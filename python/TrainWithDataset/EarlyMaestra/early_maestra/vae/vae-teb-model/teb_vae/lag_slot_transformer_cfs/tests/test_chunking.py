r"""Chunking is a memory setting, so it must move nothing but the order of a sum.

Three properties, and the third is the one whose failure is silent.

**Agreement.** A chunked pass that dropped a lag, mis-indexed the lag embedding or reduced the wrong
axis would still produce a well-shaped result. Only a comparison against the whole-array pass shows
it, and only if both passes are the same model -- which is why the builder here is the seeded one.

**A measured tolerance, not a claimed one.** Changing the chunk size reassociates the sums, and
single precision does not promise associativity. The number below was measured at this geometry
rather than predicted, and it is recorded here rather than in a comment because it is evidence.

**Nothing is detached.** A detached chunk trains a model whose lags in that chunk never update. No
shape and no metric would say so; the run would simply behave as though those lags carried no
information, which is indistinguishable from the finding this whole architecture exists to test.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_N_LAGS,
    build_tiny_model,
    tiny_streams,
)

#: The largest absolute difference actually observed between a whole-array pass and the degenerate
#: one-anchor one-lag grid, at the tiny geometry in single precision, on 2026-09-09. Recorded as a
#: measurement rather than as a bound: it is the size of the reassociation error and it is what the
#: tolerances below are set from, with a margin, rather than the other way round.
#:
#: It lands on the decoder output, which is the largest-magnitude tensor in the comparison; the
#: proposals and the summed update agree an order of magnitude more tightly.
MEASURED_MAX_ABS = 9.6e-7

#: Tolerances for chunked-against-unchunked agreement. Roughly ten times the measured error, so a
#: real drift fails while ordinary reassociation does not. They are the numerical tolerance of a
#: reassociated sum and say nothing about predictive accuracy.
CHUNK_ATOL = 1.0e-5
CHUNK_RTOL = 1.0e-4

#: Chunk grids exercised. The last is the degenerate one -- a single anchor and a single lag per
#: pass -- which is where an accumulation bug is most likely and cheapest to catch.
CHUNK_GRIDS = (
    dict(anchor_chunk=3),
    dict(lag_chunk=2),
    dict(anchor_chunk=2, lag_chunk=3),
    dict(anchor_chunk=1, lag_chunk=1),
)


def trained(**overrides):
    """A seeded model whose source pathway has been moved off zero.

    At the zero start every proposal is exactly zero, so a chunked and an unchunked pass would
    agree by producing nothing. The comparison is only a measurement once the pathway is live.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The model, in evaluation mode.
    """
    model = build_tiny_model(**overrides)
    generator = torch.Generator().manual_seed(37)
    with torch.no_grad():
        model.proposal_head.output_proj.weight.normal_(0.0, 0.4, generator=generator)
        model.proposal_head.output_proj.bias.normal_(0.0, 0.2, generator=generator)
    return model.eval()


def run(model, *, seed: int = 0, **kwargs):
    """One dense forward under a fixed noise seed.

    Args:
        model: The model.
        seed: Seed for the reparameterisation draw.
        **kwargs: Extra forward keywords.

    Returns:
        The forward's dict.
    """
    torch.manual_seed(seed)
    return model(*tiny_streams(), anchor_phase=0, anchor_stride=1, **kwargs)


@pytest.mark.parametrize("grid", CHUNK_GRIDS, ids=lambda g: "-".join(f"{k}{v}" for k, v in g.items()))
def test_chunked_and_unchunked_agree_within_the_measured_tolerance(grid) -> None:
    """Across every tensor a chunk could touch, and bitwise on the boolean one.

    Args:
        grid: The chunk sizes to compare against the whole-array pass.
    """
    whole = run(trained(), return_proposals=True)
    parts = run(trained(**grid), return_proposals=True)

    for name in (
        "mean_proposals",
        "scale_proposals",
        "raw_update_mean",
        "raw_update_logsigma",
        "update_mean",
        "update_logsigma",
        "mu_post",
        "logvar_post",
        "kld_per_anchor",
        "mu_full",
        "cancellation_ratio_mean",
        "cancellation_denominator_mean",
    ):
        left, right = whole[name], parts[name]
        assert torch.allclose(left, right, atol=CHUNK_ATOL, rtol=CHUNK_RTOL), name

    assert torch.equal(whole["lag_valid"], parts["lag_valid"])


def test_the_reassociation_error_is_the_size_it_was_measured_at() -> None:
    """Pinned to the measurement, so a real drift fails rather than hiding under a loose bound.

    A tolerance chosen far above what the code achieves passes while a genuine change in the
    accumulation goes unnoticed. This checks the margin from both sides: the error must stay under
    the declared tolerance, and it must stay near the figure it was measured at.
    """
    whole = run(trained(), return_proposals=True)
    parts = run(trained(anchor_chunk=1, lag_chunk=1), return_proposals=True)
    largest = max(
        float((whole[name] - parts[name]).abs().max())
        for name in ("mean_proposals", "raw_update_mean", "mu_post", "mu_full")
    )
    assert largest < CHUNK_ATOL, largest
    assert largest <= MEASURED_MAX_ABS * 4.0, largest


def test_the_lag_embedding_is_indexed_by_slot_and_not_by_chunk_position() -> None:
    """The failure a chunked pass makes easy: every chunk but the first reading another slot.

    Constructed by giving each slot a distinguishable embedding and checking that the chunked pass
    reproduces the whole-array proposals exactly. With positional indexing the first chunk would
    match and the rest would not.
    """
    whole_model = trained()
    with torch.no_grad():
        # One clearly distinct row per slot, so a misread slot cannot coincide with the right one.
        for slot in range(TINY_N_LAGS):
            whole_model.proposal_head.lag_embedding.weight[slot] = float(slot + 1)
    whole = run(whole_model, return_proposals=True)

    chunked_model = trained(lag_chunk=2)
    with torch.no_grad():
        for slot in range(TINY_N_LAGS):
            chunked_model.proposal_head.lag_embedding.weight[slot] = float(slot + 1)
    parts = run(chunked_model, return_proposals=True)

    assert torch.allclose(
        whole["mean_proposals"], parts["mean_proposals"], atol=CHUNK_ATOL, rtol=CHUNK_RTOL
    )


def test_no_chunk_is_detached_from_the_graph() -> None:
    """Every lag slot and every anchor chunk must carry gradient, not only the first.

    Checked on the degenerate grid, where each slot is its own chunk, so a detached accumulation
    would leave every slot but one without gradient.
    """
    model = build_tiny_model(anchor_chunk=1, lag_chunk=1)
    with torch.no_grad():
        model.proposal_head.output_proj.weight.normal_(0.0, 0.3)
    outputs = run(model)
    outputs["kld_per_anchor"].sum().backward()

    grad = model.proposal_head.lag_embedding.weight.grad
    assert grad is not None
    per_slot = grad.abs().sum(dim=-1)
    assert bool((per_slot > 0.0).all()), per_slot.tolist()


def test_chunk_sizes_are_refused_below_one() -> None:
    """A zero chunk is an empty pass that would silently score no lag at all."""
    for name in ("anchor_chunk", "lag_chunk"):
        with pytest.raises(ValueError, match=f"{name} must be >= 1"):
            build_tiny_model(**{name: 0})


def test_a_chunk_wider_than_its_axis_is_the_unchunked_pass() -> None:
    """So an operator raising a chunk size past the axis gets the whole pass, not an error."""
    whole = run(trained())
    wide = run(trained(anchor_chunk=1000, lag_chunk=1000))
    assert torch.allclose(whole["mu_post"], wide["mu_post"], atol=CHUNK_ATOL)
