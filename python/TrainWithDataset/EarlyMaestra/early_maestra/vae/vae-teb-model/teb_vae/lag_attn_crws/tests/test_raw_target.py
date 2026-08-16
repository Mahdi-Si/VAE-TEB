r"""The anchored raw gather: the one genuinely new piece of arithmetic in this cell.

Every other member of this package is inherited or bound. This one could not be, because
:func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_target` takes no anchor set -- the raw
target never needed one, since no raw-target sibling tiled -- and extending that module would edit a
file four shipped cells score through.

So there are now two expressions of the same arithmetic in the repository, and the test that matters
is the one that pins them against each other: **at the dense anchor set the anchored path must equal
the dense builder elementwise**. Everything else here guards the ways the second expression could be
wrong while every shape stayed right.

**Why a ``gather`` and not an ``index_select``.** The dense builder may use ``index_select`` because
its index is the *shared* $(T_{\mathrm{valid}}, H, R)$ grid -- the same rows for every sample. Here
the anchor set is per sample, because the tile phase is derived per segment, so the index is
$(B, A, H, R)$ and an ``index_select`` on dimension $1$ returns $(B, B \cdot A \cdot H \cdot R)$ and
fails the reshape. The property that distinguishes them is asserted directly: two rows with
different anchors must produce two different windows.

**Why the bounds check is here rather than only in the mask.** Advanced indexing on a negative index
*wraps*, so an anchor of $-1$ would gather the last legal window and return every shape correct. The
mask's own validation would catch it one call later -- but only for a batch that reached the mask,
and only as a claim about the mask rather than about the target that was already built.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import gather_anchored_future_target
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    tiny_warmup_kwargs,
)


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling, so a padded slot exists at some phase."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model, built once. The one test that mutates a buffer builds its own."""
    return build(_kwargs()).eval()


@pytest.fixture(scope="module")
def forward_outputs(model):
    """One forward at a real tiling, carrying the anchor set the objective must score at."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*make_streams(_kwargs()), 1)


def _loss(model, out, signal, **overrides):
    """The objective on a batch of unit weights, in ``mse`` so no log-variance term dominates."""
    weight = torch.ones(signal.shape[0], model.geometry.t)
    return model.compute_loss(
        out, signal, weight=weight, likelihood="mse", **overrides
    )["metrics"]


# =================================================================================================
# The gather itself
# =================================================================================================
def test_the_anchored_path_equals_the_dense_builder_at_the_dense_anchor_set(model) -> None:
    r"""The criterion that bites: at ``anchors = arange(T_valid)`` the two expressions of the same
    arithmetic must agree elementwise, not approximately and not in shape alone.

    This is the whole justification for owning a second expression at all. It is checked under
    ``torch.equal`` because both paths read the same integer grid and gather the same floats: any
    difference is an indexing mistake rather than an accumulation one.
    """
    signal = make_raw_signal(_kwargs())
    dense = build_future_target(signal, model.geometry, future_index=model.future_index)
    anchors = torch.arange(model.geometry.t_valid)[None, :].expand(BATCH, -1)

    anchored = gather_anchored_future_target(
        signal, model.geometry, anchors, future_index=model.future_index
    )

    assert tuple(anchored.shape) == tuple(dense.shape)
    assert torch.equal(anchored, dense)


def test_a_per_sample_anchor_set_yields_per_sample_windows(model) -> None:
    """The property an ``index_select`` cannot have: two rows, two different windows.

    Each row is also checked against the raw samples its own anchor names, so the test says which
    window is right rather than only that the two differ.
    """
    signal = make_raw_signal(_kwargs())
    anchors = torch.tensor([[5, 9], [6, 10]])

    windows = gather_anchored_future_target(
        signal, model.geometry, anchors, future_index=model.future_index
    )

    assert tuple(windows.shape) == (2, 2, model.horizon, model.geometry.r)
    assert not torch.equal(windows[0], windows[1])
    for row in range(2):
        for slot in range(2):
            anchor = int(anchors[row, slot])
            start = model.geometry.future_block_start(anchor)
            stop = start + model.horizon * model.geometry.r
            expected = signal[row, start:stop].reshape(model.horizon, model.geometry.r)
            assert torch.equal(windows[row, slot], expected), (row, slot)


def test_a_padded_slot_gathers_its_rows_last_valid_window(model) -> None:
    """The padding convention, at the target rather than at the index.

    A padded slot repeats its row's last real anchor, so the window it gathers is that anchor's --
    a duplicate the forecast mask then zeroes. A slot holding a distinct *legal* anchor would be
    fully live instead, and its block would be scored twice against a KL support that counts it
    once.

    Built at the **last** phase rather than at the fixture's: $A_{\\max}$ is the widest phase's
    count, so the short rows -- and therefore every padded slot there is -- appear at the largest
    phase alone.
    """
    index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=TINY_STRIDE - 1
    )
    signal = make_raw_signal(_kwargs())
    windows = gather_anchored_future_target(
        signal, model.geometry, index, future_index=model.future_index
    )

    padded = ~valid
    assert bool(padded.any()), "no phase produced a short row; the padding is untested"
    for row, slot in padded.nonzero(as_tuple=False).tolist():
        last = int(valid[row].nonzero(as_tuple=False)[-1])
        assert torch.equal(windows[row, slot], windows[row, last]), (row, slot)


# =================================================================================================
# The cached grid
# =================================================================================================
def test_the_gather_reads_the_cached_index_buffer_rather_than_rebuilding_it(
    forward_outputs,
) -> None:
    """A positive claim rather than a monkeypatched absence: moving the buffer moves the loss.

    Rebuilding the grid per step would be a second construction that could disagree with the one
    every mask and every figure is built against, and it would not raise -- both grids have the same
    shape and the same dtype. So the buffer is mutated **in place**, its address is asserted
    unchanged either side, and the objective is asserted to have followed it.
    """
    kwargs = _kwargs()
    model = build(kwargs).eval()
    signal = make_raw_signal(kwargs)
    address = model.future_index.data_ptr()

    before = _loss(model, forward_outputs, signal)
    assert model.future_index.data_ptr() == address

    # Reversing the anchor axis keeps every entry a legal raw index, so only which window each
    # anchor names changes.
    with torch.no_grad():
        model.future_index.copy_(model.future_index.flip(0))
    after = _loss(model, forward_outputs, signal)

    assert model.future_index.data_ptr() == address
    assert float(after["nll_base_block"]) != float(before["nll_base_block"])


# =================================================================================================
# The refusals
# =================================================================================================
def test_an_anchor_past_the_last_valid_one_is_refused_naming_it(model) -> None:
    r"""$\ge T_{\mathrm{valid}}$, not $\ge T$: the tail $H$ anchors have no fully observed window."""
    signal = make_raw_signal(_kwargs())
    offending = model.geometry.t_valid

    with pytest.raises(ValueError, match=str(offending)):
        gather_anchored_future_target(
            signal,
            model.geometry,
            torch.tensor([[5, offending]]),
            future_index=model.future_index,
        )


def test_a_negative_anchor_is_refused_rather_than_wrapped(model) -> None:
    """The failure that would otherwise be silent: advanced indexing wraps, so $-1$ would gather
    the last legal window and every shape would be right."""
    signal = make_raw_signal(_kwargs())

    with pytest.raises(ValueError, match="-1"):
        gather_anchored_future_target(
            signal, model.geometry, torch.tensor([[-1, 5]]), future_index=model.future_index
        )


def test_a_signal_at_another_trim_is_refused_naming_both_lengths(model) -> None:
    """A loader at a different ``trim_minutes`` shifts every window by whole minutes, and a longer
    signal would gather silently rather than raise."""
    signal = make_raw_signal(_kwargs())

    with pytest.raises(ValueError, match="raw_len"):
        gather_anchored_future_target(
            signal[:, :-16],
            model.geometry,
            torch.tensor([[5]]),
            future_index=model.future_index,
        )
    with pytest.raises(ValueError, match="2-D"):
        gather_anchored_future_target(
            signal[0], model.geometry, torch.tensor([[5]]), future_index=model.future_index
        )


def test_a_duplicate_among_the_valid_entries_is_refused_by_the_objective(
    model, forward_outputs
) -> None:
    """Uniqueness is the mask's refusal, not the gather's, and it is reached on the same call.

    The gather honours a duplicate deliberately -- that is the padding convention -- so the check
    belongs where the two per-anchor denominators it protects are built. What must not happen is
    that a duplicated *valid* anchor passes silently: its block would be scored twice by the
    reconstruction and once by the KL, and $\\beta$ would stop meaning what it means everywhere else.
    """
    signal = make_raw_signal(_kwargs())
    index = forward_outputs["anchor_index"].clone()
    offending = int(index[0, 0])
    index[0, 1] = offending
    planted = dict(forward_outputs)
    planted["anchor_index"] = index
    planted["anchor_valid"] = torch.ones_like(forward_outputs["anchor_valid"])

    with pytest.raises(ValueError, match=f"anchor {offending} appears twice"):
        _loss(model, planted, signal)


# =================================================================================================
# What the objective was handed
# =================================================================================================
def test_the_block_width_is_the_raw_grids_own(model, forward_outputs) -> None:
    r"""``block_width`` is ``geometry.r``, and it is observable rather than asserted at the call.

    It feeds only the four per-element log-variance diagnostics, so a wrong value changes no
    gradient and fails no shape check -- it rescales exactly those four numbers by a constant. The
    check is therefore to recompute one of them as a true elementwise mean over the block and
    require the reported column to be it.
    """
    signal = make_raw_signal(_kwargs())
    metrics = _loss(model, forward_outputs, signal)

    weight = torch.ones(signal.shape[0], model.geometry.t)
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

    mask, _coverage = forecast_mask(
        weight,
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=forward_outputs["anchor_index"],
        anchor_valid=forward_outputs["anchor_valid"],
    )
    elements = mask[..., None]
    expected = float(
        (forward_outputs["logvar_full"] * elements).sum()
        / (elements.sum() * model.geometry.r)
    )

    assert model.geometry.r == int(_kwargs()["raw_per_step"])
    assert float(metrics["mean_logvar_full"]) == pytest.approx(expected, rel=1e-6)


def test_the_objective_carries_the_three_kept_readouts(model, forward_outputs) -> None:
    """Merged onto the shared objective's dict, which is pinned bitwise for the shipped cells and
    therefore may not gain them there. The five that partition kept target channels are absent:
    this block's last axis counts raw samples."""
    signal = make_raw_signal(_kwargs())
    metrics = _loss(model, forward_outputs, signal)

    for name in ("anchors_per_sample", "source_lag_warmth_frac_st", "source_lag_warmth_frac_ph"):
        assert name in metrics, name
        assert torch.isfinite(metrics[name]), name
    for dropped in (
        "pred_gap_st",
        "pred_gap_ph",
        "pred_gap_warm_lo",
        "pred_gap_warm_mid",
        "pred_gap_warm_hi",
        "target_warm_frac",
    ):
        assert dropped not in metrics, dropped


def test_a_forward_with_no_anchor_set_builds_the_dense_target(model, forward_outputs) -> None:
    r"""Which is what makes a stripped anchor set a **shape refusal** rather than a wrong number.

    ``anchors=None`` means $[0, T_{\mathrm{valid}})$, so the target carries $T_{\mathrm{valid}}$
    anchors against a forecast carrying $A_{\max}$, and the score cannot broadcast.
    """
    signal = make_raw_signal(_kwargs())
    stripped = {
        name: value
        for name, value in forward_outputs.items()
        if name not in ("anchor_index", "anchor_valid")
    }

    assert model.geometry.t_valid != int(forward_outputs["anchor_index"].shape[1])
    with pytest.raises(RuntimeError):
        _loss(model, stripped, signal)
