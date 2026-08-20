r"""The tiled anchor set: which anchors are decoded, and what the padded slots cost.

The semantics are inherited whole -- ``_build_anchor_index`` is the causal-feature cell's own
function object, asserted by identity in ``test_causal_raw_inputs.py`` -- but the *failure modes* are
newly load-bearing here, because this cell gathers a **raw** window at every returned index. A padded
slot pulls a second copy of a raw window rather than a second copy of a feature block, and if that
copy were live the reconstruction would score $H \cdot R = 64$ raw samples twice against a KL support
that counts the anchor once.

Three properties carry the tiling, and each replaces something that does not work:

**$A_{\max}$ is a geometry constant.** $\lceil (T_{\mathrm{valid}} - F)/S \rceil$, independent of the
phase and of the batch, so no rank can disagree about a shape and no shape is a function of the data.
What varies is how many entries are real.

**Padding repeats the last valid anchor.** A padded slot holding a distinct *legal* index would
produce a fully live forecast-mask row. The gathered window is therefore a duplicate by design, and
what makes that sound is that the mask zeroes it -- which is asserted here as a loss identity rather
than as a mask shape.

**Nothing is drawn.** The phase arrives already derived. A draw inside the forward would consume the
global RNG stream, move the reparameterisation $\epsilon$, break every bitwise comparison in this
suite and fail to survive a checkpoint resume -- with $A_{\max}$ a constant either way, so no shape
would say so.

Every number below is derived from the geometry rather than written out, so a horizon or floor change
re-derives it instead of failing an unrelated literal.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

from .conftest import (
    BATCH,
    TINY_STRIDE,
    build,
    make_raw_signal,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: The forward keys carrying the anchor axis, which a trimmed anchor set must be sliced along
#: together. Named once so the padded-slot identity below cannot silently trim only some of them.
_ANCHOR_AXIS_KEYS = (
    "mu_base",
    "logvar_base",
    "mu_full",
    "logvar_full",
    "anchor_index",
    "anchor_valid",
)


def _kwargs() -> dict:
    """The tiny guarded keyword set at a real tiling."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)


def _geometry(model) -> tuple:
    r"""``(F, T_valid, S)`` for a built model at its configured stride."""
    return model.warmup_period, model.geometry.t_valid, model.anchor_stride


def _a_max(model, stride: int) -> int:
    r"""$A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$, computed the long way."""
    floor, t_valid, _ = _geometry(model)
    return math.ceil((t_valid - floor) / stride)


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model, built once: nothing below trains and nothing below mutates it."""
    return build(_kwargs()).eval()


# =================================================================================================
# Shape and dtype
# =================================================================================================
def test_the_anchor_axis_is_a_geometry_constant(model) -> None:
    """Not a function of the phase, and not a function of the batch."""
    _, _, stride = _geometry(model)
    expected = _a_max(model, stride)

    for phase in range(stride):
        index, valid = model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=phase
        )
        assert tuple(index.shape) == (BATCH, expected), phase
        assert tuple(valid.shape) == (BATCH, expected), phase
        assert index.dtype == torch.long and valid.dtype == torch.bool

    wider, _ = model._build_anchor_index(batch=7, device=torch.device("cpu"), anchor_phase=0)
    assert tuple(wider.shape) == (7, expected)


def test_the_dense_stride_reproduces_the_whole_anchor_range(model) -> None:
    r"""``anchor_stride: 1`` is exactly $[F, T_{\mathrm{valid}})$, which is what validation decodes."""
    floor, t_valid, _ = _geometry(model)
    index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=0, anchor_stride=1
    )

    assert bool(valid.all())
    assert index[0].tolist() == list(range(floor, t_valid))
    assert tuple(index.shape) == (BATCH, t_valid - floor)


# =================================================================================================
# The tiling itself
# =================================================================================================
def test_valid_entries_are_ascending_and_inside_the_anchor_range(model) -> None:
    """Every index in range, strictly ascending among the real entries, none below the floor."""
    floor, t_valid, stride = _geometry(model)

    for phase in range(stride):
        index, valid = model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=phase
        )
        assert bool(((index >= floor) & (index < t_valid)).all()), phase
        row = index[0][valid[0]]
        assert row.tolist() == sorted(set(row.tolist())), phase
        assert int(row[0]) == floor + phase, phase
        assert bool((row.diff() == stride).all()), phase


def test_short_rows_repeat_their_last_valid_anchor(model) -> None:
    """The convention that keeps every index in range without inventing a second legal anchor."""
    floor, t_valid, stride = _geometry(model)
    index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=stride - 1
    )

    padded = ~valid
    assert bool(padded.any()), "no phase produced a short row; the padding is untested"
    last_valid = index[0][valid[0]][-1]
    assert bool((index[0][padded[0]] == last_valid).all())
    # A duplicate exists, and only ever among the invalid entries -- which is exactly what the
    # objective's own anchor validation permits.
    assert len(set(index[0].tolist())) < index.shape[1]


def test_the_valid_count_is_the_tiles_that_fit(model) -> None:
    r"""$\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$, phase by phase."""
    floor, t_valid, stride = _geometry(model)

    counts = []
    for phase in range(stride):
        _, valid = model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=phase
        )
        expected = math.ceil((t_valid - floor - phase) / stride)
        assert int(valid[0].sum()) == expected, phase
        counts.append(expected)

    # The phases sum to the dense count, which is what makes the tiling a partition of the same
    # supervision rather than a reduction of it.
    assert sum(counts) == t_valid - floor


def test_every_phase_is_a_different_grid_and_together_they_cover_everything(model) -> None:
    """Goal of the tiling: no raw sample scored twice in a step, every anchor reached over epochs."""
    floor, t_valid, stride = _geometry(model)
    grids = []
    for phase in range(stride):
        index, valid = model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=phase
        )
        grids.append(set(index[0][valid[0]].tolist()))

    assert set().union(*grids) == set(range(floor, t_valid))
    for first in range(len(grids)):
        for second in range(first + 1, len(grids)):
            assert grids[first].isdisjoint(grids[second]), (first, second)


def test_the_phase_is_per_sample(model) -> None:
    """The batch's samples are tiled independently, which is what breaks the within-batch
    correlation an unshuffled per-recording loader would otherwise leave in place."""
    _, _, stride = _geometry(model)
    phase = torch.arange(BATCH) % stride

    index, _ = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=phase
    )
    assert not torch.equal(index[0], index[1])
    assert int(index[1, 0] - index[0, 0]) == int(phase[1] - phase[0])


# =================================================================================================
# What a padded slot costs
# =================================================================================================
def test_a_padded_slot_contributes_exactly_zero_to_the_loss(model) -> None:
    """The property the padding convention exists for, as a loss identity rather than a mask shape.

    A padded slot's gathered window *is* its row's last valid window -- that is what repeating the
    index means -- so the only thing standing between it and a doubly-scored raw block is the mask.
    Scoring the same forward with the padding trimmed away must therefore reproduce every
    reconstruction readout: a padded slot that leaked would show as a block NLL inflated by roughly
    one anchor in four against an unchanged anchor count, which no shape would report.

    Compared to a relative tolerance rather than bitwise, and the reason is the comparison rather
    than the claim. A padded slot contributes exactly zero *terms*, but the two sums are reductions
    over differently-shaped tensors, so they accumulate in a different order; the observed gap is
    $2 \\times 10^{-7}$ relative against a leak of $0.33$.
    """
    _, _, stride = _geometry(model)
    kwargs = _kwargs()
    signal = make_raw_signal(kwargs)
    weight = torch.ones(BATCH, model.geometry.t)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(kwargs), stride - 1)
    valid = out["anchor_valid"]
    assert bool((~valid).any()), "no padded slot at the widest phase; the identity is vacuous"

    count = int(valid[0].sum())
    assert bool((valid.sum(dim=1) == count).all()), "a scalar phase gives every row one count"
    trimmed = {
        name: (value[:, :count] if name in _ANCHOR_AXIS_KEYS else value)
        for name, value in out.items()
    }
    assert bool(trimmed["anchor_valid"].all())

    padded_metrics = model.compute_loss(out, signal, weight=weight, likelihood="mse")["metrics"]
    trimmed_metrics = model.compute_loss(
        trimmed, signal, weight=weight, likelihood="mse"
    )["metrics"]

    for name in ("nll_base_block", "nll_full_block", "source_conditioned_kl_raw"):
        assert float(padded_metrics[name]) == pytest.approx(
            float(trimmed_metrics[name]), rel=1e-5, abs=1e-6
        ), name
    assert float(padded_metrics["anchors_per_sample"]) == float(count)


# =================================================================================================
# What is refused
# =================================================================================================
def test_a_missing_phase_is_refused_once_the_stride_is_real(model) -> None:
    """A forgotten phase would train every sample of every epoch on one grid, at a fixed offset from
    the segment start, with no shape and no count differing."""
    with pytest.raises(ValueError, match="anchor_phase is required"):
        model._build_anchor_index(batch=BATCH, device=torch.device("cpu"))

    # And it is admitted at stride 1, where there is no grid to bias.
    _index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_stride=1
    )
    assert bool(valid.all())


def test_a_non_zero_phase_at_stride_one_is_refused(model) -> None:
    r"""$\mathcal A(\varphi)$ truncates rather than rotating, so a phase at stride $1$ would silently
    drop its first $\varphi$ anchors and shorten the anchor count."""
    with pytest.raises(ValueError, match=r"outside \[0, anchor_stride\)"):
        model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=1, anchor_stride=1
        )


def test_a_phase_at_or_above_the_stride_is_refused(model) -> None:
    """Same reason, at the real stride: it drops a leading tile instead of shifting the grid."""
    _, _, stride = _geometry(model)

    with pytest.raises(ValueError, match=str(stride)):
        model._build_anchor_index(batch=BATCH, device=torch.device("cpu"), anchor_phase=stride)
    with pytest.raises(ValueError, match="-1"):
        model._build_anchor_index(batch=BATCH, device=torch.device("cpu"), anchor_phase=-1)


def test_a_phase_of_the_wrong_length_is_refused(model) -> None:
    """The phase is per sample, so a mismatch would tile one sample at another's grid."""
    with pytest.raises(ValueError, match="per sample"):
        model._build_anchor_index(
            batch=BATCH, device=torch.device("cpu"), anchor_phase=torch.zeros(BATCH + 1)
        )


# =================================================================================================
# No randomness, and the decode itself
# =================================================================================================
def test_building_the_anchor_set_draws_no_random_number(model) -> None:
    """A draw here would move the reparameterisation stream and break every bitwise comparison in
    the suite -- and would not reproduce across a checkpoint resume, with nothing saying so."""
    _, _, stride = _geometry(model)
    torch.manual_seed(0)
    before = torch.random.get_rng_state()

    for phase in range(stride):
        model._build_anchor_index(batch=BATCH, device=torch.device("cpu"), anchor_phase=phase)

    assert torch.equal(before, torch.random.get_rng_state())


def test_a_forward_consumes_the_same_randomness_at_every_phase(model) -> None:
    """The end-to-end form. Note what is *not* claimed: the forward draws nothing.
    ``_reparameterize_shared`` calls ``randn_like`` unconditionally, so the claim is that the tiling
    does not change how much randomness is consumed."""
    _, _, stride = _geometry(model)
    streams = make_streams(_kwargs())

    states = []
    for phase in range(stride):
        torch.manual_seed(0)
        with torch.no_grad():
            model(*streams, phase)
        states.append(torch.random.get_rng_state())

    assert all(torch.equal(states[0], state) for state in states[1:])


def test_the_decoder_is_invoked_on_the_gathered_latents(model) -> None:
    """Not on the contiguous prefix: the anchors are no longer one.

    Read at the decoder's own input, so this is what the module received rather than what the
    forward meant to hand it.
    """
    streams = make_streams(_kwargs())
    seen: list = []
    handle = model.decoder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            out = model(*streams, 1)
    finally:
        handle.remove()

    index = out["anchor_index"]
    assert len(seen) == 2
    for latent, key in zip(seen, ("z_prior", "z_post")):
        expected = out[key].gather(1, index[:, :, None].expand(-1, -1, model.d_z))
        assert torch.equal(latent, expected), key
    assert tuple(out["mu_base"].shape)[:2] == tuple(index.shape)


def test_the_forward_returns_the_anchors_it_decoded(model) -> None:
    """Returned rather than recomputed, so the objective and the figures cannot disagree with it --
    which on this cell is the difference between one raw window and another."""
    streams = make_streams(_kwargs())
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams, 2)

    index, valid = model._build_anchor_index(
        batch=BATCH, device=torch.device("cpu"), anchor_phase=2
    )
    assert torch.equal(out["anchor_index"], index)
    assert torch.equal(out["anchor_valid"], valid)


# =================================================================================================
# The shipped geometry
# =================================================================================================
def test_the_shipped_geometry_tiles_as_the_budget_predicts() -> None:
    r"""$F = 134$, $S = H = 30$, $T_{\mathrm{valid}} = 270$: five tiles at $\varphi \le 15$ and four
    otherwise, mean $136/30$; and the dense validation resolution is $136$.

    The floor is the aligned one. Unaligned it was $133$, which put the five-tile boundary one phase
    later and gave $137$ dense anchors: the one anchor the common clock costs.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**shipped_warmup_kwargs()).eval()
    floor, t_valid, stride = _geometry(model)
    assert (floor, t_valid, stride) == (134, 270, 30)

    counts = []
    for phase in range(stride):
        _index, valid = model._build_anchor_index(
            batch=1, device=torch.device("cpu"), anchor_phase=phase
        )
        assert tuple(valid.shape) == (1, 5)
        counts.append(int(valid.sum()))

    assert counts[:16] == [5] * 16 and set(counts[16:]) == {4}
    assert sum(counts) == t_valid - floor == 136

    dense, valid = model._build_anchor_index(
        batch=1, device=torch.device("cpu"), anchor_stride=1
    )
    assert tuple(dense.shape) == (1, 136) and bool(valid.all())
    assert int(dense[0, 0]) == 134 and int(dense[0, -1]) == 269
