r"""The lag validity floor: which source steps the attention may read, and what it costs.

Lag attention searches $L = \mathrm{max\_lag} + 1$ steps back, so at the shipped anchor floor of
$133$ it reads source states down to step $43$ -- where much of the source is still inside its own
warm-up. The second stored source block's fastest channel is not honest until step $41$ and its
slowest until $134$, so at step $43$ a single one of its fifteen channels is warm.

The design does not resolve that; it makes it *measurable*. ``lag_floor`` generalises the mask from
$\mathbb 1[t - \ell \ge 0]$ to $\mathbb 1[t - \ell \ge F_u]$, ships at $0$ where it must be bitwise
the architecture's, and exists so a run that concentrates its attention on cold lags can be seen to.

**One consequence is stated here rather than discovered later.** At a non-zero floor the rows below
it have no admissible lag at all. The attention normalises such a row to zero rather than to NaN --
the handling the $t - \ell \ge 0$ mask already needs -- and zero is the right reading: no lag was
attended because none was available. But the lag map is $K_t$ *distributed by* those weights, so a
zeroed row carries no attribution against a non-zero $K_t$, and the identity
$\sum_\ell \widetilde K_{t,\ell} = K_t$ holds from the floor onwards rather than everywhere. Since
$F_u \le F$ in any configuration that makes sense, those rows are outside the scored anchor range
anyway.

The comparison arm at floor $0$ is the raw-signal architecture itself. That is available here and is
**not** available to the causal-feature cells: their decoder emits a different width, so a
differently-shaped head consumes a different amount of the initialisation stream and every weight
after it would differ, which would make a bitwise attention comparison meaningless. This cell is
parameter-for-parameter that architecture on the ungated arm.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import TINY_KWARGS, build, make_streams, tiny_warmup_kwargs

_TOL = 1e-6


def _forward(model, streams, **kwargs):
    """One seeded forward in ``eval()``, so two calls are the same computation twice."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, **kwargs)


# =================================================================================================
# The floor at zero is the architecture's mask
# =================================================================================================
def test_the_unfloored_mask_is_the_architectures_own_object(tiny_warmup) -> None:
    """Returned unchanged, not rebuilt equal: an unfloored causal model attends exactly as the
    architecture does."""
    model = build(tiny_warmup)
    assert model.lag_floor == 0

    mask = model.build_lag_mask(int(tiny_warmup["sequence_length"]))
    expected = SeqVaeLagAttnRws.build_lag_mask(model, int(tiny_warmup["sequence_length"]))

    assert torch.equal(mask, expected)


def test_at_lag_floor_zero_the_attention_is_bitwise_the_architectures(tiny_kwargs) -> None:
    """The ungated arm, where the two models are parameter-for-parameter identical.

    The *unguarded* keyword set is deliberate: a warm-up gives this model a mask projection the
    architecture has no counterpart for, so the two initialisations would diverge from that module
    onward and the comparison would say nothing about the mask. The mask itself is compared against
    the architecture's own implementation in the test above, which is where that claim belongs.
    """
    causal = build(tiny_kwargs).eval()
    torch.manual_seed(0)
    architecture = SeqVaeLagAttnRws(**dict(tiny_kwargs)).eval()

    streams = make_streams(tiny_kwargs)
    ours = _forward(causal, streams)
    theirs = _forward(architecture, streams)

    for key in ("attn_weights", "mu_prior", "logvar_prior", "source_state", "kld_per_t"):
        assert torch.equal(ours[key], theirs[key]), key


# =================================================================================================
# The floor at a real value
# =================================================================================================
@pytest.mark.parametrize("floor", (1, 3, 5))
def test_the_floor_zeroes_exactly_the_lags_it_names(tiny_warmup, floor: int) -> None:
    r"""Against a hand-built mask: $m_{t,\ell} = \mathbb 1[t - \ell \ge F_u]$, nothing else."""
    model = build(tiny_warmup_kwargs(tiny_warmup, lag_floor=floor))
    seq_len = int(tiny_warmup["sequence_length"])

    mask = model.build_lag_mask(seq_len)

    steps = torch.arange(seq_len)[:, None]
    lags = torch.arange(model.lag_attn.L)[None, :]
    assert torch.equal(mask, steps - lags >= floor)


@pytest.mark.parametrize("floor", (1, 3, 5))
def test_the_attention_puts_no_mass_on_a_floored_lag(tiny_warmup, floor: int) -> None:
    """The mask reaching the weights, not merely being built -- so the zeroing happens *before*
    normalisation rather than being scaled away after it."""
    model = build(tiny_warmup_kwargs(tiny_warmup, lag_floor=floor)).eval()
    weights = _forward(model, make_streams(tiny_warmup))["attn_weights"]  # (B, T, heads, L)

    seq_len = weights.shape[1]
    steps = torch.arange(seq_len)[:, None]
    lags = torch.arange(weights.shape[-1])[None, :]
    forbidden = (steps - lags < floor)[None, :, None, :]
    assert float((weights * forbidden).abs().max()) == 0.0

    # And the rows that keep a lag still normalise to one, so the floor removed mass rather than
    # rescaling everything.
    warm = weights[:, floor:]
    assert torch.allclose(warm.sum(dim=-1), torch.ones_like(warm.sum(dim=-1)), atol=_TOL)


@pytest.mark.parametrize("floor", (1, 3, 5))
def test_a_fully_masked_row_is_zero_rather_than_nan(tiny_warmup, floor: int) -> None:
    """Every row below the floor has no admissible lag; the softmax over all-$-\\infty$ is NaN
    without the handling the existing mask already needs."""
    model = build(tiny_warmup_kwargs(tiny_warmup, lag_floor=floor)).eval()
    weights = _forward(model, make_streams(tiny_warmup))["attn_weights"]

    assert not bool(torch.isnan(weights).any())
    assert float(weights[:, :floor].abs().max()) == 0.0


@pytest.mark.parametrize("floor", (0, 1, 3, 5))
def test_the_lag_map_recomposes_to_the_per_step_kl_from_the_floor_onwards(
    tiny_warmup, floor: int, perturb_posterior
) -> None:
    r"""$\sum_\ell \widetilde K_{t,\ell} = K_t$ wherever a lag is admissible.

    Perturbed first: the posterior deltas are zero-initialised, so on a fresh model both sides are
    identically $0$ and the identity would hold on a model whose lag map was wired to nothing.

    Below the floor the map is exactly zero against a non-zero $K_t$, which is asserted rather than
    tolerated -- it is the price of the floor, and it is invisible in the summed readout.
    """
    kwargs = tiny_warmup_kwargs(tiny_warmup, lag_floor=floor)
    torch.manual_seed(0)
    model = SeqVaeLagAttnCrws(**kwargs)
    perturb_posterior(model)
    model.eval()

    out = _forward(model, make_streams(tiny_warmup))
    total, lag_map = out["kld_per_t"], out["source_kl_lag_map"]
    assert float(total.abs().max()) > _TOL, "the probe is vacuous on an unperturbed model"

    warm = slice(floor, None)
    assert torch.allclose(lag_map[:, warm].sum(dim=-1), total[:, warm], rtol=1e-5, atol=_TOL)
    if floor:
        assert float(lag_map[:, :floor].abs().max()) == 0.0
        assert float(total[:, :floor].abs().max()) > 0.0


def test_the_floor_actually_moves_the_attention(tiny_warmup) -> None:
    """The paired control for every assertion above: at floor $0$ the forbidden region carries mass,
    so a model that ignored ``lag_floor`` entirely would fail here rather than pass."""
    unfloored = build(tiny_warmup).eval()
    weights = _forward(unfloored, make_streams(tiny_warmup))["attn_weights"]

    seq_len = weights.shape[1]
    steps = torch.arange(seq_len)[:, None]
    lags = torch.arange(weights.shape[-1])[None, :]
    forbidden = (steps - lags < 3)[None, :, None, :]

    assert float((weights * forbidden).abs().max()) > 0.0


def test_the_default_lag_floor_is_zero() -> None:
    """It ships at zero and no configuration moves it; the knob exists so the residual is
    measurable, and ``source_lag_warmth_frac_st`` / ``_ph`` are what measure it."""
    assert build(dict(TINY_KWARGS)).lag_floor == 0
