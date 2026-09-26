r"""The lag validity floor as this model's attention and lag map see it.

Lag attention searches $L = \mathrm{max\_lag} + 1$ steps back, so from the anchor floor it reads
source states still inside their own warm-up. ``lag_floor`` generalises the mask from
$\mathbb 1[t - \ell \ge 0]$ to $\mathbb 1[t - \ell \ge F_u]$ and ships at $0$, where the forward
must be bitwise the raw-signal architecture's -- available as a comparison arm here because the two
are parameter for parameter identical on the ungated arm. The mask itself is the causal-feature
cell's and is tested there.

At a non-zero floor the attention puts no mass on a floored lag (and a row with no admissible lag
is zero rather than NaN), the rows that keep a lag still normalise to one, and the identity
$\sum_\ell \widetilde K_{t,\ell} = K_t$ holds from the floor onwards: below it the map is exactly
zero against a non-zero $K_t$, since $F_u \le F$ keeps those rows outside the scored anchor range.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import build, make_streams, tiny_warmup_kwargs

_TOL = 1e-6


def _forward(model, streams, **kwargs):
    """One seeded forward in ``eval()``, so two calls are the same computation twice."""
    torch.manual_seed(0)
    with torch.no_grad():
        return model(*streams, **kwargs)


# =================================================================================================
# The floor at zero is the architecture's mask
# =================================================================================================
def test_at_lag_floor_zero_the_attention_is_bitwise_the_architectures(tiny_kwargs) -> None:
    """The ungated arm, where the two models are parameter-for-parameter identical.

    The *unguarded* keyword set is deliberate: a warm-up gives this model a mask projection the
    architecture has no counterpart for, so the two initialisations would diverge from that module
    onward and the comparison would say nothing about the mask.
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
def test_the_attention_puts_no_mass_on_a_floored_lag(tiny_warmup, floor: int) -> None:
    """The mask reaching the weights, not merely being built -- so the zeroing happens *before*
    normalisation rather than being scaled away after it. Every row below the floor has no
    admissible lag, and a NaN there (a softmax over all $-\\infty$) would fail the zero check too."""
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
