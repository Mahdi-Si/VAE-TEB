r"""The decoders' observation log-variance cannot collapse, and the raw decoder is gone.

Under a Gaussian likelihood the model can cheat: drive $\sigma^2 \to 0$ on the steps it predicts
well and the NLL runs to $-\infty$. The loss goes down, the forecast does not improve, and
training is over. The smooth bound makes that unreachable -- any finite raw head output maps
strictly inside $(lo, hi)$, so the log-variance is floored no matter how hard the head is driven.

This is not a hypothetical. It is exactly how a sibling model in this repository froze: the
variance collapsed, the loss EMA collapsed with it, and the spike breaker then rejected every
subsequent batch forever.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets import decoders
from teb_vae.lag_attn.nets.decoders import (
    BaselineFutureDecoder,
    HorizonDecoderCore,
    ResidualFutureDecoder,
)

_LO, _HI = -5.0, 3.0
_TOL = 1e-4
_D_MODEL, _D_Z, _D_HIDDEN, _C, _HORIZON = 32, 8, 32, 109, 4
_BATCH, _SEQ_LEN = 2, 8


def _decoders():
    torch.manual_seed(0)
    core = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON)
    baseline = BaselineFutureDecoder(
        core, d_model=_D_MODEL, out_channels=_C, d_hidden=_D_HIDDEN, dropout=0.0
    ).eval()
    residual = ResidualFutureDecoder(
        core, d_model=_D_MODEL, d_z=_D_Z, out_channels=_C, d_hidden=_D_HIDDEN, dropout=0.0
    ).eval()
    return baseline, residual


def test_decoder_logvar_is_floored_under_stress():
    baseline, residual = _decoders()
    # Extreme magnitudes, to saturate the raw log-variance heads as hard as possible.
    big_state = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL) * 1e3
    big_z = torch.randn(_BATCH, _SEQ_LEN, _D_Z) * 1e3

    with torch.no_grad():
        _, logvar_base = baseline(big_state)
        _, logvar_full = residual(big_state, big_z)

    for name, logvar in (("logvar_base", logvar_base), ("logvar_full", logvar_full)):
        assert torch.isfinite(logvar).all(), f"{name} not finite under stress"
        assert logvar.min().item() >= _LO - _TOL, f"{name} fell below lo: {logvar.min().item()}"
        assert logvar.max().item() <= _HI + _TOL, f"{name} exceeded hi: {logvar.max().item()}"


def test_decoder_output_shapes():
    baseline, residual = _decoders()
    state = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL)
    z = torch.randn(_BATCH, _SEQ_LEN, _D_Z)

    with torch.no_grad():
        mu_base, logvar_base = baseline(state)
        delta_mu_src, logvar_full = residual(state, z)

    expected = (_BATCH, _SEQ_LEN, _HORIZON, _C)
    assert mu_base.shape == logvar_base.shape == expected
    assert delta_mu_src.shape == logvar_full.shape == expected


def test_both_decoders_share_one_horizon_core():
    """Sharing is the point: the correction must live in the baseline's representation space."""
    baseline, residual = _decoders()
    assert baseline.core is residual.core


def test_the_raw_refinement_decoder_does_not_exist():
    """It was a stub whose only behaviour was to raise, and nothing ever constructed it.

    It is gone rather than kept-and-unused, which is why the forward contract no longer carries
    a ``raw_future_pred`` key holding ``None`` in a dict of tensors.
    """
    assert not hasattr(decoders, "RawRefinementDecoder")


def test_a_freshly_constructed_film_core_is_an_identity():
    """``HorizonDecoderCore`` zero-inits its FiLM generator, so a bare core starts neutral.

    Scoped to the bare core deliberately. This is *not* true of a core inside the assembled
    model: the model runs a generic weight init afterwards, which xavier-refills every ``Linear``
    including this one. See the test below -- the invariant the model actually relies on is
    carried by the residual decoder's zeroed mean head, not by FiLM.
    """
    torch.manual_seed(0)
    plain = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON, film=False)
    torch.manual_seed(0)
    filmed = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON, film=True)
    filmed.load_state_dict(plain.state_dict(), strict=False)

    h = torch.randn(_BATCH, _SEQ_LEN, _D_HIDDEN)
    with torch.no_grad():
        assert torch.allclose(plain.decode(h), filmed.decode(h), atol=1e-6)


def test_the_assembled_model_starts_neutral_even_though_film_is_not_an_identity(inputs):
    """The warm-start invariant does not depend on FiLM, and it is worth knowing why.

    The model's generic weight init xavier-refills ``film_gen`` after the core zeroed it, so
    FiLM is *random* at step 0 under the shipped ``film: true``. The invariant survives anyway
    because ``residual_decoder.mean_head`` is zeroed last: whatever FiLM does to the horizon
    features, the mean head multiplies it by zero. Pinned here so nobody "fixes" the FiLM init
    believing the invariant rests on it, or removes the mean-head zeroing believing FiLM
    covers it.
    """
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
    from teb_vae.lag_attn.tests.conftest import PROD_KWARGS

    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(PROD_KWARGS, horizon_film=True)).eval()

    assert model.horizon_core.film_gen is not None
    assert model.horizon_core.film_gen.weight.abs().max().item() > 0.0, (
        "film_gen is unexpectedly still zero; this test's premise no longer holds"
    )

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*inputs)
    assert torch.equal(out["mu_full"], out["mu_base"])
    assert out["delta_mu_src"].abs().max().item() == 0.0


def test_horizon_core_expands_over_the_forecast_axis():
    torch.manual_seed(0)
    core = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON)
    h = torch.randn(_BATCH, _SEQ_LEN, _D_HIDDEN)
    with torch.no_grad():
        assert core.decode(h).shape == (_BATCH, _SEQ_LEN, _HORIZON, _D_HIDDEN)


def test_horizon_anchors_are_refined_independently():
    """The refine stack folds ``(B, T)`` into the batch, so one anchor cannot reach another.

    If it could, an anchor at $t$ would see a later anchor's state and the forecast would be
    quietly non-causal along the axis that matters.
    """
    torch.manual_seed(0)
    core = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON)
    h = torch.randn(_BATCH, _SEQ_LEN, _D_HIDDEN)
    perturbed = h.clone()
    perturbed[:, 5:] = torch.randn(_BATCH, _SEQ_LEN - 5, _D_HIDDEN)

    with torch.no_grad():
        assert torch.allclose(core.decode(h)[:, 4], core.decode(perturbed)[:, 4], atol=1e-6)
