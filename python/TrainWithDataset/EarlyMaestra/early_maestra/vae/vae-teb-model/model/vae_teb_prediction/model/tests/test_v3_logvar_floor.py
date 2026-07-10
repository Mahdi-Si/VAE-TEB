"""S1-T05: decoder observation log-variance cannot collapse below the lower bound.

Under the smooth bound, any finite raw head output maps into the open range ``(lo, hi)``,
so the decoder observation log-variance is floored at ``lo`` (guarding against
``sigma^2 -> 0`` / ``NLL -> -inf`` collapse) even under extreme-magnitude stress input.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

_LO, _HI = -5.0, 3.0
_TOL = 1e-4


def _smooth_model(tiny_kwargs):
    return SeqVaeLagAttnV3(
        **dict(tiny_kwargs, posterior_logvar="residual", logvar_bound="smooth")
    ).eval()


def test_decoder_logvar_floored_under_stress(tiny_kwargs):
    model = _smooth_model(tiny_kwargs)
    d_model = tiny_kwargs["d_model"]
    d_z = tiny_kwargs["d_z"]

    # Extreme-magnitude decoder state / latent to saturate the raw logvar heads.
    big_state = torch.randn(2, 8, d_model) * 1e3
    big_z = torch.randn(2, 8, d_z) * 1e3

    _, logvar_base = model.baseline_decoder(big_state)
    _, logvar_full = model.residual_decoder(big_state, big_z)

    for name, lv in (("logvar_base", logvar_base), ("logvar_full", logvar_full)):
        assert torch.isfinite(lv).all(), f"{name} not finite under stress"
        assert lv.min().item() >= _LO - _TOL, f"{name} fell below lo: {lv.min().item()}"
        assert lv.max().item() <= _HI + _TOL, f"{name} exceeded hi: {lv.max().item()}"


def test_forward_logvars_bounded_on_large_inputs(tiny_kwargs):
    model = _smooth_model(tiny_kwargs)
    g = torch.Generator().manual_seed(0)
    y_st = torch.randn(2, 16, 43, generator=g) * 50.0
    y_ph = torch.randn(2, 16, 44, generator=g) * 50.0
    u = torch.randn(2, 16, 101, generator=g) * 50.0
    out = model(y_st, y_ph, u)
    for k in ("logvar_full", "logvar_base", "logvar_prior", "logvar_post"):
        lv = out[k]
        assert torch.isfinite(lv).all()
        assert lv.min().item() >= _LO - _TOL, f"{k} below lo"
        assert lv.max().item() <= _HI + _TOL, f"{k} above hi"
