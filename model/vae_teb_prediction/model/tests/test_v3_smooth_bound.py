"""S1-T01: smooth_bound helper + pre-bound raw prior logvar.

``clamp`` reproduces v1 (idempotent, zero gradient outside the range); ``smooth`` maps into
the open range with a strictly-positive gradient everywhere (including where clamp saturates);
``PriorHeadV3`` additionally exposes the pre-bound raw prior log-variance.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    PriorHeadV3,
    _apply_logvar_bound,
    smooth_bound,
)

_LO, _HI = -5.0, 3.0


def test_clamp_reproduces_torch_clamp():
    r = torch.linspace(-20.0, 20.0, 101)
    assert torch.allclose(
        _apply_logvar_bound(r, (_LO, _HI), "clamp"),
        torch.clamp(r, _LO, _HI),
        atol=0.0,
    )


def test_smooth_maps_into_open_range():
    # Kept within a range where float32 sigmoid does not round to exactly {0, 1}.
    r = torch.linspace(-15.0, 15.0, 61)
    out = smooth_bound(r, _LO, _HI)
    assert torch.all(out > _LO) and torch.all(out < _HI)
    # Monotone increasing.
    assert torch.all(out[1:] - out[:-1] >= 0)


def test_smooth_gradient_nonzero_at_extremes_but_clamp_zero():
    # float64 so sigmoid(20) is not rounded to exactly 1.0 (it is ~1 - 2e-9); the
    # smooth-bound gradient there is tiny (~1.6e-8) but strictly positive, whereas the
    # hard clamp gives an exactly-zero gradient once |r| exceeds the range.
    r = torch.tensor([20.0, -20.0], dtype=torch.float64, requires_grad=True)
    smooth_bound(r, _LO, _HI).sum().backward()
    assert torch.all(r.grad.abs() > 0), "smooth bound has zero gradient at |r|=20"

    r2 = torch.tensor([20.0, -20.0], dtype=torch.float64, requires_grad=True)
    _apply_logvar_bound(r2, (_LO, _HI), "clamp").sum().backward()
    assert torch.all(r2.grad == 0), "clamp should have zero gradient at |r|=20"


def test_prior_head_returns_pre_bound_raw_logvar():
    torch.manual_seed(0)
    head = PriorHeadV3(d_model=32, d_z=8, dropout=0.0, logvar_bound="smooth").eval()
    h = torch.randn(2, 5, 32)
    mu, logvar, dec, raw = head(h)
    assert raw.shape == logvar.shape == mu.shape == (2, 5, 8)
    # The reported logvar is the smooth bound of the returned raw value.
    assert torch.allclose(logvar, smooth_bound(raw, _LO, _HI), atol=1e-6)
    # Smooth logvar lies strictly inside the range.
    assert torch.all(logvar > _LO) and torch.all(logvar < _HI)
