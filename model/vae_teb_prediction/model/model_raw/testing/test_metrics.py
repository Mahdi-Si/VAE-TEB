r"""S6-T03: raw-waveform forecast metrics (no scattering st/ph channel split).

Exercises :func:`compute_raw_forecast_metrics` on synthetic $(B, T, H, R)$ tensors -- no model
needed. Confirms the perfect-prediction fixed points (VAF=1, MSE=0, $R^2$=1, low-pass MSE=0), the
per-horizon shape, and the absence of the scattering ``mse_st`` / ``mse_ph`` keys.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    compute_raw_forecast_metrics,
)

_B, _T, _H, _R = 3, 28, 4, 16
_WARMUP = 2
_T_VALID = _T - _H  # 24


def _mk(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(_B, _T, _H, _R, generator=g)
    x_plus = torch.randn(_B, _T_VALID, _H, _R, generator=g)
    return mu, x_plus


def test_shapes_and_no_channel_split() -> None:
    """Returns the raw metric keys with the right shapes and NO st/ph columns."""
    mu, x_plus = _mk()
    out = compute_raw_forecast_metrics(mu, x_plus, _WARMUP, _H)
    assert set(out) == {
        "raw_vaf", "raw_mse", "raw_snr", "raw_r2", "raw_mse_per_horizon", "raw_lowpass_mse",
    }
    assert tuple(out["raw_mse"].shape) == (_B,)
    assert tuple(out["raw_mse_per_horizon"].shape) == (_B, _H)
    # No scattering channel-block columns anywhere.
    for key in out:
        assert not key.endswith("_st")
        assert not key.endswith("_ph")


def test_perfect_prediction_fixed_points() -> None:
    """mu == x_plus -> VAF=1, MSE=0, R^2=1, low-pass MSE=0."""
    _mu, x_plus = _mk(seed=1)
    # Build a full-T prediction whose valid slice equals x_plus exactly.
    mu = torch.zeros(_B, _T, _H, _R)
    mu[:, _WARMUP:_T_VALID] = x_plus[:, _WARMUP:_T_VALID]
    out = compute_raw_forecast_metrics(mu, x_plus, _WARMUP, _H)
    assert torch.allclose(out["raw_mse"], torch.zeros(_B), atol=1e-6)
    assert torch.allclose(out["raw_vaf"], torch.ones(_B), atol=1e-5)
    assert torch.allclose(out["raw_r2"], torch.ones(_B), atol=1e-5)
    assert torch.allclose(out["raw_lowpass_mse"], torch.zeros(_B), atol=1e-6)
    assert torch.allclose(out["raw_mse_per_horizon"], torch.zeros(_B, _H), atol=1e-6)


def test_horizon_mismatch_raises() -> None:
    """A wrong ``horizon`` argument is caught."""
    mu, x_plus = _mk()
    try:
        compute_raw_forecast_metrics(mu, x_plus, _WARMUP, _H + 1)
    except ValueError:
        return
    raise AssertionError("expected ValueError on horizon mismatch")


def test_empty_valid_range_returns_zeros() -> None:
    """A warm-up that swallows every anchor yields all-zero metrics, not a crash."""
    mu, x_plus = _mk()
    out = compute_raw_forecast_metrics(mu, x_plus, warmup=_T, horizon=_H)
    assert torch.allclose(out["raw_mse"], torch.zeros(_B))
    assert tuple(out["raw_mse_per_horizon"].shape) == (_B, _H)


def test_lowpass_rewards_trend_over_jitter() -> None:
    """A pred matching the block-mean trend but not the jitter has low low-pass MSE."""
    g = torch.Generator().manual_seed(7)
    trend = torch.randn(_B, _T_VALID, _H, 1, generator=g)          # smooth within-block level
    jitter = 0.5 * torch.randn(_B, _T_VALID, _H, _R, generator=g)
    x_plus = trend + jitter
    mu = torch.zeros(_B, _T, _H, _R)
    mu[:, _WARMUP:_T_VALID] = trend.expand(_B, _T_VALID, _H, _R)[:, _WARMUP:_T_VALID]
    out = compute_raw_forecast_metrics(mu, x_plus, _WARMUP, _H, lowpass_scales=(4,), fs=4)
    # At a 4 s (=16-sample=R) block, the trend-only prediction matches the block mean, so the
    # low-pass MSE is far below the raw per-sample MSE (which still pays for the jitter).
    assert (out["raw_lowpass_mse"] < out["raw_mse"]).all()
