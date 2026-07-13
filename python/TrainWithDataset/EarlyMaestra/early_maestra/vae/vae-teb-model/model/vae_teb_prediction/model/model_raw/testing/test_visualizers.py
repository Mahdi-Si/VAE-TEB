r"""S7-T01: raw-domain visualizers + raw-adapted per-sample diagnostics.

Covers the standalone raw plots in :mod:`visualizers_raw` (each writes a non-empty file; the
$(H \times T)$ heatmap returns a $(H, T_v)$ array) and smoke-tests the two adapted per-sample
diagnostics (``run_sample_diagnostics`` raw overlay, ``run_kld_lag_diagnostics`` with the scattering
panel dropped) end-to-end on the tiny fixture -- proving they run on raw batches.
"""
from __future__ import annotations

import numpy as np
import torch

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_stub_batch
from model.vae_teb_prediction.model.model_raw.testing import visualizers_raw as vr

_B, _T, _H, _R = 3, 28, 4, 16
_WARMUP = 2
_T_VALID = _T - _H


def _fake(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(_B, _T, _H, _R, generator=g)
    logvar = torch.zeros(_B, _T, _H, _R)
    x_plus = torch.randn(_B, _T_VALID, _H, _R, generator=g)
    return mu, logvar, x_plus


def _nonempty(path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_forecast_overlay_writes_file(tmp_path) -> None:
    mu, logvar, x_plus = _fake()
    out = vr.plot_raw_forecast_overlay(
        mu[0, 5], x_plus[0, 5], tmp_path / "overlay.pdf",
        logvar_anchor=logvar[0, 5], fhr_stats={"mean": 140.0, "std": 15.0},
    )
    assert _nonempty(out)


def test_forecast_overlay_averaged_writes_file(tmp_path) -> None:
    mu, logvar, x_plus = _fake(1)
    out = vr.plot_raw_forecast_overlay_averaged(
        mu, x_plus, tmp_path / "overlay_avg.pdf",
        warmup=_WARMUP, horizon=_H, logvar=logvar,
    )
    assert _nonempty(out)


def test_per_horizon_error_writes_file(tmp_path) -> None:
    mse_per_h = torch.rand(_B, _H)
    out = vr.plot_raw_per_horizon_error(mse_per_h, tmp_path / "per_horizon.pdf")
    assert _nonempty(out)


def test_lowpass_error_writes_file(tmp_path) -> None:
    mu, _lv, x_plus = _fake(2)
    out = vr.plot_raw_lowpass_error(
        mu, x_plus, tmp_path / "lowpass.pdf",
        warmup=_WARMUP, horizon=_H, scales_sec=(4, 16, 32), fs=4,
    )
    assert _nonempty(out)


def test_forecast_heatmap_shape_and_file(tmp_path) -> None:
    mu, _lv, _x = _fake(3)
    out, heat = vr.plot_raw_forecast_heatmap(
        mu, tmp_path / "heatmap.pdf", warmup=_WARMUP, horizon=_H
    )
    assert _nonempty(out)
    # (H x T_valid): horizon runs down the rows, anchor across the columns.
    assert heat.shape == (_H, _T_VALID)
    assert isinstance(heat, np.ndarray)


# ---- adapted per-sample diagnostics run on raw batches -----------------------

def _runner(tiny_checkpoint, tmp_path) -> TestRunner:
    ckpt_path, _ = tiny_checkpoint
    return TestRunner.from_checkpoint(ckpt_path, tmp_path / "out")


def test_run_sample_diagnostics_raw_overlay(tiny_checkpoint, tmp_path) -> None:
    """``run_sample_diagnostics`` emits a raw overlay PDF per sample (no fhr_st needed)."""
    from model.vae_teb_prediction.model.model_raw.testing.analyses.qualitative import (
        run_sample_diagnostics,
    )

    runner = _runner(tiny_checkpoint, tmp_path)
    loader = [make_raw_stub_batch(batch_size=2, raw_len=512)]
    res = run_sample_diagnostics(runner, loader, max_samples=2, output_dir=tmp_path / "sd")
    assert res["n_plotted"] == 2
    pdfs = list((tmp_path / "sd").glob("*.pdf"))
    assert len(pdfs) == 2 and all(_nonempty(p) for p in pdfs)


def test_run_kld_lag_diagnostics_raw_no_fhr_st(tiny_checkpoint, tmp_path) -> None:
    """``run_kld_lag_diagnostics`` runs on raw batches (scattering panel dropped)."""
    from model.vae_teb_prediction.model.model_raw.testing.analyses.kld_lag_diagnostics import (
        run_kld_lag_diagnostics,
    )

    runner = _runner(tiny_checkpoint, tmp_path)
    loader = [make_raw_stub_batch(batch_size=2, raw_len=512)]
    res = run_kld_lag_diagnostics(runner, loader, max_samples=2, output_dir=tmp_path / "kld")
    assert res["n_plotted"] >= 1
    pdfs = list((tmp_path / "kld").glob("*.pdf"))
    assert len(pdfs) >= 1 and all(_nonempty(p) for p in pdfs)
