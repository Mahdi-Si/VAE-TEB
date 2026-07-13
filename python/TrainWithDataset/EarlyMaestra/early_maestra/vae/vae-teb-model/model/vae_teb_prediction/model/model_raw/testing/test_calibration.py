r"""S6-T05: raw calibration (G10) -- single-block predictive Gaussian, no scattering split.

Covers two things: (a) the calibration math is well-behaved -- a target drawn from the model's own
$\mathcal{N}(\mu, \sigma^2)$ yields near-nominal interval coverage; (b) the end-to-end analysis on
the tiny fixture writes ``per_sample.csv`` / ``per_horizon.csv`` / ``reliability.csv`` and the
per-sample table carries **no** scattering ``nll_st`` / ``nll_ph`` / ``crps_st`` / ``crps_ph`` columns.
"""
from __future__ import annotations

import pandas as pd
import torch

from model.vae_teb_prediction.model.model_raw.testing.analyses.calibration import (
    run_calibration_analysis,
)
from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.conftest import make_raw_stub_batch
from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    compute_crps,
    compute_interval_coverage,
    compute_nll,
)

_B, _T, _H, _R = 6, 28, 4, 16
_WARMUP = 2
_T_VALID = _T - _H
_LEVELS = (0.5, 0.8, 0.9, 0.95)


def test_compute_nll_crps_single_block_no_st_ph() -> None:
    """``compute_nll`` / ``compute_crps`` return total + per_horizon only (no st/ph)."""
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(_B, _T, _H, _R, generator=g)
    logvar = torch.zeros(_B, _T, _H, _R)
    x_plus = torch.randn(_B, _T_VALID, _H, _R, generator=g)
    nll = compute_nll(mu, logvar, x_plus, _WARMUP, _H)
    crps = compute_crps(mu, logvar, x_plus, _WARMUP, _H)
    assert set(nll) == {"nll_total", "nll_per_horizon"}
    assert set(crps) == {"crps_total", "crps_per_horizon"}
    assert tuple(nll["nll_per_horizon"].shape) == (_B, _H)


def test_well_calibrated_gaussian_gives_nominal_coverage() -> None:
    r"""A target drawn from $\mathcal{N}(\mu, \sigma^2)$ has interval coverage near nominal."""
    torch.manual_seed(1)
    n = 400
    mu = torch.randn(n, _T, _H, _R)
    sigma = 0.7
    logvar = torch.full((n, _T, _H, _R), float(torch.log(torch.tensor(sigma ** 2))))
    # Draw the target from the model's own predictive distribution over the valid anchors.
    x_plus = mu[:, :_T_VALID] + sigma * torch.randn(n, _T_VALID, _H, _R)
    cov = compute_interval_coverage(mu, logvar, x_plus, _WARMUP, _H, levels=_LEVELS)
    emp = cov["coverage"].mean(dim=0)  # (n_levels,)
    for j, level in enumerate(_LEVELS):
        assert abs(float(emp[j]) - level) < 0.05, f"level {level}: empirical {float(emp[j]):.3f}"


def _runner(tiny_checkpoint, tmp_path) -> TestRunner:
    ckpt_path, _ = tiny_checkpoint
    return TestRunner.from_checkpoint(ckpt_path, tmp_path / "out")


def test_run_calibration_analysis_writes_files_no_channel_split(tiny_checkpoint, tmp_path) -> None:
    """The end-to-end raw calibration writes the three CSVs with no st/ph columns."""
    runner = _runner(tiny_checkpoint, tmp_path)
    loader = [make_raw_stub_batch(batch_size=_B, raw_len=512, seed=s) for s in range(3)]
    out = tmp_path / "calib"
    summary = run_calibration_analysis(runner, loader, max_samples=None, output_dir=out)

    assert "error" not in summary
    assert (out / "per_sample.csv").exists()
    assert (out / "per_horizon.csv").exists()
    assert (out / "reliability.csv").exists()

    df = pd.read_csv(out / "per_sample.csv")
    assert {"nll", "crps", "sharpness"}.issubset(df.columns)
    for bad in ("nll_st", "nll_ph", "crps_st", "crps_ph"):
        assert bad not in df.columns
    # A coverage column for every requested level.
    for level in _LEVELS:
        assert f"coverage_{int(round(level * 100))}" in df.columns
