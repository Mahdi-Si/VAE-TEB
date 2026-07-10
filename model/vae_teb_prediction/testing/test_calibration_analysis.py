r"""S5-T02: the calibration collector and analysis, driven through a real ``TestRunner``.

``logvar_full`` was always emitted by the forward pass -- it simply had no reader. These tests
pin that it now reaches the report, that the report degrades gracefully on a model that does
not emit it, and that the constant-sigma reference is computed, because that reference is the
only thing standing between "the learned variance head is calibrated" and "the learned
variance head is untrained and nobody noticed".
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import model.vae_teb_prediction.testing.base as base_module
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3
from model.vae_teb_prediction.testing.analyses.calibration import run_calibration_analysis
from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_calibration

_KWARGS = dict(
    sequence_length=40, d_model=32, d_z=8, horizon=6, warmup_period=4,
    c_y=87, c_u=101, use_up_st=True, max_lag=8, num_heads=4, d_head=8, dropout=0.0,
    causal_norm=True, posterior_logvar="residual", logvar_bound="smooth",
    kld_support="anchor",
)
_WARMUP, _HORIZON = 4, 6


class _Batch:
    """The batch fields ``TestRunner.forward`` and ``build_future_target`` read."""

    def __init__(self, n: int = 4, T: int = 40, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.fhr_st = torch.randn(n, T, 43, generator=g)
        self.fhr_ph = torch.randn(n, T, 44, generator=g)
        self.up_st = torch.randn(n, T, 43, generator=g)
        self.up_ph = torch.randn(n, T, 58, generator=g)
        self.weight = torch.ones(n, T)


@pytest.fixture(autouse=True)
def _v3_alias(monkeypatch):
    """Point the pipeline's model-class alias at v3 for the duration of a test."""
    monkeypatch.setattr(base_module, "SeqVaeLagAttn", SeqVaeLagAttnV3)


@pytest.fixture
def runner(tmp_path) -> TestRunner:
    torch.manual_seed(0)
    model = SeqVaeLagAttnV3(**_KWARGS).eval()
    return TestRunner(
        model=model, device=torch.device("cpu"), output_dir=tmp_path,
        warmup_steps=_WARMUP, horizon=_HORIZON, max_lag=8, use_up_st=True,
    )


@pytest.fixture
def loader():
    return [_Batch(seed=0), _Batch(seed=1)]


def test_collector_surfaces_the_decoder_observation_logvar(runner, loader):
    """The Sprint-5 gap: ``logvar_full`` exists in the forward but no collector read it."""
    collected = collect_calibration(runner, loader, max_samples=8)

    per_sample = collected["per_sample"]
    assert len(per_sample) == 8
    for column in ("nll", "crps", "sharpness", "coverage_50", "coverage_95"):
        assert column in per_sample.columns
        assert per_sample[column].notna().all()
    # Sharpness is a standard deviation, floored by the smooth bound at exp(-5/2).
    assert (per_sample["sharpness"] > 0.0).all()


def test_collector_emits_one_row_per_horizon_step(runner, loader):
    collected = collect_calibration(runner, loader, max_samples=8)
    per_horizon = collected["per_horizon"]
    assert sorted(per_horizon["h"].unique()) == list(range(_HORIZON))
    assert len(per_horizon) == 8 * _HORIZON


def test_collector_fits_the_constant_sigma_reference(runner, loader):
    collected = collect_calibration(runner, loader, max_samples=8)
    summary = collected["summary"]
    assert summary["constant_sigma"] > 0.0
    assert "nll_gain_over_constant" in summary
    # An untrained variance head cannot beat a single fitted global sigma.
    assert summary["nll_gain_over_constant"] < 0.0


def test_reliability_curve_is_monotone_and_bounded(runner, loader):
    reliability = collect_calibration(runner, loader, max_samples=8)["reliability"]
    assert not reliability.empty
    assert reliability["nominal"].between(0.0, 1.0).all()
    assert reliability["empirical"].between(0.0, 1.0).all()
    for _, group in reliability.groupby("h"):
        ordered = group.sort_values("nominal")["empirical"].to_numpy()
        assert (ordered[1:] - ordered[:-1] >= -1e-6).all(), "empirical CDF decreased"


def test_analysis_writes_the_full_report(runner, loader, tmp_path):
    summary = run_calibration_analysis(runner, loader, max_samples=8)

    out = tmp_path / "calibration"
    for name in ("per_sample.csv", "per_horizon.csv", "reliability.csv", "summary.json"):
        assert (out / name).is_file(), f"{name} was not written"
    for name in ("reliability.pdf", "coverage.pdf", "sharpness_by_horizon.pdf"):
        assert (out / name).stat().st_size > 0, f"{name} is empty"

    on_disk = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["n_samples"] == 8
    for key in ("nll_mean", "crps_mean", "sharpness_mean", "constant_sigma",
                "coverage_90", "coverage_90_error", "reliability_max_deviation"):
        assert key in summary and key in on_disk, f"{key} missing from the summary"

    per_sample = pd.read_csv(out / "per_sample.csv")
    assert summary["nll_mean"] == pytest.approx(per_sample["nll"].mean(), rel=1e-6)


def test_analysis_skips_cleanly_when_the_model_emits_no_logvar(tmp_path, loader):
    """An ``mse``-era checkpoint must produce a logged skip, not a pipeline crash."""

    class _NoLogvar(SeqVaeLagAttnV3):
        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            outputs.pop("logvar_full")
            return outputs

    torch.manual_seed(0)
    runner = TestRunner(
        model=_NoLogvar(**_KWARGS).eval(), device=torch.device("cpu"), output_dir=tmp_path,
        warmup_steps=_WARMUP, horizon=_HORIZON, max_lag=8, use_up_st=True,
    )
    result = run_calibration_analysis(runner, loader, max_samples=4)

    assert "error" in result and "logvar_full" in result["error"]
    assert not (Path(tmp_path) / "calibration" / "per_sample.csv").exists()


def test_analysis_honours_the_skip_switch(runner, loader):
    assert run_calibration_analysis(runner, loader, max_samples=0) == {}


def test_a_well_calibrated_model_would_score_near_nominal(runner, loader):
    r"""Sanity-check the wiring end to end: feed the kernels a calibrated distribution.

    The untrained model's own :math:`\sigma` is meaningless, so this substitutes the residual
    scale that *is* correct by construction and asserts the pipeline recovers nominal coverage.
    That isolates a wiring bug (wrong slice, wrong axis) from a genuinely mis-calibrated model.
    """
    from model.vae_teb_prediction.testing.metrics import compute_interval_coverage

    batch = loader[0]
    with runner.inference_mode():
        outputs = runner.forward(batch)
        y_plus = runner.build_future_target(batch)

    mu = outputs["mu_full"]
    resid = y_plus[:, _WARMUP : mu.shape[1] - _HORIZON] - mu[:, _WARMUP : mu.shape[1] - _HORIZON]
    sigma = resid.std()
    logvar = torch.full_like(mu, float(2.0 * torch.log(sigma)))

    coverage = compute_interval_coverage(mu, logvar, y_plus, _WARMUP, _HORIZON)
    empirical = coverage["coverage"].mean(dim=0)
    assert torch.allclose(empirical, coverage["nominal"], atol=0.06), empirical
