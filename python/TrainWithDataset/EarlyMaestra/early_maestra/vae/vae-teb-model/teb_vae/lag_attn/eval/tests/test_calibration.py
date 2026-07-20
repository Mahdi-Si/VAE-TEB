r"""Tests for the calibration analysis.

The precondition test is the one that matters most. ``logvar_full`` is emitted on every forward
whatever the model was trained to do with it, so an analysis gated on its *presence* would score
an untrained variance head -- one that received no gradient at all -- and report a confident
calibration verdict about a quantity the objective never touched. The gate is the checkpoint's
resolved objective, and a checkpoint that fails it must produce a clean recorded skip rather than
numbers or an error.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures
from teb_vae.lag_attn.eval.analyses import calibration as calibration_analysis

PROBE = {"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4}


def _run(runner, loader, eval_config, output_dir):
    """Run the analysis and return the summary."""
    torch.manual_seed(11)
    return calibration_analysis.run_calibration_analysis(
        runner, loader, eval_config=eval_config, output_dir=output_dir, probe=PROBE
    )


# ---------------------------------------------------------------------------
# The precondition
# ---------------------------------------------------------------------------
def test_an_mse_checkpoint_produces_a_clean_recorded_skip(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Not an error: the checkpoint is valid, it simply has no predictive variance to score.

    Raising would set the run's exit code and report a healthy run as broken.
    """
    runner = make_eval_runner(
        hparams={"likelihood": "mse", "sigma_obs": 1.0}, output_dir=tmp_path / "runner"
    )
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "mse")

    assert summary["skipped"] is True
    assert "likelihood='mse'" in summary["reason"]
    assert "logvar_full is emitted on every forward" in summary["reason"]
    assert "mean_nll" not in summary
    assert not (tmp_path / "mse" / calibration_analysis.ANALYSIS_DIRNAME / "per_sample.csv").exists()


def test_a_fixed_sigma_obs_checkpoint_is_also_skipped(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Under a fixed observation noise the logvar head did not set the likelihood's variance."""
    runner = make_eval_runner(
        hparams={"likelihood": "gaussian_nll", "sigma_obs": 0.5},
        output_dir=tmp_path / "runner",
    )
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "fixed")
    assert summary["skipped"] is True
    assert "sigma_obs=0.5" in summary["reason"]


def test_the_shipped_objective_is_scored(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The shipped hparams are ``gaussian_nll`` with a learned sigma, which is scorable."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    assert calibration_analysis.is_applicable(runner) is None
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "ok")
    assert summary["skipped"] is False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_the_three_tables_are_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "tables")
    directory = tmp_path / "tables" / calibration_analysis.ANALYSIS_DIRNAME

    for name in ("per_sample.csv", "per_horizon.csv", "reliability.csv"):
        assert (directory / name).is_file(), f"{name} missing"

    frame = pd.read_csv(directory / "per_sample.csv")
    assert {"nll", "nll_homoscedastic", "nll_gain", "crps", "mean_logvar"} <= set(frame.columns)
    assert len(frame) == summary["n_samples"] == 4


def test_the_per_horizon_table_covers_every_horizon_step(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """A head calibrated at $h=1$ and over-confident at $h=H_d$ is averaged away by a scalar."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "horizon")

    per_horizon = pd.read_csv(
        tmp_path / "horizon" / calibration_analysis.ANALYSIS_DIRNAME / "per_horizon.csv"
    )
    assert set(per_horizon["horizon"]) == set(range(int(runner.model.horizon)))
    assert set(per_horizon["quantity"]) == {"nll", "coverage_2sigma"}


def test_the_reliability_table_bins_sum_to_a_uniform_density(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""The histogram is normalised to a density, so a calibrated model sits flat at $1$."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "rel")

    reliability = pd.read_csv(
        tmp_path / "rel" / calibration_analysis.ANALYSIS_DIRNAME / "reliability.csv"
    )
    assert len(reliability) == calibration_analysis.PIT_BINS
    # A density over [0, 1] integrates to 1, i.e. its mean over equal-width bins is 1.
    assert float(reliability["density"].mean()) == pytest.approx(1.0, rel=1e-6)
    assert bool((reliability["uniform"] == 1.0).all())


# ---------------------------------------------------------------------------
# The gain and its direction
# ---------------------------------------------------------------------------
def test_a_genuinely_heteroscedastic_variance_beats_the_reference(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The positive case: a variance that tracks the residual must show a positive gain.

    Substituted rather than trained, because the tiny fixture's variance head is untrained and
    would show whatever it happens to show -- which cannot distinguish a working analysis from
    one that reports a positive gain unconditionally.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    inner = runner.forecast_view

    def _oracle_view(batch, forward_outputs=None):
        view = inner(batch, forward_outputs)
        residual = (view.y_plus - view.mu_full) ** 2
        # A near-oracle variance: the squared residual itself, lightly smoothed so it stays a
        # legitimate prediction rather than a degenerate zero-variance point mass.
        object.__setattr__(
            view, "logvar_full", torch.log(residual.clamp_min(1e-3) * 0.9 + 1e-2)
        )
        return view

    runner.forecast_view = _oracle_view
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "oracle")

    assert summary["mean_nll_gain"] > 0.0
    assert summary["learned_variance_beats_homoscedastic"] is True


def test_a_constant_variance_cannot_beat_the_homoscedastic_reference(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The negative case, and the one that makes the flag meaningful.

    The reference is fitted by maximum likelihood to these very residuals, so *no* constant
    variance beats it -- the gain must come out non-positive and the flag must fire.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    inner = runner.forecast_view

    def _constant_view(batch, forward_outputs=None):
        view = inner(batch, forward_outputs)
        object.__setattr__(view, "logvar_full", torch.zeros_like(view.logvar_full))
        return view

    runner.forecast_view = _constant_view
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "constant")

    assert summary["mean_nll_gain"] <= 0.0
    assert summary["learned_variance_beats_homoscedastic"] is False


# ---------------------------------------------------------------------------
# Coverage nominals
# ---------------------------------------------------------------------------
def test_coverage_is_reported_against_the_exact_nominal_not_zero_point_ninety_five(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""$0.95$ is $\pm 1.96\sigma$, and using it would report a calibrated model as
    over-confident on every horizon."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "cov")

    assert set(summary["coverage"]) == {"1sigma", "2sigma", "3sigma"}
    assert summary["coverage"]["2sigma"]["nominal"] == pytest.approx(0.9545, abs=5e-5)
    assert summary["coverage"]["1sigma"]["nominal"] == pytest.approx(0.6827, abs=5e-5)
    for record in summary["coverage"].values():
        assert record["gap"] == pytest.approx(record["observed"] - record["nominal"])


def test_coverage_is_exact_on_synthetically_calibrated_predictions(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """End to end: substitute a truly calibrated Gaussian and the observed must hit the nominal.

    This is what proves the masking and the reduction in the analysis agree with the metric
    functions the unit tests pin -- an analysis can honour every nominal and still mask wrongly.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    inner = runner.forecast_view
    generator = torch.Generator().manual_seed(17)

    def _calibrated_view(batch, forward_outputs=None):
        view = inner(batch, forward_outputs)
        sigma = 0.75
        noise = torch.randn(view.y_plus.shape, generator=generator) * sigma
        object.__setattr__(view, "y_plus", view.mu_full + noise)
        object.__setattr__(
            view, "logvar_full", torch.full_like(view.logvar_full, 2.0 * np.log(sigma))
        )
        return view

    runner.forecast_view = _calibrated_view
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "calibrated")

    for label in ("1sigma", "2sigma", "3sigma"):
        record = summary["coverage"][label]
        assert record["observed"] == pytest.approx(record["nominal"], abs=0.01)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def test_three_figures_are_written_with_data(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch
) -> None:
    captured: dict = {}
    original = figures.render_to_pdf

    def _capture(fig, path, **kwargs):
        captured[Path(path).name] = [
            ax.has_data() for ax in fig.axes if ax.get_title()
        ]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_to_pdf", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "figs")

    assert set(captured) == {"reliability.pdf", "coverage.pdf", "sharpness.pdf"}
    assert all(all(panels) for panels in captured.values())
    assert len(summary["figures"]) == 3
    for path in summary["figures"]:
        assert Path(path).suffix == ".pdf" and Path(path).stat().st_size > 0


def test_the_reliability_figure_marks_the_uniform_reference(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path, monkeypatch
) -> None:
    """Without the reference line a PIT curve is a shape with no scale to read it against."""
    captured: dict = {}
    original = figures.render_to_pdf

    def _capture(fig, path, **kwargs):
        if Path(path).name == "reliability.pdf":
            captured["labels"] = [
                line.get_label() for line in fig.axes[0].get_lines()
            ]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_to_pdf", _capture)
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    _run(runner, tiny_loader, tiny_eval_config["eval_config"], tmp_path / "relfig")

    assert "uniform" in captured["labels"]
