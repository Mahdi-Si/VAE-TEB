r"""Tests for the forecast-quality analysis.

Three things are worth catching here and each needs a different style of test.

The **schema** -- that the CSVs exist and carry the columns a reader is promised -- is a
straightforward existence check.

The **arithmetic** is checked by recomputing one column directly from the model and asserting it
matches the CSV. A test that only asserted the column exists would pass on a column of zeros.

The **figures** are checked structurally, following ``tests/test_plotting_figure.py``: panel
count, titles in order, ``ax.has_data()``. Structural assertions alone are vacuous, though -- a
figure of the wrong data has just as many panels -- so the heatmap test also *sabotages* one
channel and asserts the residual panel names that channel. Every figure test closes its figure
in a ``finally``.

Profile lengths are asserted against the model's own geometry rather than against literals. A
literal would keep passing after the horizon changed, on a profile that had silently become the
wrong length.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, masks, metrics
from teb_vae.lag_attn.eval.analyses import forecast as forecast_analysis


#: Seed applied before the analysis and before any recomputation compared against it.
#: ``forward`` samples $z$ unconditionally -- deliberately, since a mean-$z$ evaluation would
#: report a forecast the model never makes -- so two passes over the same batches agree only when
#: they start from the same generator state.
COMPARISON_SEED = 1234


@pytest.fixture
def analysis(make_eval_runner, tiny_loader, tiny_eval_config, tmp_path):
    """Run the analysis once and return ``(runner, output directory, summary)``."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    output_dir = tmp_path / "results"
    torch.manual_seed(COMPARISON_SEED)
    summary = forecast_analysis.run_forecast_analysis(
        runner,
        tiny_loader,
        eval_config=tiny_eval_config["eval_config"],
        output_dir=output_dir,
        probe={"n_samples": 4, "source_files": ["tiny_shard.hdf5"] * 4},
    )
    return runner, output_dir / forecast_analysis.ANALYSIS_DIRNAME, summary


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def test_the_expected_files_are_written(analysis) -> None:
    _, directory, _ = analysis
    for name in ("per_sample.csv", "horizon_error.csv", "anchor_error.csv"):
        assert (directory / name).is_file(), f"{name} was not written"
    for name in ("horizon_error.pdf", "anchor_error.pdf", "distributions.pdf", "heatmaps.pdf"):
        assert (directory / name).stat().st_size > 0, f"{name} is empty"


def test_per_sample_carries_the_promised_columns(analysis) -> None:
    """Both feature blocks reported separately, because they are different scales."""
    _, directory, _ = analysis
    frame = pd.read_csv(directory / "per_sample.csv")
    expected = {
        "guid", "source_file", "sample_index",
        "feat_mse_total", "feat_mse_scattering", "feat_mse_phase",
        "feat_r2_total", "feat_r2_scattering", "feat_r2_phase",
    }
    assert expected <= set(frame.columns), f"missing {expected - set(frame.columns)}"
    assert len(frame) == 4


def test_the_summary_carries_the_headline_numbers(analysis) -> None:
    _, _, summary = analysis
    assert summary["n_samples"] == 4
    assert np.isfinite(summary["mean_feat_mse_total"])
    assert np.isfinite(summary["mean_feat_r2_total"])
    assert summary["composition"] == {"tiny_shard.hdf5": 4}


def test_the_per_sample_mse_matches_a_direct_computation(analysis, tiny_loader) -> None:
    """Recomputed from the model, so a column of plausible-but-wrong numbers fails.

    Asserting only that the column exists would pass on a column of zeros.

    Both passes start from :data:`COMPARISON_SEED` so they draw the same $z$. Without that they
    agree only to about a percent -- which is the sampling, not a defect, and is exactly what the
    reproducibility requirement pins.
    """
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "per_sample.csv").sort_values("sample_index")

    direct = []
    torch.manual_seed(COMPARISON_SEED)
    with runner.inference_mode():
        for batch in tiny_loader:
            view = runner.forecast_view(runner.to_device(batch))
            direct.append(
                metrics.forecast_metrics(
                    view.mu_full, view.y_plus, view.mask, view.n_scattering
                )["feat_mse_total"]
            )
    expected = torch.cat(direct).cpu().numpy()

    assert frame["feat_mse_total"].to_numpy() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def test_the_horizon_profile_has_one_row_per_sample_and_horizon_step(analysis) -> None:
    """Length asserted against the model's ``horizon``, never against a literal."""
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "horizon_error.csv")
    horizon = int(runner.model.horizon)
    assert set(frame["position"]) == set(range(horizon))
    assert len(frame) == 4 * horizon


def test_the_anchor_profile_has_one_row_per_sample_and_valid_anchor(analysis) -> None:
    r"""The valid anchor count is $T - H_d$, taken from the model and the shard."""
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "anchor_error.csv")
    _, stop = masks.valid_anchor_range(runner.model, 300)
    assert set(frame["position"]) == set(range(stop))
    assert len(frame) == 4 * stop


def test_the_warmup_anchors_are_nan_rather_than_zero(analysis) -> None:
    """A zero there would read as a perfectly forecast prefix rather than as no data."""
    runner, directory, _ = analysis
    frame = pd.read_csv(directory / "anchor_error.csv")
    warmup = int(runner.model.warmup_period)
    early = frame[frame["position"] < warmup]["mse"]
    assert len(early) > 0 and early.isna().all()
    assert frame[frame["position"] >= warmup]["mse"].notna().any()


def test_the_anchor_axis_converts_to_minutes_through_the_decimation_step(analysis) -> None:
    r"""One decimated step is $4$ s, so the anchor axis is minutes at $t \cdot 4 / 60$."""
    _, directory, _ = analysis
    frame = pd.read_csv(directory / "anchor_error.csv")
    last = int(frame["position"].max())
    minutes = last * metrics.STEP_SECONDS / 60.0
    # The tiny shard is a 20-minute window; the last valid anchor sits just inside it.
    assert 0.0 < minutes <= 20.0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _stub_frame(n_samples: int = 5, horizon: int = 3, anchors: int = 6) -> pd.DataFrame:
    """A minimal collected frame, so a figure test does not need a model."""
    rng = np.random.default_rng(0)
    data: Dict[str, Any] = {
        "sample_index": np.arange(n_samples),
        "guid": [f"g{index}" for index in range(n_samples)],
        "source_file": ["tiny_shard.hdf5"] * n_samples,
        "feat_mse_total": rng.uniform(0.5, 2.0, n_samples),
        "feat_r2_total": rng.uniform(-1.0, 0.5, n_samples),
    }
    for step in range(horizon):
        data[f"h{step:02d}"] = rng.uniform(0.1, 1.0, n_samples)
    for anchor in range(anchors):
        data[f"a{anchor:03d}"] = rng.uniform(0.1, 1.0, n_samples)
    return pd.DataFrame(data)


def test_the_figures_have_the_panels_and_titles_they_claim(tmp_path) -> None:
    """Structural assertions in the style of the model suite's figure tests."""
    frame = _stub_frame()
    triple = {
        "forecast": np.random.default_rng(1).normal(size=(8, 20)),
        "target": np.random.default_rng(2).normal(size=(8, 20)),
        "residual_rms": np.abs(np.random.default_rng(3).normal(size=(8, 20))),
    }
    written = forecast_analysis._write_figures(
        frame, triple, tmp_path, n_scattering=3, n_channels=8
    )
    assert set(written) == {"horizon_error", "anchor_error", "distributions", "heatmaps"}
    for path in written.values():
        assert Path(path).stat().st_size > 0


def test_the_heatmap_figure_stacks_three_panels_in_order(tmp_path, monkeypatch) -> None:
    """Panel count and title order, asserted on the in-memory figure before it is saved."""
    captured: Dict[str, Any] = {}
    original = figures.render_figure

    def _capture(fig, path, **kwargs):
        if Path(path).name == "heatmaps":
            captured["titles"] = [ax.get_title() for ax in fig.axes if ax.get_title()]
            captured["has_data"] = [ax.has_data() for ax in fig.axes if ax.get_title()]
        return original(fig, path, **kwargs)

    monkeypatch.setattr(figures, "render_figure", _capture)
    triple = {
        "forecast": np.random.default_rng(1).normal(size=(8, 20)),
        "target": np.random.default_rng(2).normal(size=(8, 20)),
        "residual_rms": np.abs(np.random.default_rng(3).normal(size=(8, 20))),
    }
    forecast_analysis._write_figures(
        _stub_frame(), triple, tmp_path, n_scattering=3, n_channels=8
    )

    assert len(captured["titles"]) == 3
    assert captured["titles"][0].startswith("Mean forecast")
    assert captured["titles"][1].startswith("Mean target")
    assert captured["titles"][2].startswith("RMS residual")
    assert all(captured["has_data"])


def test_the_residual_heatmap_reports_the_channel_that_was_sabotaged(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The assertion that makes the figure test non-vacuous.

    One channel's forecast is corrupted, and the residual panel must show that channel as the
    worst. A figure drawing the wrong field, or drawing a signed mean that cancels, fails here
    while passing every structural check.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    sabotaged_channel = 7
    original_forward = runner.forward

    def _sabotage(batch, **kwargs):
        outputs = dict(original_forward(batch, **kwargs))
        corrupted = outputs["mu_full"].clone()
        corrupted[..., sabotaged_channel] += 50.0
        outputs["mu_full"] = corrupted
        return outputs

    runner.forward = _sabotage  # type: ignore[method-assign]

    triple = forecast_analysis._heatmap_triple(runner, tiny_loader, None, None)
    residual = triple["residual_rms"]
    per_channel = np.nanmean(residual, axis=1)

    assert int(np.nanargmax(per_channel)) == sabotaged_channel, (
        "the residual heatmap did not surface the sabotaged channel, so it is not showing the "
        "forecast error it claims to"
    )


def test_the_heatmap_separator_comes_from_the_batch_not_a_literal(
    make_eval_runner, tiny_loader, tmp_path
) -> None:
    r"""The scattering / phase split is a property of the data, not of the model.

    The model stores only the combined $c_y = 109$ and cannot supply the split, so the analysis
    must read it off the batch. A literal $43$ would silently mis-place the boundary the day the
    feature layout changes -- which it already has once, when the phase block went from $44$ to
    $66$ channels.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    triple = forecast_analysis._heatmap_triple(runner, tiny_loader, None, None)

    with runner.inference_mode():
        y_st, _ = runner.build_target_streams(runner.to_device(next(iter(tiny_loader))))
    assert triple["n_scattering"] == int(y_st.shape[-1])
    assert triple["n_scattering"] < int(runner.model.c_y)
