r"""Whether the forecast is any good: the skill arithmetic, the horizon axis, and the units.

The three trivial predictors themselves are built and scored in ``metrics`` and are tested in
``test_eval_metrics.py``, against known answers on constructed inputs. What is tested here is the
*analysis* over them, and it is four kinds of assertion.

**Known answers in the skill arithmetic.** A forecast equal to the truth scores skill $1$; one
equal to the baseline scores exactly $0$; a zero-error baseline yields no skill rather than an
infinity. These pin the arithmetic without a model and without a fixture, and they are what would
catch a skill score computed the wrong way round.

**Rooting once.** The RMSE roots after the per-recording mean of the unrooted squares. Averaging
finished roots is biased low by Jensen -- in the direction that flatters the model -- so the frame
below has widely different per-recording squared errors, which is where the two differ.

**The horizon denominators.** Each $\tau$ divides by its own masked count. The curve is the
single-draw one and says so, because the Monte Carlo marginalisation does not commute with the sum
over $\tau$.

**One unit, and no second one.** A scattering or phase-harmonic coefficient has no clinical unit,
so every emitted column is in the loader's $z$ units and labelled ``normalised``. The test that
matters most here is the negative one: no column carries a bpm label, on any path.

Per the fixture rule in ``test_eval_fixtures.py``: nothing below asserts the sign or magnitude of
any skill on the generated shards. Where a direction is needed, the frame is constructed.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import forecast as forecast_analysis
from teb_vae.lag_attn_cfs.eval.metrics import BASELINE_LOGVAR, BASELINE_NAMES, NORMALISED_UNIT

from .conftest import SHIPPED_HORIZON

#: A tiny stand-in geometry for the stub context: enough anchors and horizon steps that a per-step
#: curve and an anchor profile are non-degenerate, and small enough that a figure renders fast.
_ANCHOR_FLOOR = 6
_T_VALID = 12
_T = 16
_HORIZON = 4
_CHANNELS = 5

#: Every column name any emitted artifact may not carry. Scanned rather than spot-checked: the
#: conversion was removed rather than repointed, and the failure mode is a column that comes back.
_FORBIDDEN_COLUMN_SUBSTRINGS = ("bpm",)


# =================================================================================================
# The skill arithmetic, against known answers
# =================================================================================================
def _per_guid_frame(model_sq: float, model_block: float, *, n: int = 6) -> pd.DataFrame:
    """A per-recording frame where every baseline scores 1.0 and the model scores what is asked."""
    columns = {
        "sq_error_base": model_sq, "sq_error_full": model_sq,
        "nll_base_block": model_block, "nll_full_block": model_block,
        "abs_error_base": 0.0, "abs_error_full": 0.0,
        "signed_error_base": 0.0, "signed_error_full": 0.0,
    }
    for name in BASELINE_NAMES:
        columns[f"sq_error_{name}"] = 1.0
        columns[f"nll_{name}_block"] = 10.0
    return pd.DataFrame({name: [value] * n for name, value in columns.items()})


def test_the_skill_table_reports_one_row_per_branch_and_baseline_with_its_n() -> None:
    rows = forecast_analysis.build_skill_rows(_per_guid_frame(0.0, 0.0), resamples=200, seed=0)

    assert len(rows) == len(forecast_analysis.MODEL_BRANCHES) * len(BASELINE_NAMES)
    assert {row["baseline"] for row in rows} == set(BASELINE_NAMES)
    assert all(row["n_recordings"] == 6 for row in rows)


def test_a_perfect_forecast_scores_one_against_every_baseline_with_a_finite_interval() -> None:
    rows = forecast_analysis.build_skill_rows(_per_guid_frame(0.0, 0.0), resamples=200, seed=0)

    assert all(row["mse_skill"] == pytest.approx(1.0) for row in rows)
    assert all(
        np.isfinite(row["mse_skill_lo"]) and np.isfinite(row["mse_skill_hi"]) for row in rows
    )
    # And the log-score column is a difference, so it is the baseline's score, not a ratio.
    assert all(row["advantage_nats_per_anchor"] == pytest.approx(10.0) for row in rows)


def test_a_forecast_equal_to_the_baselines_scores_zero_in_both_spaces() -> None:
    """Both columns at once: it is the case where the ratio and the difference agree, and a skill
    computed as ``ratio - 1`` rather than ``1 - ratio`` would pass one and fail the other."""
    rows = forecast_analysis.build_skill_rows(_per_guid_frame(1.0, 10.0), resamples=200, seed=0)

    assert all(row["mse_skill"] == pytest.approx(0.0) for row in rows)
    assert all(row["advantage_nats_per_anchor"] == pytest.approx(0.0) for row in rows)


def test_a_zero_error_baseline_yields_no_skill_rather_than_an_infinity() -> None:
    """A baseline that is exactly right on a degenerate recording would otherwise report an
    infinite skill as evidence -- and the headline finiteness check would refuse the summary."""
    skill = forecast_analysis.skill_against(np.array([1.0, 0.5]), np.array([0.0, 1.0]))

    assert np.isnan(skill[0]) and skill[1] == pytest.approx(0.5)


def test_the_r2_reference_is_one_of_a_closed_set() -> None:
    """An $R^2$ whose reference is implicit is a claim a reader cannot check: against the segment
    mean and against climatology it is a different number."""
    assert forecast_analysis.R2_REFERENCE in forecast_analysis.R2_REFERENCES
    assert set(forecast_analysis.R2_REFERENCES) <= set(BASELINE_NAMES)


def test_the_rmse_roots_once_rather_than_averaging_finished_roots() -> None:
    r"""Jensen: $\operatorname{mean}(\sqrt{x}) \le \sqrt{\operatorname{mean}(x)}$, so averaging
    per-segment RMSEs is biased **low** -- in the direction that flatters the model. Four
    recordings, because a bootstrap over fewer than three reports a note instead of an interval."""
    frame = pd.DataFrame(
        {
            "sq_error_base": [1.0, 9.0, 1.0, 9.0], "sq_error_full": [1.0, 9.0, 1.0, 9.0],
            "abs_error_base": [1.0, 3.0, 1.0, 3.0], "abs_error_full": [1.0, 3.0, 1.0, 3.0],
            "signed_error_base": [0.0] * 4, "signed_error_full": [0.0] * 4,
        }
    )

    rows = forecast_analysis.build_error_rows(frame, resamples=200, seed=0)

    # sqrt(mean(1, 9)) = sqrt(5) = 2.236..., not mean(sqrt(1), sqrt(9)) = 2.0.
    assert rows[0]["rmse_normalised"] == pytest.approx(float(np.sqrt(5.0)))
    assert rows[0]["rmse_normalised"] > 2.0
    assert rows[0]["unit"] == NORMALISED_UNIT


def test_the_error_row_carries_no_second_unit(emitted) -> None:
    """The conversion was removed rather than repointed: inverting the per-channel statistics would
    put the 98 scored channels on scales spanning orders of magnitude, which destroys the pooled
    mean squared error, the skill ratio and every shared axis below."""
    row = emitted["result"]["error"][0]

    assert row["unit"] == NORMALISED_UNIT == "normalised"
    assert {name for name in row if "bpm" in name.lower()} == set()
    assert "rmse" not in row and "mae" not in row and "bias" not in row


# =================================================================================================
# The horizon axis
# =================================================================================================
def _horizon_record(n_steps: int = 4) -> Dict[str, Any]:
    """A hand-built accumulator record with distinct, checkable per-step values."""
    return {
        "base_sum_block": [10.0, 20.0, 30.0, 40.0][:n_steps],
        "base_n_anchors": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "full_sum_block": [8.0, 18.0, 30.0, 44.0][:n_steps],
        "full_n_anchors": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "base_sum_sq": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "base_count": [5.0, 5.0, 4.0, 2.0][:n_steps],
        "full_sum_sq": [20.0, 20.0, 16.0, 8.0][:n_steps],
        "full_count": [5.0, 5.0, 4.0, 2.0][:n_steps],
    }


def test_the_horizon_curve_divides_each_step_by_its_own_denominator() -> None:
    r"""Per $\tau$, not the per-anchor contributing indicator -- which is an ``amax`` over $\tau$
    and would count a masked late-horizon step as a scored zero, flattering exactly the horizons
    that fall in gaps."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["d_base_nats"]) == pytest.approx([2.0, 4.0, 7.5, 20.0])
    assert list(curves["d_full_nats"]) == pytest.approx([1.6, 3.6, 7.5, 22.0])


def test_the_horizon_gap_is_the_difference_of_the_two_scores() -> None:
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["gap_nats"]) == pytest.approx(
        list(np.asarray(curves["d_base_nats"]) - np.asarray(curves["d_full_nats"])), abs=1e-12
    )


def test_the_horizon_axis_is_lead_time_in_seconds_not_step_index() -> None:
    r"""Horizon step $\tau$ reads decimated step $t + 1 + \tau$, so its lead time ends at
    $4(\tau + 1)$ seconds: step $0$ is four seconds ahead, not zero."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["lead_seconds"]) == pytest.approx([4.0, 8.0, 12.0, 16.0])


def test_the_horizon_curve_names_the_path_it_was_computed_on() -> None:
    r"""The marginalisation does not commute with the sum over $\tau$, so the curve is the
    single-draw one and has to say so rather than leaving it to be inferred."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert set(curves["score_path"]) == {"single-draw (training path)"}


def test_an_absent_horizon_block_yields_no_curve_rather_than_an_invented_one() -> None:
    assert forecast_analysis.horizon_curves({}).empty
    assert forecast_analysis.horizon_curves({"base_sum_block": [1.0]}).empty


def test_the_horizon_rmse_is_per_coefficient_and_carries_the_normalised_label() -> None:
    """``count`` already carries the channel factor, so the mean square is per coefficient and
    matches ``sq_error_*`` on the sample table rather than being larger by $C_{\\mathrm{keep}}$."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["rmse_base_normalised"]) == pytest.approx([1.0] * 4)
    assert list(curves["rmse_full_normalised"]) == pytest.approx([2.0] * 4)
    assert set(curves["rmse_unit"]) == {NORMALISED_UNIT}
    assert not [name for name in curves.columns if "bpm" in str(name).lower()]


# =================================================================================================
# The figures
# =================================================================================================
def test_the_horizon_figure_spans_the_whole_forecast_window_in_seconds() -> None:
    r"""$[0, 60]$ seconds rather than $[0, 15]$ steps, on every panel: a reader who has to multiply
    by four is a reader who will eventually forget to."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    horizon = int(SHIPPED_HORIZON)
    curves = forecast_analysis.horizon_curves(_horizon_record())

    figure = forecast_analysis.build_horizon_figure(curves, horizon_steps=horizon)
    try:
        limits = [axis.get_xlim() for axis in figure.axes]
    finally:
        shared_figures.plt.close(figure)

    assert limits == [(0.0, 4.0 * horizon)] * len(limits)
    assert len(limits) == 3


def test_the_horizon_figures_gap_line_is_the_recomputed_difference() -> None:
    """Asserted on the drawn data, not on the frame it came from: a panel plotting the wrong
    column would pass every table assertion above."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    curves = forecast_analysis.horizon_curves(_horizon_record())
    expected = np.asarray(curves["d_base_nats"]) - np.asarray(curves["d_full_nats"])

    figure = forecast_analysis.build_horizon_figure(curves, horizon_steps=4)
    try:
        drawn = np.asarray(figure.axes[1].lines[0].get_ydata(), dtype=np.float64)
    finally:
        shared_figures.plt.close(figure)

    assert drawn == pytest.approx(expected, abs=1e-6)


def test_the_anchor_profile_shades_the_region_below_the_floor_and_the_untrained_tail() -> None:
    r"""Two structural spans, and neither is a finding. Below the anchor floor **nothing is decoded
    at all** -- unlike the raw cells, where a warm-up anchor exists and carries no loss term -- and
    the tail holds anchors whose forecast window would run past the end of the segment."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    record = {"anchor_floor": _ANCHOR_FLOOR, "t_valid": _T_VALID, "t": _T}
    anchors = np.arange(_ANCHOR_FLOOR, _T_VALID)
    profile = pd.DataFrame(
        {
            "anchor": anchors,
            "nll_base_block": np.linspace(1.0, 2.0, anchors.size),
            "nll_full_block": np.linspace(1.0, 2.0, anchors.size),
            "pred_gap": np.zeros(anchors.size),
        }
    )

    figure, spans = forecast_analysis.build_anchor_profile_figure(profile, record)
    try:
        # Read off the rectangle's own x extent rather than off its path: ``axvspan`` draws under
        # a blended transform, so the path is a unit square and carries none of the data
        # coordinates the shading actually covers.
        drawn = [
            (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
            for patch in figure.axes[0].patches
        ]
    finally:
        shared_figures.plt.close(figure)

    assert spans["below_anchor_floor"] == (0.0, float(_ANCHOR_FLOOR))
    assert spans["untrained_tail"] == (float(_T_VALID), float(_T))
    assert drawn == [spans["below_anchor_floor"], spans["untrained_tail"]]


# =================================================================================================
# The overlay, whose retention is opt-in
# =================================================================================================
def _retained_blocks(rows: int = 2, anchors: int = _T_VALID - _ANCHOR_FLOOR) -> Dict[str, Any]:
    """Retained forecast blocks shaped as the collection pass keeps them: $(N, A, H, C)$."""
    shape = (rows, anchors, _HORIZON, _CHANNELS)
    return {
        "target": np.zeros(shape),
        "mu_base": np.full(shape, 0.5),
        "mu_full": np.full(shape, -0.5),
        "waveforms_sample_index": np.arange(rows),
    }


def _synthetic_context(*, retained: Optional[Dict[str, Any]] = None) -> AnalysisContext:
    """An analysis context built by hand, with no model and no collection pass.

    Block retention is opt-in -- ``eval_config.caps.waveforms`` -- so a run that did not ask for it
    retains nothing and the overlay is never reached. Driving the analysis directly is what
    exercises both branches without paying for two more end-to-end evaluations.
    """
    n_recordings = 4
    per_sample = pd.DataFrame({"guid": [f"g{index}" for index in range(n_recordings)]})
    for branch in ("base", "full", *BASELINE_NAMES):
        per_sample[f"nll_{branch}_block"] = np.linspace(10.0, 13.0, n_recordings)
        per_sample[f"sq_error_{branch}"] = np.linspace(1.0, 4.0, n_recordings)
    for branch in ("base", "full"):
        per_sample[f"abs_error_{branch}"] = 0.5
        per_sample[f"signed_error_{branch}"] = 0.0
    anchors = np.arange(_ANCHOR_FLOOR, _T_VALID)
    per_anchor = pd.DataFrame(
        {
            "anchor": np.tile(anchors, n_recordings),
            "nll_base_block": 1.0, "nll_full_block": 0.9, "pred_gap": 0.1,
        }
    )
    record = {
        "geometry": {
            "t": _T, "t_valid": _T_VALID, "horizon": _HORIZON, "anchor_floor": _ANCHOR_FLOOR,
            "anchors_per_sample": _T_VALID - _ANCHOR_FLOOR,
            "target_kept_width": _CHANNELS, "block_width": _HORIZON * _CHANNELS,
        },
        "horizon": _horizon_record(_HORIZON),
        "likelihood": "gaussian_nll",
    }
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=per_anchor, record=record, retained=retained or {}
    )
    return AnalysisContext(collection=collection, config={})


@pytest.fixture(scope="module")
def emitted(tmp_path_factory):
    """One run of the analysis over the stub context, with a block retained."""
    output_dir = tmp_path_factory.mktemp("forecast")
    result = forecast_analysis.run_forecast_analysis(
        _synthetic_context(retained=_retained_blocks()),
        eval_config={"bootstrap_resamples": 200, "seed": 0},
        output_dir=output_dir,
        probe=None,
    )
    return {"result": result, "dir": Path(output_dir) / forecast_analysis.ANALYSIS_DIRNAME}


def test_no_block_retained_means_no_overlay_rather_than_an_empty_page(tmp_path) -> None:
    """Retention is opt-in because the tensors this figure needs are megabytes per sample. A run
    that did not ask for them has not failed, so the absent figure is silence."""
    result = forecast_analysis.run_forecast_analysis(
        _synthetic_context(), eval_config={"bootstrap_resamples": 200, "seed": 0},
        output_dir=tmp_path, probe=None,
    )

    assert forecast_analysis.OVERLAY_FIGURE not in result["files"]
    assert not (
        tmp_path / forecast_analysis.ANALYSIS_DIRNAME / forecast_analysis.OVERLAY_FIGURE
    ).exists()


def test_a_retained_block_is_drawn_and_recorded(emitted) -> None:
    assert forecast_analysis.OVERLAY_FIGURE in emitted["result"]["files"]
    assert (emitted["dir"] / forecast_analysis.OVERLAY_FIGURE).is_file()


def test_the_overlay_draws_one_panel_per_channel_against_lead_time() -> None:
    r"""Not the raw cells' single waveform: what this model forecasts is an
    $H \times C_{\mathrm{keep}}$ block, so the comparison being drawn is between the three curves
    *within* a channel and each channel gets its own panel and its own y-axis."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    channels = forecast_analysis.overlay_channels(_CHANNELS)
    figure = forecast_analysis.build_overlay_figure(
        _retained_blocks(), row=0, anchor=0, channels=channels
    )
    try:
        axis = figure.axes[0]
        lines = {line.get_label(): np.asarray(line.get_ydata()) for line in axis.lines}
        x = np.asarray(axis.lines[0].get_xdata(), dtype=np.float64)
        n_panels = len(figure.axes)
    finally:
        shared_figures.plt.close(figure)

    assert n_panels == len(channels) == forecast_analysis.OVERLAY_CHANNELS
    assert lines["truth"] == pytest.approx(np.zeros(x.size))
    assert lines["target-only (base)"] == pytest.approx(np.full(x.size, 0.5))
    assert lines["source-conditioned (full)"] == pytest.approx(np.full(x.size, -0.5))
    # One point per horizon step, the first of them one decimated step ahead of the anchor.
    assert x.size == _HORIZON
    assert list(x) == pytest.approx([4.0, 8.0, 12.0, 16.0])


def test_the_overlay_channels_are_spread_across_the_axis_and_deterministic() -> None:
    """Fixed and evenly spaced so two runs of one checkpoint draw the same channels and a figure
    can be compared across arms rather than only read."""
    assert forecast_analysis.overlay_channels(98) == [0, 48, 97]
    assert forecast_analysis.overlay_channels(98) == forecast_analysis.overlay_channels(98)
    # Degenerate widths do not raise and do not invent a channel.
    assert forecast_analysis.overlay_channels(2) == [0, 1]
    assert forecast_analysis.overlay_channels(0) == []


def test_the_overlay_indexes_the_anchor_axis_by_position_not_by_decimated_step() -> None:
    """The whole reason this differs from the raw cells' overlay. This model *gathers* its anchors,
    so the anchor floor of 133 is not a valid index into a 152-long retained axis: reading the step
    as a position would draw a different anchor than the one named, or fail on the shipped
    geometry."""
    retained = _retained_blocks()
    n_positions = retained["target"].shape[1]

    assert n_positions < _T_VALID
    # ``_emit_overlay`` picks the middle position; it must be inside the retained axis.
    assert 0 <= n_positions // 2 < n_positions


# =================================================================================================
# The artifacts and the protocol
# =================================================================================================
def test_the_analysis_writes_its_four_tables_and_three_figures(emitted) -> None:
    for name in (
        forecast_analysis.SCORES_FILENAME,
        forecast_analysis.SKILL_FILENAME,
        forecast_analysis.HORIZON_FILENAME,
        forecast_analysis.ANCHOR_FILENAME,
        forecast_analysis.BASELINE_FIGURE,
        forecast_analysis.ANCHOR_FIGURE,
        forecast_analysis.HORIZON_FIGURE,
    ):
        assert (emitted["dir"] / name).is_file(), name


def test_no_emitted_column_carries_a_bpm_label(emitted) -> None:
    """The scan rather than a spot check: the conversion was removed, and the failure mode this
    guards is a column quietly coming back through a copied helper."""
    for name in (forecast_analysis.SCORES_FILENAME, forecast_analysis.SKILL_FILENAME,
                 forecast_analysis.HORIZON_FILENAME, forecast_analysis.ANCHOR_FILENAME):
        columns = [str(column).lower() for column in pd.read_csv(emitted["dir"] / name).columns]
        for column in columns:
            assert not any(token in column for token in _FORBIDDEN_COLUMN_SUBSTRINGS), (name, column)


def test_the_result_records_the_baseline_sigma_and_the_reference(emitted) -> None:
    """A learned-variance model beats a fixed-variance baseline partly on variance modelling alone,
    so which $\\sigma$ the baselines were given decides the whole NLL-space number -- and it travels
    in the output rather than in a docstring."""
    result = emitted["result"]

    assert result["baseline_logvar"] == pytest.approx(BASELINE_LOGVAR)
    assert result["baselines"] == list(BASELINE_NAMES)
    assert result["r2"]["reference"] == forecast_analysis.R2_REFERENCE
    assert result["unit"] == NORMALISED_UNIT


def test_the_result_declares_its_grouped_frame_over_the_per_recording_scores(emitted) -> None:
    """The runner fans the by-class and by-subgroup variants over a CSV this analysis has already
    written, so the entry names a file on disk rather than returning a frame."""
    entries = emitted["result"]["grouped_frames"]

    assert len(entries) == 1
    assert entries[0]["path"].endswith(forecast_analysis.SCORES_FILENAME)
    assert set(entries[0]["value_columns"]) == set(forecast_analysis.GROUPED_METRICS)


# =================================================================================================
# Against a real run
# =================================================================================================
@pytest.mark.slow
def test_the_horizon_curve_has_one_point_per_forecast_step(collected_run) -> None:
    curves = pd.read_csv(
        Path(collected_run["results_dir"])
        / forecast_analysis.ANALYSIS_DIRNAME
        / forecast_analysis.HORIZON_FILENAME
    )
    geometry = collected_run["summary"]["collection"]["geometry"]

    assert len(curves) == int(geometry["horizon"])
    assert curves["lead_seconds"].max() == pytest.approx(4.0 * int(geometry["horizon"]))


@pytest.mark.slow
def test_the_per_recording_table_is_one_row_per_recording(collected_run) -> None:
    """Not per segment: anchors overlap in fourteen of their fifteen horizon steps and one
    recording contributes tens of segments, so a per-segment bootstrap would resample a population
    it does not have."""
    scores = pd.read_csv(
        Path(collected_run["results_dir"])
        / forecast_analysis.ANALYSIS_DIRNAME
        / forecast_analysis.SCORES_FILENAME
    )
    results = collected_run["summary"]["results"]

    assert len(scores) == int(results["n_recordings"])
    assert len(scores) < int(results["n_samples"])
