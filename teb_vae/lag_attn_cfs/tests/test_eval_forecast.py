r"""Whether the forecast is any good: the skill arithmetic, the horizon axis, and the units.

The three trivial predictors themselves are built and scored in ``metrics`` and are tested in
``test_eval_metrics.py``, against known answers on constructed inputs. What is tested here is the
*analysis* over them, and it is four kinds of assertion.

**Known answers in the skill arithmetic.** A forecast equal to the truth scores skill $1$; one
equal to the baseline scores exactly $0$. These pin the arithmetic without a model and without a
fixture, and they are what would catch a skill score computed the wrong way round.

**Rooting once.** The RMSE roots after the per-recording mean of the unrooted squares. Averaging
finished roots is biased low by Jensen -- in the direction that flatters the model -- so the frame
below has widely different per-recording squared errors, which is where the two differ.

**The horizon denominators.** Each $\tau$ divides by its own masked count, and the axis is lead
time in seconds rather than step index.

**The figures and the overlay.** The drawn curves are the recomputed ones, the structural spans
are shaded where the geometry puts them, and block retention is opt-in.

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

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import forecast as forecast_analysis
from teb_vae.lag_attn_cfs.eval.metrics import BASELINE_NAMES, NORMALISED_UNIT

from .conftest import SHIPPED_HORIZON

#: A tiny stand-in geometry for the stub context: enough anchors and horizon steps that a per-step
#: curve and an anchor profile are non-degenerate, and small enough that a figure renders fast.
_ANCHOR_FLOOR = 6
_T_VALID = 12
_T = 16
_HORIZON = 4
_CHANNELS = 5

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
    assert list(curves["gap_nats"]) == pytest.approx([0.4, 0.4, 0.0, -2.0])


def test_the_horizon_axis_is_lead_time_in_seconds_not_step_index() -> None:
    r"""Horizon step $\tau$ reads decimated step $t + 1 + \tau$, so its lead time ends at
    $4(\tau + 1)$ seconds: step $0$ is four seconds ahead, not zero."""
    curves = forecast_analysis.horizon_curves(_horizon_record())

    assert list(curves["lead_seconds"]) == pytest.approx([4.0, 8.0, 12.0, 16.0])


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
    assert len(limits) == 4


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

    assert figure_filename(forecast_analysis.OVERLAY_FIGURE) not in result["files"]
    assert not (
        tmp_path / forecast_analysis.ANALYSIS_DIRNAME / figure_filename(forecast_analysis.OVERLAY_FIGURE)
    ).exists()


def test_a_retained_block_is_drawn_and_recorded(emitted) -> None:
    assert figure_filename(forecast_analysis.OVERLAY_FIGURE) in emitted["result"]["files"]
    assert (emitted["dir"] / figure_filename(forecast_analysis.OVERLAY_FIGURE)).is_file()


def test_the_overlay_draws_the_raw_rows_then_one_row_per_channel_against_time() -> None:
    r"""Not the raw cells' single waveform: what this model forecasts is an
    $H \times C_{\mathrm{keep}}$ block, so the comparison being drawn is between the curves
    *within* a channel, each channel on its own row under the raw FHR and UP rows."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    channels = forecast_analysis.overlay_channels(_CHANNELS, 3)
    retained = _retained_blocks()
    geometry = {"horizon": _HORIZON, "anchor_first": _ANCHOR_FLOOR, "anchor_stride": 1}
    samples = forecast_analysis.overlay_samples(
        retained, pd.DataFrame({"guid": ["g0", "g1"]}), None, geometry=geometry, count=2
    )
    figure = forecast_analysis.build_overlay_figure(
        retained, samples, channels, geometry=geometry, predictive_band=False
    )
    try:
        n_axes = len(figure.axes)
        # Column 0 of the first channel row, below the two raw rows.
        axis = figure.axes[2 * len(samples)]
        lines = [np.asarray(line.get_ydata(), dtype=np.float64) for line in axis.lines]
        x = np.asarray(axis.lines[1].get_xdata(), dtype=np.float64)
    finally:
        shared_figures.plt.close(figure)

    assert n_axes == (2 + len(channels)) * len(samples)
    # History (dashed), truth, base mean, full mean, in that order.
    assert lines[1] == pytest.approx(np.zeros(_HORIZON))
    assert lines[2] == pytest.approx(np.full(_HORIZON, 0.5))
    assert lines[3] == pytest.approx(np.full(_HORIZON, -0.5))
    # One point per horizon step, the first of them one decimated step ahead of the anchor.
    assert list(x) == pytest.approx([4.0, 8.0, 12.0, 16.0])


def test_the_overlay_channels_are_deterministic_and_one_per_band_when_the_map_is_known() -> None:
    """Fixed so two runs of one checkpoint draw the same channels and a figure can be compared
    across arms rather than only read; with the channel map, one scattering channel per band and
    the phase channel scored over the fewest horizon steps."""
    assert forecast_analysis.overlay_channels(98, 3) == [0, 48, 97]
    assert forecast_analysis.overlay_channels(98) == forecast_analysis.overlay_channels(98)
    # Degenerate widths do not raise and do not invent a channel.
    assert forecast_analysis.overlay_channels(2) == [0, 1]
    assert forecast_analysis.overlay_channels(0) == []

    kept = pd.DataFrame(
        {
            "kept_channel": range(8),
            "block": ["scattering"] * 5 + ["phase"] * 3,
            "band": ["slow_baseline", "beat_to_beat", "variability", "deceleration", "deceleration",
                     "beat_to_beat", "variability", "deceleration"],
        }
    )
    scored = np.asarray([30, 30, 30, 30, 30, 5, 30, 30])
    assert forecast_analysis.overlay_channels(8, kept_map=kept, scored_horizon=scored) == [0, 1, 2, 4, 5]


def test_the_history_before_an_anchor_is_an_earlier_anchors_future() -> None:
    """Horizon step tau of the anchor at step t is step t + 1 + tau, so the stored target at every
    step the decoded anchors look ahead to is recovered from the retained block alone."""
    first, anchors, horizon = 6, 4, 3
    steps = first + np.arange(anchors)[:, None] + 1 + np.arange(horizon)[None, :]
    series = forecast_analysis.step_series(steps.astype(float), first=first, stride=1)

    assert list(series[first + 1:]) == pytest.approx(list(range(first + 1, first + anchors + horizon)))
    assert np.isnan(series[: first + 1]).all()


def test_the_band_widens_to_the_marginal_variance_under_the_ar_residual() -> None:
    r"""$v_\tau = \sigma^2_\tau + \phi^2 v_{\tau-1}$: the density scores the innovation, so the
    residual accumulates earlier innovations and the band drawn around the mean must too."""
    variance = forecast_analysis.marginal_variance(np.ones(3), 0.5)

    assert list(variance) == pytest.approx([1.0, 1.25, 1.3125])
    assert list(forecast_analysis.marginal_variance(np.ones(3), None)) == [1.0, 1.0, 1.0]


# =================================================================================================
# The artifacts and the protocol
# =================================================================================================
def test_the_analysis_writes_its_four_tables_and_three_figures(emitted) -> None:
    for name in (
        forecast_analysis.SCORES_FILENAME,
        forecast_analysis.SKILL_FILENAME,
        forecast_analysis.HORIZON_FILENAME,
        forecast_analysis.ANCHOR_FILENAME,
        figure_filename(forecast_analysis.BASELINE_FIGURE),
        figure_filename(forecast_analysis.ANCHOR_FIGURE),
        figure_filename(forecast_analysis.HORIZON_FIGURE),
    ):
        assert (emitted["dir"] / name).is_file(), name


def test_the_result_declares_its_grouped_frame_over_the_per_recording_scores(emitted) -> None:
    """The runner fans the by-class and by-subgroup variants over a CSV this analysis has already
    written, so the entry names a file on disk rather than returning a frame."""
    entries = emitted["result"]["grouped_frames"]

    assert len(entries) == 1
    assert entries[0]["path"].endswith(forecast_analysis.SCORES_FILENAME)
    assert set(entries[0]["value_columns"]) == set(forecast_analysis.GROUPED_METRICS)
