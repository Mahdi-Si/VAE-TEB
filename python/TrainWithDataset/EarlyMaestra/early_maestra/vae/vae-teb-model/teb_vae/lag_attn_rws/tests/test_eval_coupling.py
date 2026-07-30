r"""The coupling readout, its denominator, and the interval on it.

Three things are pinned here, and each catches a different way a coupling number is reported
wrongly.

**Known answers on synthetic per-recording vectors.** A vector that is positive on every recording
must report a positive fraction of exactly $1.0$ with a finite interval; one that is positive on
three of four must report $0.75$. Nothing about the model is involved -- this is the arithmetic,
and it is the arithmetic that would be wrong if the fraction were computed over segments, or over
the wrong axis, or against the wrong sign.

**The denominator is visible.** ``np.nan > 0`` is ``False``, so a recording that scored no anchors
-- and therefore measured *nothing* -- silently counts as evidence against a positive gap. The
seeded-with-NaN case is what distinguishes a fraction that excluded and counted them from one that
quietly voted them down: the same data reports $2/2$ under the first and $2/4$ under the second.

**The two estimators are labelled by path.** ``pred_gap`` is one subtraction on two different
estimators of the same quantity -- the Monte Carlo marginalised score and the single-draw training
path -- and a table carrying both under one name is unreadable. Every row here names its column
and its path.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval.analyses import coupling as coupling_analysis
from teb_vae.lag_attn_rws.eval.frames import positive_fraction

#: Bootstrap settings for the tests: enough resamples for a stable interval, few enough to be
#: instant. The seed is what makes every interval below reproducible.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


def _per_guid(**columns: List[float]) -> pd.DataFrame:
    """A per-recording frame carrying the named columns and NaN for everything else it needs."""
    length = len(next(iter(columns.values())))
    frame = pd.DataFrame(columns)
    for name in coupling_analysis.VALUE_COLUMNS:
        if name not in frame.columns:
            frame[name] = np.full(length, np.nan)
    return frame


# =============================================================================
# The positive fraction, and its denominator
# =============================================================================
def test_an_all_positive_gap_reports_a_fraction_of_one_with_a_finite_interval() -> None:
    per_guid = _per_guid(
        mc_pred_gap=[0.5, 1.5, 2.5, 3.5],
        mc_nll_base_block=[10.5, 11.5, 12.5, 13.5],
        mc_nll_full_block=[10.0, 10.0, 10.0, 10.0],
    )

    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)
    headline = rows[0]

    assert headline["metric"] == "pred_gap_mc_nats"
    assert headline["positive_fraction"] == pytest.approx(1.0)
    assert headline["n_recordings_scored"] == 4
    assert np.isfinite(headline["ci_lo"]) and np.isfinite(headline["ci_hi"])
    assert headline["ci_lo"] <= headline["mean"] <= headline["ci_hi"]


def test_a_mixed_gap_reports_the_fraction_that_is_actually_positive() -> None:
    """Non-vacuity for the case above: an implementation returning 1.0 unconditionally passes it."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 1.0, 1.0, -1.0]), resamples=200, seed=0
    )

    assert rows[0]["positive_fraction"] == pytest.approx(0.75)
    assert rows[0]["n_positive"] == 3


def test_unscored_recordings_are_excluded_and_counted_rather_than_voted_down() -> None:
    r"""The one that matters. ``np.nan > 0`` is ``False``, so a recording that measured nothing
    would otherwise be counted as evidence *against* a positive gap -- and a run whose coverage
    collapsed would report a falling positive fraction rather than a falling $n$."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 2.0, np.nan, np.nan]), resamples=200, seed=0
    )

    assert rows[0]["positive_fraction"] == pytest.approx(1.0), (
        "the two NaN recordings must be excluded, not counted as non-positive"
    )
    assert rows[0]["n_recordings_scored"] == 2
    assert rows[0]["n_recordings_dropped_not_finite"] == 2


def test_the_denominator_helper_reports_both_counts_on_an_all_nan_input() -> None:
    record = positive_fraction([np.nan, np.nan])

    assert np.isnan(record["fraction"])
    assert record["n"] == 0 and record["n_dropped_not_finite"] == 2


# =============================================================================
# The two estimators
# =============================================================================
def test_both_pred_gap_estimators_are_reported_and_labelled_by_path() -> None:
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 2.0, 3.0], pred_gap=[1.1, 2.1, 3.1]), resamples=200, seed=0
    )

    assert [row["metric"] for row in rows] == [
        "pred_gap_mc_nats", "pred_gap_train_path_nats"
    ]
    assert [row["source_column"] for row in rows] == ["mc_pred_gap", "pred_gap"]
    assert "marginalised" in rows[0]["score_path"]
    assert "single-draw" in rows[1]["score_path"]
    # Each estimator's own values, not one standing for both.
    assert rows[0]["mean"] == pytest.approx(2.0)
    assert rows[1]["mean"] == pytest.approx(2.1)


def test_the_paired_test_runs_on_the_two_block_scores_the_gap_is_the_difference_of() -> None:
    """Paired rather than unpaired: each recording contributes both scores, and pairing removes
    the between-recording variance that dominates every readout here."""
    per_guid = _per_guid(
        mc_pred_gap=[0.5] * 6,
        mc_nll_base_block=[10.5] * 6,
        mc_nll_full_block=[10.0] * 6,
    )

    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)

    assert rows[0]["wilcoxon_n_pairs"] == 6
    assert rows[0]["wilcoxon_median_difference"] == pytest.approx(0.5)


def test_quantiles_travel_beside_the_mean() -> None:
    """A mean and an interval describe a symmetric distribution; these are routinely skewed."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[0.0, 1.0, 2.0, 3.0, 100.0]), resamples=200, seed=0
    )

    assert rows[0]["q50"] == pytest.approx(2.0)
    assert rows[0]["mean"] > rows[0]["q75"], "the skew is what the quantiles exist to show"


# =============================================================================
# The KL, reported as a description
# =============================================================================
def test_the_kl_rows_name_the_unfloored_value_and_the_control_beside_it() -> None:
    rows = coupling_analysis.build_kl_rows(
        _per_guid(
            source_conditioned_kl_raw=[3.0, 4.0, 5.0],
            source_conditioned_kl_shuffled_raw=[3.5, 4.5, 5.5],
        ),
        resamples=200, seed=0,
    )

    assert [row["metric"] for row in rows] == list(coupling_analysis.KL_COLUMNS)
    assert "unfloored" in rows[0]["score_path"]
    assert rows[0]["mean"] == pytest.approx(4.0)
    assert rows[1]["mean"] == pytest.approx(4.5)


def test_no_headline_kl_column_is_the_floored_one() -> None:
    """Free bits are applied per dimension per step before summing, so the floored value exceeds
    the raw one by construction and hides a collapsed source pathway."""
    assert all("train" not in column for column in coupling_analysis.KL_COLUMNS)


# =============================================================================
# The figure
# =============================================================================
def test_the_distribution_figure_marks_zero_and_the_interval_on_the_mean() -> None:
    """Zero is the finding: a distribution straddling it is a different result from one sitting
    above it, and a histogram without that line invites a reader to eyeball the sign."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    per_guid = _per_guid(mc_pred_gap=[1.0, 2.0, 3.0, 4.0], pred_gap=[1.0, 2.0, 3.0, 4.0])
    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)

    figure = coupling_analysis.build_distribution_figure(per_guid, rows)
    try:
        axis = figure.axes[0]
        # ``axvline`` draws under a blended transform, so its x data are in data coordinates.
        vertical = [
            float(line.get_xdata()[0]) for line in axis.lines
            if len(set(np.asarray(line.get_xdata(), dtype=np.float64))) == 1
        ]
        spans = [
            (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
            for patch in axis.patches
            if float(patch.get_width()) > 0.0 and patch.get_label().startswith("95%")
        ]
        n_panels = len(figure.axes)
    finally:
        shared_figures.plt.close(figure)

    assert 0.0 in vertical, "the no-improvement reference must be drawn"
    assert spans == [(pytest.approx(rows[0]["ci_lo"]), pytest.approx(rows[0]["ci_hi"]))]
    assert n_panels == 2, "the second panel is the two estimators side by side"


# =============================================================================
# End to end, on a real run
# =============================================================================
def _context(per_sample: pd.DataFrame, results: Optional[Dict[str, Any]] = None) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={},
        results=results or {},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_writes_its_tables_and_figure(tmp_path) -> None:
    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b", "c", "c"],
            "mc_pred_gap": [1.0, 1.2, 0.8, 0.9, -0.1, 0.2],
            "pred_gap": [1.1, 1.3, 0.7, 1.0, -0.2, 0.3],
            "mc_nll_base_block": [11.0] * 6,
            "mc_nll_full_block": [10.0] * 6,
            "nll_base_block": [11.0] * 6,
            "nll_full_block": [10.0] * 6,
            "source_conditioned_kl_raw": [2.0] * 6,
            "source_conditioned_kl_shuffled_raw": [2.5] * 6,
        }
    )

    result = coupling_analysis.run_coupling_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / coupling_analysis.ANALYSIS_DIRNAME
    for name in (
        coupling_analysis.PER_RECORDING_FILENAME,
        coupling_analysis.SUMMARY_FILENAME,
        coupling_analysis.DISTRIBUTION_FIGURE,
    ):
        assert (directory / name).is_file(), name
    # One row per recording, not per segment.
    assert len(pd.read_csv(directory / coupling_analysis.PER_RECORDING_FILENAME)) == 3
    assert result["composition"]["n_recordings"] == 3
    assert result["n_samples"] == 6


def test_the_real_run_reports_both_estimators_with_their_denominators(evaluated) -> None:
    block = evaluated["summary"]["results"]["coupling"]

    metrics = {row["metric"]: row for row in block["pred_gap"]}
    assert set(metrics) == {"pred_gap_mc_nats", "pred_gap_train_path_nats"}
    for row in metrics.values():
        assert row["n_recordings_scored"] == evaluated["summary"]["results"]["n_recordings"]
        assert 0.0 <= row["positive_fraction"] <= 1.0
        assert row["n_positive"] <= row["n_recordings_scored"]
