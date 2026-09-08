r"""Confusion counts, the rates derived from them, the curves, and discrimination per time bin.

Hand-checkable throughout. A confusion cell is a pair of comparisons over six labels, a rate is a
division, a curve is what ``scikit-learn`` returns for those labels, and a per-bin table is a
``groupby`` over a frame written out by hand. Nothing here builds a model, opens a shard or reads a
run directory.

Two properties are worth naming because they fail quietly rather than loudly:

* **The two zero-denominator conventions are not interchangeable.** A rate whose denominator counts
  a class is undefined when that class is absent; a rate whose denominator counts a prediction is
  zero when the rule never fires. Collapsing them either hides an unmeasurable quantity behind a
  zero, or throws away draws the bootstrap could have used.
* **A per-bin cell is measured, or it is not.** A bin carrying one outcome group has counts and no
  discrimination, and the difference has to survive into the table -- a zero there would be drawn
  as a measured failure five hours before delivery, where nothing was measured at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze, data, evaluate


# The worked example every count below is checked against. At threshold 0 the rule predicts
# positive for 0.5, 1.0 and 2.0, so it is right about two of the three adverse recordings, wrong
# about one healthy recording, and right about the other two healthy ones.
LABELS = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
SCORES = np.asarray([-2.0, -1.0, 0.5, -0.5, 1.0, 2.0], dtype=np.float64)


# =============================================================================
# The four cells
# =============================================================================
def test_the_confusion_cells_are_the_worked_example():
    assert evaluate.confusion_counts(LABELS, SCORES, threshold=0.0) == {
        "tn": 2, "fp": 1, "fn": 1, "tp": 2,
    }


def test_the_predicate_is_at_or_above_the_threshold():
    """``logit >= threshold``, the same rule ``balanced_accuracy`` applies.

    A score sitting exactly on the threshold is a positive prediction. Were the two functions to
    disagree here, a balanced accuracy and the confusion matrix printed beside it would describe
    two different decision rules.
    """
    counts = evaluate.confusion_counts([0, 1], [0.0, 0.0], threshold=0.0)
    assert counts == {"tn": 0, "fp": 1, "fn": 0, "tp": 1}

    rates = evaluate.derived_rates(counts)
    assert rates["sensitivity"] == pytest.approx(1.0)
    assert evaluate.balanced_accuracy([0, 1], [0.0, 0.0], threshold=0.0) == pytest.approx(
        0.5 * (rates["sensitivity"] + rates["specificity"])
    )


def test_the_cells_always_number_four_even_on_one_class():
    counts = evaluate.confusion_counts([0, 0, 0], [1.0, -1.0, -1.0], threshold=0.0)
    assert set(counts) == set(evaluate.CONFUSION_CELLS)
    assert counts == {"tn": 2, "fp": 1, "fn": 0, "tp": 0}


# =============================================================================
# The two zero-denominator conventions
# =============================================================================
def test_the_rates_are_the_worked_example():
    rates = evaluate.derived_rates({"tn": 2, "fp": 1, "fn": 1, "tp": 2})
    assert rates["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert rates["specificity"] == pytest.approx(2.0 / 3.0)
    assert rates["precision"] == pytest.approx(2.0 / 3.0)
    assert rates["npv"] == pytest.approx(2.0 / 3.0)
    assert rates["f1"] == pytest.approx(4.0 / 6.0)
    assert rates["accuracy"] == pytest.approx(4.0 / 6.0)
    assert set(rates) == set(evaluate.RATE_NAMES)


def test_an_absent_class_leaves_its_rate_undefined_rather_than_zero():
    """No positives means sensitivity is unmeasurable, which is not the same as bad."""
    rates = evaluate.derived_rates({"tn": 3, "fp": 1, "fn": 0, "tp": 0})
    assert np.isnan(rates["sensitivity"])
    assert rates["specificity"] == pytest.approx(0.75)

    rates = evaluate.derived_rates({"tn": 0, "fp": 0, "fn": 1, "tp": 3})
    assert np.isnan(rates["specificity"])
    assert rates["sensitivity"] == pytest.approx(0.75)

    assert np.isnan(evaluate.derived_rates({"tn": 0, "fp": 0, "fn": 0, "tp": 0})["accuracy"])


def test_a_rule_that_never_fires_scores_zero_rather_than_undefined():
    """Precision, NPV and $F_1$ follow ``scikit-learn``'s ``zero_division=0``."""
    rates = evaluate.derived_rates({"tn": 3, "fp": 0, "fn": 2, "tp": 0})
    assert rates["precision"] == 0.0
    assert rates["f1"] == 0.0
    assert rates["npv"] == pytest.approx(0.6)

    never_negative = evaluate.derived_rates({"tn": 0, "fp": 3, "fn": 0, "tp": 2})
    assert never_negative["npv"] == 0.0


def test_the_added_metrics_cost_the_bootstrap_no_draw():
    """The draws the new metrics make non-finite are exactly the ones AUROC already did.

    ``paired_bootstrap`` discards a draw in which **any** reported metric is non-finite, so a
    metric that went ``nan`` on a draw AUROC could still have used would thin every interval in the
    table without saying so. Sensitivity is undefined precisely when there are no positives and
    specificity precisely when there are no negatives -- which together are exactly the draws on
    which AUROC is undefined.
    """
    for labels in ([0, 0, 0], [1, 1, 1], [0, 1, 1]):
        measured = evaluate.recording_metrics(labels, [0.1, 0.2, 0.3], threshold=0.15)
        undefined = {
            name for name in evaluate.METRIC_NAMES
            if not np.isfinite(measured[name])
        }
        assert bool(undefined) == (not np.isfinite(measured["auroc"])), labels


def test_recording_metrics_carries_the_cells_and_the_rates():
    measured = evaluate.recording_metrics(LABELS, SCORES, threshold=0.0)
    for name in evaluate.CONFUSION_CELLS + evaluate.RATE_NAMES:
        assert name in measured
    assert measured["auroc"] == pytest.approx(8.0 / 9.0)
    assert measured["tp"] == 2
    assert measured["n_adverse"] == 3


# =============================================================================
# The curves
# =============================================================================
def test_the_roc_points_start_and_end_at_the_corners():
    curve = evaluate.roc_points(LABELS, SCORES)
    assert curve["fpr"][0] == 0.0 and curve["tpr"][0] == 0.0
    assert curve["fpr"][-1] == 1.0 and curve["tpr"][-1] == 1.0
    assert curve["auroc"] == pytest.approx(8.0 / 9.0)
    assert len(curve["fpr"]) == len(curve["tpr"]) == len(curve["thresholds"])
    assert "note" not in curve


def test_the_pr_curve_reports_the_prevalence_that_is_its_chance_level():
    curve = evaluate.pr_points(LABELS, SCORES)
    assert curve["prevalence"] == pytest.approx(0.5)
    assert curve["chance_average_precision"] == pytest.approx(0.5)
    assert len(curve["recall"]) == len(curve["precision"])
    # ``precision_recall_curve`` returns one fewer threshold than points, and the record carries
    # the library's own arrays rather than a padded copy.
    assert len(curve["thresholds"]) == len(curve["recall"]) - 1


def test_a_single_class_population_yields_no_curve_and_says_why():
    for curve in (
        evaluate.roc_points([0, 0, 0], SCORES[:3]),
        evaluate.pr_points([1, 1, 1], SCORES[:3]),
    ):
        assert curve["fpr" if "fpr" in curve else "recall"] == []
        assert curve["note"] == evaluate.SINGLE_CLASS_NOTE
        assert curve["n_recordings"] == 3


# =============================================================================
# One population, one model
# =============================================================================
def _guids(n):
    return [f"guid_{index:02d}" for index in range(n)]


def test_metric_intervals_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.asarray([0, 1] * 10)
    scores = labels + rng.normal(0.0, 0.6, size=labels.size)
    record = evaluate.metric_intervals(
        labels, scores, guids=_guids(labels.size), threshold=0.5, resamples=300, seed=1
    )

    assert record["estimable"] is True
    assert record["grouping"] == "guid"
    assert record["n_units"] == 20
    for name in evaluate.METRIC_NAMES:
        interval = record["metrics"][name]
        assert interval["point"] == pytest.approx(record["full"][name])
        assert interval["lo"] <= interval["point"] <= interval["hi"]


def test_metric_intervals_resamples_patients_where_a_mapping_exists():
    labels = np.asarray([0, 1] * 6)
    guids = _guids(labels.size)
    patients = {guid: f"patient_{index // 2}" for index, guid in enumerate(guids)}
    record = evaluate.metric_intervals(
        labels, labels + 0.1, guids=guids, threshold=0.5, patients=patients,
        resamples=50, seed=1,
    )
    assert record["grouping"] == "patient"
    assert record["n_units"] == 6


def test_metric_intervals_reports_rather_than_raises_on_a_population_it_cannot_resample():
    single = evaluate.metric_intervals(
        [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], guids=_guids(4), threshold=0.0, resamples=10
    )
    assert single["estimable"] is False
    assert single["note"] == evaluate.SINGLE_CLASS_NOTE
    assert single["n_recordings"] == 4
    assert np.isnan(single["metrics"]["auroc"]["point"])

    tiny = evaluate.metric_intervals(
        [0, 1], [0.0, 1.0], guids=_guids(2), threshold=0.5, resamples=10
    )
    assert tiny["estimable"] is False
    assert "below 3" in tiny["note"]
    # The point estimates still stand: it is the spread that could not be estimated.
    assert tiny["metrics"]["auroc"]["point"] == pytest.approx(1.0)


def test_metric_intervals_refuses_misaligned_inputs():
    with pytest.raises(evaluate.GateEvaluationError, match="must align"):
        evaluate.metric_intervals(
            [0, 1, 0], [0.0, 1.0], guids=_guids(3), threshold=0.5
        )


# =============================================================================
# Discrimination per time bin
# =============================================================================
def _scored(bins, *, models=("pretrained", "adapted"), n=8, single_class_bins=()):
    """A per-bin score table with a known separation and some deliberately empty bins."""
    rng = np.random.default_rng(11)
    rows = []
    for model in models:
        for index in bins:
            for recording in range(n):
                outcome = int(recording % 2)
                if index in single_class_bins and outcome == 1:
                    continue
                rows.append({
                    "model": model,
                    data.GUID_COLUMN: f"guid_{recording:02d}",
                    data.BIN_COLUMN: index,
                    data.BIN_LABEL_COLUMN: f"({index * 0.5:g}, {(index + 1) * 0.5:g}]",
                    data.OUTCOME_COLUMN: outcome,
                    analyze.SCORE_COLUMN: float(outcome * 2.0 + rng.normal(0.0, 0.3)),
                })
    return pd.DataFrame(rows)


def test_a_bin_table_has_one_row_per_model_and_bin_with_its_counts():
    scored = _scored(range(4))
    table = analyze.bin_classification(
        scored, thresholds={"pretrained": 1.0, "adapted": 1.0}, bin_hours=0.5,
        resamples=100, seed=2, supervised=[0, 1],
    )
    assert len(table) == 8
    assert set(table["model"]) == {"pretrained", "adapted"}
    assert sorted(table[data.BIN_COLUMN].unique().tolist()) == [0, 1, 2, 3]
    assert table["n_recordings"].tolist() == [8] * 8
    assert table["n_adverse"].tolist() == [4] * 8
    # The midpoint is the bin's own, not the median of whatever anchors landed in it.
    for _index, row in table.iterrows():
        assert row["hours_mid"] == pytest.approx((row[data.BIN_COLUMN] + 0.5) * 0.5)


def test_the_supervised_bins_are_marked_and_the_rest_are_not():
    table = analyze.bin_classification(
        _scored(range(4)), thresholds={"pretrained": 1.0}, bin_hours=0.5,
        resamples=50, seed=2, supervised=analyze.supervised_bins(
            bin_hours=0.5, supervised_hours=1.0
        ),
    )
    inside = set(table[table["supervised_window"]][data.BIN_COLUMN])
    assert inside == {0, 1}


def test_a_single_class_bin_reports_counts_and_no_discrimination():
    table = analyze.bin_classification(
        _scored(range(3), models=("pretrained",), single_class_bins=(2,)),
        thresholds={"pretrained": 1.0}, bin_hours=0.5, resamples=50, seed=2,
    )
    empty = table[table[data.BIN_COLUMN] == 2].iloc[0]
    assert empty["estimable"] is False or bool(empty["estimable"]) is False
    assert np.isnan(empty["auroc"])
    assert empty["n_recordings"] == 4
    assert empty["n_adverse"] == 0
    assert empty["note"] == evaluate.SINGLE_CLASS_NOTE
    # The cells are still counted: the rule fired, or did not, on the recordings that were there.
    assert empty["tn"] + empty["fp"] == 4


def test_a_model_without_a_threshold_is_left_out_rather_than_given_one():
    scored = _scored(range(2))
    table = analyze.bin_classification(
        scored, thresholds={"pretrained": 1.0}, bin_hours=0.5, resamples=50, seed=2
    )
    assert set(table["model"]) == {"pretrained"}


def test_the_bin_table_uses_the_threshold_it_is_given_and_never_chooses_one():
    """Two thresholds, one table each, and the counts move exactly as the rule says."""
    scored = _scored(range(1), models=("pretrained",))
    strict = analyze.bin_classification(
        scored, thresholds={"pretrained": 10.0}, bin_hours=0.5, resamples=20, seed=2
    ).iloc[0]
    permissive = analyze.bin_classification(
        scored, thresholds={"pretrained": -10.0}, bin_hours=0.5, resamples=20, seed=2
    ).iloc[0]

    assert strict["tp"] == 0 and strict["fp"] == 0
    assert permissive["tp"] == 4 and permissive["fp"] == 4
    # AUROC is threshold-free, so the two agree on it exactly.
    assert strict["auroc"] == pytest.approx(permissive["auroc"])


def test_the_roc_grid_holds_only_the_bins_that_carried_both_groups():
    scored = _scored(range(3), models=("pretrained",), single_class_bins=(2,))
    curves = analyze.bin_roc_points(
        scored, thresholds={"pretrained": 1.0}, bin_hours=0.5
    )
    assert sorted(curves[data.BIN_COLUMN].unique().tolist()) == [0, 1]
    for index in (0, 1):
        block = curves[curves[data.BIN_COLUMN] == index]
        assert block["fpr"].min() == 0.0 and block["fpr"].max() == 1.0
        assert block["n_adverse"].iloc[0] == 4


def test_an_empty_score_table_yields_empty_tables_rather_than_an_exception():
    empty = pd.DataFrame(columns=[
        "model", data.GUID_COLUMN, data.BIN_COLUMN, data.BIN_LABEL_COLUMN,
        data.OUTCOME_COLUMN, analyze.SCORE_COLUMN,
    ])
    assert analyze.bin_classification(
        empty, thresholds={"pretrained": 0.0}, bin_hours=0.5
    ).empty
    assert analyze.bin_roc_points(
        empty, thresholds={"pretrained": 0.0}, bin_hours=0.5
    ).empty
