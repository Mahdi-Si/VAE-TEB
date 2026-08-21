r"""The second clinical clock: the axis, its eligibility record, and the reduction it shares.

Three things are pinned here, and each is a way this clock could be wrong while looking right.

**The sign is the opposite of the delivery clock's, and that is not an inconsistency.** ``epoch``
is stored as time *before* delivery and is negated to give an axis; ``second_stage_onset`` is
stored as $\mathrm{domain\_start} - t_{\mathrm{SSO}}$ and is already signed the way an axis wants
it. Negating it as well would run every trajectory backwards through the second stage with
nothing raising, so the arithmetic is asserted against a known answer rather than described.

**The bin index is signed.** The delivery clock clips at zero because a window after delivery
means nothing; here the positive side is half the content, and a clip would fold every window
after onset into the first one.

**A recording is dropped for one reason only.** Missing onset excludes it; an onset recorded *at*
delivery and an onset that moves across a recording's own segments are counted and kept, because
excluding a recording changes the population every number is computed over and a count does not.
The sibling classifier pipeline was burned by the first of those, which is why it is measured here
rather than assumed away.

**The Holm family stops at this clock.** The correction runs across the windows of the second-stage
axis and is not joint with the delivery clock's, so a fixture whose two clocks disagree about how
many windows there are is what says the two families were kept apart rather than one being corrected
against the other's count.

**The figures are drawn in the natural orientation.** Every other clinical figure in this pipeline
inverts its x axis so delivery sits at the right; inverting this one would put "after the onset" on
the left with nothing on the page saying so, which is why the orientation and the onset mark are
asserted rather than eyeballed.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval import cohort
from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.analyses import second_stage as analysis

#: One hour, in the seconds both clocks are stored in.
_HOUR = 3600.0


def _offsets(rows) -> pd.DataFrame:
    """A per-sample table of ``(guid, epoch, second_stage_onset)`` triples."""
    return pd.DataFrame(
        [{"guid": guid, "epoch": epoch, cohort.SECOND_STAGE_COLUMN: offset}
         for guid, epoch, offset in rows]
    )


# =============================================================================
# The axis
# =============================================================================
def test_the_offset_is_read_as_a_signed_axis_without_a_sign_flip() -> None:
    r"""The known answer. An hour before onset is $-1$ h and half an hour after it is $+0.5$ h --
    the mirror of the delivery clock, where a negative ``epoch`` becomes a positive coordinate."""
    frame = pd.DataFrame({cohort.SECOND_STAGE_COLUMN: [-_HOUR, 0.5 * _HOUR, 0.0]})

    binned = cohort.add_second_stage_bins(frame)

    assert list(binned[cohort.SECOND_STAGE_HOURS_COLUMN]) == [-1.0, 0.5, 0.0]
    # 0.5 h windows, signed: -1.0 h lands in bin -2, +0.5 h in bin 1, 0.0 h in bin 0.
    assert list(binned[cohort.SECOND_STAGE_BIN_COLUMN]) == [-2, 1, 0]
    assert list(binned[cohort.SECOND_STAGE_BIN_CENTER_COLUMN]) == [-0.75, 0.75, 0.25]


def test_the_two_clocks_disagree_about_the_sign_of_the_same_number() -> None:
    """Non-vacuity for the rule above, stated as the contrast it exists for: handed the same
    stored value, the two axes must point in opposite directions. An implementation that negated
    both would pass every assertion about one of them alone."""
    stored = -2.0 * _HOUR
    delivery = cohort.add_time_bins(pd.DataFrame({"epoch": [stored]}))
    second_stage = cohort.add_second_stage_bins(
        pd.DataFrame({cohort.SECOND_STAGE_COLUMN: [stored]})
    )

    assert float(delivery[cohort.HOURS_COLUMN].iloc[0]) == pytest.approx(2.0)
    assert float(second_stage[cohort.SECOND_STAGE_HOURS_COLUMN].iloc[0]) == pytest.approx(-2.0)


def test_a_window_after_onset_is_not_folded_into_the_first_one() -> None:
    """The delivery clock clips at zero; this one must not, or every window after second-stage
    onset -- which is the half of the axis this clock exists to show -- collapses into one."""
    frame = pd.DataFrame({cohort.SECOND_STAGE_COLUMN: [-3.0 * _HOUR, 2.0 * _HOUR]})

    bins = list(cohort.add_second_stage_bins(frame)[cohort.SECOND_STAGE_BIN_COLUMN])

    assert bins == [-6, 4]
    assert min(bins) < 0


def test_a_non_finite_offset_is_dropped_rather_than_binned() -> None:
    frame = pd.DataFrame({cohort.SECOND_STAGE_COLUMN: [-_HOUR, np.nan, float("inf")]})

    assert len(cohort.add_second_stage_bins(frame)) == 1


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({cohort.SECOND_STAGE_COLUMN: []}),
        pd.DataFrame({cohort.SECOND_STAGE_COLUMN: [np.nan, np.nan]}),
        pd.DataFrame({"epoch": [-3600.0]}),
    ],
)
def test_a_frame_with_no_usable_offset_still_carries_the_columns(frame) -> None:
    """A cohort with no onset table at all is the ordinary outcome, not a failure -- and the
    caller still groups on these columns, so they have to exist on the empty frame."""
    binned = cohort.add_second_stage_bins(frame)

    assert len(binned) == 0
    for column in (
        cohort.SECOND_STAGE_HOURS_COLUMN,
        cohort.SECOND_STAGE_BIN_COLUMN,
        cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
    ):
        assert column in binned.columns


# =============================================================================
# The reduction, shared with the delivery clock
# =============================================================================
def test_the_per_recording_reduction_defaults_to_the_delivery_clock() -> None:
    """The second clock passes its own columns; every existing caller passes none, and must get
    exactly the frame it got before this parameter existed."""
    frame = pd.DataFrame(
        {
            "guid": ["a", "a", "b"],
            "epoch": [-3600.0, -3600.0, -3600.0],
            labels.CLASS_COLUMN: ["healthy", "healthy", "healthy"],
            "value": [1.0, 3.0, 5.0],
        }
    )
    binned = cohort.add_time_bins(frame)

    default = cohort.per_recording_in_bins(binned, ["value"], group_column=labels.CLASS_COLUMN)
    explicit = cohort.per_recording_in_bins(
        binned, ["value"], group_column=labels.CLASS_COLUMN,
        bin_column=cohort.BIN_COLUMN, center_column=cohort.BIN_CENTER_COLUMN,
    )

    pd.testing.assert_frame_equal(default, explicit)
    assert list(default["value"]) == [2.0, 5.0]


def test_the_reduction_groups_on_the_second_stage_windows_when_asked() -> None:
    """One recording either side of onset is two rows, not one: the reduction has to see the
    second clock's window column, or every segment of a recording collapses into one value."""
    frame = pd.DataFrame(
        {
            "guid": ["a"] * 4,
            cohort.SECOND_STAGE_COLUMN: [-_HOUR, -_HOUR, _HOUR, _HOUR],
            labels.CLASS_COLUMN: ["healthy"] * 4,
            "value": [1.0, 3.0, 10.0, 12.0],
        }
    )
    binned = cohort.add_second_stage_bins(frame)

    reduced = cohort.per_recording_in_bins(
        binned, ["value"], group_column=labels.CLASS_COLUMN,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
    )

    assert len(reduced) == 2
    assert sorted(reduced["value"]) == [2.0, 11.0]
    assert sorted(reduced[cohort.SECOND_STAGE_BIN_COLUMN]) == [-2, 2]


def test_the_trajectory_rows_report_a_window_and_its_centre_on_either_clock() -> None:
    """The emitted keys stay ``time_bin`` and ``bin_center_h`` on both clocks: they name a window
    and its centre, which is what they are on either, and renaming them per clock would fork every
    consumer of this table."""
    frame = pd.DataFrame(
        {
            "guid": ["a", "b", "c"],
            cohort.SECOND_STAGE_COLUMN: [_HOUR] * 3,
            labels.CLASS_COLUMN: ["healthy"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    reduced = cohort.per_recording_in_bins(
        cohort.add_second_stage_bins(frame), ["value"], group_column=labels.CLASS_COLUMN,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
    )

    rows = cohort.trajectory_rows(
        reduced, "value", metric="value",
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
    )

    assert len(rows) == 1
    assert rows[0]["time_bin"] == 2
    assert rows[0]["bin_center_h"] == pytest.approx(1.25)
    assert rows[0]["n_recordings"] == 3
    assert rows[0]["median"] == pytest.approx(2.0)


# =============================================================================
# Eligibility: one exclusion, two counted diagnostics
# =============================================================================
def test_a_recording_without_an_onset_is_the_only_one_excluded() -> None:
    """The policy, in one assertion: missing excludes, at-delivery does not."""
    table = _offsets(
        [
            ("clean", -2 * _HOUR, -_HOUR), ("clean", -_HOUR, 0.0),
            ("missing", -2 * _HOUR, -_HOUR), ("missing", -_HOUR, np.nan),
            ("at_delivery", -2 * _HOUR, -2 * _HOUR), ("at_delivery", -_HOUR, -_HOUR),
        ]
    )

    record = cohort.second_stage_eligibility(table).set_index("guid")

    assert bool(record.loc["clean", "eligible"]) is True
    assert bool(record.loc["missing", "eligible"]) is False
    assert "no second-stage onset recorded" in str(record.loc["missing", "reason"])
    # Kept, and flagged. Excluding it would change the population every number is computed over.
    assert bool(record.loc["at_delivery", "eligible"]) is True
    assert bool(record.loc["at_delivery", "onset_at_delivery"]) is True
    assert bool(record.loc["clean", "onset_at_delivery"]) is False


def test_the_implied_onset_is_the_recordings_own_and_is_reported() -> None:
    r"""$t^{\mathrm{onset}}_{\mathrm{epoch}} = \texttt{epoch} - \texttt{second\_stage\_onset}$,
    which is a property of the recording and therefore the same on all of its segments."""
    table = _offsets([("a", -5 * _HOUR, -2 * _HOUR), ("a", -4 * _HOUR, -_HOUR)])

    record = cohort.second_stage_eligibility(table)

    assert float(record["implied_onset_epoch_s"].iloc[0]) == pytest.approx(-3 * _HOUR)
    assert float(record["onset_spread_s"].iloc[0]) == pytest.approx(0.0)
    assert bool(record["inconsistent_onset"].iloc[0]) is False


def test_a_float32_rounding_spread_is_not_reported_as_an_inconsistency() -> None:
    """Both operands are stored ``float32`` at magnitudes around $4 \\times 10^{4}$ s, so an exact
    comparison would report every recording inconsistent on arithmetic alone."""
    table = _offsets(
        [("a", -40000.0, -10000.0), ("a", -40000.5, -10000.0)]
    )

    record = cohort.second_stage_eligibility(table)

    assert float(record["onset_spread_s"].iloc[0]) == pytest.approx(0.5)
    assert bool(record["inconsistent_onset"].iloc[0]) is False


def test_an_onset_that_moves_within_a_recording_is_reported_as_inconsistent() -> None:
    """Non-vacuity for the tolerance above: an onset is a property of the recording, so when it
    moves by an hour across that recording's own segments the field was written wrong."""
    table = _offsets([("a", -5 * _HOUR, -2 * _HOUR), ("a", -4 * _HOUR, -2 * _HOUR)])

    record = cohort.second_stage_eligibility(table)

    assert float(record["onset_spread_s"].iloc[0]) == pytest.approx(_HOUR)
    assert bool(record["inconsistent_onset"].iloc[0]) is True
    # Counted, not excluded -- the same rule the at-delivery case follows.
    assert bool(record["eligible"].iloc[0]) is True


def test_the_eligibility_record_of_a_table_with_nothing_to_read_is_empty_not_absent() -> None:
    for frame in (pd.DataFrame(), pd.DataFrame({"guid": ["a"]})):
        record = cohort.second_stage_eligibility(frame)
        assert len(record) == 0
        assert "eligible" in record.columns and "reason" in record.columns


# =============================================================================
# The coverage readout
# =============================================================================
def test_the_readout_counts_the_rows_it_is_missing_on() -> None:
    """The denominator of every second-stage statement. A summary that dropped these rows would
    report a trajectory over a population it does not name."""
    table = _offsets(
        [("a", -2 * _HOUR, -_HOUR), ("a", -_HOUR, 0.0), ("b", -_HOUR, np.nan)]
    )

    record = cohort.second_stage_readout(table)

    assert record["present"] is True
    assert (record["n_rows"], record["n_finite"], record["n_nan"]) == (3, 2, 1)
    assert record["nan_fraction"] == pytest.approx(1.0 / 3.0)
    # Signed hours, reported as such.
    assert record["min_hours"] == pytest.approx(-1.0)
    assert record["max_hours"] == pytest.approx(0.0)
    assert (record["n_recordings"], record["n_recordings_eligible"]) == (2, 1)
    assert record["n_recordings_missing"] == 1


def test_the_readout_carries_both_diagnostics_beside_the_coverage() -> None:
    """A cohort whose onsets were written wrong is visible in the summary rather than only in a
    figure that looks odd."""
    table = _offsets(
        [
            ("at_delivery", -2 * _HOUR, -2 * _HOUR), ("at_delivery", -_HOUR, -_HOUR),
            ("moving", -5 * _HOUR, -2 * _HOUR), ("moving", -4 * _HOUR, -2 * _HOUR),
        ]
    )

    record = cohort.second_stage_readout(table)

    assert record["n_recordings_onset_at_delivery"] == 1
    assert record["n_recordings_inconsistent_onset"] == 1
    assert record["onset_consistency_tolerance_s"] == pytest.approx(
        cohort.ONSET_CONSISTENCY_TOLERANCE_S
    )


def test_a_table_without_the_column_reports_absence_rather_than_zeroes() -> None:
    """The same shape ``labor_onset_readout`` uses: a column that is not there is a different
    statement from a column that is there and empty, and zeroes would read as the second."""
    record = cohort.second_stage_readout(pd.DataFrame({"epoch": [-3600.0, -7200.0]}))

    assert record == {"present": False, "n_rows": 2}


def test_the_cohort_block_carries_the_readout_beside_the_labour_onset_one() -> None:
    """Both clocks' coverage in the summary, or a reader has to open the analysis to learn how
    much of the cohort the second one could describe."""
    table = _offsets([("a", -2 * _HOUR, -_HOUR), ("b", -_HOUR, np.nan)])
    table[labels.CLASS_COLUMN] = ["healthy", "healthy"]

    block = cohort.build_cohort_block(table, {})

    assert block[cohort.SECOND_STAGE_COLUMN]["present"] is True
    assert block[cohort.SECOND_STAGE_COLUMN]["n_nan"] == 1
    assert "time_from_labor_onset" in block


# =============================================================================
# The analysis: what a window is worth, and to whom
# =============================================================================
#: Where the second stage sits on the delivery clock in the fixtures below. A recording's implied
#: onset is ``epoch - second_stage_onset`` and must be constant across its own segments, so the two
#: fields are generated from this rather than independently -- a fixture whose epochs did not move
#: with its offsets would report every recording inconsistent and measure the diagnostic instead of
#: the trajectory.
_ONSET_EPOCH = -4.0 * _HOUR


def _clock_rows(
    *,
    n_by_class=(("healthy", 5), ("acidosis", 5)),
    window_offsets=((-1.0, 0.0), (1.0, 50.0)),
    segments: int = 1,
    stagger: bool = False,
) -> List[Dict[str, Any]]:
    """Two classes over two second-stage windows, each window carrying its own separation.

    Args:
        n_by_class: ``(class, recordings)`` pairs. The first class is the reference; every other is
            shifted by the window's own offset, so a window can be built separated or overlapping.
        window_offsets: ``(signed hours from onset, separation)`` per window. One before the onset
            and one after it by default, which is the half of the axis this clock exists for.
        segments: Segments each recording contributes to each window, so a test can tell a count of
            recordings from a count of segments.
        stagger: Move each recording's second stage to a different point of the **delivery** clock.
            The second-stage windows are unchanged; the delivery-clock windows multiply, which is
            what lets a test tell the two families apart.
    """
    rows: List[Dict[str, Any]] = []
    for index, (name, count) in enumerate(n_by_class):
        for recording in range(count):
            onset_epoch = _ONSET_EPOCH - (2.0 * _HOUR * recording if stagger else 0.0)
            for hours, offset in window_offsets:
                for segment in range(segments):
                    value = float(recording) + (float(offset) if index else 0.0)
                    rows.append(
                        {
                            "guid": f"{name}_{recording:02d}",
                            "epoch": onset_epoch + hours * _HOUR,
                            cohort.SECOND_STAGE_COLUMN: hours * _HOUR,
                            labels.CLASS_COLUMN: name,
                            labels.SUBGROUP_COLUMN: f"{name}_cs",
                            "mc_pred_gap": value + 0.1 * segment,
                            "source_conditioned_kl_raw": value,
                        }
                    )
    return rows


def _per_sample(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """A per-sample table carrying only what this analysis reads."""
    frame = pd.DataFrame(rows)
    for column in analysis.VALUE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.full(len(frame), np.nan)
    return frame


def _context(per_sample: pd.DataFrame) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record={}, retained={}, results={},
        vectors={},
    )
    return AnalysisContext(collection=collection, config={})


def _class_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """The class-axis per-recording-per-window frame the tests run over."""
    _, eligible = analysis.eligible_rows(_per_sample(rows))
    return analysis.build_per_recording(eligible)[labels.CLASS_COLUMN]


def test_the_grid_and_the_readouts_are_the_delivery_clocks_own() -> None:
    """One grid and one pair of readouts across both clocks, bound from the layer below rather than
    restated: a window on one clock's figure is otherwise not the same duration as a window on the
    other's, and the two pages stop being comparable while both look ordinary."""
    from teb_vae.lag_attn_rws.eval.analyses import time_to_delivery as delivery_clock

    assert analysis.TRAJECTORY_BIN_HOURS is cohort.TRAJECTORY_BIN_HOURS
    assert analysis.READOUTS is cohort.CLOCK_READOUTS
    assert analysis.READOUTS == delivery_clock.READOUTS
    assert analysis.VALUE_COLUMNS == delivery_clock.VALUE_COLUMNS


def test_separated_classes_are_significant_in_the_windows_they_are_separated_in() -> None:
    """A known answer: two classes drawn fifty nats apart in every window must survive Holm."""
    record = analysis.analyse_windows(
        _class_frame(_clock_rows(window_offsets=((-1.0, 50.0), (1.0, 50.0)))), "mc_pred_gap"
    )

    assert record["tested"] is True
    assert record["n_windows_tested"] == 2
    assert record["n_significant_windows"] == 2
    assert record["significant_bin_centers_h"] == pytest.approx([-0.75, 1.25])
    assert set(record["pairwise"])


def test_overlapping_classes_are_not_significant() -> None:
    """Non-vacuity for the case above: an implementation reporting significance unconditionally
    passes it."""
    record = analysis.analyse_windows(
        _class_frame(_clock_rows(window_offsets=((-1.0, 0.0), (1.0, 0.0)))), "mc_pred_gap"
    )

    assert record["n_significant_windows"] == 0
    assert record["pairwise"] == {}


def test_the_holm_family_is_this_clocks_windows_and_not_the_delivery_clocks() -> None:
    """The statistical contract, asserted rather than described. The two clocks are two families;
    correcting one against the other's window count would change every $p$ on this page for a
    reason that has nothing to do with this clock."""
    rows = _clock_rows(window_offsets=((-1.0, 50.0), (1.0, 50.0)), stagger=True)

    record = analysis.analyse_windows(_class_frame(rows), "mc_pred_gap")

    delivery_windows = cohort.add_time_bins(_per_sample(rows))[cohort.BIN_COLUMN].nunique()
    assert record["n_windows_tested"] == 2
    assert delivery_windows > 2, "the fixture must separate the two clocks' window counts"
    for window in record["per_window"]:
        assert window["correction"] == "holm"
        assert window["n_windows_in_family"] == 2
        assert window["p_holm"] >= window["p_value"]


def test_a_windows_value_for_a_recording_is_the_mean_over_its_own_segments() -> None:
    """Three segments of one recording in one window count once, at their mean -- not three times,
    which is the pseudo-replication the whole chain exists to keep out."""
    frame = _per_sample(
        [
            {"guid": "a", "epoch": _ONSET_EPOCH - _HOUR,
             cohort.SECOND_STAGE_COLUMN: -_HOUR, labels.CLASS_COLUMN: "healthy",
             "mc_pred_gap": value}
            for value in (1.0, 2.0, 6.0)
        ]
    )

    _, eligible = analysis.eligible_rows(frame)
    per_recording = analysis.build_per_recording(eligible)[labels.CLASS_COLUMN]

    assert len(per_recording) == 1
    assert float(per_recording["mc_pred_gap"].iloc[0]) == pytest.approx(3.0)


def test_the_reported_count_is_recordings_not_segments() -> None:
    rows = _clock_rows(segments=3)
    _, eligible = analysis.eligible_rows(_per_sample(rows))

    trajectory = analysis.build_trajectory_rows(analysis.build_per_recording(eligible))
    healthy = [
        row for row in trajectory
        if row["group"] == "healthy" and row["group_column"] == labels.CLASS_COLUMN
        and row["metric"] == "pred_gap_mc_nats"
    ]

    assert healthy, "the class axis produced no rows"
    assert {row["n_recordings"] for row in healthy} == {5}
    assert len(rows) == 60, "the fixture must hold more segments than recordings"


def test_the_pooled_row_counts_a_recording_once_however_many_windows_it_spans() -> None:
    """The frame the pooled test reads is keyed per (recording, window), so counting rows would
    pseudo-replicate the p-value by the windows-per-recording factor and report an ``n`` that is not
    a recording count."""
    class_frame = _class_frame(_clock_rows(n_by_class=(("healthy", 4), ("acidosis", 4))))

    record = analysis.analyse_windows(class_frame, "mc_pred_gap")

    # Each recording contributes one segment to each of two windows.
    assert len(class_frame) == 16
    assert record["pooled"]["n_per_group"] == {"healthy": 4, "acidosis": 4}
    assert record["pooled"]["confounded_by_time"] is True


def test_the_eligibility_counts_are_exact_on_a_frame_with_all_three_cases() -> None:
    """One recording missing its onset, one with an onset at delivery, one whose onset moves. Only
    the first changes the population; the other two are counted and kept, and a count that folded
    them into the drop would silently shrink every number on the page."""
    rows = _clock_rows(n_by_class=(("healthy", 3), ("acidosis", 3)))
    rows += [
        # No onset on one of its two segments: dropped, and the only recording that is.
        {"guid": "missing", "epoch": _ONSET_EPOCH - _HOUR, cohort.SECOND_STAGE_COLUMN: -_HOUR,
         labels.CLASS_COLUMN: "hie", "mc_pred_gap": 1.0, "source_conditioned_kl_raw": 1.0},
        {"guid": "missing", "epoch": _ONSET_EPOCH + _HOUR, cohort.SECOND_STAGE_COLUMN: np.nan,
         labels.CLASS_COLUMN: "hie", "mc_pred_gap": 1.0, "source_conditioned_kl_raw": 1.0},
        # Implied onset exactly at delivery: the "not recorded" sentinel, kept and flagged.
        {"guid": "at_delivery", "epoch": -_HOUR, cohort.SECOND_STAGE_COLUMN: -_HOUR,
         labels.CLASS_COLUMN: "hie", "mc_pred_gap": 2.0, "source_conditioned_kl_raw": 2.0},
        {"guid": "at_delivery", "epoch": -2 * _HOUR, cohort.SECOND_STAGE_COLUMN: -2 * _HOUR,
         labels.CLASS_COLUMN: "hie", "mc_pred_gap": 2.0, "source_conditioned_kl_raw": 2.0},
        # An onset that moves by an hour within one recording: impossible, kept and flagged.
        {"guid": "moving", "epoch": _ONSET_EPOCH - 2 * _HOUR,
         cohort.SECOND_STAGE_COLUMN: -2 * _HOUR, labels.CLASS_COLUMN: "hie",
         "mc_pred_gap": 3.0, "source_conditioned_kl_raw": 3.0},
        {"guid": "moving", "epoch": _ONSET_EPOCH - _HOUR, cohort.SECOND_STAGE_COLUMN: -2 * _HOUR,
         labels.CLASS_COLUMN: "hie", "mc_pred_gap": 3.0, "source_conditioned_kl_raw": 3.0},
    ]
    eligibility, eligible = analysis.eligible_rows(_per_sample(rows))

    population = analysis.eligibility_summary(eligibility)

    assert population["n_recordings"] == 9
    assert population["n_eligible"] == 8
    assert population["n_dropped_no_onset"] == 1
    assert population["n_onset_at_delivery"] == 1
    assert population["n_inconsistent_onset"] == 1
    assert population["onset_consistency_tolerance_s"] == pytest.approx(
        cohort.ONSET_CONSISTENCY_TOLERANCE_S
    )
    # And the drop is the only one that changed who is scored.
    assert set(eligible["guid"]) == set(eligibility.loc[eligibility["eligible"], "guid"])
    assert "missing" not in set(eligible["guid"])


# =============================================================================
# What the analysis writes
# =============================================================================
def test_the_analysis_writes_its_five_tables_and_both_figures(tmp_path) -> None:
    result = analysis.run_second_stage_analysis(
        _context(_per_sample(_clock_rows(window_offsets=((-1.0, 50.0), (1.0, 50.0))))),
        eval_config={}, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / analysis.ANALYSIS_DIRNAME
    for name in (
        analysis.ELIGIBILITY_FILENAME, analysis.TRAJECTORY_FILENAME,
        analysis.PER_RECORDING_FILENAME, analysis.SIGNIFICANCE_FILENAME,
        analysis.PAIRWISE_FILENAME, analysis.TRAJECTORY_FIGURE, analysis.WINDOWS_FIGURE,
    ):
        assert (directory / name).is_file(), name
        assert name in result["files"], name
    # A subset population, declared as such: the coverage block compares uncapped analyses only,
    # and this one legitimately scores fewer segments than every analysis beside it.
    assert result["plan"]["capped"] is True
    assert str(result["plan"]["reason"]).strip()
    assert result["n_samples"] == 20
    assert result["composition"] == {"n_recordings": 10, "n_windows": 2}
    assert result["bin_width_hours"] == pytest.approx(0.5)
    assert [record["metric_column"] for record in result["significance"]] == list(
        analysis.VALUE_COLUMNS
    )
    # The summary carries the headline of each test, not the per-window detail, which is on disk.
    assert all("per_window" not in record for record in result["significance"])
    eligibility = pd.read_csv(directory / analysis.ELIGIBILITY_FILENAME)
    assert set(eligibility.columns) >= {"guid", "eligible", "reason", "implied_onset_epoch_s"}


def test_the_method_states_that_this_clocks_family_stands_alone() -> None:
    """The one sentence a reader needs before quoting a $p$ from both clocks at once."""
    record = analysis.analyse_windows(_class_frame(_clock_rows()), "mc_pred_gap")

    assert "NOT corrected jointly" in record["method"]
    assert "own clock" in record["method"]


@pytest.mark.parametrize(
    "rows, fragment",
    [
        ([], "empty"),
        (
            [{"guid": "a", "epoch": -_HOUR, labels.CLASS_COLUMN: "healthy", "mc_pred_gap": 1.0}],
            "second_stage_onset",
        ),
        (
            [{"guid": "a", "epoch": -_HOUR, cohort.SECOND_STAGE_COLUMN: np.nan,
              labels.CLASS_COLUMN: "healthy", "mc_pred_gap": 1.0}],
            "no recording carried a second-stage onset",
        ),
    ],
)
def test_a_population_this_clock_cannot_describe_is_a_recorded_skip(rows, fragment, tmp_path):
    """Three of the causes, each named in its own message: an empty table, a table collected before
    the column existed -- which is what an old run directory looks like -- and a cohort the
    labour-onset table has never heard of."""
    result = analysis.run_second_stage_analysis(
        _context(_per_sample(rows) if rows else pd.DataFrame()),
        eval_config={}, output_dir=tmp_path, probe=None,
    )

    assert result["skipped"] is True
    assert fragment in result["reason"]
    assert result["n_samples"] is None
    assert not (tmp_path / analysis.ANALYSIS_DIRNAME).exists()


def test_a_single_class_split_is_a_recorded_skip_naming_the_classes(tmp_path) -> None:
    """The fourth cause. Unlike the delivery clock, which still draws its trajectory: this analysis
    exists for the class contrast on a subset population, and with one class it would report a shape
    the other clock already draws over more recordings."""
    rows = [row for row in _clock_rows() if row[labels.CLASS_COLUMN] == "healthy"]

    result = analysis.run_second_stage_analysis(
        _context(_per_sample(rows)), eval_config={}, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is True
    assert "fewer than two clinical classes" in result["reason"]
    assert not (tmp_path / analysis.ANALYSIS_DIRNAME).exists()


# =============================================================================
# The two figures, and the orientation that is the whole difference
# =============================================================================
def _windows_figure(rows: List[Dict[str, Any]]):
    """Build the windows page from a hand-made per-sample table, with its significance records."""
    class_frame = _class_frame(rows)
    records = [analysis.analyse_windows(class_frame, column) for column in analysis.VALUE_COLUMNS]
    return analysis.build_windows_figure(class_frame, records), records


def _zero_lines(ax) -> int:
    """How many vertical lines this axes draws at $x = 0$ -- the onset mark."""
    return sum(
        1 for line in ax.lines
        if len(np.atleast_1d(line.get_xdata())) == 2
        and np.allclose(np.asarray(line.get_xdata(), dtype=np.float64), 0.0)
    )


def test_the_windows_page_is_not_inverted_and_marks_the_onset() -> None:
    """The delivery clock inverts its axis so delivery sits at the right; this coordinate is signed
    and reads naturally left to right, and inverting it would put "after the onset" on the left with
    nothing on the page saying so."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, _ = _windows_figure(_clock_rows(window_offsets=((-1.0, 50.0), (1.0, 50.0))))
    try:
        # Two readouts, each a violin row and a strip row, then the effect-size heatmap -- plus the
        # colourbar axes the heatmap attaches.
        assert len(figure.axes) == 6
        for index in range(4):
            low, high = figure.axes[index].get_xlim()
            assert low < high, f"panel {index} is inverted"
            assert _zero_lines(figure.axes[index]) == 1, f"panel {index} does not mark the onset"
        for index in (0, 2):
            assert figure.axes[index].get_xlim() == pytest.approx(
                figure.axes[index + 1].get_xlim()
            )
        assert all(
            "negative" in figure.axes[index].get_xlabel() for index in (1, 3)
        ), "the strips do not name the sign convention"
    finally:
        shared_figures.plt.close(figure)


def test_the_strip_clears_alpha_in_the_window_the_classes_are_separated_in() -> None:
    """A known answer in both directions on one page: the same two classes, apart in the window
    before the onset and identical in the window after it."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    figure, records = _windows_figure(
        _clock_rows(window_offsets=((-1.0, 50.0), (1.0, 0.0)))
    )
    try:
        strip = figure.axes[1]
        bars = sorted(
            (float(patch.get_x() + patch.get_width() / 2.0), float(patch.get_height()))
            for patch in strip.patches
        )
        threshold = float(-np.log10(analysis.DEFAULT_ALPHA))

        # Ascending, because the axis is not inverted: the window before the onset comes first.
        assert [centre for centre, _ in bars] == pytest.approx([-0.75, 1.25])
        assert bars[0][1] > threshold
        assert bars[1][1] < threshold
    finally:
        shared_figures.plt.close(figure)

    assert records[0]["n_significant_windows"] == 1
    assert records[0]["significant_bin_centers_h"] == pytest.approx([-0.75])


def test_the_trajectory_figure_names_the_sign_convention_and_marks_the_onset() -> None:
    """A reader who took a negative value for "after" would read the whole trajectory backwards,
    and nothing else on the page would contradict them."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    _, eligible = analysis.eligible_rows(_per_sample(_clock_rows()))
    rows = analysis.build_trajectory_rows(analysis.build_per_recording(eligible))

    figure = analysis.build_trajectory_figure(rows, labels.CLASS_COLUMN)
    try:
        for ax in figure.axes:
            low, high = ax.get_xlim()
            assert low < high
            assert "negative" in ax.get_xlabel() and "positive" in ax.get_xlabel()
            assert _zero_lines(ax) == 1
        annotations = sorted(text.get_text() for text in figure.axes[0].texts)
    finally:
        shared_figures.plt.close(figure)

    # Five recordings in each of two windows, for each of two classes.
    assert annotations == ["5"] * 4


@pytest.mark.parametrize("builder", ("trajectory", "windows"))
def test_an_empty_population_draws_the_note_rather_than_raising(builder) -> None:
    """Both builders are reachable with nothing in them -- from a test, and from a cohort whose
    readouts are all non-finite -- and the run's final step is the worst place to discover it."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    if builder == "trajectory":
        figure = analysis.build_trajectory_figure([], labels.CLASS_COLUMN)
    else:
        figure = analysis.build_windows_figure(pd.DataFrame(), [])
    try:
        notes = [text.get_text() for axis in figure.axes for text in axis.texts]
        assert notes.count(shared_figures.EMPTY_NOTE) >= 1
    finally:
        shared_figures.plt.close(figure)
