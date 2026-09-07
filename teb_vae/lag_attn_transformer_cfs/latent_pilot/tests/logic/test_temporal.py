r"""Anchor timestamps, window and bin boundaries, deduplication and the bag reductions.

Known answers, worked by hand and written down as numbers rather than as re-derivations of the
formula under test. Everything here is synthetic: invented GUIDs, invented delivery times, small
numpy value matrices. Nothing opens an HDF5 file, builds a model or runs an optimizer.

The one place a real object appears is the support adapter's check, which builds a
``TrimmedRawGeometry`` -- a validated dataclass of four integers -- and a stub carrying the three
attributes the adapter reads. That is deliberately not a model: the point of the check is that the
adapter calls the *objective's* mask rather than a second definition of support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data

#: The shipped geometry's numbers, used where a test needs a realistic one.
TRIM_MINUTES = 1.0
SEQUENCE_LENGTH = 300
HORIZON = 10


def _anchors(rows):
    """Build an anchor table with times, from ``(guid, epoch, anchor, value)`` tuples.

    Returns:
        ``(frame, values)``: the table with :data:`ROW_COLUMN` aligned to the value matrix.
    """
    frame = pd.DataFrame([
        {
            data.GUID_COLUMN: guid,
            data.EPOCH_COLUMN: float(epoch),
            data.ANCHOR_COLUMN: int(anchor),
            data.ROW_COLUMN: index,
        }
        for index, (guid, epoch, anchor, _value) in enumerate(rows)
    ])
    values = np.asarray([[float(value)] for *_rest, value in rows], dtype=np.float64)
    return data.add_anchor_times(
        frame, trim_minutes=TRIM_MINUTES, horizon=HORIZON, forecast_shift=0
    ), values


# =============================================================================
# The timestamp itself
# =============================================================================
def test_the_worked_case_is_forty_nine_minutes_before_delivery():
    """epoch -3600, one-minute trim, anchor 150 is 49 minutes out -- not 60."""
    seconds = data.anchor_seconds(-3600.0, 150, trim_minutes=TRIM_MINUTES)

    assert float(seconds) == -2940.0
    assert float(data.hours_before_delivery(seconds)) * 60.0 == pytest.approx(49.0)


def test_the_trim_offset_follows_the_loader_rather_than_sixty_times_the_minutes():
    """The loader drops whole decimated steps, so the offset is not always 60 m."""
    assert data.step_seconds() == 4.0
    assert data.trim_seconds(None) == 0.0
    # 1.0 min: 240 raw samples, 15 decimated steps, 60 s. The two conventions agree here, which is
    # why the worked case above comes out identical either way.
    assert data.trim_seconds(1.0) == 60.0
    # 1.5 min: 360 raw samples, which is 22.5 steps and therefore 22 -- 88 s, not 90. Reading it as
    # 60 m would put every anchor of such a dataset two seconds early.
    assert data.trim_seconds(1.5) == 88.0


def test_the_segment_span_and_coarse_bound_keep_a_crossing_segment():
    """A segment starting before the window still carries anchors inside it."""
    span = data.segment_span_seconds(SEQUENCE_LENGTH, TRIM_MINUTES)
    assert span == 1320.0, "300 trimmed steps of 4 s, plus one minute trimmed from each end"

    bound = data.coarse_epoch_min(3.0, span)
    assert bound == -12120.0

    # This segment starts 3h03m before delivery, so a filter at -10800 would drop it outright.
    crossing_epoch = -11000.0
    assert crossing_epoch < -10800.0
    assert crossing_epoch >= bound, "the widened bound must still admit it"

    frame, _values = _anchors([("SYNTH-CROSS", crossing_epoch, 40, 0.0)])
    hours = float(frame.loc[0, data.HOURS_COLUMN])
    assert hours == pytest.approx((11000.0 - 60.0 - 160.0) / 3600.0)
    assert hours < 3.0, "the anchor itself is inside the preserved window"
    kept = data.retained(data.mark_window_and_delivery(frame, preservation_hours=3.0))
    assert len(kept) == 1


# =============================================================================
# Window and bin boundaries
# =============================================================================
@pytest.mark.parametrize(
    "hours,inside",
    [
        (0.0, False),      # delivery itself is not in (0, 3]
        (1e-6, True),
        (0.5, True),
        (1.0, True),
        (3.0, True),       # the closed upper edge
        (3.0 + 1e-6, False),
    ],
)
def test_preservation_window_edges(hours, inside):
    """The window is half-open at zero and closed at three hours."""
    assert bool(data.in_window(hours, 0.0, 3.0)) is inside


@pytest.mark.parametrize(
    "hours,inside", [(0.0, False), (0.5, True), (1.0, True), (1.0 + 1e-6, False), (3.0, False)]
)
def test_supervised_window_edges(hours, inside):
    """An anchor exactly one hour out is supervised; one just beyond it is not."""
    assert bool(data.in_window(hours, 0.0, 1.0)) is inside


def test_the_six_fixed_bins_and_their_edges():
    """Six half-hour bins over three hours, index 0 nearest delivery, closed at the upper edge."""
    edges = data.bin_edges(bin_hours=0.5, preservation_hours=3.0)
    assert edges == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0)]

    frame = pd.DataFrame({data.HOURS_COLUMN: [0.0, 1e-6, 0.5, 0.5 + 1e-6, 1.0, 3.0, 3.1]})
    binned = data.assign_time_bins(frame, bin_hours=0.5, preservation_hours=3.0)

    assert binned[data.BIN_COLUMN].tolist() == [-1, 0, 0, 1, 1, 5, -1]
    assert binned[data.BIN_LABEL_COLUMN].tolist() == [
        "", "(0, 0.5]", "(0, 0.5]", "(0.5, 1]", "(0.5, 1]", "(2.5, 3]", "",
    ]


# =============================================================================
# Post-delivery endpoints
# =============================================================================
def test_an_anchor_before_delivery_can_still_be_scored_past_it():
    """epoch < 0 does not prove the forecast window is pre-delivery."""
    # Anchor 0 of a segment starting 200 s before delivery observes at -140 s; its ten-step
    # forecast ends at -100 s, which is still before delivery.
    frame, _values = _anchors([("SYNTH-END", -200.0, 0, 0.0)])
    assert float(frame.loc[0, data.ANCHOR_SECONDS_COLUMN]) == -140.0
    assert float(frame.loc[0, data.LATEST_SCORED_COLUMN]) == -100.0
    assert len(data.retained(data.mark_window_and_delivery(frame, preservation_hours=3.0))) == 1

    # Anchor 5 of a segment starting 100 s before delivery observes at -20 s -- inside the window --
    # but its forecast reaches +20 s, past the landmark.
    frame, _values = _anchors([("SYNTH-END", -100.0, 5, 0.0)])
    assert float(frame.loc[0, data.ANCHOR_SECONDS_COLUMN]) == -20.0
    assert float(frame.loc[0, data.LATEST_SCORED_COLUMN]) == 20.0
    marked = data.mark_window_and_delivery(frame, preservation_hours=3.0)
    assert marked.loc[0, data.EXCLUSION_COLUMN] == data.EXCLUDED_POST_DELIVERY
    assert data.anchor_exclusion_counts(marked) == {data.EXCLUDED_POST_DELIVERY: 1}


def test_a_forecast_clock_shift_moves_the_endpoint_and_the_verdict():
    """The same anchor passes under the stored clock and fails under an advancing one."""
    frame = pd.DataFrame([{
        data.GUID_COLUMN: "SYNTH-SHIFT",
        data.EPOCH_COLUMN: -200.0,
        data.ANCHOR_COLUMN: 0,
        data.ROW_COLUMN: 0,
    }])

    stored = data.add_anchor_times(
        frame, trim_minutes=TRIM_MINUTES, horizon=HORIZON, forecast_shift=0
    )
    assert float(stored.loc[0, data.LATEST_SCORED_COLUMN]) == -100.0
    assert data.mark_window_and_delivery(
        stored, preservation_hours=3.0
    ).loc[0, data.EXCLUSION_COLUMN] == ""

    # 25 steps of advance is 100 s, which lands the last scored coefficient exactly at delivery.
    shifted = data.add_anchor_times(
        frame, trim_minutes=TRIM_MINUTES, horizon=HORIZON, forecast_shift=25
    )
    assert float(shifted.loc[0, data.LATEST_SCORED_COLUMN]) == 0.0
    assert data.mark_window_and_delivery(
        shifted, preservation_hours=3.0
    ).loc[0, data.EXCLUSION_COLUMN] == data.EXCLUDED_POST_DELIVERY


def test_max_forecast_shift_reads_the_furthest_advance():
    """A delaying clock reaches less far than the unshifted one, so the maximum is the bound."""
    class _Stub:
        target_forecast_shift = None

    assert data.max_forecast_shift(_Stub()) == 0
    _Stub.target_forecast_shift = (0, 3, 7, 2)
    assert data.max_forecast_shift(_Stub()) == 7
    _Stub.target_forecast_shift = (-4, -9, -1)
    assert data.max_forecast_shift(_Stub()) == -1


# =============================================================================
# Deduplication
# =============================================================================
def test_a_repeated_segment_anchor_key_is_counted_once():
    """The same key twice is a collection defect, and the second copy is marked."""
    frame, _values = _anchors([
        ("SYNTH-DUP", -3600.0, 150, 0.0),
        ("SYNTH-DUP", -3600.0, 150, 0.0),
    ])
    deduplicated = data.deduplicate_anchors(
        data.mark_window_and_delivery(frame, preservation_hours=3.0)
    )
    assert len(data.retained(deduplicated)) == 1
    assert data.anchor_exclusion_counts(deduplicated) == {data.EXCLUDED_DUPLICATE_KEY: 1}


def test_overlapping_segments_keep_the_anchor_with_more_history():
    """Two segments supplying one instant: the survivor is the one further into its own segment."""
    # Both rows land on -2940 s: anchor 150 of a segment starting at -3600, and anchor 0 of a
    # segment starting at -3000.
    frame, _values = _anchors([
        ("SYNTH-OVERLAP", -3000.0, 0, 0.0),
        ("SYNTH-OVERLAP", -3600.0, 150, 1.0),
    ])
    assert frame[data.ANCHOR_SECONDS_COLUMN].tolist() == [-2940.0, -2940.0]

    deduplicated = data.deduplicate_anchors(
        data.mark_window_and_delivery(frame, preservation_hours=3.0)
    )
    kept = data.retained(deduplicated)
    assert len(kept) == 1
    assert int(kept.iloc[0][data.ANCHOR_COLUMN]) == 150
    assert deduplicated.loc[0, data.EXCLUSION_COLUMN] == data.EXCLUDED_DUPLICATE_TIME


def test_deduplication_does_not_touch_two_recordings_at_the_same_instant():
    """Different recordings may of course share a clock time."""
    frame, _values = _anchors([
        ("SYNTH-P1", -3600.0, 150, 0.0),
        ("SYNTH-P2", -3600.0, 150, 0.0),
    ])
    deduplicated = data.deduplicate_anchors(
        data.mark_window_and_delivery(frame, preservation_hours=3.0)
    )
    assert len(data.retained(deduplicated)) == 2


# =============================================================================
# Reductions
# =============================================================================
def _at_hours(guid, epoch, hours, value):
    """One anchor row placed at a chosen time before delivery.

    Args:
        guid: The recording.
        epoch: The segment's untrimmed start.
        hours: Where the anchor should land, in hours before delivery.
        value: Its scalar latent value.

    Returns:
        The ``(guid, epoch, anchor, value)`` tuple whose anchor index lands on that time.
    """
    anchor = (-hours * data.SECONDS_PER_HOUR - epoch - data.trim_seconds(TRIM_MINUTES)) / 4.0
    assert abs(anchor - round(anchor)) < 1e-9, "choose an epoch that puts the time on a step"
    return (guid, epoch, int(round(anchor)), value)


def test_segment_averaging_stops_a_dense_segment_from_outvoting_a_sparse_one():
    """Ten anchors in one segment and one in another weigh equally at the recording."""
    # Ten consecutive anchors of one segment, all inside the supervised hour.
    rows = [("SYNTH-R", -2000.0, 121 + step, 0.0) for step in range(10)]
    rows.append(_at_hours("SYNTH-R", -1500.0, 0.40, 1.0))
    frame, values = _anchors(rows)

    segments, segment_values = data.segment_means(frame, values)
    assert len(segments) == 2
    assert sorted(segments["n_anchors"].tolist()) == [1, 10]

    recordings, matrix = data.recording_means(segments, segment_values)
    assert len(recordings) == 1
    # The segment means are 0 and 1, so an unweighted recording mean is exactly one half. A flat
    # mean over anchors would have been 1/11.
    assert float(matrix[0, 0]) == pytest.approx(0.5)
    assert int(recordings.iloc[0]["n_anchors"]) == 11
    assert int(recordings.iloc[0]["n_segments"]) == 2


def test_the_recency_weight_favours_the_later_segment_by_its_half_life():
    """A 30-minute half-life inside the supervised hour, applied to both classes alike."""
    frame, values = _anchors([
        _at_hours("SYNTH-W", -4000.0, 0.9, 0.0),
        _at_hours("SYNTH-W", -1000.0, 0.1, 1.0),
    ])
    bags, matrix = data.recording_bags(
        frame, values, supervised_hours=1.0, halflife_hours=0.5
    )

    early, late = 2.0 ** (-0.9 / 0.5), 2.0 ** (-0.1 / 0.5)
    assert float(matrix[0, 0]) == pytest.approx(late / (early + late))
    assert float(matrix[0, 0]) == pytest.approx(0.7519, abs=1e-4)


def test_the_bag_ignores_anchors_outside_the_supervised_hour():
    """Anchors one to three hours out carry preservation supervision only, never the bag."""
    frame, values = _anchors([
        _at_hours("SYNTH-X", -8000.0, 2.0, 100.0),
        _at_hours("SYNTH-X", -2000.0, 0.4, 1.0),
    ])
    bags, matrix = data.recording_bags(
        frame, values, supervised_hours=1.0, halflife_hours=0.5
    )
    assert float(matrix[0, 0]) == 1.0
    assert int(bags.iloc[0]["n_segments"]) == 1

    # The same anchors are all teacher support: preservation spans the whole preserved window.
    assert len(data.retained(frame)) == 2


def test_bins_a_recording_has_no_anchors_in_are_absent_rather_than_filled():
    """A gap in a trajectory is a gap, not an interpolated point."""
    frame, values = _anchors([
        _at_hours("SYNTH-Y", -2000.0, 0.4, 1.0),
        _at_hours("SYNTH-Y", -11000.0, 2.8, 2.0),
    ])
    bins, matrix = data.recording_bin_means(
        frame, values, bin_hours=0.5, preservation_hours=3.0
    )

    assert bins[data.BIN_COLUMN].tolist() == [0, 5]
    assert bins[data.BIN_LABEL_COLUMN].tolist() == ["(0, 0.5]", "(2.5, 3]"]
    assert matrix[:, 0].tolist() == [1.0, 2.0]
    assert len(bins) == 2, "no row is emitted for the four unobserved bins"


def test_paired_windows_only_return_recordings_observed_in_them():
    """The paired early/late comparison is paired because an absent window stays absent."""
    frame, values = _anchors([
        _at_hours("SYNTH-BOTH", -10000.0, 2.5, 3.0),
        _at_hours("SYNTH-BOTH", -2000.0, 0.4, 1.0),
        _at_hours("SYNTH-LATE-ONLY", -2000.0, 0.4, 5.0),
    ])
    early, early_values = data.window_means(frame, values, low=2.0, high=3.0)
    late, late_values = data.window_means(frame, values, low=0.0, high=1.0)

    assert early[data.GUID_COLUMN].tolist() == ["SYNTH-BOTH"]
    assert sorted(late[data.GUID_COLUMN].tolist()) == ["SYNTH-BOTH", "SYNTH-LATE-ONLY"]
    assert float(early_values[0, 0]) == 3.0
    # Only the recording present in both windows can contribute a within-recording change.
    paired = set(early[data.GUID_COLUMN]) & set(late[data.GUID_COLUMN])
    assert paired == {"SYNTH-BOTH"}


# =============================================================================
# Late eligibility
# =============================================================================
def _recordings(guids):
    """A minimal recording table for the eligibility rules."""
    return pd.DataFrame([
        {data.GUID_COLUMN: guid, data.SPLIT_COLUMN: "train", data.EXCLUSION_COLUMN: prior}
        for guid, prior in guids
    ])


def test_late_eligibility_names_the_rule_each_recording_failed():
    """Three ways to be ineligible, each reported as itself rather than as one bucket."""
    frame, _values = _anchors([
        # Nothing inside the supervised hour at all.
        _at_hours("SYNTH-E1", -11000.0, 2.8, 0.0),
        # One late segment where two are required.
        _at_hours("SYNTH-E2", -2000.0, 0.4, 0.0),
        _at_hours("SYNTH-E2", -2000.0, 0.3, 0.0),
        # Two late segments, but nothing within the final 30 minutes.
        _at_hours("SYNTH-E3", -4000.0, 0.9, 0.0),
        _at_hours("SYNTH-E3", -3000.0, 0.8, 0.0),
        # Two late segments and an anchor 12 minutes out.
        _at_hours("SYNTH-OK", -4000.0, 0.9, 0.0),
        _at_hours("SYNTH-OK", -1000.0, 0.2, 0.0),
    ])
    recordings = _recordings([
        ("SYNTH-E1", ""), ("SYNTH-E2", ""), ("SYNTH-E3", ""), ("SYNTH-OK", ""),
    ])
    judged = data.late_eligibility(
        frame, recordings,
        supervised_hours=1.0, min_late_segments=2, final_anchor_within_minutes=30.0,
    ).set_index(data.GUID_COLUMN)

    assert judged.loc["SYNTH-E1", "eligibility_reason"] == data.INELIGIBLE_NO_LATE_ANCHOR
    assert judged.loc["SYNTH-E2", "eligibility_reason"] == data.INELIGIBLE_FEW_LATE_SEGMENTS
    assert judged.loc["SYNTH-E3", "eligibility_reason"] == data.INELIGIBLE_NO_FINAL_ANCHOR
    assert bool(judged.loc["SYNTH-OK", "eligible"]) is True
    assert judged.loc["SYNTH-OK", "eligibility_reason"] == ""

    # The actual last observation is recorded, never moved to the delivery landmark.
    assert float(judged.loc["SYNTH-OK", "last_anchor_hours"]) == pytest.approx(0.2)
    assert np.isnan(float(judged.loc["SYNTH-E1", "last_anchor_hours"]))


def test_a_recording_already_excluded_for_its_label_keeps_that_reason():
    """Coverage rules never overwrite a labelling exclusion."""
    frame, _values = _anchors([_at_hours("SYNTH-Z", -1000.0, 0.2, 0.0)])
    recordings = _recordings([("SYNTH-Z", data.EXCLUDED_CONFLICTING_CLASS)])
    judged = data.late_eligibility(
        frame, recordings,
        supervised_hours=1.0, min_late_segments=1, final_anchor_within_minutes=30.0,
    )
    assert bool(judged.iloc[0]["eligible"]) is False
    assert judged.iloc[0]["eligibility_reason"] == data.EXCLUDED_CONFLICTING_CLASS


def test_eligibility_applies_the_same_rule_to_both_classes():
    """The rule reads coverage only; nothing in it can see an outcome."""
    frame, _values = _anchors([
        _at_hours("SYNTH-HEALTHY", -1000.0, 0.2, 0.0),
        _at_hours("SYNTH-ADVERSE", -1000.0, 0.2, 0.0),
    ])
    recordings = _recordings([("SYNTH-HEALTHY", ""), ("SYNTH-ADVERSE", "")])
    recordings[data.OUTCOME_COLUMN] = [0, 1]
    judged = data.late_eligibility(
        frame, recordings,
        supervised_hours=1.0, min_late_segments=1, final_anchor_within_minutes=30.0,
    )
    assert judged["eligible"].tolist() == [True, True]
    assert judged["n_late_segments"].tolist() == [1, 1]


# =============================================================================
# The support adapter
# =============================================================================
def test_support_is_the_objective_mask_and_not_a_second_definition():
    """The warm-up prefix, the anchor's validity and the coverage floor all come from the model."""
    import torch

    from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

    geometry = TrimmedRawGeometry(raw_len=320, decimation=16, horizon=4, warmup=2)

    class _StubModel:
        """The three attributes the adapter reads. Not a model: nothing here decodes anything."""

        def __init__(self):
            self.geometry = geometry
            self.coverage_floor = 0.9

        @staticmethod
        def scored_weight(weight):
            """The stored clock's identity: no per-channel shift to pool over."""
            return weight

    anchors = torch.arange(geometry.t_valid).unsqueeze(0)
    valid = torch.ones_like(anchors, dtype=torch.bool)
    weight = torch.ones(1, geometry.t)

    support, coverage = data.contributing_support(
        _StubModel(), weight=weight, anchor_index=anchors, anchor_valid=valid
    )
    # The first two anchors are inside the warm-up, so they carry no forecast term at all.
    assert support[0, :2].tolist() == [False, False]
    assert support[0, 2:].all()
    assert float(coverage[0, -1]) == 1.0

    # A gap in the middle of one anchor's forecast window drops it below the floor entirely.
    gapped = weight.clone()
    gapped[0, 6] = 0.0
    support, _coverage = data.contributing_support(
        _StubModel(), weight=gapped, anchor_index=anchors, anchor_valid=valid
    )
    assert not bool(support[0, 3]), "anchor 3 forecasts steps 4..7, which now contains a gap"


def test_padded_anchor_slots_never_contribute():
    """A padded slot repeats a legal index, so without the validity flag it would score twice."""
    import torch

    from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

    stub_geometry = TrimmedRawGeometry(raw_len=320, decimation=16, horizon=4, warmup=2)

    class _StubModel:
        geometry = stub_geometry
        coverage_floor = 0.0

        @staticmethod
        def scored_weight(weight):
            return weight

    anchors = torch.tensor([[4, 5, 5]])
    valid = torch.tensor([[True, True, False]])
    support, _coverage = data.contributing_support(
        _StubModel(),
        weight=torch.ones(1, stub_geometry.t),
        anchor_index=anchors,
        anchor_valid=valid,
    )
    assert support[0].tolist() == [True, True, False]
