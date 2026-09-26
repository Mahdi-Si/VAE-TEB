r"""Contraction-conditioned coupling.

The readout conditions on **timing** rather than on the forecast's shape. What can go wrong in it
is not the arithmetic but the comparison: the control anchors have to be drawn *within each
recording* and matched to that recording's event count, or the difference is a
difference between recordings wearing two labels. Every assertion here therefore constructs the
condition it needs -- a synthetic anchor table whose near-contraction rows carry a known extra gap
-- rather than hoping the fixture supplies one, because the generated cohort is a fixture about
counts and shapes and no finding about coupling may be read off it.

The detector itself is tested in ``test_eval_events_detector.py``; this file starts where the
contraction timing has already reached the per-anchor table.
"""
from __future__ import annotations

import types
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import events as events_analysis
from teb_vae.lag_attn_cfs.eval.collect import CONTRACTION_AGE_COLUMN

#: Bootstrap settings for the constructed cases: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0, "event_lag_window_s": 120.0}


def _anchor_frame(*, guids: int, anchors: int, near_every: int, gap: float) -> pd.DataFrame:
    """A per-anchor table whose near-contraction anchors carry a larger gap by construction.

    The noise is zero-mean and independent of the anchor index on purpose. A ramp in the anchor
    would make the near-contraction anchors -- being a regular subgrid -- systematically earlier
    than the controls, and the "no coupling" case would then report a small but entirely one-sided
    difference that is a property of the fixture rather than of the code.

    Args:
        guids: Recordings in the table.
        anchors: Anchors per recording.
        near_every: One anchor in this many sits within the window of a contraction.
        gap: The extra ``mc_pred_gap`` those anchors carry.

    Returns:
        The table, carrying both conditioned readouts and the contraction age column.
    """
    rng = np.random.default_rng(11)
    rows = []
    for guid in range(guids):
        for anchor in range(anchors):
            near = (anchor % near_every) == 0
            rows.append(
                {
                    "guid": f"g{guid}",
                    "epoch": -1000.0 * guid,
                    "anchor": anchor,
                    "mc_pred_gap": (gap if near else 0.0) + 0.05 * float(rng.standard_normal()),
                    "kld_per_t": 1.0,
                    CONTRACTION_AGE_COLUMN: 10.0 if near else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _context(
    per_anchor: Optional[pd.DataFrame] = None,
    per_sample: Optional[pd.DataFrame] = None,
) -> AnalysisContext:
    """A context over a stub collection carrying only the two tables this analysis reads."""
    collection = types.SimpleNamespace(
        per_sample=pd.DataFrame() if per_sample is None else per_sample,
        per_anchor=pd.DataFrame() if per_anchor is None else per_anchor,
        record={},
        retained={},
        results={},
    )
    return AnalysisContext(collection=collection, config={}, task=None, loader=None)


# =================================================================================================
# The control anchors
# =================================================================================================
def test_the_control_anchors_are_count_matched_inside_each_recording() -> None:
    """Per recording, not pooled: a recording contributing forty event anchors and one
    contributing four would otherwise be compared against controls drawn mostly from the first."""
    frame = _anchor_frame(guids=4, anchors=100, near_every=5, gap=1.0)

    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    counts = split.groupby(["guid", "condition"]).size().unstack("condition")
    assert (counts["event"] == counts["control"]).all()
    assert (counts["event"] == 20).all()


def test_an_anchor_with_no_contraction_behind_it_is_a_control_rather_than_dropped() -> None:
    """NaN in the age column means no contraction preceded this anchor *at all*, which is as much
    a control as one an hour past its contraction. Dropping those rows would draw the controls
    from the far tail alone and make the comparison one between two distances rather than between
    conditioned and unconditioned."""
    frame = _anchor_frame(guids=4, anchors=100, near_every=5, gap=1.0)
    assert bool(frame[CONTRACTION_AGE_COLUMN].isna().any()), "the fixture must carry NaN ages"

    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)
    controls = split[split["condition"] == "control"]

    assert len(controls) > 0
    assert bool(controls[CONTRACTION_AGE_COLUMN].isna().all())


def test_an_anchor_beyond_the_window_is_a_control_rather_than_an_event() -> None:
    """The window is the whole definition of "conditioned", so it has to be applied rather than
    assumed: every row here has a contraction behind it and only the near ones may count."""
    frame = _anchor_frame(guids=4, anchors=40, near_every=4, gap=1.0)
    frame[CONTRACTION_AGE_COLUMN] = np.where(
        np.asarray(frame["anchor"]) % 4 == 0, 10.0, 600.0
    )

    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    events_only = split[split["condition"] == "event"]
    assert float(events_only[CONTRACTION_AGE_COLUMN].max()) <= 120.0
    assert len(events_only) == 4 * 10


def test_the_split_is_reproducible_from_its_seed_and_moves_with_it() -> None:
    """The control draw is recorded in the summary as a seed, so it has to be recoverable from
    one."""
    frame = _anchor_frame(guids=4, anchors=100, near_every=5, gap=1.0)

    first = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=1)
    again = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=1)
    other = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=2)

    pd.testing.assert_frame_equal(first, again)
    assert not first.equals(other)


# =================================================================================================
# The comparison
# =================================================================================================
def test_a_conditioned_difference_is_recovered_with_its_interval() -> None:
    """The synthetic coupled case: the gap is larger near a contraction by a known amount."""
    frame = _anchor_frame(guids=6, anchors=100, near_every=5, gap=1.0)
    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    rows, per_recording = events_analysis.conditioned_rows(
        split, pd.Series(dtype=object), resamples=200, seed=0
    )

    pooled = next(row for row in rows if row["metric"] == "pred_gap_mc_nats")
    assert pooled["mean"] == pytest.approx(1.0, abs=0.15)
    assert pooled["ci_lo"] < pooled["mean"] < pooled["ci_hi"]
    assert set(per_recording["metric"]) == {"pred_gap_mc_nats", "source_conditioned_kl_raw"}
    # The second readout is constant, so its conditioned difference must come out at exactly zero:
    # a pipeline that reported a difference there would be reporting the control draw.
    flat = next(row for row in rows if row["metric"] == "source_conditioned_kl_raw")
    assert flat["mean"] == pytest.approx(0.0)


def test_a_no_coupling_case_is_indistinguishable_from_its_control() -> None:
    """The other direction, and the one that matters more: a pipeline that manufactured a
    difference out of the control draw would pass the test above and fail this one."""
    frame = _anchor_frame(guids=6, anchors=100, near_every=5, gap=0.0)
    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    rows, _frame = events_analysis.conditioned_rows(
        split, pd.Series(dtype=object), resamples=200, seed=0
    )

    pooled = next(row for row in rows if row["metric"] == "pred_gap_mc_nats")
    assert abs(pooled["mean"]) < 0.05
    assert pooled["ci_lo"] < 0.0 < pooled["ci_hi"]


def test_the_per_class_rows_follow_the_clinical_order_rather_than_the_alphabetical_one() -> None:
    """``groupby`` orders alphabetically, which puts acidosis before hie. Every figure and every
    other table in this evaluation reads HIE, acidosis, healthy -- worst first -- and a CSV that
    did not would be read against them."""
    from teb_vae.lag_attn_cfs.eval._reuse import labels

    frame = _anchor_frame(guids=8, anchors=100, near_every=5, gap=1.0)
    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)
    classes = pd.Series(
        {f"g{index}": ("healthy" if index < 4 else "acidosis") for index in range(8)}
    )

    rows, _per_recording = events_analysis.conditioned_rows(
        split, classes, resamples=64, seed=0
    )

    cohorts = [
        row["cohort"] for row in rows if row["metric"] == "pred_gap_mc_nats"
    ]
    assert cohorts[0] == "pooled"
    assert cohorts[1:] == list(
        __import__("teb_vae.lag_attn_cfs.eval.cohort", fromlist=["cohort"]).ordered_groups(
            ["acidosis", "healthy"], labels.CLASS_COLUMN
        )
    )
    assert cohorts[1:] == ["acidosis", "healthy"]


# =================================================================================================
# The guards
# =================================================================================================
def test_the_guards_fire_on_a_small_population_and_record_a_skip(tmp_path) -> None:
    """A rate over a handful of anchors from two recordings is a description of those two."""
    frame = _anchor_frame(guids=2, anchors=20, near_every=5, gap=1.0)

    record = events_analysis._conditioned_readout(
        _context(frame).collection, tmp_path, window_s=120.0, resamples=64, seed=0
    )

    assert record["record"]["skipped"] is True
    assert str(events_analysis.MIN_EVENT_ANCHORS) in record["record"]["reason"]
    assert record["rows"] == []


def test_a_table_without_the_timing_column_skips_and_names_the_reason(tmp_path) -> None:
    """The column is written by the collection pass, which is the only pass holding the raw UP
    trace. A table without it is an older run rather than a broken one, and the analysis says so
    instead of raising inside the step wrapper."""
    frame = _anchor_frame(guids=6, anchors=200, near_every=5, gap=1.0).drop(
        columns=[CONTRACTION_AGE_COLUMN]
    )

    outcome = events_analysis.run_events_analysis(
        _context(frame), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    assert outcome["conditioned"]["skipped"] is True
    assert "contraction timing" in outcome["conditioned"]["reason"]


# =================================================================================================
# The analysis end to end, on a constructed table
# =================================================================================================
def test_the_analysis_writes_its_table_its_figure_and_the_onset_convention(tmp_path) -> None:
    """The convention has to travel with the number: the detector's onset is a level crossing of
    the peak's own prominence rather than the first sample of the rise, so a reader comparing this
    window against a clinical one is otherwise comparing two different zeros."""
    frame = _anchor_frame(guids=6, anchors=200, near_every=5, gap=1.0)
    per_sample = pd.DataFrame(
        {"guid": [f"g{index}" for index in range(6)], "clinical_class": ["healthy"] * 6}
    )

    outcome = events_analysis.run_events_analysis(
        _context(frame, per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / events_analysis.ANALYSIS_DIRNAME
    missing = [name for name in outcome["files"] if not (directory / name).is_file()]
    assert missing == []
    assert (directory / events_analysis.CONDITIONED_PER_RECORDING_FILENAME).is_file()

    record = outcome["conditioned"]
    assert record["n_event_anchors"] == record["n_control_anchors"] == 6 * 40
    assert record["n_recordings"] == 6
    assert record["onset_convention"]
    assert outcome["composition"]["n_event_anchors"] == record["n_event_anchors"]


def test_it_declares_a_grouped_frame_so_the_runner_fans_the_cohort_cuts(tmp_path) -> None:
    """The fan-out is deliberately the runner's job; an analysis that had to remember to emit its
    own grouped variants is an analysis added later that will not."""
    frame = _anchor_frame(guids=6, anchors=200, near_every=5, gap=1.0)

    outcome = events_analysis.run_events_analysis(
        _context(frame), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    declared = outcome["grouped_frames"]
    assert len(declared) == 1
    assert declared[0]["directory"] == events_analysis.ANALYSIS_DIRNAME
    assert declared[0]["value_columns"] == ["difference"]
    # The declaration names a CSV **on disk**, which is what makes the runner's fan-out work at
    # all -- and what makes ``--only events --output-dir <a finished run>`` reproduce it.
    assert (tmp_path / declared[0]["path"]).is_file()


def test_a_skipped_readout_declares_no_grouped_frame(tmp_path) -> None:
    """A declaration pointing at a file that was never written would make the runner report a
    missing source rather than a skipped analysis."""
    outcome = events_analysis.run_events_analysis(
        _context(), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    assert "grouped_frames" not in outcome


