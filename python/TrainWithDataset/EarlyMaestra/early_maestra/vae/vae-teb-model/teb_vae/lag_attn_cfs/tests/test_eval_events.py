r"""Contraction-conditioned coupling, and the two readouts this cell does not have.

The sibling's ``events`` analysis reports three things. Two of them score a bpm waveform -- a
deceleration detector run over each branch's forecast block, and that block averaged around a
contraction onset -- and this model's forecast block is $15 \times 98$ wavelet-modulus and
phase-harmonic coefficients in the loader's $z$ units. So they are absent, and the first section
below asserts the absence three ways: the symbols are gone, the record names them with a reason,
and the analysis reads no retained waveform at all.

What survives conditions on **timing** rather than on the forecast's shape, and it ports unchanged.
What can go wrong in it is not the arithmetic but the comparison: the control anchors have to be
drawn *within each recording* and matched to that recording's event count, or the difference is a
difference between recordings wearing two labels. Every assertion here therefore constructs the
condition it needs -- a synthetic anchor table whose near-contraction rows carry a known extra gap
-- rather than hoping the fixture supplies one, because the generated cohort is a fixture about
counts and shapes and no finding about coupling may be read off it.

The detector itself is tested in ``test_eval_events_detector.py``; this file starts where the
contraction timing has already reached the per-anchor table.
"""
from __future__ import annotations

import ast
import json
import types
from pathlib import Path
from typing import Any, Dict, Optional

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


def _code_names() -> set:
    """Return every name the analysis module reaches for **in code**, docstrings excluded.

    Attribute names and string literals together, because the two ways of reading a field --
    ``collection.retained`` and ``getattr(collection, "retained")`` -- look nothing alike in an
    AST. Docstrings are stripped because the module names both absent readouts in prose, which is
    the opposite of reaching for them.
    """
    tree = ast.parse(Path(events_analysis.__file__).read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    return names


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
# The two readouts that are not here
# =================================================================================================
def test_no_symbol_of_the_two_removed_readouts_survives_the_reduction() -> None:
    """A mechanical copy would have carried a deceleration detector over a coefficient block, and
    a "triggered response" averaging one. Both are arithmetic that runs, produces finite numbers,
    and means nothing -- which is exactly the failure a scan of the module's own names catches and
    a green test suite does not."""
    tree = ast.parse(Path(events_analysis.__file__).read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    } | {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    banned = ("deceleration", "triggered", "trigger", "skill_rows", "null_draws")
    offending = sorted(
        name for name in defined | assigned
        if any(fragment in name.lower() for fragment in banned)
    )

    assert offending == []
    # And the readout that stays is genuinely there, so this is not an assertion about an empty
    # module.
    assert "run_events_analysis" in defined
    assert "conditioned_anchors" in defined


def test_the_removed_readouts_are_named_in_the_record_with_a_reason(tmp_path) -> None:
    """A reader meets ``events`` in ``summary.json`` expecting the sibling's three. A key that is
    simply missing reads as a step that failed; the absence has to be stated."""
    outcome = events_analysis.run_events_analysis(
        _context(), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    removed = {entry["readout"] for entry in outcome["removed_readouts"]}
    assert removed == {"deceleration_skill", "contraction_triggered_response"}
    for entry in outcome["removed_readouts"]:
        assert "bpm" in entry["reason"] or "coefficient" in entry["reason"]


def test_the_analysis_reads_no_retained_waveform_and_says_it_is_uncapped(tmp_path) -> None:
    """The structural difference from the sibling's, and the reason this readout runs over every
    anchor of the split: the contraction timing is already on the per-anchor table, so no forecast
    block is needed and ``caps.waveforms`` does not reach here. An analysis that read ``retained``
    would silently report over the retained subsample instead."""
    names = _code_names()

    assert "retained" not in names
    assert "caps" not in names
    # Non-vacuity: the two fields it *does* read are found by the same scan.
    assert {"per_anchor", "per_sample"} <= names

    outcome = events_analysis.run_events_analysis(
        _context(), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )
    assert outcome["plan"]["capped"] is False


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
    """``groupby`` orders alphabetically, which puts acidosis before healthy. Every figure and
    every other table in this evaluation reads healthy, acidosis, HIE, and a CSV that did not
    would be read against them."""
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
    assert cohorts[1:] == ["healthy", "acidosis"]


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


# =================================================================================================
# Against the real run
# =================================================================================================
@pytest.mark.slow
def test_the_analysis_reached_the_summary_and_recorded_its_outcome(collected_run) -> None:
    """The generated cohort is eight real segments reused under distinct identities, so whether
    the guards clear is a property of the fixture rather than a finding. What must hold either way
    is that the step ran, wrote its table, and said which of the two it did."""
    block = collected_run["summary"]["results"].get("events")
    assert block is not None, "the run produced no events block at all"

    directory = Path(collected_run["results_dir"]) / events_analysis.ANALYSIS_DIRNAME
    missing = [name for name in block["files"] if not (directory / name).is_file()]
    assert missing == []

    conditioned = block["conditioned"]
    assert ("skipped" in conditioned) or ("n_event_anchors" in conditioned)
    assert {entry["readout"] for entry in block["removed_readouts"]} == {
        "deceleration_skill", "contraction_triggered_response"
    }


@pytest.mark.slow
def test_the_step_record_marks_the_analysis_as_having_succeeded(collected_run) -> None:
    """A guard that fires is a recorded skip, not a failed step: the exit code is non-zero if and
    only if a step raised."""
    steps = json.loads(
        (Path(collected_run["results_dir"]) / "steps.json").read_text(encoding="utf-8")
    )
    record = next(step for step in steps if step["name"] == "events")

    assert record["ok"] is True, record.get("error")
