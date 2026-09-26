r"""By-class and by-subgroup variants, written beside the pooled output and never in place of it.

The summary arithmetic and the emitter's own skip rules are the shared layer's and are tested
there. What is checked here is this package's wrapper around the emitter -- both axes emitted, a
cohort of one reported with its $n$ rather than dropped -- and the runner's fan-out over a frame an
analysis only *declares*. A single-group frame is a *recorded* skip rather than a one-violin
figure, and in every case the pooled output the analysis already wrote is untouched.

**The empty-frame path is a case here rather than an assumption**, because it is the one that
breaks something downstream rather than here: an empty CSV allowed through the fan-out is what
``cross_subgroup`` later reads and crashes on, and the crash arrives one analysis away from its
cause.

This file also owns :data:`GROUPED_SUFFIXES` -- the two filename endings the fan-out reserves --
derived from the group columns rather than written out, so an analysis of its own that named a
figure into that shape is caught by the analysis's own test rather than by a document.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.report_seam import emit_grouped_variants

#: The metrics a grouped variant is asked for here. Two, so a row count of ``2 x n_metrics``
#: cannot coincide with a row count of ``2 x n_groups``.
_METRICS = ["pred_gap", "source_conditioned_kl_raw"]

#: The filenames the runner's fan-out reserves, ``<stem>_by_<axis>.pdf`` for each grouping axis.
#: Derived from :data:`~teb_vae.lag_attn.eval.labels.GROUP_COLUMNS` rather than written out,
#: because the emitter builds them the same way -- a hand-kept copy would be a second definition
#: of a reserved name, which is exactly the class of mistake the reservation exists to prevent.
#:
#: An analysis that names a figure into this shape does not collide with anything; it *vanishes*,
#: because the smoke test normalises the family out of the figure manifest. So the shape is
#: published here and asserted against by the analyses that draw per-cohort figures of their own.
GROUPED_SUFFIXES = tuple(f"_by_{column}.pdf" for column in labels.GROUP_COLUMNS)


@pytest.fixture
def two_class_frame() -> pd.DataFrame:
    """Five samples per class, with two of one class's ``pred_gap`` values seeded NaN.

    The NaNs are load-bearing: ``n`` counts finite values, and a fully valid frame would leave
    that property untested while every other assertion still passed.
    """
    return pd.DataFrame(
        {
            labels.CLASS_COLUMN: ["healthy"] * 5 + ["acidosis"] * 5,
            labels.SUBGROUP_COLUMN: ["healthy_no_bg_no_cs"] * 5 + ["acidosis_cs"] * 5,
            "pred_gap": [1.0, 2.0, 3.0, np.nan, np.nan] + [10.0, 20.0, 30.0, 40.0, 50.0],
            "source_conditioned_kl_raw": [0.5] * 5 + [1.5] * 5,
        }
    )


# =============================================================================
# The happy path
# =============================================================================
def test_both_grouping_axes_emit_a_table_and_a_figure(two_class_frame, tmp_path) -> None:
    """Also the non-vacuity for every assertion made against :data:`GROUPED_SUFFIXES` elsewhere:
    the reserved endings are compared against the files the emitter actually puts on disk."""
    emitted = emit_grouped_variants(two_class_frame, tmp_path, value_columns=_METRICS)

    assert sorted(emitted) == sorted(labels.GROUP_COLUMNS)
    for axis in labels.GROUP_COLUMNS:
        record = emitted[axis]
        assert record["skipped"] is False
        table = tmp_path / f"per_sample_by_{axis}.csv"
        figure = tmp_path / f"per_sample_by_{axis}.pdf"
        assert table.is_file() and figure.is_file() and figure.stat().st_size > 0
        assert len(pd.read_csv(table)) == 2 * len(_METRICS)
        assert record["n_per_group"] == {group: 5 for group in record["groups"]}
    written = [path.name for path in tmp_path.glob("*.pdf")]
    assert written and all(name.endswith(GROUPED_SUFFIXES) for name in written)


def test_a_cohort_with_one_recording_produces_a_row_with_its_n_visible(tmp_path) -> None:
    """Dropped rather than reported is the failure this prevents: a cohort of one is a real cohort
    whose evidence is thin, and the honest rendering is a row saying $n = 1$. A pipeline that
    silently omitted it would report a two-cohort comparison over a three-cohort split."""
    frame = pd.DataFrame(
        {
            labels.CLASS_COLUMN: ["healthy", "healthy", "healthy", "hie"],
            labels.SUBGROUP_COLUMN: ["healthy_bg_cs"] * 3 + ["hie_cs"],
            "pred_gap": [1.0, 2.0, 3.0, 9.0],
        }
    )

    emitted = emit_grouped_variants(frame, tmp_path, value_columns=["pred_gap"])
    table = pd.read_csv(tmp_path / f"per_sample_by_{labels.CLASS_COLUMN}.csv")

    assert emitted[labels.CLASS_COLUMN]["skipped"] is False
    assert emitted[labels.CLASS_COLUMN]["n_per_group"] == {"healthy": 3, "hie": 1}
    lonely = table[table["group"] == "hie"].iloc[0]
    assert int(lonely["n"]) == 1
    assert float(lonely["mean"]) == pytest.approx(9.0)


# =============================================================================
# The fan-out is the runner's, not the analysis's
#
# An analysis *declares* a per-sample CSV and the columns worth resolving by group; the runner
# reads it and emits both variants. Written per analysis instead, this would be a cross-cutting
# change every analysis added later has to remember to make, and the one that forgets reports a
# pooled number over a mixed cohort with nothing saying so.
#
# The companion assertion lives in ``test_eval_protocol.py``: no module under ``eval/analyses/``
# so much as mentions the grouped emitter. Together the two say the fan-out happens *and* that no
# analysis is the thing making it happen.
# =============================================================================
def _declaring_analysis(frame: pd.DataFrame, value_columns):
    """Build a fake analysis that writes ``frame`` and declares it for grouping."""

    def _run(context, *, eval_config, output_dir, probe):
        path = Path(output_dir) / "fake_per_sample.csv"
        frame.to_csv(path, index=False)
        return {
            "n_samples": int(len(frame)),
            "composition": {},
            "plan": {"capped": False},
            "grouped_frames": [
                {"path": str(path), "value_columns": list(value_columns), "stem": "fake"}
            ],
        }

    return _run


def _run_one(analysis, output_dir):
    """Run one analysis through the runner's loop and return its recorded result."""
    from teb_vae.lag_attn_cfs.eval import run as run_module
    from teb_vae.lag_attn_cfs.eval.report_seam import Report

    report = Report()
    registry = {"fake": analysis}
    run_module.run_analyses(
        report, list(registry), registry,
        context=None, eval_config={}, output_dir=output_dir, probe=None,
    )
    assert report.exit_code() == 0, report.steps[0].traceback
    return report.results["fake"]


def test_the_runner_emits_both_variants_for_an_analysis_that_only_declares_a_frame(
    two_class_frame, tmp_path
) -> None:
    result = _run_one(_declaring_analysis(two_class_frame, _METRICS), tmp_path)

    for axis in labels.GROUP_COLUMNS:
        assert result["grouped"]["fake"][axis]["skipped"] is False
        assert (tmp_path / f"fake_by_{axis}.csv").is_file()
        assert (tmp_path / f"fake_by_{axis}.pdf").is_file()
    # The pooled frame the analysis itself wrote is untouched beside them.
    assert len(pd.read_csv(tmp_path / "fake_per_sample.csv")) == len(two_class_frame)


def test_a_single_cohort_population_records_the_skip_and_leaves_the_pooled_output(
    tmp_path
) -> None:
    """The ordinary case on the healthy-only pretraining split, and not an error."""
    frame = pd.DataFrame(
        {
            labels.CLASS_COLUMN: ["healthy"] * 3,
            labels.SUBGROUP_COLUMN: ["healthy_no_bg_no_cs"] * 3,
            "pred_gap": [1.0, 2.0, 3.0],
        }
    )

    result = _run_one(_declaring_analysis(frame, ["pred_gap"]), tmp_path)

    for axis in labels.GROUP_COLUMNS:
        assert result["grouped"]["fake"][axis]["skipped"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "fake_per_sample.csv", "steps.json",
    ]


def test_an_empty_declared_frame_is_a_recorded_skip_rather_than_a_crash(tmp_path) -> None:
    """The degenerate case with a *downstream* victim, which is why it is a case here rather than
    an assumption. An empty CSV that reached the fan-out would be written back out as an empty
    grouped table, and ``cross_subgroup`` reads those tables off disk one analysis later -- so the
    failure would surface with neither the analysis that produced it nor the cohort it came from
    anywhere in the traceback."""
    empty = pd.DataFrame(
        {labels.CLASS_COLUMN: [], labels.SUBGROUP_COLUMN: [], "pred_gap": []}
    )

    result = _run_one(_declaring_analysis(empty, ["pred_gap"]), tmp_path)

    for axis in labels.GROUP_COLUMNS:
        assert result["grouped"]["fake"][axis]["skipped"] is True
        assert result["grouped"]["fake"][axis]["reason"]
    assert not list(tmp_path.glob("fake_by_*"))


def test_an_analysis_declaring_nothing_gets_no_grouped_record(tmp_path) -> None:
    """Most analyses have no per-sample frame; the fan-out must not invent one for them."""

    def _run(context, *, eval_config, output_dir, probe):
        return {"n_samples": 0, "composition": {}, "plan": {"capped": False}}

    assert "grouped" not in _run_one(_run, tmp_path)


def test_an_unreadable_declared_frame_does_not_fail_the_analysis(tmp_path) -> None:
    """A grouped variant is an addition to a run: an analysis whose pooled output succeeded must
    not be marked failed because the variant could not be drawn."""

    def _run(context, *, eval_config, output_dir, probe):
        return {
            "n_samples": 1,
            "composition": {},
            "plan": {"capped": False},
            "grouped_frames": [
                {"path": str(Path(output_dir) / "absent.csv"), "value_columns": ["pred_gap"]}
            ],
        }

    result = _run_one(_run, tmp_path)

    assert result["grouped"]["absent"]["skipped"] is True


def test_a_relative_declaration_resolves_against_the_results_directory(
    two_class_frame, tmp_path
) -> None:
    """The form the shipped analyses declare, and the reason: an absolute path in ``summary.json``
    is a machine-specific string in a block two runs of one checkpoint must compare **equal**, and
    it stops resolving the moment the run directory is copied anywhere."""
    from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry

    def _run(context, *, eval_config, output_dir, probe):
        directory = Path(output_dir) / "fake_analysis"
        directory.mkdir(parents=True, exist_ok=True)
        two_class_frame.to_csv(directory / "fake_per_recording.csv", index=False)
        return {
            "n_samples": int(len(two_class_frame)),
            "composition": {},
            "plan": {"capped": False},
            "grouped_frames": [
                grouped_frame_entry("fake_analysis", "fake_per_recording.csv", _METRICS)
            ],
        }

    result = _run_one(_run, tmp_path)

    for axis in labels.GROUP_COLUMNS:
        record = result["grouped"]["fake_per_recording"][axis]
        assert record["skipped"] is False
        # Written where the frame lives, and recorded relative to the run directory.
        assert (tmp_path / "fake_analysis" / f"fake_per_recording_by_{axis}.csv").is_file()
        assert record["files"]["table"] == f"fake_analysis/fake_per_recording_by_{axis}.csv"
        assert not Path(record["files"]["figure"]).is_absolute()
    assert not Path(result["grouped_frames"][0]["path"]).is_absolute()
