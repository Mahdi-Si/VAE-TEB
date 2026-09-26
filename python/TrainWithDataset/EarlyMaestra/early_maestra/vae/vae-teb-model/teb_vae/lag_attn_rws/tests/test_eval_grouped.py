r"""By-class and by-subgroup variants, fanned out by the runner beside the pooled output.

The summary arithmetic and the skip rules are the shared emitter's and are tested by its owner;
this package's cohort order and palette are pinned in ``test_eval_cohort_presentation.py``. What
is checked here is what this package adds on top:

* **the quartiles** of the long-form summary, against hand-computed values;
* **the fan-out is the runner's**: an analysis only *declares* a per-sample frame, and the
  runner emits both cuts beside it, records a single-cohort population as a skip, invents nothing
  for an analysis that declared nothing, survives an unreadable declaration, and records every
  path relative to the run directory;
* **on a real run**, every participating analysis declares its frame and both cuts are emitted
  with per-recording counts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.report_seam import summarise_by_group

#: The metrics a grouped variant is asked for here. Two, so a row count of ``2 x n_metrics``
#: cannot coincide with a row count of ``2 x n_groups``.
_METRICS = ["pred_gap", "source_conditioned_kl_raw"]


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
# The summary quartiles
# =============================================================================
def test_each_quartile_matches_the_hand_computed_value(two_class_frame) -> None:
    summary = summarise_by_group(two_class_frame, labels.CLASS_COLUMN, _METRICS)
    acidosis_gap = summary[
        (summary["group"] == "acidosis") & (summary["metric"] == "pred_gap")
    ].iloc[0]

    # [10, 20, 30, 40, 50]: linear interpolation puts the quartiles on the samples themselves.
    assert float(acidosis_gap["median"]) == pytest.approx(30.0)
    assert float(acidosis_gap["q25"]) == pytest.approx(20.0)
    assert float(acidosis_gap["q75"]) == pytest.approx(40.0)


# =============================================================================
# The fan-out is the runner's, not the analysis's
#
# An analysis *declares* a per-sample CSV and the columns worth resolving by group; the runner
# reads it and emits both variants. Written per analysis instead, this would be a cross-cutting
# change every analysis added later has to remember to make, and the one that forgets reports a
# pooled number over a mixed cohort with nothing saying so.
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
    from teb_vae.lag_attn_rws.eval import run as run_module
    from teb_vae.lag_attn_rws.eval.report_seam import Report

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
    from teb_vae.lag_attn_rws.eval.frames import grouped_frame_entry

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


# =============================================================================
# Across the pipeline, on a real run
#
# The fan-out is proved above on a fake analysis, which is what says the *runner* does it. What is
# proved here is that the shipped analyses actually declare a frame -- an analysis that forgot
# would report a pooled number over a mixed cohort with nothing saying so, and no test of the
# mechanism would notice.
# =============================================================================
#: The analyses expected to declare a per-recording frame, each with the file it declares.
_PARTICIPATING = {
    "forecast": ("forecast_scores",),
    # Two stems, because a gap in nats and a KL in nats do not share a scale and so do not share
    # a page.
    "coupling": ("coupling_pred_gap", "coupling_kl"),
    "perm_control": ("perm_control_per_recording",),
    "latent": ("latent_per_recording",),
    "lag_kl": ("lag_kl_per_recording",),
    "attention": ("attention_per_recording",),
    "residual": ("residual_per_recording",),
}


def test_every_participating_analysis_emits_both_cuts_on_a_real_run(evaluated) -> None:
    """Both variants, per analysis, on the generated multi-subgroup shards -- which carry three
    clinical classes and four subgroups, so neither axis is a degenerate one."""
    results = evaluated["summary"]["results"]

    for analysis, stems in _PARTICIPATING.items():
        grouped = results[analysis].get("grouped")
        assert grouped, f"{analysis} declared no grouped frame"
        for stem, axis in ((stem, axis) for stem in stems for axis in labels.GROUP_COLUMNS):
            assert stem in grouped, f"{analysis} declared no {stem!r} frame"
            record = grouped[stem][axis]
            assert record["skipped"] is False, f"{analysis}/{axis}: {record.get('reason')}"
            for kind in ("table", "figure"):
                path = evaluated["results_dir"] / record["files"][kind]
                assert path.is_file(), path
            # The unit is the recording: the counts here are per-cohort recording counts, and
            # they must sum to the run's own recording count rather than to its segment count.
            assert sum(record["n_per_group"].values()) <= results["n_recordings"]


def test_the_grouped_tables_are_summaries_of_per_recording_values(evaluated) -> None:
    """One row per (cohort, metric), with ``n`` counting recordings -- not the long-form frame."""
    from teb_vae.lag_attn_rws.eval.analyses import coupling as coupling_analysis

    grouped = evaluated["summary"]["results"]["coupling"]["grouped"]
    # Both fan-outs, because between them they must cover every metric the analysis resolves by
    # cohort: a split that dropped one would leave the other's table looking perfectly correct.
    for stem, metrics in (
        (coupling_analysis.GROUPED_PRED_GAP_STEM, coupling_analysis.GROUPED_PRED_GAP_METRICS),
        (coupling_analysis.GROUPED_KL_STEM, coupling_analysis.GROUPED_KL_METRICS),
    ):
        record = grouped[stem]
        table = pd.read_csv(
            evaluated["results_dir"] / record[labels.CLASS_COLUMN]["files"]["table"]
        )

        groups = set(record[labels.CLASS_COLUMN]["groups"])
        assert list(table.columns) == ["group", "metric", "n", "mean", "q25", "median", "q75"]
        assert len(table) == len(groups) * len(metrics)
        assert set(table["metric"]) == set(metrics)
        assert int(table["n"].sum()) <= evaluated["summary"]["results"]["n_recordings"] * len(
            metrics
        )
