r"""By-class and by-subgroup variants, written beside the pooled output and never in place of it.

The degenerate cases are the ones the implementation is written for -- a single-cohort split, a
group column that is entirely unlabelled -- but they are *not* what this file leads with, because
every one of them would pass with the emission unimplemented. So the happy path is asserted
first, by composition and by hand-computed value: two groups of known size produce exactly
$2 \times n_{\mathrm{metrics}}$ rows, a group holding two NaNs among five values reports $n = 3$
rather than a mean over a population that looks healthy, the figure carries exactly two violin
bodies, and each median is the number a reader would compute by hand.

Then the two skips. A single-group frame is a *recorded* skip rather than a one-violin figure:
one violin invites a comparison there is nothing to compare against, and on the healthy-only
pretraining split that is the ordinary case rather than an error. And in both cases the pooled
output the analysis already wrote is untouched, so a run over a single-cohort split produces
exactly what it produced before grouped variants existed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.report_seam import emit_grouped_variants, summarise_by_group

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
# The happy path
# =============================================================================
def test_the_summary_has_one_row_per_group_and_metric(two_class_frame) -> None:
    summary = summarise_by_group(two_class_frame, labels.CLASS_COLUMN, _METRICS)

    assert len(summary) == 2 * len(_METRICS)
    assert set(summary["group"]) == {"healthy", "acidosis"}
    assert set(summary["metric"]) == set(_METRICS)
    assert list(summary.columns) == ["group", "metric", "n", "mean", "q25", "median", "q75"]


def test_n_counts_finite_values_only(two_class_frame) -> None:
    """A group of NaNs must report ``n = 0``, not a mean of NaN over a healthy-looking count."""
    summary = summarise_by_group(two_class_frame, labels.CLASS_COLUMN, _METRICS)
    healthy_gap = summary[
        (summary["group"] == "healthy") & (summary["metric"] == "pred_gap")
    ].iloc[0]

    assert int(healthy_gap["n"]) == 3
    assert float(healthy_gap["mean"]) == pytest.approx(2.0)


def test_each_quartile_matches_the_hand_computed_value(two_class_frame) -> None:
    summary = summarise_by_group(two_class_frame, labels.CLASS_COLUMN, _METRICS)
    acidosis_gap = summary[
        (summary["group"] == "acidosis") & (summary["metric"] == "pred_gap")
    ].iloc[0]

    # [10, 20, 30, 40, 50]: linear interpolation puts the quartiles on the samples themselves.
    assert float(acidosis_gap["median"]) == pytest.approx(30.0)
    assert float(acidosis_gap["q25"]) == pytest.approx(20.0)
    assert float(acidosis_gap["q75"]) == pytest.approx(40.0)


def test_both_grouping_axes_emit_a_table_and_a_figure(two_class_frame, tmp_path) -> None:
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


def test_the_figure_carries_one_violin_body_per_group(two_class_frame) -> None:
    """Read off the in-memory figure rather than the PDF: what reaches the page is what an
    operator compares, and a violin silently missing is exactly the failure a file-size check
    would pass."""
    groups = ["healthy", "acidosis"]
    values = {
        metric: {
            group: np.asarray(
                two_class_frame.loc[two_class_frame[labels.CLASS_COLUMN] == group, metric],
                dtype=np.float64,
            )
            for group in groups
        }
        for metric in _METRICS
    }

    figure, axes = shared_figures.grouped_violin_figure(values, groups)
    try:
        per_row = [
            len([a for a in axes[row, 0].collections if isinstance(a, PolyCollection)])
            for row in range(len(_METRICS))
        ]
    finally:
        shared_figures.plt.close(figure)

    assert per_row == [2, 2]


# =============================================================================
# The recorded skips
# =============================================================================
def test_a_single_cohort_split_records_a_skip_and_writes_no_figure(tmp_path) -> None:
    """The ordinary case on the healthy-only pretraining split, and not an error."""
    frame = pd.DataFrame(
        {
            labels.CLASS_COLUMN: ["healthy"] * 4,
            labels.SUBGROUP_COLUMN: ["healthy_no_bg_no_cs"] * 4,
            "pred_gap": [1.0, 2.0, 3.0, 4.0],
        }
    )

    emitted = emit_grouped_variants(frame, tmp_path, value_columns=["pred_gap"])

    for axis in labels.GROUP_COLUMNS:
        assert emitted[axis]["skipped"] is True
        assert "nothing to compare" in emitted[axis]["reason"]
    assert list(tmp_path.iterdir()) == []


def test_an_unlabelled_group_column_records_a_skip_rather_than_raising(tmp_path) -> None:
    """``None`` is not a cohort. Folding the unlabelled samples together would create one named
    after the absence, and every by-class number would then include it."""
    frame = pd.DataFrame(
        {
            labels.CLASS_COLUMN: [None, None, None],
            labels.SUBGROUP_COLUMN: [None, None, None],
            "pred_gap": [1.0, 2.0, 3.0],
        }
    )

    emitted = emit_grouped_variants(frame, tmp_path, value_columns=["pred_gap"])

    assert all(emitted[axis]["skipped"] is True for axis in labels.GROUP_COLUMNS)
    assert list(tmp_path.iterdir()) == []


def test_a_metric_absent_from_the_frame_is_skipped_rather_than_raising(
    two_class_frame, tmp_path
) -> None:
    """An analysis may name a metric it only sometimes produces; a grouped variant is an addition
    to a run and must not mark a successful analysis failed."""
    emitted = emit_grouped_variants(
        two_class_frame, tmp_path, value_columns=["pred_gap", "not_a_column"]
    )

    table = pd.read_csv(tmp_path / f"per_sample_by_{labels.CLASS_COLUMN}.csv")
    assert emitted[labels.CLASS_COLUMN]["skipped"] is False
    assert set(table["metric"]) == {"pred_gap"}


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
