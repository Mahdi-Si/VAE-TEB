r"""By-class and by-subgroup variants, written beside the pooled output and never in place of it.

The degenerate cases are the ones the implementation is written for -- a single-cohort split, a
group column that is entirely unlabelled, a cohort holding one recording -- but they are *not*
what this file leads with, because every one of them would pass with the emission unimplemented.
So the happy path is asserted first, by composition and by hand-computed value: two groups of
known size produce exactly $2 \times n_{\mathrm{metrics}}$ rows, a group holding two NaNs among
five values reports $n = 3$ rather than a mean over a population that looks healthy, the figure
carries exactly two violin bodies, and each median is the number a reader would compute by hand.

Then the skips. A single-group frame is a *recorded* skip rather than a one-violin figure: one
violin invites a comparison there is nothing to compare against, and on the healthy-only
pretraining split that is the ordinary case rather than an error. And in every case the pooled
output the analysis already wrote is untouched, so a run over a single-cohort split produces
exactly what it produced before grouped variants existed.

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
from matplotlib.collections import PolyCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.report_seam import emit_grouped_variants, summarise_by_group

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
# The reserved filenames
# =============================================================================
def test_the_reserved_suffixes_are_the_two_the_emitter_actually_writes(
    two_class_frame, tmp_path
) -> None:
    """Non-vacuity for every assertion made against :data:`GROUPED_SUFFIXES` elsewhere: the two
    strings are compared against the files the emitter puts on disk rather than against a
    convention this module would then be the only witness to."""
    emit_grouped_variants(two_class_frame, tmp_path, value_columns=_METRICS)

    written = sorted(path.name for path in tmp_path.glob("*.pdf"))

    assert GROUPED_SUFFIXES == ("_by_clinical_class.pdf", "_by_subgroup.pdf")
    assert written and all(name.endswith(GROUPED_SUFFIXES) for name in written)


def test_no_shipped_analysis_names_a_figure_into_the_reserved_shape() -> None:
    """Across the package rather than per analysis, so an analysis added later is covered by a
    test it did not have to remember to write. A figure named into the reserved family is never
    recorded in the manifest and never documented -- it reads to an operator as one of the violin
    figures it is not."""
    import ast

    from teb_vae.lag_attn_cfs.eval import analyses as analyses_package

    root = Path(analyses_package.__file__).parent
    offending = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.endswith(GROUPED_SUFFIXES):
                    offending.append(f"{path.name}: {node.value}")

    assert offending == [], offending


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


# =============================================================================
# Across the pipeline, on a real run
#
# The fan-out is proved above on a fake analysis, which is what says the *runner* does it. What is
# proved here is that the shipped analyses actually declare a frame -- an analysis that forgot
# would report a pooled number over a mixed cohort with nothing saying so, and no test of the
# mechanism would notice.
# =============================================================================
#: The registered analyses expected to declare a per-recording frame, each with the **file stems**
#: it declares. Written out rather than discovered, so an analysis that stopped declaring one fails
#: here with its own name; the entries arrive as their analyses land. ``coupling`` declares two,
#: because a gap in nats and a KL in nats do not share a scale and so do not share a page.
_PARTICIPATING = {
    "coupling": ("coupling_pred_gap", "coupling_kl"),
    "latent": ("latent_per_recording",),
}


@pytest.mark.slow
def test_every_participating_analysis_emits_both_cuts_on_a_real_run(collected_run) -> None:
    """Both variants, per analysis, on the generated multi-subgroup shards -- which carry three
    clinical classes and four subgroups, so neither axis is a degenerate one."""
    results = collected_run["summary"]["results"]

    for analysis, stems in _PARTICIPATING.items():
        grouped = results[analysis].get("grouped")
        assert grouped, f"{analysis} declared no grouped frame"
        for stem, axis in ((stem, axis) for stem in stems for axis in labels.GROUP_COLUMNS):
            assert stem in grouped, f"{analysis} declared no {stem!r} frame"
            record = grouped[stem][axis]
            assert record["skipped"] is False, f"{analysis}/{axis}: {record.get('reason')}"
            for kind in ("table", "figure"):
                path = collected_run["results_dir"] / record["files"][kind]
                assert path.is_file(), path
            # The unit is the recording: the counts here are per-cohort recording counts, and
            # they must sum to the run's own recording count rather than to its segment count.
            assert sum(record["n_per_group"].values()) <= results["n_recordings"]


@pytest.mark.slow
def test_the_grouped_tables_are_summaries_of_per_recording_values(collected_run) -> None:
    """One row per (cohort, metric), with ``n`` counting recordings -- not the long-form frame."""
    from teb_vae.lag_attn_cfs.eval.analyses import coupling as coupling_analysis

    grouped = collected_run["summary"]["results"]["coupling"]["grouped"]
    # Both fan-outs, because between them they must cover every metric the analysis resolves by
    # cohort: a split that dropped one would leave the other's table looking perfectly correct.
    for stem, metrics in (
        (coupling_analysis.GROUPED_PRED_GAP_STEM, coupling_analysis.GROUPED_PRED_GAP_METRICS),
        (coupling_analysis.GROUPED_KL_STEM, coupling_analysis.GROUPED_KL_METRICS),
    ):
        record = grouped[stem]
        table = pd.read_csv(
            collected_run["results_dir"] / record[labels.CLASS_COLUMN]["files"]["table"]
        )

        groups = set(record[labels.CLASS_COLUMN]["groups"])
        assert list(table.columns) == ["group", "metric", "n", "mean", "q25", "median", "q75"]
        assert len(table) == len(groups) * len(metrics)
        assert set(table["metric"]) == set(metrics)
        assert int(table["n"].sum()) <= collected_run["summary"]["results"][
            "n_recordings"
        ] * len(metrics)
