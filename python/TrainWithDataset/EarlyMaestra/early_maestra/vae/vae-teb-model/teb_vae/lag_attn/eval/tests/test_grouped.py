r"""Tests for the shared grouped emission helper and its adoption by the analyses.

The helper's whole value is that the *policy* lives in one place, so these tests are about the
policy rather than about any one analysis: what happens when there is one group, when there are
none, when the column is missing, and when a requested metric is not on the frame. Every one of
those is the ordinary case on some real split, and none of them may raise -- a grouped variant is
an addition to a run, and an analysis whose pooled output succeeded must not be marked failed
because its split turned out to hold one cohort.

The adoption is then checked twice over, and it needs both halves. On the committed
``tiny_shard.hdf5`` -- one file, ``target`` all zeros -- every analysis must record a clean skip;
that is the branch a repository-wide run exercises. On the generated multi-class shards the
grouped path actually runs, and without those shards *only* the skip branch would ever be tested,
which is exactly the vacuous coverage the fixture exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval import figures, labels, report
from teb_vae.lag_attn.eval.analyses import forecast as forecast_analysis
from teb_vae.lag_attn.eval.analyses import uplift as uplift_analysis


def _frame(groups, values=None, subgroups=None) -> pd.DataFrame:
    """A minimal per-sample frame carrying the two group columns."""
    size = len(groups)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sample_index": np.arange(size),
        "guid": [f"g{index}" for index in range(size)],
        "source_file": ["s.hdf5"] * size,
        labels.CLASS_COLUMN: list(groups),
        labels.SUBGROUP_COLUMN: list(subgroups if subgroups is not None else groups),
        "feat_mse_total": rng.uniform(0.5, 2.0, size) if values is None else np.asarray(values),
        "feat_r2_total": rng.uniform(-1.0, 0.5, size),
    })


# ---------------------------------------------------------------------------
# The aggregate
# ---------------------------------------------------------------------------
def test_the_summary_has_one_row_per_group_and_metric() -> None:
    """Long form: it merges across runs with no renaming and does not reshape on a new metric."""
    frame = _frame(["healthy", "healthy", "acidosis", "acidosis"])
    summary = report.summarise_by_group(
        frame, labels.CLASS_COLUMN, ["feat_mse_total", "feat_r2_total"]
    )
    assert len(summary) == 4
    assert set(summary["group"]) == {"healthy", "acidosis"}
    assert set(summary["metric"]) == {"feat_mse_total", "feat_r2_total"}


def test_the_group_statistics_match_a_direct_computation() -> None:
    """A table of plausible-but-wrong numbers would pass a shape assertion."""
    frame = _frame(["a", "a", "b", "b"], values=[1.0, 3.0, 10.0, 20.0])
    summary = report.summarise_by_group(frame, labels.CLASS_COLUMN, ["feat_mse_total"])
    rows = summary.set_index("group")

    assert rows.loc["a", "mean"] == pytest.approx(2.0)
    assert rows.loc["a", "median"] == pytest.approx(2.0)
    assert rows.loc["b", "mean"] == pytest.approx(15.0)
    assert rows.loc["b", "n"] == 2


def test_non_finite_values_are_excluded_from_the_count_as_well_as_the_mean() -> None:
    """A group of NaNs must report n = 0, not a NaN mean over a population that looks healthy."""
    frame = _frame(["a", "a", "b", "b"], values=[np.nan, np.nan, 4.0, 6.0])
    rows = report.summarise_by_group(
        frame, labels.CLASS_COLUMN, ["feat_mse_total"]
    ).set_index("group")

    assert rows.loc["a", "n"] == 0
    assert np.isnan(rows.loc["a", "mean"])
    assert rows.loc["b", "n"] == 2 and rows.loc["b", "mean"] == pytest.approx(5.0)


def test_unlabelled_samples_form_no_group_of_their_own() -> None:
    """Folding them together would create a cohort named after the absence of a label."""
    frame = _frame(["healthy", None, "acidosis", None])
    summary = report.summarise_by_group(frame, labels.CLASS_COLUMN, ["feat_mse_total"])
    assert set(summary["group"]) == {"healthy", "acidosis"}


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
def test_both_axes_are_emitted_from_one_helper(tmp_path) -> None:
    """One helper serves the class axis and the subgroup axis; only the column differs."""
    frame = _frame(
        ["healthy", "healthy", "acidosis", "acidosis"],
        subgroups=["healthy_bg_cs", "healthy_no_bg_cs", "acidosis_cs", "acidosis_no_cs"],
    )
    emitted = report.emit_grouped_variants(
        frame, tmp_path, value_columns=["feat_mse_total", "feat_r2_total"]
    )

    assert set(emitted) == set(labels.GROUP_COLUMNS)
    for column in labels.GROUP_COLUMNS:
        assert emitted[column]["skipped"] is False
        assert Path(emitted[column]["files"]["table"]).is_file()
        assert Path(emitted[column]["files"]["figure"]).stat().st_size > 0
    # The two axes must not coincide, or a by-subgroup bug would hide behind a by-class pass.
    assert emitted[labels.CLASS_COLUMN]["groups"] != emitted[labels.SUBGROUP_COLUMN]["groups"]


def test_a_single_group_falls_back_to_the_pooled_output(tmp_path) -> None:
    """One violin invites a comparison there is nothing to compare against."""
    frame = _frame(["healthy"] * 4)
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["feat_mse_total"])

    record = emitted[labels.CLASS_COLUMN]
    assert record["skipped"] is True
    assert "nothing to compare" in record["reason"]
    assert not list(tmp_path.glob("*_by_*")), "a skipped variant must write nothing"


def test_an_all_none_group_column_does_not_raise(tmp_path) -> None:
    """The ordinary case on the healthy-only pretraining split."""
    frame = _frame([None, None, None, None])
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["feat_mse_total"])

    assert emitted[labels.CLASS_COLUMN]["skipped"] is True
    assert emitted[labels.CLASS_COLUMN]["groups"] == []


def test_a_missing_group_column_is_recorded_rather_than_raising(tmp_path) -> None:
    frame = _frame(["a", "b"]).drop(columns=[labels.SUBGROUP_COLUMN])
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["feat_mse_total"])

    assert emitted[labels.SUBGROUP_COLUMN]["skipped"] is True
    assert "carries no" in emitted[labels.SUBGROUP_COLUMN]["reason"]
    assert emitted[labels.CLASS_COLUMN]["skipped"] is False


def test_a_metric_absent_from_the_frame_is_skipped_not_raised(tmp_path) -> None:
    """An analysis may name a metric it only sometimes produces."""
    frame = _frame(["a", "a", "b", "b"])
    emitted = report.emit_grouped_variants(
        frame, tmp_path, value_columns=["feat_mse_total", "not_a_column"]
    )
    table = pd.read_csv(emitted[labels.CLASS_COLUMN]["files"]["table"])
    assert set(table["metric"]) == {"feat_mse_total"}


def test_no_requested_metric_at_all_is_a_skip(tmp_path) -> None:
    frame = _frame(["a", "a", "b", "b"])
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["nothing_here"])
    assert emitted[labels.CLASS_COLUMN]["skipped"] is True
    assert "none of the requested metrics" in emitted[labels.CLASS_COLUMN]["reason"]


def test_the_per_group_counts_are_recorded(tmp_path) -> None:
    """So a variant drawn from three samples is not read as though it were drawn from three
    hundred."""
    frame = _frame(["a", "a", "a", "b"])
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["feat_mse_total"])
    assert emitted[labels.CLASS_COLUMN]["n_per_group"] == {"a": 3, "b": 1}


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def test_the_grouped_figure_has_one_panel_per_metric_and_one_violin_per_group() -> None:
    rng = np.random.default_rng(1)
    values = {
        "mse": {"healthy": rng.uniform(0, 1, 20), "acidosis": rng.uniform(2, 3, 20)},
        "r2": {"healthy": rng.uniform(0, 1, 20), "acidosis": rng.uniform(-1, 0, 20)},
    }
    figure, axes = figures.grouped_violin_figure(
        values, ["healthy", "acidosis"], title_prefix="by class: ", references={"r2": 0.0}
    )
    try:
        titles = [ax.get_title() for ax in figure.axes if ax.get_title()]
        assert titles == ["by class: mse", "by class: r2"]
        ticks = [label.get_text() for label in axes[0, 0].get_xticklabels()]
        assert ticks == ["healthy", "acidosis"]
        assert all(ax.has_data() for ax in figure.axes if ax.get_title())
    finally:
        figures.plt.close(figure)


def test_the_grouped_figure_reports_the_cohort_that_was_sabotaged() -> None:
    """The non-vacuity check: a figure drawing every group from the same column would pass the
    structural assertions above."""
    rng = np.random.default_rng(2)
    values = {
        "mse": {
            "healthy": rng.uniform(0.0, 0.5, 40),
            "acidosis": rng.uniform(20.0, 21.0, 40),
            "hie": rng.uniform(0.0, 0.5, 40),
        }
    }
    figure, axes = figures.grouped_violin_figure(values, ["healthy", "acidosis", "hie"])
    try:
        centres = [
            float(np.mean(body.get_paths()[0].vertices[:, 1]))
            for body in axes[0, 0].collections
            if hasattr(body, "get_paths") and body.get_paths()
        ]
        assert int(np.argmax(centres[:3])) == 1
    finally:
        figures.plt.close(figure)


def test_the_known_clinical_classes_keep_their_shared_colours() -> None:
    """So an eval figure and a training figure of the same cohort are the same colour."""
    colors = figures.group_colors(["healthy", "acidosis", "hie"])
    assert colors["healthy"] == figures.CLASS_COLORS_DEFAULT["healthy"]
    assert colors["acidosis"] == figures.CLASS_COLORS_DEFAULT["acidosis"]


def test_an_unknown_group_still_gets_a_distinct_colour() -> None:
    """The subgroup axis's labels are not in the class table, and must still be distinguishable."""
    colors = figures.group_colors(["healthy_bg_cs", "acidosis_cs", "hie_no_cs"])
    assert len(set(colors.values())) == 3


# ---------------------------------------------------------------------------
# Adoption
# ---------------------------------------------------------------------------
def test_the_collector_attaches_both_group_columns(multi_class_loader, make_eval_runner, tmp_path):
    """Attached once in the collector, so every analysis gets them without its own code."""
    from teb_vae.lag_attn.eval.collectors import collect_metrics

    runner = make_eval_runner(output_dir=tmp_path / "runner")
    collected = collect_metrics(
        runner, multi_class_loader, lambda _runner, batch: {}, progress_label="test"
    )
    for column in labels.GROUP_COLUMNS:
        assert column in collected.frame.columns
    assert set(collected.frame[labels.CLASS_COLUMN]) == {"healthy", "acidosis"}


@pytest.mark.parametrize(
    "analysis, dirname, stem",
    [
        (forecast_analysis.run_forecast_analysis, forecast_analysis.ANALYSIS_DIRNAME, "per_sample"),
        (uplift_analysis.run_uplift_analysis, uplift_analysis.ANALYSIS_DIRNAME, "per_sample"),
    ],
)
def test_an_analysis_emits_both_variants_on_a_multi_class_split(
    analysis, dirname, stem, make_eval_runner, multi_class_loader, tiny_eval_config, tmp_path
) -> None:
    """The branch the generated shards exist to reach."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    output_dir = tmp_path / "results"
    torch.manual_seed(7)
    summary = analysis(
        runner, multi_class_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=output_dir, probe={"n_samples": 6},
    )
    directory = output_dir / dirname

    for column in labels.GROUP_COLUMNS:
        record = summary["by_group"][column]
        assert record["skipped"] is False, f"{dirname}/{column}: {record.get('reason')}"
        assert (directory / f"{stem}_by_{column}.csv").is_file()
        assert (directory / f"{stem}_by_{column}.pdf").stat().st_size > 0

    assert summary["by_group"][labels.CLASS_COLUMN]["groups"] == ["acidosis", "healthy"]
    assert len(summary["by_group"][labels.SUBGROUP_COLUMN]["groups"]) == 3


def test_an_analysis_records_a_clean_skip_on_the_single_class_shard(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """The committed fixture is one file with an all-zero target, so both axes must skip."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    output_dir = tmp_path / "results"
    torch.manual_seed(7)
    summary = forecast_analysis.run_forecast_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=output_dir, probe={"n_samples": 4},
    )
    for column in labels.GROUP_COLUMNS:
        assert summary["by_group"][column]["skipped"] is True
    # And the pooled output is exactly what it was before any of this existed.
    assert (output_dir / forecast_analysis.ANALYSIS_DIRNAME / "per_sample.csv").is_file()
    assert not list(
        (output_dir / forecast_analysis.ANALYSIS_DIRNAME).glob("*_by_*")
    )


def test_the_pooled_numbers_are_unchanged_by_the_grouped_variant(
    make_eval_runner, multi_class_loader, tiny_eval_config, tmp_path
) -> None:
    """A grouped variant is written beside the pooled output, never instead of it."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    output_dir = tmp_path / "results"
    torch.manual_seed(7)
    summary = forecast_analysis.run_forecast_analysis(
        runner, multi_class_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=output_dir, probe={"n_samples": 6},
    )
    frame = pd.read_csv(
        output_dir / forecast_analysis.ANALYSIS_DIRNAME / "per_sample.csv"
    )
    pooled = frame["feat_mse_total"].to_numpy()
    assert summary["mean_feat_mse_total"] == pytest.approx(float(np.nanmean(pooled)), rel=1e-6)

    # The grouped counts must add up to the pooled population, or one of the two is wrong.
    counts = summary["by_group"][labels.SUBGROUP_COLUMN]["n_per_group"]
    assert sum(counts.values()) == len(frame)


def test_the_grouped_records_are_json_safe(tmp_path) -> None:
    """They land in ``summary.json``, which is written with ``allow_nan=False``."""
    frame = _frame(["a", "a", "b", "b"])
    emitted = report.emit_grouped_variants(frame, tmp_path, value_columns=["feat_mse_total"])
    json.dumps(report.json_safe(emitted), allow_nan=False)
