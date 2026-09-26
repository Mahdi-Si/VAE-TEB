r"""Tests for the cross-subgroup statistics.

Every statistic is asserted **against a direct ``scipy.stats`` computation** on the same numbers,
not against a recorded constant. A recorded constant would pass forever on an implementation that
had started passing the groups in a different order, or dropping one; recomputing catches both.

Holm and Cliff's delta themselves are pinned at their own module, in ``test_stats.py``; this file
checks the analysis that composes them: the omnibus, the ordering of the pairwise sweep, the
exclusion of small groups, and the recorded skips.

The whole module is driven by CSVs written into ``tmp_path``. That is not a convenience: the
analysis is *specified* to run against a finished run directory with no model, and building its
inputs from a model would test something else.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.analyses import cross_subgroup

#: A group size comfortably above ``MIN_GROUP_SIZE``, so the tests are about the statistics
#: rather than about the exclusion rule -- which has its own test.
GROUP_SIZE = 40


def _write_run(
    directory: Path,
    *,
    separated: bool = True,
    subgroups=("healthy_no_bg_no_cs", "acidosis_cs", "hie_cs"),
    n: int = GROUP_SIZE,
    seed: int = 0,
) -> Path:
    """Write the per-sample CSVs a finished run would have left behind.

    Args:
        directory: The results directory to populate.
        separated: Whether the subgroups are drawn from genuinely different distributions.
        subgroups: The subgroups to write.
        n: Samples per subgroup.
        seed: Seed, so the fixture is reproducible.

    Returns:
        ``directory``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for offset, subgroup in enumerate(subgroups):
        # Well separated when asked for, identically distributed when not -- so a test can assert
        # both that a real difference is found and that an absent one is not invented.
        centre = 1.0 + (3.0 * offset if separated else 0.0)
        for index in range(n):
            rows.append({
                "sample_index": offset * n + index,
                "guid": f"{subgroup}_{index:03d}",
                "source_file": f"{subgroup}.hdf5",
                labels.CLASS_COLUMN: subgroup.split("_")[0],
                labels.SUBGROUP_COLUMN: subgroup,
                "feat_mse_total": rng.normal(centre, 0.4),
                "feat_r2_total": rng.normal(0.2, 0.1),
            })
    frame = pd.DataFrame(rows)

    forecast_dir = directory / "forecast"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(forecast_dir / "per_sample.csv", index=False)
    return directory


@pytest.fixture
def separated_run(tmp_path) -> Path:
    return _write_run(tmp_path / "results", separated=True)


@pytest.fixture
def flat_run(tmp_path) -> Path:
    return _write_run(tmp_path / "results", separated=False, seed=5)


# ---------------------------------------------------------------------------
# The omnibus test
# ---------------------------------------------------------------------------
def test_the_kruskal_statistic_matches_a_direct_scipy_call(separated_run) -> None:
    """Recomputed rather than recorded: a recorded constant survives a group being dropped."""
    from scipy import stats

    record = cross_subgroup.analyse_metrics(separated_run)
    frame = pd.read_csv(separated_run / "forecast" / "per_sample.csv")
    groups = [
        frame.loc[frame[labels.SUBGROUP_COLUMN] == name, "feat_mse_total"].to_numpy()
        for name in sorted(frame[labels.SUBGROUP_COLUMN].unique())
    ]
    expected_statistic, expected_p = stats.kruskal(*groups)

    omnibus = {item["metric"]: item for item in record["omnibus"]}
    result = omnibus["forecast.feat_mse_total"]
    assert result["statistic"] == pytest.approx(float(expected_statistic))
    assert result["p_value"] == pytest.approx(float(expected_p))
    assert result["n_groups"] == 3


def test_a_genuinely_separated_metric_survives_holm(separated_run) -> None:
    record = cross_subgroup.analyse_metrics(separated_run)
    assert "forecast.feat_mse_total" in record["significant_metrics"]


def test_identically_distributed_subgroups_are_not_declared_different(flat_run) -> None:
    """The non-vacuity check: the procedure must be able to find nothing."""
    record = cross_subgroup.analyse_metrics(flat_run)
    assert record["significant_metrics"] == []
    assert record["pairwise"] == {}


def test_pairwise_tests_run_only_for_metrics_that_survived_holm(separated_run) -> None:
    """The ordering is the multiple-comparison argument, not an implementation detail."""
    record = cross_subgroup.analyse_metrics(separated_run)
    assert set(record["pairwise"]) == set(record["significant_metrics"])
    assert "forecast.feat_r2_total" not in record["pairwise"], (
        "a metric with no omnibus difference must not get 3 pairwise tests"
    )


def test_every_pair_is_compared_for_a_surviving_metric(separated_run) -> None:
    record = cross_subgroup.analyse_metrics(separated_run)
    comparisons = record["pairwise"]["forecast.feat_mse_total"]
    assert len(comparisons) == 3  # C(3, 2)
    assert len({(item["left"], item["right"]) for item in comparisons}) == 3


def test_the_pairwise_p_and_effect_size_match_a_direct_computation(separated_run) -> None:
    from scipy import stats

    record = cross_subgroup.analyse_metrics(separated_run)
    frame = pd.read_csv(separated_run / "forecast" / "per_sample.csv")

    for item in record["pairwise"]["forecast.feat_mse_total"]:
        left = frame.loc[
            frame[labels.SUBGROUP_COLUMN] == item["left"], "feat_mse_total"
        ].to_numpy()
        right = frame.loc[
            frame[labels.SUBGROUP_COLUMN] == item["right"], "feat_mse_total"
        ].to_numpy()
        statistic, p_value = stats.mannwhitneyu(left, right, alternative="two-sided")

        assert item["p_value"] == pytest.approx(float(p_value))
        assert item["cliffs_delta"] == pytest.approx(
            cross_subgroup.cliffs_delta(float(statistic), left.size, right.size)
        )


def test_the_delta_sign_says_which_group_runs_higher(separated_run) -> None:
    """Documented in the record; a reader who has the sign backwards inverts every conclusion."""
    record = cross_subgroup.analyse_metrics(separated_run)
    frame = pd.read_csv(separated_run / "forecast" / "per_sample.csv")

    for item in record["pairwise"]["forecast.feat_mse_total"]:
        left_median = frame.loc[
            frame[labels.SUBGROUP_COLUMN] == item["left"], "feat_mse_total"
        ].median()
        right_median = frame.loc[
            frame[labels.SUBGROUP_COLUMN] == item["right"], "feat_mse_total"
        ].median()
        assert np.sign(item["cliffs_delta"]) == np.sign(left_median - right_median)


# ---------------------------------------------------------------------------
# Small and missing groups
# ---------------------------------------------------------------------------
def test_a_group_below_the_minimum_size_is_excluded_and_recorded(tmp_path) -> None:
    """A rank test on two values has no power; its p-value describes the group size."""
    directory = _write_run(tmp_path / "results", n=GROUP_SIZE)
    frame = pd.read_csv(directory / "forecast" / "per_sample.csv")
    # Leave one subgroup with two samples.
    trimmed = pd.concat([
        frame[frame[labels.SUBGROUP_COLUMN] != "hie_cs"],
        frame[frame[labels.SUBGROUP_COLUMN] == "hie_cs"].head(2),
    ])
    trimmed.to_csv(directory / "forecast" / "per_sample.csv", index=False)

    record = cross_subgroup.analyse_metrics(directory)
    result = {item["metric"]: item for item in record["omnibus"]}["forecast.feat_mse_total"]
    assert result["n_groups"] == 2
    assert result["groups_excluded_as_too_small"] == {"hie_cs": 2}


def test_a_single_subgroup_run_is_skipped_rather_than_reported(tmp_path) -> None:
    """The ordinary outcome on the single-file pretraining split."""
    directory = _write_run(tmp_path / "results", subgroups=("healthy_no_bg_no_cs",))
    summary = cross_subgroup.run_cross_subgroup(directory)

    assert summary["skipped"] is True
    assert "two groups" in summary["reason"]
    assert not (directory / cross_subgroup.ANALYSIS_DIRNAME).exists(), (
        "a skipped analysis must leave no half-written directory"
    )


def test_a_missing_source_is_recorded_rather_than_raising(separated_run) -> None:
    """It is designed to run against a partial run directory, where most sources are absent."""
    record = cross_subgroup.analyse_metrics(separated_run)
    missing = {item["analysis"] for item in record["missing_sources"]}
    assert "uplift" in missing and "latent" in missing
    for item in record["missing_sources"]:
        assert item["reason"], "a missing source must say why"
    # And the sources that *were* present still produced results.
    assert record["n_metrics_tested"] == 2


def test_a_constant_metric_is_noted_rather_than_raising(tmp_path) -> None:
    """``scipy.kruskal`` raises on identical values; a constant metric is a finding, not a crash."""
    directory = _write_run(tmp_path / "results")
    frame = pd.read_csv(directory / "forecast" / "per_sample.csv")
    frame["feat_r2_total"] = 0.5
    frame.to_csv(directory / "forecast" / "per_sample.csv", index=False)

    record = cross_subgroup.analyse_metrics(directory)
    result = {item["metric"]: item for item in record["omnibus"]}["forecast.feat_r2_total"]
    assert np.isnan(result["p_value"])
    assert "constant" in result["note"]


# ---------------------------------------------------------------------------
# The run, with no model
# ---------------------------------------------------------------------------
def test_the_analysis_runs_with_no_model_and_no_loader(separated_run) -> None:
    """The specified use: re-runnable from an existing run directory."""
    summary = cross_subgroup.run_cross_subgroup_analysis(
        None, None, eval_config={}, output_dir=separated_run, probe=None
    )
    assert summary["skipped"] is False
    assert summary["n_significant"] >= 1


def test_the_significance_table_has_one_row_per_metric(separated_run) -> None:
    cross_subgroup.run_cross_subgroup(separated_run)
    table = pd.read_csv(
        separated_run / cross_subgroup.ANALYSIS_DIRNAME / "significance.csv"
    )
    assert set(table["metric"]) == {"forecast.feat_mse_total", "forecast.feat_r2_total"}
    for column in ("p_value", "p_holm", "correction", "alpha", "significant", "file"):
        assert column in table.columns


def test_the_summary_ranks_the_largest_effects_by_magnitude_not_by_p(separated_run) -> None:
    r"""At eight subgroups the smallest $p$ is usually the largest pair, not the largest effect."""
    summary = cross_subgroup.run_cross_subgroup(separated_run)
    deltas = [abs(item["cliffs_delta"]) for item in summary["largest_effects"]]
    assert deltas == sorted(deltas, reverse=True)
    assert summary["largest_effects"][0]["magnitude"] in {"small", "medium", "large"}


def test_the_same_procedure_runs_over_the_clinical_class_axis(separated_run) -> None:
    """One implementation, two axes: only the grouping column differs."""
    record = cross_subgroup.analyse_metrics(separated_run, group_column=labels.CLASS_COLUMN)
    assert record["group_column"] == labels.CLASS_COLUMN
    assert record["n_metrics_tested"] == 2


def test_the_record_is_json_safe(separated_run) -> None:
    """It lands in ``summary.json``, which is written with ``allow_nan=False``."""
    from teb_vae.lag_attn.eval.report import json_safe

    summary = cross_subgroup.run_cross_subgroup(separated_run)
    json.dumps(json_safe(summary), allow_nan=False)
