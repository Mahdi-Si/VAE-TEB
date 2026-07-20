r"""Tests for the cross-subgroup statistics.

Every statistic is asserted **against a direct ``scipy.stats`` computation** on the same numbers,
not against a recorded constant. A recorded constant would pass forever on an implementation that
had started passing the groups in a different order, or dropping one; recomputing catches both.

The Holm correction is checked against its definition rather than against ``scipy``, because it is
the one piece here that is genuinely implemented rather than delegated -- including the running
maximum that makes it monotone, which is the part an implementation gets wrong.

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
# Holm
# ---------------------------------------------------------------------------
def test_holm_matches_its_definition_on_a_worked_example() -> None:
    r"""$\tilde{p}_{(i)} = \max_{k \le i} \min((m - k + 1) p_{(k)}, 1)$."""
    adjusted = cross_subgroup.holm_adjust([0.01, 0.02, 0.03, 0.04])
    # m=4: 4*0.01=0.04, 3*0.02=0.06, 2*0.03=0.06 (held up by the running max), 1*0.04=0.06.
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_is_monotone_in_the_raw_p_values() -> None:
    """The running maximum is what guarantees it; without it the step-down order can invert."""
    raw = [0.001, 0.2, 0.02, 0.5, 0.04]
    adjusted = cross_subgroup.holm_adjust(raw)
    ordered = [adjusted[index] for index in np.argsort(raw)]
    assert ordered == sorted(ordered), "a larger raw p received a smaller adjusted one"


def test_holm_is_never_looser_than_the_raw_p_value() -> None:
    raw = [0.001, 0.01, 0.04, 0.2]
    assert all(a >= r for a, r in zip(cross_subgroup.holm_adjust(raw), raw))


def test_holm_is_at_least_as_powerful_as_bonferroni() -> None:
    """The reason it is used: uniformly more powerful at the same family-wise error rate."""
    raw = [0.001, 0.01, 0.02, 0.04]
    holm = cross_subgroup.holm_adjust(raw)
    bonferroni = [min(len(raw) * value, 1.0) for value in raw]
    assert all(h <= b for h, b in zip(holm, bonferroni))
    assert any(h < b for h, b in zip(holm, bonferroni)), "identical to Bonferroni here"


def test_holm_caps_at_one() -> None:
    assert cross_subgroup.holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_a_non_testable_metric_does_not_consume_a_rank() -> None:
    """A metric whose test could not run must not widen the correction for the ones that could."""
    with_nan = cross_subgroup.holm_adjust([0.01, float("nan"), 0.02])
    without = cross_subgroup.holm_adjust([0.01, 0.02])
    assert with_nan[0] == pytest.approx(without[0])
    assert np.isnan(with_nan[1])


def test_holm_on_an_empty_family_returns_it_unchanged() -> None:
    assert cross_subgroup.holm_adjust([]) == []


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------
def test_cliffs_delta_matches_a_direct_pair_count() -> None:
    r"""The $U$-derived form must equal the definition $P(X>Y) - P(X<Y)$."""
    from scipy import stats

    rng = np.random.default_rng(3)
    x, y = rng.normal(0.0, 1.0, 60), rng.normal(0.8, 1.0, 45)
    statistic, _ = stats.mannwhitneyu(x, y, alternative="two-sided")

    greater = int((x[:, None] > y[None, :]).sum())
    less = int((x[:, None] < y[None, :]).sum())
    expected = (greater - less) / (x.size * y.size)

    assert cross_subgroup.cliffs_delta(float(statistic), x.size, y.size) == pytest.approx(
        expected, abs=1e-12
    )


def test_disjoint_samples_give_a_delta_of_one() -> None:
    from scipy import stats

    x, y = np.arange(10.0) + 100.0, np.arange(10.0)
    statistic, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    assert cross_subgroup.cliffs_delta(float(statistic), 10, 10) == pytest.approx(1.0)


def test_identical_samples_give_a_delta_of_zero() -> None:
    """Ties count as a half, which is what makes this exactly zero rather than nearly."""
    from scipy import stats

    x = np.arange(10.0)
    statistic, _ = stats.mannwhitneyu(x, x.copy(), alternative="two-sided")
    assert cross_subgroup.cliffs_delta(float(statistic), 10, 10) == pytest.approx(0.0)


def test_an_empty_sample_gives_an_undefined_delta() -> None:
    assert np.isnan(cross_subgroup.cliffs_delta(0.0, 0, 5))


@pytest.mark.parametrize(
    "delta, expected",
    [(0.05, "negligible"), (0.2, "small"), (0.4, "medium"), (0.9, "large"), (-0.9, "large")],
)
def test_the_magnitude_labels_follow_the_conventional_thresholds(delta, expected) -> None:
    assert cross_subgroup.delta_magnitude(delta) == expected


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
        assert "left group's values run higher" in item["delta_orientation"]


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


def test_the_expected_files_are_written(separated_run) -> None:
    cross_subgroup.run_cross_subgroup(separated_run)
    directory = separated_run / cross_subgroup.ANALYSIS_DIRNAME

    for name in ("significance.csv", "pairwise.csv", cross_subgroup.RESULT_FILENAME):
        assert (directory / name).is_file(), f"{name} was not written"
    assert (directory / "cross_subgroup.pdf").stat().st_size > 0


def test_the_inference_path_is_recorded_beside_every_coefficient(separated_run) -> None:
    """A coefficient whose provenance is not recorded cannot be checked or reproduced."""
    cross_subgroup.run_cross_subgroup(separated_run)
    blob = json.loads(
        (separated_run / cross_subgroup.ANALYSIS_DIRNAME / cross_subgroup.RESULT_FILENAME)
        .read_text(encoding="utf-8")
    )

    for item in blob["omnibus"]:
        assert item["test"] == "kruskal-wallis"
        assert item["correction"] == "holm"
        assert item["source"]["file"].endswith(".csv")
        assert item["source"]["column"]
        assert "alpha" in item and "n_tests_in_family" in item
        assert "min_group_size" in item and "n_per_group" in item
    for comparisons in blob["pairwise"].values():
        for item in comparisons:
            assert item["test"] == "mann-whitney-u"
            assert "n_left" in item and "n_right" in item
    assert "Kruskal-Wallis" in blob["method"] and "Holm" in blob["method"]


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
