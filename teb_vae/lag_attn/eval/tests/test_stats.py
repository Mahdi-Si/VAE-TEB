r"""Tests for the shared rank statistics at their Layer-0 home.

These functions were extracted from ``cross_subgroup`` so a second analysis could reuse them
without an analysis-to-analysis import. ``test_cross_subgroup`` still exercises them through the
``cross_subgroup.*`` re-exports; this file pins them at their own module, so the extraction cannot
silently rot if that re-export is ever removed. Every value is checked against a direct
``scipy`` / definitional computation, never a recorded constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from teb_vae.lag_attn.eval import stats


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------
def test_holm_matches_its_definition_on_a_worked_example() -> None:
    r"""$\tilde{p}_{(i)} = \max_{k \le i} \min((m - k + 1) p_{(k)}, 1)$."""
    # m=4: 4*0.01=0.04, 3*0.02=0.06, 2*0.03=0.06 (held by the running max), 1*0.04=0.06.
    assert stats.holm_adjust([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_is_monotone_and_a_non_testable_entry_does_not_consume_a_rank() -> None:
    raw = [0.001, 0.2, 0.02, 0.5, 0.04]
    adjusted = stats.holm_adjust(raw)
    ordered = [adjusted[index] for index in np.argsort(raw)]
    assert ordered == sorted(ordered)

    with_nan = stats.holm_adjust([0.01, float("nan"), 0.02])
    assert with_nan[0] == pytest.approx(stats.holm_adjust([0.01, 0.02])[0])
    assert np.isnan(with_nan[1])


def test_holm_on_an_empty_family_returns_it_unchanged() -> None:
    assert stats.holm_adjust([]) == []


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------
def test_cliffs_delta_matches_a_direct_pair_count() -> None:
    r"""The $U$-derived form must equal $P(X>Y) - P(X<Y)$."""
    from scipy import stats as sp

    rng = np.random.default_rng(3)
    x, y = rng.normal(0.0, 1.0, 60), rng.normal(0.8, 1.0, 45)
    statistic, _ = sp.mannwhitneyu(x, y, alternative="two-sided")
    greater = int((x[:, None] > y[None, :]).sum())
    less = int((x[:, None] < y[None, :]).sum())
    expected = (greater - less) / (x.size * y.size)
    assert stats.cliffs_delta(float(statistic), x.size, y.size) == pytest.approx(expected, abs=1e-12)


def test_cliffs_delta_is_undefined_for_an_empty_sample() -> None:
    assert np.isnan(stats.cliffs_delta(0.0, 0, 5))


@pytest.mark.parametrize(
    "delta, expected",
    [(0.05, "negligible"), (0.2, "small"), (0.4, "medium"), (0.9, "large"), (-0.9, "large")],
)
def test_the_magnitude_labels_follow_the_conventional_thresholds(delta, expected) -> None:
    assert stats.delta_magnitude(delta) == expected


# ---------------------------------------------------------------------------
# Kruskal-Wallis and pairwise
# ---------------------------------------------------------------------------
def test_kruskal_matches_scipy_and_notes_a_constant_metric() -> None:
    from scipy import stats as sp

    rng = np.random.default_rng(1)
    samples = {name: rng.normal(centre, 0.4, 30) for name, centre in
               (("a", 1.0), ("b", 2.0), ("c", 3.0))}
    record = stats.kruskal_across_groups(samples)
    statistic, p_value = sp.kruskal(*samples.values())
    assert record["statistic"] == pytest.approx(float(statistic))
    assert record["p_value"] == pytest.approx(float(p_value))

    constant = {"a": np.full(5, 0.5), "b": np.full(5, 0.5)}
    noted = stats.kruskal_across_groups(constant)
    assert np.isnan(noted["p_value"])
    assert "constant" in noted["note"]


def test_pairwise_compares_every_pair_with_a_signed_effect_size() -> None:
    rng = np.random.default_rng(2)
    samples = {"low": rng.normal(0.0, 0.4, 30), "high": rng.normal(3.0, 0.4, 30)}
    records = stats.pairwise_comparisons(samples)
    assert len(records) == 1
    item = records[0]
    assert item["left"] == "high" and item["right"] == "low"  # sorted order
    # 'high' runs above 'low', so the left-vs-right delta is positive.
    assert item["cliffs_delta"] > 0.9
    assert item["magnitude"] == "large"
    assert "left group's values run higher" in item["delta_orientation"]


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank, paired
# ---------------------------------------------------------------------------
def test_wilcoxon_matches_a_hand_computed_exact_example() -> None:
    r"""Differences $[1, 2, 3, -4, 5]$: $|d|$ ranks $1..5$, so $W^- = 4$ and $W^+ = 11$.

    The exact two-sided $p$ enumerates all $2^5 = 32$ sign assignments. Seven have a
    lesser-sign rank sum $\le 4$ -- $\{\}$, $\{1\}$, $\{2\}$, $\{3\}$, $\{4\}$, $\{1,2\}$,
    $\{1,3\}$ -- so $p = 2 \cdot 7 / 32 = 0.4375$.
    """
    left = [11.0, 12.0, 13.0, 10.0, 15.0]
    right = [10.0, 10.0, 10.0, 14.0, 10.0]

    record = stats.wilcoxon_paired(left, right, label_left="full", label_right="base")

    assert record["n_pairs"] == 5
    assert record["statistic"] == pytest.approx(4.0)
    assert record["p_value"] == pytest.approx(0.4375)
    assert record["median_difference"] == pytest.approx(2.0)
    assert "full runs higher than base" in record["difference_orientation"]


def test_wilcoxon_drops_non_finite_pairs_and_counts_them() -> None:
    """A recording that scored no anchors on one branch is absent, not tied at zero.

    Non-finite on **either** side, which is what the guard says and what the callers need: the
    right-hand column is the one that goes NaN in practice, where a per-condition unstack leaves
    a recording with no control anchors empty. A left-only fixture cannot tell
    ``isfinite(x) & isfinite(y)`` from ``isfinite(x)``, and scipy returns a silent NaN p-value for
    the whole cohort if the right column slips through.
    """
    record = stats.wilcoxon_paired(
        [1.0, 2.0, 3.0, float("nan"), 5.0, 6.0], [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0]
    )
    assert record["n_pairs"] == 4
    assert record["n_dropped"] == 2


def test_wilcoxon_notes_rather_than_raises_on_degenerate_input() -> None:
    too_few = stats.wilcoxon_paired([1.0, 2.0], [0.0, 0.0])
    assert too_few["n_pairs"] < stats.MIN_GROUP_SIZE
    assert np.isnan(too_few["p_value"]) and "minimum" in too_few["note"]

    identical = stats.wilcoxon_paired([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(identical["p_value"]) and "exactly zero" in identical["note"]

    unpaired = stats.wilcoxon_paired([1.0, 2.0, 3.0], [1.0, 2.0])
    assert np.isnan(unpaired["p_value"]) and "unpaired" in unpaired["note"]


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------
def test_bootstrap_is_reproducible_from_its_seed() -> None:
    """The seed is recorded in the summary, so the interval must follow from it alone."""
    rng = np.random.default_rng(7)
    sample = rng.normal(2.0, 1.0, 40)

    first = stats.bootstrap_ci(sample, seed=11, resamples=500)
    again = stats.bootstrap_ci(sample, seed=11, resamples=500)
    different = stats.bootstrap_ci(sample, seed=12, resamples=500)

    assert (first["lo"], first["hi"]) == (again["lo"], again["hi"])
    assert (different["lo"], different["hi"]) != (first["lo"], first["hi"])


def test_bootstrap_brackets_the_mean_at_the_stated_coverage() -> None:
    """Nominal 95% coverage, checked by repeating the whole experiment rather than asserting
    the interval contains the mean once -- which a broken implementation could also manage."""
    rng = np.random.default_rng(5)
    covered = 0
    trials = 200
    for trial in range(trials):
        sample = rng.normal(3.0, 1.0, 30)
        record = stats.bootstrap_ci(sample, seed=trial, resamples=400)
        # The bootstrap estimates the sampling distribution of the sample mean about the
        # population mean, so coverage is asserted against the population value.
        covered += int(record["lo"] <= 3.0 <= record["hi"])
    # Both bounds bite. `covered / trials` is in [0, 1] by construction, so an upper bound of 1.0
    # asserts nothing -- and the direction it would have to catch is real: quantiling the
    # resampled *values* rather than the resampled *means* is the textbook percentile-bootstrap
    # mistake, and it yields intervals about six times too wide, coverage 1.000, and every
    # published `ci_lo`/`ci_hi` bracketing zero on a well-separated finding. The correct
    # implementation scores 0.945 here, so 0.99 separates the two with margin.
    assert 0.88 <= covered / trials <= 0.99

    single = stats.bootstrap_ci(rng.normal(3.0, 1.0, 60), seed=1)
    assert single["lo"] < single["point"] < single["hi"]
    assert single["n"] == 60 and single["resamples"] == stats.DEFAULT_BOOTSTRAP_RESAMPLES


def test_bootstrap_notes_rather_than_raises_on_too_few_recordings() -> None:
    record = stats.bootstrap_ci([1.0, float("nan"), 2.0])
    assert record["n"] == 2 and record["n_dropped"] == 1
    assert np.isnan(record["point"]) and "minimum" in record["note"]


def test_bootstrap_rejects_a_caller_error_rather_than_returning_a_degenerate_interval() -> None:
    """A bad ``confidence`` is a bug in the caller, not a property of the data."""
    with pytest.raises(ValueError, match="confidence"):
        stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], confidence=1.0)
    with pytest.raises(ValueError, match="resamples"):
        stats.bootstrap_ci([1.0, 2.0, 3.0, 4.0], resamples=0)
