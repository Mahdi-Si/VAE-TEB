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
