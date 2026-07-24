r"""Non-parametric rank statistics shared by more than one analysis.

These are the pieces the cross-subgroup and the KLD-to-delivery analyses both need: the Holm
step-down correction, Cliff's delta and its magnitude label, the Kruskal-Wallis omnibus wrapper,
and the pairwise Mann-Whitney sweep. They live here rather than inside either analysis because
``analyses/__init__.py``'s rule is that analyses never import one another -- so a helper two of
them share has to sit one layer down, exactly as ``masks`` and ``metrics`` do.

Everything here operates on plain ``dict``\ s of ``numpy`` arrays and Python floats. Nothing reads
a config, holds a model, or touches the filesystem, and the only third-party dependency beyond
``numpy`` is ``scipy.stats``, imported **lazily at each call site** -- so a box without SciPy loses
the analyses that reach these functions and nothing else, and the import cost is paid only when a
test actually runs rather than at module load.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

#: Smallest group that may enter a test. Below three finite values a rank test has essentially no
#: power and its $p$-value is an artifact of the group size rather than a statement about the data,
#: so such a group is *excluded and recorded* rather than silently entered.
MIN_GROUP_SIZE = 3

#: Cliff's delta magnitude thresholds, as Romano et al. give them. Reported beside every delta so
#: a reader does not have to carry the table, and so "significant" is never quoted without the
#: effect size that says whether it matters.
DELTA_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    r"""Return the Holm step-down adjusted $p$-values, in the input order.

    $$\tilde{p}_{(i)} = \max_{k \le i}\ \min\!\left((m - k + 1)\,p_{(k)},\ 1\right)$$

    The running maximum is what makes the adjustment *monotone*: without it a later, larger raw
    $p$ could receive a smaller adjusted one and the step-down procedure would reject a hypothesis
    while accepting a more significant one.

    Uniformly more powerful than Bonferroni at the same family-wise error rate, and there is no
    assumption Bonferroni satisfies that Holm does not.

    Args:
        p_values: Raw $p$-values. Non-finite entries are passed through unchanged, so a test
            that could not run does not consume a rank in the correction.

    Returns:
        The adjusted $p$-values, aligned with the input.
    """
    values = [float(value) for value in p_values]
    testable = [index for index, value in enumerate(values) if np.isfinite(value)]
    count = len(testable)
    adjusted = list(values)
    if count == 0:
        return adjusted

    order = sorted(testable, key=lambda index: values[index])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min((count - rank) * values[index], 1.0))
        adjusted[index] = running
    return adjusted


def cliffs_delta(u_statistic: float, n_x: int, n_y: int) -> float:
    r"""Cliff's delta, from the Mann-Whitney $U$ the same comparison already produced.

    $$\delta = \frac{2U}{n_x n_y} - 1
    = P(X > Y) - P(X < Y)$$

    Derived from $U$ rather than counted directly: ``scipy``'s $U_1$ is
    $\#(x > y) + \tfrac{1}{2}\#(\mathrm{ties})$, which is exactly the tie-corrected numerator
    Cliff's delta wants -- so the effect size costs nothing beyond the test that was already run,
    and it cannot disagree with it. Counting pairs directly would be $O(n_x n_y)$ per pair, or
    $16$ million comparisons per pair at a realistic subgroup size.

    $\delta = 0$ means the two distributions overlap completely; $\pm 1$ means they are disjoint.

    Args:
        u_statistic: The $U_1$ statistic for ``x`` against ``y``.
        n_x: Size of the first sample.
        n_y: Size of the second sample.

    Returns:
        The effect size in $[-1, 1]$, or ``NaN`` when either sample is empty.
    """
    if n_x <= 0 or n_y <= 0:
        return float("nan")
    return 2.0 * float(u_statistic) / (float(n_x) * float(n_y)) - 1.0


def delta_magnitude(delta: float) -> str:
    """Return the conventional magnitude label for a Cliff's delta.

    Args:
        delta: The effect size.

    Returns:
        ``'negligible'`` / ``'small'`` / ``'medium'`` / ``'large'``, or ``'undefined'``.
    """
    if not np.isfinite(delta):
        return "undefined"
    magnitude = abs(float(delta))
    for threshold, label in DELTA_THRESHOLDS:
        if magnitude < threshold:
            return label
    return "large"


def kruskal_across_groups(samples: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Run the omnibus Kruskal-Wallis test across every group of one metric.

    Args:
        samples: Group to its finite values.

    Returns:
        ``statistic``, ``p_value`` and ``n_per_group``. Both statistics are ``NaN`` when the test
        could not run -- fewer than two groups, or values that are identical throughout, which
        ``scipy`` rejects because the ranks then carry no information at all.
    """
    from scipy import stats

    record: Dict[str, Any] = {
        "test": "kruskal-wallis",
        "n_groups": len(samples),
        "n_per_group": {group: int(values.size) for group, values in samples.items()},
        "statistic": float("nan"),
        "p_value": float("nan"),
    }
    if len(samples) < 2:
        record["note"] = "fewer than two testable groups"
        return record

    pooled = np.concatenate(list(samples.values()))
    if np.unique(pooled).size < 2:
        # Not a failure: a metric that is constant across the whole split genuinely carries no
        # between-group information, and scipy raises rather than returning p = 1.
        record["note"] = "the metric is constant across every group, so the ranks carry nothing"
        return record

    statistic, p_value = stats.kruskal(*samples.values())
    record["statistic"] = float(statistic)
    record["p_value"] = float(p_value)
    return record


def pairwise_comparisons(samples: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Run every pairwise Mann-Whitney test with its Cliff's delta.

    Intended to be called only for a metric whose omnibus test survived the correction -- running
    $\\binom{k}{2}$ pairwise tests on a metric whose omnibus test found nothing is the
    multiple-comparison problem with extra steps.

    Args:
        samples: Group to its finite values.

    Returns:
        One record per unordered pair, carrying the test, its $p$-value, the effect size and its
        magnitude label. The delta's sign is oriented ``left`` against ``right``: positive means
        the left group's values run higher.
    """
    import itertools

    from scipy import stats

    records: List[Dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(samples), 2):
        x, y = samples[left], samples[right]
        try:
            statistic, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError as exc:
            # Identical constant samples. Recorded rather than dropped: a pair that could not be
            # compared is a different statement from a pair that showed no difference.
            records.append({
                "test": "mann-whitney-u",
                "left": left, "right": right,
                "n_left": int(x.size), "n_right": int(y.size),
                "p_value": float("nan"), "cliffs_delta": float("nan"),
                "magnitude": "undefined", "note": str(exc),
            })
            continue
        delta = cliffs_delta(float(statistic), int(x.size), int(y.size))
        records.append({
            "test": "mann-whitney-u",
            "left": left, "right": right,
            "n_left": int(x.size), "n_right": int(y.size),
            "u_statistic": float(statistic),
            "p_value": float(p_value),
            "cliffs_delta": delta,
            "magnitude": delta_magnitude(delta),
            "delta_orientation": "positive means the left group's values run higher",
        })
    return records
