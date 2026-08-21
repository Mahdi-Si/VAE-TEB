r"""Non-parametric rank statistics shared by more than one analysis.

These are the pieces the cross-subgroup and the KLD-to-delivery analyses both need: the Holm
step-down correction, Cliff's delta and its magnitude label, the Kruskal-Wallis omnibus wrapper,
and the pairwise Mann-Whitney sweep -- plus the two an unpaired rank sweep cannot answer: a
paired within-recording comparison (:func:`wilcoxon_paired`) and an interval on a mean over
recordings (:func:`bootstrap_ci`). They live here rather than inside either analysis because
``analyses/__init__.py``'s rule is that analyses never import one another -- so a helper two of
them share has to sit one layer down, exactly as ``masks`` and ``metrics`` do.

Everything here operates on plain ``dict``\ s of ``numpy`` arrays and Python floats. Nothing reads
a config, holds a model, or touches the filesystem, and the only third-party dependency beyond
``numpy`` is ``scipy.stats``, imported **lazily at each call site** -- so a box without SciPy loses
the analyses that reach these functions and nothing else, and the import cost is paid only when a
test actually runs rather than at module load.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Smallest group that may enter a test. Below three finite values a rank test has essentially no
#: power and its $p$-value is an artifact of the group size rather than a statement about the data,
#: so such a group is *excluded and recorded* rather than silently entered.
MIN_GROUP_SIZE = 3

#: Bootstrap resamples used when a caller does not say. Two thousand is the conventional floor for
#: a percentile interval: the $2.5$th and $97.5$th order statistics of $2000$ draws are stable to
#: about the third decimal, and the cost is one array index per draw.
DEFAULT_BOOTSTRAP_RESAMPLES = 2000

#: Cliff's delta magnitude thresholds, as Romano et al. give them. Reported beside every delta so
#: a reader does not have to carry the table, and so "significant" is never quoted without the
#: effect size that says whether it matters.
DELTA_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.330, "small"),
    (0.474, "medium"),
)

#: Recorded on a window that could not be tested at all. Deliberately says *groups* and *values*
#: rather than naming a cohort or a unit: which groups were dropped and how large each was is on
#: the caller's own exclusion record, which is the half a reader needs, and a note that named the
#: caller's unit would be wrong for every other caller.
TOO_FEW_GROUPS_NOTE = "fewer than two groups had enough values in this window"


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


def wilcoxon_paired(
    left: Iterable[float],
    right: Iterable[float],
    *,
    label_left: str = "left",
    label_right: str = "right",
) -> Dict[str, Any]:
    r"""Wilcoxon signed-rank test on two readouts measured on the *same* recordings.

    Mann-Whitney compares two independent groups; this is the paired counterpart, for the
    comparisons where each recording contributes both values -- a branch against a control on the
    same segment, a model score against a baseline's. Pairing removes the between-recording
    variance, which dominates every readout in this pipeline, so the paired test is the one with
    power here and the unpaired one would throw that away.

    The statistic is $\min(W^+, W^-)$, the sum of the ranks of the $|d_i|$ carrying one sign,
    over the pairs with $d_i = \mathrm{left}_i - \mathrm{right}_i \ne 0$. Pairs where either
    value is non-finite are dropped and counted rather than imputed: a recording that scored no
    anchors on one branch is absent from the comparison, not tied on it.

    Args:
        left: One value per recording.
        right: The paired value per recording, in the same order.
        label_left: Name of the left readout, for the orientation note.
        label_right: Name of the right readout.

    Returns:
        The test, the number of usable pairs, the statistic, the $p$-value, and the median paired
        difference. Both statistics are ``NaN`` and a ``note`` is set -- never an exception --
        when the test could not run: mismatched lengths, fewer than :data:`MIN_GROUP_SIZE`
        finite pairs, or differences that are identically zero, which ``scipy`` rejects because
        the signed ranks then carry nothing.
    """
    from scipy import stats

    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    record: Dict[str, Any] = {
        "test": "wilcoxon-signed-rank",
        "label_left": label_left,
        "label_right": label_right,
        "n_pairs": 0,
        "statistic": float("nan"),
        "p_value": float("nan"),
        "median_difference": float("nan"),
        "difference_orientation": f"positive means {label_left} runs higher than {label_right}",
    }
    if x.size != y.size:
        record["note"] = f"unpaired inputs: {x.size} left values against {y.size} right values"
        return record

    usable = np.isfinite(x) & np.isfinite(y)
    differences = x[usable] - y[usable]
    record["n_pairs"] = int(differences.size)
    record["n_dropped"] = int(x.size - differences.size)
    if differences.size < MIN_GROUP_SIZE:
        record["note"] = (
            f"only {differences.size} finite pair(s); below the minimum of {MIN_GROUP_SIZE} a "
            f"signed-rank p-value is an artifact of the pair count"
        )
        return record

    record["median_difference"] = float(np.median(differences))
    if not np.any(differences != 0.0):
        # Not a failure: the two readouts are identical on every recording, which is a finding.
        record["note"] = "every paired difference is exactly zero, so the signed ranks carry nothing"
        return record

    # Read by attribute rather than unpacked: ``wilcoxon`` returns a named result object, and
    # tuple-unpacking it loses which field is which at the call site. Typed ``Any`` because the
    # installed scipy stubs do not describe that object's fields.
    result: Any = stats.wilcoxon(differences, alternative="two-sided")
    record["statistic"] = float(result.statistic)
    record["p_value"] = float(result.pvalue)
    return record


def bootstrap_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> Dict[str, Any]:
    r"""A seeded percentile bootstrap interval for the mean of a per-recording vector.

    $$\left[\hat{\theta}^{*}_{(\alpha/2)},\ \hat{\theta}^{*}_{(1 - \alpha/2)}\right],
    \qquad \hat{\theta}^{*}_b = \frac{1}{n}\sum_{i} v_{\pi_b(i)}$$

    Percentile rather than normal-approximation, because the per-recording distributions here are
    small-$n$ and visibly skewed, and a $\pm 1.96\,\mathrm{SE}$ interval on such a sample reports
    a symmetry the data does not have.

    **Recordings, not anchors.** The caller passes one value per recording; consecutive anchors
    overlap in $29$ of their $30$ horizon steps and one recording contributes tens of segments, so
    resampling anchors would report an interval narrower than the data supports by roughly the
    pseudo-replication factor.

    Args:
        values: One value per recording. Non-finite entries are dropped and counted.
        confidence: Coverage of the interval, in $(0, 1)$.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling, so the interval is reproducible from the summary alone.

    Returns:
        The point estimate, the interval, the honest $n$, and the settings that produced it. The
        point and the bounds are ``NaN`` with a ``note`` -- never an exception -- when fewer than
        :data:`MIN_GROUP_SIZE` finite values are available.

    Raises:
        ValueError: If ``confidence`` is not in $(0, 1)$ or ``resamples`` is not positive. Those
            are caller errors rather than data conditions, and a silently degenerate interval
            would read as a measurement.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}.")
    if int(resamples) < 1:
        raise ValueError(f"resamples must be positive, got {resamples!r}.")

    sample = np.asarray(list(values), dtype=np.float64)
    finite = sample[np.isfinite(sample)]
    alpha = 1.0 - float(confidence)
    record: Dict[str, Any] = {
        "statistic": "mean",
        "n": int(finite.size),
        "n_dropped": int(sample.size - finite.size),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "seed": int(seed),
        "method": "percentile bootstrap over recordings",
        "point": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
    }
    if finite.size < MIN_GROUP_SIZE:
        record["note"] = (
            f"only {finite.size} finite value(s); below the minimum of {MIN_GROUP_SIZE} a "
            f"bootstrap interval reproduces the sample rather than estimating its spread"
        )
        return record

    record["point"] = float(finite.mean())
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, finite.size, size=(int(resamples), finite.size))
    means = finite[draws].mean(axis=1)
    record["lo"] = float(np.quantile(means, alpha / 2.0))
    record["hi"] = float(np.quantile(means, 1.0 - alpha / 2.0))
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


def windowed_group_comparisons(
    samples_by_window: Mapping[Any, Mapping[str, np.ndarray]],
    *,
    meta_by_window: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    window_field: str = "time_bin",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    r"""Test a metric across groups **inside each window**, then correct across the windows.

    The three layers every trajectory analysis in this repository runs, in this order and for
    these reasons:

    * **Per window**, an omnibus :func:`kruskal_across_groups`. Per window rather than pooled is
      what makes the result a statement about the *trajectory*: it localises where the groups
      differ instead of collapsing the whole axis into one number.
    * **Holm across the windows**, which form one family. Twenty-five windows at
      $\alpha = 0.05$ produce a false positive with near certainty by construction, and
      :func:`holm_adjust` passes a non-finite $p$ through unchanged, so a window that could not be
      tested does not consume a rank in the correction.
    * **Pairwise** :func:`pairwise_comparisons`, for the windows that survived Holm **only**.
      Running $\binom{k}{2}$ pairwise tests on a window whose omnibus found nothing is the
      multiple-comparison problem with extra steps.

    It lives here rather than in the analysis that needed it first because **five** callers now
    need it -- two clocks in each of two evaluation packages, plus the ancestor -- and an analysis
    may never import another. That is the same argument that put :func:`holm_adjust` here, and it
    is sharper for a whole procedure than for one of its steps: three copies of this loop are
    three definitions of what "significant" means, and nothing would keep them equal.

    **The caller owns the population, this function owns the arithmetic.** Which groups are in a
    window, in which order, and which were dropped for being too small are decisions about a
    cohort; they are made before the call and recorded through ``meta_by_window``. This function
    applies no :data:`MIN_GROUP_SIZE` floor of its own -- a window arriving with one usable group
    is recorded as untestable, not silently filtered.

    Args:
        samples_by_window: Ordered mapping from a window key to that window's
            ``{group: finite values}``. **Every** window belongs here, including those with fewer
            than two usable groups: they must appear in the output, or a reader cannot tell a
            window that found nothing from a window that was never looked at. Iteration order is
            preserved, so a caller passing windows in axis order gets them back in axis order.
        meta_by_window: Per-window fields merged into that window's record before the test keys --
            the window's centre, the caller's exclusion record, whatever the caller publishes.
        window_field: The key each record carries its own window identifier under. A parameter
            rather than a fixed name because three record schemas are already published on disk
            and pinned by tests, and renaming a column a reader has is an output change with no
            reader behind it.
        alpha: Family-wise error rate the Holm correction controls.

    Returns:
        ``per_window`` -- one record per window, in input order, carrying the identifier, the
        caller's metadata, the omnibus result, and then ``p_holm``, ``alpha``, ``correction``,
        ``n_windows_in_family`` and ``significant``; ``pairwise`` -- the comparisons of the
        surviving windows, keyed by ``str(window key)``; and the three counts ``n_windows``,
        ``n_windows_tested`` and ``n_significant_windows``. A window that could not be tested
        carries ``NaN`` statistics and :data:`TOO_FEW_GROUPS_NOTE` -- never an exception, because
        an untestable window is an ordinary outcome on a thin cohort.
    """
    metadata = dict(meta_by_window or {})
    windows = list(samples_by_window)
    per_window: List[Dict[str, Any]] = []
    # Only the windows a pairwise sweep may run on, keyed by the caller's own window key rather
    # than by the record's -- so a `meta_by_window` entry that happened to carry `window_field`
    # cannot silently repoint the sweep at another window's samples.
    testable: Dict[Any, Dict[str, np.ndarray]] = {}

    for window in windows:
        usable = dict(samples_by_window[window])
        record: Dict[str, Any] = {window_field: window}
        record.update(dict(metadata.get(window) or {}))
        if len(usable) >= 2:
            record.update(kruskal_across_groups(usable))
            testable[window] = usable
        else:
            record.update({
                "test": "kruskal-wallis",
                "n_groups": len(usable),
                "n_per_group": {
                    group: int(np.asarray(values).size) for group, values in usable.items()
                },
                "statistic": float("nan"),
                "p_value": float("nan"),
                "note": TOO_FEW_GROUPS_NOTE,
            })
        per_window.append(record)

    adjusted = holm_adjust([record["p_value"] for record in per_window])
    n_tested = sum(1 for record in per_window if np.isfinite(record["p_value"]))
    for record, value in zip(per_window, adjusted):
        record["p_holm"] = float(value)
        record["alpha"] = float(alpha)
        record["correction"] = "holm"
        record["n_windows_in_family"] = n_tested
        record["significant"] = bool(np.isfinite(value) and value < float(alpha))

    pairwise = {
        str(window): pairwise_comparisons(testable[window])
        for window, record in zip(windows, per_window)
        if record["significant"]
    }
    return {
        "per_window": per_window,
        "pairwise": pairwise,
        "n_windows": len(per_window),
        "n_windows_tested": n_tested,
        "n_significant_windows": sum(1 for record in per_window if record["significant"]),
    }
