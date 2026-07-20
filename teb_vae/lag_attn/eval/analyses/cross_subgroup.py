r"""Do the canonical subgroups actually differ, or does the by-subgroup table only look like they do?

A by-subgroup table invites exactly one question and cannot answer it. Eight cohorts, each with a
mean, will always produce a highest and a lowest; with nine headline metrics that is
seventy-two numbers, and *some* of them will look separated whether or not anything is there. This
analysis is the arithmetic that says which of those separations survive being asked properly.

Three layers, each answering what the one above it leaves open.

**Kruskal-Wallis**, per metric, across all subgroups: is there any difference at all? Non-parametric
because these distributions are routinely skewed and heavy-tailed -- a per-sample masked MSE has a
long right tail of poorly forecast recordings -- and a one-way ANOVA's normality assumption is not
one this data supports.

**Holm**, across metrics: nine tests at $\alpha = 0.05$ produce a false positive about a third of
the time by construction. Holm rather than Bonferroni because it is uniformly more powerful at the
same family-wise error rate, and uniformly more powerful for free is not a trade-off.

**Pairwise Mann-Whitney with Cliff's delta**, but *only for metrics that survived Holm*: this is
the ordering, not an implementation detail. Running $\binom{8}{2} = 28$ pairwise tests on a metric
whose omnibus test found nothing is the multiple-comparison problem with extra steps.

Cliff's delta comes back with every pair because a $p$-value is not an effect size. At the eight-
subgroup scale a difference of no clinical consequence reaches significance readily, and $\delta$
is the number that says whether the cohorts actually separate -- it is the probability that a
random member of one group exceeds a random member of the other, rescaled to $[-1, 1]$.

**This analysis needs no model.** It reads the per-sample CSVs the other analyses already wrote,
so it re-runs against a finished run directory in seconds:

.. code-block:: bash

    python -m teb_vae.lag_attn.eval.run --only cross_subgroup \
        --config teb_vae/lag_attn/eval/configs/eval.yaml \
        --checkpoint /path/to/ckpt --output-dir <an existing run directory>

That is also why it declares no dependency on the analyses that produce its inputs: the dependency
is on the *files*, not on the analyses having run in the same pass, and a source that is absent is
recorded as absent rather than failing the step.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import figures, labels

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "cross_subgroup"

#: Family-wise error rate the Holm correction controls.
DEFAULT_ALPHA = 0.05

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


@dataclass(frozen=True)
class MetricSource:
    """Where one headline metric is read from.

    Attributes:
        analysis: The analysis directory holding the CSV.
        filename: The CSV within it.
        column: The per-sample column to test.
        higher_is_better: Whether a larger value is a better result. Recorded rather than acted
            on, so a reader of a signed effect size knows which direction is good without going
            back to the analysis that produced the column.
    """

    analysis: str
    filename: str
    column: str
    higher_is_better: bool = False

    @property
    def name(self) -> str:
        """The metric's reported name, qualified by the analysis that produced it."""
        return f"{self.analysis}.{self.column}"

    def as_dict(self) -> Dict[str, Any]:
        """Return the source as a record -- the inference path for every coefficient below."""
        return {
            "analysis": self.analysis,
            "file": f"{self.analysis}/{self.filename}",
            "column": self.column,
            "higher_is_better": self.higher_is_better,
        }


#: The headline metrics tested, one per question the pipeline answers. Deliberately a short,
#: explicit list rather than every numeric column found on disk: `forecast/per_sample.csv` alone
#: carries three hundred profile columns, and testing all of them would bury nine real questions
#: under a multiple-comparison correction wide enough to answer none of them.
#:
#: Stated here rather than imported from each analysis's ``GROUPED_METRICS``, because
#: ``analyses/__init__.py``'s rule is that analyses do not import one another -- and because this
#: is a different choice from that one: the grouped variants are drawn, these are *tested*.
METRIC_SOURCES: Tuple[MetricSource, ...] = (
    MetricSource("forecast", "per_sample.csv", "feat_mse_total"),
    MetricSource("forecast", "per_sample.csv", "feat_r2_total", higher_is_better=True),
    MetricSource("uplift", "per_sample.csv", "uplift_rel", higher_is_better=True),
    MetricSource("residual", "per_sample.csv", "residual_ratio", higher_is_better=True),
    MetricSource("latent", "per_sample.csv", "kld_mean", higher_is_better=True),
    MetricSource("calibration", "per_sample.csv", "crps"),
    MetricSource("attention", "per_sample.csv", "argmax_lag"),
    MetricSource("te_lag", "te_lag_mean_per_sample.csv", "kld_mean", higher_is_better=True),
    MetricSource("perm_control", "per_sample.csv", "shuffle_penalty", higher_is_better=True),
)

#: File written into the analysis directory.
RESULT_FILENAME = "cross_subgroup.json"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def holm_adjust(p_values: Sequence[float]) -> List[float]:
    r"""Return the Holm step-down adjusted $p$-values, in the input order.

    $$\tilde{p}_{(i)} = \max_{k \le i}\ \min\!\left((m - k + 1)\,p_{(k)},\ 1\right)$$

    The running maximum is what makes the adjustment *monotone*: without it a later, larger raw
    $p$ could receive a smaller adjusted one and the step-down procedure would reject a hypothesis
    while accepting a more significant one.

    Uniformly more powerful than Bonferroni at the same family-wise error rate, and there is no
    assumption Bonferroni satisfies that Holm does not.

    Args:
        p_values: Raw $p$-values. Non-finite entries are passed through unchanged, so a metric
            whose test could not run does not consume a rank in the correction.

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


def _usable_groups(
    frame: pd.DataFrame, column: str, group_column: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Split one column into its per-group finite samples, dropping groups that are too small.

    Args:
        frame: The per-sample frame.
        column: The metric column.
        group_column: The grouping column.

    Returns:
        ``(usable, excluded)`` -- the groups large enough to test, and the sizes of those that
        were not. The second is returned rather than discarded because "this subgroup had two
        recordings" is the explanation for a missing comparison, and a reader who cannot see it
        will assume the comparison was made.
    """
    usable: Dict[str, np.ndarray] = {}
    excluded: Dict[str, int] = {}
    for group in labels.distinct_groups(list(frame[group_column])):
        values = np.asarray(
            frame.loc[frame[group_column].astype(str) == group, column], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size < MIN_GROUP_SIZE:
            excluded[group] = int(finite.size)
            continue
        usable[group] = finite
    return usable, excluded


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

    Called only for metrics whose omnibus test survived Holm -- see the module docstring for why
    that ordering matters.

    Args:
        samples: Group to its finite values.

    Returns:
        One record per unordered pair, carrying the test, its $p$-value, the effect size and its
        magnitude label. The delta's sign is oriented ``left`` against ``right``: positive means
        the left group's values run higher.
    """
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


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------
def load_metric_frames(
    output_dir: Any, sources: Sequence[MetricSource] = METRIC_SOURCES
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """Load the per-sample CSVs the requested metrics live in.

    Args:
        output_dir: The run's results directory.
        sources: The metrics to read.

    Returns:
        ``(frames, missing)`` -- the loaded frames keyed by relative path, and a record per source
        whose file or column was absent. Absent is recorded, never raised: this analysis is
        designed to run against a partial run directory, and a source that is missing because its
        analysis was skipped is information rather than an error.
    """
    root = Path(output_dir)
    frames: Dict[str, pd.DataFrame] = {}
    missing: List[Dict[str, Any]] = []

    for source in sources:
        key = f"{source.analysis}/{source.filename}"
        if key not in frames:
            path = root / source.analysis / source.filename
            if not path.is_file():
                missing.append({**source.as_dict(), "reason": f"{key} was not written"})
                continue
            frames[key] = pd.read_csv(path)
        if source.column not in frames[key].columns:
            missing.append({
                **source.as_dict(),
                "reason": f"{key} carries no {source.column!r} column",
            })
    return frames, missing


def analyse_metrics(
    output_dir: Any,
    *,
    group_column: str = labels.SUBGROUP_COLUMN,
    alpha: float = DEFAULT_ALPHA,
    sources: Sequence[MetricSource] = METRIC_SOURCES,
) -> Dict[str, Any]:
    """Run the whole three-layer procedure over a finished run directory.

    Args:
        output_dir: The run's results directory.
        group_column: The grouping column. Defaults to the subgroup axis; the class axis works
            identically and is what a caller passes to run the same procedure over clinical class.
        alpha: Family-wise error rate.
        sources: The metrics to test.

    Returns:
        The complete record: the omnibus tests with their Holm-adjusted $p$-values, the pairwise
        comparisons for the metrics that survived, and the inference path behind every coefficient.
    """
    frames, missing = load_metric_frames(output_dir, sources)

    omnibus: List[Dict[str, Any]] = []
    samples_by_metric: Dict[str, Dict[str, np.ndarray]] = {}
    for source in sources:
        key = f"{source.analysis}/{source.filename}"
        frame = frames.get(key)
        if frame is None or source.column not in frame.columns:
            continue
        if group_column not in frame.columns:
            missing.append({
                **source.as_dict(),
                "reason": f"{key} carries no {group_column!r} column",
            })
            continue

        usable, excluded = _usable_groups(frame, source.column, group_column)
        samples_by_metric[source.name] = usable
        record = kruskal_across_groups(usable)
        record.update({
            "metric": source.name,
            "source": source.as_dict(),
            "groups_excluded_as_too_small": excluded,
            "min_group_size": MIN_GROUP_SIZE,
        })
        omnibus.append(record)

    adjusted = holm_adjust([record["p_value"] for record in omnibus])
    for record, value in zip(omnibus, adjusted):
        record["p_holm"] = float(value)
        record["alpha"] = float(alpha)
        record["correction"] = "holm"
        record["n_tests_in_family"] = sum(
            1 for item in omnibus if np.isfinite(item["p_value"])
        )
        record["significant"] = bool(np.isfinite(value) and value < float(alpha))

    pairwise: Dict[str, List[Dict[str, Any]]] = {}
    for record in omnibus:
        if not record["significant"]:
            continue
        pairwise[record["metric"]] = pairwise_comparisons(samples_by_metric[record["metric"]])

    significant = [record["metric"] for record in omnibus if record["significant"]]
    return {
        "group_column": group_column,
        "alpha": float(alpha),
        "omnibus": omnibus,
        "pairwise": pairwise,
        "significant_metrics": significant,
        "n_metrics_tested": sum(1 for record in omnibus if np.isfinite(record["p_value"])),
        "missing_sources": missing,
        "method": (
            "Kruskal-Wallis across groups per metric; Holm step-down correction across the "
            "metrics in the family; pairwise two-sided Mann-Whitney U with Cliff's delta for the "
            "metrics significant after Holm only. Non-parametric throughout because these "
            "distributions are skewed and heavy-tailed. Groups with fewer than "
            f"{MIN_GROUP_SIZE} finite values are excluded and recorded rather than entered."
        ),
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------
def _significance_table(record: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the omnibus results into one row per metric."""
    return pd.DataFrame([
        {
            "metric": item["metric"],
            "analysis": item["source"]["analysis"],
            "column": item["source"]["column"],
            "file": item["source"]["file"],
            "n_groups": item["n_groups"],
            "n_total": sum(item["n_per_group"].values()),
            "statistic": item["statistic"],
            "p_value": item["p_value"],
            "p_holm": item.get("p_holm", float("nan")),
            "correction": item.get("correction", "holm"),
            "alpha": item.get("alpha", float("nan")),
            "significant": item.get("significant", False),
        }
        for item in record["omnibus"]
    ])


def _pairwise_table(record: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the pairwise comparisons into one row per (metric, pair)."""
    rows: List[Dict[str, Any]] = []
    for metric, comparisons in record["pairwise"].items():
        for item in comparisons:
            rows.append({"metric": metric, **item})
    return pd.DataFrame(rows)


def _write_figure(record: Dict[str, Any], directory: Path) -> Optional[str]:
    r"""Emit the significance and effect-size overview.

    Two panels because the two questions are different and are routinely confused. The upper is
    $-\log_{10}$ of the Holm-adjusted $p$ per metric against the $\alpha$ line: *is there
    anything there*. The lower is the Cliff's delta of every surviving pair: *does it matter*.
    A metric can clear the first and be negligible on the second, which at eight subgroups is the
    common case rather than the exceptional one.

    Args:
        record: The analysis record.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    table = _significance_table(record)
    pairs = _pairwise_table(record)

    figure, axes = figures.new_figure(2, height_per_row=3.2)
    try:
        finite = (
            table.loc[np.isfinite(table["p_holm"].to_numpy(dtype=np.float64))]
            if len(table) else table
        )
        if len(finite):
            # Floored so a p of exactly 0 -- which a permutation-free rank test can return at
            # large n -- does not become an infinite bar that rescales the whole axis.
            heights = -np.log10(np.clip(finite["p_holm"].to_numpy(dtype=np.float64), 1e-300, 1.0))
            positions = np.arange(len(finite))
            axes[0, 0].barh(positions, heights, color=figures.COLOR_BLUE, alpha=0.85)
            axes[0, 0].set_yticks(positions)
            axes[0, 0].set_yticklabels(list(finite["metric"]), fontsize=6)
            axes[0, 0].axvline(
                -np.log10(float(record["alpha"])), color=figures.COLOR_VERMILLION,
                linestyle="--", linewidth=1.2,
                label=f"alpha = {float(record['alpha']):g} (Holm-adjusted)",
            )
            axes[0, 0].legend(fontsize=7, loc="best")
            axes[0, 0].invert_yaxis()
        else:
            axes[0, 0].text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=axes[0, 0].transAxes,
                ha="center", va="center", fontsize=9, color=figures.COLOR_GRAY,
            )
        axes[0, 0].set_title(
            f"Omnibus significance by {record['group_column']} (Kruskal-Wallis, Holm)"
        )
        axes[0, 0].set_xlabel("$-\\log_{10}$ Holm-adjusted $p$")
        figures.style_axes(axes[0, 0])

        metrics = sorted(record["pairwise"])
        if metrics and len(pairs):
            pair_labels = sorted({f"{row.left} vs {row.right}" for row in pairs.itertuples()})
            field = np.full((len(metrics), len(pair_labels)), np.nan)
            for row in pairs.itertuples():
                field[
                    metrics.index(row.metric), pair_labels.index(f"{row.left} vs {row.right}")
                ] = row.cliffs_delta
            figures.heatmap_with_colorbar(
                figure, axes[1, 0], field,
                title="Cliff's delta for the pairs of every surviving metric",
                xlabel="", ylabel="", symmetric=True, colorbar_label="Cliff's delta",
            )
            figures.label_rows(axes[1, 0], metrics)
            axes[1, 0].set_xticks(np.arange(len(pair_labels)))
            axes[1, 0].set_xticklabels(pair_labels, rotation=30, ha="right", fontsize=5)
        else:
            figures.heatmap_with_colorbar(
                figure, axes[1, 0], np.zeros((0, 0)),
                title="Cliff's delta (no metric survived Holm)",
                symmetric=True, colorbar_label="Cliff's delta",
            )
        return str(figures.render_to_pdf(figure, directory / "cross_subgroup.pdf"))
    finally:
        figures.plt.close(figure)


def run_cross_subgroup(
    output_dir: Any,
    *,
    group_column: str = labels.SUBGROUP_COLUMN,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Run the analysis against a finished run directory, with no model and no loader.

    Args:
        output_dir: The run's results directory, holding the other analyses' per-sample CSVs.
        group_column: The grouping column.
        alpha: Family-wise error rate.

    Returns:
        The headline summary, or a ``skipped`` record when the run holds fewer than two testable
        groups -- which is the ordinary outcome on the single-file pretraining split.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    record = analyse_metrics(output_dir, group_column=group_column, alpha=alpha)

    if record["n_metrics_tested"] == 0:
        # Below two groups, or no source produced a usable column. Either way there is nothing to
        # compare and an empty table would read as "no differences found", which is a claim this
        # run cannot make.
        reason = (
            "no metric had two groups of at least "
            f"{MIN_GROUP_SIZE} samples on the {group_column!r} axis"
        )
        logger.warning(f"cross_subgroup: skipped -- {reason}")
        return {
            "skipped": True,
            "reason": reason,
            "group_column": group_column,
            "missing_sources": record["missing_sources"],
        }

    directory.mkdir(parents=True, exist_ok=True)
    _significance_table(record).to_csv(directory / "significance.csv", index=False)
    pairwise = _pairwise_table(record)
    pairwise.to_csv(directory / "pairwise.csv", index=False)
    with open(directory / RESULT_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(record), handle, indent=2)
    figure_path = _write_figure(record, directory)

    summary = {
        "skipped": False,
        "group_column": group_column,
        "alpha": float(alpha),
        "n_metrics_tested": record["n_metrics_tested"],
        "significant_metrics": record["significant_metrics"],
        "n_significant": len(record["significant_metrics"]),
        "n_pairwise_comparisons": int(len(pairwise)),
        "largest_effects": _largest_effects(pairwise),
        "missing_sources": record["missing_sources"],
        "method": record["method"],
        "figures": [figure_path] if figure_path else [],
    }
    logger.info(
        f"cross_subgroup: {summary['n_significant']} of {summary['n_metrics_tested']} metric(s) "
        f"differ across {group_column} after Holm; {summary['n_pairwise_comparisons']} pairwise "
        f"comparison(s)"
    )
    return summary


def run_cross_subgroup_analysis(
    runner: Any = None,
    loader: Any = None,
    *,
    eval_config: Optional[Dict[str, Any]] = None,
    output_dir: Any = None,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Registry adapter, so the orchestrator can call this like any other analysis.

    ``runner`` and ``loader`` are accepted and **not used**. That is the point rather than an
    oversight: the analysis reads the CSVs the earlier steps wrote, so it re-runs against a
    finished run directory with ``--only cross_subgroup`` and no forward pass at all.

    Args:
        runner: Ignored.
        loader: Ignored.
        eval_config: Ignored. Nothing here is configurable: :data:`DEFAULT_ALPHA` and
            :data:`MIN_GROUP_SIZE` are properties of the statistical procedure, not of a run, and
            an operator who could lower either could make any metric significant.
        output_dir: The run's results directory.
        probe: Ignored.

    Returns:
        The headline summary for ``summary.json``.
    """
    return run_cross_subgroup(output_dir)


def _largest_effects(pairwise: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
    """Return the pairs with the largest absolute effect size, for the summary.

    Args:
        pairwise: The flattened pairwise table.
        limit: How many to report.

    Returns:
        The strongest comparisons, largest first. Ranked by $|\\delta|$ rather than by $p$: at
        eight subgroups the smallest $p$ is usually the largest pair, not the largest difference.
    """
    if not len(pairwise) or "cliffs_delta" not in pairwise:
        return []
    finite = pairwise[np.isfinite(pairwise["cliffs_delta"])]
    if not len(finite):
        return []
    ordered = finite.reindex(
        finite["cliffs_delta"].abs().sort_values(ascending=False).index
    )
    return [
        {
            "metric": row.metric,
            "left": row.left,
            "right": row.right,
            "cliffs_delta": float(row.cliffs_delta),
            "magnitude": row.magnitude,
            "p_value": float(row.p_value),
        }
        for row in ordered.head(int(limit)).itertuples()
    ]


def _json_safe(value: Any) -> Any:
    """Convert the record for ``json.dump``, deferring to the report module's converter."""
    from teb_vae.lag_attn.eval.report import json_safe

    return json_safe(value)
