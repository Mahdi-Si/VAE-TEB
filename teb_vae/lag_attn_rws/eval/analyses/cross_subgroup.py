r"""Do the cohorts actually differ, or does the by-subgroup table only look as though they do?

A by-subgroup table invites exactly one question and cannot answer it. Eight cohorts, each with a
mean, always produce a highest and a lowest; with eight headline metrics that is sixty-four
numbers, and *some* of them will look separated whether or not anything is there. This analysis is
the arithmetic that says which of those separations survive being asked properly.

**Three layers, in this order, and the order is the point.**

* **Kruskal-Wallis**, per metric, across the cohorts: is there any difference at all?
  Non-parametric, because these distributions are routinely skewed and heavy-tailed and a one-way
  ANOVA's normality assumption is not one this data supports.
* **Holm across the metrics**, which form one family. Eight tests at $\alpha = 0.05$ produce a
  false positive about a third of the time by construction. Holm rather than Bonferroni because
  it is uniformly more powerful at the same family-wise error rate.
* **Pairwise Mann-Whitney with Cliff's delta**, for the metrics that survived Holm **only**.
  Running $\binom{8}{2} = 28$ pairwise tests on a metric whose omnibus found nothing is the
  multiple-comparison problem with extra steps.

Cliff's delta comes back with every pair because a $p$-value is not an effect size. At the
eight-subgroup scale a difference of no clinical consequence reaches significance readily, and
$\delta$ -- the probability that a random member of one cohort exceeds a random member of the
other, rescaled to $[-1, 1]$ -- is what says whether the cohorts actually separate.

**Every vector tested holds one value per recording.** The sources are the *per-recording* tables
the other analyses wrote, not the per-sample table: one recording contributes up to thirty-seven
overlapping segments, so a test over segments is pseudo-replicated by that factor and its
$p$-values are anticonservative by an amount nothing in the output would show.

**This analysis needs no model, and no analysis needs to have run in this pass.** It reads CSVs
off disk, so it re-runs against a finished run directory in seconds:

.. code-block:: bash

    python -m teb_vae.lag_attn_rws.eval.run --only cross_subgroup --output-dir <a finished run>

That is also why a source that is absent is **recorded** rather than raised: the dependency is on
files existing, not on analyses having run, and a run with ``--only`` selecting two analyses
legitimately leaves the rest of the sources missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_rws.eval.report_seam import json_safe

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "cross_subgroup"

#: What it writes.
SIGNIFICANCE_FILENAME = "cross_subgroup_significance.csv"
PAIRWISE_FILENAME = "cross_subgroup_pairwise.csv"
RESULT_FILENAME = "cross_subgroup.json"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
HEATMAP_FIGURE = "subgroup_heatmap.pdf"

#: Family-wise error rate the Holm correction controls. Deliberately not an ``eval_config`` key:
#: an operator who could raise it could make any metric significant.
DEFAULT_ALPHA = 0.05

#: How many of the strongest comparisons the summary carries. The full set is on the pairwise CSV.
LARGEST_EFFECTS = 5


@dataclass(frozen=True)
class MetricSource:
    """Where one headline metric is read from.

    Attributes:
        analysis: The analysis subdirectory holding the CSV.
        filename: The CSV within it.
        column: The per-recording column to test.
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
        """Return the source as a record -- the inference path behind every coefficient below."""
        return {
            "analysis": self.analysis,
            "file": f"{self.analysis}/{self.filename}",
            "column": self.column,
            "higher_is_better": self.higher_is_better,
        }


#: The metrics tested, one per question this pipeline answers. Deliberately a short explicit list
#: rather than every numeric column found on disk: the per-recording tables carry dozens between
#: them, and testing all of them would bury the real questions under a multiple-comparison
#: correction wide enough to answer none of them.
#:
#: Stated here rather than imported from each analysis, because an analysis may not import another
#: and because this is a different choice from the grouped variants: those are *drawn*, these are
#: **tested**.
METRIC_SOURCES: Tuple[MetricSource, ...] = (
    MetricSource("coupling", "coupling_per_recording.csv", "mc_pred_gap", higher_is_better=True),
    MetricSource(
        "coupling", "coupling_per_recording.csv", "source_conditioned_kl_raw",
        higher_is_better=True,
    ),
    MetricSource("forecast", "forecast_scores.csv", "sq_error_full"),
    MetricSource("forecast", "forecast_scores.csv", "nll_full_block"),
    MetricSource("latent", "latent_per_recording.csv", "logvar_prior_floor_frac"),
    MetricSource(
        "attention", "attention_per_recording.csv", "attention_entropy_nats",
        higher_is_better=True,
    ),
    MetricSource("residual", "residual_per_recording.csv", "forecast_difference_sq"),
    MetricSource(
        "calibration", "calibration_per_recording.csv", "mean_logvar_full"
    ),
)

#: Written into every record, so a $p$-value here is readable without this module.
METHOD = (
    "Kruskal-Wallis across cohorts per metric over one value per recording; Holm step-down "
    "correction across the metrics in the family; pairwise two-sided Mann-Whitney U with Cliff's "
    "delta for the metrics significant after Holm only. Non-parametric throughout because these "
    "distributions are skewed and heavy-tailed. Cohorts with fewer than "
    f"{shared_stats.MIN_GROUP_SIZE} finite recordings are excluded and recorded rather than "
    "entered."
)


# =============================================================================
# Reading a finished run
# =============================================================================
def load_metric_frames(
    output_dir: Any, sources: Sequence[MetricSource] = METRIC_SOURCES
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """Load the per-recording CSVs the requested metrics live in.

    Args:
        output_dir: The run's results directory.
        sources: The metrics to read.

    Returns:
        ``(frames, missing)`` -- the frames keyed by relative path, and one record per source
        whose file or column was absent. Absent is recorded, never raised: this analysis is built
        to run against a partial run directory, and a source missing because its analysis was
        skipped is information rather than an error.
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
            missing.append(
                {**source.as_dict(), "reason": f"{key} carries no {source.column!r} column"}
            )
    return frames, missing


def usable_groups(
    frame: pd.DataFrame, column: str, group_column: str
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Split one column into its per-cohort finite recordings, dropping cohorts too small to test.

    Args:
        frame: A per-recording frame carrying the cohort column.
        column: The metric column.
        group_column: The cohort axis.

    Returns:
        ``(usable, excluded)`` -- the cohorts large enough to test, and the sizes of those that
        were not. The second is returned rather than discarded because "this subgroup had two
        recordings" is the explanation for a missing comparison, and a reader who cannot see it
        will assume the comparison was made.
    """
    usable: Dict[str, np.ndarray] = {}
    excluded: Dict[str, int] = {}
    if group_column not in frame.columns or column not in frame.columns:
        return usable, excluded
    for group in labels.distinct_groups(list(frame[group_column])):
        values = np.asarray(
            frame.loc[frame[group_column].astype(str) == group, column], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size < shared_stats.MIN_GROUP_SIZE:
            excluded[group] = int(finite.size)
            continue
        usable[group] = finite
    return usable, excluded


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
        group_column: The cohort axis. The subgroup axis by default; the class axis works
            identically and is what a caller passes to run the same procedure over clinical class.
        alpha: Family-wise error rate.
        sources: The metrics to test.

    Returns:
        The complete record: the omnibus tests with their Holm-adjusted $p$-values, the pairwise
        comparisons for the metrics that survived, the sizes every test ran on, and the inference
        path behind every coefficient.
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
            missing.append(
                {**source.as_dict(), "reason": f"{key} carries no {group_column!r} column"}
            )
            continue

        usable, excluded = usable_groups(frame, source.column, group_column)
        samples_by_metric[source.name] = usable
        record = shared_stats.kruskal_across_groups(usable)
        record.update(
            {
                "metric": source.name,
                "source": source.as_dict(),
                "groups_excluded_as_too_small": excluded,
                "min_group_size": shared_stats.MIN_GROUP_SIZE,
                # The unit, recorded beside every test: these are recordings, never segments.
                "unit": "recording",
            }
        )
        omnibus.append(record)

    adjusted = shared_stats.holm_adjust([record["p_value"] for record in omnibus])
    n_tested = sum(1 for record in omnibus if np.isfinite(record["p_value"]))
    for record, value in zip(omnibus, adjusted):
        record["p_holm"] = float(value)
        record["alpha"] = float(alpha)
        record["correction"] = "holm"
        record["n_tests_in_family"] = n_tested
        record["significant"] = bool(np.isfinite(value) and value < float(alpha))

    pairwise = {
        record["metric"]: shared_stats.pairwise_comparisons(samples_by_metric[record["metric"]])
        for record in omnibus
        if record["significant"]
    }
    return {
        "group_column": group_column,
        "alpha": float(alpha),
        "omnibus": omnibus,
        "pairwise": pairwise,
        "significant_metrics": [record["metric"] for record in omnibus if record["significant"]],
        "n_metrics_tested": n_tested,
        "missing_sources": missing,
        "method": METHOD,
    }


# =============================================================================
# Emission
# =============================================================================
def significance_frame(record: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the omnibus results into one row per metric."""
    return pd.DataFrame(
        [
            {
                "metric": item["metric"],
                "analysis": item["source"]["analysis"],
                "column": item["source"]["column"],
                "file": item["source"]["file"],
                "higher_is_better": item["source"]["higher_is_better"],
                "unit": item["unit"],
                "n_groups": item["n_groups"],
                "n_recordings": sum(item["n_per_group"].values()),
                "statistic": item["statistic"],
                "p_value": item["p_value"],
                "p_holm": item.get("p_holm", float("nan")),
                "correction": item.get("correction", "holm"),
                "alpha": item.get("alpha", float("nan")),
                "significant": item.get("significant", False),
            }
            for item in record["omnibus"]
        ],
        columns=[
            "metric", "analysis", "column", "file", "higher_is_better", "unit", "n_groups",
            "n_recordings", "statistic", "p_value", "p_holm", "correction", "alpha",
            "significant",
        ],
    )


def pairwise_frame(record: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the pairwise comparisons into one row per (metric, pair)."""
    rows = [
        {"metric": metric, **item}
        for metric, comparisons in record["pairwise"].items()
        for item in comparisons
    ]
    return pd.DataFrame(rows)


def build_heatmap_figure(record: Dict[str, Any]) -> Any:
    r"""Draw the significance of every metric and the effect sizes that survived it.

    Two panels because the two questions are different and are routinely confused. The upper is
    $-\log_{10}$ of the Holm-adjusted $p$ per metric against the $\alpha$ line: *is there anything
    there*. The lower is Cliff's delta for every surviving pair: *does it matter*. A metric can
    clear the first and be negligible on the second, which at eight cohorts is the common case.

    Args:
        record: The analysis record.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2, height_per_row=3.2)
    table = significance_frame(record)
    finite = (
        table.loc[np.isfinite(table["p_holm"].to_numpy(dtype=np.float64))]
        if len(table) else table
    )
    axis = axes[0, 0]
    if len(finite):
        # Floored so a p of exactly zero -- which a rank test can return at large n -- does not
        # become an infinite bar that rescales the whole axis.
        heights = -np.log10(np.clip(finite["p_holm"].to_numpy(dtype=np.float64), 1e-300, 1.0))
        positions = np.arange(len(finite))
        axis.barh(positions, heights, color=figures.COLOR_BLUE, alpha=0.85)
        axis.set_yticks(positions)
        axis.set_yticklabels(list(finite["metric"]), fontsize=6)
        axis.axvline(
            -np.log10(float(record["alpha"])), color=figures.COLOR_VERMILLION,
            linestyle="--", linewidth=1.2,
            label=f"alpha = {float(record['alpha']):g} (Holm-adjusted)",
        )
        axis.legend(fontsize=7, loc="best")
        axis.invert_yaxis()
    else:
        axis.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=axis.transAxes,
            ha="center", va="center", fontsize=9, color=figures.COLOR_GRAY,
        )
    axis.set_title(f"Omnibus significance by {record['group_column']} (Kruskal-Wallis, Holm)")
    axis.set_xlabel("$-\\log_{10}$ Holm-adjusted $p$")
    figures.style_axes(axis)

    _draw_effect_heatmap(figure, axes[1, 0], record)
    return figure


def _draw_effect_heatmap(figure: Any, ax: Any, record: Dict[str, Any]) -> None:
    """Draw Cliff's delta for every pair of every metric that survived Holm.

    Args:
        figure: The parent figure, for the colourbar.
        ax: Target axes.
        record: The analysis record.
    """
    pairs = pairwise_frame(record)
    metrics = sorted(record["pairwise"])
    if not metrics or not len(pairs):
        figures.heatmap_with_colorbar(
            figure, ax, np.zeros((0, 0)),
            title="Cliff's delta (no metric survived Holm)",
            symmetric=True, colorbar_label="Cliff's delta",
        )
        return

    labels_x = sorted({f"{row.left} vs {row.right}" for row in pairs.itertuples()})
    field = np.full((len(metrics), len(labels_x)), np.nan)
    for row in pairs.itertuples():
        field[metrics.index(row.metric), labels_x.index(f"{row.left} vs {row.right}")] = (
            row.cliffs_delta
        )
    figures.heatmap_with_colorbar(
        figure, ax, field, title="Cliff's delta for the pairs of every surviving metric",
        symmetric=True, colorbar_label="Cliff's delta", interpolation="none",
    )
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics, fontsize=6)
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right", fontsize=5)


def largest_effects(pairs: pd.DataFrame, limit: int = LARGEST_EFFECTS) -> List[Dict[str, Any]]:
    r"""Return the pairs with the largest absolute effect size, for the summary.

    Ranked by $|\delta|$ rather than by $p$: at eight cohorts the smallest $p$ is usually the
    largest pair, not the largest difference.

    Args:
        pairs: The flattened pairwise table.
        limit: How many to report.

    Returns:
        The strongest comparisons, largest first.
    """
    if not len(pairs) or "cliffs_delta" not in pairs:
        return []
    finite = pairs[np.isfinite(pairs["cliffs_delta"])]
    if not len(finite):
        return []
    ordered = finite.reindex(finite["cliffs_delta"].abs().sort_values(ascending=False).index)
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


def run_cross_subgroup_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Test the headline metrics across the cohorts, reading the tables the run already wrote.

    Args:
        context: The analysis context. Unused: every input is a CSV on disk, which is what makes
            an offline re-run against a finished directory work with no model at all.
        eval_config: The validated block. Unused: neither the significance level nor the minimum
            cohort size is configurable -- both are properties of the procedure, and an operator
            who could lower either could make any metric significant.
        output_dir: The run's results directory, holding the other analyses' per-recording CSVs.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the omnibus outcome, the strongest surviving effects, and the
        sources that were absent. A recorded skip when no metric had two testable cohorts, which
        is the ordinary outcome on the single-cohort pretraining split.
    """
    record = analyse_metrics(output_dir)
    if record["n_metrics_tested"] == 0:
        reason = (
            f"no metric had two cohorts of at least {shared_stats.MIN_GROUP_SIZE} recordings on "
            f"the {record['group_column']!r} axis"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {
            "n_samples": None,
            "composition": {},
            "plan": {"capped": False},
            "skipped": True,
            "reason": reason,
            "group_column": record["group_column"],
            "missing_sources": record["missing_sources"],
            "files": [],
        }

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    significance = significance_frame(record)
    significance.to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    pairs = pairwise_frame(record)
    pairs.to_csv(directory / PAIRWISE_FILENAME, index=False)
    with open(directory / RESULT_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(json_safe(record), handle, indent=2)
    figure_name = str(
        figures.render_to_pdf(build_heatmap_figure(record), directory / HEATMAP_FIGURE).name
    )

    n_recordings = int(significance["n_recordings"].max()) if len(significance) else 0
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(record['significant_metrics'])} of "
        f"{record['n_metrics_tested']} metric(s) differ across {record['group_column']} after "
        f"Holm; {len(pairs)} pairwise comparison(s)"
    )
    return {
        # The population is recordings rather than segments, and the coverage block compares
        # sample counts -- so this analysis reports None and stays out of that comparison rather
        # than entering a recording count as though it were a segment count.
        "n_samples": None,
        "composition": {"n_recordings": n_recordings, "group_column": record["group_column"]},
        "plan": {"capped": False},
        "skipped": False,
        "group_column": record["group_column"],
        "alpha": float(record["alpha"]),
        "unit": "recording",
        "n_metrics_tested": record["n_metrics_tested"],
        "significant_metrics": record["significant_metrics"],
        "n_significant": len(record["significant_metrics"]),
        "n_pairwise_comparisons": int(len(pairs)),
        "largest_effects": largest_effects(pairs),
        "missing_sources": record["missing_sources"],
        "method": record["method"],
        "files": [SIGNIFICANCE_FILENAME, PAIRWISE_FILENAME, RESULT_FILENAME, figure_name],
    }
