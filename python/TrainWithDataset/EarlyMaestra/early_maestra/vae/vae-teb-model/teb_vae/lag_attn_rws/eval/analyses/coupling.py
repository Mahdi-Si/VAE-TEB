r"""What the source added, per recording, with the uncertainty on it.

Two numbers carry the whole claim of this architecture, and both are reported here on the unit the
question is asked in -- one recording, one observation.

**``pred_gap``** $= D_{\mathrm{base}} - D_{\mathrm{full}}$, in nats per anchor: how many nats the
source-conditioned forecast saves over the target-only one. It comes in two flavours and they are
never merged. The **headline** is the Monte Carlo marginalised difference,
$D = -[\operatorname{logsumexp}_r(-D_r) - \log K]$, the log of the average likelihood over $K$
latent draws. The **training-path** column is one draw scored through the objective's own
functions, and sits beside it as the parity check -- the two agree to the extent that $K$ draws
were enough, and their difference is the cost of the marginalisation rather than a second answer.

**``source_conditioned_kl_raw``** $= \bar K$, the unfloored KL between the two latents. Unfloored
because only that value is a rate: ``source_conditioned_kl_train`` has free bits applied per
dimension per step before summing, so it exceeds the raw value by construction and hides a
collapsed source pathway. It is reported here as a *description*, not as evidence of coupling --
it is inflated by an arbitrary factor whenever the prior variance sits on its clamp, which is what
the latent analysis's own verdict exists to catch, and unlike ``pred_gap`` it says nothing about
whether the forecast improved.

**The denominator is visible everywhere.** ``np.nan > 0`` is ``False``, so a recording that scored
no anchors -- and therefore measured nothing at all -- would otherwise be counted silently as
evidence *against* a positive gap. Every fraction here carries the count it was computed over and
how many recordings were dropped for carrying no finite value.

**Uncertainty is on every comparison.** A ``pred_gap`` of $10^{-6}$ nats over three recordings and
one of fifty nats over two thousand are not the same finding, and a point estimate renders them
identically. So: a percentile bootstrap over *recordings*, and a paired signed-rank test on the
per-recording block scores, which is the paired form because each recording contributes both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    positive_fraction,
    scored_sample_count,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "coupling"

#: What it writes.
PER_RECORDING_FILENAME = "coupling_per_recording.csv"
SUMMARY_FILENAME = "coupling_summary.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
DISTRIBUTION_FIGURE = "pred_gap_distribution.pdf"

#: The two ``pred_gap`` estimators, each with the path it was computed on. The label travels with
#: the column into every row this analysis writes: the block score difference is the same
#: subtraction on two different estimators of the same quantity, and a table carrying both under
#: one name would be unreadable.
PRED_GAP_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    (
        "pred_gap_mc_nats",
        "mc_pred_gap",
        "Monte Carlo marginalised: D = -[logsumexp_r(-D_r) - log K], the log of the average "
        "likelihood over K latent draws",
    ),
    (
        "pred_gap_train_path_nats",
        "pred_gap",
        "single-draw training path: the objective's own reduction, reported as the parity column",
    ),
)

#: The block scores whose difference each ``pred_gap`` is, for the paired test.
_PAIRED_SCORES: Dict[str, Tuple[str, str]] = {
    "pred_gap_mc_nats": ("mc_nll_base_block", "mc_nll_full_block"),
    "pred_gap_train_path_nats": ("nll_base_block", "nll_full_block"),
}

#: The KL columns reported beside the gap: the matched pairing, and the negative control's.
KL_COLUMNS: Tuple[str, ...] = (
    "source_conditioned_kl_raw",
    "source_conditioned_kl_shuffled_raw",
)

#: Every per-sample column this analysis reduces per recording.
VALUE_COLUMNS: Tuple[str, ...] = (
    "mc_pred_gap",
    "pred_gap",
    "mc_nll_base_block",
    "mc_nll_full_block",
    "nll_base_block",
    "nll_full_block",
    *KL_COLUMNS,
)

#: The subset worth resolving by cohort. Both ``pred_gap`` estimators and the KL beside them, not
#: the four block scores those two gaps are the differences of: a grouped figure with eight panels
#: is one nobody reads, and the block scores are dominated by how forecastable a recording is
#: rather than by what the source added to it.
GROUPED_METRICS: Tuple[str, ...] = ("mc_pred_gap", "pred_gap", *KL_COLUMNS)


def build_gap_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise each ``pred_gap`` estimator over the recordings, with its uncertainty.

    Args:
        per_guid: Per-recording means of the per-sample table.
        resamples: Bootstrap resamples, from ``eval_config.bootstrap_resamples``.
        seed: Bootstrap seed, from ``eval_config.seed``, so the interval is reproducible from the
            summary alone.

    Returns:
        One row per estimator, carrying the mean and its interval, the quartiles, the fraction of
        recordings on which the gap is positive **with its denominator**, and the paired
        signed-rank test on the two block scores the gap is the difference of.
    """
    rows: List[Dict[str, Any]] = []
    for name, column, path in PRED_GAP_COLUMNS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        positive = positive_fraction(values)
        base_column, full_column = _PAIRED_SCORES[name]
        paired = shared_stats.wilcoxon_paired(
            finite_column(per_guid, base_column),
            finite_column(per_guid, full_column),
            label_left="target-only block score",
            label_right="source-conditioned block score",
        )
        rows.append(
            {
                "metric": name,
                "source_column": column,
                "score_path": path,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "ci_method": interval["method"],
                "bootstrap_resamples": int(interval["resamples"]),
                # The fraction and the count it came from, never one without the other.
                "positive_fraction": positive["fraction"],
                "n_positive": positive["n_positive"],
                "n_recordings_scored": positive["n"],
                "n_recordings_dropped_not_finite": positive["n_dropped_not_finite"],
                "wilcoxon_p_value": paired["p_value"],
                "wilcoxon_n_pairs": paired["n_pairs"],
                "wilcoxon_median_difference": paired["median_difference"],
            }
        )
    return rows


def build_kl_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise the KL readouts over the recordings, labelled for what they are.

    Args:
        per_guid: Per-recording means.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per KL column, each carrying the note that the value is unfloored and that it is
        a description of the latent rather than evidence that the forecast improved.
    """
    rows: List[Dict[str, Any]] = []
    for column in KL_COLUMNS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        rows.append(
            {
                "metric": column,
                "source_column": column,
                "score_path": (
                    "unfloored KL between the two latents; a rate only because free bits are not "
                    "applied to it"
                    if column == "source_conditioned_kl_raw"
                    else "unfloored KL against a *stranger's* source, the negative control"
                ),
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "ci_method": interval["method"],
                "bootstrap_resamples": int(interval["resamples"]),
            }
        )
    return rows


def build_distribution_figure(
    per_guid: pd.DataFrame, gap_rows: Sequence[Dict[str, Any]]
) -> Any:
    """Draw the per-recording ``pred_gap`` distribution with its interval, and both estimators.

    Two panels rather than one. The histogram answers "how many recordings did the source help,
    and by how much" -- with zero marked, because the sign is the finding and a distribution
    straddling it is a different result from one sitting above it. The violin puts the two
    estimators side by side under their own names, so the marginalisation's cost is visible rather
    than being a difference between two figures.

    Args:
        per_guid: Per-recording means.
        gap_rows: The summary rows, read for the headline estimator's interval.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    headline = next(iter(gap_rows), {})
    axis = axes[0, 0]
    figures.histogram_panel(
        axis,
        finite_column(per_guid, PRED_GAP_COLUMNS[0][1]),
        title=(
            f"pred_gap per recording, n = {int(headline.get('n_recordings_scored') or 0)} "
            f"(Monte Carlo marginalised)"
        ),
        xlabel="nats per anchor",
        reference=0.0,
        reference_label="no improvement",
    )
    low, high = headline.get("ci_lo"), headline.get("ci_hi")
    if low is not None and high is not None and np.isfinite(low) and np.isfinite(high):
        # The interval on the *mean*, drawn over the distribution of the values it summarises.
        # Shaded rather than plotted as a point range so it cannot be misread as a data range.
        axis.axvspan(
            float(low), float(high), color=figures.COLOR_ORANGE, alpha=0.25, zorder=0,
            label=f"95% CI of the mean [{float(low):.4g}, {float(high):.4g}]",
        )
        axis.legend(fontsize=7, loc="best")

    figures.violin_panel(
        axes[1, 0],
        {name: finite_column(per_guid, column) for name, column, _ in PRED_GAP_COLUMNS},
        title="pred_gap per recording, by estimator",
        ylabel="nats per anchor",
        reference=0.0,
        reference_label="no improvement",
    )
    return figure


def run_coupling_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report both coupling readouts per recording, with intervals and honest denominators.

    Args:
        context: The analysis context, read for the collected per-sample table.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population this analysis describes is the
            set of recordings that scored an anchor, which only the table knows.

    Returns:
        The protocol's keys plus the two summary tables and the paths written.
    """
    per_sample = context.collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    gap_rows = build_gap_rows(per_guid, resamples=resamples, seed=seed)
    kl_rows = build_kl_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(gap_rows + kl_rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    figure_name = str(
        figures.render_to_pdf(
            build_distribution_figure(per_guid, gap_rows), directory / DISTRIBUTION_FIGURE
        ).name
    )
    return {
        "n_samples": scored_sample_count(per_sample, "mc_pred_gap"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "pred_gap": gap_rows,
        "kl": kl_rows,
        # Declared, not emitted: the by-class and by-subgroup variants are the runner's fan-out
        # over this frame, which carries one row per recording and the cohort each belongs to.
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [PER_RECORDING_FILENAME, SUMMARY_FILENAME, figure_name],
    }
