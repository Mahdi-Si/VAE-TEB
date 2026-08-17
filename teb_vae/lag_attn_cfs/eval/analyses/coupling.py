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

**The same answer as a percentage.** Nats are the objective's units and say nothing about
proportion: whether $3$ nats over a $2940$-coefficient block is a large improvement or a negligible
one is not readable off the number, and two checkpoints whose block scores differ in scale cannot be
compared on it at all. Three percentages are reported beside it, in the two spaces where a ratio
has a natural zero:

* ``pred_gap_rmse_pct`` $= 100\left(1 - \mathrm{RMSE}_{\mathrm{full}}/\mathrm{RMSE}_{\mathrm{base}}
  \right)$ -- the percentage of the point-forecast error the source removed. **Scale-free**, so the
  loader's per-channel $z$-scoring cancels out of it and it is the one readout here that two
  checkpoints normalised against different statistics can still be compared on.
* ``pred_gap_mse_pct`` $= 100\left(1 - \mathrm{MSE}_{\mathrm{full}}/\mathrm{MSE}_{\mathrm{base}}
  \right)$ -- the ``mse_skill`` convention the forecast analysis already reports against the
  trivial baselines, here applied source-versus-no-source.
* ``pred_gap_mc_likelihood_pct``
  $= 100\left(e^{\Delta/(H \cdot C_{\mathrm{keep}})} - 1\right)$ -- the percentage form of the
  headline nats themselves: how much more probability density the source-conditioned forecast puts
  on each observed **target coefficient**. **Under ``gaussian_nll`` only**: under ``mse`` a block
  score is a sum of squared errors rather than a log density, so its exponential is not a density
  ratio, and the column is omitted with its reason rather than emitted with a false unit.

**The likelihood percentage is budget-local, and the emitted record says so.** Its denominator is
$H \cdot C_{\mathrm{keep}}$, and $C_{\mathrm{keep}}$ is whatever the warm-up budget left standing --
$98$ of $102$ declared channels at the shipped ``causal_warmup_budget_steps``, but a different
number under a different budget. So the *same model* re-evaluated under a looser budget would report
a different percentage from the same nats, and two runs' percentages are comparable only where their
block widths are. The nats are not budget-local in that sense and neither are the two error-space
percentages, which is why all three are reported rather than the percentage alone.

**No percentage is a ratio of the two block scores**, and that is a correctness boundary rather
than a stylistic choice. $D_{\mathrm{base}}$ is a negative log *density* summed over
$H \cdot C_{\mathrm{keep}}$ coefficients: it has no natural zero and is legitimately negative for a
sharp forecast, so $\Delta / D_{\mathrm{base}}$ changes sign with its own denominator and is
unbounded near it. The forecast analysis states the same rule for the same reason, which is why its
NLL-space column is a difference rather than $1 -$ a ratio.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    positive_fraction,
    scored_sample_count,
    skill_against,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "coupling"

#: What it writes.
PER_RECORDING_FILENAME = "coupling_per_recording.csv"
SUMMARY_FILENAME = "coupling_summary.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
DISTRIBUTION_FIGURE = "pred_gap_distribution.pdf"
PERCENT_FIGURE = "pred_gap_percent.pdf"

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

#: The two branches' mean squared forecast error, per scored coefficient, base first. Reduced here
#: only because the error-space percentages are ratios of them; the forecast analysis reports the
#: errors themselves, and this one reports what the source did to them.
SQUARED_ERROR_COLUMNS: Tuple[str, str] = ("sq_error_base", "sq_error_full")

#: The likelihood the **likelihood-space** percentage is defined under, and only that one.
#:
#: Exponentiating a block score is meaningful only where the score is a log-density. Under
#: ``'mse'`` it is a plain sum of squared errors -- ``metrics.marginalise_block_scores`` says so and
#: marginalises by averaging rather than by ``logsumexp`` for the same reason -- so
#: $e^{\Delta/(H \cdot C_{\mathrm{keep}})}$ is not a density ratio, is not a probability of
#: anything, and is out by a factor of two even against the most charitable unit-variance reading.
#: The error-space percentages are unaffected: a squared error is a squared error under either
#: likelihood.
LIKELIHOOD_PERCENT_REQUIRES = "gaussian_nll"

#: Every per-sample column this analysis reduces per recording.
VALUE_COLUMNS: Tuple[str, ...] = (
    "mc_pred_gap",
    "pred_gap",
    "mc_nll_base_block",
    "mc_nll_full_block",
    "nll_base_block",
    "nll_full_block",
    *KL_COLUMNS,
    *SQUARED_ERROR_COLUMNS,
)

#: The percentage readouts, each with the space it is measured in. Derived per recording from the
#: columns above rather than accumulated in the collection pass, which is what keeps an offline
#: ``--only coupling`` re-run able to produce them from the tables alone.
#:
#: The arithmetic is written out in :func:`percent_columns` rather than tabled here as callables:
#: three explicit expressions can be checked against the documentation by reading them, and a
#: registry of lambdas cannot.
PERCENT_COLUMNS: Tuple[Tuple[str, str], ...] = (
    (
        "pred_gap_rmse_pct",
        "error space: 100 * (1 - RMSE_full / RMSE_base), the percentage of the point-forecast "
        "error the source removed. Scale-free, so the loader's per-channel z-scoring cancels",
    ),
    (
        "pred_gap_mse_pct",
        "error space: 100 * (1 - MSE_full / MSE_base), the mse_skill convention applied "
        "source-versus-no-source rather than model-versus-baseline",
    ),
    (
        "pred_gap_mc_likelihood_pct",
        "likelihood space: 100 * (exp(mc_pred_gap / (H*C_keep)) - 1), how much more probability "
        "density the source-conditioned forecast puts on each observed target coefficient. "
        "H*C_keep is the fixed block width, not a per-anchor scored-coefficient count, so this "
        "understates the improvement on any anchor with masked forecast steps; and C_keep is "
        "whatever the warm-up budget left standing, so the percentage is budget-local and two "
        "runs' values are comparable only where their block widths are",
    ),
)

#: The two error-space percentages, and the likelihood-space one, named rather than positional.
#: The figure draws the first pair on one axis and the third on its own, and the split is a
#: statement about what shares a scale -- not something a reader should have to infer from the
#: registry's ordering.
ERROR_SPACE_PERCENTS: Tuple[str, str] = ("pred_gap_rmse_pct", "pred_gap_mse_pct")
LIKELIHOOD_SPACE_PERCENT = "pred_gap_mc_likelihood_pct"

#: The one promoted to the histogram and to the by-cohort fan-out. Root-mean-square rather than
#: mean-square for the same reason ``GROUPED_METRICS`` promotes it: the two are one ratio under a
#: root, and the rooted one is the figure a reader can state in a sentence about a trace.
HEADLINE_PERCENT = ERROR_SPACE_PERCENTS[0]

#: The subset worth resolving by cohort. Both ``pred_gap`` estimators and the KL beside them, not
#: the four block scores those two gaps are the differences of: a grouped figure with eight panels
#: is one nobody reads, and the block scores are dominated by how forecastable a recording is
#: rather than by what the source added to it.
#:
#: One percentage joins them, not three. ``pred_gap_mse_pct`` is a monotone restatement of
#: ``pred_gap_rmse_pct`` -- the same recordings in the same order -- so a second panel of it would
#: cost a reader's attention and tell them nothing the first did not.
GROUPED_METRICS: Tuple[str, ...] = (
    "mc_pred_gap",
    "pred_gap",
    *KL_COLUMNS,
    HEADLINE_PERCENT,
)


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


def coefficients_per_anchor(record: Dict[str, Any]) -> Optional[float]:
    r"""$H \cdot C_{\mathrm{keep}}$, the coefficients one anchor's forecast block covers, or ``None``.

    Read off the collection record's geometry block rather than assumed, because it is the
    denominator that turns a block score into a per-coefficient one and a wrong constant there
    would rescale the likelihood percentage silently. Read as ``block_width`` rather than
    multiplied out here for a sharper reason than convenience: $C_{\mathrm{keep}}$ is not a
    constant of the architecture but the count the warm-up budget left standing, so the only
    trustworthy source for it is the model the pass actually scored.

    Args:
        record: The collection record. An offline re-run against a directory whose record carries
            no geometry block passes an empty mapping.

    Returns:
        $H \cdot C_{\mathrm{keep}}$ as a float, or ``None`` when the geometry is absent or
        degenerate. ``None`` means the likelihood-space percentage is not computed at all --
        reported as a skip rather than as a number divided by a guess.
    """
    geometry = record.get("geometry") or {}
    block_width = geometry.get("block_width")
    if block_width is None:
        return None
    total = float(block_width)
    return total if total > 0.0 else None


def likelihood_percent_support(record: Dict[str, Any]) -> Dict[str, Any]:
    r"""Whether the likelihood-space percentage is defined for this run, and if not, why not.

    Two independent preconditions, reported separately because they fail for unrelated reasons and
    an operator reading a skip needs to know which one it was: the run must have been scored under
    a likelihood whose block score is a log-density, and the block width that score is divided by
    must be known.

    The record carries ``budget_local`` whether or not it skipped, because a reader who finds the
    percentage present needs the caveat more than one who finds it absent: the denominator is
    $H \cdot C_{\mathrm{keep}}$ and $C_{\mathrm{keep}}$ is a consequence of the warm-up budget, so
    the number is comparable across runs only where their block widths are.

    Args:
        record: The collection record, read for ``likelihood`` and ``geometry``.

    Returns:
        A skip record in this package's usual shape -- ``skipped`` with a ``reason`` beside it --
        carrying the resolved block width, which is ``None`` exactly when the percentage is
        skipped, and the budget-locality note.
    """
    likelihood = str(record.get("likelihood") or "")
    samples_per_anchor = coefficients_per_anchor(record)
    if not likelihood:
        # Unknown is a skip, not a pass. The record has carried this key since before the
        # percentage existed, so an absent one means something is wrong with the tables rather
        # than that the run is old -- and the failure mode this whole precondition exists to
        # prevent is emitting a number whose units nobody checked.
        reason = "the collection record does not say which likelihood the run was scored under"
    elif likelihood != LIKELIHOOD_PERCENT_REQUIRES:
        reason = (
            f"likelihood={likelihood!r}: a block score is a sum of squared errors rather than a "
            f"log-density, so exponentiating it does not yield a density ratio"
        )
    elif samples_per_anchor is None:
        reason = "the collection record carries no usable geometry, so H*C_keep is unknown"
    else:
        reason = ""
    return {
        "skipped": bool(reason),
        "reason": reason or None,
        "likelihood": likelihood or None,
        "coefficients_per_anchor": None if reason else samples_per_anchor,
        # Stated whether or not the percentage was computed, and stated as a sentence rather than
        # as a flag: what a reader has to do with it is compare two runs' block widths before
        # comparing their percentages, and a bare ``true`` does not say that.
        "budget_local": (
            "the denominator is H*C_keep, and C_keep is the count of target channels the warm-up "
            "budget left standing rather than a constant of the architecture -- so this "
            "percentage is comparable across runs only where their block widths are. The nats it "
            "is derived from and the two error-space percentages carry no such restriction"
        ),
    }


def percent_columns(
    per_guid: pd.DataFrame, *, samples_per_anchor: Optional[float]
) -> Dict[str, np.ndarray]:
    r"""The three percentage readouts, one value per recording.

    Every one is computed **per recording and then averaged**, never as a ratio of two averages --
    the rule :func:`~teb_vae.lag_attn_cfs.eval.frames.skill_against` states and the rest of this
    pipeline's aggregation chain obeys. It is also what gives the bootstrap a per-recording
    quantity to resample.

    Args:
        per_guid: Per-recording means, carrying the two branches' squared error and the gap.
        samples_per_anchor: $H \cdot C_{\mathrm{keep}}$, or ``None`` to omit the likelihood-space
            percentage.

    Returns:
        Name to per-recording values, ``NaN`` wherever a recording measured nothing. The
        likelihood entry is **absent** rather than all-``NaN`` when ``samples_per_anchor`` is
        ``None``, so a reader can tell "not computed" from "computed and unmeasurable".
    """
    # ``skill_against`` guards the denominator at strictly positive and fails to NaN: a recording
    # whose target-only branch has zero error is degenerate, and an infinite percentage reported
    # as evidence is worse than an absent one.
    base_column, full_column = SQUARED_ERROR_COLUMNS
    mse_skill = skill_against(
        finite_column(per_guid, full_column), finite_column(per_guid, base_column)
    )
    # The RMSE ratio is $1 - \mathrm{skill}$ exactly, so taking the root of that rather than
    # dividing a second time means the two error-space percentages share one guard and cannot
    # disagree about the sign. The radicand is a ratio of two means of squares and so is never
    # negative; ``errstate`` is here for the NaNs the guard above introduces, not for that.
    with np.errstate(invalid="ignore"):
        rmse_ratio = np.sqrt(1.0 - mse_skill)
    rmse_name, mse_name = ERROR_SPACE_PERCENTS
    columns: Dict[str, np.ndarray] = {
        rmse_name: 100.0 * (1.0 - rmse_ratio),
        mse_name: 100.0 * mse_skill,
    }
    if samples_per_anchor is not None:
        # ``expm1`` rather than ``exp(x) - 1``: the exponent here is a per-anchor gap divided by
        # the block width -- over a thousand in this target domain -- so it is small, and the
        # subtraction would cancel away most of the significant digits of the answer.
        columns[LIKELIHOOD_SPACE_PERCENT] = 100.0 * np.expm1(
            finite_column(per_guid, "mc_pred_gap") / float(samples_per_anchor)
        )
    return columns


def build_percent_rows(
    per_guid: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise each percentage over the recordings, with its uncertainty and its denominator.

    Args:
        per_guid: Per-recording means, **after** :func:`percent_columns` has been written onto it.
        resamples: Bootstrap resamples, from ``eval_config.bootstrap_resamples``.
        seed: Bootstrap seed, from ``eval_config.seed``.

    Returns:
        One row per percentage present on the frame, carrying the mean and its interval, the
        quartiles, and the fraction of recordings the source helped **with its denominator**. A
        percentage the frame does not carry yields no row rather than a row of ``NaN``.
    """
    rows: List[Dict[str, Any]] = []
    for name, path in PERCENT_COLUMNS:
        if name not in per_guid.columns:
            continue
        values = finite_column(per_guid, name)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        positive = positive_fraction(values)
        rows.append(
            {
                "metric": name,
                "source_column": name,
                "score_path": path,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
                "ci_method": interval["method"],
                "bootstrap_resamples": int(interval["resamples"]),
                "positive_fraction": positive["fraction"],
                "n_positive": positive["n_positive"],
                "n_recordings_scored": positive["n"],
                "n_recordings_dropped_not_finite": positive["n_dropped_not_finite"],
            }
        )
    return rows


def percent_headline(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """The percentage point estimates, flattened for the headline registry.

    A name whose mean is not finite is **omitted** rather than carried as ``NaN``. The headline's
    own consistency check fails on a non-finite number and passes on an absent one, and the two
    say different things: ``None`` is "this run did not measure it", where ``NaN`` in a block of
    finite scalars reads as a broken readout.

    Args:
        rows: The percentage summary rows.

    Returns:
        Metric name to its mean over recordings, carrying only the finite ones.
    """
    return {
        str(row["metric"]): float(row["mean"])
        for row in rows
        if np.isfinite(row["mean"])
    }


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


def _shade_mean_interval(axis: Any, row: Dict[str, Any]) -> None:
    """Shade a summary row's bootstrap interval across a histogram, and label it as the mean's.

    Shared by the two figures this module draws rather than written twice, because they ship side
    by side: a divergence in the alpha, the colour or the label's wording would read as a
    difference between the two findings rather than as an editing accident.

    Shaded rather than drawn as a point range for the reason the label spells out -- an interval on
    the *mean* over a distribution of per-recording values is routinely far narrower than the data,
    and a point range invites reading it as the spread.

    Args:
        axis: The histogram panel to shade.
        row: A summary row carrying ``ci_lo`` and ``ci_hi``. A row missing either, or carrying a
            non-finite bound, is left undrawn rather than shaded over an invented range.
    """
    low, high = row.get("ci_lo"), row.get("ci_hi")
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high):
        return
    axis.axvspan(
        float(low), float(high), color=figures.COLOR_ORANGE, alpha=0.25, zorder=0,
        label=f"95% CI of the mean [{float(low):.4g}, {float(high):.4g}]",
    )
    axis.legend(fontsize=figures.FONT_LABEL, loc="best")


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
    _shade_mean_interval(axis, headline)

    figures.violin_panel(
        axes[1, 0],
        {name: finite_column(per_guid, column) for name, column, _ in PRED_GAP_COLUMNS},
        title="pred_gap per recording, by estimator",
        ylabel="nats per anchor",
        reference=0.0,
        reference_label="no improvement",
    )
    return figure


def build_percent_figure(
    per_guid: pd.DataFrame, percent_rows: Sequence[Dict[str, Any]]
) -> Any:
    """Draw the three percentages, with the two spaces kept on separate axes.

    Three panels rather than one, and the split is the content. The two error-space percentages
    share a panel because they are the same ratio under a root and are therefore the same order of
    magnitude. The likelihood-space one gets its own axis because it is *not*: a per-sample
    density ratio and an error reduction routinely differ by an order of magnitude, and one shared
    axis would flatten whichever is smaller into a line at zero and report it as nothing.

    Args:
        per_guid: Per-recording means carrying the percentage columns.
        percent_rows: The summary rows, read for the headline percentage's interval.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(3)
    headline = next(
        (row for row in percent_rows if row.get("metric") == HEADLINE_PERCENT), {}
    )

    axis = axes[0, 0]
    figures.histogram_panel(
        axis,
        finite_column(per_guid, HEADLINE_PERCENT),
        title=(
            f"{HEADLINE_PERCENT} per recording, "
            f"n = {int(headline.get('n_recordings_scored') or 0)}"
        ),
        xlabel="percent of the target-only forecast error removed",
        reference=0.0,
        reference_label="no improvement",
    )
    _shade_mean_interval(axis, headline)

    figures.violin_panel(
        axes[1, 0],
        {name: finite_column(per_guid, name) for name in ERROR_SPACE_PERCENTS},
        title="error space: percent of the forecast error the source removed",
        ylabel="percent",
        reference=0.0,
        reference_label="no improvement",
    )
    # A column the run did not produce reads back as all-NaN, which the panel draws as its empty
    # note -- so a run scored under 'mse', or one whose geometry was unavailable, says "not
    # measured" on the page rather than dropping a panel and changing the figure's shape.
    figures.violin_panel(
        axes[2, 0],
        {LIKELIHOOD_SPACE_PERCENT: finite_column(per_guid, LIKELIHOOD_SPACE_PERCENT)},
        title="likelihood space: extra density on each observed target coefficient",
        ylabel="percent",
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
        context: The analysis context, read for the collected per-sample table and -- for the
            block size the likelihood percentage divides by -- the collection record.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population this analysis describes is the
            set of recordings that scored an anchor, which only the table knows.

    Returns:
        The protocol's keys, the gap rows in both estimators, the three percentage rows with the
        flat block the headline registry reads, the KL rows, and the paths written.
    """
    per_sample = context.collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    # The percentages are derived here and written onto the frame *before* it is saved, so the
    # per-recording table carries them beside the values they came from -- which is what lets the
    # runner's by-cohort fan-out resolve one of them without a second reduction.
    support = likelihood_percent_support(context.collection.record or {})
    for name, values in percent_columns(
        per_guid, samples_per_anchor=support["coefficients_per_anchor"]
    ).items():
        per_guid[name] = values
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    gap_rows = build_gap_rows(per_guid, resamples=resamples, seed=seed)
    percent_rows = build_percent_rows(per_guid, resamples=resamples, seed=seed)
    kl_rows = build_kl_rows(per_guid, resamples=resamples, seed=seed)
    pd.DataFrame(gap_rows + percent_rows + kl_rows).to_csv(
        directory / SUMMARY_FILENAME, index=False
    )

    figure_names = [
        str(
            figures.render_to_pdf(
                build_distribution_figure(per_guid, gap_rows), directory / DISTRIBUTION_FIGURE
            ).name
        ),
        str(
            figures.render_to_pdf(
                build_percent_figure(per_guid, percent_rows), directory / PERCENT_FIGURE
            ).name
        ),
    ]
    return {
        "n_samples": scored_sample_count(per_sample, "mc_pred_gap"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "pred_gap": gap_rows,
        "pred_gap_percent": {
            "rows": percent_rows,
            # Flat, finite scalars only: this is what the headline registry digs into.
            "headline": percent_headline(percent_rows),
            # Why the likelihood-space percentage is absent, when it is. The two error-space ones
            # have no precondition beyond the columns they reduce, so they need no such record.
            "likelihood_space": support,
        },
        "kl": kl_rows,
        # Declared, not emitted: the by-class and by-subgroup variants are the runner's fan-out
        # over this frame, which carries one row per recording and the cohort each belongs to.
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [PER_RECORDING_FILENAME, SUMMARY_FILENAME, *figure_names],
    }
