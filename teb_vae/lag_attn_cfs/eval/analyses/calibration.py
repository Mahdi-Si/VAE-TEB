r"""Is the decoder's learned variance the spread of its own errors?

Under ``gaussian_nll`` the decoder emits a mean **and** a log-variance per target coefficient, and
the block score is a negative log density only if the second one is true. Nothing else in this
pipeline checks that. A model can drive its NLL down by shrinking $\sigma$ wherever it happens to
be right and paying for it elsewhere, and every score in every other analysis would improve.

Four readings, over the scored coefficients themselves:

* **PIT.** $u = \Phi\!\left((x - \mu)/\sigma\right)$ is uniform on $(0, 1)$ exactly when the
  observation model is right. The shape is the diagnosis: $\cup$ means the variance is too small
  (too much mass in the tails), $\cap$ means it is too large.
* **Central coverage** at $1$, $2$ and $3\sigma$, against the exact nominals
  $\operatorname{erf}(k/\sqrt{2}) = 0.6827,\ 0.9545,\ 0.9973$. The two-sigma figure is the one
  routinely quoted as $0.95$; it is not, and a model checked against $0.95$ would look half a
  point miscalibrated while being exactly right.
* **CRPS**, in closed form for a Gaussian. Proper, bounded, and in the units of what is being
  scored -- which here is a $z$-scored wavelet coefficient and stays labelled ``normalised``. The
  raw cells quote this one in bpm; there is no clinical unit to quote it in here, and inverting the
  per-channel statistics would put the $98$ scored channels on scales spanning orders of magnitude,
  which is exactly what a single pooled CRPS cannot survive.
* **The gain over the homoscedastic MLE**, one constant variance fitted to *the very residuals
  being scored*. That is deliberately the strongest form of the baseline: the comparison then says
  what the learned, input-dependent variance earned over the best possible constant one, and the
  number is a floor rather than a flattering estimate. It is reported **per coefficient**, which is
  the unit the objective's own block score divides by, and the key says so -- the sibling's
  ``per_raw_sample`` name over this denominator would be silently non-comparable.

**Both clamp fractions travel beside the mean**, because ``mean_logvar_full`` alone cannot tell a
well-spread distribution from one with half its mass pinned on each bound -- and the two ends fail
in opposite directions. On the floor the decoder is over-confident and the NLL's squared term
explodes, which is what a loss spike looks like from the inside; on the ceiling it has given up and
is predicting noise, which reads as a *healthy falling* NLL while ``pred_gap`` goes to zero.

This is the one analysis whose output changes a config value: it states a recommended revision of
``model_config.VAE_model.logvar_clamp``, derived from the observed distribution rather than from
an opinion, and says plainly when no revision is warranted.

An ``mse`` checkpoint records a **skip**. Its log-variance head is never trained, so a probability
integral transform of its output would be arithmetic over an untrained tensor -- a number, and a
meaningless one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_cfs.eval.metrics import NORMALISED_UNIT, calibration_report

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "calibration"

#: What it writes.
COVERAGE_FILENAME = "calibration_coverage.csv"
PIT_FILENAME = "calibration_pit.csv"
LOGVAR_FILENAME = "calibration_logvar.csv"
PER_RECORDING_FILENAME = "calibration_per_recording.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
PIT_FIGURE = "pit_reliability.pdf"
LOGVAR_FIGURE = "logvar_distribution.pdf"

#: The likelihood this analysis is defined under. Anything else is a recorded skip.
REQUIRED_LIKELIHOOD = "gaussian_nll"

#: The config key a revision would be written to. Named in the output rather than described, so a
#: reader can act on the recommendation without going looking for where it lands.
CLAMP_CONFIG_KEY = "model_config.VAE_model.logvar_clamp"

#: The per-recording columns reduced beside the pooled census: these *are* on the aggregation
#: chain, unlike the PIT and the coverage, which are statements about a distribution.
PER_RECORDING_COLUMNS: Tuple[str, ...] = (
    "mean_logvar_full",
    "logvar_full_floor_frac",
    "logvar_full_ceil_frac",
)

#: The metrics resolved by cohort. All three: a decoder that is over-confident on one cohort and
#: has given up on another reports a healthy pooled mean, and the two clamp fractions are what
#: separate those two failures.
GROUPED_METRICS: Tuple[str, ...] = PER_RECORDING_COLUMNS

#: Quantiles of the observed log-variance a clamp revision is proposed from. Not the extremes: a
#: clamp set to the observed minimum and maximum is a clamp that binds again the moment anything
#: moves.
_CLAMP_QUANTILES = (0.001, 0.999)

#: How much room a proposed clamp leaves beyond those quantiles, in nats of log-variance.
_CLAMP_HEADROOM = 1.0

#: How much of the log-variance must sit within a clamp's margin before a revision is worth
#: proposing.
#:
#: Deliberately **lower** than
#: :data:`~teb_vae.lag_attn_cfs.eval.metrics.DEFAULT_PINNED_VARIANCE_MAX_FRAC`, and the gap
#: between them is the difference between two questions. The verdict fails at the point where the
#: readout stops meaning what it says; a clamp is worth widening long before that, because mass
#: pressed against a bound is already distorting the scores it produces.
_CLAMP_BINDING_FRAC = 0.1


def skip_record(likelihood: str) -> Dict[str, Any]:
    """Return the recorded skip for a checkpoint with no observation variance to calibrate.

    Args:
        likelihood: The likelihood the checkpoint was trained under.

    Returns:
        The protocol's keys with ``n_samples`` set to ``None`` -- this analysis scored no
        population, and a zero there would enter the coverage block as a disagreement with every
        analysis that did.
    """
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False},
        "skipped": True,
        "likelihood": str(likelihood),
        "reason": (
            f"calibration needs a predictive distribution; this checkpoint was trained under "
            f"likelihood={likelihood!r}, whose decoder log-variance head is never fitted, so a "
            f"probability integral transform of it would be arithmetic over an untrained tensor"
        ),
        "files": [],
    }


def coverage_frame(report: Dict[str, Any]) -> pd.DataFrame:
    """Lay the central-coverage table out, observed against nominal.

    Args:
        report: The calibration report.

    Returns:
        One row per level, with the signed error and the relative tail error the verdict is
        decided on.
    """
    rows: List[Dict[str, Any]] = []
    for record in report.get("coverage") or []:
        nominal_tail = float(record.get("nominal_tail", float("nan")))
        observed_tail = float(record.get("observed_tail", float("nan")))
        rows.append(
            {
                **record,
                "coverage_error": float(record.get("observed", float("nan")))
                - float(record.get("nominal", float("nan"))),
                # The tail is where an over-confident variance shows up, and it is where the
                # three levels differ by two orders of magnitude -- so the error that matters is
                # relative to the tail rather than absolute on the coverage.
                "relative_tail_error": (
                    abs(observed_tail - nominal_tail) / nominal_tail
                    if nominal_tail > 0.0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def pit_frame(report: Dict[str, Any]) -> pd.DataFrame:
    """Lay the PIT histogram out as bins with their density.

    Args:
        report: The calibration report.

    Returns:
        One row per bin: its edges, its centre, its count and its density -- which is $1.0$ in
        every bin when the observation model is right, whatever the bin count.
    """
    block = dict(report.get("pit") or {})
    edges = np.asarray(block.get("bin_edges") or [], dtype=np.float64)
    counts = np.asarray(block.get("counts") or [], dtype=np.float64)
    if counts.size == 0 or edges.size != counts.size + 1:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count", "density"])
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
            "density": np.asarray(block.get("density") or [], dtype=np.float64),
        }
    )


def logvar_frame(report: Dict[str, Any]) -> pd.DataFrame:
    """Lay the log-variance histogram out as bins with their share of the mass.

    Args:
        report: The calibration report.

    Returns:
        One row per bin, with the fraction of scored coefficients in it.
    """
    block = dict(report.get("logvar") or {})
    edges = np.asarray(block.get("bin_edges") or [], dtype=np.float64)
    counts = np.asarray(block.get("counts") or [], dtype=np.float64)
    if counts.size == 0 or edges.size != counts.size + 1:
        return pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count", "fraction"])
    total = float(counts.sum()) or 1.0
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "count": counts,
            "fraction": counts / total,
        }
    )


def histogram_quantile(frame: pd.DataFrame, quantile: float) -> float:
    """Return a quantile of a histogrammed distribution, interpolated within its bin.

    Args:
        frame: A histogram frame carrying ``bin_left``, ``bin_right`` and ``count``.
        quantile: The quantile in $[0, 1]$.

    Returns:
        The value, or ``NaN`` on an empty histogram. Interpolated rather than snapped to a bin
        edge, because a clamp recommendation rounded to the nearest of sixty bins would move with
        the bin count.
    """
    counts = finite_column(frame, "count")
    if counts.size == 0 or counts.sum() <= 0.0:
        return float("nan")
    cumulative = np.cumsum(counts) / float(counts.sum())
    index = int(np.searchsorted(cumulative, float(quantile), side="left"))
    index = min(max(index, 0), counts.size - 1)
    left = float(finite_column(frame, "bin_left")[index])
    right = float(finite_column(frame, "bin_right")[index])
    below = float(cumulative[index - 1]) if index else 0.0
    within = float(cumulative[index]) - below
    position = 0.0 if within <= 0.0 else (float(quantile) - below) / within
    return left + min(max(position, 0.0), 1.0) * (right - left)


def clamp_recommendation(
    report: Dict[str, Any], bounds: Dict[str, Any], per_recording: Dict[str, float]
) -> Dict[str, Any]:
    """Propose a ``logvar_clamp`` revision, or state that none is warranted.

    Derived from the observed distribution rather than from an opinion: a clamp is binding when a
    non-trivial share of the log-variance sits within the margin of one of its ends, and the
    revision moves that end past the observed extreme quantile with a stated headroom. When
    neither end binds, the recommendation is explicitly *no change* -- a recommendation emitted
    unconditionally is one that gets applied unconditionally.

    The distribution behind it is per **coefficient**: the log-variance head emits one value per
    scored coefficient, which is the axis the objective's own block score reduces over, so a clamp
    proposed from anything coarser would be proposed from a quantity the model never emits.

    Args:
        report: The calibration report, for the log-variance histogram.
        bounds: The model's own bound record -- the clamp and its margin.
        per_recording: The chained floor and ceiling fractions.

    Returns:
        The config key, the current clamp, whether each end binds, and the proposed value.
    """
    clamp = list(bounds.get("logvar_clamp") or [float("nan"), float("nan")])
    margin = float(bounds.get("logvar_margin", float("nan")))
    histogram = logvar_frame(report)
    low = histogram_quantile(histogram, _CLAMP_QUANTILES[0])
    high = histogram_quantile(histogram, _CLAMP_QUANTILES[1])
    floor_frac = float(per_recording.get("logvar_full_floor_frac", float("nan")))
    ceil_frac = float(per_recording.get("logvar_full_ceil_frac", float("nan")))

    floor_binds = np.isfinite(floor_frac) and floor_frac > _CLAMP_BINDING_FRAC
    ceil_binds = np.isfinite(ceil_frac) and ceil_frac > _CLAMP_BINDING_FRAC
    proposed = list(clamp)
    if floor_binds and np.isfinite(low):
        proposed[0] = float(np.floor((low - _CLAMP_HEADROOM) * 2.0) / 2.0)
    if ceil_binds and np.isfinite(high):
        proposed[1] = float(np.ceil((high + _CLAMP_HEADROOM) * 2.0) / 2.0)
    return {
        "config_key": CLAMP_CONFIG_KEY,
        "current": clamp,
        "margin": margin,
        "floor_frac": floor_frac,
        "ceil_frac": ceil_frac,
        "floor_binds": bool(floor_binds),
        "ceil_binds": bool(ceil_binds),
        "observed_quantiles": {
            f"q{_CLAMP_QUANTILES[0]:g}": low, f"q{_CLAMP_QUANTILES[1]:g}": high
        },
        "proposed": proposed,
        "change_recommended": bool(floor_binds or ceil_binds),
        "detail": (
            f"neither end of {CLAMP_CONFIG_KEY} is binding; leave it at {clamp}"
            if not (floor_binds or ceil_binds)
            else f"the decoder log-variance is pressed against its "
                 f"{'floor' if floor_binds else ''}{' and ' if floor_binds and ceil_binds else ''}"
                 f"{'ceiling' if ceil_binds else ''}; widen {CLAMP_CONFIG_KEY} to {proposed}"
        ),
    }


def build_pit_figure(pit: pd.DataFrame, coverage: pd.DataFrame) -> Any:
    """Draw the PIT histogram and the reliability curve it implies.

    Two panels, because they fail visibly in different ways. The histogram shows *where* the
    departure is -- a $\\cup$ is an over-confident variance, a $\\cap$ an under-confident one --
    against the flat line a calibrated model produces. The reliability curve shows *how much*, as
    the empirical CDF of the PIT against the diagonal it should lie on, which is the same
    departure integrated and is what the reported deviation measures.

    Args:
        pit: The PIT table.
        coverage: The coverage table, marked on the reliability panel.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(2)
    axis = axes[0, 0]
    centers = finite_column(pit, "bin_center")
    density = finite_column(pit, "density")
    if centers.size and np.isfinite(density).any():
        width = float(finite_column(pit, "bin_right")[0] - finite_column(pit, "bin_left")[0])
        axis.bar(centers, density, width=width * 0.95, color=figures.COLOR_BLUE, alpha=0.85)
        axis.axhline(
            1.0, color=figures.COLOR_VERMILLION, linestyle="--", linewidth=figures.LINE_REGULAR,
            label="uniform (calibrated)",
        )
        axis.legend(fontsize=figures.FONT_LABEL, loc="best")
    else:
        axis.text(
            0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
        )
    axis.set_title("Probability integral transform of the scored coefficients")
    axis.set_xlabel("PIT value")
    axis.set_ylabel("density")
    figures.style_axes(axis)

    axis = axes[1, 0]
    counts = finite_column(pit, "count")
    if counts.size and counts.sum() > 0.0:
        empirical = np.cumsum(counts) / float(counts.sum())
        edges = finite_column(pit, "bin_right")
        axis.plot([0.0, 1.0], [0.0, 1.0], color=figures.COLOR_GRAY, linestyle=":",
                  linewidth=figures.LINE_REGULAR, label="calibrated")
        axis.plot(edges, empirical, color=figures.COLOR_BLUE,
                  linewidth=figures.LINE_EMPHASIS, label="observed")
        axis.legend(fontsize=figures.FONT_LABEL, loc="best")
    else:
        axis.text(
            0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
        )
    observed = ", ".join(
        f"{int(row['level_sigma'])}s {float(row['observed']):.4f}/{float(row['nominal']):.4f}"
        for row in coverage.to_dict("records")
    ) if len(coverage) else "no coverage measured"
    axis.set_title(f"Reliability: observed vs nominal ({observed})")
    axis.set_xlabel("nominal cumulative probability")
    axis.set_ylabel("observed")
    figures.style_axes(axis)
    return figure


def build_logvar_figure(histogram: pd.DataFrame, bounds: Dict[str, Any]) -> Any:
    """Draw the decoder log-variance distribution with both clamp margins marked.

    The margins, not the clamp ends, are the lines that matter: the bound is a sigmoid, so the
    asymptote is never reached and mass *at* it would be invisible. What "pinned" means in every
    number this pipeline reports is "inside the margin", and this figure draws exactly that.

    Args:
        histogram: The log-variance histogram table.
        bounds: The model's own bound record.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(1)
    axis = axes[0, 0]
    centers = finite_column(histogram, "bin_center")
    fraction = finite_column(histogram, "fraction")
    if centers.size and np.isfinite(fraction).any():
        width = float(
            finite_column(histogram, "bin_right")[0] - finite_column(histogram, "bin_left")[0]
        )
        axis.bar(centers, fraction, width=width * 0.95, color=figures.COLOR_BLUE, alpha=0.85)
    else:
        axis.text(
            0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
        )

    clamp = list(bounds.get("logvar_clamp") or [])
    margin = float(bounds.get("logvar_margin", float("nan")))
    if len(clamp) == 2 and np.isfinite(margin):
        lo, hi = float(clamp[0]), float(clamp[1])
        for position, label in (
            (lo + margin, f"floor margin {lo + margin:g}"),
            (hi - margin, f"ceiling margin {hi - margin:g}"),
        ):
            axis.axvline(
                position, color=figures.COLOR_VERMILLION, linestyle="--",
                linewidth=figures.LINE_REGULAR, label=label,
            )
        for position in (lo, hi):
            axis.axvline(position, color=figures.COLOR_GRAY, linestyle=":",
                         linewidth=figures.LINE_REGULAR)
        axis.legend(fontsize=figures.FONT_LABEL, loc="best")
    axis.set_title("Decoder log-variance over the scored coefficients")
    axis.set_xlabel("log-variance")
    axis.set_ylabel("fraction of coefficients")
    figures.style_axes(axis)
    return figure


def run_calibration_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the observation model's calibration, or record why it could not be scored.

    Args:
        context: The analysis context, read for the pass's calibration census and the per-sample
            clamp fractions.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the coverage table, the PIT, the NLL gain, the clamp fractions
        and the recommended clamp revision -- or the skip record, under a likelihood with no
        observation variance.
    """
    collection = context.collection
    record = dict(getattr(collection, "record", None) or {})
    results = dict(getattr(collection, "results", None) or {})
    likelihood = str(record.get("likelihood") or results.get("likelihood") or "")
    if likelihood != REQUIRED_LIKELIHOOD:
        return skip_record(likelihood)

    bounds = dict(record.get("bounds") or {})
    # Recomputed from the census rather than read out of ``results``, so an offline re-run against
    # a finished directory produces the identical block from the identical sums.
    report = dict(results.get("calibration") or {})
    if not report:
        report = calibration_report({}, logvar_clamp=bounds.get("logvar_clamp"))
    if not report:
        return skip_record(likelihood)

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    coverage = coverage_frame(report)
    coverage.to_csv(directory / COVERAGE_FILENAME, index=False)
    pit = pit_frame(report)
    pit.to_csv(directory / PIT_FILENAME, index=False)
    histogram = logvar_frame(report)
    histogram.to_csv(directory / LOGVAR_FILENAME, index=False)

    per_sample = collection.per_sample
    per_guid = per_recording_means(per_sample, PER_RECORDING_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    clamp_rows: List[Dict[str, Any]] = []
    for column in PER_RECORDING_COLUMNS:
        values = finite_column(per_guid, column)
        interval = shared_stats.bootstrap_ci(values, resamples=resamples, seed=seed)
        clamp_rows.append(
            {
                "metric": column,
                **{key: value for key, value in describe(values).items() if key != "metric"},
                "ci_lo": interval["lo"],
                "ci_hi": interval["hi"],
            }
        )
    chained = {row["metric"]: float(row["mean"]) for row in clamp_rows}

    written = [
        str(figures.render_to_pdf(build_pit_figure(pit, coverage), directory / PIT_FIGURE).name),
        str(
            figures.render_to_pdf(
                build_logvar_figure(histogram, bounds), directory / LOGVAR_FIGURE
            ).name
        ),
    ]
    verdicts = {
        str(verdict.get("name")): verdict
        for verdict in (results.get("verdicts") or [])
        if isinstance(verdict, dict)
    }
    return {
        "n_samples": scored_sample_count(per_sample, "mean_logvar_full"),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {"capped": False, "bootstrap_resamples": resamples, "seed": seed},
        "likelihood": likelihood,
        # The scored unit, named rather than implied: it is a target coefficient, and the sibling's
        # per-raw-sample denominator differs from it by a factor of three at this geometry.
        "n_coefficients": int(report.get("n_coefficients", 0)),
        "weighting": report.get("weighting"),
        "coverage": coverage.to_dict("records"),
        "pit": {
            key: value for key, value in (report.get("pit") or {}).items() if key != "counts"
        },
        "crps_normalised": float(report.get("crps_normalised", float("nan"))),
        "crps_unit": NORMALISED_UNIT,
        # One when the learned variance is right on average, above one when it is too small.
        "mean_standardised_sq": report.get("mean_standardised_sq"),
        "nll": report.get("nll"),
        "clamp_fractions": clamp_rows,
        "bounds": bounds,
        "recommendation": clamp_recommendation(report, bounds, chained),
        "calibration_verdict": verdicts.get("calibration_near_nominal"),
        "decoder_variance_verdict": verdicts.get("decoder_variance_not_pinned"),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            COVERAGE_FILENAME, PIT_FILENAME, LOGVAR_FILENAME, PER_RECORDING_FILENAME, *written
        ],
    }
