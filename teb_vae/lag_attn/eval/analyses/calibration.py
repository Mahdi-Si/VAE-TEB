r"""Does $\sigma_{\mathrm{full}}$ mean anything, or is it a constant the model learned to like?

The decoder emits a full predictive Gaussian $\mathcal{N}(\mu_{\mathrm{full}},
\sigma^2_{\mathrm{full}})$, and every analysis before this one uses only its mean. This one scores
the distribution: NLL, CRPS, central-interval coverage, and the PIT.

**The reference is the strongest homoscedastic one, not a straw man.** $\hat{\sigma}^2$ is fitted
by maximum likelihood to the very residuals being scored, so it is the best a *constant* variance
could possibly do. A learned head that fails to beat it has learned how uncertain the forecast is
on average -- which the residuals already say -- and nothing about *where* it is uncertain, which
is the only thing a per-element variance head is for. The gain is reported with its sign, and a
non-positive gain is flagged rather than left in a table for a reader to notice.

**The precondition is the checkpoint's objective, not the presence of a tensor.** ``logvar_full``
is emitted on every forward regardless of what the model was trained to do with it, so a
presence check would happily score an untrained variance head -- one that received no gradient at
all under ``likelihood='mse'`` -- as though its numbers meant something. The gate is therefore
``likelihood == 'gaussian_nll'`` **and** ``sigma_obs == 'learned'``, taken from the objective
resolved off the checkpoint, and a run that fails it records a clean skip instead of numbers.

**Coverage is scored against the exact nominal.** A $\pm 2\sigma$ band covers $0.9545$, not
$0.95$; $0.95$ is $\pm 1.96\sigma$. Scoring a $2\sigma$ band against $0.95$ reports a perfectly
calibrated model as over-confident on every horizon, by half a percentage point, consistently
enough to look like a real finding.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "calibration"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. A predictive variance calibrated on the majority cohort and over-confident on
#: the minority ones is the failure a pooled NLL is least able to show.
GROUPED_METRICS = ("nll", "crps", "nll_gain")

#: ``eval_config.caps`` key naming this analysis's retention cap.
CAP_NAME = "calibration"

#: Band half-widths scored, in standard deviations. Their nominal coverages are computed from
#: $\mathrm{erf}$ rather than tabulated -- see :func:`~metrics.nominal_central_coverage`.
COVERAGE_SIGMAS = (1.0, 2.0, 3.0)

#: Bins for the PIT reliability curve. Twenty gives 5% resolution, which is fine enough to show
#: the $\cup$ / $\cap$ shape that distinguishes an over- from an under-confident variance.
PIT_BINS = 20

_HORIZON_PREFIX = "h"
_PIT_PREFIX = "pit"


def is_applicable(runner: EvalRunner) -> Optional[str]:
    """Return why calibration cannot be scored on this checkpoint, or ``None`` when it can.

    Args:
        runner: The loaded runner.

    Returns:
        A reason string, or ``None``.
    """
    objective = runner.objective
    if str(objective.likelihood) != "gaussian_nll":
        return (
            f"the checkpoint was trained under likelihood={objective.likelihood!r}, so its "
            f"logvar_full head received no gradient and its values are not a predictive "
            f"variance. logvar_full is emitted on every forward regardless, which is why its "
            f"presence is not the precondition."
        )
    if not (isinstance(objective.sigma_obs, str) and objective.sigma_obs == "learned"):
        return (
            f"the checkpoint was trained with sigma_obs={objective.sigma_obs!r}, a fixed "
            f"observation noise, so the decoder's logvar head did not set the likelihood's "
            f"variance and scoring it as a predictive distribution would attribute to the model "
            f"a calibration it never optimised."
        )
    return None


def _per_batch_calibration(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    r"""Compute one batch's per-sample calibration scores and its PIT histogram.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value, with the per-horizon and PIT-bin values flattened one
        column each.
    """
    view = runner.forecast_view(batch)
    mu, y_plus, logvar, mask = view.mu_full, view.y_plus, view.logvar_full, view.mask

    nll = metrics.gaussian_log_density(mu, y_plus, logvar)
    crps = metrics.crps_gaussian(mu, y_plus, logvar)

    # Fitted per batch on that batch's own residuals. A single split-wide constant would be
    # marginally stronger, but it would need a first pass purely to fit it, and the per-batch
    # fit is if anything the *harder* reference -- it adapts to each batch.
    reference_logvar = metrics.homoscedastic_logvar(mu, y_plus, mask)
    nll_reference = metrics.gaussian_log_density(
        mu, y_plus, torch.full_like(logvar, float(reference_logvar))
    )

    columns: Dict[str, Any] = {
        "nll": metrics.masked_per_sample_mean(nll, mask),
        "nll_homoscedastic": metrics.masked_per_sample_mean(nll_reference, mask),
        "crps": metrics.masked_per_sample_mean(crps, mask),
        "mean_logvar": metrics.masked_per_sample_mean(logvar, mask),
        "homoscedastic_logvar": float(reference_logvar),
    }
    columns["nll_gain"] = columns["nll_homoscedastic"] - columns["nll"]

    for k_sigma in COVERAGE_SIGMAS:
        indicator = metrics.coverage_indicator(mu, y_plus, logvar, k_sigma)
        columns[f"coverage_{k_sigma:g}sigma"] = metrics.masked_per_sample_mean(indicator, mask)

    # Per horizon: the profile that shows a variance head calibrated at h=1 and badly
    # over-confident at h=H_d, which a single pooled number averages away entirely.
    horizon_nll = _per_horizon(nll, mask)
    for step in range(int(horizon_nll.shape[1])):
        columns[f"{_HORIZON_PREFIX}{step:03d}_nll"] = horizon_nll[:, step]
    horizon_coverage = _per_horizon(
        metrics.coverage_indicator(mu, y_plus, logvar, 2.0), mask
    )
    for step in range(int(horizon_coverage.shape[1])):
        columns[f"{_HORIZON_PREFIX}{step:03d}_cov2"] = horizon_coverage[:, step]

    pit = metrics.pit_values(mu, y_plus, logvar)
    histogram = _pit_histogram(pit, mask, PIT_BINS)
    for index in range(PIT_BINS):
        columns[f"{_PIT_PREFIX}{index:03d}"] = histogram[:, index]
    return columns


def _per_horizon(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Masked mean over anchors and channels, leaving $(B, H_d)$.

    Args:
        values: Per-element quantity, $(B, A, H_d, C)$.
        mask: Feature mask, $(B, A, H_d, 1)$.

    Returns:
        $(B, H_d)$, ``NaN`` where a (sample, horizon) cell has no unmasked entry.
    """
    channels = float(values.shape[-1])
    total = (values * mask).sum(dim=(1, 3))
    count = mask.sum(dim=(1, 3)) * channels
    return torch.where(
        count > 0, total / count.clamp_min(1.0), torch.full_like(total, float("nan"))
    )


def _pit_histogram(pit: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    r"""Per-sample normalised PIT histogram, $(B, \mathrm{bins})$.

    Accumulated per sample rather than pooled so the reliability curve can carry a spread: a
    single pooled histogram cannot show whether a departure from uniformity is systematic across
    recordings or driven by a handful of them.

    Args:
        pit: PIT values in $[0, 1]$, $(B, A, H_d, C)$.
        mask: Feature mask, $(B, A, H_d, 1)$.
        bins: Number of equal-width bins.

    Returns:
        $(B, \mathrm{bins})$, each row summing to $1$, or ``NaN`` for a fully masked sample.
    """
    batch = int(pit.shape[0])
    # clamp so a PIT of exactly 1.0 lands in the last bin rather than one past it.
    index = (pit * bins).long().clamp(0, bins - 1)
    weights = mask.expand_as(pit)

    counts = torch.zeros(batch, bins, device=pit.device, dtype=pit.dtype)
    counts.scatter_add_(1, index.reshape(batch, -1), weights.reshape(batch, -1))
    totals = counts.sum(dim=1, keepdim=True)
    return torch.where(
        totals > 0, counts / totals.clamp_min(1.0), torch.full_like(counts, float("nan"))
    )


def _prefixed(frame: pd.DataFrame, prefix: str, suffix: str = "") -> list:
    """Return the flattened columns matching ``prefix`` + digits + ``suffix``, in index order.

    Matched by pattern rather than by ``startswith``: the horizon columns carry a quantity suffix
    (``h003_nll``), so a prefix test alone would either miss them or sweep in every column whose
    name happens to begin with the same letter.

    Args:
        frame: The collected frame.
        prefix: The column-name prefix.
        suffix: Trailing text after the index, or ``''`` for none.

    Returns:
        The matching column names, sorted.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    return sorted(name for name in frame.columns if pattern.match(name))


def _horizon_columns(frame: pd.DataFrame) -> list:
    """Return every per-horizon column, both quantities, in index order."""
    return _prefixed(frame, _HORIZON_PREFIX, "_nll") + _prefixed(
        frame, _HORIZON_PREFIX, "_cov2"
    )


def _write_figures(
    frame: pd.DataFrame, directory: Path, nominal: Dict[str, float]
) -> list:
    """Draw the reliability, coverage and sharpness figures.

    Args:
        frame: The per-sample frame.
        directory: The analysis directory.
        nominal: Band label to exact nominal coverage.

    Returns:
        The three paths written.
    """
    paths = []

    pit_columns = _prefixed(frame, _PIT_PREFIX)
    figure, axes = figures.new_figure(1, height_per_row=3.0)
    try:
        ax = axes[0, 0]
        centres = (np.arange(len(pit_columns)) + 0.5) / max(len(pit_columns), 1)
        density = (
            frame[pit_columns].to_numpy(dtype=np.float64) * len(pit_columns)
            if pit_columns
            else np.zeros((0, 0))
        )
        figures.ribbon_plot(
            ax, centres, density,
            title="PIT reliability -- flat at 1.0 is calibrated; "
                  "$\\cup$ over-confident, $\\cap$ over-dispersed",
            xlabel="PIT value", ylabel="density", label="median over samples",
        )
        ax.axhline(
            1.0, color=figures.COLOR_VERMILLION, linestyle="--", linewidth=1.2,
            label="uniform",
        )
        ax.legend(fontsize=7, loc="best")
        figures.style_axes(ax)
        paths.append(str(figures.render_to_pdf(figure, directory / "reliability.pdf")))
    finally:
        figures.plt.close(figure)

    figure, axes = figures.new_figure(2)
    try:
        labels = list(nominal)
        observed = [float(frame[f"coverage_{label}"].mean()) for label in labels]
        positions = np.arange(len(labels))
        axes[0, 0].bar(
            positions - 0.2, observed, width=0.4, label="observed",
            color=figures.COLOR_BLUE, edgecolor=figures.COLOR_BLACK, linewidth=0.4,
        )
        axes[0, 0].bar(
            positions + 0.2, [nominal[label] for label in labels], width=0.4, label="nominal",
            color=figures.COLOR_GRAY, edgecolor=figures.COLOR_BLACK, linewidth=0.4,
        )
        axes[0, 0].set_xticks(positions)
        axes[0, 0].set_xticklabels(
            [f"{label}\n(nominal {nominal[label]:.4f})" for label in labels], fontsize=7
        )
        axes[0, 0].set_title("Central-interval coverage against the exact nominal")
        axes[0, 0].set_ylabel("coverage")
        axes[0, 0].legend(fontsize=7, loc="best")
        figures.style_axes(axes[0, 0])

        horizon_columns = _prefixed(frame, _HORIZON_PREFIX, "_cov2")
        figures.ribbon_plot(
            axes[1, 0], figures.sequence_axis(len(horizon_columns)),
            frame[horizon_columns].to_numpy(dtype=np.float64)
            if horizon_columns
            else np.zeros((0, 0)),
            title="$2\\sigma$ coverage by horizon step",
            xlabel="Horizon step $h$", ylabel="coverage", label="median over samples",
        )
        axes[1, 0].axhline(
            metrics.nominal_central_coverage(2.0), color=figures.COLOR_VERMILLION,
            linestyle="--", linewidth=1.2, label="nominal 0.9545",
        )
        axes[1, 0].legend(fontsize=7, loc="best")
        figures.style_axes(axes[1, 0])
        paths.append(str(figures.render_to_pdf(figure, directory / "coverage.pdf")))
    finally:
        figures.plt.close(figure)

    figure, axes = figures.new_figure(2)
    try:
        figures.histogram_panel(
            axes[0, 0], frame.get("mean_logvar", pd.Series(dtype=float)),
            title="Sharpness: per-sample mean predictive $\\log\\sigma^2$",
            xlabel="$\\log\\sigma^2$",
            reference=(
                float(frame["homoscedastic_logvar"].mean())
                if "homoscedastic_logvar" in frame
                else None
            ),
            reference_label="homoscedastic reference",
        )
        figures.histogram_panel(
            axes[1, 0], frame.get("nll_gain", pd.Series(dtype=float)),
            title="NLL gain over the homoscedastic reference (positive is better)",
            xlabel="nats", color=figures.COLOR_GREEN,
            reference=0.0, reference_label="no gain",
        )
        paths.append(str(figures.render_to_pdf(figure, directory / "sharpness.pdf")))
    finally:
        figures.plt.close(figure)
    return paths


def run_calibration_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the learned predictive Gaussian as a distribution, or record why it cannot be.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, for the sample count and per-file grouping.

    Returns:
        The headline summary for ``summary.json``, or a recorded skip.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    reason = is_applicable(runner)
    if reason is not None:
        # A recorded skip, not a failure: the checkpoint is perfectly valid, it simply has no
        # predictive variance to score. A raised error here would set the run's exit code and
        # report a healthy run as broken.
        logger.warning(f"calibration skipped: {reason}")
        return {
            "skipped": True,
            "reason": reason,
            "objective": {
                "likelihood": runner.objective.likelihood,
                "sigma_obs": runner.objective.sigma_obs,
            },
        }

    caps = eval_config.get("caps") or {}
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(
            n_total, caps.get(CAP_NAME), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_calibration,
        max_samples=eval_config.get("max_samples"), plan=plan, progress_label="calibration",
    )
    frame = collected.frame

    pit_columns = _prefixed(frame, _PIT_PREFIX)
    horizon_columns = _horizon_columns(frame)
    identity = ["sample_index", "guid", "source_file"]

    frame.drop(columns=pit_columns + horizon_columns).to_csv(
        directory / "per_sample.csv", index=False
    )

    if horizon_columns:
        per_horizon = frame[identity + horizon_columns].melt(
            id_vars=identity, var_name="column", value_name="value"
        )
        per_horizon["horizon"] = (
            per_horizon["column"].str[len(_HORIZON_PREFIX):len(_HORIZON_PREFIX) + 3].astype(int)
        )
        per_horizon["quantity"] = np.where(
            per_horizon["column"].str.endswith("_cov2"), "coverage_2sigma", "nll"
        )
        per_horizon.drop(columns=["column"]).sort_values(
            ["sample_index", "horizon", "quantity"]
        ).to_csv(directory / "per_horizon.csv", index=False)

    if pit_columns:
        density = frame[pit_columns].to_numpy(dtype=np.float64) * len(pit_columns)
        with np.errstate(invalid="ignore"):
            pd.DataFrame(
                {
                    "bin": np.arange(len(pit_columns)),
                    "bin_centre": (np.arange(len(pit_columns)) + 0.5) / len(pit_columns),
                    "density": np.nanmean(density, axis=0),
                    "density_p25": np.nanpercentile(density, 25, axis=0),
                    "density_p75": np.nanpercentile(density, 75, axis=0),
                    "uniform": 1.0,
                }
            ).to_csv(directory / "reliability.csv", index=False)

    nominal = {
        f"{k_sigma:g}sigma": metrics.nominal_central_coverage(k_sigma)
        for k_sigma in COVERAGE_SIGMAS
    }
    figure_paths = _write_figures(frame, directory, nominal)

    mean_gain = float(frame["nll_gain"].mean()) if "nll_gain" in frame else float("nan")
    gain_is_real = bool(np.isfinite(mean_gain) and mean_gain > 0.0)
    if not gain_is_real:
        logger.warning(
            f"the learned variance adds nothing: mean NLL gain over the maximum-likelihood "
            f"homoscedastic reference is {mean_gain:.4g} nats. The head has learned how "
            f"uncertain the forecast is on average -- which the residuals already say -- and "
            f"nothing about where it is uncertain."
        )

    summary: Dict[str, Any] = {
        "skipped": False,
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "mean_nll": float(frame["nll"].mean()) if "nll" in frame else float("nan"),
        "mean_nll_homoscedastic": (
            float(frame["nll_homoscedastic"].mean())
            if "nll_homoscedastic" in frame
            else float("nan")
        ),
        "mean_nll_gain": mean_gain,
        "learned_variance_beats_homoscedastic": gain_is_real,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            references={"nll_gain": 0.0},
        ),
        "mean_crps": float(frame["crps"].mean()) if "crps" in frame else float("nan"),
        "coverage": {
            label: {
                "nominal": value,
                "observed": (
                    float(frame[f"coverage_{label}"].mean())
                    if f"coverage_{label}" in frame
                    else float("nan")
                ),
            }
            for label, value in nominal.items()
        },
        "figures": figure_paths,
    }
    for record in summary["coverage"].values():
        record["gap"] = record["observed"] - record["nominal"]

    logger.info(
        f"calibration: NLL {summary['mean_nll']:.4g} vs homoscedastic "
        f"{summary['mean_nll_homoscedastic']:.4g} (gain {mean_gain:.4g}); "
        f"2-sigma coverage {summary['coverage']['2sigma']['observed']:.4f} against a nominal "
        f"{summary['coverage']['2sigma']['nominal']:.4f}"
    )
    return summary
