r"""Does the source pathway help the forecast?

$$L_{\mathrm{full}} \quad\text{versus}\quad L_{\mathrm{base}}$$

The baseline forecast is built from the target's own history alone; the full forecast adds the
source-driven correction $\delta\mu_{\mathrm{src}}$. Their difference is the uplift, and a
positive one means the source pathway earned its place.

**A near-zero uplift is not automatically a collapsed pathway.** Under ``gaussian_nll`` with
``sigma_obs='learned'`` the two losses read *different variance heads*, so they differ even when
$\delta\mu_{\mathrm{src}}$ is identically zero -- and can differ in either direction. This
analysis therefore flags a near-zero uplift rather than declaring collapse, and names the
residual analysis, which isolates the mean pathway, as the readout that settles it. The joint
verdict in the scalar pass is where the two are combined.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from teb_vae.lag_attn.eval import figures, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "uplift"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. Whether the source pathway helps *more* on the pathological cohorts is the
#: question this analysis is most often asked, and the pooled number cannot answer it.
GROUPED_METRICS = ("uplift_abs", "uplift_rel")

#: Below this absolute mean uplift the run is flagged for inspection. Not a verdict: see the
#: module docstring for why the number alone cannot distinguish the two causes.
DEFAULT_NEAR_ZERO_UPLIFT = 1e-6


def _per_batch_uplift(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    """Compute one batch's per-sample full-versus-baseline losses.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value.
    """
    view = runner.forecast_view(batch)
    return metrics.uplift_metrics(
        view.mu_full,
        view.mu_base,
        view.y_plus,
        view.logvar_full,
        view.logvar_base,
        view.mask,
        likelihood=runner.objective.likelihood,
        sigma_obs=runner.objective.sigma_obs,
    )


def run_uplift_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the full forecast against the baseline and write the CSV and figure.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, for the sample count and per-file grouping.

    Returns:
        The headline summary for ``summary.json``.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    caps = eval_config.get("caps") or {}
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(
            n_total, caps.get("uplift"), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_uplift,
        max_samples=eval_config.get("max_samples"), plan=plan, progress_label="uplift",
    )
    frame = collected.frame
    frame.to_csv(directory / "per_sample.csv", index=False)

    absolute = frame["uplift_abs"].to_numpy(dtype=np.float64) if "uplift_abs" in frame else np.zeros(0)
    relative = frame["uplift_rel"].to_numpy(dtype=np.float64) if "uplift_rel" in frame else np.zeros(0)
    finite = absolute[np.isfinite(absolute)]

    figure, axes = figures.new_figure(2)
    try:
        figures.histogram_panel(
            axes[0, 0], absolute, title="Absolute uplift $L_{base} - L_{full}$",
            xlabel="uplift_abs", reference=0.0, reference_label="no uplift",
        )
        figures.histogram_panel(
            axes[1, 0], relative, title="Relative uplift", xlabel="uplift_rel",
            reference=0.0, reference_label="no uplift",
        )
        figure_path = str(figures.render_figure(figure, directory / "uplift"))
    finally:
        figures.plt.close(figure)

    mean_uplift = float(finite.mean()) if finite.size else float("nan")
    positive_fraction = float((finite > 0).mean()) if finite.size else float("nan")
    near_zero = bool(finite.size and abs(mean_uplift) < DEFAULT_NEAR_ZERO_UPLIFT)

    if near_zero:
        logger.warning(
            f"mean uplift is {mean_uplift:.3g}, within {DEFAULT_NEAR_ZERO_UPLIFT:g} of zero. "
            f"Under likelihood='{runner.objective.likelihood}' with "
            f"sigma_obs={runner.objective.sigma_obs!r} this does not on its own mean the source "
            f"pathway collapsed -- the full and baseline losses read different variance heads. "
            f"Read it beside the residual analysis, which isolates the mean pathway."
        )

    summary = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "mean_uplift_abs": mean_uplift,
        "median_uplift_abs": float(np.median(finite)) if finite.size else float("nan"),
        "mean_uplift_rel": float(
            relative[np.isfinite(relative)].mean()
        ) if np.isfinite(relative).any() else float("nan"),
        "positive_fraction": positive_fraction,
        "near_zero_uplift": near_zero,
        "likelihood": str(runner.objective.likelihood),
        "sigma_obs": runner.objective.sigma_obs,
        "figure": figure_path,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            references={"uplift_abs": 0.0, "uplift_rel": 0.0},
        ),
    }
    for column in ("l_full", "l_base"):
        if column in frame:
            values = frame[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"mean_{column}"] = float(values.mean()) if values.size else float("nan")

    logger.info(
        f"uplift: mean={mean_uplift:.6g}, positive on "
        f"{positive_fraction:.1%} of {len(frame)} sample(s)"
    )
    return summary
