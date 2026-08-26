r"""Is the source-driven mean correction alive?

$\delta\mu_{\mathrm{src}}$ is the *entire* mean-space contribution of the source pathway: the
full forecast is $\mu_{\mathrm{base}} + \delta\mu_{\mathrm{src}}$, so if this is zero the source
changed nothing about what the model predicts, whatever the KL says.

That makes it the one readout that isolates the mean pathway. The uplift cannot, because under
``gaussian_nll`` with ``sigma_obs='learned'`` the full and baseline losses read different
variance heads and differ even when $\delta\mu_{\mathrm{src}} \equiv 0$; the KL cannot either,
because it is a property of the posterior and says nothing about whether the decoder used it.

``residual_ratio`` -- the correction's RMS relative to the forecast's -- is the reported signal
rather than the raw RMS, because an absolute magnitude is uninterpretable without the scale of
the thing it corrects.

**The per-anchor trace is the other half.** A pathway that is active early and flat later is a
different finding from one that never activates, and a per-sample scalar averages the two into
the same number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import figures, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "residual"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. A source pathway that is alive on one cohort and dead on another is a finding
#: the pooled ratio averages away entirely.
GROUPED_METRICS = ("residual_ratio", "residual_rms")

#: ``eval_config.caps`` key naming this analysis's retention cap.
CAP_NAME = "residual"


def _per_batch_residual(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    """Compute one batch's per-sample residual activity and its per-anchor trace.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value, the per-anchor trace flattened one column per anchor.
    """
    view = runner.forecast_view(batch)
    columns: Dict[str, Any] = dict(
        metrics.residual_usage(view.delta_mu_src, view.mu_full, view.mask)
    )
    trace = metrics.residual_per_anchor(view.delta_mu_src, view.mask)
    for anchor in range(int(trace.shape[1])):
        columns[f"a{anchor:03d}"] = trace[:, anchor]
    return columns


def _anchor_columns(frame: pd.DataFrame) -> list:
    """Return the per-anchor column names, in anchor order."""
    return sorted(
        name for name in frame.columns if name.startswith("a") and name[1:].isdigit()
    )


def run_residual_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure the source-driven correction's activity and flag a collapsed pathway.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block. ``health_probe_floor`` doubles as the
            collapse threshold, so the run-level probe and this analysis cannot disagree about
            what "collapsed" means.
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
            n_total, caps.get(CAP_NAME), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_residual,
        max_samples=eval_config.get("max_samples"), plan=plan, progress_label="residual",
    )
    frame = collected.frame

    anchor_columns = _anchor_columns(frame)
    identity = ["sample_index", "guid", "source_file"]
    frame.drop(columns=anchor_columns).to_csv(directory / "per_sample.csv", index=False)

    if anchor_columns:
        # Melted from a sub-frame, not from `frame`: `residual_rms` is already a per-sample
        # column there, and pandas refuses a value_name that collides with an existing one.
        per_anchor = frame[identity + anchor_columns].melt(
            id_vars=identity, value_vars=anchor_columns,
            var_name="anchor", value_name="residual_rms",
        )
        per_anchor["anchor"] = per_anchor["anchor"].str[1:].astype(int)
        per_anchor.sort_values(["sample_index", "anchor"]).to_csv(
            directory / "per_anchor.csv", index=False
        )
        traces = frame[anchor_columns].to_numpy(dtype=np.float64)
    else:
        traces = np.zeros((0, 0))

    ratios = (
        frame["residual_ratio"].to_numpy(dtype=np.float64)
        if "residual_ratio" in frame
        else np.zeros(0)
    )
    finite = ratios[np.isfinite(ratios)]
    threshold = float(eval_config.get("health_probe_floor", 0.0))
    mean_ratio = float(finite.mean()) if finite.size else float("nan")
    collapsed = bool(finite.size and mean_ratio < threshold)

    figure, axes = figures.new_figure(2)
    try:
        figures.histogram_panel(
            axes[0, 0], ratios,
            title="Per-sample residual ratio $\\mathrm{rms}(\\delta\\mu_{src}) / "
                  "\\mathrm{rms}(\\mu_{full})$",
            xlabel="residual_ratio", reference=threshold, reference_label="collapse threshold",
        )
        figures.ribbon_plot(
            axes[1, 0], figures.sequence_axis(traces.shape[1] if traces.size else 0), traces,
            title="Residual magnitude by anchor position",
            xlabel="Anchor $t$ (decimated steps)",
            ylabel="$\\mathrm{rms}(\\delta\\mu_{src})$", label="median over samples",
        )
        figure_path = str(figures.render_figure(figure, directory / "residual"))
    finally:
        figures.plt.close(figure)

    if collapsed:
        logger.warning(
            f"mean residual_ratio is {mean_ratio:.3g}, below the configured floor "
            f"{threshold:g}: the source-driven mean correction is not moving the forecast. This "
            f"is a finding to report, not a run to abort -- the checkpoint loaded, which the "
            f"weight-space preflight check established separately."
        )

    summary = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "mean_residual_ratio": mean_ratio,
        "median_residual_ratio": float(np.median(finite)) if finite.size else float("nan"),
        "min_residual_ratio": float(finite.min()) if finite.size else float("nan"),
        "max_residual_ratio": float(finite.max()) if finite.size else float("nan"),
        "collapse_threshold": threshold,
        "collapsed": collapsed,
        "figure": figure_path,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            references={"residual_ratio": threshold},
        ),
    }
    for column in ("residual_rms", "forecast_rms"):
        if column in frame:
            values = frame[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"mean_{column}"] = float(values.mean()) if values.size else float("nan")

    logger.info(
        f"residual: mean residual_ratio={mean_ratio:.6g} over {len(frame)} sample(s); "
        f"collapsed={collapsed}"
    )
    return summary
