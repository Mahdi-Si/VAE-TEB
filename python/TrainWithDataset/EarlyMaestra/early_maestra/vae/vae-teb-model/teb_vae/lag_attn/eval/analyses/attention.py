r"""Where in the UP history does the model look, and is it looking anywhere in particular?

The attention window is the model's only route from the source stream to the latent, so its shape
is the most direct statement the model makes about lag structure. This analysis reports that
shape and, just as importantly, reports whether there is a shape at all.

**``argmax_lag`` alone is not a finding.** It names a peak whether or not one exists, and on a
near-uniform row it names noise. The entropy is what separates the two: an entropy at its bound
means the head is averaging over the whole window and its argmax is meaningless. Both columns are
emitted together for that reason, and neither should be read without the other.

**That bound is not $\log L$.** Anchor $t$ has only $\min(t + 1, L)$ causally valid lags, so the
early anchors -- $60$ of the $240$ supported ones at the production geometry -- cannot reach
$\log 91 = 4.51$ at any flatness. Attention uniform over every available lag reports $4.398$,
$0.975$ of $\log L$, which read against $\log L$ looks like mild concentration and is in fact no
structure at all. ``mean_attainable_entropy_nats`` is the support-weighted bound that case
attains exactly, and is what a uniformity check must divide by; ``max_possible_entropy_nats``
remains $\log L$, the window's width.

**Head diversity answers whether head structure bought anything.** Four heads that all settled on
the same lag are one head with four times the parameters, and the per-head KL decomposition will
attribute across them regardless -- producing four confident, identical numbers.

**Every lag figure carries two axes.** The model-lag index is what the tensors are indexed by;
physical seconds is what a reader wants. The conversion is $s\ell - \Delta_{UP}$ with
$\Delta_{UP}$ from ``eval_config.up_shift_secs``, and its value is stated on every figure rather
than left implicit, because that offset has never actually been applied anywhere in this
repository before -- ``plotting.py`` reads it from a model attribute that does not exist -- so a
reader has good reason to want to see which number produced the axis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_attention, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "attention"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. Whether the selected lag differs between cohorts is the question that would
#: turn a lag readout into a clinical statement, and it is invisible in the pooled median.
GROUPED_METRICS = ("argmax_lag", "head_diversity")

#: ``eval_config.caps`` key bounding the per-sample scalar pass.
CAP_NAME = "attention"

#: ``eval_config.caps`` key bounding how many per-sample heatmaps are drawn. A separate, much
#: smaller cap than :data:`CAP_NAME`: a heatmap is read one sample at a time, so eight of them is
#: a figure and two thousand is a wasted gigabyte of retention.
HEATMAP_CAP_NAME = "samples"

#: Column prefix for the per-lag attention mass, melted into ``mass_by_lag.csv``.
_LAG_PREFIX = "l"


def _per_batch_attention(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    r"""Compute one batch's per-sample attention diagnostics.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value, with the lag profile flattened one column per lag.
    """
    outputs = runner.forward(batch)
    alpha = outputs["attn_weights"]
    support = masks.lag_readout_support(runner.model, alpha, get_field(batch, "weight"))
    diagnostics = metrics.attention_diagnostics(alpha, support)

    columns: Dict[str, Any] = {
        "argmax_lag": diagnostics["argmax_lag"],
        "entropy_mean": diagnostics["entropy_mean"].nanmean(dim=1),
        # Per sample, because the ceiling depends on which anchors that sample's support kept:
        # a recording whose gaps fall in the first $L$ steps has a different bound from one
        # whose do not, and a single pooled ceiling would misstate both.
        "attainable_entropy": diagnostics["attainable_entropy"],
        "head_diversity": diagnostics["head_diversity"],
        "n_support_anchors": diagnostics["n_support_anchors"],
    }
    for head in range(int(diagnostics["entropy_mean"].shape[1])):
        columns[f"h{head}_entropy"] = diagnostics["entropy_mean"][:, head]

    mass = diagnostics["mass_by_lag"]
    for lag in range(int(mass.shape[1])):
        columns[f"{_LAG_PREFIX}{lag:03d}"] = mass[:, lag]
    return columns


def _lag_columns(frame: pd.DataFrame) -> list:
    """Return the per-lag mass column names, in lag order."""
    return sorted(
        name
        for name in frame.columns
        if name.startswith(_LAG_PREFIX) and name[len(_LAG_PREFIX):].isdigit()
    )


def _write_heatmaps(
    runner: EvalRunner,
    loader: Any,
    plan: Optional[CollectionPlan],
    max_samples: Optional[int],
    directory: Path,
    up_shift_secs: float,
) -> Optional[str]:
    r"""Draw one head-averaged $(L, T)$ attention heatmap per retained sample.

    Head-averaged rather than one panel per head: at $M = 4$ heads and eight samples that would
    be thirty-two panels, and the per-head structure is already reported numerically by the
    entropy and diversity columns. What the heatmap adds over those is the *time course* -- an
    attention that locks onto a lag halfway through a recording looks identical to a stable one
    in any per-sample scalar.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        plan: Which samples to draw.
        max_samples: Prefix cap on iteration.
        directory: The analysis directory.
        up_shift_secs: The dataset's UP shift, for the second axis and the caption.

    Returns:
        The path written, or ``None`` when no sample was retained.
    """
    collected = collect_attention(runner, loader, plan=plan, max_samples=max_samples)
    weights = collected.arrays["attn_weights"]
    if weights.size == 0:
        return None

    n_samples, seq_len, _, num_lags = weights.shape
    warmup = int(runner.model._warmup_steps(seq_len))
    guids = list(collected.frame["guid"]) if "guid" in collected.frame else [""] * n_samples

    figure, axes = figures.new_figure(n_samples, height_per_row=2.2)
    try:
        for row in range(n_samples):
            ax = axes[row, 0]
            figures.heatmap_with_colorbar(
                figure,
                ax,
                weights[row].mean(axis=1).T,
                title=f"Head-averaged attention $\\alpha_{{t,\\ell}}$ -- {guids[row]}",
                xlabel="Anchor $t$ (decimated steps)" if row == n_samples - 1 else "",
                ylabel="Model lag $\\ell$",
                cmap="magma",
                symmetric=False,
                colorbar_label="attention mass",
                extent=(0.0, float(seq_len), float(num_lags) - 0.5, -0.5),
            )
            figures.shade_warmup(ax, warmup, float(seq_len), seq_len)
            figures.attach_lag_seconds_axis(
                ax, metrics.STEP_SECONDS, -float(up_shift_secs)
            )
        figure.suptitle(
            f"Lag axis: seconds = {metrics.STEP_SECONDS:g}$\\ell$ - "
            f"({up_shift_secs:g}) s, i.e. the delay in the original recording after undoing "
            f"the dataset's UP shift of {up_shift_secs:g} s. Shaded: warm-up.",
            fontsize=7,
            y=0.999,
        )
        return str(figures.render_to_pdf(figure, directory / "attention_heatmaps.pdf"))
    finally:
        figures.plt.close(figure)


def _write_summary_figure(
    frame: pd.DataFrame, lag_columns: list, directory: Path, up_shift_secs: float
) -> str:
    """Draw the argmax histogram, the lag-mass ribbon and the head-diversity histogram.

    Args:
        frame: The per-sample frame.
        lag_columns: The per-lag mass column names, in lag order.
        directory: The analysis directory.
        up_shift_secs: The dataset's UP shift, for the second axis and the caption.

    Returns:
        The path written.
    """
    profile = frame[lag_columns].to_numpy(dtype=np.float64) if lag_columns else np.zeros((0, 0))
    lags = np.arange(len(lag_columns), dtype=np.float64)

    figure, axes = figures.new_figure(3)
    try:
        # $-1$ is the "no supported anchor" sentinel, not a lag of minus one, and the headline
        # number already strips it (``argmax = argmax[argmax >= 0]`` below). The panel has to
        # strip it too: ``histogram_panel`` filters only non-finite values, so a sentinel row
        # would draw a bar at a lag that does not exist and would stretch the ``bins`` span to
        # $[-1, L-1]$, leaving no bin edge on an integer lag.
        argmax_lag = frame["argmax_lag"] if "argmax_lag" in frame else pd.Series(dtype=float)
        figures.histogram_panel(
            axes[0, 0],
            argmax_lag[argmax_lag >= 0],
            title="Per-sample argmax lag",
            xlabel="Model lag $\\ell$",
            bins=max(len(lag_columns), 1),
        )
        figures.ribbon_plot(
            axes[1, 0],
            lags,
            profile,
            title="Attention mass by lag, over the valid support",
            xlabel="Model lag $\\ell$",
            ylabel="$\\bar{\\alpha}_\\ell$",
            label="median over samples",
        )
        # The lag axis is the x here, not the y, so the shared secondary-axis helper -- which
        # decorates a y-axis -- does not apply; the equivalent x conversion is inlined.
        seconds = axes[1, 0].secondary_xaxis(
            "top",
            functions=(
                lambda lag: metrics.lag_to_seconds(lag, up_shift_secs=up_shift_secs),
                lambda sec: (sec + float(up_shift_secs)) / metrics.STEP_SECONDS,
            ),
        )
        seconds.set_xlabel("Physical delay (s)", fontsize=8)
        figures.histogram_panel(
            axes[2, 0],
            frame.get("head_diversity", pd.Series(dtype=float)),
            title="Head diversity (mean pairwise total variation between heads' lag profiles)",
            xlabel="head_diversity",
            color=figures.COLOR_PURPLE,
            reference=0.0,
            reference_label="all heads identical",
        )
        figure.suptitle(
            f"Physical delay = {metrics.STEP_SECONDS:g}$\\ell$ - ({up_shift_secs:g}) s, "
            f"undoing the dataset's UP shift of {up_shift_secs:g} s. Read argmax_lag only "
            f"against the entropy: near the attainable ceiling the row is flat and its peak "
            f"is noise.",
            fontsize=7,
            y=0.999,
        )
        return str(figures.render_to_pdf(figure, directory / "attention.pdf"))
    finally:
        figures.plt.close(figure)


def run_attention_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report where the attention looks, how sharply, and how differently across heads.

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
    seed = int(eval_config.get("seed", 0))
    max_samples = eval_config.get("max_samples")
    up_shift_secs = float(eval_config.get("up_shift_secs", 0.0))
    n_total = int((probe or {}).get("n_samples") or 0)
    groups = (probe or {}).get("source_files")

    def _plan(cap_name: str) -> Optional[CollectionPlan]:
        if not n_total:
            return None
        return CollectionPlan.build(n_total, caps.get(cap_name), seed, groups=groups)

    collected = collect_metrics(
        runner, loader, _per_batch_attention,
        max_samples=max_samples, plan=_plan(CAP_NAME), progress_label="attention",
    )
    frame = collected.frame

    lag_columns = _lag_columns(frame)
    identity = ["sample_index", "guid", "source_file"]

    per_sample = frame.drop(columns=lag_columns)
    # Emitted beside argmax_lag rather than derived by the reader: the offset's sign has never
    # been applied anywhere in this repository, so a CSV that shipped only the index would leave
    # every downstream consumer to rediscover the convention.
    if "argmax_lag" in per_sample:
        per_sample = per_sample.assign(
            lag_seconds_physical=metrics.lag_seconds_physical(
                per_sample["argmax_lag"].to_numpy(), up_shift_secs=up_shift_secs
            )
        )
    per_sample.to_csv(directory / "per_sample.csv", index=False)

    if lag_columns:
        mass = frame[identity + lag_columns].melt(
            id_vars=identity, value_vars=lag_columns, var_name="lag", value_name="mass",
        )
        mass["lag"] = mass["lag"].str[len(_LAG_PREFIX):].astype(int)
        mass["lag_seconds_physical"] = metrics.lag_seconds_physical(
            mass["lag"].to_numpy(), up_shift_secs=up_shift_secs
        )
        mass.sort_values(["sample_index", "lag"]).to_csv(
            directory / "mass_by_lag.csv", index=False
        )

    entropy_columns = sorted(
        name for name in frame.columns if name.startswith("h") and name.endswith("_entropy")
    )
    if entropy_columns:
        pd.DataFrame(
            {
                "head": [int(name[1:].split("_")[0]) for name in entropy_columns],
                "mean_entropy": [float(frame[name].mean()) for name in entropy_columns],
                "median_entropy": [float(frame[name].median()) for name in entropy_columns],
                "min_entropy": [float(frame[name].min()) for name in entropy_columns],
                "max_entropy": [float(frame[name].max()) for name in entropy_columns],
            }
        ).to_csv(directory / "head_entropy.csv", index=False)

    summary_figure = _write_summary_figure(frame, lag_columns, directory, up_shift_secs)
    heatmap_plan = _plan(HEATMAP_CAP_NAME)
    heatmap_figure = _write_heatmaps(
        runner, loader, heatmap_plan, max_samples, directory, up_shift_secs
    )

    # Two ceilings, and only one of them is a ceiling any anchor can reach. $\log L$ is the width
    # of the window; the attainable bound is $\log\min(t+1, L)$ averaged over the same support the
    # entropy uses, and it is strictly smaller whenever the support opens before step $L - 1$ --
    # by 2.5% at the production geometry. Read the entropy against the attainable one:
    # uniform-over-every-available-lag attention, which has no lag structure whatsoever, sits at
    # 0.975 of $\log L$, so a uniformity check with a 1% margin never fires against that bound.
    window_ceiling = float(np.log(len(lag_columns))) if len(lag_columns) > 1 else float("nan")
    attainable_ceiling = (
        float(frame["attainable_entropy"].mean())
        if "attainable_entropy" in frame
        else float("nan")
    )
    mean_entropy = float(frame["entropy_mean"].mean()) if "entropy_mean" in frame else float("nan")

    summary: Dict[str, Any] = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "heatmap_plan": None if heatmap_plan is None else heatmap_plan.describe(),
        "num_lags": len(lag_columns),
        "up_shift_secs": up_shift_secs,
        "mean_entropy_nats": mean_entropy,
        # The bound the uniformity check must divide by: the support-weighted mean of
        # log min(t+1, L), averaged over samples exactly as mean_entropy_nats is, so the ratio
        # of the two is 1 for attention uniform over every causally available lag.
        "mean_attainable_entropy_nats": attainable_ceiling,
        # log L: the width of the window, not a reachable entropy. Kept for the lag axis and the
        # figure captions; an entropy compared against it reads as more concentrated than it is.
        "max_possible_entropy_nats": window_ceiling,
        "mean_head_diversity": (
            float(frame["head_diversity"].mean()) if "head_diversity" in frame else float("nan")
        ),
        "figures": [path for path in (summary_figure, heatmap_figure) if path],
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS)
        ),
    }
    if "argmax_lag" in frame:
        argmax = frame["argmax_lag"].to_numpy()
        argmax = argmax[argmax >= 0]
        summary["median_argmax_lag"] = float(np.median(argmax)) if argmax.size else float("nan")
        summary["median_argmax_lag_seconds"] = (
            float(metrics.lag_seconds_physical(np.median(argmax), up_shift_secs=up_shift_secs))
            if argmax.size
            else float("nan")
        )
        summary["n_samples_without_support"] = int((frame["argmax_lag"].to_numpy() < 0).sum())

    logger.info(
        f"attention: median argmax lag {summary.get('median_argmax_lag')} "
        f"({summary.get('median_argmax_lag_seconds')} s); mean entropy "
        f"{mean_entropy:.3g} of an attainable {attainable_ceiling:.3g} nats "
        f"(window width log L = {window_ceiling:.3g})"
    )
    return summary
