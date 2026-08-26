r"""Is the latent being used, and is any hyperparameter bound getting in the way?

Three questions, and the third is the one most easily missed.

**How much information is the posterior carrying?** ``kld_raw`` as a scalar, and the per-dimension
KL behind it. The distribution matters more than the total: a KL of $2$ nats spread over
twenty-four dimensions is a very different model from one concentrated in two, and
``kld_active_frac`` reports the second as a count.

**Has the posterior moved at all?** ``posterior_drift`` is the mean-space companion to the KL,
computed under the same masking as ``task.py``'s ``mu_post_prior_gap_rms`` so the two are
comparable run to run.

**Is a bound binding?** ``mu_prior`` and $\mu^q - \mu^p$ are both $\tanh$-squashed to configured
scales, so a saturated element is one whose gradient has effectively vanished. A high
``delta_mu_sat_frac`` is not a property of the data -- it means ``delta_mu_scale`` is set too low
and the measured coupling is being *clipped*, which caps every transfer-entropy number the run
reports at a value the hyperparameter chose.

**Two readings of each headline diagnostic, deliberately.** The model computes all three
differently from everything else it reports: the saturation fractions apply no masking at all,
and ``kld_active_frac`` honours the KL support but ignores the per-step validity ``weight``. This
pipeline's rule is that every metric is masked exactly as the loss masks, so both are emitted --
the model's own under ``*_raw`` and this pipeline's under ``*_masked``. They are not redundant:
a large gap means the model's own diagnostic is dominated by steps the loss never scored, which
is a fact about the diagnostic rather than about the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "latent"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. How much information the latent carries per cohort is the closest thing this
#: pipeline has to a per-cohort coupling strength.
GROUPED_METRICS = ("kld_mean", "kld_dim_l2", "posterior_drift")

#: ``eval_config.caps`` key naming this analysis's retention cap. Rarely worth setting: the
#: retention is $d_z + T$ floats per sample, a few kilobytes.
CAP_NAME = "latent"

#: Column prefixes for the flattened per-dimension KL and per-step $K_t$ curve.
_DIM_PREFIX = "d"
_STEP_PREFIX = "t"

#: Batch-level diagnostics carried as broadcast columns. Averaging them over rows recovers the
#: sample-weighted pooled mean, which is what the summary wants -- but a column that is constant
#: within a batch has no per-sample meaning, so they are dropped before the CSV is written.
_BATCH_LEVEL_COLUMNS = (
    "kld_active_frac_raw",
    "mu_prior_sat_frac_raw",
    "delta_mu_sat_frac_raw",
    "kld_active_frac_masked",
    "mu_prior_sat_frac_masked",
    "delta_mu_sat_frac_masked",
)


def _per_batch_latent(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    """Compute one batch's per-sample latent aggregates and both diagnostic readings.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value, with the per-dimension KL and the $K_t$ curve flattened
        one column each.
    """
    model = runner.model
    outputs = runner.forward(batch)
    weight = get_field(batch, "weight")
    batch_size, seq_len = int(outputs["mu_post"].shape[0]), int(outputs["mu_post"].shape[1])
    mask_bt = masks.kld_mask(
        model, weight, batch_size, seq_len, device=outputs["mu_post"].device
    )

    kld_btd = metrics.kld_per_dim(outputs, model)
    aggregates = metrics.kld_aggregates(kld_btd, mask_bt)

    columns: Dict[str, Any] = {
        "kld_mean": aggregates["kld_mean"],
        "kld_sum": aggregates["kld_sum"],
        "kld_dim_l2": aggregates["kld_dim_l2"],
        "posterior_drift": metrics.posterior_drift(outputs, mask_bt),
        "n_support_steps": mask_bt.sum(dim=1),
    }

    per_dim = aggregates["kld_per_dim_mean"]
    for dim in range(int(per_dim.shape[1])):
        columns[f"{_DIM_PREFIX}{dim:03d}"] = per_dim[:, dim]

    # The K_t curve, masked to NaN outside the support so the figure renders a gap there rather
    # than a zero that reads as a genuinely silent prefix.
    kld_per_t = torch.where(
        mask_bt > 0, outputs["kld_per_t"], torch.full_like(outputs["kld_per_t"], float("nan"))
    )
    for step in range(seq_len):
        columns[f"{_STEP_PREFIX}{step:03d}"] = kld_per_t[:, step]

    raw = metrics.latent_health(outputs)
    masked = metrics.masked_latent_diagnostics(outputs, model, mask_bt)
    for name, value in raw.items():
        columns[f"{name}_raw"] = float(value)
    for name, value in masked.items():
        columns[f"{name}_masked"] = float(value)
    return columns


def _prefixed_columns(frame: pd.DataFrame, prefix: str) -> list:
    """Return the flattened columns carrying ``prefix``, in index order."""
    return sorted(
        name
        for name in frame.columns
        if name.startswith(prefix) and name[len(prefix):].isdigit()
    )


def _write_per_dim_figure(
    per_dim_mean: np.ndarray, per_dim_samples: np.ndarray, directory: Path
) -> list:
    r"""Draw the per-dimension KL bar chart and the per-dimension distribution violins.

    Symlog with $\mathrm{linthresh} = 10^{-2}$, matching the active threshold itself: a collapsed
    dimension sits at $10^{-8}$ and an active one at $10^{0}$, so a linear axis renders every
    collapsed dimension as an identical invisible sliver and a pure log axis cannot show a
    dimension at exactly zero at all.

    Args:
        per_dim_mean: Mean KL per dimension across samples, $(d_z,)$.
        per_dim_samples: Per-sample per-dimension KL, $(N, d_z)$.
        directory: The analysis directory.

    Returns:
        The two paths written.
    """
    threshold = float(metrics.KLD_ACTIVE_EPS)
    dims = np.arange(per_dim_mean.size)
    active = per_dim_mean > threshold

    paths = []
    figure, axes = figures.new_figure(1, height_per_row=3.0)
    try:
        ax = axes[0, 0]
        ax.bar(
            dims,
            per_dim_mean,
            color=np.where(active, figures.COLOR_BLUE, figures.COLOR_GRAY),
            edgecolor=figures.COLOR_BLACK,
            linewidth=0.4,
        )
        ax.axhline(
            threshold, color=figures.COLOR_VERMILLION, linestyle="--", linewidth=1.2,
            label=f"active threshold {threshold:g}",
        )
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_title(
            f"Per-dimension mean KL over the support -- "
            f"{int(active.sum())} of {per_dim_mean.size} active"
        )
        ax.set_xlabel("Latent dimension $d$")
        ax.set_ylabel("$\\overline{KL}_d$ (nats)")
        ax.legend(fontsize=7, loc="best")
        figures.style_axes(ax)
        paths.append(str(figures.render_figure(figure, directory / "per_dim_kl")))
    finally:
        figures.plt.close(figure)

    figure, axes = figures.new_figure(1, height_per_row=3.0)
    try:
        ax = axes[0, 0]
        columns = [
            per_dim_samples[np.isfinite(per_dim_samples[:, dim]), dim]
            for dim in range(per_dim_samples.shape[1])
        ]
        # A dimension with no finite value would make violinplot raise, and an all-zero one has
        # no width to draw; both are replaced by a single zero so the axis stays complete.
        columns = [column if column.size else np.zeros(1) for column in columns]
        # An empty *collection* has no ``d###`` columns at all, so the list comprehension above
        # produces no entries and the per-column repair never runs. ``violinplot([])`` raises
        # ("zero-size array to reduction operation minimum"), which would take down the run at
        # its final step -- the one thing the panels are required not to do.
        if columns:
            ax.violinplot(columns, positions=dims, showmedians=True, widths=0.8)
        else:
            ax.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
                ha="center", va="center", color=figures.COLOR_GRAY,
            )
        ax.axhline(
            threshold, color=figures.COLOR_VERMILLION, linestyle="--", linewidth=1.2,
            label=f"active threshold {threshold:g}",
        )
        ax.set_yscale("symlog", linthresh=threshold)
        ax.set_title("Per-dimension KL, distribution across samples")
        ax.set_xlabel("Latent dimension $d$")
        ax.set_ylabel("$\\overline{KL}_d$ (nats)")
        ax.legend(fontsize=7, loc="best")
        figures.style_axes(ax)
        paths.append(str(figures.render_figure(figure, directory / "per_dim_violin")))
    finally:
        figures.plt.close(figure)
    return paths


def _write_kt_figure(
    curves: np.ndarray, runner: EvalRunner, directory: Path
) -> str:
    r"""Draw the $K_t$ time course with the out-of-support regions shaded.

    Both ends are shaded, and the second one is the easily forgotten half: under
    ``kld_support='anchor'`` the final $H_d$ steps are outside the support too, because their
    forecast window runs off the end of the sequence and nothing pulls their posterior away from
    the prior. Left unshaded, the KL falling to zero there reads as the model losing interest
    late in the recording.

    Args:
        curves: Per-sample $K_t$, $(N, T)$, already ``NaN`` outside the support.
        runner: The loaded runner, for the warm-up and the support convention.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    seq_len = int(curves.shape[1]) if curves.size else 0
    warmup = int(runner.model._warmup_steps(seq_len)) if seq_len else 0

    figure, axes = figures.new_figure(1, height_per_row=3.0)
    try:
        ax = axes[0, 0]
        figures.ribbon_plot(
            ax, figures.sequence_axis(seq_len), curves,
            title="$K_t$ over the recording, with the out-of-support region shaded",
            xlabel="Step $t$ (decimated)", ylabel="$K_t$ (nats)",
            label="median over samples",
        )
        if warmup > 0:
            ax.axvspan(0, warmup, color=figures.COLOR_LIGHT_GRAY, alpha=0.6, zorder=0)
            ax.text(
                warmup / 2.0, 0.97, "warm-up", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color=figures.COLOR_GRAY,
            )
        if str(runner.model.kld_support) == "anchor" and seq_len:
            tail = seq_len - int(runner.model.horizon)
            ax.axvspan(tail, seq_len, color=figures.COLOR_LIGHT_GRAY, alpha=0.6, zorder=0)
            ax.text(
                (tail + seq_len) / 2.0, 0.97, "no forecast window",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6, color=figures.COLOR_GRAY,
            )
        figures.style_axes(ax)
        return str(figures.render_figure(figure, directory / "kt_curve"))
    finally:
        figures.plt.close(figure)


def run_latent_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report per-dimension KL, both readings of the headline diagnostics, and posterior drift.

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
            n_total, caps.get(CAP_NAME), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_latent,
        max_samples=eval_config.get("max_samples"), plan=plan, progress_label="latent",
    )
    frame = collected.frame

    dim_columns = _prefixed_columns(frame, _DIM_PREFIX)
    step_columns = _prefixed_columns(frame, _STEP_PREFIX)
    present_batch_level = [name for name in _BATCH_LEVEL_COLUMNS if name in frame]

    frame.drop(columns=dim_columns + step_columns + present_batch_level).to_csv(
        directory / "per_sample.csv", index=False
    )

    per_dim_samples = (
        frame[dim_columns].to_numpy(dtype=np.float64) if dim_columns else np.zeros((0, 0))
    )
    threshold = float(metrics.KLD_ACTIVE_EPS)
    if per_dim_samples.size:
        with np.errstate(invalid="ignore"):
            per_dim_mean = np.nanmean(per_dim_samples, axis=0)
    else:
        per_dim_mean = np.zeros(0)

    pd.DataFrame(
        {
            "dim": np.arange(per_dim_mean.size),
            "kld_mean": per_dim_mean,
            "active": per_dim_mean > threshold,
            "kld_p25": (
                np.nanpercentile(per_dim_samples, 25, axis=0)
                if per_dim_samples.size
                else np.zeros(0)
            ),
            "kld_median": (
                np.nanpercentile(per_dim_samples, 50, axis=0)
                if per_dim_samples.size
                else np.zeros(0)
            ),
            "kld_p75": (
                np.nanpercentile(per_dim_samples, 75, axis=0)
                if per_dim_samples.size
                else np.zeros(0)
            ),
        }
    ).to_csv(directory / "per_dim.csv", index=False)

    curves = (
        frame[step_columns].to_numpy(dtype=np.float64) if step_columns else np.zeros((0, 0))
    )
    figure_paths = _write_per_dim_figure(per_dim_mean, per_dim_samples, directory)
    figure_paths.append(_write_kt_figure(curves, runner, directory))

    diagnostics = {
        name: float(frame[name].mean()) if name in frame else float("nan")
        for name in _BATCH_LEVEL_COLUMNS
    }
    saturation_threshold = float(eval_config.get("saturation_flag_threshold", 0.05))
    saturated = {
        name: bool(np.isfinite(diagnostics[name]) and diagnostics[name] > saturation_threshold)
        for name in ("mu_prior_sat_frac_masked", "delta_mu_sat_frac_masked")
    }
    for name, fired in saturated.items():
        if fired:
            bound = "mu_scale" if name.startswith("mu_prior") else "delta_mu_scale"
            logger.warning(
                f"{name} is {diagnostics[name]:.3g}, above the configured "
                f"{saturation_threshold:g}: {bound} is binding on a material fraction of "
                f"elements, whose gradient has therefore vanished. This is a mis-set "
                f"hyperparameter, not a property of the data -- and on delta_mu_scale it caps "
                f"every transfer-entropy number this run reports."
            )

    summary: Dict[str, Any] = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "d_z": int(per_dim_mean.size),
        "kld_active_threshold": threshold,
        "n_active_dims": int((per_dim_mean > threshold).sum()),
        "diagnostics": diagnostics,
        "diagnostics_note": (
            "'_raw' is the model's own in-forward reading; '_masked' applies this pipeline's "
            "mask. They differ because the model's saturation fractions apply no masking at all "
            "and its kld_active_frac ignores the per-step validity weight. A large gap means "
            "the raw diagnostic is dominated by steps the loss never scored."
        ),
        "saturation_flag_threshold": saturation_threshold,
        "saturation_flagged": saturated,
        "figures": figure_paths,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS)
        ),
    }
    for column in ("kld_mean", "kld_sum", "kld_dim_l2", "posterior_drift"):
        if column in frame:
            values = frame[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"mean_{column}"] = float(values.mean()) if values.size else float("nan")

    logger.info(
        f"latent: {summary['n_active_dims']} of {summary['d_z']} dimension(s) active; "
        f"kld_active_frac raw={diagnostics['kld_active_frac_raw']:.3g} "
        f"masked={diagnostics['kld_active_frac_masked']:.3g}"
    )
    return summary
