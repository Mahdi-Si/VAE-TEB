r"""Does the forecast work?

The plainest question about a checkpoint, answered three ways.

**Per sample.** Masked MSE and $R^2$ over the whole feature vector and over each block
separately, because the scattering and phase-harmonic halves are different quantities on
different scales and a single number hides which one the model actually predicts.

**Per horizon step.** The profile the forecast is *supposed* to have: error rising with $h$,
because a step further into the future is harder. A flat profile is the signature of a model
predicting a constant, which can post a respectable aggregate MSE.

**Per anchor.** A time course, so a forecast that degrades partway through a recording is
localised rather than merely averaged in.

Every number is a *per-sample* mean -- each sample divided by its own mask sum. That is a
different quantity from ``compute_loss``'s pooled mask-weighted mean over the whole batch, and
the two do not agree unless every sample has the same mask density, which real recordings never
have. The pooled form, which does reconcile with training, is reported by the scalar pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, metrics, report
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, to_numpy

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "forecast"

#: Metrics resolved by clinical class and by canonical subgroup, when the split holds more than
#: one of either. The two headline numbers only: a grouped figure of every column would be
#: unreadable, and the per-sample CSV carries the group columns so any other cut is a ``groupby``.
GROUPED_METRICS = ("feat_mse_total", "feat_r2_total")


def _per_batch_metrics(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    """Compute one batch's per-sample forecast metrics and profiles.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to per-sample value. The horizon and anchor profiles are flattened into one
        column per position, so they land in the same frame as the scalars and a later
        ``groupby`` sees one row per sample.
    """
    view = runner.forecast_view(batch)
    columns: Dict[str, Any] = {
        name: values
        for name, values in metrics.forecast_metrics(
            view.mu_full, view.y_plus, view.mask, view.n_scattering
        ).items()
    }

    horizon_profile = metrics.horizon_error_profile(view.mu_full, view.y_plus, view.mask)
    anchor_profile = metrics.anchor_error_profile(view.mu_full, view.y_plus, view.mask)
    for step in range(int(horizon_profile.shape[1])):
        columns[f"h{step:02d}"] = horizon_profile[:, step]
    for anchor in range(int(anchor_profile.shape[1])):
        columns[f"a{anchor:03d}"] = anchor_profile[:, anchor]
    return columns


def _split_profiles(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Extract the profile columns carrying ``prefix`` into a long-form frame.

    Args:
        frame: The collected per-sample frame.
        prefix: ``'h'`` for the horizon profile, ``'a'`` for the anchor profile.

    Returns:
        One row per (sample, position), with the identity columns carried across.
    """
    columns = sorted(
        name for name in frame.columns
        if name.startswith(prefix) and name[len(prefix):].isdigit()
    )
    if not columns:
        return pd.DataFrame(columns=["guid", "source_file", "position", "mse"])
    long = frame.melt(
        id_vars=["sample_index", "guid", "source_file"],
        value_vars=columns,
        var_name="position",
        value_name="mse",
    )
    long["position"] = long["position"].str[len(prefix):].astype(int)
    return long.sort_values(["sample_index", "position"]).reset_index(drop=True)


def _profile_matrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    """Return the profile columns as a $(B, N)$ array in position order, for the ribbon figures."""
    columns = sorted(
        name for name in frame.columns
        if name.startswith(prefix) and name[len(prefix):].isdigit()
    )
    if not columns:
        return np.zeros((0, 0))
    return frame[columns].to_numpy(dtype=np.float64)


def _heatmap_triple(
    runner: EvalRunner, loader: Any, plan: Optional[CollectionPlan], max_samples: Optional[int]
) -> Dict[str, Any]:
    r"""Accumulate the mean forecast, mean target, and per-cell RMS residual, as $(c_y, T)$.

    The residual panel is an **RMS**, not a mean. A signed mean cancels: a channel the model
    over-predicts as often as it under-predicts averages to zero and reads as perfectly
    forecast, which is the opposite of the truth. The RMS cannot cancel, so a badly forecast
    channel is always a bright row.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        plan: Which samples to include.
        max_samples: Prefix cap on iteration.

    Returns:
        ``forecast``, ``target`` and ``residual_rms``, each $(c_y, T)$ with ``NaN`` where no
        anchor covered the step, plus ``n_scattering`` -- the split index, read off the batch
        because the model stores only the combined $c_y$ and cannot supply it.
    """
    from teb_vae.lag_attn.figure_primitives import average_forecast_per_channel

    horizon = int(runner.model.horizon)
    forecast_sum: Optional[np.ndarray] = None
    target_sum: Optional[np.ndarray] = None
    residual_sq_sum: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    n_scattering = 0
    global_index = 0

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        outputs = runner.forward(batch)
        y_st, y_ph = runner.build_target_streams(batch)
        n_scattering = int(y_st.shape[-1])
        target = torch.cat([y_st, y_ph], dim=-1)
        seq_len = int(target.shape[1])
        warmup = int(runner.model._warmup_steps(seq_len))

        for offset in range(int(target.shape[0])):
            index = global_index + offset
            if not (plan is None or plan.keeps(index)):
                continue
            rendered = average_forecast_per_channel(
                to_numpy(outputs["mu_full"][offset]), seq_len, horizon, warmup
            ).astype(np.float64)
            truth = to_numpy(target[offset]).astype(np.float64)
            covered = np.isfinite(rendered).all(axis=1)

            if forecast_sum is None:
                shape = rendered.shape
                forecast_sum = np.zeros(shape)
                target_sum = np.zeros(shape)
                residual_sq_sum = np.zeros(shape)
                counts = np.zeros(shape[0])

            filled = np.where(covered[:, None], rendered, 0.0)
            forecast_sum += filled
            target_sum += np.where(covered[:, None], truth, 0.0)
            residual_sq_sum += np.where(covered[:, None], (rendered - truth) ** 2, 0.0)
            counts += covered.astype(np.float64)
        global_index += int(target.shape[0])

    if forecast_sum is None or counts is None:
        empty = np.zeros((0, 0))
        return {
            "forecast": empty, "target": empty, "residual_rms": empty, "n_scattering": 0
        }

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.where(counts > 0, counts, np.nan)[:, None]
        # Transposed to (channel, time): a heatmap reads channels down and time across.
        return {
            "forecast": (forecast_sum / denom).T,
            "target": (target_sum / denom).T,
            "residual_rms": np.sqrt(residual_sq_sum / denom).T,
            "n_scattering": n_scattering,
        }


def _write_figures(
    frame: pd.DataFrame,
    triple: Dict[str, np.ndarray],
    directory: Path,
    *,
    n_scattering: int,
    n_channels: int,
) -> Dict[str, str]:
    """Emit the four figures, closing each one whatever happens.

    Args:
        frame: The per-sample frame.
        triple: The forecast / target / residual heatmap fields.
        directory: The analysis directory.
        n_scattering: Scattering block width, for the heatmap separator.
        n_channels: Total channel count.

    Returns:
        Figure name to written path.
    """
    written: Dict[str, str] = {}

    horizon = _profile_matrix(frame, "h")
    figure, axes = figures.new_figure(1)
    try:
        figures.ribbon_plot(
            axes[0, 0], figures.sequence_axis(horizon.shape[1] if horizon.size else 0), horizon,
            title="Forecast error by horizon step",
            xlabel="Horizon step $h$", ylabel="Masked MSE", label="median over samples",
        )
        written["horizon_error"] = str(
            figures.render_figure(figure, directory / "horizon_error")
        )
    finally:
        figures.plt.close(figure)

    anchor = _profile_matrix(frame, "a")
    figure, axes = figures.new_figure(1)
    try:
        figures.ribbon_plot(
            axes[0, 0], figures.sequence_axis(anchor.shape[1] if anchor.size else 0), anchor,
            title="Forecast error by anchor position",
            xlabel="Anchor $t$ (decimated steps)", ylabel="Masked MSE",
            label="median over samples",
        )
        written["anchor_error"] = str(
            figures.render_figure(figure, directory / "anchor_error")
        )
    finally:
        figures.plt.close(figure)

    figure, axes = figures.new_figure(2)
    try:
        figures.histogram_panel(
            axes[0, 0], frame.get("feat_mse_total", []), title="Per-sample masked MSE",
            xlabel="feat_mse_total",
        )
        figures.histogram_panel(
            axes[1, 0], frame.get("feat_r2_total", []), title="Per-sample $R^2$",
            xlabel="feat_r2_total", reference=0.0,
            reference_label="$R^2 = 0$ (predicting the channel mean)",
        )
        written["distributions"] = str(
            figures.render_figure(figure, directory / "distributions")
        )
    finally:
        figures.plt.close(figure)

    figure, axes = figures.new_figure(3, height_per_row=3.0)
    try:
        separator = n_scattering - 1 if 0 < n_scattering < n_channels else None
        for row, (key, title, symmetric) in enumerate(
            (
                ("forecast", "Mean forecast $\\mu_{\\mathrm{full}}$", True),
                ("target", "Mean target $Y$", True),
                ("residual_rms", "RMS residual (per channel, per step)", False),
            )
        ):
            figures.heatmap_with_colorbar(
                figure, axes[row, 0], triple.get(key, np.zeros((0, 0))), title=title,
                xlabel="Decimated step" if row == 2 else "", ylabel="Feature channel",
                symmetric=symmetric, separator_row=separator,
                colorbar_label="RMS" if key == "residual_rms" else "value",
            )
        written["heatmaps"] = str(figures.render_figure(figure, directory / "heatmaps"))
    finally:
        figures.plt.close(figure)

    return written


def run_forecast_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score the forecast against $Y^{+}$ and write the CSVs and figures.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, supplying the sample count and per-file grouping the
            capped draw stratifies over. Without it a cap falls back to an unstratified draw,
            which still covers the index space but cannot guarantee every shard appears.

    Returns:
        The headline summary for ``summary.json``.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    caps = eval_config.get("caps") or {}
    seed = int(eval_config.get("seed", 0))
    max_samples = eval_config.get("max_samples")
    n_total = int((probe or {}).get("n_samples") or 0)
    groups = (probe or {}).get("source_files")

    metric_plan = (
        CollectionPlan.build(n_total, caps.get("forecast"), seed, groups=groups)
        if n_total
        else None
    )
    collected = collect_metrics(
        runner, loader, _per_batch_metrics, max_samples=max_samples, plan=metric_plan,
        progress_label="forecast",
    )
    frame = collected.frame
    frame.to_csv(directory / "per_sample.csv", index=False)

    _split_profiles(frame, "h").to_csv(directory / "horizon_error.csv", index=False)
    _split_profiles(frame, "a").to_csv(directory / "anchor_error.csv", index=False)

    heatmap_plan = (
        CollectionPlan.build(n_total, caps.get("predictions"), seed, groups=groups)
        if n_total
        else None
    )
    triple = _heatmap_triple(runner, loader, heatmap_plan, max_samples)
    written = _write_figures(
        frame, triple, directory,
        n_scattering=int(triple["n_scattering"]),
        n_channels=int(runner.model.c_y),
    )

    summary: Dict[str, Any] = {
        "n_samples": int(len(frame)),
        "composition": collected.composition,
        "plan": collected.plan,
        "heatmap_plan": None if heatmap_plan is None else heatmap_plan.describe(),
        "figures": written,
        "by_group": report.emit_grouped_variants(
            frame, directory, value_columns=list(GROUPED_METRICS),
            references={"feat_r2_total": 0.0},
        ),
    }
    for column in ("feat_mse_total", "feat_mse_scattering", "feat_mse_phase",
                   "feat_r2_total", "feat_r2_scattering", "feat_r2_phase"):
        if column in frame:
            values = frame[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"mean_{column}"] = float(finite.mean()) if finite.size else float("nan")
            summary[f"median_{column}"] = float(np.median(finite)) if finite.size else float("nan")
            summary[f"n_finite_{column}"] = int(finite.size)

    logger.info(
        f"forecast: mean feat_mse_total={summary.get('mean_feat_mse_total'):.6g}, "
        f"mean feat_r2_total={summary.get('mean_feat_r2_total'):.4g} over {len(frame)} sample(s)"
    )
    return summary
