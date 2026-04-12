"""Temporal (horizon and anchor) analyses for the lag-attn v1 model.

This module replaces the legacy raw-FHR per-timestep accuracy analyses
with two feature-forecast equivalents:

- :func:`run_horizon_error_profile` — MSE as a function of horizon step
  ``h ∈ [0, H_d)``. Answers "how fast does the forecast decay the further
  out we look?"
- :func:`run_anchor_position_analysis` — average MSE as a function of the
  anchor index ``t ∈ [warmup, T - H_d)``. Answers "does the model trust
  its own forecast more at the start or end of the 20-minute window?"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    collect_forecast_errors_per_horizon,
)
from model.vae_teb_prediction.testing.visualizers import (
    plot_forecast_error_by_horizon,
    FONT_LABEL,
    FONT_TITLE,
    SAVE_DPI,
    COLOR_BLUE,
    COLOR_VERMILLION,
    _style_axes,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib should be available
    plt = None  # type: ignore[assignment]


def run_horizon_error_profile(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 200,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute per-horizon forecast MSE and plot the ribbon curve.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override for the output directory (defaults
            to ``runner.ensure_dir("horizon_error")``).

    Returns:
        Long-format DataFrame with columns ``[h, mse_mean, mse_ci_lo,
        mse_ci_hi, mse_st_mean, mse_ph_mean]``.
    """
    if max_samples <= 0:
        logger.info("horizon_error: skipped (max_samples <= 0)")
        return pd.DataFrame()

    logger.info(f"Collecting horizon error profile (max {max_samples} samples)...")
    per_step_df = collect_forecast_errors_per_horizon(runner, loader, max_samples)
    if per_step_df.empty:
        logger.warning("horizon_error: no rows collected.")
        return pd.DataFrame()

    if output_dir is None:
        output_dir = runner.ensure_dir("horizon_error")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate.
    def _q_lo(s: pd.Series) -> float:
        vals = s.dropna()
        return float(np.nanpercentile(vals, 2.5)) if len(vals) else float("nan")

    def _q_hi(s: pd.Series) -> float:
        vals = s.dropna()
        return float(np.nanpercentile(vals, 97.5)) if len(vals) else float("nan")

    grouped = per_step_df.groupby("h", as_index=False)
    summary = pd.DataFrame(grouped.agg(
        mse_mean=("mse_step", "mean"),
        mse_ci_lo=("mse_step", _q_lo),
        mse_ci_hi=("mse_step", _q_hi),
        mse_st_mean=("mse_st", "mean"),
        mse_ph_mean=("mse_ph", "mean"),
    ))

    summary_path = output_dir / "horizon_error.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved horizon error summary to {summary_path}")

    # Keep the raw per-step long table as well for drill-down analyses.
    per_step_path = output_dir / "forecast_errors_per_horizon.csv"
    per_step_df.to_csv(per_step_path, index=False)

    plot_forecast_error_by_horizon(per_step_df, output_dir / "horizon_error.pdf")
    return summary


def run_anchor_position_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 200,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute MSE as a function of anchor index ``t``.

    Iterates through the loader, builds ``y_plus`` via
    :meth:`TestRunner.build_future_target`, and evaluates the per-anchor
    mean squared error ``mean_{h,c}(mu_full - y_plus)^2`` on the valid
    anchor range. Averages across samples and plots a line curve with
    warmup shading.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override for the output directory (defaults
            to ``runner.ensure_dir("anchor_error")``).

    Returns:
        DataFrame with columns ``[t, mse_mean, mse_std, n_samples]``.
    """
    if max_samples <= 0:
        logger.info("anchor_error: skipped (max_samples <= 0)")
        return pd.DataFrame()

    logger.info(f"Running anchor position analysis (max {max_samples} samples)...")

    if output_dir is None:
        output_dir = runner.ensure_dir("anchor_error")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_anchor_mse: list = []
    warmup = int(runner.warmup_steps)
    H_d = int(runner.horizon)

    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)

            mu_full = outputs["mu_full"]
            T = int(mu_full.size(1))
            T_valid = max(T - H_d, 0)
            if T_valid <= 0:
                continue
            mu_v = mu_full[:, :T_valid, :, :]
            y_v = y_plus[:, :T_valid, :, :]
            diff_sq = (mu_v - y_v).pow(2).mean(dim=(2, 3))   # (B, T_valid)
            per_anchor_mse.append(diff_sq.detach().cpu().numpy())
            processed += int(mu_v.shape[0])
            if max_samples and processed >= max_samples:
                break

    if not per_anchor_mse:
        logger.warning("anchor_error: no samples collected.")
        return pd.DataFrame()

    all_mse = np.concatenate(per_anchor_mse, axis=0)  # (N_total, T_valid)
    mse_mean = np.nanmean(all_mse, axis=0)
    mse_std = np.nanstd(all_mse, axis=0)
    n_samples = np.full(mse_mean.shape, all_mse.shape[0], dtype=int)
    t_vals = np.arange(all_mse.shape[1], dtype=int)

    df = pd.DataFrame({
        "t": t_vals,
        "mse_mean": mse_mean,
        "mse_std": mse_std,
        "n_samples": n_samples,
    })
    csv_path = output_dir / "anchor_error.csv"
    df.to_csv(csv_path, index=False)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.fill_between(
            t_vals,
            mse_mean - mse_std,
            mse_mean + mse_std,
            color=COLOR_BLUE,
            alpha=0.18,
            label="±1 std",
        )
        ax.plot(t_vals, mse_mean, color=COLOR_BLUE, linewidth=1.2, label="mean")
        if warmup > 0:
            ax.axvspan(0, min(warmup, len(t_vals) - 1), color="#CCCCCC", alpha=0.4, label="warmup")
        ax.axvline(
            x=len(t_vals) - 1,
            color=COLOR_VERMILLION,
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_xlabel("Anchor t", fontsize=FONT_LABEL)
        ax.set_ylabel("mean MSE over (h, c)", fontsize=FONT_LABEL)
        ax.set_title("Forecast error vs anchor position", fontsize=FONT_TITLE, fontweight="normal")
        ax.legend(loc="upper right", frameon=True)
        _style_axes(ax, grid="major", minor_ticks=True)
        fig.tight_layout()
        fig.savefig(output_dir / "anchor_error.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved anchor error profile to {csv_path}")
    return df
