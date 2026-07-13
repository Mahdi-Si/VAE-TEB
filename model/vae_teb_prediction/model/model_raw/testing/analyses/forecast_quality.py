"""Feature-forecast quality analysis for lag-attn v1.

Evaluates the model's full feature forecast against the unfolded future
target across the valid anchor range, saves per-sample and per-horizon
CSVs, and renders the standard ribbon plot + histograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
    collect_forecast_errors_per_horizon,
)
from model.vae_teb_prediction.model.model_raw.testing.metrics import compute_forecast_metrics
from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GREEN,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
    class_color_for,
    class_label_for,
    plot_forecast_error_by_horizon,
    unique_labels_in,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def run_forecast_quality_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run per-sample feature-forecast evaluation.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("forecast_quality")``).

    Returns:
        Dict with summary statistics ``mean_mse_total, mean_r2,
        mean_mse_per_horizon`` (the last as a numpy array).
    """
    if max_samples <= 0:
        logger.info("forecast_quality: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("forecast_quality")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample metrics.
    records = []
    per_horizon_mean_accum = None
    n_accum = 0
    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)
            fcst = compute_forecast_metrics(
                outputs["mu_full"], y_plus, runner.warmup_steps, runner.horizon
            )

            batch_size = int(outputs["mu_full"].size(0))
            feat_mse_total = fcst["feat_mse_total"].cpu().numpy()
            feat_mse_st = fcst["feat_mse_st"].cpu().numpy()
            feat_mse_ph = fcst["feat_mse_ph"].cpu().numpy()
            feat_r2 = fcst["feat_r2_total"].cpu().numpy()
            feat_mse_per_horizon = fcst["feat_mse_per_horizon"].cpu().numpy()  # (B, H_d)

            if per_horizon_mean_accum is None:
                per_horizon_mean_accum = feat_mse_per_horizon.sum(axis=0)
            else:
                per_horizon_mean_accum += feat_mse_per_horizon.sum(axis=0)
            n_accum += batch_size

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                records.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "feat_mse_total": float(feat_mse_total[idx]),
                    "feat_mse_st": float(feat_mse_st[idx]),
                    "feat_mse_ph": float(feat_mse_ph[idx]),
                    "feat_r2_total": float(feat_r2[idx]),
                })
                processed += 1
            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)
    per_sample_csv = output_dir / "forecast_per_sample.csv"
    df.to_csv(per_sample_csv, index=False)

    # Per-horizon details (long format).
    per_horizon_df = collect_forecast_errors_per_horizon(runner, loader, max_samples)
    per_horizon_csv = output_dir / "forecast_per_horizon.csv"
    per_horizon_df.to_csv(per_horizon_csv, index=False)

    # Plots.
    plot_forecast_error_by_horizon(per_horizon_df, output_dir / "forecast_error_by_horizon.pdf")

    if plt is not None and not df.empty:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.hist(df["feat_mse_total"].dropna(), bins=40, color=COLOR_BLUE, alpha=0.8)
        ax.set_xlabel("feat_mse_total", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.set_title("Per-sample forecast MSE", fontsize=FONT_TITLE, fontweight="normal")
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / "feat_mse_total_hist.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.hist(df["feat_r2_total"].dropna(), bins=40, color=COLOR_GREEN, alpha=0.8)
        ax.set_xlabel("feat_r2_total", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.set_title("Per-sample forecast R²", fontsize=FONT_TITLE, fontweight="normal")
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / "feat_r2_total_hist.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Per-class variants (emitted only when >= 2 classes are present).
    classes = unique_labels_in(df.get("label") if not df.empty else None)
    if plt is not None and len(classes) >= 2:
        # Horizon-error overlay: one line per class on shared axes.
        if not per_horizon_df.empty and "h" in per_horizon_df.columns:
            fig, ax = plt.subplots(figsize=(6.0, 3.4))
            for lab in classes:
                sub = per_horizon_df[per_horizon_df["label"] == lab]
                if sub.empty:
                    continue
                grouped = sub.groupby("h")["mse_step"]
                med = grouped.median()
                q1 = grouped.quantile(0.25)
                q3 = grouped.quantile(0.75)
                xs = np.asarray(med.index.to_list(), dtype=float)
                color = class_color_for(lab)
                n_guids = int(pd.Series(sub["guid"]).nunique())
                ax.plot(
                    xs, med.to_numpy(), color=color, lw=1.2,
                    label=f"{class_label_for(lab)} (n_guids={n_guids})",
                )
                ax.fill_between(
                    xs, q1.to_numpy(), q3.to_numpy(),
                    color=color, alpha=0.18, lw=0,
                )
            ax.set_xlabel("horizon step h")
            ax.set_ylabel("per-step MSE (median, IQR)")
            ax.set_title("Forecast error by horizon — by class")
            ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND)
            _style_axes(ax, grid="major", minor_ticks=False)
            fig.tight_layout()
            fig.savefig(
                output_dir / "forecast_error_by_horizon_by_class.pdf",
                dpi=SAVE_DPI, bbox_inches="tight",
            )
            plt.close(fig)

        # Per-class MSE / R2 distributions: one subplot per class side-by-side.
        for metric, fname in (
            ("feat_mse_total", "feat_mse_total_hist_by_class.pdf"),
            ("feat_r2_total", "feat_r2_total_hist_by_class.pdf"),
        ):
            if metric not in df.columns:
                continue
            n_cls = len(classes)
            fig, axes = plt.subplots(
                1, n_cls,
                figsize=(max(3.6, 2.8 * n_cls), 3.2),
                sharey=True, squeeze=False,
            )
            for ax, lab in zip(axes[0], classes):
                vals = df.loc[df["label"] == lab, metric].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes)
                else:
                    ax.hist(vals, bins=30, color=class_color_for(lab),
                            alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.4)
                ax.set_title(f"{class_label_for(lab)} (n={vals.size})",
                             fontsize=FONT_TITLE * 0.8)
                ax.set_xlabel(metric, fontsize=FONT_LABEL * 0.9)
                _style_axes(ax, grid="major", minor_ticks=False)
            axes[0, 0].set_ylabel("Count", fontsize=FONT_LABEL * 0.9)
            fig.tight_layout()
            fig.savefig(output_dir / fname, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

    mean_per_horizon = (
        (per_horizon_mean_accum / max(n_accum, 1)) if per_horizon_mean_accum is not None else np.array([])
    )
    summary = {
        "mean_mse_total": float(df["feat_mse_total"].mean()) if not df.empty else None,
        "mean_r2": float(df["feat_r2_total"].mean()) if not df.empty else None,
        "mean_mse_per_horizon": mean_per_horizon.tolist() if isinstance(mean_per_horizon, np.ndarray) else [],
        "n_samples": int(len(df)),
        "per_sample_csv": str(per_sample_csv),
        "per_horizon_csv": str(per_horizon_csv),
    }
    logger.info(f"forecast_quality: n={summary['n_samples']} mean_mse={summary['mean_mse_total']}")
    return summary
