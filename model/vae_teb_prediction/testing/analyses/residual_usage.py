"""Residual-branch activity analysis for lag-attn v1.

Reports per-sample residual norms and the per-anchor residual norm
trace. Used to detect posterior / residual-head collapse (``mean
residual_ratio ≈ 0`` on a trained checkpoint is a red flag that the
source branch has been shut off).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.testing.metrics import compute_residual_usage
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLACK,
    COLOR_PURPLE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
    class_color_for,
    class_label_for,
    plot_residual_usage_trace,
    unique_labels_in,
)
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def run_residual_usage_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
    output_dir: Optional[Path] = None,
    collapse_threshold: float = 0.01,
) -> Dict[str, Any]:
    """Compute per-sample residual norms and the per-anchor trace.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("residual_usage")``).
        collapse_threshold: Samples with ``residual_ratio`` below this
            value are counted as "collapsed".

    Returns:
        Dict with ``mean_residual_ratio, n_collapsed, frac_collapsed,
        n_samples, csv``.
    """
    if max_samples <= 0:
        logger.info("residual_usage: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("residual_usage")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = []
    trace_rows = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            usage = compute_residual_usage(
                outputs["delta_mu_src"],
                outputs["mu_full"],
                runner.warmup_steps,
                runner.horizon,
            )
            batch_size = int(outputs["mu_full"].size(0))

            delta_norm = usage["delta_norm"].cpu().numpy()
            full_norm = usage["full_norm"].cpu().numpy()
            residual_ratio = usage["residual_ratio"].cpu().numpy()
            delta_norm_t = usage["delta_norm_t"].cpu().numpy()  # (B, T_valid)

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)
                per_sample_rows.append({
                    "guid": guid,
                    "epoch": epoch,
                    "label": label,
                    "delta_norm": float(delta_norm[idx]),
                    "full_norm": float(full_norm[idx]),
                    "residual_ratio": float(residual_ratio[idx]),
                })
                # Append long-format trace.
                for t in range(delta_norm_t.shape[1]):
                    trace_rows.append({
                        "guid": guid,
                        "t": t + int(runner.warmup_steps),
                        "delta_norm_t": float(delta_norm_t[idx, t]),
                    })
                processed += 1
            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(per_sample_rows)
    trace_df = pd.DataFrame(trace_rows)
    per_sample_csv = output_dir / "per_sample.csv"
    trace_csv = output_dir / "per_sample_trace.csv"
    df.to_csv(per_sample_csv, index=False)
    trace_df.to_csv(trace_csv, index=False)

    if df.empty:
        logger.warning("residual_usage: no samples collected.")
        return {"n_samples": 0, "csv": str(per_sample_csv)}

    if plt is not None:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.hist(df["residual_ratio"].dropna(), bins=40, color=COLOR_PURPLE, alpha=0.8)
        ax.axvline(collapse_threshold, color="red", linestyle="--", linewidth=0.8,
                   label=f"collapse<{collapse_threshold}")
        ax.set_xlabel("residual_ratio", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.set_title("Residual usage distribution", fontsize=FONT_TITLE, fontweight="normal")
        ax.legend(loc="upper right", frameon=True)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / "residual_ratio_hist.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    plot_residual_usage_trace(trace_df, output_dir / "delta_norm_trace.pdf")

    # Per-class variants: residual_ratio histograms side-by-side + per-class
    # trace overlay on shared axes. Only emitted when >= 2 classes present.
    classes = unique_labels_in(df.get("label"))
    if plt is not None and len(classes) >= 2:
        # Side-by-side residual_ratio histogram.
        n_cls = len(classes)
        fig, axes = plt.subplots(
            1, n_cls,
            figsize=(max(3.6, 2.8 * n_cls), 3.0),
            sharey=True, squeeze=False,
        )
        for ax, lab in zip(axes[0], classes):
            vals = df.loc[df["label"] == lab, "residual_ratio"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                ax.hist(vals, bins=30, color=class_color_for(lab),
                        alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.4)
            ax.axvline(collapse_threshold, color="red", ls="--", lw=0.7)
            ax.set_title(f"{class_label_for(lab)} (n={vals.size})",
                         fontsize=FONT_TITLE * 0.8)
            ax.set_xlabel("residual_ratio", fontsize=FONT_LABEL * 0.9)
            _style_axes(ax, grid="major", minor_ticks=False)
        axes[0, 0].set_ylabel("Count", fontsize=FONT_LABEL * 0.9)
        fig.tight_layout()
        fig.savefig(
            output_dir / "residual_ratio_hist_by_class.pdf",
            dpi=SAVE_DPI, bbox_inches="tight",
        )
        plt.close(fig)

        # Per-anchor delta_norm_t overlay by class. trace_df only has guid
        # keys — join back with df[guid, label] so we can group.
        if not trace_df.empty and "guid" in trace_df.columns:
            guid_labels = df.set_index("guid")["label"].to_dict()
            trace_lab = trace_df.copy()
            trace_lab["label"] = trace_lab["guid"].map(guid_labels)
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            for lab in classes:
                sub = trace_lab[trace_lab["label"] == lab]
                if sub.empty:
                    continue
                grouped = sub.groupby("t")["delta_norm_t"]
                med = grouped.median()
                q1 = grouped.quantile(0.25)
                q3 = grouped.quantile(0.75)
                xs = np.asarray(med.index.to_list(), dtype=float)
                color = class_color_for(lab)
                ax.plot(xs, med.to_numpy(), color=color, lw=1.2,
                        label=class_label_for(lab))
                ax.fill_between(
                    xs, q1.to_numpy(), q3.to_numpy(),
                    color=color, alpha=0.18, lw=0,
                )
            ax.set_xlabel("anchor timestep t")
            ax.set_ylabel("||delta_mu_src|| trace (median, IQR)")
            ax.set_title("Residual magnitude over time — by class")
            ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND)
            _style_axes(ax, grid="major", minor_ticks=False)
            fig.tight_layout()
            fig.savefig(
                output_dir / "delta_norm_trace_by_class.pdf",
                dpi=SAVE_DPI, bbox_inches="tight",
            )
            plt.close(fig)

    n_collapsed = int((df["residual_ratio"] < collapse_threshold).sum())
    summary = {
        "mean_residual_ratio": float(df["residual_ratio"].mean()),
        "n_collapsed": n_collapsed,
        "frac_collapsed": n_collapsed / max(len(df), 1),
        "n_samples": int(len(df)),
        "csv": str(per_sample_csv),
    }
    logger.info(
        f"residual_usage: mean_ratio={summary['mean_residual_ratio']:.4f}, "
        f"n_collapsed={summary['n_collapsed']}/{summary['n_samples']}"
    )
    return summary
