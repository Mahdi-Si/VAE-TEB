"""Lag-attention diagnostic analysis for lag-attn v1.

Summarises the causal lag attention produced by
:class:`SeqVaeLagAttnV1.lag_attn`. For a batch of samples it collects the
head-averaged attention map, argmax lag per anchor, per-head entropy,
inter-head diversity, and time-averaged "attention mass by lag"; then
writes CSVs and plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_attention_maps
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLACK,
    COLOR_BLUE,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
    class_color_for,
    class_label_for,
    plot_attention_mass_by_lag,
    plot_lag_attention_heatmap,
    unique_labels_in,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def run_attention_diagnostics(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 200,
    output_dir: Optional[Path] = None,
    n_heatmap_examples: int = 6,
) -> Dict[str, Any]:
    """Run lag-attention diagnostics.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("attention")``).
        n_heatmap_examples: How many per-sample attention heatmaps to
            save (bounded above by the number of collected samples).

    Returns:
        Dict with summary statistics.
    """
    if max_samples <= 0:
        logger.info("attention_diagnostics: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("attention")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_attention_maps(runner, loader, max_samples)
    if not records:
        logger.warning("attention_diagnostics: no attention maps collected.")
        return {"n_samples": 0}

    # Long-format argmax-lag table.
    argmax_rows = []
    mass_rows = []
    entropy_rows = []
    all_valid_argmax = []
    for rec in records:
        guid = rec.get("guid")
        epoch = rec.get("epoch")
        label = rec.get("label")
        argmax_lag = rec["argmax_lag"]  # (T,)
        entropy = rec["entropy"]        # (T, M)
        mass = rec["alpha_mass_by_lag"]  # (L,)

        # Per-anchor rows (skip warmup sentinels of -1).
        for t, lag in enumerate(argmax_lag):
            if lag < 0:
                continue
            argmax_rows.append({
                "guid": guid,
                "epoch": epoch,
                "label": label,
                "t": int(t),
                "argmax_lag": int(lag),
            })
            all_valid_argmax.append(int(lag))

        mass_row = {"guid": guid, "epoch": epoch, "label": label}
        for k in range(mass.shape[0]):
            mass_row[f"mass_lag_{k}"] = float(mass[k])
        mass_rows.append(mass_row)

        entropy_rows.append({
            "guid": guid,
            "epoch": epoch,
            "label": label,
            "mean_entropy": float(np.nanmean(entropy)),
            "median_entropy": float(np.nanmedian(entropy)),
        })

    argmax_df = pd.DataFrame(argmax_rows)
    mass_df = pd.DataFrame(mass_rows)
    entropy_df = pd.DataFrame(entropy_rows)

    argmax_df.to_csv(output_dir / "argmax_lag_per_sample.csv", index=False)
    mass_df.to_csv(output_dir / "alpha_mass_by_lag.csv", index=False)
    entropy_df.to_csv(output_dir / "head_entropy_summary.csv", index=False)

    # Heatmaps for a handful of representative samples.
    n_heatmap = min(n_heatmap_examples, len(records))
    for i in range(n_heatmap):
        rec = records[i]
        guid = rec.get("guid", f"sample_{i}")
        safe_guid = str(guid).replace("/", "_")
        plot_lag_attention_heatmap(
            rec["alpha_bar"],
            rec["argmax_lag"],
            warmup=int(runner.warmup_steps),
            output_path=output_dir / f"attention_heatmap_{safe_guid}.pdf",
            title=f"Lag attention - {guid}",
        )

    if plt is not None and all_valid_argmax:
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.hist(all_valid_argmax, bins=50, color=COLOR_BLUE, alpha=0.85)
        ax.set_xlabel("argmax lag k", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.set_title(
            "Argmax-lag distribution (valid anchors)",
            fontsize=FONT_TITLE, fontweight="normal",
        )
        _style_axes(ax, grid="major", minor_ticks=True)
        fig.tight_layout()
        fig.savefig(output_dir / "argmax_lag_histogram.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    plot_attention_mass_by_lag(mass_df, output_dir / "attention_mass_by_lag_bars.pdf")

    if plt is not None:
        # Head diversity histogram.
        head_div_vals = []
        for rec in records:
            arr = rec.get("head_diversity")
            if arr is None:
                continue
            head_div_vals.extend(arr[np.isfinite(arr)].tolist())
        if head_div_vals:
            fig, ax = plt.subplots(figsize=(5.4, 3.2))
            ax.hist(head_div_vals, bins=40, color=COLOR_BLUE, alpha=0.8)
            ax.set_xlabel("head_diversity", fontsize=FONT_LABEL)
            ax.set_ylabel("Count", fontsize=FONT_LABEL)
            ax.set_title("Inter-head diversity", fontsize=FONT_TITLE, fontweight="normal")
            _style_axes(ax, grid="major", minor_ticks=False)
            fig.tight_layout()
            fig.savefig(output_dir / "head_diversity_hist.pdf", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

    # Per-class argmax-lag histogram and entropy boxplot when >=2 labels.
    classes = unique_labels_in(argmax_df.get("label") if not argmax_df.empty else None)
    if plt is not None and len(classes) >= 2 and not argmax_df.empty:
        # Side-by-side argmax-lag histogram per class.
        n_cls = len(classes)
        fig, axes = plt.subplots(
            1, n_cls,
            figsize=(max(3.6, 2.8 * n_cls), 3.0),
            sharey=True, squeeze=False,
        )
        for ax, lab in zip(axes[0], classes):
            vals = argmax_df.loc[
                argmax_df["label"] == lab, "argmax_lag"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes)
            else:
                ax.hist(vals, bins=50, color=class_color_for(lab),
                        alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.3)
            ax.set_title(f"{class_label_for(lab)} (n={vals.size})",
                         fontsize=FONT_TITLE * 0.8)
            ax.set_xlabel("argmax lag k", fontsize=FONT_LABEL * 0.9)
            _style_axes(ax, grid="major", minor_ticks=False)
        axes[0, 0].set_ylabel("Count", fontsize=FONT_LABEL * 0.9)
        fig.tight_layout()
        fig.savefig(
            output_dir / "argmax_lag_histogram_by_class.pdf",
            dpi=SAVE_DPI, bbox_inches="tight",
        )
        plt.close(fig)

        # Attention mass by lag, overlaid line plot per class.
        if not mass_df.empty:
            lag_cols = sorted(
                (c for c in mass_df.columns if c.startswith("mass_lag_")),
                key=lambda c: int(c.split("_")[-1]),
            )
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            for lab in classes:
                sub = mass_df[mass_df["label"] == lab]
                if sub.empty:
                    continue
                mean_vec = sub[lag_cols].to_numpy(dtype=float).mean(axis=0)
                color = class_color_for(lab)
                ax.plot(
                    np.arange(len(mean_vec)),
                    mean_vec,
                    color=color, lw=1.2,
                    label=f"{class_label_for(lab)} (n={len(sub)})",
                )
            ax.set_xlabel("lag index k", fontsize=FONT_LABEL)
            ax.set_ylabel("mean attention mass", fontsize=FONT_LABEL)
            ax.set_title("Attention mass by lag — per class",
                         fontsize=FONT_TITLE, fontweight="normal")
            ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND)
            _style_axes(ax, grid="major", minor_ticks=False)
            fig.tight_layout()
            fig.savefig(
                output_dir / "attention_mass_by_lag_by_class.pdf",
                dpi=SAVE_DPI, bbox_inches="tight",
            )
            plt.close(fig)

    median_argmax = int(np.median(all_valid_argmax)) if all_valid_argmax else None
    mean_entropy = float(entropy_df["mean_entropy"].mean()) if not entropy_df.empty else None

    summary = {
        "n_samples": int(len(records)),
        "median_argmax_lag": median_argmax,
        "mean_head_entropy": mean_entropy,
    }
    logger.info(
        f"attention_diagnostics: n={summary['n_samples']}, "
        f"median_argmax_lag={median_argmax}, mean_entropy={mean_entropy}"
    )
    return summary
