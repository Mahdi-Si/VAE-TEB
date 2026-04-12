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
    COLOR_PURPLE,
    FONT_LABEL,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
    plot_residual_usage_trace,
)

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
