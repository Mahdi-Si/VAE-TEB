"""Baseline-vs-full uplift analysis for lag-attn v1.

Quantifies the improvement the source/latent branch provides over the
FHR-only baseline forecast. A positive ``uplift_rel`` means the source
stream is helping; values near zero on a trained checkpoint indicate
the residual head has collapsed (early-warning indicator).
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
from model.vae_teb_prediction.testing.metrics import compute_uplift_metrics
from model.vae_teb_prediction.testing.visualizers import (
    plot_feature_boxplots,
    plot_uplift_histogram,
)


def run_uplift_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 500,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run per-sample baseline-vs-full uplift analysis.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("uplift")``).

    Returns:
        Dict with ``mean_uplift_rel, frac_positive_uplift, by_class,
        n_samples, csv``.
    """
    if max_samples <= 0:
        logger.info("uplift: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("uplift")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            y_plus = runner.build_future_target(batch)
            up = compute_uplift_metrics(
                outputs["mu_full"],
                outputs["mu_base"],
                y_plus,
                runner.warmup_steps,
                runner.horizon,
            )
            batch_size = int(outputs["mu_full"].size(0))
            l_full = up["l_full"].cpu().numpy()
            l_base = up["l_base"].cpu().numpy()
            uplift_abs = up["uplift_abs"].cpu().numpy()
            uplift_rel = up["uplift_rel"].cpu().numpy()

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                records.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "l_full": float(l_full[idx]),
                    "l_base": float(l_base[idx]),
                    "uplift_abs": float(uplift_abs[idx]),
                    "uplift_rel": float(uplift_rel[idx]),
                })
                processed += 1
            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)
    per_sample_csv = output_dir / "per_sample.csv"
    df.to_csv(per_sample_csv, index=False)

    if df.empty:
        logger.warning("uplift: no samples collected.")
        return {"n_samples": 0, "csv": str(per_sample_csv)}

    plot_uplift_histogram(df, output_dir / "uplift_histogram.pdf")

    # Class-grouped boxplot of uplift_rel.
    has_labels = "label" in df.columns and bool(df["label"].notna().any())
    if has_labels:
        df_plot = df.copy()
        df_plot["class"] = df_plot["label"].astype("Int64").astype(str)
        try:
            plot_feature_boxplots(
                df_plot,
                feature_cols=["uplift_rel"],
                output_path=output_dir / "uplift_rel_by_class.pdf",
                class_col="class",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"uplift: plot_feature_boxplots failed: {exc}")

    by_class: Dict[int, float] = {}
    if has_labels:
        for lab, group in df.groupby("label"):
            if lab is None:
                continue
            try:
                by_class[int(str(lab))] = float(group["uplift_rel"].mean())
            except (TypeError, ValueError):
                continue

    summary = {
        "mean_uplift_rel": float(df["uplift_rel"].mean()),
        "frac_positive_uplift": float((df["uplift_rel"] > 0).mean()),
        "by_class": by_class,
        "n_samples": int(len(df)),
        "csv": str(per_sample_csv),
    }
    logger.info(
        f"uplift: mean_uplift_rel={summary['mean_uplift_rel']:.4f}, "
        f"frac_pos={summary['frac_positive_uplift']:.3f}, n={summary['n_samples']}"
    )
    return summary
