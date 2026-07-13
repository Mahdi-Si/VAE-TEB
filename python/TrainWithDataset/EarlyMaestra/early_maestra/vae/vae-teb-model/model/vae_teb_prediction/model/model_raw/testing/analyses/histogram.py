"""Histogram analysis for lag-attn v1 forecast metrics.

This module drives the histogram plot step of the pipeline: it runs
:func:`collect_metrics`, writes the resulting DataFrame to
``histograms/histogram_metrics.csv`` (with a ``kld`` column alias for
downstream TE consumers), and emits the standard multi-panel histogram
plot.

Example:
    >>> from testing.analyses.histogram import run_histogram_analysis
    >>> df = run_histogram_analysis(runner, test_loader, max_samples=1000)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import collect_metrics
from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
    plot_metric_histograms,
    plot_metric_histograms_by_class,
    unique_labels_in,
)


# Columns the histogram plot / metadata summary will inspect. Missing
# columns are skipped silently so the function tolerates partial configs.
_PLOT_COLUMNS = (
    "feat_mse_total",
    "feat_r2_total",
    "uplift_rel",
    "residual_ratio",
    "kld_mean",
    "kld_sum",
    "kld_l2",
)


def _save_histogram_data(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_identifier: Optional[str] = None,
) -> None:
    """Persist the collected metrics DataFrame and summary metadata.

    The CSV schema preserves ``guid``, ``epoch``, ``label``, and ``kld``
    (an alias of ``kld_mean``) so downstream consumers — notably
    ``TE_Calculated/te_kld_analysis.py::load_kld_from_metrics_csv`` —
    continue to work unchanged.

    Args:
        df: DataFrame produced by :func:`collect_metrics`.
        output_dir: Directory to write files into (typically
            ``runner.ensure_dir("histograms")``).
        dataset_identifier: Optional identifier stored in the metadata
            JSON (e.g. ``"fold_1"``, ``"test_set"``).
    """
    csv_path = output_dir / "histogram_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    summary_stats = {}
    for metric in _PLOT_COLUMNS:
        if metric in df.columns:
            series = df[metric].dropna()
            if len(series) == 0:
                continue
            summary_stats[metric] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
            }

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(df)),
        "dataset_identifier": dataset_identifier,
        "metrics_included": list(summary_stats.keys()),
        "data_file": "histogram_metrics.csv",
        "summary_statistics": summary_stats,
    }

    json_path = output_dir / "histogram_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")


def run_histogram_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    save_data: bool = True,
    dataset_identifier: Optional[str] = None,
) -> pd.DataFrame:
    """Collect per-sample metrics and render the histogram panel.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process (None = all).
        save_data: Persist the metrics DataFrame and metadata JSON.
        dataset_identifier: Optional tag saved in the metadata JSON.

    Returns:
        DataFrame returned by :func:`collect_metrics` (new schema with
        ``kld`` alias column for backward compatibility).
    """
    logger.info("Starting histogram analysis...")

    df = collect_metrics(runner, loader, max_samples)

    if df.empty:
        logger.error("No metrics collected — check the dataloader and model outputs.")
        return df

    logger.info(
        f"Collected {len(df)} samples: "
        f"feat_mse={df['feat_mse_total'].mean():.4f}±{df['feat_mse_total'].std():.4f}, "
        f"uplift_rel={df['uplift_rel'].mean():.3f}, "
        f"residual_ratio={df['residual_ratio'].mean():.3f}, "
        f"kld_mean={df['kld_mean'].mean():.4f}"
    )

    output_dir = runner.ensure_dir("histograms")
    plot_metric_histograms(df, output_dir)

    # Per-class panel: one subplot grid per class when >=2 classes are
    # present in the test set (auto-detected from the ``label`` column).
    classes_present = unique_labels_in(df.get("label"))
    if len(classes_present) >= 2:
        plot_metric_histograms_by_class(df, output_dir)
        logger.info(
            f"Per-class histograms emitted for classes {classes_present}"
        )

    if save_data:
        _save_histogram_data(df, output_dir, dataset_identifier)

    logger.info(f"Histogram plot saved to {output_dir}")
    return df
