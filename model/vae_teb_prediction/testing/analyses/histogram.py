"""
Histogram analysis for VAE-TEB reconstruction metrics.

This module provides a complete histogram analysis pipeline that collects
VAF, MSE, SNR, and KLD metrics and creates publication-quality histograms.

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

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_metrics
from model.vae_teb_prediction.testing.visualizers import plot_metric_histograms


def _save_histogram_data(
    df: pd.DataFrame,
    output_dir: Path,
    dataset_identifier: Optional[str] = None,
) -> None:
    """
    Save histogram metrics DataFrame and metadata as CSV.

    Args:
        df: DataFrame with columns [guid, epoch, label, vaf, mse, snr, kld]
        output_dir: Directory to save files (typically runner.ensure_dir("histograms"))
        dataset_identifier: Optional identifier for this dataset (e.g., "fold_1", "test_set")

    Outputs:
        - histogram_metrics.csv: Raw metrics data
        - histogram_metadata.json: Summary statistics and metadata
    """
    # Save metrics CSV
    csv_path = output_dir / "histogram_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")

    # Compute summary statistics for metadata
    summary_stats = {}
    for metric in ["vaf", "mse", "snr", "kld"]:
        if metric in df.columns:
            summary_stats[metric] = {
                "mean": float(df[metric].mean()),
                "std": float(df[metric].std()),
                "median": float(df[metric].median()),
                "min": float(df[metric].min()),
                "max": float(df[metric].max()),
            }

    # Save metadata JSON
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(df),
        "dataset_identifier": dataset_identifier,
        "metrics_included": ["vaf", "mse", "snr", "kld"],
        "data_file": "histogram_metrics.csv",
        "summary_statistics": summary_stats,
    }

    json_path = output_dir / "histogram_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {json_path}")


def run_histogram_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    save_data: bool = True,
    dataset_identifier: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run complete histogram analysis for reconstruction metrics.

    Collects VAF, MSE, SNR, and KLD metrics from the test set and creates
    a 2x2 histogram plot with statistics annotations. Optionally saves the
    underlying data as CSV for downstream statistical analysis.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process. None for all.
        save_data: Whether to save metrics DataFrame and metadata. Default True.
        dataset_identifier: Optional identifier for this dataset (e.g., "fold_1").
            Saved in metadata JSON for tracking purposes.

    Returns:
        DataFrame with columns: [guid, epoch, label, vaf, mse, snr, kld]

    Outputs (if save_data=True):
        - {output_dir}/histograms/histogram_metrics.csv: Metrics data
        - {output_dir}/histograms/histogram_metadata.json: Summary statistics

    Example:
        >>> runner = TestRunner.from_checkpoint("model.ckpt", "results/")
        >>> df = run_histogram_analysis(runner, test_loader, save_data=True,
        ...                             dataset_identifier="test_fold_1")
        >>> print(f"Mean VAF: {df['vaf'].mean():.4f}")
    """
    logger.info("Starting histogram analysis...")

    # Collect metrics from all samples
    df = collect_metrics(runner, loader, max_samples)

    if df.empty:
        logger.error("No metrics collected - check dataloader and model outputs.")
        return df

    # Log summary statistics
    logger.info(
        f"Collected {len(df)} samples: "
        f"VAF={df['vaf'].mean():.4f}±{df['vaf'].std():.4f}, "
        f"MSE={df['mse'].mean():.6f}, "
        f"SNR={df['snr'].mean():.2f} dB, "
        f"KLD={df['kld'].mean():.6f}"
    )

    # Create histogram plot
    output_dir = runner.ensure_dir("histograms")
    plot_metric_histograms(df, output_dir)

    # Save data if requested
    if save_data:
        _save_histogram_data(df, output_dir, dataset_identifier)

    logger.info(f"Histogram plot saved to {output_dir}")

    return df
