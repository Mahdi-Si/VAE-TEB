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
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    collect_metrics,
)
from model.vae_teb_prediction.testing.plot_single_samples import (
    _get_normalization_stats,
    _plot_all_single_sample_plots,
    _sanitize_folder_name,
)
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


def _plot_extreme_samples(
    runner: TestRunner,
    loader: Any,
    df: pd.DataFrame,
    n_extreme: int = 10,
) -> None:
    """
    Plot single-sample reconstructions for extreme metric values.

    For each metric (vaf, mse, snr, kld), selects the top-N highest and
    bottom-N lowest samples, then generates a consolidated reconstruction
    plot for each.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for the test data (must match df row order).
        df: DataFrame from collect_metrics with columns [guid, epoch, label, vaf, mse, snr, kld].
        n_extreme: Number of top/bottom samples per metric.
    """
    metrics = ["vaf", "mse", "snr", "kld"]

    # Build mapping: dataset index -> list of (metric, direction) categories
    index_categories: Dict[int, List[str]] = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        n = min(n_extreme, len(df))
        sorted_idx = df[metric].sort_values()
        bottom_indices = sorted_idx.head(n).index.tolist()
        top_indices = sorted_idx.tail(n).index.tolist()

        for idx in bottom_indices:
            index_categories.setdefault(idx, []).append(f"{metric}_low")
        for idx in top_indices:
            index_categories.setdefault(idx, []).append(f"{metric}_high")

    if not index_categories:
        logger.warning("No extreme samples to plot.")
        return

    unique_indices = sorted(index_categories.keys())
    logger.info(
        f"Plotting extreme samples: {len(unique_indices)} unique samples "
        f"across {len(metrics)} metrics x 2 directions"
    )

    # Create a Subset with only the extreme samples
    subset = Subset(loader.dataset, unique_indices)
    subset_loader = DataLoader(
        subset,
        batch_size=loader.batch_size or 1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=getattr(loader, "collate_fn", None),
    )

    # Map from position in subset back to original dataset index
    subset_pos_to_orig = {pos: orig_idx for pos, orig_idx in enumerate(unique_indices)}

    stats = _get_normalization_stats(loader)
    base_dir = runner.ensure_dir("histograms") / "extreme_samples"

    current_subset_pos = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(subset_loader, max_samples=None):
            batch_size = batch.fhr_st.size(0)
            outputs = runner.forward(batch)

            for idx_in_batch in range(batch_size):
                orig_idx = subset_pos_to_orig[current_subset_pos]
                categories = index_categories[orig_idx]

                guid = _extract_guid(batch, idx_in_batch)
                epoch = _extract_epoch(batch, idx_in_batch)
                sample_name = _sanitize_folder_name(
                    guid or f"sample_{orig_idx}", epoch or 0.0
                )

                for category in categories:
                    sample_dir = base_dir / category
                    _plot_all_single_sample_plots(
                        runner=runner,
                        batch=batch,
                        idx=idx_in_batch,
                        outputs=outputs,
                        sample_dir=sample_dir,
                        sample_name=sample_name,
                        stats=stats,
                    )
                    logger.debug(
                        f"Plotted extreme sample {sample_name} -> {category}"
                    )

                current_subset_pos += 1

    logger.info(f"Extreme sample plots saved to {base_dir}")


def run_histogram_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    save_data: bool = True,
    dataset_identifier: Optional[str] = None,
    plot_extremes: bool = True,
    n_extreme: int = 10,
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
        plot_extremes: Whether to generate single-sample plots for extreme metric
            values (top/bottom N per metric). Default True.
        n_extreme: Number of top/bottom samples per metric to plot. Default 10.

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

    # Plot extreme samples if enabled
    if plot_extremes:
        _plot_extreme_samples(runner, loader, df, n_extreme=n_extreme)

    return df
