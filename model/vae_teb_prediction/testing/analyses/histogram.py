"""
Histogram analysis for VAE-TEB reconstruction metrics.

This module provides a complete histogram analysis pipeline that collects
VAF, MSE, SNR, and KLD metrics and creates publication-quality histograms.

Example:
    >>> from testing.analyses.histogram import run_histogram_analysis
    >>> df = run_histogram_analysis(runner, test_loader, max_samples=1000)
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_metrics
from model.vae_teb_prediction.testing.visualizers import plot_metric_histograms


def run_histogram_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run complete histogram analysis for reconstruction metrics.

    Collects VAF, MSE, SNR, and KLD metrics from the test set and creates
    a 2x2 histogram plot with statistics annotations.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process. None for all.

    Returns:
        DataFrame with columns: [guid, epoch, label, vaf, mse, snr, kld]

    Example:
        >>> runner = TestRunner.from_checkpoint("model.ckpt", "results/")
        >>> df = run_histogram_analysis(runner, test_loader)
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

    logger.info(f"Histogram plot saved to {output_dir}")

    return df
