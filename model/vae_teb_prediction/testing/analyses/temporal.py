"""
Temporal accuracy analysis for VAE-TEB models.

This module analyzes how reconstruction quality varies across the sequence,
particularly examining the effect of the warmup period.

Example:
    >>> from testing.analyses.temporal import run_temporal_accuracy_analysis
    >>> df = run_temporal_accuracy_analysis(runner, test_loader)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from ..base import TestRunner
from ..metrics import aggregate_predictions, compute_kld_per_timestep, compute_reconstruction_metrics
from ..visualizers import plot_temporal_accuracy


def run_temporal_accuracy_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 200,
) -> pd.DataFrame:
    """
    Analyze reconstruction accuracy as a function of timestep position.

    Computes VAF and SNR at each timestep within samples to understand
    how accuracy varies across the sequence. This helps identify:
    - Warmup effects
    - Edge effects at sequence boundaries
    - Temporal patterns in reconstruction quality

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process (default 200).

    Returns:
        DataFrame with columns: [timestep, vaf, snr, sample_idx]

    Example:
        >>> df = run_temporal_accuracy_analysis(runner, test_loader)
        >>> mean_by_t = df.groupby('timestep')['vaf'].mean()
    """
    logger.info(f"Running temporal accuracy analysis (max {max_samples} samples)...")

    records: List[Dict[str, Any]] = []
    processed = 0

    # Get model parameters
    warmup = runner.warmup_steps
    stride = runner.decimation_factor

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Forward pass
            outputs = runner.forward(batch)
            mu_pr = outputs.get("mu_pr")  # (B, T, H)

            if mu_pr is None:
                continue

            T = mu_pr.size(1)  # Number of timesteps
            H = mu_pr.size(2)  # Horizon per timestep
            raw_len = batch.fhr.size(1)

            # Process each sample
            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                y_raw = batch.fhr[idx]  # (raw_len,)

                # Analyze each valid timestep
                for t in range(warmup, T):
                    # Get prediction window for this timestep
                    start_raw = t * stride
                    end_raw = start_raw + H

                    if end_raw > raw_len:
                        continue  # Skip incomplete windows

                    # Extract prediction and target for this window
                    pred_window = mu_pr[idx, t, :end_raw - start_raw]
                    true_window = y_raw[start_raw:end_raw]

                    # Compute metrics for this window
                    if pred_window.shape != true_window.shape:
                        continue

                    residual = true_window - pred_window
                    mse = (residual ** 2).mean().item()
                    var_true = true_window.var().item()
                    var_res = residual.var().item()

                    # VAF
                    vaf = max(0.0, min(1.0, 1.0 - var_res / max(var_true, 1e-12)))

                    # SNR
                    signal_power = (true_window ** 2).mean().item()
                    noise_power = mse
                    snr = 10 * np.log10(max(signal_power, 1e-12) / max(noise_power, 1e-12))

                    records.append({
                        "sample_idx": processed,
                        "timestep": t,
                        "vaf": vaf,
                        "snr": snr,
                        "mse": mse,
                    })

                processed += 1

            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)

    if df.empty:
        logger.warning("No temporal accuracy data collected.")
        return df

    logger.info(f"Collected {len(df)} timestep measurements from {processed} samples")

    # Create visualization
    output_dir = runner.ensure_dir("temporal_accuracy")
    plot_temporal_accuracy(df, output_dir, warmup_steps=warmup)

    logger.info(f"Temporal accuracy plot saved to {output_dir}")

    return df


def run_within_window_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 100,
) -> pd.DataFrame:
    """
    Analyze accuracy at different positions within prediction windows.

    Examines how reconstruction quality varies from the start to the end
    of each prediction window (horizon H).

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data.
        max_samples: Maximum samples to process.

    Returns:
        DataFrame with columns: [window_position, vaf, snr, mse]

    Example:
        >>> df = run_within_window_analysis(runner, test_loader)
        >>> df.groupby('window_position')['vaf'].mean().plot()
    """
    logger.info(f"Running within-window accuracy analysis...")

    records: List[Dict[str, Any]] = []
    processed = 0

    warmup = runner.warmup_steps
    stride = runner.decimation_factor

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            outputs = runner.forward(batch)
            mu_pr = outputs.get("mu_pr")

            if mu_pr is None:
                continue

            T = mu_pr.size(1)
            H = mu_pr.size(2)
            raw_len = batch.fhr.size(1)

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                y_raw = batch.fhr[idx]

                # Use middle timesteps to avoid edge effects
                for t in range(warmup + 10, T - 10):
                    start_raw = t * stride
                    end_raw = start_raw + H

                    if end_raw > raw_len:
                        continue

                    pred = mu_pr[idx, t, :]
                    true = y_raw[start_raw:end_raw]

                    # Analyze each position within the window
                    for pos in range(min(H, len(true))):
                        err = (true[pos] - pred[pos]).item()
                        records.append({
                            "sample_idx": processed,
                            "window_position": pos,
                            "error": err,
                            "abs_error": abs(err),
                        })

                processed += 1

            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)

    if not df.empty:
        # Aggregate by window position
        agg = df.groupby("window_position").agg({
            "abs_error": ["mean", "std"],
        }).reset_index()
        agg.columns = ["window_position", "mae_mean", "mae_std"]

        logger.info(f"Within-window analysis: collected {len(df)} position measurements")

    return df
