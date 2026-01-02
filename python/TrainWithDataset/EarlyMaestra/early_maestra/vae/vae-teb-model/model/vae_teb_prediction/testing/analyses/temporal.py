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

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.metrics import aggregate_predictions, compute_kld_per_timestep, compute_reconstruction_metrics
from model.vae_teb_prediction.testing.visualizers import plot_temporal_accuracy, plot_within_window_accuracy


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
        DataFrame with columns:
            - window_position
            - mae_mean, mae_std
            - vaf_mean, snr_mean
            - count

    Example:
        >>> df = run_within_window_analysis(runner, test_loader)
        >>> df.groupby('window_position')['vaf'].mean().plot()
    """
    logger.info(f"Running within-window accuracy analysis...")

    processed = 0

    warmup = runner.warmup_steps
    stride = runner.decimation_factor

    sum_abs_error = None
    sum_abs_error_sq = None
    sum_true = None
    sum_true_sq = None
    sum_res = None
    sum_res_sq = None
    counts = None

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

            if sum_abs_error is None:
                sum_abs_error = np.zeros(H, dtype=np.float64)
                sum_abs_error_sq = np.zeros(H, dtype=np.float64)
                sum_true = np.zeros(H, dtype=np.float64)
                sum_true_sq = np.zeros(H, dtype=np.float64)
                sum_res = np.zeros(H, dtype=np.float64)
                sum_res_sq = np.zeros(H, dtype=np.float64)
                counts = np.zeros(H, dtype=np.int64)

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

                    pred_np = pred.detach().cpu().numpy()
                    true_np = true.detach().cpu().numpy()
                    length = min(len(pred_np), len(true_np), H)
                    if length <= 0:
                        continue

                    pred_np = pred_np[:length]
                    true_np = true_np[:length]
                    residual = true_np - pred_np

                    sum_abs_error[:length] += np.abs(residual)
                    sum_abs_error_sq[:length] += residual ** 2
                    sum_true[:length] += true_np
                    sum_true_sq[:length] += true_np ** 2
                    sum_res[:length] += residual
                    sum_res_sq[:length] += residual ** 2
                    counts[:length] += 1

                processed += 1

            if max_samples and processed >= max_samples:
                break

    if counts is None or not np.any(counts):
        logger.warning("No within-window accuracy data collected.")
        return pd.DataFrame()

    count_safe = np.maximum(counts, 1)
    mae_mean = sum_abs_error / count_safe
    mae_var = sum_abs_error_sq / count_safe - mae_mean ** 2
    mae_std = np.sqrt(np.maximum(mae_var, 0.0))

    mean_true = sum_true / count_safe
    var_true = sum_true_sq / count_safe - mean_true ** 2
    mean_res = sum_res / count_safe
    var_res = sum_res_sq / count_safe - mean_res ** 2
    var_true = np.maximum(var_true, 1e-12)
    var_res = np.maximum(var_res, 0.0)

    vaf_mean = np.clip(1.0 - var_res / var_true, 0.0, 1.0)
    signal_power = sum_true_sq / count_safe
    noise_power = sum_res_sq / count_safe
    snr_mean = 10.0 * np.log10(np.maximum(signal_power, 1e-12) / np.maximum(noise_power, 1e-12))

    agg = pd.DataFrame({
        "window_position": np.arange(len(count_safe)),
        "mae_mean": mae_mean,
        "mae_std": mae_std,
        "vaf_mean": vaf_mean,
        "snr_mean": snr_mean,
        "count": counts,
    })

    logger.info("Within-window analysis: aggregated %d samples", int(np.max(counts)))

    output_dir = runner.ensure_dir("temporal_accuracy")
    plot_within_window_accuracy(agg, output_dir)

    return agg
