"""
Data collection utilities for VAE-TEB testing.

This module provides reusable iteration patterns for collecting metrics,
latent representations, and predictions from a model. Each collector
returns data in a standard format (pandas DataFrame or numpy array).

Example:
    >>> from testing.collectors import collect_metrics, collect_latents
    >>> df = collect_metrics(runner, test_loader, max_samples=1000)
    >>> latents = collect_latents(runner, test_loader, max_samples=500)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .base import TestRunner
from .metrics import (
    aggregate_predictions,
    compute_kld_per_sample,
    compute_kld_per_timestep,
    compute_reconstruction_metrics,
)


def _extract_guid(batch: Any, idx: int) -> Optional[str]:
    """
    Extract GUID string from batch at given index.

    Handles various formats: tensors, numpy arrays, lists, bytes.

    Args:
        batch: Batch object with guid attribute.
        idx: Index within the batch.

    Returns:
        GUID string or None if extraction fails.
    """
    guid_attr = getattr(batch, "guid", None)
    if guid_attr is None:
        return None

    try:
        raw = guid_attr[idx]
        # Handle tensor types
        if isinstance(raw, torch.Tensor):
            raw = raw.item() if raw.numel() == 1 else int(raw.item())
        # Handle bytes
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return str(raw)
    except Exception:
        return None


def _extract_epoch(batch: Any, idx: int) -> Optional[float]:
    """
    Extract epoch (time before birth) from batch at given index.

    Args:
        batch: Batch object with epoch attribute.
        idx: Index within the batch.

    Returns:
        Epoch value in seconds (negative) or None if extraction fails.
    """
    epoch_attr = getattr(batch, "epoch", None)
    if epoch_attr is None:
        return None

    try:
        raw = epoch_attr[idx]
        if isinstance(raw, torch.Tensor):
            return float(raw.item())
        return float(raw)
    except Exception:
        return None


def _extract_label(batch: Any, idx: int) -> Optional[int]:
    """
    Extract class label from batch at given index.

    Args:
        batch: Batch object with target attribute.
        idx: Index within the batch.

    Returns:
        Class label integer or None if extraction fails.
    """
    target_attr = getattr(batch, "target", None)
    if target_attr is None:
        return None

    try:
        raw = target_attr[idx]
        if isinstance(raw, torch.Tensor):
            # Target might be per-timestep; take mode or first valid
            if raw.dim() > 0:
                # Get most common non-zero value
                nonzero = raw[raw > 0]
                if len(nonzero) > 0:
                    return int(nonzero[0].item())
                return 0
            return int(raw.item())
        return int(raw)
    except Exception:
        return None


def collect_metrics(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collect VAF, MSE, SNR, and KLD metrics for all samples.

    Iterates through the dataloader, runs model inference, and computes
    reconstruction metrics and KL divergence for each sample.

    Args:
        runner: TestRunner with model and device.
        loader: PyTorch DataLoader yielding batches.
        max_samples: Maximum samples to process. None for all.

    Returns:
        DataFrame with columns: [guid, epoch, label, vaf, mse, snr, kld]

    Example:
        >>> df = collect_metrics(runner, test_loader)
        >>> print(f"Mean VAF: {df['vaf'].mean():.4f}")
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Forward pass
            outputs = runner.forward(batch)

            # Aggregate predictions to raw signal length
            avg_pred, valid_mask = aggregate_predictions(
                runner.model, outputs.get("mu_pr"), raw_len=batch.fhr.size(1)
            )

            if avg_pred is None:
                continue

            # Compute reconstruction metrics
            metrics = compute_reconstruction_metrics(batch.fhr, avg_pred, valid_mask)

            # Compute per-sample KLD
            kld = compute_kld_per_sample(outputs, runner.warmup_steps)

            # Extract per-sample data
            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                records.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "vaf": float(metrics["vaf"][idx].cpu().item()),
                    "mse": float(metrics["mse"][idx].cpu().item()),
                    "snr": float(metrics["snr"][idx].cpu().item()),
                    "kld": float(kld[idx].cpu().item()),
                })
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)


def collect_latents(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Collect latent representations from all samples.

    Returns a flattened array of shape (N * T, D) where N is number of
    samples, T is sequence length, and D is latent dimension.

    Args:
        runner: TestRunner with model and device.
        loader: PyTorch DataLoader yielding batches.
        max_samples: Maximum samples to process. None for all.

    Returns:
        Numpy array of shape (total_timesteps, latent_dim).

    Example:
        >>> latents = collect_latents(runner, test_loader, max_samples=500)
        >>> print(f"Latent shape: {latents.shape}")
    """
    chunks: List[np.ndarray] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Forward pass
            outputs = runner.forward(batch)
            latent = outputs.get("z")

            if latent is None:
                continue

            # Convert to numpy: (B, T, D)
            latent_np = latent.detach().cpu().numpy()

            # Flatten batch and time: (B, T, D) -> (B*T, D)
            for i in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                # Reshape single sample: (T, D)
                chunks.append(latent_np[i])
                processed += 1

            if max_samples and processed >= max_samples:
                break

    if not chunks:
        return np.array([])

    # Concatenate all chunks: (N*T, D)
    return np.concatenate(chunks, axis=0)


def collect_predictions(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Collect detailed per-sample predictions with metadata.

    Returns a list of dictionaries, each containing the ground truth,
    prediction, uncertainty, latent, and metrics for one sample.

    Args:
        runner: TestRunner with model and device.
        loader: PyTorch DataLoader yielding batches.
        max_samples: Maximum samples to process. None for all.

    Returns:
        List of dicts with keys:
            - y_true: Raw FHR signal (numpy)
            - y_pred: Aggregated prediction (numpy)
            - y_pred_std: Prediction uncertainty if available (numpy)
            - latent: Latent representation (numpy)
            - guid: Patient ID
            - epoch: Time before birth in seconds
            - label: Class label
            - metrics: {vaf, mse, snr, kld}

    Example:
        >>> samples = collect_predictions(runner, test_loader, max_samples=10)
        >>> for s in samples:
        ...     print(f"GUID: {s['guid']}, VAF: {s['metrics']['vaf']:.3f}")
    """
    samples: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Forward pass
            outputs = runner.forward(batch)

            # Aggregate predictions
            avg_pred, valid_mask = aggregate_predictions(
                runner.model, outputs.get("mu_pr"), raw_len=batch.fhr.size(1)
            )

            # Aggregate variance for uncertainty
            logvar_segments = outputs.get("logvar_pr")
            avg_std = None
            if logvar_segments is not None:
                avg_var, _ = aggregate_predictions(
                    runner.model, logvar_segments.exp(), raw_len=batch.fhr.size(1)
                )
                if avg_var is not None:
                    avg_std = torch.sqrt(avg_var.clamp_min(1e-12))

            if avg_pred is None:
                continue

            # Compute metrics
            metrics = compute_reconstruction_metrics(batch.fhr, avg_pred, valid_mask)
            kld = compute_kld_per_sample(outputs, runner.warmup_steps)
            latent = outputs.get("z")

            # Extract per-sample data
            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                sample = {
                    "y_true": batch.fhr[idx].cpu().numpy(),
                    "y_pred": avg_pred[idx].cpu().numpy(),
                    "y_pred_std": avg_std[idx].cpu().numpy() if avg_std is not None else None,
                    "latent": latent[idx].cpu().numpy() if latent is not None else None,
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                    "metrics": {
                        "vaf": float(metrics["vaf"][idx].cpu().item()),
                        "mse": float(metrics["mse"][idx].cpu().item()),
                        "snr": float(metrics["snr"][idx].cpu().item()),
                        "kld": float(kld[idx].cpu().item()),
                    },
                }
                samples.append(sample)
                processed += 1

            if max_samples and processed >= max_samples:
                break

    return samples


def collect_kld_trajectory(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collect per-timestep KLD values for trajectory analysis.

    Returns a DataFrame with KLD at each timestep, along with metadata
    for grouping by patient and time before birth.

    Args:
        runner: TestRunner with model and device.
        loader: PyTorch DataLoader yielding batches.
        max_samples: Maximum samples to process. None for all.

    Returns:
        DataFrame with columns:
            - guid: Patient ID
            - epoch: Time before birth (seconds, negative)
            - hours_before: Time before birth in hours (positive)
            - label: Class label
            - timestep: Timestep index within sample
            - kld_mean: Mean KLD over latent dimensions at this timestep
            - latent_0, latent_1, ...: Individual latent dimension values

    Example:
        >>> df = collect_kld_trajectory(runner, test_loader)
        >>> df_by_hour = df.groupby('hours_before')['kld_mean'].mean()
    """
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            # Forward pass
            outputs = runner.forward(batch)

            # Compute per-timestep KLD
            kld_t = compute_kld_per_timestep(outputs, runner.warmup_steps)
            latent = outputs.get("z")

            if kld_t is None:
                continue

            T = kld_t.size(1)

            # Extract per-sample, per-timestep data
            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break

                guid = _extract_guid(batch, idx)
                epoch = _extract_epoch(batch, idx)
                label = _extract_label(batch, idx)

                # Convert epoch to hours before birth (positive)
                hours_before = -epoch / 3600.0 if epoch is not None else None

                # Get per-timestep values
                kld_vals = kld_t[idx].cpu().numpy()
                latent_vals = latent[idx].cpu().numpy() if latent is not None else None

                for t in range(T):
                    record = {
                        "guid": guid,
                        "epoch": epoch,
                        "hours_before": hours_before,
                        "label": label,
                        "timestep": t,
                        "kld_mean": float(kld_vals[t]) if np.isfinite(kld_vals[t]) else np.nan,
                    }

                    # Add individual latent dimensions
                    if latent_vals is not None:
                        for d in range(latent_vals.shape[1]):
                            record[f"latent_{d}"] = float(latent_vals[t, d])

                    records.append(record)

                processed += 1

            if max_samples and processed >= max_samples:
                break

    return pd.DataFrame(records)
