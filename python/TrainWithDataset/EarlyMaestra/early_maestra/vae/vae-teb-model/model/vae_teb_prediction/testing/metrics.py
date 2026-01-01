"""
Pure metric computation functions for VAE-TEB testing.

This module provides stateless functions for computing reconstruction quality
metrics (VAF, MSE, SNR) and KL divergence for transfer entropy measurement.
All functions are pure with no side effects, making them easy to test.

Metrics:
    - VAF (Variance Accounted For): 1 - var(residual) / var(target), range [0, 1]
    - MSE (Mean Squared Error): mean((target - prediction)^2)
    - SNR (Signal-to-Noise Ratio): 10 * log10(signal_power / noise_power) in dB
    - KLD (KL Divergence): KL(q(z|x,y) || p(z|y)) for transfer entropy

Example:
    >>> from testing.metrics import compute_reconstruction_metrics, compute_kld
    >>> metrics = compute_reconstruction_metrics(y_true, y_pred)
    >>> print(f"VAF: {metrics['vaf'].mean():.4f}")
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def compute_reconstruction_metrics(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Compute per-sample reconstruction quality metrics.

    Calculates VAF, MSE, and SNR for each sample in the batch by comparing
    the true signal with the model's prediction.

    Args:
        y_true: Ground truth signal tensor of shape (B, L) where B is batch
            size and L is signal length.
        y_pred: Predicted signal tensor of shape (B, L), must match y_true.
        mask: Optional boolean mask of shape (B, L). If provided, metrics are
            computed only over True positions.

    Returns:
        Dict with keys:
            - 'vaf': Variance Accounted For, shape (B,), range [0, 1]
            - 'mse': Mean Squared Error, shape (B,)
            - 'snr': Signal-to-Noise Ratio in dB, shape (B,)

    Raises:
        ValueError: If y_true and y_pred shapes don't match.

    Example:
        >>> y_true = torch.randn(32, 4800)
        >>> y_pred = torch.randn(32, 4800)
        >>> metrics = compute_reconstruction_metrics(y_true, y_pred)
        >>> print(metrics['vaf'].shape)  # torch.Size([32])
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Compute residual (error signal)
    residual = y_true - y_pred

    # Determine reduction dimensions (all except batch)
    reduce_dims = tuple(range(1, residual.ndim))

    if mask is not None:
        # Masked computation: weight contributions by mask
        if mask.shape != y_true.shape:
            raise ValueError(f"Mask shape {mask.shape} must match tensor shape {y_true.shape}")

        weight = mask.float()
        denom = weight.sum(dim=reduce_dims).clamp_min(1.0)

        # Weighted MSE
        mse = ((residual ** 2) * weight).sum(dim=reduce_dims) / denom

        # Weighted signal and noise power for SNR
        signal_power = ((y_true ** 2) * weight).sum(dim=reduce_dims) / denom
        noise_power = mse  # Same as weighted squared residual mean

        # Weighted variance for VAF
        mean_true = (y_true * weight).sum(dim=reduce_dims) / denom
        mean_res = (residual * weight).sum(dim=reduce_dims) / denom
        var_true = ((y_true ** 2) * weight).sum(dim=reduce_dims) / denom - mean_true ** 2
        var_res = ((residual ** 2) * weight).sum(dim=reduce_dims) / denom - mean_res ** 2
    else:
        # Unmasked computation: simple means over signal length
        mse = (residual ** 2).mean(dim=reduce_dims)
        signal_power = (y_true ** 2).mean(dim=reduce_dims)
        noise_power = mse
        var_true = y_true.var(dim=reduce_dims, unbiased=False)
        var_res = residual.var(dim=reduce_dims, unbiased=False)

    # Clamp variances to avoid division by zero
    var_true = var_true.clamp_min(1e-12)
    var_res = var_res.clamp_min(1e-12)

    # VAF: proportion of variance explained, clamped to [0, 1]
    vaf = (1.0 - var_res / var_true).clamp(0.0, 1.0)

    # SNR: signal-to-noise ratio in decibels
    # Handle case where noise is near zero (perfect reconstruction)
    snr = torch.where(
        noise_power > 1e-12,
        10.0 * torch.log10(signal_power.clamp_min(1e-12) / noise_power.clamp_min(1e-12)),
        torch.full_like(signal_power, 100.0),  # Cap at 100 dB for near-perfect
    )

    return {"vaf": vaf, "mse": mse, "snr": snr}


def compute_kld(
    outputs: Dict[str, Tensor],
    warmup_steps: int = 30,
) -> Tensor:
    """
    Compute per-timestep KL divergence between posterior and prior.

    Uses the closed-form Gaussian KLD formula:
        KLD = 0.5 * sum_d [log(var_prior/var_post) + var_post/var_prior
                          + (mu_post - mu_prior)^2/var_prior - 1]

    The KLD measures transfer entropy: how much information the source
    signal (UP) provides beyond what the target (FHR) already contains.

    Args:
        outputs: Model forward outputs dict containing:
            - 'mu_prior': Prior mean from target encoder, shape (B, T, D)
            - 'logvar_prior': Prior log-variance, shape (B, T, D)
            - 'mu_post': Posterior mean, shape (B, T, D)
            - 'logvar_post': Posterior log-variance, shape (B, T, D)
        warmup_steps: Number of initial timesteps to mask with NaN (default 30).
            These steps are masked because the model needs context to warm up.

    Returns:
        KLD tensor of shape (B, T, D) with NaN for warmup timesteps.
        Returns None if required keys are missing from outputs.

    Example:
        >>> outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        >>> kld = compute_kld(outputs, warmup_steps=30)
        >>> kld_mean = torch.nanmean(kld)  # Mean ignoring warmup NaNs
    """
    # Extract required tensors
    mu_prior = outputs.get("mu_prior")
    logvar_prior = outputs.get("logvar_prior")
    mu_post = outputs.get("mu_post")
    logvar_post = outputs.get("logvar_post")

    # Check all required tensors are present
    if any(t is None for t in (mu_prior, logvar_prior, mu_post, logvar_post)):
        return None

    # Closed-form Gaussian KLD: KL(q || p)
    # = 0.5 * [log(var_p/var_q) + var_q/var_p + (mu_q - mu_p)^2/var_p - 1]
    kld = (
        logvar_prior  # log(var_prior)
        - logvar_post  # - log(var_post) = log(var_prior/var_post)
        + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
        - 1.0
    )
    kld = 0.5 * kld  # Shape: (B, T, D)

    # Apply warmup mask: set first warmup_steps to NaN
    if warmup_steps > 0 and kld.size(1) > warmup_steps:
        kld = kld.clone()  # Don't modify in-place
        kld[:, :warmup_steps, :] = float("nan")

    return kld


def compute_kld_per_sample(
    outputs: Dict[str, Tensor],
    warmup_steps: int = 30,
) -> Tensor:
    """
    Compute scalar KLD for each sample by averaging over time and latent dims.

    Useful for histogram plots and per-sample analysis.

    Args:
        outputs: Model forward outputs dict (see compute_kld for required keys).
        warmup_steps: Number of initial timesteps to mask (default 30).

    Returns:
        Tensor of shape (B,) with mean KLD per sample (ignoring warmup NaNs).
        Returns zeros if KLD computation fails.

    Example:
        >>> kld_samples = compute_kld_per_sample(outputs)
        >>> print(f"Mean KLD: {kld_samples.mean():.4f}")
    """
    kld = compute_kld(outputs, warmup_steps)

    if kld is None:
        # Fallback: return zeros with correct batch size
        # Try to get batch size from any available tensor
        for key in ("mu_pr", "z", "mu_prior", "mu_post"):
            if key in outputs and outputs[key] is not None:
                batch_size = outputs[key].size(0)
                device = outputs[key].device
                return torch.zeros(batch_size, device=device)
        # Last resort
        return torch.zeros(1)

    # Average over time (dim=1) and latent dimensions (dim=2)
    # Use nanmean to ignore warmup NaNs
    return torch.nanmean(kld, dim=(1, 2))


def aggregate_predictions(
    model: torch.nn.Module,
    segments: Tensor,
    raw_len: Optional[int] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Aggregate overlapping per-timestep prediction windows into a single signal.

    The model predicts a window of H samples at each of T timesteps, creating
    overlapping predictions. This function averages overlapping predictions
    to produce a single raw-length signal.

    Args:
        model: The VAE model with average_raw_prediction method.
        segments: Per-timestep prediction windows, shape (B, T, H).
        raw_len: Target raw signal length. If None, uses model's default.

    Returns:
        Tuple of:
            - avg_pred: Averaged predictions, shape (B, raw_len) or None
            - valid_mask: Boolean mask of covered positions, shape (B, raw_len)

    Example:
        >>> outputs = model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        >>> avg_pred, mask = aggregate_predictions(model, outputs['mu_pr'])
    """
    if segments is None or segments.dim() != 3:
        return None, None

    # Check if model has averaging method
    if not hasattr(model, "average_raw_prediction"):
        return None, None

    # Use model's averaging function
    avg_pred = model.average_raw_prediction(segments, raw_len=raw_len)

    # Create validity mask (positions that had predictions, not NaN)
    valid_mask = torch.isfinite(avg_pred)

    # Replace NaNs with zeros for downstream use
    avg_pred = torch.nan_to_num(avg_pred, nan=0.0)

    return avg_pred, valid_mask


def compute_kld_per_timestep(
    outputs: Dict[str, Tensor],
    warmup_steps: int = 30,
) -> Tensor:
    """
    Compute KLD averaged over latent dimensions, preserving time dimension.

    Useful for analyzing how transfer entropy varies over time within a sample.

    Args:
        outputs: Model forward outputs dict (see compute_kld for required keys).
        warmup_steps: Number of initial timesteps to mask (default 30).

    Returns:
        Tensor of shape (B, T) with mean KLD per timestep.

    Example:
        >>> kld_t = compute_kld_per_timestep(outputs)
        >>> plt.plot(kld_t[0].cpu().numpy())  # Plot KLD over time for sample 0
    """
    kld = compute_kld(outputs, warmup_steps)

    if kld is None:
        return None

    # Average over latent dimensions only (keep time)
    return torch.nanmean(kld, dim=-1)  # Shape: (B, T)
