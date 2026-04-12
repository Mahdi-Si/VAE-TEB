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

import numpy as np
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
        >>> outputs = model(y_st=y_st, y_ph=y_ph, u_stream=u_stream)
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


# -----------------------------------------------------------------------------
# Lag-Attentive V1 Feature-Forecast Metrics
# -----------------------------------------------------------------------------


def _feature_valid_slice(
    warmup: int,
    horizon: int,
    T: int,
) -> Tuple[int, int]:
    """Return the ``(start, end)`` anchor range for feature-forecast metrics.

    The valid range is ``[warmup, T - horizon)``. Both endpoints are clamped
    into ``[0, max(0, T - horizon)]`` so tiny sequences fall back to an
    empty range rather than raising.

    Args:
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.
        T: Sequence length ``T``.

    Returns:
        ``(start, end)`` anchor indices with ``0 <= start <= end``.
    """
    T_valid = max(T - int(horizon), 0)
    start = max(0, min(int(warmup), T_valid))
    return start, T_valid


def compute_forecast_metrics(
    mu_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    """Compute per-sample feature-forecast quality metrics for v1.

    Evaluates the full forecast ``mu_full`` against the unfolded future
    feature target ``y_plus`` over the valid anchor range
    ``[warmup, T - horizon)``. Channels are split at index 43 into a
    scattering block (``feat_mse_st``) and a phase-harmonic block
    (``feat_mse_ph``) so diagnostics can distinguish which kind of
    information the model is losing.

    Args:
        mu_full: Model prediction ``(B, T, H_d, C_y)``.
        y_plus: Ground-truth future feature trajectory
            ``(B, T - H_d, H_d, C_y)`` from
            :meth:`TestRunner.build_future_target`.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d`` (must equal ``mu_full.size(2)``).

    Returns:
        Dict with tensors (all shape ``(B,)`` unless noted):

        - ``feat_mse_total``: mean squared error over valid anchors,
          horizon and channels.
        - ``feat_mse_per_horizon`` ``(B, H_d)``: MSE broken out per
          horizon step.
        - ``feat_r2_total``: ``1 - SS_res / SS_tot`` using the channel-wise
          variance of ``y_plus`` as denominator.
        - ``feat_mse_st``: MSE restricted to channels ``[0, 43)``.
        - ``feat_mse_ph``: MSE restricted to channels ``[43, C_y)``.
    """
    B, T, H_d, C = mu_full.shape
    if int(H_d) != int(horizon):
        raise ValueError(
            f"horizon mismatch: mu_full.size(2)={H_d}, horizon={horizon}"
        )

    start, T_valid = _feature_valid_slice(warmup, horizon, T)

    # Align anchors. y_plus has shape (B, T - H_d, H_d, C) — already trimmed
    # to the maximum anchor count, so slice both to [start:T_valid].
    mu_valid = mu_full[:, start:T_valid, :, :]
    y_valid = y_plus[:, start:T_valid, :, :]

    if mu_valid.numel() == 0 or y_valid.numel() == 0:
        zeros = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)
        return {
            "feat_mse_total": zeros,
            "feat_mse_per_horizon": torch.zeros(
                B, int(H_d), device=mu_full.device, dtype=mu_full.dtype
            ),
            "feat_r2_total": zeros,
            "feat_mse_st": zeros,
            "feat_mse_ph": zeros,
        }

    diff = mu_valid - y_valid                    # (B, T_v, H_d, C)
    sq = diff.pow(2)

    # Per-sample scalar MSE over (T_v, H_d, C).
    feat_mse_total = sq.mean(dim=(1, 2, 3))

    # Per-horizon MSE (B, H_d), averaged over (T_v, C).
    feat_mse_per_horizon = sq.mean(dim=(1, 3))

    # Channel-block splits (scattering vs phase). 43 is hardcoded because
    # the v1 model concatenates fhr_st (43) followed by fhr_ph (44).
    c_st = min(43, int(C))
    if c_st > 0:
        feat_mse_st = sq[..., :c_st].mean(dim=(1, 2, 3))
    else:
        feat_mse_st = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)
    if c_st < int(C):
        feat_mse_ph = sq[..., c_st:].mean(dim=(1, 2, 3))
    else:
        feat_mse_ph = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)

    # Per-sample R^2 against per-channel mean of y_valid.
    y_flat = y_valid.reshape(B, -1)
    mu_flat = mu_valid.reshape(B, -1)
    y_mean = y_flat.mean(dim=1, keepdim=True)
    ss_res = (mu_flat - y_flat).pow(2).sum(dim=1)
    ss_tot = (y_flat - y_mean).pow(2).sum(dim=1).clamp_min(1e-12)
    feat_r2_total = (1.0 - ss_res / ss_tot).clamp(min=-10.0, max=1.0)

    return {
        "feat_mse_total": feat_mse_total,
        "feat_mse_per_horizon": feat_mse_per_horizon,
        "feat_r2_total": feat_r2_total,
        "feat_mse_st": feat_mse_st,
        "feat_mse_ph": feat_mse_ph,
    }


def compute_uplift_metrics(
    mu_full: Tensor,
    mu_base: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    """Compare the full forecast to the FHR-only baseline forecast.

    The lag-attn v1 model decomposes its prediction as
    ``mu_full = mu_base + delta_mu_src`` where ``mu_base`` is the
    FHR-only baseline and ``delta_mu_src`` is the residual correction
    driven by the latent. A positive uplift means the source/latent branch
    is helping; a near-zero uplift on a trained checkpoint is a red flag
    that the source branch has collapsed.

    Args:
        mu_full: Full prediction ``(B, T, H_d, C_y)``.
        mu_base: Baseline prediction ``(B, T, H_d, C_y)``.
        y_plus: Ground truth ``(B, T - H_d, H_d, C_y)``.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d``.

    Returns:
        Dict with ``l_full`` (per-sample full-MSE), ``l_base``
        (per-sample base-MSE), ``uplift_abs = l_base - l_full``,
        ``uplift_rel = uplift_abs / l_base.clamp_min(1e-12)``, all shape
        ``(B,)``.
    """
    B, T = mu_full.shape[0], mu_full.shape[1]
    start, T_valid = _feature_valid_slice(warmup, horizon, T)

    mu_full_v = mu_full[:, start:T_valid, :, :]
    mu_base_v = mu_base[:, start:T_valid, :, :]
    y_v = y_plus[:, start:T_valid, :, :]

    if mu_full_v.numel() == 0:
        zeros = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)
        return {
            "l_full": zeros,
            "l_base": zeros,
            "uplift_abs": zeros,
            "uplift_rel": zeros,
        }

    l_full = (mu_full_v - y_v).pow(2).mean(dim=(1, 2, 3))
    l_base = (mu_base_v - y_v).pow(2).mean(dim=(1, 2, 3))
    uplift_abs = l_base - l_full
    uplift_rel = uplift_abs / l_base.clamp_min(1e-12)
    return {
        "l_full": l_full,
        "l_base": l_base,
        "uplift_abs": uplift_abs,
        "uplift_rel": uplift_rel,
    }


def compute_residual_usage(
    delta_mu_src: Tensor,
    mu_full: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    """Quantify how much of the final forecast comes from the source branch.

    Returns per-sample L2 norms of ``delta_mu_src`` and ``mu_full``, the
    ratio of the two (``residual_ratio``), plus a per-anchor trace of
    ``delta_mu_src`` norm for diagnostic plots.

    Args:
        delta_mu_src: Residual correction ``(B, T, H_d, C_y)``.
        mu_full: Full prediction ``(B, T, H_d, C_y)``.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d``.

    Returns:
        Dict with keys:

        - ``delta_norm`` ``(B,)`` — ``sqrt(mean(delta^2))`` over valid
          anchors, horizon and channels.
        - ``full_norm`` ``(B,)`` — same for ``mu_full``.
        - ``residual_ratio`` ``(B,)`` — ``delta_norm / full_norm`` with
          epsilon safety.
        - ``delta_norm_t`` ``(B, T_valid)`` — per-anchor RMS of
          ``delta_mu_src`` (for plotting the residual trace over time).
    """
    B, T = mu_full.shape[0], mu_full.shape[1]
    start, T_valid = _feature_valid_slice(warmup, horizon, T)

    delta_v = delta_mu_src[:, start:T_valid, :, :]
    full_v = mu_full[:, start:T_valid, :, :]

    if delta_v.numel() == 0:
        zeros_b = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)
        zeros_bt = torch.zeros(B, 0, device=mu_full.device, dtype=mu_full.dtype)
        return {
            "delta_norm": zeros_b,
            "full_norm": zeros_b,
            "residual_ratio": zeros_b,
            "delta_norm_t": zeros_bt,
        }

    delta_norm = delta_v.pow(2).mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt()
    full_norm = full_v.pow(2).mean(dim=(1, 2, 3)).clamp_min(0.0).sqrt()
    residual_ratio = delta_norm / full_norm.clamp_min(1e-12)

    # Per-anchor trace: RMS over (H_d, C).
    delta_norm_t = delta_v.pow(2).mean(dim=(2, 3)).clamp_min(0.0).sqrt()

    return {
        "delta_norm": delta_norm,
        "full_norm": full_norm,
        "residual_ratio": residual_ratio,
        "delta_norm_t": delta_norm_t,
    }


def compute_attention_diagnostics(
    attn_weights: Tensor,
    warmup: int,
) -> Dict[str, Tensor]:
    """Summarise lag-attention weights produced by v1.

    Computes the head-averaged attention distribution, per-anchor argmax
    lag, per-head entropy, inter-head diversity, and dataset-level
    "attention mass by lag". Warmup anchors have all time-indexed outputs
    NaN-filled so aggregators can ``nanmean`` cleanly.

    Args:
        attn_weights: Raw attention probabilities ``(B, T, M, L)`` from
            ``outputs["attn_weights"]``.
        warmup: Number of initial anchors to mask.

    Returns:
        Dict with:

        - ``alpha_bar`` ``(B, T, L)`` — head-averaged attention (NaN in
          warmup).
        - ``argmax_lag`` ``(B, T)`` — head-averaged argmax lag per anchor
          (``-1`` in warmup).
        - ``entropy`` ``(B, T, M)`` — Shannon entropy per head (NaN in
          warmup).
        - ``head_diversity`` ``(B, T)`` — ``1 - mean_pairwise_cosine_sim``
          between heads at each anchor (NaN in warmup).
        - ``alpha_mass_by_lag`` ``(B, L)`` — time-averaged (valid anchors)
          head-averaged attention distribution.
    """
    if attn_weights.dim() != 4:
        raise ValueError(
            f"attn_weights must be (B, T, M, L), got {tuple(attn_weights.shape)}"
        )
    B, T, M, L = attn_weights.shape
    device = attn_weights.device
    dtype = attn_weights.dtype
    warmup = max(0, min(int(warmup), T))

    # Head-averaged attention (B, T, L).
    alpha_bar = attn_weights.mean(dim=2)

    # Argmax per head, then take head-wise mode via head-averaged argmax on
    # alpha_bar (cheaper and matches the "which lag dominates" question).
    argmax_lag = alpha_bar.argmax(dim=-1).to(torch.long)  # (B, T)

    # Per-head entropy with a numerical-stability epsilon.
    eps = 1e-12
    entropy = -(attn_weights.clamp_min(eps) * attn_weights.clamp_min(eps).log()).sum(dim=-1)

    # Head diversity: mean pairwise (1 - cosine similarity) between heads
    # at each anchor. For M heads, there are C(M, 2) pairs.
    if M >= 2:
        a = attn_weights                                   # (B, T, M, L)
        norm = a.pow(2).sum(dim=-1).clamp_min(eps).sqrt()  # (B, T, M)
        a_norm = a / norm.unsqueeze(-1)
        # sims[b, t, i, j] = <a_norm[b, t, i], a_norm[b, t, j]>
        sims = torch.einsum("btil,btjl->btij", a_norm, a_norm)
        # Exclude self-similarity and upper-triangle duplicates.
        mask = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
        pair_sims = sims[:, :, mask]                       # (B, T, C(M,2))
        head_diversity = 1.0 - pair_sims.mean(dim=-1)
    else:
        head_diversity = torch.zeros(B, T, device=device, dtype=dtype)

    # Apply warmup masking (convert to float and NaN-fill the warmup region).
    alpha_bar_f = alpha_bar.to(torch.float32).clone()
    entropy_f = entropy.to(torch.float32).clone()
    head_div_f = head_diversity.to(torch.float32).clone()
    argmax_f = argmax_lag.clone()

    if warmup > 0:
        alpha_bar_f[:, :warmup, :] = float("nan")
        entropy_f[:, :warmup, :] = float("nan")
        head_div_f[:, :warmup] = float("nan")
        argmax_f[:, :warmup] = -1

    # Time-averaged alpha_mass_by_lag over valid anchors.
    if warmup < T:
        alpha_mass_by_lag = alpha_bar[:, warmup:, :].mean(dim=1)
    else:
        alpha_mass_by_lag = torch.zeros(B, L, device=device, dtype=dtype)

    return {
        "alpha_bar": alpha_bar_f,
        "argmax_lag": argmax_f,
        "entropy": entropy_f,
        "head_diversity": head_div_f,
        "alpha_mass_by_lag": alpha_mass_by_lag,
    }


def aggregate_te_lag_map(
    te_lag_map: Tensor,
    warmup: int,
) -> Dict[str, Tensor]:
    """Aggregate the per-timestep TE lag attribution map over time.

    ``te_lag_map`` has shape ``(B, T, L)`` and carries the lag-resolved
    transfer-entropy surrogate (``kld_per_t * mean_heads(alpha)``) at each
    anchor. The aggregation averages over valid anchors to yield one
    ``(L,)`` signature per sample plus the argmax lag.

    Args:
        te_lag_map: ``(B, T, L)`` from ``outputs["te_lag_map"]``.
        warmup: Number of initial anchors to exclude.

    Returns:
        Dict with ``te_lag_mean`` ``(B, L)`` and ``te_lag_argmax`` ``(B,)``.
    """
    if te_lag_map.dim() != 3:
        raise ValueError(
            f"te_lag_map must be (B, T, L), got {tuple(te_lag_map.shape)}"
        )
    B, T, L = te_lag_map.shape
    warmup = max(0, min(int(warmup), T))

    if warmup < T:
        te_mean = te_lag_map[:, warmup:, :].mean(dim=1)
    else:
        te_mean = torch.zeros(B, L, device=te_lag_map.device, dtype=te_lag_map.dtype)

    te_argmax = te_mean.argmax(dim=-1).to(torch.long)
    return {"te_lag_mean": te_mean, "te_lag_argmax": te_argmax}


# -----------------------------------------------------------------------------
# Latent Space Preprocessing and Dimensionality Reduction
# -----------------------------------------------------------------------------

def preprocess_latent(
    latent: Tensor,
    window_length: int = 9,
    polyorder: int = 2,
    denoise: bool = True,
) -> Tensor:
    """
    Preprocess latent trajectories with robust z-score normalization and denoising.

    Uses median and MAD (Median Absolute Deviation) for robust normalization,
    and optional Savitzky-Golay filtering for denoising.

    Args:
        latent: Latent tensor of shape (B, T, D) where B is batch size,
            T is time steps, and D is latent dimension.
        window_length: Savitzky-Golay filter window length (must be odd, default 9).
        polyorder: Savitzky-Golay polynomial order (default 2).
        denoise: Whether to apply Savitzky-Golay filter (default True).

    Returns:
        Preprocessed latent tensor of same shape (B, T, D).

    Example:
        >>> latent = outputs['mu_post']  # Shape: (32, 300, 16)
        >>> preprocessed = preprocess_latent(latent)
    """
    import numpy as np

    if not denoise:
        # Simple robust normalization without denoising
        median = torch.median(latent, dim=1, keepdim=True).values
        deviations = torch.abs(latent - median)
        mad = torch.median(deviations, dim=1, keepdim=True).values
        # Avoid division by zero
        mad = torch.where(mad == 0, torch.ones_like(mad), mad)
        return (latent - median) / mad

    # Denoise with Savitzky-Golay filter
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        # Fallback to no denoising if scipy not available
        return preprocess_latent(latent, denoise=False)

    device = latent.device
    latent_np = latent.cpu().numpy()

    # Robust normalization (median and MAD)
    median = np.median(latent_np, axis=1, keepdims=True)
    mad = np.median(np.abs(latent_np - median), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1.0, mad)
    latent_normalized = (latent_np - median) / mad

    batch_size, time_steps, latent_dim = latent_normalized.shape

    # Apply Savitzky-Golay filter to each trajectory
    for b in range(batch_size):
        for d in range(latent_dim):
            if time_steps >= window_length:
                latent_normalized[b, :, d] = savgol_filter(
                    latent_normalized[b, :, d],
                    window_length=window_length,
                    polyorder=polyorder,
                )

    return torch.from_numpy(latent_normalized).to(device)


def reduce_latent_dimensionality(
    latent_data: Tensor,
    method: str = "pca",
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    return_reducer: bool = False,
):
    """
    Reduce latent trajectory dimensionality for visualization and analysis.

    Supports multiple dimensionality reduction methods including linear (PCA)
    and nonlinear manifold methods (UMAP, t-SNE, Isomap, diffusion maps).

    Args:
        latent_data: Latent tensor of shape (B, T, D) or numpy array.
        method: Reduction method - one of:
            - 'pca': Principal Component Analysis (fast, linear)
            - 'umap': Uniform Manifold Approximation (preserves global + local)
            - 'tsne': t-Distributed Stochastic Neighbor Embedding (local structure)
            - 'isomap': Isometric Mapping (geodesic distances)
            - 'diffusion': Diffusion Maps (spectral embedding)
        n_components: Target dimensions (2 or 3, default 3).
        n_neighbors: Neighbors for manifold methods (default 15).
        min_dist: UMAP minimum distance parameter (default 0.1).
        random_state: Random seed for reproducibility (default 42).
        return_reducer: Whether to return fitted reducer object (default False).

    Returns:
        If return_reducer is False:
            reduced_data: np.ndarray of shape (B, T, n_components)
        If return_reducer is True:
            Tuple of (reduced_data, reducer) where reducer is the fitted object.

    Raises:
        ValueError: If method is not recognized.
        ImportError: If required libraries are not installed.

    Example:
        >>> latent = outputs['mu_post']  # Shape: (32, 300, 16)
        >>> reduced = reduce_latent_dimensionality(latent, method='umap', n_components=3)
        >>> print(reduced.shape)  # (32, 300, 3)
    """
    import numpy as np

    # Convert to numpy if needed
    if torch.is_tensor(latent_data):
        latent_np = latent_data.cpu().numpy()
    else:
        latent_np = np.asarray(latent_data)

    batch_size, time_steps, latent_dim = latent_np.shape

    # Flatten for dimensionality reduction: (B*T, D)
    latent_flat = latent_np.reshape(-1, latent_dim)

    method = method.lower()

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == "umap":
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "UMAP is required for 'umap' method. Install with: pip install umap-learn"
            )

        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
            n_jobs=-1,
        )
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == "tsne":
        from sklearn.manifold import TSNE

        # Perplexity must be less than n_samples
        perplexity = min(30, latent_flat.shape[0] // 4)
        perplexity = max(5, perplexity)  # At least 5

        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_jobs=-1,
        )
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == "isomap":
        from sklearn.manifold import Isomap

        # n_neighbors must be less than n_samples
        effective_neighbors = min(n_neighbors, latent_flat.shape[0] - 1)

        reducer = Isomap(
            n_neighbors=effective_neighbors,
            n_components=n_components,
            n_jobs=-1,
        )
        reduced_flat = reducer.fit_transform(latent_flat)

    elif method == "diffusion":
        # Diffusion maps using spectral decomposition of the diffusion operator
        from sklearn.metrics.pairwise import rbf_kernel

        gamma = 1.0 / latent_dim
        K = rbf_kernel(latent_flat, gamma=gamma)

        # Normalize the kernel to create a Markov transition matrix
        D = np.diag(K.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L = D_inv_sqrt @ K @ D_inv_sqrt

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Use top eigenvectors (skip first constant eigenvector)
        reduced_flat = eigenvectors[:, 1 : n_components + 1] * eigenvalues[1 : n_components + 1]
        reducer = None

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from: 'pca', 'umap', 'tsne', 'isomap', 'diffusion'"
        )

    # Reshape back to (B, T, n_components)
    reduced_data = reduced_flat.reshape(batch_size, time_steps, n_components)

    if return_reducer:
        return reduced_data, reducer
    return reduced_data


# -----------------------------------------------------------------------------
# Trajectory Shape Metrics
# -----------------------------------------------------------------------------

def compute_trajectory_path_length(trajectory: np.ndarray) -> float:
    """
    Sum of consecutive L2 distances along a trajectory.

    Args:
        trajectory: Array of shape (T, D).

    Returns:
        Total path length (scalar).
    """
    if trajectory.shape[0] < 2:
        return 0.0
    diffs = np.diff(trajectory, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_trajectory_displacement(trajectory: np.ndarray) -> float:
    """
    L2 norm of (last - first) point in trajectory.

    Args:
        trajectory: Array of shape (T, D).

    Returns:
        Displacement (scalar).
    """
    if trajectory.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(trajectory[-1] - trajectory[0]))


def compute_trajectory_tortuosity(trajectory: np.ndarray) -> float:
    """
    Ratio of path length to displacement. Returns inf if displacement == 0.

    Args:
        trajectory: Array of shape (T, D).

    Returns:
        Tortuosity (scalar >= 1, or inf).
    """
    disp = compute_trajectory_displacement(trajectory)
    if disp == 0:
        return float("inf")
    return compute_trajectory_path_length(trajectory) / disp


def compute_trajectory_speed(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Per-step speed along a trajectory.

    Args:
        trajectory: Array of shape (T, D).
        dt: Time step between consecutive points.

    Returns:
        Speed array of shape (T-1,).
    """
    if trajectory.shape[0] < 2:
        return np.array([])
    diffs = np.diff(trajectory, axis=0)
    return np.linalg.norm(diffs, axis=1) / dt


def compute_trajectory_curvature(trajectory: np.ndarray) -> np.ndarray:
    """
    Angle (radians) between consecutive displacement vectors.

    Args:
        trajectory: Array of shape (T, D).

    Returns:
        Curvature array of shape (T-2,).
    """
    if trajectory.shape[0] < 3:
        return np.array([])
    diffs = np.diff(trajectory, axis=0)  # (T-1, D)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    unit_vecs = diffs / norms
    # Dot product of consecutive unit vectors
    dots = np.sum(unit_vecs[:-1] * unit_vecs[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def compute_trajectory_spread(trajectory: np.ndarray) -> float:
    """
    Log-determinant of the covariance matrix of trajectory points.

    Measures how spread out the trajectory is in latent space.

    Args:
        trajectory: Array of shape (T, D).

    Returns:
        Log-determinant of covariance (scalar). Returns -inf if singular.
    """
    if trajectory.shape[0] < trajectory.shape[1] + 1:
        return float("-inf")
    cov = np.cov(trajectory, rowvar=False)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def compute_trajectory_features(
    trajectory: np.ndarray,
    kld_series: Optional[np.ndarray] = None,
    dt: float = 4.0,
) -> Dict[str, float]:
    """
    Compute all shape metrics for a single trajectory.

    Args:
        trajectory: Array of shape (T, D) — latent trajectory.
        kld_series: Optional array of shape (T,) — per-timestep KLD values.
        dt: Time step in seconds between consecutive latent points.

    Returns:
        Dict with keys: path_length, displacement, tortuosity, mean_speed,
        std_speed, max_speed, mean_accel, mean_curvature, max_curvature,
        spread, kld_mean, kld_std, kld_slope, kld_peak, kld_range.
    """
    features: Dict[str, float] = {}

    # Shape metrics
    features["path_length"] = compute_trajectory_path_length(trajectory)
    features["displacement"] = compute_trajectory_displacement(trajectory)
    features["tortuosity"] = compute_trajectory_tortuosity(trajectory)

    speed = compute_trajectory_speed(trajectory, dt=dt)
    if speed.size > 0:
        features["mean_speed"] = float(np.mean(speed))
        features["std_speed"] = float(np.std(speed))
        features["max_speed"] = float(np.max(speed))
        # Acceleration = diff of speed
        accel = np.diff(speed) / dt
        features["mean_accel"] = float(np.mean(np.abs(accel))) if accel.size > 0 else 0.0
    else:
        features["mean_speed"] = 0.0
        features["std_speed"] = 0.0
        features["max_speed"] = 0.0
        features["mean_accel"] = 0.0

    curvature = compute_trajectory_curvature(trajectory)
    if curvature.size > 0:
        features["mean_curvature"] = float(np.mean(curvature))
        features["max_curvature"] = float(np.max(curvature))
    else:
        features["mean_curvature"] = 0.0
        features["max_curvature"] = 0.0

    features["spread"] = compute_trajectory_spread(trajectory)

    # KLD features
    if kld_series is not None:
        valid_kld = kld_series[np.isfinite(kld_series)]
        if valid_kld.size > 0:
            features["kld_mean"] = float(np.mean(valid_kld))
            features["kld_std"] = float(np.std(valid_kld))
            features["kld_peak"] = float(np.max(valid_kld))
            features["kld_range"] = float(np.max(valid_kld) - np.min(valid_kld))
            # Linear trend (slope)
            if valid_kld.size > 1:
                x = np.arange(valid_kld.size, dtype=float)
                slope, _ = np.polyfit(x, valid_kld, 1)
                features["kld_slope"] = float(slope)
            else:
                features["kld_slope"] = 0.0
        else:
            for k in ("kld_mean", "kld_std", "kld_slope", "kld_peak", "kld_range"):
                features[k] = float("nan")
    else:
        for k in ("kld_mean", "kld_std", "kld_slope", "kld_peak", "kld_range"):
            features[k] = float("nan")

    return features
