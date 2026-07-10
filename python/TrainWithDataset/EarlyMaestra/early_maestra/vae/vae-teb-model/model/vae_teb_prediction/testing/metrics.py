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

from typing import Any, Dict, Optional, Sequence, Tuple

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
        for key in ("z", "mu_prior", "mu_post"):
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


def compute_kld_aggregate_tensors(
    outputs: Dict[str, Tensor],
    warmup_steps: int = 30,
) -> Optional[Dict[str, Tensor]]:
    """Compute explicit per-time KLD aggregates over latent dimensions.

    The historical testing pipeline used ``kld_mean`` as the per-dimension
    mean KL. For TE-style interpretation, the latent dimensions are additive,
    so this helper also exposes the dimension-sum KL and the Euclidean norm of
    the per-dim KL vector.

    Args:
        outputs: Model forward dict.
        warmup_steps: Initial timesteps to NaN-mask.

    Returns:
        Dict of ``(B, T)`` tensors: ``kld_mean_t``, ``kld_sum_t``,
        ``kld_l2_t``. Returns None when KL cannot be computed.
    """
    kld = compute_kld(outputs, warmup_steps)
    if kld is None:
        return None
    all_nan = torch.isnan(kld).all(dim=-1)
    kld_sum_t = torch.nansum(kld, dim=-1)
    kld_l2_t = torch.linalg.vector_norm(torch.nan_to_num(kld, nan=0.0), dim=-1)
    nan_fill = torch.full_like(kld_sum_t, float("nan"))
    return {
        "kld_mean_t": torch.nanmean(kld, dim=-1),
        "kld_sum_t": torch.where(all_nan, nan_fill, kld_sum_t),
        "kld_l2_t": torch.where(all_nan, nan_fill, kld_l2_t),
    }


def compute_kld_aggregates_per_sample(
    outputs: Dict[str, Tensor],
    warmup_steps: int = 30,
) -> Dict[str, Tensor]:
    """Compute per-sample KLD mean, sum, and L2 aggregates.

    Returns one scalar per sample for each aggregate by averaging the
    corresponding per-time series over valid post-warmup timesteps.
    """
    agg_t = compute_kld_aggregate_tensors(outputs, warmup_steps)
    if agg_t is None:
        for key in ("z", "mu_prior", "mu_post"):
            t = outputs.get(key)
            if t is not None:
                zeros = torch.zeros(t.size(0), device=t.device, dtype=t.dtype)
                return {"kld_mean": zeros, "kld_sum": zeros, "kld_l2": zeros}
        zeros = torch.zeros(1)
        return {"kld_mean": zeros, "kld_sum": zeros, "kld_l2": zeros}
    return {
        "kld_mean": torch.nanmean(agg_t["kld_mean_t"], dim=1),
        "kld_sum": torch.nanmean(agg_t["kld_sum_t"], dim=1),
        "kld_l2": torch.nanmean(agg_t["kld_l2_t"], dim=1),
    }


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


def compute_band_forecast_metrics(
    mu_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
    partition_idx: Optional[Dict[str, "np.ndarray"]] = None,
    *,
    return_per_anchor: bool = True,
    band_combined_idx: Optional[Dict[str, "np.ndarray"]] = None,
) -> Dict[str, Dict[str, Tensor]]:
    """Compute per-sample feature-forecast metrics stratified by a partition.

    Slices the squared-error tensor along the channel axis using one
    integer index list per partition label, then aggregates over the
    same valid anchor range as :func:`compute_forecast_metrics`. The
    returned dict maps each label to a sub-dict of tensors so the
    analysis layer can iterate without unpacking individual keys.

    Generalised from the original "band-only" version: the partition
    can be the clinical 4-band split, the refined 7-band split, the
    coefficient-kind partition, the per-octave partition, or any other
    ``label -> int_array`` mapping.

    Args:
        mu_full: Model prediction ``(B, T, H_d, C_y)``.
        y_plus: Ground-truth future feature trajectory ``(B, T - H_d,
            H_d, C_y)`` from :meth:`TestRunner.build_future_target`.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d`` (must equal ``mu_full.size(2)``).
        partition_idx: Mapping ``label -> 1-D int array of channel
            indices`` into ``[0, C_y)``. Labels with empty arrays are
            returned with all-zero tensors so callers can still address
            them.
        return_per_anchor: When ``True`` also return per-anchor MSE
            tensors of shape ``(B, T_valid)`` per label — used by the
            anchor-error grid plot. Set to ``False`` to save memory if
            only horizon and total aggregates are needed.
        band_combined_idx: Backwards-compatible alias for
            ``partition_idx``. Either may be supplied (but not both);
            this kwarg exists so existing callers keep working
            unchanged.

    Returns:
        Dict ``{label -> {"mse_total", "mse_per_horizon", "r2_total",
        "n_channels", optional "mse_per_anchor"}}``. All per-sample
        tensors share the input device and dtype.
    """
    if partition_idx is None and band_combined_idx is None:
        raise TypeError(
            "compute_band_forecast_metrics requires partition_idx "
            "(or its legacy alias band_combined_idx)."
        )
    if partition_idx is not None and band_combined_idx is not None:
        raise TypeError(
            "Pass either partition_idx or band_combined_idx, not both."
        )
    if partition_idx is None:
        partition_idx = band_combined_idx  # type: ignore[assignment]
    assert partition_idx is not None  # for the type checker

    # Local alias keeps the existing loop body readable without renames.
    band_combined_idx = partition_idx

    B, T, H_d, C = mu_full.shape
    if int(H_d) != int(horizon):
        raise ValueError(
            f"horizon mismatch: mu_full.size(2)={H_d}, horizon={horizon}"
        )
    start, T_valid = _feature_valid_slice(warmup, horizon, T)

    out: Dict[str, Dict[str, Tensor]] = {}

    if (T_valid - start) <= 0:
        zero_b = torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype)
        zero_bh = torch.zeros(B, int(H_d), device=mu_full.device, dtype=mu_full.dtype)
        zero_bt = torch.zeros(
            B, max(int(T_valid - start), 0),
            device=mu_full.device, dtype=mu_full.dtype,
        )
        for band, idx in band_combined_idx.items():
            entry: Dict[str, Tensor] = {
                "mse_total": zero_b.clone(),
                "mse_per_horizon": zero_bh.clone(),
                "r2_total": zero_b.clone(),
                "n_channels": torch.tensor(int(np.asarray(idx).size)),
            }
            if return_per_anchor:
                entry["mse_per_anchor"] = zero_bt.clone()
            out[band] = entry
        return out

    mu_valid = mu_full[:, start:T_valid, :, :]                  # (B, T_v, H_d, C)
    y_valid = y_plus[:, start:T_valid, :, :]                    # (B, T_v, H_d, C)
    diff = mu_valid - y_valid
    sq = diff.pow(2)                                            # (B, T_v, H_d, C)

    for band, idx_arr in band_combined_idx.items():
        idx_np = np.asarray(idx_arr, dtype=np.int64)
        n_ch = int(idx_np.size)
        if n_ch == 0:
            entry = {
                "mse_total": torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype),
                "mse_per_horizon": torch.zeros(
                    B, int(H_d), device=mu_full.device, dtype=mu_full.dtype,
                ),
                "r2_total": torch.zeros(B, device=mu_full.device, dtype=mu_full.dtype),
                "n_channels": torch.tensor(0),
            }
            if return_per_anchor:
                entry["mse_per_anchor"] = torch.zeros(
                    B, mu_valid.size(1),
                    device=mu_full.device, dtype=mu_full.dtype,
                )
            out[band] = entry
            continue

        idx_t = torch.as_tensor(idx_np, device=mu_full.device, dtype=torch.long)
        sq_band = sq.index_select(dim=-1, index=idx_t)          # (B, T_v, H_d, n_ch)

        mse_total = sq_band.mean(dim=(1, 2, 3))                 # (B,)
        mse_per_horizon = sq_band.mean(dim=(1, 3))              # (B, H_d)

        # Per-band R^2 against per-channel mean of y_valid restricted to
        # the same channel slice.
        y_band = y_valid.index_select(dim=-1, index=idx_t)
        mu_band = mu_valid.index_select(dim=-1, index=idx_t)
        y_flat = y_band.reshape(B, -1)
        mu_flat = mu_band.reshape(B, -1)
        y_mean = y_flat.mean(dim=1, keepdim=True)
        ss_res = (mu_flat - y_flat).pow(2).sum(dim=1)
        ss_tot = (y_flat - y_mean).pow(2).sum(dim=1).clamp_min(1e-12)
        r2_total = (1.0 - ss_res / ss_tot).clamp(min=-10.0, max=1.0)

        entry = {
            "mse_total": mse_total,
            "mse_per_horizon": mse_per_horizon,
            "r2_total": r2_total,
            "n_channels": torch.tensor(n_ch),
        }
        if return_per_anchor:
            entry["mse_per_anchor"] = sq_band.mean(dim=(2, 3))   # (B, T_v)
        out[band] = entry

    return out


def compute_per_channel_forecast_metrics(
    mu_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    """Compute per-sample × per-channel forecast MSE / R² over valid anchors.

    Mirrors :func:`compute_band_forecast_metrics` but keeps the channel
    axis intact instead of pooling. Used by the per-channel diagnostics
    that ask "which exact channel is hardest to forecast?" without
    pre-binning into a frequency band.

    Args:
        mu_full: Model prediction ``(B, T, H_d, C_y)``.
        y_plus: Ground-truth future feature trajectory ``(B, T - H_d,
            H_d, C_y)`` from :meth:`TestRunner.build_future_target`.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d`` (must equal ``mu_full.size(2)``).

    Returns:
        Dict with two keys:

        * ``"mse_per_channel"`` : ``(B, C_y)`` — mean squared error per
          (sample, channel), averaged over valid anchors and horizon
          steps.
        * ``"r2_per_channel"`` : ``(B, C_y)`` — coefficient of
          determination per (sample, channel), clamped to
          ``[-10, 1]`` to suppress runaway values when the per-channel
          target variance is near zero.

        Per-anchor tensors are *not* returned: keeping memory at
        ``O(B × C)`` rather than ``O(B × T_v × C)`` lets the caller
        write the per-channel CSV without the long-format anchor blow-up.
    """
    B, T, H_d, _C = mu_full.shape
    if int(H_d) != int(horizon):
        raise ValueError(
            f"horizon mismatch: mu_full.size(2)={H_d}, horizon={horizon}"
        )
    start, T_valid = _feature_valid_slice(warmup, horizon, T)

    if (T_valid - start) <= 0:
        zero_bc = torch.zeros(
            B, int(_C), device=mu_full.device, dtype=mu_full.dtype,
        )
        return {
            "mse_per_channel": zero_bc.clone(),
            "r2_per_channel": zero_bc.clone(),
        }

    mu_valid = mu_full[:, start:T_valid, :, :]                  # (B, T_v, H_d, C)
    y_valid = y_plus[:, start:T_valid, :, :]                    # (B, T_v, H_d, C)

    # Per-channel MSE: average over (T_v, H_d) only.
    diff = mu_valid - y_valid                                   # (B, T_v, H_d, C)
    mse_per_channel = diff.pow(2).mean(dim=(1, 2))              # (B, C)

    # Per-channel R²: 1 - SS_res / SS_tot, with SS_tot computed against
    # the per-(sample, channel) mean of y. ss_tot is clamped to a tiny
    # floor so flat targets don't blow R² up.
    ss_res = diff.pow(2).sum(dim=(1, 2))                        # (B, C)
    y_mean = y_valid.mean(dim=(1, 2), keepdim=True)             # (B, 1, 1, C)
    ss_tot = (y_valid - y_mean).pow(2).sum(dim=(1, 2)).clamp_min(1e-12)
    r2_per_channel = (1.0 - ss_res / ss_tot).clamp(min=-10.0, max=1.0)

    return {
        "mse_per_channel": mse_per_channel,
        "r2_per_channel": r2_per_channel,
    }


def compute_per_channel_per_horizon_forecast_metrics(
    mu_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    r"""Compute per-(sample, channel, horizon-step) forecast MSE and $R^2$.

    Mirrors :func:`compute_per_channel_forecast_metrics` but keeps **both**
    the channel axis and the horizon axis intact instead of pooling them
    together. Used by the channel $\times$ horizon heatmap analysis that
    asks "which exact channel is hardest to forecast at each look-ahead
    step?".

    Memory cost is $O(B \cdot C \cdot H_d)$ which is harmless for the
    typical lag-attn v1 setting ($B \le 32$, $C = 87$, $H_d \le 16$).

    Args:
        mu_full: Model prediction ``(B, T, H_d, C_y)``.
        y_plus: Ground-truth future feature trajectory
            ``(B, T - H_d, H_d, C_y)`` from
            :meth:`TestRunner.build_future_target`.
        warmup: Warmup anchors to skip.
        horizon: Forecast horizon ``H_d`` (must equal ``mu_full.size(2)``).

    Returns:
        Dict with three keys:

        * ``"mse_per_channel_per_horizon"`` : ``(B, C_y, H_d)`` — squared
          error averaged over the valid-anchor axis only.
        * ``"r2_per_channel_per_horizon"`` : ``(B, C_y, H_d)`` —
          coefficient of determination per (sample, channel,
          horizon-step), clamped to $[-10, 1]$ to suppress runaway
          values when the per-(channel, horizon-step) target variance
          is near zero.
        * ``"n_valid_anchors"`` : 0-D tensor — number of valid anchors
          contributing to the average. Lets a streaming caller weight
          per-batch contributions correctly even if ``warmup`` ever
          changes between batches.
    """
    B, T, H_d, _C = mu_full.shape
    if int(H_d) != int(horizon):
        raise ValueError(
            f"horizon mismatch: mu_full.size(2)={H_d}, horizon={horizon}"
        )
    start, T_valid = _feature_valid_slice(warmup, horizon, T)
    n_valid = max(0, int(T_valid - start))

    if n_valid <= 0:
        zero = torch.zeros(
            B, int(_C), int(H_d), device=mu_full.device, dtype=mu_full.dtype,
        )
        return {
            "mse_per_channel_per_horizon": zero.clone(),
            "r2_per_channel_per_horizon": zero.clone(),
            "n_valid_anchors": torch.tensor(
                0, device=mu_full.device, dtype=torch.long,
            ),
        }

    mu_valid = mu_full[:, start:T_valid, :, :]                  # (B, T_v, H_d, C)
    y_valid = y_plus[:, start:T_valid, :, :]                    # (B, T_v, H_d, C)
    diff = mu_valid - y_valid                                   # (B, T_v, H_d, C)

    # Collapse the valid-anchor axis only -> (B, H_d, C); permute to
    # (B, C, H_d) so the channel axis comes first as in the per-channel
    # variant.
    mse_h_c = diff.pow(2).mean(dim=1)                           # (B, H_d, C)
    mse_per_channel_per_horizon = mse_h_c.permute(0, 2, 1).contiguous()  # (B, C, H_d)

    # R² with SS_tot computed against the per-(sample, channel, h) mean
    # of y across the valid-anchor axis. Floors ss_tot to keep flat
    # targets from blowing R² up.
    ss_res = diff.pow(2).sum(dim=1)                             # (B, H_d, C)
    y_mean = y_valid.mean(dim=1, keepdim=True)                  # (B, 1, H_d, C)
    ss_tot = (y_valid - y_mean).pow(2).sum(dim=1).clamp_min(1e-12)  # (B, H_d, C)
    r2_h_c = (1.0 - ss_res / ss_tot).clamp(min=-10.0, max=1.0)
    r2_per_channel_per_horizon = r2_h_c.permute(0, 2, 1).contiguous()   # (B, C, H_d)

    return {
        "mse_per_channel_per_horizon": mse_per_channel_per_horizon,
        "r2_per_channel_per_horizon": r2_per_channel_per_horizon,
        "n_valid_anchors": torch.tensor(
            n_valid, device=mu_full.device, dtype=torch.long,
        ),
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
# Additional TE surrogate helpers (lag-attn v1)
# -----------------------------------------------------------------------------


def compute_posterior_drift(
    mu_prior: Tensor,
    mu_post: Tensor,
    warmup: int,
) -> Tensor:
    """Compute the per-sample mean squared posterior drift.

    The squared L2 norm ``||mu_post - mu_prior||^2`` over the latent
    dimension at every anchor, averaged over the post-warmup time range.
    This is one of the additive terms inside the closed-form KL between
    two diagonal Gaussians and behaves as an alternative TE surrogate
    that is independent of the variance heads.

    Args:
        mu_prior: Prior mean tensor ``(B, T, d_z)``.
        mu_post: Posterior mean tensor ``(B, T, d_z)``.
        warmup: Number of initial anchors to skip.

    Returns:
        Per-sample drift ``(B,)``.
    """
    if mu_prior.shape != mu_post.shape:
        raise ValueError(
            f"mu_prior and mu_post must have the same shape, "
            f"got {tuple(mu_prior.shape)} vs {tuple(mu_post.shape)}"
        )
    if mu_prior.dim() != 3:
        raise ValueError(
            f"mu_prior must be (B, T, d_z), got {tuple(mu_prior.shape)}"
        )

    B, T, _ = mu_prior.shape
    warm = max(0, min(int(warmup), T))
    if warm >= T:
        return torch.zeros(B, device=mu_prior.device, dtype=mu_prior.dtype)
    drift_t = (mu_post[:, warm:, :] - mu_prior[:, warm:, :]).pow(2).sum(dim=-1)
    return drift_t.mean(dim=1)


def fit_pca_kld_per_dim(
    kld_per_dim_t: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
):
    """Fit a PCA on flattened per-time per-dim KL trajectories.

    Args:
        kld_per_dim_t: Array of shape ``(N_samples, T, d_z)`` carrying
            per-time per-dim KL contributions for every collected sample.
            NaN entries (e.g. warmup) are dropped before fitting.
        n_components: Number of PCA components to retain (default 3).
        random_state: Seed for reproducibility (default 42).

    Returns:
        Tuple ``(pca_model, projected, explained_variance_ratio)`` where:
            - ``pca_model`` is the fitted ``sklearn.decomposition.PCA``.
            - ``projected`` is ``(N_samples, T, n_components)`` with NaN
              kept where the input was NaN.
            - ``explained_variance_ratio`` is a ``(n_components,)`` array.
    """
    from sklearn.decomposition import PCA

    arr = np.asarray(kld_per_dim_t)
    if arr.ndim != 3:
        raise ValueError(
            f"kld_per_dim_t must be (N, T, d_z), got {arr.shape}"
        )
    N, T, dz = arr.shape
    flat = arr.reshape(-1, dz)
    finite_mask = np.all(np.isfinite(flat), axis=1)
    if not finite_mask.any():
        raise ValueError(
            "fit_pca_kld_per_dim received no finite rows in kld_per_dim_t"
        )

    n_components = max(1, min(int(n_components), dz, int(finite_mask.sum())))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(flat[finite_mask])

    projected = np.full((flat.shape[0], n_components), np.nan, dtype=np.float32)
    projected[finite_mask] = pca.transform(flat[finite_mask]).astype(np.float32)
    projected = projected.reshape(N, T, n_components)
    return pca, projected, np.asarray(pca.explained_variance_ratio_, dtype=np.float32)


def project_kld_per_dim(
    kld_per_dim_t: np.ndarray,
    pca_model,
) -> np.ndarray:
    """Project per-time per-dim KL trajectories through a fitted PCA.

    Args:
        kld_per_dim_t: Array of shape ``(N, T, d_z)``.
        pca_model: Fitted sklearn ``PCA`` instance.

    Returns:
        Projected array ``(N, T, n_components)`` with NaN preserved.
    """
    arr = np.asarray(kld_per_dim_t)
    if arr.ndim != 3:
        raise ValueError(
            f"kld_per_dim_t must be (N, T, d_z), got {arr.shape}"
        )
    N, T, dz = arr.shape
    n_components = int(getattr(pca_model, "n_components_", 0))
    if n_components <= 0:
        raise ValueError("pca_model has no fitted components")

    flat = arr.reshape(-1, dz)
    finite_mask = np.all(np.isfinite(flat), axis=1)
    out = np.full((flat.shape[0], n_components), np.nan, dtype=np.float32)
    if finite_mask.any():
        out[finite_mask] = pca_model.transform(flat[finite_mask]).astype(np.float32)
    return out.reshape(N, T, n_components)


def _rankdata_1d(values: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank helper for Spearman scores."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=float)
    sorted_x = x[order]
    start = 0
    while start < sorted_x.size:
        end = start + 1
        while end < sorted_x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _safe_abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return ``abs(Spearman rho)`` with finite/constant guards."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    xr = _rankdata_1d(x[mask])
    yr = _rankdata_1d(y[mask])
    sx = float(np.std(xr))
    sy = float(np.std(yr))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return abs(float(np.corrcoef(xr, yr)[0, 1]))


def _class_contrast_eta2(values: np.ndarray, labels: np.ndarray) -> float:
    """Effect-size contrast for one PC across outcome labels."""
    x = np.asarray(values, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    mask = np.isfinite(x) & pd_notna_array(y)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    classes = np.unique(y)
    if classes.size < 2:
        return 0.0
    overall = float(np.mean(x))
    total_ss = float(np.sum((x - overall) ** 2))
    if total_ss <= 1e-12:
        return 0.0
    between = 0.0
    for cls in classes:
        vals = x[y == cls]
        if vals.size == 0:
            continue
        between += float(vals.size) * (float(np.mean(vals)) - overall) ** 2
    return max(0.0, min(1.0, between / total_ss))


def pd_notna_array(values: np.ndarray) -> np.ndarray:
    """Pandas-like notna for object/numeric numpy arrays without importing pandas."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return np.isfinite(arr.astype(float))
    return np.asarray([v is not None and str(v).lower() != "nan" for v in arr])


def select_pca_components(
    projected: np.ndarray,
    explained_variance_ratio: Sequence[float],
    *,
    n_select: int = 3,
    labels: Optional[Sequence[Any]] = None,
    te_values: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Select PCA components by eigenvalue weighted contrast.

    Preference order:
    1. empirical-TE association when ``te_values`` are supplied;
    2. outcome-label contrast when labels contain at least two classes;
    3. eigenvalue rank when neither contrast is usable.

    Args:
        projected: PCA scores ``(N, T, K)``.
        explained_variance_ratio: PCA explained-variance ratio for K PCs.
        n_select: Number of PCs to select.
        labels: Optional sample labels for class-contrast ranking.
        te_values: Optional empirical TE values for association ranking.

    Returns:
        Dict with selected zero-based component indices, sign alignment, scores,
        contrast type, and per-sample PC means.
    """
    arr = np.asarray(projected, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"projected must be (N, T, K), got {arr.shape}")
    with np.errstate(invalid="ignore"):
        pc_means = np.nanmean(arr, axis=1)
    N, K = pc_means.shape
    ev = np.asarray(explained_variance_ratio, dtype=float)[:K]
    if ev.size < K:
        ev = np.pad(ev, (0, K - ev.size), constant_values=0.0)

    contrast = np.ones(K, dtype=float)
    contrast_type = "eigenvalue"

    if te_values is not None:
        te = np.asarray(te_values, dtype=float)
        if te.size == N and np.isfinite(te).sum() >= 3:
            contrast = np.asarray(
                [_safe_abs_spearman(pc_means[:, k], te) for k in range(K)],
                dtype=float,
            )
            if np.any(contrast > 0):
                contrast_type = "empirical_te"

    if contrast_type == "eigenvalue" and labels is not None:
        lab = np.asarray(labels)
        if lab.size == N and np.unique(lab[pd_notna_array(lab)]).size >= 2:
            contrast = np.asarray(
                [_class_contrast_eta2(pc_means[:, k], lab) for k in range(K)],
                dtype=float,
            )
            if np.any(contrast > 0):
                contrast_type = "label_contrast"

    score = np.nan_to_num(ev, nan=0.0) * np.nan_to_num(contrast, nan=0.0)
    if not np.any(score > 0):
        score = np.nan_to_num(ev, nan=0.0)
        contrast = np.ones(K, dtype=float)
        contrast_type = "eigenvalue"

    n_keep = max(1, min(int(n_select), K))
    selected = np.argsort(score)[::-1][:n_keep]

    signs = np.ones(n_keep, dtype=float)
    ref: Optional[np.ndarray] = None
    if contrast_type == "empirical_te" and te_values is not None:
        ref = np.asarray(te_values, dtype=float)
    elif contrast_type == "label_contrast" and labels is not None:
        try:
            ref = np.asarray(labels, dtype=float)
        except (TypeError, ValueError):
            ref = None
    for i, pc_idx in enumerate(selected):
        x = pc_means[:, pc_idx]
        if ref is not None and ref.size == N:
            mask = np.isfinite(x) & np.isfinite(ref)
            if mask.sum() >= 3 and np.std(x[mask]) > 0 and np.std(ref[mask]) > 0:
                corr = float(np.corrcoef(x[mask], ref[mask])[0, 1])
                signs[i] = 1.0 if corr >= 0 else -1.0
                continue
        # Deterministic fallback: largest-magnitude loading/score direction positive.
        finite_x = x[np.isfinite(x)]
        if finite_x.size and abs(float(np.nanmin(finite_x))) > abs(float(np.nanmax(finite_x))):
            signs[i] = -1.0

    return {
        "selected_indices": selected.astype(int),
        "selected_1based": (selected + 1).astype(int),
        "signs": signs.astype(float),
        "score": score.astype(float),
        "contrast": contrast.astype(float),
        "contrast_type": contrast_type,
        "pc_means": pc_means.astype(float),
    }


def aggregate_selected_pca_scores(
    pc_means: np.ndarray,
    selected_indices: Sequence[int],
    signs: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate selected PCA component scores per sample."""
    means = np.asarray(pc_means, dtype=float)
    idx = np.asarray(selected_indices, dtype=int)
    if idx.size == 0:
        raise ValueError("selected_indices must not be empty")
    selected = means[:, idx]
    sign_arr = (
        np.ones(idx.size, dtype=float)
        if signs is None
        else np.asarray(signs, dtype=float)[: idx.size]
    )
    signed = selected * sign_arr[None, :]
    return {
        "selected_scores": signed,
        "l2": np.sqrt(np.nansum(selected ** 2, axis=1)),
        "abs_sum": np.nansum(np.abs(selected), axis=1),
        "signed_sum": np.nansum(signed, axis=1),
    }


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


# =============================================================================
# Calibration metrics (G10)
# =============================================================================
#
# These kernels judge the *predictive distribution* the decoder emits under
# ``sigma_obs='learned'``, not merely its mean. They all slice the anchor axis with
# ``_feature_valid_slice``, so they cover exactly the anchors ``[warmup, T - H_d)`` that
# ``compute_forecast_metrics`` and the training loss use, and are therefore directly
# comparable to both.
#
# Caveat worth stating loudly: on a checkpoint trained with a *fixed* ``sigma_obs`` the
# decoder's ``logvar_full`` head receives no gradient, so these numbers describe an untrained
# head. The model does not record which likelihood it was trained under, so callers cannot
# detect that case automatically -- always report ``fit_constant_sigma`` alongside as the
# homoscedastic reference.

#: 0.5 * log(2*pi). The training NLL omits this constant; a proper scoring rule needs it.
_HALF_LOG_2PI = 0.9189385332046727

#: 1 / sqrt(pi), the closed-form Gaussian CRPS constant.
_INV_SQRT_PI = 0.5641895835477563


def _standard_normal_cdf(z: Tensor) -> Tensor:
    r"""Standard-normal CDF :math:`\Phi(z) = \tfrac12\left[1 + \mathrm{erf}(z/\sqrt2)\right]`."""
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476))


def _standard_normal_pdf(z: Tensor) -> Tensor:
    r"""Standard-normal PDF :math:`\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}`."""
    return torch.exp(-0.5 * z * z) * 0.3989422804014327


def _valid_triplet(
    mu_full: Tensor,
    logvar_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Slice ``(mu, sigma, y)`` down to the supervised anchor range.

    ``mu_full`` / ``logvar_full`` run over all ``T`` anchors while ``y_plus`` has only
    ``T - H_d`` of them; both are cut to ``[warmup, T - H_d)``.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        logvar_full: Forecast observation log-variance ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.

    Returns:
        ``(mu, sigma, y)``, each ``(B, T_v, H_d, C)``, with
        :math:`\sigma = \exp(\tfrac12 \log\sigma^2)`.
    """
    T = int(mu_full.shape[1])
    start, end = _feature_valid_slice(warmup, horizon, T)
    mu = mu_full[:, start:end]
    logvar = logvar_full[:, start:end]
    y = y_plus[:, start:end]
    return mu, torch.exp(0.5 * logvar), y


def _split_channel_blocks(per_elem: Tensor) -> Dict[str, Tensor]:
    """Reduce an elementwise score to total / per-horizon / scattering / phase summaries.

    The channel split index 43 is hardcoded, matching ``compute_forecast_metrics``: the model
    concatenates ``fhr_st`` (43 channels) followed by ``fhr_ph`` (44).
    """
    B, _, _, C = per_elem.shape
    c_st = min(43, int(C))
    zeros = torch.zeros(B, device=per_elem.device, dtype=per_elem.dtype)
    return {
        "total": per_elem.mean(dim=(1, 2, 3)),
        "per_horizon": per_elem.mean(dim=(1, 3)),
        "st": per_elem[..., :c_st].mean(dim=(1, 2, 3)) if c_st > 0 else zeros,
        "ph": per_elem[..., c_st:].mean(dim=(1, 2, 3)) if c_st < int(C) else zeros.clone(),
    }


def _empty_score(B: int, horizon: int, prefix: str, ref: Tensor) -> Dict[str, Tensor]:
    """Zero-filled result for a degenerate (empty) anchor range."""
    zeros = torch.zeros(B, device=ref.device, dtype=ref.dtype)
    return {
        f"{prefix}_total": zeros,
        f"{prefix}_per_horizon": torch.zeros(B, int(horizon), device=ref.device, dtype=ref.dtype),
        f"{prefix}_st": zeros.clone(),
        f"{prefix}_ph": zeros.clone(),
    }


def compute_nll(
    mu_full: Tensor,
    logvar_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
    *,
    include_const: bool = True,
) -> Dict[str, Tensor]:
    r"""Per-sample Gaussian negative log-likelihood of the future target, in nats.

    .. math::

        -\log p(y \mid \mu, \sigma) = \tfrac12 \log(2\pi) + \log\sigma
        + \frac{(y-\mu)^2}{2\sigma^2}

    Note:
        The **training** loss (``SeqVaeLagAttnV1.compute_loss``) omits the
        :math:`\tfrac12\log 2\pi` constant. Pass ``include_const=False`` to reproduce it
        exactly; the default includes it so the value is a proper scoring rule. The two differ
        by :math:`0.9189385` nats per element.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        logvar_full: Forecast observation log-variance ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.
        include_const: Whether to add :math:`\tfrac12\log 2\pi`.

    Returns:
        ``nll_total`` ``(B,)``, ``nll_per_horizon`` ``(B, H_d)``, and ``nll_st`` / ``nll_ph``
        ``(B,)`` for the scattering and phase channel blocks. Lower is better.
    """
    mu, sigma, y = _valid_triplet(mu_full, logvar_full, y_plus, warmup, horizon)
    if mu.shape[1] == 0:
        return _empty_score(mu.shape[0], horizon, "nll", mu)

    z = (y - mu) / sigma
    per_elem = 0.5 * z.pow(2) + torch.log(sigma)
    if include_const:
        per_elem = per_elem + _HALF_LOG_2PI
    return {f"nll_{k}": v for k, v in _split_channel_blocks(per_elem).items()}


def crps_gaussian(mu: Tensor, sigma: Tensor, y: Tensor) -> Tensor:
    r"""Closed-form CRPS of a Gaussian predictive distribution, elementwise.

    .. math::

        \mathrm{CRPS}\big(\mathcal{N}(\mu,\sigma^2), y\big)
        = \sigma\left[z\big(2\Phi(z) - 1\big) + 2\varphi(z) - \frac{1}{\sqrt{\pi}}\right],
        \qquad z = \frac{y - \mu}{\sigma}

    At :math:`y = \mu` this reduces to :math:`\sigma(\sqrt{2}-1)/\sqrt{\pi}
    \approx 0.2337\,\sigma`.

    Args:
        mu: Predictive mean.
        sigma: Predictive standard deviation (strictly positive).
        y: Observation.

    Returns:
        Elementwise CRPS with the same shape as the inputs. Lower is better.
    """
    z = (y - mu) / sigma
    return sigma * (
        z * (2.0 * _standard_normal_cdf(z) - 1.0)
        + 2.0 * _standard_normal_pdf(z)
        - _INV_SQRT_PI
    )


def crps_sample(samples: Tensor, y: Tensor) -> Tensor:
    r"""Sample-based CRPS estimator, for predictive distributions with no closed form.

    .. math::

        \mathrm{CRPS} = \mathbb{E}\lvert X - y\rvert
        - \tfrac12 \mathbb{E}\lvert X - X'\rvert

    with :math:`X, X'` independent draws from the predictive distribution. The spread term is
    the unbiased Gini mean difference, evaluated through its sorted-order identity

    .. math::

        \mathbb{E}\lvert X - X'\rvert
        = \frac{2}{n(n-1)} \sum_{i=1}^{n} \bigl(2i - n - 1\bigr)\, x_{(i)},

    so the estimator costs :math:`O(n \log n)` time and :math:`O(n)` memory rather than
    materialising the :math:`n \times n` pairwise-difference tensor.

    Args:
        samples: Draws ``(n, ...)`` with the sample axis first.
        y: Observation, broadcastable to ``samples[0]``.

    Returns:
        CRPS with the sample axis reduced.

    Raises:
        ValueError: If fewer than two draws are supplied.
    """
    n = int(samples.shape[0])
    if n < 2:
        raise ValueError(f"crps_sample needs at least 2 draws, got {n}")

    term_obs = (samples - y.unsqueeze(0)).abs().mean(dim=0)
    ordered, _ = torch.sort(samples, dim=0)
    rank = torch.arange(1, n + 1, device=samples.device, dtype=samples.dtype)
    coeff = (2.0 * rank - n - 1).reshape(-1, *([1] * (samples.dim() - 1)))
    gini = (coeff * ordered).sum(dim=0) * (2.0 / (n * (n - 1)))
    return term_obs - 0.5 * gini


def compute_crps(
    mu_full: Tensor,
    logvar_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Dict[str, Tensor]:
    r"""Per-sample continuous ranked probability score over the supervised anchors.

    CRPS rewards a forecast for being accurate *and* appropriately sharp. Unlike NLL it stays
    finite when an observation lands far in the tail, so one outlier cannot dominate the
    report.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        logvar_full: Forecast observation log-variance ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.

    Returns:
        ``crps_total`` ``(B,)``, ``crps_per_horizon`` ``(B, H_d)``, ``crps_st`` / ``crps_ph``
        ``(B,)``. Lower is better, and the score is in the target's units.
    """
    mu, sigma, y = _valid_triplet(mu_full, logvar_full, y_plus, warmup, horizon)
    if mu.shape[1] == 0:
        return _empty_score(mu.shape[0], horizon, "crps", mu)

    per_elem = crps_gaussian(mu, sigma, y)
    return {f"crps_{k}": v for k, v in _split_channel_blocks(per_elem).items()}


def compute_interval_coverage(
    mu_full: Tensor,
    logvar_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> Dict[str, Tensor]:
    r"""Empirical coverage of central prediction intervals, per sample and per level.

    For nominal level :math:`p` the interval is
    :math:`\mu \pm \Phi^{-1}\!\left(\tfrac{1+p}{2}\right)\sigma`. A calibrated forecaster
    covers the truth a fraction :math:`p` of the time. Systematic under-coverage means the
    learned :math:`\sigma` is too small -- the over-confidence that a variance-collapsed model
    hides from MSE entirely.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        logvar_full: Forecast observation log-variance ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.
        levels: Nominal central-interval levels, each strictly inside ``(0, 1)``.

    Returns:
        ``coverage`` ``(B, n_levels)``; ``nominal`` ``(n_levels,)``; ``sharpness`` ``(B,)``,
        the mean predictive :math:`\sigma` in the target's units; and
        ``sharpness_per_horizon`` ``(B, H_d)``.

    Raises:
        ValueError: If any level lies outside ``(0, 1)``.
    """
    if any(not (0.0 < float(p) < 1.0) for p in levels):
        raise ValueError(f"levels must lie strictly inside (0, 1), got {tuple(levels)}")

    mu, sigma, y = _valid_triplet(mu_full, logvar_full, y_plus, warmup, horizon)
    B = int(mu.shape[0])
    nominal = torch.tensor([float(p) for p in levels], device=mu.device, dtype=mu.dtype)
    if mu.shape[1] == 0:
        return {
            "coverage": torch.zeros(B, len(levels), device=mu.device, dtype=mu.dtype),
            "nominal": nominal,
            "sharpness": torch.zeros(B, device=mu.device, dtype=mu.dtype),
            "sharpness_per_horizon": torch.zeros(
                B, int(horizon), device=mu.device, dtype=mu.dtype
            ),
        }

    abs_z = ((y - mu) / sigma).abs()
    # Inverse standard-normal CDF at (1+p)/2, via erfinv.
    z_crit = 1.4142135623730951 * torch.erfinv(nominal)
    inside = abs_z.unsqueeze(-1) <= z_crit.view(1, 1, 1, 1, -1)
    return {
        "coverage": inside.to(mu.dtype).mean(dim=(1, 2, 3)),
        "nominal": nominal,
        "sharpness": sigma.mean(dim=(1, 2, 3)),
        "sharpness_per_horizon": sigma.mean(dim=(1, 3)),
    }


def compute_reliability_by_horizon(
    mu_full: Tensor,
    logvar_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
    *,
    n_bins: int = 20,
) -> Dict[str, Tensor]:
    r"""Probability-integral-transform reliability curves, resolved by horizon step.

    The PIT of a Gaussian forecast is :math:`u = \Phi\!\left((y-\mu)/\sigma\right)`. If the
    forecast is calibrated then :math:`u \sim \mathrm{Uniform}(0,1)`, so the empirical CDF of
    :math:`u` traces the diagonal. The shape of any deviation is diagnostic: an S-curve means
    the variance is misspecified, a shifted curve means the mean is biased.

    Pooled over batch, anchors and channels, but kept separate per horizon step, because
    calibration typically degrades with lead time.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        logvar_full: Forecast observation log-variance ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.
        n_bins: Number of quantile levels at which to evaluate the empirical CDF.

    Returns:
        ``nominal`` ``(n_bins,)`` -- the quantile levels;
        ``empirical`` ``(H_d, n_bins)`` -- observed fraction of PIT values at or below each
        level, per horizon step;
        ``empirical_pooled`` ``(n_bins,)`` -- the same, pooled over horizons;
        ``pit_mean`` / ``pit_var`` ``(H_d,)`` -- :math:`1/2` and :math:`1/12` when calibrated;
        ``ks_stat`` ``(H_d,)`` -- Kolmogorov-Smirnov distance from uniform.
    """
    mu, sigma, y = _valid_triplet(mu_full, logvar_full, y_plus, warmup, horizon)
    device, dtype = mu.device, mu.dtype
    H_d = int(mu.shape[2]) if mu.shape[1] > 0 else int(horizon)
    nominal = torch.linspace(0.0, 1.0, int(n_bins) + 1, device=device, dtype=dtype)[1:]

    if mu.shape[1] == 0:
        nan_h = torch.full((H_d,), float("nan"), device=device, dtype=dtype)
        return {
            "nominal": nominal,
            "empirical": torch.full((H_d, int(n_bins)), float("nan"), device=device, dtype=dtype),
            "empirical_pooled": torch.full(
                (int(n_bins),), float("nan"), device=device, dtype=dtype
            ),
            "pit_mean": nan_h,
            "pit_var": nan_h.clone(),
            "ks_stat": nan_h.clone(),
        }

    pit = _standard_normal_cdf((y - mu) / sigma)          # (B, T_v, H_d, C)
    pit_by_h = pit.permute(2, 0, 1, 3).reshape(H_d, -1)   # (H_d, N)

    empirical = (pit_by_h.unsqueeze(-1) <= nominal.view(1, 1, -1)).to(dtype).mean(dim=1)
    pooled = (pit_by_h.reshape(-1, 1) <= nominal.view(1, -1)).to(dtype).mean(dim=0)

    return {
        "nominal": nominal,
        "empirical": empirical,
        "empirical_pooled": pooled,
        "pit_mean": pit_by_h.mean(dim=1),
        "pit_var": pit_by_h.var(dim=1, unbiased=False),
        "ks_stat": (empirical - nominal.view(1, -1)).abs().max(dim=1).values,
    }


def fit_constant_sigma(
    mu_full: Tensor,
    y_plus: Tensor,
    warmup: int,
    horizon: int,
) -> Tensor:
    r"""Maximum-likelihood homoscedastic :math:`\sigma` over the supervised anchors.

    The reference every calibration report needs: the best a *single* global noise scale can
    do. If the learned heteroscedastic :math:`\sigma` does not beat this on NLL and CRPS, the
    learned variance is buying nothing. It also supplies a meaningful predictive distribution
    for a checkpoint trained with a fixed ``sigma_obs``, whose ``logvar_full`` head never
    received a gradient.

    Args:
        mu_full: Forecast mean ``(B, T, H_d, C)``.
        y_plus: Unfolded future target ``(B, T - H_d, H_d, C)``.
        warmup: Number of initial anchors to skip.
        horizon: Forecast horizon ``H_d``.

    Returns:
        A scalar tensor
        :math:`\hat{\sigma} = \sqrt{\operatorname{mean}\left[(y-\mu)^2\right]}`.
    """
    T = int(mu_full.shape[1])
    start, end = _feature_valid_slice(warmup, horizon, T)
    if end <= start:
        return torch.zeros((), device=mu_full.device, dtype=mu_full.dtype)
    resid = y_plus[:, start:end] - mu_full[:, start:end]
    return resid.pow(2).mean().sqrt()
