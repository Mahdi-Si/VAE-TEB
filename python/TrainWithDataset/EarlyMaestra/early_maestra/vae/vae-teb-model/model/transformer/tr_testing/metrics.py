"""Pure metric computation functions for the transformer testing pipeline.

All functions are stateless and operate on tensors or numpy arrays.  No model
or I/O side-effects.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Target extraction (mirrors CausalTransformerLoss._extract_targets)
# ---------------------------------------------------------------------------

def extract_targets(
    Y: Tensor,
    anchor_indices: Tensor,
    horizon: int,
    guard_gap: int,
) -> Tensor:
    """Extract future FHR target blocks for given anchors and horizon.

    For anchor *a* and guard gap *g*, extracts
    ``Y[:, a+g+1 : a+g+1+h, :]``.

    Args:
        Y: FHR scattering features ``(B, T, d_F)``.
        anchor_indices: Anchor positions ``(B, K)``.
        horizon: Number of future time steps to extract.
        guard_gap: Guard gap between anchor and target start.

    Returns:
        Target blocks ``(B*K, h, d_F)``.
    """
    B, T, d_f = Y.shape
    K = anchor_indices.shape[1]
    starts = anchor_indices + guard_gap + 1  # (B, K)
    offsets = torch.arange(horizon, device=Y.device)  # (h,)
    time_idx = starts.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    time_idx_exp = time_idx.unsqueeze(-1).expand(B, K, horizon, d_f)
    Y_exp = Y.unsqueeze(1).expand(B, K, T, d_f)
    targets = torch.gather(Y_exp, dim=2, index=time_idx_exp)
    return targets.reshape(B * K, horizon, d_f)


# ---------------------------------------------------------------------------
# Forecasting metrics
# ---------------------------------------------------------------------------

def compute_per_anchor_mae(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
) -> Dict[int, Tensor]:
    """MAE per anchor for one forecast head, across all horizons.

    Args:
        Y_hat: ``{horizon: (B*K, h, d_f)}`` predictions from one head.
        Y: FHR input ``(B, T, d_F)``.
        anchor_indices: ``(B, K)`` anchor positions.
        guard_gap: Guard gap.

    Returns:
        ``{horizon: (B*K,)}`` per-anchor MAE.
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        result[h] = (pred - targets).abs().mean(dim=(1, 2))
    return result


def compute_per_anchor_mse(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
) -> Dict[int, Tensor]:
    """MSE per anchor for one forecast head.

    Returns:
        ``{horizon: (B*K,)}`` per-anchor MSE.
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        result[h] = (pred - targets).pow(2).mean(dim=(1, 2))
    return result


def compute_per_anchor_vaf(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
) -> Dict[int, Tensor]:
    """Variance Accounted For per anchor.

    ``VAF = 1 - var(target - pred) / var(target)``

    Returns:
        ``{horizon: (B*K,)}`` per-anchor VAF (higher is better, 1.0 = perfect).
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        error = targets - pred
        # Flatten time and channel dims for variance
        flat_target = targets.reshape(targets.shape[0], -1)
        flat_error = error.reshape(error.shape[0], -1)
        var_target = flat_target.var(dim=1).clamp(min=1e-10)
        var_error = flat_error.var(dim=1)
        result[h] = 1.0 - var_error / var_target
    return result


def compute_per_anchor_snr(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
) -> Dict[int, Tensor]:
    """Signal-to-Noise Ratio per anchor in dB.

    ``SNR = 10 * log10(var(target) / var(error))``

    Returns:
        ``{horizon: (B*K,)}`` per-anchor SNR in dB.
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        error = targets - pred
        flat_target = targets.reshape(targets.shape[0], -1)
        flat_error = error.reshape(error.shape[0], -1)
        var_target = flat_target.var(dim=1).clamp(min=1e-10)
        var_error = flat_error.var(dim=1).clamp(min=1e-10)
        result[h] = 10.0 * torch.log10(var_target / var_error)
    return result


def compute_per_anchor_huber(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
    delta: float = 1.0,
) -> Dict[int, Tensor]:
    """Huber loss per anchor.

    Returns:
        ``{horizon: (B*K,)}`` per-anchor Huber loss.
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        # Per-element Huber, then mean over time and channels per anchor
        loss = F.huber_loss(pred, targets, reduction="none", delta=delta)
        result[h] = loss.mean(dim=(1, 2))
    return result


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def compute_kl_per_anchor(
    mu_post: Tensor,
    logvar_post: Tensor,
    mu_prior: Tensor,
    logvar_prior: Tensor,
) -> Tensor:
    """Total KL divergence per anchor (summed over latent dims).

    Args:
        mu_post: Posterior mean ``(N, d_z)``.
        logvar_post: Posterior log-variance ``(N, d_z)``.
        mu_prior: Prior mean ``(N, d_z)``.
        logvar_prior: Prior log-variance ``(N, d_z)``.

    Returns:
        ``(N,)`` total KL per anchor.
    """
    kl = compute_kl_per_dimension(mu_post, logvar_post, mu_prior, logvar_prior)
    return kl.sum(dim=-1)


def compute_kl_per_dimension(
    mu_post: Tensor,
    logvar_post: Tensor,
    mu_prior: Tensor,
    logvar_prior: Tensor,
) -> Tensor:
    """Per-dimension KL divergence between posterior and prior Gaussians.

    Args:
        mu_post: ``(N, d_z)``.
        logvar_post: ``(N, d_z)``.
        mu_prior: ``(N, d_z)``.
        logvar_prior: ``(N, d_z)``.

    Returns:
        ``(N, d_z)`` per-dimension KL.
    """
    return 0.5 * (
        logvar_prior - logvar_post
        + (logvar_post.exp() + (mu_post - mu_prior).pow(2))
        / logvar_prior.exp()
        - 1.0
    )


# ---------------------------------------------------------------------------
# TE residual
# ---------------------------------------------------------------------------

def compute_te_residual_norm(
    R_hat: Dict[int, Tensor],
) -> Dict[int, Tensor]:
    """L2 norm of TE residual predictions per anchor per horizon.

    Args:
        R_hat: ``{horizon: (B*K, h, d_f)}`` residual predictions.

    Returns:
        ``{horizon: (B*K,)}`` L2 norm of residual.
    """
    result = {}
    for h, r in R_hat.items():
        # Flatten time×channels, compute L2 norm per anchor
        result[h] = r.reshape(r.shape[0], -1).norm(dim=-1)
    return result


# ---------------------------------------------------------------------------
# Fusion and gate
# ---------------------------------------------------------------------------

def compute_fusion_contribution(
    H_FU: Tensor,
    H_F: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute fusion contribution per timestep.

    Args:
        H_FU: Fused encoder states ``(B, T, d)``.
        H_F: FHR-only encoder states ``(B, T, d)``.

    Returns:
        Tuple of:
            - ``(B, T)`` L2 distance ``||H_FU - H_F||``.
            - ``(B, T)`` relative change ``||H_FU - H_F|| / (||H_F|| + eps)``.
    """
    diff = H_FU - H_F
    l2_dist = diff.norm(dim=-1)  # (B, T)
    h_f_norm = H_F.norm(dim=-1).clamp(min=1e-8)
    relative = l2_dist / h_f_norm
    return l2_dist, relative


def compute_gate_statistics(gate: Tensor) -> Dict[str, Tensor]:
    """Compute summary statistics of gate activations.

    Args:
        gate: Gate tensor ``(B, T, d)``.

    Returns:
        Dictionary with keys ``mean``, ``std``, ``min``, ``max`` (each ``(B,)``)
        and ``temporal_profile`` ``(B, T)`` (mean across hidden dim).
    """
    temporal = gate.mean(dim=-1)  # (B, T)
    return {
        "mean": temporal.mean(dim=-1),     # (B,)
        "std": temporal.std(dim=-1),       # (B,)
        "min": temporal.min(dim=-1).values,  # (B,)
        "max": temporal.max(dim=-1).values,  # (B,)
        "temporal_profile": temporal,        # (B, T)
    }


# ---------------------------------------------------------------------------
# Per-channel forecast error
# ---------------------------------------------------------------------------

def compute_per_channel_mae(
    Y_hat: Dict[int, Tensor],
    Y: Tensor,
    anchor_indices: Tensor,
    guard_gap: int,
) -> Dict[int, Tensor]:
    """MAE per channel per horizon, averaged across anchors.

    Returns:
        ``{horizon: (d_f,)}`` mean MAE per channel.
    """
    result = {}
    for h, pred in Y_hat.items():
        targets = extract_targets(Y, anchor_indices, h, guard_gap)
        # (B*K, h, d_f) -> mean over B*K and h -> (d_f,)
        result[h] = (pred - targets).abs().mean(dim=(0, 1))
    return result
