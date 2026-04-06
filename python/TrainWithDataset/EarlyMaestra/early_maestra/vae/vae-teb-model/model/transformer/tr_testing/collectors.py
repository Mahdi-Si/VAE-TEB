"""Data collection functions for the transformer testing pipeline.

Each collector iterates through a DataLoader batch-by-batch, runs model
inference, extracts specific quantities, and returns structured data.
All collectors preserve full segment identification metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import Tensor

from .base import TransformerTestRunner
from .metrics import (
    compute_fusion_contribution,
    compute_gate_statistics,
    compute_kl_per_anchor,
    compute_kl_per_dimension,
    compute_per_anchor_huber,
    compute_per_anchor_mae,
    compute_per_anchor_mse,
    compute_per_anchor_snr,
    compute_per_anchor_vaf,
    compute_per_channel_mae,
    compute_te_residual_norm,
    extract_targets,
)


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

def _extract_guid(batch: Any, idx: int) -> str:
    """Extract GUID string from a batch at index *idx*."""
    guid = getattr(batch, "guid", None)
    if guid is None:
        return "unknown"
    if isinstance(guid, (list, tuple)):
        val = guid[idx]
    elif isinstance(guid, Tensor):
        val = guid[idx].item() if guid.dim() > 0 else guid.item()
    else:
        val = str(guid)
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    return str(val)


def _extract_scalar(batch: Any, field: str, idx: int,
                    default: float = float("nan")) -> float:
    """Extract a scalar metadata field from a batch."""
    val = getattr(batch, field, None)
    if val is None:
        return default
    if isinstance(val, Tensor):
        return float(val[idx].item()) if val.dim() > 0 else float(val.item())
    if isinstance(val, (list, tuple)):
        return float(val[idx])
    return float(val)


def _extract_subgroup(batch: Any, idx: int) -> str:
    """Derive subgroup name from source HDF5 filename."""
    basename = getattr(batch, "source_file_basename", None)
    if basename is None:
        basename = getattr(batch, "source_file", None)
    if basename is None:
        return "unknown"
    if isinstance(basename, (list, tuple)):
        basename = basename[idx]
    if isinstance(basename, Tensor):
        basename = str(basename[idx].item())
    if isinstance(basename, bytes):
        basename = basename.decode("utf-8", errors="replace")
    name = str(basename)
    # Strip extension: "acidosis_cs.hdf5" -> "acidosis_cs"
    return Path(name).stem


def _build_metadata_row(
    batch: Any,
    idx: int,
    class_label: str,
) -> Dict[str, Any]:
    """Build common metadata dict for one sample in a batch.

    Args:
        batch: DataLoader batch (AttributeDict).
        idx: Sample index within the batch.
        class_label: Class name string.

    Returns:
        Metadata dictionary.
    """
    epoch = _extract_scalar(batch, "epoch", idx)
    return {
        "guid": _extract_guid(batch, idx),
        "epoch": epoch,
        "epoch_minutes": epoch / 60.0,
        "epoch_hours": epoch / 3600.0,
        "class_label": class_label,
        "cs_label": int(_extract_scalar(batch, "cs_label", idx, 0)),
        "bg_label": int(_extract_scalar(batch, "bg_label", idx, 0)),
        "subgroup": _extract_subgroup(batch, idx),
        "tlo_seconds": _extract_scalar(
            batch, "time_from_labor_onset", idx
        ),
    }


# ---------------------------------------------------------------------------
# Collector: forecast metrics
# ---------------------------------------------------------------------------

def collect_forecast_metrics(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect MAE, MSE, VAF, SNR, and Huber loss for all heads and horizons.

    One row per (segment, head, horizon).

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        max_samples: Maximum number of segments to process.

    Returns:
        DataFrame with columns: [common metadata, head, horizon, mae,
        mse, vaf, snr, huber_loss].
    """
    rows = []
    g = runner.config.guard_gap
    head_keys = {
        "self": "Y_hat_self",
        "fused": "Y_hat_fus",
        "te": "Y_hat_te",
    }

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            outputs = runner.forward_with_anchors(Y, U)
            anchors = outputs["anchor_indices"]
            K = anchors.shape[1]

            for head_name, head_key in head_keys.items():
                Y_hat = outputs[head_key]
                mae = compute_per_anchor_mae(Y_hat, Y, anchors, g)
                mse = compute_per_anchor_mse(Y_hat, Y, anchors, g)
                vaf = compute_per_anchor_vaf(Y_hat, Y, anchors, g)
                snr = compute_per_anchor_snr(Y_hat, Y, anchors, g)
                huber = compute_per_anchor_huber(Y_hat, Y, anchors, g)

                for h in runner.config.horizons:
                    # Average across anchors per segment
                    mae_h = mae[h].reshape(B, K).mean(dim=1).cpu().numpy()
                    mse_h = mse[h].reshape(B, K).mean(dim=1).cpu().numpy()
                    vaf_h = vaf[h].reshape(B, K).mean(dim=1).cpu().numpy()
                    snr_h = snr[h].reshape(B, K).mean(dim=1).cpu().numpy()
                    hub_h = huber[h].reshape(B, K).mean(dim=1).cpu().numpy()

                    for i in range(B):
                        row = _build_metadata_row(batch, i, class_label)
                        row.update({
                            "head": head_name,
                            "horizon": h,
                            "mae": mae_h[i],
                            "mse": mse_h[i],
                            "vaf": vaf_h[i],
                            "snr": snr_h[i],
                            "huber_loss": hub_h[i],
                        })
                        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collector: loss components
# ---------------------------------------------------------------------------

def collect_loss_components(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-sample loss decomposition.

    One row per segment.

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        max_samples: Maximum number of segments.

    Returns:
        DataFrame with columns: [common metadata, L_fus, L_delta,
        L_self, L_te, L_kl, total_loss].
    """
    rows = []
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            outputs = runner.forward_with_anchors(Y, U)
            losses = runner.compute_losses(outputs, Y)

            for i in range(B):
                row = _build_metadata_row(batch, i, class_label)
                for loss_name in ("L_fus", "L_delta", "L_self",
                                  "L_te", "L_kl", "total_loss"):
                    # These are batch-level scalars from the loss fn;
                    # store the same value for each sample in the batch
                    row[loss_name] = float(losses[loss_name].item())
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collector: TE latent data (two-level)
# ---------------------------------------------------------------------------

def collect_te_latent_data(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect TE latent data at anchor and segment levels.

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        max_samples: Maximum number of segments.

    Returns:
        Tuple of (anchor_level_df, segment_level_df).
    """
    d_z = runner.config.d_z
    anchor_rows = []
    segment_rows = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            outputs = runner.forward_with_anchors(Y, U)
            anchors = outputs["anchor_indices"]
            K = anchors.shape[1]

            mu_post = outputs["mu_post"].cpu()      # (B*K, d_z)
            logvar_post = outputs["logvar_post"].cpu()
            mu_prior = outputs["mu_prior"].cpu()
            logvar_prior = outputs["logvar_prior"].cpu()

            # KL per dimension and total
            kl_dim = compute_kl_per_dimension(
                mu_post, logvar_post, mu_prior, logvar_prior
            ).numpy()  # (B*K, d_z)
            kl_total = kl_dim.sum(axis=-1)  # (B*K,)

            # TE residual norms
            r_norms = compute_te_residual_norm(outputs["R_hat"])
            r_norms = {
                h: v.reshape(B, K).cpu().numpy() for h, v in r_norms.items()
            }

            # Reshape to (B, K, ...)
            mu_post_np = mu_post.numpy().reshape(B, K, d_z)
            logvar_post_np = logvar_post.numpy().reshape(B, K, d_z)
            mu_prior_np = mu_prior.numpy().reshape(B, K, d_z)
            logvar_prior_np = logvar_prior.numpy().reshape(B, K, d_z)
            kl_dim_np = kl_dim.reshape(B, K, d_z)
            kl_total_np = kl_total.reshape(B, K)
            anchors_np = anchors.cpu().numpy()

            for i in range(B):
                meta = _build_metadata_row(batch, i, class_label)

                # --- Anchor-level rows ---
                for k in range(K):
                    arow = dict(meta)
                    arow["anchor_idx"] = k
                    arow["anchor_timestep"] = int(anchors_np[i, k])
                    arow["kl_total"] = kl_total_np[i, k]
                    for d in range(d_z):
                        arow[f"kl_dim_{d}"] = kl_dim_np[i, k, d]
                        arow[f"mu_post_{d}"] = mu_post_np[i, k, d]
                        arow[f"mu_prior_{d}"] = mu_prior_np[i, k, d]
                        arow[f"logvar_post_{d}"] = logvar_post_np[i, k, d]
                        arow[f"logvar_prior_{d}"] = logvar_prior_np[i, k, d]
                    for h in runner.config.horizons:
                        arow[f"residual_norm_h{h}"] = r_norms[h][i, k]
                    anchor_rows.append(arow)

                # --- Segment-level row (aggregated across anchors) ---
                srow = dict(meta)
                srow["n_anchors"] = K
                kl_seg = kl_total_np[i]
                srow["kl_mean"] = kl_seg.mean()
                srow["kl_max"] = kl_seg.max()
                srow["kl_min"] = kl_seg.min()
                srow["kl_std"] = kl_seg.std()

                for d in range(d_z):
                    kd = kl_dim_np[i, :, d]
                    srow[f"kl_dim_mean_{d}"] = kd.mean()
                    srow[f"kl_dim_max_{d}"] = kd.max()
                    srow[f"kl_dim_min_{d}"] = kd.min()

                    mp = mu_post_np[i, :, d]
                    srow[f"mu_post_mean_{d}"] = mp.mean()
                    srow[f"mu_post_max_{d}"] = mp.max()
                    srow[f"mu_post_min_{d}"] = mp.min()

                    mpr = mu_prior_np[i, :, d]
                    srow[f"mu_prior_mean_{d}"] = mpr.mean()
                    srow[f"mu_prior_max_{d}"] = mpr.max()
                    srow[f"mu_prior_min_{d}"] = mpr.min()

                    srow[f"logvar_post_mean_{d}"] = logvar_post_np[i, :, d].mean()
                    srow[f"logvar_prior_mean_{d}"] = logvar_prior_np[i, :, d].mean()

                for h in runner.config.horizons:
                    rn = r_norms[h][i]
                    srow[f"residual_norm_mean_h{h}"] = rn.mean()
                    srow[f"residual_norm_max_h{h}"] = rn.max()
                    srow[f"residual_norm_min_h{h}"] = rn.min()

                segment_rows.append(srow)

    return pd.DataFrame(anchor_rows), pd.DataFrame(segment_rows)


# ---------------------------------------------------------------------------
# Collector: embeddings
# ---------------------------------------------------------------------------

def collect_embeddings(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect window embeddings and their component decomposition.

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        max_samples: Maximum number of segments.

    Returns:
        Dictionary with:
            - ``"e_win"``: ``(N, output_dim)`` full embedding array.
            - ``"e_F"``: ``(N, 2*d)`` FHR component.
            - ``"e_FU"``: ``(N, 6*d)`` fused component.
            - ``"e_TE"``: ``(N, 2*d_z)`` TE component.
            - ``"metadata"``: DataFrame with common metadata.
    """
    d = runner.config.d_model
    d_z = runner.config.d_z
    boundary_f = 2 * d
    boundary_fu = boundary_f + 6 * d

    all_ewin = []
    meta_rows = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            e_win = runner.forward_for_embedding(Y, U).cpu().numpy()
            all_ewin.append(e_win)

            for i in range(B):
                meta_rows.append(
                    _build_metadata_row(batch, i, class_label)
                )

    if not all_ewin:
        empty = np.empty((0, 8 * d + 2 * d_z))
        return {
            "e_win": empty,
            "e_F": empty[:, :boundary_f],
            "e_FU": empty[:, boundary_f:boundary_fu],
            "e_TE": empty[:, boundary_fu:],
            "metadata": pd.DataFrame(),
        }

    e_win = np.concatenate(all_ewin, axis=0)
    return {
        "e_win": e_win,
        "e_F": e_win[:, :boundary_f],
        "e_FU": e_win[:, boundary_f:boundary_fu],
        "e_TE": e_win[:, boundary_fu:],
        "metadata": pd.DataFrame(meta_rows),
    }


# ---------------------------------------------------------------------------
# Collector: gate and fusion contribution
# ---------------------------------------------------------------------------

def collect_gate_and_fusion(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Collect per-segment gate activation and fusion contribution stats.

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        max_samples: Maximum number of segments.

    Returns:
        DataFrame with [common metadata, gate stats, fusion stats].
    """
    rows = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            intermediates = runner.extract_intermediates(Y, U)
            gate = intermediates["gate"]
            H_F = intermediates["H_F"]
            H_FU = intermediates["H_FU"]

            gate_stats = compute_gate_statistics(gate)
            l2_dist, relative = compute_fusion_contribution(H_FU, H_F)

            for i in range(B):
                row = _build_metadata_row(batch, i, class_label)
                row["mean_gate"] = float(gate_stats["mean"][i].item())
                row["std_gate"] = float(gate_stats["std"][i].item())
                row["min_gate"] = float(gate_stats["min"][i].item())
                row["max_gate"] = float(gate_stats["max"][i].item())
                row["mean_fusion_dist"] = float(l2_dist[i].mean().item())
                row["max_fusion_dist"] = float(l2_dist[i].max().item())
                row["relative_fusion_mean"] = float(relative[i].mean().item())

                # Gate temporal profile as columns
                profile = gate_stats["temporal_profile"][i].cpu().numpy()
                for t in range(len(profile)):
                    row[f"gate_t{t:03d}"] = profile[t]

                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Collector: full sample data (for per-sample diagnostics)
# ---------------------------------------------------------------------------

def collect_full_sample_data(
    runner: TransformerTestRunner,
    loader: Any,
    class_label: str,
    n_samples: int = 10,
) -> List[Dict[str, Any]]:
    """Collect complete data for per-sample diagnostic figures.

    Runs both anchor-based forward, intermediate extraction, and embedding
    forward for a limited number of samples.

    Args:
        runner: TransformerTestRunner instance.
        loader: DataLoader for one class.
        class_label: Class name string.
        n_samples: Number of samples to collect.

    Returns:
        List of dicts, each containing all tensors and metadata for one sample.
    """
    samples = []

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, n_samples):
            Y = batch.fhr_st
            U = batch.up_st
            B = Y.shape[0]

            # All forward passes
            outputs = runner.forward_with_anchors(Y, U)
            intermediates = runner.extract_intermediates(Y, U)
            e_win = runner.forward_for_embedding(Y, U)

            for i in range(B):
                if len(samples) >= n_samples:
                    break

                K = outputs["anchor_indices"].shape[1]

                sample = _build_metadata_row(batch, i, class_label)
                sample["Y"] = Y[i].cpu().numpy()
                sample["U"] = U[i].cpu().numpy()

                # Raw signals if available
                fhr_raw = getattr(batch, "fhr", None)
                up_raw = getattr(batch, "up", None)
                if fhr_raw is not None:
                    sample["fhr_raw"] = fhr_raw[i].cpu().numpy()
                if up_raw is not None:
                    sample["up_raw"] = up_raw[i].cpu().numpy()

                # Encoder states
                sample["H_F"] = intermediates["H_F"][i].cpu().numpy()
                sample["H_U"] = intermediates["H_U"][i].cpu().numpy()
                sample["H_FU"] = intermediates["H_FU"][i].cpu().numpy()
                sample["gate"] = intermediates["gate"][i].cpu().numpy()

                # Anchor indices
                sample["anchor_indices"] = (
                    outputs["anchor_indices"][i].cpu().numpy()
                )

                # Forecasts (per-anchor for this sample)
                for head_key in ("Y_hat_self", "Y_hat_fus", "Y_hat_te",
                                 "R_hat"):
                    sample[head_key] = {}
                    for h in runner.config.horizons:
                        sample[head_key][h] = (
                            outputs[head_key][h][i * K:(i + 1) * K]
                            .cpu().numpy()
                        )

                # TE latent parameters
                sample["mu_post"] = (
                    outputs["mu_post"][i * K:(i + 1) * K].cpu().numpy()
                )
                sample["mu_prior"] = (
                    outputs["mu_prior"][i * K:(i + 1) * K].cpu().numpy()
                )
                sample["logvar_post"] = (
                    outputs["logvar_post"][i * K:(i + 1) * K].cpu().numpy()
                )
                sample["logvar_prior"] = (
                    outputs["logvar_prior"][i * K:(i + 1) * K].cpu().numpy()
                )

                # Window embedding
                sample["e_win"] = e_win[i].cpu().numpy()

                samples.append(sample)

    logger.info(
        f"Collected {len(samples)} full samples for {class_label}"
    )
    return samples
