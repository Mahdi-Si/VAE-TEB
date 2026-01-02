"""
Qualitative prediction diagnostics for the VAE-TEB testing pipeline.

This module mirrors key plots from the legacy testing script:
- Per-sample model analysis panels
- VAE reconstruction diagnostics (scattering/phase harmonics)
- Single prediction window visualization (no averaging)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import _extract_epoch, _extract_guid
from model.vae_teb_prediction.testing.metrics import aggregate_predictions, compute_kld, compute_reconstruction_metrics
from utils.plot_utils import plot_model_analysis, plot_single_prediction_windows, plot_vae_reconstruction


def _get_normalization_stats(loader: Any) -> Optional[Dict[str, Any]]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "get_normalization_stats"):
        return None
    try:
        return dataset.get_normalization_stats()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch normalization stats: %s", exc)
        return None


def _denormalize_tensor(
    tensor: torch.Tensor,
    field: str,
    stats: Optional[Dict[str, Any]],
    *,
    raw_start: Optional[int] = None,
    length: Optional[int] = None,
) -> torch.Tensor:
    if not stats or field not in stats:
        return tensor
    try:
        field_stats = stats[field] or {}
        mean = field_stats.get("mean_tensor", field_stats.get("mean", 0.0))
        std = field_stats.get("std_tensor", field_stats.get("std", 1.0))

        mean_t = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
        std_t = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)

        if (
            mean_t.dim() > 0
            and raw_start is not None
            and length is not None
            and mean_t.size(-1) >= raw_start + length
        ):
            mean_t = mean_t.narrow(-1, raw_start, length)
        if (
            std_t.dim() > 0
            and raw_start is not None
            and length is not None
            and std_t.size(-1) >= raw_start + length
        ):
            std_t = std_t.narrow(-1, raw_start, length)

        while mean_t.dim() < tensor.dim():
            mean_t = mean_t.unsqueeze(0)
        while std_t.dim() < tensor.dim():
            std_t = std_t.unsqueeze(0)

        return tensor * (std_t + 1e-8) + mean_t
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to denormalize %s: %s. Returning tensor as-is.", field, exc)
        return tensor


def _denormalize_std(
    std_tensor: torch.Tensor,
    field: str,
    stats: Optional[Dict[str, Any]],
    *,
    raw_start: Optional[int] = None,
    length: Optional[int] = None,
) -> torch.Tensor:
    if not stats or field not in stats:
        return std_tensor
    try:
        field_stats = stats[field] or {}
        scale = field_stats.get("std_tensor", field_stats.get("std", 1.0))
        scale_tensor = torch.as_tensor(scale, dtype=std_tensor.dtype, device=std_tensor.device)

        if (
            scale_tensor.dim() > 0
            and raw_start is not None
            and length is not None
            and scale_tensor.size(-1) >= raw_start + length
        ):
            scale_tensor = scale_tensor.narrow(-1, raw_start, length)
        while scale_tensor.dim() < std_tensor.dim():
            scale_tensor = scale_tensor.unsqueeze(0)

        return std_tensor * (scale_tensor + 1e-8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to denormalize std for %s: %s. Returning std as-is.", field, exc)
        return std_tensor


def _extract_reconstruction_features(
    linear_output: Optional[torch.Tensor],
    st_channels: int,
    ph_channels: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if (
        linear_output is None
        or linear_output.dim() != 3
        or linear_output.size(-1) < st_channels + ph_channels
    ):
        return None, None
    linear_np = linear_output[0].detach().cpu().numpy()
    recon_st = linear_np[:, :st_channels].T
    recon_ph = linear_np[:, st_channels : st_channels + ph_channels].T
    return recon_st, recon_ph


def _aggregate_segments(
    model: torch.nn.Module,
    segments: Optional[torch.Tensor],
    logvar_segments: Optional[torch.Tensor],
    *,
    raw_len: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if segments is None:
        return None, None, None

    if segments.dim() == 2:
        segments = segments.unsqueeze(0)
    if segments.dim() != 3:
        return None, None, None

    avg_mu, valid_mask = aggregate_predictions(model, segments, raw_len=raw_len)

    avg_logvar = None
    if logvar_segments is not None:
        logvar_use = logvar_segments
        if logvar_use.dim() == 2:
            logvar_use = logvar_use.unsqueeze(0)
        if logvar_use.dim() == 3:
            avg_var, _ = aggregate_predictions(model, logvar_use.exp(), raw_len=raw_len)
            if avg_var is not None:
                avg_logvar = avg_var.clamp_min(1e-12).log()
                if valid_mask is not None:
                    avg_logvar = torch.where(valid_mask, avg_logvar, torch.tensor(float("nan"), device=avg_logvar.device))

    if avg_mu is not None and avg_mu.dim() == 2:
        avg_mu = avg_mu[0]
    if avg_logvar is not None and avg_logvar.dim() == 2:
        avg_logvar = avg_logvar[0]
    if valid_mask is not None and valid_mask.dim() == 2:
        valid_mask = valid_mask[0]

    return avg_mu, avg_logvar, valid_mask


def _prepare_single_prediction_windows(
    *,
    predictions: torch.Tensor,
    logvar_predictions: Optional[torch.Tensor],
    raw_norm: torch.Tensor,
    raw_denorm: torch.Tensor,
    start_index: int,
    step_size: Optional[int],
    max_windows: int,
    stats: Optional[Dict[str, Any]],
    stride: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    if predictions.dim() == 3:
        predictions = predictions.squeeze(0)
    if predictions.dim() != 2:
        return []

    horizon = predictions.size(-1)
    effective_start = max(0, max(start_index, warmup))
    if step_size is None or step_size <= 0:
        effective_step = max(1, horizon // max(1, stride))
    else:
        effective_step = int(step_size)

    raw_norm = raw_norm.view(-1)
    raw_denorm = raw_denorm.view(-1)
    raw_len = raw_norm.size(0)

    windows: List[Dict[str, Any]] = []
    total_steps = predictions.size(0)
    t_idx = effective_start
    while t_idx < total_steps and len(windows) < max_windows:
        raw_start = t_idx * stride
        raw_end = raw_start + horizon
        if raw_end > raw_len:
            break

        pred_segment = predictions[t_idx]
        logvar_segment = logvar_predictions[t_idx] if logvar_predictions is not None else None

        target_norm = raw_norm[raw_start:raw_end]
        target_denorm = raw_denorm[raw_start:raw_end]

        pred_denorm = _denormalize_tensor(
            pred_segment.unsqueeze(0),
            "fhr",
            stats,
            raw_start=raw_start,
            length=horizon,
        )[0]

        std_denorm_np = None
        std_norm_np = None
        if logvar_segment is not None:
            std_norm = torch.exp(0.5 * logvar_segment)
            std_norm_np = std_norm.detach().cpu().numpy()
            std_denorm = _denormalize_std(
                std_norm.unsqueeze(0),
                "fhr",
                stats,
                raw_start=raw_start,
                length=horizon,
            )[0]
            std_denorm_np = std_denorm.detach().cpu().numpy()

        metrics = compute_reconstruction_metrics(
            target_denorm.unsqueeze(0),
            pred_denorm.unsqueeze(0),
        )
        metrics_map = (
            {key: float(val.detach().cpu().item()) for key, val in metrics.items()}
            if metrics is not None
            else {}
        )

        windows.append(
            {
                "t_index": int(t_idx),
                "raw_start": int(raw_start),
                "raw_end": int(raw_end),
                "prediction": pred_denorm.detach().cpu().numpy(),
                "target": target_denorm.detach().cpu().numpy(),
                "prediction_norm": pred_segment.detach().cpu().numpy(),
                "target_norm": target_norm.detach().cpu().numpy(),
                "uncertainty": std_denorm_np,
                "uncertainty_norm": std_norm_np,
                "metrics": metrics_map,
            }
        )
        t_idx += effective_step

    return windows


def run_reconstruction_analysis(
    runner: TestRunner,
    loader: Any,
    *,
    max_samples: int = 10,
    output_dir: Optional[Path] = None,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Generate detailed per-sample analysis plots (model analysis + VAE reconstruction).
    """
    if max_samples <= 0:
        logger.info("Reconstruction analysis skipped (max_samples <= 0).")
        return {"n_samples": 0}

    out_dir = Path(output_dir) if output_dir is not None else runner.ensure_dir("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _get_normalization_stats(loader)

    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)

            outputs = runner.forward(batch)
            mu_pr = outputs.get("mu_pr")
            logvar_pr = outputs.get("logvar_pr")
            latent = outputs.get("z")
            linear_output = outputs.get("linear_output")
            kld_tensor = compute_kld(outputs, runner.warmup_steps)

            for idx in range(batch_size):
                if processed >= max_samples:
                    break

                y_st = batch.fhr_st[idx : idx + 1]
                y_ph = batch.fhr_ph[idx : idx + 1]
                x_ph = batch.fhr_up_ph[idx : idx + 1]
                y_raw = batch.fhr[idx : idx + 1]

                up_raw_tensor = getattr(batch, "up", None)
                up_raw = up_raw_tensor[idx : idx + 1] if up_raw_tensor is not None else torch.zeros_like(y_raw)

                if mu_pr is None or latent is None:
                    logger.warning("Missing prediction outputs for sample %d; skipping.", processed)
                    continue

                avg_mu, avg_logvar, valid_mask = _aggregate_segments(
                    runner.model,
                    mu_pr[idx],
                    logvar_pr[idx] if logvar_pr is not None else None,
                    raw_len=y_raw.size(1),
                )
                if avg_mu is None:
                    logger.warning("Aggregated predictions missing for sample %d; skipping.", processed)
                    continue

                sample_outputs: Dict[str, Any] = {}
                for key, val in outputs.items():
                    if torch.is_tensor(val):
                        sample_outputs[key] = val[idx : idx + 1]
                    else:
                        sample_outputs[key] = val
                loss_dict = None
                try:
                    loss_dict = runner.model.compute_loss(
                        forward_outputs=sample_outputs,
                        y_st=y_st,
                        y_ph=y_ph,
                        y_raw=y_raw,
                        beta=beta,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Loss computation failed for sample %d: %s", processed, exc)

                loss_floats: Dict[str, float] = {}
                if loss_dict:
                    for key, val in loss_dict.items():
                        if torch.is_tensor(val):
                            loss_floats[key] = float(torch.nan_to_num(val.detach()).cpu().item())
                        else:
                            loss_floats[key] = float(val)

                kld_tensor_np = None
                kld_mean_np = None
                if kld_tensor is not None:
                    kld_sample = kld_tensor[idx]
                    kld_tensor_np = kld_sample.detach().cpu().numpy().T
                    kld_mean_np = torch.nanmean(kld_sample, dim=-1).detach().cpu().numpy()

                fhr_denorm = _denormalize_tensor(y_raw, "fhr", stats)
                up_denorm = _denormalize_tensor(up_raw, "up", stats)

                raw_fhr_norm_np = y_raw[0].detach().cpu().numpy()
                raw_up_norm_np = up_raw[0].detach().cpu().numpy()
                raw_fhr_denorm_np = fhr_denorm[0].detach().cpu().numpy()
                raw_up_denorm_np = up_denorm[0].detach().cpu().numpy()

                fhr_st_np = y_st[0].detach().cpu().numpy().T
                fhr_ph_np = y_ph[0].detach().cpu().numpy().T
                fhr_up_ph_np = x_ph[0].detach().cpu().numpy().T
                latent_np = latent[idx].detach().cpu().numpy().T

                recon_np = avg_mu.detach().cpu().numpy()
                logvar_np = avg_logvar.detach().cpu().numpy() if avg_logvar is not None else None
                if valid_mask is not None and logvar_np is not None:
                    mask_np = valid_mask.detach().cpu().numpy()
                    logvar_np = np.where(mask_np, logvar_np, np.nan)

                recon_st_np, recon_ph_np = _extract_reconstruction_features(
                    linear_output[idx : idx + 1] if linear_output is not None else None,
                    st_channels=fhr_st_np.shape[0],
                    ph_channels=fhr_ph_np.shape[0],
                )

                plot_model_analysis(
                    output_dir=str(out_dir),
                    raw_fhr=raw_fhr_denorm_np,
                    raw_up=raw_up_denorm_np,
                    fhr_st=fhr_st_np,
                    fhr_ph=fhr_ph_np,
                    fhr_up_ph=fhr_up_ph_np,
                    latent_z=latent_np,
                    reconstructed_fhr_mu=recon_np,
                    reconstructed_fhr_logvar=logvar_np,
                    kld_tensor=kld_tensor_np,
                    kld_mean_over_channels=kld_mean_np,
                    batch_idx=processed,
                    loss_dict=loss_floats,
                    raw_fhr_normalized=raw_fhr_norm_np,
                    raw_up_normalized=raw_up_norm_np,
                )

                plot_vae_reconstruction(
                    output_dir=str(out_dir),
                    raw_fhr_unnormalized=raw_fhr_denorm_np,
                    raw_up_unnormalized=raw_up_denorm_np,
                    raw_fhr_normalized=raw_fhr_norm_np,
                    raw_up_normalized=raw_up_norm_np,
                    reconstructed_fhr=recon_np,
                    original_scattering_transform=fhr_st_np,
                    reconstructed_scattering_transform=recon_st_np,
                    original_phase_harmonic=fhr_ph_np,
                    reconstructed_phase_harmonic=recon_ph_np,
                    scattering_channel_data=None,
                    batch_idx=processed,
                    loss_dict=loss_floats,
                )

                processed += 1
            if processed >= max_samples:
                break

    logger.info("Reconstruction analysis plots saved to %s (samples=%d)", out_dir, processed)
    return {"output_dir": str(out_dir), "n_samples": processed}


def run_single_prediction_windows(
    runner: TestRunner,
    loader: Any,
    *,
    max_samples: int = 5,
    start_index: int = 20,
    step_size: Optional[int] = None,
    windows_per_sample: int = 4,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate single-window prediction plots (no averaging) for a subset of samples.
    """
    if max_samples <= 0 or windows_per_sample <= 0:
        logger.info("Single prediction windows skipped (max_samples<=0 or windows_per_sample<=0).")
        return {"n_samples": 0}

    out_dir = Path(output_dir) if output_dir is not None else runner.ensure_dir("single_prediction_windows")
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _get_normalization_stats(loader)

    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            batch_size = batch.fhr_st.size(0)
            outputs = runner.forward(batch)
            mu_pr = outputs.get("mu_pr")
            logvar_pr = outputs.get("logvar_pr")

            if mu_pr is None:
                logger.warning("Model outputs missing 'mu_pr'; skipping batch.")
                continue

            for idx in range(batch_size):
                if processed >= max_samples:
                    break

                raw_norm = batch.fhr[idx]
                raw_denorm = _denormalize_tensor(raw_norm, "fhr", stats)

                windows = _prepare_single_prediction_windows(
                    predictions=mu_pr[idx],
                    logvar_predictions=logvar_pr[idx] if logvar_pr is not None else None,
                    raw_norm=raw_norm,
                    raw_denorm=raw_denorm,
                    start_index=start_index,
                    step_size=step_size,
                    max_windows=windows_per_sample,
                    stats=stats,
                    stride=runner.decimation_factor,
                    warmup=runner.warmup_steps,
                )
                if not windows:
                    continue

                raw_fhr_norm_np = raw_norm.detach().cpu().numpy()
                agg_pred_norm = np.full_like(raw_fhr_norm_np, np.nan)
                agg_uncert_norm = np.full_like(raw_fhr_norm_np, np.nan)
                for window in windows:
                    start = window["raw_start"]
                    end = window["raw_end"]
                    pred_norm = window.get("prediction_norm")
                    if pred_norm is not None:
                        agg_pred_norm[start:end] = pred_norm
                    uncert_norm = window.get("uncertainty_norm")
                    if uncert_norm is not None:
                        agg_uncert_norm[start:end] = uncert_norm

                raw_fhr_denorm_np = raw_denorm.detach().cpu().numpy()
                plot_single_prediction_windows(
                    output_dir=str(out_dir),
                    raw_fhr_unnormalized=raw_fhr_denorm_np,
                    raw_fhr_normalized=raw_fhr_norm_np,
                    windows=windows,
                    aggregated_pred_norm=agg_pred_norm,
                    aggregated_uncertainty_norm=agg_uncert_norm,
                    sample_idx=processed,
                    sample_guid=_extract_guid(batch, idx),
                    epoch=_extract_epoch(batch, idx),
                )
                processed += 1

            if processed >= max_samples:
                break

    if processed == 0:
        logger.warning("Single prediction windows did not generate any plots.")
    else:
        logger.info("Single prediction windows saved %d samples to %s", processed, out_dir)

    return {"output_dir": str(out_dir), "n_samples": processed}
