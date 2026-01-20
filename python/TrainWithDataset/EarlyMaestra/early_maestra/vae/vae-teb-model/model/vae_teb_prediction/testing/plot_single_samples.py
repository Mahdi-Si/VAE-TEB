"""
Single sample plotting utility for VAE-TEB testing pipeline.

This module provides functionality to randomly select samples from the test
dataset and generate all possible single-sample plots for each, organized
in folders named by GUID and epoch.

Example:
    >>> from model.vae_teb_prediction.testing.plot_single_samples import plot_single_samples
    >>> results = plot_single_samples(
    ...     config_path="model/vae_teb_prediction/config.yaml",
    ...     n_samples=5,
    ... )
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime

import numpy as np
import torch
from loguru import logger
import yaml

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.testing.metrics import (
    aggregate_predictions,
    compute_kld,
    compute_kld_per_sample,
    compute_reconstruction_metrics,
    reduce_latent_dimensionality,
)
from model.vae_teb_prediction.testing.visualizers import (
    plot_reconstruction_sample,
    plot_coherence_signals,
    plot_time_frequency_coherence,
    plot_latent_trajectory_2d,
    plot_latent_trajectory_3d,
    plot_psd_comparison,
    plot_cross_correlation,
)
from model.vae_teb_prediction.testing.visualizers_interactive import (
    plot_reconstruction_interactive,
    plot_latent_trajectory_3d_interactive,
    HAS_PLOTLY,
)
from model.vae_teb_prediction.testing.analyses.coherence import (
    compute_stft_coherence_map,
    compute_wavelet_coherence,
    compute_welch_psd,
    compute_cross_correlation,
)

# Import data loading utilities
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

# Import legacy plot utils for detailed analysis plots
from utils.plot_utils import (
    plot_model_analysis,
    plot_vae_reconstruction,
    plot_single_prediction_windows,
)


def _load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}


def _resolve_settings(
    config_path: Union[str, Path],
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve settings from config and explicit arguments."""
    config = _load_config(config_path)
    model_cfg = config.get("model_config", {}) or {}
    dataset_cfg = config.get("dataset_config", {}) or {}
    dataloader_cfg = dataset_cfg.get("dataloader_config", {}) or {}
    folders_cfg = config.get("general_config", {}).get("folders_config", {}) or {}

    # Resolve checkpoint
    resolved_checkpoint = checkpoint_path or model_cfg.get("core_model_checkpoint")
    if not resolved_checkpoint:
        raise ValueError(
            "checkpoint_path is required unless config provides model_config.core_model_checkpoint."
        )

    # Resolve data paths
    resolved_data_paths: List[str] = []
    if data_path:
        if isinstance(data_path, str):
            resolved_data_paths = [data_path]
        else:
            resolved_data_paths = list(data_path)
    else:
        resolved_data_paths = list(dataset_cfg.get("vae_test_datasets", []) or [])

    if not resolved_data_paths:
        raise ValueError(
            "data_path is required unless config provides dataset_config.vae_test_datasets."
        )

    # Resolve output directory
    resolved_output = output_dir
    if resolved_output is None:
        base_dir = folders_cfg.get("out_dir_base")
        if base_dir:
            now = datetime.now()
            run_date = now.strftime("%Y-%m-%d--[%H-%M-%S]") + f"--{now.microsecond:06d}-"
            experiment_tag = config.get("general_config", {}).get("tag", "test")
            tag_dir = Path(base_dir) / experiment_tag
            timestamped_dir = tag_dir / run_date
            resolved_output = str(timestamped_dir / "single_sample_plots")
        else:
            resolved_output = "single_sample_plots"

    # Resolve other settings
    resolved_stats = stats_path or dataset_cfg.get("stat_path")
    resolved_batch_size = batch_size or config.get("general_config", {}).get("batch_size", {}).get("test", 32)
    resolved_workers = num_workers if num_workers is not None else dataloader_cfg.get("num_workers", 0)
    resolved_normalize_fields = normalize_fields or dataloader_cfg.get("normalize_fields")
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}

    return {
        "checkpoint_path": resolved_checkpoint,
        "data_paths": resolved_data_paths,
        "output_dir": Path(resolved_output),
        "stats_path": resolved_stats,
        "batch_size": resolved_batch_size,
        "num_workers": resolved_workers,
        "normalize_fields": resolved_normalize_fields,
        "dataset_kwargs": dataset_kwargs,
        "config": config,
    }


def _get_normalization_stats(loader: Any) -> Optional[Dict[str, Any]]:
    """Get normalization stats from dataset."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None or not hasattr(dataset, "get_normalization_stats"):
        return None
    try:
        return dataset.get_normalization_stats()
    except Exception as exc:
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
    """Denormalize a tensor using statistics."""
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
    except Exception as exc:
        logger.warning("Failed to denormalize %s: %s. Returning tensor as-is.", field, exc)
        return tensor


def _sanitize_folder_name(guid: str, epoch: Optional[float]) -> str:
    """Create a sanitized folder name from guid and epoch."""
    guid_safe = str(guid).replace("/", "_").replace("\\", "_").replace(":", "_")
    if epoch is not None:
        epoch_minutes = abs(epoch / 60.0)
        folder_name = f"{guid_safe}_epoch_{epoch_minutes:.1f}min"
    else:
        folder_name = f"{guid_safe}_epoch_unknown"
    return folder_name


def _extract_reconstruction_features(
    linear_output: Optional[torch.Tensor],
    st_channels: int,
    ph_channels: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract scattering and phase harmonic reconstructions from linear output."""
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


def _plot_all_single_sample_plots(
    runner: TestRunner,
    batch: Any,
    idx: int,
    outputs: Dict[str, Any],
    sample_dir: Path,
    stats: Optional[Dict[str, Any]],
    fs: float = 4.0,
    beta: float = 1.0,
    skip_interactive: bool = False,
    dim_reduction_method: str = "pca",
) -> Dict[str, Any]:
    """
    Plot all available single-sample plots for a given sample.

    Args:
        runner: TestRunner instance.
        batch: Current batch from dataloader.
        idx: Index within the batch.
        outputs: Model outputs dict.
        sample_dir: Directory to save plots.
        stats: Normalization statistics.
        fs: Sampling frequency in Hz.
        beta: Beta value for loss computation.
        skip_interactive: Whether to skip Plotly interactive plots.
        dim_reduction_method: Method for latent dimensionality reduction.

    Returns:
        Dict with paths to generated plots.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, Any] = {}

    # Extract sample data
    y_st = batch.fhr_st[idx : idx + 1]
    y_ph = batch.fhr_ph[idx : idx + 1]
    x_ph = batch.fhr_up_ph[idx : idx + 1]
    y_raw = batch.fhr[idx : idx + 1]
    up_raw_tensor = getattr(batch, "up", None)
    up_raw = up_raw_tensor[idx : idx + 1] if up_raw_tensor is not None else torch.zeros_like(y_raw)

    mu_pr = outputs.get("mu_pr")
    logvar_pr = outputs.get("logvar_pr")
    latent = outputs.get("z")
    linear_output = outputs.get("linear_output")

    if mu_pr is None or latent is None:
        logger.warning("Missing prediction outputs; skipping sample.")
        return plot_paths

    # Aggregate predictions
    avg_mu, valid_mask = aggregate_predictions(
        runner.model, mu_pr[idx], raw_len=y_raw.size(1)
    )
    if avg_mu is None:
        logger.warning("Aggregated predictions missing; skipping sample.")
        return plot_paths

    # Aggregate variance if available
    avg_std = None
    if logvar_pr is not None:
        logvar_seg = logvar_pr[idx]
        if logvar_seg.dim() == 2:
            logvar_seg = logvar_seg.unsqueeze(0)
        if logvar_seg.dim() == 3:
            avg_var, _ = aggregate_predictions(runner.model, logvar_seg.exp(), raw_len=y_raw.size(1))
            if avg_var is not None:
                avg_std = torch.sqrt(avg_var.clamp_min(1e-12))

    # Compute metrics
    metrics = compute_reconstruction_metrics(y_raw, avg_mu.unsqueeze(0), valid_mask.unsqueeze(0) if valid_mask is not None else None)
    kld = compute_kld_per_sample(outputs, runner.warmup_steps)

    # Prepare numpy arrays
    y_true_np = y_raw[0].detach().cpu().numpy()
    y_pred_np = avg_mu.detach().cpu().numpy() if avg_mu.dim() == 1 else avg_mu[0].detach().cpu().numpy()
    y_pred_std_np = avg_std[0].detach().cpu().numpy() if avg_std is not None else None
    latent_np = latent[idx].detach().cpu().numpy()

    guid = _extract_guid(batch, idx)
    epoch = _extract_epoch(batch, idx)
    label = _extract_label(batch, idx)

    # Sample dict for reconstruction plots
    sample_dict = {
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "y_pred_std": y_pred_std_np,
        "latent": latent_np,
        "guid": guid,
        "epoch": epoch,
        "label": label,
        "metrics": {
            "vaf": float(metrics["vaf"][0].cpu().item()) if metrics else np.nan,
            "mse": float(metrics["mse"][0].cpu().item()) if metrics else np.nan,
            "snr": float(metrics["snr"][0].cpu().item()) if metrics else np.nan,
            "kld": float(kld[idx].cpu().item()) if kld is not None else np.nan,
        },
    }

    # -------------------------------------------------------------------
    # 1. Basic reconstruction sample plot
    # -------------------------------------------------------------------
    try:
        plot_path = sample_dir / "reconstruction_sample.svg"
        plot_reconstruction_sample(sample_dict, plot_path, fs=fs)
        plot_paths["reconstruction_sample"] = str(plot_path)
        logger.debug("Saved reconstruction_sample.svg")
    except Exception as e:
        logger.warning(f"Failed to plot reconstruction_sample: {e}")

    # -------------------------------------------------------------------
    # 2. Interactive reconstruction plot (Plotly)
    # -------------------------------------------------------------------
    if not skip_interactive and HAS_PLOTLY:
        try:
            plot_path = sample_dir / "reconstruction_interactive.html"
            plot_reconstruction_interactive(sample_dict, plot_path)
            plot_paths["reconstruction_interactive"] = str(plot_path)
            logger.debug("Saved reconstruction_interactive.html")
        except Exception as e:
            logger.warning(f"Failed to plot reconstruction_interactive: {e}")

    # -------------------------------------------------------------------
    # 3. Coherence signals plot
    # -------------------------------------------------------------------
    try:
        fhr_denorm = _denormalize_tensor(y_raw, "fhr", stats)
        up_denorm = _denormalize_tensor(up_raw, "up", stats)
        fhr_orig_np = fhr_denorm[0].detach().cpu().numpy()
        up_np = up_denorm[0].detach().cpu().numpy()

        # Denormalize prediction
        pred_denorm = _denormalize_tensor(avg_mu.unsqueeze(0), "fhr", stats)
        fhr_pred_np = pred_denorm[0].detach().cpu().numpy()

        plot_path = sample_dir / "coherence_signals.svg"
        plot_coherence_signals(
            up_signal=up_np if np.any(up_np != 0) else None,
            fhr_original=fhr_orig_np,
            fhr_reconstructed=fhr_pred_np,
            output_path=plot_path,
            fs=fs,
            title=f"Signals - GUID: {guid}",
        )
        plot_paths["coherence_signals"] = str(plot_path)
        logger.debug("Saved coherence_signals.svg")
    except Exception as e:
        logger.warning(f"Failed to plot coherence_signals: {e}")

    # -------------------------------------------------------------------
    # 4. Time-frequency coherence (STFT)
    # -------------------------------------------------------------------
    try:
        stft_result = compute_stft_coherence_map(
            y_true_np, y_pred_np, fs=fs, nperseg=64
        )
        if stft_result["coherence"].size > 0:
            plot_path = sample_dir / "time_frequency_coherence_stft.svg"
            plot_time_frequency_coherence(
                stft_result["frequencies"],
                stft_result["times"],
                stft_result["coherence"],
                plot_path,
                max_freq=0.5,
                title=f"STFT Coherence - GUID: {guid}",
            )
            plot_paths["time_frequency_coherence_stft"] = str(plot_path)
            logger.debug("Saved time_frequency_coherence_stft.svg")
    except Exception as e:
        logger.warning(f"Failed to plot time_frequency_coherence_stft: {e}")

    # -------------------------------------------------------------------
    # 5. Time-frequency coherence (Wavelet) - optional
    # -------------------------------------------------------------------
    try:
        wavelet_result = compute_wavelet_coherence(
            y_true_np, y_pred_np, fs=fs, num_scales=40
        )
        if wavelet_result["coherence"].size > 0:
            plot_path = sample_dir / "time_frequency_coherence_wavelet.svg"
            plot_time_frequency_coherence(
                wavelet_result["frequencies"],
                wavelet_result["times"],
                wavelet_result["coherence"],
                plot_path,
                max_freq=0.5,
                title=f"Wavelet Coherence - GUID: {guid}",
            )
            plot_paths["time_frequency_coherence_wavelet"] = str(plot_path)
            logger.debug("Saved time_frequency_coherence_wavelet.svg")
    except ImportError:
        logger.debug("PyWavelets not available, skipping wavelet coherence plot.")
    except Exception as e:
        logger.warning(f"Failed to plot time_frequency_coherence_wavelet: {e}")

    # -------------------------------------------------------------------
    # 6. PSD comparison
    # -------------------------------------------------------------------
    try:
        freqs_orig, psd_orig = compute_welch_psd(y_true_np, fs=fs, nperseg=256)
        _, psd_pred = compute_welch_psd(y_pred_np, fs=fs, nperseg=256)

        if freqs_orig.size > 0 and psd_orig.size > 0:
            plot_path = sample_dir / "psd_comparison.svg"
            plot_psd_comparison(
                freqs_orig,
                psd_orig.reshape(1, -1),  # shape (1, F) for mean/std calc
                psd_pred.reshape(1, -1),
                sample_dir,
                filename="psd_comparison.svg",
            )
            plot_paths["psd_comparison"] = str(plot_path)
            logger.debug("Saved psd_comparison.svg")
    except Exception as e:
        logger.warning(f"Failed to plot psd_comparison: {e}")

    # -------------------------------------------------------------------
    # 7. Cross-correlation
    # -------------------------------------------------------------------
    try:
        lags, corr = compute_cross_correlation(y_true_np, y_pred_np, fs=fs, max_lag_sec=60.0)
        if lags.size > 0:
            plot_path = sample_dir / "cross_correlation.svg"
            plot_cross_correlation(
                lags,
                corr,
                np.zeros_like(corr),  # no std for single sample
                sample_dir,
                filename="cross_correlation.svg",
            )
            plot_paths["cross_correlation"] = str(plot_path)
            logger.debug("Saved cross_correlation.svg")
    except Exception as e:
        logger.warning(f"Failed to plot cross_correlation: {e}")

    # -------------------------------------------------------------------
    # 8. Latent trajectory 2D
    # -------------------------------------------------------------------
    try:
        if latent_np.shape[0] > 2 and latent_np.shape[1] >= 2:
            # Reduce to 2D
            latent_2d = reduce_latent_dimensionality(
                latent_np, method=dim_reduction_method, n_components=2
            )
            plot_path = sample_dir / "latent_trajectory_2d.svg"
            plot_latent_trajectory_2d(
                latent_2d,
                plot_path,
                sample_id=str(guid),
                color_by_time=True,
            )
            plot_paths["latent_trajectory_2d"] = str(plot_path)
            logger.debug("Saved latent_trajectory_2d.svg")
    except Exception as e:
        logger.warning(f"Failed to plot latent_trajectory_2d: {e}")

    # -------------------------------------------------------------------
    # 9. Latent trajectory 3D
    # -------------------------------------------------------------------
    try:
        if latent_np.shape[0] > 2 and latent_np.shape[1] >= 3:
            # Reduce to 3D
            latent_3d = reduce_latent_dimensionality(
                latent_np, method=dim_reduction_method, n_components=3
            )
            plot_path = sample_dir / "latent_trajectory_3d.svg"
            plot_latent_trajectory_3d(
                latent_3d,
                plot_path,
                sample_id=str(guid),
                color_by_time=True,
            )
            plot_paths["latent_trajectory_3d"] = str(plot_path)
            logger.debug("Saved latent_trajectory_3d.svg")

            # Interactive 3D
            if not skip_interactive and HAS_PLOTLY:
                plot_path = sample_dir / "latent_trajectory_3d_interactive.html"
                plot_latent_trajectory_3d_interactive(
                    latent_3d,
                    plot_path,
                    sample_id=str(guid),
                    color_by_time=True,
                )
                plot_paths["latent_trajectory_3d_interactive"] = str(plot_path)
                logger.debug("Saved latent_trajectory_3d_interactive.html")
    except Exception as e:
        logger.warning(f"Failed to plot latent_trajectory_3d: {e}")

    # -------------------------------------------------------------------
    # Prepare common data for detailed plots (model_analysis, vae_reconstruction)
    # -------------------------------------------------------------------
    # Denormalize signals
    fhr_denorm_full = _denormalize_tensor(y_raw, "fhr", stats)
    up_denorm_full = _denormalize_tensor(up_raw, "up", stats)

    raw_fhr_norm_np = y_raw[0].detach().cpu().numpy()
    raw_up_norm_np = up_raw[0].detach().cpu().numpy()
    raw_fhr_denorm_np = fhr_denorm_full[0].detach().cpu().numpy()
    raw_up_denorm_np = up_denorm_full[0].detach().cpu().numpy()

    fhr_st_np = y_st[0].detach().cpu().numpy().T
    fhr_ph_np = y_ph[0].detach().cpu().numpy().T
    fhr_up_ph_np = x_ph[0].detach().cpu().numpy().T

    # Compute KLD tensor
    kld_tensor = compute_kld(outputs, runner.warmup_steps)
    kld_tensor_np = None
    kld_mean_np = None
    if kld_tensor is not None:
        kld_sample = kld_tensor[idx]
        kld_tensor_np = kld_sample.detach().cpu().numpy().T
        kld_mean_np = torch.nanmean(kld_sample, dim=-1).detach().cpu().numpy()

    # Compute loss for annotations
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
    except Exception as exc:
        logger.warning("Loss computation failed: %s", exc)

    loss_floats: Dict[str, float] = {}
    if loss_dict:
        for key, val in loss_dict.items():
            if torch.is_tensor(val):
                loss_floats[key] = float(torch.nan_to_num(val.detach()).cpu().item())
            else:
                loss_floats[key] = float(val)

    # Denormalize prediction for detailed plots
    pred_denorm_full = _denormalize_tensor(avg_mu.unsqueeze(0), "fhr", stats)
    recon_np = pred_denorm_full[0].detach().cpu().numpy()

    # Logvar for detailed plots
    logvar_np = None
    if logvar_pr is not None:
        avg_logvar_seg = logvar_pr[idx]
        if avg_logvar_seg.dim() == 2:
            avg_logvar_seg = avg_logvar_seg.unsqueeze(0)
        if avg_logvar_seg.dim() == 3:
            avg_logvar, _ = aggregate_predictions(runner.model, avg_logvar_seg, raw_len=y_raw.size(1))
            if avg_logvar is not None:
                logvar_np = avg_logvar.detach().cpu().numpy()

    # -------------------------------------------------------------------
    # 10. Model analysis (detailed panel)
    # -------------------------------------------------------------------
    try:
        plot_model_analysis(
            output_dir=str(sample_dir),
            raw_fhr=raw_fhr_denorm_np,
            raw_up=raw_up_denorm_np,
            fhr_st=fhr_st_np,
            fhr_ph=fhr_ph_np,
            fhr_up_ph=fhr_up_ph_np,
            latent_z=latent_np.T,
            reconstructed_fhr_mu=recon_np,
            reconstructed_fhr_logvar=logvar_np,
            kld_tensor=kld_tensor_np,
            kld_mean_over_channels=kld_mean_np,
            batch_idx=0,
            loss_dict=loss_floats,
            raw_fhr_normalized=raw_fhr_norm_np,
            raw_up_normalized=raw_up_norm_np,
        )
        plot_paths["model_analysis"] = str(sample_dir / "model_analysis_0.svg")
        logger.debug("Saved model_analysis.svg")
    except Exception as e:
        logger.warning(f"Failed to plot model_analysis: {e}")

    # -------------------------------------------------------------------
    # 11. VAE reconstruction diagnostic
    # -------------------------------------------------------------------
    try:
        recon_st_np, recon_ph_np = _extract_reconstruction_features(
            linear_output[idx : idx + 1] if linear_output is not None else None,
            st_channels=fhr_st_np.shape[0],
            ph_channels=fhr_ph_np.shape[0],
        )

        plot_vae_reconstruction(
            output_dir=str(sample_dir),
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
            batch_idx=0,
            loss_dict=loss_floats,
        )
        plot_paths["vae_reconstruction"] = str(sample_dir / "vae_reconstruction_0.svg")
        logger.debug("Saved vae_reconstruction.svg")
    except Exception as e:
        logger.warning(f"Failed to plot vae_reconstruction: {e}")

    # -------------------------------------------------------------------
    # 12. Single prediction windows
    # -------------------------------------------------------------------
    try:
        mu_pr_sample = mu_pr[idx]
        logvar_pr_sample = logvar_pr[idx] if logvar_pr is not None else None

        if mu_pr_sample.dim() == 2:
            mu_pr_sample = mu_pr_sample.unsqueeze(0)
        if mu_pr_sample.dim() != 3:
            raise ValueError("mu_pr must be 3D for window extraction")

        # Prepare windows
        predictions = mu_pr_sample.squeeze(0)
        logvar_predictions = logvar_pr_sample.squeeze(0) if logvar_pr_sample is not None else None

        horizon = predictions.size(-1)
        stride = runner.decimation_factor
        warmup = runner.warmup_steps

        raw_norm = y_raw[0]
        raw_denorm = _denormalize_tensor(raw_norm.unsqueeze(0), "fhr", stats)[0]
        raw_len = raw_norm.size(0)

        # Extract several windows
        windows: List[Dict[str, Any]] = []
        total_steps = predictions.size(0)
        step_size = max(1, horizon // stride)
        t_idx = warmup
        max_windows = 4

        while t_idx < total_steps and len(windows) < max_windows:
            raw_start = t_idx * stride
            raw_end = raw_start + horizon
            if raw_end > raw_len:
                break

            pred_segment = predictions[t_idx]
            logvar_segment = logvar_predictions[t_idx] if logvar_predictions is not None else None

            target_norm = raw_norm[raw_start:raw_end]
            target_denorm = raw_denorm[raw_start:raw_end]

            pred_denorm_win = _denormalize_tensor(
                pred_segment.unsqueeze(0), "fhr", stats, raw_start=raw_start, length=horizon
            )[0]

            std_denorm_np = None
            std_norm_np = None
            if logvar_segment is not None:
                std_norm = torch.exp(0.5 * logvar_segment)
                std_norm_np = std_norm.detach().cpu().numpy()
                std_denorm = _denormalize_tensor(
                    std_norm.unsqueeze(0), "fhr", stats, raw_start=raw_start, length=horizon
                )[0]
                std_denorm_np = std_denorm.detach().cpu().numpy()

            win_metrics = compute_reconstruction_metrics(
                target_denorm.unsqueeze(0), pred_denorm_win.unsqueeze(0)
            )
            win_metrics_map = (
                {key: float(val.detach().cpu().item()) for key, val in win_metrics.items()}
                if win_metrics else {}
            )

            windows.append({
                "t_index": int(t_idx),
                "raw_start": int(raw_start),
                "raw_end": int(raw_end),
                "prediction": pred_denorm_win.detach().cpu().numpy(),
                "target": target_denorm.detach().cpu().numpy(),
                "prediction_norm": pred_segment.detach().cpu().numpy(),
                "target_norm": target_norm.detach().cpu().numpy(),
                "uncertainty": std_denorm_np,
                "uncertainty_norm": std_norm_np,
                "metrics": win_metrics_map,
            })
            t_idx += step_size

        if windows:
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
                output_dir=str(sample_dir),
                raw_fhr_unnormalized=raw_fhr_denorm_np,
                raw_fhr_normalized=raw_fhr_norm_np,
                windows=windows,
                aggregated_pred_norm=agg_pred_norm,
                aggregated_uncertainty_norm=agg_uncert_norm,
                sample_idx=0,
                sample_guid=guid,
                epoch=epoch,
            )
            plot_paths["single_prediction_windows"] = str(sample_dir / "single_pred_windows_0.svg")
            logger.debug("Saved single_prediction_windows.svg")
    except Exception as e:
        logger.warning(f"Failed to plot single_prediction_windows: {e}")

    return plot_paths


def plot_single_samples(
    config_path: Union[str, Path],
    n_samples: int = 5,
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    skip_interactive: bool = False,
    beta: float = 1.0,
    dim_reduction_method: str = "pca",
    fs: float = 4.0,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot all available single-sample plots for randomly selected samples.

    Args:
        config_path: Path to YAML configuration file.
        n_samples: Number of samples to randomly select and plot.
        checkpoint_path: Path to model checkpoint (optional, uses config if not provided).
        data_path: Path(s) to test data (optional, uses config if not provided).
        output_dir: Output directory (optional, uses config if not provided).
        stats_path: Path to normalization statistics (optional).
        device: Device string ("cuda:0", "cpu"). Auto-detects if None.
        batch_size: Batch size for data loading.
        num_workers: Number of dataloader workers.
        normalize_fields: Fields to normalize.
        seed: Random seed for sample selection.
        skip_interactive: Whether to skip Plotly interactive plots.
        beta: Beta value for loss computation.
        dim_reduction_method: Method for latent dimensionality reduction (pca, umap, tsne).
        fs: Sampling frequency in Hz.
        **model_kwargs: Additional model architecture parameters.

    Returns:
        Dict with results including paths to all generated plots.

    Example:
        >>> results = plot_single_samples(
        ...     config_path="model/vae_teb_prediction/config.yaml",
        ...     n_samples=5,
        ...     seed=42,
        ... )
    """
    # Resolve settings
    settings = _resolve_settings(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        output_dir=output_dir,
        stats_path=stats_path,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_fields=normalize_fields,
    )

    # Auto-detect device
    device_str = device
    if device_str is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_torch = torch.device(device_str)

    logger.info(f"Checkpoint: {settings['checkpoint_path']}")
    logger.info(f"Data: {settings['data_paths']}")
    logger.info(f"Output: {settings['output_dir']}")
    logger.info(f"Device: {device_torch}")
    logger.info(f"Number of samples: {n_samples}")

    # Create TestRunner
    logger.info("Loading model from checkpoint...")
    runner = TestRunner.from_checkpoint(
        checkpoint_path=str(settings["checkpoint_path"]),
        output_dir=str(settings["output_dir"]),
        device=device_torch,
        **model_kwargs,
    )

    # Create DataLoader
    logger.info("Creating test dataloader...")
    resolved_kwargs = settings.get("dataset_kwargs", {}) or {}
    if "pin_memory" not in resolved_kwargs:
        resolved_kwargs["pin_memory"] = True

    loader = create_optimized_dataloader(
        hdf5_files=[str(p) for p in settings["data_paths"]],
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        shuffle=False,
        stats_path=settings["stats_path"],
        normalize_fields=settings["normalize_fields"],
        rank=0,
        world_size=1,
        **resolved_kwargs,
    )

    total_samples = len(loader.dataset)
    logger.info(f"Total samples in dataset: {total_samples}")

    # Get normalization stats
    stats = _get_normalization_stats(loader)

    # Select random sample indices
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_samples = min(n_samples, total_samples)
    selected_indices = sorted(random.sample(range(total_samples), n_samples))
    logger.info(f"Selected sample indices: {selected_indices}")

    # Create output directory
    output_path = Path(settings["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Process selected samples
    results: Dict[str, Any] = {
        "n_samples": n_samples,
        "selected_indices": selected_indices,
        "samples": [],
    }

    processed = 0
    current_batch_start = 0

    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples=None):
            batch_size_actual = batch.fhr_st.size(0)
            batch_end = current_batch_start + batch_size_actual

            # Check if any selected indices are in this batch
            batch_indices = [
                (i, idx - current_batch_start)
                for i, idx in enumerate(selected_indices)
                if current_batch_start <= idx < batch_end
            ]

            if batch_indices:
                # Run forward pass
                outputs = runner.forward(batch)

                for _, idx_in_batch in batch_indices:
                    # Extract sample info
                    guid = _extract_guid(batch, idx_in_batch)
                    epoch = _extract_epoch(batch, idx_in_batch)
                    label = _extract_label(batch, idx_in_batch)

                    # Create folder for this sample
                    folder_name = _sanitize_folder_name(guid or f"sample_{processed}", epoch or 0.0)
                    sample_dir = output_path / folder_name
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Processing sample {processed + 1}/{n_samples}: GUID={guid}, Epoch={epoch}")

                    # Plot all single-sample plots
                    plot_paths = _plot_all_single_sample_plots(
                        runner=runner,
                        batch=batch,
                        idx=idx_in_batch,
                        outputs=outputs,
                        sample_dir=sample_dir,
                        stats=stats,
                        fs=fs,
                        beta=beta,
                        skip_interactive=skip_interactive,
                        dim_reduction_method=dim_reduction_method,
                    )

                    results["samples"].append({
                        "guid": guid,
                        "epoch": epoch,
                        "label": label,
                        "folder": str(sample_dir),
                        "plots": plot_paths,
                    })

                    processed += 1

            current_batch_start = batch_end

            if processed >= n_samples:
                break

    logger.info(f"Completed! Generated plots for {processed} samples in {output_path}")
    return results


# ----- Main entry point -----
if __name__ == "__main__":
    # Default configuration - edit these for your setup
    CONFIG_PATH = "model/vae_teb_prediction/config.yaml"
    N_SAMPLES = 5
    SEED = 42

    # Run the plotting
    results = plot_single_samples(
        config_path=CONFIG_PATH,
        n_samples=N_SAMPLES,
        seed=SEED,
        skip_interactive=False,
        dim_reduction_method="pca",
    )

    # Print summary
    print(f"\n=== Single Sample Plots ===")
    print(f"Samples processed: {results['n_samples']}")
    for sample in results["samples"]:
        print(f"\n  GUID: {sample['guid']}")
        print(f"  Epoch: {sample['epoch']}")
        print(f"  Folder: {sample['folder']}")
        print(f"  Plots: {len(sample['plots'])}")
