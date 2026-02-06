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
)
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from model.vae_teb_prediction.testing.visualizers import (
    # Colors for consistent styling
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_SKY,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    COLOR_GRAY,
    COLOR_BLACK,
    COLOR_LIGHT_GRAY,
    COLOR_SAGE,
    SAVE_DPI,
)

# Import data loading utilities
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

# Import legacy plot utils for detailed analysis plots



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


def _apply_publication_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": SAVE_DPI,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLOR_BLACK,
        "axes.labelcolor": COLOR_BLACK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.color": COLOR_BLACK,
        "ytick.color": COLOR_BLACK,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.3,
        "grid.color": COLOR_LIGHT_GRAY,
        "grid.linestyle": "-",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": COLOR_GRAY,
        "legend.shadow": False,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "errorbar.capsize": 3,
        "mathtext.default": "regular",
    })


def _style_axes(ax: plt.Axes, *, grid: str = "major") -> None:
    """Apply clean styling to axes with all four spines visible."""
    ax.set_axisbelow(True)
    if grid in ("both", "major"):
        ax.grid(True, linestyle="-", alpha=0.4, linewidth=0.4, color=COLOR_LIGHT_GRAY)
    if grid == "both":
        ax.grid(True, which="minor", linestyle=":", alpha=0.25, linewidth=0.3, color=COLOR_LIGHT_GRAY)
        ax.minorticks_on()
    # Show all four spines
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)


def _add_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: plt.Axes,
    *,
    label: Optional[str] = None,
) -> plt.Axes:
    """Attach aligned colorbar."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=8, color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    return cbar


def _plot_coefficient_heatmap(
    coefficients: np.ndarray,
    output_path: Path,
    *,
    title: str,
    ylabel: str = "Channel",
    xlabel: str = "Time Steps",
    cmap: str = "bwr",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fs: float = 4.0,
) -> None:
    """Plot a heatmap of coefficient values (ST, PH, etc.)."""
    _apply_publication_style()

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Auto-scale if not provided
    if vmin is None or vmax is None:
        vabs = np.nanmax(np.abs(coefficients))
        vmin = -vabs
        vmax = vabs

    im = ax.imshow(
        coefficients,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="normal", pad=8)
    ax.grid(False)

    _add_colorbar(fig, im, ax, label="Value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_coefficient_error_heatmap(
    original: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    *,
    title: str,
    ylabel: str = "Channel",
) -> None:
    """Plot reconstruction error heatmap."""
    _apply_publication_style()

    error = np.abs(original - reconstructed)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True)

    # Original
    vabs = np.nanmax(np.abs(original))
    im0 = axes[0].imshow(original, aspect="auto", cmap="bwr", origin="upper", vmin=-vabs, vmax=vabs)
    axes[0].set_title("Original Coefficients", fontsize=9)
    axes[0].set_ylabel(ylabel, fontsize=8)
    axes[0].grid(False)
    _add_colorbar(fig, im0, axes[0], label="Value")

    # Reconstructed
    im1 = axes[1].imshow(reconstructed, aspect="auto", cmap="bwr", origin="upper", vmin=-vabs, vmax=vabs)
    axes[1].set_title("Reconstructed Coefficients", fontsize=9)
    axes[1].set_ylabel(ylabel, fontsize=8)
    axes[1].grid(False)
    _add_colorbar(fig, im1, axes[1], label="Value")

    # Error
    im2 = axes[2].imshow(error, aspect="auto", cmap="Reds", origin="upper")
    axes[2].set_title(f"Absolute Error (MAE: {np.nanmean(error):.4f})", fontsize=9)
    axes[2].set_xlabel("Time Steps", fontsize=8)
    axes[2].set_ylabel(ylabel, fontsize=8)
    axes[2].grid(False)
    _add_colorbar(fig, im2, axes[2], label="|Error|")

    fig.suptitle(title, fontsize=10, fontweight="normal", y=1.01)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_heatmap(
    latent: np.ndarray,
    output_path: Path,
    *,
    title: str = "Latent Space z(t)",
    warmup_steps: int = 0,
) -> None:
    """Plot latent space heatmap with optional warmup marking."""
    _apply_publication_style()

    fig, ax = plt.subplots(figsize=(8, 3))

    # latent shape: (T, D) - transpose to (D, T) for display
    latent_T = latent.T if latent.ndim == 2 else latent

    vabs = np.nanmax(np.abs(latent_T))
    im = ax.imshow(latent_T, aspect="auto", cmap="bwr", origin="lower", vmin=-vabs, vmax=vabs)

    # Mark warmup boundary
    if warmup_steps > 0:
        ax.axvline(x=warmup_steps - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.text(warmup_steps + 1, latent_T.shape[0] * 0.95, "Warmup end", color="white", fontsize=7, va="top")

    ax.set_xlabel("Time Steps", fontsize=8)
    ax.set_ylabel("Latent Dimension", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="normal", pad=8)
    ax.grid(False)

    _add_colorbar(fig, im, ax, label="Activation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_kld_per_dimension(
    kld_tensor: np.ndarray,
    output_path: Path,
    *,
    title: str = "KLD per Latent Dimension",
    warmup_steps: int = 0,
) -> None:
    """Plot KLD heatmap and mean trace per latent dimension."""
    _apply_publication_style()

    # kld_tensor shape: (T, D) or (D, T) - ensure (D, T)
    if kld_tensor.ndim != 2:
        return

    # Assume (D, T) format based on typical usage
    kld_T = kld_tensor if kld_tensor.shape[0] < kld_tensor.shape[1] else kld_tensor.T

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), gridspec_kw={"height_ratios": [2, 1]})

    # Heatmap
    im = axes[0].imshow(kld_T, aspect="auto", cmap="viridis", origin="lower")
    if warmup_steps > 0:
        axes[0].axvline(x=warmup_steps - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Latent Dimension", fontsize=8)
    axes[0].set_title("KLD per Dimension over Time", fontsize=9)
    axes[0].grid(False)
    _add_colorbar(fig, im, axes[0], label="KLD (bits)")

    # Mean trace
    if kld_T.size == 0:
        return

    finite_mask = np.isfinite(kld_T)
    if np.any(finite_mask):
        kld_mean = np.nanmean(kld_T, axis=0)
        overall_mean = float(np.nanmean(kld_mean))
    else:
        kld_mean = np.full(kld_T.shape[1], np.nan, dtype=float)
        overall_mean = 0.0

    t = np.arange(len(kld_mean))
    axes[1].plot(t, kld_mean, color=COLOR_PURPLE, linewidth=0.8)
    if warmup_steps > 0:
        axes[1].axvspan(0, warmup_steps, alpha=0.1, color=COLOR_GRAY)
        axes[1].axvline(x=warmup_steps, color=COLOR_GRAY, linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Time Steps", fontsize=8)
    axes[1].set_ylabel("Mean KLD", fontsize=8)
    axes[1].set_title(f"Mean KLD over Time (Overall: {overall_mean:.4f})", fontsize=9)
    _style_axes(axes[1], grid="both")
    axes[1].set_xlim(0, len(kld_mean))

    fig.suptitle(title, fontsize=10, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    *,
    title: str = "Residual Distribution",
) -> None:
    """Plot histogram of reconstruction residuals."""
    _apply_publication_style()

    residual = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Histogram
    n_bins = min(100, max(30, int(np.sqrt(len(residual)) * 2)))
    counts, bins, patches = axes[0].hist(
        residual, bins=n_bins, density=True,
        color=COLOR_GREEN, alpha=0.7, edgecolor=COLOR_BLACK, linewidth=0.3
    )

    # Fit normal distribution
    mean_val = np.mean(residual)
    std_val = np.std(residual)
    x_fit = np.linspace(bins[0], bins[-1], 200)
    y_fit = scipy_stats.norm.pdf(x_fit, mean_val, std_val)
    axes[0].plot(x_fit, y_fit, color=COLOR_VERMILLION, linewidth=1.2, label=f"N({mean_val:.3f}, {std_val:.3f})")

    # Reference lines
    axes[0].axvline(mean_val, color=COLOR_ORANGE, linewidth=0.9, linestyle="--", label=f"Mean: {mean_val:.4f}")
    axes[0].axvline(0, color=COLOR_BLACK, linewidth=0.6, alpha=0.5)

    axes[0].set_xlabel("Residual (True - Pred)", fontsize=8)
    axes[0].set_ylabel("Density", fontsize=8)
    axes[0].set_title("Residual Histogram", fontsize=9)
    axes[0].legend(fontsize=7, framealpha=0.95)
    _style_axes(axes[0], grid="major")

    # Q-Q plot
    scipy_stats.probplot(residual, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normal)", fontsize=9)
    axes[1].get_lines()[0].set_markersize(2)
    axes[1].get_lines()[0].set_color(COLOR_BLUE)
    axes[1].get_lines()[1].set_color(COLOR_VERMILLION)
    _style_axes(axes[1], grid="major")

    # Stats box
    skewness = scipy_stats.skew(residual)
    kurtosis = scipy_stats.kurtosis(residual)
    stats_text = f"Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}\nRMSE: {np.sqrt(np.mean(residual**2)):.4f}"
    axes[0].text(
        0.98, 0.98, stats_text,
        transform=axes[0].transAxes, fontsize=7,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLOR_LIGHT_GRAY, alpha=0.95)
    )

    fig.suptitle(title, fontsize=10, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_channel_timeseries(
    coefficients: np.ndarray,
    output_path: Path,
    *,
    title: str,
    n_channels: int = 8,
    reconstructed: Optional[np.ndarray] = None,
    fs: float = 4.0,
) -> None:
    """Plot individual channel time series (first n_channels)."""
    _apply_publication_style()

    n_ch = min(n_channels, coefficients.shape[0])
    n_rows = (n_ch + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2 * n_rows), sharex=True)
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes

    t = np.arange(coefficients.shape[1])

    for i in range(n_ch):
        ax = axes[i]
        ax.plot(t, coefficients[i], color=COLOR_BLUE, linewidth=0.6, label="Original")
        if reconstructed is not None and i < reconstructed.shape[0]:
            ax.plot(t, reconstructed[i], color=COLOR_ORANGE, linewidth=0.8, alpha=0.8, label="Reconstructed")
            mae = np.mean(np.abs(coefficients[i] - reconstructed[i]))
            ax.set_title(f"Channel {i} (MAE: {mae:.4f})", fontsize=8)
        else:
            ax.set_title(f"Channel {i}", fontsize=8)
        ax.set_ylabel("Value", fontsize=7)
        _style_axes(ax, grid="major")
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.9)

    # Hide unused axes
    for i in range(n_ch, len(axes)):
        axes[i].set_visible(False)

    axes[-2].set_xlabel("Time Steps", fontsize=8)
    if len(axes) > 1:
        axes[-1].set_xlabel("Time Steps", fontsize=8)

    fig.suptitle(title, fontsize=10, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_error_summary(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    fhr_st_orig: np.ndarray,
    fhr_st_recon: Optional[np.ndarray],
    fhr_ph_orig: np.ndarray,
    fhr_ph_recon: Optional[np.ndarray],
    output_path: Path,
    *,
    title: str = "Reconstruction Error Summary",
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Plot comprehensive error summary with bar charts and statistics."""
    _apply_publication_style()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # 1. FHR reconstruction metrics bar chart
    ax = axes[0, 0]
    if metrics:
        metric_names = ["VAF", "MSE", "SNR", "KLD"]
        metric_values = [
            metrics.get("vaf", np.nan),
            metrics.get("mse", np.nan),
            metrics.get("snr", np.nan),
            metrics.get("kld", np.nan),
        ]
        colors = [COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_PURPLE]
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor=COLOR_BLACK, linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel("Value", fontsize=8)
        ax.set_title("FHR Reconstruction Metrics", fontsize=9)
        _style_axes(ax, grid="major")

    # 2. Per-channel ST MAE distribution
    ax = axes[0, 1]
    if fhr_st_recon is not None:
        st_mae_per_channel = np.mean(np.abs(fhr_st_orig - fhr_st_recon), axis=1)
        ax.bar(range(len(st_mae_per_channel)), st_mae_per_channel, color=COLOR_SKY, alpha=0.8,
               edgecolor=COLOR_BLACK, linewidth=0.3)
        ax.axhline(np.mean(st_mae_per_channel), color=COLOR_VERMILLION, linestyle="--",
                   linewidth=1.0, label=f"Mean: {np.mean(st_mae_per_channel):.4f}")
        ax.set_xlabel("ST Channel", fontsize=8)
        ax.set_ylabel("MAE", fontsize=8)
        ax.set_title("Scattering Transform Error by Channel", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9)
        _style_axes(ax, grid="major")
    else:
        ax.text(0.5, 0.5, "No ST reconstruction available", ha="center", va="center", fontsize=8)
        ax.set_title("Scattering Transform Error by Channel", fontsize=9)

    # 3. Per-channel PH MAE distribution
    ax = axes[1, 0]
    if fhr_ph_recon is not None:
        ph_mae_per_channel = np.mean(np.abs(fhr_ph_orig - fhr_ph_recon), axis=1)
        ax.bar(range(len(ph_mae_per_channel)), ph_mae_per_channel, color=COLOR_SAGE, alpha=0.8,
               edgecolor=COLOR_BLACK, linewidth=0.3)
        ax.axhline(np.mean(ph_mae_per_channel), color=COLOR_VERMILLION, linestyle="--",
                   linewidth=1.0, label=f"Mean: {np.mean(ph_mae_per_channel):.4f}")
        ax.set_xlabel("PH Channel", fontsize=8)
        ax.set_ylabel("MAE", fontsize=8)
        ax.set_title("Phase Harmonic Error by Channel", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.9)
        _style_axes(ax, grid="major")
    else:
        ax.text(0.5, 0.5, "No PH reconstruction available", ha="center", va="center", fontsize=8)
        ax.set_title("Phase Harmonic Error by Channel", fontsize=9)

    # 4. FHR error over time
    ax = axes[1, 1]
    residual = y_true_np - y_pred_np
    window_size = max(1, len(residual) // 50)
    # Compute rolling MAE
    rolling_mae = np.array([
        np.mean(np.abs(residual[max(0, i - window_size):i + window_size + 1]))
        for i in range(len(residual))
    ])
    t = np.arange(len(residual))
    ax.plot(t, rolling_mae, color=COLOR_PURPLE, linewidth=0.6, label="Rolling MAE")
    ax.axhline(np.mean(np.abs(residual)), color=COLOR_ORANGE, linestyle="--",
               linewidth=0.9, label=f"Mean MAE: {np.mean(np.abs(residual)):.4f}")
    ax.set_xlabel("Sample Index", fontsize=8)
    ax.set_ylabel("MAE", fontsize=8)
    ax.set_title("FHR Reconstruction Error Over Time", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.set_xlim(0, len(residual))
    _style_axes(ax, grid="major")

    fig.suptitle(title, fontsize=10, fontweight="normal", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_all_single_sample_plots(
    runner: TestRunner,
    batch: Any,
    idx: int,
    outputs: Dict[str, Any],
    sample_dir: Path,
    sample_name: str,
    stats: Optional[Dict[str, Any]],
    fs: float = 4.0,
    beta: float = 1.0,
    skip_interactive: bool = False,
    dim_reduction_method: str = "pca",
) -> Dict[str, Any]:
    """
    Plot a single consolidated summary figure for a sample.

    Args:
        sample_dir: Output directory (root folder).
        sample_name: Unique identifier for the sample (used in filename).

    Returns:
        Dict with paths to generated plots.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: Dict[str, Any] = {}

    y_st = batch.fhr_st[idx : idx + 1]
    y_ph = batch.fhr_ph[idx : idx + 1]
    x_ph = batch.fhr_up_ph[idx : idx + 1]
    y_raw = batch.fhr[idx : idx + 1]
    up_raw_tensor = getattr(batch, "up", None)
    up_raw = up_raw_tensor[idx : idx + 1] if up_raw_tensor is not None else torch.zeros_like(y_raw)

    mu_pr = outputs.get("mu_pr")
    logvar_pr = outputs.get("logvar_pr")
    latent = outputs.get("z")

    if mu_pr is None or latent is None:
        logger.warning("Missing prediction outputs; skipping sample.")
        return plot_paths

    avg_mu, valid_mask = aggregate_predictions(
        runner.model, mu_pr[idx : idx + 1], raw_len=y_raw.size(1)
    )
    if avg_mu is None:
        logger.warning("Aggregated predictions missing; skipping sample.")
        return plot_paths

    avg_mu = avg_mu[0]
    valid_mask_np = None
    if valid_mask is not None:
        valid_mask_np = valid_mask[0].detach().cpu().numpy().astype(bool)

    kld_tensor = compute_kld(outputs, runner.warmup_steps)
    kld_mean_np = None
    kld_std_np = None
    kld_sample_np = None
    if kld_tensor is not None:
        kld_sample = kld_tensor[idx]
        kld_mean_np = torch.nanmean(kld_sample, dim=-1).detach().cpu().numpy()
        kld_std_np = kld_sample.detach().cpu().numpy()
        kld_std_np = np.nanstd(kld_std_np, axis=-1)  # (T,)
        kld_sample_np = kld_sample.detach().cpu().numpy()  # (T, D)

    fhr_st_np = y_st[0].detach().cpu().numpy().T
    fhr_ph_np = y_ph[0].detach().cpu().numpy().T
    fhr_up_ph_np = x_ph[0].detach().cpu().numpy().T
    latent_np = latent[idx].detach().cpu().numpy().T

    raw_norm_np = y_raw[0].detach().cpu().numpy().reshape(-1)
    raw_denorm_np = _denormalize_tensor(y_raw, "fhr", stats)[0].detach().cpu().numpy().reshape(-1)
    up_denorm_np = _denormalize_tensor(up_raw, "up", stats)[0].detach().cpu().numpy().reshape(-1)

    avg_pred_np = avg_mu.detach().cpu().numpy().reshape(-1)
    if valid_mask_np is not None:
        avg_pred_np = np.where(valid_mask_np, avg_pred_np, np.nan)

    pred_concat = np.full_like(raw_norm_np, np.nan, dtype=float)
    std_concat = None
    predictions = mu_pr[idx]
    logvar_predictions = logvar_pr[idx] if logvar_pr is not None else None
    if predictions.dim() == 3:
        predictions = predictions.squeeze(0)
    if logvar_predictions is not None and logvar_predictions.dim() == 3:
        logvar_predictions = logvar_predictions.squeeze(0)

    horizon = predictions.size(-1)
    stride = runner.decimation_factor
    warmup = runner.warmup_steps
    raw_len = raw_norm_np.shape[0]
    total_steps = predictions.size(0)
    step_size = max(1, horizon // max(1, stride))

    if logvar_predictions is not None:
        std_concat = np.full_like(raw_norm_np, np.nan, dtype=float)

    # Track prediction window boundaries for visualization
    prediction_boundaries = []

    t_idx = warmup
    while t_idx < total_steps:
        raw_start = t_idx * stride
        if raw_start >= raw_len:
            break
        raw_end = raw_start + horizon

        pred_segment = predictions[t_idx].detach().cpu().numpy()
        seg_len = raw_end - raw_start
        if raw_end > raw_len:
            seg_len = raw_len - raw_start
            raw_end = raw_len
            pred_segment = pred_segment[:seg_len]

        pred_concat[raw_start:raw_end] = pred_segment
        prediction_boundaries.append(raw_start)

        if std_concat is not None:
            logvar_segment = logvar_predictions[t_idx]
            std_segment = torch.exp(0.5 * logvar_segment).detach().cpu().numpy()
            if std_segment.shape[0] > seg_len:
                std_segment = std_segment[:seg_len]
            std_concat[raw_start:raw_end] = std_segment

        t_idx += step_size

    _apply_publication_style()
    fig, axes = plt.subplots(9, 1, figsize=(16, 27), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    def _style_heatmap(ax: plt.Axes) -> None:
        _style_axes(ax, grid="none")
        ax.grid(False)

    t_raw = np.arange(raw_norm_np.shape[0]) / fs

    ax = axes[0]
    ax.plot(t_raw, raw_denorm_np, color=COLOR_BLUE, linewidth=1.2, label="FHR")
    ax.plot(t_raw, up_denorm_np, color=COLOR_GREEN, linewidth=1.2, label="UP")
    ax.set_title("Original FHR and UP")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(t_raw[0], t_raw[-1])
    ax.margins(x=0.0)
    _style_axes(ax, grid="both")

    ax = axes[1]
    im = ax.imshow(fhr_st_np, aspect="auto", cmap="bwr", origin="upper")
    ax.set_title("FHR Scattering Transform")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("ST Channel")
    _style_heatmap(ax)
    _add_colorbar(fig, im, ax, label="Coeff")

    ax = axes[2]
    im = ax.imshow(fhr_ph_np, aspect="auto", cmap="bwr", origin="upper")
    ax.set_title("FHR Phase Harmonics")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("PH Channel")
    _style_heatmap(ax)
    _add_colorbar(fig, im, ax, label="Coeff")

    ax = axes[3]
    im = ax.imshow(fhr_up_ph_np, aspect="auto", cmap="bwr", origin="upper")
    ax.set_title("FHR-UP Cross-Channel Phase Harmonics")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Cross-PH Channel")
    _style_heatmap(ax)
    _add_colorbar(fig, im, ax, label="Coeff")

    ax = axes[4]
    im = ax.imshow(latent_np, aspect="auto", cmap="bwr", origin="lower")
    # Add warmup boundary
    if warmup > 0:
        ax.axvline(x=warmup - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title("Latent Representation")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Latent Dim")
    _style_heatmap(ax)
    _add_colorbar(fig, im, ax, label="Activation")

    ax = axes[5]
    if kld_mean_np is None:
        ax.text(0.5, 0.5, "KLD unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        t_kld = np.arange(len(kld_mean_np))
        ax.plot(t_kld, kld_mean_np, color=COLOR_PURPLE, linewidth=1.1, label="Mean KLD")
        # Add warmup shading
        if warmup > 0:
            ax.axvspan(0, warmup, alpha=0.15, color=COLOR_GRAY, label="Warmup")
            ax.axvline(x=warmup, color=COLOR_GRAY, linestyle="--", linewidth=0.8)
        overall_mean = float(np.nanmean(kld_mean_np[warmup:]))
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Mean KLD", color=COLOR_PURPLE)
        ax.tick_params(axis="y", labelcolor=COLOR_PURPLE)
        ax.set_xlim(0, len(kld_mean_np))
        _style_axes(ax, grid="both")

        # Second y-axis for KLD std across dimensions
        if kld_std_np is not None:
            ax2 = ax.twinx()
            ax2.plot(t_kld, kld_std_np, color=COLOR_ORANGE, linewidth=0.9, alpha=0.85, label="Std KLD")
            ax2.set_ylabel("Std KLD", color=COLOR_ORANGE, fontsize=8)
            ax2.tick_params(axis="y", labelcolor=COLOR_ORANGE)
            overall_std = float(np.nanmean(kld_std_np[warmup:]))
            ax.set_title(
                f"KLD over Dimensions (Mean: {overall_mean:.4f}, Std: {overall_std:.4f})"
            )
            # Combined legend from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, framealpha=0.95)
        else:
            ax.set_title(f"Mean KLD over Dimensions (Overall: {overall_mean:.4f})")

    # KLD heatmap across all latent dimensions
    ax = axes[6]
    if kld_sample_np is None:
        ax.text(0.5, 0.5, "KLD unavailable", ha="center", va="center")
        ax.set_axis_off()
    else:
        # kld_sample_np shape: (T, D) — transpose to (D, T) for display
        kld_heatmap = kld_sample_np.T
        im = ax.imshow(kld_heatmap, aspect="auto", cmap="viridis", origin="lower")
        if warmup > 0:
            ax.axvline(x=warmup - 0.5, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_title("KLD per Latent Dimension")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Latent Dimension")
        ax.grid(False)
        _add_colorbar(fig, im, ax, label="KLD (nats)")

    ax = axes[7]
    ax.plot(t_raw, raw_norm_np, color=COLOR_BLUE, linewidth=1.2, label="FHR (norm)")
    ax.plot(t_raw, avg_pred_np, color=COLOR_ORANGE, linewidth=1.2, label="Avg Prediction")
    ax.set_title("Normalized FHR vs Average Prediction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(t_raw[0], t_raw[-1])
    ax.margins(x=0.0)
    _style_axes(ax, grid="both")

    ax = axes[8]
    ax.plot(t_raw, raw_norm_np, color=COLOR_BLUE, linewidth=1.2, label="FHR (norm)")
    ax.plot(t_raw, pred_concat, color=COLOR_ORANGE, linewidth=1.2, label="Single Predictions")
    # Add fill_between for uncertainty with more visible color
    if std_concat is not None:
        upper = pred_concat + std_concat
        lower = pred_concat - std_concat
        ax.fill_between(t_raw, lower, upper, color=COLOR_SKY, alpha=0.35, label="+/-1SD")
    # Add vertical lines at prediction window boundaries
    for i, boundary in enumerate(prediction_boundaries):
        boundary_sec = boundary / fs
        ax.axvline(x=boundary_sec, color=COLOR_GRAY, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_title("Single-Window Predictions (Normalized)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_xlim(t_raw[0], t_raw[-1])
    ax.margins(x=0.0)
    _style_axes(ax, grid="both")

    guid = _extract_guid(batch, idx)
    epoch = _extract_epoch(batch, idx)
    title = f"Sample Summary | GUID={guid} | Epoch={epoch}"
    fig.suptitle(title, fontsize=14, fontweight="normal", y=1.01, color=COLOR_BLUE)

    plot_path = sample_dir / f"vae_reconstruction_analysis_{sample_name}.pdf"
    fig.savefig(plot_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    plot_paths["vae_reconstruction"] = str(plot_path)
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

                    # Create unique sample name for filename
                    sample_name = _sanitize_folder_name(guid or f"sample_{processed}", epoch or 0.0)

                    logger.info(f"Processing sample {processed + 1}/{n_samples}: GUID={guid}, Epoch={epoch}")

                    # Plot all single-sample plots (save directly to output_path)
                    plot_paths = _plot_all_single_sample_plots(
                        runner=runner,
                        batch=batch,
                        idx=idx_in_batch,
                        outputs=outputs,
                        sample_dir=output_path,
                        sample_name=sample_name,
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
                        "output_dir": str(output_path),
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
