"""
Matplotlib visualization functions for VAE-TEB testing.

This module provides pure plotting functions that take data and output paths,
with no side effects other than saving figures. All functions use a clean,
publication-quality style.

Example:
    >>> from testing.visualizers import plot_metric_histograms
    >>> plot_metric_histograms(metrics_df, Path("results/histograms"))
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean publication style (with fallback for older matplotlib)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # Use default style if seaborn styles unavailable


def plot_metric_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "metrics_histograms.png",
) -> None:
    """
    Create a 2x2 grid of histograms for VAF, MSE, SNR, and KLD.

    Args:
        df: DataFrame with columns 'vaf', 'mse', 'snr', 'kld'.
        output_dir: Directory to save the plot.
        filename: Output filename (default: metrics_histograms.png).

    Example:
        >>> plot_metric_histograms(metrics_df, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Define metrics and their properties
    metrics_config = [
        ("vaf", "VAF (Variance Accounted For)", "#4C72B0", "[0, 1]"),
        ("mse", "MSE (Mean Squared Error)", "#55A868", "lower is better"),
        ("snr", "SNR (Signal-to-Noise Ratio, dB)", "#C44E52", "higher is better"),
        ("kld", "KLD (Transfer Entropy)", "#8172B2", "bits"),
    ]

    for ax, (col, title, color, unit) in zip(axes, metrics_config):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"No {col} data", ha="center", va="center")
            ax.set_title(title)
            continue

        # Get finite values only
        values = df[col].dropna().values
        values = values[np.isfinite(values)]

        if len(values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.set_title(title)
            continue

        # Plot histogram
        ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor="white")

        # Add statistics annotation
        mean_val = np.mean(values)
        std_val = np.std(values)
        median_val = np.median(values)

        stats_text = f"n={len(values):,}\nmean={mean_val:.4f}\nstd={std_val:.4f}\nmedian={median_val:.4f}"
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"{col.upper()} ({unit})")
        ax.set_ylabel("Count")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_latent_distributions(
    latents: np.ndarray,
    output_dir: Path,
    filename: str = "latent_distributions.png",
) -> None:
    """
    Create a grid of histograms for each latent dimension.

    Args:
        latents: Array of shape (N, D) where N is samples and D is latent dim.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Example:
        >>> plot_latent_distributions(latents, Path("results/latent/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if latents.size == 0:
        return

    latent_dim = latents.shape[1]
    cols = 4
    rows = math.ceil(latent_dim / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        if idx < latent_dim:
            # Get values for this dimension
            values = latents[:, idx]
            values = values[np.isfinite(values)]

            if len(values) > 0:
                ax.hist(values, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
                ax.set_title(f"z[{idx}]", fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("Count" if col == 0 else "")
        else:
            ax.axis("off")

    fig.suptitle("Latent Space Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_sample(
    sample: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Create a 3-panel figure for a single sample analysis.

    Panels:
        1. Top: Raw signal with prediction overlay and uncertainty band
        2. Middle: Residual (error) plot
        3. Bottom: Latent representation heatmap

    Args:
        sample: Dict with keys 'y_true', 'y_pred', 'y_pred_std', 'latent', 'metrics'.
        output_path: Full path to save the figure.

    Example:
        >>> plot_reconstruction_sample(samples[0], Path("results/sample_0.png"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    y_pred_std = sample.get("y_pred_std")
    latent = sample.get("latent")
    metrics = sample.get("metrics", {})

    # Create time axis (assuming 4 Hz sampling)
    time = np.arange(len(y_true)) / 4.0 / 60.0  # Convert to minutes

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 2]})

    # ----- Panel 1: Signal with prediction -----
    ax1 = axes[0]
    ax1.plot(time, y_true, label="Ground Truth", color="#4C72B0", alpha=0.8, linewidth=0.8)
    ax1.plot(time, y_pred, label="Prediction", color="#C44E52", alpha=0.8, linewidth=0.8)

    # Add uncertainty band if available
    if y_pred_std is not None:
        ax1.fill_between(
            time,
            y_pred - 2 * y_pred_std,
            y_pred + 2 * y_pred_std,
            alpha=0.2, color="#C44E52", label="±2σ",
        )

    # Add metrics annotation
    metrics_text = f"VAF: {metrics.get('vaf', 0):.3f}  |  SNR: {metrics.get('snr', 0):.1f} dB  |  KLD: {metrics.get('kld', 0):.4f}"
    ax1.text(
        0.02, 0.98, metrics_text,
        transform=ax1.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("FHR (normalized)")
    ax1.set_title(f"Sample: GUID={sample.get('guid', 'N/A')}, Epoch={sample.get('epoch', 'N/A')}")
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, time[-1])

    # ----- Panel 2: Residual -----
    ax2 = axes[1]
    residual = y_true - y_pred
    ax2.plot(time, residual, color="#55A868", alpha=0.8, linewidth=0.5)
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.fill_between(time, residual, 0, alpha=0.3, color="#55A868")
    ax2.set_xlabel("")
    ax2.set_ylabel("Residual")
    ax2.set_xlim(0, time[-1])

    # ----- Panel 3: Latent heatmap -----
    ax3 = axes[2]
    if latent is not None and latent.size > 0:
        # latent shape: (T, D)
        im = ax3.imshow(
            latent.T, aspect="auto", cmap="RdBu_r",
            extent=[0, latent.shape[0], latent.shape[1] - 0.5, -0.5],
        )
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Latent Dim")
        ax3.set_title("Latent Representation z(t)")
        plt.colorbar(im, ax=ax3, label="Value")
    else:
        ax3.text(0.5, 0.5, "No latent data", ha="center", va="center")
        ax3.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    warmup_steps: int = 30,
    filename: str = "temporal_accuracy.png",
) -> None:
    """
    Plot VAF and SNR as a function of timestep position.

    Shows how reconstruction quality varies across the sequence, with
    warmup region clearly marked.

    Args:
        df: DataFrame with 'timestep', 'vaf', 'snr' columns (per-timestep data).
        output_dir: Directory to save the plot.
        warmup_steps: Number of warmup steps to mark (default 30).
        filename: Output filename.

    Example:
        >>> plot_temporal_accuracy(temporal_df, Path("results/temporal/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "timestep" not in df.columns:
        return

    # Aggregate by timestep
    agg = df.groupby("timestep").agg({
        "vaf": ["mean", "std"],
        "snr": ["mean", "std"],
    }).reset_index()
    agg.columns = ["timestep", "vaf_mean", "vaf_std", "snr_mean", "snr_std"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ----- VAF plot -----
    ax1.plot(agg["timestep"], agg["vaf_mean"], color="#4C72B0", linewidth=1.5, label="VAF")
    ax1.fill_between(
        agg["timestep"],
        agg["vaf_mean"] - agg["vaf_std"],
        agg["vaf_mean"] + agg["vaf_std"],
        alpha=0.3, color="#4C72B0",
    )
    # Mark warmup region
    ax1.axvspan(0, warmup_steps, alpha=0.2, color="gray", label="Warmup")
    ax1.axvline(warmup_steps, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("VAF")
    ax1.set_title("Reconstruction Quality vs Timestep")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1)

    # ----- SNR plot -----
    ax2.plot(agg["timestep"], agg["snr_mean"], color="#C44E52", linewidth=1.5, label="SNR")
    ax2.fill_between(
        agg["timestep"],
        agg["snr_mean"] - agg["snr_std"],
        agg["snr_mean"] + agg["snr_std"],
        alpha=0.3, color="#C44E52",
    )
    ax2.axvspan(0, warmup_steps, alpha=0.2, color="gray")
    ax2.axvline(warmup_steps, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("SNR (dB)")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_kld_trajectory(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "kld_trajectory.png",
) -> None:
    """
    Plot KLD (transfer entropy) as a function of time before birth.

    If a 'label' column exists, plots separate lines per class.

    Args:
        df: DataFrame with 'hours_before', 'kld_mean', and optionally 'label'.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Example:
        >>> plot_kld_trajectory(trajectory_df, Path("results/trajectory/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "hours_before" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Check if we have class labels
    has_labels = "label" in df.columns and df["label"].notna().any()

    if has_labels:
        # Plot by class
        label_colors = {0: "#55A868", 1: "#4C72B0", 2: "#C44E52", 3: "#8172B2"}
        label_names = {0: "Unknown", 1: "Healthy", 2: "Acidosis", 3: "HIE"}

        for label in sorted(df["label"].dropna().unique()):
            label = int(label)
            subset = df[df["label"] == label]

            # Bin by hours and compute mean
            subset = subset.copy()
            subset["hour_bin"] = (subset["hours_before"] * 2).round() / 2  # 30-min bins
            agg = subset.groupby("hour_bin")["kld_mean"].agg(["mean", "std"]).reset_index()

            color = label_colors.get(label, "#666666")
            name = label_names.get(label, f"Class {label}")

            ax.plot(
                agg["hour_bin"], agg["mean"],
                color=color, linewidth=2, label=name, marker="o", markersize=4,
            )
            ax.fill_between(
                agg["hour_bin"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.2, color=color,
            )
    else:
        # Single line for all data
        df = df.copy()
        df["hour_bin"] = (df["hours_before"] * 2).round() / 2
        agg = df.groupby("hour_bin")["kld_mean"].agg(["mean", "std"]).reset_index()

        ax.plot(
            agg["hour_bin"], agg["mean"],
            color="#4C72B0", linewidth=2, marker="o", markersize=4,
        )
        ax.fill_between(
            agg["hour_bin"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.3, color="#4C72B0",
        )

    ax.set_xlabel("Hours Before Birth")
    ax.set_ylabel("KLD (Transfer Entropy)")
    ax.set_title("Transfer Entropy Evolution Before Delivery")
    ax.invert_xaxis()  # Time flows right-to-left toward birth
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coherence_analysis(
    frequencies: np.ndarray,
    coherence_original: np.ndarray,
    coherence_reconstructed: np.ndarray,
    output_dir: Path,
    filename: str = "coherence_analysis.png",
) -> None:
    """
    Plot UP-FHR coherence comparison between original and reconstructed signals.

    Args:
        frequencies: Frequency array in Hz.
        coherence_original: Coherence between UP and original FHR.
        coherence_reconstructed: Coherence between UP and reconstructed FHR.
        output_dir: Directory to save the plot.
        filename: Output filename.

    Example:
        >>> plot_coherence_analysis(freqs, coh_orig, coh_recon, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(frequencies, coherence_original, color="#4C72B0", linewidth=2, label="Original UP-FHR")
    ax.plot(frequencies, coherence_reconstructed, color="#C44E52", linewidth=2, label="UP-Reconstructed FHR", linestyle="--")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title("UP-FHR Spectral Coherence Comparison")
    ax.set_xlim(0, min(2.0, frequencies.max()))  # Focus on relevant frequencies
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
