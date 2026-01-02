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
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Set publication-quality style - optimized for high-impact journals
plt.style.use("default")  # Start from clean slate

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.titleweight": "bold",
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
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.3,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": "#000000",
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

# Colorblind-friendly palette (Okabe-Ito, optimized for accessibility and print)
# Validated for deuteranopia, protanopia, and tritanopia
COLOR_BLUE = "#0173B2"        # Sapphire blue
COLOR_ORANGE = "#DE8F05"      # Deep orange
COLOR_GREEN = "#029E73"       # Teal green
COLOR_SKY = "#56B4E9"         # Sky blue
COLOR_PURPLE = "#CC78BC"      # Magenta/purple
COLOR_VERMILLION = "#D55E00"  # Red-orange
COLOR_GRAY = "#555555"        # Dark gray
COLOR_BLACK = "#000000"       # Pure black
COLOR_LIGHT_GRAY = "#999999"  # Light gray for auxiliary elements

# Multi-line palettes for complex figures
PALETTE_PRIMARY = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]
PALETTE_EXTENDED = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE,
                    COLOR_SKY, COLOR_VERMILLION, COLOR_GRAY]
SAVE_DPI = 600
FONT_LABEL = plt.rcParams["axes.labelsize"]
FONT_TITLE = plt.rcParams["axes.titlesize"]
FONT_TICK = plt.rcParams["xtick.labelsize"]
FONT_LEGEND = plt.rcParams["legend.fontsize"]


def _style_axes(ax: plt.Axes, *, grid: str = "major", minor_ticks: bool = True) -> None:
    """
    Apply clean styling to axes with thin black borders.

    Args:
        ax: Matplotlib axes object.
        grid: Grid style - "major", "both", or "none".
        minor_ticks: Whether to show minor ticks.
    """
    ax.set_axisbelow(True)

    # Configure grid - subtle
    if grid in ("both", "major"):
        ax.grid(True, which="major", alpha=0.2, linewidth=0.3, color="#E0E0E0")
    if grid == "both":
        ax.grid(True, which="minor", alpha=0.1, linewidth=0.2, color="#F0F0F0")

    # Configure ticks
    if minor_ticks and grid == "both":
        ax.minorticks_on()

    # Ensure all spines are thin and black
    for spine in ["left", "bottom", "top", "right"]:
        if spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(0.6)


def _add_colorbar(
    fig: plt.Figure,
    mappable: Any,
    ax: plt.Axes,
    *,
    label: Optional[str] = None,
    shrink: float = 0.8,
    pad: float = 0.02,
) -> plt.Axes:
    """Attach a single-column aligned colorbar matching plot_utils.py."""
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=pad)
    if label:
        cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    return cbar


def _format_stats_box(n: int, mean: float, std: float, median: float, **kwargs) -> str:
    """
    Create a formatted statistics text box string.

    Args:
        n: Sample size.
        mean: Mean value.
        std: Standard deviation.
        median: Median value.
        **kwargs: Additional statistics (e.g., ci95=(low, high)).

    Returns:
        Formatted string for statistics annotation.
    """
    text = f"$n$ = {n:,}\n$\\mu$ = {mean:.4f}\n$\\sigma$ = {std:.4f}\n$\\tilde{{x}}$ = {median:.4f}"

    if "ci95" in kwargs:
        low, high = kwargs["ci95"]
        text += f"\n95% CI: [{low:.4f}, {high:.4f}]"

    return text


def plot_metric_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "metrics_histograms.png",
    *,
    add_kde: bool = False,
    add_ci: bool = True,
) -> None:
    """
    Create a 2x2 grid of histograms for VAF, MSE, SNR, and KLD.

    Args:
        df: DataFrame with columns 'vaf', 'mse', 'snr', 'kld'.
        output_dir: Directory to save the plot.
        filename: Output filename (default: metrics_histograms.png).
        add_kde: Whether to add kernel density estimate overlay (default: False).
        add_ci: Whether to include 95% confidence intervals in stats box (default: True).

    Example:
        >>> plot_metric_histograms(metrics_df, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use appropriate figure size for journal (double-column width)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    axes = axes.flatten()

    # Define metrics and their properties
    metrics_config = [
        ("vaf", "Variance Accounted For (VAF)", COLOR_BLUE, ""),
        ("mse", "Mean Squared Error (MSE)", COLOR_GREEN, ""),
        ("snr", "Signal-to-Noise Ratio (SNR)", COLOR_ORANGE, "dB"),
        ("kld", "Transfer Entropy (KLD)", COLOR_PURPLE, "bits"),
    ]

    for ax, (col, title, color, unit) in zip(axes, metrics_config):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"No {col} data", ha="center", va="center", fontsize=FONT_LABEL)
            ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
            continue

        # Get finite values only
        values = df[col].dropna().values
        values = values[np.isfinite(values)]

        if len(values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=FONT_LABEL)
            ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold")
            continue

        # Compute statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        median_val = np.median(values)
        sem = scipy_stats.sem(values)
        ci95 = scipy_stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=sem)

        # Plot histogram with better binning
        n_bins = min(50, max(20, int(np.sqrt(len(values)))))
        counts, bins, patches = ax.hist(
            values, bins=n_bins, density=True,
            color=color, alpha=0.7, edgecolor="#000000",
            linewidth=0.5
        )

        # Add reference lines - thin and clean
        ax.axvline(mean_val, color=COLOR_BLACK, linewidth=0.9,
                  alpha=0.8, label="Mean")
        ax.axvline(median_val, color=COLOR_GRAY, linewidth=0.7,
                  alpha=0.8, label="Median")

        # Add statistics box with proper formatting
        kwargs = {"ci95": ci95} if add_ci else {}
        stats_text = _format_stats_box(len(values), mean_val, std_val, median_val, **kwargs)
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes, fontsize=FONT_LEGEND,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor="#CCCCCC", alpha=0.95, linewidth=0.8),
        )

        # Labels and styling
        xlabel = f"{col.upper()}" + (f" ({unit})" if unit else "")
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=8)
        _style_axes(ax, grid="major", minor_ticks=False)

        # Add subtle legend only to first panel
        if ax == axes[0]:
            ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.9)

    fig.tight_layout(pad=1.5)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latent_distributions(
    latents: np.ndarray,
    output_dir: Path,
    filename: str = "latent_distributions.png",
    *,
    add_gaussian: bool = False,
) -> None:
    """
    Create a grid of histograms for each latent dimension.

    Args:
        latents: Array of shape (N, D) where N is samples and D is latent dim.
        output_dir: Directory to save the plot.
        filename: Output filename.
        add_gaussian: Whether to overlay N(0,1) reference (default: False).

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

    fig_width = 7.0 if cols >= 3 else 3.5
    fig_height = 2.0 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes).reshape(-1)  # Flatten for easier indexing

    for idx in range(rows * cols):
        ax = axes[idx]

        if idx < latent_dim:
            # Get values for this dimension
            values = latents[:, idx]
            values = values[np.isfinite(values)]

            if len(values) > 0:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1))

                # Normalized histogram - clean and simple
                n_bins = min(40, max(15, int(np.sqrt(len(values)))))
                ax.hist(values, bins=n_bins, density=True,
                       color=COLOR_BLUE, alpha=0.7, edgecolor="#000000",
                       linewidth=0.3)

                # Reference lines - thin and minimal
                ax.axvline(mean_val, color=COLOR_BLACK, linewidth=0.7,
                          alpha=0.7)
                ax.axvline(0.0, color=COLOR_LIGHT_GRAY, linewidth=0.5,
                          alpha=0.6)

                # Styling
                ax.set_title(f"$z_{{{idx}}}$", fontsize=FONT_LABEL, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Density" if idx % cols == 0 else "", fontsize=FONT_LEGEND)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)

                _style_axes(ax, grid="major", minor_ticks=False)
        else:
            ax.axis("off")

    fig.suptitle("Latent Space Distributions", fontsize=FONT_TITLE, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_sample(
    sample: Dict[str, Any],
    output_path: Path,
    *,
    fs: float = 4.0,
) -> None:
    """
    Create a 3-panel figure for a single sample analysis with enhanced quality.

    Panels:
        1. Top: Raw signal with prediction overlay and uncertainty band
        2. Middle: Residual (error) plot with distribution
        3. Bottom: Latent representation heatmap

    Args:
        sample: Dict with keys 'y_true', 'y_pred', 'y_pred_std', 'latent', 'metrics'.
        output_path: Full path to save the figure.
        fs: Sampling frequency in Hz (default: 4.0).

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

    # Create time axis
    time = np.arange(len(y_true)) / fs / 60.0  # Convert to minutes

    # Compute correlation coefficient
    corr_coef, p_value = scipy_stats.pearsonr(y_true, y_pred)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0),
                            gridspec_kw={"height_ratios": [2.5, 1, 1.5]})

    # ----- Panel 1: Signal with prediction -----
    ax1 = axes[0]
    ax1.plot(time, y_true, label="Ground truth",
            color=COLOR_BLUE, alpha=0.9, linewidth=0.8, zorder=2)
    ax1.plot(time, y_pred, label="Reconstruction",
            color=COLOR_ORANGE, alpha=0.9, linewidth=0.8, zorder=3)

    # Add uncertainty band if available
    if y_pred_std is not None:
        ax1.fill_between(
            time,
            y_pred - 2 * y_pred_std,
            y_pred + 2 * y_pred_std,
            alpha=0.25, color=COLOR_ORANGE, label="95% CI",
            zorder=1, linewidth=0,
        )

    # Add metrics annotation with improved formatting
    vaf = metrics.get('vaf', np.nan)
    snr = metrics.get('snr', np.nan)
    kld = metrics.get('kld', np.nan)

    metrics_text = (
        f"VAF = {vaf:.3f}  |  SNR = {snr:.1f} dB\n"
        f"KLD = {kld:.4f} bits  |  $r$ = {corr_coef:.3f}"
    )
    ax1.text(
        0.02, 0.98, metrics_text,
        transform=ax1.transAxes, fontsize=FONT_LEGEND,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                 edgecolor="#CCCCCC", alpha=0.95, linewidth=0.8),
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    ax1.set_title(
        f"Reconstruction Quality: GUID={sample.get('guid', 'N/A')}, Epoch={sample.get('epoch', 'N/A')}",
        fontsize=FONT_TITLE, fontweight="bold", pad=8
    )
    ax1.legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)
    ax1.set_xlim(0, time[-1])
    _style_axes(ax1, grid="both", minor_ticks=True)

    # ----- Panel 2: Residual -----
    ax2 = axes[1]
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(residual**2))

    ax2.plot(time, residual, color=COLOR_GREEN, alpha=0.8, linewidth=0.6)
    ax2.axhline(y=0, color=COLOR_BLACK, linewidth=0.6, alpha=0.6, zorder=3)
    ax2.fill_between(time, residual, 0, alpha=0.15, color=COLOR_GREEN, linewidth=0)

    # Add RMSE annotation
    ax2.text(
        0.98, 0.95, f"RMSE = {rmse:.4f}",
        transform=ax2.transAxes, fontsize=FONT_LEGEND,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="#CCCCCC", alpha=0.95, linewidth=0.8),
    )

    ax2.set_xlabel("")
    ax2.set_ylabel("Residual", fontsize=FONT_LABEL)
    ax2.set_xlim(0, time[-1])
    _style_axes(ax2, grid="major", minor_ticks=False)

    # ----- Panel 3: Latent heatmap -----
    ax3 = axes[2]
    if latent is not None and latent.size > 0:
        # latent shape: (T, D)
        # Use symmetric colormap centered at zero
        vmax = np.abs(latent).max()
        im = ax3.imshow(
            latent.T, aspect="auto", cmap="RdBu_r",
            extent=[0, time[-1] * 60 * fs, latent.shape[1] - 0.5, -0.5],
            interpolation="nearest",
            vmin=-vmax, vmax=vmax,
        )
        ax3.set_xlabel("Time (min)", fontsize=FONT_LABEL)
        ax3.set_ylabel("Latent dimension", fontsize=FONT_LABEL)
        ax3.set_title("Latent Representation $\\mathbf{z}(t)$", fontsize=FONT_TITLE, fontweight="bold")

        _add_colorbar(fig, im, ax3, label="Value", shrink=0.8, pad=0.02)
        _style_axes(ax3, grid="major", minor_ticks=False)
    else:
        ax3.text(0.5, 0.5, "No latent data available",
                ha="center", va="center", fontsize=FONT_LABEL)
        ax3.axis("off")

    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    warmup_steps: int = 30,
    filename: str = "temporal_accuracy.png",
) -> None:
    """
    Plot VAF and SNR as a function of timestep position with publication quality.

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
        "vaf": ["mean", "std", "sem"],
        "snr": ["mean", "std", "sem"],
    }).reset_index()
    agg.columns = ["timestep", "vaf_mean", "vaf_std", "vaf_sem",
                  "snr_mean", "snr_std", "snr_sem"]

    # Use wider figure for better visibility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)

    # ----- VAF plot -----
    marker_every = max(1, len(agg) // 20)
    ax1.plot(
        agg["timestep"], agg["vaf_mean"],
        color=COLOR_BLUE, linewidth=0.5, label="VAF",
        marker="o", markersize=1, markevery=marker_every,
        markerfacecolor=COLOR_BLUE, markeredgewidth=0,
    )
    ax1.fill_between(
        agg["timestep"],
        agg["vaf_mean"] - agg["vaf_std"],
        agg["vaf_mean"] + agg["vaf_std"],
        alpha=0.15, color=COLOR_BLUE, linewidth=0,
    )

    # Mark warmup region with subtle styling
    ax1.axvspan(0, warmup_steps, alpha=0.08, color=COLOR_GRAY,
               label="Warmup", zorder=0, linewidth=0)
    ax1.axvline(warmup_steps, color=COLOR_GRAY,
               linewidth=0.6, alpha=0.6, zorder=1)

    ax1.set_ylabel("Variance Accounted For", fontsize=FONT_LABEL)
    ax1.set_title("Reconstruction Quality vs. Timestep",
                 fontsize=FONT_TITLE, fontweight="bold", pad=8)
    ax1.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    ax1.set_ylim(0, 1.05)
    _style_axes(ax1, grid="both", minor_ticks=True)

    # ----- SNR plot -----
    ax2.plot(
        agg["timestep"], agg["snr_mean"],
        color=COLOR_ORANGE, linewidth=0.5, label="SNR",
        marker="s", markersize=2.5, markevery=marker_every,
        markerfacecolor=COLOR_ORANGE, markeredgewidth=0,
    )
    ax2.fill_between(
        agg["timestep"],
        agg["snr_mean"] - agg["snr_std"],
        agg["snr_mean"] + agg["snr_std"],
        alpha=0.15, color=COLOR_ORANGE, linewidth=0,
    )

    ax2.axvspan(0, warmup_steps, alpha=0.08, color=COLOR_GRAY, zorder=0, linewidth=0)
    ax2.axvline(warmup_steps, color=COLOR_GRAY,
               linewidth=0.6, alpha=0.6, zorder=1)

    ax2.set_xlabel("Timestep (prediction index)", fontsize=FONT_LABEL)
    ax2.set_ylabel("Signal-to-Noise Ratio (dB)", fontsize=FONT_LABEL)
    ax2.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax2, grid="both", minor_ticks=True)

    fig.tight_layout(pad=1.0)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_within_window_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    fs: float = 4.0,
    filename: str = "within_window_accuracy.png",
) -> None:
    """
    Plot reconstruction accuracy as a function of index within the prediction window.

    Args:
        df: DataFrame with columns ['window_position', 'abs_error'] or aggregated
            ['window_position', 'mae_mean', 'mae_std', 'vaf_mean', 'snr_mean'].
        output_dir: Directory to save the plot.
        fs: Sampling frequency in Hz.
        filename: Output filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "window_position" not in df.columns:
        return

    if "mae_mean" in df.columns:
        cols = ["window_position", "mae_mean", "mae_std"]
        if "vaf_mean" in df.columns:
            cols.append("vaf_mean")
        if "snr_mean" in df.columns:
            cols.append("snr_mean")
        agg = df[cols].copy()
    else:
        if "abs_error" not in df.columns:
            return
        agg = df.groupby("window_position")["abs_error"].agg(["mean", "std"]).reset_index()
        agg.columns = ["window_position", "mae_mean", "mae_std"]

    has_vaf = "vaf_mean" in agg.columns
    has_snr = "snr_mean" in agg.columns
    n_panels = 1 + int(has_vaf) + int(has_snr)

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(3.5, 2.4 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    x = agg["window_position"].values
    axes[0].plot(
        x,
        agg["mae_mean"],
        color=COLOR_BLUE,
        linewidth=0.5,
        marker="o",
        markersize=1,
        markevery=max(1, len(x) // 30),
        label="Mean absolute error",
    )
    axes[0].fill_between(
        x,
        agg["mae_mean"] - agg["mae_std"],
        agg["mae_mean"] + agg["mae_std"],
        color=COLOR_BLUE,
        alpha=0.2,
    )

    axes[0].set_ylabel("Absolute error", fontsize=FONT_LABEL)
    axes[0].set_title("Within-Window Accuracy", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(axes[0], grid="both", minor_ticks=True)

    secax = axes[0].secondary_xaxis("top", functions=(lambda v: v / fs, lambda s: s * fs))
    secax.set_xlabel("Time from window start (seconds)", fontsize=FONT_LABEL)
    secax.tick_params(labelsize=FONT_TICK)

    panel_idx = 1
    if has_vaf:
        axes[panel_idx].plot(
            x,
            agg["vaf_mean"],
            color=COLOR_GREEN,
            linewidth=0.6,
            marker="o",
            markersize=1,
            markevery=max(1, len(x) // 30),
            label="VAF",
        )
        axes[panel_idx].set_ylabel("VAF", fontsize=FONT_LABEL)
        axes[panel_idx].set_ylim(0.0, 1.0)
        axes[panel_idx].legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
        _style_axes(axes[panel_idx], grid="both", minor_ticks=True)
        panel_idx += 1

    if has_snr:
        axes[panel_idx].plot(
            x,
            agg["snr_mean"],
            color=COLOR_ORANGE,
            linewidth=0.6,
            marker="o",
            markersize=1,
            markevery=max(1, len(x) // 30),
            label="SNR",
        )
        axes[panel_idx].set_ylabel("SNR (dB)", fontsize=FONT_LABEL)
        axes[panel_idx].legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
        _style_axes(axes[panel_idx], grid="both", minor_ticks=True)

    axes[-1].set_xlabel("Index from window start (samples)", fontsize=FONT_LABEL)
    axes[-1].set_xlim(0, np.max(x) if len(x) else 1)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Check if we have class labels
    has_labels = "label" in df.columns and df["label"].notna().any()
    if has_labels and not pd.api.types.is_numeric_dtype(df["label"]):
        has_labels = False

    if has_labels:
        # Plot by class
        label_colors = {
            0: COLOR_GRAY,
            1: COLOR_BLUE,
            2: COLOR_ORANGE,
            3: COLOR_PURPLE,
        }
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
                color=color, linewidth=0.5, label=name,
                marker="o", markersize=1, markevery=max(1, len(agg) // 20),
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
            color=COLOR_BLUE, linewidth=0.5, marker="o", markersize=1,
            markevery=max(1, len(agg) // 20),
        )
        ax.fill_between(
            agg["hour_bin"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.25, color=COLOR_BLUE,
        )

    ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
    ax.set_ylabel("KLD (Transfer Entropy)", fontsize=FONT_LABEL)
    ax.set_title("Transfer Entropy Evolution Before Delivery", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.invert_xaxis()  # Time flows right-to-left toward birth
    if has_labels:
        ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.plot(frequencies, coherence_original, color=COLOR_BLUE, linewidth=0.6, label="UP vs FHR")
    ax.plot(
        frequencies,
        coherence_reconstructed,
        color=COLOR_ORANGE,
        linewidth=0.9,
        label="UP vs Reconstructed FHR",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title("UP-FHR Spectral Coherence Comparison", fontsize=FONT_LABEL, fontweight="bold", pad=6)
    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    ax.set_xlim(0, max_freq)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_coherence(
    frequencies: np.ndarray,
    coherence_mean: np.ndarray,
    coherence_std: np.ndarray,
    output_dir: Path,
    filename: str = "reconstruction_coherence.png",
) -> None:
    """
    Plot coherence between original and reconstructed FHR signals.

    Args:
        frequencies: Frequency array in Hz.
        coherence_mean: Mean coherence across samples.
        coherence_std: Std coherence across samples.
        output_dir: Directory to save the plot.
        filename: Output filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(frequencies, coherence_mean, color=COLOR_BLUE, linewidth=0.5, label="FHR vs Reconstructed FHR")
    ax.fill_between(
        frequencies,
        coherence_mean - coherence_std,
        coherence_mean + coherence_std,
        color=COLOR_BLUE,
        alpha=0.2,
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title("FHR Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    ax.set_xlim(0, max_freq)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_psd_comparison(
    frequencies: np.ndarray,
    psd_orig_mean: np.ndarray,
    psd_orig_std: np.ndarray,
    psd_recon_mean: np.ndarray,
    psd_recon_std: np.ndarray,
    output_dir: Path,
    *,
    psd_residual_mean: Optional[np.ndarray] = None,
    psd_residual_std: Optional[np.ndarray] = None,
    filename: str = "psd_comparison.png",
) -> None:
    """
    Plot Welch PSD comparison between original and reconstructed signals.

    Args:
        frequencies: Frequency array in Hz.
        psd_orig_mean: Mean PSD of original signal.
        psd_orig_std: Std PSD of original signal.
        psd_recon_mean: Mean PSD of reconstructed signal.
        psd_recon_std: Std PSD of reconstructed signal.
        output_dir: Directory to save plot.
        psd_residual_mean: Optional mean PSD of residual.
        psd_residual_std: Optional std PSD of residual.
        filename: Output filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    def _to_db(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(x, 1e-12))

    def _band_db(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        low = np.maximum(mean - std, 1e-12)
        high = np.maximum(mean + std, 1e-12)
        return _to_db(low), _to_db(high)

    orig_db = _to_db(psd_orig_mean)
    recon_db = _to_db(psd_recon_mean)

    # Plot with clean styling
    ax.plot(frequencies, orig_db, color=COLOR_BLUE, linewidth=0.6,
           label="Original FHR", zorder=3)
    ax.plot(
        frequencies,
        recon_db,
        color=COLOR_ORANGE,
        linewidth=0.9,
        label="Reconstructed FHR",
        zorder=3,
    )

    # Uncertainty bands with proper dB conversion
    orig_low, orig_high = _band_db(psd_orig_mean, psd_orig_std)
    recon_low, recon_high = _band_db(psd_recon_mean, psd_recon_std)
    ax.fill_between(frequencies, orig_low, orig_high,
                    color=COLOR_BLUE, alpha=0.2, linewidth=0, zorder=1)
    ax.fill_between(frequencies, recon_low, recon_high,
                    color=COLOR_ORANGE, alpha=0.2, linewidth=0, zorder=2)

    if psd_residual_mean is not None:
        resid_db = _to_db(psd_residual_mean)
        ax.plot(frequencies, resid_db, color=COLOR_GREEN, linewidth=0.5,
               label="Residual", zorder=3)
        if psd_residual_std is not None:
            resid_low, resid_high = _band_db(psd_residual_mean, psd_residual_std)
            ax.fill_between(
                frequencies, resid_low, resid_high,
                color=COLOR_GREEN,
                alpha=0.2,
                linewidth=0,
                zorder=1,
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("PSD (dB/Hz)", fontsize=FONT_LABEL)
    ax.set_title("Power Spectral Density Comparison (Welch)", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    ax.set_xlim(0, max_freq)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cross_correlation(
    lags_sec: np.ndarray,
    corr_mean: np.ndarray,
    corr_std: np.ndarray,
    output_dir: Path,
    filename: str = "cross_correlation.png",
    *,
    annotate_peak: bool = True,
) -> None:
    """
    Plot cross-correlation between original and reconstructed signals with peak annotation.

    Args:
        lags_sec: Lag array in seconds.
        corr_mean: Mean correlation values.
        corr_std: Std correlation values.
        output_dir: Directory to save plot.
        filename: Output filename.
        annotate_peak: Whether to annotate the peak correlation (default: True).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Plot main curve with clean styling
    ax.plot(lags_sec, corr_mean, color=COLOR_PURPLE, linewidth=0.5,
           label="Cross-correlation", zorder=3)
    ax.fill_between(
        lags_sec,
        corr_mean - corr_std,
        corr_mean + corr_std,
        color=COLOR_PURPLE,
        alpha=0.2,
        linewidth=0,
        zorder=1,
    )

    # Find and annotate peak if requested
    if annotate_peak and len(corr_mean) > 0:
        peak_idx = np.argmax(np.abs(corr_mean))
        peak_lag = lags_sec[peak_idx]
        peak_corr = corr_mean[peak_idx]

        ax.plot(peak_lag, peak_corr, 'o', color=COLOR_ORANGE,
               markersize=4, markeredgecolor='white', markeredgewidth=0.3,
               zorder=4, label=f"Peak: $r$={peak_corr:.3f} at {peak_lag:.2f}s")

    # Reference lines - thin and clean
    ax.axvline(0.0, color=COLOR_BLACK, linewidth=0.7,
              alpha=0.6, zorder=2, label="Zero lag")
    ax.axhline(0.0, color=COLOR_LIGHT_GRAY,
              linewidth=0.6, alpha=0.5, zorder=0)

    ax.set_xlabel("Lag (seconds)", fontsize=FONT_LABEL)
    ax.set_ylabel("Normalized correlation", fontsize=FONT_LABEL)
    ax.set_title("Cross-Correlation Analysis", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95, loc="best")
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coherence_signals(
    up_signal: Optional[np.ndarray],
    fhr_original: np.ndarray,
    fhr_reconstructed: np.ndarray,
    output_path: Path,
    *,
    fs: float = 4.0,
    title: Optional[str] = None,
) -> None:
    """
    Plot FHR signals for a single sample, with optional UP context.

    Args:
        up_signal: UP signal array (optional).
        fhr_original: Original FHR array.
        fhr_reconstructed: Reconstructed FHR array.
        output_path: Output path for the figure.
        fs: Sampling frequency in Hz (default 4.0).
        title: Optional title for the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_len = min(len(fhr_original), len(fhr_reconstructed))
    if up_signal is not None:
        min_len = min(min_len, len(up_signal))
        up_signal = up_signal[:min_len]
    fhr_original = fhr_original[:min_len]
    fhr_reconstructed = fhr_reconstructed[:min_len]

    time_min = np.arange(min_len) / fs / 60.0

    if up_signal is None:
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)
        axes[0].plot(time_min, fhr_original, color=COLOR_BLUE, linewidth=0.6, label="Original FHR")
        axes[0].plot(
            time_min,
            fhr_reconstructed,
            color=COLOR_ORANGE,
            linewidth=0.9,
            label="Reconstructed FHR",
        )
        axes[0].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
        axes[0].set_title(title or "FHR Reconstruction", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        axes[0].legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)

        residual = fhr_original - fhr_reconstructed
        axes[1].plot(time_min, residual, color=COLOR_GREEN, linewidth=0.6)
        axes[1].axhline(0.0, color=COLOR_BLACK, alpha=0.5, linewidth=0.7)
        axes[1].set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
        axes[1].set_ylabel("Residual", fontsize=FONT_LABEL)
        _style_axes(axes[0], grid="both", minor_ticks=True)
        _style_axes(axes[1], grid="major", minor_ticks=False)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.5), sharex=True)
        axes[0].plot(time_min, up_signal, color=COLOR_PURPLE, linewidth=0.5)
        axes[0].set_ylabel("UP (normalized)", fontsize=FONT_LABEL)
        axes[0].set_title(title or "UP and FHR Signals", fontsize=FONT_TITLE, fontweight="bold", pad=6)

        axes[1].plot(time_min, fhr_original, color=COLOR_BLUE, linewidth=0.7, label="Original FHR")
        axes[1].plot(
            time_min,
            fhr_reconstructed,
            color=COLOR_ORANGE,
            linewidth=0.9,
            label="Reconstructed FHR",
        )
        axes[1].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
        axes[1].legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)

        residual = fhr_original - fhr_reconstructed
        axes[2].plot(time_min, residual, color=COLOR_GREEN, linewidth=0.5)
        axes[2].axhline(0.0, color=COLOR_BLACK, alpha=0.5, linewidth=0.6)
        axes[2].set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
        axes[2].set_ylabel("Residual", fontsize=FONT_LABEL)
        _style_axes(axes[0], grid="both", minor_ticks=True)
        _style_axes(axes[1], grid="both", minor_ticks=True)
        _style_axes(axes[2], grid="major", minor_ticks=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_time_frequency_coherence(
    frequencies: np.ndarray,
    times: np.ndarray,
    coherence_original: np.ndarray,
    output_path: Path,
    *,
    coherence_reconstructed: Optional[np.ndarray] = None,
    max_freq: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot time-frequency coherence maps for original and reconstructed signals.

    Args:
        frequencies: Frequency array in Hz.
        times: Time array in seconds.
        coherence_original: Coherence matrix (freq x time) for a reference pairing.
        coherence_reconstructed: Optional coherence matrix (freq x time) for a comparison pairing.
        output_path: Output path for the figure.
        max_freq: Optional max frequency to display.
        title: Optional title for the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freq_mask = slice(None)
    if max_freq is not None:
        freq_mask = frequencies <= max_freq

    freqs = frequencies[freq_mask]
    coh_orig = coherence_original[freq_mask, :]
    time_min = times / 60.0

    if coherence_reconstructed is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
        im = ax.pcolormesh(time_min, freqs, coh_orig, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
        ax.set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
        ax.set_title("FHR Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        _add_colorbar(fig, im, ax, label="Coherence", shrink=0.8, pad=0.02)
        _style_axes(ax, grid="major")

        if title:
            fig.suptitle(title, fontsize=FONT_TITLE, y=0.98)

        fig.tight_layout()
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    coh_recon = coherence_reconstructed[freq_mask, :]
    coh_diff = coh_recon - coh_orig

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)

    im0 = axes[0].pcolormesh(time_min, freqs, coh_orig, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[0].set_title("Reference Coherence", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    _add_colorbar(fig, im0, axes[0], label="Coherence", shrink=0.8, pad=0.02)

    im1 = axes[1].pcolormesh(time_min, freqs, coh_recon, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[1].set_title("Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    _add_colorbar(fig, im1, axes[1], label="Coherence", shrink=0.8, pad=0.02)

    im2 = axes[2].pcolormesh(time_min, freqs, coh_diff, shading="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[2].set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
    axes[2].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[2].set_title("Coherence Difference (Reconstruction - Reference)", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    _add_colorbar(fig, im2, axes[2], label="Delta", shrink=0.8, pad=0.02)

    for ax in axes:
        _style_axes(ax, grid="major")

    if title:
        fig.suptitle(title, fontsize=FONT_TITLE, y=0.98)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Trajectory Visualization Functions
# -----------------------------------------------------------------------------

def plot_latent_trajectory_2d(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    color_by_time: bool = True,
    point_size: int = 12,
    show_arrows: bool = True,
) -> None:
    """
    Plot 2D latent trajectory with temporal coloring and directional arrows.

    Args:
        trajectory: Trajectory array of shape (T, 2) where T is time steps.
        output_path: Path to save the figure.
        sample_id: Sample identifier for the title.
        color_by_time: Whether to color points by time progression.
        point_size: Size of trajectory points.
        show_arrows: Whether to show directional arrows.

    Example:
        >>> plot_latent_trajectory_2d(trajectory_2d, Path("results/traj_2d.png"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    time_steps = trajectory.shape[0]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Color by time
    if color_by_time:
        colors = plt.cm.viridis(np.linspace(0, 1, time_steps))
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1],
            c=np.arange(time_steps), cmap="viridis",
            s=point_size, zorder=3, edgecolor="white", linewidth=0.3
        )
    else:
        colors = [COLOR_BLUE] * time_steps
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1],
            c=COLOR_BLUE, s=point_size, zorder=3,
            edgecolor="white", linewidth=0.3
        )

    # Draw arrows
    if show_arrows:
        for i in range(time_steps - 1):
            ax.annotate(
                "",
                xy=trajectory[i + 1],
                xytext=trajectory[i],
                arrowprops=dict(
                    arrowstyle="->",
                    lw=0.8,
                    color=colors[i] if color_by_time else COLOR_BLUE,
                    alpha=0.6,
                ),
            )

    # Mark start and end
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1],
        c=COLOR_GREEN, s=50, marker="o", label="Start",
        zorder=4, edgecolor=COLOR_BLACK, linewidth=0.4
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1],
        c=COLOR_VERMILLION, s=50, marker="X", label="End",
        zorder=4, edgecolor=COLOR_BLACK, linewidth=0.4
    )

    if color_by_time:
        _add_colorbar(fig, scatter, ax, label="Time Step", shrink=0.8, pad=0.02)

    ax.set_xlabel("Latent Dim 1", fontsize=FONT_LABEL)
    ax.set_ylabel("Latent Dim 2", fontsize=FONT_LABEL)
    ax.set_title(f"Latent Trajectory - {sample_id}", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="major", minor_ticks=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latent_trajectory_3d(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    color_by_time: bool = True,
    point_size: int = 12,
) -> None:
    """
    Plot 3D latent trajectory with temporal coloring.

    Args:
        trajectory: Trajectory array of shape (T, 3) where T is time steps.
        output_path: Path to save the figure.
        sample_id: Sample identifier for the title.
        color_by_time: Whether to color points by time progression.
        point_size: Size of trajectory points.

    Example:
        >>> plot_latent_trajectory_3d(trajectory_3d, Path("results/traj_3d.png"))
    """
    from mpl_toolkits.mplot3d import Axes3D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    time_steps = trajectory.shape[0]

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    # Color by time
    if color_by_time:
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=np.arange(time_steps), cmap="viridis",
            s=point_size, depthshade=True
        )
    else:
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=COLOR_BLUE, s=point_size, depthshade=True
        )

    # Draw trajectory line
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color=COLOR_BLUE, alpha=0.4, linewidth=0.5
    )

    # Mark start and end
    ax.scatter(
        [trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
        c=COLOR_GREEN, s=50, marker="o", label="Start", depthshade=False
    )
    ax.scatter(
        [trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
        c=COLOR_VERMILLION, s=50, marker="X", label="End", depthshade=False
    )

    if color_by_time:
        _add_colorbar(fig, scatter, ax, label="Time Step", shrink=0.8, pad=0.02)

    ax.set_xlabel("Latent Dim 1", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Latent Dim 2", fontsize=FONT_LABEL, labelpad=10)
    ax.set_zlabel("Latent Dim 3", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(f"Latent Trajectory 3D - {sample_id}", fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=FONT_LEGEND)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latent_changepoints_with_raw(
    latent_mean: np.ndarray,
    fhr: np.ndarray,
    changepoint_results: Dict[str, Any],
    output_path: Path,
    *,
    sample_id: str = "sample",
    cmap: str = "viridis",
    decimation_factor: int = 16,
) -> None:
    """
    Visualize latent trajectory alongside raw FHR signal with changepoints marked.

    Creates a 3-panel figure:
        1. Latent representation heatmap with changepoints
        2. FHR signal with latent-derived changepoints
        3. FHR signal with raw-detected changepoints (if available)

    Args:
        latent_mean: Latent trajectory of shape (T, D).
        fhr: Raw FHR signal of shape (L,).
        changepoint_results: Dict from detect_changepoints with keys:
            - 'latent_changepoints': Changepoint indices in latent space
            - 'raw_changepoints': Mapped indices in raw space
            - 'raw_detected_changepoints': Directly detected in raw (optional)
        output_path: Path to save the figure.
        sample_id: Sample identifier for the title.
        cmap: Colormap for latent heatmap.
        decimation_factor: Ratio between raw and latent lengths.

    Example:
        >>> plot_latent_changepoints_with_raw(latent, fhr, cp_results, Path("results/cp.png"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latent_cps = changepoint_results.get("latent_changepoints", np.array([]))
    raw_cps_from_latent = changepoint_results.get("raw_changepoints", np.array([]))
    raw_detected_cps = changepoint_results.get("raw_detected_changepoints", np.array([]))

    latent_time_steps, latent_dims = latent_mean.shape
    raw_len = len(fhr)

    # Map latent changepoints to raw indices if not provided
    if len(raw_cps_from_latent) == 0 and len(latent_cps) > 0:
        raw_cps_from_latent = np.array([
            min(raw_len - 1, int(cp * decimation_factor)) for cp in latent_cps
        ])

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1, 1]})

    # Panel 1: Latent heatmap
    x_max = max(raw_len - 1, 0)
    extent = (0, x_max, -0.5, latent_dims - 0.5)
    im = axes[0].imshow(
        latent_mean.T, aspect="auto", origin="lower", cmap=cmap,
        interpolation="nearest", extent=extent
    )
    axes[0].set_xlim(extent[0], extent[1])
    axes[0].set_ylim(extent[2], extent[3])
    axes[0].set_ylabel("Latent Dimension", fontsize=FONT_LABEL)
    axes[0].set_title(f"Latent Representation - {sample_id}", fontsize=FONT_TITLE, fontweight="bold", pad=6)

    for raw_cp in raw_cps_from_latent:
        axes[0].axvline(raw_cp, color="white", linewidth=0.7, alpha=0.8)

    cbar = _add_colorbar(fig, im, axes[0], label="Activation", shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Activation", rotation=270, labelpad=12, fontsize=plt.rcParams["axes.labelsize"])

    # Panel 2: FHR with latent-derived changepoints
    time_axis = np.arange(raw_len)
    axes[1].plot(time_axis, fhr, color=COLOR_VERMILLION, linewidth=0.5, label="FHR")
    axes[1].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    axes[1].set_title("FHR with Latent Changepoints", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    axes[1].set_xlim(0, x_max)
    _style_axes(axes[1], grid="major", minor_ticks=False)
    axes[1].legend(loc="upper right", fontsize=FONT_LEGEND)

    for raw_cp in raw_cps_from_latent:
        axes[1].axvline(raw_cp, color=COLOR_BLACK, linewidth=0.5, alpha=0.6)

    # Panel 3: FHR with both latent and raw changepoints
    axes[2].plot(time_axis, fhr, color=COLOR_VERMILLION, linewidth=0.5, label="FHR")
    axes[2].set_xlabel("Raw Time Index", fontsize=FONT_LABEL)
    axes[2].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    axes[2].set_title("FHR with Latent (gray) and Raw (green) Changepoints", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    axes[2].set_xlim(0, x_max)
    _style_axes(axes[2], grid="major", minor_ticks=False)

    latent_line_added = False
    raw_line_added = False
    for raw_cp in raw_cps_from_latent:
        axes[2].axvline(
            raw_cp, color=COLOR_GRAY, linewidth=0.5, alpha=0.6,
            label="Latent CPs" if not latent_line_added else None
        )
        latent_line_added = True
    for raw_cp in raw_detected_cps:
        axes[2].axvline(
            raw_cp, color=COLOR_GREEN, linestyle="-", linewidth=0.9, alpha=0.75,
            label="Raw CPs" if not raw_line_added else None
        )
        raw_line_added = True

    if latent_line_added or raw_line_added:
        axes[2].legend(loc="upper right", fontsize=FONT_LEGEND)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_segment_statistics(
    segment_stats: List[Dict[str, Any]],
    output_dir: Path,
    *,
    filename_prefix: str = "segment",
) -> Optional[pd.DataFrame]:
    """
    Visualize aggregated latent segment statistics from changepoint analysis.

    Creates multiple plots:
        1. Segment duration histogram
        2. Mean speed vs start time scatter
        3. Dominant latent dimension bar chart
        4. Segments per sample bar chart

    Also exports segment data to CSV.

    Args:
        segment_stats: List of per-sample dicts from summarize_latent_segments.
        output_dir: Directory to save plots.
        filename_prefix: Prefix for output files.

    Returns:
        DataFrame with flattened segment statistics, or None if no data.

    Example:
        >>> df = plot_segment_statistics(segment_stats, Path("results/segments/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not segment_stats:
        return None

    # Flatten to DataFrame
    rows: List[Dict[str, Any]] = []
    for entry in segment_stats:
        sample_id = entry.get("sample_id")
        epoch_raw_index = entry.get("epoch_raw_index")
        for seg in entry.get("segments", []):
            row = {
                "sample_id": sample_id,
                "epoch_raw_index": epoch_raw_index,
                "segment_index": seg.get("segment_index"),
                "start_step": seg.get("start_step"),
                "end_step": seg.get("end_step"),
                "length_steps": seg.get("length_steps"),
                "duration_seconds": seg.get("duration_seconds"),
                "start_minutes_rel_delivery": seg.get("start_minutes_rel_delivery"),
                "end_minutes_rel_delivery": seg.get("end_minutes_rel_delivery"),
                "dominant_latent_dim": seg.get("dominant_latent_dim"),
                "mean_speed": seg.get("mean_speed"),
                "mean_activation_norm": seg.get("mean_activation_norm"),
            }
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / f"{filename_prefix}_stats.csv"
    df.to_csv(csv_path, index=False)

    # Plot 1: Duration histogram
    durations = df["duration_seconds"].dropna()
    if not durations.empty:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        durations_minutes = durations / 60.0
        ax.hist(durations_minutes, bins=20, color=COLOR_BLUE, alpha=0.75, edgecolor=COLOR_BLACK, linewidth=0.5)
        ax.set_xlabel("Duration (minutes)", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.set_title("Segment Duration Distribution", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_duration_hist.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Plot 2: Mean speed vs start time
    start_speed = df[["start_minutes_rel_delivery", "mean_speed"]].dropna()
    if not start_speed.empty:
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.scatter(
            start_speed["start_minutes_rel_delivery"],
            start_speed["mean_speed"],
            s=12, c=COLOR_ORANGE, alpha=0.7, edgecolors=COLOR_BLACK, linewidths=0.2
        )
        ax.set_xlabel("Start Minutes Rel. Delivery", fontsize=FONT_LABEL)
        ax.set_ylabel("Mean Latent Speed", fontsize=FONT_LABEL)
        ax.set_title("Latent Speed vs Start Time", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        ax.axvline(0.0, color=COLOR_GRAY, linewidth=0.6, alpha=0.6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_speed_vs_start.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Plot 3: Dominant latent dimension counts
    dominant_dims = df["dominant_latent_dim"].dropna()
    if not dominant_dims.empty:
        dominant_counts = dominant_dims.astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.bar(dominant_counts.index.astype(str), dominant_counts.values, color=COLOR_GREEN, alpha=0.8, edgecolor=COLOR_BLACK, linewidth=0.5)
        ax.set_xlabel("Latent Dimension Index", fontsize=FONT_LABEL)
        ax.set_ylabel("Segment Count", fontsize=FONT_LABEL)
        ax.set_title("Dominant Latent Dimension", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_dominant_dim.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Plot 4: Segments per sample
    segments_per_sample = df.groupby("sample_id")["segment_index"].count()
    if not segments_per_sample.empty and len(segments_per_sample) <= 30:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        segments_per_sample.sort_values(ascending=False).plot(
            kind="bar", ax=ax, color=COLOR_PURPLE, alpha=0.8, edgecolor=COLOR_BLACK, linewidth=0.5
        )
        ax.set_xlabel("Sample ID", fontsize=FONT_LABEL)
        ax.set_ylabel("Number of Segments", fontsize=FONT_LABEL)
        ax.set_title("Segments per Sample", fontsize=FONT_TITLE, fontweight="bold", pad=6)
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_TICK)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_per_sample.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    return df


def plot_trajectory_comparison(
    trajectories: Dict[str, np.ndarray],
    output_dir: Path,
    *,
    n_components: int = 2,
    filename: str = "trajectory_comparison.png",
) -> None:
    """
    Compare latent trajectories across multiple classes in a single plot.

    Args:
        trajectories: Dict mapping class names to trajectory arrays.
            Each array has shape (N, T, D) where N is samples, T is time, D is dims.
        output_dir: Directory to save the plot.
        n_components: Number of dimensions to plot (2 or 3).
        filename: Output filename.

    Example:
        >>> trajectories = {"healthy": healthy_trajs, "acidosis": acidosis_trajs}
        >>> plot_trajectory_comparison(trajectories, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color map for classes
    class_colors = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
        "unknown": COLOR_GRAY,
    }

    if n_components == 3:
        fig = plt.figure(figsize=(8.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))

    for class_name, class_trajectories in trajectories.items():
        color = class_colors.get(class_name.lower(), COLOR_BLUE)

        if class_trajectories.ndim == 2:
            class_trajectories = class_trajectories[None, ...]

        for i, traj in enumerate(class_trajectories):
            alpha = 0.3 + 0.5 * (i == 0)  # Highlight first trajectory

            if n_components == 3 and traj.shape[1] >= 3:
                ax.plot(
                    traj[:, 0], traj[:, 1], traj[:, 2],
                    color=color, alpha=alpha, linewidth=0.7,
                    label=class_name if i == 0 else None
                )
                ax.scatter(
                    [traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                    c=color, s=25, marker="o", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2
                )
                ax.scatter(
                    [traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                    c=color, s=25, marker="X", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2
                )
            else:
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=color, alpha=alpha, linewidth=0.7,
                    label=class_name if i == 0 else None
                )
                ax.scatter(
                    traj[0, 0], traj[0, 1],
                    c=color, s=25, marker="o", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2
                )
                ax.scatter(
                    traj[-1, 0], traj[-1, 1],
                    c=color, s=25, marker="X", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2
                )

    if n_components == 3:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL, labelpad=10)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL, labelpad=10)
        ax.set_zlabel("PC3", fontsize=FONT_LABEL, labelpad=10)
    else:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL)
        _style_axes(ax, grid="major", minor_ticks=False)

    ax.set_title("Trajectory Comparison by Class", fontsize=FONT_TITLE, fontweight="bold", pad=6)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
