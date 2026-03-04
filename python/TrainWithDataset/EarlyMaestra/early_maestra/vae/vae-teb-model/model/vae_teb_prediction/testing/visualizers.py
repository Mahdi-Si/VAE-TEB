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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

# Set publication-quality style - optimized for high-impact journals
plt.style.use("default")  # Start from clean slate

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
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
    "axes.edgecolor": "#222831",
    "axes.labelcolor": "#222831",
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
    "xtick.color": "#222831",
    "ytick.color": "#222831",
    "grid.alpha": 0.2,
    "grid.linewidth": 0.3,
    "grid.color": "#EEEEEE",
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": "#393E46",
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

# Testing palette (user-provided)
COLOR_BLUE = "#3F72AF"
COLOR_ORANGE = "#FFB200"
COLOR_GREEN = "#609966"
COLOR_SKY = "#00ADB5"
COLOR_PURPLE = "#112D4E"
COLOR_VERMILLION = "#EB5B00"
COLOR_GRAY = "#393E46"
COLOR_BLACK = "#222831"
COLOR_LIGHT_GRAY = "#EEEEEE"
COLOR_SAGE = "#9DC08B"
COLOR_TEAL_DARK = "#0D7377"

# Multi-line palettes for complex figures
PALETTE_PRIMARY = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_SKY]
PALETTE_EXTENDED = [
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_SKY,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    COLOR_SAGE,
    COLOR_TEAL_DARK,
]
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
        ax.grid(True, which="major", alpha=0.25, linewidth=0.3, color=COLOR_LIGHT_GRAY)
    if grid == "both":
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.2, color=COLOR_LIGHT_GRAY)

    # Configure ticks
    if minor_ticks and grid == "both":
        ax.minorticks_on()

    # Ensure all spines are thin and black
    for spine in ["left", "bottom", "top", "right"]:
        if spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(COLOR_BLACK)
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"], color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"], colors=COLOR_BLACK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    return cbar


def _tighten_xaxis(ax: plt.Axes, x_vals: np.ndarray) -> None:
    """Remove extra horizontal padding so traces align to axis edges."""
    if x_vals is None or len(x_vals) == 0:
        return
    finite = np.asarray(x_vals)[np.isfinite(x_vals)]
    if finite.size == 0:
        return
    ax.set_xlim(float(np.min(finite)), float(np.max(finite)))
    ax.margins(x=0.0)


def _set_ylim_from_data(
    ax: plt.Axes,
    y_vals: np.ndarray,
    *,
    pad_frac: float = 0.05,
    min_zero: bool = False,
    clamp: Optional[Tuple[float, float]] = None,
) -> None:
    """Set y-limits based on data with minimal padding."""
    if y_vals is None:
        return
    finite = np.asarray(y_vals)[np.isfinite(y_vals)]
    if finite.size == 0:
        return
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if min_zero:
        y_min = 0.0
    span = max(y_max - y_min, 1e-12)
    y_min -= span * pad_frac
    y_max += span * pad_frac
    if clamp is not None:
        y_min = max(clamp[0], y_min)
        y_max = min(clamp[1], y_max)
    ax.set_ylim(y_min, y_max)


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
    filename: str = "metrics_histograms.pdf",
    *,
    add_kde: bool = False,
    add_ci: bool = True,
) -> None:
    """
    Create a 2x2 grid of histograms for VAF, MSE, SNR, and KLD.

    Args:
        df: DataFrame with columns 'vaf', 'mse', 'snr', 'kld'.
        output_dir: Directory to save the plot.
        filename: Output filename (default: metrics_histograms.pdf).
        add_kde: Whether to add kernel density estimate overlay (default: False).
        add_ci: Whether to include 95% confidence intervals in stats box (default: True).

    Example:
        >>> plot_metric_histograms(metrics_df, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Single-column layout with wider panels
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.5))
    axes = np.atleast_1d(axes)

    # Define metrics and their properties
    metrics_config = [
        ("vaf", "Variance Accounted For (VAF)", COLOR_BLUE, ""),
        ("mse", "Mean Squared Error (MSE)", COLOR_GREEN, ""),
        ("snr", "Signal-to-Noise Ratio (SNR)", COLOR_ORANGE, "dB"),
        ("kld", "Transfer Entropy (KLD)", COLOR_PURPLE, "nats"),
    ]

    for ax, (col, title, color, unit) in zip(axes, metrics_config):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"No {col} data", ha="center", va="center", fontsize=FONT_LABEL)
            ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
            continue

        # Get finite values only
        values = df[col].dropna().values
        values = values[np.isfinite(values)]

        if len(values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=FONT_LABEL)
            ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
            continue

        # Compute statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        median_val = np.median(values)
        sem = scipy_stats.sem(values)
        ci95 = scipy_stats.t.interval(0.95, len(values) - 1, loc=mean_val, scale=sem)

        # Plot histogram with finer binning
        n_bins = min(120, max(40, int(np.sqrt(len(values)) * 2)))
        counts, bins, patches = ax.hist(
            values, bins=n_bins, density=True,
            color=color, alpha=0.7, edgecolor=COLOR_BLACK,
            linewidth=0.5
        )

        # Add reference lines with distinct colors and dashed styles
        ax.axvline(mean_val, color=COLOR_ORANGE, linewidth=0.9,
                   linestyle="--", alpha=0.9, label="Mean")
        ax.axvline(median_val, color=COLOR_PURPLE, linewidth=0.8,
                   linestyle="-.", alpha=0.9, label="Median")

        # Add statistics box with proper formatting
        kwargs = {"ci95": ci95} if add_ci else {}
        stats_text = _format_stats_box(len(values), mean_val, std_val, median_val, **kwargs)
        ax.text(
            1.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=FONT_LEGEND,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     edgecolor=COLOR_LIGHT_GRAY, alpha=0.95, linewidth=0.8),
            clip_on=False,
        )

        # Labels and styling
        xlabel = f"{col.upper()}" + (f" ({unit})" if unit else "")
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal", pad=8)

        # Use log scale for KLD x-axis by default
        if col == "kld":
            ax.set_xscale('log')

        _style_axes(ax, grid="major", minor_ticks=False)
        _tighten_xaxis(ax, values)
        ax.set_ylim(bottom=0.0)

        # Add legend outside, below the stats box
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 0.62),
            fontsize=FONT_LEGEND,
            framealpha=0.9,
        )

    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0], pad=0.8)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_latent_distributions(
    latents: np.ndarray,
    output_dir: Path,
    filename: str = "latent_distributions.pdf",
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

    fig_width = 6.5 if cols >= 3 else 4.5
    fig_height = 2.2 * rows
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

                # Normalized histogram with finer binning
                n_bins = min(100, max(30, int(np.sqrt(len(values)) * 2)))
                ax.hist(values, bins=n_bins, density=True,
                       color=COLOR_BLUE, alpha=0.7, edgecolor=COLOR_BLACK,
                       linewidth=0.3)

                # Reference lines with distinct colors and dashed styles
                median_val = float(np.median(values))
                ax.axvline(mean_val, color=COLOR_ORANGE, linewidth=0.7,
                           linestyle="--", alpha=0.8)
                ax.axvline(median_val, color=COLOR_PURPLE, linewidth=0.7,
                           linestyle="-.", alpha=0.8)

                # Styling
                ax.set_title(f"$z_{{{idx}}}$", fontsize=FONT_LABEL, fontweight="normal")
                ax.set_xlabel("")
                ax.set_ylabel("Density" if idx % cols == 0 else "", fontsize=FONT_LEGEND)
                ax.tick_params(axis='both', which='major', labelsize=FONT_TICK)

                _style_axes(ax, grid="major", minor_ticks=False)
                _tighten_xaxis(ax, values)
        else:
            ax.axis("off")

    fig.suptitle("Latent Space Distributions", fontsize=FONT_TITLE, fontweight="normal", y=0.99)
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
        >>> plot_reconstruction_sample(samples[0], Path("results/sample_0.pdf"))
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

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6.5),
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
                 edgecolor=COLOR_LIGHT_GRAY, alpha=0.95, linewidth=0.8),
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    ax1.set_title(
        f"Reconstruction Quality: GUID={sample.get('guid', 'N/A')}, Epoch={sample.get('epoch', 'N/A')}",
        fontsize=FONT_TITLE, fontweight="normal", pad=8
    )
    ax1.legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)
    _tighten_xaxis(ax1, time)
    _style_axes(ax1, grid="both", minor_ticks=True)

    # ----- Panel 2: Residual -----
    ax2 = axes[1]
    residual = y_true - y_pred
    rmse = np.sqrt(np.mean(residual**2))

    ax2.plot(time, residual, color=COLOR_GREEN, alpha=0.8, linewidth=0.6, label="Residual")
    ax2.axhline(y=0, color=COLOR_BLACK, linewidth=0.6, alpha=0.6, zorder=3)
    ax2.fill_between(
        time,
        residual,
        0,
        alpha=0.15,
        color=COLOR_GREEN,
        linewidth=0,
        label="Residual area",
    )

    # Add RMSE annotation
    ax2.text(
        0.98, 0.95, f"RMSE = {rmse:.4f}",
        transform=ax2.transAxes, fontsize=FONT_LEGEND,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor=COLOR_LIGHT_GRAY, alpha=0.95, linewidth=0.8),
    )

    ax2.set_xlabel("")
    ax2.set_ylabel("Residual", fontsize=FONT_LABEL)
    ax2.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.95)
    _tighten_xaxis(ax2, time)
    _style_axes(ax2, grid="major", minor_ticks=False)

    # ----- Panel 3: Latent heatmap -----
    ax3 = axes[2]
    if latent is not None and latent.size > 0:
        # latent shape: (T, D)
        # Use symmetric colormap centered at zero
        vmax = np.abs(latent).max()
        im = ax3.imshow(
            latent.T, aspect="auto", cmap="bwr",
            extent=[0, time[-1] * 60 * fs, latent.shape[1] - 0.5, -0.5],
            interpolation="nearest",
            vmin=-vmax, vmax=vmax,
        )
        ax3.set_xlabel("Time (min)", fontsize=FONT_LABEL)
        ax3.set_ylabel("Latent dimension", fontsize=FONT_LABEL)
        ax3.set_title("Latent Representation $\\mathbf{z}(t)$", fontsize=FONT_TITLE, fontweight="normal")

        _add_colorbar(fig, im, ax3, label="Value", shrink=0.8, pad=0.02)
        _style_axes(ax3, grid="major", minor_ticks=False)
        _tighten_xaxis(ax3, time)
    else:
        ax3.text(0.5, 0.5, "No latent data available",
                ha="center", va="center", fontsize=FONT_LABEL)
        ax3.axis("off")

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    warmup_steps: int = 30,
    filename: str = "temporal_accuracy.pdf",
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True)

    # ----- VAF plot -----
    marker_every = max(1, len(agg) // 20)
    ax1.plot(
        agg["timestep"], agg["vaf_mean"],
        color=COLOR_BLUE, linewidth=0.5, label="VAF mean",
        marker="o", markersize=1, markevery=marker_every,
        markerfacecolor=COLOR_BLUE, markeredgewidth=0,
    )
    ax1.fill_between(
        agg["timestep"],
        agg["vaf_mean"] - agg["vaf_std"],
        agg["vaf_mean"] + agg["vaf_std"],
        alpha=0.15, color=COLOR_BLUE, linewidth=0,
        label="VAF +/- 1 SD",
    )

    # Mark warmup region with subtle styling
    ax1.axvspan(0, warmup_steps, alpha=0.08, color=COLOR_GRAY,
               label="Warmup", zorder=0, linewidth=0)
    ax1.axvline(warmup_steps, color=COLOR_GRAY,
               linewidth=0.6, alpha=0.6, zorder=1)

    ax1.set_ylabel("Variance Accounted For", fontsize=FONT_LABEL)
    ax1.set_title("Reconstruction Quality vs. Timestep",
                 fontsize=FONT_TITLE, fontweight="normal", pad=8)
    ax1.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    _set_ylim_from_data(
        ax1,
        np.concatenate([agg["vaf_mean"].values, (agg["vaf_mean"] + agg["vaf_std"]).values]),
        min_zero=True,
        clamp=(0.0, 1.0),
    )
    _style_axes(ax1, grid="both", minor_ticks=True)
    _tighten_xaxis(ax1, agg["timestep"].values)

    # ----- SNR plot -----
    ax2.plot(
        agg["timestep"], agg["snr_mean"],
        color=COLOR_ORANGE, linewidth=0.5, label="SNR mean",
        marker="s", markersize=2.5, markevery=marker_every,
        markerfacecolor=COLOR_ORANGE, markeredgewidth=0,
    )
    ax2.fill_between(
        agg["timestep"],
        agg["snr_mean"] - agg["snr_std"],
        agg["snr_mean"] + agg["snr_std"],
        alpha=0.15, color=COLOR_ORANGE, linewidth=0,
        label="SNR +/- 1 SD",
    )

    ax2.axvspan(0, warmup_steps, alpha=0.08, color=COLOR_GRAY, zorder=0, linewidth=0)
    ax2.axvline(warmup_steps, color=COLOR_GRAY,
               linewidth=0.6, alpha=0.6, zorder=1)

    ax2.set_xlabel("Timestep (prediction index)", fontsize=FONT_LABEL)
    ax2.set_ylabel("Signal-to-Noise Ratio (dB)", fontsize=FONT_LABEL)
    ax2.legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax2, grid="both", minor_ticks=True)
    _set_ylim_from_data(
        ax2,
        np.concatenate([agg["snr_mean"].values, (agg["snr_mean"] + agg["snr_std"]).values]),
        pad_frac=0.05,
    )
    _tighten_xaxis(ax2, agg["timestep"].values)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_within_window_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    fs: float = 4.0,
    filename: str = "within_window_accuracy.pdf",
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
        figsize=(6.5, 2.4 * n_panels),
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
        label="MAE mean",
    )
    axes[0].fill_between(
        x,
        agg["mae_mean"] - agg["mae_std"],
        agg["mae_mean"] + agg["mae_std"],
        color=COLOR_BLUE,
        alpha=0.2,
        label="MAE +/- 1 SD",
    )

    axes[0].set_ylabel("Absolute error", fontsize=FONT_LABEL)
    axes[0].set_title("Within-Window Accuracy", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _set_ylim_from_data(
        axes[0],
        np.concatenate([agg["mae_mean"].values, (agg["mae_mean"] + agg["mae_std"]).values]),
        min_zero=True,
        pad_frac=0.05,
    )
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
            label="VAF mean",
        )
        axes[panel_idx].set_ylabel("VAF", fontsize=FONT_LABEL)
        _set_ylim_from_data(
            axes[panel_idx],
            agg["vaf_mean"].values,
            min_zero=True,
            clamp=(0.0, 1.0),
        )
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
            label="SNR mean",
        )
        axes[panel_idx].set_ylabel("SNR (dB)", fontsize=FONT_LABEL)
        axes[panel_idx].legend(loc="lower right", fontsize=FONT_LEGEND, framealpha=0.95)
        _style_axes(axes[panel_idx], grid="both", minor_ticks=True)
        _set_ylim_from_data(
            axes[panel_idx],
            agg["snr_mean"].values,
            pad_frac=0.05,
        )

    axes[-1].set_xlabel("Index from window start (samples)", fontsize=FONT_LABEL)
    for ax in axes:
        _tighten_xaxis(ax, x)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_trajectory(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "kld_trajectory.pdf",
    *,
    by_class: bool = False,
) -> None:
    """
    Plot KLD (transfer entropy) as a function of time before birth.

    If by_class is True and 'label' exists, plots separate lines per class.

    Args:
        df: DataFrame with 'hours_before', 'kld_mean', and optionally 'label'.
        output_dir: Directory to save the plot.
        filename: Output filename.
        by_class: Whether to plot class-wise trajectories (default False).

    Example:
        >>> plot_kld_trajectory(trajectory_df, Path("results/trajectory/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "hours_before" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    # Check if we have class labels and class-wise plotting is enabled
    has_labels = by_class and "label" in df.columns and df["label"].notna().any()
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

            color = label_colors.get(label, COLOR_GRAY)
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
                alpha=0.2,
                color=color,
                label=f"{name} +/- 1 SD",
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
            label="KLD mean",
        )
        ax.fill_between(
            agg["hour_bin"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.25,
            color=COLOR_BLUE,
            label="KLD +/- 1 SD",
        )

    ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
    ax.set_ylabel("KLD (Transfer Entropy)", fontsize=FONT_LABEL)
    ax.set_title("Transfer Entropy Evolution Before Delivery", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _tighten_xaxis(ax, agg["hour_bin"].values)
    ax.invert_xaxis()  # Time flows right-to-left toward birth
    ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_guid_trajectory(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    guid: str,
    filename: Optional[str] = None,
    show_std: bool = True,
) -> None:
    """
    Plot per-epoch KLD mean trajectory for a single GUID.

    Args:
        df: DataFrame filtered to a single GUID with 'hours_before', 'kld_mean',
            and optionally 'kld_std'.
        output_dir: Directory to save the plot.
        guid: GUID identifier for title/filename.
        filename: Output filename override (optional).
        show_std: Whether to plot +/- std if available.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty or "hours_before" not in df.columns or "kld_mean" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7.6, 2.8))

    plot_df = df.copy().sort_values("hours_before", ascending=False)
    x = plot_df["hours_before"].values
    y = plot_df["kld_mean"].values

    ax.plot(
        x,
        y,
        color=COLOR_BLUE,
        linewidth=0.8,
        marker="o",
        markersize=2.2,
        label="KLD mean",
    )

    if show_std and "kld_std" in plot_df.columns:
        std = plot_df["kld_std"].values
        ax.fill_between(
            x,
            y - std,
            y + std,
            color=COLOR_BLUE,
            alpha=0.18,
            label="KLD +/- 1 SD",
        )

    ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
    ax.set_ylabel("KLD (Transfer Entropy)", fontsize=FONT_LABEL)
    ax.set_title(f"KLD Trajectory - {guid}", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _tighten_xaxis(ax, x)
    _set_ylim_from_data(ax, y, pad_frac=0.08)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)

    fig.tight_layout()
    out_name = filename or f"kld_guid_{guid}.pdf"
    fig.savefig(output_dir / out_name, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coherence_analysis(
    frequencies: np.ndarray,
    coherence_original: np.ndarray,
    coherence_reconstructed: np.ndarray,
    output_dir: Path,
    filename: str = "coherence_analysis.pdf",
) -> None:
    """
    Plot optional UP-FHR coherence comparison between original and reconstructed signals.

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

    fig, ax = plt.subplots(figsize=(7.6, 2.6))

    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]
    coh_orig = coherence_original[freq_mask]
    coh_recon = coherence_reconstructed[freq_mask]

    ax.plot(
        freqs,
        coh_orig,
        color=COLOR_BLUE,
        linewidth=0.6,
        label="UP vs FHR (reference)",
    )
    ax.plot(
        freqs,
        coh_recon,
        color=COLOR_ORANGE,
        linewidth=0.9,
        label="UP vs FHR (reconstruction)",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title("UP-FHR Spectral Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, max_freq)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_coherence(
    frequencies: np.ndarray,
    coherence_mean: np.ndarray,
    coherence_std: np.ndarray,
    output_dir: Path,
    filename: str = "reconstruction_coherence.pdf",
    significance_threshold: Optional[float] = None,
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

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]
    coh_mean = coherence_mean[freq_mask]
    coh_std = coherence_std[freq_mask]

    ax.plot(freqs, coh_mean, color=COLOR_BLUE, linewidth=0.5, label="Mean coherence")
    ax.fill_between(
        freqs,
        coh_mean - coh_std,
        coh_mean + coh_std,
        color=COLOR_BLUE,
        alpha=0.2,
        label="Coherence +/- 1 SD",
    )
    if significance_threshold is not None and np.isfinite(significance_threshold):
        ax.axhline(
            float(significance_threshold),
            color=COLOR_ORANGE,
            linestyle="--",
            linewidth=0.9,
            label="Significance threshold",
        )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title("FHR Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, max_freq)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coherence_spectrum(
    frequencies: np.ndarray,
    coherence: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
    max_freq: Optional[float] = 0.5,
) -> None:
    """
    Plot a single coherence spectrum (one line, no aggregation).

    Args:
        frequencies: Frequency array in Hz.
        coherence: Coherence values in [0, 1].
        output_path: Output path for the figure.
        title: Optional title override.
        max_freq: Optional max frequency to display.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frequencies.size == 0 or coherence.size == 0:
        return

    if max_freq is None:
        max_freq = float(np.nanmax(frequencies))
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]
    coh = coherence[freq_mask]

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(freqs, coh, color=COLOR_BLUE, linewidth=0.8, label="Coherence")
    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title(title or "Coherence Spectrum", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, float(max_freq))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_spectra(
    frequencies: np.ndarray,
    early_mean: np.ndarray,
    early_std: np.ndarray,
    late_mean: np.ndarray,
    late_std: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
    max_freq: Optional[float] = 0.5,
    early_label: str = "Early horizon",
    late_label: str = "Late horizon",
) -> None:
    """
    Plot early vs late horizon coherence spectra with uncertainty bands.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frequencies.size == 0:
        return

    if max_freq is None:
        max_freq = float(np.nanmax(frequencies))
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]
    early_m = early_mean[freq_mask]
    late_m = late_mean[freq_mask]
    early_s = early_std[freq_mask] if early_std.size else np.zeros_like(early_m)
    late_s = late_std[freq_mask] if late_std.size else np.zeros_like(late_m)

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(freqs, early_m, color=COLOR_BLUE, linewidth=0.8, label=early_label)
    ax.fill_between(
        freqs,
        early_m - early_s,
        early_m + early_s,
        color=COLOR_BLUE,
        alpha=0.2,
        label=f"{early_label} +/- 1 SD",
    )
    ax.plot(freqs, late_m, color=COLOR_ORANGE, linewidth=0.9, linestyle="--", label=late_label)
    ax.fill_between(
        freqs,
        late_m - late_s,
        late_m + late_s,
        color=COLOR_ORANGE,
        alpha=0.15,
        label=f"{late_label} +/- 1 SD",
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Coherence", fontsize=FONT_LABEL)
    ax.set_title(title or "Early vs Late Horizon Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, float(max_freq))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_delta(
    frequencies: np.ndarray,
    delta_mean: np.ndarray,
    delta_std: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
    max_freq: Optional[float] = 0.5,
    label: str = "Late - Early",
) -> None:
    """
    Plot delta spectrum (late minus early) with uncertainty.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frequencies.size == 0:
        return

    if max_freq is None:
        max_freq = float(np.nanmax(frequencies))
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]
    delta_m = delta_mean[freq_mask]
    delta_s = delta_std[freq_mask] if delta_std.size else np.zeros_like(delta_m)

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    ax.plot(freqs, delta_m, color=COLOR_BLUE, linewidth=0.8, label=label)
    ax.fill_between(
        freqs,
        delta_m - delta_s,
        delta_m + delta_s,
        color=COLOR_BLUE,
        alpha=0.2,
        label=f"{label} +/- 1 SD",
    )
    ax.axhline(0.0, color=COLOR_BLACK, linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("Delta Coherence", fontsize=FONT_LABEL)
    ax.set_title(title or "Horizon Delta Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, float(max_freq))
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_time_frequency_map(
    frequencies: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    max_freq: Optional[float] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "Value",
) -> None:
    """
    Plot a generic time-frequency map (not constrained to coherence [0,1]).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frequencies.size == 0 or times.size == 0 or values.size == 0:
        return

    freq_mask = slice(None)
    if max_freq is not None:
        freq_mask = frequencies <= max_freq

    freqs = frequencies[freq_mask]
    vals = values[freq_mask, :]
    time_min = times / 60.0

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 2.6))
    im = ax.pcolormesh(
        time_min,
        freqs,
        vals,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
    ax.set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_title(title or "Time-Frequency Map", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    if freqs.size:
        ax.set_ylim(float(np.min(freqs)), float(np.max(freqs)))
    _add_colorbar(fig, im, ax, label=colorbar_label, shrink=0.8, pad=0.02)
    _style_axes(ax, grid="major")
    ax.margins(x=0.0, y=0.0)

    _tighten_xaxis(ax, time_min)
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
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
    filename: str = "psd_comparison.pdf",
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

    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    max_freq = min(0.5, float(np.nanmax(frequencies))) if frequencies.size else 0.5
    freq_mask = frequencies <= max_freq
    freqs = frequencies[freq_mask]

    def _to_db(x: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(x, 1e-12))

    def _band_db(mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        low = np.maximum(mean - std, 1e-12)
        high = np.maximum(mean + std, 1e-12)
        return _to_db(low), _to_db(high)

    orig_db = _to_db(psd_orig_mean)[freq_mask]
    recon_db = _to_db(psd_recon_mean)[freq_mask]

    # Plot with clean styling
    ax.plot(freqs, orig_db, color=COLOR_BLUE, linewidth=0.6,
           label="Original mean", zorder=3)
    ax.plot(
        freqs,
        recon_db,
        color=COLOR_ORANGE,
        linewidth=0.9,
        label="Reconstruction mean",
        zorder=3,
    )

    # Uncertainty bands with proper dB conversion
    orig_low, orig_high = _band_db(psd_orig_mean, psd_orig_std)
    recon_low, recon_high = _band_db(psd_recon_mean, psd_recon_std)
    ax.fill_between(freqs, orig_low[freq_mask], orig_high[freq_mask],
                    color=COLOR_BLUE, alpha=0.2, linewidth=0, zorder=1,
                    label="Original +/- 1 SD")
    ax.fill_between(freqs, recon_low[freq_mask], recon_high[freq_mask],
                    color=COLOR_ORANGE, alpha=0.2, linewidth=0, zorder=2,
                    label="Reconstruction +/- 1 SD")

    if psd_residual_mean is not None:
        resid_db = _to_db(psd_residual_mean)[freq_mask]
        ax.plot(freqs, resid_db, color=COLOR_GREEN, linewidth=0.5,
               label="Residual mean", zorder=3)
        if psd_residual_std is not None:
            resid_low, resid_high = _band_db(psd_residual_mean, psd_residual_std)
            ax.fill_between(
                freqs,
                resid_low[freq_mask],
                resid_high[freq_mask],
                color=COLOR_GREEN,
                alpha=0.2,
                linewidth=0,
                zorder=1,
                label="Residual +/- 1 SD",
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax.set_ylabel("PSD (dB/Hz)", fontsize=FONT_LABEL)
    ax.set_title("Power Spectral Density Comparison (Welch)", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.set_xlim(0, max_freq)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95)
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, freqs)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cross_correlation(
    lags_sec: np.ndarray,
    corr_mean: np.ndarray,
    corr_std: np.ndarray,
    output_dir: Path,
    filename: str = "cross_correlation.pdf",
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

    fig, ax = plt.subplots(figsize=(7.6, 2.4))

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
        label="Correlation +/- 1 SD",
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
    ax.set_title("Cross-Correlation Analysis", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.95, loc="best")
    _style_axes(ax, grid="both", minor_ticks=True)
    _tighten_xaxis(ax, lags_sec)

    fig.tight_layout(pad=0.6)
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
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 3.2), sharex=True)
        axes[0].plot(time_min, fhr_original, color=COLOR_BLUE, linewidth=0.6, label="Original FHR")
        axes[0].plot(
            time_min,
            fhr_reconstructed,
            color=COLOR_ORANGE,
            linewidth=0.9,
            label="Reconstructed FHR",
        )
        axes[0].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
        axes[0].set_title(title or "FHR Reconstruction", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        axes[0].legend(loc="upper right", fontsize=FONT_LEGEND, framealpha=0.95)

        residual = fhr_original - fhr_reconstructed
        axes[1].plot(time_min, residual, color=COLOR_GREEN, linewidth=0.6)
        axes[1].axhline(0.0, color=COLOR_BLACK, alpha=0.5, linewidth=0.7)
        axes[1].set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
        axes[1].set_ylabel("Residual", fontsize=FONT_LABEL)
        _style_axes(axes[0], grid="both", minor_ticks=True)
        _style_axes(axes[1], grid="major", minor_ticks=False)
        _tighten_xaxis(axes[0], time_min)
        _tighten_xaxis(axes[1], time_min)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.2), sharex=True)
        axes[0].plot(time_min, up_signal, color=COLOR_PURPLE, linewidth=0.5)
        axes[0].set_ylabel("UP (normalized)", fontsize=FONT_LABEL)
        axes[0].set_title(title or "UP and FHR Signals", fontsize=FONT_TITLE, fontweight="normal", pad=6)

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
        _tighten_xaxis(axes[0], time_min)
        _tighten_xaxis(axes[1], time_min)
        _tighten_xaxis(axes[2], time_min)

    fig.tight_layout(pad=0.6)
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
        fig, ax = plt.subplots(1, 1, figsize=(7.6, 2.6))
        im = ax.pcolormesh(time_min, freqs, coh_orig, shading="auto", cmap="bwr", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
        ax.set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
        ax.set_title("FHR Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        if freqs.size:
            ax.set_ylim(float(np.min(freqs)), float(np.max(freqs)))
        _add_colorbar(fig, im, ax, label="Coherence", shrink=0.8, pad=0.02)
        _style_axes(ax, grid="major")
        ax.margins(x=0.0, y=0.0)

        if title:
            fig.suptitle(title, fontsize=FONT_TITLE, y=0.98)

        _tighten_xaxis(ax, time_min)
        fig.tight_layout(pad=0.6)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    coh_recon = coherence_reconstructed[freq_mask, :]
    coh_diff = coh_recon - coh_orig

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 4.8), sharex=True)

    im0 = axes[0].pcolormesh(time_min, freqs, coh_orig, shading="auto", cmap="bwr", vmin=0.0, vmax=1.0)
    axes[0].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[0].set_title("Reference Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    if freqs.size:
        axes[0].set_ylim(float(np.min(freqs)), float(np.max(freqs)))
    _add_colorbar(fig, im0, axes[0], label="Coherence", shrink=0.8, pad=0.02)

    im1 = axes[1].pcolormesh(time_min, freqs, coh_recon, shading="auto", cmap="bwr", vmin=0.0, vmax=1.0)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[1].set_title("Reconstruction Coherence", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    if freqs.size:
        axes[1].set_ylim(float(np.min(freqs)), float(np.max(freqs)))
    _add_colorbar(fig, im1, axes[1], label="Coherence", shrink=0.8, pad=0.02)

    im2 = axes[2].pcolormesh(time_min, freqs, coh_diff, shading="auto", cmap="bwr", vmin=-1.0, vmax=1.0)
    axes[2].set_xlabel("Time (minutes)", fontsize=FONT_LABEL)
    axes[2].set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes[2].set_title("Coherence Difference (Reconstruction - Reference)", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    if freqs.size:
        axes[2].set_ylim(float(np.min(freqs)), float(np.max(freqs)))
    _add_colorbar(fig, im2, axes[2], label="Delta", shrink=0.8, pad=0.02)

    for ax in axes:
        _style_axes(ax, grid="major")
        _tighten_xaxis(ax, time_min)
        ax.margins(x=0.0, y=0.0)

    if title:
        fig.suptitle(title, fontsize=FONT_TITLE, y=0.98)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Trajectory Visualization Functions
# -----------------------------------------------------------------------------

def plot_guid_absolute_trajectory(
    df: pd.DataFrame,
    output_path: Path,
    *,
    guid: str,
    x_col: str,
    y_col: str,
    color_by: str = "t_abs_sec",
    show_epoch_boundaries: bool = True,
) -> None:
    """
    Plot concatenated latent trajectory for a GUID, ordered by absolute time.

    Args:
        df: DataFrame containing a single GUID's latent rows with time fields.
        output_path: Output path for the figure.
        guid: GUID identifier for the title.
        x_col: Column name for x-axis coordinates.
        y_col: Column name for y-axis coordinates.
        color_by: Column name for color progression (default: 't_abs_sec').
        show_epoch_boundaries: Mark transitions between epochs if available.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    sort_col = color_by if color_by in df.columns else "t_abs_sec"
    if sort_col not in df.columns:
        sort_col = "t_sec" if "t_sec" in df.columns else None
    if sort_col is None:
        return

    sub = df.sort_values(sort_col)
    X = sub[[x_col, y_col]].to_numpy()
    if len(X) < 2:
        return

    color_vals = sub[sort_col].to_numpy()
    points = X.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="bwr", linewidths=2.0)
    lc.set_array(color_vals[:-1])

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.add_collection(lc)
    ax.scatter(X[0, 0], X[0, 1], s=80, marker="o", color=COLOR_GREEN, edgecolors=COLOR_BLACK, linewidth=0.6, label="Start", zorder=5)
    ax.scatter(X[-1, 0], X[-1, 1], s=90, marker="X", color=COLOR_VERMILLION, edgecolors=COLOR_BLACK, linewidth=0.6, label="End", zorder=5)

    if show_epoch_boundaries and "epoch_sec" in sub.columns:
        epochs = sub["epoch_sec"].to_numpy()
        for idx in range(1, len(epochs)):
            if epochs[idx] != epochs[idx - 1]:
                ax.scatter(
                    X[idx, 0],
                    X[idx, 1],
                    s=28,
                    marker="s",
                    color=COLOR_LIGHT_GRAY,
                    edgecolors=COLOR_GRAY,
                    linewidth=0.4,
                )

    if "t_abs_sec" in sub.columns:
        abs_times = sub["t_abs_sec"].to_numpy()
        if abs_times.size:
            zero_idx = int(np.argmin(np.abs(abs_times)))
            ax.scatter(
                X[zero_idx, 0],
                X[zero_idx, 1],
                s=50,
                marker="^",
                color=COLOR_ORANGE,
                edgecolors=COLOR_BLACK,
                linewidth=0.4,
                label="t~0",
            )

    ax.set_xlabel(x_col.upper(), fontsize=FONT_LABEL)
    ax.set_ylabel(y_col.upper(), fontsize=FONT_LABEL)
    ax.set_title(f"Absolute Trajectory - {guid}", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _style_axes(ax, grid="major", minor_ticks=False)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)

    _add_colorbar(fig, lc, ax, label=color_by, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

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
        >>> plot_latent_trajectory_2d(trajectory_2d, Path("results/traj_2d.pdf"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    time_steps = trajectory.shape[0]

    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    # Color by time — bwr colormap: blue → white → red
    if color_by_time:
        colors = plt.cm.bwr(np.linspace(0, 1, time_steps))
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1],
            c=np.arange(time_steps), cmap="bwr",
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
                    lw=1.0,
                    color=colors[i] if color_by_time else COLOR_BLUE,
                    alpha=0.7,
                ),
            )

    # Mark start and end
    ax.scatter(
        trajectory[0, 0], trajectory[0, 1],
        c=COLOR_GREEN, s=80, marker="o", label="Start",
        zorder=5, edgecolor=COLOR_BLACK, linewidth=0.6
    )
    ax.scatter(
        trajectory[-1, 0], trajectory[-1, 1],
        c=COLOR_VERMILLION, s=80, marker="X", label="End",
        zorder=5, edgecolor=COLOR_BLACK, linewidth=0.6
    )

    if color_by_time:
        _add_colorbar(fig, scatter, ax, label="Time Step", shrink=0.8, pad=0.02)

    ax.set_xlabel("Latent Dim 1", fontsize=FONT_LABEL)
    ax.set_ylabel("Latent Dim 2", fontsize=FONT_LABEL)
    ax.set_title(f"Latent Trajectory - {sample_id}", fontsize=FONT_TITLE, fontweight="normal", pad=6)
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
        >>> plot_latent_trajectory_3d(trajectory_3d, Path("results/traj_3d.pdf"))
    """
    from mpl_toolkits.mplot3d import Axes3D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    time_steps = trajectory.shape[0]

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    # Color by time — bwr colormap: blue → white → red
    if color_by_time:
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=np.arange(time_steps), cmap="bwr",
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
        color=COLOR_GRAY, alpha=0.5, linewidth=0.8
    )

    # Mark start and end
    ax.scatter(
        [trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
        c=COLOR_GREEN, s=80, marker="o", label="Start", depthshade=False
    )
    ax.scatter(
        [trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
        c=COLOR_VERMILLION, s=80, marker="X", label="End", depthshade=False
    )

    if color_by_time:
        _add_colorbar(fig, scatter, ax, label="Time Step", shrink=0.8, pad=0.02)

    ax.set_xlabel("Latent Dim 1", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Latent Dim 2", fontsize=FONT_LABEL, labelpad=10)
    ax.set_zlabel("Latent Dim 3", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(f"Latent Trajectory 3D - {sample_id}", fontsize=FONT_TITLE, fontweight="normal", pad=10)
    ax.legend(loc="best", fontsize=FONT_LEGEND)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_guid_trajectory_3d(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "guid",
    time_axis: Optional[np.ndarray] = None,
    epoch_boundaries: Optional[list] = None,
    point_size: int = 10,
    cmap: str = "bwr",
) -> None:
    """
    Plot a full stitched GUID-level 3D trajectory with temporal coloring.

    Shows how a patient's latent trajectory evolves over the full recording
    (potentially many hours), with epoch boundaries marked.

    Args:
        trajectory: Stitched trajectory array of shape (T_total, 3).
        output_path: Path to save the figure.
        sample_id: GUID or identifier for the title.
        time_axis: Absolute time values of shape (T_total,) in seconds.
            If provided, colorbar shows hours before birth.
        epoch_boundaries: Indices where new epochs start (marked with diamonds).
        point_size: Size of trajectory points.
        cmap: Colormap name (default 'bwr').
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)
    if trajectory.shape[1] < 3:
        return

    T = trajectory.shape[0]

    # Determine color values and label
    if time_axis is not None and len(time_axis) == T:
        color_vals = np.abs(time_axis) / 3600.0  # hours before birth
        cbar_label = "Hours before birth"
    else:
        color_vals = np.arange(T)
        cbar_label = "Time Step"

    n_epochs = len(epoch_boundaries) if epoch_boundaries else 1
    if time_axis is not None and len(time_axis) > 1:
        hours = abs(float(time_axis[-1]) - float(time_axis[0])) / 3600.0
    else:
        hours = T * 4.0 / 3600.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Main scatter
    scatter = ax.scatter(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        c=color_vals, cmap=cmap, s=point_size, depthshade=True,
    )

    # Trajectory line
    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color=COLOR_GRAY, alpha=0.4, linewidth=0.6,
    )

    # Epoch boundaries
    if epoch_boundaries:
        for eb in epoch_boundaries:
            if 0 <= eb < T:
                ax.scatter(
                    [trajectory[eb, 0]], [trajectory[eb, 1]], [trajectory[eb, 2]],
                    c=COLOR_GRAY, s=20, marker="D", alpha=0.6, depthshade=False,
                )

    # Start and end markers
    ax.scatter(
        [trajectory[0, 0]], [trajectory[0, 1]], [trajectory[0, 2]],
        c=COLOR_GREEN, s=80, marker="o", label="Start", depthshade=False,
    )
    ax.scatter(
        [trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]],
        c=COLOR_VERMILLION, s=80, marker="X", label="End", depthshade=False,
    )

    _add_colorbar(fig, scatter, ax, label=cbar_label, shrink=0.8, pad=0.02)

    ax.set_xlabel("PC1", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("PC2", fontsize=FONT_LABEL, labelpad=10)
    ax.set_zlabel("PC3", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(
        f"GUID Trajectory 3D - {sample_id} ({n_epochs} epochs, {hours:.1f}h)",
        fontsize=FONT_TITLE, fontweight="normal", pad=10,
    )
    ax.legend(loc="best", fontsize=FONT_LEGEND)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_trajectory_3d(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    color_by_time: bool = True,
    point_size: int = 12,
) -> None:
    """
    Plot 3D KLD trajectory (KLD, KLD velocity, KLD acceleration).

    Args:
        trajectory: Array of shape (T, 3) with columns [kld, kld_velocity, kld_accel].
        output_path: Path to save the figure.
        sample_id: Sample identifier for the title.
        color_by_time: Whether to color points by time progression.
        point_size: Size of trajectory points.
    """
    from mpl_toolkits.mplot3d import Axes3D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if trajectory.ndim == 3:
        trajectory = trajectory.squeeze(0)

    if trajectory.shape[1] < 3:
        return

    time_steps = trajectory.shape[0]

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    if color_by_time:
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=np.arange(time_steps), cmap="cividis",
            s=point_size, depthshade=True
        )
    else:
        scatter = ax.scatter(
            trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            c=COLOR_BLUE, s=point_size, depthshade=True
        )

    ax.plot(
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        color=COLOR_BLUE, alpha=0.4, linewidth=0.5
    )

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

    ax.set_xlabel("KLD", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("KLD Velocity", fontsize=FONT_LABEL, labelpad=10)
    ax.set_zlabel("KLD Acceleration", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(f"KLD Trajectory 3D - {sample_id}", fontsize=FONT_TITLE, fontweight="normal", pad=10)
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
    cmap: str = "bwr",
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
        >>> plot_latent_changepoints_with_raw(latent, fhr, cp_results, Path("results/cp.pdf"))
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
    axes[0].set_title(f"Latent Representation - {sample_id}", fontsize=FONT_TITLE, fontweight="normal", pad=6)

    for raw_cp in raw_cps_from_latent:
        axes[0].axvline(raw_cp, color="white", linewidth=0.7, alpha=0.8)

    cbar = _add_colorbar(fig, im, axes[0], label="Activation", shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Activation", rotation=270, labelpad=12, fontsize=plt.rcParams["axes.labelsize"])

    # Panel 2: FHR with latent-derived changepoints
    time_axis = np.arange(raw_len)
    axes[1].plot(time_axis, fhr, color=COLOR_VERMILLION, linewidth=0.5, label="FHR")
    axes[1].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    axes[1].set_title("FHR with Latent Changepoints", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    axes[1].set_xlim(0, x_max)
    _style_axes(axes[1], grid="major", minor_ticks=False)
    axes[1].legend(loc="upper right", fontsize=FONT_LEGEND)

    for raw_cp in raw_cps_from_latent:
        axes[1].axvline(raw_cp, color=COLOR_BLACK, linewidth=0.5, alpha=0.6)

    # Panel 3: FHR with both latent and raw changepoints
    axes[2].plot(time_axis, fhr, color=COLOR_VERMILLION, linewidth=0.5, label="FHR")
    axes[2].set_xlabel("Raw Time Index", fontsize=FONT_LABEL)
    axes[2].set_ylabel("FHR (normalized)", fontsize=FONT_LABEL)
    axes[2].set_title("FHR with Latent (gray) and Raw (green) Changepoints", fontsize=FONT_TITLE, fontweight="normal", pad=6)
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
        ax.set_title("Segment Duration Distribution", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_duration_hist.pdf", dpi=SAVE_DPI, bbox_inches="tight")
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
        ax.set_title("Latent Speed vs Start Time", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        ax.axvline(0.0, color=COLOR_GRAY, linewidth=0.6, alpha=0.6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_speed_vs_start.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    # Plot 3: Dominant latent dimension counts
    dominant_dims = df["dominant_latent_dim"].dropna()
    if not dominant_dims.empty:
        dominant_counts = dominant_dims.astype(int).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        ax.bar(dominant_counts.index.astype(str), dominant_counts.values, color=COLOR_GREEN, alpha=0.8, edgecolor=COLOR_BLACK, linewidth=0.5)
        ax.set_xlabel("Latent Dimension Index", fontsize=FONT_LABEL)
        ax.set_ylabel("Segment Count", fontsize=FONT_LABEL)
        ax.set_title("Dominant Latent Dimension", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_dominant_dim.pdf", dpi=SAVE_DPI, bbox_inches="tight")
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
        ax.set_title("Segments per Sample", fontsize=FONT_TITLE, fontweight="normal", pad=6)
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_TICK)
        _style_axes(ax, grid="major", minor_ticks=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{filename_prefix}_per_sample.pdf", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)

    return df


def plot_trajectory_comparison(
    trajectories: Dict[str, np.ndarray],
    output_dir: Path,
    *,
    n_components: int = 2,
    filename: str = "trajectory_comparison.pdf",
    per_class_cmaps: Optional[Dict[str, str]] = None,
) -> None:
    """
    Compare latent trajectories across multiple classes in a single plot.

    Args:
        trajectories: Dict mapping class names to trajectory arrays.
            Each array has shape (N, T, D) where N is samples, T is time, D is dims.
        output_dir: Directory to save the plot.
        n_components: Number of dimensions to plot (2 or 3).
        filename: Output filename.
        per_class_cmaps: Dict mapping class names to colormap names for temporal
            coloring within each class. Default: Greens/Oranges/Blues.

    Example:
        >>> trajectories = {"healthy": healthy_trajs, "acidosis": acidosis_trajs}
        >>> plot_trajectory_comparison(trajectories, Path("results/"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default per-class colormaps for temporal coloring
    default_cmaps = {"healthy": "Greens", "acidosis": "Oranges", "hie": "Blues"}
    if per_class_cmaps is None:
        per_class_cmaps = default_cmaps

    # Flat color fallbacks for classes without a colormap
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
        cmap_name = per_class_cmaps.get(class_name.lower())
        flat_color = class_colors.get(class_name.lower(), COLOR_BLUE)

        if class_trajectories.ndim == 2:
            class_trajectories = class_trajectories[None, ...]

        for i, traj in enumerate(class_trajectories):
            T = traj.shape[0]

            # Use per-class colormap for temporal coloring if available
            if cmap_name is not None and T > 1:
                cmap_obj = plt.get_cmap(cmap_name)
                traj_colors = cmap_obj(np.linspace(0.3, 0.9, T))
            else:
                traj_colors = None

            alpha = 0.3 + 0.5 * (i == 0)

            if n_components == 3 and traj.shape[1] >= 3:
                if traj_colors is not None:
                    # Temporal coloring: scatter with colormap
                    ax.scatter(
                        traj[:, 0], traj[:, 1], traj[:, 2],
                        c=traj_colors, s=8, alpha=alpha, depthshade=True,
                        label=class_name if i == 0 else None,
                    )
                    ax.plot(
                        traj[:, 0], traj[:, 1], traj[:, 2],
                        color=flat_color, alpha=alpha * 0.5, linewidth=0.5,
                    )
                else:
                    ax.plot(
                        traj[:, 0], traj[:, 1], traj[:, 2],
                        color=flat_color, alpha=alpha, linewidth=0.7,
                        label=class_name if i == 0 else None,
                    )
                ax.scatter(
                    [traj[0, 0]], [traj[0, 1]], [traj[0, 2]],
                    c=flat_color, s=25, marker="o", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2,
                )
                ax.scatter(
                    [traj[-1, 0]], [traj[-1, 1]], [traj[-1, 2]],
                    c=flat_color, s=25, marker="X", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2,
                )
            else:
                if traj_colors is not None:
                    ax.scatter(
                        traj[:, 0], traj[:, 1],
                        c=traj_colors, s=8, alpha=alpha,
                        label=class_name if i == 0 else None,
                    )
                    ax.plot(
                        traj[:, 0], traj[:, 1],
                        color=flat_color, alpha=alpha * 0.5, linewidth=0.5,
                    )
                else:
                    ax.plot(
                        traj[:, 0], traj[:, 1],
                        color=flat_color, alpha=alpha, linewidth=0.7,
                        label=class_name if i == 0 else None,
                    )
                ax.scatter(
                    traj[0, 0], traj[0, 1],
                    c=flat_color, s=25, marker="o", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2,
                )
                ax.scatter(
                    traj[-1, 0], traj[-1, 1],
                    c=flat_color, s=25, marker="X", alpha=alpha, edgecolors=COLOR_BLACK, linewidths=0.2,
                )

    if n_components == 3:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL, labelpad=10)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL, labelpad=10)
        ax.set_zlabel("PC3", fontsize=FONT_LABEL, labelpad=10)
    else:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL)
        _style_axes(ax, grid="major", minor_ticks=False)

    ax.set_title("Trajectory Comparison by Class", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_recurrence(
    trajectory: np.ndarray,
    output_path: Path,
    *,
    sample_id: str = "sample",
    threshold: Optional[float] = None,
) -> None:
    """
    Plot a recurrence plot (pairwise distance matrix) for a single trajectory.

    Reveals periodicity, attractor structure, and regime transitions.

    Args:
        trajectory: Array of shape (T, D) — latent trajectory points.
        output_path: Path to save the figure.
        sample_id: Label for the plot title.
        threshold: If provided, binarize the distance matrix at this threshold.
            If None, show continuous distances as a heatmap.
    """
    from scipy.spatial.distance import pdist, squareform

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    T = trajectory.shape[0]
    if T < 3:
        return

    # Compute pairwise distance matrix
    dist_matrix = squareform(pdist(trajectory, metric="euclidean"))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    if threshold is not None:
        # Binary recurrence plot
        recurrence = (dist_matrix <= threshold).astype(float)
        im = ax.imshow(recurrence, cmap="Greys", origin="lower", aspect="equal")
        _add_colorbar(fig, im, ax, label="Recurrence")
    else:
        # Continuous distance heatmap — inferno for high contrast
        im = ax.imshow(dist_matrix, cmap="inferno", origin="lower", aspect="equal")
        _add_colorbar(fig, im, ax, label="L2 distance")

    ax.set_xlabel("Timestep", fontsize=FONT_LABEL)
    ax.set_ylabel("Timestep", fontsize=FONT_LABEL)
    ax.set_title(f"Recurrence Plot — {sample_id}", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _style_axes(ax, grid="none", minor_ticks=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: list,
    output_path: Path,
    *,
    class_col: str = "class",
) -> None:
    """
    Box/violin plots of trajectory features grouped by class.

    Args:
        df: DataFrame with feature columns and a class column.
        feature_cols: List of feature column names to plot.
        output_path: Path to save the figure.
        class_col: Column name for class labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_features = len(feature_cols)
    if n_features == 0:
        return

    cols = 4
    rows = math.ceil(n_features / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = np.atleast_2d(axes).reshape(-1)

    classes = sorted(df[class_col].unique())
    class_colors_map = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
    }

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        if feat not in df.columns:
            ax.set_visible(False)
            continue

        # Skip features with all NaN or inf
        valid = df[feat].replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            ax.set_visible(False)
            continue

        data_by_class = []
        labels = []
        colors = []
        for cls in classes:
            vals = df.loc[df[class_col] == cls, feat].replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size > 0:
                data_by_class.append(vals)
                labels.append(cls)
                colors.append(class_colors_map.get(cls.lower(), COLOR_BLUE))

        if not data_by_class:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(data_by_class, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(feat, fontsize=FONT_LABEL, fontweight="normal")
        ax.tick_params(axis="x", rotation=30, labelsize=FONT_TICK)
        _style_axes(ax, grid="major", minor_ticks=False)

    # Hide unused axes
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Trajectory Features by Class", fontsize=FONT_TITLE, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Class Comparison Visualizations
# ---------------------------------------------------------------------------

def plot_class_mean_trajectory(
    trajectories_by_class: Dict[str, list],
    output_path: Path,
    *,
    n_components: int = 3,
) -> None:
    """
    Plot mean trajectory per class with ±1 std confidence ribbon.

    Shows the "average journey" through latent space for each class with
    shaded uncertainty regions, revealing systematic differences in how
    healthy vs pathological babies evolve.

    Args:
        trajectories_by_class: Dict mapping class names to lists of
            trajectory arrays, each shape (T, D). Trajectories within
            a class may have different lengths — they are resampled to
            a common grid before averaging.
        output_path: Path to save figure.
        n_components: 2 or 3 PCs to plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_colors = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
        "unknown": COLOR_GRAY,
    }

    def _resample_to_grid(trajs: list, n_points: int = 100) -> np.ndarray:
        """Resample variable-length trajectories to a common grid."""
        resampled = []
        for traj in trajs:
            T = traj.shape[0]
            if T < 2:
                continue
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, n_points)
            interp = np.column_stack(
                [np.interp(new_t, old_t, traj[:, d]) for d in range(traj.shape[1])]
            )
            resampled.append(interp)
        return np.array(resampled) if resampled else np.empty((0, n_points, 0))

    if n_components == 3:
        fig = plt.figure(figsize=(9.0, 7.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for class_name, traj_list in trajectories_by_class.items():
        if not traj_list:
            continue

        n_dims = min(traj_list[0].shape[1], n_components)
        stacked = _resample_to_grid(traj_list, n_points=100)

        if stacked.shape[0] < 2:
            # Only 1 trajectory — plot without confidence
            traj = traj_list[0]
            color = class_colors.get(class_name.lower(), COLOR_BLUE)
            if n_components == 3 and n_dims >= 3:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                        color=color, linewidth=1.5, label=f"{class_name} (n=1)")
            else:
                ax.plot(traj[:, 0], traj[:, 1],
                        color=color, linewidth=1.5, label=f"{class_name} (n=1)")
            continue

        mean_traj = stacked.mean(axis=0)  # (100, D)
        std_traj = stacked.std(axis=0)    # (100, D)
        color = class_colors.get(class_name.lower(), COLOR_BLUE)
        n_trajs = stacked.shape[0]

        if n_components == 3 and n_dims >= 3:
            ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2],
                    color=color, linewidth=2.0, label=f"{class_name} (n={n_trajs})")
            # Mark start and end of mean
            ax.scatter([mean_traj[0, 0]], [mean_traj[0, 1]], [mean_traj[0, 2]],
                       c=color, s=40, marker="o", edgecolors=COLOR_BLACK, linewidths=0.5, zorder=5)
            ax.scatter([mean_traj[-1, 0]], [mean_traj[-1, 1]], [mean_traj[-1, 2]],
                       c=color, s=40, marker="X", edgecolors=COLOR_BLACK, linewidths=0.5, zorder=5)
            # Plot individual trajectories faintly
            for i in range(min(n_trajs, 8)):
                ax.plot(stacked[i, :, 0], stacked[i, :, 1], stacked[i, :, 2],
                        color=color, alpha=0.15, linewidth=0.5)
        else:
            ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                    color=color, linewidth=2.0, label=f"{class_name} (n={n_trajs})")
            # Confidence ribbon using distance from mean
            for alpha, n_std in [(0.2, 1.0), (0.08, 2.0)]:
                ax.fill_between(
                    mean_traj[:, 0],
                    mean_traj[:, 1] - n_std * std_traj[:, 1],
                    mean_traj[:, 1] + n_std * std_traj[:, 1],
                    color=color, alpha=alpha,
                )
            ax.scatter(mean_traj[0, 0], mean_traj[0, 1],
                       c=color, s=40, marker="o", edgecolors=COLOR_BLACK, linewidths=0.5, zorder=5)
            ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1],
                       c=color, s=40, marker="X", edgecolors=COLOR_BLACK, linewidths=0.5, zorder=5)

    if n_components == 3:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL, labelpad=10)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL, labelpad=10)
        ax.set_zlabel("PC3", fontsize=FONT_LABEL, labelpad=10)
    else:
        ax.set_xlabel("PC1", fontsize=FONT_LABEL)
        ax.set_ylabel("PC2", fontsize=FONT_LABEL)
        _style_axes(ax, grid="major", minor_ticks=False)

    ax.set_title("Mean Trajectory by Class (±1 std)", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.legend(loc="best", fontsize=FONT_LEGEND, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_latent_density(
    latent_df: pd.DataFrame,
    output_path: Path,
    *,
    pc_x: str = "pc1",
    pc_y: str = "pc2",
    label_col: str = "label",
) -> None:
    """
    2D scatter with KDE contours showing class-level latent density separation.

    Reveals whether healthy vs HIE babies occupy distinct regions of latent
    space or overlap significantly.

    Args:
        latent_df: DataFrame with pc1, pc2, and label columns.
        output_path: Path to save figure.
        pc_x: Column name for x-axis PC.
        pc_y: Column name for y-axis PC.
        label_col: Column name for class label.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pc_x not in latent_df.columns or pc_y not in latent_df.columns:
        return

    class_colors = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
        "unknown": COLOR_GRAY,
    }

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    labels = [l for l in latent_df[label_col].unique() if l != "unknown"]
    for label in sorted(labels):
        subset = latent_df[latent_df[label_col] == label]
        x = subset[pc_x].dropna().values
        y = subset[pc_y].dropna().values

        if len(x) < 10:
            continue

        color = class_colors.get(label.lower(), COLOR_BLUE)

        # Scatter points
        ax.scatter(x, y, c=color, s=3, alpha=0.15, rasterized=True)

        # KDE contours
        try:
            from scipy.stats import gaussian_kde
            # Subsample for KDE if too many points
            if len(x) > 5000:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(x), 5000, replace=False)
                x_kde, y_kde = x[idx], y[idx]
            else:
                x_kde, y_kde = x, y

            kde = gaussian_kde(np.vstack([x_kde, y_kde]))
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            margin = 0.1
            xr = xmax - xmin
            yr = ymax - ymin
            xx, yy = np.meshgrid(
                np.linspace(xmin - margin * xr, xmax + margin * xr, 100),
                np.linspace(ymin - margin * yr, ymax + margin * yr, 100),
            )
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contour(xx, yy, zz, levels=5, colors=[color], linewidths=0.8, alpha=0.7)
            ax.contourf(xx, yy, zz, levels=5, colors=[color],
                        alpha=np.linspace(0.0, 0.15, 6))
        except Exception:
            pass  # KDE can fail with degenerate data

        # Add class label at centroid
        ax.annotate(
            label.capitalize(), xy=(np.median(x), np.median(y)),
            fontsize=FONT_LABEL, fontweight="bold", color=color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8),
        )

    ax.set_xlabel(pc_x.upper(), fontsize=FONT_LABEL)
    ax.set_ylabel(pc_y.upper(), fontsize=FONT_LABEL)
    ax.set_title(f"Latent Space Density — {pc_x.upper()} vs {pc_y.upper()}", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_pc_evolution(
    latent_df: pd.DataFrame,
    output_path: Path,
    *,
    label_col: str = "label",
    time_col: str = "t_abs_sec",
    n_pcs: int = 3,
    bin_minutes: float = 30.0,
) -> None:
    """
    Per-PC temporal evolution by class (mean ± std in time bins).

    Shows how each principal component changes over time for each class,
    revealing whether classes diverge, converge, or remain separated.

    Args:
        latent_df: DataFrame with pc1..pcN, label, and time columns.
        output_path: Path to save figure.
        label_col: Column name for class labels.
        time_col: Column name for absolute time in seconds.
        n_pcs: Number of principal components to plot.
        bin_minutes: Time bin width in minutes for aggregation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pc_cols = [f"pc{i}" for i in range(1, n_pcs + 1) if f"pc{i}" in latent_df.columns]
    if not pc_cols or time_col not in latent_df.columns or label_col not in latent_df.columns:
        return

    class_colors = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
        "unknown": COLOR_GRAY,
    }

    labels = [l for l in latent_df[label_col].unique() if l != "unknown"]
    if not labels:
        return

    n_cols = len(pc_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.5), squeeze=False)
    axes = axes[0]

    # Convert time to hours before birth
    df = latent_df.copy()
    df["hours_before"] = -df[time_col] / 3600.0
    bin_hours = bin_minutes / 60.0
    df["time_bin"] = (df["hours_before"] / bin_hours).round() * bin_hours

    for col_idx, pc_col in enumerate(pc_cols):
        ax = axes[col_idx]

        for label in sorted(labels):
            subset = df[df[label_col] == label]
            color = class_colors.get(label.lower(), COLOR_BLUE)

            agg = subset.groupby("time_bin")[pc_col].agg(["mean", "std", "count"]).reset_index()
            agg = agg[agg["count"] >= 3]  # Require at least 3 points per bin
            agg = agg.sort_values("time_bin")

            if agg.empty:
                continue

            ax.plot(agg["time_bin"], agg["mean"], color=color, linewidth=1.5,
                    label=label.capitalize(), marker=".", markersize=3)
            ax.fill_between(
                agg["time_bin"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
        ax.set_ylabel(pc_col.upper(), fontsize=FONT_LABEL)
        ax.set_title(f"{pc_col.upper()} by Class", fontsize=FONT_LABEL + 1, fontweight="normal")
        ax.invert_xaxis()
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.9)
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle("Principal Component Evolution by Class", fontsize=FONT_TITLE, fontweight="normal", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_class_dynamics_comparison(
    latent_df: pd.DataFrame,
    output_path: Path,
    *,
    label_col: str = "label",
) -> None:
    """
    Violin/box plots comparing latent dynamics (speed, acceleration) by class.

    Shows whether pathological babies have different trajectory dynamics
    (faster/slower transitions, more/less acceleration) than healthy ones.

    Args:
        latent_df: DataFrame with speed, accel, and label columns.
        output_path: Path to save figure.
        label_col: Column name for class labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_cols = [c for c in ["speed", "accel", "kld_velocity"] if c in latent_df.columns]
    if not dynamic_cols or label_col not in latent_df.columns:
        return

    class_colors = {
        "healthy": COLOR_GREEN,
        "acidosis": COLOR_VERMILLION,
        "hie": COLOR_PURPLE,
        "unknown": COLOR_GRAY,
    }

    labels = sorted([l for l in latent_df[label_col].unique() if l != "unknown"])
    if not labels:
        return

    n_cols = len(dynamic_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4.0), squeeze=False)
    axes = axes[0]

    for col_idx, dyn_col in enumerate(dynamic_cols):
        ax = axes[col_idx]

        data_by_class = []
        colors = []
        for label in labels:
            vals = latent_df.loc[
                latent_df[label_col] == label, dyn_col
            ].replace([np.inf, -np.inf], np.nan).dropna().values
            if vals.size > 0:
                data_by_class.append(vals)
                colors.append(class_colors.get(label.lower(), COLOR_BLUE))

        if not data_by_class:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(data_by_class, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.5)
        for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(COLOR_BLACK)
                parts[key].set_linewidth(0.6)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([l.capitalize() for l in labels], fontsize=FONT_TICK)

        # Clean up column name for title
        titles = {"speed": "Latent Speed", "accel": "Latent Acceleration", "kld_velocity": "KLD Velocity"}
        ax.set_title(titles.get(dyn_col, dyn_col), fontsize=FONT_LABEL + 1, fontweight="normal")
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle("Trajectory Dynamics by Class", fontsize=FONT_TITLE, fontweight="normal", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_significance_heatmap(
    dim_comparison_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Heatmap of per-latent-dimension p-values between class pairs.

    Identifies which latent dimensions carry the most discriminative
    information between classes. Darker cells = more significant differences.

    Args:
        dim_comparison_df: DataFrame from compare_dimensions_by_class()
            with columns: dimension, pair, p_value, effect_size_r.
        output_path: Path to save figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dim_comparison_df.empty:
        return

    # Pivot to get dimensions x pairs matrix
    pivot = dim_comparison_df.pivot_table(
        index="dimension", columns="pair", values="p_value", aggfunc="first"
    )

    # Sort dimensions numerically
    def _dim_sort_key(d: str) -> int:
        try:
            return int(d.replace("z", ""))
        except ValueError:
            return 999
    pivot = pivot.reindex(sorted(pivot.index, key=_dim_sort_key))

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2.5), max(4, len(pivot) * 0.35)))

    # Use -log10(p) for better visual contrast
    display_vals = -np.log10(pivot.values.astype(float) + 1e-300)

    im = ax.imshow(display_vals, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    _add_colorbar(fig, im, ax, label="-log₁₀(p-value)")

    # Add text annotations
    for i in range(display_vals.shape[0]):
        for j in range(display_vals.shape[1]):
            p_val = pivot.values[i, j]
            if np.isfinite(p_val):
                text = f"{p_val:.1e}"
                text_color = "white" if display_vals[i, j] > display_vals.max() * 0.6 else COLOR_BLACK
                ax.text(j, i, text, ha="center", va="center", fontsize=5.5, color=text_color)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=FONT_TICK, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=FONT_TICK)
    ax.set_xlabel("Class Pair", fontsize=FONT_LABEL)
    ax.set_ylabel("Latent Dimension", fontsize=FONT_LABEL)
    ax.set_title("Dimension Significance (Mann-Whitney U)", fontsize=FONT_TITLE, fontweight="normal", pad=6)

    # Add significance threshold line in colorbar
    sig_threshold = -np.log10(0.05)
    ax.axhline(y=-0.5, color="none")  # No-op to ensure consistent rendering

    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distributional_distances(
    pairwise_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Bar chart of pairwise FID and MMD between classes.

    Summarizes overall latent distribution separability between classes.
    Higher FID/MMD = more distinct latent representations.

    Args:
        pairwise_df: DataFrame with columns: pair, FID, MMD.
        output_path: Path to save figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pairwise_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))

    pairs = pairwise_df["pair"].values
    x = np.arange(len(pairs))
    bar_colors = [COLOR_VERMILLION, COLOR_PURPLE, COLOR_GREEN, COLOR_BLUE]

    for ax, metric in zip(axes, ["FID", "MMD"]):
        vals = pairwise_df[metric].values
        colors = [bar_colors[i % len(bar_colors)] for i in range(len(vals))]
        bars = ax.bar(x, vals, color=colors, alpha=0.75, edgecolor=COLOR_BLACK, linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=FONT_TICK)

        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_vs_", " vs ") for p in pairs],
                           fontsize=FONT_TICK, rotation=20, ha="right")
        ax.set_ylabel(metric, fontsize=FONT_LABEL)
        ax.set_title(f"Pairwise {metric}", fontsize=FONT_LABEL + 1, fontweight="normal")
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle("Distributional Distances Between Classes", fontsize=FONT_TITLE, fontweight="normal", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Class Separation Plots
# ============================================================================

def plot_class_separation_scatter_2d(
    reduced_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    centroids: Optional[Dict[int, np.ndarray]] = None,
    method_name: str = "PCA",
    label_map: Optional[Dict[int, str]] = None,
    explained_var: Optional[np.ndarray] = None,
) -> None:
    """2D scatter plot of latent space colored by class with optional centroids.

    Args:
        reduced_2d: Array of shape ``(N, 2)`` with reduced coordinates.
        labels: Integer class labels of shape ``(N,)``.
        output_path: Path to save the figure.
        centroids: Optional dict mapping class int → 2D centroid array.
        method_name: Name of the dimensionality reduction method for titles.
        label_map: Optional dict mapping int labels to display strings.
        explained_var: Optional PCA explained variance ratios (length 2).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_colors = [COLOR_GREEN, COLOR_VERMILLION, COLOR_PURPLE, COLOR_BLUE, COLOR_SKY, COLOR_ORANGE]
    unique_labels = sorted(np.unique(labels))

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    for i, c in enumerate(unique_labels):
        mask = labels == c
        color = class_colors[i % len(class_colors)]
        display_name = label_map.get(c, str(c)) if label_map else str(c)

        ax.scatter(
            reduced_2d[mask, 0], reduced_2d[mask, 1],
            c=color, s=4, alpha=0.25, label=display_name, rasterized=True,
        )

        if centroids is not None and c in centroids:
            cx, cy = centroids[c]
            ax.scatter(cx, cy, c=color, s=120, marker="*", edgecolors=COLOR_BLACK,
                       linewidth=0.8, zorder=10)
            ax.annotate(
                display_name, xy=(cx, cy), fontsize=FONT_LABEL, fontweight="bold",
                color=color, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85),
            )

    # Axis labels
    if explained_var is not None and len(explained_var) >= 2:
        ax.set_xlabel(f"{method_name} 1 ({explained_var[0]*100:.1f}%)", fontsize=FONT_LABEL)
        ax.set_ylabel(f"{method_name} 2 ({explained_var[1]*100:.1f}%)", fontsize=FONT_LABEL)
    else:
        ax.set_xlabel(f"{method_name} 1", fontsize=FONT_LABEL)
        ax.set_ylabel(f"{method_name} 2", fontsize=FONT_LABEL)

    ax.set_title(f"Latent Space — {method_name} Projection", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.legend(loc="best", fontsize=FONT_LEGEND, markerscale=3, framealpha=0.95)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_per_dimension_boxplots(
    X: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    label_map: Optional[Dict[int, str]] = None,
    dim_names: Optional[list] = None,
    max_dims: int = 16,
) -> None:
    """Per-dimension boxplots grouped by class.

    Creates side-by-side boxplots for each latent dimension, showing the
    distribution of each class.

    Args:
        X: Feature matrix of shape ``(N, D)``.
        labels: Integer class labels of shape ``(N,)``.
        output_path: Path to save the figure.
        label_map: Optional dict mapping int labels to display strings.
        dim_names: Optional list of dimension names.
        max_dims: Maximum number of dimensions to plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_dims = min(X.shape[1], max_dims)
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)

    class_colors = [COLOR_GREEN, COLOR_VERMILLION, COLOR_PURPLE, COLOR_BLUE, COLOR_SKY, COLOR_ORANGE]

    n_cols = 4
    n_rows = math.ceil(n_dims / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    axes = np.atleast_2d(axes)

    for d in range(n_dims):
        ax = axes[d // n_cols, d % n_cols]
        data = [X[labels == c, d] for c in unique_labels]
        names = [label_map.get(c, str(c)) if label_map else str(c) for c in unique_labels]

        bp = ax.boxplot(
            data, labels=names, patch_artist=True,
            widths=0.6, showfliers=False,
            medianprops=dict(color=COLOR_BLACK, linewidth=1.0),
        )
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(class_colors[j % len(class_colors)])
            patch.set_alpha(0.6)

        dim_label = dim_names[d] if dim_names and d < len(dim_names) else f"z{d}"
        ax.set_title(dim_label, fontsize=FONT_LABEL)
        _style_axes(ax, grid="major", minor_ticks=False)

    # Hide unused axes
    for d in range(n_dims, n_rows * n_cols):
        axes[d // n_cols, d % n_cols].set_visible(False)

    fig.suptitle("Per-Dimension Distribution by Class", fontsize=FONT_TITLE, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_distance_heatmap(
    between_class_distances: Dict[str, float],
    centroids: Dict[int, Any],
    output_path: Path,
    *,
    label_map: Optional[Dict[int, str]] = None,
) -> None:
    """Pairwise centroid distance heatmap.

    Args:
        between_class_distances: Dict mapping ``"i_vs_j"`` → L2 distance.
        centroids: Dict mapping class int → centroid (list or ndarray).
        output_path: Path to save the figure.
        label_map: Optional dict mapping int labels to display strings.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = sorted(centroids.keys())
    n = len(classes)
    dist_matrix = np.zeros((n, n))

    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i == j:
                continue
            key = f"{ci}_vs_{cj}" if ci < cj else f"{cj}_vs_{ci}"
            dist_matrix[i, j] = between_class_distances.get(key, 0.0)

    tick_labels = [label_map.get(c, str(c)) if label_map else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=FONT_TICK)
    ax.set_yticklabels(tick_labels, fontsize=FONT_TICK)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            if val > 0:
                text_color = "white" if val > dist_matrix.max() * 0.6 else COLOR_BLACK
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=FONT_TICK, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="L2 Distance", shrink=0.8)
    ax.set_title("Pairwise Centroid Distances", fontsize=FONT_TITLE, fontweight="normal", pad=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distance_to_centroid_violins(
    per_class_dists: Dict[int, list],
    output_path: Path,
    *,
    label_map: Optional[Dict[int, str]] = None,
    title: str = "Distance to Class Centroid",
    foreign_dists: Optional[Dict[int, list]] = None,
) -> None:
    """Violin plots of per-class distance-to-centroid distributions.

    Args:
        per_class_dists: Dict mapping class int → list of own-centroid
            distances.
        output_path: Path to save the figure.
        label_map: Optional dict mapping int labels to display strings.
        title: Plot title.
        foreign_dists: Optional dict mapping class int → list of nearest
            foreign-centroid distances.  If provided, shown alongside own
            distances for comparison.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = sorted(per_class_dists.keys())
    class_colors = [COLOR_GREEN, COLOR_VERMILLION, COLOR_PURPLE, COLOR_BLUE, COLOR_SKY, COLOR_ORANGE]

    has_foreign = foreign_dists is not None and len(foreign_dists) > 0

    if has_foreign:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 5.0))
    else:
        fig, ax1 = plt.subplots(figsize=(6.0, 5.0))

    # Own-centroid distances
    data_own = [np.array(per_class_dists[c]) for c in classes]
    names = [label_map.get(c, str(c)) if label_map else str(c) for c in classes]

    parts = ax1.violinplot(data_own, positions=range(len(classes)), showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(class_colors[i % len(class_colors)])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color(COLOR_BLACK)
    parts["cmedians"].set_color(COLOR_VERMILLION)

    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(names, fontsize=FONT_TICK)
    ax1.set_ylabel("L2 Distance", fontsize=FONT_LABEL)
    ax1.set_title("Dist. to Own Centroid", fontsize=FONT_LABEL + 1, fontweight="normal")
    _style_axes(ax1, grid="major", minor_ticks=False)

    # Foreign-centroid distances (if available)
    if has_foreign:
        data_foreign = [np.array(foreign_dists[c]) for c in classes]
        parts_f = ax2.violinplot(data_foreign, positions=range(len(classes)), showmeans=True, showmedians=True)
        for i, pc in enumerate(parts_f["bodies"]):
            pc.set_facecolor(class_colors[i % len(class_colors)])
            pc.set_alpha(0.6)
        parts_f["cmeans"].set_color(COLOR_BLACK)
        parts_f["cmedians"].set_color(COLOR_VERMILLION)

        ax2.set_xticks(range(len(classes)))
        ax2.set_xticklabels(names, fontsize=FONT_TICK)
        ax2.set_ylabel("L2 Distance", fontsize=FONT_LABEL)
        ax2.set_title("Dist. to Nearest Foreign Centroid", fontsize=FONT_LABEL + 1, fontweight="normal")
        _style_axes(ax2, grid="major", minor_ticks=False)

    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="normal", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_class_separation(
    temporal_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """2x2 grid showing class separation metrics vs time-to-birth.

    Args:
        temporal_df: DataFrame from ``compute_temporal_separation`` with
            columns: ``bin_center``, ``silhouette``, ``davies_bouldin``,
            ``calinski_harabasz``, ``fisher_ratio``, etc.
        output_path: Path to save the figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    t = temporal_df["bin_center"].values

    # Panel definitions: (column, ylabel, title, color, higher_is_better)
    panels = [
        ("silhouette", "Silhouette Score", "Silhouette vs Time", COLOR_BLUE, True),
        ("fisher_ratio", "Fisher Ratio", "Fisher Ratio vs Time", COLOR_GREEN, True),
        ("davies_bouldin", "Davies-Bouldin Index", "Davies-Bouldin vs Time", COLOR_VERMILLION, False),
        ("calinski_harabasz", "CH Index", "Calinski-Harabasz vs Time", COLOR_PURPLE, True),
    ]

    for ax, (col, ylabel, title, color, _) in zip(axes.flat, panels):
        if col in temporal_df.columns:
            vals = temporal_df[col].values
            valid = np.isfinite(vals)
            ax.plot(t[valid], vals[valid], color=color, linewidth=1.2, marker="o", markersize=3)
            ax.fill_between(t[valid], vals[valid], alpha=0.15, color=color)
        ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_LABEL + 1, fontweight="normal")
        ax.invert_xaxis()
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle("Class Separation Evolution Over Time", fontsize=FONT_TITLE, fontweight="normal", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
