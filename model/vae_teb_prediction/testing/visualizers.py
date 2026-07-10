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
from matplotlib.patches import Patch

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

# Canonical class metadata: label ID -> human-readable name + plotting colour.
# Used by every class-aware plotter so panels look identical across analyses.
CLASS_NAMES: Dict[int, str] = {1: "HEALTHY", 2: "ACIDOSIS", 3: "HIE"}
CLASS_COLORS: Dict[int, str] = {
    1: COLOR_BLUE,
    2: COLOR_ORANGE,
    3: COLOR_VERMILLION,
}


def unique_labels_in(values: Any) -> list:
    """Return the sorted list of integer class IDs actually present.

    Accepts a numpy array, pandas Series, list, or None. Non-finite
    and non-numeric values are ignored. Result is sorted ascending.

    Note:
        ``np.unique`` calls ``.sort()`` on the underlying array, which
        raises ``TypeError`` when the array is object-dtype containing
        only ``None`` (Python's ``None < None`` is undefined). We
        therefore pre-filter to integer-coercible values *before*
        deduplicating, which keeps the function safe on label-free
        datasets (e.g. synthetic-TE batches) where ``df["label"]`` is
        all-``None``.
    """
    if values is None:
        return []
    try:
        arr = np.asarray(values)
    except Exception:
        return []
    if arr.size == 0:
        return []
    seen: set = set()
    iterable = arr.tolist() if hasattr(arr, "tolist") else arr
    for v in iterable:
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv in CLASS_NAMES:
            seen.add(iv)
    return sorted(seen)


def class_label_for(label_id: int) -> str:
    """Pretty name for a class id, falling back to ``f"class {id}"``."""
    return CLASS_NAMES.get(int(label_id), f"class {int(label_id)}")


def class_color_for(label_id: int, fallback: str = COLOR_GRAY) -> str:
    """Palette colour for a class id, falling back to gray when unknown."""
    return CLASS_COLORS.get(int(label_id), fallback)

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
FONT_LABEL = plt.rcParams["axes.labelsize"]*1.5
FONT_TITLE = plt.rcParams["axes.titlesize"]*1.5
FONT_TICK = plt.rcParams["xtick.labelsize"]*1.5
FONT_LEGEND = plt.rcParams["legend.fontsize"]*1.5


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
    """Create a single-column histogram panel for the v1 feature-forecast metrics.

    Draws histograms for ``feat_mse_total``, ``uplift_rel``,
    ``residual_ratio``, ``kld_mean``, ``kld_sum`` and ``kld_l2`` (falls back to the legacy
    ``vaf/mse/snr/kld`` set when none of the new columns are present).

    Args:
        df: DataFrame produced by :func:`collect_metrics` (new schema) or
            any legacy file that still carries the old column names.
        output_dir: Directory to save the plot.
        filename: Output filename.
        add_kde: Whether to overlay a KDE trace (currently unused).
        add_ci: Whether to include 95% confidence intervals in the stats
            box.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lag-attn v1 primary schema.
    v1_metrics = [
        ("feat_mse_total", "Feature Forecast MSE", COLOR_BLUE, ""),
        ("uplift_rel", "Baseline - Full Uplift (rel.)", COLOR_GREEN, ""),
        ("residual_ratio", "Residual Usage Ratio", COLOR_ORANGE, ""),
        ("kld_mean", "KL Mean / Dim", COLOR_PURPLE, "nats"),
        ("kld_sum", "KL Sum Across Dims", COLOR_VERMILLION, "nats"),
        ("kld_l2", "KL Vector L2", COLOR_SKY, "nats"),
    ]
    # Legacy fallback (old histogram_metrics.csv).
    legacy_metrics = [
        ("vaf", "Variance Accounted For (VAF)", COLOR_BLUE, ""),
        ("mse", "Mean Squared Error (MSE)", COLOR_GREEN, ""),
        ("snr", "Signal-to-Noise Ratio (SNR)", COLOR_ORANGE, "dB"),
        ("kld", "Transfer Entropy (KLD)", COLOR_PURPLE, "nats"),
    ]
    if any(m[0] in df.columns for m in v1_metrics):
        metrics_config = v1_metrics
    else:
        metrics_config = legacy_metrics

    # Single-column layout with wider panels
    fig, axes = plt.subplots(
        len(metrics_config), 1, figsize=(6.5, max(8.5, 2.1 * len(metrics_config)))
    )
    axes = np.atleast_1d(axes)

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
        xlabel = f"{col}" + (f" ({unit})" if unit else "")
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
        ax.set_ylabel("Density", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal", pad=8)

        # Use log scale for KL / MSE x-axis by default (values span orders of magnitude).
        if col in ("kld", "kld_mean", "kld_sum", "kld_l2", "feat_mse_total") and (values > 0).all():
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


def plot_metric_histograms_by_class(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "metrics_histograms_by_class.pdf",
    *,
    label_col: str = "label",
    metrics: Optional[list] = None,
    max_classes_side_by_side: int = 4,
) -> None:
    """Per-class version of :func:`plot_metric_histograms`.

    Produces a grid where each row is a metric and each column is a
    class. When more than ``max_classes_side_by_side`` classes are
    present the layout collapses to one column with overlaid densities
    per class (histtype=``stepfilled``).

    Args:
        df: DataFrame with at least ``label_col`` and the metric
            columns listed in ``metrics`` (or auto-detected v1 set).
        output_dir: Directory to save the plot.
        filename: Output filename.
        label_col: Column carrying the integer class id (1/2/3).
        metrics: Optional explicit list of ``(col, title, unit)``
            triples. When ``None`` the v1 default set is used.
        max_classes_side_by_side: Threshold above which the grid
            collapses to a single column with per-class overlays.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if label_col not in df.columns:
        # No labels available — fall back to the pooled plot.
        plot_metric_histograms(df, output_dir, filename=filename)
        return

    if metrics is None:
        metrics = [
            ("feat_mse_total", "Feature Forecast MSE", ""),
            ("feat_r2_total", "Feature Forecast R^2", ""),
            ("uplift_rel", "Uplift (relative)", ""),
            ("residual_ratio", "Residual Usage Ratio", ""),
            ("kld_mean", "KL Mean / Dim", "nats"),
            ("kld_sum", "KL Sum Across Dims", "nats"),
            ("kld_l2", "KL Vector L2", "nats"),
        ]
    metrics = [m for m in metrics if m[0] in df.columns]
    if not metrics:
        return

    classes = unique_labels_in(df[label_col])
    if len(classes) < 2:
        # Nothing to split — emit the pooled panel and return.
        plot_metric_histograms(df, output_dir, filename=filename)
        return

    overlay_mode = len(classes) > max_classes_side_by_side
    n_rows = len(metrics)
    n_cols = 1 if overlay_mode else len(classes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(4.0, 3.0 * n_cols) if not overlay_mode else 6.0, 2.2 * n_rows),
        squeeze=False,
    )

    for r, (col, title, unit) in enumerate(metrics):
        for c, label_id in enumerate(classes):
            if overlay_mode:
                ax = axes[r, 0]
            else:
                ax = axes[r, c]
            sub = df[df[label_col] == label_id]
            vals = sub[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                if not overlay_mode:
                    ax.text(0.5, 0.5, "—", ha="center", va="center")
                    _style_axes(ax, grid="major", minor_ticks=False)
                continue
            color = class_color_for(label_id)
            n_bins = min(80, max(20, int(np.sqrt(vals.size) * 2)))
            if overlay_mode:
                ax.hist(
                    vals, bins=n_bins, density=True, color=color,
                    alpha=0.35, histtype="stepfilled",
                    edgecolor=COLOR_BLACK, linewidth=0.4,
                    label=f"{class_label_for(label_id)} (n={vals.size})",
                )
            else:
                ax.hist(
                    vals, bins=n_bins, density=True, color=color,
                    alpha=0.8, edgecolor=COLOR_BLACK, linewidth=0.4,
                )
                ax.set_title(
                    f"{class_label_for(label_id)} (n={vals.size})",
                    fontsize=FONT_TITLE * 0.75,
                )
            ax.axvline(float(np.mean(vals)), color=COLOR_ORANGE, lw=0.7, ls="--")
            ax.axvline(float(np.median(vals)), color=COLOR_PURPLE, lw=0.7, ls="-.")
            if col in ("kld", "kld_mean", "kld_sum", "kld_l2", "feat_mse_total") and np.all(vals > 0):
                ax.set_xscale("log")
            _style_axes(ax, grid="major", minor_ticks=False)
        # Row-level labelling
        axes[r, 0].set_ylabel(
            f"{title}" + (f" ({unit})" if unit else ""),
            fontsize=FONT_LABEL * 0.9,
        )
        if overlay_mode:
            axes[r, 0].legend(loc="best", fontsize=FONT_LEGEND * 0.9, frameon=True)

    for ax in axes[-1]:
        ax.set_xlabel("value", fontsize=FONT_LABEL * 0.9)

    mode_name = "overlay" if overlay_mode else "grid"
    fig.suptitle(
        f"Per-class metric distributions ({mode_name}, "
        f"{len(classes)} classes)",
        fontsize=FONT_TITLE, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Frequency-band forecast plotters (lag-attn v1)
# -----------------------------------------------------------------------------

# Stable band-to-color mapping. Order matches BAND_NAMES in
# ``model/vae_teb_prediction/testing/band_partition.py`` so the legend
# reads from slow-baseline (cooler) to beat-to-beat (warmer) bands.
_BAND_COLORS: Dict[str, str] = {
    "slow_baseline": COLOR_PURPLE,
    "deceleration": COLOR_BLUE,
    "variability": COLOR_GREEN,
    "beat_to_beat": COLOR_VERMILLION,
}


def _band_color_for(band: str, fallback: str = COLOR_GRAY) -> str:
    """Return a stable color for a band/partition label.

    Known clinical bands ride on the canonical palette; arbitrary labels
    (refined7, by_kind, by_octave, ...) are projected through a stable
    hash into a cyclic palette so the same label always gets the same
    color across runs.
    """
    name = str(band)
    if name in _BAND_COLORS:
        return _BAND_COLORS[name]
    fallback_palette = (
        COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_VERMILLION,
        COLOR_PURPLE, COLOR_TEAL_DARK, COLOR_GRAY,
    )
    h = abs(hash(name)) % len(fallback_palette)
    return fallback_palette[h] or fallback


_CANONICAL_4BAND_ORDER: Tuple[str, ...] = (
    "slow_baseline", "deceleration", "variability", "beat_to_beat",
)
_CANONICAL_REFINED7_ORDER: Tuple[str, ...] = (
    "baseline", "early_decel", "late_decel",
    "lf_var", "mf_var", "beat_to_beat", "nyquist_edge",
)
_CANONICAL_KIND_ORDER: Tuple[str, ...] = (
    "st_S0", "st_S1", "ph_diag", "ph_h2", "ph_h3", "ph_other",
)


def _ordered_bands(
    values: Any,
    *,
    canonical: Optional[Tuple[str, ...]] = None,
) -> list:
    """Return present band/partition labels in a stable display order.

    Args:
        values: Iterable of label values (typically a DataFrame column).
        canonical: Optional canonical ordering. When ``None`` the helper
            tries the clinical-4 / refined-7 / by-kind orders in turn,
            and falls back to ``sorted(seen)`` when none match (e.g.
            ``octave_*`` labels — ``sorted`` keeps them in numeric order).

    Returns:
        List of label names in the chosen order, filtered to those
        actually present in ``values``.
    """
    seen = set()
    try:
        for v in pd.Series(values).dropna().unique():
            seen.add(str(v))
    except Exception:
        return []
    if not seen:
        return []
    if canonical is not None:
        ordered = [b for b in canonical if b in seen]
        if ordered:
            return ordered + sorted(seen.difference(canonical))
        return sorted(seen)
    # Auto-detect ordering by which canonical list shares the most labels.
    for canon in (
        _CANONICAL_4BAND_ORDER,
        _CANONICAL_REFINED7_ORDER,
        _CANONICAL_KIND_ORDER,
    ):
        match = [b for b in canon if b in seen]
        if len(match) >= max(1, int(len(seen) * 0.6)):
            return match + sorted(seen.difference(canon))
    # Fallback: alphabetical (octave_0..octave_10 sort numerically with
    # zero-padding only — the kymatio bank gives them as 'octave_0' etc.,
    # so we sort by trailing integer when possible).
    def _sort_key(name: str) -> Tuple[int, str]:
        if name.startswith("octave_"):
            tail = name.split("_", 1)[1]
            try:
                return (int(tail), name)
            except ValueError:
                return (10**9, name)
        return (10**9, name)
    return sorted(seen, key=_sort_key)


def _format_band_label_with_hz(
    band: str,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]],
) -> str:
    r"""Append the explicit Hz range to a band name when known.

    Falls back to the bare name when ``band_hz_ranges`` is ``None`` or
    does not contain the band (e.g. ``by_kind`` partitions, where the
    label is a coefficient kind rather than a band).

    Examples:
        ``("deceleration", {...})`` $\rightarrow$
        ``"deceleration\n(0.008–0.04 Hz)"``.
    """
    if band_hz_ranges is None:
        return str(band)
    rng = band_hz_ranges.get(str(band))
    if rng is None:
        return str(band)
    lo, hi = float(rng[0]), float(rng[1])
    lo_finite = np.isfinite(lo) and lo > 0.0
    hi_finite = np.isfinite(hi)
    # Degenerate "covers everything" range (e.g. ``octave_dc``) — collapse
    # to a DC tag rather than emit a misleading "(> 0 Hz)" string.
    if not lo_finite and not hi_finite:
        return f"{band}\n(DC)"
    if not hi_finite:
        return f"{band}\n(> {lo:g} Hz)"
    if not lo_finite:
        return f"{band}\n(< {hi:g} Hz)"
    return f"{band}\n({lo:g}–{hi:g} Hz)"


def plot_band_violin(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    *,
    title: Optional[str] = None,
    n_channels_by_band: Optional[Dict[str, int]] = None,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Violin plot of ``value_col`` per band on a single axes.

    Expected DataFrame columns: ``band`` (string) and ``value_col``.
    Empty bands or all-NaN values are skipped silently.

    Args:
        df: Long-format dataframe with ``band`` and ``value_col`` columns.
        value_col: Numeric column to plot.
        output_path: Destination PDF.
        title: Optional figure title.
        n_channels_by_band: Channel-count annotation per band.
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` mapping. When
            provided, every x-tick label is suffixed with the explicit
            Hz range so reviewers don't have to look up the cutoffs.
    """
    bands = _ordered_bands(df.get("band"))
    if not bands or value_col not in df.columns:
        return

    data = []
    labels = []
    colors = []
    for band in bands:
        vals = pd.to_numeric(
            df.loc[df["band"] == band, value_col], errors="coerce"
        ).to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        data.append(vals)
        n_ch = (
            n_channels_by_band.get(band) if n_channels_by_band is not None else None
        )
        band_hz_label = _format_band_label_with_hz(band, band_hz_ranges)
        if n_ch is None or n_ch <= 0:
            labels.append(f"{band_hz_label}\n(n={vals.size})")
        else:
            labels.append(f"{band_hz_label}\nch={n_ch}, n={vals.size}")
        colors.append(_band_color_for(band))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(4.4, 1.5 * len(data) + 1.6), 3.4))
    parts = ax.violinplot(
        data, showmeans=False, showmedians=True, showextrema=False,
    )
    for body, color in zip(parts.get("bodies", []), colors):
        body.set_facecolor(color)
        body.set_edgecolor(COLOR_BLACK)
        body.set_alpha(0.55)
        body.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color(COLOR_BLACK)
        parts["cmedians"].set_linewidth(0.8)

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=FONT_LABEL * 0.85)
    ax.set_ylabel(value_col, fontsize=FONT_LABEL)
    ax.set_title(
        title or f"{value_col} per frequency band",
        fontsize=FONT_TITLE, fontweight="normal",
    )
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_band_violin_by_class(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    *,
    title: Optional[str] = None,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Grouped violin plot: one cluster per band, one violin per class.

    Expected columns: ``band``, ``label``, ``value_col``. Bands without
    samples in any class are skipped. Falls back to the pooled plot
    when fewer than 2 classes are present.

    Args:
        df: Long-format dataframe with ``band``, ``label``, ``value_col``.
        value_col: Numeric column to plot.
        output_path: Destination PDF.
        title: Optional figure title.
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` for explicit Hz
            tick suffixes (see :func:`plot_band_violin`).
    """
    bands = _ordered_bands(df.get("band"))
    classes = unique_labels_in(df.get("label"))
    if not bands or value_col not in df.columns:
        return
    if len(classes) < 2:
        plot_band_violin(
            df, value_col, output_path, title=title,
            band_hz_ranges=band_hz_ranges,
        )
        return

    fig, ax = plt.subplots(
        figsize=(max(5.0, 1.7 * len(bands) + 1.6), 3.6),
    )
    width = 0.8 / max(len(classes), 1)
    legend_handles: list = []
    legend_added = set()
    for b_idx, band in enumerate(bands):
        for c_idx, lab in enumerate(classes):
            vals = pd.to_numeric(
                df.loc[(df["band"] == band) & (df["label"] == lab), value_col],
                errors="coerce",
            ).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            pos = b_idx + 1 + (c_idx - (len(classes) - 1) / 2.0) * width
            parts = ax.violinplot(
                [vals], positions=[pos], widths=width * 0.9,
                showmeans=False, showmedians=True, showextrema=False,
            )
            color = class_color_for(lab)
            for body in parts.get("bodies", []):
                body.set_facecolor(color)
                body.set_edgecolor(COLOR_BLACK)
                body.set_alpha(0.55)
                body.set_linewidth(0.5)
            if "cmedians" in parts:
                parts["cmedians"].set_color(COLOR_BLACK)
                parts["cmedians"].set_linewidth(0.7)
            if lab not in legend_added:
                legend_handles.append(
                    Patch(facecolor=color, edgecolor=COLOR_BLACK,
                          alpha=0.55, label=class_label_for(lab))
                )
                legend_added.add(lab)

    ax.set_xticks(range(1, len(bands) + 1))
    band_tick_labels = [
        _format_band_label_with_hz(b, band_hz_ranges) for b in bands
    ]
    ax.set_xticklabels(band_tick_labels, fontsize=FONT_LABEL * 0.85)
    ax.set_ylabel(value_col, fontsize=FONT_LABEL)
    ax.set_title(
        title or f"{value_col} per frequency band — by class",
        fontsize=FONT_TITLE, fontweight="normal",
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="best",
            frameon=True, fontsize=FONT_LEGEND,
        )
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _ribbon_plot_per_band(
    df: pd.DataFrame,
    *,
    x_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_to_minutes: Optional[float] = None,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Median + IQR ribbon, one line per band, all on shared axes."""
    bands = _ordered_bands(df.get("band"))
    if not bands or x_col not in df.columns or value_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plotted = False
    for band in bands:
        sub = df[df["band"] == band]
        if sub.empty:
            continue
        grouped = sub.groupby(x_col)[value_col]
        med = grouped.median()
        q1 = grouped.quantile(0.25)
        q3 = grouped.quantile(0.75)
        if med.empty:
            continue
        xs = np.asarray(med.index.to_list(), dtype=float)
        if x_to_minutes is not None:
            xs = xs * float(x_to_minutes)
        color = _band_color_for(band)
        legend_label = _format_band_label_with_hz(
            band, band_hz_ranges,
        ).replace("\n", "  ")
        ax.plot(xs, med.to_numpy(), color=color, lw=1.1, label=legend_label)
        ax.fill_between(
            xs, q1.to_numpy(), q3.to_numpy(),
            color=color, alpha=0.18, lw=0,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_band_horizon_error(
    per_horizon_df: pd.DataFrame,
    output_path: Path,
    *,
    value_col: str = "mse",
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Per-band median+IQR ribbon of forecast MSE vs horizon step ``h``.

    Args:
        per_horizon_df: Long-format dataframe with ``band, h, value_col``.
        output_path: Destination PDF.
        value_col: Numeric column to plot (default ``"mse"``).
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` for explicit Hz
            suffixes in the legend.
    """
    _ribbon_plot_per_band(
        per_horizon_df,
        x_col="h",
        value_col=value_col,
        output_path=output_path,
        title="Forecast error by horizon step — per frequency band",
        xlabel="horizon step h",
        ylabel=f"{value_col} (median, IQR)",
        band_hz_ranges=band_hz_ranges,
    )


def plot_band_anchor_error(
    per_anchor_df: pd.DataFrame,
    output_path: Path,
    *,
    value_col: str = "mse",
    decim_step_seconds: float = 4.0,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Per-band median+IQR ribbon of forecast MSE vs anchor position ``t``.

    The anchor axis is rescaled to minutes via ``t * decim_step_seconds /
    60`` so the figure is directly readable in clinical time. Default
    ``decim_step_seconds=4`` matches the v1 model's 16x decimation at
    ``fs=4 Hz``.

    Args:
        per_anchor_df: Long-format dataframe with ``band, t, value_col``.
        output_path: Destination PDF.
        value_col: Numeric column to plot.
        decim_step_seconds: Physical seconds per decimated step.
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` for explicit Hz
            suffixes in the legend.
    """
    _ribbon_plot_per_band(
        per_anchor_df,
        x_col="t",
        value_col=value_col,
        output_path=output_path,
        title="Forecast error by anchor position — per frequency band",
        xlabel="anchor position (min into segment)",
        ylabel=f"{value_col} (median, IQR)",
        x_to_minutes=decim_step_seconds / 60.0,
        band_hz_ranges=band_hz_ranges,
    )


def plot_band_horizon_heatmap(
    per_horizon_df: pd.DataFrame,
    output_path: Path,
    *,
    horizon: int,
    fs_hz: float,
    n_samples: int,
    value_col: str = "mse",
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    log_color: bool = True,
    metric_label: str = "Forecast MSE",
) -> None:
    r"""Band $\times$ horizon-step heatmap of mean MSE.

    Pivots ``per_horizon_df`` by ``(band, h)``, takes the cross-sample
    mean per cell, and renders an ``imshow`` with rows = bands (high
    frequency at top, descending) and columns = horizon step
    $h \in [0, H_d)$. Y-axis labels carry explicit Hz ranges via
    :func:`_format_band_label_with_hz`; x-axis carries dual
    horizon-step / elapsed-time labels at $f_s$ Hz.

    Layout uses the same 3-column gridspec (heatmap | label header |
    colorbar) as the freq-horizon channel heatmaps, so labels and
    colorbar each get a guaranteed column.

    Args:
        per_horizon_df: Long-format dataframe with ``band, h, value_col``
            (one row per (sample, band, horizon-step)).
        output_path: Destination PDF path.
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz.
        n_samples: Number of samples averaged into each cell.
        value_col: Numeric column to aggregate (default ``"mse"``).
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` used to append Hz
            suffixes to row labels.
        log_color: Use a log-scaled colormap (default).
        metric_label: Colorbar label prefix.
    """
    if per_horizon_df is None or per_horizon_df.empty:
        return
    if "band" not in per_horizon_df.columns or "h" not in per_horizon_df.columns:
        return
    if value_col not in per_horizon_df.columns:
        return

    bands = _ordered_bands(per_horizon_df["band"])
    if not bands:
        return

    grid = (
        per_horizon_df.groupby(["band", "h"])[value_col]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(index=bands, columns=range(horizon))
    )
    mse_grid = grid.to_numpy(dtype=float)
    n_rows, n_cols = mse_grid.shape
    if n_rows == 0 or n_cols == 0:
        return

    finite_vals = mse_grid[np.isfinite(mse_grid) & (mse_grid > 0.0)]
    if log_color and finite_vals.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_vals, 1.0))
        vmax = float(np.percentile(finite_vals, 99.0))
        if vmin <= 0 or not np.isfinite(vmin):
            vmin = float(np.min(finite_vals))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(
            vmin=float(np.nanmin(mse_grid)),
            vmax=float(np.nanmax(mse_grid)),
        )

    fig_h = max(2.6, 0.42 * float(n_rows) + 1.6)
    fig_w = max(7.0, 0.4 * float(n_cols) + 4.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        width_ratios=[1.0, 0.025],
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        mse_grid, aspect="auto", origin="upper",
        cmap="magma", norm=norm,
        interpolation="nearest",
    )

    yticks = list(range(n_rows))
    yticklabels = [
        _format_band_label_with_hz(band, band_hz_ranges)
        for band in bands
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=FONT_LABEL * 0.85)
    ax.set_ylabel("Frequency band", fontsize=FONT_LABEL)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([
        f"$h={k}$\n({k / float(fs_hz):.2f} s)" for k in range(n_cols)
    ])
    ax.set_xlabel(
        f"Horizon step (and elapsed time at $f_s = {fs_hz:g}$ Hz)",
        fontsize=FONT_LABEL,
    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        f"{metric_label} (log scale)" if log_color else metric_label,
        fontsize=FONT_LABEL,
    )
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    cbar.ax.set_title(
        f"avg over\n{n_samples} samples",
        fontsize=plt.rcParams["axes.labelsize"] * 0.7, pad=4,
    )

    ax.set_title(
        "Forecast error — band × horizon step (cross-sample mean)\n"
        rf"$f_s = {fs_hz:g}$ Hz, $H_d = {horizon}$, $n$ = {n_samples} samples",
        fontsize=FONT_TITLE * 0.95, pad=10,
    )
    _style_axes(ax, grid="none", minor_ticks=False)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.18)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_band_horizon_meanse_lines(
    per_horizon_df: pd.DataFrame,
    output_path: Path,
    *,
    horizon: int,
    fs_hz: float,
    n_samples: int,
    value_col: str = "mse",
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    r"""Mean $\pm$ SE line plot of forecast error vs horizon step, one line per band.

    Cleaner alternative to the median+IQR :func:`plot_band_horizon_error`
    when reviewers want a quick read of how forecast quality decays with
    horizon for each band. Each band's line is labelled at its right
    edge ($h = H_d - 1$), so a separate legend is not needed.

    SE is computed as $\mathrm{std}(value\_col) / \sqrt{n}$ across
    samples per ``(band, h)``.

    Args:
        per_horizon_df: Long-format dataframe with ``band, h, value_col``.
        output_path: Destination PDF.
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz.
        n_samples: Number of samples (used in title only).
        value_col: Numeric column to aggregate (default ``"mse"``).
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` used in
            right-edge labels.
    """
    if per_horizon_df is None or per_horizon_df.empty:
        return
    if "band" not in per_horizon_df.columns or "h" not in per_horizon_df.columns:
        return
    if value_col not in per_horizon_df.columns:
        return

    bands = _ordered_bands(per_horizon_df["band"])
    if not bands:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    plotted = False
    label_y_positions: List[Tuple[float, str, str]] = []  # (y, label, color)
    for band in bands:
        sub = per_horizon_df[per_horizon_df["band"] == band]
        if sub.empty:
            continue
        grouped = sub.groupby("h")[value_col]
        mean = grouped.mean()
        std = grouped.std()
        n = grouped.count().clip(lower=1)
        se = std / np.sqrt(n.to_numpy())
        if mean.empty:
            continue
        xs = np.asarray(mean.index.to_list(), dtype=float)
        color = _band_color_for(band)
        ys = mean.to_numpy()
        ax.plot(xs, ys, color=color, lw=1.6)
        ax.fill_between(
            xs, ys - se.to_numpy(), ys + se.to_numpy(),
            color=color, alpha=0.18, lw=0,
        )
        # Right-edge label — replace newline in the Hz-suffix with " "
        # so the inline label stays one row tall.
        right_label = _format_band_label_with_hz(
            band, band_hz_ranges,
        ).replace("\n", " ")
        label_y_positions.append((float(ys[-1]), right_label, color))
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    # Annotate each line at its right edge. Sort y positions to nudge
    # overlapping labels apart by a small offset (visual de-collision).
    xmax = float(horizon - 1)
    for y, label, color in sorted(label_y_positions, key=lambda t: t[0]):
        ax.annotate(
            label,
            xy=(xmax, y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=FONT_LEGEND,
            color=color,
            annotation_clip=False,
        )

    ax.set_xlabel(
        f"Horizon step h (elapsed time at $f_s = {fs_hz:g}$ Hz: 0..{(horizon-1)/fs_hz:.2f} s)",
        fontsize=FONT_LABEL,
    )
    ax.set_ylabel(f"{value_col} (mean ± SE)", fontsize=FONT_LABEL)
    ax.set_title(
        "Forecast error vs horizon step — per band\n"
        rf"$H_d = {horizon}$, $n$ = {n_samples} samples",
        fontsize=FONT_TITLE,
    )
    ax.set_xlim(-0.3, xmax + 0.6)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.subplots_adjust(left=0.10, right=0.78, top=0.88, bottom=0.13)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_channel_horizon_heatmap_band_grouped(
    long_df: pd.DataFrame,
    output_path: Path,
    *,
    horizon: int,
    fs_hz: float,
    n_samples: int,
    band_col: str,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    log_color: bool = True,
    metric_label: str = "Forecast MSE",
) -> None:
    r"""Channel $\times$ horizon heatmap with rows visually grouped by band.

    Same data primitive as the existing scattering / phase channel
    heatmaps (long-form ``channel_horizon_mse.csv``), but channels in
    the same band are placed contiguously and separated by thin
    horizontal divider lines. Band names appear in the right-margin
    label column. One PDF per partition (``clinical_4band``,
    ``clinical_7band``, ``by_octave`` — ``by_kind`` is already covered
    by the dedicated phase-by-kind heatmap).

    Args:
        long_df: Long-format dataframe with one row per
            ``(channel, h)``; must include ``channel``, ``h``,
            ``mse_mean`` and the band-id column ``band_col``. Optional
            ``freq_hz_primary`` is used for the inner-row label
            (channel id + Hz).
        output_path: Destination PDF.
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz.
        n_samples: Number of samples averaged into each cell.
        band_col: Which band column to group by — one of ``"band"``,
            ``"refined_band"``, ``"octave"``.
        band_hz_ranges: ``{band -> (low_hz, high_hz)}`` for Hz suffixes.
        log_color: Use ``LogNorm`` (default).
        metric_label: Colorbar label prefix.
    """
    if long_df is None or long_df.empty:
        return
    needed = {"channel", "h", "mse_mean", band_col}
    if not needed.issubset(long_df.columns):
        return

    # Build the (channel × horizon) MSE matrix with bands kept
    # contiguous in row order. Within each band, rows are sorted by
    # ``freq_hz_primary`` descending (high freq at the top of the
    # band's strip).
    bands = _ordered_bands(long_df[band_col])
    if not bands:
        return

    sort_freq_col = (
        "freq_hz_primary" if "freq_hz_primary" in long_df.columns else None
    )

    row_order: List[int] = []
    band_of_row: List[str] = []
    band_boundaries: List[int] = [0]   # row index where each band starts
    band_centres: List[float] = []
    band_labels: List[str] = []

    for band in bands:
        sub_meta = (
            long_df[long_df[band_col] == band]
            [["channel"] + ([sort_freq_col] if sort_freq_col else [])]
            .drop_duplicates(subset=["channel"]).copy()
        )
        if sub_meta.empty:
            continue
        if sort_freq_col is not None:
            f = sub_meta[sort_freq_col].astype(float).to_numpy()
            f_sort = np.where(np.isnan(f), -np.inf, f)
            order = np.argsort(-f_sort, kind="stable")
            sub_meta = sub_meta.iloc[order].reset_index(drop=True)
        ch_list = sub_meta["channel"].astype(int).tolist()
        start_idx = len(row_order)
        row_order.extend(ch_list)
        band_of_row.extend([str(band)] * len(ch_list))
        band_boundaries.append(len(row_order))
        band_centres.append(0.5 * (start_idx + len(row_order) - 1))
        band_labels.append(_format_band_label_with_hz(band, band_hz_ranges))

    if not row_order:
        return

    wide = (
        long_df.pivot_table(
            index="channel", columns="h", values="mse_mean", aggfunc="mean",
        )
        .reindex(row_order)
        .reindex(columns=range(horizon))
    )
    mse_grid = wide.to_numpy(dtype=float)
    n_rows, n_cols = mse_grid.shape
    if n_rows == 0 or n_cols == 0:
        return

    finite_vals = mse_grid[np.isfinite(mse_grid) & (mse_grid > 0.0)]
    if log_color and finite_vals.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_vals, 1.0))
        vmax = float(np.percentile(finite_vals, 99.0))
        if vmin <= 0 or not np.isfinite(vmin):
            vmin = float(np.min(finite_vals))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(
            vmin=float(np.nanmin(mse_grid)),
            vmax=float(np.nanmax(mse_grid)),
        )

    fig_h = max(4.5, 0.18 * float(n_rows) + 1.8)
    fig_w = max(8.5, 0.45 * float(n_cols) + 5.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 0.18, 0.025],
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_label = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    im = ax.imshow(
        mse_grid, aspect="auto", origin="upper",
        cmap="magma", norm=norm,
        interpolation="nearest",
    )

    # Channel-id ticks on the left axis (subsampled if too many).
    label_step = 1 if n_rows <= 30 else int(np.ceil(n_rows / 30.0))
    yticks = list(range(0, n_rows, label_step))
    if (n_rows - 1) not in yticks:
        yticks.append(n_rows - 1)
    yticklabels = [f"ch {row_order[i]}" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=plt.rcParams["ytick.labelsize"])
    ax.set_ylabel("Channel index (87-ch space)", fontsize=FONT_LABEL)

    # Band divider lines.
    for boundary in band_boundaries[1:-1]:
        ax.axhline(boundary - 0.5, color=COLOR_BLACK, linestyle="-",
                   linewidth=0.8, alpha=0.85)

    # X-axis horizon ticks.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([
        f"$h={k}$\n({k / float(fs_hz):.2f} s)" for k in range(n_cols)
    ])
    ax.set_xlabel(
        f"Horizon step (elapsed time at $f_s = {fs_hz:g}$ Hz)",
        fontsize=FONT_LABEL,
    )

    # Right-margin band labels at the centre of each band's row span.
    ax_label.set_xlim(0.0, 1.0)
    ax_label.set_ylim(ax.get_ylim())
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)
    for centre, label in zip(band_centres, band_labels):
        ax_label.text(
            0.02, float(centre), label,
            ha="left", va="center",
            fontsize=FONT_LABEL * 0.78, fontweight="bold",
            transform=ax_label.transData,
        )
    ax_label.set_title(
        "Band\nlabel", fontsize=FONT_LABEL * 0.75, pad=4, loc="left",
    )

    # Colorbar.
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        f"{metric_label} (log scale)" if log_color else metric_label,
        fontsize=FONT_LABEL,
    )
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    cbar.ax.set_title(
        f"avg over\n{n_samples} samples",
        fontsize=plt.rcParams["axes.labelsize"] * 0.7, pad=4,
    )

    ax.set_title(
        f"Channel × horizon — band-grouped ({band_col})\n"
        rf"$f_s = {fs_hz:g}$ Hz, $H_d = {horizon}$, $n$ = {n_samples} samples",
        fontsize=FONT_TITLE * 0.95, pad=10,
    )
    _style_axes(ax, grid="none", minor_ticks=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _grid_ribbon_by_class(
    df: pd.DataFrame,
    *,
    x_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    x_to_minutes: Optional[float] = None,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Grid plot: rows = bands, cols = classes. Each cell is a ribbon."""
    bands = _ordered_bands(df.get("band"))
    classes = unique_labels_in(df.get("label"))
    if not bands or not classes:
        return
    if len(classes) < 2:
        # Fall back to single-axes pooled plot
        _ribbon_plot_per_band(
            df, x_col=x_col, value_col=value_col,
            output_path=output_path, title=title,
            xlabel=xlabel, ylabel=ylabel,
            x_to_minutes=x_to_minutes,
            band_hz_ranges=band_hz_ranges,
        )
        return

    n_rows = len(bands)
    n_cols = len(classes)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(3.0, 2.6 * n_cols), max(2.4, 1.8 * n_rows)),
        sharex=True, sharey="row", squeeze=False,
    )

    plotted_any = False
    for r, band in enumerate(bands):
        for c, lab in enumerate(classes):
            ax = axes[r, c]
            sub = df[(df["band"] == band) & (df["label"] == lab)]
            if sub.empty or x_col not in sub.columns:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=FONT_LABEL * 0.9)
                _style_axes(ax, grid="major", minor_ticks=False)
                if c == 0:
                    ax.set_ylabel(
                        _format_band_label_with_hz(band, band_hz_ranges),
                        fontsize=FONT_LABEL * 0.85,
                    )
                if r == 0:
                    ax.set_title(class_label_for(lab),
                                 fontsize=FONT_TITLE * 0.85,
                                 fontweight="normal")
                continue
            grouped = sub.groupby(x_col)[value_col]
            med = grouped.median()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)
            if med.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=FONT_LABEL * 0.9)
                _style_axes(ax, grid="major", minor_ticks=False)
                if c == 0:
                    ax.set_ylabel(
                        _format_band_label_with_hz(band, band_hz_ranges),
                        fontsize=FONT_LABEL * 0.85,
                    )
                if r == 0:
                    ax.set_title(class_label_for(lab),
                                 fontsize=FONT_TITLE * 0.85,
                                 fontweight="normal")
                continue
            xs = np.asarray(med.index.to_list(), dtype=float)
            if x_to_minutes is not None:
                xs = xs * float(x_to_minutes)
            color = class_color_for(lab)
            ax.plot(xs, med.to_numpy(), color=color, lw=1.0)
            ax.fill_between(
                xs, q1.to_numpy(), q3.to_numpy(),
                color=color, alpha=0.20, lw=0,
            )
            plotted_any = True
            if c == 0:
                ax.set_ylabel(
                    _format_band_label_with_hz(band, band_hz_ranges),
                    fontsize=FONT_LABEL * 0.85,
                )
            if r == 0:
                ax.set_title(class_label_for(lab),
                             fontsize=FONT_TITLE * 0.85,
                             fontweight="normal")
            if r == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=FONT_LABEL * 0.85)
            _style_axes(ax, grid="major", minor_ticks=False)

    if not plotted_any:
        plt.close(fig)
        return

    fig.suptitle(title, fontsize=FONT_TITLE, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_band_horizon_error_by_class(
    per_horizon_df: pd.DataFrame,
    output_path: Path,
    *,
    value_col: str = "mse",
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Grid: per-band horizon-error ribbons split by class."""
    _grid_ribbon_by_class(
        per_horizon_df,
        x_col="h",
        value_col=value_col,
        output_path=output_path,
        title="Forecast error by horizon — per band, per class",
        xlabel="horizon step h",
        ylabel=value_col,
        band_hz_ranges=band_hz_ranges,
    )


def plot_band_anchor_error_by_class(
    per_anchor_df: pd.DataFrame,
    output_path: Path,
    *,
    value_col: str = "mse",
    decim_step_seconds: float = 4.0,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Grid: per-band anchor-error ribbons split by class."""
    _grid_ribbon_by_class(
        per_anchor_df,
        x_col="t",
        value_col=value_col,
        output_path=output_path,
        title="Forecast error by anchor position — per band, per class",
        xlabel="anchor position (min)",
        ylabel=value_col,
        x_to_minutes=decim_step_seconds / 60.0,
        band_hz_ranges=band_hz_ranges,
    )


def _kind_color_for(kind: str) -> str:
    """Stable color per coefficient-kind label (st_S0, st_S1, ph_*)."""
    palette = {
        "st_S0":    COLOR_GRAY,
        "st_S1":    COLOR_BLUE,
        "ph_diag":  COLOR_GREEN,
        "ph_h2":    COLOR_ORANGE,
        "ph_h3":    COLOR_VERMILLION,
        "ph_other": COLOR_PURPLE,
    }
    return palette.get(str(kind), COLOR_GRAY)


def plot_per_channel_mse_vs_freq(
    per_channel_df: pd.DataFrame,
    channel_metadata_df: pd.DataFrame,
    output_path: Path,
    *,
    band_hz_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    by_class: bool = False,
) -> None:
    """Mean per-channel forecast MSE vs centre frequency, coloured by kind.

    Two stacked panels: scattering (st_S0 + st_S1) on top, phase (ph_*)
    on bottom. Each point is one channel; x = ``freq_hz_primary`` (log
    scale), y = mean forecast MSE across all samples (per class when
    ``by_class=True``).

    Args:
        per_channel_df: Long-format frame from
            ``frequency_band_forecast/per_channel/per_channel_forecast.csv``;
            must carry ``channel, kind, freq_hz_primary, mse_total``.
        channel_metadata_df: Static channel→band map (unused for plotting
            here but kept in the signature so downstream callers can pass
            it without an extra branch).
        output_path: Destination PDF.
        band_hz_ranges: Optional dict of band→(low, high) Hz; when
            supplied, vertical dashed lines are drawn at each band
            boundary in the scattering panel for orientation.
        by_class: When True, plot one curve per class on each panel
            instead of pooling.

    No-ops silently if the input is empty.
    """
    if per_channel_df is None or per_channel_df.empty:
        return
    needed = {"channel", "kind", "freq_hz_primary", "mse_total"}
    if not needed.issubset(per_channel_df.columns):
        return

    if by_class and "label" in per_channel_df.columns:
        groupers: list = ["channel", "kind", "freq_hz_primary", "label"]
    else:
        groupers = ["channel", "kind", "freq_hz_primary"]
    agg = per_channel_df.groupby(groupers, as_index=False)["mse_total"].mean()
    agg = agg.rename(columns={"mse_total": "mean_mse"})

    # Pull harmonic_ratio per channel for the phase-panel legend table.
    if "harmonic_ratio" in per_channel_df.columns:
        ch_harmonic = (
            per_channel_df.groupby("channel", as_index=False)["harmonic_ratio"]
            .first()
            .set_index("channel")["harmonic_ratio"]
        )
    else:
        ch_harmonic = pd.Series(dtype=float)

    st_mask = agg["kind"].astype(str).isin(("st_S0", "st_S1"))
    ph_mask = agg["kind"].astype(str).str.startswith("ph_")

    # Two-column layout: left column = the two scatter panels (scattering
    # on top, phase on bottom); right column = a side legend table per
    # panel. Replacing the previous inline ``ax.annotate`` worst/best
    # labels — which had no collision avoidance and piled up at similar
    # frequencies — with circled-number badges on the data points and a
    # tabular legend in its own axis means collisions are impossible by
    # construction.
    fig = plt.figure(figsize=(10.5, 6.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[1.0, 0.55],
        hspace=0.32, wspace=0.06,
    )
    axes = [fig.add_subplot(gs[r, 0]) for r in range(2)]
    axes_legend = [fig.add_subplot(gs[r, 1]) for r in range(2)]

    panels = [
        ("Scattering channels (S0 + S1)", st_mask, axes[0], axes_legend[0], False),
        ("Phase channels (ph_diag / ph_h2 / ph_h3 / ph_other)",
         ph_mask, axes[1], axes_legend[1], True),
    ]

    classes = unique_labels_in(agg.get("label")) if by_class else []

    for title, mask, ax, ax_legend, is_phase_panel in panels:
        sub = agg.loc[mask]
        ax_legend.set_axis_off()
        if sub.empty:
            ax.text(
                0.5, 0.5, "no channels for this kind group",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=FONT_LABEL,
            )
            ax.set_title(title, fontsize=FONT_TITLE * 0.9, fontweight="normal")
            _style_axes(ax, grid="major", minor_ticks=False)
            continue

        if by_class and classes:
            for lab in classes:
                cls_sub = sub[sub["label"] == lab]
                if cls_sub.empty:
                    continue
                cls_sub = cls_sub.sort_values("freq_hz_primary")
                xs = cls_sub["freq_hz_primary"].to_numpy(dtype=float)
                ys = cls_sub["mean_mse"].to_numpy(dtype=float)
                ax.plot(
                    xs, ys, "o-",
                    color=class_color_for(lab),
                    markersize=3.5, lw=0.9, alpha=0.85,
                    label=class_label_for(lab),
                )
        else:
            for kind, kind_sub in sub.groupby("kind"):
                kind_sub = kind_sub.sort_values("freq_hz_primary")
                xs = kind_sub["freq_hz_primary"].to_numpy(dtype=float)
                ys = kind_sub["mean_mse"].to_numpy(dtype=float)
                ax.scatter(
                    xs, ys, s=18,
                    color=_kind_color_for(str(kind)),
                    edgecolors=COLOR_BLACK, linewidths=0.3,
                    label=str(kind), alpha=0.9,
                )

        # Vertical dashed lines at clinical-4-band boundaries (skip
        # 0.0 and inf so we don't pollute the log scale).
        if band_hz_ranges is not None:
            for _, (lo, hi) in band_hz_ranges.items():
                for boundary in (lo, hi):
                    if 0.0 < boundary < float("inf"):
                        ax.axvline(
                            boundary, color=COLOR_GRAY, ls="--",
                            lw=0.4, alpha=0.5,
                        )

        # Numbered badges for the top-3 worst (1..3, vermillion) and
        # top-3 best (4..6, green) channels — pooled view only. The
        # per-class layout already has too many series to make
        # numbered annotations useful, so we skip badges in that mode.
        if not by_class:
            pooled = sub.groupby(
                ["channel", "kind", "freq_hz_primary"], as_index=False,
            )["mean_mse"].mean()
            worst3 = pooled.sort_values("mean_mse", ascending=False).head(3).copy()
            best3 = pooled.sort_values("mean_mse", ascending=True).head(3).copy()
            worst3["rank"] = np.arange(1, len(worst3) + 1)
            worst3["color"] = COLOR_VERMILLION
            worst3["label_kind"] = "worst"
            best3["rank"] = np.arange(4, 4 + len(best3))
            best3["color"] = COLOR_GREEN
            best3["label_kind"] = "best"

            for _, row in pd.concat([worst3, best3], ignore_index=True).iterrows():
                if not (float(row["freq_hz_primary"]) > 0):
                    continue
                ax.scatter(
                    float(row["freq_hz_primary"]),
                    float(row["mean_mse"]),
                    marker="o", s=110,
                    facecolors="white",
                    edgecolors=row["color"], linewidths=1.6,
                    zorder=10,
                )
                ax.text(
                    float(row["freq_hz_primary"]),
                    float(row["mean_mse"]),
                    str(int(row["rank"])),
                    ha="center", va="center",
                    fontsize=FONT_LABEL * 0.75,
                    color=row["color"], fontweight="bold",
                    zorder=11,
                )

            # Side-panel legend: render a compact table that maps each
            # numbered badge back to (channel, kind, freq, MSE) — plus
            # ``harmonic_ratio`` for phase channels. Reviewers can scan
            # the table for context without colours getting in the way.
            header = ["#", "ch", "kind", "Hz", "MSE"]
            if is_phase_panel:
                header.append("p")

            def _row_to_cells(row: Any) -> List[str]:
                cells = [
                    f"{int(row['rank'])}",
                    f"{int(row['channel'])}",
                    str(row["kind"]),
                    _format_hz_tick(float(row["freq_hz_primary"])),
                    f"{float(row['mean_mse']):.4g}",
                ]
                if is_phase_panel:
                    p_val = ch_harmonic.get(int(row["channel"]), float("nan"))
                    if p_val is None or not np.isfinite(float(p_val)):
                        cells.append("—")
                    else:
                        cells.append(f"{float(p_val):.2g}")
                return cells

            cell_rows: List[List[str]] = []
            row_colors: List[str] = []
            for _, row in pd.concat([worst3, best3], ignore_index=True).iterrows():
                cell_rows.append(_row_to_cells(row))
                row_colors.append(str(row["color"]))

            if cell_rows:
                table = ax_legend.table(
                    cellText=cell_rows,
                    colLabels=header,
                    cellLoc="center",
                    loc="upper center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(FONT_LEGEND * 0.85)
                table.scale(1.0, 1.25)
                # Colour the leading "#" cell of each row to match its
                # badge so the cross-reference is unambiguous even in
                # greyscale prints.
                for r_idx, color in enumerate(row_colors):
                    cell = table[(r_idx + 1, 0)]    # +1 for header row
                    cell.set_text_props(color=color, fontweight="bold")
                ax_legend.set_title(
                    "Top-3 worst (red) / best (green)",
                    fontsize=FONT_LABEL * 0.85, pad=2,
                )

        ax.set_xscale("log")
        ax.set_xlabel("centre frequency (Hz)", fontsize=FONT_LABEL)
        ax.set_ylabel("mean forecast MSE", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE * 0.9, fontweight="normal")
        ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND * 0.85)
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.10)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_phase_harmonic_mse(
    per_channel_df: pd.DataFrame,
    channel_metadata_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Mean per-channel MSE vs phase harmonic ratio (phase channels only).

    Uses the per-channel CSV plus the channel-metadata table to plot
    one point per phase channel: x = ``harmonic_ratio`` (the power
    ``p`` from ``select_fhr_phase_coefficients``), y = mean forecast
    MSE across all samples, color = ``freq_hz_primary`` (target wavelet
    centre frequency in Hz, viridis), marker = coefficient ``kind``.

    Args:
        per_channel_df: Long-format per-channel CSV.
        channel_metadata_df: Static channel metadata. Used to filter to
            phase channels and pull the ``xi_j_hz`` color value.
        output_path: Destination PDF.

    No-ops if the input has no phase channels.
    """
    if per_channel_df is None or per_channel_df.empty:
        return
    needed = {"channel", "kind", "harmonic_ratio", "freq_hz_primary", "mse_total"}
    if not needed.issubset(per_channel_df.columns):
        return

    sub = per_channel_df.loc[
        per_channel_df["kind"].astype(str).str.startswith("ph_")
    ]
    if sub.empty:
        return

    agg = sub.groupby(
        ["channel", "kind", "harmonic_ratio", "freq_hz_primary"],
        as_index=False,
    )["mse_total"].mean().rename(columns={"mse_total": "mean_mse"})
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    # Colour scale = primary (target) frequency in Hz.
    freqs = agg["freq_hz_primary"].to_numpy(dtype=float)
    finite_freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    if finite_freqs.size > 0:
        vmin = float(np.min(finite_freqs))
        vmax = float(np.max(finite_freqs))
    else:
        vmin, vmax = 0.0, 1.0
    cmap = plt.colormaps.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    marker_for_kind = {
        "ph_diag":  "o",
        "ph_h2":    "s",
        "ph_h3":    "^",
        "ph_other": "D",
    }
    for kind, kind_sub in agg.groupby("kind"):
        marker = marker_for_kind.get(str(kind), "x")
        xs = kind_sub["harmonic_ratio"].to_numpy(dtype=float)
        ys = kind_sub["mean_mse"].to_numpy(dtype=float)
        cs = kind_sub["freq_hz_primary"].to_numpy(dtype=float)
        ax.scatter(
            xs, ys, c=cs, cmap=cmap, norm=norm,
            marker=marker, s=42, edgecolors=COLOR_BLACK, linewidths=0.4,
            label=str(kind), alpha=0.95,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("xi_j (target Hz)", fontsize=FONT_LABEL * 0.85)
    cbar.ax.tick_params(labelsize=FONT_LABEL * 0.7)

    ax.set_xlabel("phase harmonic ratio p (xi_j / xi_i)", fontsize=FONT_LABEL)
    ax.set_ylabel("mean forecast MSE", fontsize=FONT_LABEL)
    ax.set_title(
        "Per-channel forecast MSE vs phase harmonic ratio",
        fontsize=FONT_TITLE * 0.9, fontweight="normal",
    )
    ax.legend(
        loc="best", frameon=True, fontsize=FONT_LEGEND * 0.85,
        title="kind",
    )
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Frequency x horizon-step heatmaps + phase-harmonic comodulograms.
#
# These render the channel x horizon MSE grid produced by
# :func:`run_frequency_band_forecast_analysis`. Three flavours:
#
# 1. :func:`plot_freq_horizon_heatmap_scattering` — flat Hz-labelled
#    heatmap for the 43 scattering channels (1-D frequency identity).
# 2. :func:`plot_freq_horizon_heatmap_phase_by_kind` — 4 row-stacked
#    heatmaps for the phase channels split by ``kind`` (``ph_diag``,
#    ``ph_h2``, ``ph_h3``, ``ph_other``), each row annotated with the
#    driver/response frequency pair $(\xi_i, \xi_j)$ and harmonic
#    ratio $p$.
# 3. :func:`plot_phase_comodulograms_by_horizon` — small-multiples
#    comodulogram per horizon snapshot, axes $\log_2 \xi_i$ vs
#    $\log_2 \xi_j$, with dashed harmonic ridges $\xi_j = p\,\xi_i$.
# ----------------------------------------------------------------------


def _format_hz_tick(value: float) -> str:
    """Format a Hz frequency for axis ticks with adaptive precision."""
    if not np.isfinite(value):
        return "DC"
    if value <= 0.0:
        return "DC"
    if value >= 1.0:
        return f"{value:.2f} Hz"
    if value >= 0.1:
        return f"{value:.3f} Hz"
    if value >= 0.01:
        return f"{value:.4f} Hz"
    return f"{value:.5f} Hz"


def plot_freq_horizon_heatmap_scattering(
    mse_grid: np.ndarray,
    freq_hz: np.ndarray,
    channel_ids: np.ndarray,
    horizon: int,
    fs_hz: float,
    output_path: Path,
    *,
    n_samples: int,
    n_anchors_total: int = 0,
    log_color: bool = True,
    metric_label: str = "Forecast MSE",
) -> None:
    r"""Render a (frequency $\times$ horizon-step) MSE heatmap for the scattering block.

    Rows index the 43 scattering channels sorted from highest to lowest
    centre frequency $\xi$ (in Hz), so the top of the figure shows
    beat-to-beat content and the bottom shows the slow-baseline /
    DC channel. The DC ``st_S0`` row (no centre frequency) is rendered
    at the bottom with an explicit ``DC`` tick label.

    Each cell shows the cross-sample mean of the per-(sample, channel,
    horizon-step) MSE produced by
    :func:`compute_per_channel_per_horizon_forecast_metrics`.

    Args:
        mse_grid: ``(n_rows, H_d)`` array of MSE values, already sorted
            so that ``mse_grid[0]`` is the highest-frequency row.
        freq_hz: ``(n_rows,)`` array of centre frequencies in Hz.
            ``np.nan`` indicates the DC / S_0 channel.
        channel_ids: ``(n_rows,)`` array of original channel indices in
            the 87-channel forecast space, used in the right-margin
            annotations.
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz (used to render the secondary
            x-axis in seconds).
        output_path: Destination PDF path. A sibling ``.png`` is also
            written.
        n_samples: Number of samples averaged into ``mse_grid``.
        n_anchors_total: Total ``samples * valid_anchors`` averaged.
        log_color: Use a log-scaled colormap (default).
        metric_label: Colorbar label prefix (default ``"Forecast MSE"``).
    """
    if mse_grid.ndim != 2 or mse_grid.size == 0:
        return
    n_rows, n_cols = mse_grid.shape
    if int(n_cols) != int(horizon):
        return

    # Layout uses an explicit 3-column gridspec so the heatmap, the
    # right-margin channel-id column and the colorbar each get a
    # guaranteed column. This avoids the legacy ``tight_layout`` /
    # ``twinx``-spillover that caused ch labels to overlap the colorbar.
    fig_h = max(4.0, 0.18 * float(n_rows) + 1.5)
    fig_w = max(7.5, 0.45 * float(n_cols) + 4.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, 0.10, 0.025],
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_label = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    finite_vals = mse_grid[np.isfinite(mse_grid) & (mse_grid > 0.0)]
    if log_color and finite_vals.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_vals, 1.0))
        vmax = float(np.percentile(finite_vals, 99.0))
        if vmin <= 0 or not np.isfinite(vmin):
            vmin = float(np.min(finite_vals))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(
            vmin=float(np.nanmin(mse_grid)),
            vmax=float(np.nanmax(mse_grid)),
        )

    im = ax.imshow(
        mse_grid, aspect="auto", origin="upper",
        cmap="magma", norm=norm,
        interpolation="nearest",
    )

    # Y-axis: exact Hz tick labels, descending. Suppress every other
    # label past 30 rows so they remain readable.
    label_step = 1 if n_rows <= 30 else int(np.ceil(n_rows / 30.0))
    yticks = list(range(0, n_rows, label_step))
    if (n_rows - 1) not in yticks:
        yticks.append(n_rows - 1)
    yticklabels = [_format_hz_tick(float(freq_hz[i])) for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(r"Centre frequency $\xi$ (Hz, high $\rightarrow$ low)",
                  fontsize=FONT_LABEL)

    # Mark DC band (NaN freq) with a dashed horizontal line if present.
    dc_rows = np.where(~np.isfinite(freq_hz))[0]
    for r in dc_rows:
        ax.axhline(r - 0.5, color=COLOR_BLACK, linestyle="--",
                   linewidth=0.5, alpha=0.7)

    # X-axis: horizon step + seconds annotation.
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([
        f"$h={k}$\n({k / float(fs_hz):.2f} s)" for k in range(n_cols)
    ])
    ax.set_xlabel(
        "Horizon step (and elapsed time at "
        f"$f_s = {fs_hz:g}$ Hz)",
        fontsize=FONT_LABEL,
    )

    # Right-margin annotations: original channel index per row, on a
    # dedicated axis so labels never extend into the colorbar's column.
    ax_label.set_xlim(0.0, 1.0)
    ax_label.set_ylim(ax.get_ylim())
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    for spine in ax_label.spines.values():
        spine.set_visible(False)
    for i in yticks:
        ax_label.text(
            0.02, float(i),
            f"ch {int(channel_ids[i])}",
            ha="left", va="center",
            fontsize=plt.rcParams["ytick.labelsize"],
            transform=ax_label.transData,
        )
    ax_label.set_title(
        "Channel ID\n(87-ch space)",
        fontsize=FONT_LABEL * 0.75, pad=4, loc="left",
    )

    # Colorbar on its own dedicated axis (no make_axes_locatable, no
    # subplots_adjust magic).
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        f"{metric_label} (log scale)" if log_color else metric_label,
        fontsize=FONT_LABEL,
    )
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    cbar.ax.set_title(
        f"avg over\n{n_samples} samples",
        fontsize=plt.rcParams["axes.labelsize"] * 0.7, pad=4,
    )

    suptitle = (
        "Scattering forecast quality — channel × horizon step\n"
        rf"$f_s = {fs_hz:g}$ Hz, $H_d = {horizon}$, $n$ = {n_samples} samples"
    )
    if n_anchors_total > 0:
        suptitle += f" × {n_anchors_total // max(n_samples, 1)} valid anchors"
    ax.set_title(suptitle, fontsize=FONT_TITLE * 0.95, pad=10)

    _style_axes(ax, grid="none", minor_ticks=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.13)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    png_path = Path(str(output_path).replace(".pdf", ".png"))
    if png_path != output_path:
        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


_PHASE_KIND_ORDER: Tuple[str, ...] = ("ph_diag", "ph_h2", "ph_h3", "ph_other")
_PHASE_KIND_TITLE: Dict[str, str] = {
    "ph_diag":  r"$p \approx 1$ — autocorrelation ($\xi_i = \xi_j$)",
    "ph_h2":    r"$p \approx 2$ — second-harmonic coupling",
    "ph_h3":    r"$p \approx 3$ — third-harmonic coupling",
    "ph_other": r"residual cross-frequency pairs",
}
_PHASE_KIND_MARKER: Dict[str, str] = {
    "ph_diag": "o", "ph_h2": "^", "ph_h3": "s", "ph_other": "P",
}


def plot_freq_horizon_heatmap_phase_by_kind(
    long_df: pd.DataFrame,
    horizon: int,
    fs_hz: float,
    output_path: Path,
    *,
    n_samples: int,
    log_color: bool = True,
    metric_label: str = "Forecast MSE",
) -> None:
    r"""Phase-harmonic forecast quality heatmap, faceted by coefficient kind.

    Renders one panel per ``kind`` in $\{$``ph_diag``, ``ph_h2``,
    ``ph_h3``, ``ph_other``$\}$. Each panel sorts rows by the response
    frequency $\xi_j$ in descending order and shows a (channel
    $\times$ horizon-step) MSE strip with a shared color scale.

    Each row carries a right-margin annotation with the
    $(\xi_i, \xi_j, p)$ triple so the dual-frequency identity of the
    coefficient is preserved on the page.

    Args:
        long_df: Long-format dataframe (one row per channel x horizon)
            with columns ``channel, kind, freq_hz_primary,
            freq_hz_secondary, harmonic_ratio, h, mse_mean``. Phase
            channels only (``kind`` starts with ``ph_``).
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz.
        output_path: Destination PDF path.
        n_samples: Number of samples in the underlying mean.
        log_color: Use ``LogNorm`` for the shared colorbar (default).
        metric_label: Colorbar label prefix.
    """
    if long_df is None or long_df.empty:
        return
    needed = {
        "channel", "kind", "freq_hz_primary", "freq_hz_secondary",
        "harmonic_ratio", "h", "mse_mean",
    }
    if not needed.issubset(long_df.columns):
        return

    panels: Dict[str, np.ndarray] = {}
    panel_meta: Dict[str, pd.DataFrame] = {}
    for kind in _PHASE_KIND_ORDER:
        sub = long_df[long_df["kind"] == kind]
        if sub.empty:
            continue
        sub_meta = sub[
            ["channel", "freq_hz_primary", "freq_hz_secondary", "harmonic_ratio"]
        ].drop_duplicates(subset=["channel"]).copy()
        # Sort by xi_j (response) descending; NaN sinks to the bottom.
        sort_keys = sub_meta["freq_hz_primary"].astype(float).to_numpy()
        sort_keys = np.where(np.isnan(sort_keys), -np.inf, sort_keys)
        order = np.argsort(-sort_keys, kind="stable")
        sub_meta = sub_meta.iloc[order].reset_index(drop=True)
        ch_order = sub_meta["channel"].astype(int).tolist()
        wide = (
            sub.pivot_table(
                index="channel", columns="h", values="mse_mean", aggfunc="mean",
            )
            .reindex(ch_order)
            .reindex(columns=range(horizon))
        )
        panels[kind] = wide.to_numpy(dtype=float)
        panel_meta[kind] = sub_meta

    if not panels:
        return

    # Shared color norm across panels.
    all_vals = np.concatenate([
        g[np.isfinite(g)] for g in panels.values()
    ])
    finite_pos = all_vals[all_vals > 0.0]
    if log_color and finite_pos.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_pos, 1.0))
        vmax = float(np.percentile(finite_pos, 99.0))
        if vmin <= 0:
            vmin = float(np.min(finite_pos))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif all_vals.size > 0:
        norm = plt.Normalize(
            vmin=float(np.nanmin(all_vals)),
            vmax=float(np.nanmax(all_vals)),
        )
    else:
        return

    n_panels = len(panels)
    height_ratios = [max(1.0, panels[k].shape[0] * 0.18 + 0.5)
                     for k in panels]
    total_h = float(sum(height_ratios)) + 1.5
    fig_w = max(9.0, 0.6 * float(horizon) + 6.5)

    # Three-column gridspec: heatmap | (xi_i, p) annotation column |
    # shared colorbar (spans every row). The annotation column gets a
    # generous width because phase labels are long
    # (e.g. ``\xi_i=0.456, p=2.0``); putting them on a dedicated axis
    # rather than a ``twinx`` removes any chance of bleeding into the
    # colorbar's column.
    fig = plt.figure(figsize=(fig_w, total_h))
    gs = fig.add_gridspec(
        nrows=n_panels, ncols=3,
        width_ratios=[1.0, 0.20, 0.022],
        height_ratios=height_ratios,
        wspace=0.04, hspace=0.22,
    )
    axes_main: List[Any] = []
    axes_label: List[Any] = []
    for r in range(n_panels):
        axes_main.append(fig.add_subplot(gs[r, 0]))
        axes_label.append(fig.add_subplot(gs[r, 1]))
    cax = fig.add_subplot(gs[:, 2])

    im = None
    panel_items = list(panels.items())
    for r, ((kind, grid), ax, ax_label) in enumerate(
        zip(panel_items, axes_main, axes_label)
    ):
        meta = panel_meta[kind]
        im = ax.imshow(
            grid, aspect="auto", origin="upper",
            cmap="magma", norm=norm, interpolation="nearest",
        )
        n_rows = grid.shape[0]
        # Y-axis: response frequency xi_j in Hz.
        label_step = 1 if n_rows <= 14 else int(np.ceil(n_rows / 14.0))
        yticks = list(range(0, n_rows, label_step))
        if (n_rows - 1) not in yticks:
            yticks.append(n_rows - 1)
        yticklabels = [
            _format_hz_tick(float(meta.iloc[i]["freq_hz_primary"]))
            for i in yticks
        ]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel(
            rf"$\xi_j$ (Hz)" + "\n" + _PHASE_KIND_TITLE[kind],
            fontsize=FONT_LABEL * 0.85,
        )

        # X-axis only on the bottom panel.
        if r == n_panels - 1:
            ax.set_xticks(np.arange(grid.shape[1]))
            ax.set_xticklabels([
                f"$h={k}$\n({k / float(fs_hz):.2f} s)"
                for k in range(grid.shape[1])
            ])
            ax.set_xlabel(
                "Horizon step (and elapsed time at "
                f"$f_s = {fs_hz:g}$ Hz)",
                fontsize=FONT_LABEL,
            )
        else:
            ax.set_xticks([])

        # Right-margin annotations on a dedicated axis (no twinx). For
        # ``ph_diag`` the (xi_i, xi_j) pair is degenerate so we fall
        # back to a single ``(autocorr)`` tag.
        if kind == "ph_diag":
            annot = ["(autocorr)" for _ in yticks]
        else:
            annot = []
            for i in yticks:
                xi_i = float(meta.iloc[i]["freq_hz_secondary"])
                p = float(meta.iloc[i]["harmonic_ratio"])
                annot.append(
                    rf"$\xi_i$={_format_hz_tick(xi_i)},  $p$={p:.2g}"
                )
        ax_label.set_xlim(0.0, 1.0)
        ax_label.set_ylim(ax.get_ylim())
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for spine in ax_label.spines.values():
            spine.set_visible(False)
        for i, label in zip(yticks, annot):
            ax_label.text(
                0.02, float(i), label,
                ha="left", va="center",
                fontsize=plt.rcParams["ytick.labelsize"] * 0.85,
                transform=ax_label.transData,
            )

        ax.set_title(
            f"{kind}  ($n_{{\\mathrm{{ch}}}} = {n_rows}$)",
            fontsize=FONT_TITLE * 0.85, loc="left", pad=4,
        )
        _style_axes(ax, grid="none", minor_ticks=False)

    # Shared colorbar on the right of every panel.
    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(
            f"{metric_label} (log scale)" if log_color else metric_label,
            fontsize=FONT_LABEL,
        )
        cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)

    fig.suptitle(
        "Phase-harmonic forecast quality — by coefficient kind\n"
        rf"$f_s = {fs_hz:g}$ Hz, $H_d = {horizon}$, "
        rf"$n$ = {n_samples} samples",
        fontsize=FONT_TITLE * 1.0, y=0.995,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.10)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    png_path = Path(str(output_path).replace(".pdf", ".png"))
    if png_path != output_path:
        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_phase_comodulograms_by_horizon(
    long_df: pd.DataFrame,
    horizon: int,
    fs_hz: float,
    output_path: Path,
    *,
    n_samples: int,
    snapshot_horizons: Optional[Tuple[int, ...]] = None,
    log_color: bool = True,
    metric_label: str = "Forecast MSE",
) -> None:
    r"""Cross-frequency comodulograms of phase-harmonic forecast quality.

    Renders small-multiples panels at $K$ horizon snapshots; each
    panel scatters the phase channels in a $\log_2 \xi_i$ (driver,
    x-axis) vs $\log_2 \xi_j$ (response, y-axis) plane, with marker
    colour encoding MSE at that horizon step. Dashed reference lines
    at $\xi_j = p \xi_i$ for $p \in \{1, 2, 3\}$ make the harmonic
    ridges visible.

    Inspired by the comodulogram convention from phase-amplitude
    coupling literature (Tort 2010, PMC2941206; Brainstorm PAC
    tutorial) and the wavelet-phase-harmonic covariance matrices of
    Mallat & Zhang 2020 (IMA).

    Args:
        long_df: Long-format phase-channel MSE dataframe; same schema
            as :func:`plot_freq_horizon_heatmap_phase_by_kind`.
        horizon: Forecast horizon $H_d$.
        fs_hz: Sampling frequency in Hz.
        output_path: Destination PDF.
        n_samples: Number of samples averaged in.
        snapshot_horizons: Horizon steps to render. Defaults to
            ``(0, H_d // 4, H_d // 2, H_d - 1)`` (deduplicated).
        log_color: Use ``LogNorm`` for shared colorbar.
        metric_label: Colorbar label prefix.
    """
    if long_df is None or long_df.empty:
        return
    needed = {
        "kind", "freq_hz_primary", "freq_hz_secondary",
        "harmonic_ratio", "h", "mse_mean",
    }
    if not needed.issubset(long_df.columns):
        return

    if snapshot_horizons is None:
        raw = (0, max(0, horizon // 4), max(0, horizon // 2),
               max(0, horizon - 1))
        seen: set = set()
        snapshots: list = []
        for k in raw:
            ki = int(k)
            if 0 <= ki < int(horizon) and ki not in seen:
                seen.add(ki)
                snapshots.append(ki)
        snapshot_horizons = tuple(snapshots)
    if not snapshot_horizons:
        return

    df = long_df.dropna(subset=["freq_hz_primary", "freq_hz_secondary"]).copy()
    df = df[(df["freq_hz_primary"] > 0) & (df["freq_hz_secondary"] > 0)]
    if df.empty:
        return

    df["log2_xi_i"] = np.log2(df["freq_hz_secondary"].astype(float))
    df["log2_xi_j"] = np.log2(df["freq_hz_primary"].astype(float))

    snapshot_dfs: Dict[int, pd.DataFrame] = {
        h: df[df["h"] == int(h)].copy() for h in snapshot_horizons
    }
    snapshot_dfs = {h: d for h, d in snapshot_dfs.items() if not d.empty}
    if not snapshot_dfs:
        return

    all_mse = np.concatenate([
        d["mse_mean"].to_numpy(dtype=float) for d in snapshot_dfs.values()
    ])
    finite_pos = all_mse[np.isfinite(all_mse) & (all_mse > 0.0)]
    if log_color and finite_pos.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_pos, 1.0))
        vmax = float(np.percentile(finite_pos, 99.0))
        if vmin <= 0:
            vmin = float(np.min(finite_pos))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(
            vmin=float(np.nanmin(all_mse)),
            vmax=float(np.nanmax(all_mse)),
        )

    cmap = plt.colormaps.get_cmap("magma")

    n_panels = len(snapshot_dfs)
    fig_w = max(4.0, 4.0 * float(n_panels))
    fig, axes = plt.subplots(
        1, n_panels, figsize=(fig_w, 4.5),
        squeeze=False, sharex=True, sharey=True,
    )
    axes = axes.flatten()

    # Shared axis range across panels for visual comparability.
    xi_i_all = df["log2_xi_i"].to_numpy(dtype=float)
    xi_j_all = df["log2_xi_j"].to_numpy(dtype=float)
    pad = 0.2
    ax_xmin, ax_xmax = float(xi_i_all.min()) - pad, float(xi_i_all.max()) + pad
    ax_ymin, ax_ymax = float(xi_j_all.min()) - pad, float(xi_j_all.max()) + pad

    sc = None
    for ax, (h_step, d) in zip(axes, snapshot_dfs.items()):
        for kind, kind_df in d.groupby("kind"):
            marker = _PHASE_KIND_MARKER.get(str(kind), "x")
            sc = ax.scatter(
                kind_df["log2_xi_i"].to_numpy(dtype=float),
                kind_df["log2_xi_j"].to_numpy(dtype=float),
                c=kind_df["mse_mean"].to_numpy(dtype=float),
                cmap=cmap, norm=norm,
                marker=marker, s=42,
                edgecolors=COLOR_BLACK, linewidths=0.4,
                label=str(kind), alpha=0.95,
            )

        # Harmonic ridges: xi_j = p * xi_i  ->  log2(xi_j) = log2(p) + log2(xi_i)
        xs = np.linspace(ax_xmin, ax_xmax, 100)
        for p_val, lbl in [(1.0, r"$p=1$"), (2.0, r"$p=2$"), (3.0, r"$p=3$")]:
            ys = np.log2(p_val) + xs
            ax.plot(
                xs, ys, color=COLOR_BLACK, linestyle="--",
                linewidth=0.6, alpha=0.6,
            )
            # Place label near top of line within axis bounds.
            if ys[-1] < ax_ymax:
                ax.text(
                    xs[-1], ys[-1], lbl, fontsize=FONT_LEGEND * 0.85,
                    ha="right", va="bottom", color=COLOR_BLACK, alpha=0.8,
                )

        ax.set_xlim(ax_xmin, ax_xmax)
        ax.set_ylim(ax_ymin, ax_ymax)
        ax.set_xlabel(r"$\log_2 \xi_i$ (driver, Hz scale)",
                      fontsize=FONT_LABEL)
        if ax is axes[0]:
            ax.set_ylabel(r"$\log_2 \xi_j$ (response, Hz scale)",
                          fontsize=FONT_LABEL)
        ax.set_title(
            rf"$h = {int(h_step)}$  ($t = {int(h_step) / float(fs_hz):.2f}$ s)",
            fontsize=FONT_TITLE * 0.95,
        )
        _style_axes(ax, grid="major", minor_ticks=False)

        # Translate a few x-ticks to actual Hz for readability.
        xt = ax.get_xticks()
        ax.set_xticks(xt)
        ax.set_xticklabels([
            f"{2.0 ** v:.3g}" if np.isfinite(v) else "" for v in xt
        ])
        yt = ax.get_yticks()
        ax.set_yticks(yt)
        ax.set_yticklabels([
            f"{2.0 ** v:.3g}" if np.isfinite(v) else "" for v in yt
        ])

    if sc is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.65])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(
            f"{metric_label} (log scale)" if log_color else metric_label,
            fontsize=FONT_LABEL,
        )
        cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
        cbar.outline.set_linewidth(0.6)
        cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)

    # Single legend collected from one panel.
    handles = []
    for kind in _PHASE_KIND_ORDER:
        if (long_df["kind"] == kind).any():
            handles.append(plt.Line2D(
                [0], [0],
                marker=_PHASE_KIND_MARKER.get(kind, "x"),
                color="white", markerfacecolor=COLOR_LIGHT_GRAY,
                markeredgecolor=COLOR_BLACK, markersize=7,
                linestyle="", label=kind,
            ))
    if handles:
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles),
            frameon=False, fontsize=FONT_LEGEND * 0.95,
            bbox_to_anchor=(0.45, -0.02),
        )

    fig.suptitle(
        "Phase-harmonic forecast quality — comodulogram by horizon snapshot\n"
        rf"axes = $(\log_2 \xi_i,\, \log_2 \xi_j)$;  dashed = harmonic ridges "
        rf"$\xi_j = p\,\xi_i$;  $n$ = {n_samples} samples,  "
        rf"$f_s = {fs_hz:g}$ Hz",
        fontsize=FONT_TITLE * 0.95, y=1.02,
    )
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    png_path = Path(str(output_path).replace(".pdf", ".png"))
    if png_path != output_path:
        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# KLD per-dim & PC visualisations (Phase 1 + Phase 2 share these primitives
# via the ``group_col`` keyword: pass ``"label"`` for per-class views and
# ``"subgroup"`` for cross-subgroup overlays).
# ---------------------------------------------------------------------------


def _kld_dim_columns(df: pd.DataFrame, d_z: int) -> List[str]:
    """Return the present ``kld_dim_*`` columns up to ``d_z`` in numeric order."""
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        if not isinstance(c, str) or not c.startswith("kld_dim_"):
            continue
        tail = c.split("_", 2)[-1]
        try:
            i = int(tail)
        except ValueError:
            continue
        if 0 <= i < d_z:
            cols.append((i, c))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]


def plot_kld_per_dim_heatmap(
    df: pd.DataFrame,
    d_z: int,
    output_path: Path,
    *,
    group_col: str = "label",
    n_samples_per_group: Optional[Dict[str, int]] = None,
    metric_label: str = "mean KLD per dim",
    log_color: bool = True,
) -> None:
    r"""Per-latent-dim mean KLD as a heatmap.

    Pivots ``df`` (typically ``histograms/histogram_metrics.csv``) on the
    ``kld_dim_0..kld_dim_{d_z-1}`` columns and renders rows = latent
    dim, columns = group (class label or subgroup name), color = mean
    KLD across samples in that group.

    Args:
        df: DataFrame with at least ``group_col`` and ``kld_dim_*``
            columns. Rows are per-sample.
        d_z: Number of latent dimensions to consider (the maximum
            ``kld_dim_*`` column index $+ 1$).
        output_path: Destination PDF path.
        group_col: Column to pivot on (default ``"label"``;
            Phase 2 callers pass ``"subgroup"``).
        n_samples_per_group: Optional ``{group_value -> n}`` mapping;
            when supplied, n-samples are appended to each column label.
        metric_label: Colorbar label (default ``"mean KLD per dim"``).
        log_color: Use ``LogNorm`` for the heatmap (default).
    """
    if df is None or df.empty or group_col not in df.columns:
        return
    dim_cols = _kld_dim_columns(df, int(d_z))
    if not dim_cols:
        return

    grouped = df.groupby(group_col)[dim_cols].mean()
    if grouped.empty:
        return

    # Stable column order (same as encountered) and float matrix.
    groups = list(grouped.index)
    matrix = grouped.to_numpy(dtype=float).T   # (d_z, n_groups)

    finite_pos = matrix[np.isfinite(matrix) & (matrix > 0.0)]
    if log_color and finite_pos.size > 0:
        from matplotlib.colors import LogNorm
        vmin = float(np.percentile(finite_pos, 1.0))
        vmax = float(np.percentile(finite_pos, 99.0))
        if vmin <= 0:
            vmin = float(np.min(finite_pos))
        if vmax <= vmin:
            vmax = vmin * 10.0 if vmin > 0 else 1.0
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(
            vmin=float(np.nanmin(matrix)),
            vmax=float(np.nanmax(matrix)),
        )

    n_rows, n_cols = matrix.shape
    fig_h = max(4.0, 0.28 * float(n_rows) + 1.5)
    fig_w = max(5.5, 0.9 * float(n_cols) + 3.5)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[1.0, 0.035], wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(
        matrix, aspect="auto", origin="lower",
        cmap="viridis", norm=norm, interpolation="nearest",
    )

    # Y-axis: latent dim index. Annotate every cell with its value to
    # make the heatmap readable in print as well as on screen.
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"dim {i}" for i in range(n_rows)],
                       fontsize=plt.rcParams["ytick.labelsize"])
    ax.set_ylabel("latent dimension", fontsize=FONT_LABEL)

    # X-axis: group labels with optional n-samples suffix.
    def _col_label(g: Any) -> str:
        if n_samples_per_group is None:
            return str(g)
        n = n_samples_per_group.get(g, n_samples_per_group.get(str(g)))
        if n is None:
            return str(g)
        return f"{g}\n(n={int(n)})"
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([_col_label(g) for g in groups], rotation=20,
                       ha="right", fontsize=plt.rcParams["xtick.labelsize"])
    ax.set_xlabel(group_col, fontsize=FONT_LABEL)

    # Per-cell numeric annotations — use a contrasting text colour so
    # the value is readable on both the bright and dark ends of the
    # colormap.
    if n_rows * n_cols <= 200:
        threshold = float(np.nanmedian(matrix))
        for r in range(n_rows):
            for c in range(n_cols):
                v = float(matrix[r, c])
                if not np.isfinite(v):
                    continue
                txt_color = "white" if v < threshold else "black"
                ax.text(
                    c, r, f"{v:.3g}",
                    ha="center", va="center",
                    fontsize=plt.rcParams["ytick.labelsize"] * 0.75,
                    color=txt_color,
                )

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        f"{metric_label} (log scale)" if log_color else metric_label,
        fontsize=FONT_LABEL,
    )
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"])
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)

    ax.set_title(
        f"Per-dim mean KLD by {group_col}",
        fontsize=FONT_TITLE * 0.95, pad=8,
    )
    _style_axes(ax, grid="none", minor_ticks=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.16)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_per_dim_violins_by_class(
    df: pd.DataFrame,
    d_z: int,
    output_path: Path,
    *,
    group_col: str = "label",
) -> None:
    r"""One violin per latent dim, coloured / split by ``group_col``.

    Distributional view of the same data the per-dim heatmap aggregates.
    Useful for spotting class-discriminative dimensions where the
    *spread* of KLD differs across groups, not just the mean.

    The grid is laid out 4 panels wide; rows wrap as needed for the
    full $d_z$ dims. Empty / single-class panels are silently filled
    with a placeholder.

    Args:
        df: DataFrame with ``group_col`` and ``kld_dim_*`` columns.
        d_z: Number of latent dimensions.
        output_path: Destination PDF path.
        group_col: Column to split violins by (default ``"label"``).
    """
    if df is None or df.empty or group_col not in df.columns:
        return
    dim_cols = _kld_dim_columns(df, int(d_z))
    if not dim_cols:
        return

    groups = list(df[group_col].dropna().unique())
    if not groups:
        return

    n_panels = len(dim_cols)
    ncols = 4
    nrows = int(np.ceil(n_panels / ncols))
    fig_h = max(2.6 * nrows, 3.5)
    fig_w = max(3.0 * ncols, 8.0)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h),
        sharex=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    palette: Dict[Any, str] = {}
    fallback_palette = (
        COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_VERMILLION,
        COLOR_PURPLE, COLOR_TEAL_DARK, COLOR_GRAY,
    )
    for i, g in enumerate(groups):
        # Use the canonical class palette for {1, 2, 3} when the
        # caller is feeding clinical labels; otherwise hash into the
        # fallback palette so the same group always gets the same
        # colour across runs.
        try:
            palette[g] = class_color_for(int(g))
        except Exception:
            palette[g] = fallback_palette[i % len(fallback_palette)]

    for idx, dim_col in enumerate(dim_cols):
        ax = axes_flat[idx]
        data: List[np.ndarray] = []
        positions: List[float] = []
        colors: List[str] = []
        labels: List[str] = []
        for j, g in enumerate(groups):
            sub = df[df[group_col] == g][dim_col].to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            data.append(sub)
            positions.append(float(j))
            colors.append(palette[g])
            labels.append(str(g))
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL * 0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(dim_col, fontsize=FONT_LABEL * 0.85)
            continue
        parts = ax.violinplot(
            data, positions=positions, widths=0.85,
            showmeans=False, showmedians=True,
        )
        for body, color in zip(parts.get("bodies", []), colors):
            body.set_facecolor(color)
            body.set_edgecolor(COLOR_BLACK)
            body.set_alpha(0.7)
        for spine_key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if spine_key in parts:
                parts[spine_key].set_edgecolor(COLOR_BLACK)
                parts[spine_key].set_linewidth(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=plt.rcParams["xtick.labelsize"] * 0.85,
                           rotation=20, ha="right")
        ax.set_title(dim_col, fontsize=FONT_LABEL * 0.85)
        _style_axes(ax, grid="major", minor_ticks=False)

    # Hide unused panels.
    for k in range(n_panels, nrows * ncols):
        axes_flat[k].set_visible(False)

    fig.suptitle(
        f"Per-dim KLD distributions by {group_col}",
        fontsize=FONT_TITLE, y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.06)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_trajectory_by_group(
    traj_df: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str = "label",
    metric_col: str = "kld_mean",
    time_col: str = "timestep",
    title: Optional[str] = None,
) -> None:
    r"""Mean $\pm$ SE trajectory of a per-time-step KLD column, one line per group.

    Aggregates ``traj_df`` (typically ``kld_pca/kld_pc_trajectory.csv``)
    by ``(group_col, time_col)`` and plots a mean $\pm$ SE ribbon for
    each group on shared axes.

    Args:
        traj_df: Per-(sample, timestep) dataframe. Must contain
            ``group_col``, ``time_col``, and ``metric_col``.
        output_path: Destination PDF.
        group_col: Column whose distinct values become separate
            lines / ribbons (default ``"label"``).
        metric_col: Numeric column to aggregate per timestep
            (default ``"kld_mean"``).
        time_col: Time-index column (default ``"timestep"``).
        title: Optional plot title; defaults to a sensible auto-title.
    """
    if traj_df is None or traj_df.empty:
        return
    needed = {group_col, time_col, metric_col}
    if not needed.issubset(traj_df.columns):
        return

    groups = list(traj_df[group_col].dropna().unique())
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    plotted = False
    for g in groups:
        sub = traj_df[traj_df[group_col] == g]
        if sub.empty:
            continue
        grouped = sub.groupby(time_col)[metric_col]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        n = grouped.count().clip(lower=1)
        se = std / np.sqrt(n.to_numpy())
        if mean.empty:
            continue
        xs = np.asarray(mean.index.to_list(), dtype=float)
        ys = mean.to_numpy()
        try:
            color = class_color_for(int(g))
        except Exception:
            color = _band_color_for(str(g))
        ax.plot(xs, ys, color=color, lw=1.6, label=str(g))
        ax.fill_between(
            xs, ys - se.to_numpy(), ys + se.to_numpy(),
            color=color, alpha=0.18, lw=0,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(f"{time_col} (model time index)", fontsize=FONT_LABEL)
    ax.set_ylabel(f"{metric_col} (mean ± SE)", fontsize=FONT_LABEL)
    ax.set_title(
        title or f"{metric_col} trajectory by {group_col}",
        fontsize=FONT_TITLE,
    )
    ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND * 0.9, title=group_col)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.14)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_selected_pc_trajectories_grid(
    traj_df: pd.DataFrame,
    selection: Optional[Dict[str, Any]],
    output_path: Path,
    *,
    group_col: str = "label",
    time_col: str = "timestep",
) -> None:
    r"""Small-multiples grid of selected-PC trajectories, one panel per PC.

    Reads the contrast-selected PC indices from ``selection.json``
    (written by :func:`select_pca_components`) and renders one panel
    per selected PC. Each panel is a mean $\pm$ SE trajectory of
    ``kld_pc_selected_{i}_t`` (or ``kld_pc{i}_t`` as a fallback) over
    ``time_col``, with one line per group.

    Args:
        traj_df: Per-(sample, timestep) dataframe with the per-time PC
            columns (``kld_pc_selected_{i}_t`` preferred,
            ``kld_pc{i}_t`` accepted as a fallback).
        selection: ``selection.json`` payload (may be ``None`` —
            in which case the function falls back to plotting
            ``kld_pc1_t``, ``kld_pc2_t``, ``kld_pc3_t``).
        output_path: Destination PDF.
        group_col: Column to split lines by (default ``"label"``).
        time_col: Time-index column (default ``"timestep"``).
    """
    if traj_df is None or traj_df.empty:
        return
    if group_col not in traj_df.columns or time_col not in traj_df.columns:
        return

    # Resolve which PC trajectory columns to plot.
    pc_cols: List[Tuple[str, str]] = []   # (column_name, panel_title)
    if selection and isinstance(selection.get("selected_indices_1based"), list):
        for one_based in selection["selected_indices_1based"]:
            col_sel = f"kld_pc_selected_{int(one_based)}_t"
            col_raw = f"kld_pc{int(one_based)}_t"
            if col_sel in traj_df.columns:
                pc_cols.append((col_sel, f"PC {int(one_based)} (selected)"))
            elif col_raw in traj_df.columns:
                pc_cols.append((col_raw, f"PC {int(one_based)}"))
    if not pc_cols:
        for k in range(1, 4):
            col = f"kld_pc{k}_t"
            if col in traj_df.columns:
                pc_cols.append((col, f"PC {k}"))
    if not pc_cols:
        return

    contrast_type = ""
    if selection and isinstance(selection.get("contrast_type"), str):
        contrast_type = selection["contrast_type"]

    # When ``group_col`` has no non-null values (label-free runs), fall
    # back to a single pooled trace per PC so the grid is informative
    # instead of an array of "no data" panels.
    groups = list(traj_df[group_col].dropna().unique())
    pooled_mode = len(groups) == 0

    n_panels = len(pc_cols)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig_h = max(3.0 * nrows, 3.5)
    fig_w = max(4.0 * ncols, 7.5)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h),
        sharex=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (col, panel_title) in enumerate(pc_cols):
        ax = axes_flat[idx]
        plotted = False
        iter_groups = [None] if pooled_mode else groups
        for g in iter_groups:
            if pooled_mode:
                sub = traj_df
                label = "pooled"
                color = COLOR_BLUE
            else:
                sub = traj_df[traj_df[group_col] == g]
                if sub.empty:
                    continue
                label = str(g)
                try:
                    color = class_color_for(int(g))
                except Exception:
                    color = _band_color_for(str(g))
            grouped = sub.groupby(time_col)[col]
            mean = grouped.mean()
            std = grouped.std().fillna(0.0)
            n = grouped.count().clip(lower=1)
            se = std / np.sqrt(n.to_numpy())
            if mean.empty:
                continue
            xs = np.asarray(mean.index.to_list(), dtype=float)
            ys = mean.to_numpy()
            ax.plot(xs, ys, color=color, lw=1.4, label=label)
            ax.fill_between(
                xs, ys - se.to_numpy(), ys + se.to_numpy(),
                color=color, alpha=0.18, lw=0,
            )
            plotted = True
        ax.set_title(panel_title, fontsize=FONT_LABEL * 0.95)
        ax.set_xlabel(time_col, fontsize=FONT_LABEL * 0.85)
        ax.set_ylabel("mean ± SE", fontsize=FONT_LABEL * 0.85)
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL * 0.8)
        _style_axes(ax, grid="major", minor_ticks=False)

    # Hide unused panels.
    for k in range(n_panels, nrows * ncols):
        axes_flat[k].set_visible(False)

    # One shared legend at the figure level.
    handles_labels = axes_flat[0].get_legend_handles_labels()
    if handles_labels and handles_labels[0]:
        fig.legend(
            handles_labels[0], handles_labels[1],
            loc="lower center", ncol=min(len(handles_labels[0]), 4),
            frameon=False, fontsize=FONT_LEGEND * 0.95,
            bbox_to_anchor=(0.5, -0.02), title=group_col,
        )

    suptitle = f"Selected-PC KLD trajectories by {group_col}"
    if contrast_type:
        suptitle += f" — selection: {contrast_type}"
    fig.suptitle(suptitle, fontsize=FONT_TITLE, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pc12_mean_trajectory_overlay(
    traj_df: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str = "subgroup",
    time_col: str = "timestep",
    pc1_col: str = "kld_pc1_t",
    pc2_col: str = "kld_pc2_t",
) -> None:
    r"""Per-group mean trajectory in (PC1, PC2) space with directional arrows.

    For each group, computes the mean PC1 and PC2 per timestep, draws
    the resulting curve, and overlays small arrows showing the
    direction of time progression.

    Args:
        traj_df: Per-(sample, timestep) trajectory frame.
        output_path: Destination PDF.
        group_col: Column whose unique values become separate
            trajectories (default ``"subgroup"``).
        time_col: Per-sample time-step column.
        pc1_col, pc2_col: PC trajectory columns (defaults match the
            standard ``kld_pc_trajectory.csv`` schema).
    """
    if traj_df is None or traj_df.empty:
        return
    needed = {group_col, time_col, pc1_col, pc2_col}
    if not needed.issubset(traj_df.columns):
        return

    groups = list(traj_df[group_col].dropna().unique())
    if not groups:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    plotted = False
    fallback_palette = (
        COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_VERMILLION,
        COLOR_PURPLE, COLOR_TEAL_DARK, COLOR_GRAY,
    )
    for i, g in enumerate(groups):
        sub = traj_df[traj_df[group_col] == g]
        if sub.empty:
            continue
        mean_per_t = sub.groupby(time_col)[[pc1_col, pc2_col]].mean()
        if mean_per_t.empty:
            continue
        xs = mean_per_t[pc1_col].to_numpy(dtype=float)
        ys = mean_per_t[pc2_col].to_numpy(dtype=float)
        try:
            color = class_color_for(int(g))
        except Exception:
            color = fallback_palette[i % len(fallback_palette)]
        ax.plot(xs, ys, "-o", color=color, lw=1.5, markersize=2.0,
                label=str(g))
        # Arrow at the median timestep -> later timestep to show direction.
        if xs.size >= 4:
            mid = xs.size // 2
            ax.annotate(
                "",
                xy=(xs[mid + 1], ys[mid + 1]),
                xytext=(xs[mid], ys[mid]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
            )
        # Mark start and end points.
        ax.scatter(xs[0], ys[0], color=color, marker="s",
                   s=40, edgecolors=COLOR_BLACK, linewidths=0.6,
                   zorder=11)
        ax.scatter(xs[-1], ys[-1], color=color, marker="*",
                   s=110, edgecolors=COLOR_BLACK, linewidths=0.6,
                   zorder=11)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("PC1 (mean over samples)", fontsize=FONT_LABEL)
    ax.set_ylabel("PC2 (mean over samples)", fontsize=FONT_LABEL)
    ax.set_title(
        f"Per-{group_col} mean trajectory in (PC1, PC2)\n"
        "■ = first timestep, ★ = last timestep",
        fontsize=FONT_TITLE * 0.95,
    )
    ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND * 0.9,
              title=group_col)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_kld_violins(
    df: pd.DataFrame,
    output_path: Path,
    *,
    group_col: str = "subgroup",
    metric_cols: Sequence[str] = ("kld_mean", "kld_sum", "kld_l2"),
) -> None:
    r"""Side-by-side violins of aggregate KLD metrics, one panel per metric.

    Args:
        df: Per-sample DataFrame with at least ``group_col`` and the
            requested metric columns.
        output_path: Destination PDF.
        group_col: Column to split violins by (default ``"subgroup"``).
        metric_cols: Per-sample metric columns to render, one panel each.
    """
    if df is None or df.empty or group_col not in df.columns:
        return
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return

    groups = list(df[group_col].dropna().unique())
    if not groups:
        return

    n_panels = len(metric_cols)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(max(4.5 * n_panels, 7.0), 3.6),
        sharey=False, squeeze=False,
    )
    axes_flat = axes.flatten()

    fallback_palette = (
        COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_VERMILLION,
        COLOR_PURPLE, COLOR_TEAL_DARK, COLOR_GRAY,
    )
    palette: Dict[Any, str] = {}
    for i, g in enumerate(groups):
        try:
            palette[g] = class_color_for(int(g))
        except Exception:
            palette[g] = fallback_palette[i % len(fallback_palette)]

    for idx, metric in enumerate(metric_cols):
        ax = axes_flat[idx]
        data: List[np.ndarray] = []
        positions: List[float] = []
        colors: List[str] = []
        labels: List[str] = []
        for j, g in enumerate(groups):
            sub = df[df[group_col] == g][metric].to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            data.append(sub)
            positions.append(float(j))
            colors.append(palette[g])
            labels.append(str(g))
        if not data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL * 0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(metric, fontsize=FONT_LABEL)
            continue
        parts = ax.violinplot(
            data, positions=positions, widths=0.85,
            showmeans=False, showmedians=True,
        )
        for body, color in zip(parts.get("bodies", []), colors):
            body.set_facecolor(color)
            body.set_edgecolor(COLOR_BLACK)
            body.set_alpha(0.7)
        for spine_key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if spine_key in parts:
                parts[spine_key].set_edgecolor(COLOR_BLACK)
                parts[spine_key].set_linewidth(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right",
                           fontsize=plt.rcParams["xtick.labelsize"] * 0.85)
        ax.set_title(metric, fontsize=FONT_LABEL)
        ax.set_ylabel(metric, fontsize=FONT_LABEL * 0.9)
        _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle(
        f"Aggregate KLD metrics by {group_col}",
        fontsize=FONT_TITLE, y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.20, wspace=0.30)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
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


_KLD_METRIC_LABELS: Dict[str, str] = {
    "kld_mean":  r"per-segment mean KLD",
    "kld_l2sq":  r"per-segment $\|\mathrm{KLD}\|_2^2$",
    "kld_max":   r"per-segment max KLD",
    "kld_std":   r"per-segment std KLD",
}


def plot_kld_segment_summary_vs_time(
    df: pd.DataFrame,
    output_path: Path,
    *,
    metric_col: str,
    title: Optional[str] = None,
    group_col: Optional[str] = None,
    n_samples: Optional[int] = None,
    palette: Optional[Dict[str, str]] = None,
) -> None:
    r"""Per-segment KLD summary as a function of time-before-birth.

    Aggregates per-(guid, epoch) rows of ``df`` (typically
    ``trajectory/kld_trajectory_raw.csv``) into 30-minute bins along the
    ``hours_before`` axis and plots mean $\pm$ SE for the chosen
    ``metric_col``. The function generalises ``plot_kld_trajectory``
    along two axes:

    * ``metric_col`` — supports ``"kld_mean"``, ``"kld_l2sq"`` (squared
      $L_2$ norm of the per-timestep KLD), ``"kld_max"`` (per-segment
      maximum), or any other numeric column.
    * ``group_col`` — when set (e.g. ``"label"`` for Phase 1,
      ``"subgroup"`` for Phase 2), one line per group is rendered on the
      same axes.

    Args:
        df: DataFrame with ``hours_before`` and ``metric_col``; may also
            carry ``group_col``.
        output_path: Destination PDF.
        metric_col: Numeric column to summarise per epoch (already a
            per-segment scalar — no further per-timestep aggregation).
        title: Optional figure title; auto-generated if ``None``.
        group_col: When set, render one mean $\pm$ SE line per group.
        n_samples: Optional sample-count annotation for the title.
        palette: Optional ``{group_value: color}`` mapping; unspecified
            groups fall back to ``_band_color_for``.
    """
    if df is None or df.empty:
        return
    if "hours_before" not in df.columns or metric_col not in df.columns:
        return

    work = df.copy()
    work["hour_bin"] = (work["hours_before"] * 2).round() / 2  # 30-min bins
    work = work[work["hour_bin"].notna() & work[metric_col].notna()]
    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    def _draw(sub: pd.DataFrame, color: str, label: str) -> bool:
        grouped = sub.groupby("hour_bin")[metric_col]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        n = grouped.count().clip(lower=1)
        se = std / np.sqrt(n.to_numpy())
        if mean.empty:
            return False
        xs = np.asarray(mean.index.to_list(), dtype=float)
        ys = mean.to_numpy()
        ax.plot(xs, ys, color=color, lw=1.4, marker="o", markersize=2.5,
                markevery=max(1, len(xs) // 20), label=label)
        ax.fill_between(xs, ys - se.to_numpy(), ys + se.to_numpy(),
                        alpha=0.20, color=color, lw=0)
        return True

    plotted = False
    if group_col and group_col in work.columns:
        groups = [g for g in work[group_col].dropna().unique().tolist()
                  if str(g) != "unknown"]
        for g in groups:
            sub = work[work[group_col] == g]
            if sub.empty:
                continue
            color: Optional[str] = None
            if palette is not None:
                color = palette.get(g) or palette.get(str(g))
            if color is None:
                try:
                    color = class_color_for(int(g))
                except Exception:
                    color = _band_color_for(str(g))
            plotted = _draw(sub, color, str(g)) or plotted
    else:
        plotted = _draw(work, COLOR_BLUE, _KLD_METRIC_LABELS.get(metric_col, metric_col))

    if not plotted:
        plt.close(fig)
        return

    metric_label = _KLD_METRIC_LABELS.get(metric_col, metric_col)
    ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL)
    ax.set_ylabel(f"{metric_label} (mean $\\pm$ SE)", fontsize=FONT_LABEL)
    auto_title = f"{metric_label} vs time before delivery"
    if group_col:
        auto_title = f"{auto_title} — by {group_col}"
    if n_samples is not None:
        auto_title = f"{auto_title} (n={int(n_samples)} samples)"
    ax.set_title(title or auto_title, fontsize=FONT_TITLE, fontweight="normal", pad=6)
    ax.invert_xaxis()
    ax.legend(loc="best", frameon=True, fontsize=FONT_LEGEND * 0.9,
              title=group_col if group_col else None)
    _style_axes(ax, grid="both", minor_ticks=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_kld_pc_trajectory_grid(
    df: pd.DataFrame,
    output_path: Path,
    *,
    n_components: int = 6,
    group_col: Optional[str] = None,
    n_samples: Optional[int] = None,
    palette: Optional[Dict[str, str]] = None,
    pc_prefix: str = "kld_pc_top",
) -> None:
    r"""Small-multiples grid (3 $\times$ 2) of the first ``n_components`` KLD-PC
    trajectories vs time before birth.

    For each PC $i \in [1, N]$, summarises the per-(guid, epoch)
    average of ``{pc_prefix}{i}_t`` (mean over the segment's timesteps)
    and plots mean $\pm$ SE in 30-minute ``hours_before`` bins. When
    ``group_col`` is set, one line per group is overlaid in each panel.

    The default ``pc_prefix="kld_pc_top"`` reads the
    first-N-by-eigenvalue PC trajectory columns emitted by
    :func:`collect_kld_trajectory` when called with ``pca_model_top``.
    Use ``pc_prefix="kld_pc"`` to plot the contrast-selected PCs
    instead.

    Args:
        df: Per-(sample, timestep) trajectory dataframe. Must contain
            ``hours_before`` and ``{pc_prefix}{i}_t`` for the chosen
            ``i``s.
        output_path: Destination PDF.
        n_components: Number of PCs to render (default 6 → 3 $\times$ 2 grid).
        group_col: Optional grouping column (``"label"`` Phase 1,
            ``"subgroup"`` Phase 2).
        n_samples: Optional sample-count annotation for the suptitle.
        palette: Optional ``{group_value: color}`` mapping.
        pc_prefix: Column-name prefix; expects ``{pc_prefix}{i}_t``.
    """
    if df is None or df.empty or "hours_before" not in df.columns:
        return
    if "guid" not in df.columns or "epoch" not in df.columns:
        return

    pc_cols = [f"{pc_prefix}{i + 1}_t" for i in range(int(n_components))
               if f"{pc_prefix}{i + 1}_t" in df.columns]
    if not pc_cols:
        return

    # Per-epoch summary: mean over the segment's timesteps for each PC.
    agg_cols = pc_cols + ["hours_before"]
    if group_col and group_col in df.columns:
        agg_cols.append(group_col)
    work = df[["guid", "epoch"] + agg_cols].copy()
    work = work[work["hours_before"].notna()]
    if work.empty:
        return
    group_keys: List[str] = ["guid", "epoch", "hours_before"]
    if group_col and group_col in work.columns:
        group_keys.append(group_col)
    epoch_means = work.groupby(group_keys, dropna=False)[pc_cols].mean().reset_index()
    epoch_means["hour_bin"] = (epoch_means["hours_before"] * 2).round() / 2
    epoch_means = epoch_means[epoch_means["hour_bin"].notna()]
    if epoch_means.empty:
        return

    n_panels = len(pc_cols)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.6, 2.4 * n_rows + 0.6),
        sharex=True,
        squeeze=False,
    )

    def _color_for(g: Any) -> str:
        if palette is not None:
            c = palette.get(g) or palette.get(str(g))
            if c:
                return c
        try:
            return class_color_for(int(g))
        except Exception:
            return _band_color_for(str(g))

    panel_handles: Dict[str, Any] = {}
    panel_labels: Dict[str, str] = {}

    for k, pc_col in enumerate(pc_cols):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        plotted = False
        if group_col and group_col in epoch_means.columns:
            groups = [g for g in epoch_means[group_col].dropna().unique().tolist()
                      if str(g) != "unknown"]
            for g in groups:
                sub = epoch_means[epoch_means[group_col] == g]
                grouped = sub.groupby("hour_bin")[pc_col]
                mean = grouped.mean()
                std = grouped.std().fillna(0.0)
                n = grouped.count().clip(lower=1)
                se = std / np.sqrt(n.to_numpy())
                if mean.empty:
                    continue
                xs = np.asarray(mean.index.to_list(), dtype=float)
                ys = mean.to_numpy()
                color = _color_for(g)
                line, = ax.plot(xs, ys, color=color, lw=1.2,
                                marker="o", markersize=2.0,
                                markevery=max(1, len(xs) // 20), label=str(g))
                ax.fill_between(xs, ys - se.to_numpy(), ys + se.to_numpy(),
                                alpha=0.18, color=color, lw=0)
                panel_handles[str(g)] = line
                panel_labels[str(g)] = str(g)
                plotted = True
        else:
            grouped = epoch_means.groupby("hour_bin")[pc_col]
            mean = grouped.mean()
            std = grouped.std().fillna(0.0)
            n = grouped.count().clip(lower=1)
            se = std / np.sqrt(n.to_numpy())
            if not mean.empty:
                xs = np.asarray(mean.index.to_list(), dtype=float)
                ys = mean.to_numpy()
                ax.plot(xs, ys, color=COLOR_BLUE, lw=1.2,
                        marker="o", markersize=2.0,
                        markevery=max(1, len(xs) // 20))
                ax.fill_between(xs, ys - se.to_numpy(), ys + se.to_numpy(),
                                alpha=0.22, color=COLOR_BLUE, lw=0)
                plotted = True

        ax.set_title(f"PC{k + 1}", fontsize=FONT_LABEL, pad=2)
        if r == n_rows - 1:
            ax.set_xlabel("Hours Before Birth", fontsize=FONT_LABEL * 0.9)
        if c == 0:
            ax.set_ylabel(f"{pc_col} (mean $\\pm$ SE)", fontsize=FONT_LABEL * 0.85)
        ax.invert_xaxis()
        _style_axes(ax, grid="major", minor_ticks=False)
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT_LABEL * 0.85)

    # Hide any unused axes in the final row.
    for k in range(n_panels, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].set_visible(False)

    if panel_handles:
        fig.legend(
            panel_handles.values(), panel_labels.values(),
            loc="upper center", ncol=min(len(panel_handles), 4),
            frameon=True, fontsize=FONT_LEGEND * 0.85,
            title=group_col,
            bbox_to_anchor=(0.5, 1.0),
        )

    suptitle = f"First {n_panels} KLD-PC trajectories vs time before delivery"
    if group_col:
        suptitle = f"{suptitle} — by {group_col}"
    if n_samples is not None:
        suptitle = f"{suptitle} (n={int(n_samples)} samples)"
    fig.suptitle(suptitle, fontsize=FONT_TITLE, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94 if panel_handles else 0.97])
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
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
    # Pre-filter None to keep ``np.unique`` from raising on object-dtype
    # arrays that hold only ``None`` (label-free datasets).
    _valid = [x for x in np.asarray(labels).tolist() if x is not None]
    unique_labels = sorted({x for x in _valid})

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
    # Pre-filter None to keep ``np.unique`` from raising on object-dtype
    # arrays that hold only ``None`` (label-free datasets).
    _valid = [x for x in np.asarray(labels).tolist() if x is not None]
    unique_labels = sorted({x for x in _valid})
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


# -----------------------------------------------------------------------------
# Lag-Attentive V1 Feature-Forecast Primitives
# -----------------------------------------------------------------------------


def plot_feature_forecast_heatmap(
    mu_full_avg: np.ndarray,
    y_plus_avg: np.ndarray,
    warmup: int,
    output_path: Path,
    *,
    fhr_st_end: int = 43,
    title: str = "Feature Forecast Heatmap",
) -> None:
    """Plot a 3-row prediction / truth / residual heatmap for a sample.

    Rows stack the 87-channel future forecast as ``(channels, T_valid)``
    imshow panels: row 0 prediction ``mu_full``, row 1 ground truth
    ``y_plus``, row 2 residual ``(mu_full - y_plus)``. A horizontal line
    marks the scattering/phase channel split at ``fhr_st_end`` (default 43).

    Args:
        mu_full_avg: Prediction averaged over horizon, shape ``(C, T_valid)``.
        y_plus_avg: Ground truth averaged over horizon, shape ``(C, T_valid)``.
        warmup: Number of initial anchors that are invalid (shaded gray).
        output_path: Destination PDF/PNG.
        fhr_st_end: Channel index separating scattering from phase (default 43).
        title: Figure title.
    """
    if mu_full_avg.shape != y_plus_avg.shape:
        raise ValueError(
            f"Shape mismatch: mu_full_avg {mu_full_avg.shape} vs "
            f"y_plus_avg {y_plus_avg.shape}"
        )
    C, T_valid = mu_full_avg.shape
    residual = mu_full_avg - y_plus_avg
    vmax = max(abs(mu_full_avg).max(), abs(y_plus_avg).max()) + 1e-12
    res_max = abs(residual).max() + 1e-12

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=True)
    titles = ("Prediction (mu_full)", "Truth (y_plus)", "Residual (mu_full - y_plus)")
    data = (mu_full_avg, y_plus_avg, residual)
    vlims = (vmax, vmax, res_max)

    for ax, panel, panel_title, vlim in zip(axes, data, titles, vlims):
        im = ax.imshow(
            panel,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        if 0 < fhr_st_end < C:
            ax.axhline(y=fhr_st_end - 0.5, color=COLOR_BLACK, linewidth=0.6, linestyle="--")
        if warmup > 0 and warmup < T_valid:
            ax.axvspan(-0.5, warmup - 0.5, color=COLOR_GRAY, alpha=0.15)
        ax.set_ylabel(panel_title, fontsize=FONT_LABEL)
        _add_colorbar(fig, im, ax, label="")

    axes[-1].set_xlabel("Anchor t", fontsize=FONT_LABEL)
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="normal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_error_by_horizon(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Forecast Error by Horizon Step",
) -> None:
    """Plot per-horizon-step MSE as a median + IQR ribbon.

    Expects a long-format DataFrame with columns ``[h, mse_step]`` and
    optionally ``mse_st``/``mse_ph`` for the channel-block overlays.

    Args:
        df: DataFrame produced by
            :func:`collect_forecast_errors_per_horizon`.
        output_path: Destination PDF/PNG.
        title: Figure title.
    """
    if df.empty or "h" not in df.columns:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    grouped = df.groupby("h")
    stats = grouped["mse_step"].agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75), "mean"])
    stats.columns = ["median", "q25", "q75", "mean"]
    h_vals = stats.index.to_numpy()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.fill_between(h_vals, stats["q25"], stats["q75"], color=COLOR_BLUE, alpha=0.20, label="IQR")
    ax.plot(h_vals, stats["median"], color=COLOR_BLUE, linewidth=1.2, label="median")
    ax.plot(h_vals, stats["mean"], color=COLOR_VERMILLION, linewidth=1.0, linestyle="--", label="mean")

    if "mse_st" in df.columns:
        st_med = grouped["mse_st"].median()
        ax.plot(h_vals, st_med, color=COLOR_GREEN, linewidth=0.9, label="mse_st median")
    if "mse_ph" in df.columns:
        ph_med = grouped["mse_ph"].median()
        ax.plot(h_vals, ph_med, color=COLOR_ORANGE, linewidth=0.9, label="mse_ph median")

    ax.set_xlabel("Horizon step h", fontsize=FONT_LABEL)
    ax.set_ylabel("MSE", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    ax.legend(loc="upper left", frameon=True)
    _style_axes(ax, grid="major", minor_ticks=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_uplift_histogram(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Baseline vs Full Uplift",
) -> None:
    """Two-panel histogram comparing baseline and full losses plus uplift.

    Left panel: overlaid histograms of ``l_full`` and ``l_base``. Right
    panel: histogram of ``uplift_rel`` with a zero line marker.

    Args:
        df: DataFrame with columns ``l_full, l_base, uplift_rel``.
        output_path: Destination PDF/PNG.
        title: Figure title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    ax = axes[0]
    l_full = df["l_full"].to_numpy() if "l_full" in df.columns else df.get("full_mse", pd.Series(dtype=float)).to_numpy()
    l_base = df["l_base"].to_numpy() if "l_base" in df.columns else df.get("base_mse_total", pd.Series(dtype=float)).to_numpy()
    bins = 40
    if l_full.size:
        ax.hist(l_full, bins=bins, color=COLOR_BLUE, alpha=0.6, label="full MSE",
                edgecolor=COLOR_BLACK, linewidth=0.5)
    if l_base.size:
        ax.hist(l_base, bins=bins, color=COLOR_ORANGE, alpha=0.5, label="baseline MSE",
                edgecolor=COLOR_BLACK, linewidth=0.5)
    ax.set_xlabel("Per-sample MSE", fontsize=FONT_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    ax.legend(loc="upper right", frameon=True)
    _style_axes(ax, grid="major", minor_ticks=False)

    ax = axes[1]
    up = df["uplift_rel"].to_numpy() if "uplift_rel" in df.columns else np.array([])
    if up.size:
        ax.hist(up, bins=bins, color=COLOR_GREEN, alpha=0.8,
                edgecolor=COLOR_BLACK, linewidth=0.5)
        ax.axvline(x=0.0, color=COLOR_BLACK, linewidth=0.8, linestyle="--")
        median = float(np.nanmedian(up))
        ax.axvline(x=median, color=COLOR_VERMILLION, linewidth=0.8, linestyle="-", label=f"median={median:.3f}")
        ax.legend(loc="upper right", frameon=True)
    ax.set_xlabel("uplift_rel = (l_base - l_full) / l_base", fontsize=FONT_LABEL)
    ax.set_ylabel("Count", fontsize=FONT_LABEL)
    _style_axes(ax, grid="major", minor_ticks=False)

    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="normal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residual_usage_trace(
    trace_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Residual Usage Over Time",
) -> None:
    """Plot the per-anchor residual norm trace as median + IQR across samples.

    Args:
        trace_df: Long-format DataFrame with columns ``[guid, t, delta_norm_t]``.
        output_path: Destination PDF/PNG.
        title: Figure title.
    """
    if trace_df.empty or "t" not in trace_df.columns:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    grouped = trace_df.groupby("t")["delta_norm_t"]
    t_vals = np.sort(trace_df["t"].unique())
    median = grouped.median().reindex(t_vals).to_numpy()
    q25 = grouped.quantile(0.25).reindex(t_vals).to_numpy()
    q75 = grouped.quantile(0.75).reindex(t_vals).to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(t_vals, q25, q75, color=COLOR_PURPLE, alpha=0.20, label="IQR")
    ax.plot(t_vals, median, color=COLOR_PURPLE, linewidth=1.2, label="median")
    ax.set_xlabel("Anchor t", fontsize=FONT_LABEL)
    ax.set_ylabel("||delta_mu_src||", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    ax.legend(loc="upper right", frameon=True)
    _style_axes(ax, grid="major", minor_ticks=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_lag_attention_heatmap(
    alpha_bar: np.ndarray,
    argmax_lag: np.ndarray,
    warmup: int,
    output_path: Path,
    *,
    title: str = "Lag Attention",
) -> None:
    """Plot the head-averaged lag attention map for a single sample.

    Draws a ``(T, L)`` heatmap with the argmax-lag curve overlaid and
    the warmup region shaded out. NaN entries (warmup) render as white.

    Args:
        alpha_bar: Head-averaged attention, shape ``(T, L)``. Warmup
            rows expected to contain NaN.
        argmax_lag: Argmax lag per anchor, shape ``(T,)``. Warmup entries
            should be ``-1``.
        warmup: Number of initial anchors that are invalid.
        output_path: Destination PDF/PNG.
        title: Figure title.
    """
    if alpha_bar.ndim != 2:
        raise ValueError(f"alpha_bar must be (T, L), got {alpha_bar.shape}")
    T, L = alpha_bar.shape

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    masked = np.ma.masked_invalid(alpha_bar)
    im = ax.imshow(
        masked.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=(-0.5, T - 0.5, -0.5, L - 0.5),
    )
    _add_colorbar(fig, im, ax, label="attention")

    # Overlay argmax lag curve, skipping warmup sentinel (-1).
    valid = argmax_lag >= 0
    if valid.any():
        t_idx = np.arange(T)
        ax.plot(
            t_idx[valid],
            argmax_lag[valid],
            color=COLOR_VERMILLION,
            linewidth=0.9,
            alpha=0.9,
            label="argmax lag",
        )
        ax.legend(loc="upper right", frameon=True)

    if warmup > 0:
        ax.axvspan(-0.5, warmup - 0.5, color=COLOR_GRAY, alpha=0.22)

    ax.set_xlabel("Anchor t", fontsize=FONT_LABEL)
    ax.set_ylabel("Lag k", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    _style_axes(ax, grid="none", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_te_lag_distribution(
    te_lag_mean: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    *,
    class_names: Optional[Dict[int, str]] = None,
    n_bootstrap: int = 500,
    title: str = "TE Lag Distribution by Class",
) -> None:
    """Plot per-class mean TE-lag trace with bootstrap 95% CI.

    Args:
        te_lag_mean: Per-sample time-averaged lag profile, shape ``(N, L)``.
        labels: Per-sample class labels, shape ``(N,)``.
        output_path: Destination PDF/PNG.
        class_names: Optional mapping ``{label: display_name}``. Defaults
            to integer labels.
        n_bootstrap: Bootstrap resamples for the class-mean CI.
        title: Figure title.
    """
    if te_lag_mean.ndim != 2:
        raise ValueError(f"te_lag_mean must be (N, L), got {te_lag_mean.shape}")
    N, L = te_lag_mean.shape

    if class_names is None:
        class_names = {1: "HEALTHY", 2: "ACIDOSIS", 3: "HIE"}

    # Pre-filter None before deduplicating: ``np.unique`` calls
    # ``.sort()`` on the underlying array, which raises ``TypeError``
    # on an object-dtype array of only ``None``.
    _valid_labels = [x for x in np.asarray(labels).tolist() if x is not None]
    unique_labels = sorted({int(x) for x in _valid_labels})
    palette = [COLOR_BLUE, COLOR_VERMILLION, COLOR_GREEN, COLOR_ORANGE, COLOR_PURPLE]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(L)
    rng = np.random.default_rng(0)
    plotted_any = False

    if unique_labels:
        for ci, lab in enumerate(unique_labels):
            mask = labels == lab
            if not mask.any():
                continue
            subset = te_lag_mean[mask]
            mean_vec = np.nanmean(subset, axis=0)
            # Bootstrap CI over samples.
            idxs = rng.integers(0, subset.shape[0], size=(n_bootstrap, subset.shape[0]))
            boot_means = np.nanmean(subset[idxs], axis=1)
            ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
            ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)

            color = palette[ci % len(palette)]
            name = class_names.get(lab, f"class {lab}")
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.18)
            ax.plot(x, mean_vec, color=color, linewidth=1.2, label=f"{name} (n={int(mask.sum())})")
            plotted_any = True
    else:
        # Label-free input (synthetic-TE runs etc.): emit a single
        # pooled trace so the PDF carries the lag profile instead of an
        # empty axes + "No artists with labels" legend warning.
        mean_vec = np.nanmean(te_lag_mean, axis=0)
        if N > 0:
            idxs = rng.integers(0, N, size=(n_bootstrap, N))
            boot_means = np.nanmean(te_lag_mean[idxs], axis=1)
            ci_lo = np.nanpercentile(boot_means, 2.5, axis=0)
            ci_hi = np.nanpercentile(boot_means, 97.5, axis=0)
            ax.fill_between(x, ci_lo, ci_hi, color=COLOR_BLUE, alpha=0.18)
        ax.plot(x, mean_vec, color=COLOR_BLUE, linewidth=1.2,
                label=f"pooled (n={int(N)})")
        plotted_any = True

    ax.set_xlabel("Lag k", fontsize=FONT_LABEL)
    ax.set_ylabel("mean te_lag_map", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    if plotted_any:
        ax.legend(loc="upper right", frameon=True)
    _style_axes(ax, grid="major", minor_ticks=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_attention_mass_by_lag(
    mass_df: pd.DataFrame,
    output_path: Path,
    *,
    fs: float = 4.0,
    decim: int = 16,
    title: str = "Attention Mass by Lag Bin",
) -> None:
    """Grouped bar chart: fraction of attention mass in coarse lag bins.

    Args:
        mass_df: Wide-format DataFrame with one row per sample, columns
            ``[label, mass_lag_0, mass_lag_1, ...]``. Per-sample row sums
            should be ~1.
        output_path: Destination PDF/PNG.
        fs: Sampling rate of the raw signal (4 Hz).
        decim: Decimation factor (16) used to convert lag index to seconds.
        title: Figure title.
    """
    lag_cols = [c for c in mass_df.columns if c.startswith("mass_lag_")]
    if not lag_cols:
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.text(0.5, 0.5, "No lag mass columns", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    # Convert lag index -> seconds using decimation and fs.
    lag_indices = np.array(sorted(int(c.split("_")[-1]) for c in lag_cols))
    seconds_per_step = decim / float(fs)
    lag_seconds = lag_indices * seconds_per_step

    # Coarse bins (seconds).
    bin_edges = [0.0, 10.0, 30.0, 60.0, 120.0, 360.0, 1e9]
    bin_labels = ["0-10s", "10-30s", "30-60s", "60-120s", "120-360s", ">=360s"]
    mass_matrix = mass_df[[f"mass_lag_{i}" for i in lag_indices]].to_numpy(dtype=float)

    bin_totals = np.zeros((mass_matrix.shape[0], len(bin_labels)), dtype=float)
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (lag_seconds >= lo) & (lag_seconds < hi)
        if in_bin.any():
            bin_totals[:, i] = mass_matrix[:, in_bin].sum(axis=1)

    labels_col = mass_df.get("label")
    if labels_col is None or labels_col.isna().all():
        class_groups: Dict[str, np.ndarray] = {"all": bin_totals.mean(axis=0)}
    else:
        class_groups = {}
        for lab in sorted(labels_col.dropna().unique()):
            mask = (labels_col == lab).to_numpy()
            if mask.any():
                class_groups[str(int(lab))] = bin_totals[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    n_groups = len(class_groups)
    n_bins = len(bin_labels)
    width = 0.8 / max(n_groups, 1)
    x = np.arange(n_bins)
    palette = [COLOR_BLUE, COLOR_VERMILLION, COLOR_GREEN, COLOR_ORANGE]
    for gi, (gname, means) in enumerate(class_groups.items()):
        ax.bar(
            x + gi * width - (n_groups - 1) * width / 2,
            means,
            width=width,
            color=palette[gi % len(palette)],
            alpha=0.85,
            label=gname,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.set_xlabel("Lag bin (seconds)", fontsize=FONT_LABEL)
    ax.set_ylabel("Mean attention mass", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="normal")
    ax.legend(loc="upper right", frameon=True)
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Calibration plots (G10)
# =============================================================================


def plot_reliability_curve(
    df: pd.DataFrame,
    output_path: Any,
    *,
    title: str = "Reliability (PIT) by horizon",
) -> None:
    r"""Plot empirical vs nominal PIT quantiles, one line per horizon step.

    A calibrated forecast traces the diagonal. Bowing *above* the diagonal at low quantiles
    and below it at high quantiles means the predictive distribution is too wide; the mirror
    shape means it is over-confident. A curve shifted bodily off the diagonal indicates a
    biased mean rather than a mis-scaled variance.

    Args:
        df: Long-format frame with columns ``h``, ``nominal``, ``empirical``.
        output_path: Destination path; the extension selects the format.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _style_axes(ax)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    horizons = sorted(df["h"].unique())
    cmap = plt.get_cmap("viridis")
    for i, h in enumerate(horizons):
        sub = df[df["h"] == h].sort_values("nominal")
        shade = cmap(i / max(len(horizons) - 1, 1))
        ax.plot(sub["nominal"], sub["empirical"], color=shade, linewidth=1.1, alpha=0.85)

    ax.plot([0, 1], [0, 1], color=COLOR_BLACK, linestyle="--", linewidth=1.2,
            label="perfect calibration")
    pooled = df.groupby("nominal", as_index=False)["empirical"].mean()
    ax.plot(pooled["nominal"], pooled["empirical"], color=COLOR_VERMILLION, linewidth=2.0,
            label="pooled over horizons")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max(horizons)))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("horizon step $h$", fontsize=FONT_LABEL * 0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal quantile", fontsize=FONT_LABEL * 0.7)
    ax.set_ylabel("empirical PIT fraction", fontsize=FONT_LABEL * 0.7)
    ax.set_title(title, fontsize=FONT_TITLE * 0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=FONT_LEGEND * 0.55)
    _style_axes(ax)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_vs_nominal(
    df: pd.DataFrame,
    output_path: Any,
    *,
    title: str = "Interval coverage vs nominal level",
) -> None:
    r"""Plot empirical central-interval coverage against its nominal level.

    Points below the diagonal mean under-coverage: the learned :math:`\sigma` is too small and
    the model is over-confident. This is the failure that mean-squared error cannot see, and
    the reason the calibration report exists.

    Args:
        df: Per-sample frame carrying ``coverage_50``, ``coverage_80``, ... columns.
        output_path: Destination path.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    columns = [c for c in df.columns if c.startswith("coverage_") and not c.endswith("_error")] \
        if df is not None and not df.empty else []
    if not columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _style_axes(ax)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    levels = sorted(int(c.rsplit("_", 1)[1]) / 100.0 for c in columns)
    means, los, his = [], [], []
    for level in levels:
        values = df[f"coverage_{int(round(level * 100))}"].to_numpy()
        means.append(float(np.mean(values)))
        los.append(float(np.quantile(values, 0.25)))
        his.append(float(np.quantile(values, 0.75)))

    ax.plot([0, 1], [0, 1], color=COLOR_BLACK, linestyle="--", linewidth=1.2,
            label="perfect calibration")
    ax.fill_between(levels, los, his, color=COLOR_BLUE, alpha=0.2, linewidth=0,
                    label="per-sample IQR")
    ax.plot(levels, means, color=COLOR_BLUE, marker="o", markersize=5, linewidth=1.6,
            label="mean coverage")

    for level, mean in zip(levels, means):
        ax.annotate(f"{mean - level:+.3f}", (level, mean), textcoords="offset points",
                    xytext=(6, -12), fontsize=FONT_TICK * 0.5, color=COLOR_GRAY)

    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("nominal level $p$", fontsize=FONT_LABEL * 0.7)
    ax.set_ylabel("empirical coverage", fontsize=FONT_LABEL * 0.7)
    ax.set_title(title, fontsize=FONT_TITLE * 0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=FONT_LEGEND * 0.55)
    _style_axes(ax)
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sharpness_by_horizon(
    df: pd.DataFrame,
    output_path: Any,
    *,
    constant_sigma: Optional[float] = None,
    title: str = "Predictive spread and score by horizon",
) -> None:
    r"""Plot the learned :math:`\sigma`, the NLL, and the CRPS against lead time.

    Sharpness alone is meaningless -- a collapsed variance is maximally sharp and useless -- so
    it is drawn beside the two proper scoring rules, and against the homoscedastic
    :math:`\hat{\sigma}` the model has to beat to justify its variance head.

    Args:
        df: Long-format frame with columns ``h``, ``sharpness``, ``nll``, ``crps``.
        output_path: Destination path.
        constant_sigma: The fitted global :math:`\hat{\sigma}` reference line, if available.
        title: Figure suptitle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    if df is None or df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            _style_axes(ax)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    grouped = df.groupby("h")
    steps = np.asarray(sorted(df["h"].unique()))
    panels = (
        ("sharpness", r"predictive $\sigma$", COLOR_BLUE),
        ("nll", "NLL [nats]", COLOR_VERMILLION),
        ("crps", "CRPS", COLOR_ORANGE),
    )
    for ax, (column, ylabel, colour) in zip(axes, panels):
        median = grouped[column].median().to_numpy()
        q25 = grouped[column].quantile(0.25).to_numpy()
        q75 = grouped[column].quantile(0.75).to_numpy()
        ax.fill_between(steps, q25, q75, color=colour, alpha=0.2, linewidth=0, label="IQR")
        ax.plot(steps, median, color=colour, linewidth=1.6, label="median")
        if column == "sharpness" and constant_sigma is not None and np.isfinite(constant_sigma):
            ax.axhline(constant_sigma, color=COLOR_BLACK, linestyle=":", linewidth=1.2,
                       label=r"constant $\hat{\sigma}$")
        ax.set_xlabel("horizon step $h$", fontsize=FONT_LABEL * 0.7)
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL * 0.7)
        ax.legend(loc="best", frameon=False, fontsize=FONT_LEGEND * 0.55)
        _style_axes(ax)

    fig.suptitle(title, fontsize=FONT_TITLE * 0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def _rank_1d(values: np.ndarray) -> np.ndarray:
    """Average-rank of the finite entries (NaN-preserving), scaled to ``[0, 1]``."""
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    mask = np.isfinite(x)
    if int(mask.sum()) < 2:
        return out
    sub = x[mask]
    order = np.argsort(sub, kind="mergesort")
    ranks = np.empty_like(sub)
    ranks[order] = np.arange(1, sub.size + 1, dtype=float)
    out[mask] = (ranks - 1.0) / (sub.size - 1.0)
    return out


def plot_cmi_comparison(
    comparison_table: pd.DataFrame,
    per_sample: pd.DataFrame,
    output_path: Any,
    *,
    title: str = r"$K_{\mathrm{raw}}$ vs neural CMI and empirical TE",
) -> None:
    r"""Plot the CMI corroboration: rank correlations and per-sample rank agreement (G11).

    :math:`K_{\mathrm{raw}}`, the neural CMI bounds, and empirical TE live on different scales
    (nats/step, nats, bits), so their magnitudes are not directly comparable -- the deliverable
    is **rank-level** agreement. The left panel shows Spearman :math:`\rho` of each independent
    estimate against :math:`K_{\mathrm{raw}}`; the right panel shows the per-sample rank scatter
    that those correlations summarise. Per-quantity means and patient-level bootstrap CIs are in
    ``comparison_table.csv`` (they are not plotted because their scales differ).

    Args:
        comparison_table: Frame with columns ``quantity`` and ``spearman_vs_kraw``.
        per_sample: Frame with ``k_raw`` and ``cmi_*`` (and optional ``ite_valid``) columns.
        output_path: Destination path; the extension selects the format.
        title: Figure suptitle.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    has_table = comparison_table is not None and not comparison_table.empty
    has_samples = per_sample is not None and not per_sample.empty
    if not has_table or not has_samples:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            _style_axes(ax)
        fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    _pretty = {"cmi_infonce": "InfoNCE", "cmi_mine": "MINE", "ite_valid": "empirical TE"}
    _colours = {"cmi_infonce": COLOR_BLUE, "cmi_mine": COLOR_VERMILLION, "ite_valid": COLOR_ORANGE}

    # Panel 1: rank correlation with K_raw for every non-K_raw quantity.
    ax = axes[0]
    rows = comparison_table[comparison_table["quantity"] != "k_raw"]
    names = [_pretty.get(q, q) for q in rows["quantity"]]
    rhos = rows["spearman_vs_kraw"].to_numpy(dtype=float)
    bar_colours = [_colours.get(q, COLOR_GRAY) for q in rows["quantity"]]
    ypos = np.arange(len(names))
    ax.barh(ypos, rhos, color=bar_colours, alpha=0.85, height=0.6)
    ax.axvline(0.0, color=COLOR_BLACK, linewidth=1.0)
    for yp, rho in zip(ypos, rhos):
        ax.annotate(f"{rho:+.2f}", (rho, yp), textcoords="offset points",
                    xytext=(4 if rho >= 0 else -22, -3), fontsize=FONT_TICK * 0.55,
                    color=COLOR_BLACK)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=FONT_LABEL * 0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(r"Spearman $\rho$ with $K_{\mathrm{raw}}$", fontsize=FONT_LABEL * 0.7)
    ax.set_title("rank corroboration", fontsize=FONT_TITLE * 0.55)
    _style_axes(ax)

    # Panel 2: per-sample rank scatter, K_raw vs each estimate.
    ax = axes[1]
    kr = _rank_1d(per_sample["k_raw"].to_numpy(dtype=float))
    ax.plot([0, 1], [0, 1], color=COLOR_BLACK, linestyle="--", linewidth=1.1)
    for col in ("cmi_infonce", "cmi_mine", "ite_valid"):
        if col not in per_sample.columns:
            continue
        yr = _rank_1d(per_sample[col].to_numpy(dtype=float))
        m = np.isfinite(kr) & np.isfinite(yr)
        if int(m.sum()) < 2:
            continue
        ax.scatter(kr[m], yr[m], s=14, alpha=0.5, color=_colours.get(col, COLOR_GRAY),
                   edgecolors="none", label=_pretty.get(col, col))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"normalised rank of $K_{\mathrm{raw}}$", fontsize=FONT_LABEL * 0.7)
    ax.set_ylabel("normalised rank of estimate", fontsize=FONT_LABEL * 0.7)
    ax.set_title("per-sample rank agreement", fontsize=FONT_TITLE * 0.55)
    ax.legend(loc="upper left", frameon=False, fontsize=FONT_LEGEND * 0.55)
    _style_axes(ax)

    fig.suptitle(title, fontsize=FONT_TITLE * 0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
