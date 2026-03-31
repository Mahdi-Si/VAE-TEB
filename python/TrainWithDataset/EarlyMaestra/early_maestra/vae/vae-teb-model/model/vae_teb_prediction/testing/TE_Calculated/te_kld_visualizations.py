"""Matplotlib visualisations for the TE vs. KLD comparison analysis.

All functions follow the publication-quality style defined in
``testing/visualizers.py`` and accept pre-computed DataFrames or dicts
as input (no model inference happens here).

Example:
    >>> from te_kld_visualizations import plot_pooled_scatter
    >>> plot_pooled_scatter(merged_df, Path("results"), pooled_stats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    COLOR_GREEN,
    COLOR_GRAY,
    COLOR_BLACK,
    COLOR_LIGHT_GRAY,
    SAVE_DPI,
    _style_axes,
)


# ---------------------------------------------------------------------------
# 1. Pooled scatter: ite_valid vs KLD
# ---------------------------------------------------------------------------


def plot_pooled_scatter(
    merged_df: pd.DataFrame,
    output_dir: Path,
    pooled_stats: Dict[str, Any],
) -> Path:
    """Scatter plot of empirical TE vs VAE-KLD for all matched epochs.

    Includes a linear regression line and annotated Pearson/Spearman
    correlation statistics.

    Args:
        merged_df: Merged TE-KLD DataFrame from ``merge_te_kld()``.
        output_dir: Directory to save the figure.
        pooled_stats: Output of ``compute_pooled_correlation()``.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = merged_df["ite_valid"].values
    y = merged_df["kld"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(4.5, 4))

    ax.scatter(x, y, s=6, alpha=0.25, color=COLOR_BLUE, edgecolors="none",
               rasterized=True)

    # Regression line
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLOR_VERMILLION,
                linewidth=1.2, label="Linear fit")

    ax.set_xlabel("Empirical TE (ite_valid)")
    ax.set_ylabel("VAE KLD")
    ax.set_title("Empirical TE vs. VAE KLD (all epochs)")

    # Annotation
    r = pooled_stats.get("pearson_r", np.nan)
    rho = pooled_stats.get("spearman_rho", np.nan)
    n = pooled_stats.get("n", 0)
    r_p = pooled_stats.get("pearson_p", np.nan)
    rho_p = pooled_stats.get("spearman_p", np.nan)
    ann = (
        f"n = {n}\n"
        f"Pearson r = {r:.3f} (p = {r_p:.2e})\n"
        f"Spearman ρ = {rho:.3f} (p = {rho_p:.2e})"
    )
    ax.text(
        0.03, 0.97, ann, transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_GRAY, alpha=0.9),
    )

    _style_axes(ax)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()

    out_path = output_dir / "pooled_scatter_te_vs_kld.png"
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2. Per-GUID correlation histogram
# ---------------------------------------------------------------------------


def plot_per_guid_correlation_histogram(
    per_guid_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Histogram of per-GUID Pearson r and Spearman rho values.

    Args:
        per_guid_df: Output of ``compute_per_guid_correlations()``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for ax, col, label, color in [
        (axes[0], "pearson_r", "Pearson r", COLOR_BLUE),
        (axes[1], "spearman_rho", "Spearman ρ", COLOR_ORANGE),
    ]:
        vals = per_guid_df[col].dropna().values
        if len(vals) == 0:
            ax.set_title(f"{label} (no data)")
            continue

        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.3)
        ax.axvline(0, color=COLOR_BLACK, linewidth=0.8, linestyle="--",
                   alpha=0.5)
        ax.axvline(np.mean(vals), color=COLOR_VERMILLION, linewidth=1.0,
                   linestyle="-", label=f"Mean = {np.mean(vals):.3f}")
        ax.axvline(np.median(vals), color=COLOR_GREEN, linewidth=1.0,
                   linestyle="--", label=f"Median = {np.median(vals):.3f}")

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Per-GUID {label} distribution (n={len(vals)})")
        ax.legend(fontsize=6)
        _style_axes(ax)

    fig.tight_layout()
    out_path = output_dir / "per_guid_correlation_histogram.png"
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3. Sample dual-axis trajectory plots
# ---------------------------------------------------------------------------


def plot_sample_dual_axis_trajectories(
    merged_df: pd.DataFrame,
    output_dir: Path,
    guids_per_page: int = 8,
    max_pages: int = 10,
) -> list[Path]:
    """Dual-axis trajectory plots for many GUIDs across multiple figures.

    Ranks GUIDs by number of matched epochs and produces up to *max_pages*
    figures, each showing *guids_per_page* GUIDs.  Every subplot uses the
    full time range available for that GUID (no fixed hour window), with
    the x-axis running from the earliest data to delivery (0 h).

    Args:
        merged_df: Merged TE-KLD DataFrame.
        output_dir: Directory to save figures.
        guids_per_page: Number of GUIDs per figure (laid out 2 columns).
        max_pages: Maximum number of figures to generate.

    Returns:
        List of paths to the saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rank GUIDs by number of matched epochs (descending)
    guid_counts = merged_df.groupby("guid").size().sort_values(ascending=False)
    all_guids = guid_counts.index.tolist()

    total_guids_needed = guids_per_page * max_pages
    selected_guids = all_guids[:total_guids_needed]

    # Split into pages
    pages: list[list[str]] = [
        selected_guids[i : i + guids_per_page]
        for i in range(0, len(selected_guids), guids_per_page)
    ]
    pages = pages[:max_pages]

    saved_paths: list[Path] = []

    for page_idx, page_guids in enumerate(pages):
        n_guids = len(page_guids)
        nrows = (n_guids + 1) // 2
        ncols = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.0 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for i, guid in enumerate(page_guids):
            ax = axes_flat[i]
            sub = merged_df[merged_df["guid"] == guid].sort_values(
                "domain_start_rounded"
            )
            hours = -sub["domain_start_rounded"].values / 3600.0

            # TE on left axis
            ax.plot(
                hours, sub["ite_valid"].values,
                color=COLOR_BLUE, linewidth=1.2,
                marker=".", markersize=4, label="TE",
            )
            ax.set_ylabel("TE (ite_valid)", color=COLOR_BLUE, fontsize=7)
            ax.tick_params(axis="y", labelcolor=COLOR_BLUE, labelsize=6)

            # KLD on right axis
            ax2 = ax.twinx()
            ax2.plot(
                hours, sub["kld"].values,
                color=COLOR_VERMILLION, linewidth=1.4,
                marker=".", markersize=4, label="KLD",
            )
            ax2.set_ylabel("KLD", color=COLOR_VERMILLION, fontsize=7)
            ax2.tick_params(axis="y", labelcolor=COLOR_VERMILLION, labelsize=6)

            ax.set_xlabel("Hours before delivery", fontsize=7)
            h_min, h_max = hours.min(), hours.max()
            ax.set_xlim(h_max + 0.2, max(h_min - 0.2, 0))
            ax.set_title(
                f"GUID ...{guid[-6:]}  (n={len(sub)}, "
                f"{h_max:.1f}\u2013{h_min:.1f} h)",
                fontsize=7,
            )
            _style_axes(ax, grid="major")

        # Hide unused axes
        for j in range(len(page_guids), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout()
        out_path = output_dir / f"dual_axis_trajectories_{page_idx + 1:02d}.png"
        fig.savefig(out_path, dpi=SAVE_DPI)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


# ---------------------------------------------------------------------------
# 4. Cross-GUID scatter: mean TE vs mean KLD
# ---------------------------------------------------------------------------


def plot_cross_guid_scatter(
    per_guid_df: pd.DataFrame,
    output_dir: Path,
    cross_guid_stats: Dict[str, Any],
) -> Path:
    """Scatter plot of per-GUID mean TE vs per-GUID mean KLD.

    Each point represents one patient.

    Args:
        per_guid_df: Output of ``compute_per_guid_correlations()``.
        output_dir: Directory to save the figure.
        cross_guid_stats: Output of ``compute_cross_guid_correlation()``.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = per_guid_df["mean_ite_valid"].values
    y = per_guid_df["mean_kld"].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(x, y, s=20, alpha=0.6, color=COLOR_BLUE, edgecolors="white",
               linewidth=0.3)

    # Regression line
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=COLOR_VERMILLION,
                linewidth=1.2, label="Linear fit")

    ax.set_xlabel("Mean empirical TE per GUID")
    ax.set_ylabel("Mean VAE KLD per GUID")
    ax.set_title("Cross-GUID: mean TE vs. mean KLD")

    r = cross_guid_stats.get("pearson_r", np.nan)
    rho = cross_guid_stats.get("spearman_rho", np.nan)
    n = cross_guid_stats.get("n_guids", 0)
    ann = (
        f"n = {n} GUIDs\n"
        f"Pearson r = {r:.3f}\n"
        f"Spearman ρ = {rho:.3f}"
    )
    ax.text(
        0.03, 0.97, ann, transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=COLOR_GRAY, alpha=0.9),
    )

    _style_axes(ax)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()

    out_path = output_dir / "cross_guid_scatter.png"
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5. Per-GUID correlation bar chart
# ---------------------------------------------------------------------------


def plot_per_guid_correlation_bar(
    per_guid_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Sorted bar chart of per-GUID Spearman rho, coloured by significance.

    Args:
        per_guid_df: Output of ``compute_per_guid_correlations()``.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = per_guid_df.dropna(subset=["spearman_rho"]).copy()
    if len(df) == 0:
        return output_dir / "per_guid_correlation_bar.png"

    df = df.sort_values("spearman_rho")
    colors = [
        COLOR_VERMILLION if p < 0.05 else COLOR_LIGHT_GRAY
        for p in df["spearman_p"]
    ]

    fig, ax = plt.subplots(figsize=(6, max(3, len(df) * 0.15)))
    ax.barh(range(len(df)), df["spearman_rho"].values, color=colors,
            edgecolor="none", height=0.7)
    ax.axvline(0, color=COLOR_BLACK, linewidth=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"...{g[-6:]}" for g in df["guid"].values], fontsize=5
    )
    ax.set_xlabel("Spearman ρ (TE vs KLD)")
    ax.set_title(
        f"Per-GUID Spearman ρ (red = p < 0.05, n={len(df)})"
    )
    _style_axes(ax, grid="major")

    fig.tight_layout()
    out_path = output_dir / "per_guid_correlation_bar.png"
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6. Bootstrap CI plot
# ---------------------------------------------------------------------------


def plot_bootstrap_ci(
    pooled_stats: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Bootstrap distribution of pooled Pearson r and Spearman rho with CIs.

    Args:
        pooled_stats: Output of ``compute_pooled_correlation()`` (must
            include ``bootstrap_pearson_samples`` and
            ``bootstrap_spearman_samples`` arrays).
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    for ax, key, obs_key, ci_lo_key, ci_hi_key, label, color in [
        (
            axes[0], "bootstrap_pearson_samples", "pearson_r",
            "pearson_ci_lo", "pearson_ci_hi", "Pearson r", COLOR_BLUE,
        ),
        (
            axes[1], "bootstrap_spearman_samples", "spearman_rho",
            "spearman_ci_lo", "spearman_ci_hi", "Spearman ρ", COLOR_ORANGE,
        ),
    ]:
        samples = pooled_stats.get(key)
        if samples is None:
            ax.set_title(f"{label} (no bootstrap data)")
            continue

        samples = samples[np.isfinite(samples)]
        obs = pooled_stats.get(obs_key, np.nan)
        ci_lo = pooled_stats.get(ci_lo_key, np.nan)
        ci_hi = pooled_stats.get(ci_hi_key, np.nan)

        ax.hist(samples, bins=60, color=color, alpha=0.6, edgecolor="none")
        ax.axvline(obs, color=COLOR_BLACK, linewidth=1.2,
                   label=f"Observed = {obs:.3f}")
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=COLOR_VERMILLION,
                   label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.axvline(0, color=COLOR_GRAY, linewidth=0.6, linestyle="--",
                   alpha=0.5)

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Bootstrap {label}")
        ax.legend(fontsize=6)
        _style_axes(ax)

    fig.tight_layout()
    out_path = output_dir / "bootstrap_ci.png"
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    return out_path
