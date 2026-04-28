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
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
                f"GUID {_format_guid(guid)}  "
                f"(n={len(sub)}, {h_max:.1f}\u2013{h_min:.1f} h)",
                fontsize=5.5, family="monospace", loc="left",
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

    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.18)))
    ax.barh(range(len(df)), df["spearman_rho"].values, color=colors,
            edgecolor="none", height=0.7)
    ax.axvline(0, color=COLOR_BLACK, linewidth=0.6)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [_format_guid(g) for g in df["guid"].values],
        fontsize=4.5, family="monospace",
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


# =============================================================================
# Extended plots — ported from the old pipeline + new time-alignment diagnostic
# =============================================================================


from typing import List, Optional, Sequence  # noqa: E402


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value, or empty string."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_guid(guid: Any, *, max_chars: Optional[int] = None) -> str:
    """Render a GUID for display.

    Defaults to the full GUID string. Existing call-sites that previously
    truncated to a leading-hex prefix can pass ``max_chars`` to keep the
    same compact behaviour. The pipeline standardises on full GUIDs in
    every figure so per-row labels can always be traced back to a
    recording in the metrics CSV; ``max_chars`` is reserved for the rare
    legend / annotation slot that does not have room for 32 chars.
    """
    s = str(guid)
    if max_chars is not None and max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "..."
    return s


# Backwards-compat alias so any external import keeps working. New code
# should call ``_format_guid`` directly. The default behaviour is now to
# return the full GUID, matching the project-wide policy.
def _short_guid(guid: Any, n: Optional[int] = None) -> str:  # noqa: D401
    """Backwards-compat shim around :func:`_format_guid`."""
    return _format_guid(guid, max_chars=n)


# ---------------------------------------------------------------------------
# Data quality summary
# ---------------------------------------------------------------------------


def plot_data_quality_summary(
    quality_report: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Three-panel data-quality diagnostic: GUID overlap, stats, per-GUID match.

    Args:
        quality_report: Output of
            :func:`~te_data_loader.compute_data_quality_report`.
        output_dir: Directory for the saved figure.

    Returns:
        Path to the saved PDF.
    """
    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.0, 1.4], wspace=0.35)

    ax_overlap = fig.add_subplot(gs[0, 0])
    te_only = quality_report.get("te_only_guids", 0)
    kld_only = quality_report.get("kld_only_guids", 0)
    common = quality_report.get("common_guids", 0)
    ax_overlap.barh(
        ["TE only", "Common", "KLD only"],
        [te_only, common, kld_only],
        color=[COLOR_VERMILLION, COLOR_GREEN, COLOR_BLUE],
    )
    ax_overlap.set_title("GUID overlap", fontsize=9)
    ax_overlap.set_xlabel("Number of GUIDs", fontsize=8)
    for spine in ax_overlap.spines.values():
        spine.set_linewidth(0.4)
    _style_axes(ax_overlap)

    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    gap_stats = quality_report.get("time_gap_stats", {})
    lines = [
        f"TE epochs: {quality_report.get('te_total_epochs', '?')}",
        f"KLD segments: {quality_report.get('kld_total_segments', '?')}",
        f"Matched pairs: {quality_report.get('matched_pairs', '?')}",
        f"Matched GUIDs: {quality_report.get('matched_guids', '?')}",
        "",
        f"Max gap tolerance: ±{quality_report.get('max_gap_minutes', 0):.1f} min",
    ]
    if gap_stats:
        lines += [
            f"Mean gap: {gap_stats.get('mean', 0):.1f} s",
            f"Median gap: {gap_stats.get('median', 0):.1f} s",
            f"Max gap: {gap_stats.get('max', 0):.1f} s",
        ]
    ax_text.text(
        0.0, 1.0, "\n".join(lines),
        fontsize=9, va="top", ha="left", family="monospace",
    )

    ax_bar = fig.add_subplot(gs[0, 2])
    per_guid_match = quality_report.get("per_guid_matching", []) or []
    if per_guid_match:
        df = pd.DataFrame(per_guid_match).sort_values(
            "n_matched", ascending=False
        )
        labels = [_format_guid(g) for g in df["guid"].astype(str).tolist()]
        ax_bar.bar(
            range(len(df)), df["n_matched"],
            color=COLOR_BLUE, alpha=0.85,
        )
        ax_bar.set_xticks(range(len(df)))
        ax_bar.set_xticklabels(
            labels, rotation=75, ha="right", fontsize=4.5,
            family="monospace",
        )
        ax_bar.set_ylabel("n matched", fontsize=8)
        for i, (_, row) in enumerate(df.iterrows()):
            ax_bar.text(
                i, row["n_matched"], f"{row['mean_gap']:.0f}s",
                ha="center", va="bottom", fontsize=5, color=COLOR_GRAY,
            )
    else:
        ax_bar.text(
            0.5, 0.5, "no matches", ha="center", va="center",
            transform=ax_bar.transAxes,
        )
    ax_bar.set_title("Matches per GUID (annotated: mean time gap)", fontsize=9)
    _style_axes(ax_bar)

    fig.suptitle("Data quality summary", fontsize=11, y=1.02)
    out_path = output_dir / "data_quality_summary.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    output_dir: Path,
    *,
    title: str = "Spearman correlation: KLD measures vs TE",
    filename: str = "correlation_heatmap.pdf",
) -> Path:
    """Diverging-colour correlation heatmap with significance stars."""
    if len(corr_df) == 0 or len(corr_df.columns) == 0:
        return output_dir / filename

    fig, ax = plt.subplots(
        figsize=(max(3.2, 0.9 * len(corr_df.columns) + 2.0),
                 max(2.4, 0.5 * len(corr_df) + 1.5)),
    )
    vmax = float(np.nanmax(np.abs(corr_df.values))) if corr_df.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        corr_df.values, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index, fontsize=8)

    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            r = corr_df.values[i, j]
            p = pval_df.values[i, j]
            if not np.isfinite(r):
                continue
            ax.text(
                j, i, f"{r:.2f}{_sig_stars(float(p))}",
                ha="center", va="center",
                fontsize=7,
                color="white" if abs(r) > vmax * 0.6 else COLOR_BLACK,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("ρ", fontsize=8)
    ax.set_title(title, fontsize=10)
    _style_axes(ax)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-dimension KL heatmap
# ---------------------------------------------------------------------------


def plot_per_dimension_kl_heatmap(
    per_dim_df: pd.DataFrame,
    output_dir: Path,
) -> Optional[Path]:
    """Row-per-dim heatmap of ρ(kld_dim_d, ite_valid) with permutation stars."""
    if per_dim_df is None or len(per_dim_df) == 0:
        return None
    df = per_dim_df.sort_values("dimension").reset_index(drop=True)
    rho = df["spearman_rho"].values.astype(float).reshape(-1, 1)
    pvals = df["permutation_p"].values.astype(float).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(2.6, 0.28 * len(df) + 1.4))
    vmax = float(np.nanmax(np.abs(rho))) if rho.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    im = ax.imshow(
        rho, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"dim {int(d)}" for d in df["dimension"]], fontsize=7)
    ax.set_xticks([0])
    ax.set_xticklabels(["ite_valid"], fontsize=8)
    for i in range(len(df)):
        r = rho[i, 0]
        p = pvals[i, 0]
        if not np.isfinite(r):
            continue
        ax.text(
            0, i, f"{r:.2f}{_sig_stars(float(p))}",
            ha="center", va="center", fontsize=7,
            color="white" if abs(r) > vmax * 0.6 else COLOR_BLACK,
        )
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("ρ", fontsize=8)
    ax.set_title("Per-dim KL vs empirical TE", fontsize=10)
    _style_axes(ax)
    out_path = output_dir / "per_dimension_kl_heatmap.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Scatter matrix
# ---------------------------------------------------------------------------


def plot_scatter_matrix(
    merged_df: pd.DataFrame,
    kld_cols: Sequence[str],
    te_cols: Sequence[str],
    output_dir: Path,
    *,
    per_pair_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """Pairwise scatter for each (kld_col, te_col) with ρ + permutation p.

    Args:
        merged_df: Merged DataFrame.
        kld_cols: KLD-side columns (rows of the grid).
        te_cols: TE-side columns (columns of the grid).
        output_dir: Output directory.
        per_pair_stats: Optional mapping ``"{kld_col}__{te_col}" -> dict``
            with ``spearman_rho`` and ``permutation_p`` to annotate each
            panel. When missing, falls back to scipy spearmanr.
    """
    from scipy import stats as sp_stats  # local import to keep module light
    rows = len(kld_cols)
    cols = len(te_cols)
    if rows == 0 or cols == 0:
        return output_dir / "scatter_matrix.pdf"

    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False,
    )
    guids = sorted(merged_df["guid"].unique())
    palette = [COLOR_BLUE, COLOR_ORANGE, COLOR_VERMILLION, COLOR_GREEN]
    guid_color = {g: palette[i % len(palette)] for i, g in enumerate(guids)}

    for i, kc in enumerate(kld_cols):
        for j, tc in enumerate(te_cols):
            ax = axes[i][j]
            y = merged_df[kc].values.astype(float)
            x = merged_df[tc].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[mask], y[mask]
            color_vec = [
                guid_color.get(str(g), COLOR_GRAY)
                for g in merged_df["guid"].values[mask]
            ]
            ax.scatter(xv, yv, s=8, c=color_vec, alpha=0.7, edgecolors="none")
            if len(xv) >= 3 and np.std(xv) > 0 and np.std(yv) > 0:
                slope, intercept = np.polyfit(xv, yv, 1)
                xs = np.linspace(xv.min(), xv.max(), 50)
                ax.plot(xs, slope * xs + intercept, color=COLOR_GRAY,
                        linewidth=0.8, linestyle="--")

            key = f"{kc}__{tc}"
            if per_pair_stats and key in per_pair_stats:
                rho = per_pair_stats[key].get("spearman_rho", float("nan"))
                p = per_pair_stats[key].get("permutation_p", float("nan"))
            else:
                rho, p = (
                    (sp_stats.spearmanr(xv, yv).statistic,
                     sp_stats.spearmanr(xv, yv).pvalue)
                    if len(xv) >= 3 else (float("nan"), float("nan"))
                )
            ann = f"ρ={float(rho):.2f} p={float(p):.3f}"
            ax.text(
                0.98, 0.97, ann,
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.75,
                          edgecolor=COLOR_GRAY, linewidth=0.3),
            )
            if i == rows - 1:
                ax.set_xlabel(tc, fontsize=8)
            if j == 0:
                ax.set_ylabel(kc, fontsize=8)
            _style_axes(ax)

    fig.suptitle("Scatter matrix: KLD measures vs empirical TE", fontsize=10,
                 y=1.02)
    fig.tight_layout()
    out_path = output_dir / "scatter_matrix.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Trend agreement
# ---------------------------------------------------------------------------


def plot_trend_agreement(
    trend_result: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Per-GUID agreement bars + overall rate + binomial p annotation."""
    per_guid = trend_result.get("per_guid_agreement", {}) or {}
    n_guids = len(per_guid)
    # Switch to a horizontal bar so full GUIDs fit on the y-axis.
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.18 * n_guids + 2.5)))
    if per_guid:
        labels = list(per_guid.keys())
        formatted = [_format_guid(g) for g in labels]
        rates = [per_guid[g]["agreement_rate"] for g in labels]
        n_trans = [per_guid[g]["n_transitions"] for g in labels]
        colors = [
            COLOR_GREEN if r >= 0.5 else COLOR_VERMILLION for r in rates
        ]
        ax.barh(range(len(labels)), rates, color=colors, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(formatted, fontsize=4.5, family="monospace")
        for i, (r, n) in enumerate(zip(rates, n_trans)):
            ax.text(r + 0.01, i, f"n={n}", ha="left", va="center",
                    fontsize=6, color=COLOR_GRAY)
    ax.axvline(0.5, color=COLOR_GRAY, linewidth=0.6, linestyle=":",
               label="chance (0.5)")
    overall = trend_result.get("sign_agreement_rate", float("nan"))
    binom_p = trend_result.get("binomial_p", float("nan"))
    if np.isfinite(overall):
        ax.axvline(overall, color=COLOR_BLUE, linewidth=1.2,
                   label=f"overall={overall:.2%}")
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Sign agreement rate", fontsize=8)
    ax.set_title(
        f"Trend agreement — binomial p = {binom_p:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="upper right")
    _style_axes(ax)
    out_path = output_dir / "trend_agreement.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Leave-one-GUID-out
# ---------------------------------------------------------------------------


def plot_leave_one_out(
    loo_result: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Bar chart: full ρ vs each leave-one-GUID-out ρ."""
    leave_out = loo_result.get("leave_out_correlations", {}) or {}
    full_r = loo_result.get("full_correlation", float("nan"))
    most = loo_result.get("most_influential_guid")

    labels = ["full"] + [_format_guid(g) for g in leave_out.keys()]
    values = [full_r] + list(leave_out.values())
    bar_colors = [COLOR_BLACK] + [
        COLOR_VERMILLION if g == most else COLOR_BLUE
        for g in leave_out.keys()
    ]

    # Horizontal bars so full GUIDs render comfortably as y-tick labels.
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.18 * len(labels) + 2.0)))
    ax.barh(range(len(labels)), values, color=bar_colors, alpha=0.85)
    ax.axvline(0.0, color=COLOR_GRAY, linewidth=0.6, linestyle=":")
    if np.isfinite(full_r):
        ax.axvline(full_r, color=COLOR_GRAY, linewidth=0.6, linestyle="--",
                   label="full ρ")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=4.5, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel(f"{loo_result.get('method', 'spearman')} ρ", fontsize=8)
    title_bits = ["Leave-one-GUID-out sensitivity"]
    if most:
        title_bits.append(f"(most influential: {_format_guid(str(most))})")
    ax.set_title(" ".join(title_bits), fontsize=10)
    if any(np.isfinite(v) for v in values):
        ax.legend(fontsize=7, loc="best")
    _style_axes(ax)
    out_path = output_dir / "leave_one_out.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Cluster-aware bootstrap distribution
# ---------------------------------------------------------------------------


def plot_bootstrap_distribution(
    cluster_result: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Histogram of cluster-aware block-bootstrap samples for Pearson + Spearman."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, label in [
        (axes[0], "pearson", "Pearson r"),
        (axes[1], "spearman", "Spearman ρ"),
    ]:
        sub = cluster_result.get(key, {}) or {}
        samples = sub.get("bootstrap_samples")
        if samples is None or not hasattr(samples, "__len__") or len(samples) == 0:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                    transform=ax.transAxes, color=COLOR_GRAY)
            ax.set_title(label)
            continue
        ax.hist(samples, bins=40, color=COLOR_BLUE, alpha=0.7,
                edgecolor=COLOR_GRAY, linewidth=0.3)
        observed = sub.get("observed", float("nan"))
        ci_lo = sub.get("ci_lo", float("nan"))
        ci_hi = sub.get("ci_hi", float("nan"))
        if np.isfinite(observed):
            ax.axvline(observed, color=COLOR_VERMILLION, linewidth=1.2,
                       label=f"observed={observed:.3f}")
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.axvspan(ci_lo, ci_hi, color=COLOR_GREEN, alpha=0.18,
                       label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        ax.axvline(0.0, color=COLOR_GRAY, linewidth=0.6, linestyle=":")
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title(f"Cluster bootstrap ({label})", fontsize=9)
        _style_axes(ax)
    fig.suptitle(
        f"Block bootstrap (n_guids={cluster_result.get('n_guids', '?')})",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "bootstrap_distribution.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------


def plot_permutation_null(
    perm_result: Dict[str, Any],
    output_dir: Path,
    *,
    title: str = "Permutation null distribution",
    filename: str = "permutation_null.pdf",
) -> Optional[Path]:
    """Histogram of permutation null with observed value marked."""
    null_dist = perm_result.get("null_distribution")
    if null_dist is None or not hasattr(null_dist, "__len__") or len(null_dist) == 0:
        return None
    observed = perm_result.get("observed", float("nan"))
    p = perm_result.get("p_value", float("nan"))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(null_dist, bins=50, color=COLOR_GRAY, alpha=0.6,
            edgecolor=COLOR_BLACK, linewidth=0.3)
    if np.isfinite(observed):
        ax.axvline(observed, color=COLOR_VERMILLION, linewidth=1.3,
                   label=f"observed={observed:.3f}")
        # Shade tails beyond |observed|
        ax.axvspan(abs(observed), np.nanmax(null_dist), color=COLOR_VERMILLION,
                   alpha=0.08)
        ax.axvspan(np.nanmin(null_dist), -abs(observed), color=COLOR_VERMILLION,
                   alpha=0.08)
    ax.axvline(0.0, color=COLOR_GRAY, linewidth=0.5, linestyle=":")
    ax.set_xlabel("null correlation", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(f"{title}  (p = {p:.4f})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    _style_axes(ax)
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Mutual information vs correlation
# ---------------------------------------------------------------------------


def plot_mutual_information_comparison(
    mi_value: float,
    pooled_stats: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Grouped bar chart: |Spearman ρ| and |Pearson r| vs MI (nats)."""
    spearman_rho = abs(float(pooled_stats.get("spearman_rho", float("nan"))))
    pearson_r = abs(float(pooled_stats.get("pearson_r", float("nan"))))
    labels = ["|Pearson r|", "|Spearman ρ|", "MI (nats)"]
    values = [pearson_r, spearman_rho, float(mi_value)]
    colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN]

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Strength", fontsize=8)
    ax.set_title("Linear vs non-linear association", fontsize=10)
    _style_axes(ax)
    out_path = output_dir / "mutual_information_comparison.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# DTW trajectory plot with ±band envelope + warping-path connectors
# ---------------------------------------------------------------------------


def plot_dtw_trajectories(
    dtw_results: Dict[str, Any],
    output_dir: Path,
) -> Optional[Path]:
    """Per-GUID z-scored DTW alignment with ±band envelope + warp connectors.

    Args:
        dtw_results: Output of :func:`~te_dtw.dtw_align_per_guid`.
        output_dir: Output directory.

    Returns:
        Path to the saved PDF, or ``None`` when ``dtw_results`` is
        unavailable / empty.
    """
    if not dtw_results or not dtw_results.get("available", True):
        return None
    per_guid = dtw_results.get("per_guid", {}) or {}
    guids = [g for g in per_guid if per_guid[g].get("n_te", 0) >= 3
             and per_guid[g].get("n_kld", 0) >= 3]
    if not guids:
        return None

    band_seconds = float(dtw_results.get("band_seconds", 300.0))
    backend = dtw_results.get("method", dtw_results.get("backend", "dtw"))

    fig, axes = plt.subplots(
        len(guids), 1, figsize=(10.5, 2.5 * len(guids)), squeeze=False,
    )
    for row, guid in enumerate(guids):
        ax = axes[row][0]
        entry = per_guid[guid]
        t_times_h = np.asarray(entry["te_times"]) / 3600.0
        k_times_h = np.asarray(entry["kld_times"]) / 3600.0
        t_z = np.asarray(entry["te_z"])
        k_z = np.asarray(entry["kld_z"])

        # ±band envelope around every empirical timestamp (in hours).
        band_h = band_seconds / 3600.0
        for th in t_times_h:
            ax.axvspan(th - band_h, th + band_h, color=COLOR_GREEN, alpha=0.06)

        # Warping-path connectors — sub-sample to at most ~200 lines so
        # dense alignments don't turn the plot into a solid block.
        path_te = np.asarray(entry.get("path_te", []), dtype=int)
        path_kld = np.asarray(entry.get("path_kld", []), dtype=int)
        if len(path_te) > 0:
            step = max(1, len(path_te) // 200)
            for i, j in zip(path_te[::step], path_kld[::step]):
                if i < len(t_times_h) and j < len(k_times_h):
                    ax.plot(
                        [t_times_h[i], k_times_h[j]],
                        [t_z[i], k_z[j]],
                        color=COLOR_GRAY, linewidth=0.35, alpha=0.5,
                    )

        ax.plot(t_times_h, t_z, color=COLOR_VERMILLION, marker="s",
                markersize=3, linewidth=1.0, label="TE (z)")
        ax.plot(k_times_h, k_z, color=COLOR_BLUE, marker="o",
                markersize=3, linewidth=1.0, label="KLD (z)")

        dist = entry.get("distance", float("nan"))
        norm = entry.get("normalized", float("nan"))
        band_steps = entry.get("band_steps", 0)
        method = entry.get("method", backend)
        ann = (
            f"{method} dist={dist:.2f} "
            f"(norm={norm:.3f})\n"
            f"band={band_seconds:.0f}s ({band_steps} steps)\n"
            f"n_te={entry.get('n_te', 0)}, n_kld={entry.get('n_kld', 0)}"
        )
        ax.text(
            0.99, 0.97, ann, transform=ax.transAxes, fontsize=6,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor=COLOR_GRAY,
                      linewidth=0.3),
        )

        # Full GUID stays on the title (monospace) so DTW pages remain
        # traceable even after re-ordering by domain_start.
        ax.set_title(
            f"GUID {_format_guid(guid)}",
            fontsize=6.5, family="monospace", loc="left",
        )
        ax.set_ylabel("z-score", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        _style_axes(ax)
        if row == len(guids) - 1:
            ax.set_xlabel("Time (hours, negative = pre-delivery)", fontsize=8)

    fig.suptitle(
        f"DTW alignment: TE (ite_valid) vs KLD — ±{band_seconds/60.0:.1f} min band",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_path = output_dir / "dtw_trajectories.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Time-alignment diagnostic (new)
# ---------------------------------------------------------------------------


def plot_time_alignment_diagnostic(
    merged_df: pd.DataFrame,
    output_dir: Path,
    *,
    max_gap_seconds: float = 300.0,
) -> Optional[Path]:
    """Scatter of TE epoch vs KLD epoch for every matched pair.

    - Colored by time gap (seconds).
    - y=x reference line with ±max_gap diagonal band.
    - Side panel: histogram of time gaps with cutoff line.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld` (fuzzy
            mode; carries ``time_gap_seconds``, ``domain_start``,
            ``epoch`` columns).
        output_dir: Output directory.
        max_gap_seconds: Tolerance used when matching (for display only).

    Returns:
        Path to saved PDF. ``None`` when no matches or required columns
        are missing.
    """
    if (
        len(merged_df) == 0
        or "domain_start" not in merged_df.columns
        or "epoch" not in merged_df.columns
        or "time_gap_seconds" not in merged_df.columns
    ):
        return None

    te_hours = merged_df["domain_start"].values / 3600.0
    kld_hours = merged_df["epoch"].values / 3600.0
    gaps = merged_df["time_gap_seconds"].values.astype(float)

    fig = plt.figure(figsize=(9.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(
        kld_hours, te_hours, c=gaps, cmap="viridis", s=14,
        vmin=0.0, vmax=max_gap_seconds, alpha=0.85, edgecolors="none",
    )
    lo = min(float(np.nanmin(kld_hours)), float(np.nanmin(te_hours)))
    hi = max(float(np.nanmax(kld_hours)), float(np.nanmax(te_hours)))
    xs = np.linspace(lo, hi, 50)
    band_h = max_gap_seconds / 3600.0
    ax.fill_between(xs, xs - band_h, xs + band_h, color=COLOR_GREEN,
                    alpha=0.12, label=f"±{max_gap_seconds/60.0:.1f} min")
    ax.plot(xs, xs, color=COLOR_GRAY, linewidth=0.8, linestyle="--",
            label="y = x")
    ax.set_xlabel("KLD epoch (hours)", fontsize=8)
    ax.set_ylabel("TE domain_start (hours)", fontsize=8)
    ax.set_title("Matched-pair time alignment", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("|time gap| (s)", fontsize=8)
    _style_axes(ax)

    ax_h = fig.add_subplot(gs[0, 1])
    ax_h.hist(gaps, bins=30, color=COLOR_BLUE, alpha=0.75,
              edgecolor=COLOR_GRAY, linewidth=0.3)
    ax_h.axvline(max_gap_seconds, color=COLOR_VERMILLION, linewidth=1.2,
                 linestyle="--", label=f"cutoff {max_gap_seconds:.0f}s")
    ax_h.set_xlabel("time gap (s)", fontsize=8)
    ax_h.set_ylabel("count", fontsize=8)
    ax_h.set_title("Gap distribution", fontsize=10)
    ax_h.legend(fontsize=7, loc="upper right")
    _style_axes(ax_h)

    out_path = output_dir / "time_alignment_diagnostic.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-GUID trajectory overlays (z-scored)
# ---------------------------------------------------------------------------


def plot_per_guid_trajectory_overlays(
    merged_df: pd.DataFrame,
    output_dir: Path,
    *,
    te_col: str = "ite_valid",
    kld_col: str = "kld",
    top_n: int = 12,
) -> Optional[Path]:
    """Grid of z-scored TE+KLD trajectories for the top-N GUIDs by match count."""
    if len(merged_df) == 0:
        return None
    counts = merged_df.groupby("guid").size().sort_values(ascending=False)
    guids = counts.head(top_n).index.tolist()
    if not guids:
        return None

    ncols = 2
    nrows = int(np.ceil(len(guids) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 2.4 * nrows), squeeze=False,
    )
    time_col = "domain_start" if "domain_start" in merged_df.columns else "epoch"
    for i, guid in enumerate(guids):
        ax = axes[i // ncols][i % ncols]
        sub = merged_df[merged_df["guid"] == guid].sort_values(time_col)
        if len(sub) < 2:
            ax.text(0.5, 0.5, "<2 pts", transform=ax.transAxes,
                    ha="center", va="center", color=COLOR_GRAY)
            continue
        hours = sub[time_col].values / 3600.0
        tv = sub[te_col].values.astype(float)
        kv = sub[kld_col].values.astype(float)
        tv_z = (
            (tv - np.mean(tv)) / np.std(tv)
            if np.std(tv) > 0 else np.zeros_like(tv)
        )
        kv_z = (
            (kv - np.mean(kv)) / np.std(kv)
            if np.std(kv) > 0 else np.zeros_like(kv)
        )
        ax.plot(hours, tv_z, color=COLOR_VERMILLION, marker="s",
                markersize=3, linewidth=1.0, label="TE (z)")
        ax.plot(hours, kv_z, color=COLOR_BLUE, marker="o",
                markersize=3, linewidth=1.0, label="KLD (z)")
        ax.set_title(
            f"{_format_guid(guid)}  (n={len(sub)})",
            fontsize=5.5, family="monospace", loc="left",
        )
        ax.legend(fontsize=6, loc="upper left")
        _style_axes(ax)
        if i // ncols == nrows - 1:
            ax.set_xlabel("hours (neg = pre-delivery)", fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel("z-score", fontsize=7)

    for j in range(len(guids), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Per-GUID trajectory overlays (top {len(guids)} by matches)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    out_path = output_dir / "per_guid_trajectory_overlays.pdf"
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Lag-attn v1: extra empirical-vs-model comparison plots
# ---------------------------------------------------------------------------


def plot_xcorr_lag_hist(xcorr_df: pd.DataFrame, out_path: Path) -> Path:
    """Histogram of per-GUID best lag and best xcorr value.

    Args:
        xcorr_df: Output of
            :func:`te_kld_comparison.cross_correlation_per_guid` —
            must contain ``best_lag`` and ``best_xcorr`` columns.
        out_path: Path of the PDF to write.

    Returns:
        ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if xcorr_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].hist(xcorr_df["best_lag"], bins=21, color=COLOR_BLUE, alpha=0.8)
    axes[0].axvline(0, color=COLOR_BLACK, lw=0.6)
    axes[0].set_xlabel("best lag (epochs)")
    axes[0].set_ylabel("# GUIDs")
    axes[0].set_title("Per-GUID best cross-correlation lag")
    _style_axes(axes[0])
    axes[1].hist(xcorr_df["best_xcorr"], bins=20, color=COLOR_VERMILLION, alpha=0.8)
    axes[1].axvline(0, color=COLOR_BLACK, lw=0.6)
    axes[1].set_xlabel("|best xcorr|")
    axes[1].set_ylabel("# GUIDs")
    axes[1].set_title("Per-GUID best xcorr value")
    _style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bland_altman(
    merged_df: pd.DataFrame,
    bland_stats: Dict[str, float],
    out_path: Path,
    *,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    standardize: bool = True,
) -> Path:
    """Bland-Altman agreement scatter for two scores.

    Args:
        merged_df: Merged TE-KLD DataFrame.
        bland_stats: Output of :func:`te_kld_comparison.bland_altman`.
        out_path: PDF path.
        score_col: Model-side score column.
        te_col: Empirical-TE column.
        standardize: Mirror the standardisation flag used when computing
            ``bland_stats``.

    Returns:
        ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(merged_df[score_col], dtype=float)
    y = np.asarray(merged_df[te_col], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    if standardize:
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
    diff = x - y
    avg = (x + y) / 2.0
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(avg, diff, s=10, alpha=0.45, color=COLOR_BLUE)
    ax.axhline(bland_stats["mean_diff"], color=COLOR_BLACK, lw=0.8,
               label=f"mean diff = {bland_stats['mean_diff']:.3f}")
    ax.axhline(bland_stats["loa_low"], color=COLOR_VERMILLION, lw=0.6, ls="--",
               label=f"LoA = ({bland_stats['loa_low']:.3f}, {bland_stats['loa_high']:.3f})")
    ax.axhline(bland_stats["loa_high"], color=COLOR_VERMILLION, lw=0.6, ls="--")
    ax.set_xlabel(f"mean of standardised {score_col} and {te_col}")
    ax.set_ylabel(f"{score_col} − {te_col} (standardised)")
    ax.set_title("Bland-Altman agreement")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(
    merged_df: pd.DataFrame,
    roc_stats: Dict[str, Any],
    out_path: Path,
    *,
    score_col: str = "kld",
    te_col: str = "ite_valid",
) -> Path:
    """ROC curve of "high empirical TE" detection by a model score.

    Args:
        merged_df: Merged TE-KLD DataFrame.
        roc_stats: Output of :func:`te_kld_comparison.roc_auc_high_te`
            (used for the threshold + AUC annotation).
        out_path: PDF path.
        score_col: Model-side score column.
        te_col: Empirical-TE column.

    Returns:
        ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(merged_df[score_col], dtype=float)
    y = np.asarray(merged_df[te_col], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if x.size == 0 or not np.isfinite(roc_stats.get("threshold", float("nan"))):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    thr = float(roc_stats["threshold"])
    pos = y >= thr
    if pos.sum() == 0 or (~pos).sum() == 0:
        ax.text(0.5, 0.5, "Degenerate split", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    sort_idx = np.argsort(-x)
    pos_sorted = pos[sort_idx]
    tpr = np.cumsum(pos_sorted) / max(int(pos.sum()), 1)
    fpr = np.cumsum(~pos_sorted) / max(int((~pos).sum()), 1)
    ax.plot([0, 1], [0, 1], color=COLOR_GRAY, lw=0.6, ls="--")
    ax.plot(fpr, tpr, color=COLOR_BLUE, lw=1.2,
            label=f"AUC = {roc_stats.get('auc', float('nan')):.3f}")
    ax.set_xlabel(f"FPR (negatives: {te_col} < {thr:.3f})")
    ax.set_ylabel(f"TPR (positives: {te_col} ≥ {thr:.3f})")
    ax.set_title(f"ROC: detect high {te_col} via {score_col}")
    ax.legend(loc="lower right", frameon=True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_guid_slope_hist(reg_df: pd.DataFrame, out_path: Path) -> Path:
    """Histogram of per-GUID regression slopes and R² values.

    Args:
        reg_df: Output of :func:`te_kld_comparison.per_guid_regression`.
        out_path: PDF path.

    Returns:
        ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if reg_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    axes[0].hist(reg_df["slope"], bins=20, color=COLOR_BLUE, alpha=0.8)
    axes[0].axvline(0, color=COLOR_BLACK, lw=0.6)
    axes[0].set_xlabel("regression slope (model = a · TE + b)")
    axes[0].set_ylabel("# GUIDs")
    axes[0].set_title("Per-GUID slope distribution")
    _style_axes(axes[0])
    axes[1].hist(reg_df["r2"], bins=20, color=COLOR_GREEN, alpha=0.8, range=(0, 1))
    axes[1].set_xlabel("per-GUID R²")
    axes[1].set_ylabel("# GUIDs")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Per-GUID R²")
    _style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_conditional_ks_grid(
    ks_df: pd.DataFrame,
    out_path: Path,
    *,
    score_col: str = "kld",
    te_col: str = "ite_valid",
) -> Path:
    """Bar chart of KS statistics by empirical-TE quartile.

    Args:
        ks_df: Output of
            :func:`te_kld_comparison.conditional_ks_by_quartile`.
        out_path: PDF path.
        score_col: Model-side score column (used in axis label).
        te_col: Empirical-TE column (used in axis label).

    Returns:
        ``out_path``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if ks_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    quartiles = ks_df["quartile"].astype(int).to_list()
    stats = ks_df["ks_stat"].astype(float).to_list()
    pvals = ks_df["p_value"].astype(float).to_list()
    bars = ax.bar(quartiles, stats, color=COLOR_BLUE, alpha=0.8)
    for q, p, b in zip(quartiles, pvals, bars):
        if np.isfinite(p):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.005,
                    f"p={p:.2g}",
                    ha="center", va="bottom", fontsize=7, color=COLOR_BLACK)
    ax.set_xlabel(f"{te_col} quartile (vs Q1 reference)")
    ax.set_ylabel(f"KS statistic on {score_col}")
    ax.set_title("Conditional KS by TE quartile")
    ax.set_xticks(quartiles)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Distance-based comparisons (Euclidean / RMSE / NRMSE / cosine / Frechet)
# ---------------------------------------------------------------------------


_DISTANCE_METRIC_ORDER: Sequence[str] = (
    "euclidean", "rmse", "nrmse", "cosine", "frechet_discrete",
)

# Metrics where higher = better (similarity) rather than lower = better
# (distance). Plot helpers that rank GUIDs or color cells by "alignment
# quality" must invert these so the convention "dark / top = best" is
# uniform across the figure.
_HIGHER_IS_BETTER_METRICS: frozenset = frozenset({"cosine"})


def _alignment_quality(values: np.ndarray, metric: str) -> np.ndarray:
    """Map raw metric values to a "lower = better" alignment quality.

    For distance metrics this is the identity; for similarity metrics
    (currently just ``cosine``) this returns ``-values`` so the same
    direction logic ("smaller is better") works for sorting and
    colormap normalisation.
    """
    arr = np.asarray(values, dtype=float)
    if metric in _HIGHER_IS_BETTER_METRICS:
        return -arr
    return arr


def _ensure_path(out_path: Path) -> Path:
    """Coerce to ``Path`` and create parent directories."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_per_guid_distance_heatmap(
    per_guid_dist_df: pd.DataFrame,
    out_path: Path,
    *,
    metrics: Optional[Sequence[str]] = None,
    score_col: Optional[str] = None,
) -> Path:
    """Heatmap of per-GUID distances across metrics (rows=GUIDs, cols=metrics).

    Each metric column is independently min-max scaled so the colormap
    is comparable across metrics with different units. The numeric value
    of each cell is annotated in the original scale.

    Args:
        per_guid_dist_df: Output of
            :func:`te_kld_distance_metrics.compute_per_guid_distances`.
        out_path: PDF target.
        metrics: Optional subset of distance columns to render. Defaults
            to all metrics from ``_DISTANCE_METRIC_ORDER`` that exist on
            the DataFrame.
        score_col: Optional score column name to surface in the title
            (e.g. ``"kld_pca_l2_top3"``).

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if per_guid_dist_df is None or per_guid_dist_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    metrics = list(metrics) if metrics else [
        m for m in _DISTANCE_METRIC_ORDER if m in per_guid_dist_df.columns
    ]
    if not metrics:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No distance metrics", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Rank GUIDs by mean rank across metrics on the "lower = better"
    # alignment-quality scale, so cosine (a similarity) does not pull
    # well-aligned GUIDs to the bottom.
    df = per_guid_dist_df.copy()
    quality_cols = pd.DataFrame(
        {m: _alignment_quality(df[m].to_numpy(dtype=float), m) for m in metrics},
        index=df.index,
    )
    df["__rank"] = quality_cols.rank(method="average").mean(axis=1)
    df = df.sort_values("__rank").reset_index(drop=True)
    df = df.drop(columns="__rank")

    arr = df[metrics].to_numpy(dtype=float)
    norm = np.full_like(arr, np.nan, dtype=float)
    for j, m in enumerate(metrics):
        # Normalise on the alignment-quality scale so that, in every
        # column, "dark = good alignment" and "bright = bad alignment".
        # Cell text below still reports the raw value.
        quality = _alignment_quality(arr[:, j], m)
        finite = np.isfinite(quality)
        if finite.sum() == 0:
            continue
        lo = float(np.nanmin(quality))
        hi = float(np.nanmax(quality))
        span = hi - lo
        if span <= 1e-12:
            norm[finite, j] = 0.0
        else:
            norm[finite, j] = (quality[finite] - lo) / span

    n_rows = len(df)
    height = max(3.0, 0.18 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, height))
    im = ax.imshow(
        norm, aspect="auto", cmap="viridis",
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [_format_guid(g) for g in df["guid"].astype(str)],
        fontsize=4.5, family="monospace",
    )

    for i in range(n_rows):
        for j in range(len(metrics)):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=4.5, color="white" if norm[i, j] > 0.55 else COLOR_BLACK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("min-max within metric", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    suffix = f" — score={score_col}" if score_col else ""
    ax.set_title(f"Per-GUID distances{suffix}", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distance_distribution(
    per_guid_dist_df: pd.DataFrame,
    out_path: Path,
    *,
    score_col: Optional[str] = None,
) -> Path:
    """One-panel-per-metric box+strip plot of per-GUID distances.

    The strip overlay surfaces every patient as a single point, which
    makes outlier GUIDs visually obvious. The top 3 GUIDs (largest
    distance) on each metric are annotated with their full ID.

    Args:
        per_guid_dist_df: Output of
            :func:`te_kld_distance_metrics.compute_per_guid_distances`.
        out_path: PDF target.
        score_col: Optional score column name surfaced in the title.

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    metrics = [m for m in _DISTANCE_METRIC_ORDER if m in per_guid_dist_df.columns]
    if per_guid_dist_df is None or per_guid_dist_df.empty or not metrics:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n + 0.6, 3.4))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(7)
    for ax, metric in zip(axes, metrics):
        vals = per_guid_dist_df[metric].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        finite_vals = vals[finite]
        if finite_vals.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(metric, fontsize=8)
            continue
        ax.boxplot(
            finite_vals, vert=True, widths=0.55, showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_LIGHT_GRAY, color=COLOR_BLACK,
                          linewidth=0.8),
            medianprops=dict(color=COLOR_VERMILLION, linewidth=1.0),
            whiskerprops=dict(color=COLOR_BLACK, linewidth=0.6),
            capprops=dict(color=COLOR_BLACK, linewidth=0.6),
        )
        jitter = rng.uniform(0.85, 1.15, size=finite_vals.size)
        ax.scatter(jitter, finite_vals, s=10, color=COLOR_BLUE,
                   alpha=0.75, edgecolors="none")
        # Annotate the 3 worst-aligned GUIDs (= largest distance for
        # "lower-is-better" metrics, smallest similarity for cosine).
        # Ranking on the alignment-quality scale keeps the convention
        # uniform across every panel.
        quality = _alignment_quality(finite_vals, metric)
        order = np.argsort(quality)[::-1][:3]
        guids_finite = per_guid_dist_df.loc[finite, "guid"].astype(str).to_numpy()
        for k in order:
            ax.annotate(
                _format_guid(guids_finite[k]),
                xy=(jitter[k], finite_vals[k]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=4, family="monospace", color=COLOR_GRAY,
                va="center", ha="left",
            )
        ax.set_title(metric, fontsize=8)
        ax.set_xticks([1])
        ax.set_xticklabels([f"n={finite_vals.size}"], fontsize=7)
        _style_axes(ax)

    suffix = f" — score={score_col}" if score_col else ""
    fig.suptitle(f"Per-GUID distance distribution{suffix}", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_guid_distance_bar(
    per_guid_dist_df: pd.DataFrame,
    out_path: Path,
    *,
    metric: str = "euclidean",
    score_col: Optional[str] = None,
) -> Path:
    """Sorted horizontal bar chart of one distance metric per GUID.

    Bars are sorted ascending (best alignment at the top) and coloured
    by outcome class when a ``label`` column is present.

    Args:
        per_guid_dist_df: Output of
            :func:`te_kld_distance_metrics.compute_per_guid_distances`.
        out_path: PDF target.
        metric: Distance column to plot (default ``"euclidean"``).
        score_col: Optional score column for the title.

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if (
        per_guid_dist_df is None or per_guid_dist_df.empty
        or metric not in per_guid_dist_df.columns
    ):
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    df = per_guid_dist_df.dropna(subset=[metric]).copy()
    # Sort on the alignment-quality scale so that "best at top" holds
    # for both distance metrics (lower is better) and similarity metrics
    # like cosine (higher is better).
    df["__quality"] = _alignment_quality(df[metric].to_numpy(dtype=float), metric)
    df = df.sort_values("__quality", ascending=True).reset_index(drop=True)
    df = df.drop(columns="__quality")

    has_label = "label" in df.columns
    if has_label:
        try:
            from model.vae_teb_prediction.testing.visualizers import class_color_for
            colors = [
                class_color_for(int(lab)) if pd.notna(lab) else COLOR_BLUE
                for lab in df["label"]
            ]
        except Exception:  # noqa: BLE001
            colors = [COLOR_BLUE] * len(df)
    else:
        colors = [COLOR_BLUE] * len(df)

    height = max(3.0, 0.18 * len(df) + 1.6)
    fig, ax = plt.subplots(figsize=(7.5, height))
    ax.barh(range(len(df)), df[metric].to_numpy(dtype=float),
            color=colors, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [_format_guid(g) for g in df["guid"].astype(str)],
        fontsize=4.5, family="monospace",
    )
    ax.invert_yaxis()
    ax.set_xlabel(metric, fontsize=8)

    if "n_pairs" in df.columns:
        for i, (val, n_pairs) in enumerate(zip(
            df[metric].to_numpy(dtype=float), df["n_pairs"].to_numpy()
        )):
            if np.isfinite(val):
                ax.text(val, i, f"  n={int(n_pairs)}",
                        ha="left", va="center", fontsize=5,
                        color=COLOR_GRAY)

    suffix = f" — score={score_col}" if score_col else ""
    ax.set_title(f"Per-GUID {metric}{suffix}", fontsize=10, loc="left")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_residual_trajectory_grid(
    merged_df: pd.DataFrame,
    per_guid_dist_df: pd.DataFrame,
    out_path: Path,
    *,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    metric: str = "euclidean",
    n_top: int = 6,
    n_bottom: int = 6,
) -> Path:
    """Side-by-side trajectories of best- and worst-aligned GUIDs.

    Z-scores both TE and the elected score per GUID, then renders the
    paired trajectories on a shared y-axis with the residual highlighted
    as a shaded band. The "top" rows show the GUIDs with the **smallest**
    distance (best alignment), and the "bottom" rows show the **largest**
    distance (worst alignment).

    Args:
        merged_df: Output of :func:`merge_te_kld`.
        per_guid_dist_df: Output of
            :func:`te_kld_distance_metrics.compute_per_guid_distances`.
        out_path: PDF target.
        score_col: Model-side score column to plot.
        te_col: Empirical TE column.
        metric: Distance column used to rank GUIDs.
        n_top: Number of best-aligned GUIDs (top of figure).
        n_bottom: Number of worst-aligned GUIDs (bottom of figure).

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if (
        merged_df is None or merged_df.empty
        or per_guid_dist_df is None or per_guid_dist_df.empty
        or metric not in per_guid_dist_df.columns
    ):
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # Rank on alignment-quality so the "best" head and "worst" tail
    # are correct for both distance metrics (lower=better) and the
    # cosine similarity (higher=better).
    ranked = per_guid_dist_df.dropna(subset=[metric]).copy()
    ranked["__quality"] = _alignment_quality(
        ranked[metric].to_numpy(dtype=float), metric,
    )
    ranked = ranked.sort_values("__quality").drop(columns="__quality")
    if ranked.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "All distances NaN", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    top = ranked.head(n_top)
    bottom = ranked.tail(n_bottom)
    if metric in _HIGHER_IS_BETTER_METRICS:
        best_label = f"Best-aligned (largest {metric})"
        worst_label = f"Worst-aligned (smallest {metric})"
    else:
        best_label = f"Best-aligned (smallest {metric})"
        worst_label = f"Worst-aligned (largest {metric})"
    pages = [(best_label, top), (worst_label, bottom)]

    time_col = (
        "domain_start" if "domain_start" in merged_df.columns
        else "epoch" if "epoch" in merged_df.columns
        else merged_df.columns[1]
    )

    n_rows = max(len(top), len(bottom))
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(11, max(2.2, 1.4 * n_rows + 0.5)),
        squeeze=False,
    )
    for col_idx, (header, sub) in enumerate(pages):
        guids = sub["guid"].astype(str).tolist()
        for row_idx in range(n_rows):
            ax = axes[row_idx][col_idx]
            if row_idx >= len(guids):
                ax.set_visible(False)
                continue
            guid = guids[row_idx]
            metric_val = float(sub.iloc[row_idx][metric])
            data = merged_df[merged_df["guid"] == guid].sort_values(time_col)
            hours = data[time_col].to_numpy(dtype=float) / 3600.0
            t_vals = data[te_col].to_numpy(dtype=float)
            s_vals = data[score_col].to_numpy(dtype=float)

            def _z(arr: np.ndarray) -> np.ndarray:
                a = np.asarray(arr, dtype=float)
                if a.size < 2 or np.std(a[np.isfinite(a)]) <= 1e-12:
                    return np.zeros_like(a)
                return (a - np.nanmean(a)) / np.nanstd(a)

            tz = _z(t_vals)
            sz = _z(s_vals)
            ax.plot(hours, tz, color=COLOR_VERMILLION, marker="s",
                    markersize=2.5, linewidth=1.0, label=f"{te_col} (z)")
            ax.plot(hours, sz, color=COLOR_BLUE, marker="o",
                    markersize=2.5, linewidth=1.0, label=f"{score_col} (z)")
            ax.fill_between(hours, sz, tz,
                             color=COLOR_GRAY, alpha=0.18, linewidth=0)
            ax.set_title(
                f"{_format_guid(guid)}  ({metric}={metric_val:.2f})",
                fontsize=5.5, family="monospace", loc="left",
            )
            ax.tick_params(labelsize=6)
            _style_axes(ax)
            if row_idx == 0:
                ax.legend(fontsize=5, loc="upper left")
            if row_idx == n_rows - 1:
                ax.set_xlabel("hours (negative = pre-delivery)", fontsize=6)

    fig.suptitle(
        f"Residual trajectory grid — {te_col} vs {score_col}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PCA-Euclidean diagnostics
# ---------------------------------------------------------------------------


def plot_pca_distance_curve(
    pca_search_summary_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Pooled & macro-mean Euclidean distance vs number of PCs included.

    One line per (source, ranking, aggregator) tuple, plus stars marking
    the elected best subsets per source. Only ``aggregator == "l2"`` is
    plotted by default to keep the figure readable; the underlying CSV
    contains both aggregators for downstream analysis.

    Args:
        pca_search_summary_df: ``summary_df`` from
            :func:`te_kld_distance_metrics.pca_distance_search`.
        out_path: PDF target.

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if pca_search_summary_df is None or pca_search_summary_df.empty:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No PCA inputs", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    df = pca_search_summary_df[
        pca_search_summary_df["aggregator"] == "l2"
    ].copy()
    if df.empty:
        df = pca_search_summary_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    palette = {
        "existing|variance": (COLOR_BLUE, "-"),
        "existing|te_corr":  (COLOR_BLUE, "--"),
        "refit|variance":    (COLOR_VERMILLION, "-"),
        "refit|te_corr":     (COLOR_VERMILLION, "--"),
    }
    for ax, ycol, title in (
        (axes[0], "pooled_euclidean", "Pooled Euclidean"),
        (axes[1], "macro_mean_euclidean", "Macro-mean per-GUID Euclidean"),
    ):
        for (source, ranking), grp in df.groupby(["source", "ranking"]):
            ordered = grp.sort_values("k")
            color, ls = palette.get(
                f"{source}|{ranking}", (COLOR_GRAY, "-")
            )
            ax.plot(
                ordered["k"].to_numpy(dtype=float),
                ordered[ycol].to_numpy(dtype=float),
                marker="o", markersize=4, linewidth=1.2,
                color=color, linestyle=ls,
                label=f"{source} | {ranking}",
            )
            # mark best per (source, ranking)
            arr = ordered[ycol].to_numpy(dtype=float)
            finite = np.isfinite(arr)
            if finite.any():
                idx = int(np.nanargmin(np.where(finite, arr, np.inf)))
                ax.scatter(
                    [ordered["k"].to_numpy()[idx]],
                    [arr[idx]],
                    marker="*", color=color, s=70,
                    edgecolors=COLOR_BLACK, linewidths=0.4, zorder=5,
                )
        ax.set_xlabel("# PCs included (cumulative)", fontsize=7)
        ax.set_ylabel(ycol, fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6, loc="best")
        _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_pc_distance_heatmap(
    per_pc_distance_df: pd.DataFrame,
    out_path: Path,
    *,
    metric: str = "euclidean",
) -> Path:
    """Heatmap rows=GUIDs cols=PCs cells=single-PC distance to TE.

    Useful for asking *which PC* aligns best with TE for *which patient*.

    Args:
        per_pc_distance_df: ``per_pc_distance_df`` from
            :func:`te_kld_distance_metrics.pca_distance_search`. Must
            contain ``source``, ``pc``, ``guid`` and a distance column.
        out_path: PDF target.
        metric: Distance column to render (default ``"euclidean"``).

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if (
        per_pc_distance_df is None or per_pc_distance_df.empty
        or metric not in per_pc_distance_df.columns
    ):
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    sources = sorted(per_pc_distance_df["source"].astype(str).unique())
    n_sources = len(sources)
    fig, axes = plt.subplots(
        1, n_sources,
        figsize=(5.5 * n_sources, max(3.2, 0.18 * per_pc_distance_df["guid"].nunique() + 1.8)),
        squeeze=False,
    )

    for ax, source in zip(axes[0], sources):
        sub = per_pc_distance_df[per_pc_distance_df["source"] == source]
        pivot = sub.pivot_table(
            index="guid", columns="pc", values=metric, aggfunc="mean",
        )
        # Order PCs naturally (kld_pc1, ..., refit_pc5).
        cols = list(pivot.columns)
        cols.sort(key=lambda c: (
            0 if str(c).startswith("kld_pc") else 1,
            int("".join(ch for ch in str(c) if ch.isdigit()) or "0"),
        ))
        pivot = pivot[cols]
        # Order rows by alignment-quality (best-aligned at top), so the
        # row order works for both distance metrics and cosine.
        quality_arr = _alignment_quality(pivot.to_numpy(dtype=float), metric)
        row_quality = np.nanmean(quality_arr, axis=1)
        order_idx = np.argsort(row_quality)
        pivot = pivot.iloc[order_idx]

        arr = pivot.to_numpy(dtype=float)
        if arr.size == 0:
            ax.set_visible(False)
            continue

        im = ax.imshow(
            arr, aspect="auto", cmap="magma_r",
            interpolation="nearest",
        )
        ax.set_xticks(range(arr.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=6, rotation=30, ha="right")
        ax.set_yticks(range(arr.shape[0]))
        ax.set_yticklabels(
            [_format_guid(g) for g in pivot.index.astype(str)],
            fontsize=4.5, family="monospace",
        )
        ax.set_title(
            f"source={source}: {metric} of single-PC trajectory vs TE",
            fontsize=8, loc="left",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label(metric, fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_joint_te_kld_pca(
    joint_pca_result: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Two-panel diagnostic for the joint TE-KLD PCA.

    Left panel: bar chart of the loadings of every input variable on
    PC1..PCk (with the TE row highlighted). Right panel: PC1 vs PC2
    scatter coloured by ``ite_valid`` quartile, with per-GUID centroids
    annotated with their full GUID.

    Args:
        joint_pca_result: Output of
            :func:`te_kld_distance_metrics.joint_te_kld_pca`.
        out_path: PDF target.

    Returns:
        ``out_path``.
    """
    out_path = _ensure_path(out_path)
    if (
        joint_pca_result is None
        or "loadings_df" not in joint_pca_result
        or "scores_df" not in joint_pca_result
    ):
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        msg = "joint PCA unavailable"
        if isinstance(joint_pca_result, dict) and "error" in joint_pca_result:
            msg = f"joint PCA: {joint_pca_result['error']}"
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    loadings_df: pd.DataFrame = joint_pca_result["loadings_df"]
    scores_df: pd.DataFrame = joint_pca_result["scores_df"]
    evr = joint_pca_result.get("explained_variance_ratio", [])
    te_col = joint_pca_result.get("te_col", "ite_valid")

    fig = plt.figure(figsize=(11.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.25)
    ax_load = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])

    pc_cols = [c for c in loadings_df.columns if c.startswith("PC")]
    n_pc = len(pc_cols)
    bar_width = 0.8 / max(n_pc, 1)
    x = np.arange(len(loadings_df))
    palette = [COLOR_BLUE, COLOR_VERMILLION, COLOR_GREEN,
               COLOR_ORANGE, COLOR_GRAY, COLOR_BLACK]
    for i, pc in enumerate(pc_cols):
        ev = evr[i] if i < len(evr) else float("nan")
        offset = (i - (n_pc - 1) / 2.0) * bar_width
        ax_load.bar(
            x + offset, loadings_df[pc].to_numpy(dtype=float),
            width=bar_width,
            color=palette[i % len(palette)],
            alpha=0.85,
            label=f"{pc} ({ev:.1%})",
        )
    ax_load.axhline(0.0, color=COLOR_BLACK, linewidth=0.5)
    ax_load.set_xticks(x)
    ax_load.set_xticklabels(loadings_df["variable"], rotation=20, ha="right",
                            fontsize=7)
    # Highlight the TE row
    for tick in ax_load.get_xticklabels():
        if tick.get_text() == te_col:
            tick.set_color(COLOR_VERMILLION)
            tick.set_fontweight("bold")
    ax_load.set_ylabel("Loading", fontsize=8)
    ax_load.set_title("Joint TE-KLD PCA: loadings", fontsize=9, loc="left")
    ax_load.legend(fontsize=6, loc="best")
    _style_axes(ax_load)

    if "PC1" in scores_df.columns and "PC2" in scores_df.columns:
        scatter_df = scores_df.dropna(subset=["PC1", "PC2"])
        if te_col in scatter_df.columns:
            te_vals = scatter_df[te_col].to_numpy(dtype=float)
            try:
                quartiles = pd.qcut(te_vals, 4, labels=False, duplicates="drop")
                quartiles = np.asarray(quartiles, dtype=float)
            except Exception:  # noqa: BLE001
                quartiles = np.zeros_like(te_vals)
            colormap = plt.colormaps.get_cmap("viridis")
            colors = colormap(np.where(np.isfinite(quartiles), quartiles, 0) / 3.0)
            ax_scatter.scatter(
                scatter_df["PC1"].to_numpy(dtype=float),
                scatter_df["PC2"].to_numpy(dtype=float),
                s=10, c=colors, alpha=0.65, edgecolors="none",
                rasterized=True,
            )
            sm = plt.cm.ScalarMappable(
                cmap=colormap, norm=plt.Normalize(vmin=0, vmax=3),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_scatter, fraction=0.04, pad=0.02)
            cbar.set_label(f"{te_col} quartile (Q1-Q4)", fontsize=6)
            cbar.ax.tick_params(labelsize=5)
        else:
            ax_scatter.scatter(
                scatter_df["PC1"].to_numpy(dtype=float),
                scatter_df["PC2"].to_numpy(dtype=float),
                s=10, color=COLOR_BLUE, alpha=0.6, edgecolors="none",
            )
        # Per-GUID centroids; annotate the top loaders in |TE| of PC1.
        if "guid" in scatter_df.columns:
            grouped = scatter_df.groupby("guid")[["PC1", "PC2"]].mean()
            ax_scatter.scatter(
                grouped["PC1"].to_numpy(dtype=float),
                grouped["PC2"].to_numpy(dtype=float),
                s=18, marker="x", color=COLOR_BLACK, linewidths=0.6,
            )
            # Annotate top-N centroids by |PC1| value.
            order = grouped["PC1"].abs().sort_values(ascending=False).index[:6]
            for guid in order:
                row = grouped.loc[guid]
                ax_scatter.annotate(
                    _format_guid(guid),
                    xy=(float(row["PC1"]), float(row["PC2"])),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=4, family="monospace",
                    color=COLOR_BLACK,
                )
        ax_scatter.set_xlabel(
            f"PC1 ({evr[0]:.1%})" if evr else "PC1", fontsize=7,
        )
        ax_scatter.set_ylabel(
            f"PC2 ({evr[1]:.1%})" if len(evr) > 1 else "PC2", fontsize=7,
        )
        ax_scatter.set_title(
            "PC1 vs PC2 — colored by empirical TE quartile",
            fontsize=9, loc="left",
        )
        _style_axes(ax_scatter)
    else:
        ax_scatter.text(0.5, 0.5, "PC1/PC2 unavailable",
                         transform=ax_scatter.transAxes,
                         ha="center", va="center")

    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
