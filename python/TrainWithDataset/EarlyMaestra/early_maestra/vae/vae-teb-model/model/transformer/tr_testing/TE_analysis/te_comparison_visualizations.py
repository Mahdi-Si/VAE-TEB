"""Publication-quality visualisations for model TE vs empirical TE comparison.

All plots follow the conventions in ``model.transformer.tr_testing.style``:
serif fonts, DPI 600, four-spine axes, and the project colour palette.

Example:
    >>> from model.transformer.tr_testing.TE_analysis.te_comparison_visualizations import (
    ...     plot_correlation_heatmap, plot_trajectory_overlay,
    ... )
    >>> plot_correlation_heatmap(corr_df, pval_df, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.patches import FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from model.transformer.tr_testing.style import (
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    COLOR_GREEN,
    COLOR_SKY,
    COLOR_GRAY,
    COLOR_BLACK,
    COLOR_LIGHT_GRAY,
    SAVE_DPI,
    apply_publication_style,
    style_axes,
    save_figure,
    add_colorbar,
)

from loguru import logger

# Per-GUID colours (3 distinct colours for the 3 overlapping GUIDs)
GUID_COLORS = [COLOR_BLUE, COLOR_ORANGE, COLOR_VERMILLION]


def _get_guid_color_map(guids: Sequence[str]) -> Dict[str, str]:
    """Assign a colour to each GUID."""
    return {g: GUID_COLORS[i % len(GUID_COLORS)] for i, g in enumerate(sorted(guids))}


def _short_guid(guid: str, n: int = 8) -> str:
    """Truncate GUID for display."""
    return guid[:n] + "..."


# ---------------------------------------------------------------------------
# 1. Data quality summary
# ---------------------------------------------------------------------------


def plot_data_quality_summary(
    quality_report: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Data quality infographic with GUID overlap, stats table, and time-gap histogram.

    Three-panel figure:
      - Left: annotated text boxes showing GUID overlap counts.
      - Centre: summary statistics table.
      - Right: histogram of ``time_gap_seconds`` for matched pairs.

    Args:
        quality_report: Output of
            :func:`~te_data_loader.compute_data_quality_report`.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # --- Left: GUID overlap diagram ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("GUID Overlap", fontsize=9, fontweight="bold", pad=8)

    n_model = quality_report["model_unique_guids"]
    n_emp = quality_report["empirical_unique_guids"]
    n_common = quality_report["common_guids"]

    ax.text(2.5, 7.5, f"Model\n{n_model} GUIDs", ha="center", va="center",
            fontsize=9, fontweight="bold", color=COLOR_BLUE,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_BLUE, alpha=0.15))
    ax.text(7.5, 7.5, f"Empirical\n{n_emp} GUIDs", ha="center", va="center",
            fontsize=9, fontweight="bold", color=COLOR_VERMILLION,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_VERMILLION, alpha=0.15))
    ax.text(5.0, 4.5, f"Overlap\n{n_common} GUIDs", ha="center", va="center",
            fontsize=10, fontweight="bold", color=COLOR_GREEN,
            bbox=dict(boxstyle="round,pad=0.6", facecolor=COLOR_GREEN, alpha=0.2))
    ax.annotate("", xy=(3.8, 5.0), xytext=(3.2, 6.8),
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=0.8))
    ax.annotate("", xy=(6.2, 5.0), xytext=(6.8, 6.8),
                arrowprops=dict(arrowstyle="->", color=COLOR_GRAY, lw=0.8))

    ax.text(5.0, 2.0,
            f"Model-only: {quality_report['model_only_guids']}\n"
            f"Empirical-only: {quality_report['empirical_only_guids']}",
            ha="center", va="center", fontsize=7, color=COLOR_GRAY)

    # --- Centre: summary table ---
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Matching Summary", fontsize=9, fontweight="bold", pad=8)

    gap_stats = quality_report.get("time_gap_stats", {})
    table_data = [
        ["Matched pairs", str(quality_report["matched_pairs"])],
        ["Matched GUIDs", str(quality_report["matched_guids"])],
        ["Mean time gap", f"{gap_stats.get('mean', 0):.1f} s"],
        ["Max time gap", f"{gap_stats.get('max', 0):.1f} s"],
        ["Model segments", str(quality_report["model_total_segments"])],
        ["Empirical epochs", str(quality_report["empirical_total_epochs"])],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLOR_LIGHT_GRAY)
        if row == 0:
            cell.set_facecolor(COLOR_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white")

    # --- Right: time-gap histogram ---
    ax = axes[2]
    style_axes(ax)
    ax.set_title("Time Gap Distribution", fontsize=9, fontweight="bold", pad=8)

    per_guid = quality_report.get("per_guid_matching", [])
    if per_guid:
        # We don't have individual gaps in the report, so show per-GUID bars
        guids_short = [_short_guid(g["guid"]) for g in per_guid]
        n_matched = [g["n_matched"] for g in per_guid]
        mean_gaps = [g["mean_gap"] for g in per_guid]
        colors = GUID_COLORS[:len(per_guid)]

        x_pos = np.arange(len(per_guid))
        bars = ax.bar(x_pos, n_matched, color=colors, edgecolor=COLOR_BLACK, linewidth=0.4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(guids_short, rotation=30, ha="right")
        ax.set_ylabel("Matched pairs", fontsize=8)
        ax.set_xlabel("GUID", fontsize=8)

        # Annotate with mean gap
        for i, (bar, mg) in enumerate(zip(bars, mean_gaps)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{mg:.0f}s", ha="center", va="bottom", fontsize=6,
                    color=COLOR_GRAY)
    else:
        ax.text(0.5, 0.5, "No matched data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color=COLOR_GRAY)

    fig.tight_layout()
    path = output_dir / "data_quality_summary.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 2. Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    output_dir: Path,
    title: str = "Model vs Empirical TE: Spearman Correlation",
) -> Path:
    """Heatmap of Spearman rho between model and empirical measures.

    Annotated with rho values and significance stars
    (* p < 0.05, ** p < 0.01, *** p < 0.001).

    Args:
        corr_df: Correlation DataFrame (model rows x empirical cols).
        pval_df: P-value DataFrame (same shape).
        output_dir: Directory to save the figure.
        title: Figure title.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    n_rows, n_cols = corr_df.shape
    fig, ax = plt.subplots(figsize=(max(4, n_cols * 1.2 + 1), max(3, n_rows * 0.8 + 1)))

    data = corr_df.values
    vabs = max(np.nanmax(np.abs(data)), 0.1)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    # Annotations
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            pval = pval_df.values[i, j]
            if np.isnan(val):
                text = "—"
            else:
                stars = ""
                if np.isfinite(pval):
                    if pval < 0.001:
                        stars = "***"
                    elif pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                text = f"{val:.2f}{stars}"

            color = "white" if abs(val) > vabs * 0.6 else COLOR_BLACK
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(corr_df.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(corr_df.index, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=10)

    add_colorbar(fig, im, ax, label="Spearman ρ")

    fig.tight_layout()
    path = output_dir / "correlation_heatmap.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 3. Per-dimension KL heatmap
# ---------------------------------------------------------------------------


def plot_per_dimension_kl_heatmap(
    dim_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Heatmap of Spearman rho between each KL dimension and empirical TE measures.

    Rows = latent dimensions 0..15, columns = empirical measures.  Annotated
    with significance stars from permutation tests.

    Args:
        dim_df: Output of
            :func:`~te_comparison_analysis.per_dimension_analysis`.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()

    empirical_measures = dim_df["empirical_measure"].unique()
    n_dims = dim_df["dimension"].nunique()
    n_emp = len(empirical_measures)

    data = np.full((n_dims, n_emp), np.nan)
    pvals = np.full((n_dims, n_emp), np.nan)

    for _, row in dim_df.iterrows():
        d = int(row["dimension"])
        j = list(empirical_measures).index(row["empirical_measure"])
        data[d, j] = row["spearman_rho"]
        pvals[d, j] = row["permutation_p"]

    fig, ax = plt.subplots(figsize=(max(4, n_emp * 1.3 + 1), max(4, n_dims * 0.35 + 1)))
    vabs = max(np.nanmax(np.abs(data)), 0.1)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    for i in range(n_dims):
        for j in range(n_emp):
            val = data[i, j]
            pval = pvals[i, j]
            if np.isnan(val):
                text = "—"
            else:
                stars = ""
                if np.isfinite(pval):
                    if pval < 0.01:
                        stars = "**"
                    elif pval < 0.05:
                        stars = "*"
                text = f"{val:.2f}{stars}"
            color = "white" if abs(val) > vabs * 0.6 else COLOR_BLACK
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)

    ax.set_xticks(range(n_emp))
    ax.set_xticklabels(empirical_measures, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels([f"dim {d}" for d in range(n_dims)], fontsize=6)
    ax.set_title("KL Dimension vs Empirical TE: Spearman ρ", fontsize=9,
                 fontweight="bold", pad=10)
    ax.set_xlabel("Empirical TE measure", fontsize=8)
    ax.set_ylabel("KL latent dimension", fontsize=8)

    add_colorbar(fig, im, ax, label="Spearman ρ")

    fig.tight_layout()
    path = output_dir / "per_dimension_kl_heatmap.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 4. Scatter matrix
# ---------------------------------------------------------------------------


def plot_scatter_matrix(
    merged_df: pd.DataFrame,
    model_cols: List[str],
    empirical_cols: List[str],
    output_dir: Path,
    perm_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """Grid of scatter plots: each model measure vs each empirical measure.

    Points are coloured by GUID.  Each panel shows Spearman rho and
    permutation p-value (if available).

    Args:
        merged_df: Merged DataFrame.
        model_cols: Model measure column names.
        empirical_cols: Empirical measure column names.
        output_dir: Directory to save the figure.
        perm_results: Optional permutation test results dict.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    n_r, n_c = len(model_cols), len(empirical_cols)
    fig, axes = plt.subplots(n_r, n_c, figsize=(n_c * 2.5, n_r * 2.5),
                             squeeze=False)

    guid_cmap = _get_guid_color_map(merged_df["guid"].unique())

    from scipy import stats as sp_stats

    for i, mc in enumerate(model_cols):
        for j, ec in enumerate(empirical_cols):
            ax = axes[i, j]
            style_axes(ax)

            for guid, color in guid_cmap.items():
                mask = merged_df["guid"] == guid
                ax.scatter(
                    merged_df.loc[mask, ec],
                    merged_df.loc[mask, mc],
                    c=color, s=18, alpha=0.8, edgecolors=COLOR_BLACK,
                    linewidths=0.3, label=_short_guid(guid),
                    zorder=3,
                )

            # Regression line
            x = merged_df[ec].values.astype(float)
            y = merged_df[mc].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 3:
                rho, _ = sp_stats.spearmanr(x[m], y[m])
                slope, intercept = np.polyfit(x[m], y[m], 1)
                x_line = np.linspace(x[m].min(), x[m].max(), 50)
                ax.plot(x_line, slope * x_line + intercept, color=COLOR_GRAY,
                        lw=0.8, ls="--", alpha=0.7)

                # Annotation
                p_text = ""
                key = f"{mc}_vs_{ec}"
                if perm_results and key in perm_results:
                    p_val = perm_results[key].get("p_value", None)
                    if p_val is not None:
                        p_text = f"\np(perm)={p_val:.3f}"
                ax.text(0.05, 0.95, f"ρ={rho:.2f}{p_text}",
                        transform=ax.transAxes, fontsize=6, va="top",
                        color=COLOR_BLACK,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

            if i == n_r - 1:
                ax.set_xlabel(ec, fontsize=7)
            if j == 0:
                ax.set_ylabel(mc, fontsize=7)
            if i == 0:
                ax.set_title(ec, fontsize=7, fontweight="bold")

    # Legend on first axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(guid_cmap),
                   fontsize=6, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()
    path = output_dir / "scatter_matrix.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 5. Trajectory overlay (z-scored, single axis)
# ---------------------------------------------------------------------------


def plot_trajectory_overlay(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    output_dir: Path,
) -> Path:
    """Z-scored trajectory overlay for each GUID on a single y-axis.

    Both measures are z-scored (zero mean, unit variance) and plotted on
    the **same** y-axis to avoid misleading dual-axis scales.  Shaded
    region between curves highlights divergence.

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    guids = sorted(merged_df["guid"].unique())
    n_guids = len(guids)

    fig, axes = plt.subplots(n_guids, 1, figsize=(8, 2.5 * n_guids),
                             squeeze=False)
    guid_cmap = _get_guid_color_map(guids)
    time_col = "epoch" if "epoch" in merged_df.columns else "domain_start"

    for idx, guid in enumerate(guids):
        ax = axes[idx, 0]
        style_axes(ax)

        sub = merged_df[merged_df["guid"] == guid].sort_values(time_col).copy()
        if len(sub) < 2:
            ax.text(0.5, 0.5, f"GUID {_short_guid(guid)}: <2 points",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=COLOR_GRAY)
            continue

        hours = sub[time_col].values / 3600.0  # seconds → hours before delivery

        m_vals = sub[model_col].values.astype(float)
        e_vals = sub[empirical_col].values.astype(float)

        # Z-score normalisation
        m_std = np.std(m_vals)
        e_std = np.std(e_vals)
        m_z = (m_vals - np.mean(m_vals)) / m_std if m_std > 0 else np.zeros_like(m_vals)
        e_z = (e_vals - np.mean(e_vals)) / e_std if e_std > 0 else np.zeros_like(e_vals)

        ax.plot(hours, m_z, color=COLOR_BLUE, marker="o", markersize=3,
                lw=1.0, label=f"Model: {model_col} (z-scored)")
        ax.plot(hours, e_z, color=COLOR_VERMILLION, marker="s", markersize=3,
                lw=1.0, label=f"Empirical: {empirical_col} (z-scored)")

        # Shaded divergence region
        ax.fill_between(hours, m_z, e_z, alpha=0.1, color=COLOR_GRAY)

        ax.set_ylabel("Z-score", fontsize=8)
        ax.set_title(f"GUID: {_short_guid(guid)} (n={len(sub)})",
                     fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left")

        if idx == n_guids - 1:
            ax.set_xlabel("Hours before delivery", fontsize=8)

    fig.suptitle(f"Z-Scored Trajectory: {model_col} vs {empirical_col}",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / "trajectory_overlay.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 6. Trend agreement
# ---------------------------------------------------------------------------


def plot_trend_agreement(
    trend_results: Dict[str, Any],
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    output_dir: Path,
) -> Path:
    """Step plot showing sign agreement of temporal derivatives per GUID.

    Green markers for same-sign transitions, red for opposite-sign.
    Summary bar at bottom showing overall agreement rate.

    Args:
        trend_results: Output of
            :func:`~te_comparison_analysis.trend_agreement_analysis`.
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    time_col = "epoch" if "epoch" in merged_df.columns else "domain_start"

    per_guid = trend_results.get("per_guid_agreement", {})
    guids = sorted(per_guid.keys())

    if not guids:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No transitions available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
        path = output_dir / "trend_agreement.pdf"
        save_figure(fig, path)
        return path

    fig, axes = plt.subplots(len(guids) + 1, 1,
                             figsize=(8, 1.8 * (len(guids) + 1)),
                             gridspec_kw={"height_ratios": [1] * len(guids) + [0.6]},
                             squeeze=False)

    for idx, guid in enumerate(guids):
        ax = axes[idx, 0]
        style_axes(ax)

        sub = merged_df[merged_df["guid"] == guid].sort_values(time_col)
        if len(sub) < 2:
            continue

        hours = sub[time_col].values / 3600.0
        m_vals = sub[model_col].values.astype(float)
        e_vals = sub[empirical_col].values.astype(float)

        m_diff = np.diff(m_vals)
        e_diff = np.diff(e_vals)

        for t in range(len(m_diff)):
            if m_diff[t] == 0 or e_diff[t] == 0:
                color = COLOR_GRAY
                marker = "d"
            elif np.sign(m_diff[t]) == np.sign(e_diff[t]):
                color = COLOR_GREEN
                marker = "^"
            else:
                color = COLOR_VERMILLION
                marker = "v"

            mid_h = (hours[t] + hours[t + 1]) / 2
            ax.scatter(mid_h, 0, color=color, marker=marker, s=40, zorder=3,
                       edgecolors=COLOR_BLACK, linewidths=0.3)

        ax.set_xlim(hours.min() - 0.5, hours.max() + 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ag_rate = per_guid[guid]["agreement_rate"]
        ax.set_title(f"GUID {_short_guid(guid)}: {ag_rate:.0%} agreement",
                     fontsize=7, fontweight="bold")
        if idx == len(guids) - 1:
            ax.set_xlabel("Hours before delivery", fontsize=8)

    # Summary bar
    ax = axes[-1, 0]
    ax.axis("off")
    rate = trend_results.get("sign_agreement_rate", 0)
    p_val = trend_results.get("binomial_p", 1.0)
    n_tr = trend_results.get("n_transitions", 0)
    ax.text(0.5, 0.5,
            f"Overall: {rate:.1%} agreement ({trend_results.get('n_agree', 0)}/{n_tr}), "
            f"binomial p = {p_val:.3f}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_GREEN if rate > 0.5 else COLOR_VERMILLION,
                      alpha=0.15))

    fig.suptitle(f"Trend Agreement: {model_col} vs {empirical_col}",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / "trend_agreement.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 7. Leave-one-GUID-out
# ---------------------------------------------------------------------------


def plot_leave_one_out(
    sensitivity_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Bar chart showing pooled correlation with each GUID removed.

    Four bars: full dataset + one per GUID.  Highlights the most
    influential GUID.

    Args:
        sensitivity_results: Output of
            :func:`~te_comparison_analysis.leave_one_guid_out_sensitivity`.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    style_axes(ax)

    full_r = sensitivity_results.get("full_correlation", np.nan)
    leave_out = sensitivity_results.get("leave_out_correlations", {})
    method = sensitivity_results.get("method", "spearman")
    most_infl = sensitivity_results.get("most_influential_guid")

    labels = ["Full"]
    values = [full_r]
    colors = [COLOR_BLUE]

    guid_colors = _get_guid_color_map(list(leave_out.keys()))
    for guid, val in sorted(leave_out.items()):
        labels.append(f"w/o {_short_guid(guid)}")
        values.append(val)
        c = COLOR_VERMILLION if guid == most_infl else guid_colors.get(guid, COLOR_GRAY)
        colors.append(c)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor=COLOR_BLACK, linewidth=0.4)

    ax.axhline(0, color=COLOR_GRAY, lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"{method.capitalize()} ρ", fontsize=8)
    ax.set_title("Leave-One-GUID-Out Sensitivity", fontsize=9, fontweight="bold")

    # Annotate values
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * np.sign(val),
                    f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=6, color=COLOR_BLACK)

    fig.tight_layout()
    path = output_dir / "leave_one_out.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 8. Bootstrap distribution
# ---------------------------------------------------------------------------


def plot_bootstrap_distribution(
    bootstrap_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Histogram of cluster-aware bootstrap Spearman rho samples.

    Shows observed value, 95% CI shading, and null reference at zero.

    Args:
        bootstrap_results: Output of
            :func:`~te_comparison_analysis.cluster_aware_bootstrap`.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    style_axes(ax)

    samples = bootstrap_results.get("bootstrap_samples", np.array([]))
    if isinstance(samples, list):
        samples = np.array(samples)
    if len(samples) == 0:
        ax.text(0.5, 0.5, "No bootstrap samples", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
        path = output_dir / "bootstrap_distribution.pdf"
        save_figure(fig, path)
        return path

    observed = bootstrap_results.get("observed", np.nan)
    ci_lo = bootstrap_results.get("ci_lo", np.nan)
    ci_hi = bootstrap_results.get("ci_hi", np.nan)
    n_guids = bootstrap_results.get("n_guids", "?")

    ax.hist(samples, bins=30, color=COLOR_BLUE, alpha=0.6,
            edgecolor=COLOR_BLACK, linewidth=0.3)

    # CI shading
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        ax.axvspan(ci_lo, ci_hi, alpha=0.15, color=COLOR_GREEN,
                   label=f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Observed line
    if np.isfinite(observed):
        ax.axvline(observed, color=COLOR_VERMILLION, lw=1.5, ls="-",
                   label=f"Observed ρ = {observed:.3f}")

    # Null reference
    ax.axvline(0, color=COLOR_GRAY, lw=0.8, ls="--", label="Null (ρ = 0)")

    ax.set_xlabel("Spearman ρ", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(f"Cluster-Aware Bootstrap (n_guids={n_guids})",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)

    fig.tight_layout()
    path = output_dir / "bootstrap_distribution.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 9. Permutation null
# ---------------------------------------------------------------------------


def plot_permutation_null(
    permutation_results: Dict[str, Any],
    output_dir: Path,
    title: str = "Permutation Test",
) -> Path:
    """Histogram of permutation null distribution with observed statistic.

    Shades the tail beyond the observed value to visualise the p-value.

    Args:
        permutation_results: Output of
            :func:`~te_comparison_analysis.permutation_test_correlation`.
        output_dir: Directory to save the figure.
        title: Figure title.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    style_axes(ax)

    null_dist = permutation_results.get("null_distribution", np.array([]))
    if isinstance(null_dist, list):
        null_dist = np.array(null_dist)
    if len(null_dist) == 0:
        ax.text(0.5, 0.5, "No null distribution", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
        path = output_dir / "permutation_null.pdf"
        save_figure(fig, path)
        return path

    observed = permutation_results.get("observed", np.nan)
    p_value = permutation_results.get("p_value", np.nan)

    _, bins, patches = ax.hist(null_dist, bins=50, color=COLOR_BLUE, alpha=0.5,
                                edgecolor=COLOR_BLACK, linewidth=0.2)

    # Shade tails beyond |observed|
    if np.isfinite(observed):
        for patch, left_edge in zip(patches, bins[:-1]):
            right_edge = left_edge + (bins[1] - bins[0])
            if abs(left_edge) >= abs(observed) or abs(right_edge) >= abs(observed):
                patch.set_facecolor(COLOR_VERMILLION)
                patch.set_alpha(0.7)

        ax.axvline(observed, color=COLOR_VERMILLION, lw=1.5, ls="-",
                   label=f"Observed = {observed:.3f}")
        ax.axvline(-observed, color=COLOR_VERMILLION, lw=1.0, ls=":",
                   alpha=0.5)

    ax.set_xlabel("Correlation under null", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    p_str = f"p = {p_value:.4f}" if np.isfinite(p_value) else "p = N/A"
    ax.set_title(f"{title} ({p_str})", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6)

    fig.tight_layout()
    path = output_dir / "permutation_null.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 10. Mutual information comparison
# ---------------------------------------------------------------------------


def plot_mutual_information_comparison(
    mi_results: Dict[str, float],
    corr_results: Dict[str, float],
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing MI estimates with Spearman rho.

    Highlights cases where MI detects associations that linear correlation
    misses (non-linear coupling).

    Args:
        mi_results: Dict mapping ``"model_vs_empirical"`` keys to MI values.
        corr_results: Dict mapping same keys to Spearman rho values.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()

    keys = sorted(set(mi_results.keys()) & set(corr_results.keys()))
    if not keys:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
        path = output_dir / "mi_vs_correlation.pdf"
        save_figure(fig, path)
        return path

    mi_vals = [mi_results[k] for k in keys]
    corr_vals = [abs(corr_results[k]) if np.isfinite(corr_results[k]) else 0 for k in keys]

    # Normalise MI to [0, 1] range for visual comparison
    mi_max = max(mi_vals) if max(mi_vals) > 0 else 1.0
    mi_norm = [v / mi_max for v in mi_vals]

    # Short labels
    labels = [k.replace("_vs_", " ↔\n").replace("_", " ") for k in keys]

    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.2), 4))
    style_axes(ax)

    ax.bar(x - w / 2, corr_vals, w, label="|Spearman ρ|", color=COLOR_BLUE,
           edgecolor=COLOR_BLACK, linewidth=0.3)
    ax.bar(x + w / 2, mi_norm, w, label="MI (normalised)", color=COLOR_GREEN,
           edgecolor=COLOR_BLACK, linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Value (normalised)", fontsize=8)
    ax.set_title("Mutual Information vs Spearman Correlation", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = output_dir / "mi_vs_correlation.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 11. DTW trajectories
# ---------------------------------------------------------------------------


def plot_dtw_trajectories(
    dtw_results: Dict[str, Any],
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    output_dir: Path,
) -> Path:
    """Z-scored trajectories per GUID annotated with DTW distance.

    Args:
        dtw_results: Output of
            :func:`~te_comparison_analysis.dtw_trajectory_similarity`.
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()
    time_col = "epoch" if "epoch" in merged_df.columns else "domain_start"
    guids = sorted(merged_df["guid"].unique())
    per_guid = dtw_results.get("per_guid", {})
    method = dtw_results.get("method", "euclidean")

    fig, axes = plt.subplots(len(guids), 1, figsize=(8, 2.5 * len(guids)),
                             squeeze=False)

    for idx, guid in enumerate(guids):
        ax = axes[idx, 0]
        style_axes(ax)

        sub = merged_df[merged_df["guid"] == guid].sort_values(time_col)
        if len(sub) < 2:
            ax.text(0.5, 0.5, f"GUID {_short_guid(guid)}: <2 points",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=COLOR_GRAY)
            continue

        hours = sub[time_col].values / 3600.0
        m_vals = sub[model_col].values.astype(float)
        e_vals = sub[empirical_col].values.astype(float)

        m_std = np.std(m_vals)
        e_std = np.std(e_vals)
        m_z = (m_vals - np.mean(m_vals)) / m_std if m_std > 0 else np.zeros_like(m_vals)
        e_z = (e_vals - np.mean(e_vals)) / e_std if e_std > 0 else np.zeros_like(e_vals)

        ax.plot(hours, m_z, color=COLOR_BLUE, marker="o", markersize=3,
                lw=1.0, label="Model (z)")
        ax.plot(hours, e_z, color=COLOR_VERMILLION, marker="s", markersize=3,
                lw=1.0, label="Empirical (z)")

        # DTW distance annotation
        guid_dtw = per_guid.get(guid, {})
        dist = guid_dtw.get("distance", np.nan)
        norm_dist = guid_dtw.get("normalized", np.nan)
        dist_str = f"{method.upper()} dist: {dist:.2f}" if np.isfinite(dist) else "N/A"
        norm_str = f"(norm: {norm_dist:.3f})" if np.isfinite(norm_dist) else ""

        ax.text(0.98, 0.95, f"{dist_str} {norm_str}",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=COLOR_GRAY,
                          linewidth=0.3))

        ax.set_ylabel("Z-score", fontsize=8)
        ax.set_title(f"GUID: {_short_guid(guid)} (n={len(sub)})",
                     fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left")

        if idx == len(guids) - 1:
            ax.set_xlabel("Hours before delivery", fontsize=8)

    fig.suptitle(f"DTW Trajectory Comparison: {model_col} vs {empirical_col}",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = output_dir / "dtw_trajectories.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 12. Summary table
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 13. Model GUID trajectories (standalone, no matching required)
# ---------------------------------------------------------------------------


def _select_diverse_guids(
    df: pd.DataFrame,
    n: int = 10,
    class_col: str = "class_label",
    guid_col: str = "guid",
    count_col: str = "epoch",
) -> List[str]:
    """Select *n* GUIDs balanced across classes, preferring data-rich ones.

    Args:
        df: DataFrame with guid and class columns.
        n: Number of GUIDs to select.
        class_col: Column containing class labels.
        guid_col: Column containing GUID identifiers.
        count_col: Column to count for richness.

    Returns:
        List of selected GUID strings.
    """
    guid_counts = (
        df.groupby([guid_col, class_col])[count_col]
        .count()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    selected: List[str] = []
    if class_col in df.columns:
        classes = sorted(df[class_col].unique())
        per_class = max(1, n // len(classes))
        remainder = n - per_class * len(classes)

        for cls in classes:
            cls_guids = guid_counts[guid_counts[class_col] == cls][guid_col].tolist()
            take = per_class + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
            for g in cls_guids:
                if g not in selected and len([s for s in selected if guid_counts[guid_counts[guid_col] == s][class_col].values[0] == cls]) < take:
                    selected.append(g)
                if len(selected) >= n:
                    break
    else:
        selected = guid_counts[guid_col].head(n).tolist()

    return selected[:n]


def plot_model_guid_trajectories(
    model_df: pd.DataFrame,
    output_dir: Path,
    empirical_df: Optional[pd.DataFrame] = None,
    common_guids: Optional[List[str]] = None,
    n_guids: int = 10,
    model_col: str = "kl_mean",
    empirical_col: str = "ite_valid",
) -> Path:
    """Plot model KL trajectories for diverse GUIDs, with empirical overlay.

    For each of 10 GUIDs, shows the full ``kl_mean`` trajectory.  For
    overlap GUIDs (those also in the empirical data), a z-scored overlay
    of model and empirical TE is shown on a twin axis so the reader can
    compare temporal dynamics.

    Args:
        model_df: Full model TE DataFrame (all GUIDs).
        output_dir: Directory to save the figure.
        empirical_df: Full empirical TE DataFrame (for overlay on common
            GUIDs).  If *None*, no overlay is drawn.
        common_guids: GUIDs present in both datasets.
        n_guids: Number of GUIDs to plot.
        model_col: Model measure column.
        empirical_col: Empirical measure column for overlay.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()

    if common_guids is None:
        common_guids = []
    common_set = set(common_guids)

    # Select diverse GUIDs, ensuring common GUIDs are included first
    selected = []
    for g in common_guids:
        if g in model_df["guid"].values:
            selected.append(g)
    remaining_n = n_guids - len(selected)

    if remaining_n > 0:
        candidates = _select_diverse_guids(
            model_df[~model_df["guid"].isin(selected)],
            n=remaining_n,
        )
        selected.extend(candidates)
    selected = selected[:n_guids]

    from model.transformer.tr_testing.style import get_class_colors
    classes_present = model_df[model_df["guid"].isin(selected)]["class_label"].unique()
    class_cmap = get_class_colors(classes_present)

    n_rows, n_cols = 5, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.0 * n_rows),
                             squeeze=False)

    for idx, guid in enumerate(selected):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        style_axes(ax)

        sub = model_df[model_df["guid"] == guid].sort_values("epoch")
        hours = sub["epoch"].values / 3600.0
        y_main = sub[model_col].values.astype(float)
        cls = sub["class_label"].iloc[0] if "class_label" in sub.columns else "unknown"
        color = class_cmap.get(cls, COLOR_GRAY)

        is_overlap = guid in common_set

        if is_overlap and empirical_df is not None:
            # Z-scored overlay: both measures on the same scale
            e_sub = empirical_df[empirical_df["guid"] == guid].sort_values("domain_start")
            e_hours = e_sub["domain_start"].values / 3600.0
            e_vals = e_sub[empirical_col].values.astype(float)

            m_std = np.std(y_main)
            e_std = np.std(e_vals)
            m_z = (y_main - np.mean(y_main)) / m_std if m_std > 0 else np.zeros_like(y_main)
            e_z = (e_vals - np.mean(e_vals)) / e_std if e_std > 0 else np.zeros_like(e_vals)

            ax.plot(hours, m_z, color=color, marker="*", markersize=5,
                    lw=1.2, label=f"Model {model_col} (z)", zorder=3)
            ax.plot(e_hours, e_z, color=COLOR_VERMILLION, marker="s",
                    markersize=2.5, lw=0.9,
                    label=f"Empirical {empirical_col} (z)", zorder=2)
            ax.set_ylabel("Z-score", fontsize=7)

            overlap_tag = f" [OVERLAP: model={len(sub)}, emp={len(e_sub)}]"
        else:
            ax.plot(hours, y_main, color=color, marker="o", markersize=3,
                    lw=1.2, label=model_col, zorder=3)
            ax.set_ylabel(model_col, fontsize=7)
            overlap_tag = ""

        ax.set_title(f"{_short_guid(guid)}  ({cls}, n={len(sub)}){overlap_tag}",
                     fontsize=7, fontweight="bold", color=color)

        if row == n_rows - 1:
            ax.set_xlabel("Hours before delivery", fontsize=7)

        ax.legend(fontsize=5, loc="upper right")

    for idx in range(len(selected), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("Model TE Trajectories (with empirical overlay for overlap GUIDs)",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = output_dir / "model_guid_trajectories.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 14. Empirical GUID trajectories (with model overlay for common GUIDs)
# ---------------------------------------------------------------------------


def plot_empirical_guid_trajectories(
    empirical_df: pd.DataFrame,
    output_dir: Path,
    model_df: Optional[pd.DataFrame] = None,
    common_guids: Optional[List[str]] = None,
    n_guids: int = 10,
    empirical_col: str = "ite_valid",
    model_col: str = "kl_mean",
) -> Path:
    """Plot empirical TE trajectories for data-rich GUIDs, with model overlay.

    For each of 10 GUIDs, shows the full ``ite_valid`` trajectory.  For
    overlap GUIDs, a z-scored overlay of both model and empirical TE is
    drawn so temporal dynamics can be compared visually.

    Args:
        empirical_df: Full empirical TE DataFrame (all GUIDs).
        output_dir: Directory to save the figure.
        model_df: Full model TE DataFrame (for overlay on common GUIDs).
        common_guids: GUIDs present in both datasets.
        n_guids: Number of GUIDs to plot.
        empirical_col: Empirical measure column.
        model_col: Model measure column for overlay.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()

    if common_guids is None:
        common_guids = []
    common_set = set(common_guids)

    # Common GUIDs first, then richest by epoch count
    selected = []
    for g in common_guids:
        if g in empirical_df["guid"].values:
            selected.append(g)
    remaining_n = n_guids - len(selected)

    if remaining_n > 0:
        richest = (
            empirical_df[~empirical_df["guid"].isin(selected)]
            .groupby("guid")["domain_start"]
            .count()
            .sort_values(ascending=False)
            .head(remaining_n)
            .index.tolist()
        )
        selected.extend(richest)
    selected = selected[:n_guids]

    n_rows, n_cols = 5, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.0 * n_rows),
                             squeeze=False)

    for idx, guid in enumerate(selected):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        style_axes(ax)

        sub = empirical_df[empirical_df["guid"] == guid].sort_values("domain_start")
        hours = sub["domain_start"].values / 3600.0
        y_main = sub[empirical_col].values.astype(float)
        is_overlap = guid in common_set

        if is_overlap and model_df is not None:
            m_sub = model_df[model_df["guid"] == guid].sort_values("epoch")
            m_hours = m_sub["epoch"].values / 3600.0
            m_vals = m_sub["kl_mean"].values.astype(float)

            e_std = np.std(y_main)
            m_std = np.std(m_vals)
            e_z = (y_main - np.mean(y_main)) / e_std if e_std > 0 else np.zeros_like(y_main)
            m_z = (m_vals - np.mean(m_vals)) / m_std if m_std > 0 else np.zeros_like(m_vals)

            ax.plot(hours, e_z, color=COLOR_VERMILLION, marker="*",
                    markersize=5, lw=1.0,
                    label=f"Empirical {empirical_col} (z)", zorder=3)
            ax.plot(m_hours, m_z, color=COLOR_BLUE, marker="s",
                    markersize=2.5, lw=0.9,
                    label=f"Model {model_col} (z)", zorder=2)
            ax.set_ylabel("Z-score", fontsize=7)

            overlap_tag = f" [OVERLAP: emp={len(sub)}, model={len(m_sub)}]"
            title_color = COLOR_VERMILLION
        else:
            ax.plot(hours, y_main, color=COLOR_SKY, marker="o",
                    markersize=2, lw=1.0, label=empirical_col, zorder=3)
            ax.set_ylabel(empirical_col, fontsize=7)
            overlap_tag = ""
            title_color = COLOR_BLACK

        ax.set_title(f"{_short_guid(guid)}  (n={len(sub)}){overlap_tag}",
                     fontsize=7, fontweight="bold", color=title_color)

        if row == n_rows - 1:
            ax.set_xlabel("Hours before delivery", fontsize=7)

        ax.legend(fontsize=5, loc="upper right")

    for idx in range(len(selected), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle("Empirical TE Trajectories (with model overlay for overlap GUIDs)",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    path = output_dir / "empirical_guid_trajectories.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 15. Summary table
# ---------------------------------------------------------------------------


def generate_summary_table(
    all_results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Publication-ready summary table rendered as a matplotlib figure.

    Rows = primary measure pairs.  Columns = Spearman rho, Kendall tau,
    permutation p, MI, trend agreement %, concordance index.

    Args:
        all_results: Comprehensive results dict from
            :func:`~te_comparison_analysis.run_full_comparison`.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    apply_publication_style()

    # Build table rows from correlation matrices
    corr_sp = all_results.get("correlation_spearman")
    corr_ke = all_results.get("correlation_kendall")
    perm_tests = all_results.get("permutation_tests", {})
    mi = all_results.get("mutual_information", {})
    concordance = all_results.get("concordance", {})
    trend = all_results.get("trend_agreement", {})

    if corr_sp is None:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No results available", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color=COLOR_GRAY)
        path = output_dir / "summary_table.pdf"
        save_figure(fig, path)
        return path

    rows = []
    col_headers = ["Measure Pair", "Spearman ρ", "Kendall τ",
                    "Perm. p", "MI", "Trend %", "C-index"]

    for mc in corr_sp.index:
        for ec in corr_sp.columns:
            key = f"{mc}_vs_{ec}"
            sp_val = corr_sp.loc[mc, ec]
            ke_val = corr_ke.loc[mc, ec] if corr_ke is not None else np.nan
            perm_p = perm_tests.get(key, {}).get("p_value", np.nan)
            mi_val = mi.get(key, np.nan)

            row = [
                f"{mc} ↔ {ec}",
                f"{sp_val:.3f}" if np.isfinite(sp_val) else "—",
                f"{ke_val:.3f}" if np.isfinite(ke_val) else "—",
                f"{perm_p:.4f}" if np.isfinite(perm_p) else "—",
                f"{mi_val:.4f}" if np.isfinite(mi_val) else "—",
                "",  # trend filled for primary only
                "",  # c-index filled for primary only
            ]
            rows.append(row)

    # Fill trend + concordance for primary pair (first row)
    if rows:
        tr_rate = trend.get("sign_agreement_rate", np.nan)
        ci_val = concordance.get("concordance_index", np.nan)
        rows[0][5] = f"{tr_rate:.1%}" if np.isfinite(tr_rate) else "—"
        rows[0][6] = f"{ci_val:.3f}" if np.isfinite(ci_val) else "—"

    fig_height = max(2, 0.35 * len(rows) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.auto_set_column_width(list(range(len(col_headers))))
    table.scale(1.0, 1.3)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLOR_LIGHT_GRAY)
        if row == 0:
            cell.set_facecolor(COLOR_BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8F9FA")
        else:
            cell.set_facecolor("white")

    ax.set_title("Model TE vs Empirical TE: Summary Statistics",
                 fontsize=10, fontweight="bold", pad=15)

    fig.tight_layout()
    path = output_dir / "summary_table.pdf"
    save_figure(fig, path)
    logger.info(f"Saved: {path.name}")
    return path
