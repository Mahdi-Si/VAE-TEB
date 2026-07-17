"""Publication-quality evaluation plots for the transformer classifier.

Uses the shared style from ``model/transformer/tr_testing/style.py`` to
produce 600-DPI publication-grade figures with Times New Roman, thin
lines, and the project colour palette.

All plot functions follow the same pattern:
    1. ``apply_publication_style()`` sets global rcParams.
    2. Figure is created with publication-appropriate sizing.
    3. ``style_axes(ax)`` applies consistent spine, grid, tick styling.
    4. ``save_figure(fig, path)`` saves at 600 DPI and closes.

Example::

    from model.transformer.classification.evaluation_plots import (
        plot_metric_curves,
    )
    plot_metric_curves(metrics_df, "committed_overall", output_dir)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from model.transformer.tr_testing.style import (
    apply_publication_style,
    style_axes,
    save_figure,
    get_class_colors,
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


# ====================================================================== #
#  Constants                                                               #
# ====================================================================== #

METRIC_COLORS: Dict[str, str] = {
    "sensitivity": COLOR_GREEN,
    "specificity": COLOR_BLUE,
    "fpr": COLOR_VERMILLION,
}

METRIC_MARKERS: Dict[str, str] = {
    "sensitivity": "o",
    "specificity": "s",
    "fpr": "^",
}

METRIC_LABELS: Dict[str, str] = {
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "fpr": "False Positive Rate",
}

METRIC_TYPE_LABELS: Dict[str, str] = {
    "instantaneous": "Instantaneous",
    "committed_cumulative": "Committed Cumulative",
    "committed_overall": "Committed Overall",
}

DIAGNOSIS_COLORS: Dict[str, str] = {
    "healthy": COLOR_BLUE,
    "acidosis": COLOR_ORANGE,
    "hie": COLOR_VERMILLION,
    "unhealthy": COLOR_PURPLE,
}

# Subgroup stratification colours.
CS_COLORS = {"pos": COLOR_SKY, "neg": COLOR_SAGE}
BG_COLORS = {"pos": COLOR_ORANGE, "neg": COLOR_PURPLE}

# Dynamic marker list for subgroups.
SUBGROUP_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


# ====================================================================== #
#  Metric Curve Plots (5 variants)                                         #
# ====================================================================== #


def plot_metric_curves(
    metrics_df: pd.DataFrame,
    metric_type: str,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Generate 5 metric-vs-time plot variants for one metric type.

    Produces:
        - ``sensitivity_vs_time.png``
        - ``sensitivity_specificity_vs_time.png``
        - ``sensitivity_fpr_vs_time.png``
        - ``all_metrics_vs_time.png``
        - ``fpr_vs_time.png``

    Args:
        metrics_df: DataFrame with columns ``bin_center``,
            ``sensitivity``, ``specificity``, ``fpr``.
        metric_type: One of ``"instantaneous"``,
            ``"committed_cumulative"``, ``"committed_overall"``.
        output_dir: Directory to save plots.
        title_suffix: Optional suffix for plot titles.
    """
    if not HAS_MPL:
        logger.warning("Matplotlib not available — skipping metric plots")
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    df = metrics_df.dropna(subset=["bin_center"]).sort_values(
        "bin_center", ascending=False
    )
    if df.empty:
        logger.warning("Empty metrics DataFrame — skipping plots")
        return

    x = df["bin_center"].values
    mt_label = METRIC_TYPE_LABELS.get(metric_type, metric_type)

    # Variant definitions: (filename, metrics_to_plot, ylabel).
    variants = [
        ("sensitivity_vs_time", ["sensitivity"], "Sensitivity"),
        ("sensitivity_specificity_vs_time", ["sensitivity", "specificity"],
         "Value"),
        ("sensitivity_fpr_vs_time", ["sensitivity", "fpr"], "Value"),
        ("all_metrics_vs_time", ["sensitivity", "specificity", "fpr"],
         "Value"),
        ("fpr_vs_time", ["fpr"], "False Positive Rate"),
    ]

    for fname, metrics, ylabel in variants:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        for m in metrics:
            if m not in df.columns:
                continue
            y = df[m].values
            valid = ~np.isnan(y)
            ax.plot(
                x[valid], y[valid],
                color=METRIC_COLORS[m],
                marker=METRIC_MARKERS[m],
                label=METRIC_LABELS[m],
            )

        ax.set_xlabel("Hours before birth")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.invert_xaxis()
        ax.legend(loc="best")
        title = f"{mt_label}"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        style_axes(ax)

        save_figure(fig, output_dir / f"{fname}.png")


# ====================================================================== #
#  Metric Type Comparison                                                  #
# ====================================================================== #


def plot_metric_comparison(
    metrics_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Compare all three metric types in a 1x3 panel.

    Each panel shows sensitivity and FPR for one metric type.

    Args:
        metrics_dict: Dict mapping metric type name to DataFrame.
        output_dir: Directory to save the comparison plot.
        title_suffix: Optional suffix for the suptitle.
    """
    if not HAS_MPL:
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    types = ["instantaneous", "committed_cumulative", "committed_overall"]
    available = [t for t in types if t in metrics_dict]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, mt in zip(axes, available):
        df = metrics_dict[mt].dropna(subset=["bin_center"]).sort_values(
            "bin_center", ascending=False
        )
        if df.empty:
            continue
        x = df["bin_center"].values
        for m in ("sensitivity", "fpr"):
            if m not in df.columns:
                continue
            y = df[m].values
            valid = ~np.isnan(y)
            ax.plot(
                x[valid], y[valid],
                color=METRIC_COLORS[m],
                marker=METRIC_MARKERS[m],
                label=METRIC_LABELS[m],
            )
        ax.set_title(METRIC_TYPE_LABELS.get(mt, mt), fontsize=8)
        ax.set_xlabel("Hours before birth", fontsize=7)
        ax.set_ylim(-0.02, 1.05)
        ax.invert_xaxis()
        ax.legend(loc="best", fontsize=6)
        style_axes(ax)

    axes[0].set_ylabel("Value")
    suptitle = "Metric Type Comparison"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle, fontsize=9, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir / "metric_type_comparison.png")


# ====================================================================== #
#  Subgroup Analysis Plots                                                 #
# ====================================================================== #


def plot_subgroup_analysis(
    subgroup_metrics: Dict[str, pd.DataFrame],
    metric_type: str,
    output_dir: Path,
    title_suffix: str = "",
    subgroup_guid_counts: Optional[Dict[str, int]] = None,
) -> None:
    """Generate subgroup comparison plots for one metric type.

    Produces:
        - ``diagnosis_comparison.png``: healthy vs acidosis vs HIE
        - ``cs_stratification.png``: CS+ vs CS- per diagnosis
        - ``bg_stratification.png``: BG+ vs BG- for acidosis
        - ``healthy_subgroups.png``: healthy CS/BG combinations

    Args:
        subgroup_metrics: Dict mapping subgroup name to metrics
            DataFrame (with ``bin_center``, ``sensitivity``).
        metric_type: Metric type label for titles.
        output_dir: Directory to save plots.
        title_suffix: Optional suffix for titles.
        subgroup_guid_counts: Optional dict of GUID counts per
            subgroup for legend annotations.
    """
    if not HAS_MPL:
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)
    counts = subgroup_guid_counts or {}
    mt_label = METRIC_TYPE_LABELS.get(metric_type, metric_type)

    def _make_label(name: str) -> str:
        n = counts.get(name)
        return f"{name} (n={n})" if n is not None else name

    # --- Diagnosis comparison ---
    _plot_subgroup_set(
        subgroup_metrics,
        names=["healthy", "acidosis", "hie", "unhealthy"],
        colors=DIAGNOSIS_COLORS,
        metric_col="sensitivity",
        ylabel="Sensitivity",
        title=f"Diagnosis Comparison — {mt_label}",
        title_suffix=title_suffix,
        output_path=output_dir / "diagnosis_comparison.png",
        label_fn=_make_label,
    )

    # --- CS stratification ---
    cs_names = [
        k for k in subgroup_metrics
        if "cs_pos" in k or "cs_neg" in k
    ]
    if cs_names:
        _plot_subgroup_set(
            subgroup_metrics,
            names=cs_names,
            colors=None,
            metric_col="sensitivity",
            ylabel="Sensitivity",
            title=f"CS Stratification — {mt_label}",
            title_suffix=title_suffix,
            output_path=output_dir / "cs_stratification.png",
            label_fn=_make_label,
        )

    # --- BG stratification ---
    bg_names = [
        k for k in subgroup_metrics
        if "bg_pos" in k or "bg_neg" in k
    ]
    if bg_names:
        _plot_subgroup_set(
            subgroup_metrics,
            names=bg_names,
            colors=None,
            metric_col="sensitivity",
            ylabel="Sensitivity",
            title=f"BG Stratification — {mt_label}",
            title_suffix=title_suffix,
            output_path=output_dir / "bg_stratification.png",
            label_fn=_make_label,
        )

    # --- Healthy subgroups (specificity) ---
    healthy_names = [
        k for k in subgroup_metrics
        if k.startswith("healthy_")
    ]
    if healthy_names:
        _plot_subgroup_set(
            subgroup_metrics,
            names=healthy_names,
            colors=None,
            metric_col="specificity",
            ylabel="Specificity",
            title=f"Healthy Subgroups — {mt_label}",
            title_suffix=title_suffix,
            output_path=output_dir / "healthy_subgroups.png",
            label_fn=_make_label,
        )


def _plot_subgroup_set(
    subgroup_metrics: Dict[str, pd.DataFrame],
    names: List[str],
    colors: Optional[Dict[str, str]],
    metric_col: str,
    ylabel: str,
    title: str,
    title_suffix: str,
    output_path: Path,
    label_fn=None,
) -> None:
    """Plot overlaid subgroup curves on a single axis."""
    available = [n for n in names if n in subgroup_metrics]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    if colors is None:
        palette = get_class_colors(available)
    else:
        palette = colors

    for idx, name in enumerate(available):
        df = subgroup_metrics[name]
        if df.empty or "bin_center" not in df.columns:
            continue
        df = df.dropna(subset=["bin_center"]).sort_values(
            "bin_center", ascending=False
        )
        if metric_col not in df.columns:
            continue

        x = df["bin_center"].values
        y = df[metric_col].values
        valid = ~np.isnan(y)

        color = palette.get(name, COLOR_GRAY)
        marker = SUBGROUP_MARKERS[idx % len(SUBGROUP_MARKERS)]
        label = label_fn(name) if label_fn else name

        ax.plot(
            x[valid], y[valid],
            color=color,
            marker=marker,
            label=label,
        )

    ax.set_xlabel("Hours before birth")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.05)
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=6)
    full_title = title
    if title_suffix:
        full_title += f" — {title_suffix}"
    ax.set_title(full_title)
    style_axes(ax)
    save_figure(fig, output_path)


# ====================================================================== #
#  ROC Curves                                                              #
# ====================================================================== #


def plot_roc_curve(
    roc_data: Dict,
    output_path: Path,
    title_suffix: str = "",
    threshold: Optional[float] = None,
) -> None:
    """Plot a single ROC curve with AUC annotation.

    Args:
        roc_data: Dict with ``fpr`` (list), ``tpr`` (list), ``auc``
            (float).
        output_path: File path to save the plot.
        title_suffix: Optional suffix for the title.
        threshold: If provided, marks the operating point.
    """
    if not HAS_MPL:
        return

    apply_publication_style()

    fpr = np.array(roc_data.get("fpr", []))
    tpr = np.array(roc_data.get("tpr", []))
    auc_val = roc_data.get("auc", 0.0)

    if len(fpr) == 0:
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Diagonal reference.
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_LIGHT_GRAY,
            linewidth=0.6)

    # ROC curve.
    ax.plot(fpr, tpr, color=COLOR_BLUE,
            label=f"AUC = {auc_val:.3f}")

    # Operating point.
    if threshold is not None and "thresholds" in roc_data:
        thresholds = np.array(roc_data["thresholds"])
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.plot(fpr[idx], tpr[idx], marker="o", color=COLOR_VERMILLION,
                markersize=5, zorder=5,
                label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6)

    title = "ROC Curve"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    style_axes(ax)

    os.makedirs(output_path.parent, exist_ok=True)
    save_figure(fig, output_path)


def plot_aggregated_roc(
    all_roc_data: List[Dict],
    output_dir: Path,
    n_folds: int,
    title_suffix: str = "",
) -> None:
    """Overlay per-fold ROC curves with mean +/- std band.

    Args:
        all_roc_data: List of per-fold ROC dicts (each with
            ``fpr``, ``tpr``, ``auc``).
        output_dir: Directory to save the plot.
        n_folds: Total number of folds (for title annotation).
        title_suffix: Optional suffix for the title.
    """
    if not HAS_MPL or not all_roc_data:
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Diagonal.
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_LIGHT_GRAY,
            linewidth=0.6)

    # Interpolate all folds onto common FPR grid.
    mean_fpr = np.linspace(0, 1, 200)
    tpr_interp = []
    aucs = []

    for fold_roc in all_roc_data:
        fpr = np.array(fold_roc.get("fpr", []))
        tpr = np.array(fold_roc.get("tpr", []))
        auc_val = fold_roc.get("auc", 0.0)
        if len(fpr) < 2:
            continue

        # Per-fold curve (thin, transparent).
        ax.plot(fpr, tpr, color=COLOR_BLUE, alpha=0.2, linewidth=0.4)

        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tpr_interp.append(interp)
        aucs.append(auc_val)

    if tpr_interp:
        tpr_arr = np.array(tpr_interp)
        mean_tpr = tpr_arr.mean(axis=0)
        std_tpr = tpr_arr.std(axis=0)
        mean_tpr[-1] = 1.0

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr, mean_tpr,
            color=COLOR_BLUE, linewidth=1.2,
            label=f"Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}",
        )
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=COLOR_BLUE, alpha=0.15,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6)

    title = f"Aggregated ROC ({n_folds} folds)"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    style_axes(ax)
    save_figure(fig, output_dir / "aggregated_roc_curves.png")


# ====================================================================== #
#  Dataset Statistics Plots                                                #
# ====================================================================== #


def plot_dataset_statistics(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """Generate dataset overview and composition plots.

    Produces:
        - ``dataset_overview.png``: 2x2 overview grid
        - ``epochs_per_time_bin.png``: stacked bar chart by diagnosis

    Args:
        df: Predictions DataFrame with ``guid``, ``epoch``,
            ``binary_target``, ``target``, ``cs_label``, ``bg_label``.
        time_bins: Time bin edges from ``compute_time_bins``.
        output_dir: Directory to save plots.
        title_suffix: Optional suffix for titles.
    """
    if not HAS_MPL:
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # --- Dataset overview (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))

    # Panel 1: Epochs per GUID histogram.
    ax = axes[0, 0]
    epochs_per_guid = df.groupby("guid").size()
    ax.hist(epochs_per_guid.values, bins=30, color=COLOR_BLUE, alpha=0.8,
            edgecolor=COLOR_BLACK, linewidth=0.3)
    ax.set_xlabel("Segments per GUID")
    ax.set_ylabel("Count")
    ax.set_title("Segments per GUID")
    style_axes(ax)

    # Panel 2: Epoch time distribution.
    ax = axes[0, 1]
    if "epoch" in df.columns:
        epoch_hours = -df["epoch"].values / 3600.0
        ax.hist(epoch_hours, bins=50, color=COLOR_SKY, alpha=0.8,
                edgecolor=COLOR_BLACK, linewidth=0.3)
        ax.set_xlabel("Hours before birth")
        ax.set_ylabel("Count")
        ax.set_title("Epoch Distribution")
    style_axes(ax)

    # Panel 3: Class distribution.
    ax = axes[1, 0]
    if "binary_target" in df.columns:
        guid_labels = df.groupby("guid")["binary_target"].max()
        counts = guid_labels.value_counts().sort_index()
        labels = ["Healthy", "Unhealthy"]
        colors = [COLOR_BLUE, COLOR_VERMILLION]
        bars = ax.bar(
            range(len(counts)), counts.values,
            color=colors[:len(counts)],
            edgecolor=COLOR_BLACK, linewidth=0.3,
        )
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels[:len(counts)])
        ax.set_ylabel("GUIDs")
        ax.set_title("Class Distribution")
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=7,
            )
    style_axes(ax)

    # Panel 4: Summary text.
    ax = axes[1, 1]
    ax.axis("off")
    n_guids = df["guid"].nunique()
    n_epochs = len(df)
    n_healthy = df.groupby("guid")["binary_target"].max().eq(0).sum()
    n_unhealthy = n_guids - n_healthy
    summary = (
        f"Total GUIDs: {n_guids}\n"
        f"Total segments: {n_epochs}\n"
        f"Healthy GUIDs: {n_healthy}\n"
        f"Unhealthy GUIDs: {n_unhealthy}\n"
        f"Imbalance ratio: {n_healthy / max(n_unhealthy, 1):.1f}:1"
    )
    ax.text(
        0.1, 0.5, summary,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor=COLOR_LIGHT_GRAY, alpha=0.5),
    )

    suptitle = "Dataset Overview"
    if title_suffix:
        suptitle += f" — {title_suffix}"
    fig.suptitle(suptitle, fontsize=9, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir / "dataset_overview.png")

    # --- Epochs per time bin (stacked) ---
    if time_bins is not None and len(time_bins) > 1:
        _plot_epochs_per_time_bin(df, time_bins, output_dir, title_suffix)


def _plot_epochs_per_time_bin(
    df: pd.DataFrame,
    time_bins: np.ndarray,
    output_dir: Path,
    title_suffix: str,
) -> None:
    """Stacked bar chart of epochs per time bin, coloured by diagnosis."""
    if "epoch" not in df.columns or "target" not in df.columns:
        return

    epoch_hours = -df["epoch"].values / 3600.0
    bin_centres = (time_bins[:-1] + time_bins[1:]) / 2
    bin_width = np.median(np.diff(time_bins)) * 0.85

    # Bin assignment.
    bin_idx = np.digitize(epoch_hours, time_bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_centres) - 1)

    # Count per diagnosis per bin.
    diag_map = {1: "Healthy", 2: "Acidosis", 3: "HIE"}
    diag_colors = {
        "Healthy": COLOR_BLUE,
        "Acidosis": COLOR_ORANGE,
        "HIE": COLOR_VERMILLION,
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bottom = np.zeros(len(bin_centres))

    for target_val, label in sorted(diag_map.items()):
        mask = df["target"].values == target_val
        counts = np.zeros(len(bin_centres))
        for i, m in enumerate(mask):
            if m:
                counts[bin_idx[i]] += 1
        ax.bar(
            bin_centres, counts, width=bin_width,
            bottom=bottom, color=diag_colors.get(label, COLOR_GRAY),
            edgecolor=COLOR_BLACK, linewidth=0.3,
            label=label, alpha=0.85,
        )
        bottom += counts

    ax.set_xlabel("Hours before birth")
    ax.set_ylabel("Segments")
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=6)
    title = "Segments per Time Bin"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    style_axes(ax)
    save_figure(fig, output_dir / "epochs_per_time_bin.png")


# ====================================================================== #
#  Aggregated Metric Plots (Cross-Fold)                                    #
# ====================================================================== #


def plot_aggregated_metrics(
    metric_type: str,
    all_fold_dfs: List[pd.DataFrame],
    output_dir: Path,
    n_folds: int,
    title_suffix: str = "",
) -> None:
    """Plot aggregated (mean + min/max band) metrics across folds.

    Generates the same 5 variants as ``plot_metric_curves`` but with
    cross-fold aggregation (mean line with min/max shaded band).

    Args:
        metric_type: Metric type label.
        all_fold_dfs: List of per-fold metrics DataFrames.
        output_dir: Directory to save plots.
        n_folds: Total fold count for titles.
        title_suffix: Optional suffix.
    """
    if not HAS_MPL or not all_fold_dfs:
        return

    apply_publication_style()
    os.makedirs(output_dir, exist_ok=True)

    # Interpolate all folds onto a common time grid.
    all_x = np.concatenate(
        [df["bin_center"].dropna().values for df in all_fold_dfs]
    )
    if len(all_x) == 0:
        return
    x_grid = np.sort(np.unique(all_x))[::-1]  # Descending (far→near)

    mt_label = METRIC_TYPE_LABELS.get(metric_type, metric_type)

    variants = [
        ("sensitivity_vs_time_aggregated", ["sensitivity"], "Sensitivity"),
        ("sensitivity_specificity_vs_time_aggregated",
         ["sensitivity", "specificity"], "Value"),
        ("sensitivity_fpr_vs_time_aggregated",
         ["sensitivity", "fpr"], "Value"),
        ("all_metrics_vs_time_aggregated",
         ["sensitivity", "specificity", "fpr"], "Value"),
        ("fpr_vs_time_aggregated", ["fpr"], "False Positive Rate"),
    ]

    for fname, metrics, ylabel in variants:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))

        for m in metrics:
            # Collect per-fold curves interpolated onto common grid.
            fold_curves = []
            for fold_df in all_fold_dfs:
                fdf = fold_df.dropna(subset=["bin_center", m]).sort_values(
                    "bin_center"
                )
                if fdf.empty:
                    continue
                interp = np.interp(
                    x_grid, fdf["bin_center"].values[::-1],
                    fdf[m].values[::-1],
                )
                fold_curves.append(interp)

            if not fold_curves:
                continue

            arr = np.array(fold_curves)
            mean = arr.mean(axis=0)
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)

            color = METRIC_COLORS[m]
            ax.plot(
                x_grid, mean,
                color=color, marker=METRIC_MARKERS[m],
                label=f"{METRIC_LABELS[m]} (mean)",
                markevery=max(1, len(x_grid) // 15),
            )
            ax.fill_between(
                x_grid, lo, hi,
                color=color, alpha=0.15,
            )

        ax.set_xlabel("Hours before birth")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.invert_xaxis()
        ax.legend(loc="best", fontsize=6)
        title = f"{mt_label} — Aggregated ({n_folds} folds)"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        style_axes(ax)
        save_figure(fig, output_dir / f"{fname}.png")
