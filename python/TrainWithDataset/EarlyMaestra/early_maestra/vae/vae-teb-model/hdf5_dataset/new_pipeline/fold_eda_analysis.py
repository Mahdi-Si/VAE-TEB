"""Fold-level exploratory data analysis for the CTG dataset pipeline.

Reads the actual HDF5 files produced for a single fold and generates
diagnostic plots + a summary report that validate the dataset structure,
class balance, TLO distribution, and signal coverage.

Usage::

    from fold_eda_analysis import run_fold_eda
    run_fold_eda(fold_dir, output_dir)
"""
from __future__ import annotations

import glob
import logging
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Journal-quality matplotlib defaults (mirrors guid_analysis.py)
# ---------------------------------------------------------------------------
_JOURNAL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------
_SUBGROUP_COLORS = {
    "acidosis_cs": "#e06666",
    "acidosis_no_cs": "#ea9999",
    "hie_cs": "#b4a7d6",
    "hie_no_cs": "#d5a6bd",
    "healthy_bg_cs": "#6fa8dc",
    "healthy_bg_no_cs": "#93c47d",
    "healthy_no_bg_cs": "#f6b26b",
    "healthy_no_bg_no_cs": "#ffd966",
}

_SUBGROUP_ORDER = [
    "acidosis_cs", "acidosis_no_cs",
    "hie_cs", "hie_no_cs",
    "healthy_bg_cs", "healthy_bg_no_cs",
    "healthy_no_bg_cs", "healthy_no_bg_no_cs",
]

_PARTITION_COLORS = {
    "train": "#6fa8dc",
    "val": "#f6b26b",
    "test": "#93c47d",
}

_PARTITION_ORDER = ["train", "val", "test"]

_CLASS_COLORS = {
    "unhealthy": "#e06666",
    "healthy": "#6fa8dc",
}

_UNHEALTHY_SUBGROUPS = {
    "acidosis_cs", "acidosis_no_cs", "hie_cs", "hie_no_cs",
}

_BIN_LABELS = ["short", "medium", "long", "unknown"]
_BIN_COLORS = ["#6fa8dc", "#f6b26b", "#e06666", "#cccccc"]
_N_DURATION_BINS = 3


def _remove_spines(ax: plt.Axes) -> None:
    """Remove top and right spines from an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compute_duration_bins(
    durations: pd.Series,
    n_bins: int = _N_DURATION_BINS,
) -> pd.Series:
    """Bin labour durations into quantile groups.

    Args:
        durations: Series of labour duration in hours (NaN for unknown).
        n_bins: Number of quantile bins for known durations.

    Returns:
        Series of string bin labels aligned with input.
    """
    known_mask = durations.notna()
    labels = pd.Series("unknown", index=durations.index)

    if known_mask.sum() == 0:
        return labels

    known_vals = durations[known_mask].values
    quantiles = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
    boundaries = np.quantile(known_vals, quantiles)
    bin_idx = np.digitize(known_vals, boundaries)

    bin_names = _BIN_LABELS[:n_bins]
    labels.loc[known_mask] = [bin_names[i] for i in bin_idx]
    return labels


# ============================================================================
# Data extraction from HDF5 files
# ============================================================================
def _read_hdf5_guid_summary(hdf5_path: str) -> pd.DataFrame:
    """Read an HDF5 file and return a per-GUID summary DataFrame.

    Args:
        hdf5_path: Path to a subgroup HDF5 file.

    Returns:
        DataFrame with one row per GUID containing segment counts,
        TLO statistics, epoch range, and mean weight.
    """
    rows: List[Dict[str, Any]] = []
    try:
        with h5py.File(hdf5_path, "r") as f:
            n_samples = f["guid"].shape[0]
            if n_samples == 0:
                return pd.DataFrame()

            guids = f["guid"][:]
            # Decode bytes to str if needed
            if isinstance(guids[0], bytes):
                guids = np.array([g.decode("utf-8") for g in guids])

            tlo = f["time_from_labor_onset"][:]
            epochs = f["epoch"][:]
            weights = f["weight"][:]  # shape (N, 330)
            mean_weights = np.nanmean(weights, axis=1)

        # Group by GUID
        unique_guids = np.unique(guids)
        for guid in unique_guids:
            mask = guids == guid
            g_tlo = tlo[mask]
            g_epochs = epochs[mask]
            g_weights = mean_weights[mask]

            # TLO: non-NaN values indicate labour onset is known
            valid_tlo = g_tlo[~np.isnan(g_tlo)]
            has_tlo = len(valid_tlo) > 0

            # Labour duration estimate: max TLO across segments
            # (segment closest to delivery gives best total duration)
            labour_dur_hours = float(np.max(valid_tlo) / 3600.0) if has_tlo else float("nan")

            rows.append({
                "guid": guid,
                "n_segments": int(mask.sum()),
                "has_tlo": has_tlo,
                "labour_duration_hours": labour_dur_hours,
                "epoch_min": float(np.min(g_epochs)),
                "epoch_max": float(np.max(g_epochs)),
                "mean_weight": float(np.mean(g_weights)),
            })

    except Exception as e:
        logger.error(f"Failed to read {hdf5_path}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _build_fold_dataframe(
    fold_dir: str,
    test_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Build a GUID-level DataFrame from fold HDF5 files.

    Reads all ``*.hdf5`` files from ``{fold_dir}/{train,val,test}/`` and
    optionally from a separate shared test directory.

    Args:
        fold_dir: Path to the fold directory (e.g. ``.../fold_1``).
        test_dir: Optional separate test directory (holdout mode).

    Returns:
        DataFrame with one row per GUID per partition, with columns:
        ``partition``, ``subgroup``, ``guid``, ``n_segments``, ``has_tlo``,
        ``labour_duration_hours``, ``epoch_min``, ``epoch_max``,
        ``mean_weight``, ``class_label``.
    """
    all_dfs: List[pd.DataFrame] = []

    # Discover partitions in fold_dir
    for partition in _PARTITION_ORDER:
        part_dir = os.path.join(fold_dir, partition)
        if not os.path.isdir(part_dir):
            continue
        for hdf5_path in sorted(glob.glob(os.path.join(part_dir, "*.hdf5"))):
            subgroup = os.path.splitext(os.path.basename(hdf5_path))[0]
            df = _read_hdf5_guid_summary(hdf5_path)
            if df.empty:
                continue
            df["partition"] = partition
            df["subgroup"] = subgroup
            all_dfs.append(df)

    # Holdout mode: separate test directory
    if test_dir and os.path.isdir(test_dir):
        for hdf5_path in sorted(glob.glob(os.path.join(test_dir, "*.hdf5"))):
            subgroup = os.path.splitext(os.path.basename(hdf5_path))[0]
            df = _read_hdf5_guid_summary(hdf5_path)
            if df.empty:
                continue
            df["partition"] = "test"
            df["subgroup"] = subgroup
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No HDF5 data found — returning empty DataFrame")
        return pd.DataFrame()

    fold_df = pd.concat(all_dfs, ignore_index=True)
    fold_df["class_label"] = fold_df["subgroup"].apply(
        lambda sg: "unhealthy" if sg in _UNHEALTHY_SUBGROUPS else "healthy"
    )
    return fold_df


# ============================================================================
# Plot 1: TLO distribution by partition
# ============================================================================
def _plot_tlo_distribution(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Overlaid histograms of labour duration for train/val/test.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    df = fold_df.dropna(subset=["labour_duration_hours"])
    if df.empty:
        logger.warning("No TLO data for distribution plot — skipping")
        return None

    out_path = os.path.join(output_dir, "tlo_distribution.png")

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        partitions = [p for p in _PARTITION_ORDER if p in df["partition"].unique()]
        stats_lines: List[str] = []

        for part in partitions:
            subset = df[df["partition"] == part]["labour_duration_hours"]
            if subset.empty:
                continue
            ax.hist(
                subset, bins=20, alpha=0.45,
                color=_PARTITION_COLORS[part], label=part,
                edgecolor="white", linewidth=0.3,
            )
            stats_lines.append(
                f"{part}: n={len(subset)}, "
                f"mean={subset.mean():.1f}h, "
                f"med={subset.median():.1f}h"
            )

        ax.set_xlabel("Labour Duration (hours)")
        ax.set_ylabel("GUID Count")
        ax.set_title("Labour Duration Distribution by Partition")
        ax.legend(frameon=False)
        _remove_spines(ax)

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="#cccccc", linewidth=0.3),
        )

        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved TLO distribution plot: {out_path}")
    return out_path


# ============================================================================
# Plot 2: TLO presence ratio per subgroup per partition
# ============================================================================
def _plot_tlo_presence_ratio(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Multi-panel bar chart of TLO presence per subgroup per partition.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    if fold_df.empty:
        return None

    partitions = [p for p in _PARTITION_ORDER if p in fold_df["partition"].unique()]
    active_sgs = [sg for sg in _SUBGROUP_ORDER if sg in fold_df["subgroup"].unique()]
    if not active_sgs:
        return None

    out_path = os.path.join(output_dir, "tlo_presence_ratio.png")

    with plt.rc_context(_JOURNAL_RC):
        n_panels = len(partitions)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(2.5 * n_panels, 3.5), sharey=True,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, part in zip(axes, partitions):
            part_df = fold_df[fold_df["partition"] == part]
            with_tlo = []
            without_tlo = []
            labels = []
            for sg in active_sgs:
                sg_df = part_df[part_df["subgroup"] == sg]
                n_with = sg_df["has_tlo"].sum()
                n_without = len(sg_df) - n_with
                with_tlo.append(n_with)
                without_tlo.append(n_without)
                labels.append(sg.replace("_", "\n"))

            x = np.arange(len(labels))
            width = 0.35
            ax.bar(
                x - width / 2, with_tlo, width, label="Has TLO",
                color="#6fa8dc", edgecolor="white", linewidth=0.3,
            )
            ax.bar(
                x + width / 2, without_tlo, width, label="No TLO",
                color="#cccccc", edgecolor="white", linewidth=0.3,
            )

            for i in range(len(labels)):
                total = with_tlo[i] + without_tlo[i]
                if total > 0:
                    pct = with_tlo[i] / total * 100
                    ax.text(
                        x[i], max(with_tlo[i], without_tlo[i]) + 0.3,
                        f"{pct:.0f}%", ha="center", fontsize=5.5,
                    )

            ax.set_title(part, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=5)
            _remove_spines(ax)
            if ax == axes[0]:
                ax.set_ylabel("GUID Count")

        axes[-1].legend(frameon=False, loc="upper right", fontsize=6)
        fig.suptitle(
            "TLO Presence Ratio by Subgroup",
            fontsize=9, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved TLO presence ratio plot: {out_path}")
    return out_path


# ============================================================================
# Plot 3: Duration bin distribution by partition
# ============================================================================
def _plot_duration_bin_distribution(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Grouped bar chart of duration bin counts per partition.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    if fold_df.empty:
        return None

    df = fold_df.copy()
    df["duration_bin"] = _compute_duration_bins(df["labour_duration_hours"])

    partitions = [p for p in _PARTITION_ORDER if p in df["partition"].unique()]
    bins_present = [b for b in _BIN_LABELS if b in df["duration_bin"].unique()]
    if not bins_present:
        return None

    out_path = os.path.join(output_dir, "duration_bin_distribution.png")

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(5.0, 3.5))

        x = np.arange(len(partitions))
        n_bins = len(bins_present)
        total_width = 0.75
        bar_width = total_width / n_bins

        for i, bin_label in enumerate(bins_present):
            counts = []
            for part in partitions:
                n = (
                    (df["partition"] == part)
                    & (df["duration_bin"] == bin_label)
                ).sum()
                counts.append(n)
            color_idx = _BIN_LABELS.index(bin_label)
            offset = (i - (n_bins - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, counts, bar_width,
                label=bin_label, color=_BIN_COLORS[color_idx],
                edgecolor="white", linewidth=0.3,
            )
            for bar, c in zip(bars, counts):
                if c > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        str(c), ha="center", va="bottom", fontsize=5.5,
                    )

        ax.set_xlabel("Partition")
        ax.set_ylabel("GUID Count")
        ax.set_title("Labour Duration Bin Distribution by Partition")
        ax.set_xticks(x)
        ax.set_xticklabels(partitions)
        ax.legend(frameon=False, fontsize=6)
        _remove_spines(ax)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved duration bin plot: {out_path}")
    return out_path


# ============================================================================
# Plot 4: Subgroup composition (GUIDs + segments)
# ============================================================================
def _plot_subgroup_composition(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Two-row panel: GUID counts and segment counts per subgroup.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    if fold_df.empty:
        return None

    partitions = [p for p in _PARTITION_ORDER if p in fold_df["partition"].unique()]
    active_sgs = [sg for sg in _SUBGROUP_ORDER if sg in fold_df["subgroup"].unique()]
    if not active_sgs:
        return None

    out_path = os.path.join(output_dir, "subgroup_composition.png")

    with plt.rc_context(_JOURNAL_RC):
        fig, (ax_guid, ax_seg) = plt.subplots(
            2, 1, figsize=(6.5, 5.5), sharex=True,
        )

        x = np.arange(len(active_sgs))
        n_parts = len(partitions)
        total_width = 0.75
        bar_width = total_width / n_parts

        for row_idx, (ax, metric, ylabel) in enumerate([
            (ax_guid, "guid_count", "GUID Count"),
            (ax_seg, "n_segments", "Segment Count"),
        ]):
            for i, part in enumerate(partitions):
                counts = []
                for sg in active_sgs:
                    mask = (fold_df["partition"] == part) & (fold_df["subgroup"] == sg)
                    if metric == "guid_count":
                        counts.append(mask.sum())
                    else:
                        counts.append(fold_df.loc[mask, "n_segments"].sum())
                offset = (i - (n_parts - 1) / 2) * bar_width
                bars = ax.bar(
                    x + offset, counts, bar_width,
                    label=part if row_idx == 0 else None,
                    color=_PARTITION_COLORS[part],
                    edgecolor="white", linewidth=0.3,
                )
                for bar, c in zip(bars, counts):
                    if c > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), str(c),
                            ha="center", va="bottom", fontsize=4.5,
                        )
            ax.set_ylabel(ylabel)
            _remove_spines(ax)

        ax_seg.set_xticks(x)
        ax_seg.set_xticklabels(
            [sg.replace("_", "\n") for sg in active_sgs], fontsize=5,
        )
        ax_guid.set_title("Subgroup Composition: GUIDs and Segments")
        ax_guid.legend(frameon=False, fontsize=6)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved subgroup composition plot: {out_path}")
    return out_path


# ============================================================================
# Plot 5: Segment coverage per GUID (box + strip)
# ============================================================================
def _plot_segment_coverage(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Box + strip plot of segment count per GUID per subgroup.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    if fold_df.empty:
        return None

    active_sgs = [sg for sg in _SUBGROUP_ORDER if sg in fold_df["subgroup"].unique()]
    if not active_sgs:
        return None

    out_path = os.path.join(output_dir, "segment_coverage.png")

    with plt.rc_context(_JOURNAL_RC):
        fig, (ax_seg, ax_span) = plt.subplots(
            1, 2, figsize=(9.0, 3.5),
        )

        # Left panel: segments per GUID
        box_data_seg = []
        positions_seg = []
        colors_seg = []
        for i, sg in enumerate(active_sgs):
            vals = fold_df[fold_df["subgroup"] == sg]["n_segments"].values
            if len(vals) > 0:
                box_data_seg.append(vals)
                positions_seg.append(i)
                colors_seg.append(_SUBGROUP_COLORS.get(sg, "#999999"))

        if box_data_seg:
            bp = ax_seg.boxplot(
                box_data_seg, positions=positions_seg, widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=0.6),
                whiskerprops=dict(linewidth=0.4),
                capprops=dict(linewidth=0.4),
            )
            for patch, c in zip(bp["boxes"], colors_seg):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)

            rng = np.random.RandomState(42)
            for pos, vals, c in zip(positions_seg, box_data_seg, colors_seg):
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax_seg.scatter(
                    pos + jitter, vals, s=6, alpha=0.5, color=c,
                    edgecolors="none", zorder=3,
                )

        ax_seg.set_xticks(range(len(active_sgs)))
        ax_seg.set_xticklabels(
            [sg.replace("_", "\n") for sg in active_sgs], fontsize=5,
        )
        ax_seg.set_ylabel("Segments per GUID")
        ax_seg.set_title("Segment Count per GUID")
        _remove_spines(ax_seg)

        # Right panel: epoch span (hours of coverage)
        box_data_span = []
        positions_span = []
        colors_span = []
        for i, sg in enumerate(active_sgs):
            sg_df = fold_df[fold_df["subgroup"] == sg]
            if len(sg_df) > 0:
                span_hours = (sg_df["epoch_max"] - sg_df["epoch_min"]).abs() / 3600.0
                box_data_span.append(span_hours.values)
                positions_span.append(i)
                colors_span.append(_SUBGROUP_COLORS.get(sg, "#999999"))

        if box_data_span:
            bp2 = ax_span.boxplot(
                box_data_span, positions=positions_span, widths=0.5,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=0.6),
                whiskerprops=dict(linewidth=0.4),
                capprops=dict(linewidth=0.4),
            )
            for patch, c in zip(bp2["boxes"], colors_span):
                patch.set_facecolor(c)
                patch.set_alpha(0.5)

            rng2 = np.random.RandomState(42)
            for pos, vals, c in zip(positions_span, box_data_span, colors_span):
                jitter = rng2.uniform(-0.12, 0.12, size=len(vals))
                ax_span.scatter(
                    pos + jitter, vals, s=6, alpha=0.5, color=c,
                    edgecolors="none", zorder=3,
                )

        ax_span.set_xticks(range(len(active_sgs)))
        ax_span.set_xticklabels(
            [sg.replace("_", "\n") for sg in active_sgs], fontsize=5,
        )
        ax_span.set_ylabel("Epoch Span (hours)")
        ax_span.set_title("Signal Coverage Span per GUID")
        _remove_spines(ax_span)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved segment coverage plot: {out_path}")
    return out_path


# ============================================================================
# Plot 6: Class balance per partition
# ============================================================================
def _plot_class_balance(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> Optional[str]:
    """Side-by-side bars of unhealthy/healthy GUID counts per partition.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the plot.

    Returns:
        Path to saved figure, or ``None`` if skipped.
    """
    if fold_df.empty:
        return None

    partitions = [p for p in _PARTITION_ORDER if p in fold_df["partition"].unique()]
    out_path = os.path.join(output_dir, "class_balance.png")

    with plt.rc_context(_JOURNAL_RC):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        x = np.arange(len(partitions))
        width = 0.35

        unhealthy_counts = []
        healthy_counts = []
        for part in partitions:
            pdf = fold_df[fold_df["partition"] == part]
            unhealthy_counts.append((pdf["class_label"] == "unhealthy").sum())
            healthy_counts.append((pdf["class_label"] == "healthy").sum())

        ax.bar(
            x - width / 2, unhealthy_counts, width,
            label="Unhealthy", color=_CLASS_COLORS["unhealthy"],
            edgecolor="white", linewidth=0.3,
        )
        ax.bar(
            x + width / 2, healthy_counts, width,
            label="Healthy", color=_CLASS_COLORS["healthy"],
            edgecolor="white", linewidth=0.3,
        )

        for i in range(len(partitions)):
            total = unhealthy_counts[i] + healthy_counts[i]
            if total > 0:
                pct_u = unhealthy_counts[i] / total * 100
                y_pos = max(unhealthy_counts[i], healthy_counts[i]) + 0.5
                ax.text(
                    x[i], y_pos, f"{pct_u:.0f}/{100 - pct_u:.0f}%",
                    ha="center", fontsize=6,
                )

        ax.set_xlabel("Partition")
        ax.set_ylabel("GUID Count")
        ax.set_title("Class Balance by Partition")
        ax.set_xticks(x)
        ax.set_xticklabels(partitions)
        ax.legend(frameon=False, fontsize=6)
        _remove_spines(ax)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    logger.info(f"Saved class balance plot: {out_path}")
    return out_path


# ============================================================================
# Markdown report
# ============================================================================
def _write_fold_eda_report(
    fold_df: pd.DataFrame,
    output_dir: str,
) -> str:
    """Write a Markdown summary report of the fold dataset.

    Args:
        fold_df: GUID-level DataFrame from ``_build_fold_dataframe()``.
        output_dir: Directory for saving the report.

    Returns:
        Path to saved report file.
    """
    out_path = os.path.join(output_dir, "fold_eda_report.md")
    partitions = [p for p in _PARTITION_ORDER if p in fold_df["partition"].unique()]
    active_sgs = [sg for sg in _SUBGROUP_ORDER if sg in fold_df["subgroup"].unique()]

    lines: List[str] = []
    lines.append("# Fold EDA Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # --- GUID and segment counts per subgroup per partition ---
    lines.append("\n## Subgroup Composition\n")
    header = "| Subgroup | " + " | ".join(
        f"{p} GUIDs / Segs" for p in partitions
    ) + " |"
    sep = "|----------|" + "|".join(
        ["--------------:" for _ in partitions]
    ) + "|"
    lines.append(header)
    lines.append(sep)
    for sg in active_sgs:
        cells = []
        for p in partitions:
            mask = (fold_df["partition"] == p) & (fold_df["subgroup"] == sg)
            n_guids = mask.sum()
            n_segs = fold_df.loc[mask, "n_segments"].sum()
            cells.append(f"{n_guids} / {n_segs}")
        lines.append(f"| {sg} | " + " | ".join(cells) + " |")

    # Totals row
    cells = []
    for p in partitions:
        mask = fold_df["partition"] == p
        cells.append(f"**{mask.sum()}** / **{fold_df.loc[mask, 'n_segments'].sum()}**")
    lines.append(f"| **Total** | " + " | ".join(cells) + " |")

    # --- Class balance ---
    lines.append("\n## Class Balance\n")
    lines.append("| Partition | Unhealthy | Healthy | Unhealthy % |")
    lines.append("|-----------|----------:|--------:|------------:|")
    for p in partitions:
        pdf = fold_df[fold_df["partition"] == p]
        n_u = (pdf["class_label"] == "unhealthy").sum()
        n_h = (pdf["class_label"] == "healthy").sum()
        total = n_u + n_h
        pct = f"{n_u / total * 100:.1f}%" if total > 0 else "—"
        lines.append(f"| {p} | {n_u} | {n_h} | {pct} |")

    # --- TLO coverage ---
    lines.append("\n## TLO Coverage\n")
    lines.append(
        "| Partition | GUIDs | Has TLO | % TLO | "
        "Mean Dur (h) | Median Dur (h) |"
    )
    lines.append(
        "|-----------|------:|--------:|------:|"
        "-------------:|---------------:|"
    )
    for p in partitions:
        pdf = fold_df[fold_df["partition"] == p]
        n_total = len(pdf)
        n_tlo = pdf["has_tlo"].sum()
        pct = f"{n_tlo / n_total * 100:.1f}%" if n_total > 0 else "—"
        vals = pdf["labour_duration_hours"].dropna()
        mean_d = f"{vals.mean():.1f}" if len(vals) > 0 else "—"
        med_d = f"{vals.median():.1f}" if len(vals) > 0 else "—"
        lines.append(
            f"| {p} | {n_total} | {n_tlo} | {pct} | {mean_d} | {med_d} |"
        )

    # --- Duration bins ---
    lines.append("\n## Duration Bin Distribution\n")
    df_binned = fold_df.copy()
    df_binned["duration_bin"] = _compute_duration_bins(
        df_binned["labour_duration_hours"],
    )
    bins_present = [b for b in _BIN_LABELS if b in df_binned["duration_bin"].unique()]
    header = "| Partition | " + " | ".join(bins_present) + " |"
    sep_line = "|-----------|" + "|".join(["------:" for _ in bins_present]) + "|"
    lines.append(header)
    lines.append(sep_line)
    for p in partitions:
        pdf = df_binned[df_binned["partition"] == p]
        counts = [str((pdf["duration_bin"] == b).sum()) for b in bins_present]
        lines.append(f"| {p} | " + " | ".join(counts) + " |")

    # --- Signal quality ---
    lines.append("\n## Signal Quality (Mean Weight)\n")
    lines.append(
        "| Subgroup | n GUIDs | Mean | Median | Min | Max |"
    )
    lines.append(
        "|----------|--------:|-----:|-------:|----:|----:|"
    )
    for sg in active_sgs:
        vals = fold_df[fold_df["subgroup"] == sg]["mean_weight"]
        if len(vals) > 0:
            lines.append(
                f"| {sg} | {len(vals)} | {vals.mean():.3f} | "
                f"{vals.median():.3f} | {vals.min():.3f} | "
                f"{vals.max():.3f} |"
            )
        else:
            lines.append(f"| {sg} | 0 | — | — | — | — |")

    report_text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Saved fold EDA report: {out_path}")
    return out_path


# ============================================================================
# Public entry point
# ============================================================================
def run_fold_eda(
    fold_dir: str,
    output_dir: str,
    test_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run fold-level EDA on created HDF5 files and generate plots + report.

    Reads all HDF5 files from the fold directory's train/val/test
    partitions and produces diagnostic plots that validate the dataset
    structure, class balance, TLO distribution, and signal coverage.

    Args:
        fold_dir: Path to the fold directory (e.g. ``.../fold_1``).
        output_dir: Directory for output plots and report.
        test_dir: Optional separate test directory (holdout mode where
            the test set is shared across folds).

    Returns:
        Summary dict with paths to generated files.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Running fold EDA on {fold_dir} -> {output_dir}")

    fold_df = _build_fold_dataframe(fold_dir, test_dir=test_dir)
    if fold_df.empty:
        logger.warning("Empty fold DataFrame — skipping all EDA plots")
        return {"status": "skipped", "reason": "empty fold_df"}

    n_guids = len(fold_df)
    n_segs = fold_df["n_segments"].sum()
    logger.info(
        f"Fold EDA: {n_guids} GUID entries, {n_segs} total segments "
        f"across {fold_df['partition'].nunique()} partitions"
    )

    generated: Dict[str, Optional[str]] = {}

    generated["tlo_distribution"] = _plot_tlo_distribution(fold_df, output_dir)
    generated["tlo_presence"] = _plot_tlo_presence_ratio(fold_df, output_dir)
    generated["duration_bins"] = _plot_duration_bin_distribution(fold_df, output_dir)
    generated["subgroup_composition"] = _plot_subgroup_composition(fold_df, output_dir)
    generated["segment_coverage"] = _plot_segment_coverage(fold_df, output_dir)
    generated["class_balance"] = _plot_class_balance(fold_df, output_dir)
    generated["report"] = _write_fold_eda_report(fold_df, output_dir)

    n_generated = sum(1 for v in generated.values() if v is not None)
    logger.info(
        f"Fold EDA complete: {n_generated}/{len(generated)} outputs generated"
    )

    return {
        "status": "ok",
        "output_dir": output_dir,
        "n_guids": n_guids,
        "n_segments": int(n_segs),
        "generated_files": {
            k: v for k, v in generated.items() if v is not None
        },
    }
