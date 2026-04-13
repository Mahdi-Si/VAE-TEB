"""
Dataset statistics analysis for VAE-TEB testing.

This module collects and visualizes statistics about the test dataset:
- Number of unique GUIDs (patients/babies)
- Number of epochs (20-minute segments)
- Distribution of epochs over time before birth (based on actual epoch values)
- Label distributions (cs_label, bg_label)
- Epochs per GUID distribution

The epoch field in the dataset represents seconds before birth (negative values).
Each epoch is a 20-minute segment, so consecutive epochs from the same GUID
are typically spaced by ~1200 seconds (20 minutes) or less if overlapping.

Example:
    >>> from testing.analyses.dataset_stats import run_dataset_stats_analysis
    >>> stats = run_dataset_stats_analysis(loader, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# Import visualization constants from visualizers module
try:
    from model.vae_teb_prediction.testing.visualizers import (
        COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_SKY,
        COLOR_LIGHT_GRAY, SAVE_DPI, _style_axes,
    )
except ImportError:
    # Fallback colors
    COLOR_BLUE = "#3F72AF"
    COLOR_ORANGE = "#FFB200"
    COLOR_GREEN = "#609966"
    COLOR_SKY = "#00ADB5"
    COLOR_LIGHT_GRAY = "#EEEEEE"
    SAVE_DPI = 600
    _style_axes = None


def seconds_to_hhmm(seconds: float) -> str:
    """Convert seconds before birth to HH:MM format (absolute value)."""
    abs_seconds = abs(seconds)
    hours = int(abs_seconds // 3600)
    minutes = int((abs_seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


def seconds_to_hours_minutes(seconds: float) -> Tuple[int, int]:
    """Convert seconds to (hours, minutes) tuple."""
    abs_seconds = abs(seconds)
    hours = int(abs_seconds // 3600)
    minutes = int((abs_seconds % 3600) // 60)
    return hours, minutes


def collect_dataset_stats(loader: Any) -> pd.DataFrame:
    """Collect metadata from all samples in the dataloader.

    Fields ``guid`` and ``epoch`` are required. ``cs_label`` and
    ``bg_label`` are optional — when a batch doesn't carry them (because
    they weren't listed in the config's ``dataset_kwargs.load_fields``,
    or the HDF5 file doesn't have those datasets), the corresponding
    columns default to ``0`` for all samples and a one-time warning is
    logged so downstream label-by-class plots register the absence
    rather than crashing.

    Args:
        loader: PyTorch DataLoader for test data.

    Returns:
        DataFrame with columns ``[guid, epoch_seconds, cs_label,
        bg_label]`` where ``epoch_seconds`` is seconds before birth
        (negative values).
    """
    records = []
    warned_missing: set = set()

    def _as_array(value: Any) -> Any:
        """Normalise a batch field to something indexable (or None)."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "numpy"):
            try:
                value = value.numpy()
            except Exception:  # noqa: BLE001
                pass
        return value

    def _safe_len(value: Any) -> int:
        """Return ``len(value)`` when defined, else 0."""
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 0

    for batch in loader:
        # Required metadata.
        guids = _as_array(batch.get("guid", None))
        epochs = _as_array(batch.get("epoch", None))
        if guids is None or epochs is None:
            if "required" not in warned_missing:
                logger.warning(
                    "dataset_stats: batch is missing 'guid' or 'epoch' — "
                    "skipping. Ensure these are in dataset_kwargs.load_fields."
                )
                warned_missing.add("required")
            continue

        batch_size = _safe_len(guids) or _safe_len(epochs) or 1

        # Optional label fields. If missing (or present but shorter than
        # the batch), fall back to zeros.
        cs_labels = _as_array(batch.get("cs_label", None))
        bg_labels = _as_array(batch.get("bg_label", None))
        if cs_labels is None or _safe_len(cs_labels) < batch_size:
            if "cs_label" not in warned_missing:
                logger.warning(
                    "dataset_stats: 'cs_label' not found in batch — "
                    "defaulting to 0 for all samples. Add 'cs_label' to "
                    "dataset_kwargs.load_fields to populate it."
                )
                warned_missing.add("cs_label")
            cs_labels = None
        if bg_labels is None or _safe_len(bg_labels) < batch_size:
            if "bg_label" not in warned_missing:
                logger.warning(
                    "dataset_stats: 'bg_label' not found in batch — "
                    "defaulting to 0 for all samples. Add 'bg_label' to "
                    "dataset_kwargs.load_fields to populate it."
                )
                warned_missing.add("bg_label")
            bg_labels = None

        def _index(arr: Any, i: int, default: Any) -> Any:
            if arr is None:
                return default
            try:
                return arr[i]
            except (IndexError, KeyError, TypeError):
                return default

        for i in range(batch_size):
            guid = _index(guids, i, "unknown")
            epoch = _index(epochs, i, 0.0)
            cs_label = _index(cs_labels, i, 0)
            bg_label = _index(bg_labels, i, 0)

            if isinstance(guid, bytes):
                guid = guid.decode("utf-8")

            records.append({
                "guid": str(guid),
                "epoch_seconds": float(epoch),  # seconds before birth (negative)
                "cs_label": int(bool(cs_label)),
                "bg_label": int(bool(bg_label)),
            })

    return pd.DataFrame(records)


def compute_stats_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute summary statistics from the collected metadata.

    Args:
        df: DataFrame with columns [guid, epoch_seconds, cs_label, bg_label]

    Returns:
        Dictionary with computed statistics.
    """
    if df.empty:
        return {"error": "No data collected"}

    # Basic counts
    n_samples = len(df)
    n_guids = df["guid"].nunique()
    unique_guids = df["guid"].unique().tolist()

    # Epochs per GUID
    epochs_per_guid = df.groupby("guid").size()
    epochs_per_guid_stats = {
        "min": int(epochs_per_guid.min()),
        "max": int(epochs_per_guid.max()),
        "mean": float(epochs_per_guid.mean()),
        "median": float(epochs_per_guid.median()),
        "std": float(epochs_per_guid.std()),
    }

    # Epoch (time before birth) statistics - convert to hours and minutes
    epoch_seconds = df["epoch_seconds"]
    epoch_minutes = epoch_seconds / 60.0
    epoch_hours = epoch_seconds / 3600.0

    # Get actual unique epoch values (sorted)
    unique_epochs_seconds = sorted(df["epoch_seconds"].unique())

    min_sec = float(epoch_seconds.min())
    max_sec = float(epoch_seconds.max())
    epoch_stats = {
        "min_seconds": min_sec,
        "max_seconds": max_sec,
        "min_minutes": float(epoch_minutes.min()),
        "max_minutes": float(epoch_minutes.max()),
        "min_hours": float(epoch_hours.min()),
        "max_hours": float(epoch_hours.max()),
        "mean_minutes": float(epoch_minutes.mean()),
        "median_minutes": float(epoch_minutes.median()),
        "std_minutes": float(epoch_minutes.std()),
        "min_hhmm": seconds_to_hhmm(min_sec),
        "max_hhmm": seconds_to_hhmm(max_sec),
        "n_unique_epochs": len(unique_epochs_seconds),
    }

    # Label distributions (epoch counts)
    cs_label_dist = df["cs_label"].value_counts().to_dict()
    bg_label_dist = df["bg_label"].value_counts().to_dict()

    # Detailed stats by cs_label
    stats_by_cs_label = {}
    for cs_val in [0, 1]:
        subset = df[df["cs_label"] == cs_val]
        if len(subset) > 0:
            guid_counts = subset.groupby("guid").size()
            max_sec = float(subset["epoch_seconds"].max())
            min_sec = float(subset["epoch_seconds"].min())
            stats_by_cs_label[cs_val] = {
                "n_epochs": len(subset),
                "n_guids": int(subset["guid"].nunique()),
                "epochs_per_guid_mean": float(guid_counts.mean()),
                "epochs_per_guid_std": float(guid_counts.std()) if len(guid_counts) > 1 else 0.0,
                "time_range_hhmm": f"{seconds_to_hhmm(max_sec)} to {seconds_to_hhmm(min_sec)}",
            }
        else:
            stats_by_cs_label[cs_val] = {"n_epochs": 0, "n_guids": 0}

    # Detailed stats by bg_label
    stats_by_bg_label = {}
    for bg_val in [0, 1]:
        subset = df[df["bg_label"] == bg_val]
        if len(subset) > 0:
            guid_counts = subset.groupby("guid").size()
            max_sec = float(subset["epoch_seconds"].max())
            min_sec = float(subset["epoch_seconds"].min())
            stats_by_bg_label[bg_val] = {
                "n_epochs": len(subset),
                "n_guids": int(subset["guid"].nunique()),
                "epochs_per_guid_mean": float(guid_counts.mean()),
                "epochs_per_guid_std": float(guid_counts.std()) if len(guid_counts) > 1 else 0.0,
                "time_range_hhmm": f"{seconds_to_hhmm(max_sec)} to {seconds_to_hhmm(min_sec)}",
            }
        else:
            stats_by_bg_label[bg_val] = {"n_epochs": 0, "n_guids": 0}

    # Detailed stats by label combination (cs_label x bg_label)
    stats_by_label_combo = {}
    for cs_val in [0, 1]:
        for bg_val in [0, 1]:
            key = f"cs{cs_val}_bg{bg_val}"
            subset = df[(df["cs_label"] == cs_val) & (df["bg_label"] == bg_val)]
            if len(subset) > 0:
                guid_counts = subset.groupby("guid").size()
                max_sec = float(subset["epoch_seconds"].max())
                min_sec = float(subset["epoch_seconds"].min())
                stats_by_label_combo[key] = {
                    "n_epochs": len(subset),
                    "n_guids": int(subset["guid"].nunique()),
                    "epochs_per_guid_mean": float(guid_counts.mean()),
                    "epochs_per_guid_std": float(guid_counts.std()) if len(guid_counts) > 1 else 0.0,
                    "time_range_hhmm": f"{seconds_to_hhmm(max_sec)} to {seconds_to_hhmm(min_sec)}",
                    "guids": subset["guid"].unique().tolist(),
                }
            else:
                stats_by_label_combo[key] = {"n_epochs": 0, "n_guids": 0, "guids": []}

    # GUID-level label info (what labels does each GUID have)
    guid_label_info = df.groupby("guid").agg({
        "cs_label": ["first", "nunique"],  # Check if consistent within GUID
        "bg_label": ["first", "nunique"],
        "epoch_seconds": ["count", "min", "max"],
    })
    guid_label_info.columns = ["cs_label", "cs_nunique", "bg_label", "bg_nunique", "n_epochs", "min_epoch", "max_epoch"]
    guid_label_info = guid_label_info.reset_index()

    # Count GUIDs by label (not epochs)
    guids_by_cs = guid_label_info.groupby("cs_label").size().to_dict()
    guids_by_bg = guid_label_info.groupby("bg_label").size().to_dict()

    # Time distribution based on actual unique epoch values
    # Group by the actual epoch values and count samples at each
    epoch_value_counts = df["epoch_seconds"].value_counts().sort_index()
    time_distribution_by_epoch = {
        seconds_to_hhmm(float(k)): int(v) for k, v in epoch_value_counts.items()
    }

    # Also create minute-level bins for visualization
    # Round to nearest minute
    epoch_minutes_rounded = (epoch_seconds / 60).round().astype(int)
    minute_distribution = epoch_minutes_rounded.value_counts().sort_index().to_dict()

    return {
        "n_samples": n_samples,
        "n_guids": n_guids,
        "unique_guids": unique_guids,
        "epochs_per_guid": epochs_per_guid_stats,
        "epoch_time": epoch_stats,
        "cs_label_distribution": cs_label_dist,
        "bg_label_distribution": bg_label_dist,
        "guids_by_cs_label": guids_by_cs,
        "guids_by_bg_label": guids_by_bg,
        "stats_by_cs_label": stats_by_cs_label,
        "stats_by_bg_label": stats_by_bg_label,
        "stats_by_label_combo": stats_by_label_combo,
        "time_distribution_by_epoch": time_distribution_by_epoch,
        "minute_distribution": minute_distribution,
        "unique_epochs_seconds": unique_epochs_seconds,
    }


def plot_dataset_statistics(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Create visualization plots for dataset statistics.

    Args:
        df: DataFrame with metadata.
        stats: Computed statistics dictionary.
        output_dir: Directory to save plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a 2x2 figure with all key statistics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Test Dataset Statistics", fontsize=12, fontweight="bold", y=0.98)

    # --- Plot 1: Epochs per GUID distribution ---
    ax = axes[0, 0]
    epochs_per_guid = df.groupby("guid").size()
    ax.hist(epochs_per_guid, bins=min(30, len(epochs_per_guid.unique())),
            color=COLOR_BLUE, edgecolor="white", alpha=0.8)
    ax.axvline(float(epochs_per_guid.mean()), color=COLOR_ORANGE, linestyle="--",
               linewidth=1.5, label=f"Mean: {epochs_per_guid.mean():.1f}")
    ax.axvline(float(epochs_per_guid.median()), color=COLOR_GREEN, linestyle=":",
               linewidth=1.5, label=f"Median: {epochs_per_guid.median():.1f}")
    ax.set_xlabel("Epochs per GUID (baby)")
    ax.set_ylabel("Number of GUIDs")
    ax.set_title(f"Epochs per Baby (n={stats['n_guids']} GUIDs)")
    ax.legend(fontsize=7)
    if _style_axes:
        _style_axes(ax)

    # --- Plot 2: Time distribution based on actual epoch values ---
    ax = axes[0, 1]
    epoch_minutes = df["epoch_seconds"] / 60.0

    # Use actual unique epoch values as bin edges (in minutes)
    unique_epochs_minutes = sorted(set(epoch_minutes))
    if len(unique_epochs_minutes) > 1:
        # Create bins centered on actual epoch values
        bin_edges = []
        step = unique_epochs_minutes[1] - unique_epochs_minutes[0] if len(unique_epochs_minutes) > 1 else 20.0
        for i, val in enumerate(unique_epochs_minutes):
            if i == 0:
                # First bin edge: half step before first value
                bin_edges.append(val - step / 2)
            if i < len(unique_epochs_minutes) - 1:
                next_gap = unique_epochs_minutes[i + 1] - val
                bin_edges.append(val + next_gap / 2)
            else:
                bin_edges.append(val + step / 2)

        ax.hist(epoch_minutes, bins=bin_edges, color=COLOR_SKY, edgecolor="white", alpha=0.8)
    else:
        ax.hist(epoch_minutes, bins=20, color=COLOR_SKY, edgecolor="white", alpha=0.8)

    ax.set_xlabel("Minutes before birth")
    ax.set_ylabel("Number of epochs")
    ax.set_title(f"Epoch Distribution ({stats['epoch_time']['n_unique_epochs']} unique time points)")
    ax.invert_xaxis()  # More negative = further from birth
    if _style_axes:
        _style_axes(ax)

    # --- Plot 3: Label distribution ---
    ax = axes[1, 0]
    labels = ["CS=0", "CS=1", "BG=0", "BG=1"]
    cs_dist = stats["cs_label_distribution"]
    bg_dist = stats["bg_label_distribution"]
    counts = [
        cs_dist.get(0, 0),
        cs_dist.get(1, 0),
        bg_dist.get(0, 0),
        bg_dist.get(1, 0),
    ]
    bars = ax.bar(labels, counts, color=[COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_SKY],
                  edgecolor="white", alpha=0.8)
    ax.set_ylabel("Number of epochs")
    ax.set_title("Label Distribution")
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count}", ha="center", va="bottom", fontsize=8)
    if _style_axes:
        _style_axes(ax)

    # --- Plot 4: Summary text box ---
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
    Dataset Summary
    ---------------
    Total Samples (epochs): {stats['n_samples']:,}
    Unique GUIDs (babies): {stats['n_guids']:,}
    Unique Time Points: {stats['epoch_time']['n_unique_epochs']}

    Epochs per GUID:
      Min: {stats['epochs_per_guid']['min']}
      Max: {stats['epochs_per_guid']['max']}
      Mean: {stats['epochs_per_guid']['mean']:.1f}
      Median: {stats['epochs_per_guid']['median']:.1f}

    Time Before Birth:
      Range: {stats['epoch_time']['max_hhmm']} to {stats['epoch_time']['min_hhmm']} (HH:MM)
      Range: {abs(stats['epoch_time']['max_minutes']):.0f} to {abs(stats['epoch_time']['min_minutes']):.0f} min
      Mean: {abs(stats['epoch_time']['mean_minutes']):.1f} min

    Labels:
      CS=1: {cs_dist.get(1, 0):,} ({100*cs_dist.get(1, 0)/stats['n_samples']:.1f}%)
      BG=1: {bg_dist.get(1, 0):,} ({100*bg_dist.get(1, 0)/stats['n_samples']:.1f}%)
    """

    ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_LIGHT_GRAY, alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / "dataset_statistics.pdf", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Detailed plot: Distribution at each unique epoch time point ---
    _plot_epoch_time_distribution(df, stats, output_dir)

    # --- Plot: Epochs per GUID ranked ---
    _plot_epochs_per_guid_ranked(df, stats, output_dir)

    # --- Plot: Label-based statistics ---
    _plot_label_statistics(df, stats, output_dir)

    logger.info(f"Dataset statistics plots saved to {output_dir}")


def _plot_epoch_time_distribution(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot detailed time distribution based on actual epoch values."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Get counts at each unique epoch time
    epoch_counts = df["epoch_seconds"].value_counts().sort_index()
    epoch_minutes = np.array(epoch_counts.index) / 60.0
    counts = np.array(epoch_counts.values)

    # Bar plot at actual epoch positions
    # Calculate bar width based on typical spacing (or minimum gap)
    if len(epoch_minutes) > 1:
        gaps = np.diff(epoch_minutes)
        bar_width = max(min(abs(gaps)) * 0.8, 1)  # At least 1 minute wide
    else:
        bar_width = 20  # Default 20 minutes

    ax.bar(epoch_minutes, counts, width=bar_width, color=COLOR_BLUE, edgecolor="white", alpha=0.8)

    ax.set_xlabel("Time before birth (minutes)")
    ax.set_ylabel("Number of epochs at this time point")
    ax.set_title(
        f"Epoch Distribution by Actual Time Points\n"
        f"(n={stats['n_samples']} epochs, {stats['epoch_time']['n_unique_epochs']} unique times, "
        f"range: {stats['epoch_time']['max_hhmm']} to {stats['epoch_time']['min_hhmm']})"
    )
    ax.invert_xaxis()

    # Add hour markers
    min_minutes = epoch_minutes.min()
    max_minutes = epoch_minutes.max()
    hour_marks = np.arange(np.floor(min_minutes / 60) * 60, np.ceil(max_minutes / 60) * 60 + 1, 60)
    for hm in hour_marks:
        if min_minutes <= hm <= max_minutes:
            ax.axvline(hm, color=COLOR_ORANGE, linestyle="--", linewidth=0.5, alpha=0.5)
            hours = int(abs(hm) // 60)
            ax.text(hm, ax.get_ylim()[1] * 0.95, f"{hours}h", fontsize=7, ha="center", color=COLOR_ORANGE)

    if _style_axes:
        _style_axes(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "time_distribution_detailed.pdf", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Also create a table/list of unique epoch times ---
    _save_epoch_time_table(df, output_dir)


def _save_epoch_time_table(df: pd.DataFrame, output_dir: Path) -> None:
    """Save a detailed table of epoch times and counts."""
    epoch_counts = df["epoch_seconds"].value_counts().sort_index()

    rows = []
    for epoch_sec, count in epoch_counts.items():
        epoch_sec_float = float(epoch_sec)
        hours, _ = seconds_to_hours_minutes(epoch_sec_float)
        rows.append({
            "epoch_seconds": epoch_sec_float,
            "hours_before_birth": hours,
            "minutes_before_birth": int(abs(epoch_sec_float) // 60),
            "time_hhmm": seconds_to_hhmm(epoch_sec_float),
            "count": int(count),
        })

    time_df = pd.DataFrame(rows)
    time_df.to_csv(output_dir / "epoch_time_distribution.csv", index=False)


def _plot_label_statistics(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot detailed label-based statistics."""
    # Create a 2x2 figure for label statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Dataset Statistics by Labels", fontsize=12, fontweight="bold", y=0.98)

    # --- Plot 1: GUIDs and Epochs by CS label ---
    ax = axes[0, 0]
    cs_stats = stats.get("stats_by_cs_label", {})
    x_labels = ["CS=0", "CS=1"]
    guids = [cs_stats.get(0, {}).get("n_guids", 0), cs_stats.get(1, {}).get("n_guids", 0)]
    epochs = [cs_stats.get(0, {}).get("n_epochs", 0), cs_stats.get(1, {}).get("n_epochs", 0)]

    x = np.arange(len(x_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, guids, width, label="GUIDs", color=COLOR_BLUE, alpha=0.8)
    bars2 = ax.bar(x + width/2, epochs, width, label="Epochs", color=COLOR_ORANGE, alpha=0.8)

    ax.set_ylabel("Count")
    ax.set_title("Distribution by CS Label (Caesarean Section)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Add count labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}",
                ha="center", va="bottom", fontsize=8)
    if _style_axes:
        _style_axes(ax)

    # --- Plot 2: GUIDs and Epochs by BG label ---
    ax = axes[0, 1]
    bg_stats = stats.get("stats_by_bg_label", {})
    x_labels = ["BG=0", "BG=1"]
    guids = [bg_stats.get(0, {}).get("n_guids", 0), bg_stats.get(1, {}).get("n_guids", 0)]
    epochs = [bg_stats.get(0, {}).get("n_epochs", 0), bg_stats.get(1, {}).get("n_epochs", 0)]

    bars1 = ax.bar(x - width/2, guids, width, label="GUIDs", color=COLOR_GREEN, alpha=0.8)
    bars2 = ax.bar(x + width/2, epochs, width, label="Epochs", color=COLOR_SKY, alpha=0.8)

    ax.set_ylabel("Count")
    ax.set_title("Distribution by BG Label (Blood Gas Available)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}",
                ha="center", va="bottom", fontsize=8)
    if _style_axes:
        _style_axes(ax)

    # --- Plot 3: Label combination matrix (GUIDs) ---
    ax = axes[1, 0]
    combo_stats = stats.get("stats_by_label_combo", {})
    matrix_guids = np.array([
        [combo_stats.get("cs0_bg0", {}).get("n_guids", 0), combo_stats.get("cs0_bg1", {}).get("n_guids", 0)],
        [combo_stats.get("cs1_bg0", {}).get("n_guids", 0), combo_stats.get("cs1_bg1", {}).get("n_guids", 0)],
    ])

    im = ax.imshow(matrix_guids, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["BG=0", "BG=1"])
    ax.set_yticklabels(["CS=0", "CS=1"])
    ax.set_xlabel("BG Label")
    ax.set_ylabel("CS Label")
    ax.set_title("GUIDs by Label Combination")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = matrix_guids[i, j]
            text_color = "white" if val > matrix_guids.max() / 2 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.6)

    # --- Plot 4: Label combination matrix (Epochs) ---
    ax = axes[1, 1]
    matrix_epochs = np.array([
        [combo_stats.get("cs0_bg0", {}).get("n_epochs", 0), combo_stats.get("cs0_bg1", {}).get("n_epochs", 0)],
        [combo_stats.get("cs1_bg0", {}).get("n_epochs", 0), combo_stats.get("cs1_bg1", {}).get("n_epochs", 0)],
    ])

    im = ax.imshow(matrix_epochs, cmap="Oranges", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["BG=0", "BG=1"])
    ax.set_yticklabels(["CS=0", "CS=1"])
    ax.set_xlabel("BG Label")
    ax.set_ylabel("CS Label")
    ax.set_title("Epochs by Label Combination")

    for i in range(2):
        for j in range(2):
            val = matrix_epochs[i, j]
            text_color = "white" if val > matrix_epochs.max() / 2 else "black"
            ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=12, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_dir / "label_statistics.pdf", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Save detailed label statistics to CSV ---
    _save_label_statistics_csv(stats, output_dir)


def _save_label_statistics_csv(stats: Dict[str, Any], output_dir: Path) -> None:
    """Save detailed label statistics to CSV files."""
    # CS label stats
    cs_rows = []
    for cs_val in [0, 1]:
        cs_stats = stats.get("stats_by_cs_label", {}).get(cs_val, {})
        cs_rows.append({
            "cs_label": cs_val,
            "n_guids": cs_stats.get("n_guids", 0),
            "n_epochs": cs_stats.get("n_epochs", 0),
            "epochs_per_guid_mean": cs_stats.get("epochs_per_guid_mean", 0),
            "epochs_per_guid_std": cs_stats.get("epochs_per_guid_std", 0),
            "time_range": cs_stats.get("time_range_hhmm", "N/A"),
        })
    pd.DataFrame(cs_rows).to_csv(output_dir / "stats_by_cs_label.csv", index=False)

    # BG label stats
    bg_rows = []
    for bg_val in [0, 1]:
        bg_stats = stats.get("stats_by_bg_label", {}).get(bg_val, {})
        bg_rows.append({
            "bg_label": bg_val,
            "n_guids": bg_stats.get("n_guids", 0),
            "n_epochs": bg_stats.get("n_epochs", 0),
            "epochs_per_guid_mean": bg_stats.get("epochs_per_guid_mean", 0),
            "epochs_per_guid_std": bg_stats.get("epochs_per_guid_std", 0),
            "time_range": bg_stats.get("time_range_hhmm", "N/A"),
        })
    pd.DataFrame(bg_rows).to_csv(output_dir / "stats_by_bg_label.csv", index=False)

    # Combination stats
    combo_rows = []
    for cs_val in [0, 1]:
        for bg_val in [0, 1]:
            key = f"cs{cs_val}_bg{bg_val}"
            combo_stats = stats.get("stats_by_label_combo", {}).get(key, {})
            combo_rows.append({
                "cs_label": cs_val,
                "bg_label": bg_val,
                "n_guids": combo_stats.get("n_guids", 0),
                "n_epochs": combo_stats.get("n_epochs", 0),
                "epochs_per_guid_mean": combo_stats.get("epochs_per_guid_mean", 0),
                "epochs_per_guid_std": combo_stats.get("epochs_per_guid_std", 0),
                "time_range": combo_stats.get("time_range_hhmm", "N/A"),
            })
    pd.DataFrame(combo_rows).to_csv(output_dir / "stats_by_label_combo.csv", index=False)


def _plot_epochs_per_guid_ranked(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot epochs per GUID in ranked order."""
    fig, ax = plt.subplots(figsize=(12, 4))

    epochs_per_guid = df.groupby("guid").size()
    epochs_per_guid_sorted = np.sort(np.array(epochs_per_guid.values))[::-1]
    x = np.arange(len(epochs_per_guid_sorted))

    ax.bar(x, epochs_per_guid_sorted, color=COLOR_BLUE, alpha=0.8, width=1.0)
    ax.set_xlabel("GUID rank (sorted by epoch count)")
    ax.set_ylabel("Number of epochs")
    ax.set_title(f"Epochs per GUID (ranked, n={stats['n_guids']} GUIDs)")

    mean_val = float(epochs_per_guid.mean())
    ax.axhline(mean_val, color=COLOR_ORANGE, linestyle="--",
               linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.legend()

    if _style_axes:
        _style_axes(ax)

    plt.tight_layout()
    plt.savefig(output_dir / "epochs_per_guid_ranked.pdf", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def run_dataset_stats_analysis(
    loader: Any,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Run complete dataset statistics analysis.

    Collects metadata from the dataloader, computes statistics,
    creates visualizations, and saves results.

    The epoch field is expected to be in seconds before birth (negative values).

    Args:
        loader: PyTorch DataLoader for test data.
        output_dir: Directory to save results.

    Returns:
        Dictionary with dataset statistics and output paths.

    Example:
        >>> stats = run_dataset_stats_analysis(test_loader, Path("results/dataset"))
        >>> print(f"Total samples: {stats['n_samples']}")
        >>> print(f"Unique GUIDs: {stats['n_guids']}")
        >>> print(f"Time range: {stats['epoch_time']['min_hhmm']} to {stats['epoch_time']['max_hhmm']}")
    """
    logger.info("Starting dataset statistics analysis...")

    # Collect metadata
    df = collect_dataset_stats(loader)

    if df.empty:
        logger.error("No data collected from dataloader.")
        return {"error": "No data collected"}

    # Compute statistics
    stats = compute_stats_summary(df)

    # Log summary
    logger.info(
        f"Dataset: {stats['n_samples']} epochs from {stats['n_guids']} GUIDs, "
        f"{stats['epoch_time']['n_unique_epochs']} unique time points, "
        f"range: {stats['epoch_time']['max_hhmm']} to {stats['epoch_time']['min_hhmm']} before birth"
    )

    # Log label breakdown
    cs_stats = stats.get("stats_by_cs_label", {})
    bg_stats = stats.get("stats_by_bg_label", {})
    logger.info(
        f"By CS label: CS=0: {cs_stats.get(0, {}).get('n_guids', 0)} GUIDs / {cs_stats.get(0, {}).get('n_epochs', 0)} epochs, "
        f"CS=1: {cs_stats.get(1, {}).get('n_guids', 0)} GUIDs / {cs_stats.get(1, {}).get('n_epochs', 0)} epochs"
    )
    logger.info(
        f"By BG label: BG=0: {bg_stats.get(0, {}).get('n_guids', 0)} GUIDs / {bg_stats.get(0, {}).get('n_epochs', 0)} epochs, "
        f"BG=1: {bg_stats.get(1, {}).get('n_guids', 0)} GUIDs / {bg_stats.get(1, {}).get('n_epochs', 0)} epochs"
    )

    # Create visualizations
    output_dir = Path(output_dir)
    plot_dataset_statistics(df, stats, output_dir)

    # Save statistics to JSON (exclude large lists and GUID lists from combo stats)
    import json

    # Clean combo stats to exclude GUID lists
    combo_stats_clean = {}
    for key, val in stats.get("stats_by_label_combo", {}).items():
        combo_stats_clean[key] = {k: v for k, v in val.items() if k != "guids"}

    stats_json = {
        k: v for k, v in stats.items()
        if k not in ("unique_guids", "unique_epochs_seconds", "time_distribution_by_epoch", "minute_distribution", "stats_by_label_combo")
    }
    stats_json["stats_by_label_combo"] = combo_stats_clean
    stats_json["unique_guids_count"] = len(stats["unique_guids"])
    stats_json["unique_epochs_count"] = len(stats.get("unique_epochs_seconds", []))

    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats_json, f, indent=2)

    # Save detailed CSV with all metadata
    df.to_csv(output_dir / "dataset_metadata.csv", index=False)

    logger.info(f"Dataset statistics saved to {output_dir}")

    # Return stats with output paths
    return {
        **stats,
        "output_dir": str(output_dir),
        "plots": [
            str(output_dir / "dataset_statistics.pdf"),
            str(output_dir / "time_distribution_detailed.pdf"),
            str(output_dir / "epochs_per_guid_ranked.pdf"),
            str(output_dir / "label_statistics.pdf"),
        ],
        "data_csv": str(output_dir / "dataset_metadata.csv"),
        "time_table_csv": str(output_dir / "epoch_time_distribution.csv"),
        "label_stats_csv": {
            "by_cs_label": str(output_dir / "stats_by_cs_label.csv"),
            "by_bg_label": str(output_dir / "stats_by_bg_label.csv"),
            "by_combo": str(output_dir / "stats_by_label_combo.csv"),
        },
    }
