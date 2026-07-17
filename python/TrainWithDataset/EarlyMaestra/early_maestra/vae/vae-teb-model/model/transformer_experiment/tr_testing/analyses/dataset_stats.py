"""Dataset statistics analysis (Category 7).

Characterizes the test data distribution before any model inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger


def _collect_metadata(loader, class_label: str) -> pd.DataFrame:
    """Iterate loader and collect metadata for all samples.

    Args:
        loader: DataLoader to iterate over.
        class_label: Class label string to assign to all samples.

    Returns:
        DataFrame with columns: guid, epoch, epoch_hours, class_label,
        cs_label, bg_label.
    """
    rows = []
    for batch in loader:
        B = batch.fhr_st.shape[0]
        for i in range(B):
            guid = batch.guid[i] if isinstance(batch.guid, (list, tuple)) else str(batch.guid)
            if isinstance(guid, bytes):
                guid = guid.decode("utf-8", errors="replace")
            epoch_val = float(batch.epoch[i]) if hasattr(batch.epoch, '__getitem__') else float(batch.epoch)
            cs = int(batch.cs_label[i]) if hasattr(batch.cs_label, '__getitem__') else int(batch.cs_label)
            bg = int(batch.bg_label[i]) if hasattr(batch.bg_label, '__getitem__') else int(batch.bg_label)
            rows.append({
                "guid": str(guid),
                "epoch": epoch_val,
                "epoch_hours": epoch_val / 3600.0,
                "class_label": class_label,
                "cs_label": cs,
                "bg_label": bg,
            })
    return pd.DataFrame(rows)


def _collect_st_stats(loader, class_label: str, max_batches: int = 20) -> Dict[str, Any]:
    """Collect scattering transform channel statistics.

    Args:
        loader: DataLoader to iterate over.
        class_label: Class label string.
        max_batches: Maximum number of batches to process.

    Returns:
        Dict with class_label and per-channel mean/std arrays for
        fhr_st and optionally up_st.
    """
    fhr_vals = []
    up_vals = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        fhr_vals.append(batch.fhr_st.cpu().numpy())
        if hasattr(batch, 'up_st') and batch.up_st is not None:
            up_vals.append(batch.up_st.cpu().numpy())

    result = {"class_label": class_label}
    if fhr_vals:
        fhr = np.concatenate(fhr_vals, axis=0)  # (N, T, C)
        result["fhr_st_mean"] = fhr.mean(axis=(0, 1))  # (C,)
        result["fhr_st_std"] = fhr.std(axis=(0, 1))
    if up_vals:
        up = np.concatenate(up_vals, axis=0)
        result["up_st_mean"] = up.mean(axis=(0, 1))
        result["up_st_std"] = up.std(axis=(0, 1))
    return result


def run_dataset_stats_analysis(
    class_loaders: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run dataset statistics analysis.

    Args:
        class_loaders: Dict mapping class names to DataLoaders.
        output_dir: Output directory for plots and data.

    Returns:
        Summary statistics dict.
    """
    from model.transformer.tr_testing.visualizers import plot_dataset_overview, plot_time_distribution, plot_st_coefficient_stats

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metadata from all classes
    all_meta = []
    for class_name, loader in class_loaders.items():
        logger.info(f"  Collecting metadata for {class_name}...")
        df = _collect_metadata(loader, class_name)
        all_meta.append(df)

    meta_df = pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
    meta_df.to_csv(output_dir / "metadata.csv", index=False)

    # Compute summary statistics
    summary = {}
    if not meta_df.empty:
        summary["n_samples"] = len(meta_df)
        summary["n_guids"] = meta_df["guid"].nunique()
        summary["n_classes"] = meta_df["class_label"].nunique()

        for cls in meta_df["class_label"].unique():
            cls_df = meta_df[meta_df["class_label"] == cls]
            summary[f"{cls}_n_samples"] = len(cls_df)
            summary[f"{cls}_n_guids"] = cls_df["guid"].nunique()
            summary[f"{cls}_epochs_per_guid_mean"] = cls_df.groupby("guid").size().mean()

    # Collect ST statistics
    st_stats = []
    for class_name, loader in class_loaders.items():
        st_stats.append(_collect_st_stats(loader, class_name))

    # Build dicts expected by visualizers from collected DataFrames/lists
    plot_stats = {}
    if not meta_df.empty:
        plot_stats["samples_per_class"] = (
            meta_df.groupby("class_label").size().to_dict()
        )
        plot_stats["guids_per_class"] = (
            meta_df.groupby("class_label")["guid"].nunique().to_dict()
        )
        plot_stats["epochs_per_guid"] = (
            meta_df.groupby("guid").size().tolist()
        )
        plot_stats["time_distribution"] = {
            cls: grp["epoch_hours"].values
            for cls, grp in meta_df.groupby("class_label")
        }

    st_plot_stats: Dict[str, Any] = {
        "fhr_st_stats": {},
        "up_st_stats": {},
    }
    for entry in st_stats:
        cls = entry["class_label"]
        if "fhr_st_mean" in entry:
            st_plot_stats["fhr_st_stats"][cls] = {
                "mean": entry["fhr_st_mean"],
                "std": entry["fhr_st_std"],
            }
        if "up_st_mean" in entry:
            st_plot_stats["up_st_stats"][cls] = {
                "mean": entry["up_st_mean"],
                "std": entry["up_st_std"],
            }

    # Generate plots
    try:
        plot_dataset_overview(plot_stats, output_dir)
    except Exception as e:
        logger.warning(f"Dataset overview plot failed: {e}")

    try:
        plot_time_distribution(plot_stats, output_dir)
    except Exception as e:
        logger.warning(f"Time distribution plot failed: {e}")

    try:
        plot_st_coefficient_stats(st_plot_stats, output_dir)
    except Exception as e:
        logger.warning(f"ST coefficient stats plot failed: {e}")

    # Save summary
    import json
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Dataset stats: {summary.get('n_samples', 0)} samples, "
                f"{summary.get('n_guids', 0)} GUIDs")
    return summary
