"""Extract per-GUID metadata from the classification dataset pickle file.

Reads ``classification_dataset_records.pickle`` produced by
``create_hdf5_dataset.create_records()`` and generates a CSV with one row
per unique GUID, including subgroup labels, domain-start range, and
time-from-labor-onset (TLO).

Two operating modes:
    **Full mode** (default): runs the lightweight MIMO prescreen on each
    GUID to obtain valid domain-start values.  Requires access to the
    original ``.mat`` files.

    **Light mode** (``--skip_mimo``): skips MIMO processing; domain-start
    and segment-count columns will be NaN.  Useful when ``.mat`` files are
    not accessible.

Usage examples::

    # Full mode (with MIMO prescreen)
    python extract_classification_guids.py \\
        --pickle_path /data1/.../classification_dataset_records.pickle \\
        --labor_onset_csv labor_onset_sample.csv \\
        --output_csv classification_guids.csv

    # Light mode (pickle + CSV only, no .mat access needed)
    python extract_classification_guids.py \\
        --pickle_path /data1/.../classification_dataset_records.pickle \\
        --labor_onset_csv labor_onset_sample.csv \\
        --skip_mimo \\
        --output_csv classification_guids.csv
"""

import math
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Ensure local imports work when running from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from create_hdf5_dataset import (
    _normalize_guid,
    load_labor_onset_data,
    prescreen_guid,
)

# Mapping from subgroup key (as stored in the pickle) to classification
# metadata.  class_label mirrors the three-class scheme used by the
# classifier (Healthy / Acidosis / HIE).
SUBGROUP_META: Dict[str, Dict[str, Any]] = {
    "healthy_no_bg_no_cs": {"class_label": "Healthy",  "cs_label": False, "bg_label": False},
    "healthy_no_bg_cs":    {"class_label": "Healthy",  "cs_label": True,  "bg_label": False},
    "healthy_bg_cs":       {"class_label": "Healthy",  "cs_label": True,  "bg_label": True},
    "healthy_bg_no_cs":    {"class_label": "Healthy",  "cs_label": False, "bg_label": True},
    "acidosis_cs":         {"class_label": "Acidosis", "cs_label": True,  "bg_label": True},
    "acidosis_no_cs":      {"class_label": "Acidosis", "cs_label": False, "bg_label": True},
    "hie_cs":              {"class_label": "HIE",      "cs_label": True,  "bg_label": True},
    "hie_no_cs":           {"class_label": "HIE",      "cs_label": False, "bg_label": True},
}


def extract_unique_guids(
    pickle_path: str,
) -> Dict[str, Tuple[str, str]]:
    """Extract unique GUIDs and their subgroup assignment from the folds pickle.

    Each GUID appears in multiple folds (as train/val/test), but always
    belongs to a single subgroup.  This function deduplicates across all
    folds and partitions.

    Args:
        pickle_path: Path to ``classification_dataset_records.pickle``.

    Returns:
        Dict mapping GUID string to ``(subgroup_name, record_file_path)``.
    """
    with open(pickle_path, "rb") as f:
        folds = pickle.load(f)

    guid_info: Dict[str, Tuple[str, str]] = {}

    for fold_name, fold_data in folds.items():
        for partition, subgroups in fold_data.items():
            for subgroup, file_list in subgroups.items():
                for fpath in file_list:
                    guid = os.path.splitext(os.path.basename(fpath))[0]
                    if guid not in guid_info:
                        guid_info[guid] = (subgroup, fpath)

    return guid_info


def build_guid_dataframe(
    guid_info: Dict[str, Tuple[str, str]],
    labor_onset_map: Optional[Dict[str, float]] = None,
    run_mimo: bool = True,
    base_block_size: int = 3520,
    overlap_percentage: float = 1 / 22,
) -> pd.DataFrame:
    """Build a DataFrame with one row per unique GUID.

    Args:
        guid_info: Output of :func:`extract_unique_guids`.
        labor_onset_map: Optional GUID -> labor-onset-seconds mapping from
            :func:`load_labor_onset_data`.
        run_mimo: If ``True``, run :func:`prescreen_guid` on each GUID to
            obtain domain-start and segment-count information.  If ``False``,
            those columns are filled with NaN.
        base_block_size: Base block size passed to ``prescreen_guid``.
        overlap_percentage: Overlap fraction passed to ``prescreen_guid``.

    Returns:
        DataFrame sorted by ``(class_label, subgroup, guid)``.
    """
    rows: List[Dict[str, Any]] = []

    iterator = tqdm(
        guid_info.items(),
        desc="Processing GUIDs" if run_mimo else "Collecting GUIDs",
    )

    for guid, (subgroup, fpath) in iterator:
        meta = SUBGROUP_META.get(subgroup, {
            "class_label": "Unknown", "cs_label": False, "bg_label": False,
        })

        # --- Labor onset lookup (from CSV, independent of MIMO) ---
        labor_onset_hours = float("nan")
        if labor_onset_map:
            normalized = _normalize_guid(guid)
            lo_sec = labor_onset_map.get(normalized, float("nan"))
            if not math.isnan(lo_sec):
                labor_onset_hours = lo_sec / 3600.0

        # --- MIMO prescreen for domain-start info ---
        n_valid = float("nan")
        n_total = float("nan")
        min_ds_sec = float("nan")
        max_ds_sec = float("nan")
        estimated_hours = float("nan")
        n_post_delivery = float("nan")

        if run_mimo:
            result = prescreen_guid(
                fpath,
                base_block_size=base_block_size,
                overlap_percentage=overlap_percentage,
                labor_onset_map=labor_onset_map,
            )
            if not result.error:
                n_total = result.n_total_segments
                n_valid = result.n_valid_segments
                estimated_hours = result.estimated_valid_hours
                n_post_delivery = result.n_post_delivery
                if not math.isnan(result.domain_start_range[0]):
                    min_ds_sec = result.domain_start_range[0]
                if not math.isnan(result.domain_start_range[1]):
                    max_ds_sec = result.domain_start_range[1]
            else:
                print(f"  [WARN] {guid}: prescreen error: {result.error_msg}")

        # --- Compute TLO at min domain_start ---
        lo_sec_val = labor_onset_hours * 3600.0 if not math.isnan(labor_onset_hours) else float("nan")
        tlo_min_sec = (min_ds_sec - lo_sec_val
                       if not math.isnan(min_ds_sec) and not math.isnan(lo_sec_val)
                       else float("nan"))
        tlo_min_hours = tlo_min_sec / 3600.0 if not math.isnan(tlo_min_sec) else float("nan")

        rows.append({
            "guid": guid,
            "record_path": fpath,
            "subgroup": subgroup,
            "class_label": meta["class_label"],
            "cs_label": meta["cs_label"],
            "bg_label": meta["bg_label"],
            "n_total_segments": n_total,
            "n_valid_segments": n_valid,
            "min_domain_start_sec": min_ds_sec,
            "max_domain_start_sec": max_ds_sec,
            "min_domain_start_hours": min_ds_sec / 3600.0 if not math.isnan(min_ds_sec) else float("nan"),
            "max_domain_start_hours": max_ds_sec / 3600.0 if not math.isnan(max_ds_sec) else float("nan"),
            "estimated_valid_hours": estimated_hours,
            "n_post_delivery": n_post_delivery,
            "labor_onset_hours": labor_onset_hours,
            "tlo_at_min_domain_start_sec": tlo_min_sec,
            "tlo_at_min_domain_start_hours": tlo_min_hours,
        })

    df = pd.DataFrame(rows)

    # Sort for readability
    class_order = {"Healthy": 0, "Acidosis": 1, "HIE": 2, "Unknown": 3}
    df["_sort"] = df["class_label"].map(class_order)
    df = df.sort_values(["_sort", "subgroup", "guid"]).drop(columns=["_sort"])
    df = df.reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary table to stdout.

    Args:
        df: DataFrame returned by :func:`build_guid_dataframe`.
    """
    print("\n" + "=" * 80)
    print("CLASSIFICATION DATASET GUID SUMMARY")
    print("=" * 80)

    summary = (
        df.groupby(["class_label", "subgroup"])
        .agg(
            n_guids=("guid", "count"),
            n_valid_segs=("n_valid_segments", lambda x: int(x.sum()) if x.notna().any() else 0),
            min_ds_h=("min_domain_start_hours", "min"),
            max_ds_h=("max_domain_start_hours", "max"),
            tlo_min_h=("tlo_at_min_domain_start_hours", "min"),
            tlo_max_h=("tlo_at_min_domain_start_hours", "max"),
            pct_has_tlo=("labor_onset_hours", lambda x: f"{x.notna().mean():.0%}"),
        )
        .reset_index()
    )

    print(f"\n{'Class':<10} {'Subgroup':<25} {'GUIDs':>6} {'Segs':>6} "
          f"{'MinDS(h)':>9} {'MaxDS(h)':>9} {'TLO_min(h)':>11} {'TLO_max(h)':>11} {'HasTLO':>7}")
    print("-" * 105)

    for _, row in summary.iterrows():
        min_ds = f"{row['min_ds_h']:.1f}" if pd.notna(row["min_ds_h"]) else "N/A"
        max_ds = f"{row['max_ds_h']:.1f}" if pd.notna(row["max_ds_h"]) else "N/A"
        tlo_min = f"{row['tlo_min_h']:.1f}" if pd.notna(row["tlo_min_h"]) else "N/A"
        tlo_max = f"{row['tlo_max_h']:.1f}" if pd.notna(row["tlo_max_h"]) else "N/A"
        print(f"{row['class_label']:<10} {row['subgroup']:<25} {row['n_guids']:>6} "
              f"{row['n_valid_segs']:>6} {min_ds:>9} {max_ds:>9} "
              f"{tlo_min:>11} {tlo_max:>11} {row['pct_has_tlo']:>7}")

    # Totals
    print("-" * 105)
    n_total = len(df)
    n_has_tlo = df["labor_onset_hours"].notna().sum()
    print(f"{'TOTAL':<10} {'':<25} {n_total:>6} "
          f"{int(df['n_valid_segments'].sum()) if df['n_valid_segments'].notna().any() else 'N/A':>6}")
    print(f"\nGUIDs with labor onset data: {n_has_tlo}/{n_total} "
          f"({n_has_tlo / n_total:.0%})")
    print("=" * 80)


def run(
    pickle_path: str,
    labor_onset_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    skip_mimo: bool = False,
    base_block_size: int = 3520,
    overlap_percentage: float = 1 / 22,
) -> pd.DataFrame:
    """Run the full extraction pipeline.

    Args:
        pickle_path: Path to ``classification_dataset_records.pickle``.
        labor_onset_csv: Path to labor onset CSV with columns
            ``trace_guid`` and ``labor_onset_hours``.  ``None`` to skip.
        output_csv: Output CSV path.  Defaults to
            ``<pickle_dir>/classification_guids.csv``.
        skip_mimo: If ``True``, skip MIMO prescreen (domain-start and
            segment-count columns will be NaN).
        base_block_size: Base block size for MIMO ``prepare_data``.
        overlap_percentage: Overlap fraction for ``split_long``.

    Returns:
        The resulting DataFrame (also saved to ``output_csv``).
    """
    # --- Extract unique GUIDs from pickle ---
    guid_info = extract_unique_guids(pickle_path)
    print(f"Found {len(guid_info)} unique GUIDs in "
          f"{os.path.basename(pickle_path)}")

    # --- Load labor onset data ---
    labor_onset_map = None
    if labor_onset_csv:
        labor_onset_map = load_labor_onset_data(labor_onset_csv)

    # --- Build per-GUID DataFrame ---
    df = build_guid_dataframe(
        guid_info,
        labor_onset_map=labor_onset_map,
        run_mimo=not skip_mimo,
        base_block_size=base_block_size,
        overlap_percentage=overlap_percentage,
    )

    # --- Print summary ---
    print_summary(df)

    # --- Save CSV ---
    out_path = output_csv or os.path.join(
        os.path.dirname(os.path.abspath(pickle_path)),
        "classification_guids.csv",
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} GUIDs to {out_path}")

    return df


if __name__ == "__main__":
    # ---- Configure paths here and run directly ----
    PICKLE_PATH = r"/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours/classification_dataset_records.pickle"
    LABOR_ONSET_CSV = r"labor_onset_sample.csv"
    OUTPUT_CSV = r"classification_guids.csv"
    SKIP_MIMO = False
    BASE_BLOCK_SIZE = 3520
    OVERLAP_PERCENTAGE = 1 / 22

    run(
        pickle_path=PICKLE_PATH,
        labor_onset_csv=LABOR_ONSET_CSV,
        output_csv=OUTPUT_CSV,
        skip_mimo=SKIP_MIMO,
        base_block_size=BASE_BLOCK_SIZE,
        overlap_percentage=OVERLAP_PERCENTAGE,
    )
