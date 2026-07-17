"""Load and match model-based TE data with empirical TE data from IDTxl.

Handles GUID normalisation, ``#``-prefixed CSV headers, nearest-neighbour
fuzzy time matching, and data quality diagnostics.

Example:
    >>> from model.transformer.tr_testing.TE_analysis.te_data_loader import (
    ...     load_model_te_data, load_empirical_te_data, fuzzy_time_match,
    ... )
    >>> model_df = load_model_te_data("te_segment_data.csv")
    >>> emp_df = load_empirical_te_data("te_record_epoch.csv")
    >>> merged = fuzzy_time_match(model_df, emp_df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# GUID normalisation
# ---------------------------------------------------------------------------


def normalize_guid(guid_str: str) -> str:
    """Normalise a GUID string to 32-char uppercase hex without dashes.

    Args:
        guid_str: Raw GUID string (may contain dashes, mixed case, or
            whitespace).

    Returns:
        32-character uppercase hexadecimal string.
    """
    return guid_str.strip().upper().replace("-", "").replace("_", "").replace(" ", "")


def _raise_on_duplicate_keys(
    df: pd.DataFrame,
    key_cols: List[str],
    dataset_name: str,
) -> None:
    """Raise if *df* contains duplicate rows for the specified key columns."""
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not dup_mask.any():
        return

    dup_examples = (
        df.loc[dup_mask, key_cols]
        .drop_duplicates()
        .head(10)
        .to_dict(orient="records")
    )
    raise ValueError(
        f"{dataset_name} contains duplicate rows for keys {key_cols}. "
        f"Expected unique start times per GUID. Examples: {dup_examples}"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_model_te_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load model TE segment data from ``te_segment_data.csv``.

    Normalises GUIDs and ensures the ``epoch`` column is numeric.

    Args:
        csv_path: Path to the model segment CSV (produced by
            ``collect_te_latent_data`` in the testing pipeline).

    Returns:
        DataFrame with normalised GUIDs and all KL / residual columns.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Model TE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"guid", "epoch", "kl_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Model CSV missing columns: {missing}")

    df["guid"] = df["guid"].astype(str).apply(normalize_guid)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    n_nan = df["epoch"].isna().sum()
    if n_nan > 0:
        logger.warning(f"Dropped {n_nan} rows with non-numeric epoch values.")
        df = df.dropna(subset=["epoch"]).copy()

    _raise_on_duplicate_keys(df, ["guid", "epoch"], "Model TE CSV")

    logger.info(
        f"Loaded {len(df)} model segments from {csv_path.name}, "
        f"{df['guid'].nunique()} unique GUIDs."
    )
    return df


def load_empirical_te_data(
    csv_path: Union[str, Path],
    min_ite_valid_pc: float = 0.0,
) -> pd.DataFrame:
    """Load empirical TE data from an IDTxl CSV export.

    Handles the ``#`` prefix on the header line that IDTxl exports produce.
    Renames ``tracing_guid`` to ``guid`` and normalises GUIDs.

    Args:
        csv_path: Path to the IDTxl TE CSV (e.g. ``te_record_epoch.csv``).
        min_ite_valid_pc: Minimum ``ite_valid_pc`` fraction to keep an epoch.
            Set to 0.0 (default) to keep everything — important when sample
            size is small.

    Returns:
        DataFrame with normalised GUIDs, ``domain_start`` in seconds, and
        all empirical TE columns.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Empirical TE CSV not found: {csv_path}")

    # Read first line to handle # prefix
    with open(csv_path, "r") as f:
        first_line = f.readline()

    if first_line.startswith("#"):
        # Strip the '#' from header, read rest normally
        header = first_line.lstrip("#").strip().split(",")
        df = pd.read_csv(csv_path, skiprows=1, header=None, names=header)
    else:
        df = pd.read_csv(csv_path)

    # Rename tracing_guid -> guid
    if "tracing_guid" in df.columns:
        df = df.rename(columns={"tracing_guid": "guid"})

    if "guid" not in df.columns:
        raise ValueError(
            "Empirical CSV must have a 'tracing_guid' or 'guid' column."
        )

    required = {"domain_start", "ite_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Empirical CSV missing columns: {missing}")

    df["guid"] = df["guid"].astype(str).apply(normalize_guid)
    df["domain_start"] = pd.to_numeric(df["domain_start"], errors="coerce")
    df["ite_valid"] = pd.to_numeric(df["ite_valid"], errors="coerce")

    n_bad_time = df["domain_start"].isna().sum()
    n_bad_ite = df["ite_valid"].isna().sum()
    if n_bad_time > 0 or n_bad_ite > 0:
        logger.warning(
            "Dropped {} empirical rows with invalid domain_start or ite_valid.",
            int((df["domain_start"].isna() | df["ite_valid"].isna()).sum()),
        )
        df = df.dropna(subset=["domain_start", "ite_valid"]).copy()

    # Filter on validity if requested
    n_before = len(df)
    if min_ite_valid_pc > 0.0 and "ite_valid_pc" in df.columns:
        df = df[df["ite_valid_pc"] >= min_ite_valid_pc].copy()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                f"Dropped {n_dropped} epochs with "
                f"ite_valid_pc < {min_ite_valid_pc}."
            )

    # Drop rows with empty dataset_name (artifact rows)
    if "dataset_name" in df.columns:
        empty_mask = df["dataset_name"].astype(str).str.strip().eq("")
        n_empty = empty_mask.sum()
        if n_empty > 0:
            df = df[~empty_mask].copy()
            logger.info(f"Dropped {n_empty} rows with empty dataset_name.")

    _raise_on_duplicate_keys(df, ["guid", "domain_start"], "Empirical TE CSV")

    logger.info(
        f"Loaded {len(df)} empirical epochs from {csv_path.name}, "
        f"{df['guid'].nunique()} unique GUIDs."
    )
    return df


# ---------------------------------------------------------------------------
# Fuzzy time matching
# ---------------------------------------------------------------------------


def fuzzy_time_match(
    model_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    max_gap_seconds: float = 600.0,
    model_time_col: str = "epoch",
    empirical_time_col: str = "domain_start",
) -> pd.DataFrame:
    """Match model and empirical rows by nearest-neighbour time within each GUID.

    For each common GUID, pairs model segments with empirical epochs using
    greedy 1-to-1 nearest-neighbour matching: candidate pairs are sorted by
    absolute time gap and greedily assigned (each row used at most once).

    Args:
        model_df: Model TE DataFrame (from :func:`load_model_te_data`).
        empirical_df: Empirical TE DataFrame (from
            :func:`load_empirical_te_data`).
        max_gap_seconds: Maximum allowed absolute time difference (seconds)
            for a match.  Default 600 s (10 minutes).
        model_time_col: Time column name in *model_df*.
        empirical_time_col: Time column name in *empirical_df*.

    Returns:
        Merged DataFrame with columns from both sources plus
        ``time_gap_seconds`` showing the actual time difference.
        Model columns keep original names; empirical columns that collide
        get an ``_empirical`` suffix.
    """
    model_guids = set(model_df["guid"].unique())
    empirical_guids = set(empirical_df["guid"].unique())
    common_guids = sorted(model_guids & empirical_guids)

    if not common_guids:
        logger.warning("No common GUIDs between model and empirical data!")
        return pd.DataFrame()

    logger.info(
        f"Fuzzy matching: {len(common_guids)} common GUIDs, "
        f"max_gap={max_gap_seconds}s"
    )

    matched_rows: List[Dict[str, Any]] = []

    for guid in common_guids:
        m_sub = model_df[model_df["guid"] == guid].copy()
        e_sub = empirical_df[empirical_df["guid"] == guid].copy()

        m_times = m_sub[model_time_col].values
        e_times = e_sub[empirical_time_col].values

        # Build all candidate pairs with their time gaps
        candidates = []
        for m_idx, m_t in enumerate(m_times):
            for e_idx, e_t in enumerate(e_times):
                gap = abs(m_t - e_t)
                if gap <= max_gap_seconds:
                    candidates.append((gap, m_idx, e_idx))

        # Greedy 1-to-1 matching: sort by gap, assign closest first
        candidates.sort(
            key=lambda x: (x[0], m_times[x[1]], e_times[x[2]], x[1], x[2])
        )
        used_m = set()
        used_e = set()

        for gap, m_idx, e_idx in candidates:
            if m_idx in used_m or e_idx in used_e:
                continue
            used_m.add(m_idx)
            used_e.add(e_idx)

            m_row = m_sub.iloc[m_idx].to_dict()
            e_row = e_sub.iloc[e_idx].to_dict()

            # Merge: model columns keep names, empirical get suffix on collision
            combined = {}
            for k, v in m_row.items():
                combined[k] = v
            for k, v in e_row.items():
                if k == "guid":
                    continue  # already from model side
                if k in combined:
                    combined[k + "_empirical"] = v
                else:
                    combined[k] = v

            combined["time_gap_seconds"] = gap
            matched_rows.append(combined)

        logger.debug(
            f"  GUID {guid[:8]}...: {len(used_m)} matches "
            f"(model={len(m_sub)}, empirical={len(e_sub)})"
        )

    merged = pd.DataFrame(matched_rows)

    if len(merged) == 0:
        logger.warning("No time-matched pairs found within tolerance!")
        return merged

    _raise_on_duplicate_keys(merged, ["guid", model_time_col], "Merged TE comparison data")
    _raise_on_duplicate_keys(
        merged, ["guid", empirical_time_col], "Merged TE comparison data"
    )

    too_far = merged["time_gap_seconds"] > max_gap_seconds
    if too_far.any():
        raise ValueError(
            "Merged TE comparison data contains matches beyond the allowed "
            f"time gap of {max_gap_seconds} seconds."
        )

    logger.info(
        f"Fuzzy matching result: {len(merged)} matched pairs across "
        f"{merged['guid'].nunique()} GUIDs. "
        f"Mean time gap: {merged['time_gap_seconds'].mean():.1f}s, "
        f"max: {merged['time_gap_seconds'].max():.1f}s"
    )
    return merged


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------


def compute_data_quality_report(
    model_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    max_gap_seconds: float = 600.0,
) -> Dict[str, Any]:
    """Produce a comprehensive data quality diagnostic report.

    Args:
        model_df: Full model TE DataFrame.
        empirical_df: Full empirical TE DataFrame.
        merged_df: Merged DataFrame from :func:`fuzzy_time_match`.

    Returns:
        Dict with diagnostic fields for logging and JSON export.
    """
    model_guids = set(model_df["guid"].unique())
    empirical_guids = set(empirical_df["guid"].unique())
    common_guids = model_guids & empirical_guids

    report: Dict[str, Any] = {
        "model_total_segments": len(model_df),
        "model_unique_guids": len(model_guids),
        "empirical_total_epochs": len(empirical_df),
        "empirical_unique_guids": len(empirical_guids),
        "common_guids": len(common_guids),
        "common_guid_list": sorted(common_guids),
        "model_only_guids": len(model_guids - empirical_guids),
        "empirical_only_guids": len(empirical_guids - model_guids),
        "matched_pairs": len(merged_df),
        "matched_guids": int(merged_df["guid"].nunique()) if len(merged_df) > 0 else 0,
        "max_gap_seconds": float(max_gap_seconds),
        "max_gap_minutes": float(max_gap_seconds / 60.0),
        "model_duplicate_guid_epoch": int(
            model_df.duplicated(subset=["guid", "epoch"]).sum()
        ),
        "empirical_duplicate_guid_domain_start": int(
            empirical_df.duplicated(subset=["guid", "domain_start"]).sum()
        ),
    }

    # Model class distribution
    if "class_label" in model_df.columns:
        report["model_class_distribution"] = (
            model_df["class_label"].value_counts().to_dict()
        )

    # Empirical dataset distribution
    if "dataset_name" in empirical_df.columns:
        report["empirical_dataset_distribution"] = (
            empirical_df["dataset_name"].value_counts().to_dict()
        )

    # Per-GUID matching stats
    if len(merged_df) > 0:
        per_guid = merged_df.groupby("guid").agg(
            n_matched=("time_gap_seconds", "count"),
            mean_gap=("time_gap_seconds", "mean"),
            max_gap=("time_gap_seconds", "max"),
        ).reset_index()
        report["per_guid_matching"] = per_guid.to_dict(orient="records")

        report["time_gap_stats"] = {
            "mean": float(merged_df["time_gap_seconds"].mean()),
            "median": float(merged_df["time_gap_seconds"].median()),
            "max": float(merged_df["time_gap_seconds"].max()),
            "min": float(merged_df["time_gap_seconds"].min()),
            "std": float(merged_df["time_gap_seconds"].std()),
        }
        report["merged_duplicate_guid_epoch"] = int(
            merged_df.duplicated(subset=["guid", "epoch"]).sum()
        )
        report["merged_duplicate_guid_domain_start"] = int(
            merged_df.duplicated(subset=["guid", "domain_start"]).sum()
        )

        # Class distribution of matched data
        if "class_label" in merged_df.columns:
            report["matched_class_distribution"] = (
                merged_df["class_label"].value_counts().to_dict()
            )

    # Coverage: what fraction of each common GUID's segments were matched?
    coverage = {}
    for guid in common_guids:
        n_model = len(model_df[model_df["guid"] == guid])
        n_empirical = len(empirical_df[empirical_df["guid"] == guid])
        n_matched = len(merged_df[merged_df["guid"] == guid]) if len(merged_df) > 0 else 0
        coverage[guid] = {
            "model_segments": n_model,
            "empirical_epochs": n_empirical,
            "matched": n_matched,
            "model_coverage_pct": round(100 * n_matched / n_model, 1) if n_model > 0 else 0.0,
            "empirical_coverage_pct": round(100 * n_matched / n_empirical, 1) if n_empirical > 0 else 0.0,
        }
    report["per_guid_coverage"] = coverage

    # Log summary
    logger.info(
        f"Data quality: {report['model_unique_guids']} model GUIDs, "
        f"{report['empirical_unique_guids']} empirical GUIDs, "
        f"{report['common_guids']} common, "
        f"{report['matched_pairs']} matched pairs"
    )

    return report
