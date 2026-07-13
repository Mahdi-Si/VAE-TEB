"""Load empirical Transfer Entropy data and match it to VAE-KLD outputs.

This module handles the IDTxl CSV exports produced by the clinical data
pipeline. It exposes three main operations:

1. :func:`load_te_data` — parses the empirical CSV (handling the ``#``
   prefixed header some IDTxl exports emit), normalises GUIDs, and filters
   by validity fraction.
2. :func:`fuzzy_time_match` — greedy 1-to-1 nearest-neighbour matching of
   TE epochs to KLD epochs within a ``max_gap_seconds`` tolerance
   (default ±5 min).
3. :func:`compute_data_quality_report` — GUID overlap, coverage, and
   time-gap diagnostics for the merged dataset.

Example:
    >>> from te_data_loader import load_te_data, fuzzy_time_match
    >>> te_df = load_te_data("te_record_epoch.csv")
    >>> merged = fuzzy_time_match(te_df, kld_df, max_gap_seconds=300.0)
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
    """Normalise a GUID string to 32-char uppercase hex (no dashes).

    Mirrors the convention used by ``hdf5_dataset/create_hdf5_dataset.py`` so
    GUIDs from different sources can be matched reliably.

    Args:
        guid_str: Raw GUID string (may contain dashes, mixed case, or
            whitespace).

    Returns:
        32-character uppercase hexadecimal string.
    """
    return (
        guid_str.strip()
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )


def _raise_on_duplicate_keys(
    df: pd.DataFrame,
    key_cols: List[str],
    dataset_name: str,
) -> None:
    """Raise if *df* contains duplicate rows for the specified key columns.

    Args:
        df: DataFrame to validate.
        key_cols: Column names that must uniquely identify a row.
        dataset_name: Human-readable name used in the error message.

    Raises:
        ValueError: When any ``key_cols`` tuple appears more than once.
    """
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
# Epoch-grid rounding (legacy, retained for exact-grid matching mode)
# ---------------------------------------------------------------------------


def round_domain_start(value: float, grid_spacing: int = 1200) -> int:
    """Round a ``domain_start`` value to the nearest epoch-grid boundary.

    The IDTxl CSV has a systematic -1 s offset (e.g. -30001 instead of
    -30000). Rounding to the nearest ``grid_spacing`` absorbs this offset
    and produces values that can be compared with HDF5 epoch fields when
    exact-grid matching is used instead of fuzzy nearest-neighbour
    matching.

    Args:
        value: Raw domain_start in seconds (negative = before delivery).
        grid_spacing: Grid spacing in seconds. Default 1200 (20 min
            epochs).

    Returns:
        Rounded domain_start as an integer.
    """
    return int(round(value / grid_spacing) * grid_spacing)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_te_data(
    csv_path: Union[str, Path],
    min_ite_valid_pc: float = 0.5,
    grid_spacing: int = 1200,
) -> pd.DataFrame:
    """Load empirical TE data, normalise GUIDs, round epochs, and filter.

    Handles both CSV formats seen in practice:

    - IDTxl exports with a ``#``-prefixed header line (e.g. the file at
      ``model/transformer/tr_testing/TE_analysis/te_record_epoch.csv``).
    - Plain CSVs whose first line is already a valid pandas header.

    Args:
        csv_path: Path to the IDTxl TE CSV file.
        min_ite_valid_pc: Minimum fraction of valid instantaneous TE samples
            required to keep an epoch. Epochs with
            ``ite_valid_pc < min_ite_valid_pc`` are dropped. Set to 0 to
            retain every row.
        grid_spacing: Epoch-grid spacing in seconds for the legacy
            ``domain_start_rounded`` column used by exact-grid matching.

    Returns:
        DataFrame with columns:
            ``guid``, ``domain_start``, ``domain_start_rounded``,
            ``ite_valid``, ``ite_valid_pc`` (if present), plus any
            auxiliary columns found in the CSV (``omnibus_te``, ``BD``,
            ``pH``, ``dataset_name``, ...).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"TE CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        first_line = f.readline()

    if first_line.startswith("#"):
        header = first_line.lstrip("#").strip().split(",")
        df = pd.read_csv(csv_path, skiprows=1, header=None, names=header)
    else:
        df = pd.read_csv(csv_path)

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

    n_before = len(df)
    df = df.dropna(subset=["domain_start", "ite_valid"]).copy()
    n_dropped_nan = n_before - len(df)

    if min_ite_valid_pc > 0.0 and "ite_valid_pc" in df.columns:
        df = df[df["ite_valid_pc"] >= min_ite_valid_pc].copy()

    if "dataset_name" in df.columns:
        empty_mask = df["dataset_name"].astype(str).str.strip().eq("")
        if empty_mask.any():
            df = df[~empty_mask].copy()

    df["domain_start_rounded"] = df["domain_start"].apply(
        lambda v: round_domain_start(v, grid_spacing)
    )

    _raise_on_duplicate_keys(df, ["guid", "domain_start"], "Empirical TE CSV")

    logger.info(
        f"Loaded {len(df)} TE epochs from {csv_path.name}, "
        f"dropped {n_dropped_nan} NaN / validity-failed rows, "
        f"{df['guid'].nunique()} unique GUIDs. "
        f"domain_start range: "
        f"[{df['domain_start'].min()}, {df['domain_start'].max()}]"
    )
    return df


def get_te_guids(csv_path: Union[str, Path]) -> List[str]:
    """Return unique normalised GUIDs present in the TE CSV.

    Reads only the GUID column so this is lightweight even for large
    files. Handles both plain and ``#``-prefixed IDTxl header lines.

    Args:
        csv_path: Path to the IDTxl TE CSV file.

    Returns:
        Sorted list of unique 32-character uppercase hex GUID strings.
    """
    csv_path = Path(csv_path)
    with open(csv_path, "r") as f:
        first_line = f.readline()

    if first_line.startswith("#"):
        header = first_line.lstrip("#").strip().split(",")
        df = pd.read_csv(
            csv_path, skiprows=1, header=None, names=header,
            usecols=["tracing_guid"],
        )
    else:
        df = pd.read_csv(csv_path, usecols=["tracing_guid"])

    guids = sorted(
        df["tracing_guid"].astype(str).apply(normalize_guid).unique().tolist()
    )
    logger.info(f"Found {len(guids)} unique GUIDs in {csv_path.name}.")
    return guids


# ---------------------------------------------------------------------------
# Fuzzy time matching
# ---------------------------------------------------------------------------


def fuzzy_time_match(
    te_df: pd.DataFrame,
    kld_df: pd.DataFrame,
    max_gap_seconds: float = 180.0,
    te_time_col: str = "domain_start",
    kld_time_col: str = "epoch",
) -> pd.DataFrame:
    """Match TE and KLD rows by nearest-neighbour time within each GUID.

    For each common GUID, enumerates all ``(te_idx, kld_idx)`` candidate
    pairs whose absolute time difference is ``<= max_gap_seconds``, then
    assigns them greedily in ascending-gap order so every row is used at
    most once (1-to-1 matching). Ties are broken on times then indices to
    make the result deterministic.

    Args:
        te_df: Empirical TE DataFrame (from :func:`load_te_data`).
        kld_df: KLD DataFrame (must contain ``guid`` and ``kld`` columns
            and a time column named ``kld_time_col``).
        max_gap_seconds: Maximum allowed absolute time difference in
            seconds. Default 300 s (±5 minutes).
        te_time_col: Time column name in ``te_df``.
        kld_time_col: Time column name in ``kld_df``.

    Returns:
        Merged DataFrame with columns from both inputs plus
        ``time_gap_seconds`` (actual time difference). Columns unique to
        TE keep their original names; KLD columns that collide with TE
        columns are suffixed with ``_kld``.

    Raises:
        ValueError: If the resulting frame somehow contains duplicate
            ``(guid, time)`` keys or a gap larger than ``max_gap_seconds``.
    """
    te_guids = set(te_df["guid"].unique())
    kld_guids = set(kld_df["guid"].unique())
    common_guids = sorted(te_guids & kld_guids)

    if not common_guids:
        logger.warning("No common GUIDs between TE and KLD data!")
        return pd.DataFrame()

    logger.info(
        f"Fuzzy matching: {len(common_guids)} common GUIDs, "
        f"max_gap={max_gap_seconds}s (±{max_gap_seconds / 60.0:.1f} min)"
    )

    matched_rows: List[Dict[str, Any]] = []

    for guid in common_guids:
        t_sub = te_df[te_df["guid"] == guid].reset_index(drop=True)
        k_sub = kld_df[kld_df["guid"] == guid].reset_index(drop=True)

        t_times = t_sub[te_time_col].values
        k_times = k_sub[kld_time_col].values

        candidates: List[tuple] = []
        for t_idx, t_t in enumerate(t_times):
            for k_idx, k_t in enumerate(k_times):
                gap = abs(float(t_t) - float(k_t))
                if gap <= max_gap_seconds:
                    candidates.append(
                        (gap, float(t_t), float(k_t), t_idx, k_idx)
                    )

        candidates.sort()
        used_t: set = set()
        used_k: set = set()

        for gap, _t_t, _k_t, t_idx, k_idx in candidates:
            if t_idx in used_t or k_idx in used_k:
                continue
            used_t.add(t_idx)
            used_k.add(k_idx)

            t_row = t_sub.iloc[t_idx].to_dict()
            k_row = k_sub.iloc[k_idx].to_dict()

            combined: Dict[str, Any] = dict(t_row)
            for k, v in k_row.items():
                if k == "guid":
                    continue
                if k in combined:
                    combined[f"{k}_kld"] = v
                else:
                    combined[k] = v
            combined["time_gap_seconds"] = gap
            matched_rows.append(combined)

        logger.debug(
            f"  GUID {guid[:8]}...: {len(used_t)} matches "
            f"(TE={len(t_sub)}, KLD={len(k_sub)})"
        )

    merged = pd.DataFrame(matched_rows)

    if len(merged) == 0:
        logger.warning("No time-matched pairs found within tolerance!")
        return merged

    _raise_on_duplicate_keys(merged, ["guid", te_time_col], "Merged TE/KLD")
    if kld_time_col != te_time_col and kld_time_col in merged.columns:
        _raise_on_duplicate_keys(
            merged, ["guid", kld_time_col], "Merged TE/KLD"
        )

    too_far = merged["time_gap_seconds"] > max_gap_seconds
    if too_far.any():
        raise ValueError(
            "Merged TE/KLD data contains matches beyond the allowed "
            f"time gap of {max_gap_seconds} seconds."
        )

    logger.info(
        f"Fuzzy matching result: {len(merged)} matched pairs across "
        f"{merged['guid'].nunique()} GUIDs. "
        f"Mean gap: {merged['time_gap_seconds'].mean():.1f}s, "
        f"max: {merged['time_gap_seconds'].max():.1f}s"
    )
    return merged


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------


def compute_data_quality_report(
    te_df: pd.DataFrame,
    kld_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    max_gap_seconds: float = 300.0,
) -> Dict[str, Any]:
    """Produce a comprehensive data-quality diagnostic report.

    Quantifies GUID overlap, per-GUID coverage, and time-gap statistics
    so the user can see at a glance whether the matching is finding most
    of the available data or silently dropping segments.

    Args:
        te_df: Full empirical TE DataFrame.
        kld_df: Full KLD DataFrame.
        merged_df: Merged DataFrame returned by :func:`fuzzy_time_match`
            or :func:`~te_kld_analysis.merge_te_kld`.
        max_gap_seconds: Tolerance used when building ``merged_df`` (for
            reporting only).

    Returns:
        Dict with diagnostic fields suitable for JSON export.
    """
    te_guids = set(te_df["guid"].unique())
    kld_guids = set(kld_df["guid"].unique())
    common_guids = te_guids & kld_guids

    report: Dict[str, Any] = {
        "te_total_epochs": len(te_df),
        "te_unique_guids": len(te_guids),
        "kld_total_segments": len(kld_df),
        "kld_unique_guids": len(kld_guids),
        "common_guids": len(common_guids),
        "common_guid_list": sorted(common_guids),
        "te_only_guids": len(te_guids - kld_guids),
        "kld_only_guids": len(kld_guids - te_guids),
        "matched_pairs": len(merged_df),
        "matched_guids": (
            int(merged_df["guid"].nunique()) if len(merged_df) > 0 else 0
        ),
        "max_gap_seconds": float(max_gap_seconds),
        "max_gap_minutes": float(max_gap_seconds / 60.0),
        "te_duplicate_guid_domain_start": int(
            te_df.duplicated(subset=["guid", "domain_start"]).sum()
        ),
        "kld_duplicate_guid_epoch": int(
            kld_df.duplicated(subset=["guid", "epoch"]).sum()
        ),
    }

    if "dataset_name" in te_df.columns:
        report["te_dataset_distribution"] = (
            te_df["dataset_name"].value_counts().to_dict()
        )

    if len(merged_df) > 0 and "time_gap_seconds" in merged_df.columns:
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

    coverage: Dict[str, Dict[str, Any]] = {}
    for guid in common_guids:
        n_te = len(te_df[te_df["guid"] == guid])
        n_kld = len(kld_df[kld_df["guid"] == guid])
        n_matched = (
            len(merged_df[merged_df["guid"] == guid])
            if len(merged_df) > 0 else 0
        )
        coverage[guid] = {
            "te_epochs": n_te,
            "kld_segments": n_kld,
            "matched": n_matched,
            "te_coverage_pct": (
                round(100 * n_matched / n_te, 1) if n_te > 0 else 0.0
            ),
            "kld_coverage_pct": (
                round(100 * n_matched / n_kld, 1) if n_kld > 0 else 0.0
            ),
        }
    report["per_guid_coverage"] = coverage

    logger.info(
        f"Data quality: {report['te_unique_guids']} TE GUIDs, "
        f"{report['kld_unique_guids']} KLD GUIDs, "
        f"{report['common_guids']} common, "
        f"{report['matched_pairs']} matched pairs"
    )
    return report
