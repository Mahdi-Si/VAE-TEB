"""Load and preprocess empirical Transfer Entropy data from IDTxl CSV exports.

Handles GUID normalisation, epoch-grid rounding, and validity filtering so
the TE records can be matched against VAE-KLD outputs on a common
(guid, domain_start_rounded) key.

Example:
    >>> from te_data_loader import load_te_data, get_te_guids
    >>> te_df = load_te_data("te_record_epoch_HIE_NoCS.csv")
    >>> guids = get_te_guids("te_record_epoch_HIE_NoCS.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
from loguru import logger


def normalize_guid(guid_str: str) -> str:
    """Normalise a GUID string to 32-char uppercase hex (no dashes).

    Mirrors the convention used by ``hdf5_dataset/create_hdf5_dataset.py``
    so that GUIDs from different sources can be matched reliably.

    Args:
        guid_str: Raw GUID string (may contain dashes, mixed case, or
            whitespace).

    Returns:
        32-character uppercase hexadecimal string.
    """
    return guid_str.strip().upper().replace("-", "")


def round_domain_start(value: float, grid_spacing: int = 1200) -> int:
    """Round a domain_start value to the nearest epoch-grid boundary.

    The empirical TE CSV has a systematic -1 s offset (e.g. -30001 instead
    of -30000).  Rounding to the nearest ``grid_spacing`` absorbs this
    offset and produces values that can be compared with HDF5 epoch fields.

    Args:
        value: Raw domain_start in seconds (negative = before delivery).
        grid_spacing: Grid spacing in seconds.  Default 1200 (20 min epochs).

    Returns:
        Rounded domain_start as an integer.
    """
    return int(round(value / grid_spacing) * grid_spacing)


def load_te_data(
    csv_path: Union[str, Path],
    min_ite_valid_pc: float = 0.5,
    grid_spacing: int = 1200,
) -> pd.DataFrame:
    """Load empirical TE data, normalise GUIDs, round epochs, and filter.

    Args:
        csv_path: Path to the IDTxl TE CSV file (e.g.
            ``te_record_epoch_HIE_NoCS.csv``).
        min_ite_valid_pc: Minimum fraction of valid instantaneous TE
            samples required to keep an epoch.  Epochs with
            ``ite_valid_pc < min_ite_valid_pc`` are dropped.
        grid_spacing: Epoch-grid spacing in seconds for rounding
            ``domain_start``.

    Returns:
        DataFrame with columns:
            guid, domain_start, domain_start_rounded, ite_valid,
            omnibus_te, ite_valid_pc, BD, pH, dataset_name.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"TE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename tracing_guid -> guid for consistency with VAE outputs
    df = df.rename(columns={"tracing_guid": "guid"})

    # Normalise GUIDs
    df["guid"] = df["guid"].astype(str).apply(normalize_guid)

    # Round domain_start to grid
    df["domain_start_rounded"] = df["domain_start"].apply(
        lambda v: round_domain_start(v, grid_spacing)
    )

    n_before = len(df)
    df = df[df["ite_valid_pc"] >= min_ite_valid_pc].copy()
    n_dropped = n_before - len(df)
    logger.info(
        f"Loaded {n_before} TE epochs from {csv_path.name}, "
        f"dropped {n_dropped} with ite_valid_pc < {min_ite_valid_pc}, "
        f"kept {len(df)}."
    )
    logger.info(
        f"  Unique GUIDs after filter: {df['guid'].nunique()}, "
        f"domain_start range: [{df['domain_start'].min()}, "
        f"{df['domain_start'].max()}]"
    )

    return df


def get_te_guids(csv_path: Union[str, Path]) -> List[str]:
    """Return unique normalised GUIDs present in the TE CSV.

    Reads only the ``tracing_guid`` column, so this is lightweight even for
    large CSVs.

    Args:
        csv_path: Path to the IDTxl TE CSV file.

    Returns:
        Sorted list of unique 32-char uppercase hex GUID strings.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, usecols=["tracing_guid"])
    guids = sorted(
        df["tracing_guid"].astype(str).apply(normalize_guid).unique().tolist()
    )
    logger.info(f"Found {len(guids)} unique GUIDs in {csv_path.name}.")
    return guids
