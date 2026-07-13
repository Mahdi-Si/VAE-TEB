r"""Thin adapter exposing the empirical transfer-entropy tooling to the CMI comparison (S6-T03).

The CMI comparison (``analyses/cmi_comparison.py``) summarises each sample to a single
per-sample :math:`K_{\mathrm{raw}}`, so it aligns to empirical TE at the **patient (GUID)**
level rather than the epoch-by-epoch trajectory that :func:`te_kld_analysis.merge_te_kld`
consumes. This shim loads a precomputed IDTxl empirical-TE CSV and reduces it to a
``{guid: mean ite_valid}`` mapping, degrading gracefully to an empty mapping on any failure so
a missing or malformed CSV becomes a logged skip rather than a pipeline crash.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from loguru import logger


def load_empirical_te_by_guid(
    csv_path: Union[str, Path],
    *,
    te_col: str = "ite_valid",
    guid_col: str = "guid",
) -> Dict[str, float]:
    r"""Return ``{guid: mean empirical TE}`` from a precomputed IDTxl CSV.

    Args:
        csv_path: Path to the empirical-TE CSV (IDTxl output). Must carry ``guid_col`` and
            ``te_col`` columns.
        te_col: Column holding the empirical TE (default ``ite_valid``).
        guid_col: Column holding the patient GUID.

    Returns:
        A ``{guid: mean_te}`` dict, or an **empty** dict if the file is missing, unreadable, or
        lacks the required columns (a logged skip -- never raises).
    """
    try:
        import pandas as pd

        path = Path(csv_path)
        if not path.is_file():
            logger.warning(f"empirical-TE adapter: no CSV at {path}; skipping")
            return {}
        df = pd.read_csv(path)
        if guid_col not in df.columns or te_col not in df.columns:
            logger.warning(
                f"empirical-TE adapter: CSV missing '{guid_col}'/'{te_col}' "
                f"(has {list(df.columns)}); skipping"
            )
            return {}
        df = df[[guid_col, te_col]].dropna()
        if df.empty:
            logger.warning("empirical-TE adapter: no finite rows; skipping")
            return {}
        grouped = df.groupby(guid_col)[te_col].mean()
        return {str(g): float(v) for g, v in grouped.items()}
    except Exception as exc:  # noqa: BLE001 - a bad CSV must degrade, not crash the pipeline
        logger.warning(f"empirical-TE adapter: failed to load ({exc}); skipping")
        return {}
