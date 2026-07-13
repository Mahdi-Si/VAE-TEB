"""Dynamic Time Warping alignment with a Sakoe-Chiba band.

This module aligns per-GUID empirical TE trajectories to the
corresponding VAE-KLD trajectories using DTW, bounded by a
Sakoe-Chiba band whose half-width is expressed in real seconds
(default ``±5 min``). The warping path is captured so the runner can
compute alignment-aware correlations on the warped pairs, not just a
scalar distance.

Backends are tried in order:

1. ``dtw-python`` — full DTW with explicit ``sakoechiba`` window,
   exposes ``index1``/``index2`` for the warping path.
2. ``tslearn.metrics.dtw_path`` — same algorithm, different API
   (returns ``(path, distance)``).
3. Euclidean truncation fallback — used only when neither library is
   importable. The band is NOT enforced in this mode and a warning is
   emitted.

Example:
    >>> from te_dtw import dtw_align_per_guid, paired_dataset_from_dtw
    >>> dtw_res = dtw_align_per_guid(te_df, kld_df, common_guids)
    >>> aligned = paired_dataset_from_dtw(dtw_res, te_df, kld_df)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger


# Probe optional DTW backends once at import time.
try:  # pragma: no cover - import-time dependency detection
    from dtw import dtw as _dtw_python  # type: ignore
    _HAVE_DTW_PYTHON = True
except ImportError:
    _dtw_python = None  # type: ignore[assignment]
    _HAVE_DTW_PYTHON = False

try:  # pragma: no cover
    from tslearn.metrics import dtw_path as _tslearn_dtw_path  # type: ignore
    _HAVE_TSLEARN = True
except ImportError:
    _tslearn_dtw_path = None  # type: ignore[assignment]
    _HAVE_TSLEARN = False


def dtw_backend_name() -> str:
    """Return the backend name that :func:`dtw_align_per_guid` will use.

    Returns:
        ``"dtw-python"`` if available, else ``"tslearn"``, else
        ``"euclidean"`` (unbounded fallback).
    """
    if _HAVE_DTW_PYTHON:
        return "dtw-python"
    if _HAVE_TSLEARN:
        return "tslearn"
    return "euclidean"


def _band_steps(
    t_times: np.ndarray, k_times: np.ndarray, band_seconds: float
) -> int:
    """Convert a band half-width in seconds to a step count.

    Uses the median sampling interval from whichever side has more
    samples — DTW's step-based band radius must cover the coarser of
    the two signals for the tolerance to hold.

    Args:
        t_times: TE timestamps in seconds.
        k_times: KLD timestamps in seconds.
        band_seconds: Desired half-width in seconds.

    Returns:
        Integer step radius (``>= 1``).
    """
    if band_seconds is None or band_seconds <= 0:
        return 0

    def _median_dt(times: np.ndarray) -> float:
        if len(times) < 2:
            return float("inf")
        diffs = np.abs(np.diff(np.sort(times)))
        diffs = diffs[diffs > 0]
        return float(np.median(diffs)) if diffs.size > 0 else float("inf")

    dt_t = _median_dt(t_times)
    dt_k = _median_dt(k_times)
    median_dt = min(dt_t, dt_k)  # smaller dt => more steps fit in the band
    if not math.isfinite(median_dt) or median_dt <= 0:
        return 0
    return max(1, int(math.ceil(float(band_seconds) / median_dt)))


def _run_dtw_backend(
    t_z: np.ndarray,
    k_z: np.ndarray,
    band_steps: int,
) -> Dict[str, Any]:
    """Run DTW on z-scored trajectories, returning distance + path.

    The band is enforced when a real DTW backend is available; the
    Euclidean fallback ignores it and warns.

    Args:
        t_z: TE z-scored values (length ``N_t``).
        k_z: KLD z-scored values (length ``N_k``).
        band_steps: Sakoe-Chiba half-width in steps. ``0`` means no
            band constraint.

    Returns:
        Dict with ``distance`` (float), ``path_te`` (np.ndarray[int]),
        ``path_kld`` (np.ndarray[int]), ``method`` (str).
    """
    if _HAVE_DTW_PYTHON and _dtw_python is not None:
        kwargs: Dict[str, Any] = {"keep_internals": False}
        if band_steps > 0:
            kwargs["window_type"] = "sakoechiba"
            kwargs["window_args"] = {"window_size": int(band_steps)}
        result = _dtw_python(t_z, k_z, **kwargs)
        return {
            "distance": float(result.distance),
            "path_te": np.asarray(result.index1, dtype=int),
            "path_kld": np.asarray(result.index2, dtype=int),
            "method": "dtw-python",
        }

    if _HAVE_TSLEARN and _tslearn_dtw_path is not None:
        kwargs = {}
        if band_steps > 0:
            kwargs["sakoe_chiba_radius"] = int(band_steps)
        path, dist = _tslearn_dtw_path(t_z, k_z, **kwargs)
        path_arr = np.asarray(path, dtype=int)
        return {
            "distance": float(dist),
            "path_te": path_arr[:, 0],
            "path_kld": path_arr[:, 1],
            "method": "tslearn",
        }

    logger.warning(
        "DTW libraries not available — falling back to Euclidean distance. "
        "The Sakoe-Chiba band is NOT enforced in this mode."
    )
    n_min = min(len(t_z), len(k_z))
    dist = float(np.sqrt(np.sum((t_z[:n_min] - k_z[:n_min]) ** 2)))
    idx = np.arange(n_min, dtype=int)
    return {
        "distance": dist,
        "path_te": idx,
        "path_kld": idx,
        "method": "euclidean",
    }


def dtw_align_per_guid(
    te_df: pd.DataFrame,
    kld_df: pd.DataFrame,
    common_guids: Sequence[str],
    te_col: str = "ite_valid",
    kld_col: str = "kld",
    te_time_col: str = "domain_start",
    kld_time_col: str = "epoch",
    band_seconds: float = 300.0,
    standardize: bool = True,
) -> Dict[str, Any]:
    """Align empirical TE and VAE-KLD trajectories per GUID via DTW.

    Uses the *complete* trajectory from each dataset (not just matched
    time points). Trajectories are z-scored before alignment so the
    shape, not the scale, drives the distance. A Sakoe-Chiba band of
    ``band_seconds`` is enforced when a DTW backend is available.

    Args:
        te_df: Full empirical TE DataFrame.
        kld_df: Full KLD DataFrame.
        common_guids: GUIDs present in both datasets (e.g.
            ``sorted(set(te_df.guid) & set(kld_df.guid))``).
        te_col: TE value column (default ``"ite_valid"``).
        kld_col: KLD value column (default ``"kld"``).
        te_time_col: TE time column (default ``"domain_start"``).
        kld_time_col: KLD time column (default ``"epoch"``).
        band_seconds: Sakoe-Chiba half-width in seconds. Default
            ``300`` (±5 min). Set to ``0`` or negative to disable.
        standardize: If True (default), z-score each trajectory before
            alignment.

    Returns:
        Dict with keys:
            - ``available`` (bool) — False if inputs are empty.
            - ``method`` — one of ``"dtw-python" | "tslearn" |
              "euclidean"`` (the last one is mixed — per-GUID entries
              may still report a different method if some had enough
              data and others did not).
            - ``band_seconds`` — value passed in.
            - ``per_guid`` — dict keyed by GUID with per-trajectory
              results (see below).
            - ``mean_distance`` / ``std_distance`` — aggregate summary.
            - ``backend`` — :func:`dtw_backend_name` snapshot.

        Each per-GUID entry contains ``distance``, ``normalized``,
        ``n_te``, ``n_kld``, ``band_seconds``, ``band_steps``,
        ``path_te``, ``path_kld``, ``te_times``, ``kld_times``,
        ``te_values``, ``kld_values``, ``te_z``, ``kld_z``, ``method``.
    """
    per_guid: Dict[str, Dict[str, Any]] = {}
    if not common_guids:
        return {
            "available": False,
            "reason": "no_common_guids",
            "per_guid": {},
            "band_seconds": float(band_seconds),
            "backend": dtw_backend_name(),
            "method": dtw_backend_name(),
        }

    methods_used: List[str] = []

    for guid in common_guids:
        t_sub = te_df[te_df["guid"] == guid].sort_values(te_time_col)
        k_sub = kld_df[kld_df["guid"] == guid].sort_values(kld_time_col)

        t_times = t_sub[te_time_col].values.astype(float)
        k_times = k_sub[kld_time_col].values.astype(float)
        t_vals = t_sub[te_col].values.astype(float)
        k_vals = k_sub[kld_col].values.astype(float)

        t_mask = np.isfinite(t_vals) & np.isfinite(t_times)
        k_mask = np.isfinite(k_vals) & np.isfinite(k_times)
        t_vals, t_times = t_vals[t_mask], t_times[t_mask]
        k_vals, k_times = k_vals[k_mask], k_times[k_mask]

        if len(t_vals) < 3 or len(k_vals) < 3:
            per_guid[guid] = {
                "distance": np.nan,
                "normalized": np.nan,
                "n_te": len(t_vals),
                "n_kld": len(k_vals),
                "band_seconds": float(band_seconds),
                "band_steps": 0,
                "path_te": np.array([], dtype=int),
                "path_kld": np.array([], dtype=int),
                "te_times": t_times,
                "kld_times": k_times,
                "te_values": t_vals,
                "kld_values": k_vals,
                "te_z": np.array([], dtype=float),
                "kld_z": np.array([], dtype=float),
                "method": "skipped",
            }
            continue

        if standardize:
            t_std = float(np.std(t_vals))
            k_std = float(np.std(k_vals))
            if t_std == 0.0 or k_std == 0.0:
                per_guid[guid] = {
                    "distance": np.nan,
                    "normalized": np.nan,
                    "n_te": len(t_vals),
                    "n_kld": len(k_vals),
                    "band_seconds": float(band_seconds),
                    "band_steps": 0,
                    "path_te": np.array([], dtype=int),
                    "path_kld": np.array([], dtype=int),
                    "te_times": t_times,
                    "kld_times": k_times,
                    "te_values": t_vals,
                    "kld_values": k_vals,
                    "te_z": np.array([], dtype=float),
                    "kld_z": np.array([], dtype=float),
                    "method": "skipped_zero_std",
                }
                continue
            t_z = (t_vals - float(np.mean(t_vals))) / t_std
            k_z = (k_vals - float(np.mean(k_vals))) / k_std
        else:
            t_z = t_vals.copy()
            k_z = k_vals.copy()

        band_steps = _band_steps(t_times, k_times, band_seconds)
        out = _run_dtw_backend(t_z, k_z, band_steps)
        methods_used.append(out["method"])

        max_len = max(len(t_vals), len(k_vals))
        per_guid[guid] = {
            "distance": out["distance"],
            "normalized": out["distance"] / max_len if max_len > 0 else np.nan,
            "n_te": len(t_vals),
            "n_kld": len(k_vals),
            "band_seconds": float(band_seconds),
            "band_steps": int(band_steps),
            "path_te": out["path_te"],
            "path_kld": out["path_kld"],
            "te_times": t_times,
            "kld_times": k_times,
            "te_values": t_vals,
            "kld_values": k_vals,
            "te_z": t_z,
            "kld_z": k_z,
            "method": out["method"],
        }

    distances = [
        v["distance"] for v in per_guid.values() if np.isfinite(v["distance"])
    ]

    overall_method = (
        max(set(methods_used), key=methods_used.count)
        if methods_used else dtw_backend_name()
    )

    summary: Dict[str, Any] = {
        "available": True,
        "backend": dtw_backend_name(),
        "method": overall_method,
        "band_seconds": float(band_seconds),
        "per_guid": per_guid,
        "mean_distance": (
            float(np.mean(distances)) if distances else float("nan")
        ),
        "std_distance": (
            float(np.std(distances)) if len(distances) > 1 else float("nan")
        ),
        "n_guids": len(per_guid),
    }
    logger.info(
        f"DTW alignment ({te_col} vs {kld_col}): backend={overall_method}, "
        f"band={band_seconds:.0f}s, "
        f"mean_dist={summary['mean_distance']:.4f} over "
        f"{len(distances)} GUIDs"
    )
    return summary


def paired_dataset_from_dtw(
    dtw_results: Dict[str, Any],
    te_df: Optional[pd.DataFrame] = None,
    kld_df: Optional[pd.DataFrame] = None,
    enforce_band: bool = True,
    band_seconds: Optional[float] = None,
) -> pd.DataFrame:
    """Convert DTW warping paths into a paired DataFrame.

    Each row corresponds to one warp step ``(i, j)`` — an alignment
    from ``te_df.iloc[i]`` to ``kld_df.iloc[j]`` after the per-GUID
    sort. This dataset feeds alignment-aware correlation: pair-wise
    Pearson / Spearman on DTW-matched values gives a different (and
    typically more optimistic) picture than the greedy 1-to-1 fuzzy
    match.

    Args:
        dtw_results: Output of :func:`dtw_align_per_guid`.
        te_df: Optional full TE DataFrame — unused internally because
            paths index per-GUID sorted arrays already stored inside
            ``dtw_results``. Kept as a parameter to mirror the planned
            API.
        kld_df: Same as ``te_df``.
        enforce_band: If True (default), drop pairs whose real-time gap
            exceeds ``band_seconds``. The step-based Sakoe-Chiba band
            is expressed in sample counts, not seconds, so this
            catches any pairs that slip through when sampling is very
            irregular.
        band_seconds: Override the band tolerance for the filter. If
            None, reuse ``dtw_results["band_seconds"]``.

    Returns:
        DataFrame with columns ``guid, te_time, kld_time,
        time_gap_seconds, te_value, kld_value, pair_kind``.
    """
    del te_df, kld_df  # mirror planned signature; unused.
    per_guid = dtw_results.get("per_guid", {}) or {}
    if not per_guid:
        return pd.DataFrame(
            columns=[
                "guid", "te_time", "kld_time", "time_gap_seconds",
                "te_value", "kld_value", "pair_kind",
            ]
        )

    if band_seconds is None:
        band_seconds = float(dtw_results.get("band_seconds", float("inf")))
    else:
        band_seconds = float(band_seconds)

    rows: List[Dict[str, Any]] = []
    for guid, entry in per_guid.items():
        path_te = entry.get("path_te")
        path_kld = entry.get("path_kld")
        if path_te is None or path_kld is None:
            continue
        if len(path_te) == 0 or len(path_kld) == 0:
            continue

        t_times = np.asarray(entry["te_times"], dtype=float)
        k_times = np.asarray(entry["kld_times"], dtype=float)
        t_vals = np.asarray(entry["te_values"], dtype=float)
        k_vals = np.asarray(entry["kld_values"], dtype=float)

        for i, j in zip(path_te, path_kld):
            i = int(i)
            j = int(j)
            if i >= len(t_times) or j >= len(k_times):
                continue
            gap = float(abs(t_times[i] - k_times[j]))
            if (
                enforce_band
                and math.isfinite(band_seconds)
                and gap > band_seconds
            ):
                continue
            rows.append({
                "guid": guid,
                "te_time": float(t_times[i]),
                "kld_time": float(k_times[j]),
                "time_gap_seconds": gap,
                "te_value": float(t_vals[i]),
                "kld_value": float(k_vals[j]),
                "pair_kind": "dtw",
            })

    df = pd.DataFrame(rows)
    logger.info(
        f"DTW paired dataset: {len(df)} aligned pairs across "
        f"{df['guid'].nunique() if len(df) > 0 else 0} GUIDs"
        f"{' (band filtered)' if enforce_band else ''}"
    )
    return df
