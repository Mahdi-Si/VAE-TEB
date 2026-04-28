"""Distance-based comparisons between empirical TE and model KLD.

This module complements the correlation-driven helpers in
``te_kld_analysis.py`` with metric / shape distances between the
empirical transfer-entropy trajectory and the model-side score
trajectories (per-step KLD, posterior drift, attention concentration,
PCA aggregates, ...).

The functions here are pure: they consume the merged DataFrame produced
by :func:`te_kld_analysis.merge_te_kld` (one row per matched epoch) and
return DataFrames / dicts that downstream visualisers can plot directly.

Design summary (all helpers respect the ``min_pairs`` floor and skip
GUIDs with too few matched epochs):

* :func:`compute_pair_residuals`     – per-pair (matched-epoch) residuals
                                        in raw, global-z or per-GUID-z space.
* :func:`compute_per_guid_distances` – Euclidean / RMSE / NRMSE / cosine /
                                        discrete-Frechet distances per
                                        GUID.
* :func:`compute_pooled_distances`   – pooled scalars over all epochs +
                                        mean-of-per-GUID summaries.
* :func:`pca_distance_search`        – sweep PC subsets (existing or
                                        refit) and rank by per-GUID and
                                        pooled Euclidean distance to TE.
* :func:`joint_te_kld_pca`           – joint PCA on stacked
                                        [TE, KLD, PCs] columns to expose
                                        shared variance modes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats


__all__ = [
    "compute_pair_residuals",
    "compute_per_guid_distances",
    "compute_pooled_distances",
    "discrete_frechet_distance",
    "joint_te_kld_pca",
    "pca_distance_search",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_time_col(df: pd.DataFrame) -> str:
    """Return the temporal ordering column present in ``df``.

    Prefers ``domain_start`` (TE side), falls back to ``epoch`` (KLD
    side), and finally to the first non-GUID column. Used to ensure
    per-GUID trajectories are always sorted in real time before
    distances are computed.
    """
    for cand in ("domain_start", "epoch"):
        if cand in df.columns:
            return cand
    cols = [c for c in df.columns if c != "guid"]
    if not cols:
        raise ValueError("DataFrame has no columns other than 'guid'")
    return cols[0]


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Numerically-safe z-score; constant arrays map to zeros."""
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 2:
        return np.zeros_like(arr)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=0))
    if std <= 1e-12:
        out = np.zeros_like(arr)
        out[~finite] = np.nan
        return out
    return (arr - mean) / std


def _normalise_per_guid(
    df: pd.DataFrame,
    col: str,
    mode: str,
) -> np.ndarray:
    """Return ``col`` after the requested normalisation, preserving NaNs.

    ``mode`` ∈ {``"raw"``, ``"zscore_global"``, ``"zscore_per_guid"``,
    ``"minmax_per_guid"``}.
    """
    values = df[col].to_numpy(dtype=float)
    if mode == "raw":
        return values
    if mode == "zscore_global":
        return _zscore(values)
    if mode == "zscore_per_guid":
        out = np.full_like(values, np.nan, dtype=float)
        for _guid, idx in df.groupby("guid").indices.items():
            out[idx] = _zscore(values[idx])
        return out
    if mode == "minmax_per_guid":
        out = np.full_like(values, np.nan, dtype=float)
        for _guid, idx in df.groupby("guid").indices.items():
            slice_vals = values[idx]
            finite = np.isfinite(slice_vals)
            if finite.sum() < 2:
                out[idx] = 0.0
                continue
            lo = float(np.nanmin(slice_vals))
            hi = float(np.nanmax(slice_vals))
            span = hi - lo
            if span <= 1e-12:
                out[idx] = 0.0
            else:
                out[idx] = (slice_vals - lo) / span
        return out
    raise ValueError(f"Unknown normalisation mode: {mode!r}")


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Return the elementwise ``isfinite`` AND of the input arrays."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _flip_pc_signs(
    loadings: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip each PC so its largest-|loading| entry is positive.

    sklearn and the NumPy SVD fallback pick the sign of each component
    independently, so loadings barplots and signed-sum aggregators can
    flip across runs / environments. Forcing the largest-magnitude
    loading to be positive gives a deterministic convention; both the
    loading row and its score column are flipped together so the fit is
    preserved.

    Args:
        loadings: ``(n_components, n_features)`` matrix.
        scores: ``(n_samples, n_components)`` matrix.

    Returns:
        ``(loadings, scores)`` with the sign convention applied.
    """
    if loadings.size == 0 or scores.size == 0:
        return loadings, scores
    loadings = np.array(loadings, copy=True)
    scores = np.array(scores, copy=True)
    n_components = loadings.shape[0]
    for k in range(n_components):
        row = loadings[k]
        if row.size == 0:
            continue
        max_idx = int(np.argmax(np.abs(row)))
        if row[max_idx] < 0:
            loadings[k] = -row
            if scores.shape[1] > k:
                scores[:, k] = -scores[:, k]
    return loadings, scores


def _argmin_row(df: pd.DataFrame, col: str) -> Optional[Dict[str, Any]]:
    """Return the row of ``df`` minimising ``col``, as a plain dict.

    Skips NaN entries. Returns ``None`` when the column is missing,
    empty, or entirely non-finite.
    """
    if df.empty or col not in df.columns:
        return None
    arr = np.asarray(df[col].to_numpy(), dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    masked = np.where(finite, arr, np.inf)
    idx = int(np.argmin(masked))
    row = df.iloc[idx]
    return {k: row[k] for k in df.columns}


def discrete_frechet_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Return the discrete Frechet distance between two 1-D curves.

    The discrete Frechet distance walks both sequences with
    monotonically advancing pointers and reports the minimum of the
    longest leash needed across all valid traversals. It is the natural
    non-elastic complement to DTW: it penalises sequences that have to
    skip large gaps to align, where DTW averages those gaps away.

    The implementation is pure NumPy / Python (O(N²) time and memory),
    so it adds no external dependency.

    Args:
        p: Float array of length ``N``.
        q: Float array of length ``M``.

    Returns:
        Frechet distance (non-negative). Returns ``nan`` when either
        input is empty after filtering for finite values.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.size == 0 or q.size == 0:
        return float("nan")

    n, m = len(p), len(q)
    ca = np.full((n, m), -1.0, dtype=float)

    # Iterative version of the recursion to avoid Python recursion limits.
    ca[0, 0] = abs(p[0] - q[0])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], abs(p[i] - q[0]))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], abs(p[0] - q[j]))
    for i in range(1, n):
        for j in range(1, m):
            step = min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1])
            ca[i, j] = max(step, abs(p[i] - q[j]))
    return float(ca[n - 1, m - 1])


# ---------------------------------------------------------------------------
# Pair-level residuals
# ---------------------------------------------------------------------------


def compute_pair_residuals(
    merged_df: pd.DataFrame,
    score_col: str,
    te_col: str = "ite_valid",
    *,
    mode: str = "zscore_per_guid",
) -> pd.DataFrame:
    """Augment ``merged_df`` with normalised values and residuals.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        score_col: Model-side score column (e.g. ``"kld"``,
            ``"kld_pc1"``, ``"kld_pca_l2_top3"``).
        te_col: Empirical TE column (default ``"ite_valid"``).
        mode: Normalisation strategy applied to each side independently.
            One of ``"raw"``, ``"zscore_global"``, ``"zscore_per_guid"``
            (default), ``"minmax_per_guid"``.

    Returns:
        Copy of ``merged_df`` augmented with three columns:
        ``f"{score_col}_norm"``, ``f"{te_col}_norm"``, and
        ``f"residual_{score_col}_vs_{te_col}"`` (= score_norm - te_norm).
    """
    if score_col not in merged_df.columns:
        raise KeyError(f"score_col {score_col!r} not in merged_df")
    if te_col not in merged_df.columns:
        raise KeyError(f"te_col {te_col!r} not in merged_df")

    out = merged_df.copy()
    out[f"{score_col}_norm"] = _normalise_per_guid(out, score_col, mode)
    out[f"{te_col}_norm"] = _normalise_per_guid(out, te_col, mode)
    out[f"residual_{score_col}_vs_{te_col}"] = (
        out[f"{score_col}_norm"] - out[f"{te_col}_norm"]
    )
    return out


# ---------------------------------------------------------------------------
# Per-GUID trajectory distances
# ---------------------------------------------------------------------------


def _per_guid_distance_row(
    guid: str,
    score_vals: np.ndarray,
    te_vals: np.ndarray,
) -> Dict[str, Any]:
    """Compute the family of trajectory distances for one GUID."""
    finite = _finite_mask(score_vals, te_vals)
    n = int(finite.sum())
    if n < 2:
        return {
            "guid": guid,
            "n_pairs": n,
            "euclidean": float("nan"),
            "rmse": float("nan"),
            "nrmse": float("nan"),
            "cosine": float("nan"),
            "frechet_discrete": float("nan"),
            "pearson_r": float("nan"),
        }
    s = score_vals[finite]
    t = te_vals[finite]
    diff = s - t
    euclidean = float(np.linalg.norm(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    te_span = float(np.nanmax(t) - np.nanmin(t)) if n >= 2 else 0.0
    nrmse = float(rmse / te_span) if te_span > 1e-12 else float("nan")
    s_norm = float(np.linalg.norm(s))
    t_norm = float(np.linalg.norm(t))
    if s_norm > 1e-12 and t_norm > 1e-12:
        cosine = float(np.dot(s, t) / (s_norm * t_norm))
    else:
        cosine = float("nan")
    frechet = discrete_frechet_distance(s, t)
    if n >= 3 and np.std(s) > 1e-12 and np.std(t) > 1e-12:
        pearson_r = float(np.corrcoef(s, t)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "guid": guid,
        "n_pairs": n,
        "euclidean": euclidean,
        "rmse": rmse,
        "nrmse": nrmse,
        "cosine": cosine,
        "frechet_discrete": frechet,
        "pearson_r": pearson_r,
    }


def compute_per_guid_distances(
    merged_df: pd.DataFrame,
    score_col: str,
    te_col: str = "ite_valid",
    *,
    normalize: str = "zscore_per_guid",
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Per-GUID trajectory distances between empirical TE and a score.

    Each GUID's matched epochs are sorted by ``domain_start`` (or the
    next available temporal column), normalised per the ``normalize``
    flag, and compared with five distance measures. Per-GUID Pearson
    is included as an interpretation aid.

    Args:
        merged_df: Output of :func:`merge_te_kld`. Must contain ``guid``,
            ``score_col``, and ``te_col``.
        score_col: Model-side score column name.
        te_col: Empirical TE column.
        normalize: ``"raw"``, ``"zscore_global"``, ``"zscore_per_guid"``
            (default), or ``"minmax_per_guid"``.
        min_pairs: Skip GUIDs with fewer matched epochs than this.

    Returns:
        DataFrame with columns ``guid, n_pairs, euclidean, rmse, nrmse,
        cosine, frechet_discrete, pearson_r``. ``label`` is propagated
        when present in ``merged_df``.
    """
    if score_col not in merged_df.columns or te_col not in merged_df.columns:
        return pd.DataFrame(columns=[
            "guid", "n_pairs", "euclidean", "rmse", "nrmse",
            "cosine", "frechet_discrete", "pearson_r",
        ])
    augmented = compute_pair_residuals(
        merged_df, score_col, te_col, mode=normalize,
    )
    time_col = _resolve_time_col(augmented)
    score_norm = f"{score_col}_norm"
    te_norm = f"{te_col}_norm"

    has_label = "label" in augmented.columns
    rows: List[Dict[str, Any]] = []
    for guid, group in augmented.groupby("guid"):
        if len(group) < min_pairs:
            continue
        ordered = group.sort_values(time_col)
        score_vals = ordered[score_norm].to_numpy(dtype=float)
        te_vals = ordered[te_norm].to_numpy(dtype=float)
        row = _per_guid_distance_row(str(guid), score_vals, te_vals)
        if has_label:
            label_vals = np.asarray(ordered["label"].to_numpy())
            mask = np.asarray(pd.notna(label_vals))
            finite_label = label_vals[mask]
            if finite_label.size > 0:
                first = finite_label[0]
                try:
                    row["label"] = int(first)
                except (TypeError, ValueError):
                    row["label"] = None
            else:
                row["label"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled (cross-GUID) distance metrics
# ---------------------------------------------------------------------------


def compute_pooled_distances(
    merged_df: pd.DataFrame,
    score_col: str,
    te_col: str = "ite_valid",
    *,
    normalize: str = "zscore_per_guid",
    min_pairs: int = 5,
) -> Dict[str, Any]:
    """Pooled distance scalars across all matched epochs.

    Combines two views:

    * a true *pooled* view that concatenates every matched epoch into a
      single pair of arrays (uses the elected normalisation), and
    * a *macro* view that averages the per-GUID distances. The macro
      view dampens GUIDs with many epochs and is closer to clinical
      patient-level reporting.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        score_col: Model-side score column.
        te_col: Empirical TE column.
        normalize: Same options as :func:`compute_per_guid_distances`.
        min_pairs: GUIDs with fewer matched epochs are excluded from the
            macro-mean view. Forwarded to
            :func:`compute_per_guid_distances` so the macro statistics
            and the CSV stay consistent. Defaults to ``5`` to match the
            per-GUID default.

    Returns:
        Flat dict that JSON-serialises cleanly:
        ``pooled_euclidean``, ``pooled_rmse``, ``pooled_nrmse``,
        ``pooled_cosine``, ``pooled_frechet_discrete``,
        ``macro_mean_euclidean``, ``macro_mean_rmse``,
        ``macro_mean_nrmse``, ``macro_mean_cosine``,
        ``macro_mean_frechet_discrete``, ``n_pairs``, ``n_guids``.
    """
    if score_col not in merged_df.columns or te_col not in merged_df.columns:
        return {"error": "missing_columns",
                "score_col": score_col, "te_col": te_col}

    augmented = compute_pair_residuals(
        merged_df, score_col, te_col, mode=normalize,
    )
    score_norm = augmented[f"{score_col}_norm"].to_numpy(dtype=float)
    te_norm = augmented[f"{te_col}_norm"].to_numpy(dtype=float)
    finite = _finite_mask(score_norm, te_norm)
    s = score_norm[finite]
    t = te_norm[finite]

    if s.size < 2:
        pooled = {k: float("nan") for k in (
            "pooled_euclidean", "pooled_rmse", "pooled_nrmse",
            "pooled_cosine", "pooled_frechet_discrete",
        )}
    else:
        diff = s - t
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        te_span = float(np.nanmax(t) - np.nanmin(t))
        s_norm = float(np.linalg.norm(s))
        t_norm = float(np.linalg.norm(t))
        cosine = (
            float(np.dot(s, t) / (s_norm * t_norm))
            if s_norm > 1e-12 and t_norm > 1e-12 else float("nan")
        )
        pooled = {
            "pooled_euclidean": float(np.linalg.norm(diff)),
            "pooled_rmse": rmse,
            "pooled_nrmse": (
                float(rmse / te_span) if te_span > 1e-12 else float("nan")
            ),
            "pooled_cosine": cosine,
            "pooled_frechet_discrete": discrete_frechet_distance(s, t),
        }

    per_guid_df = compute_per_guid_distances(
        merged_df, score_col, te_col,
        normalize=normalize, min_pairs=min_pairs,
    )
    macro = {}
    if not per_guid_df.empty:
        for col in (
            "euclidean", "rmse", "nrmse", "cosine", "frechet_discrete",
        ):
            macro[f"macro_mean_{col}"] = float(
                np.nanmean(per_guid_df[col].to_numpy(dtype=float))
            )
    else:
        for col in (
            "euclidean", "rmse", "nrmse", "cosine", "frechet_discrete",
        ):
            macro[f"macro_mean_{col}"] = float("nan")

    return {
        **pooled,
        **macro,
        "n_pairs": int(finite.sum()),
        "n_guids": int(merged_df["guid"].nunique()),
        "n_guids_evaluated": int(len(per_guid_df)),
        "normalize": normalize,
    }


# ---------------------------------------------------------------------------
# PCA-Euclidean: search the best low-dim KLD subspace
# ---------------------------------------------------------------------------


def _detect_existing_pc_columns(df: pd.DataFrame, max_components: int) -> List[str]:
    """Return ``kld_pcN`` columns sorted by N (excludes selected aliases)."""
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        if not c.startswith("kld_pc"):
            continue
        if c.startswith("kld_pc_selected_") or c.startswith("kld_pca_"):
            continue
        suffix = c[len("kld_pc"):]
        if suffix.isdigit():
            cols.append((int(suffix), c))
    cols.sort()
    return [c for _, c in cols[:max_components]]


def _detect_kld_dim_columns(df: pd.DataFrame) -> List[str]:
    """Return per-dimension KLD columns sorted by dim index."""
    cols: List[Tuple[int, str]] = []
    for c in df.columns:
        if c.startswith("kld_dim_"):
            suffix = c[len("kld_dim_"):]
        elif c.startswith("kld_per_dim_"):
            suffix = c[len("kld_per_dim_"):]
        else:
            continue
        if suffix.isdigit():
            cols.append((int(suffix), c))
    cols.sort()
    return [c for _, c in cols]


def _build_subset_trajectories(
    df: pd.DataFrame,
    pc_cols: Sequence[str],
    aggregator: str,
) -> np.ndarray:
    """Aggregate a subset of PC columns into a single per-row trajectory."""
    if not pc_cols:
        return np.full(len(df), np.nan, dtype=float)
    arr = df[list(pc_cols)].to_numpy(dtype=float)
    if aggregator == "l2":
        return np.sqrt(np.nansum(arr ** 2, axis=1))
    if aggregator == "signed_sum":
        return np.nansum(arr, axis=1)
    if aggregator == "abs_sum":
        return np.nansum(np.abs(arr), axis=1)
    raise ValueError(f"Unknown aggregator {aggregator!r}")


def _rank_pcs_by_te_corr(
    df: pd.DataFrame,
    pc_cols: Sequence[str],
    te_col: str,
) -> List[str]:
    """Rank PC columns by descending |Spearman(PC, TE)|."""
    rhos: List[Tuple[float, str]] = []
    te_vals = df[te_col].to_numpy(dtype=float)
    for col in pc_cols:
        pc_vals = df[col].to_numpy(dtype=float)
        m = _finite_mask(pc_vals, te_vals)
        if m.sum() < 5 or np.std(pc_vals[m]) <= 1e-12 or np.std(te_vals[m]) <= 1e-12:
            rhos.append((0.0, col))
            continue
        rho = sp_stats.spearmanr(pc_vals[m], te_vals[m]).statistic
        rhos.append((float(abs(rho)) if np.isfinite(rho) else 0.0, col))
    rhos.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in rhos]


def _refit_pca_from_dims(
    df: pd.DataFrame,
    dim_cols: Sequence[str],
    n_components: int,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Fit PCA on per-dim KLD; return df augmented with refit_pcN columns.

    The refit uses sklearn when available, otherwise falls back to a
    dense SVD on the centred matrix. Returns ``(df_with_pcs, meta)`` or
    ``(None, None)`` when there are not enough finite rows to fit.
    """
    arr = df[list(dim_cols)].to_numpy(dtype=float)
    finite = np.all(np.isfinite(arr), axis=1)
    arr_finite = arr[finite]
    if arr_finite.shape[0] < max(5, n_components + 1):
        logger.warning(
            f"_refit_pca_from_dims: only {arr_finite.shape[0]} finite rows; "
            f"need >= {max(5, n_components + 1)}. Skipping refit."
        )
        return None, None

    n_components = min(n_components, arr_finite.shape[1], arr_finite.shape[0] - 1)
    try:
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=n_components)
        pca.fit(arr_finite)
        loadings = pca.components_
        evr = pca.explained_variance_ratio_
        mean_ = pca.mean_
        scores_finite_arr = pca.transform(arr_finite)
        backend = "sklearn"
    except Exception:  # noqa: BLE001
        mean_ = arr_finite.mean(axis=0)
        centred = arr_finite - mean_
        u, s, vh = np.linalg.svd(centred, full_matrices=False)
        loadings = vh[:n_components]
        evr_full = (s ** 2) / np.maximum((centred.shape[0] - 1) * np.var(centred, axis=0).sum(), 1e-12)
        evr = evr_full[:n_components]
        scores_finite_arr = u[:, :n_components] * s[:n_components]
        backend = "numpy_svd"

    # Pin component signs so loadings / signed-sum aggregators are
    # reproducible across sklearn / NumPy backends and across runs.
    loadings, scores_finite_arr = _flip_pc_signs(loadings, scores_finite_arr)

    scores_full = np.full((arr.shape[0], n_components), np.nan, dtype=float)
    scores_full[finite] = scores_finite_arr

    out = df.copy()
    refit_cols: List[str] = []
    for k in range(n_components):
        col = f"refit_pc{k + 1}"
        out[col] = scores_full[:, k]
        refit_cols.append(col)

    meta = {
        "backend": backend,
        "n_components": int(n_components),
        "explained_variance_ratio": [float(x) for x in evr],
        "n_finite_rows": int(arr_finite.shape[0]),
        "refit_columns": refit_cols,
        "source_dim_columns": list(dim_cols),
        # Loadings are sign-pinned (largest |loading| in each PC is positive)
        # so the row order matches ``source_dim_columns``.
        "loadings": [[float(v) for v in row] for row in loadings],
        "sign_convention": "max_abs_loading_positive",
    }
    return out, meta


def _evaluate_subset(
    df: pd.DataFrame,
    pc_cols: Sequence[str],
    te_col: str,
    aggregator: str,
    normalize: str,
    min_pairs: int,
    label: str,
) -> Dict[str, Any]:
    """Materialise a synthetic score column for the subset and compute distances."""
    score_col = f"__pca_subset_{label}"
    df_aug = df.copy()
    df_aug[score_col] = _build_subset_trajectories(df_aug, pc_cols, aggregator)
    pooled = compute_pooled_distances(
        df_aug, score_col, te_col,
        normalize=normalize, min_pairs=min_pairs,
    )
    per_guid = compute_per_guid_distances(
        df_aug, score_col, te_col, normalize=normalize, min_pairs=min_pairs,
    )
    return {
        "label": label,
        "members": list(pc_cols),
        "k": int(len(pc_cols)),
        "aggregator": aggregator,
        "pooled_euclidean": pooled.get("pooled_euclidean", float("nan")),
        "pooled_rmse": pooled.get("pooled_rmse", float("nan")),
        "pooled_cosine": pooled.get("pooled_cosine", float("nan")),
        "pooled_frechet_discrete": pooled.get(
            "pooled_frechet_discrete", float("nan"),
        ),
        "macro_mean_euclidean": pooled.get(
            "macro_mean_euclidean", float("nan"),
        ),
        "macro_mean_rmse": pooled.get("macro_mean_rmse", float("nan")),
        "macro_mean_cosine": pooled.get("macro_mean_cosine", float("nan")),
        "macro_mean_frechet_discrete": pooled.get(
            "macro_mean_frechet_discrete", float("nan"),
        ),
        "mean_pearson_r": float(
            np.nanmean(per_guid["pearson_r"].to_numpy(dtype=float))
        ) if not per_guid.empty else float("nan"),
        "n_guids_evaluated": int(len(per_guid)),
    }


def pca_distance_search(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    *,
    max_components: int = 5,
    refit_if_dims_present: bool = True,
    normalize: str = "zscore_per_guid",
    min_pairs: int = 5,
) -> Dict[str, Any]:
    """Search PC subsets that minimise the Euclidean distance to TE.

    For every available source of principal components — the existing
    ``kld_pc1/2/...`` columns produced by ``collect_metrics()`` and, if
    per-dim KLD columns are present, a fresh PCA refit — this routine
    enumerates two ranking strategies:

    * cumulative top-k by **variance** (the natural PCA order), and
    * cumulative top-k by **|Spearman(PC, TE)|** (which PCs explain the
      empirical signal best).

    Each enumerated subset is condensed via L2 norm and signed sum,
    standardised per the ``normalize`` flag, and scored with
    :func:`compute_pooled_distances` plus :func:`compute_per_guid_distances`.

    Args:
        merged_df: Output of :func:`merge_te_kld` augmented with
            ``kld_pcN`` and/or ``kld_dim_*`` columns.
        te_col: Empirical TE column.
        max_components: Maximum number of components to retain (capped
            by what is available on each side).
        refit_if_dims_present: When True (default), and ``kld_dim_*``
            columns exist, also refit a fresh PCA on those columns.
        normalize: Forwarded to the distance helpers.
        min_pairs: Minimum matched epochs per GUID for per-GUID stats.

    Returns:
        Dict with:

        * ``summary_df`` (``pd.DataFrame``): one row per subset_id. Columns:
          ``source`` (``"existing"``/``"refit"``), ``ranking``
          (``"variance"``/``"te_corr"``), ``aggregator`` (``"l2"``/
          ``"signed_sum"``), ``k``, ``members``, plus pooled / macro
          distances.
        * ``per_pc_distance_df`` (``pd.DataFrame``): one row per
          (source, GUID, PC) — the distance of the single-PC trajectory
          to TE. Useful for the per-PC heatmap.
        * ``selected`` (``dict``): the best subset by ``pooled_euclidean``
          and by ``macro_mean_euclidean`` (one entry per source).
        * ``meta`` (``dict``): info about column origins, available PCs,
          refit explained-variance, etc.
    """
    if te_col not in merged_df.columns:
        return {
            "summary_df": pd.DataFrame(),
            "per_pc_distance_df": pd.DataFrame(),
            "selected": {},
            "meta": {"error": f"missing te_col {te_col!r}"},
        }

    sources: List[Tuple[str, pd.DataFrame, List[str]]] = []

    existing_pcs = _detect_existing_pc_columns(merged_df, max_components)
    if existing_pcs:
        sources.append(("existing", merged_df, existing_pcs))

    refit_meta: Optional[Dict[str, Any]] = None
    refit_df: Optional[pd.DataFrame] = None
    if refit_if_dims_present:
        dim_cols = _detect_kld_dim_columns(merged_df)
        if dim_cols:
            refit_df, refit_meta = _refit_pca_from_dims(
                merged_df, dim_cols, max_components,
            )
            if refit_df is not None and refit_meta is not None:
                sources.append((
                    "refit", refit_df, list(refit_meta["refit_columns"]),
                ))

    if not sources:
        return {
            "summary_df": pd.DataFrame(),
            "per_pc_distance_df": pd.DataFrame(),
            "selected": {},
            "meta": {
                "warning": "no PCA inputs found",
                "existing_pc_columns": existing_pcs,
            },
        }

    summary_rows: List[Dict[str, Any]] = []
    per_pc_frames: List[pd.DataFrame] = []

    for source_name, df_src, pc_cols in sources:
        # Per-PC single-component distance (used by the per-PC heatmap).
        for pc in pc_cols:
            sub = compute_per_guid_distances(
                df_src, pc, te_col, normalize=normalize, min_pairs=min_pairs,
            )
            if sub.empty:
                continue
            sub.insert(0, "pc", pc)
            sub.insert(0, "source", source_name)
            per_pc_frames.append(sub)

        # Cumulative top-k subsets, two rankings, two aggregators.
        rank_variance = list(pc_cols)
        rank_te = _rank_pcs_by_te_corr(df_src, pc_cols, te_col)
        rankings = (("variance", rank_variance), ("te_corr", rank_te))
        for ranking_name, ordered_pcs in rankings:
            for k in range(1, len(ordered_pcs) + 1):
                subset = ordered_pcs[:k]
                for aggregator in ("l2", "signed_sum"):
                    label = f"{source_name}_{ranking_name}_{aggregator}_top{k}"
                    row = _evaluate_subset(
                        df_src, subset, te_col, aggregator,
                        normalize, min_pairs, label,
                    )
                    row["source"] = source_name
                    row["ranking"] = ranking_name
                    summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    per_pc_df = (
        pd.concat(per_pc_frames, ignore_index=True)
        if per_pc_frames else pd.DataFrame()
    )

    selected: Dict[str, Any] = {}
    if not summary_df.empty:
        for source_name in summary_df["source"].unique():
            sub_df = summary_df.loc[summary_df["source"] == source_name].copy()
            best_pooled = _argmin_row(sub_df, "pooled_euclidean")
            best_macro = _argmin_row(sub_df, "macro_mean_euclidean")
            selected[source_name] = {
                "best_pooled_euclidean": best_pooled,
                "best_macro_mean_euclidean": best_macro,
            }

    meta = {
        "sources": [src for src, *_ in sources],
        "existing_pc_columns": existing_pcs,
        "refit_meta": refit_meta,
        "te_col": te_col,
        "normalize": normalize,
    }
    return {
        "summary_df": summary_df,
        "per_pc_distance_df": per_pc_df,
        "selected": selected,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Joint TE-KLD PCA
# ---------------------------------------------------------------------------


def joint_te_kld_pca(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    *,
    n_components: int = 3,
    score_cols: Sequence[str] = ("kld", "kld_pc1", "kld_pc2", "kld_pc3"),
    standardize: bool = True,
) -> Dict[str, Any]:
    """Joint PCA on stacked [TE, score columns] to expose shared modes.

    Stacks ``te_col`` next to every available column in ``score_cols``
    (z-scored by default), drops rows with any NaN, and fits PCA.
    Loadings tell you how much each input variable contributes to each
    component; high |loading| on TE indicates a shared variance mode
    between empirical TE and the model surrogates.

    Args:
        merged_df: Output of :func:`merge_te_kld`.
        te_col: Empirical TE column.
        n_components: Maximum number of components to keep.
        score_cols: Candidate score columns; absent ones are skipped.
        standardize: When True (default), z-score each column before
            PCA so loadings are unit-free.

    Returns:
        Dict with ``explained_variance_ratio`` (list[float]),
        ``loadings_df`` (rows=columns, cols=PC1..PCk),
        ``scores_df`` (per-row PCA scores with the original ``guid`` /
        ``domain_start`` carried through), ``n_used`` (rows kept),
        ``columns_used`` (input column order). Returns
        ``{"error": "insufficient_data"}`` when fewer than 5 rows or
        2 columns are usable.
    """
    cols = [te_col] + [c for c in score_cols if c in merged_df.columns]
    cols = [c for c in cols if c in merged_df.columns]
    cols = list(dict.fromkeys(cols))  # dedupe, preserve order
    if len(cols) < 2:
        return {"error": "insufficient_columns", "columns_used": cols}

    sub = merged_df[cols].to_numpy(dtype=float)
    finite_rows = np.all(np.isfinite(sub), axis=1)
    sub_finite = sub[finite_rows]
    if sub_finite.shape[0] < max(5, n_components + 1):
        return {
            "error": "insufficient_rows",
            "n_used": int(sub_finite.shape[0]),
            "columns_used": cols,
        }

    if standardize:
        mean_ = sub_finite.mean(axis=0)
        std_ = sub_finite.std(axis=0, ddof=0)
        std_[std_ <= 1e-12] = 1.0
        z = (sub_finite - mean_) / std_
    else:
        mean_ = sub_finite.mean(axis=0)
        std_ = np.ones_like(mean_)
        z = sub_finite - mean_

    n_components = min(n_components, z.shape[1], z.shape[0] - 1)
    try:
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=n_components)
        pca.fit(z)
        loadings = pca.components_
        evr = pca.explained_variance_ratio_
        scores_finite = pca.transform(z)
        backend = "sklearn"
    except Exception:  # noqa: BLE001
        u, s, vh = np.linalg.svd(z, full_matrices=False)
        loadings = vh[:n_components]
        evr_full = (s ** 2) / np.maximum((z.shape[0] - 1) * np.var(z, axis=0).sum(), 1e-12)
        evr = evr_full[:n_components]
        scores_finite = (u[:, :n_components] * s[:n_components])
        backend = "numpy_svd"

    # Pin component signs so loadings (especially the TE row) and the
    # signed PC scores stay reproducible across runs / backends.
    loadings, scores_finite = _flip_pc_signs(loadings, scores_finite)

    scores_full = np.full(
        (merged_df.shape[0], n_components), np.nan, dtype=float,
    )
    scores_full[finite_rows] = scores_finite

    pc_names = [f"PC{i + 1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(
        loadings.T,
        index=cols,
        columns=pc_names,
    ).rename_axis("variable").reset_index()

    scores_df = pd.DataFrame(scores_full, columns=pc_names)
    for tag in ("guid", "domain_start", "epoch", "label", te_col):
        if tag in merged_df.columns:
            scores_df.insert(0, tag, merged_df[tag].to_numpy())

    return {
        "explained_variance_ratio": [float(x) for x in evr],
        "loadings_df": loadings_df,
        "scores_df": scores_df,
        "n_used": int(sub_finite.shape[0]),
        "n_components": int(n_components),
        "columns_used": cols,
        "te_col": te_col,
        "mean": mean_.tolist(),
        "std": std_.tolist(),
        "backend": backend,
        "sign_convention": "max_abs_loading_positive",
    }
