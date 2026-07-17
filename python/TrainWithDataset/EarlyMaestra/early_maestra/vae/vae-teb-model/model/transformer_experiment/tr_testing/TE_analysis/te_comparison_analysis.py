"""Core statistical analyses comparing model-based TE with empirical TE.

All methods are chosen for validity at small sample sizes (N ~ 26, 3 GUIDs).
Uses permutation tests for exact p-values, block bootstrap for cluster-aware
CIs, and non-parametric rank-based measures throughout.

Example:
    >>> from model.transformer.tr_testing.TE_analysis.te_comparison_analysis import (
    ...     run_full_comparison,
    ... )
    >>> results = run_full_comparison(merged_df, output_dir)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Measure definitions
# ---------------------------------------------------------------------------

MODEL_TE_MEASURES: Dict[str, str] = {
    "kl_mean": "Mean KL divergence",
    "kl_max": "Max KL divergence",
    "residual_norm_mean_h8": "Residual norm (h=8)",
    "residual_norm_mean_h15": "Residual norm (h=15)",
    "residual_norm_mean_h30": "Residual norm (h=30)",
    "te_forecast_gain_mean": "TE forecast gain (all h)",
    "te_forecast_gain_mean_h8": "TE forecast gain (h=8)",
    "te_forecast_gain_mean_h15": "TE forecast gain (h=15)",
    "te_forecast_gain_mean_h30": "TE forecast gain (h=30)",
    "te_relative_gain_mean": "TE relative gain (all h)",
}

EMPIRICAL_TE_MEASURES: Dict[str, str] = {
    "ite_valid": "Valid instantaneous TE",
}


def _available_columns(
    df: pd.DataFrame,
    candidates: Dict[str, str],
) -> Dict[str, str]:
    """Filter measure dict to columns that exist in the DataFrame."""
    return {k: v for k, v in candidates.items() if k in df.columns}


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def compute_correlation_matrix(
    merged_df: pd.DataFrame,
    model_cols: Optional[List[str]] = None,
    empirical_cols: Optional[List[str]] = None,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise correlation matrix between model and empirical measures.

    Args:
        merged_df: Merged DataFrame from fuzzy time matching.
        model_cols: Model-side column names.  Defaults to keys of
            :data:`MODEL_TE_MEASURES` that exist in *merged_df*.
        empirical_cols: Empirical-side column names.  Defaults to keys of
            :data:`EMPIRICAL_TE_MEASURES` that exist in *merged_df*.
        method: ``"spearman"`` or ``"kendall"``.

    Returns:
        Tuple of ``(correlation_df, pvalue_df)`` with shape
        ``(len(model_cols), len(empirical_cols))``.
    """
    if model_cols is None:
        model_cols = list(_available_columns(merged_df, MODEL_TE_MEASURES).keys())
    if empirical_cols is None:
        empirical_cols = list(_available_columns(merged_df, EMPIRICAL_TE_MEASURES).keys())

    corr_vals = np.full((len(model_cols), len(empirical_cols)), np.nan)
    pval_vals = np.full((len(model_cols), len(empirical_cols)), np.nan)

    for i, mc in enumerate(model_cols):
        for j, ec in enumerate(empirical_cols):
            x = merged_df[mc].values.astype(float)
            y = merged_df[ec].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) < 3:
                continue

            if method == "spearman":
                r, p = sp_stats.spearmanr(x, y)
            elif method == "kendall":
                r, p = sp_stats.kendalltau(x, y)
            else:
                raise ValueError(f"Unknown method: {method}")

            corr_vals[i, j] = r
            pval_vals[i, j] = p

    corr_df = pd.DataFrame(corr_vals, index=model_cols, columns=empirical_cols)
    pval_df = pd.DataFrame(pval_vals, index=model_cols, columns=empirical_cols)

    logger.info(
        f"Correlation matrix ({method}): "
        f"{len(model_cols)} model x {len(empirical_cols)} empirical measures"
    )
    return corr_df, pval_df


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def permutation_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Permutation-based significance test for correlation.

    Permutes *y* to build a null distribution and reports the exact
    (two-sided) p-value.  Essential because parametric p-values are
    unreliable at N ~ 26.

    Args:
        x: First variable array.
        y: Second variable array (same length as *x*).
        method: ``"spearman"`` or ``"kendall"``.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``observed``, ``p_value``, ``null_mean``, ``null_std``,
        ``null_ci_lo``, ``null_ci_hi``.
    """
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"observed": np.nan, "p_value": np.nan, "error": "insufficient_data"}

    corr_fn = sp_stats.spearmanr if method == "spearman" else sp_stats.kendalltau
    observed = corr_fn(x, y).statistic

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_dist[i] = corr_fn(x, y_perm).statistic

    # Two-sided p-value
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))

    return {
        "observed": float(observed),
        "p_value": p_value,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "null_ci_lo": float(np.percentile(null_dist, 2.5)),
        "null_ci_hi": float(np.percentile(null_dist, 97.5)),
        "null_distribution": null_dist,
    }


# ---------------------------------------------------------------------------
# Within-GUID correlations
# ---------------------------------------------------------------------------


def compute_within_guid_correlations(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Compute per-GUID temporal Spearman correlation.

    For each GUID with at least *min_pairs* matched epochs, computes
    the Spearman correlation on time-ordered values.

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column name.
        empirical_col: Empirical measure column name.
        min_pairs: Minimum matched pairs required per GUID.

    Returns:
        DataFrame with one row per eligible GUID containing
        ``guid``, ``n_pairs``, ``spearman_rho``, ``spearman_p``,
        ``mean_model``, ``mean_empirical``.
    """
    records: List[Dict[str, Any]] = []
    time_col = "epoch" if "epoch" in merged_df.columns else merged_df.columns[1]

    for guid, group in merged_df.groupby("guid"):
        if len(group) < min_pairs:
            continue

        group = group.sort_values(time_col)
        x = group[model_col].values.astype(float)
        y = group[empirical_col].values.astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < min_pairs:
            continue

        if np.std(x) == 0 or np.std(y) == 0:
            rho, p = np.nan, np.nan
        else:
            rho, p = sp_stats.spearmanr(x, y)

        records.append({
            "guid": guid,
            "n_pairs": len(x),
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_p": float(p) if np.isfinite(p) else np.nan,
            "mean_model": float(np.mean(x)),
            "mean_empirical": float(np.mean(y)),
        })

    result = pd.DataFrame(records)
    logger.info(
        f"Within-GUID correlations ({model_col} vs {empirical_col}): "
        f"{len(result)} GUIDs with >= {min_pairs} pairs"
    )
    return result


# ---------------------------------------------------------------------------
# Concordance analysis
# ---------------------------------------------------------------------------


def concordance_analysis(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
) -> Dict[str, Any]:
    """Compute Kendall's tau-b and concordance statistics.

    Kendall's tau-b handles ties properly and is directly interpretable
    as the excess probability of concordance over discordance.

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.

    Returns:
        Dict with ``kendall_tau``, ``kendall_p``, ``n_concordant``,
        ``n_discordant``, ``n_tied``, ``n_pairs``.
    """
    x = merged_df[model_col].values.astype(float)
    y = merged_df[empirical_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"error": "insufficient_data", "n_pairs": len(x)}

    tau, p = sp_stats.kendalltau(x, y)

    # Count concordant / discordant pairs
    n = len(x)
    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            prod = dx * dy
            if prod > 0:
                n_concordant += 1
            elif prod < 0:
                n_discordant += 1
            else:
                n_tied += 1

    concordance_index = n_concordant / (n_concordant + n_discordant) if (n_concordant + n_discordant) > 0 else 0.5

    result = {
        "kendall_tau": float(tau),
        "kendall_p": float(p),
        "concordance_index": float(concordance_index),
        "n_concordant": n_concordant,
        "n_discordant": n_discordant,
        "n_tied": n_tied,
        "n_pairs": len(x),
    }
    logger.info(
        f"Concordance ({model_col} vs {empirical_col}): "
        f"tau={tau:.4f} (p={p:.2e}), C-index={concordance_index:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Trend agreement
# ---------------------------------------------------------------------------


def trend_agreement_analysis(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
) -> Dict[str, Any]:
    """Assess whether temporal derivatives agree in sign.

    For each GUID, computes first differences between consecutive matched
    time points.  Counts how often model and empirical deltas have the
    same sign (both increasing or both decreasing).

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.

    Returns:
        Dict with ``sign_agreement_rate``, ``n_transitions``,
        ``per_guid_agreement``, ``binomial_p`` (test vs 0.5 chance).
    """
    time_col = "epoch" if "epoch" in merged_df.columns else "domain_start"
    per_guid: Dict[str, Dict[str, Any]] = {}
    all_agrees = []

    for guid, group in merged_df.groupby("guid"):
        if len(group) < 2:
            continue

        group = group.sort_values(time_col)
        m_vals = group[model_col].values.astype(float)
        e_vals = group[empirical_col].values.astype(float)

        m_diff = np.diff(m_vals)
        e_diff = np.diff(e_vals)

        # Exclude zero-change transitions
        nonzero = (m_diff != 0) & (e_diff != 0)
        if nonzero.sum() == 0:
            continue

        m_sign = np.sign(m_diff[nonzero])
        e_sign = np.sign(e_diff[nonzero])
        agrees = (m_sign == e_sign).astype(int)

        all_agrees.extend(agrees.tolist())
        per_guid[guid] = {
            "n_transitions": int(nonzero.sum()),
            "n_agree": int(agrees.sum()),
            "agreement_rate": float(agrees.mean()),
        }

    if len(all_agrees) == 0:
        return {"error": "no_transitions", "n_transitions": 0}

    all_agrees_arr = np.array(all_agrees)
    overall_rate = float(all_agrees_arr.mean())
    n_total = len(all_agrees_arr)

    # Binomial test: is agreement rate significantly different from 0.5?
    n_agree = int(all_agrees_arr.sum())
    binom_p = float(sp_stats.binomtest(n_agree, n_total, 0.5).pvalue)

    result = {
        "sign_agreement_rate": overall_rate,
        "n_transitions": n_total,
        "n_agree": n_agree,
        "binomial_p": binom_p,
        "per_guid_agreement": per_guid,
    }
    logger.info(
        f"Trend agreement ({model_col} vs {empirical_col}): "
        f"{overall_rate:.1%} ({n_agree}/{n_total}), "
        f"binomial p={binom_p:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Per-dimension analysis
# ---------------------------------------------------------------------------


def per_dimension_analysis(
    merged_df: pd.DataFrame,
    empirical_cols: Optional[List[str]] = None,
    n_dims: int = 16,
    n_permutations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Correlate each KL dimension with each empirical TE measure.

    Identifies which latent dimensions carry the most coupling information
    by computing Spearman rho between ``kl_dim_mean_{d}`` and each
    empirical measure.

    Args:
        merged_df: Merged DataFrame.
        empirical_cols: Empirical measure columns.  Defaults to all
            available :data:`EMPIRICAL_TE_MEASURES`.
        n_dims: Number of KL dimensions (default 16).
        n_permutations: Permutations for significance testing.
        seed: Random seed.

    Returns:
        DataFrame with columns: ``dimension``, ``empirical_measure``,
        ``spearman_rho``, ``parametric_p``, ``permutation_p``,
        ``mean_kl``, ``std_kl``.
    """
    if empirical_cols is None:
        empirical_cols = list(
            _available_columns(merged_df, EMPIRICAL_TE_MEASURES).keys()
        )

    records = []
    rng = np.random.default_rng(seed)

    for d in range(n_dims):
        kl_col = f"kl_dim_mean_{d}"
        if kl_col not in merged_df.columns:
            continue

        x = merged_df[kl_col].values.astype(float)
        mask_x = np.isfinite(x)

        for ec in empirical_cols:
            y = merged_df[ec].values.astype(float)
            mask = mask_x & np.isfinite(y)
            xm, ym = x[mask], y[mask]

            if len(xm) < 3 or np.std(xm) == 0 or np.std(ym) == 0:
                records.append({
                    "dimension": d,
                    "empirical_measure": ec,
                    "spearman_rho": np.nan,
                    "parametric_p": np.nan,
                    "permutation_p": np.nan,
                    "mean_kl": float(np.mean(xm)) if len(xm) > 0 else np.nan,
                    "std_kl": float(np.std(xm)) if len(xm) > 0 else np.nan,
                })
                continue

            rho, p_param = sp_stats.spearmanr(xm, ym)

            # Permutation p-value
            null_vals = np.empty(n_permutations)
            for i in range(n_permutations):
                null_vals[i] = sp_stats.spearmanr(xm, rng.permutation(ym)).statistic
            p_perm = float(np.mean(np.abs(null_vals) >= np.abs(rho)))

            records.append({
                "dimension": d,
                "empirical_measure": ec,
                "spearman_rho": float(rho),
                "parametric_p": float(p_param),
                "permutation_p": p_perm,
                "mean_kl": float(np.mean(xm)),
                "std_kl": float(np.std(xm)),
            })

    result = pd.DataFrame(records)
    logger.info(
        f"Per-dimension analysis: {n_dims} dims x "
        f"{len(empirical_cols)} empirical measures"
    )
    return result


# ---------------------------------------------------------------------------
# DTW trajectory similarity
# ---------------------------------------------------------------------------


def dtw_trajectory_similarity(
    model_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    common_guids: List[str],
    model_col: str = "kl_mean",
    empirical_col: str = "ite_valid",
) -> Dict[str, Any]:
    """Compute DTW distance between full per-GUID trajectories.

    Uses the **complete** trajectory from each dataset (not just matched
    time points), since DTW naturally handles different-length sequences.
    Both trajectories are z-scored before comparison.

    Args:
        model_df: Full model TE DataFrame (all segments for common GUIDs).
        empirical_df: Full empirical TE DataFrame (all epochs for common
            GUIDs).
        common_guids: List of GUIDs present in both datasets.
        model_col: Model measure column.
        empirical_col: Empirical measure column.

    Returns:
        Dict with per-GUID DTW distances, normalised DTW, and summary
        statistics.  Each GUID entry includes ``n_model`` and
        ``n_empirical`` showing the full trajectory lengths used.
    """
    per_guid: Dict[str, Dict[str, Any]] = {}
    use_dtw = True

    try:
        from dtw import dtw as dtw_func
    except ImportError:
        try:
            from tslearn.metrics import dtw as tslearn_dtw

            def dtw_func(x, y, **_kwargs):
                """Wrapper around tslearn DTW."""
                class _R:
                    distance = tslearn_dtw(x.reshape(-1, 1), y.reshape(-1, 1))
                return _R()
        except ImportError:
            use_dtw = False
            logger.info("DTW libraries not available, using Euclidean distance.")

    for guid in common_guids:
        m_sub = model_df[model_df["guid"] == guid].sort_values("epoch")
        e_sub = empirical_df[empirical_df["guid"] == guid].sort_values("domain_start")

        m_vals = m_sub[model_col].values.astype(float)
        e_vals = e_sub[empirical_col].values.astype(float)

        # Remove NaNs
        m_vals = m_vals[np.isfinite(m_vals)]
        e_vals = e_vals[np.isfinite(e_vals)]

        if len(m_vals) < 3 or len(e_vals) < 3:
            per_guid[guid] = {
                "distance": np.nan, "normalized": np.nan,
                "n_model": len(m_vals), "n_empirical": len(e_vals),
            }
            continue

        # Z-score normalisation
        m_std, e_std = np.std(m_vals), np.std(e_vals)
        if m_std == 0 or e_std == 0:
            per_guid[guid] = {
                "distance": np.nan, "normalized": np.nan,
                "n_model": len(m_vals), "n_empirical": len(e_vals),
            }
            continue

        m_z = (m_vals - np.mean(m_vals)) / m_std
        e_z = (e_vals - np.mean(e_vals)) / e_std

        if use_dtw:
            result = dtw_func(m_z, e_z, keep_internals=False)
            dist = float(result.distance)
        else:
            # Euclidean fallback: truncate to shorter length
            n_min = min(len(m_z), len(e_z))
            dist = float(np.sqrt(np.sum((m_z[:n_min] - e_z[:n_min]) ** 2)))

        max_len = max(len(m_vals), len(e_vals))
        per_guid[guid] = {
            "distance": dist,
            "normalized": dist / max_len,
            "n_model": len(m_vals),
            "n_empirical": len(e_vals),
        }

    distances = [v["distance"] for v in per_guid.values() if np.isfinite(v["distance"])]

    result = {
        "method": "dtw" if use_dtw else "euclidean",
        "per_guid": per_guid,
        "mean_distance": float(np.mean(distances)) if distances else np.nan,
        "std_distance": float(np.std(distances)) if len(distances) > 1 else np.nan,
    }
    logger.info(
        f"Trajectory similarity ({model_col} vs {empirical_col}): "
        f"method={result['method']}, mean_dist={result['mean_distance']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Leave-one-GUID-out sensitivity
# ---------------------------------------------------------------------------


def leave_one_guid_out_sensitivity(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    method: str = "spearman",
) -> Dict[str, Any]:
    """Leave-one-GUID-out analysis to assess correlation robustness.

    With only 3 GUIDs, each has outsized influence.  Computes the pooled
    correlation with each GUID removed in turn.

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.
        method: ``"spearman"`` or ``"kendall"``.

    Returns:
        Dict with ``full_correlation``, ``leave_out_correlations`` (per GUID),
        ``min_correlation``, ``max_correlation``, ``range``,
        ``most_influential_guid``.
    """
    corr_fn = sp_stats.spearmanr if method == "spearman" else sp_stats.kendalltau

    x_full = merged_df[model_col].values.astype(float)
    y_full = merged_df[empirical_col].values.astype(float)
    mask = np.isfinite(x_full) & np.isfinite(y_full)

    full_r = float(corr_fn(x_full[mask], y_full[mask]).statistic) if mask.sum() >= 3 else np.nan

    guids = merged_df["guid"].unique()
    leave_out: Dict[str, float] = {}

    for guid in guids:
        subset = merged_df[merged_df["guid"] != guid]
        x = subset[model_col].values.astype(float)
        y = subset[empirical_col].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)

        if m.sum() < 3:
            leave_out[guid] = np.nan
        else:
            leave_out[guid] = float(corr_fn(x[m], y[m]).statistic)

    valid_lo = [v for v in leave_out.values() if np.isfinite(v)]

    # Most influential = GUID whose removal changes correlation the most
    if valid_lo and np.isfinite(full_r):
        changes = {g: abs(full_r - v) for g, v in leave_out.items() if np.isfinite(v)}
        most_influential = max(changes, key=changes.get) if changes else None
    else:
        most_influential = None

    result = {
        "full_correlation": full_r,
        "leave_out_correlations": leave_out,
        "min_correlation": float(min(valid_lo)) if valid_lo else np.nan,
        "max_correlation": float(max(valid_lo)) if valid_lo else np.nan,
        "range": float(max(valid_lo) - min(valid_lo)) if len(valid_lo) >= 2 else np.nan,
        "most_influential_guid": most_influential,
        "method": method,
    }
    logger.info(
        f"Leave-one-out ({model_col} vs {empirical_col}): "
        f"full={full_r:.4f}, range=[{result['min_correlation']:.4f}, "
        f"{result['max_correlation']:.4f}]"
    )
    return result


# ---------------------------------------------------------------------------
# Cluster-aware bootstrap
# ---------------------------------------------------------------------------


def cluster_aware_bootstrap(
    merged_df: pd.DataFrame,
    model_col: str,
    empirical_col: str,
    guid_col: str = "guid",
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Block bootstrap resampling GUIDs to produce cluster-aware CIs.

    Standard bootstrap assumes IID observations, violating the temporal
    nesting within patients.  This resamples at the GUID level: each
    iteration draws ``n_guids`` GUIDs with replacement and includes all
    their observations.

    Args:
        merged_df: Merged DataFrame.
        model_col: Model measure column.
        empirical_col: Empirical measure column.
        guid_col: GUID column name.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with ``observed``, ``bootstrap_mean``, ``bootstrap_std``,
        ``ci_lo``, ``ci_hi``, ``bootstrap_samples``, ``n_guids``.
    """
    rng = np.random.default_rng(seed)
    guids = merged_df[guid_col].unique()
    n_guids = len(guids)
    alpha = (1 - ci) / 2

    # Full-data correlation
    x_full = merged_df[model_col].values.astype(float)
    y_full = merged_df[empirical_col].values.astype(float)
    mask = np.isfinite(x_full) & np.isfinite(y_full)
    observed = float(sp_stats.spearmanr(x_full[mask], y_full[mask]).statistic) if mask.sum() >= 3 else np.nan

    # Group data by GUID for efficient resampling
    guid_groups = {
        g: merged_df[merged_df[guid_col] == g]
        for g in guids
    }

    samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample GUIDs with replacement
        selected = rng.choice(guids, size=n_guids, replace=True)
        boot_df = pd.concat(
            [guid_groups[g] for g in selected], ignore_index=True
        )

        x = boot_df[model_col].values.astype(float)
        y = boot_df[empirical_col].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)

        if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            samples[b] = np.nan
        else:
            samples[b] = sp_stats.spearmanr(x[m], y[m]).statistic

    valid_samples = samples[np.isfinite(samples)]

    result = {
        "observed": observed,
        "bootstrap_mean": float(np.mean(valid_samples)) if len(valid_samples) > 0 else np.nan,
        "bootstrap_std": float(np.std(valid_samples)) if len(valid_samples) > 0 else np.nan,
        "ci_lo": float(np.percentile(valid_samples, alpha * 100)) if len(valid_samples) > 0 else np.nan,
        "ci_hi": float(np.percentile(valid_samples, (1 - alpha) * 100)) if len(valid_samples) > 0 else np.nan,
        "bootstrap_samples": valid_samples,
        "n_guids": n_guids,
        "n_valid_samples": len(valid_samples),
    }
    logger.info(
        f"Block bootstrap ({model_col} vs {empirical_col}): "
        f"observed={observed:.4f}, "
        f"95% CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}], "
        f"n_guids={n_guids}"
    )
    return result


# ---------------------------------------------------------------------------
# Mutual information (KSG estimator)
# ---------------------------------------------------------------------------


def mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """Estimate mutual information using the KSG k-nearest-neighbour method.

    Implements the first estimator from Kraskov, Stoegbauer, Grassberger
    (2004).  Falls back to ``sklearn.feature_selection.mutual_info_regression``
    if available, otherwise uses a simple binned estimator.

    Args:
        x: First variable array.
        y: Second variable array (same length as *x*).
        k: Number of neighbours for the KSG estimator.

    Returns:
        Estimated mutual information in nats.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < k + 1:
        return np.nan

    # Try sklearn first (robust, well-tested implementation)
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(
            x.reshape(-1, 1), y, n_neighbors=k, random_state=42
        )
        return float(mi[0])
    except ImportError:
        pass

    # Fallback: simple binned MI estimator
    n_bins = max(3, int(np.sqrt(len(x))))
    hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Avoid log(0)
    mask_nz = pxy > 0
    mi = np.sum(
        pxy[mask_nz] * np.log(pxy[mask_nz] / (px[:, None] * py[None, :])[mask_nz])
    )
    return float(max(0.0, mi))


# ---------------------------------------------------------------------------
# Full comparison orchestrator
# ---------------------------------------------------------------------------


def run_full_comparison(
    merged_df: pd.DataFrame,
    output_dir: Union[str, Path],
    model_df: Optional[pd.DataFrame] = None,
    empirical_df: Optional[pd.DataFrame] = None,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """Run the complete analysis suite.

    Orchestrates all statistical analyses on the merged model-empirical
    DataFrame and saves CSV / JSON summaries.

    Args:
        merged_df: Merged DataFrame from fuzzy time matching.
        output_dir: Directory for output files.
        model_df: Full model TE DataFrame (used for DTW on complete
            trajectories).  If *None*, DTW uses matched points only.
        empirical_df: Full empirical TE DataFrame (used for DTW).
        n_permutations: Number of permutations for significance tests.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Comprehensive results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(merged_df) == 0:
        logger.warning("Empty merged DataFrame — no analyses to run.")
        return {"error": "no_data"}

    model_cols = list(_available_columns(merged_df, MODEL_TE_MEASURES).keys())
    empirical_cols = list(_available_columns(merged_df, EMPIRICAL_TE_MEASURES).keys())

    results: Dict[str, Any] = {
        "n_matched_pairs": len(merged_df),
        "n_guids": int(merged_df["guid"].nunique()),
    }

    # 1. Correlation matrices
    logger.info("Computing correlation matrices...")
    corr_spearman, pval_spearman = compute_correlation_matrix(
        merged_df, model_cols, empirical_cols, method="spearman"
    )
    corr_kendall, pval_kendall = compute_correlation_matrix(
        merged_df, model_cols, empirical_cols, method="kendall"
    )
    results["correlation_spearman"] = corr_spearman
    results["pvalue_spearman"] = pval_spearman
    results["correlation_kendall"] = corr_kendall
    results["pvalue_kendall"] = pval_kendall

    # Save correlation CSVs
    corr_spearman.to_csv(output_dir / "correlation_spearman.csv")
    pval_spearman.to_csv(output_dir / "pvalue_spearman.csv")
    corr_kendall.to_csv(output_dir / "correlation_kendall.csv")
    pval_kendall.to_csv(output_dir / "pvalue_kendall.csv")

    # 2. Permutation tests for primary measure pair
    logger.info("Running permutation tests...")
    primary_model = "kl_mean" if "kl_mean" in model_cols else model_cols[0]
    primary_empirical = "ite_valid" if "ite_valid" in empirical_cols else empirical_cols[0]

    perm_results = {}
    for mc in model_cols:
        for ec in empirical_cols:
            x = merged_df[mc].values.astype(float)
            y = merged_df[ec].values.astype(float)
            perm_results[f"{mc}_vs_{ec}"] = permutation_test_correlation(
                x, y, n_permutations=n_permutations
            )
    results["permutation_tests"] = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
        for k, v in perm_results.items()
    }
    results["_permutation_tests_full"] = perm_results  # with null distributions

    # 3. Within-GUID correlations
    logger.info("Computing within-GUID correlations...")
    within_guid = compute_within_guid_correlations(
        merged_df, primary_model, primary_empirical, min_pairs=5
    )
    results["within_guid_correlations"] = within_guid
    if len(within_guid) > 0:
        within_guid.to_csv(output_dir / "within_guid_correlations.csv", index=False)

    # 4. Concordance analysis
    logger.info("Computing concordance analysis...")
    concordance = concordance_analysis(merged_df, primary_model, primary_empirical)
    results["concordance"] = concordance

    # 5. Trend agreement
    logger.info("Computing trend agreement...")
    trends = trend_agreement_analysis(merged_df, primary_model, primary_empirical)
    results["trend_agreement"] = {
        k: v for k, v in trends.items() if k != "per_guid_agreement"
    }
    results["_trend_agreement_full"] = trends

    # 6. Per-dimension analysis
    logger.info("Running per-dimension analysis...")
    dim_df = per_dimension_analysis(
        merged_df, empirical_cols=empirical_cols, n_permutations=min(n_permutations, 5000)
    )
    results["per_dimension"] = dim_df
    dim_df.to_csv(output_dir / "per_dimension_analysis.csv", index=False)

    # 7. DTW trajectory similarity (full per-GUID trajectories)
    logger.info("Computing DTW trajectory similarity...")
    common_guids = sorted(merged_df["guid"].unique().tolist())
    if model_df is not None and empirical_df is not None:
        dtw_results = dtw_trajectory_similarity(
            model_df, empirical_df, common_guids,
            model_col=primary_model, empirical_col=primary_empirical,
        )
    else:
        dtw_results = {"method": "skipped", "per_guid": {}, "mean_distance": np.nan}
    results["dtw"] = {
        k: v for k, v in dtw_results.items()
        if k != "per_guid"
    }
    results["_dtw_full"] = dtw_results

    # 8. Leave-one-GUID-out sensitivity
    logger.info("Running leave-one-GUID-out sensitivity...")
    loo = leave_one_guid_out_sensitivity(
        merged_df, primary_model, primary_empirical
    )
    results["leave_one_out"] = loo

    # 9. Cluster-aware bootstrap
    logger.info("Running cluster-aware bootstrap...")
    boot = cluster_aware_bootstrap(
        merged_df, primary_model, primary_empirical, n_bootstrap=n_bootstrap
    )
    results["bootstrap"] = {
        k: v for k, v in boot.items() if not isinstance(v, np.ndarray)
    }
    results["_bootstrap_full"] = boot

    # 10. Mutual information
    logger.info("Computing mutual information...")
    mi_results: Dict[str, float] = {}
    for mc in model_cols:
        for ec in empirical_cols:
            x = merged_df[mc].values.astype(float)
            y = merged_df[ec].values.astype(float)
            mi_results[f"{mc}_vs_{ec}"] = mutual_information_knn(x, y)
    results["mutual_information"] = mi_results

    # Export JSON summary (exclude DataFrames and arrays)
    json_safe = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, pd.DataFrame):
            json_safe[k] = v.to_dict(orient="records")
        elif isinstance(v, np.ndarray):
            json_safe[k] = v.tolist()
        elif isinstance(v, dict):
            json_safe[k] = {
                kk: (vv.tolist() if isinstance(vv, np.ndarray) else
                     float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                for kk, vv in v.items()
                if not isinstance(vv, (pd.DataFrame, np.ndarray))
            }
        else:
            json_safe[k] = v

    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(json_safe, f, indent=2, default=str)

    logger.info(f"Analysis results exported to {output_dir}")
    return results
