"""Merge empirical TE and VAE-KLD data, compute correlations and tests.

Provides functions to:
1. Load KLD from a pre-computed metrics CSV or via live model inference.
2. Merge TE and KLD DataFrames on (guid, rounded domain_start).
3. Compute pooled, per-GUID, and cross-GUID correlations with bootstrap CIs.
4. Run population-level statistical tests (Fisher z, Wilcoxon).

Example:
    >>> from te_kld_analysis import load_kld_from_metrics_csv, merge_te_kld
    >>> kld_df = load_kld_from_metrics_csv("metrics.csv")
    >>> merged = merge_te_kld(te_df, kld_df)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as sp_stats

from model.vae_teb_prediction.testing.TE_Calculated.te_data_loader import (
    fuzzy_time_match,
    normalize_guid,
    round_domain_start,
)

# ---------------------------------------------------------------------------
# KLD data loading
# ---------------------------------------------------------------------------


def load_kld_from_metrics_csv(
    csv_path: Union[str, Path],
    grid_spacing: int = 1200,
) -> pd.DataFrame:
    """Load per-epoch KLD values from a pre-computed metrics CSV.

    The CSV is expected to come from ``collect_metrics()`` in
    ``testing/collectors.py`` and must contain at least ``guid``, ``epoch``,
    and ``kld`` columns. All other columns are preserved verbatim, so
    optional TE-surrogate columns added by the v1 collector — ``label``,
    ``kld_pc1`` / ``kld_pc2`` / ``kld_pc3``, ``posterior_drift_norm``,
    ``attention_entropy_mean``, ``attention_concentration_mean``,
    ``te_lag_peak``, ``te_lag_total_mass``, ``delta_src_norm`` — flow
    straight into the merged comparison DataFrame.

    Args:
        csv_path: Path to the metrics CSV file.
        grid_spacing: Epoch-grid spacing for rounding ``epoch`` values.

    Returns:
        DataFrame with all CSV columns plus an ``epoch_rounded`` column
        used by the ``exact_grid`` matching mode.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"KLD metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"guid", "epoch", "kld"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"KLD CSV missing columns: {missing}")

    df["guid"] = df["guid"].astype(str).apply(normalize_guid)
    df["epoch_rounded"] = df["epoch"].apply(
        lambda v: round_domain_start(v, grid_spacing)
    )

    # Drop rows with NaN KLD
    n_before = len(df)
    df = df.dropna(subset=["kld"]).copy()
    if len(df) < n_before:
        logger.warning(
            f"Dropped {n_before - len(df)} rows with NaN KLD values."
        )

    logger.info(
        f"Loaded {len(df)} KLD epochs from {csv_path.name}, "
        f"{df['guid'].nunique()} unique GUIDs."
    )
    return df


def load_kld_from_inference(
    config_path: Union[str, Path],
    te_guids: List[str],
    checkpoint_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    device: Optional[str] = None,
    grid_spacing: int = 1200,
) -> pd.DataFrame:
    """Run VAE inference on specified GUIDs and return per-epoch KLD.

    Uses ``TestRunner`` + ``collect_metrics()`` with a GUID-filtered
    dataloader restricted to the provided GUIDs.

    Args:
        config_path: Path to the VAE config YAML (used to resolve checkpoint
            and data paths if not explicitly provided).
        te_guids: List of normalised GUIDs to include.
        checkpoint_path: Model checkpoint path.  If *None*, resolved from
            config.
        data_path: HDF5 dataset path(s).  If *None*, resolved from config.
        device: Torch device string (e.g. ``"cuda:0"``).  Auto-detected if
            *None*.
        grid_spacing: Epoch-grid spacing for rounding.

    Returns:
        DataFrame with columns: guid, epoch, epoch_rounded, kld.
    """
    import torch
    import yaml

    from hdf5_dataset.hdf5_dataset import (
        build_guid_filtered_dataloader,
    )
    from model.vae_teb_prediction.testing.base import TestRunner
    from model.vae_teb_prediction.testing.collectors import collect_metrics

    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve checkpoint
    if checkpoint_path is None:
        checkpoint_path = cfg.get("model_config", {}).get(
            "core_model_checkpoint"
        )
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path not provided and not found in config."
        )

    # Resolve data paths
    if data_path is None:
        data_path = cfg.get("dataset_config", {}).get("vae_test_datasets")
    if data_path is None:
        raise ValueError("data_path not provided and not found in config.")
    if isinstance(data_path, str):
        data_path = [data_path]

    # Resolve stats_path, normalize_fields, and dataset_kwargs from config.
    # Accept both spellings: v1 configs use "stat_path", older ones use
    # "stats_path".
    dataset_cfg = cfg.get("dataset_config", {}) or {}
    stats_path = dataset_cfg.get("stat_path") or dataset_cfg.get("stats_path")
    dataloader_cfg = dataset_cfg.get("dataloader_config", {}) or {}
    normalize_fields = dataloader_cfg.get("normalize_fields") or dataset_cfg.get("normalize_fields")
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}

    # Remove epoch filtering so we get data all the way to delivery
    dataset_kwargs.pop("epoch_max", None)
    dataset_kwargs.pop("epoch_min", None)

    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {checkpoint_path} on {device}")
    runner = TestRunner.from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=str(Path(config_path).parent / "te_kld_tmp"),
        config_path=config_path,
        device=torch.device(device),
    )

    logger.info(
        f"Building GUID-filtered dataloader for {len(te_guids)} GUIDs..."
    )
    _, guid_loader = build_guid_filtered_dataloader(
        dataset_paths=data_path,
        min_samples=1,
        max_guids=None,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        **dataset_kwargs,
    )

    logger.info("Running inference to collect KLD metrics...")
    metrics_df = collect_metrics(runner, guid_loader)

    # Normalise GUIDs and round epochs
    metrics_df["guid"] = metrics_df["guid"].astype(str).apply(normalize_guid)
    metrics_df["epoch_rounded"] = metrics_df["epoch"].apply(
        lambda v: round_domain_start(v, grid_spacing)
    )

    # Filter to only the requested GUIDs
    te_guid_set = set(te_guids)
    metrics_df = metrics_df[metrics_df["guid"].isin(te_guid_set)].copy()

    logger.info(
        f"Collected KLD for {len(metrics_df)} epochs, "
        f"{metrics_df['guid'].nunique()} matching GUIDs."
    )
    return metrics_df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_te_kld(
    te_df: pd.DataFrame,
    kld_df: pd.DataFrame,
    matching_mode: str = "fuzzy",
    max_gap_seconds: float = 300.0,
) -> pd.DataFrame:
    """Merge empirical TE and VAE-KLD DataFrames on (guid, time).

    Supports two matching strategies:

    * ``"fuzzy"`` (default): greedy 1-to-1 nearest-neighbour matching
      within ``±max_gap_seconds`` via
      :func:`te_data_loader.fuzzy_time_match`. Every matched pair carries
      an explicit ``time_gap_seconds`` column.
    * ``"exact_grid"``: legacy behaviour — inner-joins
      ``domain_start_rounded`` to ``epoch_rounded`` assuming both sides
      were rounded to the same grid at load time. Retained for
      backward-compatibility diff checks.

    The returned DataFrame always carries the canonical columns
    ``guid, domain_start, epoch, ite_valid, kld`` (the latter two
    automatically renamed / preserved regardless of matching mode) so
    downstream analyses don't need to know which mode was used.

    Args:
        te_df: Empirical TE DataFrame from ``load_te_data()``.
        kld_df: KLD DataFrame from ``load_kld_from_metrics_csv()`` or
            ``load_kld_from_inference()``.
        matching_mode: ``"fuzzy"`` or ``"exact_grid"``.
        max_gap_seconds: Only used when ``matching_mode="fuzzy"``.
            Default 300 s (±5 min).

    Returns:
        Merged DataFrame.
    """
    n_te_guids = te_df["guid"].nunique()
    n_kld_guids = kld_df["guid"].nunique()

    if matching_mode == "fuzzy":
        merged = fuzzy_time_match(
            te_df, kld_df,
            max_gap_seconds=max_gap_seconds,
            te_time_col="domain_start",
            kld_time_col="epoch",
        )
    elif matching_mode == "exact_grid":
        if "domain_start_rounded" not in te_df.columns:
            raise ValueError(
                "exact_grid mode requires 'domain_start_rounded' column "
                "on te_df (produced by load_te_data)."
            )
        if "epoch_rounded" not in kld_df.columns:
            raise ValueError(
                "exact_grid mode requires 'epoch_rounded' column on "
                "kld_df (produced by load_kld_from_*)."
            )
        merged = pd.merge(
            te_df,
            kld_df[["guid", "epoch_rounded", "kld"]],
            left_on=["guid", "domain_start_rounded"],
            right_on=["guid", "epoch_rounded"],
            how="inner",
        )
    else:
        raise ValueError(
            f"Unknown matching_mode: {matching_mode!r}. "
            "Use 'fuzzy' or 'exact_grid'."
        )

    n_merged_guids = (
        int(merged["guid"].nunique()) if len(merged) > 0 else 0
    )
    logger.info(
        f"Merge result ({matching_mode}): {len(merged)} matched epochs, "
        f"{n_merged_guids} GUIDs (TE had {n_te_guids}, KLD had {n_kld_guids})."
    )
    if n_merged_guids == 0:
        logger.warning("No GUIDs matched between TE and KLD data!")

    return merged


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute bootstrap confidence intervals for Pearson and Spearman.

    Args:
        x: First variable array.
        y: Second variable array (same length as *x*).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: pearson_r, spearman_rho, pearson_ci_lo,
        pearson_ci_hi, spearman_ci_lo, spearman_ci_hi,
        pearson_samples, spearman_samples.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    alpha = (1 - ci) / 2

    pearson_samples = np.empty(n_bootstrap)
    spearman_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        # Skip degenerate resamples
        if np.std(xi) == 0 or np.std(yi) == 0:
            pearson_samples[i] = np.nan
            spearman_samples[i] = np.nan
            continue
        pearson_samples[i] = np.corrcoef(xi, yi)[0, 1]
        spearman_samples[i] = sp_stats.spearmanr(xi, yi).statistic

    return {
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_rho": float(sp_stats.spearmanr(x, y).statistic),
        "pearson_ci_lo": float(np.nanpercentile(pearson_samples, alpha * 100)),
        "pearson_ci_hi": float(
            np.nanpercentile(pearson_samples, (1 - alpha) * 100)
        ),
        "spearman_ci_lo": float(
            np.nanpercentile(spearman_samples, alpha * 100)
        ),
        "spearman_ci_hi": float(
            np.nanpercentile(spearman_samples, (1 - alpha) * 100)
        ),
        "pearson_samples": pearson_samples,
        "spearman_samples": spearman_samples,
    }


def compute_pooled_correlation(
    merged_df: pd.DataFrame,
    n_bootstrap: int = 10000,
) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation on all matched epochs.

    Args:
        merged_df: Output of ``merge_te_kld()``.
        n_bootstrap: Number of bootstrap iterations for CIs.

    Returns:
        Dict with correlation values, p-values, and bootstrap CIs.
    """
    x = merged_df["ite_valid"].values
    y = merged_df["kld"].values

    # Remove any remaining NaN pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        logger.warning("Too few valid pairs for pooled correlation.")
        return {"n": len(x), "error": "insufficient_data"}

    pearson_r, pearson_p = sp_stats.pearsonr(x, y)
    spearman_rho, spearman_p = sp_stats.spearmanr(x, y)

    boot = bootstrap_correlation(x, y, n_bootstrap=n_bootstrap)

    result = {
        "n": int(len(x)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_ci_lo": boot["pearson_ci_lo"],
        "pearson_ci_hi": boot["pearson_ci_hi"],
        "spearman_ci_lo": boot["spearman_ci_lo"],
        "spearman_ci_hi": boot["spearman_ci_hi"],
        "bootstrap_pearson_samples": boot["pearson_samples"],
        "bootstrap_spearman_samples": boot["spearman_samples"],
    }

    logger.info(
        f"Pooled correlation (n={len(x)}): "
        f"Pearson r={pearson_r:.4f} (p={pearson_p:.2e}), "
        f"Spearman ρ={spearman_rho:.4f} (p={spearman_p:.2e})"
    )
    return result


def compute_per_guid_correlations(
    merged_df: pd.DataFrame,
    min_epochs: int = 5,
) -> pd.DataFrame:
    """Compute within-GUID Pearson and Spearman correlations.

    For each GUID with at least ``min_epochs`` matched epochs, computes the
    temporal correlation between TE and KLD trajectories.

    Args:
        merged_df: Output of ``merge_te_kld()``.
        min_epochs: Minimum matched epochs required per GUID.

    Returns:
        DataFrame with columns: guid, n_epochs, mean_ite_valid, mean_kld,
        pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    records: List[Dict[str, Any]] = []
    grouped = merged_df.groupby("guid")

    for guid, group in grouped:
        if len(group) < min_epochs:
            continue

        group_sorted = group.sort_values("domain_start_rounded")
        x = group_sorted["ite_valid"].values
        y = group_sorted["kld"].values

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if len(x) < min_epochs:
            continue

        # Handle constant arrays
        if np.std(x) == 0 or np.std(y) == 0:
            records.append({
                "guid": guid,
                "n_epochs": len(x),
                "mean_ite_valid": float(np.mean(x)),
                "mean_kld": float(np.mean(y)),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            })
            continue

        pr, pp = sp_stats.pearsonr(x, y)
        sr, sp_val = sp_stats.spearmanr(x, y)

        records.append({
            "guid": guid,
            "n_epochs": len(x),
            "mean_ite_valid": float(np.mean(x)),
            "mean_kld": float(np.mean(y)),
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_rho": float(sr),
            "spearman_p": float(sp_val),
        })

    result = pd.DataFrame(records)
    logger.info(
        f"Per-GUID correlations: {len(result)} GUIDs with >= {min_epochs} "
        f"epochs (out of {grouped.ngroups} total)."
    )
    if len(result) > 0:
        median_r = result["pearson_r"].median()
        median_rho = result["spearman_rho"].median()
        logger.info(
            f"  Median Pearson r = {median_r:.4f}, "
            f"Median Spearman ρ = {median_rho:.4f}"
        )
    return result


def compute_cross_guid_correlation(
    per_guid_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Correlate per-GUID mean TE with per-GUID mean KLD.

    Tests whether patients with higher average TE also exhibit higher
    average KLD.

    Args:
        per_guid_df: Output of ``compute_per_guid_correlations()``.

    Returns:
        Dict with Pearson/Spearman r, p-values, and sample size.
    """
    x = per_guid_df["mean_ite_valid"].values
    y = per_guid_df["mean_kld"].values

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        logger.warning("Too few GUIDs for cross-GUID correlation.")
        return {"n_guids": len(x), "error": "insufficient_data"}

    pr, pp = sp_stats.pearsonr(x, y)
    sr, sp_val = sp_stats.spearmanr(x, y)

    result = {
        "n_guids": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp_val),
    }
    logger.info(
        f"Cross-GUID correlation (n={len(x)}): "
        f"Pearson r={pr:.4f} (p={pp:.2e}), "
        f"Spearman ρ={sr:.4f} (p={sp_val:.2e})"
    )
    return result


def population_level_test(
    per_guid_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Test whether population mean of per-GUID correlations differs from 0.

    Uses Fisher z-transformation on per-GUID Pearson r values and a
    Wilcoxon signed-rank test as a non-parametric alternative.

    Args:
        per_guid_df: Output of ``compute_per_guid_correlations()``.

    Returns:
        Dict with Fisher z-test and Wilcoxon results for both Pearson
        and Spearman per-GUID values.
    """
    result: Dict[str, Any] = {}

    for metric, col in [("pearson", "pearson_r"), ("spearman", "spearman_rho")]:
        vals = per_guid_df[col].dropna().values
        if len(vals) < 3:
            result[metric] = {"error": "insufficient_data", "n": len(vals)}
            continue

        # Fisher z-transform (for Pearson; applied to Spearman as approx.)
        z_vals = np.arctanh(np.clip(vals, -0.9999, 0.9999))
        z_mean = np.mean(z_vals)
        z_se = np.std(z_vals, ddof=1) / np.sqrt(len(z_vals))
        z_stat = z_mean / z_se if z_se > 0 else 0.0
        z_p = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))

        # Wilcoxon signed-rank test (H0: median = 0)
        try:
            w_stat, w_p = sp_stats.wilcoxon(vals, alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        result[metric] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)),
            "fisher_z_mean": float(z_mean),
            "fisher_z_se": float(z_se),
            "fisher_z_stat": float(z_stat),
            "fisher_z_p": float(z_p),
            "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else None,
            "wilcoxon_p": float(w_p) if np.isfinite(w_p) else None,
        }
        logger.info(
            f"Population test ({metric}, n={len(vals)}): "
            f"mean={np.mean(vals):.4f}, "
            f"Fisher z p={z_p:.2e}, Wilcoxon p={w_p:.2e}"
        )

    return result


# ---------------------------------------------------------------------------
# Permutation test for correlation significance
# ---------------------------------------------------------------------------


def permutation_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Permutation-based significance test for a correlation coefficient.

    Permutes ``y`` to build a null distribution and reports the exact
    two-sided p-value. Essential because parametric correlation
    p-values are unreliable at the small sample sizes typical of this
    comparison.

    Args:
        x: First variable array.
        y: Second variable array (same length as ``x``).
        method: ``"spearman"`` or ``"kendall"``.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        Dict with ``observed``, ``p_value``, ``null_mean``, ``null_std``,
        ``null_ci_lo``, ``null_ci_hi``, and ``null_distribution`` (a
        numpy array, not exported to JSON).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {
            "observed": float("nan"),
            "p_value": float("nan"),
            "n": int(len(x)),
            "error": "insufficient_data",
        }

    corr_fn = (
        sp_stats.spearmanr if method == "spearman" else sp_stats.kendalltau
    )
    observed = float(corr_fn(x, y).statistic)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_dist[i] = corr_fn(x, y_perm).statistic

    p_value = float(np.mean(np.abs(null_dist) >= abs(observed)))
    return {
        "observed": observed,
        "p_value": p_value,
        "n": int(len(x)),
        "method": method,
        "n_permutations": int(n_permutations),
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist)),
        "null_ci_lo": float(np.percentile(null_dist, 2.5)),
        "null_ci_hi": float(np.percentile(null_dist, 97.5)),
        "null_distribution": null_dist,
    }


# ---------------------------------------------------------------------------
# Concordance analysis
# ---------------------------------------------------------------------------


def concordance_analysis(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    kld_col: str = "kld",
) -> Dict[str, Any]:
    """Compute Kendall's tau-b and pair-level concordance statistics.

    Kendall's tau-b handles ties correctly and is directly interpretable
    as the excess probability that a randomly drawn pair has the same
    relative ordering in both variables.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_col: TE column name.
        kld_col: KLD column name.

    Returns:
        Dict with ``kendall_tau``, ``kendall_p``, ``concordance_index``,
        ``n_concordant``, ``n_discordant``, ``n_tied``, ``n_pairs``.
    """
    x = merged_df[te_col].values.astype(float)
    y = merged_df[kld_col].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"error": "insufficient_data", "n_pairs": int(len(x))}

    tau, p = sp_stats.kendalltau(x, y)

    n = len(x)
    n_concordant = 0
    n_discordant = 0
    n_tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            prod = (x[j] - x[i]) * (y[j] - y[i])
            if prod > 0:
                n_concordant += 1
            elif prod < 0:
                n_discordant += 1
            else:
                n_tied += 1

    denom = n_concordant + n_discordant
    concordance_index = (
        n_concordant / denom if denom > 0 else 0.5
    )
    result = {
        "kendall_tau": float(tau),
        "kendall_p": float(p),
        "concordance_index": float(concordance_index),
        "n_concordant": int(n_concordant),
        "n_discordant": int(n_discordant),
        "n_tied": int(n_tied),
        "n_pairs": int(len(x)),
    }
    logger.info(
        f"Concordance ({te_col} vs {kld_col}): "
        f"tau={tau:.4f} (p={p:.2e}), C-index={concordance_index:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Trend agreement (sign of first differences)
# ---------------------------------------------------------------------------


def trend_agreement_analysis(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    kld_col: str = "kld",
) -> Dict[str, Any]:
    """Assess whether temporal derivatives of TE and KLD agree in sign.

    For each GUID, sorts by time and computes first differences. Counts
    how often both variables move in the same direction (both up or
    both down). Stable (zero-change) transitions are excluded from the
    denominator. Tests the overall agreement rate against chance
    (0.5) with a binomial test.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_col: TE column name.
        kld_col: KLD column name.

    Returns:
        Dict with ``sign_agreement_rate``, ``n_transitions``,
        ``n_agree``, ``binomial_p``, ``per_guid_agreement``.
    """
    time_col = (
        "domain_start" if "domain_start" in merged_df.columns
        else "epoch" if "epoch" in merged_df.columns
        else merged_df.columns[1]
    )
    per_guid: Dict[str, Dict[str, Any]] = {}
    all_agrees: List[int] = []

    for guid, group in merged_df.groupby("guid"):
        if len(group) < 2:
            continue
        g = group.sort_values(time_col)
        m_vals = g[te_col].values.astype(float)
        e_vals = g[kld_col].values.astype(float)
        m_diff = np.diff(m_vals)
        e_diff = np.diff(e_vals)

        nonzero = (m_diff != 0) & (e_diff != 0)
        if nonzero.sum() == 0:
            continue
        agrees = (
            np.sign(m_diff[nonzero]) == np.sign(e_diff[nonzero])
        ).astype(int)
        all_agrees.extend(agrees.tolist())
        per_guid[str(guid)] = {
            "n_transitions": int(nonzero.sum()),
            "n_agree": int(agrees.sum()),
            "agreement_rate": float(agrees.mean()),
        }

    if len(all_agrees) == 0:
        return {"error": "no_transitions", "n_transitions": 0}

    arr = np.asarray(all_agrees)
    overall_rate = float(arr.mean())
    n_total = int(len(arr))
    n_agree = int(arr.sum())
    binom_p = float(sp_stats.binomtest(n_agree, n_total, 0.5).pvalue)

    result = {
        "sign_agreement_rate": overall_rate,
        "n_transitions": n_total,
        "n_agree": n_agree,
        "binomial_p": binom_p,
        "per_guid_agreement": per_guid,
    }
    logger.info(
        f"Trend agreement ({te_col} vs {kld_col}): "
        f"{overall_rate:.1%} ({n_agree}/{n_total}), binomial p={binom_p:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Leave-one-GUID-out sensitivity
# ---------------------------------------------------------------------------


def leave_one_guid_out_sensitivity(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    kld_col: str = "kld",
    method: str = "spearman",
) -> Dict[str, Any]:
    """Assess correlation robustness by removing one GUID at a time.

    With small GUID counts each patient has outsized leverage. Recomputes
    the pooled correlation with each GUID removed and flags whichever
    removal causes the biggest change.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_col: TE column name.
        kld_col: KLD column name.
        method: ``"spearman"`` or ``"kendall"``.

    Returns:
        Dict with ``full_correlation``, ``leave_out_correlations`` (per
        GUID), ``min_correlation``, ``max_correlation``, ``range``,
        ``most_influential_guid``, ``method``.
    """
    corr_fn = (
        sp_stats.spearmanr if method == "spearman" else sp_stats.kendalltau
    )
    x_full = merged_df[te_col].values.astype(float)
    y_full = merged_df[kld_col].values.astype(float)
    mask = np.isfinite(x_full) & np.isfinite(y_full)
    full_r = (
        float(corr_fn(x_full[mask], y_full[mask]).statistic)
        if mask.sum() >= 3 else float("nan")
    )

    leave_out: Dict[str, float] = {}
    for guid in merged_df["guid"].unique():
        subset = merged_df[merged_df["guid"] != guid]
        x = subset[te_col].values.astype(float)
        y = subset[kld_col].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            leave_out[str(guid)] = float("nan")
        else:
            leave_out[str(guid)] = float(corr_fn(x[m], y[m]).statistic)

    valid = [v for v in leave_out.values() if np.isfinite(v)]
    if valid and np.isfinite(full_r):
        changes = {
            g: abs(full_r - v) for g, v in leave_out.items() if np.isfinite(v)
        }
        most_influential = (
            max(changes, key=changes.get) if changes else None
        )
    else:
        most_influential = None

    result = {
        "method": method,
        "full_correlation": full_r,
        "leave_out_correlations": leave_out,
        "min_correlation": float(min(valid)) if valid else float("nan"),
        "max_correlation": float(max(valid)) if valid else float("nan"),
        "range": (
            float(max(valid) - min(valid)) if len(valid) >= 2
            else float("nan")
        ),
        "most_influential_guid": most_influential,
    }
    logger.info(
        f"LOO ({te_col} vs {kld_col}): full={full_r:.4f}, "
        f"range=[{result['min_correlation']:.4f}, "
        f"{result['max_correlation']:.4f}]"
    )
    return result


# ---------------------------------------------------------------------------
# Cluster-aware block bootstrap
# ---------------------------------------------------------------------------


def cluster_aware_bootstrap(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    kld_col: str = "kld",
    guid_col: str = "guid",
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Block bootstrap resampling GUIDs to produce cluster-aware CIs.

    Observations from the same patient are temporally correlated, so a
    plain IID bootstrap is anti-conservative. This function resamples
    at the GUID level — each iteration draws ``n_guids`` GUIDs with
    replacement and uses *all* their matched rows. Returns CIs for both
    Pearson and Spearman.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_col: TE column name.
        kld_col: KLD column name.
        guid_col: GUID column name.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence level (e.g. 0.95).
        seed: Random seed.

    Returns:
        Dict with ``pearson`` and ``spearman`` sub-dicts, each
        containing ``observed``, ``bootstrap_mean``, ``bootstrap_std``,
        ``ci_lo``, ``ci_hi``, ``bootstrap_samples``. Also includes
        ``n_guids``, ``n_bootstrap``, ``ci``.
    """
    rng = np.random.default_rng(seed)
    guids = list(merged_df[guid_col].unique())
    n_guids = len(guids)
    alpha = (1 - ci) / 2

    x_full = merged_df[te_col].values.astype(float)
    y_full = merged_df[kld_col].values.astype(float)
    mask = np.isfinite(x_full) & np.isfinite(y_full)
    if mask.sum() < 3:
        return {
            "error": "insufficient_data",
            "n_pairs": int(mask.sum()),
            "n_guids": n_guids,
        }
    observed_pearson = float(
        sp_stats.pearsonr(x_full[mask], y_full[mask]).statistic
    )
    observed_spearman = float(
        sp_stats.spearmanr(x_full[mask], y_full[mask]).statistic
    )

    guid_groups = {g: merged_df[merged_df[guid_col] == g] for g in guids}

    pearson_samples = np.full(n_bootstrap, np.nan)
    spearman_samples = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        selected = rng.choice(np.array(guids), size=n_guids, replace=True)
        boot_df = pd.concat(
            [guid_groups[g] for g in selected], ignore_index=True
        )
        x = boot_df[te_col].values.astype(float)
        y = boot_df[kld_col].values.astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
            continue
        pearson_samples[b] = sp_stats.pearsonr(x[m], y[m]).statistic
        spearman_samples[b] = sp_stats.spearmanr(x[m], y[m]).statistic

    def _summary(samples: np.ndarray, observed: float) -> Dict[str, Any]:
        valid = samples[np.isfinite(samples)]
        if len(valid) == 0:
            return {"observed": observed, "error": "all_nan"}
        return {
            "observed": observed,
            "bootstrap_mean": float(np.mean(valid)),
            "bootstrap_std": float(np.std(valid)),
            "ci_lo": float(np.percentile(valid, alpha * 100)),
            "ci_hi": float(np.percentile(valid, (1 - alpha) * 100)),
            "bootstrap_samples": valid,
            "n_valid": int(len(valid)),
        }

    result = {
        "n_guids": n_guids,
        "n_bootstrap": int(n_bootstrap),
        "ci": float(ci),
        "pearson": _summary(pearson_samples, observed_pearson),
        "spearman": _summary(spearman_samples, observed_spearman),
    }
    logger.info(
        f"Cluster bootstrap (n_guids={n_guids}, n_iter={n_bootstrap}): "
        f"Pearson {observed_pearson:.4f} "
        f"CI=[{result['pearson'].get('ci_lo', float('nan')):.4f}, "
        f"{result['pearson'].get('ci_hi', float('nan')):.4f}]; "
        f"Spearman {observed_spearman:.4f} "
        f"CI=[{result['spearman'].get('ci_lo', float('nan')):.4f}, "
        f"{result['spearman'].get('ci_hi', float('nan')):.4f}]"
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
    """Estimate mutual information using a KSG-style k-NN estimator.

    Prefers ``sklearn.feature_selection.mutual_info_regression`` and
    falls back to a simple 2D histogram estimator when sklearn is
    absent. Captures non-linear association that rank correlations
    cannot.

    Args:
        x: First variable array.
        y: Second variable array (same length as ``x``).
        k: Number of neighbours for the KSG estimator.

    Returns:
        Estimated mutual information in nats (``>= 0``). Returns NaN
        when there are fewer than ``k + 1`` finite samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < k + 1:
        return float("nan")

    try:  # pragma: no cover
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(
            x.reshape(-1, 1), y, n_neighbors=k, random_state=42
        )
        return float(mi[0])
    except ImportError:
        pass

    n_bins = max(3, int(np.sqrt(len(x))))
    hist, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    mask_nz = (pxy > 0) & (px_py > 0)
    mi = float(np.sum(pxy[mask_nz] * np.log(pxy[mask_nz] / px_py[mask_nz])))
    return max(0.0, mi)


# ---------------------------------------------------------------------------
# Per-dimension KL analysis
# ---------------------------------------------------------------------------


def _detect_kld_dim_columns(df: pd.DataFrame) -> List[str]:
    """Return the sorted list of per-dim KLD columns present in ``df``."""
    out: List[str] = []
    for c in df.columns:
        if c.startswith("kld_dim_") or c.startswith("kld_per_dim_"):
            out.append(c)
    def _dim_index(col: str) -> int:
        tail = col.rsplit("_", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return 10**9
    return sorted(out, key=_dim_index)


def per_dimension_kl_analysis(
    merged_df: pd.DataFrame,
    te_col: str = "ite_valid",
    n_permutations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Correlate each per-dim KLD column with empirical TE.

    Identifies which latent dimensions drive the coupling with ``ite_valid``.
    Requires per-dim KLD columns (``kld_dim_0, kld_dim_1, ...``) produced
    by :func:`~collectors.collect_metrics` when the model emits
    ``kld_per_dim``. Gated on column presence — returns an empty
    DataFrame with an info log if no per-dim columns are found.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_col: Empirical TE column.
        n_permutations: Permutations for significance testing.
        seed: Random seed.

    Returns:
        DataFrame with one row per dimension: ``dimension``,
        ``column``, ``spearman_rho``, ``parametric_p``, ``permutation_p``,
        ``mean_kl``, ``std_kl``, ``n``.
    """
    dim_cols = _detect_kld_dim_columns(merged_df)
    if not dim_cols:
        logger.info(
            "per_dimension_kl_analysis: no 'kld_dim_*' columns present; "
            "skipping. Enable by re-emitting metrics after updating "
            "collectors.collect_metrics to flatten kld_per_dim."
        )
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    y_full = merged_df[te_col].values.astype(float)
    mask_y = np.isfinite(y_full)

    records: List[Dict[str, Any]] = []
    for col in dim_cols:
        try:
            dim_idx = int(col.rsplit("_", 1)[-1])
        except ValueError:
            dim_idx = -1

        x = merged_df[col].values.astype(float)
        mask = np.isfinite(x) & mask_y
        xm, ym = x[mask], y_full[mask]
        if len(xm) < 3 or np.std(xm) == 0 or np.std(ym) == 0:
            records.append({
                "dimension": dim_idx,
                "column": col,
                "spearman_rho": float("nan"),
                "parametric_p": float("nan"),
                "permutation_p": float("nan"),
                "mean_kl": float(np.mean(xm)) if len(xm) > 0 else float("nan"),
                "std_kl": float(np.std(xm)) if len(xm) > 0 else float("nan"),
                "n": int(len(xm)),
            })
            continue

        rho, p_param = sp_stats.spearmanr(xm, ym)
        null_vals = np.empty(n_permutations)
        for i in range(n_permutations):
            null_vals[i] = sp_stats.spearmanr(xm, rng.permutation(ym)).statistic
        p_perm = float(np.mean(np.abs(null_vals) >= abs(rho)))

        records.append({
            "dimension": dim_idx,
            "column": col,
            "spearman_rho": float(rho),
            "parametric_p": float(p_param),
            "permutation_p": p_perm,
            "mean_kl": float(np.mean(xm)),
            "std_kl": float(np.std(xm)),
            "n": int(len(xm)),
        })

    out = pd.DataFrame(records).sort_values("dimension").reset_index(drop=True)
    logger.info(
        f"Per-dim KL analysis: {len(out)} dims vs {te_col}"
    )
    return out


# ---------------------------------------------------------------------------
# Correlation matrix (multi-measure)
# ---------------------------------------------------------------------------


# Model-side KLD-family aggregates worth correlating against empirical TE
# when the columns exist in the merged dataframe.
CANDIDATE_KLD_COLS: Sequence[str] = (
    "kld",
    "kld_max",
    "feat_mse_total",
    "uplift_abs",
    "residual_ratio",
    # Lag-attn v1 TE surrogates (only present when the metrics CSV was
    # produced by the new collect_metrics implementation).
    "kld_pc1",
    "kld_pc2",
    "kld_pc3",
    "kld_pca_l2_top3",
    "posterior_drift_norm",
    "attention_concentration_mean",
    "attention_entropy_mean",
    "te_lag_total_mass",
    "delta_src_norm",
)
CANDIDATE_TE_COLS: Sequence[str] = (
    "ite_valid",
    "omnibus_te",
    "ite_valid_pc",
)


def correlation_matrix(
    merged_df: pd.DataFrame,
    te_cols: Optional[Sequence[str]] = None,
    kld_cols: Optional[Sequence[str]] = None,
    method: str = "spearman",
) -> Dict[str, pd.DataFrame]:
    """Compute pairwise correlations across multiple TE and KLD measures.

    Args:
        merged_df: Merged DataFrame from :func:`merge_te_kld`.
        te_cols: TE-side columns. Defaults to ``CANDIDATE_TE_COLS``
            intersected with the columns actually present.
        kld_cols: KLD-side columns. Defaults to ``CANDIDATE_KLD_COLS``
            intersected with the columns actually present.
        method: ``"spearman"`` or ``"kendall"``.

    Returns:
        Dict with ``correlation`` and ``p_value`` DataFrames, both
        shaped ``(len(kld_cols), len(te_cols))``.
    """
    if te_cols is None:
        te_cols = [c for c in CANDIDATE_TE_COLS if c in merged_df.columns]
    if kld_cols is None:
        kld_cols = [c for c in CANDIDATE_KLD_COLS if c in merged_df.columns]

    corr_arr = np.full((len(kld_cols), len(te_cols)), np.nan)
    pval_arr = np.full((len(kld_cols), len(te_cols)), np.nan)

    for i, kc in enumerate(kld_cols):
        for j, tc in enumerate(te_cols):
            x = merged_df[kc].values.astype(float)
            y = merged_df[tc].values.astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            xm, ym = x[mask], y[mask]
            if np.std(xm) == 0 or np.std(ym) == 0:
                continue
            if method == "spearman":
                r, p = sp_stats.spearmanr(xm, ym)
            elif method == "kendall":
                r, p = sp_stats.kendalltau(xm, ym)
            else:
                raise ValueError(f"Unknown method: {method}")
            corr_arr[i, j] = r
            pval_arr[i, j] = p

    corr_df = pd.DataFrame(
        corr_arr, index=list(kld_cols), columns=list(te_cols)
    )
    pval_df = pd.DataFrame(
        pval_arr, index=list(kld_cols), columns=list(te_cols)
    )
    logger.info(
        f"Correlation matrix ({method}): "
        f"{len(kld_cols)} KLD measures x {len(te_cols)} TE measures"
    )
    return {"correlation": corr_df, "p_value": pval_df}


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------


def _strip_arrays(obj: Any) -> Any:
    """Recursively strip numpy arrays for JSON-friendly output."""
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if not (
            isinstance(v, np.ndarray) and v.size > 16
        )}
    if isinstance(obj, list):
        return [_strip_arrays(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def export_summary(
    output_dir: Path,
    merged_df: pd.DataFrame,
    per_guid_df: pd.DataFrame,
    pooled_stats: Dict[str, Any],
    cross_guid_stats: Dict[str, Any],
    population_stats: Dict[str, Any],
    data_quality: Optional[Dict[str, Any]] = None,
    permutation_primary: Optional[Dict[str, Any]] = None,
    concordance: Optional[Dict[str, Any]] = None,
    trend_agreement: Optional[Dict[str, Any]] = None,
    leave_one_out: Optional[Dict[str, Any]] = None,
    cluster_bootstrap: Optional[Dict[str, Any]] = None,
    mutual_information: Optional[Dict[str, Any]] = None,
    dtw: Optional[Dict[str, Any]] = None,
    dtw_pooled_correlation: Optional[Dict[str, Any]] = None,
    per_dimension: Optional[pd.DataFrame] = None,
    correlation_matrices: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Export all results to CSV and JSON files.

    Every optional argument corresponds to a block of analyses that
    may or may not have been run. When ``None``, the block is omitted
    from the JSON summary and no CSV is written for it.

    Args:
        output_dir: Directory to write output files.
        merged_df: Merged TE-KLD DataFrame.
        per_guid_df: Per-GUID correlation DataFrame.
        pooled_stats: Pooled correlation results dict.
        cross_guid_stats: Cross-GUID correlation results dict.
        population_stats: Population-level test results dict.
        data_quality: Output of :func:`~te_data_loader.compute_data_quality_report`.
        permutation_primary: Output of :func:`permutation_test_correlation`
            on the primary ``ite_valid`` vs ``kld`` pair.
        concordance: Output of :func:`concordance_analysis`.
        trend_agreement: Output of :func:`trend_agreement_analysis`.
        leave_one_out: Output of :func:`leave_one_guid_out_sensitivity`.
        cluster_bootstrap: Output of :func:`cluster_aware_bootstrap`.
        mutual_information: Dict with MI estimate(s).
        dtw: Output of :func:`~te_dtw.dtw_align_per_guid`. Large
            per-GUID arrays are stripped from the JSON.
        dtw_pooled_correlation: Pooled correlation on DTW-aligned pairs.
        per_dimension: Output of :func:`per_dimension_kl_analysis`.
        correlation_matrices: Output of :func:`correlation_matrix`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(output_dir / "merged_te_kld.csv", index=False)
    per_guid_df.to_csv(output_dir / "per_guid_correlations.csv", index=False)

    if per_dimension is not None and len(per_dimension) > 0:
        per_dimension.to_csv(output_dir / "per_dimension.csv", index=False)
    if correlation_matrices is not None:
        for name, df in correlation_matrices.items():
            df.to_csv(output_dir / f"correlation_{name}.csv")

    summary: Dict[str, Any] = {
        "pooled": _strip_arrays(pooled_stats),
        "cross_guid": cross_guid_stats,
        "population_tests": population_stats,
        "data_summary": {
            "n_matched_epochs": int(len(merged_df)),
            "n_matched_guids": (
                int(merged_df["guid"].nunique()) if len(merged_df) else 0
            ),
            "n_guids_with_correlation": int(len(per_guid_df)),
        },
    }
    if len(merged_df) > 0:
        summary["data_summary"]["ite_valid_range"] = [
            float(merged_df["ite_valid"].min()),
            float(merged_df["ite_valid"].max()),
        ]
        summary["data_summary"]["kld_range"] = [
            float(merged_df["kld"].min()),
            float(merged_df["kld"].max()),
        ]
        if "time_gap_seconds" in merged_df.columns:
            summary["data_summary"]["time_gap_mean"] = float(
                merged_df["time_gap_seconds"].mean()
            )
            summary["data_summary"]["time_gap_max"] = float(
                merged_df["time_gap_seconds"].max()
            )

    if data_quality is not None:
        summary["data_quality"] = _strip_arrays(data_quality)
    if permutation_primary is not None:
        summary["permutation_primary"] = _strip_arrays(permutation_primary)
    if concordance is not None:
        summary["concordance"] = concordance
    if trend_agreement is not None:
        summary["trend_agreement"] = trend_agreement
    if leave_one_out is not None:
        summary["leave_one_out"] = leave_one_out
    if cluster_bootstrap is not None:
        summary["cluster_bootstrap"] = _strip_arrays(cluster_bootstrap)
    if mutual_information is not None:
        summary["mutual_information"] = mutual_information
    if dtw is not None:
        summary["dtw"] = _strip_arrays(dtw)
    if dtw_pooled_correlation is not None:
        summary["dtw_pooled_correlation"] = _strip_arrays(
            dtw_pooled_correlation
        )

    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if data_quality is not None:
        with open(output_dir / "data_quality.json", "w") as f:
            json.dump(_strip_arrays(data_quality), f, indent=2, default=str)

    logger.info(f"Results exported to {output_dir}")


# ---------------------------------------------------------------------------
# Lag-attn v1: PCA-derived score & per-class stratification
# ---------------------------------------------------------------------------


def pca_trajectory(
    df: pd.DataFrame,
    mode: str = "pc1",
) -> pd.Series:
    """Synthesize a PCA-based score column from the merged DataFrame.

    The metrics CSV produced by the v1 collector carries the per-sample
    means of the top-3 PCA components (``kld_pc1``, ``kld_pc2``,
    ``kld_pc3``). This helper combines them into a single score
    suitable for correlation against empirical TE.

    Args:
        df: Merged DataFrame (output of :func:`merge_te_kld`) that
            contains the ``kld_pc*`` columns.
        mode: One of:
            * ``"pc1"``   — use the first principal component as the score.
            * ``"l2_top3"`` — Euclidean norm of the top-3 components.
            * ``"sum_top3"`` — signed sum of the top-3 components.

    Returns:
        Pandas Series of the chosen score, indexed identically to ``df``.
        The series is filled with NaN when the required columns are not
        present (callers should ``dropna`` before using it).
    """
    cols = [c for c in ("kld_pc1", "kld_pc2", "kld_pc3") if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, name=f"kld_pca_{mode}")

    arr = df[cols].to_numpy(dtype=float)
    if mode == "pc1":
        out = arr[:, 0]
    elif mode == "l2_top3":
        out = np.sqrt(np.nansum(arr ** 2, axis=1))
    elif mode == "sum_top3":
        out = np.nansum(arr, axis=1)
    else:
        raise ValueError(
            f"Unknown pca_trajectory mode: {mode!r}. "
            "Use one of 'pc1', 'l2_top3', 'sum_top3'."
        )
    return pd.Series(out, index=df.index, name=f"kld_pca_{mode}")


def _label_to_folder(label_id: int) -> Optional[str]:
    """Map class id (1/2/3) to a stable subfolder name."""
    return {1: "te_kld_class_healthy", 2: "te_kld_class_acidosis", 3: "te_kld_class_hie"}.get(label_id)


def run_te_kld_pipeline_stratified(
    merged_df: pd.DataFrame,
    output_dir: Union[str, Path],
    pipeline_fn,
    *,
    pipeline_kwargs: Optional[Dict[str, Any]] = None,
    label_col: str = "label",
) -> Dict[str, Any]:
    """Re-run an existing TE-KLD pipeline once per outcome class.

    The wrapper writes:

    - ``<output_dir>/te_kld_class_all/`` (pooled, original behaviour)
    - ``<output_dir>/te_kld_class_healthy/``
    - ``<output_dir>/te_kld_class_acidosis/``
    - ``<output_dir>/te_kld_class_hie/``

    Each subfolder receives the full output of ``pipeline_fn`` for the
    matching class subset of ``merged_df``. ``pipeline_fn`` must accept
    ``(merged_df, output_dir, **pipeline_kwargs)`` and write artifacts
    to the directory it is given (compatible with the helpers in
    ``te_kld_comparison.py``).

    Args:
        merged_df: The output of :func:`merge_te_kld` after surrogate
            columns have been propagated through. Must include the
            ``label`` column for stratification to occur.
        output_dir: Root directory under which the per-class subfolders
            are written.
        pipeline_fn: Callable invoked once per class subset.
        pipeline_kwargs: Optional keyword arguments forwarded to every
            invocation of ``pipeline_fn``.
        label_col: Column carrying integer class IDs (1=HEALTHY,
            2=ACIDOSIS, 3=HIE). Defaults to ``"label"``.

    Returns:
        Dict ``{folder_name: pipeline_result}`` mapping each subfolder to
        the value returned by ``pipeline_fn``.
    """
    pipeline_kwargs = pipeline_kwargs or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    pooled_dir = output_dir / "te_kld_class_all"
    pooled_dir.mkdir(parents=True, exist_ok=True)
    try:
        results["te_kld_class_all"] = pipeline_fn(
            merged_df, pooled_dir, **pipeline_kwargs
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"stratified pipeline (pooled) failed: {exc}")
        results["te_kld_class_all"] = {"error": str(exc)}

    if label_col not in merged_df.columns:
        logger.warning(
            f"run_te_kld_pipeline_stratified: column {label_col!r} not "
            f"found, skipping per-class subdirectories."
        )
        return results

    for label_id in (1, 2, 3):
        folder_name = _label_to_folder(label_id)
        if folder_name is None:
            continue
        sub = merged_df[merged_df[label_col] == label_id]
        if sub.empty:
            logger.info(f"stratified pipeline: skipping {folder_name} (no rows)")
            continue
        sub_dir = output_dir / folder_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        try:
            results[folder_name] = pipeline_fn(
                pd.DataFrame(sub.copy()), sub_dir, **pipeline_kwargs
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"stratified pipeline ({folder_name}) failed: {exc}")
            results[folder_name] = {"error": str(exc)}

    return results
