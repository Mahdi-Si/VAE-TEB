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
    and ``kld`` columns.

    Args:
        csv_path: Path to the metrics CSV file.
        grid_spacing: Epoch-grid spacing for rounding ``epoch`` values.

    Returns:
        DataFrame with columns: guid, epoch, epoch_rounded, kld.
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

    # Resolve stats_path, normalize_fields, and dataset_kwargs from config
    stats_path = cfg.get("dataset_config", {}).get("stats_path")
    normalize_fields = cfg.get("dataset_config", {}).get("normalize_fields")
    dataloader_cfg = cfg.get("dataset_config", {}).get("dataloader_config", {})
    dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}

    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {checkpoint_path} on {device}")
    runner = TestRunner.from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=str(Path(config_path).parent / "te_kld_tmp"),
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
) -> pd.DataFrame:
    """Merge empirical TE and VAE-KLD DataFrames on (guid, rounded epoch).

    Both DataFrames must already have normalised GUIDs and rounded epoch
    columns (``domain_start_rounded`` for TE, ``epoch_rounded`` for KLD).

    Args:
        te_df: Empirical TE DataFrame from ``load_te_data()``.
        kld_df: KLD DataFrame from ``load_kld_from_metrics_csv()`` or
            ``load_kld_from_inference()``.

    Returns:
        Merged DataFrame with columns from both inputs, joined on
        ``(guid, epoch_rounded)``.
    """
    merged = pd.merge(
        te_df,
        kld_df[["guid", "epoch_rounded", "kld"]],
        left_on=["guid", "domain_start_rounded"],
        right_on=["guid", "epoch_rounded"],
        how="inner",
    )

    n_te_guids = te_df["guid"].nunique()
    n_kld_guids = kld_df["guid"].nunique()
    n_merged_guids = merged["guid"].nunique()

    logger.info(
        f"Merge result: {len(merged)} matched epochs, "
        f"{n_merged_guids} GUIDs "
        f"(TE had {n_te_guids}, KLD had {n_kld_guids})."
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
# Summary export
# ---------------------------------------------------------------------------


def export_summary(
    output_dir: Path,
    merged_df: pd.DataFrame,
    per_guid_df: pd.DataFrame,
    pooled_stats: Dict[str, Any],
    cross_guid_stats: Dict[str, Any],
    population_stats: Dict[str, Any],
) -> None:
    """Export all results to CSV and JSON files.

    Args:
        output_dir: Directory to write output files.
        merged_df: Merged TE-KLD DataFrame.
        per_guid_df: Per-GUID correlation DataFrame.
        pooled_stats: Pooled correlation results dict.
        cross_guid_stats: Cross-GUID correlation results dict.
        population_stats: Population-level test results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merged data
    merged_df.to_csv(output_dir / "merged_te_kld.csv", index=False)

    # Per-GUID correlations
    per_guid_df.to_csv(output_dir / "per_guid_correlations.csv", index=False)

    # JSON summary (exclude numpy arrays)
    summary = {
        "pooled": {
            k: v
            for k, v in pooled_stats.items()
            if not isinstance(v, np.ndarray)
        },
        "cross_guid": cross_guid_stats,
        "population_tests": population_stats,
        "data_summary": {
            "n_matched_epochs": int(len(merged_df)),
            "n_matched_guids": int(merged_df["guid"].nunique()),
            "n_guids_with_correlation": int(len(per_guid_df)),
            "ite_valid_range": [
                float(merged_df["ite_valid"].min()),
                float(merged_df["ite_valid"].max()),
            ],
            "kld_range": [
                float(merged_df["kld"].min()),
                float(merged_df["kld"].max()),
            ],
        },
    }

    with open(output_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Results exported to {output_dir}")
