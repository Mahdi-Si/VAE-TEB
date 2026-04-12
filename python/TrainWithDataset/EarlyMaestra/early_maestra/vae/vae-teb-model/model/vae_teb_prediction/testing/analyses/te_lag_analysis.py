"""Lag-resolved transfer-entropy analysis grouped by outcome class.

Aggregates ``outputs["te_lag_map"]`` (``(B, T, L) = kld_per_t ×
mean_heads(alpha)``) over time and samples, groups by outcome class,
and runs a per-lag Kruskal-Wallis test to detect lags that are
systematically different across classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.collectors import collect_te_lag_maps
from model.vae_teb_prediction.testing.visualizers import plot_te_lag_distribution

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None  # type: ignore[assignment]


def _bootstrap_ci(data: np.ndarray, n_boot: int = 500, alpha: float = 0.05) -> tuple:
    """Return ``(ci_lo, ci_hi)`` arrays for a per-lag bootstrap mean CI."""
    if data.size == 0:
        shape = (data.shape[1],) if data.ndim == 2 else (0,)
        return np.full(shape, np.nan), np.full(shape, np.nan)
    rng = np.random.default_rng(0)
    N = data.shape[0]
    idx = rng.integers(0, N, size=(n_boot, N))
    boot_means = np.nanmean(data[idx], axis=1)                # (n_boot, L)
    lo = np.nanpercentile(boot_means, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(boot_means, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def run_te_lag_class_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 1000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Aggregate TE lag attribution by class and test for group differences.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("te_lag")``).

    Returns:
        Dict with ``n_samples, best_lag_by_class, median_p_value,
        n_significant_lags``.
    """
    if max_samples <= 0:
        logger.info("te_lag_analysis: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("te_lag")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_te_lag_maps(runner, loader, max_samples)
    if df.empty:
        logger.warning("te_lag_analysis: no samples collected.")
        return {"n_samples": 0}

    per_sample_csv = output_dir / "te_lag_mean_per_sample.csv"
    df.to_csv(per_sample_csv, index=False)

    lag_cols = sorted(
        (c for c in df.columns if c.startswith("te_lag_mean_")),
        key=lambda c: int(c.split("_")[-1]),
    )
    lag_matrix = df[lag_cols].to_numpy(dtype=float)          # (N, L)
    labels = df["label"].to_numpy() if "label" in df.columns else np.full(len(df), None)
    N, L = lag_matrix.shape

    # Per-class mean + bootstrap CI.
    class_rows = []
    unique_labels = sorted(int(x) for x in np.unique(labels) if x is not None)
    for lab in unique_labels:
        mask = labels == lab
        subset = lag_matrix[mask]
        if subset.size == 0:
            continue
        mean_vec = np.nanmean(subset, axis=0)
        ci_lo, ci_hi = _bootstrap_ci(subset)
        for k in range(L):
            class_rows.append({
                "label": int(lab),
                "lag": k,
                "mean": float(mean_vec[k]),
                "ci_lo": float(ci_lo[k]),
                "ci_hi": float(ci_hi[k]),
                "n": int(mask.sum()),
            })
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(output_dir / "te_lag_by_class.csv", index=False)

    # Per-lag Kruskal-Wallis across classes.
    sig_rows = []
    if scipy_stats is not None and len(unique_labels) >= 2:
        for k in range(L):
            groups = [
                lag_matrix[labels == lab, k][np.isfinite(lag_matrix[labels == lab, k])]
                for lab in unique_labels
            ]
            groups = [g for g in groups if g.size > 0]
            if len(groups) < 2:
                sig_rows.append({"lag": k, "H_stat": np.nan, "p_value": np.nan})
                continue
            try:
                h, p = scipy_stats.kruskal(*groups)
            except ValueError:
                h, p = np.nan, np.nan
            sig_rows.append({"lag": k, "H_stat": float(h), "p_value": float(p)})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(output_dir / "significance.csv", index=False)

    # Class-mean plot.
    if lag_matrix.size > 0:
        class_names = {1: "HEALTHY", 2: "ACIDOSIS", 3: "HIE"}
        plot_te_lag_distribution(
            lag_matrix,
            labels,
            output_dir / "te_lag_by_class.pdf",
            class_names=class_names,
        )

    best_by_class: Dict[int, int] = {}
    for lab in unique_labels:
        mask = labels == lab
        subset = lag_matrix[mask]
        if subset.size == 0:
            continue
        mean_vec = np.nanmean(subset, axis=0)
        if np.all(np.isnan(mean_vec)):
            continue
        best_by_class[int(lab)] = int(np.nanargmax(mean_vec))

    if not sig_df.empty and sig_df["p_value"].notna().any():
        median_p = float(sig_df["p_value"].median())
        n_sig = int((sig_df["p_value"] < 0.05).sum())
    else:
        median_p = None
        n_sig = 0

    summary = {
        "n_samples": int(N),
        "best_lag_by_class": best_by_class,
        "median_p_value": median_p,
        "n_significant_lags": n_sig,
    }
    logger.info(
        f"te_lag_analysis: n={summary['n_samples']}, best_by_class={best_by_class}, "
        f"median_p={median_p}, n_sig={n_sig}"
    )
    return summary
