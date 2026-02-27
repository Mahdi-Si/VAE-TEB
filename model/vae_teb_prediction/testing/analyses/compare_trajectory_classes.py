"""
Standalone cross-class trajectory comparison script.

Loads per-GUID trajectory feature CSVs from multiple runs (each labeled with
a class) and performs statistical comparison between classes.

Usage:
    python -m model.vae_teb_prediction.testing.analyses.compare_trajectory_classes \
        --runs /path/to/healthy_run/trajectory/guid_trajectory_features.csv:healthy \
               /path/to/acidosis_run/trajectory/guid_trajectory_features.csv:acidosis \
        --stitched /path/to/healthy_run/trajectory/latent_trajectories_stitched.csv:healthy \
                   /path/to/acidosis_run/trajectory/latent_trajectories_stitched.csv:acidosis \
        --output comparison_results/

The --runs arg points to the per-GUID feature CSVs with a class label after
the colon. The --stitched arg is optional — if provided, the script also
computes FID and MMD on the full latent distributions from the stitched CSVs.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats
from scipy.linalg import sqrtm


# ---------------------------------------------------------------------------
# Distributional distance metrics
# ---------------------------------------------------------------------------

def compute_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """
    Compute the Frechet distance between two multivariate Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Args:
        mu1: Mean of distribution 1, shape (D,).
        sigma1: Covariance of distribution 1, shape (D, D).
        mu2: Mean of distribution 2, shape (D,).
        sigma2: Covariance of distribution 2, shape (D, D).

    Returns:
        Frechet distance (scalar >= 0).
    """
    diff = mu1 - mu2
    mean_term = float(diff @ diff)

    covmean = sqrtm(sigma1 @ sigma2)
    # sqrtm can return complex values due to numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_term = float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return mean_term + trace_term


def compute_mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Args:
        X: Samples from distribution 1, shape (N, D).
        Y: Samples from distribution 2, shape (M, D).
        gamma: RBF kernel bandwidth (1 / (2 * sigma^2)).
            If None, uses median heuristic.

    Returns:
        MMD value (scalar >= 0).
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if gamma is None:
        # Median heuristic for bandwidth
        XY = np.vstack([X, Y])
        from scipy.spatial.distance import pdist
        dists = pdist(XY, metric="euclidean")
        median_dist = np.median(dists)
        if median_dist > 0:
            gamma = 1.0 / (2.0 * median_dist ** 2)
        else:
            gamma = 1.0

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    n = K_xx.shape[0]
    m = K_yy.shape[0]

    # Unbiased estimator: exclude diagonal
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    mmd_sq = (
        K_xx.sum() / (n * (n - 1))
        + K_yy.sum() / (m * (m - 1))
        - 2.0 * K_xy.sum() / (n * m)
    )
    return float(max(0.0, mmd_sq) ** 0.5)


# ---------------------------------------------------------------------------
# Per-feature statistical comparison
# ---------------------------------------------------------------------------

def compare_features_by_class(
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Run Kruskal-Wallis H-test across all classes for each trajectory feature,
    plus post-hoc pairwise Mann-Whitney U with Bonferroni correction.

    Args:
        merged_df: Merged DataFrame with features and class column.
        feature_cols: List of feature column names.
        class_col: Column name for class labels.

    Returns:
        DataFrame with one row per (feature, pair) combination, containing
        test statistics, p-values, and effect sizes.
    """
    classes = sorted(merged_df[class_col].unique())
    results = []

    for feat in feature_cols:
        if feat not in merged_df.columns:
            continue

        # Clean data
        df_clean = merged_df[[feat, class_col]].replace([np.inf, -np.inf], np.nan).dropna()
        groups = [df_clean.loc[df_clean[class_col] == c, feat].values for c in classes]
        groups = [g for g in groups if g.size > 0]

        if len(groups) < 2:
            continue

        # Kruskal-Wallis omnibus test
        try:
            h_stat, kw_p = scipy_stats.kruskal(*groups)
        except ValueError:
            h_stat, kw_p = np.nan, np.nan

        # Pairwise Mann-Whitney U with Bonferroni correction
        n_pairs = len(list(combinations(range(len(classes)), 2)))
        for i, j in combinations(range(len(classes)), 2):
            ci, cj = classes[i], classes[j]
            xi = df_clean.loc[df_clean[class_col] == ci, feat].values
            xj = df_clean.loc[df_clean[class_col] == cj, feat].values

            if xi.size < 2 or xj.size < 2:
                continue

            try:
                u_stat, mw_p = scipy_stats.mannwhitneyu(xi, xj, alternative="two-sided")
                # Rank-biserial correlation as effect size
                n1, n2 = xi.size, xj.size
                r = 1.0 - (2.0 * u_stat) / (n1 * n2)
            except ValueError:
                u_stat, mw_p, r = np.nan, np.nan, np.nan

            bonferroni_p = min(mw_p * n_pairs, 1.0) if np.isfinite(mw_p) else np.nan

            results.append({
                "feature": feat,
                "pair": f"{ci}_vs_{cj}",
                "class_1": ci,
                "class_2": cj,
                "n_1": xi.size,
                "n_2": xj.size,
                "kruskal_wallis_H": h_stat,
                "kruskal_wallis_p": kw_p,
                "mann_whitney_U": u_stat,
                "mann_whitney_p": mw_p,
                "bonferroni_p": bonferroni_p,
                "effect_size_r": r,
            })

    return pd.DataFrame(results)


def compare_dimensions_by_class(
    merged_df: pd.DataFrame,
    z_cols: List[str],
    class_col: str = "class",
) -> pd.DataFrame:
    """
    Mann-Whitney U on each latent dimension between class pairs.

    Identifies which latent dimensions differ most between classes.

    Args:
        merged_df: Merged DataFrame with z-columns and class column.
        z_cols: List of latent dimension column names (e.g., ["z0", "z1", ...]).
        class_col: Column name for class labels.

    Returns:
        DataFrame with test results per (dimension, pair).
    """
    classes = sorted(merged_df[class_col].unique())
    results = []

    for z_col in z_cols:
        if z_col not in merged_df.columns:
            continue

        for ci, cj in combinations(classes, 2):
            xi = merged_df.loc[merged_df[class_col] == ci, z_col].dropna().values
            xj = merged_df.loc[merged_df[class_col] == cj, z_col].dropna().values

            if xi.size < 2 or xj.size < 2:
                continue

            try:
                u_stat, p_val = scipy_stats.mannwhitneyu(xi, xj, alternative="two-sided")
                n1, n2 = xi.size, xj.size
                r = 1.0 - (2.0 * u_stat) / (n1 * n2)
            except ValueError:
                u_stat, p_val, r = np.nan, np.nan, np.nan

            results.append({
                "dimension": z_col,
                "pair": f"{ci}_vs_{cj}",
                "class_1": ci,
                "class_2": cj,
                "n_1": xi.size,
                "n_2": xj.size,
                "mann_whitney_U": u_stat,
                "p_value": p_val,
                "effect_size_r": r,
                "mean_1": float(np.mean(xi)),
                "mean_2": float(np.mean(xj)),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_feature_boxplots(
    merged_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
    class_col: str = "class",
) -> None:
    """Generate box plots of features by class using the visualizer."""
    try:
        from model.vae_teb_prediction.testing.visualizers import plot_feature_boxplots
        plot_feature_boxplots(
            merged_df,
            feature_cols,
            output_dir / "trajectory_features_by_class.pdf",
            class_col=class_col,
        )
    except Exception as e:
        logger.warning(f"Feature boxplots failed: {e}")


def _plot_dimension_heatmap(
    dim_comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate heatmap of per-dimension p-values between class pairs."""
    try:
        from model.vae_teb_prediction.testing.visualizers import plot_dimension_significance_heatmap
        plot_dimension_significance_heatmap(
            dim_comparison_df,
            output_dir / "dimension_significance_heatmap.pdf",
        )
    except Exception as e:
        logger.warning(f"Dimension heatmap failed: {e}")


def _plot_distributional_distances(
    pairwise_summary: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate bar chart of FID/MMD between class pairs."""
    if not pairwise_summary:
        return
    try:
        from model.vae_teb_prediction.testing.visualizers import plot_distributional_distances
        summary_df = pd.DataFrame(pairwise_summary)
        plot_distributional_distances(
            summary_df,
            output_dir / "distributional_distances.pdf",
        )
    except Exception as e:
        logger.warning(f"Distributional distance plots failed: {e}")


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def run_comparison(
    run_specs: List[Tuple[str, str]],
    output_dir: str,
    stitched_specs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Run full cross-class trajectory comparison.

    Args:
        run_specs: List of (csv_path, class_label) tuples pointing to
            guid_trajectory_features.csv files.
        output_dir: Directory for saving outputs.
        stitched_specs: Optional list of (csv_path, class_label) tuples
            pointing to latent_trajectories_stitched.csv files.

    Returns:
        Dict with comparison results.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and merge per-GUID feature CSVs ----
    dfs = []
    for csv_path, class_label in run_specs:
        p = Path(csv_path)
        if not p.exists():
            logger.warning(f"Feature CSV not found: {p}")
            continue
        df = pd.read_csv(p)
        df["class"] = class_label
        dfs.append(df)
        logger.info(f"Loaded {len(df)} GUIDs from {p} (class={class_label})")

    if not dfs:
        logger.error("No valid feature CSVs found")
        return {"error": "no data"}

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(out / "merged_features.csv", index=False)
    logger.info(f"Merged {len(merged)} GUIDs across {len(dfs)} classes")

    results: Dict[str, Any] = {"n_guids": len(merged), "classes": sorted(merged["class"].unique().tolist())}

    # ---- 2. Per-feature comparison ----
    feature_cols = [
        c for c in merged.columns
        if c not in ("guid", "label", "class", "n_epochs", "duration_hours")
        and not c.startswith("z")
    ]
    if feature_cols:
        feat_comparison = compare_features_by_class(merged, feature_cols)
        feat_comparison.to_csv(out / "feature_comparison.csv", index=False)
        results["n_features_tested"] = len(feature_cols)
        results["significant_features"] = int((feat_comparison["bonferroni_p"] < 0.05).sum())
        logger.info(f"Feature comparison: {results['significant_features']}/{len(feature_cols)} significant (p<0.05 Bonferroni)")

        # Box plots
        _plot_feature_boxplots(merged, feature_cols, out)

    # ---- 3. Distributional comparison (FID, MMD) from stitched CSVs ----
    pairwise_summary = []

    if stitched_specs:
        stitched_dfs = []
        for csv_path, class_label in stitched_specs:
            p = Path(csv_path)
            if not p.exists():
                logger.warning(f"Stitched CSV not found: {p}")
                continue
            df = pd.read_csv(p)
            df["class"] = class_label
            stitched_dfs.append(df)
            logger.info(f"Loaded {len(df)} timesteps from {p} (class={class_label})")

        if len(stitched_dfs) >= 2:
            stitched_merged = pd.concat(stitched_dfs, ignore_index=True)
            z_cols = [c for c in stitched_merged.columns if c.startswith("z") and c[1:].isdigit()]

            if z_cols:
                classes = sorted(stitched_merged["class"].unique())

                # Pairwise FID and MMD
                for ci, cj in combinations(classes, 2):
                    Xi = stitched_merged.loc[stitched_merged["class"] == ci, z_cols].dropna().values
                    Xj = stitched_merged.loc[stitched_merged["class"] == cj, z_cols].dropna().values

                    if Xi.shape[0] < 10 or Xj.shape[0] < 10:
                        continue

                    # Subsample for efficiency if too large
                    max_samples = 10000
                    if Xi.shape[0] > max_samples:
                        Xi = Xi[np.random.RandomState(42).choice(Xi.shape[0], max_samples, replace=False)]
                    if Xj.shape[0] > max_samples:
                        Xj = Xj[np.random.RandomState(42).choice(Xj.shape[0], max_samples, replace=False)]

                    # FID
                    mu1, sigma1 = Xi.mean(axis=0), np.cov(Xi, rowvar=False)
                    mu2, sigma2 = Xj.mean(axis=0), np.cov(Xj, rowvar=False)
                    fid = compute_frechet_distance(mu1, sigma1, mu2, sigma2)

                    # MMD
                    mmd = compute_mmd_rbf(Xi, Xj)

                    pairwise_summary.append({
                        "pair": f"{ci}_vs_{cj}",
                        "class_1": ci,
                        "class_2": cj,
                        "n_1": Xi.shape[0],
                        "n_2": Xj.shape[0],
                        "FID": fid,
                        "MMD": mmd,
                    })
                    logger.info(f"{ci} vs {cj}: FID={fid:.4f}, MMD={mmd:.6f}")

                # Per-dimension comparison
                dim_comparison = compare_dimensions_by_class(stitched_merged, z_cols)
                dim_comparison.to_csv(out / "dimension_comparison.csv", index=False)
                results["n_dimensions_tested"] = len(z_cols)

                # Dimension significance heatmap
                _plot_dimension_heatmap(dim_comparison, out)

    # Distributional distance plots
    _plot_distributional_distances(pairwise_summary, out)

    if pairwise_summary:
        summary_df = pd.DataFrame(pairwise_summary)
        summary_df.to_csv(out / "class_comparison_summary.csv", index=False)
        results["pairwise"] = pairwise_summary

    # ---- 4. Save full report ----
    with open(out / "class_comparison_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Comparison results saved to {out}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_spec(spec: str) -> Tuple[str, str]:
    """Parse 'path:label' specification."""
    parts = spec.rsplit(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid spec '{spec}'. Expected format: /path/to/csv:class_label"
        )
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Compare latent trajectory features across outcome classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m model.vae_teb_prediction.testing.analyses.compare_trajectory_classes \\
        --runs output_healthy/trajectory/guid_trajectory_features.csv:healthy \\
               output_acidosis/trajectory/guid_trajectory_features.csv:acidosis \\
        --stitched output_healthy/trajectory/latent_trajectories_stitched.csv:healthy \\
                   output_acidosis/trajectory/latent_trajectories_stitched.csv:acidosis \\
        --output comparison_results/
        """,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Per-GUID feature CSV paths with class labels (format: path:label)",
    )
    parser.add_argument(
        "--stitched",
        nargs="*",
        default=None,
        help="Optional stitched trajectory CSV paths with class labels (format: path:label). "
             "If provided, FID and MMD are computed on the latent distributions.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    run_specs = [_parse_spec(s) for s in args.runs]
    stitched_specs = [_parse_spec(s) for s in args.stitched] if args.stitched else None

    run_comparison(run_specs, args.output, stitched_specs)


if __name__ == "__main__":
    main()
