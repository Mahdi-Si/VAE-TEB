"""
Statistical significance testing for comparing histogram metrics across 3 independent datasets.

This module performs non-parametric statistical tests to compare VAF, MSE, SNR, and KLD
metrics across three independent datasets (e.g., cross-validation folds).

Usage:
    Set data_paths in __main__ and run:
    python significance_tests.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# Import visualization constants from existing module
from model.vae_teb_prediction.testing.visualizers import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    SAVE_DPI,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_histogram_data(path: Path) -> pd.DataFrame:
    """
    Load histogram metrics from CSV file.

    Args:
        path: Path to histogram_metrics.csv

    Returns:
        DataFrame with columns [guid, epoch, label, vaf, mse, snr, kld]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    required_cols = ["vaf", "mse", "snr", "kld"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df)} samples from {path}")
    return df


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def compute_descriptive_stats(data: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for a single dataset."""
    return {
        "n": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data, ddof=1)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def kruskal_wallis_test(
    datasets: List[np.ndarray], labels: List[str]
) -> Dict[str, Any]:
    """
    Perform Kruskal-Wallis H-test for 3 independent samples.

    Non-parametric test for comparing 3+ independent groups.
    Null hypothesis: All groups have the same distribution.

    Returns:
        Dict with test statistic, p-value, effect size (epsilon-squared)
    """
    h_stat, p_value = stats.kruskal(*datasets)

    # Compute effect size (epsilon-squared)
    n_total = sum(len(d) for d in datasets)
    epsilon_sq = (h_stat - len(datasets) + 1) / (n_total - len(datasets))

    return {
        "test": "Kruskal-Wallis",
        "statistic": float(h_stat),
        "p_value": float(p_value),
        "is_significant": p_value < 0.05,
        "effect_size": float(epsilon_sq),
        "effect_interpretation": _interpret_effect_size(epsilon_sq),
    }


def mann_whitney_pairwise(
    datasets: List[np.ndarray], labels: List[str], alpha: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Perform pairwise Mann-Whitney U tests with Bonferroni correction.

    Tests all pairs: (0 vs 1), (0 vs 2), (1 vs 2)

    Args:
        datasets: List of 3 arrays
        labels: List of 3 dataset names
        alpha: Significance level (default 0.05)

    Returns:
        List of 3 pairwise comparison results
    """
    n_comparisons = 3
    alpha_bonferroni = alpha / n_comparisons

    pairs = [(0, 1), (0, 2), (1, 2)]
    results = []

    for i, j in pairs:
        u_stat, p_value = stats.mannwhitneyu(
            datasets[i], datasets[j], alternative="two-sided"
        )

        # Compute rank-biserial correlation (effect size)
        n1, n2 = len(datasets[i]), len(datasets[j])
        r = 1 - (2 * u_stat) / (n1 * n2)

        results.append({
            "comparison": f"{labels[i]} vs {labels[j]}",
            "test": "Mann-Whitney U",
            "statistic": float(u_stat),
            "p_value": float(p_value),
            "p_bonferroni": float(p_value * n_comparisons),
            "is_significant": p_value < alpha_bonferroni,
            "alpha_bonferroni": alpha_bonferroni,
            "effect_size": float(r),
            "effect_interpretation": _interpret_effect_size(abs(r)),
        })

    return results


def _interpret_effect_size(effect: float) -> str:
    """Interpret effect size magnitude."""
    if effect < 0.1:
        return "negligible"
    elif effect < 0.3:
        return "small"
    elif effect < 0.5:
        return "medium"
    else:
        return "large"


# ============================================================================
# COMPARISON LOGIC
# ============================================================================

def compare_metric_across_datasets(
    datasets: List[pd.DataFrame],
    labels: List[str],
    metric_name: str,
) -> Dict[str, Any]:
    """
    Compare a single metric across 3 datasets.

    Args:
        datasets: List of 3 DataFrames with metric columns
        labels: List of 3 dataset names
        metric_name: Name of metric column (vaf, mse, snr, kld)

    Returns:
        Dict with descriptive stats, overall test, and pairwise tests
    """
    logger.info(f"Comparing {metric_name.upper()} across {len(datasets)} datasets")

    # Extract metric values
    metric_arrays = [df[metric_name].values for df in datasets]

    # Descriptive statistics
    descriptive = {
        label: compute_descriptive_stats(data)
        for label, data in zip(labels, metric_arrays)
    }

    # Overall test (Kruskal-Wallis)
    overall_test = kruskal_wallis_test(metric_arrays, labels)

    # Pairwise tests (only if overall test is significant)
    pairwise_tests = None
    if overall_test["is_significant"]:
        pairwise_tests = mann_whitney_pairwise(metric_arrays, labels)
    else:
        logger.info(f"  Overall test not significant (p={overall_test['p_value']:.4f}), skipping pairwise tests")

    return {
        "metric": metric_name,
        "descriptive_statistics": descriptive,
        "overall_test": overall_test,
        "pairwise_tests": pairwise_tests,
    }


def compare_histogram_datasets(
    data_paths: List[Path],
    dataset_labels: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    MAIN FUNCTION: Compare histogram metrics across 3 datasets.

    Performs statistical comparisons for VAF, MSE, SNR, and KLD metrics
    across 3 independent datasets and generates comparison visualizations.

    Args:
        data_paths: List of 3 paths to histogram_metrics.csv files
        dataset_labels: List of 3 dataset names (e.g., ["Fold 1", "Fold 2", "Fold 3"])
        output_dir: Directory to save results

    Returns:
        Dict with results for all metrics
    """
    if len(data_paths) != 3:
        raise ValueError(f"Expected 3 data paths, got {len(data_paths)}")
    if len(dataset_labels) != 3:
        raise ValueError(f"Expected 3 labels, got {len(dataset_labels)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Starting significance testing for histogram metrics")
    logger.info("="*80)

    # Load all datasets
    datasets = [load_histogram_data(path) for path in data_paths]

    # Compare each metric
    metrics = ["vaf", "mse", "snr", "kld"]
    results = {}

    for metric in metrics:
        result = compare_metric_across_datasets(datasets, dataset_labels, metric)
        results[metric] = result

        # Log summary
        logger.info(f"\n{metric.upper()} Comparison:")
        logger.info(f"  Overall test: p={result['overall_test']['p_value']:.4e}, "
                   f"significant={result['overall_test']['is_significant']}")
        if result['pairwise_tests']:
            for pw in result['pairwise_tests']:
                logger.info(f"    {pw['comparison']}: p={pw['p_value']:.4e}, "
                          f"p_bonf={pw['p_bonferroni']:.4f}, "
                          f"sig={pw['is_significant']}")

    # Save results
    _save_results(results, dataset_labels, output_dir)

    # Create visualizations
    _plot_all_comparisons(datasets, dataset_labels, results, output_dir)

    logger.info("="*80)
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    logger.info("="*80)

    return {"results": results, "dataset_labels": dataset_labels}


# ============================================================================
# VISUALIZATION
# ============================================================================

def _plot_metric_comparison(
    datasets: List[pd.DataFrame],
    labels: List[str],
    metric_name: str,
    result: Dict[str, Any],
    output_path: Path,
) -> None:
    """Create box+violin plot for a single metric comparison."""
    metric_config = {
        "vaf": ("Variance Accounted For (VAF)", "", COLOR_BLUE),
        "mse": ("Mean Squared Error (MSE)", "", COLOR_GREEN),
        "snr": ("Signal-to-Noise Ratio (SNR)", "dB", COLOR_ORANGE),
        "kld": ("Transfer Entropy (KLD)", "bits", COLOR_PURPLE),
    }

    title, unit, color = metric_config[metric_name]
    xlabel = f"{metric_name.upper()}" + (f" ({unit})" if unit else "")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    data_arrays = [df[metric_name].values for df in datasets]
    positions = np.arange(1, len(labels) + 1)

    # Violin plots
    parts = ax.violinplot(
        data_arrays,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    # Box plots (overlaid)
    bp = ax.boxplot(
        data_arrays,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        boxprops=dict(facecolor=color, alpha=0.6),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color=color),
        capprops=dict(color=color),
    )

    # Statistical annotations
    overall = result["overall_test"]
    stats_text = (
        f"Kruskal-Wallis H-test:\n"
        f"  H = {overall['statistic']:.2f}\n"
        f"  p = {overall['p_value']:.4e}\n"
        f"  Effect size (ε²) = {overall['effect_size']:.3f} ({overall['effect_interpretation']})\n"
        f"  {'**SIGNIFICANT**' if overall['is_significant'] else 'Not significant'}"
    )

    if result['pairwise_tests']:
        stats_text += "\n\nPairwise (Mann-Whitney + Bonferroni):"
        for pw in result['pairwise_tests']:
            sig_marker = "***" if pw['is_significant'] else "n.s."
            stats_text += f"\n  {pw['comparison']}: p={pw['p_bonferroni']:.4f} {sig_marker}"

    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
    )

    # Descriptive stats table
    desc_text = "Median (IQR):\n"
    for label in labels:
        stats = result['descriptive_statistics'][label]
        desc_text += f"  {label}: {stats['median']:.4f} ({stats['iqr']:.4f})\n"

    ax.text(
        0.02, 0.97, desc_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
    )

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved comparison plot: {output_path}")


def _plot_all_comparisons(
    datasets: List[pd.DataFrame],
    labels: List[str],
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate comparison plots for all metrics."""
    logger.info("\nGenerating comparison plots...")

    for metric in ["vaf", "mse", "snr", "kld"]:
        output_path = output_dir / f"{metric}_comparison.png"
        _plot_metric_comparison(datasets, labels, metric, results[metric], output_path)

    # Summary plot (2x2 grid with median bars)
    _plot_summary_comparison(datasets, labels, results, output_dir)


def _plot_summary_comparison(
    datasets: List[pd.DataFrame],
    labels: List[str],
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create 2x2 summary plot with median comparisons."""
    metrics_config = [
        ("vaf", "VAF", COLOR_BLUE),
        ("mse", "MSE", COLOR_GREEN),
        ("snr", "SNR (dB)", COLOR_ORANGE),
        ("kld", "KLD (bits)", COLOR_PURPLE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric, label, color) in enumerate(metrics_config):
        ax = axes[idx]
        result = results[metric]

        # Extract medians
        medians = [result['descriptive_statistics'][lbl]['median'] for lbl in labels]
        iqrs = [result['descriptive_statistics'][lbl]['iqr'] for lbl in labels]

        # Bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, medians, yerr=iqrs, capsize=5, color=color, alpha=0.7, edgecolor='black')

        # Add significance marker
        if result['overall_test']['is_significant']:
            ax.text(
                0.5, 0.95, '**SIGNIFICANT**',
                transform=ax.transAxes,
                ha='center', va='top',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
            )

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"{label} Comparison", fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle("Metrics Comparison Summary (Median ± IQR)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison_summary.png", dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved summary plot: {output_dir / 'metrics_comparison_summary.png'}")


# ============================================================================
# OUTPUT SAVING
# ============================================================================

def _save_results(
    results: Dict[str, Any],
    labels: List[str],
    output_dir: Path,
) -> None:
    """Save statistical results in JSON, CSV, and text formats."""
    # JSON (full results)
    json_path = output_dir / "statistical_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved full results: {json_path}")

    # CSV (summary table)
    csv_rows = []
    for metric, result in results.items():
        overall = result["overall_test"]
        row = {
            "metric": metric.upper(),
            "kruskal_wallis_H": overall["statistic"],
            "kruskal_wallis_p": overall["p_value"],
            "overall_significant": overall["is_significant"],
            "effect_size_epsilon_sq": overall["effect_size"],
            "effect_interpretation": overall["effect_interpretation"],
        }

        # Add pairwise results
        if result["pairwise_tests"]:
            for pw in result["pairwise_tests"]:
                comp_key = pw["comparison"].replace(" ", "_").replace("vs", "vs")
                row[f"{comp_key}_p_bonferroni"] = pw["p_bonferroni"]
                row[f"{comp_key}_significant"] = pw["is_significant"]

        csv_rows.append(row)

    csv_path = output_dir / "statistical_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Saved summary table: {csv_path}")

    # Text report (human-readable)
    txt_path = output_dir / "statistical_report.txt"
    with open(txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL SIGNIFICANCE TESTING REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Datasets compared: {', '.join(labels)}\n")
        f.write(f"Statistical test: Kruskal-Wallis H-test (non-parametric ANOVA)\n")
        f.write(f"Post-hoc test: Mann-Whitney U with Bonferroni correction (α=0.05/3=0.0167)\n\n")

        for metric, result in results.items():
            f.write("-"*80 + "\n")
            f.write(f"{metric.upper()} Analysis\n")
            f.write("-"*80 + "\n\n")

            # Descriptive stats
            f.write("Descriptive Statistics:\n")
            for label in labels:
                stats = result['descriptive_statistics'][label]
                f.write(f"  {label}:\n")
                f.write(f"    n = {stats['n']}\n")
                f.write(f"    median = {stats['median']:.6f}, IQR = {stats['iqr']:.6f}\n")
                f.write(f"    mean = {stats['mean']:.6f}, std = {stats['std']:.6f}\n")
                f.write(f"    range = [{stats['min']:.6f}, {stats['max']:.6f}]\n\n")

            # Overall test
            overall = result['overall_test']
            f.write(f"Kruskal-Wallis H-test:\n")
            f.write(f"  H = {overall['statistic']:.4f}\n")
            f.write(f"  p-value = {overall['p_value']:.4e}\n")
            f.write(f"  Significant: {overall['is_significant']} (α=0.05)\n")
            f.write(f"  Effect size (ε²) = {overall['effect_size']:.4f} ({overall['effect_interpretation']})\n\n")

            # Pairwise tests
            if result['pairwise_tests']:
                f.write("Pairwise Comparisons (Mann-Whitney U + Bonferroni):\n")
                for pw in result['pairwise_tests']:
                    f.write(f"  {pw['comparison']}:\n")
                    f.write(f"    U = {pw['statistic']:.2f}\n")
                    f.write(f"    p-value = {pw['p_value']:.4e}\n")
                    f.write(f"    p-value (Bonferroni) = {pw['p_bonferroni']:.4f}\n")
                    f.write(f"    Significant: {pw['is_significant']} (α=0.0167)\n")
                    f.write(f"    Effect size (r) = {pw['effect_size']:.4f} ({pw['effect_interpretation']})\n\n")
            else:
                f.write("Pairwise comparisons not performed (overall test not significant)\n\n")

    logger.info(f"Saved text report: {txt_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - SET YOUR PATHS HERE
    # ========================================================================

    # Paths to the 3 histogram_metrics.csv files
    DATA_PATHS = [
        "results/fold_1/histograms/histogram_metrics.csv",
        "results/fold_2/histograms/histogram_metrics.csv",
        "results/fold_3/histograms/histogram_metrics.csv",
    ]

    # Labels for the 3 datasets
    DATASET_LABELS = [
        "Fold 1",
        "Fold 2",
        "Fold 3",
    ]

    # Output directory for significance testing results
    OUTPUT_DIR = "results/significance_analysis"

    # ========================================================================
    # RUN ANALYSIS
    # ========================================================================

    try:
        results = compare_histogram_datasets(
            data_paths=[Path(p) for p in DATA_PATHS],
            dataset_labels=DATASET_LABELS,
            output_dir=Path(OUTPUT_DIR),
        )
        logger.info("\n✓ Analysis completed successfully!")

    except Exception as e:
        logger.error(f"\n✗ Analysis failed: {e}")
        raise
