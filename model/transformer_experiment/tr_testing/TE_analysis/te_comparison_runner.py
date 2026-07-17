"""Orchestrator for model TE vs empirical TE comparison pipeline.

Wires together data loading, statistical analysis, and visualisation into
a single entry-point that can run standalone or be called from the testing
pipeline.

Example (standalone)::

    python -m model.transformer.tr_testing.TE_analysis.te_comparison_runner

Example (from code)::

    from model.transformer.tr_testing.TE_analysis.te_comparison_runner import (
        run_te_comparison,
    )
    results = run_te_comparison(
        model_csv_path="te_segment_data.csv",
        empirical_csv_path="te_record_epoch.csv",
        output_dir="results",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from loguru import logger

from model.transformer.tr_testing.TE_analysis.te_data_loader import (
    load_model_te_data,
    load_empirical_te_data,
    fuzzy_time_match,
    compute_data_quality_report,
)
from model.transformer.tr_testing.TE_analysis.te_comparison_analysis import (
    MODEL_TE_MEASURES,
    EMPIRICAL_TE_MEASURES,
    run_full_comparison,
)
from model.transformer.tr_testing.TE_analysis.te_comparison_visualizations import (
    plot_data_quality_summary,
    plot_correlation_heatmap,
    plot_per_dimension_kl_heatmap,
    plot_scatter_matrix,
    plot_trend_agreement,
    plot_leave_one_out,
    plot_bootstrap_distribution,
    plot_permutation_null,
    plot_mutual_information_comparison,
    plot_dtw_trajectories,
    plot_model_guid_trajectories,
    plot_empirical_guid_trajectories,
    generate_summary_table,
)


def run_te_comparison(
    model_csv_path: Union[str, Path],
    empirical_csv_path: Union[str, Path],
    output_dir: Union[str, Path],
    max_gap_seconds: float = 600.0,
    min_ite_valid_pc: float = 0.0,
    n_permutations: int = 10000,
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """Run the full model-TE vs empirical-TE comparison pipeline.

    Steps:
        1. Load both CSVs with GUID normalisation.
        2. Fuzzy nearest-neighbour time match.
        3. Compute data quality report.
        4. Run all statistical analyses.
        5. Generate all publication-quality plots.
        6. Export results (CSVs, JSON summary).
        7. Log console summary.

    Args:
        model_csv_path: Path to ``te_segment_data.csv`` (model TE output).
        empirical_csv_path: Path to ``te_record_epoch.csv`` (IDTxl output).
        output_dir: Directory for all output files.
        max_gap_seconds: Maximum time gap for fuzzy matching (seconds).
        min_ite_valid_pc: Minimum ``ite_valid_pc`` to keep empirical epochs.
        n_permutations: Number of permutations for significance tests.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Comprehensive results dict containing all analysis outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Step 1: Load data
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Model TE vs Empirical TE Comparison Pipeline")
    logger.info("=" * 60)

    model_df = load_model_te_data(model_csv_path)
    empirical_df = load_empirical_te_data(empirical_csv_path, min_ite_valid_pc)

    # ---------------------------------------------------------------
    # Step 2: Fuzzy time matching
    # ---------------------------------------------------------------
    merged_df = fuzzy_time_match(
        model_df, empirical_df, max_gap_seconds=max_gap_seconds
    )

    if len(merged_df) == 0:
        logger.error("No matched pairs found. Aborting pipeline.")
        return {"error": "no_matches", "model_guids": len(model_df["guid"].unique()),
                "empirical_guids": len(empirical_df["guid"].unique())}

    # Save merged data
    merged_df.to_csv(output_dir / "merged_data.csv", index=False)

    # ---------------------------------------------------------------
    # Step 3: Data quality report
    # ---------------------------------------------------------------
    quality_report = compute_data_quality_report(
        model_df, empirical_df, merged_df, max_gap_seconds=max_gap_seconds
    )

    with open(output_dir / "data_quality.json", "w") as f:
        json.dump(quality_report, f, indent=2, default=str)

    # ---------------------------------------------------------------
    # Step 4: Statistical analyses
    # ---------------------------------------------------------------
    results = run_full_comparison(
        merged_df, output_dir,
        model_df=model_df,
        empirical_df=empirical_df,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )

    # ---------------------------------------------------------------
    # Step 5: Visualisations
    # ---------------------------------------------------------------
    logger.info("Generating visualisations...")
    plots: Dict[str, Optional[str]] = {}

    def _try_plot(name: str, fn, *args, **kwargs) -> None:
        """Attempt a plot and log failures without crashing."""
        try:
            path = fn(*args, **kwargs)
            plots[name] = str(path)
        except Exception as e:
            logger.warning(f"Plot '{name}' failed: {e}")
            plots[name] = None

    # Available columns for scatter matrix
    model_cols = [c for c in MODEL_TE_MEASURES if c in merged_df.columns]
    empirical_cols = [c for c in EMPIRICAL_TE_MEASURES if c in merged_df.columns]
    primary_model = "kl_mean" if "kl_mean" in model_cols else model_cols[0] if model_cols else None
    primary_empirical = "ite_valid" if "ite_valid" in empirical_cols else empirical_cols[0] if empirical_cols else None

    # 5a. Data quality summary
    _try_plot("data_quality_summary",
              plot_data_quality_summary, quality_report, output_dir)

    # 5b. Correlation heatmap
    corr_sp = results.get("correlation_spearman")
    pval_sp = results.get("pvalue_spearman")
    if corr_sp is not None and pval_sp is not None:
        _try_plot("correlation_heatmap",
                  plot_correlation_heatmap, corr_sp, pval_sp, output_dir)

    # 5c. Per-dimension heatmap
    dim_df = results.get("per_dimension")
    if dim_df is not None and len(dim_df) > 0:
        _try_plot("per_dimension_heatmap",
                  plot_per_dimension_kl_heatmap, dim_df, output_dir)

    # 5d. Scatter matrix
    perm_tests_full = results.get("_permutation_tests_full", {})
    # Extract observed correlations for scatter annotations
    perm_for_scatter = {
        k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
        for k, v in perm_tests_full.items()
    }
    if model_cols and empirical_cols:
        _try_plot("scatter_matrix",
                  plot_scatter_matrix, merged_df, model_cols, empirical_cols,
                  output_dir, perm_results=perm_for_scatter)

    # 5e. (removed — trajectory overlay merged into 5l/5m GUID trajectory figures)

    # 5f. Trend agreement
    trend_full = results.get("_trend_agreement_full", {})
    if trend_full and "error" not in trend_full and primary_model and primary_empirical:
        _try_plot("trend_agreement",
                  plot_trend_agreement, trend_full, merged_df,
                  primary_model, primary_empirical, output_dir)

    # 5g. Leave-one-out
    loo = results.get("leave_one_out", {})
    if "error" not in loo:
        _try_plot("leave_one_out",
                  plot_leave_one_out, loo, output_dir)

    # 5h. Bootstrap distribution
    boot_full = results.get("_bootstrap_full", {})
    if boot_full:
        _try_plot("bootstrap_distribution",
                  plot_bootstrap_distribution, boot_full, output_dir)

    # 5i. Permutation null (primary pair)
    if primary_model and primary_empirical:
        perm_key = f"{primary_model}_vs_{primary_empirical}"
        perm_primary = perm_tests_full.get(perm_key, {})
        if perm_primary and "error" not in perm_primary:
            _try_plot("permutation_null",
                      plot_permutation_null, perm_primary, output_dir,
                      title=f"Permutation Test: {primary_model} vs {primary_empirical}")

    # 5j. MI vs correlation comparison
    mi = results.get("mutual_information", {})
    corr_for_mi = {}
    if corr_sp is not None:
        for mc in corr_sp.index:
            for ec in corr_sp.columns:
                corr_for_mi[f"{mc}_vs_{ec}"] = corr_sp.loc[mc, ec]
    if mi and corr_for_mi:
        _try_plot("mi_vs_correlation",
                  plot_mutual_information_comparison, mi, corr_for_mi, output_dir)

    # 5k. DTW trajectories
    dtw_full = results.get("_dtw_full", {})
    if dtw_full and primary_model and primary_empirical:
        _try_plot("dtw_trajectories",
                  plot_dtw_trajectories, dtw_full, merged_df,
                  primary_model, primary_empirical, output_dir)

    # 5l. Model GUID trajectories (10 GUIDs, with empirical overlay)
    common_guid_list = sorted(merged_df["guid"].unique().tolist())
    _try_plot("model_guid_trajectories",
              plot_model_guid_trajectories, model_df, output_dir,
              empirical_df=empirical_df,
              common_guids=common_guid_list, n_guids=10)

    # 5m. Empirical GUID trajectories (10 GUIDs, with model overlay)
    _try_plot("empirical_guid_trajectories",
              plot_empirical_guid_trajectories, empirical_df, output_dir,
              model_df=model_df,
              common_guids=common_guid_list, n_guids=10)

    # 5n. Summary table
    _try_plot("summary_table",
              generate_summary_table, results, output_dir)

    results["plots"] = plots

    # ---------------------------------------------------------------
    # Step 6: Console summary
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info(f"  Matched pairs: {len(merged_df)}")
    logger.info(f"  Matched GUIDs: {merged_df['guid'].nunique()}")
    logger.info(f"  Plots generated: {sum(1 for v in plots.values() if v)}/{len(plots)}")
    logger.info(f"  Output directory: {output_dir}")

    if primary_model and primary_empirical:
        boot = results.get("bootstrap", {})
        logger.info(f"  Primary pair: {primary_model} vs {primary_empirical}")
        logger.info(f"    Spearman ρ: {boot.get('observed', 'N/A')}")
        logger.info(f"    95% CI (block): [{boot.get('ci_lo', 'N/A')}, {boot.get('ci_hi', 'N/A')}]")
        perm_key = f"{primary_model}_vs_{primary_empirical}"
        perm_p = results.get("permutation_tests", {}).get(perm_key, {}).get("p_value", "N/A")
        logger.info(f"    Permutation p: {perm_p}")

    logger.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default paths: CSVs co-located with this script
    _this_dir = Path(__file__).resolve().parent
    _model_csv = _this_dir / "te_segment_data.csv"
    _empirical_csv = _this_dir / "te_record_epoch.csv"
    _output = _this_dir / "results"

    run_te_comparison(
        model_csv_path=_model_csv,
        empirical_csv_path=_empirical_csv,
        output_dir=_output,
        max_gap_seconds=600.0,
        min_ite_valid_pc=0.0,
        n_permutations=10000,
        n_bootstrap=5000,
    )
