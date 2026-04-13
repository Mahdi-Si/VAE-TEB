"""Main orchestrator for empirical TE vs VAE-KLD comparison analysis.

Loads empirical Transfer Entropy (from IDTxl CSV) and VAE KL divergence
values, matches them with a fuzzy ±5-min tolerance, and runs a full
statistical and visualization suite.

Supports two KLD sources:

1. **Pre-computed**: load per-epoch KLD from a metrics CSV produced by a
   prior test run (fast, no GPU needed).
2. **Live inference**: run model inference on-the-fly for the TE GUIDs
   (requires checkpoint + HDF5 data + GPU).

All parameters are set in the ``if __name__ == "__main__"`` block below
so the script can be run as-is.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

# Ensure project root is importable.
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model.vae_teb_prediction.testing.TE_Calculated.te_data_loader import (
    compute_data_quality_report,
    get_te_guids,
    load_te_data,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (
    CANDIDATE_KLD_COLS,
    CANDIDATE_TE_COLS,
    cluster_aware_bootstrap,
    compute_cross_guid_correlation,
    compute_per_guid_correlations,
    compute_pooled_correlation,
    concordance_analysis,
    correlation_matrix,
    export_summary,
    leave_one_guid_out_sensitivity,
    load_kld_from_inference,
    load_kld_from_metrics_csv,
    merge_te_kld,
    mutual_information_knn,
    per_dimension_kl_analysis,
    permutation_test_correlation,
    population_level_test,
    trend_agreement_analysis,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_dtw import (
    dtw_align_per_guid,
    dtw_backend_name,
    paired_dataset_from_dtw,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_visualizations import (
    plot_bootstrap_ci,
    plot_bootstrap_distribution,
    plot_correlation_heatmap,
    plot_cross_guid_scatter,
    plot_data_quality_summary,
    plot_dtw_trajectories,
    plot_leave_one_out,
    plot_mutual_information_comparison,
    plot_per_guid_correlation_bar,
    plot_per_guid_correlation_histogram,
    plot_per_guid_trajectory_overlays,
    plot_per_dimension_kl_heatmap,
    plot_permutation_null,
    plot_pooled_scatter,
    plot_sample_dual_axis_trajectories,
    plot_scatter_matrix,
    plot_time_alignment_diagnostic,
    plot_trend_agreement,
)


def _try_plot(
    name: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke ``fn`` defensively so one bad plot doesn't abort the run."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Plot '{name}' failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def run_comparison(
    te_csv_path: str,
    output_dir: str,
    kld_csv_path: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Any] = None,
    device: Optional[str] = None,
    min_ite_valid_pc: float = 0.5,
    min_epochs_per_guid: int = 5,
    n_bootstrap: int = 10000,
    n_cluster_bootstrap: int = 5000,
    n_permutations: int = 10000,
    max_gap_seconds: float = 300.0,
    matching_mode: str = "fuzzy",
    band_seconds: float = 300.0,
    include_dtw: bool = True,
    include_per_dim: bool = True,
    epoch_grid_spacing: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the full TE vs KLD comparison pipeline.

    Args:
        te_csv_path: Path to the IDTxl empirical TE CSV.
        output_dir: Directory for all outputs (plots, CSVs, JSON).
        kld_csv_path: Path to pre-computed KLD metrics CSV. If provided,
            live inference is skipped.
        config_path: VAE config YAML path (for live inference).
        checkpoint_path: Model checkpoint (live inference only).
        data_path: HDF5 data path(s) (live inference only).
        device: Torch device string (live inference only).
        min_ite_valid_pc: Minimum valid-sample fraction for TE epochs.
        min_epochs_per_guid: Minimum matched epochs for per-GUID
            correlation.
        n_bootstrap: IID bootstrap iterations (simple correlation CI).
        n_cluster_bootstrap: Block bootstrap iterations (cluster-aware).
        n_permutations: Permutations for the primary permutation test.
        max_gap_seconds: Fuzzy-matching tolerance in seconds
            (default 300 s = ±5 min).
        matching_mode: ``"fuzzy"`` (default) or ``"exact_grid"``.
        band_seconds: Sakoe-Chiba half-width for DTW alignment.
        include_dtw: If True (default), run DTW alignment + plot.
        include_per_dim: If True (default), attempt per-dim KL
            analysis (auto-skips when columns absent).
        epoch_grid_spacing: Epoch-grid spacing (exact_grid mode only).
        seed: Random seed for permutation and bootstrap determinism.

    Returns:
        Dict with all statistical results.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"DTW backend: {dtw_backend_name()}")

    # ------------------------------------------------------------------
    # 1. Load empirical TE data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: Loading empirical TE data")
    logger.info("=" * 60)
    te_df = load_te_data(
        te_csv_path,
        min_ite_valid_pc=min_ite_valid_pc,
        grid_spacing=epoch_grid_spacing,
    )

    # ------------------------------------------------------------------
    # 2. Load or compute KLD data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Loading / computing KLD data")
    logger.info("=" * 60)
    if kld_csv_path is not None:
        kld_df = load_kld_from_metrics_csv(
            kld_csv_path, grid_spacing=epoch_grid_spacing
        )
    elif config_path is not None:
        te_guids = get_te_guids(te_csv_path)
        kld_df = load_kld_from_inference(
            config_path=config_path,
            te_guids=te_guids,
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            device=device,
            grid_spacing=epoch_grid_spacing,
        )
    else:
        raise ValueError(
            "Either kld_csv_path (pre-computed) or config_path "
            "(live inference) must be provided."
        )

    # ------------------------------------------------------------------
    # 3. Match
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(
        f"Step 3: Matching TE and KLD data (mode={matching_mode}, "
        f"max_gap={max_gap_seconds}s)"
    )
    logger.info("=" * 60)
    merged_df = merge_te_kld(
        te_df, kld_df,
        matching_mode=matching_mode,
        max_gap_seconds=max_gap_seconds,
    )
    if len(merged_df) == 0:
        logger.error("No matched epochs found. Aborting.")
        return {"error": "no_matches"}

    data_quality = compute_data_quality_report(
        te_df, kld_df, merged_df, max_gap_seconds=max_gap_seconds,
    )

    # ------------------------------------------------------------------
    # 4. Core correlations (existing + ported)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Correlations and statistical tests")
    logger.info("=" * 60)

    pooled_stats = compute_pooled_correlation(
        merged_df, n_bootstrap=n_bootstrap
    )
    per_guid_df = compute_per_guid_correlations(
        merged_df, min_epochs=min_epochs_per_guid
    )
    cross_guid_stats = compute_cross_guid_correlation(per_guid_df)
    pop_stats = population_level_test(per_guid_df)

    import numpy as np
    x = merged_df["ite_valid"].values.astype(float)
    y = merged_df["kld"].values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)

    permutation_primary = permutation_test_correlation(
        x[mask], y[mask],
        method="spearman", n_permutations=n_permutations, seed=seed,
    )
    concordance = concordance_analysis(merged_df)
    trend_agreement = trend_agreement_analysis(merged_df)
    loo = leave_one_guid_out_sensitivity(merged_df)
    cluster_boot = cluster_aware_bootstrap(
        merged_df, n_bootstrap=n_cluster_bootstrap, seed=seed,
    )
    mi_value = mutual_information_knn(x[mask], y[mask], k=3)

    corr_sp = correlation_matrix(merged_df, method="spearman")
    corr_ke = correlation_matrix(merged_df, method="kendall")
    correlation_matrices = {
        "spearman": corr_sp["correlation"],
        "spearman_p": corr_sp["p_value"],
        "kendall": corr_ke["correlation"],
        "kendall_p": corr_ke["p_value"],
    }

    per_dim = None
    if include_per_dim:
        per_dim = per_dimension_kl_analysis(
            merged_df, n_permutations=min(n_permutations, 5000), seed=seed,
        )
        if per_dim is not None and len(per_dim) == 0:
            per_dim = None

    # ------------------------------------------------------------------
    # 4c. DTW alignment (on full trajectories)
    # ------------------------------------------------------------------
    dtw_result: Optional[Dict[str, Any]] = None
    dtw_pooled: Optional[Dict[str, Any]] = None
    if include_dtw:
        logger.info("=" * 60)
        logger.info("Step 4c: DTW alignment")
        logger.info("=" * 60)
        common_guids = sorted(
            set(te_df["guid"].unique()) & set(kld_df["guid"].unique())
        )
        dtw_result = dtw_align_per_guid(
            te_df, kld_df, common_guids=common_guids,
            band_seconds=band_seconds,
        )
        aligned_df = paired_dataset_from_dtw(
            dtw_result, enforce_band=True, band_seconds=band_seconds,
        )
        if len(aligned_df) >= 3:
            aligned_df = aligned_df.rename(
                columns={"te_value": "ite_valid", "kld_value": "kld"}
            )
            dtw_pooled = compute_pooled_correlation(
                aligned_df, n_bootstrap=min(n_bootstrap, 5000)
            )

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Generating plots")
    logger.info("=" * 60)

    _try_plot("pooled_scatter", plot_pooled_scatter,
              merged_df, output_dir_path, pooled_stats)
    _try_plot("per_guid_correlation_histogram",
              plot_per_guid_correlation_histogram,
              per_guid_df, output_dir_path)
    _try_plot("sample_dual_axis_trajectories",
              plot_sample_dual_axis_trajectories,
              merged_df, output_dir_path)
    _try_plot("cross_guid_scatter", plot_cross_guid_scatter,
              per_guid_df, output_dir_path, cross_guid_stats)
    _try_plot("per_guid_correlation_bar", plot_per_guid_correlation_bar,
              per_guid_df, output_dir_path)
    _try_plot("bootstrap_ci", plot_bootstrap_ci,
              pooled_stats, output_dir_path)

    _try_plot("data_quality_summary", plot_data_quality_summary,
              data_quality, output_dir_path)
    _try_plot("time_alignment_diagnostic", plot_time_alignment_diagnostic,
              merged_df, output_dir_path, max_gap_seconds=max_gap_seconds)
    _try_plot(
        "correlation_heatmap_spearman", plot_correlation_heatmap,
        correlation_matrices["spearman"], correlation_matrices["spearman_p"],
        output_dir_path,
        title="Spearman ρ: KLD measures vs TE measures",
        filename="correlation_heatmap_spearman.pdf",
    )
    _try_plot(
        "correlation_heatmap_kendall", plot_correlation_heatmap,
        correlation_matrices["kendall"], correlation_matrices["kendall_p"],
        output_dir_path,
        title="Kendall τ: KLD measures vs TE measures",
        filename="correlation_heatmap_kendall.pdf",
    )
    kld_cols_avail = [c for c in CANDIDATE_KLD_COLS if c in merged_df.columns]
    te_cols_avail = [c for c in CANDIDATE_TE_COLS if c in merged_df.columns]
    _try_plot(
        "scatter_matrix", plot_scatter_matrix,
        merged_df, kld_cols_avail, te_cols_avail, output_dir_path,
    )
    _try_plot("trend_agreement", plot_trend_agreement,
              trend_agreement, output_dir_path)
    _try_plot("leave_one_out", plot_leave_one_out,
              loo, output_dir_path)
    _try_plot("bootstrap_distribution", plot_bootstrap_distribution,
              cluster_boot, output_dir_path)
    _try_plot("permutation_null", plot_permutation_null,
              permutation_primary, output_dir_path,
              title="Permutation null: ite_valid ↔ kld")
    _try_plot(
        "mutual_information_comparison",
        plot_mutual_information_comparison,
        mi_value, pooled_stats, output_dir_path,
    )
    _try_plot("per_guid_trajectory_overlays",
              plot_per_guid_trajectory_overlays,
              merged_df, output_dir_path)
    if per_dim is not None:
        _try_plot("per_dimension_kl_heatmap", plot_per_dimension_kl_heatmap,
                  per_dim, output_dir_path)
    if dtw_result is not None:
        _try_plot("dtw_trajectories", plot_dtw_trajectories,
                  dtw_result, output_dir_path)

    # ------------------------------------------------------------------
    # 6. Export
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 6: Exporting results")
    logger.info("=" * 60)

    export_summary(
        output_dir=output_dir_path,
        merged_df=merged_df,
        per_guid_df=per_guid_df,
        pooled_stats=pooled_stats,
        cross_guid_stats=cross_guid_stats,
        population_stats=pop_stats,
        data_quality=data_quality,
        permutation_primary=permutation_primary,
        concordance=concordance,
        trend_agreement=trend_agreement,
        leave_one_out=loo,
        cluster_bootstrap=cluster_boot,
        mutual_information={"pooled_ite_kld_nats": float(mi_value)},
        dtw=dtw_result,
        dtw_pooled_correlation=dtw_pooled,
        per_dimension=per_dim,
        correlation_matrices=correlation_matrices,
    )

    # ------------------------------------------------------------------
    # 7. Console summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Matched epochs:  {len(merged_df)}")
    logger.info(f"  Matched GUIDs:   {merged_df['guid'].nunique()}")
    logger.info(
        f"  GUIDs with >= {min_epochs_per_guid} epochs: {len(per_guid_df)}"
    )
    if "pearson_r" in pooled_stats:
        logger.info(
            f"  Pooled Pearson r:  {pooled_stats['pearson_r']:.4f} "
            f"(p = {pooled_stats['pearson_p']:.2e})"
        )
        logger.info(
            f"  Pooled Spearman ρ: {pooled_stats['spearman_rho']:.4f} "
            f"(p = {pooled_stats['spearman_p']:.2e})"
        )
    if permutation_primary and "p_value" in permutation_primary:
        logger.info(
            f"  Permutation p (Spearman): {permutation_primary['p_value']:.4f}"
        )
    if concordance and "kendall_tau" in concordance:
        logger.info(
            f"  Concordance: τ = {concordance['kendall_tau']:.4f}, "
            f"C-index = {concordance['concordance_index']:.3f}"
        )
    if dtw_pooled and "pearson_r" in dtw_pooled:
        logger.info(
            f"  DTW-aligned Pearson r: {dtw_pooled['pearson_r']:.4f} "
            f"(p = {dtw_pooled.get('pearson_p', float('nan')):.2e})"
        )
    logger.info(f"  MI (nats): {mi_value:.4f}")
    logger.info(f"  Results saved to: {output_dir_path}")

    return {
        "pooled": pooled_stats,
        "per_guid": per_guid_df,
        "cross_guid": cross_guid_stats,
        "population": pop_stats,
        "merged_df": merged_df,
        "data_quality": data_quality,
        "permutation_primary": permutation_primary,
        "concordance": concordance,
        "trend_agreement": trend_agreement,
        "leave_one_out": loo,
        "cluster_bootstrap": cluster_boot,
        "mutual_information": float(mi_value),
        "dtw": dtw_result,
        "dtw_pooled_correlation": dtw_pooled,
        "per_dimension": per_dim,
        "correlation_matrices": correlation_matrices,
    }


# ======================================================================
# __main__ — edit paths and knobs here, then run the file directly.
# ======================================================================

if __name__ == "__main__":
    # ── Paths ──
    # Default to the empirical CSV bundled with the old transformer
    # pipeline (9507 rows, 273 GUIDs). Override freely.
    TE_CSV_PATH = (
        "model/transformer/tr_testing/TE_analysis/te_record_epoch.csv"
    )
    OUTPUT_DIR = "model/vae_teb_prediction/testing/TE_Calculated/results"

    # ── KLD source: pre-computed CSV (set to None for live inference) ──
    KLD_CSV_PATH = None  # e.g., "path/to/metrics.csv"

    # ── Live inference settings (used when KLD_CSV_PATH is None) ──
    CONFIG_PATH = "model/vae_teb_prediction/config.yaml"
    CHECKPOINT_PATH = None  # None → read from config
    DATA_PATH = None  # None → read from config
    DEVICE = None  # None → auto-detect

    # ── Matching / DTW tolerances ──
    MAX_GAP_SECONDS = 300.0       # ±5 min fuzzy matching tolerance
    MATCHING_MODE = "fuzzy"       # "fuzzy" or "exact_grid"
    BAND_SECONDS = 300.0          # DTW Sakoe-Chiba half-width

    # ── Analysis parameters ──
    MIN_ITE_VALID_PC = 0.9
    MIN_EPOCHS_PER_GUID = 5
    N_BOOTSTRAP = 10000
    N_CLUSTER_BOOTSTRAP = 5000
    N_PERMUTATIONS = 10000
    EPOCH_GRID_SPACING = 1200     # only used in exact_grid mode
    INCLUDE_DTW = True
    INCLUDE_PER_DIM = True
    SEED = 42

    results = run_comparison(
        te_csv_path=TE_CSV_PATH,
        output_dir=OUTPUT_DIR,
        kld_csv_path=KLD_CSV_PATH,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        data_path=DATA_PATH,
        device=DEVICE,
        min_ite_valid_pc=MIN_ITE_VALID_PC,
        min_epochs_per_guid=MIN_EPOCHS_PER_GUID,
        n_bootstrap=N_BOOTSTRAP,
        n_cluster_bootstrap=N_CLUSTER_BOOTSTRAP,
        n_permutations=N_PERMUTATIONS,
        max_gap_seconds=MAX_GAP_SECONDS,
        matching_mode=MATCHING_MODE,
        band_seconds=BAND_SECONDS,
        include_dtw=INCLUDE_DTW,
        include_per_dim=INCLUDE_PER_DIM,
        epoch_grid_spacing=EPOCH_GRID_SPACING,
        seed=SEED,
    )
