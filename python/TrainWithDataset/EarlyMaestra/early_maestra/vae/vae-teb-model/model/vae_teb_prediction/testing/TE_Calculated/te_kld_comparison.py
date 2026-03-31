"""Main orchestrator for empirical TE vs. VAE-KLD comparison analysis.

Loads empirical Transfer Entropy (from IDTxl CSV) and VAE KL divergence
values, merges them on (guid, epoch), and runs a full statistical
comparison pipeline with publication-quality plots.

Supports two modes:
    1. **Pre-computed**: load KLD from a metrics CSV produced by a prior
       test run (fast, no GPU needed).
    2. **Live inference**: run model inference on-the-fly for the TE GUIDs
       (requires checkpoint + HDF5 data + GPU).

All parameters are set in the ``if __name__ == "__main__"`` block below.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Ensure project root is importable
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model.vae_teb_prediction.testing.TE_Calculated.te_data_loader import (
    load_te_data,
    get_te_guids,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (
    load_kld_from_metrics_csv,
    load_kld_from_inference,
    merge_te_kld,
    compute_pooled_correlation,
    compute_per_guid_correlations,
    compute_cross_guid_correlation,
    population_level_test,
    export_summary,
)
from model.vae_teb_prediction.testing.TE_Calculated.te_kld_visualizations import (
    plot_pooled_scatter,
    plot_per_guid_correlation_histogram,
    plot_sample_dual_axis_trajectories,
    plot_cross_guid_scatter,
    plot_per_guid_correlation_bar,
    plot_bootstrap_ci,
)


def run_comparison(
    te_csv_path: str,
    output_dir: str,
    kld_csv_path: str | None = None,
    config_path: str | None = None,
    checkpoint_path: str | None = None,
    data_path: str | list[str] | None = None,
    device: str | None = None,
    min_ite_valid_pc: float = 0.5,
    min_epochs_per_guid: int = 5,
    n_bootstrap: int = 10000,
    epoch_grid_spacing: int = 1200,
) -> dict:
    """Run the full TE vs. KLD comparison pipeline.

    Args:
        te_csv_path: Path to the IDTxl empirical TE CSV.
        output_dir: Directory for all outputs (plots, CSVs, JSON).
        kld_csv_path: Path to pre-computed KLD metrics CSV.  If provided,
            live inference is skipped.
        config_path: VAE config YAML path (for live inference).
        checkpoint_path: Model checkpoint path (for live inference; if
            *None*, resolved from config).
        data_path: HDF5 data path(s) (for live inference; if *None*,
            resolved from config).
        device: Torch device string (for live inference).
        min_ite_valid_pc: Minimum valid-sample fraction for TE epochs.
        min_epochs_per_guid: Minimum matched epochs for per-GUID
            correlation.
        n_bootstrap: Number of bootstrap iterations.
        epoch_grid_spacing: Epoch grid spacing in seconds.

    Returns:
        Dict with all statistical results.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

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
    # 3. Merge
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Merging TE and KLD data")
    logger.info("=" * 60)
    merged_df = merge_te_kld(te_df, kld_df)

    if len(merged_df) == 0:
        logger.error("No matched epochs found. Aborting.")
        return {"error": "no_matches"}

    # ------------------------------------------------------------------
    # 4. Compute statistics
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Computing correlations and statistical tests")
    logger.info("=" * 60)

    pooled_stats = compute_pooled_correlation(merged_df, n_bootstrap=n_bootstrap)
    per_guid_df = compute_per_guid_correlations(
        merged_df, min_epochs=min_epochs_per_guid
    )
    cross_guid_stats = compute_cross_guid_correlation(per_guid_df)
    pop_stats = population_level_test(per_guid_df)

    # ------------------------------------------------------------------
    # 5. Generate plots
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Generating plots")
    logger.info("=" * 60)

    plot_pooled_scatter(merged_df, output_dir_path, pooled_stats)
    plot_per_guid_correlation_histogram(per_guid_df, output_dir_path)
    plot_sample_dual_axis_trajectories(merged_df, output_dir_path)
    plot_cross_guid_scatter(per_guid_df, output_dir_path, cross_guid_stats)
    plot_per_guid_correlation_bar(per_guid_df, output_dir_path)
    plot_bootstrap_ci(pooled_stats, output_dir_path)

    # ------------------------------------------------------------------
    # 6. Export results
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
    )

    # ------------------------------------------------------------------
    # 7. Print console summary
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
    if "pearson_r" in cross_guid_stats:
        logger.info(
            f"  Cross-GUID Pearson r:  {cross_guid_stats['pearson_r']:.4f}"
        )
        logger.info(
            f"  Cross-GUID Spearman ρ: {cross_guid_stats['spearman_rho']:.4f}"
        )
    if "pearson" in pop_stats and "fisher_z_p" in pop_stats["pearson"]:
        logger.info(
            f"  Population test (Pearson): Fisher z p = "
            f"{pop_stats['pearson']['fisher_z_p']:.2e}"
        )
    logger.info(f"  Results saved to: {output_dir_path}")

    return {
        "pooled": pooled_stats,
        "per_guid": per_guid_df,
        "cross_guid": cross_guid_stats,
        "population": pop_stats,
        "merged_df": merged_df,
    }


# ======================================================================
# __main__ — all parameters defined here
# ======================================================================

if __name__ == "__main__":
    # ── Paths ──
    TE_CSV_PATH = "model/vae_teb_prediction/testing/TE_Calculated/te_record_epoch_HIE_NoCS.csv"
    OUTPUT_DIR = "model/vae_teb_prediction/testing/TE_Calculated/results"

    # ── KLD source: pre-computed CSV (set to None for live inference) ──
    KLD_CSV_PATH = None  # e.g., "path/to/metrics.csv"

    # ── Live inference settings (used when KLD_CSV_PATH is None) ──
    CONFIG_PATH = "model/vae_teb_prediction/config.yaml"
    CHECKPOINT_PATH = None  # None → read from config
    DATA_PATH = None  # None → read from config
    DEVICE = None  # None → auto-detect

    # ── Analysis parameters ──
    MIN_ITE_VALID_PC = 0.9  # Min valid sample % for TE epochs
    MIN_EPOCHS_PER_GUID = 5  # Min matched epochs for per-GUID correlation
    N_BOOTSTRAP = 10000  # Bootstrap iterations
    EPOCH_GRID_SPACING = 1200  # Seconds between epoch boundaries

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
        epoch_grid_spacing=EPOCH_GRID_SPACING,
    )
