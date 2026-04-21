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
from typing import Any, Callable, Dict, Optional, Sequence

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
    # 4b. PCA vs per-dim surrogate comparison.
    #
    # Ranks every model-side TE surrogate present in ``merged_df``
    # (raw ``kld``, PCA scores ``kld_pc1/2/3``, L2 norm of top-3
    # components, posterior drift, attention concentration, TE-lag
    # mass, residual/uplift) by Pearson / Spearman / Kendall / MI
    # against the empirical ``ite_valid``. Writes
    # ``pca_vs_dims_summary.csv`` to the output directory.
    #
    # Requires the histogram CSV to carry the lag-attn v1 TE-surrogate
    # columns (written automatically by ``collect_metrics`` in v1).
    # Skips gracefully if they are absent.
    # ------------------------------------------------------------------
    pca_vs_dims_df: Optional["pd.DataFrame"] = None
    try:
        pca_vs_dims_df = run_pca_vs_dims_comparison(
            merged_df, output_dir_path, te_col="ite_valid",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"run_pca_vs_dims_comparison failed: {exc}")

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
    # 5b. Extended empirical-vs-model comparisons (v1).
    #
    # For each model-side score column present in ``merged_df``
    # (``kld`` + any PCA / TE-surrogate columns propagated from the
    # histogram metrics CSV), compute:
    #   - cross-correlation per GUID (lag tolerance)
    #   - Bland-Altman agreement (standardised scales)
    #   - ROC-AUC for "high empirical TE" event detection
    #   - per-GUID linear regression (slope + R²)
    #   - conditional KS across ite_valid quartiles
    #
    # Each helper writes a CSV; each plotter writes a PDF. Loops
    # over the candidate score columns so PCA surrogates are
    # exercised alongside raw ``kld``.
    # ------------------------------------------------------------------
    extra_comparisons: Dict[str, Dict[str, Any]] = {}
    try:
        extra_score_cols = ["kld"]
        if {"kld_pc1", "kld_pc2", "kld_pc3"}.issubset(merged_df.columns):
            extra_score_cols.append("kld_pca_l2_top3")   # synth by run_pca_vs_dims_comparison
            extra_score_cols.append("kld_pc1")
        for opt_col in (
            "posterior_drift_norm", "attention_concentration_mean",
            "te_lag_total_mass",
        ):
            if opt_col in merged_df.columns:
                extra_score_cols.append(opt_col)

        # ``kld_pca_l2_top3`` is synthesised by
        # ``run_pca_vs_dims_comparison`` above; mirror it here so the
        # extended comparisons can use it too.
        if (
            "kld_pca_l2_top3" not in merged_df.columns
            and {"kld_pc1", "kld_pc2", "kld_pc3"}.issubset(merged_df.columns)
        ):
            from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (  # noqa: E501
                pca_trajectory as _pca_traj,
            )
            merged_df["kld_pca_l2_top3"] = _pca_traj(merged_df, "l2_top3")

        from model.vae_teb_prediction.testing.TE_Calculated.te_kld_visualizations import (  # noqa: E501
            plot_bland_altman,
            plot_conditional_ks_grid,
            plot_per_guid_slope_hist,
            plot_roc_curve,
            plot_xcorr_lag_hist,
        )

        for score_col in extra_score_cols:
            if score_col not in merged_df.columns:
                continue
            sub_dir = output_dir_path / f"extended_vs_{score_col}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            try:
                xcorr_df = cross_correlation_per_guid(
                    merged_df, score_col=score_col, te_col="ite_valid",
                )
                xcorr_df.to_csv(sub_dir / "xcorr_per_guid.csv", index=False)
                _try_plot(f"xcorr_{score_col}", plot_xcorr_lag_hist,
                          xcorr_df, sub_dir / "xcorr_lag_hist.pdf")

                bland = bland_altman(merged_df, score_col, "ite_valid")
                _try_plot(f"bland_altman_{score_col}", plot_bland_altman,
                          merged_df, bland, sub_dir / "bland_altman.pdf",
                          score_col=score_col, te_col="ite_valid")

                roc = roc_auc_high_te(merged_df, score_col, "ite_valid")
                _try_plot(f"roc_{score_col}", plot_roc_curve,
                          merged_df, roc, sub_dir / "roc_high_te.pdf",
                          score_col=score_col, te_col="ite_valid")

                reg = per_guid_regression(merged_df, score_col, "ite_valid")
                reg.to_csv(sub_dir / "per_guid_regression.csv", index=False)
                _try_plot(f"slope_{score_col}", plot_per_guid_slope_hist,
                          reg, sub_dir / "per_guid_slope_hist.pdf")

                ks_df = conditional_ks_by_quartile(
                    merged_df, score_col, "ite_valid",
                )
                ks_df.to_csv(sub_dir / "conditional_ks.csv", index=False)
                _try_plot(f"ks_{score_col}", plot_conditional_ks_grid,
                          ks_df, sub_dir / "conditional_ks_grid.pdf",
                          score_col=score_col, te_col="ite_valid")

                extra_comparisons[score_col] = {
                    "bland_altman": bland,
                    "roc_auc_high_te": roc,
                    "n_guids_xcorr": int(len(xcorr_df)),
                    "n_guids_regression": int(len(reg)),
                }
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"extended comparisons failed for {score_col!r}: {exc}"
                )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"extended comparisons block failed: {exc}")

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

    # PCA-vs-dims surrogate ranking (when available).
    if pca_vs_dims_df is not None and not pca_vs_dims_df.empty:
        top = pca_vs_dims_df.iloc[0]
        logger.info(
            f"  Top TE surrogate by Spearman ρ: "
            f"{top['surrogate']!r} (rho={top.get('spearman_rho', float('nan')):.4f}, "
            f"pearson_r={top.get('pearson_r', float('nan')):.4f}, "
            f"MI={top.get('mutual_information', float('nan')):.4f})"
        )
        if "kld_pc1" in pca_vs_dims_df["surrogate"].values:
            row = pca_vs_dims_df[pca_vs_dims_df["surrogate"] == "kld_pc1"].iloc[0]
            logger.info(
                f"  KLD PC1 alone:       rho={row['spearman_rho']:.4f}, "
                f"pearson_r={row['pearson_r']:.4f}"
            )
        if "kld_pca_l2_top3" in pca_vs_dims_df["surrogate"].values:
            row = pca_vs_dims_df[
                pca_vs_dims_df["surrogate"] == "kld_pca_l2_top3"
            ].iloc[0]
            logger.info(
                f"  KLD PCA L2(top-3):   rho={row['spearman_rho']:.4f}, "
                f"pearson_r={row['pearson_r']:.4f}"
            )
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
        "pca_vs_dims_summary": pca_vs_dims_df,
        "extended_comparisons": extra_comparisons,
    }


# ======================================================================
# Lag-attn v1: extra empirical-vs-model comparison helpers
# ======================================================================

import numpy as np
import pandas as pd
from scipy import stats as _sp_stats

from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (
    mutual_information_knn,
    pca_trajectory,
)


def _finite_pair(x: Any, y: Any) -> "tuple[np.ndarray, np.ndarray]":
    """Return aligned, finite numpy arrays from two array-likes."""
    arr_x = np.asarray(x, dtype=float).ravel()
    arr_y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    return arr_x[mask], arr_y[mask]


def cross_correlation_per_guid(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    max_lag: int = 10,
    min_n: int = 5,
) -> pd.DataFrame:
    """Per-GUID cross-correlation of a model score vs empirical TE.

    For each GUID with at least ``min_n`` matched epochs, computes the
    normalised cross-correlation of the score and TE traces over lags
    ``[-max_lag, +max_lag]`` and reports the lag of the maximum
    absolute correlation.

    Args:
        merged: Output of ``merge_te_kld``, sorted by ``epoch``.
        score_col: Model-side score column (e.g. ``kld``, ``kld_pc1``).
        te_col: Empirical-TE column (e.g. ``ite_valid``).
        max_lag: Maximum absolute lag to evaluate (epochs, integer).
        min_n: Minimum matched epochs required per GUID.

    Returns:
        DataFrame with columns ``guid, n_epochs, best_lag, best_xcorr``.
    """
    rows = []
    for guid, sub in merged.groupby("guid"):
        sub = sub.sort_values("epoch").reset_index(drop=True)
        x, y = _finite_pair(sub[score_col], sub[te_col])
        n = x.size
        if n < min_n:
            continue
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
        best_lag = 0
        best_val = 0.0
        max_l = min(max_lag, n - 1)
        for lag in range(-max_l, max_l + 1):
            if lag < 0:
                xc = x[-lag:]
                yc = y[: n + lag]
            elif lag > 0:
                xc = x[: n - lag]
                yc = y[lag:]
            else:
                xc, yc = x, y
            if xc.size < 2:
                continue
            corr = float(np.dot(xc, yc) / xc.size)
            if abs(corr) > abs(best_val):
                best_val = corr
                best_lag = lag
        rows.append({
            "guid": str(guid),
            "n_epochs": int(n),
            "best_lag": int(best_lag),
            "best_xcorr": best_val,
        })
    return pd.DataFrame(rows)


def bland_altman(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    standardize: bool = True,
) -> Dict[str, float]:
    """Bland-Altman agreement summary between two scores.

    When ``standardize=True`` (default) both columns are z-scored before
    differencing — the empirical TE and the model surrogate live in
    different units, so the agreement of interest is on standardised
    scales.

    Args:
        merged: Merged DataFrame.
        score_col: Model-side score column.
        te_col: Empirical-TE column.
        standardize: Whether to z-score both series before computing
            the differences.

    Returns:
        Dict with ``n``, ``mean_diff``, ``std_diff``, ``loa_low``,
        ``loa_high`` (95% limits of agreement = mean_diff ± 1.96*std_diff).
    """
    x, y = _finite_pair(merged[score_col], merged[te_col])
    if x.size < 3:
        return {"n": int(x.size), "mean_diff": float("nan"),
                "std_diff": float("nan"), "loa_low": float("nan"),
                "loa_high": float("nan")}
    if standardize:
        x = (x - x.mean()) / (x.std() + 1e-12)
        y = (y - y.mean()) / (y.std() + 1e-12)
    diff = x - y
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    return {
        "n": int(diff.size),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_low": mean_diff - 1.96 * std_diff,
        "loa_high": mean_diff + 1.96 * std_diff,
    }


def roc_auc_high_te(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    quantile: float = 0.9,
) -> Dict[str, Any]:
    """ROC analysis: detect "high empirical TE" events using a model score.

    Binarises ``te_col`` at the given quantile (default 90th percentile)
    and treats ``score_col`` as the discriminator. Computes the
    Mann-Whitney U statistic and converts to AUC.

    Args:
        merged: Merged DataFrame.
        score_col: Model-side score column (higher = positive).
        te_col: Empirical-TE column.
        quantile: Quantile threshold for the positive class.

    Returns:
        Dict with ``n_pos``, ``n_neg``, ``auc``, ``mw_p``, ``threshold``.
    """
    x, y = _finite_pair(merged[score_col], merged[te_col])
    if x.size < 8:
        return {"n_pos": 0, "n_neg": 0, "auc": float("nan"),
                "mw_p": float("nan"), "threshold": float("nan")}
    thr = float(np.quantile(y, quantile))
    pos = x[y >= thr]
    neg = x[y < thr]
    if pos.size == 0 or neg.size == 0:
        return {"n_pos": int(pos.size), "n_neg": int(neg.size),
                "auc": float("nan"), "mw_p": float("nan"),
                "threshold": thr}
    u, p = _sp_stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(u) / float(pos.size * neg.size)
    return {
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "auc": auc,
        "mw_p": float(p),
        "threshold": thr,
    }


def per_guid_regression(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    min_n: int = 5,
) -> pd.DataFrame:
    """Per-GUID linear regression ``score = slope * te + intercept``.

    Args:
        merged: Merged DataFrame.
        score_col: Model-side dependent variable.
        te_col: Empirical-TE independent variable.
        min_n: Minimum matched epochs required per GUID.

    Returns:
        DataFrame with ``guid, n, slope, intercept, r2`` columns.
    """
    rows = []
    for guid, sub in merged.groupby("guid"):
        x, y = _finite_pair(sub[te_col], sub[score_col])
        if x.size < min_n or np.std(x) == 0:
            continue
        res = _sp_stats.linregress(x, y)
        slope_val = float(res[0])     # slope
        intercept_val = float(res[1]) # intercept
        r_val = float(res[2])         # r-value
        rows.append({
            "guid": str(guid),
            "n": int(x.size),
            "slope": slope_val,
            "intercept": intercept_val,
            "r2": r_val * r_val,
        })
    return pd.DataFrame(rows)


def conditional_ks_by_quartile(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
) -> pd.DataFrame:
    """Two-sample KS of the model score across empirical-TE quartiles.

    Splits ``te_col`` into 4 quartiles and runs a KS test of the
    score distribution in the highest vs lowest quartile and each
    intermediate quartile vs the lowest, providing a non-parametric
    check that the model score *shifts* with the empirical TE.

    Args:
        merged: Merged DataFrame.
        score_col: Model-side score column.
        te_col: Empirical-TE column.

    Returns:
        DataFrame ``[quartile, n_q, n_ref, ks_stat, p_value]``.
    """
    x, y = _finite_pair(merged[score_col], merged[te_col])
    if x.size < 16:
        return pd.DataFrame(columns=["quartile", "n_q", "n_ref", "ks_stat", "p_value"])
    qs = np.quantile(y, [0.25, 0.5, 0.75])
    bins = np.digitize(y, qs)  # 0..3
    ref = x[bins == 0]
    rows = []
    for q in (1, 2, 3):
        sub = x[bins == q]
        if sub.size < 4 or ref.size < 4:
            rows.append({"quartile": int(q), "n_q": int(sub.size),
                         "n_ref": int(ref.size), "ks_stat": float("nan"),
                         "p_value": float("nan")})
            continue
        try:
            ks = _sp_stats.ks_2samp(sub, ref, alternative="two-sided")
            stat = float(getattr(ks, "statistic", ks[0]))
            p_val = float(getattr(ks, "pvalue", ks[1]))
        except Exception:  # noqa: BLE001
            stat = float("nan")
            p_val = float("nan")
        rows.append({
            "quartile": int(q),
            "n_q": int(sub.size),
            "n_ref": int(ref.size),
            "ks_stat": stat,
            "p_value": p_val,
        })
    return pd.DataFrame(rows)


def per_guid_r2(
    merged: pd.DataFrame,
    score_col: str = "kld",
    te_col: str = "ite_valid",
    min_n: int = 5,
) -> pd.DataFrame:
    """Per-GUID R² between a model score and empirical TE.

    Convenience wrapper around :func:`per_guid_regression` that returns
    only the GUID and its R² (handy for histogramming).

    Args:
        merged: Merged DataFrame.
        score_col: Model-side score column.
        te_col: Empirical-TE column.
        min_n: Minimum matched epochs.

    Returns:
        DataFrame ``[guid, n, r2]``.
    """
    df = per_guid_regression(merged, score_col, te_col, min_n=min_n)
    if df.empty:
        return df
    return df[["guid", "n", "r2"]].copy()


def run_pca_vs_dims_comparison(
    merged: pd.DataFrame,
    output_dir: Path,
    *,
    te_col: str = "ite_valid",
    candidate_scores: Optional[Sequence[str]] = None,
    n_bootstrap: int = 0,
) -> pd.DataFrame:
    """Score every available model surrogate against empirical TE.

    Loops over the model-side TE surrogates that are present in
    ``merged`` (raw ``kld``, PCA scores, posterior drift, attention
    concentration, TE-lag mass, etc.), and reports Pearson, Spearman,
    Kendall, and KSG-MI for each. Results are saved to a CSV.

    Args:
        merged: Output of :func:`merge_te_kld`. Must contain ``te_col``.
        output_dir: Where to write ``pca_vs_dims_summary.csv``.
        te_col: Empirical-TE column (default ``"ite_valid"``).
        candidate_scores: Iterable of model-side columns to compare.
            When ``None``, defaults to a sensible v1 list (and falls
            back gracefully when columns are missing).
        n_bootstrap: When > 0, runs an IID bootstrap on Pearson r.

    Returns:
        DataFrame with one row per surrogate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if candidate_scores is None:
        candidate_scores = (
            "kld",
            "kld_pc1",
            "kld_pc2",
            "kld_pc3",
            "kld_pca_l2_top3",
            "posterior_drift_norm",
            "attention_concentration_mean",
            "te_lag_total_mass",
            "delta_src_norm",
            "uplift_abs",
            "residual_ratio",
        )

    df_local = merged.copy()
    if (
        "kld_pca_l2_top3" not in df_local.columns
        and {"kld_pc1", "kld_pc2", "kld_pc3"}.issubset(df_local.columns)
    ):
        df_local["kld_pca_l2_top3"] = pca_trajectory(df_local, "l2_top3")

    rows = []
    for col in candidate_scores:
        if col not in df_local.columns or te_col not in df_local.columns:
            continue
        x, y = _finite_pair(df_local[col], df_local[te_col])
        if x.size < 8:
            rows.append({"surrogate": col, "n": int(x.size)})
            continue
        pear_res = _sp_stats.pearsonr(x, y)
        spear_res = _sp_stats.spearmanr(x, y)
        ken_res = _sp_stats.kendalltau(x, y)
        try:
            mi = float(mutual_information_knn(x, y, k=3))
        except Exception:  # noqa: BLE001
            mi = float("nan")
        row = {
            "surrogate": col,
            "n": int(x.size),
            "pearson_r": float(pear_res[0]),
            "pearson_p": float(pear_res[1]),
            "spearman_rho": float(spear_res[0]),
            "spearman_p": float(spear_res[1]),
            "kendall_tau": float(ken_res[0]),
            "kendall_p": float(ken_res[1]),
            "mutual_information": mi,
        }
        if n_bootstrap > 0:
            rng = np.random.default_rng(42)
            samples = np.empty(n_bootstrap)
            for i in range(n_bootstrap):
                idx = rng.integers(0, x.size, size=x.size)
                xs, ys = x[idx], y[idx]
                if np.std(xs) == 0 or np.std(ys) == 0:
                    samples[i] = np.nan
                else:
                    samples[i] = float(np.corrcoef(xs, ys)[0, 1])
            row["pearson_r_ci_lo"] = float(np.nanpercentile(samples, 2.5))
            row["pearson_r_ci_hi"] = float(np.nanpercentile(samples, 97.5))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="spearman_rho", ascending=False, na_position="last")
        df.to_csv(output_dir / "pca_vs_dims_summary.csv", index=False)
        logger.info(
            f"run_pca_vs_dims_comparison: ranked {len(df)} surrogates against "
            f"{te_col}; top: {df.iloc[0]['surrogate'] if len(df) else 'n/a'}"
        )
    return df


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
