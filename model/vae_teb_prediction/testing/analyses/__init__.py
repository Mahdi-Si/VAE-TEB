"""Analysis modules for the VAE-TEB lag-attn v1 testing pipeline.

This package composes the testing building blocks (collectors, metrics,
visualizers) into end-to-end workflows. Each ``run_*`` function takes
``(runner, loader, ...)``, writes CSVs and plots to a subdirectory under
``runner.output_dir``, and returns a summary dict.

Available analyses:

- ``run_dataset_stats_analysis`` — dataset coverage statistics
- ``run_histogram_analysis`` — per-sample metric histograms
- ``run_forecast_quality_analysis`` — feature forecast quality
- ``run_horizon_error_profile`` — MSE vs horizon step
- ``run_anchor_position_analysis`` — MSE vs anchor index
- ``run_uplift_analysis`` — baseline vs full uplift
- ``run_residual_usage_analysis`` — residual-branch activity / collapse
- ``run_attention_diagnostics`` — lag attention diagnostics
- ``run_te_lag_class_analysis`` — lag-resolved TE by class
- ``run_encoder_probe`` — classifier-probe on encoder features
- ``run_latent_distribution_analysis`` — per-dim latent histograms
- ``run_latent_space_visualization`` — 3D PCA latent plot
- ``run_latent_interpolation`` — stub (unsupported under v1)
- ``run_trajectory_analysis`` — per-patient latent trajectory / KLD
- ``run_class_separation_analysis`` — latent class separability
- ``run_sample_diagnostics`` — per-sample multi-row diagnostic PDFs

Example:
    >>> results = run_all_analyses(runner, standard_loader, trajectory_loader=guid_loader)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner

# Dataset-level analysis (model-agnostic).
from model.vae_teb_prediction.testing.analyses.dataset_stats import (
    collect_dataset_stats,
    compute_stats_summary,
    run_dataset_stats_analysis,
)

# Histogram and forecast-quality analyses.
from model.vae_teb_prediction.testing.analyses.histogram import run_histogram_analysis
from model.vae_teb_prediction.testing.analyses.forecast_quality import (
    run_forecast_quality_analysis,
)
from model.vae_teb_prediction.testing.analyses.temporal import (
    run_anchor_position_analysis,
    run_horizon_error_profile,
)
from model.vae_teb_prediction.testing.analyses.uplift import run_uplift_analysis
from model.vae_teb_prediction.testing.analyses.residual_usage import (
    run_residual_usage_analysis,
)
from model.vae_teb_prediction.testing.analyses.attention_diagnostics import (
    run_attention_diagnostics,
)
from model.vae_teb_prediction.testing.analyses.te_lag_analysis import (
    run_te_lag_class_analysis,
)
from model.vae_teb_prediction.testing.analyses.encoder_probe import run_encoder_probe
from model.vae_teb_prediction.testing.analyses.kld_lag_diagnostics import (
    run_kld_lag_diagnostics,
)
from model.vae_teb_prediction.testing.analyses.kld_pca import run_kld_pca_analysis
from model.vae_teb_prediction.testing.analyses.per_class_breakdown import (
    run_per_class_breakdown,
)

# Latent / trajectory / class-separation analyses.
from model.vae_teb_prediction.testing.analyses.latent import (
    run_latent_distribution_analysis,
    run_latent_interpolation,
    run_latent_space_visualization,
)
from model.vae_teb_prediction.testing.analyses.trajectory import (
    TrajectoryAnalyzer,
    run_trajectory_analysis,
)
from model.vae_teb_prediction.testing.analyses.class_separation import (
    compute_center_loss_effectiveness,
    compute_class_cohesion_separation,
    compute_cluster_quality_metrics,
    compute_linear_separability,
    compute_temporal_separation,
    load_discriminative_centers,
    run_class_separation_analysis,
)
from model.vae_teb_prediction.testing.analyses.compare_trajectory_classes import (
    compare_dimensions_by_class,
    compare_features_by_class,
    compute_frechet_distance,
    compute_mmd_rbf,
    run_comparison as run_trajectory_comparison,
)
from model.vae_teb_prediction.testing.analyses.qualitative import run_sample_diagnostics

# Optional changepoint dependency.
try:
    from model.vae_teb_prediction.testing.analyses.changepoint import (
        create_changepoint_detector,
        detect_changepoints,
        summarize_latent_segments,
        summarize_trajectory,
    )
    HAS_CHANGEPOINT = True
except ImportError:
    HAS_CHANGEPOINT = False
    create_changepoint_detector = None  # type: ignore[assignment]
    detect_changepoints = None  # type: ignore[assignment]
    summarize_trajectory = None  # type: ignore[assignment]
    summarize_latent_segments = None  # type: ignore[assignment]


def _safe(name: str, fn, *args, **kwargs) -> Any:
    """Run ``fn`` and convert any exception into an ``{"error": ...}`` dict."""
    try:
        logger.info("=" * 50)
        logger.info(f"Running {name} ...")
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{name} failed: {exc}")
        return {"error": str(exc)}


def run_all_analyses(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    *,
    skip_trajectory: bool = False,
    skip_attention: bool = False,
    skip_forecast_heatmaps: bool = False,
    skip_kld_pca: bool = False,
    skip_per_class_breakdown: bool = False,
    trajectory_loader: Optional[Any] = None,
    trajectory_dim_reduction: str = "pca",
    trajectory_n_changepoints: int = 5,
    trajectory_plot_3d: bool = True,
    trajectory_plot_animations: bool = False,
) -> Dict[str, Any]:
    """Run every lag-attn v1 analysis with sensible defaults.

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: Standard PyTorch DataLoader.
        max_samples: Maximum samples for per-sample analyses.
        skip_trajectory: Skip the per-GUID trajectory analysis.
        skip_attention: Skip the attention diagnostics and TE lag class analysis.
        skip_forecast_heatmaps: Skip the per-sample diagnostic PDFs.
        trajectory_loader: Optional GUID-based DataLoader (preferred for
            trajectory analysis).
        trajectory_dim_reduction: DR method passed to trajectory analysis.
        trajectory_n_changepoints: Changepoint count per trajectory.
        trajectory_plot_3d: Include 3D trajectory plots.
        trajectory_plot_animations: Include animated trajectory GIFs.

    Returns:
        Dict mapping analysis names to result dicts (or ``{"error": ...}``
        on failure).
    """
    results: Dict[str, Any] = {}

    results["histogram"] = _safe(
        "histogram_analysis", run_histogram_analysis,
        runner, loader, max_samples=max_samples or 1000,
    )
    results["forecast_quality"] = _safe(
        "forecast_quality", run_forecast_quality_analysis,
        runner, loader, max_samples=max_samples or 500,
    )
    results["horizon_error"] = _safe(
        "horizon_error_profile", run_horizon_error_profile,
        runner, loader, max_samples=min(200, max_samples or 200),
    )
    results["anchor_error"] = _safe(
        "anchor_position_analysis", run_anchor_position_analysis,
        runner, loader, max_samples=min(200, max_samples or 200),
    )
    results["uplift"] = _safe(
        "uplift_analysis", run_uplift_analysis,
        runner, loader, max_samples=max_samples or 500,
    )
    results["residual_usage"] = _safe(
        "residual_usage", run_residual_usage_analysis,
        runner, loader, max_samples=max_samples or 500,
    )

    if not skip_attention:
        results["attention"] = _safe(
            "attention_diagnostics", run_attention_diagnostics,
            runner, loader, max_samples=min(200, max_samples or 200),
        )
        results["te_lag"] = _safe(
            "te_lag_class_analysis", run_te_lag_class_analysis,
            runner, loader, max_samples=max_samples or 1000,
        )

    results["encoder_probe"] = _safe(
        "encoder_probe", run_encoder_probe,
        runner, loader, max_samples=min(2000, max_samples or 2000),
    )
    results["latent_distribution"] = _safe(
        "latent_distribution", run_latent_distribution_analysis,
        runner, loader, max_samples=max_samples or 500,
    )
    results["latent_space"] = _safe(
        "latent_space", run_latent_space_visualization,
        runner, loader, max_samples=max_samples or 500,
    )

    if not skip_trajectory:
        traj_loader = trajectory_loader if trajectory_loader is not None else loader
        results["trajectory"] = _safe(
            "trajectory_analysis", run_trajectory_analysis,
            runner, traj_loader,
            time_range_hours=12.0,
            min_epochs_per_guid=3,
            skip_dashboards=True,
            dim_reduction_method=trajectory_dim_reduction,
            n_changepoints=trajectory_n_changepoints,
            plot_3d=trajectory_plot_3d,
            plot_animations=trajectory_plot_animations,
        )

    if not skip_forecast_heatmaps:
        results["sample_diagnostics"] = _safe(
            "sample_diagnostics", run_sample_diagnostics,
            runner, loader, max_samples=min(10, max_samples or 10),
        )
        results["kld_lag_diagnostics"] = _safe(
            "kld_lag_diagnostics", run_kld_lag_diagnostics,
            runner, loader, max_samples=min(10, max_samples or 10),
        )

    if not skip_kld_pca:
        results["kld_pca"] = _safe(
            "kld_pca_analysis", run_kld_pca_analysis,
            runner, loader, max_samples=max_samples or 500,
        )

    # Per-class breakdown is a pure post-processor over the CSVs the
    # earlier analyses just wrote, so it must run *last*.
    if not skip_per_class_breakdown:
        results["per_class_breakdown"] = _safe(
            "per_class_breakdown", run_per_class_breakdown,
            runner.output_dir,
        )

    logger.info("=" * 50)
    logger.info("All analyses complete!")
    logger.info(f"Results saved to: {runner.output_dir}")
    return results


__all__ = [
    # Base
    "TestRunner",
    # Individual analyses
    "run_histogram_analysis",
    "run_forecast_quality_analysis",
    "run_horizon_error_profile",
    "run_anchor_position_analysis",
    "run_uplift_analysis",
    "run_residual_usage_analysis",
    "run_attention_diagnostics",
    "run_te_lag_class_analysis",
    "run_encoder_probe",
    "run_kld_lag_diagnostics",
    "run_kld_pca_analysis",
    "run_per_class_breakdown",
    "run_latent_distribution_analysis",
    "run_latent_space_visualization",
    "run_latent_interpolation",
    "run_trajectory_analysis",
    "TrajectoryAnalyzer",
    "run_sample_diagnostics",
    "run_dataset_stats_analysis",
    "collect_dataset_stats",
    "compute_stats_summary",
    # Changepoint analysis
    "create_changepoint_detector",
    "detect_changepoints",
    "summarize_trajectory",
    "summarize_latent_segments",
    "HAS_CHANGEPOINT",
    # Cross-class comparison
    "run_trajectory_comparison",
    "compute_frechet_distance",
    "compute_mmd_rbf",
    "compare_features_by_class",
    "compare_dimensions_by_class",
    # Class separation
    "run_class_separation_analysis",
    "compute_cluster_quality_metrics",
    "compute_class_cohesion_separation",
    "compute_linear_separability",
    "compute_temporal_separation",
    "load_discriminative_centers",
    "compute_center_loss_effectiveness",
    # Combined
    "run_all_analyses",
]
