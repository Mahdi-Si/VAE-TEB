"""VAE-TEB Lag-Attentive v1 testing pipeline.

A modular testing framework tailored to :class:`SeqVaeLagAttnV1`. It
provides:

- ``TestRunner`` — minimal harness for model loading, device management,
  batch iteration, and forward dispatch.
- Pure metric functions over the v1 forward dict (feature forecast,
  uplift, residual usage, attention diagnostics, TE lag maps, closed-form
  KL).
- Data collectors that surface those metrics in DataFrame / numpy /
  list-of-dict formats for downstream analyses.
- Matplotlib visualizers (feature-forecast heatmaps, attention maps,
  uplift histograms, etc.) and Plotly interactive alternatives.
- Composable analysis modules under ``analyses/``.

Example:
    >>> from testing import TestRunner, run_all_analyses
    >>> runner = TestRunner.from_checkpoint(
    ...     "best.ckpt", "results/", config_path="config_lag_attn_v1.yaml"
    ... )
    >>> results = run_all_analyses(runner, test_loader)
"""

from __future__ import annotations

# Core components
from model.vae_teb_prediction.testing.base import TestRunner

# Metric functions
from model.vae_teb_prediction.testing.metrics import (
    aggregate_te_lag_map,
    compute_attention_diagnostics,
    compute_forecast_metrics,
    compute_kld,
    compute_kld_per_sample,
    compute_kld_per_timestep,
    compute_reconstruction_metrics,
    compute_residual_usage,
    compute_uplift_metrics,
)

# Data collectors
from model.vae_teb_prediction.testing.collectors import (
    collect_attention_maps,
    collect_forecast_errors_per_horizon,
    collect_kld_trajectory,
    collect_latents,
    collect_metrics,
    collect_predictions,
    collect_te_lag_maps,
)

# Static visualizers (Matplotlib)
from model.vae_teb_prediction.testing.visualizers import (
    plot_attention_mass_by_lag,
    plot_feature_forecast_heatmap,
    plot_forecast_error_by_horizon,
    plot_guid_absolute_trajectory,
    plot_kld_guid_trajectory,
    plot_kld_trajectory,
    plot_kld_trajectory_3d,
    plot_lag_attention_heatmap,
    plot_latent_distributions,
    plot_latent_trajectory_3d,
    plot_metric_histograms,
    plot_residual_usage_trace,
    plot_te_lag_distribution,
    plot_uplift_histogram,
)

# Interactive visualizers (Plotly)
from model.vae_teb_prediction.testing.visualizers_interactive import (
    plot_kld_trajectory_3d_interactive,
    plot_kld_trajectory_interactive,
    plot_latent_interpolation_interactive,
    plot_latent_space_3d,
    plot_latent_trajectory_3d_interactive,
    plot_metrics_comparison_interactive,
)

# Analysis modules
from model.vae_teb_prediction.testing.analyses import (
    TrajectoryAnalyzer,
    run_all_analyses,
    run_anchor_position_analysis,
    run_attention_diagnostics,
    run_class_separation_analysis,
    run_dataset_stats_analysis,
    run_encoder_probe,
    run_forecast_quality_analysis,
    run_histogram_analysis,
    run_horizon_error_profile,
    run_kld_lag_diagnostics,
    run_latent_distribution_analysis,
    run_latent_interpolation,
    run_latent_space_visualization,
    run_residual_usage_analysis,
    run_sample_diagnostics,
    run_te_lag_class_analysis,
    run_trajectory_analysis,
    run_uplift_analysis,
)

# Single sample plotting
from model.vae_teb_prediction.testing.plot_single_samples import (
    plot_sample_lag_attention,
    plot_sample_lag_attn_diagnostic,
    plot_sample_signals_kld,
)


__all__ = [
    # Core
    "TestRunner",
    # Metrics
    "compute_reconstruction_metrics",
    "compute_kld",
    "compute_kld_per_sample",
    "compute_kld_per_timestep",
    "compute_forecast_metrics",
    "compute_uplift_metrics",
    "compute_residual_usage",
    "compute_attention_diagnostics",
    "aggregate_te_lag_map",
    # Collectors
    "collect_metrics",
    "collect_latents",
    "collect_predictions",
    "collect_kld_trajectory",
    "collect_attention_maps",
    "collect_te_lag_maps",
    "collect_forecast_errors_per_horizon",
    # Static visualizers
    "plot_metric_histograms",
    "plot_latent_distributions",
    "plot_kld_trajectory",
    "plot_kld_guid_trajectory",
    "plot_kld_trajectory_3d",
    "plot_guid_absolute_trajectory",
    "plot_latent_trajectory_3d",
    "plot_feature_forecast_heatmap",
    "plot_forecast_error_by_horizon",
    "plot_uplift_histogram",
    "plot_residual_usage_trace",
    "plot_lag_attention_heatmap",
    "plot_te_lag_distribution",
    "plot_attention_mass_by_lag",
    # Interactive visualizers
    "plot_kld_trajectory_interactive",
    "plot_kld_trajectory_3d_interactive",
    "plot_latent_space_3d",
    "plot_latent_interpolation_interactive",
    "plot_latent_trajectory_3d_interactive",
    "plot_metrics_comparison_interactive",
    # Analyses
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
    "run_latent_distribution_analysis",
    "run_latent_space_visualization",
    "run_latent_interpolation",
    "run_trajectory_analysis",
    "TrajectoryAnalyzer",
    "run_sample_diagnostics",
    "run_dataset_stats_analysis",
    "run_class_separation_analysis",
    "run_all_analyses",
    # Single sample plotting
    "plot_sample_lag_attn_diagnostic",
    "plot_sample_signals_kld",
    "plot_sample_lag_attention",
]
