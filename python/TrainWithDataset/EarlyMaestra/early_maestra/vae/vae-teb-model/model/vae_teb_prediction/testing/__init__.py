"""
VAE-TEB Testing Pipeline.

A modular, reusable testing framework for VAE-TEB models that provides:
- Standardized test running with TestRunner dataclass
- Pure metric computation functions (VAF, MSE, SNR, KLD)
- Flexible data collection patterns
- Publication-quality Matplotlib plots
- Interactive Plotly visualizations
- Composable analysis modules

Example:
    >>> from testing import TestRunner, run_all_analyses
    >>> runner = TestRunner.from_checkpoint("model.pt", "results/")
    >>> results = run_all_analyses(runner, test_loader)

For individual analyses:
    >>> from testing.analyses import run_histogram_analysis, run_trajectory_analysis
    >>> df = run_histogram_analysis(runner, test_loader)
"""

from __future__ import annotations

# Core components
from model.vae_teb_prediction.testing.base import TestRunner

# Metric functions
from model.vae_teb_prediction.testing.metrics import (
    aggregate_predictions,
    compute_kld,
    compute_kld_per_sample,
    compute_kld_per_timestep,
    compute_reconstruction_metrics,
)

# Data collectors
from model.vae_teb_prediction.testing.collectors import (
    collect_kld_trajectory,
    collect_latents,
    collect_metrics,
    collect_predictions,
)

# Static visualizers (Matplotlib)
from model.vae_teb_prediction.testing.visualizers import (
    plot_coherence_analysis,
    plot_coherence_signals,
    plot_coherence_spectrum,
    plot_cross_correlation,
    plot_guid_absolute_trajectory,
    plot_kld_guid_trajectory,
    plot_kld_trajectory_3d,
    plot_kld_trajectory,
    plot_latent_distributions,
    plot_latent_trajectory_3d,
    plot_metric_histograms,
    plot_psd_comparison,
    plot_reconstruction_coherence,
    plot_reconstruction_sample,
    plot_temporal_accuracy,
    plot_within_window_accuracy,
    plot_time_frequency_coherence,
)

# Interactive visualizers (Plotly)
from model.vae_teb_prediction.testing.visualizers_interactive import (
    plot_kld_trajectory_interactive,
    plot_kld_trajectory_3d_interactive,
    plot_latent_interpolation_interactive,
    plot_latent_space_3d,
    plot_latent_trajectory_3d_interactive,
    plot_metrics_comparison_interactive,
    plot_reconstruction_interactive,
)

# Analysis modules
from model.vae_teb_prediction.testing.analyses import (
    TrajectoryAnalyzer,
    run_all_analyses,
    run_coherence_analysis,
    run_histogram_analysis,
    run_latent_distribution_analysis,
    run_latent_interpolation,
    run_latent_space_visualization,
    run_temporal_accuracy_analysis,
    run_trajectory_analysis,
    run_within_window_analysis,
    run_reconstruction_analysis,
    run_single_prediction_windows,
)


__all__ = [
    # Core
    "TestRunner",
    # Metrics
    "compute_reconstruction_metrics",
    "compute_kld",
    "compute_kld_per_sample",
    "compute_kld_per_timestep",
    "aggregate_predictions",
    # Collectors
    "collect_metrics",
    "collect_latents",
    "collect_predictions",
    "collect_kld_trajectory",
    # Static visualizers
    "plot_metric_histograms",
    "plot_latent_distributions",
    "plot_reconstruction_sample",
    "plot_temporal_accuracy",
    "plot_kld_trajectory",
    "plot_kld_guid_trajectory",
    "plot_kld_trajectory_3d",
    "plot_guid_absolute_trajectory",
    "plot_coherence_analysis",
    "plot_coherence_signals",
    "plot_coherence_spectrum",
    "plot_reconstruction_coherence",
    "plot_psd_comparison",
    "plot_cross_correlation",
    "plot_time_frequency_coherence",
    "plot_within_window_accuracy",
    # Interactive visualizers
    "plot_reconstruction_interactive",
    "plot_kld_trajectory_interactive",
    "plot_kld_trajectory_3d_interactive",
    "plot_latent_space_3d",
    "plot_latent_interpolation_interactive",
    "plot_latent_trajectory_3d_interactive",
    "plot_metrics_comparison_interactive",
    # Analyses
    "run_histogram_analysis",
    "run_latent_distribution_analysis",
    "run_latent_space_visualization",
    "run_latent_interpolation",
    "run_temporal_accuracy_analysis",
    "run_within_window_analysis",
    "run_reconstruction_analysis",
    "run_single_prediction_windows",
    "run_coherence_analysis",
    "run_trajectory_analysis",
    "TrajectoryAnalyzer",
    "run_all_analyses",
]
