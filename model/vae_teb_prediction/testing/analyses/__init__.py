"""
Analysis modules for VAE-TEB testing.

This package provides complete analysis pipelines that compose the basic
building blocks (collectors, metrics, visualizers) into useful workflows.

Available analyses:
    - run_histogram_analysis: VAF, MSE, SNR, KLD histograms
    - run_latent_distribution_analysis: Per-dimension latent distributions
    - run_latent_space_visualization: 3D PCA latent space plot
    - run_latent_interpolation: Interpolate between latent codes
    - run_temporal_accuracy_analysis: Accuracy vs timestep position
    - run_coherence_analysis: FHR reconstruction coherence (optional UP coupling)
    - run_trajectory_analysis: KLD evolution over time before birth
    - run_reconstruction_analysis: Detailed per-sample diagnostics plots
    - run_single_prediction_windows: Non-averaged prediction window plots
    - run_all_analyses: Run all analyses with sensible defaults

IMPORTANT: Trajectory analysis works best with a GUID-based DataLoader where
each batch contains all epochs from a single patient. Use
`build_guid_filtered_dataloader` from hdf5_dataset.hdf5_dataset for this.

Example with separate loaders:
    >>> from hdf5_dataset.hdf5_dataset import build_guid_filtered_dataloader
    >>> guids, guid_loader = build_guid_filtered_dataloader(["test.h5"], min_samples=3)
    >>> results = run_all_analyses(runner, standard_loader, trajectory_loader=guid_loader)

Example with single loader (trajectory uses same loader):
    >>> results = run_all_analyses(runner, test_loader)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from model.vae_teb_prediction.testing.base import TestRunner

# Import individual analyses
from model.vae_teb_prediction.testing.analyses.histogram import run_histogram_analysis
from model.vae_teb_prediction.testing.analyses.latent import (
    run_latent_distribution_analysis,
    run_latent_interpolation,
    run_latent_space_visualization,
)
from model.vae_teb_prediction.testing.analyses.temporal import run_temporal_accuracy_analysis, run_within_window_analysis
from model.vae_teb_prediction.testing.analyses.coherence import run_coherence_analysis
from model.vae_teb_prediction.testing.analyses.trajectory import run_trajectory_analysis, TrajectoryAnalyzer
from model.vae_teb_prediction.testing.analyses.qualitative import (
    run_reconstruction_analysis,
    run_single_prediction_windows,
)


def run_all_analyses(
    runner: TestRunner,
    loader: Any,
    max_samples: Optional[int] = None,
    skip_trajectory: bool = False,
    skip_coherence: bool = False,
    trajectory_loader: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run all available analyses with sensible defaults.

    Each analysis runs independently with error handling, so a failure in
    one analysis doesn't prevent others from completing.

    Args:
        runner: TestRunner with model and device configured.
        loader: PyTorch DataLoader for test data (standard batching).
        max_samples: Maximum samples for per-sample analyses. None for all.
        skip_trajectory: If True, skip trajectory analysis (can be slow).
        skip_coherence: If True, skip coherence analysis.
        trajectory_loader: Optional GUID-based DataLoader for trajectory analysis.
            If None, uses the standard loader. For best results, pass a loader
            created with build_guid_filtered_dataloader where each batch contains
            all epochs from a single patient.

    Returns:
        Dict mapping analysis names to their results or error messages.

    Example with separate loaders (recommended for trajectory):
        >>> from hdf5_dataset.hdf5_dataset import build_guid_filtered_dataloader
        >>> guids, guid_loader = build_guid_filtered_dataloader(["test.h5"], min_samples=3)
        >>> results = run_all_analyses(runner, standard_loader, trajectory_loader=guid_loader)

    Example with single loader:
        >>> results = run_all_analyses(runner, test_loader, max_samples=500)
    """
    results: Dict[str, Any] = {}

    # Histogram analysis
    try:
        logger.info("=" * 50)
        logger.info("Running histogram analysis...")
        results["histogram"] = run_histogram_analysis(
            runner, loader, max_samples=max_samples or 1000
        )
    except Exception as e:
        logger.error(f"Histogram analysis failed: {e}")
        results["histogram"] = {"error": str(e)}

    # Latent distribution
    try:
        logger.info("=" * 50)
        logger.info("Running latent distribution analysis...")
        results["latent_distribution"] = run_latent_distribution_analysis(
            runner, loader, max_samples=max_samples or 500
        )
    except Exception as e:
        logger.error(f"Latent distribution analysis failed: {e}")
        results["latent_distribution"] = {"error": str(e)}

    # Latent space 3D
    try:
        logger.info("=" * 50)
        logger.info("Running latent space visualization...")
        results["latent_space"] = run_latent_space_visualization(
            runner, loader, max_samples=max_samples or 500
        )
    except Exception as e:
        logger.error(f"Latent space visualization failed: {e}")
        results["latent_space"] = {"error": str(e)}

    # Temporal accuracy
    try:
        logger.info("=" * 50)
        logger.info("Running temporal accuracy analysis...")
        results["temporal"] = run_temporal_accuracy_analysis(
            runner, loader, max_samples=max_samples or 200
        )
    except Exception as e:
        logger.error(f"Temporal accuracy analysis failed: {e}")
        results["temporal"] = {"error": str(e)}

    # Coherence analysis (optional)
    if not skip_coherence:
        try:
            logger.info("=" * 50)
            logger.info("Running coherence analysis...")
            results["coherence"] = run_coherence_analysis(
                runner, loader, max_samples=min(max_samples or 50, 50)
            )
        except Exception as e:
            logger.error(f"Coherence analysis failed: {e}")
            results["coherence"] = {"error": str(e)}

    # Trajectory analysis (optional - uses GUID-based loader if provided)
    if not skip_trajectory:
        try:
            logger.info("=" * 50)
            # Use GUID-based loader if provided, otherwise fall back to standard
            traj_loader = trajectory_loader if trajectory_loader is not None else loader
            if trajectory_loader is not None:
                logger.info("Running trajectory analysis with GUID-based batching...")
            else:
                logger.info("Running trajectory analysis (standard batching)...")
            results["trajectory"] = run_trajectory_analysis(
                runner, traj_loader,
                time_range_hours=12.0,
                min_epochs_per_guid=3,
                skip_dashboards=True,  # Skip for speed
            )
        except Exception as e:
            logger.error(f"Trajectory analysis failed: {e}")
            results["trajectory"] = {"error": str(e)}

    logger.info("=" * 50)
    logger.info("All analyses complete!")
    logger.info(f"Results saved to: {runner.output_dir}")

    return results


# Public API
__all__ = [
    # Base
    "TestRunner",
    # Individual analyses
    "run_histogram_analysis",
    "run_latent_distribution_analysis",
    "run_latent_space_visualization",
    "run_latent_interpolation",
    "run_temporal_accuracy_analysis",
    "run_within_window_analysis",
    "run_coherence_analysis",
    "run_trajectory_analysis",
    "TrajectoryAnalyzer",
    "run_reconstruction_analysis",
    "run_single_prediction_windows",
    # Combined
    "run_all_analyses",
]
