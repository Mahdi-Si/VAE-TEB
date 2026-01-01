"""
Simple script to run VAE-TEB testing pipeline.

Usage:
    from testing.run_tests import run_full_test_pipeline

    results = run_full_test_pipeline(
        checkpoint_path="path/to/model.ckpt",
        data_path="path/to/test_data.h5",
        output_dir="results/",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from loguru import logger


def run_full_test_pipeline(
    checkpoint_path: str,
    data_path: Union[str, List[str]],
    output_dir: str = "test_results",
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    skip_trajectory: bool = False,
    skip_coherence: bool = False,
    skip_interactive: bool = False,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the complete VAE-TEB testing pipeline.

    Uses two different dataloaders:
    - Standard batch loader for histogram, latent, temporal, and coherence analyses
    - GUID-based loader for trajectory analysis (each batch = one patient)

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt or .pt).
        data_path: Path(s) to test dataset(s). Can be a single HDF5 file path
            or a list of paths for multiple files.
        output_dir: Directory for saving results.
        stats_path: Path to normalization statistics HDF5 file (optional).
        device: Device string ("cuda:0", "cpu"). Auto-detects if None.
        max_samples: Maximum samples to process for standard analyses. None for all.
        batch_size: Batch size for standard data loading.
        skip_trajectory: Skip trajectory analysis (faster).
        skip_coherence: Skip coherence analysis (faster).
        skip_interactive: Skip Plotly interactive plots.
        min_epochs_per_guid: Minimum epochs per patient for trajectory analysis.
        max_guids: Maximum patients for trajectory analysis (None for all).
        **model_kwargs: Additional model architecture parameters
            (e.g., latent_dim=16, hidden_dim=64).

    Returns:
        Dict with all analysis results and output paths.

    Example:
        >>> results = run_full_test_pipeline(
        ...     checkpoint_path="checkpoints/best_model.ckpt",
        ...     data_path="data/test.h5",
        ...     output_dir="results/experiment_1",
        ...     stats_path="data/stats.h5",
        ...     max_samples=500,
        ...     min_epochs_per_guid=5,  # Only patients with 5+ epochs
        ... )
        >>> print(f"Mean VAF: {results['histogram']['vaf'].mean():.4f}")
    """
    # Import here to avoid circular imports
    from .base import TestRunner
    from .analyses import (
        run_histogram_analysis,
        run_latent_distribution_analysis,
        run_temporal_accuracy_analysis,
        run_coherence_analysis,
        run_trajectory_analysis,
    )
    from .collectors import collect_predictions
    from .visualizers import plot_reconstruction_sample
    from .visualizers_interactive import (
        plot_reconstruction_interactive,
        plot_metrics_comparison_interactive,
    )

    # Resolve paths
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    # Normalize data_path to list of strings for CombinedHDF5Dataset
    if isinstance(data_path, str):
        data_paths = [data_path]
    else:
        data_paths = list(data_path)

    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    # ----- Step 1: Create TestRunner -----
    logger.info("Loading model from checkpoint...")
    runner = TestRunner.from_checkpoint(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=device,
        **model_kwargs,
    )

    # ----- Step 2: Create DataLoaders -----
    # Standard loader for most analyses
    logger.info("Creating standard test dataloader...")
    standard_loader = _create_dataloader(data_paths, batch_size, stats_path)

    # GUID-based loader for trajectory analysis (each batch = one patient)
    guid_loader = None
    if not skip_trajectory:
        logger.info("Creating GUID-based dataloader for trajectory analysis...")
        _, guid_loader = _create_guid_dataloader(
            data_paths,
            stats_path=stats_path,
            min_epochs_per_guid=min_epochs_per_guid,
            max_guids=max_guids,
        )

    # ----- Step 3: Run analyses -----
    results: Dict[str, Any] = {}

    # Histogram analysis (standard loader)
    logger.info("Running histogram analysis...")
    results["histogram"] = run_histogram_analysis(runner, standard_loader, max_samples)

    # Latent distribution analysis (standard loader)
    logger.info("Running latent distribution analysis...")
    results["latent"] = run_latent_distribution_analysis(runner, standard_loader, max_samples=min(500, max_samples or 500))

    # Temporal accuracy analysis (standard loader)
    logger.info("Running temporal accuracy analysis...")
    results["temporal"] = run_temporal_accuracy_analysis(runner, standard_loader, max_samples=min(200, max_samples or 200))

    # Coherence analysis (standard loader)
    if not skip_coherence:
        logger.info("Running coherence analysis...")
        results["coherence"] = run_coherence_analysis(runner, standard_loader, max_samples=min(50, max_samples or 50))

    # Trajectory analysis (GUID-based loader - each batch is one patient)
    if not skip_trajectory and guid_loader is not None:
        logger.info("Running trajectory analysis with GUID-based batching...")
        results["trajectory"] = run_trajectory_analysis(
            runner,
            guid_loader,
            time_range_hours=12.0,
            min_epochs_per_guid=min_epochs_per_guid,
        )

    # ----- Step 4: Generate sample reconstruction plots -----
    logger.info("Generating sample reconstructions...")
    samples = collect_predictions(runner, standard_loader, max_samples=10)

    samples_dir = runner.ensure_dir("samples")
    for i, sample in enumerate(samples[:5]):
        # Static plot
        plot_reconstruction_sample(sample, samples_dir / f"sample_{i}.png")

        # Interactive plot
        if not skip_interactive:
            plot_reconstruction_interactive(sample, samples_dir / f"sample_{i}.html")

    # ----- Step 5: Interactive metrics comparison -----
    if not skip_interactive and "histogram" in results:
        logger.info("Generating interactive metrics comparison...")
        plot_metrics_comparison_interactive(
            results["histogram"],
            output_dir / "metrics_comparison.html",
        )

    # ----- Step 6: Save summary -----
    _save_summary(results, output_dir)

    logger.info(f"Testing complete! Results saved to {output_dir}")
    return results


def _create_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    batch_size: int = 32,
    stats_path: Optional[str] = None,
) -> Any:
    """
    Create a standard DataLoader for the test dataset.

    Args:
        data_path: Path or list of paths to HDF5 test data file(s).
        batch_size: Batch size for loading.
        stats_path: Optional path to normalization statistics HDF5 file.

    Returns:
        DataLoader for the test dataset.
    """
    from torch.utils.data import DataLoader
    from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset

    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    dataset = CombinedHDF5Dataset(
        paths=[str(p) for p in paths],
        stats_path=stats_path,
        cache_size=500,
        pin_memory=torch.cuda.is_available(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"Loaded {len(dataset)} test samples")
    return loader


def _create_guid_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    stats_path: Optional[str] = None,
    min_epochs_per_guid: int = 3,
    max_guids: Optional[int] = None,
) -> Any:
    """
    Create a GUID-based DataLoader where each batch contains all samples from one GUID.

    This is essential for trajectory analysis where we need to see the full
    temporal evolution of each patient.

    Args:
        data_path: Path or list of paths to HDF5 test data file(s).
        stats_path: Optional path to normalization statistics HDF5 file.
        min_epochs_per_guid: Minimum epochs required per GUID to be included.
        max_guids: Maximum number of GUIDs to include (None for all).

    Returns:
        Tuple of (eligible_guids, DataLoader).
    """
    from hdf5_dataset.hdf5_dataset import build_guid_filtered_dataloader

    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    eligible_guids, loader = build_guid_filtered_dataloader(
        dataset_paths=[str(p) for p in paths],
        min_samples=min_epochs_per_guid,
        max_guids=max_guids,
        sampler_shuffle=False,
        stats_path=stats_path,
        cache_size=500,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(f"GUID-based loader: {len(eligible_guids)} patients with >= {min_epochs_per_guid} epochs")
    return eligible_guids, loader


def _save_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """Save a text summary of results."""
    import json

    summary_path = output_dir / "test_summary.json"

    # Extract serializable metrics
    summary = {}

    if "histogram" in results:
        df = results["histogram"]
        summary["metrics"] = {
            "n_samples": len(df),
            "vaf_mean": float(df["vaf"].mean()),
            "vaf_std": float(df["vaf"].std()),
            "mse_mean": float(df["mse"].mean()),
            "snr_mean": float(df["snr"].mean()),
            "kld_mean": float(df["kld"].mean()),
        }

    if "trajectory" in results and isinstance(results["trajectory"], dict):
        summary["trajectory"] = {
            k: v for k, v in results["trajectory"].items()
            if isinstance(v, (int, float, str))
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


# ----- Convenience function for quick testing -----
def quick_test(
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "quick_test_results",
    stats_path: Optional[str] = None,
    n_samples: int = 100,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Run a quick test with limited samples (for debugging).

    Args:
        checkpoint_path: Path to checkpoint.
        data_path: Path to test data.
        output_dir: Output directory.
        stats_path: Path to normalization statistics (optional).
        n_samples: Number of samples to process (default 100).
        **model_kwargs: Model architecture parameters.

    Returns:
        Results dict.
    """
    return run_full_test_pipeline(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        output_dir=output_dir,
        stats_path=stats_path,
        max_samples=n_samples,
        skip_trajectory=True,
        skip_coherence=True,
        skip_interactive=True,
        **model_kwargs,
    )


# ----- Example usage -----
if __name__ == "__main__":
    # Example: Edit these paths for your setup
    CHECKPOINT = "path/to/your/model.ckpt"
    DATA = ["path/to/your/test_data.h5"]  # Can be a single path or list of paths
    STATS = "path/to/your/stats.h5"  # Optional normalization stats
    OUTPUT = "test_results"

    # Run full pipeline
    results = run_full_test_pipeline(
        checkpoint_path=CHECKPOINT,
        data_path=DATA,
        output_dir=OUTPUT,
        stats_path=STATS,  # Optional
        max_samples=None,  # Process all samples
    )

    # Print summary
    if "histogram" in results:
        df = results["histogram"]
        print(f"\n=== Test Results ===")
        print(f"Samples: {len(df)}")
        print(f"VAF: {df['vaf'].mean():.4f} ± {df['vaf'].std():.4f}")
        print(f"MSE: {df['mse'].mean():.6f}")
        print(f"SNR: {df['snr'].mean():.2f} dB")
        print(f"KLD: {df['kld'].mean():.6f}")
