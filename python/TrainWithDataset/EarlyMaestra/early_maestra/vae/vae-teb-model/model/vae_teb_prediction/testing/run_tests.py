"""
Simple script to run VAE-TEB testing pipeline.

Usage:
    from testing.run_tests import run_full_test_pipeline

    results = run_full_test_pipeline(
        checkpoint_path=None,
        data_path=None,
        output_dir=None,
        config_path="model/vae_teb_prediction/config.yaml",
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime

import torch
from loguru import logger
import yaml

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parents[4]  # Go up 4 levels to reach project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import testing components
from model.vae_teb_prediction.testing.base import TestRunner
from model.vae_teb_prediction.testing.analyses import (
    run_histogram_analysis,
    run_latent_distribution_analysis,
    run_temporal_accuracy_analysis,
    run_within_window_analysis,
    run_coherence_analysis,
    run_trajectory_analysis,
    run_reconstruction_analysis,
    run_single_prediction_windows,
    run_dataset_stats_analysis,
)
from model.vae_teb_prediction.testing.collectors import collect_predictions
from model.vae_teb_prediction.testing.visualizers import plot_reconstruction_sample
from model.vae_teb_prediction.testing.visualizers_interactive import (
    plot_reconstruction_interactive,
    plot_metrics_comparison_interactive,
)

# Import data loading utilities
from hdf5_dataset.hdf5_dataset import create_optimized_dataloader, build_guid_filtered_dataloader

# Import standard library
import json


def run_full_test_pipeline(
    checkpoint_path: Optional[str],
    data_path: Optional[Union[str, List[str]]],
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    skip_trajectory: bool = False,
    skip_coherence: bool = False,
    skip_interactive: bool = False,
    analysis_samples: int = 5,
    analysis_beta: float = 1.0,
    single_prediction_samples: int = 5,
    single_prediction_start_index: int = 20,
    single_prediction_step_size: Optional[int] = None,
    single_prediction_windows_per_sample: int = 4,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    config_path: Optional[Union[str, Path]] = None,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the complete VAE-TEB testing pipeline.

    Uses two different dataloaders:
    - Standard batch loader for histogram, latent, temporal, and coherence analyses
    - GUID-based loader for trajectory analysis (each batch = one patient)

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt or .pt). If None, uses
            model_config.core_model_checkpoint from config_path.
        data_path: Path(s) to test dataset(s). Can be a single HDF5 file path
            or a list of paths for multiple files. If None, config_path must be
            provided and include dataset_config.vae_test_datasets.
        output_dir: Directory for saving results. If None and config_path is provided,
            uses general_config.folders_config.out_dir_base/test_results.
        stats_path: Path to normalization statistics HDF5 file (optional).
        device: Device string ("cuda:0", "cpu"). Auto-detects if None.
        max_samples: Maximum samples to process for standard analyses. None for all.
        batch_size: Batch size for standard data loading. If None and config_path
            is provided, uses general_config.batch_size.test.
        skip_trajectory: Skip trajectory analysis (faster).
        skip_coherence: Skip coherence analysis (faster).
        skip_interactive: Skip Plotly interactive plots.
        analysis_samples: Number of samples for detailed reconstruction plots.
        analysis_beta: Beta value for loss annotation in reconstruction plots.
        single_prediction_samples: Number of samples for single-window plots.
        single_prediction_start_index: First prediction index for window plots.
        single_prediction_step_size: Optional step size between windows.
        single_prediction_windows_per_sample: Windows per sample for window plots.
        min_epochs_per_guid: Minimum epochs per patient for trajectory analysis.
        max_guids: Maximum patients for trajectory analysis (None for all).
        config_path: Optional path to a YAML config matching trainer.py.
        num_workers: DataLoader worker count. If None and config_path is provided,
            uses dataset_config.dataloader_config.num_workers.
        normalize_fields: Fields to normalize (None uses config or stats defaults).
        dataset_kwargs: Additional CombinedHDF5Dataset kwargs. Merged over config.
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
    # Resolve paths
    checkpoint_path, output_dir = _resolve_runner_settings(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    # Normalize data_path to list of strings
    data_paths: List[str] = []
    if isinstance(data_path, str):
        data_paths = [data_path]
    elif data_path is not None:
        data_paths = list(data_path)

    data_paths, stats_path, batch_size, num_workers, normalize_fields, dataset_kwargs = _resolve_dataloader_settings(
        data_paths=data_paths,
        stats_path=stats_path,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize_fields=normalize_fields,
        dataset_kwargs=dataset_kwargs,
        config_path=config_path,
    )

    if not data_paths:
        raise ValueError(
            "No test data provided. Pass data_path or supply config_path with "
            "dataset_config.vae_test_datasets."
        )

    # Auto-detect device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Data: {data_paths}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Stats: {stats_path}")
    if config_path is not None:
        logger.info(f"Config: {config_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")

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
    standard_loader = _create_dataloader(
        data_paths,
        batch_size,
        stats_path,
        normalize_fields=normalize_fields,
        num_workers=num_workers,
        dataset_kwargs=dataset_kwargs,
    )

    # GUID-based loader for trajectory analysis (each batch = one patient)
    guid_loader = None
    if not skip_trajectory:
        logger.info("Creating GUID-based dataloader for trajectory analysis...")
        _, guid_loader = _create_guid_dataloader(
            data_paths,
            stats_path=stats_path,
            min_epochs_per_guid=min_epochs_per_guid,
            max_guids=max_guids,
            normalize_fields=normalize_fields,
            num_workers=num_workers,
            dataset_kwargs=dataset_kwargs,
        )

    # ----- Step 3: Run analyses -----
    results: Dict[str, Any] = {}

    # Dataset statistics (does not require model, only dataloader)
    logger.info("Running dataset statistics analysis...")
    dataset_stats_dir = Path(output_dir) / "dataset_stats"
    results["dataset_stats"] = run_dataset_stats_analysis(standard_loader, dataset_stats_dir)

    # Histogram analysis (standard loader)
    logger.info("Running histogram analysis...")
    results["histogram"] = run_histogram_analysis(runner, standard_loader, max_samples)

    # Latent distribution analysis (standard loader)
    logger.info("Running latent distribution analysis...")
    results["latent"] = run_latent_distribution_analysis(runner, standard_loader, max_samples=min(500, max_samples or 500))

    # Temporal accuracy analysis (standard loader)
    logger.info("Running temporal accuracy analysis...")
    results["temporal"] = run_temporal_accuracy_analysis(runner, standard_loader, max_samples=min(200, max_samples or 200))

    # Within-window temporal accuracy (standard loader)
    logger.info("Running within-window accuracy analysis...")
    results["within_window"] = run_within_window_analysis(
        runner,
        standard_loader,
        max_samples=min(100, max_samples or 100),
    )

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

    # ----- Step 5: Detailed reconstruction + single-window plots -----
    if analysis_samples > 0:
        logger.info("Generating detailed reconstruction analysis plots...")
        results["analysis"] = run_reconstruction_analysis(
            runner,
            standard_loader,
            max_samples=min(analysis_samples, max_samples or analysis_samples),
            beta=analysis_beta,
        )

    if single_prediction_samples > 0:
        logger.info("Generating single prediction window plots...")
        results["single_prediction_windows"] = run_single_prediction_windows(
            runner,
            standard_loader,
            max_samples=min(single_prediction_samples, max_samples or single_prediction_samples),
            start_index=single_prediction_start_index,
            step_size=single_prediction_step_size,
            windows_per_sample=single_prediction_windows_per_sample,
        )

    # ----- Step 6: Interactive metrics comparison -----
    if not skip_interactive and "histogram" in results:
        logger.info("Generating interactive metrics comparison...")
        plot_metrics_comparison_interactive(
            results["histogram"],
            output_dir / "metrics_comparison.html",
        )

    # ----- Step 7: Save summary -----
    _save_summary(results, output_dir)

    logger.info(f"Testing complete! Results saved to {output_dir}")
    return results


def _create_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    batch_size: int,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    num_workers: int = 0,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
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
    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    if "pin_memory" not in resolved_kwargs:
        resolved_kwargs["pin_memory"] = True

    loader = create_optimized_dataloader(
        hdf5_files=[str(p) for p in paths],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        rank=0,
        world_size=1,
        **resolved_kwargs,
    )

    logger.info(f"Loaded {len(loader.dataset)} test samples")
    return loader


def _create_guid_dataloader(
    data_path: Union[str, Path, Sequence[Union[str, Path]]],
    stats_path: Optional[str] = None,
    min_epochs_per_guid: int = 3,
    max_guids: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    num_workers: Optional[int] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
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
    paths = list(data_path) if isinstance(data_path, (list, tuple)) else [data_path]
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)
    if "pin_memory" not in resolved_kwargs:
        resolved_kwargs["pin_memory"] = True

    loader_overrides = {}
    if num_workers is not None:
        loader_overrides["num_workers"] = num_workers
    eligible_guids, loader = build_guid_filtered_dataloader(
        dataset_paths=[str(p) for p in paths],
        min_samples=min_epochs_per_guid,
        max_guids=max_guids,
        sampler_shuffle=False,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        dataloader_overrides=loader_overrides if loader_overrides else None,
        **resolved_kwargs,
    )

    logger.info(f"GUID-based loader: {len(eligible_guids)} patients with >= {min_epochs_per_guid} epochs")
    return eligible_guids, loader


def _load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config or {}


def _resolve_runner_settings(
    *,
    checkpoint_path: Optional[str],
    output_dir: Optional[str],
    config_path: Optional[Union[str, Path]],
) -> tuple[Path, Path]:
    resolved_checkpoint = checkpoint_path
    resolved_output = output_dir

    if config_path is not None:
        config = _load_config(config_path)
        model_cfg = config.get("model_config", {}) or {}
        folders_cfg = config.get("general_config", {}).get("folders_config", {}) or {}

        if not resolved_checkpoint:
            resolved_checkpoint = model_cfg.get("core_model_checkpoint")

        if resolved_output is None:
            base_dir = folders_cfg.get("out_dir_base")
            if base_dir:
                # Create timestamped folder structure similar to training pipeline
                now = datetime.now()
                run_date = now.strftime("%Y-%m-%d--[%H-%M-%S]") + f"--{now.microsecond:06d}-"
                experiment_tag = config.get("general_config", {}).get("tag", "test")

                # Structure: out_dir_base / {tag} / {timestamp} / test_results
                tag_dir = Path(base_dir) / experiment_tag
                timestamped_dir = tag_dir / run_date
                resolved_output = str(timestamped_dir / "test_results")

    if not resolved_checkpoint:
        raise ValueError(
            "checkpoint_path is required unless config_path provides model_config.core_model_checkpoint."
        )

    if not resolved_output:
        resolved_output = "test_results"

    return Path(resolved_checkpoint), Path(resolved_output)


def _resolve_dataloader_settings(
    *,
    data_paths: List[str],
    stats_path: Optional[str],
    batch_size: Optional[int],
    num_workers: Optional[int],
    normalize_fields: Optional[Sequence[str]],
    dataset_kwargs: Optional[Dict[str, Any]],
    config_path: Optional[Union[str, Path]],
) -> tuple[List[str], Optional[str], int, int, Optional[Sequence[str]], Dict[str, Any]]:
    resolved_paths = list(data_paths) if data_paths else []
    resolved_stats = stats_path
    resolved_batch_size = batch_size
    resolved_workers = num_workers
    resolved_normalize_fields = normalize_fields
    resolved_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)

    if config_path is not None:
        config = _load_config(config_path)
        dataset_cfg = config.get("dataset_config", {}) or {}
        dataloader_cfg = dataset_cfg.get("dataloader_config", {}) or {}

        if not resolved_paths:
            resolved_paths = list(dataset_cfg.get("vae_test_datasets", []) or [])
        if resolved_stats is None:
            resolved_stats = dataset_cfg.get("stat_path")
        if resolved_batch_size is None:
            resolved_batch_size = (
                config.get("general_config", {})
                .get("batch_size", {})
                .get("test")
            )
        if resolved_workers is None:
            resolved_workers = dataloader_cfg.get("num_workers", 0)
        if resolved_normalize_fields is None:
            resolved_normalize_fields = dataloader_cfg.get("normalize_fields")

        config_dataset_kwargs = dataloader_cfg.get("dataset_kwargs", {}) or {}
        merged_kwargs = dict(config_dataset_kwargs)
        merged_kwargs.update(resolved_kwargs)
        resolved_kwargs = merged_kwargs

    if resolved_batch_size is None:
        resolved_batch_size = 32
    if resolved_workers is None:
        resolved_workers = 0

    return (
        resolved_paths,
        resolved_stats,
        resolved_batch_size,
        resolved_workers,
        resolved_normalize_fields,
        resolved_kwargs,
    )


def _save_summary(results: Dict[str, Any], output_dir: Path) -> None:
    """Save a text summary of results."""
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

    if "dataset_stats" in results and isinstance(results["dataset_stats"], dict):
        ds = results["dataset_stats"]
        summary["dataset_stats"] = {
            "n_samples": ds.get("n_samples"),
            "n_guids": ds.get("n_guids"),
            "epochs_per_guid": ds.get("epochs_per_guid"),
            "epoch_time": ds.get("epoch_time"),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


# ----- Convenience function for quick testing -----
def quick_test(
    checkpoint_path: Optional[str],
    data_path: Optional[str],
    output_dir: str = "quick_test_results",
    stats_path: Optional[str] = None,
    n_samples: int = 100,
    config_path: Optional[Union[str, Path]] = None,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Run a quick test with limited samples (for debugging).

    Args:
        checkpoint_path: Path to checkpoint (optional if config_path provides core_model_checkpoint).
        data_path: Path to test data (optional if config_path is provided).
        output_dir: Output directory.
        stats_path: Path to normalization statistics (optional).
        n_samples: Number of samples to process (default 100).
        config_path: Optional path to a YAML config matching trainer.py.
        num_workers: DataLoader worker count.
        normalize_fields: Fields to normalize.
        dataset_kwargs: Additional CombinedHDF5Dataset kwargs.
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
        analysis_samples=0,
        single_prediction_samples=0,
        config_path=config_path,
        num_workers=num_workers,
        normalize_fields=normalize_fields,
        dataset_kwargs=dataset_kwargs,
        **model_kwargs,
    )


# ----- Example usage -----
if __name__ == "__main__":
    # Example: Edit these paths for your setup
    CHECKPOINT = None  # Use config_path to pull core_model_checkpoint
    DATA = None  # Use config_path to pull dataset settings
    STATS = None  # Optional normalization stats if not using config_path
    CONFIG = "path/to/your/config.yaml"
    OUTPUT = None  # Use config_path to pull out_dir_base/test_results

    # Run full pipeline
    results = run_full_test_pipeline(
        checkpoint_path=CHECKPOINT,
        data_path=DATA,
        output_dir=OUTPUT,
        stats_path=STATS,  # Optional
        config_path=CONFIG,
        max_samples=None,  # Process all samples
    )

    # Print summary
    if "histogram" in results:
        df = results["histogram"]
        print(f"\n=== Test Results ===")
        print(f"Samples: {len(df)}")
        print(f"VAF: {df['vaf'].mean():.4f} +/- {df['vaf'].std():.4f}")
        print(f"MSE: {df['mse'].mean():.6f}")
        print(f"SNR: {df['snr'].mean():.2f} dB")
        print(f"KLD: {df['kld'].mean():.6f}")
