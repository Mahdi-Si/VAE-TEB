"""Main entry point for the transformer testing and analysis pipeline.

Provides ``run_full_test_pipeline()`` and ``quick_test()`` functions that
create DataLoaders, load the model, and run all analyses.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from .base import TransformerTestRunner


def _create_class_loaders(
    class_data_paths: Dict[str, List[str]],
    batch_size: int = 64,
    num_workers: int = 4,
    stats_path: Optional[str] = None,
    shuffle: bool = False,
) -> Dict[str, Any]:
    """Create standard DataLoaders, one per class.

    Args:
        class_data_paths: Dict mapping class names to lists of HDF5 paths.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        stats_path: Path to normalization statistics file.
        shuffle: Whether to shuffle.

    Returns:
        Dict mapping class names to DataLoaders.
    """
    from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

    loaders = {}
    for class_name, paths in class_data_paths.items():
        logger.info(
            f"Creating DataLoader for {class_name} "
            f"({len(paths)} HDF5 files)"
        )
        loaders[class_name] = create_optimized_dataloader(
            hdf5_files=paths,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            stats_path=stats_path,
            normalize_fields=["fhr_st", "up_st"],
            load_fields=["fhr_st", "up_st", "fhr", "up"],
            trim_minutes=1.0,
        )
    return loaders


def _create_guid_loaders(
    class_data_paths: Dict[str, List[str]],
    stats_path: Optional[str] = None,
    min_epochs_per_guid: int = 5,
    max_guids: Optional[int] = None,
) -> Dict[str, Any]:
    """Create GUID-based DataLoaders, one per class.

    Each batch contains all epochs for one patient.

    Args:
        class_data_paths: Dict mapping class names to lists of HDF5 paths.
        stats_path: Path to normalization statistics file.
        min_epochs_per_guid: Minimum epochs for a GUID to be included.
        max_guids: Maximum number of GUIDs per class.

    Returns:
        Dict mapping class names to GUID-based DataLoaders.
    """
    from hdf5_dataset.hdf5_dataset import build_guid_filtered_dataloader

    loaders = {}
    for class_name, paths in class_data_paths.items():
        logger.info(
            f"Creating GUID DataLoader for {class_name}..."
        )
        try:
            eligible_guids, loader = build_guid_filtered_dataloader(
                dataset_paths=paths,
                min_samples=min_epochs_per_guid,
                max_guids=max_guids,
                stats_path=stats_path,
                normalize_fields=["fhr_st", "up_st"],
                load_fields=["fhr_st", "up_st", "fhr", "up"],
                trim_minutes=1.0,
            )
            loaders[class_name] = loader
            logger.info(
                f"  {class_name}: {len(eligible_guids)} eligible GUIDs"
            )
        except Exception as e:
            logger.warning(
                f"Could not create GUID loader for {class_name}: {e}"
            )
    return loaders


def run_full_test_pipeline(
    checkpoint_path: str,
    class_data_paths: Dict[str, List[str]],
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    skip_trajectory: bool = False,
    skip_per_sample: bool = False,
    n_diagnostic_samples: int = 10,
    min_epochs_per_guid: int = 5,
    max_guids: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the complete testing and analysis pipeline.

    Args:
        checkpoint_path: Path to trained model ``.ckpt`` file.
        class_data_paths: Dict mapping class names to lists of HDF5 file
            paths.  Keys are arbitrary class names (e.g. ``"healthy"``,
            ``"acidosis"``, ``"hie"``).  May contain 1, 2, or 3+ classes.
        output_dir: Where to save results.  Defaults to a timestamped
            directory next to the checkpoint.
        stats_path: Path to normalization statistics ``.hdf5`` file.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        max_samples: Limit per class (``None`` = all).
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        skip_trajectory: Skip trajectory analysis (requires GUID loaders).
        skip_per_sample: Skip per-sample diagnostics (slow).
        n_diagnostic_samples: Samples per class for per-sample diagnostics.
        min_epochs_per_guid: Minimum epochs for GUID trajectory analysis.
        max_guids: Maximum GUIDs per class for trajectory analysis.

    Returns:
        Summary dict with all results and figure paths.
    """
    from .analyses import run_all_analyses

    # Default output dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(
            Path(checkpoint_path).parent / f"test_results_{ts}"
        )

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Classes: {list(class_data_paths.keys())}")

    # 1. Load model
    runner = TransformerTestRunner.from_checkpoint(
        checkpoint_path, output_dir, device=device
    )

    # 2. Create standard DataLoaders
    class_loaders = _create_class_loaders(
        class_data_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        stats_path=stats_path,
    )

    # 3. Create GUID DataLoaders (for trajectory analysis)
    guid_loaders = None
    if not skip_trajectory:
        guid_loaders = _create_guid_loaders(
            class_data_paths,
            stats_path=stats_path,
            min_epochs_per_guid=min_epochs_per_guid,
            max_guids=max_guids,
        )

    # 4. Run all analyses
    results = run_all_analyses(
        runner=runner,
        class_loaders=class_loaders,
        guid_loaders=guid_loaders,
        max_samples=max_samples,
        skip_trajectory=skip_trajectory,
        skip_per_sample=skip_per_sample,
        n_diagnostic_samples=n_diagnostic_samples,
    )

    # 5. Save summary
    summary_path = Path(output_dir) / "test_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_path}")
    except Exception as e:
        logger.warning(f"Could not save summary: {e}")

    return results


def quick_test(
    checkpoint_path: str,
    class_data_paths: Dict[str, List[str]],
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    max_samples: int = 10,
) -> Dict[str, Any]:
    """Fast validation with limited samples, no trajectory, no per-sample.

    Useful for verifying that the pipeline works end-to-end.

    Args:
        checkpoint_path: Path to checkpoint.
        class_data_paths: Dict of class name to HDF5 paths.
        output_dir: Output directory.
        stats_path: Normalization stats path.
        device: Compute device.
        max_samples: Samples per class (default 10).

    Returns:
        Summary dict.
    """
    return run_full_test_pipeline(
        checkpoint_path=checkpoint_path,
        class_data_paths=class_data_paths,
        output_dir=output_dir,
        stats_path=stats_path,
        device=device,
        max_samples=max_samples,
        batch_size=16,
        num_workers=0,
        skip_trajectory=True,
        skip_per_sample=True,
        n_diagnostic_samples=0,
    )


# =========================================================================
# __main__ — Edit the settings below and run:
#   python -m model.transformer.tr_testing.run_tests
# =========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------
    # PATHS — Set these before running
    # -----------------------------------------------------------------

    # Trained model checkpoint (.ckpt or .pt)
    CHECKPOINT_PATH = None  # e.g. "./output/transformer/checkpoints/best.ckpt"

    # Normalization statistics file (.hdf5)
    STATS_PATH = None  # e.g. "./output/transformer/stats.hdf5"

    # Output directory (None = auto-generated next to checkpoint)
    OUTPUT_DIR = None  # e.g. "./output/transformer_test_results"

    # -----------------------------------------------------------------
    # HDF5 DATASET PATHS — One list per class
    # Add/remove classes as needed. Works with 1, 2, or 3+ classes.
    # -----------------------------------------------------------------

    CLASS_DATA_PATHS = {
        "healthy": [
            # "./data/k_fold/fold_1/test/healthy_no_bg_no_cs.hdf5",
            # "./data/k_fold/fold_1/test/healthy_no_bg_cs.hdf5",
            # "./data/k_fold/fold_1/test/healthy_bg_cs.hdf5",
            # "./data/k_fold/fold_1/test/healthy_bg_no_cs.hdf5",
        ],
        "acidosis": [
            # "./data/k_fold/fold_1/test/acidosis_cs.hdf5",
            # "./data/k_fold/fold_1/test/acidosis_no_cs.hdf5",
        ],
        "hie": [
            # "./data/k_fold/fold_1/test/hie_cs.hdf5",
            # "./data/k_fold/fold_1/test/hie_no_cs.hdf5",
        ],
    }

    # -----------------------------------------------------------------
    # PIPELINE OPTIONS
    # -----------------------------------------------------------------

    DEVICE = None               # "cuda", "cuda:0", "cpu", or None (auto)
    BATCH_SIZE = 64             # DataLoader batch size
    NUM_WORKERS = 4             # DataLoader workers (0 for debugging)
    MAX_SAMPLES = None          # None = all; set to e.g. 100 for faster runs
    SKIP_TRAJECTORY = False     # Skip per-GUID trajectory analysis
    SKIP_PER_SAMPLE = False     # Skip per-sample 17-row diagnostic figures
    N_DIAGNOSTIC_SAMPLES = 10   # Diagnostic figures per class
    MIN_EPOCHS_PER_GUID = 5    # Min epochs for GUID trajectory
    MAX_GUIDS = None            # Max GUIDs per class for trajectory (None = all)

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    # Remove empty class entries
    CLASS_DATA_PATHS = {k: v for k, v in CLASS_DATA_PATHS.items() if v}

    if not CHECKPOINT_PATH:
        print(
            "ERROR: Set CHECKPOINT_PATH in the __main__ section of "
            "run_tests.py before running."
        )
    elif not CLASS_DATA_PATHS:
        print(
            "ERROR: Add HDF5 paths to CLASS_DATA_PATHS in the __main__ "
            "section of run_tests.py before running."
        )
    else:
        run_full_test_pipeline(
            checkpoint_path=CHECKPOINT_PATH,
            class_data_paths=CLASS_DATA_PATHS,
            output_dir=OUTPUT_DIR,
            stats_path=STATS_PATH,
            device=DEVICE,
            max_samples=MAX_SAMPLES,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            skip_trajectory=SKIP_TRAJECTORY,
            skip_per_sample=SKIP_PER_SAMPLE,
            n_diagnostic_samples=N_DIAGNOSTIC_SAMPLES,
            min_epochs_per_guid=MIN_EPOCHS_PER_GUID,
            max_guids=MAX_GUIDS,
        )
