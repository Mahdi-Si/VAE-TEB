"""
Run latent trajectory analysis and cross-class comparison.

Supports two workflows:

1. **Single-run** (default): Load model + data, run trajectory analysis for all
   classes at once, then split by label and compare automatically.

2. **Compare-only**: Skip model inference entirely and compare pre-existing
   CSV outputs from previous trajectory runs.

Usage:
    # Single-run: analyse + compare in one shot
    python -m model.vae_teb_prediction.testing.run_latent_analysis_comparison \
        --config model/vae_teb_prediction/config.yaml

    # Compare-only: point at existing CSVs
    python -m model.vae_teb_prediction.testing.run_latent_analysis_comparison \
        --compare-only \
        --runs output_healthy/trajectory/guid_trajectory_features.csv:healthy \
               output_acidosis/trajectory/guid_trajectory_features.csv:acidosis \
        --stitched output_healthy/trajectory/latent_trajectories_stitched.csv:healthy \
                   output_acidosis/trajectory/latent_trajectories_stitched.csv:acidosis \
        --output comparison_results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from loguru import logger

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.vae_teb_prediction.testing.analyses.compare_trajectory_classes import (
    run_comparison,
)


# ---------------------------------------------------------------------------
# Single-run workflow helpers
# ---------------------------------------------------------------------------

def _run_trajectory_analysis(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    time_range_hours: float = 12.0,
    dim_reduction_method: str = "pca",
    n_changepoints: int = 5,
    num_workers: Optional[int] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    changepoint_algo: str = "pelt",
    preprocess_latent: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Run trajectory analysis using the existing pipeline.

    Returns:
        (trajectory_output_dir, results_dict)
    """
    import torch
    from model.vae_teb_prediction.testing.base import TestRunner
    from model.vae_teb_prediction.testing.analyses.trajectory import run_trajectory_analysis
    from model.vae_teb_prediction.testing.run_tests import (
        _resolve_runner_settings,
        _resolve_dataloader_settings,
        _create_guid_dataloader,
    )

    # Resolve config
    ckpt_path, out_dir = _resolve_runner_settings(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        config_path=config_path,
    )

    data_paths: List[str] = []
    if isinstance(data_path, str):
        data_paths = [data_path]
    elif data_path is not None:
        data_paths = list(data_path)

    data_paths, stats_path, _, resolved_workers, resolved_nf, resolved_kwargs = (
        _resolve_dataloader_settings(
            data_paths=data_paths,
            stats_path=stats_path,
            batch_size=None,
            num_workers=num_workers,
            normalize_fields=normalize_fields,
            dataset_kwargs=dataset_kwargs,
            config_path=config_path,
        )
    )

    if not data_paths:
        raise ValueError("No test data. Pass --data-path or config with vae_test_datasets.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Data: {data_paths}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Device: {torch_device}")

    # Load model
    runner = TestRunner.from_checkpoint(
        checkpoint_path=ckpt_path,
        output_dir=out_dir,
        device=torch_device,
    )

    # GUID-based loader
    _, guid_loader = _create_guid_dataloader(
        data_paths,
        stats_path=stats_path,
        min_epochs_per_guid=min_epochs_per_guid,
        max_guids=max_guids,
        normalize_fields=resolved_nf,
        num_workers=resolved_workers,
        dataset_kwargs=resolved_kwargs,
    )

    # Run trajectory analysis
    results = run_trajectory_analysis(
        runner,
        guid_loader,
        time_range_hours=time_range_hours,
        min_epochs_per_guid=min_epochs_per_guid,
        dim_reduction_method=dim_reduction_method,
        n_changepoints=n_changepoints,
        skip_dashboards=True,
        plot_3d=True,
        changepoint_algo=changepoint_algo,
        preprocess_latent_trajectories=preprocess_latent,
    )

    trajectory_dir = runner.ensure_dir("trajectory")
    return trajectory_dir, results


def _split_and_compare(
    trajectory_dir: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Split trajectory CSVs by label and run cross-class comparison.

    Reads guid_trajectory_features.csv and latent_trajectories_stitched.csv,
    groups by label, and feeds each group to the comparison pipeline.
    """
    features_csv = trajectory_dir / "guid_trajectory_features.csv"
    stitched_csv = trajectory_dir / "latent_trajectories_stitched.csv"

    if not features_csv.exists():
        logger.error(f"Features CSV not found: {features_csv}")
        return {"error": "no features CSV"}

    features_df = pd.read_csv(features_csv)

    if "label" not in features_df.columns:
        logger.error("Features CSV has no 'label' column — cannot split by class")
        return {"error": "no label column"}

    labels = [l for l in features_df["label"].unique() if l != "unknown"]
    if len(labels) < 2:
        logger.warning(f"Only {len(labels)} class(es) found ({labels}). Need >= 2 for comparison.")
        return {"error": f"only {len(labels)} class(es)", "labels": labels}

    logger.info(f"Found {len(labels)} classes: {labels}")

    # Split features CSV into per-class temp files
    split_dir = trajectory_dir / "_class_splits"
    split_dir.mkdir(exist_ok=True)

    run_specs: List[Tuple[str, str]] = []
    for label in labels:
        subset = features_df[features_df["label"] == label]
        split_path = split_dir / f"features_{label}.csv"
        subset.to_csv(split_path, index=False)
        run_specs.append((str(split_path), label))
        logger.info(f"  {label}: {len(subset)} GUIDs")

    # Split stitched CSV if available
    stitched_specs: Optional[List[Tuple[str, str]]] = None
    if stitched_csv.exists():
        stitched_df = pd.read_csv(stitched_csv)
        if "label" in stitched_df.columns:
            stitched_specs = []
            for label in labels:
                subset = stitched_df[stitched_df["label"] == label]
                split_path = split_dir / f"stitched_{label}.csv"
                subset.to_csv(split_path, index=False)
                stitched_specs.append((str(split_path), label))

    # Run comparison
    if output_dir is None:
        output_dir = trajectory_dir / "class_comparison"

    results = run_comparison(run_specs, str(output_dir), stitched_specs)
    logger.info(f"Comparison results saved to {output_dir}")
    return results


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_spec(spec: str) -> Tuple[str, str]:
    """Parse 'path:label' specification."""
    parts = spec.rsplit(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid spec '{spec}'. Expected format: /path/to/csv:class_label"
        )
    return parts[0], parts[1]


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of the comparison results."""
    print("\n=== Comparison Summary ===")
    if "error" in results:
        print(f"  Error: {results['error']}")
    else:
        print(f"  Classes: {results.get('classes', '?')}")
        print(f"  GUIDs: {results.get('n_guids', '?')}")
        if "pairwise" in results:
            for pw in results["pairwise"]:
                print(f"  {pw['pair']}: FID={pw['FID']:.4f}, MMD={pw['MMD']:.6f}")
        if "significant_features" in results:
            print(f"  Significant features (p<0.05): "
                  f"{results['significant_features']}/{results.get('n_features_tested', '?')}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_single_run(
    config_path: str = "model/vae_teb_prediction/config.yaml",
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    time_range_hours: float = 12.0,
    dim_reduction_method: str = "pca",
    n_changepoints: int = 5,
    changepoint_algo: str = "pelt",
    preprocess_latent: bool = False,
) -> Dict[str, Any]:
    """
    Run trajectory analysis on all data, then split by label and compare.

    Args:
        config_path: Path to YAML config file.
        checkpoint_path: Model checkpoint path (overrides config).
        data_path: HDF5 test data path(s) (overrides config).
        output_dir: Output directory (overrides config).
        stats_path: Normalization stats HDF5 path.
        device: Device string (e.g. "cuda:0", "cpu"). Auto-detects if None.
        min_epochs_per_guid: Minimum epochs per GUID to include.
        max_guids: Maximum GUIDs to process (None = all).
        time_range_hours: Hours before birth to analyse.
        dim_reduction_method: Dimensionality reduction ('pca', 'umap', 'tsne', etc.).
        n_changepoints: Changepoints per sample.
        changepoint_algo: Changepoint detection algorithm.
        preprocess_latent: If True, apply robust normalization/denoising.

    Returns:
        Comparison results dict.
    """
    trajectory_dir, traj_results = _run_trajectory_analysis(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        output_dir=output_dir,
        stats_path=stats_path,
        device=device,
        min_epochs_per_guid=min_epochs_per_guid,
        max_guids=max_guids,
        time_range_hours=time_range_hours,
        dim_reduction_method=dim_reduction_method,
        n_changepoints=n_changepoints,
        changepoint_algo=changepoint_algo,
        preprocess_latent=preprocess_latent,
    )

    logger.info(f"Trajectory analysis done: {traj_results.get('n_guids', '?')} GUIDs, "
                f"{traj_results.get('n_epochs', '?')} epochs")

    comparison_output = Path(output_dir) / "class_comparison" if output_dir else None
    results = _split_and_compare(trajectory_dir, comparison_output)
    _print_summary(results)
    return results


def run_compare_only(
    runs: List[str],
    output: str,
    stitched: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compare pre-existing trajectory CSVs from separate runs.

    Args:
        runs: List of "path:label" specs pointing to guid_trajectory_features.csv files.
        output: Output directory for comparison results.
        stitched: Optional list of "path:label" specs pointing to
            latent_trajectories_stitched.csv files. Enables FID/MMD computation.

    Returns:
        Comparison results dict.
    """
    run_specs = [_parse_spec(s) for s in runs]
    stitched_specs = [_parse_spec(s) for s in stitched] if stitched else None

    results = run_comparison(run_specs, output, stitched_specs)
    _print_summary(results)
    return results


def main(
    compare_only: bool = False,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    data_path: Optional[Union[str, List[str]]] = None,
    output_dir: Optional[str] = None,
    stats_path: Optional[str] = None,
    device: Optional[str] = None,
    min_epochs_per_guid: int = 10,
    max_guids: Optional[int] = None,
    time_range_hours: float = 12.0,
    dim_reduction_method: str = "pca",
    n_changepoints: int = 5,
    runs: Optional[List[str]] = None,
    stitched: Optional[List[str]] = None,
    changepoint_algo: str = "pelt",
    preprocess_latent: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for latent trajectory comparison.

    Can be called directly from Python or driven by the CLI below.

    Args:
        compare_only: If True, skip model inference and compare existing CSVs.
        config_path: Path to YAML config file (single-run mode).
        checkpoint_path: Model checkpoint path (single-run mode, overrides config).
        data_path: HDF5 test data path(s) (single-run mode, overrides config).
        output_dir: Output directory.
        stats_path: Normalization stats HDF5 path.
        device: Device string (e.g. "cuda:0", "cpu"). Auto-detects if None.
        min_epochs_per_guid: Minimum epochs per GUID to include.
        max_guids: Maximum GUIDs to process (None = all).
        time_range_hours: Hours before birth to analyse.
        dim_reduction_method: Dimensionality reduction ('pca', 'umap', 'tsne', etc.).
        n_changepoints: Changepoints per sample.
        runs: List of "path:label" specs for compare-only mode.
        stitched: Optional list of "path:label" specs for compare-only mode (FID/MMD).
        changepoint_algo: Changepoint detection algorithm.
        preprocess_latent: If True, apply robust normalization/denoising.

    Returns:
        Comparison results dict.
    """
    if compare_only:
        if not runs:
            raise ValueError("'runs' is required in compare-only mode")
        if not output_dir:
            raise ValueError("'output_dir' is required in compare-only mode")
        return run_compare_only(runs=runs, output=output_dir, stitched=stitched)
    else:
        if not config_path and not checkpoint_path:
            raise ValueError("Either 'config_path' or 'checkpoint_path' is required in single-run mode")
        assert config_path is not None  # validated above
        return run_single_run(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            data_path=data_path,
            output_dir=output_dir,
            stats_path=stats_path,
            device=device,
            min_epochs_per_guid=min_epochs_per_guid,
            max_guids=max_guids,
            time_range_hours=time_range_hours,
            dim_reduction_method=dim_reduction_method,
            n_changepoints=n_changepoints,
            changepoint_algo=changepoint_algo,
            preprocess_latent=preprocess_latent,
        )


def _cli():
    """Parse command-line arguments and call main()."""
    parser = argparse.ArgumentParser(
        description="Run latent trajectory analysis and cross-class comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-run: analyse + compare in one shot
  python -m model.vae_teb_prediction.testing.run_latent_analysis_comparison \\
      --config model/vae_teb_prediction/config.yaml

  # Compare-only: point at existing CSVs from separate runs
  python -m model.vae_teb_prediction.testing.run_latent_analysis_comparison \\
      --compare-only \\
      --runs output_healthy/trajectory/guid_trajectory_features.csv:healthy \\
             output_acidosis/trajectory/guid_trajectory_features.csv:acidosis \\
      --output comparison_results/

  # Single-run with custom settings
  python -m model.vae_teb_prediction.testing.run_latent_analysis_comparison \\
      --config model/vae_teb_prediction/config.yaml \\
      --min-epochs 5 --time-range 8.0 --dim-reduction umap
        """,
    )

    # --- Mode selection ---
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip model inference. Only compare pre-existing CSV outputs.",
    )

    # --- Single-run args ---
    single = parser.add_argument_group("single-run options (ignored with --compare-only)")
    single.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    single.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    single.add_argument("--data-path", type=str, nargs="+", default=None, help="HDF5 test data path(s).")
    single.add_argument("--stats-path", type=str, default=None, help="Normalization stats HDF5.")
    single.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu).")
    single.add_argument("--min-epochs", type=int, default=10, help="Min epochs per GUID (default: 10).")
    single.add_argument("--max-guids", type=int, default=None, help="Max GUIDs to process.")
    single.add_argument("--time-range", type=float, default=12.0, help="Hours before birth (default: 12).")
    single.add_argument("--dim-reduction", type=str, default="pca", help="Dim reduction method (default: pca).")
    single.add_argument("--n-changepoints", type=int, default=5, help="Changepoints per sample (default: 5).")
    single.add_argument(
        "--changepoint-algo", type=str, default="pelt",
        choices=["pelt", "binseg", "bottomup", "window", "dynp", "gradient"],
        help="Changepoint detection algorithm (default: pelt).",
    )
    single.add_argument("--preprocess-latent", action="store_true", help="Apply robust normalization/denoising to latent z-columns.")

    # --- Compare-only args ---
    compare = parser.add_argument_group("compare-only options")
    compare.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Per-GUID feature CSVs with class labels (format: path:label).",
    )
    compare.add_argument(
        "--stitched",
        nargs="*",
        default=None,
        help="Optional stitched trajectory CSVs with class labels (format: path:label).",
    )

    # --- Shared ---
    parser.add_argument("--output", type=str, default=None, help="Output directory for comparison results.")

    args = parser.parse_args()

    data_path = args.data_path
    if data_path and len(data_path) == 1:
        data_path = data_path[0]

    main(
        compare_only=args.compare_only,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_path=data_path,
        output_dir=args.output,
        stats_path=args.stats_path,
        device=args.device,
        min_epochs_per_guid=args.min_epochs,
        max_guids=args.max_guids,
        time_range_hours=args.time_range,
        dim_reduction_method=args.dim_reduction,
        n_changepoints=args.n_changepoints,
        runs=args.runs,
        stitched=args.stitched,
        changepoint_algo=args.changepoint_algo,
        preprocess_latent=args.preprocess_latent,
    )


if __name__ == "__main__":
    # If CLI arguments are provided, use argparse-based entry point.
    if len(sys.argv) > 1:
        _cli()
        sys.exit(0)

    # ----------------------------------------------------------------
    # Edit these variables to run directly without CLI arguments.
    # Set COMPARE_ONLY = True to skip model inference and compare
    # pre-existing CSVs, or False to run the full pipeline.
    # ----------------------------------------------------------------

    COMPARE_ONLY = False

    # --- Single-run settings (used when COMPARE_ONLY = False) ---
    CONFIG_PATH = "model/vae_teb_prediction/config.yaml"
    CHECKPOINT_PATH = None       # None = read from config
    DATA_PATH = None             # None = read from config
    OUTPUT_DIR = None            # None = read from config
    STATS_PATH = None            # None = read from config
    DEVICE = None                # None = auto-detect
    MIN_EPOCHS_PER_GUID = 10
    MAX_GUIDS = None             # None = all
    TIME_RANGE_HOURS = 12.0
    DIM_REDUCTION_METHOD = "pca"
    N_CHANGEPOINTS = 5
    CHANGEPOINT_ALGO = "pelt"
    PREPROCESS_LATENT = False

    # --- Compare-only settings (used when COMPARE_ONLY = True) ---
    RUNS = [
        # "output_healthy/trajectory/guid_trajectory_features.csv:healthy",
        # "output_acidosis/trajectory/guid_trajectory_features.csv:acidosis",
        # "output_hie/trajectory/guid_trajectory_features.csv:HIE",
    ]
    STITCHED = [
        # "output_healthy/trajectory/latent_trajectories_stitched.csv:healthy",
        # "output_acidosis/trajectory/latent_trajectories_stitched.csv:acidosis",
        # "output_hie/trajectory/latent_trajectories_stitched.csv:HIE",
    ]
    COMPARE_OUTPUT = "comparison_results/"

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    if COMPARE_ONLY:
        main(
            compare_only=True,
            runs=RUNS,
            stitched=STITCHED or None,
            output_dir=COMPARE_OUTPUT,
        )
    else:
        main(
            compare_only=False,
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            stats_path=STATS_PATH,
            device=DEVICE,
            min_epochs_per_guid=MIN_EPOCHS_PER_GUID,
            max_guids=MAX_GUIDS,
            time_range_hours=TIME_RANGE_HOURS,
            dim_reduction_method=DIM_REDUCTION_METHOD,
            n_changepoints=N_CHANGEPOINTS,
            changepoint_algo=CHANGEPOINT_ALGO,
            preprocess_latent=PREPROCESS_LATENT,
        )
