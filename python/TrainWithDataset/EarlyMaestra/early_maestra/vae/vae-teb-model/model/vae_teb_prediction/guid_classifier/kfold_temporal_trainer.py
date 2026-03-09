"""K-Fold parallel training and aggregation for the temporal VAE classifier.

Orchestrates multi-fold training by dispatching each fold to a separate process
(one per GPU) using :class:`concurrent.futures.ProcessPoolExecutor` with a
``'spawn'`` multiprocessing context for CUDA safety.

Typical usage::

    python -m model.vae_teb_prediction.guid_classifier.kfold_temporal_trainer

All settings are read from ``config_temporal.yaml``.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import time
import traceback
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger


# ---------------------------------------------------------------------------
#  Single-fold entry point (designed for subprocess)
# ---------------------------------------------------------------------------


def train_single_fold_temporal(
    fold_id: int,
    gpu_id: int,
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
    parent_run_id: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Train and evaluate one fold of the temporal classifier.

    Designed to run in a subprocess launched by
    :func:`run_kfold_temporal_parallel`.  Sets ``CUDA_VISIBLE_DEVICES``
    **before** any PyTorch CUDA initialisation so the subprocess sees only
    the assigned GPU.

    The function delegates training to
    :func:`temporal_classifier_trainer.train_fold` and evaluation to
    :func:`evaluate_temporal_classifier.evaluate_single_fold_temporal`.

    Args:
        fold_id: Fold number (1-10).
        gpu_id: GPU device ID to expose via ``CUDA_VISIBLE_DEVICES``.
        base_config_path: Path to ``config_temporal.yaml``.
        kfold_base_path: Path to the ``k_fold_dataset/`` directory.
        output_base_dir: Root output directory (``fold_N/`` is created
            beneath this).
        vae_checkpoint: Path to the frozen VAE ``.ckpt`` file.
        parent_run_id: Optional MLflow parent run ID for nested grouping.
        **kwargs: Additional config overrides (e.g. ``target_fpr``).

    Returns:
        Dict with fold results including ``fold_id``, ``status``,
        ``training_time_minutes``, ``primary_threshold``,
        ``test_sensitivity_mean``, etc.
    """
    logger.info(
        "Starting temporal fold {} on GPU {} (pid={})",
        fold_id, gpu_id, os.getpid(),
    )

    # Set CUDA_VISIBLE_DEVICES early — before any torch.cuda call
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Load config
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override paths in config
        config.setdefault("dataset_config", {})["kfold_base_path"] = kfold_base_path
        config.setdefault("model_config", {})["vae_checkpoint"] = vae_checkpoint

        fold_output_dir = Path(output_base_dir) / f"fold_{fold_id}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        config["general_config"]["folders_config"]["out_dir_base"] = str(
            fold_output_dir
        )

        # Apply any additional overrides from kwargs
        for key, value in kwargs.items():
            if "." in key:
                parts = key.split(".")
                cfg = config
                for part in parts[:-1]:
                    cfg = cfg.setdefault(part, {})
                cfg[parts[-1]] = value

        # ----- Training -------------------------------------------------------
        from model.vae_teb_prediction.guid_classifier.temporal_classifier_trainer import (
            train_fold,
        )

        start_time = time.time()
        checkpoint_dir, trainer_instance = train_fold(
            fold_id=fold_id,
            config=config,
            gpu_id=0,  # Already mapped via CUDA_VISIBLE_DEVICES
        )
        training_time_min = (time.time() - start_time) / 60.0

        # ----- Evaluation -----------------------------------------------------
        from model.vae_teb_prediction.guid_classifier.evaluate_temporal_classifier import (
            evaluate_single_fold_temporal,
        )

        eval_cfg = config.get("model_config", {}).get("evaluation", {}) or {}
        target_fpr = float(
            kwargs.get("target_fpr", eval_cfg.get("target_fpr", 0.2))
        )
        exclude_last_minutes = float(
            eval_cfg.get("exclude_last_minutes", 30.0)
        )
        decision_time_hours = float(
            eval_cfg.get("decision_time_hours", 1.0)
        )
        max_gap_multiplier = eval_cfg.get("max_gap_multiplier")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        eval_results = evaluate_single_fold_temporal(
            fold_dir=str(fold_output_dir),
            config=config,
            device=device,
            target_fpr=target_fpr,
            exclude_last_minutes=exclude_last_minutes,
            decision_time_hours=decision_time_hours,
            max_gap_multiplier=max_gap_multiplier,
        )

        # Merge training metadata into evaluation results
        eval_results["training_time_minutes"] = training_time_min
        eval_results["gpu_id"] = gpu_id
        eval_results["checkpoint_dir"] = checkpoint_dir

        # Save combined fold results
        results_path = fold_output_dir / "fold_results.json"
        with open(results_path, "w") as f:
            json.dump(_serialise(eval_results), f, indent=2)

        logger.info(
            "Fold {} completed — training {:.1f} min, threshold {:.4f}, "
            "test sens {:.3f}",
            fold_id,
            training_time_min,
            eval_results.get("primary_threshold", 0.0),
            eval_results.get("test_sensitivity_mean", 0.0),
        )

        # Cleanup
        del trainer_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return eval_results

    except Exception as exc:
        logger.exception("Fold {} failed:", fold_id)
        return {
            "fold_id": fold_id,
            "gpu_id": gpu_id,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
#  K-Fold parallel orchestrator
# ---------------------------------------------------------------------------


def run_kfold_temporal_parallel(
    num_folds: int,
    gpu_ids: List[int],
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
    max_parallel: Optional[int] = None,
    fold_ids: Optional[List[int]] = None,
    sequential: bool = False,
    **kwargs,
) -> List[Dict]:
    """Run k-fold temporal classifier training in parallel.

    Uses :class:`ProcessPoolExecutor` with ``'spawn'`` context for CUDA
    safety.  GPU assignment is round-robin:
    ``gpu_ids[job_idx % len(gpu_ids)]``.

    Args:
        num_folds: Total number of folds configured in the dataset (e.g. 10).
        gpu_ids: List of GPU device IDs to use (e.g. ``[0, 1, 2, 3]``).
        base_config_path: Path to ``config_temporal.yaml``.
        kfold_base_path: Path to ``k_fold_dataset/`` directory.
        output_base_dir: Root output directory.
        vae_checkpoint: Path to frozen VAE ``.ckpt`` file.
        max_parallel: Maximum simultaneous fold processes.  Defaults to
            ``min(len(gpu_ids), len(selected_folds))``.
        fold_ids: Optional subset of fold IDs to run.  ``None`` = all
            ``1..num_folds``.
        sequential: If ``True``, run folds sequentially on a single GPU
            (useful for debugging).
        **kwargs: Additional config overrides forwarded to each fold.

    Returns:
        List of per-fold result dicts, sorted by ``fold_id``.
    """
    # --- Resolve fold selection ------------------------------------------------
    if fold_ids is None:
        selected_folds = list(range(1, num_folds + 1))
    else:
        selected_folds = sorted({int(fid) for fid in fold_ids})
        invalid = [fid for fid in selected_folds if fid < 1 or fid > num_folds]
        if invalid:
            raise ValueError(
                f"Invalid fold IDs: {invalid} (valid range 1..{num_folds})"
            )
    if not selected_folds:
        raise ValueError("No folds requested for execution.")

    requested_count = len(selected_folds)

    if max_parallel is None:
        max_parallel = min(len(gpu_ids), requested_count)
    else:
        max_parallel = max(1, min(max_parallel, requested_count))

    execution_mode = (
        "parallel" if (not sequential and max_parallel > 1) else "sequential"
    )

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("TEMPORAL K-FOLD CROSS-VALIDATION")
    logger.info("=" * 80)
    logger.info("Requested folds: {}", selected_folds)
    logger.info("Execution mode: {}", execution_mode)
    logger.info("GPUs: {}", gpu_ids)
    if execution_mode == "parallel":
        logger.info("Max parallel folds: {}", max_parallel)

    # --- Execute folds ---------------------------------------------------------
    all_results: List[Dict] = []

    common_kwargs = dict(
        base_config_path=base_config_path,
        kfold_base_path=kfold_base_path,
        output_base_dir=output_base_dir,
        vae_checkpoint=vae_checkpoint,
        parent_run_id=None,
        **kwargs,
    )

    if sequential or max_parallel == 1:
        logger.info("Running folds sequentially...")
        for job_idx, fold_id in enumerate(selected_folds):
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            result = train_single_fold_temporal(
                fold_id=fold_id, gpu_id=gpu_id, **common_kwargs,
            )
            all_results.append(result)
            logger.info("Fold {} status: {}", fold_id, result.get("status"))
    else:
        logger.info("Running folds in parallel...")
        spawn_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_parallel, mp_context=spawn_ctx,
        ) as executor:
            futures = {}
            for job_idx, fold_id in enumerate(selected_folds):
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                future = executor.submit(
                    train_single_fold_temporal,
                    fold_id=fold_id,
                    gpu_id=gpu_id,
                    **common_kwargs,
                )
                futures[future] = fold_id

            for future in as_completed(futures):
                fold_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    logger.info(
                        "Fold {} status: {}", fold_id, result.get("status"),
                    )
                except Exception as exc:
                    logger.exception("Fold {} raised exception:", fold_id)
                    all_results.append({
                        "fold_id": fold_id,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })

    all_results.sort(key=lambda x: x.get("fold_id", 0))

    # --- Save per-fold results -------------------------------------------------
    kfold_results_path = Path(output_base_dir) / "kfold_results.json"
    with open(kfold_results_path, "w") as f:
        json.dump(_serialise(all_results), f, indent=2)

    # --- Compute summary statistics -------------------------------------------
    successful = [r for r in all_results if r.get("status") == "success"]

    summary = _build_kfold_summary(
        all_results=all_results,
        successful=successful,
        num_folds=num_folds,
        selected_folds=selected_folds,
    )

    summary_path = Path(output_base_dir) / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_serialise(summary), f, indent=2)

    # --- Cross-fold aggregation ------------------------------------------------
    if successful:
        try:
            aggregate_temporal_results(
                output_base_dir=str(output_base_dir),
                fold_results=successful,
            )
        except Exception as exc:
            logger.error("Cross-fold aggregation failed: {}", exc)
            logger.error(traceback.format_exc())

    # --- Print summary ---------------------------------------------------------
    _print_summary(summary, successful)

    # --- Save execution metadata -----------------------------------------------
    execution_metadata = {
        "num_folds_configured": num_folds,
        "requested_folds": selected_folds,
        "successful_folds": [r["fold_id"] for r in successful],
        "failed_folds": [
            r["fold_id"]
            for r in all_results
            if r.get("status") != "success"
        ],
        "execution_mode": execution_mode,
        "max_parallel": max_parallel if not sequential else 1,
        "gpu_ids": gpu_ids,
        "execution_timestamp": datetime.now().isoformat(),
    }
    metadata_path = Path(output_base_dir) / "execution_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(execution_metadata, f, indent=2)

    return all_results


# ---------------------------------------------------------------------------
#  Cross-fold aggregation
# ---------------------------------------------------------------------------


def aggregate_temporal_results(
    output_base_dir: str,
    fold_ids: Optional[List[int]] = None,
    fold_results: Optional[List[Dict]] = None,
    regenerate_from_predictions: bool = False,
    exclude_last_minutes: float = 30.0,
) -> Dict:
    """Aggregate evaluation results across folds.

    Loads ``fold_results.json`` from each fold directory (or uses
    pre-loaded ``fold_results``), calls
    :func:`generate_aggregated_plots` from ``evaluate_classifier.py``
    to produce cross-fold plots with mean + min/max bands, and saves
    ``aggregated_results.json``.

    Args:
        output_base_dir: Root directory containing ``fold_N/`` subdirs.
        fold_ids: Optional subset of fold IDs to include.  ``None`` =
            discover automatically.
        fold_results: Pre-loaded list of fold result dicts.  If provided,
            ``fold_ids`` is ignored.
        regenerate_from_predictions: If ``True``, re-run evaluation from
            cached prediction CSVs before aggregating.
        exclude_last_minutes: Minutes to exclude in time-based analysis
            (forwarded to aggregated plots).

    Returns:
        Aggregated result dict with cross-fold statistics.
    """
    output_base_dir = Path(output_base_dir)

    # --- Collect fold results --------------------------------------------------
    if fold_results is None:
        fold_results = _load_fold_results_from_disk(output_base_dir, fold_ids)

    successful = [r for r in fold_results if r.get("status") == "success"]

    if not successful:
        logger.warning("No successful folds to aggregate")
        return {"status": "failed", "n_successful": 0}

    logger.info(
        "Aggregating {} successful folds: {}",
        len(successful),
        [r["fold_id"] for r in successful],
    )

    # --- Generate aggregated plots ---------------------------------------------
    try:
        from model.vae_teb_prediction.evaluate_classifier import (
            generate_aggregated_plots,
        )

        generate_aggregated_plots(
            all_fold_results=successful,
            output_base_dir=output_base_dir,
            n_folds=len(successful),
        )
        logger.info(
            "Aggregated plots saved to: {}",
            output_base_dir / "aggregated_plots",
        )
    except Exception as exc:
        logger.error("generate_aggregated_plots failed: {}", exc)
        logger.error(traceback.format_exc())

    # --- Compute aggregated statistics -----------------------------------------
    from model.vae_teb_prediction.guid_classifier.evaluate_temporal_classifier import (
        _aggregate_temporal_results,
    )

    aggregated = _aggregate_temporal_results(successful)
    aggregated["successful_folds"] = [r["fold_id"] for r in successful]

    agg_path = output_base_dir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(_serialise(aggregated), f, indent=2)

    logger.info("Aggregated results saved to: {}", agg_path)
    return aggregated


# ---------------------------------------------------------------------------
#  Private helpers
# ---------------------------------------------------------------------------


def _load_fold_results_from_disk(
    output_base_dir: Path,
    fold_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """Load ``fold_results.json`` from each fold directory on disk.

    Args:
        output_base_dir: Root directory containing ``fold_N/`` subdirs.
        fold_ids: Optional subset of fold IDs.  ``None`` = discover all.

    Returns:
        List of fold result dicts.
    """
    all_fold_dirs = sorted(
        [
            d
            for d in output_base_dir.iterdir()
            if d.is_dir() and d.name.startswith("fold_")
        ],
        key=lambda x: int(x.name.split("_")[1]),
    )

    if fold_ids is not None:
        fold_id_set = set(fold_ids)
        all_fold_dirs = [
            d
            for d in all_fold_dirs
            if int(d.name.split("_")[1]) in fold_id_set
        ]

    results: List[Dict] = []
    for fold_dir in all_fold_dirs:
        results_path = fold_dir / "fold_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results.append(json.load(f))
        else:
            logger.warning("No fold_results.json in {}", fold_dir)

    return results


def _build_kfold_summary(
    all_results: List[Dict],
    successful: List[Dict],
    num_folds: int,
    selected_folds: List[int],
) -> Dict:
    """Build a summary dict with mean/std statistics across folds.

    Args:
        all_results: All fold result dicts (including failures).
        successful: Only successful fold result dicts.
        num_folds: Total configured folds.
        selected_folds: Fold IDs that were requested.

    Returns:
        Summary dict.
    """
    import numpy as np

    summary: Dict = {
        "configured_num_folds": num_folds,
        "requested_folds": selected_folds,
        "successful_folds": len(successful),
        "failed_folds": len(all_results) - len(successful),
    }

    if not successful:
        return summary

    def _ms(key: str):
        vals = [r.get(key, 0.0) for r in successful]
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std())

    # Training metrics
    train_time_mean, train_time_std = _ms("training_time_minutes")
    summary["training_metrics"] = {
        "mean_training_time_minutes": train_time_mean,
        "std_training_time_minutes": train_time_std,
    }

    # Validation metrics
    val_sens_m, val_sens_s = _ms("validation_sensitivity")
    val_spec_m, val_spec_s = _ms("validation_specificity")
    val_fpr_m, val_fpr_s = _ms("validation_fpr")
    thresh_m, thresh_s = _ms("primary_threshold")
    summary["validation_metrics"] = {
        "mean_threshold": thresh_m,
        "std_threshold": thresh_s,
        "mean_sensitivity": val_sens_m,
        "std_sensitivity": val_sens_s,
        "mean_specificity": val_spec_m,
        "std_specificity": val_spec_s,
        "mean_fpr": val_fpr_m,
        "std_fpr": val_fpr_s,
    }

    # Test metrics (PRIMARY — committed_overall)
    test_sens_m, test_sens_s = _ms("test_sensitivity_mean")
    test_spec_m, test_spec_s = _ms("test_specificity_mean")
    test_fpr_m, test_fpr_s = _ms("test_fpr_mean")
    summary["test_metrics_primary"] = {
        "mean_sensitivity": test_sens_m,
        "std_sensitivity": test_sens_s,
        "mean_specificity": test_spec_m,
        "std_specificity": test_spec_s,
        "mean_fpr": test_fpr_m,
        "std_fpr": test_fpr_s,
    }

    summary["individual_results"] = all_results

    return summary


def _print_summary(summary: Dict, successful: List[Dict]) -> None:
    """Print a human-readable k-fold summary to the log.

    Args:
        summary: Summary dict from :func:`_build_kfold_summary`.
        successful: Successful fold result dicts.
    """
    logger.info("=" * 80)
    logger.info("TEMPORAL K-FOLD CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "Total requested: {} | Successful: {} | Failed: {}",
        len(summary.get("requested_folds", [])),
        summary.get("successful_folds", 0),
        summary.get("failed_folds", 0),
    )

    if not successful:
        logger.warning("No successful folds — no metrics to report.")
        return

    val = summary.get("validation_metrics", {})
    logger.info("")
    logger.info("Validation Metrics:")
    logger.info(
        "  Threshold: {:.4f} +/- {:.4f}",
        val.get("mean_threshold", 0.0),
        val.get("std_threshold", 0.0),
    )
    logger.info(
        "  Sensitivity: {:.4f} +/- {:.4f}",
        val.get("mean_sensitivity", 0.0),
        val.get("std_sensitivity", 0.0),
    )
    logger.info(
        "  Specificity: {:.4f} +/- {:.4f}",
        val.get("mean_specificity", 0.0),
        val.get("std_specificity", 0.0),
    )

    test = summary.get("test_metrics_primary", {})
    logger.info("")
    logger.info("Test Metrics (PRIMARY — committed_overall):")
    logger.info(
        "  Sensitivity: {:.4f} +/- {:.4f}",
        test.get("mean_sensitivity", 0.0),
        test.get("std_sensitivity", 0.0),
    )
    logger.info(
        "  Specificity: {:.4f} +/- {:.4f}",
        test.get("mean_specificity", 0.0),
        test.get("std_specificity", 0.0),
    )
    logger.info(
        "  FPR: {:.4f} +/- {:.4f}",
        test.get("mean_fpr", 0.0),
        test.get("std_fpr", 0.0),
    )
    logger.info("=" * 80)


def _serialise(obj):
    """Recursively convert non-JSON-serialisable types.

    Handles ``torch.Tensor``, ``numpy`` scalars, ``Path``, ``DataFrame``
    and similar types that may appear in fold result dicts.

    Args:
        obj: Any Python object to serialise.

    Returns:
        JSON-compatible object.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas DataFrame
    if hasattr(obj, "to_dict"):
        return obj.to_dict("records")
    return obj


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for temporal k-fold training.

    Reads ``config_temporal.yaml`` from the same directory as this module,
    runs training and evaluation for all configured folds, and generates
    cross-fold aggregated plots.
    """
    base_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config_temporal.yaml",
    )

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(
            f"Config file not found: {base_config_path}"
        )

    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    general_cfg = config.get("general_config", {})
    model_cfg = config.get("model_config", {})
    dataset_cfg = config.get("dataset_config", {})

    gpu_ids = general_cfg.get("cuda_devices", [0])
    max_parallel = general_cfg.get("max_parallel_folds", 1)
    output_base_dir = general_cfg.get("folders_config", {}).get(
        "out_dir_base", os.getcwd()
    )
    vae_checkpoint = model_cfg.get("vae_checkpoint")
    kfold_base_path = dataset_cfg.get("kfold_base_path", "")
    num_folds = dataset_cfg.get("num_folds", 10)
    fold_ids_cfg = dataset_cfg.get("fold_ids")  # null = all

    run_parallel = max_parallel > 1

    logger.info("Config file: {}", base_config_path)
    logger.info("Output base dir: {}", output_base_dir)
    logger.info("K-fold base path: {}", kfold_base_path)
    logger.info("GPUs: {}", gpu_ids)
    logger.info("Max parallel folds: {}", max_parallel)
    logger.info("Num folds: {}", num_folds)
    logger.info("Fold IDs: {}", fold_ids_cfg if fold_ids_cfg else "all")

    # Validate paths
    if vae_checkpoint and not os.path.exists(vae_checkpoint):
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_checkpoint}"
        )

    results = run_kfold_temporal_parallel(
        num_folds=num_folds,
        gpu_ids=gpu_ids,
        base_config_path=base_config_path,
        kfold_base_path=kfold_base_path,
        output_base_dir=output_base_dir,
        vae_checkpoint=vae_checkpoint,
        max_parallel=max_parallel,
        fold_ids=fold_ids_cfg,
        sequential=not run_parallel,
    )

    successful = [r for r in results if r.get("status") == "success"]
    logger.info(
        "K-Fold Cross-Validation completed: {}/{} folds successful",
        len(successful),
        len(results),
    )
    logger.info("Results saved to: {}", output_base_dir)


if __name__ == "__main__":
    main()
