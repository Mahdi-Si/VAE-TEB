"""K-Fold parallel training for the transformer-based GRU classifier.

Orchestrates multi-fold training by dispatching each fold to a separate
process (one per GPU) using :class:`concurrent.futures.ProcessPoolExecutor`
with a ``'spawn'`` multiprocessing context for CUDA safety.

Includes MLflow integration: parent/child run nesting, per-fold metric
and artifact logging, and cross-fold summary logging.

Typical usage::

    python -m model.transformer.classification.kfold_classification_trainer

All settings are read from ``config_classification.yaml``.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import time
import traceback
import yaml
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger


# ---------------------------------------------------------------------------
#  MLflow helpers
# ---------------------------------------------------------------------------


def create_parent_mlflow_run(
    config_data: Optional[Dict],
    experiment_tag: str,
    extra_tags: Optional[Dict[str, str]] = None,
):
    """Create a parent MLflow run for the k-fold execution.

    All per-fold runs are nested under this parent via the
    ``mlflow.parentRunId`` tag.

    Args:
        config_data: Full config dict.
        experiment_tag: Fallback experiment name.
        extra_tags: Additional tags to attach to the parent run.

    Returns:
        Tuple of ``(MlflowClient, parent_run_id)`` or
        ``(None, None)`` if MLflow is disabled or unavailable.
    """
    if not config_data:
        return None, None

    mlflow_cfg = (
        config_data.get("advanced_config", {})
        .get("tracking", {})
        .get("mlflow", {})
        or {}
    )
    if not mlflow_cfg.get("enabled"):
        return None, None

    try:
        import mlflow

        tracking_uri = mlflow_cfg.get("tracking_uri")
        experiment_name = (
            mlflow_cfg.get("experiment_name") or experiment_tag
        )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        elif experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
            experiment_id = experiment.experiment_id
            logger.info(
                "Restored deleted MLflow experiment '{}' (id={})",
                experiment_name, experiment_id,
            )
        else:
            experiment_id = experiment.experiment_id

        tags = dict(mlflow_cfg.get("tags") or {})
        tags["kfold.role"] = "parent"
        tags["kfold.model_type"] = "transformer_classifier"
        if extra_tags:
            tags.update({k: str(v) for k, v in extra_tags.items()})

        run_name = f"{experiment_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
        parent_run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        )

        parent_run_id = parent_run.info.run_id
        logger.info(
            "Created parent MLflow run: {} (id={})",
            run_name, parent_run_id,
        )
        return client, parent_run_id

    except Exception as exc:
        logger.warning("Failed to create parent MLflow run: {}", exc)
        return None, None


def _inject_mlflow_fold_tags(
    config: Dict,
    fold_id: int,
    gpu_id: int,
    fold_datasets: Optional[Dict] = None,
    parent_run_id: Optional[str] = None,
) -> None:
    """Inject per-fold MLflow tags into the config dict (in-place).

    Args:
        config: Full config dict to modify in-place.
        fold_id: Fold number.
        gpu_id: Assigned GPU ID.
        fold_datasets: Optional dict with train/val/test file lists.
        parent_run_id: Optional parent run ID for nesting.
    """
    advanced_cfg = config.setdefault("advanced_config", {})
    tracking_cfg = advanced_cfg.setdefault("tracking", {})
    mlflow_cfg = tracking_cfg.setdefault("mlflow", {})

    tags = dict(mlflow_cfg.get("tags") or {})
    tags.update({
        "kfold.role": "fold",
        "kfold.model_type": "transformer_classifier",
        "kfold.fold_id": str(fold_id),
        "kfold.gpu_id": str(gpu_id),
    })
    if fold_datasets:
        tags["kfold.train_files"] = str(
            len(fold_datasets.get("train", []))
        )
        tags["kfold.val_files"] = str(
            len(fold_datasets.get("val", []))
        )
        tags["kfold.test_files"] = str(
            len(fold_datasets.get("test", []))
        )
    if parent_run_id:
        tags["mlflow.parentRunId"] = parent_run_id

    mlflow_cfg["tags"] = tags
    if not mlflow_cfg.get("run_name"):
        mlflow_cfg["run_name"] = f"classification-fold-{fold_id:02d}"


def _log_fold_mlflow_metrics(
    mlflow_logger,
    metrics: Dict[str, float],
    step: Optional[int] = None,
) -> None:
    """Log a dict of metrics to the fold's MLflow run.

    Args:
        mlflow_logger: Lightning MLFlowLogger instance (may be ``None``).
        metrics: Dict of metric name -> value.
        step: Optional global step.
    """
    if mlflow_logger is None or not metrics:
        return

    run_id = getattr(mlflow_logger, "run_id", None)
    client = getattr(mlflow_logger, "experiment", None)
    if run_id is None or client is None:
        return

    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.item()
        try:
            client.log_metric(run_id, key, float(value), step=step or 0)
        except Exception as exc:
            logger.warning("Failed to log MLflow metric '{}': {}", key, exc)


def _log_fold_mlflow_artifact(
    mlflow_logger,
    path,
    *,
    is_dir: bool = False,
) -> None:
    """Log a file or directory as an MLflow artifact.

    Args:
        mlflow_logger: Lightning MLFlowLogger instance (may be ``None``).
        path: File or directory path.
        is_dir: If ``True``, log all files in the directory.
    """
    if mlflow_logger is None or not path:
        return

    run_id = getattr(mlflow_logger, "run_id", None)
    client = getattr(mlflow_logger, "experiment", None)
    if run_id is None or client is None:
        return

    path_obj = Path(path)
    if not path_obj.exists():
        return
    try:
        if is_dir:
            client.log_artifacts(run_id, str(path_obj))
        else:
            client.log_artifact(run_id, str(path_obj))
    except Exception as exc:
        logger.warning("Failed to log MLflow artifact {}: {}", path_obj, exc)


def _serialise(obj):
    """Make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, float):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
#  Single-fold entry point (designed for subprocess)
# ---------------------------------------------------------------------------


def train_single_fold_classification(
    fold_id: int,
    gpu_id: int,
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    transformer_checkpoint: str,
    parent_run_id: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Train one fold of the transformer classifier.

    Designed to run in a subprocess launched by
    :func:`run_kfold_classification_parallel`.  Sets
    ``CUDA_VISIBLE_DEVICES`` before any PyTorch CUDA initialisation.

    Note:
        No evaluation is run — that will be implemented separately.

    Args:
        fold_id: Fold number (1-10).
        gpu_id: GPU device ID.
        base_config_path: Path to ``config_classification.yaml``.
        kfold_base_path: Path to the ``k_fold_dataset/`` directory.
        output_base_dir: Root output directory.
        transformer_checkpoint: Path to the transformer checkpoint.
        parent_run_id: Optional MLflow parent run ID.
        **kwargs: Additional config overrides.

    Returns:
        Dict with fold results.
    """
    logger.info(
        "Starting classification fold {} on GPU {} (pid={})",
        fold_id, gpu_id, os.getpid(),
    )

    # Set CUDA_VISIBLE_DEVICES early.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Load config.
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override paths in config.
        config.setdefault("dataset_config", {})[
            "kfold_base_path"
        ] = kfold_base_path
        config.setdefault("model_config", {})[
            "transformer_checkpoint"
        ] = transformer_checkpoint

        fold_output_dir = Path(output_base_dir) / f"fold_{fold_id}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        config["general_config"]["folders_config"]["out_dir_base"] = str(
            output_base_dir
        )

        # Inject per-fold MLflow tags.
        _inject_mlflow_fold_tags(
            config,
            fold_id=fold_id,
            gpu_id=gpu_id,
            parent_run_id=parent_run_id,
        )

        # Apply additional overrides from kwargs.
        for key, value in kwargs.items():
            if "." in key:
                parts = key.split(".")
                cfg = config
                for part in parts[:-1]:
                    cfg = cfg.setdefault(part, {})
                cfg[parts[-1]] = value

        # ----- Training ---------------------------------------------------
        from model.transformer.classification.classification_trainer import (
            train_fold,
        )

        start_time = time.time()
        checkpoint_dir, trainer_instance = train_fold(
            fold_id=fold_id,
            config=config,
            gpu_id=gpu_id,
        )
        training_time_min = (time.time() - start_time) / 60.0

        # Grab MLflow logger and training results.
        graph_model = trainer_instance
        mlflow_logger = getattr(graph_model, "mlflow_logger", None)
        best_ckpt_path = getattr(
            getattr(graph_model, "checkpoint_callback", None),
            "best_model_path", None,
        )

        # Read training results.
        _train_results_path = fold_output_dir / "fold_results.json"
        _train_results: Dict = {}
        if _train_results_path.exists():
            try:
                with open(_train_results_path) as _f:
                    _train_results = json.load(_f)
            except Exception:
                pass
        best_val_loss = _train_results.get("best_val_loss_training")
        best_val_accuracy = _train_results.get("best_val_accuracy_training")

        # Build results dict.
        fold_results = {
            "fold_id": fold_id,
            "gpu_id": gpu_id,
            "status": "success",
            "training_time_minutes": training_time_min,
            "best_val_loss_training": best_val_loss,
            "best_val_accuracy_training": best_val_accuracy,
            "checkpoint_dir": checkpoint_dir,
        }

        # ----- Post-training evaluation ----------------------------------
        eval_cfg = config.get("model_config", {}).get("evaluation", {})
        run_eval = eval_cfg.get("run_after_training", True)

        if run_eval:
            try:
                from model.transformer.classification.evaluate_transformer_classifier import (
                    evaluate_single_fold_transformer,
                )

                logger.info("Fold {}: Running post-training evaluation...",
                            fold_id)
                eval_results = evaluate_single_fold_transformer(
                    fold_dir=str(fold_output_dir),
                    config=config,
                    device="cuda:0",
                    target_fpr=float(eval_cfg.get("target_fpr", 0.2)),
                    exclude_last_minutes=float(
                        eval_cfg.get("exclude_last_minutes", 30.0)
                    ),
                    decision_time_hours=float(
                        eval_cfg.get("decision_time_hours", 1.0)
                    ),
                    max_gap_multiplier=eval_cfg.get("max_gap_multiplier"),
                    regenerate_predictions=True,
                )
                fold_results.update({
                    k: v for k, v in eval_results.items()
                    if k != "fold_id" and k != "status"
                })
                fold_results["evaluation_status"] = "success"
                logger.info(
                    "Fold {}: Evaluation complete — threshold={:.4f}, "
                    "ROC AUC={:.3f}",
                    fold_id,
                    eval_results.get("primary_threshold", 0.0),
                    eval_results.get("roc_auc", 0.0),
                )
            except Exception as eval_exc:
                logger.warning(
                    "Fold {} evaluation failed: {}", fold_id, eval_exc,
                )
                fold_results["evaluation_status"] = "failed"
                fold_results["evaluation_error"] = str(eval_exc)

        # Save results.
        results_path = fold_output_dir / "fold_results.json"
        with open(results_path, "w") as f:
            json.dump(_serialise(fold_results), f, indent=2)

        logger.info(
            "Fold {} completed — training {:.1f} min, "
            "best val_loss={}, best val_acc={}",
            fold_id, training_time_min,
            best_val_loss, best_val_accuracy,
        )

        # ----- Per-fold MLflow logging ------------------------------------
        fold_mlflow_metrics = {
            "train/training_time_minutes": training_time_min,
            "train/best_val_loss": best_val_loss,
            "train/best_val_accuracy": best_val_accuracy,
        }
        lightning_trainer = getattr(graph_model, "_trainer", None)
        _log_fold_mlflow_metrics(
            mlflow_logger, fold_mlflow_metrics,
            step=(
                getattr(lightning_trainer, "global_step", None)
                if lightning_trainer else None
            ),
        )

        # Log artifacts.
        fold_config_path = fold_output_dir / "config.yaml"
        if fold_config_path.exists():
            _log_fold_mlflow_artifact(mlflow_logger, fold_config_path)
        _log_fold_mlflow_artifact(mlflow_logger, results_path)
        if best_ckpt_path:
            _log_fold_mlflow_artifact(mlflow_logger, best_ckpt_path)

        # Cleanup.
        del trainer_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return fold_results

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


def run_kfold_classification_parallel(
    num_folds: int,
    gpu_ids: List[int],
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    transformer_checkpoint: str,
    max_parallel: Optional[int] = None,
    fold_ids: Optional[List[int]] = None,
    sequential: bool = False,
    fold_timeout_hours: float = 6.0,
    **kwargs,
) -> List[Dict]:
    """Run k-fold classification training in parallel.

    Uses :class:`ProcessPoolExecutor` with ``'spawn'`` context for CUDA
    safety.  GPU assignment is round-robin.

    Args:
        num_folds: Total number of folds in the dataset.
        gpu_ids: List of GPU device IDs.
        base_config_path: Path to ``config_classification.yaml``.
        kfold_base_path: Path to ``k_fold_dataset/`` directory.
        output_base_dir: Root output directory.
        transformer_checkpoint: Path to transformer checkpoint.
        max_parallel: Maximum simultaneous fold processes.
        fold_ids: Optional subset of fold IDs.
        sequential: If ``True``, run folds sequentially.
        fold_timeout_hours: Max hours per fold before timeout.
        **kwargs: Additional config overrides.

    Returns:
        List of per-fold result dicts, sorted by ``fold_id``.
    """
    # --- Resolve fold selection ------------------------------------------- #
    if fold_ids is None:
        selected_folds = list(range(1, num_folds + 1))
    else:
        selected_folds = sorted({int(fid) for fid in fold_ids})
        invalid = [
            fid for fid in selected_folds
            if fid < 1 or fid > num_folds
        ]
        if invalid:
            raise ValueError(
                f"Invalid fold IDs: {invalid} (valid range 1..{num_folds})"
            )
    if not selected_folds:
        raise ValueError("No folds requested for execution.")

    requested_count = len(selected_folds)

    if max_parallel is None:
        max_parallel = min(len(gpu_ids), requested_count)

    logger.info(
        "K-fold classification: {} folds on {} GPUs "
        "(max_parallel={})",
        requested_count, len(gpu_ids), max_parallel,
    )

    # --- MLflow parent run ------------------------------------------------ #
    with open(base_config_path, "r") as f:
        config_data = yaml.safe_load(f)

    client, parent_run_id = create_parent_mlflow_run(
        config_data, "transformer-cls-kfold"
    )

    # --- Output directory ------------------------------------------------- #
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Execute folds ---------------------------------------------------- #
    results: List[Dict] = []

    if sequential:
        for fold_id in selected_folds:
            gpu_id = gpu_ids[0]
            result = train_single_fold_classification(
                fold_id=fold_id,
                gpu_id=gpu_id,
                base_config_path=base_config_path,
                kfold_base_path=kfold_base_path,
                output_base_dir=output_base_dir,
                transformer_checkpoint=transformer_checkpoint,
                parent_run_id=parent_run_id,
                **kwargs,
            )
            results.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=max_parallel,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            futures = {}
            for job_idx, fold_id in enumerate(selected_folds):
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]
                future = executor.submit(
                    train_single_fold_classification,
                    fold_id=fold_id,
                    gpu_id=gpu_id,
                    base_config_path=base_config_path,
                    kfold_base_path=kfold_base_path,
                    output_base_dir=output_base_dir,
                    transformer_checkpoint=transformer_checkpoint,
                    parent_run_id=parent_run_id,
                    **kwargs,
                )
                futures[future] = fold_id

            try:
                for future in as_completed(
                    futures,
                    timeout=fold_timeout_hours * 3600,
                ):
                    fold_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        logger.exception("Fold {} failed:", fold_id)
                        results.append({
                            "fold_id": fold_id,
                            "status": "failed",
                            "error": str(exc),
                        })
            except TimeoutError:
                timed_out_folds = [
                    fid for fut, fid in futures.items()
                    if not fut.done()
                ]
                logger.error(
                    "Parallel execution timed out after {:.1f}h. "
                    "Timed-out folds: {}",
                    fold_timeout_hours, timed_out_folds,
                )
                for future, fid in futures.items():
                    if not future.done():
                        future.cancel()
                        results.append({
                            "fold_id": fid,
                            "status": "failed",
                            "error": "Timed out",
                        })

    # --- Sort and summarise ----------------------------------------------- #
    results.sort(key=lambda r: r.get("fold_id", 0))

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    logger.info(
        "K-fold complete: {}/{} successful, {}/{} failed",
        len(successful), requested_count,
        len(failed), requested_count,
    )

    if successful:
        val_losses = [
            r["best_val_loss_training"]
            for r in successful
            if r.get("best_val_loss_training") is not None
        ]
        val_accs = [
            r["best_val_accuracy_training"]
            for r in successful
            if r.get("best_val_accuracy_training") is not None
        ]
        if val_losses:
            import numpy as np
            logger.info(
                "Cross-fold val_loss: mean={:.4f} ± {:.4f}",
                np.mean(val_losses), np.std(val_losses),
            )
        if val_accs:
            import numpy as np
            logger.info(
                "Cross-fold val_accuracy: mean={:.4f} ± {:.4f}",
                np.mean(val_accs), np.std(val_accs),
            )

    # --- Cross-fold evaluation aggregation -------------------------------- #
    if successful:
        try:
            from model.transformer.classification.evaluate_transformer_classifier import (
                aggregate_transformer_results,
            )
            aggregated = aggregate_transformer_results(
                output_base_dir=output_base_dir,
                fold_results=successful,
            )
            logger.info(
                "Cross-fold aggregation complete — "
                "ROC AUC={:.3f}±{:.3f}",
                aggregated.get("roc_auc_mean", 0),
                aggregated.get("roc_auc_std", 0),
            )
        except Exception as agg_exc:
            logger.warning(
                "Cross-fold evaluation aggregation failed: {}", agg_exc,
            )

    # --- Save summary ----------------------------------------------------- #
    summary_path = Path(output_base_dir) / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_serialise(results), f, indent=2)
    logger.info("Summary saved to: {}", summary_path)

    return results


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: run k-fold classification training."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config_classification.yaml",
    )

    logger.info("Loading config from: {}", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    general_cfg = config.get("general_config", {})
    ds_cfg = config.get("dataset_config", {})
    model_cfg = config.get("model_config", {})

    gpu_ids = general_cfg.get("cuda_devices", [0])
    max_parallel = general_cfg.get("max_parallel_folds", len(gpu_ids))
    num_folds = ds_cfg.get("num_folds", 10)
    fold_ids = ds_cfg.get("fold_ids", None)
    kfold_base_path = ds_cfg.get("kfold_base_path", "")
    transformer_checkpoint = model_cfg.get("transformer_checkpoint", "")

    output_base_dir = (
        general_cfg.get("folders_config", {}).get(
            "out_dir_base", "./classification_results"
        )
    )

    run_kfold_classification_parallel(
        num_folds=num_folds,
        gpu_ids=gpu_ids,
        base_config_path=config_path,
        kfold_base_path=kfold_base_path,
        output_base_dir=output_base_dir,
        transformer_checkpoint=transformer_checkpoint,
        max_parallel=max_parallel,
        fold_ids=fold_ids,
    )


if __name__ == "__main__":
    main()
