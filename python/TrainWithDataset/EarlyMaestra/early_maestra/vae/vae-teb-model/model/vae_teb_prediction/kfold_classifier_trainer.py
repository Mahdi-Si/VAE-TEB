"""
K-Fold Cross-Validation Trainer for VAE + Classifier

This script runs multiple folds in parallel across available GPUs.
Each fold trains independently with its own train/val/test splits.

Label mapping:
- healthy → target value 1 → binary class 0
- acidosis → target value 2 → binary class 1
- HIE → target value 3 → binary class 1
"""

import os
import yaml
import torch
import time
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import pandas as pd
import h5py

from model.vae_teb_prediction.evaluate_classifier import generate_aggregated_plots


def get_fold_datasets(base_path: str, fold_id: int) -> Dict[str, List[str]]:
    """Get train, val, and test dataset paths for a specific fold.

    Supports two directory layouts:

    * **Per-fold test** (augmented / legacy): ``fold_N/test/`` contains HDF5
      files specific to that fold.
    * **Shared test** (holdout): ``base_path/test/`` is a single test
      directory shared across all folds (``fold_N/`` only has train + val).

    The function auto-detects the layout: it checks the fold-level test
    directory first, then falls back to the shared test directory.

    Args:
        base_path: Base path to k-fold dataset directory.
        fold_id: Fold number (1-10).

    Returns:
        Dictionary with ``'train'``, ``'val'``, ``'test'`` keys containing
        lists of HDF5 file paths.
    """
    fold_dir = Path(base_path) / f"fold_{fold_id}"

    datasets: Dict[str, List[str]] = {
        'train': [],
        'val': [],
        'test': []
    }

    for split in ['train', 'val']:
        split_dir = fold_dir / split
        if split_dir.exists():
            hdf5_files = sorted(split_dir.glob("*.hdf5"))
            datasets[split] = [str(f) for f in hdf5_files]
            logger.info(f"Fold {fold_id} {split}: found {len(datasets[split])} files")
        else:
            logger.warning(f"Split directory not found: {split_dir}")

    # Test: check fold-level first, then shared test directory
    test_dir_fold = fold_dir / "test"
    test_dir_shared = Path(base_path) / "test"

    if test_dir_fold.exists() and list(test_dir_fold.glob("*.hdf5")):
        hdf5_files = sorted(test_dir_fold.glob("*.hdf5"))
        datasets["test"] = [str(f) for f in hdf5_files]
        logger.info(f"Fold {fold_id} test (per-fold): found {len(datasets['test'])} files")
    elif test_dir_shared.exists() and list(test_dir_shared.glob("*.hdf5")):
        hdf5_files = sorted(test_dir_shared.glob("*.hdf5"))
        datasets["test"] = [str(f) for f in hdf5_files]
        logger.info(f"Fold {fold_id} test (shared): found {len(datasets['test'])} files")
    else:
        logger.warning(
            f"No test directory found for fold {fold_id}: "
            f"checked {test_dir_fold} and {test_dir_shared}"
        )

    # Validate we have data for all splits
    for split in ['train', 'val', 'test']:
        if not datasets[split]:
            raise ValueError(f"Fold {fold_id} has no {split} files! Check dataset structure at {fold_dir}")

    return datasets


def create_parent_mlflow_run(
    config_data: Optional[Dict],
    experiment_tag: str,
    extra_tags: Optional[Dict[str, str]] = None,
):
    """
    Create a parent MLflow run for a k-fold execution.

    Returns:
        Tuple of (mlflow.MlflowClient, parent_run_id) or (None, None) if disabled.
    """
    if not config_data:
        return None, None

    mlflow_cfg = (
        config_data.get('advanced_config', {})
        .get('tracking', {})
        .get('mlflow', {})
        or {}
    )
    if not mlflow_cfg.get('enabled'):
        return None, None

    try:
        import mlflow

        tracking_uri = mlflow_cfg.get('tracking_uri')
        experiment_name = mlflow_cfg.get('experiment_name') or experiment_tag

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.MlflowClient()

        # Get or create experiment (restore if soft-deleted)
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

        # Build tags for the parent run
        tags = dict(mlflow_cfg.get('tags') or {})
        tags['kfold.role'] = 'parent'
        if extra_tags:
            tags.update({k: str(v) for k, v in extra_tags.items()})

        run_name = f"{experiment_tag}-{time.strftime('%Y%m%d-%H%M%S')}"
        parent_run = client.create_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tags,
        )

        parent_run_id = parent_run.info.run_id
        logger.info(f"Created parent MLflow run: {run_name} (id={parent_run_id})")
        return client, parent_run_id

    except Exception as e:
        logger.warning(f"Failed to create parent MLflow run: {e}")
        return None, None


def estimate_class_weights(hdf5_files: List[str], chunk_size: int = 512) -> Optional[List[float]]:
    """
    Estimate class weights (binary) by scanning the HDF5 target datasets.
    """
    if not hdf5_files:
        return None

    counts = np.zeros(2, dtype=np.int64)

    for path in hdf5_files:
        h5_path = Path(path)
        if not h5_path.exists():
            logger.warning(f"Training dataset missing for class weight estimation: {path}")
            continue
        with h5py.File(h5_path, "r") as h5f:
            if "target" not in h5f:
                logger.warning(f"'target' dataset missing in {path}")
                continue
            target_ds = h5f["target"]
            total_samples = target_ds.shape[0]
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                targets = target_ds[start:end]
                labels = targets.max(axis=1)
                counts[0] += np.sum(labels <= 1)
                counts[1] += np.sum(labels > 1)

    total = counts.sum()
    if total == 0:
        return None
    weights = []
    num_classes = len(counts)
    for class_count in counts:
        if class_count == 0:
            weights.append(1.0)
        else:
            weights.append(float(total / (num_classes * class_count)))
    logger.info(f"Estimated class counts (healthy, unhealthy): {counts.tolist()}, weights: {weights}")
    return weights


def train_single_fold(
    fold_id: int,
    gpu_id: int,
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
    parent_run_id: Optional[str] = None,
    **kwargs
) -> Dict[str, any]:
    """
    Train a single fold on a specific GPU.

    Args:
        fold_id: Fold number (1-10)
        gpu_id: GPU device ID to use
        base_config_path: Path to base config.yaml
        kfold_base_path: Base path to k-fold dataset directory
        output_base_dir: Base directory for saving results
        vae_checkpoint: Path to pre-trained VAE checkpoint
        parent_run_id: Optional parent MLflow run ID for nested run grouping
        **kwargs: Additional config overrides

    Returns:
        Dictionary with fold results (accuracy, loss, etc.)
    """
    logger.info(f"Starting fold {fold_id} on GPU {gpu_id}")

    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    fold_datasets = get_fold_datasets(kfold_base_path, fold_id)

    config['dataset_config']['classifier_train_datasets'] = fold_datasets['train']
    config['dataset_config']['classifier_val_datasets'] = fold_datasets['val']
    config['dataset_config']['classifier_test_datasets'] = fold_datasets['test']

    config['general_config']['cuda_devices'] = [0]

    advanced_cfg = config.setdefault('advanced_config', {})
    tracking_cfg = advanced_cfg.setdefault('tracking', {})
    mlflow_cfg = tracking_cfg.setdefault('mlflow', {})
    mlflow_tags = mlflow_cfg.get('tags') or {}
    mlflow_tags.update({
        'kfold.role': 'fold',
        'kfold.fold_id': str(fold_id),
        'kfold.gpu_id': str(gpu_id),
        'kfold.train_files': str(len(fold_datasets['train'])),
        'kfold.val_files': str(len(fold_datasets['val'])),
        'kfold.test_files': str(len(fold_datasets['test'])),
        'kfold.base_path': str(kfold_base_path),
    })
    if parent_run_id:
        mlflow_tags['mlflow.parentRunId'] = parent_run_id
    mlflow_cfg['tags'] = mlflow_tags
    if not mlflow_cfg.get('run_name'):
        mlflow_cfg['run_name'] = f"fold-{fold_id:02d}"

    fold_output_dir = Path(output_base_dir) / f"fold_{fold_id}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    config['general_config']['folders_config']['out_dir_base'] = str(fold_output_dir)

    if 'model_config' not in config:
        config['model_config'] = {}
    if 'classifier' not in config['model_config']:
        config['model_config']['classifier'] = {}
    config['model_config']['classifier']['vae_checkpoint'] = vae_checkpoint
    class_weights = estimate_class_weights(fold_datasets['train'])
    if class_weights is not None:
        config['model_config']['classifier']['class_weights'] = class_weights
    else:
        config['model_config']['classifier'].pop('class_weights', None)

    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested keys like 'general_config.lr'
            parts = key.split('.')
            cfg = config
            for part in parts[:-1]:
                cfg = cfg.setdefault(part, {})
            cfg[parts[-1]] = value
        else:
            config[key] = value

    fold_config_path = fold_output_dir / "config.yaml"
    with open(fold_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Fold {fold_id} config saved to {fold_config_path}")

    from model.vae_teb_prediction.classifier_trainer import GraphModelClassifierTrainer
    from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
    import numpy as np
    import random

    # Set random seeds for reproducibility
    random.seed(42 + fold_id)
    np.random.seed(42 + fold_id)
    torch.manual_seed(42 + fold_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + fold_id)
    # Keep benchmark mode for performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        dataset_config = config.get('dataset_config')
        dataloader_config = dataset_config.get('dataloader_config')
        dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
        normalized_fields = dataloader_config.get('normalize_fields')
        stat_path = dataset_config.get('stat_path')

        logger.info(f"Fold {fold_id}: Creating dataloaders...")

        train_dataloader = create_optimized_dataloader(
            hdf5_files=fold_datasets['train'],
            batch_size=config['general_config']['batch_size']['train'],
            num_workers=dataloader_config.get('num_workers', 4),
            shuffle=True,
            stats_path=stat_path,
            normalize_fields=normalized_fields,
            prefetch_factor=dataloader_config.get('prefetch_factor', 2),
            pin_memory=True,
            rank=0,
            world_size=1,
            **dataset_kwargs
        )

        val_dataloader = create_optimized_dataloader(
            hdf5_files=fold_datasets['val'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=dataloader_config.get('num_workers', 4),
            shuffle=False,
            stats_path=stat_path,
            normalize_fields=normalized_fields,
            prefetch_factor=dataloader_config.get('prefetch_factor', 2),
            pin_memory=True,
            rank=0,
            world_size=1,
            **dataset_kwargs
        )

        logger.info(f"Fold {fold_id}: Creating model...")
        graph_model = GraphModelClassifierTrainer(config_file_path=str(fold_config_path))
        graph_model.setup_config()
        graph_model.create_model()

        logger.info(f"Fold {fold_id}: Starting training...")
        start_time = time.time()
        trainer = graph_model.train_model(train_dataloader, val_dataloader)
        training_time = (time.time() - start_time) / 60

        logger.info(f"Fold {fold_id}: Training completed in {training_time:.2f} minutes")

        best_val_acc = trainer.callback_metrics.get('val/accuracy', 0.0)
        best_val_loss = trainer.callback_metrics.get('val/loss', float('inf'))

        best_ckpt_path = graph_model.checkpoint_callback.best_model_path
        logger.info(f"Fold {fold_id}: Best checkpoint path: {best_ckpt_path}")

        mlflow_logger = getattr(graph_model, "mlflow_logger", None)

        def _log_mlflow_metrics(metrics: Dict[str, float], step: Optional[int] = None):
            """Log metrics using MlflowClient directly (works on terminated runs)."""
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

        def _log_mlflow_artifact(path: Path | str, *, is_dir: bool = False):
            """Log artifacts using MlflowClient directly (works on terminated runs)."""
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
                logger.warning("Failed to log MLflow artifact '{}': {}", path_obj, exc)

# NEW Clean Evaluation Implementation for kfold_classifier_trainer.py
# Replace lines 332-493 with this code

        # --------------------------------------------------------------------
        # POST-TRAINING EVALUATION - NEW Implementation
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Starting post-training evaluation...")

        from model.vae_teb_prediction.evaluate_classifier import (
            run_inference,
            find_threshold_for_committed_overall_fpr_at_1h,
            find_threshold_for_committed_cumulative_fpr_at_1h,
            find_threshold_for_instantaneous_fpr_at_1h,
            apply_clinical_decision_rule,
            fill_missing_epochs,
            generate_three_metric_type_analysis,
            compute_guid_level_roc,
            plot_roc_curve,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        evaluation_dir = fold_output_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Get evaluation config (from config file, with kwargs override)
        eval_cfg = config.get('model_config', {}).get('classifier', {}).get('evaluation', {}) or {}
        target_fpr = float(kwargs.get('target_fpr', eval_cfg.get('target_fpr', 0.15)))
        exclude_last_minutes = float(eval_cfg.get('exclude_last_minutes', 30.0))
        max_gap_multiplier = eval_cfg.get('max_gap_multiplier', None)
        decision_time_hours = float(eval_cfg.get('decision_time_hours', 1.0))

        # --------------------------------------------------------------------
        # VALIDATION SET INFERENCE
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Running validation inference...")
        val_df_raw = run_inference(graph_model.pytorch_model, val_dataloader, device)
        val_df_raw.to_csv(evaluation_dir / "validation_predictions_raw.csv", index=False)
        logger.info(f"Fold {fold_id}: Validation predictions saved ({len(val_df_raw)} rows)")

        # --------------------------------------------------------------------
        # FIND THRESHOLDS (one per metric type)
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Finding thresholds (target_fpr={target_fpr}, time={decision_time_hours}h)...")

        # PRIMARY — committed_overall
        primary_threshold, threshold_metrics = find_threshold_for_committed_overall_fpr_at_1h(
            val_df_raw,
            target_fpr=target_fpr,
            time_window_hours=decision_time_hours,
            fallback_tolerance_hours=0.5,
        )

        # Committed cumulative
        threshold_cumulative, _ = find_threshold_for_committed_cumulative_fpr_at_1h(
            val_df_raw,
            target_fpr=target_fpr,
            time_window_hours=decision_time_hours,
            fallback_tolerance_hours=0.5,
        )

        # Instantaneous
        threshold_instantaneous, _ = find_threshold_for_instantaneous_fpr_at_1h(
            val_df_raw,
            target_fpr=target_fpr,
            time_window_hours=decision_time_hours,
            fallback_tolerance_hours=0.5,
        )

        all_thresholds = {
            'instantaneous': threshold_instantaneous,
            'committed_cumulative': threshold_cumulative,
            'committed_overall': primary_threshold,
        }

        logger.info(f"Fold {fold_id}: Thresholds — overall={primary_threshold:.4f}, "
                     f"cumulative={threshold_cumulative:.4f}, "
                     f"instantaneous={threshold_instantaneous:.4f}")

        # Apply clinical decision rule to validation set (primary threshold)
        val_df_clinical = apply_clinical_decision_rule(val_df_raw.copy(), primary_threshold)
        val_df_clinical = fill_missing_epochs(val_df_clinical, max_gap_multiplier=max_gap_multiplier)
        val_df_clinical.to_csv(evaluation_dir / "validation_predictions_clinical.csv", index=False)

        # --------------------------------------------------------------------
        # TEST SET INFERENCE
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Running test inference...")

        # Create test dataloader
        test_dataloader = create_optimized_dataloader(
            hdf5_files=fold_datasets['test'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=dataloader_config.get('num_workers', 4),
            shuffle=False,
            stats_path=stat_path,
            normalize_fields=normalized_fields,
            prefetch_factor=dataloader_config.get('prefetch_factor', 2),
            pin_memory=True,
            rank=0,
            world_size=1,
            **dataset_kwargs
        )

        test_df_raw = run_inference(graph_model.pytorch_model, test_dataloader, device)
        test_df_raw.to_csv(evaluation_dir / "test_predictions_raw.csv", index=False)
        logger.info(f"Fold {fold_id}: Test predictions saved ({len(test_df_raw)} rows)")

        # Save CDR'd test predictions using primary (committed_overall) threshold
        test_df_clinical = apply_clinical_decision_rule(test_df_raw.copy(), primary_threshold)
        test_df_clinical = fill_missing_epochs(test_df_clinical, max_gap_multiplier=max_gap_multiplier)
        test_df_clinical.to_csv(evaluation_dir / "test_predictions_clinical.csv", index=False)

        # --------------------------------------------------------------------
        # GENERATE THREE METRIC TYPE ANALYSIS (separate thresholds)
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Generating three metric type analysis...")
        try:
            three_metric_results = generate_three_metric_type_analysis(
                test_df_raw,
                thresholds=all_thresholds,
                output_base_dir=evaluation_dir,
                exclude_last_minutes=exclude_last_minutes,
                title_suffix=f"Fold {fold_id}",
                max_gap_multiplier=max_gap_multiplier,
                decision_time_hours=decision_time_hours,
            )
            logger.info(f"Fold {fold_id}: Three metric type analysis complete")

            # Log the three metric type directory to MLflow
            three_metric_dir = evaluation_dir / "three_metric_types"
            if three_metric_dir.exists():
                _log_mlflow_artifact(three_metric_dir, is_dir=True)

        except Exception as e:
            logger.warning(f"Fold {fold_id}: Three metric type analysis failed: {e}")
            three_metric_results = {}

        # --------------------------------------------------------------------
        # COMPUTE TEST METRICS
        # --------------------------------------------------------------------
        # Extract PRIMARY metric (committed_overall, mean across bins)
        primary_metrics = {}
        if three_metric_results and 'summary' in three_metric_results:
            primary_summary = three_metric_results['summary'].get('metric_types', {}).get('committed_overall', {})
            primary_metrics = {
                'test_sensitivity_mean': primary_summary.get('sensitivity_mean', 0.0),
                'test_sensitivity_std': primary_summary.get('sensitivity_std', 0.0),
                'test_specificity_mean': primary_summary.get('specificity_mean', 0.0),
                'test_specificity_std': primary_summary.get('specificity_std', 0.0),
                'test_fpr_mean': primary_summary.get('fpr_mean', 0.0),
                'test_fpr_std': primary_summary.get('fpr_std', 0.0),
            }

        # Extract decision-point metrics for all three metric types
        decision_point = three_metric_results.get("decision_point_metrics", {}) if three_metric_results else {}

        # --- ROC curve ------------------------------------------------------
        roc_data = {}
        try:
            roc_data = compute_guid_level_roc(test_df_raw, decision_time_hours=decision_time_hours)
            if roc_data:
                plot_roc_curve(
                    roc_data,
                    evaluation_dir / "roc_curve.png",
                    title_suffix=f"Fold {fold_id}",
                    threshold=primary_threshold,
                )
                roc_csv = pd.DataFrame({'fpr': roc_data['fpr'], 'tpr': roc_data['tpr']})
                roc_csv.to_csv(evaluation_dir / "roc_data.csv", index=False)
                logger.info(f"Fold {fold_id}: ROC AUC = {roc_data['auc']:.4f}")
        except Exception as e:
            logger.warning(f"Fold {fold_id}: ROC computation failed: {e}")

        # Save threshold and metrics info
        threshold_info = {
            'primary_threshold': float(primary_threshold),
            'all_thresholds': {k: float(v) for k, v in all_thresholds.items()},
            'target_fpr': float(target_fpr),
            'validation_metrics': {
                'sensitivity': float(threshold_metrics.get('sensitivity', 0)),
                'specificity': float(threshold_metrics.get('specificity', 0)),
                'fpr': float(threshold_metrics.get('fpr', 0)),
                'accuracy': float(threshold_metrics.get('accuracy', 0)),
                'time_window_hours': float(threshold_metrics.get('time_window_hours', 1.0)),
            },
            'test_metrics_primary': primary_metrics,
            'decision_point_metrics': decision_point,
            'roc_auc': roc_data.get('auc'),
        }

        # Save threshold info
        with open(evaluation_dir / "threshold_info.json", 'w') as f:
            json.dump(threshold_info, f, indent=2)

        logger.info(f"Fold {fold_id}: Evaluation completed")

        # --------------------------------------------------------------------
        # RESULTS DICTIONARY
        # --------------------------------------------------------------------
        results = {
            'fold_id': fold_id,
            'gpu_id': gpu_id,
            'training_time_minutes': training_time,
            'best_val_accuracy_training': float(best_val_acc),
            'best_val_loss_training': float(best_val_loss),

            # PRIMARY threshold and validation metrics
            'primary_threshold': float(primary_threshold),
            'all_thresholds': {k: float(v) for k, v in all_thresholds.items()},
            'validation_sensitivity': float(threshold_metrics.get('sensitivity', 0)),
            'validation_specificity': float(threshold_metrics.get('specificity', 0)),
            'validation_fpr': float(threshold_metrics.get('fpr', 0)),
            'validation_accuracy': float(threshold_metrics.get('accuracy', 0)),

            # Test metrics (PRIMARY - committed_overall, mean across bins)
            'test_sensitivity_mean': primary_metrics.get('test_sensitivity_mean', 0.0),
            'test_sensitivity_std': primary_metrics.get('test_sensitivity_std', 0.0),
            'test_specificity_mean': primary_metrics.get('test_specificity_mean', 0.0),
            'test_specificity_std': primary_metrics.get('test_specificity_std', 0.0),
            'test_fpr_mean': primary_metrics.get('test_fpr_mean', 0.0),
            'test_fpr_std': primary_metrics.get('test_fpr_std', 0.0),

            # Decision-point metrics (FPR at 1h for each metric type)
            'decision_point_metrics': decision_point,

            # ROC
            'roc_auc': roc_data.get('auc'),
            'roc_data': {
                'fpr': roc_data.get('fpr', []),
                'tpr': roc_data.get('tpr', []),
            } if roc_data else {},

            'status': 'success'
        }

        results_path = fold_output_dir / "fold_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Log summary
        logger.info(f"Fold {fold_id} summary:")
        logger.info(f"  Training val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")
        logger.info(f"  PRIMARY threshold: {results['primary_threshold']:.4f}")
        logger.info(f"  Validation sensitivity: {results['validation_sensitivity']:.3f}, specificity: {results['validation_specificity']:.3f}")
        logger.info(f"  Test sensitivity: {results['test_sensitivity_mean']:.3f} ± {results['test_sensitivity_std']:.3f}")
        logger.info(f"  Test specificity: {results['test_specificity_mean']:.3f} ± {results['test_specificity_std']:.3f}")
        logger.info(f"  Test FPR: {results['test_fpr_mean']:.3f} ± {results['test_fpr_std']:.3f}")
        if roc_data.get('auc') is not None:
            logger.info(f"  ROC AUC: {roc_data['auc']:.4f}")

        # Log metrics to MLflow
        mlflow_metrics = {
            "train/training_time_minutes": training_time,
            "val/best_accuracy": best_val_acc,
            "val/best_loss": best_val_loss,
            "eval/primary_threshold": results['primary_threshold'],
            "eval/validation_sensitivity": results['validation_sensitivity'],
            "eval/validation_specificity": results['validation_specificity'],
            "eval/validation_fpr": results['validation_fpr'],
            "eval/test_sensitivity_mean": results['test_sensitivity_mean'],
            "eval/test_specificity_mean": results['test_specificity_mean'],
            "eval/test_fpr_mean": results['test_fpr_mean'],
        }
        if roc_data.get('auc') is not None:
            mlflow_metrics["eval/roc_auc"] = roc_data['auc']
        _log_mlflow_metrics(mlflow_metrics, step=getattr(trainer, "global_step", None))

        # Log artifacts to MLflow
        _log_mlflow_artifact(fold_config_path)
        _log_mlflow_artifact(results_path)
        if best_ckpt_path:
            _log_mlflow_artifact(best_ckpt_path)
        if evaluation_dir.exists():
            _log_mlflow_artifact(evaluation_dir, is_dir=True)

        # Explicit cleanup to prevent memory accumulation
        try:
            del train_dataloader, val_dataloader, test_dataloader
            del graph_model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"Fold {fold_id}: Memory cleanup completed")
        except Exception as e_cleanup:
            logger.warning(f"Fold {fold_id}: Cleanup warning: {e_cleanup}")

        # Explicit cleanup to prevent memory accumulation
        try:
            del train_dataloader, val_dataloader
            del graph_model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"Fold {fold_id}: Memory cleanup completed")
        except Exception as e_cleanup:
            logger.warning(f"Fold {fold_id}: Cleanup warning: {e_cleanup}")

        return results

    except Exception as e:
        import traceback
        logger.exception(f"Fold {fold_id} failed:")
        return {
            'fold_id': fold_id,
            'gpu_id': gpu_id,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_kfold_parallel(
    num_folds: int,
    gpu_ids: List[int],
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
    max_parallel: int = None,
    fold_ids: Optional[List[int]] = None,
    sequential: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Run k-fold cross-validation in parallel across multiple GPUs.

    Args:
        num_folds: Number of folds (e.g., 10)
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
        base_config_path: Path to base config.yaml
        kfold_base_path: Base path to k-fold dataset directory
        output_base_dir: Base directory for saving results
        vae_checkpoint: Path to pre-trained VAE checkpoint
        max_parallel: Maximum number of parallel processes (defaults to len(gpu_ids))
        fold_ids: Optional subset of folds to run (default: all 1..num_folds)
        sequential: If True, run folds sequentially on GPUs instead of spawning processes
        **kwargs: Additional config overrides

    Returns:
        List of result dictionaries, one per fold
    """
    if fold_ids is None:
        selected_folds = list(range(1, num_folds + 1))
    else:
        selected_folds = sorted({int(fid) for fid in fold_ids})
        invalid = [fid for fid in selected_folds if fid < 1 or fid > num_folds]
        if invalid:
            raise ValueError(f"Invalid fold IDs requested: {invalid} (valid range 1..{num_folds})")
    if not selected_folds:
        raise ValueError("No folds requested for execution.")

    requested_fold_count = len(selected_folds)

    if max_parallel is None:
        max_parallel = min(len(gpu_ids), requested_fold_count)
    else:
        max_parallel = max(1, min(max_parallel, requested_fold_count))

    base_cfg_data = None
    if base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, 'r') as cfg_file:
            base_cfg_data = yaml.safe_load(cfg_file)

    experiment_tag = (base_cfg_data or {}).get('general_config', {}).get('tag', 'classifier')

    execution_mode = "parallel" if (not sequential and max_parallel > 1) else "sequential"

    logger.info(f"Starting k-fold cross-validation (configured folds={num_folds})")
    logger.info(f"Requested folds: {selected_folds}")
    logger.info(f"Execution mode: {execution_mode}")
    logger.info(f"Using GPUs: {gpu_ids}")
    if execution_mode == "parallel":
        logger.info(f"Max parallel folds: {max_parallel}")

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    # Create parent MLflow run — all fold runs will be nested under it
    mlflow_client, parent_run_id = create_parent_mlflow_run(
        base_cfg_data,
        experiment_tag,
        extra_tags={
            'kfold.num_folds': str(num_folds),
            'kfold.requested_folds': str(selected_folds),
            'kfold.execution_mode': execution_mode,
        },
    )

    all_results = []

    if sequential or max_parallel == 1:
        logger.info("Running folds sequentially...")
        for job_idx, fold_id in enumerate(selected_folds):
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            result = train_single_fold(
                fold_id=fold_id,
                gpu_id=gpu_id,
                base_config_path=base_config_path,
                kfold_base_path=kfold_base_path,
                output_base_dir=output_base_dir,
                vae_checkpoint=vae_checkpoint,
                parent_run_id=parent_run_id,
                **kwargs
            )
            all_results.append(result)
            logger.info(f"Fold {fold_id} completed: {result}")
    else:
        logger.info("Running folds in parallel...")
        spawn_ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=max_parallel, mp_context=spawn_ctx) as executor:
            futures = {}
            for job_idx, fold_id in enumerate(selected_folds):
                gpu_id = gpu_ids[job_idx % len(gpu_ids)]

                future = executor.submit(
                    train_single_fold,
                    fold_id=fold_id,
                    gpu_id=gpu_id,
                    base_config_path=base_config_path,
                    kfold_base_path=kfold_base_path,
                    output_base_dir=output_base_dir,
                    vae_checkpoint=vae_checkpoint,
                    parent_run_id=parent_run_id,
                    **kwargs
                )
                futures[future] = fold_id

            for future in as_completed(futures):
                fold_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    logger.info(f"Fold {fold_id} completed: {result}")
                except Exception as e:
                    import traceback
                    logger.exception(f"Fold {fold_id} raised exception:")
                    all_results.append({
                        'fold_id': fold_id,
                        'status': 'failed',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })

    all_results.sort(key=lambda x: x['fold_id'])

    aggregated_results_path = Path(output_base_dir) / "kfold_results.json"
    with open(aggregated_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    successful_folds = [r for r in all_results if r['status'] == 'success']

    if successful_folds:
        n = len(successful_folds)

        # Training metrics
        avg_val_acc_train = sum(r['best_val_accuracy_training'] for r in successful_folds) / n
        std_val_acc_train = (sum((r['best_val_accuracy_training'] - avg_val_acc_train) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        # Test metrics (PRIMARY - committed_overall averaged across folds)
        avg_test_sens = sum(r.get('test_sensitivity_mean', 0.0) for r in successful_folds) / n
        std_test_sens = (sum((r.get('test_sensitivity_mean', 0.0) - avg_test_sens) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        avg_test_spec = sum(r.get('test_specificity_mean', 0.0) for r in successful_folds) / n
        std_test_spec = (sum((r.get('test_specificity_mean', 0.0) - avg_test_spec) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        avg_test_fpr = sum(r.get('test_fpr_mean', 0.0) for r in successful_folds) / n
        std_test_fpr = (sum((r.get('test_fpr_mean', 0.0) - avg_test_fpr) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        # Decision-point FPR aggregation per metric type
        decision_point_agg = {}
        for mt in ['instantaneous', 'committed_cumulative', 'committed_overall']:
            fpr_vals = [
                r.get('decision_point_metrics', {}).get(mt, {}).get('fpr_at_decision')
                for r in successful_folds
            ]
            fpr_vals = [v for v in fpr_vals if v is not None]
            sens_vals = [
                r.get('decision_point_metrics', {}).get(mt, {}).get('sensitivity_at_decision')
                for r in successful_folds
            ]
            sens_vals = [v for v in sens_vals if v is not None]
            if fpr_vals:
                arr = np.array(fpr_vals)
                decision_point_agg[f'fpr_at_decision_{mt}_mean'] = float(arr.mean())
                decision_point_agg[f'fpr_at_decision_{mt}_std'] = float(arr.std())
            if sens_vals:
                arr = np.array(sens_vals)
                decision_point_agg[f'sensitivity_at_decision_{mt}_mean'] = float(arr.mean())
                decision_point_agg[f'sensitivity_at_decision_{mt}_std'] = float(arr.std())

        # ROC AUC aggregation
        roc_aucs = [r.get('roc_auc') for r in successful_folds if r.get('roc_auc') is not None]
        roc_auc_agg = {}
        if roc_aucs:
            arr = np.array(roc_aucs)
            roc_auc_agg = {'roc_auc_mean': float(arr.mean()), 'roc_auc_std': float(arr.std())}

        summary = {
            'configured_num_folds': num_folds,
            'requested_folds': selected_folds,
            'successful_folds': len(successful_folds),
            'failed_folds': requested_fold_count - len(successful_folds),
            'training_metrics': {
                'mean_val_accuracy': avg_val_acc_train,
                'std_val_accuracy': std_val_acc_train,
            },
            'test_metrics_primary': {
                'mean_sensitivity': avg_test_sens,
                'std_sensitivity': std_test_sens,
                'mean_specificity': avg_test_spec,
                'std_specificity': std_test_spec,
                'mean_fpr': avg_test_fpr,
                'std_fpr': std_test_fpr,
            },
            'decision_point_metrics': decision_point_agg,
            **roc_auc_agg,
            'individual_results': all_results
        }

        summary_path = Path(output_base_dir) / "kfold_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Log summary metrics and artifacts to the parent MLflow run
        if mlflow_client and parent_run_id:
            try:
                summary_metrics = {
                    "summary/successful_folds": float(len(successful_folds)),
                    "summary/failed_folds": float(requested_fold_count - len(successful_folds)),
                    "summary/mean_val_accuracy": avg_val_acc_train,
                    "summary/std_val_accuracy": std_val_acc_train,
                    "summary/mean_test_sensitivity": avg_test_sens,
                    "summary/std_test_sensitivity": std_test_sens,
                    "summary/mean_test_specificity": avg_test_spec,
                    "summary/std_test_specificity": std_test_spec,
                    "summary/mean_test_fpr": avg_test_fpr,
                    "summary/std_test_fpr": std_test_fpr,
                }
                for key, value in summary_metrics.items():
                    mlflow_client.log_metric(parent_run_id, key, value)
                mlflow_client.log_artifact(parent_run_id, str(aggregated_results_path))
                mlflow_client.log_artifact(parent_run_id, str(summary_path))
            except Exception as e:
                logger.warning(f"Failed to log summary to parent MLflow run: {e}")

        logger.info("=" * 80)
        logger.info("K-FOLD CROSS-VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total requested folds: {requested_fold_count}")
        logger.info(f"Successful: {len(successful_folds)}")
        logger.info(f"Failed: {requested_fold_count - len(successful_folds)}")
        logger.info("")
        logger.info("Training Metrics:")
        logger.info(f"  Mean validation accuracy: {avg_val_acc_train:.4f} ± {std_val_acc_train:.4f}")
        logger.info("")
        logger.info("Test Metrics (PRIMARY - committed_overall):")
        logger.info(f"  Mean sensitivity: {avg_test_sens:.4f} ± {std_test_sens:.4f}")
        logger.info(f"  Mean specificity: {avg_test_spec:.4f} ± {std_test_spec:.4f}")
        logger.info(f"  Mean FPR: {avg_test_fpr:.4f} ± {std_test_fpr:.4f}")
        logger.info("=" * 80)

        # CRITICAL FIX: Always aggregate completed folds (not just when all folds requested)
        completed_fold_ids = [r['fold_id'] for r in all_results if r['status'] == 'success']

        if len(completed_fold_ids) > 0:
            logger.info("")
            logger.info("=" * 80)
            logger.info("CROSS-FOLD AGGREGATION")
            logger.info("=" * 80)
            logger.info(f"Completed folds: {completed_fold_ids}")

            successful_results = [r for r in all_results if r['status'] == 'success']

            try:
                from model.vae_teb_prediction.evaluate_classifier import (
                    plot_aggregated_roc_curves,
                )

                generate_aggregated_plots(
                    all_fold_results=successful_results,
                    output_base_dir=Path(output_base_dir),
                    n_folds=len(successful_results)
                )

                # Generate aggregated ROC plot
                all_roc = [r.get('roc_data', {}) for r in successful_results]
                normalised_roc = []
                for i, rd in enumerate(all_roc):
                    if rd and 'fpr' in rd and 'tpr' in rd and rd['fpr']:
                        entry = dict(rd)
                        if 'auc' not in entry:
                            entry['auc'] = successful_results[i].get('roc_auc', 0.0)
                        normalised_roc.append(entry)
                if normalised_roc:
                    agg_plots_dir = Path(output_base_dir) / 'aggregated_plots'
                    agg_plots_dir.mkdir(parents=True, exist_ok=True)
                    plot_aggregated_roc_curves(
                        normalised_roc,
                        agg_plots_dir / 'aggregated_roc_curves.png',
                        n_folds=len(successful_results),
                    )

                logger.info("Cross-fold aggregation completed successfully")
                logger.info(f"Aggregated plots saved to: {Path(output_base_dir) / 'aggregated_plots'}")
            except Exception as e:
                logger.error(f"Cross-fold aggregation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.warning("Individual fold results still available in each fold's 'three_metric_types' directory")

            logger.info("=" * 80)
        else:
            logger.warning("No successful folds to aggregate")

    # Save execution metadata for reproducibility and debugging
    execution_metadata = {
        'num_folds_configured': num_folds,
        'requested_folds': selected_folds,
        'successful_folds': [r['fold_id'] for r in all_results if r['status'] == 'success'],
        'failed_folds': [r['fold_id'] for r in all_results if r['status'] == 'failed'],
        'execution_mode': 'sequential' if sequential else 'parallel',
        'max_parallel': max_parallel if not sequential else 1,
        'gpu_ids': gpu_ids,
        'execution_timestamp': datetime.now().isoformat()
    }

    metadata_path = Path(output_base_dir) / "execution_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(execution_metadata, f, indent=2)
    logger.info(f"Saved execution metadata: {metadata_path}")

    # Log final artifacts and terminate the parent MLflow run
    if mlflow_client and parent_run_id:
        try:
            mlflow_client.log_artifact(parent_run_id, str(metadata_path))
            aggregated_plots_dir = Path(output_base_dir) / "aggregated_plots"
            if aggregated_plots_dir.exists():
                mlflow_client.log_artifacts(parent_run_id, str(aggregated_plots_dir), artifact_path="aggregated_plots")
            mlflow_client.set_terminated(parent_run_id)
            logger.info(f"Parent MLflow run terminated: {parent_run_id}")
        except Exception as e:
            logger.warning(f"Failed to finalize parent MLflow run: {e}")

    return all_results


def main():
    """
    Main entry point for k-fold cross-validation.

    All settings are read from config_cls.yaml. Override here only if needed.

    Usage:
        python kfold_classifier_trainer.py
    """
    # Path to the classifier config file
    BASE_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config_cls.yaml"
    )

    if not os.path.exists(BASE_CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {BASE_CONFIG_PATH}")

    with open(BASE_CONFIG_PATH, 'r') as f:
        base_config = yaml.safe_load(f)

    # Read settings from config
    general_cfg = base_config.get('general_config', {})
    model_cfg = base_config.get('model_config', {})
    dataset_cfg = base_config.get('dataset_config', {})

    GPU_IDS = general_cfg.get('cuda_devices', [0])
    MAX_PARALLEL = general_cfg.get('max_parallel_folds', 1)
    OUTPUT_BASE_DIR = general_cfg.get('folders_config', {}).get('out_dir_base', os.getcwd())
    VAE_CHECKPOINT = model_cfg.get('classifier', {}).get('vae_checkpoint')
    KFOLD_BASE_PATH = dataset_cfg.get('kfold_base_path', '')
    NUM_FOLDS = dataset_cfg.get('num_folds', 10)
    FOLDS_TO_RUN = dataset_cfg.get('fold_ids', None)  # null = all folds

    RUN_IN_PARALLEL = MAX_PARALLEL > 1

    logger.info(f"Config file: {BASE_CONFIG_PATH}")
    logger.info(f"Output base dir: {OUTPUT_BASE_DIR}")
    logger.info(f"K-fold base path: {KFOLD_BASE_PATH}")
    logger.info(f"GPU IDs: {GPU_IDS}")
    logger.info(f"Max parallel folds: {MAX_PARALLEL}")
    logger.info(f"Num folds: {NUM_FOLDS}")
    logger.info(f"Fold IDs: {FOLDS_TO_RUN if FOLDS_TO_RUN else 'all'}")

    # Validate VAE checkpoint
    if VAE_CHECKPOINT:
        if not os.path.exists(VAE_CHECKPOINT):
            raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CHECKPOINT}")
        logger.info(f"VAE checkpoint validated: {VAE_CHECKPOINT}")
    else:
        logger.warning("No VAE checkpoint specified - will attempt to train from scratch")

    logger.info("Starting K-Fold Cross-Validation for Classifier")

    results = run_kfold_parallel(
        num_folds=NUM_FOLDS,
        gpu_ids=GPU_IDS,
        base_config_path=BASE_CONFIG_PATH,
        kfold_base_path=KFOLD_BASE_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        vae_checkpoint=VAE_CHECKPOINT,
        max_parallel=MAX_PARALLEL,
        fold_ids=FOLDS_TO_RUN,
        sequential=not RUN_IN_PARALLEL,
    )

    logger.info("K-Fold Cross-Validation completed!")
    logger.info(f"Results saved to: {OUTPUT_BASE_DIR}")


if __name__ == '__main__':
    main()
