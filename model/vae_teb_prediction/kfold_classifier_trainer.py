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
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger
import json
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_fold_datasets(base_path: str, fold_id: int) -> Dict[str, List[str]]:
    """
    Get train, val, and test dataset paths for a specific fold.

    Args:
        base_path: Base path to k-fold dataset directory
        fold_id: Fold number (1-10)

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing lists of HDF5 file paths
    """
    fold_dir = Path(base_path) / f"fold_{fold_id}"

    datasets = {
        'train': [],
        'val': [],
        'test': []
    }

    for split in ['train', 'val', 'test']:
        split_dir = fold_dir / split
        if split_dir.exists():
            hdf5_files = sorted(split_dir.glob("*.hdf5"))
            datasets[split] = [str(f) for f in hdf5_files]
            logger.info(f"Fold {fold_id} {split}: found {len(datasets[split])} files")
        else:
            logger.warning(f"Split directory not found: {split_dir}")

    # Validate we have data for all splits
    for split in ['train', 'val', 'test']:
        if not datasets[split]:
            raise ValueError(f"Fold {fold_id} has no {split} files! Check dataset structure at {fold_dir}")

    return datasets


def create_mlflow_logger_from_config(
    config_data: Optional[Dict],
    run_name: str,
    extra_tags: Optional[Dict[str, str]] = None,
) -> Optional[MLFlowLogger]:
    """
    Build an MLflow logger (outside Lightning) using the base configuration.
    """
    if not config_data:
        return None

    mlflow_cfg = (
        config_data.get('advanced_config', {})
        .get('tracking', {})
        .get('mlflow', {})
        or {}
    )
    if not mlflow_cfg.get('enabled'):
        return None

    general_cfg = config_data.get('general_config', {})
    tags = dict(mlflow_cfg.get('tags') or {})
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})

    save_dir = (
        general_cfg.get('folders_config', {}).get('out_dir_base')
        or os.getcwd()
    )

    return MLFlowLogger(
        experiment_name=mlflow_cfg.get('experiment_name')
        or general_cfg.get('tag', 'kfold'),
        run_name=run_name,
        tracking_uri=mlflow_cfg.get('tracking_uri'),
        artifact_location=mlflow_cfg.get('artifact_location'),
        log_model=bool(mlflow_cfg.get('log_model', False)),
        tags=tags or None,
        save_dir=save_dir,
    )


def train_single_fold(
    fold_id: int,
    gpu_id: int,
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
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
        'kfold.fold_id': str(fold_id),
        'kfold.gpu_id': str(gpu_id),
        'kfold.train_files': str(len(fold_datasets['train'])),
        'kfold.val_files': str(len(fold_datasets['val'])),
        'kfold.test_files': str(len(fold_datasets['test'])),
        'kfold.base_path': str(kfold_base_path),
    })
    mlflow_cfg['tags'] = mlflow_tags
    if not mlflow_cfg.get('run_name'):
        experiment_tag = config['general_config'].get('tag', 'classifier')
        mlflow_cfg['run_name'] = f"{experiment_tag}-fold{fold_id:02d}"

    fold_output_dir = Path(output_base_dir) / f"fold_{fold_id}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    config['general_config']['folders_config']['out_dir_base'] = str(fold_output_dir)

    if 'model_config' not in config:
        config['model_config'] = {}
    if 'classifier' not in config['model_config']:
        config['model_config']['classifier'] = {}
    config['model_config']['classifier']['vae_checkpoint'] = vae_checkpoint

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
            if mlflow_logger is None or not metrics:
                return
            cleaned = {}
            for key, value in metrics.items():
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.item()
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if cleaned:
                mlflow_logger.log_metrics(cleaned, step=step)

        def _log_mlflow_artifact(path: Path | str, *, is_dir: bool = False):
            if mlflow_logger is None or not path:
                return
            path_obj = Path(path)
            if not path_obj.exists():
                return
            if is_dir:
                mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id, str(path_obj))
            else:
                mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(path_obj))

        # --------------------------------------------------------------------
        # POST-TRAINING EVALUATION
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Starting post-training evaluation...")

        from model.vae_teb_prediction.evaluate_classifier import evaluate_fold

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        eval_results = evaluate_fold(
            model=graph_model.pytorch_model,
            fold_dir=str(fold_output_dir),
            config_path=str(fold_config_path),
            checkpoint_path=best_ckpt_path,
            target_fpr=kwargs.get('target_fpr', 0.15),  # Default 15% FPR
            device=device
        )

        logger.info(f"Fold {fold_id}: Evaluation completed")

        results = {
            'fold_id': fold_id,
            'gpu_id': gpu_id,
            'training_time_minutes': training_time,
            'best_val_accuracy_training': float(best_val_acc),
            'best_val_loss_training': float(best_val_loss),
            'threshold': eval_results['validation_metrics']['threshold'],
            'validation_accuracy': eval_results['validation_metrics']['accuracy'],
            'validation_sensitivity': eval_results['validation_metrics']['sensitivity'],
            'validation_specificity': eval_results['validation_metrics']['specificity'],
            'test_accuracy': eval_results['test_metrics']['accuracy'],
            'test_sensitivity': eval_results['test_metrics']['sensitivity'],
            'test_specificity': eval_results['test_metrics']['specificity'],
            'test_fpr': eval_results['test_metrics']['fpr'],
            'status': 'success'
        }

        results_path = fold_output_dir / "fold_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Fold {fold_id} summary:")
        logger.info(f"  Training val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")
        logger.info(f"  Threshold: {results['threshold']:.4f}")
        logger.info(f"  Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"  Test sensitivity: {results['test_sensitivity']:.4f}, specificity: {results['test_specificity']:.4f}")

        _log_mlflow_metrics(
            {
                "train/training_time_minutes": training_time,
                "val/best_accuracy": best_val_acc,
                "val/best_loss": best_val_loss,
            },
            step=getattr(trainer, "global_step", None),
        )
        _log_mlflow_metrics(
            {
                "eval/threshold": results['threshold'],
                "eval/val_accuracy": results['validation_accuracy'],
                "eval/val_sensitivity": results['validation_sensitivity'],
                "eval/val_specificity": results['validation_specificity'],
                "eval/test_accuracy": results['test_accuracy'],
                "eval/test_sensitivity": results['test_sensitivity'],
                "eval/test_specificity": results['test_specificity'],
                "eval/test_fpr": results['test_fpr'],
            },
            step=getattr(trainer, "global_step", None),
        )

        _log_mlflow_artifact(fold_config_path)
        _log_mlflow_artifact(results_path)
        if best_ckpt_path:
            _log_mlflow_artifact(best_ckpt_path)
        evaluation_dir = fold_output_dir / "evaluation"
        if evaluation_dir.exists():
            _log_mlflow_artifact(evaluation_dir, is_dir=True)

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

    execution_mode = "parallel" if (not sequential and max_parallel > 1) else "sequential"

    logger.info(f"Starting k-fold cross-validation (configured folds={num_folds})")
    logger.info(f"Requested folds: {selected_folds}")
    logger.info(f"Execution mode: {execution_mode}")
    logger.info(f"Using GPUs: {gpu_ids}")
    if execution_mode == "parallel":
        logger.info(f"Max parallel folds: {max_parallel}")

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

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
                **kwargs
            )
            all_results.append(result)
            logger.info(f"Fold {fold_id} completed: {result}")
    else:
        logger.info("Running folds in parallel...")
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
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

        # Test metrics
        avg_test_acc = sum(r['test_accuracy'] for r in successful_folds) / n
        std_test_acc = (sum((r['test_accuracy'] - avg_test_acc) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        avg_test_sens = sum(r['test_sensitivity'] for r in successful_folds) / n
        std_test_sens = (sum((r['test_sensitivity'] - avg_test_sens) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        avg_test_spec = sum(r['test_specificity'] for r in successful_folds) / n
        std_test_spec = (sum((r['test_specificity'] - avg_test_spec) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        avg_test_fpr = sum(r['test_fpr'] for r in successful_folds) / n
        std_test_fpr = (sum((r['test_fpr'] - avg_test_fpr) ** 2 for r in successful_folds) / (n - 1)) ** 0.5 if n > 1 else 0.0

        summary = {
            'configured_num_folds': num_folds,
            'requested_folds': selected_folds,
            'successful_folds': len(successful_folds),
            'failed_folds': requested_fold_count - len(successful_folds),
            'training_metrics': {
                'mean_val_accuracy': avg_val_acc_train,
                'std_val_accuracy': std_val_acc_train,
            },
            'test_metrics': {
                'mean_accuracy': avg_test_acc,
                'std_accuracy': std_test_acc,
                'mean_sensitivity': avg_test_sens,
                'std_sensitivity': std_test_sens,
                'mean_specificity': avg_test_spec,
                'std_specificity': std_test_spec,
                'mean_fpr': avg_test_fpr,
                'std_fpr': std_test_fpr,
            },
            'individual_results': all_results
        }

        summary_path = Path(output_base_dir) / "kfold_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        experiment_tag = (
            (base_cfg_data or {}).get('general_config', {}).get('tag', 'classifier')
        )
        summary_run_name = f"{experiment_tag}-kfold-summary-{time.strftime('%Y%m%d-%H%M%S')}"
        summary_logger = create_mlflow_logger_from_config(
            base_cfg_data,
            run_name=summary_run_name,
            extra_tags={
                'kfold.summary': True,
                'kfold.configured_num_folds': num_folds,
                'kfold.requested_folds': selected_folds,
                'kfold.successful_folds': len(successful_folds),
            },
        )

        if summary_logger:
            summary_logger.log_metrics(
                {
                    "summary/successful_folds": len(successful_folds),
                    "summary/failed_folds": requested_fold_count - len(successful_folds),
                    "summary/mean_val_accuracy": avg_val_acc_train,
                    "summary/std_val_accuracy": std_val_acc_train,
                    "summary/mean_test_accuracy": avg_test_acc,
                    "summary/std_test_accuracy": std_test_acc,
                    "summary/mean_test_sensitivity": avg_test_sens,
                    "summary/std_test_sensitivity": std_test_sens,
                    "summary/mean_test_specificity": avg_test_spec,
                    "summary/std_test_specificity": std_test_spec,
                    "summary/mean_test_fpr": avg_test_fpr,
                    "summary/std_test_fpr": std_test_fpr,
                }
            )
            summary_logger.experiment.log_artifact(
                summary_logger.run_id, str(aggregated_results_path)
            )
            summary_logger.experiment.log_artifact(
                summary_logger.run_id, str(summary_path)
            )

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
        logger.info("Test Metrics (with threshold):")
        logger.info(f"  Mean accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
        logger.info(f"  Mean sensitivity (TPR): {avg_test_sens:.4f} ± {std_test_sens:.4f}")
        logger.info(f"  Mean specificity (TNR): {avg_test_spec:.4f} ± {std_test_spec:.4f}")
        logger.info(f"  Mean FPR: {avg_test_fpr:.4f} ± {std_test_fpr:.4f}")
        logger.info("=" * 80)

        need_full_aggregation = (fold_ids is None)
        if need_full_aggregation:
            logger.info("")
            logger.info("=" * 80)
            logger.info("RUNNING SUBGROUP AGGREGATION ACROSS FOLDS")
            logger.info("=" * 80)

            try:
                from model.vae_teb_prediction.evaluate_classifier import run_subgroup_aggregation

                run_subgroup_aggregation(
                    kfold_results_dir=output_base_dir,
                    num_folds=num_folds
                )

                logger.info("Subgroup aggregation completed successfully!")
            except Exception as e:
                import traceback
                logger.exception("Subgroup aggregation failed:")
                logger.warning("K-fold results are still valid, but aggregated subgroup analysis is missing")

            logger.info("=" * 80)

            if summary_logger:
                aggregated_dir = Path(output_base_dir) / "aggregated"
                if aggregated_dir.exists():
                    summary_logger.experiment.log_artifacts(
                        summary_logger.run_id, str(aggregated_dir)
                    )
                subgroup_dir = Path(output_base_dir) / "aggregated_analysis"
                if subgroup_dir.exists():
                    summary_logger.experiment.log_artifacts(
                        summary_logger.run_id, str(subgroup_dir)
                    )
        else:
            logger.info("Skipping aggregated subgroup analysis (disabled when running a subset of folds).")

    return all_results


def main():
    """
    Main entry point for k-fold cross-validation.

    Usage:
        python kfold_classifier_trainer.py
    """
    BASE_CONFIG_PATH = "config.yaml"
    KFOLD_BASE_PATH = "/data1/fetal-heart-tracing/HDF5_Datasets/last_12_hours_20_sec_delay/k_fold_cross_validation_dataset"
    OUTPUT_BASE_DIR = "/data/deid/isilon/MS_model/classifier_kfold_results"
    VAE_CHECKPOINT = None

    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    NUM_FOLDS = 10
    MAX_PARALLEL = 8
    RUN_IN_PARALLEL = True  # Set to False to run folds sequentially
    FOLDS_TO_RUN = None     # Example: [1, 3, 5] to run only specific folds

    if os.path.exists(BASE_CONFIG_PATH):
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config = yaml.safe_load(f)
            vae_ckpt = base_config.get('model_config', {}).get('classifier', {}).get('vae_checkpoint')
            if vae_ckpt:
                VAE_CHECKPOINT = vae_ckpt

    # Validate VAE checkpoint exists before starting folds
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
