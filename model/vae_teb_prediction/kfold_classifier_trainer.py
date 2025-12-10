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
import sys
import yaml
import torch
import subprocess
import time
from pathlib import Path
from typing import List, Dict
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

    return datasets


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

    config['general_config']['cuda_devices'] = [gpu_id]

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

    # Set random seeds
    np.random.seed(42 + fold_id)
    torch.manual_seed(42 + fold_id)

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

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

        # --------------------------------------------------------------------
        # POST-TRAINING EVALUATION
        # --------------------------------------------------------------------
        logger.info(f"Fold {fold_id}: Starting post-training evaluation...")

        from model.vae_teb_prediction.evaluate_classifier import evaluate_fold

        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
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

        return results

    except Exception as e:
        logger.error(f"Fold {fold_id} failed with error: {str(e)}")
        return {
            'fold_id': fold_id,
            'gpu_id': gpu_id,
            'status': 'failed',
            'error': str(e)
        }


def run_kfold_parallel(
    num_folds: int,
    gpu_ids: List[int],
    base_config_path: str,
    kfold_base_path: str,
    output_base_dir: str,
    vae_checkpoint: str,
    max_parallel: int = None,
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
        **kwargs: Additional config overrides

    Returns:
        List of result dictionaries, one per fold
    """
    if max_parallel is None:
        max_parallel = len(gpu_ids)

    logger.info(f"Starting {num_folds}-fold cross-validation")
    logger.info(f"Using GPUs: {gpu_ids}")
    logger.info(f"Max parallel folds: {max_parallel}")

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)

    all_results = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for fold_id in range(1, num_folds + 1):
            gpu_id = gpu_ids[(fold_id - 1) % len(gpu_ids)]

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
                logger.error(f"Fold {fold_id} raised exception: {e}")
                all_results.append({
                    'fold_id': fold_id,
                    'status': 'failed',
                    'error': str(e)
                })

    all_results.sort(key=lambda x: x['fold_id'])

    aggregated_results_path = Path(output_base_dir) / "kfold_results.json"
    with open(aggregated_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    successful_folds = [r for r in all_results if r['status'] == 'success']

    if successful_folds:
        # Training metrics
        avg_val_acc_train = sum(r['best_val_accuracy_training'] for r in successful_folds) / len(successful_folds)
        std_val_acc_train = (sum((r['best_val_accuracy_training'] - avg_val_acc_train) ** 2 for r in successful_folds) / len(successful_folds)) ** 0.5

        # Test metrics
        avg_test_acc = sum(r['test_accuracy'] for r in successful_folds) / len(successful_folds)
        std_test_acc = (sum((r['test_accuracy'] - avg_test_acc) ** 2 for r in successful_folds) / len(successful_folds)) ** 0.5

        avg_test_sens = sum(r['test_sensitivity'] for r in successful_folds) / len(successful_folds)
        std_test_sens = (sum((r['test_sensitivity'] - avg_test_sens) ** 2 for r in successful_folds) / len(successful_folds)) ** 0.5

        avg_test_spec = sum(r['test_specificity'] for r in successful_folds) / len(successful_folds)
        std_test_spec = (sum((r['test_specificity'] - avg_test_spec) ** 2 for r in successful_folds) / len(successful_folds)) ** 0.5

        avg_test_fpr = sum(r['test_fpr'] for r in successful_folds) / len(successful_folds)
        std_test_fpr = (sum((r['test_fpr'] - avg_test_fpr) ** 2 for r in successful_folds) / len(successful_folds)) ** 0.5

        summary = {
            'num_folds': num_folds,
            'successful_folds': len(successful_folds),
            'failed_folds': num_folds - len(successful_folds),
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

        logger.info("=" * 80)
        logger.info("K-FOLD CROSS-VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total folds: {num_folds}")
        logger.info(f"Successful: {len(successful_folds)}")
        logger.info(f"Failed: {num_folds - len(successful_folds)}")
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

        # Run subgroup aggregation across all folds
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
            logger.error(f"Subgroup aggregation failed: {e}")
            logger.warning("K-fold results are still valid, but aggregated subgroup analysis is missing")

        logger.info("=" * 80)

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

    if os.path.exists(BASE_CONFIG_PATH):
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config = yaml.safe_load(f)
            vae_ckpt = base_config.get('model_config', {}).get('classifier', {}).get('vae_checkpoint')
            if vae_ckpt:
                VAE_CHECKPOINT = vae_ckpt

    logger.info("Starting K-Fold Cross-Validation for Classifier")

    results = run_kfold_parallel(
        num_folds=NUM_FOLDS,
        gpu_ids=GPU_IDS,
        base_config_path=BASE_CONFIG_PATH,
        kfold_base_path=KFOLD_BASE_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        vae_checkpoint=VAE_CHECKPOINT,
        max_parallel=MAX_PARALLEL,
    )

    logger.info("K-Fold Cross-Validation completed!")
    logger.info(f"Results saved to: {OUTPUT_BASE_DIR}")


if __name__ == '__main__':
    main()
