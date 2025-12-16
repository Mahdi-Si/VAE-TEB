from train.graph_model_base import GraphModelBase
from train.pl_model_base import LightningModelBase
from train.callbacks import (
    LossPlotCallback,
    HyperparameterLoggingCallback,
    MetricsLoggingCallback,
)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from loguru import logger

from train.graph_models_utils import load_checkpoint_strict
from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
from model.vae_teb_prediction.prediction_classification_model import (
    VaeTebTimeSeriesClassifier,
    LSTMClassifier,
    CNN1DClassifier,
    BiLSTMAttentionClassifier,
    TransformerClassifier,
)

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
import lightning as pl

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

from typing import List, Optional

import numpy as np
import torch
import time
import yaml


class PlSeqVaeClassifier(LightningModelBase):
    """
    PyTorch Lightning wrapper for VaeTebTimeSeriesClassifier.

    This class handles the training/validation logic for the combined VAE+Classifier model,
    following the same pattern as SeqVaePl from the VAE trainer.
    """

    def __init__(self, *args, class_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
            logger.info(f"PlSeqVaeClassifier: using class weights {class_weights}")
        else:
            self.class_weights = None

    def compute_loss_and_metrics(self, batch, batch_idx, stage: str):
        """
        Compute loss and metrics for binary classification task.

        Target sequence contains values 1, 2, or 3 at each timestep.
        Binary mapping: 1 → class 0, 2&3 → class 1

        Args:
            batch: Batch from dataloader with target shape (B, len_sequence)
            batch_idx: Batch index
            stage: Training stage ('train', 'val', or 'test')

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Extract features from batch
        y_st = batch.fhr_st      # (B, T, 43)
        y_ph = batch.fhr_ph      # (B, T, 44)
        x_ph = batch.fhr_up_ph   # (B, T, 130)

        # Get target sequence (B, len_sequence) with values 1, 2, or 3
        target_seq = batch.target  # (B, len_sequence)

        # Aggregate sequence to single label per sample (take max across sequence)
        labels = target_seq.max(dim=1)[0]  # (B,) - values will be 1, 2, or 3

        # Map to binary: 1 → 0, 2&3 → 1
        binary_labels = (labels > 1).long()  # (B,) - class 0 or 1

        # Forward pass through the model
        outputs = self.model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        logits = outputs["logits"]  # (B, 2)
        preds = outputs["preds"]    # (B,)

        # Compute loss with optional class weights
        weight = getattr(self, "class_weights", None)
        loss = torch.nn.functional.cross_entropy(
            logits,
            binary_labels,
            weight=weight,
        )

        # Compute accuracy
        accuracy = (preds == binary_labels).float().mean()

        # Compute per-class accuracy
        per_class_acc = {}
        for c in range(2):  # Binary classification: class 0 and class 1
            mask = binary_labels == c
            if mask.sum() > 0:
                per_class_acc[f"class_{c}_acc"] = (preds[mask] == binary_labels[mask]).float().mean()

        # Build metrics dictionary
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            **per_class_acc,
        }

        return loss, metrics


class GraphModelClassifierTrainer(GraphModelBase):
    """
    Training pipeline for VAE + Classifier model.

    This class creates the combined model, loads pre-trained VAE weights,
    and sets up the training loop following the same pattern as GraphModelVaeTebSmallTrainer.
    """
    def __init__(self, config_file_path=None):
        super(GraphModelClassifierTrainer, self).__init__(config_file_path)

    def create_model(self):
        """
        Create the combined VAE + Classifier model.

        This method:
        1. Loads the pre-trained VAE model
        2. Creates the classifier head
        3. Combines them into VaeTebTimeSeriesClassifier
        4. Wraps in PyTorch Lightning
        """
        # Get classifier config
        classifier_config = self.config.get('model_config', {}).get('classifier', {})
        vae_checkpoint = classifier_config.get('vae_checkpoint')

        if vae_checkpoint is None:
            raise ValueError("VAE checkpoint path must be provided in config under model_config.classifier.vae_checkpoint")

        # Create VAE model
        vae_model = SeqVae()

        # Load pre-trained VAE weights
        load_checkpoint_strict(
            model=vae_model,
            checkpoint=vae_checkpoint,
        )
        logger.info(f"VAE model loaded from checkpoint: {vae_checkpoint}")

        # Get classifier architecture and parameters
        classifier_type = classifier_config.get('type', 'lstm')  # 'lstm', 'cnn', 'bilstm_attention', 'transformer'
        latent_dim = classifier_config.get('latent_dim', 16)
        num_classes = classifier_config.get('num_classes', 2)
        freeze_vae = classifier_config.get('freeze_vae', True)
        use_posterior = classifier_config.get('use_posterior', True)
        sample_latent = classifier_config.get('sample_latent', False)

        classifier = LSTMClassifier(
            input_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_config.get('hidden_dim', 128),
            num_layers=classifier_config.get('num_layers', 2),
            bidirectional=classifier_config.get('bidirectional', False),
            dropout=classifier_config.get('dropout', 0.1),
        )

        logger.info(f"Created classifier: {classifier_type}")

        class_weights = classifier_config.get('class_weights')
        if class_weights is not None:
            if not isinstance(class_weights, (list, tuple)):
                raise ValueError("class_weights must be a list or tuple when provided")
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"class_weights length ({len(class_weights)}) must match num_classes ({num_classes})"
                )
            class_weights = [float(w) for w in class_weights]
            logger.info(f"Using class weights: {class_weights}")

        # Combine VAE + Classifier
        self.pytorch_model = VaeTebTimeSeriesClassifier(
            vae_model=vae_model,
            classifier=classifier,
            freeze_vae=freeze_vae,
            use_posterior=use_posterior,
            sample_latent=sample_latent,
            class_weights=class_weights,
        )

        # Log parameter counts
        total_params = sum(p.numel() for p in self.pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

        # Create PyTorch Lightning wrapper
        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
        }

        self.pl_model = PlSeqVaeClassifier(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            class_weights=class_weights,
        )

        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    def train_model(self, train_dataloader, validation_dataloader):
        """
        Train the classifier model using PyTorch Lightning.

        Args:
            train_dataloader: Training data loader
            validation_dataloader: Validation data loader

        Returns:
            trainer: PyTorch Lightning Trainer instance
        """
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})

        # Setup callbacks
        self.metrics_callback = MetricsLoggingCallback()
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get("plot_frequency", 1),
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            output_dir=self.train_results_dir,
            plot_frequency=10,
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/accuracy",
            filename="classifier-model-epoch={epoch:02d}-acc={val/accuracy:.4f}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode="max",  # Maximize accuracy
        )

        self.early_stopping_callback = EarlyStopping(
            monitor="val/accuracy",
            patience=30,
            mode="max",
            verbose=True,
        )
        
        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.checkpoint_callback,
            self.early_stopping_callback
        ]

        # Setup trainer configuration
        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        precision = trainer_cfg.get("precision", "32-true")
        gradient_clip_val = trainer_cfg.get("gradient_clip_val")
        gradient_clip_algorithm = trainer_cfg.get("gradient_clip_algorithm", "norm")
        logger_reference = self.lightning_loggers if self.lightning_loggers else True

        trainer_kwargs = {
            "max_epochs": self.epochs_num,
            "callbacks": callback_list,
            "default_root_dir": self.train_results_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": precision,
            "deterministic": trainer_cfg.get("deterministic", False),
            "benchmark": trainer_cfg.get("benchmark", True),
            "gradient_clip_val": gradient_clip_val,
            "gradient_clip_algorithm": gradient_clip_algorithm,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "use_distributed_sampler": True,
            "sync_batchnorm": len(self.cuda_devices) > 1,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
            "logger": logger_reference,
        }

        # Add GPU/CPU configuration
        if torch.cuda.is_available():
            trainer_kwargs.update(
                {
                    "accelerator": "gpu",
                    "devices": self.cuda_devices,
                    "strategy": "ddp" if len(self.cuda_devices) > 1 else "auto",
                }
            )
        else:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})
        # Create and run trainer
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)

        return trainer


def main():
    """
    Main training script for VAE + Classifier model.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()

    # Load config
    with open(r'config.yaml') as f:
        config = yaml.safe_load(f)

    # Setup dataset
    dataset_config = config.get('dataset_config')
    dataloader_config = dataset_config.get('dataloader_config')
    dataset_kwargs = dataloader_config.get('dataset_kwargs')
    normalized_fields = dataloader_config.get('normalize_fields')
    stat_path = dataset_config.get('stat_path')

    if stat_path is None:
        raise ValueError("stat_path must be provided")

    logger.info(f"Normalized fields: {normalized_fields}")

    # Create dataloaders
    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get('classifier_train_datasets', []),
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

    validation_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get('classifier_test_datasets', []),
        batch_size=config['general_config']['batch_size']['test'],
        num_workers=dataloader_config.get('num_workers', 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        rank=0,
        world_size=1,
        **dataset_kwargs
    )

    # Create and train model
    graph_model = GraphModelClassifierTrainer(config_file_path=r'config.yaml')
    graph_model.setup_config()
    graph_model.create_model()
    trainer = graph_model.train_model(train_dataloader, validation_dataloader)

    end_time = time.time()
    logger.info(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')


if __name__ == '__main__':
    main()
