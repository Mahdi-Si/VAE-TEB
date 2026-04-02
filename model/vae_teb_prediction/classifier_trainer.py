from train.graph_model_base import GraphModelBase
from train.pl_model_base import LightningModelBase
from train.callbacks import (
    LossPlotCallback,
    HyperparameterLoggingCallback,
    MetricsLoggingCallback,
)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import copy

from loguru import logger

from train.graph_models_utils import load_checkpoint_strict
from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
from model.vae_teb_prediction.prediction_classification_model import (
    VaeTebTimeSeriesClassifier,
    LSTMClassifier,
    CNNLSTMClassifier,
    CNN1DClassifier,
    BiLSTMAttentionClassifier,
    TransformerClassifier,
    MambaClassifier,
    MultiScaleConvAttentionClassifier,
    CausalCNNLSTMClassifier,
    FocalBCEWithLogitsLoss,
    map_to_hierarchical_labels,
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


class EMACallback(pl.Callback):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of trainable parameters updated as:
    ``shadow = decay * shadow + (1 - decay) * param``.

    At validation time, swaps model weights with EMA weights so that
    checkpoints and metrics reflect the averaged model. Swaps back
    after validation.

    Args:
        decay: EMA decay factor (0.999 is typical).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow: dict = {}
        self.backup: dict = {}

    def on_fit_start(self, trainer, pl_module):
        """Initialize shadow weights from model parameters."""
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA shadow weights after each training batch."""
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay,
                )

    def on_validation_epoch_start(self, trainer, pl_module):
        """Swap to EMA weights for validation."""
        self.backup = {}
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Restore training weights after validation."""
        for name, param in pl_module.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class PlSeqVaeClassifier(LightningModelBase):
    """PyTorch Lightning wrapper for VaeTebTimeSeriesClassifier.

    Supports both binary (cross-entropy) and hierarchical (focal BCE)
    label modes, latent-space mixup during training, and EMA model
    averaging via external callback.
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[List[float]] = None,
        label_mode: str = "binary",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        bit_weights: Optional[List[float]] = None,
        mixup_alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_mode = label_mode
        self.mixup_alpha = mixup_alpha

        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", weight_tensor)
            logger.info(f"PlSeqVaeClassifier: using class weights {class_weights}")
        else:
            self.class_weights = None

        # Focal loss for hierarchical mode
        if label_mode == "hierarchical":
            bw = torch.as_tensor(bit_weights, dtype=torch.float32) if bit_weights else None
            self.focal_loss = FocalBCEWithLogitsLoss(
                gamma=focal_gamma, alpha=bw, label_smoothing=label_smoothing,
            )
            logger.info(
                f"PlSeqVaeClassifier: hierarchical mode, focal_gamma={focal_gamma}, "
                f"label_smoothing={label_smoothing}, bit_weights={bit_weights}"
            )
        if mixup_alpha > 0:
            logger.info(f"PlSeqVaeClassifier: mixup enabled, alpha={mixup_alpha}")

    def _build_targets(self, labels: torch.Tensor):
        """Build float targets from raw labels for loss computation.

        Args:
            labels: Raw labels ``(B,)`` with values ``{1, 2, 3}``.

        Returns:
            Float targets suitable for the current ``label_mode``.
            Hierarchical: ``(B, 3)`` multi-hot. Binary: ``(B, 2)`` one-hot.
        """
        if self.label_mode == "hierarchical":
            return map_to_hierarchical_labels(labels)
        binary = (labels > 1).long()
        return torch.nn.functional.one_hot(binary, num_classes=2).float()

    def _compute_loss_from_logits(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from logits and soft float targets.

        Works with both mixed (soft) and non-mixed (hard) targets.

        Args:
            logits: ``(B, C)`` raw logits.
            targets: ``(B, C)`` float targets (may be soft from mixup).

        Returns:
            Scalar loss.
        """
        if self.label_mode == "hierarchical":
            return self.focal_loss(logits, targets)
        # Soft cross-entropy: -sum(target * log_softmax(logits))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

    def compute_loss_and_metrics(self, batch, batch_idx, stage: str):
        """Compute loss and metrics for classification.

        Applies latent-space mixup during training when ``mixup_alpha > 0``.

        Args:
            batch: Batch from dataloader with target shape ``(B, len_sequence)``.
            batch_idx: Batch index.
            stage: Training stage (``'train'``, ``'val'``, or ``'test'``).

        Returns:
            Tuple of ``(loss, metrics_dict)``.
        """
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        x_ph = batch.fhr_up_ph
        target_seq = batch.target
        labels = target_seq.max(dim=1)[0]  # (B,) values {1, 2, 3}
        tlo = batch.time_from_labor_onset if hasattr(batch, 'time_from_labor_onset') else None

        # --- Encode features ---
        z = self.model.encode_and_prepare(y_st=y_st, y_ph=y_ph, x_ph=x_ph, tlo=tlo)
        targets = self._build_targets(labels)  # (B, C) float

        # --- Mixup (training only) ---
        if self.training and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 for stability
            idx = torch.randperm(z.shape[0], device=z.device)
            z = lam * z + (1.0 - lam) * z[idx]
            targets = lam * targets + (1.0 - lam) * targets[idx]

        # --- Classify mixed features ---
        outputs = self.model.classify_features(z)
        logits = outputs["logits"]

        # --- Loss ---
        loss = self._compute_loss_from_logits(logits, targets)

        # --- Metrics (always on unmixed binary labels for consistency) ---
        binary_labels = (labels > 1).long()
        if self.label_mode == "hierarchical":
            preds = (torch.sigmoid(logits[:, 1]) > 0.5).long()
        else:
            preds = logits.argmax(dim=-1)

        accuracy = (preds == binary_labels).float().mean()
        per_class_acc = {}
        for c in range(2):
            mask = binary_labels == c
            if mask.sum() > 0:
                per_class_acc[f"class_{c}_acc"] = (preds[mask] == binary_labels[mask]).float().mean()

        metrics = {"loss": loss, "accuracy": accuracy, **per_class_acc}
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
        """Create the combined VAE + Classifier model.

        Loads the pre-trained VAE, creates the classifier head (with
        enriched features, attention pooling, and label_mode support),
        combines them, and wraps in PyTorch Lightning.
        """
        classifier_config = self.config.get('model_config', {}).get('classifier', {})
        vae_checkpoint = classifier_config.get('vae_checkpoint')

        if vae_checkpoint is None:
            raise ValueError("VAE checkpoint path must be provided in config under model_config.classifier.vae_checkpoint")

        vae_model = SeqVae()
        load_checkpoint_strict(model=vae_model, checkpoint=vae_checkpoint)
        logger.info(f"VAE model loaded from checkpoint: {vae_checkpoint}")

        # --- Core config ---
        classifier_type = classifier_config.get('type', 'lstm')
        latent_dim = classifier_config.get('latent_dim', 16)
        num_classes = classifier_config.get('num_classes', 2)
        freeze_vae = classifier_config.get('freeze_vae', True)
        use_posterior = classifier_config.get('use_posterior', True)
        sample_latent = classifier_config.get('sample_latent', False)

        # --- Enhancement config ---
        enriched_features = classifier_config.get('enriched_features', False)
        label_mode = classifier_config.get('label_mode', 'binary')
        attention_pool = classifier_config.get('attention_pool', False)
        focal_gamma = classifier_config.get('focal_gamma', 2.0)
        label_smoothing = classifier_config.get('label_smoothing', 0.0)
        bit_weights = classifier_config.get('bit_weights', None)

        # Augmentation config
        aug_config = classifier_config.get('augmentation', {})
        augment_posterior_sample = aug_config.get('posterior_sampling', False)
        augment_noise_scale = aug_config.get('noise_scale', 0.5)
        augment_temporal_jitter = aug_config.get('temporal_jitter', 0)

        # --- Compute classifier input dim ---
        tlo_embed_dim = classifier_config.get('tlo_embed_dim', 0)
        tlo_dropout = classifier_config.get('tlo_dropout', 0.1)
        feature_dim = latent_dim * 4 if enriched_features else latent_dim
        classifier_input_dim = feature_dim + tlo_embed_dim

        if enriched_features:
            logger.info(f"Enriched features enabled: {latent_dim}-dim -> {feature_dim}-dim "
                        f"(mu_post + logvar_post + residual + kld)")
        if tlo_embed_dim > 0:
            logger.info(f"TLO embedding enabled: embed_dim={tlo_embed_dim}, "
                        f"classifier input_dim={classifier_input_dim}")
        if label_mode == "hierarchical":
            logger.info(f"Hierarchical label mode: 3-bit [healthy, unhealthy, severe], "
                        f"focal_gamma={focal_gamma}, label_smoothing={label_smoothing}")

        # --- Shared kwargs for classifiers that support new features ---
        shared_kwargs = dict(label_mode=label_mode)
        pool_kwargs = dict(attention_pool=attention_pool, **shared_kwargs)

        if classifier_type == 'lstm':
            classifier = LSTMClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                hidden_dim=classifier_config.get('hidden_dim', 128),
                num_layers=classifier_config.get('num_layers', 2),
                bidirectional=classifier_config.get('bidirectional', False),
                dropout=classifier_config.get('dropout', 0.1),
                **pool_kwargs,
            )
        elif classifier_type == 'cnn_lstm':
            classifier = CNNLSTMClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                num_filters=classifier_config.get('num_filters', 32),
                kernel_sizes=tuple(classifier_config.get('kernel_sizes', [3, 5, 7])),
                cnn_out_dim=classifier_config.get('cnn_out_dim', 64),
                lstm_hidden=classifier_config.get('lstm_hidden', 128),
                lstm_layers=classifier_config.get('lstm_layers', 2),
                dropout=classifier_config.get('dropout', 0.1),
                pooling=classifier_config.get('pooling', 'mean_max'),
                **pool_kwargs,
            )
        elif classifier_type == 'cnn':
            classifier = CNN1DClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                num_filters=classifier_config.get('num_filters', 64),
                kernel_sizes=tuple(classifier_config.get('kernel_sizes', [3, 5, 7])),
                dropout=classifier_config.get('dropout', 0.1),
                **shared_kwargs,
            )
        elif classifier_type == 'bilstm_attention':
            classifier = BiLSTMAttentionClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                hidden_dim=classifier_config.get('hidden_dim', 128),
                num_layers=classifier_config.get('num_layers', 1),
                attn_dim=classifier_config.get('attn_dim', 64),
                dropout=classifier_config.get('dropout', 0.1),
                **shared_kwargs,
            )
        elif classifier_type == 'transformer':
            classifier = TransformerClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                d_model=classifier_config.get('d_model', 128),
                n_heads=classifier_config.get('n_heads', 4),
                num_layers=classifier_config.get('num_layers', 2),
                dim_feedforward=classifier_config.get('dim_feedforward', 256),
                dropout=classifier_config.get('dropout', 0.1),
                pooling=classifier_config.get('pooling', 'mean'),
                **shared_kwargs,
            )
        elif classifier_type == 'mamba':
            classifier = MambaClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                d_model=classifier_config.get('d_model', 64),
                d_state=classifier_config.get('d_state', 16),
                expand=classifier_config.get('expand', 2),
                conv_kernel=classifier_config.get('conv_kernel', 4),
                n_layers=classifier_config.get('n_layers', 3),
                dropout=classifier_config.get('dropout', 0.1),
                pooling=classifier_config.get('pooling', 'mean_max'),
                mlp_multiplier=classifier_config.get('mlp_multiplier', 2.0),
                **pool_kwargs,
            )
        elif classifier_type == 'multiscale_conv_attention':
            classifier = MultiScaleConvAttentionClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                num_filters=classifier_config.get('num_filters', 32),
                kernel_sizes=tuple(classifier_config.get('kernel_sizes', [5, 19, 39])),
                n_inception_blocks=classifier_config.get('n_inception_blocks', 2),
                se_reduction=classifier_config.get('se_reduction', 8),
                n_attn_heads=classifier_config.get('n_attn_heads', 4),
                attn_dropout=classifier_config.get('attn_dropout', 0.1),
                dropout=classifier_config.get('dropout', 0.1),
                mlp_multiplier=classifier_config.get('mlp_multiplier', 2.0),
                **shared_kwargs,
            )
        elif classifier_type == 'causal_cnn_lstm':
            classifier = CausalCNNLSTMClassifier(
                input_dim=classifier_input_dim,
                num_classes=num_classes,
                conv_channels=list(classifier_config.get('conv_channels', [32, 64, 128])),
                kernel_sizes=list(classifier_config.get('kernel_sizes', [5, 7, 11])),
                dilations=list(classifier_config.get('dilations', [1, 2, 4])),
                lstm_hidden=classifier_config.get('lstm_hidden', 128),
                lstm_layers=classifier_config.get('lstm_layers', 2),
                dropout=classifier_config.get('dropout', 0.1),
                pooling=classifier_config.get('pooling', 'mean_max'),
                mlp_multiplier=classifier_config.get('mlp_multiplier', 2.0),
                **pool_kwargs,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        logger.info(f"Created classifier: {classifier_type}")

        class_weights = classifier_config.get('class_weights')
        if class_weights is not None:
            if not isinstance(class_weights, (list, tuple)):
                raise ValueError("class_weights must be a list or tuple when provided")
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
            tlo_embed_dim=tlo_embed_dim,
            tlo_dropout=tlo_dropout,
            enriched_features=enriched_features,
            label_mode=label_mode,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            bit_weights=bit_weights,
            augment_posterior_sample=augment_posterior_sample,
            augment_noise_scale=augment_noise_scale,
            augment_temporal_jitter=augment_temporal_jitter,
        )

        total_params = sum(p.numel() for p in self.pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

        # Training procedure config
        training_cfg = classifier_config.get('training', {})
        mixup_alpha = training_cfg.get('mixup_alpha', 0.0)
        weight_decay = training_cfg.get('weight_decay', 0.01)
        scheduler_type = training_cfg.get('scheduler_type', 'cosine')
        warmup_epochs = training_cfg.get('warmup_epochs', 10)
        min_lr = training_cfg.get('min_lr', 1e-6)
        max_epochs = self.config.get('general_config', {}).get('epochs', 1000)

        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "weight_decay": weight_decay,
            "scheduler_type": scheduler_type,
            "warmup_epochs": warmup_epochs,
            "min_lr": min_lr,
            "max_epochs": max_epochs,
        }

        self.pl_model = PlSeqVaeClassifier(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            class_weights=class_weights,
            label_mode=label_mode,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            bit_weights=bit_weights,
            mixup_alpha=mixup_alpha,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
            max_epochs=max_epochs,
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
        ckpt_cfg = callbacks_cfg.get("model_checkpoint", {})
        ckpt_monitor = ckpt_cfg.get("monitor", "val/accuracy")
        ckpt_mode = ckpt_cfg.get("mode", "max")
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor=ckpt_monitor,
            filename="classifier-model-{epoch:02d}",
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            mode=ckpt_mode,
            auto_insert_metric_name=False,
        )

        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.checkpoint_callback,
        ]

        # Early stopping
        es_cfg = callbacks_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", True):
            es_monitor = es_cfg.get("monitor", "val/accuracy")
            es_mode = es_cfg.get("mode", "max")
            self.early_stopping_callback = EarlyStopping(
                monitor=es_monitor,
                patience=es_cfg.get("patience", 50),
                mode=es_mode,
                verbose=True,
            )
            callback_list.append(self.early_stopping_callback)

        # EMA model averaging
        training_cfg = self.config.get("model_config", {}).get("classifier", {}).get("training", {})
        ema_decay = training_cfg.get("ema_decay", 0.0)
        if ema_decay > 0:
            self.ema_callback = EMACallback(decay=ema_decay)
            callback_list.append(self.ema_callback)
            logger.info(f"EMA enabled with decay={ema_decay}")

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
