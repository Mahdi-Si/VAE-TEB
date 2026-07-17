"""Trainer orchestrator for the Causal Multimodal Forecasting Transformer.

Subclasses ``GraphModelBase`` to provide config-driven model creation,
callback setup, and Lightning Trainer construction with multi-GPU DDP support
and MLflow experiment tracking.

Usage::

    python -m model.transformer.training.trainer --config model/transformer/config.yaml
"""

import argparse
import os
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.transformer.model import (
    CausalMultimodalTransformer,
    CausalTransformerLoss,
    TransformerConfig,
)
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
)
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import load_checkpoint_strict

from .lightning_module import PlCausalTransformer
from .plotting_callback import TransformerPlotCallback


class CausalTransformerTrainer(GraphModelBase):
    """Trainer orchestrator for the causal multimodal transformer.

    Reads a YAML config, builds the model and Lightning module, sets up
    callbacks (including MLflow), constructs dataloaders, and runs training
    with multi-GPU DDP support.

    Args:
        config_file_path: Path to the YAML configuration file.
    """

    def create_model(self) -> None:
        """Instantiate model, loss, and Lightning wrapper from config.

        Reads ``model_config.transformer`` for architecture, loads an optional
        checkpoint, and wraps everything in ``PlCausalTransformer``.
        """
        # --- Build TransformerConfig from YAML ---
        transformer_dict = dict(self.config["model_config"]["transformer"])
        checkpoint_path = transformer_dict.pop("checkpoint_path", None)

        # Convert list fields to tuples (YAML loads lists, dataclass expects tuples)
        for key, value in transformer_dict.items():
            if isinstance(value, list):
                transformer_dict[key] = tuple(value)

        transformer_config = TransformerConfig(**transformer_dict)

        # --- Instantiate model and loss (share config) ---
        model = CausalMultimodalTransformer(transformer_config)
        loss_fn = CausalTransformerLoss(transformer_config)

        # --- Optional checkpoint loading ---
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            load_checkpoint_strict(model=model, checkpoint=checkpoint_path)

        # --- Read training schedule ---
        schedule = self.config.get("training_schedule", {})
        stage1 = schedule.get("stage1_epochs", 60)
        stage2 = schedule.get("stage2_epochs", 60)
        stage3 = schedule.get("stage3_epochs", 180)
        beta_max = schedule.get("beta_max", schedule.get("beta_max_transfer", 1e-4))
        warmup_steps = schedule.get("warmup_steps", 5000)

        # Override total epochs
        self.epochs_num = stage1 + stage2 + stage3

        # --- Wrap in Lightning module ---
        self.pl_model = PlCausalTransformer(
            base_model=model,
            loss_fn=loss_fn,
            transformer_config=transformer_config,
            lr=self.lr,
            lr_milestones=self.lr_milestones if self.lr_milestones else None,
            weight_decay=self.config["general_config"].get("weight_decay", 1e-4),
            stage1_epochs=stage1,
            stage2_epochs=stage2,
            stage3_epochs=stage3,
            beta_max=beta_max,
            warmup_steps=warmup_steps,
        )

        # --- Log hyperparameters to MLflow ---
        if self.mlflow_logger is not None:
            hparams = {
                "d_model": transformer_config.d_model,
                "n_heads": transformer_config.n_heads,
                "total_encoder_layers": (
                    transformer_config.fhr_encoder_layers
                    + transformer_config.up_encoder_layers
                    + transformer_config.fused_encoder_layers
                ),
                "d_z_transfer": transformer_config.d_z_transfer,
                "fusion_layers": transformer_config.fusion_layers,
                "dropout": transformer_config.dropout,
                "num_anchors": transformer_config.num_anchors,
                "horizons": str(transformer_config.horizons),
                "stage1_epochs": stage1,
                "stage2_epochs": stage2,
                "stage3_epochs": stage3,
                "beta_max": beta_max,
                "warmup_steps": warmup_steps,
                "total_epochs": self.epochs_num,
                "use_swiglu": transformer_config.use_swiglu,
                "use_rmsnorm": transformer_config.use_rmsnorm,
                "gradient_checkpointing": transformer_config.gradient_checkpointing,
            }
            self.mlflow_logger.log_hyperparams(hparams)

        # --- Resume overrides (applied after checkpoint restore) ---
        # These allow changing LR, beta_max, stage epochs, etc. when resuming.
        # The values come from the CURRENT config, overwriting whatever the
        # checkpoint saved.  Only non-None keys are applied.
        self._resume_overrides = {
            "lr": self.lr,
            "weight_decay": self.config["general_config"].get("weight_decay", 1e-4),
            "lr_milestones": self.lr_milestones if self.lr_milestones else None,
            "stage1_epochs": stage1,
            "stage2_epochs": stage2,
            "stage3_epochs": stage3,
            "beta_max": beta_max,
            "warmup_steps": warmup_steps,
        }

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model created: {total_params:,} parameters, "
            f"{self.epochs_num} epochs ({stage1}+{stage2}+{stage3})"
        )

    def train_model(
        self,
        train_loader,
        val_loader,
        resume_ckpt_path: Optional[str] = None,
    ) -> None:
        """Build callbacks, create Lightning Trainer, and run training.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            resume_ckpt_path: Path to a Lightning ``.ckpt`` file to resume
                from.  Restores model weights, optimizer state, epoch counter,
                and the 3-stage schedule state (beta, stage, global step).
        """
        # --- Apply config overrides when resuming ---
        # On resume, Lightning restores hparams from the checkpoint.  We
        # overwrite them with the CURRENT config values so that the user can
        # change LR, beta_max, stage epochs, etc. between runs.
        if resume_ckpt_path is not None:
            self.apply_config_hyperparameters(
                self._resume_overrides, self.pl_model,
            )

        # --- Callbacks ---
        callbacks = []

        # Checkpoint
        cb_cfg = self.config.get("advanced_config", {}).get("callbacks", {})
        ckpt_cfg = cb_cfg.get("model_checkpoint", {})
        checkpoint_cb = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor=ckpt_cfg.get("monitor", "val/L_fus"),
            filename="transformer-epoch={epoch:03d}-Lfus={val/L_fus:.4f}",
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            mode=ckpt_cfg.get("mode", "min"),
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_cb)

        # Early stopping
        es_cfg = cb_cfg.get("early_stopping", {})
        if es_cfg.get("enabled", False):
            callbacks.append(EarlyStopping(
                monitor=es_cfg.get("monitor", "val/L_fus"),
                patience=es_cfg.get("patience", 50),
                mode=es_cfg.get("mode", "min"),
                verbose=True,
            ))

        # Loss plotting
        callbacks.append(LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            mlflow_logger=self.mlflow_logger,
        ))

        # Hyperparameter tracking (beta, stage, lr)
        callbacks.append(HyperparameterLoggingCallback(
            tracked_keys=(
                "train/beta", "train/stage",
                "val/beta", "val/stage",
                "lr", "learning_rate",
            ),
            output_dir=self.train_results_dir,
        ))

        # Transformer diagnostic plots
        callbacks.append(TransformerPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            num_examples=2,
            mlflow_logger=self.mlflow_logger,
        ))

        # --- Trainer configuration ---
        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        use_cuda = torch.cuda.is_available() and len(self.cuda_devices) > 0
        multi_gpu = use_cuda and len(self.cuda_devices) > 1

        trainer_kwargs = {
            "max_epochs": self.epochs_num,
            "callbacks": callbacks,
            "default_root_dir": self.train_results_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": trainer_cfg.get("precision", "16-mixed"),
            "gradient_clip_val": trainer_cfg.get("gradient_clip_val", 1.0),
            "gradient_clip_algorithm": trainer_cfg.get(
                "gradient_clip_algorithm", "norm"
            ),
            "deterministic": trainer_cfg.get("deterministic", False),
            "benchmark": trainer_cfg.get("benchmark", True),
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 2,
            "use_distributed_sampler": True,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
        }

        # Logger(s)
        if self.lightning_loggers:
            trainer_kwargs["logger"] = self.lightning_loggers

        # Accelerator
        # find_unused_parameters=True is required because window_export's
        # attention pools are only used in inference mode, not during training.
        if use_cuda:
            trainer_kwargs.update({
                "accelerator": "gpu",
                "devices": self.cuda_devices,
                "strategy": (
                    "ddp_find_unused_parameters_true" if multi_gpu else "auto"
                ),
                "sync_batchnorm": multi_gpu,
            })
        else:
            trainer_kwargs.update({
                "accelerator": "cpu",
                "devices": 1,
            })

        trainer = L.Trainer(**trainer_kwargs)

        logger.info(
            f"Starting training: {self.epochs_num} epochs, "
            f"devices={self.cuda_devices}, "
            f"precision={trainer_kwargs['precision']}, "
            f"strategy={trainer_kwargs.get('strategy', 'auto')}"
        )
        trainer.fit(
            self.pl_model, train_loader, val_loader,
            ckpt_path=resume_ckpt_path,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    config_path: Optional[str] = None,
    resume_ckpt_path: Optional[str] = None,
) -> None:
    """Run the full training pipeline.

    Args:
        config_path: Path to the YAML config. Defaults to
            ``model/transformer/config.yaml``.
        resume_ckpt_path: Path to a Lightning ``.ckpt`` file to resume from.
            Restores model weights, optimizer, epoch, and stage state.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )

    trainer_obj = CausalTransformerTrainer(config_path)
    trainer_obj.setup_config()
    trainer_obj.create_model()

    # --- Build dataloaders ---
    ds_cfg = trainer_obj.config["dataset_config"]
    dl_cfg = ds_cfg.get("dataloader_config", {})
    dataset_kwargs = dict(dl_cfg.get("dataset_kwargs", {}))
    # pin_memory must NOT be set on the dataset — it causes every dataloader
    # worker to initialize a CUDA context on GPU 0 (~260 MiB each).  Lightning
    # handles pin_memory at the DataLoader level automatically on CUDA.
    dataset_kwargs.pop("pin_memory", None)

    common_dl_kwargs = {
        "stats_path": ds_cfg.get("stat_path"),
        "normalize_fields": dl_cfg.get("normalize_fields"),
    }

    train_loader = create_optimized_dataloader(
        hdf5_files=ds_cfg["train_datasets"],
        batch_size=trainer_obj.batch_size_train,
        num_workers=dl_cfg.get("num_workers", 4),
        shuffle=True,
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
        **common_dl_kwargs,
        **dataset_kwargs,
    )

    val_loader = create_optimized_dataloader(
        hdf5_files=ds_cfg["val_datasets"],
        batch_size=trainer_obj.batch_size_test,
        num_workers=0,
        shuffle=False,
        **common_dl_kwargs,
        **dataset_kwargs,
    )

    trainer_obj.train_model(
        train_loader, val_loader, resume_ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Causal Multimodal Forecasting Transformer."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (default: model/transformer/config.yaml)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a Lightning .ckpt file to resume training from.",
    )
    args = parser.parse_args()
    main(config_path=args.config, resume_ckpt_path=args.resume)
