"""Lightning wrapper + Graph-model trainer for ``SeqVaeLagAttnV1``.

This file mirrors the layout of :mod:`model.vae_teb_prediction.training.trainer`
but targets the new lag-attentive v1 model defined in
:mod:`model.vae_teb_prediction.model.vae_teb_lag_attn_v1`. The two trainers
coexist and the original ``GraphModelVaeTebSmallTrainer`` is untouched.

Usage:
    python -m model.vae_teb_prediction.training.trainer_lag_attn_v1 \\
        --config model/vae_teb_prediction/model/config_lag_attn_v1.yaml
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import lightning as pl
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (
    LagAttnV1PlotCallback,
)
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsLoggingCallback,
)
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import load_checkpoint_strict
from train.pl_model_base import LightningModelBase


# =============================================================================
# Lightning wrapper
# =============================================================================


class SeqVaeLagAttnPl(LightningModelBase):
    """Lightning wrapper for :class:`SeqVaeLagAttnV1`.

    Reads the four model-facing fields directly from the batch:

    * ``batch.fhr_st`` — FHR scattering (target stream, 43 ch)
    * ``batch.fhr_ph`` — FHR phase harmonics (target stream, 44 ch)
    * ``batch.up_st`` — UP scattering (source stream, 43 ch; optional,
      skipped when the model was built with ``use_up_st=False``)
    * ``batch.up_ph`` — UP self-phase harmonics (source stream, 58 ch)

    Both ``up_st`` and ``up_ph`` are first-class HDF5 datasets with their
    own per-channel asinh/log stats. They are not derived from
    ``fhr_up_ph``. The two tensors are concatenated on the channel axis
    here when ``use_up_st=True`` to form the 101-channel source stream;
    otherwise ``up_ph`` alone is used (58 channels).

    Expected keys in ``self.hparams`` (merged via ``apply_config_hyperparameters``):

    * ``kld_beta`` — weight on the KL term (default 0.01).
    * ``lambda_full`` — weight on ``L_feat`` (default 1.0).
    * ``lambda_base`` — weight on ``L_base`` (default 0.5).
    """

    #: Progress bar shows total + feature losses.
    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss")

    def _build_source_stream(self, batch: Any) -> torch.Tensor:
        """Build the ``u_stream`` tensor consumed by ``SeqVaeLagAttnV1.forward``.

        When ``use_up_st=True`` the stream is ``[up_st, up_ph]`` concatenated
        along the channel axis → ``(B, T, 101)``. When ``use_up_st=False`` it
        collapses to just ``up_ph`` → ``(B, T, 58)``. Both fields are read
        directly from the batch as independent HDF5 datasets.
        """
        up_ph = getattr(batch, "up_ph", None)
        if up_ph is None:
            raise RuntimeError(
                "batch has no `up_ph` field. Make sure 'up_ph' is listed in "
                "`dataset_kwargs.load_fields` of the config and that the HDF5 "
                "files were built with the new_pipeline (which writes up_ph as "
                "a first-class 58-channel dataset)."
            )
        use_up_st = bool(getattr(self.orig_model, "use_up_st", False))
        if not use_up_st:
            return up_ph
        up_st = getattr(batch, "up_st", None)
        if up_st is None:
            raise RuntimeError(
                "SeqVaeLagAttnV1 was constructed with use_up_st=True but the "
                "batch does not contain `up_st`. Either add 'up_st' to "
                "load_fields in the config, rebuild the HDF5 with up_st, or "
                "set use_up_st=False (and c_u=58) on the model."
            )
        return torch.cat([up_st, up_ph], dim=-1)

    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the forward pass and build the unified metrics dict."""
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        u_stream = self._build_source_stream(batch)

        forward_outputs = self.model(y_st, y_ph, u_stream)

        beta = float(self.hparams.get("kld_beta", 0.01))
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 0.5))

        loss_dict = self.orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            beta=beta,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
        )
        total_loss = loss_dict["total_loss"]
        metrics = {
            "total_loss": total_loss,
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            "kld_beta": beta,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
        }
        return total_loss, metrics


# =============================================================================
# Graph-model trainer
# =============================================================================


class GraphModelVaeTebLagAttnV1Trainer(GraphModelBase):
    """Experiment driver for the lag-attentive v1 model.

    Mirrors ``GraphModelVaeTebSmallTrainer`` but builds ``SeqVaeLagAttnV1``
    from config and uses the new Lightning wrapper.
    """

    def __init__(self, config_file_path: str | None = None) -> None:
        super().__init__(config_file_path)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Translate the ``VAE_model`` config section into constructor kwargs."""
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        kwargs: Dict[str, Any] = {
            "sequence_length": int(vae_cfg.get("sequence_length", 300)),
            "d_model": int(vae_cfg.get("d_model", 128)),
            "d_z": int(vae_cfg.get("d_z", 24)),
            "horizon": int(vae_cfg.get("horizon", 30)),
            "warmup_period": int(vae_cfg.get("warmup_period", 30)),
            "c_y": int(vae_cfg.get("c_y", 87)),
            "c_u": int(vae_cfg.get("c_u", 101)),
            "use_up_st": bool(vae_cfg.get("use_up_st", True)),
            "max_lag": int(vae_cfg.get("max_lag", 90)),
            "num_heads": int(vae_cfg.get("num_heads", 4)),
            "d_head": int(vae_cfg.get("d_head", 32)),
            "lstm_layers": int(vae_cfg.get("lstm_layers", 2)),
            "dropout": float(vae_cfg.get("dropout", 0.1)),
            "decoder_hidden": int(vae_cfg.get("decoder_hidden", 128)),
            "use_entmax": bool(vae_cfg.get("use_entmax", False)),
            "attention_grad_checkpoint": bool(
                vae_cfg.get("attention_grad_checkpoint", True)
            ),
        }
        logvar_clamp = vae_cfg.get("logvar_clamp")
        if isinstance(logvar_clamp, (list, tuple)) and len(logvar_clamp) == 2:
            kwargs["logvar_clamp"] = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        return kwargs

    def create_model(self) -> None:
        """Instantiate ``SeqVaeLagAttnV1`` and wrap it in ``SeqVaeLagAttnPl``."""
        model_kwargs = self._build_model_kwargs()
        logger.info(
            "Building SeqVaeLagAttnV1 with kwargs: "
            + ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
        )
        self.pytorch_model = SeqVaeLagAttnV1(**model_kwargs)

        self.checkpoint = self.config.get("model_config", {}).get("core_model_checkpoint")
        if self.checkpoint is not None:
            load_checkpoint_strict(
                model=self.pytorch_model,
                checkpoint=self.checkpoint,
            )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": vae_cfg.get("kld_beta", 0.01),
            "lambda_full": vae_cfg.get("lambda_full", 1.0),
            "lambda_base": vae_cfg.get("lambda_base", 0.5),
        }
        self.pl_model = SeqVaeLagAttnPl(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_model(self, train_dataloader, validation_dataloader):
        """Build callbacks + Lightning Trainer and run ``trainer.fit``."""
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})
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
            monitor="val/total_loss",
            filename="lag-attn-v1-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode=callbacks_cfg.get("model_checkpoint", {}).get("mode", "min"),
        )
        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
        ]

        # Diagnostic plotting — enabled by default, configurable via the
        # `advanced_config.callbacks.lag_attn_plotting` section.
        plot_cfg = callbacks_cfg.get("lag_attn_plotting", {}) or {}
        if plot_cfg.get("enabled", True):
            self.lag_attn_plot_callback = LagAttnV1PlotCallback(
                output_dir=self.train_results_dir,
                plot_frequency=int(plot_cfg.get("plot_frequency", 5)),
                num_examples=int(plot_cfg.get("num_examples", 2)),
                file_format=str(plot_cfg.get("file_format", "pdf")),
                mlflow_logger=self.mlflow_logger,
                forecast_channels=tuple(
                    int(c) for c in plot_cfg.get("forecast_channels", [0, 43, 80])
                ),
                forecast_anchor_frac=float(plot_cfg.get("forecast_anchor_frac", 0.6)),
            )
            callback_list.append(self.lag_attn_plot_callback)

        callback_list.append(self.checkpoint_callback)

        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        precision = trainer_cfg.get("precision", "32-true")
        gradient_clip_val = trainer_cfg.get("gradient_clip_val")
        gradient_clip_algorithm = trainer_cfg.get("gradient_clip_algorithm", "norm")
        logger_reference = self.lightning_loggers if self.lightning_loggers else True
        trainer_kwargs: Dict[str, Any] = {
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
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)
        return trainer


# =============================================================================
# Entry point
# =============================================================================


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "config_lag_attn_v1.yaml",
)


def main(config_path: str = _DEFAULT_CONFIG) -> None:
    """Build data loaders, model, trainer and run ``fit``."""
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config.get("dataset_config")
    dataloader_config = dataset_config.get("dataloader_config")
    dataset_kwargs = dataloader_config.get("dataset_kwargs")
    normalize_fields = dataloader_config.get("normalize_fields")
    stat_path = dataset_config.get("stat_path")
    if stat_path is None:
        raise ValueError("stat_path must be provided")
    logger.info(f"normalized fields: {normalize_fields}")
    logger.info(f"load fields:       {dataset_kwargs.get('load_fields')}")

    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_train_datasets", []),
        batch_size=config["general_config"]["batch_size"]["train"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=True,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        pin_memory=True,
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    validation_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_test_datasets", []),
        batch_size=config["general_config"]["batch_size"]["test"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    graph_model = GraphModelVaeTebLagAttnV1Trainer(config_file_path=config_path)
    graph_model.setup_config()
    graph_model.create_model()
    graph_model.train_model(train_dataloader, validation_dataloader)
    end_time = time.time()
    logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to the YAML config (default: config_lag_attn_v1.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
