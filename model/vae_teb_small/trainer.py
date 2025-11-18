from train.graph_model_base import GraphModelBase
from train.pl_model_base import LightningModelBase
from train.callbacks import (
    LossPlotCallback,
    HyperparameterLoggingCallback,
    MetricsLoggingCallback,
    PlottingCallBack
)

from loguru import logger

from train.graph_models_utils import load_checkpoint_strict
from vae_teb_model_small import SeqVaeCore

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
import lightning as pl

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

import numpy as np
import torch
import time
import yaml


class SeqVaeCorePl(LightningModelBase):

    def compute_loss_and_metrics(self, batch, batch_idx, stage: str):
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        x_ph = batch.fhr_up_ph
        y_raw  = batch.fhr
        forward_outputs = self.model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)
        loss_dict = self.orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            beta=self.hparams.get('kld_beta', 0.001),
        )
        total_loss = loss_dict['total_loss']
        metrics = {
            "total_loss": total_loss,
            "mse_loss": loss_dict["mse_loss"],
            "nll_loss": loss_dict["nll_loss"],
            "kld_loss": loss_dict["kld_loss"],
        }
        return total_loss, metrics


class GraphModelVaeTebSmallTrainer(GraphModelBase):
    def __init__(self, config_file_path=None):
        super(GraphModelVaeTebSmallTrainer, self).__init__(config_file_path)

    def create_model(self):
        self.pytorch_model = SeqVaeCore()
        self.checkpoint = self.config.get('model_config').get('core_model_checkpoint')
        if self.checkpoint is not None:
            load_checkpoint_strict(
                model=self.pytorch_model,
                checkpoint=self.checkpoint,
            )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")
        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": self.config["model_config"]["VAE_model"].get("kld_beta"),
        }
        self.pl_model = SeqVaeCorePl(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)
    
    def train_model(self, train_dataloader, validation_dataloader):
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
        self.plotting_callback = PlottingCallBack(
            output_dir=self.train_results_dir,
            plot_every_epoch=self.plot_every_epoch,
            input_channel_num=0,
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/total_loss",
            filename="core-model-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode=callbacks_cfg.get("model_checkpoint", {}).get("mode", "min"),
        )
        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.plotting_callback,
            self.checkpoint_callback,
        ]
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
        
        
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    start_time = time.time()
    
    with open(r'config.yaml') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset_config')
    dataloader_config = dataset_config.get('dataloader_config')
    dataset_kwargs = dataloader_config.get('dataset_kwargs')
    normalized_fields = dataloader_config.get('normalize_fields')
    stat_path = dataset_config.get('stat_path')
    if stat_path is None:
        raise ValueError("stat_path must be provided")
    logger.info(f"normalized fields: {normalized_fields}")
    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get('vae_train_datasets', []),
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
        hdf5_files=dataset_config.get('vae_test_datasets', []),
        batch_size=config['general_config']['batch_size']['test'],
        num_workers=dataloader_config.get('num_workers', 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalized_fields,
        rank=0,
        world_size=1,
        **dataset_kwargs
    )
    
    graph_model = GraphModelVaeTebSmallTrainer(config_file_path=r'config.yaml')
    graph_model.setup_config()
    graph_model.create_model()
    trainer = graph_model.train_model(train_dataloader, validation_dataloader)
    end_time = time.time()
    logger.info(f'Training completed in {(end_time - start_time)/60:.2f} minutes.')
    
    
if __name__ == '__main__':
    main()
