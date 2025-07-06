import lightning as L
import sklearn.utils
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import torch.nn as nn
import torch
from torch.utils.data import \
    DataLoader, \
    random_split, \
    Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import logging
import os
from collections import OrderedDict
import yaml
import logging
from datetime import datetime
import sys
import pickle
import argparse
from tqdm import tqdm
import time
import numpy as np

from utils.custom_logger import setup_logging, InterceptHandler

from utils.data_utils import \
    plot_forward_pass, \
    plot_averaged_results, \
    plot_generated_samples, \
    plot_distributions, \
    plot_histogram, \
    plot_loss_dict, \
    plot_latent_interpolation, \
    animate_latent_interpolation, \
    plot_original_reconstructed, \
    plot_prediction_st, \
    plot_forward_pass_kld

from utils.graph_model_utils import \
    calculate_log_likelihood, \
    interpolate_latent, \
    calculate_vaf

from loguru import logger

from pytorch_lightning_modules import *

from sklearn.manifold import TSNE
import pandas as pd

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from vae_teb_model_improved import SeqVaeTeb
from pytorch_lightning_modules import LightSeqVaeTeb

from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist

# Add this line to enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "7"

matplotlib.use('Agg')



class SeqVAEGraphModel:
    def __init__(self, config_file_path=None):
        super(SeqVAEGraphModel, self).__init__()
        if config_file_path is None:
            self.config_file_path = os.path.dirname(os.path.realpath(__file__)) + '/seqvae_configs/config_args.yaml'
        else:
            self.config_file_path = config_file_path

        with open(self.config_file_path) as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        now = datetime.now()
        run_date = now.strftime("%Y-%m-%d--[%H-%M]-")
        self.experiment_tag = self.config['general_config']['tag']
        self.cuda_devices = self.config['general_config']['cuda_devices']

        self.output_base_dir = os.path.normpath(self.config['folders_config']['out_dir_base'])
        self.base_folder = f'{run_date}-{self.experiment_tag}'
        self.train_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'train_results')
        self.test_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'test_results')
        self.model_checkpoint_dir = os.path.join(self.output_base_dir, self.base_folder, 'model_checkpoints')
        self.aux_dir = os.path.join(self.output_base_dir, self.base_folder, 'aux_test_HIE')
        self.tensorboard_dir = os.path.join(self.output_base_dir, self.base_folder, 'tensorboard_log')
        self.log_file = None
        self.logger = None

        # logger.info yaml file properly -------------------------------------------------------------------------------------
        logger.info(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        logger.info('==' * 50)
        self.stat_path = os.path.normpath(self.config['dataset_config']['stat_path'])

        self.plot_every_epoch = self.config['general_config']['plot_frequency']

        self.raw_input_size = self.config['model_config']['VAE_model']['raw_input_size']
        self.input_size = self.config['model_config']['VAE_model']['input_size']

        self.input_dim = self.config['model_config']['VAE_model']['input_dim']
        self.input_channel_num = self.config['model_config']['VAE_model']['channel_num']

        self.latent_dim = self.config['model_config']['VAE_model']['latent_size']
        self.num_layers = self.config['model_config']['VAE_model']['num_RNN_layers']
        self.rnn_hidden_dim = self.config['model_config']['VAE_model']['RNN_hidden_dim']
        self.y_module_only = self.config['model_config']['VAE_model']['Y_module_only']
        self.epochs_num = self.config['general_config']['epochs']
        self.lr = self.config['general_config']['lr']
        self.lr_milestones = self.config['general_config']['lr_milestone']
        self.kld_beta_ = float(self.config['model_config']['VAE_model']['kld_beta'])
        self.seqvae_ckp = self.config['model_config']['seqvae_checkpoint']

        self.train_classifier = self.config['general_config']['train_classifier']

        self.freeze_seqvae = self.config['model_config']['VAE_model']['freeze_seqvae']
        self.batch_size_train = self.config['general_config']['batch_size']['train']
        self.batch_size_test = self.config['general_config']['batch_size']['test']
        self.accumulate_grad_batches = self.config['general_config'].get('accumulate_grad_batches', 1)

        self.test_checkpoint_path = None
        self.seqvae_testing_checkpoint = self.config['seqvae_testing']['test_checkpoint_path']
        self.base_model_checkpoint = self.config['model_config']['base_model_checkpoint']

        self.inv_scattering_checkpoint = self.config['inv_scattering_model']['inv_st_checkpoint']
        self.do_inv_st = self.config['inv_scattering_model']['do_inv_st']
        self.train_inv_st = self.config['inv_scattering_model']['train_inv_st']

        self.zero_source = self.config['model_config']['VAE_model']['zero_source']
        self.clip = 10
        plt.ion()

        self.log_stat = None
        self.latent_stats = None
        self.model = None
        self.seqvae_lightning_model = None
        self.classifier = None
        self.inv_scattering_model = None
        self.csv_logger = None
        self.plotting_callback = None
        self.classification_performance_callback = None
        self.base_model = None
        self.pytorch_model = None
        self.prd_base_model = None
        self.checkpoint_callback = None
        self.early_stop_callback = None
        self.loss_plot_callback = None
        self.metrics_callback = None
        self.lightning_base_model = None


    def setup_config(self):
        folders_list = [
            self.output_base_dir,
            self.train_results_dir,
            self.test_results_dir,
            self.model_checkpoint_dir,
            self.aux_dir,
            self.tensorboard_dir
        ]
        for folder in folders_list:
            os.makedirs(folder, exist_ok=True)

        self.log_file = os.path.join(self.train_results_dir, 'full.log')
        
        # Reconfiguring logger to be multiprocessing-safe
        logger.remove() # Removes the default handler
        logger.add(sys.stderr, level="INFO") # Keep console logging
        logger.add(
            self.log_file,
            level="INFO",
            rotation="100 MB",
            retention="14 days",
            compression="zip",
            enqueue=True,  # This is the key for multiprocessing safety
            backtrace=True,
            diagnose=True,
            serialize=False,
        )

        # sys.stdout = StreamToLogger(self.logger, logging.INFO)
        logger.info(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        logger.info('==' * 50)

    def load_checkpoint(self):
        """
        Loads a checkpoint for both the PyTorch and PyTorch Lightning models.
        If a checkpoint path is provided in the configuration, this function will:
        1. Load the PyTorch Lightning module (`LightSeqVaeTeb`) from the checkpoint.
        2. The underlying PyTorch model (`SeqVaeTeb`) will be part of the loaded Lightning module.
        """
        if self.base_model_checkpoint and os.path.exists(self.base_model_checkpoint):
            logger.info(f"Loading model from checkpoint: {self.base_model_checkpoint}")

            # Instantiate a base model to pass to load_from_checkpoint
            # The actual weights will be overwritten by the checkpoint's state_dict.
            base_model_for_loading = SeqVaeTeb(
                input_channels=76,  # Replace with config values if available
                sequence_length=300,
                prediction_horizon=30,
                warmup_period=30,
                kld_beta=1.0
            )

            try:
                self.lightning_base_model = LightSeqVaeTeb.load_from_checkpoint(
                    self.base_model_checkpoint,
                    seqvae_teb_model=base_model_for_loading,
                    strict=False 
                )
                self.base_model = self.lightning_base_model.model
                logger.info("Successfully loaded Lightning model and base PyTorch model from checkpoint.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.error("Initializing models from scratch.")
                self.base_model_checkpoint = None  # Reset checkpoint path to prevent further errors
                self.create_model() # Re-initialize models without checkpoint

        else:
            if self.base_model_checkpoint:
                logger.warning(f"Checkpoint file not found at {self.base_model_checkpoint}. Initializing models from scratch.")
            else:
                logger.info("No checkpoint provided. Initializing models from scratch.")
            
            self.base_model = SeqVaeTeb(
                input_channels=76,
                sequence_length=300,
                prediction_horizon=30,
                warmup_period=30,
                kld_beta=1.0
            )
            self.lightning_base_model = LightSeqVaeTeb(
                seqvae_teb_model=self.base_model,
                lr=self.lr,
                lr_milestones=self.lr_milestones,
                beta_schedule="constant",
                beta_const_val=self.kld_beta_
            )

    def load_pytorch_checkpoint(self):
        if self.seqvae_ckp is not None:
            logger.info(f"Loading checkpoint: {self.seqvae_ckp}")
            # checkpoint = torch.load(self.seqvae_checkpoint_path,  map_location=self.device)
            checkpoint = torch.load(self.seqvae_ckp)
            state_dict = checkpoint['state_dict']
            # filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'scattering_transform' not in k}
            state_dict = {k.replace('seqvae_model.', ''): v for k, v in state_dict.items()}
            self.pytorch_model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint '{self.seqvae_ckp}' (epoch {checkpoint['epoch']})")

    def create_model(self):
        self.setup_config()
        self.load_checkpoint()

    def set_cuda_devices(self, device_list=None):
        self.cuda_devices = device_list if device_list is not None else [0]

    @staticmethod
    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    def train_base_model(self, train_loader, validation_loader):
        """
        Trains the base SeqVaeTeb model using PyTorch Lightning.

        This function configures and runs the training process, leveraging multi-GPU
        support, callbacks for early stopping, model checkpointing, and real-time
        loss plotting with Plotly.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            validation_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            dict: A dictionary containing the training history.
        """
        logger.info("Setting up trainer for the base model...")

        self.plotting_callback = PlottingCallBack(
            output_dir=self.train_results_dir,
            plot_every_epoch=self.plot_every_epoch,
            input_channel_num=self.input_channel_num,
        )

        self.metrics_callback = MetricsLoggingCallback()

        # Callback for early stopping to prevent overfitting
        self.early_stop_callback = EarlyStopping(
            monitor="val/total_loss",
            min_delta=0.0001,
            patience=100,
            verbose=True,
            mode="min"
        )

        # Callback for saving the best model checkpoint
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val/total_loss",
            mode="min",
            dirpath=self.model_checkpoint_dir,
            filename="base-model-best-{epoch:02d}-{val/total_loss:.2f}",
            save_top_k=1,
            save_last=False,
        )

        # Callback for plotting losses using Plotly
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=1
        )

        # Profiler for performance analysis
        profiler = SimpleProfiler(dirpath=self.train_results_dir, filename="profiler_base_model.txt")

        # Configure devices and strategy for training
        if self.cuda_devices and len(self.cuda_devices) > 0:
            loging_steps = (len(train_loader.dataset) // self.batch_size_train) // len(self.cuda_devices) if self.batch_size_train > 0 else 1
            process_group_backend = "gloo" if sys.platform == "win32" else "nccl"
            strategy = DDPStrategy(find_unused_parameters=True, process_group_backend=process_group_backend)
            accelerator = "gpu"
            devices = self.cuda_devices
        else:
            loging_steps = (len(train_loader.dataset) // self.batch_size_train) if self.batch_size_train > 0 else 1
            strategy = "auto"
            accelerator = "auto"
            devices = "auto"
        
        if loging_steps == 0:
            loging_steps = 1 # log at least once per epoch.

        callbacks_list = [
            ModelSummary(max_depth=-1),
            self.plotting_callback,
            self.checkpoint_callback,
            self.loss_plot_callback,
            self.early_stop_callback,
        ]

        # Instantiate the PyTorch Lightning Trainer
        trainer = L.Trainer(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            log_every_n_steps=loging_steps,
            gradient_clip_val=0.5,
            max_epochs=self.epochs_num,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=os.path.normpath(self.train_results_dir),
            profiler=profiler,
            num_sanity_val_steps=0,
            callbacks=callbacks_list,
            precision="16-mixed",
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

        # Find optimal learning rate
        logger.info("Finding optimal learning rate using PyTorch Lightning's tuner...")
        tuner = Tuner(trainer)

        # Run learning rate finder
        lr_finder = tuner.lr_find(
            self.lightning_base_model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader
        )

        # Get suggestion and update model
        if lr_finder and lr_finder.suggestion():
            new_lr = lr_finder.suggestion()
            self.lightning_base_model.hparams.lr = new_lr
            self.lightning_base_model.lr = new_lr  # Also update attribute if used directly
            logger.info(f"Found new optimal learning rate: {new_lr}")

            # Plot results
            fig = lr_finder.plot(suggest=True)
            plot_path = os.path.join(self.train_results_dir, 'lr_finder_plot.png')
            fig.savefig(plot_path)
            plt.close(fig)
            logger.info(f"Learning rate finder plot saved to {plot_path}")
        else:
            logger.warning("Could not find a new learning rate. Using the one from config.")

        logger.info(f"Starting training of the base model for {self.epochs_num} epochs.")
        trainer.fit(
            self.lightning_base_model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader
        )
        logger.info("Finished training the base model.")

        # Save training history
        training_hist = self.loss_plot_callback.history
        path_save_hist = os.path.join(self.train_results_dir, 'base_model_history.pkl')
        with open(path_save_hist, 'wb') as f:
            pickle.dump(training_hist, f)
        
        logger.info(f"Training history saved to {path_save_hist}")

        return training_hist

    def train_base_model_pytorch(self, train_loader, validation_loader):
        """
        Trains the base SeqVaeTeb model using a pure PyTorch DDP setup.

        This function manually implements the training loop, including multi-GPU
        support via DDP, checkpointing, early stopping, and loss plotting.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            validation_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            dict: A dictionary containing the training history.
        """
        logger.info("Setting up PyTorch DDP training for the base model with optimized loss computation...")

        is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        if is_distributed:
            # In a distributed setup, device is the local rank
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = local_rank
            torch.cuda.set_device(device)
            self.base_model.to(device)
            # Find all buffer tensors and move them to the correct device
            for buffer in self.base_model.buffers():
                buffer.to(device)
            # Wrap the model with DDP
            model = DDP(self.base_model, device_ids=[device], find_unused_parameters=False)
            rank = dist.get_rank()
        else:
            # For single GPU or CPU
            rank = 0
            device = f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu"
            self.base_model.to(device)
            model = self.base_model

        # Use AdamW with weight decay for better optimization
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4, eps=1e-8)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.5)
        
        # Enable mixed precision for faster training
        scaler = torch.cuda.amp.GradScaler() if device != "cpu" else None

        # Re-create callbacks/utilities for PyTorch loop
        loss_plotter = LossPlotCallback(output_dir=self.train_results_dir, plot_frequency=1)
        best_val_loss = float('inf')
        early_stop_patience = 100
        patience_counter = 0

        logger.info(f"Starting optimized PyTorch DDP training on device: {device}")
        logger.info(f"Using separate loss computation and backward passes for optimal gradient flow")
        logger.info(f"Mixed precision enabled: {scaler is not None}")
        logger.info(f"Gradient clipping enabled with max_norm=1.0")
        logger.info(f"Using AdamW optimizer with weight_decay=1e-4")
        
        # Correctly get the underlying model in both DDP and single-GPU cases
        plain_model = model.module if is_distributed else model

        for epoch in range(self.epochs_num):
            model.train()
            
            # Note: A distributed sampler is automatically applied by the trainer,
            # so we don't need to manually set the epoch.
            
            train_loss_dict = {
                'total_loss': 0.0, 'recon_loss': 0.0, 'kld_loss': 0.0,
                'scattering_loss': 0.0, 'phase_loss': 0.0
            }
            
            # --- Training Loop ---
            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs_num} [Train]", disable=(rank != 0)):
                # Access data using correct HDF5 dataset field names
                y_st = batch_data.fhr_st.to(device)      # Scattering transform features
                y_ph = batch_data.fhr_ph.to(device)      # Phase harmonic features  
                x_ph = batch_data.fhr_up_ph.to(device)   # Cross-phase features

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        forward_outputs = model(y_st, y_ph, x_ph)
                        
                        # Compute all losses separately for optimal gradient flow
                        scattering_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=True,
                            compute_phase_loss=False,
                            compute_kld_loss=False
                        )
                        
                        phase_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=False,
                            compute_phase_loss=True,
                            compute_kld_loss=False
                        )
                        
                        kld_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=False,
                            compute_phase_loss=False,
                            compute_kld_loss=True
                        )
                    
                    # Separate backward passes with gradient scaling
                    if scattering_loss_dict['scattering_loss'].item() > 0:
                        scaler.scale(scattering_loss_dict['scattering_loss']).backward(retain_graph=True)
                    
                    if phase_loss_dict['phase_loss'].item() > 0:
                        scaler.scale(phase_loss_dict['phase_loss']).backward(retain_graph=True)
                    
                    if kld_loss_dict['kld_loss'].item() > 0:
                        scaler.scale(kld_loss_dict['kld_loss']).backward()
                    
                    # Gradient clipping and optimizer step with scaling
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Forward pass (shared computation)
                    forward_outputs = model(y_st, y_ph, x_ph)
                    
                    # Separate loss computation and backward passes for optimal PyTorch performance
                    # 1. Scattering loss backward pass
                    scattering_loss_dict = plain_model.compute_loss(
                        forward_outputs, y_st, y_ph,
                        compute_scattering_loss=True,
                        compute_phase_loss=False,
                        compute_kld_loss=False
                    )
                    if scattering_loss_dict['scattering_loss'].item() > 0:
                        scattering_loss_dict['scattering_loss'].backward(retain_graph=True)
                    
                    # 2. Phase loss backward pass
                    phase_loss_dict = plain_model.compute_loss(
                        forward_outputs, y_st, y_ph,
                        compute_scattering_loss=False,
                        compute_phase_loss=True,
                        compute_kld_loss=False
                    )
                    if phase_loss_dict['phase_loss'].item() > 0:
                        phase_loss_dict['phase_loss'].backward(retain_graph=True)
                    
                    # 3. KLD loss backward pass
                    kld_loss_dict = plain_model.compute_loss(
                        forward_outputs, y_st, y_ph,
                        compute_scattering_loss=False,
                        compute_phase_loss=False,
                        compute_kld_loss=True
                    )
                    if kld_loss_dict['kld_loss'].item() > 0:
                        kld_loss_dict['kld_loss'].backward()
                    
                    # Gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Accumulate losses for logging
                train_loss_dict['scattering_loss'] += scattering_loss_dict['scattering_loss'].item()
                train_loss_dict['phase_loss'] += phase_loss_dict['phase_loss'].item()
                train_loss_dict['kld_loss'] += kld_loss_dict['kld_loss'].item()
                train_loss_dict['recon_loss'] += (scattering_loss_dict['scattering_loss'] + phase_loss_dict['phase_loss']).item()
                train_loss_dict['total_loss'] += (scattering_loss_dict['scattering_loss'] + phase_loss_dict['phase_loss'] + kld_loss_dict['kld_loss']).item()
            
            # --- Validation Loop ---
            model.eval()
            val_loss_dict = {
                'total_loss': 0.0, 'recon_loss': 0.0, 'kld_loss': 0.0,
                'scattering_loss': 0.0, 'phase_loss': 0.0
            }
            with torch.no_grad():
                for batch_data in tqdm(validation_loader, desc=f"Epoch {epoch+1}/{self.epochs_num} [Val]", disable=(rank != 0)):
                    # Access data using correct HDF5 dataset field names
                    y_st = batch_data.fhr_st.to(device)      # Scattering transform features
                    y_ph = batch_data.fhr_ph.to(device)      # Phase harmonic features  
                    x_ph = batch_data.fhr_up_ph.to(device)   # Cross-phase features
                    
                    # Use mixed precision for validation if available
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            forward_outputs = model(y_st, y_ph, x_ph)
                            
                            # Compute all losses separately for validation (no backward pass needed)
                            scattering_loss_dict = plain_model.compute_loss(
                                forward_outputs, y_st, y_ph,
                                compute_scattering_loss=True,
                                compute_phase_loss=False,
                                compute_kld_loss=False
                            )
                            phase_loss_dict = plain_model.compute_loss(
                                forward_outputs, y_st, y_ph,
                                compute_scattering_loss=False,
                                compute_phase_loss=True,
                                compute_kld_loss=False
                            )
                            kld_loss_dict = plain_model.compute_loss(
                                forward_outputs, y_st, y_ph,
                                compute_scattering_loss=False,
                                compute_phase_loss=False,
                                compute_kld_loss=True
                            )
                    else:
                        forward_outputs = model(y_st, y_ph, x_ph)
                        
                        # Compute all losses separately for validation (no backward pass needed)
                        scattering_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=True,
                            compute_phase_loss=False,
                            compute_kld_loss=False
                        )
                        phase_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=False,
                            compute_phase_loss=True,
                            compute_kld_loss=False
                        )
                        kld_loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph,
                            compute_scattering_loss=False,
                            compute_phase_loss=False,
                            compute_kld_loss=True
                        )
                    
                    # Accumulate losses
                    val_loss_dict['scattering_loss'] += scattering_loss_dict['scattering_loss'].item()
                    val_loss_dict['phase_loss'] += phase_loss_dict['phase_loss'].item()
                    val_loss_dict['kld_loss'] += kld_loss_dict['kld_loss'].item()
                    val_loss_dict['recon_loss'] += (scattering_loss_dict['scattering_loss'] + phase_loss_dict['phase_loss']).item()
                    val_loss_dict['total_loss'] += (scattering_loss_dict['scattering_loss'] + phase_loss_dict['phase_loss'] + kld_loss_dict['kld_loss']).item()
            
            scheduler.step()

            # --- Logging & Checkpointing (on rank 0) ---
            if rank == 0:
                # Average losses
                for k in train_loss_dict: train_loss_dict[k] /= len(train_loader)
                for k in val_loss_dict: val_loss_dict[k] /= len(validation_loader)

                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {train_loss_dict['total_loss']:.4f} "
                    f"(Scat: {train_loss_dict['scattering_loss']:.4f}, "
                    f"Phase: {train_loss_dict['phase_loss']:.4f}, "
                    f"KLD: {train_loss_dict['kld_loss']:.4f}), "
                    f"Val Loss: {val_loss_dict['total_loss']:.4f} "
                    f"(Scat: {val_loss_dict['scattering_loss']:.4f}, "
                    f"Phase: {val_loss_dict['phase_loss']:.4f}, "
                    f"KLD: {val_loss_dict['kld_loss']:.4f})"
                )
                
                # Update history for plotting
                loss_plotter.history['epoch'].append(epoch)
                for k,v in train_loss_dict.items(): loss_plotter.history[f'train/{k}'].append(v)
                for k,v in val_loss_dict.items(): loss_plotter.history[f'val/{k}'].append(v)
                loss_plotter.plot_losses()

                # Checkpointing
                if val_loss_dict['total_loss'] < best_val_loss:
                    best_val_loss = val_loss_dict['total_loss']
                    patience_counter = 0
                    save_path = os.path.join(self.model_checkpoint_dir, "base-model-best-pytorch.pt")
                    torch.save(plain_model.state_dict(), save_path)
                    logger.info(f"Saved new best model to {save_path} with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1

                # Early Stopping
                if patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered.")
                    break
        
        if rank == 0:
            logger.info("Finished training the base model with PyTorch DDP.")
            training_hist = loss_plotter.history
            path_save_hist = os.path.join(self.train_results_dir, 'base_model_history_pytorch.pkl')
            with open(path_save_hist, 'wb') as f: pickle.dump(training_hist, f)
            logger.info(f"Training history saved to {path_save_hist}")
            return training_hist
        
        return None

    def train_seqvae_model(self, train_loader_seqvae=None,
                           validation_loader_seqvae=None):
        self.early_stop_callback = EarlyStopping(
            monitor="validation_loss",
            min_delta=0.0,
            patience=100,
            verbose=True,
            mode="min"
        )
        self.checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            mode="min",
            dirpath=self.model_checkpoint_dir,
            filename="best_checkpoint-{epoch:02d}-{validation_loss:.2f}",
            save_top_k=1,
            save_last=True,
        )
        self.loss_plot_callback = LossPlotCallback(output_dir=self.train_results_dir, plot_frequency=self.plot_every_epoch)
        profiler_g = SimpleProfiler(dirpath=self.train_results_dir, filename="profiler_log.txt")
        if len(self.cuda_devices) > 0:
            loging_steps = (len(train_loader_seqvae.dataset) // self.batch_size_train) // len(self.cuda_devices)
        else:
            loging_steps = (len(train_loader_seqvae.dataset) // self.batch_size_train)

        callbacks_list = [
            ModelSummary(max_depth=-1),
            self.plotting_callback,
            self.checkpoint_callback,
            self.loss_plot_callback
        ]
        if self.train_classifier:
            callbacks_list.append(self.metrics_callback)
            callbacks_list.append(self.classification_performance_callback)
            callbacks_list.append(self.early_stop_callback)

        trainer_graph_model = L.Trainer(
            devices=self.cuda_devices,
            log_every_n_steps=loging_steps,
            gradient_clip_val=0.5,
            accelerator="gpu",
            max_epochs=self.epochs_num,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=os.path.normpath(self.train_results_dir),
            profiler=profiler_g,
            num_sanity_val_steps=0,
            # accumulate_grad_batches=2,
            callbacks=callbacks_list,
            strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="gloo" if sys.platform == "win32" else "nccl")
        )

        # tuner = Tuner(trainer_graph_model)
        # lr_finder = tuner.lr_find(self.seqvae_lightning_model, train_loader_seqvae, validation_loader_seqvae)
        # logger.info(lr_finder.results)
        # fig = lr_finder.plot(suggest=True)
        # plt.savefig(os.path.join(self.train_results_dir, "lr.png"))
        # plt.close()

        trainer_graph_model.fit(self.seqvae_lightning_model, train_dataloaders=train_loader_seqvae,
                                val_dataloaders=validation_loader_seqvae)
        logger.info('=' * 50)
        training_hist = self.loss_plot_callback.history
        path_save_hist = os.path.join(self.train_results_dir, 'history.pkl')
        with open(path_save_hist, 'wb') as f:
            pickle.dump(training_hist, f)
        return training_hist


    def seqvae_prediction_plot(self, dataloader, prediction_idx, device):
        self.pytorch_model.eval()
        num_predictions_ = int((300 - prediction_idx) / 30)
        for idx, batched_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_data = batched_data[0].to(device)
            guids_list = batched_data[2]
            epoch_nums_list = batched_data[3].to(device)
            save_dir_prediction = os.path.join(self.test_results_dir, 'predictions_st')
            os.makedirs(save_dir_prediction, exist_ok=True)

            predicted_list_mean = []
            predicted_list_var = []
            selected_idx = [1, 10, 20, 30, 35, 58, 62, 29, 50, 60, 69, 100, 119, 169, 170, 179, 190]
            with torch.no_grad():
                vline_indices = []
                for j in range(num_predictions_):
                    prediction_idx_m = prediction_idx + (j*30)
                    vline_indices.append(prediction_idx_m)
                    scattering_original, prediction_mean, prediction_logvar = \
                        self.pytorch_model.predict_next(input_data, prediction_index=prediction_idx_m, epoch_num=epoch_nums_list, zero_source=self.zero_source)
                    predicted_list_mean.append(prediction_mean)
                    predicted_list_var.append(torch.exp(prediction_logvar))
                prediction_mean = torch.cat(predicted_list_mean, dim=2)
                prediction_var = torch.cat(predicted_list_var, dim=2)
                # prediction_mean = prediction_mean.permute(0, 2, 1)
                for k in selected_idx:
                    try:
                        plot_prediction_st(input_data[k].unsqueeze(-1).detach().cpu().numpy(),
                                           sx=scattering_original[k].permute(1, 0).detach().cpu().numpy(),
                                           sx_pmean=prediction_mean[k].detach().cpu().numpy(),
                                           sx_pvar=prediction_var[k].detach().cpu().numpy(),
                                           plot_second_channel= (self.input_channel_num==2),
                                           plot_dir=save_dir_prediction,
                                           prediction_idx=prediction_idx,
                                           vline_indices=vline_indices,
                                           plot_title=f'{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}',
                                           tag=f'{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}')
                    except Exception as e:
                        logger.info(f'{e}')

                result = self.pytorch_model(input_data, epoch_num=epoch_nums_list, zero_source=self.zero_source)
                # if window is not None:
                #     sxr = result.decoder_mean[0].detach().cpu().numpy()
                #     sxr_std = result.decoder_std[0].detach().cpu().numpy()
                save_dir_prediction = os.path.join(self.test_results_dir, 'test_results')
                os.makedirs(save_dir_prediction, exist_ok=True)
                for k in selected_idx:
                    try:
                        plot_forward_pass(signal=input_data[k].detach().cpu().numpy(),
                                          fhr_st=result.sx.permute(1, 2, 0)[k][:, :].detach().cpu().numpy(),
                                          meta=None,
                                          plot_second_channel=(self.input_channel_num == 2),
                                          fhr_st_pr=result.decoder_mean[k][:, :].detach().cpu().numpy(),
                                          Sxr_std=result.decoder_std[k][:, :].detach().cpu().numpy(),
                                          z_latent=result.z_latent[k][:, :].detach().cpu().numpy(),
                                          plot_dir=save_dir_prediction,
                                          plot_title=f"",
                                          tag=f'{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}')
                        plt.close('all')
                    except Exception as e:
                        logger.info(f'{e}')

    def test_seqvae_torch_model(self, dataloader, device):
        save_dir_prediction = os.path.join(self.test_results_dir, 'testing_kls')
        os.makedirs(save_dir_prediction, exist_ok=True)
        self.pytorch_model.eval()
        with torch.no_grad():
            for idx, batched_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_data = batched_data[0].to(device)
                guids_list = batched_data[2]
                epoch_nums_list = batched_data[3].to(device)
                result = self.pytorch_model(input_data, zero_source=False, epoch_num=epoch_nums_list)
                result_no_source = self.pytorch_model(input_data, zero_source=True, epoch_num=epoch_nums_list)
                selected_idx = [119, 169, 170, 179, 190]
                for k in selected_idx:
                    try:
                        kld_diff_mean = result.decoder_mean[k][:, 3:] - result_no_source.decoder_mean[k][:, 3:]
                        kld_diff_std = result.decoder_std[k][:, 3:] - result_no_source.decoder_std[k][:, 3:]
                        latent_diff = result.z_latent[k][:, 3:] - result_no_source.z_latent[k][:, 3:]
                        kld_kld_diff = result.kld_values[k][:, 3:] - result_no_source.kld_values[k][:, 3:]
                        plot_forward_pass_kld(signal=input_data[k].detach().cpu().numpy(),
                                              Sx=result.sx.permute(1, 2, 0)[k][:, 3:].detach().cpu().numpy(),
                                              meta=None,
                                              plot_second_channel=(self.input_channel_num == 2),
                                              Sxr=result.decoder_mean[k][:, 3:].detach().cpu().numpy(),
                                              Sxr_std=result.decoder_std[k][:, 3:].detach().cpu().numpy(),
                                              z_latent=result.z_latent[k][:, 3:].detach().cpu().numpy(),
                                              plot_dir=save_dir_prediction,
                                              plot_title=f"{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}",
                                              kld_elements=result.kld_values[k][:, 3:].detach().cpu().numpy(),
                                              tag=f'{k}--{idx}-with-source',)
                        plot_forward_pass_kld(signal=input_data[k].detach().cpu().numpy(),
                                              Sx=result_no_source.sx.permute(1, 2, 0)[k][:, 3:].detach().cpu().numpy(),
                                              meta=None,
                                              plot_second_channel=(self.input_channel_num == 2),
                                              Sxr=result_no_source.decoder_mean[k][:, 3:].detach().cpu().numpy(),
                                              Sxr_std=result_no_source.decoder_std[k][:, 3:].detach().cpu().numpy(),
                                              z_latent=result_no_source.z_latent[k][:, 3:].detach().cpu().numpy(),
                                              plot_dir=save_dir_prediction,
                                              plot_title=f"{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}",
                                              kld_elements=result_no_source.kld_values[k][:, 3:].detach().cpu().numpy(),
                                              tag=f'{k}--{idx}-without-source')

                        plot_forward_pass_kld(signal=input_data[k].detach().cpu().numpy(),
                                              Sx=result.sx.permute(1, 2, 0)[k][:, 3:].detach().cpu().numpy(),
                                              meta=None,
                                              plot_second_channel=(self.input_channel_num == 2),
                                              Sxr=kld_diff_mean.detach().cpu().numpy(),
                                              Sxr_std=kld_diff_std.detach().cpu().numpy(),
                                              z_latent=latent_diff.detach().cpu().numpy(),
                                              plot_dir=save_dir_prediction,
                                              plot_title=f"{guids_list[k]}-{epoch_nums_list[k].item()}-{idx}",
                                              kld_elements=kld_kld_diff.detach().cpu().numpy(),
                                              tag=f'{k}--{idx}-difference')
                    except Exception as e:
                        logger.info(e)
                logger.info('done')


    def seqvae_mse_test(self, seqvae_mse_test_dataloader,  tag="error_stats", device=None):
        base_dir = self.test_results_dir
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        mse_all_list = []
        mse_energy_norm_list = []
        vaf_all_list = []
        log_likelihood_list = []
        st_list = []
        snr_all_list = []

        with torch.no_grad():
            for idx, batched_data in tqdm(enumerate(seqvae_mse_test_dataloader), total=len(seqvae_mse_test_dataloader)):
                input_data = batched_data[0].to(device)
                guids_list = batched_data[2]
                epoch_nums_list = batched_data[3].to(device)
                results_t = self.pytorch_model(input_data, zero_source=self.zero_source, epoch_num=epoch_nums_list)
                dec_mean_t_ = results_t.decoder_mean[:, :, 20:280]  # (batch, input_dim, length)
                dec_std_t_ = torch.sqrt(torch.exp(results_t.decoder_std))[:, :, 20:280]
                sx_t_ = results_t.sx.permute(1, 2, 0)[:, :, 20:280]  # (batch, input_dim, length)
                # MSE per channel
                mse_per_ce = torch.mean((sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
                # Energy of the original signal
                energy_per_coeff = torch.mean(sx_t_ ** 2, dim=2)  # (batch, input_dim)

                # Energy-normalized MSE
                energy_normalized_mse = mse_per_ce / (energy_per_coeff + 1e-12)

                # VAF calculation
                _, vaf = calculate_vaf(sx_t_, dec_mean_t_)  # (input_dim,)
                # vaf = vaf.unsqueeze(0)  # make it (1, input_dim) for concatenation

                # Log-likelihood calculation
                log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, sx_t_)

                # SNR calculation (in dB)
                signal_power = torch.mean(sx_t_ ** 2, dim=2)  # (batch, input_dim)
                noise_power = torch.mean((sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
                snr = 10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12))  # (batch, input_dim)

                # Accumulate results
                mse_all_list.append(mse_per_ce)
                mse_energy_norm_list.append(energy_normalized_mse)
                vaf_all_list.append(vaf)
                log_likelihood_list.extend(log_likelihoods)
                st_list.append(sx_t_)
                snr_all_list.append(snr)

        tag_hist = tag + 'loglikelihood_'
        save_dir_hist = os.path.join(base_dir, tag_hist)
        os.makedirs(save_dir_hist, exist_ok=True)

        # Concatenate all data
        mse_all_data = torch.cat(mse_all_list, dim=0)  # (N, input_dim)
        mse_energy_normalized = torch.cat(mse_energy_norm_list, dim=0)  # (N, input_dim)
        vaf_all_data = torch.cat(vaf_all_list, dim=0)  # (N, input_dim)
        all_st_tensor = torch.cat(st_list, dim=0)  # (N, input_dim, length)
        snr_all_data = torch.cat(snr_all_list, dim=0)  # (N, input_dim)
        save_path_ttest = os.path.join(save_dir_hist, f'{tag}-snr-t-test.npy')
        np.save(save_path_ttest, snr_all_data.detach().cpu().numpy())
        # Mean and std of the entire dataset
        all_st_mean = all_st_tensor.mean(dim=0)  # (input_dim, length)
        all_st_std = all_st_tensor.std(dim=0)  # (input_dim, length)

        # Plot distributions of Sx
        plot_distributions(
            sx_mean=all_st_mean.detach().cpu().numpy(),
            sx_std=all_st_std.detach().cpu().numpy(),
            plot_second_channel=False,
            plot_sample=False,
            plot_dir=save_dir_hist,
            plot_dataset_average=True,
            tag='st_mean'
        )

        # Plot histogram of log-likelihood
        plot_histogram(
            data=np.array(log_likelihood_list),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='loglikelihood_original'
        )

        # Save VAF data
        vaf_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels.npy')
        np.save(vaf_path, vaf_all_data.detach().cpu().numpy())

        # Averages across channels for MSE
        mse_all_data_averaged = torch.mean(mse_all_data, dim=1)  # (N,)
        mse_energy_normalized_averaged = torch.mean(mse_energy_normalized, dim=1)  # (N,)

        # Save MSE averaged data
        mse_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_averaged.npy')
        np.save(mse_avg_path, mse_all_data_averaged.detach().cpu().numpy())

        mse_norm_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_normalized_averaged.npy')
        np.save(mse_norm_avg_path, mse_energy_normalized_averaged.detach().cpu().numpy())

        # Plot histograms for MSE distributions
        plot_histogram(
            data=mse_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='mse-all_dist'
        )

        plot_histogram(
            data=mse_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='mse-all-data-per'
        )

        # ---------- NEW: SNR and VAF single-channel averaged distributions ----------

        # SNR averaged per sample
        snr_all_data_averaged = torch.mean(snr_all_data, dim=1)  # (N,)
        snr_hist_path = os.path.join(save_dir_hist, f'{tag}-snr_all_data.npy')
        np.save(snr_hist_path, snr_all_data.detach().cpu().numpy())

        # Plot SNR histogram for all data (per-channel)
        plot_histogram(
            data=snr_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='snr-all-data-per'
        )

        # Plot SNR histogram averaged over channels (similar to mse-all_dist)
        plot_histogram(
            data=snr_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='snr-all_dist'
        )

        # VAF averaged per sample
        vaf_all_data_averaged = torch.mean(vaf_all_data, dim=1)  # (N,)
        vaf_hist_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels_averaged.npy')
        np.save(vaf_hist_path, vaf_all_data_averaged.detach().cpu().numpy())

        # Plot VAF histogram for all data (per-channel)
        plot_histogram(
            data=vaf_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='vaf-all-data-per'
        )

        # Plot VAF histogram averaged over channels (similar to mse-all_dist)
        plot_histogram(
            data=vaf_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='vaf-all_dist'
        )

        return all_st_tensor

    # todo: you can make one function for accuracy analysis and combine both
    def seqvae_prediction_accuracy_test(self, seqvae_mse_test_dataloader,  tag="prediction_error_stats", prediction_idx=30, device=None):
        base_dir = self.test_results_dir
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        mse_all_list = []
        mse_energy_norm_list = []
        vaf_all_list = []
        log_likelihood_list = []
        st_list = []
        snr_all_list = []
        num_predictions = int((300 - prediction_idx) / 30)
        with torch.no_grad():
            for idx, batched_data in tqdm(enumerate(seqvae_mse_test_dataloader), total=len(seqvae_mse_test_dataloader)):
                input_data = batched_data[0].to(device)
                guids_list = batched_data[2]
                epoch_nums_list = batched_data[3].to(device)
                predicted_list_mean = []
                predicted_list_var = []
                for j in range(num_predictions):
                    prediction_idx_m = prediction_idx + (j * 30)
                    scattering_original, prediction_mean, prediction_logvar = \
                        self.pytorch_model.predict_next(input_data, prediction_index=prediction_idx_m, epoch_num=epoch_nums_list, zero_source=self.zero_source)
                    predicted_list_mean.append(prediction_mean)
                    predicted_list_var.append(torch.exp(prediction_logvar))
                dec_mean_t_ = torch.cat(predicted_list_mean, dim=2)  # (batch, input_dim, length)
                dec_std_t_ = torch.cat(predicted_list_var, dim=2)
                sx_t_ = scattering_original.permute(0, 2, 1)[:, :, prediction_idx:]
                # MSE per channel
                mse_per_ce = torch.mean((sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
                # Energy of the original signal
                energy_per_coeff = torch.mean(sx_t_ ** 2, dim=2)  # (batch, input_dim)

                # Energy-normalized MSE
                energy_normalized_mse = mse_per_ce / (energy_per_coeff + 1e-12)

                # VAF calculation
                _, vaf = calculate_vaf(sx_t_, dec_mean_t_)  # (input_dim,)
                # vaf = vaf.unsqueeze(0)  # make it (1, input_dim) for concatenation

                # Log-likelihood calculation
                log_likelihoods = calculate_log_likelihood(dec_mean_t_, dec_std_t_, sx_t_)

                # SNR calculation (in dB)
                signal_power = torch.mean(sx_t_ ** 2, dim=2)  # (batch, input_dim)
                noise_power = torch.mean((sx_t_ - dec_mean_t_) ** 2, dim=2)  # (batch, input_dim)
                snr = 10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12))  # (batch, input_dim)

                # Accumulate results
                mse_all_list.append(mse_per_ce)
                mse_energy_norm_list.append(energy_normalized_mse)
                vaf_all_list.append(vaf)
                log_likelihood_list.extend(log_likelihoods)
                st_list.append(sx_t_)
                snr_all_list.append(snr)

        tag_hist = tag + 'loglikelihood_'
        save_dir_hist = os.path.join(base_dir, tag_hist)
        os.makedirs(save_dir_hist, exist_ok=True)

        # Concatenate all data
        mse_all_data = torch.cat(mse_all_list, dim=0)  # (N, input_dim)
        mse_energy_normalized = torch.cat(mse_energy_norm_list, dim=0)  # (N, input_dim)
        vaf_all_data = torch.cat(vaf_all_list, dim=0)  # (N, input_dim)
        all_st_tensor = torch.cat(st_list, dim=0)  # (N, input_dim, length)
        snr_all_data = torch.cat(snr_all_list, dim=0)  # (N, input_dim)
        save_path_ttest = os.path.join(save_dir_hist, f'{tag}-snr-t-test.npy')
        np.save(save_path_ttest, snr_all_data.detach().cpu().numpy())
        # Mean and std of the entire dataset
        all_st_mean = all_st_tensor.mean(dim=0)  # (input_dim, length)
        all_st_std = all_st_tensor.std(dim=0)  # (input_dim, length)

        # Plot distributions of Sx
        plot_distributions(
            sx_mean=all_st_mean.detach().cpu().numpy(),
            sx_std=all_st_std.detach().cpu().numpy(),
            plot_second_channel=False,
            plot_sample=False,
            plot_dir=save_dir_hist,
            plot_dataset_average=True,
            tag='st_mean'
        )

        # Plot histogram of log-likelihood
        plot_histogram(
            data=np.array(log_likelihood_list),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='loglikelihood_original'
        )

        # Save VAF data
        vaf_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels.npy')
        np.save(vaf_path, vaf_all_data.detach().cpu().numpy())

        # Averages across channels for MSE
        mse_all_data_averaged = torch.mean(mse_all_data, dim=1)  # (N,)
        mse_energy_normalized_averaged = torch.mean(mse_energy_normalized, dim=1)  # (N,)

        # Save MSE averaged data
        mse_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_averaged.npy')
        np.save(mse_avg_path, mse_all_data_averaged.detach().cpu().numpy())

        mse_norm_avg_path = os.path.join(save_dir_hist, f'{tag}-mse_all_data_normalized_averaged.npy')
        np.save(mse_norm_avg_path, mse_energy_normalized_averaged.detach().cpu().numpy())

        # Plot histograms for MSE distributions
        plot_histogram(
            data=mse_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='mse-all_dist'
        )

        plot_histogram(
            data=mse_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='mse-all-data-per'
        )

        # ---------- NEW: SNR and VAF single-channel averaged distributions ----------

        # SNR averaged per sample
        snr_all_data_averaged = torch.mean(snr_all_data, dim=1)  # (N,)
        snr_hist_path = os.path.join(save_dir_hist, f'{tag}-snr_all_data.npy')
        np.save(snr_hist_path, snr_all_data.detach().cpu().numpy())

        # Plot SNR histogram for all data (per-channel)
        plot_histogram(
            data=snr_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='snr-all-data-per'
        )

        # Plot SNR histogram averaged over channels (similar to mse-all_dist)
        plot_histogram(
            data=snr_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='snr-all_dist'
        )

        # VAF averaged per sample
        vaf_all_data_averaged = torch.mean(vaf_all_data, dim=1)  # (N,)
        vaf_hist_path = os.path.join(save_dir_hist, f'{tag}-vaf_all_data_all_channels_averaged.npy')
        np.save(vaf_hist_path, vaf_all_data_averaged.detach().cpu().numpy())

        # Plot VAF histogram for all data (per-channel)
        plot_histogram(
            data=vaf_all_data.detach().cpu().numpy(),
            single_channel=False,
            bins=160,
            save_dir=save_dir_hist,
            tag='vaf-all-data-per'
        )

        # Plot VAF histogram averaged over channels (similar to mse-all_dist)
        plot_histogram(
            data=vaf_all_data_averaged.detach().cpu().numpy(),
            single_channel=True,
            bins=160,
            save_dir=save_dir_hist,
            tag='vaf-all_dist'
        )

        return all_st_tensor


    def do_seqvae_tests(self, test_dataloader):
        self.load_pytorch_checkpoint()
        cuda_device = f"cuda:{self.cuda_devices[0]}"
        self.pytorch_model.to(cuda_device)
        self.seqvae_prediction_accuracy_test(seqvae_mse_test_dataloader=test_dataloader, device=cuda_device)
        self.seqvae_prediction_plot(test_dataloader, 30, device=cuda_device)
        self.test_seqvae_torch_model(test_dataloader, device=cuda_device)
        self.seqvae_mse_test(test_dataloader, tag='__', device=cuda_device)


def main(train_SeqVAE=-1, test_SeqVAE=-1):
    np.random.seed(42)
    torch.manual_seed(42)
    sklearn.utils.check_random_state(42)
    start = time.time()

    config_file_path = 'model/config.yaml'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(project_root, config_file_path)

    config_file_path = os.path.normpath(config_file_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at the resolved path: {config_file_path}")
        logger.error("This might be because the file is missing or the path is incorrect.")
        logger.error(f"The path was set to 'model/config.yaml'.")
        logger.error("Please check your project structure and the config path.")
        sys.exit(1)

    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    # DDP setup: Initialize process group if launched with torchrun
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        # The 'env://' init method is used by default and reads the DDP env vars
        # set by torchrun.
        dist.init_process_group(backend="gloo" if sys.platform == "win32" else "nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Initialized DDP on rank {rank}/{world_size}.")
    else:
        rank = 0
        world_size = 1

    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')

    def resolve_path(p):
        if not p or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(project_root, p))

    if 'dataset_config' in config:
        if 'vae_train_datasets' in config['dataset_config']:
            config['dataset_config']['vae_train_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_train_datasets']]
        if 'vae_test_datasets' in config['dataset_config']:
            config['dataset_config']['vae_test_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_test_datasets']]
        if 'stat_path' in config['dataset_config']:
            config['dataset_config']['stat_path'] = resolve_path(config['dataset_config']['stat_path'])
    
    if 'seqvae_testing' in config and 'test_data_dir' in config['seqvae_testing']:
        config['seqvae_testing']['test_data_dir'] = resolve_path(config['seqvae_testing']['test_data_dir'])
    
    if train_SeqVAE > -1:
        cuda_device_list = config['general_config']['cuda_devices']
        # Dataloader configuration
        dataloader_config = config['dataset_config'].get('dataloader_config', {})
        dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
        # Set num_workers to 0 to avoid multiprocessing pickling issues on Windows
        num_workers = 0  # Changed from dataloader_config.get('num_workers', 4)
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # For distributed training, rank and world_size are now correctly set
        # before this point. The dataloader will use a DistributedSampler if world_size > 1.
        
        train_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_train_datasets'],
            batch_size=config['general_config']['batch_size']['train'],
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        # For validation, we typically don't need a distributed sampler,
        # and run it on a single GPU (rank 0).
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        graph_model.train_base_model(train_loader=train_loader_seqvae, validation_loader=validation_loader_seqvae)

    # Clean up the process group
    if is_ddp:
        dist.destroy_process_group()


def main_pytorch(train_SeqVAE=-1, test_SeqVAE=-1):
    """
    Alternative main function that uses the PyTorch DDP implementation instead of PyTorch Lightning.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    sklearn.utils.check_random_state(42)
    start = time.time()

    config_file_path = 'model/config.yaml'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(project_root, config_file_path)

    config_file_path = os.path.normpath(config_file_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at the resolved path: {config_file_path}")
        logger.error("This might be because the file is missing or the path is incorrect.")
        logger.error(f"The path was set to 'model/config.yaml'.")
        logger.error("Please check your project structure and the config path.")
        sys.exit(1)

    with open(config_file_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    
    # DDP setup: Initialize process group if launched with torchrun
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        # The 'env://' init method is used by default and reads the DDP env vars
        # set by torchrun.
        dist.init_process_group(backend="gloo" if sys.platform == "win32" else "nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Initialized DDP on rank {rank}/{world_size}.")
    else:
        rank = 0
        world_size = 1

    # Set matmul precision for Tensor Cores
    torch.set_float32_matmul_precision('high')

    def resolve_path(p):
        if not p or os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(project_root, p))

    if 'dataset_config' in config:
        if 'vae_train_datasets' in config['dataset_config']:
            config['dataset_config']['vae_train_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_train_datasets']]
        if 'vae_test_datasets' in config['dataset_config']:
            config['dataset_config']['vae_test_datasets'] = [resolve_path(p) for p in config['dataset_config']['vae_test_datasets']]
        if 'stat_path' in config['dataset_config']:
            config['dataset_config']['stat_path'] = resolve_path(config['dataset_config']['stat_path'])
    
    if 'seqvae_testing' in config and 'test_data_dir' in config['seqvae_testing']:
        config['seqvae_testing']['test_data_dir'] = resolve_path(config['seqvae_testing']['test_data_dir'])
    
    if train_SeqVAE > -1:
        cuda_device_list = config['general_config']['cuda_devices']
        # Dataloader configuration
        dataloader_config = config['dataset_config'].get('dataloader_config', {})
        dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
        # Set num_workers to 0 to avoid multiprocessing pickling issues on Windows
        num_workers = 0  # Changed from dataloader_config.get('num_workers', 4)
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # For distributed training, rank and world_size are now correctly set
        # before this point. The dataloader will use a DistributedSampler if world_size > 1.
        
        train_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_train_datasets'],
            batch_size=config['general_config']['batch_size']['train'],
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        # For validation, we typically don't need a distributed sampler,
        # and run it on a single GPU (rank 0).
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        # Use the PyTorch DDP training method instead of PyTorch Lightning
        graph_model.train_base_model_pytorch(train_loader=train_loader_seqvae, validation_loader=validation_loader_seqvae)

    # Clean up the process group
    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    # Set training parameters directly
    use_pytorch_ddp = True  # Set to True to use PyTorch DDP, False for PyTorch Lightning
    train_model = 1  # 1 to train, -1 to skip
    test_model = -1  # 1 to test, -1 to skip
    
    if use_pytorch_ddp:
        main_pytorch(train_SeqVAE=train_model, test_SeqVAE=test_model)
    else:
        main(train_SeqVAE=train_model, test_SeqVAE=test_model)
