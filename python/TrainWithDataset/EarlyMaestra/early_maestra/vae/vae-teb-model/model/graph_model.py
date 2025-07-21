import lightning as L
import sklearn.utils
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.distributed as dist
from lightning.pytorch.tuner import Tuner
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import os
import yaml
from datetime import datetime
import sys
import pickle
from tqdm import tqdm
import time
import numpy as np

from utils.data_utils import \
    plot_distributions, \
    plot_histogram

from utils.graph_model_utils import \
    calculate_log_likelihood, \
    calculate_vaf

from loguru import logger

from pytorch_lightning_modules import *

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from vae_teb_model import SeqVaeTeb
from pytorch_lightning_modules import LightSeqVaeTeb

from torch.optim.lr_scheduler import MultiStepLR

# Add this line to enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"   # set to 1 only in debugging 

matplotlib.use('Agg')

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'

def log_gpu_memory_usage(prefix=""):
    """Log current GPU memory usage for debugging memory issues."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        logger.info(f"{prefix} GPU {device}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
    else:
        logger.info(f"{prefix} CUDA not available")

def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_memory_threshold(threshold_gb=10.0):
    """Check if GPU memory usage exceeds threshold and clear cache if needed."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        if allocated > threshold_gb:
            logger.warning(f"GPU memory usage ({allocated:.2f}GB) exceeds threshold ({threshold_gb}GB). Clearing cache...")
            clear_gpu_memory()
            return True
    return False

def find_optimal_batch_size(model, sample_batch, device, max_batch_size=64, min_batch_size=1):
    """
    Find the optimal batch size that fits in GPU memory.
    
    Args:
        model: The model to test
        sample_batch: A sample batch to use for testing
        device: The device to test on
        max_batch_size: Maximum batch size to try
        min_batch_size: Minimum batch size to try
    
    Returns:
        int: Optimal batch size
    """
    model.eval()
    optimal_batch_size = min_batch_size
    
    for batch_size in range(min_batch_size, max_batch_size + 1, 2):
        try:
            # Clear memory before test
            clear_gpu_memory()
            
            # Create test batch with current batch size
            test_y_st = sample_batch.fhr_st[:batch_size].to(device)
            test_y_ph = sample_batch.fhr_ph[:batch_size].to(device)
            test_x_ph = sample_batch.fhr_up_ph[:batch_size].to(device)
            test_y_raw = sample_batch.fhr[:batch_size].to(device)
            
            # Test forward pass
            with torch.no_grad():
                forward_outputs = model(test_y_st, test_y_ph, test_x_ph)
                loss_dict = model.compute_loss(forward_outputs, test_y_raw, compute_kld_loss=True)
                loss = loss_dict['total_loss']
                
            # Test backward pass (without updating weights)
            loss.backward()
            
            # If we get here, this batch size works
            optimal_batch_size = batch_size
            logger.info(f"Batch size {batch_size} successful")
            
            # Clean up test tensors
            del test_y_st, test_y_ph, test_x_ph, test_y_raw
            del forward_outputs, loss_dict, loss
            clear_gpu_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"Batch size {batch_size} failed: OOM")
                break
            else:
                logger.error(f"Batch size {batch_size} failed with error: {e}")
                break
        except Exception as e:
            logger.error(f"Batch size {batch_size} failed with unexpected error: {e}")
            break
    
    # Use 80% of the maximum working batch size for safety margin
    safe_batch_size = max(1, int(optimal_batch_size * 0.8))
    logger.info(f"Optimal batch size found: {safe_batch_size} (80% of max working size {optimal_batch_size})")
    
    # Reset model to training mode
    model.train()
    clear_gpu_memory()
    
    return safe_batch_size

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
            # self.aux_dir,
            # self.tensorboard_dir
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
        
        # Log initial GPU memory status - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("Initial setup")
        
        # Clear any residual GPU memory - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # clear_gpu_memory()
        
        # Reset memory statistics
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

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
                decimation_factor=16,
            )

            try:
                self.lightning_base_model = LightSeqVaeTeb.load_from_checkpoint(
                    self.base_model_checkpoint,
                    seqvae_teb_model=base_model_for_loading,
                    strict=False 
                )
                self.base_model = self.lightning_base_model.model
                self.pytorch_model = self.base_model  # Set pytorch_model reference
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
                decimation_factor=16,
                warmup_period=30,
            )
            self.lightning_base_model = LightSeqVaeTeb(
                seqvae_teb_model=self.base_model,
                lr=self.lr,
                lr_milestones=self.lr_milestones,
                beta_schedule="constant",
                beta_const_val=self.kld_beta_
            )
            self.pytorch_model = self.base_model  # Set pytorch_model reference

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
        
        # Log memory before setting up training - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("Before training setup")

        self.plotting_callback = PlottingCallBack(
            output_dir=self.train_results_dir,
            plot_every_epoch=self.plot_every_epoch,
            input_channel_num=self.input_channel_num,
        )

        self.metrics_callback = MetricsLoggingCallback()

        # Optimized memory monitoring for smaller batch sizes - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # self.memory_monitor_callback = MemoryMonitorCallback(
        #     threshold_gb=6.0,  # Lower threshold for aggressive cleanup
        #     log_frequency=50   # More frequent monitoring
        # )

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
            filename="base-model-best-{epoch}",
            save_top_k=2,
            save_last=False,
        )

        # Callback for plotting losses using Plotly with memory optimization
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            max_history_size=19900  # Limit history to prevent memory issues
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
            # self.memory_monitor_callback,  # COMMENTED OUT FOR MULTI-GPU PERFORMANCE
            self.plotting_callback,
            self.checkpoint_callback,
            self.loss_plot_callback,
            self.early_stop_callback,
        ]

        # Log memory after callback setup - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("After callback setup")

        # Instantiate the PyTorch Lightning Trainer with memory optimizations
        trainer = L.Trainer(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            log_every_n_steps=loging_steps,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",  # Specify clipping algorithm
            max_epochs=self.epochs_num,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=os.path.normpath(self.train_results_dir),
            profiler=profiler,
            num_sanity_val_steps=0,
            callbacks=callbacks_list,
            precision="16-mixed",
            accumulate_grad_batches=max(self.accumulate_grad_batches, 1),  # Force higher accumulation
            # Enhanced memory optimization settings
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            sync_batchnorm=True if len(self.cuda_devices) > 1 else False,
            detect_anomaly=False,
            deterministic=False,
            benchmark=True,
            # Additional memory optimizations
            enable_model_summary=False,  # Disable to save memory
        )

        # Log memory after trainer setup - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("After trainer setup")

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

            # Clean up lr_finder to free memory
            del lr_finder, fig
        else:
            logger.warning("Could not find a new learning rate. Using the one from config.")

        # Log memory before training starts - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("Before training starts")

        logger.info(f"Starting training of the base model for {self.epochs_num} epochs.")
        trainer.fit(
            self.lightning_base_model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader
        )
        logger.info("Finished training the base model.")

        # Log memory after training completes - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("After training completes")

        # Save training history
        training_hist = self.loss_plot_callback.history
        path_save_hist = os.path.join(self.train_results_dir, 'base_model_history.pkl')
        with open(path_save_hist, 'wb') as f:
            pickle.dump(training_hist, f)
        
        logger.info(f"Training history saved to {path_save_hist}")

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            # In a distributed setup, get rank and determine device from config
            rank = dist.get_rank()
            device_id = self.cuda_devices[rank]
            device = f"cuda:{device_id}"
            torch.cuda.set_device(device)

            # Add diagnostic logging
            logger.info(f"[Process Rank: {rank}] - Running on GPU: {device}")
            self.base_model.to(device)
            # Find all buffer tensors and move them to the correct device
            for buffer in self.base_model.buffers():
                buffer.to(device)
            # Wrap the model with DDP
            model = DDP(self.base_model, device_ids=[device_id], find_unused_parameters=False)
        else:
            # For single GPU or CPU
            rank = 0
            device = f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu"
            logger.info(f"[Single Process] - Running on device: {device}")
            self.base_model.to(device)
            model = self.base_model

        # Optimized AdamW configuration for memory efficiency
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.lr, 
            weight_decay=1e-4, 
            eps=1e-8,
            betas=(0.9, 0.98)  # Slightly modified for VAE training
        )
        
        # Use cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.epochs_num, 
            eta_min=self.lr * 0.01
        )
        
        # Enable mixed precision for faster training
        scaler = torch.amp.GradScaler('cuda') if device != "cpu" else None

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
                'total_loss': 0.0, 'reconstruction_loss': 0.0, 'kld_loss': 0.0,
                'raw_signal_loss': 0.0
            }
            
            # --- Training Loop ---
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs_num} [Train]", disable=(rank != 0))
            for i, batch_data in enumerate(progress_bar):
                # Access data using correct HDF5 dataset field names
                y_st = batch_data.fhr_st.to(device)      # Scattering transform features
                y_ph = batch_data.fhr_ph.to(device)      # Phase harmonic features  
                x_ph = batch_data.fhr_up_ph.to(device)   # Cross-phase features
                y_raw = batch_data.fhr.to(device)        # Raw signal for reconstruction

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        forward_outputs = model(y_st, y_ph, x_ph)
                        
                        # Compute loss with new API for raw signal reconstruction
                        loss_dict = plain_model.compute_loss(
                            forward_outputs, y_raw, compute_kld_loss=True)
                        
                        total_loss = loss_dict['total_loss']

                    # Single backward pass on the combined loss
                    scaler.scale(total_loss).backward()
                    
                    # Gradient clipping and optimizer step with scaling
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                else:
                    # Forward pass (shared computation)
                    forward_outputs = model(y_st, y_ph, x_ph)
                    
                    # Compute loss with new API for raw signal reconstruction
                    loss_dict = plain_model.compute_loss(
                        forward_outputs, y_raw, compute_kld_loss=True)

                    total_loss = loss_dict['total_loss']

                    # Single backward pass
                    total_loss.backward()
                    
                    # Gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Accumulate losses for logging
                reconstruction_loss_item = loss_dict['reconstruction_loss'].item()
                kld_loss_item = loss_dict['kld_loss'].item()
                raw_signal_loss_item = loss_dict['raw_signal_loss'].item()

                train_loss_dict['reconstruction_loss'] += reconstruction_loss_item
                train_loss_dict['kld_loss'] += kld_loss_item
                train_loss_dict['raw_signal_loss'] += raw_signal_loss_item
                train_loss_dict['total_loss'] += total_loss.item()
                
                # Update progress bar
                if rank == 0:
                    lr = scheduler.get_last_lr()[0]
                    num_batches_processed = i + 1
                    postfix_dict = {
                        'lr': f'{lr:.2e}',
                        'recon': f'{train_loss_dict["reconstruction_loss"] / num_batches_processed:.4f}',
                        'kld': f'{train_loss_dict["kld_loss"] / num_batches_processed:.4f}',
                        'total': f'{train_loss_dict["total_loss"] / num_batches_processed:.4f}'
                    }
                    progress_bar.set_postfix(postfix_dict)
            
            # --- Validation Loop ---
            model.eval()
            val_loss_dict = {
                'total_loss': 0.0, 'reconstruction_loss': 0.0, 'kld_loss': 0.0,
                'raw_signal_loss': 0.0
            }
            with torch.no_grad():
                val_progress_bar = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{self.epochs_num} [Val]", disable=(rank != 0))
                for i, batch_data in enumerate(val_progress_bar):
                    # Access data using correct HDF5 dataset field names
                    y_st = batch_data.fhr_st.to(device)      # Scattering transform features
                    y_ph = batch_data.fhr_ph.to(device)      # Phase harmonic features  
                    x_ph = batch_data.fhr_up_ph.to(device)   # Cross-phase features
                    y_raw = batch_data.fhr.to(device)        # Raw signal for reconstruction
                    
                    # Use mixed precision for validation if available
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            forward_outputs = model(y_st, y_ph, x_ph)
                            
                            # Compute loss with new API for raw signal reconstruction
                            loss_dict = plain_model.compute_loss(
                                forward_outputs, y_raw, compute_kld_loss=True)
                    else:
                        forward_outputs = model(y_st, y_ph, x_ph)
                        
                        # Compute loss with new API for raw signal reconstruction
                        loss_dict = plain_model.compute_loss(
                            forward_outputs, y_raw, compute_kld_loss=True)
                    
                    # Accumulate losses
                    reconstruction_loss_item = loss_dict['reconstruction_loss'].item()
                    kld_loss_item = loss_dict['kld_loss'].item()
                    raw_signal_loss_item = loss_dict['raw_signal_loss'].item()
                    current_total_loss = loss_dict['total_loss'].item()

                    val_loss_dict['reconstruction_loss'] += reconstruction_loss_item
                    val_loss_dict['kld_loss'] += kld_loss_item
                    val_loss_dict['raw_signal_loss'] += raw_signal_loss_item
                    val_loss_dict['total_loss'] += current_total_loss

                    if rank == 0:
                        num_batches_processed = i + 1
                        postfix_dict = {
                            'recon': f'{val_loss_dict["reconstruction_loss"] / num_batches_processed:.4f}',
                            'kld': f'{val_loss_dict["kld_loss"] / num_batches_processed:.4f}',
                            'total': f'{val_loss_dict["total_loss"] / num_batches_processed:.4f}'
                        }
                        val_progress_bar.set_postfix(postfix_dict)
            
            # Create tensors of the summed losses for this process
            train_losses_local = torch.tensor(list(train_loss_dict.values()), device=device)
            val_losses_local = torch.tensor(list(val_loss_dict.values()), device=device)

            # Sum losses across all processes
            if is_distributed:
                # Add diagnostic logging to debug potential hangs
                if rank == 0: logger.debug(f"Epoch {epoch+1}: Rank 0 entering all_reduce.")
                dist.all_reduce(train_losses_local, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_losses_local, op=dist.ReduceOp.SUM)
                if rank == 0: logger.debug(f"Epoch {epoch+1}: Rank 0 finished all_reduce.")

            scheduler.step()

            # --- Logging & Checkpointing (on rank 0) ---
            if rank == 0:
                # Average losses over all batches across all GPUs
                world_size_val = dist.get_world_size() if is_distributed else 1
                
                # The length of the dataloader is per-process, so multiply by world_size for total batches
                num_total_train_batches = len(train_loader) * world_size_val
                num_total_val_batches = len(validation_loader) * world_size_val
                
                # Avoid division by zero
                if num_total_train_batches == 0: num_total_train_batches = 1
                if num_total_val_batches == 0: num_total_val_batches = 1

                # Calculate final average losses
                avg_train_losses = {k: v.item() / num_total_train_batches for k, v in zip(train_loss_dict.keys(), train_losses_local)}
                avg_val_losses = {k: v.item() / num_total_val_batches for k, v in zip(val_loss_dict.keys(), val_losses_local)}

                logger.info(
                    f"Epoch {epoch+1}: Train Loss: {avg_train_losses['total_loss']:.4f} "
                    f"(Recon: {avg_train_losses['reconstruction_loss']:.4f}, "
                    f"Raw: {avg_train_losses['raw_signal_loss']:.4f}, "
                    f"KLD: {avg_train_losses['kld_loss']:.4f}), "
                    f"Val Loss: {avg_val_losses['total_loss']:.4f} "
                    f"(Recon: {avg_val_losses['reconstruction_loss']:.4f}, "
                    f"Raw: {avg_val_losses['raw_signal_loss']:.4f}, "
                    f"KLD: {avg_val_losses['kld_loss']:.4f})"
                )
                
                # Update history for plotting
                loss_plotter.history['epoch'].append(epoch)
                for k,v in avg_train_losses.items(): loss_plotter.history[f'train/{k}'].append(v)
                for k,v in avg_val_losses.items(): loss_plotter.history[f'val/{k}'].append(v)
                loss_plotter.plot_losses()

                # Checkpointing
                if avg_val_losses['total_loss'] < best_val_loss:
                    best_val_loss = avg_val_losses['total_loss']
                    patience_counter = 0
                    save_path = os.path.join(self.model_checkpoint_dir, "base-model-best-pytorch.pt")
                    torch.save(plain_model.state_dict(), save_path)
                    logger.info(f"Saved new best model to {save_path} with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1

                # Early Stopping Check
                should_stop_tensor = torch.tensor(0, device=device)
                if patience_counter >= early_stop_patience:
                    logger.info("Early stopping triggered.")
                    should_stop_tensor = torch.tensor(1, device=device)
            else:
                # Other ranks need a placeholder tensor
                should_stop_tensor = torch.tensor(0, device=device)
            
            # Broadcast the stop signal from rank 0 to all other processes
            if is_distributed:
                dist.broadcast(should_stop_tensor, src=0)

            # All processes check the signal and break the loop if needed
            if should_stop_tensor.item() == 1:
                if rank == 0:
                    logger.info("All processes are stopping.")
                break
        
        if rank == 0:
            logger.info("Finished training the base model with PyTorch DDP.")
            training_hist = loss_plotter.history
            path_save_hist = os.path.join(self.train_results_dir, 'base_model_history_pytorch.pkl')
            with open(path_save_hist, 'wb') as f: pickle.dump(training_hist, f)
            logger.info(f"Training history saved to {path_save_hist}")
            return training_hist
        
        return None


    def seqvae_raw_signal_plot(self, dataloader, device):
        """
        Plot raw signal predictions using the new SeqVaeTeb model.
        This replaces the old seqvae_prediction_plot method to work with raw signal architecture.
        """
        if not hasattr(self, 'base_model') or self.base_model is None:
            logger.error("Base model not initialized. Cannot perform raw signal plotting.")
            return
            
        self.base_model.eval()
        save_dir_prediction = os.path.join(self.test_results_dir, 'raw_signal_predictions')
        os.makedirs(save_dir_prediction, exist_ok=True)
        
        selected_idx = [1, 10, 20, 30, 35, 58, 62, 29, 50, 60, 69, 100, 119, 169, 170, 179, 190]
        
        for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Access data using correct HDF5 dataset field names
            y_st = batch_data.fhr_st.to(device)      # Scattering transform features
            y_ph = batch_data.fhr_ph.to(device)      # Phase harmonic features  
            x_ph = batch_data.fhr_up_ph.to(device)   # Cross-phase features
            y_raw = batch_data.fhr.to(device)        # Raw signal ground truth
            guids_list = batch_data.guid if hasattr(batch_data, 'guid') else [f'sample_{i}' for i in range(y_st.size(0))]
            epochs_list = batch_data.epoch if hasattr(batch_data, 'epoch') else torch.zeros(y_st.size(0))
            
            with torch.no_grad():
                # Forward pass to get raw signal predictions
                forward_outputs = self.base_model(y_st, y_ph, x_ph)
                raw_predictions = forward_outputs['raw_predictions']
                
                # Note: raw_predictions now contains (B, 480) single future window predictions
                raw_signal_mu = raw_predictions['raw_signal_mu']  # (B, 480)
                raw_signal_logvar = raw_predictions['raw_signal_logvar']  # (B, 480)
                raw_signal_std = torch.exp(0.5 * raw_signal_logvar)
                z_latent = forward_outputs['z']  # (B, S, latent_dim)
                
                # Get model parameters for plotting
                warmup_period = getattr(self.base_model, 'warmup_period', 30)
                decimation_factor = getattr(self.base_model, 'decimation_factor', 16)
                
                # Plot for selected samples
                for k in selected_idx:
                    if k >= y_raw.size(0):
                        continue
                        
                    try:
                        # Prepare data for plotting
                        y_raw_sample = y_raw[k].squeeze().detach().cpu().numpy()  # Ground truth raw signal
                        
                        # Extract prediction data for this sample
                        pred_mu_future = raw_signal_mu[k].detach().cpu().numpy()  # (480,)
                        pred_std_future = raw_signal_std[k].detach().cpu().numpy()  # (480,)
                        z_sample = z_latent[k].permute(1, 0).detach().cpu().numpy()  # (latent_dim, seq_len)
                        
                        # Create extended arrays for plotting
                        raw_signal_length = len(y_raw_sample)
                        prediction_horizon = len(pred_mu_future)
                        
                        # The model predicts the future window after the sequence ends
                        sequence_end = raw_signal_length - prediction_horizon
                        
                        if sequence_end > 0:
                            # Create extended ground truth and prediction arrays
                            extended_length = raw_signal_length + prediction_horizon
                            extended_ground_truth = np.zeros(extended_length)
                            extended_ground_truth[:raw_signal_length] = y_raw_sample
                            
                            extended_prediction = np.zeros(extended_length)
                            extended_prediction_std = np.zeros(extended_length)
                            extended_prediction[raw_signal_length:] = pred_mu_future
                            extended_prediction_std[raw_signal_length:] = pred_std_future
                            
                            sequence_mask = np.arange(extended_length) < raw_signal_length
                            future_mask = np.arange(extended_length) >= raw_signal_length
                        else:
                            # Fallback for short sequences
                            extended_length = max(raw_signal_length, prediction_horizon)
                            extended_ground_truth = np.zeros(extended_length)
                            extended_ground_truth[:len(y_raw_sample)] = y_raw_sample
                            extended_prediction = np.zeros(extended_length)
                            extended_prediction_std = np.zeros(extended_length)
                            extended_prediction[:len(pred_mu_future)] = pred_mu_future
                            extended_prediction_std[:len(pred_std_future)] = pred_std_future
                            sequence_mask = np.arange(extended_length) < len(y_raw_sample)
                            future_mask = np.arange(extended_length) >= len(y_raw_sample)
                        
                        # Create a comprehensive plot with 5 subplots
                        fig, axes = plt.subplots(5, 1, figsize=(15, 18))
                        
                        # Plot 1: Ground truth raw signal
                        axes[0].plot(y_raw_sample, 'b-', linewidth=1, label='Ground Truth')
                        axes[0].set_title('Ground Truth Raw Signal')
                        axes[0].set_ylabel('Amplitude')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # Plot 2: Future window prediction with uncertainty
                        if np.any(future_mask):
                            # Plot the sequence part in gray
                            sequence_indices = np.arange(extended_length)[sequence_mask]
                            axes[1].plot(sequence_indices, extended_ground_truth[sequence_mask], 'gray', linewidth=1, alpha=0.6, label='Historical Ground Truth')
                            
                            # Plot the future prediction
                            future_pred = extended_prediction[future_mask]
                            future_std = extended_prediction_std[future_mask]
                            future_indices = np.arange(extended_length)[future_mask]
                            
                            axes[1].plot(future_indices, future_pred, 'r-', linewidth=1, label='Future Prediction')
                            axes[1].fill_between(future_indices, 
                                               future_pred - future_std, 
                                               future_pred + future_std, 
                                               alpha=0.3, color='red', label='±1 Std')
                            
                            # Add vertical line to separate sequence from prediction
                            axes[1].axvline(x=raw_signal_length, color='blue', linestyle='--', alpha=0.5, label='Prediction Start')
                            
                        axes[1].set_title('Future Window Prediction with Uncertainty')
                        axes[1].set_ylabel('Amplitude')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                        
                        # Plot 3: Prediction detail view
                        if np.any(future_mask):
                            # Show transition from sequence to prediction
                            transition_samples = 100
                            transition_start = max(0, raw_signal_length - transition_samples)
                            
                            # Plot the transition region
                            axes[2].plot(range(transition_start, raw_signal_length), 
                                        y_raw_sample[transition_start:], 
                                        'b-', linewidth=1, label='Ground Truth (End of Sequence)', alpha=0.8)
                            
                            # Plot the future prediction
                            future_pred = extended_prediction[future_mask]
                            future_std = extended_prediction_std[future_mask]
                            future_indices = np.arange(extended_length)[future_mask]
                            
                            axes[2].fill_between(future_indices, future_pred - future_std, future_pred + future_std,
                                                alpha=0.2, color='red', label='±1 Std')
                            axes[2].plot(future_indices, future_pred, 'r-', linewidth=1, label='Future Prediction', alpha=0.8)
                            
                            # Add vertical line to separate sequence from prediction
                            axes[2].axvline(x=raw_signal_length, color='blue', linestyle='--', alpha=0.7, label='Prediction Boundary')
                            
                        axes[2].set_title('Transition from Sequence to Future Prediction')
                        axes[2].set_ylabel('Amplitude')
                        axes[2].legend()
                        axes[2].grid(True, alpha=0.3)
                        
                        # Plot 4: Full sequence overview
                        axes[3].plot(y_raw_sample, 'b-', linewidth=1, alpha=0.7, label='Ground Truth (Full Sequence)')
                        if np.any(future_mask):
                            future_pred = extended_prediction[future_mask]
                            future_indices = np.arange(extended_length)[future_mask]
                            axes[3].plot(future_indices, future_pred, 'r--', linewidth=1, alpha=0.7, label='Future Prediction')
                            axes[3].axvline(x=raw_signal_length, color='blue', linestyle='--', alpha=0.5, label='Prediction Start')
                        axes[3].set_title('Full Sequence with Future Prediction')
                        axes[3].set_ylabel('Amplitude')
                        axes[3].legend()
                        axes[3].grid(True, alpha=0.3)
                        
                        # Plot 5: Latent representation
                        im = axes[4].imshow(z_sample, aspect='auto', cmap='viridis', interpolation='nearest')
                        axes[4].set_title('Latent Representation')
                        axes[4].set_xlabel('Time Steps')
                        axes[4].set_ylabel('Latent Dimensions')
                        plt.colorbar(im, ax=axes[4])
                        
                        # Overall title
                        guid = guids_list[k] if isinstance(guids_list, list) else guids_list[k].item()
                        epoch = epochs_list[k].item() if hasattr(epochs_list[k], 'item') else epochs_list[k]
                        plt.suptitle(f'Raw Signal Prediction - GUID: {guid}, Epoch: {epoch}, Batch: {idx}\nFuture Window: {prediction_horizon} samples ({prediction_horizon/4.0/60.0:.1f} min)')
                        
                        plt.tight_layout()
                        
                        # Save plot
                        plot_filename = f'raw_signal_pred_{guid}_{epoch}_{idx}_sample_{k}.png'
                        plot_path = os.path.join(save_dir_prediction, plot_filename)
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close('all')
                        
                        logger.info(f"Saved raw signal plot: {plot_path}")
                        
                    except Exception as e:
                        logger.error(f'Error plotting sample {k}: {e}')
                        plt.close('all')
                
                # Cleanup
                del forward_outputs, raw_predictions, raw_signal_mu, raw_signal_logvar, raw_signal_std, z_latent

    def test_raw_signal_model(self, dataloader, device):
        """
        Test raw signal model by analyzing KLD differences and raw signal reconstruction quality.
        This replaces the old test_seqvae_torch_model method.
        """
        if not hasattr(self, 'base_model') or self.base_model is None:
            logger.error("Base model not initialized. Cannot perform testing.")
            return
            
        save_dir_prediction = os.path.join(self.test_results_dir, 'raw_signal_testing')
        os.makedirs(save_dir_prediction, exist_ok=True)
        
        self.base_model.eval()
        selected_idx = [1, 10, 20, 30, 35] # Reduced for efficiency
        
        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Access data using correct HDF5 dataset field names
                y_st = batch_data.fhr_st.to(device)
                y_ph = batch_data.fhr_ph.to(device)
                x_ph = batch_data.fhr_up_ph.to(device)
                y_raw = batch_data.fhr.to(device)
                guids_list = batch_data.guid if hasattr(batch_data, 'guid') else [f'sample_{i}' for i in range(y_st.size(0))]
                epochs_list = batch_data.epoch if hasattr(batch_data, 'epoch') else torch.zeros(y_st.size(0))
                
                # Forward pass with and without source information (zero out x_ph)
                forward_outputs = self.base_model(y_st, y_ph, x_ph)
                forward_outputs_no_source = self.base_model(y_st, y_ph, torch.zeros_like(x_ph))
                
                # Extract predictions
                raw_pred = forward_outputs['raw_predictions']['raw_signal_mu']
                raw_pred_no_source = forward_outputs_no_source['raw_predictions']['raw_signal_mu']
                z_latent = forward_outputs['z']
                z_latent_no_source = forward_outputs_no_source['z']
                
                # Compute differences
                raw_signal_diff = raw_pred - raw_pred_no_source
                latent_diff = z_latent - z_latent_no_source
                
                for k in selected_idx:
                    if k >= y_raw.size(0):
                        continue
                        
                    try:
                        # Prepare data for plotting
                        guid = guids_list[k] if isinstance(guids_list, list) else guids_list[k].item()
                        epoch = epochs_list[k].item() if hasattr(epochs_list[k], 'item') else epochs_list[k]
                        
                        # Create comprehensive comparison plots
                        fig, axes = plt.subplots(5, 1, figsize=(15, 20))
                        
                        # Plot 1: Ground truth
                        y_raw_sample = y_raw[k].squeeze().detach().cpu().numpy()
                        axes[0].plot(y_raw_sample, 'k-', linewidth=1, label='Ground Truth')
                        axes[0].set_title('Ground Truth Raw Signal')
                        axes[0].set_ylabel('Amplitude')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # Plot 2: With source vs without source
                        raw_with_source = raw_pred[k].squeeze().detach().cpu().numpy()
                        raw_without_source = raw_pred_no_source[k].squeeze().detach().cpu().numpy()
                        
                        axes[1].plot(raw_with_source, 'b-', linewidth=1, alpha=0.7, label='With Source')
                        axes[1].plot(raw_without_source, 'r-', linewidth=1, alpha=0.7, label='Without Source')
                        axes[1].set_title('Predicted Raw Signal: With vs Without Source')
                        axes[1].set_ylabel('Amplitude')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                        
                        # Plot 3: Source influence (difference)
                        raw_diff = raw_signal_diff[k].squeeze().detach().cpu().numpy()
                        axes[2].plot(raw_diff, 'g-', linewidth=1, label='Source Influence')
                        axes[2].set_title('Source Influence on Raw Signal Prediction')
                        axes[2].set_ylabel('Amplitude Difference')
                        axes[2].legend()
                        axes[2].grid(True, alpha=0.3)
                        
                        # Plot 4: Latent space differences
                        latent_diff_sample = latent_diff[k].permute(1, 0).detach().cpu().numpy()
                        im1 = axes[3].imshow(latent_diff_sample, aspect='auto', cmap='RdBu_r', 
                                           interpolation='nearest', vmin=-latent_diff_sample.std(), 
                                           vmax=latent_diff_sample.std())
                        axes[3].set_title('Latent Space Difference (With - Without Source)')
                        axes[3].set_ylabel('Latent Dimensions')
                        plt.colorbar(im1, ax=axes[3])
                        
                        # Plot 5: Reconstruction quality comparison
                        axes[4].plot(y_raw_sample, 'k-', linewidth=1.5, alpha=0.8, label='Ground Truth')
                        axes[4].plot(raw_with_source, 'b--', linewidth=1, alpha=0.7, label='With Source')
                        axes[4].plot(raw_without_source, 'r:', linewidth=1, alpha=0.7, label='Without Source')
                        axes[4].set_title('Reconstruction Quality Comparison')
                        axes[4].set_xlabel('Time Steps')
                        axes[4].set_ylabel('Amplitude')
                        axes[4].legend()
                        axes[4].grid(True, alpha=0.3)
                        
                        plt.suptitle(f'Raw Signal Model Test - GUID: {guid}, Epoch: {epoch}, Batch: {idx}')
                        plt.tight_layout()
                        
                        # Save plot
                        plot_filename = f'raw_signal_test_{guid}_{epoch}_{idx}_sample_{k}.png'
                        plot_path = os.path.join(save_dir_prediction, plot_filename)
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close('all')
                        
                        # Compute and log quality metrics
                        mse_with_source = np.mean((y_raw_sample - raw_with_source) ** 2)
                        mse_without_source = np.mean((y_raw_sample - raw_without_source) ** 2)
                        source_influence_magnitude = np.mean(np.abs(raw_diff))
                        
                        logger.info(f"Sample {k}: MSE with source: {mse_with_source:.6f}, "
                                  f"MSE without source: {mse_without_source:.6f}, "
                                  f"Source influence: {source_influence_magnitude:.6f}")
                        
                    except Exception as e:
                        logger.error(f'Error testing sample {k}: {e}')
                        plt.close('all')
                
                # Cleanup
                del forward_outputs, forward_outputs_no_source, raw_pred, raw_pred_no_source
                del z_latent, z_latent_no_source, raw_signal_diff, latent_diff
                
        logger.info('Raw signal model testing completed.')


    def raw_signal_evaluation_test(self, dataloader, tag="raw_signal_error_stats", device=None):
        """
        Comprehensive evaluation of raw signal reconstruction quality.
        This replaces the old seqvae_mse_test method to work with raw signal architecture.
        """
        if not hasattr(self, 'base_model') or self.base_model is None:
            logger.error("Base model not initialized. Cannot perform evaluation.")
            return None
            
        base_dir = self.test_results_dir
        self.base_model.to(device)
        self.base_model.eval()
        
        # Metrics for raw signal evaluation
        mse_all_list = []
        mse_energy_norm_list = []
        vaf_all_list = []
        log_likelihood_list = []
        raw_signal_list = []
        snr_all_list = []
        pearson_corr_list = []

        with torch.no_grad():
            for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Access data using correct HDF5 dataset field names
                y_st = batch_data.fhr_st.to(device)
                y_ph = batch_data.fhr_ph.to(device) 
                x_ph = batch_data.fhr_up_ph.to(device)
                y_raw = batch_data.fhr.to(device)  # Ground truth raw signal
                
                # Forward pass to get raw signal predictions
                forward_outputs = self.base_model(y_st, y_ph, x_ph)
                raw_predictions = forward_outputs['raw_predictions']
                
                # Extract predictions and ground truth
                raw_pred_mu = raw_predictions['raw_signal_mu']  # (B, 480)
                raw_pred_logvar = raw_predictions['raw_signal_logvar']  # (B, 480)
                
                # Ground truth y_raw is expected to be (B, S*16).
                # Extract the corresponding future window for evaluation
                B, prediction_horizon = raw_pred_mu.shape
                raw_signal_length = y_raw.shape[1]
                
                # Extract the future window from ground truth
                sequence_end = raw_signal_length - prediction_horizon
                if sequence_end <= 0:
                    continue  # Skip if not enough data
                    
                target_future = y_raw[:, sequence_end:sequence_end + prediction_horizon]  # (B, 480)
                raw_std = torch.exp(0.5 * raw_pred_logvar)  # (B, 480)
                
                # MSE calculation on future window
                mse_per_sample = torch.mean((target_future - raw_pred_mu) ** 2, dim=1)  # (B,)
                
                # Energy of the target future window
                energy_per_sample = torch.mean(target_future ** 2, dim=1)  # (B,)
                
                # Energy-normalized MSE
                energy_normalized_mse = mse_per_sample / (energy_per_sample + 1e-12)
                
                # VAF calculation (Variance Accounted For)
                target_centered = target_future - torch.mean(target_future, dim=1, keepdim=True)
                pred_centered = raw_pred_mu - torch.mean(raw_pred_mu, dim=1, keepdim=True)
                
                numerator = torch.sum(target_centered * pred_centered, dim=1) ** 2
                denominator = torch.sum(target_centered ** 2, dim=1) * torch.sum(pred_centered ** 2, dim=1)
                vaf = numerator / (denominator + 1e-12)  # (B,)
                
                # Log-likelihood calculation using predicted mean and std
                log_likelihood = -0.5 * (raw_pred_logvar + 
                                        ((target_future - raw_pred_mu) ** 2) / 
                                        (torch.exp(raw_pred_logvar) + 1e-12))
                log_likelihood_per_sample = torch.mean(log_likelihood, dim=1)  # (B,)
                
                # SNR calculation (in dB)
                signal_power = torch.mean(target_future ** 2, dim=1)  # (B,)
                noise_power = torch.mean((target_future - raw_pred_mu) ** 2, dim=1)  # (B,)
                snr = 10.0 * torch.log10((signal_power + 1e-12) / (noise_power + 1e-12))  # (B,)
                
                # Pearson correlation coefficient
                pearson_corr = torch.zeros(target_future.size(0), device=device)
                for i in range(target_future.size(0)):
                    y_true = target_future[i]
                    y_pred = raw_pred_mu[i]
                    
                    y_true_centered = y_true - torch.mean(y_true)
                    y_pred_centered = y_pred - torch.mean(y_pred)
                    
                    numerator = torch.sum(y_true_centered * y_pred_centered)
                    denominator = torch.sqrt(torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2))
                    pearson_corr[i] = numerator / (denominator + 1e-12)
                
                # Accumulate results
                mse_all_list.append(mse_per_sample)
                mse_energy_norm_list.append(energy_normalized_mse)
                vaf_all_list.append(vaf)
                log_likelihood_list.append(log_likelihood_per_sample)
                raw_signal_list.append(target_future)
                snr_all_list.append(snr)
                pearson_corr_list.append(pearson_corr)

        # Create results directory
        save_dir_hist = os.path.join(base_dir, f'{tag}_results')
        os.makedirs(save_dir_hist, exist_ok=True)

        # Concatenate all data
        mse_all_data = torch.cat(mse_all_list, dim=0)  # (N,)
        mse_energy_normalized = torch.cat(mse_energy_norm_list, dim=0)  # (N,)
        vaf_all_data = torch.cat(vaf_all_list, dim=0)  # (N,)
        log_likelihood_all = torch.cat(log_likelihood_list, dim=0)  # (N,)
        all_raw_signals = torch.cat(raw_signal_list, dim=0)  # (N, signal_length)
        snr_all_data = torch.cat(snr_all_list, dim=0)  # (N,)
        pearson_all_data = torch.cat(pearson_corr_list, dim=0)  # (N,)
        
        # Convert to numpy for saving
        mse_np = mse_all_data.detach().cpu().numpy()
        mse_energy_norm_np = mse_energy_normalized.detach().cpu().numpy()
        vaf_np = vaf_all_data.detach().cpu().numpy()
        log_likelihood_np = log_likelihood_all.detach().cpu().numpy()
        snr_np = snr_all_data.detach().cpu().numpy()
        pearson_np = pearson_all_data.detach().cpu().numpy()
        
        # Calculate statistics
        stats = {
            'mse_mean': np.mean(mse_np),
            'mse_std': np.std(mse_np),
            'mse_energy_norm_mean': np.mean(mse_energy_norm_np),
            'mse_energy_norm_std': np.std(mse_energy_norm_np),
            'vaf_mean': np.mean(vaf_np),
            'vaf_std': np.std(vaf_np),
            'log_likelihood_mean': np.mean(log_likelihood_np),
            'log_likelihood_std': np.std(log_likelihood_np),
            'snr_mean': np.mean(snr_np),
            'snr_std': np.std(snr_np),
            'pearson_mean': np.mean(pearson_np),
            'pearson_std': np.std(pearson_np),
        }
        
        # Log statistics
        logger.info("Raw Signal Evaluation Results:")
        logger.info(f"MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
        logger.info(f"Energy-normalized MSE: {stats['mse_energy_norm_mean']:.6f} ± {stats['mse_energy_norm_std']:.6f}")
        logger.info(f"VAF: {stats['vaf_mean']:.4f} ± {stats['vaf_std']:.4f}")
        logger.info(f"Log-likelihood: {stats['log_likelihood_mean']:.4f} ± {stats['log_likelihood_std']:.4f}")
        logger.info(f"SNR (dB): {stats['snr_mean']:.2f} ± {stats['snr_std']:.2f}")
        logger.info(f"Pearson correlation: {stats['pearson_mean']:.4f} ± {stats['pearson_std']:.4f}")
        
        # Save raw data
        np.save(os.path.join(save_dir_hist, f'{tag}_mse.npy'), mse_np)
        np.save(os.path.join(save_dir_hist, f'{tag}_mse_energy_norm.npy'), mse_energy_norm_np)
        np.save(os.path.join(save_dir_hist, f'{tag}_vaf.npy'), vaf_np)
        np.save(os.path.join(save_dir_hist, f'{tag}_log_likelihood.npy'), log_likelihood_np)
        np.save(os.path.join(save_dir_hist, f'{tag}_snr.npy'), snr_np)
        np.save(os.path.join(save_dir_hist, f'{tag}_pearson_corr.npy'), pearson_np)
        
        # Save statistics
        import json
        with open(os.path.join(save_dir_hist, f'{tag}_statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create plots using the available plotting utilities
        plot_histogram(
            data=log_likelihood_np,
            single_channel=True,
            bins=50,
            save_dir=save_dir_hist,
            tag=f'{tag}_log_likelihood'
        )
        
        plot_histogram(
            data=mse_np,
            single_channel=True,
            bins=50,
            save_dir=save_dir_hist,
            tag=f'{tag}_mse'
        )
        
        plot_histogram(
            data=snr_np,
            single_channel=True,
            bins=50,
            save_dir=save_dir_hist,
            tag=f'{tag}_snr'
        )
        
        plot_histogram(
            data=vaf_np,
            single_channel=True,
            bins=50,
            save_dir=save_dir_hist,
            tag=f'{tag}_vaf'
        )
        
        plot_histogram(
            data=pearson_np,
            single_channel=True,
            bins=50,
            save_dir=save_dir_hist,
            tag=f'{tag}_pearson_correlation'
        )

        return all_raw_signals

    # todo: you can make one function for accuracy analysis and combine both


    def do_raw_signal_tests(self, test_dataloader):
        """
        Run comprehensive testing suite for raw signal TEB-VAE model.
        
        Args:
            test_dataloader: DataLoader containing test data
        """
        logger.info("Starting comprehensive raw signal testing suite")
        
        self.load_pytorch_checkpoint()
        cuda_device = f"cuda:{self.cuda_devices[0]}"
        self.pytorch_model.to(cuda_device)
        
        # Run all raw signal evaluation methods
        logger.info("Running raw signal prediction plots...")
        self.seqvae_raw_signal_plot(test_dataloader, device=cuda_device)
        
        logger.info("Running raw signal model testing...")
        self.test_raw_signal_model(test_dataloader, device=cuda_device)
        
        logger.info("Running comprehensive raw signal evaluation...")
        self.raw_signal_evaluation_test(test_dataloader, tag='comprehensive_eval', device=cuda_device)
        
        logger.info("Raw signal testing suite completed successfully")


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
    
    # For PyTorch Lightning, DDP is handled by the Trainer.
    # We initialize rank and world_size for single-process dataloader creation.
    # Lightning will correctly handle distributed sampling when the DDP strategy is active.
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

        # For validation, we also use a distributed sampler so all GPUs are utilized.
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        graph_model.train_base_model(train_loader=train_loader_seqvae, validation_loader=validation_loader_seqvae)

    if test_SeqVAE > -1:
        # Dataloader configuration for testing
        dataloader_config = config['dataset_config'].get('dataloader_config', {})
        dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
        num_workers = 0
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # Create test dataloader
        test_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        # Initialize model for testing
        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        
        # Run comprehensive raw signal testing suite
        graph_model.do_raw_signal_tests(test_loader_seqvae)

    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()


def main_pytorch(rank, world_size, train_SeqVAE, test_SeqVAE):
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

    # DDP setup for torch.multiprocessing.spawn
    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" else "nccl",
        world_size=world_size,
        rank=rank
    )

    # Set the device for this process based on its rank and the config file
    cuda_devices = config['general_config']['cuda_devices']
    if rank < len(cuda_devices):
        device = cuda_devices[rank]
        torch.cuda.set_device(device)
        logger.info(f"Initialized DDP on rank {rank}/{world_size} using GPU {device}.")
    else:
        logger.error(f"Rank {rank} is out of range for configured cuda_devices with length {len(cuda_devices)}")
        return

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

        # For validation, we also use a distributed sampler so all GPUs are utilized.
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        # Use the PyTorch DDP training method instead of PyTorch Lightning
        graph_model.train_base_model_pytorch(train_loader=train_loader_seqvae, validation_loader=validation_loader_seqvae)

    if test_SeqVAE > -1 and rank == 0:  # Only run testing on rank 0 to avoid duplicate tests
        # Dataloader configuration for testing
        dataloader_config = config['dataset_config'].get('dataloader_config', {})
        dataset_kwargs = dataloader_config.get('dataset_kwargs', {})
        num_workers = 0
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # Create test dataloader
        test_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=num_workers,
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        # Initialize model for testing
        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.create_model()
        
        # Run comprehensive raw signal testing suite
        graph_model.do_raw_signal_tests(test_loader_seqvae)

    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    # Set training parameters directly
    use_pytorch_ddp = False  # Set to True to use PyTorch DDP, False for PyTorch Lightning
    train_model = 1  # 1 to train, -1 to skip
    test_model = -1  # 1 to test, -1 to skip
    
    if use_pytorch_ddp:
        import yaml
        from torch import multiprocessing as mp

        # Load config to determine the number of GPUs to use
        config_file_path = 'model/config.yaml'
        project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if not os.path.isabs(config_file_path):
            config_file_path = os.path.join(project_root, config_file_path)

        with open(config_file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        
        # Determine world size from the number of specified CUDA devices
        cuda_devices = config['general_config'].get('cuda_devices', [])
        world_size = len(cuda_devices)

        if world_size == 0:
            logger.error("No CUDA devices specified in the config file for DDP training.")
        else:
            logger.info(f"Spawning {world_size} processes for DDP training on devices: {cuda_devices}")
            mp.spawn(
                main_pytorch,
                args=(world_size, train_model, test_model),
                nprocs=world_size,
                join=True
            )
    else:
        main(train_SeqVAE=train_model, test_SeqVAE=test_model)
