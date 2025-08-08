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

from  utils.plot_utils import (
    plot_model_analysis,
    plot_vae_reconstruction,
    plot_transfer_entropy_vs_shift,
    plot_metrics_histograms,
    plot_te_ablation_results,
    plot_te_gain_sweep,
)
from loguru import logger
from hdf5_dataset.kymatio_frequency_analysis import analyze_scattering_frequencies
from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D
from hdf5_dataset.hdf5_dataset import normalize_tensor_data

from pytorch_lightning_modules import *

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from vae_teb_model import SeqVaeTeb
from pytorch_lightning_modules import LightSeqVaeTeb

from torch.optim.lr_scheduler import MultiStepLR

# SPEED OPTIMIZATION: Enable cuDNN benchmarking and other optimizations for maximum training speed
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
torch.backends.cudnn.allow_tf32 = True  # Enable TF32 on Ampere GPUs for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix operations

def denormalize_signal_data(normalized_data: torch.Tensor, field_name: str, normalization_stats: dict) -> torch.Tensor:
    """
    Denormalize FHR or UP signal data using normalization statistics.
    
    Args:
        normalized_data: Normalized tensor data (shape: any)
        field_name: Name of the field ('fhr' or 'up')
        normalization_stats: Dictionary containing normalization statistics
        
    Returns:
        Denormalized tensor data
    """
    if field_name not in normalization_stats:
        logger.warning(f"No normalization stats found for field '{field_name}'. Returning data as-is.")
        return normalized_data
    
    if field_name not in ['fhr', 'up']:
        logger.warning(f"Denormalization only supported for 'fhr' and 'up' fields, got '{field_name}'. Returning data as-is.")
        return normalized_data
    
    stats = normalization_stats[field_name]
    
    # Get mean and std tensors (these should be scalars for fhr/up)
    if 'mean_tensor' in stats and 'std_tensor' in stats:
        mean_tensor = stats['mean_tensor']
        std_tensor = stats['std_tensor']
    else:
        # Fallback to creating tensors from scalar values
        mean_tensor = torch.tensor(stats['mean'], dtype=normalized_data.dtype, device=normalized_data.device)
        std_tensor = torch.tensor(stats['std'], dtype=normalized_data.dtype, device=normalized_data.device)
    
    # Denormalize: original = normalized * std + mean
    epsilon = 1e-8
    denormalized_data = normalized_data * (std_tensor + epsilon) + mean_tensor
    
    return denormalized_data

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
                loss_dict = model.compute_loss(forward_outputs, test_y_st, test_y_ph, test_y_raw, compute_kld_loss=True, beta=1.0)
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
            
            # SPEED OPTIMIZATION: Compile model for faster execution (PyTorch 2.0+)
            try:
                self.base_model = torch.compile(self.base_model, mode='max-autotune')
                logger.info("Model successfully compiled with torch.compile for maximum speed")
            except Exception as e:
                logger.warning(f"torch.compile failed, proceeding without compilation: {e}")
            
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
            # self.early_stop_callback,
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
        log_gpu_memory_usage("After trainer setup")

        # Find optimal batch size first
        logger.info("Finding optimal batch size using PyTorch Lightning's tuner...")
        tuner = Tuner(trainer)
        
        # try:
        #     # Use the built-in batch size finder
        #     optimal_batch_size = tuner.scale_batch_size(
        #         self.lightning_base_model,
        #         train_dataloaders=train_loader,
        #         val_dataloaders=validation_loader,
        #         mode='power',  # Use 'power' mode (doubles batch size) or 'binary' for binary search
        #         init_val=self.batch_size_train,
        #         max_trials=10,
        #         batch_arg_name=None  # We'll handle batch size manually
        #     )
            
        #     if optimal_batch_size and optimal_batch_size != self.batch_size_train:
        #         logger.info(f"Found optimal batch size: {optimal_batch_size} (was {self.batch_size_train})")
        #         # Note: You would need to recreate dataloaders with new batch size
        #         # For now, we'll log the suggestion but keep using current batch size
        #         logger.info("Continuing with current batch size. Consider updating config for next run.")
        #     else:
        #         logger.info(f"Current batch size {self.batch_size_train} appears optimal")
                
        # except Exception as e:
        #     logger.warning(f"Batch size finding failed: {e}. Using configured batch size {self.batch_size_train}")

        # # Find optimal learning rate
        # logger.info("Finding optimal learning rate using PyTorch Lightning's tuner...")

        # # Run learning rate finder
        # lr_finder = tuner.lr_find(
        #     self.lightning_base_model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=validation_loader
        # )

        # # Get suggestion and update model
        # if lr_finder and lr_finder.suggestion():
        #     new_lr = lr_finder.suggestion()
        #     self.lightning_base_model.hparams.lr = new_lr
        #     self.lightning_base_model.lr = new_lr  # Also update attribute if used directly
        #     logger.info(f"Found new optimal learning rate: {new_lr}")

        #     # Plot results
        #     fig = lr_finder.plot(suggest=True)
        #     plot_path = os.path.join(self.train_results_dir, 'lr_finder_plot.png')
        #     fig.savefig(plot_path)
        #     plt.close(fig)
        #     logger.info(f"Learning rate finder plot saved to {plot_path}")

        #     # Clean up lr_finder to free memory
        #     del lr_finder, fig
        # else:
        #     logger.warning("Could not find a new learning rate. Using the one from config.")

        # Log memory before training starts - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        log_gpu_memory_usage("Before training starts")

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
                'mse_loss': 0.0, 'nll_loss': 0.0
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
                            forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.kld_beta_)
                        
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
                        forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.kld_beta_)

                    total_loss = loss_dict['total_loss']

                    # Single backward pass
                    total_loss.backward()
                    
                    # Gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # Accumulate losses for logging
                reconstruction_loss_item = loss_dict['reconstruction_loss'].item()
                kld_loss_item = loss_dict['kld_loss'].item()
                mse_loss_item = loss_dict['mse_loss'].item()
                nll_loss_item = loss_dict['nll_loss'].item()

                train_loss_dict['reconstruction_loss'] += reconstruction_loss_item
                train_loss_dict['kld_loss'] += kld_loss_item
                train_loss_dict['mse_loss'] += mse_loss_item
                train_loss_dict['nll_loss'] += nll_loss_item
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
                'mse_loss': 0.0, 'nll_loss': 0.0
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
                                forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.kld_beta_)
                    else:
                        forward_outputs = model(y_st, y_ph, x_ph)
                        
                        # Compute loss with new API for raw signal reconstruction
                        loss_dict = plain_model.compute_loss(
                            forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.kld_beta_)
                    
                    # Accumulate losses
                    reconstruction_loss_item = loss_dict['reconstruction_loss'].item()
                    kld_loss_item = loss_dict['kld_loss'].item()
                    mse_loss_item = loss_dict['mse_loss'].item()
                    nll_loss_item = loss_dict['nll_loss'].item()
                    current_total_loss = loss_dict['total_loss'].item()

                    val_loss_dict['reconstruction_loss'] += reconstruction_loss_item
                    val_loss_dict['kld_loss'] += kld_loss_item
                    val_loss_dict['mse_loss'] += mse_loss_item
                    val_loss_dict['nll_loss'] += nll_loss_item
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
                    f"MSE: {avg_train_losses['mse_loss']:.4f}, "
                    f"NLL: {avg_train_losses['nll_loss']:.4f}, "
                    f"KLD: {avg_train_losses['kld_loss']:.4f}), "
                    f"Val Loss: {avg_val_losses['total_loss']:.4f} "
                    f"(Recon: {avg_val_losses['reconstruction_loss']:.4f}, "
                    f"MSE: {avg_val_losses['mse_loss']:.4f}, "
                    f"NLL: {avg_val_losses['nll_loss']:.4f}, "
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

    def run_tests(self, test_loader):
        """
        Runs tests on the SeqVaeTeb model by performing analysis and plotting on random samples.
        """
        self.run_analysis_and_plot(test_loader, 200)
        self.run_transfer_entropy_shift_analysis(test_loader)
        self.run_metrics_histogram_analysis(test_loader)
        # New: Demonstrate information flow using UP ablation and gain sweep
        self.run_up_ablation_analysis(test_loader)
        self.run_up_gain_sweep_analysis(test_loader)


    def run_analysis_and_plot(self, test_loader, num_samples=200):
        """
        Runs a full analysis on randomly selected samples from the test loader and plots the results.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of random samples to analyze and plot (default: 50)
        """
        logger.info(f"Starting model analysis and plotting on {num_samples} random samples...")
        self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()

        # Get normalization stats from the dataset for denormalization
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            if normalization_stats:
                logger.info("Found normalization stats for denormalizing FHR and UP signals")
            else:
                logger.warning("No normalization stats available - will use normalized data for plotting")

        # Get scattering transform frequency analysis for channel annotations
        scattering_analysis = None
        try:
            # Parameters from fhr_st_setting.md - J=11, Q=4, T=16, sampling_rate=4Hz
            scattering_analysis = analyze_scattering_frequencies(
                J=11, Q=4, T=16, sampling_rate=4.0, signal_duration_minutes=20.0,
                analyze_phase_harmonics=True, analyze_cross_phase=True
            )
            logger.info("Generated scattering transform frequency analysis for channel annotations")
        except Exception as e:
            logger.warning(f"Could not generate scattering frequency analysis: {e}")
            scattering_analysis = None

        # Collect all samples from the test loader
        logger.info("Collecting all samples from test loader...")
        all_samples = []
        try:
            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc="Collecting samples"):
                    batch_size = batch_data.fhr_st.size(0)
                    for i in range(batch_size):
                        sample = {
                            'fhr_st': batch_data.fhr_st[i],
                            'fhr_ph': batch_data.fhr_ph[i],
                            'fhr_up_ph': batch_data.fhr_up_ph[i],
                            'fhr': batch_data.fhr[i],
                            'up': batch_data.up[i]
                        }
                        all_samples.append(sample)
        except Exception as e:
            logger.error(f"Error collecting samples: {e}")
            return

        if len(all_samples) == 0:
            logger.error("No samples found in test loader. Cannot perform analysis.")
            return

        np.random.seed(42)  # For reproducibility
        total_samples = len(all_samples)
        num_samples = min(num_samples, total_samples)
        selected_indices = np.random.choice(total_samples, size=num_samples, replace=False)
        
        logger.info(f"Selected {num_samples} random samples from {total_samples} total samples")
        logger.info(f"Selected sample indices: {selected_indices[:10]}..." if num_samples > 10 else f"Selected sample indices: {selected_indices}")

        # Process each selected sample
        with torch.no_grad():
            for plot_idx, sample_idx in enumerate(tqdm(selected_indices, desc="Processing selected samples")):
                try:
                    sample = all_samples[sample_idx]
                    
                    # Move sample data to device and add batch dimension
                    y_st = sample['fhr_st'].unsqueeze(0).to(device)
                    y_ph = sample['fhr_ph'].unsqueeze(0).to(device)
                    x_ph = sample['fhr_up_ph'].unsqueeze(0).to(device)
                    y_raw = sample['fhr'].unsqueeze(0).to(device)
                    up_raw = sample['up'].unsqueeze(0).to(device)

                    # Get model outputs
                    forward_outputs = self.pytorch_model(y_st, y_ph, x_ph)
                    latent_z = forward_outputs['z']
                    reconstructed_fhr_mu = forward_outputs['mu_pr']
                    reconstructed_fhr_logvar = forward_outputs['logvar_pr']

                    # Compute loss the same way as training to get consistent KLD values
                    loss_dict = self.pytorch_model.compute_loss(
                        forward_outputs, y_st, y_ph, y_raw, compute_kld_loss=True, beta=self.kld_beta_)
                    
                    # Also get KLD tensor for detailed analysis (original method)
                    kld_tensor = self.pytorch_model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=False)
                    kld_mean_over_channels = kld_tensor.mean(dim=-1)

                    # Always keep normalized versions for reconstruction comparison
                    raw_fhr_normalized_np = y_raw[0].cpu().numpy()
                    raw_up_normalized_np = up_raw[0].cpu().numpy()
                    
                    # Denormalize FHR and UP signals if normalization stats are available
                    if normalization_stats:
                        # Denormalize the normalized signals to get the original raw signals for first plot
                        raw_fhr_denormalized = denormalize_signal_data(y_raw[0], 'fhr', normalization_stats)
                        raw_up_denormalized = denormalize_signal_data(up_raw[0], 'up', normalization_stats)
                        raw_fhr_unnormalized_np = raw_fhr_denormalized.cpu().numpy()
                        raw_up_unnormalized_np = raw_up_denormalized.cpu().numpy()
                        
                        # Log info for first sample to confirm denormalization is working
                        if plot_idx == 0:
                            logger.info(f"Using denormalized FHR and UP signals for first plot, normalized for reconstruction plot")
                            logger.info(f"Unnormalized FHR range: [{raw_fhr_unnormalized_np.min():.2f}, {raw_fhr_unnormalized_np.max():.2f}]")
                            logger.info(f"Unnormalized UP range: [{raw_up_unnormalized_np.min():.2f}, {raw_up_unnormalized_np.max():.2f}]")
                            logger.info(f"Normalized FHR range: [{raw_fhr_normalized_np.min():.2f}, {raw_fhr_normalized_np.max():.2f}]")
                    else:
                        # Use normalized data if no stats available
                        raw_fhr_unnormalized_np = raw_fhr_normalized_np
                        raw_up_unnormalized_np = raw_up_normalized_np
                        
                        # Log warning for first sample
                        if plot_idx == 0:
                            logger.warning("Using normalized FHR and UP signals for plotting (no denormalization stats available)")
                        
                    # Move other data to CPU and convert to numpy for plotting (remove batch dimension)
                    fhr_st_np = y_st[0].cpu().numpy().T
                    fhr_ph_np = y_ph[0].cpu().numpy().T
                    fhr_up_ph_np = x_ph[0].cpu().numpy().T
                    latent_z_np = latent_z[0].cpu().numpy().T
                    reconstructed_fhr_mu_np = reconstructed_fhr_mu[0].cpu().numpy()
                    reconstructed_fhr_logvar_np = reconstructed_fhr_logvar[0].cpu().numpy()
                    kld_tensor_np = kld_tensor[0].cpu().numpy().T
                    kld_mean_over_channels_np = kld_mean_over_channels[0].cpu().numpy()
                    
                    # Extract reconstructed scattering and phase harmonic coefficients from linear_output
                    linear_output = forward_outputs['linear_output']  # Shape: (1, 300, 87)
                    linear_output_np = linear_output[0].cpu().numpy()  # Shape: (300, 87)
                    
                    # Split into scattering (43) and phase harmonic (44) components
                    reconstructed_st_np = linear_output_np[:, :43].T  # Shape: (43, 300)
                    reconstructed_ph_np = linear_output_np[:, 43:].T  # Shape: (44, 300)

                    # Generate plots for this sample
                    plot_model_analysis(
                        output_dir=self.test_results_dir,
                        raw_fhr=raw_fhr_unnormalized_np,  # Unnormalized for first plot
                        raw_up=raw_up_unnormalized_np,    # Unnormalized for first plot
                        fhr_st=fhr_st_np,
                        fhr_ph=fhr_ph_np,
                        fhr_up_ph=fhr_up_ph_np,
                        latent_z=latent_z_np,
                        reconstructed_fhr_mu=reconstructed_fhr_mu_np,
                        reconstructed_fhr_logvar=reconstructed_fhr_logvar_np,
                        kld_tensor=kld_tensor_np,
                        kld_mean_over_channels=kld_mean_over_channels_np,
                        batch_idx=sample_idx,  # Use original sample index for unique file naming
                        loss_dict=loss_dict,  # Pass training-consistent loss values
                        # Pass normalized versions for reconstruction comparison
                        raw_fhr_normalized=raw_fhr_normalized_np,
                        raw_up_normalized=raw_up_normalized_np
                    )
                    
                    # Generate VAE reconstruction plots for this sample
                    plot_vae_reconstruction(
                        output_dir=self.test_results_dir,
                        raw_fhr_unnormalized=raw_fhr_unnormalized_np,
                        raw_up_unnormalized=raw_up_unnormalized_np,
                        raw_fhr_normalized=raw_fhr_normalized_np,
                        raw_up_normalized=raw_up_normalized_np,
                        reconstructed_fhr=reconstructed_fhr_mu_np,
                        original_scattering_transform=fhr_st_np,  # Already transposed to (43, 300)
                        reconstructed_scattering_transform=reconstructed_st_np,  # Shape: (43, 300)
                        original_phase_harmonic=fhr_ph_np,  # Already transposed to (44, 300)
                        reconstructed_phase_harmonic=reconstructed_ph_np,  # Shape: (44, 300)
                        scattering_channel_data=scattering_analysis,  # Frequency analysis data
                        batch_idx=sample_idx,
                        loss_dict=loss_dict
                    )
                    
                    # Log progress every 10 samples
                    if (plot_idx + 1) % 10 == 0:
                        logger.info(f"Completed analysis for {plot_idx + 1}/{num_samples} samples")
                        
                except Exception as e:
                    logger.warning(f"Failed to process sample {sample_idx}: {e}")
                    continue

        logger.info(f"Model analysis and plotting complete for {num_samples} samples.")
        logger.info(f"Plots saved to: {self.test_results_dir}")

    def run_transfer_entropy_shift_analysis(self, test_loader, num_samples=None, max_left_shift_seconds=60, step_seconds=1):
        """
        Analyze transfer entropy (KLD) vs left shifts of UP (UP lags FHR).

        Workflow:
        1) Load raw unnormalized UP and FHR without trimming
        2) Apply circular LEFT shifts to UP only (negative seconds → left)
        3) Recompute cross-channel phase coefficients via KymatioPhaseScattering1D
        4) Normalize new coefficients with existing fhr_up_ph stats
        5) Apply 2-minute trimming to both signals and coefficients
        6) Measure KLD per shift and average across samples
        7) Plot KLD vs shift and example signals

        Args:
            test_loader: DataLoader for test data
            num_samples (int | None): Number of samples to analyze (None = all)
            max_left_shift_seconds (int): Max left shift in seconds (default: 60)
            step_seconds (int): Step size in seconds (default: 1)
        """
        logger.info(
            f"Starting transfer entropy shift analysis (LEFT shifts only) on {('ALL' if num_samples is None else num_samples)} samples, max_left_shift_seconds={max_left_shift_seconds}, step={step_seconds}..."
        )
        self.create_model()
        
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting shift analysis.")
            return
            
        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        
        # Get normalization stats for the fhr_up_ph field
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            if not normalization_stats or 'fhr_up_ph' not in normalization_stats:
                logger.error("No normalization stats found for fhr_up_ph field. Cannot proceed with analysis.")
                return
        else:
            logger.error("Dataset does not provide normalization stats. Cannot proceed with analysis.")
            return
            
        # Initialize scattering transform for cross-phase computation
        # Use parameters matching dataset creation: J=11, Q=4, T=16, shape=5760, max_order=1
        scattering_transform = KymatioPhaseScattering1D(
            J=11, Q=4, T=16, shape=5760, device=device, tukey_alpha=None, max_order=1
        )
        scattering_transform.to(device)
        scattering_transform.eval()
        
        # Get optimal coefficient selection masks (same as dataset creation)
        optimal_selection = scattering_transform.get_optimal_coefficients_for_fhr(11, 4, 16)
        cross_mask = optimal_selection['recommendations']['use_cross_mask']
        logger.info(f"Using cross-channel mask with {cross_mask.sum().item()} selected coefficients")
        
        # Create a temporary dataset without trimming to get raw signals
        logger.info("Creating dataset without trimming to access raw signals...")
        from hdf5_dataset.hdf5_dataset import CombinedHDF5Dataset
        
        # Get dataset config from test_loader
        dataset_paths = test_loader.dataset.paths
        stats_path = test_loader.dataset.stats_path
        allowed_guids = None
        if hasattr(test_loader.dataset, 'allowed_guids'):
            allowed_guids = list(test_loader.dataset.allowed_guids) if test_loader.dataset.allowed_guids else None
            
        # Create dataset without trimming for raw signal access
        raw_dataset = CombinedHDF5Dataset(
            paths=dataset_paths,
            load_fields=['fhr', 'up', 'fhr_st', 'fhr_ph'],  # Load necessary fields
            allowed_guids=allowed_guids,
            stats_path=stats_path,
            trim_minutes=None,  # No trimming to get full raw signals
            normalize_fields=['fhr_st', 'fhr_ph']  # Only normalize what we need, keep fhr/up raw
        )
        
        # Collect samples for analysis
        logger.info("Collecting samples for shift analysis...")
        all_samples = []
        sample_count = 0
        
        try:
            total_items = len(raw_dataset)
            for i in range(total_items):
                if num_samples is not None and sample_count >= num_samples:
                    break
                    
                sample = raw_dataset[i]
                # Store both raw (unnormalized) and normalized versions
                all_samples.append({
                    'fhr': sample['fhr'],           # Raw FHR signal
                    'up': sample['up'],             # Raw UP signal  
                    'fhr_st': sample['fhr_st'],         # Normalized scattering coefficients
                    'fhr_ph': sample['fhr_ph'],         # Normalized phase coefficients
                })
                sample_count += 1
                
        except Exception as e:
            logger.error(f"Error collecting raw samples: {e}")
            return
            
        if len(all_samples) == 0:
            logger.error("No samples collected for analysis.")
            return
            
        logger.info(f"Collected {len(all_samples)} samples for analysis")
        
        # Define LEFT shift range only: [-max_left_shift_seconds, 0] in step_seconds increments
        # At 4Hz sampling rate: 1 second = 4 samples
        sampling_rate = 4.0  # Hz
        shift_seconds = np.arange(-int(max_left_shift_seconds), 0 + 1, int(step_seconds))
        shift_samples = (shift_seconds * sampling_rate).astype(int)
        
        logger.info(f"Testing {len(shift_samples)} left shifts from {shift_seconds[0]}s to {shift_seconds[-1]}s")
        
        # Calculate 2-minute trimming parameters
        trim_minutes = 2.0
        trim_samples_raw = int(4 * 60 * trim_minutes)  # 480 samples at 4Hz
        trim_samples_decimated = trim_samples_raw // 16  # 30 samples for coefficients
        
        # Storage for results
        kld_results = []
        valid_shifts = []
        
        # For plotting individual signals (save first few samples)
        plot_samples = min(3, len(all_samples))
        signal_plot_data = []
        
        with torch.no_grad():
            for shift_idx, (shift_sec, shift_samp) in enumerate(zip(shift_seconds, shift_samples)):
                logger.info(f"Processing left shift {shift_idx+1}/{len(shift_samples)}: {shift_sec}s ({shift_samp} samples)")
                
                sample_klds = []
                
                for sample_idx, sample in enumerate(all_samples):
                    try:
                        # Get raw signals (these are already tensors from the dataset)
                        fhr_raw = sample['fhr'].cpu().numpy()  # Shape: (5760,)
                        up_raw = sample['up'].cpu().numpy()    # Shape: (5760,)
                        fhr_st = sample['fhr_st']  # Already normalized, shape: (300, 43)
                        fhr_ph = sample['fhr_ph']  # Already normalized, shape: (300, 44)
                        
                        # Apply circular shift to UP signal
                        up_shifted = self._apply_circular_shift(up_raw, shift_samp)
                        
                        # Prepare signals exactly as in dataset creation
                        # Stack [fhr, up_shifted] as in create_hdf5_dataset.py line 418
                        st_input = torch.from_numpy(np.stack([fhr_raw, up_shifted], axis=0)).float().unsqueeze(0).to(device)  # (1, 2, 5760)
                        
                        # Compute cross-channel phase coefficients exactly as in dataset creation (lines 427-432)
                        st_results_cross = scattering_transform(x=st_input,
                                                               compute_phase=False,
                                                               compute_cross_phase=True,
                                                               scattering_channel=0,
                                                               phase_channels=[0, 1])
                        
                        # Extract the full cross-phase coefficients (line 437)
                        fhr_up_cc_phase_full = st_results_cross.get('cross_phase_corr')
                        
                        # Apply optimal selection mask exactly as in dataset creation (line 441)
                        cross_phase_raw = fhr_up_cc_phase_full[:, cross_mask, :] if fhr_up_cc_phase_full is not None else None
                        
                        # Convert to expected format: (batch, seq_len, channels)
                        cross_phase_formatted = cross_phase_raw.transpose(1, 2)  # (1, 300, 130)
                        
                        # Normalize using existing stats
                        cross_phase_normalized = normalize_tensor_data(
                            data=cross_phase_formatted,
                            field_name='fhr_up_ph',
                            normalization_stats=normalization_stats,
                            log_norm_channels_config=raw_dataset.log_norm_channels_config,
                            asinh_norm_channels_config=raw_dataset.asinh_norm_channels_config,
                            log_epsilon=raw_dataset.log_epsilon,
                            pin_memory=False,
                            normalize_fields=raw_dataset.normalize_fields,
                            dtype=torch.float32
                        )
                        
                        # Apply 2-minute trimming to coefficients (remove beginning and end)
                        if trim_samples_decimated > 0:
                            cross_phase_trimmed = cross_phase_normalized[:, trim_samples_decimated:-trim_samples_decimated, :]
                            fhr_st_trimmed = fhr_st[trim_samples_decimated:-trim_samples_decimated, :]
                            fhr_ph_trimmed = fhr_ph[trim_samples_decimated:-trim_samples_decimated, :]
                        else:
                            cross_phase_trimmed = cross_phase_normalized
                            fhr_st_trimmed = fhr_st
                            fhr_ph_trimmed = fhr_ph
                        
                        # Prepare inputs for model 
                        y_st_input = fhr_st_trimmed.unsqueeze(0).to(device)  # (1, seq_len, 43)
                        y_ph_input = fhr_ph_trimmed.unsqueeze(0).to(device)  # (1, seq_len, 44)
                        x_ph_input = cross_phase_trimmed.to(device)  # (1, seq_len, 130)
                        
                        # Measure transfer entropy (KLD) for this shift
                        kld_tensor = self.pytorch_model.measure_transfer_entropy(
                            y_st=y_st_input,
                            y_ph=y_ph_input, 
                            x_ph=x_ph_input,
                            reduce_mean=False  # Get full tensor for analysis
                        )
                        
                        # Average KLD over sequence length and latent dimensions
                        sample_kld = kld_tensor.mean().item()
                        sample_klds.append(sample_kld)
                        
                        # Store data for plotting (first few samples and shifts)
                        if sample_idx < plot_samples and len(signal_plot_data) < plot_samples * 5:  # Store 5 shifts per sample
                            if len(shift_seconds) > 1 and shift_idx % max(1, (len(shift_seconds) // 5)) == 0:  # Sample every few shifts
                                # Apply trimming to raw signals for plotting
                                if trim_samples_raw > 0:
                                    fhr_trimmed = fhr_raw[trim_samples_raw:-trim_samples_raw]
                                    up_trimmed = up_raw[trim_samples_raw:-trim_samples_raw]
                                    up_shifted_trimmed = up_shifted[trim_samples_raw:-trim_samples_raw]
                                else:
                                    fhr_trimmed = fhr_raw
                                    up_trimmed = up_raw
                                    up_shifted_trimmed = up_shifted
                                    
                                signal_plot_data.append({
                                    'sample_idx': sample_idx,
                                    'shift_sec': shift_sec,
                                    'fhr': fhr_trimmed,
                                    'up_original': up_trimmed,
                                    'up_shifted': up_shifted_trimmed,
                                    'kld': sample_kld
                                })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process sample {sample_idx} with shift {shift_sec}s: {e}")
                        continue
                
                if len(sample_klds) > 0:
                    # Average KLD across all samples for this shift
                    avg_kld = np.mean(sample_klds)
                    kld_results.append(avg_kld)
                    valid_shifts.append(shift_sec)
                    logger.info(f"Shift {shift_sec}s: Average KLD = {avg_kld:.6f} ({len(sample_klds)} samples)")
                else:
                    logger.warning(f"No valid samples for shift {shift_sec}s")
        
        # Plot results
        if len(kld_results) > 0:
            # Plot transfer entropy vs shift
            plot_path = plot_transfer_entropy_vs_shift(valid_shifts, kld_results, self.test_results_dir)
            
            # Plot individual signal examples
            if signal_plot_data:
                self._plot_signal_shift_examples(signal_plot_data, sampling_rate)
            
            # Save results to file
            results_data = {
                'shifts_seconds': valid_shifts,
                'average_kld': kld_results,
                'num_samples': len(all_samples),
                'sampling_rate': sampling_rate,
                'signal_examples': signal_plot_data
            }
            
            results_path = os.path.join(self.test_results_dir, 'transfer_entropy_shift_analysis.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Transfer entropy shift analysis complete.")
            logger.info(f"Results saved to: {results_path}")
            logger.info(f"Plot saved to: {plot_path}")
        else:
            logger.error("No valid results obtained from shift analysis.")

    def _apply_circular_shift(self, signal, shift_samples):
        """
        Apply circular shift to a signal (no zero-padding, preserves all information).
        
        Args:
            signal: 1D numpy array of signal values
            shift_samples: Number of samples to shift (positive = shift right/delay, negative = shift left/advance)
        
        Returns:
            Circularly shifted signal of the same length
        """
        if shift_samples == 0:
            return signal.copy()
        
        # Use numpy's roll for circular shift
        return np.roll(signal, shift_samples)

    def _plot_signal_shift_examples(self, signal_plot_data, sampling_rate):
        """
        Plot examples of FHR, original UP, and shifted UP signals.
        
        Args:
            signal_plot_data: List of dictionaries containing signal data for different shifts
            sampling_rate: Sampling rate in Hz
        """
        if not signal_plot_data:
            return
            
        # Group data by sample
        samples_data = {}
        for data in signal_plot_data:
            sample_idx = data['sample_idx']
            if sample_idx not in samples_data:
                samples_data[sample_idx] = []
            samples_data[sample_idx].append(data)
        
        # Plot each sample
        for sample_idx, sample_shifts in samples_data.items():
            fig, axes = plt.subplots(len(sample_shifts), 1, figsize=(16, len(sample_shifts) * 4), constrained_layout=True)
            if len(sample_shifts) == 1:
                axes = [axes]
                
            for i, data in enumerate(sample_shifts):
                t = np.arange(len(data['fhr'])) / sampling_rate
                
                axes[i].plot(t, data['fhr'], color='#055C9A', label='FHR', linewidth=1.2, alpha=0.8)
                axes[i].plot(t, data['up_original'], color='#0DD8A2', label='UP Original', linewidth=1.2, alpha=0.8)
                axes[i].plot(t, data['up_shifted'], color='#BB3E00', label=f'UP Shifted ({data["shift_sec"]}s)', linewidth=1.2, alpha=0.8)
                
                axes[i].set_title(f'Sample {sample_idx} - Shift: {data["shift_sec"]}s - KLD: {data["kld"]:.6f}', fontweight='normal', pad=12)
                axes[i].set_ylabel('Amplitude', fontweight='normal')
                axes[i].legend(loc='upper right', framealpha=0.95)
                axes[i].grid(True, alpha=0.3)
                
                if i == len(sample_shifts) - 1:
                    axes[i].set_xlabel('Time (s)', fontweight='normal')
            
            fig.suptitle(f'Signal Shift Examples - Sample {sample_idx}', fontsize=14, fontweight='normal', y=0.98)
            
            # Save plot
            plot_path = os.path.join(self.test_results_dir, f'signal_shift_examples_sample_{sample_idx}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            logger.info(f"Signal shift examples for sample {sample_idx} saved to: {plot_path}")

    def run_metrics_histogram_analysis(self, test_loader, num_samples=None):
        """
        Calculate VAF, MSE, SNR between normalized raw FHR and reconstructed FHR,
        and KLD loss for each sample, then plot histograms of these metrics.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to analyze (None = all samples)
        """
        logger.info("Starting metrics histogram analysis...")
        self.create_model()
        
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting metrics analysis.")
            return
            
        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        
        # Get normalization stats for denormalization
        normalization_stats = None
        if hasattr(test_loader.dataset, 'get_normalization_stats'):
            normalization_stats = test_loader.dataset.get_normalization_stats()
            
        # Collect all samples
        all_samples = []
        sample_count = 0
        max_samples = num_samples if num_samples is not None else float('inf')
        
        try:
            with torch.no_grad():
                for batch_data in tqdm(test_loader, desc="Collecting samples"):
                    if sample_count >= max_samples:
                        break
                        
                    batch_size = batch_data.fhr_st.size(0)
                    for i in range(batch_size):
                        if sample_count >= max_samples:
                            break
                            
                        sample = {
                            'fhr_st': batch_data.fhr_st[i],
                            'fhr_ph': batch_data.fhr_ph[i], 
                            'fhr_up_ph': batch_data.fhr_up_ph[i],
                            'fhr': batch_data.fhr[i]
                        }
                        all_samples.append(sample)
                        sample_count += 1
                        
        except Exception as e:
            logger.error(f"Error collecting samples: {e}")
            return
            
        if len(all_samples) == 0:
            logger.error("No samples found in test loader.")
            return
            
        logger.info(f"Analyzing {len(all_samples)} samples for metrics calculation")
        
        # Storage for metrics
        vaf_values = []
        mse_values = []
        snr_values = []
        kld_values = []
        
        # Process each sample
        with torch.no_grad():
            for sample_idx, sample in enumerate(tqdm(all_samples, desc="Computing metrics")):
                try:
                    # Move sample data to device and add batch dimension
                    y_st = sample['fhr_st'].unsqueeze(0).to(device)
                    y_ph = sample['fhr_ph'].unsqueeze(0).to(device) 
                    x_ph = sample['fhr_up_ph'].unsqueeze(0).to(device)
                    y_raw = sample['fhr'].unsqueeze(0).to(device)
                    
                    # Get model outputs
                    forward_outputs = self.pytorch_model(y_st, y_ph, x_ph)
                    reconstructed_fhr_mu = forward_outputs['mu_pr']  # (1, 4800)
                    
                    # Compute KLD using the model's method
                    kld_tensor = self.pytorch_model.measure_transfer_entropy(
                        y_st, y_ph, x_ph, reduce_mean=False
                    )
                    # Average KLD over sequence length and latent dimensions
                    sample_kld = kld_tensor.mean().item()
                    kld_values.append(sample_kld)
                    
                    # Move to CPU for metric calculations
                    y_raw_np = y_raw[0].cpu().numpy()  # (4800,)
                    reconstructed_fhr_np = reconstructed_fhr_mu[0].cpu().numpy()  # (4800,)
                    
                    # Handle normalization - we want normalized versions for fair comparison
                    if normalization_stats and 'fhr' in normalization_stats:
                        # Both signals should be normalized to compute metrics fairly
                        original_fhr_normalized = y_raw_np  # Already normalized from dataset
                        reconstructed_fhr_normalized = reconstructed_fhr_np  # Model output should be in same scale
                    else:
                        # Use as-is if no normalization stats
                        original_fhr_normalized = y_raw_np
                        reconstructed_fhr_normalized = reconstructed_fhr_np
                    
                    # Calculate VAF (Variance Accounted For)
                    # VAF = 1 - var(original - reconstructed) / var(original)
                    residual = original_fhr_normalized - reconstructed_fhr_normalized
                    var_residual = np.var(residual)
                    var_original = np.var(original_fhr_normalized)
                    
                    if var_original > 1e-12:  # Avoid division by zero
                        vaf = 1.0 - (var_residual / var_original)
                        vaf = max(0.0, min(1.0, vaf))  # Clamp to [0, 1]
                    else:
                        vaf = 0.0
                    vaf_values.append(vaf)
                    
                    # Calculate MSE
                    mse = np.mean((original_fhr_normalized - reconstructed_fhr_normalized) ** 2)
                    mse_values.append(mse)
                    
                    # Calculate SNR (Signal-to-Noise Ratio) in dB
                    # SNR = 10 * log10(signal_power / noise_power)
                    signal_power = np.mean(original_fhr_normalized ** 2)
                    noise_power = np.mean(residual ** 2)
                    
                    if noise_power > 1e-12:  # Avoid division by zero
                        snr_db = 10.0 * np.log10(signal_power / noise_power)
                    else:
                        snr_db = 100.0  # Very high SNR when noise is negligible
                    snr_values.append(snr_db)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {sample_idx}: {e}")
                    continue
        
        # Log statistics
        logger.info(f"Computed metrics for {len(vaf_values)} samples")
        logger.info(f"VAF - Mean: {np.mean(vaf_values):.4f}, Std: {np.std(vaf_values):.4f}")
        logger.info(f"MSE - Mean: {np.mean(mse_values):.6f}, Std: {np.std(mse_values):.6f}")
        logger.info(f"SNR - Mean: {np.mean(snr_values):.2f} dB, Std: {np.std(snr_values):.2f} dB")
        logger.info(f"KLD - Mean: {np.mean(kld_values):.6f}, Std: {np.std(kld_values):.6f}")
        
        # Plot histograms using the plotting function from utils
        plot_metrics_histograms(vaf_values, mse_values, snr_values, kld_values, self.test_results_dir)
        
        # Save metrics data
        metrics_data = {
            'vaf': vaf_values,
            'mse': mse_values, 
            'snr': snr_values,
            'kld': kld_values,
            'num_samples': len(vaf_values),
            'statistics': {
                'vaf': {'mean': np.mean(vaf_values), 'std': np.std(vaf_values)},
                'mse': {'mean': np.mean(mse_values), 'std': np.std(mse_values)},
                'snr': {'mean': np.mean(snr_values), 'std': np.std(snr_values)},
                'kld': {'mean': np.mean(kld_values), 'std': np.std(kld_values)}
            }
        }
        
        results_path = os.path.join(self.test_results_dir, 'metrics_histogram_analysis.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(metrics_data, f)
            
        logger.info(f"Metrics histogram analysis complete. Results saved to: {results_path}")

    def run_up_ablation_analysis(self, test_loader, num_samples=None):
        """Compare TE (KLD) and reconstruction quality (VAF) with and without UP input.

        Args:
            test_loader (DataLoader): Loader providing normalized tensors. e.g., create_optimized_dataloader(...)
            num_samples (int | None): Limit number of samples evaluated. e.g., 200; None = all.

        Returns:
            None: Saves an ablation plot showing distributions and mean±std bars.
        """
        logger.info("Starting UP ablation analysis (with vs without UP)...")
        self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting ablation analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        model = self.pytorch_model.to(device)
        model.eval()

        kld_with_up, kld_without_up = [], []
        vaf_with_up, vaf_without_up = [], []

        processed = 0
        max_samples = num_samples if num_samples is not None else float('inf')

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="UP Ablation"):
                # Respect sample cap
                batch_size = batch.fhr_st.size(0)
                if processed >= max_samples:
                    break
                take = min(batch_size, int(max_samples - processed))

                y_st = batch.fhr_st[:take].to(device)
                y_ph = batch.fhr_ph[:take].to(device)
                x_ph = batch.fhr_up_ph[:take].to(device)
                y_raw = batch.fhr[:take].to(device)

                # With UP
                out_up = model(y_st, y_ph, x_ph)
                mu_pr_up = out_up['mu_pr']  # (B, 4800)
                kld_tensor_up = model.measure_transfer_entropy(y_st, y_ph, x_ph, reduce_mean=False)
                kld_up = kld_tensor_up.mean(dim=(1, 2))  # per-sample

                # Without UP (zeroed source)
                x_zero = torch.zeros_like(x_ph)
                out_no = model(y_st, y_ph, x_zero)
                mu_pr_no = out_no['mu_pr']
                kld_tensor_no = model.measure_transfer_entropy(y_st, y_ph, x_zero, reduce_mean=False)
                kld_no = kld_tensor_no.mean(dim=(1, 2))

                # VAF per-sample (normalized space)
                for i in range(take):
                    gt = y_raw[i].detach().cpu().numpy()
                    pr_up = mu_pr_up[i].detach().cpu().numpy()
                    pr_no = mu_pr_no[i].detach().cpu().numpy()

                    res_up = gt - pr_up
                    res_no = gt - pr_no
                    var_gt = np.var(gt)
                    if var_gt > 1e-12:
                        vaf_w = 1.0 - (np.var(res_up) / var_gt)
                        vaf_wo = 1.0 - (np.var(res_no) / var_gt)
                        # Keep within [0,1] as elsewhere
                        vaf_w = max(0.0, min(1.0, float(vaf_w)))
                        vaf_wo = max(0.0, min(1.0, float(vaf_wo)))
                    else:
                        vaf_w = 0.0
                        vaf_wo = 0.0

                    vaf_with_up.append(vaf_w)
                    vaf_without_up.append(vaf_wo)
                    kld_with_up.append(float(kld_up[i].item()))
                    kld_without_up.append(float(kld_no[i].item()))

                processed += take

        # Plot
        try:
            plot_te_ablation_results(kld_with_up, kld_without_up, vaf_with_up, vaf_without_up, self.test_results_dir)
            logger.info("UP ablation analysis complete.")
        except Exception as e:
            logger.warning(f"Failed to plot ablation analysis: {e}")

    def run_up_gain_sweep_analysis(self, test_loader, gains=None, num_samples=None):
        """Sweep multiplicative gains on UP features and track TE (KLD) and VAF trends.

        Args:
            test_loader (DataLoader): Loader for normalized tensors.
            gains (List[float] | None): Multiplicative gains to apply to UP features. e.g., [0.0, 0.5, 1.0, 1.5, 2.0]
            num_samples (int | None): Limit the number of samples. None = all.

        Returns:
            None: Saves a plot of mean KLD and VAF vs gain.
        """
        logger.info("Starting UP gain sweep analysis...")
        self.create_model()

        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded. Aborting gain sweep analysis.")
            return

        device = torch.device(f"cuda:{self.cuda_devices[0]}" if self.cuda_devices and torch.cuda.is_available() else "cpu")
        model = self.pytorch_model.to(device)
        model.eval()

        gains = gains if gains is not None else [0.0, 0.5, 1.0, 1.5, 2.0]

        # Accumulators per gain
        kld_sums = {g: 0.0 for g in gains}
        vaf_sums = {g: 0.0 for g in gains}
        counts = 0
        max_samples = num_samples if num_samples is not None else float('inf')

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="UP Gain Sweep"):
                if counts >= max_samples:
                    break
                batch_size = batch.fhr_st.size(0)
                take = min(batch_size, int(max_samples - counts))

                y_st = batch.fhr_st[:take].to(device)
                y_ph = batch.fhr_ph[:take].to(device)
                x_ph_base = batch.fhr_up_ph[:take].to(device)
                y_raw = batch.fhr[:take].to(device)

                for g in gains:
                    x_scaled = x_ph_base * float(g)
                    out = model(y_st, y_ph, x_scaled)
                    mu_pr = out['mu_pr']
                    kld_tensor = model.measure_transfer_entropy(y_st, y_ph, x_scaled, reduce_mean=False)

                    # Per-sample KLD mean
                    kld_ps = kld_tensor.mean(dim=(1, 2))  # (B,)

                    # Per-sample VAF
                    for i in range(take):
                        gt = y_raw[i].detach().cpu().numpy()
                        pr = mu_pr[i].detach().cpu().numpy()
                        res = gt - pr
                        var_gt = np.var(gt)
                        if var_gt > 1e-12:
                            vaf = 1.0 - (np.var(res) / var_gt)
                            vaf = max(0.0, min(1.0, float(vaf)))
                        else:
                            vaf = 0.0

                        kld_sums[g] += float(kld_ps[i].item())
                        vaf_sums[g] += vaf

                counts += take

        if counts == 0:
            logger.warning("No samples processed for gain sweep.")
            return

        gains_list = list(gains)
        kld_means = [kld_sums[g] / counts for g in gains_list]
        vaf_means = [vaf_sums[g] / counts for g in gains_list]

        try:
            plot_te_gain_sweep(gains_list, kld_means, vaf_means, self.test_results_dir)
            logger.info("UP gain sweep analysis complete.")
        except Exception as e:
            logger.warning(f"Failed to plot gain sweep analysis: {e}")


def main(train_SeqVAE=1, test_SeqVAE=-1):
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
        # Set num_workers=0 to avoid pickle issues with thread locks
        num_workers = 0
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # For distributed training, rank and world_size are now correctly set
        # before this point. The dataloader will use a DistributedSampler if world_size > 1.
        
        # SPEED OPTIMIZED: Enhanced dataloader with prefetching and pinned memory
        train_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_train_datasets'],
            batch_size=config['general_config']['batch_size']['train'],
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            pin_memory=True,  # Speed optimization
            **dataset_kwargs
        )

        # SPEED OPTIMIZED: Enhanced validation dataloader with prefetching
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=0,  # Set to 0 to avoid pickle issues
            rank=rank,
            world_size=world_size,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            pin_memory=True,  # Speed optimization
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

        # SPEED OPTIMIZED: Enhanced test dataloader
        test_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=0,  # Set to 0 to avoid pickle issues
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            pin_memory=True,  # Speed optimization
            **dataset_kwargs
        )

        # Initialize model for testing
        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.run_tests(test_loader_seqvae)

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
        # Set num_workers=0 to avoid pickle issues with thread locks
        num_workers = 0
        normalize_fields = dataloader_config.get('normalize_fields', None)
        stat_path = config['dataset_config'].get('stat_path')

        # For distributed training, rank and world_size are now correctly set
        # before this point. The dataloader will use a DistributedSampler if world_size > 1.
        
        # SPEED OPTIMIZED: Enhanced dataloader with prefetching and pinned memory
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

        # SPEED OPTIMIZED: Enhanced validation dataloader with prefetching
        validation_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=0,  # Set to 0 to avoid pickle issues
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

        # SPEED OPTIMIZED: Enhanced test dataloader
        test_loader_seqvae = create_optimized_dataloader(
            hdf5_files=config['dataset_config']['vae_test_datasets'],
            batch_size=config['general_config']['batch_size']['test'],
            num_workers=0,  # Set to 0 to avoid pickle issues
            rank=0,
            world_size=1,
            stats_path=stat_path,
            normalize_fields=normalize_fields,
            **dataset_kwargs
        )

        # Initialize model for testing
        graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
        graph_model.run_analysis_and_plot(test_loader_seqvae)

    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    # Set training parameters directly
    use_pytorch_ddp = False  # Set to True to use PyTorch DDP, False for PyTorch Lightning
    train_model = -1  # 1 to train, -1 to skip
    test_model = 1  # 1 to test, -1 to skip
    
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
