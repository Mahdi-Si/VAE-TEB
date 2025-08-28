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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"   # set to 1 only in debugging 

matplotlib.use('Agg')

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'


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
        # Beta scheduling configuration (optional in config). Defaults keep current behavior.
        vae_cfg = self.config['model_config']['VAE_model']
        self.beta_schedule = vae_cfg.get('beta_schedule', 'linear')
        self.beta_start = float(vae_cfg.get('beta_start', 0.0))
        self.beta_end = float(vae_cfg.get('beta_end', 6.0))  # Default β-TCVAE value
        self.beta_anneal_epochs = int(vae_cfg.get('beta_anneal_epochs', 50))
        self.beta_cycle_len = int(vae_cfg.get('beta_cycle_len', 1000))
        # If beta_const_val not provided, fall back to kld_beta_ from config for constant schedule
        self.beta_const_val = float(vae_cfg.get('beta_const_val', self.kld_beta_))
        
        self.seqvae_ckp = self.config['model_config']['seqvae_checkpoint']

        self.train_classifier = self.config['general_config']['train_classifier']

        self.freeze_seqvae = self.config['model_config']['VAE_model']['freeze_seqvae']
        self.batch_size_train = self.config['general_config']['batch_size']['train']
        self.batch_size_test = self.config['general_config']['batch_size']['test']
        self.accumulate_grad_batches = self.config['general_config'].get('accumulate_grad_batches', 1)
        
        # Additional model parameters from config
        self.decimation_factor = int(vae_cfg.get('decimation_factor', 16))  # From config or default
        self.warmup_period = int(vae_cfg.get('warmup_period', 30))          # From config or default

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
        IMPORTANT: Config file hyperparameters ALWAYS override checkpoint hyperparameters.
        """
        if self.base_model_checkpoint and os.path.exists(self.base_model_checkpoint):
            logger.info(f"Loading model from checkpoint: {self.base_model_checkpoint}")
            logger.info("⚠️  Config hyperparameters will OVERRIDE checkpoint hyperparameters")

            # Create base model with current config parameters
            base_model_for_loading = SeqVaeTeb(
                sequence_length=self.input_size,  # Use config values
                latent_dim_source=self.latent_dim,
                latent_dim_target=self.latent_dim, 
                latent_dim_z=self.latent_dim,
                decimation_factor=self.decimation_factor if hasattr(self, 'decimation_factor') else 16,
                warmup_period=self.warmup_period if hasattr(self, 'warmup_period') else 30,
            )

            try:
                # Load from checkpoint but FORCE all hyperparameters from current config
                self.lightning_base_model = LightSeqVaeTeb.load_from_checkpoint(
                    self.base_model_checkpoint,
                    seqvae_teb_model=base_model_for_loading,
                    strict=False,
                    # FORCE config hyperparameters (these override checkpoint values)
                    lr=self.lr,
                    lr_milestones=self.lr_milestones,
                    beta_schedule=self.beta_schedule,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                    beta_anneal_epochs=self.beta_anneal_epochs,
                    beta_cycle_len=self.beta_cycle_len,
                    beta_const_val=self.beta_const_val,
                )
                
                # CRITICAL: Manually override hparams to ensure config takes precedence
                logger.info("🔧 Enforcing config hyperparameters over checkpoint...")
                self.lightning_base_model.hparams.lr = self.lr
                self.lightning_base_model.hparams.lr_milestones = self.lr_milestones
                self.lightning_base_model.hparams.beta_schedule = self.beta_schedule
                self.lightning_base_model.hparams.beta_start = self.beta_start
                self.lightning_base_model.hparams.beta_end = self.beta_end
                self.lightning_base_model.hparams.beta_anneal_epochs = self.beta_anneal_epochs
                self.lightning_base_model.hparams.beta_cycle_len = self.beta_cycle_len
                self.lightning_base_model.hparams.beta_const_val = self.beta_const_val
                
                logger.info("✅ Successfully ENFORCED config hyperparameters:")
                logger.info(f"  lr: {self.lr}")
                logger.info(f"  beta_schedule: {self.beta_schedule}")
                logger.info(f"  beta_start: {self.beta_start}, beta_end: {self.beta_end}")
                logger.info(f"  beta_anneal_epochs: {self.beta_anneal_epochs}")
                logger.info(f"  beta_const_val: {self.beta_const_val}")
                
                self.base_model = self.lightning_base_model.model
                self.pytorch_model = self.base_model
                logger.info("Successfully loaded checkpoint with CONFIG hyperparameters enforced.")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.error("Initializing models from scratch.")
                self.base_model_checkpoint = None
                self._create_fresh_model()

        else:
            if self.base_model_checkpoint:
                logger.warning(f"Checkpoint file not found at {self.base_model_checkpoint}. Initializing models from scratch.")
            else:
                logger.info("No checkpoint provided. Initializing models from scratch.")
            self._create_fresh_model()

    def _create_fresh_model(self):
        """Create fresh model instance with config parameters."""
        logger.info("🔧 Creating fresh model with config parameters...")
        
        self.base_model = SeqVaeTeb(
            sequence_length=self.input_size,  # Use config values
            latent_dim_source=self.latent_dim,
            latent_dim_target=self.latent_dim, 
            latent_dim_z=self.latent_dim,
            decimation_factor=self.decimation_factor if hasattr(self, 'decimation_factor') else 16,
            warmup_period=self.warmup_period if hasattr(self, 'warmup_period') else 30,
        )
        
        # SPEED OPTIMIZATION: Advanced model compilation (PyTorch 2.0+)
        try:
            # Try aggressive optimization first
            compile_options = {
                'mode': 'max-autotune',  # Most aggressive optimization
                'fullgraph': False,      # Allow graph breaks for complex models
                'dynamic': True,         # Handle dynamic shapes efficiently
            }
            self.base_model = torch.compile(self.base_model, **compile_options)
            logger.info("Model successfully compiled with torch.compile (max-autotune mode)")
        except Exception as e:
            logger.warning(f"max-autotune compilation failed: {e}, trying reduce-overhead mode...")
            try:
                # Fallback to safer compilation
                compile_options = {
                    'mode': 'reduce-overhead',
                    'options': {'triton.cudagraphs': False}
                }
                self.base_model = torch.compile(self.base_model, **compile_options)
                logger.info("Model compiled with reduce-overhead mode")
            except Exception as e2:
                logger.warning(f"All compilation failed, proceeding without compilation: {e2}")
        
        self.lightning_base_model = LightSeqVaeTeb(
            seqvae_teb_model=self.base_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_anneal_epochs=self.beta_anneal_epochs,
            beta_cycle_len=self.beta_cycle_len,
            beta_const_val=self.beta_const_val,
        )
        self.pytorch_model = self.base_model  # Set pytorch_model reference
        
        logger.info("✅ Fresh model created with config parameters:")
        logger.info(f"  sequence_length: {self.input_size}")
        logger.info(f"  latent_dim: {self.latent_dim}")
        logger.info(f"  lr: {self.lr}")
        logger.info(f"  beta_schedule: {self.beta_schedule}")
        logger.info(f"  beta_const_val: {self.beta_const_val}")

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
        """Create model ensuring config parameters take precedence over any checkpoint values."""
        logger.info("🚀 Creating SeqVaeTeb model with config enforcement...")
        logger.info(f"📋 Using config parameters:")
        logger.info(f"  - sequence_length: {self.input_size}")
        logger.info(f"  - latent_dim: {self.latent_dim}")
        logger.info(f"  - decimation_factor: {self.decimation_factor}")
        logger.info(f"  - lr: {self.lr}")
        logger.info(f"  - beta_schedule: {self.beta_schedule}")
        logger.info(f"  - beta_const_val: {self.beta_const_val}")
        
        self.setup_config()
        self.load_checkpoint()
        
        # Final validation that our model has correct parameters
        if self.pytorch_model is not None:
            logger.info("✅ Model created successfully with enforced config parameters")

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
        self.hyperparameter_callback = HyperparameterLoggingCallback()

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
            self.hyperparameter_callback,
            # self.early_stop_callback,
        ]

        # Log memory after callback setup - COMMENTED OUT FOR MULTI-GPU PERFORMANCE
        # log_gpu_memory_usage("After callback setup")

        # Instantiate the PyTorch Lightning Trainer with SOTA optimizations
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
            accumulate_grad_batches=max(self.accumulate_grad_batches, 1),
            # Enhanced memory and speed optimization settings
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            sync_batchnorm=True if len(self.cuda_devices) > 1 else False,
            detect_anomaly=False,
            deterministic=False,
            benchmark=True,
            enable_model_summary=False,  # Disable to save memory
            # SOTA optimizations for faster training
            reload_dataloaders_every_n_epochs=0,  # Don't reload dataloaders
            use_distributed_sampler=True if len(self.cuda_devices) > 1 else False,
            inference_mode=False,        # Use training mode for better performance
            barebones=False,            # Use full trainer features
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
        
        # Save hyperparameter history
        hyperparameter_hist = self.hyperparameter_callback.history
        path_save_hyperparams = os.path.join(self.train_results_dir, 'hyperparameter_history.pkl')
        with open(path_save_hyperparams, 'wb') as f:
            pickle.dump(hyperparameter_hist, f)
        
        logger.info(f"Hyperparameter history saved to {path_save_hyperparams}")
        
        # Generate final hyperparameter plot
        if trainer.is_global_zero:
            self.hyperparameter_callback.plot_hyperparameters(self.train_results_dir)

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return training_hist


def main():
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
    
    # SPEED OPTIMIZED: Enhanced dataloader with advanced prefetching and memory optimizations
    train_loader_seqvae = create_optimized_dataloader(
        hdf5_files=config['dataset_config']['vae_train_datasets'],
        batch_size=config['general_config']['batch_size']['train'],
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        pin_memory=True,           # Speed optimization
        persistent_workers=True,   # SOTA: Keep workers alive between epochs
        prefetch_factor=4,         # SOTA: Prefetch multiple batches per worker
        drop_last=True,           # SOTA: Avoid irregular batch sizes
        **dataset_kwargs
    )
    
    # Update dataset size for β-TCVAE MWS computation
    dataset_size = len(train_loader_seqvae.dataset)
    logger.info(f"Training dataset size: {dataset_size} samples")

    # SPEED OPTIMIZED: Enhanced validation dataloader with optimizations
    validation_loader_seqvae = create_optimized_dataloader(
        hdf5_files=config['dataset_config']['vae_test_datasets'],
        batch_size=config['general_config']['batch_size']['test'],
        num_workers=0,             # Set to 0 to avoid pickle issues
        rank=rank,
        world_size=world_size,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        pin_memory=True,           # Speed optimization
        persistent_workers=False,  # Not needed for validation
        prefetch_factor=2,         # Lower prefetch for validation
        drop_last=False,          # Keep all validation samples
        **dataset_kwargs
    )

    graph_model = SeqVAEGraphModel(config_file_path=config_file_path)
    graph_model.create_model()
    graph_model.train_base_model(train_loader=train_loader_seqvae, validation_loader=validation_loader_seqvae)


    # Clean up the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
