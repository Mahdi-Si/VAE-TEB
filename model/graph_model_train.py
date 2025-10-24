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
from typing import Optional

from loguru import logger
from hdf5_dataset.kymatio_frequency_analysis import analyze_scattering_frequencies
from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D
from hdf5_dataset.hdf5_dataset import normalize_tensor_data

from pytorch_lightning_modules import *

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from vae_teb_model import (
    SeqVaeTeb,
    SeqVaeNoForecast,
    DEFAULT_COMPILE_ATTEMPTS,
    ensure_compiled_module,
)
from pytorch_lightning_modules import LightSeqVaeTeb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

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
    
    if 'mean_tensor' in stats and 'std_tensor' in stats:
        mean_tensor = stats['mean_tensor']
        std_tensor = stats['std_tensor']
    else:
        mean_tensor = torch.tensor(stats['mean'], dtype=normalized_data.dtype, device=normalized_data.device)
        std_tensor = torch.tensor(stats['std'], dtype=normalized_data.dtype, device=normalized_data.device)    
    epsilon = 1e-8
    denormalized_data = normalized_data * (std_tensor + epsilon) + mean_tensor
    
    return denormalized_data


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

        logger.info(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        logger.info('==' * 50)
        self.stat_path = os.path.normpath(self.config['dataset_config']['stat_path'])

        self.plot_every_epoch = self.config['general_config']['plot_frequency']

        self.epochs_num = self.config['general_config']['epochs']
        self.lr = self.config['general_config']['lr']
        self.lr_milestones = self.config['general_config']['lr_milestone']
        self.kld_beta_ = float(self.config['model_config']['VAE_model']['kld_beta'])
        vae_cfg = self.config['model_config']['VAE_model']
        self.beta_schedule = vae_cfg.get('beta_schedule', 'linear')
        self.beta_start = float(vae_cfg.get('beta_start', 0.0))
        self.beta_end = float(vae_cfg.get('beta_end', 6.0))  # Default β-TCVAE value
        self.beta_anneal_epochs = int(vae_cfg.get('beta_anneal_epochs', 50))
        self.beta_cycle_len = int(vae_cfg.get('beta_cycle_len', 1000))
        self.beta_const_val = float(vae_cfg.get('beta_const_val', self.kld_beta_))

        # SeqVAE forecasting parameters
        self.model_context_len = int(vae_cfg.get('context_len', 75))
        self.model_horizon_len = int(vae_cfg.get('horizon_len', 30))
        self.forecast_weight = float(vae_cfg.get('forecast_weight', vae_cfg.get('predictive_weight', 0.0)))
        self.latent_nll_weight = float(vae_cfg.get('latent_nll_weight', vae_cfg.get('latent_consistency_weight', 0.0)))
        self.predictive_kl_weight = float(vae_cfg.get('predictive_kl_weight', 0.0))
        self.stability_weight = float(vae_cfg.get('stability_weight', 0.0))
        # Legacy aliases (for backwards compatibility in downstream utilities)
        self.predictive_weight = self.forecast_weight
        self.latent_consistency_weight = self.latent_nll_weight

        predictive_horizon_cfg = vae_cfg.get('predictive_horizon')
        if predictive_horizon_cfg is None:
            self.predictive_horizon = self.model_horizon_len
        else:
            self.predictive_horizon = max(1, int(predictive_horizon_cfg))

        predictive_context_cfg = vae_cfg.get('predictive_context_len')
        if predictive_context_cfg is None:
            self.predictive_context_len = self.model_context_len
        else:
            self.predictive_context_len = max(1, int(predictive_context_cfg))

        predictive_max_cfg = vae_cfg.get('predictive_max_anchors', 0)
        if predictive_max_cfg in (None, "", False):
            self.predictive_max_anchors = 0
        else:
            self.predictive_max_anchors = int(predictive_max_cfg)
        self.log_forecast_metrics = bool(vae_cfg.get('log_forecast_metrics', True))
        self.monitor_metric = vae_cfg.get('monitor_metric', 'val/predictive_loss')
        self.monitor_mode = vae_cfg.get('monitor_mode', 'min')

        default_forecaster_flag = (
            self.forecast_weight > 0.0
            or self.latent_nll_weight > 0.0
            or self.predictive_kl_weight > 0.0
            or self.stability_weight > 0.0
        )
        self.enable_forecaster = bool(vae_cfg.get('enable_forecaster', default_forecaster_flag))
        if not self.enable_forecaster:
            self.predictive_weight = 0.0
            self.latent_consistency_weight = 0.0
            self.forecast_weight = 0.0
            self.latent_nll_weight = 0.0
            self.predictive_kl_weight = 0.0
            self.stability_weight = 0.0
            self.log_forecast_metrics = False
            if 'predictive' in self.monitor_metric:
                self.monitor_metric = 'val/total_loss'

        self.freeze_seqvae = self.config['model_config']['VAE_model']['freeze_seqvae']
        self.freeze_core_model = self.config['model_config']['VAE_model'].get('freeze_core_model', False)
        self.batch_size_train = self.config['general_config']['batch_size']['train']
        self.batch_size_test = self.config['general_config']['batch_size']['test']
        self.accumulate_grad_batches = self.config['general_config'].get('accumulate_grad_batches', 1)

        model_cfg = self.config.get('model_config', {})
        self.base_model_checkpoint = model_cfg.get('base_model_checkpoint')
        if self.base_model_checkpoint:
            if not os.path.isabs(self.base_model_checkpoint):
                base_dir = os.path.dirname(self.config_file_path)
                self.base_model_checkpoint = os.path.normpath(os.path.join(base_dir, self.base_model_checkpoint))
            else:
                self.base_model_checkpoint = os.path.normpath(self.base_model_checkpoint)

        self.legacy_seqvae_checkpoint = model_cfg.get('legacy_seqvae_checkpoint')
        if self.legacy_seqvae_checkpoint:
            if not os.path.isabs(self.legacy_seqvae_checkpoint):
                base_dir = os.path.dirname(self.config_file_path)
                self.legacy_seqvae_checkpoint = os.path.normpath(os.path.join(base_dir, self.legacy_seqvae_checkpoint))
            else:
                self.legacy_seqvae_checkpoint = os.path.normpath(self.legacy_seqvae_checkpoint)

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
        self._skip_compilation = False


    def setup_config(self):
        folders_list = [
            self.output_base_dir,
            self.train_results_dir,
            self.test_results_dir,
            self.model_checkpoint_dir,
        ]
        for folder in folders_list:
            os.makedirs(folder, exist_ok=True)

        self.log_file = os.path.join(self.train_results_dir, 'full.log')
        
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(
            self.log_file,
            level="INFO",
            rotation="100 MB",
            retention="14 days",
            compression="zip",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            serialize=False,
        )
        logger.info(yaml.dump(self.config, sort_keys=False, default_flow_style=False))
        logger.info('==' * 50)
        
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
            logger.info("Config hyperparameters will OVERRIDE checkpoint hyperparameters")

            compile_attempts = [None]
            compile_attempts.extend(DEFAULT_COMPILE_ATTEMPTS)

            load_success = False
            last_error: Optional[Exception] = None

            model_cls = SeqVaeTeb if self.enable_forecaster else SeqVaeNoForecast

            for compile_options in compile_attempts:
                if self.enable_forecaster:
                    base_model_for_loading = model_cls(
                        context_len=self.model_context_len,
                        horizon_len=self.model_horizon_len,
                        use_latent_forecaster=True,
                    )
                else:
                    base_model_for_loading = model_cls(
                        context_len=self.model_context_len,
                        horizon_len=self.model_horizon_len,
                    )
                self._resolve_predictive_anchor_cap(getattr(base_model_for_loading, 'sequence_length', 300))

                if compile_options is not None:
                    compiled_module, compiled_ok = ensure_compiled_module(
                        base_model_for_loading,
                        module_name=f"{model_cls.__name__} preload",
                        attempts=[compile_options],
                    )
                    if not compiled_ok:
                        logger.warning(f"Pre-load compilation failed with options {compile_options}; retrying with next strategy")
                        continue
                    base_model_for_loading = compiled_module

                try:
                    logger.info("Loading Lightning checkpoint with strict architecture matching...")
                    self.lightning_base_model = LightSeqVaeTeb.load_from_checkpoint(
                        self.base_model_checkpoint,
                        seqvae_teb_model=base_model_for_loading,
                        strict=False,
                        lr=self.lr,
                        lr_milestones=self.lr_milestones,
                        beta_schedule=self.beta_schedule,
                        beta_start=self.beta_start,
                        beta_end=self.beta_end,
                        beta_anneal_epochs=self.beta_anneal_epochs,
                        beta_cycle_len=self.beta_cycle_len,
                        beta_const_val=self.beta_const_val,
                        predictive_weight=self.predictive_weight,
                        latent_consistency_weight=self.latent_consistency_weight,
                        predictive_horizon=self.predictive_horizon,
                        predictive_context_len=self.predictive_context_len,
                        predictive_max_anchors=self.predictive_max_anchors,
                        log_forecast_metrics=self.log_forecast_metrics,
                        forecast_weight=self.forecast_weight,
                        latent_nll_weight=self.latent_nll_weight,
                        predictive_kl_weight=self.predictive_kl_weight,
                        stability_weight=self.stability_weight,
                    )
                    load_success = True
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Checkpoint load failed with compile options={compile_options}: {e}")

            if load_success:
                logger.info("Enforcing config hyperparameters over checkpoint...")
                self._enforce_lightning_hparams(self.lightning_base_model)

                self.base_model = self.lightning_base_model.model
                self.base_model, compiled_flag = ensure_compiled_module(
                    self.base_model,
                    module_name=f"{model_cls.__name__} post-checkpoint",
                )
                self.lightning_base_model.model = self.base_model
                self._skip_compilation = compiled_flag
                self.pytorch_model = self.base_model

                logger.info("Successfully loaded checkpoint with CONFIG hyperparameters enforced.")
            else:
                logger.error(f"Failed to load checkpoint: {last_error}")
                logger.error("Initializing models from scratch.")
                self.base_model_checkpoint = None
                self._create_fresh_model()

        else:
            if self.base_model_checkpoint:
                logger.warning(f"Checkpoint file not found at {self.base_model_checkpoint}. Initializing models from scratch.")
            elif self.legacy_seqvae_checkpoint and os.path.exists(self.legacy_seqvae_checkpoint):
                logger.info("No full checkpoint provided. Falling back to legacy SeqVAE checkpoint.")
            else:
                logger.info("No checkpoint provided. Initializing models from scratch.")
            self._create_fresh_model()

    def _resolve_predictive_anchor_cap(self, sequence_length: int) -> None:
        """Set predictive_max_anchors to the maximum feasible value if unset or oversized."""
        if not self.enable_forecaster:
            self.predictive_max_anchors = 0
            logger.info("Latent forecaster disabled; predictive_max_anchors set to 0")
            return
        seq_len = max(1, int(sequence_length))
        context = max(1, int(self.predictive_context_len))
        horizon = max(1, int(self.predictive_horizon))

        start = max(0, context - 1)
        end = max(start - 1, seq_len - 1 - horizon)
        if end < start:
            max_possible = 0
        else:
            max_possible = end - start + 1

        if self.predictive_max_anchors <= 0:
            self.predictive_max_anchors = max_possible
        else:
            self.predictive_max_anchors = min(self.predictive_max_anchors, max_possible)

        logger.info(
            "Resolved predictive_max_anchors=%s (max possible=%s, sequence_len=%s, context=%s, horizon=%s)",
            self.predictive_max_anchors,
            max_possible,
            seq_len,
            context,
            horizon,
        )

    def _create_fresh_model(self):
        """Create fresh model instance with config parameters."""
        logger.info("Creating fresh model with config parameters...")

        init_kwargs = {
            'context_len': self.model_context_len,
            'horizon_len': self.model_horizon_len,
        }
        model_cls = SeqVaeTeb if self.enable_forecaster else SeqVaeNoForecast
        if self.enable_forecaster:
            init_kwargs['use_latent_forecaster'] = True
        base_model = None
        legacy_ckpt = getattr(self, 'legacy_seqvae_checkpoint', None)
        if legacy_ckpt:
            if os.path.exists(legacy_ckpt):
                try:
                    logger.info(f"Initializing {model_cls.__name__} from legacy checkpoint: {legacy_ckpt}")
                    base_model = model_cls.from_legacy_checkpoint(
                        legacy_ckpt,
                        strict=False,
                        init_kwargs=init_kwargs,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to load legacy SeqVAE checkpoint {legacy_ckpt}: {exc}")
                    logger.warning("Falling back to random initialization.")
            else:
                logger.warning(f"Legacy SeqVAE checkpoint not found at {legacy_ckpt}. Proceeding with random initialization.")

        if base_model is None:
            base_model = model_cls(**init_kwargs)

        self.base_model = base_model
        self._resolve_predictive_anchor_cap(getattr(self.base_model, 'sequence_length', 300))

        # Apply robust compilation with fallback attempts (matching checkpoint loading logic)
        compile_attempts = [None]
        compile_attempts.extend(DEFAULT_COMPILE_ATTEMPTS)

        compilation_success = False
        last_error: Optional[Exception] = None

        for compile_options in compile_attempts:
            try:
                if compile_options is not None:
                    # Try specific compilation options
                    compiled_module, compiled_ok = ensure_compiled_module(
                        self.base_model,
                        module_name="SeqVaeTeb fresh init",
                        attempts=[compile_options],
                    )
                    if compiled_ok:
                        self.base_model = compiled_module
                        compilation_success = True
                        break
                else:
                    # Try default compilation
                    self.base_model, compiled_flag = ensure_compiled_module(
                        self.base_model,
                        module_name="SeqVaeTeb fresh init",
                    )
                    compilation_success = compiled_flag
                    break
            except Exception as exc:
                last_error = exc
                logger.warning(f"Fresh model compilation failed with options={compile_options}: {exc}")

        if not compilation_success:
            logger.warning(f"All fresh model compilation attempts failed. Last error: {last_error}")
            logger.warning("Proceeding with uncompiled model.")

        self._skip_compilation = compilation_success
        
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
            predictive_weight=self.predictive_weight,
            latent_consistency_weight=self.latent_consistency_weight,
            predictive_horizon=self.predictive_horizon,
            predictive_context_len=self.predictive_context_len,
            predictive_max_anchors=self.predictive_max_anchors,
            log_forecast_metrics=self.log_forecast_metrics,
            forecast_weight=self.forecast_weight,
            latent_nll_weight=self.latent_nll_weight,
            predictive_kl_weight=self.predictive_kl_weight,
            stability_weight=self.stability_weight,
        )
        self._enforce_lightning_hparams(self.lightning_base_model)
        self.pytorch_model = self.base_model  # Set pytorch_model reference

    def _enforce_lightning_hparams(self, lightning_model: LightSeqVaeTeb) -> None:
        """Force Lightning module hyperparameters/attributes to match current config."""
        if lightning_model is None:
            return

        enforce_map = {
            'lr': self.lr,
            'lr_milestones': self.lr_milestones,
            'beta_schedule': self.beta_schedule,
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'beta_anneal_epochs': self.beta_anneal_epochs,
            'beta_cycle_len': self.beta_cycle_len,
            'beta_const_val': self.beta_const_val,
            'predictive_weight': self.predictive_weight,
            'latent_consistency_weight': self.latent_consistency_weight,
            'forecast_weight': self.forecast_weight,
            'latent_nll_weight': self.latent_nll_weight,
            'predictive_kl_weight': self.predictive_kl_weight,
            'stability_weight': self.stability_weight,
            'predictive_horizon': self.predictive_horizon,
            'predictive_context_len': self.predictive_context_len,
            'predictive_max_anchors': self.predictive_max_anchors,
            'log_forecast_metrics': self.log_forecast_metrics,
            'enable_forecaster': self.enable_forecaster,
        }

        for key, value in enforce_map.items():
            try:
                setattr(lightning_model.hparams, key, value)
            except AttributeError:
                pass

        # Synchronize model attributes if present
        if hasattr(lightning_model, 'model') and lightning_model.model is not None:
            lightning_model.model.context_len = self.model_context_len
            lightning_model.model.horizon_len = self.model_horizon_len

        logger.info(
            "Enforced Lightning hparams from config: %s",
            {k: getattr(lightning_model.hparams, k, None) for k in enforce_map.keys()},
        )

    def _apply_freeze_core(self) -> None:
        """Apply freeze_core to the model if latent forecaster is available."""
        if self.pytorch_model is None:
            logger.error("Cannot apply freeze_core: pytorch_model is None")
            return

        # Get the actual model (unwrap if compiled)
        model = self.pytorch_model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod

        # Check if model has latent forecaster
        if not hasattr(model, 'has_forecaster') or not model.has_forecaster():
            logger.error(
                "Cannot freeze core: latent forecaster is not available. "
                "Set enable_forecaster=true in config to use freeze_core_model."
            )
            self.freeze_core_model = False
            return

        # Apply freeze
        model.freeze_core()

        # Get and log parameter info
        param_info = model.get_trainable_params_info()
        logger.info("=" * 80)
        logger.info("FREEZE CORE MODEL APPLIED:")
        logger.info(f"  Core VAE - Trainable: {param_info['core_trainable']:,} | Frozen: {param_info['core_frozen']:,}")
        logger.info(f"  Latent Forecaster - Trainable: {param_info['forecaster_trainable']:,} | Frozen: {param_info['forecaster_frozen']:,}")
        logger.info(f"  TOTAL - Trainable: {param_info['total_trainable']:,} | Frozen: {param_info['total_frozen']:,}")
        logger.info(f"  Training only {param_info['forecaster_trainable']:,} / {param_info['total_trainable'] + param_info['total_frozen']:,} parameters ({100.0 * param_info['forecaster_trainable'] / (param_info['total_trainable'] + param_info['total_frozen']):.2f}%)")
        logger.info("=" * 80)

        # Also apply to lightning model if it exists
        if hasattr(self, 'lightning_base_model') and self.lightning_base_model is not None:
            lightning_model = self.lightning_base_model.model
            if hasattr(lightning_model, '_orig_mod'):
                lightning_model = lightning_model._orig_mod
            if hasattr(lightning_model, 'freeze_core'):
                lightning_model.freeze_core()
                logger.info("Freeze core also applied to Lightning module")

    def _validate_compilation_state(self) -> None:
        """Validate that the model compilation state is correct and log status."""
        if self.pytorch_model is None:
            logger.error("Cannot validate compilation: pytorch_model is None")
            return

        from vae_teb_model import is_compiled_module

        is_compiled = is_compiled_module(self.pytorch_model)
        expected_compiled = self._skip_compilation

        if is_compiled:
            logger.info("✓ Model compilation validation: Model is compiled")
            if hasattr(self.pytorch_model, '_compile_options'):
                compile_opts = getattr(self.pytorch_model, '_compile_options', {})
                logger.info(f"  Compilation options: {compile_opts}")
        else:
            logger.warning("⚠ Model compilation validation: Model is NOT compiled")
            if expected_compiled:
                logger.warning("  Expected compiled=True but found compiled=False")
            else:
                logger.info("  Running in eager mode (compilation was skipped/failed)")

        # Validate Lightning model compilation state matches
        if hasattr(self, 'lightning_base_model') and self.lightning_base_model is not None:
            lightning_compiled = is_compiled_module(self.lightning_base_model.model)
            if lightning_compiled != is_compiled:
                logger.warning(
                    f"⚠ Compilation state mismatch: pytorch_model compiled={is_compiled}, "
                    f"lightning_model compiled={lightning_compiled}"
                )


    def create_model(self):
        """Create model ensuring config parameters take precedence over any checkpoint values."""
        self.setup_config()
        self.load_checkpoint()
        if self.pytorch_model is not None:
            logger.info("Model created successfully with enforced config parameters")

            # Apply freeze_core_model if configured
            if self.freeze_core_model:
                logger.info("Applying freeze_core_model configuration...")
                self._apply_freeze_core()

            # Validate compilation state
            self._validate_compilation_state()
        else:
            logger.error("Failed to create model - pytorch_model is None")

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
        logger.info(
            "Forecasting configuration | forecast_weight=%.4f | latent_nll_weight=%.4f | predictive_kl_weight=%.4f | stability_weight=%.4f | horizon=%s | context=%s | max_anchors=%s | log_metrics=%s",
            self.forecast_weight,
            self.latent_nll_weight,
            self.predictive_kl_weight,
            self.stability_weight,
            self.predictive_horizon,
            self.predictive_context_len,
            self.predictive_max_anchors,
            self.log_forecast_metrics,
        )
        logger.info("Latent forecaster enabled: %s", self.enable_forecaster)
        self.plotting_callback = PlottingCallBack(
            output_dir=self.train_results_dir,
            plot_every_epoch=self.plot_every_epoch,
            predictive_horizon=self.predictive_horizon,
        )

        self.metrics_callback = MetricsLoggingCallback()
        self.hyperparameter_callback = HyperparameterLoggingCallback()

        callbacks_cfg = self.config.get('advanced_config', {}).get('callbacks', {})
        early_cfg = callbacks_cfg.get('early_stopping', {})
        ckpt_cfg = callbacks_cfg.get('model_checkpoint', {})

        monitor_metric = ckpt_cfg.get('monitor', self.monitor_metric)
        monitor_mode = ckpt_cfg.get('mode', self.monitor_mode)
        if (not self.enable_forecaster) and monitor_metric and 'predictive' in monitor_metric:
            monitor_metric = 'val/total_loss'
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        ckpt_filename = ckpt_cfg.get('filename', 'base-model-best-{epoch}')
        ckpt_save_top_k = ckpt_cfg.get('save_top_k', 2)
        ckpt_save_last = ckpt_cfg.get('save_last', False)

        self.checkpoint_callback = ModelCheckpoint(
            monitor=monitor_metric,
            mode=monitor_mode,
            dirpath=self.model_checkpoint_dir,
            filename=ckpt_filename,
            save_top_k=ckpt_save_top_k,
            save_last=ckpt_save_last,
        )
        logger.info(f"Checkpoint callback monitoring '{monitor_metric}' (mode={monitor_mode})")

        early_enabled = bool(early_cfg.get('enabled', False))
        early_monitor = early_cfg.get('monitor', monitor_metric)
        if (not self.enable_forecaster) and early_monitor and 'predictive' in early_monitor:
            early_monitor = 'val/total_loss'

        self.early_stop_callback = None
        if early_enabled:
            self.early_stop_callback = EarlyStopping(
                monitor=early_monitor,
                min_delta=float(early_cfg.get('min_delta', 1e-4)),
                patience=int(early_cfg.get('patience', 100)),
                verbose=True,
                mode=early_cfg.get('mode', monitor_mode),
            )
            logger.info(f"Early stopping enabled: monitoring '{self.early_stop_callback.monitor}' (mode={self.early_stop_callback.mode})")
        
        if self.base_model_checkpoint and os.path.exists(self.base_model_checkpoint):
            logger.info("Created fresh checkpoint callback - will monitor validation loss from retrain start")
            logger.info("New checkpoints will be saved based on validation loss improvements from current point")
        else:
            logger.info("Created fresh checkpoint callback for new training")

        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            max_history_size=19900 
        )

        profiler = SimpleProfiler(dirpath=self.train_results_dir, filename="profiler_base_model.txt")

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
            loging_steps = 1

        callbacks_list = [
            ModelSummary(max_depth=-1),
            self.plotting_callback,
            self.checkpoint_callback,
            self.loss_plot_callback,
            self.hyperparameter_callback,
        ]
        if self.metrics_callback is not None:
            callbacks_list.append(self.metrics_callback)
        if self.early_stop_callback is not None:
            callbacks_list.append(self.early_stop_callback)

        trainer = L.Trainer(
            devices=devices,
            accelerator=accelerator,
            strategy=strategy,
            log_every_n_steps=loging_steps,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            max_epochs=self.epochs_num,
            enable_checkpointing=True,
            enable_progress_bar=True,
            default_root_dir=os.path.normpath(self.train_results_dir),
            profiler=profiler,
            num_sanity_val_steps=0,
            callbacks=callbacks_list,
            precision="16-mixed",
            accumulate_grad_batches=max(self.accumulate_grad_batches, 1),
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            sync_batchnorm=False if len(self.cuda_devices) > 1 else False,
            detect_anomaly=False,
            deterministic=False,
            benchmark=True,
            enable_model_summary=False,  # Disable to save memory
            reload_dataloaders_every_n_epochs=0,  # Don't reload dataloaders
            use_distributed_sampler=True if len(self.cuda_devices) > 1 else False,
            inference_mode=False,        # Use training mode for better performance
            barebones=False,            # Use full trainer features
        )

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

        # Find optimal learning rate
        logger.info("Finding optimal learning rate using PyTorch Lightning's tuner...")

        # Run learning rate finder
        # lr_finder = tuner.lr_find(
        #     self.lightning_base_model,
        #     train_dataloaders=train_loader,
        #     val_dataloaders=validation_loader
        # )
        #
        # # Get suggestion and update model
        # if lr_finder and lr_finder.suggestion():
        #     new_lr = lr_finder.suggestion()
        #     self.lightning_base_model.hparams.lr = new_lr
        #     self.lightning_base_model.lr = new_lr  # Also update attribute if used directly
        #     logger.info(f"Found new optimal learning rate: {new_lr}")
        #
        #     # Plot results
        #     fig = lr_finder.plot(suggest=True)
        #     plot_path = os.path.join(self.train_results_dir, 'lr_finder_plot.png')
        #     fig.savefig(plot_path)
        #     plt.close(fig)
        #     logger.info(f"Learning rate finder plot saved to {plot_path}")
        #
        #     # Clean up lr_finder to free memory
        #     del lr_finder, fig
        # else:
        #     logger.warning("Could not find a new learning rate. Using the one from config.")


        logger.info(f"Starting training of the base model for {self.epochs_num} epochs.")
        trainer.fit(
            self.lightning_base_model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader
        )
        logger.info("Finished training the base model.")

        training_hist = self.loss_plot_callback.history
        path_save_hist = os.path.join(self.train_results_dir, 'base_model_history.pkl')
        with open(path_save_hist, 'wb') as f:
            pickle.dump(training_hist, f)
        
        logger.info(f"Training history saved to {path_save_hist}")
        
        hyperparameter_hist = self.hyperparameter_callback.history
        path_save_hyperparams = os.path.join(self.train_results_dir, 'hyperparameter_history.pkl')
        with open(path_save_hyperparams, 'wb') as f:
            pickle.dump(hyperparameter_hist, f)
        
        logger.info(f"Hyperparameter history saved to {path_save_hyperparams}")
        
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
        # persistent_workers=True,   # SOTA: Keep workers alive between epochs
        # prefetch_factor=4,         # SOTA: Prefetch multiple batches per worker
        # drop_last=True,           # SOTA: Avoid irregular batch sizes
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
        # normalize_fields=normalize_fields,
        # pin_memory=True,           # Speed optimization
        # persistent_workers=False,  # Not needed for validation
        # prefetch_factor=2,         # Lower prefetch for validation
        # drop_last=False,          # Keep all validation samples
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
