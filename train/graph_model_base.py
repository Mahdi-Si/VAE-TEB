
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from train.graph_models_utils import load_checkpoint_torch

import matplotlib.pyplot as plt
import matplotlib

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
import yaml
import os

from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import MLFlowLogger
from utils.custom_logger import setup_logging
from loguru import logger

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


class GraphModelBase(ABC):
    """Shared scaffolding for SeqVAE experiment pipelines.

    Concrete subclasses are expected to implement the abstract hooks
    (``create_model``, ``configure_trainable_params``, ``train_base_model``) so
    they can produce Lightning-ready models and run experiment-specific training
    loops. This base class centralizes config parsing, logging setup, and
    filesystem bookkeeping so each experiment can focus on its modeling logic.
    """

    def __init__(self, config_file_path=None):
        """Initialize run directories, logger, and general hyperparameters."""
        super(GraphModelBase, self).__init__()
        if config_file_path is None:
            self.config_file_path = os.path.dirname(os.path.realpath(__file__)) + '/seqvae_configs/config_args.yaml'
        else:
            self.config_file_path = config_file_path

        with open(self.config_file_path) as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        self._config_dump = yaml.dump(self.config, sort_keys=False, default_flow_style=False)
        now = datetime.now()
        run_date = now.strftime("%Y-%m-%d--[%H-%M-%S]") + f"--{now.microsecond:06d}-"
        self.experiment_tag = self.config['general_config']['tag']
        self.cuda_devices = self.config['general_config']['cuda_devices']

        advanced_cfg = self.config.get('advanced_config', {})
        tracking_cfg = advanced_cfg.get('tracking', {}).get('mlflow', {}) or {}
        self._mlflow_settings = tracking_cfg
        self.mlflow_logger: Optional[MLFlowLogger] = None
        self.lightning_loggers: List[LightningLogger] = []

        self.output_base_dir = os.path.normpath(self.config['general_config']['folders_config']['out_dir_base'])
        self.base_folder = f'{run_date}-{self.experiment_tag}'
        self.train_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'train_results')
        self.test_results_dir = os.path.join(self.output_base_dir, self.base_folder, 'test_results')
        self.model_checkpoint_dir = os.path.join(self.output_base_dir, self.base_folder, 'model_checkpoints')
        self.aux_dir = os.path.join(self.output_base_dir, self.base_folder, 'aux_tests')
        self.tensorboard_dir = os.path.join(self.output_base_dir, self.base_folder, 'tensorboard_log')
        self.log_file = None
        self.logger = None

        self.plot_every_epoch = self.config['general_config']['plot_frequency']

        self.epochs_num = self.config['general_config']['epochs']
        self.lr = self.config['general_config']['lr']
        self.lr_milestones = self.config['general_config']['lr_milestone']
    
        self.batch_size_train = self.config['general_config']['batch_size']['train']
        self.batch_size_test = self.config['general_config']['batch_size']['test']
        self.accumulate_grad_batches = self.config['general_config'].get('accumulate_grad_batches', 1)

        self.clip = 10
        plt.ion()

    def setup_config(self):
        """Initialize logging sinks and ensure output directories exist."""
        folders_list = [
            self.output_base_dir,
            self.train_results_dir,
            self.test_results_dir,
            self.model_checkpoint_dir,
        ]
        for folder in folders_list:
            os.makedirs(folder, exist_ok=True)

        self.log_file = os.path.join(self.train_results_dir, 'full.log')

        setup_logging(
            log_to_file=True,
            log_to_console=True,
            file_path=self.log_file,
            file_level="INFO",
            console_level="INFO",
            rotation="100 MB",
            retention="14 days",
            compression="zip",
            serialize=False,
            backtrace=True,
            diagnose=True,
        )
        logger.info(self._config_dump)
        logger.info('==' * 50)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        self._init_mlflow_logger()

    @abstractmethod
    def create_model(self):
        """Construct the complete Lightning model used for training.

        Expected workflow for subclasses:

        1. Instantiate the raw ``torch.nn.Module`` (SeqVAE variant) based on the
            values stored in ``self.config``.
        2. Pass that module to ``self.configure_trainable_params`` so freezing
            and fine-tuning logic stays centralized.
        3. Wrap the configured module in a Lightning wrapper
            (e.g., ``LightningModelBase`` derivative) and return it.
        """
        raise NotImplementedError("Subclasses must implement create_model()")

    # @abstractmethod
    # def configure_trainable_params(self, model: torch.nn.Module) -> None:
    #     """Set ``requires_grad`` flags before the optimizer sees the parameters.
    #
    #     Example:
    #         >>> def configure_trainable_params(self, model):
    #         ...     # Freeze the pretrained encoder
    #         ...     for param in model.encoder.parameters():
    #         ...         param.requires_grad = False
    #         ...     # Train only the decoder head
    #         ...     for param in model.decoder.parameters():
    #         ...         param.requires_grad = True
    #     """
    #     raise NotImplementedError("Subclasses must implement configure_trainable_params()")

    def set_cuda_devices(self, device_list=None):
        """Override CUDA device IDs used by the training run."""
        self.cuda_devices = device_list if device_list is not None else [0]

    @staticmethod
    def freeze_model(model):
        """Utility to freeze all parameters in-place."""
        for param in model.parameters():
            param.requires_grad = False

    def apply_config_hyperparameters(self, hparams_dict, lightning_module) -> None:
        """Force LightningModule hyperparameters to match explicit override values.

        Args:
            hparams_dict: Mapping of hyperparameter names to the desired values.
                Example::

                    {
                        "lr": 1e-4,
                        "lr_milestones": [1000, 2000],
                        "beta_schedule": "constant",
                        "beta_const_val": 0.01,
                        "predictive_horizon": 30,
                    }

            lightning_module: The instantiated Lightning wrapper whose
                ``hparams`` namespace should be updated.

        Loading from checkpoints restores the hyperparameters that were saved
        with that run. Call this helper immediately after instantiating (and
        optionally loading weights into) the Lightning module to ensure the
        experiment always uses the values declared in the current config file.
        """
        hparams = getattr(lightning_module, "hparams")
        overrides = hparams_dict
        updated = []
        for key, value in overrides.items():
            if value is None:
                continue
            setattr(hparams, key, value)
            updated.append(f"{key}={value}")

        if updated:
            logger.info("apply_config_hyperparameters: {}", ", ".join(updated))
        else:
            logger.info("apply_config_hyperparameters: no overrides were applied; config missing relevant keys")

    @abstractmethod
    def train_model(self, train_loader, validation_loader):
        """Run the full training loop using Lightning's Trainer."""
        raise NotImplementedError("Subclasses must implement train_base_model()")

    def _init_mlflow_logger(self) -> None:
        """Create an MLflow logger when enabled in the experiment config."""
        settings = self._mlflow_settings
        enabled = bool(settings.get('enabled', False))
        if not enabled:
            self.mlflow_logger = None
            self.lightning_loggers = []
            return
        experiment_name = settings.get('experiment_name') or self.experiment_tag
        run_name = settings.get('run_name') or self.base_folder
        tracking_uri = settings.get('tracking_uri')
        artifact_location = settings.get('artifact_location')
        log_model = bool(settings.get('log_model', False))
        tags = settings.get('tags') or None
        self.mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            artifact_location=artifact_location,
            log_model=log_model,
            tags=tags,
            save_dir=self.train_results_dir,
        )
        basic_params = {
            "tag": self.experiment_tag,
            "lr": self.lr,
            "epochs": self.epochs_num,
            "batch_size_train": self.batch_size_train,
            "batch_size_test": self.batch_size_test,
            "run_directory": self.train_results_dir,
        }
        self.mlflow_logger.log_hyperparams(basic_params)
        self.lightning_loggers = [self.mlflow_logger]


if __name__ == '__main__':
    pass
