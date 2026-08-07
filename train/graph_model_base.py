
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

from train.graph_models_utils import load_checkpoint_torch

import matplotlib.pyplot as plt
import matplotlib

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Union
import atexit
import sys
import yaml
import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import Logger as LightningLogger
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.strategies import Strategy
from utils.custom_logger import setup_logging
from loguru import logger

# Backend/determinism flags are NOT set at import time: doing so silently contradicts
# any ``Trainer(deterministic=...)`` request and cannot see the config. They are applied
# per-run in ``configure_determinism()`` instead. The environment and matplotlib settings
# below are not determinism concerns and stay at import scope.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYDEVD_USE_CYTHON']="NO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"   # set to 1 only in debugging 

matplotlib.use('Agg')

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'


# --- Config-validation schema -------------------------------------------------
# Keys the framework itself consumes when building the Trainer. Anything else under
# ``advanced_config.trainer`` is drift and gets a warning (never a hard failure).
_KNOWN_TRAINER_KEYS = {
    "precision", "gradient_clip_val", "gradient_clip_algorithm",
    "sync_batchnorm", "deterministic", "benchmark", "compile",
    "log_every_n_steps", "num_sanity_val_steps", "use_distributed_sampler",
    "profiler",
}

# Blocks the framework recognises under ``advanced_config``. ``tracking`` and
# ``spike_breaker`` are reserved for wiring the framework does not yet own; ``memory``
# is dead but tolerated for backward compatibility (see ``_DEAD_CONFIG_KEYS``);
# ``callbacks`` holds both wired and consumer-owned (plotting) sub-blocks.
_KNOWN_ADVANCED_BLOCKS = {
    "trainer", "memory", "tracking", "callbacks", "spike_breaker", "logging",
}

# Removed knobs that unmigrated consumer configs may still carry. Presence warns; it
# never raises, so those configs keep loading unchanged.
_DEAD_CONFIG_KEYS = {
    "advanced_config.memory": "memory monitoring was removed and is no longer wired",
}

# Carries the run's timestamp from the launching process to its DDP children.
_RUN_STAMP_ENV = "TEB_RUN_STAMP"


def _resolve_run_stamp() -> str:
    """Return the run's timestamp string, identical across every DDP rank.

    ``strategy="ddp"`` re-executes the whole training script once per GPU, so each
    rank would otherwise call ``datetime.now()`` itself. At the shipped minute
    resolution that usually collides harmlessly, but a spawn that straddles a minute
    boundary silently splits one run into two sibling output directories with the
    checkpoints, plots and logs divided between them (Lightning issue #3071).

    ``strategy.broadcast`` cannot fix this: no process group exists yet when this
    runs. The only channel available this early is the environment — Lightning's
    subprocess launcher does ``os.environ.copy()`` into each child — so the parent
    stamps the environment once and children inherit the exact string.

    The inherit branch is keyed on a **non-zero** ``LOCAL_RANK`` deliberately.
    Lightning's launcher keeps rank $0$ for the parent and spawns children for
    $1..N-1$ only, and it sets ``LOCAL_RANK=0`` in the parent's own environment once
    ``fit`` starts. Inheriting on any set ``LOCAL_RANK`` would therefore make a
    *second* run in the same process reuse the first run's directory.

    lean-limit: covers Lightning's ``ddp``/``ddp_spawn`` launchers; under
    ``torchrun`` every rank is launched independently and inherits nothing, so the
    job script must export ``TEB_RUN_STAMP`` itself.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    inherited = os.environ.get(_RUN_STAMP_ENV)
    if inherited and local_rank not in (None, "0"):
        return inherited

    stamp = datetime.now().strftime("%Y-%m-%d--[%H-%M]")
    os.environ[_RUN_STAMP_ENV] = stamp
    return stamp


def _config_get_by_path(cfg, dotted_path):
    """Resolve a dotted ``a.b.c`` path in a nested dict.

    Returns ``(present, value)``; ``present`` is ``False`` if any segment is missing
    or a non-dict is encountered mid-path.
    """
    node = cfg
    for part in dotted_path.split('.'):
        if not isinstance(node, dict) or part not in node:
            return (False, None)
        node = node[part]
    return (True, node)


def _config_type_ok(value, expected):
    """Type-check that treats ``bool`` as distinct from ``int`` (Python conflates them).

    ``bool`` requires an actual bool; ``int`` rejects bools; ``float`` accepts int/float
    but not bool. Any other ``expected`` (including a tuple of types) uses plain
    ``isinstance``.
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def _type_name(expected):
    """Render an ``expected`` type (or tuple of types) for an error message.

    A tuple has no ``__name__``, so join the members' names (e.g. ``bool or str``);
    this keeps the ``ValueError`` message readable instead of raising ``AttributeError``.
    """
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


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

        # encoding is explicit: config.yaml is UTF-8 (contains non-ASCII comment glyphs),
        # and Windows' default cp1252 would raise UnicodeDecodeError on it.
        with open(self.config_file_path, encoding="utf-8") as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        self._config_dump = yaml.dump(self.config, sort_keys=False, default_flow_style=False)
        run_date = _resolve_run_stamp()
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
        """Validate config, seed, set backend flags, then init logging and output dirs."""
        # Fail fast on a broken config, then seed + set backend flags, before anything
        # (model, dataloader, logger) that RNG or determinism state could influence.
        self.validate_config()
        self.configure_determinism()

        folders_list = [
            self.output_base_dir,
            self.train_results_dir,
            self.test_results_dir,
            self.model_checkpoint_dir,
        ]
        for folder in folders_list:
            os.makedirs(folder, exist_ok=True)

        self.log_file = os.path.join(self.train_results_dir, 'full.log')
        self.json_log_file = os.path.join(self.train_results_dir, 'run.jsonl')

        logging_cfg = self.config.get('advanced_config', {}).get('logging', {}) or {}
        self._log_paths = setup_logging(
            log_to_file=True,
            log_to_console=True,
            file_path=self.log_file,
            # JSON Lines mirror of the same records: one object per line, so a killed
            # run still leaves every completed line parseable.
            json_path=self.json_log_file if logging_cfg.get('json_log', True) else None,
            file_level=str(logging_cfg.get('file_level', 'INFO')),
            console_level=str(logging_cfg.get('console_level', 'INFO')),
            rotation=str(logging_cfg.get('rotation', '100 MB')),
            retention=str(logging_cfg.get('retention', '14 days')),
            compression="zip",
            serialize=False,
            backtrace=True,
            # Never on the persisted sinks: diagnose renders every local in the
            # traceback, and a training_step frame holds the input batch.
            diagnose=False,
            console_diagnose=bool(logging_cfg.get('console_diagnose', False)),
        )
        logger.info(self._config_dump)
        logger.info('==' * 50)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        self._init_mlflow_logger()
        # Model-independent MLflow logging happens here (config/hparams/provenance are
        # known before the model is built); architecture and final-model logging happen
        # in MLflowRunLoggingCallback, which build_trainer attaches when tracking is on.
        self._log_run_metadata_to_mlflow()
        self._start_system_metrics_monitor()
        self._register_run_log_upload()

    def configure_determinism(self) -> None:
        """Seed every RNG and set the backend flags from config for reproducible runs.

        Reads ``general_config.seed`` (default $42$) and
        ``advanced_config.trainer.deterministic`` (default ``False``). Seeds Python,
        NumPy and torch — including DataLoader workers — via
        ``lightning.pytorch.seed_everything(seed, workers=True)``, then reconciles the
        cuDNN / TF32 backend flags with the requested determinism level so they no
        longer contradict a ``Trainer(deterministic=...)`` request:

        * ``deterministic=False`` (default): ``cudnn.benchmark=True``,
          ``cudnn.deterministic=False``, ``cudnn.allow_tf32=True``, matmul precision
          ``"high"`` — the historical fast path.
        * ``deterministic=True``: the inverse, with matmul precision ``"highest"``.
          ``cudnn.allow_tf32`` (convolution TF32) is distinct from matmul precision and
          is forced ``False`` here for a genuinely deterministic run.
        """
        general_cfg = self.config.get('general_config', {})
        trainer_cfg = self.config.get('advanced_config', {}).get('trainer', {})
        seed = int(general_cfg.get('seed', 42))
        deterministic = bool(trainer_cfg.get('deterministic', False))

        seed_everything(seed, workers=True)

        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
        # set_float32_matmul_precision supersedes the old import-time
        # ``cuda.matmul.allow_tf32`` flag: "high" enables matmul TF32, "highest" is full FP32.
        torch.set_float32_matmul_precision("highest" if deterministic else "high")
        logger.info(
            "configure_determinism: seed={}, deterministic={}", seed, deterministic,
        )

    def validate_config(self) -> None:
        """Fail fast on a broken config; warn (never raise) on dead/unknown keys.

        The asymmetry is a backward-compatibility requirement: unmigrated consumer
        configs still carry removed keys (e.g. ``advanced_config.memory.*``) and must
        keep loading, so unknown and denylisted keys only produce a warning. Missing
        required keys and mistyped known keys raise ``ValueError`` naming the key.
        """
        cfg = self.config

        # Required keys — hard fail on absence. Container types are checked where the
        # framework immediately indexes into the value; scalar values (epochs, lr) are
        # presence-only so a valid config is never rejected over an int/float distinction.
        required = [
            ("general_config", dict),
            ("general_config.tag", None),
            ("general_config.cuda_devices", list),
            ("general_config.epochs", None),
            ("general_config.lr", None),
            ("general_config.batch_size", dict),
            ("general_config.folders_config", dict),
            ("advanced_config", dict),
            ("advanced_config.trainer", dict),
        ]
        for path, expected in required:
            present, value = _config_get_by_path(cfg, path)
            if not present:
                raise ValueError(f"config: required key '{path}' is missing")
            if expected is not None and not _config_type_ok(value, expected):
                raise ValueError(
                    f"config: key '{path}' must be {_type_name(expected)}, "
                    f"got {type(value).__name__}"
                )

        # Optional-but-known keys — hard fail only on a wrong type when present.
        typed = [
            ("general_config.seed", int),
            ("advanced_config.trainer.precision", str),
            ("advanced_config.trainer.sync_batchnorm", bool),
            ("advanced_config.trainer.deterministic", bool),
            ("advanced_config.trainer.benchmark", bool),
            ("advanced_config.trainer.compile", bool),
            ("advanced_config.trainer.log_every_n_steps", int),
            ("advanced_config.trainer.num_sanity_val_steps", int),
            ("advanced_config.trainer.use_distributed_sampler", bool),
            ("advanced_config.spike_breaker.enabled", bool),
            ("advanced_config.spike_breaker.multiplier", float),
            ("advanced_config.spike_breaker.ema_decay", float),
            ("advanced_config.spike_breaker.ema_floor", float),
            ("advanced_config.spike_breaker.additive_margin", float),
            ("advanced_config.spike_breaker.warmup_batches", int),
            ("advanced_config.spike_breaker.max_consecutive_skips", int),
            ("advanced_config.spike_breaker.comparison_metric", str),
            ("advanced_config.logging.console_level", str),
            ("advanced_config.logging.file_level", str),
            ("advanced_config.logging.json_log", bool),
            ("advanced_config.logging.console_diagnose", bool),
            ("advanced_config.logging.rotation", str),
            ("advanced_config.logging.retention", str),
            ("advanced_config.tracking.mlflow.enabled", bool),
            ("advanced_config.tracking.mlflow.tracking_uri", str),
            ("advanced_config.tracking.mlflow.log_model", bool),
            # log_checkpoints threads into MLFlowLogger(log_model=...); accepts bool or
            # the string "all", so both types are permitted.
            ("advanced_config.tracking.mlflow.log_checkpoints", (bool, str)),
            ("advanced_config.tracking.mlflow.log_config_artifact", bool),
        ]
        for path, expected in typed:
            present, value = _config_get_by_path(cfg, path)
            if present and not _config_type_ok(value, expected):
                raise ValueError(
                    f"config: key '{path}' must be {_type_name(expected)}, "
                    f"got {type(value).__name__}"
                )

        # Denylisted dead keys — warn only.
        for path, reason in _DEAD_CONFIG_KEYS.items():
            present, _ = _config_get_by_path(cfg, path)
            if present:
                logger.warning(
                    "config: deprecated key '{}' is present but ignored ({})",
                    path, reason,
                )

        # Unknown keys under advanced_config — warn only.
        _, trainer = _config_get_by_path(cfg, 'advanced_config.trainer')
        if isinstance(trainer, dict):
            for key in trainer:
                if key not in _KNOWN_TRAINER_KEYS:
                    logger.warning(
                        "config: unrecognized key 'advanced_config.trainer.{}' is ignored",
                        key,
                    )
        _, advanced = _config_get_by_path(cfg, 'advanced_config')
        if isinstance(advanced, dict):
            for key in advanced:
                if key not in _KNOWN_ADVANCED_BLOCKS:
                    logger.warning(
                        "config: unrecognized block 'advanced_config.{}' is ignored", key,
                    )

    def select_ddp_strategy(
        self, num_devices: int, config: dict, model=None
    ) -> Union[str, Strategy]:
        """Return the DDP strategy for ``num_devices`` GPUs.

        Default reproduces the canonical choice: ``"ddp"`` for more than one device,
        ``"auto"`` otherwise. ``config`` and ``model`` are accepted (and ignored by the
        default) so consumers whose models need ``find_unused_parameters=True`` — dead
        logvar heads, structured latent, curriculum — can override this hook wholesale.

        The return type admits a ``Strategy`` instance as well as a shorthand string,
        because ``Trainer(strategy=...)`` accepts either and the strings can express
        ``find_unused_parameters`` and nothing else. A consumer that needs any other DDP
        setting — ``broadcast_buffers``, ``gradient_as_bucket_view``, a comm hook — has
        to return an instance, so the annotation says so rather than making every such
        override a type error.
        """
        return "ddp" if num_devices > 1 else "auto"

    def _build_profiler(self, profiler_spec):
        """Construct the profiler named by ``profiler_spec`` (``None`` disables it).

        Accepts ``"simple"``/``True`` (SimpleProfiler, the default), ``"advanced"``
        (AdvancedProfiler), or a falsy/``"none"`` value (no profiler).
        """
        if profiler_spec in (None, False, "none", "off", ""):
            return None
        if profiler_spec in ("simple", True):
            return SimpleProfiler(dirpath=self.train_results_dir)
        if profiler_spec == "advanced":
            from lightning.pytorch.profilers import AdvancedProfiler
            return AdvancedProfiler(dirpath=self.train_results_dir)
        logger.warning("Unknown profiler '{}'; disabling profiler", profiler_spec)
        return None

    def _build_trainer_kwargs(self, callbacks, model=None) -> dict:
        """Assemble the ``Trainer`` kwargs dict from config (the capturable half).

        Kept separate from :meth:`build_trainer` so tests assert on the dict without
        constructing a ``Trainer`` (which would validate device availability against
        real hardware). Every kwarg that differs across the consumer trainers —
        ``use_distributed_sampler``, ``num_sanity_val_steps``, ``log_every_n_steps``,
        ``profiler``, early stopping, ``sync_batchnorm``, and the DDP strategy — is
        config-driven, so one builder covers them all without drift.
        """
        advanced_cfg = self.config.get('advanced_config', {})
        trainer_cfg = advanced_cfg.get('trainer', {})
        callbacks_cfg = advanced_cfg.get('callbacks', {}) or {}

        callback_list = list(callbacks) if callbacks else []

        # Early stopping: appended from config when enabled (e.g. the classifier path).
        es_cfg = callbacks_cfg.get('early_stopping', {}) or {}
        if es_cfg.get('enabled', False):
            callback_list.append(EarlyStopping(
                monitor=es_cfg.get('monitor', 'val/total_loss'),
                patience=int(es_cfg.get('patience', 30)),
                min_delta=float(es_cfg.get('min_delta', 0.0)),
                mode=es_cfg.get('mode', 'min'),
            ))

        # LR monitor is always attached (previously re-added per consumer).
        callback_list.append(LearningRateMonitor(logging_interval='epoch'))

        # When tracking is on, attach the run-logging callback (architecture at fit
        # start, final model + registry at fit end). Imported lazily so importing this
        # module never pulls the plotting stack in callbacks.py.
        if self.mlflow_logger is not None:
            from train.callbacks import MLflowRunLoggingCallback
            callback_list.append(MLflowRunLoggingCallback(
                mlflow_logger=self.mlflow_logger,
                experiment_tag=self.experiment_tag,
                log_model=bool(self._mlflow_settings.get('log_model', False)),
            ))

        num_devices = len(self.cuda_devices)
        # SyncBatchNorm's forward needs an initialized process group, so it is only safe
        # when more than one device is actually in play — never the config value alone.
        sync_batchnorm = bool(trainer_cfg.get('sync_batchnorm', False)) and num_devices > 1

        # A deterministic run forces cuDNN autotuning off (configure_determinism already
        # set cudnn.benchmark=False); passing benchmark=True here would let the Trainer
        # re-enable it and silently break reproducibility, so the two are reconciled.
        deterministic = bool(trainer_cfg.get('deterministic', False))
        benchmark = bool(trainer_cfg.get('benchmark', True)) and not deterministic

        kwargs = {
            'max_epochs': self.epochs_num,
            'callbacks': callback_list,
            'default_root_dir': self.train_results_dir,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'precision': trainer_cfg.get('precision', '32-true'),
            'deterministic': deterministic,
            'benchmark': benchmark,
            'gradient_clip_val': trainer_cfg.get('gradient_clip_val'),
            'gradient_clip_algorithm': trainer_cfg.get('gradient_clip_algorithm', 'norm'),
            'enable_checkpointing': True,
            'log_every_n_steps': int(trainer_cfg.get('log_every_n_steps', 1)),
            'num_sanity_val_steps': int(trainer_cfg.get('num_sanity_val_steps', 0)),
            'use_distributed_sampler': bool(trainer_cfg.get('use_distributed_sampler', True)),
            'sync_batchnorm': sync_batchnorm,
            # tqdm redraws with a carriage return, which a TTY collapses into one
            # updating line but a redirected stream (nohup, SLURM, CI) expands into
            # thousands of near-identical lines that bury the real log. Gated on
            # *stdout* because that is the stream TQDMProgressBar writes to
            # (``file=sys.stdout`` in tqdm_progress.py); stderr can be a TTY while
            # stdout is a pipe, so testing the wrong one silently keeps the bar on.
            'enable_progress_bar': sys.stdout.isatty(),
            'logger': self.lightning_loggers if self.lightning_loggers else True,
        }

        # Profiler is opt-out: defaults to SimpleProfiler, or none when configured off.
        profiler = self._build_profiler(trainer_cfg.get('profiler', 'simple'))
        if profiler is not None:
            kwargs['profiler'] = profiler

        # Accelerator / devices / strategy — mirror the consumer main()s.
        if torch.cuda.is_available():
            kwargs['accelerator'] = 'gpu'
            kwargs['devices'] = self.cuda_devices
            kwargs['strategy'] = self.select_ddp_strategy(num_devices, self.config, model)
        else:
            kwargs['accelerator'] = 'cpu'
            kwargs['devices'] = 1

        return kwargs

    def build_trainer(self, callbacks, model=None) -> Trainer:
        """Construct the configured Lightning ``Trainer`` from the shared kwargs dict."""
        return Trainer(**self._build_trainer_kwargs(callbacks, model=model))

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

    def _validate_tracking_uri(self, uri: str) -> bool:
        """Validate MLflow tracking URI."""
        if not uri:
            return False
        # Basic validation
        if uri.startswith(('http://', 'https://', 'file://', 'databricks://')):
            return True
        # Local file path
        from pathlib import Path
        try:
            if Path(uri).parent.exists():
                return True
        except Exception:
            pass
        return False

    def _init_mlflow_logger(self) -> None:
        """Create an MLflow logger when enabled in the experiment config.

        The ``MLFlowLogger`` constructor is lazy — it does NOT connect to the
        tracking server.  The first real network call happens when the
        ``experiment`` property is accessed (which creates the MLflow run).
        We therefore explicitly trigger ``experiment`` here and treat any
        failure as a hard error that disables MLflow for this run, rather than
        attaching a broken logger that silently drops every metric during
        training.
        """
        settings = self._mlflow_settings
        enabled = bool(settings.get('enabled', False))
        if not enabled:
            self.mlflow_logger = None
            self.lightning_loggers = []
            return

        tracking_uri = settings.get('tracking_uri')
        experiment_name = settings.get('experiment_name') or self.experiment_tag
        run_name = settings.get('run_name') or self.base_folder

        try:
            # Validate tracking URI if provided
            if tracking_uri and not self._validate_tracking_uri(tracking_uri):
                logger.warning(
                    "Invalid tracking URI: {}, falling back to default",
                    tracking_uri,
                )
                tracking_uri = None

            # Restore the experiment if it was soft-deleted, before
            # Lightning's MLFlowLogger tries to create a run in it.
            try:
                import mlflow
                _pre_client = mlflow.MlflowClient(tracking_uri=tracking_uri)
                _pre_expt = _pre_client.get_experiment_by_name(experiment_name)
                if _pre_expt is not None and _pre_expt.lifecycle_stage == "deleted":
                    _pre_client.restore_experiment(_pre_expt.experiment_id)
                    logger.info(
                        "Restored deleted MLflow experiment '{}' (id={})",
                        experiment_name, _pre_expt.experiment_id,
                    )
            except Exception as e_restore:
                logger.debug("MLflow experiment pre-check skipped: {}", e_restore)

            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                run_name=run_name,
                tracking_uri=tracking_uri,
                artifact_location=settings.get('artifact_location'),
                # Lightning's own checkpoint-artifact logging. Prefer the explicit
                # ``log_checkpoints`` key (the decoupled control that lets a build_trainer
                # run avoid storing weights twice), but fall back to ``log_model`` when it
                # is absent so existing consumer configs — which set ``log_model: true``
                # and build their own Trainer without the run-logging callback — keep
                # uploading checkpoints exactly as before. No bool() coercion so an "all"
                # value survives as-is.
                log_model=settings.get('log_checkpoints', settings.get('log_model', False)),
                tags=settings.get('tags') or None,
                save_dir=self.train_results_dir,
            )

            # Force lazy initialisation: this creates the MlflowClient,
            # the experiment (if needed), and the run.  If the tracking
            # server is unreachable, the call raises and we fall through
            # to the except block that disables MLflow entirely.
            _ = mlflow_logger.experiment

            # Ensure the run name is set explicitly via the mlflow.runName
            # tag.  Some Lightning versions don't pass run_name through to
            # client.create_run(), causing MLflow to auto-generate names
            # like "run_1".
            if run_name and mlflow_logger.run_id:
                try:
                    mlflow_logger.experiment.set_tag(
                        mlflow_logger.run_id, "mlflow.runName", run_name,
                    )
                except Exception:
                    pass

            # Connection verified — log hyperparameters
            basic_params = {
                "tag": self.experiment_tag,
                "lr": self.lr,
                "epochs": self.epochs_num,
                "batch_size_train": self.batch_size_train,
                "batch_size_test": self.batch_size_test,
                "run_directory": self.train_results_dir,
            }
            try:
                mlflow_logger.log_hyperparams(basic_params)
            except Exception as e_params:
                logger.warning("Failed to log hyperparameters: {}", e_params)

            # Only attach the logger after connectivity is confirmed
            self.mlflow_logger = mlflow_logger
            self.lightning_loggers = [mlflow_logger]
            logger.info(
                "MLflow logger initialised — experiment='{}', run='{}', "
                "run_id={}, tracking_uri={}",
                experiment_name,
                run_name,
                mlflow_logger.run_id,
                tracking_uri or "(default)",
            )

        except Exception as e:
            logger.error(
                "MLflow initialisation FAILED (tracking_uri={}): {}",
                tracking_uri,
                e,
            )
            logger.warning(
                "Training will continue WITHOUT MLflow logging. "
                "Check that the MLflow server is running and reachable."
            )
            self.mlflow_logger = None
            self.lightning_loggers = []

    @staticmethod
    def _flatten_config(node, prefix: str, *, max_len: int = 500) -> dict:
        """Flatten a nested config mapping into dotted MLflow param keys.

        Nested dicts recurse into dotted keys (``a.b.c``); non-scalar leaves are
        JSON-encoded; every value is stringified and truncated to ``max_len``
        characters (MLflow caps a param value at 500).

        Args:
            node: The (possibly nested) config mapping to flatten.
            prefix: Dotted prefix prepended to every key at this level.
            max_len: Maximum rendered value length before truncation.

        Returns:
            A flat ``{dotted_key: str_value}`` mapping.
        """
        flat: dict = {}
        if not isinstance(node, dict):
            return flat
        for key, value in node.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(GraphModelBase._flatten_config(value, dotted, max_len=max_len))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat[dotted] = str(value)[:max_len]
            else:
                import json
                flat[dotted] = json.dumps(value, default=str)[:max_len]
        return flat

    def _collect_provenance_tags(self) -> dict:
        """Assemble reproducibility tags for the run (git, versions, host, data).

        Git information is best-effort: when ``git`` is missing or this is not a
        checkout, the git tags are simply omitted rather than raising.
        """
        import platform

        tags: dict = {}
        # Git commit SHA + dirty flag (best-effort).
        try:
            import subprocess
            repo_dir = os.path.dirname(os.path.realpath(__file__))
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL,
            ).decode().strip()
            tags["git_commit"] = sha
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL,
            ).decode().strip()
            tags["git_dirty"] = str(bool(dirty))
        except Exception:
            pass  # git absent / not a repo -> omit git tags, never raise

        # Library + platform versions.
        try:
            import lightning
            import mlflow as _mlflow
            tags["torch_version"] = str(torch.__version__)
            tags["lightning_version"] = str(lightning.__version__)
            tags["mlflow_version"] = str(_mlflow.__version__)
            tags["cuda_version"] = str(torch.version.cuda or "cpu")
        except Exception as exc:
            logger.debug("MLflow version tags skipped: {}", exc)

        tags["host"] = platform.node()
        tags["world_size"] = str(len(self.cuda_devices))
        trainer_cfg = self.config.get('advanced_config', {}).get('trainer', {})
        tags["precision"] = str(trainer_cfg.get('precision', '32-true'))

        dataset_cfg = self.config.get('dataset_config', {}) or {}
        for name in ("vae_train_datasets", "vae_test_datasets"):
            value = dataset_cfg.get(name)
            if value is not None:
                tags[name] = str(value)[:500]
        return tags

    def _log_run_metadata_to_mlflow(self) -> None:
        """Log the resolved config, all hyperparameters, and provenance to the run.

        Runs after ``_init_mlflow_logger`` on the run-creating process. A missing run
        id means a non-zero DDP rank whose logger is a no-op, so it is skipped. Every
        write uses the run-bound client API (never a fluent call, which would orphan a
        new run) and is fail-closed so a tracking outage never kills the run.
        """
        if self.mlflow_logger is None or not getattr(self.mlflow_logger, "run_id", None):
            return
        run_id = self.mlflow_logger.run_id
        experiment = self.mlflow_logger.experiment
        settings = self._mlflow_settings

        # Resolved config as an artifact (activates the previously-unread flag).
        if settings.get('log_config_artifact', True):
            try:
                experiment.log_text(run_id, self._config_dump, "config/resolved_config.yaml")
            except Exception as exc:
                logger.warning("MLflow config-artifact logging failed: {}", exc)

        # Every general + model hyperparameter, flattened to searchable params. Logged
        # in one batched call (via the logger's log_hyperparams, which chunks + is
        # rank-safe) rather than a per-key client loop — fewer round-trips at startup and
        # one bad key cannot abort the rest.
        try:
            params = self._flatten_config(self.config.get('general_config', {}), 'general_config')
            params.update(self._flatten_config(self.config.get('model_config', {}), 'model_config'))
            self.mlflow_logger.log_hyperparams(params)
        except Exception as exc:
            logger.warning("MLflow hyperparameter logging failed: {}", exc)

        # Provenance tags (git, versions, host, precision, datasets).
        try:
            for key, value in self._collect_provenance_tags().items():
                experiment.set_tag(run_id, key, value)
        except Exception as exc:
            logger.warning("MLflow provenance tagging failed: {}", exc)

    def upload_run_logs(self) -> None:
        """Upload this rank's run logs to the MLflow run (fail-closed, idempotent).

        MLflow has no console capture and no append/streaming artifact API, so the
        local files are the source of truth and MLflow gets whole-file snapshots
        (``log_artifact`` replaces the artifact each time). Safe to call repeatedly.

        Not rank-$0$-guarded on purpose. Lightning's ``MLFlowLogger.experiment`` is
        already a no-op ``_DummyExperiment`` on ranks $> 0$, and every rank writes a
        *differently named* file (``full.log.rank2``), so when a non-zero rank is the
        one that hangs — the common DDP failure — its log can still reach the run
        rather than being dropped by a guard.
        """
        if self.mlflow_logger is None or not getattr(self.mlflow_logger, "run_id", None):
            return
        paths = getattr(self, "_log_paths", None)
        if paths is None:
            return
        for path in (paths.text_log, paths.json_log):
            if not path or not os.path.exists(path):
                continue
            try:
                self.mlflow_logger.experiment.log_artifact(
                    self.mlflow_logger.run_id, path, artifact_path="logs",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow run-log upload failed for {}: {}", path, exc)

    def _register_run_log_upload(self) -> None:
        """Ship the run logs to MLflow at interpreter exit.

        Registered here rather than in a Trainer callback because the consumers still
        build their own ``Trainer`` and never call ``build_trainer``, so a callback
        would not fire for them. ``atexit`` covers both a clean finish and an
        unhandled exception — but not ``SIGKILL``/OOM-kill, which is the standing
        reason the local file, not the MLflow copy, is authoritative.
        """
        if self.mlflow_logger is None or not getattr(self.mlflow_logger, "run_id", None):
            return
        atexit.register(self.upload_run_logs)

    def _start_system_metrics_monitor(self) -> None:
        """Start MLflow's system-metrics monitor bound to the client-managed run.

        Lightning's ``MLFlowLogger`` never enters a fluent run, so the fluent
        ``enable_system_metrics_logging`` helper would attach no monitor. An explicit
        ``SystemMetricsMonitor`` bound to the run id is started instead (rank-0,
        fail-closed).
        """
        if self.mlflow_logger is None or not getattr(self.mlflow_logger, "run_id", None):
            return
        try:
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
            monitor = SystemMetricsMonitor(run_id=self.mlflow_logger.run_id)
            monitor.start()
            self._system_metrics_monitor = monitor
        except Exception as exc:
            logger.warning("MLflow system-metrics monitor failed to start: {}", exc)


if __name__ == '__main__':
    pass
