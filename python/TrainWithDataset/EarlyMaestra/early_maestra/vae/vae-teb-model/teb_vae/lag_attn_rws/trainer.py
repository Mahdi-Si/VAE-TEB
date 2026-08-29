r"""The experiment driver: build the model from config, build the callbacks, run the fit.

Run from the repository root, which is what puts ``teb_vae``, ``train`` and ``utils`` on
``sys.path``:

.. code-block:: bash

    # Single GPU / local smoke
    python -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/tiny.yaml

    # Prod box. The rank count must equal len(general_config.cuda_devices) -- default.yaml ships
    # 7 -- or Lightning rejects the device/world-size mismatch at Trainer construction. Export
    # the stamp so ranks 1..N-1 share rank 0's run directory.
    TEB_RUN_STAMP="$(date '+%Y-%m-%d--[%H-%M]')" torchrun --nproc_per_node=7 \
        -m teb_vae.lag_attn_rws.trainer --config teb_vae/lag_attn_rws/configs/default.yaml

From an IDE's Run button, with no command line: ``RUN_CONFIG`` at the bottom of this file names
the config to use. Note a Run-button launch of ``default.yaml`` is a *single* process whose seven
``cuda_devices`` make Lightning spawn DDP workers underneath it; for a single-device smoke run
point ``RUN_CONFIG`` at ``configs/tiny.yaml``.

Everything about *running* an experiment -- run directories, seeding, log sinks, MLflow, the
``Trainer`` itself -- is the framework's. This module supplies the three things the framework
cannot know: how to turn ``model_config.VAE_model`` into a net, which callbacks this model
wants, and which DDP strategy its parameter-usage pattern permits.
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

#: Repository root: ``teb_vae/lag_attn_rws/trainer.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the `teb_vae.` and `train.` imports below would fail
# with ModuleNotFoundError before __main__ is ever reached. Launching as
# `python -m teb_vae.lag_attn_rws.trainer` from the repo root sets __package__ and needs none of
# this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402
from lightning.pytorch.callbacks import Callback, ModelCheckpoint  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import resolve_config_file  # noqa: E402
from teb_vae.lag_attn.channel_reach import (  # noqa: E402
    ChannelBudget,
    resolve_stream_budgets,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws  # noqa: E402
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask  # noqa: E402
from train.callbacks import (  # noqa: E402
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsHistoryCsvCallback,
    MetricsLoggingCallback,
)
from train.data_module import GraphDataModule  # noqa: E402
from train.graph_model_base import GraphModelBase  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: Metric suffixes the task emits on every stage. Kept here rather than imported from the task
#: because it is the trainer that must tell ``MetricsLoggingCallback`` what to collect -- that
#: callback hardcodes its own default list, which does not match this model. A test drives the
#: task and fails if this list ever falls behind what it emits.
_METRIC_SUFFIXES = (
    "total_loss", "main_loss",
    "nll_full_block", "nll_base_block", "nll_full_sample", "nll_base_sample",
    "pred_gap",
    "source_conditioned_kl_raw", "source_conditioned_kl_train",
    "kld_active_frac", "kld_beta",
    # The prior scale rate and its weight. The rate is reported whether or not a config opts
    # into the anchor term, so a collapsing prior is visible in any run's CSV; the echoed
    # weight is what lets a metrics_history.csv identify its own arm.
    "prior_rate", "beta_prior",
    # The three auxiliary shape terms and their echoed weights. Unlike the prior rate these are
    # zero whenever their weight is zero -- the objective does not compute an unweighted shape
    # term -- so the pair of columns is what distinguishes "the term was off" from "the term was
    # on and small". They are L1/Huber quantities, not nats: total_loss carries them and the
    # nll_* columns are the pure-nats readouts.
    "aux_multiscale", "aux_derivative", "aux_boundary",
    "lambda_ms", "lambda_deriv", "lambda_boundary",
    "anchor_coverage_frac",
    # The decoder-output variances (mean_logvar_full/base) do NOT detect a prior variance
    # pinned on its clamp; mean_logvar_prior and logvar_prior_floor_frac are the watch for the
    # failure that inflates the coupling readout while everything else looks healthy.
    "mean_logvar_full", "mean_logvar_base",
    # Whether the decoder's logvar_clamp is binding, at each end. The shipped [-5, 3] is
    # inherited from a feature-coefficient decoder and is re-derived from these two columns.
    "logvar_full_floor_frac", "logvar_full_ceil_frac",
    "mean_logvar_prior", "mean_logvar_post", "logvar_prior_floor_frac",
    "delta_mu_rms", "mu_post_prior_gap_rms",
    # The tanh-bounded latent heads' saturation fractions: a bound that is always active is a
    # silently mis-set hyperparameter.
    "mu_prior_sat_frac", "delta_mu_sat_frac",
)

#: Suffixes the permutation control emits, which runs on validation batches only. Tracking a
#: ``train/`` variant of these would produce a column that is NaN in every row of every run.
_VAL_ONLY_SUFFIXES = ("nll_shuffled_block", "kld_shuffled", "shuffle_penalty")

#: Suffixes emitted on training batches only. Two are the *framework's*, injected when the spike
#: breaker is enabled: they are its whole diagnostic surface -- the per-step skip decision and
#: the EMA it compares against -- and this repository has already lost a run that trained
#: normally and then skipped every batch forever, a failure only those two columns can show. The
#: third is the task's pre-clip gradient norm, which exists on the training path alone and is
#: what the provisional ``gradient_clip_val`` is re-derived from. The fourth is the fraction of
#: optimizer steps whose pre-clip norm exceeded that threshold: ``grad_norm``'s epoch value is
#: an aggregate, so without the fraction "how often did the clip bind" is not recoverable from
#: any recorded quantity.
_TRAIN_ONLY_SUFFIXES = ("spike_skipped", "spike_ema_loss", "grad_norm", "grad_clip_frac")

#: The names the framework actually puts in ``callback_metrics``: every task metric is logged as
#: ``{stage}/{name}``, plus the bare ``lr`` the base logs once per epoch. A bare suffix here
#: would match nothing and produce a column that is NaN for every epoch of every run.
_TRACKED_METRICS = (
    tuple(f"{stage}/{name}" for stage in ("train", "val") for name in _METRIC_SUFFIXES)
    + tuple(f"val/{name}" for name in _VAL_ONLY_SUFFIXES)
    + tuple(f"train/{name}" for name in _TRAIN_ONLY_SUFFIXES)
    + ("lr",)
)

#: The one ``VAE_model`` key naming a real constructor argument that must never be forwarded:
#: weight initialisation is not a config decision.
_NON_CONSTRUCTOR_KEYS = frozenset({"init_weights"})

#: Constructor kwargs holding one entry per channel. Abbreviated in the startup log, in full in
#: the resolved config.
_CHANNEL_TUPLE_KEYS = frozenset(
    {"target_keep_index", "target_delays", "source_keep_index", "source_delays"}
)

#: Filename of the fully resolved configuration, written beside a run's checkpoints. The
#: evaluation entry point derives this path from the checkpoint it is given rather than taking a
#: second config file that could drift from the one that trained, so the name is shared.
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

#: Where the resolved causal guard is recorded in that file. Under ``model_config`` rather than
#: inside ``VAE_model``, and under a name that is not a constructor argument, because it is a
#: *record* of what ``causal_reach_budget_s`` resolved to -- not a second input. Written into
#: ``VAE_model`` it would become a competing source of truth, and re-running from the written
#: config would both forward it and re-resolve the budget.
RESOLVED_BUDGET_KEY = "resolved_causal_budget"


class LagAttnRwsTrainer(GraphModelBase):
    """Experiment driver for :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.

    Everything below the three class attributes is model-independent -- the config-to-constructor
    sweep, the reach-budget resolution, the callback assembly and the DDP strategy selection -- so
    a sibling architecture that keeps this objective, this data contract and this metric surface
    reuses all of it by subclassing and re-pointing the attributes, rather than by copying a
    driver that would then be free to drift.
    """

    #: The net this driver builds. A class attribute rather than a literal at each construction
    #: site so a sibling architecture can be trained through this driver without duplicating the
    #: kwarg sweep, the guards or the callback assembly.
    MODEL_CLS = SeqVaeLagAttnRws

    #: The Lightning task the net is wrapped in.
    TASK_CLS = SeqVaeLagAttnRwsTask

    #: Checkpoint filename stem, before the epoch placeholder. Lightning auto-prefixes each
    #: placeholder with its own name, so this must not itself contain ``epoch=``.
    CHECKPOINT_STEM = "lag-attn-rws"

    #: Loader fields the reconstruction target is built from, checked by ``main``'s normalisation
    #: guard. A class attribute beside the three above and for the same reason: a sibling
    #: forecasting another domain re-points it instead of copying the guard, and the guard keeps
    #: running from ``main`` where no subclass can drop it.
    TARGET_FIELDS: Tuple[str, ...] = ("fhr",)

    #: The framework metric names this driver collects into ``metrics_history.csv``. An attribute
    #: rather than the module global read directly, so a sibling emitting extra metrics adds them
    #: here instead of overriding ``train_model`` -- which is 75 lines of callback assembly whose
    #: copy would be free to drift from this one.
    TRACKED_METRICS: Tuple[str, ...] = _TRACKED_METRICS

    #: Config block the diagnostic plotter reads its settings from. Deliberately not derived from
    #: the package name: a sibling that renames it to match its own package gets no figure, no
    #: error and nothing in the log saying why.
    PLOT_CONFIG_KEY = "lag_attn_rws_plotting"

    #: The causal guard this run resolved, populated by :meth:`_build_model_kwargs` and read by
    #: :meth:`create_model` for the startup log. ``None`` means no reach budget is configured.
    resolved_budget: Optional[ChannelBudget] = None

    #: The second checkpoint criterion's callback, built by :meth:`train_model` only when the
    #: config names a ``secondary_monitor``. Declared here rather than left to appear on the
    #: instance so a caller asking whether a run kept two criteria reads ``None`` instead of an
    #: ``AttributeError`` on every run that kept one.
    secondary_checkpoint_callback: Optional[ModelCheckpoint] = None

    @classmethod
    def plot_callback_cls(cls) -> type[Callback]:
        """Return the diagnostic-plot callback class, importing it on the way.

        A method rather than a class attribute because the import must stay **lazy**: the
        callback pulls matplotlib, and a module-level attribute would import it in every run
        whether or not the config asked for a figure. Called only inside ``train_model``'s
        enabled branch, so enabling the flag before the module exists still fails loudly at the
        import rather than silently plotting nothing.

        Returns:
            The callback class ``train_model`` constructs.
        """
        from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback

        return LagAttnRwsPlotCallback

    @classmethod
    def preflight(cls, config: Dict[str, Any]) -> None:
        """Refuse a launch this driver's architecture cannot serve, before anything is built.

        A documented no-op here, and deliberately so: every guard this architecture needs is one
        of the four module-level ``_check_*`` functions, which ``main`` calls by name from its own
        call site. A subclass that forgets ``super().preflight(config)`` therefore cannot drop an
        inherited check -- there are none to drop -- which is the property that makes the hook
        safe to override with a bare body.

        Called after those four and before ``setup_config``, so a refusal raised here still
        leaves no run directory, no log sink and no MLflow run behind.

        Args:
            config: The already-loaded resolved config dict, not a path. The driver has read the
                file by the time this runs and a second read could only disagree with it.
        """

    def causal_standing_message(self) -> str:
        """Return the one-line statement of this run's causal standing, for the startup log.

        A run's log should say what its history states are a function of, because that is the
        premise every coupling number it produces rests on and it is otherwise recoverable only
        by reading the architecture. For this model the answer is decided by the reach budget:
        pruned-and-delayed channels if one is configured, and otherwise the unguarded statement
        that the stored two-sided features let step $t$ read its own future.

        Called from :meth:`create_model` *before* ``self.pytorch_model`` is assigned, which is
        deliberate: the sentence belongs beside the kwargs it describes, and a launch that dies
        in the constructor should still have said what it was about to build. An override whose
        answer depends on the built network must therefore derive it from the constructor kwargs
        or from a module constant rather than from ``self.pytorch_model``, which is not there yet.

        Returns:
            The message, already formatted for a single ``logger.info`` call.
        """
        if self.resolved_budget is not None:
            return self.resolved_budget.summary()
        return (
            "causal reach budget: none (all channels, no delay) -- input features at step t "
            "read up to 974 s into their own future, so the source-conditioned KL is not a "
            "transfer entropy."
        )

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Translate the ``model_config.VAE_model`` block into constructor kwargs.

        One ``inspect.signature`` sweep: any key naming a real constructor argument is
        forwarded, and anything else -- the loss weights, the likelihood, the schedule -- is left
        for the task. Deliberately no table of defaults: the constructor already owns those, and
        a second copy here could only ever disagree with it. No shape translations either: the
        constructor coerces ``logvar_clamp`` and ``encoder_extra_dilations`` from YAML lists
        itself.

        The one key that is *translated* rather than forwarded is ``causal_reach_budget_s``,
        which names no constructor argument: it is resolved here into the four concrete channel
        tuples the net takes. Resolved here specifically, rather than in ``create_model``, so the
        tuples land in the ``model_kwargs`` written into every checkpoint -- the input adapters'
        widths depend on them, so a checkpoint that recorded only the budget could not be rebuilt
        without re-running the resolution.

        Returns:
            Constructor kwargs for the net.
        """
        vae_config = (self.config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        valid_parameters = set(inspect.signature(self.MODEL_CLS.__init__).parameters)
        model_kwargs = {
            name: value
            for name, value in vae_config.items()
            if name in valid_parameters
            and name not in _NON_CONSTRUCTOR_KEYS
            and value is not None
        }

        # Held on the instance so ``create_model`` can log the run's causal standing without
        # resolving it a second time from a second read of the same config block.
        self.resolved_budget = resolve_stream_budgets(vae_config)
        if self.resolved_budget is not None:
            model_kwargs.update(
                target_keep_index=self.resolved_budget.target_keep_index,
                target_delays=self.resolved_budget.target_delays,
                source_keep_index=self.resolved_budget.source_keep_index,
                source_delays=self.resolved_budget.source_delays,
            )
        return model_kwargs

    def create_model(self) -> None:
        """Build the net, optionally load a checkpoint into it, and wrap it in its task.

        Raises:
            RuntimeError: If ``core_model_checkpoint`` is set but cannot be aligned into the
                model. ``load_checkpoint_strict`` returns ``None`` rather than raising when
                nothing lines up, so an unchecked call trains a randomly-initialised model that
                was supposed to be warm-started -- and reports nothing.
        """
        model_kwargs = self._build_model_kwargs()
        logger.info(
            f"Building {self.MODEL_CLS.__name__} with kwargs: "
            + ", ".join(
                # The four channel tuples are hundreds of integers long and are recorded in full
                # in the resolved config; here they would bury every other kwarg.
                f"{key}=<{len(value)} channels>" if key in _CHANNEL_TUPLE_KEYS else f"{key}={value}"
                for key, value in model_kwargs.items()
            )
        )
        # The run's causal standing, stated in its own log rather than left to be inferred from
        # the config. Behind a method because the sentence is architecture-specific and this one
        # is false for a model whose inputs are not the stored two-sided features.
        logger.info(self.causal_standing_message())
        self.pytorch_model = self.MODEL_CLS(**model_kwargs)

        # ``getattr`` with a safe default rather than a bare read: an architecture with no
        # time-pooling normaliser has no ``causal_norm`` argument at all, and the default states
        # what such a model is -- causal by construction, so there is nothing to warn about.
        if not getattr(self.pytorch_model, "causal_norm", True):
            logger.warning(
                "causal_norm=False: the encoders' GroupNorm pools statistics across time, so "
                "the prior conditions on the future and the source-conditioned KL is NOT a "
                "coupling readout."
            )

        model_config = self.config.get("model_config", {}) or {}
        vae_config = model_config.get("VAE_model", {}) or {}

        self.checkpoint = model_config.get("core_model_checkpoint")
        if self.checkpoint is not None:
            blob = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            # Before the load, not after: a blob from another model may align by accident.
            check_model_class(blob, self.MODEL_CLS.__name__)
            if load_checkpoint_strict(model=self.pytorch_model, checkpoint=blob) is None:
                raise RuntimeError(
                    f"could not align core_model_checkpoint {self.checkpoint!r} into "
                    f"{self.MODEL_CLS.__name__} (no matching module keys). Training would "
                    f"otherwise continue from random weights."
                )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        self.pl_model = self.TASK_CLS(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
            model_kwargs=model_kwargs,
            spike_breaker=self.config.get("advanced_config", {}).get("spike_breaker"),
            beta_schedule=vae_config.get("beta_schedule"),
            kld_beta=vae_config.get("kld_beta", 1.0),
            beta_prior=vae_config.get("beta_prior", 0.0),
            lambda_full=vae_config.get("lambda_full", 1.0),
            lambda_base=vae_config.get("lambda_base", 1.0),
            likelihood=vae_config.get("likelihood", "gaussian_nll"),
            free_bits=vae_config.get("free_bits", 0.0),
            lambda_ms=vae_config.get("lambda_ms", 0.0),
            lambda_deriv=vae_config.get("lambda_deriv", 0.0),
            lambda_boundary=vae_config.get("lambda_boundary", 0.0),
            compile_model=self.compile_model_requested(),
        )
        # Re-forces the config's values onto hparams, so a checkpoint-restored run follows the
        # config it was launched with rather than the one it was originally trained under.
        self.apply_config_hyperparameters(
            {"lr": self.lr, "lr_milestones": self.lr_milestones}, self.pl_model
        )

    def compile_model_requested(self) -> bool:
        """Whether to wrap this architecture's net in ``torch.compile``.

        Always ``False`` here, and ``advanced_config.trainer.compile`` is deliberately **not**
        read: this net's LSTM encoders defeat TorchInductor unconditionally, so honouring the
        key would let a config turn on a path that cannot work. The key stays in the schema
        because the framework validates it and because an architecture whose net compiles can
        override this method to honour it -- which is what makes the refusal a property of the
        net rather than of the objective, the task or the training step, none of which change.

        Returns:
            ``False``.
        """
        return False

    def ddp_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        r"""The ``DistributedDataParallel`` settings every raw-signal architecture here runs under.

        Split out from :meth:`select_ddp_strategy` so the *decision* is assertable without
        reaching into Lightning's ``DDPStrategy._ddp_kwargs``, which is private and whose name is
        not a contract.

        ``find_unused_parameters`` is the only correctness entry and the only one that reads
        config: under ``False`` the reducer expects **every** parameter to be marked ready in
        every backward, and exactly one group can starve -- the decoder log-variance heads, which
        are consumed only under ``likelihood: gaussian_nll``. (The attention output projection
        ``W_o``, the other starvation source in this family, is frozen unconditionally by the
        net's constructor, so it is never in the expectation set at all.)

        The other two are performance settings and are constants:

        * ``broadcast_buffers=False``. DDP re-broadcasts every buffer from rank $0$ on **each
          forward**, which here is $1.5$ MiB -- the eight fixed anti-alias filter banks, fourteen
          rotary tables, three causal attention masks and the raw-target index grid. Every one is
          a deterministic function of the config, built identically in each rank's constructor,
          so the broadcast restores values that were never going to differ. Safe *because* there
          is no ``BatchNorm`` anywhere in this family -- a running statistic is the one kind of
          buffer that genuinely diverges per rank, and ``sync_batchnorm`` is off for the same
          reason. A model that gained one would need this back.
        * ``gradient_as_bucket_view=True``. Points ``param.grad`` at the reduction bucket instead
          of a separate allocation, saving one model-sized gradient copy per step.

        **``static_graph`` is deliberately absent**, and that is a correctness call rather than an
        omission. It promises DDP that the autograd graph is identical on every iteration, and the
        loss-spike circuit breaker breaks exactly that promise: on a skipped batch
        ``LightningModelBase._apply_spike_breaker`` substitutes a loss summed over every trainable
        parameter times zero, which is a structurally different backward from the one the first
        iteration recorded. The breaker is enabled in the shipped configs, so the promise would be
        false on precisely the batches that already went wrong.

        Args:
            config: The resolved config.

        Returns:
            Keyword arguments for ``DDPStrategy``.
        """
        vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
        gaussian = str(vae_config.get("likelihood", "gaussian_nll")) == "gaussian_nll"
        return {
            "find_unused_parameters": not gaussian,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }

    def select_ddp_strategy(
        self, num_devices: int, config: Dict[str, Any], model=None
    ) -> Union[str, DDPStrategy]:
        r"""Select the Lightning ``strategy`` from the configured parameter usage.

        A ``DDPStrategy`` instance rather than one of the ``'ddp'`` /
        ``'ddp_find_unused_parameters_true'`` shorthand strings, because those strings can express
        ``find_unused_parameters`` and nothing else, and two of the three settings this family
        wants (:meth:`ddp_kwargs`) have no shorthand. The instance is equivalent to the string it
        replaces on the one axis the string could carry.

        Everything is read from ``config`` and the ``model`` argument goes unused, which is
        deliberate rather than lazy: the framework passes the *Lightning module* here, not the
        raw net, so reaching for a net attribute would find nothing and silently regress a
        correct run to the slower strategy.

        Args:
            num_devices: Number of CUDA devices for the run.
            config: The resolved config.
            model: The Lightning module, unused. See above.

        Returns:
            ``'auto'`` on a single device -- there is no process group to configure -- otherwise a
            configured ``DDPStrategy``.
        """
        if num_devices <= 1:
            return "auto"
        return DDPStrategy(**self.ddp_kwargs(config))

    def train_model(self, train_loader, validation_loader):
        """Build this model's callbacks and run the fit.

        Args:
            train_loader: Training dataloader.
            validation_loader: Validation dataloader.

        Returns:
            The fitted Lightning ``Trainer``.
        """
        callbacks_config = self.config.get("advanced_config", {}).get("callbacks", {}) or {}
        checkpoint_config = callbacks_config.get("model_checkpoint", {}) or {}

        self.metrics_callback = MetricsLoggingCallback(tracked_metrics=self.TRACKED_METRICS)
        self.metrics_csv_callback = MetricsHistoryCsvCallback(
            source=self.metrics_callback, output_dir=self.train_results_dir
        )
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.plot_every_epoch,
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            # Explicit, because the default list names keys this model never emits: it asks for
            # a bare `kld_beta`, which the framework renames to `train/kld_beta` on the way out.
            # Left to default, every series is NaN and the beta ramp -- the knob the config
            # tells the operator to retune -- silently vanishes from hyperparameters.html.
            tracked_keys=("train/kld_beta", "lr"),
            output_dir=self.train_results_dir,
            plot_frequency=10,
            mlflow_logger=self.mlflow_logger,
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            # From config: hardcoding it here would make the config key a decoration.
            monitor=checkpoint_config.get("monitor", "val/total_loss"),
            # Lightning auto-prefixes each placeholder with its own name, so
            # "model-epoch={epoch}" would render as "model-epoch=epoch=00".
            filename=f"{self.CHECKPOINT_STEM}-{{epoch:02d}}",
            save_top_k=checkpoint_config.get("save_top_k", 3),
            mode=checkpoint_config.get("mode", "min"),
        )
        callback_list: List[Callback] = [
            self.metrics_callback,
            self.metrics_csv_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
            self.checkpoint_callback,
        ]

        # A SECOND checkpoint criterion, built only when a config names one. The composite optimum
        # and the best conditioned forecast are different epochs -- 336 against 278 on the H = 15
        # diagnosed run -- so a run keeping only the first has no copy of the weights that forecast
        # best, and nothing in its metrics recovers them once the epoch has passed.
        #
        # One optional key rather than a second full block: `save_top_k` and `mode` do not vary
        # between the two criteria (both are minimised nats, both keep three), and a duplicated
        # surface would be two places for them to disagree. The stem is derived from the monitor
        # rather than fixed, so the two ModelCheckpoints can never write the same filename however
        # many criteria a config eventually names -- Lightning would otherwise have each overwrite
        # the other's file at the same epoch, leaving one criterion's best silently unsaved.
        secondary_monitor = checkpoint_config.get("secondary_monitor")
        if secondary_monitor:
            self.secondary_checkpoint_callback = ModelCheckpoint(
                dirpath=self.model_checkpoint_dir,
                monitor=str(secondary_monitor),
                filename=(
                    f"{self.CHECKPOINT_STEM}-{_monitor_stem(secondary_monitor)}-{{epoch:02d}}"
                ),
                save_top_k=checkpoint_config.get("save_top_k", 3),
                mode=checkpoint_config.get("mode", "min"),
            )
            callback_list.append(self.secondary_checkpoint_callback)

        # The diagnostic plotter is opt-in and pulls matplotlib, so the class -- and with it the
        # import -- is resolved only when the config asks for it.
        plot_config = callbacks_config.get(self.PLOT_CONFIG_KEY, {}) or {}
        if plot_config.get("enabled", False):
            # Bound loosely: the seam's contract is "a Callback", while the keywords below are
            # this callback's own and a sibling's replacement is free to take more.
            plot_callback_cls: Any = self.plot_callback_cls()
            callback_list.append(
                plot_callback_cls(
                    output_dir=self.train_results_dir,
                    # The same knob the HTML loss plot above uses -- `general_config.
                    # plot_frequency` -- and deliberately not a second key under this block.
                    # The two figures are read together (the curves say a metric moved, the
                    # diagnostic says what the model was doing when it moved), so two
                    # independent cadences produce epochs where one exists and the other does
                    # not, and nothing marks which epochs those are.
                    plot_frequency=self.plot_every_epoch,
                    num_examples=plot_config.get("num_examples", 2),
                    file_format=plot_config.get("file_format", "pdf"),
                    mlflow_logger=self.mlflow_logger,
                )
            )

        trainer = self.build_trainer(callback_list, model=self.pl_model)
        trainer.fit(self.pl_model, train_loader, validation_loader)
        return trainer


def _monitor_stem(monitor: str) -> str:
    """Turn a monitored metric name into a filename fragment.

    ``val/nll_full_block`` becomes ``val-nll_full_block``. Only the stage separator is rewritten:
    a checkpoint filename is what an operator reads to decide which of two files to evaluate, and a
    fragment that dropped the stage or abbreviated the metric would need the config open beside it
    to be read at all.

    Args:
        monitor: The metric name, as Lightning receives it.

    Returns:
        The fragment, safe on every filesystem this trains on.
    """
    return str(monitor).replace("/", "-")


#: The driver a call to :func:`main` returns, which is the class it was handed rather than this
#: module's own -- a sibling cell's entry point delegates here with its own driver and gets that
#: driver back.
_TrainerT = TypeVar("_TrainerT", bound=LagAttnRwsTrainer)


def main(
    config_path: str, trainer_cls: type[_TrainerT] = LagAttnRwsTrainer
) -> _TrainerT:
    """Resolve the config, build everything, and run the fit.

    The call order is load-bearing and nothing chains it. ``setup_config`` is what seeds the
    run, creates the output directories, opens the log sinks and connects MLflow; building the
    model before it means no seeding, no logs, nowhere to write, and ``mlflow_logger is None``
    -- which silently drops the MLflow callback from the fit. The pre-flight guards run *before*
    ``setup_config`` so a doomed launch leaves no run directory and no MLflow run behind.

    ``trainer_cls`` is a parameter rather than a literal because the four pre-flight guards, the
    resolved-config persistence and the temporary-file dance around them are model-independent:
    a sibling architecture's entry point delegates here with its own driver instead of copying
    them, which is the only way they cannot drift. What such a driver adds of its own goes in its
    ``preflight`` classmethod, called here after the four and before ``setup_config``.

    Args:
        config_path: Path to the YAML config. Its ``base:`` chain is resolved first.
        trainer_cls: The driver class to construct. Defaults to this module's.

    Returns:
        The driver, after the fit. Returned rather than discarded because it is the only handle on
        where the run went: ``model_checkpoint_dir``, ``train_results_dir`` and the
        ``ModelCheckpoint`` callback's ``best_model_path`` are all decided inside ``setup_config``
        from a timestamped directory name, so a caller that needs the checkpoint it just produced
        would otherwise have to guess it by scanning the output tree for the newest directory --
        which is wrong the moment two runs land in the same second.
    """
    start_time = time.time()

    # Resolved to a file because the driver reads a path and takes no dict, and into a temporary
    # directory because the run directory does not exist until setup_config, which needs the
    # file first. Deleted on the way out: the durable record of what ran is the framework's own
    # config dump in the run log and the resolved_config.yaml MLflow artifact.
    with tempfile.TemporaryDirectory(prefix="teb_vae_rws_") as resolved_dir:
        resolved_path = resolve_config_file(config_path, resolved_dir)
        logger.info(f"resolved config {config_path} -> {resolved_path}")

        graph_model = trainer_cls(config_file_path=resolved_path)
        _check_stat_path(graph_model.config)
        _check_declared_widths_against_shard(graph_model.config)
        _check_raw_target_normalized(
            graph_model.config, fields=trainer_cls.TARGET_FIELDS
        )
        _check_causal_budget_resolves(graph_model.config)
        # After the four, before setup_config: the four are this architecture's and stay called
        # by name here (three other places import them individually and assert on this order),
        # while a sibling architecture's own guards go behind the hook. Handed the already-loaded
        # config, so the hook cannot read a different file than the one that was just checked.
        trainer_cls.preflight(graph_model.config)

        graph_model.setup_config()
        # After setup_config, which is what creates the run directories, and still inside the
        # temporary directory's lifetime.
        _persist_resolved_config(resolved_path, graph_model.model_checkpoint_dir)

    data_module = GraphDataModule(graph_model.config)

    graph_model.create_model()
    graph_model.train_model(data_module.train_dataloader(), data_module.val_dataloader())
    logger.info(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
    return graph_model


def _persist_resolved_config(resolved_path: str, checkpoint_dir: str) -> None:
    """Write the fully resolved config, plus the resolved causal guard, beside the checkpoints.

    A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive, and neither is a file the evaluation
    entry point can open. Written beside the checkpoints specifically, so a checkpoint directory
    copied off the production box carries what is needed to rebuild the run's data contract.

    The one addition to what the config file already says is :data:`RESOLVED_BUDGET_KEY`: the
    surviving channel indices and per-channel delays that ``causal_reach_budget_s`` resolved to.
    Without it a run records the budget it was *asked* for but not the guard it actually got, and
    the two are separated by a filter bank. It is a record and nothing reads it back; see the
    constant for why it is not written into ``VAE_model``.

    Rank $0$ only. Under ``torchrun`` every rank executes this module, and while the bytes are
    identical the writes are not atomic against each other.

    Args:
        resolved_path: The resolved config file, inside the caller's temporary directory.
        checkpoint_dir: The run's checkpoint directory, already created by ``setup_config``.
    """
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return
    destination = os.path.join(checkpoint_dir, RESOLVED_CONFIG_FILENAME)
    try:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        model_config = config.setdefault("model_config", {}) or {}
        budget = resolve_stream_budgets(model_config.get("VAE_model", {}) or {})
        model_config[RESOLVED_BUDGET_KEY] = None if budget is None else budget.as_record()
        config["model_config"] = model_config
        with open(destination, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        logger.info(f"wrote {destination}")
    except (OSError, yaml.YAMLError) as exc:
        # Not fatal: this file serves the *later* evaluation run, and refusing to train because a
        # write failed would cost the multi-day run it exists to describe.
        logger.warning(f"could not write {destination}: {exc}")


def _check_stat_path(config: Dict[str, Any]) -> None:
    """Refuse to start without a normalization statistics file that actually exists.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: If ``stat_path`` is unset or names a file that is not there.
    """
    dataset_config = config.get("dataset_config", {}) or {}
    stat_path = dataset_config.get("stat_path")
    if stat_path is None:
        # The loader passes None straight through and the dataset merely skips normalization,
        # so an absent or misspelled stat_path (the config key is `stat_path`; the loader's
        # parameter is `stats_path`) would otherwise train on raw-scale inputs -- and here the
        # raw-scale *target* -- and report nothing.
        raise ValueError(
            "dataset_config.stat_path must be set; without it normalization is silently "
            "disabled and the model trains on unnormalized inputs and an unnormalized raw "
            "target."
        )
    if not os.path.isfile(str(stat_path)):
        # Set-but-wrong is the same failure as unset, and it is the likelier one: the loader
        # emits `UserWarning: Statistics file not found ... Normalization disabled` and carries
        # on, so a mistyped or not-yet-generated path costs a full run.
        raise ValueError(
            f"dataset_config.stat_path does not exist: {stat_path!r}. The loader would only "
            f"warn and silently disable normalization. Generate the stats for this dataset "
            f"with hdf5_dataset/calculate_dataset_stats.py at trim_minutes=1.0, matching "
            f"dataloader_config.dataset_kwargs.trim_minutes."
        )


def _check_declared_widths_against_shard(config: Dict[str, Any]) -> None:
    r"""Compare the configured $c_y$ / $c_u$ against the first training shard, before the fit.

    The task already checks this against every real batch, which is the authoritative check.
    This one exists only to move the failure earlier: without it a width mismatch surfaces
    inside ``training_step``, by which point every rank has initialised and the run directory
    and MLflow run exist.

    Deliberately not fatal on anything but a genuine mismatch: a missing file, a missing field
    or an unreadable shard is left to the data module, which reports those far better than a
    pre-flight peek can.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: If a declared width disagrees with the shard's own channel counts.
    """
    dataset_config = config.get("dataset_config", {}) or {}
    shards = dataset_config.get("vae_train_datasets") or []
    vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    if not shards or "c_y" not in vae_config or "c_u" not in vae_config:
        return

    try:
        import h5py

        with h5py.File(str(shards[0]), "r") as handle:
            # Stored layout is (N, C, T); the loader transposes to (T, C) per sample.
            widths = {name: int(handle[name].shape[1]) for name in
                      ("fhr_st", "fhr_ph", "up_st", "up_ph")}
    except Exception:  # noqa: BLE001 - the data module reports these properly; see docstring.
        return

    use_up_st = bool(vae_config.get("use_up_st", True))
    expected_c_y = widths["fhr_st"] + widths["fhr_ph"]
    expected_c_u = widths["up_st"] + widths["up_ph"] if use_up_st else widths["up_ph"]

    problems = []
    if int(vae_config["c_y"]) != expected_c_y:
        problems.append(
            f"c_y={vae_config['c_y']} but the shard gives {expected_c_y} "
            f"(fhr_st={widths['fhr_st']} + fhr_ph={widths['fhr_ph']})"
        )
    if int(vae_config["c_u"]) != expected_c_u:
        composition = (
            f"up_st={widths['up_st']} + up_ph={widths['up_ph']}"
            if use_up_st
            else f"up_ph={widths['up_ph']}"
        )
        problems.append(
            f"c_u={vae_config['c_u']} but the shard gives {expected_c_u} "
            f"({composition}, use_up_st={use_up_st})"
        )
    if problems:
        raise ValueError(
            "model_config.VAE_model channel widths disagree with "
            f"{shards[0]}: " + "; ".join(problems) + ". The widths are a property of the HDF5: "
            "either fix the config or point dataset_config at the shards these widths were "
            "chosen for. Note 58 is both the current use_up_st=true c_u and the old phase-only "
            "one, so decide from use_up_st before trusting the number."
        )


def _check_raw_target_normalized(
    config: Dict[str, Any], *, fields: Sequence[str] = ("fhr",)
) -> None:
    """Refuse to start unless every field the target is built from is loaded *and* normalized.

    Without a target field in ``load_fields`` the task fails on the first batch (late, after
    every rank initialised); without it in ``normalize_fields`` nothing fails at all -- the
    target arrives at ~140 bpm scale, the Gaussian NLL is computed against a z-scale variance
    model, and the run trains a meaningless objective to completion.

    ``fields`` is a parameter rather than the literal ``'fhr'`` because which fields carry the
    target is the one thing a sibling architecture forecasting another domain changes about this
    check. The default is this model's, so the three places that call the guard with one argument
    -- the evaluation's preflight among them -- are unaffected; ``main`` passes the driver's
    ``TARGET_FIELDS``. The guard deliberately stays in ``main``'s by-name list rather than moving
    behind the ``preflight`` hook, whose documented no-op property is what makes a bare-bodied
    override safe.

    Args:
        config: The resolved run config.
        fields: Loader field names the reconstruction target is built from.

    Raises:
        ValueError: Naming the offending field and the offending list.
    """
    dataloader_config = (config.get("dataset_config", {}) or {}).get(
        "dataloader_config", {}
    ) or {}
    load_fields = (dataloader_config.get("dataset_kwargs", {}) or {}).get("load_fields") or []
    normalize_fields = dataloader_config.get("normalize_fields") or []

    # Field by field, so a config carrying one of a two-field target is refused naming the one
    # it is missing rather than the pair.
    missing = []
    for field in fields:
        if field not in load_fields:
            missing.append(
                f"'{field}' in dataset_config.dataloader_config.dataset_kwargs.load_fields"
            )
        if field not in normalize_fields:
            missing.append(
                f"'{field}' in dataset_config.dataloader_config.normalize_fields"
            )
    if missing:
        raise ValueError(
            "this model's reconstruction target requires " + " and ".join(missing) + ": an "
            "unloaded target field fails on the first batch and an unnormalized one makes the "
            "Gaussian NLL meaningless with nothing else raising."
        )


def _check_causal_budget_resolves(config: Dict[str, Any]) -> None:
    """Resolve the causal reach budget before anything is built, so a bad one fails here.

    The resolution is done again inside ``_build_model_kwargs`` -- it is a pure function of the
    config and the filter bank, so the two cannot disagree. Doing it once here moves a budget
    that keeps no channel, or whose delay exceeds ``warmup_period``, out of the middle of model
    construction (after directories, log sinks and an MLflow run exist) and into the pre-flight,
    where the message is the only thing on screen.

    Args:
        config: The resolved run config.

    Raises:
        ValueError: Propagated from the resolution, naming the config key and the two values.
    """
    resolve_stream_budgets((config.get("model_config", {}) or {}).get("VAE_model", {}) or {})


def _resolve_cli_config_path(config_path: str) -> str:
    """Resolve a command-line config path against the repository root.

    Every documented invocation runs from the repo root and uses repo-root-relative paths; an
    IDE's working directory is not something this module can rely on. Absolute paths pass
    through untouched.

    Args:
        config_path: The path as supplied on the command line or via ``RUN_CONFIG``.

    Returns:
        An absolute path.
    """
    if os.path.isabs(config_path):
        return config_path
    return os.path.join(_REPO_ROOT, config_path)


#: Config used when the module is launched with no ``--config`` -- i.e. an IDE's Run button.
#: ``--config`` on the command line always wins over this value. A relative path is resolved
#: against the repository root, not the working directory.
RUN_CONFIG: str | None = "teb_vae/lag_attn_rws/configs/default.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the YAML config, e.g. teb_vae/lag_attn_rws/configs/default.yaml. Run "
        "from the repo root. Optional only if RUN_CONFIG is set in this file (for an IDE Run "
        "button).",
    )
    _args = parser.parse_args()

    _config_path = _args.config or RUN_CONFIG
    if _config_path is None:
        parser.error(
            "--config is required. To launch from an IDE Run button instead, set RUN_CONFIG "
            "near the bottom of this file to a config path."
        )

    _config_path = _resolve_cli_config_path(_config_path)

    # The paths *inside* a config are repo-root-relative too (see configs/tiny.yaml), and under
    # an IDE Run button the working directory is whatever the IDE chose -- a relative shard path
    # then resolves to nothing and the loader dies as "No samples match the specified filters"
    # with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)

    if _args.config is None:
        logger.info(f"no --config given; using RUN_CONFIG={_config_path}")

    main(_config_path)
