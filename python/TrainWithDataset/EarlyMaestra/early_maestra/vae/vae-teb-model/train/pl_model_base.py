from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast
from abc import ABC, abstractmethod
from pathlib import Path

import lightning as L
import torch

from loguru import logger
from torch import nn
from torch.distributed import ReduceOp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

MetricDict = Dict[str, torch.Tensor]
"""Convenience alias used for the metric dictionaries each step returns."""


class LightningModelBase(L.LightningModule, ABC):
    """Base Lightning wrapper providing a repeatable training/validation skeleton.

    The class standardizes boilerplate required to convert an ordinary
    ``torch.nn.Module`` into a fully functional Lightning module. Sub-classes are
    encouraged to implement only the domain-specific logic, primarily
    ``compute_loss_and_metrics`` and optionally optimizer/scheduler builders or
    epoch hooks. Everything else—compiling the model, saving hyperparameters,
    logging learning-rate/metrics, and filtering trainable parameters—is handled
    centrally here so downstream Lightning modules remain concise.

    Metric logging workflow:

    * ``compute_loss_and_metrics`` must return ``(loss, metrics_dict)`` where keys
      are short metric names (``total_loss``, ``kld_loss``, ``beta`` ...).
    * ``LightningModelBase`` prefixes those keys with the trainer stage
      (``train/``, ``val/``, ``test/``) inside ``_log_metrics`` so every callback
      sees consistent names in ``trainer.callback_metrics``.
    * Built-in callbacks such as ``LossPlotCallback`` or Lightning's
      ``ModelCheckpoint`` simply reference those keys via their ``monitor``
      argument (e.g., ``monitor='val/total_loss'``) and need no extra wiring.
    * Any metric logged via ``self.log(...)`` or returned in ``metrics_dict`` is
      immediately available to externally supplied callbacks, so you can monitor
      reconstruction losses, hyper-parameters, or custom scalars just by adding
      them to the metrics dict.
    """

    prog_bar_metrics: Tuple[str, ...] = ("total_loss",)
    """Metric suffixes that should surface in Lightning's progress bar."""

    sync_dist_stages: Tuple[str, ...] = ("val", "test")
    """Trainer stages that require distributed metric synchronization."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        lr_gamma: float = 0.1,
        lr_warmup_epochs: int = 0,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
        compile_model: bool = True,
        spike_breaker: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            base_model: The raw ``nn.Module`` that performs inference and loss work.
            lr: Default learning rate stored in ``self.hparams``.
            lr_milestones: Optional milestone epochs for the scheduler helper.
            lr_gamma: Multiplicative LR decay applied at each milestone (default $0.1$).
            lr_warmup_epochs: When $> 0$, prepend a linear LR warmup of this many epochs
                before the milestone decay (see ``build_lr_scheduler``). Default $0$
                preserves the bare ``MultiStepLR`` behaviour.
            weight_decay: AdamW weight decay applied across parameters.
            module_name: Friendly name used in logs/debug messages.
            compile_model: When ``True`` (default, preserving historical behaviour)
                ``self.model`` is the ``torch.compile`` wrapper around ``base_model``.
                When ``False`` ``self.model`` is ``base_model`` itself, so subclasses
                that previously bypassed this constructor with a grandparent
                ``LightningModule.__init__`` hack to run eager can call
                ``super().__init__(..., compile_model=False)`` instead. Consumers
                source this flag from ``advanced_config.trainer.compile``.
            spike_breaker: Optional loss-spike circuit-breaker configuration mapping.
                ``None`` (default) leaves the breaker disabled and the training step
                byte-for-byte unchanged. When provided with ``enabled: true`` it turns
                on the train-only breaker described in ``_apply_spike_breaker``.
                Consumers source it from ``advanced_config.spike_breaker``.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])
        self._orig_model = base_model  # Reference to the original module before compilation/wrapping
        self._wrapper_name = module_name or self.__class__.__name__  # Used in logs to identify this wrapper
        # Compile is opt-out: the default keeps the compiled fast path; compile_model=False
        # leaves the eager module in place. torch.compile is lazy — it only invokes the
        # backend on the first forward — so constructing either path here is cheap.
        self.model = torch.compile(base_model) if compile_model else base_model

        # Loss-spike circuit-breaker running state. All zero/None until the first
        # training batch and only mutated when the breaker is enabled; kept here so the
        # counters survive across steps and are visible to logging.
        self._spike_ema_loss: Optional[float] = None
        self._spike_consecutive: int = 0
        self._spike_batches_seen: int = 0
        self._spike_skips_total: int = 0
        self._spike_forced_accepts_total: int = 0

    @property
    def orig_model(self) -> nn.Module:
        """Return the underlying, non-Lightning module."""
        return self._orig_model

    def forward(self, *args, **kwargs):
        """Delegate forwards to the wrapped PyTorch module."""
        return self.model(*args, **kwargs)

    def on_train_epoch_start(self) -> None:
        """Expose LR telemetry and let subclasses refresh state before a new epoch."""
        self._log_learning_rate()
        self._on_train_epoch_start_hook()

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Stamp the eager model's class name into every saved checkpoint.

        Records ``checkpoint["model_class"] = type(self._orig_model).__name__`` — the
        portable module class, unaffected by the ``_orig_mod.`` prefix ``torch.compile``
        would otherwise introduce. The stamp is additive metadata that lets downstream
        tooling identify which architecture produced a checkpoint; it never alters the
        weights.

        Subclasses that override this hook should call
        ``super().on_save_checkpoint(checkpoint)`` to keep the stamp alongside their own
        keys. Lightning mutates ``checkpoint`` in place, so this returns ``None``.
        """
        checkpoint["model_class"] = type(self._orig_model).__name__

    def training_step(self, batch, batch_idx):
        """Shared training step that delegates to ``compute_loss_and_metrics``."""
        return self._dispatch_stage_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        """Shared validation step mirroring ``training_step`` without grads."""
        return self._dispatch_stage_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        """Shared test step using the same metric dispatch path."""
        return self._dispatch_stage_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        """Create the optimizer + optional scheduler using helper builders."""
        # Materialise once: the overview and the optimizer both consume this, so a
        # ``configure_param_groups`` override that returns a lazy generator (e.g.
        # ``self.parameters()``) would otherwise be exhausted by the first consumer,
        # leaving the optimizer with an empty parameter list.
        param_groups = list(self.configure_param_groups())
        self._log_parameter_overview(param_groups)
        optimizer = self.build_optimizer(param_groups)
        scheduler = self.build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @abstractmethod
    def compute_loss_and_metrics(self, batch, batch_idx: int, stage: str) -> Tuple[torch.Tensor, MetricDict]:
        """Perform the forward/loss computation for the current stage.

        Implementations should:

        1. Run the wrapped model forward pass given the ``batch`` data.
        2. Compute the scalar loss tensor to backpropagate (typically named
            ``total_loss`` or similar).
        3. Build a metric dictionary where each value is either a tensor or a
            float/int convertible to tensor. Metrics can include the loss itself
            (e.g., ``{'total_loss': loss}``) or any auxiliary quantities.
        4. Return ``(loss, metrics)`` where ``loss`` participates in gradient
            calculation and ``metrics`` feeds the unified logging helper.

        Args:
            batch: The Lightning batch object passed into the stage step.
            batch_idx: Index of the batch within the epoch.
            stage: Literal string ``'train'``, ``'val'``, or ``'test'`` used to
                scope metric names (e.g., ``train/total_loss``).

        Returns:
            A tuple containing the scalar loss tensor and a dictionary of metrics
            to log. Missing metrics or ``None`` values are ignored gracefully.

        Example:
            >>> loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, "train")
            >>> metrics
            {
                "total_loss": loss,
                "recon_loss": recon_loss,
                "kld_loss": kld,
                "beta": beta_value,  # logged as train/beta
            }
            The helper will automatically prefix each key with the current stage
            unless the name already contains '/'.
        Example Implementation:
        ```Python
        (
            y_st,        # scattering inputs
            y_ph,        # phase-harmonic inputs
            x_ph,        # cross-phase inputs (if used)
            y_raw,       # raw waveform target
            meta,        # optional extra info (guid, epoch, etc.)
        ) = batch

        # forward pass through SeqVaeTeb (compiled handle for speed)
        outputs = self.model(
            y_st=y_st,
            y_ph=y_ph,
            x_ph=x_ph,
            meta=meta,
        )

        # SeqVaeTeb already exposes a loss helper
        loss_dict = self.orig_model.compute_loss(
            forward_outputs=outputs,
            y_raw=y_raw,
            y_st=y_st,
            y_ph=y_ph,
            beta_override=self.hparams.get("kld_beta"),
            log_forecast_metrics=self.hparams.get("log_forecast_metrics", True),
        )

        total_loss = loss_dict["total_loss"]

        metrics = {
            "total_loss": total_loss,
            "reconstruction_loss": loss_dict.get("reconstruction_loss"),
            "kld_loss": loss_dict.get("kld_loss"),
            "forecast_loss": loss_dict.get("forecast_loss"),
            "beta": loss_dict.get("beta"),
        }
        return total_loss, metrics
        ```
        """

    def configure_param_groups(
        self,
    ) -> Iterable[Union[torch.nn.Parameter, Dict[str, Any]]]:
        r"""Return the parameter collection the optimizer is built over.

        The default returns the flat list of parameters with ``requires_grad`` set —
        identical to the historical behaviour, so default runs are numerically
        unchanged. Override to return a list of optimizer *param-group dicts* when a
        model needs different hyperparameters per sub-module, e.g. an encoder and a
        head at learning rates $\eta_{\mathrm{enc}} \ne \eta_{\mathrm{head}}$::

            return [
                {"params": encoder_params, "lr": lr_encoder},
                {"params": head_params,    "lr": lr_head},
            ]

        ``build_optimizer`` consumes whatever this returns, so differential-LR
        subclasses can express their grouping here instead of bypassing the optimizer
        builder entirely. Group dicts use the base ``lr``/``weight_decay`` from
        ``self.hparams`` as the fallback for any key they omit.

        Returns:
            Either an iterable of ``torch.nn.Parameter`` (the default flat list) or a
            list of ``AdamW`` param-group dictionaries.
        """
        return self._trainable_parameters()

    def build_optimizer(
        self, trainable_params: Iterable[Union[torch.nn.Parameter, Dict[str, Any]]]
    ) -> Optimizer:
        """Construct the optimizer; override for custom optimizers.

        The default uses ``torch.optim.AdamW`` with the learning-rate and
        weight-decay pulled from ``self.hparams``. Sub-classes can override this
        method to:

        * Swap in entirely different optimizers (SGD, Adam, Lion, etc.).
        * Group parameters with different hyperparameters.
        * Introduce optimizer-specific keyword arguments.

        Always return an ``Optimizer`` instance ready to be consumed by
        Lightning's ``configure_optimizers`` flow.
        """
        lr = float(getattr(self.hparams, "lr", 1e-4))
        weight_decay = float(getattr(self.hparams, "weight_decay", 1e-4))
        # AdamW accepts either a flat parameter list or a list of group dicts; the
        # cast expresses that ``configure_param_groups`` may return either homogeneous
        # form, which the optimizer's stub type cannot represent as one iterable.
        return torch.optim.AdamW(
            cast(Any, list(trainable_params)),
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95),
        )

    def build_lr_scheduler(
        self,
        optimizer: Optimizer,
    ) -> Optional[Union[_LRScheduler, Dict[str, Union[_LRScheduler, str, int]]]]:
        r"""Optional epoch-wise LR scheduler builder with an opt-in warmup.

        Reads three hyperparameters, all with backward-compatible defaults:

        * ``lr_milestones`` — epochs at which the LR is multiplied by ``lr_gamma``.
        * ``lr_gamma`` — the decay factor (default $0.1$).
        * ``lr_warmup_epochs`` — when $> 0$, prepend a linear warmup.

        Behaviour:

        * No milestones and no warmup → ``None`` (no scheduler).
        * ``lr_warmup_epochs <= 0`` → a bare ``MultiStepLR`` (historical behaviour).
        * ``lr_warmup_epochs > 0`` → a ``LinearLR`` ramp over the warmup epochs
          composed with a ``MultiStepLR`` via ``SequentialLR``. The milestones are
          shifted back by ``lr_warmup_epochs`` (``max(0, m - warmup)``) because
          ``SequentialLR`` restarts the second scheduler's epoch counter at the
          switch point; without the shift the decay would fire ``lr_warmup_epochs``
          epochs too late, so the shift keeps each decay at its intended *absolute*
          epoch.

        Override to return ``None``, a plain scheduler, or the richer dict Lightning
        expects when extra metadata (interval/frequency) is required.
        """
        milestones = list(getattr(self.hparams, "lr_milestones", None) or [])
        warmup_epochs = int(getattr(self.hparams, "lr_warmup_epochs", 0) or 0)
        gamma = float(getattr(self.hparams, "lr_gamma", 0.1))

        if not milestones and warmup_epochs <= 0:
            return None

        from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

        if warmup_epochs <= 0:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            # Shift milestones so the decay lands at the intended absolute epoch: SequentialLR
            # resets the decay scheduler's epoch clock to 0 at the switch (epoch warmup_epochs).
            decay = MultiStepLR(
                optimizer,
                milestones=[max(0, m - warmup_epochs) for m in milestones],
                gamma=gamma,
            )
            scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_epochs])

        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

    def _dispatch_stage_step(self, batch, batch_idx: int, stage: str):
        """Helper shared by train/val/test steps for consistent logging.

        Centralizes the boilerplate of calling ``compute_loss_and_metrics`` and
        logging the returned metrics. Sub-classes generally should not override
        ``training_step``/``validation_step``/``test_step`` directly unless they
        need non-standard behavior—customization typically happens inside
        ``compute_loss_and_metrics``.

        When the loss-spike circuit breaker is enabled it runs here — but only on the
        ``train`` stage. The dispatcher is shared with val/test, and fabricating a loss
        on a spiky *validation* batch would poison ``val/total_loss`` and corrupt the
        ``ModelCheckpoint`` selection, so the breaker never touches val/test.
        """
        loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, stage)
        spike_cfg = self.hparams.get("spike_breaker", None) if hasattr(self, "hparams") else None
        if stage == "train" and spike_cfg and spike_cfg.get("enabled", False):
            loss = self._apply_spike_breaker(loss, metrics, spike_cfg)
        self._log_metrics(metrics, stage=stage, on_step=True)
        return loss

    def _apply_spike_breaker(
        self, loss: torch.Tensor, metrics: MetricDict, cfg: Dict[str, Any]
    ) -> torch.Tensor:
        r"""Loss-spike circuit breaker (train-only); returns the loss to backpropagate.

        Folds the behaviour that VAE consumers previously hand-rolled as a
        ``training_step`` override. Tracks an exponential moving average (EMA) of a
        watched loss and, when the current loss spikes far above it, performs a
        *zero-gradient step* instead of a real update. The step is realised as a loss
        summed over **every** trainable parameter times $0$ — not ``None`` — because
        the forward already armed DDP's gradient reducer, which then expects a hook to
        fire for each parameter; a bare ``None`` (a true skip) would desynchronise the
        reducer under DDP.

        Config keys (all optional, defaults in parentheses):

        * ``multiplier`` ($5.0$) — spike when
          $\ell > \mathrm{multiplier}\cdot\max(\mathrm{EMA},\,\mathrm{ema\_floor})$.
        * ``ema_decay`` ($0.02$) — EMA momentum $m$ in
          $\mathrm{EMA}\leftarrow m\,\ell + (1-m)\,\mathrm{EMA}$, updated only on
          accepted batches so a skipped spike never pollutes the baseline.
        * ``ema_floor`` ($0.0$) — floor on the comparison base only, so a collapsed EMA
          cannot make the threshold unreachably small.
        * ``warmup_batches`` ($100$) — never skip during the first this-many batches
          (the EMA has not stabilised yet).
        * ``max_consecutive_skips`` ($25$) — after this many consecutive skips,
          force-accept the next batch to break a frozen-EMA deadlock; on force-accept
          the EMA is hard re-seeded to the current loss (a momentum step barely moves a
          collapsed EMA).
        * ``comparison_metric`` (``"total_loss"``) — ``"total_loss"`` watches the
          returned loss; ``"main_loss"`` watches ``metrics["main_loss"]`` (for models
          whose returned loss is periodically perturbed by an auxiliary term).

        Args:
            loss: The scalar loss tensor from ``compute_loss_and_metrics``.
            metrics: The metric dict; spike diagnostics are added in place for logging.
            cfg: The resolved ``spike_breaker`` config mapping.

        Returns:
            Either the original ``loss`` (accepted / force-accepted) or a
            zero-gradient no-op loss (skipped).
        """
        multiplier = float(cfg.get("multiplier", 5.0))
        ema_decay = float(cfg.get("ema_decay", 0.02))
        ema_floor = float(cfg.get("ema_floor", 0.0))
        warmup_batches = int(cfg.get("warmup_batches", 100))
        max_consecutive_skips = int(cfg.get("max_consecutive_skips", 25))
        comparison_metric = str(cfg.get("comparison_metric", "total_loss"))

        # Select the watched value. main_loss is a perm-free scalar some models expose
        # in metrics; fall back to the returned loss if it is absent.
        watched = loss
        if comparison_metric == "main_loss":
            candidate = metrics.get("main_loss")
            if candidate is not None:
                watched = candidate
        loss_value = float(watched.detach().item()) if isinstance(watched, torch.Tensor) else float(watched)
        # The non-finite guard must protect the loss that is actually backpropagated
        # (``loss``/total_loss), not only the watched value: in ``main_loss`` mode the
        # watched main_loss can stay finite while total_loss blows up, and returning a
        # non-finite loss would corrupt every weight — the exact failure the breaker exists
        # to prevent. So a spike is forced when EITHER value is non-finite.
        returned_value = float(loss.detach().item()) if isinstance(loss, torch.Tensor) else float(loss)
        is_nonfinite = not (math.isfinite(loss_value) and math.isfinite(returned_value))

        self._spike_batches_seen += 1
        ema_before = self._spike_ema_loss

        # Local skip decision.
        if is_nonfinite:
            is_spike_local = True
        elif self._spike_batches_seen <= warmup_batches or ema_before is None:
            is_spike_local = False
        else:
            ema_ref = max(ema_before, ema_floor)
            is_spike_local = loss_value > multiplier * ema_ref

        # Skip if ANY rank skips (MAX): ranks must agree so their optimizer steps stay
        # in lockstep under DDP.
        is_spike = self._reduce_spike_decision(is_spike_local, "max")

        # Escape hatch: only after the cap, only if EVERY rank agrees (MIN), and never
        # when a loss is non-finite (a finite forced step is safe, a NaN one is not).
        forced = False
        if is_spike:
            forced_local = (
                not is_nonfinite
                and max_consecutive_skips > 0
                and self._spike_consecutive >= max_consecutive_skips
            )
            forced = self._reduce_spike_decision(forced_local, "min")

        if forced:
            self._spike_ema_loss = loss_value  # hard re-seed the collapsed EMA
            self._spike_consecutive = 0
            self._spike_forced_accepts_total += 1
            is_spike = False
        elif is_spike:
            self._spike_consecutive += 1
            self._spike_skips_total += 1
        else:
            # Accept: advance (or seed) the EMA and clear the run length.
            if ema_before is None:
                self._spike_ema_loss = loss_value
            else:
                self._spike_ema_loss = ema_decay * loss_value + (1.0 - ema_decay) * ema_before
            self._spike_consecutive = 0

        # Diagnostics for the logger (tensors so the metric-logging contract holds).
        # ``spike_ema_loss`` (instantaneous) and ``spike_skipped`` (0/1, whose epoch mean
        # is the skip rate) aggregate meaningfully; the monotonic cumulative counters are
        # left as instance attributes (``self._spike_skips_total`` /
        # ``self._spike_forced_accepts_total``) rather than logged, because an on_epoch
        # mean of a running total is meaningless.
        ema_report = self._spike_ema_loss if self._spike_ema_loss is not None else 0.0
        metrics["spike_ema_loss"] = torch.tensor(float(ema_report), device=self.device)
        metrics["spike_skipped"] = torch.tensor(1.0 if is_spike else 0.0, device=self.device)

        if is_spike:
            # Replace the poisoned loss metric(s) with the finite EMA so a skipped
            # NaN/huge value does not corrupt the epoch-aggregated training curve (the
            # optimizer never saw it — see the zero-gradient step below).
            safe_loss = torch.tensor(float(ema_report), device=self.device)
            if "total_loss" in metrics:
                metrics["total_loss"] = safe_loss
            if comparison_metric == "main_loss" and "main_loss" in metrics:
                metrics["main_loss"] = safe_loss

            # Zero-gradient step touching every trainable parameter. The forward already
            # armed DDP's reducer, which expects a gradient hook per parameter; a no-op
            # built from all of them keeps every hook firing. nan_to_num keeps the value
            # finite if a parameter itself went NaN (its gradient is 0 regardless), and
            # the *0.0 makes every gradient exactly zero. Note this is a *zero-gradient*
            # step, not a true skip: under automatic optimization the optimizer still
            # steps, so AdamW's decoupled weight decay and carried momentum still nudge
            # the weights slightly — matching what the consumer breakers do today and
            # kept because returning ``None`` (a true skip) is DDP-unsafe once the reducer
            # is armed.
            zero_grad_loss = torch.zeros((), device=self.device)
            for param in self.parameters():
                if param.requires_grad:
                    zero_grad_loss = zero_grad_loss + torch.nan_to_num(param).sum()
            return zero_grad_loss * 0.0
        return loss

    def _reduce_spike_decision(self, flag: bool, op: str) -> bool:
        """Reduce a boolean skip/force-accept decision across DDP ranks.

        No-op (returns ``flag`` unchanged) when the module is unattached or running on a
        single process. Otherwise reduces a $\\{0,1\\}$ tensor through the trainer
        strategy with ``ReduceOp.MAX`` (``op="max"`` — skip if *any* rank skips) or
        ``ReduceOp.MIN`` (``op="min"`` — force-accept only if *every* rank agrees).
        """
        trainer = getattr(self, "_trainer", None)
        strategy = getattr(trainer, "strategy", None) if trainer is not None else None
        world_size = getattr(strategy, "world_size", 1) if strategy is not None else 1
        if strategy is None or not world_size or world_size <= 1:
            return flag
        reduce_op = ReduceOp.MAX if op == "max" else ReduceOp.MIN
        flag_tensor = torch.tensor(1.0 if flag else 0.0, device=self.device)
        reduced = strategy.reduce(flag_tensor, reduce_op=reduce_op)
        value = reduced.item() if isinstance(reduced, torch.Tensor) else float(reduced)
        return value > 0.5

    def _log_learning_rate(self) -> None:
        """Report the first parameter group's LR once per epoch."""
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        if not optimizer or not optimizer.param_groups:
            return
        lr_value = optimizer.param_groups[0].get("lr")
        if lr_value is None:
            return
        self.log("lr", lr_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def _log_metrics(self, metrics: MetricDict, *, stage: str, on_step: bool) -> None:
        """Unified metric logger framing keys as ``stage/name``."""
        if not metrics:
            return
        for raw_name, value in metrics.items():
            if value is None:
                continue
            name = raw_name if "/" in raw_name else f"{stage}/{raw_name}"
            metric_tensor = self._as_tensor(value)
            prog_bar = self._should_log_on_prog_bar(name)
            sync_dist = stage in self.sync_dist_stages
            self.log(name, metric_tensor, on_step=on_step, on_epoch=True, prog_bar=prog_bar, logger=True, sync_dist=sync_dist)

    def _trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        """Collect parameters with ``requires_grad`` for optimizer construction."""
        return [param for param in self.parameters() if param.requires_grad]

    def _log_parameter_overview(
        self, trainable_params: Iterable[Union[torch.nn.Parameter, Dict[str, Any]]]
    ) -> None:
        """Emit a short breakdown of trainable vs total parameters.

        Accepts either a flat parameter iterable or a list of optimizer param-group
        dicts (as ``configure_param_groups`` may return); group dicts are flattened via
        their ``"params"`` entry so the ``numel`` accounting never sees a dict.
        """
        trainable_params = list(trainable_params)
        # Flatten any param-group dicts so numel accounting only iterates real tensors.
        flat_trainable = [
            param
            for item in trainable_params
            for param in (item["params"] if isinstance(item, dict) else [item])
        ]
        total_params = sum(param.numel() for param in self.parameters())
        trainable_count = sum(param.numel() for param in flat_trainable)
        frozen_count = total_params - trainable_count
        if total_params == 0:
            logger.warning("[{}] No parameters detected in model", self._wrapper_name)
            return
        logger.info("=" * 80)
        logger.info("[{}] Parameter overview", self._wrapper_name)
        logger.info("  Total parameters: {:,}", total_params)
        logger.info("  Trainable parameters: {:,} ({:.2f}%)", trainable_count, 100.0 * trainable_count / total_params)
        logger.info("  Frozen parameters: {:,} ({:.2f}%)", frozen_count, 100.0 * frozen_count / total_params)
        logger.info("=" * 80)

    def _should_log_on_prog_bar(self, name: str) -> bool:
        """Check whether the metric suffix is part of ``prog_bar_metrics``."""
        metric_name = name.split("/")[-1]
        return metric_name in self.prog_bar_metrics

    def _as_tensor(self, value) -> torch.Tensor:
        """Convert scalars/None into detached tensors for ``self.log``."""
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (float, int)):
            tensor = torch.tensor(float(value), device=self.device)
        else:
            tensor = torch.tensor(0.0, device=self.device)
        return tensor.detach()

    def _on_train_epoch_start_hook(self) -> None:
        """Optional hook for subclasses that need to refresh schedulers or state."""
