r"""The training task: the loss, the metrics, and the validation-only permutation control.

This is the only place that knows both the net and the data. The net takes tensors and knows
nothing about batches or config; the experiment driver knows about config and nothing about
tensors; this module turns one into the other and computes the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
  + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}} + \lambda_{\Delta} \mathcal{L}_{\Delta}
  + \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}},
$$

with the first four terms in nats per anchor (the net's ``compute_loss`` owns the reduction
convention) and the three shape terms in $L_1$/Huber units, which makes ``total_loss`` a
mixed-unit criterion and the ``nll_*`` metrics the pure-nats readouts. Each shape weight
defaults to $0.0$, at which its term is not computed and its metric is an exact zero.

Everything else is inherited. There is no ``training_step`` here: the framework's own step
dispatches to :meth:`compute_loss_and_metrics`, logs the returned metrics, and runs the
loss-spike circuit breaker from ``advanced_config.spike_breaker``. ``compile_model`` **defaults**
to ``False`` and that is the only correct value for *this* net -- its LSTM encoders defeat
TorchInductor on their own, and a checkpointed attention region defeats it again whenever
``attention_grad_checkpoint`` is set.

It is a default rather than a hard-wired constant because those are two facts about this
architecture, not about the objective, and a subclass over a different net may have neither. The
third blocker once recorded here -- the data-dependent mask indexing behind ``kld_active_frac`` --
is **not** a reason for any subclass: it lives in ``compute_loss``, which
:meth:`compute_loss_and_metrics` reaches through ``self.orig_model``. Only the forward is ever
compiled, so that indexing stays eager by construction rather than by a graph break.

The permutation control runs on **validation batches only** and never enters the loss. It is a
readout: the shuffled-forecast score ``nll_shuffled_block`` against ``nll_base_block`` and
``nll_full_block`` is the negative control the acceptance ordering
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ is read from. Its metrics are
*absent*, never zero-filled, on steps where it did not run: the framework logs every metric with
``on_epoch=True``, whose epoch value is the mean over the steps that reported it, so a zero
placeholder would scale the epoch aggregate down and corrupt the very ordering the control
exists to check.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from loguru import logger
from torch import nn

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask
from train.pl_model_base import LightningModelBase, MetricDict


class SeqVaeLagAttnRwsTask(LightningModelBase):
    r"""Lightning task for :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.

    Implements :meth:`compute_loss_and_metrics` and the checkpoint contract, and nothing else:
    the optimizer, the scheduler, the metric logging, the spike breaker and the ``model_class``
    stamp all come from the base.
    """

    #: Metric suffixes shown on the progress bar. ``source_conditioned_kl_raw`` rather than the
    #: trained KL: the raw value is the only one readable as an information rate, and watching
    #: the floored one hides a collapsed source pathway.
    prog_bar_metrics: Tuple[str, ...] = (
        "total_loss",
        "nll_full_block",
        "source_conditioned_kl_raw",
    )

    #: Step stride of the gradient-norm hook: the norm costs one reduction kernel per
    #: parameter, and computing it on every step buys nothing the CSV records (see
    #: :meth:`on_before_optimizer_step` for what IS recorded). The epoch's last batch always
    #: logs regardless of the stride, which is what keeps the CSV's sample exact.
    GRAD_NORM_LOG_EVERY_N_STEPS: int = 25

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Any] = None,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
        spike_breaker: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        beta_schedule: Optional[Dict[str, Any]] = None,
        kld_beta: float = 1.0,
        beta_prior: float = 0.0,
        lambda_full: float = 1.0,
        lambda_base: float = 1.0,
        likelihood: str = "gaussian_nll",
        free_bits: float = 0.0,
        lambda_ms: float = 0.0,
        lambda_deriv: float = 0.0,
        lambda_boundary: float = 0.0,
        compile_model: bool = False,
    ) -> None:
        r"""Initialize the task.

        Every argument except ``base_model`` lands in ``self.hparams`` and therefore in the
        checkpoint, so a run's objective is recoverable from its checkpoint alone. There is
        deliberately no observation-noise knob: under ``'gaussian_nll'`` the decoder's learned
        log-variance heads *are* the observation model, unconditionally.

        Args:
            base_model: The net to wrap.
            lr: Learning rate.
            lr_milestones: Epoch milestones for the LR scheduler.
            weight_decay: AdamW weight decay.
            module_name: Friendly name used in logs.
            spike_breaker: The ``advanced_config.spike_breaker`` block, consumed by the base.
                ``None`` or ``{'enabled': False}`` disables the breaker.
            model_kwargs: The exact constructor kwargs used to build ``base_model``, written
                into every checkpoint so the architecture can be rebuilt without a config file.
            beta_schedule: Structured $\beta$ schedule; see :meth:`_resolve_beta`. ``None``
                falls back to the constant ``kld_beta``.
            kld_beta: Constant $\beta$ used when no schedule is configured.
            beta_prior: Weight $\beta_p$ of the prior scale rate. A constant, never a
                schedule: the prior-variance collapse this term prevents completes within the
                first epoch, so a warm-up would arrive after the damage. ``0.0`` restores the
                historical three-term objective exactly.
            lambda_full: Weight of the full (source-conditioned) reconstruction term.
            lambda_base: Weight of the base (target-only) reconstruction term.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-dim per-step KL floor in nats; enters the trained KL only.
            lambda_ms: Weight $\lambda_{\mathrm{ms}}$ of the multiscale $L_1$ shape term. At
                ``0.0`` -- the default -- the term is not computed and its metric is an exact
                zero. Constants, never schedules, for the same reason ``beta_prior`` is one:
                these shape the mean the reconstruction is already training, so nothing is
                gained by letting them arrive late.
            lambda_deriv: Weight $\lambda_{\Delta}$ of the derivative Huber shape term.
            lambda_boundary: Weight $\lambda_{\mathrm{boundary}}$ of the boundary-continuity
                shape term.
            compile_model: Wrap the net in ``torch.compile``. Defaults to ``False``, which is
                the only correct value **for this net**: its LSTM encoders defeat TorchInductor
                on their own, and a checkpointed attention region defeats it again whenever
                ``attention_grad_checkpoint`` is set. The keyword exists because those are
                facts about *this architecture*, not about the objective or the training step,
                and a subclass whose net has neither may pass ``True``.

                What compiling does **not** reach, in any subclass: only the forward is
                compiled. :meth:`compute_loss_and_metrics` runs the objective through
                ``self.orig_model``, so the data-dependent boolean mask indexing behind
                ``kld_active_frac`` and the prior-floor watch stays eager by construction
                rather than by a graph break. That separation is what makes the keyword safe
                to offer at all.
        """
        super().__init__(
            base_model,
            lr=lr,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            module_name=module_name,
            compile_model=compile_model,
            spike_breaker=spike_breaker,
        )
        self.save_hyperparameters(
            "beta_schedule",
            "kld_beta",
            "beta_prior",
            "lambda_full",
            "lambda_base",
            "likelihood",
            "free_bits",
            "lambda_ms",
            "lambda_deriv",
            "lambda_boundary",
        )
        self._model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        self._perm_generator: Optional[torch.Generator] = None
        self._peak_memory_logged = False

    def setup(self, stage: str) -> None:
        """Seed the derangement generator, once the module knows its rank.

        Deliberately not ``__init__``: the module is unattached there, so ``self.global_rank``
        is ``0`` on every rank and a rank-seeded generator would produce identical shuffles
        everywhere while claiming otherwise. Ranks hold different data, so their shuffles should
        differ; the run/skip *decision* is what must stay rank-invariant, and that is reduced in
        :meth:`_sync_perm_decision`.
        """
        if self._perm_generator is None:
            self._perm_generator = torch.Generator()
            self._perm_generator.manual_seed(1234 + int(self.global_rank))

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Log this rank's peak CUDA memory once, right after the first training step.

        The experiment driver resets the CUDA peak-memory counters before the fit and nothing
        else ever reads them back, so without this line a run reports no memory telemetry at
        all. The first step is the honest high-water mark to size against: it covers the
        weights, the optimizer state and one full forward/backward at the configured batch.
        Both counters are reported because their gap is diagnostic -- reserved far above
        allocated points at allocator fragmentation, which has a different fix than a model
        that is simply too large.

        Args:
            outputs: The step output Lightning passes through; unused.
            batch: The batch Lightning passes through; unused.
            batch_idx: The batch index within the epoch; unused (a flag gates the log so it
                fires once per run, not once per epoch).
        """
        if self._peak_memory_logged or self.device.type != "cuda":
            return
        self._peak_memory_logged = True
        allocated_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
        reserved_gib = torch.cuda.max_memory_reserved(self.device) / 2**30
        logger.info(
            f"peak CUDA memory after the first training step (rank {self.global_rank}): "
            f"{allocated_gib:.2f} GiB allocated, {reserved_gib:.2f} GiB reserved"
        )

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        r"""Log the pre-clip gradient $L_2$ norm and whether it exceeded the clip, per step.

        ``advanced_config.trainer.gradient_clip_val`` ships at a provisional $250$ -- scaled from
        the sibling's $0.5$ by the $\approx 2 \times 480$ change in loss magnitude, not measured
        -- and the config says to re-derive it from the first real run. Nothing else records the
        quantity it must be derived from: Lightning removed ``track_grad_norm`` in 2.x, so a run
        reports the clip threshold it used and never what it was clipping. A threshold far below
        the typical norm rescales every step (the run is really training at a much lower
        effective learning rate); far above it, the clip is decoration and a genuine blow-up
        reaches the weights.

        Two columns, because they answer different questions. ``train/grad_norm`` is the norm
        itself; whether the threshold *bound* on a given step is not recoverable from it.
        ``train/grad_clip_frac`` logs the per-step exceedance indicator -- $1.0$ is
        normalised-gradient descent in disguise, $0.0$ is a clip that never fires. Omitted
        entirely when the trainer configures no clipping -- an absent threshold *or* a
        non-positive one, which is what Lightning itself treats as disabled -- since a fraction
        against no threshold would be an answer to no question.

        **What reaches ``metrics_history.csv`` is one step per epoch, not the epoch's
        aggregate.** ``MetricsLoggingCallback`` reads ``trainer.callback_metrics`` from
        ``on_validation_epoch_end``, which runs before the *training* epoch is reduced, so for a
        metric logged from this hook with both ``on_step`` and ``on_epoch`` the bare key still
        holds the last step's value. That is a usable sample rather than a defect -- a threshold
        is a per-step question, and one step per epoch is a usable draw from the per-step
        distribution -- but it governs how the columns are read: ``grad_clip_frac`` is $0$ or $1$
        per row and its **mean over epochs** estimates the exceedance fraction. Every ``val/``
        column is a true epoch mean; these two are the exception.

        Because only that one-step-per-epoch sample is ever recorded, the hook runs every
        :attr:`GRAD_NORM_LOG_EVERY_N_STEPS` optimizer steps rather than every step -- one norm
        reduction kernel per parameter is pure overhead on the steps in between. The epoch's
        last batch is exempt from the stride, so the value the CSV samples is computed on
        exactly the step it always sampled. The on_epoch aggregate and the MLflow step series
        become every-Nth systematic samples of the per-step distribution, which reads the same
        way as before for both columns.

        The hook order is what makes the numbers right: Lightning's precision plugin calls
        this hook and *then* ``_clip_gradients``, so what is measured here is the pre-clip norm --
        the only version comparable against ``gradient_clip_val``. Under DDP the gradients have
        already been all-reduced by the end of the backward, so every rank measures the same
        value and no synchronisation is needed. On a batch the spike breaker skipped the norm is
        $\approx 0$ by construction; ``train/spike_skipped`` is the column that says so.

        Args:
            optimizer: The optimizer Lightning is about to step; unused (the gradients are read
                off the module's own parameters, which is the same set).
        """
        trainer = self.trainer
        if not (
            trainer.is_last_batch
            or trainer.global_step % self.GRAD_NORM_LOG_EVERY_N_STEPS == 0
        ):
            return
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach())
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return
        grad_norm = torch.linalg.vector_norm(torch.stack(norms))
        self.log(
            "train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        clip_val = self.trainer.gradient_clip_val
        # ``> 0`` and not merely ``is not None``, because that is the predicate Lightning itself
        # clips on: ``Precision.clip_gradients`` returns immediately for a non-positive threshold,
        # so a config carrying ``gradient_clip_val: 0`` -- the "0 disables it" convention, which
        # still appears elsewhere in this repository -- clips nothing while every norm exceeds
        # zero. Gated on ``is not None`` alone that run would report an exceedance fraction of
        # 1.000 in every row: the one value read as "the threshold rescaled every step", for a run
        # with no threshold at all.
        if clip_val is not None and float(clip_val) > 0.0:
            self.log(
                "train/grad_clip_frac",
                (grad_norm > float(clip_val)).to(grad_norm.dtype),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

    # ------------------------------------------------------------------
    # Batch -> model inputs
    # ------------------------------------------------------------------
    def _build_target_streams(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(fhr_st, fhr_ph)``, having checked their joint width against $c_y$.

        Checked here rather than in the net's constructor because this is the only place that
        can see both: the model's ``c_y`` is a number from a config file, this is the data.

        Args:
            batch: A batch from the data module.

        Returns:
            The target scattering and phase-harmonic blocks.

        Raises:
            RuntimeError: If their concatenated width disagrees with the model's ``c_y``.
        """
        y_st, y_ph = batch.fhr_st, batch.fhr_ph
        expected = int(self.orig_model.c_y)
        got = int(y_st.shape[-1]) + int(y_ph.shape[-1])
        if got != expected:
            raise RuntimeError(
                f"target stream is {got} channels (fhr_st={int(y_st.shape[-1])} + "
                f"fhr_ph={int(y_ph.shape[-1])}) but the model was built with c_y={expected}. "
                f"These widths come from the HDF5, not from the model: either set "
                f"model_config.VAE_model.c_y={got}, or point dataset_config at the shards this "
                f"c_y was chosen for."
            )
        return y_st, y_ph

    def _build_source_stream(self, batch: Any) -> torch.Tensor:
        r"""Assemble the source stream $u$ consumed by the net's forward.

        With ``use_up_st=True`` the stream is ``[up_st, up_ph]`` concatenated along the channel
        axis; otherwise it is ``up_ph`` alone.

        Args:
            batch: A batch from the data module.

        Returns:
            The source stream, $(B, T, c_u)$.

        Raises:
            RuntimeError: If a field the model's configuration requires is absent from the
                batch, or if the assembled stream is not as wide as the model was built for.
        """
        up_ph = getattr(batch, "up_ph", None)
        if up_ph is None:
            raise RuntimeError(
                "batch has no `up_ph` field. Add 'up_ph' to dataset_kwargs.load_fields in the "
                "config, and check the HDF5 files were built by the pipeline that writes up_ph "
                "as a first-class dataset."
            )
        if not bool(getattr(self.orig_model, "use_up_st", False)):
            return self._checked_source(up_ph, up_st=None, up_ph=up_ph)
        up_st = getattr(batch, "up_st", None)
        if up_st is None:
            raise RuntimeError(
                "the model was built with use_up_st=True but the batch has no `up_st` field. "
                "Either add 'up_st' to dataset_kwargs.load_fields, rebuild the HDF5 with up_st, "
                "or set use_up_st=false and c_u to up_ph's own width in model_config.VAE_model."
            )
        return self._checked_source(
            torch.cat([up_st, up_ph], dim=-1), up_st=up_st, up_ph=up_ph
        )

    def _checked_source(
        self,
        stream: torch.Tensor,
        *,
        up_st: Optional[torch.Tensor],
        up_ph: torch.Tensor,
    ) -> torch.Tensor:
        """Return the source stream, having checked its width against the model's ``c_u``.

        Args:
            stream: The assembled source stream.
            up_st: The scattering block, or ``None`` under the ``use_up_st=False`` ablation.
            up_ph: The phase-harmonic block.

        Returns:
            ``stream`` unchanged.

        Raises:
            RuntimeError: If the stream's width disagrees with the model's ``c_u``.
        """
        expected, got = int(self.orig_model.c_u), int(stream.shape[-1])
        if got == expected:
            return stream
        breakdown = (
            f"up_ph={int(up_ph.shape[-1])}"
            if up_st is None
            else f"up_st={int(up_st.shape[-1])} + up_ph={int(up_ph.shape[-1])}"
        )
        raise RuntimeError(
            f"source stream is {got} channels ({breakdown}) but the model was built with "
            f"c_u={expected} (use_up_st={bool(self.orig_model.use_up_st)}). These widths come "
            f"from the HDF5, not from the model: either set model_config.VAE_model.c_u={got}, "
            f"or point dataset_config at the shards this c_u was chosen for. Note 58 is both "
            f"the current use_up_st=true width and the old phase-only width, so decide from "
            f"use_up_st before trusting the number."
        )

    def _build_forward_inputs(self, batch: Any) -> Tuple[torch.Tensor, ...]:
        """Return the positional tensors the net's ``forward`` takes, in order.

        The one seam between a batch and a net's input signature. This model feeds its net the
        two target feature blocks and the assembled source stream; a sibling architecture over a
        different *input representation* keeps this whole class -- the objective, the metric
        surface, the permutation control, the checkpoint contract -- and overrides only this
        method. Everything downstream reads ``inputs[0]`` for the batch size and device rather
        than a named tensor, so an override is free to return tensors of a different rank
        entirely.

        Deliberately not merged with :meth:`_build_raw_target`: the raw FHR is the
        reconstruction *target*, scored by the loss, and it stays behind its own builder even for
        an architecture that also feeds it to the net. One source of the target is what stops a
        model from being scored against a tensor other than the one it was shown.

        Args:
            batch: A batch from the data module.

        Returns:
            ``(y_st, y_ph, u_stream)``, splatted into the net's forward.
        """
        y_st, y_ph = self._build_target_streams(batch)
        return y_st, y_ph, self._build_source_stream(batch)

    def _build_raw_target(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fhr, weight)``: the raw reconstruction target and its validity signal.

        Unlike the feature-target sibling, both fields are hard requirements here: the raw
        signal *is* the target, and the decimated ``weight`` is the only trustworthy gap signal
        (gaps are stored as $0$ bpm, roughly $-11\\sigma$ after z-scoring -- not a detectable
        sentinel).

        Args:
            batch: A batch from the data module.

        Returns:
            The raw target ``(B, L_raw)`` and the validity weight ``(B, T)``.

        Raises:
            RuntimeError: If either field is absent, naming the config key that fixes it.
        """
        fhr = getattr(batch, "fhr", None)
        if fhr is None:
            raise RuntimeError(
                "batch has no `fhr` field, and the raw FHR is this model's reconstruction "
                "target. Add 'fhr' to dataset_kwargs.load_fields AND to "
                "dataloader_config.normalize_fields -- without the latter the target arrives "
                "in bpm and the Gaussian NLL is meaningless, with nothing raising."
            )
        weight = getattr(batch, "weight", None)
        if weight is None:
            raise RuntimeError(
                "batch has no `weight` field. The decimated weight is the only trustworthy "
                "validity signal for the raw target; add 'weight' to "
                "dataset_kwargs.load_fields."
            )
        return fhr, weight

    # ------------------------------------------------------------------
    # Beta schedule
    # ------------------------------------------------------------------
    def _resolve_beta(self, epoch: int) -> float:
        r"""Resolve the KL weight $\beta$ for ``epoch``.

        Supported ``beta_schedule.kind`` values:

        * ``constant`` -- returns ``beta_schedule.value`` when present, else the ``kld_beta``
          hparam.
        * ``linear_warmup`` -- ramps linearly from ``start`` to ``end`` over the first
          ``warmup_epochs`` epochs, then holds:

          $$\beta(e) = \mathrm{start} + (\mathrm{end} - \mathrm{start})
                       \min\!\left(1, \frac{e}{\mathrm{warmup\_epochs}}\right).$$

        The warm-up starts at $0$ here by configuration, not by code: $z$ is the only route to
        the decoder, so a nonzero $\beta$ before the decoder can use the latent at all is the
        standard route to posterior collapse.

        Args:
            epoch: The current training epoch.

        Returns:
            The scalar $\beta$ weighting the trained KL this epoch.

        Raises:
            ValueError: If ``beta_schedule.kind`` is not a supported value. Silently falling
                back to a constant would train a different objective than the config describes.
        """
        schedule = self.hparams.get("beta_schedule")
        if not isinstance(schedule, dict):
            return float(self.hparams.get("kld_beta", 1.0))
        kind = str(schedule.get("kind", "constant"))
        if kind == "constant":
            value = schedule.get("value")
            return float(value) if value is not None else float(self.hparams.get("kld_beta", 1.0))
        if kind == "linear_warmup":
            start = float(schedule.get("start", 0.0))
            end = float(schedule.get("end", 1.0))
            warmup_epochs = int(schedule.get("warmup_epochs", 50))
            if warmup_epochs <= 0:
                return end
            fraction = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
            return start + (end - start) * fraction
        raise ValueError(
            f"unknown beta_schedule.kind={kind!r}; expected 'constant' or 'linear_warmup'."
        )

    # ------------------------------------------------------------------
    # Validation-only permutation control
    # ------------------------------------------------------------------
    def _should_run_perm(self, batch_size: int, stage: str) -> bool:
        """Decide whether to evaluate the permutation control on this step, locally.

        Validation only, every batch: the control is a readout and it is cheap under
        ``no_grad``. It never runs on a training batch and never contributes to the objective.
        A degenerate batch ($B < 2$) cannot be deranged and skips.

        Args:
            batch_size: Local batch size.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            Whether this rank *can* run the control, before the cross-rank reduction.
        """
        return stage != "train" and batch_size >= 2

    def _sync_perm_decision(self, do_perm: bool, device: torch.device) -> bool:
        """MIN-reduce ``do_perm`` so no rank runs the control alone.

        The control's metrics are logged with ``sync_dist=True`` on validation, so a rank that
        logs ``val/kld_shuffled`` while a peer (whose last uneven batch is degenerate) does not
        would hang the metric sync. The reduction makes the decision collective: run only if
        every rank can.

        Args:
            do_perm: This rank's local decision.
            device: Device for the reduction tensor.

        Returns:
            The decision every rank agrees on.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return do_perm
        flag = torch.tensor([1.0 if do_perm else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() > 0.0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _mu_gap_rms(
        self, forward_outputs: Dict[str, torch.Tensor], weight: torch.Tensor
    ) -> torch.Tensor:
        r"""Masked RMS of the per-step latent mean gap $\lVert \mu^q_t - \mu^p_t \rVert_2$.

        Uses the KL's own anchor support (rebuilt through the same two functions the loss uses,
        at the model's own ``coverage_floor``, so the two cannot drift): anchors with no
        reconstruction term -- the tail $H$, and any anchor the gap masking or the coverage
        floor drops -- have nothing pulling the posterior off the prior, and averaging the gap
        over them would read systematically low against the reported KL. Distinct from
        ``delta_mu_rms`` (the per-element RMS the net reports): this one sums over $d_z$ first,
        so it is the size of the *belief shift* per step rather than a per-coordinate figure.

        Args:
            forward_outputs: The net's forward dict.
            weight: Per-step validity, $(B, T)$.

        Returns:
            A scalar tensor.
        """
        with torch.no_grad():
            model = self.orig_model
            forecast, _coverage = forecast_mask(
                weight, model.geometry, coverage_floor=model.coverage_floor
            )
            support = kl_mask(forecast, model.geometry)
            gap_sq = (
                (forward_outputs["mu_post"] - forward_outputs["mu_prior"]) ** 2
            ).sum(dim=-1)
            return torch.sqrt((gap_sq * support).sum() / support.sum().clamp_min(1.0))

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, MetricDict]:
        r"""Run the forward pass, build the loss, and report the diagnostics.

        Args:
            batch: A batch from the data module.
            batch_idx: Index of the current batch.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            ``(loss, metrics)``. The loss is always the perm-free objective -- the control never
            enters it on any stage -- and ``metrics['main_loss']`` carries its detached value
            under exactly that name, which is what the spike breaker watches.
        """
        inputs = self._build_forward_inputs(batch)
        fhr_raw, weight = self._build_raw_target(batch)

        forward_outputs = self.model(*inputs)

        beta = self._resolve_beta(self.current_epoch)
        beta_prior = float(self.hparams.get("beta_prior", 0.0))
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 1.0))
        likelihood = str(self.hparams.get("likelihood", "gaussian_nll"))
        free_bits = float(self.hparams.get("free_bits", 0.0))
        lambda_ms = float(self.hparams.get("lambda_ms", 0.0))
        lambda_deriv = float(self.hparams.get("lambda_deriv", 0.0))
        lambda_boundary = float(self.hparams.get("lambda_boundary", 0.0))

        loss_metrics = self.orig_model.compute_loss(
            forward_outputs,
            fhr_raw,
            weight=weight,
            beta=beta,
            beta_prior=beta_prior,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            free_bits=free_bits,
            lambda_ms=lambda_ms,
            lambda_deriv=lambda_deriv,
            lambda_boundary=lambda_boundary,
        )["metrics"]
        main_loss = loss_metrics["total_loss"]

        # The net's metric dict is all tensors by contract (the likelihood string lives outside
        # it), so splatting it is safe; the task adds what only it can see.
        metrics: Dict[str, Any] = dict(loss_metrics)
        # Unprefixed and detached: the breaker watches metrics['main_loss'] by exact name and
        # falls back to the returned loss, silently, if it is missing.
        metrics["main_loss"] = main_loss.detach()
        metrics["mu_prior_sat_frac"] = forward_outputs["mu_prior_sat_frac"]
        metrics["delta_mu_sat_frac"] = forward_outputs["delta_mu_sat_frac"]
        metrics["mu_post_prior_gap_rms"] = self._mu_gap_rms(forward_outputs, weight)

        # ``inputs[0]`` rather than a named tensor: the batch size and the device are properties
        # of any of the net's inputs, and reading them off the first one is what keeps the
        # control working for a subclass whose ``_build_forward_inputs`` returns something else.
        #
        # The reduction is skipped on training steps entirely, not merely answered False: the
        # control is validation-only, the stage is identical on every rank, so each rank skips
        # the collective in lockstep -- while calling ``_sync_perm_decision`` here would cost an
        # ``all_reduce`` plus a ``.item()`` GPU sync on EVERY training step to confirm a
        # constant False, stalling the CPU behind the forward each time.
        do_perm = stage != "train" and self._sync_perm_decision(
            self._should_run_perm(inputs[0].size(0), stage), inputs[0].device
        )
        if do_perm:
            # Readout only, under no_grad: the shuffled branch re-scores the forecast under a
            # stranger's source and never touches the objective.
            with torch.no_grad():
                permuted = controls.perm_forward_outputs(
                    self.orig_model, forward_outputs, generator=self._perm_generator
                )
                shuffled = self.orig_model.compute_loss(
                    permuted,
                    fhr_raw,
                    weight=weight,
                    beta=0.0,
                    # Zero like beta: the control re-scores the full branch under a stranger's
                    # source and leaves the prior untouched, so no objective weight belongs here.
                    beta_prior=0.0,
                    lambda_full=lambda_full,
                    lambda_base=lambda_base,
                    likelihood=likelihood,
                    free_bits=0.0,
                    # Zero for the same reason as beta_prior, and one more: only this call's NLL
                    # is consumed, so computing the shape terms here would buy a full extra pass
                    # over both forecast blocks for numbers nothing reads.
                    lambda_ms=0.0,
                    lambda_deriv=0.0,
                    lambda_boundary=0.0,
                )["metrics"]
            metrics["nll_shuffled_block"] = shuffled["nll_full_block"]
            # Raw, not floored: only the raw KL is readable as an information rate, and the
            # shuffled one is reported with the same semantics so the two curves compare.
            metrics["kld_shuffled"] = shuffled["source_conditioned_kl_raw"]
            metrics["shuffle_penalty"] = (
                shuffled["nll_full_block"] - loss_metrics["nll_full_block"].detach()
            )
        # No else-branch zero-fill: an epoch-aggregated metric is the mean over the steps that
        # reported it, so zeros on skipped steps would scale it toward nothing and invert the
        # D_full < D_base < D_shuffled reading on a healthy model.

        return main_loss, metrics

    # ------------------------------------------------------------------
    # Checkpoint contract
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Add the constructor kwargs beside the base's ``model_class`` stamp.

        With both fields present a checkpoint is self-describing: the architecture can be
        rebuilt from ``model_kwargs`` with no config file, and ``check_model_class`` can refuse
        a blob written by a different model before that rebuild is attempted.

        Args:
            checkpoint: The checkpoint dict, mutated in place.
        """
        super().on_save_checkpoint(checkpoint)  # stamps model_class; must run first
        checkpoint["model_kwargs"] = dict(self._model_kwargs)
