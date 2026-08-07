r"""The training task: the loss, the metrics, and the source-permutation control.

This is the only place that knows both the net and the data. The net takes tensors and knows
nothing about batches or config; the experiment driver knows about config and nothing about
tensors; this module turns one into the other and computes the objective:

$$
L = \lambda_{\mathrm{full}} L_{\mathrm{feat}} + \lambda_{\mathrm{base}} L_{\mathrm{base}}
  + \beta(e)\, L_{\mathrm{KL}} + \lambda_{\mathrm{lag}} L_{\mathrm{smooth}}
  \;+\; \lambda_{\mathrm{perm}} L_{\mathrm{perm}} .
$$

Everything else is inherited. There is no ``training_step`` here: the framework's own step already
dispatches to :meth:`compute_loss_and_metrics`, logs the returned metrics, and runs the loss-spike
circuit breaker from ``advanced_config.spike_breaker``. There is no
``pl.LightningModule.__init__`` bypass either -- ``compile_model=False`` is a constructor argument
now, and the model requires it permanently (its LSTM encoders, the checkpointed attention, and the
data-dependent mask indexing behind ``kld_active_frac`` each defeat TorchInductor independently).

Four behaviours differ from a hand-rolled step that did its own spike handling, all deliberate:

1. On a skipped batch the framework overwrites ``metrics["total_loss"]`` and ``metrics["main_loss"]``
   with the running EMA rather than the raw spiked value, so the logged series stays readable.
2. ``spike_skips_total`` and ``spike_forced_accepts_total`` are no longer emitted; ``spike_skipped``
   and ``spike_ema_loss`` carry the same signal per step.
3. The framework guards the *returned* loss for non-finiteness as well as the watched one, which is
   strictly stricter.
4. There is no per-skip log line. The metrics carry it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from teb_vae.lag_attn.nets import controls
from train.pl_model_base import LightningModelBase, MetricDict


class SeqVaeLagAttnTask(LightningModelBase):
    r"""Lightning task for :class:`~teb_vae.lag_attn.nets.model.SeqVaeLagAttn`.

    Implements :meth:`compute_loss_and_metrics` and the checkpoint contract, and nothing else: the
    optimizer, the scheduler, the metric logging, the spike breaker and the ``model_class`` stamp
    all come from the base.
    """

    #: Metric suffixes shown on the progress bar. ``kld_raw`` rather than ``kld_loss``: the former
    #: is the transfer-entropy surrogate, the latter is the free-bit-floored quantity that enters
    #: the loss, and watching the wrong one hides a collapsed source pathway.
    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss", "kld_raw")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
        spike_breaker: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        beta_schedule: Optional[Dict[str, Any]] = None,
        kld_beta: float = 0.01,
        lambda_full: float = 1.0,
        lambda_base: float = 0.5,
        likelihood: str = "gaussian_nll",
        sigma_obs: Any = "learned",
        free_bits: float = 0.0,
        detach_baseline_in_full: bool = False,
        lambda_lag: float = 0.0,
    ) -> None:
        r"""Initialize the task.

        Every argument except ``base_model`` lands in ``self.hparams`` and therefore in the
        checkpoint, so a run's objective is recoverable from its checkpoint alone.

        Args:
            base_model: The net to wrap.
            lr: Learning rate.
            lr_milestones: Epoch milestones for the LR scheduler.
            weight_decay: AdamW weight decay.
            module_name: Friendly name used in logs.
            spike_breaker: The ``advanced_config.spike_breaker`` block, consumed by the base.
                ``None`` or ``{'enabled': False}`` disables the breaker.
            model_kwargs: The exact constructor kwargs used to build ``base_model``, written into
                every checkpoint so the architecture can be rebuilt without a config file.
            beta_schedule: Structured $\beta$ schedule; see :meth:`_resolve_beta`. ``None`` falls
                back to the constant ``kld_beta``.
            kld_beta: Constant $\beta$ used when no schedule is configured.
            lambda_full: Weight of the source-conditioned forecast term.
            lambda_base: Weight of the source-free baseline term.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            sigma_obs: Observation-noise scalar, or ``'learned'`` to consume the decoder
                log-variance heads.
            free_bits: Per-dim per-step KL floor, in nats.
            detach_baseline_in_full: Whether to stop-gradient the baseline inside the full term.
            lambda_lag: Weight of the lag-embedding smoothness penalty.
        """
        super().__init__(
            base_model,
            lr=lr,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            module_name=module_name,
            # Permanent, not a default: four independent things in this net break inductor.
            compile_model=False,
            spike_breaker=spike_breaker,
        )
        self.save_hyperparameters(
            "beta_schedule",
            "kld_beta",
            "lambda_full",
            "lambda_base",
            "likelihood",
            "sigma_obs",
            "free_bits",
            "detach_baseline_in_full",
            "lambda_lag",
        )
        self._model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        self._perm_generator: Optional[torch.Generator] = None

    def setup(self, stage: str) -> None:
        """Seed the derangement generator, once the module knows its rank.

        Deliberately not ``__init__``: the module is unattached there, so ``self.global_rank`` is
        ``0`` on every rank and a rank-seeded generator would produce identical shuffles everywhere
        while claiming otherwise. Ranks hold different data, so their shuffles should differ; the
        *schedule* is what must stay rank-invariant, and that is decided by ``batch_idx``.
        """
        if self._perm_generator is None:
            self._perm_generator = torch.Generator()
            self._perm_generator.manual_seed(1234 + int(self.global_rank))

    # ------------------------------------------------------------------
    # Batch -> model inputs
    # ------------------------------------------------------------------
    def _build_source_stream(self, batch: Any) -> torch.Tensor:
        r"""Assemble the source stream $u$ consumed by the net's forward.

        With ``use_up_st=True`` the stream is ``[up_st, up_ph]`` concatenated along the channel
        axis, $(B, T, 58)$; otherwise it is ``up_ph`` alone, $(B, T, 15)$. Both are independent
        first-class HDF5 datasets.

        Args:
            batch: A batch from the data module.

        Returns:
            The source stream, $(B, T, c_u)$.

        Raises:
            RuntimeError: If a field the model's configuration requires is absent from the batch,
                or if the assembled stream is not as wide as the model was built for. The net
                would otherwise fail later with a channel-count error naming neither the missing
                field nor the config key that would fix it.
        """
        up_ph = getattr(batch, "up_ph", None)
        if up_ph is None:
            raise RuntimeError(
                "batch has no `up_ph` field. Add 'up_ph' to dataset_kwargs.load_fields in the "
                "config, and check the HDF5 files were built by the pipeline that writes up_ph as "
                "a first-class dataset."
            )
        if not bool(getattr(self.orig_model, "use_up_st", False)):
            return self._checked_source(up_ph, up_st=None, up_ph=up_ph)
        up_st = getattr(batch, "up_st", None)
        if up_st is None:
            raise RuntimeError(
                "the model was built with use_up_st=True but the batch has no `up_st` field. "
                "Either add 'up_st' to dataset_kwargs.load_fields, rebuild the HDF5 with up_st, or "
                "set use_up_st=false and c_u to up_ph's own width in model_config.VAE_model."
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
        r"""Return the source stream, having checked its width against the model's $c_u$.

        Checked here rather than in the net's constructor because this is the only place that can
        see both. The model's $c_u$ is a number from a config file; this is the data. The check
        that used to live in the constructor compared the config against *module constants* --
        that is, against another copy of the config -- so it went stale the moment the dataset
        pipeline changed its phase-harmonic selection, and it could never catch a correct config
        pointed at a stale shard.

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

    def _build_target_streams(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return ``(fhr_st, fhr_ph)``, having checked their joint width against $c_y$.

        The mirror of :meth:`_build_source_stream` for the target. ``c_y`` had no validation of
        any kind before the widths moved to the data boundary.

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

        A weak early $\beta$ lets the residual decoder learn to use $z$ before the bottleneck
        tightens for a calibrated TE reading.

        Args:
            epoch: The current training epoch.

        Returns:
            The scalar $\beta$ weighting ``kld_loss`` this epoch.

        Raises:
            ValueError: If ``beta_schedule.kind`` is not a supported value. Silently falling back
                to a constant would train a different objective than the config describes.
        """
        schedule = self.hparams.get("beta_schedule")
        if not isinstance(schedule, dict):
            return float(self.hparams.get("kld_beta", 0.01))
        kind = str(schedule.get("kind", "constant"))
        if kind == "constant":
            value = schedule.get("value")
            return float(value) if value is not None else float(self.hparams.get("kld_beta", 0.01))
        if kind == "linear_warmup":
            start = float(schedule.get("start", 1.0e-4))
            end = float(schedule.get("end", 0.1))
            warmup_epochs = int(schedule.get("warmup_epochs", 50))
            if warmup_epochs <= 0:
                return end
            fraction = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
            return start + (end - start) * fraction
        raise ValueError(
            f"unknown beta_schedule.kind={kind!r}; expected 'constant' or 'linear_warmup'."
        )

    # ------------------------------------------------------------------
    # Source-permutation control
    # ------------------------------------------------------------------
    def _sync_perm_decision(self, do_perm: bool, device: torch.device) -> bool:
        """MIN-reduce ``do_perm`` so no rank runs the control alone.

        ``batch_idx`` is already rank-invariant, but a rank whose local batch is degenerate
        ($B < 2$) cannot be deranged -- the last batch of an unevenly-sized shard is exactly that.
        Without this reduction that rank builds a different autograd graph from its peers and the
        all-reduce in ``backward`` deadlocks: no error, no output, the job simply stops.

        Args:
            do_perm: This rank's local decision.
            device: Device for the reduction tensor.

        Returns:
            The decision every rank agrees on: run the control only if all of them can.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return do_perm
        flag = torch.tensor([1.0 if do_perm else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() > 0.0)

    def _should_run_perm(self, batch_idx: int, batch_size: int, stage: str) -> bool:
        r"""Decide whether to evaluate the source-permutation control on this step.

        The control runs as a readout regardless of $\lambda_{\mathrm{perm}}$; the weight only
        decides whether it also enters the loss. Validation runs it every step -- ``kld_shuffled``
        and ``feat_loss_shuffled`` are the headline diagnostics and are cheap under ``no_grad``.
        Training subsamples it on a rank-invariant ``batch_idx`` schedule.

        Args:
            batch_idx: Index of the current batch.
            batch_size: Local batch size.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            Whether to run the control, before the cross-rank reduction.
        """
        if batch_size < 2:
            return False  # a derangement needs at least two samples to swap
        return self._perm_scheduled(batch_idx, stage)

    def _perm_scheduled(self, batch_idx: int, stage: str) -> bool:
        """The rank-invariant half of the control decision: stage and batch schedule.

        Split out of :meth:`_should_run_perm` so the call site can skip the cross-rank
        reduction whenever the schedule itself says no: every rank sees the same stage and
        the same ``batch_idx`` sequence, so on an unscheduled step they all agree without a
        collective -- while reducing anyway would cost an ``all_reduce`` plus a ``.item()``
        GPU sync on most training steps to confirm a constant False. Only the batch-size
        viability (the degenerate last batch of an uneven shard) differs per rank, and that
        is what the reduction remains for on scheduled steps.

        Args:
            batch_idx: Index of the current batch.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            Whether the schedule selects this step, ignoring batch-size viability.
        """
        if stage != "train":
            return True
        every = max(int(self.orig_model.perm_every_n_batches), 1)
        return batch_idx % every == 0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _compute_residual_diagnostics(
        self, *, forward_outputs: Dict[str, torch.Tensor], weight: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        r"""Return the masked RMS of the source-driven mean shift and of the latent mean gap.

        Both use the same masking rules as the loss terms they sit beside, so the numbers are
        directly comparable run to run:

        * ``delta_mu_rms`` uses the full (warmup $\times$ anchor $\times$ future) mask that
          $L_{\mathrm{feat}}$ uses.
        * ``mu_post_prior_gap_rms`` uses the KL's own time support, taken from the model rather
          than rebuilt here. Rebuilding it is how the two drift apart: under
          ``kld_support: anchor`` the KL is reduced over $[\mathrm{warmup}, T - H_d)$, and a
          warm-up-only mask would average the gap over the final $H_d$ steps too -- exactly the
          steps whose posterior is pulled to the prior with no forecast term pulling back. The
          gap would then read systematically low against a ``kld_raw`` computed elsewhere, which
          is the opposite of comparable.

        Args:
            forward_outputs: The net's forward dict.
            weight: Per-step validity, $(B, T)$, or ``None``.

        Returns:
            A dict with ``delta_mu_rms`` and ``mu_post_prior_gap_rms``, both scalars.
        """
        delta_mu = forward_outputs["delta_mu_src"]            # (B, T, Hd, C)
        batch, seq_len, horizon, _ = delta_mu.shape
        n_anchors = seq_len - horizon
        device, dtype = delta_mu.device, delta_mu.dtype

        # --- Feature-window mask (as in feat_loss) --------------------------
        warmup = int(self.orig_model._warmup_steps(seq_len))
        warmup_mask = torch.zeros(n_anchors, dtype=dtype, device=device)
        if warmup < n_anchors:
            warmup_mask[warmup:] = 1.0

        if weight is not None:
            w = weight.to(device=device, dtype=dtype)
            anchor_w = w[:, :n_anchors]                                    # (B, T_valid)
            target_w = w[:, 1:].unfold(dimension=1, size=horizon, step=1)  # (B, T_valid, Hd)
            feat_mask = warmup_mask[None, :, None] * anchor_w[:, :, None] * target_w
        else:
            feat_mask = warmup_mask[None, :, None].expand(batch, n_anchors, horizon)

        delta_sq = (delta_mu[:, :n_anchors] ** 2).sum(dim=-1)              # (B, T_valid, Hd)
        delta_mu_rms = torch.sqrt(
            (delta_sq * feat_mask).sum() / feat_mask.sum().clamp_min(1.0)
        )

        # --- Latent-window mask: the KL's own support, not a copy of it ------
        mu_prior = forward_outputs["mu_prior"]                             # (B, T, d_z)
        mu_post = forward_outputs["mu_post"]
        full_len = mu_prior.size(1)
        time_mask = self.orig_model._kld_support_mask(full_len, device=device, dtype=dtype)
        latent_mask = time_mask.unsqueeze(0).expand(batch, full_len)
        if weight is not None:
            latent_mask = latent_mask * weight.to(device=device, dtype=dtype)

        gap_sq = ((mu_post - mu_prior) ** 2).sum(dim=-1)                   # (B, T)
        mu_post_prior_gap_rms = torch.sqrt(
            (gap_sq * latent_mask).sum() / latent_mask.sum().clamp_min(1.0)
        )

        return {"delta_mu_rms": delta_mu_rms, "mu_post_prior_gap_rms": mu_post_prior_gap_rms}

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, MetricDict]:
        r"""Run the forward pass, build the loss, and report the diagnostics.

        The permutation control is fused into this single forward rather than re-running it: the
        source encoder is batch-independent, so permuting ``source_state`` is exactly equivalent to
        re-encoding a permuted source, and it keeps the whole step inside one forward and one
        backward. That is not merely an optimisation -- under plain ``'ddp'`` a second backward
        that does not touch every parameter raises.

        Args:
            batch: A batch from the data module.
            batch_idx: Index of the current batch.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            ``(loss, metrics)``, where ``loss`` is
            $L_{\mathrm{main}} + \lambda_{\mathrm{perm}} L_{\mathrm{perm}}$ on scheduled training
            steps and $L_{\mathrm{main}}$ otherwise. ``metrics['main_loss']`` always carries the
            detached perm-free value, which is what the spike breaker watches.
        """
        y_st, y_ph = self._build_target_streams(batch)
        u_stream = self._build_source_stream(batch)
        # Per-step validity. Gaps (weight ~ 0) would otherwise pollute every loss term, and the KL
        # curve is only trustworthy if they do not.
        weight = getattr(batch, "weight", None)

        forward_outputs = self.model(y_st, y_ph, u_stream)

        beta = self._resolve_beta(self.current_epoch)
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 0.5))
        likelihood = str(self.hparams.get("likelihood", "gaussian_nll"))
        sigma_obs = self.hparams.get("sigma_obs", "learned")
        if not isinstance(sigma_obs, str):
            sigma_obs = float(sigma_obs)
        free_bits = float(self.hparams.get("free_bits", 0.0))
        detach_baseline_in_full = bool(self.hparams.get("detach_baseline_in_full", False))
        lambda_lag = float(self.hparams.get("lambda_lag", 0.0))

        loss_dict = self.orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            weight=weight,
            beta=beta,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            sigma_obs=sigma_obs,
            free_bits=free_bits,
            detach_baseline_in_full=detach_baseline_in_full,
            lambda_lag=lambda_lag,
        )
        main_loss = loss_dict["total_loss"]
        kld_raw = loss_dict["kld_raw"]
        diagnostics = self._compute_residual_diagnostics(
            forward_outputs=forward_outputs, weight=weight
        )

        # Built key by key, not splatted from loss_dict: that dict carries `likelihood`, a str, and
        # the metric logger coerces a non-numeric value to a clean 0.0 rather than raising.
        metrics: Dict[str, Any] = {
            "total_loss": main_loss,
            # Unprefixed and perm-free: the breaker watches metrics['main_loss'] by exact name and
            # falls back to the returned loss, silently, if it is missing.
            "main_loss": main_loss.detach(),
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            # Only kld_raw may be read as a TE surrogate; kld_train is the free-bit-floored
            # quantity that actually enters total_loss.
            "kld_raw": kld_raw,
            "kld_train": loss_dict["kld_train"],
            "kld_active_frac": loss_dict["kld_active_frac"],
            "kld_beta": beta,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            # Watch these for variance collapse (sigma^2 -> exp(logvar_clamp[0])).
            "mean_logvar_full": loss_dict["mean_logvar_full"],
            "mean_logvar_base": loss_dict["mean_logvar_base"],
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "delta_mu_rms": diagnostics["delta_mu_rms"],
            "mu_post_prior_gap_rms": diagnostics["mu_post_prior_gap_rms"],
            "pred_gap": loss_dict["base_loss"] - loss_dict["feat_loss"],
            "lag_smoothness": loss_dict["lag_smoothness"],
        }

        lambda_perm = float(self.orig_model.lambda_perm)
        # The schedule short-circuit skips the collective in lockstep on unscheduled steps;
        # see _perm_scheduled for why that is DDP-safe.
        do_perm = self._perm_scheduled(batch_idx, stage) and self._sync_perm_decision(
            self._should_run_perm(batch_idx, y_st.size(0), stage), y_st.device
        )
        train_loss = main_loss
        if do_perm:
            # Build an autograd graph for the control only when it actually enters the loss.
            optimise_perm = stage == "train" and lambda_perm > 0.0
            with torch.set_grad_enabled(optimise_perm and torch.is_grad_enabled()):
                perm = controls.perm_kl_from_forward(
                    self.orig_model,
                    forward_outputs,
                    weight=weight,
                    generator=self._perm_generator,
                )
            kld_shuffled = perm["kld_shuffled"]
            if optimise_perm:
                train_loss = main_loss + lambda_perm * perm["perm_kl"]
                metrics["perm_loss"] = (lambda_perm * perm["perm_kl"]).detach()
            else:
                metrics["perm_loss"] = torch.zeros_like(kld_shuffled)
            metrics["kld_shuffled"] = kld_shuffled
            metrics["kld_shuffled_ratio"] = kld_shuffled / kld_raw.detach().clamp_min(1e-8)
            metrics["total_loss"] = train_loss

            # The prediction-space control. The KL-space one does not discriminate: a deranged UP
            # is still a UP, and the posterior -- trained only on matched pairs -- reacts to it out
            # of distribution, so K_shuffled >= K_true even when the source is genuinely used. The
            # forecast tells the truth. A model exploiting the source has
            #     feat_loss  <  base_loss  <  feat_loss_shuffled,
            # i.e. a wrong source is worse than no source at all.
            with torch.no_grad():
                permuted = controls.perm_forward_outputs(
                    self.orig_model, forward_outputs, perm_index=perm["perm_index"]
                )
                shuffled = self.orig_model.compute_loss(
                    forward_outputs=permuted,
                    y_st=y_st,
                    y_ph=y_ph,
                    weight=weight,
                    compute_kld_loss=False,
                    beta=0.0,
                    lambda_full=lambda_full,
                    lambda_base=lambda_base,
                    likelihood=likelihood,
                    sigma_obs=sigma_obs,
                    detach_baseline_in_full=detach_baseline_in_full,
                )
            metrics["feat_loss_shuffled"] = shuffled["feat_loss"]
            metrics["shuffle_penalty"] = shuffled["feat_loss"] - loss_dict["feat_loss"]
        # No else-branch zero-fill. Filling the control metrics with zeros on the steps the control
        # did not run would look like a harmless placeholder and is not: the framework logs every
        # metric with `on_epoch=True`, whose epoch value is the *mean* over the steps that reported
        # it. Zeros on 3 of every 4 training steps therefore scale the epoch-aggregated
        # `train/feat_loss_shuffled` and `train/shuffle_penalty` to a quarter of their real value,
        # which inverts the ordering the whole control exists to check
        # (feat_loss < base_loss < feat_loss_shuffled) and reads as a collapsed source pathway on a
        # perfectly healthy model. Omitted, the mean is taken over the perm steps alone and is
        # right. Validation runs the control on every step, so its series is dense either way.

        return train_loss, metrics

    # ------------------------------------------------------------------
    # Checkpoint contract
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Add the constructor kwargs beside the base's ``model_class`` stamp.

        With both fields present a checkpoint is self-describing: the architecture can be rebuilt
        from ``model_kwargs`` with no config file, and ``check_model_class`` can refuse a blob
        written by a different model before that rebuild is attempted.

        Args:
            checkpoint: The checkpoint dict, mutated in place.
        """
        super().on_save_checkpoint(checkpoint)  # stamps model_class; must run first
        checkpoint["model_kwargs"] = dict(self._model_kwargs)
