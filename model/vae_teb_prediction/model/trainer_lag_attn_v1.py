"""Lightning wrapper + Graph-model trainer for ``SeqVaeLagAttnV1``.

This file mirrors the layout of :mod:`model.vae_teb_prediction.training.trainer`
but targets the new lag-attentive v1 model defined in
:mod:`model.vae_teb_prediction.model.vae_teb_lag_attn_v1`. The two trainers
coexist and the original ``GraphModelVaeTebSmallTrainer`` is untouched.

Usage:
    python -m model.vae_teb_prediction.training.trainer_lag_attn_v1 \\
        --config model/vae_teb_prediction/model/config_lag_attn_v1.yaml
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import lightning as pl
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import SimpleProfiler
from loguru import logger

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.plotting_callback_lag_attn_v1 import (
    LagAttnV1PlotCallback,
)
from train.callbacks import (
    HyperparameterLoggingCallback,
    LossPlotCallback,
    MetricsLoggingCallback,
)
from train.graph_model_base import GraphModelBase
from train.graph_models_utils import load_checkpoint_strict
from train.pl_model_base import LightningModelBase


# =============================================================================
# Lightning wrapper
# =============================================================================


class SeqVaeLagAttnPl(LightningModelBase):
    """Lightning wrapper for :class:`SeqVaeLagAttnV1`.

    Reads the four model-facing fields directly from the batch:

    * ``batch.fhr_st`` — FHR scattering (target stream, 43 ch)
    * ``batch.fhr_ph`` — FHR phase harmonics (target stream, 44 ch)
    * ``batch.up_st`` — UP scattering (source stream, 43 ch; optional,
      skipped when the model was built with ``use_up_st=False``)
    * ``batch.up_ph`` — UP self-phase harmonics (source stream, 58 ch)

    Both ``up_st`` and ``up_ph`` are first-class HDF5 datasets with their
    own per-channel asinh/log stats. They are not derived from
    ``fhr_up_ph``. The two tensors are concatenated on the channel axis
    here when ``use_up_st=True`` to form the 101-channel source stream;
    otherwise ``up_ph`` alone is used (58 channels).

    Expected keys in ``self.hparams`` (merged via ``apply_config_hyperparameters``):

    * ``kld_beta`` — weight on the KL term (default 0.01).
    * ``lambda_full`` — weight on ``L_feat`` (default 1.0).
    * ``lambda_base`` — weight on ``L_base`` (default 0.5).
    """

    #: Progress bar shows total + feature losses.
    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss")

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        module_name: Optional[str] = None,
    ) -> None:
        """Initialize the wrapper while bypassing ``torch.compile``.

        ``LightningModelBase.__init__`` wraps ``base_model`` with
        ``torch.compile`` unconditionally. That path is incompatible with
        :class:`SeqVaeLagAttnV1` because ``LagCrossAttention.forward`` wraps
        its inner ``_attend`` call in
        ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` (see
        the ``attention_grad_checkpoint`` option in the config). When the
        outer module is compiled, AOT autograd's
        ``min_cut_rematerialization_partition`` asserts with
        ``Node <name> was invalid, but is output`` during the first backward
        pass, because forward-only nodes inside the activation-checkpointed
        region get marked as backward-graph outputs across the partition cut.
        Disabling ``torch._dynamo.config.optimize_ddp`` was not sufficient —
        the assertion reappears on ``constant_pad_nd_1`` from the lag memory
        builder's ``F.pad`` call even with Inductor as the top-level backend.

        We replicate the base ``__init__`` manually, skipping the
        ``torch.compile(base_model)`` step. The model runs eager, preserving
        activation checkpointing (which is necessary to fit the ~900 MB lag
        memory bank at ``B=64``). This mirrors the pattern already used by
        :class:`PlTemporalClassifier` for a similar incompatibility.

        Args:
            base_model: The :class:`SeqVaeLagAttnV1` instance to wrap.
            lr: Learning rate stored in ``self.hparams``.
            lr_milestones: Optional epoch milestones for the LR scheduler.
            weight_decay: AdamW weight decay applied across parameters.
            module_name: Friendly name used in logs and debug messages.
        """
        # NB: we intentionally skip ``LightningModelBase.__init__`` and call
        # the grandparent Lightning ``__init__`` directly so the
        # ``torch.compile`` line in the base class is never executed.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["base_model"])
        self._orig_model = base_model
        self._wrapper_name = module_name or self.__class__.__name__
        self.model = base_model  # Eager mode — no torch.compile

        # Loss-spike circuit breaker state. Populated lazily from ``self.hparams``
        # on the first training step because ``apply_config_hyperparameters``
        # runs AFTER ``__init__``. See ``_spike_cfg`` / ``_maybe_skip_step``.
        self._spike_ema_loss: Optional[float] = None
        self._spike_batches_seen: int = 0
        self._spike_skips_total: int = 0

    def _build_source_stream(self, batch: Any) -> torch.Tensor:
        """Build the ``u_stream`` tensor consumed by ``SeqVaeLagAttnV1.forward``.

        When ``use_up_st=True`` the stream is ``[up_st, up_ph]`` concatenated
        along the channel axis → ``(B, T, 101)``. When ``use_up_st=False`` it
        collapses to just ``up_ph`` → ``(B, T, 58)``. Both fields are read
        directly from the batch as independent HDF5 datasets.
        """
        up_ph = getattr(batch, "up_ph", None)
        if up_ph is None:
            raise RuntimeError(
                "batch has no `up_ph` field. Make sure 'up_ph' is listed in "
                "`dataset_kwargs.load_fields` of the config and that the HDF5 "
                "files were built with the new_pipeline (which writes up_ph as "
                "a first-class 58-channel dataset)."
            )
        use_up_st = bool(getattr(self.orig_model, "use_up_st", False))
        if not use_up_st:
            return up_ph
        up_st = getattr(batch, "up_st", None)
        if up_st is None:
            raise RuntimeError(
                "SeqVaeLagAttnV1 was constructed with use_up_st=True but the "
                "batch does not contain `up_st`. Either add 'up_st' to "
                "load_fields in the config, rebuild the HDF5 with up_st, or "
                "set use_up_st=False (and c_u=58) on the model."
            )
        return torch.cat([up_st, up_ph], dim=-1)

    # ------------------------------------------------------------------
    # Loss-spike circuit breaker (new_architecture.md §8 stability notes)
    # ------------------------------------------------------------------

    _SPIKE_DEFAULTS: Dict[str, Any] = {
        # Skip optimizer step when ``total_loss > multiplier * EMA``. 5× is
        # conservative enough to ignore normal variance while catching the
        # order-of-magnitude jumps that corrupt Adam's second moment.
        "enabled": True,
        "multiplier": 5.0,
        # EMA smoothing: ``ema ← m * loss + (1 - m) * ema``. Smaller = longer
        # memory of the healthy regime, harder for a slow drift to raise the
        # threshold into the spike zone.
        "ema_momentum": 0.02,
        # Number of priming batches: during this window spikes are never
        # flagged and the EMA is updated unconditionally so the running
        # average reaches a realistic scale before gating kicks in.
        "warmup_batches": 100,
        # Write a warning to loguru each time a batch is skipped.
        "warn_on_skip": True,
    }

    @property
    def _spike_cfg(self) -> Dict[str, Any]:
        """Resolve spike-skip config from ``self.hparams`` with defaults.

        Lives here (not in ``__init__``) because ``apply_config_hyperparameters``
        runs after construction, so the hparams namespace is not populated until
        training actually starts.
        """
        cfg = dict(self._SPIKE_DEFAULTS)
        override = self.hparams.get("loss_spike_skip")
        if isinstance(override, dict):
            for key in cfg:
                if key in override and override[key] is not None:
                    cfg[key] = override[key]
        return cfg

    def _sync_skip_decision_across_ranks(
        self, is_spike: bool, device: torch.device
    ) -> bool:
        """MAX-reduce ``is_spike`` so every DDP rank takes the same branch.

        Diverging branches produce mismatched autograd graphs across ranks,
        which deadlocks the all-reduce inside ``backward``.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return is_spike
        flag = torch.tensor([1.0 if is_spike else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        """Forward + loss, gated by a spike check.

        On a spike (non-finite loss, or loss > ``multiplier × EMA``) returns a
        finite zero-valued no-op loss instead of ``loss``. Lightning rejects
        ``None`` returns under DDP, so we need a real tensor connected to a
        parameter; ``anchor.sum() * 0.0`` keeps the all-reduce participating
        while leaving Adam's moments untouched.
        """
        loss, metrics = self.compute_loss_and_metrics(batch, batch_idx, stage="train")

        cfg = self._spike_cfg
        loss_value = float(loss.detach().item())
        is_nonfinite = not math.isfinite(loss_value)

        ema_before = self._spike_ema_loss
        seen_before = self._spike_batches_seen
        self._spike_batches_seen += 1

        is_spike = False
        if cfg["enabled"]:
            if is_nonfinite:
                is_spike = True
            elif seen_before < int(cfg["warmup_batches"]):
                # Still priming the EMA — never flag a spike.
                pass
            elif ema_before is not None and ema_before > 0.0:
                threshold = float(cfg["multiplier"]) * ema_before
                if loss_value > threshold:
                    is_spike = True

        # Ensure all DDP ranks take the same branch (or backward deadlocks).
        is_spike = self._sync_skip_decision_across_ranks(is_spike, device=loss.device)

        # Update the EMA only on accepted batches so a spike cannot raise the
        # bar for the next one. During warmup we always update (priming).
        if not is_spike:
            m = float(cfg["ema_momentum"])
            if ema_before is None:
                self._spike_ema_loss = loss_value
            else:
                self._spike_ema_loss = m * loss_value + (1.0 - m) * ema_before

        if is_spike:
            self._spike_skips_total += 1

        # Surface the circuit-breaker state to the unified metrics logger.
        ema_for_log = (
            self._spike_ema_loss if self._spike_ema_loss is not None else loss_value
        )
        metrics["spike_ema_loss"] = self._as_tensor(ema_for_log)
        metrics["spike_skipped"] = self._as_tensor(1.0 if is_spike else 0.0)
        metrics["spike_skips_total"] = self._as_tensor(self._spike_skips_total)
        self._log_metrics(metrics, stage="train", on_step=True)

        if is_spike:
            if cfg["warn_on_skip"]:
                ema_str = "n/a" if ema_before is None else f"{ema_before:.4e}"
                threshold_str = (
                    "n/a"
                    if ema_before is None or ema_before <= 0.0
                    else f"{float(cfg['multiplier']) * ema_before:.4e}"
                )
                logger.warning(
                    "[spike-skip] batch_idx={} loss={:.4e} ema={} "
                    "threshold={} nonfinite={} total_skips={}",
                    batch_idx,
                    loss_value,
                    ema_str,
                    threshold_str,
                    is_nonfinite,
                    self._spike_skips_total,
                )
            # No-op loss built from a parameter (not the possibly-NaN ``loss``)
            # so backward stays finite and DDP all-reduce participates.
            anchor = next(p for p in self.parameters() if p.requires_grad)
            return anchor.sum() * 0.0

        return loss

    def _resolve_beta(self, epoch: int) -> float:
        r"""Resolve the KL weight :math:`\beta` for ``epoch`` (A2).

        Reads the structured ``beta_schedule`` hparam. Supported kinds:

        * ``constant`` — returns ``beta_schedule.value`` when present, else
          the legacy ``kld_beta`` hparam (default ``0.01``).
        * ``linear_warmup`` — linearly ramps from ``start`` to ``end`` over the
          first ``warmup_epochs`` epochs, then holds at ``end``:

          .. math::
              \beta(e) = start + (end - start)\,
                         \min\!\left(1, \tfrac{e}{warmup\_epochs}\right).

        A weak early :math:`\beta` lets the residual decoder *learn to use*
        :math:`z` before the bottleneck tightens for a calibrated TE reading.
        When no ``beta_schedule`` dict is configured the legacy constant
        ``kld_beta`` is used, preserving pre-A2 behaviour.

        Args:
            epoch: Current training epoch (``self.current_epoch``).

        Returns:
            The scalar :math:`\beta` weighting ``kld_loss`` this epoch.
        """
        sched = self.hparams.get("beta_schedule")
        if not isinstance(sched, dict):
            return float(self.hparams.get("kld_beta", 0.01))
        kind = str(sched.get("kind", "constant"))
        if kind == "constant":
            value = sched.get("value")
            if value is not None:
                return float(value)
            return float(self.hparams.get("kld_beta", 0.01))
        if kind == "linear_warmup":
            start = float(sched.get("start", 1.0e-4))
            end = float(sched.get("end", 0.1))
            warmup_epochs = int(sched.get("warmup_epochs", 50))
            if warmup_epochs <= 0:
                return end
            frac = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
            return start + (end - start) * frac
        raise ValueError(
            f"Unknown beta_schedule.kind={kind!r}; expected "
            "'constant' or 'linear_warmup'."
        )

    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the forward pass and build the unified metrics dict."""
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        u_stream = self._build_source_stream(batch)

        forward_outputs = self.model(y_st, y_ph, u_stream)

        # A2: epoch-dependent β from the structured schedule (falls back to the
        # constant ``kld_beta`` when no schedule is configured).
        beta = self._resolve_beta(self.current_epoch)
        lambda_full = float(self.hparams.get("lambda_full", 1.0))
        lambda_base = float(self.hparams.get("lambda_base", 0.5))

        # A1: calibrated ELBO. ``likelihood='gaussian_nll'`` puts feat / base
        # losses in nats so they are directly comparable to ``kld_loss``;
        # ``sigma_obs`` is the fixed observation-noise scalar (or 'learned' to
        # activate the decoder logvar heads). A2: ``free_bits`` floors the
        # per-dim KL. A3: ``detach_baseline_in_full`` stop-gradients the
        # baseline inside the full-prediction term.
        likelihood = str(self.hparams.get("likelihood", "mse"))
        sigma_obs = self.hparams.get("sigma_obs", 1.0)
        if not isinstance(sigma_obs, str):
            sigma_obs = float(sigma_obs)
        free_bits = float(self.hparams.get("free_bits", 0.0))
        detach_baseline_in_full = bool(
            self.hparams.get("detach_baseline_in_full", False)
        )
        # B4: lag-embedding smoothness weight.
        lambda_lag = float(self.hparams.get("lambda_lag", 0.0))

        # Pass the dataset per-step validity mask so gaps (weight ≈ 0) do
        # not pollute feat / base / KL losses. Required for a trustworthy
        # TE curve. ``weight`` is always present in the HDF5 schema.
        weight = getattr(batch, "weight", None)

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
        total_loss = loss_dict["total_loss"]

        # --- Residual-branch diagnostics ------------------------------------
        # These three metrics answer the question "is the residual decoder
        # doing anything useful, and in which direction?". All are masked to
        # match the anchor / future windows used by feat_loss and kld_loss
        # so magnitudes are directly comparable across the loss plot.
        #
        # * ``delta_mu_rms``          — how much the residual moves the
        #   forecast at each valid anchor. ``→ 0`` means "UP contributes
        #   nothing" (healthy if KL is also ≈ 0); growing over epochs while
        #   ``pred_gap < 0`` is the signature of double-counting.
        # * ``mu_post_prior_gap_rms`` — Euclidean distance between posterior
        #   and prior means in latent space, one of the two terms inside
        #   ``KL(q‖p)``. Complements ``kld_loss`` by isolating the mean
        #   component from the variance component.
        # * ``pred_gap``              — prediction-space TE surrogate
        #   ``base_loss − feat_loss``. Positive means the residual helps;
        #   negative means it hurts. Compare against ``kld_loss`` — they
        #   should track each other up to a Jensen gap in a healthy run.
        diag = self._compute_residual_diagnostics(
            forward_outputs=forward_outputs,
            weight=weight,
        )
        pred_gap = loss_dict["base_loss"] - loss_dict["feat_loss"]

        metrics = {
            "total_loss": total_loss,
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            "kld_beta": beta,
            "lambda_full": lambda_full,
            "lambda_base": lambda_base,
            # Tanh-bound saturation diagnostics. Expect ~0 when the bounds
            # are dormant; sustained > 0.05 means the prior / posterior
            # wants to drift beyond the configured scale — bump mu_scale
            # or delta_mu_scale in config_lag_attn_v1.yaml.
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            # Residual-branch magnitudes + prediction-space TE surrogate.
            "delta_mu_rms": diag["delta_mu_rms"],
            "mu_post_prior_gap_rms": diag["mu_post_prior_gap_rms"],
            "pred_gap": pred_gap,
            # B4 lag-embedding smoothness penalty (raw, pre-weighting).
            "lag_smoothness": loss_dict["lag_smoothness"],
        }
        return total_loss, metrics

    def _compute_residual_diagnostics(
        self,
        *,
        forward_outputs: Dict[str, torch.Tensor],
        weight: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return masked RMS of ``delta_mu_src`` and the latent mean gap.

        Both are computed under the same masking rules used by the loss so
        the numbers are directly comparable to ``feat_loss`` / ``kld_loss``
        on a run-to-run basis:

        * ``delta_mu_rms`` uses the full (warmup × anchor_weight × future_weight)
          mask that ``L_feat`` uses.
        * ``mu_post_prior_gap_rms`` uses the (warmup × anchor_weight) mask
          that ``L_KL`` uses.
        """
        warmup = int(self.orig_model._warmup_steps(
            forward_outputs["mu_full"].size(1)
        ))

        delta_mu = forward_outputs["delta_mu_src"]          # (B, T, Hd, C)
        B, T, Hd, _ = delta_mu.shape
        T_valid = T - Hd
        device = delta_mu.device
        dtype = delta_mu.dtype

        # --- Feature-window mask (same as feat_loss) -----------------------
        warmup_t = torch.zeros(T_valid, dtype=dtype, device=device)
        if warmup < T_valid:
            warmup_t[warmup:] = 1.0                          # (T_valid,)

        if weight is not None:
            w = weight.to(device=device, dtype=dtype)        # (B, T)
            anchor_w = w[:, :T_valid]                        # (B, T_valid)
            target_w = w[:, 1:].unfold(
                dimension=1, size=Hd, step=1
            )                                                # (B, T_valid, Hd)
            feat_mask = (
                warmup_t[None, :, None]
                * anchor_w[:, :, None]
                * target_w
            )                                                # (B, T_valid, Hd)
        else:
            feat_mask = warmup_t[None, :, None].expand(
                B, T_valid, Hd
            )

        delta_valid = delta_mu[:, :T_valid, :, :]            # (B, T_valid, Hd, C)
        # Per-(b, t, tau) squared norm summed over channels, then masked.
        delta_sq = (delta_valid ** 2).sum(dim=-1)            # (B, T_valid, Hd)
        denom = feat_mask.sum().clamp_min(1.0)
        delta_mu_rms = torch.sqrt(
            (delta_sq * feat_mask).sum() / denom
        )

        # --- Latent-window mask (same as kld_loss) -------------------------
        mu_prior = forward_outputs["mu_prior"]               # (B, T, d_z)
        mu_post = forward_outputs["mu_post"]                 # (B, T, d_z)
        T_full = mu_prior.size(1)
        time_mask = torch.ones(T_full, dtype=dtype, device=device)
        warm_full = int(self.orig_model._warmup_steps(T_full))
        if warm_full > 0:
            time_mask[:warm_full] = 0.0
        lat_mask = time_mask.unsqueeze(0).expand(B, T_full)   # (B, T_full)
        if weight is not None:
            lat_mask = lat_mask * weight.to(
                device=device, dtype=dtype
            )

        gap_sq = ((mu_post - mu_prior) ** 2).sum(dim=-1)     # (B, T_full)
        lat_denom = lat_mask.sum().clamp_min(1.0)
        mu_post_prior_gap_rms = torch.sqrt(
            (gap_sq * lat_mask).sum() / lat_denom
        )

        return {
            "delta_mu_rms": delta_mu_rms,
            "mu_post_prior_gap_rms": mu_post_prior_gap_rms,
        }


# =============================================================================
# Graph-model trainer
# =============================================================================


class GraphModelVaeTebLagAttnV1Trainer(GraphModelBase):
    """Experiment driver for the lag-attentive v1 model.

    Mirrors ``GraphModelVaeTebSmallTrainer`` but builds ``SeqVaeLagAttnV1``
    from config and uses the new Lightning wrapper.
    """

    def __init__(self, config_file_path: str | None = None) -> None:
        super().__init__(config_file_path)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Translate the ``VAE_model`` config section into constructor kwargs."""
        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        kwargs: Dict[str, Any] = {
            "sequence_length": int(vae_cfg.get("sequence_length", 300)),
            "d_model": int(vae_cfg.get("d_model", 128)),
            "d_z": int(vae_cfg.get("d_z", 24)),
            "horizon": int(vae_cfg.get("horizon", 30)),
            "warmup_period": int(vae_cfg.get("warmup_period", 30)),
            "c_y": int(vae_cfg.get("c_y", 87)),
            "c_u": int(vae_cfg.get("c_u", 101)),
            "use_up_st": bool(vae_cfg.get("use_up_st", True)),
            "max_lag": int(vae_cfg.get("max_lag", 90)),
            "num_heads": int(vae_cfg.get("num_heads", 4)),
            "d_head": int(vae_cfg.get("d_head", 32)),
            "lstm_layers": int(vae_cfg.get("lstm_layers", 2)),
            "dropout": float(vae_cfg.get("dropout", 0.1)),
            "decoder_hidden": int(vae_cfg.get("decoder_hidden", 128)),
            "use_entmax": bool(vae_cfg.get("use_entmax", False)),
            "attention_grad_checkpoint": bool(
                vae_cfg.get("attention_grad_checkpoint", True)
            ),
            # B4 — lag-bias init: 'normal' (default) or 'alibi_decay'.
            "lag_bias_init": str(vae_cfg.get("lag_bias_init", "normal")),
            # C7 — head-structured latent (per-head additive KL).
            "head_structured_latent": bool(
                vae_cfg.get("head_structured_latent", False)
            ),
        }
        # D8 — shared horizon decoder core (depth / kernel / FiLM).
        horizon_cfg = vae_cfg.get("horizon_refine", {}) or {}
        kwargs["horizon_depth"] = int(horizon_cfg.get("depth", 2))
        kwargs["horizon_kernel"] = int(horizon_cfg.get("kernel", 3))
        kwargs["horizon_film"] = bool(horizon_cfg.get("film", False))
        # E11 — extra encoder dilations for a longer receptive field.
        encoder_cfg = vae_cfg.get("encoder", {}) or {}
        kwargs["encoder_extra_dilations"] = tuple(
            int(x) for x in (encoder_cfg.get("extra_dilations", []) or [])
        )
        logvar_clamp = vae_cfg.get("logvar_clamp")
        if isinstance(logvar_clamp, (list, tuple)) and len(logvar_clamp) == 2:
            kwargs["logvar_clamp"] = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        if "mu_scale" in vae_cfg:
            kwargs["mu_scale"] = float(vae_cfg["mu_scale"])
        if "delta_mu_scale" in vae_cfg:
            kwargs["delta_mu_scale"] = float(vae_cfg["delta_mu_scale"])
        if "latent_stats_momentum" in vae_cfg:
            kwargs["latent_stats_momentum"] = float(vae_cfg["latent_stats_momentum"])
        return kwargs

    def create_model(self) -> None:
        """Instantiate ``SeqVaeLagAttnV1`` and wrap it in ``SeqVaeLagAttnPl``."""
        model_kwargs = self._build_model_kwargs()
        logger.info(
            "Building SeqVaeLagAttnV1 with kwargs: "
            + ", ".join(f"{k}={v}" for k, v in model_kwargs.items())
        )
        self.pytorch_model = SeqVaeLagAttnV1(**model_kwargs)

        self.checkpoint = self.config.get("model_config", {}).get("core_model_checkpoint")
        if self.checkpoint is not None:
            load_checkpoint_strict(
                model=self.pytorch_model,
                checkpoint=self.checkpoint,
            )
            logger.info(f"Model loaded from checkpoint: {self.checkpoint}")

        vae_cfg = self.config.get("model_config", {}).get("VAE_model", {}) or {}
        trainer_hparams = {
            "lr": self.lr,
            "lr_milestones": self.lr_milestones,
            "kld_beta": vae_cfg.get("kld_beta", 0.01),
            "lambda_full": vae_cfg.get("lambda_full", 1.0),
            "lambda_base": vae_cfg.get("lambda_base", 0.5),
            # A1/A2/A3 calibration core. ``beta_schedule`` (dict) is resolved
            # per-epoch by ``SeqVaeLagAttnPl._resolve_beta``; ``likelihood`` /
            # ``sigma_obs`` select the reconstruction NLL; ``free_bits`` floors
            # the per-dim KL; ``detach_baseline_in_full`` stop-gradients the
            # baseline inside the full-prediction term.
            "beta_schedule": vae_cfg.get("beta_schedule"),
            "likelihood": vae_cfg.get("likelihood", "mse"),
            "sigma_obs": vae_cfg.get("sigma_obs", 1.0),
            "free_bits": vae_cfg.get("free_bits", 0.0),
            "detach_baseline_in_full": vae_cfg.get(
                "detach_baseline_in_full", False
            ),
            # B4 — lag-embedding smoothness weight (lambda_lag).
            "lambda_lag": vae_cfg.get("lag_smoothness_lambda", 0.0),
            # Loss-spike circuit breaker. Consumed in SeqVaeLagAttnPl._spike_cfg.
            # Missing keys fall back to the class-level ``_SPIKE_DEFAULTS``.
            "loss_spike_skip": vae_cfg.get("loss_spike_skip", {}) or {},
        }
        self.pl_model = SeqVaeLagAttnPl(
            self.pytorch_model,
            lr=self.lr,
            lr_milestones=self.lr_milestones,
        )
        self.apply_config_hyperparameters(trainer_hparams, self.pl_model)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_model(self, train_dataloader, validation_dataloader):
        """Build callbacks + Lightning Trainer and run ``trainer.fit``."""
        callbacks_cfg = self.config.get("advanced_config", {}).get("callbacks", {})
        self.metrics_callback = MetricsLoggingCallback()
        self.loss_plot_callback = LossPlotCallback(
            output_dir=self.train_results_dir,
            plot_frequency=self.config["general_config"].get("plot_frequency", 1),
            mlflow_logger=self.mlflow_logger,
        )
        self.hyperparam_callback = HyperparameterLoggingCallback(
            output_dir=self.train_results_dir,
            plot_frequency=10,
        )
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_checkpoint_dir,
            monitor="val/total_loss",
            filename="lag-attn-v1-epoch={epoch:02d}",
            save_top_k=callbacks_cfg.get("model_checkpoint", {}).get("save_top_k", 3),
            mode=callbacks_cfg.get("model_checkpoint", {}).get("mode", "min"),
        )
        callback_list = [
            self.metrics_callback,
            self.loss_plot_callback,
            self.hyperparam_callback,
        ]

        # Diagnostic plotting — enabled by default, configurable via the
        # `advanced_config.callbacks.lag_attn_plotting` section.
        plot_cfg = callbacks_cfg.get("lag_attn_plotting", {}) or {}
        if plot_cfg.get("enabled", True):
            self.lag_attn_plot_callback = LagAttnV1PlotCallback(
                output_dir=self.train_results_dir,
                plot_frequency=int(plot_cfg.get("plot_frequency", 5)),
                num_examples=int(plot_cfg.get("num_examples", 2)),
                file_format=str(plot_cfg.get("file_format", "pdf")),
                mlflow_logger=self.mlflow_logger,
                forecast_channels=tuple(
                    int(c) for c in plot_cfg.get("forecast_channels", [0, 43, 80])
                ),
                forecast_anchor_frac=float(plot_cfg.get("forecast_anchor_frac", 0.6)),
            )
            callback_list.append(self.lag_attn_plot_callback)

        callback_list.append(self.checkpoint_callback)

        trainer_cfg = self.config.get("advanced_config", {}).get("trainer", {})
        precision = trainer_cfg.get("precision", "32-true")
        gradient_clip_val = trainer_cfg.get("gradient_clip_val")
        gradient_clip_algorithm = trainer_cfg.get("gradient_clip_algorithm", "norm")
        logger_reference = self.lightning_loggers if self.lightning_loggers else True
        trainer_kwargs: Dict[str, Any] = {
            "max_epochs": self.epochs_num,
            "callbacks": callback_list,
            "default_root_dir": self.train_results_dir,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "precision": precision,
            "deterministic": trainer_cfg.get("deterministic", False),
            "benchmark": trainer_cfg.get("benchmark", True),
            "gradient_clip_val": gradient_clip_val,
            "gradient_clip_algorithm": gradient_clip_algorithm,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "num_sanity_val_steps": 0,
            "use_distributed_sampler": True,
            "sync_batchnorm": len(self.cuda_devices) > 1,
            "enable_progress_bar": True,
            "profiler": SimpleProfiler(dirpath=self.train_results_dir),
            "logger": logger_reference,
        }
        if torch.cuda.is_available():
            # D10 — DDP strategy depends on whether the decoder logvar heads
            # receive gradients. ``SeqVaeLagAttnV1`` keeps two auxiliary logvar
            # heads (``BaselineFutureDecoder.logvar_head`` /
            # ``ResidualFutureDecoder.logvar_head``) wired into forward. They
            # are consumed by ``compute_loss`` ONLY when
            # ``likelihood='gaussian_nll'`` AND ``sigma_obs='learned'``
            # (heteroscedastic NLL). In every other config (MSE, or
            # gaussian_nll with a fixed scalar ``sigma_obs``) they receive
            # ``None`` gradients, so DDP's first-iteration bucket rebuild trips
            # with "parameters that were not used in producing the loss" unless
            # we set ``find_unused_parameters``. Once ``sigma_obs='learned'``
            # activates the heads, plain ``'ddp'`` is correct and drops the
            # extra post-backward scan.
            vae_cfg = (
                self.config.get("model_config", {}).get("VAE_model", {}) or {}
            )
            sigma_obs = vae_cfg.get("sigma_obs", 1.0)
            likelihood = str(vae_cfg.get("likelihood", "mse"))
            logvar_heads_consumed = (
                likelihood == "gaussian_nll"
                and isinstance(sigma_obs, str)
                and sigma_obs == "learned"
            )
            if len(self.cuda_devices) > 1:
                ddp_strategy = (
                    "ddp"
                    if logvar_heads_consumed
                    else "ddp_find_unused_parameters_true"
                )
            else:
                ddp_strategy = "auto"
            trainer_kwargs.update(
                {
                    "accelerator": "gpu",
                    "devices": self.cuda_devices,
                    "strategy": ddp_strategy,
                }
            )
        else:
            trainer_kwargs.update({"accelerator": "cpu", "devices": 1})
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self.pl_model, train_dataloader, validation_dataloader)
        return trainer


# =============================================================================
# Entry point
# =============================================================================


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "config_lag_attn_v1.yaml",
)


def main(config_path: str = _DEFAULT_CONFIG) -> None:
    """Build data loaders, model, trainer and run ``fit``."""
    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config.get("dataset_config")
    dataloader_config = dataset_config.get("dataloader_config")
    dataset_kwargs = dataloader_config.get("dataset_kwargs")
    normalize_fields = dataloader_config.get("normalize_fields")
    stat_path = dataset_config.get("stat_path")
    if stat_path is None:
        raise ValueError("stat_path must be provided")
    logger.info(f"normalized fields: {normalize_fields}")
    logger.info(f"load fields:       {dataset_kwargs.get('load_fields')}")

    # NB: ``pin_memory`` is intentionally *not* passed here. It now lives in
    # ``dataloader_config.dataset_kwargs.pin_memory`` (set to ``false`` in the
    # config) so it flows through ``**dataset_kwargs`` to
    # ``HDF5Dataset.__init__``. Passing it here as well would trigger a
    # ``TypeError: multiple values for keyword argument 'pin_memory'``.
    # Historical note: the previous ``pin_memory=True`` argument was
    # misleading — ``create_optimized_dataloader`` never forwarded it to
    # ``DataLoader(pin_memory=...)``; it only fed the dataset's in-worker
    # ``tensor.pin_memory()`` call, which forced each worker to spawn a
    # CUDA context on ``cuda:0`` and caused the OOM at the epoch-0
    # train→val transition.
    train_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_train_datasets", []),
        batch_size=config["general_config"]["batch_size"]["train"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=True,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    validation_dataloader = create_optimized_dataloader(
        hdf5_files=dataset_config.get("vae_test_datasets", []),
        batch_size=config["general_config"]["batch_size"]["test"],
        num_workers=dataloader_config.get("num_workers", 4),
        shuffle=False,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2),
        rank=0,
        world_size=1,
        **dataset_kwargs,
    )

    graph_model = GraphModelVaeTebLagAttnV1Trainer(config_file_path=config_path)
    graph_model.setup_config()
    graph_model.create_model()
    graph_model.train_model(train_dataloader, validation_dataloader)
    end_time = time.time()
    logger.info(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to the YAML config (default: config_lag_attn_v1.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
