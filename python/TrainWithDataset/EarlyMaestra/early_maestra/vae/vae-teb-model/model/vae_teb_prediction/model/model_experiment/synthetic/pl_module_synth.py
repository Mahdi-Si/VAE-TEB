r"""Lightning wrapper for ``SeqVaeLagAttnV1`` on the **synthetic** TE benchmarks.

This is the model side of the multi-GPU (DDP) path for the single large
``G1_mix`` final-model run (see :mod:`train_ddp`). It is the synthetic-pipeline
analogue of :class:`model.vae_teb_prediction.model.trainer_lag_attn_v1.SeqVaeLagAttnPl`
(the real-HDF5 wrapper), which is left untouched. The two differ in exactly one
respect: this wrapper threads the synthetic calibration knobs
(``likelihood`` / ``sigma_obs`` / ``free_bits``) through to
:meth:`SeqVaeLagAttnV1.compute_loss` so a DDP run reproduces the
single-GPU :mod:`train_minimal` loss bit-for-bit (same ``orig_model.compute_loss``
call, same arguments).

Two patterns are copied verbatim from ``SeqVaeLagAttnPl`` because they are
load-bearing for *this* model under DDP:

* **No ``torch.compile``.** ``LightningModelBase.__init__`` compiles
  ``base_model`` unconditionally; that path is incompatible with the model's
  activation checkpointing (``attention_grad_checkpoint``) -- AOT autograd
  asserts in the first backward. We replicate the base ``__init__`` manually,
  skipping the compile line (see ``trainer_lag_attn_v1.py`` for the full
  diagnosis).
* **Spike-skip with cross-rank sync.** A non-finite or order-of-magnitude loss
  is turned into a no-op step. The skip decision is ``MAX``-reduced across ranks
  so every rank takes the same branch -- diverging branches mismatch the
  autograd graph and deadlock the backward all-reduce.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from model.vae_teb_prediction.model.model_experiment.synthetic.dataset import (
    build_u_stream,
)
from train.pl_model_base import LightningModelBase


class SyntheticSeqVaeLagAttnPl(LightningModelBase):
    r"""Lightning wrapper for :class:`SeqVaeLagAttnV1` on synthetic TE data.

    Reads the four model-facing fields from each batch -- ``fhr_st`` (43 ch),
    ``fhr_ph`` (44 ch) for the target stream and ``up_st`` (43 ch) + ``up_ph``
    (58 ch) concatenated into the 101-channel source stream via
    :func:`dataset.build_u_stream`. The synthetic caches always carry the full
    source, matching the ``use_up_st=True`` / ``c_u=101`` model build.

    Hyperparameters captured in ``self.hparams`` (so they survive checkpointing
    and are visible to callbacks): ``lr``, ``lr_milestones``, ``lr_gamma``,
    ``weight_decay``, plus the synthetic loss knobs ``kld_beta``,
    ``lambda_full``, ``lambda_base``, ``likelihood``, ``sigma_obs``,
    ``free_bits``, the ``warmup_epochs`` for LR warmup and the
    ``loss_spike_skip`` config.
    """

    #: Progress bar shows total + feature losses.
    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss")

    #: Spike-skip defaults (mirrors ``SeqVaeLagAttnPl._SPIKE_DEFAULTS``).
    _SPIKE_DEFAULTS: Dict[str, Any] = {
        "enabled": True,
        "multiplier": 5.0,
        "ema_momentum": 0.02,
        "warmup_batches": 100,
        "warn_on_skip": True,
    }

    def __init__(
        self,
        base_model: nn.Module,
        *,
        lr: float = 1e-3,
        lr_milestones: Optional[Iterable[int]] = None,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        kld_beta: float = 1e-3,
        lambda_full: float = 1.0,
        lambda_base: float = 0.5,
        likelihood: str = "mse",
        sigma_obs: "float | str" = 1.0,
        free_bits: float = 0.0,
        warmup_epochs: int = 0,
        loss_spike_skip: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None,
    ) -> None:
        """Initialize the wrapper while bypassing ``torch.compile``.

        Mirrors :meth:`SeqVaeLagAttnPl.__init__`: call the Lightning grandparent
        ``__init__`` directly so the ``torch.compile(base_model)`` line in
        :class:`LightningModelBase` never runs (incompatible with the model's
        activation checkpointing). All loss knobs are explicit kwargs so
        :meth:`save_hyperparameters` captures them.

        Args:
            base_model: The :class:`SeqVaeLagAttnV1` instance to wrap.
            lr: Learning rate (already LR-scaled by the caller for DDP).
            lr_milestones: Epoch milestones for the post-warmup ``MultiStepLR``.
            lr_gamma: Multiplicative decay applied at each milestone.
            weight_decay: AdamW weight decay.
            kld_beta: Weight on the KL term ($\\beta$).
            lambda_full: Weight on the residual feature loss $\\mathcal L_{feat}$.
            lambda_base: Weight on the baseline loss $\\mathcal L_{base}$.
            likelihood: ``'mse'`` or ``'gaussian_nll'`` (synthetic Sprint-5).
            sigma_obs: Positive scalar or the literal ``'learned'``.
            free_bits: Per-dim per-step KL floor (``0.0`` is a no-op).
            warmup_epochs: Linear LR warmup length; ``0`` disables it.
            loss_spike_skip: Optional overrides for the spike circuit breaker.
            module_name: Friendly name used in logs.
        """
        # Skip ``LightningModelBase.__init__`` (it compiles ``base_model``);
        # call the Lightning grandparent directly. See class docstring.
        pl.LightningModule.__init__(self)
        # Pass an explicit dict rather than relying on ``save_hyperparameters``'
        # frame inspection: skipping the parent ``__init__`` (and the keyword-only
        # signature) leaves the inspected frame empty, so the knobs would silently
        # vanish from ``self.hparams``. The explicit dict is deterministic and
        # keeps every loss / schedule knob available to ``compute_loss_and_metrics``.
        self.save_hyperparameters(
            {
                "lr": lr,
                "lr_milestones": list(lr_milestones) if lr_milestones else None,
                "lr_gamma": lr_gamma,
                "weight_decay": weight_decay,
                "kld_beta": kld_beta,
                "lambda_full": lambda_full,
                "lambda_base": lambda_base,
                "likelihood": likelihood,
                "sigma_obs": sigma_obs,
                "free_bits": free_bits,
                "warmup_epochs": warmup_epochs,
                "loss_spike_skip": loss_spike_skip,
                "module_name": module_name,
            }
        )
        self._orig_model = base_model
        self._wrapper_name = module_name or self.__class__.__name__
        self.model = base_model  # Eager mode -- no torch.compile.

        # Lazily-primed spike-skip state (see ``training_step``).
        self._spike_ema_loss: Optional[float] = None
        self._spike_batches_seen: int = 0
        self._spike_skips_total: int = 0

    # ------------------------------------------------------------------
    # Loss / metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward + synthetic loss, returning ``(total_loss, metrics)``.

        Calls ``orig_model.compute_loss`` with the exact knobs the single-GPU
        :mod:`train_minimal` loop uses, so DDP and single-GPU losses match.
        """
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        u_stream = build_u_stream(batch)

        forward_outputs = self.model(y_st, y_ph, u_stream)

        hp = self.hparams
        loss_dict = self.orig_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            weight=getattr(batch, "weight", None),
            beta=float(hp["kld_beta"]),
            lambda_full=float(hp["lambda_full"]),
            lambda_base=float(hp["lambda_base"]),
            likelihood=str(hp["likelihood"]),
            sigma_obs=hp["sigma_obs"],
            free_bits=float(hp["free_bits"]),
        )
        total_loss = loss_dict["total_loss"]
        pred_gap = loss_dict["base_loss"] - loss_dict["feat_loss"]

        metrics = {
            "total_loss": total_loss,
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            "pred_gap": pred_gap,
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "mean_logvar_full": loss_dict.get("mean_logvar_full"),
            "mean_logvar_base": loss_dict.get("mean_logvar_base"),
            "kld_beta": float(hp["kld_beta"]),
        }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Spike-skip circuit breaker (copied from SeqVaeLagAttnPl)
    # ------------------------------------------------------------------
    @property
    def _spike_cfg(self) -> Dict[str, Any]:
        """Resolve the spike-skip config from ``self.hparams`` with defaults."""
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
        """``MAX``-reduce ``is_spike`` so every DDP rank takes the same branch."""
        if not (dist.is_available() and dist.is_initialized()):
            return is_spike
        flag = torch.tensor([1.0 if is_spike else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        """Forward + loss gated by a cross-rank-synced spike check."""
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
                pass  # Still priming the EMA -- never flag a spike.
            elif ema_before is not None and ema_before > 0.0:
                if loss_value > float(cfg["multiplier"]) * ema_before:
                    is_spike = True

        is_spike = self._sync_skip_decision_across_ranks(is_spike, device=loss.device)

        # Update the EMA only on accepted batches (a spike must not raise the
        # bar for the next one). During warmup we always update (priming).
        if not is_spike:
            m = float(cfg["ema_momentum"])
            if ema_before is None:
                self._spike_ema_loss = loss_value
            else:
                self._spike_ema_loss = m * loss_value + (1.0 - m) * ema_before

        if is_spike:
            self._spike_skips_total += 1

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
                logger.warning(
                    "[spike-skip] batch_idx={} loss={:.4e} ema={} nonfinite={} "
                    "total_skips={}",
                    batch_idx, loss_value, ema_str, is_nonfinite,
                    self._spike_skips_total,
                )
            # No-op loss built from a parameter (not the possibly-NaN ``loss``)
            # so backward stays finite and DDP all-reduce keeps participating.
            anchor = next(p for p in self.parameters() if p.requires_grad)
            return anchor.sum() * 0.0

        return loss

    # ------------------------------------------------------------------
    # LR schedule: optional linear warmup -> MultiStepLR
    # ------------------------------------------------------------------
    def build_lr_scheduler(self, optimizer):
        r"""Linear warmup for ``warmup_epochs`` then the ``MultiStepLR`` decay.

        With ``warmup_epochs == 0`` this returns the plain ``MultiStepLR``
        (identical to :class:`LightningModelBase`'s default and to
        :func:`train_minimal.build_scheduler`). The DDP LR scaling
        (``lr * n_gpus``) is applied by the caller before construction; the
        warmup ramps from a small fraction up to that scaled ``lr``.
        """
        milestones = list(self.hparams.get("lr_milestones") or [])
        gamma = float(self.hparams.get("lr_gamma", 0.1))
        warmup_epochs = int(self.hparams.get("warmup_epochs", 0) or 0)

        from torch.optim.lr_scheduler import (
            LinearLR,
            MultiStepLR,
            SequentialLR,
        )

        if not milestones and warmup_epochs <= 0:
            return None
        if warmup_epochs <= 0:
            scheduler: Any = MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )
        else:
            warmup = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs,
            )
            # Shift milestones so the decay schedule counts epochs *after*
            # warmup, preserving the absolute-epoch milestone semantics of the
            # single-GPU loop.
            shifted = MultiStepLR(
                optimizer,
                milestones=[max(0, m - warmup_epochs) for m in milestones],
                gamma=gamma,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, shifted], milestones=[warmup_epochs]
            )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
