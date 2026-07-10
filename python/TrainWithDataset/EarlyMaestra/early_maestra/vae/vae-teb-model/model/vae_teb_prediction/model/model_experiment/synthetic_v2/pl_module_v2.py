r"""Lightning wrapper + training driver for ``SeqVaeLagAttn`` on ``synthetic_v2``.

Sprint 5 (S5-T01/T02/T03/T04). This module trains the **unchanged** VAE-TEB
lag-attention model on the cached ``synthetic_v2`` splits and is the analogue of the
v1 ``synthetic/pl_module_synth.py`` + ``synthetic/train_ddp.py`` pair, folded into one
standalone v2 file (the v2 layout has no separate ``train_v2.py``).

It provides:

* :class:`SyntheticSeqVaeLagAttnV2Pl` -- a :class:`train.pl_model_base.LightningModelBase`
  subclass that reads the four model-facing fields from each batch (``fhr_st`` (43 ch),
  ``fhr_ph`` (44 ch), and ``up_st`` (43) + ``up_ph`` (58) concatenated to the
  101-channel source stream via :func:`dataset_v2.build_u_stream`), runs the model
  ``forward`` + ``compute_loss``, threads the calibration knobs
  (``likelihood`` / ``sigma_obs`` / ``free_bits``), logs ``kld_nats`` (the
  $\bar K$ TE-surrogate scale), and carries a cross-rank-synced loss-spike circuit
  breaker + optional linear LR warmup.
* :func:`build_model`, :func:`save_checkpoint_v2`, :func:`train_v2`, and
  :func:`beta_select` -- the driver stack: build the model on CPU, wrap it, build the
  :class:`datamodule_v2.SyntheticTEDataModuleV2`, fit a single-GPU (or DDP when
  ``devices > 1``) :class:`lightning.pytorch.Trainer`, run a post-fit latent-stats pass,
  and export ``final.ckpt`` / ``best.ckpt`` in the v1-compatible format that
  :func:`train.graph_models_utils.load_checkpoint_strict` reads.

Two load-bearing patterns are copied from the v1 wrapper: **no** ``torch.compile`` (the
base ``__init__`` compiles unconditionally, which breaks activation checkpointing and is
fragile on Windows), and a spike-skip step whose decision is ``MAX``-reduced across ranks
so every DDP rank takes the same autograd branch.

See ``SYNTHETIC_V2_SPEC_AND_SPRINTS.md`` Sprint 5 and
``SYNTHETIC_V2_RAW_TE_PIPELINE_EXPLAINED.md`` §13.
"""
from __future__ import annotations

import inspect
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# The model imports ``utils.custom_logger`` from the repo root, but the repo has an
# irregular package layout: ``model/vae_teb_prediction/utils`` is a regular package
# WITHOUT ``custom_logger``, and pytest / a bare script run can insert
# ``model/vae_teb_prediction`` onto ``sys.path`` ahead of the repo root, shadowing the
# real ``utils`` the model needs. Force the repo root (six levels up:
# synthetic_v2 -> model_experiment -> model -> vae_teb_prediction -> model -> root) to
# the front before importing the model / train helpers.
_REPO_ROOT = str(Path(__file__).resolve().parents[5])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import lightning as pl  # noqa: E402  (import after the sys.path bootstrap)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from loguru import logger  # noqa: E402

# Canonical model-class alias -- comment-toggle to switch v1 <-> v2 in one line.
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1 as SeqVaeLagAttn  # noqa: E402
# from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import SeqVaeLagAttnV2 as SeqVaeLagAttn  # noqa: E402
from train.graph_models_utils import load_checkpoint_strict  # noqa: E402
from train.pl_model_base import LightningModelBase  # noqa: E402

from .dataset_v2 import build_u_stream  # noqa: E402
from .datamodule_v2 import SyntheticTEDataModuleV2  # noqa: E402

_MODULE_DIR = Path(__file__).resolve().parent

# Trainer defaults, mirroring ``synthetic/train_ddp.py::_DDP_DEFAULTS`` but with a
# single-GPU (``devices=1``) local default and ``warmup_epochs=0`` (warmup is applied
# only under DDP LR scaling). Overridden by the config ``ddp`` block.
_DDP_DEFAULTS: Dict[str, Any] = {
    "devices": 1,
    "strategy": "ddp_find_unused_parameters_true",
    "precision": "32-true",
    "sync_batchnorm": True,
    "lr_scaling": True,
    "warmup_epochs": 0,
    "num_sanity_val_steps": 0,
    "gradient_clip_val": 0.5,
    "gradient_clip_algorithm": "norm",
}

#: Early-stopping defaults, mirroring the ``ddp.early_stopping`` YAML block. Off by
#: default: the block is *honoured* (read and acted on), but enabling it is opt-in.
_EARLY_STOPPING_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "monitor": "val/total_loss",
    "mode": "min",
    "patience": 15,
    "min_delta": 0.0,
}


def _resolve_epoch_metric(name: str) -> str:
    r"""Resolve a bare metric name to its Lightning epoch-logged key.

    Metrics logged with ``on_step=on_epoch=True`` (as all metrics are here, via
    ``train.pl_model_base.LightningModelBase._log_metrics``) are forked by Lightning
    into ``<name>_step`` and ``<name>_epoch`` columns; the *bare* ``<name>`` never
    lands in ``trainer.callback_metrics`` or ``metrics.csv``. Epoch-level monitoring
    (early stopping, best-checkpoint selection, post-fit stamping) wants the
    ``_epoch`` variant, so this appends ``_epoch`` unless ``name`` is already
    suffixed. Idempotent; mirrors the resolver in
    :func:`visualize_v2._read_metric_series`.

    Args:
        name: A metric name, e.g. ``"val/total_loss"``.

    Returns:
        ``name`` unchanged if it ends in ``_epoch``/``_step``, else ``f"{name}_epoch"``.
    """
    return name if name.endswith(("_epoch", "_step")) else f"{name}_epoch"


# ============================================================================
# Lightning module (S5-T01)
# ============================================================================
class SyntheticSeqVaeLagAttnV2Pl(LightningModelBase):
    r"""Lightning wrapper for :class:`SeqVaeLagAttn` on ``synthetic_v2`` TE data.

    Reads the four model-facing fields from each batch and calls
    ``orig_model.compute_loss`` with the synthetic calibration knobs, so a run
    reproduces the loss the R0 / evaluation math expects. The objective is
    $\mathcal L = \lambda_{\mathrm{full}}\,\mathcal L_{\mathrm{feat}}
    + \lambda_{\mathrm{base}}\,\mathcal L_{\mathrm{base}} + \beta\,\mathcal L_{\mathrm{KL}}$
    (model :meth:`compute_loss`), and the logged ``kld_nats``
    $= \mathrm{KL}\cdot d_z$ is the nat-scale surrogate $\bar K$ that Sprint 6
    calibrates against $\mathrm{TE}_{\mathrm{inj}}$ / $\mathrm{TE}_{\mathrm{scat}}$.
    """

    #: Progress bar shows total + feature losses.
    prog_bar_metrics: Tuple[str, ...] = ("total_loss", "feat_loss")

    #: Spike-skip defaults (mirrors the v1 wrapper).
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
        detach_baseline_in_full: bool = False,
        lambda_lag: float = 0.0,
        warmup_epochs: int = 0,
        beta_schedule: Optional[Dict[str, Any]] = None,
        loss_spike_skip: Optional[Dict[str, Any]] = None,
        curriculum: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None,
    ) -> None:
        r"""Initialize the wrapper while bypassing ``torch.compile``.

        Calls the Lightning grandparent ``__init__`` directly so the
        ``torch.compile(base_model)`` line in :class:`LightningModelBase` never
        runs, then saves every loss / schedule knob into ``self.hparams`` via an
        explicit dict (frame inspection is empty when the parent ``__init__`` is
        skipped).

        Args:
            base_model: The :class:`SeqVaeLagAttn` instance to wrap.
            lr: Learning rate (already LR-scaled by the caller for DDP).
            lr_milestones: Epoch milestones for the post-warmup ``MultiStepLR``.
            lr_gamma: Multiplicative decay applied at each milestone.
            weight_decay: AdamW weight decay.
            kld_beta: Weight on the KL term ($\beta$).
            lambda_full: Weight on the residual feature loss $\mathcal L_{\mathrm{feat}}$.
            lambda_base: Weight on the baseline loss $\mathcal L_{\mathrm{base}}$.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            sigma_obs: Positive scalar or the literal ``'learned'``.
            free_bits: Per-dim per-step KL floor (``0.0`` is a no-op).
            detach_baseline_in_full: Baseline/residual gradient separation for
                $\mathcal L_{\mathrm{feat}}$ (forward values unchanged).
            warmup_epochs: Linear LR warmup length; ``0`` disables it.
            loss_spike_skip: Optional overrides for the spike circuit breaker.
            module_name: Friendly name used in logs.
        """
        # Skip ``LightningModelBase.__init__`` (it compiles ``base_model``); call
        # the Lightning grandparent directly. See the module docstring.
        pl.LightningModule.__init__(self)
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
                "detach_baseline_in_full": detach_baseline_in_full,
                "lambda_lag": lambda_lag,
                "warmup_epochs": warmup_epochs,
                "beta_schedule": beta_schedule,
                "loss_spike_skip": loss_spike_skip,
                "curriculum": curriculum,
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

        # CPU generator for the source-permutation control's batch derangement (S3-T01).
        # Seeded per rank so the shuffles differ across ranks (their data differs); the
        # *schedule* is rank-invariant, which is what DDP actually requires. Present
        # unconditionally -- ``perm_kl_from_forward(generator=...)`` dereferences it, and a
        # v1 model simply never reaches that call.
        self._perm_generator = torch.Generator()
        self._perm_generator.manual_seed(1234 + int(getattr(self, "global_rank", 0) or 0))
        self._perm_lambda_warned: bool = False

    # ------------------------------------------------------------------
    # Source-permutation control (S3-T01) -- readout only
    # ------------------------------------------------------------------
    def _sync_perm_decision(self, do_perm: bool, device: torch.device) -> bool:
        r"""MIN-reduce ``do_perm`` so no rank runs the control alone.

        Ported from ``trainer_lag_attn_v3._sync_perm_decision``. ``batch_idx`` is already
        rank-invariant, but a rank whose local batch is degenerate ($B < 2$) cannot be
        deranged. Reducing with ``MIN`` means one such rank vetoes the control everywhere,
        so every rank takes the same branch.

        Args:
            do_perm: This rank's local decision.
            device: Device to allocate the reduction buffer on.

        Returns:
            The agreed decision (``do_perm`` verbatim outside a distributed run).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return do_perm
        flag = torch.tensor([1.0 if do_perm else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item() > 0.0)

    def _should_run_perm(self, batch_idx: int, batch_size: int, stage: str) -> bool:
        r"""Decide whether to evaluate the source-permutation control on this step.

        Ported from ``trainer_lag_attn_v3._should_run_perm``, with the model attribute read
        through ``getattr`` because the synthetic wrapper also hosts v1 / v2 models that have
        no ``perm_every_n_batches``. Validation runs the control on every step (under
        ``no_grad`` it is cheap, and ``val/shuffle_penalty`` is the headline readout);
        training subsamples it on a rank-invariant ``batch_idx`` schedule.

        Args:
            batch_idx: Batch index within the epoch (rank-invariant).
            batch_size: Local batch size; a batch with $B < 2$ cannot be deranged.
            stage: ``'train'`` / ``'val'`` / ``'test'``.

        Returns:
            Whether this rank wants to run the control (before the MIN reduction).
        """
        if batch_size < 2:
            return False
        if stage != "train":
            return True
        every = max(int(getattr(self.orig_model, "perm_every_n_batches", 4)), 1)
        return batch_idx % every == 0

    def _apply_curriculum(self, epoch: int) -> Optional[float]:
        r"""Apply the v2 curriculum stage for ``epoch`` and return its $\beta$.

        Reads the ``curriculum`` hparam (``{enabled, stages}``); when enabled and
        the wrapped model supports it (v2), calls
        ``model.set_curriculum_stage(epoch, stages)`` to flip the branch flags and
        ``active_lags`` in place and writes the resolved per-epoch $\beta$ into
        ``self.hparams['kld_beta']`` (read verbatim by
        :meth:`compute_loss_and_metrics`). A no-op for v1 or when disabled.

        Args:
            epoch: The epoch whose stage to apply (``self.current_epoch``).

        Returns:
            The resolved $\beta$, or ``None`` when the curriculum is inactive.
        """
        cur = self.hparams.get("curriculum")
        if not isinstance(cur, dict) or not cur.get("enabled", False):
            return None
        stages = cur.get("stages")
        if not stages or not hasattr(self.orig_model, "set_curriculum_stage"):
            return None
        beta = float(self.orig_model.set_curriculum_stage(int(epoch), stages))
        self.hparams["kld_beta"] = beta
        return beta

    def _resolve_beta(self, epoch: int) -> float:
        r"""Resolve $\beta$ for ``epoch`` from the ``beta_schedule`` hparam.

        Ported from ``trainer_lag_attn_v1._resolve_beta``. Supports two kinds:

        * ``constant`` -- ``value`` if given, else the current ``kld_beta``;
        * ``linear_warmup`` -- ``start + (end - start) * min(1, epoch / warmup_epochs)``,
          holding at ``end`` afterwards (``warmup_epochs <= 0`` returns ``end`` immediately).

        When no ``beta_schedule`` is configured this returns the current ``kld_beta`` verbatim,
        so writing it back is a no-op and the constant-$\beta$ (v1) / curriculum (v2) behaviour
        is preserved. This is not cosmetic for v3: it starts at $K \equiv 0$ and must *grow*
        $K$ to earn prediction gain, whereas v1 starts with $K > 0$ from a random log-variance
        head and $\beta$ pushes it down.

        Args:
            epoch: The epoch to resolve (``self.current_epoch``).

        Returns:
            The resolved $\beta$.

        Raises:
            ValueError: On an unknown ``beta_schedule.kind``.
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
            f"Unknown beta_schedule.kind={kind!r}; expected 'constant' or 'linear_warmup'."
        )

    def _on_train_epoch_start_hook(self) -> None:
        r"""Resolve the per-epoch $\beta$ at the start of each training epoch.

        Precedence (S2-T01b): the v2 curriculum runs first (a no-op for v1 / v3), then the
        ``beta_schedule`` is resolved and wins. When ``beta_schedule`` is absent,
        :meth:`_resolve_beta` returns the current ``kld_beta`` unchanged, so a curriculum's
        per-stage $\beta$ (v2) or the fixed ``kld_beta`` (v1) survives. Configuring BOTH a
        curriculum and a ``beta_schedule`` therefore lets the schedule override the curriculum;
        ``beta_select`` nulls the schedule so its swept $\beta$ is not overwritten (S2-T01b).
        """
        self._apply_curriculum(self.current_epoch)
        self.hparams["kld_beta"] = self._resolve_beta(self.current_epoch)

    # ------------------------------------------------------------------
    # Loss / metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Forward + synthetic loss, returning ``(total_loss, metrics)``.

        Args:
            batch: An :class:`dataset_v2.AttributeDict` carrying ``fhr_st``,
                ``fhr_ph``, ``up_st``, ``up_ph`` and ``weight``.
            batch_idx: Batch index within the epoch (unused; kept for the API).
            stage: ``'train'`` / ``'val'`` / ``'test'``.

        Returns:
            ``(total_loss, metrics)`` where ``metrics['kld_nats']`` is the
            dim-summed KL in nats/step (the $\bar K$ surrogate scale).
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
            detach_baseline_in_full=bool(hp["detach_baseline_in_full"]),
            lambda_lag=float(hp.get("lambda_lag", 0.0)),
        )
        total_loss = loss_dict["total_loss"]
        pred_gap = loss_dict["base_loss"] - loss_dict["feat_loss"]

        metrics = {
            "total_loss": total_loss,
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            # KL in nats/step (the scale of the TE surrogate $\bar K$). v1's ``kld_loss`` is
            # a per-dim mean, so nats/step $= \mathrm{kld} \cdot d_z$. Under v3 with
            # ``free_bits > 0`` the optimised ``kld_loss`` is the FLOORED ``kld_train``, which
            # is NOT a TE surrogate; only the raw KL may be read as one (G4). So read
            # ``kld_raw`` when the model exposes it, else fall back to ``kld_loss`` (they
            # coincide at ``free_bits = 0``). v2's ``kld_loss`` is already summed over heads
            # and latent dims (nats/step) and is used directly (detected by its decomposed-KL
            # return keys).
            "kld_nats": (
                loss_dict["kld_loss"]
                if "kld_content_loss" in loss_dict
                else loss_dict.get("kld_raw", loss_dict["kld_loss"])
                * float(forward_outputs["mu_post"].shape[-1])
            ),
            "pred_gap": pred_gap,
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "kld_beta": float(hp["kld_beta"]),
        }

        # --- v3 diagnostics (S2-T03), key-presence guarded ------------------
        # kld_raw (TE surrogate) vs kld_train (free-bit-floored optimised KL) so the beta
        # schedule's effect on the bottleneck is visible; kld_active_frac = fraction of latent
        # dims with K_j > eps (G4); mean_logvar_{full,base} monitor variance collapse. All are
        # ``.get``-guarded and, crucially, only added when NOT None -- ``_log_metrics`` cannot
        # log a None (v1 / mse runs leave the logvar means unset).
        for _k in ("kld_raw", "kld_train", "kld_active_frac"):
            _v = loss_dict.get(_k)
            if _v is None:
                _v = forward_outputs.get(_k)
            if _v is not None:
                metrics[_k] = _v
        for _k in ("mean_logvar_full", "mean_logvar_base"):
            _v = loss_dict.get(_k)
            if _v is not None:
                metrics[_k] = _v

        # --- v2-only diagnostics (S7-T03), key-presence guarded -------------
        # ``delta_l`` is the production ``pred_gap`` under a different name; add
        # it so the synthetic side logs the section-26 gain too. The rest are
        # emitted only when the v2 model provides them (no-op under v1).
        metrics["delta_l"] = loss_dict.get("delta_l", pred_gap)
        for _k in ("kld_lag_loss", "kld_content_loss", "kld_content_raw",
                   "r_lag", "rms_src", "lag_tv", "lag_entropy_reg"):
            _v = loss_dict.get(_k)
            if _v is not None:
                metrics[_k] = _v
        for _k in ("lag_entropy", "n_active"):
            _v = forward_outputs.get(_k)
            if _v is not None:
                metrics[_k] = _v
        _exp_lag = forward_outputs.get("expected_lag")
        if _exp_lag is not None:
            metrics["expected_lag_mean"] = _exp_lag.mean()
            if hasattr(self.orig_model, "expected_lag_seconds"):
                metrics["expected_lag_sec_mean"] = (
                    self.orig_model.expected_lag_seconds(_exp_lag).mean()
                )

        self._add_perm_control_metrics(
            metrics, forward_outputs, loss_dict, y_st=y_st, y_ph=y_ph,
            weight=getattr(batch, "weight", None), batch_idx=batch_idx, stage=stage,
        )
        return total_loss, metrics

    def _add_perm_control_metrics(
        self,
        metrics: Dict[str, torch.Tensor],
        forward_outputs: Dict[str, torch.Tensor],
        loss_dict: Dict[str, torch.Tensor],
        *,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        weight: Optional[torch.Tensor],
        batch_idx: int,
        stage: str,
    ) -> None:
        r"""Add the source-permutation control metrics in place (S3-T01, readout only).

        Deranges ``source_state`` along the batch axis and re-runs only the lag attention +
        posterior head (``perm_kl_from_forward``) and the source-dependent decoder path
        (``perm_forward_outputs``), reusing the completed forward's encoder states. Two
        readouts fall out:

        * $K_{\mathrm{shuffled}} / K_{\mathrm{raw}}$ -- the KL-space ratio. Per v3 Finding F2
          this sits near $1$ on an honest model and is **not** expected to vanish:
          $\mathrm{KL}(q \,\|\, p)$ measures "the source moved my belief", not "...correctly".
        * $\mathcal L_{\mathrm{feat}}^{\pi(U)} - \mathcal L_{\mathrm{feat}}$ -- the
          prediction-space penalty, which *does* discriminate: a model exploiting the source
          must forecast worse under a wrong source.

        Everything runs under ``no_grad``. The synthetic pipeline pins ``lambda_perm: 0.0``
        (a hard non-goal), so the control never enters the objective and never builds an
        autograd graph; ``total_loss`` is left untouched, which matters because
        :meth:`training_step` feeds the spike circuit breaker the *returned* loss.

        Naming: this is a ``source_state`` **batch derangement**, logged as ``feat_loss_perm``.
        The evaluation-time controls in ``eval_v2`` corrupt the **input stream** (``shuffle`` /
        ``reverse``) and are logged as ``feat_loss_shuffle`` / ``feat_loss_reverse``. They are
        different corruptions and must never be tabulated as comparable.

        Args:
            metrics: The metrics dict to extend in place.
            forward_outputs: The completed forward's outputs.
            loss_dict: The completed ``compute_loss`` result (supplies ``kld_raw``/``feat_loss``).
            y_st: Scattering-coefficient FHR target $(B, T, 43)$.
            y_ph: Phase-harmonic FHR target $(B, T, 44)$.
            weight: Optional per-sample weight broadcastable to $(B, T)$.
            batch_idx: Batch index within the epoch.
            stage: ``'train'`` / ``'val'`` / ``'test'``.
        """
        if not hasattr(self.orig_model, "perm_kl_from_forward"):
            return  # v1 / v2: no perm API, nothing is logged and nothing changes.

        hp = self.hparams
        lambda_perm = float(getattr(self.orig_model, "lambda_perm", 0.0))
        if lambda_perm > 0.0 and not self._perm_lambda_warned:
            self._perm_lambda_warned = True
            logger.warning(
                "lambda_perm={} is ignored: the synthetic_v2 wrapper runs the permutation "
                "control as a no-grad READOUT only (v3 Finding F2). Use "
                "trainer_lag_attn_v3 to optimise L_perm.",
                lambda_perm,
            )

        # ``kld_raw`` is the un-floored KL, i.e. the only term readable as a TE surrogate (G4).
        kld_raw = loss_dict.get("kld_raw", loss_dict["kld_loss"])
        keys = ("kld_shuffled", "kld_shuffled_ratio", "feat_loss_perm", "shuffle_penalty")

        do_perm = self._sync_perm_decision(
            self._should_run_perm(batch_idx, int(y_st.size(0)), stage), y_st.device
        )
        if not do_perm:
            # Zero-fill so the CSV columns stay dense; ``_log_metrics`` cannot log ``None``.
            zero = torch.zeros_like(kld_raw.detach())
            for key in keys:
                metrics[key] = zero
            return

        with torch.no_grad():
            perm = self.orig_model.perm_kl_from_forward(
                forward_outputs, weight=weight, generator=self._perm_generator
            )
            permuted = self.orig_model.perm_forward_outputs(
                forward_outputs, perm_index=perm["perm_index"]
            )
            shuffled = self.orig_model.compute_loss(
                forward_outputs=permuted,
                y_st=y_st,
                y_ph=y_ph,
                weight=weight,
                compute_kld_loss=False,
                beta=0.0,
                lambda_full=float(hp["lambda_full"]),
                lambda_base=float(hp["lambda_base"]),
                likelihood=str(hp["likelihood"]),
                sigma_obs=hp["sigma_obs"],
                detach_baseline_in_full=bool(hp["detach_baseline_in_full"]),
            )

        kld_shuffled = perm["kld_shuffled"]
        metrics["kld_shuffled"] = kld_shuffled
        metrics["kld_shuffled_ratio"] = kld_shuffled / kld_raw.detach().clamp_min(1e-8)
        metrics["feat_loss_perm"] = shuffled["feat_loss"]
        metrics["shuffle_penalty"] = shuffled["feat_loss"] - loss_dict["feat_loss"].detach()

    # ------------------------------------------------------------------
    # Spike-skip circuit breaker
    # ------------------------------------------------------------------
    @property
    def _spike_cfg(self) -> Dict[str, Any]:
        r"""Resolve the spike-skip config from ``self.hparams`` with defaults."""
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
        r"""``MAX``-reduce ``is_spike`` so every DDP rank takes the same branch."""
        if not (dist.is_available() and dist.is_initialized()):
            return is_spike
        flag = torch.tensor([1.0 if is_spike else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0.0)

    def training_step(self, batch: Any, batch_idx: int):  # type: ignore[override]
        r"""Forward + loss gated by a cross-rank-synced spike check."""
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

        # Update the EMA only on accepted batches (a spike must not raise the bar
        # for the next one). During warmup we always update (priming).
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
            # No-op loss built from the parameters (not the possibly-NaN ``loss``) so
            # backward stays finite. It must touch EVERY trainable parameter: the real
            # forward above already armed DDP's reducer, which then expects a gradient
            # hook to fire for each one. A single-parameter anchor leaves the rest
            # unreduced and the next iteration raises ``Expected to have finished
            # reduction in the prior iteration`` -- under find_unused_parameters=False
            # AND =True alike. ``nan_to_num`` keeps the *value* finite when a parameter
            # has itself gone NaN (its gradient is 0 either way, but a NaN loss would be
            # logged and monitored).
            return sum(
                torch.nan_to_num(p).sum()
                for p in self.parameters()
                if p.requires_grad
            ) * 0.0

        return loss

    # ------------------------------------------------------------------
    # LR schedule: optional linear warmup -> MultiStepLR
    # ------------------------------------------------------------------
    def build_lr_scheduler(self, optimizer):
        r"""Linear warmup for ``warmup_epochs`` then the ``MultiStepLR`` decay.

        With ``warmup_epochs == 0`` this is the plain ``MultiStepLR``. The DDP LR
        scaling (``lr * n_gpus``) is applied by the caller before construction; the
        warmup ramps from a small fraction up to that scaled ``lr``.
        """
        milestones = list(self.hparams.get("lr_milestones") or [])
        gamma = float(self.hparams.get("lr_gamma", 0.1))
        warmup_epochs = int(self.hparams.get("warmup_epochs", 0) or 0)

        from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR

        if not milestones and warmup_epochs <= 0:
            return None
        if warmup_epochs <= 0:
            scheduler: Any = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            warmup = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            # Shift milestones so the decay counts epochs *after* warmup.
            shifted = MultiStepLR(
                optimizer,
                milestones=[max(0, m - warmup_epochs) for m in milestones],
                gamma=gamma,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, shifted], milestones=[warmup_epochs]
            )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}


# ============================================================================
# Builders (ported from synthetic/train_minimal.py + train_ddp.py)
# ============================================================================
# Config-driven model-class registry. The committed :data:`SeqVaeLagAttn` toggle alias
# stays the default (v1 today); the ``synthetic_v3`` ablation ladder selects
# ``SeqVaeLagAttnV3`` by an optional ``model.class`` string instead of flipping the
# toggle, so the v1/v2 synthetic paths and ``test_rollback_v1`` are untouched. The v1
# key is the alias's *runtime* ``__name__`` (never the bare literal), keeping the alias
# the single source of truth for the default and satisfying the alias-seam guards.
_KNOWN_MODEL_CLASSES: Tuple[str, ...] = (
    SeqVaeLagAttn.__name__,
    "SeqVaeLagAttnV2",
    "SeqVaeLagAttnV3",
)


def _resolve_model_class(name: Optional[str]) -> type:
    r"""Resolve a ``model.class`` name to its class, defaulting to the committed alias.

    Resolution is lazy: only the toggle alias is imported at module load; the v2 / v3
    classes are imported here, on demand, the first time a config selects them. An absent
    ``name`` returns the alias, so every existing v1 / v2 caller is byte-unchanged.

    Args:
        name: The ``model.class`` value, or ``None`` to use the committed alias.

    Returns:
        The resolved model class (an :class:`torch.nn.Module` subclass).

    Raises:
        ValueError: If ``name`` is a non-empty string outside
            :data:`_KNOWN_MODEL_CLASSES`.
    """
    if name is None:
        return SeqVaeLagAttn
    name = str(name)
    if name == SeqVaeLagAttn.__name__:
        # Whatever the committed toggle currently points at (v1 today).
        return SeqVaeLagAttn
    if name == "SeqVaeLagAttnV3":
        from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
            SeqVaeLagAttnV3,
        )

        return SeqVaeLagAttnV3
    if name == "SeqVaeLagAttnV2":
        from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (
            SeqVaeLagAttnV2,
        )

        return SeqVaeLagAttnV2
    raise ValueError(
        f"unknown model.class {name!r}; known classes: {_KNOWN_MODEL_CLASSES}"
    )


def build_model(
    model_cfg: Dict[str, Any], device: torch.device
) -> Tuple[SeqVaeLagAttn, Dict[str, Any]]:
    r"""Construct :class:`SeqVaeLagAttn` from the ``model`` config block.

    The block is a 1:1 map of the keyword-only constructor arguments. Two extra
    conventions make it a version-robust drop-in for the ``SeqVaeLagAttn``
    comment-toggle alias:

    * An optional ``class`` key selects the constructor from the config-driven
      :func:`_resolve_model_class` registry (e.g. ``SeqVaeLagAttnV3`` for the
      ``synthetic_v3`` arms); absent, it falls back to the committed
      :data:`SeqVaeLagAttn` toggle alias. The resolved class drives **both** the
      ``inspect.signature`` filter and the construction, so a v3-only kwarg is never
      dropped just because the toggle still points at v1.
    * Two nested overlays hold version-specific tuning: ``v2`` (``use_entmax``, the
      source lag-atom scales, the model-owned lag-regularizer weights, the
      physical-time / optional-feature flags) and ``v3`` (``causal_norm``,
      ``posterior_logvar``, ``logvar_bound``, ``kld_support``, ...). Each is merged
      over the flat keys only when the *resolved* class accepts its sentinel
      parameter (``source_scales`` for v2, ``posterior_logvar`` for v3) and both are
      always stripped before construction, so the shared block stays a clean v1
      reproduction under a v1 class.
    * Any remaining key the resolved constructor does not accept is dropped (with a
      debug log), mirroring the production trainer's ``inspect.signature`` guard,
      so a superset config never raises a bare ``TypeError``.

    Args:
        model_cfg: The ``model`` block (an optional ``class`` key, flat v1 kwargs,
            plus optional nested ``v2`` / ``v3`` overlays). ``c_y=87`` / ``c_u=101``
            for the v2 features.
        device: Device to move the model onto.

    Returns:
        ``(model, model_kwargs)`` where ``model_kwargs`` is the exact resolved
        (flat, overlay-merged, signature-filtered) constructor kwargs, stored
        verbatim in the checkpoint so downstream phases rebuild the architecture
        without the config. The resolved class is recoverable as ``type(model)``.
    """
    kwargs = deepcopy(dict(model_cfg))
    # Resolve the constructor from the optional ``model.class`` (default: the toggle
    # alias). ``cls`` drives the signature filter AND the construction -- pointing
    # ``accepted`` at the alias while constructing a v3 would silently drop every
    # v3-only kwarg and build v3's defaults, so both must use the same resolved class.
    cls = _resolve_model_class(kwargs.pop("class", None))
    accepted = set(inspect.signature(cls.__init__).parameters)
    # Strip the nested version overlays (never constructor args) and merge each only
    # when the resolved class accepts its sentinel parameter. A rebuild from a
    # checkpoint's flat ``model_kwargs`` carries no overlay key, so this is a no-op there.
    v2_overlay = kwargs.pop("v2", None)
    v3_overlay = kwargs.pop("v3", None)
    if v2_overlay and "source_scales" in accepted:
        kwargs.update(deepcopy(dict(v2_overlay)))
    if v3_overlay and "posterior_logvar" in accepted:
        kwargs.update(deepcopy(dict(v3_overlay)))
    # Drop any key the resolved constructor does not accept (e.g. v3-only keys under
    # a v1 class). Keeps the shared config a superset without a TypeError.
    dropped = [k for k in kwargs if k not in accepted]
    for key in dropped:
        del kwargs[key]
    if dropped:
        logger.debug(
            "[build_model] dropped {} config key(s) not accepted by {}: {}",
            len(dropped), cls.__name__, dropped,
        )
    # YAML gives ``logvar_clamp`` as a list; the constructor expects a tuple.
    clamp = kwargs.get("logvar_clamp")
    if clamp is not None:
        kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))
    # ``LagCrossAttention`` silently degrades ``use_entmax=True`` to softmax when the
    # optional ``entmax`` package is missing, while ``model_kwargs`` still records
    # ``True``. On the ablation ladder that would drop the sparse-attention leg of the
    # parity -> v3_noncausal contrast and leave the checkpoint's provenance lying about
    # it. Refuse to build instead.
    if kwargs.get("use_entmax"):
        try:
            import entmax  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "model.use_entmax is true but the 'entmax' package is not importable, "
                "so LagCrossAttention would silently fall back to softmax while the "
                "checkpoint recorded use_entmax=True. Install it (`pip install entmax`) "
                "or set use_entmax: false."
            ) from exc
    model = cls(**kwargs)
    model.to(device)
    return model, kwargs


def resolved_model_class_name(model_cfg: Dict[str, Any]) -> str:
    r"""The class name a ``model`` block resolves to (via ``model.class`` or the alias).

    Lets a grading path state the class it *expects* from a checkpoint without importing any
    concrete model class (so the alias-seam guards stay green).
    """
    return _resolve_model_class(model_cfg.get("class")).__name__


def rebuild_model_from_checkpoint(
    blob: Dict[str, Any],
    device: torch.device,
    *,
    expected_class: Optional[str] = None,
) -> Tuple[SeqVaeLagAttn, Dict[str, Any]]:
    r"""Rebuild a model from a checkpoint blob's OWN class + kwargs (architecture-faithful).

    A checkpoint must always be graded with the architecture it was trained under. The three
    ``synthetic_v3`` arms share one flat kwarg layout but differ in ``posterior_logvar`` (and,
    for ``v3_prod``, ``causal_norm``), so rebuilding from ``blob["model_kwargs"]`` -- rather
    than from the run config -- is what stops arm B's checkpoint being silently graded as arm
    C. The stored ``model_class`` selects the constructor through the registry, so a v3
    checkpoint never falls back to the v1 alias (which a bare ``build_model(model_kwargs)``
    would, since the flat kwargs carry no ``class`` key).

    Args:
        blob: A deserialised checkpoint dict (carrying ``model_class`` / ``model_kwargs``).
        device: Device to build the rebuilt model on.
        expected_class: When given, the class name the caller expects (e.g. the configured
            ``model.class``). A mismatch raises via
            :func:`vae_teb_lag_attn_trfr.check_model_class`; a blob with no stored
            ``model_class`` (a legacy checkpoint) warns and falls back to the committed alias.
            Absent, no cross-check is made.

    Returns:
        ``(model, model_kwargs)`` -- the model built on ``device`` and its resolved kwargs.

    Raises:
        ValueError: When ``expected_class`` is given and the blob's ``model_class`` differs.
    """
    from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import check_model_class

    if expected_class is not None:
        check_model_class(blob, expected_class)
    rebuild_cfg = dict(blob.get("model_kwargs") or {})
    stored_class = blob.get("model_class")
    if stored_class:
        # Route the checkpoint's own class through the registry (never the flat-kwargs alias
        # fallback), so an arm trained as v3 rebuilds as v3.
        rebuild_cfg["class"] = stored_class
    return build_model(rebuild_cfg, device)


def _resolve_loss_settings(loss_cfg: Dict[str, Any]) -> Dict[str, Any]:
    r"""Resolve the synthetic loss settings from the ``loss`` config block.

    Parses ``sigma_obs`` (positive float or the literal ``'learned'``) and the
    remaining calibration knobs. Mirrors the v1 driver so the objective is
    identical across pipelines.

    Args:
        loss_cfg: The ``loss`` block.

    Returns:
        ``{beta, lambda_full, lambda_base, likelihood, sigma_obs, free_bits,
        detach_baseline_in_full}``.

    Raises:
        ValueError: If ``sigma_obs`` is a non-numeric string other than
            ``'learned'``.
    """
    beta = float(loss_cfg["kld_beta"])
    lambda_full = float(loss_cfg["lambda_full"])
    lambda_base = float(loss_cfg["lambda_base"])
    likelihood = str(loss_cfg.get("likelihood", "mse"))
    sigma_obs_raw = loss_cfg.get("sigma_obs", 1.0)
    if isinstance(sigma_obs_raw, str) and sigma_obs_raw != "learned":
        try:
            sigma_obs: "float | str" = float(sigma_obs_raw)
        except ValueError as exc:
            raise ValueError(
                f"loss.sigma_obs must be a positive float or 'learned', "
                f"got {sigma_obs_raw!r}"
            ) from exc
    else:
        sigma_obs = (
            sigma_obs_raw if isinstance(sigma_obs_raw, str) else float(sigma_obs_raw)
        )
    free_bits = float(loss_cfg.get("free_bits", 0.0))
    detach_baseline_in_full = bool(loss_cfg.get("detach_baseline_in_full", False))
    lambda_lag = float(loss_cfg.get("lag_smoothness_lambda", 0.0))
    # Optional per-epoch beta schedule (v3): a ``{kind: constant|linear_warmup, ...}`` dict
    # resolved by :meth:`SyntheticSeqVaeLagAttnV2Pl._resolve_beta` each epoch. Absent, the
    # constant ``kld_beta`` above is used (v1 / v2 behaviour unchanged).
    beta_schedule = loss_cfg.get("beta_schedule")
    beta_schedule = dict(beta_schedule) if isinstance(beta_schedule, dict) else None
    return {
        "beta": beta,
        "lambda_full": lambda_full,
        "lambda_base": lambda_base,
        "likelihood": likelihood,
        "sigma_obs": sigma_obs,
        "free_bits": free_bits,
        "detach_baseline_in_full": detach_baseline_in_full,
        "lambda_lag": lambda_lag,
        "beta_schedule": beta_schedule,
    }


def _parse_devices(spec: Any) -> Tuple[Any, int]:
    r"""Parse a ``devices`` spec into ``(lightning_devices, n_gpus)``.

    Accepts an int (use the first N GPUs), a comma list ``"0,1,2"`` (explicit
    indices), or an actual list. Returns the value to hand to :class:`pl.Trainer`
    and the GPU count used for LR scaling / strategy selection.
    """
    if isinstance(spec, bool):  # guard: bool is an int subclass
        return int(spec), int(spec)
    if isinstance(spec, int):
        return spec, int(spec)
    if isinstance(spec, (list, tuple)):
        idx = [int(x) for x in spec]
        return idx, len(idx)
    text = str(spec).strip()
    if "," in text:
        idx = [int(x) for x in text.split(",") if x != ""]
        return idx, len(idx)
    return int(text), int(text)


def _batch_to_inputs(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Map a batch to ``(y_st, y_ph, u_stream)`` for :meth:`fit_latent_stats`."""
    return batch.fhr_st, batch.fhr_ph, build_u_stream(batch)


def _maybe_fit_latent_stats(
    model: SeqVaeLagAttn, loader: Any, device: torch.device
) -> int:
    r"""Run :meth:`SeqVaeLagAttn.fit_latent_stats` over the training set.

    Replaces the noisy EMA ``mu_post_running_*`` buffers with exact statistics.
    Non-fatal: returns ``0`` on any failure (or if the method is absent) so the
    training run still completes and writes its checkpoint.

    Args:
        model: The trained model.
        loader: A full-train, unshuffled :class:`DataLoader`.
        device: Compute device.

    Returns:
        The number of time-step samples aggregated, or ``0`` on failure.
    """
    fit = getattr(model, "fit_latent_stats", None)
    if fit is None:
        return 0
    try:
        return int(fit(loader, device=device, batch_to_inputs=_batch_to_inputs))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[train_v2] fit_latent_stats failed (non-fatal): {}", exc)
        return 0


# ============================================================================
# Persistence (v1-compatible checkpoint format)
# ============================================================================
def save_checkpoint_v2(
    path: Path,
    *,
    model: SeqVaeLagAttn,
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    epoch: int,
    val_loss: float,
    loss_settings: Dict[str, Any],
    latent_stats_fitted: bool,
    train_metrics: Optional[Dict[str, float]] = None,
    arm: Optional[str] = None,
) -> None:
    r"""Save a training checkpoint in the v1-compatible format.

    The bare (unprefixed) ``state_dict`` is stored under ``model_state_dict`` -- the
    key :func:`train.graph_models_utils.load_checkpoint_strict` scans for. A
    downstream phase rebuilds the model via ``SeqVaeLagAttn(**model_kwargs)`` then
    loads this file ``strict=True``. The write is atomic (temp + ``os.replace``).

    Args:
        path: Destination ``.ckpt`` path.
        model: The model to checkpoint.
        model_kwargs: Exact resolved constructor kwargs.
        config: The effective (post-override) config.
        data_meta: The dataset ``meta.json`` (carries the generator seeds).
        epoch: Epoch index this checkpoint corresponds to.
        val_loss: Validation total loss at this checkpoint (may be ``nan``).
        loss_settings: The resolved loss settings dict.
        latent_stats_fitted: Whether :meth:`fit_latent_stats` ran (exact buffers)
            vs. the noisy EMA buffers.
        train_metrics: Optional last-epoch training metrics.
        arm: The resolved ``synthetic_v3`` arm name (``parity`` / ``v3_noncausal`` /
            ``v3_prod``), or ``None`` for the arm-less v1 / v2 path. Recorded in the
            blob so any figure traces back to the exact arm that produced it.
    """
    # Persist the CURRENT runtime curriculum state. active_lags / enable_* are
    # constructor params but are mutated in place per-epoch by the curriculum and
    # are NOT part of the state_dict, so refresh model_kwargs from the live model
    # to keep a rebuilt (SeqVaeLagAttn(**model_kwargs)) model faithful to the stage
    # the weights were trained under. No-op for v1 (attributes absent).
    model_kwargs = dict(model_kwargs)
    for _key in ("active_lags", "enable_source", "enable_residual", "enable_kl"):
        if hasattr(model, _key):
            model_kwargs[_key] = getattr(model, _key)

    ckpt = {
        "model_class": type(model).__name__,
        "arm": arm,
        "model_state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "config": config,
        "data_meta": data_meta,
        "epoch": int(epoch),
        "val_total_loss": float(val_loss),
        "train_metrics": {k: float(v) for k, v in (train_metrics or {}).items()},
        "loss_settings": loss_settings,
        "latent_stats_fitted": bool(latent_stats_fitted),
        "torch_version": torch.__version__,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


# ============================================================================
# Driver
# ============================================================================
def _results_dir(config: Dict[str, Any], benchmark: str) -> Path:
    r"""Resolve the ``results/<tag>/`` output directory (matches ``run_pipeline_v2``)."""
    tag = str(config.get("experiment", {}).get("tag", benchmark))
    results_dir = Path(config.get("paths", {}).get("results_dir", "./results"))
    if not results_dir.is_absolute():
        results_dir = _MODULE_DIR / results_dir
    return results_dir / tag


def _run_dir(config: Dict[str, Any], benchmark: str,
             arm: Optional[str] = None) -> Path:
    r"""Resolve the per-run output root ``results/<tag>/<arm>/`` (matches ``run_pipeline_v2``).

    Arm-scopes the checkpoint / log / figure write sites so the three ``synthetic_v3`` arms
    coexist under one ``experiment.tag``. ``arm=None`` reproduces :func:`_results_dir`
    exactly (the v1 / v2 single-arm path).
    """
    base = _results_dir(config, benchmark)
    return base / str(arm) if arm else base


def _callback_metrics_to_floats(trainer: pl.Trainer) -> Dict[str, float]:
    r"""Extract the trainer's ``callback_metrics`` as a plain ``{name: float}`` dict."""
    out: Dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _select_ddp_strategy(
    num_devices: int,
    likelihood: str,
    sigma_obs: Any,
    curriculum_enabled: bool = False,
    *,
    head_structured_latent: bool = False,
    freeze_unused_attn_proj: bool = False,
) -> str:
    r"""Resolve the DDP strategy string for a v3 run (ported from ``trainer_lag_attn_v3``).

    Returns plain ``'ddp'`` (``find_unused_parameters=False``) only when EVERY parameter is
    guaranteed a gradient every step, so the reducer never waits on a variable that is not
    marked ready. Two starvation sources are accounted for:

    * the decoder log-variance heads are dead unless the objective consumes them
      (``gaussian_nll`` + ``sigma_obs='learned'``);
    * ``head_structured_latent`` starves ``lag_attn.W_o`` unless it is frozen
      (``freeze_unused_attn_proj``), in which case it carries no grad by design.

    A curriculum (which switches branches on/off per epoch) always needs
    ``find_unused_parameters``. Single-device runs return ``'auto'``.

    Args:
        num_devices: Number of training devices.
        likelihood: The reconstruction likelihood (``'mse'`` / ``'gaussian_nll'``).
        sigma_obs: The observation-noise setting (a float, or the literal ``'learned'``).
        curriculum_enabled: Whether the v2 curriculum is active.
        head_structured_latent: Whether the latent is head-structured.
        freeze_unused_attn_proj: Whether ``lag_attn.W_o`` is frozen.

    Returns:
        ``'auto'`` (single device), ``'ddp'`` (full grad coverage), or
        ``'ddp_find_unused_parameters_true'``.
    """
    if num_devices <= 1:
        return "auto"
    if curriculum_enabled:
        return "ddp_find_unused_parameters_true"
    logvar_heads_consumed = (
        likelihood == "gaussian_nll"
        and isinstance(sigma_obs, str)
        and sigma_obs == "learned"
    )
    attn_proj_starved = bool(head_structured_latent) and not bool(freeze_unused_attn_proj)
    if logvar_heads_consumed and not attn_proj_starved:
        return "ddp"
    return "ddp_find_unused_parameters_true"


def _build_trainer_v2(
    *,
    ddp_cfg: Dict[str, Any],
    results_dir: Path,
    epochs: int,
    devices: Any,
    n_gpus: int,
    has_val: bool,
    overrides: Dict[str, Any],
    plotting_cfg: Optional[Dict[str, Any]] = None,
) -> pl.Trainer:
    r"""Build the Lightning :class:`Trainer` (single-GPU default, DDP when ``n_gpus>1``).

    Registers, in order: a :class:`LearningRateMonitor`; a :class:`ModelCheckpoint`
    (when ``has_val``); an :class:`EarlyStopping` (when ``has_val`` and
    ``ddp_cfg['early_stopping'].enabled``); and a live
    :class:`~model.vae_teb_prediction.model.model_experiment.synthetic_v2.callbacks_v2.LossPlotHtmlCallback`
    (when ``plotting_cfg`` enables the interactive HTML curve). All metric monitors are
    resolved to their epoch-logged keys via :func:`_resolve_epoch_metric`, so the config
    can keep the clean ``val/total_loss`` name.

    Args:
        ddp_cfg: The merged ``ddp`` config (precision, grad-clip, sanity steps,
            and the optional ``early_stopping`` sub-block).
        results_dir: Output root (``lightning_ckpts/`` + ``logs/`` land here).
        epochs: ``max_epochs``.
        devices: Lightning ``devices`` spec (from :func:`_parse_devices`).
        n_gpus: GPU count (selects the strategy / sync-batchnorm).
        has_val: Whether a validation split is available (enables checkpointing and
            early stopping — both need a monitored ``val/*`` metric).
        overrides: Extra knobs; ``limit_train_batches`` / ``limit_val_batches``
            support the ``--pilot`` smoke.
        plotting_cfg: The ``plotting`` config block; when ``enabled`` and ``html``
            are truthy a :class:`LossPlotHtmlCallback` refreshes an interactive
            Plotly loss curve every ``plot_every`` epochs.

    Returns:
        A configured :class:`lightning.pytorch.Trainer`.
    """
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger

    callbacks: List[Any] = [LearningRateMonitor(logging_interval="epoch")]
    if has_val:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(results_dir / "lightning_ckpts"),
                monitor=_resolve_epoch_metric("val/total_loss"),
                mode="min",
                save_top_k=1,
                filename="best-{epoch:03d}",
            )
        )

    # Early stopping (opt-in via ``ddp.early_stopping.enabled``). Needs a val split for
    # a monitored metric, so gate on ``has_val`` exactly like ``ModelCheckpoint``.
    es_cfg = {**_EARLY_STOPPING_DEFAULTS, **(ddp_cfg.get("early_stopping") or {})}
    if es_cfg.get("enabled"):
        if has_val:
            callbacks.append(
                EarlyStopping(
                    monitor=_resolve_epoch_metric(str(es_cfg["monitor"])),
                    mode=str(es_cfg["mode"]),
                    patience=int(es_cfg["patience"]),
                    min_delta=float(es_cfg["min_delta"]),
                    strict=True,
                    verbose=True,
                )
            )
        else:
            logger.warning(
                "[train_v2] early_stopping.enabled but no validation split; "
                "skipping EarlyStopping (nothing to monitor)."
            )

    # Live interactive Plotly HTML loss curve (rewritten every ``plot_every`` epochs
    # and finalised at fit end). Non-fatal, rank-0-only inside the callback itself.
    plotting_cfg = plotting_cfg or {}
    if plotting_cfg.get("enabled", True) and plotting_cfg.get("html", True):
        from .callbacks_v2 import LossPlotHtmlCallback

        callbacks.append(
            LossPlotHtmlCallback(
                out_stem=results_dir / "figures" / "training_curves",
                every_n_epochs=int(plotting_cfg.get("plot_every", 10)),
            )
        )

    trainer_kwargs: Dict[str, Any] = {
        "max_epochs": epochs,
        "callbacks": callbacks,
        "default_root_dir": str(results_dir),
        "precision": ddp_cfg["precision"],
        "deterministic": False,
        "benchmark": True,
        "gradient_clip_val": ddp_cfg["gradient_clip_val"],
        "gradient_clip_algorithm": ddp_cfg["gradient_clip_algorithm"],
        "enable_checkpointing": has_val,
        "log_every_n_steps": 1,
        "num_sanity_val_steps": int(ddp_cfg["num_sanity_val_steps"]),
        "use_distributed_sampler": True,
        "enable_progress_bar": bool(overrides.get("progress_bar", True)),
        "logger": CSVLogger(save_dir=str(results_dir), name="logs"),
    }
    if "limit_train_batches" in overrides:
        trainer_kwargs["limit_train_batches"] = overrides["limit_train_batches"]
    if "limit_val_batches" in overrides:
        trainer_kwargs["limit_val_batches"] = overrides["limit_val_batches"]

    if torch.cuda.is_available() and n_gpus >= 1:
        strategy = str(ddp_cfg["strategy"]) if n_gpus > 1 else "auto"
        trainer_kwargs.update(
            {"accelerator": "gpu", "devices": devices, "strategy": strategy}
        )
        if n_gpus > 1:
            trainer_kwargs["sync_batchnorm"] = bool(ddp_cfg["sync_batchnorm"])
    else:
        trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

    return pl.Trainer(**trainer_kwargs)


def train_v2(
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Train the unchanged VAE-TEB model on the ``synthetic_v2`` cache.

    Single-GPU by default (``devices=1``); pass ``overrides['devices'] > 1`` (or a
    device list) for DDP on the prod box. Builds the model on CPU, wraps it in
    :class:`SyntheticSeqVaeLagAttnV2Pl`, fits over
    :class:`datamodule_v2.SyntheticTEDataModuleV2`, runs a post-fit latent-stats
    pass, and exports ``final.ckpt`` / ``best.ckpt`` under ``results/<tag>/`` in the
    :func:`load_checkpoint_strict`-compatible format.

    Args:
        config: The parsed ``config_synth_v2.yaml`` tree.
        overrides: Optional run overrides. Recognised keys: ``epochs``,
            ``devices``, ``batch_size``, ``limit_train_batches``,
            ``limit_val_batches``, ``loss`` (a dict merged over ``config['loss']``,
            e.g. ``{'kld_beta': ...}``), ``resume_ckpt`` (warm-start path),
            ``skip_checkpoint`` (skip the latent-stats + checkpoint export, used by
            :func:`beta_select`), and ``progress_bar``.
        benchmark: Active benchmark key (defaults to ``experiment.benchmark``).

    Returns:
        A dict with ``checkpoint`` / ``best`` paths (``None`` when
        ``skip_checkpoint``), ``metrics_csv``, ``figures`` (loss-curve paths),
        ``metrics`` (final ``callback_metrics``), ``epochs``, and ``n_stats``.
    """
    overrides = dict(overrides or {})
    benchmark = str(benchmark or config.get("experiment", {}).get("benchmark", "G1_raw"))
    # The resolved arm name (``None`` for the v1 / v2 single-arm path). Carried through
    # ``overrides`` by the driver so a DDP re-exec keeps the same arm; used to arm-scope
    # the run dir and to stamp the checkpoint provenance.
    arm = overrides.get("arm")

    seed = int(
        config.get("seeds", {}).get(
            "base_seed", config.get("experiment", {}).get("seed", 0)
        )
    )
    pl.seed_everything(seed, workers=True)

    ddp_cfg = {**_DDP_DEFAULTS, **(config.get("ddp") or {})}
    devices_spec = overrides.get("devices", 1)  # local single-GPU default; --devices overrides
    devices, n_gpus = _parse_devices(devices_spec)

    optim_cfg = config["optim"]
    loss_cfg = {**config["loss"], **(overrides.get("loss") or {})}
    loss_settings = _resolve_loss_settings(loss_cfg)

    base_lr = float(optim_cfg["lr"])
    do_scale = bool(ddp_cfg.get("lr_scaling", True)) and n_gpus > 1
    effective_lr = base_lr * n_gpus if do_scale else base_lr

    # Model on CPU first (Lightning assigns devices inside ``fit``).
    model, model_kwargs = build_model(config["model"], torch.device("cpu"))

    # Re-seed AFTER model construction (S2-T06). The three arms are structurally different
    # (``parity`` has an independent log-variance head and no ALiBi lag bias, so it draws a
    # different number of RNG samples while initialising), which would leave the global RNG in
    # a different state per arm and hence give each arm a DIFFERENT DataLoader shuffle order.
    # Section 7 promises all arms share one cache order, so that a gamma difference is
    # attributable to the model alone. The weights are already drawn; this pins only the data
    # order (and the workers' seeds).
    pl.seed_everything(seed, workers=True)

    # Resolve the DDP strategy when the config defers (``strategy`` absent or ``auto``): a v3
    # run with learned-variance NLL + head-structured latent + frozen attn proj has full grad
    # coverage, so it resolves to plain ``'ddp'`` rather than paying for
    # ``find_unused_parameters`` every step. An explicit ``ddp.strategy`` still wins (S2-T04).
    if str(ddp_cfg.get("strategy", "auto")) in ("", "auto", "None"):
        curriculum_cfg = overrides.get("curriculum", config.get("curriculum")) or {}
        ddp_cfg = {**ddp_cfg, "strategy": _select_ddp_strategy(
            n_gpus,
            loss_settings["likelihood"],
            loss_settings["sigma_obs"],
            curriculum_enabled=bool(curriculum_cfg.get("enabled", False)),
            head_structured_latent=bool(model_kwargs.get("head_structured_latent", False)),
            freeze_unused_attn_proj=bool(model_kwargs.get("freeze_unused_attn_proj", False)),
        )}

    # Startup provenance: the headline runbook's first watch-signal is that this reads
    # ``DDP strategy: ddp`` and not ``ddp_find_unused_parameters_true``. ``n_gpus <= 1``
    # forces ``'auto'`` in ``_build_trainer_v2``, so log the *effective* value.
    logger.info(
        "[train_v2] devices={} | n_gpus={} | DDP strategy: {}",
        devices, n_gpus, str(ddp_cfg["strategy"]) if n_gpus > 1 else "auto",
    )

    resume = overrides.get("resume_ckpt")
    if resume:
        from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (
            check_model_class,
        )

        logger.info("[train_v2] warm-starting from {}", resume)
        # Guard the model class BEFORE the strict load so a class mismatch fails with an
        # actionable message instead of a cryptic state_dict key error. Validate against the
        # class actually being trained (``type(model)``), not the committed alias, so a v3 arm
        # accepts a v3 warm-start while still rejecting a foreign-class checkpoint.
        blob = torch.load(str(resume), map_location="cpu", weights_only=False)
        check_model_class(blob, type(model).__name__)
        if load_checkpoint_strict(model, blob) is None:
            raise RuntimeError(
                f"could not align resume checkpoint {resume!r} into "
                f"{type(model).__name__} (no matching module keys); refusing "
                f"to silently train from random initial weights."
            )

    # Beta schedule: the headline run uses ``loss.beta_schedule``; a ``--pilot`` run swaps in
    # ``train.pilot_beta_schedule`` (a short warm-up) so the zero-KL bottleneck can open inside
    # the few-epoch pilot budget -- a 20-epoch warm-up would keep beta pinned near ``start`` the
    # whole pilot and the S2-T05 bottleneck-health gate could never fire (S2-T05).
    beta_schedule = loss_settings.get("beta_schedule")
    if overrides.get("pilot"):
        pilot_sched = (config.get("train") or {}).get("pilot_beta_schedule")
        if isinstance(pilot_sched, dict):
            beta_schedule = dict(pilot_sched)
    # An explicit top-level ``beta_schedule`` override wins over both (``beta_select`` forces
    # it to ``None`` so its swept fixed beta is not overwritten each epoch -- S2-T01b).
    if "beta_schedule" in overrides:
        beta_schedule = overrides["beta_schedule"]
    # Record the schedule that was ACTUALLY used (the pilot swap / beta_select null included)
    # in the checkpoint's loss_settings, so a run's provenance never claims the headline ramp
    # when a short pilot ramp produced the weights.
    loss_settings = {**loss_settings, "beta_schedule": beta_schedule}

    pl_model = SyntheticSeqVaeLagAttnV2Pl(
        model,
        lr=effective_lr,
        lr_milestones=optim_cfg.get("lr_milestones"),
        lr_gamma=float(optim_cfg.get("lr_gamma", 0.1)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        kld_beta=loss_settings["beta"],
        lambda_full=loss_settings["lambda_full"],
        lambda_base=loss_settings["lambda_base"],
        likelihood=loss_settings["likelihood"],
        sigma_obs=loss_settings["sigma_obs"],
        free_bits=loss_settings["free_bits"],
        detach_baseline_in_full=loss_settings["detach_baseline_in_full"],
        lambda_lag=loss_settings["lambda_lag"],
        warmup_epochs=int(ddp_cfg["warmup_epochs"]) if do_scale else 0,
        beta_schedule=beta_schedule,
        loss_spike_skip=config.get("loss_spike_skip"),
        # Overrides win over config so callers (e.g. beta_select) can disable the
        # curriculum for a fixed-beta run without mutating the shared config.
        curriculum=overrides.get("curriculum", config.get("curriculum")),
    )

    batch_size = int(overrides.get("batch_size", optim_cfg["batch_size"]))
    dm = SyntheticTEDataModuleV2(config, batch_size=batch_size, benchmark=benchmark)
    dm.setup("fit")
    has_val = dm.val_dataloader() is not None

    epochs = int(overrides.get("epochs", optim_cfg["epochs"]))
    results_dir = _run_dir(config, benchmark, arm)
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer_v2(
        ddp_cfg=ddp_cfg,
        results_dir=results_dir,
        epochs=epochs,
        devices=devices,
        n_gpus=n_gpus,
        has_val=has_val,
        overrides=overrides,
        plotting_cfg=config.get("plotting") or {},
    )
    trainer.fit(pl_model, datamodule=dm)

    final_metrics = _callback_metrics_to_floats(trainer)
    result: Dict[str, Any] = {
        "checkpoint": None,
        "best": None,
        "metrics_csv": None,
        "figures": [],
        "metrics": final_metrics,
        "epochs": epochs,
        "n_stats": 0,
    }

    # Interactive training-curve HTML from the Lightning CSV log (rank-0 only). This
    # finalises the same self-contained ``training_curves.html`` the live
    # ``LossPlotHtmlCallback`` rewrites each epoch, and records its path in
    # ``result["figures"]``. Skipped on the ``skip_checkpoint`` path
    # (``beta_select``'s throwaway per-beta runs) so they do not overwrite the headline
    # run's figure. ``plot_loss_curves_html`` returns ``None`` if plotly is missing.
    if getattr(trainer, "is_global_zero", True):
        metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"
        result["metrics_csv"] = metrics_csv
        if metrics_csv.is_file() and not overrides.get("skip_checkpoint"):
            try:
                from .visualize_v2 import plot_loss_curves_html

                result["figures"] = plot_loss_curves_html(
                    metrics_csv, results_dir / "figures" / "training_curves"
                ) or []
            except Exception as exc:  # pragma: no cover - plotting is non-fatal
                logger.warning(
                    "[train_v2] loss-curve render failed (non-fatal): {}", exc
                )

    if overrides.get("skip_checkpoint"):
        return result

    # Post-fit latent-stats pass (non-fatal), then export the checkpoints.
    n_stats = _maybe_fit_latent_stats(
        pl_model.orig_model, dm.make_plain_train_loader(), pl_model.device
    )
    result["n_stats"] = n_stats

    if getattr(trainer, "is_global_zero", True):
        data_meta = dm.data_meta
        val_loss = final_metrics.get(
            _resolve_epoch_metric("val/total_loss"), float("nan")
        )
        final_path = results_dir / "final.ckpt"
        save_checkpoint_v2(
            final_path,
            model=pl_model.orig_model,
            model_kwargs=model_kwargs,
            config=config,
            data_meta=data_meta,
            epoch=int(trainer.current_epoch),
            val_loss=val_loss,
            loss_settings=loss_settings,
            latent_stats_fitted=n_stats > 0,
            train_metrics=final_metrics,
            arm=arm,
        )
        best_path = results_dir / "best.ckpt"
        _export_best(
            best_path,
            trainer=trainer,
            fallback_model=pl_model.orig_model,
            model_cfg=config["model"],
            model_kwargs=model_kwargs,
            config=config,
            data_meta=data_meta,
            epoch=int(trainer.current_epoch),
            val_loss=val_loss,
            loss_settings=loss_settings,
            latent_stats_fitted=n_stats > 0,
            has_val=has_val,
            arm=arm,
        )
        result["checkpoint"] = final_path
        result["best"] = best_path

    return result


def _export_best(
    best_path: Path,
    *,
    trainer: pl.Trainer,
    fallback_model: SeqVaeLagAttn,
    model_cfg: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    epoch: int,
    val_loss: float,
    loss_settings: Dict[str, Any],
    has_val: bool,
    latent_stats_fitted: bool,
    arm: Optional[str] = None,
) -> None:
    r"""Write ``best.ckpt`` from the Lightning best snapshot, else from the final model.

    When a validation split produced a :class:`ModelCheckpoint` best path, load it
    into a fresh :class:`SeqVaeLagAttn` (stripping the Lightning wrapper prefixes
    via :func:`load_checkpoint_strict`) and re-save it in the v2 format. Any failure
    falls back to saving the current in-memory model (i.e. ``best == final``).
    """
    best_src = ""
    ckpt_cb = getattr(trainer, "checkpoint_callback", None)
    if has_val and ckpt_cb is not None:
        best_src = str(getattr(ckpt_cb, "best_model_path", "") or "")

    if best_src and Path(best_src).is_file():
        try:
            fresh, _ = build_model(model_cfg, torch.device("cpu"))
            loaded = load_checkpoint_strict(fresh, best_src, map_location="cpu")
            if loaded is None:
                raise RuntimeError("load_checkpoint_strict returned None")
            save_checkpoint_v2(
                best_path, model=fresh, model_kwargs=model_kwargs, config=config,
                data_meta=data_meta, epoch=epoch, val_loss=val_loss,
                loss_settings=loss_settings, latent_stats_fitted=False, arm=arm,
            )
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "[train_v2] best-ckpt export from {} failed ({}); "
                "falling back to the final model.", best_src, exc
            )

    save_checkpoint_v2(
        best_path, model=fallback_model, model_kwargs=model_kwargs, config=config,
        data_meta=data_meta, epoch=epoch, val_loss=val_loss,
        loss_settings=loss_settings, latent_stats_fitted=latent_stats_fitted, arm=arm,
    )


# ============================================================================
# Optional beta-selection stage (S5-T03)
# ============================================================================
def beta_select(
    config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
    *,
    benchmark: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Short runs over a ``beta_grid`` to pick the least-collapsed KL weight (S5-T03).

    Skippable: when ``beta_select.enabled`` is false and it is not force-run
    (``overrides['force']``), this is a no-op that reports the fixed
    ``loss.kld_beta``. When run, it trains a short model (``beta_select.epochs``, no
    checkpoint export) for each $\beta$ in ``beta_select.beta_grid``, records the
    final ``kld_nats`` and ``total_loss``, and selects the $\beta$ with the highest
    ``kld_nats`` (least posterior collapse). Writes ``results/<tag>/beta_select.json``.

    Args:
        config: The parsed config tree.
        overrides: Optional overrides. ``force`` runs even when disabled;
            ``beta_grid`` / ``epochs`` override the config; the remaining
            :func:`train_v2` overrides (``limit_*_batches``, ``batch_size``,
            ``devices``) are forwarded to each short run.
        benchmark: Active benchmark key.

    Returns:
        ``{enabled, selected_beta, results, out_path}`` where ``results`` is a list
        of ``{beta, kld_nats, total_loss}`` (empty on the no-op path).
    """
    overrides = dict(overrides or {})
    benchmark = str(benchmark or config.get("experiment", {}).get("benchmark", "G1_raw"))
    arm = overrides.get("arm")  # arm-scope the sweep output; forwarded to each train_v2 run
    bs_cfg = dict(config.get("beta_select") or {})
    enabled = bool(bs_cfg.get("enabled", False)) or bool(overrides.pop("force", False))

    fixed_beta = float(config.get("loss", {}).get("kld_beta", 1e-3))
    if not enabled:
        logger.info(
            "[beta_select] disabled; using the fixed loss.kld_beta={}", fixed_beta
        )
        return {
            "enabled": False,
            "selected_beta": fixed_beta,
            "results": [],
            "out_path": None,
        }

    beta_grid = [
        float(b)
        for b in overrides.pop("beta_grid", bs_cfg.get("beta_grid", [fixed_beta]))
    ]
    epochs = int(overrides.pop("epochs", bs_cfg.get("epochs", 5)))

    results: List[Dict[str, float]] = []
    for beta in beta_grid:
        run_over = {
            **overrides,
            "epochs": epochs,
            "loss": {**(overrides.get("loss") or {}), "kld_beta": beta,
                     "beta_schedule": None},
            # The sweep needs the swept beta to stick; BOTH the per-epoch curriculum hook and
            # the beta_schedule would otherwise overwrite kld_beta every epoch, so disable the
            # curriculum AND force the schedule off (the top-level override wins over the pilot
            # schedule too) for these fixed-beta calibration runs (S2-T01b).
            "curriculum": {"enabled": False},
            "beta_schedule": None,
            "skip_checkpoint": True,
            "progress_bar": overrides.get("progress_bar", False),
        }
        out = train_v2(config, run_over, benchmark=benchmark)
        metrics = out["metrics"]
        kld_nats = metrics.get(
            _resolve_epoch_metric("train/kld_nats"),
            metrics.get(_resolve_epoch_metric("val/kld_nats"), float("nan")),
        )
        total_loss = metrics.get(
            _resolve_epoch_metric("val/total_loss"),
            metrics.get(_resolve_epoch_metric("train/total_loss"), float("nan")),
        )
        results.append(
            {"beta": beta, "kld_nats": float(kld_nats), "total_loss": float(total_loss)}
        )
        logger.info(
            "[beta_select] beta={:.3e}  kld_nats={:.4f}  total_loss={:.4f}",
            beta, kld_nats, total_loss,
        )

    # Least-collapsed = highest kld_nats among finite results (fallback: fixed beta).
    finite = [r for r in results if math.isfinite(r["kld_nats"])]
    selected = (
        max(finite, key=lambda r: r["kld_nats"])["beta"] if finite else fixed_beta
    )

    out_path = _run_dir(config, benchmark, arm) / "beta_select.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"selected_beta": selected, "epochs": epochs, "results": results},
            handle,
            indent=2,
        )
    logger.info("[beta_select] selected beta={:.3e} -> {}", selected, out_path)

    return {
        "enabled": True,
        "selected_beta": selected,
        "results": results,
        "out_path": out_path,
    }
