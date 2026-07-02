r"""Lightning wrapper + training driver for ``SeqVaeLagAttnV1`` on ``synthetic_v2``.

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

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1  # noqa: E402
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
    r"""Lightning wrapper for :class:`SeqVaeLagAttnV1` on ``synthetic_v2`` TE data.

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
        warmup_epochs: int = 0,
        loss_spike_skip: Optional[Dict[str, Any]] = None,
        module_name: Optional[str] = None,
    ) -> None:
        r"""Initialize the wrapper while bypassing ``torch.compile``.

        Calls the Lightning grandparent ``__init__`` directly so the
        ``torch.compile(base_model)`` line in :class:`LightningModelBase` never
        runs, then saves every loss / schedule knob into ``self.hparams`` via an
        explicit dict (frame inspection is empty when the parent ``__init__`` is
        skipped).

        Args:
            base_model: The :class:`SeqVaeLagAttnV1` instance to wrap.
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
        )
        total_loss = loss_dict["total_loss"]
        pred_gap = loss_dict["base_loss"] - loss_dict["feat_loss"]

        metrics = {
            "total_loss": total_loss,
            "feat_loss": loss_dict["feat_loss"],
            "base_loss": loss_dict["base_loss"],
            "kld_loss": loss_dict["kld_loss"],
            # Dim-summed KL in nats/step ($\mathrm{KL} = \sum_d \mathrm{KL}_d$) --
            # the scale of the TE surrogate $\bar K$, vs the per-dim ``kld_loss``.
            "kld_nats": loss_dict["kld_loss"]
            * float(forward_outputs["mu_post"].shape[-1]),
            "pred_gap": pred_gap,
            "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
            "delta_mu_sat_frac": forward_outputs["delta_mu_sat_frac"],
            "mean_logvar_full": loss_dict.get("mean_logvar_full"),
            "mean_logvar_base": loss_dict.get("mean_logvar_base"),
            "kld_beta": float(hp["kld_beta"]),
        }
        return total_loss, metrics

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
            # No-op loss built from a parameter (not the possibly-NaN ``loss``) so
            # backward stays finite and DDP all-reduce keeps participating.
            anchor = next(p for p in self.parameters() if p.requires_grad)
            return anchor.sum() * 0.0

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
def build_model(
    model_cfg: Dict[str, Any], device: torch.device
) -> Tuple[SeqVaeLagAttnV1, Dict[str, Any]]:
    r"""Construct :class:`SeqVaeLagAttnV1` from the ``model`` config block.

    Args:
        model_cfg: The ``model`` block -- a 1:1 map of the keyword-only
            constructor arguments (``c_y=87`` / ``c_u=101`` for the v2 features).
        device: Device to move the model onto.

    Returns:
        ``(model, model_kwargs)`` where ``model_kwargs`` is the exact resolved
        constructor kwargs (stored verbatim in the checkpoint so downstream phases
        rebuild the architecture without the config).
    """
    kwargs = deepcopy(dict(model_cfg))
    # YAML gives ``logvar_clamp`` as a list; the constructor expects a tuple.
    clamp = kwargs.get("logvar_clamp")
    if clamp is not None:
        kwargs["logvar_clamp"] = (float(clamp[0]), float(clamp[1]))
    model = SeqVaeLagAttnV1(**kwargs)
    model.to(device)
    return model, kwargs


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
    return {
        "beta": beta,
        "lambda_full": lambda_full,
        "lambda_base": lambda_base,
        "likelihood": likelihood,
        "sigma_obs": sigma_obs,
        "free_bits": free_bits,
        "detach_baseline_in_full": detach_baseline_in_full,
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
    model: SeqVaeLagAttnV1, loader: Any, device: torch.device
) -> int:
    r"""Run :meth:`SeqVaeLagAttnV1.fit_latent_stats` over the training set.

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
    model: SeqVaeLagAttnV1,
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    epoch: int,
    val_loss: float,
    loss_settings: Dict[str, Any],
    latent_stats_fitted: bool,
    train_metrics: Optional[Dict[str, float]] = None,
) -> None:
    r"""Save a training checkpoint in the v1-compatible format.

    The bare (unprefixed) ``state_dict`` is stored under ``model_state_dict`` -- the
    key :func:`train.graph_models_utils.load_checkpoint_strict` scans for. A
    downstream phase rebuilds the model via ``SeqVaeLagAttnV1(**model_kwargs)`` then
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
    """
    ckpt = {
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


def _callback_metrics_to_floats(trainer: pl.Trainer) -> Dict[str, float]:
    r"""Extract the trainer's ``callback_metrics`` as a plain ``{name: float}`` dict."""
    out: Dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


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

    resume = overrides.get("resume_ckpt")
    if resume:
        logger.info("[train_v2] warm-starting from {}", resume)
        load_checkpoint_strict(model, str(resume), map_location="cpu")

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
        warmup_epochs=int(ddp_cfg["warmup_epochs"]) if do_scale else 0,
        loss_spike_skip=config.get("loss_spike_skip"),
    )

    batch_size = int(overrides.get("batch_size", optim_cfg["batch_size"]))
    dm = SyntheticTEDataModuleV2(config, batch_size=batch_size, benchmark=benchmark)
    dm.setup("fit")
    has_val = dm.val_dataloader() is not None

    epochs = int(overrides.get("epochs", optim_cfg["epochs"]))
    results_dir = _results_dir(config, benchmark)
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

    # Loss curves from the Lightning CSV log (rank-0 only). Skipped on the
    # ``skip_checkpoint`` path (``beta_select``'s throwaway per-beta runs) so they do
    # not overwrite the headline run's ``training_curves`` figure.
    if getattr(trainer, "is_global_zero", True):
        metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"
        result["metrics_csv"] = metrics_csv
        if metrics_csv.is_file() and not overrides.get("skip_checkpoint"):
            try:
                from .visualize_v2 import plot_loss_curves

                result["figures"] = plot_loss_curves(
                    metrics_csv, results_dir / "figures" / "training_curves"
                )
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
        )
        result["checkpoint"] = final_path
        result["best"] = best_path

    return result


def _export_best(
    best_path: Path,
    *,
    trainer: pl.Trainer,
    fallback_model: SeqVaeLagAttnV1,
    model_cfg: Dict[str, Any],
    model_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    data_meta: Dict[str, Any],
    epoch: int,
    val_loss: float,
    loss_settings: Dict[str, Any],
    latent_stats_fitted: bool,
    has_val: bool,
) -> None:
    r"""Write ``best.ckpt`` from the Lightning best snapshot, else from the final model.

    When a validation split produced a :class:`ModelCheckpoint` best path, load it
    into a fresh :class:`SeqVaeLagAttnV1` (stripping the Lightning wrapper prefixes
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
                loss_settings=loss_settings, latent_stats_fitted=False,
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
        loss_settings=loss_settings, latent_stats_fitted=latent_stats_fitted,
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
            "loss": {**(overrides.get("loss") or {}), "kld_beta": beta},
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

    out_path = _results_dir(config, benchmark) / "beta_select.json"
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
