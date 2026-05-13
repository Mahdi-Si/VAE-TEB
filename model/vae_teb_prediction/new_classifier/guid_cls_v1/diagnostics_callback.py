"""Training diagnostics for ``guid_cls_v1``.

Adds a single Lightning :class:`Callback` that exposes the scalars
needed to debug bursty / class-imbalanced training without changing the
existing model surface:

* Global L2 gradient norm and per-parameter-group learning rate (so both
  the classifier and the live-VAE param groups stay visible after the
  stage-2 unfreeze).
* Global L2 weight norm at the end of each train epoch.
* The per-epoch validation summary computed by
  :meth:`PlGuidClassifier.on_validation_epoch_end` (per-class
  precision / recall / F1, confusion matrix, support, Brier, ECE,
  buffer sizes), persisted to ``epoch_summary.jsonl`` for later
  inspection.

Everything is opt-in via constructor flags; the trainer is responsible
for wiring those flags from the YAML ``logging.diagnostics_callback``
block. The callback does NOT cache any tensors longer than one event,
so memory overhead is negligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import lightning as L
import torch
from loguru import logger

from model.vae_teb_prediction.new_classifier.guid_cls_v1.logging_utils import (
    append_jsonl,
)


def _global_grad_norm(parameters) -> float:
    """Compute the global L2 grad norm across every parameter with a grad.

    Mirrors ``torch.nn.utils.clip_grad_norm_``'s pre-clip calculation so
    the reported value matches what ``gradient_clip_val`` is comparing
    against. Returns 0.0 when no parameter has a grad (e.g. before the
    first optimizer step or when every param is frozen).
    """
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        # ``.detach().norm(2)`` is the same expression Lightning uses
        # internally for ``grad_norm`` callbacks. Cast to float32 to
        # avoid bf16 / fp16 truncation when summing across many params.
        g = p.grad.detach()
        total_sq += float(g.float().pow(2).sum().item())
    return float(total_sq ** 0.5)


def _global_weight_norm(parameters) -> float:
    """Global L2 weight norm across all parameters that ``require_grad``."""
    total_sq = 0.0
    for p in parameters:
        if not p.requires_grad:
            continue
        total_sq += float(p.detach().float().pow(2).sum().item())
    return float(total_sq ** 0.5)


class TrainingDiagnosticsCallback(L.Callback):
    """Per-step / per-epoch diagnostics for ``PlGuidClassifier``.

    Attributes:
        log_dir: Directory the callback writes ``epoch_summary.jsonl``
            into (typically ``fold_{k}/logs/``).
        log_grad_norm: When True, ``train/grad_norm`` is computed in
            :meth:`on_before_optimizer_step` and logged via
            ``pl_module.log`` for the CSVLogger.
        log_weight_norm: When True, ``train/weight_norm`` is computed in
            :meth:`on_train_epoch_end`.
        log_param_group_lrs: When True, per-parameter-group learning
            rates are logged as ``train/lr_group{i}`` (so both the
            classifier and the optional live-VAE group are visible).
        log_per_class_metrics: When True, the structured validation
            summary populated by :meth:`PlGuidClassifier.on_validation_epoch_end`
            is appended as one row to ``epoch_summary.jsonl``.
        log_calibration: Pure documentation — Brier / ECE are logged
            from the module itself; the callback forwards them into
            ``epoch_summary.jsonl`` whenever the validation summary
            includes them.
        epoch_summary_filename: Filename within ``log_dir``. Defaults
            to ``epoch_summary.jsonl``.
    """

    def __init__(
        self,
        *,
        log_dir: Path,
        log_grad_norm: bool = True,
        log_weight_norm: bool = True,
        log_param_group_lrs: bool = True,
        log_per_class_metrics: bool = True,
        log_calibration: bool = True,
        epoch_summary_filename: str = "epoch_summary.jsonl",
    ) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_grad_norm = bool(log_grad_norm)
        self.log_weight_norm = bool(log_weight_norm)
        self.log_param_group_lrs = bool(log_param_group_lrs)
        self.log_per_class_metrics = bool(log_per_class_metrics)
        self.log_calibration = bool(log_calibration)
        self.epoch_summary_filename = str(epoch_summary_filename)
        # Cache the latest grad norm so ``on_train_epoch_end`` can
        # surface the max-over-the-epoch into ``epoch_summary.jsonl``
        # without needing a second backward pass.
        self._grad_norm_running_max: float = 0.0
        self._grad_norm_running_sum: float = 0.0
        self._grad_norm_running_count: int = 0

    @property
    def _summary_path(self) -> Path:
        return self.log_dir / self.epoch_summary_filename

    # ------------------------------------------------------------------
    # Per-step: grad norm + LR
    # ------------------------------------------------------------------

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        self._grad_norm_running_max = 0.0
        self._grad_norm_running_sum = 0.0
        self._grad_norm_running_count = 0

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.log_grad_norm:
            grad_norm = _global_grad_norm(pl_module.parameters())
            # ``on_step=True, on_epoch=False`` so each step's value is
            # recorded individually; the CSV logger handles aggregation
            # automatically when both are set elsewhere.
            pl_module.log(
                "train/grad_norm",
                grad_norm,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )
            if grad_norm > self._grad_norm_running_max:
                self._grad_norm_running_max = grad_norm
            self._grad_norm_running_sum += grad_norm
            self._grad_norm_running_count += 1
        if self.log_param_group_lrs:
            for i, group in enumerate(optimizer.param_groups):
                lr_val = group.get("lr")
                if lr_val is None:
                    continue
                pl_module.log(
                    f"train/lr_group{i}",
                    float(lr_val),
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

    # ------------------------------------------------------------------
    # Per train epoch: weight norm
    # ------------------------------------------------------------------

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self.log_weight_norm:
            weight_norm = _global_weight_norm(pl_module.parameters())
            pl_module.log(
                "train/weight_norm",
                weight_norm,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    # ------------------------------------------------------------------
    # Per val epoch: persist structured summary
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not (self.log_per_class_metrics or self.log_calibration):
            return
        # The module sets this attribute at the very end of
        # ``on_validation_epoch_end``. Lightning calls module hooks
        # before callback hooks, so the attribute is already populated
        # by the time this runs.
        last_summary = getattr(pl_module, "_last_val_summary", None)
        if not last_summary:
            return

        record: Dict[str, Any] = dict(last_summary)
        # Augment with the train-side stats so the JSONL row is a
        # complete per-epoch snapshot. Read directly from
        # ``trainer.callback_metrics`` (where every ``self.log`` value
        # is mirrored) so the values include the epoch aggregate, not
        # only the validation buffer.
        cm = trainer.callback_metrics or {}
        record.setdefault("epoch", int(getattr(trainer, "current_epoch", -1)))
        record.setdefault("global_step", int(getattr(trainer, "global_step", 0)))
        snapshot_keys = [
            "train/total_loss",
            "train/ce_3",
            "train/bce_bin",
            "train/acc_3",
            "train/acc_bin",
            "train/grad_norm",
            "train/weight_norm",
            "train/loss_positions_total",
            "train/loss_positions_class0",
            "train/loss_positions_class1",
            "train/loss_positions_class2",
            "val/total_loss",
            "val/ce_3",
            "val/bce_bin",
            "val/acc_3",
            "val/acc_bin",
            "val/macro_f1",
            "val/binary_auroc",
            "val/brier",
            "val/ece",
            "lr",
        ]
        # Add the per-class val keys dynamically so an arbitrary number
        # of classes is supported (currently 3).
        for c in range(3):
            snapshot_keys.append(f"val/precision_class{c}")
            snapshot_keys.append(f"val/recall_class{c}")
            snapshot_keys.append(f"val/f1_class{c}")
        # Add live-VAE group LRs whenever the optimizer has more than
        # one group (we don't know the count without peeking at the
        # optimizer; fetch generously and skip absent keys).
        for i in range(8):
            snapshot_keys.append(f"train/lr_group{i}")

        for key in snapshot_keys:
            val = cm.get(key)
            if val is None:
                continue
            record[_safe_key(key)] = _coerce(val)

        # Grad-norm summary stats for the epoch (the per-step values
        # are already in the CSV; this is a compact aggregation).
        if self._grad_norm_running_count > 0:
            record["grad_norm_max_epoch"] = float(self._grad_norm_running_max)
            record["grad_norm_mean_epoch"] = float(
                self._grad_norm_running_sum / float(self._grad_norm_running_count)
            )

        try:
            append_jsonl(self._summary_path, record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"TrainingDiagnosticsCallback: failed to append epoch summary "
                f"to {self._summary_path}: {exc}"
            )


def _safe_key(key: str) -> str:
    """Replace ``/`` with ``_`` so the JSONL is friendly to flat parsers."""
    return key.replace("/", "_")


def _coerce(value: Any) -> Any:
    """Convert a tensor / numpy scalar to a plain python scalar."""
    if isinstance(value, torch.Tensor):
        try:
            return float(value.detach().cpu().item())
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return float(value)
    except Exception:
        return value


__all__ = [
    "TrainingDiagnosticsCallback",
]
