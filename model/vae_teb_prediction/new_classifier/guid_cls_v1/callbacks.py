"""Lightning callbacks for ``guid_cls_v1`` (live-VAE / two-stage path).

The cached-VAE path uses only the standard callbacks bundled by
:func:`single_fold_trainer._build_callbacks` (model checkpoint, early
stopping, loss plot, hparams, metrics). The live-VAE path additionally
needs to flip ``requires_grad`` on a documented subset of VAE submodules
at the stage-1 → stage-2 boundary, snapshot ``θ^{(0)}`` for the L2
sparsity anchor, and refresh the optimizer so the new parameter group is
picked up.

See :mod:`classifier_description` §12.3 and §18.9 for the math; the
unfreezing scope (target/source adapters + encoders + prior + posterior
heads, lag-attention and decoder kept frozen) follows the doc verbatim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from loguru import logger

from model.vae_teb_prediction.new_classifier.guid_cls_v1.logging_utils import (
    append_jsonl,
)


# Submodules of ``SeqVaeLagAttnV1`` that get unfrozen at stage 2.
# Lag-attention (``lag_bank`` / ``lag_attn``) and the decoders
# (``baseline_decoder`` / ``residual_decoder``) intentionally stay
# frozen — see §12.3.
_DEFAULT_UNFREEZE_SUBMODULES: Tuple[str, ...] = (
    "target_adapter",
    "source_adapter",
    "target_encoder",
    "source_encoder",
    "prior_head",
    "posterior_head",
)


def _iter_unfreeze_params(
    vae: nn.Module,
    submodule_names: Iterable[str],
) -> List[Tuple[str, nn.Parameter]]:
    """Return ``(qualified_name, param)`` for every param in the unfreeze set.

    Args:
        vae: The :class:`SeqVaeLagAttnV1` instance owned by the classifier.
        submodule_names: Attribute names on ``vae`` whose parameters
            should be unfrozen at stage 2.

    Returns:
        A flat list of ``(name, parameter)`` pairs sorted by name so the
        order is deterministic across runs (important for the L2 anchor
        bookkeeping).
    """
    out: List[Tuple[str, nn.Parameter]] = []
    for sub_name in submodule_names:
        if not hasattr(vae, sub_name):
            logger.warning(
                f"TwoStageVaeUnfreeze: VAE has no submodule '{sub_name}'; "
                "skipping (this likely indicates a VAE-version mismatch)"
            )
            continue
        sub = getattr(vae, sub_name)
        for pname, param in sub.named_parameters():
            out.append((f"{sub_name}.{pname}", param))
    out.sort(key=lambda kv: kv[0])
    return out


class TwoStageVaeUnfreeze(L.Callback):
    """Stage-2 unfreeze callback for the live-VAE path.

    At ``on_train_start``:
      * snapshot the to-be-unfrozen VAE params as ``θ^{(0)}`` on the
        Lightning module (``pl_module._vae_theta0``);
      * keep ``requires_grad=False`` on every VAE param for stage 1 and
        zero out ``gamma_vae`` / ``lambda_sp`` so the auxiliary terms
        are inert.

    At ``on_train_epoch_start`` once ``trainer.current_epoch`` reaches
    ``stage1_epochs``:
      * set ``requires_grad=True`` on the documented subset;
      * restore ``gamma_vae`` / ``lambda_sp`` to their YAML values;
      * rebuild the optimizer via ``pl_module.build_optimizer`` (which
        already produces a two-group AdamW when the classifier owns a
        ``vae`` submodule with trainable parameters).

    Args:
        stage1_epochs: Number of frozen-VAE epochs before stage 2.
        gamma_vae_stage2: VAE-aux loss weight to restore at stage 2.
        lambda_sp_stage2: L2-anchor weight to restore at stage 2.
        unfreeze_submodules: Optional override of the unfreeze set.
        log_path: Optional path to a JSONL file. When provided, every
            stage transition (stage 1 begin, stage 2 fire,
            EarlyStopping reset) is also appended as a structured
            record so the per-fold log bundle has a machine-readable
            trail of when each transition happened. ``None`` (default)
            preserves the prior console-only behaviour.
    """

    def __init__(
        self,
        *,
        stage1_epochs: int,
        gamma_vae_stage2: float,
        lambda_sp_stage2: float,
        unfreeze_submodules: Iterable[str] = _DEFAULT_UNFREEZE_SUBMODULES,
        log_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.stage1_epochs = int(stage1_epochs)
        self.gamma_vae_stage2 = float(gamma_vae_stage2)
        self.lambda_sp_stage2 = float(lambda_sp_stage2)
        self.unfreeze_submodules: Tuple[str, ...] = tuple(unfreeze_submodules)
        self.log_path: Optional[Path] = (
            Path(log_path) if log_path is not None else None
        )
        self._fired = False

    def _record(self, event: str, **payload) -> None:
        """Append a stage-transition record to ``log_path`` when configured."""
        if self.log_path is None:
            return
        record = {
            "event": event,
            "iso_timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            append_jsonl(self.log_path, record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                f"TwoStageVaeUnfreeze: failed to append stage transition "
                f"record to {self.log_path}: {exc}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_vae(pl_module: L.LightningModule) -> nn.Module:
        """Return the VAE submodule attached to the classifier."""
        base = getattr(pl_module, "_orig_model", pl_module)
        vae = getattr(base, "vae", None)
        if vae is None:
            raise RuntimeError(
                "TwoStageVaeUnfreeze: classifier has no .vae attribute. "
                "The callback should only be registered on the live-VAE path."
            )
        return vae

    def _get_loss_weights(self, pl_module: L.LightningModule):
        """Return the mutable :class:`LossWeights` instance on the wrapper."""
        loss_weights = getattr(pl_module, "loss_weights", None)
        if loss_weights is None:
            raise RuntimeError(
                "TwoStageVaeUnfreeze: pl_module has no .loss_weights"
            )
        return loss_weights

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        vae = self._get_vae(pl_module)

        # Stage-1 invariants: every VAE param frozen, aux losses inert.
        for p in vae.parameters():
            p.requires_grad_(False)
        loss_weights = self._get_loss_weights(pl_module)
        loss_weights.gamma_vae = 0.0
        loss_weights.lambda_sp = 0.0

        # Snapshot θ^{(0)} for the L2-to-init anchor used in stage 2.
        # Stored as a dict of detached clones so the values follow the
        # device the model lives on.
        unfreeze_params = _iter_unfreeze_params(vae, self.unfreeze_submodules)
        theta0: Dict[str, torch.Tensor] = {
            name: p.detach().clone() for name, p in unfreeze_params
        }
        # Preserve the unfreeze-set name list so the loss step can iterate
        # parameters by name without re-scanning the VAE.
        pl_module._vae_theta0 = theta0  # type: ignore[attr-defined]
        pl_module._vae_unfreeze_names = [n for n, _ in unfreeze_params]  # type: ignore[attr-defined]

        logger.info(
            f"TwoStageVaeUnfreeze: stage 1 begin (frozen VAE, {len(theta0)} "
            f"params snapshotted for stage-2 anchor; stage1_epochs={self.stage1_epochs})"
        )
        self._record(
            "stage1_begin",
            epoch=int(trainer.current_epoch),
            n_params_snapshotted=int(len(theta0)),
            stage1_epochs=int(self.stage1_epochs),
            unfreeze_submodules=list(self.unfreeze_submodules),
        )

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if self._fired:
            return
        if trainer.current_epoch < self.stage1_epochs:
            return

        vae = self._get_vae(pl_module)
        unfreeze_params = _iter_unfreeze_params(vae, self.unfreeze_submodules)
        for _, p in unfreeze_params:
            p.requires_grad_(True)

        loss_weights = self._get_loss_weights(pl_module)
        loss_weights.gamma_vae = self.gamma_vae_stage2
        loss_weights.lambda_sp = self.lambda_sp_stage2

        # Add the newly-trainable VAE params as a fresh low-LR param
        # group on the **existing** optimizer. Rebuilding from scratch
        # would leave the LR scheduler bound to the old optimizer object;
        # ``add_param_group`` keeps the scheduler relationship intact and
        # AdamW initialises moment buffers for the new params on first
        # step.
        if not trainer.optimizers:
            raise RuntimeError(
                "TwoStageVaeUnfreeze: trainer has no optimizer at stage 2"
            )
        optimizer = trainer.optimizers[0]
        vae_lr = float(getattr(pl_module, "vae_lr", 1e-5))
        new_params = [p for _, p in unfreeze_params]
        if new_params:
            optimizer.add_param_group(
                {
                    "params": new_params,
                    "lr": vae_lr,
                    "weight_decay": 0.0,
                }
            )

        # Reset any EarlyStopping callback so stage 2 starts with a fresh
        # patience window. Stage 1's val/total_loss is computed under a
        # frozen VAE and zeroed VAE-aux / sparsity terms; once stage 2
        # restores those weights and unfreezes the encoder, the loss
        # surface changes discontinuously and the stage-1 ``best_score``
        # is no longer a meaningful target. Without this reset, an
        # EarlyStopping that was already close to firing in stage 1
        # would terminate stage 2 within a handful of epochs because the
        # post-unfreeze val loss is compared against a stale baseline.
        # ``min_epochs = stage1_epochs + 1`` (set in
        # :func:`single_fold_trainer.train_fold`) guarantees the run
        # reaches this point at all.
        self._reset_early_stopping(trainer)

        self._fired = True
        logger.info(
            f"TwoStageVaeUnfreeze: stage 2 begin at epoch "
            f"{trainer.current_epoch} — unfroze {len(unfreeze_params)} params "
            f"across {len(self.unfreeze_submodules)} submodules; "
            f"new VAE param group lr={vae_lr}; "
            f"gamma_vae={self.gamma_vae_stage2}, "
            f"lambda_sp={self.lambda_sp_stage2}"
        )
        self._record(
            "stage2_begin",
            epoch=int(trainer.current_epoch),
            n_params_unfrozen=int(len(unfreeze_params)),
            n_submodules=int(len(self.unfreeze_submodules)),
            unfreeze_submodules=list(self.unfreeze_submodules),
            vae_lr=float(vae_lr),
            gamma_vae=float(self.gamma_vae_stage2),
            lambda_sp=float(self.lambda_sp_stage2),
        )

    @staticmethod
    def _reset_early_stopping(trainer: L.Trainer) -> None:
        """Clear ``best_score`` / ``wait_count`` on every EarlyStopping callback.

        Iterates over ``trainer.callbacks`` (multiple EarlyStopping
        instances are technically allowed, though we register at most
        one) and sets:

        * ``best_score`` to ``+inf`` for ``mode='min'`` callbacks (or
          ``-inf`` for ``mode='max'``), matching what Lightning does
          at trainer construction;
        * ``wait_count`` to 0 so the patience counter starts over;
        * ``stopped_epoch`` to 0 so a previously-armed terminator is
          re-armed cleanly.

        The reset is best-effort: if Lightning's internal attribute
        names change, we log and continue rather than crash the run.
        """
        from lightning.pytorch.callbacks import EarlyStopping  # noqa: WPS433

        for cb in trainer.callbacks:
            if not isinstance(cb, EarlyStopping):
                continue
            try:
                mode = getattr(cb, "mode", "min")
                # Match the dtype / device of the live ``current_score``
                # Lightning will compare against — under bf16-mixed or
                # fp16, a fp32 CPU ``inf`` would force a silent dtype
                # promotion every step.
                ref = trainer.callback_metrics.get(getattr(cb, "monitor", "")) \
                    if hasattr(trainer, "callback_metrics") else None
                ref_dtype = ref.dtype if isinstance(ref, torch.Tensor) else torch.float32
                ref_device = (
                    ref.device if isinstance(ref, torch.Tensor) else torch.device("cpu")
                )
                fill = float("inf") if mode == "min" else float("-inf")
                cb.best_score = torch.tensor(fill, dtype=ref_dtype, device=ref_device)
                cb.wait_count = 0
                cb.stopped_epoch = 0
                logger.info(
                    f"TwoStageVaeUnfreeze: reset EarlyStopping(monitor="
                    f"{getattr(cb, 'monitor', '?')!r}, mode={mode!r}) — "
                    f"stage 2 starts with a fresh patience window"
                )
            except (AttributeError, TypeError) as exc:  # pragma: no cover
                logger.warning(
                    f"TwoStageVaeUnfreeze: could not reset EarlyStopping "
                    f"(Lightning internal layout may have changed): {exc}"
                )


__all__ = [
    "TwoStageVaeUnfreeze",
    "_iter_unfreeze_params",
]
