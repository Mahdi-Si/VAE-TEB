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

from typing import Dict, Iterable, List, Tuple

import lightning as L
import torch
import torch.nn as nn
from loguru import logger


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
    """

    def __init__(
        self,
        *,
        stage1_epochs: int,
        gamma_vae_stage2: float,
        lambda_sp_stage2: float,
        unfreeze_submodules: Iterable[str] = _DEFAULT_UNFREEZE_SUBMODULES,
    ) -> None:
        super().__init__()
        self.stage1_epochs = int(stage1_epochs)
        self.gamma_vae_stage2 = float(gamma_vae_stage2)
        self.lambda_sp_stage2 = float(lambda_sp_stage2)
        self.unfreeze_submodules: Tuple[str, ...] = tuple(unfreeze_submodules)
        self._fired = False

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

        self._fired = True
        logger.info(
            f"TwoStageVaeUnfreeze: stage 2 begin at epoch "
            f"{trainer.current_epoch} — unfroze {len(unfreeze_params)} params "
            f"across {len(self.unfreeze_submodules)} submodules; "
            f"new VAE param group lr={vae_lr}; "
            f"gamma_vae={self.gamma_vae_stage2}, "
            f"lambda_sp={self.lambda_sp_stage2}"
        )


__all__ = [
    "TwoStageVaeUnfreeze",
    "_iter_unfreeze_params",
]
