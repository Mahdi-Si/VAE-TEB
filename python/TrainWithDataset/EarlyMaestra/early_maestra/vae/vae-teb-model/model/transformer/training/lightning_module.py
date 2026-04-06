"""PyTorch Lightning wrapper for the Causal Multimodal Forecasting Transformer (v2).

Implements the 3-stage training schedule from model.md §19:
    - Stage 1: Deterministic warm start (L_fus + L_delta + L_delta2 + L_spectral + L_self only).
    - Stage 2: Activate TE residual head (add L_te, beta_self ramps, beta_transfer=0).
    - Stage 3: Full KL warmup (both beta_self and beta_transfer ramp to their maxes).
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from loguru import logger
from torch import Tensor

from model.transformer.model import (
    CausalMultimodalTransformer,
    CausalTransformerLoss,
    TransformerConfig,
    sample_anchors,
)
from train.pl_model_base import LightningModelBase


class PlCausalTransformer(LightningModelBase):
    """Lightning module for self-supervised pretraining of the causal transformer (v2).

    Wraps ``CausalMultimodalTransformer`` and ``CausalTransformerLoss``, handles
    anchor sampling, 3-stage training schedule, and per-step dual KL beta warmup.

    v2 changes:
        - Dual KL scheduling: ``beta_self`` for z_self and ``beta_transfer`` for z_transfer.
        - ``beta_self`` starts ramping in Stage 2 (z_self benefits from early regularization).
        - ``beta_transfer`` ramps only in Stage 3 (same as v1's single beta).
        - Logs additional metrics: L_delta2, L_spectral, L_kl_self, L_kl_transfer.

    Args:
        base_model: The raw ``CausalMultimodalTransformer`` module.
        loss_fn: The ``CausalTransformerLoss`` module (shares config with model).
        transformer_config: The ``TransformerConfig`` instance shared by model
            and loss.  Its ``lambda_te`` field is mutated at stage boundaries.
        lr: Learning rate for AdamW.
        lr_milestones: Epoch milestones for MultiStepLR decay.
        weight_decay: AdamW weight decay.
        stage1_epochs: Number of epochs for Stage 1 (deterministic warm start).
        stage2_epochs: Number of epochs for Stage 2 (TE head active, beta_self ramps).
        stage3_epochs: Number of epochs for Stage 3 (both betas ramp).
        beta_max_self: Maximum KL weight for z_self.
        beta_max_transfer: Maximum KL weight for z_transfer.
        warmup_steps: Number of training steps over which betas ramp.
    """

    prog_bar_metrics = ("total_loss", "L_fus", "beta_transfer")

    def __init__(
        self,
        base_model: CausalMultimodalTransformer,
        loss_fn: CausalTransformerLoss,
        transformer_config: TransformerConfig,
        *,
        lr: float = 8e-5,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        stage1_epochs: int = 60,
        stage2_epochs: int = 60,
        stage3_epochs: int = 180,
        beta_max_self: float = 5e-4,
        beta_max_transfer: float = 2e-4,
        warmup_steps: int = 8000,
        # Legacy compat: accept beta_max and map to beta_max_transfer
        beta_max: Optional[float] = None,
    ) -> None:
        super().__init__(
            base_model,
            lr=lr,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            module_name="PlCausalTransformer",
        )
        self.loss_fn = loss_fn
        self._transformer_config = transformer_config
        self._original_lambda_te = transformer_config.lambda_te

        # Handle legacy beta_max -> beta_max_transfer
        if beta_max is not None and beta_max_transfer == 2e-4:
            beta_max_transfer = beta_max

        # Save schedule hparams (accessible via self.hparams.*)
        self.save_hyperparameters(
            ignore=["base_model", "loss_fn", "transformer_config", "beta_max"],
        )

        # Stage tracking state
        self._current_stage: int = 1
        self._current_beta_self: float = 0.0
        self._current_beta_transfer: float = 0.0
        self._stage2_global_step: Optional[int] = None
        self._stage3_global_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Core training logic
    # ------------------------------------------------------------------

    def compute_loss_and_metrics(
        self,
        batch,
        batch_idx: int,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """Forward pass, loss computation, and metric assembly.

        Args:
            batch: A batch ``AttributeDict`` from ``CombinedHDF5Dataset``
                containing at least ``fhr_st`` and ``up_st``.
            batch_idx: Index of the current batch.
            stage: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            Tuple of ``(loss, metrics_dict)`` where ``loss`` is the scalar to
            backpropagate and ``metrics_dict`` maps metric names to values.
        """
        Y = batch["fhr_st"]
        U = batch["up_st"]

        anchors = sample_anchors(
            Y, U, self._transformer_config, training=(stage == "train"),
        )
        outputs = self.model(Y, U, anchor_indices=anchors)
        losses = self.loss_fn(outputs, Y)

        # Apply dual KL weights
        total = (
            losses["total_loss"]
            + self._current_beta_self * losses["L_kl_self"]
            + self._current_beta_transfer * losses["L_kl_transfer"]
        )

        metrics = {
            "total_loss": total,
            "L_fus": losses["L_fus"],
            "L_delta": losses["L_delta"],
            "L_delta2": losses["L_delta2"],
            "L_spectral": losses["L_spectral"],
            "L_self": losses["L_self"],
            "L_te": losses["L_te"],
            "L_kl_self": losses["L_kl_self"],
            "L_kl_transfer": losses["L_kl_transfer"],
            "beta_self": self._current_beta_self,
            "beta_transfer": self._current_beta_transfer,
            "stage": float(self._current_stage),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Hyperparameter override on resume
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Sync optimizer LR with hparams after checkpoint restore.

        When resuming, ``apply_config_hyperparameters`` updates
        ``self.hparams.lr`` from the current config, but the optimizer's
        param groups still hold the checkpoint's LR.  This hook overwrites
        the optimizer LR so the config value actually takes effect.
        """
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]
        if optimizer is None:
            return
        target_lr = self.hparams.get("lr")
        if target_lr is None:
            return
        for pg in optimizer.param_groups:
            if pg["lr"] != target_lr:
                logger.info(
                    f"Overriding optimizer LR: {pg['lr']:.2e} -> {target_lr:.2e}"
                )
                pg["lr"] = target_lr

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def _on_train_epoch_start_hook(self) -> None:
        """Transition between training stages at epoch boundaries.

        Stage 1 (epoch < stage1_epochs):
            lambda_te = 0, beta_self = 0, beta_transfer = 0.
            Only fused + dynamics + self-only losses.

        Stage 2 (epoch < stage1 + stage2):
            lambda_te restored, beta_self ramps, beta_transfer = 0.
            TE head active, z_self gets early KL regularization.

        Stage 3 (remaining epochs):
            lambda_te restored, both betas ramp to their maxes.
        """
        epoch = self.current_epoch
        s1 = self.hparams.stage1_epochs
        s2 = s1 + self.hparams.stage2_epochs

        if epoch < s1:
            new_stage = 1
            self._transformer_config.lambda_te = 0.0
            self._current_beta_self = 0.0
            self._current_beta_transfer = 0.0
        elif epoch < s2:
            new_stage = 2
            self._transformer_config.lambda_te = self._original_lambda_te
            self._current_beta_transfer = 0.0
            if self._stage2_global_step is None:
                self._stage2_global_step = self.global_step
        else:
            new_stage = 3
            self._transformer_config.lambda_te = self._original_lambda_te
            if self._stage3_global_step is None:
                self._stage3_global_step = self.global_step

        if new_stage != self._current_stage:
            logger.info(
                f"Stage transition: {self._current_stage} -> {new_stage} "
                f"at epoch {epoch}"
            )
        self._current_stage = new_stage

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        """Update KL betas with step-level granularity.

        Stage 2: beta_self ramps, beta_transfer stays 0.
        Stage 3: both beta_self and beta_transfer ramp.

        Args:
            batch: Current training batch (unused).
            batch_idx: Index of the current batch (unused).
        """
        warmup = max(self.hparams.warmup_steps, 1)

        if self._current_stage == 2 and self._stage2_global_step is not None:
            steps_in_stage2 = self.global_step - self._stage2_global_step
            self._current_beta_self = self.hparams.beta_max_self * min(
                1.0, steps_in_stage2 / warmup
            )

        if self._current_stage == 3 and self._stage3_global_step is not None:
            steps_in_stage3 = self.global_step - self._stage3_global_step
            ramp = min(1.0, steps_in_stage3 / warmup)
            self._current_beta_self = self.hparams.beta_max_self * ramp
            self._current_beta_transfer = self.hparams.beta_max_transfer * ramp

    # ------------------------------------------------------------------
    # Checkpoint state (survives resume)
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist stage-tracking state into the Lightning checkpoint.

        Args:
            checkpoint: The checkpoint dict being saved.
        """
        checkpoint["stage_state"] = {
            "current_stage": self._current_stage,
            "current_beta_self": self._current_beta_self,
            "current_beta_transfer": self._current_beta_transfer,
            "stage2_global_step": self._stage2_global_step,
            "stage3_global_step": self._stage3_global_step,
            "original_lambda_te": self._original_lambda_te,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore stage-tracking state from a Lightning checkpoint.

        Args:
            checkpoint: The checkpoint dict being loaded.
        """
        state = checkpoint.get("stage_state")
        if state is None:
            logger.info("No stage_state in checkpoint; starting fresh.")
            return

        self._current_stage = state["current_stage"]
        self._original_lambda_te = state["original_lambda_te"]

        # Handle v1 checkpoint format (single beta)
        if "current_beta" in state:
            self._current_beta_transfer = state["current_beta"]
            self._current_beta_self = 0.0
            self._stage2_global_step = None
            self._stage3_global_step = state.get("stage3_global_step")
        else:
            self._current_beta_self = state["current_beta_self"]
            self._current_beta_transfer = state["current_beta_transfer"]
            self._stage2_global_step = state.get("stage2_global_step")
            self._stage3_global_step = state.get("stage3_global_step")

        # Apply the restored stage immediately
        if self._current_stage < 2:
            self._transformer_config.lambda_te = 0.0
        else:
            self._transformer_config.lambda_te = self._original_lambda_te
        logger.info(
            f"Restored stage state from checkpoint: stage={self._current_stage}, "
            f"beta_self={self._current_beta_self:.6f}, "
            f"beta_transfer={self._current_beta_transfer:.6f}, "
            f"stage2_global_step={self._stage2_global_step}, "
            f"stage3_global_step={self._stage3_global_step}"
        )
