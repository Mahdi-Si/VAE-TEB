"""PyTorch Lightning wrapper for the Causal Multimodal Forecasting Transformer.

Implements the 3-stage training schedule from model.md §19:
    - Stage 1: Deterministic warm start (L_fus + L_delta + L_self only).
    - Stage 2: Activate TE residual head (add L_te, beta still 0).
    - Stage 3: KL warmup (ramp beta from 0 to beta_max).
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
    """Lightning module for self-supervised pretraining of the causal transformer.

    Wraps ``CausalMultimodalTransformer`` and ``CausalTransformerLoss``, handles
    anchor sampling, 3-stage training schedule, and per-step KL beta warmup.

    Args:
        base_model: The raw ``CausalMultimodalTransformer`` module.
        loss_fn: The ``CausalTransformerLoss`` module (shares config with model).
        transformer_config: The ``TransformerConfig`` instance shared by model
            and loss.  Its ``lambda_te`` field is mutated at stage boundaries.
        lr: Learning rate for AdamW.
        lr_milestones: Epoch milestones for MultiStepLR decay.
        weight_decay: AdamW weight decay.
        stage1_epochs: Number of epochs for Stage 1 (deterministic warm start).
        stage2_epochs: Number of epochs for Stage 2 (TE head active, no KL).
        stage3_epochs: Number of epochs for Stage 3 (KL warmup).
        beta_max: Maximum KL weight reached at end of warmup.
        warmup_steps: Number of training steps over which beta ramps in Stage 3.
    """

    prog_bar_metrics = ("total_loss", "L_fus", "beta")

    def __init__(
        self,
        base_model: CausalMultimodalTransformer,
        loss_fn: CausalTransformerLoss,
        transformer_config: TransformerConfig,
        *,
        lr: float = 1e-4,
        lr_milestones: Optional[Iterable[int]] = None,
        weight_decay: float = 1e-4,
        stage1_epochs: int = 50,
        stage2_epochs: int = 50,
        stage3_epochs: int = 200,
        beta_max: float = 1e-4,
        warmup_steps: int = 5000,
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

        # Save schedule hparams (accessible via self.hparams.*)
        self.save_hyperparameters(
            ignore=["base_model", "loss_fn", "transformer_config"],
        )

        # Stage tracking state
        self._current_stage: int = 1
        self._current_beta: float = 0.0
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

        total = losses["total_loss"] + self._current_beta * losses["L_kl"]

        metrics = {
            "total_loss": total,
            "L_fus": losses["L_fus"],
            "L_delta": losses["L_delta"],
            "L_self": losses["L_self"],
            "L_te": losses["L_te"],
            "L_kl": losses["L_kl"],
            "beta": self._current_beta,
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
            lambda_te = 0, beta = 0.  Only fused + dynamics + self-only losses.
        Stage 2 (epoch < stage1 + stage2):
            lambda_te restored, beta = 0.  TE head active but no KL pressure.
        Stage 3 (remaining epochs):
            lambda_te restored, beta ramps via on_train_batch_start.
        """
        epoch = self.current_epoch
        s1 = self.hparams.stage1_epochs
        s2 = s1 + self.hparams.stage2_epochs

        if epoch < s1:
            new_stage = 1
            self._transformer_config.lambda_te = 0.0
            self._current_beta = 0.0
        elif epoch < s2:
            new_stage = 2
            self._transformer_config.lambda_te = self._original_lambda_te
            self._current_beta = 0.0
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
        """Update KL beta with step-level granularity during Stage 3.

        Args:
            batch: Current training batch (unused).
            batch_idx: Index of the current batch (unused).
        """
        if self._current_stage == 3 and self._stage3_global_step is not None:
            steps_in_stage3 = self.global_step - self._stage3_global_step
            self._current_beta = self.hparams.beta_max * min(
                1.0, steps_in_stage3 / max(self.hparams.warmup_steps, 1)
            )

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
            "current_beta": self._current_beta,
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
        self._current_beta = state["current_beta"]
        self._stage3_global_step = state["stage3_global_step"]
        self._original_lambda_te = state["original_lambda_te"]
        # Apply the restored stage immediately
        if self._current_stage < 2:
            self._transformer_config.lambda_te = 0.0
        else:
            self._transformer_config.lambda_te = self._original_lambda_te
        logger.info(
            f"Restored stage state from checkpoint: stage={self._current_stage}, "
            f"beta={self._current_beta:.6f}, "
            f"stage3_global_step={self._stage3_global_step}"
        )
