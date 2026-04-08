"""PyTorch Lightning wrapper for the document-aligned causal transformer.

Implements the 3-stage training schedule from model.md §19:
    - Stage 1: deterministic warm start (no TE, no KL)
    - Stage 2: TE residual head active, KL kept near zero
    - Stage 3: TE KL warmup with a single beta
"""

from typing import Dict, Iterable, Optional, Tuple

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
    """Lightning module for self-supervised pretraining."""

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
        stage1_epochs: int = 60,
        stage2_epochs: int = 60,
        stage3_epochs: int = 180,
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

        self.save_hyperparameters(ignore=["base_model", "loss_fn", "transformer_config"])

        self._current_stage: int = 1
        self._current_beta: float = 0.0
        self._stage3_global_step: Optional[int] = None

    def compute_loss_and_metrics(
        self,
        batch,
        batch_idx: int,
        stage: str,
    ) -> Tuple[Tensor, Dict[str, object]]:
        Y = batch["fhr_st"]
        U = batch["up_st"]

        anchors = sample_anchors(Y, U, self._transformer_config, training=(stage == "train"))
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

    def on_train_start(self) -> None:
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

    def _on_train_epoch_start_hook(self) -> None:
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
        warmup = max(self.hparams.warmup_steps, 1)
        if self._current_stage == 3 and self._stage3_global_step is not None:
            steps_in_stage3 = self.global_step - self._stage3_global_step
            self._current_beta = self.hparams.beta_max * min(1.0, steps_in_stage3 / warmup)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["stage_state"] = {
            "current_stage": self._current_stage,
            "current_beta": self._current_beta,
            "stage3_global_step": self._stage3_global_step,
            "original_lambda_te": self._original_lambda_te,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state = checkpoint.get("stage_state")
        if state is None:
            logger.info("No stage_state in checkpoint; starting fresh.")
            return

        self._current_stage = state["current_stage"]
        self._original_lambda_te = state["original_lambda_te"]
        self._stage3_global_step = state.get("stage3_global_step")

        if "current_beta" in state:
            self._current_beta = state["current_beta"]
        else:
            self._current_beta = state.get("current_beta_transfer", 0.0)

        if self._current_stage < 2:
            self._transformer_config.lambda_te = 0.0
        else:
            self._transformer_config.lambda_te = self._original_lambda_te

        logger.info(
            f"Restored stage state from checkpoint: stage={self._current_stage}, "
            f"beta={self._current_beta:.6f}, "
            f"stage3_global_step={self._stage3_global_step}"
        )
