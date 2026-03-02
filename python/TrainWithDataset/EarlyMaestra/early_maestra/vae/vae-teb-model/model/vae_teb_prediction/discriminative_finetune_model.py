"""Discriminative fine-tuning wrapper for the pretrained SeqVae.

This module defines ``DiscriminativeSeqVae``, a thin wrapper around an existing
``SeqVae`` instance that adds class-discriminative losses (center loss +
auxiliary classifier) **without modifying** the SeqVae class itself.

The wrapper supports two training phases:

* **Phase 1** — *Heads-only warmup*: all VAE parameters are frozen; only the
  auxiliary classifier head trains while center loss accumulates class centroids.
* **Phase 2** — *Encoder fine-tuning*: the decoder stays frozen (preserving
  the reconstruction mapping) while all three encoders (source, target,
  conditional) are unfrozen with a low learning rate.

See ``discriminative_training.md`` for full mathematical derivation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from model.vae_teb_prediction.discriminative_losses import (
    AuxiliaryClassifierHead,
    TemporalCenterLoss,
)
from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae


class DiscriminativeSeqVae(nn.Module):
    """SeqVae wrapper that adds class-discriminative training objectives.

    Composes a pretrained ``SeqVae`` with a ``TemporalCenterLoss`` and an
    ``AuxiliaryClassifierHead`` to produce an augmented loss function:

    .. math::

        \\mathcal{L} = \\alpha_r \\mathcal{L}_{\\text{NLL}}
                     + \\alpha_k \\beta \\mathcal{L}_{\\text{KLD}}
                     + \\alpha_c \\mathcal{L}_{\\text{center}}
                     + \\alpha_s \\mathcal{L}_{\\text{cls}}

    The wrapper **never** modifies the internals of ``SeqVae``; it calls
    ``SeqVae.forward()`` and ``SeqVae.compute_loss()`` as-is and then
    computes the additional discriminative terms on the returned latent ``z``.

    Attributes:
        vae_model: The underlying pretrained SeqVae.
        center_loss: TemporalCenterLoss module.
        classifier_head: AuxiliaryClassifierHead module.
    """

    def __init__(
        self,
        vae_model: SeqVae,
        *,
        num_classes: int = 3,
        classifier_hidden_dim: int = 32,
        center_ema_decay: float = 0.99,
        alpha_recon: float = 1.0,
        alpha_kld: float = 1.0,
        alpha_center: float = 0.1,
        alpha_cls: float = 0.5,
    ) -> None:
        """Initialize the discriminative wrapper.

        Args:
            vae_model: A pretrained ``SeqVae`` instance (weights already
                loaded).
            num_classes: Number of clinical outcome classes.
            classifier_hidden_dim: Hidden dimension for the auxiliary
                classifier MLP.
            center_ema_decay: EMA decay for centroid updates.
            alpha_recon: Weight for reconstruction (NLL) loss.
            alpha_kld: Weight for KL divergence loss.
            alpha_center: Weight for temporal center loss.
            alpha_cls: Weight for auxiliary classification loss.
        """
        super().__init__()
        self.vae_model = vae_model
        self.num_classes = num_classes

        # Loss weights
        self.alpha_recon = alpha_recon
        self.alpha_kld = alpha_kld
        self.alpha_center = alpha_center
        self.alpha_cls = alpha_cls

        latent_dim = vae_model.latent_dim_z

        # Discriminative modules
        self.center_loss = TemporalCenterLoss(
            num_classes=num_classes,
            latent_dim=latent_dim,
            ema_decay=center_ema_decay,
        )
        self.classifier_head = AuxiliaryClassifierHead(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
        )

    def freeze_for_phase(self, phase: int) -> None:
        """Apply parameter freezing strategy for the given training phase.

        Args:
            phase: Training phase number.

                * ``1`` — Freeze **all** VAE parameters.  Only the auxiliary
                  classifier head and center-loss buffers are active.
                * ``2`` — Freeze **decoder only**.  Unfreeze source_encoder,
                  target_encoder, and conditional_encoder for fine-tuning.

        Raises:
            ValueError: If *phase* is not 1 or 2.
        """
        if phase == 1:
            # Freeze entire VAE
            for param in self.vae_model.parameters():
                param.requires_grad = False
            logger.info(
                "Phase 1: ALL VAE parameters frozen. "
                "Training classifier head only."
            )
        elif phase == 2:
            # Freeze decoder only, unfreeze all encoders
            for param in self.vae_model.decoder.parameters():
                param.requires_grad = False
            for module in [
                self.vae_model.source_encoder,
                self.vae_model.target_encoder,
                self.vae_model.conditional_encoder,
            ]:
                for param in module.parameters():
                    param.requires_grad = True
            logger.info(
                "Phase 2: Decoder frozen. "
                "source_encoder, target_encoder, conditional_encoder unfrozen."
            )
        else:
            raise ValueError(f"Unknown training phase: {phase}. Must be 1 or 2.")

        # Classifier head is always trainable
        for param in self.classifier_head.parameters():
            param.requires_grad = True

        self._log_param_counts()

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        x_ph: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run full VAE forward pass plus auxiliary classifier.

        Args:
            y_st: Target scattering features, shape ``(B, T, 43)``.
            y_ph: Target phase harmonic features, shape ``(B, T, 44)``.
            x_ph: Source cross-phase features, shape ``(B, T, 137)``.

        Returns:
            Dictionary merging SeqVae forward outputs with classifier outputs:
                - All keys from ``SeqVae.forward()`` (z, mu_pr, logvar_pr,
                  mu_prior, logvar_prior, mu_post, logvar_post, warmup_mask).
                - ``cls_logits``: Shape ``(B, C)``.
                - ``cls_probs``: Shape ``(B, C)``.
                - ``cls_preds``: Shape ``(B,)``.
        """
        vae_outputs = self.vae_model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

        z = vae_outputs["z"]  # (B, T, D)
        warmup_mask = vae_outputs["warmup_mask"]  # (T,)

        cls_outputs = self.classifier_head(z, warmup_mask=warmup_mask)

        return {
            **vae_outputs,
            "cls_logits": cls_outputs["logits"],
            "cls_probs": cls_outputs["probs"],
            "cls_preds": cls_outputs["preds"],
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        y_raw: torch.Tensor,
        labels: torch.Tensor,
        *,
        beta: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined discriminative + reconstruction loss.

        Args:
            forward_outputs: Dictionary returned by ``self.forward()``.
            y_st: Target scattering features, shape ``(B, T, 43)``.
            y_ph: Target phase harmonic features, shape ``(B, T, 44)``.
            y_raw: Raw target signal, shape ``(B, R)`` or ``(B, R, 1)``.
            labels: Per-sample class labels, shape ``(B,)`` with values in
                ``{1, 2, 3}``.
            beta: KL divergence weight (same as pretrained model's beta).

        Returns:
            Dictionary with all loss components and auxiliary metrics:
                - ``total_loss``: Combined scalar loss for backpropagation.
                - ``nll_loss``: Reconstruction NLL.
                - ``kld_loss``: KL divergence (transfer entropy).
                - ``center_loss``: Temporal center loss.
                - ``cls_loss``: Auxiliary classification cross-entropy.
                - ``cls_accuracy``: Classification accuracy (detached).
                - ``kld_beta``: Current beta value (for logging).
        """
        # --- VAE reconstruction losses ---
        vae_loss_dict = self.vae_model.compute_loss(
            forward_outputs=forward_outputs,
            y_st=y_st,
            y_ph=y_ph,
            y_raw=y_raw,
            beta=beta,
        )
        nll_loss = vae_loss_dict["nll_loss"]
        kld_loss = vae_loss_dict["kld_loss"]

        # --- Center loss ---
        z = forward_outputs["z"]  # (B, T, D)
        warmup_mask = forward_outputs["warmup_mask"]  # (T,)
        center_loss_val = self.center_loss(z, labels, warmup_mask=warmup_mask)

        # --- Classification loss ---
        cls_logits = forward_outputs["cls_logits"]  # (B, C)
        # Map labels {1,2,3} -> {0,1,2} for cross-entropy
        cls_targets = (labels - 1).long()
        cls_loss = F.cross_entropy(cls_logits, cls_targets)

        # Classification accuracy (detached, for monitoring only)
        cls_preds = forward_outputs["cls_preds"]
        cls_accuracy = (cls_preds == cls_targets).float().mean()

        # --- Combined loss ---
        total_loss = (
            self.alpha_recon * nll_loss
            + self.alpha_kld * beta * kld_loss
            + self.alpha_center * center_loss_val
            + self.alpha_cls * cls_loss
        )

        return {
            "total_loss": total_loss,
            "nll_loss": nll_loss,
            "kld_loss": kld_loss,
            "center_loss": center_loss_val,
            "cls_loss": cls_loss,
            "cls_accuracy": cls_accuracy.detach(),
            "kld_beta": torch.tensor(beta, device=total_loss.device),
        }

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Return parameters from all three VAE encoders.

        Returns:
            List of parameters from source_encoder, target_encoder, and
            conditional_encoder (used for differential learning rate).
        """
        params = []
        for module in [
            self.vae_model.source_encoder,
            self.vae_model.target_encoder,
            self.vae_model.conditional_encoder,
        ]:
            params.extend(p for p in module.parameters() if p.requires_grad)
        return params

    def get_head_params(self) -> list[nn.Parameter]:
        """Return parameters from the auxiliary classifier head.

        Returns:
            List of trainable classifier head parameters.
        """
        return [p for p in self.classifier_head.parameters() if p.requires_grad]

    def get_center_distances(self) -> torch.Tensor:
        """Retrieve pairwise centroid distances for monitoring.

        Returns:
            Symmetric distance matrix of shape ``(num_classes, num_classes)``.
        """
        return self.center_loss.get_center_distances()

    def _log_param_counts(self) -> None:
        """Log trainable vs frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        logger.info(
            f"DiscriminativeSeqVae — Total: {total:,} | "
            f"Trainable: {trainable:,} ({100 * trainable / max(total, 1):.1f}%) | "
            f"Frozen: {frozen:,}"
        )
