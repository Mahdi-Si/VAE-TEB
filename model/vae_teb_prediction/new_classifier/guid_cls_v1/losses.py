"""Combined loss for ``guid_cls_v1`` (PRD §8.1).

Implements the four classifier loss components plus optional VAE-aux + L2
sparsity terms used by the unfrozen two-stage path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Per-term coefficients (PRD §8.1)."""

    lambda_3: float = 1.0
    lambda_2: float = 0.5
    lambda_aux: float = 0.2
    lambda_aux_bin: float = 0.1
    gamma_vae: float = 0.0          # 0 in stage 1; 0.1 in stage 2 (set externally)
    lambda_sp: float = 0.0          # 0 in stage 1; 1e-4 in stage 2 (set externally)


class GuidClassifierLoss(nn.Module):
    """Combined loss with class weights and masked auxiliary heads.

    Args:
        weights: Per-term loss weights.
        class_weights_3: Optional length-3 tensor of inverse-frequency
            weights for the multi-class head. Registered as a buffer so the
            tensor follows ``.to(device)`` automatically.
        class_weights_bin: Optional length-2 tensor for the binary head;
            stored as ``[w_neg, w_pos]``.
    """

    def __init__(
        self,
        weights: LossWeights,
        *,
        class_weights_3: Optional[torch.Tensor] = None,
        class_weights_bin: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.weights = weights
        if class_weights_3 is not None:
            self.register_buffer("class_weights_3", class_weights_3.to(torch.float32))
        else:
            self.class_weights_3 = None  # type: ignore[assignment]
        if class_weights_bin is not None:
            self.register_buffer("class_weights_bin", class_weights_bin.to(torch.float32))
        else:
            self.class_weights_bin = None  # type: ignore[assignment]

    def _bin_weight_per_sample(self, target: torch.Tensor) -> torch.Tensor:
        """Per-sample weight for binary cross-entropy.

        Args:
            target: Float tensor in ``{0., 1.}`` of any shape.

        Returns:
            Tensor with the same shape as ``target`` carrying the corresponding
            ``[w_neg, w_pos]`` weight (or ones if no class weights set).
        """
        if self.class_weights_bin is None:
            return torch.ones_like(target)
        w_neg, w_pos = self.class_weights_bin[0], self.class_weights_bin[1]
        return torch.where(target > 0.5, w_pos, w_neg)

    def _ce_3_guid(
        self, logits_3: torch.Tensor, target_3: torch.Tensor
    ) -> torch.Tensor:
        """Per-GUID 3-class CE.

        Spec §10 wants ``L_3 = −(1/B) Σ_g Σ_c α_c y_{g,c} log p_{g,c}``
        (divide by batch size B). PyTorch's ``F.cross_entropy(weight=w,
        reduction='mean')`` instead divides by ``Σ_i w[y_i]``, which rescales
        gradients under class imbalance. Using ``reduction='none'`` here
        keeps the per-sample class weight (applied internally to the log
        term) and then ``.mean()`` normalises by B exactly, matching the
        spec.
        """
        per_sample = F.cross_entropy(
            logits_3, target_3, weight=self.class_weights_3, reduction="none"
        )
        return per_sample.mean()

    def _bce_guid(
        self, logit_bin: torch.Tensor, target_bin: torch.Tensor
    ) -> torch.Tensor:
        """Per-GUID binary CE with optional class weights."""
        per_sample = F.binary_cross_entropy_with_logits(
            logit_bin, target_bin, reduction="none"
        )
        w = self._bin_weight_per_sample(target_bin)
        return (per_sample * w).mean()

    def _ce_3_aux(
        self,
        aux_logits_3: torch.Tensor,
        target_3_per_seg: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary 3-class CE summed over valid segments and normalised."""
        if segment_mask.sum() == 0:
            return aux_logits_3.new_zeros(())
        flat_logits = aux_logits_3.reshape(-1, aux_logits_3.shape[-1])
        flat_target = target_3_per_seg.reshape(-1)
        flat_mask = segment_mask.reshape(-1)
        per_step = F.cross_entropy(
            flat_logits, flat_target, weight=self.class_weights_3, reduction="none"
        )
        return (per_step * flat_mask.to(per_step.dtype)).sum() / flat_mask.float().sum().clamp_min(1.0)

    def _bce_aux(
        self,
        aux_logit_bin: torch.Tensor,
        target_bin_per_seg: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary binary CE over valid segments."""
        if segment_mask.sum() == 0:
            return aux_logit_bin.new_zeros(())
        per_step = F.binary_cross_entropy_with_logits(
            aux_logit_bin, target_bin_per_seg, reduction="none"
        )
        w = self._bin_weight_per_sample(target_bin_per_seg)
        per_step = per_step * w
        flat_per = per_step.reshape(-1)
        flat_mask = segment_mask.reshape(-1).to(per_step.dtype)
        return (flat_per * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        vae_loss: Optional[torch.Tensor] = None,
        sparsity_term: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss + per-component breakdown.

        Args:
            outputs: Forward dict from :class:`GuidOutcomeClassifier`.
            batch: Collated batch dict (must contain ``label_3``, ``label_bin``,
                ``segment_mask``).
            vae_loss: Optional VAE-aux loss scalar (live-VAE path only).
            sparsity_term: Optional L2 sparsity scalar (stage 2 only).

        Returns:
            Dict with at minimum ``total_loss`` plus each component for
            logging.
        """
        target_3 = batch["label_3"].long()                  # (B,)
        target_bin = batch["label_bin"].float()             # (B,)
        seg_mask = batch["segment_mask"]                    # (B, N)

        ce_3 = self._ce_3_guid(outputs["logits_3"], target_3)
        bce_bin = self._bce_guid(outputs["logit_bin"], target_bin)

        # Broadcast GUID labels to per-segment for the auxiliary head.
        aux_target_3 = target_3.unsqueeze(1).expand(-1, seg_mask.shape[1])
        aux_target_bin = target_bin.unsqueeze(1).expand(-1, seg_mask.shape[1])
        aux_ce_3 = self._ce_3_aux(
            outputs["aux_logits_3"], aux_target_3, seg_mask
        )
        aux_bce_bin = self._bce_aux(
            outputs["aux_logit_bin"], aux_target_bin, seg_mask
        )

        w = self.weights
        total = (
            w.lambda_3 * ce_3
            + w.lambda_2 * bce_bin
            + w.lambda_aux * aux_ce_3
            + w.lambda_aux_bin * aux_bce_bin
        )
        if vae_loss is not None and w.gamma_vae > 0.0:
            total = total + w.gamma_vae * vae_loss
        if sparsity_term is not None and w.lambda_sp > 0.0:
            total = total + w.lambda_sp * sparsity_term

        components: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "ce_3": ce_3,
            "bce_bin": bce_bin,
            "aux_ce_3": aux_ce_3,
            "aux_bce_bin": aux_bce_bin,
        }
        if vae_loss is not None:
            components["vae_loss"] = vae_loss
        if sparsity_term is not None:
            components["sparsity"] = sparsity_term
        return components


def estimate_inverse_frequency_class_weights_3(
    labels_3: Sequence[int],
) -> torch.Tensor:
    """3-class inverse-frequency weights at the GUID level.

    Args:
        labels_3: Iterable of class ids in ``{0, 1, 2}``.

    Returns:
        Length-3 ``torch.float32`` tensor. Classes absent from ``labels_3``
        receive a neutral weight of 1.0 (rather than ``total / K``, which is
        what the naive ``total / (K * counts.clamp_min(1))`` formula would
        produce).
    """
    counts = torch.zeros(3, dtype=torch.float32)
    for c in labels_3:
        counts[int(c)] += 1.0
    total = counts.sum().clamp_min(1.0)
    weights = total / (3.0 * counts.clamp_min(1.0))
    weights[counts == 0] = 1.0
    return weights


def estimate_inverse_frequency_class_weights_bin(
    labels_bin: Sequence[int],
) -> torch.Tensor:
    """Binary inverse-frequency weights.

    Args:
        labels_bin: Iterable of class ids in ``{0, 1}``.

    Returns:
        Length-2 ``torch.float32`` tensor ``[w_neg, w_pos]``. Classes absent
        from ``labels_bin`` receive a neutral weight of 1.0.
    """
    counts = torch.zeros(2, dtype=torch.float32)
    for c in labels_bin:
        counts[int(c)] += 1.0
    total = counts.sum().clamp_min(1.0)
    weights = total / (2.0 * counts.clamp_min(1.0))
    weights[counts == 0] = 1.0
    return weights


__all__ = [
    "GuidClassifierLoss",
    "LossWeights",
    "estimate_inverse_frequency_class_weights_3",
    "estimate_inverse_frequency_class_weights_bin",
]
