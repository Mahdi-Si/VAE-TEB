"""Combined loss for ``guid_cls_v1`` (causal autoregressive design).

Each forward pass produces per-position 3-class and binary logits of
shape ``(B, N, 3)`` and ``(B, N)``. The loss is the GUID-level label
broadcast to every visible position, then reduced with a **two-level
masked mean**: first over valid positions per GUID, then over the batch.
This restores per-GUID equal weighting (long GUIDs do not dominate the
gradient) while still putting a learning signal at every observable time
point.

Class weights (inverse frequency, computed at GUID level) apply
unchanged: each per-position loss term is weighted by its class prior.

Optional VAE-aux + L2 sparsity terms are still supported for the
unfrozen two-stage path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossWeights:
    """Per-term coefficients (causal-AR design).

    The auxiliary segment-head terms (``lambda_aux``, ``lambda_aux_bin``)
    are gone: the per-position GUID head provides the same signal at
    every visible position with full predictive capacity.

    ``position_weight_alpha`` controls how strongly later positions are
    up-weighted in the per-GUID reduction. Position ``t`` (0-indexed
    rank within valid entries) gets weight ``((rank + 1) / n_valid) ** α``,
    so the last valid position has weight 1.0 and earlier positions
    decay toward 0. ``α = 0`` recovers uniform weighting (the original
    behaviour); ``α = 1.5`` is the recommended default — late positions
    carry roughly 80% of the per-GUID loss for a typical N_g ≈ 10.
    Padded positions always receive zero weight.
    """

    lambda_3: float = 1.0
    lambda_2: float = 0.5
    gamma_vae: float = 0.0          # 0 in stage 1; 0.1 in stage 2 (set externally)
    lambda_sp: float = 0.0          # 0 in stage 1; 1e-4 in stage 2 (set externally)
    position_weight_alpha: float = 0.0  # 0 = uniform; 1.5 = recommended late-bias


class GuidClassifierLoss(nn.Module):
    """Per-position masked-mean loss with two-level reduction.

    Args:
        weights: Per-term loss weights.
        class_weights_3: Optional length-3 tensor of inverse-frequency
            weights for the multi-class head. Registered as a buffer so
            the tensor follows ``.to(device)`` automatically.
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

    # ------------------------------------------------------------------
    # Per-position loss reductions
    # ------------------------------------------------------------------

    @staticmethod
    def _position_weights_from_mask(
        mask: torch.Tensor,
        alpha: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build per-position weights with optional late-position bias.

        For ``alpha == 0`` returns ``mask`` cast to ``dtype`` — equivalent
        to uniform weighting and identical to the prior behaviour. For
        ``alpha > 0`` returns ``((rank + 1) / n_valid) ** alpha`` at each
        valid position, where ``rank`` is the 0-based count of valid
        positions seen so far in the row. Padded positions get zero
        weight.

        Args:
            mask: ``(B, N)`` segment-validity mask (bool or castable).
            alpha: Power on the rank ratio. 0 → uniform; 1.5 → recommended
                late-bias (Phase B of the diagnosis plan).
            dtype: Output dtype.

        Returns:
            ``(B, N)`` weight tensor, NOT normalised. The two-level mean
            divides by ``weights.sum(dim=-1)`` per row.
        """
        mask_f = mask.to(dtype)
        if alpha <= 0.0:
            return mask_f
        # 1-based rank within valid entries: cumsum on mask_f gives the
        # running count of valid steps. Padded positions (mask_f = 0)
        # would otherwise inherit the carried cumsum value, so multiply
        # by mask_f at the end to zero them.
        n_valid = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        rank = mask_f.cumsum(dim=-1)                          # (B, N)
        ratio = (rank / n_valid).clamp(min=0.0, max=1.0)
        weights = ratio.pow(float(alpha)) * mask_f
        return weights

    @staticmethod
    def _two_level_mean(
        per_step: torch.Tensor,
        mask: torch.Tensor,
        *,
        position_weight_alpha: float = 0.0,
    ) -> torch.Tensor:
        """Two-level (per-GUID, then batch) weighted mean.

        Args:
            per_step: ``(B, N)`` per-position scalar loss values.
            mask: ``(B, N)`` segment-validity mask.
            position_weight_alpha: Power on the rank ratio inside
                :meth:`_position_weights_from_mask`. ``0`` reproduces the
                prior uniform reduction; ``1.5`` up-weights late positions.

        Returns:
            Scalar tensor — first weighted-average over valid positions
            per row, then average across rows. Rows with zero valid
            positions contribute zero loss and are excluded from the
            batch denominator.
        """
        weights = GuidClassifierLoss._position_weights_from_mask(
            mask, position_weight_alpha, per_step.dtype
        )                                                     # (B, N)
        weight_sum = weights.sum(dim=-1)                      # (B,)
        per_guid = (per_step * weights).sum(dim=-1) / weight_sum.clamp_min(1.0)
        any_valid = (weight_sum > 0).to(per_step.dtype)       # (B,)
        denom = any_valid.sum().clamp_min(1.0)
        return (per_guid * any_valid).sum() / denom

    def _bin_weight_per_pos(self, target: torch.Tensor) -> torch.Tensor:
        """Per-position weight for binary cross-entropy.

        Args:
            target: ``(B, N)`` float tensor in ``{0., 1.}``.

        Returns:
            Tensor with the same shape carrying the corresponding
            ``[w_neg, w_pos]`` weight (or ones if no class weights set).
        """
        if self.class_weights_bin is None:
            return torch.ones_like(target)
        w_neg, w_pos = self.class_weights_bin[0], self.class_weights_bin[1]
        return torch.where(target > 0.5, w_pos, w_neg)

    def _ce_3_per_pos(
        self,
        logits_3: torch.Tensor,
        target_3: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position 3-class CE with two-level reduction.

        Args:
            logits_3: ``(B, N, 3)`` per-position logits.
            target_3: ``(B,)`` long GUID-level class id (broadcast to N).
            mask: ``(B, N)`` segment-validity mask.

        Returns:
            Scalar loss.
        """
        B, N, C = logits_3.shape
        target_per_pos = target_3.unsqueeze(1).expand(B, N).reshape(-1)        # (B*N,)
        flat_logits = logits_3.reshape(-1, C)
        per_step = F.cross_entropy(
            flat_logits, target_per_pos, weight=self.class_weights_3, reduction="none"
        ).reshape(B, N)
        return self._two_level_mean(
            per_step,
            mask,
            position_weight_alpha=float(self.weights.position_weight_alpha),
        )

    def _bce_per_pos(
        self,
        logit_bin: torch.Tensor,
        target_bin: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position binary CE with two-level reduction.

        Args:
            logit_bin: ``(B, N)`` per-position logits.
            target_bin: ``(B,)`` float in ``{0., 1.}`` (broadcast to N).
            mask: ``(B, N)`` segment-validity mask.

        Returns:
            Scalar loss.
        """
        B, N = logit_bin.shape
        target_per_pos = target_bin.unsqueeze(1).expand(B, N)                  # (B, N)
        per_step = F.binary_cross_entropy_with_logits(
            logit_bin, target_per_pos, reduction="none"
        )
        per_step = per_step * self._bin_weight_per_pos(target_per_pos)
        return self._two_level_mean(
            per_step,
            mask,
            position_weight_alpha=float(self.weights.position_weight_alpha),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        *,
        vae_loss: Optional[torch.Tensor] = None,
        sparsity_term: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined per-position loss + per-component breakdown.

        Args:
            outputs: Forward dict from :class:`GuidOutcomeClassifier`.
                Must contain ``logits_3 (B, N, 3)`` and ``logit_bin (B, N)``.
            batch: Collated batch dict (must contain ``label_3``,
                ``label_bin``, ``segment_mask``).
            vae_loss: Optional VAE-aux loss scalar (live-VAE path only).
            sparsity_term: Optional L2 sparsity scalar (stage 2 only).

        Returns:
            Dict with at minimum ``total_loss`` plus each component for
            logging.
        """
        target_3 = batch["label_3"].long()                  # (B,)
        target_bin = batch["label_bin"].float()             # (B,)
        seg_mask = batch["segment_mask"]                    # (B, N)

        ce_3 = self._ce_3_per_pos(outputs["logits_3"], target_3, seg_mask)
        bce_bin = self._bce_per_pos(outputs["logit_bin"], target_bin, seg_mask)

        w = self.weights
        total = w.lambda_3 * ce_3 + w.lambda_2 * bce_bin
        if vae_loss is not None and w.gamma_vae > 0.0:
            total = total + w.gamma_vae * vae_loss
        if sparsity_term is not None and w.lambda_sp > 0.0:
            total = total + w.lambda_sp * sparsity_term

        components: Dict[str, torch.Tensor] = {
            "total_loss": total,
            "ce_3": ce_3,
            "bce_bin": bce_bin,
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
