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
from typing import Dict, Optional, Sequence, Tuple

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

    ``loss_warmup_positions`` (``K_warm`` in §18.17.3 step E) excludes
    the first K positions of every GUID from the loss reduction.
    Position $n < K$ has access to only $n+1$ segments of history under
    the causal mask; with `min_samples_per_guid=3` this means positions
    0,1,2 are intrinsically near-unsolvable and otherwise pull the loss
    toward the class prior solution while soaking up gradient mass that
    should drive the model on the late, informative positions. The
    masked positions still produce predictions (used by the prefix
    sweep at eval); they just don't contribute gradient. Default 0
    (no skip) preserves prior behaviour.
    """

    lambda_3: float = 1.0
    lambda_2: float = 0.5
    gamma_vae: float = 0.0          # 0 in stage 1; 0.1 in stage 2 (set externally)
    lambda_sp: float = 0.0          # 0 in stage 1; 1e-4 in stage 2 (set externally)
    position_weight_alpha: float = 0.0  # 0 = uniform; 1.5 = recommended late-bias
    loss_warmup_positions: int = 0      # K_warm position skip (§18.17.3 E)


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
        loss_warmup_positions: int = 0,
    ) -> torch.Tensor:
        """Two-level (per-GUID, then batch) weighted mean.

        Args:
            per_step: ``(B, N)`` per-position scalar loss values.
            mask: ``(B, N)`` segment-validity mask.
            position_weight_alpha: Power on the rank ratio inside
                :meth:`_position_weights_from_mask`. ``0`` reproduces the
                prior uniform reduction; ``1.5`` up-weights late positions.
            loss_warmup_positions: Skip the first ``K`` positions of every
                row from the loss reduction. ``0`` keeps every valid
                position; ``3`` is the recommended default (matches
                ``min_samples_per_guid``). Padded positions are skipped
                regardless.

        Returns:
            Scalar tensor — first weighted-average over valid positions
            per row, then average across rows. Rows with zero valid
            positions (e.g. all-padded, or all-skipped by ``K_warm``)
            contribute zero loss and are excluded from the batch
            denominator.
        """
        if loss_warmup_positions > 0:
            # Rank-based skip: drop the first ``K_warm`` *valid* positions
            # of every row. ``cumsum`` over the bool mask gives a 1-based
            # rank at valid slots and a carried value at padded slots; we
            # AND with the original mask so padded slots stay False
            # regardless of the rank carry-over. This is robust to any
            # padding placement (right-padded, interior gaps, etc.).
            #
            # NOTE: a row whose *entire* valid run is shorter than
            # ``loss_warmup_positions`` ends up with zero kept positions
            # and contributes zero loss — it is excluded from the batch
            # denominator below. The recommended config pairs
            # ``min_samples_per_guid > loss_warmup_positions`` so this
            # never happens; with ``min_samples_per_guid == K_warm`` (the
            # current YAML default 3 / 3), every minimally-qualifying
            # GUID contributes zero. This is the documented behaviour
            # (§18.17.3 E) but worth being aware of when tuning the two
            # thresholds together.
            rank_one_based = mask.long().cumsum(dim=-1)        # (B, N)
            warm_keep = (rank_one_based > int(loss_warmup_positions)) & mask
            mask = warm_keep
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
            loss_warmup_positions=int(self.weights.loss_warmup_positions),
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
            loss_warmup_positions=int(self.weights.loss_warmup_positions),
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


_VALID_CLASS_WEIGHT_MODES: Tuple[str, ...] = (
    "none",
    "inverse_frequency",
    "sqrt_inverse_frequency",
)


def _class_count_weights(
    counts: torch.Tensor, mode: str
) -> torch.Tensor:
    """Convert per-class counts into a length-K weight tensor under a mode.

    Args:
        counts: Length-K float32 tensor of per-class GUID counts.
        mode: One of ``{"none", "inverse_frequency",
            "sqrt_inverse_frequency"}``.

    Returns:
        Length-K float32 weights. ``"none"`` returns ones.
        ``"inverse_frequency"`` returns ``N_total / (K · N_c)`` per
        class (so weights sum to K and the *mean* is 1).
        ``"sqrt_inverse_frequency"`` returns the same expression with a
        square root applied — the recommended middle ground when
        inverse-frequency overshoots on bursty mini-batches (§18.17.3
        step B). Classes absent from the labels get a neutral weight of
        1.0 in every mode.
    """
    mode = str(mode)
    if mode not in _VALID_CLASS_WEIGHT_MODES:
        raise ValueError(
            f"class weight mode {mode!r} not in {_VALID_CLASS_WEIGHT_MODES}"
        )
    K = int(counts.numel())
    if mode == "none":
        return torch.ones(K, dtype=torch.float32)
    total = counts.sum().clamp_min(1.0)
    raw = total / (float(K) * counts.clamp_min(1.0))
    if mode == "sqrt_inverse_frequency":
        raw = raw.sqrt()
    raw = raw.to(torch.float32)
    raw[counts == 0] = 1.0
    return raw


def estimate_class_weights_3(
    labels_3: Sequence[int],
    mode: str = "none",
) -> torch.Tensor:
    """3-class GUID-level class weights under a configurable mode.

    Args:
        labels_3: Iterable of class ids in ``{0, 1, 2}``.
        mode: ``"none"`` (uniform — recommended after §18.17.3),
            ``"inverse_frequency"`` (legacy default), or
            ``"sqrt_inverse_frequency"`` (softer middle ground).

    Returns:
        Length-3 ``torch.float32`` tensor.
    """
    counts = torch.zeros(3, dtype=torch.float32)
    for c in labels_3:
        counts[int(c)] += 1.0
    return _class_count_weights(counts, mode)


def estimate_class_weights_bin(
    labels_bin: Sequence[int],
    mode: str = "none",
) -> torch.Tensor:
    """Binary GUID-level class weights under a configurable mode.

    Args:
        labels_bin: Iterable of class ids in ``{0, 1}``.
        mode: Same options as :func:`estimate_class_weights_3`.

    Returns:
        Length-2 ``torch.float32`` tensor ``[w_neg, w_pos]``.
    """
    counts = torch.zeros(2, dtype=torch.float32)
    for c in labels_bin:
        counts[int(c)] += 1.0
    return _class_count_weights(counts, mode)


def estimate_inverse_frequency_class_weights_3(
    labels_3: Sequence[int],
) -> torch.Tensor:
    """Back-compat shim — calls :func:`estimate_class_weights_3` with
    ``mode="inverse_frequency"``. Prefer the new function for new code.
    """
    return estimate_class_weights_3(labels_3, mode="inverse_frequency")


def estimate_inverse_frequency_class_weights_bin(
    labels_bin: Sequence[int],
) -> torch.Tensor:
    """Back-compat shim — calls :func:`estimate_class_weights_bin` with
    ``mode="inverse_frequency"``. Prefer the new function for new code.
    """
    return estimate_class_weights_bin(labels_bin, mode="inverse_frequency")


def class_priors_3(labels_3: Sequence[int]) -> torch.Tensor:
    """Empirical 3-class GUID-level prior ``p(y=c)`` in ``{0, 1, 2}``.

    Args:
        labels_3: Iterable of class ids.

    Returns:
        Length-3 ``torch.float32`` simplex (sums to 1). Empty input
        falls back to the uniform prior so callers can use it as a
        default when the train fold is empty.
    """
    counts = torch.zeros(3, dtype=torch.float32)
    for c in labels_3:
        counts[int(c)] += 1.0
    total = counts.sum()
    if total <= 0:
        return torch.full((3,), 1.0 / 3.0, dtype=torch.float32)
    return counts / total


def class_prior_bin(labels_bin: Sequence[int]) -> torch.Tensor:
    """Empirical positive-class prior for the binary head.

    Args:
        labels_bin: Iterable of binary labels in ``{0, 1}``.

    Returns:
        Scalar ``torch.float32`` tensor in ``(0, 1)``. Returns 0.5 when
        ``labels_bin`` is empty so callers have a safe default.
    """
    counts = torch.zeros(2, dtype=torch.float32)
    for c in labels_bin:
        counts[int(c)] += 1.0
    total = counts.sum()
    if total <= 0:
        return torch.tensor(0.5, dtype=torch.float32)
    return (counts[1] / total).to(torch.float32)


__all__ = [
    "GuidClassifierLoss",
    "LossWeights",
    "class_prior_bin",
    "class_priors_3",
    "estimate_class_weights_3",
    "estimate_class_weights_bin",
    "estimate_inverse_frequency_class_weights_3",
    "estimate_inverse_frequency_class_weights_bin",
]
