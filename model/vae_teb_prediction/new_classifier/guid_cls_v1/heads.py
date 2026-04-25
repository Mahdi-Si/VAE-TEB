"""Output heads for ``guid_cls_v1`` (PRD §7.2-§7.3).

Two heads sit on top of the temporal-transformer output:

* :class:`GuidOutcomeHead` — primary head. Pools (attentive + last-valid) the
  per-segment context, concatenates 6-d global GUID stats, projects, and
  emits 3-class + binary logits.
* :class:`SegmentAuxHead` — shared light MLP applied to *every* segment
  context vector. Produces auxiliary 3-class + binary logits used by the
  prefix-eval pipeline and the auxiliary loss term (PRD §8.1).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MaskedAttentivePoolOverN(nn.Module):
    """Additive attentive pool across the GUID's segment axis.

    Args:
        d_model: Hidden width of the input segment context.
        hidden_dim: Bottleneck width inside the score MLP.
    """

    def __init__(self, d_model: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, h: torch.Tensor, segment_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool across N.

        Args:
            h: ``(B, N, d_model)``.
            segment_mask: ``(B, N)`` bool — True for valid segments.

        Returns:
            Tuple ``(pooled (B, d_model), weights (B, N))``.
        """
        logits = self.score(torch.tanh(self.fc(h))).squeeze(-1)  # (B, N)
        # -inf on padded positions is safe because every batch row has at
        # least one valid segment (enforced by min_samples_per_guid).
        logits = logits.masked_fill(~segment_mask, float("-inf"))
        weights = F.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=-2)
        return pooled, weights


def _gather_last_valid(h: torch.Tensor, segment_mask: torch.Tensor) -> torch.Tensor:
    """Pick the last-valid segment context per row.

    Args:
        h: ``(B, N, d_model)`` segment contexts.
        segment_mask: ``(B, N)`` bool — True for valid positions.

    Returns:
        ``(B, d_model)`` — the row-wise context at index ``argmax(j: mask[b, j])``.
    """
    # Use the position of the last True in segment_mask. Force at least 0.
    pos = segment_mask.float().cumsum(dim=-1) * segment_mask.float()
    last_idx = pos.argmax(dim=-1).clamp(min=0)                 # (B,)
    batch_idx = torch.arange(h.shape[0], device=h.device)
    return h[batch_idx, last_idx]                              # (B, d_model)


def build_guid_global_stats(
    *,
    num_segments: torch.Tensor,
    iota_sso: torch.Tensor,
    segment_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the 2-d ``g_glob`` vector per GUID (PRD §7.3 / desc §8.5).

    Components (per row):
        ``[log(1+N), mean ι_sso]``

    where the ``mean`` reduction is taken over **valid** segments only.

    Excluded by design:
        - Signal-quality summaries (mean of ``hat_w``, fraction of valid
          steps): describe sensor validity, not physiology.
        - Cumulative monitoring time ``ψ(c_N)``, mean Δt ``ψ(mean Δt)``,
          and max gap ``log(1+max κ)``: all derive from the *spans* of
          observed segments, which are biased by the dataset's quality
          filter — a noisier patient has more early segments rejected, so
          their first surviving ``epoch[0]`` is later, shrinking every
          downstream cumulative/span statistic. We refuse to feed that
          quality-correlated bias into the head.

    The pairwise ``Δt`` is still consumed by the relative-time attention
    bias inside the transformer (it's structural, not a feature). And
    ``hat_w`` is still consumed inside the segment tokenizer purely as a
    masking signal.

    Args:
        num_segments: ``(B,)`` long.
        iota_sso: ``(B, N)`` per-segment "in second stage" indicator.
        segment_mask: ``(B, N)`` bool.

    Returns:
        ``(B, 2)`` global stats tensor.
    """
    mask_f = segment_mask.float()
    n_valid = mask_f.sum(dim=-1).clamp_min(1.0)

    log_n = torch.log1p(num_segments.float())
    mean_iota = (iota_sso * mask_f).sum(dim=-1) / n_valid

    return torch.stack(
        [log_n, mean_iota],
        dim=-1,
    )                                                              # (B, 2)


class GuidOutcomeHead(nn.Module):
    """Primary GUID-level head (3-class + binary).

    Args:
        d_model: Width of the temporal-transformer output (default 256).
        global_stats_dim: Dimensionality of ``g_glob`` (default 2 —
            ``[log(1+N), mean ι_sso]``).
        num_classes_multi: Multi-class output size (default 3).
        pool_hidden_dim: Bottleneck width of the attentive-pool score MLP.
        dropout: Dropout used in the post-pool MLP.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        global_stats_dim: int = 2,
        num_classes_multi: int = 3,
        pool_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes_multi = int(num_classes_multi)

        self.pool = _MaskedAttentivePoolOverN(d_model, hidden_dim=pool_hidden_dim)

        proj_in = 2 * d_model + global_stats_dim
        self.proj = nn.Sequential(
            nn.Linear(proj_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_3 = nn.Linear(d_model, num_classes_multi)
        self.head_bin = nn.Linear(d_model, 1)

        # Zero-init the binary head bias so initial prob_bin ≈ 0.5.
        nn.init.zeros_(self.head_bin.bias)

    def forward(
        self,
        h: torch.Tensor,
        segment_mask: torch.Tensor,
        g_glob: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute primary GUID logits.

        Args:
            h: ``(B, N, d_model)`` per-segment context vectors.
            segment_mask: ``(B, N)`` bool.
            g_glob: ``(B, global_stats_dim)`` global GUID statistics.

        Returns:
            Dict with:
              * ``logits_3``: ``(B, num_classes_multi)``
              * ``logit_bin``: ``(B,)``
              * ``prob_3``: ``(B, num_classes_multi)``
              * ``prob_bin``: ``(B,)``
              * ``segment_importance``: ``(B, N)`` attention weights over N
              * ``h_guid``: ``(B, d_model)`` final GUID representation
        """
        h_pool, weights = self.pool(h, segment_mask)
        h_last = _gather_last_valid(h, segment_mask)
        h_guid = self.proj(torch.cat([h_pool, h_last, g_glob], dim=-1))
        logits_3 = self.head_3(h_guid)
        logit_bin = self.head_bin(h_guid).squeeze(-1)
        return {
            "logits_3": logits_3,
            "logit_bin": logit_bin,
            "prob_3": F.softmax(logits_3, dim=-1),
            "prob_bin": torch.sigmoid(logit_bin),
            "segment_importance": weights,
            "h_guid": h_guid,
        }


class SegmentAuxHead(nn.Module):
    """Auxiliary per-segment head (shared MLP, tied across positions).

    Each segment's context vector ``h_{g, n}`` is independently pushed
    through ``Linear(d_model -> d_model/2) -> GELU -> Dropout`` then through
    parallel 3-class and binary linear heads. The binary head is zero-inited
    so the auxiliary BCE term starts near 0.5 prior, matching the description
    of "stage-1 starts with aux contribution near zero".

    Args:
        d_model: Width of the input segment context.
        hidden_dim: Bottleneck width (default ``d_model // 2``).
        num_classes_multi: 3-class output size.
        dropout: Dropout used in the bottleneck.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_classes_multi: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_3 = nn.Linear(hidden_dim, num_classes_multi)
        self.head_bin = nn.Linear(hidden_dim, 1)
        # Zero-init both heads → aux logits ~ 0 at step 0.
        nn.init.zeros_(self.head_3.weight)
        nn.init.zeros_(self.head_3.bias)
        nn.init.zeros_(self.head_bin.weight)
        nn.init.zeros_(self.head_bin.bias)

    def forward(
        self, h: torch.Tensor, segment_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Apply the shared auxiliary head.

        Args:
            h: ``(B, N, d_model)`` per-segment context.
            segment_mask: ``(B, N)`` bool — used only to zero padded outputs.

        Returns:
            Dict with:
              * ``aux_logits_3``: ``(B, N, num_classes_multi)``
              * ``aux_logit_bin``: ``(B, N)``
              * ``aux_prob_3``: ``(B, N, num_classes_multi)``
              * ``aux_prob_bin``: ``(B, N)``
        """
        z = self.body(h)
        logits_3 = self.head_3(z)
        logit_bin = self.head_bin(z).squeeze(-1)
        mask_f = segment_mask.to(h.dtype)
        logits_3 = logits_3 * mask_f.unsqueeze(-1)
        logit_bin = logit_bin * mask_f
        return {
            "aux_logits_3": logits_3,
            "aux_logit_bin": logit_bin,
            "aux_prob_3": F.softmax(logits_3, dim=-1),
            "aux_prob_bin": torch.sigmoid(logit_bin),
        }


__all__ = [
    "GuidOutcomeHead",
    "SegmentAuxHead",
    "build_guid_global_stats",
]
