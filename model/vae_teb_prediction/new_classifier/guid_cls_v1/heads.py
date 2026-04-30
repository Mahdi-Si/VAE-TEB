"""Output head for ``guid_cls_v1`` (causal autoregressive design).

Under the causal design, the temporal transformer's output at position
``n`` already encodes "history up to position ``n``". A single shared head
is therefore applied **independently at every position** and produces
``(B, N, *)`` per-position 3-class and binary logits — the per-position
output at position ``n`` is the model's GUID-level prediction *given the
prefix ``1..n``*.

This replaces the earlier two-head design (``GuidOutcomeHead`` doing
attentive-pool + last-valid + global-stats fusion, plus a tied
``SegmentAuxHead`` applied per position). The pool / last / global-stats
path was tied to the "single GUID-level prediction" framing of the
stochastic-prefix-sampling era, and the auxiliary head was a workaround
for sparse prefix supervision; both are obsolete once every position
contributes a per-position loss in one forward pass.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerPositionOutcomeHead(nn.Module):
    """Per-position GUID outcome head (3-class + binary).

    Applied independently at every segment position. Combined with the
    causal mask in :class:`RelativeTimeMultiHeadSelfAttention`, this gives
    a genuine "predict GUID outcome given history up to here" output at
    every observed segment, in a single forward pass per GUID.

    Position 0 has access only to itself under the causal mask, so its
    prediction necessarily reflects only the most-recent segment's signal
    plus the model prior. Low skill at very early positions is expected
    and clinically acceptable.

    Args:
        d_model: Width of the temporal-transformer output (default 256).
        num_classes_multi: Multi-class output size (default 3 — H/A/HIE).
        hidden_dim: Width of the internal MLP. Defaults to ``d_model``
            when ``None`` is passed.
        dropout: Dropout used in the head MLP.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        num_classes_multi: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes_multi = int(num_classes_multi)
        hidden = int(hidden_dim) if hidden_dim is not None else self.d_model

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_3 = nn.Linear(d_model, num_classes_multi)
        self.head_bin = nn.Linear(d_model, 1)

        # Zero-init binary head so initial prob_bin ≈ 0.5 (smoke-test invariant).
        nn.init.zeros_(self.head_bin.weight)
        nn.init.zeros_(self.head_bin.bias)

    def forward(
        self,
        h: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply the per-position head.

        Args:
            h: ``(B, N, d_model)`` per-segment context vectors from the
                causal temporal transformer. Position ``n`` carries
                "history-up-to-``n``" information.
            segment_mask: ``(B, N)`` bool — True for valid positions.
                Padded positions are zeroed in the output for safety.

        Returns:
            Dict with:
              * ``logits_3``  — ``(B, N, num_classes_multi)``
              * ``logit_bin`` — ``(B, N)``
              * ``prob_3``    — ``(B, N, num_classes_multi)``  softmax
              * ``prob_bin``  — ``(B, N)``                     sigmoid
        """
        z = self.proj(h)                                 # (B, N, d_model)
        logits_3 = self.head_3(z)                        # (B, N, C)
        logit_bin = self.head_bin(z).squeeze(-1)         # (B, N)

        mask_f = segment_mask.to(h.dtype)
        logits_3 = logits_3 * mask_f.unsqueeze(-1)
        logit_bin = logit_bin * mask_f
        prob_3 = F.softmax(logits_3, dim=-1) * mask_f.unsqueeze(-1)
        prob_bin = torch.sigmoid(logit_bin) * mask_f
        return {
            "logits_3": logits_3,
            "logit_bin": logit_bin,
            "prob_3": prob_3,
            "prob_bin": prob_bin,
        }


__all__ = [
    "PerPositionOutcomeHead",
]
