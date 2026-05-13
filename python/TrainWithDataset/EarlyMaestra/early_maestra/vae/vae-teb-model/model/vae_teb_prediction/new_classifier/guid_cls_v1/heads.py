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

import math
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
        enable_three_class: bool = True,
        enable_binary: bool = True,
    ) -> None:
        super().__init__()
        if not (enable_three_class or enable_binary):
            raise ValueError(
                "PerPositionOutcomeHead: at least one of "
                "``enable_three_class`` / ``enable_binary`` must be True."
            )
        self.d_model = int(d_model)
        self.num_classes_multi = int(num_classes_multi)
        self.enable_three_class: bool = bool(enable_three_class)
        self.enable_binary: bool = bool(enable_binary)
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
        # Conditional head linears. Disabled heads are simply not
        # instantiated — they hold no parameters, never produce gradients,
        # and never appear in the forward output. Consumers must check
        # ``if "logits_3" in out`` / ``if "logit_bin" in out``.
        if self.enable_three_class:
            self.head_3 = nn.Linear(d_model, num_classes_multi)
        else:
            self.head_3 = None  # type: ignore[assignment]
        if self.enable_binary:
            self.head_bin = nn.Linear(d_model, 1)
        else:
            self.head_bin = None  # type: ignore[assignment]

        # Zero-init both head weight matrices so step-0 logits depend
        # ONLY on the head bias. Combined with
        # :meth:`init_class_bias_from_prior`, this puts the model
        # exactly at the empirical prior at step 0:
        #   * prob_bin = sigmoid(b_bin) = p^+
        #   * prob_3   = softmax(b_3)   = p_3
        # Without zero-initing ``head_3.weight``, the Kaiming-init
        # ``W·z`` term adds a non-trivial random offset to the 3-class
        # logits and the docstring's "model already predicts the prior"
        # guarantee silently fails for the 3-class branch.
        if self.head_3 is not None:
            nn.init.zeros_(self.head_3.weight)
            nn.init.zeros_(self.head_3.bias)
        if self.head_bin is not None:
            nn.init.zeros_(self.head_bin.weight)
            nn.init.zeros_(self.head_bin.bias)

    def init_class_bias_from_prior(
        self,
        *,
        prior_3: Optional[torch.Tensor] = None,
        prior_bin: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> None:
        """Bias-init the heads from the train-fold class prior (§18.17.3 D).

        Sets the head's output bias to the class log-odds so that at
        step 0, before any features are learned, the model already
        predicts the empirical prior. Combined with the segment-token
        zero-pad on padded positions, this means position 0 (which only
        sees one segment of history under the causal mask) emits the
        prior — *exactly* the behaviour we want, because the rare-class
        gradient noise at early positions otherwise dominates the loss
        signal and pushes AdamW into a class-prior-collapse basin.

        For the 3-class head (multinomial logits), bias is set to
        $b_c = \\log(p_c + \\varepsilon)$. The unnormalised softmax
        of bias-only logits then matches the prior by construction
        (the constant offset cancels in softmax).

        For the binary head (sigmoid logit), bias is set to the
        positive-class logit
        $b = \\log\\big(p^+ + \\varepsilon\\big)
              - \\log\\big(1 - p^+ + \\varepsilon\\big)$.
        With zero-init weights this gives ``prob_bin = sigmoid(b) = p^+``
        at step 0.

        Both ``head_3.weight`` and ``head_bin.weight`` are zero-init in
        :meth:`__init__`, so the prior-from-bias guarantee is exact for
        both heads at step 0: the ``W·z`` term contributes nothing and
        the softmax/sigmoid sees only the bias. The first gradient step
        breaks this symmetry as the weights begin to capture
        feature-conditional class boundaries; the prior is the
        starting point, not a constraint.

        Args:
            prior_3: Optional length-3 simplex tensor (sums to 1). If
                provided, sets ``head_3.bias = log(prior_3 + eps)``.
            prior_bin: Optional scalar in $(0, 1)$ (positive-class
                prior). If provided, sets the binary head's bias to
                the corresponding logit.
            eps: Numerical guard so log doesn't blow up on zero counts.
        """
        with torch.no_grad():
            if prior_3 is not None and self.head_3 is not None:
                p3 = prior_3.to(dtype=self.head_3.bias.dtype,
                                device=self.head_3.bias.device)
                if p3.numel() != self.head_3.bias.numel():
                    raise ValueError(
                        f"prior_3 size {p3.numel()} does not match "
                        f"head_3 bias size {self.head_3.bias.numel()}"
                    )
                self.head_3.bias.copy_(torch.log(p3 + eps))
            if prior_bin is not None and self.head_bin is not None:
                p = float(prior_bin.item() if isinstance(prior_bin, torch.Tensor) else prior_bin)
                p = min(max(p, eps), 1.0 - eps)
                logit = math.log(p) - math.log(1.0 - p)
                self.head_bin.bias.fill_(logit)

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
            Dict carrying *only* the enabled heads' outputs. When both
            heads are enabled (default):

              * ``logits_3``  — ``(B, N, num_classes_multi)``
              * ``logit_bin`` — ``(B, N)``
              * ``prob_3``    — ``(B, N, num_classes_multi)``  softmax
              * ``prob_bin``  — ``(B, N)``                     sigmoid

            When ``enable_three_class`` is False the four 3-class keys
            are omitted; when ``enable_binary`` is False the two binary
            keys are omitted. Consumers must check key presence with
            ``"key" in out`` rather than assume both heads exist.
        """
        z = self.proj(h)                                 # (B, N, d_model)
        mask_f = segment_mask.to(h.dtype)

        out: Dict[str, torch.Tensor] = {}
        if self.head_3 is not None:
            logits_3 = self.head_3(z)                    # (B, N, C)
            logits_3 = logits_3 * mask_f.unsqueeze(-1)
            prob_3 = F.softmax(logits_3, dim=-1) * mask_f.unsqueeze(-1)
            out["logits_3"] = logits_3
            out["prob_3"] = prob_3
        if self.head_bin is not None:
            logit_bin = self.head_bin(z).squeeze(-1)     # (B, N)
            logit_bin = logit_bin * mask_f
            prob_bin = torch.sigmoid(logit_bin) * mask_f
            out["logit_bin"] = logit_bin
            out["prob_bin"] = prob_bin
        return out


__all__ = [
    "PerPositionOutcomeHead",
]
