"""Causal transformer encoders and cross-attention fusion.

Contains:
    - CausalTransformerBlock: Single pre-norm causal self-attention + FFN block.
    - CausalTransformerEncoder: Stack of N blocks + final LayerNorm (spec §8).
    - CausalCrossAttentionFusion: Cross-attention + gated residual fusion (spec §9).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .layers import CausalCrossAttention, CausalSelfAttention, FeedForward


class CausalTransformerBlock(nn.Module):
    """Single pre-norm causal transformer block.

    Architecture::

        x = x + CausalSelfAttention(x)
        x = x + FeedForward(x)

    Supports gradient checkpointing when ``use_checkpoint=True``.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        ff_expansion: Feed-forward expansion ratio.
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, ff_expansion, dropout)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: Tensor) -> Tensor:
        """Core forward logic (separated for checkpointing)."""
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Output tensor of shape ``(B, T, d)``.
        """
        if self.use_checkpoint and self.training:
            return grad_checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class CausalTransformerEncoder(nn.Module):
    """Stack of causal transformer blocks with final layer normalization.

    Used for the FHR-only encoder, UP-only encoder, and fused encoder
    (spec §8.3).  Each instance is parameterized independently.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked transformer blocks.
        ff_expansion: Feed-forward expansion ratio.
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing in blocks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model, n_heads, ff_expansion, dropout, use_checkpoint
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Encoded tensor of shape ``(B, T, d)``.
        """
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class CausalCrossAttentionFusion(nn.Module):
    """Cross-attention with gated residual fusion (spec §9).

    At each time t, the FHR state queries the UP state history (s <= t)
    via causal cross-attention.  The attended UP context is then gated and
    added to the FHR state as a residual.

    Architecture::

        C_t = CrossAttention(Q=H_F, K/V=H_U)     # (B, T, d)
        G_t = sigmoid(Linear([H_F_t | C_t]))      # (B, T, d)
        H_tilde_t = H_F_t + G_t * C_t             # gated residual

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = CausalCrossAttention(d_model, n_heads, dropout)
        self.gate_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, h_fhr: Tensor, h_up: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h_fhr: FHR encoder states of shape ``(B, T, d)``.
            h_up: UP encoder states of shape ``(B, T, d)``.

        Returns:
            Fused states of shape ``(B, T, d)``, ready for the fused encoder.
        """
        context = self.cross_attn(target=h_fhr, source=h_up)  # (B, T, d)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([h_fhr, context], dim=-1))
        )                                                       # (B, T, d)
        return h_fhr + gate * context
