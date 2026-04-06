"""Causal transformer encoders and cross-attention fusion.

Contains:
    - CausalTransformerBlock: Single pre-norm causal self-attention + FFN block.
    - CausalTransformerEncoder: Stack of N blocks + final normalization (spec §8).
    - CausalCrossAttentionFusion: Multi-layer cross-attention + gated residual
      fusion (spec §9, v2 extension).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .layers import (
    CausalCrossAttention,
    CausalSelfAttention,
    FeedForward,
    _make_norm,
)


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
        use_swiglu: Whether to use SwiGLU feed-forward.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout, use_rmsnorm=use_rmsnorm,
        )
        self.ff = FeedForward(
            d_model, ff_expansion, dropout,
            use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm,
        )
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
    """Stack of causal transformer blocks with final normalization.

    Used for the FHR-only encoder, UP-only encoder, and fused encoder
    (spec §8.3).  Each instance is parameterized independently.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked transformer blocks.
        ff_expansion: Feed-forward expansion ratio.
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing in blocks.
        use_swiglu: Whether to use SwiGLU feed-forward.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model, n_heads, ff_expansion, dropout,
                use_checkpoint, use_swiglu, use_rmsnorm,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = _make_norm(d_model, use_rmsnorm)

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
    """Multi-layer cross-attention with gated residual fusion (spec §9, v2).

    Each fusion layer performs:

    1. Cross-attention from the fused stream (query) to the UP stream (key/value).
    2. Sigmoid gate on the concatenation of the fused state and attended context.
    3. Gated residual addition.
    4. A full causal transformer block (self-attention + FFN) to integrate.

    With ``n_layers=1`` this reduces to the original v1 design (single
    cross-attention + gate, no self-attention refinement within fusion).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of stacked fusion layers.
        ff_expansion: Feed-forward expansion ratio for the internal blocks.
        dropout: Dropout probability.
        use_swiglu: Whether to use SwiGLU in the internal transformer blocks.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int = 2,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers

        self.cross_attns = nn.ModuleList([
            CausalCrossAttention(d_model, n_heads, dropout, use_rmsnorm)
            for _ in range(n_layers)
        ])
        self.gate_projs = nn.ModuleList([
            nn.Linear(2 * d_model, d_model)
            for _ in range(n_layers)
        ])
        # Each fusion layer (except possibly the first in v1-compat mode)
        # has a self-attention + FFN block for integration
        self.refine_blocks = nn.ModuleList([
            CausalTransformerBlock(
                d_model, n_heads, ff_expansion, dropout,
                use_swiglu=use_swiglu, use_rmsnorm=use_rmsnorm,
            )
            for _ in range(n_layers)
        ])

    def forward(self, h_fhr: Tensor, h_up: Tensor) -> Tensor:
        """Forward pass.

        Args:
            h_fhr: FHR encoder states of shape ``(B, T, d)``.
            h_up: UP encoder states of shape ``(B, T, d)``.

        Returns:
            Fused states of shape ``(B, T, d)``, ready for the fused encoder.
        """
        h = h_fhr
        for i in range(self.n_layers):
            context = self.cross_attns[i](target=h, source=h_up)  # (B, T, d)
            gate = torch.sigmoid(
                self.gate_projs[i](torch.cat([h, context], dim=-1))
            )                                                       # (B, T, d)
            h = h + gate * context
            h = self.refine_blocks[i](h)
        return h
