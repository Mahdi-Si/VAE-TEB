"""Primitive reusable layers for the Causal Multimodal Forecasting Transformer.

Contains:
    - CausalConv1d: 1D convolution with left-only padding for causality.
    - CausalConvBlock: Residual depthwise-separable causal conv block (spec §7.3).
    - CausalSelfAttention: Pre-norm multi-head self-attention with causal masking.
    - CausalCrossAttention: Pre-norm multi-head cross-attention with causal masking.
    - FeedForward: Pre-norm position-wise feed-forward network.
    - AttentionPool: Learned attention-weighted pooling over a sequence dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """1D convolution with explicit left-only padding to enforce causality.

    At time t the output depends only on inputs at times <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Dilation factor.
        bias: Whether to add a learnable bias.
        groups: Number of blocked connections (use ``in_channels`` for depthwise).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C_in, L)``.

        Returns:
            Output tensor of shape ``(B, C_out, L)`` (length-preserving when
            stride=1).
        """
        if self.left_padding > 0:
            x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """Residual causal convolution block matching spec §7.3.

    Data flow::

        x (B, T, d)
        → LayerNorm
        → transpose to (B, d, T)
        → DWConv_causal (depthwise, groups=d)
        → transpose to (B, T, d)
        → Linear(d → expansion*d) → GELU
        → Linear(expansion*d → d) → Dropout
        → + x  (residual)

    Args:
        d_model: Feature dimension.
        kernel_size: Kernel size for the depthwise causal convolution.
        dilation: Dilation factor for the depthwise causal convolution.
        expansion: Pointwise expansion ratio.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int = 1,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw_conv = CausalConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
        )
        self.pw_up = nn.Linear(d_model, expansion * d_model)
        self.pw_down = nn.Linear(expansion * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Output tensor of shape ``(B, T, d)`` with residual connection.
        """
        residual = x
        h = self.norm(x)                         # (B, T, d)
        h = h.transpose(1, 2)                    # (B, d, T)
        h = self.dw_conv(h)                       # (B, d, T)
        h = h.transpose(1, 2)                    # (B, T, d)
        h = F.gelu(self.pw_up(h))                # (B, T, expansion*d)
        h = self.drop(self.pw_down(h))           # (B, T, d)
        return residual + h


class CausalSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention with causal masking.

    Uses ``F.scaled_dot_product_attention(is_causal=True)`` which automatically
    selects FlashAttention or the memory-efficient backend when available.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability on attention weights (applied only during
            training).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Output tensor of shape ``(B, T, d)`` (pre-norm residual is added
            by the caller).
        """
        B, T, _ = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                      # each (B, T, H, d_h)
        q = q.transpose(1, 2)                              # (B, H, T, d_h)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                  # (B, H, T, d_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d)
        return self.drop(self.out_proj(attn_out))


class CausalCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention with causal masking.

    Query comes from the target stream, key/value from the source stream.
    Both streams share the same time dimension T, and causality enforces
    that source position s is only visible to query position t when s <= t.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        """Forward pass.

        Args:
            target: Query stream of shape ``(B, T, d)`` (e.g. FHR encoder states).
            source: Key/Value stream of shape ``(B, T, d)`` (e.g. UP encoder states).

        Returns:
            Attended context of shape ``(B, T, d)``.
        """
        B, T, _ = target.shape
        q = self.q_proj(self.norm_q(target)).reshape(
            B, T, self.n_heads, self.d_head
        ).transpose(1, 2)                                    # (B, H, T, d_h)

        kv = self.kv_proj(self.norm_kv(source)).reshape(
            B, T, 2, self.n_heads, self.d_head
        )
        k, v = kv.unbind(dim=2)                              # each (B, T, H, d_h)
        k = k.transpose(1, 2)                                # (B, H, T, d_h)
        v = v.transpose(1, 2)

        # Both streams share time dim T; causal mask enforces s <= t.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                     # (B, H, T, d_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.drop(self.out_proj(attn_out))


class FeedForward(nn.Module):
    """Pre-norm position-wise feed-forward network.

    Architecture: ``LayerNorm → Linear(d → ff_dim) → GELU → Dropout
    → Linear(ff_dim → d) → Dropout``.

    Args:
        d_model: Input and output dimension.
        expansion: Expansion ratio for the hidden layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ff_dim = expansion * d_model
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Output tensor of shape ``(B, T, d)`` (residual is added by
            the caller).
        """
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return self.drop(h)


class AttentionPool(nn.Module):
    """Learned attention-weighted pooling over a sequence dimension.

    Computes::

        score_t = w^T tanh(W h_t + b)
        alpha   = softmax(score, dim=T)
        output  = sum(alpha * h, dim=T)

    Args:
        d_model: Dimension of the input hidden states.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(*, T, d)``.

        Returns:
            Pooled tensor of shape ``(*, d)``.
        """
        scores = self.net(x)                         # (*, T, 1)
        alpha = F.softmax(scores, dim=-2)            # (*, T, 1)
        pooled = (alpha * x).sum(dim=-2)             # (*, d)
        return pooled
