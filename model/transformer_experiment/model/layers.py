"""Primitive reusable layers for the Causal Multimodal Forecasting Transformer.

Contains:
    - RMSNorm: Root mean square normalization (LLaMA-style).
    - RotaryEmbedding: Rotary positional encoding for Q/K vectors.
    - CausalConv1d: 1D convolution with left-only padding for causality.
    - CausalConvBlock: Residual depthwise-separable causal conv block (spec §7.3).
    - CausalSelfAttention: Pre-norm multi-head self-attention with causal masking and RoPE.
    - CausalCrossAttention: Pre-norm multi-head cross-attention with causal masking and RoPE.
    - FeedForward: Pre-norm position-wise feed-forward network (GELU or SwiGLU).
    - AttentionPool: Learned attention-weighted pooling over a sequence dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    Normalizes by the RMS of the input (no mean centering). Faster and
    empirically equivalent to LayerNorm for transformer architectures.

    Args:
        d_model: Dimension of the input.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., d)``.

        Returns:
            Normalized tensor of the same shape.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _make_norm(d_model: int, use_rmsnorm: bool = True) -> nn.Module:
    """Factory for normalization layers.

    Args:
        d_model: Dimension of the input.
        use_rmsnorm: If True, return RMSNorm; otherwise LayerNorm.

    Returns:
        Normalization module.
    """
    if use_rmsnorm:
        return RMSNorm(d_model)
    return nn.LayerNorm(d_model)


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding for causal self-/cross-attention.

    Precomputes sinusoidal frequencies and caches cos/sin tables up to a
    maximum sequence length.  Applies rotation to query and key tensors so
    that dot-product attention becomes a function of relative position.

    Args:
        d_head: Dimension per attention head (must be even).
        max_seq_len: Maximum sequence length to precompute (can grow dynamically).
        theta_base: Base for the geometric frequency schedule.
    """

    def __init__(
        self,
        d_head: int,
        max_seq_len: int = 512,
        theta_base: float = 10000.0,
    ) -> None:
        super().__init__()
        assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"
        self.d_head = d_head

        # Frequencies: theta_i = 1 / (base^(2i/d))  for i in [0, d/2)
        inv_freq = 1.0 / (
            theta_base ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, d/2)
        # Duplicate for full d_head: [cos, cos] and [sin, sin]
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, d)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: Tensor, k: Tensor) -> tuple:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape ``(B, H, T, d_head)``.
            k: Key tensor of shape ``(B, H, T, d_head)``.

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes.
        """
        T = q.shape[2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, d)
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the last dimension."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding: x * cos + rotate_half(x) * sin."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Causal Conv1d
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Causal Conv Block
# ---------------------------------------------------------------------------

class CausalConvBlock(nn.Module):
    """Residual causal convolution block matching spec §7.3.

    Data flow::

        x (B, T, d)
        -> Norm
        -> transpose to (B, d, T)
        -> DWConv_causal (depthwise, groups=d)
        -> transpose to (B, T, d)
        -> Linear(d -> expansion*d) -> GELU/SiLU
        -> Linear(expansion*d -> d) -> Dropout
        -> + x  (residual)

    Args:
        d_model: Feature dimension.
        kernel_size: Kernel size for the depthwise causal convolution.
        dilation: Dilation factor for the depthwise causal convolution.
        expansion: Pointwise expansion ratio.
        dropout: Dropout probability.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
        use_swiglu: Whether to use SiLU activation (matching SwiGLU style).
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int = 1,
        expansion: int = 2,
        dropout: float = 0.1,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
    ) -> None:
        super().__init__()
        self.norm = _make_norm(d_model, use_rmsnorm)
        self.dw_conv = CausalConv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=d_model,
        )
        self.act = F.silu if use_swiglu else F.gelu
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
        h = self.act(self.pw_up(h))              # (B, T, expansion*d)
        h = self.drop(self.pw_down(h))           # (B, T, d)
        return residual + h


# ---------------------------------------------------------------------------
# Causal Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention with causal masking and RoPE.

    Uses ``F.scaled_dot_product_attention(is_causal=True)`` which automatically
    selects FlashAttention or the memory-efficient backend when available.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability on attention weights (applied only during
            training).
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm = _make_norm(d_model, use_rmsnorm)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.rotary = RotaryEmbedding(self.d_head)

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

        # Apply rotary positional encoding to Q, K
        q, k = self.rotary(q, k)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                  # (B, H, T, d_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)  # (B, T, d)
        return self.drop(self.out_proj(attn_out))


# ---------------------------------------------------------------------------
# Causal Cross-Attention
# ---------------------------------------------------------------------------

class CausalCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention with causal masking and RoPE.

    Query comes from the target stream, key/value from the source stream.
    Both streams share the same time dimension T, and causality enforces
    that source position s is only visible to query position t when s <= t.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability on attention weights.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = _make_norm(d_model, use_rmsnorm)
        self.norm_kv = _make_norm(d_model, use_rmsnorm)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.rotary = RotaryEmbedding(self.d_head)

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

        # Apply rotary positional encoding to Q, K
        q, k = self.rotary(q, k)

        # Both streams share time dim T; causal mask enforces s <= t.
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )                                                     # (B, H, T, d_h)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.drop(self.out_proj(attn_out))


# ---------------------------------------------------------------------------
# Feed-Forward Network (GELU or SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Pre-norm position-wise feed-forward network.

    Supports two modes:

    - **GELU** (default when ``use_swiglu=False``):
      ``Norm -> Linear(d -> ff_dim) -> GELU -> Dropout -> Linear(ff_dim -> d) -> Dropout``
    - **SwiGLU** (when ``use_swiglu=True``):
      ``Norm -> W3(SiLU(W1(x)) * W2(x)) -> Dropout``
      Uses ``ff_dim = int(expansion * d * 2/3)`` to keep ~same param count.

    Args:
        d_model: Input and output dimension.
        expansion: Expansion ratio for the hidden layer.
        dropout: Dropout probability.
        use_swiglu: Whether to use SwiGLU gated activation.
        use_rmsnorm: Whether to use RMSNorm instead of LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        dropout: float = 0.1,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_swiglu = use_swiglu
        self.norm = _make_norm(d_model, use_rmsnorm)
        self.drop = nn.Dropout(dropout)

        if use_swiglu:
            # SwiGLU: use 2/3 factor to keep param count similar to GELU path
            ff_dim = int(expansion * d_model * 2 / 3)
            self.w1 = nn.Linear(d_model, ff_dim)   # gate path
            self.w2 = nn.Linear(d_model, ff_dim)   # value path
            self.w3 = nn.Linear(ff_dim, d_model)   # output
        else:
            ff_dim = expansion * d_model
            self.fc1 = nn.Linear(d_model, ff_dim)
            self.fc2 = nn.Linear(ff_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, d)``.

        Returns:
            Output tensor of shape ``(B, T, d)`` (residual is added by
            the caller).
        """
        h = self.norm(x)
        if self.use_swiglu:
            h = self.w3(F.silu(self.w1(h)) * self.w2(h))
            return self.drop(h)
        else:
            h = F.gelu(self.fc1(h))
            h = self.drop(h)
            h = self.fc2(h)
            return self.drop(h)


# ---------------------------------------------------------------------------
# Attention Pool
# ---------------------------------------------------------------------------

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
