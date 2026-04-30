"""Relative-time transformer for ``guid_cls_v1`` (PRD §7).

Pre-norm transformer with per-head log-bucketed relative-time biases. Takes
the precomputed ``rel_bucket_idx`` from the collate function so the model
itself never reads ``epoch``.

Under the causal-autoregressive design (default), a lower-triangular mask
is AND'd with the key-validity mask so position ``n`` can only attend to
positions ``j <= n``. This makes the per-position outputs genuine
"history-up-to-here" predictions and matches the per-position
``PerPositionOutcomeHead`` consumed by the classifier head.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeTimeMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with per-head additive relative-time bias.

    Args:
        d_model: Model width (must equal ``n_heads * d_head``).
        n_heads: Number of attention heads.
        d_head: Per-head dimensionality.
        n_buckets: Number of relative-time buckets (PRD §7.1, default 32).
        dropout: Attention dropout probability.
        causal: If True (default), AND a lower-triangular ``(N, N)`` mask
            with ``key_mask`` so position ``n`` only attends to positions
            ``j <= n``. Required for autoregressive per-position prediction.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_buckets: int,
        dropout: float = 0.1,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError(
                f"n_heads * d_head ({n_heads}*{d_head}) must equal d_model ({d_model})"
            )
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.n_buckets = int(n_buckets)
        self.causal = bool(causal)
        self.scale = 1.0 / math.sqrt(d_head)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Per-head additive bias table indexed by the (i, j) bucket id.
        self.bias = nn.Parameter(torch.zeros(n_heads, n_buckets))
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

        # Lazily allocated lower-triangular template grown on demand.
        self.register_buffer(
            "_causal_template",
            torch.empty(0, 0, dtype=torch.bool),
            persistent=False,
        )

    def _get_causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """Return a cached ``(n, n)`` lower-triangular bool mask.

        Grows the cached buffer on demand. ``True`` at ``[i, j]`` means
        position ``i`` is allowed to attend to position ``j``.
        """
        cur = self._causal_template
        if cur.shape[0] >= n and cur.device == device:
            return cur[:n, :n]
        bigger = max(n, max(64, cur.shape[0] * 2))
        new = torch.tril(torch.ones(bigger, bigger, dtype=torch.bool, device=device))
        self._causal_template = new
        return new[:n, :n]

    def forward(
        self,
        x: torch.Tensor,
        segment_mask: torch.Tensor,
        rel_bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative-time self-attention.

        Args:
            x: ``(B, N, d_model)`` input tokens.
            segment_mask: ``(B, N)`` bool — True for valid keys/queries.
            rel_bucket_idx: ``(B, N, N)`` long — bucket id per pair.

        Returns:
            ``(B, N, d_model)`` updated tokens.
        """
        B, N, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.W_k(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (B, H, N, N)

        # Per-head additive bias gather: bias[H, n_buckets] -> (B, H, N, N).
        # rel_bucket_idx is (B, N, N); expand head dim and gather along bucket axis.
        idx = rel_bucket_idx.unsqueeze(1).expand(B, H, N, N)            # (B, H, N, N)
        bias_table = self.bias.unsqueeze(0).expand(B, H, self.n_buckets)
        # Use advanced indexing: for each head h, look up bias_table[b, h, idx[b, h, i, j]].
        bias = torch.gather(
            bias_table.unsqueeze(2).expand(B, H, N, self.n_buckets),
            dim=-1,
            index=idx,
        )                                                                # (B, H, N, N)
        scores = scores + bias

        # Mask invalid keys (and rows whose query is also invalid produce a
        # safe but unused output; downstream segment_mask zeros those tokens).
        # Under ``causal=True``, AND a lower-triangular mask so position ``n``
        # cannot attend to positions ``j > n`` — required for the
        # autoregressive per-position head.
        key_mask = segment_mask.view(B, 1, 1, N)                         # (B, 1, 1, N)
        if self.causal:
            causal_mask = self._get_causal_mask(N, x.device).view(1, 1, N, N)
            valid = key_mask & causal_mask
        else:
            valid = key_mask
        scores = scores.masked_fill(~valid, float("-inf"))

        # ``softmax(all -inf)`` would produce NaN; ``nan_to_num`` rescues
        # the rare row where every key is invalid (downstream segment_mask
        # then zeros the corresponding output token). With ``causal=True``,
        # position 0 has at least itself as a valid key when ``segment_mask[:, 0]``
        # is True, so the rescue still only fires for fully-padded GUIDs.
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)                                       # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)


class RelativeTimeTransformerBlock(nn.Module):
    """One pre-norm transformer block with relative-time attention.

    Args:
        d_model: Hidden width.
        n_heads: Attention heads.
        d_head: Per-head dim.
        d_ff: Feed-forward inner width.
        n_buckets: Number of relative-time buckets.
        dropout: Dropout for attention probs and FFN.
        causal: Forwarded to :class:`RelativeTimeMultiHeadSelfAttention`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_buckets: int,
        dropout: float = 0.1,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RelativeTimeMultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            n_buckets=n_buckets,
            dropout=dropout,
            causal=causal,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        segment_mask: torch.Tensor,
        rel_bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(B, N, d_model)`` input.
            segment_mask: ``(B, N)`` bool.
            rel_bucket_idx: ``(B, N, N)`` int64.

        Returns:
            ``(B, N, d_model)`` updated representation.
        """
        x = x + self.attn(self.ln1(x), segment_mask, rel_bucket_idx)
        x = x + self.ffn(self.ln2(x))
        return x


class RelativeTimeTransformer(nn.Module):
    """Stack of pre-norm relative-time transformer blocks.

    Args:
        d_model: Model width (default 256).
        n_heads: Heads per block (default 4).
        d_head: Per-head dim (default 64; ``n_heads * d_head == d_model``).
        n_layers: Number of blocks (default 3).
        d_ff: Feed-forward inner width (default 512).
        n_buckets: Number of bias buckets (default 32, PRD §7.1).
        dropout: Dropout used in attn and FFN (default 0.1).
        causal: If True (default), every block attends causally.
    """

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_heads: int = 4,
        d_head: int = 64,
        n_layers: int = 3,
        d_ff: int = 512,
        n_buckets: int = 32,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if n_heads * d_head != d_model:
            raise ValueError(
                f"n_heads * d_head ({n_heads}*{d_head}) must equal d_model ({d_model})"
            )
        self.d_model = d_model
        self.causal = bool(causal)
        self.layers = nn.ModuleList(
            [
                RelativeTimeTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_head=d_head,
                    d_ff=d_ff,
                    n_buckets=n_buckets,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        segment_mask: torch.Tensor,
        rel_bucket_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply all blocks.

        Args:
            x: ``(B, N, d_model)`` input tokens.
            segment_mask: ``(B, N)`` bool.
            rel_bucket_idx: ``(B, N, N)`` int64.

        Returns:
            ``(B, N, d_model)`` final representation, with padded rows zeroed.
        """
        for layer in self.layers:
            x = layer(x, segment_mask, rel_bucket_idx)
        x = self.final_norm(x)
        return x * segment_mask.to(x.dtype).unsqueeze(-1)


__all__ = [
    "RelativeTimeTransformer",
    "RelativeTimeTransformerBlock",
    "RelativeTimeMultiHeadSelfAttention",
]
