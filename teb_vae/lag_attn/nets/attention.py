r"""Lag cross-attention: the mechanism that decides *when* the source mattered.

The model asks a question with a delay in it. A uterine contraction does not move the fetal heart
rate at the same instant; the response arrives some lag $\ell$ later, the lag is not known in
advance, and it is not the same for every recording. So rather than fixing a lag, the target
state at step $t$ attends over a window of source states $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and
learns where to look. The attention weights are then themselves a readout: they say which lag the
model found informative.

Two design choices here are load-bearing.

**The window is a view, not a bank.** Keys and values are projected once from $H^u$ and the lag
window is formed with strided ``unfold`` views over the projected tensors. Materialising the
window as a real $(B, T, L, d_{model})$ tensor costs roughly $L\times$ the activation memory --
at $L = 91$ that is the difference between fitting a batch and not. The result is numerically
identical.

**Long lags start penalised.** With ``lag_bias_init='alibi_decay'`` each head is seeded with a
negative-slope bias in $\ell$, so the model begins biased toward short lags and must *earn* a
long-lag reading. Without it, a randomly-initialised head can find spurious long-lag structure
early and never leave it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, cast

import torch
import torch.nn.functional as F
from entmax import entmax15
from torch import nn
from torch.utils.checkpoint import checkpoint


def alibi_slopes(num_heads: int) -> torch.Tensor:
    r"""ALiBi-style geometric per-head slopes for the lag-decay bias.

    Returns positive slopes $m_h$, one per head, following the power-of-two schedule of Press et
    al. (2022). A score bias $-m_h\,\ell$ then penalises larger lags. The slopes are geometric so
    the heads span scales: the steepest head sees essentially only the current step, the
    shallowest is nearly flat across the window, and between them the set covers short, medium
    and long lags without any head being told which to take.

    Args:
        num_heads: Number of attention heads.

    Returns:
        Tensor of shape ``(num_heads,)`` holding the per-head slopes.
    """

    def _pow2_slopes(n: int) -> list:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        return [start * (start**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _pow2_slopes(num_heads)
    else:  # pragma: no cover - num_heads is a power of two in practice
        closest = 2 ** int(math.floor(math.log2(num_heads)))
        slopes = _pow2_slopes(closest)
        extra = _pow2_slopes(2 * closest)[0::2][: num_heads - closest]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)


class LagCrossAttention(nn.Module):
    """Multi-head cross-attention from the target state to lagged source memory.

    Each head learns a per-lag additive key bias (Shaw-style relative position encoding) and,
    optionally, a per-(head, lag) scalar score bias. Scores are masked by lag validity, then
    normalised with ``softmax`` or ``entmax15``.

    ``entmax15`` is worth the dependency: unlike ``softmax`` it can assign a lag *exactly* zero
    weight rather than merely a small one. When the output is read as "which lag mattered", the
    difference between $0$ and $10^{-4}$ across $91$ lags is the difference between a clean answer
    and a smear.

    Shapes:
        ``h_y``: ``(B, T, d_model)``; ``h_u``: ``(B, T, d_model)``
        Output: ``(B, T, d_model)``, ``(B, T, num_heads, L)``, ``(B, T, num_heads, d_head)``
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        d_head: int = 32,
        max_lag: int = 90,
        dropout: float = 0.1,
        use_entmax: bool = False,
        grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
    ) -> None:
        """Initialize the attention module.

        Args:
            d_model: Query/key/value embedding width; must equal ``num_heads * d_head``.
            num_heads: Number of attention heads.
            d_head: Per-head width.
            max_lag: Maximum past lag; the window is $L = \\mathrm{max\\_lag} + 1$ wide.
            dropout: Dropout on the attention probabilities.
            use_entmax: Use ``entmax15`` instead of ``softmax``, giving exactly-zero weights.
            grad_checkpoint: Recompute the attention in the backward pass instead of storing it.
                Trades compute for memory; with the view-based projection the retained memory is
                already small, so this can usually stay ``False``.
            lag_bias_init: ``'normal'`` registers no extra parameter. ``'alibi_decay'`` seeds a
                learnable per-(head, lag) scalar score bias with a negative slope in $\\ell$.
            alibi_slope_scale: Multiplier on the ``'alibi_decay'`` slopes. Values below $1$
                soften the long-lag penalty so a genuinely long coupling is reachable at init;
                $0$ gives a flat but still learnable bias. Ignored when ``lag_bias_init`` is
                ``'normal'``.

        Raises:
            ValueError: If ``num_heads * d_head != d_model``, or ``lag_bias_init`` is not one of
                the two accepted values.
        """
        super().__init__()
        if num_heads * d_head != d_model:
            raise ValueError(
                f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
            )
        if lag_bias_init not in ("normal", "alibi_decay"):
            raise ValueError(
                f"lag_bias_init must be 'normal' or 'alibi_decay', got {lag_bias_init!r}"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.L = int(max_lag) + 1
        self.scale = 1.0 / math.sqrt(d_head)
        self.use_entmax = bool(use_entmax)
        self.grad_checkpoint = bool(grad_checkpoint)

        # Pre-norm, the standard transformer pattern: it stabilises the dot-product scores
        # before the normaliser saturates on them.
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Per-lag learned bias added to the keys (Shaw-style relative position encoding).
        self.lag_embeddings = nn.Parameter(torch.zeros(self.L, num_heads, d_head))
        nn.init.normal_(self.lag_embeddings, mean=0.0, std=0.02)

        self.lag_bias_init = lag_bias_init
        self.alibi_slope_scale = float(alibi_slope_scale)
        if lag_bias_init == "alibi_decay":
            slopes = alibi_slopes(num_heads) * float(alibi_slope_scale)  # (num_heads,)
            lags = torch.arange(self.L, dtype=torch.float32)             # (L,)
            decay = -slopes[:, None] * lags[None, :]                     # (num_heads, L)
            self.lag_score_bias = nn.Parameter(decay)
        else:
            self.register_parameter("lag_score_bias", None)

    def build_lag_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Return the lag-validity mask: ``True`` where the lagged step exists.

        Lag $\\ell$ at step $t$ refers to source step $t - \\ell$, which does not exist for
        $\\ell > t$. Early steps therefore have fewer valid lags than late ones.

        Public because the model composes this with its own band mask. It was private, and the
        model reached through the underscore to get at it -- which is not encapsulation, only
        the appearance of it.

        Args:
            seq_len: Sequence length $T$.
            device: Device to build the mask on.

        Returns:
            Bool tensor of shape ``(T, L)``, ``True`` iff ``t - l >= 0``.
        """
        steps = torch.arange(seq_len, device=device)[:, None]
        lags = torch.arange(self.L, device=device)[None, :]
        return steps - lags >= 0

    def _attend(
        self,
        h_y: torch.Tensor,
        h_u: torch.Tensor,
        m_lag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the attention.

        Split out from ``forward`` so gradient checkpointing has a single function to wrap.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            h_u: Source state ``(B, T, d_model)``.
            m_lag: Lag-validity mask ``(T, L)``.

        Returns:
            The attended output, the attention weights in lag order, and the per-head summaries.
        """
        batch, seq_len = h_y.shape[0], h_y.shape[1]
        num_lags = self.L
        heads = self.num_heads
        width = self.d_head

        # Project Q from the target state and K/V from the source state exactly once.
        q = self.W_q(self.q_norm(h_y)).view(batch, seq_len, heads, width)
        h_u_normed = self.kv_norm(h_u)
        k = self.W_k(h_u_normed).view(batch, seq_len, heads, width)
        v = self.W_v(h_u_normed).view(batch, seq_len, heads, width)

        # Banded lag windows as strided views. Window index j maps to lag L-1-j (oldest first);
        # every lag-indexed tensor (bias, mask, returned alpha) is flipped on the small L axis
        # so the large windows can stay views rather than copies.
        k_padded = F.pad(k, (0, 0, 0, 0, num_lags - 1, 0))  # (B, T+L-1, Mh, d)
        v_padded = F.pad(v, (0, 0, 0, 0, num_lags - 1, 0))
        k_window = k_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view
        v_window = v_padded.unfold(1, num_lags, 1)          # (B, T, Mh, d, L) view

        # Content score <q_t, k_{t-l}> plus the Shaw bias <q_t, r_l>, in window order.
        scores = torch.einsum("btmd,btmdj->btmj", q, k_window)
        scores = scores + torch.einsum("btmd,jmd->btmj", q, self.lag_embeddings.flip(0))
        scores = scores * self.scale
        if self.lag_score_bias is not None:
            scores = scores + self.lag_score_bias.flip(-1)[None, None, :, :]

        # Lag validity in window order: position j is valid iff t-(L-1-j) >= 0.
        mask_window = m_lag.flip(-1).to(torch.bool)
        scores = scores.masked_fill(~mask_window[None, :, None, :], float("-inf"))

        # entmax ships no type stubs, hence the cast.
        alpha_window = (
            cast(torch.Tensor, entmax15(scores, dim=-1))
            if self.use_entmax
            else F.softmax(scores, dim=-1)
        )
        # A row with no valid lag is all -inf, which normalises to NaN. Such rows exist by
        # construction at t=0 under a restrictive band mask; zero is the right reading -- no lag
        # was attended because none was available.
        alpha_window = torch.nan_to_num(alpha_window, nan=0.0)
        alpha_window = self.attn_dropout(alpha_window)

        head_out = torch.einsum("btmj,btmdj->btmd", alpha_window, v_window)
        out = self.W_o(head_out.reshape(batch, seq_len, heads * width))

        alpha = alpha_window.flip(-1)  # back to lag order
        # head_out is the per-head attended summary before W_o. The head-structured posterior
        # consumes it; the flat path ignores it.
        return out, alpha, head_out

    def forward(
        self,
        h_y: torch.Tensor,
        h_u: torch.Tensor,
        m_lag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attend from ``h_y`` over lagged projections of ``h_u``.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            h_u: Source state ``(B, T, d_model)``.
            m_lag: Optional lag-validity mask. ``(T, L)`` is preferred; a ``(B, T, L)`` mask is
                accepted and collapsed, since it can only depend on ``(t, l)``. Built on the fly
                when ``None``.

        Returns:
            ``(A, alpha, a_heads)``: the attended output ``(B, T, d_model)``; the attention
            weights ``(B, T, num_heads, L)`` in lag order, index $0$ being the current step; and
            the per-head summaries ``(B, T, num_heads, d_head)`` taken before ``W_o``.
        """
        if m_lag is None:
            m_lag = self.build_lag_mask(h_y.shape[1], device=h_y.device)
        elif m_lag.dim() == 3:
            m_lag = m_lag[0]
        if self.grad_checkpoint and self.training:
            return cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                checkpoint(self._attend, h_y, h_u, m_lag, use_reentrant=False),
            )
        return self._attend(h_y, h_u, m_lag)
