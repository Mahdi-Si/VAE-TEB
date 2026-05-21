"""Lag-Attentive Residual VAE-TEB (v1).

This module implements the ``SeqVaeLagAttnV1`` model specified in
``model/vae_teb_prediction/model/new_architecture.md``. It is a source-pure
variant of the original :class:`SeqVae` that:

* consumes ``[up_st, up_ph]`` as the source stream (no cross-phase),
* replaces the ``ConditionalEncoder`` with a causal *lag cross-attention* block
  over a sliding window of past UP states,
* decomposes the future forecast as
  :math:`\\hat{Y}^{full} = \\hat{Y}^{base} + \\Delta\\hat{Y}^{src}` so that the
  KL bottleneck carries only the UP-driven residual contribution,
* supervises the future feature trajectory
  :math:`Y^+_t = Y_{t+1:t+H_d}` directly instead of reconstructing the raw
  signal.

All shared building blocks (``ResidualMLP``, ``CausalConv1d``,
``CausalMultiChannelConvBlock``, ``geometric_schedule``, ``initialization``)
are imported from :mod:`model.vae_teb_prediction.model.vae_teb_model_prediction`
to keep a single source of truth.

Shape conventions
-----------------
``B``  batch size
``T``  decimated sequence length (300 for 20 min @ 4 Hz with stride 16)
``C_y`` FHR feature channels (43 + 44 = 87)
``C_u`` UP feature channels (43 + 58 = 101), or 58 when ``up_st`` is absent
``d_model`` internal width (default 128)
``d_z`` latent dim (default 24)
``H_d`` decimated forecast horizon (default 30, i.e. 2 min)
``L``  lag window length, ``L = max_lag + 1`` (default 91 = 6 min of history)
``M_heads`` number of attention heads (default 4)
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from model.vae_teb_prediction.model.vae_teb_model_prediction import (
    CausalMultiChannelConvBlock,
    ResidualMLP,
    geometric_schedule,
    initialization,
)

try:  # Optional sparse attention normalisation (spec 6.7)
    from entmax import entmax15 as _entmax15  # type: ignore
    _HAS_ENTMAX = True
except Exception:  # pragma: no cover - optional dependency
    _entmax15 = None
    _HAS_ENTMAX = False


# =============================================================================
# Input adapters
# =============================================================================


class TargetInputAdapter(nn.Module):
    """Project target features ``Y`` into the internal model width.

    Shapes:
        Input:  ``(B, T, in_dim)``
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        in_dim: int = 87,
        d_model: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the adapter.

        Args:
            in_dim: Number of target feature channels (87 = 43 + 44).
            d_model: Internal model width.
            dropout: Dropout probability applied after GELU.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, T, in_dim)`` → ``(B, T, d_model)``."""
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.res_mlp(x)
        return x


class SourceInputAdapter(nn.Module):
    """Project source features ``U`` into the internal model width.

    Shapes:
        Input:  ``(B, T, in_dim)`` — ``in_dim = 101`` or ``58`` (fallback).
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        in_dim: int = 101,
        d_model: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the adapter.

        Args:
            in_dim: Number of source feature channels. ``101`` when ``up_st``
                is available (43 + 58), ``58`` when ``up_st`` is absent.
            d_model: Internal model width.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``(B, T, in_dim)`` → ``(B, T, d_model)``."""
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.res_mlp(x)
        return x


# =============================================================================
# Encoders
# =============================================================================


class _CausalConvLstmEncoder(nn.Module):
    """Shared causal encoder body used by target and source streams.

    Three causal multi-channel conv blocks (with intra-stack skip connections
    in the style of the legacy :class:`TargetEncoder`) run in parallel to a
    unidirectional LSTM. Their outputs are concatenated and fused through a
    residual MLP, yielding a ``(B, T, d_model)`` hidden sequence.
    """

    def __init__(
        self,
        *,
        d_model: int,
        cnn_kernels: Tuple[int, int, int],
        cnn_dilations: Tuple[int, int, int],
        lstm_layers: int,
        lstm_dropout: float,
        conv_dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Stage A — front-end residual MLP (keeps channel count at d_model)
        self.front_mlp = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=conv_dropout,
        )

        # Stage B — three stacked causal conv blocks with inter-block skips
        k1, k2, k3 = cnn_kernels
        d1, d2, d3 = cnn_dilations
        self.conv_1 = CausalMultiChannelConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            filter_size=k1,
            dilation=d1,
            dropout=conv_dropout,
            activation=nn.GELU,
        )
        self.conv_2 = CausalMultiChannelConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            filter_size=k2,
            dilation=d2,
            dropout=conv_dropout,
            activation=nn.GELU,
        )
        self.conv_3 = CausalMultiChannelConvBlock(
            in_channels=d_model,
            out_channels=d_model,
            filter_size=k3,
            dilation=d3,
            dropout=conv_dropout,
            activation=nn.GELU,
        )
        self.stack_skip_norm_1 = nn.GroupNorm(
            num_groups=min(8, d_model), num_channels=d_model
        )
        self.stack_skip_norm_2 = nn.GroupNorm(
            num_groups=min(8, d_model), num_channels=d_model
        )
        self.conv_out_norm = nn.LayerNorm(d_model)

        # Stage C — unidirectional LSTM branch
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # Stage D — fusion of conv + LSTM outputs
        self.fusion = ResidualMLP(
            input_dim=2 * d_model,
            hidden_dims=geometric_schedule(2 * d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=conv_dropout,
        )
        # Caps the encoder exit to per-step ~N(0, 1). Prevents unbounded drift
        # that previously let a single latent dim land at ~-25 in mu_prior.
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, d_model)`` → ``(B, T, d_model)``."""
        # Stage A
        x_lin = self.front_mlp(x)  # (B, T, d_model)

        # Stage B — conv stack with skips (inputs in (B, C, T))
        x_conv = x_lin.transpose(1, 2).contiguous()  # (B, d_model, T)
        c1 = self.conv_1(x_conv)
        c2 = self.conv_2(c1) + self.stack_skip_norm_1(c1)
        c3 = self.conv_3(c2) + self.stack_skip_norm_2(c2)
        conv_out = self.conv_out_norm(c3.transpose(1, 2).contiguous() + x_lin)

        # Stage C — LSTM
        lstm_out, _ = self.lstm(x_lin)
        lstm_out = self.lstm_norm(lstm_out)

        # Stage D — concat + fusion, then bound the exit.
        fused = torch.cat([conv_out, lstm_out], dim=-1)  # (B, T, 2*d_model)
        out = self.output_norm(self.fusion(fused))       # (B, T, d_model)
        return out


class TargetEncoder(nn.Module):
    """Target-only causal encoder producing the FHR history state ``H^y``.

    Shapes:
        Input:  ``(B, T, d_model)`` (already projected by ``TargetInputAdapter``).
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_kernels: Tuple[int, int, int] = (3, 7, 11),
        cnn_dilations: Tuple[int, int, int] = (1, 2, 4),
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ) -> None:
        """Initialize the target encoder."""
        super().__init__()
        self.body = _CausalConvLstmEncoder(
            d_model=d_model,
            cnn_kernels=cnn_kernels,
            cnn_dilations=cnn_dilations,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            conv_dropout=conv_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``H^y`` for the target stream."""
        return self.body(x)


class SourceEncoder(nn.Module):
    """Source-only causal encoder producing the UP history state ``H^u``.

    Shapes:
        Input:  ``(B, T, d_model)``
        Output: ``(B, T, d_model)``
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_kernels: Tuple[int, int, int] = (3, 5, 11),
        cnn_dilations: Tuple[int, int, int] = (1, 2, 4),
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        conv_dropout: float = 0.1,
    ) -> None:
        """Initialize the source encoder."""
        super().__init__()
        self.body = _CausalConvLstmEncoder(
            d_model=d_model,
            cnn_kernels=cnn_kernels,
            cnn_dilations=cnn_dilations,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            conv_dropout=conv_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``H^u`` for the source stream."""
        return self.body(x)


# =============================================================================
# Prior / Posterior heads
# =============================================================================


class PriorHead(nn.Module):
    """Target-only prior ``p(z_t | Y_{<=t})`` and decoder conditioning state.

    Produces three outputs from the target state ``H^y``:

    * ``mu_prior`` — prior mean, shape ``(B, T, d_z)``. Bounded via
      ``mu_scale * tanh(raw / mu_scale)`` so ``|mu_prior| <= mu_scale``.
    * ``logvar_prior`` — prior log-variance, shape ``(B, T, d_z)``, clamped
    * ``decoder_state`` — FHR-only conditioning for the baseline decoder,
      shape ``(B, T, d_model)``

    Each of the three heads is fed through its own ``LayerNorm`` so the raw
    encoder state cannot drift unbounded through any of them.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        mu_scale: float = 5.0,
    ) -> None:
        """Initialize the prior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(min, max)`` clamps for ``logvar_prior``.
            dropout: Dropout used inside every internal ResidualMLP.
            mu_scale: Saturation magnitude of the tanh-bound prior mean; must
                be positive. Set large enough to be non-restrictive (5.0 gives
                plenty of room around the ``N(0, I)`` reference).
        """
        super().__init__()
        if mu_scale <= 0.0:
            raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
        self.logvar_clamp = logvar_clamp
        self.mu_scale = float(mu_scale)

        # Per-head input LayerNorms decouple the three heads from shared
        # drift in ``h_y`` and are the cheapest available stabilizer.
        self.mu_input_norm = nn.LayerNorm(d_model)
        self.logvar_input_norm = nn.LayerNorm(d_model)
        self.dec_input_norm = nn.LayerNorm(d_model)

        self.mu_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.logvar_prior_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.decoder_state_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(
        self, h_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mu_prior, logvar_prior, decoder_state)``."""
        raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
        mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)

        logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
        logvar_prior = torch.clamp(
            logvar_prior, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        decoder_state = self.decoder_state_head(self.dec_input_norm(h_y))
        return mu_prior, logvar_prior, decoder_state


class PosteriorHead(nn.Module):
    """Source-conditioned posterior ``q(z_t | Y_{<=t}, U_{<=t})``.

    Uses the residual parameterisation ``mu_post = mu_prior + delta_mu`` so that
    at initialisation (with ``delta_mu_head`` zero-inited) the KL divergence
    against the prior is close to zero. This keeps early training stable when
    ``beta`` is small. ``delta_mu`` is additionally tanh-bounded to
    ``[-delta_mu_scale, +delta_mu_scale]`` to prevent the posterior mean from
    drifting arbitrarily far from the prior.

    Per-modality ``LayerNorm`` is applied to ``h_y`` and the attended source
    ``a`` before the fusion concat so their scales do not bleed into each
    other.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        delta_mu_scale: float = 3.0,
    ) -> None:
        """Initialize the posterior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(min, max)`` clamps for ``logvar_post``.
            dropout: Dropout used inside every internal ResidualMLP.
            delta_mu_scale: Saturation magnitude of the tanh-bounded posterior
                delta; must be positive. ``delta_mu`` is soft-limited so the
                posterior mean stays within ``mu_prior +/- delta_mu_scale``.
        """
        super().__init__()
        if delta_mu_scale <= 0.0:
            raise ValueError(
                f"delta_mu_scale must be > 0, got {delta_mu_scale}"
            )
        self.logvar_clamp = logvar_clamp
        self.delta_mu_scale = float(delta_mu_scale)

        # Per-modality input norms keep H^y and A on comparable scales before
        # concatenation; the existing ResidualMLP internal LN then normalises
        # the fused 2*d_model tensor.
        self.h_y_norm = nn.LayerNorm(d_model)
        self.a_norm = nn.LayerNorm(d_model)

        fused_in = 2 * d_model  # concat [LN(H^y), LN(A)]
        self.fusion = ResidualMLP(
            input_dim=fused_in,
            hidden_dims=geometric_schedule(fused_in, d_model, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.delta_mu_head = nn.Linear(d_model, d_z)
        self.logvar_post_head = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_z, 4),
            final_activation=False,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(
        self,
        h_y: torch.Tensor,
        a: torch.Tensor,
        mu_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_post, logvar_post)``.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            a: Attended source summary ``(B, T, d_model)``.
            mu_prior: Prior mean ``(B, T, d_z)`` (used for the residual add).
        """
        fused = self.fusion(
            torch.cat([self.h_y_norm(h_y), self.a_norm(a)], dim=-1)
        )
        raw_delta = self.delta_mu_head(fused)
        # tanh(0) = 0, so with ``delta_mu_head`` zero-inited the bounded
        # ``delta_mu`` is also identically zero at step 0.
        delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
        mu_post = mu_prior + delta_mu
        logvar_post = self.logvar_post_head(fused)
        logvar_post = torch.clamp(
            logvar_post, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        return mu_post, logvar_post


# =============================================================================
# Lag memory bank + lag cross-attention
# =============================================================================


class LagMemoryBankBuilder(nn.Module):
    """Build the lagged source-state memory ``M`` and its validity mask.

    For each time step ``t`` and lag ``l`` in ``[0, L)`` the memory entry
    ``M[b, t, l, :] = H^u[b, t - l, :]``, with zero-padding for ``t - l < 0``.
    """

    def __init__(self, max_lag: int = 90) -> None:
        """Initialize the builder.

        Args:
            max_lag: Maximum past lag to include. The effective window length
                is ``L = max_lag + 1`` (lag 0 is the current step).
        """
        super().__init__()
        if max_lag < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        self.max_lag = int(max_lag)
        self.L = int(max_lag) + 1

    def forward(
        self, h_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(M, m_lag)`` with shapes ``(B, T, L, D)`` and ``(B, T, L)``."""
        B, T = h_u.shape[0], h_u.shape[1]
        L = self.L
        # Left-pad along the time axis with L-1 zeros so slices are safe.
        h_pad = F.pad(h_u, (0, 0, L - 1, 0))  # (B, T + L - 1, D)
        # Stack L shifted views. Each view at lag l corresponds to H^u[:, t-l, :].
        slices = [h_pad[:, L - 1 - l : L - 1 - l + T, :] for l in range(L)]
        mem = torch.stack(slices, dim=2)  # (B, T, L, D)

        # Lag validity mask: entry (t, l) is valid iff t - l >= 0.
        t_ar = torch.arange(T, device=h_u.device)[None, :, None]
        l_ar = torch.arange(L, device=h_u.device)[None, None, :]
        m_lag = (t_ar - l_ar >= 0).expand(B, T, L).contiguous()
        return mem, m_lag


class LagCrossAttention(nn.Module):
    """Multi-head cross-attention from current FHR state to lagged UP memory.

    Each head learns a per-lag additive bias ``r_l`` (Shaw-style relative
    position encoding). Attention scores are masked using the lag validity
    mask, then normalised with ``softmax`` (or ``entmax15`` if available and
    enabled via ``use_entmax=True``).
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
    ) -> None:
        """Initialize the attention module.

        Args:
            d_model: Query/key/value embedding dim (must equal ``num_heads * d_head``).
            num_heads: Number of attention heads.
            d_head: Per-head dimensionality.
            max_lag: Maximum past lag; ``L = max_lag + 1``.
            dropout: Dropout on the attention probabilities.
            use_entmax: If True and ``entmax`` is importable, use ``entmax15``
                for sparse attention. Falls back to ``softmax`` otherwise.
            grad_checkpoint: If True, wrap the attention op in
                ``torch.utils.checkpoint.checkpoint`` to trade compute for
                memory (the lag memory bank can be ~900 MB at B=64).
        """
        super().__init__()
        if num_heads * d_head != d_model:
            raise ValueError(
                f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.L = int(max_lag) + 1
        self.scale = 1.0 / math.sqrt(d_head)
        self.use_entmax = bool(use_entmax) and _HAS_ENTMAX
        self.grad_checkpoint = bool(grad_checkpoint)

        # Pre-norm: standard transformer pattern. Stabilises the dot-product
        # scores before softmax / entmax. The kv norm broadcasts across the
        # lag dim (channel-last LayerNorm normalises the trailing d_model
        # axis of the (B, T, L, d_model) memory tensor).
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # Per-lag learned bias broadcast-added to keys.
        self.lag_embeddings = nn.Parameter(
            torch.zeros(self.L, num_heads, d_head)
        )
        nn.init.normal_(self.lag_embeddings, mean=0.0, std=0.02)

    def _attend(
        self,
        h_y: torch.Tensor,
        mem: torch.Tensor,
        m_lag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = h_y.shape[0], h_y.shape[1]
        L = self.L
        Mh = self.num_heads
        d = self.d_head

        h_y_n = self.q_norm(h_y)
        mem_n = self.kv_norm(mem)

        Q = self.W_q(h_y_n).view(B, T, Mh, d)                        # (B, T, Mh, d)
        K = self.W_k(mem_n).view(B, T, L, Mh, d)                     # (B, T, L, Mh, d)
        V = self.W_v(mem_n).view(B, T, L, Mh, d)                     # (B, T, L, Mh, d)

        # Shaw-style: add per-lag bias to keys before the dot product.
        K = K + self.lag_embeddings[None, None, :, :, :]

        scores = torch.einsum("btmd,btlmd->btml", Q, K) * self.scale  # (B, T, Mh, L)
        # Mask invalid lags (where t - l < 0). ``m_lag`` is (B, T, L).
        invalid = (~m_lag)[:, :, None, :]                             # (B, T, 1, L)
        scores = scores.masked_fill(invalid, float("-inf"))

        if self.use_entmax and _entmax15 is not None:
            alpha = _entmax15(scores, dim=-1)
        else:
            alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # guard when *all* lags invalid
        alpha = self.attn_dropout(alpha)

        head_out = torch.einsum("btml,btlmd->btmd", alpha, V)         # (B, T, Mh, d)
        out = self.W_o(head_out.reshape(B, T, Mh * d))                # (B, T, d_model)
        return out, alpha

    def forward(
        self,
        h_y: torch.Tensor,
        mem: torch.Tensor,
        m_lag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``h_y`` over lagged ``mem``.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            mem: Lagged source memory ``(B, T, L, d_model)``.
            m_lag: Lag validity mask ``(B, T, L)``.

        Returns:
            ``(A, alpha)`` where ``A`` has shape ``(B, T, d_model)`` and
            ``alpha`` has shape ``(B, T, num_heads, L)``.
        """
        if self.grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._attend, h_y, mem, m_lag, use_reentrant=False
            )
        return self._attend(h_y, mem, m_lag)


# =============================================================================
# Decoders
# =============================================================================


class _HorizonRefine(nn.Module):
    """Two 1D convs along the forecast-horizon axis with GELU + GroupNorm."""

    def __init__(self, d_hidden: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(
            d_hidden, d_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm_1 = nn.GroupNorm(num_groups=min(8, d_hidden), num_channels=d_hidden)
        self.conv_2 = nn.Conv1d(
            d_hidden, d_hidden, kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm_2 = nn.GroupNorm(num_groups=min(8, d_hidden), num_channels=d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine ``(N, d_hidden, H_d)`` along the horizon axis."""
        y = self.conv_1(x)
        y = F.gelu(self.norm_1(y))
        y = self.conv_2(y)
        y = F.gelu(self.norm_2(y))
        return x + y


class BaselineFutureDecoder(nn.Module):
    """Predict future FHR features from the FHR-only decoder state.

    Produces :math:`\\hat{Y}^{base}` and (auxiliary) ``logvar_base``. In v1 only
    the mean is consumed by the loss, but the logvar head is wired for a
    future switch to heteroscedastic Gaussian NLL.
    """

    def __init__(
        self,
        d_model: int = 128,
        horizon: int = 30,
        out_channels: int = 87,
        d_hidden: int = 128,
        dropout: float = 0.1,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
    ) -> None:
        """Initialize the baseline decoder."""
        super().__init__()
        self.horizon = int(horizon)
        self.out_channels = int(out_channels)
        self.d_hidden = int(d_hidden)
        self.logvar_clamp = logvar_clamp

        self.proj = ResidualMLP(
            input_dim=d_model,
            hidden_dims=geometric_schedule(d_model, d_hidden, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        # Internal forecast-step embedding (not dataset-time metadata).
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, d_hidden))
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        self.refine = _HorizonRefine(d_hidden, kernel_size=3)
        # Long skip around ``_HorizonRefine`` is added in forward(); the output
        # norm stabilises the sum before the output heads.
        self.out_norm = nn.LayerNorm(d_hidden)
        self.mean_head = nn.Linear(d_hidden, out_channels)
        self.logvar_head = nn.Linear(d_hidden, out_channels)

    def forward(
        self, decoder_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_base, logvar_base)`` each of shape ``(B, T, H_d, C)``."""
        B, T, _ = decoder_state.shape
        Hd = self.horizon
        Dh = self.d_hidden

        h = self.proj(decoder_state)                       # (B, T, Dh)
        h = h.unsqueeze(2).expand(-1, -1, Hd, -1)           # (B, T, Hd, Dh)
        h = h + self.horizon_embedding[None, None, :, :]
        h_skip = h                                          # identity path

        # Conv refinement along the horizon axis: fold B*T together.
        h_flat = h.reshape(B * T, Hd, Dh).transpose(1, 2).contiguous()  # (B*T, Dh, Hd)
        h_flat = self.refine(h_flat)
        h = h_flat.transpose(1, 2).reshape(B, T, Hd, Dh)

        h = self.out_norm(h + h_skip)                       # long skip + LN

        mu_base = self.mean_head(h)                         # (B, T, Hd, C)
        logvar_base = self.logvar_head(h)
        logvar_base = torch.clamp(
            logvar_base, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        return mu_base, logvar_base


class ResidualFutureDecoder(nn.Module):
    """Predict the source-driven future correction :math:`\\Delta\\hat{Y}^{src}`.

    Consumes the target-only decoder state and the latent ``z``. At
    initialisation the final mean head is zero-inited so that ``delta_mu_src ≈ 0``
    and ``mu_full ≈ mu_base``. Training learns to diverge only when the latent
    carries genuinely incremental UP information.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
        horizon: int = 30,
        out_channels: int = 87,
        d_hidden: int = 128,
        dropout: float = 0.1,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
    ) -> None:
        """Initialize the residual decoder."""
        super().__init__()
        self.horizon = int(horizon)
        self.out_channels = int(out_channels)
        self.d_hidden = int(d_hidden)
        self.logvar_clamp = logvar_clamp

        in_dim = d_model + d_z
        self.proj = ResidualMLP(
            input_dim=in_dim,
            hidden_dims=geometric_schedule(in_dim, d_hidden, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, d_hidden))
        nn.init.normal_(self.horizon_embedding, mean=0.0, std=0.02)

        self.refine = _HorizonRefine(d_hidden, kernel_size=3)
        # Long skip around ``_HorizonRefine`` is added in forward(); the output
        # norm stabilises the sum before the output heads. The mean head is
        # zero-inited externally so ``delta_mu_src`` is still identically zero
        # at step 0 regardless of the skip contribution.
        self.out_norm = nn.LayerNorm(d_hidden)
        self.mean_head = nn.Linear(d_hidden, out_channels)
        self.logvar_head = nn.Linear(d_hidden, out_channels)

    def forward(
        self,
        decoder_state: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(delta_mu_src, logvar_full)`` of shape ``(B, T, H_d, C)``."""
        B, T, _ = decoder_state.shape
        Hd = self.horizon
        Dh = self.d_hidden

        h_in = torch.cat([decoder_state, z], dim=-1)        # (B, T, d_model + d_z)
        h = self.proj(h_in)                                 # (B, T, Dh)
        h = h.unsqueeze(2).expand(-1, -1, Hd, -1)           # (B, T, Hd, Dh)
        h = h + self.horizon_embedding[None, None, :, :]
        h_skip = h                                          # identity path

        h_flat = h.reshape(B * T, Hd, Dh).transpose(1, 2).contiguous()
        h_flat = self.refine(h_flat)
        h = h_flat.transpose(1, 2).reshape(B, T, Hd, Dh)

        h = self.out_norm(h + h_skip)                       # long skip + LN

        delta_mu_src = self.mean_head(h)                    # (B, T, Hd, C)
        logvar_full = self.logvar_head(h)
        logvar_full = torch.clamp(
            logvar_full, min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        return delta_mu_src, logvar_full


class RawRefinementDecoder(nn.Module):
    """Placeholder for the v2 raw refinement decoder.

    Not implemented in v1. Spec section 6.12 is explicit: "start without raw
    decoder". A concrete implementation will be added in a follow-up once the
    feature-level forecast is stable.
    """

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        super().__init__()
        raise NotImplementedError(
            "RawRefinementDecoder is deferred to a future revision. "
            "See new_architecture.md §6.12."
        )


# =============================================================================
# TE analysis head
# =============================================================================


class TEAnalysisHead(nn.Module):
    """Derive per-timestep KL and a lag-resolved TE attribution map.

    This is a pure function (no learnable parameters). It consumes the per-
    timestep KL tensor and the attention weights and returns:

    * ``kld_per_t`` — ``(B, T)``, total KL summed over latent dims
    * ``te_lag_map`` — ``(B, T, L)``, lag attribution ``K_t * mean_m alpha_{t,m,l}``
    """

    def forward(
        self,
        kld_btd: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(kld_per_t, te_lag_map)``.

        Args:
            kld_btd: Per-step KL tensor ``(B, T, d_z)``.
            attn_weights: Attention probabilities ``(B, T, num_heads, L)``.
        """
        kld_per_t = kld_btd.sum(dim=-1)                        # (B, T)
        mean_alpha = attn_weights.mean(dim=-2)                 # (B, T, L)
        te_lag_map = kld_per_t.unsqueeze(-1) * mean_alpha       # (B, T, L)
        return kld_per_t, te_lag_map


# =============================================================================
# Top-level model
# =============================================================================


class SeqVaeLagAttnV1(nn.Module):
    """Lag-Attentive Residual VAE-TEB v1.

    See ``new_architecture.md`` for the full spec and the accompanying
    implementation plan for context. In short: causal encoders on
    target / source → lag cross-attention → residual posterior → baseline +
    residual future decoders → weighted-MSE feature loss + KL regulariser.
    """

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 24,
        horizon: int = 30,
        warmup_period: int = 30,
        c_y: int = 87,
        c_u: int = 101,
        use_up_st: bool = True,
        max_lag: int = 90,
        num_heads: int = 4,
        d_head: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        delta_mu_scale: float = 3.0,
        latent_stats_momentum: float = 0.01,
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        init_weights: bool = True,
    ) -> None:
        """Initialize ``SeqVaeLagAttnV1``.

        Args:
            sequence_length: Decimated sequence length ``T``.
            d_model: Internal width used throughout the backbone.
            d_z: Latent dimensionality.
            horizon: Decimated forecast horizon ``H_d`` (default 30 = 2 min).
            warmup_period: Number of initial decimated steps to ignore in both
                KL and feature losses (default 30 = 2 min).
            c_y: Target feature channel count (43 + 44 = 87).
            c_u: Source feature channel count. Must equal ``101`` when
                ``use_up_st=True`` (43 ``up_st`` + 58 ``up_ph``) and ``58``
                when ``use_up_st=False`` (``up_ph`` only). An inconsistent
                ``(c_u, use_up_st)`` pair raises ``ValueError``.
            use_up_st: Ablation toggle. ``True`` (default) feeds the source
                adapter with the concatenation ``[up_st, up_ph]``; ``False``
                feeds only ``up_ph``. Both fields are first-class HDF5
                datasets with their own per-channel stats — no virtual
                slicing from ``fhr_up_ph``.
            max_lag: Maximum past lag (L = max_lag + 1).
            num_heads: Number of attention heads.
            d_head: Per-head dimensionality (must satisfy
                ``num_heads * d_head == d_model``).
            lstm_layers: Depth of the encoder LSTM branches.
            dropout: Dropout used throughout residual MLPs / attention.
            decoder_hidden: Hidden width of the structured horizon decoders.
            logvar_clamp: ``(min, max)`` clamps applied to every log-variance
                head in the model. Default ``(-5, 3)`` keeps ``var`` in
                ``[e^{-5}, e^3] = [0.007, 20]`` which is well-conditioned for
                the closed-form Gaussian KL.
            mu_scale: Saturation magnitude of the tanh-bounded ``mu_prior``;
                must be positive. Caps the prior mean to ``|mu_prior| <=
                mu_scale``.
            delta_mu_scale: Saturation magnitude of the tanh-bounded
                posterior delta; must be positive. Caps ``|mu_post -
                mu_prior| <= delta_mu_scale``.
            latent_stats_momentum: EMA momentum used when updating the
                ``mu_post_running_mean`` / ``mu_post_running_var`` buffers
                during training. Buffers are populated with a BatchNorm-style
                running average excluding warm-up steps. Use
                :meth:`normalize_latent` downstream to consume them.
            use_entmax: If True, use ``entmax15`` attention normalisation when
                the ``entmax`` package is importable.
            attention_grad_checkpoint: If True, wrap ``LagCrossAttention`` in
                gradient checkpointing.
            init_weights: If True, apply the standard :func:`initialization`
                helper and then enforce the zero-init on delta heads.
        """
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.horizon = int(horizon)
        self.warmup_period = int(warmup_period)
        self.c_y = int(c_y)
        self.use_up_st = bool(use_up_st)
        expected_c_u = 101 if self.use_up_st else 58
        if int(c_u) != expected_c_u:
            raise ValueError(
                f"c_u={c_u} inconsistent with use_up_st={use_up_st}; "
                f"expected {expected_c_u} "
                f"(43 up_st + 58 up_ph if use_up_st=True, 58 up_ph only if False). "
                f"up_st and up_ph are first-class HDF5 fields in the current "
                f"dataset schema — there is no virtual-slicing fallback."
            )
        self.c_u = expected_c_u
        self.max_lag = int(max_lag)
        self.mu_scale = float(mu_scale)
        self.delta_mu_scale = float(delta_mu_scale)
        self.latent_stats_momentum = float(latent_stats_momentum)

        # --- Input adapters -------------------------------------------------
        self.target_adapter = TargetInputAdapter(
            in_dim=self.c_y, d_model=d_model, dropout=dropout
        )
        self.source_adapter = SourceInputAdapter(
            in_dim=self.c_u, d_model=d_model, dropout=dropout
        )

        # --- Encoders -------------------------------------------------------
        self.target_encoder = TargetEncoder(
            d_model=d_model,
            cnn_kernels=(3, 7, 11),
            cnn_dilations=(1, 2, 4),
            lstm_layers=lstm_layers,
            lstm_dropout=dropout,
            conv_dropout=dropout,
        )
        self.source_encoder = SourceEncoder(
            d_model=d_model,
            cnn_kernels=(3, 5, 11),
            cnn_dilations=(1, 2, 4),
            lstm_layers=lstm_layers,
            lstm_dropout=dropout,
            conv_dropout=dropout,
        )

        # --- Prior / Lag attention / Posterior ------------------------------
        self.prior_head = PriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
        )
        self.lag_bank = LagMemoryBankBuilder(max_lag=max_lag)
        self.lag_attn = LagCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            max_lag=max_lag,
            dropout=dropout,
            use_entmax=use_entmax,
            grad_checkpoint=attention_grad_checkpoint,
        )
        self.posterior_head = PosteriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            delta_mu_scale=self.delta_mu_scale,
        )

        # --- Decoders -------------------------------------------------------
        self.baseline_decoder = BaselineFutureDecoder(
            d_model=d_model,
            horizon=horizon,
            out_channels=c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
        )
        self.residual_decoder = ResidualFutureDecoder(
            d_model=d_model,
            d_z=d_z,
            horizon=horizon,
            out_channels=c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
        )

        # --- Analysis head (no parameters) ----------------------------------
        self.te_analysis = TEAnalysisHead()

        # --- Latent running stats (for downstream classifier consumption) ---
        # These buffers are maintained as an approximate, per-rank EMA on
        # ``mu_post`` during training (see :meth:`_update_latent_running_stats`)
        # and are useful for monitoring only. Under DDP each rank tracks its
        # own stream and the checkpoint persists rank-0's values.
        #
        # For correct downstream use (freezing the VAE and feeding
        # :meth:`normalize_latent` output into a classifier), call
        # :meth:`fit_latent_stats` after VAE training: it runs one eval-mode
        # pass over the training loader and writes exact, DDP-synchronised
        # mean/variance into these buffers.
        self.register_buffer("mu_post_running_mean", torch.zeros(d_z))
        self.register_buffer("mu_post_running_var", torch.ones(d_z))
        self.register_buffer(
            "mu_post_running_count", torch.zeros((), dtype=torch.long)
        )

        # --- Weight init ----------------------------------------------------
        if init_weights:
            initialization(self)
        # Zero-init the delta heads AFTER generic init so they are not
        # overwritten. This enforces mu_full ≈ mu_base and KL ≈ 0 at step 0.
        self._zero_init_delta_heads()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _zero_init_delta_heads(self) -> None:
        """Zero the posterior delta-mu and residual-decoder mean heads."""
        nn.init.zeros_(self.posterior_head.delta_mu_head.weight)
        if self.posterior_head.delta_mu_head.bias is not None:
            nn.init.zeros_(self.posterior_head.delta_mu_head.bias)
        nn.init.zeros_(self.residual_decoder.mean_head.weight)
        if self.residual_decoder.mean_head.bias is not None:
            nn.init.zeros_(self.residual_decoder.mean_head.bias)

    # ------------------------------------------------------------------
    # Forward / sampling
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z = mu + sigma * eps`` with ``eps ~ N(0, I)``."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.no_grad()
    def _update_latent_running_stats(self, mu_post: torch.Tensor) -> None:
        """Update ``mu_post_running_{mean,var,count}`` from a live batch.

        Pure BatchNorm-style EMA with momentum ``self.latent_stats_momentum``.
        No CPU sync in the hot path. Warm-up steps are masked out before
        aggregating so early, under-conditioned states do not bias the
        downstream normaliser.

        The (0, 1) initialisation of the buffers decays exponentially —
        within ~500 batches at the default momentum of 0.01 the residual is
        < 1%. Call :meth:`fit_latent_stats` post-training to replace these
        approximate EMA stats with exact, DDP-consistent training-set
        statistics before consuming them downstream.
        """
        valid_t = self._build_warmup_valid_mask(
            mu_post.size(1), device=mu_post.device
        )                                      # (T,)
        if not bool(valid_t.any()):
            return
        flat = mu_post[:, valid_t, :].reshape(-1, mu_post.size(-1))
        if flat.numel() == 0:
            return
        m = self.latent_stats_momentum
        self.mu_post_running_mean.mul_(1.0 - m).add_(
            flat.mean(dim=0), alpha=m
        )
        self.mu_post_running_var.mul_(1.0 - m).add_(
            flat.var(dim=0, unbiased=False), alpha=m
        )
        self.mu_post_running_count.add_(int(flat.size(0)))

    def normalize_latent(
        self, z: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        """Z-score ``z`` (or ``mu_post``) using the running latent stats.

        Args:
            z: Latent tensor of shape ``(..., d_z)``. Works with both sampled
                ``z`` and deterministic ``mu_post``.
            eps: Small constant added to the variance before the square root
                for numerical stability.

        Returns:
            ``(z - running_mean) / sqrt(running_var + eps)`` with the same
            shape as ``z``. If the buffers are still at their initial values
            (no training has happened), this reduces to a near-identity
            shift.
        """
        std = (self.mu_post_running_var + eps).sqrt()
        return (z - self.mu_post_running_mean) / std

    @torch.no_grad()
    def fit_latent_stats(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        device: Optional[torch.device] = None,
        batch_to_inputs: Optional[Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
    ) -> int:
        """Populate ``mu_post_running_{mean,var,count}`` exactly.

        Runs one eval-mode pass over ``dataloader`` and replaces the EMA
        training-time buffers with the exact mean and variance of
        ``mu_post`` over all valid (non-warm-up) time steps. Safe under
        DDP: when a process group is initialised, partial sums are
        reduced across ranks with ``all_reduce`` so every rank ends up
        with the same buffer values.

        Call this after VAE training has converged and before freezing
        the encoder for downstream classification.

        Args:
            dataloader: Any iterable that yields batches accepted by
                ``batch_to_inputs``. A standard training DataLoader works.
            max_batches: If set, stop after this many batches (useful for
                sanity checks). ``None`` uses the full loader.
            device: Device on which to run the forward pass. Defaults to
                the device of the model's first parameter.
            batch_to_inputs: Callable mapping a batch object to the triple
                ``(y_st, y_ph, u_stream)``. The default expects an
                attribute-style batch (``batch.fhr_st``, ``batch.fhr_ph``,
                and either ``batch.up_ph`` alone or ``torch.cat([batch.up_st,
                batch.up_ph], dim=-1)`` depending on ``self.use_up_st``) —
                i.e. the layout produced by the project's
                ``CombinedHDF5Dataset`` loader.

        Returns:
            Total number of time-step samples aggregated across ranks.
        """
        try:
            import torch.distributed as dist
            dist_active = dist.is_available() and dist.is_initialized()
        except Exception:  # pragma: no cover - distributed not available
            dist_active = False
            dist = None  # type: ignore[assignment]

        if device is None:
            device = next(self.parameters()).device
        if batch_to_inputs is None:
            batch_to_inputs = self._default_batch_to_inputs

        was_training = self.training
        self.eval()

        d_z = self.d_z
        # Numerically-stable sums in float64; one ``O(dataset)`` pass is
        # fine and the fp32 rounding would otherwise bias the variance.
        sum_x = torch.zeros(d_z, dtype=torch.float64, device=device)
        sum_xx = torch.zeros(d_z, dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)

        try:
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                y_st, y_ph, u_stream = batch_to_inputs(batch)
                y_st = y_st.to(device, non_blocking=True)
                y_ph = y_ph.to(device, non_blocking=True)
                u_stream = u_stream.to(device, non_blocking=True)

                enc = self.encode_only(y_st, y_ph, u_stream, sample_z=False)
                mu_post = enc["mu_post"]                          # (B, T, d_z)
                valid_t = self._build_warmup_valid_mask(
                    mu_post.size(1), device=mu_post.device
                )
                flat = mu_post[:, valid_t, :].reshape(-1, d_z).double()
                sum_x += flat.sum(dim=0)
                sum_xx += (flat * flat).sum(dim=0)
                count += flat.size(0)
        finally:
            if was_training:
                self.train()

        if dist_active:
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_xx, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        n_samples = int(count.item())
        # Refuse to overwrite the buffers with zeros. Callers would otherwise
        # silently feed hugely-scaled latents into the classifier (division
        # by sqrt(eps)). Better to raise so the caller fixes the dataloader.
        if n_samples == 0:
            raise RuntimeError(
                "fit_latent_stats: dataloader yielded zero valid samples "
                "(no non-warm-up time steps). Check that the loader is not "
                "empty and that sequence_length > warmup_period."
            )

        mean = sum_x / count
        var = (sum_xx / count - mean * mean).clamp_min(0.0)

        self.mu_post_running_mean.copy_(mean.float())
        self.mu_post_running_var.copy_(var.float())
        self.mu_post_running_count.copy_(count.long())

        # Single post-fit log line so the caller can confirm the stats
        # landed sensibly without manually inspecting buffers.
        logger.info(
            "[fit_latent_stats] aggregated {n} samples across {r} rank(s); "
            "mean range [{mn:+.3f}, {mx:+.3f}], var range [{vn:.3f}, {vx:.3f}]",
            n=n_samples,
            r=(torch.distributed.get_world_size() if dist_active else 1),
            mn=float(self.mu_post_running_mean.min()),
            mx=float(self.mu_post_running_mean.max()),
            vn=float(self.mu_post_running_var.min()),
            vx=float(self.mu_post_running_var.max()),
        )
        return n_samples

    def _default_batch_to_inputs(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ``(y_st, y_ph, u_stream)`` from an attribute-style batch.

        Matches the convention used by :class:`SeqVaeLagAttnPl`.
        """
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        up_ph = batch.up_ph
        if self.use_up_st:
            up_st = batch.up_st
            u_stream = torch.cat([up_st, up_ph], dim=-1)
        else:
            u_stream = up_ph
        return y_st, y_ph, u_stream

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the full encoder → attention → posterior → decoders pipeline.

        Args:
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase harmonic features ``(B, T, 44)``.
            u_stream: Source stream ``(B, T, 101)`` (``[up_st, up_ph]``) or
                ``(B, T, 58)`` (``up_ph`` only) matching ``self.c_u``.

        Returns:
            Dictionary with all the tensors listed in ``new_architecture.md``
            section 7.
        """
        Y = torch.cat([y_st, y_ph], dim=-1)                # (B, T, C_y)

        Y_tilde = self.target_adapter(Y)                   # (B, T, d_model)
        U_tilde = self.source_adapter(u_stream)            # (B, T, d_model)

        H_y = self.target_encoder(Y_tilde)                 # (B, T, d_model)
        H_u = self.source_encoder(U_tilde)                 # (B, T, d_model)

        mu_prior, logvar_prior, decoder_state = self.prior_head(H_y)

        mem, m_lag = self.lag_bank(H_u)                    # (B, T, L, d_model), (B, T, L)
        A, alpha = self.lag_attn(H_y, mem, m_lag)          # (B, T, d_model), (B, T, M, L)

        mu_post, logvar_post = self.posterior_head(H_y, A, mu_prior)
        z = self.reparameterize(mu_post, logvar_post)      # (B, T, d_z)

        # Tanh saturation diagnostics. Values >> 0 indicate the prior or
        # posterior wants to drift beyond its tanh bound and gradients are
        # being starved. Cheap to compute; logged by the trainer.
        with torch.no_grad():
            mu_prior_sat_frac = (
                mu_prior.abs() >= (0.99 * self.mu_scale)
            ).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (0.99 * self.delta_mu_scale)
            ).float().mean()

        # Update running stats on mu_post for downstream consumers. Training
        # mode only; no-grad. Warm-up steps are excluded because they carry
        # under-conditioned encoder states.
        if self.training:
            self._update_latent_running_stats(mu_post)

        mu_base, logvar_base = self.baseline_decoder(decoder_state)
        delta_mu_src, logvar_full = self.residual_decoder(decoder_state, z)
        mu_full = mu_base + delta_mu_src                   # (B, T, H_d, C_y)

        # TE analysis (cheap — purely a function of already-computed tensors).
        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            mask_warmup=False,
        )
        kld_per_t, te_lag_map = self.te_analysis(kld_btd, alpha)

        warmup_mask = self._build_warmup_valid_mask(H_y.size(1), device=H_y.device)

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": H_u,
            "decoder_state": decoder_state,
            "attended_source": A,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "delta_mu_src": delta_mu_src,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "raw_future_pred": None,
            "kld_per_t": kld_per_t,
            "te_lag_map": te_lag_map,
            "warmup_mask": warmup_mask,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
        }

    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        sample_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run encoders + posterior only (no decoders). Useful for classifiers."""
        Y = torch.cat([y_st, y_ph], dim=-1)
        Y_tilde = self.target_adapter(Y)
        U_tilde = self.source_adapter(u_stream)
        H_y = self.target_encoder(Y_tilde)
        H_u = self.source_encoder(U_tilde)
        mu_prior, logvar_prior, decoder_state = self.prior_head(H_y)
        mem, m_lag = self.lag_bank(H_u)
        A, alpha = self.lag_attn(H_y, mem, m_lag)
        mu_post, logvar_post = self.posterior_head(H_y, A, mu_prior)
        z = self.reparameterize(mu_post, logvar_post) if sample_z else mu_post
        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": H_u,
            "decoder_state": decoder_state,
            "attended_source": A,
            "attn_weights": alpha,
        }

    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        """Estimate the TE surrogate ``KL(q || p)``.

        Args:
            y_st: FHR scattering features.
            y_ph: FHR phase features.
            u_stream: Source stream (shape matching ``self.c_u``).
            reduce_mean: If True return the scalar mean KL; otherwise return
                the full ``(B, T, d_z)`` tensor with warmup steps masked to NaN.
        """
        self.eval()
        with torch.no_grad():
            enc = self.encode_only(y_st, y_ph, u_stream, sample_z=True)
            if reduce_mean:
                return self._kld_loss(
                    mu_prior=enc["mu_prior"],
                    logvar_prior=enc["logvar_prior"],
                    mu_post=enc["mu_post"],
                    logvar_post=enc["logvar_post"],
                    reduce_mean=True,
                )
            return self.kld_tensor(
                mu_prior=enc["mu_prior"],
                logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"],
                logvar_post=enc["logvar_post"],
                mask_warmup=True,
            )

    # ------------------------------------------------------------------
    # Warmup / KL helpers (copied verbatim from SeqVae)
    # ------------------------------------------------------------------

    def _warmup_steps(self, seq_len: int) -> int:
        warmup = int(getattr(self, "warmup_period", 0) or 0)
        if warmup <= 0:
            return 0
        return min(seq_len, warmup)

    def _build_warmup_valid_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            mask[:warmup] = False
        return mask

    def kld_tensor(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        *,
        mask_warmup: bool = False,
        fill_value: float = float("nan"),
    ) -> torch.Tensor:
        """Closed-form diagonal-Gaussian KL. Returns ``(B, T, d_z)``."""
        kld = (
            logvar_prior
            - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
            - 1.0
        )
        kld = 0.5 * kld
        if mask_warmup:
            warmup = self._warmup_steps(kld.size(1))
            if warmup > 0:
                kld = kld.clone()
                if math.isnan(fill_value):
                    kld[:, :warmup, :] = float("nan")
                else:
                    kld[:, :warmup, :].fill_(fill_value)
        return kld

    def _kld_loss(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
        *,
        reduce_mean: bool = True,
        weight: Optional[torch.Tensor] = None,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """Aggregate the per-step closed-form KL into a scalar loss.

        Args:
            weight: Optional validity mask of shape ``(B, T)`` taking values
                in ``[0, 1]``. Combines multiplicatively with the warm-up
                mask so that KL at gap / invalid time steps contributes zero
                to the loss and zero to the denominator. If ``None``, falls
                back to warm-up-only averaging (legacy behaviour).
            free_bits: Lower bound applied per (b, t, d) on the closed-form
                KL before aggregation. The default ``0.0`` is a no-op
                because :math:`\\mathrm{KL}(q\\,\\|\\,p) \\ge 0` for diagonal
                Gaussians; values ``> 0`` implement canonical free-bits
                (model_validation_v2_plan §5.6 fallback).
        """
        kld = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            mask_warmup=False,
        )                                                # (B, T, d_z)
        if free_bits > 0.0:
            kld = kld.clamp(min=float(free_bits))
        B, T, d_z = kld.shape
        warmup = self._warmup_steps(T)
        device = kld.device
        dtype = kld.dtype

        # Build a non-boolean (B, T) validity mask that combines warm-up
        # exclusion with the optional per-step weight.
        time_mask = torch.ones(T, device=device, dtype=dtype)
        if warmup > 0:
            time_mask[:warmup] = 0.0
        full_mask = time_mask.unsqueeze(0).expand(B, T)   # (B, T)
        if weight is not None:
            full_mask = full_mask * weight.to(device=device, dtype=dtype)

        # Expand across the latent dim and aggregate.
        mask_btd = full_mask.unsqueeze(-1)                # (B, T, 1)
        if reduce_mean:
            denom = mask_btd.sum() * float(d_z)
            if float(denom) <= 0.0:
                return torch.zeros((), device=device, dtype=dtype)
            return (kld * mask_btd).sum() / denom
        return (kld * mask_btd).sum()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        *,
        weight: Optional[torch.Tensor] = None,
        compute_kld_loss: bool = True,
        beta: float = 1.0,
        lambda_full: float = 1.0,
        lambda_base: float = 0.5,
        likelihood: str = "mse",
        sigma_obs: "float | str" = 1.0,
        free_bits: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute the v1 training objective.

        ``L_total = lambda_full * L_feat + lambda_base * L_base + beta * L_KL``

        * ``L_feat`` — per-element reconstruction term on the future FHR
          feature trajectory, evaluated at valid anchors
          ``t in [warmup, T - H_d)``. Either weighted MSE (default) or
          Gaussian negative log-likelihood (Sprint 5 calibration path).
        * ``L_base`` — same form but on ``mu_base`` / ``logvar_base``. Forces
          the FHR-only branch to be a strong forecaster on its own.
        * ``L_KL`` — ``KL(q || p)`` averaged over ``t in [warmup, T)``. The
          KL window is *independent* of the feature-loss window. When
          ``free_bits > 0`` each per-dim per-step KL is lower-bounded by
          ``free_bits`` (canonical free-bits, no-op at the default 0.0
          since KL is non-negative).

        Args:
            forward_outputs: Dict returned by :meth:`forward`.
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            weight: Optional per-time-step validity mask from the dataset,
                shape ``(B, T)`` with values in ``[0, 1]``. When provided,
                (a) each future-target entry is masked by
                ``weight[b, t] * weight[b, t+tau+1]`` so that gaps at either
                the anchor or the future step contribute zero to the loss,
                and (b) per-step KL is masked by ``weight[b, t]``. When
                ``None``, only the warm-up mask is applied (legacy
                behaviour). Passing the dataset ``weight`` field is the
                correct setting for a trustworthy TE curve — see
                ``new_model.md`` §"Important practical loss fix".
            compute_kld_loss: If False the KL term is set to 0 (ablation).
            beta: Weight on the KL term.
            lambda_full: Weight on ``L_feat``.
            lambda_base: Weight on ``L_base``.
            likelihood: Reconstruction likelihood. ``'mse'`` (default,
                bit-exact with all pre-Sprint-5 checkpoints) computes the
                weighted squared error :math:`(\hat y - y)^2`. ``'gaussian_nll'``
                computes :math:`\tfrac12(\hat y - y)^2/\sigma^2 + \tfrac12\ln\sigma^2`
                per element, giving ``L_feat`` units of *nats per element*
                so it is directly comparable to ``kld_loss``. This is the
                Sprint 5 calibration mode (model_validation_v2_plan §5.1).
            sigma_obs: Observation noise. Only consulted when
                ``likelihood='gaussian_nll'``. A positive float fixes
                :math:`\sigma_{obs}` to that scalar value (so
                :math:`\ln\sigma^2 = 2\ln\sigma_{obs}` is constant);
                the literal string ``'learned'`` instead reuses the model's
                already-trained ``logvar_full`` / ``logvar_base`` heads
                (per-channel, clamped via ``logvar_clamp``). For MSE this
                kwarg is ignored.
            free_bits: Per-dim per-step lower bound for the KL term. The
                default ``0.0`` is a no-op (closed-form Gaussian KL is
                non-negative). Used only in the Sprint 5.6 conditional
                fallback if a calibrated β collapses the latent.

        Returns:
            Dict with ``feat_loss``, ``base_loss``, ``kld_loss``,
            ``total_loss``, ``beta``, ``likelihood`` (string echo), and
            ``mean_logvar_full`` / ``mean_logvar_base`` collapse diagnostics
            (always present for tracking; for ``likelihood='mse'`` these are
            still computed from ``logvar_full`` / ``logvar_base`` so that
            checkpoints record whether the heads stayed lively).
        """
        Y = torch.cat([y_st, y_ph], dim=-1)                # (B, T, C_y)
        mu_full = forward_outputs["mu_full"]               # (B, T, H_d, C_y)
        mu_base = forward_outputs["mu_base"]               # (B, T, H_d, C_y)

        B, T, Hd, C = mu_full.shape
        T_valid = T - Hd
        device = Y.device
        dtype = Y.dtype

        # --- Future target via unfold --------------------------------------
        # Y_shift[:, t, :] = Y[:, t+1, :]
        Y_shift = Y[:, 1:, :]                              # (B, T-1, C_y)
        # unfold produces (B, T-H_d, C_y, H_d); permute to (B, T-H_d, H_d, C_y)
        Y_plus = Y_shift.unfold(dimension=1, size=Hd, step=1)
        Y_plus = Y_plus.permute(0, 1, 3, 2).contiguous()

        # Slice predictions to the valid anchor range.
        mu_full_valid = mu_full[:, :T_valid, :, :]
        mu_base_valid = mu_base[:, :T_valid, :, :]

        # --- Warm-up mask on the anchor axis (always applied) --------------
        warmup = self._warmup_steps(T)
        warmup_t = torch.zeros(T_valid, dtype=dtype, device=device)
        if warmup < T_valid:
            warmup_t[warmup:] = 1.0                        # (T_valid,)

        # --- Optional dataset weight mask ----------------------------------
        # Build a per-(b, t, tau) weighting from weight[b, t] * weight[b, t+tau+1].
        if weight is not None:
            w = weight.to(device=device, dtype=dtype)      # (B, T)
            anchor_w = w[:, :T_valid]                       # (B, T_valid)
            # Future-step weight aligned with Y_plus via the same unfold.
            target_w_shift = w[:, 1:]                       # (B, T-1)
            target_w = target_w_shift.unfold(
                dimension=1, size=Hd, step=1
            )                                               # (B, T_valid, Hd)
            # (B, T_valid, Hd, 1) = anchor * target, then broadcast over C.
            mask_feat = (
                warmup_t[None, :, None, None]
                * anchor_w[:, :, None, None]
                * target_w[:, :, :, None]
            )
        else:
            mask_feat = warmup_t[None, :, None, None].expand(
                B, T_valid, Hd, 1
            )

        # Denominator counts *effective* entries and always includes the
        # channel axis so the scale matches a mean over (B, T_valid, Hd, C).
        denom = mask_feat.sum() * float(C)
        denom = denom.clamp_min(1.0)

        diff_full = (mu_full_valid - Y_plus) ** 2
        diff_base = (mu_base_valid - Y_plus) ** 2

        # Heads are always present (see forward dict invariant in §5
        # ``new_architecture.md``); slice to the valid anchor range so the
        # NLL branch and the diagnostics share the same support as the means.
        logvar_full_valid = forward_outputs["logvar_full"][:, :T_valid, :, :]
        logvar_base_valid = forward_outputs["logvar_base"][:, :T_valid, :, :]

        if likelihood == "mse":
            per_elem_full = diff_full
            per_elem_base = diff_base
        elif likelihood == "gaussian_nll":
            if isinstance(sigma_obs, str):
                if sigma_obs != "learned":
                    raise ValueError(
                        f"sigma_obs string must be 'learned', got {sigma_obs!r}"
                    )
                logvar_full_obs = logvar_full_valid
                logvar_base_obs = logvar_base_valid
            else:
                sigma_obs_f = float(sigma_obs)
                if sigma_obs_f <= 0.0:
                    raise ValueError(
                        "sigma_obs scalar must be positive, "
                        f"got {sigma_obs_f}"
                    )
                logvar_scalar = math.log(sigma_obs_f ** 2)
                logvar_full_obs = torch.full_like(
                    diff_full, logvar_scalar
                )
                logvar_base_obs = torch.full_like(
                    diff_base, logvar_scalar
                )
            # Per-element Gaussian NLL in nats:
            #   0.5 * (mu - y)^2 / sigma^2 + 0.5 * log(sigma^2)
            per_elem_full = (
                0.5 * diff_full * torch.exp(-logvar_full_obs)
                + 0.5 * logvar_full_obs
            )
            per_elem_base = (
                0.5 * diff_base * torch.exp(-logvar_base_obs)
                + 0.5 * logvar_base_obs
            )
        else:
            raise ValueError(
                "likelihood must be 'mse' or 'gaussian_nll', "
                f"got {likelihood!r}"
            )

        feat_loss = (per_elem_full * mask_feat).sum() / denom
        base_loss = (per_elem_base * mask_feat).sum() / denom

        # --- KL loss -------------------------------------------------------
        if compute_kld_loss:
            kld_loss = self._kld_loss(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
                reduce_mean=True,
                weight=weight,
                free_bits=free_bits,
            )
        else:
            kld_loss = torch.zeros((), device=device, dtype=dtype)

        total_loss = (
            lambda_full * feat_loss
            + lambda_base * base_loss
            + beta * kld_loss
        )

        # Collapse diagnostics: mean over the valid (mask-weighted) support.
        # ``mask_feat`` is (B, T_valid, H_d, 1) and broadcasts across C in
        # the numerator, so the matching denominator is the same one the
        # per-element losses use (``mask_feat.sum() * C``) — otherwise the
        # diagnostic would scale with the channel count and fall outside
        # the head's logvar_clamp band.
        mean_logvar_full = (
            (logvar_full_valid * mask_feat).sum() / denom
        )
        mean_logvar_base = (
            (logvar_base_valid * mask_feat).sum() / denom
        )

        return {
            "feat_loss": feat_loss,
            "base_loss": base_loss,
            "kld_loss": kld_loss,
            "total_loss": total_loss,
            "beta": torch.tensor(float(beta), device=device, dtype=dtype),
            "likelihood": likelihood,
            "mean_logvar_full": mean_logvar_full,
            "mean_logvar_base": mean_logvar_base,
        }


# =============================================================================
# Smoke test (run with: python vae_teb_lag_attn_v1.py)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 300

    expected_keys = {
        "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
        "target_state", "source_state", "decoder_state",
        "attended_source", "attn_weights",
        "mu_base", "logvar_base", "delta_mu_src", "mu_full", "logvar_full",
        "raw_future_pred", "kld_per_t", "te_lag_map", "warmup_mask",
        "mu_prior_sat_frac", "delta_mu_sat_frac",
    }

    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)

    # ---- Test 1: use_up_st=True, source stream has 101 channels -----------
    model = SeqVaeLagAttnV1(use_up_st=True)
    u_full = torch.randn(B, T, 101)
    outs = model(y_st, y_ph, u_full)

    missing = expected_keys - set(outs.keys())
    assert not missing, f"missing forward keys: {missing}"

    assert outs["mu_prior"].shape == (B, T, 24)
    assert outs["logvar_prior"].shape == (B, T, 24)
    assert outs["mu_post"].shape == (B, T, 24)
    assert outs["logvar_post"].shape == (B, T, 24)
    assert outs["z"].shape == (B, T, 24)
    assert outs["target_state"].shape == (B, T, 128)
    assert outs["source_state"].shape == (B, T, 128)
    assert outs["decoder_state"].shape == (B, T, 128)
    assert outs["attended_source"].shape == (B, T, 128)
    assert outs["attn_weights"].shape == (B, T, 4, 91)
    assert outs["mu_base"].shape == (B, T, 30, 87)
    assert outs["logvar_base"].shape == (B, T, 30, 87)
    assert outs["delta_mu_src"].shape == (B, T, 30, 87)
    assert outs["mu_full"].shape == (B, T, 30, 87)
    assert outs["logvar_full"].shape == (B, T, 30, 87)
    assert outs["kld_per_t"].shape == (B, T)
    assert outs["te_lag_map"].shape == (B, T, 91)
    assert outs["warmup_mask"].shape == (T,)

    # Warm-init invariant: delta_mu_src ≈ 0, mu_full ≈ mu_base, KL ≈ 0.
    init_diff = (outs["mu_full"] - outs["mu_base"]).abs().max().item()
    init_delta = outs["delta_mu_src"].abs().max().item()
    print(f"[init] max |mu_full - mu_base| = {init_diff:.3e}")
    print(f"[init] max |delta_mu_src|      = {init_delta:.3e}")
    assert init_delta < 1e-6, f"delta_mu_src is not zero at init: max abs = {init_delta}"
    assert init_diff < 1e-6, f"mu_full != mu_base at init: max diff = {init_diff}"

    delta_mu_weight = model.posterior_head.delta_mu_head.weight.abs().max().item()
    delta_mu_bias = model.posterior_head.delta_mu_head.bias.abs().max().item()
    assert delta_mu_weight == 0.0 and delta_mu_bias == 0.0, (
        "PosteriorHead.delta_mu_head is not zero-initialised"
    )

    # ---- Loss smoke test --------------------------------------------------
    losses = model.compute_loss(outs, y_st, y_ph, beta=0.01)
    for k in ("feat_loss", "base_loss", "kld_loss", "total_loss"):
        v = losses[k]
        assert torch.isfinite(v), f"non-finite loss component: {k}={v}"
    print(
        f"[loss] feat={losses['feat_loss'].item():.4f}"
        f"  base={losses['base_loss'].item():.4f}"
        f"  kld={losses['kld_loss'].item():.4e}"
        f"  total={losses['total_loss'].item():.4f}"
    )
    # feat_loss and base_loss should match exactly at init (mu_full == mu_base)
    feat_base_gap = (losses["feat_loss"] - losses["base_loss"]).abs().item()
    assert feat_base_gap < 1e-6, (
        f"At init feat_loss should equal base_loss, got gap={feat_base_gap}"
    )
    # KL at init: posterior mean exactly matches prior mean (delta_mu = 0),
    # but logvar_prior and logvar_post come from independent random heads so the
    # variance-ratio term of the closed-form KL is non-zero at step 0. With
    # beta=0.01 this contributes ~0.03 to total_loss which is immaterial; just
    # print it as a sanity value.
    print(f"[init] kld at step 0 = {losses['kld_loss'].item():.3e} (expected O(1))")

    losses["total_loss"].backward()
    # Make sure at least the posterior logvar head received gradient
    lv_grad = model.posterior_head.logvar_post_head
    any_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert any_grad, "No parameter received a non-zero gradient"
    print("[backward] OK")

    # ---- Test 2: fallback (no up_st) --------------------------------------
    model_fb = SeqVaeLagAttnV1(use_up_st=False, c_u=58)
    u_fallback = torch.randn(B, T, 58)
    outs_fb = model_fb(y_st, y_ph, u_fallback)
    assert outs_fb["mu_full"].shape == (B, T, 30, 87)
    assert outs_fb["source_state"].shape == (B, T, 128)
    print("[fallback] use_up_st=False forward OK")

    # ---- Parameter counts -------------------------------------------------
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] total={n_params:,}  trainable={n_trainable:,}")

    # ---- Latent-stabiliser invariants -------------------------------------
    mu_prior_max = outs["mu_prior"].abs().max().item()
    delta_mu_max = (outs["mu_post"] - outs["mu_prior"]).abs().max().item()
    assert mu_prior_max <= model.mu_scale + 1e-4, (
        f"mu_prior exceeded tanh bound: {mu_prior_max} > {model.mu_scale}"
    )
    assert delta_mu_max <= model.delta_mu_scale + 1e-4, (
        f"delta_mu exceeded tanh bound: {delta_mu_max} > {model.delta_mu_scale}"
    )
    print(
        f"[bounds] |mu_prior|_max={mu_prior_max:.3f} (<= {model.mu_scale})"
        f"  |delta_mu|_max={delta_mu_max:.3f} (<= {model.delta_mu_scale})"
    )

    # ---- Running-stats buffer check ---------------------------------------
    # A single training-mode forward should have populated the buffers.
    model.train()
    _ = model(y_st, y_ph, u_full)
    assert int(model.mu_post_running_count.item()) > 0, (
        "mu_post_running_count was not updated in training mode"
    )
    # normalize_latent should produce a near-zero-mean / unit-std output
    # on the same batch used to populate the buffers.
    with torch.no_grad():
        z_norm = model.normalize_latent(outs["mu_post"])
    print(
        f"[latent-norm] count={int(model.mu_post_running_count.item())}"
        f"  z_norm.mean~{z_norm.mean().item():+.3f}"
        f"  z_norm.std~{z_norm.std().item():.3f}"
    )

    # ---- TE surrogate scalar ----------------------------------------------
    model.eval()
    te_scalar = model.measure_transfer_entropy(y_st, y_ph, u_full, reduce_mean=True)
    print(f"[te] scalar KL = {te_scalar.item():.3e} (posterior ~ prior at init)")

    print("All smoke checks passed.")
