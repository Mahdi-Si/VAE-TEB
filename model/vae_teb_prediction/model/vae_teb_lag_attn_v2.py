r"""Lag-Attentive Residual VAE-TEB (v2) - ``SeqVaeLagAttnV2``.

This module implements the ELS-VTEB-v2 architecture specified in
``model/vae_teb_prediction/model/vae-teb-lat-attn-v2.md`` (roadmap:
``model/vae_teb_prediction/model/vae-teb-lag-attn-v2-spec-and-sprints.md``). It is a
drop-in successor to :class:`SeqVaeLagAttnV1`
(``model/vae_teb_prediction/model/vae_teb_lag_attn_v1.py``) that makes the lag a
first-class variational object:

* a multi-scale gated depthwise-separable **causal target encoder** replaces the
  v1 conv+LSTM target branch (bounded, interpretable receptive field);
* a **bounded source lag-atom encoder** produces $r^u_s$ whose receptive field is
  at most $21$ decimated steps, so attending at lag $\ell$ isolates the source
  information at lag $\ell$ (unlike v1's LSTM state $h^u_{t-\ell}$);
* an explicit **discrete lag posterior** $\alpha^{(m)}_{t,:}=\operatorname{entmax}_{1.5}(s)$
  per head with a **target-only lag prior** $\pi^{(m)}$;
* a **lag-specific continuous latent** with an exact KL decomposition into a
  lag-selection term $K^R=\operatorname{KL}(\alpha\|\pi)$ and a source-content term
  $K^Z$, so $K_t=\sum_m(K^{R,(m)}+K^{Z,(m)})$.

The transfer-entropy interpretation is preserved because the prior and the
baseline decoder stay target-only and the full reconstruction term detaches the
baseline.

Drop-in contract
----------------
v2 is a mechanical drop-in for v1: consumers reference a canonical alias
``SeqVaeLagAttn`` through a one-line comment-toggle import, so swapping v1 and v2
is a single commented line. v2 exposes every v1 constructor parameter (same names,
same defaults) plus new keyword-only v2 parameters, and emits a **superset** of
the v1 forward dictionary. The v2-to-v1 key mapping is:

======================  ========================================================
v1 forward key          v2 internal quantity
======================  ========================================================
``attn_weights``        $\alpha^{(m)}_{t,\ell}$ lag posterior $(B,T,M,L)$
``te_lag_map``          $K_{t,\ell}$ lag-resolved KL $(B,T,L)$
``kld_per_t``           $K_t=\sum_m K^{(m)}_t$ $(B,T)$
``kld_per_t_per_head``  $K^{(m)}_t=K^{R,(m)}+K^{Z,(m)}$ $(B,T,M)$
``source_state``        $r^u_s$ source lag-atom states $(B,T,d)$
``decoder_state``       $b_t$ baseline conditioning state $(B,T,d)$
``attended_source``     active-$\alpha$-weighted value summary $(B,T,d)$
``attended_source_heads`` per-head active value summary $(B,T,M,d_v)$
``mu_prior``/``mu_post``  per-head prior / mixture-moment posterior $(B,T,d_z)$
======================  ========================================================

New additive v2 keys (``pi_lag``, ``active_lag_indices``, ``mu_prior_heads``,
``mu_post_active``, ``kld_lag``, ``kld_content``, ``expected_lag``, ...) are
appended by the full forward path (Sprint 3).

Shape conventions
-----------------
``B``  batch size;
``T``  decimated sequence length (300 for 20 min @ 4 Hz, stride 16);
``C_y`` FHR feature channels (43 + 44 = 87);
``C_u`` UP feature channels (43 + 58 = 101), or 58 when ``up_st`` is absent;
``d``  internal width ``d_model`` (default 128);
``d_z`` latent dim (default 24), partitioned into ``M`` heads of ``d_z_m = d_z / M``;
``H_d`` decimated forecast horizon (default 30, i.e. 2 min);
``L``  lag window length, ``L = max_lag + 1`` (default 91 = 6 min of history);
``M``  number of heads (``num_heads``, default 4);
``d_k``/``d_v`` lag-attention key/value dims (16 / 32);
``K_a`` active lag count (top-$K_a$; default 8, 16 during warm-up).

Implementation status: this file is built incrementally across Sprints 0-2. The
constructor exposes the full parameter surface from Sprint 0; sub-modules and the
forward/loss paths are wired in Sprints 1-2.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (used by sub-modules in S1-S2)
from loguru import logger

from model.vae_teb_prediction.model.vae_teb_model_prediction import (  # noqa: F401
    ResidualMLP,
    geometric_schedule,
    initialization,
)

# Reuse the v1 horizon decoder core and both future decoders verbatim (they are
# defined in ``vae_teb_lag_attn_v1.py``, NOT in ``vae_teb_model_prediction.py``).
# Wired in S1-T03b (baseline) and S3-T03 (residual).
from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: F401
    BaselineFutureDecoder,
    HorizonDecoderCore,
    ResidualFutureDecoder,
)
from model.vae_teb_prediction.model.outcome_head_v2 import (  # noqa: F401
    OutcomeHead,
    outcome_loss,
)


# =============================================================================
# Sparse attention normalisation: self-contained entmax_1.5 (S0-T02)
# =============================================================================


def _make_ix_like(x: torch.Tensor, dim: int) -> torch.Tensor:
    r"""Return $[1, 2, \ldots, d]$ shaped to broadcast against ``x`` along ``dim``.

    Args:
        x: Reference tensor.
        dim: Axis along which the index ramp is laid out.

    Returns:
        A tensor of shape ``1 x ... x d x ... x 1`` (size ``d`` on ``dim``) with
        ``x``'s dtype and device, holding the support-size candidates $\rho$.
    """
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[dim] = d
    return rho.view(view)


class _Entmax15Fn(torch.autograd.Function):
    r"""Differentiable $\operatorname{entmax}_{1.5}$ (Peters, Niculae, Martins 2019).

    Solves, along ``dim``,
    $$\alpha^\star = \operatorname*{arg\,max}_{p \in \Delta}\ \langle p, x\rangle +
    H_{1.5}(p),\qquad H_{1.5}(p) = \tfrac{1}{1.5(1.5-1)}\sum_i (p_i - p_i^{1.5}),$$
    which admits the exact threshold form $\alpha_i = [\,x_i/2 - \tau^\star\,]_+^2$
    with $\tau^\star$ found by sorting. The map is sparse (produces exact zeros)
    and differentiable; the analytic backward matches the KKT-derived Jacobian.

    Masked entries (non-finite scores, e.g. invalid lags) are replaced by a finite
    sentinel so they receive exactly zero probability; a fully-masked row returns
    all zeros with a finite (zero) gradient rather than a NaN.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int) -> torch.Tensor:  # type: ignore[override]
        r"""Compute $\operatorname{entmax}_{1.5}(x)$ along ``dim``."""
        ctx.dim = dim
        finite = torch.isfinite(x)
        row_all_masked = ~finite.any(dim=dim, keepdim=True)
        # Finite sentinel keeps every arithmetic op well-defined; masked entries
        # fall far below the threshold and clamp to exact zero.
        x = torch.where(finite, x, torch.full_like(x, -1e9))

        max_val, _ = x.max(dim=dim, keepdim=True)
        x = x - max_val          # shift-invariant; numerical stability
        x = x / 2.0              # reduce the alpha=1.5 problem to the threshold form

        x_srt, _ = torch.sort(x, descending=True, dim=dim)
        rho = _make_ix_like(x, dim)
        mean = x_srt.cumsum(dim) / rho
        mean_sq = (x_srt * x_srt).cumsum(dim) / rho
        ss = rho * (mean_sq - mean * mean)
        delta = torch.clamp((1.0 - ss) / rho, min=0.0)
        tau = mean - torch.sqrt(delta)
        support_size = (tau <= x_srt).sum(dim=dim, keepdim=True)
        tau_star = tau.gather(dim, support_size - 1)

        p = torch.clamp(x - tau_star, min=0.0) ** 2
        p = p.masked_fill(row_all_masked, 0.0)

        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        r"""Analytic backward $dX = s\odot(dY - \tfrac{\sum s\odot dY}{\sum s})$, $s=\sqrt{p}$."""
        (p,) = ctx.saved_tensors
        dim = ctx.dim
        s = p.sqrt()
        d_x = grad_output * s
        s_sum = s.sum(dim=dim, keepdim=True)
        # Guard fully-masked rows (s_sum == 0) so q is 0 rather than NaN.
        q = d_x.sum(dim=dim, keepdim=True) / s_sum.clamp_min(1e-12)
        d_x = d_x - q * s
        return d_x, None


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Sparse $\operatorname{entmax}_{1.5}$ normalisation along ``dim``.

    A drop-in, dependency-free alternative to ``softmax`` that yields exact zeros
    for low-scored entries. Non-finite entries (masked positions) map to zero
    probability; a fully-masked row maps to all zeros without NaNs.

    Args:
        x: Score tensor.
        dim: Axis to normalise over (default last).

    Returns:
        A tensor the same shape as ``x`` that is non-negative and sums to 1 along
        ``dim`` (or all-zero for a fully-masked row).
    """
    return _Entmax15Fn.apply(x, dim)


def sparse_normalize(
    scores: torch.Tensor, dim: int = -1, use_entmax: bool = True
) -> torch.Tensor:
    r"""Normalise ``scores`` along ``dim`` with $\operatorname{entmax}_{1.5}$ or softmax.

    Args:
        scores: Score tensor (masked positions may be non-finite / large-negative).
        dim: Axis to normalise over.
        use_entmax: If ``True`` use :func:`entmax15` (sparse); otherwise
            :func:`torch.nn.functional.softmax` (dense fallback).

    Returns:
        The normalised weights along ``dim``.
    """
    if use_entmax:
        return entmax15(scores, dim=dim)
    return F.softmax(scores, dim=dim)


# =============================================================================
# Checkpoint model-class guard (S0-T04)
# =============================================================================


def check_model_class(ckpt: Any, active_cls_name: str) -> None:
    r"""Guard that a checkpoint's ``model_class`` matches the active alias class.

    Runs on the raw checkpoint dict BEFORE any ``SeqVaeLagAttn(**model_kwargs)``
    reconstruction. Because the v1/v2 constructors are keyword-only with no
    ``**kwargs``, loading a v2 checkpoint's ``model_kwargs`` into v1 (or vice
    versa) would otherwise raise a cryptic ``TypeError`` at construction; this
    guard fails first with an actionable message.

    Args:
        ckpt: The deserialised checkpoint dict (non-dicts are skipped).
        active_cls_name: ``__name__`` of the currently-active alias class, e.g.
            ``"SeqVaeLagAttnV1"`` or ``"SeqVaeLagAttnV2"``.

    Raises:
        ValueError: If the checkpoint records a ``model_class`` differing from
            ``active_cls_name``.
    """
    if not isinstance(ckpt, dict):
        return
    stored = ckpt.get("model_class")
    if stored is None:
        warnings.warn(
            "checkpoint has no 'model_class' field (pre-guard checkpoint); "
            f"assuming it matches the active class {active_cls_name!r}. The "
            "rebuild will still fail loudly if the constructor kwargs are "
            "incompatible.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if str(stored) != str(active_cls_name):
        raise ValueError(
            f"checkpoint model_class={stored!r} does not match the active model "
            f"class {active_cls_name!r}. Toggle the SeqVaeLagAttn alias import to "
            f"the matching version before loading this checkpoint."
        )


# =============================================================================
# Deterministic backbone: target causal encoder (S1-T01)
# =============================================================================


class GatedCausalConvBlock(nn.Module):
    r"""Pre-LN gated depthwise-separable causal conv block (arch spec section 5).

    Given a block input $x_t \in \mathbb{R}^{d}$ the block computes
    $$\bar x_t = \operatorname{LN}(x_t),\quad v_t = W_v\bar x_t,\quad
    g_t = \sigma(W_g\bar x_t),$$
    $$r_{1:T} = W_o\!\left(\operatorname{DWConv}^{\mathrm{causal}}_{k,\delta}(v_{1:T})
    \odot g_{1:T}\right),\qquad x_t \leftarrow x_t + \operatorname{Dropout}(r_t).$$

    The gate is a sigmoid on a separate pointwise projection of the SAME LN'd
    input; the depthwise causal conv (left-only padding $(k-1)\delta$) is applied
    to the value path and multiplied by the gate AFTER the conv; $W_o$ is the
    separable pointwise mixing projection. Strict causality holds because
    $\operatorname{LN}$, the gate, and the pointwise projections are per-timestep,
    and the conv only looks backward.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the block.

        Args:
            d_model: Channel width $d$.
            kernel_size: Depthwise conv kernel size $k$.
            dilation: Depthwise conv dilation $\delta$.
            dropout: Dropout probability applied to the residual branch.
        """
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size, groups=d_model, dilation=dilation
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the block. Input/output ``(B, T, d)``."""
        x_bar = self.norm(x)
        v = self.value_proj(x_bar)                    # (B, T, d)
        g = torch.sigmoid(self.gate_proj(x_bar))      # (B, T, d)
        v_c = v.transpose(1, 2)                        # (B, d, T)
        v_c = F.pad(v_c, (self.left_pad, 0))           # causal left pad
        v_c = self.dwconv(v_c)                         # (B, d, T)
        v_c = v_c.transpose(1, 2)                       # (B, T, d)
        r = self.out_proj(v_c * g)
        return x + self.drop(r)


class TargetCausalEncoderV2(nn.Module):
    r"""Multi-scale gated causal target encoder (arch spec section 5).

    Stacks $N_y$ :class:`GatedCausalConvBlock` blocks with an exponential dilation
    schedule and a final $\operatorname{LN}$, producing the target state
    $h^y_t \in \mathbb{R}^{d}$. The receptive field is
    $R = 1 + (k-1)\sum_i \delta_i$; the default $(k, \delta) = (5, \{1,2,4,8,16,32\})$
    gives $R = 253$ decimated steps ($\approx 16.9$ min).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_blocks: int = 6,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the encoder.

        Args:
            d_model: Channel width $d$.
            num_blocks: Number of gated conv blocks $N_y$; must equal
                ``len(dilations)``.
            kernel_size: Depthwise conv kernel size $k$.
            dilations: Per-block dilation schedule.
            dropout: Dropout probability per block.
        """
        super().__init__()
        if len(dilations) != num_blocks:
            raise ValueError(
                f"num_blocks={num_blocks} must equal len(dilations)="
                f"{len(dilations)}"
            )
        self.blocks = nn.ModuleList(
            GatedCausalConvBlock(d_model, kernel_size, d, dropout)
            for d in dilations
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.receptive_field = 1 + (kernel_size - 1) * sum(int(d) for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode ``(B, T, d)`` target-adapter states into $h^y$ ``(B, T, d)``."""
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


# =============================================================================
# Deterministic backbone: source lag-atom encoder (S1-T02)
# =============================================================================


class SourceInputAdapterV2(nn.Module):
    r"""Project the source stream $U$ into the source-adapter width $d_u$.

    Same LN/Linear/GELU/Dropout/ResMLP form as v1's adapters (arch spec section
    4). Per-timestep, so it does not extend the source receptive field.

    Shapes:
        Input:  ``(B, T, in_dim)`` -- ``in_dim = c_u`` (101 or 58).
        Output: ``(B, T, d_u)``.
    """

    def __init__(self, in_dim: int = 101, d_u: int = 96, dropout: float = 0.1) -> None:
        r"""Initialize the source adapter.

        Args:
            in_dim: Source feature channels $c_u$.
            d_u: Source adapter width $d_u$.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, d_u)
        self.norm = nn.LayerNorm(d_u)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res_mlp = ResidualMLP(
            input_dim=d_u,
            hidden_dims=geometric_schedule(d_u, d_u, 3),
            final_activation=True,
            use_skip_connection=True,
            use_input_layer_norm=True,
            activation=nn.GELU,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Project ``(B, T, in_dim)`` into ``(B, T, d_u)``."""
        x = self.drop(self.act(self.norm(self.linear(x))))
        return self.res_mlp(x)


class _CausalLocalConv(nn.Module):
    r"""Single-scale causal local convolution with left-only padding.

    A ``Conv1d`` with kernel $r$ and left padding $r-1$, so the output at position
    $s$ depends only on inputs $[s-(r-1), s]$ (bounded, causal).
    """

    def __init__(self, d_in: int, d_out: int, kernel: int) -> None:
        r"""Initialize the local conv.

        Args:
            d_in: Input channel width.
            d_out: Output channel width.
            kernel: Kernel size $r$ (look-back is $r-1$ steps).
        """
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, kernel)
        self.left_pad = kernel - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Convolve ``(B, T, d_in)`` causally into ``(B, T, d_out)``."""
        x_c = x.transpose(1, 2)                 # (B, d_in, T)
        x_c = F.pad(x_c, (self.left_pad, 0))    # causal left pad
        x_c = self.conv(x_c)                    # (B, d_out, T)
        return x_c.transpose(1, 2)              # (B, T, d_out)


class SourceLagAtomEncoder(nn.Module):
    r"""Bounded-receptive-field source lag-atom encoder (arch spec section 6).

    Multi-scale local causal convs at scales $\mathcal{R}$ produce per-position
    atoms that are concatenated and projected to the model width:
    $$r^u_s = W_r\,[a^{u,3}_s\,|\,a^{u,9}_s\,|\,a^{u,21}_s] + b_r \in \mathbb{R}^{d}.$$
    Each atom depends on at most $\max(\mathcal{R})$ past decimated steps
    (default 21), so $r^u_s$ isolates a bounded, local window of the source --
    unlike a recurrent state which summarises all of $U_{\le s}$.
    """

    def __init__(
        self,
        d_u: int = 96,
        d_model: int = 128,
        scales: Tuple[int, ...] = (3, 9, 21),
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the lag-atom encoder.

        Args:
            d_u: Source adapter width $d_u$ (input width).
            d_model: Output width $d$.
            scales: Local causal conv scales $\mathcal{R}$.
            dropout: Dropout probability applied after projection.
        """
        super().__init__()
        self.scales = tuple(int(r) for r in scales)
        self.convs = nn.ModuleList(
            _CausalLocalConv(d_u, d_u, r) for r in self.scales
        )
        self.act = nn.GELU()
        self.proj = nn.Linear(len(self.scales) * d_u, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        # Bounded look-back: the largest scale reaches back ``max(scales) - 1``
        # steps, so an atom at ``s`` depends on inputs in ``[s - max_lookback, s]``.
        self.max_lookback = max(self.scales) - 1

    def forward(self, u_tilde: torch.Tensor) -> torch.Tensor:
        r"""Encode source-adapter states ``(B, T, d_u)`` into $r^u$ ``(B, T, d)``."""
        feats = [self.act(conv(u_tilde)) for conv in self.convs]
        cat = torch.cat(feats, dim=-1)          # (B, T, len(scales) * d_u)
        r_u = self.drop(self.proj(cat))
        return self.norm(r_u)


# =============================================================================
# Deterministic backbone: target adapter, prior head, shared lag embedding
# (S1-T03a)
# =============================================================================


class TargetInputAdapterV2(nn.Module):
    r"""Project target features (and optional context) into the model width.

    Same LN/Linear/GELU/Dropout/ResMLP form as v1's adapters (arch spec section
    4), with an optional temporal context vector $c_t$ concatenated to $Y$:
    $\tilde y_t = A_y([y_t\,|\,c_t])$. Context defaults to zeros, so the output is
    identical to the no-context path (the zero columns contribute nothing to the
    input ``Linear``).

    Shapes:
        Input:  ``(B, T, in_dim)`` target, optional ``(B, T, context_dim)`` context.
        Output: ``(B, T, d_model)``.
    """

    def __init__(
        self,
        in_dim: int = 87,
        d_model: int = 128,
        context_dim: int = 5,
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the target adapter.

        Args:
            in_dim: Target feature channels $c_y$.
            d_model: Internal model width $d$.
            context_dim: Temporal context width $C_c$ (0 disables context).
            dropout: Dropout probability.
        """
        super().__init__()
        self.context_dim = int(context_dim)
        self.linear = nn.Linear(in_dim + self.context_dim, d_model)
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

    def forward(
        self, y: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Project ``(B, T, in_dim)`` (plus optional context) into ``(B, T, d)``."""
        if self.context_dim > 0:
            if context is None:
                context = y.new_zeros(*y.shape[:-1], self.context_dim)
            x = torch.cat([y, context], dim=-1)
        else:
            x = y
        x = self.drop(self.act(self.norm(self.linear(x))))
        return self.res_mlp(x)


class LagEmbedding(nn.Module):
    r"""Shared lag-embedding table $e_\ell \in \mathbb{R}^{d_e}$ (arch spec section 9).

    A single table shared across heads (shape $(L, d_e)$); per-head specialisation
    comes from downstream per-head projections. Consumed by the target-only lag
    prior (S2-T03), the lag-posterior latent MLP (S3-T01), and the expected-lag
    embedding (S3-T02).
    """

    def __init__(self, num_lags: int, d_e: int) -> None:
        r"""Initialize the lag-embedding table.

        Args:
            num_lags: Number of lag bins $L = L_{\max} + 1$.
            d_e: Lag-embedding dimensionality $d_e$.
        """
        super().__init__()
        self.num_lags = int(num_lags)
        self.d_e = int(d_e)
        self.table = nn.Parameter(torch.empty(num_lags, d_e))
        nn.init.normal_(self.table, std=0.02)

    def forward(self) -> torch.Tensor:
        r"""Return the shared lag-embedding table ``(L, d_e)``."""
        return self.table


class PriorHeadV2(nn.Module):
    r"""Per-head continuous latent prior + baseline conditioning state.

    For each head $m$ (arch spec sections 12 and 16):
    $$\mu^{p,(m)}_t = \mu_{\mathrm{scale}}\tanh\!\Big(\tfrac{W^{p,m}_\mu
    \operatorname{LN}(h^y_t)}{\mu_{\mathrm{scale}}}\Big),\qquad
    \log(\sigma^{p,(m)}_t)^2 = \operatorname{clamp}_{[\ell_{\min},\ell_{\max}]}
    (W^{p,m}_\sigma \operatorname{LN}(h^y_t)),$$
    plus the target-only baseline conditioning state $b_t = B_\psi(h^y_t)$ used by
    the baseline future decoder.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_z_m: int,
        mu_scale: float,
        logvar_clamp: Tuple[float, float],
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the prior head.

        Args:
            d_model: Model width $d$.
            num_heads: Number of heads $M$.
            d_z_m: Per-head latent dim $d_z^{(m)}$.
            mu_scale: tanh bound on the prior mean.
            logvar_clamp: ``(min, max)`` log-variance clamp.
            dropout: Dropout used in the baseline-state MLP.
        """
        super().__init__()
        self.M = int(num_heads)
        self.d_z_m = int(d_z_m)
        self.mu_scale = float(mu_scale)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, self.M * self.d_z_m)
        self.logvar_head = nn.Linear(d_model, self.M * self.d_z_m)
        self.baseline_state = ResidualMLP(
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
        r"""Return ``(mu_heads (B,T,M,d_z_m), logvar_heads, b_t (B,T,d))``."""
        B, T, _ = h_y.shape
        h = self.norm(h_y)
        mu = self.mu_scale * torch.tanh(self.mu_head(h) / self.mu_scale)
        logvar = self.logvar_head(h).clamp(
            min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        mu_heads = mu.view(B, T, self.M, self.d_z_m)
        logvar_heads = logvar.view(B, T, self.M, self.d_z_m)
        b_t = self.baseline_state(h_y)
        return mu_heads, logvar_heads, b_t


# =============================================================================
# Lag posterior: smooth lag bias (S2-T01)
# =============================================================================


class SmoothLagBias(nn.Module):
    r"""Smooth per-head lag bias over fixed Gaussian bases (arch spec sections 9/20).

    With $J_b$ fixed Gaussian bases $\varphi_j(\ell) = \exp(-(\ell-c_j)^2/(2 s_j^2))$
    (fixed centers $c_j$, fixed widths $s_j$) and per-head learned weights
    $\theta^{(m)}_j$, the bias is
    $$b^{(m)}_\ell = \sum_{j=1}^{J_b} \theta^{(m)}_j \varphi_j(\ell) \in \mathbb{R},$$
    added to the lag scores before normalisation. The weights initialise to zero,
    so the bias starts flat (no lag preference). The smoothness penalty is
    $$\mathcal{L}_{\mathrm{bias}} = \frac{1}{M(L-1)} \sum_{m}\sum_{\ell=0}^{L-2}
    \big(b^{(m)}_{\ell+1} - b^{(m)}_\ell\big)^2.$$
    """

    def __init__(
        self,
        num_heads: int,
        num_lags: int,
        centers: Tuple[float, ...],
        widths: Optional[Tuple[float, ...]] = None,
    ) -> None:
        r"""Initialize the smooth lag bias.

        Args:
            num_heads: Number of heads $M$.
            num_lags: Number of lag bins $L$.
            centers: Fixed Gaussian-basis centers $c_j$.
            widths: Fixed Gaussian-basis widths $s_j$; ``None`` derives them from
                local inter-center spacing (floored at 2).
        """
        super().__init__()
        self.M = int(num_heads)
        self.L = int(num_lags)
        centers_t = torch.tensor([float(c) for c in centers], dtype=torch.float32)
        if widths is None:
            widths_t = self._derive_widths(centers_t)
        else:
            widths_t = torch.tensor([float(w) for w in widths], dtype=torch.float32)
        if widths_t.numel() != centers_t.numel():
            raise ValueError("centers and widths must have equal length")
        ells = torch.arange(self.L, dtype=torch.float32)          # (L,)
        phi = torch.exp(
            -((ells[None, :] - centers_t[:, None]) ** 2)
            / (2.0 * widths_t[:, None] ** 2)
        )                                                          # (J_b, L)
        self.register_buffer("phi", phi)
        self.theta = nn.Parameter(torch.zeros(self.M, phi.shape[0]))  # (M, J_b)

    @staticmethod
    def _derive_widths(centers: torch.Tensor) -> torch.Tensor:
        r"""Derive per-basis widths $s_j$ from local inter-center spacing."""
        J = centers.numel()
        if J == 1:
            # A single center has no neighbour spacing; fall back to the floor.
            return torch.full((1,), 2.0)
        widths = torch.empty(J)
        for j in range(J):
            if j == 0:
                widths[j] = centers[1] - centers[0]
            elif j == J - 1:
                widths[j] = centers[J - 1] - centers[J - 2]
            else:
                widths[j] = 0.5 * (centers[j + 1] - centers[j - 1])
        return widths.clamp_min(2.0)

    def forward(self) -> torch.Tensor:
        r"""Return the per-head lag bias $b^{(m)}_\ell$ of shape ``(M, L)``."""
        return self.theta @ self.phi

    def smoothness_penalty(self) -> torch.Tensor:
        r"""Return the mean-squared first-difference smoothness penalty $\mathcal{L}_{bias}$."""
        b = self.forward()
        diff = b[:, 1:] - b[:, :-1]
        return (diff ** 2).mean()


# =============================================================================
# Lag posterior: scores, entmax, strided KV, active set (S2-T02/T04)
# =============================================================================

# Finite sentinel for masked (invalid) lag scores -- entmax maps it to exactly
# zero probability without the NaNs a raw ``-inf`` would create.
_MASK_NEG = -1e9


class LagPosteriorAttention(nn.Module):
    r"""Discrete lag posterior over past source atoms (arch spec sections 8/15).

    Projects a query from the target state $h^y$ and keys/values from the source
    lag-atom states $r^u$, scores each lag with a strided view (no
    $(B,T,L,\cdot)$ bank is materialised), adds the smooth lag bias (and an
    optional cross-phase bias), masks invalid lags ($t-\ell<0$), and normalises
    with $\operatorname{entmax}_{1.5}$ (or softmax):
    $$s^{(m)}_{t,\ell} = \frac{\langle q^{(m)}_t, k^{(m)}_{t-\ell}\rangle}{\sqrt{d_k}}
    + b^{(m)}_\ell + \rho^{(m)}_{t,\ell},\qquad
    \alpha^{(m)}_{t,:} = \operatorname{entmax}_{1.5}(s^{(m)}_{t,:}).$$
    The top-$K_a$ active set and renormalisation are added in S2-T04.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: int,
        d_v: int,
        num_lags: int,
        use_entmax: bool = False,
    ) -> None:
        r"""Initialize the lag posterior attention.

        Args:
            d_model: Model width $d$.
            num_heads: Number of heads $M$.
            d_k: Key/query dim $d_k$.
            d_v: Value dim $d_v$.
            num_lags: Number of lag bins $L$.
            use_entmax: If ``True`` normalise with $\operatorname{entmax}_{1.5}$,
                else softmax.
        """
        super().__init__()
        self.M = int(num_heads)
        self.d_k = int(d_k)
        self.d_v = int(d_v)
        self.L = int(num_lags)
        self.use_entmax = bool(use_entmax)
        self.scale = 1.0 / math.sqrt(self.d_k)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, self.M * self.d_k)
        self.k_proj = nn.Linear(d_model, self.M * self.d_k)
        self.v_proj = nn.Linear(d_model, self.M * self.d_v)

    def project(
        self, h_y: torch.Tensor, r_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Project into per-head ``q (B,T,M,d_k)``, ``k (B,T,M,d_k)``, ``v (B,T,M,d_v)``."""
        B, T, _ = h_y.shape
        q = self.q_proj(self.norm_q(h_y)).view(B, T, self.M, self.d_k)
        k_in = self.norm_kv(r_u)
        k = self.k_proj(k_in).view(B, T, self.M, self.d_k)
        v = self.v_proj(k_in).view(B, T, self.M, self.d_v)
        return q, k, v

    def lag_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        lag_bias: torch.Tensor,
        cross_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Score every lag via a strided view. Returns ``(B, T, M, L)``.

        Left-pads $k$ by $L-1$ and takes a size-$L$ ``unfold`` window (a strided
        view, no copy), contracts $d_k$, then flips the window axis so index
        $\ell=0$ is the current step. No $(B, T, L, d)$ tensor is allocated.
        """
        L = self.L
        # Front-pad the time axis by L-1 so each output position sees L past keys.
        k_pad = F.pad(k, (0, 0, 0, 0, L - 1, 0))            # (B, T+L-1, M, d_k)
        k_win = k_pad.unfold(dimension=1, size=L, step=1)   # (B, T, M, d_k, L) view
        raw = torch.einsum("btmd,btmdl->btml", q, k_win) * self.scale
        scores = raw.flip(-1)                                # window pos -> lag l
        scores = scores + lag_bias[None, None, :, :]
        if cross_bias is not None:
            scores = scores + cross_bias
        return scores

    @staticmethod
    def valid_mask(T: int, L: int, device: torch.device) -> torch.Tensor:
        r"""Return the banded validity mask ``valid[t, l] = (t - l >= 0)`` ``(T, L)``."""
        t_idx = torch.arange(T, device=device)[:, None]
        l_idx = torch.arange(L, device=device)[None, :]
        return l_idx <= t_idx

    def forward(
        self,
        h_y: torch.Tensor,
        r_u: torch.Tensor,
        lag_bias: torch.Tensor,
        cross_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return the lag posterior ``alpha (B,T,M,L)`` and values ``v (B,T,M,d_v)``."""
        _, T, _ = h_y.shape
        q, k, v = self.project(h_y, r_u)
        scores = self.lag_scores(q, k, lag_bias, cross_bias)
        valid = self.valid_mask(T, self.L, h_y.device)          # (T, L)
        scores = scores.masked_fill(~valid[None, :, None, :], _MASK_NEG)
        alpha = sparse_normalize(scores, dim=-1, use_entmax=self.use_entmax)
        return alpha, v

    def select_active(
        self,
        alpha: torch.Tensor,
        pi: torch.Tensor,
        v: torch.Tensor,
        active_lags: int,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        r"""Select the top-$K_a$ active lags and gather the active values.

        Renormalises the posterior and prior on the active set and gathers the
        active values into ``(B, T, M, K_a, d_v)`` directly (never a full-$L$
        value bank), for the $O(BTMK_a d_v)$ memory budget. Invalid picks
        ($t-\ell<0$) are masked out of both the renormalisation and the gather.

        Args:
            alpha: Posterior lag weights ``(B, T, M, L)``.
            pi: Prior lag weights ``(B, T, M, L)``.
            v: Per-head values ``(B, T, M, d_v)``.
            active_lags: Active count $K_a$.
            eps: Renormalisation floor.

        Returns:
            A dict with ``active_lag_indices (B,T,M,Ka)``, ``alpha_bar``,
            ``pi_bar`` (both ``(B,T,M,Ka)`` and summing to 1 over valid picks),
            ``active_v (B,T,M,Ka,d_v)``, and ``kld_lag (B,T,M)`` = the truncated
            $\bar K^R = \operatorname{KL}(\bar\alpha\|\bar\pi)$.
        """
        B, T, M, L = alpha.shape
        Ka = min(int(active_lags), L)
        active_idx = alpha.topk(Ka, dim=-1).indices               # (B,T,M,Ka)

        t_arange = torch.arange(T, device=alpha.device).view(1, T, 1, 1)
        valid_pick = active_idx <= t_arange                        # (B,T,M,Ka)

        alpha_act = torch.gather(alpha, -1, active_idx) * valid_pick
        pi_act = torch.gather(pi, -1, active_idx) * valid_pick
        alpha_bar = alpha_act / (alpha_act.sum(-1, keepdim=True) + eps)
        pi_bar = pi_act / (pi_act.sum(-1, keepdim=True) + eps)

        # Gather active values at source times t - lag, without a full-L bank.
        src_t = (t_arange - active_idx).clamp_min(0)               # (B,T,M,Ka)
        b_idx = torch.arange(B, device=alpha.device).view(B, 1, 1, 1)
        m_idx = torch.arange(M, device=alpha.device).view(1, 1, M, 1)
        active_v = v[b_idx, src_t, m_idx, :]                        # (B,T,M,Ka,d_v)
        active_v = active_v * valid_pick.unsqueeze(-1)

        kld_lag = discrete_lag_kl(alpha_bar, pi_bar, eps=eps)       # (B,T,M)

        return {
            "active_lag_indices": active_idx,
            "alpha_bar": alpha_bar,
            "pi_bar": pi_bar,
            "active_v": active_v,
            "kld_lag": kld_lag,
        }


# =============================================================================
# Lag posterior: target-only lag prior + discrete KL (S2-T03)
# =============================================================================


def discrete_lag_kl(
    alpha: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    r"""Discrete KL $\operatorname{KL}(\alpha\|\pi)$ over the last axis.

    $$K^R = \sum_\ell \alpha_\ell \log\frac{\alpha_\ell + \epsilon}{\pi_\ell + \epsilon}.$$
    Sparse ``alpha`` entries (exact zeros) contribute $0$ because
    $0 \cdot \log(\cdot)$ vanishes with the finite $\epsilon$ smoothing.

    Args:
        alpha: Posterior lag weights ``(..., L)`` (a simplex, possibly sparse).
        pi: Prior lag weights ``(..., L)`` (a strictly-positive simplex).
        eps: Numerical floor inside the logarithms.

    Returns:
        The per-row discrete KL ``(...)`` (last axis reduced).
    """
    return (alpha * (torch.log(alpha + eps) - torch.log(pi + eps))).sum(dim=-1)


class LagPriorHead(nn.Module):
    r"""Target-only discrete lag prior $\pi^{(m)}$ (arch spec section 11).

    $$g^{(m)}_{t,\ell} = a^{(m)\top}\operatorname{GELU}\!\big(W_h^{(m)}\operatorname{LN}(h^y_t)
    + W_e^{(m)} e_\ell\big) + b^{p,(m)}_\ell,\qquad
    \pi^{(m)}_{t,:} = \operatorname{softmax}(g^{(m)}_{t,:}).$$
    It depends ONLY on the target state $h^y$ and the shared lag embedding $e_\ell$
    (never the source), preserving the transfer-entropy interpretation, and uses a
    dense softmax (not entmax) so the KL stays finite.

    Note:
        The intermediate $\operatorname{GELU}$ activation has shape
        $(B, T, M, L, d_h)$; at production scale this is the dominant transient of
        the prior. ``d_hidden`` defaults to $d_e$; reduce it (or add time-chunking)
        if the memory smoke (S6-T04) flags it.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_lags: int,
        d_e: int,
        d_hidden: Optional[int] = None,
    ) -> None:
        r"""Initialize the target-only lag prior.

        Args:
            d_model: Model width $d$.
            num_heads: Number of heads $M$.
            num_lags: Number of lag bins $L$.
            d_e: Lag-embedding dim $d_e$.
            d_hidden: Hidden width of the prior scorer (defaults to $d_e$).
        """
        super().__init__()
        self.M = int(num_heads)
        self.L = int(num_lags)
        self.d_e = int(d_e)
        self.d_hidden = int(d_hidden) if d_hidden is not None else int(d_e)
        self.norm = nn.LayerNorm(d_model)
        self.W_h = nn.Linear(d_model, self.M * self.d_hidden)
        self.W_e = nn.Linear(self.d_e, self.M * self.d_hidden)
        self.a = nn.Parameter(torch.empty(self.M, self.d_hidden))
        nn.init.normal_(self.a, std=0.02)
        self.prior_lag_bias = nn.Parameter(torch.zeros(self.M, self.L))

    def scores(self, h_y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        r"""Return the prior logits $g^{(m)}_{t,\ell}$ of shape ``(B, T, M, L)``."""
        B, T, _ = h_y.shape
        H = self.W_h(self.norm(h_y)).view(B, T, self.M, self.d_hidden)   # (B,T,M,dh)
        E = self.W_e(e).view(self.L, self.M, self.d_hidden)             # (L,M,dh)
        pre = H[:, :, :, None, :] + E.permute(1, 0, 2)[None, None]      # (B,T,M,L,dh)
        act = F.gelu(pre)
        g = torch.einsum("btmld,md->btml", act, self.a)
        return g + self.prior_lag_bias[None, None, :, :]

    def forward(self, h_y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        r"""Return the target-only lag prior $\pi^{(m)}$ of shape ``(B, T, M, L)``."""
        return F.softmax(self.scores(h_y, e), dim=-1)


class CrossPhaseLagBias(nn.Module):
    r"""Score-only cross-phase lag-proposal bias $\rho^{(m)}_{t,\ell}$ (arch spec section 10).

    The dataset carries an FHR--UP cross-phase field $\boldsymbol{x}^{yu}_t$. It is
    NOT admitted into the transfer bottleneck as a source value (which would mix a
    joint FHR--UP stream into the latent content); instead a causal interaction
    state $h^c_t = E_c(\boldsymbol{x}^{yu}_{\le t})$ produces a per-head, per-lag
    bias that is added to the lag SCORES only:

    $$\rho^{(m)}_{t,\ell} = w_\rho^{(m)\top}\operatorname{GELU}\!\big(
    W_c^{(m)} h^c_t + W_e^{(m)} e_\ell\big).$$

    Because $\rho$ enters only the pre-softmax lag score, it influences the lag
    posterior $\alpha^{(m)}_{t,\ell}$ (hence which lags the latent aggregates over),
    but never the source values $v$, the per-lag content $\mu^{q}$, or the residual
    decoder input -- the latent CONTENT stays UP-only (the design goal of section
    10). This is a default-off ablation enabled after the source-pure v2 is stable.

    Note:
        The final projection $w_\rho$ is deliberately NOT zero-initialised. A zero
        init would give $\partial\rho/\partial x^{yu} = 0$, so the ablation would
        contribute nothing (and its gradient-path test would see a zero gradient).
        Standard init lets $\rho$ influence lag selection from the first step; the
        flag is only turned on deliberately, so no warm-start neutrality is needed.
    """

    def __init__(
        self,
        c_cross: int,
        d_model: int,
        num_heads: int,
        num_lags: int,
        d_e: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2),
    ) -> None:
        r"""Initialize the cross-phase lag-bias module.

        Args:
            c_cross: Cross-phase feature channels $C_c$ (79 for FHR--UP cross-phase).
            d_model: Model width $d$ of the causal interaction state.
            num_heads: Number of heads $M$.
            num_lags: Number of lag bins $L$.
            d_e: Shared lag-embedding dim $d_e$.
            d_hidden: Hidden width of the per-lag scorer (defaults to $d_e$).
            dropout: Dropout probability.
            num_blocks: Number of causal gated-conv blocks in $E_c$; must equal
                ``len(dilations)``.
            kernel_size: Depthwise conv kernel size of $E_c$.
            dilations: Per-block dilation schedule of $E_c$.
        """
        super().__init__()
        if len(dilations) != num_blocks:
            raise ValueError(
                f"num_blocks={num_blocks} must equal len(dilations)={len(dilations)}"
            )
        self.M = int(num_heads)
        self.L = int(num_lags)
        self.d_e = int(d_e)
        self.d_hidden = int(d_hidden) if d_hidden is not None else int(d_e)
        # Cross-phase adapter (LN/Linear/GELU/Dropout), mirroring the source adapter.
        self.in_linear = nn.Linear(int(c_cross), d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.in_act = nn.GELU()
        self.in_drop = nn.Dropout(dropout)
        # Causal interaction encoder $E_c$ (reuses the target encoder's gated block).
        self.blocks = nn.ModuleList(
            GatedCausalConvBlock(d_model, kernel_size, d, dropout) for d in dilations
        )
        self.enc_norm = nn.LayerNorm(d_model)
        # Per-head / per-lag raw scorer (structurally like ``LagPriorHead.scores``
        # but consuming $h^c$ and emitting raw scores -- no softmax, no prior bias).
        self.norm = nn.LayerNorm(d_model)
        self.W_c = nn.Linear(d_model, self.M * self.d_hidden)
        self.W_e = nn.Linear(self.d_e, self.M * self.d_hidden)
        self.w_rho = nn.Parameter(torch.empty(self.M, self.d_hidden))
        nn.init.normal_(self.w_rho, std=0.02)   # NOT zero -- gradient must flow.

    def encode(self, x_cross: torch.Tensor) -> torch.Tensor:
        r"""Causal interaction state $h^c_t = E_c(x^{yu}_{\le t})$ of shape ``(B, T, d)``."""
        h = self.in_drop(self.in_act(self.in_norm(self.in_linear(x_cross))))
        for block in self.blocks:
            h = block(h)
        return self.enc_norm(h)

    def forward(self, x_cross: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        r"""Return the score-only bias $\rho^{(m)}_{t,\ell}$ of shape ``(B, T, M, L)``.

        Args:
            x_cross: Cross-phase field ``(B, T, c_cross)``.
            e: Shared lag-embedding table ``(L, d_e)``.

        Returns:
            The additive lag-score bias ``(B, T, M, L)`` (same shape as the lag
            scores, so it broadcasts directly onto them).
        """
        h_c = self.encode(x_cross)
        B, T, _ = h_c.shape
        H = self.W_c(self.norm(h_c)).view(B, T, self.M, self.d_hidden)   # (B,T,M,dh)
        E = self.W_e(e).view(self.L, self.M, self.d_hidden)             # (L,M,dh)
        pre = H[:, :, :, None, :] + E.permute(1, 0, 2)[None, None]      # (B,T,M,L,dh)
        act = F.gelu(pre)
        return torch.einsum("btmld,md->btml", act, self.w_rho)         # (B,T,M,L)


# =============================================================================
# Lag-specific continuous latent posterior + content KL (S3-T01)
# =============================================================================


def content_gaussian_kl(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form diagonal-Gaussian content KL per active lag (arch spec section 12).

    For the per-active-lag posterior
    $q = \mathcal{N}(\mu^{q,(m)}_{t,\ell}, \operatorname{diag}[(\sigma^{q,(m)}_{t,\ell})^2])$
    and the target-only prior
    $p = \mathcal{N}(\mu^{p,(m)}_t, \operatorname{diag}[(\sigma^{p,(m)}_t)^2])$
    (broadcast over the active lags),
    $$K^{Z,(m)}_{t,\ell} = \frac{1}{2}\sum_{j}\left[
    \log(\sigma^{p,(m)}_{t,j})^2 - \log(\sigma^{q,(m)}_{t,\ell,j})^2
    + \frac{(\sigma^{q,(m)}_{t,\ell,j})^2 + (\mu^{q,(m)}_{t,\ell,j}-\mu^{p,(m)}_{t,j})^2}{
    (\sigma^{p,(m)}_{t,j})^2} - 1\right] \ge 0.$$

    Args:
        mu_q: Posterior means $(B, T, M, K_a, d_z^{(m)})$.
        logvar_q: Posterior log-variances $(B, T, M, K_a, d_z^{(m)})$.
        mu_p: Prior means $(B, T, M, d_z^{(m)})$; broadcast over $K_a$.
        logvar_p: Prior log-variances $(B, T, M, d_z^{(m)})$; broadcast over $K_a$.

    Returns:
        The per-active-lag content KL $(B, T, M, K_a)$, non-negative.
    """
    mu_p = mu_p.unsqueeze(-2)
    logvar_p = logvar_p.unsqueeze(-2)
    kz = 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
        - 1.0
    )
    return kz.sum(dim=-1)


class LagLatentPosteriorHead(nn.Module):
    r"""Lag-specific continuous latent posterior (arch spec section 12).

    For each head $m$ and active lag $\ell$, a fused state is built from the target
    state, the (already gathered) source value $v^{(m)}_{t,\ell}$, and the lag
    embedding $e_\ell$:
    $$g^{(m)}_{t,\ell} = \operatorname{MLP}\big([\operatorname{LN}(h^y_t)\,|\,
    \operatorname{LN}(v^{(m)}_{t,\ell})\,|\,e_\ell]\big).$$
    The posterior mean is a bounded residual on the prior mean and the log-variance
    is clamped:
    $$\Delta\mu^{(m)}_{t,\ell} = \delta\mu_{\mathrm{scale}}\tanh\!\Big(
    \frac{W^q_\mu g^{(m)}_{t,\ell}}{\delta\mu_{\mathrm{scale}}}\Big),\quad
    \mu^{q,(m)}_{t,\ell} = \mu^{p,(m)}_t + \Delta\mu^{(m)}_{t,\ell},\quad
    \log(\sigma^{q,(m)}_{t,\ell})^2 = \operatorname{clamp}(W^q_\sigma g^{(m)}_{t,\ell}).$$

    A single shared trunk is used for all heads, with per-head behaviour coming from
    (a) the per-head source values $v^{(m)}$ and (b) a per-head additive
    ``head_embed``. The mean head is a plain :class:`torch.nn.Linear` named
    ``delta_mu_head`` so :meth:`SeqVaeLagAttnV2._zero_init_delta_heads` can zero it
    for the warm start ($\Delta\mu \equiv 0 \Rightarrow \mu^q = \mu^p$ at init).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_z_m: int,
        d_v: int,
        d_e: int,
        delta_mu_scale: float,
        logvar_clamp: Tuple[float, float],
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        r"""Initialize the lag-latent posterior head.

        Args:
            d_model: Model width $d$.
            num_heads: Number of heads $M$.
            d_z_m: Per-head latent dim $d_z^{(m)}$.
            d_v: Lag-attention value dim $d_v$ (source-value width).
            d_e: Lag-embedding dim $d_e$.
            delta_mu_scale: tanh bound on the posterior mean residual $\Delta\mu$.
            logvar_clamp: ``(min, max)`` clamp on the posterior log-variance.
            d_hidden: Hidden width of the fused trunk (defaults to $d_e$; kept small
                because the fused input $(B,T,M,K_a,d+d_v+d_e)$ is the dominant
                transient).
            dropout: Dropout applied inside the trunk.
        """
        super().__init__()
        self.M = int(num_heads)
        self.d_z_m = int(d_z_m)
        self.delta_mu_scale = float(delta_mu_scale)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.d_hidden = int(d_hidden) if d_hidden is not None else int(d_e)
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_v)
        d_in = d_model + d_v + d_e
        self.fc1 = nn.Linear(d_in, self.d_hidden)
        self.head_embed = nn.Parameter(torch.empty(self.M, self.d_hidden))
        nn.init.normal_(self.head_embed, std=0.02)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.d_hidden, self.d_hidden)
        self.drop = nn.Dropout(dropout)
        self.delta_mu_head = nn.Linear(self.d_hidden, self.d_z_m)
        self.logvar_head = nn.Linear(self.d_hidden, self.d_z_m)

    def forward(
        self,
        h_y: torch.Tensor,
        active_v: torch.Tensor,
        e_active: torch.Tensor,
        mu_prior_heads: torch.Tensor,
        logvar_prior_heads: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Return ``(mu_q, logvar_q, kz, dmu)`` over the active lags.

        Args:
            h_y: Target state $(B, T, d)$.
            active_v: Gathered active source values $(B, T, M, K_a, d_v)$ (invalid
                picks already zeroed by the caller).
            e_active: Gathered lag embeddings $(B, T, M, K_a, d_e)$.
            mu_prior_heads: Per-head prior means $(B, T, M, d_z^{(m)})$.
            logvar_prior_heads: Per-head prior log-variances $(B, T, M, d_z^{(m)})$.

        Returns:
            ``mu_q`` / ``logvar_q`` $(B, T, M, K_a, d_z^{(m)})$, the content KL
            ``kz`` $(B, T, M, K_a)$, and the residual mean ``dmu``
            $(B, T, M, K_a, d_z^{(m)})$ (for the saturation diagnostic).
        """
        B, T, M, Ka, _ = active_v.shape
        h_ln = self.norm_h(h_y)[:, :, None, None, :].expand(B, T, M, Ka, -1)
        v_ln = self.norm_v(active_v)
        g_in = torch.cat([h_ln, v_ln, e_active], dim=-1)          # (B,T,M,Ka,d_in)
        g = self.fc1(g_in) + self.head_embed[None, None, :, None, :]
        g = self.act(g)
        g = self.drop(self.act(self.fc2(g)))
        raw_delta = self.delta_mu_head(g)
        dmu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
        mu_q = mu_prior_heads.unsqueeze(-2) + dmu
        logvar_q = self.logvar_head(g).clamp(
            min=self.logvar_clamp[0], max=self.logvar_clamp[1]
        )
        kz = content_gaussian_kl(
            mu_q, logvar_q, mu_prior_heads, logvar_prior_heads
        )
        return mu_q, logvar_q, kz, dmu


# =============================================================================
# Exact KL decomposition + lag-resolved transfer map (S3-T04a)
# =============================================================================


class TEDecompositionHead(nn.Module):
    r"""Exact head-wise KL decomposition and lag-resolved map (arch spec section 13).

    Given the renormalized active weights $\bar\alpha^{(m)}_{t,\ell}$,
    $\bar\pi^{(m)}_{t,\ell}$, the per-active-lag content KL $K^{Z,(m)}_{t,\ell}$, and
    the truncated lag KL $K^{R,(m)}_t = \operatorname{KL}(\bar\alpha^{(m)}\|\bar\pi^{(m)})$,
    the module returns
    $$K^{Z,(m)}_t = \sum_\ell \bar\alpha^{(m)}_{t,\ell} K^{Z,(m)}_{t,\ell},\quad
    K^{(m)}_t = K^{R,(m)}_t + K^{Z,(m)}_t,\quad K_t = \sum_m K^{(m)}_t,$$
    and the lag-resolved transfer map scattered from the active set to all $L$ lags
    $$K_{t,\ell} = \sum_m \bar\alpha^{(m)}_{t,\ell}\left[
    \log\frac{\bar\alpha^{(m)}_{t,\ell}+\epsilon}{\bar\pi^{(m)}_{t,\ell}+\epsilon}
    + K^{Z,(m)}_{t,\ell}\right].$$

    The construction reuses the SAME $\bar\alpha, \bar\pi, K^R, \epsilon$ that define
    ``kld_lag``, so the identities $\sum_\ell K_{t,\ell} = K_t$ and
    $K_t = \sum_m(K^{R,(m)} + K^{Z,(m)})$ hold exactly (up to float round-off).
    ``te_lag_map`` is NOT clamped: individual $K_{t,\ell}$ entries may be slightly
    negative where $\bar\alpha_\ell < \bar\pi_\ell$, but the row-sum (a KL) is
    non-negative; clamping would break the load-bearing $\sum_\ell$ identity.
    """

    def __init__(self, num_heads: int, num_lags: int, eps: float = 1e-8) -> None:
        r"""Initialize the decomposition head.

        Args:
            num_heads: Number of heads $M$.
            num_lags: Number of lag bins $L$.
            eps: $\epsilon$ inside the discrete-KL logarithms (must match the value
                used to compute ``kld_lag`` in :meth:`LagPosteriorAttention.select_active`).
        """
        super().__init__()
        self.M = int(num_heads)
        self.L = int(num_lags)
        self.eps = float(eps)

    def forward(
        self,
        alpha_bar: torch.Tensor,
        pi_bar: torch.Tensor,
        kz: torch.Tensor,
        kld_lag: torch.Tensor,
        active_lag_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Return the KL decomposition dict.

        Args:
            alpha_bar: Renormalized active posterior weights $(B, T, M, K_a)$.
            pi_bar: Renormalized active prior weights $(B, T, M, K_a)$.
            kz: Per-active-lag content KL $(B, T, M, K_a)$.
            kld_lag: Truncated discrete lag KL $K^{R,(m)}_t$ $(B, T, M)$.
            active_lag_indices: Active lag indices $(B, T, M, K_a)$ (long).

        Returns:
            A dict with ``kld_content`` $(B,T,M)$, ``kld_per_t_per_head`` $(B,T,M)$,
            ``kld_per_t`` $(B,T)$, and ``te_lag_map`` $(B,T,L)$.
        """
        B, T, M, _ = alpha_bar.shape
        kld_content = (alpha_bar * kz).sum(-1)                    # (B,T,M)
        kld_per_t_per_head = kld_lag + kld_content                # (B,T,M)
        kld_per_t = kld_per_t_per_head.sum(-1)                    # (B,T)

        contrib = alpha_bar * (
            torch.log(alpha_bar + self.eps)
            - torch.log(pi_bar + self.eps)
            + kz
        )                                                          # (B,T,M,Ka)
        te_bm = alpha_bar.new_zeros(B, T, M, self.L)
        te_bm.scatter_add_(-1, active_lag_indices, contrib)
        te_lag_map = te_bm.sum(dim=2)                             # (B,T,L)
        return {
            "kld_content": kld_content,
            "kld_per_t_per_head": kld_per_t_per_head,
            "kld_per_t": kld_per_t,
            "te_lag_map": te_lag_map,
        }


class SeqVaeLagAttnV2(nn.Module):
    r"""Lag-Attentive Residual VAE-TEB (v2).

    A source-pure variational UP-to-FHR transfer model with an explicit discrete
    lag variable per head and an exact KL decomposition (see the module
    docstring). It is a drop-in successor to :class:`SeqVaeLagAttnV1`: the
    constructor accepts every v1 keyword-only parameter with identical defaults,
    plus the new v2 parameters below.

    The forward pass and loss are assembled across Sprints 1-3; at Sprint 0 the
    constructor validates and stores the full parameter surface and registers the
    latent running-statistics buffers, while :meth:`forward` and
    :meth:`compute_loss` raise :class:`NotImplementedError`.
    """

    def __init__(
        self,
        *,
        # --- v1 parameters (verbatim names + defaults; drop-in parity) --------
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
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = False,
        encoder_extra_dilations: Tuple[int, ...] = (),
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        delta_mu_scale: float = 3.0,
        latent_stats_momentum: float = 0.01,
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        head_structured_latent: bool = False,
        init_weights: bool = True,
        # --- new v2 parameters (keyword-only) --------------------------------
        target_encoder_blocks: int = 6,
        target_kernel: int = 5,
        target_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        source_scales: Tuple[int, ...] = (3, 9, 21),
        d_u: int = 96,
        d_k: int = 16,
        d_e: int = 32,
        lag_basis_centers: Tuple[float, ...] = (
            0, 4, 8, 12, 16, 24, 32, 40, 52, 64, 78, 90,
        ),
        lag_basis_widths: Optional[Tuple[float, ...]] = None,
        active_lags: int = 8,
        active_lags_warmup: int = 16,
        kl_eps: float = 1e-8,
        context_dim: int = 5,
        kappa_z: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_ent: float = 0.0,
        enable_source: bool = True,
        enable_residual: bool = True,
        enable_kl: bool = True,
        use_crossphase_bias: bool = False,
        c_cross: int = 79,
        use_outcome_head: bool = False,
        outcome_classes: int = 3,
        step_seconds: float = 4.0,
        delta_up_seconds: float = 0.0,
    ) -> None:
        r"""Initialize ``SeqVaeLagAttnV2``.

        Args:
            sequence_length: Decimated sequence length $T$. Informational in v2
                (warm-up sizing is derived from the actual input length).
            d_model: Internal width $d$ used throughout the backbone.
            d_z: Total latent dimensionality; partitioned into ``num_heads``
                groups of ``d_z_m = d_z / num_heads`` (must divide evenly).
            horizon: Decimated forecast horizon $H_d$ (default 30 = 2 min).
            warmup_period: Number of initial decimated steps ignored in both KL
                and feature losses (default 30 = 2 min).
            c_y: Target feature channel count (43 + 44 = 87).
            c_u: Source feature channel count. Must equal ``101`` when
                ``use_up_st=True`` and ``58`` when ``use_up_st=False``; an
                inconsistent ``(c_u, use_up_st)`` pair raises ``ValueError``.
            use_up_st: If ``True`` (default) feed the source adapter with
                ``[up_st, up_ph]``; if ``False`` feed only ``up_ph``.
            max_lag: Maximum past lag $L_{\max}$; the lag window is
                ``L = max_lag + 1`` bins ($\ell \in \{0, \ldots, L_{\max}\}$).
            num_heads: Number of lag-attention heads $M$.
            d_head: v1 per-head dim. In v2 this is remapped to the lag-attention
                **value** dim $d_v$; v2 does NOT require ``num_heads * d_head ==
                d_model``.
            lstm_layers: No effect in v2 (no recurrence). Accepted and stored for
                config round-trip parity with v1.
            dropout: Dropout used throughout adapters / encoders / MLPs.
            decoder_hidden: Hidden width of the reused v1 horizon decoders.
            horizon_depth: Depth of the reused v1 horizon refine core.
            horizon_kernel: Horizon-conv kernel size of the reused core.
            horizon_film: FiLM toggle of the reused v1 horizon core (default off;
                the v2 spec's core is additive, but v2 reuses the v1 decoder).
            encoder_extra_dilations: No effect in v2 (the target encoder uses the
                fixed ``target_dilations`` schedule). Accepted and stored for
                config round-trip parity.
            logvar_clamp: ``(min, max)`` clamp applied to every log-variance head.
            mu_scale: tanh saturation magnitude of ``mu_prior``; caps
                $|\mu_{prior}| \le \mu_{scale}$.
            delta_mu_scale: tanh saturation magnitude of the posterior delta;
                caps $|\mu_{post} - \mu_{prior}| \le \delta\mu_{scale}$.
            latent_stats_momentum: EMA momentum for the ``mu_post_running_*``
                buffers during training.
            use_entmax: If ``True`` the lag posterior $\alpha$ uses
                $\operatorname{entmax}_{1.5}$; otherwise softmax. Default ``False``
                for strict v1 drop-in parity (the v2 config enables it).
            attention_grad_checkpoint: Reserved; no effect in S0-S2 (the strided
                KV attention is already memory-light). Accepted and stored.
            lag_bias_init: No effect in v2 (the lag bias is a Gaussian-basis
                :class:`SmoothLagBias`). Accepted and stored for parity.
            head_structured_latent: No effect in v2 (v2 is always head-structured
                with an explicit per-head latent). Accepted and stored for parity.
            init_weights: If ``True`` apply :func:`initialization` then enforce
                the zero-init on the residual mean head (warm-start).
            target_encoder_blocks: Number of gated causal conv blocks $N_y$ in the
                target encoder (default 6).
            target_kernel: Kernel size of each target conv block (default 5).
            target_dilations: Per-block dilation schedule of the target encoder
                (default $\{1,2,4,8,16,32\}$; receptive field $R = 1 + (k-1)
                \sum_i \delta_i = 253$).
            source_scales: Local causal conv scales of the source lag-atom encoder
                (default $\{3,9,21\}$; bounded receptive field $\le 21$ steps).
            d_u: Source adapter width $d_u$ (default 96); the atom encoder projects
                up to ``d_model``.
            d_k: Lag-attention key/query dim $d_k$ (default 16).
            d_e: Shared lag-embedding dim $d_e$ (default 32). The expected-lag
                embedding has width ``num_heads * d_e``.
            lag_basis_centers: Fixed Gaussian-basis centers $c_j$ (default 12
                centers spanning $[0, 90]$).
            lag_basis_widths: Fixed Gaussian-basis widths $s_j$; ``None`` (default)
                derives them per-basis from local inter-center spacing.
            active_lags: Active lag count $K_a$ (default 8; settable at runtime and
                by the curriculum from each stage's ``active_lags`` entry).
            active_lags_warmup: Reference warm-up active-lag count (default 16). The
                EFFECTIVE per-stage count is read from each curriculum stage dict's
                ``active_lags`` field, not from this attribute; it is a documented
                default for authoring those stages and is not read by the forward.
            kl_eps: $\epsilon$ added inside KL logarithms (default $10^{-8}$).
            context_dim: Dimensionality of the (deferred) temporal context vector
                $c_t$; defaults to zero-valued so the output is identical to the
                no-context path.
            kappa_z: Content-KL free-bits budget per latent dim (default 0.0;
                consumed by ``compute_loss`` in Sprint 4).
            lambda_tv: Weight of the temporal lag TV regularizer (Sprint 4).
            lambda_ent: Weight of the lag-entropy regularizer (Sprint 4).
            enable_source: Curriculum flag; when ``False`` the forward runs the
                baseline-only (target-pure) path.
            enable_residual: Curriculum flag for the source residual decoder
                (Sprint 3/4).
            enable_kl: Curriculum flag for the KL term (Sprint 4).
            use_crossphase_bias: Optional section-10 score-only cross-phase lag
                bias (default off; Sprint 7).
            c_cross: Cross-phase field channel count (used only when
                ``use_crossphase_bias``).
            use_outcome_head: Optional section-22 supervised outcome head (default
                off; Sprint 7).
            outcome_classes: Number of outcome classes for the outcome head.
            step_seconds: Physical seconds per decimated step (default 4.0), used
                for lag-in-seconds diagnostics (Sprint 7).
            delta_up_seconds: UP alignment offset $\Delta_{UP}$ in seconds for the
                physical-time lag axis (Sprint 7).
        """
        super().__init__()

        # --- v1-parity scalar attributes ------------------------------------
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
                f"dataset schema - there is no virtual-slicing fallback."
            )
        self.c_u = expected_c_u
        self.max_lag = int(max_lag)
        self.num_heads = int(num_heads)
        self.d_head = int(d_head)
        self.dropout = float(dropout)
        self.decoder_hidden = int(decoder_hidden)
        self.horizon_depth = int(horizon_depth)
        self.horizon_kernel = int(horizon_kernel)
        self.horizon_film = bool(horizon_film)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.mu_scale = float(mu_scale)
        if self.mu_scale <= 0.0:
            raise ValueError(f"mu_scale must be > 0, got {mu_scale}")
        self.delta_mu_scale = float(delta_mu_scale)
        if self.delta_mu_scale <= 0.0:
            raise ValueError(f"delta_mu_scale must be > 0, got {delta_mu_scale}")
        self.latent_stats_momentum = float(latent_stats_momentum)
        self.use_entmax = bool(use_entmax)
        self.init_weights = bool(init_weights)

        # Dead-but-kept v1 parameters (no effect in v2; stored for config
        # round-trip fidelity and ``inspect.signature`` auto-forwarding).
        self.lstm_layers = int(lstm_layers)
        self.encoder_extra_dilations = tuple(int(x) for x in encoder_extra_dilations)
        self.attention_grad_checkpoint = bool(attention_grad_checkpoint)
        self.lag_bias_init = str(lag_bias_init)
        self.head_structured_latent = bool(head_structured_latent)

        # --- v2 structural attributes ---------------------------------------
        if self.d_z % self.num_heads != 0:
            raise ValueError(
                f"d_z must be divisible by num_heads, got d_z={self.d_z}, "
                f"num_heads={self.num_heads}"
            )
        self.M = self.num_heads
        self.d_z_m = self.d_z // self.num_heads
        self.d_v = self.d_head
        self.d_k = int(d_k)
        self.d_e = int(d_e)
        self.L = self.max_lag + 1

        self.target_encoder_blocks = int(target_encoder_blocks)
        self.target_kernel = int(target_kernel)
        self.target_dilations = tuple(int(x) for x in target_dilations)
        self.source_scales = tuple(int(x) for x in source_scales)
        self.d_u = int(d_u)
        self.lag_basis_centers = tuple(float(x) for x in lag_basis_centers)
        self.lag_basis_widths = (
            None if lag_basis_widths is None
            else tuple(float(x) for x in lag_basis_widths)
        )
        self.active_lags = int(active_lags)
        self.active_lags_warmup = int(active_lags_warmup)
        self.kl_eps = float(kl_eps)
        self.context_dim = int(context_dim)

        # Loss weights carried on the model (consumed in Sprints 4/7).
        self.kappa_z = float(kappa_z)
        self.lambda_tv = float(lambda_tv)
        self.lambda_ent = float(lambda_ent)

        # Curriculum branch flags.
        self.enable_source = bool(enable_source)
        self.enable_residual = bool(enable_residual)
        self.enable_kl = bool(enable_kl)

        # Optional flag-gated features (Sprint 7).
        self.use_crossphase_bias = bool(use_crossphase_bias)
        self.c_cross = int(c_cross)
        self.use_outcome_head = bool(use_outcome_head)
        self.outcome_classes = int(outcome_classes)
        # The cross-phase score-only lag bias (arch spec section 10, S7-T01) and
        # the supervised outcome head (section 22, S7-T02) are wired below and
        # constructed only when their flags are on; both default off.

        # Physical-time diagnostics (Sprint 7).
        self.step_seconds = float(step_seconds)
        self.delta_up_seconds = float(delta_up_seconds)

        # Width of the augmented latent fed to the reused residual decoder in
        # Sprint 3: ``[z | expected-lag-embedding]``. Sized now so the decoder
        # can be constructed in S1-T03b without a later ``__init__`` edit.
        self.expected_lag_embed_dim = self.M * self.d_e
        self.d_z_residual = self.d_z + self.expected_lag_embed_dim
        # Lower bound on the mixture posterior variance before ``log`` (S3-T02).
        # Equals the smallest valid per-lag variance $\exp(\ell_{\min})$, which is
        # also the natural floor of the law-of-total-variance mixture when the
        # active weights sum to 1, so it is a no-op except on degenerate rows.
        self._var_floor = float(math.exp(self.logvar_clamp[0]))

        # --- Sub-modules ----------------------------------------------------
        # Input adapters (target adapter accepts optional zero-default context).
        self.target_adapter = TargetInputAdapterV2(
            in_dim=self.c_y,
            d_model=self.d_model,
            context_dim=self.context_dim,
            dropout=self.dropout,
        )
        self.source_adapter = SourceInputAdapterV2(
            in_dim=self.c_u, d_u=self.d_u, dropout=self.dropout
        )
        # Deterministic backbone encoders.
        self.target_encoder = TargetCausalEncoderV2(
            d_model=self.d_model,
            num_blocks=self.target_encoder_blocks,
            kernel_size=self.target_kernel,
            dilations=self.target_dilations,
            dropout=self.dropout,
        )
        self.source_encoder = SourceLagAtomEncoder(
            d_u=self.d_u,
            d_model=self.d_model,
            scales=self.source_scales,
            dropout=self.dropout,
        )
        # Prior head + shared lag-embedding table.
        self.prior_head = PriorHeadV2(
            d_model=self.d_model,
            num_heads=self.M,
            d_z_m=self.d_z_m,
            mu_scale=self.mu_scale,
            logvar_clamp=self.logvar_clamp,
            dropout=self.dropout,
        )
        self.lag_embedding = LagEmbedding(num_lags=self.L, d_e=self.d_e)
        # Reused v1 horizon decoders (shared core). The residual decoder consumes
        # the augmented latent [z | expected-lag-embedding] of width d_z_residual
        # (wired in Sprint 3); its mean head is zero-inited for warm-start.
        self.horizon_core = HorizonDecoderCore(
            d_hidden=self.decoder_hidden,
            horizon=self.horizon,
            kernel_size=self.horizon_kernel,
            depth=self.horizon_depth,
            film=self.horizon_film,
        )
        self.baseline_decoder = BaselineFutureDecoder(
            core=self.horizon_core,
            d_model=self.d_model,
            out_channels=self.c_y,
            d_hidden=self.decoder_hidden,
            dropout=self.dropout,
            logvar_clamp=self.logvar_clamp,
        )
        self.residual_decoder = ResidualFutureDecoder(
            core=self.horizon_core,
            d_model=self.d_model,
            d_z=self.d_z_residual,
            out_channels=self.c_y,
            d_hidden=self.decoder_hidden,
            dropout=self.dropout,
            logvar_clamp=self.logvar_clamp,
        )
        # Lag posterior sub-modules (Sprint 2).
        self.lag_bias = SmoothLagBias(
            num_heads=self.M,
            num_lags=self.L,
            centers=self.lag_basis_centers,
            widths=self.lag_basis_widths,
        )
        self.lag_posterior = LagPosteriorAttention(
            d_model=self.d_model,
            num_heads=self.M,
            d_k=self.d_k,
            d_v=self.d_v,
            num_lags=self.L,
            use_entmax=self.use_entmax,
        )
        self.lag_prior = LagPriorHead(
            d_model=self.d_model,
            num_heads=self.M,
            num_lags=self.L,
            d_e=self.d_e,
        )
        # Lag-specific continuous latent posterior + exact KL decomposition
        # (Sprint 3). Constructed BEFORE ``_zero_init_delta_heads`` so its
        # ``delta_mu_head`` is zeroed for the warm start.
        self.lag_latent_head = LagLatentPosteriorHead(
            d_model=self.d_model,
            num_heads=self.M,
            d_z_m=self.d_z_m,
            d_v=self.d_v,
            d_e=self.d_e,
            delta_mu_scale=self.delta_mu_scale,
            logvar_clamp=self.logvar_clamp,
            dropout=self.dropout,
        )
        self.te_decomposition = TEDecompositionHead(
            num_heads=self.M, num_lags=self.L, eps=self.kl_eps
        )
        # Optional cross-phase score-only lag bias (S7-T01; arch spec section 10).
        # Default off; when on, its rho is added to the lag scores only. It has
        # an effect ONLY if the caller passes ``x_cross`` to ``forward`` -- as of
        # this writing neither ``trainer_lag_attn_v1.py`` nor ``pl_module_v2.py``
        # does, so enabling the flag alone trains an inert submodule with zero
        # gradient contribution. Warn loudly so this is never a silent no-op.
        if self.use_crossphase_bias:
            warnings.warn(
                "use_crossphase_bias=True constructs CrossPhaseLagBias, but it "
                "only affects the forward pass when the caller explicitly passes "
                "x_cross to forward(...). No current production or synthetic "
                "Lightning module wires x_cross through, so this ablation will "
                "train inert (zero-gradient) parameters unless you supply "
                "x_cross yourself at the call site.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.crossphase_bias = (
            CrossPhaseLagBias(
                c_cross=self.c_cross,
                d_model=self.d_model,
                num_heads=self.M,
                num_lags=self.L,
                d_e=self.d_e,
                dropout=self.dropout,
            )
            if self.use_crossphase_bias
            else None
        )
        # Optional supervised outcome head (S7-T02; arch spec section 22).
        # Default off; labels never enter ``forward`` -- it consumes the forward
        # dict and is trained in a separate Stage-4 pass (see ``freeze_vae``).
        self.outcome_head = (
            OutcomeHead(
                d_model=self.d_model,
                d_z=self.d_z,
                num_lags=self.L,
                num_classes=self.outcome_classes,
                dropout=self.dropout,
            )
            if self.use_outcome_head
            else None
        )

        # --- Latent running stats (downstream classifier consumption) -------
        self.register_buffer("mu_post_running_mean", torch.zeros(self.d_z))
        self.register_buffer("mu_post_running_var", torch.ones(self.d_z))
        self.register_buffer(
            "mu_post_running_count", torch.zeros((), dtype=torch.long)
        )

        # --- Weight init ----------------------------------------------------
        if self.init_weights:
            initialization(self)
        # Zero-init the residual mean head AFTER generic init so ``delta_mu_src``
        # is exactly zero and ``mu_full == mu_base`` at step 0 (warm-start).
        self._zero_init_delta_heads()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        """Zero a ``Linear`` layer's weight (and bias if present)."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _zero_init_delta_heads(self) -> None:
        r"""Zero the residual-decoder mean head so ``delta_mu_src = 0`` at init.

        This enforces the warm-start invariant $\mu^{full} = \mu^{base}$ at step 0.
        The lag-latent posterior delta head (Sprint 3) is zeroed here as well once
        it exists.
        """
        self._zero_linear(self.residual_decoder.mean_head)
        lag_latent = getattr(self, "lag_latent_head", None)
        if lag_latent is not None and hasattr(lag_latent, "delta_mu_head"):
            self._zero_linear(lag_latent.delta_mu_head)

    # ------------------------------------------------------------------
    # Forward / sampling
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        r"""Sample $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim N(0, I)$."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        x_cross: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the v2 pipeline and emit the superset forward dictionary.

        With ``enable_source=False`` this runs the target-only baseline path
        (adapters + target encoder + prior head + baseline decoder) and emits the
        full v1 key set with the source-dependent keys zero-filled and the
        posterior set to the prior (S1-T04a). The source path (lag posterior,
        continuous latent, residual decoder) is assembled in Sprints 2-3.

        Args:
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            u_stream: Source stream ``(B, T, c_u)``.
            x_cross: Optional FHR--UP cross-phase field ``(B, T, c_cross)`` for the
                score-only lag bias (S7-T01). Used only when
                ``use_crossphase_bias=True``; ``None`` (the default) preserves the
                three-tensor call sites and the source-pure forward.

        Returns:
            The superset forward dictionary (see the module docstring).
        """
        Y = torch.cat([y_st, y_ph], dim=-1)               # (B, T, c_y)
        Y_tilde = self.target_adapter(Y)                   # (B, T, d)
        H_y = self.target_encoder(Y_tilde)                 # (B, T, d)

        mu_heads, logvar_heads, decoder_state = self.prior_head(H_y)
        B, T, _ = H_y.shape
        mu_prior = mu_heads.reshape(B, T, self.d_z)
        logvar_prior = logvar_heads.reshape(B, T, self.d_z)

        mu_base, logvar_base = self.baseline_decoder(decoder_state)

        if not self.enable_source:
            return self._baseline_only_outputs(
                H_y=H_y,
                mu_prior=mu_prior,
                logvar_prior=logvar_prior,
                decoder_state=decoder_state,
                mu_base=mu_base,
                logvar_base=logvar_base,
            )

        return self._source_outputs(
            H_y=H_y,
            u_stream=u_stream,
            mu_heads=mu_heads,
            logvar_heads=logvar_heads,
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            decoder_state=decoder_state,
            mu_base=mu_base,
            logvar_base=logvar_base,
            x_cross=x_cross,
        )

    def _source_outputs(
        self,
        *,
        H_y: torch.Tensor,
        u_stream: torch.Tensor,
        mu_heads: torch.Tensor,
        logvar_heads: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        decoder_state: torch.Tensor,
        mu_base: torch.Tensor,
        logvar_base: torch.Tensor,
        x_cross: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Assemble the full source-conditioned forward dict (S3-T04b).

        Runs the source lag-atom encoder, the lag posterior + target-only lag
        prior, the top-$K_a$ active set, the lag-specific continuous latent, the
        latent aggregation + mixture moments, the exact KL decomposition, and the
        source residual decoder, then returns the superset of the v1 forward keys
        plus the new v2 keys (see the module docstring).

        When ``use_crossphase_bias`` is on and ``x_cross`` is supplied, the
        score-only cross-phase bias $\rho$ is added to the lag scores (S7-T01);
        it reaches ``alpha`` only, never the source values or latent content.
        """
        B, T, _ = H_y.shape

        # Source lag-atom states + shared lag embedding + smooth lag bias.
        u_tilde = self.source_adapter(u_stream)                 # (B,T,d_u)
        r_u = self.source_encoder(u_tilde)                       # (B,T,d)
        e_table = self.lag_embedding()                           # (L,d_e)
        lag_bias_mat = self.lag_bias()                           # (M,L)

        # Optional score-only cross-phase lag bias (S7-T01). Added to the lag
        # SCORES only -- flows to alpha, never to the source values v or content.
        cross_bias = None
        if (
            self.use_crossphase_bias
            and self.crossphase_bias is not None
            and x_cross is not None
        ):
            cross_bias = self.crossphase_bias(x_cross, e_table)  # (B,T,M,L)

        # Lag posterior (entmax alpha + values) and target-only lag prior.
        alpha, v = self.lag_posterior(
            H_y, r_u, lag_bias_mat, cross_bias=cross_bias
        )                                                        # (B,T,M,L),(B,T,M,d_v)
        pi = self.lag_prior(H_y, e_table)                        # (B,T,M,L)

        # Top-Ka active set with renormalized weights and truncated K^R.
        active = self.lag_posterior.select_active(
            alpha, pi, v, active_lags=self.active_lags, eps=self.kl_eps
        )
        active_idx = active["active_lag_indices"]                # (B,T,M,Ka)
        alpha_bar = active["alpha_bar"]                          # (B,T,M,Ka)
        pi_bar = active["pi_bar"]                                # (B,T,M,Ka)
        active_v = active["active_v"]                            # (B,T,M,Ka,d_v)
        kld_lag = active["kld_lag"]                              # (B,T,M)

        # Lag-specific continuous posterior + per-active-lag content KL.
        e_active = e_table[active_idx]                           # (B,T,M,Ka,d_e)
        mu_q, logvar_q, kz, dmu = self.lag_latent_head(
            H_y, active_v, e_active, mu_heads, logvar_heads
        )

        # Aggregation, mixture moments, expected lag (S3-T02).
        z_heads, mu_post_h, logvar_post_h = self._aggregate_latent(
            alpha_bar, mu_q, logvar_q
        )
        z = z_heads.reshape(B, T, self.d_z)
        mu_post = mu_post_h.reshape(B, T, self.d_z)
        logvar_post = logvar_post_h.reshape(B, T, self.d_z)

        lag_vals = active_idx.to(alpha_bar.dtype)
        expected_lag = (alpha_bar * lag_vals).sum(-1)            # (B,T,M)
        exp_e_heads = torch.einsum("btmk,btmke->btme", alpha_bar, e_active)
        expected_lag_embedding = exp_e_heads.reshape(
            B, T, self.expected_lag_embed_dim
        )                                                         # (B,T,M*d_e)

        attended_source_heads = torch.einsum(
            "btmk,btmkv->btmv", alpha_bar, active_v
        )                                                         # (B,T,M,d_v)
        attended_source = attended_source_heads.reshape(
            B, T, self.M * self.d_v
        )                                                         # (B,T,d)

        # Exact KL decomposition + lag-resolved map (S3-T04a).
        decomp = self.te_decomposition(alpha_bar, pi_bar, kz, kld_lag, active_idx)
        kld_content = decomp["kld_content"]
        kld_per_t_per_head = decomp["kld_per_t_per_head"]
        kld_per_t = decomp["kld_per_t"]
        te_lag_map = decomp["te_lag_map"]

        # Source residual decoder (S3-T03); warm-started to zero at init. The
        # ``enable_residual`` curriculum flag gates the residual correction: when
        # off, mu_full == mu_base and logvar_full falls back to logvar_base (the
        # source latent / KL are still computed, but do not affect the forecast).
        if self.enable_residual:
            z_aug = torch.cat([z, expected_lag_embedding], dim=-1)  # (B,T,d_z_residual)
            delta_mu_src, logvar_full = self.residual_decoder(decoder_state, z_aug)
            mu_full = mu_base + delta_mu_src
        else:
            delta_mu_src = torch.zeros_like(mu_base)
            logvar_full = logvar_base
            mu_full = mu_base

        if self.training:
            self._update_latent_running_stats(mu_post)

        with torch.no_grad():
            mu_prior_sat_frac = (
                mu_prior.abs() >= (0.99 * self.mu_scale)
            ).float().mean()
            delta_mu_sat_frac = (
                dmu.abs() >= (0.99 * self.delta_mu_scale)
            ).float().mean()
            lag_entropy = (
                -(alpha * torch.log(alpha + self.kl_eps)).sum(-1).mean()
            )
            n_active = (alpha > 0).float().sum(-1).mean()

        warmup_mask = self._build_warmup_valid_mask(T, device=H_y.device)

        return {
            # --- v1 keys (mapped from v2 internals) --------------------------
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": r_u,
            "decoder_state": decoder_state,
            "attended_source": attended_source,
            "attended_source_heads": attended_source_heads,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "delta_mu_src": delta_mu_src,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "raw_future_pred": None,
            "kld_per_t": kld_per_t,
            "kld_per_t_per_head": kld_per_t_per_head,
            "te_lag_map": te_lag_map,
            "warmup_mask": warmup_mask,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
            # --- new additive v2 keys ----------------------------------------
            "pi_lag": pi,
            "active_lag_indices": active_idx,
            "alpha_bar": alpha_bar,
            "pi_bar": pi_bar,
            "mu_prior_heads": mu_heads,
            "logvar_prior_heads": logvar_heads,
            "mu_post_active": mu_q,
            "logvar_post_active": logvar_q,
            "kld_lag": kld_lag,
            "kld_content": kld_content,
            "expected_lag": expected_lag,
            "expected_lag_embedding": expected_lag_embedding,
            "lag_entropy": lag_entropy,
            "n_active": n_active,
        }

    def _aggregate_latent(
        self,
        alpha_bar: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Aggregate per-active-lag Gaussians into a head latent + mixture moments.

        The head latent is the active-weighted reparameterized sample (mean at
        inference):
        $$z^{(m)}_t = \sum_\ell \bar\alpha^{(m)}_{t,\ell} z^{(m)}_{t,\ell}.$$
        The flat posterior moments follow the law of total variance
        (arch spec section 14),
        $$\mu^{q,(m)}_t = \sum_\ell \bar\alpha^{(m)}_{t,\ell}\mu^{q,(m)}_{t,\ell},\quad
        (\sigma^{q,(m)}_t)^2 = \sum_\ell \bar\alpha^{(m)}_{t,\ell}\big[
        (\sigma^{q,(m)}_{t,\ell})^2 + (\mu^{q,(m)}_{t,\ell})^2\big]
        - (\mu^{q,(m)}_t)^2,$$
        floored at $\exp(\ell_{\min})$ before the ``log`` for numerical safety.

        Args:
            alpha_bar: Renormalized active weights $(B, T, M, K_a)$.
            mu_q: Per-active-lag posterior means $(B, T, M, K_a, d_z^{(m)})$.
            logvar_q: Per-active-lag posterior log-variances (same shape).

        Returns:
            ``(z_heads, mu_post_heads, logvar_post_heads)``, each
            $(B, T, M, d_z^{(m)})$.
        """
        if self.training:
            z_lag = self.reparameterize(mu_q, logvar_q)
        else:
            z_lag = mu_q
        z_heads = torch.einsum("btmk,btmkj->btmj", alpha_bar, z_lag)
        mu_post_h = torch.einsum("btmk,btmkj->btmj", alpha_bar, mu_q)
        second = torch.einsum(
            "btmk,btmkj->btmj", alpha_bar, logvar_q.exp() + mu_q ** 2
        )
        var_post_h = (second - mu_post_h ** 2).clamp_min(self._var_floor)
        logvar_post_h = var_post_h.log()
        return z_heads, mu_post_h, logvar_post_h

    def _baseline_only_outputs(
        self,
        *,
        H_y: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        decoder_state: torch.Tensor,
        mu_base: torch.Tensor,
        logvar_base: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Assemble the target-only baseline forward dict (23 v1 keys)."""
        B, T, _ = H_y.shape
        device, dtype = H_y.device, H_y.dtype
        M, L, d_v, d = self.M, self.L, self.d_v, self.d_model

        # No source: the posterior collapses to the prior (deterministic z).
        mu_post = mu_prior
        logvar_post = logvar_prior
        z = mu_prior

        if self.training:
            self._update_latent_running_stats(mu_post)

        with torch.no_grad():
            mu_prior_sat_frac = (
                mu_prior.abs() >= (0.99 * self.mu_scale)
            ).float().mean()
            delta_mu_sat_frac = torch.zeros((), device=device, dtype=dtype)

        delta_mu_src = torch.zeros_like(mu_base)
        warmup_mask = self._build_warmup_valid_mask(T, device=device)

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": torch.zeros(B, T, d, device=device, dtype=dtype),
            "decoder_state": decoder_state,
            # Width M*d_v (the active-value summary width), matching the source
            # path's ``attended_source`` so the key's shape is stable across
            # curriculum stages even when ``num_heads * d_head != d_model``.
            "attended_source": torch.zeros(
                B, T, M * d_v, device=device, dtype=dtype
            ),
            "attended_source_heads": torch.zeros(
                B, T, M, d_v, device=device, dtype=dtype
            ),
            "attn_weights": torch.zeros(B, T, M, L, device=device, dtype=dtype),
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "delta_mu_src": delta_mu_src,
            "mu_full": mu_base,
            "logvar_full": logvar_base,
            "raw_future_pred": None,
            "kld_per_t": torch.zeros(B, T, device=device, dtype=dtype),
            "kld_per_t_per_head": torch.zeros(B, T, M, device=device, dtype=dtype),
            "te_lag_map": torch.zeros(B, T, L, device=device, dtype=dtype),
            "warmup_mask": warmup_mask,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
        }

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
        detach_baseline_in_full: bool = False,
        lambda_lag: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        r"""Compute the v2 training objective (v1-compatible signature + keys).

        The reconstruction terms (``feat_loss`` on $\mu^{full}$, ``base_loss`` on
        $\mu^{base}$) reuse v1's unfold / warm-up / dataset-weight masking. The KL
        term is the exact v2 decomposition (arch spec section 19),
        $$\mathcal{L}_{\mathrm{KL}} = \frac{\sum_{b,t} m^{\mathrm{KL}}_{b,t}
        \sum_m\big[K^{R,(m)}_{b,t} + \max(K^{Z,(m)}_{b,t}, \kappa_z d_z^{(m)})\big]}{
        \sum_{b,t} m^{\mathrm{KL}}_{b,t}},$$
        normalized by the number of valid $(b,t)$ (NOT $\times d_z$), so $\beta$ is
        on the spec scale. The free-bits floor $\kappa_z$ applies to the content KL
        only; the lag KL $K^R$ is unfloored. The reported ``kld_per_t`` in the
        forward dict stays raw / unfloored. Three lag regularizers
        ($\mathcal{L}_{\mathrm{bias}}$, $\mathcal{L}_{\mathrm{tv}}$,
        $\mathcal{L}_{\mathrm{ent}}$) are added with model-owned weights. A
        zero-valued grad-mask keeps every conditional-branch parameter connected to
        ``total_loss`` for DDP.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            weight: Optional per-step dataset weight ``(B, T)`` in ``[0, 1]``.
            compute_kld_loss: If ``False`` (or ``self.enable_kl`` is ``False``, or
                the forward dict lacks the decomposition keys) the KL term is zero.
            beta: KL weight $\beta$.
            lambda_full: Weight on the full reconstruction term.
            lambda_base: Weight on the baseline reconstruction term.
            likelihood: ``"mse"`` or ``"gaussian_nll"``.
            sigma_obs: Observation noise (scalar) or ``"learned"`` (uses the model
                log-variance heads).
            free_bits: DEPRECATED and ignored in v2 (a warning is emitted when
                non-zero); the content-KL free-bits floor is governed by the model
                attribute ``kappa_z``. Kept in the signature for v1 call parity.
            detach_baseline_in_full: If ``True`` rebuild
                $\mu^{full} = \mathrm{sg}(\mu^{base}) + \Delta\mu^{src}$.
            lambda_lag: Weight on the smooth lag-bias penalty
                $\mathcal{L}_{\mathrm{bias}}$.

        Returns:
            A dict with the nine v1 loss keys plus ``kld_lag_loss``,
            ``kld_content_loss``, ``lag_tv``, and ``lag_entropy_reg``.
        """
        Y = torch.cat([y_st, y_ph], dim=-1)                # (B, T, C_y)
        mu_base = forward_outputs["mu_base"]               # (B, T, H_d, C_y)
        if detach_baseline_in_full:
            mu_full = mu_base.detach() + forward_outputs["delta_mu_src"]
        else:
            mu_full = forward_outputs["mu_full"]

        B, T, Hd, C = mu_full.shape
        T_valid = T - Hd
        device = Y.device
        dtype = Y.dtype

        # Future target via unfold: Y_plus[:, t, tau] = Y[:, t + 1 + tau].
        Y_shift = Y[:, 1:, :]                              # (B, T-1, C_y)
        Y_plus = Y_shift.unfold(dimension=1, size=Hd, step=1)
        Y_plus = Y_plus.permute(0, 1, 3, 2).contiguous()   # (B, T_valid, Hd, C_y)

        mu_full_valid = mu_full[:, :T_valid, :, :]
        mu_base_valid = mu_base[:, :T_valid, :, :]

        # Warm-up mask on the anchor axis (always applied).
        warmup = self._warmup_steps(T)
        warmup_t = torch.zeros(T_valid, dtype=dtype, device=device)
        if warmup < T_valid:
            warmup_t[warmup:] = 1.0

        # Optional dataset weight: weight[b, t] * weight[b, t + tau + 1].
        if weight is not None:
            w = weight.to(device=device, dtype=dtype)
            anchor_w = w[:, :T_valid]
            target_w = w[:, 1:].unfold(dimension=1, size=Hd, step=1)
            mask_feat = (
                warmup_t[None, :, None, None]
                * anchor_w[:, :, None, None]
                * target_w[:, :, :, None]
            )
        else:
            mask_feat = warmup_t[None, :, None, None].expand(B, T_valid, Hd, 1)

        denom = (mask_feat.sum() * float(C)).clamp_min(1.0)

        diff_full = (mu_full_valid - Y_plus) ** 2
        diff_base = (mu_base_valid - Y_plus) ** 2

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
                        f"sigma_obs scalar must be positive, got {sigma_obs_f}"
                    )
                logvar_scalar = math.log(sigma_obs_f ** 2)
                logvar_full_obs = torch.full_like(diff_full, logvar_scalar)
                logvar_base_obs = torch.full_like(diff_base, logvar_scalar)
            per_elem_full = (
                0.5 * diff_full * torch.exp(-logvar_full_obs) + 0.5 * logvar_full_obs
            )
            per_elem_base = (
                0.5 * diff_base * torch.exp(-logvar_base_obs) + 0.5 * logvar_base_obs
            )
        else:
            raise ValueError(
                f"likelihood must be 'mse' or 'gaussian_nll', got {likelihood!r}"
            )

        feat_loss = (per_elem_full * mask_feat).sum() / denom
        base_loss = (per_elem_base * mask_feat).sum() / denom

        # --- Decomposed KL (S4-T01) -----------------------------------------
        # Spec-form (arch spec section 19): mean over valid (b,t) of
        # sum_m [K^R + max(K^Z, kappa_z * d_z_m)]; the free-bits floor applies to
        # the content KL only and the lag KL K^R is never floored. The reported
        # ``kld_per_t`` in the forward dict stays raw / unfloored.
        if free_bits:
            warnings.warn(
                "compute_loss(free_bits=...) is deprecated in v2 and ignored; the "
                "content-KL free-bits floor is governed by the model attribute "
                "``kappa_z`` (nats per latent dim). Set kappa_z in the "
                "constructor / config instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        have_decomp = (
            "kld_lag" in forward_outputs and "kld_content" in forward_outputs
        )
        if compute_kld_loss and self.enable_kl and have_decomp:
            kld_lag_bt = forward_outputs["kld_lag"]            # (B,T,M)
            kld_content_bt = forward_outputs["kld_content"]    # (B,T,M)
            kcont_floored = kld_content_bt.clamp(
                min=self.kappa_z * float(self.d_z_m)
            )
            per_bt = (kld_lag_bt + kcont_floored).sum(-1)      # (B,T)
            mask_kl = self._kl_mask(
                kld_lag_bt.size(0), kld_lag_bt.size(1),
                weight=weight, device=device, dtype=dtype,
            )
            denom_kl = mask_kl.sum().clamp_min(1.0)
            kld_loss = (per_bt * mask_kl).sum() / denom_kl
            kld_lag_loss = (kld_lag_bt.sum(-1) * mask_kl).sum() / denom_kl
            kld_content_loss = (kcont_floored.sum(-1) * mask_kl).sum() / denom_kl
            # Reporting content KL is UNFLOORED (arch spec section 19): the
            # kappa_z free-bits floor is an optimisation device only. r_lag (below)
            # must use this raw K^Z, not the floored ``kld_content_loss``.
            kld_content_raw = (kld_content_bt.sum(-1) * mask_kl).sum() / denom_kl
        else:
            kld_loss = torch.zeros((), device=device, dtype=dtype)
            kld_lag_loss = torch.zeros((), device=device, dtype=dtype)
            kld_content_loss = torch.zeros((), device=device, dtype=dtype)
            kld_content_raw = torch.zeros((), device=device, dtype=dtype)

        # --- Lag regularizers (S4-T02) --------------------------------------
        lag_bias = getattr(self, "lag_bias", None)
        if lambda_lag > 0.0 and lag_bias is not None:
            lag_smoothness = lag_bias.smoothness_penalty()
        else:
            lag_smoothness = torch.zeros((), device=device, dtype=dtype)

        alpha = forward_outputs.get("attn_weights")
        if self.lambda_tv > 0.0 and alpha is not None:
            d_alpha = alpha[:, 1:, :, :] - alpha[:, :-1, :, :]
            lag_tv = d_alpha.abs().sum(-1).mean()
        else:
            lag_tv = torch.zeros((), device=device, dtype=dtype)
        if self.lambda_ent > 0.0 and alpha is not None:
            lag_entropy_reg = (
                -(alpha * torch.log(alpha + self.kl_eps)).sum(-1).mean()
            )
        else:
            lag_entropy_reg = torch.zeros((), device=device, dtype=dtype)

        total_loss = (
            lambda_full * feat_loss
            + lambda_base * base_loss
            + beta * kld_loss
            + lambda_lag * lag_smoothness
            + self.lambda_tv * lag_tv
            + self.lambda_ent * lag_entropy_reg
        )

        # --- DDP-safe grad-mask (S4-T03) ------------------------------------
        # Add a zero-valued touch of every conditional-branch parameter so a
        # disabled curriculum branch still receives a zero (not None) gradient
        # from ``total_loss``; the plain ``ddp`` reducer then finds every
        # parameter (no ``find_unused_parameters`` required). The numeric loss is
        # unchanged (scaled by 0.0) and the touched set is fixed across steps
        # (static-graph safe).
        total_loss = total_loss + 0.0 * self._grad_touch(total_loss)

        mean_logvar_full = (logvar_full_valid * mask_feat).sum() / denom
        mean_logvar_base = (logvar_base_valid * mask_feat).sum() / denom

        # --- Section-26 diagnostics (S7-T03) --------------------------------
        # delta_L = L_base - L_full: prediction gain of the source residual
        # (a healthy source pathway drives it > 0).
        delta_l = base_loss - feat_loss
        # Channel-normalised residual RMS: sqrt(sum m_feat*(delta_mu_src)^2 /
        # (C_y * sum m_feat + eps)) -- section 26. Distinct from the trainer's
        # existing ``delta_mu_rms`` (which omits the 1/C_y channel average).
        delta_src_valid = forward_outputs["delta_mu_src"][:, :T_valid, :, :]
        rms_src = torch.sqrt(
            ((delta_src_valid ** 2) * mask_feat).sum() / denom
        )
        # r_lag = E[K^R] / E[K^R + K^Z]: fraction of the transfer surrogate spent
        # on lag selection vs source content (uses the RAW, unfloored K^Z).
        r_lag = kld_lag_loss / (kld_lag_loss + kld_content_raw + self.kl_eps)

        return {
            "feat_loss": feat_loss,
            "base_loss": base_loss,
            "kld_loss": kld_loss,
            "kld_lag_loss": kld_lag_loss,
            "kld_content_loss": kld_content_loss,
            "kld_content_raw": kld_content_raw,
            "total_loss": total_loss,
            "beta": torch.tensor(float(beta), device=device, dtype=dtype),
            "likelihood": likelihood,
            "mean_logvar_full": mean_logvar_full,
            "mean_logvar_base": mean_logvar_base,
            "lag_smoothness": lag_smoothness,
            "lag_tv": lag_tv,
            "lag_entropy_reg": lag_entropy_reg,
            # Section-26 diagnostics.
            "delta_l": delta_l,
            "rms_src": rms_src,
            "r_lag": r_lag,
        }

    # ------------------------------------------------------------------
    # Loss helpers (KL mask, DDP grad-mask) -- Sprint 4
    # ------------------------------------------------------------------

    def _kl_mask(
        self,
        batch: int,
        seq_len: int,
        *,
        weight: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""Return the per-step KL mask $m^{\mathrm{KL}}_{b,t}$ of shape ``(B, T)``.

        Combines the warm-up exclusion ($t < w_0 \to 0$) with the optional per-step
        dataset weight multiplicatively.
        """
        time_mask = torch.ones(seq_len, device=device, dtype=dtype)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            time_mask[:warmup] = 0.0
        mask = time_mask.unsqueeze(0).expand(batch, seq_len)
        if weight is not None:
            mask = mask * weight.to(device=device, dtype=dtype)
        return mask

    def _conditional_touch_params(self):
        r"""Yield the parameters of the source-conditioned (curriculum) branches.

        These are the parameters that can become disconnected from ``total_loss``
        when a curriculum stage disables a branch. The always-on backbone
        (target adapter/encoder, the baseline conditioning state, the shared
        horizon core, and the baseline decoder) is excluded because ``base_loss``
        always exercises it; the shared ``horizon_core`` is likewise excluded from
        the residual-decoder group (iterated as ``proj``/``mean_head``/
        ``logvar_head`` only) to avoid a double touch.
        """
        groups = (
            self.source_adapter.parameters(),
            self.source_encoder.parameters(),
            self.prior_head.norm.parameters(),
            self.prior_head.mu_head.parameters(),
            self.prior_head.logvar_head.parameters(),
            self.lag_embedding.parameters(),
            self.lag_bias.parameters(),
            self.lag_posterior.parameters(),
            self.lag_prior.parameters(),
            self.lag_latent_head.parameters(),
            self.residual_decoder.proj.parameters(),
            self.residual_decoder.mean_head.parameters(),
            self.residual_decoder.logvar_head.parameters(),
        )
        for group in groups:
            yield from group

    def _grad_touch(self, ref: torch.Tensor) -> torch.Tensor:
        r"""Return $\sum_p \operatorname{sum}(p)$ over the conditional branch params.

        Used as ``total_loss += 0.0 * _grad_touch(total_loss)`` so disabled
        branches receive a zero (not ``None``) gradient. The value is scaled by 0
        by the caller, so only its graph connectivity matters.
        """
        params = [
            p for p in self._conditional_touch_params() if p.requires_grad
        ]
        if not params:
            return ref.new_zeros(())
        acc = params[0].sum()
        for p in params[1:]:
            acc = acc + p.sum()
        return acc.to(device=ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------
    # Warm-up masking helpers (v1-compatible)
    # ------------------------------------------------------------------

    def _warmup_steps(self, seq_len: int) -> int:
        """Return the number of leading warm-up steps for ``seq_len``."""
        warmup = int(getattr(self, "warmup_period", 0) or 0)
        if warmup <= 0:
            return 0
        return min(seq_len, warmup)

    def _build_warmup_valid_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Return a bool ``(seq_len,)`` mask that is ``False`` on warm-up steps."""
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            mask[:warmup] = False
        return mask

    # ------------------------------------------------------------------
    # Latent running-statistics helpers (v1-compatible)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_latent_running_stats(self, mu_post: torch.Tensor) -> None:
        r"""Update ``mu_post_running_{mean,var,count}`` from a live batch.

        BatchNorm-style EMA with momentum ``self.latent_stats_momentum``; warm-up
        steps are masked out before aggregating.
        """
        valid_t = self._build_warmup_valid_mask(
            mu_post.size(1), device=mu_post.device
        )
        if not bool(valid_t.any()):
            return
        flat = mu_post[:, valid_t, :].reshape(-1, mu_post.size(-1))
        if flat.numel() == 0:
            return
        m = self.latent_stats_momentum
        self.mu_post_running_mean.mul_(1.0 - m).add_(flat.mean(dim=0), alpha=m)
        self.mu_post_running_var.mul_(1.0 - m).add_(
            flat.var(dim=0, unbiased=False), alpha=m
        )
        self.mu_post_running_count.add_(int(flat.size(0)))

    def normalize_latent(self, z: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        r"""Z-score ``z`` (or ``mu_post``) using the running latent statistics."""
        std = (self.mu_post_running_var + eps).sqrt()
        return (z - self.mu_post_running_mean) / std

    # ------------------------------------------------------------------
    # Closed-form KL helpers (v1-compatible)
    # ------------------------------------------------------------------

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
        r"""Closed-form diagonal-Gaussian KL $\operatorname{KL}(q\|p)$; ``(B,T,d_z)``."""
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
        r"""Aggregate the per-step closed-form KL into a scalar loss.

        Combines the warm-up mask with an optional per-step ``weight`` ``(B, T)``
        multiplicatively; ``free_bits > 0`` applies a per-element floor.
        """
        kld = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            mask_warmup=False,
        )
        if free_bits > 0.0:
            kld = kld.clamp(min=float(free_bits))
        B, T, d_z = kld.shape
        warmup = self._warmup_steps(T)
        device = kld.device
        dtype = kld.dtype
        time_mask = torch.ones(T, device=device, dtype=dtype)
        if warmup > 0:
            time_mask[:warmup] = 0.0
        full_mask = time_mask.unsqueeze(0).expand(B, T)
        if weight is not None:
            full_mask = full_mask * weight.to(device=device, dtype=dtype)
        mask_btd = full_mask.unsqueeze(-1)
        if reduce_mean:
            denom = mask_btd.sum() * float(d_z)
            if float(denom) <= 0.0:
                return torch.zeros((), device=device, dtype=dtype)
            return (kld * mask_btd).sum() / denom
        return (kld * mask_btd).sum()

    # ------------------------------------------------------------------
    # Batch adapter (v1-compatible)
    # ------------------------------------------------------------------

    def _default_batch_to_inputs(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract ``(y_st, y_ph, u_stream)`` from an attribute-style batch."""
        y_st = batch.fhr_st
        y_ph = batch.fhr_ph
        up_ph = batch.up_ph
        if self.use_up_st:
            up_st = batch.up_st
            u_stream = torch.cat([up_st, up_ph], dim=-1)
        else:
            u_stream = up_ph
        return y_st, y_ph, u_stream

    # ------------------------------------------------------------------
    # Encode-only / transfer-entropy helpers (S4-T05)
    # ------------------------------------------------------------------

    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        sample_z: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the encode path and return the latent + lag quantities (no decoders).

        Returns the prior/posterior latent parameters, the aggregated latent $z$
        (sampled iff the module is in training mode; otherwise the posterior mean),
        the target/source states, the lag posterior, and the KL decomposition. The
        decoder outputs are omitted.

        Args:
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            u_stream: Source stream ``(B, T, c_u)``.
            sample_z: If ``True`` return the model's aggregated latent ``z``;
                otherwise return the posterior mean ``mu_post`` as ``z``.

        Returns:
            A dict with ``mu_prior``, ``logvar_prior``, ``mu_post``,
            ``logvar_post``, ``z``, ``target_state``, ``source_state``,
            ``attended_source``, ``attended_source_heads``, ``attn_weights``,
            ``kld_per_t``, ``kld_lag``, ``kld_content``.
        """
        out = self.forward(y_st, y_ph, u_stream)
        z = out["z"] if sample_z else out["mu_post"]
        return {
            "mu_prior": out["mu_prior"],
            "logvar_prior": out["logvar_prior"],
            "mu_post": out["mu_post"],
            "logvar_post": out["logvar_post"],
            "z": z,
            "target_state": out["target_state"],
            "source_state": out["source_state"],
            "attended_source": out["attended_source"],
            "attended_source_heads": out["attended_source_heads"],
            "attn_weights": out["attn_weights"],
            "kld_per_t": out["kld_per_t"],
            "kld_lag": out.get("kld_lag"),
            "kld_content": out.get("kld_content"),
        }

    def physical_lag_axis(self) -> torch.Tensor:
        r"""Return the lag axis in physical seconds (arch spec section 27).

        Converts each decimated lag index $\ell \in \{0, \ldots, L-1\}$ to physical
        seconds via $\mathrm{lag}_{\mathrm{phys}}(\ell) = s\,\ell + \Delta_{UP}$,
        where $s$ is ``step_seconds`` (4 s/step) and $\Delta_{UP}$ is the fixed
        preprocessing UP shift ``delta_up_seconds``. Use it to label the lag axis of
        ``te_lag_map`` / ``expected_lag`` in both model-lag and second coordinates.

        Returns:
            A ``(L,)`` tensor of lag values in seconds.
        """
        ell = torch.arange(self.L, dtype=torch.float32)
        return self.step_seconds * ell + self.delta_up_seconds

    def expected_lag_seconds(self, expected_lag: torch.Tensor) -> torch.Tensor:
        r"""Convert a decimated expected lag $\bar\ell$ to seconds (section 27).

        Applies $s\,\bar\ell + \Delta_{UP}$ elementwise, so an ``expected_lag``
        tensor of any shape is mapped to physical seconds.

        Args:
            expected_lag: Expected lag(s) in decimated steps (any shape).

        Returns:
            The same-shaped tensor in physical seconds.
        """
        return self.step_seconds * expected_lag + self.delta_up_seconds

    # ------------------------------------------------------------------
    # Supervised outcome head (Stage 4) -- S7-T02
    # ------------------------------------------------------------------
    def freeze_vae(self) -> None:
        r"""Freeze every variational parameter, leaving only the outcome head trainable.

        Stage 4a of the curriculum (arch spec section 23): the classifier trains on
        a frozen transfer-entropy bottleneck so the supervised signal cannot distort
        the variational representation. A no-op on the VAE params if the outcome head
        is disabled.
        """
        self.requires_grad_(False)
        if self.outcome_head is not None:
            self.outcome_head.requires_grad_(True)

    def unfreeze_finetune(self) -> None:
        r"""Enable a light Stage-4b fine-tune subset (arch spec section 23).

        Starting from :meth:`freeze_vae`, re-enables the last target-encoder block
        (plus its final norm), the source lag-atom encoder, and the outcome head, so
        end-to-end fine-tuning perturbs only the shallow adapters near the head while
        the deep bottleneck stays fixed.
        """
        self.freeze_vae()
        if len(self.target_encoder.blocks) > 0:
            self.target_encoder.blocks[-1].requires_grad_(True)
        self.target_encoder.final_norm.requires_grad_(True)
        self.source_encoder.requires_grad_(True)

    def outcome_logits(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Return outcome-head logits ``(B, num_classes)`` from a forward dict.

        Args:
            forward_outputs: A source-path forward dict (must carry
                ``target_state``, ``z``, ``kld_per_t``, ``attn_weights``).
            valid_mask: Optional ``(B, T)`` pooling weight (warm-up exclusion times
                the dataset weight). Defaults to the warm-up mask when ``None``.

        Returns:
            Segment class logits ``(B, num_classes)``.

        Raises:
            RuntimeError: If the outcome head is not enabled.
        """
        if self.outcome_head is None:
            raise RuntimeError(
                "outcome_logits called but use_outcome_head=False; construct the "
                "model with use_outcome_head=True to enable the Stage-4 head."
            )
        if valid_mask is None:
            T = forward_outputs["target_state"].size(1)
            valid_mask = self._build_warmup_valid_mask(
                T, device=forward_outputs["target_state"].device
            ).unsqueeze(0)
        return self.outcome_head(forward_outputs, valid_mask)

    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        r"""Return the v2 transfer surrogate $K_t$ (arch spec sections 13/19).

        Unlike v1 (which returned the per-dimension flat KL ``(B, T, d_z)``), v2
        reports the model's own decomposed transfer surrogate
        $K_t = \sum_m (K^{R,(m)}_t + K^{Z,(m)}_t)$, so:

        * ``reduce_mean=False`` returns the per-step $K_t$ of shape ``(B, T)`` with
          the warm-up steps filled with ``NaN`` (matching v1's masking convention);
        * ``reduce_mean=True`` returns the scalar mean of $K_t$ over the non-warm-up
          steps.

        Runs under :func:`torch.no_grad` in ``eval`` mode.

        Args:
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            u_stream: Source stream ``(B, T, c_u)``.
            reduce_mean: If ``True`` return the scalar mean; else the ``(B, T)`` map.

        Returns:
            The scalar mean $K_t$, or the ``(B, T)`` per-step $K_t$ (warm-up NaN).
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                out = self.forward(y_st, y_ph, u_stream)
                kpt = out["kld_per_t"]                       # (B, T)
                warmup = self._warmup_steps(kpt.size(1))
                if reduce_mean:
                    mask = torch.ones_like(kpt)
                    if warmup > 0:
                        mask[:, :warmup] = 0.0
                    denom = mask.sum().clamp_min(1.0)
                    result = (kpt * mask).sum() / denom
                else:
                    kpt = kpt.clone()
                    if warmup > 0:
                        kpt[:, :warmup] = float("nan")
                    result = kpt
        finally:
            if was_training:
                self.train()
        return result

    @torch.no_grad()
    def fit_latent_stats(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        device: Optional[torch.device] = None,
        batch_to_inputs: Optional[
            Callable[[Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> int:
        r"""Fit the ``mu_post_running_*`` buffers over a loader (v1-compatible).

        Accumulates the mean and variance of ``mu_post`` over all non-warm-up time
        steps (DDP-summed when a process group is active) and overwrites the running
        buffers. Refuses a loader that yields zero valid samples.

        Args:
            dataloader: An iterable of batches consumable by ``batch_to_inputs``.
            max_batches: Optional cap on the number of batches consumed.
            device: Device to run on (defaults to the model's device).
            batch_to_inputs: Batch -> ``(y_st, y_ph, u_stream)`` extractor (defaults
                to :meth:`_default_batch_to_inputs`).

        Returns:
            The number of valid samples aggregated.

        Raises:
            RuntimeError: If the loader yields zero valid (non-warm-up) samples.
        """
        import torch.distributed as dist

        dist_active = dist.is_available() and dist.is_initialized()
        if device is None:
            device = next(self.parameters()).device
        if batch_to_inputs is None:
            batch_to_inputs = self._default_batch_to_inputs

        was_training = self.training
        self.eval()
        sum_x = torch.zeros(self.d_z, dtype=torch.float64, device=device)
        sum_xx = torch.zeros(self.d_z, dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)
        try:
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                y_st, y_ph, u_stream = batch_to_inputs(batch)
                y_st = y_st.to(device)
                y_ph = y_ph.to(device)
                u_stream = u_stream.to(device)
                enc = self.encode_only(y_st, y_ph, u_stream, sample_z=False)
                mu_post = enc["mu_post"]
                valid_t = self._build_warmup_valid_mask(
                    mu_post.size(1), device=mu_post.device
                )
                flat = mu_post[:, valid_t, :].reshape(-1, self.d_z).double()
                if flat.numel() == 0:
                    continue
                sum_x += flat.sum(0)
                sum_xx += (flat * flat).sum(0)
                count += flat.size(0)
        finally:
            if was_training:
                self.train()

        if dist_active:
            dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_xx, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        n_samples = int(count.item())
        if n_samples == 0:
            raise RuntimeError(
                "fit_latent_stats: dataloader yielded zero valid samples (no "
                "non-warm-up time steps). Check that the loader is not empty and "
                "that the sequence length exceeds warmup_period."
            )
        mean = sum_x / count
        var = (sum_xx / count - mean * mean).clamp_min(0.0)
        self.mu_post_running_mean.copy_(mean.to(self.mu_post_running_mean.dtype))
        self.mu_post_running_var.copy_(var.to(self.mu_post_running_var.dtype))
        self.mu_post_running_count.copy_(
            torch.tensor(
                n_samples,
                dtype=self.mu_post_running_count.dtype,
                device=self.mu_post_running_count.device,
            )
        )
        logger.info(
            f"fit_latent_stats: fitted latent statistics over {n_samples} "
            f"non-warm-up samples (d_z={self.d_z})."
        )
        return n_samples

    # ------------------------------------------------------------------
    # Curriculum stage-schedule mapping (S4-T04)
    # ------------------------------------------------------------------

    @staticmethod
    def default_curriculum_stages() -> "list[dict]":
        r"""Return the arch-spec section 23 default curriculum stage schedule.

        Stage 1 pretrains the target-only baseline; Stage 2 attaches the source
        residual with a weak KL and $K_a = 16$; Stage 3 warms $\beta$ to
        $\beta_{\max}$ and anneals $K_a$ to 8. Epoch boundaries are illustrative
        defaults; production runs override them via config.
        """
        return [
            {
                "start_epoch": 0, "active_lags": 8, "enable_source": False,
                "enable_residual": False, "enable_kl": False, "beta": 0.0,
            },
            {
                "start_epoch": 5, "active_lags": 16, "enable_source": True,
                "enable_residual": True, "enable_kl": True,
                "beta": {"kind": "constant", "value": 1.0e-4},
            },
            {
                "start_epoch": 10, "active_lags": 8, "enable_source": True,
                "enable_residual": True, "enable_kl": True,
                "beta": {
                    "kind": "linear_warmup", "start": 1.0e-4, "end": 5.0e-2,
                    "warmup_epochs": 50, "warmup_start_epoch": 10,
                },
            },
        ]

    @staticmethod
    def _resolve_stage_beta(spec: Any, epoch: int) -> float:
        r"""Resolve a per-stage $\beta$ from a scalar or a schedule dict.

        A number is returned verbatim. A dict with ``kind='constant'`` returns
        ``value``; ``kind='linear_warmup'`` ramps ``start`` -> ``end`` linearly over
        ``warmup_epochs`` epochs starting at ``warmup_start_epoch`` (default 0).
        """
        if isinstance(spec, bool):  # bool is an int subclass; treat as its value
            return float(spec)
        if isinstance(spec, (int, float)):
            return float(spec)
        if not isinstance(spec, dict):
            raise ValueError(
                f"beta spec must be a number or dict, got {type(spec).__name__!r}"
            )
        kind = str(spec.get("kind", "constant"))
        if kind == "constant":
            return float(spec.get("value", 0.0))
        if kind == "linear_warmup":
            start = float(spec.get("start", 1.0e-4))
            end = float(spec.get("end", 5.0e-2))
            warmup_epochs = int(spec.get("warmup_epochs", 50))
            warmup_start = int(spec.get("warmup_start_epoch", 0))
            if warmup_epochs <= 0:
                return end
            frac = min(
                1.0,
                max(0.0, (float(epoch) - warmup_start) / float(warmup_epochs)),
            )
            return start + (end - start) * frac
        raise ValueError(
            f"unknown beta kind {kind!r}; expected 'constant' or 'linear_warmup'"
        )

    @staticmethod
    def _resolve_stage(epoch: int, stages) -> Dict[str, Any]:
        r"""Map ``epoch`` to the active curriculum stage settings (pure function).

        Selects the stage with the GREATEST ``start_epoch`` that is $\le$ ``epoch``
        (order-independent; falling back to the earliest stage before any boundary)
        and resolves its $\beta$.

        Args:
            epoch: Current epoch index.
            stages: List of stage dicts (see :meth:`default_curriculum_stages`);
                need not be pre-sorted.

        Returns:
            A dict ``{active_lags, enable_source, enable_residual, enable_kl,
            beta}``.
        """
        if not stages:
            raise ValueError("stages schedule must be a non-empty list")
        eligible = [
            s for s in stages if int(s.get("start_epoch", 0)) <= int(epoch)
        ]
        if eligible:
            active = max(eligible, key=lambda s: int(s.get("start_epoch", 0)))
        else:
            # ``epoch`` precedes every stage boundary: use the earliest stage.
            active = min(stages, key=lambda s: int(s.get("start_epoch", 0)))
        beta = SeqVaeLagAttnV2._resolve_stage_beta(active.get("beta", 0.0), epoch)
        return {
            "active_lags": int(active.get("active_lags", 8)),
            "enable_source": bool(active.get("enable_source", True)),
            "enable_residual": bool(active.get("enable_residual", True)),
            "enable_kl": bool(active.get("enable_kl", True)),
            "beta": float(beta),
        }

    @staticmethod
    def _resolve_active_lags(epoch: int, stages) -> int:
        r"""Return the active-lag count $K_a$ for ``epoch`` from ``stages``."""
        return int(SeqVaeLagAttnV2._resolve_stage(epoch, stages)["active_lags"])

    def set_curriculum_stage(self, epoch: int, stages) -> float:
        r"""Apply the resolved stage flags / active-lag count and return its $\beta$.

        Mutates ``self.enable_source``, ``self.enable_residual``,
        ``self.enable_kl``, and ``self.active_lags`` in place; the returned $\beta$
        is fed to :meth:`compute_loss` by the Lightning epoch hook (S5-T02).
        """
        st = self._resolve_stage(epoch, stages)
        self.enable_source = st["enable_source"]
        self.enable_residual = st["enable_residual"]
        self.enable_kl = st["enable_kl"]
        self.active_lags = st["active_lags"]
        return st["beta"]


if __name__ == "__main__":
    # Sprint 0 skeleton smoke: construct the model from a full v1 kwarg dict, the
    # fallback (no up_st) config, and confirm buffers/attributes exist. The
    # forward/loss paths are assembled in Sprints 1-3, so they are not exercised
    # here. This smoke grows per sprint.
    torch.manual_seed(0)

    model = SeqVaeLagAttnV2(use_up_st=True)
    assert model.c_u == 101
    assert model.M == 4 and model.d_z_m == 6 and model.d_v == 32
    assert model.L == 91
    assert model.d_z_residual == 24 + 4 * 32
    assert model.mu_post_running_mean.shape == (24,)
    assert int(model.mu_post_running_count.item()) == 0

    model_fb = SeqVaeLagAttnV2(use_up_st=False, c_u=58)
    assert model_fb.c_u == 58

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[skeleton] built SeqVaeLagAttnV2; params so far = {n_params:,}")

    # entmax_1.5 sanity: sums to 1, non-negative, sparse.
    _scores = torch.randn(2, 4, 91)
    _scores[..., 0] += 8.0
    _alpha = entmax15(_scores, dim=-1)
    assert torch.allclose(_alpha.sum(-1), torch.ones(2, 4), atol=1e-6)
    assert (_alpha == 0.0).any()
    print(f"[entmax] entmax15 OK; nnz-frac = {(_alpha > 0).float().mean().item():.3f}")

    # Sprint 1 baseline-only forward + base loss (source disabled). T is well
    # past warmup_period + horizon so the base loss has valid anchor support.
    B, T = 2, 128
    base_model = SeqVaeLagAttnV2(enable_source=False).eval()
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u_full = torch.randn(B, T, 101)
    outs = base_model(y_st, y_ph, u_full)
    assert outs["mu_base"].shape == (B, T, 30, 87)
    assert torch.equal(outs["mu_full"], outs["mu_base"])
    assert float(outs["delta_mu_src"].abs().max()) == 0.0
    losses = base_model.compute_loss(outs, y_st, y_ph, beta=0.01)
    for k in ("feat_loss", "base_loss", "kld_loss", "total_loss"):
        assert torch.isfinite(losses[k])
    losses["total_loss"].backward()
    n_params = sum(p.numel() for p in base_model.parameters())
    print(
        f"[baseline] forward+loss OK; params={n_params:,}"
        f"  base_loss={losses['base_loss'].item():.4f}"
    )

    # Sprint 3-4: full source-enabled forward + decomposed KL + backward.
    src_model = SeqVaeLagAttnV2(use_entmax=True)
    src_model.eval()
    outs = src_model(y_st, y_ph, u_full)
    # Warm start: residual is zero, so mu_full == mu_base at init.
    assert float(outs["delta_mu_src"].abs().max()) < 1e-6
    assert torch.allclose(outs["mu_full"], outs["mu_base"], atol=1e-6)
    # Exact KL decomposition identities.
    add1 = (outs["kld_per_t"] - (outs["kld_lag"] + outs["kld_content"]).sum(-1)).abs().max()
    add2 = (outs["te_lag_map"].sum(-1) - outs["kld_per_t"]).abs().max()
    assert float(add1) < 1e-4 and float(add2) < 1e-4
    print(
        f"[source] full forward OK; K_t additivity={float(add1):.2e}"
        f"  sum_l te==K_t={float(add2):.2e}"
    )

    src_model.train()
    outs = src_model(y_st, y_ph, u_full)
    losses = src_model.compute_loss(
        outs, y_st, y_ph, beta=5e-2, likelihood="gaussian_nll", sigma_obs=1.0,
        detach_baseline_in_full=True,
    )
    for k in ("feat_loss", "base_loss", "kld_loss", "kld_lag_loss",
              "kld_content_loss", "total_loss", "lag_tv", "lag_entropy_reg"):
        assert torch.isfinite(losses[k]), k
    losses["total_loss"].backward()
    print(
        f"[source] decomposed loss+backward OK; kld_loss={losses['kld_loss'].item():.4f}"
        f"  feat={losses['feat_loss'].item():.4f}"
    )

    # Curriculum stage mapping + transfer-entropy helper.
    stages = SeqVaeLagAttnV2.default_curriculum_stages()
    st = SeqVaeLagAttnV2._resolve_stage(0, stages)
    assert st["enable_source"] is False and st["active_lags"] == 8
    te = src_model.measure_transfer_entropy(y_st, y_ph, u_full, reduce_mean=True)
    assert torch.isfinite(te)
    print(f"[curriculum] stage map + measure_transfer_entropy OK; TE_mean={te.item():.4f}")

    print("[v2] Sprint 0-4 smoke OK.")
