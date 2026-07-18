r"""Lag-Attentive Residual VAE-TEB (v3).

FROZEN. Superseded by the flattened tree at ``teb_vae/lag_attn/``; fixes and new work land there,
not here. Kept only because ``model_experiment/``, ``model_raw/`` and ``new_classifier/`` still
import this lag-attention cluster -- delete it once those retire. Freezing v1 and v3 together also
freezes ``SeqVaeRawV4``'s base.

This module implements :class:`SeqVaeLagAttnV3`, a *scientific-cleanliness* fork of
:class:`~model.vae_teb_prediction.model.vae_teb_lag_attn_v1.SeqVaeLagAttnV1`. It keeps
v1's architecture byte-for-byte (all backbone blocks are imported unchanged) and changes
only how the latent log-variances are produced, bounded, and how the reported
transfer-entropy (TE) surrogate KL is masked and summarised.

The reported TE surrogate is the per-step KL
:math:`K_t = \mathrm{KL}\!\left(q(z_t \mid Y_{\le t}, U_{\le t}) \,\|\,
p(z_t \mid Y_{\le t})\right)`. In v1 the posterior log-variance is an *independent*
random head, so at initialisation :math:`q \neq p` and :math:`K_t` is a nonzero
artifact of head mismatch rather than *earned* by source conditioning. v3 fixes this.

Goals implemented here (spec ``vae-teb-lag-attn-v3-spec-and-sprints.md``, Sprints 0-3):

* **G0 - Causal history states.** ``causal_norm=True`` replaces the
  :class:`torch.nn.GroupNorm` modules inside both encoders with :class:`CausalGroupNorm`.
  v1 applies ``GroupNorm`` to ``(B, C, T)`` tensors, which pools its normalising statistics
  *across time*, so :math:`H_y[t]` is a function of :math:`Y_{>t}` and the "prior"
  :math:`p(z_t \mid Y_{\le t})` secretly conditions on the future. Without this fix
  :math:`K_t` is not a transfer-entropy surrogate at all, whatever G1-G6 do to it.
* **G1 - Posterior log-variance residual + zero-init.** With
  ``posterior_logvar='residual'`` the posterior log-variance is
  :math:`\log\sigma^{2,q}_t = \mathrm{smoothbound}\!\left(\widetilde{\log\sigma^{2,p}_t}
  + \Delta\ell_t\right)`, where :math:`\widetilde{\log\sigma^{2,p}_t}` is the prior's
  *pre-bound* raw log-variance and :math:`\Delta\ell_t = s_\ell \tanh(\widetilde{\Delta
  \ell}_t / s_\ell)` is produced by a zero-initialised residual head. At init
  :math:`\Delta\ell_t = 0`, so :math:`\log\sigma^{2,q}_t = \log\sigma^{2,p}_t` exactly and
  (with the delta-mean head also zeroed) :math:`K_t \equiv 0`. The
  ``posterior_logvar='independent'`` path reproduces v1's independent clamped head for
  golden parity.
* **G2 - Smooth log-variance bounds.** ``logvar_bound='smooth'`` replaces the hard
  ``torch.clamp`` with :math:`\ell = \mathrm{lo} + (\mathrm{hi}-\mathrm{lo})\,\sigma(r)`
  over the same effective range :math:`[-5, 3]`, eliminating the zero-gradient plateaus of
  clamp. ``logvar_bound='clamp'`` (idempotent) reproduces exact v1 numerics.
* **G3 - KL support aligned to forecast anchors.** ``kld_support='anchor'`` trains the KL
  only over supervised anchors :math:`t \in [w_{\mathrm{warm}}, T-H)`;
  ``kld_support='full'`` reproduces v1's warm-up-only support.
* **G4 - Raw-vs-free-bit KL reporting.** :meth:`compute_loss` returns ``kld_raw``,
  ``kld_train`` (free-bit-floored, the optimised term), and ``kld_active_frac`` distinctly;
  only ``kld_train`` enters ``total_loss``.
* **G5 - ALiBi lag decay.** ``lag_bias_init='alibi_decay'`` turns on v1's learnable
  ``(num_heads, L)`` score bias, seeded with a per-head negative slope :math:`-m_h \ell`, so
  the attention starts with a physiologic short-lag prior instead of a flat one.
* **G6 - Source-permutation control.** :meth:`permutation_kl` and its fused training-time
  twin :meth:`perm_kl_from_forward` measure
  :math:`K_{\mathrm{shuffled}} = \mathrm{KL}(q(z \mid Y, \pi(U)) \,\|\, p(z \mid Y))` under a
  batch derangement :math:`\pi` (see :func:`make_derangement`). Used as an auxiliary loss it
  pushes :math:`q` back onto :math:`p` when the source is uninformative; used as a readout it
  is the negative control that establishes :math:`K_{\mathrm{raw}}` is source-specific.

The forward dict gains exactly two additive keys over v1 -- ``kld_active_frac`` (G4) and
``raw_logvar_prior`` (G6) -- so the testing pipeline's forward contract is preserved.

The **parity configuration** ``posterior_logvar='independent'``, ``logvar_bound='clamp'``,
``kld_support='full'``, ``causal_norm=False`` (with ``sigma_obs=1.0``,
``lag_bias_init='normal'``, ``lambda_perm=0``) reduces v3 to v1 exactly, tensor-for-tensor and
loss-term-for-loss-term.

Shape conventions follow v1 (``B, T, C_y, C_u, d_model, d_z, H_d, L, M_heads``).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (
    BaselineFutureDecoder,
    PosteriorHead,
    PriorHead,
    ResidualFutureDecoder,
    SeqVaeLagAttnV1,
)

# Latent dim is counted "active" when its mean per-step KL exceeds this many nats.
_KLD_ACTIVE_EPS = 1e-2

_BOUND_CHOICES = ("clamp", "smooth")
_POSTERIOR_LOGVAR_CHOICES = ("independent", "residual")
_KLD_SUPPORT_CHOICES = ("full", "anchor")


# =============================================================================
# Causal normalisation (G0)
# =============================================================================
class CausalGroupNorm(nn.Module):
    r"""Group normalisation over channels only, with **no pooling across time**.

    The blocks v1 calls "causal" normalise with :class:`torch.nn.GroupNorm` applied to a
    ``(B, C, T)`` tensor. ``GroupNorm`` reduces over *every* non-batch dimension inside a
    group, i.e. over :math:`(C/G, T)` -- so the mean and variance at step :math:`t` are
    functions of the whole sequence, including :math:`t' > t`. Every "history" state therefore
    carries a low-bandwidth image of the future, which silently invalidates the
    transfer-entropy reading of :math:`K_t` (whose prior is supposed to condition on
    :math:`Y_{\le t}` alone).

    This module reduces over the channels of each group **at each timestep independently**:

    $$\hat{x}_{b,c,t} = \frac{x_{b,c,t} - \mu_{b,g(c),t}}{\sqrt{\sigma^2_{b,g(c),t} +
    \epsilon}}\,\gamma_c + \beta_c, \qquad
    \mu_{b,g,t} = \frac{G}{C}\sum_{c \in g} x_{b,c,t}.$$

    It registers exactly the parameters :class:`torch.nn.GroupNorm` does (``weight`` and
    ``bias``, both of shape ``(C,)``) under the same names, so a v1 ``state_dict`` still aligns
    key-for-key and shape-for-shape and warm-start is unaffected.

    Shapes:
        Input:  ``(B, C, T)``
        Output: ``(B, C, T)``
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """Initialize with the same signature as :class:`torch.nn.GroupNorm`.

        Args:
            num_groups: Number of channel groups :math:`G`; must divide ``num_channels``.
            num_channels: Channel count :math:`C`.
            eps: Numerical-stability term added to the variance.

        Raises:
            ValueError: If ``num_channels`` is not divisible by ``num_groups``.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``(B, C, T)`` per group per timestep."""
        B, C, T = x.shape
        grouped = x.view(B, self.num_groups, C // self.num_groups, T)
        mean = grouped.mean(dim=2, keepdim=True)
        var = grouped.var(dim=2, unbiased=False, keepdim=True)
        normed = ((grouped - mean) / torch.sqrt(var + self.eps)).view(B, C, T)
        return normed * self.weight[None, :, None] + self.bias[None, :, None]

    def extra_repr(self) -> str:
        """Mirror :class:`torch.nn.GroupNorm`'s repr for readable module trees."""
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}"


def causalize_norms(module: nn.Module) -> int:
    """Recursively replace every :class:`torch.nn.GroupNorm` with :class:`CausalGroupNorm`.

    The replacement inherits the original affine parameters, so this is numerically a no-op
    for the affine transform and changes only which elements the normalising statistics are
    pooled over.

    Args:
        module: Subtree to rewrite **in place**.

    Returns:
        The number of modules replaced.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            causal = CausalGroupNorm(child.num_groups, child.num_channels, child.eps)
            if child.affine:
                with torch.no_grad():
                    causal.weight.copy_(child.weight)
                    causal.bias.copy_(child.bias)
            causal.to(child.weight.device if child.affine else torch.device("cpu"))
            setattr(module, name, causal)
            replaced += 1
        else:
            replaced += causalize_norms(child)
    return replaced


# =============================================================================
# Bound helpers (G2)
# =============================================================================
def smooth_bound(r: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    r"""Smoothly map a raw value into ``(lo, hi)`` via a scaled sigmoid.

    Computes :math:`\ell = \mathrm{lo} + (\mathrm{hi}-\mathrm{lo})\,\sigma(r)`. Unlike
    :func:`torch.clamp`, the gradient is strictly positive everywhere (no zero-gradient
    plateaus), so a saturated log-variance can still recover. The map is **not**
    idempotent -- callers must apply it to a *raw* value, never to an already-bounded one.

    Args:
        r: Raw pre-bound tensor.
        lo: Lower asymptote of the output range.
        hi: Upper asymptote of the output range.

    Returns:
        A tensor with the same shape as ``r`` lying in the open interval ``(lo, hi)``.
    """
    return lo + (hi - lo) * torch.sigmoid(r)


def _apply_logvar_bound(
    raw: torch.Tensor, clamp: Tuple[float, float], bound: str
) -> torch.Tensor:
    """Bound a raw log-variance by clamp (idempotent, v1) or smooth sigmoid (v3).

    Args:
        raw: Raw pre-bound log-variance tensor.
        clamp: ``(lo, hi)`` effective range shared by both bound modes.
        bound: Either ``'clamp'`` or ``'smooth'``.

    Returns:
        The bounded log-variance tensor.
    """
    lo, hi = float(clamp[0]), float(clamp[1])
    if bound == "clamp":
        return torch.clamp(raw, min=lo, max=hi)
    return smooth_bound(raw, lo, hi)


def _validate_choice(value: str, choices: Tuple[str, ...], name: str) -> str:
    """Validate that ``value`` is one of ``choices`` (raise ``ValueError`` otherwise)."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


# =============================================================================
# Source-permutation control helpers (G6)
# =============================================================================
def make_derangement(
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Draw a batch-index derangement :math:`\pi` with :math:`\pi(i) \neq i` for all
    :math:`i`.

    Uses **Sattolo's algorithm**, which draws uniformly from the *cyclic* permutations of
    :math:`\{0, \dots, B-1\}`. Every cyclic permutation of length :math:`B \ge 2` is
    fixed-point-free, so the guarantee is structural rather than probabilistic -- unlike
    rejection sampling from :func:`torch.randperm`, this always terminates in :math:`O(B)`
    and can never leak an identity mapping into the permutation control.

    Args:
        batch_size: Batch size :math:`B`. Must be at least 2.
        generator: Optional CPU :class:`torch.Generator` for reproducibility.
        device: Device of the returned index tensor. Defaults to CPU.

    Returns:
        A ``(B,)`` ``torch.long`` tensor holding :math:`\pi`, with no fixed points.

    Raises:
        ValueError: If ``batch_size < 2``, where no derangement exists.
    """
    if batch_size < 2:
        raise ValueError(
            f"a derangement requires batch_size >= 2, got {batch_size}; callers must "
            "skip the permutation control for degenerate batches"
        )
    perm = list(range(batch_size))
    for i in range(batch_size - 1, 0, -1):
        j = int(torch.randint(0, i, (1,), generator=generator).item())
        perm[i], perm[j] = perm[j], perm[i]
    return torch.tensor(perm, dtype=torch.long, device=device)


# =============================================================================
# Bound-only head subclasses (G2)
# =============================================================================
class PriorHeadV3(PriorHead):
    r"""v1 :class:`PriorHead` with a selectable log-variance bound.

    Adds the ``logvar_bound`` switch and additionally returns the prior's *pre-bound* raw
    log-variance :math:`\widetilde{\log\sigma^{2,p}_t}`, which :class:`PosteriorHeadV3`
    needs to build an exact residual (so that ``smooth_bound(raw + 0) == smooth_bound(raw)``
    gives KL :math:`\equiv 0` at init).
    """

    def __init__(self, *args, logvar_bound: str = "clamp", **kwargs) -> None:
        """Initialize the prior head.

        Args:
            *args: Forwarded to :class:`PriorHead`.
            logvar_bound: ``'clamp'`` (v1) or ``'smooth'`` (v3).
            **kwargs: Forwarded to :class:`PriorHead`.
        """
        super().__init__(*args, **kwargs)
        self.logvar_bound = _validate_choice(logvar_bound, _BOUND_CHOICES, "logvar_bound")

    def forward(
        self, h_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(mu_prior, logvar_prior, decoder_state, raw_logvar_prior)``."""
        raw_mu = self.mu_prior_head(self.mu_input_norm(h_y))
        mu_prior = self.mu_scale * torch.tanh(raw_mu / self.mu_scale)

        raw_logvar_prior = self.logvar_prior_head(self.logvar_input_norm(h_y))
        logvar_prior = _apply_logvar_bound(
            raw_logvar_prior, self.logvar_clamp, self.logvar_bound
        )
        decoder_state = self.decoder_state_head(self.dec_input_norm(h_y))
        return mu_prior, logvar_prior, decoder_state, raw_logvar_prior


class BaselineFutureDecoderV3(BaselineFutureDecoder):
    """v1 :class:`BaselineFutureDecoder` with a selectable log-variance bound (G2)."""

    def __init__(self, *args, logvar_bound: str = "clamp", **kwargs) -> None:
        """Initialize the baseline decoder (see :class:`BaselineFutureDecoder`)."""
        super().__init__(*args, **kwargs)
        self.logvar_bound = _validate_choice(logvar_bound, _BOUND_CHOICES, "logvar_bound")

    def forward(
        self, decoder_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_base, logvar_base)`` each of shape ``(B, T, H_d, C)``."""
        h = self.proj(decoder_state)
        feat = self.core.decode(h)
        mu_base = self.mean_head(feat)
        logvar_base = _apply_logvar_bound(
            self.logvar_head(feat), self.logvar_clamp, self.logvar_bound
        )
        return mu_base, logvar_base


class ResidualFutureDecoderV3(ResidualFutureDecoder):
    """v1 :class:`ResidualFutureDecoder` with a selectable log-variance bound (G2)."""

    def __init__(self, *args, logvar_bound: str = "clamp", **kwargs) -> None:
        """Initialize the residual decoder (see :class:`ResidualFutureDecoder`)."""
        super().__init__(*args, **kwargs)
        self.logvar_bound = _validate_choice(logvar_bound, _BOUND_CHOICES, "logvar_bound")

    def forward(
        self, decoder_state: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(delta_mu_src, logvar_full)`` of shape ``(B, T, H_d, C)``."""
        h_in = torch.cat([decoder_state, z], dim=-1)
        h = self.proj(h_in)
        feat = self.core.decode(h)
        delta_mu_src = self.mean_head(feat)
        logvar_full = _apply_logvar_bound(
            self.logvar_head(feat), self.logvar_clamp, self.logvar_bound
        )
        return delta_mu_src, logvar_full


# =============================================================================
# Posterior head (G1)
# =============================================================================
class PosteriorHeadV3(PosteriorHead):
    r"""Source-conditioned posterior with a zero-init residual log-variance (G1).

    In ``posterior_logvar='residual'`` mode the independent v1 log-variance head is removed
    and replaced by a zero-initialised residual head ``delta_logvar_head`` mirroring
    ``delta_mu_head``. The posterior log-variance is then

    .. math::
        \log\sigma^{2,q}_t = \mathrm{smoothbound}\!\left(\widetilde{\log\sigma^{2,p}_t}
        + s_\ell \tanh(\widetilde{\Delta\ell}_t / s_\ell)\right),

    using the prior's **pre-bound** raw log-variance :math:`\widetilde{\log\sigma^{2,p}_t}`.
    Because the delta head is zero-initialised, at init :math:`\log\sigma^{2,q}_t =
    \log\sigma^{2,p}_t` exactly, and with the zero-init delta-mean head :math:`K_t \equiv 0`.

    In ``posterior_logvar='independent'`` mode the head is byte-identical to v1's
    :class:`PosteriorHead` (used for golden parity).
    """

    def __init__(
        self,
        d_model: int = 128,
        d_z: int = 24,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        dropout: float = 0.1,
        delta_mu_scale: float = 3.0,
        head_structured: bool = False,
        num_heads: int = 4,
        d_head: int = 32,
        *,
        logvar_bound: str = "clamp",
        posterior_logvar: str = "independent",
        delta_logvar_scale: float = 2.0,
    ) -> None:
        r"""Initialize the posterior head.

        Args:
            d_model: Encoder state width.
            d_z: Latent dimensionality.
            logvar_clamp: ``(lo, hi)`` effective range of the log-variance bound.
            dropout: Dropout used inside every internal ResidualMLP.
            delta_mu_scale: Saturation magnitude of the tanh-bounded posterior mean delta.
            head_structured: C7 toggle (partition the latent into ``num_heads`` groups).
            num_heads: Number of latent groups when ``head_structured``.
            d_head: Per-head summary width when ``head_structured``.
            logvar_bound: ``'clamp'`` (v1) or ``'smooth'`` (v3).
            posterior_logvar: ``'independent'`` (v1 clamped head) or ``'residual'`` (zero-init
                residual around the prior's pre-bound raw log-variance).
            delta_logvar_scale: Saturation magnitude :math:`s_\ell` of the tanh-bounded
                log-variance delta (residual mode only). Must be positive.
        """
        super().__init__(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            delta_mu_scale=delta_mu_scale,
            head_structured=head_structured,
            num_heads=num_heads,
            d_head=d_head,
        )
        self.logvar_bound = _validate_choice(logvar_bound, _BOUND_CHOICES, "logvar_bound")
        self.posterior_logvar = _validate_choice(
            posterior_logvar, _POSTERIOR_LOGVAR_CHOICES, "posterior_logvar"
        )
        if delta_logvar_scale <= 0.0:
            raise ValueError(
                f"delta_logvar_scale must be > 0, got {delta_logvar_scale}"
            )
        self.delta_logvar_scale = float(delta_logvar_scale)

        if self.posterior_logvar == "residual":
            # Replace the independent v1 log-variance head with a zero-init residual head
            # mirroring ``delta_mu_head``. The independent head is removed so it does not
            # dangle as an unused parameter (DDP no-unused-params in learned-variance runs).
            if self.head_structured:
                fuse_out = max(2 * self.group, 16)
                self.delta_logvar_head = nn.ModuleList(
                    [nn.Linear(fuse_out, self.group) for _ in range(self.num_heads)]
                )
            else:
                self.delta_logvar_head = nn.Linear(d_model, d_z)
            del self.logvar_post_head

    def forward(
        self,
        h_y: torch.Tensor,
        a: torch.Tensor,
        mu_prior: torch.Tensor,
        raw_logvar_prior: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_post, logvar_post)``.

        Args:
            h_y: Target state ``(B, T, d_model)``.
            a: Attended source summary -- ``(B, T, d_model)`` (flat) or
                ``(B, T, num_heads, d_head)`` (head-structured).
            mu_prior: Prior mean ``(B, T, d_z)`` for the residual add.
            raw_logvar_prior: Prior *pre-bound* raw log-variance ``(B, T, d_z)``. Required
                when ``posterior_logvar='residual'``; ignored otherwise.
        """
        residual = self.posterior_logvar == "residual"
        if self.head_structured:
            hy = self.h_y_norm(h_y)
            raw_deltas, logvar_terms = [], []
            for m in range(self.num_heads):
                a_m = self.a_head_norm(a[:, :, m, :])
                fused_m = self.fusion[m](torch.cat([hy, a_m], dim=-1))
                raw_deltas.append(self.delta_mu_head[m](fused_m))
                head = self.delta_logvar_head[m] if residual else self.logvar_post_head[m]
                logvar_terms.append(head(fused_m))
            raw_delta = torch.cat(raw_deltas, dim=-1)
            logvar_term = torch.cat(logvar_terms, dim=-1)
        else:
            fused = self.fusion(
                torch.cat([self.h_y_norm(h_y), self.a_norm(a)], dim=-1)
            )
            raw_delta = self.delta_mu_head(fused)
            head = self.delta_logvar_head if residual else self.logvar_post_head
            logvar_term = head(fused)

        # tanh(0) = 0, so with the delta heads zero-inited both deltas are identically 0
        # at step 0 (warm-start / zero-KL invariant).
        delta_mu = self.delta_mu_scale * torch.tanh(raw_delta / self.delta_mu_scale)
        mu_post = mu_prior + delta_mu

        if residual:
            if raw_logvar_prior is None:
                raise ValueError(
                    "posterior_logvar='residual' requires raw_logvar_prior; "
                    "call via SeqVaeLagAttnV3.forward / encode_only."
                )
            delta_logvar = self.delta_logvar_scale * torch.tanh(
                logvar_term / self.delta_logvar_scale
            )
            # Bound the *summed raw* value (smooth_bound is not idempotent), so at init
            # (delta_logvar == 0) logvar_post == logvar_prior exactly.
            logvar_post = _apply_logvar_bound(
                raw_logvar_prior + delta_logvar, self.logvar_clamp, self.logvar_bound
            )
        else:
            logvar_post = _apply_logvar_bound(
                logvar_term, self.logvar_clamp, self.logvar_bound
            )
        return mu_post, logvar_post


# =============================================================================
# Top-level model
# =============================================================================
class SeqVaeLagAttnV3(SeqVaeLagAttnV1):
    r"""Lag-attentive residual VAE-TEB, v3.

    A subclass of :class:`SeqVaeLagAttnV1` that swaps the four log-variance-producing heads
    for their v3 variants and rewires the KL support/reporting. All backbone blocks
    (adapters, encoders, lag attention, shared horizon core, TE-analysis head) are inherited
    unchanged. The parity configuration (``posterior_logvar='independent'``,
    ``logvar_bound='clamp'``, ``kld_support='full'``) reduces v3 to v1 exactly.
    """

    #: Stored in checkpoints for the version-agnostic testing load-path guard.
    model_class = "SeqVaeLagAttnV3"

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
        alibi_slope_scale: float = 1.0,
        head_structured_latent: bool = False,
        init_weights: bool = True,
        logvar_bound: str = "clamp",
        posterior_logvar: str = "independent",
        delta_logvar_scale: float = 2.0,
        kld_support: str = "full",
        lambda_perm: float = 0.0,
        perm_every_n_batches: int = 4,
        causal_norm: bool = False,
        freeze_unused_attn_proj: bool = False,
    ) -> None:
        r"""Initialize ``SeqVaeLagAttnV3``.

        All :class:`SeqVaeLagAttnV1` constructor arguments are listed explicitly (rather than
        absorbed into ``**kwargs``) so the testing pipeline's ``inspect.signature``-based
        ``_lag_attn_kwargs_from_config`` discovers them; see the v1 docstring for their
        meaning. The v3-specific arguments are:

        Args:
            logvar_bound: Log-variance bound mode ``'clamp'`` (v1) or ``'smooth'`` (v3, G2),
                applied to the prior, posterior, and both decoder log-variance heads.
            posterior_logvar: ``'independent'`` (v1 clamped head, for parity) or
                ``'residual'`` (zero-init residual around the prior, G1).
            delta_logvar_scale: Saturation magnitude :math:`s_\ell` of the posterior
                log-variance delta (residual mode).
            kld_support: Training-KL time support -- ``'full'`` (``[warmup, T)``, v1) or
                ``'anchor'`` (``[warmup, T-H)``, G3). Read by :meth:`_kld_loss` /
                :meth:`_kld_support_mask`.
            lambda_perm: Weight of the source-permutation control :math:`L_{\mathrm{perm}}`
                (G6). Consumed by the trainer, not by :meth:`compute_loss`.
            perm_every_n_batches: Permutation-control schedule period (G6). Consumed by the
                trainer.
            causal_norm: If True, replace every :class:`torch.nn.GroupNorm` inside the target
                and source encoders with :class:`CausalGroupNorm`, restoring strict causality
                of :math:`H_y` and :math:`H_u` (G0). ``False`` keeps v1's time-pooling
                ``GroupNorm`` and is required for Sprint-0 golden parity. The decoders and the
                shared horizon core are left alone: their ``GroupNorm``s pool over the
                *forecast-horizon* axis within a single anchor, which mixes outputs of one
                anchor rather than inputs across time, and a direct probe confirms they leak
                nothing.
            freeze_unused_attn_proj: Only meaningful with ``head_structured_latent=True``,
                where the posterior consumes the per-head summaries ``A_heads`` and the
                attention's output projection ``lag_attn.W_o`` feeds nothing but the diagnostic
                ``attended_source`` key. ``W_o`` therefore receives no gradient and is never
                updated by the optimiser -- in v1 either. Setting this flag makes that explicit
                by clearing ``requires_grad``, which is numerically a no-op but removes the
                parameter from DDP's expectation set, so the run can use plain ``'ddp'``
                instead of paying for ``ddp_find_unused_parameters_true`` on every step.
        """
        super().__init__(
            sequence_length=sequence_length,
            d_model=d_model,
            d_z=d_z,
            horizon=horizon,
            warmup_period=warmup_period,
            c_y=c_y,
            c_u=c_u,
            use_up_st=use_up_st,
            max_lag=max_lag,
            num_heads=num_heads,
            d_head=d_head,
            lstm_layers=lstm_layers,
            dropout=dropout,
            decoder_hidden=decoder_hidden,
            horizon_depth=horizon_depth,
            horizon_kernel=horizon_kernel,
            horizon_film=horizon_film,
            encoder_extra_dilations=encoder_extra_dilations,
            logvar_clamp=logvar_clamp,
            mu_scale=mu_scale,
            delta_mu_scale=delta_mu_scale,
            latent_stats_momentum=latent_stats_momentum,
            use_entmax=use_entmax,
            attention_grad_checkpoint=attention_grad_checkpoint,
            lag_bias_init=lag_bias_init,
            alibi_slope_scale=alibi_slope_scale,
            head_structured_latent=head_structured_latent,
            init_weights=init_weights,
        )

        self.logvar_bound = _validate_choice(logvar_bound, _BOUND_CHOICES, "logvar_bound")
        self.posterior_logvar = _validate_choice(
            posterior_logvar, _POSTERIOR_LOGVAR_CHOICES, "posterior_logvar"
        )
        self.delta_logvar_scale = float(delta_logvar_scale)
        self.kld_support = _validate_choice(
            kld_support, _KLD_SUPPORT_CHOICES, "kld_support"
        )
        self.lambda_perm = float(lambda_perm)
        self.perm_every_n_batches = int(perm_every_n_batches)

        self.causal_norm = bool(causal_norm)
        if self.causal_norm:
            self.n_causalized_norms = causalize_norms(
                self.target_encoder
            ) + causalize_norms(self.source_encoder)
        else:
            self.n_causalized_norms = 0

        # In head-structured mode the posterior reads ``A_heads`` (pre-projection), so
        # ``lag_attn.W_o`` reaches nothing but the diagnostic ``attended_source`` key. It gets
        # no gradient and the optimiser never touches it. Clearing ``requires_grad`` states
        # that explicitly and, more usefully, drops it from DDP's expectation set so
        # ``find_unused_parameters=False`` (plain ``'ddp'``) stays legal.
        self.frozen_attn_proj = bool(
            freeze_unused_attn_proj and head_structured_latent
        )
        if self.frozen_attn_proj:
            for param in self.lag_attn.W_o.parameters():
                param.requires_grad_(False)

        self._install_v3_heads(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=(float(logvar_clamp[0]), float(logvar_clamp[1])),
            dropout=dropout,
            num_heads=num_heads,
            d_head=d_head,
            decoder_hidden=decoder_hidden,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _install_v3_heads(
        self,
        *,
        d_model: int,
        d_z: int,
        logvar_clamp: Tuple[float, float],
        dropout: float,
        num_heads: int,
        d_head: int,
        decoder_hidden: int,
    ) -> None:
        """Replace the four v1 log-variance heads with v3 variants in place.

        The parent :meth:`__init__` has already built *and initialised* the full v1 model
        (generic init + zero-init delta heads). Each v3 head is a structural subclass, so we
        construct it and **transfer the already-initialised weights** via ``load_state_dict``.
        This makes v3's parameters equal v1's byte-for-byte in the parity configuration
        (independent of the RNG seed), which is what the golden-parity test relies on. In
        ``residual`` mode the newly-added ``delta_logvar_head`` has no v1 counterpart and is
        left to be zeroed by :meth:`_zero_init_delta_heads`.
        """
        lb = self.logvar_bound

        old_prior = self.prior_head
        prior = PriorHeadV3(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
            logvar_bound=lb,
        )
        prior.load_state_dict(old_prior.state_dict(), strict=True)
        self.prior_head = prior

        old_base = self.baseline_decoder
        baseline = BaselineFutureDecoderV3(
            core=self.horizon_core,
            d_model=d_model,
            out_channels=self.c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
            logvar_bound=lb,
        )
        baseline.load_state_dict(old_base.state_dict(), strict=True)
        self.baseline_decoder = baseline

        old_res = self.residual_decoder
        residual = ResidualFutureDecoderV3(
            core=self.horizon_core,
            d_model=d_model,
            d_z=d_z,
            out_channels=self.c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
            logvar_bound=lb,
        )
        residual.load_state_dict(old_res.state_dict(), strict=True)
        self.residual_decoder = residual

        old_post = self.posterior_head
        posterior = PosteriorHeadV3(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            delta_mu_scale=self.delta_mu_scale,
            head_structured=self.head_structured_latent,
            num_heads=num_heads,
            d_head=d_head,
            logvar_bound=lb,
            posterior_logvar=self.posterior_logvar,
            delta_logvar_scale=self.delta_logvar_scale,
        )
        # Independent (parity) mode is a structural match with v1, so require an exact load
        # (strict=True) -- this keeps the parity/warm-start guard that would catch any future
        # key/shape drift. Residual mode legitimately differs (the independent logvar head is
        # gone, delta_logvar_head is new), so it must relax to strict=False.
        posterior.load_state_dict(
            old_post.state_dict(), strict=(self.posterior_logvar == "independent")
        )
        self.posterior_head = posterior

        # Re-assert the zero-init on all delta heads: idempotent for the mean deltas (already
        # zero in v1), and zeroes the freshly-built delta_logvar_head in residual mode.
        self._zero_init_delta_heads()

    def _zero_init_delta_heads(self) -> None:
        """Zero the delta-mean heads (v1) and, in residual mode, the delta-logvar head (G1).

        Extends :meth:`SeqVaeLagAttnV1._zero_init_delta_heads`. During the parent
        ``__init__`` this runs while ``posterior_head`` is still the v1 head (no
        ``delta_logvar_head``); it then runs again after :meth:`_install_v3_heads` with the
        v3 posterior in place.

        Warning:
            This also zeroes ``delta_mu_head`` and ``residual_decoder.mean_head``. It is an
            *initialisation* routine and must never be called after loading trained weights --
            use :meth:`zero_init_delta_logvar_head` for that.
        """
        super()._zero_init_delta_heads()
        self.zero_init_delta_logvar_head()

    def zero_init_delta_logvar_head(self) -> None:
        r"""Zero **only** the posterior's delta-logvar head, leaving every other head intact.

        Safe to call after a warm-start: it re-asserts :math:`\Delta\ell_t \equiv 0` (so
        :math:`\log\sigma^{2,q} = \log\sigma^{2,p}` exactly) without touching the trained
        ``delta_mu_head`` or ``residual_decoder.mean_head`` that the checkpoint supplied.
        """
        ph = getattr(self, "posterior_head", None)
        if ph is None or getattr(ph, "posterior_logvar", None) != "residual":
            return
        delta_logvar_head = getattr(ph, "delta_logvar_head", None)
        if delta_logvar_head is None:
            return
        if isinstance(delta_logvar_head, nn.ModuleList):
            for layer in delta_logvar_head:
                self._zero_linear(layer)
        else:
            self._zero_linear(delta_logvar_head)

    # ------------------------------------------------------------------
    # Interventional lag-band masking (G-F)
    # ------------------------------------------------------------------
    def _combined_lag_mask(
        self,
        seq_len: int,
        device: torch.device,
        lag_band_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Intersect an ablation band mask with the causal lag-validity mask.

        :class:`LagCrossAttention` applies **exactly one** mask: whatever reaches
        ``forward(h_y, h_u, m_lag)`` replaces the internally-built validity mask rather than
        intersecting with it. Passing a bare band mask straight through would therefore
        silently destroy the causal constraint :math:`t - \ell \ge 0`. A ``(L,)`` mask would
        additionally raise inside ``_attend`` (which needs ``(T, L)``), and a ``(B, T, L)``
        mask would be collapsed to sample ``0``'s row -- so per-sample masks are *not*
        expressible through this API.

        This helper closes all three gaps: it broadcasts ``(L,) -> (T, L)``, logical-ANDs with
        :meth:`LagCrossAttention._build_lag_mask`, and returns the result in **lag order**
        (index :math:`0` = lag :math:`0` = the current step), which is the orientation
        ``_attend`` expects before it flips to window order itself.

        **Dead anchors.** When the masked band contains lag :math:`0`, every causally valid
        lag at anchors :math:`t < \min(\text{band})` is removed and the attention row becomes
        all :math:`-\infty`. Under ``softmax`` that row degrades gracefully
        (``nan_to_num`` -> :math:`\alpha = 0`), but ``entmax15`` **raises**: its support size
        is :math:`0` and it gathers at index :math:`-1`. The causal mask alone can never
        produce such a row -- lag :math:`0` is valid at every anchor -- so band masking is the
        first thing that can. Lag :math:`0` is therefore forced back on at those anchors
        purely to keep the activation well-posed, and the resulting rows are discarded by
        :meth:`_ablate_dead_anchors`, which reimposes the ``softmax`` semantics for **both**
        activations.

        Args:
            seq_len: Sequence length :math:`T`.
            device: Device on which to build the validity mask.
            lag_band_mask: Boolean keep-mask of shape ``(L,)`` or ``(T, L)`` in lag order,
                where ``True`` keeps the lag. ``None`` disables band masking entirely.

        Returns:
            ``(m_lag, dead)``. ``m_lag`` is the combined ``(T, L)`` boolean mask and ``dead``
            the ``(T,)`` boolean mask of anchors with no surviving valid lag. Both are
            ``None`` when ``lag_band_mask is None``, so the caller invokes attention exactly
            as it does without this feature -- which is what makes the default path bit-exact.

        Raises:
            ValueError: If ``lag_band_mask`` is not 1-D or 2-D, or its lag axis is not
                :math:`L = \texttt{max\_lag} + 1`, or its time axis is not :math:`T`.
        """
        if lag_band_mask is None:
            return None, None

        L = int(self.lag_attn.L)
        band = lag_band_mask.to(device=device, dtype=torch.bool)
        if band.dim() == 1:
            if band.shape[0] != L:
                raise ValueError(
                    f"lag_band_mask of shape {tuple(lag_band_mask.shape)} has lag axis "
                    f"{band.shape[0]}, expected L={L}"
                )
            band = band.unsqueeze(0).expand(int(seq_len), L)
        elif band.dim() == 2:
            if band.shape != (int(seq_len), L):
                raise ValueError(
                    f"lag_band_mask of shape {tuple(lag_band_mask.shape)} is not (T, L) = "
                    f"({int(seq_len)}, {L})"
                )
        else:
            raise ValueError(
                f"lag_band_mask must be 1-D (L,) or 2-D (T, L); got "
                f"{tuple(lag_band_mask.shape)}"
            )

        validity = self.lag_attn._build_lag_mask(int(seq_len), device=device)
        combined = validity & band
        dead = ~combined.any(dim=-1)                       # (T,)
        if bool(dead.any()):
            combined = combined.clone()
            combined[dead, 0] = True                       # always causally valid
        return combined, dead

    def _ablate_dead_anchors(
        self,
        A: torch.Tensor,
        alpha: torch.Tensor,
        A_heads: torch.Tensor,
        dead: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Zero the attention at anchors whose every valid lag was masked.

        Reimposes, for both ``softmax`` and ``entmax15``, exactly what ``softmax``'s
        all-:math:`-\infty` path produces: :math:`\alpha = 0`, per-head summary
        :math:`a^{(m)} = 0`, and the fused source :math:`A = W_o(0)` -- which is ``W_o``'s
        bias, *not* necessarily zero. Ablation, not renormalisation: the surviving lags are
        never rescaled to recover unit mass.

        Args:
            A: Fused attended source ``(B, T, d_model)``.
            alpha: Attention weights ``(B, T, num_heads, L)`` in lag order.
            A_heads: Per-head attended summaries ``(B, T, num_heads, d_head)``, pre-``W_o``.
            dead: ``(T,)`` boolean mask of anchors with no surviving valid lag, or ``None``.

        Returns:
            The ``(A, alpha, A_heads)`` triple with the dead anchors overwritten.
        """
        if dead is None or not bool(dead.any()):
            return A, alpha, A_heads

        alpha = alpha.clone()
        alpha[:, dead] = 0.0
        A_heads = A_heads.clone()
        A_heads[:, dead] = 0.0
        A = A.clone()
        bias = self.lag_attn.W_o.bias
        A[:, dead, :] = 0.0 if bias is None else bias.to(dtype=A.dtype)
        return A, alpha, A_heads

    # ------------------------------------------------------------------
    # Forward / sampling
    # ------------------------------------------------------------------
    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full pipeline. Additive to v1: ``kld_active_frac`` + ``raw_logvar_prior``.

        Identical to :meth:`SeqVaeLagAttnV1.forward` except that the prior head returns the
        pre-bound raw log-variance (threaded into the posterior for the residual path) and
        the returned dict gains two additive keys:

        * ``kld_active_frac`` -- scalar fraction of latent dims with :math:`\overline{K_j} >
          \epsilon` (G4).
        * ``raw_logvar_prior`` -- ``(B, T, d_z)`` pre-bound raw prior log-variance
          :math:`\widetilde{\log\sigma^{2,p}_t}`, required by :meth:`perm_kl_from_forward`
          to rebuild the posterior under a permuted source without a second encoder pass
          (G6).

        Every v1 key keeps its v1 shape and semantics, so the testing pipeline's forward
        contract is preserved.

        Args:
            y_st: Target scattering features ``(B, T, c_y_st)``.
            y_ph: Target phase features ``(B, T, c_y_ph)``.
            u_stream: Source stream ``(B, T, c_u)``.
            lag_band_mask: Optional boolean keep-mask over lags, ``(L,)`` or ``(T, L)`` in lag
                order, ``True`` = keep. Intersected with the causal validity mask by
                :meth:`_combined_lag_mask`. ``None`` (the default) is a **bit-exact no-op**.

        Returns:
            The 25-key forward dict.

        Note:
            Band masking is an **ablation, not a renormalisation**. The surviving lags are
            *not* re-scaled to recover unit attention mass: ``_attend`` fills the masked
            scores with :math:`-\infty` before the softmax/entmax, so an ``alpha`` row over a
            partially-masked set of valid lags still sums to :math:`1`, while a row whose
            *every* valid lag was masked collapses to :math:`\alpha = 0` (and per-head
            summary :math:`a^{(m)} = 0`, hence fused :math:`A = W_o(0)`, i.e. ``W_o``'s bias
            rather than necessarily zero). That collapse happens at anchors :math:`t < D`
            whenever the masked band contains lag :math:`0`; those anchors lie inside the
            warm-up prefix, below the clean evaluation window
            :math:`[\max(\text{warmup}, D-1),\, T-H)`, so they never enter a forecast loss.
            See :meth:`_ablate_dead_anchors` for why the two activations need help agreeing.
        """
        Y = torch.cat([y_st, y_ph], dim=-1)

        Y_tilde = self.target_adapter(Y)
        U_tilde = self.source_adapter(u_stream)

        H_y = self.target_encoder(Y_tilde)
        H_u = self.source_encoder(U_tilde)

        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(H_y)

        m_lag, dead = self._combined_lag_mask(H_y.size(1), H_y.device, lag_band_mask)
        A, alpha, A_heads = self.lag_attn(H_y, H_u, m_lag)
        A, alpha, A_heads = self._ablate_dead_anchors(A, alpha, A_heads, dead)

        post_src = A_heads if self.head_structured_latent else A
        mu_post, logvar_post = self.posterior_head(
            H_y, post_src, mu_prior, raw_logvar_prior
        )
        z = self.reparameterize(mu_post, logvar_post)

        with torch.no_grad():
            mu_prior_sat_frac = (
                mu_prior.abs() >= (0.99 * self.mu_scale)
            ).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (0.99 * self.delta_mu_scale)
            ).float().mean()

        if self.training:
            self._update_latent_running_stats(mu_post)

        mu_base, logvar_base = self.baseline_decoder(decoder_state)
        delta_mu_src, logvar_full = self.residual_decoder(decoder_state, z)
        mu_full = mu_base + delta_mu_src

        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
            mask_warmup=False,
        )
        kld_per_t, te_lag_map, kld_per_t_per_head = self.te_analysis(
            kld_btd, alpha, head_structured=self.head_structured_latent
        )

        warmup_mask = self._build_warmup_valid_mask(H_y.size(1), device=H_y.device)
        kld_active_frac = self._kld_active_frac(kld_btd)

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": H_y,
            "source_state": H_u,
            "decoder_state": decoder_state,
            "attended_source": A,
            "attended_source_heads": A_heads,
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
            "kld_active_frac": kld_active_frac,
        }

    def encode_only(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        sample_z: bool = True,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run encoders + posterior only (no decoders). Same 11-key contract as v1.

        Args:
            y_st: Target scattering features ``(B, T, c_y_st)``.
            y_ph: Target phase features ``(B, T, c_y_ph)``.
            u_stream: Source stream ``(B, T, c_u)``.
            sample_z: Reparameterise when True, else return the posterior mean as ``z``.
            lag_band_mask: Optional lag keep-mask, ``(L,)`` or ``(T, L)`` in lag order. See
                :meth:`forward` for the ablation semantics. ``None`` is a bit-exact no-op.

        Returns:
            The 11-key encode dict.
        """
        Y = torch.cat([y_st, y_ph], dim=-1)
        Y_tilde = self.target_adapter(Y)
        U_tilde = self.source_adapter(u_stream)
        H_y = self.target_encoder(Y_tilde)
        H_u = self.source_encoder(U_tilde)
        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(H_y)
        m_lag, dead = self._combined_lag_mask(H_y.size(1), H_y.device, lag_band_mask)
        A, alpha, A_heads = self.lag_attn(H_y, H_u, m_lag)
        A, alpha, A_heads = self._ablate_dead_anchors(A, alpha, A_heads, dead)
        post_src = A_heads if self.head_structured_latent else A
        mu_post, logvar_post = self.posterior_head(
            H_y, post_src, mu_prior, raw_logvar_prior
        )
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
            "attended_source_heads": A_heads,
            "attn_weights": alpha,
        }

    # ------------------------------------------------------------------
    # Source-permutation control (G6)
    # ------------------------------------------------------------------
    def _perm_posterior(
        self,
        h_y: torch.Tensor,
        h_u_perm: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        raw_logvar_prior: Optional[torch.Tensor],
        detach_prior: bool,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Re-run lag attention + posterior against a batch-permuted source state.

        Args:
            h_y: Target state :math:`H_y` ``(B, T, d_model)``.
            h_u_perm: Batch-permuted source state :math:`H_u[\pi]` ``(B, T, d_model)``.
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Bounded prior log-variance ``(B, T, d_z)``.
            raw_logvar_prior: Pre-bound raw prior log-variance ``(B, T, d_z)``; may be ``None``
                when ``posterior_logvar='independent'``.
            detach_prior: If True, detach the prior statistics before they enter either the
                posterior's residual base or the KL, so the control's gradient flows *only*
                through source-encoder :math:`\to` attention :math:`\to` posterior deltas.
                Without this, :math:`L_{\mathrm{perm}}` could be minimised by dragging the
                prior toward :math:`q` rather than by collapsing the source-driven deltas.
            lag_band_mask: Optional lag keep-mask, ``(L,)`` or ``(T, L)`` in lag order. See
                :meth:`forward` for the ablation semantics. ``None`` is a bit-exact no-op, and
                is what every permutation-control caller passes.

        Returns:
            ``(mu_prior, logvar_prior, mu_post_perm, logvar_post_perm)``, where the first two
            are the (possibly detached) prior statistics the caller must use as the KL
            reference.
        """
        if detach_prior:
            mu_prior = mu_prior.detach()
            logvar_prior = logvar_prior.detach()
            if raw_logvar_prior is not None:
                raw_logvar_prior = raw_logvar_prior.detach()

        m_lag, dead = self._combined_lag_mask(h_y.size(1), h_y.device, lag_band_mask)
        a_perm, alpha_perm, a_heads_perm = self.lag_attn(h_y, h_u_perm, m_lag)
        a_perm, _, a_heads_perm = self._ablate_dead_anchors(
            a_perm, alpha_perm, a_heads_perm, dead
        )
        post_src = a_heads_perm if self.head_structured_latent else a_perm
        mu_post_perm, logvar_post_perm = self.posterior_head(
            h_y, post_src, mu_prior, raw_logvar_prior
        )
        return mu_prior, logvar_prior, mu_post_perm, logvar_post_perm

    def _resolve_perm_index(
        self,
        batch_size: int,
        perm_index: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
        device: torch.device,
    ) -> torch.Tensor:
        """Validate a supplied ``perm_index`` or draw a fresh derangement of ``batch_size``."""
        if perm_index is None:
            return make_derangement(batch_size, generator=generator, device=device)
        perm_index = perm_index.to(device=device, dtype=torch.long)
        if perm_index.shape != (batch_size,):
            raise ValueError(
                f"perm_index must have shape ({batch_size},), got {tuple(perm_index.shape)}"
            )
        return perm_index

    def _perm_kl_result(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post_perm: torch.Tensor,
        logvar_post_perm: torch.Tensor,
        perm_index: torch.Tensor,
        weight: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Reduce the permuted posterior to the anchor-masked control loss + its readouts.

        Returns:
            ``perm_kl`` -- the differentiable anchor-masked mean (the auxiliary loss);
            ``kld_shuffled`` -- its detached scalar readout;
            ``kld_shuffled_per_t`` -- ``(B, T)`` raw per-step KL under :math:`\\pi`, matching
            ``kld_per_t``'s semantics (raw, full-:math:`T`, summed over latent dims) so the two
            curves can be plotted against each other;
            ``perm_index`` -- the derangement used.
        """
        perm_kl = self._kld_loss(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post_perm,
            logvar_post=logvar_post_perm,
            reduce_mean=True,
            weight=weight,
            free_bits=0.0,
        )
        with torch.no_grad():
            kld_shuffled_per_t = self.kld_tensor(
                mu_prior=mu_prior,
                logvar_prior=logvar_prior,
                mu_post=mu_post_perm,
                logvar_post=logvar_post_perm,
                mask_warmup=False,
            ).sum(dim=-1)
        return {
            "perm_kl": perm_kl,
            "kld_shuffled": perm_kl.detach(),
            "kld_shuffled_per_t": kld_shuffled_per_t,
            "perm_index": perm_index,
        }

    def permutation_kl(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        *,
        weight: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        perm_index: Optional[torch.Tensor] = None,
        detach_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Source-permutation control :math:`L_{\mathrm{perm}}` (G6), re-encoding the source.

        Deranges the batch on the source stream, re-encodes :math:`\pi(U)` through the source
        adapter and encoder, and returns the anchor-masked mean

        $$L_{\mathrm{perm}} = \mathrm{KL}\!\left(q\big(z \mid Y, \pi(U)\big) \,\big\|\,
        p\big(z \mid Y\big)\right)$$

        over the *same* time support as :meth:`_kld_loss` (hence the same support as
        :meth:`measure_transfer_entropy`), so :math:`K_{\mathrm{true}}` and
        :math:`K_{\mathrm{shuffled}}` are directly comparable. The free-bit floor is
        deliberately **not** applied: a positive floor would zero the gradient in exactly the
        low-KL regime the control exists to drive the model into.

        This is the eval-time / diagnostic entry point. Training uses the equivalent but
        cheaper :meth:`perm_kl_from_forward`, which reuses the source state that
        :meth:`forward` already computed.

        Args:
            y_st: FHR scattering features ``(B, T, 43)``.
            y_ph: FHR phase features ``(B, T, 44)``.
            u_stream: Source stream ``(B, T, c_u)``.
            weight: Optional per-sample dataset weight broadcastable to ``(B, T)``.
            generator: Optional CPU generator seeding the derangement.
            perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
            detach_prior: See :meth:`_perm_posterior`.

        Returns:
            ``{"perm_kl", "kld_shuffled", "perm_index"}``. ``perm_kl`` carries gradient through
            the source/posterior path; ``kld_shuffled`` is its detached readout.

        Raises:
            ValueError: If the batch is too small to derange (``B < 2``).
        """
        perm_index = self._resolve_perm_index(
            y_st.size(0), perm_index, generator, y_st.device
        )

        Y = torch.cat([y_st, y_ph], dim=-1)
        h_y = self.target_encoder(self.target_adapter(Y))
        h_u_perm = self.source_encoder(self.source_adapter(u_stream[perm_index]))
        mu_prior, logvar_prior, _, raw_logvar_prior = self.prior_head(h_y)

        mu_p, logvar_p, mu_q, logvar_q = self._perm_posterior(
            h_y, h_u_perm, mu_prior, logvar_prior, raw_logvar_prior, detach_prior
        )
        return self._perm_kl_result(mu_p, logvar_p, mu_q, logvar_q, perm_index, weight)

    def perm_kl_from_forward(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        *,
        weight: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        perm_index: Optional[torch.Tensor] = None,
        detach_prior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Fused :math:`L_{\mathrm{perm}}` reusing the states of a completed forward (G6).

        The source path (``source_adapter`` :math:`\to` ``source_encoder``) contains no
        batch-coupled operator -- only causal convolutions, an LSTM, and LayerNorm -- so it
        acts independently on each batch element and

        $$\mathrm{SourceEncoder}\big(\mathrm{SourceAdapter}(\pi(U))\big)_i \;=\; H_u[\pi(i)].$$

        Permuting the already-computed ``source_state`` along the batch axis is therefore
        *exactly* equivalent to re-encoding a permuted source stream, and only the lag attention
        and the posterior head must be re-run. This keeps the whole control inside the single
        main forward/backward: under DDP a parameter used twice in one graph shares one
        ``AccumulateGrad`` node and is marked ready exactly once, so ``automatic_optimization``
        -- and with it Lightning's gradient clipping, gradient accumulation, LR scheduling, and
        the loss-spike circuit breaker -- is preserved.

        Equivalence to :meth:`permutation_kl` is asserted by ``tests/test_v3_perm_kl.py``.

        Args:
            forward_outputs: The dict returned by :meth:`forward`. Requires ``target_state``,
                ``source_state``, ``mu_prior``, ``logvar_prior`` and (in residual mode)
                ``raw_logvar_prior``.
            weight: Optional per-sample dataset weight broadcastable to ``(B, T)``.
            generator: Optional CPU generator seeding the derangement.
            perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
            detach_prior: See :meth:`_perm_posterior`.

        Returns:
            ``{"perm_kl", "kld_shuffled", "perm_index"}``, as in :meth:`permutation_kl`.

        Raises:
            ValueError: If the batch is too small to derange (``B < 2``).
        """
        h_y = forward_outputs["target_state"]
        h_u = forward_outputs["source_state"]
        perm_index = self._resolve_perm_index(
            h_u.size(0), perm_index, generator, h_u.device
        )

        mu_p, logvar_p, mu_q, logvar_q = self._perm_posterior(
            h_y,
            h_u[perm_index],
            forward_outputs["mu_prior"],
            forward_outputs["logvar_prior"],
            forward_outputs.get("raw_logvar_prior"),
            detach_prior,
        )
        return self._perm_kl_result(mu_p, logvar_p, mu_q, logvar_q, perm_index, weight)

    @torch.no_grad()
    def perm_forward_outputs(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        *,
        perm_index: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Rebuild the forecast under a batch-deranged source, for the prediction-space control.

        The KL-space control :meth:`permutation_kl` answers "did the source move my belief?",
        which a *mismatched* source does too -- often more strongly, since the posterior only
        ever trained on matched pairs. Empirically :math:`K_{\mathrm{shuffled}} \gtrsim
        K_{\mathrm{true}}` even when the source is unmistakably being used, so the raw KL alone
        does **not** establish source specificity.

        The forecast does. Feeding the decoder a latent drawn from
        :math:`q(z \mid Y, \pi(U))` and re-scoring against the true future gives
        ``feat_loss_shuffled``. On a model that genuinely exploits the source this lands
        *above* the target-only ``base_loss`` -- a wrong source is worse than no source -- while
        ``feat_loss`` lands well below it. That ordering is the negative control that
        discriminates, and it needs no auxiliary loss.

        Only the source-dependent tensors are recomputed; the encoders, the prior, and the
        target-only baseline decoder are reused untouched.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            perm_index: Optional precomputed derangement ``(B,)``; drawn if omitted.
            generator: Optional CPU generator seeding the derangement.

        Returns:
            A shallow copy of ``forward_outputs`` with ``mu_post``, ``logvar_post``, ``z``,
            ``delta_mu_src``, ``mu_full`` and ``logvar_full`` recomputed under :math:`\pi(U)`,
            plus the ``perm_index`` used. Suitable to hand straight to :meth:`compute_loss`
            with ``compute_kld_loss=False``.

        Raises:
            ValueError: If the batch is too small to derange (``B < 2``).
        """
        h_y = forward_outputs["target_state"]
        h_u = forward_outputs["source_state"]
        perm_index = self._resolve_perm_index(
            h_u.size(0), perm_index, generator, h_u.device
        )

        _, _, mu_post_perm, logvar_post_perm = self._perm_posterior(
            h_y,
            h_u[perm_index],
            forward_outputs["mu_prior"],
            forward_outputs["logvar_prior"],
            forward_outputs.get("raw_logvar_prior"),
            detach_prior=True,
        )
        z_perm = self.reparameterize(mu_post_perm, logvar_post_perm)
        decoder_state = forward_outputs["decoder_state"]
        delta_mu_perm, logvar_full_perm = self.residual_decoder(decoder_state, z_perm)

        permuted = dict(forward_outputs)
        permuted.update(
            mu_post=mu_post_perm,
            logvar_post=logvar_post_perm,
            z=z_perm,
            delta_mu_src=delta_mu_perm,
            mu_full=forward_outputs["mu_base"] + delta_mu_perm,
            logvar_full=logvar_full_perm,
            perm_index=perm_index,
        )
        return permuted

    # ------------------------------------------------------------------
    # KL support + reporting (G3, G4)
    # ------------------------------------------------------------------
    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        """Estimate the TE surrogate :math:`\\mathrm{KL}(q\\,\\|\\,p)` over the KL support.

        Overrides :meth:`SeqVaeLagAttnV1.measure_transfer_entropy` so **both** return modes
        share the same time support. The scalar (``reduce_mean=True``) already routes through
        the overridden :meth:`_kld_loss` (which honours ``self.kld_support``); the per-step
        tensor (``reduce_mean=False``) is masked to the same support -- out-of-support steps
        are set to ``NaN`` -- so for an ``'anchor'`` model the plotted per-step curve and the
        reported scalar are consistent (the scalar is the ``nanmean`` of the curve summed over
        latent dims), and the untrained final-``H`` tail is not shown as a spurious KL spike.
        In ``'full'`` mode this reproduces v1 exactly (only the warm-up prefix is masked).
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
            kld = self.kld_tensor(
                mu_prior=enc["mu_prior"],
                logvar_prior=enc["logvar_prior"],
                mu_post=enc["mu_post"],
                logvar_post=enc["logvar_post"],
                mask_warmup=False,
            )
            support = self._kld_support_mask(kld.size(1), device=kld.device) > 0  # (T,)
            kld = kld.clone()
            kld[:, ~support, :] = float("nan")
            return kld

    def _kld_support_mask(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        r"""Build the ``(T,)`` training-KL time-support mask.

        ``kld_support='full'`` masks only the warm-up prefix (``[warmup, T)``, v1 behaviour).
        ``kld_support='anchor'`` additionally masks the final :math:`H` steps
        (``[warmup, T-H)``), which are the anchors with no fully-observed forecast window and
        therefore carry no supervised gradient in :meth:`compute_loss`.

        Args:
            seq_len: Sequence length ``T``.
            device: Target device for the mask.
            dtype: Target floating dtype for the mask.

        Returns:
            A ``(T,)`` tensor of 1.0 (in support) / 0.0 (excluded).
        """
        mask = torch.ones(seq_len, device=device, dtype=dtype)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            mask[:warmup] = 0.0
        if self.kld_support == "anchor":
            horizon = int(self.horizon)
            if horizon > 0:
                upper = max(seq_len - horizon, 0)
                mask[upper:] = 0.0
        return mask

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
        """Aggregate the closed-form KL over the configured support (G3).

        Identical to :meth:`SeqVaeLagAttnV1._kld_loss` except the time mask comes from
        :meth:`_kld_support_mask` (which honours ``self.kld_support``). The call signature is
        unchanged, so the inherited :meth:`compute_loss` uses this override transparently.
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
        device = kld.device
        dtype = kld.dtype

        time_mask = self._kld_support_mask(T, device=device, dtype=dtype)
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

    def _kld_active_frac(self, kld_btd: torch.Tensor) -> torch.Tensor:
        r"""Fraction of latent dims whose mean per-step KL exceeds ``_KLD_ACTIVE_EPS`` (G4).

        Args:
            kld_btd: Per-step per-dim raw KL ``(B, T, d_z)``.

        Returns:
            A scalar tensor in ``[0, 1]``: the fraction of the ``d_z`` latent dimensions with
            :math:`\overline{K_j} > \epsilon`, averaged over batch and the **configured KL
            support** (so an ``'anchor'`` model excludes the untrained final-``H`` tail, whose
            KL carries no supervised gradient; in ``'full'`` mode this is the warm-up-only
            support, unchanged from the diagnostic's original meaning).
        """
        with torch.no_grad():
            support = self._kld_support_mask(kld_btd.size(1), device=kld_btd.device) > 0
            if not bool(support.any()):
                return torch.zeros((), device=kld_btd.device, dtype=kld_btd.dtype)
            kld_dim_mean = kld_btd[:, support, :].mean(dim=(0, 1))  # (d_z,)
            return (kld_dim_mean > _KLD_ACTIVE_EPS).to(kld_btd.dtype).mean()

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
        """Compute the loss with additive raw/train/active KL reporting (G4).

        Delegates to :meth:`SeqVaeLagAttnV1.compute_loss` (whose ``kld_loss`` is the
        anchor-masked, free-bit-floored optimised term that feeds ``total_loss``), then adds:

        * ``kld_train`` -- alias of ``kld_loss`` (the optimised term).
        * ``kld_raw`` -- the un-floored KL over the same support (reporting only; detached, so
          ``total_loss`` gradients flow *only* through ``kld_train``).
        * ``kld_active_frac`` -- surfaced from ``forward_outputs`` (G4 diagnostic).

        Because free-bits clamps every per-dim KL up to ``free_bits`` before masking,
        ``kld_train >= kld_raw`` always holds.
        """
        out = super().compute_loss(
            forward_outputs,
            y_st,
            y_ph,
            weight=weight,
            compute_kld_loss=compute_kld_loss,
            beta=beta,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            sigma_obs=sigma_obs,
            free_bits=free_bits,
            detach_baseline_in_full=detach_baseline_in_full,
            lambda_lag=lambda_lag,
        )

        if compute_kld_loss:
            with torch.no_grad():
                kld_raw = self._kld_loss(
                    mu_prior=forward_outputs["mu_prior"],
                    logvar_prior=forward_outputs["logvar_prior"],
                    mu_post=forward_outputs["mu_post"],
                    logvar_post=forward_outputs["logvar_post"],
                    reduce_mean=True,
                    weight=weight,
                    free_bits=0.0,
                )
        else:
            kld_raw = torch.zeros_like(out["kld_loss"])

        out["kld_raw"] = kld_raw
        out["kld_train"] = out["kld_loss"]
        out["kld_active_frac"] = forward_outputs.get(
            "kld_active_frac", torch.zeros_like(out["kld_loss"])
        )
        return out


# =============================================================================
# Smoke test (run with: python -m model.vae_teb_prediction.model.vae_teb_lag_attn_v3)
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 300

    # v1's 23 forward keys plus the two additive v3 keys.
    expected_keys = {
        "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z",
        "target_state", "source_state", "decoder_state",
        "attended_source", "attended_source_heads", "attn_weights",
        "mu_base", "logvar_base", "delta_mu_src", "mu_full", "logvar_full",
        "raw_future_pred", "kld_per_t", "kld_per_t_per_head", "te_lag_map",
        "warmup_mask", "mu_prior_sat_frac", "delta_mu_sat_frac",
        "kld_active_frac", "raw_logvar_prior",
    }

    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u_full = torch.randn(B, T, 101)

    # ---- Test 1: parity config (independent + clamp + full) ---------------
    model = SeqVaeLagAttnV3(
        use_up_st=True,
        posterior_logvar="independent",
        logvar_bound="clamp",
        kld_support="full",
    )
    outs = model(y_st, y_ph, u_full)

    missing = expected_keys - set(outs.keys())
    assert not missing, f"missing forward keys: {missing}"
    assert outs["mu_prior"].shape == (B, T, 24)
    assert outs["mu_full"].shape == (B, T, 30, 87)
    assert outs["te_lag_map"].shape == (B, T, 91)
    assert outs["warmup_mask"].shape == (T,)
    assert outs["kld_active_frac"].shape == ()
    init_delta = outs["delta_mu_src"].abs().max().item()
    assert init_delta < 1e-6, f"delta_mu_src not zero at init: {init_delta}"
    losses = model.compute_loss(outs, y_st, y_ph, beta=0.01, free_bits=0.1)
    for k in ("feat_loss", "base_loss", "kld_loss", "kld_raw", "kld_train",
              "kld_active_frac", "total_loss"):
        assert torch.isfinite(losses[k]), f"non-finite loss: {k}={losses[k]}"
    assert (losses["kld_train"] >= losses["kld_raw"] - 1e-6), "kld_train < kld_raw"
    losses["total_loss"].backward()
    print(
        f"[parity] kld={losses['kld_loss'].item():.4e}"
        f"  kld_raw={losses['kld_raw'].item():.4e}"
        f"  kld_train={losses['kld_train'].item():.4e}"
        f"  active_frac={losses['kld_active_frac'].item():.3f}"
    )
    print("[parity] independent+clamp+full forward/loss/backward OK")

    # ---- Test 2: production config (residual + smooth + anchor) -----------
    torch.manual_seed(0)
    model_prod = SeqVaeLagAttnV3(
        use_up_st=True,
        posterior_logvar="residual",
        logvar_bound="smooth",
        kld_support="anchor",
    ).eval()
    outs_prod = model_prod(y_st, y_ph, u_full)
    kld_max = outs_prod["kld_per_t"].abs().max().item()
    print(f"[prod] max |kld_per_t| at init = {kld_max:.3e} (expected ~0)")
    assert kld_max < 1e-6, f"KL is not zero at init under residual+smooth: {kld_max}"
    L = model_prod.compute_loss(
        outs_prod, y_st, y_ph, beta=0.1,
        likelihood="gaussian_nll", sigma_obs="learned", free_bits=0.1,
        detach_baseline_in_full=True,
    )
    for k in ("feat_loss", "base_loss", "kld_raw", "kld_train", "total_loss"):
        assert torch.isfinite(L[k]), f"non-finite {k}"
    assert L["kld_train"] >= L["kld_raw"] - 1e-6
    L["total_loss"].backward()
    print("[prod] residual+smooth+anchor zero-KL-init + learned-var loss/backward OK")

    # ---- Test 3: head-structured production zero-KL-init ------------------
    torch.manual_seed(0)
    model_hs = SeqVaeLagAttnV3(
        use_up_st=True,
        head_structured_latent=True,
        posterior_logvar="residual",
        logvar_bound="smooth",
        kld_support="anchor",
    ).eval()
    outs_hs = model_hs(y_st, y_ph, u_full)
    kld_max_hs = outs_hs["kld_per_t"].abs().max().item()
    print(f"[hs] max |kld_per_t| at init = {kld_max_hs:.3e} (expected ~0)")
    assert kld_max_hs < 1e-6, f"head-structured KL not zero at init: {kld_max_hs}"
    assert outs_hs["kld_per_t_per_head"].shape == (B, T, 4)
    model_hs.compute_loss(outs_hs, y_st, y_ph, beta=0.1)["total_loss"].backward()
    print("[hs] head-structured residual zero-KL-init OK")

    # ---- Test 4: ALiBi lag decay + source-permutation control (G5, G6) ----
    torch.manual_seed(0)
    model_g6 = SeqVaeLagAttnV3(
        use_up_st=True,
        posterior_logvar="residual",
        logvar_bound="smooth",
        kld_support="anchor",
        lag_bias_init="alibi_decay",
        lambda_perm=0.1,
    ).eval()
    assert isinstance(model_g6.lag_attn.lag_score_bias, nn.Parameter)
    assert model_g6.lag_attn.lag_score_bias.shape == (4, 91)

    outs_g6 = model_g6(y_st, y_ph, u_full)
    assert outs_g6["raw_logvar_prior"].shape == (B, T, 24)

    g = torch.Generator().manual_seed(7)
    perm = make_derangement(B, generator=g)
    assert bool((perm != torch.arange(B)).all()), "derangement has a fixed point"

    fused = model_g6.perm_kl_from_forward(outs_g6, perm_index=perm)
    exact = model_g6.permutation_kl(y_st, y_ph, u_full, perm_index=perm)
    gap = (fused["perm_kl"] - exact["perm_kl"]).abs().item()
    print(f"[g6] |fused - re-encoded| perm_kl = {gap:.3e} (expected ~0)")
    assert gap < 1e-6, f"fused perm_kl disagrees with the re-encoded one: {gap}"
    # At init the posterior deltas are zero, so a shuffled source still yields KL == 0.
    assert fused["kld_shuffled"].item() < 1e-6
    fused["perm_kl"].backward()
    print("[g6] alibi_decay + derangement + perm_kl equivalence/backward OK")

    print("All smoke checks passed.")
