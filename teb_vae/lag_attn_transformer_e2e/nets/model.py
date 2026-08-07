r"""The end-to-end causal conv-Transformer lag-attention sequential VAE over raw signals.

The question is the sibling's: *how much does the source's recent past tell us about the target's
near raw future, that the target's own past did not already say -- and at what delay?* What differs
is the one thing that decides whether the answer is a forecast at all: **what the encoders read**.

The machinery, per $4$-second anchor $t$:

1. Two :class:`~teb_vae.lag_attn_transformer_e2e.nets.frontend.CausalRawFrontend` instances map the
   raw $4$ Hz target and source signals onto the token grid. Token $t$ is a function of raw samples
   at index $\le 16t + 15$ -- exactly ``TrimmedRawGeometry.n_raw(t)`` -- and of nothing later.
2. Two causal conv-Transformer encoders build history states $H^y_t$ and $H^u_t$ from those tokens.
   The target reads its full causal prefix; the source reads a bounded causal window, so a source
   state summarises a *local* neighbourhood and the lag attention keeps its ability to tell adjacent
   delays apart.
3. A full-latent target-only **prior** $p_\theta(z_t \mid Y_{\le t}) =
   \mathcal{N}(\mu^p_t, \operatorname{diag} e^{\ell^p_t})$ is read off $H^y$.
4. Lag cross-attention, queried by a projection of $\mu^p_t$, looks back over
   $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and returns per-head attended summaries.
5. A head-structured **posterior** $q_\phi(z_t \mid Y_{\le t}, U_{\le t})$ is a bounded
   zero-initialised residual on the prior, in the same coordinate system.
6. One noise draw $\epsilon_t$ serves both distributions (common random numbers), and **one shared
   decoder**, invoked twice on $z^p_t$ and $z^q_t$ and receiving nothing else, emits the base and
   full forecasts of the next $H \cdot R = 480$ raw target samples.

Everything from step 2 down is imported unchanged from the packages that own it, including both
encoders and the objective. That is the experiment: *same everything, different input*, so a
difference in results is attributable to the input representation and to nothing else. A retyped
encoder would let an encoder difference masquerade as a result about the input; a second copy of the
objective would make the comparison partly a comparison of two losses.

Five structural properties carry the design, each enforced here rather than by convention:

* **Raw-signal causality.** $H_t = f(x_{\le 16t + 15})$, at raw-sample resolution rather than at
  token resolution. This is the property the package exists for, and it is what the sibling cannot
  have: its inputs are two-sided wavelet and phase-harmonic coefficients, so a token-causal model
  over them still conditions on part of the interval it is asked to forecast. Here every operation
  on the history path is either position-wise on the channel axis or left-padded in time.
* **No decoder bypass.** The decoder's forward accepts exactly one tensor, $z$. There is no
  ``decoder_state``; gradient reaches the decoder only through the latent, so $\mu^p_t$ must carry
  the target's predictive state and $\mu^q_t - \mu^p_t$ is the additional source-derived
  predictive information.
* **Source purity.** The source pathway never sees a target tensor, and the prior never sees the
  source, so $\mathrm{KL}(q_t \Vert p_t)$ measures what the source added. Two independently
  parameterised front ends at identical settings, never one shared instance.
* **Exact zero at init.** The posterior deltas are zero-initialised *after* the generic weight
  initialisation, one $\epsilon$ serves both samples, and the twice-invoked decoder and both sets of
  attention probabilities carry no dropout -- so at initialisation the KL is exactly $0$ and the
  base and full forecasts are bitwise identical, in train mode. Front-end and encoder dropout do not
  disturb that: prior and posterior are built from one common target forward, so both branches see
  the same dropped activations.
* **The warm-up covers the front end's reach.** The front ends are constructed against a budget of
  $\texttt{warmup\_period} \times \texttt{raw\_per\_step}$ raw samples and refuse a stack that
  reaches further, so no *trained* anchor reads the zero-padded convolution transient at the
  segment's start.

Shapes, with $B$ batch, $T$ steps, $L = \mathrm{max\_lag}+1$ lags, $H$ horizon tokens, $R$ raw
samples per token: inputs are two ``(B, T*R)`` raw signals and a ``(B, T)`` validity weight; the
latent is ``(B, T, d_z)``; forecasts are ``(B, T - H, H, R)`` -- decoded over the valid anchor range
only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.blocks import initialization, validate_choice
from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
from teb_vae.lag_attn.nets.heads import POSTERIOR_LOGVAR_MODES, PosteriorHead, TEAnalysisHead
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
from teb_vae.lag_attn_rws.nets.losses import kld_tensor as closed_form_kld
from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index
from teb_vae.lag_attn_transformer_e2e.nets.frontend import (
    FRONTEND_KERNELS,
    CausalRawFrontend,
)
from teb_vae.lag_attn_transformer_rws.nets.blocks import init_depthwise_
from teb_vae.lag_attn_transformer_rws.nets.encoders import CausalConvTransformerEncoder


#: How the base branch obtains its latent. ``'sample'`` draws $z^p$ from the prior under the
#: shared $\epsilon$; ``'mean'`` decodes at $\mu^p$. A choice rather than a bool so the config
#: says which of two named behaviours it wants, and so a third mode has somewhere to go.
BASE_DECODE_CHOICES = ("sample", "mean")


class SeqVaeLagAttnTrfE2E(nn.Module):
    r"""Single-latent lag-attentive VAE reading the raw signals through causal front ends.

    See the module docstring for what the pieces are for. The constructor's job is to build them
    consistently and to refuse inconsistent geometry loudly, before any of it reaches a forward.

    A standalone module rather than a mode of the model it is compared against, for the same reason
    that one is standalone rather than a subclass of *its* sibling: the constructor schemas genuinely
    differ. There are no stored feature blocks here, so ``c_y``, ``c_u``, ``use_up_st`` and the four
    causal-reach channel tuples describe nothing -- and a front-end kernel schedule appears, which
    that constructor has never heard of. Absorbing the difference behind a flag would leave half the
    keyword surface dead on every run. What the two share is their *objective* and everything under
    it, imported from the modules that own them, rather than their construction.
    """

    #: The cached raw-target index grid, declared so it types as a tensor rather than as the
    #: ``Tensor | Module`` a bare submodule attribute would: it is handed to the shared objective as
    #: a tensor argument.
    future_index: torch.Tensor

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 48,
        horizon: int = 30,
        raw_per_step: int = 16,
        warmup_period: int = 30,
        max_lag: int = 90,
        num_heads: int = 4,
        d_head: int = 32,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = False,
        horizon_embed_std: float = 0.02,
        head_init_calibration: bool = False,
        a_head_gain: float = 1.0,
        frontend_kernels: Sequence[int] = FRONTEND_KERNELS,
        frontend_reach_budget_s: Optional[float] = None,
        encoder_conv_kernels: Sequence[int] = (5, 9),
        encoder_conv_dilations: Sequence[int] = (1, 2),
        encoder_num_heads: int = 4,
        encoder_d_ff: int = 256,
        target_attention_blocks: int = 4,
        source_attention_blocks: int = 3,
        source_attention_window: Optional[int] = 16,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        delta_mu_scale: float = 3.0,
        delta_logvar_scale: float = 2.0,
        posterior_logvar_mode: str = "residual",
        source_dropout: Optional[float] = None,
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        query_uses_logvar: bool = False,
        coverage_floor: float = 0.9,
        base_decode: str = "sample",
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Args:
            sequence_length: Decimated sequence length $T$. Together with ``raw_per_step`` it fixes
                the trimmed-grid geometry the raw target lives on **and** the raw length the two
                inputs must arrive at, and it sizes the rotary tables and the window masks -- all
                fixed at construction, so a longer batch raises rather than silently reallocating.
            d_model: Internal width used throughout the backbone, and the width each front end
                emits. The front-end stage widths are derived from it as
                $(d/4,\, d/2,\, 3d/4,\, d)$, so it must be divisible by $4$.
            d_z: Latent dimensionality. Must be divisible by ``num_heads``: the posterior is
                head-structured, so latent group $m$ is written only by attention head $m$ and the
                per-head KL is a genuine additive decomposition.
            horizon: Forecast horizon $H$ in decimated steps.
            raw_per_step: Raw samples per decimated step $R$; each horizon token emits this many raw
                samples, and it is also the front ends' total decimation, which they check against
                their own stride schedule.
            warmup_period: Initial steps excluded from every loss. It does double duty here: it is
                also the front ends' **reach budget** in steps, because an anchor inside the warm-up
                is the only one allowed to see the zero-padded convolution transient at the
                segment's start.
            max_lag: Maximum past lag; the attention window is $L = \mathrm{max\_lag}+1$ wide.
            num_heads: Number of **lag-attention** heads and latent groups. Unrelated to
                ``encoder_num_heads``; nothing may couple them.
            d_head: Per-head width of the lag attention; must satisfy $\mathrm{num\_heads} \cdot
                d_{head} = d_{model}$.
            dropout: Dropout used inside the front ends, the encoders and the latent heads. The
                shared decoder, the lag-attention probabilities and the encoder attention
                probabilities are always built at zero dropout -- see the construction sites below
                for why each is a correctness requirement rather than a preference.
            decoder_hidden: Hidden width of the shared horizon decoder.
            horizon_depth: Number of dilated blocks in the horizon core.
            horizon_kernel: Horizon-convolution kernel width.
            horizon_film: Whether to FiLM-condition each horizon step on the decoder state.
            horizon_embed_std: Standard deviation the horizon-step embedding is re-seeded at,
                *after* the generic initialisation. The default $0.02$ leaves the core's own seed in
                place; a larger value (shipped $0.8$) breaks the near-degeneracy of the $H$ horizon
                tokens so per-block FiLM has token-specific structure to modulate.
            head_init_calibration: Calibrate every distribution head onto the trivial
                $\mathcal{N}(0, 1)$ predictor at init: the shared decoder's output heads onto
                $\mu = 0, \sigma = 1$, so the raw-target NLL starts near that predictor's level
                rather than orders of magnitude above it, and the prior head's log-variance onto
                exactly $0$ ($\sigma_p = 1$, the scale anchor's optimum).
            a_head_gain: LayerNorm gain applied to the attended source summary inside the
                head-structured posterior fusion. The shipped $2.0 = \sqrt{d_{model}/d_{head}}$
                rescales the $d_{head}$-wide summary up so it is not out-columned by the
                $d_{model}$-wide target state in the fusion.
            frontend_kernels: One depthwise kernel per front-end stage. A constructor default rather
                than a configuration key, following the precedent ``ROPE_BASE`` sets in the
                sibling's blocks: no arm varies it, and the reach guard bounds any future choice.
                Both streams are built at the same schedule -- a per-stream difference would be one
                nobody has measured.
            frontend_reach_budget_s: Cap on each front end's **backward** receptive field, in
                seconds; ``None`` derives it from the warm-up as before. This is the one quantity
                in this model that plays the role ``causal_reach_budget_s`` plays in the
                feature-based siblings, and it is deliberately not that key: those models prune
                channels whose *forward* reach reads past the anchor, and these front ends are
                one-sided by construction, so there is no forward leak to bound. What there is
                instead is a backward reach, and a front end that reached further back than the
                warm-up would train its earliest anchors against the zero-padded transient at the
                segment's start. Configured rather than derived because it is the knob a depth or
                kernel arm has to move together with the stack, and a silently derived bound would
                make such an arm fail at construction with a number nobody chose.
            encoder_conv_kernels: Kernel width per block of each encoder's causal depthwise
                convolution stem. An empty schedule builds no stem, which is a stem-free
                architecture arm.
            encoder_conv_dilations: Dilation per stem block; positional against
                ``encoder_conv_kernels``.
            encoder_num_heads: Encoder self-attention heads $H_e$. The derived head width
                $d_{model} / H_e$ must be even, which rotary position encoding requires.
            encoder_d_ff: Encoder feed-forward hidden width $d_{\mathrm{ff}}$.
            target_attention_blocks: Causal Transformer blocks in the target encoder.
            source_attention_blocks: Causal Transformer blocks in the source encoder.
            source_attention_window: Causal window $W_U$ of the source encoder's attention, in
                steps, or ``None`` for the full causal prefix. Bounded by default and by design: the
                source state's reach is then shorter than the lag search range, so the encoder
                characterises a local neighbourhood and the lag attention selects which neighbourhood
                matters. The target's context is deliberately not a parameter -- it is the full
                prefix in every arm.
            logvar_clamp: ``(lo, hi)`` effective range of every log-variance in the model.
            mu_scale: Saturation magnitude of the tanh-bounded prior mean.
            delta_mu_scale: Saturation magnitude of the tanh-bounded posterior mean delta.
            delta_logvar_scale: Saturation magnitude of the posterior log-variance delta.
            source_dropout: Dropout rate for the SOURCE pathway alone -- its input adapter or
                front end, its encoder, and the attended source summary inside the posterior
                fusion. ``None`` reproduces the pre-key model exactly, which is *not* one number:
                the adapter and encoder always ran at ``dropout`` and keep doing so, while the
                posterior fusion's dropout on the attended source summary is a site this key
                introduced and so resolves to $0$. Any explicit value applies to all three.
                Deliberately not applied to the lag attention (its probabilities must stay
                dropout-free or the per-lag KL attribution stops being exact) nor to the
                shared decoder (one module invoked twice would draw two masks).
            posterior_logvar_mode: ``'residual'`` builds the posterior log-variance as a
                bounded delta on the prior's pre-bound raw value; ``'independent'`` gives it
                its own head. ``'independent'`` is shipped because the residual form routes
                the full branch's pressure to sharpen $\sigma^q$ onto the PRIOR's tensor, so
                $D_1$ drags the prior's scale down alongside its own -- the second of the two
                paths driving the prior-variance collapse, and the one that survives turning
                off the base branch's noise. See :class:`PosteriorHead` for what it costs.
            use_entmax: Use ``entmax15`` lag attention, which can assign a lag exactly zero weight.
            attention_grad_checkpoint: Recompute the lag attention in the backward pass.
            lag_bias_init: ``'normal'`` or ``'alibi_decay'``.
            alibi_slope_scale: Multiplier on the ``'alibi_decay'`` slopes.
            query_uses_logvar: Whether the lag-attention query reads the prior log-variance as well
                as its mean. Both forms are target-only, so source purity is untouched.
            base_decode: How the base branch's latent is obtained -- ``'sample'`` draws
                $z^p = \mu^p + \sigma^p \epsilon$ under the shared $\epsilon$, ``'mean'``
                decodes at $z^p = \mu^p$. ``'mean'`` is shipped, and it is the only change
                that removes a *direct* downward pressure on the prior's scale: $D_0$ decodes
                a **sample** from the prior, sampling noise can only degrade a forecast, so
                its gradient on $\ell^p$ points down without limit and nothing in the
                objective opposes it from below. Under ``'mean'`` the prior's scale is left
                to the KL, which pulls it *up* to cover $q$, and to the scale anchor. The
                posterior branch stays stochastic either way, so the latent the coupling is
                read from is still sampled.
            coverage_floor: Minimum valid fraction of an anchor's forecast window for the anchor to
                enter the loss at all.
            init_weights: Apply the standard initialisation, and the depthwise correction it needs,
                before the delta heads are zeroed.

        Raises:
            ValueError: If $\mathrm{num\_heads} \cdot d_{head} \ne d_{model}$, if ``max_lag`` is
                negative, if $d_z$ is not divisible by ``num_heads``, if the derived raw geometry is
                degenerate; from the front end, if ``d_model`` is not divisible by $4$, if
                ``frontend_kernels`` is not one kernel per stage, if ``raw_per_step`` disagrees with
                the front end's own total stride, or if the accumulated reach exceeds
                $\texttt{warmup\_period} \times \texttt{raw\_per\_step}$; and from the encoder, if
                the stem schedules disagree in length, fewer than one attention block is requested,
                the window is not positive, or $d_{model}$ does not divide into
                ``encoder_num_heads`` even head widths.
        """
        super().__init__()

        # Validate the geometry before building anything: each of these produces a model that is
        # wrong rather than one that fails. There are no channel-count arguments to check -- the two
        # inputs are single-channel raw signals, and their *length* is a geometry fact the forward
        # checks against the trimmed grid rather than a declared width the constructor takes on
        # trust.
        if int(num_heads) * int(d_head) != int(d_model):
            raise ValueError(
                f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
            )
        # A negative max_lag gives an empty attention window: the attended source collapses to a
        # bias, and the model trains to completion having never read the source at all -- then
        # reports its KL as a coupling measurement of it.
        if int(max_lag) < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        # Unconditional: the posterior is head-structured by design, not by flag, because only
        # head-aligned latent groups make the per-head KL an additive decomposition.
        if int(d_z) % int(num_heads) != 0:
            raise ValueError(
                f"the head-structured latent requires d_z % num_heads == 0, "
                f"got d_z={d_z}, num_heads={num_heads}"
            )

        # The trimmed-grid geometry is an explicit attribute, validated on construction. Its index
        # identities (forecast of anchor t starts at raw 16*(t+1)) are what every raw target and mask
        # downstream is built against -- and, uniquely to this model, what the *inputs* are checked
        # against too, since the front ends consume the same raw grid the target is cut from.
        self.geometry = TrimmedRawGeometry(
            raw_len=int(sequence_length) * int(raw_per_step),
            decimation=int(raw_per_step),
            horizon=int(horizon),
            warmup=int(warmup_period),
        )

        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.horizon = int(horizon)
        self.raw_per_step = int(raw_per_step)
        self.warmup_period = int(warmup_period)
        self.max_lag = int(max_lag)
        self.num_heads = int(num_heads)
        self.mu_scale = float(mu_scale)
        self.delta_mu_scale = float(delta_mu_scale)
        self.delta_logvar_scale = float(delta_logvar_scale)
        self.posterior_logvar_mode = validate_choice(
            posterior_logvar_mode, POSTERIOR_LOGVAR_MODES, "posterior_logvar_mode"
        )
        # Two resolutions of the one key, because the source-side sites did not all start from
        # the same place and "unchanged" therefore means different numbers at each.
        #
        # The PATHWAY sites -- input adapter/front end and source encoder -- already existed and
        # always ran at the global `dropout`, so an unset key must resolve to `dropout` there.
        # The posterior fusion's dropout on the attended source summary is a NEW site introduced
        # with this key; before it, `a` entered the fusion with no dropout at all, so an unset key
        # must resolve to 0.0 there. Resolving both to `dropout` would add p=dropout inside the
        # posterior of every run that leaves the key unset -- invisible in eval mode, and enough
        # to make the source_dropout sweep arms measure against the wrong baseline.
        #
        # Setting the key moves both together, which is the intent: one source-pathway rate.
        self.source_dropout = float(dropout if source_dropout is None else source_dropout)
        self.posterior_source_dropout = 0.0 if source_dropout is None else float(source_dropout)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.coverage_floor = float(coverage_floor)
        self.base_decode = validate_choice(base_decode, BASE_DECODE_CHOICES, "base_decode")
        # Init-policy bundle (zero-parameter, applied in the post-init block below). Each stores its
        # configured value here and re-initialises after the generic init; the recorded defaults are
        # exact no-ops.
        self.horizon_embed_std = float(horizon_embed_std)
        self.head_init_calibration = bool(head_init_calibration)
        self.a_head_gain = float(a_head_gain)

        # The raw-target index grid is geometry-fixed, so it is built once and cached as a buffer
        # (moves with the module across devices). Non-persistent: a geometry-shaped tensor in the
        # state_dict would make a checkpoint unloadable across geometries, and the load failure would
        # name misaligned keys rather than the geometry.
        self.register_buffer(
            "future_index", build_future_index(self.geometry), persistent=False
        )

        # Two independently parameterised front ends at identical settings, in place of the two
        # stored-feature adapters. The budget bounds how far BACK a front end may see: an anchor
        # outside the warm-up that reached further would be trained against the zero-padded
        # transient at the segment's start. `None` derives it from the warm-up, which is what the
        # model did before the key existed; a configured value is converted from seconds at the
        # raw sampling rate and may only tighten that ceiling, never raise it.
        self.frontend_reach_budget = self._resolve_frontend_budget(frontend_reach_budget_s)
        self.target_frontend = CausalRawFrontend(
            d_model=self.d_model,
            raw_per_step=self.raw_per_step,
            reach_budget=self.frontend_reach_budget,
            kernels=frontend_kernels,
            dropout=dropout,
        )
        self.source_frontend = CausalRawFrontend(
            d_model=self.d_model,
            raw_per_step=self.raw_per_step,
            reach_budget=self.frontend_reach_budget,
            kernels=frontend_kernels,
            dropout=self.source_dropout,
        )

        # Two independent encoders, differing only in depth and in how much context the attention
        # admits. Separate instances, never a shared one: a shared encoder would make the source
        # state a function of the target and destroy the purity the KL readout rests on.
        self.target_encoder = CausalConvTransformerEncoder(
            d_model=d_model,
            sequence_length=self.sequence_length,
            conv_kernels=encoder_conv_kernels,
            conv_dilations=encoder_conv_dilations,
            num_attention_blocks=target_attention_blocks,
            num_heads=encoder_num_heads,
            d_ff=encoder_d_ff,
            attention_window=None,
            dropout=dropout,
        )
        self.source_encoder = CausalConvTransformerEncoder(
            d_model=d_model,
            sequence_length=self.sequence_length,
            conv_kernels=encoder_conv_kernels,
            conv_dilations=encoder_conv_dilations,
            num_attention_blocks=source_attention_blocks,
            num_heads=encoder_num_heads,
            d_ff=encoder_d_ff,
            attention_window=source_attention_window,
            dropout=self.source_dropout,
        )

        self.prior_head = FullLatentPriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
        )

        # The attention query is a projection of the prior belief, not of H^y: the question asked of
        # the source memory is "what would move *this* belief", posed from the latent the posterior
        # will then residually correct. Both forms are target-only (read off the prior head on h_y),
        # so source purity holds either way.
        self.query_uses_logvar = bool(query_uses_logvar)
        query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
        self.query_proj = nn.Linear(query_in, d_model)

        # Attention dropout must be zero, not merely small: dropout is applied to the attention
        # probabilities before they are returned, and the per-lag KL attribution is exact only if the
        # returned weights are the ones the posterior actually consumed.
        self.lag_attn = LagCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            max_lag=max_lag,
            dropout=0.0,
            use_entmax=use_entmax,
            grad_checkpoint=attention_grad_checkpoint,
            lag_bias_init=lag_bias_init,
            alibi_slope_scale=alibi_slope_scale,
        )
        self.posterior_head = PosteriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            delta_mu_scale=self.delta_mu_scale,
            head_structured=True,
            num_heads=num_heads,
            d_head=d_head,
            delta_logvar_scale=self.delta_logvar_scale,
            posterior_logvar_mode=self.posterior_logvar_mode,
            # The posterior-fusion resolution, not the pathway one: see __init__ for why an
            # unset key means 0.0 here and `dropout` at the pathway sites.
            source_dropout=self.posterior_source_dropout,
        )
        self.te_analysis = TEAnalysisHead()

        # ONE shared decoder, invoked twice per forward -- on z^p and on z^q -- and receiving
        # nothing else. Its input width is d_z (the latent, not an encoder state), and out_channels
        # here counts RAW SAMPLES PER HORIZON TOKEN, not feature channels. Dropout must be zero, not
        # stylistic: invoking one module twice draws two independent dropout masks, so base and full
        # would differ at initialisation even with z^p == z^q, and independent noise would enter the
        # base-minus-full readout on every training step.
        self.horizon_core = HorizonDecoderCore(
            d_hidden=decoder_hidden,
            horizon=horizon,
            kernel_size=horizon_kernel,
            depth=horizon_depth,
            film=horizon_film,
            film_per_block=True,
        )
        self.decoder = BaselineFutureDecoder(
            core=self.horizon_core,
            d_model=d_z,
            out_channels=self.raw_per_step,
            d_hidden=decoder_hidden,
            dropout=0.0,
            logvar_clamp=logvar_clamp,
        )

        # The posterior consumes the per-head summaries, so the attention's output projection W_o
        # feeds nothing: it receives no gradient and is never updated either way. Clearing
        # requires_grad makes that explicit and drops it from DDP's expectation set.
        for parameter in self.lag_attn.W_o.parameters():
            parameter.requires_grad_(False)

        # The initialisation order is load-bearing, top to bottom.
        #
        #: How many depthwise convolutions the variance-preserving pass re-initialised. Recorded
        #: because the count is the only evidence the pass was not a silent no-op, and because it is
        #: strictly larger here than in the sibling at equal stem settings: the front ends contribute
        #: one depthwise filter bank per stage on top of each encoder's stem.
        self.n_depthwise_init = 0
        if init_weights:
            initialization(self)
            # Immediately after the generic pass, never before it. `initialization` xavier-fills
            # every nn.Conv1d, and on a depthwise (C, 1, k) weight Xavier reads fan_in = k against
            # fan_out = Ck -- a standard deviation a factor sqrt((1 + C)/2) = 8.03 too small at
            # C = 128, independent of k, so the stem and the front-end stages would start an order of
            # magnitude too quiet and no kernel sweep could reveal it. Tied to `init_weights` because
            # it exists to repair that pass: torch's own Conv1d default already reads the depthwise
            # fan correctly, so without the generic pass there is nothing to repair.
            #
            # The front ends' fixed anti-alias filters are untouched by both passes: they are
            # non-persistent buffers applied with F.conv1d rather than nn.Conv1d weights, which is
            # exactly what keeps `initialization` from replacing them with random values.
            self.n_depthwise_init = init_depthwise_(self)
            # The third thing the generic pass undoes, and the least visible. `initialization`
            # zeros every nn.Linear bias, which includes the four stage projections per front end
            # -- the only biases in that stack, and the one mechanism keeping a fully invalid
            # window off an exactly zero token. Every other front-end operator is bias-free, so a
            # zeroed projection makes the whole cascade homogeneous: featurise gives the zero
            # vector, RMSNorm(0) = 0, LayerScale(0) = 0, and the emitted token is exactly zero.
            # A zero token entering repeated pre-normalisation is the accident the sibling's input
            # adapter records reaching gradient norms around 1e26; measured here, a 26-step gap
            # takes the global norm from 2.5e4 to 2.1e17, all of it on stage 0's bias, which under
            # gradient_clip_val rescales the whole batch's gradient to nothing without the loss
            # ever spiking. Tied to `init_weights` for the same reason the depthwise repair is:
            # torch's own Linear default is already non-zero, so with no generic pass there is
            # nothing to repair.
            self._restore_frontend_stage_bias()
        # After the generic init, never before: initialization xavier-fills every nn.Linear and would
        # otherwise undo the zeroing -- and with it the exact zero-KL start.
        self._zero_init_delta_heads()
        # The same undoing hits the per-block FiLM generators: the core zero-inits them for an
        # identity-at-init, and `initialization` xavier-refills them. Re-zeroing here is what makes
        # the identity actually true -- at step 0 the per-block-FiLM decoder is bitwise the FiLM-free
        # decoder, so the latent enters the trajectory only as training moves the generators off
        # zero.
        self._zero_init_film_generators()

        # Init-policy bundle: three zero-parameter re-initialisations, each applied only when its
        # config value leaves the constructor default, so a default-flag model is bitwise the
        # pre-bundle one. They run after the generic init and after the zeroings above, and touch
        # disjoint parameters, so their order is immaterial.
        if self.horizon_embed_std != 0.02:
            self._reinit_horizon_embedding()
        if self.head_init_calibration:
            self._calibrate_output_heads()
            self._calibrate_prior_scale()
        if self.a_head_gain != 1.0:
            self._set_a_head_gain()

    def _resolve_frontend_budget(self, budget_s: Optional[float]) -> int:
        r"""Resolve the front ends' backward reach ceiling, in raw samples.

        The warm-up is the hard ceiling either way: an anchor at step $w$ reaches back
        $w \cdot R$ raw samples before the segment starts, and everything before that is
        zero padding rather than signal. A configured budget may tighten that ceiling and may
        not raise it, so the guard is a real bound rather than a way to opt out of one.

        Args:
            budget_s: The configured backward reach in seconds, or ``None`` to derive the
                ceiling from ``warmup_period``.

        Returns:
            The budget in raw samples.

        Raises:
            ValueError: If ``budget_s`` is not positive, or exceeds the warm-up ceiling -- the
                second names both numbers, because the fix is to raise ``warmup_period`` and a
                message that named only the budget would send the reader to the wrong key.
        """
        ceiling = self.warmup_period * self.raw_per_step
        if budget_s is None:
            return ceiling

        budget = float(budget_s)
        if budget <= 0.0:
            raise ValueError(
                f"frontend_reach_budget_s must be > 0, got {budget_s}; use null to derive the "
                f"ceiling from warmup_period"
            )
        # Seconds -> raw samples through the decimated grid: R samples per step, SECONDS_PER_STEP
        # seconds per step. Floored, so a budget always buys at most what it asked for.
        samples = int(budget * self.raw_per_step / SECONDS_PER_STEP)
        if samples > ceiling:
            raise ValueError(
                f"frontend_reach_budget_s={budget_s} is {samples} raw samples, past the "
                f"{ceiling} the warm-up allows (warmup_period={self.warmup_period} steps x "
                f"raw_per_step={self.raw_per_step}). A front end reaching further back than the "
                f"warm-up trains its earliest anchors against the segment's zero-padded "
                f"transient; raise warmup_period to admit this budget."
            )
        return samples

    @property
    def source_delay_steps(self) -> int:
        r"""The causal input delay $\delta$ the source stream is read with, in decimated steps.

        Always zero, and structurally so: there is no channel gate to delay anything. The stored
        features the sibling reads are two-sided, so it delays a channel to keep the leak inside a
        budget; a raw signal read through a strictly one-sided front end has nothing to compensate
        for, and a lag $\ell$ here refers to source content exactly $\ell$ steps in the past.

        Exposed anyway, and asserted in the suite, because the diagnostic figure reads it through a
        silent ``getattr(model, "source_delay_steps", 0)``: a model that stopped exposing it would
        keep drawing the same figure, with no error, and the default would happen to be right today
        and wrong the moment anything delays a stream.
        """
        return 0

    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        """Zero a linear layer's weight and, if present, its bias."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _restore_frontend_stage_bias(self) -> None:
        r"""Undo the generic pass's zeroing of the front-end stage projections' biases.

        :func:`initialization` zeros every ``nn.Linear`` bias it walks, which is right for this
        model everywhere except one place. The front-end stage projection is the single biased
        operator in either front end -- the gated convolution block's projections, its depthwise
        convolution, ``RMSNorm`` and ``LayerScale`` are all bias-free -- so with its bias at zero the
        whole cascade is homogeneous in its input: a fully invalid window featurises to the zero
        vector and every stage maps zero to zero, so the emitted token is *exactly* zero rather than
        the learnable "this window is empty" constant the stage was given a bias to provide.

        The consequence is not a small one. An exactly zero token entering repeated
        pre-normalisation produces derivatives of order $1/\sqrt{\epsilon}$; measured on this model,
        a $26$-step ($104$ s) validity gap takes the global gradient norm from $2.5 \times 10^4$ to
        $2.1 \times 10^{17}$, entirely on the first stage's bias. Under a global-norm clip that
        rescales *every* parameter's gradient for that batch to nothing, while the loss itself moves
        too little for the spike breaker to fire -- so the run silently learns from a fraction of
        its data. Gaps of that length are routine in this dataset.

        Restores torch's own ``nn.Linear`` default, $\mathcal U(-1/\sqrt{C_{\mathrm{in}}},
        1/\sqrt{C_{\mathrm{in}}})$, rather than inventing a scale: the bias is a starting point for
        a learnable constant, and the layer's own default is what it would have had if the generic
        pass had not walked it.

        Initialisation only, and called only when the generic pass ran -- torch's default is already
        non-zero, so with no generic pass there is nothing to repair.
        """
        for frontend in (self.target_frontend, self.source_frontend):
            for stage in frontend.stage_modules:
                bound = 1.0 / math.sqrt(float(stage.proj.in_features))
                nn.init.uniform_(cast(nn.Linear, stage.proj).bias, -bound, bound)

    def _zero_init_delta_heads(self) -> None:
        r"""Zero the posterior delta heads, so the model starts asserting the source says nothing.

        With the mean and log-variance deltas at zero, $q \equiv p$ exactly, the shared $\epsilon$
        makes $z^q = z^p$ sample by sample, and the shared decoder makes the base and full forecasts
        bitwise identical. Any coupling the model later reports had to be learned against that null.

        Initialisation only. Calling it on a trained model would silently discard exactly the
        parameters that carry what was learned.
        """
        for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
            # delta_logvar_head is None under posterior_logvar_mode='independent', where the
            # posterior's log-variance is not a delta on anything and is handled below.
            if module is None:
                continue
            layers = list(module) if isinstance(module, nn.ModuleList) else [module]
            for layer in layers:
                self._zero_linear(cast(nn.Linear, layer))

        # The independent log-variance head has no zero that means "agree with the prior": its
        # output IS the posterior's log-variance, so zeroing it would assert the midpoint of the
        # clamp. Seeded instead at the pre-image of log-variance 0 -- zero weight, bias at
        # log((0 - lo) / (hi - 0)) -- so the head starts input-independent at sigma_q = 1. That
        # is exactly where head_init_calibration puts the PRIOR, so under the shipped flags the
        # two agree and the KL is still exactly zero at init. Without that flag the prior starts
        # elsewhere and the init KL is small but nonzero, which is the accepted cost of the mode.
        independent = self.posterior_head.logvar_post_head
        if independent is not None:
            # Read off the head that owns the bound, not off the model: only some of these models
            # keep logvar_clamp as an attribute, and the head is the thing that applies it.
            lo, hi = self.posterior_head.logvar_clamp
            if not lo < 0.0 < hi:
                raise ValueError(
                    f"posterior_logvar_mode='independent' seeds the head at unit scale, which "
                    f"needs 0 inside logvar_clamp; got ({lo}, {hi})"
                )
            bias_value = math.log((0.0 - lo) / (hi - 0.0))
            layers = (
                list(independent)
                if isinstance(independent, nn.ModuleList)
                else [independent]
            )
            for layer in layers:
                linear = cast(nn.Linear, layer)
                nn.init.zeros_(linear.weight)
                linear.bias.data.fill_(bias_value)

    def _zero_init_film_generators(self) -> None:
        r"""Zero every FiLM generator in the horizon core, so per-block FiLM starts as an identity.

        The core zero-inits its generators for an identity at construction, but the generic
        :func:`initialization` xavier-fills every ``nn.Linear`` afterwards and undoes it. Re-zeroing
        restores it, so at step 0 the decoder ignores $z$ inside the FiLM path and consults it only
        as training drives the generators off zero.

        Initialisation only. Calling it on a trained model would discard exactly the modulation the
        latent learned to apply.
        """
        core = self.horizon_core
        film_layers: List[nn.Module] = []
        if core.film_gen is not None:
            film_layers.append(core.film_gen)
        if core.refine.film is not None:
            film_layers.extend(core.refine.film)
        for layer in film_layers:
            self._zero_linear(cast(nn.Linear, layer))

    def _reinit_horizon_embedding(self) -> None:
        r"""Re-seed the horizon-step embedding at ``horizon_embed_std``, to break token symmetry.

        The core seeds ``horizon_embedding`` at $\mathcal{N}(0, 0.02^2)$ -- a few percent of the
        magnitude of the broadcast projected latent -- so the $H$ horizon tokens enter the refine
        stack almost perfectly correlated and the stack must manufacture the whole trajectory shape
        in a $z$-independent direction. Re-seeding at a larger std gives each token a distinct
        starting offset, so per-block FiLM has token-specific structure to modulate from step $0$.

        Initialisation only; gated by the caller on a non-default std.
        """
        nn.init.normal_(
            self.horizon_core.horizon_embedding, mean=0.0, std=self.horizon_embed_std
        )

    def _calibrate_output_heads(self) -> None:
        r"""Calibrate the shared decoder's output heads onto the trivial predictor at init.

        Xavier-filled heads emit a high-variance mean and an over-confident low log-variance, so the
        init NLL of the raw z-scored target sits orders of magnitude above the trivial
        $\mu = 0, \sigma = 1$ predictor's -- pressure the optimiser spends its first epochs undoing,
        and a confound in any comparison that reads the first epochs. Three edits move the decoder
        onto that predictor at step $0$: shrink the mean head so $\hat\mu \approx 0$ (scaled, not
        zeroed, so a posterior perturbation still moves the two forecasts apart), set the
        log-variance bias to $\log(5/3)$ so ``smooth_bound(-5, 3)`` maps it to exactly $0$
        (i.e. $\sigma = 1$, since $\mathrm{sigmoid}(\log(5/3)) = 5/8$ and $-5 + 8 \cdot 5/8 = 0$),
        and shrink the log-variance weight so the init spread around that centre is small.

        Both forecasts share this one decoder, so base and full stay calibrated identically and every
        bitwise-at-init contract is preserved. Initialisation only; gated by the caller.
        """
        self.decoder.mean_head.weight.data.mul_(0.02)
        self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
        self.decoder.logvar_head.weight.data.mul_(0.1)

    def _calibrate_prior_scale(self) -> None:
        r"""Pin the prior head's log-variance at unit scale ($\log\sigma_p^2 = 0$) at init.

        Nothing else in the initialisation places the prior's scale: Xavier-filled, the head starts
        around $-3$ nats and nothing in the objective but the scale anchor pushes it back up, so it
        collapses onto the clamp floor within an epoch on real data. This is the remaining half of
        the trivial-predictor calibration -- the decoder starts at $\mathcal{N}(0, 1)$ over the
        target, the prior at $\sigma_p = 1$ over the latent.

        The head is a ``ResidualMLP`` returning ``body(x) + skip_proj(x)`` with no single bias
        governing its output level, so the decoder's shrink-plus-bias recipe does not transfer -- and
        a shrink could not be exact anyway, since ``smooth_bound`` is a sigmoid and the mean of the
        bound is not the bound of the mean. Instead the posterior deltas' own zero-weight recipe:
        zero the final body layer's weight and the whole skip projection, and seed the final bias at
        the pre-image of log-variance $0$ under ``smooth_bound(*logvar_clamp)`` -- $\log(5/3)$ at the
        shipped $(-5, 3)$ -- so the raw output is input-independent and the bounded output exactly
        $0$. The zeroed layers still receive gradient (the final layer against its activations, the
        skip against its input), and the posterior residual is built on the same raw tensor, so the
        exact zero-KL start is untouched. Initialisation only; gated by the caller.

        Raises:
            ValueError: If the clamp interval does not contain $0$, which makes unit scale
                unreachable, or if the head's skip path is an identity, which the recipe cannot
                silence.
        """
        lo, hi = self.logvar_clamp
        if not lo < 0.0 < hi:
            raise ValueError(
                f"prior scale calibration needs 0 inside logvar_clamp, got ({lo}, {hi})"
            )
        head = self.prior_head.logvar_prior_head
        if not isinstance(head.skip_proj, nn.Linear):
            raise ValueError(
                "prior scale calibration requires a projected skip on the log-variance head; "
                "with d_model == d_z the skip is an identity and the output cannot be pinned"
            )
        self._zero_linear(head.skip_proj)
        final = cast(nn.Linear, head.body[-1])
        nn.init.zeros_(final.weight)
        final.bias.data.fill_(math.log((0.0 - lo) / (hi - 0.0)))

    def _set_a_head_gain(self) -> None:
        r"""Set the posterior fusion's attended-source LayerNorm gain to ``a_head_gain``.

        The head-structured posterior fuses a $d_{model}$-wide target state with a $d_{head}$-wide
        attended source summary; at unit gain the summary is out-columned $d_{model} : d_{head}$ and
        receives correspondingly less gradient while the KL is open. Setting the ``a_head_norm`` gain
        to $\sqrt{d_{model}/d_{head}} = 2$ rescales the summary up so the two inputs enter the fusion
        at comparable magnitude.

        The posterior deltas are still zero at init, so this does not perturb the exact zero-KL
        start. Initialisation only; gated by the caller on a non-unit gain.
        """
        nn.init.constant_(self.posterior_head.a_head_norm.weight, self.a_head_gain)

    def _reparameterize_shared(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Draw one $\epsilon$ and produce both latents from it.

        Common random numbers: the posterior is always $z^q = \mu^q + \sigma^q \epsilon$,
        and under ``base_decode='sample'`` the prior sample $z^p = \mu^p + \sigma^p \epsilon$
        comes from the *same* draw -- so $p_t = q_t$ implies $z^p_t = z^q_t$ sample by sample
        and the base-minus-full readout carries no independent sampling noise.

        Under ``base_decode='mean'`` the base branch is decoded at $z^p = \mu^p$ instead,
        and the identity above becomes an approximation: at initialisation the two forecasts
        differ by the posterior's own noise rather than being bitwise equal. That is
        deliberate. The draw is still taken -- it still serves the posterior, and taking it
        unconditionally keeps a run's RNG consumption identical across the two modes, so an
        arm that flips this key differs by the mode alone, not by every subsequent random
        number.

        The branch reads a constructor-fixed string, never a tensor, so it is identical on
        every rank at every step and cannot drop a parameter from the graph.

        Args:
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Prior log-variance ``(B, T, d_z)``.
            mu_post: Posterior mean ``(B, T, d_z)``.
            logvar_post: Posterior log-variance ``(B, T, d_z)``.

        Returns:
            ``(z_prior, z_post)``, both ``(B, T, d_z)``.
        """
        epsilon = torch.randn_like(mu_prior)
        z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
        if self.base_decode == "mean":
            # logvar_prior is untouched here, and still reaches the graph through the KL and the
            # prior scale rate -- so the prior's log-variance head keeps receiving gradient, from
            # the two terms that want it WIDE rather than from the one that wanted it narrow.
            return mu_prior, z_post
        z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
        return z_prior, z_post

    def forward(
        self,
        y_raw: torch.Tensor,
        u_raw: torch.Tensor,
        weight: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full pipeline.

        The two signals are named for their **roles** rather than for the batch fields that carry
        them: a net takes tensors, and ``nets/`` may not know what anything was called on disk. It is
        also the argument order the loss then scores against -- ``y_raw`` is both the target stream's
        input and, through the task, the reconstruction target -- so a swapped pair is a genuine
        hazard here in a way it is not for the sibling, whose two streams have different widths and
        would fail a shape check. It is caught by the source-purity probes instead.

        Args:
            y_raw: Loader-normalized raw target signal ``(B, T*R)``.
            u_raw: Loader-normalized raw source signal ``(B, T*R)``.
            weight: Decimated validity signal ``(B, T)``, shared by both front ends.

        Returns:
            Exactly these keys -- the same set, at the same shapes, as the model this one is compared
            against, and there is deliberately no ``decoder_state`` and no ``delta_mu_src``, because
            neither pathway exists in this architecture:

            * ``mu_prior``, ``logvar_prior``, ``raw_logvar_prior`` -- the target-only prior, each
              ``(B, T, d_z)``.
            * ``mu_post``, ``logvar_post`` -- the source-conditioned posterior, ``(B, T, d_z)``.
            * ``z_prior``, ``z_post`` -- the two latents the decoder is invoked on,
              ``(B, T, d_z)``. Under ``base_decode='sample'`` they are paired samples under
              one $\epsilon$. Under the shipped ``'mean'`` only ``z_post`` is sampled and
              ``z_prior`` **is** the ``mu_prior`` tensor -- the same object, not a copy -- so
              a caller computing a prior-sample statistic off it gets the mean instead, and
              an in-place write to it corrupts ``mu_prior`` for every other consumer of this
              dict. Read ``self.base_decode`` before treating it as a draw.
            * ``target_state``, ``source_state`` -- encoder history states, ``(B, T, d_model)``.
            * ``attended_source_heads`` -- per-head attended summaries
              ``(B, T, num_heads, d_head)``.
            * ``attn_weights`` -- lag-attention probabilities ``(B, T, num_heads, L)`` in lag order.
            * ``mu_base``, ``logvar_base``, ``mu_full``, ``logvar_full`` -- the two raw forecasts,
              each ``(B, T - H, H, R)``.
            * ``kld_per_t`` -- the per-step KL, summed over $d_z$, ``(B, T)``.
            * ``kld_per_t_per_head`` -- its additive per-latent-group split, ``(B, T, num_heads)``.
            * ``source_kl_lag_map`` -- the KL attributed across lags by the attention weights,
              ``(B, T, L)``; sums over lags to ``kld_per_t`` exactly, because the attention
              probabilities carry no dropout.
            * ``mu_prior_sat_frac``, ``delta_mu_sat_frac`` -- scalar saturation diagnostics.

        Raises:
            ValueError: If either raw signal is not ``(B, sequence_length * raw_per_step)`` or the
                weight is not ``(B, sequence_length)``.
        """
        # Both guards read shape metadata only, never tensor content: a forward that branched on a
        # value would drop parameters from the graph on some ranks and not others, which hangs a DDP
        # run under find_unused_parameters=False rather than failing it. A shape guard that raises
        # fails the run, which is the outcome wanted here.
        if (
            y_raw.ndim != 2
            or u_raw.ndim != 2
            or y_raw.shape[-1] != self.geometry.raw_len
            or u_raw.shape[-1] != self.geometry.raw_len
        ):
            raise ValueError(
                f"expected two raw signals of shape (B, {self.geometry.raw_len}) = "
                f"(B, sequence_length {self.sequence_length} * raw_per_step "
                f"{self.raw_per_step}), got {tuple(y_raw.shape)} and {tuple(u_raw.shape)}; a "
                f"loader running at a different trim_minutes is what produces this"
            )
        if weight.ndim != 2 or weight.shape[-1] != self.sequence_length:
            raise ValueError(
                f"expected a weight of shape (B, {self.sequence_length}) on the trimmed decimated "
                f"grid, got {tuple(weight.shape)}"
            )

        # The one architectural difference from the model this is compared against, in two lines:
        # a learned strictly causal map from the raw grid instead of a projection of stored
        # two-sided coefficients. Everything below is the sibling's, unchanged.
        h_y = self.target_encoder(self.target_frontend(y_raw, weight))
        h_u = self.source_encoder(self.source_frontend(u_raw, weight))

        mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)

        # The attended output (W_o's fused projection) is discarded: the head-structured posterior
        # consumes the per-head summaries, and W_o is frozen. The query is posed from the prior
        # belief -- mu^p, or [mu^p || logvar^p] under query_uses_logvar -- both target-only.
        query = (
            torch.cat([mu_prior, logvar_prior], dim=-1)
            if self.query_uses_logvar
            else mu_prior
        )
        _, alpha, attended_heads = self.lag_attn(self.query_proj(query), h_u)

        mu_post, logvar_post = self.posterior_head(
            h_y, attended_heads, mu_prior, raw_logvar_prior
        )
        z_prior, z_post = self._reparameterize_shared(
            mu_prior, logvar_prior, mu_post, logvar_post
        )

        # Saturation diagnostics: a bound that is always active is a bound that is binding, and a
        # binding bound is a silently mis-set hyperparameter.
        with torch.no_grad():
            mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
            ).float().mean()

        # Decode the valid anchor range only. The tail H anchors have no fully observed raw future
        # and would be discarded by the loss anyway.
        t_valid = self.geometry.t_valid
        mu_base, logvar_base = self.decoder(z_prior[:, :t_valid])
        mu_full, logvar_full = self.decoder(z_post[:, :t_valid])

        # The per-lag attribution: K_t says how much the source moved the belief, the attention
        # weights say from which lag. Head-structured, so the split is an additive decomposition
        # rather than an arbitrary slice of a shared latent.
        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
        )
        kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
            kld_btd, alpha, head_structured=True
        )

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z_prior": z_prior,
            "z_post": z_post,
            "target_state": h_y,
            "source_state": h_u,
            "attended_source_heads": attended_heads,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "kld_per_t": kld_per_t,
            "kld_per_t_per_head": kld_per_t_per_head,
            "source_kl_lag_map": source_kl_lag_map,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
        }

    def kld_tensor(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
    ) -> torch.Tensor:
        r"""Closed-form KL between two diagonal Gaussians, per step and per dimension.

        Delegates to :func:`~teb_vae.lag_attn_rws.nets.losses.kld_tensor`, the one definition every
        raw-signal architecture here reports this quantity from. The method stays because every
        consumer -- the permutation control, the diagnostic figures, an offline evaluation -- reaches
        it through the model it was handed.

        Args:
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Prior log-variance ``(B, T, d_z)``.
            mu_post: Posterior mean ``(B, T, d_z)``.
            logvar_post: Posterior log-variance ``(B, T, d_z)``.

        Returns:
            The per-step per-dimension KL ``(B, T, d_z)``.
        """
        return closed_form_kld(mu_prior, logvar_prior, mu_post, logvar_post)

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        fhr_raw: torch.Tensor,
        *,
        weight: torch.Tensor,
        beta: float = 1.0,
        beta_prior: float = 0.0,
        lambda_full: float = 1.0,
        lambda_base: float = 1.0,
        likelihood: str = "gaussian_nll",
        free_bits: float = 0.0,
    ) -> Dict[str, Any]:
        r"""Compute the four-term objective in nats per anchor.

        $$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
        + \beta\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p$$

        Delegates to :func:`~teb_vae.lag_attn_rws.nets.losses.compute_loss`, supplying this model's
        geometry, its cached raw-target index grid and its two scalar bounds. Shared rather than
        reimplemented, and that is the point: this architecture exists to be compared against the one
        it changes the input of, and a second copy of the objective would make the comparison partly
        a comparison of two losses.

        The signature keeps the sibling's argument name, so a caller holding either model reaches
        the objective the same way -- which is what lets one task, one plotting callback and one
        metric surface serve both.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            fhr_raw: Raw target signal ``(B, L_raw)``, loader-normalized. The same tensor
                :meth:`forward` was handed as ``y_raw``: one source of the target is what stops a
                model being scored against a tensor other than the one it was shown.
            weight: Decimated validity signal ``(B, T)``.
            beta: Weight on the trained KL term.
            beta_prior: Weight on the prior scale rate; ``0.0`` leaves the historical three-term
                objective while ``prior_rate`` is still reported.
            lambda_full: Weight on the full-forecast reconstruction.
            lambda_base: Weight on the base-forecast reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-dimension per-step KL floor; enters the trained KL only.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``; see the shared implementation for the metric
            list.

        Raises:
            ValueError: On an unknown ``likelihood``, a raw length that does not match the geometry,
                or a ``weight`` that does not match the trimmed grid.
        """
        return compute_raw_objective(
            forward_outputs,
            fhr_raw,
            weight=weight,
            geometry=self.geometry,
            future_index=self.future_index,
            coverage_floor=self.coverage_floor,
            logvar_clamp=self.logvar_clamp,
            beta=beta,
            beta_prior=beta_prior,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            free_bits=free_bits,
        )
