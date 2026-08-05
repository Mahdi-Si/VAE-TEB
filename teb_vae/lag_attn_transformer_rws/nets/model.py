r"""The causal conv-Transformer raw-signal lag-attention sequential VAE.

The question the model answers is: *how much does the source's recent past tell us about the
target's near raw future, that the target's own past did not already say -- and at what delay?*

The machinery, per $4$-second anchor $t$:

1. Two causal conv-Transformer encoders build history states $H^y_t$ and $H^u_t$ from the target
   and source feature streams. The target reads its full causal prefix; the source reads a bounded
   causal window, so a source state summarises a *local* neighbourhood and the lag attention below
   keeps its ability to tell adjacent delays apart.
2. A full-latent target-only **prior** $p_\theta(z_t \mid Y_{\le t}) =
   \mathcal{N}(\mu^p_t, \operatorname{diag} e^{\ell^p_t})$ is read off $H^y$.
3. Lag cross-attention, queried by a projection of $\mu^p_t$, looks back over
   $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and returns per-head attended summaries.
4. A head-structured **posterior** $q_\phi(z_t \mid Y_{\le t}, U_{\le t})$ is a bounded
   zero-initialised residual on the prior, in the same coordinate system.
5. One noise draw $\epsilon_t$ serves both distributions (common random numbers), and **one shared
   decoder**, invoked twice on $z^p_t$ and $z^q_t$ and receiving nothing else, emits the base and
   full forecasts of the next $H \cdot R = 480$ raw target samples.

Everything from step 2 down is imported unchanged from the packages that own it, including the
objective: what two architectures optimise must never diverge, or the comparison between them
measures the loss as much as the model.

Four structural properties carry the design, each enforced here rather than by convention:

* **Token causality.** $H_t = f(X_{\le t})$ per block and per encoder: the stem pads on the left
  only, the attention admits $j \le t$ only, and every normaliser reduces over channels at one
  position. Note what this does *not* claim: the input features are two-sided transforms, so
  token causality is not raw-signal causality. The channel gate bounds that leak; it does not
  remove it.
* **No decoder bypass.** The decoder's forward accepts exactly one tensor, $z$. There is no
  ``decoder_state``; gradient reaches the decoder only through the latent, so $\mu^p_t$ must carry
  the target's predictive state and $\mu^q_t - \mu^p_t$ is the additional source-derived
  predictive information.
* **Source purity.** The source pathway never sees a target tensor, and the prior never sees the
  source, so $\mathrm{KL}(q_t \Vert p_t)$ measures what the source added.
* **Exact zero at init.** The posterior deltas are zero-initialised *after* the generic weight
  initialisation, one $\epsilon$ serves both samples, and the twice-invoked decoder and the
  attention probabilities carry no dropout -- so at initialisation the KL is exactly $0$ and the
  base and full forecasts are bitwise identical, in train mode. Encoder dropout does not disturb
  that: prior and posterior are built from one common encoder forward, so both see the same
  dropped activations. Every nat of measured coupling is earned during training.

Shapes, with $B$ batch, $T$ steps, $L = \mathrm{max\_lag}+1$ lags, $H$ horizon tokens, $R$ raw
samples per token: inputs are ``(B, T, 43)``, ``(B, T, 66)`` and ``(B, T, c_u)``; the latent is
``(B, T, d_z)``; forecasts are ``(B, T - H, H, R)`` -- decoded over the valid anchor range only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.blocks import initialization
from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
from teb_vae.lag_attn.nets.heads import PosteriorHead, TEAnalysisHead
from teb_vae.lag_attn_rws.nets.delays import ChannelGate
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.heads import FullLatentPriorHead
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
from teb_vae.lag_attn_rws.nets.losses import kld_tensor as closed_form_kld
from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index
from teb_vae.lag_attn_transformer_rws.nets.blocks import init_depthwise_
from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    AvailabilityInputAdapter,
    CausalConvTransformerEncoder,
)


class SeqVaeLagAttnTrfRws(nn.Module):
    r"""Single-latent lag-attentive VAE with causal conv-Transformer history encoders.

    See the module docstring for what the pieces are for. The constructor's job is to build them
    consistently and to refuse inconsistent geometry loudly, before any of it reaches a forward.

    A standalone module rather than a subclass of the model it is compared against: that
    constructor builds the two encoders being replaced and validates a keyword schema this model
    does not have -- no recurrent depth, no extra dilation schedule, no time-pooling normaliser to
    causalise -- so the two share their *objective*, imported from the module that owns it, rather
    than their construction.
    """

    #: The causal input guards, or ``None`` when no reach budget is configured. Declared so they
    #: type as gates rather than as the ``Tensor | Module`` a bare submodule attribute would.
    target_gate: Optional[ChannelGate]
    source_gate: Optional[ChannelGate]

    #: The cached raw-target index grid, declared for the same reason: it is handed to the shared
    #: objective as a tensor argument.
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
        c_y: int = 109,
        c_u: int = 58,
        use_up_st: bool = True,
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
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        query_uses_logvar: bool = False,
        coverage_floor: float = 0.9,
        target_keep_index: Optional[Sequence[int]] = None,
        target_delays: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_delays: Optional[Sequence[int]] = None,
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Args:
            sequence_length: Decimated sequence length $T$. Together with ``raw_per_step`` it
                fixes the trimmed-grid geometry the raw target lives on, and it sizes the rotary
                tables, the window masks and the availability patterns -- all of which are fixed
                at construction, so a longer batch raises rather than silently reallocating.
            d_model: Internal width used throughout the backbone.
            d_z: Latent dimensionality. Must be divisible by ``num_heads``: the posterior is
                head-structured, so latent group $m$ is written only by attention head $m$ and the
                per-head KL is a genuine additive decomposition.
            horizon: Forecast horizon $H$ in decimated steps.
            raw_per_step: Raw samples per decimated step $R$; each horizon token emits this many
                raw samples.
            warmup_period: Initial steps excluded from every loss. The encoders need history
                before their states mean anything.
            c_y: Target feature channel count ($43$ scattering plus $66$ phase-harmonic).
            c_u: Source feature channel count. Only positivity is checked here: the widths are
                dataset facts, validated against the first real batch at the data boundary.
            use_up_st: Whether the source stream includes its scattering channels. An ablation
                toggle; ``False`` feeds the phase-harmonic channels alone.
            max_lag: Maximum past lag; the attention window is $L = \mathrm{max\_lag}+1$ wide.
            num_heads: Number of **lag-attention** heads and latent groups. Unrelated to
                ``encoder_num_heads``; nothing may couple them.
            d_head: Per-head width of the lag attention; must satisfy $\mathrm{num\_heads} \cdot
                d_{head} = d_{model}$.
            dropout: Dropout used inside the encoders and the latent heads. The shared decoder,
                the lag-attention probabilities and the encoder attention probabilities are always
                built at zero dropout -- see the construction sites below for why each is a
                correctness requirement rather than a preference.
            decoder_hidden: Hidden width of the shared horizon decoder.
            horizon_depth: Number of dilated blocks in the horizon core.
            horizon_kernel: Horizon-convolution kernel width.
            horizon_film: Whether to FiLM-condition each horizon step on the decoder state.
            horizon_embed_std: Standard deviation the horizon-step embedding is re-seeded at,
                *after* the generic initialisation. The default $0.02$ leaves the core's own seed
                in place; a larger value (shipped $0.8$) breaks the near-degeneracy of the $H$
                horizon tokens so per-block FiLM has token-specific structure to modulate.
            head_init_calibration: Calibrate every distribution head onto the trivial
                $\mathcal{N}(0, 1)$ predictor at init: the shared decoder's output heads onto
                $\mu = 0, \sigma = 1$, so the raw-target NLL starts near that predictor's level
                rather than orders of magnitude above it, and the prior head's log-variance onto
                exactly $0$ ($\sigma_p = 1$, the scale anchor's optimum).
            a_head_gain: LayerNorm gain applied to the attended source summary inside the
                head-structured posterior fusion. The shipped $2.0 = \sqrt{d_{model}/d_{head}}$
                rescales the $d_{head}$-wide summary up so it is not out-columned by the
                $d_{model}$-wide target state in the fusion.
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
                steps, or ``None`` for the full causal prefix. Bounded by default and by design:
                the source state's reach $R_U = R_{\mathrm{conv}} + N_U (W_U - 1)$ is then shorter
                than the lag search range, so the encoder characterises a local neighbourhood and
                the lag attention selects which neighbourhood matters. The target's context is
                deliberately not a parameter -- it is the full prefix in every arm.
            logvar_clamp: ``(lo, hi)`` effective range of every log-variance in the model.
            mu_scale: Saturation magnitude of the tanh-bounded prior mean.
            delta_mu_scale: Saturation magnitude of the tanh-bounded posterior mean delta.
            delta_logvar_scale: Saturation magnitude of the posterior log-variance delta.
            use_entmax: Use ``entmax15`` lag attention, which can assign a lag exactly zero weight.
            attention_grad_checkpoint: Recompute the lag attention in the backward pass.
            lag_bias_init: ``'normal'`` or ``'alibi_decay'``.
            alibi_slope_scale: Multiplier on the ``'alibi_decay'`` slopes.
            query_uses_logvar: Whether the lag-attention query reads the prior log-variance as
                well as its mean. Both forms are target-only, so source purity is untouched.
            coverage_floor: Minimum valid fraction of an anchor's forecast window for the anchor
                to enter the loss at all.
            target_keep_index: Indices of the target channels that survive the configured causal
                reach budget, into the declared ``c_y``. ``None`` keeps every channel. The model
                is still *built* with the full declared width -- the gather happens inside the
                forward, after the data boundary has checked the batch against ``c_y``.
            target_delays: One delay in decimated steps per surviving target channel, in the same
                order as ``target_keep_index``.
            source_keep_index: The same for the source stream, into the declared ``c_u``.
            source_delays: The same for the source stream.
            init_weights: Apply the standard initialisation, and the depthwise correction it
                needs, before the delta heads are zeroed.

        Raises:
            ValueError: If ``c_y`` or ``c_u`` is not positive, if $\mathrm{num\_heads} \cdot
                d_{head} \ne d_{model}$, if ``max_lag`` is negative, if $d_z$ is not divisible by
                ``num_heads``, if a keep-index is empty, out of range or not strictly ascending,
                if a delay vector does not match its keep-index, if the derived raw geometry is
                degenerate, or -- from the encoder -- if the stem schedules disagree in length,
                fewer than one attention block is requested, the window is not positive, or
                $d_{model}$ does not divide into ``encoder_num_heads`` even head widths.
        """
        super().__init__()

        # Validate the geometry before building anything: each of these produces a model that is
        # wrong rather than one that fails. The *values* of c_y and c_u are dataset facts and are
        # checked against the first real batch at the data boundary, not here; their positivity is
        # checked because nn.Linear(0, d) is legal and returns its bias, so a zero width builds a
        # model that trains to completion having never read that stream.
        if int(c_y) < 1 or int(c_u) < 1:
            raise ValueError(
                f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
            )
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
        # identities (forecast of anchor t starts at raw 16*(t+1)) are what every raw target and
        # mask downstream is built against.
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
        self.c_y = int(c_y)
        self.c_u = int(c_u)
        self.use_up_st = bool(use_up_st)
        self.max_lag = int(max_lag)
        self.num_heads = int(num_heads)
        self.mu_scale = float(mu_scale)
        self.delta_mu_scale = float(delta_mu_scale)
        self.delta_logvar_scale = float(delta_logvar_scale)
        self.logvar_clamp = (float(logvar_clamp[0]), float(logvar_clamp[1]))
        self.coverage_floor = float(coverage_floor)
        # Init-policy bundle (zero-parameter, applied in the post-init block below). Each stores
        # its configured value here and re-initialises after the generic init; the recorded
        # defaults are exact no-ops.
        self.horizon_embed_std = float(horizon_embed_std)
        self.head_init_calibration = bool(head_init_calibration)
        self.a_head_gain = float(a_head_gain)

        # The raw-target index grid is geometry-fixed, so it is built once and cached as a buffer
        # (moves with the module across devices). Non-persistent: a geometry-shaped tensor in the
        # state_dict would make a checkpoint unloadable across geometries, and the load failure
        # would name misaligned keys rather than the geometry.
        self.register_buffer(
            "future_index", build_future_index(self.geometry), persistent=False
        )

        # The causal input guard, if one is configured. Applied inside the forward, between the
        # data boundary (which sees the full declared widths) and the input adapters (which see
        # only the survivors), so a nonzero budget changes what the model reads without changing
        # what the batch must contain.
        self.target_gate = self._build_channel_gate(
            self.c_y, target_keep_index, target_delays
        )
        self.source_gate = self._build_channel_gate(
            self.c_u, source_keep_index, source_delays
        )

        # The adapters' availability terms are derived from the gates themselves, at the gate's
        # own emitted width, rather than from the constructor arguments a second time: the gate is
        # what the forward actually applies, so a delay vector read from anywhere else could
        # describe a guard the stream never got.
        self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
        self.source_adapter = self._build_adapter(self.source_gate, self.c_u, dropout)

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
            dropout=dropout,
        )

        self.prior_head = FullLatentPriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
        )

        # The attention query is a projection of the prior belief, not of H^y: the question asked
        # of the source memory is "what would move *this* belief", posed from the latent the
        # posterior will then residually correct. Both forms are target-only (read off the prior
        # head on h_y), so source purity holds either way.
        self.query_uses_logvar = bool(query_uses_logvar)
        query_in = 2 * self.d_z if self.query_uses_logvar else self.d_z
        self.query_proj = nn.Linear(query_in, d_model)

        # Attention dropout must be zero, not merely small: dropout is applied to the attention
        # probabilities before they are returned, and the per-lag KL attribution is exact only if
        # the returned weights are the ones the posterior actually consumed.
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
        )
        self.te_analysis = TEAnalysisHead()

        # ONE shared decoder, invoked twice per forward -- on z^p and on z^q -- and receiving
        # nothing else. Its input width is d_z (the latent, not an encoder state), and
        # out_channels here counts RAW SAMPLES PER HORIZON TOKEN, not feature channels. Dropout
        # must be zero, not stylistic: invoking one module twice draws two independent dropout
        # masks, so base and full would differ at initialisation even with z^p == z^q, and
        # independent noise would enter the base-minus-full readout on every training step.
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
        #: because the count is the only evidence the pass was not a silent no-op, and a
        #: stem-free arm legitimately has none.
        self.n_depthwise_init = 0
        if init_weights:
            initialization(self)
            # Immediately after the generic pass, never before it. `initialization` xavier-fills
            # every nn.Conv1d, and on a depthwise (C, 1, k) weight Xavier reads fan_in = k against
            # fan_out = Ck -- a standard deviation a factor sqrt((1 + C)/2) = 8.03 too small at
            # C = 128, independent of k, so the stem would start an order of magnitude too quiet
            # and no kernel sweep could reveal it. Tied to `init_weights` because it exists to
            # repair that pass: torch's own Conv1d default already reads the depthwise fan
            # correctly, so without the generic pass there is nothing to repair.
            self.n_depthwise_init = init_depthwise_(self)
        # After the generic init, never before: initialization xavier-fills every nn.Linear and
        # would otherwise undo the zeroing -- and with it the exact zero-KL start.
        self._zero_init_delta_heads()
        # The same undoing hits the per-block FiLM generators: the core zero-inits them for an
        # identity-at-init, and `initialization` xavier-refills them. Re-zeroing here is what makes
        # the identity actually true -- at step 0 the per-block-FiLM decoder is bitwise the
        # FiLM-free decoder, so the latent enters the trajectory only as training moves the
        # generators off zero.
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

    @staticmethod
    def _build_channel_gate(
        declared_width: int,
        keep_index: Optional[Sequence[int]],
        delays: Optional[Sequence[int]],
    ) -> Optional[ChannelGate]:
        r"""Build one stream's causal input guard, or ``None`` when it has none.

        With neither argument the stream is ungated and **no** module is created, so the unguarded
        model is structurally identical to one built before the guard existed. With either, a gate
        is built and the missing half is filled in -- a missing delay vector becomes zeros, a
        missing keep-index becomes the identity -- because a gather without delays (or the
        reverse) is far more likely to be a resolution bug than an intent.

        Args:
            declared_width: The stream's full declared channel count.
            keep_index: Surviving channel indices, or ``None`` for all of them.
            delays: Per-survivor delay in steps, or ``None`` for none.

        Returns:
            The gate, or ``None``.

        Raises:
            ValueError: Propagated from :class:`ChannelGate`, which owns the validation.
        """
        if keep_index is None and delays is None:
            return None
        return ChannelGate(
            declared_width=int(declared_width), keep_index=keep_index, delays=delays
        )

    def _build_adapter(
        self, gate: Optional[ChannelGate], declared_width: int, dropout: float
    ) -> AvailabilityInputAdapter:
        r"""Build one stream's input adapter at the width and delays its gate actually emits.

        The gate is the single source of truth for both. Reading the delays back off the
        constructed gate rather than off the constructor arguments means the availability pattern
        $m_{t,c} = \mathbb 1[t \ge \delta_c]$ cannot describe a guard the stream never received:
        the gate fills in a missing delay vector with zeros and a missing keep-index with the
        identity, and either substitution would leave the two out of step.

        Args:
            gate: The stream's guard, or ``None`` when it is unguarded.
            declared_width: The stream's declared channel count, used when there is no gate.
            dropout: Dropout probability inside the projection stack.

        Returns:
            The adapter, carrying whichever availability terms the delays call for.
        """
        width = declared_width if gate is None else gate.out_channels
        delays = None if gate is None else [int(value) for value in gate.delay.delay_steps]
        return AvailabilityInputAdapter(
            in_dim=width,
            d_model=self.d_model,
            sequence_length=self.sequence_length,
            dropout=dropout,
            delays=delays,
        )

    @property
    def source_delay_steps(self) -> int:
        r"""The causal input delay $\delta$ the source stream is read with, in decimated steps.

        Zero when no reach budget is configured. **Every lag report must add this back**: an
        attention peak at lag $\ell$ refers to source content $\ell + \delta$ steps in the past,
        so a figure or summary that omits it understates the physiological delay -- by up to two
        minutes at the $120$ s budget, with nothing failing.

        The source channels are delayed individually, so no single $\delta$ describes them all.
        The maximum is reported, which makes a lag computed from it an upper bound. Exposed here,
        on the model, because the model is what was trained: the diagnostic figure reads this
        through a silent ``getattr(model, "source_delay_steps", 0)``, so a model that stopped
        exposing it would shift the figure's lag axis with no error at all.
        """
        return 0 if self.source_gate is None else self.source_gate.max_delay

    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        """Zero a linear layer's weight and, if present, its bias."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _zero_init_delta_heads(self) -> None:
        r"""Zero the posterior delta heads, so the model starts asserting the source says nothing.

        With the mean and log-variance deltas at zero, $q \equiv p$ exactly, the shared $\epsilon$
        makes $z^q = z^p$ sample by sample, and the shared decoder makes the base and full
        forecasts bitwise identical. Any coupling the model later reports had to be learned
        against that null.

        Initialisation only. Calling it on a trained model would silently discard exactly the
        parameters that carry what was learned.
        """
        for module in (self.posterior_head.delta_mu_head, self.posterior_head.delta_logvar_head):
            layers = list(module) if isinstance(module, nn.ModuleList) else [module]
            for layer in layers:
                self._zero_linear(cast(nn.Linear, layer))

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

        Xavier-filled heads emit a high-variance mean and an over-confident low log-variance, so
        the init NLL of the raw z-scored target sits orders of magnitude above the trivial
        $\mu = 0, \sigma = 1$ predictor's -- pressure the optimiser spends its first epochs
        undoing, and a confound in any comparison that reads the first epochs. Three edits move the
        decoder onto that predictor at step $0$: shrink the mean head so $\hat\mu \approx 0$
        (scaled, not zeroed, so a posterior perturbation still moves the two forecasts apart), set
        the log-variance bias to $\log(5/3)$ so ``smooth_bound(-5, 3)`` maps it to exactly $0$
        (i.e. $\sigma = 1$, since $\mathrm{sigmoid}(\log(5/3)) = 5/8$ and $-5 + 8 \cdot 5/8 = 0$),
        and shrink the log-variance weight so the init spread around that centre is small.

        Both forecasts share this one decoder, so base and full stay calibrated identically and
        every bitwise-at-init contract is preserved. Initialisation only; gated by the caller.
        """
        self.decoder.mean_head.weight.data.mul_(0.02)
        self.decoder.logvar_head.bias.data.fill_(math.log(5.0 / 3.0))
        self.decoder.logvar_head.weight.data.mul_(0.1)

    def _calibrate_prior_scale(self) -> None:
        r"""Pin the prior head's log-variance at unit scale ($\log\sigma_p^2 = 0$) at init.

        Nothing else in the initialisation places the prior's scale: Xavier-filled, the head
        starts around $-3$ nats and nothing in the objective but the scale anchor pushes it back
        up, so it collapses onto the clamp floor within an epoch on real data. This is the
        remaining half of the trivial-predictor calibration -- the decoder starts at
        $\mathcal{N}(0, 1)$ over the target, the prior at $\sigma_p = 1$ over the latent.

        The head is a ``ResidualMLP`` returning ``body(x) + skip_proj(x)`` with no single bias
        governing its output level, so the decoder's shrink-plus-bias recipe does not transfer
        -- and a shrink could not be exact anyway, since ``smooth_bound`` is a sigmoid and the
        mean of the bound is not the bound of the mean. Instead the posterior deltas' own
        zero-weight recipe: zero the final body layer's weight and the whole skip projection,
        and seed the final bias at the pre-image of log-variance $0$ under
        ``smooth_bound(*logvar_clamp)`` -- $\log(5/3)$ at the shipped $(-5, 3)$ -- so the raw
        output is input-independent and the bounded output exactly $0$. The zeroed layers still
        receive gradient (the final layer against its activations, the skip against its input),
        and the posterior residual is built on the same raw tensor, so the exact zero-KL start
        is untouched. Initialisation only; gated by the caller.

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
        attended source summary; at unit gain the summary is out-columned $d_{model} : d_{head}$
        and receives correspondingly less gradient while the KL is open. Setting the
        ``a_head_norm`` gain to $\sqrt{d_{model}/d_{head}} = 2$ rescales the summary up so the two
        inputs enter the fusion at comparable magnitude.

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
        r"""Draw one $\epsilon$ and sample both latents with it.

        Common random numbers: $z^p = \mu^p + \sigma^p \epsilon$ and
        $z^q = \mu^q + \sigma^q \epsilon$ share the draw, so $p_t = q_t$ implies $z^p_t = z^q_t$
        sample by sample and the base-minus-full readout carries no independent sampling noise.

        Args:
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Prior log-variance ``(B, T, d_z)``.
            mu_post: Posterior mean ``(B, T, d_z)``.
            logvar_post: Posterior log-variance ``(B, T, d_z)``.

        Returns:
            ``(z_prior, z_post)``, both ``(B, T, d_z)``.
        """
        epsilon = torch.randn_like(mu_prior)
        z_prior = mu_prior + epsilon * torch.exp(0.5 * logvar_prior)
        z_post = mu_post + epsilon * torch.exp(0.5 * logvar_post)
        return z_prior, z_post

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full pipeline.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            u_stream: Source stream ``(B, T, c_u)``.

        Returns:
            Exactly these keys -- there is deliberately no ``decoder_state`` and no
            ``delta_mu_src``, because neither pathway exists in this architecture:

            * ``mu_prior``, ``logvar_prior``, ``raw_logvar_prior`` -- the target-only prior, each
              ``(B, T, d_z)``.
            * ``mu_post``, ``logvar_post`` -- the source-conditioned posterior, ``(B, T, d_z)``.
            * ``z_prior``, ``z_post`` -- paired samples under one $\epsilon$, ``(B, T, d_z)``.
            * ``target_state``, ``source_state`` -- encoder history states, ``(B, T, d_model)``.
            * ``attended_source_heads`` -- per-head attended summaries
              ``(B, T, num_heads, d_head)``.
            * ``attn_weights`` -- lag-attention probabilities ``(B, T, num_heads, L)`` in lag
              order.
            * ``mu_base``, ``logvar_base``, ``mu_full``, ``logvar_full`` -- the two raw forecasts,
              each ``(B, T - H, H, R)``.
            * ``kld_per_t`` -- the per-step KL, summed over $d_z$, ``(B, T)``.
            * ``kld_per_t_per_head`` -- its additive per-latent-group split,
              ``(B, T, num_heads)``.
            * ``source_kl_lag_map`` -- the KL attributed across lags by the attention weights,
              ``(B, T, L)``; sums over lags to ``kld_per_t`` exactly, because the attention
              probabilities carry no dropout. Named for what it is -- a source-conditioned KL --
              not for the transfer entropy it is not yet.
            * ``mu_prior_sat_frac``, ``delta_mu_sat_frac`` -- scalar saturation diagnostics.
        """
        # Concatenate at the declared widths, then gate: the surviving-channel indices are
        # positional into the full stream, and the delay is applied before the adapters so the
        # encoders never see a channel at a step it could not causally have known.
        target = torch.cat([y_st, y_ph], dim=-1)
        if self.target_gate is not None:
            target = self.target_gate(target)
        source = u_stream if self.source_gate is None else self.source_gate(u_stream)

        h_y = self.target_encoder(self.target_adapter(target))
        h_u = self.source_encoder(self.source_adapter(source))

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

        Delegates to :func:`~teb_vae.lag_attn_rws.nets.losses.kld_tensor`, the one definition both
        raw-signal architectures report this quantity from. The method stays because every
        consumer -- the permutation control, the diagnostic figures, an offline evaluation --
        reaches it through the model it was handed.

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
        reimplemented, and that is the point: this architecture exists to be compared against the
        one it replaces, and a second copy of the objective would make the comparison partly a
        comparison of two losses.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            fhr_raw: Raw target signal ``(B, L_raw)``, loader-normalized.
            weight: Decimated validity signal ``(B, T)``.
            beta: Weight on the trained KL term.
            beta_prior: Weight on the prior scale rate; ``0.0`` leaves the historical
                three-term objective while ``prior_rate`` is still reported.
            lambda_full: Weight on the full-forecast reconstruction.
            lambda_base: Weight on the base-forecast reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-dimension per-step KL floor; enters the trained KL only.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``; see the shared implementation for the metric
            list.

        Raises:
            ValueError: On an unknown ``likelihood``, a raw length that does not match the
                geometry, or a ``weight`` that does not match the trimmed grid.
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
