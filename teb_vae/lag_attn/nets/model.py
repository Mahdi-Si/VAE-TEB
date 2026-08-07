r"""The lag-attention sequential VAE.

The question the model answers is: *how much does the source's recent past tell us about the
target's near future, that the target's own past did not already say -- and at what delay?*

The machinery, per step $t$:

1. Two causal encoders build history states $H^y_t$ and $H^u_t$ from the target and source.
2. A target-only **prior** $p(z_t \mid Y_{\le t})$ is read off $H^y$.
3. Lag cross-attention lets $H^y_t$ look back over $\{H^u_{t-\ell}\}_{\ell=0}^{L-1}$ and pick
   the delay that matters.
4. A **posterior** $q(z_t \mid Y_{\le t}, U_{\le t})$ is built as a bounded residual on the prior,
   conditioned on what the attention found.
5. Two decoders forecast the next $H_d$ steps: a baseline from the target alone, and a
   correction driven by the latent.

$K_t = \mathrm{KL}(q \,\|\, p)$ is then the per-step answer, and the attention weights split it
across lags. Everything in the design exists to keep that reading honest: the prior never sees
the source, the encoders never see their own future, the baseline decoder is trained to be good
on its own so the correction cannot win by laundering target information, and the whole thing
starts at $K_t \equiv 0$ so any measured coupling had to be learned.

Shapes, with $B$ batch, $T$ steps, $L = \mathrm{max\_lag}+1$ lags, $H_d$ forecast horizon:
inputs are ``(B, T, 43)``, ``(B, T, 66)`` and ``(B, T, c_u)``; the latent is ``(B, T, d_z)``;
forecasts are ``(B, T, H_d, c_y)``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, cast

import torch
from torch import nn

from teb_vae.lag_attn.nets.attention import LagCrossAttention
from teb_vae.lag_attn.nets.blocks import causalize_norms, initialization, validate_choice
from teb_vae.lag_attn.nets.decoders import (
    BaselineFutureDecoder,
    HorizonDecoderCore,
    ResidualFutureDecoder,
)
from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter, CausalConvLstmEncoder
from teb_vae.lag_attn.nets.heads import POSTERIOR_LOGVAR_MODES, PosteriorHead, PriorHead, TEAnalysisHead

_KLD_SUPPORT_CHOICES = ("full", "anchor")
_LIKELIHOOD_CHOICES = ("mse", "gaussian_nll")

# A latent dimension counts as carrying information once its mean per-step KL clears this. Well
# below any meaningful coupling, well above float noise on a collapsed dimension.
_KLD_ACTIVE_EPS = 1e-2



class SeqVaeLagAttn(nn.Module):
    r"""Lag-attentive residual VAE with a transfer-entropy readout.

    See the module docstring for what the pieces are for. The constructor's job is to build them
    consistently and to refuse inconsistent geometry loudly, before any of it reaches a forward.
    """

    #: The causal input guards, or ``None`` when no reach budget is configured. Declared so they
    #: type as gates rather than as the ``Tensor | Module`` a bare submodule attribute would.
    target_gate: Optional[ChannelGate]
    source_gate: Optional[ChannelGate]

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 24,
        horizon: int = 30,
        warmup_period: int = 30,
        c_y: int = 109,
        c_u: int = 58,
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
        delta_logvar_scale: float = 2.0,
        posterior_logvar_mode: str = "residual",
        source_dropout: Optional[float] = None,
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        head_structured_latent: bool = False,
        kld_support: str = "full",
        lambda_perm: float = 0.0,
        perm_every_n_batches: int = 4,
        causal_norm: bool = False,
        freeze_unused_attn_proj: bool = False,
        target_keep_index: Optional[Sequence[int]] = None,
        target_delays: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_delays: Optional[Sequence[int]] = None,
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Args:
            sequence_length: Decimated sequence length $T$.
            d_model: Internal width used throughout the backbone.
            d_z: Latent dimensionality.
            horizon: Decimated forecast horizon $H_d$.
            warmup_period: Initial steps ignored by both the KL and the feature loss. The
                encoders need history before their states mean anything.
            c_y: Target feature channel count ($43$ scattering plus $66$ phase-harmonic).
            c_u: Source feature channel count. Must match what the task actually assembles:
                $43 + 15 = 58$ with ``use_up_st=True``, $15$ without. Checked against the first
                real batch rather than here -- see the note in the constructor body.
            use_up_st: Whether the source stream includes its scattering channels. An ablation
                toggle: ``False`` feeds the phase-harmonic channels alone.
            max_lag: Maximum past lag; the attention window is $L = \mathrm{max\_lag}+1$ wide.
            num_heads: Number of attention heads.
            d_head: Per-head width; must satisfy $\mathrm{num\_heads} \cdot d_{head} = d_{model}$.
            lstm_layers: Depth of the encoder LSTM branches.
            dropout: Dropout used throughout.
            decoder_hidden: Hidden width of the horizon decoders.
            horizon_depth: Number of dilated blocks in the shared horizon core.
            horizon_kernel: Horizon-convolution kernel width.
            horizon_film: Whether to FiLM-condition each horizon step on the decoder state.
            encoder_extra_dilations: Extra dilations appended to both encoders' convolution
                stacks for a longer receptive field. Each appended block uses kernel size $15$.
            logvar_clamp: ``(lo, hi)`` effective range of every log-variance in the model.
                The default keeps the variance in $[e^{-5}, e^{3}]$, which is well-conditioned
                for the closed-form Gaussian KL.
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
            use_entmax: Use ``entmax15`` attention, which can assign a lag exactly zero weight.
            attention_grad_checkpoint: Recompute the attention in the backward pass.
            lag_bias_init: ``'normal'`` or ``'alibi_decay'``.
            alibi_slope_scale: Multiplier on the ``'alibi_decay'`` slopes.
            head_structured_latent: Partition the latent into ``num_heads`` groups, each
                conditioned only on its attention head's summary. This is what makes the
                per-head KL an additive decomposition rather than an arbitrary slice. Requires
                $d_z \bmod \mathrm{num\_heads} = 0$.
            kld_support: Time support of the *training* KL -- ``'full'`` is $[warmup, T)$,
                ``'anchor'`` is $[warmup, T-H_d)$, i.e. only the steps that actually have a
                full forecast target to be judged against.
            lambda_perm: Weight of the source-permutation control. Read by the training task,
                not by this module.
            perm_every_n_batches: Period of the permutation-control schedule. Read by the
                training task.
            causal_norm: Replace every ``GroupNorm`` inside the two encoders with
                :class:`CausalGroupNorm`, restoring strict causality of $H^y$ and $H^u$. The
                decoders and horizon core are deliberately left alone: their normalisers pool
                over the forecast axis of a single anchor, not across input time.
            freeze_unused_attn_proj: Only meaningful with ``head_structured_latent``, where the
                posterior consumes the per-head summaries and the attention's output projection
                ``W_o`` feeds nothing but a diagnostic. It therefore receives no gradient and is
                never updated either way; clearing ``requires_grad`` makes that explicit and,
                more usefully, drops it from DDP's expectation set, so the run can use plain
                ``'ddp'`` instead of paying for ``find_unused_parameters=True`` every step.
            target_keep_index: Indices of the target channels that survive the configured causal
                reach budget, into the declared ``c_y``. ``None`` keeps every channel. The model
                is still *built* with the full declared width -- the gather happens inside the
                forward, after the data boundary has checked the batch against ``c_y``.
            target_delays: One delay in decimated steps per surviving target channel, in the same
                order as ``target_keep_index``.
            source_keep_index: The same for the source stream, into the declared ``c_u``.
            source_delays: The same for the source stream.
            init_weights: Apply the standard initialisation before the delta heads are zeroed.

        Raises:
            ValueError: If ``c_y`` or ``c_u`` is not positive, if
                $\mathrm{num\_heads} \cdot d_{head} \ne d_{model}$, if ``max_lag`` is negative,
                or if ``head_structured_latent`` is set and $d_z$ is not divisible by
                ``num_heads``.
        """
        super().__init__()

        # Validate the geometry before building anything: each of these produces a model that is
        # wrong rather than one that fails, and would otherwise only surface as a shape error
        # somewhere deep in a forward.
        #
        # The *values* of `c_y` and `c_u` are deliberately not validated here. They are properties
        # of the dataset, not of the model, and this constructor cannot see the dataset. The check
        # that used to live here compared them against module constants -- which is exactly what
        # went stale when the pipeline's phase-harmonic channel selection changed, and which also
        # made every pre-change checkpoint un-rebuildable. Agreement with the data is now checked
        # against the real batch in the task, where the per-field channel counts are in hand.
        #
        # Their positivity *is* checked, because it cannot go stale and because zero is silent:
        # `nn.Linear(0, d_model)` is legal and returns its bias broadcast over the batch, so a
        # zero width builds a model that trains to completion having never read that stream --
        # the same failure mode the `max_lag` guard below exists to prevent.
        if int(c_y) < 1 or int(c_u) < 1:
            raise ValueError(
                f"c_y and c_u are channel counts and must be >= 1, got c_y={c_y}, c_u={c_u}"
            )
        if int(num_heads) * int(d_head) != int(d_model):
            raise ValueError(
                f"num_heads * d_head ({num_heads}*{d_head}) must equal d_model ({d_model})"
            )
        # A negative max_lag gives an empty attention window (L = max_lag + 1 <= 0). Nothing
        # downstream objects: the einsums reduce over a zero-length axis, the attended source
        # collapses to W_o's bias, and the model trains to completion having never read the
        # source at all -- then reports its KL as a transfer-entropy measurement of it.
        if int(max_lag) < 0:
            raise ValueError(f"max_lag must be >= 0, got {max_lag}")
        if head_structured_latent and int(d_z) % int(num_heads) != 0:
            raise ValueError(
                f"head_structured_latent requires d_z % num_heads == 0, "
                f"got d_z={d_z}, num_heads={num_heads}"
            )

        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.horizon = int(horizon)
        self.warmup_period = int(warmup_period)
        self.c_y = int(c_y)
        self.use_up_st = bool(use_up_st)
        self.c_u = int(c_u)
        self.max_lag = int(max_lag)
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
        self.head_structured_latent = bool(head_structured_latent)
        self.kld_support = validate_choice(kld_support, _KLD_SUPPORT_CHOICES, "kld_support")
        self.lambda_perm = float(lambda_perm)
        self.perm_every_n_batches = int(perm_every_n_batches)

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

        # Built at the width and delays the gate actually emits, read back off the constructed
        # gate rather than off the constructor arguments a second time. Unguarded, no availability
        # term is constructed and the module is parameter-for-parameter the plain InputAdapter it
        # replaces, so a model built without a budget is unchanged.
        self.target_adapter = self._build_adapter(self.target_gate, self.c_y, dropout)
        self.source_adapter = self._build_adapter(
            self.source_gate, self.c_u, self.source_dropout
        )

        # One extra dilated block per requested dilation, at kernel 15, for a longer receptive
        # field. The two streams differ only in their base kernel schedule.
        extra_dilations = tuple(int(x) for x in encoder_extra_dilations)
        extra_kernels = tuple(15 for _ in extra_dilations)
        encoder_dilations = (1, 2, 4) + extra_dilations
        self.target_encoder = CausalConvLstmEncoder(
            d_model=d_model,
            cnn_kernels=(3, 7, 11) + extra_kernels,
            cnn_dilations=encoder_dilations,
            lstm_layers=lstm_layers,
            lstm_dropout=dropout,
            conv_dropout=dropout,
        )
        self.source_encoder = CausalConvLstmEncoder(
            d_model=d_model,
            cnn_kernels=(3, 5, 11) + extra_kernels,
            cnn_dilations=encoder_dilations,
            lstm_layers=lstm_layers,
            lstm_dropout=self.source_dropout,
            conv_dropout=self.source_dropout,
        )

        self.prior_head = PriorHead(
            d_model=d_model,
            d_z=d_z,
            logvar_clamp=logvar_clamp,
            dropout=dropout,
            mu_scale=self.mu_scale,
        )
        self.lag_attn = LagCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            max_lag=max_lag,
            dropout=dropout,
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
            head_structured=self.head_structured_latent,
            num_heads=num_heads,
            d_head=d_head,
            delta_logvar_scale=self.delta_logvar_scale,
            posterior_logvar_mode=self.posterior_logvar_mode,
            # The posterior-fusion resolution, not the pathway one: see __init__ for why an
            # unset key means 0.0 here and `dropout` at the pathway sites.
            source_dropout=self.posterior_source_dropout,
        )

        # One horizon core, shared, so the residual decoder corrects the baseline inside the
        # baseline's own representation space.
        self.horizon_core = HorizonDecoderCore(
            d_hidden=decoder_hidden,
            horizon=horizon,
            kernel_size=horizon_kernel,
            depth=horizon_depth,
            film=horizon_film,
        )
        self.baseline_decoder = BaselineFutureDecoder(
            core=self.horizon_core,
            d_model=d_model,
            out_channels=c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
        )
        self.residual_decoder = ResidualFutureDecoder(
            core=self.horizon_core,
            d_model=d_model,
            d_z=d_z,
            out_channels=c_y,
            d_hidden=decoder_hidden,
            dropout=dropout,
            logvar_clamp=logvar_clamp,
        )

        self.te_analysis = TEAnalysisHead()

        self.causal_norm = bool(causal_norm)
        if self.causal_norm:
            self.n_causalized_norms = causalize_norms(self.target_encoder) + causalize_norms(
                self.source_encoder
            )
        else:
            self.n_causalized_norms = 0

        self.frozen_attn_proj = bool(freeze_unused_attn_proj and head_structured_latent)
        if self.frozen_attn_proj:
            for param in self.lag_attn.W_o.parameters():
                param.requires_grad_(False)

        if init_weights:
            initialization(self)
        # After the generic init, never before: it would overwrite them.
        self._zero_init_delta_heads()

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
        constructed gate rather than off the constructor arguments a second time means the
        availability pattern $m_{t,c} = \mathbb 1[t \ge \delta_c]$ cannot describe a guard the
        stream never received: the gate fills in a missing delay vector with zeros and a missing
        keep-index with the identity, and either substitution would leave the two out of step.

        This model keeps its **gated** residual seam (``post_residual_activation=True``, the
        original here) while the raw-signal siblings run the ungated one, so an unguarded model
        built through this method is bitwise the plain-``InputAdapter`` model it replaces --
        which is what ``tests/test_shared_code_stability.py`` pins.

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
            post_residual_activation=True,
        )

    @property
    def source_delay_steps(self) -> int:
        r"""The causal input delay $\delta$ the source stream is read with, in decimated steps.

        Zero when no reach budget is configured. **Every lag report must add this back**: an
        attention peak at lag $\ell$ refers to source content $\ell + \delta$ steps in the past,
        so a figure or summary that omits it understates the physiological delay -- by up to two
        minutes at the $120$ s budget, with nothing failing.

        The source channels are delayed individually, so no single $\delta$ describes them all.
        The maximum is reported, which makes a lag computed from it an upper bound.
        """
        return 0 if self.source_gate is None else self.source_gate.max_delay

    @staticmethod
    def _zero_linear(layer: nn.Linear) -> None:
        """Zero a linear layer's weight and, if present, its bias."""
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _zero_init_delta_heads(self) -> None:
        r"""Zero every head whose output is a delta on something else.

        This is what makes the model start honest. With the posterior's mean and log-variance
        deltas at zero, $q \equiv p$ and $K_t \equiv 0$: the model begins asserting that the
        source says nothing. With the residual decoder's mean head at zero, the full forecast
        equals the baseline. Any coupling the model later reports had to be learned against
        that null.

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
        self._zero_linear(self.residual_decoder.mean_head)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        r"""Sample $z = \mu + \sigma \epsilon$ with $\epsilon \sim N(0, I)$.

        The noise is a separate factor rather than being drawn from $N(\mu, \sigma^2)$ directly,
        which is what keeps $\mu$ and $\sigma$ on the gradient path.

        Args:
            mu: Distribution mean.
            logvar: Distribution log-variance.

        Returns:
            A sample of the same shape.
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _warmup_steps(self, seq_len: int) -> int:
        """Number of leading steps to ignore, clipped to the sequence length."""
        warmup = int(getattr(self, "warmup_period", 0) or 0)
        if warmup <= 0:
            return 0
        return min(seq_len, warmup)

    def _build_warmup_valid_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Return a ``(T,)`` bool mask, ``False`` across the warm-up prefix.

        The encoders need history before their states mean anything; the first steps are
        conditioned on almost nothing and would otherwise contribute noise to every loss.

        Args:
            seq_len: Sequence length $T$.
            device: Device to build the mask on.

        Returns:
            Bool tensor ``(T,)``, ``True`` where the step is usable.
        """
        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            mask[:warmup] = False
        return mask

    def _combined_lag_mask(
        self,
        seq_len: int,
        device: torch.device,
        lag_band_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""Intersect an ablation band mask with the causal lag-validity mask.

        The attention applies **exactly one** mask: whatever reaches its forward *replaces* the
        internally-built validity mask rather than intersecting with it. Passing a bare band
        mask straight through would therefore silently destroy the causal constraint
        $t - \ell \ge 0$ -- the model would quietly start attending to lags that do not exist,
        and nothing would raise. This helper closes that gap, broadcasts ``(L,) -> (T, L)``, and
        returns the result in lag order.

        **Dead anchors.** When the kept band excludes lag $0$, every causally valid lag at
        anchors $t < \min(\mathrm{band})$ is removed and the attention row is all $-\infty$.
        ``softmax`` degrades gracefully there; ``entmax15`` **raises**, because its support size
        is $0$ and it gathers at index $-1$. The causal mask alone can never produce such a row
        -- lag $0$ is valid everywhere -- so band masking is the first thing that can. Lag $0$ is
        forced back on at those anchors purely to keep the activation well-posed, and the
        resulting rows are then discarded by :meth:`_ablate_dead_anchors`.

        Args:
            seq_len: Sequence length $T$.
            device: Device on which to build the validity mask.
            lag_band_mask: Boolean keep-mask ``(L,)`` or ``(T, L)`` in lag order, ``True``
                keeps the lag. ``None`` disables band masking.

        Returns:
            ``(m_lag, dead)``: the combined ``(T, L)`` mask, and the ``(T,)`` mask of anchors
            with no surviving valid lag. Both ``None`` when ``lag_band_mask`` is ``None``, so
            the default path calls attention exactly as it would without this feature -- which
            is what makes the no-mask case bit-exact rather than merely equivalent.

        Raises:
            ValueError: If the mask is not 1-D or 2-D, or its axes do not match $(T, L)$.
        """
        if lag_band_mask is None:
            return None, None

        num_lags = int(self.lag_attn.L)
        band = lag_band_mask.to(device=device, dtype=torch.bool)
        if band.dim() == 1:
            if band.shape[0] != num_lags:
                raise ValueError(
                    f"lag_band_mask of shape {tuple(lag_band_mask.shape)} has lag axis "
                    f"{band.shape[0]}, expected L={num_lags}"
                )
            band = band.unsqueeze(0).expand(int(seq_len), num_lags)
        elif band.dim() == 2:
            if band.shape != (int(seq_len), num_lags):
                raise ValueError(
                    f"lag_band_mask of shape {tuple(lag_band_mask.shape)} is not (T, L) = "
                    f"({int(seq_len)}, {num_lags})"
                )
        else:
            raise ValueError(
                f"lag_band_mask must be 1-D (L,) or 2-D (T, L); got "
                f"{tuple(lag_band_mask.shape)}"
            )

        validity = self.lag_attn.build_lag_mask(int(seq_len), device=device)
        combined = validity & band
        dead = ~combined.any(dim=-1)
        if bool(dead.any()):
            combined = combined.clone()
            combined[dead, 0] = True  # lag 0 is always causally valid
        return combined, dead

    def _ablate_dead_anchors(
        self,
        attended: torch.Tensor,
        alpha: torch.Tensor,
        attended_heads: torch.Tensor,
        dead: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Zero the attention at anchors whose every valid lag was masked away.

        Reimposes, for both normalisers, exactly what ``softmax``'s all-$-\infty$ path produces:
        $\alpha = 0$, per-head summary $a^{(m)} = 0$, and fused $A = W_o(0)$ -- which is $W_o$'s
        *bias*, not necessarily zero. Without this the two normalisers would disagree at
        precisely the anchors an ablation experiment creates.

        Ablation, not renormalisation: the surviving lags are never rescaled to recover unit
        mass. Rescaling would report the same total coupling through fewer lags, which is the
        opposite of what an ablation is asking.

        Args:
            attended: Fused attended source ``(B, T, d_model)``.
            alpha: Attention weights ``(B, T, num_heads, L)`` in lag order.
            attended_heads: Per-head summaries ``(B, T, num_heads, d_head)``, pre-``W_o``.
            dead: ``(T,)`` mask of anchors with no surviving valid lag, or ``None``.

        Returns:
            The triple with the dead anchors overwritten.
        """
        if dead is None or not bool(dead.any()):
            return attended, alpha, attended_heads

        alpha = alpha.clone()
        alpha[:, dead] = 0.0
        attended_heads = attended_heads.clone()
        attended_heads[:, dead] = 0.0
        attended = attended.clone()
        bias = self.lag_attn.W_o.bias
        attended[:, dead, :] = 0.0 if bias is None else bias.to(dtype=attended.dtype)
        return attended, alpha, attended_heads

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        *,
        lag_band_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full pipeline.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            u_stream: Source stream ``(B, T, c_u)``.
            lag_band_mask: Optional boolean keep-mask over lags, ``(L,)`` or ``(T, L)`` in lag
                order. Intersected with the causal validity mask. ``None`` is a bit-exact
                no-op.

        Returns:
            The forward dict: the prior and posterior parameters, the latent, both history
            states, the attention and its weights, both forecasts, and the KL readouts.

        Note:
            Band masking is an **ablation, not a renormalisation**. A row over a partially
            masked set of valid lags still sums to $1$; a row whose every valid lag was masked
            collapses to $\alpha = 0$. That collapse happens at anchors inside the warm-up
            prefix, which never enter a forecast loss.
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

        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(h_y)

        m_lag, dead = self._combined_lag_mask(h_y.size(1), h_y.device, lag_band_mask)
        attended, alpha, attended_heads = self.lag_attn(h_y, h_u, m_lag)
        attended, alpha, attended_heads = self._ablate_dead_anchors(
            attended, alpha, attended_heads, dead
        )

        # Head-structured mode reads the per-head summaries directly, which is what makes the
        # per-head KL attributable; the flat path reads the fused projection.
        posterior_source = attended_heads if self.head_structured_latent else attended
        mu_post, logvar_post = self.posterior_head(
            h_y, posterior_source, mu_prior, raw_logvar_prior
        )
        z = self.reparameterize(mu_post, logvar_post)

        # Saturation diagnostics: a bound that is always active is a bound that is binding, and
        # a binding bound is a silently mis-set hyperparameter.
        with torch.no_grad():
            mu_prior_sat_frac = (mu_prior.abs() >= (0.99 * self.mu_scale)).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (0.99 * self.delta_mu_scale)
            ).float().mean()

        mu_base, logvar_base = self.baseline_decoder(decoder_state)
        delta_mu_src, logvar_full = self.residual_decoder(decoder_state, z)
        mu_full = mu_base + delta_mu_src

        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
        )
        kld_per_t, te_lag_map, kld_per_t_per_head = self.te_analysis(
            kld_btd, alpha, head_structured=self.head_structured_latent
        )

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": h_y,
            "source_state": h_u,
            "decoder_state": decoder_state,
            "attended_source": attended,
            "attended_source_heads": attended_heads,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "delta_mu_src": delta_mu_src,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "kld_per_t": kld_per_t,
            "kld_per_t_per_head": kld_per_t_per_head,
            "te_lag_map": te_lag_map,
            "warmup_mask": self._build_warmup_valid_mask(h_y.size(1), device=h_y.device),
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
            "kld_active_frac": self._kld_active_frac(kld_btd),
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
        """Run the encoders, the attention and the posterior, but not the decoders.

        The decoders are most of the compute and none of the latent, so analysis that only
        needs $z$ or $K_t$ should not pay for them.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            u_stream: Source stream ``(B, T, c_u)``.
            sample_z: Reparameterise when ``True``, else return the posterior mean as ``z``.
            lag_band_mask: Optional lag keep-mask; see :meth:`forward`.

        Returns:
            The encode dict: the prior and posterior parameters, the latent, both history
            states, the decoder conditioning state, and the attention.
        """
        # Gated exactly as in `forward`; see the comment there.
        target = torch.cat([y_st, y_ph], dim=-1)
        if self.target_gate is not None:
            target = self.target_gate(target)
        source = u_stream if self.source_gate is None else self.source_gate(u_stream)
        h_y = self.target_encoder(self.target_adapter(target))
        h_u = self.source_encoder(self.source_adapter(source))

        mu_prior, logvar_prior, decoder_state, raw_logvar_prior = self.prior_head(h_y)
        m_lag, dead = self._combined_lag_mask(h_y.size(1), h_y.device, lag_band_mask)
        attended, alpha, attended_heads = self.lag_attn(h_y, h_u, m_lag)
        attended, alpha, attended_heads = self._ablate_dead_anchors(
            attended, alpha, attended_heads, dead
        )

        posterior_source = attended_heads if self.head_structured_latent else attended
        mu_post, logvar_post = self.posterior_head(
            h_y, posterior_source, mu_prior, raw_logvar_prior
        )
        z = self.reparameterize(mu_post, logvar_post) if sample_z else mu_post

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z": z,
            "target_state": h_y,
            "source_state": h_u,
            "decoder_state": decoder_state,
            "attended_source": attended,
            "attended_source_heads": attended_heads,
            "attn_weights": alpha,
        }

    def kld_tensor(
        self,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
        mu_post: torch.Tensor,
        logvar_post: torch.Tensor,
    ) -> torch.Tensor:
        r"""Closed-form KL between two diagonal Gaussians, per step and per dimension.

        $$\mathrm{KL} = \tfrac{1}{2}\left[\log\sigma^{2,p} - \log\sigma^{2,q}
        + \frac{\sigma^{2,q} + (\mu^q - \mu^p)^2}{\sigma^{2,p}} - 1\right]$$

        Closed-form rather than sampled: this quantity is the model's output, not an
        intermediate, and a Monte-Carlo estimate would put variance straight into the number
        being reported.

        Returns the raw KL over the full sequence, unmasked. Masking is the caller's job and
        every caller wants a different window: the training term masks to ``kld_support``, the
        reported curve masks the warm-up prefix to ``NaN``, and the permutation control masks
        nothing. A ``mask_warmup`` flag here was carried over from the original and no call site
        ever set it.

        Args:
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Prior log-variance ``(B, T, d_z)``.
            mu_post: Posterior mean ``(B, T, d_z)``.
            logvar_post: Posterior log-variance ``(B, T, d_z)``.

        Returns:
            The per-step per-dimension KL ``(B, T, d_z)``.
        """
        return 0.5 * (
            logvar_prior
            - logvar_post
            + (logvar_post.exp() + (mu_post - mu_prior) ** 2) / logvar_prior.exp()
            - 1.0
        )

    def _kld_support_mask(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        r"""Build the ``(T,)`` time-support mask for the training KL.

        ``'full'`` masks only the warm-up prefix. ``'anchor'`` additionally masks the final
        $H_d$ steps: those anchors have no fully-observed forecast window, so they receive no
        supervised gradient from the reconstruction term. Left in, their KL is regularised
        toward the prior by $\beta$ with nothing pulling the other way, and the resulting
        collapse at the tail is easy to misread as a real drop in coupling.

        Args:
            seq_len: Sequence length $T$.
            device: Target device.
            dtype: Target floating dtype.

        Returns:
            A ``(T,)`` tensor of $1.0$ in support, $0.0$ outside.
        """
        mask = torch.ones(seq_len, device=device, dtype=dtype)
        warmup = self._warmup_steps(seq_len)
        if warmup > 0:
            mask[:warmup] = 0.0
        if self.kld_support == "anchor":
            horizon = int(self.horizon)
            if horizon > 0:
                mask[max(seq_len - horizon, 0) :] = 0.0
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
        """Aggregate the closed-form KL over the configured support.

        Args:
            mu_prior: Prior mean ``(B, T, d_z)``.
            logvar_prior: Prior log-variance ``(B, T, d_z)``.
            mu_post: Posterior mean ``(B, T, d_z)``.
            logvar_post: Posterior log-variance ``(B, T, d_z)``.
            reduce_mean: Return the support-mean rather than the sum.
            weight: Optional per-step validity weight ``(B, T)``.
            free_bits: Per-dimension per-step floor on the KL before masking. The default is a
                no-op, since the closed-form Gaussian KL is already non-negative.

        Returns:
            A scalar.
        """
        kld = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
        )
        if free_bits > 0.0:
            kld = kld.clamp(min=float(free_bits))

        batch, seq_len, d_z = kld.shape
        device, dtype = kld.device, kld.dtype

        time_mask = self._kld_support_mask(seq_len, device=device, dtype=dtype)
        full_mask = time_mask.unsqueeze(0).expand(batch, seq_len)
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
        r"""Fraction of latent dimensions carrying more than $\epsilon$ of mean KL.

        The headline diagnostic for posterior collapse. A model can post a healthy total KL
        while routing all of it through one dimension; this is what distinguishes that from a
        latent that is actually being used.

        Args:
            kld_btd: Per-step per-dimension raw KL ``(B, T, d_z)``.

        Returns:
            A scalar in $[0, 1]$, averaged over the batch and the configured KL support.
        """
        with torch.no_grad():
            support = self._kld_support_mask(kld_btd.size(1), device=kld_btd.device) > 0
            if not bool(support.any()):
                return torch.zeros((), device=kld_btd.device, dtype=kld_btd.dtype)
            kld_dim_mean = kld_btd[:, support, :].mean(dim=(0, 1))
            return (kld_dim_mean > _KLD_ACTIVE_EPS).to(kld_btd.dtype).mean()

    def measure_transfer_entropy(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        reduce_mean: bool = False,
    ) -> torch.Tensor:
        r"""Estimate the transfer-entropy surrogate $\mathrm{KL}(q \,\|\, p)$ over the support.

        Both return modes share one time support, deliberately. The scalar routes through
        :meth:`_kld_loss`, which honours ``kld_support``; the per-step tensor is masked to the
        same support with ``NaN`` outside it. Were they allowed to differ, an ``'anchor'``
        model's plotted curve would show a KL spike across the untrained final-$H_d$ tail that
        the reported scalar does not contain, and the two would silently disagree.

        Args:
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            u_stream: Source stream ``(B, T, c_u)``.
            reduce_mean: Return the support-mean scalar rather than the per-step tensor.

        Returns:
            A scalar, or the per-step per-dimension KL ``(B, T, d_z)`` with ``NaN`` outside the
            support.
        """
        # Eval mode is needed (dropout would add noise to a measurement), but it is restored
        # afterwards: this is routinely called from a plotting callback mid-training, and a
        # method that silently left the module in eval would disable dropout for the rest of
        # the run -- training a differently-regularised model than the config asked for, with
        # no error and no log line.
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                encoded = self.encode_only(y_st, y_ph, u_stream, sample_z=True)
                if reduce_mean:
                    return self._kld_loss(
                        mu_prior=encoded["mu_prior"],
                        logvar_prior=encoded["logvar_prior"],
                        mu_post=encoded["mu_post"],
                        logvar_post=encoded["logvar_post"],
                        reduce_mean=True,
                    )
                kld = self.kld_tensor(
                    mu_prior=encoded["mu_prior"],
                    logvar_prior=encoded["logvar_prior"],
                    mu_post=encoded["mu_post"],
                    logvar_post=encoded["logvar_post"],
                )
                support = self._kld_support_mask(kld.size(1), device=kld.device) > 0
                kld = kld.clone()
                kld[:, ~support, :] = float("nan")
                return kld
        finally:
            self.train(was_training)

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
    ) -> Dict[str, Any]:
        r"""Compute the training objective.

        $$L = \lambda_{full} L_{feat} + \lambda_{base} L_{base} + \beta L_{KL}
        + \lambda_{lag} L_{smooth}$$

        * $L_{feat}$ -- reconstruction of the future target trajectory from the *full*
          forecast, at valid anchors $t \in [warmup, T - H_d)$.
        * $L_{base}$ -- the same, from the baseline forecast. This term is what stops the model
          cheating: without it the baseline could be left weak so that target-explainable
          variance gets pushed through the latent, inflating $K_t$ with information the source
          never supplied.
        * $L_{KL}$ -- the KL over its own support, which is deliberately independent of the
          feature window.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            y_st: Target scattering features ``(B, T, 43)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            weight: Optional per-step validity weight ``(B, T)``.
            compute_kld_loss: Include the KL term.
            beta: Weight on the KL term.
            lambda_full: Weight on the full reconstruction.
            lambda_base: Weight on the baseline reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            sigma_obs: Under ``'gaussian_nll'``, either a positive scalar observation noise or
                the string ``'learned'`` to use the decoders' own log-variance heads.
            free_bits: Per-dimension per-step floor on the KL.
            detach_baseline_in_full: Stop-gradient the baseline inside the full term, so
                $L_{feat}$ trains only the residual and source path while $L_{base}$ alone
                shapes the baseline.
            lambda_lag: Weight on a smoothness penalty over the lag embeddings, encouraging a
                physiologically plausible (smooth) lag profile rather than a spiky one.

        Returns:
            The loss dict: the three terms, the total, and the reporting keys ``kld_train``
            (the optimised KL) and ``kld_raw`` (the un-floored KL over the same support,
            detached). ``kld_train >= kld_raw`` always holds, because free-bits clamps each
            per-dimension KL upward before masking.

            The values are tensors except ``likelihood``, which echoes the string it was
            given -- hence the loose annotation. A caller that forwards this dict straight to a
            metric logger should drop that key rather than let it be coerced.

        Raises:
            ValueError: If ``likelihood`` is unknown, or ``sigma_obs`` is an unusable string or
                a non-positive scalar.
        """
        validate_choice(likelihood, _LIKELIHOOD_CHOICES, "likelihood")

        target = torch.cat([y_st, y_ph], dim=-1)
        mu_base = forward_outputs["mu_base"]
        if detach_baseline_in_full:
            mu_full = mu_base.detach() + forward_outputs["delta_mu_src"]
        else:
            mu_full = forward_outputs["mu_full"]

        batch, seq_len, horizon, channels = mu_full.shape
        valid_steps = seq_len - horizon
        device, dtype = target.device, target.dtype

        # The forecast target for anchor t is Y[t+1 : t+1+H_d]. unfold builds every anchor's
        # window as a view rather than materialising B*T copies.
        shifted = target[:, 1:, :]
        future = shifted.unfold(dimension=1, size=horizon, step=1)
        future = future.permute(0, 1, 3, 2).contiguous()

        mu_full_valid = mu_full[:, :valid_steps, :, :]
        mu_base_valid = mu_base[:, :valid_steps, :, :]

        warmup = self._warmup_steps(seq_len)
        warmup_t = torch.zeros(valid_steps, dtype=dtype, device=device)
        if warmup < valid_steps:
            warmup_t[warmup:] = 1.0

        if weight is not None:
            # An entry counts only if both its anchor and its forecast target are valid.
            step_weight = weight.to(device=device, dtype=dtype)
            anchor_weight = step_weight[:, :valid_steps]
            target_weight = step_weight[:, 1:].unfold(dimension=1, size=horizon, step=1)
            mask_feat = (
                warmup_t[None, :, None, None]
                * anchor_weight[:, :, None, None]
                * target_weight[:, :, :, None]
            )
        else:
            mask_feat = warmup_t[None, :, None, None].expand(batch, valid_steps, horizon, 1)

        # Count effective entries, including the channel axis, so the scale matches a mean over
        # (B, T_valid, H_d, C) rather than drifting with the mask density.
        denom = (mask_feat.sum() * float(channels)).clamp_min(1.0)

        diff_full = (mu_full_valid - future) ** 2
        diff_base = (mu_base_valid - future) ** 2

        logvar_full_valid = forward_outputs["logvar_full"][:, :valid_steps, :, :]
        logvar_base_valid = forward_outputs["logvar_base"][:, :valid_steps, :, :]

        if likelihood == "mse":
            per_elem_full = diff_full
            per_elem_base = diff_base
        else:
            if isinstance(sigma_obs, str):
                if sigma_obs != "learned":
                    raise ValueError(f"sigma_obs string must be 'learned', got {sigma_obs!r}")
                logvar_full_obs = logvar_full_valid
                logvar_base_obs = logvar_base_valid
            else:
                sigma_obs_value = float(sigma_obs)
                if sigma_obs_value <= 0.0:
                    raise ValueError(
                        f"sigma_obs scalar must be positive, got {sigma_obs_value}"
                    )
                logvar_scalar = math.log(sigma_obs_value**2)
                logvar_full_obs = torch.full_like(diff_full, logvar_scalar)
                logvar_base_obs = torch.full_like(diff_base, logvar_scalar)
            # Per-element Gaussian NLL in nats, dropping the constant term.
            per_elem_full = 0.5 * diff_full * torch.exp(-logvar_full_obs) + 0.5 * logvar_full_obs
            per_elem_base = 0.5 * diff_base * torch.exp(-logvar_base_obs) + 0.5 * logvar_base_obs

        feat_loss = (per_elem_full * mask_feat).sum() / denom
        base_loss = (per_elem_base * mask_feat).sum() / denom

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
            kld_loss = torch.zeros((), device=device, dtype=dtype)
            kld_raw = torch.zeros((), device=device, dtype=dtype)

        if lambda_lag > 0.0:
            lag_embeddings = self.lag_attn.lag_embeddings
            lag_diff = lag_embeddings[1:] - lag_embeddings[:-1]
            lag_smoothness = (lag_diff**2).mean()
        else:
            lag_smoothness = torch.zeros((), device=device, dtype=dtype)

        total_loss = (
            lambda_full * feat_loss
            + lambda_base * base_loss
            + beta * kld_loss
            + lambda_lag * lag_smoothness
        )

        # Collapse diagnostics over the same mask-weighted support the losses use, so they stay
        # inside the head's own bound band rather than scaling with the channel count.
        mean_logvar_full = (logvar_full_valid * mask_feat).sum() / denom
        mean_logvar_base = (logvar_base_valid * mask_feat).sum() / denom

        return {
            "feat_loss": feat_loss,
            "base_loss": base_loss,
            "kld_loss": kld_loss,
            "total_loss": total_loss,
            "beta": torch.tensor(float(beta), device=device, dtype=dtype),
            "likelihood": likelihood,
            "mean_logvar_full": mean_logvar_full,
            "mean_logvar_base": mean_logvar_base,
            "lag_smoothness": lag_smoothness,
            "kld_raw": kld_raw,
            "kld_train": kld_loss,
            "kld_active_frac": forward_outputs.get(
                "kld_active_frac", torch.zeros((), device=device, dtype=dtype)
            ),
        }
