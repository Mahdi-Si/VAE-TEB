r"""The composed model: two target-domain mixins, one architecture base, one anchored forward.

:class:`SeqVaeLagResidualTrfCfs` is a composition, and the order of its bases is load-bearing.

* :class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs` owns the input warm-up mask,
  the channel alignment, the tiled anchor set and the geometry refusals.
* :class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` owns the
  decoder's width, the gathered one-sided target block, the scored clock and the channel weights.
* :class:`~teb_vae.lag_slot_transformer_cfs.nets.core.LagResidualCore` builds the modules.

The mixins come first so their construction hooks win method resolution over the base's. What this
class then overrides against **both** of them is exactly three members, and each has a reason the
inherited version cannot be used:

``forward``
    The inherited one poses a query from the prior, calls a lag cross-attention and fuses the
    result through a head-structured posterior. This architecture has none of those, so the forward
    is written here rather than adapted.

``build_lag_mask``
    The inherited one reads the lag count off the attention module. There is none, and the count is
    a property of the configuration.

``_prior_clock``
    The inherited one encodes a stream of exact zeros through the source pathway and hands the
    result to the prior. This architecture's clock is an explicit function of stored position,
    built once at construction, and it is read by the proposal head as well as by the prior.

**The constructor writes out its own keyword schema in full**, and cannot delegate it. The
experiment driver builds a run's kwargs by sweeping ``inspect.signature`` on the model class, so a
``**kwargs`` signature would forward a handful of keys and silently build an all-defaults model. The
same sweep is why the keywords this architecture *refuses* appear in the signature at all: a key the
signature does not mention is dropped without a word, and a run configured for attention over a
model with none would train quietly to completion.

**The forward is anchor-indexed from the latent onward.** Every latent tensor's second axis is a
position in the decoded anchor set, not a stored step, and the two are different lengths and
different orders. A dense export must map one to the other explicitly; feeding an anchor-indexed
tensor to a reader that assumes a time axis is the one mistake here whose symptom is a plausible
number rather than an exception, which is why the divergence keys are named for the axis they carry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_slot_transformer_cfs.nets.core import (
    REFUSED_KEYWORDS,
    LagResidualCore,
    refuse_incompatible_keywords,
)
from teb_vae.lag_slot_transformer_cfs.nets.objective import compute_residual_objective
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    bound_update,
    cancellation_from_parts,
    residual_kl,
    residual_parameters,
)
from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import lag_validity

#: This model's identity, stamped into every checkpoint. A model kind rather than a version of the
#: lag-attentive one: matching latent and decoder dimensions do not make the old source fusion
#: semantically compatible, and a checkpoint that could be loaded into the wrong architecture would
#: report one model's numbers under the other's name.
MODEL_KIND = "fhr_lag_residual_cfs_v1"

#: What the composing constructor removes from its own ``locals()`` before forwarding the rest to
#: the base: the mixins' own keywords, the refused ones, and the two implicit entries.
FORWARDED_EXCLUSIONS_HERE: Tuple[str, ...] = FORWARDED_EXCLUSIONS + tuple(REFUSED_KEYWORDS)


class SeqVaeLagResidualTrfCfs(
    CausalWarmupInputs, CausalFeatureForecastTarget, LagResidualCore
):
    r"""Single-latent lag-residual conv-Transformer forecaster over one-sided coefficients.

    One $d_z$-dimensional latent defined by a target-only prior; uterine activity makes a bounded
    residual correction to the parameters of a second Gaussian over the same space; one shared
    decoder is invoked on a paired sample from each.

    Attributes:
        MODEL_KIND: This architecture's identity, for the checkpoint contract.
    """

    #: Stamped into checkpoints and read by the evaluation binding's class check.
    MODEL_KIND: str = MODEL_KIND

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 64,
        horizon: int = 10,
        raw_per_step: int = 16,
        warmup_period: int = 134,
        c_y: int = 80,
        c_u: int = 46,
        use_up_st: bool = True,
        max_lag: int = 90,
        dropout: float = 0.1,
        decoder_hidden: int = 256,
        horizon_depth: int = 4,
        horizon_kernel: int = 3,
        horizon_film: bool = True,
        horizon_attention_blocks: int = 2,
        horizon_embed_std: float = 0.8,
        head_init_calibration: bool = True,
        encoder_conv_kernels: Sequence[int] = (5, 9),
        encoder_conv_dilations: Sequence[int] = (1, 2),
        encoder_num_heads: int = 4,
        encoder_d_ff: int = 512,
        target_attention_blocks: int = 6,
        logvar_clamp: Tuple[float, float] = (-5.0, 3.0),
        mu_scale: float = 5.0,
        coverage_floor: float = 0.9,
        persistence_residual: bool = True,
        horizon_weight_halflife_steps: Optional[float] = None,
        residual_mu_scale: float = 3.0,
        residual_logsigma_scale: float = 1.0,
        lag_embed_dim: int = 8,
        proposal_hidden: Optional[int] = None,
        mean_only_residual: bool = False,
        source_scalar_lift: bool = False,
        source_disabled: bool = False,
        source_values_withheld: bool = False,
        source_stem: str = "pointwise",
        lag_fusion: str = "local",
        lag_attention_heads: int = 4,
        lag_scale: Optional[float] = None,
        anchor_chunk: Optional[int] = None,
        lag_chunk: Optional[int] = None,
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
        target_align_delays: Optional[Sequence[int]] = None,
        source_align_delays: Optional[Sequence[int]] = None,
        anchor_stride: int = 1,
        lag_floor: int = 0,
        target_weight_st: float = 1.0,
        target_weight_ph: float = 1.0,
        target_novelty_frac: Optional[Sequence[float]] = None,
        target_forecast_shift: Optional[Sequence[int]] = None,
        init_weights: bool = True,
        # ----------------------------------------------------------------------------------
        # Refused. Present in the signature and nowhere else, because the experiment driver
        # forwards a configuration key only when the constructor names it -- so a key omitted
        # here would be dropped in silence and the run would describe machinery it never built.
        # Every one of them raises through ``refuse_incompatible_keywords``, naming the reason.
        # ----------------------------------------------------------------------------------
        num_heads: Optional[int] = None,
        d_head: Optional[int] = None,
        source_attention_blocks: Optional[int] = None,
        source_attention_window: Optional[int] = None,
        source_dropout: Optional[float] = None,
        lag_kv_source: Optional[str] = None,
        use_entmax: Optional[bool] = None,
        lag_bias_init: Optional[str] = None,
        alibi_slope_scale: Optional[float] = None,
        attention_grad_checkpoint: Optional[bool] = None,
        query_uses_logvar: Optional[bool] = None,
        posterior_logvar_mode: Optional[str] = None,
        delta_mu_scale: Optional[float] = None,
        delta_logvar_scale: Optional[float] = None,
        a_head_gain: Optional[float] = None,
        base_decode: Optional[str] = None,
        prior_availability_input: Optional[bool] = None,
    ) -> None:
        r"""Initialize the model.

        Args:
            sequence_length: Stored steps $T$ after trimming.
            d_model: Encoder and conditioning-state width, which must be even for the clock.
            d_z: Latent width. Unconstrained by any head count, because nothing partitions it.
            horizon: Future steps per forecast $H$.
            raw_per_step: Raw samples per stored step, carried for the geometry alone.
            warmup_period: The anchor floor $F$.
            c_y: Declared target channels.
            c_u: Declared source channels.
            use_up_st: Whether the source stream carries its first stored block.
            max_lag: Furthest candidate lag, so $L = \texttt{max\_lag} + 1$.
            dropout: Dropout in the target adapter, target encoder and prior heads.
            decoder_hidden: Shared decoder hidden width.
            horizon_depth: Dilated blocks in the horizon core.
            horizon_kernel: Horizon convolution kernel width.
            horizon_film: Whether the latent modulates each horizon block.
            horizon_attention_blocks: Attention blocks over the generated forecast tokens.
            horizon_embed_std: Spread the horizon-step embedding is re-seeded at.
            head_init_calibration: Place the decoder and prior scale on the trivial predictor at
                initialisation. For a freshly trained model only.
            encoder_conv_kernels: Kernel width per target stem block.
            encoder_conv_dilations: Dilation per target stem block.
            encoder_num_heads: Attention heads inside the target encoder.
            encoder_d_ff: Feed-forward width inside the target encoder.
            target_attention_blocks: Causal Transformer blocks in the target encoder.
            logvar_clamp: The prior's log-variance bound and the observation log-variance bound.
            mu_scale: Bound on the prior mean.
            coverage_floor: Minimum valid fraction of an anchor's forecast window.
            persistence_residual: Whether the decoder mean carries the anchor's own target vector.
            horizon_weight_halflife_steps: Half-life of the horizon weighting, or ``None``.
            residual_mu_scale: $a_{\max}$, the mean bound in **prior standard deviations**. Not
                the same quantity as the lag-attentive model's raw-unit bound, which is why that
                keyword is refused rather than reused.
            residual_logsigma_scale: $b_{\max}$, the bound on the log-standard-deviation residual.
            lag_embed_dim: Width of the lag embedding.
            proposal_hidden: Body width of the proposal head, or ``None`` for ``d_model``.
            mean_only_residual: Build no scale-proposal parameters at all.
            source_scalar_lift: Widen each source coefficient's own representation.
            source_disabled: Build no source pathway at all, giving the target-only arm. The full
                distribution is then the prior, the divergence is exactly zero, and the two decoded
                forecasts are bitwise identical -- so this is a target-only forecaster scored
                through the same objective, the same decoder and the same reduction as the joint
                candidate it is the baseline for.
            source_values_withheld: The capacity control. Build the source pathway whole and emit
                an exact zero in place of every source coefficient, so a head of the candidate's
                trainable capacity sees lag identity and the availability announcement and no
                value. A gain on this arm is evidence about extra nonlinear target capacity rather
                than about the source, which is the confound the arm exists to measure.
            source_stem: ``'pointwise'`` for the recommended per-coefficient representation, or
                ``'conv'`` for the comparator's bounded causal convolution stack. The second exists
                so that removing the source temporal convolution can be isolated from the four
                other things that changed alongside it.
            lag_fusion: ``'local'`` for the recommended explicit sum of one proposal per lag, or
                ``'attention'`` for the comparator's learned distribution over lags. Everything
                downstream is identical on both.
            lag_attention_heads: Attention heads on the comparator's fusion, which must divide
                ``d_model``. It splits the attention alone; nothing here partitions the latent.
            lag_scale: $c_L$, or ``None`` for $L^{-1/2}$. Refused under attention fusion, which
                performs no summation to scale.
            anchor_chunk: Anchors per proposal chunk, or ``None`` for all of them. A memory
                setting: it changes floating-point summation order and nothing else, so a run that
                changes it must record it.
            lag_chunk: Lags per proposal chunk, or ``None`` for all of them.
            target_keep_index: Surviving target channels.
            target_warmup_steps: $W'_c$ per surviving target channel.
            source_keep_index: Surviving source channels.
            source_warmup_steps: $W'_j$ per surviving source channel.
            target_align_delays: Per-survivor target shift onto a common reference clock.
            source_align_delays: The same for the source stream.
            anchor_stride: $S$, the spacing between decoded anchors.
            lag_floor: $F_u$, the earliest stored step a lag may read.
            target_weight_st: Relative reconstruction weight of the first stored target block.
            target_weight_ph: The same for the second.
            target_novelty_frac: Per-declared-channel novelty share, a readout only.
            target_forecast_shift: The forecast clock's signed re-indexing of the scored element.
            init_weights: Run the generic initialisation pass and the repairs that follow it.
            num_heads: Refused; see :data:`~teb_vae.lag_slot_transformer_cfs.nets.core.REFUSED_KEYWORDS`.
            d_head: Refused.
            source_attention_blocks: Refused.
            source_attention_window: Refused.
            source_dropout: Refused.
            lag_kv_source: Refused.
            use_entmax: Refused.
            lag_bias_init: Refused.
            alibi_slope_scale: Refused.
            attention_grad_checkpoint: Refused.
            query_uses_logvar: Refused.
            posterior_logvar_mode: Refused.
            delta_mu_scale: Refused; set ``residual_mu_scale``, whose units differ.
            delta_logvar_scale: Refused; set ``residual_logsigma_scale``.
            a_head_gain: Refused.
            base_decode: Refused; both branches are always sampled under one draw.
            prior_availability_input: Refused; the metadata clock is unconditional here.

        Raises:
            ValueError: On any refused keyword that carries a value; on a geometry the base
                refuses; or on a stride, floor or warm-up pairing the causal mixin refuses.
        """
        # Before anything else, and before the base builds a single module: a refused keyword
        # names a mechanism that does not exist, so there is nothing to build for it and no point
        # constructing a model that is about to be thrown away.
        refuse_incompatible_keywords(locals())

        # Captured before any local is added below, so the forwarded set is exactly this signature
        # minus the mixins' keywords and the refused ones. Written out as explicit pairs it would be
        # the same dict with one silent failure mode: a keyword added to the base and forgotten
        # here would be forwarded at its default with nothing raising.
        forwarded = {
            name: value
            for name, value in locals().items()
            if name not in FORWARDED_EXCLUSIONS_HERE
        }

        # Before the base constructor, which builds the adapters that read the warm-up vectors and
        # the source encoder that reads the combined source steps.
        self._set_causal_inputs(
            horizon=horizon,
            target_keep_index=target_keep_index,
            target_warmup_steps=target_warmup_steps,
            source_keep_index=source_keep_index,
            source_warmup_steps=source_warmup_steps,
            anchor_stride=anchor_stride,
            lag_floor=lag_floor,
            target_forecast_shift=target_forecast_shift,
        )
        self._set_channel_weights(
            target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
        )
        self._set_target_novelty(target_novelty_frac=target_novelty_frac)

        # The alignment shifts reach the base under ITS names, which is the one place in this
        # family where the channel delay does any work. They arrive under names of their own
        # because a run configures a *reference* and the resolver turns it into these vectors.
        super().__init__(
            **forwarded,
            target_delays=target_align_delays,
            source_delays=source_align_delays,
        )

        # After the base, which is what validates the geometry the anchor checks read, and what
        # resolves the gate the channel weights are positional over.
        self._validate_causal_geometry()
        self._register_channel_weights()

    # ------------------------------------------------------------------
    # The objective seam
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        target_features: torch.Tensor,
        *,
        weight: torch.Tensor,
        beta: float = 1.0,
        beta_prior: float = 0.0,
        lambda_full: float = 1.0,
        lambda_base: float = 1.0,
        likelihood: str = "gaussian_nll",
        free_bits: float = 0.0,
        lambda_ms: float = 0.0,
        lambda_deriv: float = 0.0,
        lambda_boundary: float = 0.0,
    ) -> Dict[str, Any]:
        r"""Score the forward against the target, under this package's own reduction.

        **This override is load-bearing and is the easiest member of this class to lose.** Without
        it, ``compute_loss`` resolves through the two mixins into the shared implementation, which
        divides a rank's own numerator by a rank's own denominator and floors that denominator at
        one. Both are behaviours this architecture rejects, and neither raises: the model would
        train, converge and report plausible nats against the wrong reduction.

        Two things happen here that the shared path also does, and they are done here because this
        method does not reach it. The validity signal is pooled onto the scored clock first, which
        is the identity object under the stored clock and the whole seam under any other; and the
        forecast target is gathered at the anchors the forward decoded, from the mixin that owns
        how this target domain is gathered.

        Args:
            forward_outputs: The dict :meth:`forward` returned.
            target_features: The target stream $(B, T, c_y)$, loader-normalized, in the declared
                channel order the keep-index is positional into.
            weight: Decimated validity signal $(B, T)$, as the loader delivered it.
            beta: Weight on the divergence.
            beta_prior: Weight on the prior scale rate.
            lambda_full: Weight on the full-forecast reconstruction.
            lambda_base: Weight on the base-forecast reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-coordinate floor on the divergence entering the loss.
            lambda_ms: Accepted and refused above zero; see below.
            lambda_deriv: The same.
            lambda_boundary: The same.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``.

        Raises:
            ValueError: If any shape-term weight is nonzero. The three terms read a forecast
                block's last axis as a trajectory -- pooled neighbourhoods, first differences, a
                boundary sample identified with the previous anchor's -- and here that axis counts
                channels, which have no order and no continuity with anything. They are accepted as
                keywords rather than dropped from the signature so that a configuration setting one
                fails at the first step instead of being silently ignored.
        """
        for name, value in (
            ("lambda_ms", lambda_ms),
            ("lambda_deriv", lambda_deriv),
            ("lambda_boundary", lambda_boundary),
        ):
            if float(value) != 0.0:
                raise ValueError(
                    f"{name}={value} was given, and this objective has no shape terms: they read "
                    f"the forecast block's last axis as a trajectory, and here it counts channels. "
                    f"Set it to 0.0, which is what every configuration of this target domain ships."
                )
        scored = self.scored_weight(weight)
        target = self._build_forecast_target(
            target_features, forward_outputs["anchor_index"]
        )
        return compute_residual_objective(
            forward_outputs,
            target,
            weight=scored,
            geometry=self.geometry,
            # The block's last axis counts surviving target channels, which is exactly what the
            # decoder emits, so the two cannot disagree.
            block_width=self.decoder_out_channels,
            coverage_floor=self.coverage_floor,
            logvar_clamp=self.logvar_clamp,
            # ``getattr`` rather than an attribute read: both are buffers where they exist and
            # absent otherwise, and ``None`` means the score skips the multiplication rather than
            # multiplying by ones.
            channel_weight=getattr(self, "target_channel_weight", None),
            horizon_weight=getattr(self, "horizon_weight", None),
            beta=beta,
            beta_prior=beta_prior,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            free_bits=free_bits,
        )

    # ------------------------------------------------------------------
    # The three overrides
    # ------------------------------------------------------------------
    def build_lag_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        r"""Index support per anchor and lag: $\mathbb 1[F_u \le t - \ell]$.

        Written here rather than inherited because the inherited version reads the lag count off
        an attention module this architecture does not build.

        **This is the index half of availability and not the whole of it.** A lag can be perfectly
        in range while the source channel it would read has not warmed up, and the two conditions
        fail at different anchors for different reasons. The per-channel half lives in the source
        encoder's own mask and is combined at the gather; keeping them apart is what lets the
        per-lag exposure readout say which of the two is limiting a bin.

        Args:
            seq_len: Sequence length $T$.
            device: Device to build the mask on.

        Returns:
            A boolean $(T, L)$ mask, ``True`` where the lagged stored step is readable.
        """
        steps = torch.arange(int(seq_len), device=device)[:, None]
        lags = torch.arange(self.n_lags, device=device)[None, :]
        source_step = steps - lags
        return (source_step >= int(self.lag_floor)) & (source_step >= 0)

    def _prior_clock(self, u_stream: torch.Tensor) -> torch.Tensor:
        r"""The metadata clock, broadcast over the batch.

        Overridden against the inherited version, which encodes a stream of exact zeros through the
        source pathway. That tensor carries no source *value* either, but it is a function of the
        source pathway's own parameters, and this architecture's source pathway holds no parameters
        at all on the recommended arm -- so there would be nothing to encode. The clock here is an
        explicit function of stored position, built once at construction.

        The forward does not call this method: it calls
        :meth:`~teb_vae.lag_slot_transformer_cfs.nets.core.LagResidualCore.conditioning_state`,
        which adds the projected clock to the target state once and hands the result to both heads.
        This exists so that a caller reaching for the inherited name gets this model's actual clock
        rather than an attribute error about a source encoder that was never built.

        Args:
            u_stream: The source stream, read for its shape, dtype and device only.

        Returns:
            The clock, $(1, T, d_{\mathrm{model}})$.
        """
        return self.metadata_clock[: u_stream.shape[1]].unsqueeze(0)

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        anchor_phase: Optional[Union[int, torch.Tensor]] = None,
        anchor_stride: Optional[int] = None,
        *,
        selector: Optional[torch.Tensor] = None,
        return_proposals: bool = False,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the pipeline and decode at a tiled anchor set.

        Eleven steps, in this order and for these reasons: the anchor set is built first because
        every later tensor is indexed by it; the persistence input is gathered before the gate
        because it is the target's own declared-order vector; the conditioning state is formed
        before the prior because the proposal head reads the same tensor; and the source encoding
        is computed once for the whole batch because it is pointwise and shares nothing with the
        anchor axis.

        Args:
            y_st: Target scattering features $(B, T, \cdot)$.
            y_ph: Target phase features $(B, T, \cdot)$, concatenated after ``y_st`` in the
                declared channel order the keep-index is positional into.
            u_stream: Source stream $(B, T, c_u)$ at the declared width.
            anchor_phase: $\varphi$ per sample; required once the resolved stride exceeds one.
            anchor_stride: $S$, or ``None`` for the model's configured stride.
            selector: $s_{t,\ell}$, broadcastable to $(B, A, L)$, externally set and never
                learned. ``None`` is the unintervened forward. This is the whole of the
                suppression interface: it is not an attention weight and carries no gradient.
            return_proposals: Retain and return the per-lag proposals and the per-channel source
                mask. Off by default because the proposal array is the largest tensor this
                architecture can hold, and a training step has no use for it.

        Returns:
            The anchor-indexed contract. Latent and forecast tensors carry an **anchor** axis, not
            a time axis:

            * ``anchor_index``, ``anchor_valid`` -- the decoded anchors $(B, A)$ and which of them
              are real. Padded slots repeat the row's last real anchor.
            * ``mu_prior``, ``logvar_prior``, ``raw_logvar_prior`` -- the target-only prior.
            * ``mu_post``, ``logvar_post`` -- the full, source-conditioned distribution. Named for
              the family's key, not for a posterior: neither branch observes a future label.
            * ``z_prior``, ``z_post`` -- the paired samples under one shared noise draw.
            * ``mu_base``, ``logvar_base``, ``mu_full``, ``logvar_full`` -- the two forecasts
              $(B, A, H, C_{\mathrm{keep}})$.
            * ``kld_per_anchor_dim``, ``kld_per_anchor`` -- the residual-form divergence, per
              coordinate and summed. Named for the axis they carry.
            * ``update_mean``, ``update_logsigma`` -- the bounded updates $a$ and $b$; the second
              is absent on the mean-only arm rather than a zero tensor.
            * ``raw_update_mean``, ``raw_update_logsigma`` -- the same before the limiter, so
              saturation is measurable rather than inferred.
            * ``cancellation_*`` -- the ratio and both of its parts, for each proposal channel.
              Absent under attention fusion, which sums no per-lag updates and therefore has
              nothing that can cancel.
            * ``lag_valid`` -- which anchor-lag pairs carry any available channel $(B, A, L)$.
            * ``target_state`` -- the target encoder's output over the stored grid $(B, T, d_h)$.
            * ``conditioning_state`` -- $h_t$ at the decoded anchors $(B, A, d_h)$.
            * ``persistence`` -- the anchor's own target vector, only where the decoder was built
              with the residual.
            * ``source_channel_mask`` -- under ``return_proposals``, on every arm with a source
              pathway. ``mean_proposals`` and ``scale_proposals`` join it only where the arm
              produces per-lag updates at all, which the local fusion does and the attention
              fusion does not.

            There is deliberately no ``attn_weights``, no ``attended_source_heads``, no
            ``source_kl_lag_map`` and no ``kld_per_t_per_head``: this architecture computes none of
            them, and a fabricated tensor under one of those names would let an evaluator report a
            per-lag attribution that does not exist.
        """
        anchor_index, anchor_valid = self._build_anchor_index(
            batch=int(y_st.shape[0]),
            device=y_st.device,
            anchor_phase=anchor_phase,
            anchor_stride=anchor_stride,
        )

        target = torch.cat([y_st, y_ph], dim=-1)
        # Before the gate: the gather is the target mixin's own and reads the declared channel
        # order. Target-only, so it opens no source bypass, and both decoder calls receive the
        # identical tensor so the base-minus-full gap stays a pure source readout.
        persistence = (
            self._anchor_target_values(target, anchor_index)
            if self.persistence_residual
            else None
        )
        if self.target_gate is not None:
            target = self.target_gate(target)
        source = u_stream if self.source_gate is None else self.source_gate(u_stream)

        target_state = self.target_encoder(self.target_adapter(target))
        # One clock, added once, read by both heads below.
        conditioning = self.conditioning_state(target_state)

        # Gathered before the prior rather than after it: the prior is needed at the decoded
        # anchors only, and running its two multilayer perceptrons over the whole stored grid would
        # compute several times as many rows as the objective scores.
        gather_state = anchor_index[:, :, None].expand(-1, -1, self.d_model)
        anchor_conditioning = conditioning.gather(1, gather_state)
        mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(anchor_conditioning)

        # Pointwise, once for the batch: the encoding shares nothing with the anchor axis, so
        # recomputing it per chunk would be the same tensor built many times. On the target-only
        # arm there is no encoder to run and no head to run it through, and the update is the zero
        # a sum over no proposals gives -- written out rather than reached by evaluating modules
        # that were never built.
        if self.source_disabled:
            proposals = self._absent_proposals(anchor_conditioning)
        else:
            encoded, channel_mask = self.source_encoder(source)
            accumulate = (
                self._attend_proposals
                if self.lag_fusion == "attention"
                else self._accumulate_proposals
            )
            proposals = accumulate(
                anchor_index=anchor_index,
                anchor_conditioning=anchor_conditioning,
                encoded=encoded,
                channel_mask=channel_mask,
                selector=selector,
                return_proposals=return_proposals,
            )

        raw_a = self.lag_scale * proposals["mean_total"]
        a = bound_update(raw_a, self.residual_mu_scale)
        raw_b = (
            None
            if proposals["scale_total"] is None
            else self.lag_scale * proposals["scale_total"]
        )
        b = None if raw_b is None else bound_update(raw_b, self.residual_logsigma_scale)

        mu_post, logvar_post = residual_parameters(mu_prior, logvar_prior, a, b)
        z_prior, z_post = self.reparameterize_shared(
            mu_prior, logvar_prior, mu_post, logvar_post
        )

        # One decoder, invoked twice, with the identical persistence tensor and weights. Nothing
        # source-derived reaches it except through the latent it is handed.
        mu_base, logvar_base = self.decoder(z_prior, persistence=persistence)
        mu_full, logvar_full = self.decoder(z_post, persistence=persistence)

        kld_per_anchor_dim = residual_kl(a, b)

        outputs: Dict[str, torch.Tensor] = {
            "anchor_index": anchor_index,
            "anchor_valid": anchor_valid,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z_prior": z_prior,
            "z_post": z_post,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "kld_per_anchor_dim": kld_per_anchor_dim,
            "kld_per_anchor": kld_per_anchor_dim.sum(dim=-1),
            "update_mean": a,
            "raw_update_mean": raw_a,
            "lag_valid": proposals["lag_valid"],
            "target_state": target_state,
            "conditioning_state": anchor_conditioning,
        }
        if b is not None and raw_b is not None:
            # Absent rather than zero on the mean-only arm: a zero tensor here would let a reader
            # report a scale update the model cannot make.
            outputs["update_logsigma"] = b
            outputs["raw_update_logsigma"] = raw_b
        outputs.update(self.saturation_fractions(mu_prior, a, b))

        for channel in ("mean", "scale"):
            total = proposals[f"{channel}_total"]
            norm_sum = proposals[f"{channel}_norm_sum"]
            if total is None or norm_sum is None:
                continue
            ratio, numerator, denominator = cancellation_from_parts(total, norm_sum)
            outputs[f"cancellation_ratio_{channel}"] = ratio
            outputs[f"cancellation_numerator_{channel}"] = numerator
            outputs[f"cancellation_denominator_{channel}"] = denominator

        if persistence is not None:
            outputs["persistence"] = persistence
        if return_proposals:
            # Each key appears only where the arm actually produced it, for the reason every
            # optional key here is absent rather than zero-filled: a caller reaching for the
            # proposals of a model that has no source pathway, or for the per-lag updates of a
            # normalised aggregation that has none, should find nothing rather than an array of
            # zeros it could mistake for a measurement. The gathered channel mask is the one the
            # exposure readout needs and every arm with a source pathway has it, whichever
            # representation formed the window.
            if proposals["channel_mask"] is not None:
                outputs["source_channel_mask"] = proposals["channel_mask"]
            if proposals["mean_proposals"] is not None:
                outputs["mean_proposals"] = proposals["mean_proposals"]
            if proposals["scale_proposals"] is not None:
                outputs["scale_proposals"] = proposals["scale_proposals"]
        return outputs

    # ------------------------------------------------------------------
    # The two proposal passes
    # ------------------------------------------------------------------
    def _absent_proposals(self, anchor_conditioning: torch.Tensor) -> Dict[str, Any]:
        r"""What the accumulation returns when there is no source pathway to accumulate.

        A sum over no proposals is zero by definition, so the update is zero, the full distribution
        is the prior and the divergence is exactly zero. Written out rather than reached by
        evaluating a head that was never built.

        The per-lag norm sum is zero too, which makes the cancellation ratio zero over its own
        epsilon -- a ratio the readout will report as near-zero beside a denominator of zero, which
        is exactly the "nothing to cancel" reading it exists to distinguish.

        Args:
            anchor_conditioning: $h_t$ at the decoded anchors, read for its shape, dtype and device.

        Returns:
            The same keys :meth:`_accumulate_proposals` returns, every one of them empty.
        """
        batch, n_anchors = anchor_conditioning.shape[0], anchor_conditioning.shape[1]
        device, dtype = anchor_conditioning.device, anchor_conditioning.dtype
        zeros = torch.zeros(batch, n_anchors, self.d_z, device=device, dtype=dtype)
        return {
            "mean_total": zeros,
            # ``None`` rather than a zero tensor, matching the mean-only arm: it is what keeps the
            # scale keys out of the returned contract instead of reporting an update the model
            # cannot make.
            "scale_total": None,
            "mean_norm_sum": torch.zeros(batch, n_anchors, device=device, dtype=dtype),
            "scale_norm_sum": None,
            "lag_valid": torch.zeros(
                batch, n_anchors, self.n_lags, device=device, dtype=torch.bool
            ),
            "mean_proposals": None,
            "scale_proposals": None,
            "channel_mask": None,
        }

    def _accumulate_proposals(
        self,
        *,
        anchor_index: torch.Tensor,
        anchor_conditioning: torch.Tensor,
        encoded: torch.Tensor,
        channel_mask: torch.Tensor,
        selector: Optional[torch.Tensor],
        return_proposals: bool,
    ) -> Dict[str, Any]:
        r"""Evaluate every lag's proposal and reduce it, in chunks, without leaving the graph.

        The summed update is what the model needs and the per-lag array is what it would cost, so
        the two axes are walked in chunks and only the reductions are kept. Both reductions are
        additive over lags -- the sum itself, and the sum of per-lag norms the cancellation ratio
        needs -- so a chunked pass reports exactly what an unchunked one would, up to
        floating-point summation order.

        **Nothing is detached.** A detached chunk would train a model whose lags in that chunk
        never update, which no shape and no metric would reveal; the accumulation is an ordinary
        sum of graph-connected tensors and the gradient flows through every chunk.

        Chunk sizes change floating-point summation order and therefore the last digits of a
        result. That is a real effect rather than a rounding curiosity, which is why they are
        configuration and are recorded with a run.

        Args:
            anchor_index: The decoded anchors $(B, A)$.
            anchor_conditioning: $h_t$ at those anchors $(B, A, d_h)$.
            encoded: The pointwise source encoding $(B, T, C_U, W)$.
            channel_mask: Its availability $(B, T, C_U)$.
            selector: $s_{t,\ell}$ broadcastable to $(B, A, L)$, or ``None``.
            return_proposals: Retain the per-lag arrays and the gathered channel mask.

        Returns:
            The accumulated totals, the per-lag norm sums, the lag-validity indicator, and the
            retained arrays when they were asked for.
        """
        batch, n_anchors = anchor_conditioning.shape[0], anchor_conditioning.shape[1]
        device, dtype = anchor_conditioning.device, anchor_conditioning.dtype
        anchor_step = n_anchors if self.anchor_chunk is None else self.anchor_chunk
        lag_step = self.n_lags if self.lag_chunk is None else self.lag_chunk

        mean_total = torch.zeros(batch, n_anchors, self.d_z, device=device, dtype=dtype)
        scale_total = (
            None
            if self.mean_only_residual
            else torch.zeros(batch, n_anchors, self.d_z, device=device, dtype=dtype)
        )
        # Diagnostics, so they carry no gradient and are accumulated outside the graph.
        mean_norm_sum = torch.zeros(batch, n_anchors, device=device, dtype=dtype)
        scale_norm_sum = (
            None
            if self.mean_only_residual
            else torch.zeros(batch, n_anchors, device=device, dtype=dtype)
        )
        lag_valid = torch.zeros(
            batch, n_anchors, self.n_lags, device=device, dtype=torch.bool
        )
        mean_kept: List[List[torch.Tensor]] = []
        scale_kept: List[List[torch.Tensor]] = []
        mask_kept: List[List[torch.Tensor]] = []

        for start in range(0, n_anchors, anchor_step):
            stop = min(start + anchor_step, n_anchors)
            anchors = anchor_index[:, start:stop]
            state = anchor_conditioning[:, start:stop]
            mean_row: List[torch.Tensor] = []
            scale_row: List[torch.Tensor] = []
            mask_row: List[torch.Tensor] = []

            for lag_start in range(0, self.n_lags, lag_step):
                lag_stop = min(lag_start + lag_step, self.n_lags)
                # The encoder's own gather, because the two source representations this package
                # builds are read differently -- one carries a channel axis and one has mixed it
                # away -- and this loop must not be the place that knows which.
                window, window_mask = self.source_encoder.gather(
                    encoded,
                    channel_mask,
                    anchors,
                    n_lags=lag_stop - lag_start,
                    lag_floor=self.lag_floor,
                    lag_offset=lag_start,
                )
                valid = lag_validity(window_mask)
                lag_valid[:, start:stop, lag_start:lag_stop] = valid

                chunk_selector = (
                    None
                    if selector is None
                    else selector[:, start:stop, lag_start:lag_stop]
                )
                mean_chunk, scale_chunk = self.proposal_head(
                    state,
                    window,
                    lag_valid=valid,
                    selector=chunk_selector,
                    lag_index=torch.arange(lag_start, lag_stop, device=device),
                )

                mean_total[:, start:stop] = mean_total[:, start:stop] + mean_chunk.sum(dim=2)
                with torch.no_grad():
                    mean_norm_sum[:, start:stop] += mean_chunk.norm(dim=-1).sum(dim=2)
                if scale_chunk is not None and scale_total is not None:
                    scale_total[:, start:stop] = (
                        scale_total[:, start:stop] + scale_chunk.sum(dim=2)
                    )
                    if scale_norm_sum is not None:
                        with torch.no_grad():
                            scale_norm_sum[:, start:stop] += scale_chunk.norm(dim=-1).sum(dim=2)

                if return_proposals:
                    mean_row.append(mean_chunk)
                    mask_row.append(window_mask)
                    if scale_chunk is not None:
                        scale_row.append(scale_chunk)

            if return_proposals:
                mean_kept.append(mean_row)
                mask_kept.append(mask_row)
                if scale_row:
                    scale_kept.append(scale_row)

        result: Dict[str, Any] = {
            "mean_total": mean_total,
            "scale_total": scale_total,
            "mean_norm_sum": mean_norm_sum,
            "scale_norm_sum": scale_norm_sum,
            "lag_valid": lag_valid,
            "mean_proposals": None,
            "scale_proposals": None,
            "channel_mask": None,
        }
        if return_proposals:
            result["mean_proposals"] = _reassemble(mean_kept)
            result["channel_mask"] = _reassemble(mask_kept)
            if scale_kept:
                result["scale_proposals"] = _reassemble(scale_kept)
        return result


    def _attend_proposals(
        self,
        *,
        anchor_index: torch.Tensor,
        anchor_conditioning: torch.Tensor,
        encoded: torch.Tensor,
        channel_mask: torch.Tensor,
        selector: Optional[torch.Tensor],
        return_proposals: bool,
    ) -> Dict[str, Any]:
        r"""The comparator's fusion: one distribution over lags per anchor, in anchor chunks.

        The same signature and the same returned keys as the local accumulation, so the forward
        chooses between them and holds no case for either. What differs is what the returned totals
        **are**: there is no per-lag update to sum, so the totals are the fusion's own output and
        the per-lag arrays and the cancellation parts are absent rather than zero-filled.

        **Absent, not zero.** The cancellation ratio measures how much of a sum of per-lag updates
        survives the sum, and a normalised aggregation has no such sum: every weight is
        non-negative and they add to one, so nothing can cancel and a ratio reported here would be
        a constant dressed as a diagnostic. The suppression readout is the intervention both arms
        share, and it is the one the evaluation reports for this arm.

        **Only the anchor axis is chunked.** The softmax is normalised over the whole lag axis, so
        a lag chunk would renormalise within each chunk and compute a different model rather than
        the same one in a different order. The constructor refuses ``lag_chunk`` on this arm for
        that reason, and this loop therefore walks one axis where the local one walks two.

        Args:
            anchor_index: The decoded anchors $(B, A)$.
            anchor_conditioning: $h_t$ at those anchors $(B, A, d_h)$.
            encoded: The source encoding over the stored grid, in whatever shape its encoder emits.
            channel_mask: Its per-channel availability $(B, T, C_U)$.
            selector: $s_{t,\ell}$ broadcastable to $(B, A, L)$, or ``None``.
            return_proposals: Retain the gathered channel mask, which the exposure readout needs.
                There is no per-lag update array to retain.

        Returns:
            The same keys :meth:`_accumulate_proposals` returns, with the per-lag arrays and the
            per-lag norm sums empty.
        """
        batch, n_anchors = anchor_conditioning.shape[0], anchor_conditioning.shape[1]
        device, dtype = anchor_conditioning.device, anchor_conditioning.dtype
        anchor_step = n_anchors if self.anchor_chunk is None else self.anchor_chunk

        mean_total = torch.zeros(batch, n_anchors, self.d_z, device=device, dtype=dtype)
        scale_total = (
            None
            if self.mean_only_residual
            else torch.zeros(batch, n_anchors, self.d_z, device=device, dtype=dtype)
        )
        lag_valid = torch.zeros(
            batch, n_anchors, self.n_lags, device=device, dtype=torch.bool
        )
        mask_kept: List[List[torch.Tensor]] = []

        for start in range(0, n_anchors, anchor_step):
            stop = min(start + anchor_step, n_anchors)
            anchors = anchor_index[:, start:stop]
            state = anchor_conditioning[:, start:stop]
            window, window_mask = self.source_encoder.gather(
                encoded,
                channel_mask,
                anchors,
                n_lags=self.n_lags,
                lag_floor=self.lag_floor,
            )
            valid = lag_validity(window_mask)
            lag_valid[:, start:stop] = valid

            mean_chunk, scale_chunk = self.proposal_head(
                state,
                window,
                lag_valid=valid,
                selector=None if selector is None else selector[:, start:stop],
            )
            # Assignment rather than accumulation: an anchor's update is produced whole by one
            # normalised aggregation, so the chunk holds the answer for its anchors rather than a
            # partial sum over them.
            mean_total[:, start:stop] = mean_chunk
            if scale_chunk is not None and scale_total is not None:
                scale_total[:, start:stop] = scale_chunk
            if return_proposals:
                mask_kept.append([window_mask])

        return {
            "mean_total": mean_total,
            "scale_total": scale_total,
            # No sum of per-lag norms exists to compare a total against, so the cancellation
            # readout is absent on this arm rather than reported as a number it cannot mean.
            "mean_norm_sum": None,
            "scale_norm_sum": None,
            "lag_valid": lag_valid,
            "mean_proposals": None,
            "scale_proposals": None,
            "channel_mask": _reassemble(mask_kept) if return_proposals else None,
        }


def _reassemble(chunks: List[List[torch.Tensor]]) -> torch.Tensor:
    """Rebuild a full anchor-by-lag array from the chunk grid it was evaluated in.

    Args:
        chunks: Rows of anchor chunks, each holding that row's lag chunks in order.

    Returns:
        The concatenated array, anchors on axis one and lags on axis two.
    """
    return torch.cat([torch.cat(row, dim=2) for row in chunks], dim=1)


__all__ = ["MODEL_KIND", "SeqVaeLagResidualTrfCfs"]
