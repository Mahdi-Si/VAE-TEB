r"""The conv-Transformer causal-feature forecaster: two mixins, one architecture, one constructor.

:class:`SeqVaeLagAttnTrfCfs` is a composition. Both causal conv-Transformer encoders, the channel
gates, the availability-aware input adapters, the prior and posterior heads, the lag
cross-attention, the shared horizon decoder and the paired reparameterisation are
:class:`~teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`'s. The input warm-up mask,
the lag validity floor, the tiled anchor set and the forward that decodes at it are
:class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs`'s. The decoder's width, the
gathered one-sided target block, the budget-and-floor refusal and the eleven resolved readouts are
:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget`'s. No network
code is written here.

**Why two mixins and not two inheritances.** ``SeqVaeLagAttnCfs`` subclasses the conv-LSTM model,
while ``SeqVaeLagAttnTrfRws`` derives from ``nn.Module`` directly, so
``class X(SeqVaeLagAttnCfs, SeqVaeLagAttnTrfRws)`` linearises as
``X -> Cfs -> ... -> Rws -> TrfRws -> Module`` and runs the **conv-LSTM** constructor: a model that
builds, trains and reports, and is not this architecture. Every member of this target domain names
no encoder, so both cells compose the same two plain objects instead.

**The order of the bases is load-bearing.** The mixins come first, which is what makes the width
hook win method resolution over the architecture's ``raw_per_step`` one and the tiled forward win
over the dense one. Reversed, the decoder would be built at $R = 16$ and a $C_{\mathrm{keep}}$-wide
feature block scored against it. That failure is loud, but not where a reader would look for it:
``block_width`` would not catch it, since it feeds only the four log-variance diagnostics and no
shape check, while ``raw_sample_score`` computes $(\text{target} - \mu)^2$ on
$(B, A, H, C_{\mathrm{keep}})$ against $(B, A, H, 16)$, which is not broadcastable.

**The constructor is the one member of this class**, and only because the experiment driver builds a
run's kwargs by sweeping ``inspect.signature(MODEL_CLS.__init__)``. This architecture's keyword
schema is not the conv-LSTM cell's -- five keys absent, seven encoder keys added -- so the schema
has to be written where the class is. It holds no validation: what the four causal keywords need is
:meth:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs._set_causal_inputs` before the
base constructor and
:meth:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs._validate_causal_geometry` after
it, both shared with the conv-LSTM cell.

The unit consequence, restated where a reader will look for it: the reconstruction is summed over
$H \cdot C_{\mathrm{keep}} = 30 \times 98 = 2940$ coefficients at the shipped budget, against the
raw variant's $H \cdot R$ samples and against the two-sided feature variant's
$30 \times 78 = 2340$, so the nats are comparable to neither -- nor across warm-up budgets within
this model, since $C_{\mathrm{keep}}$ moves with the budget. The *horizon* now agrees with every
other cell of the grid; only $C_{\mathrm{keep}}$ separates this block from the two-sided one.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws


class SeqVaeLagAttnTrfCfs(
    CausalWarmupInputs, CausalFeatureForecastTarget, SeqVaeLagAttnTrfRws
):
    r"""Single-latent lag-attentive conv-Transformer VAE forecasting one-sided coefficients.

    Constructed exactly as :class:`SeqVaeLagAttnTrfRws` is -- same keywords, same defaults, same
    refusals -- except that ``target_delays`` and ``source_delays`` are gone and eight keywords of
    this target domain take their place -- the two warm-up vectors, the two alignment shifts, the
    anchor stride, the lag floor and the two block weights. Removing the two is the point: a warm-up is a leading
    *mask* and ``ChannelDelay`` is a *shift*, so a warm-up routed under a delay name would train a
    different model with every shape intact, and a checkpoint's ``model_kwargs`` would be ambiguous
    between two families under one key name.

    There is no ``decoder_out_channels`` keyword, as there is none on the architecture parent: the
    decoder width follows the target gate, so a run configured with a warm-up budget gets a decoder
    for the channels that budget kept and one configured without gets a decoder for all $c_y$ --
    and because no keyword records it, no second field can disagree with the gate.
    """

    def __init__(
        self,
        *,
        sequence_length: int = 300,
        d_model: int = 128,
        d_z: int = 48,
        horizon: int = 30,
        raw_per_step: int = 16,
        warmup_period: int = 134,
        c_y: int = 102,
        c_u: int = 51,
        use_up_st: bool = True,
        max_lag: int = 90,
        num_heads: int = 4,
        d_head: int = 32,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = False,
        horizon_attention_blocks: int = 0,
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
        posterior_logvar_mode: str = "residual",
        source_dropout: Optional[float] = None,
        lag_kv_source: str = "encoder",
        use_entmax: bool = False,
        attention_grad_checkpoint: bool = False,
        lag_bias_init: str = "normal",
        alibi_slope_scale: float = 1.0,
        query_uses_logvar: bool = False,
        prior_availability_input: bool = False,
        coverage_floor: float = 0.9,
        base_decode: str = "sample",
        persistence_residual: bool = False,
        horizon_weight_halflife_steps: Optional[float] = None,
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
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Every keyword the architecture parent takes is forwarded unchanged or renamed; only the
        nine below are
        this target domain's. The defaults that differ from the parent's -- ``warmup_period`` $134$,
        ``c_y`` $102$, ``c_u`` $51$ -- are this target domain's geometry rather than a preference,
        and a run that left them at the parent's values would be describing a dataset that does not
        exist. ``horizon`` is **no longer** among them: at $30$ it agrees with the architecture
        parent and with every other cell of the grid, so the forecast question is shared even though
        the block is not.

        Args:
            target_warmup_steps: $W'_c$ per **surviving** target channel, positional against
                ``target_keep_index``. ``None`` builds no warm-up mask, which is the ungated model.
            source_warmup_steps: The same for the source stream.
            target_align_delays: $d_c$ per **surviving** target channel, the shift that brings
                every one of them onto a common reference clock. Forwarded to the base as
                ``target_delays``, so it reaches ``ChannelDelay`` and the gate stops being a pure
                gather. ``None`` -- the default and the shipped setting -- builds a model bitwise
                identical to one constructed before the keyword existed.
            source_align_delays: The same for the source stream.
            anchor_stride: $S$, the spacing between decoded anchors, in $[1, H]$. Defaults to $1$
                -- the dense range every sibling decodes, and the inert value -- so a model
                constructed without an opinion behaves like the rest of the family. The tiling is a
                configuration decision, and the shipped configuration states it.
            lag_floor: $F_u$, the earliest source step lag attention may read. Ships at $0$,
                where the lag mask is bitwise the architecture parent's.
            target_weight_st: Relative reconstruction weight of the first stored target block,
                the scattering coefficients. Both weights default to $1.0$, where the objective is
                bitwise the uniform one.
            target_weight_ph: The same for the second stored block, the phase-harmonic
                coefficients. The pair is renormalised to leave the block scale unchanged, so what
                the configuration states is a **ratio**: $(1.0, 0.1)$ and $(10.0, 1.0)$ describe
                the same objective, agreeing to float32 rounding rather than bitwise.
            target_novelty_frac: The share of each **declared** target channel's coefficient drawn
                from raw samples the anchor has not seen, as the shards record it. A readout alone:
                it ranks the kept channels into three groups that ``pred_gap`` is reported split by,
                and changes no width, no mask and no parameter. Gathered through
                ``target_keep_index`` rather than taken per survivor, so a model built with no gate
                still receives a vector of the right width. ``None`` -- the ungated arm and every
                unit construction -- reports the split over the declared channel order instead,
                which is a partition of the axis and not a measurement.

        Raises:
            ValueError: If ``anchor_stride`` is outside $[1, H]$ or leaves a phase with no anchor;
                if ``lag_floor`` is negative; if a warm-up vector arrives without its keep-index;
                or if ``warmup_period`` is below the floor the kept channels require. Everything
                else is the architecture parent's own validation, including the five encoder
                refusals a conv-LSTM cell has no analogue of.
        """
        # Captured before anything else runs, so the forwarded set is exactly this signature minus
        # the four keywords the mixin owns. Written out as forty explicit `name=name` pairs it would
        # be the same dict with one silent failure mode: a keyword added to the architecture parent
        # and forgotten here would be forwarded at its default with nothing raising.
        forwarded = {
            name: value
            for name, value in locals().items()
            if name not in FORWARDED_EXCLUSIONS
        }

        # Before the base constructor, which builds the input adapters that read the two vectors.
        self._set_causal_inputs(
            horizon=horizon,
            target_keep_index=target_keep_index,
            target_warmup_steps=target_warmup_steps,
            source_keep_index=source_keep_index,
            source_warmup_steps=source_warmup_steps,
            anchor_stride=anchor_stride,
            lag_floor=lag_floor,
        )
        # Separate from the call above because the RAW-target causal cells compose that mixin too
        # and have no stored-block split to weight; this half is the feature target's own.
        self._set_channel_weights(
            target_weight_st=target_weight_st, target_weight_ph=target_weight_ph
        )
        # The third feature-target-only keyword, and the only one of them that is a pure readout:
        # it changes no width, no mask and no parameter. Stashed before the base for the same
        # reason the weights are -- the keep-index it is gathered through does not exist yet.
        self._set_target_novelty(target_novelty_frac=target_novelty_frac)

        # The alignment shifts reach the base under ITS names, which is the one place in this
        # family where ``ChannelDelay`` does any work. They arrive under names of their own because
        # a run configures a *reference* and the resolver turns it into these vectors, while the
        # base's names carry the two-sided reach guard -- a different quantity, measured on a bank
        # that did not produce these coefficients, and still refused here as a constructor keyword.
        super().__init__(
            **forwarded,
            target_delays=target_align_delays,
            source_delays=source_align_delays,
        )

        # After the base, which is what validates the geometry the anchor checks read.
        self._validate_causal_geometry()
        # After the geometry check, which is what resolves the gate the weights
        # are positional over.
        self._register_channel_weights()
