r"""The causal-input raw-target forecaster: the raw-signal model, told where its inputs begin.

:class:`SeqVaeLagAttnCrws` is :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws` composed with
:class:`~teb_vae.lag_attn_crws.nets.causal_raw_inputs.CausalRawInputs`, and it holds **nothing but a
constructor**. The mixin brings the input warm-up mask, the lag validity floor, the tiled anchor set,
the forward that decodes at it, and the objective that gathers the raw future window at those
anchors; the architecture brings everything else.

**One mixin, not two, and the missing one is the point.** The causal-feature cells compose a target
mixin beside the input one because a stored-coefficient target changes the decoder's width, its
gather and its readouts. A raw target changes none of them: ``_default_decoder_out_channels``
already returns ``raw_per_step`` on the architecture, which is exactly this cell's width, so the
correct thing to do about the width hook is to not define one. Composing
:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` in here by
mistake would build the decoder at $C_{\mathrm{keep}} = 98$ against a $(B, A, H, 16)$ target, and the
first symptom would be ``raw_sample_score`` computing $(\text{target} - \mu)^2$ on shapes that do not
broadcast, three frames below the decision that caused it.

**Which is also why ``persistence_residual`` is not on the signature below.** The decoder's
persistence term carries the anchor's own value forward per *channel*, and this block's last axis
counts raw samples of one signal -- there is no per-channel level for it to carry. Leaving the
keyword off is what makes the exclusion structural: the driver's ``inspect.signature`` sweep can
reach only what a cell re-lists, so no configuration of this cell can set the flag, and the
architecture parent would refuse it by name if one could. The *other* half of horizon-aware
decoding, ``horizon_weight_halflife_steps``, **is** here: a decaying weight reads the forecast's own
$\tau$ axis, which this block has exactly as a feature block does.

**The base order is load-bearing.** The mixin comes first, so its ``forward`` -- the tiled one --
wins method resolution over the architecture's dense one, and its ``_build_adapter`` builds each
stream's availability terms from the warm-up rather than from a gate whose delays are all zero.

What cannot be shared is this file's one member: the experiment driver builds a run's kwargs by
sweeping ``inspect.signature(MODEL_CLS.__init__)``, so each cell writes out its **own**
architecture's keyword list in full. A ``**kwargs`` signature would forward four keys and silently
build an all-defaults model at $d_{model} = 128$ on a tiny smoke config.

``target_delays`` and ``source_delays`` are the only names removed from the base's list, and removing
them is the point: a warm-up is a leading *mask*, ``ChannelDelay`` is a *shift*, and a warm-up routed
under a delay name would train a different model with every shape intact. The channel alignment *is*
a shift and does reach that module -- under ``target_align_delays`` and ``source_align_delays``,
which the constructor renames on the way through, so what a checkpoint records is which of the two
quantities it was built with rather than a name that could be either.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from teb_vae.lag_attn_cfs.nets.causal_inputs import FORWARDED_EXCLUSIONS
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


class SeqVaeLagAttnCrws(CausalRawInputs, SeqVaeLagAttnRws):
    r"""Single-latent lag-attentive VAE forecasting raw samples from one-sided inputs.

    The mixin comes **first** in the bases, which is what makes the tiled forward, the warm-up
    adapter, the floored lag mask and the anchored objective win method resolution over the
    architecture's; reversing them would decode the dense anchor range, return no anchor set at all,
    and score a $(B, T_{\mathrm{valid}}, H, R)$ target against it.

    Its **absence** matters as much: nothing here or on the mixin defines
    ``_default_decoder_out_channels``, so the width *defaults* to the architecture's
    ``raw_per_step`` and a config that says nothing about it gets a decoder that matches the raw
    target by construction.

    Note the limit of that, because the conv-Transformer cell of this row has the stronger property
    and this one does not. ``decoder_out_channels`` is still a keyword of the signature below -- it
    is the architecture parent's and the signature is written out in full for the driver's sweep --
    and the driver forwards ``model_config.VAE_model`` keys onto the constructor by name. So
    ``decoder_out_channels: 98`` in a config really does build the decoder at $98$ against a
    $(B, A, H, 16)$ target. It fails loudly rather than quietly, inside ``raw_sample_score``, on the
    first batch of the first step -- but it fails there rather than at the config, so an operator
    who sets that key gets a broadcast error about two shapes instead of a sentence about the key.
    ``SeqVaeLagAttnTrfCrws`` cannot be misconfigured that way at all, because its architecture
    parent declares no such keyword.

    The constructor signature is the base's, written out in full rather than narrowed to the new
    keywords. That is not style: the experiment driver builds a run's kwargs by sweeping
    ``inspect.signature(MODEL_CLS.__init__)``, so a ``**kwargs`` signature would forward four keys
    and silently build an all-defaults model.
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
        lstm_layers: int = 2,
        dropout: float = 0.1,
        decoder_hidden: int = 128,
        decoder_out_channels: Optional[int] = None,
        horizon_depth: int = 2,
        horizon_kernel: int = 3,
        horizon_film: bool = False,
        horizon_attention_blocks: int = 0,
        horizon_embed_std: float = 0.02,
        head_init_calibration: bool = False,
        a_head_gain: float = 1.0,
        encoder_extra_dilations: Tuple[int, ...] = (),
        encoder_extra_kernel: int = 15,
        conv_norm_groups: Optional[int] = None,
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
        causal_norm: bool = False,
        coverage_floor: float = 0.9,
        base_decode: str = "sample",
        horizon_weight_halflife_steps: Optional[float] = None,
        target_keep_index: Optional[Sequence[int]] = None,
        target_warmup_steps: Optional[Sequence[int]] = None,
        source_keep_index: Optional[Sequence[int]] = None,
        source_warmup_steps: Optional[Sequence[int]] = None,
        target_align_delays: Optional[Sequence[int]] = None,
        source_align_delays: Optional[Sequence[int]] = None,
        anchor_stride: int = 1,
        lag_floor: int = 0,
        init_weights: bool = True,
    ) -> None:
        r"""Initialize the model.

        Every keyword the base takes is forwarded unchanged or renamed; only the six below are
        this input domain's, and only ``target_delays`` / ``source_delays`` are gone as names. The defaults that differ
        from the base's -- ``horizon`` $15$, ``warmup_period`` $134$, ``c_y`` $102$, ``c_u`` $51$ --
        are the causal dataset's geometry rather than a preference, and a run that leaves them at
        the base's values would be describing a dataset that does not exist.

        Note which of those four the *target* forces: none of them. A raw sample is honest at every
        step, so no validity constraint ties the floor to the resolved budget here; the values are
        held at the causal-feature cells' so that the two cells differ in exactly one variable --
        what the decoder emits.

        Args:
            target_warmup_steps: $W'_c$ per **surviving** target-stream channel, positional against
                ``target_keep_index``. ``None`` builds no warm-up mask, which is the ungated model.
                The stream is an *input* here, which is why the vector's own name says nothing about
                a target block being forecast.
            source_warmup_steps: The same for the source stream.
            target_align_delays: $d_c$ per **surviving** target channel, the shift that brings
                every one of them onto a common reference clock. Forwarded to the base as
                ``target_delays``, so it reaches ``ChannelDelay`` and the gate stops being a pure
                gather. ``None`` -- the default and the shipped setting -- builds a model bitwise
                identical to one constructed before the keyword existed.
            source_align_delays: The same for the source stream.
            anchor_stride: $S$, the spacing between decoded anchors, in $[1, H]$. Defaults to $1$ --
                the dense range every sibling decodes, and the inert value -- so a model constructed
                without an opinion behaves like the rest of the family. The tiling is a
                configuration decision, and the shipped configuration states it.
            lag_floor: $F_u$, the earliest source step lag attention may read. Ships at $0$, where
                the lag mask is bitwise the sibling's.

        Raises:
            ValueError: If ``anchor_stride`` is outside $[1, H]$ or leaves a phase with no anchor;
                if ``lag_floor`` is negative; if a warm-up vector arrives without its keep-index; or
                if ``warmup_period`` is below the input-warmth policy's floor. Everything else is
                the base's own validation.
        """
        # Captured before anything else runs, so the forwarded set is exactly this signature minus
        # the four keywords the mixin owns. Written out as forty explicit `name=name` pairs it would
        # be the same dict with one silent failure mode: a keyword added to the base and forgotten
        # here would be forwarded at its default with nothing raising.
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
