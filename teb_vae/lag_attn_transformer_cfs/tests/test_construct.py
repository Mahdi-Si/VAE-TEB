r"""Construction: what the composition changes, what it inherits, and what it still refuses.

A composition is only worth having if it is provably one, so the tests split in two.

The first half pins the *difference* against the architecture parent, and there is exactly one:
each horizon token emits one value per surviving target channel instead of $R$ raw samples. The
decoder width follows the gate, the guard and the alignment move nothing but the adapters, and the
decoded block is the conv-LSTM causal cell's head for head.

The second half pins the *sameness*: the class body holds a constructor and nothing else, the
target-domain members are the conv-LSTM causal cell's own objects, the constructor re-lists every
architecture-parent keyword except the two delays, and every construction-time refusal fires.
Then the alignment, the revision's architecture switches at their off-state, and the lag
attention's key/value memory arms.

**Why there is a constructor here and nowhere else in the composition.** The experiment driver
builds a run's kwargs by sweeping ``inspect.signature(MODEL_CLS.__init__)``, and this architecture's
keyword schema is not the conv-LSTM causal cell's -- five keys absent, seven encoder keys added --
so the schema has to be written where the class is. It holds no logic: the causal keywords are
validated and set by the mixin, before and after ``super().__init__``, through the same methods
the conv-LSTM cell calls.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    shipped_warmup_kwargs as conv_lstm_shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    CAUSAL_C_Y,
    CONV_LSTM_ONLY_KEYS,
    SHIPPED_KWARGS,
    TINY_STRIDE,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

#: Surviving target channels at the shipped warm-up budget, and the declared width.
_KEPT_CHANNELS = 98


def _model(kwargs, cls=SeqVaeLagAttnTrfCfs, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------------------------------
# The geometry, and the width that follows the budget
# ---------------------------------------------------------------------------------------
def test_the_decoder_width_is_the_surviving_channel_count():
    """$98$ at the shipped warm-up budget. Not a configuration key -- and, as on the architecture
    parent, not even a constructor keyword -- so the width follows the gate and a run cannot decode
    a width its target does not have."""
    model = _model(shipped_warmup_kwargs())

    assert model.target_gate is not None
    assert model.target_gate.out_channels == _KEPT_CHANNELS
    assert model.decoder_out_channels == _KEPT_CHANNELS
    assert model.decoder.out_channels == _KEPT_CHANNELS
    assert model.decoder.mean_head.out_features == _KEPT_CHANNELS
    assert model.decoder.logvar_head.out_features == _KEPT_CHANNELS


def test_without_a_budget_the_decoder_width_is_the_declared_width(shipped_kwargs):
    """$102$ with no warm-up budget resolved. The unguarded arm is a configuration rather than an
    unhandled case, and its block cardinality is $H \\cdot c_y$ -- but it is not a run anyone should
    make on this dataset, because the leading region then enters the encoder as signal."""
    model = _model(shipped_kwargs)

    assert model.target_gate is None
    assert model.decoder_out_channels == CAUSAL_C_Y == model.c_y
    assert model.decoder.out_channels == CAUSAL_C_Y


def test_the_width_follows_a_gate_of_any_size(tiny_warmup):
    """The rule is the gate's count, not a constant."""
    model = _model(tiny_warmup)

    assert model.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)
    assert model.raw_per_step == 16, "the raw grid is geometry, not the decoder width"


def test_the_guard_and_the_alignment_move_nothing_but_the_adapters():
    r"""Measured on constructed models rather than restated as totals. The budget narrows the
    decoder, and the alignment narrows the source adapter's two $d_{\mathrm{model}}$-wide linears
    while bringing both start-of-record vectors into existence -- and nothing else may move."""
    gated = _model(shipped_warmup_kwargs())
    ungated = _model(SHIPPED_KWARGS)
    unaligned = _model(shipped_warmup_kwargs(align=False))

    assert gated.decoder_out_channels < ungated.decoder_out_channels

    dropped = unaligned.source_gate.out_channels - gated.source_gate.out_channels
    d_model = int(gated.d_model)
    assert dropped > 0
    assert _n_parameters(gated) - _n_parameters(unaligned) == -dropped * d_model * 2 + 2 * d_model


def test_the_encoder_edge_decodes_the_same_block_as_the_conv_lstm_cell():
    """The grid's premise: at the same guard both causal cells forecast the same block over the
    same horizon and tiling, so a difference between them is the encoder alone. Asserted on the
    heads themselves, because a parameter total would also be satisfied by two models that differed
    in the head and compensated elsewhere."""
    conv_lstm = _model(conv_lstm_shipped_warmup_kwargs(), cls=SeqVaeLagAttnCfs)
    transformer = _model(shipped_warmup_kwargs())

    for name in ("mean_head", "logvar_head"):
        assert (
            getattr(conv_lstm.decoder, name).weight.shape
            == getattr(transformer.decoder, name).weight.shape
        ), name
    assert conv_lstm.decoder_out_channels == transformer.decoder_out_channels
    assert conv_lstm.horizon == transformer.horizon
    assert conv_lstm.anchor_stride == transformer.anchor_stride


# ---------------------------------------------------------------------------------------
# What the class is
# ---------------------------------------------------------------------------------------
def test_the_class_defines_a_constructor_and_nothing_else():
    """Set equality over ``vars``, and the set is exactly ``{'__init__'}``.

    Not a line count, which passes a class that overrode ``forward`` in 140 lines. With nothing else
    defined here, the twenty-two forward keys, the posterior's structure, the lag map, the anchor
    tiling and the objective's metric set cannot have moved, because they are the two mixins' and
    the architecture parent's own code objects.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfCfs).items()
        if callable(value) and not isinstance(value, type)
    }

    assert own == {"__init__"}
    assert {name for name in vars(SeqVaeLagAttnTrfCfs) if not name.startswith("__")} == set()
    assert "forward" not in vars(SeqVaeLagAttnTrfCfs)


@pytest.mark.parametrize(
    "name",
    ["forward", "_build_anchor_index", "_build_adapter", "build_lag_mask",
     "_resolved_forecast_gaps", "_default_decoder_out_channels", "TARGET_BLOCK_SPLIT"],
)
def test_the_two_causal_cells_share_every_target_domain_member(name):
    """The other direction, and the one that matters for the comparison: the conv-LSTM causal cell
    reaches the same objects. A member that had drifted onto one model would make a difference in
    results attributable to something other than the encoder."""
    assert getattr(SeqVaeLagAttnTrfCfs, name) is getattr(SeqVaeLagAttnCfs, name)


# ---------------------------------------------------------------------------------------
# The constructor signature
# ---------------------------------------------------------------------------------------
def test_the_signature_re_lists_every_architecture_keyword_but_the_two_delays():
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``.
    A narrowed signature would forward no configuration at all and silently build an all-defaults
    model -- no error, no shape mismatch, a run at the wrong widths.

    The two delay keywords are the only removals, and removing them is the point: a warm-up routed
    under a delay name would reach ``ChannelDelay``, which shifts rather than masks, and would train
    a different model with every shape intact.
    """
    parameters = inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters

    assert set(base) - set(parameters) == {"target_delays", "source_delays"}
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_the_conv_lstm_only_keywords_are_refused(tiny_kwargs, key):
    """Each names a component this architecture does not have, so each must fail loudly rather than
    be accepted and ignored -- and each *is* a keyword of the conv-LSTM causal cell, which is what
    makes a config copied from that package fail here by name."""
    with pytest.raises(TypeError, match=key):
        _model(tiny_kwargs, **{key: 1})
    assert key in inspect.signature(SeqVaeLagAttnCfs.__init__).parameters


# ---------------------------------------------------------------------------------------
# The inherited refusals, with the inherited messages
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(d_z=9), r"d_z=9.*num_heads=4"),
        (dict(d_head=16), "d_model"),
        (dict(max_lag=-1), "max_lag"),
        (dict(c_y=0), "c_y"),
        (dict(c_u=0), "c_u"),
        (dict(encoder_conv_kernels=(3, 3, 3)), "equal length"),
        (dict(target_attention_blocks=0), "at least 1"),
        (dict(source_attention_window=0), "at least 1 step"),
        (dict(encoder_num_heads=5), "divisible"),
    ],
    ids=["indivisible-latent", "head-geometry", "negative-lag", "zero-c_y", "zero-c_u",
         "stem-schedules", "no-attention", "zero-window", "indivisible-heads"],
)
def test_the_construction_refusals_are_the_architectures(tiny_kwargs, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_kwargs, **overrides)


# ---------------------------------------------------------------------------------------
# The causal refusals, with the conv-LSTM cell's messages
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(anchor_stride=0), r"anchor_stride must be in"),
        (dict(anchor_stride=99), r"anchor_stride must be in"),
        (dict(lag_floor=-1), r"lag_floor must be >= 0"),
    ],
    ids=["stride-zero", "stride-above-horizon", "negative-floor"],
)
def test_the_causal_refusals_fire_here_too(tiny_warmup, overrides, match):
    with pytest.raises(ValueError, match=match):
        _model(tiny_warmup, **overrides)


def test_a_warmup_without_its_keep_index_is_refused(tiny_kwargs):
    """The one arrangement that would misroute a vector: the adapters are told apart by which gate
    they were handed, so a warm-up with no gate would leave both gates ``None`` and route the
    target's vector into both streams."""
    with pytest.raises(ValueError, match="target_warmup_steps was given without"):
        _model(tiny_kwargs, target_warmup_steps=(0, 1))


def test_a_floor_below_the_kept_channels_warmup_is_refused(tiny_warmup):
    """$F \\ge B - 1$, enforced at construction rather than assumed. Below it the objective scores
    the assumed pre-recording history of the slowest kept channel as signal, with every shape
    correct and every warm-fraction readout still reporting $1.0$."""
    with pytest.raises(ValueError, match="below the anchor floor"):
        _model(tiny_warmup, warmup_period=1)


# ---------------------------------------------------------------------------------------
# The five constants the readouts are computed against
# ---------------------------------------------------------------------------------------
def test_the_warmup_readout_constants_are_resolved_at_construction(tiny_warmup):
    """Resolved once from the budget and the geometry, and registered as **non-persistent** buffers
    so a checkpoint trained at one budget fails to load at another as a budget mismatch rather than
    as misaligned keys."""
    model = _model(tiny_warmup)

    assert model.target_warm_frac == 1.0
    assert model.warm_tertile_id.shape == (len(TINY_TARGET_KEEP_INDEX),)
    # The second partition of the same kept axis, by how much of each coefficient the anchor has
    # not seen. Shaped by the survivors although the vector it comes from is declared-width, which
    # is the gather this cell inherits along with the mixin.
    assert model.novelty_tertile_id.shape == (len(TINY_TARGET_KEEP_INDEX),)
    assert model.source_block_warm_st.shape == (model.sequence_length,)
    assert model.source_block_warm_ph.shape == (model.sequence_length,)

    state = model.state_dict()
    for name in (
        "warm_tertile_id",
        "novelty_tertile_id",
        "source_block_warm_st",
        "source_block_warm_ph",
    ):
        assert name not in state, f"{name} is persistent; a budget change would read as key drift"


def test_the_tiling_geometry_is_the_configured_one(tiny_warmup):
    """$A_{\\max} = \\lceil (T_{\\mathrm{valid}} - F)/S \\rceil$, a geometry constant no rank can
    disagree about."""
    model = _model(tiny_warmup, anchor_stride=TINY_STRIDE)
    anchors, valid = model._build_anchor_index(
        batch=2, device=torch.device("cpu"), anchor_phase=0, anchor_stride=TINY_STRIDE
    )

    span = model.geometry.t_valid - model.warmup_period
    assert anchors.shape == (2, -(-span // TINY_STRIDE))
    assert valid.all()
    assert anchors[0].tolist() == list(
        range(model.warmup_period, model.geometry.t_valid, TINY_STRIDE)
    )


# =================================================================================================
# The channel alignment
# =================================================================================================
def test_omitting_the_alignment_keywords_builds_todays_model_bitwise(tiny_warmup) -> None:
    """The path an old checkpoint's saved kwargs dict actually exercises.

    A checkpoint written before these keywords existed carries neither, so construction from it
    must produce the object graph and the tensor values of the model that was trained -- not an
    equivalent one with an identity ``ChannelDelay`` in it. Asserted over the state dict *and* over
    the buffer names, because an identity shift is numerically invisible and structurally is not.
    """
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, target_align_delays=None, source_align_delays=None))

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))
    assert without.target_gate.max_delay == explicit.target_gate.max_delay == 0
    assert without.source_delay_steps == explicit.source_delay_steps == 0
    assert without.target_adapter.start_embed is None
    assert without.source_adapter.start_embed is None


def test_the_shift_reaches_the_gate_and_the_adapter_carries_warm_up_plus_shift(
    tiny_align,
) -> None:
    r"""The silent half of the alignment, on this cell's own composition.

    The keywords are renamed on the way to the base, so the gate is where they land; and a
    gathered-and-delayed channel is honest only once the step index has reached **both** $W'_c$ and
    $d_c$, so the vector the availability mask and the announcement are built from is the sum. Fed
    the warm-up alone, the adapter would call a channel warm $d_c$ steps early and every shape,
    every metric and every gradient would be exactly as they are now.
    """
    model = build(tiny_align)
    combined = tuple(
        wait + shift
        for wait, shift in zip(TINY_TARGET_WARMUP_STEPS, TINY_TARGET_ALIGN_DELAYS)
    )

    assert tuple(int(value) for value in model.target_gate.delay.delay_steps) == (
        TINY_TARGET_ALIGN_DELAYS
    )
    assert model.target_adapter.max_delay == max(combined)
    assert model.target_adapter.min_delay == min(combined)
    pattern = model.target_adapter.availability
    for channel, delay in enumerate(combined):
        column = pattern[:, channel]
        assert not bool(column[:delay].any()), channel
        assert bool(column[delay:].all()), channel


def test_a_non_null_reference_builds_the_start_of_record_embedding(
    tiny_warmup, tiny_align
) -> None:
    r"""A construction-time change no shipped configuration of this family has ever made.

    ``AvailabilityInputAdapter`` builds ``start_embed`` when $\min_c \delta_c > 0$. Unaligned, both
    streams have a channel at $W' = 0$, so the token is permanently inert and is not built. Under
    the shift the minimum of $W'_c + d_c$ lifts off zero on both streams and it comes into
    existence: a new learned parameter of width $d_{\mathrm{model}}$ per stream, and a live token in
    the forward pass. This is wanted, and it must be asserted rather than discovered in a parameter
    total.
    """
    off = build(tiny_warmup)
    on = build(tiny_align)

    assert off.target_adapter.start_embed is None and off.source_adapter.start_embed is None
    for adapter in (on.target_adapter, on.source_adapter):
        assert adapter.start_embed is not None
        assert adapter.start_embed.shape == (int(on.d_model),)
        assert bool(adapter.start_indicator.any()), "an inert token is not a token"

    assert sum(p.numel() for p in on.parameters()) - sum(
        p.numel() for p in off.parameters()
    ) == 2 * int(on.d_model)


def test_the_anchor_floor_rises_to_the_shifted_warmth(tiny_align, tiny_warmup) -> None:
    r"""Both requirements, and they do not move together.

    The scored target is never shifted, so its half stays where it was. The *inputs* are, and an
    aligned channel vector at step $t$ asserts one physical instant -- an assertion that is false,
    not partially true, while any entry has not arrived. So the floor must clear
    $\max_c(W'_c + d_c)$, which costs exactly one anchor here as it does at the shipped reference.
    The unaligned floor is unmoved, which is the control: a check applying the second half
    unconditionally would refuse the shipped configuration.
    """
    floor = int(tiny_align["warmup_period"])
    assert floor == max(TINY_TARGET_WARMUP_STEPS), "the flat combined vector is what this pins"

    with pytest.raises(ValueError) as error:
        build(dict(tiny_align, warmup_period=floor - 1))
    assert f"warmup_period={floor - 1}" in str(error.value)
    assert f"at least {floor}" in str(error.value)

    assert build(dict(tiny_align, warmup_period=floor)) is not None
    assert build(dict(tiny_warmup, warmup_period=floor - 1)) is not None


# =================================================================================================
# The revision's five switches, and the off-state of each
#
# Pinned per cell rather than once on the parent, because the failure this catches is *this* cell's:
# the driver builds a run's kwargs by sweeping the constructor's signature and silently drops any
# key the class does not re-list, so a switch threaded through the parent and forgotten here would
# train the baseline under the arm's name with no error and no metric saying so.
#
# The other half is the off-state. Every mechanism must reproduce, bitwise and key for key, the
# model that was trained before it existed -- that is what makes an arm comparable to a record, and
# what a checkpoint written under one setting and read under another silently violates.
# =================================================================================================
#: The five keywords the revision added to this constructor, at their off-values. Written out
#: rather than derived from the signature's defaults: comparing the defaults against themselves
#: would pass on any edit, and what has to hold is that these particular values reproduce the
#: pre-revision model -- which also pins that each keyword defaults to its off-value, since the
#: model built without them is the reference.
_SWITCHES_OFF = dict(
    lag_kv_source="encoder",
    prior_availability_input=False,
    persistence_residual=False,
    horizon_weight_halflife_steps=None,
    alibi_slope_scale=1.0,
)


def test_every_switch_at_its_off_value_is_bitwise_the_model_without_the_keywords(
    tiny_warmup,
) -> None:
    """The whole off-state claim in one comparison, over the state dict and the buffer names.

    Values as well as keys, because a switch that added a zero-initialised parameter would leave
    the totals standing and change the object; and buffer names as well as parameters, because a
    non-persistent buffer is invisible to a ``state_dict`` comparison and is exactly how the
    horizon weight and the availability announcement are carried.
    """
    without = _model(tiny_warmup)
    explicit = _model(tiny_warmup, **_SWITCHES_OFF)

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))


@pytest.mark.parametrize(
    "keyword, absent",
    [
        ("prior_availability_input", "prior_head.clock_proj.weight"),
        ("persistence_residual", "decoder.persistence_weight"),
    ],
)
def test_an_off_switch_builds_no_parameter_at_all(tiny_warmup, keyword, absent) -> None:
    """Absent rather than present-and-zero, and the difference is a distributed run's: a parameter
    built and left inert has no gradient path, which is what ``find_unused_parameters=False``
    refuses. This is the encoder whose reachability suite runs, so the two halves have to agree."""
    off = _model(tiny_warmup, **{keyword: False})
    on = _model(tiny_warmup, **{keyword: True})

    assert absent not in dict(off.named_parameters())
    assert absent in dict(on.named_parameters())


def test_the_horizon_weight_is_a_non_persistent_buffer_or_nothing(tiny_warmup) -> None:
    r"""Null builds no buffer; a half-life builds one that a checkpoint does not carry.

    Non-persistent is the load-bearing half. The weight is $(H,)$, so a persistent one would put
    the horizon into the state dict and make a checkpoint unloadable at any other horizon -- for a
    tensor that is a pure function of two numbers the constructor already has.
    """
    off = _model(tiny_warmup, horizon_weight_halflife_steps=None)
    on = _model(tiny_warmup, horizon_weight_halflife_steps=5.0)

    assert "horizon_weight" not in dict(off.named_buffers())
    assert "horizon_weight" in dict(on.named_buffers())
    assert on.horizon_weight.shape == (on.horizon,)
    assert float(on.horizon_weight.sum()) == pytest.approx(float(on.horizon), rel=1e-6)
    assert not any("horizon_weight" in name for name in on.state_dict())


# =================================================================================================
# The lag attention's key/value memory
# =================================================================================================
def test_an_unknown_kv_source_is_refused_naming_the_choices(tiny_warmup) -> None:
    """By name, with the admitted set in the message. The value reaches a branch that would
    otherwise fall through to one of the arms, so an unrecognised string would silently train the
    fall-through arm under the misspelt one's name."""
    with pytest.raises(ValueError, match=r"lag_kv_source must be one of"):
        _model(tiny_warmup, lag_kv_source="conv-stem")


@pytest.mark.parametrize("arm", ["conv_stem", "adapter"])
def test_a_local_kv_arm_does_not_build_the_deep_source_encoder(tiny_warmup, arm) -> None:
    """The windowed source encoder leaves the *model*, not just the lag path.

    Nothing else consumes the source state, so under a local arm it would be a whole attention
    stack no forward reaches. On this encoder that is the larger of the two savings and the one
    the design's parameter table is read on, which is why it is asserted by state-dict prefix
    rather than by a total: a total cannot say which stack went.
    """
    deep = _model(tiny_warmup, lag_kv_source="encoder")
    local = _model(tiny_warmup, lag_kv_source=arm)

    assert [name for name in deep.state_dict() if name.startswith("source_encoder.")]
    assert [name for name in local.state_dict() if name.startswith("source_encoder.")] == []
    assert getattr(local, "source_encoder", None) is None
    assert sum(p.numel() for p in local.parameters()) < sum(p.numel() for p in deep.parameters())


def test_the_conv_stem_arm_builds_a_bounded_stem_and_the_adapter_arm_builds_nothing(
    tiny_warmup,
) -> None:
    r"""What each local arm puts in the encoder's place, and what resolves the arm.

    ``source_kv_body`` is the single place the arm becomes a module, and every consumer -- the
    forward, both source controls, the prior clock, the norm guard -- goes through it or through
    ``encode_source_kv``, so pinning it is pinning that they cannot disagree.

    **The stem's reach is asserted, not assumed.** The whole content of a local arm is that the
    value at lag $\ell$ is a function of a bounded window rather than of the prefix; a stem that
    inherited a long dilation tail would satisfy every structural assertion above and still be
    effectively whole-prefix, at which point the arm tests nothing.
    """
    stem = _model(tiny_warmup, lag_kv_source="conv_stem")
    adapter = _model(tiny_warmup, lag_kv_source="adapter")
    deep = _model(tiny_warmup, lag_kv_source="encoder")

    assert stem.source_kv_body() is stem.source_kv_stem
    assert adapter.source_kv_body() is None
    assert deep.source_kv_body() is deep.source_encoder

    assert adapter.source_kv_modules() == (adapter.source_adapter,)
    assert stem.source_kv_modules() == (stem.source_adapter, stem.source_kv_stem)
    assert 1 < stem.source_kv_stem.receptive_field < stem.sequence_length
