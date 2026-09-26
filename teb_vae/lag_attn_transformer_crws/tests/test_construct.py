r"""Construction: what the composition changes, what it inherits, and what it still refuses.

This cell composes one input mixin over the conv-Transformer architecture and writes one member of
its own, the constructor. It has to be written here because the experiment driver builds a run's
kwargs by sweeping ``inspect.signature(MODEL_CLS.__init__)``, and this architecture's keyword schema
is not the conv-LSTM cell's -- five keys absent, seven encoder keys added -- so the schema has to
live where the class is. Everything asserted below is therefore about that constructor: that it
forwards every architecture keyword and refuses every one it does not have, that each forwarded
switch reaches the network and is inert at its off value, that the warm-up and the alignment shift
reach the gate and the adapters, and that what it builds differs from its two neighbours in the grid
exactly where the design says -- the input adapters against the raw-signal architecture parent, the
two history encoders against the conv-LSTM twin.

The decoder is the one width no budget can move: it emits $R$ raw samples per horizon token, where a
causal-*feature* composition would emit the target gate's surviving-channel count.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    shipped_warmup_kwargs as conv_lstm_shipped_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws

from .conftest import (
    CONV_LSTM_ONLY_KEYS,
    SHIPPED_KWARGS,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    shipped_warmup_kwargs,
)

#: The seven keywords that are this architecture's own. Each names a component the conv-LSTM cell
#: has no analogue of, and together they are the entire declared difference between the two cells of
#: this row.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: The four keywords the revision added to this constructor, at their off-values. Written out
#: rather than derived from the signature's defaults: comparing the defaults against themselves
#: would pass on any edit, and what has to hold is that these particular values reproduce the
#: pre-revision model.
_SWITCHES_OFF = dict(
    lag_kv_source="encoder",
    prior_availability_input=False,
    horizon_weight_halflife_steps=None,
    alibi_slope_scale=1.0,
)


def _model(kwargs, cls=SeqVaeLagAttnTrfCrws, **overrides):
    torch.manual_seed(0)
    return cls(**dict(kwargs, **overrides))


def _n_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# =================================================================================================
# The constructor signature: every architecture keyword forwarded, every foreign one refused
# =================================================================================================
def test_the_signature_is_the_architecture_parents_with_the_delays_replaced() -> None:
    """The trainer builds its kwargs from an ``inspect.signature`` sweep of ``MODEL_CLS.__init__``,
    so a keyword the architecture parent gains and this constructor does not re-list is one no
    config can reach: the arm trains at the parent's default with no error and no shape differing.

    Removing the two delay keywords is the point: a warm-up routed under a delay name would reach
    ``ChannelDelay``, which shifts rather than masks. The alignment shifts arrive under names of
    their own for the same reason. ``persistence_residual`` adds a term in the target's own stored
    coefficient to the decoder mean; this row's target is the raw signal, so the mechanism is
    declined rather than defaulted off.
    """
    parameters = inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters

    assert set(base) - set(parameters) == {
        "target_delays",
        "source_delays",
        "persistence_residual",
    }
    assert set(parameters) - set(base) == {
        "target_warmup_steps",
        "source_warmup_steps",
        "anchor_stride",
        "lag_floor",
        "target_align_delays",
        "source_align_delays",
    }
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def test_the_seven_encoder_keywords_reach_the_encoders(tiny_kwargs) -> None:
    """Accepted is not enough: a keyword silently swallowed into ``forwarded`` and never read would
    satisfy the signature check above. Moving the two depth keys must move the parameter total, and
    moving the window must move what the source encoder is allowed to attend to."""
    base = _model(tiny_kwargs)
    deeper = _model(tiny_kwargs, target_attention_blocks=3, source_attention_blocks=3)
    wider = _model(tiny_kwargs, encoder_d_ff=128)

    assert _n_parameters(deeper) > _n_parameters(base)
    assert _n_parameters(wider) > _n_parameters(base)
    assert base.source_encoder.attention_window == tiny_kwargs["source_attention_window"]
    assert base.target_encoder.attention_window is None, "the target reads the full causal prefix"


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS + ("persistence_residual",))
def test_a_keyword_this_architecture_does_not_have_is_refused_by_name(tiny_kwargs, key) -> None:
    """Each conv-LSTM key names a component this architecture does not have, and the persistence
    residual is declined by this row, so each must fail loudly rather than be accepted and ignored
    -- which is what makes a config copied from the conv-LSTM cell fail here by name."""
    with pytest.raises(TypeError, match=key):
        _model(tiny_kwargs, **{key: 1})


@pytest.mark.parametrize(
    "overrides",
    [dict(d_z=9), dict(d_head=16), dict(encoder_num_heads=5), dict(target_attention_blocks=0)],
    ids=["indivisible-latent", "head-geometry", "indivisible-heads", "no-attention"],
)
def test_the_refusal_messages_are_the_architecture_parents(tiny_kwargs, overrides) -> None:
    """Not merely "both raise": a composition that re-derived its own guards would drift in wording
    first and in behaviour later."""
    messages = []
    for cls in (SeqVaeLagAttnTrfRws, SeqVaeLagAttnTrfCrws):
        with pytest.raises(ValueError) as excinfo:
            _model(tiny_kwargs, cls=cls, **overrides)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


def test_the_causal_refusal_messages_are_the_conv_lstm_cells(tiny_warmup) -> None:
    """The two cells share one input domain, so they must share its refusals *verbatim* -- and the
    floor check firing here at all is what shows the constructor runs the causal half's geometry
    validation after the architecture is built."""
    conv_lstm_kwargs = dict(tiny_warmup, lstm_layers=1)
    for key in _ENCODER_KEYS:
        conv_lstm_kwargs.pop(key)

    messages = []
    for cls, kwargs in (
        (SeqVaeLagAttnTrfCrws, tiny_warmup),
        (SeqVaeLagAttnCrws, conv_lstm_kwargs),
    ):
        with pytest.raises(ValueError) as excinfo:
            _model(kwargs, cls=cls, warmup_period=1)
        messages.append(str(excinfo.value))

    assert messages[0] == messages[1]


# =================================================================================================
# What the composition builds, against its two neighbours in the grid
# =================================================================================================
def test_the_shipped_decoder_emits_raw_samples_rather_than_the_surviving_channel_count() -> None:
    r"""The load-bearing **absence**: no width hook is composed in, so the decoder resolves to the
    architecture's ``raw_per_step`` while the target gate reaches the input adapters alone. A
    feature-target composition would emit the gate's surviving-channel count per horizon token
    against a $(B, A, H, R)$ raw target."""
    model = _model(shipped_warmup_kwargs())

    assert model.target_gate is not None
    assert model.decoder.mean_head.out_features == model.raw_per_step
    assert model.decoder.logvar_head.out_features == model.raw_per_step
    assert model.decoder_out_channels == model.raw_per_step != model.target_gate.out_channels


def test_the_ungated_model_is_parameter_for_parameter_the_architecture_parent(tiny_kwargs) -> None:
    """This class adds no parameter of its own: what it changes is which anchors are decoded.

    Compared against the raw-target architecture at the same keywords -- the comparison that is
    available here and is *not* available to the causal-feature cells, whose decoder is a different
    width. A parameter creeping in here (a second adapter, a learned phase, a floor embedding) fails
    rather than being absorbed into a total nobody re-derives.
    """
    causal = _model(tiny_kwargs)
    torch.manual_seed(0)
    raw = SeqVaeLagAttnTrfRws(**dict(tiny_kwargs))

    assert {name: tuple(p.shape) for name, p in causal.named_parameters()} == {
        name: tuple(p.shape) for name, p in raw.named_parameters()
    }


def test_the_encoder_edge_is_the_two_history_encoders_and_nothing_else() -> None:
    """The grid's premise, measured on the shipped geometry and budget. Both cells of this row
    forecast the same raw block at the same budget over the same tiling, so every parameter outside
    the two history encoders -- adapters, gates, prior, posterior, lag attention, decoder -- must be
    name-for-name and shape-for-shape the conv-LSTM twin's, and a difference in results is then
    attributable to the encoder alone."""
    encoders = ("target_encoder.", "source_encoder.")
    conv_lstm = _model(conv_lstm_shipped_warmup_kwargs(), cls=SeqVaeLagAttnCrws)
    transformer = _model(shipped_warmup_kwargs())

    def _outside_encoders(model):
        return {
            name: tuple(p.shape)
            for name, p in model.named_parameters()
            if not name.startswith(encoders)
        }

    assert _outside_encoders(transformer) == _outside_encoders(conv_lstm)
    assert conv_lstm.anchor_stride == transformer.anchor_stride == transformer.horizon


def test_the_budget_and_the_alignment_move_only_the_input_adapters() -> None:
    r"""Nothing in this target domain widens a head, so every parameter the resolved guard adds is
    under an input adapter and none is removed.

    The alignment arm, measured rather than asserted absent: the aligned budget drops every channel
    slower than the reference, each dropped channel costs its two $d_{\mathrm{model}}$-wide adapter
    projections, and both start embeddings are built because the shifted minimum is positive while
    the unaligned one is not. Derived from the gates, so a reference change moves the arithmetic
    with the model.
    """
    gated = _model(shipped_warmup_kwargs())
    ungated = _model(SHIPPED_KWARGS)
    unaligned = _model(shipped_warmup_kwargs(align=False))

    gated_names = {name for name, _ in gated.named_parameters()}
    ungated_names = {name for name, _ in ungated.named_parameters()}
    added = gated_names - ungated_names
    assert added, "the guarded model added no parameter at all"
    assert all("adapter" in name for name in added), sorted(added)
    assert ungated_names - gated_names == set()

    d_model = int(gated.d_model)
    narrowing = (
        gated.target_gate.out_channels
        - unaligned.target_gate.out_channels
        + gated.source_gate.out_channels
        - unaligned.source_gate.out_channels
    )
    assert narrowing != 0, "the alignment moved no channel; the arithmetic below proves nothing"
    assert _n_parameters(gated) - _n_parameters(unaligned) == narrowing * d_model * 2 + 2 * d_model


# =================================================================================================
# The guard and the alignment the constructor threads through
# =================================================================================================
def test_the_adapter_is_built_at_the_warm_up_and_not_at_the_gates_delays(tiny_warmup) -> None:
    r"""The failure the causal half's ``_build_adapter`` exists to prevent, and the one the base order
    decides: the architecture parent's own version builds the guard from ``gate.delay.delay_steps``,
    which carries the alignment shifts $d_c$ and nothing about the warm-up -- all zeros under this
    unaligned fixture, so no availability buffer, no mask projection, and a leading region of
    real-valued pre-recording history entering the encoder as though it were signal.
    """
    model = _model(tiny_warmup)

    assert model.target_adapter.max_delay == max(TINY_TARGET_WARMUP_STEPS)
    assert model.source_adapter.max_delay == max(TINY_SOURCE_WARMUP_STEPS)
    assert model.target_adapter.mask_proj is not None
    assert model.source_adapter.mask_proj is not None
    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.source_gate is not None
    assert model.source_gate.max_delay == 0, "the gate is a gather; the warm-up is not a shift"


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


# =================================================================================================
# The revision's switches, and the off state an old checkpoint rebuilds
#
# Pinned per cell rather than once on the parent, because the failure this catches is *this* cell's:
# the driver silently drops any key the class does not re-list, so a switch threaded through the
# parent and forgotten here would train the baseline under the arm's name with no error.
# =================================================================================================
@pytest.mark.parametrize(
    "explicit_off",
    [dict(target_align_delays=None, source_align_delays=None), _SWITCHES_OFF],
    ids=["alignment", "switches"],
)
def test_omitted_keywords_build_the_model_their_off_values_build_bitwise(
    tiny_warmup, explicit_off
) -> None:
    """The path an old checkpoint's saved kwargs dict exercises: it carries none of these keywords,
    so construction from it must produce the object graph and the tensor values of the model that
    was trained -- not an equivalent one with an identity ``ChannelDelay`` or a zero-initialised
    parameter in it.

    Values as well as keys, because a switch that added a zero-initialised parameter would leave the
    totals standing and change the object; and buffer names as well as parameters, because a
    non-persistent buffer is invisible to a ``state_dict`` comparison and is exactly how the horizon
    weight and the availability announcement are carried.
    """
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, **explicit_off))

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))


def test_the_prior_clock_builds_no_parameter_when_it_is_off(tiny_warmup) -> None:
    """Absent rather than present-and-zero, and the difference is a distributed run's: a parameter
    built and left inert has no gradient path, which is what ``find_unused_parameters=False``
    refuses. Both directions, so the absence is not the absence of a working mechanism."""
    off = build(dict(tiny_warmup, prior_availability_input=False))
    on = build(dict(tiny_warmup, prior_availability_input=True))

    assert "prior_head.clock_proj.weight" not in dict(off.named_parameters())
    assert "prior_head.clock_proj.weight" in dict(on.named_parameters())


def test_the_horizon_weight_is_a_non_persistent_buffer_or_nothing(tiny_warmup) -> None:
    r"""Null builds no buffer; a half-life builds one that a checkpoint does not carry.

    Non-persistent is the load-bearing half. The weight is $(H,)$, so a persistent one would put
    the horizon into the state dict and make a checkpoint unloadable at any other horizon -- for a
    tensor that is a pure function of two numbers the constructor already has.
    """
    off = build(dict(tiny_warmup, horizon_weight_halflife_steps=None))
    on = build(dict(tiny_warmup, horizon_weight_halflife_steps=5.0))

    assert "horizon_weight" not in dict(off.named_buffers())
    assert "horizon_weight" in dict(on.named_buffers())
    assert on.horizon_weight.shape == (on.horizon,)
    assert float(on.horizon_weight.sum()) == pytest.approx(float(on.horizon), rel=1e-6)
    assert not any("horizon_weight" in name for name in on.state_dict())


def test_an_unknown_kv_source_is_refused_naming_the_choices(tiny_warmup) -> None:
    """By name, with the admitted set in the message. The value reaches a branch that would
    otherwise fall through to one of the arms, so an unrecognised string would silently train the
    fall-through arm under the misspelt one's name."""
    with pytest.raises(ValueError, match=r"lag_kv_source must be one of"):
        build(dict(tiny_warmup, lag_kv_source="conv-stem"))


@pytest.mark.parametrize("arm", ["conv_stem", "adapter"])
def test_a_local_kv_arm_does_not_build_the_deep_source_encoder(tiny_warmup, arm) -> None:
    """The deep source encoder leaves the *model*, not just the lag path: nothing else consumes the
    source state, so under a local arm it would be a whole stack of parameters no forward reaches.
    Asserted by state-dict prefix rather than by a total, because a total cannot say which stack
    went."""
    deep = build(dict(tiny_warmup, lag_kv_source="encoder"))
    local = build(dict(tiny_warmup, lag_kv_source=arm))

    assert [name for name in deep.state_dict() if name.startswith("source_encoder.")]
    assert [name for name in local.state_dict() if name.startswith("source_encoder.")] == []
    assert getattr(local, "source_encoder", None) is None
