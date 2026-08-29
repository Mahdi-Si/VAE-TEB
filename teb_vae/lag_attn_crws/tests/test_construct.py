r"""The model is built, its inputs begin where the warm-up says they do, and its decoder is raw.

This cell composes one mixin over the raw-signal architecture, and the two ways that goes wrong are
opposite in kind.

**Too little.** A member that should be inherited is written here instead, and the conv-Transformer
twin composing the same mixin silently does not get it. ``vars()`` is asserted to be exactly
``{'__init__'}`` for that reason, and the constructor is the one member that genuinely cannot be
shared: the experiment driver builds a run's kwargs by sweeping
``inspect.signature(MODEL_CLS.__init__)``, so a ``**kwargs`` signature would forward four keys and
build an all-defaults model at $d_{model} = 128$ on a tiny smoke config.

**Too much.** The causal-feature target mixin is composed in as well -- which is what the design's
first draft did, and it reads perfectly plausibly. Then ``_default_decoder_out_channels`` resolves to
the *feature* one, the decoder emits $C_{\mathrm{keep}} = 98$ channels per horizon token, and the
target is $(B, A, H, 16)$. ``block_width`` does not catch it: it feeds only the four log-variance
diagnostics and no shape check. The first symptom is ``raw_sample_score`` computing
$(\text{target} - \mu)^2$ on shapes that do not broadcast, three frames below the class body that
caused it. So the wrong composition is built here and its decoder width asserted, and the reason the
mixin is excluded becomes a passing test rather than a comment.

The warm-up half is the causal-feature cell's machinery reached by identity -- ``_build_adapter`` is
asserted to be that mixin's own function object in ``test_causal_raw_inputs.py`` -- so what is
checked here is that it was *reached*: the adapter is built at $W'_c + d_c$ rather than at the
gate's own ``delay_steps``, which carry the alignment shifts $d_c$ alone -- all zero on an unaligned
config, where the gate is a pure gather, and under an alignment reference a different vector from
the warm-up rather than a smaller one -- and no gradient flows from inside the masked region.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import (
    FORWARDED_EXCLUSIONS,
    CausalWarmupInputs,
)
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    SHIPPED_HORIZON,
    SHIPPED_WARMUP_PERIOD,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    shipped_warmup_kwargs,
)

#: What the shipped budget keeps of the $102$ declared target-stream channels. A literal here rather
#: than a resolved count, because the wrong-composition guard below is about a width that must *not*
#: appear on this cell's decoder, and deriving it from the same resolution the model uses would let
#: both move together.
_KEPT_TARGET_CHANNELS = 38


# =================================================================================================
# The class
# =================================================================================================
def test_the_mixin_comes_first_and_the_mro_is_pinned() -> None:
    """As a list of names, because the *order* is what makes the tiled forward, the warm-up adapter
    and the floored lag mask resolve to the causal ones.

    Reversed, the architecture's dense forward wins: it decodes $[0, T_{\\mathrm{valid}})$, returns
    no anchor set at all, and the objective then gathers a dense target for it -- a different model
    with every shape self-consistent.
    """
    assert SeqVaeLagAttnCrws.__bases__ == (CausalRawInputs, SeqVaeLagAttnRws)
    assert [cls.__name__ for cls in SeqVaeLagAttnCrws.__mro__] == [
        "SeqVaeLagAttnCrws",
        "CausalRawInputs",
        "CausalWarmupInputs",
        "SeqVaeLagAttnRws",
        "Module",
        "object",
    ]


def test_the_model_carries_nothing_but_its_constructor() -> None:
    """Everything else is encoder-agnostic and lives on the mixin the second cell composes too."""
    assert {name for name in vars(SeqVaeLagAttnCrws) if not name.startswith("__")} == set()
    assert "__init__" in vars(SeqVaeLagAttnCrws)

    for shared in ("forward", "_build_anchor_index", "_build_adapter", "build_lag_mask"):
        assert shared not in vars(SeqVaeLagAttnCrws), shared
        assert shared in vars(CausalWarmupInputs), shared
    for shared in ("compute_loss", "_check_anchor_floor"):
        assert shared not in vars(SeqVaeLagAttnCrws), shared
        assert shared in vars(CausalRawInputs), shared


def test_the_constructor_takes_warm_ups_and_refuses_delays() -> None:
    """The names are the whole guard: a warm-up routed under a delay name trains a different model.

    The full parameter list is asserted rather than only the four new names, because the failure
    that matters is the *opposite* one -- a narrowed signature. The driver builds a run's kwargs by
    sweeping this signature.

    Three of the base's keywords are absent, and the third is a decision rather than a rename.
    ``persistence_residual`` adds $w_{\\tau,c}\\, y_{t,c}$ to the decoder mean, where $y_t$ is the
    target's own stored value; this row's target is the raw signal, whose per-step samples are a
    different object from a stored coefficient, so the mechanism is deliberately not offered here.
    Leaving it off the signature is what makes the driver's ``inspect.signature`` sweep unable to
    reach it -- a config that set it would be refused by name rather than silently ignored.
    """
    parameters = inspect.signature(SeqVaeLagAttnCrws.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnRws.__init__).parameters

    assert "target_warmup_steps" in parameters and "source_warmup_steps" in parameters
    assert "anchor_stride" in parameters and "lag_floor" in parameters
    for banned in ("target_delays", "source_delays", "persistence_residual"):
        assert banned not in parameters, banned
        assert banned in base, f"{banned} is meant to be the base's, removed here"

    assert set(base) - set(parameters) == {
        "target_delays",
        "source_delays",
        "persistence_residual",
    }
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def test_the_forwarded_set_is_this_signature_minus_the_mixins_own_keywords() -> None:
    """Captured from ``locals()`` rather than written out a second time.

    Forty explicit ``name=name`` pairs would be the same dict with one silent failure mode: a
    keyword added to the base and forgotten here would be forwarded at its default, with nothing
    raising and no shape differing. The exclusion list lives on the mixin that owns those keywords,
    so a keyword removed there cannot be left behind in a filter here.

    The right-hand union names the three the base has and this row does not: the two delay
    keywords the warm-ups replace, and the target-only persistence residual this row declines.
    """
    parameters = set(inspect.signature(SeqVaeLagAttnCrws.__init__).parameters)
    base = set(inspect.signature(SeqVaeLagAttnRws.__init__).parameters)

    forwarded = parameters - set(FORWARDED_EXCLUSIONS)
    assert (
        forwarded | {"self", "target_delays", "source_delays", "persistence_residual"} == base
    )

    source = Path(inspect.getfile(SeqVaeLagAttnCrws)).read_text(encoding="utf-8")
    assert "locals().items()" in source and "FORWARDED_EXCLUSIONS" in source


def test_a_base_keyword_no_other_test_names_still_reaches_the_base(tiny_kwargs) -> None:
    """The behavioural half of the claim above, on the two keywords a hand-written forward dict
    would be likeliest to drop: neither shapes anything, so neither would fail loudly."""
    model = build(dict(tiny_kwargs, coverage_floor=0.25, base_decode="mean"))

    assert model.coverage_floor == 0.25
    assert model.base_decode == "mean"


def test_the_geometry_defaults_are_the_causal_datasets() -> None:
    """A run left at the base's defaults would be describing a dataset that does not exist.

    None of the four is forced by this target -- a raw sample is honest at every step, so no
    validity constraint ties the floor to the resolved budget here. They are held at the
    causal-feature cells' values so that the two differ in exactly one variable.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters.items()
    }

    assert defaults["c_y"] == CAUSAL_C_Y
    assert defaults["c_u"] == CAUSAL_C_U
    assert defaults["horizon"] == SHIPPED_HORIZON
    assert defaults["warmup_period"] == SHIPPED_WARMUP_PERIOD
    assert defaults["raw_per_step"] == 16
    # The inert defaults: a model built with no opinion decodes densely and floors no lag, which is
    # what every sibling does. The tiling is a configuration decision and the config states it.
    assert defaults["anchor_stride"] == 1
    assert defaults["lag_floor"] == 0


# =================================================================================================
# The decoder's width, and the mixin that must not be composed
# =================================================================================================
def test_the_decoder_emits_raw_samples_because_no_width_hook_is_defined(tiny_warmup) -> None:
    r"""The load-bearing **absence**: neither this class nor its mixin defines
    ``_default_decoder_out_channels``, so it resolves to the architecture's ``raw_per_step``."""
    model = build(tiny_warmup)

    assert model.decoder.mean_head.out_features == 16 == model.raw_per_step
    assert model.decoder_out_channels == 16
    assert "_default_decoder_out_channels" not in vars(SeqVaeLagAttnCrws)
    assert "_default_decoder_out_channels" not in vars(CausalRawInputs)
    assert (
        SeqVaeLagAttnCrws._default_decoder_out_channels
        is vars(SeqVaeLagAttnRws)["_default_decoder_out_channels"]
    )


def test_composing_the_feature_target_mixin_builds_the_wrong_decoder() -> None:
    r"""Why that absence is load-bearing, as a passing test rather than as a comment.

    The design's first draft composed three bases -- this cell's input mixin, the causal-feature
    *target* mixin, and the architecture -- and it reads plausibly: both mixins are "the causal
    half". But the second one brings a width hook, and the decoder is then built at
    $C_{\mathrm{keep}} = 98$ while the target this cell gathers is $(B, A, H, 16)$.

    Composed here as ``(CausalFeatureForecastTarget, SeqVaeLagAttnCrws)`` rather than as the draft's
    literal base tuple, for a mechanical reason worth recording: that tuple leaves ``__init__``
    resolving to the architecture's, which does not accept this domain's four keywords, so the wrong
    model could not be constructed at all and the guard would prove nothing. Subclassing this cell
    keeps its constructor and moves only the resolution order, which is exactly the mistake.
    """
    wrong_composition = type(
        "SeqVaeLagAttnCrwsWithFeatureTarget",
        (CausalFeatureForecastTarget, SeqVaeLagAttnCrws),
        {},
    )
    kwargs = shipped_warmup_kwargs()

    torch.manual_seed(0)
    wrong = wrong_composition(**dict(kwargs))
    right = build(kwargs)

    assert wrong.decoder.mean_head.out_features == _KEPT_TARGET_CHANNELS
    assert right.decoder.mean_head.out_features == 16 == right.raw_per_step
    # The cause, stated beside the symptom: which class the width hook resolved to.
    assert wrong_composition._default_decoder_out_channels is not (
        SeqVaeLagAttnCrws._default_decoder_out_channels
    )


# =================================================================================================
# The guard the constructor builds
# =================================================================================================
def test_the_source_is_gathered_whole(tiny_warmup) -> None:
    """The source keep-index is the identity, so its gate is a gather that changes nothing.

    It is still a gate rather than ``None``: the warm-up vector travels beside it and the adapter is
    built at the gate's width, so a stream with a warm-up always has one.
    """
    model = build(tiny_warmup)

    assert model.source_gate is not None
    assert model.source_gate.out_channels == CAUSAL_C_U
    assert model.source_gate.max_delay == 0, "the gate is a gather; the warm-up is not a shift"


def test_the_target_stream_gate_keeps_the_channels_the_budget_kept(tiny_warmup) -> None:
    """And the adapter is built at that width -- while the *decoder* is not, which is the whole
    difference between this cell and the causal-feature one."""
    model = build(tiny_warmup)

    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.decoder_out_channels == 16 != len(TINY_TARGET_KEEP_INDEX)


def test_the_adapter_is_built_at_the_warm_up_and_not_at_the_gates_delays(tiny_warmup) -> None:
    r"""The specific failure the inherited ``_build_adapter`` exists to prevent.

    ``gate.delay.delay_steps`` is all zeros under a pure gather, so the architecture's own version
    would give ``max_delay = 0`` -- no availability buffer, no mask projection, and a leading region
    of real-valued pre-recording history entering the encoder as though it were signal.
    """
    model = build(tiny_warmup)

    assert model.target_adapter.max_delay == max(TINY_TARGET_WARMUP_STEPS)
    assert model.source_adapter.max_delay == max(TINY_SOURCE_WARMUP_STEPS)
    assert model.target_adapter.mask_proj is not None
    assert model.source_adapter.mask_proj is not None
    expected = torch.tensor(TINY_TARGET_WARMUP_STEPS)
    assert torch.equal(model.target_adapter.availability.argmax(dim=0), expected)


def test_no_gradient_flows_from_inside_the_warm_up(tiny_warmup) -> None:
    """By gradient, which is the stronger half of the masking claim.

    A value check passes on a model that happens to emit zeros in that region for some other
    reason; a zero gradient says the output is not a function of those inputs at all. The paired
    control -- the same channel past its warm-up is live -- is what makes it a statement about the
    warm-up rather than about a dead pathway.
    """
    model = build(tiny_warmup).eval()
    y_st, y_ph, u_stream = make_streams(tiny_warmup)
    y_ph = y_ph.clone().requires_grad_(True)

    torch.manual_seed(0)
    model(y_st, y_ph, u_stream)["mu_prior"].sum().backward()
    grad = y_ph.grad
    assert grad is not None

    checked = 0
    for position, declared in enumerate(TINY_TARGET_KEEP_INDEX):
        steps = TINY_TARGET_WARMUP_STEPS[position]
        if declared < CAUSAL_ST_WIDTH or steps == 0:
            continue
        channel = declared - CAUSAL_ST_WIDTH
        assert float(grad[:, :steps, channel].abs().max()) == 0.0, declared
        assert float(grad[:, steps:, channel].abs().max()) > 0.0, declared
        checked += 1
    assert checked > 0, "no kept phase-block channel had a warm-up; the probe proved nothing"


# =================================================================================================
# The ungated arm
# =================================================================================================
def test_without_a_budget_no_gate_and_no_availability_term_is_built(tiny_kwargs) -> None:
    """An unguarded run gets the model that has no guard, not an identity one."""
    model = build(tiny_kwargs)

    assert model.target_gate is None and model.source_gate is None
    assert model.target_warmup_steps is None and model.source_warmup_steps is None
    for adapter in (model.target_adapter, model.source_adapter):
        assert adapter.mask_proj is None
        assert adapter.start_embed is None
        assert not hasattr(adapter, "availability")
    assert model.decoder_out_channels == 16


def test_the_ungated_model_is_parameter_for_parameter_the_raw_signal_sibling(tiny_kwargs) -> None:
    """This class adds no parameter of its own: what it changes is which anchors are decoded.

    Compared against the raw-target architecture at the same keywords -- which is the comparison
    that is available here and is *not* available to the causal-feature cells, whose decoder is a
    different width. A parameter creeping in here (a second adapter, a learned phase, a floor
    embedding) fails rather than being absorbed into a total nobody re-derives.
    """
    causal = build(tiny_kwargs)
    torch.manual_seed(0)
    raw = SeqVaeLagAttnRws(**dict(tiny_kwargs))

    assert {name: tuple(p.shape) for name, p in causal.named_parameters()} == {
        name: tuple(p.shape) for name, p in raw.named_parameters()
    }
    assert sum(p.numel() for p in causal.parameters()) == sum(
        p.numel() for p in raw.parameters()
    )


# =================================================================================================
# The refusals the constructor inherits
# =================================================================================================
def test_a_warm_up_without_its_keep_index_is_refused(tiny_kwargs) -> None:
    """Unpaired, both gates stay ``None`` and the target stream's vector would route into both."""
    with pytest.raises(ValueError, match="target_keep_index"):
        SeqVaeLagAttnCrws(**dict(tiny_kwargs, target_warmup_steps=TINY_TARGET_WARMUP_STEPS))
    with pytest.raises(ValueError, match="source_keep_index"):
        SeqVaeLagAttnCrws(**dict(tiny_kwargs, source_warmup_steps=TINY_SOURCE_WARMUP_STEPS))


def test_a_stride_outside_the_horizon_is_refused_naming_it(tiny_warmup) -> None:
    """Above $H$ the decoded windows leave gaps no phase ever covers; below $1$ there is no set."""
    horizon = int(tiny_warmup["horizon"])
    for stride in (0, -1, horizon + 1):
        with pytest.raises(ValueError, match=str(stride)):
            SeqVaeLagAttnCrws(**dict(tiny_warmup, anchor_stride=stride))
    assert build(dict(tiny_warmup, anchor_stride=horizon)) is not None


def test_a_stride_wider_than_the_anchor_span_is_refused(tiny_warmup) -> None:
    """At the last phase the first anchor would not exist, and the sample would decode nothing."""
    model = build(tiny_warmup)
    span = model.geometry.t_valid - model.warmup_period

    with pytest.raises(ValueError, match="anchor_stride"):
        SeqVaeLagAttnCrws(
            **dict(tiny_warmup, warmup_period=model.geometry.t_valid - 1, anchor_stride=2)
        )
    assert span >= int(tiny_warmup["horizon"]), "the tiny geometry admits the shipped stride"


def test_a_negative_lag_floor_is_refused_naming_it(tiny_warmup) -> None:
    """It would admit lags reading before the start of the sequence."""
    with pytest.raises(ValueError, match="-3"):
        SeqVaeLagAttnCrws(**dict(tiny_warmup, lag_floor=-3))


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


def test_the_availability_of_an_aligned_channel_is_the_unaligned_one_shifted_right(
    tiny_warmup, tiny_align
) -> None:
    """Stated as the relation rather than as two independent patterns: the whole claim of the
    alignment is that a channel's content moved later by $d_c$ steps and nothing else about it
    changed, so its availability must be the same staircase translated by the same $d_c$."""
    unaligned = build(tiny_warmup).target_adapter.availability
    aligned = build(tiny_align).target_adapter.availability
    steps = int(unaligned.shape[0])

    for channel, shift in enumerate(TINY_TARGET_ALIGN_DELAYS):
        expected = torch.zeros(steps, dtype=aligned.dtype)
        expected[shift:] = unaligned[: steps - shift, channel]
        assert torch.equal(aligned[:, channel], expected), channel


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


#: This cell's own class, bound once: half of the block below asserts a property of the *signature*
#: and needs no model at all, and the other half builds through the package's own ``build`` helper
#: so two builds differ only where a keyword made them.
_CELL_CLS = SeqVaeLagAttnCrws


# =================================================================================================
# The revision's switches: the four this row takes, and the one it declines
#
# Pinned per cell rather than once on the parent, because the failure this catches is *this* cell's:
# the driver builds a run's kwargs by sweeping the constructor's signature and silently drops any
# key the class does not re-list, so a switch threaded through the parent and forgotten here would
# train the baseline under the arm's name with no error and no metric saying so.
#
# The decline is the mirror image and is asserted the same way. ``persistence_residual`` adds a term
# in the TARGET's own stored coefficient to the decoder mean; this row's target is the raw signal,
# whose per-step samples are a different object, so the mechanism is refused at the constructor
# rather than shipped off. Absence from the signature is what makes it unreachable -- the sweep
# cannot forward a key that is not there.
# =================================================================================================
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


def test_every_switch_is_a_keyword_of_this_cell_at_its_off_default() -> None:
    """Both halves of the sweep hazard: present, and defaulting off.

    Present, because a key absent here is a key the driver cannot forward -- the arm would train as
    the baseline. Defaulting off, because the default is what an old checkpoint's saved kwargs dict
    falls back to, and a default that moved would silently rebuild an old run as a new architecture.
    """
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(_CELL_CLS.__init__).parameters.items()
        if name in _SWITCHES_OFF
    }

    assert defaults == _SWITCHES_OFF


def test_the_persistence_residual_is_declined_rather_than_defaulted_off(tiny_warmup) -> None:
    """The decline, asserted where it is enforced. The keyword is not on this signature at all, so
    a config carrying it is refused by name instead of training a model its own config does not
    describe -- and the parent's own guard refuses the mechanism on a raw-target architecture even
    if the keyword were reached some other way."""
    assert "persistence_residual" not in inspect.signature(_CELL_CLS.__init__).parameters

    with pytest.raises(TypeError, match="persistence_residual"):
        build(dict(tiny_warmup, persistence_residual=True))


def test_every_switch_at_its_off_value_is_bitwise_the_model_without_the_keywords(
    tiny_warmup,
) -> None:
    """The whole off-state claim in one comparison, over the state dict and the buffer names.

    Values as well as keys, because a switch that added a zero-initialised parameter would leave
    the totals standing and change the object; and buffer names as well as parameters, because a
    non-persistent buffer is invisible to a ``state_dict`` comparison and is exactly how the
    horizon weight and the availability announcement are carried.
    """
    without = build(tiny_warmup)
    explicit = build(dict(tiny_warmup, **_SWITCHES_OFF))

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


@pytest.mark.parametrize("arm", ["encoder", "conv_stem", "adapter"])
def test_the_kv_representation_is_the_pathway_composed_in_order(tiny_warmup, arm) -> None:
    """``encode_source_kv`` is what the forward and both controls call, so what it computes has to
    be exactly the pathway's modules in order -- a helper that dropped or reordered one would move
    the keys, the values, the null control's re-encode and the prior's clock together, and every
    one of them would still be the right shape."""
    model = build(dict(tiny_warmup, lag_kv_source=arm)).eval()
    gated = torch.randn(
        2,
        model.sequence_length,
        model.source_gate.keep_index.numel(),
        generator=torch.Generator().manual_seed(3),
    )

    with torch.no_grad():
        encoded = model.encode_source_kv(gated)
        expected = gated
        for module in model.source_kv_modules():
            expected = module(expected)

    assert torch.equal(encoded, expected)
    assert encoded.shape == (2, model.sequence_length, model.d_model)
