r"""The model is built, and its inputs begin where the warm-up says they do.

The claim this package exists to make is that no coefficient the model reads was computed from
anything but the past. Half of that is the dataset's; the other half is here, and it is easy to get
silently wrong in three ways.

**The warm-up region is not empty.** A one-sided filter emits real float values before $W'_c$, from
assumed pre-recording history, and the normalisation constants were accumulated while deliberately
excluding exactly that region -- so those numbers are on no defined scale, and a model that reads
them is training on pad with nothing raising.

**The gate does not mask it.** ``ChannelGate`` here selects channels and -- under an alignment
reference -- shifts them; it never masks. On an unaligned config it is built with ``delays=None``, a
pure gather, so ``gate.delay.delay_steps`` is all zeros; under the shipped ``causal_align_reference``
those entries are the alignment shifts $d_c$, a different quantity from $W'_c$ and unrelated to it.
A model that built its adapter from them -- as the base does, correctly, for a reach-budget guard --
would mask at $\mathbb 1[t \ge d_c]$ rather than at $\mathbb 1[t \ge W'_c + d_c]$: ``max_delay = 0``
and **neither** availability term in the unaligned case, and in the aligned one a mask that
announces every channel warm $W'_c$ steps before it is. Either way the encoder sees the warm-up
region as signal.

**A warm-up under a delay name would shift instead of masking.** ``target_delays`` reaches
``ChannelDelay``, whose output at step $t$ is the input at $t - \delta_c$; the content is then
permanently late rather than removed. Every shape survives it.

So the tests below assert the mask by **gradient** as well as by value: a value check passes on a
model that happens to emit zeros there, while a zero gradient is a statement about what the output
is a function of.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.tests.conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_ST_WIDTH,
    TINY_SOURCE_ALIGN_DELAYS,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_align_kwargs,
    tiny_warmup_kwargs,
)


@pytest.fixture
def tiny_align():
    """A fresh copy of the tiny kwargs carrying the guard and the alignment (safe to mutate)."""
    return tiny_align_kwargs()
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws


# =================================================================================================
# The class
# =================================================================================================
def test_the_base_order_puts_the_target_domain_first() -> None:
    r"""Reversed, the decoder is built at $R = 16$ and a $C_{\mathrm{keep}}$-wide block is scored
    against it -- which ``block_width`` would not catch, because it feeds only the four
    log-variance diagnostics and no shape check.

    Two mixins, not one, and both ahead of the architecture: the input warm-up, the lag floor and
    the tiled forward name no encoder either, so they sit beside the target domain rather than on
    the model -- which is what lets a second architecture compose the identical pair.
    """
    assert SeqVaeLagAttnCfs.__mro__ == (
        SeqVaeLagAttnCfs,
        CausalWarmupInputs,
        CausalFeatureForecastTarget,
        SeqVaeLagAttnFs.__mro__[1],  # FeatureForecastTarget, reached through the parent it extends
        SeqVaeLagAttnRws,
        torch.nn.Module,
        object,
    )
    assert SeqVaeLagAttnCfs.__bases__ == (
        CausalWarmupInputs,
        CausalFeatureForecastTarget,
        SeqVaeLagAttnRws,
    )


def test_the_model_carries_nothing_but_its_constructor() -> None:
    """Everything else is encoder-agnostic and lives on a mixin the second cell composes too.

    Asserted by set equality rather than by a line count: a member added here would be one the
    conv-Transformer cell silently does not get, and the two models would stop being the same
    target domain over two architectures.
    """
    own = {name for name in vars(SeqVaeLagAttnCfs) if not name.startswith("__")}
    assert own == set()
    assert "__init__" in vars(SeqVaeLagAttnCfs)
    for shared in ("forward", "_build_anchor_index", "_build_adapter", "build_lag_mask"):
        assert shared not in vars(SeqVaeLagAttnCfs), shared
        assert shared in vars(CausalWarmupInputs), shared


def test_the_constructor_takes_warm_ups_and_refuses_delays() -> None:
    """The names are the whole guard: a warm-up routed under a delay name trains a different model.

    The full parameter list is asserted rather than only the four new names, because the failure
    that matters is the *opposite* one -- a narrowed signature. The driver builds a run's kwargs by
    sweeping this signature, so a ``**kwargs`` constructor would forward four keys and silently
    build an all-defaults model at ``d_model=128`` on a tiny smoke config.
    """
    parameters = inspect.signature(SeqVaeLagAttnCfs.__init__).parameters
    base = inspect.signature(SeqVaeLagAttnRws.__init__).parameters

    assert "target_warmup_steps" in parameters and "source_warmup_steps" in parameters
    assert "anchor_stride" in parameters and "lag_floor" in parameters
    for banned in ("target_delays", "source_delays"):
        assert banned not in parameters, banned
        assert banned in base, f"{banned} is meant to be the base's, removed here"

    # Everything else the base takes is still reachable, which is what the signature sweep needs.
    assert set(base) - set(parameters) == {"target_delays", "source_delays"}
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def test_the_geometry_defaults_are_the_causal_ones() -> None:
    """A cfs run left at the base's defaults would be describing a dataset that does not exist."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnCfs.__init__).parameters.items()
    }
    assert defaults["c_y"] == CAUSAL_C_Y
    assert defaults["c_u"] == CAUSAL_C_U
    assert defaults["horizon"] == 30
    assert defaults["warmup_period"] == 134
    # The inert defaults: a model built with no opinion decodes densely and floors no lag, which is
    # what every sibling does. The tiling is a configuration decision and the config states it.
    assert defaults["anchor_stride"] == 1
    assert defaults["lag_floor"] == 0


# =================================================================================================
# The guard the constructor builds
# =================================================================================================
def test_the_source_is_gathered_whole(tiny_warmup) -> None:
    """The source keep-index is the identity, so its gate is a gather that changes nothing.

    It is still a gate rather than ``None``: the warm-up vector travels beside it and the adapter
    is built at the gate's width, so a stream with a warm-up always has one.
    """
    model = build(tiny_warmup)

    assert model.source_gate is not None
    assert model.source_gate.out_channels == CAUSAL_C_U
    assert model.source_gate.keep_index.tolist() == list(range(CAUSAL_C_U))
    assert model.source_gate.max_delay == 0, "the gate is a gather; the warm-up is not a shift"


def test_the_target_gate_keeps_the_channels_the_budget_kept(tiny_warmup) -> None:
    """And the adapter is built at that width, not at the declared one."""
    model = build(tiny_warmup)

    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)


def test_the_adapter_is_built_at_the_warm_up_and_not_at_the_gates_delays(tiny_warmup) -> None:
    r"""The specific failure the ``_build_adapter`` override exists to prevent.

    ``gate.delay.delay_steps`` is all zeros under a pure gather, so the base's version would give
    ``max_delay = 0`` -- no availability buffer, no mask projection, and a leading region of
    real-valued pre-recording history entering the encoder as signal.
    """
    model = build(tiny_warmup)

    assert model.target_adapter.max_delay == max(TINY_TARGET_WARMUP_STEPS)
    assert model.source_adapter.max_delay == max(TINY_SOURCE_WARMUP_STEPS)
    assert model.target_adapter.mask_proj is not None
    assert model.source_adapter.mask_proj is not None
    # The pattern the adapter masks with is exactly the resolved vector, per channel.
    expected = torch.tensor(TINY_TARGET_WARMUP_STEPS)
    first_valid = model.target_adapter.availability.argmax(dim=0)
    assert torch.equal(first_valid, expected)


def test_the_start_token_is_built_only_when_every_channel_waits(tiny_warmup) -> None:
    r"""$e_{\mathrm{start}}$ exists iff $\min_c W'_c > 0$: the indicator is non-zero only when
    *every* channel is still cold, so a mixed vector leaves it permanently inert.

    Both streams are asserted, so a budget change that flips either is visible rather than
    inferred. Dropping the source scattering block is the configuration that flips one: the phase
    block alone has no channel honest from step $0$.
    """
    model = build(tiny_warmup)
    assert min(TINY_TARGET_WARMUP_STEPS) == 0 and min(TINY_SOURCE_WARMUP_STEPS) == 0
    assert model.target_adapter.start_embed is None
    assert model.source_adapter.start_embed is None

    phase_only = tiny_warmup_kwargs(
        use_up_st=False,
        c_u=CAUSAL_C_U - CAUSAL_ST_WIDTH,
        source_keep_index=tuple(range(CAUSAL_C_U - CAUSAL_ST_WIDTH)),
        source_warmup_steps=TINY_SOURCE_WARMUP_STEPS[CAUSAL_ST_WIDTH:],
    )
    assert min(phase_only["source_warmup_steps"]) > 0
    narrowed = build(phase_only)
    assert narrowed.source_adapter.start_embed is not None
    assert narrowed.target_adapter.start_embed is None


def test_the_shipped_budget_builds_both_start_tokens() -> None:
    r"""The alignment brings a parameter into existence that no shipped configuration had before.

    The adapter builds ``start_embed`` only when every channel of its stream is still pre-warm-up at
    step $0$. Unaligned, both minima are $0$ -- some channel is honest immediately on each stream --
    so neither token existed. Under the shipped reference the adapter is fed $W'_c + d_c$, whose
    minimum is $91$ on both streams, so both are constructed: a learned vector of width
    $d_{\mathrm{model}}$ per stream, and a live "everything here is still pre-warm-up" token in the
    forward pass. Asserted rather than discovered, because the alternative is finding it as a
    parameter-count disagreement in ``test_docs.py``.
    """
    kwargs = shipped_warmup_kwargs()
    assert min(kwargs["target_warmup_steps"]) == 0
    assert min(kwargs["source_warmup_steps"]) == 0
    combined = [
        min(warm + shift for warm, shift in zip(kwargs[f"{name}_warmup_steps"],
                                                kwargs[f"{name}_align_delays"]))
        for name in ("target", "source")
    ]
    assert combined == [80, 80]

    model = build(kwargs)
    assert model.target_adapter.start_embed is not None
    assert model.source_adapter.start_embed is not None
    assert model.target_adapter.start_embed.shape[-1] == model.d_model
    assert model.source_adapter.start_embed.shape[-1] == model.d_model
    assert model.target_gate is not None and model.target_gate.out_channels == 98
    assert model.decoder_out_channels == 98
    # The source loses the four channels above the reference, and only those.
    assert model.source_gate is not None and model.source_gate.out_channels == 47


def test_the_unaligned_budget_builds_neither_start_token() -> None:
    """The comparison arm, and the assertion the test above replaced: with no reference the
    adapter sees the warm-up alone, whose minimum is $0$ on both streams."""
    kwargs = shipped_warmup_kwargs(align=False)
    assert "target_align_delays" not in kwargs and "source_align_delays" not in kwargs

    model = build(kwargs)
    assert model.target_adapter.start_embed is None
    assert model.source_adapter.start_embed is None
    assert model.source_gate is not None and model.source_gate.out_channels == 51


# =================================================================================================
# What the encoder can see
# =================================================================================================
def _plant(stream: torch.Tensor, channel: int, steps: int, value: float) -> torch.Tensor:
    """A copy of ``stream`` with one channel's leading ``steps`` positions set to ``value``."""
    planted = stream.clone()
    planted[:, :steps, channel] = value
    return planted


def test_a_value_inside_the_warm_up_reaches_no_encoder_input(tiny_warmup) -> None:
    """By value: two streams differing only inside the warm-up produce the same adapter output."""
    model = build(tiny_warmup).eval()
    y_st, y_ph, u_stream = make_streams(tiny_warmup)

    # A kept target channel whose warm-up is non-empty, in declared coordinates.
    position = next(
        index for index, step in enumerate(TINY_TARGET_WARMUP_STEPS) if step > 0
    )
    declared = TINY_TARGET_KEEP_INDEX[position]
    steps = TINY_TARGET_WARMUP_STEPS[position]

    seen: list = []
    handle = model.target_adapter.register_forward_hook(
        lambda module, args, output: seen.append(output)
    )
    try:
        with torch.no_grad():
            torch.manual_seed(0)
            model(y_st, y_ph, u_stream)
            torch.manual_seed(0)
            if declared < CAUSAL_ST_WIDTH:
                model(_plant(y_st, declared, steps, 1e3), y_ph, u_stream)
            else:
                model(y_st, _plant(y_ph, declared - CAUSAL_ST_WIDTH, steps, 1e3), u_stream)
    finally:
        handle.remove()

    assert len(seen) == 2
    assert torch.equal(seen[0], seen[1])


def test_no_gradient_flows_from_inside_the_warm_up(tiny_warmup) -> None:
    """By gradient, which is the stronger half.

    A value check passes on a model that happens to emit zeros in that region for some other
    reason; a zero gradient says the output is not a function of those inputs at all. The control
    below is what makes it a statement about the warm-up rather than about a dead pathway.
    """
    model = build(tiny_warmup).eval()
    y_st, y_ph, u_stream = make_streams(tiny_warmup)
    y_ph = y_ph.clone().requires_grad_(True)

    torch.manual_seed(0)
    model(y_st, y_ph, u_stream)["mu_prior"].sum().backward()
    grad = y_ph.grad
    assert grad is not None

    warmup_by_declared = {
        TINY_TARGET_KEEP_INDEX[position]: step
        for position, step in enumerate(TINY_TARGET_WARMUP_STEPS)
    }
    checked = 0
    for declared, steps in warmup_by_declared.items():
        if declared < CAUSAL_ST_WIDTH or steps == 0:
            continue
        channel = declared - CAUSAL_ST_WIDTH
        assert float(grad[:, :steps, channel].abs().max()) == 0.0, declared
        # The control: the same channel past its warm-up is live.
        assert float(grad[:, steps:, channel].abs().max()) > 0.0, declared
        checked += 1
    assert checked > 0, "no kept phase channel had a warm-up; the probe proved nothing"


def test_a_dropped_channel_reaches_nothing_at_any_step(tiny_warmup) -> None:
    """The gather's own half: a channel the budget dropped is not read anywhere, warm or not."""
    model = build(tiny_warmup).eval()
    y_st, y_ph, u_stream = make_streams(tiny_warmup)
    y_ph = y_ph.clone().requires_grad_(True)

    torch.manual_seed(0)
    model(y_st, y_ph, u_stream)["mu_prior"].sum().backward()
    grad = y_ph.grad
    assert grad is not None

    dropped = [
        declared - CAUSAL_ST_WIDTH
        for declared in range(CAUSAL_ST_WIDTH, CAUSAL_C_Y)
        if declared not in set(TINY_TARGET_KEEP_INDEX)
    ]
    assert dropped, "the tiny budget dropped no phase channel; the probe proved nothing"
    assert float(grad[:, :, dropped].abs().max()) == 0.0


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
    assert model.decoder_out_channels == CAUSAL_C_Y


def test_the_ungated_model_is_parameter_for_parameter_the_two_sided_sibling(tiny_kwargs) -> None:
    """This class adds no parameter of its own: what it changes is which anchors are decoded.

    Compared against the feature sibling at the same keywords, so a parameter creeping in here --
    a second adapter, a learned phase, a floor embedding -- fails rather than being absorbed into
    a total nobody re-derives.
    """
    causal = build(tiny_kwargs)
    torch.manual_seed(0)
    two_sided = SeqVaeLagAttnFs(**dict(tiny_kwargs))

    causal_shapes = {name: tuple(p.shape) for name, p in causal.named_parameters()}
    sibling_shapes = {name: tuple(p.shape) for name, p in two_sided.named_parameters()}
    assert causal_shapes == sibling_shapes
    assert sum(p.numel() for p in causal.parameters()) == sum(
        p.numel() for p in two_sided.parameters()
    )


# =================================================================================================
# The refusals
# =================================================================================================
def test_a_warm_up_without_its_keep_index_is_refused(tiny_kwargs) -> None:
    """Unpaired, both gates stay ``None`` and the target's vector would route into both streams."""
    with pytest.raises(ValueError, match="target_keep_index"):
        SeqVaeLagAttnCfs(**dict(tiny_kwargs, target_warmup_steps=TINY_TARGET_WARMUP_STEPS))
    with pytest.raises(ValueError, match="source_keep_index"):
        SeqVaeLagAttnCfs(**dict(tiny_kwargs, source_warmup_steps=TINY_SOURCE_WARMUP_STEPS))


def test_an_anchor_floor_below_the_kept_channels_budget_is_refused(tiny_warmup) -> None:
    r"""The budget-and-floor pairing, enforced rather than assumed.

    A floor one step too low scores the slowest kept channel's assumed pre-recording history with
    every shape correct -- the objective's mask is $(B, A, H)$ and broadcasts over channels, so
    nothing about it depends on which channel is honest when.
    """
    budget = max(TINY_TARGET_WARMUP_STEPS)
    with pytest.raises(ValueError) as error:
        SeqVaeLagAttnCfs(**dict(tiny_warmup, warmup_period=budget - 2))
    message = str(error.value)
    assert f"warmup_period={budget - 2}" in message
    assert str(budget) in message and str(budget - 1) in message

    # Exactly B - 1 is admitted: an anchor reads target step t + 1 at the earliest.
    assert build(dict(tiny_warmup, warmup_period=budget - 1)) is not None


def test_a_stride_outside_the_horizon_is_refused_naming_it(tiny_warmup) -> None:
    """Above $H$ the decoded windows leave gaps no phase ever covers; below $1$ there is no set."""
    horizon = int(tiny_warmup["horizon"])
    for stride in (0, -1, horizon + 1):
        with pytest.raises(ValueError, match=str(stride)):
            SeqVaeLagAttnCfs(**dict(tiny_warmup, anchor_stride=stride))
    assert build(dict(tiny_warmup, anchor_stride=horizon)) is not None


def test_a_stride_wider_than_the_anchor_span_is_refused(tiny_warmup) -> None:
    """At the last phase the first anchor would not exist, and the sample would decode nothing."""
    model = build(tiny_warmup)
    span = model.geometry.t_valid - model.warmup_period
    with pytest.raises(ValueError, match="anchor_stride"):
        SeqVaeLagAttnCfs(
            **dict(tiny_warmup, warmup_period=model.geometry.t_valid - 1, anchor_stride=2)
        )
    assert span >= int(tiny_warmup["horizon"]), "the tiny geometry admits the shipped stride"


def test_a_negative_lag_floor_is_refused_naming_it(tiny_warmup) -> None:
    """It would admit lags reading before the start of the sequence."""
    with pytest.raises(ValueError, match="-3"):
        SeqVaeLagAttnCfs(**dict(tiny_warmup, lag_floor=-3))


# =================================================================================================
# The channel alignment
#
# The gate stops being a pure gather. Three things follow, and exactly one of them is loud: the
# widths move (loud), the anchor floor moves (loud, refused by name), and the availability mask has
# to move with the shift (SILENT -- it would announce a channel warm by up to its own shift before
# it is, with no crash, no shape change and no metric moving). The last is what most of the block
# below is about.
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
    # And neither adapter has a start-of-record token, because the warm-up alone reaches zero.
    assert without.target_adapter.start_embed is None
    assert without.source_adapter.start_embed is None


def test_the_shift_reaches_the_gate_and_nothing_else_moves_the_keep_index(tiny_align) -> None:
    """The keywords are renamed on the way to the base, so the gate is where they must land."""
    model = build(tiny_align)

    assert tuple(int(value) for value in model.target_gate.delay.delay_steps) == (
        TINY_TARGET_ALIGN_DELAYS
    )
    assert tuple(int(value) for value in model.source_gate.delay.delay_steps) == (
        TINY_SOURCE_ALIGN_DELAYS
    )
    assert tuple(int(value) for value in model.target_gate.keep_index) == TINY_TARGET_KEEP_INDEX
    assert model.target_gate.out_channels == len(TINY_TARGET_KEEP_INDEX)


def test_the_adapter_is_told_the_warm_up_plus_the_shift(tiny_align) -> None:
    r"""The silent one. A gathered-and-delayed channel is honest only once the step index has
    reached **both** $W'_c$ and $d_c$, so the vector the mask and the announcement are built from is
    the sum. Fed the warm-up alone, the adapter would call a channel warm $d_c$ steps early and
    every shape, every metric and every gradient would be exactly as they are now."""
    model = build(tiny_align)
    combined = tuple(
        wait + shift
        for wait, shift in zip(TINY_TARGET_WARMUP_STEPS, TINY_TARGET_ALIGN_DELAYS)
    )

    assert model.target_adapter.max_delay == max(combined)
    assert model.target_adapter.min_delay == min(combined)
    # The pattern itself, which is what the forward multiplies by.
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
    r"""A construction-time change that no shipped configuration has ever made before.

    ``AvailabilityInputAdapter`` builds ``start_embed`` when $\min_c \delta_c > 0$ -- when *every*
    channel of the stream is still absent for some leading steps, so the "nothing has arrived yet"
    indicator is non-zero somewhere. Unaligned, both streams have a channel at $W' = 0$ and the
    token is permanently inert, so it is not built. Under the shift the minimum of $W'_c + d_c$
    lifts off zero on both streams and the token comes into existence: a new learned parameter of
    width $d_{\mathrm{model}}$ per stream, and a live token in the forward pass. This is wanted,
    and it must be asserted rather than discovered in a parameter total.
    """
    off = build(tiny_warmup)
    on = build(tiny_align)

    assert off.target_adapter.start_embed is None and off.source_adapter.start_embed is None
    for adapter in (on.target_adapter, on.source_adapter):
        assert adapter.start_embed is not None
        assert adapter.start_embed.shape == (int(on.d_model),)
        assert bool(adapter.start_indicator.any()), "an inert token is not a token"

    # And it is the only structural addition: two vectors of the model width, nothing else.
    assert sum(p.numel() for p in on.parameters()) - sum(
        p.numel() for p in off.parameters()
    ) == 2 * int(on.d_model)


def test_the_anchor_floor_rises_to_the_shifted_warmth(tiny_align) -> None:
    r"""Both requirements, and they do not move together.

    The scored target is never shifted, so its half stays at $F \ge B - 1$. The *inputs* are, and an
    aligned channel vector at step $t$ asserts one physical instant -- an assertion that is false,
    not partially true, while any entry has not arrived. So the floor must clear
    $\max_c(W'_c + d_c)$, which costs exactly one anchor here as it does at the shipped reference.
    """
    floor = int(tiny_align["warmup_period"])
    assert floor == max(TINY_TARGET_WARMUP_STEPS), "the flat combined vector is what this pins"

    with pytest.raises(ValueError) as error:
        SeqVaeLagAttnCfs(**dict(tiny_align, warmup_period=floor - 1))
    message = str(error.value)
    assert "shifted inputs" in message
    assert f"warmup_period={floor - 1}" in message and f"at least {floor}" in message

    assert build(dict(tiny_align, warmup_period=floor)) is not None


def test_the_unaligned_floor_is_unchanged_by_the_second_requirement(tiny_warmup) -> None:
    r"""The negative control for the test above, and the reason the two cases are kept apart.

    With no shift the input at step $t$ *is* the stored coefficient at $t$: a cold one is masked and
    announced, which is the policy this family ships, and it is why $F = B - 1$ is admitted with the
    slowest channel still cold at the anchor itself. A check that applied the input-warmth half
    unconditionally would refuse the shipped configuration.
    """
    budget = max(TINY_TARGET_WARMUP_STEPS)
    assert build(dict(tiny_warmup, warmup_period=budget - 1)) is not None
    # And a zero-filled shift vector is the same case, not a shift that ran: the constructor hands
    # the gate's own delays to the check, and an unaligned gate's are all zeros.
    assert build(
        dict(
            tiny_warmup,
            warmup_period=budget - 1,
            target_align_delays=tuple(0 for _ in TINY_TARGET_WARMUP_STEPS),
            source_align_delays=tuple(0 for _ in TINY_SOURCE_WARMUP_STEPS),
        )
    ) is not None


def test_a_negative_shift_is_refused_by_name(tiny_align) -> None:
    """A negative shift reads a channel from its own future, which is the property the whole causal
    construction exists for. Refused by ``ChannelDelay``, which the alignment reaches through, so
    the message names the offending entries rather than a shape."""
    shifts = list(TINY_TARGET_ALIGN_DELAYS)
    shifts[2] = -1
    with pytest.raises(ValueError, match="own future"):
        SeqVaeLagAttnCfs(**dict(tiny_align, target_align_delays=tuple(shifts)))


def test_the_old_delay_keywords_are_still_refused(tiny_align) -> None:
    """The alignment gaining its own names does not reopen the two that were removed: those carry
    the two-sided reach guard, measured on a bank that did not produce these coefficients."""
    for banned in ("target_delays", "source_delays"):
        with pytest.raises(TypeError, match=banned):
            SeqVaeLagAttnCfs(**dict(tiny_align, **{banned: TINY_TARGET_ALIGN_DELAYS}))


# =================================================================================================
# The revision's five switches, and the off-state of each
#
# Every mechanism this family added is a keyword whose off-state must reproduce the model that was
# trained before it existed -- not approximately, and not "equivalently": bitwise, key for key. That
# is what makes an arm comparable to a record, and it is what a checkpoint written under one
# setting and read under another silently violates. So the off-states are pinned against a model
# built with none of the keywords at all, which is the object graph an old saved kwargs dict
# actually produces.
#
# Two of the five cannot be pinned by construction and are pinned where they live instead. The
# horizon weighting is a loss-side switch, so its null state is an equality between loss VALUES and
# is asserted in the objective owner's own file; the flat lag bias moves a seeded value rather than
# a structure, so what is asserted here is the shape and the seeding rather than an absence.
# =================================================================================================
#: The five keywords the revision added to this constructor, at their off-values. Written out
#: rather than derived from the signature's defaults: comparing the defaults against themselves
#: would pass on any edit, and what has to hold is that these particular values are the ones that
#: reproduce the pre-revision model.
_SWITCHES_OFF = dict(
    lag_kv_source="encoder",
    prior_availability_input=False,
    persistence_residual=False,
    horizon_weight_halflife_steps=None,
    alibi_slope_scale=1.0,
)


def test_the_switch_defaults_are_the_off_values() -> None:
    """The defaults themselves, read off the signature. A default that moved would make every
    "bitwise off" claim below true of a model nobody builds -- the configs all set these keys
    explicitly, so the constructor's default is what an old checkpoint's kwargs dict falls back to
    and is the only thing standing between it and a different architecture."""
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(SeqVaeLagAttnCfs.__init__).parameters.items()
        if name in _SWITCHES_OFF
    }

    assert defaults == _SWITCHES_OFF


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


@pytest.mark.parametrize(
    "keyword, absent",
    [
        ("prior_availability_input", "prior_head.clock_proj.weight"),
        ("persistence_residual", "decoder.persistence_weight"),
    ],
)
def test_an_off_switch_builds_no_parameter_at_all(tiny_warmup, keyword, absent) -> None:
    """Absent rather than present-and-zero, and the difference is a distributed run's.

    A parameter built and left inert has no gradient path, which is what
    ``find_unused_parameters=False`` refuses -- so an off switch that built its tensor anyway would
    turn every multi-GPU run of the baseline into an error, or into a flag that tolerates real dead
    parameters as well. Both directions are checked: absent when off, present when on.
    """
    off = build(dict(tiny_warmup, **{keyword: False}))
    on = build(dict(tiny_warmup, **{keyword: True}))

    assert absent not in dict(off.named_parameters())
    assert absent in dict(on.named_parameters())


def test_the_horizon_weight_is_a_non_persistent_buffer_or_nothing(tiny_warmup) -> None:
    r"""Null builds no buffer; a half-life builds one that a checkpoint does not carry.

    Non-persistent is the load-bearing half. The weight is $(H,)$, so a persistent one would put
    the horizon into the state dict and make a checkpoint unloadable at any other horizon -- for a
    tensor that is a pure function of two numbers the constructor already has.
    """
    off = build(dict(tiny_warmup, horizon_weight_halflife_steps=None))
    on = build(dict(tiny_warmup, horizon_weight_halflife_steps=5.0))

    # Not registered at all when unset, rather than registered as None: the delegation sites read
    # it with a default, so an absent buffer is what "no weighting" is spelled as.
    assert "horizon_weight" not in dict(off.named_buffers())
    assert "horizon_weight" in dict(on.named_buffers())
    assert on.horizon_weight.shape == (on.horizon,)
    assert float(on.horizon_weight.sum()) == pytest.approx(float(on.horizon), rel=1e-6)
    assert not any("horizon_weight" in name for name in on.state_dict())


def test_the_flat_lag_bias_seed_is_a_learnable_parameter_of_the_shipped_shape(
    tiny_warmup,
) -> None:
    r"""The one switch whose off-state is a *value* rather than a structure, pinned as such.

    ``alibi_slope_scale`` multiplies the seed of ``lag_score_bias``. At $0$ the seed is flat and at
    $1$ it is $-s_h \ell$, and **both build the same learnable $(M, L)$ parameter** -- which is the
    whole reason the flat arm is this key rather than ``lag_bias_init: 'normal'``, a mode that
    registers no bias parameter at all. So what is asserted is that the two arms differ in the
    tensor's values and in nothing else about it.
    """
    # `lag_bias_init` is named explicitly because the two are one decision and the tiny fixture
    # leaves the mode at the constructor's default: the seed only exists under `alibi_decay`, so a
    # scale set against any other mode reaches nothing.
    flat = build(dict(tiny_warmup, lag_bias_init="alibi_decay", alibi_slope_scale=0.0))
    decaying = build(dict(tiny_warmup, lag_bias_init="alibi_decay", alibi_slope_scale=1.0))
    normal = build(dict(tiny_warmup, lag_bias_init="normal"))

    seed_flat = dict(flat.named_parameters())["lag_attn.lag_score_bias"]
    seed_decay = dict(decaying.named_parameters())["lag_attn.lag_score_bias"]

    assert seed_flat.shape == seed_decay.shape
    assert seed_flat.requires_grad and seed_decay.requires_grad
    assert torch.equal(seed_flat, torch.zeros_like(seed_flat)), "the flat seed is not flat"
    # Monotone in the lag on every head, which is what makes it a prior for lag 0.
    assert torch.all(seed_decay[:, :-1] > seed_decay[:, 1:])
    # And the mode that is NOT the flat arm, stated so the two cannot be confused: `normal` builds
    # no bias parameter at all, so it is a third arm about a different object.
    assert "lag_attn.lag_score_bias" not in dict(normal.named_parameters())


# =================================================================================================
# The lag attention's key/value memory
# =================================================================================================
def test_an_unknown_kv_source_is_refused_naming_the_choices(tiny_warmup) -> None:
    """By name, with the admitted set in the message. The value reaches a branch that would
    otherwise fall through to one of the arms, so an unrecognised string would silently train the
    fall-through arm under the misspelt one's name."""
    with pytest.raises(ValueError, match=r"lag_kv_source must be one of"):
        build(dict(tiny_warmup, lag_kv_source="conv-stem"))


@pytest.mark.parametrize("arm", ["conv_stem", "adapter"])
def test_a_local_kv_arm_does_not_build_the_deep_source_encoder(tiny_warmup, arm) -> None:
    """The deep encoder leaves the *model*, not just the lag path.

    Nothing else consumes the source state, so under a local arm the encoder would be a whole stack
    of parameters no forward reaches -- the same distributed hazard as an inert switch, at hundreds
    of thousands of parameters rather than one matrix. Asserted by state-dict prefix rather than by
    a total, because a total cannot say *which* stack went.
    """
    deep = build(dict(tiny_warmup, lag_kv_source="encoder"))
    local = build(dict(tiny_warmup, lag_kv_source=arm))

    assert [name for name in deep.state_dict() if name.startswith("source_encoder.")]
    assert [name for name in local.state_dict() if name.startswith("source_encoder.")] == []
    assert getattr(local, "source_encoder", None) is None


def test_the_conv_stem_arm_builds_a_stem_and_the_adapter_arm_builds_nothing(
    tiny_warmup,
) -> None:
    """What each local arm puts in the encoder's place, and what resolves the arm.

    ``source_kv_body`` is the single place the arm becomes a module, and every consumer -- the
    forward, both source controls, the prior clock, the norm-causalisation guard -- goes through it
    or through ``encode_source_kv``. Pinning it here is pinning that they cannot disagree.
    """
    stem = build(dict(tiny_warmup, lag_kv_source="conv_stem"))
    adapter = build(dict(tiny_warmup, lag_kv_source="adapter"))
    deep = build(dict(tiny_warmup, lag_kv_source="encoder"))

    assert stem.source_kv_body() is stem.source_kv_stem
    assert adapter.source_kv_body() is None
    assert deep.source_kv_body() is deep.source_encoder

    # The adapter's own output IS the representation there, so the pathway is one module.
    assert adapter.source_kv_modules() == (adapter.source_adapter,)
    assert stem.source_kv_modules() == (stem.source_adapter, stem.source_kv_stem)


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
