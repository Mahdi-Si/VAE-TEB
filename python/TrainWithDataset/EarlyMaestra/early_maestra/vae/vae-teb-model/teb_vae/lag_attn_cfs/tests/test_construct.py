r"""The model is built, and its inputs begin where the warm-up says they do.

The claim this package exists to make is that no coefficient the model reads was computed from
anything but the past. Half of that is the dataset's; the other half is here, and it is easy to get
silently wrong in three ways.

**The warm-up region is not empty.** A one-sided filter emits real float values before $W'_c$, from
assumed pre-recording history, and the normalisation constants were accumulated while deliberately
excluding exactly that region -- so those numbers are on no defined scale, and a model that reads
them is training on pad with nothing raising.

**The gate does not mask it.** This family's ``ChannelGate`` is built with ``delays=None``, a pure
gather, so ``gate.delay.delay_steps`` is all zeros. A model that built its adapter from that -- as
the base does, correctly, for a reach-budget guard -- would get ``max_delay = 0`` and **neither**
availability term: no mask, no announcement, and the encoder would see the region as signal.

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
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)
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
    assert defaults["warmup_period"] == 133
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


def test_the_shipped_budget_builds_neither_start_token() -> None:
    """At the real budget both minima are $0$, and a change to either must be visible here."""
    kwargs = shipped_warmup_kwargs()
    assert min(kwargs["target_warmup_steps"]) == 0
    assert min(kwargs["source_warmup_steps"]) == 0

    model = build(kwargs)
    assert model.target_adapter.start_embed is None
    assert model.source_adapter.start_embed is None
    assert model.target_gate is not None and model.target_gate.out_channels == 98
    assert model.decoder_out_channels == 98


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
