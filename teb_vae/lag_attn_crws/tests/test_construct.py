r"""The model is built, its inputs begin where the warm-up says they do, and its decoder is raw.

This cell composes one mixin over the raw-signal architecture and holds nothing but a constructor,
which is the one member that cannot be shared: the experiment driver builds a run's kwargs by
sweeping ``inspect.signature(MODEL_CLS.__init__)``, so the signature must be the architecture's with
exactly the two delay keywords and the declined ``persistence_residual`` swapped for this domain's
own, and every other keyword must reach the base.

What the construction must produce: the target-stream gate keeps the budget's channels and the input
adapter is built at that width while the decoder stays $R$ wide; the adapter is built at the warm-up
(plus the alignment shift, which the constructor renames onto the gate) rather than at the gate's
own delays; no gradient flows from inside the warm-up; the ungated model is parameter for parameter
the raw-signal sibling; the off-values of the alignment and of the revision's switches rebuild the
model built without them, which is what an old checkpoint's kwargs exercise; and the anchor floor
rises to the shifted warmth under an alignment.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_inputs import FORWARDED_EXCLUSIONS
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

from .conftest import (
    CAUSAL_ST_WIDTH,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
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


# =================================================================================================
# The signature the driver sweeps
# =================================================================================================
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
    # And no catch-all: a ``**kwargs`` signature would hide every keyword from the sweep.
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(SeqVaeLagAttnCrws.__init__).parameters.values()
    )


def test_a_base_keyword_no_other_test_names_still_reaches_the_base(tiny_kwargs) -> None:
    """The behavioural half of the claim above, on the two keywords a hand-written forward dict
    would be likeliest to drop: neither shapes anything, so neither would fail loudly."""
    model = build(dict(tiny_kwargs, coverage_floor=0.25, base_decode="mean"))

    assert model.coverage_floor == 0.25
    assert model.base_decode == "mean"


# =================================================================================================
# The guard the constructor builds
# =================================================================================================
def test_the_target_stream_gate_keeps_the_channels_the_budget_kept(tiny_warmup) -> None:
    """And the adapter is built at that width -- while the *decoder* is not, which is the whole
    difference between this cell and the causal-feature one."""
    model = build(tiny_warmup)

    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(TINY_TARGET_KEEP_INDEX)
    assert model.target_adapter.in_dim == len(TINY_TARGET_KEEP_INDEX)
    assert model.decoder.mean_head.out_features == model.raw_per_step
    assert model.raw_per_step != len(TINY_TARGET_KEEP_INDEX)


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
# The ungated arm, and the off-values an old checkpoint rebuilds at
# =================================================================================================
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


def test_the_off_values_rebuild_the_model_built_without_the_keywords(tiny_warmup) -> None:
    """The path an old checkpoint's saved kwargs dict actually exercises.

    A checkpoint written before the alignment keywords and the revision's switches existed carries
    none of them, so construction from it must produce the object graph and the tensor values of
    the model that was trained. Compared over the state dict *and* the buffer names, because a
    non-persistent buffer is invisible to a state-dict comparison; and with no identity
    ``ChannelDelay`` in the unaligned gate, because an identity shift is numerically invisible and
    structurally is not.
    """
    without = build(tiny_warmup)
    explicit = build(
        dict(tiny_warmup, target_align_delays=None, source_align_delays=None, **_SWITCHES_OFF)
    )

    assert list(without.state_dict()) == list(explicit.state_dict())
    for name, tensor in without.state_dict().items():
        assert torch.equal(tensor, explicit.state_dict()[name]), name
    assert sorted(dict(without.named_buffers())) == sorted(dict(explicit.named_buffers()))
    assert without.target_gate.max_delay == 0
    assert without.source_delay_steps == 0


# =================================================================================================
# The channel alignment
# =================================================================================================
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
