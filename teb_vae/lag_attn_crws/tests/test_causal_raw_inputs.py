r"""What this mixin overrides, what it inherits, and what it merely points at.

:class:`~teb_vae.lag_attn_crws.nets.causal_raw_inputs.CausalRawInputs` is
:class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs` with two members replaced and
five bound by reference, and every one of those three categories fails silently if it slips:

* **An inherited member that quietly became an override** is a second copy of the tiled forward, the
  warm-up adapter or the anchor set, free to drift from the one six shipped cells score through. So
  the six inherited names are asserted *absent* from ``vars()`` **and** identical to that mixin's own
  function objects -- absence alone would pass on a member re-declared with the same body.
* **A bound member that became a copy** is the same failure at the readout level, and the check is
  ``is`` identity rather than sampled equality, because two implementations of one quantity agree on
  the cases a test happens to try.
* **A bound ``staticmethod`` that was not re-wrapped** is the trap this composition actually hit.
  ``Owner.some_staticmethod`` returns the *plain function* -- the descriptor has already resolved --
  and a plain function assigned in a class body becomes an **instance** method, so ``self`` arrives
  as its first positional argument. Identity holds either way and the call fails three frames below.
  That is why every bound callable is *called through an instance* at the arity its owner declares.

The two overrides are checked for what they now say rather than for existing: the floor refusal must
state the **input-warmth** policy, because the validity requirement it inherited is about a target
this cell does not have; and the readout resolution must register the two source patterns and
**nothing else**, because a target warm fraction over a raw block would be a vacuous $1.0$ column and
a tertile assignment over raw samples would partition an axis that has no filter to rank.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

from .conftest import (
    BATCH,
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    tiny_warmup_kwargs,
)

#: Everything this mixin declares, by name. A set rather than a count: a member added here is one
#: the conv-Transformer cell composing the same object silently gets too, and a member removed is a
#: behaviour that fell back to the causal-feature cells' target-coupled version.
_OWN_MEMBERS = {
    "_check_anchor_floor",
    "_resolve_warmup_readout_constants",
    "compute_loss",
    "SOURCE_BLOCK_SPLIT",
    "TARGET_BLOCK_SPLIT",
    "_resolve_block_warm_steps",
    "_anchors_per_sample",
    "_source_lag_warmth",
}

#: The members that must still be the parent mixin's own function objects. Each is target-domain-free
#: -- these cells take the identical three input tensors -- so a copy here would be a second
#: definition of machinery the causal-feature cells are the reference for.
_INHERITED = (
    "_set_causal_inputs",
    "_build_adapter",
    "build_lag_mask",
    "_build_anchor_index",
    "forward",
    "_validate_causal_geometry",
)

#: The five members bound from the causal-feature target mixin, and the arity each owner declares.
#: The arity is what catches an unwrapped ``staticmethod``; the identity beside it is what catches a
#: copy. Neither alone is enough.
_BOUND_CALLABLES = ("_resolve_block_warm_steps", "_anchors_per_sample", "_source_lag_warmth")
_BOUND_CONSTANTS = ("SOURCE_BLOCK_SPLIT", "TARGET_BLOCK_SPLIT")


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model at a real tiling, built once; nothing below mutates it."""
    return build(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()


@pytest.fixture(scope="module")
def outputs(model):
    """One forward, so the two bound readouts can be called with real arguments."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*make_streams(kwargs), 1)
    target = torch.zeros(
        BATCH, out["anchor_index"].shape[1], model.horizon, model.geometry.r
    )
    return out, target


# =================================================================================================
# What the class declares
# =================================================================================================
def test_the_mixin_declares_exactly_the_members_this_target_changes() -> None:
    """Set equality, so neither an addition nor a removal passes."""
    own = {name for name in vars(CausalRawInputs) if not name.startswith("__")}

    assert own == _OWN_MEMBERS


def test_the_mixin_extends_the_causal_input_half_rather_than_restating_it() -> None:
    """A sibling relationship rather than a copy: the parent is the object six cells score through."""
    assert CausalRawInputs.__bases__ == (CausalWarmupInputs,)
    assert CausalRawInputs.__mro__ == (CausalRawInputs, CausalWarmupInputs, object)


@pytest.mark.parametrize("name", _INHERITED)
def test_an_inherited_member_is_the_parents_own_function_object(name: str) -> None:
    """Absent from ``vars()`` *and* identical by object.

    Absence alone would pass on a member re-declared with the same body under another name, and
    identity alone would pass on a member shadowed by an equal-looking copy -- so both.
    """
    assert name not in vars(CausalRawInputs), name
    assert getattr(CausalRawInputs, name) is vars(CausalWarmupInputs)[name]


# =================================================================================================
# The five bound members
# =================================================================================================
@pytest.mark.parametrize("name", _BOUND_CALLABLES + _BOUND_CONSTANTS)
def test_each_bound_member_is_the_owning_mixins_own_object(name: str) -> None:
    """Identity, not equality. Binding is what makes drift structurally impossible instead of
    merely test-detected: the object here *is* the object there, and the owner never learns it has
    a second consumer."""
    assert getattr(CausalRawInputs, name) is getattr(CausalFeatureForecastTarget, name)


def test_the_bound_source_block_pattern_is_callable_at_its_owners_arity(model) -> None:
    """The ``staticmethod`` trap, as a call rather than as a comment.

    Unwrapped, ``_resolve_block_warm_steps`` binds as an instance method and ``self`` arrives where
    ``block_warmup_steps`` belongs -- so it would fail on ``len()`` of a model, three frames below
    the class body that caused it, with the identity assertion above still green.
    """
    pattern = model._resolve_block_warm_steps([0, 2, 4], model.sequence_length)

    assert tuple(pattern.shape) == (model.sequence_length,)
    assert pattern.dtype == torch.bool
    # And the descriptor really is a staticmethod on this class rather than a plain function.
    assert isinstance(vars(CausalRawInputs)["_resolve_block_warm_steps"], staticmethod)
    assert list(inspect.signature(CausalRawInputs._resolve_block_warm_steps).parameters) == [
        "block_warmup_steps",
        "sequence_length",
    ]


def test_the_two_bound_readouts_are_callable_at_their_owners_arity(model, outputs) -> None:
    """These two take ``self``, so they bind as they are -- and the paired assertion is that they
    were *not* wrapped, which would swallow the first real argument instead."""
    out, target = outputs

    anchors_per_sample = model._anchors_per_sample(out, target)
    warmth = model._source_lag_warmth(out, target)

    assert anchors_per_sample.dim() == 0
    assert set(warmth) == {"source_lag_warmth_frac_st", "source_lag_warmth_frac_ph"}
    for name in _BOUND_CALLABLES[1:]:
        assert not isinstance(vars(CausalRawInputs)[name], staticmethod), name


# =================================================================================================
# The floor, restated as a policy
# =================================================================================================
def test_the_floor_refusal_states_the_input_warmth_policy() -> None:
    r"""$F \ge B - 1$ over the kept **target-stream** channels, and a message that says so.

    The inequality is the causal-feature cells' and what it enforces is not: there a lower floor
    scores a stored coefficient's assumed pre-recording history as signal, and here the target is a
    raw sample that is honest at every step. A message inherited unchanged would tell an operator
    the objective is invalid when it is merely reading colder inputs than the run claims.

    **The stream is named, and that is what this asserts.** The check is fed
    ``target_warmup_steps`` alone -- see the source-stream test below for the numbers -- so a
    message reading "every kept *input* channel" would send an operator to compute the floor from
    the wrong set. It is asserted as an absent substring as well as a present one, because the
    wider sentence is the one the message is most likely to drift back into.
    """
    with pytest.raises(ValueError) as error:
        CausalRawInputs._check_anchor_floor(4, (0, 3, 6))
    message = str(error.value)

    assert "warmup_period=4" in message
    assert "every kept TARGET-STREAM input channel is warm by the first forecast step" in message
    assert "every kept input channel is warm" not in message
    assert "6" in message and "5" in message


def test_the_floor_refusal_says_the_source_stream_is_not_covered() -> None:
    """The half of the policy an operator acts on, and the only one that is falsifiable here.

    The source stream is never gated, so it keeps channels far colder than any floor this cell
    ships -- and the refusal names that explicitly, together with the columns that measure the
    residual, so "the message told me to raise the floor" cannot become the reason a run decodes
    eight anchors instead of a hundred and fifty-two.
    """
    with pytest.raises(ValueError) as error:
        CausalRawInputs._check_anchor_floor(4, (0, 3, 6))
    message = str(error.value)

    assert "source stream is never gated" in message
    assert "source_lag_warmth_frac_st" in message
    assert "do not raise the floor" in message.lower()


def test_the_floor_refusal_fires_through_the_constructor(tiny_warmup) -> None:
    """Reached from ``_validate_causal_geometry``, which is inherited: the override is the check,
    not the call site, so the stride-versus-span refusal beside it cannot drift."""
    budget = max(TINY_TARGET_WARMUP_STEPS)
    with pytest.raises(ValueError, match="input-warmth policy"):
        SeqVaeLagAttnCrws(**dict(tiny_warmup, warmup_period=budget - 2))

    # Exactly B - 1 is admitted: a forecast at anchor t covers target steps from t + 1.
    assert build(dict(tiny_warmup, warmup_period=budget - 1)) is not None


def test_an_ungated_model_has_no_floor_to_check(tiny_kwargs) -> None:
    """No budget, no kept-channel vector, and therefore no policy to state -- the model with no
    guard rather than an identity one."""
    assert CausalRawInputs._check_anchor_floor(0, ()) is None
    assert build(dict(tiny_kwargs, warmup_period=1)) is not None


# =================================================================================================
# What the readout resolution registers
# =================================================================================================
def test_only_the_two_source_patterns_are_resolved(model) -> None:
    """The three target-channel constants are dropped rather than re-pointed.

    A raw block's last axis counts raw samples: they have no warm-up to be past, so
    ``target_warm_frac`` would be a vacuous $1.0$ in every row of every run's CSV, and no filter to
    rank, so the tertile assignment would partition nothing. Both would read as measurements.
    """
    names = dict(model.named_buffers())

    assert "source_block_warm_st" in names and "source_block_warm_ph" in names
    assert "warm_tertile_id" not in names
    assert not hasattr(model, "warm_tertile_id")
    assert not hasattr(model, "target_warm_frac")


def test_the_two_patterns_are_non_persistent_and_shaped_by_the_sequence(model) -> None:
    """Non-persistent for the family's reason: their contents follow the resolved budget, so a
    persistent copy would make a checkpoint trained at one budget fail to load at another and
    report it as misaligned keys rather than as a budget mismatch."""
    state = model.state_dict()

    for name in ("source_block_warm_st", "source_block_warm_ph"):
        pattern = getattr(model, name)
        assert name not in state, name
        assert tuple(pattern.shape) == (model.sequence_length,), name
        assert pattern.dtype == torch.bool, name


def test_the_patterns_split_the_source_stream_at_its_declared_boundary(model) -> None:
    """Each block's pattern is the one its own channels imply, so a pooled figure cannot hide the
    slower block behind the faster one."""
    split = CausalRawInputs.SOURCE_BLOCK_SPLIT
    for name, block in (
        ("st", TINY_SOURCE_WARMUP_STEPS[:split]),
        ("ph", TINY_SOURCE_WARMUP_STEPS[split:]),
    ):
        expected = CausalRawInputs._resolve_block_warm_steps(
            [int(step) for step in block], model.sequence_length
        )
        assert torch.equal(getattr(model, f"source_block_warm_{name}"), expected), name


def test_an_ungated_stream_is_warm_at_every_step(tiny_kwargs) -> None:
    """No warm-up to wait out, so the pattern is all-``True`` -- and it exists, rather than being
    absent, so the readouts have something to normalise against on an unguarded run."""
    model = build(tiny_kwargs)

    for name in ("source_block_warm_st", "source_block_warm_ph"):
        assert bool(getattr(model, name).all()), name
