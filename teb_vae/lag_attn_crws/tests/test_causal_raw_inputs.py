r"""The two members this mixin overrides: the floor policy and the source-block readout constants.

:class:`~teb_vae.lag_attn_crws.nets.causal_raw_inputs.CausalRawInputs` replaces two members of
:class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs`, and each is checked for what it
computes:

* ``_check_anchor_floor`` enforces $F \ge \max(B - 1, \max_c(W'_c + d_c))$ over the kept
  target-stream channels, where the second half applies only under a non-zero shift, and names which
  half binds;
* ``_resolve_warmup_readout_constants`` registers the two source-block warmth patterns -- each the
  pattern its own declared channels imply, non-persistent so a checkpoint does not carry a
  budget-shaped tensor -- and an ungated stream is warm at every step.

The bound readouts and the gather arithmetic are exercised through ``compute_loss`` elsewhere.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.nets.causal_raw_inputs import CausalRawInputs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws

from .conftest import (
    TINY_SOURCE_WARMUP_STEPS,
    TINY_STRIDE,
    TINY_TARGET_WARMUP_STEPS,
    build,
    tiny_warmup_kwargs,
)


@pytest.fixture(scope="module")
def model():
    """The tiny guarded model at a real tiling, built once; nothing below mutates it."""
    return build(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)).eval()


# =================================================================================================
# The floor, restated as a policy
# =================================================================================================
@pytest.mark.parametrize(
    "floor,waits,shifts,refusal",
    [
        # No kept channel: an ungated model has no floor to check.
        (0, (), (), None),
        # Unshifted: B - 1 is admitted, one below it is refused by the first half.
        (5, (0, 3, 6), (), None),
        (4, (0, 3, 6), (), ("warm by the first forecast step", "at least 5")),
        # An all-zero shift vector is the unshifted reading, not a satisfied second half.
        (5, (0, 3, 6), (0, 0, 0), None),
        (4, (0, 3, 6), (0, 0, 0), ("warm by the first forecast step", "at least 5")),
        # Shifted: max_c(W'_c + d_c) = 7 binds at kept channel 2, gathered from t - 1.
        (6, (0, 3, 6), (2, 1, 1), ("AT THE ANCHOR", "at least 7", "channel 2", "t - 1")),
        (7, (0, 3, 6), (2, 1, 1), None),
    ],
)
def test_the_floor_policy_binds_at_the_half_the_shift_selects(floor, waits, shifts, refusal):
    r"""$F \ge B - 1$ unshifted, $F \ge \max_c(W'_c + d_c)$ once a shift is non-zero.

    The constructor hands the check the gate's own delays, which are all zero on an unaligned gate,
    so an empty vector and an all-zero one must give the same answer or the shipped unaligned arm
    would be refused. The message names which half binds and the floor it requires.
    """
    if refusal is None:
        assert CausalRawInputs._check_anchor_floor(floor, waits, shifts) is None
        return
    with pytest.raises(ValueError) as error:
        CausalRawInputs._check_anchor_floor(floor, waits, shifts)
    message = str(error.value)
    assert f"warmup_period={floor}" in message
    for fragment in refusal:
        assert fragment in message, fragment


def test_the_floor_refusal_fires_through_the_constructor(tiny_warmup) -> None:
    """Reached from ``_validate_causal_geometry``, which is inherited: the override is the check,
    not the call site, so the stride-versus-span refusal beside it cannot drift."""
    budget = max(TINY_TARGET_WARMUP_STEPS)
    with pytest.raises(ValueError, match="input-warmth policy"):
        SeqVaeLagAttnCrws(**dict(tiny_warmup, warmup_period=budget - 2))

    # Exactly B - 1 is admitted: a forecast at anchor t covers target steps from t + 1.
    build(dict(tiny_warmup, warmup_period=budget - 1))


# =================================================================================================
# What the readout resolution registers
# =================================================================================================
def test_the_patterns_split_the_source_stream_at_its_declared_boundary(model) -> None:
    """Each block's pattern is the one its own channels imply, so a pooled figure cannot hide the
    slower block behind the faster one.

    Non-persistent for the family's reason: their contents follow the resolved budget, so a
    persistent copy would make a checkpoint trained at one budget fail to load at another and
    report it as misaligned keys rather than as a budget mismatch."""
    split = CausalRawInputs.SOURCE_BLOCK_SPLIT
    state = model.state_dict()
    for name, block in (
        ("st", TINY_SOURCE_WARMUP_STEPS[:split]),
        ("ph", TINY_SOURCE_WARMUP_STEPS[split:]),
    ):
        expected = CausalRawInputs._resolve_block_warm_steps(
            [int(step) for step in block], model.sequence_length
        )
        assert torch.equal(getattr(model, f"source_block_warm_{name}"), expected), name
        assert f"source_block_warm_{name}" not in state, name


def test_an_ungated_stream_is_warm_at_every_step(tiny_kwargs) -> None:
    """No warm-up to wait out, so the pattern is all-``True`` -- and it exists, rather than being
    absent, so the readouts have something to normalise against on an unguarded run."""
    model = build(tiny_kwargs)

    for name in ("source_block_warm_st", "source_block_warm_ph"):
        assert bool(getattr(model, name).all()), name
