r"""Self-attention over the horizon tokens: inert by default, and correct when it is on.

The shared horizon core is the one module both future decoders run, and it is invoked **twice per
forward** -- once on $z^p$ and once on $z^q$. That makes every property here a correctness
boundary rather than a preference:

* a core that was not asked for attention must be the core that existed before the knob did, or
  the untouched feature sibling's numbers move for no stated reason;
* the attention must be deterministic in train mode, or the base-minus-full readout picks up noise
  that has nothing to do with the source;
* the blocks must not mix anchors, or an anchor's forecast reads a neighbour it is not conditioned
  on;
* every attention parameter must reach the graph, or a DDP run hangs waiting for a gradient.

The identity checks are exact (``torch.equal``) rather than tolerant: they are the same
computation on the same weights, and anything less than equality would be evidence that the
attention is not the pure residual its placement claims.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.decoders import HorizonDecoderCore

#: A tiny but faithful geometry: the width divides by the head count and the horizon is long
#: enough that "every token attends to every other" is not the same as "attends to itself".
_D_HIDDEN = 16
_HORIZON = 6
_HEADS = 4
_DEPTH = 2


def _core(**overrides) -> HorizonDecoderCore:
    """Build a core at the fixed geometry, seeded so two builds are comparable."""
    kwargs = dict(
        d_hidden=_D_HIDDEN, horizon=_HORIZON, depth=_DEPTH, attention_heads=_HEADS
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return HorizonDecoderCore(**kwargs)


def _state(batch: int = 3, seq_len: int = 4, *, seed: int = 1) -> torch.Tensor:
    """A projected decoder state ``(B, T, d_hidden)``."""
    return torch.randn(
        batch, seq_len, _D_HIDDEN, generator=torch.Generator().manual_seed(seed)
    )


# ---------------------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------------------
def test_the_default_core_carries_no_attention_state_dict_key():
    """A checkpoint written by the default core must load into a core built before this knob
    existed, which is only true if the key set did not grow."""
    core = _core()
    keys = list(core.state_dict())

    assert core.attention is None
    assert keys, "the core has no parameters at all; this probe is vacuous"
    assert [key for key in keys if "attention" in key] == []


def test_a_width_that_no_attention_will_see_is_not_held_to_the_head_constraint():
    """The divisibility rule belongs to the attention, not to the core. Existing geometries -- the
    oracle probe's narrow core among them -- must keep constructing at whatever width they use."""
    torch.manual_seed(0)
    core = HorizonDecoderCore(d_hidden=6, horizon=_HORIZON, depth=1, attention_heads=_HEADS)

    assert core.attention is None


# ---------------------------------------------------------------------------------------
# What turning it on costs and produces
# ---------------------------------------------------------------------------------------
def test_the_blocks_add_only_their_own_keys_and_leave_the_rest_of_the_core_unchanged():
    """Every blockless key survives at its own shape, so a block that quietly widened something
    else fails here, and a blockless checkpoint's weights still fit an attended core."""
    blockless = _core().state_dict()
    attended = _core(attention_blocks=2).state_dict()

    assert set(blockless) <= set(attended)
    for key, value in blockless.items():
        assert attended[key].shape == value.shape, key
    added = set(attended) - set(blockless)
    assert added and all(key.startswith("attention.") for key in added)


def test_the_decode_shape_is_the_one_the_decoders_expect():
    core = _core(attention_blocks=2)
    state = _state()

    out = core.decode(state)

    assert out.shape == (state.shape[0], state.shape[1], _HORIZON, _D_HIDDEN)


@pytest.mark.parametrize(
    "heads, message",
    [(3, r"attention_heads=3.*d_hidden=16"), (0, "attention_heads=0")],
    ids=["indivisible", "zero"],
)
def test_a_bad_head_count_is_refused_naming_the_values(heads, message):
    """An indivisible count names both values; a zero one is refused rather than divided by."""
    with pytest.raises(ValueError, match=message):
        _core(attention_blocks=1, attention_heads=heads)


# ---------------------------------------------------------------------------------------
# The invariants the twice-invoked decoder depends on
# ---------------------------------------------------------------------------------------
def test_two_train_mode_decodes_are_bitwise_equal():
    """The property one module invoked twice must have. Seeded *differently* between the two
    calls, so a stochastic path would have to produce the same numbers from two RNG states."""
    core = _core(attention_blocks=2).train()
    state = _state()

    torch.manual_seed(11)
    first = core.decode(state)
    torch.manual_seed(22)
    second = core.decode(state)

    assert torch.equal(first, second)


def test_no_anchor_reads_another_anchors_horizon():
    """The isolation the fold into the batch is supposed to give. One anchor's state is perturbed;
    every other anchor's forecast must be bit-for-bit what it was."""
    core = _core(attention_blocks=2).eval()
    state = _state()

    with torch.no_grad():
        before = core.decode(state)
        moved = state.clone()
        moved[1, 2] += 5.0
        after = core.decode(moved)

    assert not torch.equal(before[1, 2], after[1, 2]), "the perturbation did nothing; probe vacuous"
    for batch in range(state.shape[0]):
        for step in range(state.shape[1]):
            if (batch, step) != (1, 2):
                assert torch.equal(before[batch, step], after[batch, step]), (batch, step)


def test_gradient_reaches_every_attention_parameter():
    """Reachability under ``find_unused_parameters=False``, checked where the parameters live: a
    block whose gain started at exactly zero would leave its four projections with a zeros
    gradient, so the assertion is that each is genuinely on the graph *and* moving."""
    core = _core(attention_blocks=2)

    core.decode(_state()).pow(2).sum().backward()

    assert core.attention is not None
    for index, block in enumerate(core.attention):
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, f"attention.{index}.{name} received no gradient"
            assert float(parameter.grad.abs().sum()) > 0.0, f"attention.{index}.{name} is inert"


# ---------------------------------------------------------------------------------------
# Placement: a pure residual before the untouched output norm
# ---------------------------------------------------------------------------------------
def _blockless_twin(core: HorizonDecoderCore) -> HorizonDecoderCore:
    """A blockless core holding ``core``'s shared weights; the attention keys have no counterpart."""
    twin = HorizonDecoderCore(d_hidden=_D_HIDDEN, horizon=_HORIZON, depth=_DEPTH)
    twin.load_state_dict(core.state_dict(), strict=False)
    return twin


def test_at_zero_gain_the_decode_is_exactly_the_blockless_one():
    """What "each block is its own residual, and the skip and output norm are untouched" means
    numerically. If the blocks had been inserted before the skip was taken, or inside the
    ``out_norm(feat + skip)`` composition, zeroing the gains would not recover this."""
    core = _core(attention_blocks=2)
    assert core.attention is not None
    with torch.no_grad():
        for block in core.attention:
            block.residual_gain.zero_()

    state = _state()
    with torch.no_grad():
        assert torch.equal(core.decode(state), _blockless_twin(core).decode(state))


def test_at_its_own_initialisation_the_attention_changes_the_forecast():
    """The negative control for the test above: the gain is initialised small so the stack starts
    *near* identity, not at it. Without this, a stack that never ran would pass."""
    core = _core(attention_blocks=2)
    state = _state()

    with torch.no_grad():
        assert not torch.equal(core.decode(state), _blockless_twin(core).decode(state))
