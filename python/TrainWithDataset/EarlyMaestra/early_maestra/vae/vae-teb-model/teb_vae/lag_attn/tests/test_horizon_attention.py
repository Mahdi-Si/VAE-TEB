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
from torch import nn

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
def test_the_default_core_builds_no_attention_module_at_all():
    """Not a zero-length stack and not a disabled one -- nothing. An empty ``ModuleList`` would
    still put a name in the module tree, and "the core is what it was" is a claim about the tree
    as well as about the numbers."""
    core = _core()

    assert core.attention_blocks == 0
    assert core.attention is None
    assert [name for name, _ in core.named_modules() if "attention" in name] == []


def test_the_default_core_carries_no_attention_state_dict_key():
    """A checkpoint written by the default core must load into a core built before this knob
    existed, which is only true if the key set did not grow."""
    keys = list(_core().state_dict())

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
def test_the_blocks_add_exactly_their_own_parameters_and_nothing_else():
    r"""Per block: a ``LayerNorm`` pair, four square bias-free projections and one residual gain,
    $2d + 4d^2 + 1$. Asserted as a *delta* against the blockless core and as a key-set difference,
    so a block that quietly widened something else fails here."""
    blockless, attended = _core(), _core(attention_blocks=2)

    expected = 2 * (4 * _D_HIDDEN**2 + 2 * _D_HIDDEN + 1)
    delta = sum(p.numel() for p in attended.parameters()) - sum(
        p.numel() for p in blockless.parameters()
    )

    assert delta == expected
    added = set(attended.state_dict()) - set(blockless.state_dict())
    assert added == {
        f"attention.{index}.{name}"
        for index in (0, 1)
        for name in (
            "norm.weight", "norm.bias", "residual_gain",
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight",
        )
    }
    assert set(blockless.state_dict()) - set(attended.state_dict()) == set()


def test_the_projections_are_bias_free():
    """A bias on q/k after a pre-norm shifts every logit by the same constant and buys nothing;
    stated as a structural assertion because the parameter arithmetic above depends on it."""
    core = _core(attention_blocks=1)
    assert core.attention is not None
    block = core.attention[0]

    for projection in (block.q_proj, block.k_proj, block.v_proj, block.out_proj):
        assert projection.bias is None


def test_the_decode_shape_is_the_one_the_decoders_expect():
    core = _core(attention_blocks=2)
    state = _state()

    out = core.decode(state)

    assert out.shape == (state.shape[0], state.shape[1], _HORIZON, _D_HIDDEN)


def test_an_indivisible_head_count_is_refused_naming_both_values():
    with pytest.raises(ValueError, match=r"attention_heads=3.*d_hidden=16"):
        _core(attention_blocks=1, attention_heads=3)


def test_a_zero_head_count_is_refused_rather_than_dividing_by_it():
    with pytest.raises(ValueError, match="attention_heads=0"):
        _core(attention_blocks=1, attention_heads=0)


# ---------------------------------------------------------------------------------------
# The invariants the twice-invoked decoder depends on
# ---------------------------------------------------------------------------------------
def test_the_attention_stack_holds_no_dropout_of_any_kind():
    """Module-level, checked structurally; the functional kind is checked by the determinism test
    below, which is the only way to see a ``dropout_p`` passed to the attention call."""
    core = _core(attention_blocks=2)

    assert [name for name, m in core.named_modules() if isinstance(m, nn.Dropout)] == []


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
