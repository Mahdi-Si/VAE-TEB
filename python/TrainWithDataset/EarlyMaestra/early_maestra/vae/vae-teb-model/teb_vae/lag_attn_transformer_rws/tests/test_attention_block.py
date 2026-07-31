r"""Causal self-attention: the masks, the kernel against an explicit softmax, and the window edge.

Three things are pinned here that nothing downstream would catch.

The **masks** are checked as tensors, before any attention wraps them, because a mask is the one
place a causality bug is a single comparison operator and is invisible in every loss curve.

The **kernel** is checked against a softmax written out longhand.
``scaled_dot_product_attention`` dispatches to different implementations by backend and by mask
type, and this is the only place in the package where the architecture's own equation and the
kernel's behaviour are compared directly.

The **window boundary** is pinned from both sides. A window that is off by one still passes every
causality test -- it is still causal, just wider or narrower than the receptive-field arithmetic
the source-locality argument rests on.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    CausalSelfAttention,
    CausalTransformerBlock,
    build_causal_window_mask,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import assert_token_causal

#: Geometry the probes run at. $T$ is comfortably longer than the test window so the band has
#: interior rows, and $d$ divides into four even-width heads.
BATCH, SEQ_LEN, D_MODEL, NUM_HEADS, D_FF = 2, 12, 16, 4, 32

#: The test window. Small enough that its boundary sits well inside the sequence.
WINDOW = 4

#: The shipped widths, where the parameter arithmetic is pinned against the architecture's numbers.
SHIPPED_D, SHIPPED_D_FF = 128, 256

#: float32 round-off between a fused kernel and a longhand softmax over $O(1)$ activations.
_KERNEL_TOL = 1e-5


def _sequence(seed: int = 0) -> torch.Tensor:
    """A seeded $(B, T, d)$ activation tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, generator=generator)


def _resample_step(x: torch.Tensor, step: int, *, seed: int = 1) -> torch.Tensor:
    """Return a copy of ``x`` with exactly one step drawn afresh."""
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    perturbed[:, step] = torch.randn(
        x.shape[0], x.shape[2], generator=generator, dtype=x.dtype
    )
    return perturbed


def _reference_attention(module: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Recompute the module's output longhand from its own weights.

    Rebuilds the whole sublayer -- pre-norm, the three projections, the rotary rotation, the mask,
    an explicit softmax, the head merge and the output projection -- rather than reusing anything
    the module computes. So this pins the head layout and the mask semantics, not only the kernel.

    Args:
        module: The attention sublayer under test, in eval mode.
        x: Its input, ``(B, T, d)``.

    Returns:
        The reference output, ``(B, T, d)``.
    """
    batch, seq_len, _ = x.shape
    shape = (batch, seq_len, module.num_heads, module.d_head)
    normed = module.norm(x)

    query = module.rope(module.q_proj(normed).view(shape).transpose(1, 2))
    key = module.rope(module.k_proj(normed).view(shape).transpose(1, 2))
    value = module.v_proj(normed).view(shape).transpose(1, 2)

    scores = query @ key.transpose(-1, -2) / math.sqrt(module.d_head)
    allowed = build_causal_window_mask(seq_len, module.window)
    scores = scores.masked_fill(~allowed, float("-inf"))
    attended = scores.softmax(dim=-1) @ value

    merged = attended.transpose(1, 2).reshape(batch, seq_len, module.d_model)
    return module.out_proj(merged)


# ---------------------------------------------------------------------------------------
# The masks
# ---------------------------------------------------------------------------------------


def test_the_unwindowed_mask_is_exactly_the_lower_triangle():
    mask = build_causal_window_mask(SEQ_LEN, None)
    assert torch.equal(mask, torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool).tril())


def test_the_windowed_mask_is_exactly_the_band():
    mask = build_causal_window_mask(SEQ_LEN, WINDOW)
    positions = torch.arange(SEQ_LEN)
    displacement = positions[:, None] - positions[None, :]
    assert torch.equal(mask, (displacement >= 0) & (displacement < WINDOW))


@pytest.mark.parametrize("window", [None, 1, WINDOW, SEQ_LEN, SEQ_LEN + 4])
def test_every_row_admits_its_own_position_so_no_row_can_be_fully_masked(window):
    """An all-masked row would divide by zero in the softmax and return NaN.

    It cannot happen here, and that is structural rather than lucky: both mask forms always admit
    $j = t$, and the encoder attention does no data-driven validity masking. So there is no NaN
    path to defend against and none is written.
    """
    mask = build_causal_window_mask(SEQ_LEN, window)
    assert bool(mask.diagonal().all())
    assert bool(mask.any(dim=-1).all())


def test_a_non_positive_window_raises_naming_the_value():
    with pytest.raises(ValueError, match="got 0"):
        build_causal_window_mask(SEQ_LEN, 0)
    with pytest.raises(ValueError, match="got -3"):
        build_causal_window_mask(SEQ_LEN, -3)


# ---------------------------------------------------------------------------------------
# The kernel against an explicit softmax
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [None, WINDOW])
def test_the_kernel_matches_an_explicit_masked_softmax(window):
    module = CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=window)
    module.eval()
    x = _sequence(seed=2)

    assert torch.allclose(module(x), _reference_attention(module, x), atol=_KERNEL_TOL)


def test_the_two_masking_mechanisms_are_mutually_exclusive():
    """Never both: the kernel flag for the full prefix, an explicit band mask for a window."""
    full = CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN)
    windowed = CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=WINDOW)

    assert full.is_causal is True and full.attn_mask is None
    assert windowed.is_causal is False and windowed.attn_mask is not None
    assert windowed.attn_mask.dtype == torch.bool


# ---------------------------------------------------------------------------------------
# Causality and the window boundary
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [None, WINDOW])
@pytest.mark.parametrize("cut", [0, 1, SEQ_LEN - 2])
def test_the_attention_sublayer_is_token_causal(window, cut):
    module = CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=window)
    module.eval()
    assert_token_causal(module, _sequence(seed=3), cut, label=f"attention(window={window})")


@pytest.mark.parametrize("window", [None, WINDOW])
@pytest.mark.parametrize("cut", [0, 1, SEQ_LEN - 2])
def test_the_transformer_block_is_token_causal(window, cut):
    block = CausalTransformerBlock(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN, window=window)
    block.eval()
    assert_token_causal(block, _sequence(seed=4), cut, label=f"block(window={window})")


@pytest.mark.parametrize(
    "module_factory",
    [
        lambda: CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=WINDOW),
        lambda: CausalTransformerBlock(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN, window=WINDOW),
    ],
    ids=["attention", "block"],
)
def test_the_window_boundary_is_pinned_from_both_sides(module_factory):
    r"""One block reads exactly $[t - W + 1,\ t]$.

    The excluded side is asserted **bitwise**: a masked key contributes a softmax weight of exactly
    zero, so its value never enters the sum at any precision. The included side is asserted as a
    difference, since one step's influence through a $10^{-2}$ LayerScale is small but must not be
    nothing.
    """
    module = module_factory()
    module.eval()
    x = _sequence(seed=5)
    anchor = SEQ_LEN - 1

    reference = module(x)
    outside = module(_resample_step(x, anchor - WINDOW))
    inside = module(_resample_step(x, anchor - WINDOW + 1))

    assert torch.equal(reference[:, anchor], outside[:, anchor]), (
        f"step {anchor - WINDOW} is {WINDOW} steps back and must be outside a window of {WINDOW}"
    )
    assert not torch.equal(reference[:, anchor], inside[:, anchor]), (
        f"step {anchor - WINDOW + 1} is inside the window and must reach the output"
    )


# ---------------------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------------------


def test_the_block_parameter_count_is_the_architecture_arithmetic():
    r"""$4d^2$ attention projections, $3d\,d_{\mathrm{ff}}$ SwiGLU, $4d$ for two norms and two
    LayerScale vectors."""
    block = CausalTransformerBlock(SHIPPED_D, 4, SHIPPED_D_FF, 300)
    arithmetic = 4 * SHIPPED_D**2 + 3 * SHIPPED_D * SHIPPED_D_FF + 4 * SHIPPED_D
    assert sum(p.numel() for p in block.parameters()) == arithmetic == 164_352


def test_an_indivisible_head_split_raises_naming_both_values():
    with pytest.raises(ValueError, match=r"d_model \(16\).*num_heads \(3\)"):
        CausalSelfAttention(16, 3, SEQ_LEN)


def test_an_odd_head_width_raises_naming_it():
    """$d_h$ must be even because the rotation acts on coordinate pairs."""
    with pytest.raises(ValueError, match="d_head=5"):
        CausalSelfAttention(10, 2, SEQ_LEN)


def test_declaring_both_a_window_and_the_causal_flag_raises():
    """Double-specifying causality is refused, in both directions, so the two cannot silently
    disagree after a later edit."""
    with pytest.raises(ValueError, match="contradicts"):
        CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=WINDOW, is_causal=True)
    with pytest.raises(ValueError, match="contradicts"):
        CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, window=None, is_causal=False)


def test_the_declared_consistent_combinations_are_accepted():
    """The guard must not reject a caller that states what is already true."""
    assert CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, is_causal=True).window is None
    assert (
        CausalSelfAttention(
            D_MODEL, NUM_HEADS, SEQ_LEN, window=WINDOW, is_causal=False
        ).window
        == WINDOW
    )


def test_the_block_preserves_shape_and_reports_its_window():
    block = CausalTransformerBlock(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN, window=WINDOW)
    x = _sequence(seed=6)
    assert block(x).shape == x.shape
    assert block.window == WINDOW
    assert CausalTransformerBlock(D_MODEL, NUM_HEADS, D_FF, SEQ_LEN).window is None


def test_attention_probability_dropout_is_structurally_absent():
    """The architecture sets it to zero and it is not a configuration key: the ``dropout``
    argument is the sublayer's *output* dropout, and the kernel is always called at ``p = 0``.

    Neutralising the one dropout the sublayer legitimately owns is what makes this falsifiable.
    Anything still stochastic in train mode afterwards is dropout inside the attention itself.
    """
    module = CausalSelfAttention(D_MODEL, NUM_HEADS, SEQ_LEN, dropout=0.5)
    module.train()
    assert float(module.dropout.p) == 0.5
    module.dropout = nn.Identity()
    x = _sequence(seed=7)

    torch.manual_seed(0)
    first = module(x)
    torch.manual_seed(1)
    second = module(x)

    assert torch.equal(first, second)
