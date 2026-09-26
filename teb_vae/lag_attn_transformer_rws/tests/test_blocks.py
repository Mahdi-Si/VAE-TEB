r"""The position-wise primitives, the causal depthwise convolution, and the gated conv block.

``RMSNorm`` and ``SwiGLUFeedForward`` are pinned against their equations written out longhand, which
also pins them as position-wise. The depthwise convolution and the gated block are probed for
token causality -- bit-stability at the cut paired with movement at the end, so a module returning
zeros cannot pass -- and for the two initialisation properties the encoder relies on: the
depthwise filters are variance preserving, and a fresh gated block is near, but not at, the
identity.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    RMS_NORM_EPS,
    CausalDepthwiseConv1d,
    GatedCausalConvBlock,
    RMSNorm,
    SwiGLUFeedForward,
    init_depthwise_,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import assert_token_causal, relative_change

#: Geometry the position-wise probes run at. Small; these modules carry no time dependence, so
#: nothing here scales with $T$.
BATCH, SEQ_LEN, D_MODEL = 2, 12, 16

#: The shipped model width, where the initialisation properties are measured.
SHIPPED_D = 128

#: The module computes ``x * rsqrt(v)`` and the reference below computes ``x / sqrt(v)``. Those
#: are the same number in exact arithmetic and differ in the last bit or two in float32, so the
#: comparison is tolerant rather than bitwise -- the identity being checked is the formula, not the
#: reciprocal-square-root routine.
_REDUCTION_TOL = 1e-6

#: Sample standard deviations are noisy. Over $Ck \ge 640$ draws the relative standard error is
#: about $3\%$, so a $10\%$ band is roughly three sigma: wide enough not to flake, tight enough to
#: separate $1/\sqrt k$ from the eightfold-smaller value the generic initialiser would produce.
_STD_BAND = 0.10


def _sequence(seed: int = 0, *, batch: int = BATCH, seq_len: int = SEQ_LEN, dim: int = D_MODEL):
    """A seeded $(B, T, d)$ activation tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, dim, generator=generator)


# ---------------------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------------------


def test_rms_norm_matches_the_hand_written_reduction():
    """Pinned against the equation itself, not against another normaliser."""
    module = RMSNorm(D_MODEL)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.5, 1.5, D_MODEL))
    x = _sequence(seed=3)

    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    expected = expected * module.weight

    assert torch.allclose(module(x), expected, atol=_REDUCTION_TOL, rtol=0.0)


# ---------------------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------------------


def test_swiglu_matches_its_definition_with_the_gate_on_the_silu_branch():
    """Rebuilt from the module's own weights, so which branch carries the nonlinearity is pinned.

    Zeroing a projection alone cannot distinguish the two branches -- either one zeroes the
    product -- so the assignment has to be checked against the expression.
    """
    module = SwiGLUFeedForward(D_MODEL, 4 * D_MODEL)
    x = _sequence(seed=8)

    expected = module.out_proj(
        torch.nn.functional.silu(module.gate_proj(x)) * module.value_proj(x)
    )

    assert torch.equal(module(x), expected)


def test_swiglu_zeroing_the_gate_projection_gives_exactly_zero():
    """Bias-free throughout, so a dead gate produces an exact zero rather than a residual bias."""
    module = SwiGLUFeedForward(D_MODEL, 4 * D_MODEL)
    with torch.no_grad():
        module.gate_proj.weight.zero_()

    output = module(_sequence(seed=9))

    assert torch.equal(output, torch.zeros_like(output))


# ---------------------------------------------------------------------------------------
# Causal depthwise convolution
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_size,dilation", [(3, 1), (5, 2), (9, 4), (15, 8)])
def test_depthwise_preserves_length(kernel_size, dilation):
    module = CausalDepthwiseConv1d(D_MODEL, kernel_size, dilation)
    x = torch.randn(BATCH, D_MODEL, SEQ_LEN)
    assert module(x).shape == x.shape


@pytest.mark.parametrize("cut", [0, 1, SEQ_LEN - 2])
def test_depthwise_is_token_causal(cut):
    module = CausalDepthwiseConv1d(D_MODEL, kernel_size=5, dilation=2)
    module.eval()

    def forward(x: torch.Tensor) -> torch.Tensor:
        return module(x.transpose(1, 2)).transpose(1, 2)

    assert_token_causal(forward, _sequence(seed=10), cut, label="CausalDepthwiseConv1d")


@pytest.mark.parametrize("kernel_size", [5, 9])
def test_depthwise_initialiser_is_variance_preserving(kernel_size):
    """The generic Xavier pass reads this shape's fans wrongly by a factor of $8.03$ at $C = 128$.

    ``fan_in`` is $k$ and ``fan_out`` is $Ck$ on a $(C, 1, k)$ weight, so Xavier gives
    $\\sqrt{2/(k + Ck)}$ where variance preservation needs $1/\\sqrt k$.
    """
    module = CausalDepthwiseConv1d(SHIPPED_D, kernel_size)
    torch.manual_seed(0)

    replaced = init_depthwise_(module)

    assert replaced == 1, "the initialiser found no depthwise convolution to re-initialise"
    target = 1.0 / math.sqrt(kernel_size)
    measured = float(module.conv.weight.std())
    assert abs(measured - target) / target < _STD_BAND, (
        f"depthwise std {measured:.4f} is not within {_STD_BAND:.0%} of {target:.4f}"
    )

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.conv.weight)
    xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
    assert fan_in == kernel_size and fan_out == SHIPPED_D * kernel_size
    assert measured > 5.0 * xavier_std, (
        f"depthwise std {measured:.4f} is not clear of the Xavier value {xavier_std:.4f} the "
        f"generic pass would have left"
    )


def test_depthwise_initialiser_reaches_every_convolution_in_a_subtree():
    """It is applied to the whole model after the generic pass, so it must walk, not just act."""
    stack = nn.Sequential(
        CausalDepthwiseConv1d(8, 3), nn.Linear(8, 8), CausalDepthwiseConv1d(8, 3)
    )
    assert init_depthwise_(stack) == 2


# ---------------------------------------------------------------------------------------
# Gated causal convolution block
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [0, 1, SEQ_LEN - 2])
def test_gated_conv_block_is_token_causal(cut):
    block = GatedCausalConvBlock(D_MODEL, kernel_size=3, dilation=2)
    block.eval()
    assert_token_causal(block, _sequence(seed=12), cut, label="GatedCausalConvBlock")


def test_gated_conv_block_starts_near_the_identity_without_being_disconnected():
    """LayerScale at $10^{-2}$ makes a fresh block almost a pass-through.

    Almost, not exactly: the lower bound is what a residual branch that was never wired into the
    sum would fail, and that failure is otherwise indistinguishable from good behaviour.
    """
    block = GatedCausalConvBlock(SHIPPED_D, kernel_size=5)
    block.eval()
    x = _sequence(seed=13, dim=SHIPPED_D)

    movement = relative_change(x, block(x))

    assert movement < 0.05, f"block moved its input by {movement:.3e} at initialisation"
    assert movement > 1e-6, "block is an exact pass-through -- the residual branch is dead"

