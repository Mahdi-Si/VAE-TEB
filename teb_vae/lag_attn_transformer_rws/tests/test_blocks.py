r"""The position-wise primitives, the causal depthwise convolution, and the gated conv block.

Every test here comes in pairs, because each of these modules has a way of passing while doing
nothing. A module returning zeros is position-wise, causal, near the identity in norm ratio only
if you forget to check the ratio is non-zero, and silently fatal. So each probe asserts both that
the output did not move where it must not, **and** that it did move where it must.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.nets.blocks import (
    LAYER_SCALE_INIT,
    RMS_NORM_EPS,
    CausalDepthwiseConv1d,
    GatedCausalConvBlock,
    LayerScale,
    RMSNorm,
    SwiGLUFeedForward,
    init_depthwise_,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    MOVEMENT_TOL,
    assert_token_causal,
    relative_change,
)

#: Geometry the position-wise probes run at. Small; these modules carry no time dependence, so
#: nothing here scales with $T$.
BATCH, SEQ_LEN, D_MODEL = 2, 12, 16

#: The shipped widths, where the parameter arithmetic is pinned against the architecture's own
#: numbers rather than against a miniature's.
SHIPPED_D, SHIPPED_D_FF = 128, 256

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


def _resample_steps(x: torch.Tensor, steps: tuple[int, ...], *, seed: int = 1) -> torch.Tensor:
    """Return a copy of ``x`` with the given steps drawn afresh and the rest untouched."""
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    for step in steps:
        perturbed[:, step] = torch.randn(
            x.shape[0], x.shape[2], generator=generator, dtype=x.dtype
        )
    return perturbed


def _assert_position_wise(module: nn.Module, x: torch.Tensor, steps: tuple[int, ...]) -> None:
    """Assert the module mixes channels but never time.

    Resamples ``steps`` and requires every other step's output to be **bitwise** identical -- a
    position-wise module reads nothing else, so a tolerance would be a concession to a bug -- while
    the resampled steps themselves move measurably. The second half is what a module returning a
    constant would fail.

    Args:
        module: The module under test.
        x: A $(B, T, d)$ input.
        steps: Which steps to resample.
    """
    reference = module(x)
    perturbed = module(_resample_steps(x, steps))
    for step in range(x.shape[1]):
        if step in steps:
            movement = relative_change(reference[:, step], perturbed[:, step])
            assert movement > MOVEMENT_TOL, (
                f"step {step} was resampled but its output moved by only {movement:.3e} -- the "
                f"module is not reading its input"
            )
        else:
            assert torch.equal(reference[:, step], perturbed[:, step]), (
                f"step {step} moved when only steps {steps} were resampled; a position-wise "
                f"module cannot see another step at all"
            )


# ---------------------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------------------


def test_rms_norm_is_position_wise():
    _assert_position_wise(RMSNorm(D_MODEL), _sequence(), (0, 5, SEQ_LEN - 1))


def test_rms_norm_matches_the_hand_written_reduction():
    """Pinned against the equation itself, not against another normaliser."""
    module = RMSNorm(D_MODEL)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(0.5, 1.5, D_MODEL))
    x = _sequence(seed=3)

    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    expected = expected * module.weight

    assert torch.allclose(module(x), expected, atol=_REDUCTION_TOL, rtol=0.0)


def test_rms_norm_does_not_centre_so_a_constant_offset_survives():
    """The distinguishing property against ``LayerNorm``, checked without invoking ``LayerNorm``.

    A centring normaliser maps $x + c\\mathbf 1$ and $x$ to the same output. This one must not:
    the offset changes the root mean square, so it reaches the output.
    """
    module = RMSNorm(D_MODEL)
    x = _sequence(seed=4)
    offset = x + 3.0

    plain = module(x)
    shifted = module(offset)

    assert not torch.allclose(plain, shifted, atol=1e-4), "the offset was centred away"
    expected = offset / torch.sqrt(offset.pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS)
    assert torch.allclose(shifted, expected * module.weight, atol=_REDUCTION_TOL, rtol=0.0)


def test_rms_norm_scale_starts_at_one_and_is_the_only_parameter():
    module = RMSNorm(D_MODEL)
    assert [name for name, _ in module.named_parameters()] == ["weight"]
    assert module.weight.shape == (D_MODEL,)
    assert torch.equal(module.weight, torch.ones(D_MODEL))


# ---------------------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------------------


def test_layer_scale_starts_at_exactly_the_initialisation():
    module = LayerScale(D_MODEL)
    assert module.weight.shape == (D_MODEL,)
    assert torch.equal(module.weight, torch.full((D_MODEL,), LAYER_SCALE_INIT))
    assert LAYER_SCALE_INIT == 1e-2


def test_layer_scale_is_position_wise():
    _assert_position_wise(LayerScale(D_MODEL), _sequence(seed=5), (1, SEQ_LEN - 2))


def test_layer_scale_multiplies_channel_wise():
    module = LayerScale(D_MODEL)
    x = _sequence(seed=6)
    assert torch.equal(module(x), x * module.weight)


# ---------------------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------------------


def test_swiglu_has_exactly_three_d_dff_parameters():
    module = SwiGLUFeedForward(SHIPPED_D, SHIPPED_D_FF)
    total = sum(parameter.numel() for parameter in module.parameters())
    assert total == 3 * SHIPPED_D * SHIPPED_D_FF == 98_304
    # Bias-free is what makes the count exactly three matrices, and what makes a dead gate
    # produce an exact zero rather than a leftover bias.
    for projection in (module.gate_proj, module.value_proj, module.out_proj):
        assert projection.bias is None


def test_swiglu_is_position_wise():
    _assert_position_wise(SwiGLUFeedForward(D_MODEL, 4 * D_MODEL), _sequence(seed=7), (2, 9))


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


@pytest.mark.parametrize("kernel_size", [3, 5, 9, 15])
@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_depthwise_preserves_length(kernel_size, dilation):
    module = CausalDepthwiseConv1d(D_MODEL, kernel_size, dilation)
    x = torch.randn(BATCH, D_MODEL, SEQ_LEN)
    assert module(x).shape == x.shape


def test_depthwise_weight_is_one_filter_per_channel_with_no_bias_and_no_padding():
    kernel_size = 5
    module = CausalDepthwiseConv1d(D_MODEL, kernel_size, dilation=2)

    assert module.conv.weight.shape == (D_MODEL, 1, kernel_size)
    assert sum(p.numel() for p in module.parameters()) == D_MODEL * kernel_size
    assert module.conv.bias is None
    # The padding argument would be symmetric, so it must stay zero: the left pad is explicit.
    assert module.conv.padding == (0,)
    assert module.left_padding == (kernel_size - 1) * 2


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


@pytest.mark.parametrize("kernel_size,expected", [(5, 50_176), (9, 50_688)])
def test_gated_conv_block_parameter_count(kernel_size, expected):
    """$2d^2$ gated input, $d^2$ output, $d$ per norm, $d$ LayerScale, $dk$ depthwise."""
    block = GatedCausalConvBlock(SHIPPED_D, kernel_size)
    arithmetic = (
        2 * SHIPPED_D**2 + SHIPPED_D**2 + 2 * SHIPPED_D + SHIPPED_D + SHIPPED_D * kernel_size
    )
    assert sum(p.numel() for p in block.parameters()) == arithmetic == expected


def test_gated_conv_block_preserves_shape():
    block = GatedCausalConvBlock(D_MODEL, kernel_size=3, dilation=2)
    x = _sequence(seed=11)
    assert block(x).shape == x.shape


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


def test_gated_conv_block_reports_its_reach():
    block = GatedCausalConvBlock(D_MODEL, kernel_size=9, dilation=2)
    assert block.receptive_field == 1 + (9 - 1) * 2
