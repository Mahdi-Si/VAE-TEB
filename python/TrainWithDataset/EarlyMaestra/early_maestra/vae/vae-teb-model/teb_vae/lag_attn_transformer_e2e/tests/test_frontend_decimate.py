r"""The fixed anti-alias decimator: its offset, its kernel, and its survival of initialisation.

Three failures this module could suffer are all silent, which is why each gets a paired test rather
than an assertion.

The **offset** could be left or centred instead of right. Nothing would raise; token $t$ would
simply read raw sample $16t + 16$, or would discard the newest quarter-second of every token, and
every downstream number would be quietly wrong. It is checked against a hand-written convolution
rather than against the module's own arithmetic.

The **kernel** could be clobbered. ``initialization`` Xavier-fills every ``nn.Conv1d`` weight in the
model, so storing the coefficients as a layer would replace them with random values -- and a random
low-pass still low-passes something, so shapes stay right and only the aliasing comes back. The
survival test is therefore paired with a test-local ``nn.Conv1d`` holding the *same* coefficients,
which the same pass must visibly change; without that half the assertion would also pass against an
``initialization`` that walked nothing at all.

The **filtering** could stop happening. A binomial kernel has an exact null at the old Nyquist, so
that is testable at $10^{-12}$ in float64 rather than against a threshold somebody chose -- and the
paired control is the unfiltered subsample, whose output at the same input has magnitude $1$.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from teb_vae.lag_attn.nets.blocks import initialization
from teb_vae.lag_attn_transformer_e2e.nets.frontend import (
    ANTI_ALIAS_TAPS,
    CausalAntiAliasDecimate,
    binomial_lowpass,
)

CHANNELS = 4
STRIDE = 2
LENGTH = 64


def _manual_causal_fir(x: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    r"""Apply $\tilde x[n] = \sum_i h_i x[n-i]$ by hand, treating $x[n<0]$ as zero.

    Written out rather than delegated to ``F.conv1d`` so that the offset identity below is checked
    against an independent implementation. A probe that borrowed the module's own arithmetic could
    not catch that arithmetic being wrong.

    Args:
        x: Input of shape ``(B, C, L)``.
        coefficients: The FIR taps, ``(tau,)``.

    Returns:
        The filtered signal, shaped like ``x``.
    """
    filtered = torch.zeros_like(x)
    for index in range(int(x.shape[-1])):
        accumulator = torch.zeros_like(x[..., 0])
        for tap in range(int(coefficients.numel())):
            if index - tap >= 0:
                accumulator = accumulator + coefficients[tap] * x[..., index - tap]
        filtered[..., index] = accumulator
    return filtered


# ---------------------------------------------------------------------------------------
# The kernel itself
# ---------------------------------------------------------------------------------------
def test_the_binomial_kernel_is_unit_sum_symmetric_and_null_at_nyquist():
    r"""$H(0) = 1$ preserves the DC level, symmetry is what makes the delay a pure integer, and
    $H(\pi) = 0$ is the property the alias test asserts against."""
    coefficients = binomial_lowpass(ANTI_ALIAS_TAPS).to(torch.float64)

    assert coefficients.numel() == ANTI_ALIAS_TAPS
    assert float(coefficients.sum()) == pytest.approx(1.0)
    assert torch.allclose(coefficients, coefficients.flip(0))
    alternating = coefficients * torch.tensor(
        [(-1.0) ** index for index in range(ANTI_ALIAS_TAPS)], dtype=torch.float64
    )
    assert abs(float(alternating.sum())) < 1e-15


def test_the_binomial_kernel_is_the_binomial_row():
    """Pinned against ``math.comb`` so a rewrite of the three-line expression cannot drift."""
    expected = torch.tensor(
        [math.comb(ANTI_ALIAS_TAPS - 1, index) for index in range(ANTI_ALIAS_TAPS)],
        dtype=torch.float32,
    )
    assert torch.allclose(binomial_lowpass(ANTI_ALIAS_TAPS), expected / expected.sum())


def test_a_single_tap_kernel_is_refused():
    """One tap is the identity, which is decimation without anti-aliasing -- the case this module
    deliberately does not offer."""
    with pytest.raises(ValueError, match="at least 2 taps"):
        binomial_lowpass(1)


# ---------------------------------------------------------------------------------------
# Shape and offset
# ---------------------------------------------------------------------------------------
def test_the_output_length_is_the_input_length_over_the_stride():
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE)

    out = decimate(torch.randn(2, CHANNELS, LENGTH))

    assert out.shape == (2, CHANNELS, LENGTH // STRIDE)


def test_the_offset_is_right_checked_against_a_hand_built_index():
    r"""$\mathrm{out}[t] = \tilde x[s t + s - 1]$: the last sample of each stride group, never the
    first and never the middle. Composed over four stride-2 stages this is what puts token $t$'s
    newest input at raw index $16t + 15$."""
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE).double()
    x = torch.randn(2, CHANNELS, 32, dtype=torch.float64)

    out = decimate(x)

    filtered = _manual_causal_fir(x, binomial_lowpass(ANTI_ALIAS_TAPS).to(torch.float64))
    for token in range(int(out.shape[-1])):
        expected = filtered[..., STRIDE * token + STRIDE - 1]
        assert torch.allclose(out[..., token], expected, atol=1e-12), f"token {token}"


def test_a_wrong_channel_count_is_refused_by_name():
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE)

    with pytest.raises(ValueError, match=f"expected \\(B, {CHANNELS}, L\\)"):
        decimate(torch.randn(2, CHANNELS + 1, LENGTH))


def test_a_non_positive_geometry_is_refused():
    with pytest.raises(ValueError, match="must both be positive"):
        CausalAntiAliasDecimate(0, STRIDE)
    with pytest.raises(ValueError, match="must both be positive"):
        CausalAntiAliasDecimate(CHANNELS, 0)


# ---------------------------------------------------------------------------------------
# The coefficients are a buffer, and they survive initialisation
# ---------------------------------------------------------------------------------------
def test_the_coefficients_are_a_non_persistent_buffer_and_not_a_layer():
    """Three separate claims, each with its own failure mode: a ``Parameter`` would be trained, an
    ``nn.Conv1d`` would be Xavier-overwritten, and a persistent buffer would make a checkpoint fail
    to load the moment the tap count changed."""
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE)

    assert "fir" in dict(decimate.named_buffers())
    assert not isinstance(decimate.fir, nn.Parameter)
    assert list(decimate.parameters()) == []
    assert not any(isinstance(child, nn.Conv1d) for child in decimate.modules())
    assert decimate.state_dict() == {}


def test_a_full_initialization_pass_leaves_the_coefficients_bitwise_identical():
    """The paired control is what makes this mean anything: the same pass must visibly change a
    ``nn.Conv1d`` holding the *same* coefficients, or the assertion above would also pass against an
    ``initialization`` that walked nothing."""

    class _Holder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE)
            self.as_a_layer = nn.Conv1d(
                CHANNELS, CHANNELS, kernel_size=ANTI_ALIAS_TAPS, groups=CHANNELS, bias=False
            )
            with torch.no_grad():
                self.as_a_layer.weight.copy_(self.decimate.fir)

    holder = _Holder()
    fir_before = holder.decimate.fir.clone()
    layer_before = holder.as_a_layer.weight.detach().clone()

    initialization(holder)

    assert torch.equal(holder.decimate.fir, fir_before)
    assert not torch.equal(holder.as_a_layer.weight.detach(), layer_before)


# ---------------------------------------------------------------------------------------
# The anti-aliasing itself
# ---------------------------------------------------------------------------------------
def test_the_old_nyquist_is_annihilated_and_the_unfiltered_control_passes_it_through():
    r"""Drive $x[n] = (-1)^n$, the frequency a factor-2 decimation folds onto DC. A binomial kernel
    has $|H(\pi)| = 0$ exactly, so the steady-state output is round-off; plain subsampling, computed
    inline here, returns the alternating signal at full amplitude. The first
    $\lceil (\tau - 1) / s \rceil$ outputs are skipped because they see the zero left pad."""
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE).double()
    alternating = torch.tensor(
        [(-1.0) ** index for index in range(LENGTH)], dtype=torch.float64
    ).expand(2, CHANNELS, LENGTH)

    filtered = decimate(alternating)
    unfiltered = alternating[..., STRIDE - 1 :: STRIDE]

    transient = math.ceil((ANTI_ALIAS_TAPS - 1) / STRIDE)
    assert float(filtered[..., transient:].abs().max()) < 1e-12
    assert float(unfiltered[..., transient:].abs().min()) > 0.9


def test_a_constant_signal_passes_through_unchanged():
    """$H(0) = 1$: the filter is a low-pass, not an attenuator. A kernel that failed to normalise
    would shrink every token's level by a factor the downstream norms would then hide."""
    decimate = CausalAntiAliasDecimate(CHANNELS, STRIDE).double()
    constant = torch.full((2, CHANNELS, LENGTH), 3.0, dtype=torch.float64)

    out = decimate(constant)

    transient = math.ceil((ANTI_ALIAS_TAPS - 1) / STRIDE)
    assert torch.allclose(out[..., transient:], torch.full_like(out[..., transient:], 3.0))
