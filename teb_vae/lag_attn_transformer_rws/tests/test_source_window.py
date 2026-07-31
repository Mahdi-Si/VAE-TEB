r"""The measured source receptive-field bound, and the structural bans on non-causal operations.

Two things the arithmetic alone does not establish.

**The bound is measured, not computed.** $R_U = R_{\mathrm{conv}} + N_U(W_U - 1)$ is a claim about
how a stack of masks and paddings compose, and the whole source-locality argument rests on it: the
encoder is meant to characterise a *local* source neighbourhood so the late lag cross-attention
keeps its ability to tell adjacent delays apart. So the edge is probed from both sides -- one step
outside the bound must leave the output bitwise unchanged, one step inside must not.

That probe runs in **float64 and asserts with** ``torch.equal``. The edge-most path traverses every
attention block, each scaled by a LayerScale at $10^{-2}$ times an attention weight of order
$1/W_U$, so the movement at the inside step is around $10^{-7}$ of the output and can round to
exactly zero in float32 -- which would make the "inside" half of the probe pass for the wrong
reason and, worse, would make a too-narrow window look correct.

**The bans are asserted structurally.** Time-pooled normalisation, recurrence and symmetric padding
are not style choices here; each is a causality violation, and each is cheap to reintroduce by
accident in a later edit.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    SEQ_LEN,
    TINY_KWARGS,
    build_stream_encoder,
)

BATCH = 1
D_MODEL = int(TINY_KWARGS["d_model"])

#: Amplitude of the resampled step. Large because the path being probed is the narrowest one in the
#: encoder; irrelevant to the *first* normalisation, which is scale-invariant, but it keeps the
#: difference well clear of float64's denormal range at the far end.
_PERTURBATION_AMPLITUDE = 50.0

#: Modules that would break token causality if any of them appeared inside a history encoder:
#: recurrence carries state the wrong way only if bidirectional, but no recurrence belongs here at
#: all, and every one of these normalisers pools statistics across the time axis.
_BANNED_TYPES = (
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
)


def _sequence(seed: int = 0, *, dtype=torch.float64) -> torch.Tensor:
    """A seeded $(B, T, d)$ encoder input."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, generator=generator, dtype=torch.float32).to(dtype)


def _resample_step(x: torch.Tensor, step: int, *, seed: int = 1) -> torch.Tensor:
    """Return a copy of ``x`` with exactly one step redrawn at a large amplitude."""
    generator = torch.Generator().manual_seed(seed)
    perturbed = x.clone()
    draw = torch.randn(x.shape[0], x.shape[2], generator=generator, dtype=torch.float32)
    perturbed[:, step] = (_PERTURBATION_AMPLITUDE * draw).to(x.dtype)
    return perturbed


@pytest.mark.parametrize("window", [int(TINY_KWARGS["source_attention_window"]), 3])
def test_the_measured_bound_equals_the_reported_one(window):
    r"""Perturbing at $t - R_U$ must not reach $t$; perturbing at $t - R_U + 1$ must."""
    encoder = build_stream_encoder("source", attention_window=window).double()
    reach = encoder.receptive_field
    anchor = SEQ_LEN - 1

    assert reach is not None
    assert reach < SEQ_LEN, (
        f"a bound of {reach} clamps at the sequence length, so every step is inside it and the "
        f"probe below cannot fail"
    )
    assert anchor - reach >= 0
    x = _sequence()

    reference = encoder(x)[:, anchor]
    outside = encoder(_resample_step(x, anchor - reach))[:, anchor]
    inside = encoder(_resample_step(x, anchor - reach + 1))[:, anchor]

    assert torch.equal(reference, outside), (
        f"step {anchor - reach} is {reach} steps back, outside a reported bound of {reach}, but "
        f"it moved the output at {anchor}"
    )
    assert not torch.equal(reference, inside), (
        f"step {anchor - reach + 1} is the earliest step inside the reported bound of {reach} and "
        f"must reach the output at {anchor}; the encoder's reach is narrower than it reports"
    )


def test_the_reported_bound_is_the_architecture_arithmetic():
    r"""$R_U = R_{\mathrm{conv}} + N_U(W_U - 1)$, from the tiny configuration's own numbers."""
    encoder = build_stream_encoder("source")
    kernels = TINY_KWARGS["encoder_conv_kernels"]
    dilations = TINY_KWARGS["encoder_conv_dilations"]
    conv_reach = 1 + sum((k - 1) * r for k, r in zip(kernels, dilations))
    blocks = int(TINY_KWARGS["source_attention_blocks"])
    window = int(TINY_KWARGS["source_attention_window"])

    assert encoder.conv_reach == conv_reach
    assert encoder.receptive_field == conv_reach + blocks * (window - 1)


@pytest.mark.parametrize("stream", ["target", "source"])
def test_no_recurrent_or_time_pooling_module_exists_in_either_encoder(stream):
    encoder = build_stream_encoder(stream)

    offenders = [
        f"{name or '<root>'}: {type(module).__name__}"
        for name, module in encoder.named_modules()
        if isinstance(module, _BANNED_TYPES)
    ]

    assert not offenders, (
        f"{stream} encoder contains {offenders} -- each of these either pools statistics across "
        f"time or carries recurrent state, and both make a history state a function of its own "
        f"future"
    )


@pytest.mark.parametrize("stream", ["target", "source"])
def test_every_convolution_pads_explicitly_rather_than_through_the_padding_argument(stream):
    """``nn.Conv1d``'s ``padding`` is symmetric, so a non-zero value would read the future.

    Every causal convolution here pads on the left before the call, which is why the argument must
    stay at zero.
    """
    encoder = build_stream_encoder(stream)

    convolutions = [
        (name, module)
        for name, module in encoder.named_modules()
        if isinstance(module, nn.Conv1d)
    ]

    assert convolutions or len(encoder.conv_blocks) == 0
    for name, module in convolutions:
        assert module.padding == (0,), f"{name} has padding={module.padding}"
        assert module.groups == module.in_channels, f"{name} is not depthwise"


@pytest.mark.parametrize("stream", ["target", "source"])
def test_no_pooling_module_exists_in_either_encoder(stream):
    """Adaptive or global pooling reduces over time, so at $t$ it would read every other step."""
    encoder = build_stream_encoder(stream)

    offenders = [
        f"{name}: {type(module).__name__}"
        for name, module in encoder.named_modules()
        if "Pool" in type(module).__name__
    ]

    assert not offenders, f"{stream} encoder contains pooling modules {offenders}"


def test_the_ban_list_catches_what_it_names():
    """A guard that cannot fire is not a guard."""

    class _Leaky(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.GroupNorm(1, 4)
            self.recurrent = nn.LSTM(4, 4, batch_first=True)

    offenders = [
        type(module).__name__
        for _, module in _Leaky().named_modules()
        if isinstance(module, _BANNED_TYPES)
    ]

    assert sorted(offenders) == ["GroupNorm", "LSTM"]
