r"""Token causality, per block and per encoder, for both streams.

$H_t = f(X_{\le t})$ is the property the whole model rests on: the prior is supposed to condition
on $Y_{\le t}$ alone, and a history state that has seen its own future answers the coupling
question with the answer already in it. The leak would be small, entirely invisible in a loss
curve, and would corrupt only the quantity the model exists to measure.

Two things about how it is measured here.

The perturbation is a **random resample**, not a constant offset. The input adapter upstream ends
in a ``LayerNorm``, which removes a uniform channel shift outright; a constant-offset probe would
therefore report causality it never tested. ``RMSNorm`` inside the encoder does not centre, but the
probe runs through both.

The second half of every assertion -- that the output at $T-1$ *moved* -- is the negative control.
This architecture has no time-pooling normaliser to flip, which is how the encoder it replaces
built its leaky counterfactual, so the control is positional instead: the perturbation is shown to
reach the module before its absence at the cut is allowed to mean anything. Nothing in production
code exists only for this test.
"""
from __future__ import annotations

from typing import List, Tuple

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    SEQ_LEN,
    TINY_KWARGS,
    assert_token_causal,
    build_stream_encoder,
)

BATCH = 2
D_MODEL = int(TINY_KWARGS["d_model"])

#: Both streams, because their attention masks differ and only one of them is windowed.
STREAMS = ("target", "source")

#: Cutoffs, including the first timestep -- where a convolution's left padding and a rotary table's
#: zero offset both have their edge case -- and the last one that leaves a future to perturb.
CUTS = (0, 1, SEQ_LEN - 2)


def _sequence(seed: int = 0) -> torch.Tensor:
    """A seeded $(B, T, d)$ encoder input."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, generator=generator)


def _labelled_blocks(encoder: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Every block of an encoder in forward order, each with a label naming its stage."""
    return [
        *((f"conv{index}", block) for index, block in enumerate(encoder.conv_blocks)),
        *((f"attn{index}", block) for index, block in enumerate(encoder.attention_blocks)),
    ]


#: One case per (stream, block), built once at collection so a failure names the block rather than
#: an index into an anonymous stack.
BLOCK_CASES = [
    (stream, label)
    for stream in STREAMS
    for label, _ in _labelled_blocks(build_stream_encoder(stream))
]


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("stream", STREAMS)
def test_the_encoder_is_token_causal(stream, cut):
    encoder = build_stream_encoder(stream)
    assert_token_causal(encoder, _sequence(), cut, label=f"{stream} encoder")


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize(
    "stream,label", BLOCK_CASES, ids=[f"{stream}-{label}" for stream, label in BLOCK_CASES]
)
def test_every_block_is_token_causal(stream, label, cut):
    """Per block as well as per encoder: composition can hide a leak that a stack-level probe
    happens to attenuate below the tolerance."""
    encoder = build_stream_encoder(stream)
    block = dict(_labelled_blocks(encoder))[label]
    assert_token_causal(block, _sequence(seed=1), cut, label=f"{stream}.{label}")


def test_the_case_list_covers_every_block_of_both_encoders():
    """A silently-empty or short case list would leave the parametrised test vacuous."""
    for stream in STREAMS:
        encoder = build_stream_encoder(stream)
        expected = len(encoder.conv_blocks) + len(encoder.attention_blocks)
        assert expected == sum(1 for name, _ in BLOCK_CASES if name == stream)
        assert expected > 1, f"{stream} encoder has too few blocks for the probe to mean anything"


def test_the_probe_reaches_the_deepest_block_of_the_stack():
    """The stack-level negative control, stated once rather than implied by every case above.

    A perturbation at the last step must move the encoder's own output, or every bit-stability
    assertion in this file is a statement about a module that is not being driven.
    """
    encoder = build_stream_encoder("source")
    x = _sequence(seed=2)
    perturbed = x.clone()
    perturbed[:, -1] = torch.randn(BATCH, D_MODEL)

    assert not torch.equal(encoder(x)[:, -1], encoder(perturbed)[:, -1])
