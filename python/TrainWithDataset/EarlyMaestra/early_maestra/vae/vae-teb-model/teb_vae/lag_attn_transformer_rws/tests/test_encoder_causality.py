r"""Token causality of each whole encoder, for both streams.

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

Each block type is probed on its own in ``test_blocks.py`` and ``test_attention_block.py``; this
file probes the stacks the model actually builds.
"""
from __future__ import annotations

import pytest
import torch

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


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("stream", STREAMS)
def test_the_encoder_is_token_causal(stream, cut):
    encoder = build_stream_encoder(stream)
    assert_token_causal(encoder, _sequence(), cut, label=f"{stream} encoder")

