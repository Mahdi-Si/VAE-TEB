r"""The planted pattern is load-bearing, so its ground truth gets its own tests.

Every target assertion in this package is made against
:func:`~teb_vae.lag_attn_fs.tests.conftest.make_patterned_batch`, and the whole point of it is
that a *wrong* gather cannot produce the right values. That rests on three properties, each of
which would fail silently if the pattern were ever weakened to a random draw:

* every element is distinct, so an off-by-one anchor, a wrong channel or a transposed read lands on
  a value that exists nowhere in the correct answer (distinctness at the declared $c_y$ also
  requires the step stride to exceed the channel count);
* every value is exactly representable in float32 at the largest geometry, so comparisons against
  it can be ``torch.equal``;
* the two stored target blocks concatenate back into one pattern, so the value at channel $c$ of
  the concatenated stream is $c$, the index the reach budget's keep-index is positional into.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_fs.tests.conftest import (
    BATCH,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    make_patterned_batch,
    patterned_feature_stream,
)


def test_every_element_of_the_pattern_is_distinct():
    """Uniqueness is what makes a mismatch legible: the value names the position it came from."""
    stream = patterned_feature_stream(BATCH, SEQ_LEN, TINY_KWARGS["c_y"])
    assert stream.unique().numel() == stream.numel()


def test_the_pattern_is_exactly_representable_in_float32():
    """Comparisons against it are ``torch.equal``, not ``allclose``, so this has to hold at the
    largest geometry the suite builds rather than merely at the tiny one."""
    largest = patterned_feature_stream(
        BATCH, SHIPPED_KWARGS["sequence_length"], SHIPPED_KWARGS["c_y"]
    )
    assert torch.equal(largest, largest.double().float())
    assert float(largest.max()) < 2.0**24


def test_the_two_blocks_concatenate_back_into_one_pattern():
    """The split at the block boundary is what makes the value at channel $c$ of the concatenated
    stream equal to $c$ -- and $c$ is what the reach budget's keep-index indexes into."""
    batch = make_patterned_batch()
    stream = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)
    assert torch.equal(stream, patterned_feature_stream(BATCH, SEQ_LEN, TINY_KWARGS["c_y"]))
