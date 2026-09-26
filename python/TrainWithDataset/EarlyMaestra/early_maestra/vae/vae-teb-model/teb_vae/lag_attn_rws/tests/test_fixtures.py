"""The stub batch's planted gap, which every mask test in the suite relies on.

A stub batch whose ``weight`` stopped planting its gap would leave every mask test green whether
or not the masks work, so the gap and its position inside the trained-anchor range are pinned.
"""
from __future__ import annotations

from teb_vae.lag_attn_rws.tests.conftest import (
    SEQ_LEN,
    STUB_GAP_STEP,
    TINY_KWARGS,
    make_stub_batch,
)


def test_the_stub_batch_plants_its_gap():
    """A silently gap-free fixture would leave every mask test vacuous."""
    batch = make_stub_batch()
    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert (batch.weight == 0.0).any()
    # The gap sits inside the tiny trained-anchor range, where every mask can see it.
    assert TINY_KWARGS["warmup_period"] <= STUB_GAP_STEP
    assert STUB_GAP_STEP < SEQ_LEN - TINY_KWARGS["horizon"]
