r"""The planted ground truth the rest of the suite relies on.

A stub batch whose ``weight`` stopped planting its gap -- or planted it where the warm-up trims it
away -- would leave every mask test green whether or not the masks work. This package raises the
tiny warm-up to give the front end reach budget, which is exactly the change that could push the
gap out of the trained-anchor range $[w,\, T - H)$.
"""
from __future__ import annotations

from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    STUB_GAP_STEP,
    TINY_KWARGS,
    TINY_WARMUP_PERIOD,
    make_stub_batch,
)


def test_the_stub_batch_plants_its_gap_inside_the_trained_anchor_range():
    batch = make_stub_batch(BATCH, SEQ_LEN)

    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert TINY_WARMUP_PERIOD <= STUB_GAP_STEP < SEQ_LEN - int(TINY_KWARGS["horizon"])
