r"""The trimmed-grid index arithmetic, pinned.

The formulas here are one crop-offset away from the untrimmed variant's, and applying either on
the other grid mis-aligns every forecast target by exactly one minute with nothing failing
loudly. So the numbers are pinned as literals -- ``future_block_start(0) == 16``, distinct from
the untrimmed ``256`` and from ``0`` -- rather than re-derived through the same arithmetic being
tested.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry


def _production() -> TrimmedRawGeometry:
    return TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)


def test_the_production_geometry_derives_the_documented_table():
    geometry = _production()
    assert geometry.t == 300
    assert geometry.t_valid == 270
    assert geometry.r == 16


def test_anchor_zero_forecast_starts_at_sixteen_not_the_untrimmed_value():
    """16, not 256 (the untrimmed grid's 16*(0+16)) and not 0 (the anchor's own block)."""
    geometry = _production()
    assert geometry.future_block_start(0) == 16
    assert geometry.future_block_start(0) != 16 * 16
    assert geometry.future_block_start(0) != 0
    assert geometry.n_raw(0) == 15


def test_the_last_valid_anchor_window_lands_exactly_at_the_raw_end():
    geometry = _production()
    last = geometry.t_valid - 1
    assert last == 269
    start = geometry.future_block_start(last)
    assert start == 4320
    assert start + geometry.horizon * geometry.r == 4800 == geometry.raw_len


def test_the_forecast_start_is_one_past_the_causal_endpoint():
    geometry = _production()
    for t in (0, 1, 137, 269):
        assert geometry.future_block_start(t) == geometry.n_raw(t) + 1 == 16 * (t + 1)


def test_the_trained_anchor_range_is_warmup_to_t_valid():
    assert _production().valid_anchor_range() == range(30, 270)


def test_the_tiny_test_geometry_constructs():
    """The suite's stub geometry: 16 steps of 16 raw samples, horizon 4, warmup 2."""
    geometry = TrimmedRawGeometry(raw_len=256, decimation=16, horizon=4, warmup=2)
    assert geometry.t == 16
    assert geometry.t_valid == 12
    assert geometry.future_block_start(0) == 16
    assert geometry.future_block_start(11) + 4 * 16 == 256


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(raw_len=4801, decimation=16, horizon=30, warmup=30), "divisible"),
        (dict(raw_len=4800, decimation=0, horizon=30, warmup=30), "decimation"),
        (dict(raw_len=4800, decimation=16, horizon=0, warmup=30), "horizon"),
        (dict(raw_len=4800, decimation=16, horizon=300, warmup=30), "degenerate"),
        (dict(raw_len=4800, decimation=16, horizon=30, warmup=270), "warmup"),
        (dict(raw_len=4800, decimation=16, horizon=30, warmup=-1), "warmup"),
    ],
)
def test_invalid_geometry_is_rejected_at_construction(kwargs, match):
    """An unvalidated instance must not exist; the constructor is the only gate."""
    with pytest.raises(ValueError, match=match):
        TrimmedRawGeometry(**kwargs)


def test_the_geometry_is_immutable():
    """Frozen, so a consumer cannot drift its grid after validation."""
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        _production().raw_len = 5280  # type: ignore[misc]
