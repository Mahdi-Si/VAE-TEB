r"""The raw future-target gather, checked against hand-written slices.

The comparisons are against explicit ``fhr[:, a:b]`` slices rather than against the index
builder itself: a wrong formula shared by builder and gather would agree with itself perfectly.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index, build_future_target

_PROD = TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30)
_TINY = TrimmedRawGeometry(raw_len=256, decimation=16, horizon=4, warmup=2)


def test_the_index_grid_has_the_documented_shape_and_bounds():
    idx = build_future_index(_PROD)
    assert idx.shape == (270, 30, 16)
    assert idx.dtype == torch.long
    assert int(idx.min()) == 16          # anchor 0's first future sample
    assert int(idx.max()) == 4799        # anchor 269's last future sample


@pytest.mark.parametrize(
    ("anchor", "start", "stop"),
    [
        (0, 16, 496),  # the first future window
        (100, 1616, 2096),  # [16*101, 16*101 + 480)
        (269, 4320, 4800),  # the last one, ending exactly at the raw end
    ],
    ids=["first", "middle", "last"],
)
def test_an_anchor_gathers_its_own_future_window(anchor, start, stop):
    fhr = torch.randn(2, 4800)
    target = build_future_target(fhr, _PROD)
    assert target.shape == (2, 270, 30, 16)
    assert torch.equal(target[:, anchor].reshape(2, -1), fhr[:, start:stop])


def test_a_precomputed_index_grid_is_used_verbatim():
    """The hot-path cache must gather identically to the built-on-the-fly grid."""
    fhr = torch.randn(2, 256)
    idx = build_future_index(_TINY)
    assert torch.equal(
        build_future_target(fhr, _TINY, future_index=idx),
        build_future_target(fhr, _TINY),
    )


def test_a_wrong_raw_length_is_rejected_naming_both_lengths_and_the_trim():
    """A loader at the wrong trim_minutes must fail loudly, not gather shifted targets."""
    with pytest.raises(ValueError, match=r"5280.*4800.*trim_minutes") as excinfo:
        build_future_target(torch.randn(2, 5280), _PROD)
    assert "trim_minutes" in str(excinfo.value)


def test_a_non_2d_signal_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        build_future_target(torch.randn(4800), _PROD)
