r"""S0-T01: config-driven raw/low-rate geometry and the crop-offset identities.

These tests pin the single most error-prone point of the raw model -- the crop offset -- so a
regression that shifts the forecast target by one low-rate step (predicting the anchor's own
encoded present) fails loudly.
"""
from __future__ import annotations

import pytest

from model.vae_teb_prediction.model.model_raw import geometry as geo


# ---------------------------------------------------------------------------
# Derived-constant identities.
# ---------------------------------------------------------------------------
def test_module_constants() -> None:
    assert geo.RAW_LEN == 5280
    assert geo.D == 16
    assert geo.T_TILDE == geo.RAW_LEN // geo.D == 330
    assert geo.T == geo.T_TILDE - 2 * geo.CROP == 300
    assert geo.R == geo.D == 16
    assert geo.H == 30
    assert geo.WARMUP == 30
    assert geo.H * geo.R == 480


def test_valid_anchor_range() -> None:
    assert geo.valid_anchor_range() == range(30, 270)
    # Trained anchors exclude the warm-up: [w, T-H) has length T - H - w = 240.
    assert len(geo.valid_anchor_range()) == geo.T - geo.H - geo.WARMUP == 240
    # All 270 anchors in [0, T-H) have a full horizon (t_valid); warm-up excludes the first 30.
    assert geo.GEOMETRY.t_valid == geo.T - geo.H == 270


# ---------------------------------------------------------------------------
# Crop-offset (critical).
# ---------------------------------------------------------------------------
def test_crop_offset_identities_all_valid_t() -> None:
    for t in geo.valid_anchor_range():
        assert geo.future_block_start(t) == 16 * (t + 16)
        assert geo.future_block_start(t) == geo.n_raw(t) + 1
        assert geo.future_block_start(t) == 16 * (t + geo.CROP + 1)


def test_future_block_start_t0_is_256_not_240() -> None:
    # The forecast block starts one sample AFTER the anchor's causal endpoint.
    assert geo.future_block_start(0) == 256
    # The anchor's OWN present block starts at 240 -- a distinct value the m_low mask uses.
    own_present_start = 16 * (0 + geo.CROP)
    assert own_present_start == 240
    assert geo.future_block_start(0) != own_present_start
    # And it is NOT the naive 16*(t+1) either.
    assert geo.future_block_start(0) != 16 * (0 + 1)


def test_n_raw_endpoint() -> None:
    assert geo.n_raw(0) == 16 * 16 - 1 == 255
    # A cropped anchor t is the uncropped token t + CROP.
    for t in (0, 50, 150, 269):
        assert geo.n_raw(t) == geo.token_endpoint_uncropped(t + geo.CROP)


def test_last_anchor_bound() -> None:
    last = geo.valid_anchor_range()[-1]
    assert last == 269
    end = geo.future_block_start(last) + geo.H * geo.R - 1
    assert end == 5039
    assert end < geo.RAW_LEN


def test_future_block_indices_shape_and_values() -> None:
    grid = geo.future_block_indices(0)
    assert len(grid) == geo.H
    assert all(len(row) == geo.R for row in grid)
    # First target sample is future_block_start(0) == 256.
    assert grid[0][0] == 256
    # Contiguous within the block: index [tau, r] == start + 16*tau + r.
    start = geo.future_block_start(0)
    assert grid[1][0] == start + 16
    assert grid[0][15] == start + 15
    assert grid[geo.H - 1][geo.R - 1] == start + 16 * (geo.H - 1) + (geo.R - 1)


# ---------------------------------------------------------------------------
# Config-driven derivation (a non-5280 dataset must not reuse the 5280 numbers).
# ---------------------------------------------------------------------------
def test_derive_geometry_default_matches_module() -> None:
    g = geo.derive_geometry(5280, 16)
    assert (g.t_tilde, g.t, g.r) == (330, 300, 16)
    assert g.future_block_start(0) == 256
    assert g.valid_anchor_range() == range(30, 270)


def test_derive_geometry_alternate_len() -> None:
    g = geo.derive_geometry(5760, 16)
    assert g.t_tilde == 360
    assert g.t == 330
    assert g.r == 16
    # Must NOT reuse the 5280-geometry numbers.
    assert g.t_tilde != geo.T_TILDE
    assert g.t != geo.T
    # Its own crop-offset identity still holds.
    assert g.future_block_start(0) == 16 * (0 + g.crop + 1) == 256
    last = g.valid_anchor_range()[-1]
    assert g.future_block_start(last) + g.horizon * g.r - 1 < 5760


def test_derive_geometry_rejects_indivisible() -> None:
    with pytest.raises(ValueError):
        geo.derive_geometry(5281, 16)


def test_derive_geometry_rejects_degenerate_crop() -> None:
    # crop so large that T <= 0.
    with pytest.raises(ValueError):
        geo.derive_geometry(320, 16, crop=15)
