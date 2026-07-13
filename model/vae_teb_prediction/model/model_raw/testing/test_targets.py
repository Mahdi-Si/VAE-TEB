"""S3-T01: crop-aligned raw future-target extraction (the single most error-prone offset)."""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.model_raw.geometry import GEOMETRY, RAW_LEN, derive_geometry
from model.vae_teb_prediction.model.model_raw.raw_targets import (
    build_future_index,
    build_future_target,
)


def test_first_target_sample_is_future_block_start():
    # An identity signal x[n] = n makes X_plus[b, t, tau, r] equal the gathered raw index itself.
    fhr = torch.arange(RAW_LEN, dtype=torch.float32).unsqueeze(0)  # (1, 5280)
    xp = build_future_target(fhr, GEOMETRY)
    # First target sample of anchor t=0 is future_block_start(0) = 16*(0+16) = 256, NOT 240, NOT 16.
    assert int(xp[0, 0, 0, 0].item()) == GEOMETRY.future_block_start(0) == 256
    assert int(xp[0, 0, 0, 0].item()) != GEOMETRY.own_present_start(0)  # 240
    assert int(xp[0, 0, 0, 0].item()) != GEOMETRY.decimation  # 16*(t+1) at t=0


def test_target_matches_future_block_start_all_anchors():
    fhr = torch.arange(RAW_LEN, dtype=torch.float32).unsqueeze(0)
    xp = build_future_target(fhr, GEOMETRY)
    d = GEOMETRY.decimation
    for t in (0, 1, 30, 150, GEOMETRY.t_valid - 1):
        for tau in (0, 1, GEOMETRY.horizon - 1):
            for r in (0, GEOMETRY.r - 1):
                expected = GEOMETRY.future_block_start(t) + d * tau + r
                assert int(xp[0, t, tau, r].item()) == expected


def test_shape_and_last_anchor_bound():
    fhr = torch.arange(RAW_LEN, dtype=torch.float32).unsqueeze(0)
    xp = build_future_target(fhr, GEOMETRY)
    assert xp.shape == (1, GEOMETRY.t_valid, GEOMETRY.horizon, GEOMETRY.r)
    assert GEOMETRY.t_valid == GEOMETRY.t - GEOMETRY.horizon == 270
    # Last valid anchor's final target sample lands strictly inside the loaded window.
    assert int(xp[0, -1, -1, -1].item()) == 5039 < RAW_LEN


def test_build_future_index_matches_geometry_helper():
    idx = build_future_index(GEOMETRY)
    assert idx.shape == (GEOMETRY.t_valid, GEOMETRY.horizon, GEOMETRY.r)
    for t in (0, 5, GEOMETRY.t_valid - 1):
        assert idx[t].tolist() == GEOMETRY.future_block_indices(t)


def test_hand_computed_small_example():
    # A tiny geometry (raw_len=512, decimation=16, crop=2) -> future_block_start(t) = 16*(t+3).
    geo = derive_geometry(512, 16, crop=2, horizon=4, warmup=2)
    fhr = torch.arange(512, dtype=torch.float32).unsqueeze(0)
    xp = build_future_target(fhr, geo)
    assert xp.shape == (1, geo.t_valid, 4, 16)
    assert int(xp[0, 0, 0, 0].item()) == 16 * (0 + 3) == 48
    assert int(xp[0, 1, 0, 0].item()) == 16 * (1 + 3) == 64
    assert int(xp[0, -1, -1, -1].item()) < 512
