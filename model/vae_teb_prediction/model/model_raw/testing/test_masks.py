"""S3-T02: multi-resolution mask construction (+CROP symmetry, warm-up, sentinel modes, guards)."""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.model_raw.geometry import GEOMETRY, RAW_LEN
from model.vae_teb_prediction.model.model_raw.raw_masks import (
    forecast_mask,
    frontend_mask,
    kl_mask,
    low_rate_mask,
    raw_validity_mask,
)


def _all_valid_weight(b=1):
    return torch.ones(b, GEOMETRY.t_tilde)


def test_frontend_mask_upsamples_and_guards():
    w = _all_valid_weight()
    m = frontend_mask(w, RAW_LEN, GEOMETRY.decimation)
    assert m.shape == (1, RAW_LEN)
    assert m.sum().item() == RAW_LEN
    # Length guard: a trimmed loader (wrong T_tilde) must fail loudly.
    with pytest.raises(ValueError):
        frontend_mask(torch.ones(1, 300), RAW_LEN, GEOMETRY.decimation)


def test_raw_validity_modes():
    w = _all_valid_weight()
    fhr = torch.randn(1, RAW_LEN)
    m_weight = raw_validity_mask(fhr, w, mask_mode="weight_only")
    assert m_weight.sum().item() == RAW_LEN
    # sentinel_refine additionally drops sentinel/non-finite samples.
    fhr2 = fhr.clone()
    fhr2[0, 100] = 0.0  # a sentinel
    fhr2[0, 200] = float("nan")
    m_ref = raw_validity_mask(fhr2, w, mask_mode="sentinel_refine", sentinel=0.0)
    assert m_ref[0, 100] == 0.0 and m_ref[0, 200] == 0.0
    assert m_ref.sum().item() == RAW_LEN - 2
    with pytest.raises(ValueError):
        raw_validity_mask(fhr2, w, mask_mode="sentinel_refine", sentinel=None)
    with pytest.raises(ValueError):
        raw_validity_mask(fhr2, w, mask_mode="bogus")


def test_low_rate_mask_plus_crop_symmetry():
    # Zeroing decimated weight[k] must mask exactly low-rate anchor k - CROP (via m_low = weight[t+CROP]).
    k = 100
    w = _all_valid_weight()
    w[0, k] = 0.0
    m_raw = frontend_mask(w, RAW_LEN, GEOMETRY.decimation)
    m_low = low_rate_mask(m_raw, GEOMETRY)
    anchor = k - GEOMETRY.crop
    assert m_low[0, anchor] == 0.0
    # Its neighbours stay valid.
    assert m_low[0, anchor - 1] == 1.0
    assert m_low[0, anchor + 1] == 1.0


def test_forecast_mask_gates_on_future_block_start():
    # Zeroing weight at the anchor's OWN present (weight[t+CROP]) must NOT gate the forecast (which
    # depends on weight[t+CROP+1 ...]); zeroing a future step must.
    w = _all_valid_weight()
    m_raw = frontend_mask(w, RAW_LEN, GEOMETRY.decimation)
    m_low = low_rate_mask(m_raw, GEOMETRY)
    fmask = forecast_mask(m_raw, m_low, GEOMETRY)
    assert fmask.shape == (1, GEOMETRY.t_valid, GEOMETRY.horizon, GEOMETRY.r)
    # Warm-up anchors are fully masked.
    assert fmask[:, : GEOMETRY.warmup].sum().item() == 0.0
    # A valid mid anchor has a fully-valid forecast block.
    assert fmask[0, 100].min().item() == 1.0

    # Zero the first future step of anchor t=100: weight index = (100 + CROP + 1) = t+CROP+1.
    t = 100
    w2 = _all_valid_weight()
    w2[0, t + GEOMETRY.crop + 1] = 0.0
    m_raw2 = frontend_mask(w2, RAW_LEN, GEOMETRY.decimation)
    m_low2 = low_rate_mask(m_raw2, GEOMETRY)
    fmask2 = forecast_mask(m_raw2, m_low2, GEOMETRY)
    assert fmask2[0, t, 0].sum().item() == 0.0  # tau=0 block gated out
    assert fmask2[0, t, 1].min().item() == 1.0  # tau=1 block still valid


def test_kl_mask_warmup():
    w = _all_valid_weight()
    m_raw = frontend_mask(w, RAW_LEN, GEOMETRY.decimation)
    m_low = low_rate_mask(m_raw, GEOMETRY)
    klm = kl_mask(m_low, GEOMETRY)
    assert klm.shape == (1, GEOMETRY.t)
    assert klm[:, : GEOMETRY.warmup].sum().item() == 0.0
    assert klm[0, GEOMETRY.warmup] == 1.0


def test_all_zero_weight_row():
    w = torch.zeros(1, GEOMETRY.t_tilde)
    m_raw = frontend_mask(w, RAW_LEN, GEOMETRY.decimation)
    m_low = low_rate_mask(m_raw, GEOMETRY)
    fmask = forecast_mask(m_raw, m_low, GEOMETRY)
    klm = kl_mask(m_low, GEOMETRY)
    assert m_raw.sum().item() == 0.0
    assert fmask.sum().item() == 0.0
    assert klm.sum().item() == 0.0
