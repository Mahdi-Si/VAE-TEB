"""S2-T02 / S2-T03: raw future decoders (shapes, bounds, decoder-head modes, zero-init residual)."""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    TINY_FRONTEND,
    make_raw_batch,
    make_tiny_raw_model,
)


def _logvar_bounds(model):
    lo, hi = model.baseline_decoder.logvar_clamp
    return lo, hi


@pytest.mark.parametrize("decoder_head", ["learned_basis", "linear"])
def test_baseline_decoder_shapes_and_bound(decoder_head):
    m = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": decoder_head}).eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    B, T, Hh, R = 2, m.geometry.t, m.geometry.horizon, m.geometry.r
    assert out["mu_base"].shape == (B, T, Hh, R)
    assert out["logvar_base"].shape == (B, T, Hh, R)
    lo, hi = _logvar_bounds(m)
    assert out["logvar_base"].min() >= lo - 1e-4
    assert out["logvar_base"].max() <= hi + 1e-4


@pytest.mark.parametrize("decoder_head", ["learned_basis", "linear"])
def test_residual_decoder_shapes_and_bound(decoder_head):
    m = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": decoder_head}).eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    B, T, Hh, R = 2, m.geometry.t, m.geometry.horizon, m.geometry.r
    assert out["delta_mu_src"].shape == (B, T, Hh, R)
    assert out["logvar_full"].shape == (B, T, Hh, R)
    lo, hi = _logvar_bounds(m)
    assert out["logvar_full"].min() >= lo - 1e-4
    assert out["logvar_full"].max() <= hi + 1e-4


def test_residual_forward_two_positional_args():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    enc = m.encode_only(fhr, up, mask)
    # 2-arg forward(decoder_state, z), matching ResidualFutureDecoderV3.
    delta_mu, logvar_full = m.residual_decoder(enc["decoder_state"], enc["z"])
    assert delta_mu.shape == (2, m.geometry.t, m.geometry.horizon, m.geometry.r)
    assert logvar_full.shape == delta_mu.shape


@pytest.mark.parametrize("decoder_head", ["learned_basis", "linear"])
def test_residual_delta_zero_at_init(decoder_head):
    m = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": decoder_head}).eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    # mean_head zero-inited -> delta exactly 0 -> mu_full == mu_base, for both head modes.
    assert out["delta_mu_src"].abs().max().item() == 0.0
    assert torch.equal(out["mu_full"], out["mu_base"])


def test_baseline_mean_nonzero_at_init():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    # The baseline mean_head is NOT zeroed; the FHR-only baseline must be a live forecaster.
    assert out["mu_base"].abs().max().item() > 0.0


def test_no_unused_decoder_parameters():
    """Construction-gated head: exactly one of {basis, extra-width mean_head} exists per mode."""
    m_basis = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": "learned_basis"})
    assert hasattr(m_basis.baseline_decoder, "basis")
    assert m_basis.baseline_decoder.mean_head.out_features == m_basis.baseline_decoder.basis_size

    m_lin = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": "linear"})
    assert not hasattr(m_lin.baseline_decoder, "basis")
    assert m_lin.baseline_decoder.mean_head.out_features == m_lin.geometry.r
