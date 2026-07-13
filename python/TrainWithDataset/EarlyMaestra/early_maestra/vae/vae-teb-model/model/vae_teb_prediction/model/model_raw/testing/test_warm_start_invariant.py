"""S2-T05: warm-start / zero-KL invariant (K == 0, delta_mu_src == 0, mu_full == mu_base at init)."""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    TINY_FRONTEND,
    make_raw_batch,
    make_tiny_raw_model,
)


@pytest.mark.parametrize("decoder_head", ["learned_basis", "linear"])
def test_zero_init_invariant(decoder_head):
    torch.manual_seed(0)
    m = make_tiny_raw_model(frontend={**TINY_FRONTEND, "decoder_head": decoder_head}).eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    # G1/G2: zero-init residual mean AND log-var deltas => posterior == prior exactly, K == 0.
    assert torch.allclose(out["mu_post"], out["mu_prior"], atol=1e-6)
    assert torch.allclose(out["logvar_post"], out["logvar_prior"], atol=1e-6)
    assert out["kld_per_t"].abs().max().item() < 1e-6
    # Residual decoder starts at zero, so the full forecast equals the baseline forecast.
    assert out["delta_mu_src"].abs().max().item() == 0.0
    assert torch.allclose(out["mu_full"], out["mu_base"])


def test_construction_does_not_raise():
    # The parent __init__ calls _zero_init_delta_heads before the raw decoders exist (it zeroes the
    # discarded v3 decoder); v4 re-calls it after binding. Neither call must raise.
    make_tiny_raw_model()


def test_residual_mean_head_is_zeroed():
    m = make_tiny_raw_model()
    w = m.residual_decoder.mean_head.weight
    b = m.residual_decoder.mean_head.bias
    assert torch.count_nonzero(w) == 0
    assert b is None or torch.count_nonzero(b) == 0
