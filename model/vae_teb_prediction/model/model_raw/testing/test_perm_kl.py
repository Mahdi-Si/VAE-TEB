"""S2-T07: raw ``permutation_kl`` override and its equivalence to the fused ``perm_kl_from_forward``."""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    make_raw_batch,
    make_tiny_raw_model,
)


def _fixed_perm(batch_size: int) -> torch.Tensor:
    # A simple fixed-point-free permutation (reverse) for a deterministic, seed-free control.
    return torch.arange(batch_size - 1, -1, -1, dtype=torch.long)


def test_permutation_kl_runs_and_keys():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch(batch_size=4)
    res = m.permutation_kl(fhr, up, mask, perm_index=_fixed_perm(4))
    assert {"perm_kl", "kld_shuffled", "perm_index"} <= set(res.keys())
    assert torch.isfinite(res["perm_kl"])


def test_permutation_kl_zero_at_init():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch(batch_size=4)
    res = m.permutation_kl(fhr, up, mask, perm_index=_fixed_perm(4))
    # At init q == p regardless of the (permuted) source, so the shuffled KL is ~0.
    assert res["kld_shuffled"].item() < 1e-6


def test_fused_matches_reencoded_reference():
    """The cheap fused control (permute source_state) must equal the eval-time re-encode."""
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch(batch_size=4)
    pi = _fixed_perm(4)

    out = m(fhr, up, mask)
    fused = m.perm_kl_from_forward(out, perm_index=pi)
    reference = m.permutation_kl(fhr, up, mask, perm_index=pi)

    assert torch.allclose(fused["perm_kl"], reference["perm_kl"], atol=1e-5)
    assert torch.allclose(fused["kld_shuffled"], reference["kld_shuffled"], atol=1e-5)


def test_fused_matches_reference_on_trained_like_weights():
    """Break the zero-init symmetry (so KL != 0) and re-check the equivalence holds numerically."""
    m = make_tiny_raw_model().eval()
    # Perturb the posterior delta heads so q != p and the control is non-trivial.
    with torch.no_grad():
        for p in m.posterior_head.delta_mu_head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    fhr, up, mask = make_raw_batch(batch_size=4)
    pi = _fixed_perm(4)

    out = m(fhr, up, mask)
    fused = m.perm_kl_from_forward(out, perm_index=pi)
    reference = m.permutation_kl(fhr, up, mask, perm_index=pi)

    assert reference["kld_shuffled"].item() > 1e-6  # genuinely non-zero now
    assert torch.allclose(fused["perm_kl"], reference["perm_kl"], atol=1e-5)
