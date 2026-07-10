r"""Sprint 0 (S0-T02): self-contained ``entmax15`` + softmax fallback.

Checks the simplex/sparsity properties, a fully-masked row (all ``-inf``) returning
zeros without NaNs, gradient correctness on well-separated inputs, forward/backward
agreement with the reference ``entmax`` package (when importable), and fp32
real-magnitude stability. See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` S0-T02.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    entmax15,
    sparse_normalize,
)


def test_sum_to_one_nonneg_sparse() -> None:
    """Rows sum to 1, are non-negative, and produce exact zeros when separated."""
    torch.manual_seed(0)
    scores = torch.randn(4, 7, 91)
    # Make one lag dominate per row so the sparse map zeroes far-off entries.
    scores[..., 0] += 8.0
    p = entmax15(scores, dim=-1)
    assert torch.all(p >= 0.0)
    sums = p.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    # Sparsity: some entries are exactly zero.
    assert (p == 0.0).any()


def test_fully_masked_row() -> None:
    """An all-``-inf`` row maps to all zeros with a finite (zero) gradient."""
    scores = torch.zeros(3, 5, requires_grad=True)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[1, :] = True  # row 1 fully masked
    masked = scores.masked_fill(mask, float("-inf"))
    p = entmax15(masked, dim=-1)
    assert torch.isfinite(p).all()
    assert torch.allclose(p[1], torch.zeros(5))
    # Other rows still normalise to 1.
    assert torch.allclose(p[0].sum(), torch.tensor(1.0), atol=1e-6)
    loss = p.sum()
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_invalid_entries_get_zero_mass() -> None:
    """Masked (non-finite) entries receive exactly zero probability."""
    scores = torch.randn(2, 10)
    scores[:, 5:] = float("-inf")
    p = entmax15(scores, dim=-1)
    assert torch.all(p[:, 5:] == 0.0)
    assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_grad_matches_reference() -> None:
    """``gradcheck`` passes on well-separated float64 inputs (stable support)."""
    torch.manual_seed(1)
    # Well-separated so the support size is locally constant (avoids the
    # non-smooth support boundary that would trip a finite-difference check).
    x = torch.tensor(
        [[3.0, 1.0, -1.0, -4.0], [2.5, 0.5, -2.0, -5.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda t: entmax15(t, dim=-1), (x,), atol=1e-6)


@pytest.mark.parametrize("shape", [(6, 91), (3, 4, 40)])
def test_agrees_with_entmax_package(shape) -> None:
    """Forward and backward match the reference ``entmax`` package within 1e-5."""
    entmax_pkg = pytest.importorskip("entmax")
    torch.manual_seed(2)
    base = torch.randn(*shape, dtype=torch.float64)

    x_ref = base.clone().requires_grad_(True)
    x_ours = base.clone().requires_grad_(True)

    p_ref = entmax_pkg.entmax15(x_ref, dim=-1)
    p_ours = entmax15(x_ours, dim=-1)
    assert torch.allclose(p_ref, p_ours, atol=1e-5)

    # Backward: contract with a fixed random cotangent and compare input grads.
    g = torch.randn(*shape, dtype=torch.float64)
    (p_ref * g).sum().backward()
    (p_ours * g).sum().backward()
    assert torch.allclose(x_ref.grad, x_ours.grad, atol=1e-5)


def test_fp32_realmag_stability() -> None:
    """fp32 scores of realistic magnitude stay finite and normalised."""
    torch.manual_seed(3)
    scores = torch.randn(8, 4, 91, dtype=torch.float32) * 5.0
    p = entmax15(scores, dim=-1)
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(dim=-1), torch.ones(8, 4), atol=1e-6)


def test_sparse_normalize_softmax_fallback() -> None:
    """``sparse_normalize(use_entmax=False)`` matches ``softmax`` exactly."""
    torch.manual_seed(4)
    scores = torch.randn(5, 91)
    got = sparse_normalize(scores, dim=-1, use_entmax=False)
    assert torch.allclose(got, torch.softmax(scores, dim=-1))
    # entmax path differs (sparse) but still normalises.
    sp = sparse_normalize(scores, dim=-1, use_entmax=True)
    assert torch.allclose(sp.sum(dim=-1), torch.ones(5), atol=1e-6)
