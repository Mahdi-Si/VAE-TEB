r"""Sprint 3 (S3-T04a): exact KL decomposition and additivity identities.

Checks the two load-bearing identities $K_t = \sum_m(K^R + K^Z)$ and
$\sum_\ell K_{t,\ell} = K_t$, both on the standalone :class:`TEDecompositionHead`
with hand-built active tensors and end-to-end through the full model forward. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 3.
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
    SeqVaeLagAttnV2,
    TEDecompositionHead,
    discrete_lag_kl,
)


def test_decomposition_head_identities() -> None:
    """Standalone head: additivity holds for arbitrary active weights and content KL."""
    torch.manual_seed(0)
    B, T, M, Ka, L = 2, 6, 4, 5, 20
    eps = 1e-8
    # Distinct active indices per (b,t,m) via topk of random scores.
    active_idx = torch.rand(B, T, M, L, dtype=torch.float64).topk(Ka, dim=-1).indices
    alpha_bar = torch.rand(B, T, M, Ka, dtype=torch.float64)
    alpha_bar = alpha_bar / alpha_bar.sum(-1, keepdim=True)
    pi_bar = torch.rand(B, T, M, Ka, dtype=torch.float64)
    pi_bar = pi_bar / pi_bar.sum(-1, keepdim=True)
    kz = torch.rand(B, T, M, Ka, dtype=torch.float64)          # content KL >= 0
    kld_lag = discrete_lag_kl(alpha_bar, pi_bar, eps=eps)

    head = TEDecompositionHead(num_heads=M, num_lags=L, eps=eps)
    d = head(alpha_bar, pi_bar, kz, kld_lag, active_idx)

    assert d["kld_content"].shape == (B, T, M)
    assert d["kld_per_t_per_head"].shape == (B, T, M)
    assert d["kld_per_t"].shape == (B, T)
    assert d["te_lag_map"].shape == (B, T, L)

    assert torch.allclose(d["kld_content"], (alpha_bar * kz).sum(-1), atol=1e-12)
    # K_t == sum_m (K^R + K^Z).
    assert torch.allclose(
        d["kld_per_t"], (kld_lag + d["kld_content"]).sum(-1), atol=1e-10
    )
    # sum_l te_lag_map == K_t (load-bearing).
    assert (d["te_lag_map"].sum(-1) - d["kld_per_t"]).abs().max().item() < 1e-9


@pytest.mark.parametrize("use_entmax", [True, False])
def test_model_decomposition_identities(use_entmax) -> None:
    """End-to-end forward: both additivity identities hold; KL terms non-negative."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=use_entmax).eval()
    B, T = 2, 50
    out = model(torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))

    err1 = (
        out["kld_per_t"] - (out["kld_lag"] + out["kld_content"]).sum(-1)
    ).abs().max().item()
    err2 = (out["te_lag_map"].sum(-1) - out["kld_per_t"]).abs().max().item()
    assert err1 < 1e-4, err1
    assert err2 < 1e-4, err2

    assert torch.all(out["kld_lag"] >= -1e-6)
    assert torch.all(out["kld_content"] >= -1e-6)
    assert torch.all(out["kld_per_t"] >= -1e-6)
