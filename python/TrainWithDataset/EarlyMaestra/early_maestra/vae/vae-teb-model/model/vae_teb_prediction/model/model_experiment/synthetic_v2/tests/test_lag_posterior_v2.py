r"""Sprint 2 (S2-T01..T04): explicit lag posterior and discrete lag KL.

Checks the smooth Gaussian-basis lag bias (S2-T01), the strided-view lag scores +
entmax posterior (S2-T02), the target-only lag prior + discrete KL (S2-T03), and the
top-$K_a$ active set + emitted truncated $K^R$ (S2-T04). See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

import pytest  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    LagPosteriorAttention,
    LagPriorHead,
    SmoothLagBias,
    discrete_lag_kl,
    entmax15,
    sparse_normalize,
)

_CENTERS = (0, 4, 8, 12, 16, 24, 32, 40, 52, 64, 78, 90)


def _dense_alpha(attn, q, k, lag_bias):
    r"""Materialised-bank reference for ``alpha`` (portable, tiny dims only)."""
    B, T, M, d_k = q.shape
    L = attn.L
    k_full = q.new_zeros(B, T, L, M, d_k)
    for lag in range(L):
        k_full[:, lag:, lag] = k[:, : T - lag]  # k_full[b,t,l] = k[b, t-l]
    scores = torch.einsum("btmd,btlmd->btml", q, k_full) * attn.scale
    scores = scores + lag_bias[None, None, :, :]
    valid = attn.valid_mask(T, L, q.device)
    scores = scores.masked_fill(~valid[None, :, None, :], -1e9)
    return sparse_normalize(scores, dim=-1, use_entmax=attn.use_entmax)


def test_smooth_bias() -> None:
    """Bias shape ``(M, L)``; penalty non-negative, finite, zero when flat."""
    torch.manual_seed(0)
    M, L = 4, 91
    bias = SmoothLagBias(num_heads=M, num_lags=L, centers=_CENTERS)

    b = bias()
    assert b.shape == (M, L)

    # At init theta = 0 => flat bias => zero smoothness penalty.
    pen0 = bias.smoothness_penalty()
    assert torch.isfinite(pen0)
    assert float(pen0) == 0.0

    # A constant per-head offset (theta scaled so the bias is near-constant) still
    # has a tiny penalty; a rough random theta gives a finite, non-negative one.
    with torch.no_grad():
        bias.theta.copy_(torch.randn(M, bias.theta.shape[1]))
    pen = bias.smoothness_penalty()
    assert torch.isfinite(pen)
    assert float(pen) >= 0.0
    assert float(pen) > 0.0  # a non-trivial bias is not flat

    # Derived widths are all positive and floored at 2.
    widths = SmoothLagBias._derive_widths(
        torch.tensor([float(c) for c in _CENTERS])
    )
    assert torch.all(widths >= 2.0)
    assert widths.numel() == len(_CENTERS)


def test_smooth_bias_single_center() -> None:
    """A single lag-basis center builds without IndexError and yields a finite bias."""
    bias = SmoothLagBias(num_heads=2, num_lags=5, centers=(2.0,))
    b = bias()
    assert b.shape == (2, 5)
    assert torch.isfinite(b).all()
    w = SmoothLagBias._derive_widths(torch.tensor([2.0]))
    assert w.shape == (1,) and float(w[0]) >= 2.0


@pytest.mark.parametrize("use_entmax", [True, False])
def test_alpha_scores(use_entmax) -> None:
    """Shapes, simplex over valid lags, invalid-lag zeros, and dense-ref agreement."""
    torch.manual_seed(0)
    B, T, d_model, M, d_k, d_v, L = 2, 12, 16, 4, 8, 8, 9
    attn = LagPosteriorAttention(
        d_model=d_model, num_heads=M, d_k=d_k, d_v=d_v, num_lags=L,
        use_entmax=use_entmax,
    ).double().eval()
    h_y = torch.randn(B, T, d_model, dtype=torch.float64)
    r_u = torch.randn(B, T, d_model, dtype=torch.float64)
    lag_bias = torch.randn(M, L, dtype=torch.float64)

    with torch.no_grad():
        alpha, v = attn(h_y, r_u, lag_bias)
    assert alpha.shape == (B, T, M, L)
    assert v.shape == (B, T, M, d_v)

    # Rows sum to 1 over valid lags; invalid lags are exactly 0.
    assert torch.allclose(alpha.sum(-1), torch.ones(B, T, M, dtype=torch.float64), atol=1e-6)
    valid = attn.valid_mask(T, L, alpha.device)              # (T, L)
    invalid = ~valid[None, :, None, :].expand_as(alpha)
    assert torch.all(alpha[invalid] == 0.0)

    # Strided scoring equals the dense materialised-bank reference.
    q, k, _ = attn.project(h_y, r_u)
    alpha_ref = _dense_alpha(attn, q, k, lag_bias)
    assert torch.allclose(alpha, alpha_ref, atol=1e-5)


def test_alpha_forced_score_concentrates() -> None:
    """A large lag-bias spike at l* concentrates ``alpha`` at l*."""
    torch.manual_seed(0)
    B, T, d_model, M, d_k, d_v, L = 2, 12, 16, 4, 8, 8, 9
    attn = LagPosteriorAttention(
        d_model=d_model, num_heads=M, d_k=d_k, d_v=d_v, num_lags=L, use_entmax=True,
    ).eval()
    h_y = torch.randn(B, T, d_model)
    r_u = torch.randn(B, T, d_model)
    ell_star = 5
    lag_bias = torch.zeros(M, L)
    lag_bias[:, ell_star] = 50.0
    with torch.no_grad():
        alpha, _ = attn(h_y, r_u, lag_bias)
    # For positions where lag l* is valid (t >= l*), alpha concentrates at l*.
    valid_t = alpha[:, ell_star:, :, :]
    peak = valid_t.argmax(dim=-1)
    assert torch.all(peak == ell_star), "alpha did not concentrate at the forced lag"
    assert torch.all(valid_t[..., ell_star] > 0.99)


def test_lag_prior_kl() -> None:
    """Prior is a strictly-positive simplex, target-only; discrete KR is non-negative."""
    torch.manual_seed(0)
    B, T, d_model, M, L, d_e = 2, 12, 16, 4, 9, 8
    prior = LagPriorHead(
        d_model=d_model, num_heads=M, num_lags=L, d_e=d_e
    ).double().eval()
    h_y = torch.randn(B, T, d_model, dtype=torch.float64)
    e = torch.randn(L, d_e, dtype=torch.float64)

    pi = prior(h_y, e)
    assert pi.shape == (B, T, M, L)
    assert torch.allclose(pi.sum(-1), torch.ones(B, T, M, dtype=torch.float64), atol=1e-6)
    assert torch.all(pi > 0.0), "softmax prior must be strictly positive"

    # Discrete KL against an entmax posterior: non-negative; zero when alpha == pi.
    alpha = entmax15(torch.randn(B, T, M, L, dtype=torch.float64), dim=-1)
    kr = discrete_lag_kl(alpha, pi, eps=1e-8)
    assert kr.shape == (B, T, M)
    assert torch.all(kr >= -1e-6), "discrete KL must be non-negative"
    kr_self = discrete_lag_kl(pi, pi, eps=1e-8)
    assert torch.allclose(kr_self, torch.zeros_like(kr_self), atol=1e-6)

    # Target-only: pi has no gradient path to a source stream.
    r_u = torch.randn(B, T, d_model, dtype=torch.float64, requires_grad=True)
    pi2 = prior(h_y, e)
    (grad_ru,) = torch.autograd.grad(
        pi2.sum(), r_u, allow_unused=True, retain_graph=True
    )
    assert grad_ru is None, "prior must not depend on the source stream"


def test_active_set() -> None:
    """Top-Ka gather, renormalisation, emitted truncated K^R, and settable Ka."""
    torch.manual_seed(0)
    B, T, d_model, M, d_k, d_v, L = 2, 12, 16, 4, 8, 8, 9
    attn = LagPosteriorAttention(
        d_model=d_model, num_heads=M, d_k=d_k, d_v=d_v, num_lags=L, use_entmax=True,
    ).double().eval()
    prior = LagPriorHead(
        d_model=d_model, num_heads=M, num_lags=L, d_e=8
    ).double().eval()
    h_y = torch.randn(B, T, d_model, dtype=torch.float64)
    r_u = torch.randn(B, T, d_model, dtype=torch.float64)
    e = torch.randn(L, 8, dtype=torch.float64)
    lag_bias = torch.zeros(M, L, dtype=torch.float64)

    with torch.no_grad():
        alpha, v = attn(h_y, r_u, lag_bias)
        pi = prior(h_y, e)

    Ka = 4
    out = attn.select_active(alpha, pi, v, active_lags=Ka)
    assert out["active_lag_indices"].shape == (B, T, M, Ka)
    assert out["active_v"].shape == (B, T, M, Ka, d_v)
    assert out["kld_lag"].shape == (B, T, M)

    # Renormalised weights sum to 1 over the active set for positions with >= Ka
    # valid lags (t >= Ka - 1). K^R is non-negative.
    for name in ("alpha_bar", "pi_bar"):
        s = out[name][:, Ka:, :, :].sum(-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-6), name
    assert torch.all(out["kld_lag"] >= -1e-6)

    # Active-value gather matches an explicit reference (no full-L bank).
    idx = out["active_lag_indices"]
    ref_v = torch.zeros(B, T, M, Ka, d_v, dtype=torch.float64)
    for b in range(B):
        for t in range(T):
            for m in range(M):
                for kk in range(Ka):
                    lag = int(idx[b, t, m, kk])
                    if t - lag >= 0:
                        ref_v[b, t, m, kk] = v[b, t - lag, m]
    assert torch.allclose(out["active_v"], ref_v, atol=1e-9)

    # Emitted kld_lag equals KL(alpha_bar || pi_bar).
    kr_ref = discrete_lag_kl(out["alpha_bar"], out["pi_bar"], eps=1e-8)
    assert torch.allclose(out["kld_lag"], kr_ref, atol=1e-9)

    # Ka is settable: a larger active count yields a wider active set.
    out16 = attn.select_active(alpha, pi, v, active_lags=8)
    assert out16["active_lag_indices"].shape == (B, T, M, 8)


def test_model_active_lags_settable() -> None:
    """The model exposes a settable ``active_lags`` attribute (default 8)."""
    from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import SeqVaeLagAttnV2

    model = SeqVaeLagAttnV2()
    assert model.active_lags == 8
    model.active_lags = 16
    assert model.active_lags == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only peak-memory check")
def test_peak_memory() -> None:
    """Strided scoring peak stays well below a dense ``(B, T, L, d_model)`` bank."""
    dev = torch.device("cuda")
    B, T, d_model, M, d_k, d_v, L = 32, 300, 128, 4, 16, 32, 91
    attn = LagPosteriorAttention(
        d_model=d_model, num_heads=M, d_k=d_k, d_v=d_v, num_lags=L, use_entmax=True,
    ).to(dev).eval()
    h_y = torch.randn(B, T, d_model, device=dev)
    r_u = torch.randn(B, T, d_model, device=dev)
    lag_bias = torch.zeros(M, L, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        attn(h_y, r_u, lag_bias)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(dev)
    dense_bank_bytes = B * T * L * d_model * 4  # a materialised (B,T,L,d) fp32 bank
    assert peak < dense_bank_bytes, (
        f"peak {peak / 1e6:.1f} MB >= dense bank {dense_bank_bytes / 1e6:.1f} MB"
    )
