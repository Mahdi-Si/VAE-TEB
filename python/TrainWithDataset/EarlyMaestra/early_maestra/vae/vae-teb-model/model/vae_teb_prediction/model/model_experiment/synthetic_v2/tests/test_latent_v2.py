r"""Sprint 3 (S3-T01..T03): continuous latent, content KL, aggregation, residual.

Checks the closed-form content KL and posterior bounds (S3-T01), the latent
aggregation + law-of-total-variance mixture moments (S3-T02), and the source
residual decoder warm start + baseline detachment (S3-T03). See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    LagLatentPosteriorHead,
    SeqVaeLagAttnV2,
    content_gaussian_kl,
)


def _ref_content_kl(mu_q, logvar_q, mu_p, logvar_p):
    r"""Brute-force diagonal-Gaussian $\operatorname{KL}(q\|p)$, prior broadcast over $K_a$."""
    mu_p = mu_p.unsqueeze(-2)
    logvar_p = logvar_p.unsqueeze(-2)
    per_dim = 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp()
        - 1.0
    )
    return per_dim.sum(-1)


def test_content_kl() -> None:
    """Closed-form content KL matches a brute-force reference; non-negative; self-KL 0."""
    torch.manual_seed(0)
    B, T, M, Ka, dz = 2, 5, 4, 8, 6
    mu_q = torch.randn(B, T, M, Ka, dz, dtype=torch.float64)
    logvar_q = torch.randn(B, T, M, Ka, dz, dtype=torch.float64).clamp(-5, 3)
    mu_p = torch.randn(B, T, M, dz, dtype=torch.float64)
    logvar_p = torch.randn(B, T, M, dz, dtype=torch.float64).clamp(-5, 3)

    kz = content_gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
    assert kz.shape == (B, T, M, Ka)
    assert torch.allclose(kz, _ref_content_kl(mu_q, logvar_q, mu_p, logvar_p), atol=1e-9)
    assert torch.all(kz >= -1e-9), "content KL must be non-negative"

    # Self-KL (q == p, broadcast over Ka) is exactly zero.
    mu_pe = mu_p.unsqueeze(-2).expand(B, T, M, Ka, dz).contiguous()
    lv_pe = logvar_p.unsqueeze(-2).expand(B, T, M, Ka, dz).contiguous()
    kz0 = content_gaussian_kl(mu_pe, lv_pe, mu_p, logvar_p)
    assert torch.allclose(kz0, torch.zeros_like(kz0), atol=1e-9)


def test_lag_latent_head_bounds() -> None:
    """Posterior mean residual and log-variance are bounded; zero-init => mu_q == mu_prior."""
    torch.manual_seed(0)
    B, T, M, Ka, dz, d_model, d_v, d_e = 2, 5, 4, 8, 6, 16, 32, 8
    head = LagLatentPosteriorHead(
        d_model=d_model, num_heads=M, d_z_m=dz, d_v=d_v, d_e=d_e,
        delta_mu_scale=3.0, logvar_clamp=(-5.0, 3.0),
    ).eval()
    h_y = torch.randn(B, T, d_model)
    active_v = torch.randn(B, T, M, Ka, d_v)
    e_active = torch.randn(B, T, M, Ka, d_e)
    mu_p = torch.randn(B, T, M, dz)
    logvar_p = torch.randn(B, T, M, dz).clamp(-5, 3)

    mu_q, logvar_q, kz, dmu = head(h_y, active_v, e_active, mu_p, logvar_p)
    assert mu_q.shape == (B, T, M, Ka, dz)
    assert logvar_q.shape == (B, T, M, Ka, dz)
    assert kz.shape == (B, T, M, Ka)
    assert (mu_q - mu_p.unsqueeze(-2)).abs().max().item() <= 3.0 + 1e-5
    assert logvar_q.min() >= -5.0 - 1e-6 and logvar_q.max() <= 3.0 + 1e-6

    # Zero the delta head => Delta mu == 0 => mu_q == mu_prior (the warm-start form).
    torch.nn.init.zeros_(head.delta_mu_head.weight)
    torch.nn.init.zeros_(head.delta_mu_head.bias)
    mu_q2, _, _, dmu2 = head(h_y, active_v, e_active, mu_p, logvar_p)
    assert torch.allclose(mu_q2, mu_p.unsqueeze(-2).expand_as(mu_q2), atol=1e-6)
    assert float(dmu2.abs().max()) == 0.0


def test_aggregation_moments() -> None:
    """Flat mu_post/logvar_post match the law-of-total-variance reference; inference z==mu_post."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    B, T = 2, 40
    out = model(torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))

    ab = out["alpha_bar"]                       # (B,T,M,Ka)
    muq = out["mu_post_active"]                 # (B,T,M,Ka,dz_m)
    lvq = out["logvar_post_active"]
    mu_ref = (ab.unsqueeze(-1) * muq).sum(-2)
    sec = (ab.unsqueeze(-1) * (lvq.exp() + muq ** 2)).sum(-2)
    var_ref = (sec - mu_ref ** 2).clamp_min(model._var_floor)
    lv_ref = var_ref.log().reshape(B, T, 24)
    mu_ref = mu_ref.reshape(B, T, 24)

    assert torch.allclose(out["mu_post"], mu_ref, atol=1e-5)
    assert torch.allclose(out["logvar_post"], lv_ref, atol=1e-5)
    # Inference mode uses posterior means (no sampling noise).
    assert torch.allclose(out["z"], out["mu_post"], atol=1e-6)


def test_residual_decoder_warmstart() -> None:
    """delta_mu_src == 0 and mu_full == mu_base at init; logvar_full defined."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    B, T = 2, 40
    out = model(torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))
    assert out["delta_mu_src"].shape == (B, T, 30, 87)
    assert float(out["delta_mu_src"].abs().max()) < 1e-6
    assert torch.allclose(out["mu_full"], out["mu_base"], atol=1e-6)
    assert out["logvar_full"].shape == (B, T, 30, 87)


def _baseline_meanhead_grad_mag(model, y_st, y_ph, u, detach):
    r"""Backward the full term only (``lambda_base=0``) and return the baseline
    mean-head gradient magnitude (0.0 if ``None``)."""
    model.zero_grad(set_to_none=True)
    out = model(y_st, y_ph, u)
    losses = model.compute_loss(
        out, y_st, y_ph, lambda_full=1.0, lambda_base=0.0,
        compute_kld_loss=False, detach_baseline_in_full=detach,
    )
    losses["total_loss"].backward()
    g = model.baseline_decoder.mean_head.weight.grad
    return 0.0 if g is None else float(g.abs().sum())


def test_baseline_detach_blocks_baseline_grad() -> None:
    """detach_baseline_in_full stops the full term from training baseline-only params."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).train()
    # T must exceed warmup_period + horizon (30 + 30) so the loss mask has valid
    # anchor steps; otherwise every term is masked to zero and no branch sees grad.
    B, T = 2, 80
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u = torch.randn(B, T, 101)

    # Without detachment the full reconstruction reaches the baseline mean head
    # (mu_full = mu_base + delta_mu_src). With detachment its contribution is
    # stop-gradiented, so the baseline mean head receives no (nonzero) gradient
    # from the full term. (lambda_base=0 leaves a zero-valued 0*base_loss edge, so
    # the magnitude, not None-ness, is the load-bearing check.)
    mag_nodetach = _baseline_meanhead_grad_mag(model, y_st, y_ph, u, detach=False)
    mag_detach = _baseline_meanhead_grad_mag(model, y_st, y_ph, u, detach=True)
    assert mag_nodetach > 0.0, "full term should reach baseline mean head without detach"
    assert mag_detach == 0.0, "detached full term must not train the baseline mean head"

    # The residual mean head is on the full-term path in both cases.
    assert model.residual_decoder.mean_head.weight.grad is not None
