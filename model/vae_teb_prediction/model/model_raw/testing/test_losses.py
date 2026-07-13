"""S3-T03..T06: raw NLL, KL adapter, low-pass/smoothness, and the assembled compute_loss."""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.model_raw.raw_losses import (
    kld_terms,
    lowpass_loss,
    raw_nll,
    smooth_loss,
)
from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    make_raw_batch,
    make_tiny_raw_model,
)

_LOSS_KEYS = {
    "feat_loss",
    "raw_loss",
    "base_loss",
    "kld_loss",
    "kld_raw",
    "kld_train",
    "kld_active_frac",
    "lowpass_loss",
    "smooth_loss",
    "lag_smoothness",
    "raw_mae",
    "mean_logvar_full",
    "mean_logvar_base",
    "total_loss",
    "beta",
}


def _shapes(b=2, tv=24, h=4, r=16):
    return b, tv, h, r


# -- S3-T03: raw NLL --------------------------------------------------------
def test_raw_nll_finite_and_mean_logvar():
    b, tv, h, r = _shapes()
    mu = torch.randn(b, tv, h, r)
    logvar = torch.zeros(b, tv, h, r)
    target = torch.randn(b, tv, h, r)
    mask = torch.ones(b, tv, h, r)
    loss, mean_logvar = raw_nll(mu, logvar, target, mask)
    assert torch.isfinite(loss)
    # logvar == 0 -> NLL reduces to 0.5 * masked MSE.
    mse = ((mu - target) ** 2).mean()
    assert torch.allclose(loss, 0.5 * mse, atol=1e-5)
    assert torch.allclose(mean_logvar, torch.zeros(()))


def test_raw_nll_nan_safe():
    b, tv, h, r = _shapes()
    mu = torch.randn(b, tv, h, r)
    logvar = torch.zeros(b, tv, h, r)
    target = torch.randn(b, tv, h, r)
    mask = torch.ones(b, tv, h, r)
    # Poison masked-out positions with NaN/sentinel; they must not leak into the loss.
    mask[0, 0] = 0.0
    target[0, 0] = float("nan")
    loss, _ = raw_nll(mu, logvar, target, mask)
    assert torch.isfinite(loss)


def test_raw_nll_all_zero_mask_contributes_zero():
    b, tv, h, r = _shapes()
    mu = torch.randn(b, tv, h, r)
    logvar = torch.zeros(b, tv, h, r)
    target = torch.randn(b, tv, h, r)
    full = torch.ones(b, tv, h, r)
    zeroed = full.clone()
    zeroed[0] = 0.0  # drop sample 0 entirely
    loss_all, _ = raw_nll(mu, logvar, target, full)
    loss_one = raw_nll(mu[1:], logvar[1:], target[1:], full[1:])[0]
    loss_masked = raw_nll(mu, logvar, target, zeroed)[0]
    assert torch.allclose(loss_masked, loss_one, atol=1e-6)


# -- S3-T04: KL adapter -----------------------------------------------------
def test_kld_terms_nonneg_and_ordering():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    # Break the zero-init symmetry so KL > 0.
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.posterior_head.delta_mu_head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    out = m(fhr, up, mask)
    kl_w = torch.ones(2, m.geometry.t)
    kld_train, kld_raw, _ = kld_terms(m, out, weight=kl_w, free_bits=0.1)
    assert kld_train >= 0 and kld_raw >= 0
    # Free bits floor each per-dim KL, so the trained term is >= the raw (un-floored) term.
    assert kld_train >= kld_raw - 1e-6


def test_kld_terms_disabled():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    kld_train, kld_raw, _ = kld_terms(m, out, compute_kld_loss=False)
    assert kld_train.item() == 0.0 and kld_raw.item() == 0.0


# -- S3-T05: low-pass + smoothness -----------------------------------------
def test_lowpass_and_smooth_zero_when_equal():
    b, tv, h, r = _shapes()
    x = torch.randn(b, tv, h, r)
    mask = torch.ones(b, tv, h, r)
    assert torch.allclose(lowpass_loss(x, x, mask), torch.zeros(()), atol=1e-6)
    assert torch.allclose(smooth_loss(x, x, mask), torch.zeros(()), atol=1e-6)


def test_lowpass_positive_when_different():
    b, tv, h, r = _shapes()
    x = torch.randn(b, tv, h, r)
    y = x + 1.0
    mask = torch.ones(b, tv, h, r)
    assert lowpass_loss(x, y, mask).item() > 0.0
    assert smooth_loss(x, y * 2, mask).item() >= 0.0


# -- S3-T06: assembled compute_loss ----------------------------------------
def test_compute_loss_key_contract_and_finite():
    m = make_tiny_raw_model()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    loss = m.compute_loss(out, fhr, mask, beta=0.1, free_bits=0.1, lambda_lag=1e-3)
    assert set(loss.keys()) == _LOSS_KEYS
    for k, v in loss.items():
        assert torch.isfinite(v).all(), f"non-finite {k}"
    # feat_loss is the full-NLL alias.
    assert torch.equal(loss["feat_loss"], loss["raw_loss"])


def test_compute_loss_backward_reaches_front_ends_and_decoders():
    m = make_tiny_raw_model()
    # Break the zero-init warm-start symmetry so the source path is live (at strict init the source
    # has zero influence by design -- G1 -- and legitimately receives no gradient). Perturb the
    # posterior delta-mean and the residual mean heads to emulate a slightly-trained model.
    torch.manual_seed(2)
    with torch.no_grad():
        for p in m.posterior_head.delta_mu_head.parameters():
            p.add_(0.1 * torch.randn_like(p))
        for p in m.residual_decoder.mean_head.parameters():
            p.add_(0.1 * torch.randn_like(p))
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    loss = m.compute_loss(out, fhr, mask, beta=0.1, free_bits=0.1, lambda_lag=1e-3)
    loss["total_loss"].backward()
    # Gradients must reach the front ends, both encoders, the posterior, and both decoders.
    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert _has_grad(m.frontend_y)
    assert _has_grad(m.frontend_u)
    assert _has_grad(m.target_encoder)
    assert _has_grad(m.source_encoder)
    assert _has_grad(m.posterior_head)
    assert _has_grad(m.baseline_decoder)
    assert _has_grad(m.residual_decoder)


def test_compute_loss_nan_safe_under_gaps():
    m = make_tiny_raw_model()
    fhr, up, mask = make_raw_batch()
    # Introduce a raw gap: zero part of the validity mask (and NaN the signal there).
    mask[:, 200:260] = 0.0
    fhr = fhr.clone()
    fhr[:, 200:260] = float("nan")
    out = m(fhr.nan_to_num(), up, mask)  # the front end also sanitises internally
    loss = m.compute_loss(out, fhr, mask, beta=0.1, free_bits=0.1)
    assert torch.isfinite(loss["total_loss"]).all()
