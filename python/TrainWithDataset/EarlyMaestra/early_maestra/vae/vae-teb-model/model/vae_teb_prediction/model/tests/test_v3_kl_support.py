"""S2-T01: anchor-aligned training-KL support.

``kld_support='full'`` masks only the warm-up prefix (v1 behaviour); ``kld_support='anchor'``
additionally drops the final ``horizon`` steps (``[warmup, T-H)``). The reduce-mean KL
denominator counts ``d_z * mask.sum()`` over the configured support, and ``'full'``
reproduces v1's denominator/value exactly.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def _latents(d_z, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = 16
    B = 2
    return (
        torch.randn(B, T, d_z, generator=g),           # mu_prior
        torch.randn(B, T, d_z, generator=g) * 0.3,     # logvar_prior
        torch.randn(B, T, d_z, generator=g),           # mu_post
        torch.randn(B, T, d_z, generator=g) * 0.3,     # logvar_post
    )


def test_support_masks(tiny_kwargs):
    T, warmup, H = 16, tiny_kwargs["warmup_period"], tiny_kwargs["horizon"]
    m_full = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="full"))._kld_support_mask(T)
    m_anchor = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="anchor"))._kld_support_mask(T)

    assert m_full.sum().item() == float(T - warmup)                 # [warmup, T)
    assert m_anchor.sum().item() == float((T - H) - warmup)         # [warmup, T-H)
    assert torch.all(m_full[:warmup] == 0.0) and torch.all(m_full[warmup:] == 1.0)
    assert torch.all(m_anchor[T - H:] == 0.0)
    assert torch.all(m_anchor[warmup:T - H] == 1.0)


def test_reduce_mean_denominator(tiny_kwargs):
    d_z = tiny_kwargs["d_z"]
    T, warmup, H = 16, tiny_kwargs["warmup_period"], tiny_kwargs["horizon"]
    lat = _latents(d_z)

    B = lat[0].shape[0]
    for support in ("full", "anchor"):
        model = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support=support))
        got = model._kld_loss(*lat, reduce_mean=True, free_bits=0.0)

        kld = model.kld_tensor(*lat, mask_warmup=False)
        # Denominator counts d_z over every in-support (batch, step) entry (v1 convention:
        # mask_btd.sum() * d_z == B * support_count * d_z).
        mask_btd = model._kld_support_mask(T).view(1, T, 1).expand(B, T, 1)
        expected = (kld * mask_btd).sum() / (mask_btd.sum() * d_z)
        assert torch.allclose(got, expected, atol=1e-6), f"denominator mismatch ({support})"


def test_full_support_matches_v1(tiny_kwargs):
    d_z = tiny_kwargs["d_z"]
    lat = _latents(d_z, seed=3)
    v1 = SeqVaeLagAttnV1(**tiny_kwargs)
    v3_full = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="full"))

    k1 = v1._kld_loss(*lat, reduce_mean=True, free_bits=0.1)
    k3 = v3_full._kld_loss(*lat, reduce_mean=True, free_bits=0.1)
    assert torch.allclose(k1, k3, atol=1e-6), "full-support KL diverges from v1"


def test_measure_transfer_entropy_support_consistent(tiny_kwargs, inputs):
    # For an anchor model the scalar (reduce_mean=True) and the per-step curve
    # (reduce_mean=False) must share the same time support: warm-up + tail are NaN, and the
    # scalar equals the nanmean of the per-step per-dim KL.
    model = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="anchor")).eval()
    T, warmup, H = 16, tiny_kwargs["warmup_period"], tiny_kwargs["horizon"]

    scalar = model.measure_transfer_entropy(*inputs, reduce_mean=True)
    tensor = model.measure_transfer_entropy(*inputs, reduce_mean=False)  # (B, T, d_z)

    assert torch.isnan(tensor[:, :warmup, :]).all(), "warm-up not masked"
    assert torch.isnan(tensor[:, T - H:, :]).all(), "anchor tail not masked"
    assert torch.isfinite(tensor[:, warmup:T - H, :]).all(), "support region has NaNs"
    assert torch.allclose(scalar, torch.nanmean(tensor), atol=1e-6), (
        "scalar TE is not the nanmean of the per-step curve"
    )


def test_anchor_excludes_tail(tiny_kwargs):
    # A KL spike confined to the final horizon steps must not affect the anchor KL,
    # but must raise the full-support KL.
    d_z = tiny_kwargs["d_z"]
    T, H = 16, tiny_kwargs["horizon"]
    mu_prior, logvar_prior, mu_post, logvar_post = _latents(d_z, seed=7)
    mu_post = mu_post.clone()
    mu_post[:, T - H:, :] += 50.0  # huge posterior drift in the tail only

    anchor = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="anchor"))
    full = SeqVaeLagAttnV3(**dict(tiny_kwargs, kld_support="full"))
    k_anchor = anchor._kld_loss(mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=True)
    k_full = full._kld_loss(mu_prior, logvar_prior, mu_post, logvar_post, reduce_mean=True)
    assert k_full > k_anchor, "tail spike should inflate full support but not anchor support"
