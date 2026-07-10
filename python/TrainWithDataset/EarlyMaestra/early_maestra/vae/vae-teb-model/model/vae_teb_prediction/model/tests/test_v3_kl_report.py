"""S2-T02 / S2-T03: additive kld_active_frac key + raw/train KL reporting.

The forward gains exactly one additive key (``kld_active_frac``); ``kld_per_t`` stays the raw
full-T sum and ``te_lag_map`` still derives from it. ``compute_loss`` reports ``kld_raw`` /
``kld_train`` / ``kld_active_frac``: ``kld_train >= kld_raw`` (free-bit floor), and only
``kld_train`` carries a gradient into ``total_loss`` (``kld_raw`` is detached).
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3


def _prod_model(tiny_kwargs, head_structured=False):
    return SeqVaeLagAttnV3(
        **dict(
            tiny_kwargs,
            head_structured_latent=head_structured,
            posterior_logvar="residual",
            logvar_bound="smooth",
            kld_support="anchor",
        )
    ).eval()


def test_kld_active_frac_present_and_bounded(tiny_kwargs, inputs):
    out = _prod_model(tiny_kwargs)(*inputs)
    assert "kld_active_frac" in out
    kaf = out["kld_active_frac"]
    assert kaf.shape == ()
    assert 0.0 <= float(kaf) <= 1.0


def test_kld_per_t_is_raw_full_t_sum(tiny_kwargs, inputs):
    model = _prod_model(tiny_kwargs)
    out = model(*inputs)
    kld_btd = model.kld_tensor(
        out["mu_prior"], out["logvar_prior"], out["mu_post"], out["logvar_post"],
        mask_warmup=False,
    )
    # kld_per_t is the raw (un-floored, full-T) per-step KL summed over latent dims.
    assert torch.allclose(out["kld_per_t"], kld_btd.sum(-1), atol=1e-6)


def test_te_lag_map_derives_from_raw_kld(tiny_kwargs, inputs):
    model = _prod_model(tiny_kwargs)
    out = model(*inputs)
    kld_btd = model.kld_tensor(
        out["mu_prior"], out["logvar_prior"], out["mu_post"], out["logvar_post"],
        mask_warmup=False,
    )
    _, te_ref, _ = model.te_analysis(
        kld_btd, out["attn_weights"], head_structured=model.head_structured_latent
    )
    assert torch.allclose(out["te_lag_map"], te_ref, atol=1e-6)


def test_kld_train_ge_raw_and_gradient_path(tiny_kwargs, inputs):
    model = _prod_model(tiny_kwargs)
    out = model(*inputs)
    losses = model.compute_loss(out, inputs[0], inputs[1], beta=0.1, free_bits=0.1)

    for k in ("kld_raw", "kld_train", "kld_active_frac"):
        assert k in losses

    assert float(losses["kld_train"]) >= float(losses["kld_raw"]) - 1e-6

    # Only kld_train feeds total_loss; kld_raw is a detached diagnostic.
    assert not losses["kld_raw"].requires_grad
    assert losses["kld_train"].requires_grad
    assert losses["total_loss"].requires_grad
