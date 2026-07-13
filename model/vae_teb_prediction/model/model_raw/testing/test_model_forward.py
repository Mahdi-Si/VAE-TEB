"""S2-T04 / S2-T06: forward key contract, disable-source ablation, encode_only / TE / batch adapter."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from model.vae_teb_prediction.model.model_raw.testing.conftest import (
    TINY_RAW_LEN,
    make_raw_batch,
    make_tiny_raw_model,
)

_V3_FORWARD_KEYS = {
    "mu_prior",
    "logvar_prior",
    "raw_logvar_prior",
    "mu_post",
    "logvar_post",
    "z",
    "target_state",
    "source_state",
    "decoder_state",
    "attended_source",
    "attended_source_heads",
    "attn_weights",
    "mu_base",
    "logvar_base",
    "delta_mu_src",
    "mu_full",
    "logvar_full",
    "raw_future_pred",
    "kld_per_t",
    "kld_per_t_per_head",
    "te_lag_map",
    "warmup_mask",
    "mu_prior_sat_frac",
    "delta_mu_sat_frac",
    "kld_active_frac",
}


def test_forward_key_contract_and_shapes():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    assert set(out.keys()) == _V3_FORWARD_KEYS
    B, T, Hh, R, dz, nh, L = 2, m.geometry.t, m.geometry.horizon, m.geometry.r, 24, 4, 8 + 1
    assert out["mu_full"].shape == (B, T, Hh, R)
    assert out["attn_weights"].shape == (B, T, nh, L)
    assert out["te_lag_map"].shape == (B, T, L)
    assert out["kld_per_t"].shape == (B, T)
    assert out["z"].shape == (B, T, dz)
    # raw_future_pred is now the (non-null) full raw forecast.
    assert out["raw_future_pred"] is not None
    assert torch.equal(out["raw_future_pred"], out["mu_full"])


def test_disable_source_collapses_kl():
    m = make_tiny_raw_model(disable_source=True).eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    # No-UP ablation: attended source zeroed -> q ~ p -> KL ~ 0.
    assert out["attended_source"].abs().max().item() == 0.0
    assert out["kld_per_t"].abs().max().item() < 1e-5


def test_forward_all_finite():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    out = m(fhr, up, mask)
    for k, v in out.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v).all(), f"non-finite in {k}"


def test_encode_only_contract():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    enc = m.encode_only(fhr, up, mask)
    assert set(enc.keys()) == {
        "mu_prior",
        "logvar_prior",
        "mu_post",
        "logvar_post",
        "z",
        "target_state",
        "source_state",
        "decoder_state",
        "attended_source",
        "attended_source_heads",
        "attn_weights",
    }
    assert enc["z"].shape == (2, m.geometry.t, 24)


def test_measure_transfer_entropy_modes():
    m = make_tiny_raw_model().eval()
    fhr, up, mask = make_raw_batch()
    te = m.measure_transfer_entropy(fhr, up, mask, reduce_mean=False)
    assert te.shape == (2, m.geometry.t, 24)
    # Out-of-support steps (warm-up prefix + final H) are NaN.
    assert torch.isnan(te[:, : m.geometry.warmup]).all()
    assert torch.isnan(te[:, m.geometry.t - m.geometry.horizon :]).all()
    # In-support steps are finite.
    lo, hi = m.geometry.warmup, m.geometry.t - m.geometry.horizon
    assert torch.isfinite(te[:, lo:hi]).all()

    scalar = m.measure_transfer_entropy(fhr, up, mask, reduce_mean=True)
    assert scalar.ndim == 0
    assert torch.isfinite(scalar)


def test_default_batch_to_inputs_and_fit_latent_stats():
    m = make_tiny_raw_model()
    fhr, up, mask = make_raw_batch()
    weight = torch.ones(2, m.geometry.t_tilde)  # decimated validity (B, 32)
    batch = SimpleNamespace(fhr=fhr, up=up, weight=weight)
    inp = m._default_batch_to_inputs(batch)
    assert len(inp) == 3
    fhr_out, up_out, mask_out = inp
    assert fhr_out.shape == (2, TINY_RAW_LEN)
    assert mask_out.shape == (2, TINY_RAW_LEN)

    # The inherited fit_latent_stats must run one batch through the raw encode path w/o shape error.
    # It returns the total number of valid (non-warm-up) time-step samples: B * (T - warmup).
    loader = [batch]
    n = m.fit_latent_stats(loader, max_batches=1, device=torch.device("cpu"))
    assert n == 2 * (m.geometry.t - m.geometry.warmup)
