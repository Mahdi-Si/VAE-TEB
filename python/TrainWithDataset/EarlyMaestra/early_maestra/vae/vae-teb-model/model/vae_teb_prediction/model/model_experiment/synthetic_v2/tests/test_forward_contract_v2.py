r"""Sprint 3 (S3-T04b): master forward-contract test.

Asserts the v2 forward emits every v1 key (by name, cross-checked against a live
v1 forward) with v1 shapes, plus the new v2 keys; that the warm-start invariants
hold; and that a proxy-scalar backward produces gradients on the posterior and
source parameters. See ``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (  # noqa: E402
    SeqVaeLagAttnV1,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_NEW_V2_KEYS = {
    "pi_lag", "active_lag_indices", "alpha_bar", "pi_bar", "mu_prior_heads",
    "logvar_prior_heads", "mu_post_active", "logvar_post_active", "kld_lag",
    "kld_content", "expected_lag", "expected_lag_embedding", "lag_entropy",
    "n_active",
}

_SHARED_SHAPE_KEYS = [
    "mu_prior", "logvar_prior", "mu_post", "logvar_post", "z", "target_state",
    "source_state", "decoder_state", "attended_source", "attended_source_heads",
    "attn_weights", "mu_base", "logvar_base", "delta_mu_src", "mu_full",
    "logvar_full", "kld_per_t", "kld_per_t_per_head", "te_lag_map", "warmup_mask",
]


def test_forward_superset_keys_and_shapes() -> None:
    """v2 forward is a superset of v1's keys with matching shapes on shared keys."""
    torch.manual_seed(0)
    B, T = 2, 48
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u = torch.randn(B, T, 101)

    v1 = SeqVaeLagAttnV1(use_entmax=True).eval()
    v2 = SeqVaeLagAttnV2(use_entmax=True).eval()
    v1_out = v1(y_st, y_ph, u)
    v2_out = v2(y_st, y_ph, u)

    missing = set(v1_out.keys()) - set(v2_out.keys())
    assert not missing, f"v2 is missing v1 keys: {missing}"

    for k in _SHARED_SHAPE_KEYS:
        assert v2_out[k].shape == v1_out[k].shape, (
            k, tuple(v2_out[k].shape), tuple(v1_out[k].shape)
        )
    assert v2_out["mu_full"].shape[-1] == 87

    assert _NEW_V2_KEYS <= set(v2_out.keys())
    assert v2_out["active_lag_indices"].shape == (B, T, 4, 8)
    assert v2_out["mu_post_active"].shape == (B, T, 4, 8, 6)
    assert v2_out["expected_lag"].shape == (B, T, 4)
    assert v2_out["expected_lag_embedding"].shape == (B, T, 4 * 32)


def test_forward_warmstart_and_backward() -> None:
    """Warm-start invariants at init; a proxy scalar produces posterior/source grads."""
    torch.manual_seed(0)
    B, T = 2, 48
    y_st = torch.randn(B, T, 43)
    y_ph = torch.randn(B, T, 44)
    u = torch.randn(B, T, 101)

    model = SeqVaeLagAttnV2(use_entmax=True)
    model.eval()
    out = model(y_st, y_ph, u)
    assert float(out["delta_mu_src"].abs().max()) < 1e-6
    assert torch.allclose(out["mu_full"], out["mu_base"], atol=1e-6)

    model.train()
    out2 = model(y_st, y_ph, u)
    proxy = out2["kld_per_t"].mean() + out2["mu_full"].pow(2).mean()
    proxy.backward()

    for name, p in (
        ("lag_posterior.q_proj", model.lag_posterior.q_proj.weight),
        ("lag_latent_head.logvar_head", model.lag_latent_head.logvar_head.weight),
        ("source_encoder.proj", model.source_encoder.proj.weight),
        ("lag_prior.W_h", model.lag_prior.W_h.weight),
    ):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, (
            f"no gradient reached {name}"
        )
