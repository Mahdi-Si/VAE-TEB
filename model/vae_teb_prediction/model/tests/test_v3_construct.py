"""S0-T01b: SeqVaeLagAttnV3 constructs and honours the additive forward/loss contract.

The v3 forward must return every key a live v1 forward returns, plus exactly two additive
keys -- ``kld_active_frac`` (G4) and ``raw_logvar_prior`` (G6) -- with identical shapes;
``compute_loss`` must return the v1 loss keys plus the v3 reporting keys. Keeping this
strict is what lets the existing testing pipeline consume a v3 checkpoint unchanged.
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import SeqVaeLagAttnV3

_V1_LOSS_KEYS = {
    "feat_loss", "base_loss", "kld_loss", "total_loss", "beta",
    "likelihood", "mean_logvar_full", "mean_logvar_base", "lag_smoothness",
}


def test_v3_construct_and_forward_contract(tiny_kwargs, inputs):
    v1 = SeqVaeLagAttnV1(**tiny_kwargs).eval()
    v3 = SeqVaeLagAttnV3(**tiny_kwargs).eval()  # defaults = parity (independent/clamp/full)

    o1 = v1(*inputs)
    o3 = v3(*inputs)

    v1_keys = set(o1.keys())
    v3_keys = set(o3.keys())
    # Additive contract: v3 == v1 keys + exactly {kld_active_frac, raw_logvar_prior}.
    assert v3_keys == v1_keys | {"kld_active_frac", "raw_logvar_prior"}, (
        f"unexpected key delta: {v3_keys.symmetric_difference(v1_keys)}"
    )

    # Every shared tensor key keeps its v1 shape.
    for k, v in o1.items():
        if isinstance(v, torch.Tensor):
            assert o3[k].shape == v.shape, f"shape drift on {k}: {o3[k].shape} vs {v.shape}"

    assert o3["kld_active_frac"].shape == ()
    assert 0.0 <= float(o3["kld_active_frac"]) <= 1.0
    assert o3["raw_logvar_prior"].shape == o3["logvar_prior"].shape


def test_v3_compute_loss_keys(tiny_kwargs, inputs):
    v3 = SeqVaeLagAttnV3(**tiny_kwargs).eval()
    o3 = v3(*inputs)
    losses = v3.compute_loss(o3, inputs[0], inputs[1], beta=0.1, free_bits=0.1)
    assert _V1_LOSS_KEYS.issubset(set(losses.keys()))
    for k in ("kld_raw", "kld_train", "kld_active_frac"):
        assert k in losses, f"missing v3 loss-report key {k}"


def test_v3_construct_no_up_st():
    kw = dict(
        sequence_length=16, d_model=32, d_z=8, horizon=4, warmup_period=2,
        c_y=87, c_u=58, use_up_st=False, max_lag=8, num_heads=4, d_head=8, dropout=0.0,
    )
    v3 = SeqVaeLagAttnV3(**kw).eval()
    g = torch.Generator().manual_seed(0)
    y_st = torch.randn(2, 16, 43, generator=g)
    y_ph = torch.randn(2, 16, 44, generator=g)
    u = torch.randn(2, 16, 58, generator=g)
    o3 = v3(y_st, y_ph, u)
    assert o3["mu_full"].shape == (2, 16, 4, 87)
    assert o3["source_state"].shape == (2, 16, 32)
