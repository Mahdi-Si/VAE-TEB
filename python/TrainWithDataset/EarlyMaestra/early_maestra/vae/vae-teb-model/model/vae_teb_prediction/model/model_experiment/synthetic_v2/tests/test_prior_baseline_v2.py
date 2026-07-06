r"""Sprint 1 (S1-T03a/T03b): prior head, lag embedding, and baseline decoder.

Checks :class:`TargetInputAdapterV2` (shape + zero-context identity),
:class:`PriorHeadV2` (bounded per-head prior mean, clamped log-variance, baseline
state), :class:`LagEmbedding` (shared table shape), and the reused v1
:class:`BaselineFutureDecoder` producing ``mu_base`` / ``logvar_base``. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 1.
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
    BaselineFutureDecoder,
    HorizonDecoderCore,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v2 import (  # noqa: E402
    LagEmbedding,
    PriorHeadV2,
    TargetInputAdapterV2,
)


def test_prior_and_embedding() -> None:
    """Prior mean is tanh-bounded, log-variance clamped, context zero-default inert."""
    torch.manual_seed(0)
    B, T = 2, 20
    d_model, M, d_z_m = 32, 4, 6
    mu_scale, clamp = 5.0, (-5.0, 3.0)

    adapter = TargetInputAdapterV2(
        in_dim=87, d_model=d_model, context_dim=5, dropout=0.0
    ).eval()
    y = torch.randn(B, T, 87)
    with torch.no_grad():
        h_none = adapter(y)
        h_zero = adapter(y, torch.zeros(B, T, 5))
    assert h_none.shape == (B, T, d_model)
    assert torch.allclose(h_none, h_zero, atol=1e-6), (
        "zero-default context must be identical to explicit zero context"
    )

    prior = PriorHeadV2(
        d_model=d_model,
        num_heads=M,
        d_z_m=d_z_m,
        mu_scale=mu_scale,
        logvar_clamp=clamp,
        dropout=0.0,
    ).eval()
    h_y = torch.randn(B, T, d_model)
    with torch.no_grad():
        mu_heads, logvar_heads, b_t = prior(h_y)
    assert mu_heads.shape == (B, T, M, d_z_m)
    assert logvar_heads.shape == (B, T, M, d_z_m)
    assert b_t.shape == (B, T, d_model)

    mu_flat = mu_heads.reshape(B, T, M * d_z_m)
    assert mu_flat.shape == (B, T, 24)
    assert mu_flat.abs().max().item() <= mu_scale + 1e-4
    assert logvar_heads.min().item() >= clamp[0] - 1e-6
    assert logvar_heads.max().item() <= clamp[1] + 1e-6

    emb = LagEmbedding(num_lags=91, d_e=32)
    assert emb().shape == (91, 32)


def test_baseline_decoder() -> None:
    """The reused v1 baseline decoder emits ``mu_base`` / ``logvar_base`` (B,T,30,87)."""
    torch.manual_seed(0)
    B, T = 2, 16
    d_model, horizon, c_y = 32, 30, 87
    core = HorizonDecoderCore(
        d_hidden=d_model, horizon=horizon, kernel_size=3, depth=2, film=False
    )
    decoder = BaselineFutureDecoder(
        core=core, d_model=d_model, out_channels=c_y, d_hidden=d_model,
        dropout=0.0, logvar_clamp=(-5.0, 3.0),
    ).eval()
    decoder_state = torch.randn(B, T, d_model)
    with torch.no_grad():
        mu_base, logvar_base = decoder(decoder_state)
    assert mu_base.shape == (B, T, horizon, c_y)
    assert logvar_base.shape == (B, T, horizon, c_y)
    assert torch.isfinite(mu_base).all() and torch.isfinite(logvar_base).all()
