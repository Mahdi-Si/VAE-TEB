"""S0-T01a: subclassed bound-only heads reproduce v1 under the parity settings.

Each v3 head is a structural subclass of its v1 counterpart, so transferring the v1 head's
weights (``load_state_dict``) and running the same input must reproduce v1's outputs exactly
under ``logvar_bound='clamp'`` (and ``posterior_logvar='independent'`` for the posterior).
"""
from __future__ import annotations

import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import (
    BaselineFutureDecoder,
    HorizonDecoderCore,
    PosteriorHead,
    PriorHead,
    ResidualFutureDecoder,
)
from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import (
    BaselineFutureDecoderV3,
    PosteriorHeadV3,
    PriorHeadV3,
    ResidualFutureDecoderV3,
)

_ATOL = 1e-6
_DM, _DZ, _NH, _DH = 32, 8, 4, 8
_B, _T = 2, 5


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_prior_head_v3_matches_v1_clamp():
    torch.manual_seed(0)
    v1 = PriorHead(d_model=_DM, d_z=_DZ, dropout=0.0).eval()
    v3 = PriorHeadV3(d_model=_DM, d_z=_DZ, dropout=0.0, logvar_bound="clamp").eval()
    v3.load_state_dict(v1.state_dict(), strict=True)

    h = _rand(_B, _T, _DM)
    mu1, lv1, ds1 = v1(h)
    mu3, lv3, ds3, raw3 = v3(h)

    assert torch.allclose(mu1, mu3, atol=_ATOL)
    assert torch.allclose(lv1, lv3, atol=_ATOL)
    assert torch.allclose(ds1, ds3, atol=_ATOL)
    # The extra return is the *pre-bound* raw prior logvar; clamp(raw) == lv3.
    assert raw3.shape == lv3.shape
    assert torch.allclose(torch.clamp(raw3, -5.0, 3.0), lv3, atol=_ATOL)


def test_posterior_head_v3_matches_v1_independent_flat():
    torch.manual_seed(0)
    v1 = PosteriorHead(d_model=_DM, d_z=_DZ, dropout=0.0, head_structured=False).eval()
    v3 = PosteriorHeadV3(
        d_model=_DM, d_z=_DZ, dropout=0.0, head_structured=False,
        logvar_bound="clamp", posterior_logvar="independent",
    ).eval()
    v3.load_state_dict(v1.state_dict(), strict=True)

    h = _rand(_B, _T, _DM, seed=1)
    a = _rand(_B, _T, _DM, seed=2)
    mu_prior = _rand(_B, _T, _DZ, seed=3)
    mu1, lv1 = v1(h, a, mu_prior)
    mu3, lv3 = v3(h, a, mu_prior)  # raw_logvar_prior ignored in independent mode

    assert torch.allclose(mu1, mu3, atol=_ATOL)
    assert torch.allclose(lv1, lv3, atol=_ATOL)


def test_posterior_head_v3_matches_v1_independent_head_structured():
    torch.manual_seed(0)
    v1 = PosteriorHead(
        d_model=_DM, d_z=_DZ, dropout=0.0, head_structured=True,
        num_heads=_NH, d_head=_DH,
    ).eval()
    v3 = PosteriorHeadV3(
        d_model=_DM, d_z=_DZ, dropout=0.0, head_structured=True,
        num_heads=_NH, d_head=_DH,
        logvar_bound="clamp", posterior_logvar="independent",
    ).eval()
    v3.load_state_dict(v1.state_dict(), strict=True)

    h = _rand(_B, _T, _DM, seed=1)
    a = _rand(_B, _T, _NH, _DH, seed=2)
    mu_prior = _rand(_B, _T, _DZ, seed=3)
    mu1, lv1 = v1(h, a, mu_prior)
    mu3, lv3 = v3(h, a, mu_prior)

    assert torch.allclose(mu1, mu3, atol=_ATOL)
    assert torch.allclose(lv1, lv3, atol=_ATOL)


def _make_core():
    return HorizonDecoderCore(d_hidden=_DM, horizon=4, kernel_size=3, depth=2, film=False)


def test_baseline_decoder_v3_matches_v1_clamp():
    torch.manual_seed(0)
    core = _make_core()
    v1 = BaselineFutureDecoder(core=core, d_model=_DM, out_channels=87, d_hidden=_DM, dropout=0.0).eval()
    v3 = BaselineFutureDecoderV3(
        core=core, d_model=_DM, out_channels=87, d_hidden=_DM, dropout=0.0, logvar_bound="clamp"
    ).eval()
    v3.load_state_dict(v1.state_dict(), strict=True)

    ds = _rand(_B, _T, _DM, seed=5)
    mu1, lv1 = v1(ds)
    mu3, lv3 = v3(ds)
    assert torch.allclose(mu1, mu3, atol=_ATOL)
    assert torch.allclose(lv1, lv3, atol=_ATOL)


def test_residual_decoder_v3_matches_v1_clamp():
    torch.manual_seed(0)
    core = _make_core()
    v1 = ResidualFutureDecoder(core=core, d_model=_DM, d_z=_DZ, out_channels=87, d_hidden=_DM, dropout=0.0).eval()
    v3 = ResidualFutureDecoderV3(
        core=core, d_model=_DM, d_z=_DZ, out_channels=87, d_hidden=_DM, dropout=0.0, logvar_bound="clamp"
    ).eval()
    v3.load_state_dict(v1.state_dict(), strict=True)

    ds = _rand(_B, _T, _DM, seed=5)
    z = _rand(_B, _T, _DZ, seed=6)
    mu1, lv1 = v1(ds, z)
    mu3, lv3 = v3(ds, z)
    assert torch.allclose(mu1, mu3, atol=_ATOL)
    assert torch.allclose(lv1, lv3, atol=_ATOL)
