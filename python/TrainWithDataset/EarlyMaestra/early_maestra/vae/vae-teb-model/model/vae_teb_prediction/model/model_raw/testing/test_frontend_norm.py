r"""S1-T02a/T02b: the causal front-end norm factory and the forbidden-norm guard.

Every front-end normaliser must be (a) per-sample -- permuting the batch permutes the output
identically -- and (b) causal -- ``out[:, :, t]`` depends only on ``x[:, :, <= t]``. The guard
must reject any time-pooling / batch-coupling norm.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.vae_teb_prediction.model.model_raw.raw_frontend import (
    ChannelAffine,
    CumulativeLayerNorm,
    assert_no_time_pooling_norm,
    make_frontend_norm,
)
from model.vae_teb_prediction.model.model_raw.reuse import CausalGroupNorm


_KINDS = ["causal_group_norm", "cln", "channel_affine"]


# ---------------------------------------------------------------------------
# S1-T02a: norm factory
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", _KINDS)
def test_norm_factory_shape(kind) -> None:
    m = make_frontend_norm(kind, 16, num_groups=8).eval()
    y = m(torch.randn(2, 16, 40))
    assert y.shape == (2, 16, 40)


@pytest.mark.parametrize("kind", _KINDS)
def test_norm_factory_per_sample(kind) -> None:
    # Permuting the batch must permute the output identically (no cross-sample coupling).
    m = make_frontend_norm(kind, 16, num_groups=8).eval()
    x = torch.randn(3, 16, 40)
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(m(x)[perm], m(x[perm]), atol=1e-6)


@pytest.mark.parametrize("kind", _KINDS)
def test_norm_factory_causal(kind) -> None:
    # Backprop a fixed RANDOM PROJECTION of the token, not a plain channel-sum: for the
    # channel-centering norms (causal_group_norm, cln) sum_c out[c, t] is identically zero, so a
    # channel-sum probe gives zero gradient and both the leak and positive-control checks would be
    # vacuous (a full-sequence-pooling regression would still pass). The projection makes both real.
    # 16 channels / 4 groups = 4 channels per group (non-degenerate; groups==channels would make
    # CausalGroupNorm output an input-independent 0, which no probe could exercise).
    m = make_frontend_norm(kind, 16, num_groups=4).eval()
    base = torch.randn(2, 16, 32)
    for t in (0, 5, 20):
        x = base.clone().detach().requires_grad_(True)
        tok = m(x)[:, :, t]  # (B, C)
        gen = torch.Generator().manual_seed(0)
        (tok * torch.randn(tok.shape, generator=gen)).sum().backward()
        assert torch.all(x.grad[:, :, t + 1 :] == 0), f"{kind} leaks the future at t={t}"
        assert torch.any(x.grad[:, :, : t + 1] != 0), f"{kind} vacuously immune at t={t}"


def test_norm_factory_kind_returns_expected_types() -> None:
    assert isinstance(make_frontend_norm("causal_group_norm", 16), CausalGroupNorm)
    assert isinstance(make_frontend_norm("cln", 16), CumulativeLayerNorm)
    assert isinstance(make_frontend_norm("channel_affine", 16), ChannelAffine)


def test_norm_factory_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make_frontend_norm("batchnorm", 16)


def test_norm_factory_rejects_indivisible_groups() -> None:
    # A misconfigured num_groups must fail loudly, not silently degrade to a single group.
    with pytest.raises(ValueError):
        make_frontend_norm("causal_group_norm", 96, num_groups=7)


def test_cln_matches_reference_running_stats() -> None:
    # cLN pools over channels AND cumulative time; check against a naive prefix implementation.
    m = CumulativeLayerNorm(4, eps=1e-5).eval()
    x = torch.randn(2, 4, 10)
    out = m(x)
    B, _, T = x.shape
    ref = torch.empty_like(x)
    for t in range(T):
        window = x[:, :, : t + 1]                       # (B, C, t+1)
        mean = window.mean(dim=(1, 2), keepdim=False)   # (B,)
        var = window.var(dim=(1, 2), unbiased=False)    # (B,)
        for b in range(B):
            ref[b, :, t] = (x[b, :, t] - mean[b]) / torch.sqrt(var[b] + 1e-5)
    assert torch.allclose(out, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# S1-T02b: forbidden-norm guard
# ---------------------------------------------------------------------------
def test_guard_raises_on_batchnorm() -> None:
    bad = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
    with pytest.raises(ValueError):
        assert_no_time_pooling_norm(bad)


def test_guard_raises_on_groupnorm() -> None:
    bad = nn.Sequential(nn.Conv1d(8, 8, 1), nn.GroupNorm(2, 8))
    with pytest.raises(ValueError):
        assert_no_time_pooling_norm(bad)


def test_guard_raises_on_instancenorm() -> None:
    bad = nn.Sequential(nn.Conv1d(4, 4, 1), nn.InstanceNorm1d(4))
    with pytest.raises(ValueError):
        assert_no_time_pooling_norm(bad)


@pytest.mark.parametrize("kind", _KINDS)
def test_guard_passes_on_allowed_norms(kind) -> None:
    ok = nn.Sequential(nn.Conv1d(16, 16, 1), make_frontend_norm(kind, 16, num_groups=8))
    assert_no_time_pooling_norm(ok)  # must not raise


def test_guard_passes_on_channel_layernorm() -> None:
    # A channel-axis LayerNorm (as the inherited v3 core uses) is NOT time-pooling -> allowed.
    ok = nn.Sequential(nn.Conv1d(16, 16, 1), nn.LayerNorm(16))
    assert_no_time_pooling_norm(ok)  # must not raise
