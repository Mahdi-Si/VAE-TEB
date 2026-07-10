r"""Sprint 4 (S4-T05): helper-method parity and latent-statistics buffers.

Checks ``encode_only`` shapes, the v2 ``measure_transfer_entropy`` ``(B, T)``
contract, that a training-mode forward updates ``mu_post_running_count``, and that
``fit_latent_stats`` normalizes correctly and refuses zero-sample loaders. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` Sprint 4.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)


def _batch(B, T):
    return SimpleNamespace(
        fhr_st=torch.randn(B, T, 43),
        fhr_ph=torch.randn(B, T, 44),
        up_st=torch.randn(B, T, 43),
        up_ph=torch.randn(B, T, 58),
    )


def test_encode_only_shapes() -> None:
    """encode_only returns the latent + lag quantities with correct shapes."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    B, T = 2, 60
    y_st, y_ph, u = torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101)
    enc = model.encode_only(y_st, y_ph, u, sample_z=False)
    assert enc["mu_post"].shape == (B, T, 24)
    assert enc["z"].shape == (B, T, 24)
    assert torch.allclose(enc["z"], enc["mu_post"])          # sample_z=False -> mean
    assert enc["attn_weights"].shape == (B, T, 4, 91)
    assert enc["kld_per_t"].shape == (B, T)
    assert enc["kld_lag"].shape == (B, T, 4)


def test_measure_transfer_entropy_contract() -> None:
    """v2 measure_transfer_entropy returns (B, T) (warm-up NaN) or a scalar mean."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True)
    B, T = 2, 60
    y_st, y_ph, u = torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101)

    te = model.measure_transfer_entropy(y_st, y_ph, u, reduce_mean=False)
    assert te.shape == (B, T)
    w = model.warmup_period
    assert torch.isnan(te[:, :w]).all()
    assert not torch.isnan(te[:, w:]).any()

    te_mean = model.measure_transfer_entropy(y_st, y_ph, u, reduce_mean=True)
    assert te_mean.ndim == 0 and torch.isfinite(te_mean)


def test_training_forward_updates_running_count() -> None:
    """A training-mode forward aggregates non-warm-up steps into the running count."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_entmax=True).train()
    B, T = 2, 60
    c0 = int(model.mu_post_running_count.item())
    model(torch.randn(B, T, 43), torch.randn(B, T, 44), torch.randn(B, T, 101))
    c1 = int(model.mu_post_running_count.item())
    assert c1 > c0


def test_fit_latent_stats_and_normalize() -> None:
    """fit_latent_stats aggregates and yields near zero-mean / unit-std normalization."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(use_up_st=True, use_entmax=True).eval()
    B, T = 2, 60
    loader = [_batch(B, T) for _ in range(3)]

    n = model.fit_latent_stats(loader)
    valid = int(model._build_warmup_valid_mask(T).sum())
    assert n == 3 * B * valid
    assert int(model.mu_post_running_count.item()) == n

    # Mean-centering: normalizing the running mean gives ~0.
    z0 = model.normalize_latent(model.mu_post_running_mean)
    assert torch.allclose(z0, torch.zeros_like(z0), atol=1e-4)

    # Unit-std over the actual fitting data.
    mus = []
    with torch.no_grad():
        for b in loader:
            u = torch.cat([b.up_st, b.up_ph], dim=-1)
            enc = model.encode_only(b.fhr_st, b.fhr_ph, u, sample_z=False)
            vt = model._build_warmup_valid_mask(enc["mu_post"].size(1))
            mus.append(enc["mu_post"][:, vt, :].reshape(-1, 24))
    zn = model.normalize_latent(torch.cat(mus, dim=0))
    assert zn.mean(0).abs().max().item() < 1e-3
    assert (zn.std(0, unbiased=False) - 1.0).abs().max().item() < 1e-2


def test_fit_latent_stats_refuses_zero_samples() -> None:
    """A loader with no non-warm-up steps raises RuntimeError."""
    model = SeqVaeLagAttnV2(use_up_st=True).eval()
    short_loader = [_batch(2, 20)]                            # T=20 < warmup_period=30
    with pytest.raises(RuntimeError):
        model.fit_latent_stats(short_loader)
