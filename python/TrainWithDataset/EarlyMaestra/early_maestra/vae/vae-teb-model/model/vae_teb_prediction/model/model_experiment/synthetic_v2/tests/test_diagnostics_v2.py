r"""Sprint 7 (S7-T03): extended diagnostics (section 26) + physical-time lag (section 27).

Checks that ``compute_loss`` emits the section-26 diagnostics
($\Delta\mathcal{L}$, $\mathrm{RMS}_{src}$, the $K^R$/$K^Z$ split and ratio
$r_{lag}$ over the RAW unfloored $K^Z$), that the model maps lag indices to
physical seconds ($\mathrm{lag}_{phys}(\ell)=s\ell+\Delta_{UP}$), and that the new
trainer/pl-module log keys are key-presence guarded so the shared metrics builder
stays valid when the ``SeqVaeLagAttn`` alias is flipped back to v1. See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` S7-T03.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

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

_T = 40   # > warmup + horizon so the loss masks are non-empty

_COMMON = {
    "sequence_length": _T,
    "d_model": 16,
    "d_z": 4,
    "horizon": 4,
    "warmup_period": 2,
    "c_y": 87,
    "c_u": 101,
    "use_up_st": True,
    "max_lag": 8,
    "num_heads": 2,
    "d_head": 8,
    "dropout": 0.0,
    "decoder_hidden": 16,
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
}

_TINY_V2: Dict[str, Any] = {
    **_COMMON,
    "use_entmax": True,
    "target_encoder_blocks": 2,
    "target_kernel": 3,
    "target_dilations": (1, 2),
    "source_scales": (3, 5),
    "d_u": 16,
    "d_k": 8,
    "d_e": 8,
    "active_lags": 4,
    "active_lags_warmup": 6,
    "kappa_z": 0.05,
    "step_seconds": 4.0,
    "delta_up_seconds": 20.0,
}

_TINY_V1: Dict[str, Any] = {**_COMMON, "use_entmax": False}


def _streams(n: int = 2, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n, _T, 43, generator=g),
        torch.randn(n, _T, 44, generator=g),
        torch.randn(n, _T, 101, generator=g),
    )


# ---------------------------------------------------------------------------
# Section-26 diagnostics in compute_loss
# ---------------------------------------------------------------------------
def test_compute_loss_diagnostics_present_and_finite() -> None:
    r"""``compute_loss`` returns delta_l / rms_src / r_lag / K^R / raw K^Z, all finite."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_TINY_V2)
    y_st, y_ph, u = _streams(seed=1)
    out = model(y_st, y_ph, u)
    loss = model.compute_loss(
        forward_outputs=out, y_st=y_st, y_ph=y_ph,
        weight=torch.ones(2, _T), beta=5.0e-2, lambda_full=1.0, lambda_base=0.5,
        likelihood="gaussian_nll", sigma_obs=1.0, detach_baseline_in_full=True,
        lambda_lag=1.0e-3,
    )
    for key in ("delta_l", "rms_src", "r_lag", "kld_lag_loss",
                "kld_content_loss", "kld_content_raw"):
        assert key in loss, key
        assert torch.isfinite(torch.as_tensor(loss[key])), key
    # r_lag = E[K^R] / E[K^R + K^Z] is a fraction.
    assert 0.0 <= float(loss["r_lag"]) <= 1.0
    # rms_src is a non-negative RMS.
    assert float(loss["rms_src"]) >= 0.0


# ---------------------------------------------------------------------------
# Physical-time lag (section 27)
# ---------------------------------------------------------------------------
def test_physical_lag_axis_and_expected_lag_seconds() -> None:
    r"""$\mathrm{lag}_{phys}(\ell) = 4\ell + 20$: monotone, step 4, offset 20."""
    model = SeqVaeLagAttnV2(**_TINY_V2)
    axis = model.physical_lag_axis()
    assert axis.shape == (model.L,)
    assert float(axis[0]) == pytest.approx(20.0)
    assert float(axis[1]) == pytest.approx(24.0)
    assert float(axis[-1]) == pytest.approx(4.0 * (model.L - 1) + 20.0)
    diffs = (axis[1:] - axis[:-1])
    assert torch.all(diffs > 0)                     # strictly increasing
    assert torch.allclose(diffs, torch.full_like(diffs, 4.0))

    # expected_lag_seconds applies the same affine map elementwise.
    ell = torch.tensor([[0.0, 2.5], [8.0, 4.0]])
    sec = model.expected_lag_seconds(ell)
    assert torch.allclose(sec, 4.0 * ell + 20.0)


def test_physical_lag_axis_default_offset() -> None:
    r"""With the default ``delta_up_seconds=0`` the axis is pure ``4*ell``."""
    model = SeqVaeLagAttnV2(**{**_TINY_V2, "delta_up_seconds": 0.0})
    axis = model.physical_lag_axis()
    assert float(axis[0]) == pytest.approx(0.0)
    assert float(axis[3]) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Key-presence guarding in the production trainer's metrics builder
# ---------------------------------------------------------------------------
def _prod_module(model):
    trainer_mod = pytest.importorskip(
        "model.vae_teb_prediction.model.trainer_lag_attn_v1"
    )
    module = trainer_mod.SeqVaeLagAttnPl(model, lr=1e-3, lr_milestones=[])
    module.hparams["kld_beta"] = 5.0e-2
    module.hparams["likelihood"] = "gaussian_nll"
    module.hparams["sigma_obs"] = 1.0
    module.hparams["detach_baseline_in_full"] = True
    return module


def _batch(n: int = 2, *, seed: int = 0):
    from types import SimpleNamespace
    y_st, y_ph, u_full = _streams(n, seed=seed)
    # u_full is [up_st | up_ph]; split back for the production batch fields.
    return SimpleNamespace(
        fhr_st=y_st, fhr_ph=y_ph,
        up_st=u_full[..., :43], up_ph=u_full[..., 43:],
        weight=torch.ones(n, _T),
    )


_V2_DIAG_KEYS = (
    "kld_lag_loss", "kld_content_raw", "r_lag", "rms_src",
    "lag_entropy", "n_active", "expected_lag_mean",
)


def test_metrics_has_v2_keys_under_v2() -> None:
    r"""Under a v2 model the guarded diagnostics appear in the metrics dict."""
    torch.manual_seed(0)
    module = _prod_module(SeqVaeLagAttnV2(**_TINY_V2))
    _, metrics = module.compute_loss_and_metrics(_batch(seed=1), 0, stage="train")
    for key in _V2_DIAG_KEYS:
        assert key in metrics, key
        assert torch.isfinite(torch.as_tensor(metrics[key])), key


def test_metrics_guarded_under_v1() -> None:
    r"""Under a v1 model the shared builder runs with NO v2 keys and no KeyError."""
    torch.manual_seed(0)
    module = _prod_module(SeqVaeLagAttnV1(**_TINY_V1))
    total_loss, metrics = module.compute_loss_and_metrics(
        _batch(seed=1), 0, stage="train"
    )
    assert torch.isfinite(total_loss)
    for key in _V2_DIAG_KEYS:
        assert key not in metrics, f"v2-only key {key} leaked under v1"
