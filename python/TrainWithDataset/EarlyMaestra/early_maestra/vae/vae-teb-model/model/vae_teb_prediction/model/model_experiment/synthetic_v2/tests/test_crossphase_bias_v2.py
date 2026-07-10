r"""Sprint 7 (S7-T01): cross-phase score-only lag bias (arch spec section 10).

Checks the default-off ablation and the source-purity invariant. The bias
$\rho^{(m)}_{t,\ell}$ is added to the lag SCORES only, so:

* Off (default): ``x_cross`` is ignored -- forward is bit-identical with and
  without it, and ``model.crossphase_bias is None``.
* On: $\rho$ reaches the lag posterior $\alpha$ (hence lag-driven quantities such
  as ``expected_lag``), but NOT the source values (``source_state``) or the
  per-lag latent content (``mu_post_active``) -- the latent content stays UP-only.

Note: a naive ``grad(attn_weights.sum(), x_cross)`` is a false test -- each row of
``attn_weights`` is a simplex summing to 1, so its sum is constant and its
gradient is identically zero. The positive test therefore uses ``expected_lag``
(a genuine, non-constant function of $\alpha$). See
``vae-teb-lag-attn-v2-spec-and-sprints.md`` S7-T01 and decision 2.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

_T = 32
_C_CROSS = 79

_TINY_KW: Dict[str, Any] = {
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
    "use_entmax": True,
    "horizon_depth": 1,
    "horizon_kernel": 3,
    "horizon_film": False,
    "target_encoder_blocks": 2,
    "target_kernel": 3,
    "target_dilations": (1, 2),
    "source_scales": (3, 5),
    "d_u": 16,
    "d_k": 8,
    "d_e": 8,
    "active_lags": 4,
    "active_lags_warmup": 6,
    "c_cross": _C_CROSS,
}


def _streams(n: int = 2, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    y_st = torch.randn(n, _T, 43, generator=g)
    y_ph = torch.randn(n, _T, 44, generator=g)
    u_stream = torch.randn(n, _T, 101, generator=g)
    return y_st, y_ph, u_stream


# ---------------------------------------------------------------------------
# Default-off: x_cross is ignored
# ---------------------------------------------------------------------------
def test_crossphase_off_by_default_ignores_x_cross() -> None:
    r"""With the flag off, ``crossphase_bias is None`` and ``x_cross`` is a no-op."""
    model = SeqVaeLagAttnV2(**_TINY_KW)   # use_crossphase_bias defaults False
    assert model.crossphase_bias is None
    model.eval()

    y_st, y_ph, u = _streams(seed=1)
    x_cross = torch.randn(2, _T, _C_CROSS)
    with torch.no_grad():
        out_none = model(y_st, y_ph, u)
        out_xc = model(y_st, y_ph, u, x_cross=x_cross)
    # x_cross must not change anything when the feature is off.
    for key in ("attn_weights", "expected_lag", "mu_full", "z"):
        assert torch.allclose(out_none[key], out_xc[key], atol=0.0), key


def test_crossphase_constructs_when_enabled() -> None:
    r"""Enabling the flag builds the module (no longer raises NotImplementedError)."""
    model = SeqVaeLagAttnV2(**_TINY_KW, use_crossphase_bias=True)
    assert model.crossphase_bias is not None


# ---------------------------------------------------------------------------
# On: gradient-path (rho -> alpha only; content stays UP-only)
# ---------------------------------------------------------------------------
def _forward_with_grad(model, x_cross):
    y_st, y_ph, u = _streams(seed=2)
    return model(y_st, y_ph, u, x_cross=x_cross)


def test_crossphase_reaches_alpha_but_not_content() -> None:
    r"""$\rho$ influences lag-driven ``expected_lag`` but never values / content."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_TINY_KW, use_crossphase_bias=True)
    model.eval()   # posterior means -> deterministic graph, grads still flow

    x_cross = torch.randn(2, _T, _C_CROSS, requires_grad=True)
    out = _forward_with_grad(model, x_cross)

    # POSITIVE: expected_lag (a non-constant function of alpha) depends on x_cross.
    (g_lag,) = torch.autograd.grad(
        out["expected_lag"].sum(), x_cross, retain_graph=True, allow_unused=True,
    )
    assert g_lag is not None and float(g_lag.abs().sum()) > 0.0

    # NEGATIVE (value purity): the source state r^u never sees cross-phase.
    (g_src,) = torch.autograd.grad(
        out["source_state"].sum(), x_cross, retain_graph=True, allow_unused=True,
    )
    assert g_src is None

    # NEGATIVE (content purity): per-lag posterior means are cross-phase-independent
    # (the only path from x_cross into the active set is the non-differentiable topk).
    (g_content,) = torch.autograd.grad(
        out["mu_post_active"].sum(), x_cross, retain_graph=True, allow_unused=True,
    )
    assert g_content is None


def test_crossphase_none_bypasses_module() -> None:
    r"""With ``x_cross=None`` the bias module carries no gradient (bypassed)."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnV2(**_TINY_KW, use_crossphase_bias=True)
    model.eval()
    y_st, y_ph, u = _streams(seed=3)
    out = model(y_st, y_ph, u, x_cross=None)
    (g,) = torch.autograd.grad(
        out["expected_lag"].sum(), model.crossphase_bias.w_rho, allow_unused=True,
    )
    assert g is None   # crossphase params unused when x_cross is None
