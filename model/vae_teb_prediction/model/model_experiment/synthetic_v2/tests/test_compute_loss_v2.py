r"""Sprint 4 (S4-T01/T02): decomposed-KL compute_loss and lag regularizers.

Checks the v1-compatible return keys plus the new v2 keys, the warm-start
``feat_loss == base_loss`` identity, the content-only free-bits floor, the
``mse``/``gaussian_nll`` branches, the ``free_bits`` deprecation, and the TV /
entropy / bias regularizers. See ``vae-teb-lag-attn-v2-spec-and-sprints.md``
Sprint 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.vae_teb_lag_attn_trfr import (  # noqa: E402
    SeqVaeLagAttnV2,
)

# T must exceed warmup_period + horizon (30 + 30) so the loss mask is non-empty.
_B, _T = 2, 80

_V1_KEYS = {
    "feat_loss", "base_loss", "kld_loss", "total_loss", "beta", "likelihood",
    "mean_logvar_full", "mean_logvar_base", "lag_smoothness",
}
_NEW_KEYS = {"kld_lag_loss", "kld_content_loss", "lag_tv", "lag_entropy_reg"}


def _inputs():
    torch.manual_seed(0)
    return (
        torch.randn(_B, _T, 43),
        torch.randn(_B, _T, 44),
        torch.randn(_B, _T, 101),
    )


def test_return_keys_and_warmstart() -> None:
    """All v1 + new keys present and finite; feat_loss == base_loss at init."""
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    y_st, y_ph, u = _inputs()
    out = model(y_st, y_ph, u)
    losses = model.compute_loss(out, y_st, y_ph, beta=0.05)

    assert _V1_KEYS <= set(losses.keys())
    assert _NEW_KEYS <= set(losses.keys())
    for k in _V1_KEYS | _NEW_KEYS:
        if k == "likelihood":
            continue
        assert torch.isfinite(torch.as_tensor(losses[k])).all(), k
    # Warm start: mu_full == mu_base => feat_loss == base_loss.
    assert torch.allclose(losses["feat_loss"], losses["base_loss"], atol=1e-6)


@pytest.mark.parametrize("likelihood", ["mse", "gaussian_nll"])
def test_likelihood_branches(likelihood) -> None:
    """Both likelihoods run, including gaussian_nll with learned sigma."""
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    y_st, y_ph, u = _inputs()
    out = model(y_st, y_ph, u)
    sigma = "learned" if likelihood == "gaussian_nll" else 1.0
    losses = model.compute_loss(
        out, y_st, y_ph, beta=0.05, likelihood=likelihood, sigma_obs=sigma
    )
    assert torch.isfinite(losses["feat_loss"])
    assert torch.isfinite(losses["total_loss"])


def test_free_bits_floors_content_only() -> None:
    """kappa_z floors the content KL, never the lag KL."""
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    y_st, y_ph, u = _inputs()
    out = model(y_st, y_ph, u)

    model.kappa_z = 0.0
    lo = model.compute_loss(out, y_st, y_ph, beta=0.05)
    model.kappa_z = 10.0
    hi = model.compute_loss(out, y_st, y_ph, beta=0.05)

    assert hi["kld_content_loss"] > lo["kld_content_loss"]
    # Lag KL is unfloored -> identical across kappa_z.
    assert torch.allclose(hi["kld_lag_loss"], lo["kld_lag_loss"], atol=1e-6)


def test_free_bits_kwarg_deprecated() -> None:
    """A non-zero free_bits kwarg emits a deprecation warning and is ignored."""
    model = SeqVaeLagAttnV2(use_entmax=True).eval()
    y_st, y_ph, u = _inputs()
    out = model(y_st, y_ph, u)
    with pytest.warns(RuntimeWarning):
        losses = model.compute_loss(out, y_st, y_ph, beta=0.05, free_bits=0.1)
    assert torch.isfinite(losses["total_loss"])


def test_regularizers() -> None:
    """TV / entropy / bias terms compute, enter total_loss, and skip on zero weight."""
    model = SeqVaeLagAttnV2(
        use_entmax=True, lambda_tv=1e-3, lambda_ent=1e-3
    ).eval()
    y_st, y_ph, u = _inputs()
    out = model(y_st, y_ph, u)
    losses = model.compute_loss(out, y_st, y_ph, beta=0.05, lambda_lag=1e-3)
    assert torch.isfinite(losses["lag_tv"]) and float(losses["lag_tv"]) >= 0.0
    assert torch.isfinite(losses["lag_entropy_reg"])
    assert torch.isfinite(losses["lag_smoothness"])

    # Zero weights => the regularizer terms are exact zeros (skipped compute).
    model.lambda_tv = 0.0
    model.lambda_ent = 0.0
    losses0 = model.compute_loss(out, y_st, y_ph, beta=0.05, lambda_lag=0.0)
    assert float(losses0["lag_tv"]) == 0.0
    assert float(losses0["lag_entropy_reg"]) == 0.0
    assert float(losses0["lag_smoothness"]) == 0.0


def test_new_weights_autoforward_via_signature() -> None:
    """kappa_z / lambda_tv / lambda_ent are real constructor params (config-forwardable)."""
    import inspect

    params = set(inspect.signature(SeqVaeLagAttnV2.__init__).parameters)
    assert {"kappa_z", "lambda_tv", "lambda_ent"} <= params
