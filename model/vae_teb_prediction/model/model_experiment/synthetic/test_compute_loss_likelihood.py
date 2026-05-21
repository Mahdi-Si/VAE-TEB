"""Pytest checks for ``SeqVaeLagAttnV1.compute_loss`` likelihood switch.

Sprint 5.1 of ``model_validation_v2_plan.md``: ``compute_loss`` gained a
``likelihood ∈ {'mse', 'gaussian_nll'}`` kwarg, a ``sigma_obs`` (positive
float or the literal string ``'learned'``), and a ``free_bits >= 0.0`` KL
floor. The tests below verify:

* The MSE path is bit-exact with the pre-Sprint-5 ``\\|\\hat y - y\\|^2``
  formula (no silent renormalisation).
* Gaussian-NLL with a fixed unit ``sigma_obs`` matches the MSE feat / base
  losses up to the additive ``0.5 \\ln(\\sigma_{obs}^2) = 0`` constant.
* Gaussian-NLL with a non-unit fixed ``sigma_obs`` matches the closed-form
  ``0.5\\,\\mathrm{MSE}/\\sigma^2 + 0.5\\ln\\sigma^2``.
* Gaussian-NLL with ``sigma_obs='learned'`` is finite, differentiable, and
  reuses the model's ``logvar_full`` / ``logvar_base`` heads (verified by
  checking gradients flow through them, which they did not in the MSE path).
* ``free_bits = 0.0`` is a no-op (per-dim Gaussian KL is non-negative);
  ``free_bits > 0`` raises the KL floor monotonically.
* Bad arguments raise ``ValueError``.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v1 import SeqVaeLagAttnV1


# Use a small but realistic shape: full T to exercise the unfold path,
# warmup mask, and the diagonal-Gaussian KL aggregator.
_B = 2
_T = 80          # short enough to keep CPU runtime sub-second
_HORIZON = 8
_WARMUP = 4


@pytest.fixture(scope="module")
def model() -> SeqVaeLagAttnV1:
    """A tiny but architecturally-faithful instance for loss-only tests."""
    torch.manual_seed(0)
    m = SeqVaeLagAttnV1(
        sequence_length=_T,
        d_model=32,
        d_z=8,
        horizon=_HORIZON,
        warmup_period=_WARMUP,
        c_y=87,
        c_u=101,
        use_up_st=True,
        max_lag=24,
        num_heads=4,
        d_head=8,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch() -> dict:
    """Fixed-seed inputs so all tests share the same forward pass."""
    torch.manual_seed(1)
    y_st = torch.randn(_B, _T, 43)
    y_ph = torch.randn(_B, _T, 44)
    u_stream = torch.randn(_B, _T, 101)
    return {"y_st": y_st, "y_ph": y_ph, "u_stream": u_stream}


@pytest.fixture(scope="module")
def forward_out(model: SeqVaeLagAttnV1, batch: dict) -> dict:
    with torch.no_grad():
        out = model(batch["y_st"], batch["y_ph"], batch["u_stream"])
    return out


# ----------------------------------------------------------------------
# 1. MSE path: backwards-compat sanity
# ----------------------------------------------------------------------

def test_mse_default_matches_explicit(model, batch, forward_out):
    """Default kwargs and ``likelihood='mse'`` produce identical numbers."""
    default = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"]
    )
    explicit = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"],
        likelihood="mse", sigma_obs=1.0,
    )
    for key in ("feat_loss", "base_loss", "kld_loss", "total_loss"):
        torch.testing.assert_close(default[key], explicit[key], rtol=0, atol=0)
    assert default["likelihood"] == "mse"


def test_mse_recovers_masked_mean(model, batch, forward_out):
    """The MSE feat_loss equals the explicit masked-mean formula."""
    out = forward_out
    losses = model.compute_loss(
        out, batch["y_st"], batch["y_ph"], likelihood="mse",
    )
    # Reconstruct the masked MSE by hand and confirm equality.
    y_st, y_ph = batch["y_st"], batch["y_ph"]
    Y = torch.cat([y_st, y_ph], dim=-1)
    Y_plus = Y[:, 1:, :].unfold(dimension=1, size=_HORIZON, step=1)
    Y_plus = Y_plus.permute(0, 1, 3, 2).contiguous()
    T_valid = _T - _HORIZON
    mu_full = out["mu_full"][:, :T_valid, :, :]
    diff = (mu_full - Y_plus) ** 2
    warmup_t = torch.zeros(T_valid)
    warmup_t[_WARMUP:] = 1.0
    # ``compute_loss`` uses ``mask_feat`` of shape (B, T_valid, H_d, 1) and
    # divides by ``mask_feat.sum() * C`` — the (1)-axis broadcasts into the
    # channel sum in the numerator, so the matching denom counts mask
    # once and multiplies by C.
    mask = warmup_t[None, :, None, None].expand(_B, T_valid, _HORIZON, 1)
    denom = mask.sum() * Y.shape[-1]
    expected = (diff * mask).sum() / denom
    torch.testing.assert_close(losses["feat_loss"], expected, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------
# 2. Gaussian-NLL: scalar sigma_obs
# ----------------------------------------------------------------------

def test_nll_unit_sigma_matches_half_mse(model, batch, forward_out):
    """At ``sigma_obs=1.0``, NLL per element = 0.5 * (y-mu)^2 + 0. So
    feat_loss_NLL == 0.5 * feat_loss_MSE (both branches mask + normalise the
    same way)."""
    mse = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], likelihood="mse",
    )
    nll = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"],
        likelihood="gaussian_nll", sigma_obs=1.0,
    )
    torch.testing.assert_close(
        nll["feat_loss"], 0.5 * mse["feat_loss"], rtol=1e-6, atol=1e-7,
    )
    torch.testing.assert_close(
        nll["base_loss"], 0.5 * mse["base_loss"], rtol=1e-6, atol=1e-7,
    )
    # KL term is unaffected by likelihood switch.
    torch.testing.assert_close(
        nll["kld_loss"], mse["kld_loss"], rtol=1e-6, atol=1e-7,
    )
    assert nll["likelihood"] == "gaussian_nll"


def test_nll_nonunit_sigma_matches_closed_form(model, batch, forward_out):
    """``sigma_obs=2.0`` ⇒ per-element loss = 0.5 * MSE/4 + 0.5 * log(4)."""
    sigma = 2.0
    mse = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], likelihood="mse",
    )
    nll = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"],
        likelihood="gaussian_nll", sigma_obs=sigma,
    )
    expected_feat = (
        0.5 * mse["feat_loss"] / (sigma ** 2) + 0.5 * math.log(sigma ** 2)
    )
    expected_base = (
        0.5 * mse["base_loss"] / (sigma ** 2) + 0.5 * math.log(sigma ** 2)
    )
    torch.testing.assert_close(nll["feat_loss"], expected_feat, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(nll["base_loss"], expected_base, rtol=1e-6, atol=1e-7)


# ----------------------------------------------------------------------
# 3. Gaussian-NLL: sigma_obs='learned'
# ----------------------------------------------------------------------

def test_nll_learned_sigma_runs_and_is_finite(model, batch, forward_out):
    """The 'learned' branch reuses ``logvar_full`` / ``logvar_base`` heads."""
    losses = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"],
        likelihood="gaussian_nll", sigma_obs="learned",
    )
    for key in ("feat_loss", "base_loss", "kld_loss", "total_loss"):
        v = losses[key]
        assert torch.isfinite(v).all(), f"{key} not finite: {v}"
    # mean_logvar_full / base are the collapse diagnostics; both must be
    # finite and inside the model's logvar_clamp band (-5, 3) at init.
    for key in ("mean_logvar_full", "mean_logvar_base"):
        v = losses[key]
        assert torch.isfinite(v).all()
        assert -5.0 <= float(v) <= 3.0, f"{key}={float(v)} outside clamp"


def test_nll_learned_sigma_propagates_gradients_to_logvar_head():
    """Under MSE, gradient on the ``logvar_full`` linear head is zero (the
    head is never read by the loss). Under NLL with ``sigma_obs='learned'``
    the head should pick up a non-trivial gradient."""
    torch.manual_seed(2)
    m = SeqVaeLagAttnV1(
        sequence_length=_T, d_model=32, d_z=8, horizon=_HORIZON,
        warmup_period=_WARMUP, c_y=87, c_u=101, use_up_st=True,
        max_lag=24, num_heads=4, d_head=8,
    )
    y_st = torch.randn(_B, _T, 43)
    y_ph = torch.randn(_B, _T, 44)
    u_stream = torch.randn(_B, _T, 101)

    head = m.residual_decoder.logvar_head

    def _grad_norm_of(likelihood: str, sigma_obs):
        m.zero_grad(set_to_none=True)
        out = m(y_st, y_ph, u_stream)
        losses = m.compute_loss(
            out, y_st, y_ph,
            likelihood=likelihood, sigma_obs=sigma_obs,
        )
        losses["total_loss"].backward()
        if head.weight.grad is None:
            return 0.0
        return float(head.weight.grad.detach().norm())

    g_mse = _grad_norm_of("mse", 1.0)
    g_nll = _grad_norm_of("gaussian_nll", "learned")

    assert g_mse == 0.0, (
        f"MSE path leaked gradient into logvar_head (norm={g_mse}); "
        "expected the head to be untouched."
    )
    assert g_nll > 1e-6, (
        f"NLL(learned) failed to backprop through logvar_head (norm={g_nll})."
    )


# ----------------------------------------------------------------------
# 4. free_bits floor
# ----------------------------------------------------------------------

def test_free_bits_zero_is_noop(model, batch, forward_out):
    """``free_bits=0.0`` must reproduce the legacy KL exactly because the
    closed-form Gaussian KL is non-negative."""
    a = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], free_bits=0.0,
    )
    b = model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"],
    )
    torch.testing.assert_close(a["kld_loss"], b["kld_loss"], rtol=0, atol=0)


def test_free_bits_positive_raises_kld_monotonically(model, batch, forward_out):
    """Free-bits floors per-dim KL, so increasing the floor cannot decrease
    the aggregate KL loss."""
    k0 = float(model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], free_bits=0.0,
    )["kld_loss"])
    k1 = float(model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], free_bits=0.5,
    )["kld_loss"])
    k2 = float(model.compute_loss(
        forward_out, batch["y_st"], batch["y_ph"], free_bits=2.0,
    )["kld_loss"])
    assert k0 <= k1 + 1e-7 <= k2 + 1e-7, (
        f"free_bits not monotone: k0={k0}, k1={k1}, k2={k2}"
    )


# ----------------------------------------------------------------------
# 5. Argument validation
# ----------------------------------------------------------------------

def test_unknown_likelihood_raises(model, batch, forward_out):
    with pytest.raises(ValueError, match="likelihood"):
        model.compute_loss(
            forward_out, batch["y_st"], batch["y_ph"], likelihood="huber",
        )


def test_unknown_sigma_string_raises(model, batch, forward_out):
    with pytest.raises(ValueError, match="learned"):
        model.compute_loss(
            forward_out, batch["y_st"], batch["y_ph"],
            likelihood="gaussian_nll", sigma_obs="oracle",
        )


def test_nonpositive_sigma_raises(model, batch, forward_out):
    with pytest.raises(ValueError, match="positive"):
        model.compute_loss(
            forward_out, batch["y_st"], batch["y_ph"],
            likelihood="gaussian_nll", sigma_obs=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        model.compute_loss(
            forward_out, batch["y_st"], batch["y_ph"],
            likelihood="gaussian_nll", sigma_obs=-1.0,
        )


def test_return_dict_invariants(model, batch, forward_out):
    """Return dict must always carry the diagnostic keys, regardless of
    likelihood (so MSE checkpoints still log logvar_full collapse)."""
    expected = {
        "feat_loss", "base_loss", "kld_loss", "total_loss", "beta",
        "likelihood", "mean_logvar_full", "mean_logvar_base",
    }
    for like, sig in (("mse", 1.0), ("gaussian_nll", 1.0),
                      ("gaussian_nll", "learned")):
        losses = model.compute_loss(
            forward_out, batch["y_st"], batch["y_ph"],
            likelihood=like, sigma_obs=sig,
        )
        assert set(losses.keys()) == expected, (
            f"likelihood={like}: missing/extra keys "
            f"{expected.symmetric_difference(losses.keys())}"
        )
        assert losses["likelihood"] == like
