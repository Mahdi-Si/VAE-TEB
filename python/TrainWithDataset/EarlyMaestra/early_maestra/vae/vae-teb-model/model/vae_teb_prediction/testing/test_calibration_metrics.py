r"""S5-T01: unit tests for the calibration kernels.

Calibration numbers are silent when wrong -- a mis-signed CRPS or an off-by-one quantile still
produces a plausible-looking curve. So every kernel here is pinned against a value known in
closed form, or against a distribution whose calibration is known by construction.

Shapes follow the model's forecast contract: ``mu_full``/``logvar_full`` are
``(B, T, H_d, C)`` over all ``T`` anchors, while ``y_plus`` is ``(B, T - H_d, H_d, C)``; the
kernels slice both to ``[warmup, T - H_d)``.
"""
from __future__ import annotations

import math

import pytest
import torch

from model.vae_teb_prediction.testing.metrics import (
    compute_crps,
    compute_forecast_metrics,
    compute_interval_coverage,
    compute_nll,
    compute_reliability_by_horizon,
    crps_gaussian,
    crps_sample,
    fit_constant_sigma,
)

_HALF_LOG_2PI = 0.9189385332046727


def _calibrated_gaussian(B=6, T=40, H=3, C=20, sigma=1.0, seed=0):
    r"""A forecast that is calibrated by construction: :math:`y \sim \mathcal{N}(\mu, \sigma^2)`."""
    torch.manual_seed(seed)
    mu_full = torch.zeros(B, T, H, C)
    logvar_full = torch.full((B, T, H, C), 2.0 * math.log(sigma))
    y_plus = mu_full[:, : T - H] + sigma * torch.randn(B, T - H, H, C)
    return mu_full, logvar_full, y_plus


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------
def test_crps_gaussian_matches_the_pinned_closed_form():
    r""":math:`\mathrm{CRPS}(\mathcal{N}(0,1), 0) = (\sqrt2 - 1)/\sqrt\pi \approx 0.2337`."""
    got = crps_gaussian(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0))
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert float(got) == pytest.approx(expected, abs=1e-6)
    assert float(got) == pytest.approx(0.2336950, abs=1e-6)  # the literal, spelled out


@pytest.mark.parametrize("sigma", [0.25, 1.0, 4.0])
def test_crps_gaussian_scales_linearly_with_sigma(sigma):
    """At ``y = mu`` the score is exactly ``0.2337 * sigma``."""
    got = crps_gaussian(torch.tensor(3.0), torch.tensor(sigma), torch.tensor(3.0))
    expected = sigma * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert float(got) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("mu,sigma,y", [(1.3, 0.7, 2.1), (-0.5, 2.0, -0.5), (0.0, 1.0, 3.0)])
def test_crps_sample_agrees_with_the_closed_form(mu, sigma, y):
    torch.manual_seed(0)
    mu_t, sigma_t, y_t = torch.tensor(mu), torch.tensor(sigma), torch.tensor(y)
    draws = mu_t + sigma_t * torch.randn(200_000)
    assert float(crps_sample(draws, y_t)) == pytest.approx(
        float(crps_gaussian(mu_t, sigma_t, y_t)), abs=0.01
    )


def test_crps_sample_is_exact_for_two_draws():
    r""":math:`\tfrac12(|0-1| + |4-1|) - \tfrac12|4-0| = 2 - 2 = 0`."""
    got = crps_sample(torch.tensor([0.0, 4.0]), torch.tensor(1.0))
    assert float(got) == pytest.approx(0.0, abs=1e-6)


def test_crps_sample_rejects_a_single_draw():
    with pytest.raises(ValueError, match="at least 2 draws"):
        crps_sample(torch.zeros(1), torch.tensor(0.0))


def test_crps_rewards_sharpness_when_the_mean_is_right():
    """Among correct means, the sharper forecast wins."""
    mu, _, y = _calibrated_gaussian(sigma=1.0)
    sharp = torch.full_like(mu, 2.0 * math.log(0.5))
    wide = torch.full_like(mu, 2.0 * math.log(4.0))
    a = compute_crps(mu, sharp, y, 4, 3)["crps_total"].mean()
    b = compute_crps(mu, wide, y, 4, 3)["crps_total"].mean()
    assert float(a) < float(b)


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------
def test_nll_without_the_constant_matches_the_training_convention():
    r"""``SeqVaeLagAttnV1.compute_loss`` omits :math:`\tfrac12\log 2\pi`."""
    mu, logvar, y = _calibrated_gaussian(seed=1)
    with_const = compute_nll(mu, logvar, y, 4, 3, include_const=True)["nll_total"]
    without = compute_nll(mu, logvar, y, 4, 3, include_const=False)["nll_total"]
    assert torch.allclose(with_const - without, torch.full_like(without, _HALF_LOG_2PI))


def test_nll_is_minimised_by_the_true_sigma():
    """A Gaussian NLL is a proper scoring rule: mis-stating sigma costs you."""
    mu, logvar, y = _calibrated_gaussian(sigma=1.0, seed=2)
    truth = compute_nll(mu, logvar, y, 4, 3)["nll_total"].mean()
    too_sharp = compute_nll(mu, logvar - 2 * math.log(3), y, 4, 3)["nll_total"].mean()
    too_wide = compute_nll(mu, logvar + 2 * math.log(3), y, 4, 3)["nll_total"].mean()
    assert float(truth) < float(too_sharp)
    assert float(truth) < float(too_wide)


def test_nll_at_the_mean_with_unit_sigma_is_the_gaussian_constant():
    mu = torch.zeros(1, 5, 2, 3)
    logvar = torch.zeros(1, 5, 2, 3)
    y = torch.zeros(1, 3, 2, 3)
    got = compute_nll(mu, logvar, y, 0, 2)["nll_total"]
    assert float(got) == pytest.approx(_HALF_LOG_2PI, abs=1e-6)


def test_nll_channel_blocks_average_to_the_total():
    """43 scattering channels + 44 phase channels; the blocks must reconstruct the mean."""
    mu, logvar, y = _calibrated_gaussian(C=87, seed=3)
    out = compute_nll(mu, logvar, y, 4, 3)
    blended = (43.0 * out["nll_st"] + 44.0 * out["nll_ph"]) / 87.0
    assert torch.allclose(blended, out["nll_total"], atol=1e-5)


def test_nll_per_horizon_has_one_entry_per_lead_time():
    mu, logvar, y = _calibrated_gaussian(H=3)
    out = compute_nll(mu, logvar, y, 4, 3)
    assert out["nll_per_horizon"].shape == (6, 3)
    assert torch.allclose(out["nll_per_horizon"].mean(dim=1), out["nll_total"], atol=1e-5)


# ---------------------------------------------------------------------------
# Interval coverage
# ---------------------------------------------------------------------------
def test_coverage_matches_nominal_on_calibrated_gaussian_draws():
    mu, logvar, y = _calibrated_gaussian(B=8, T=60, C=30, seed=4)
    out = compute_interval_coverage(mu, logvar, y, 4, 3)

    assert torch.allclose(out["nominal"], torch.tensor([0.5, 0.8, 0.9, 0.95]))
    empirical = out["coverage"].mean(dim=0)
    assert torch.allclose(empirical, out["nominal"], atol=0.01), empirical


def test_coverage_reports_the_sharpness_in_target_units():
    mu, logvar, y = _calibrated_gaussian(sigma=2.5, seed=5)
    out = compute_interval_coverage(mu, logvar, y, 4, 3)
    assert float(out["sharpness"].mean()) == pytest.approx(2.5, rel=1e-5)
    assert out["sharpness_per_horizon"].shape == (6, 3)


def test_an_overconfident_forecast_under_covers():
    """Halving sigma must be caught -- this is the failure MSE cannot see."""
    mu, logvar, y = _calibrated_gaussian(B=8, T=60, C=30, seed=6)
    honest = compute_interval_coverage(mu, logvar, y, 4, 3)["coverage"].mean(dim=0)
    cocky = compute_interval_coverage(
        mu, logvar - 2 * math.log(2), y, 4, 3
    )["coverage"].mean(dim=0)
    assert bool((cocky < honest - 0.1).all()), (honest, cocky)
    assert float(cocky[-1]) < 0.75  # nominal 0.95


def test_coverage_rejects_degenerate_levels():
    mu, logvar, y = _calibrated_gaussian()
    for bad in ((0.0, 0.9), (0.5, 1.0), (-0.1,)):
        with pytest.raises(ValueError, match=r"strictly inside \(0, 1\)"):
            compute_interval_coverage(mu, logvar, y, 4, 3, levels=bad)


# ---------------------------------------------------------------------------
# Reliability / PIT
# ---------------------------------------------------------------------------
def test_reliability_is_near_diagonal_for_a_calibrated_forecast():
    mu, logvar, y = _calibrated_gaussian(B=8, T=60, C=30, seed=7)
    out = compute_reliability_by_horizon(mu, logvar, y, 4, 3, n_bins=10)

    assert out["nominal"].shape == (10,)
    assert out["empirical"].shape == (3, 10)
    assert float(out["ks_stat"].max()) < 0.05
    # PIT ~ Uniform(0, 1): mean 1/2, variance 1/12.
    assert float(out["pit_mean"].mean()) == pytest.approx(0.5, abs=0.02)
    assert float(out["pit_var"].mean()) == pytest.approx(1.0 / 12.0, abs=0.01)


def test_reliability_detects_a_misspecified_variance():
    mu, logvar, y = _calibrated_gaussian(B=8, T=60, C=30, seed=8)
    good = compute_reliability_by_horizon(mu, logvar, y, 4, 3, n_bins=10)
    bad = compute_reliability_by_horizon(mu, logvar - 2 * math.log(2), y, 4, 3, n_bins=10)
    assert float(bad["ks_stat"].max()) > 5.0 * float(good["ks_stat"].max())
    # An over-confident forecast pushes PIT mass into both tails.
    assert float(bad["pit_var"].mean()) > float(good["pit_var"].mean())


def test_reliability_detects_a_biased_mean():
    mu, logvar, y = _calibrated_gaussian(B=8, T=60, C=30, seed=9)
    shifted = compute_reliability_by_horizon(mu + 1.0, logvar, y, 4, 3, n_bins=10)
    assert float(shifted["pit_mean"].mean()) < 0.35  # y sits below mu, so PIT skews low


def test_reliability_empirical_cdf_is_monotone():
    mu, logvar, y = _calibrated_gaussian(seed=10)
    out = compute_reliability_by_horizon(mu, logvar, y, 4, 3, n_bins=12)
    diffs = out["empirical"][:, 1:] - out["empirical"][:, :-1]
    assert bool((diffs >= -1e-6).all()), "an empirical CDF cannot decrease"


# ---------------------------------------------------------------------------
# Constant-sigma reference
# ---------------------------------------------------------------------------
def test_fit_constant_sigma_recovers_the_residual_scale():
    mu, _, y = _calibrated_gaussian(B=8, T=60, C=30, sigma=1.7, seed=11)
    assert float(fit_constant_sigma(mu, y, 4, 3)) == pytest.approx(1.7, rel=0.02)


def test_fit_constant_sigma_agrees_with_the_forecast_mse():
    mu, _, y = _calibrated_gaussian(seed=12)
    sigma = float(fit_constant_sigma(mu, y, 4, 3))
    mse = float(compute_forecast_metrics(mu, y, 4, 3)["feat_mse_total"].mean())
    assert sigma ** 2 == pytest.approx(mse, rel=1e-4)


def test_a_learned_heteroscedastic_sigma_beats_the_constant_when_it_is_right():
    """The comparison the calibration report exists to make."""
    torch.manual_seed(13)
    B, T, H, C = 6, 40, 3, 20
    mu = torch.zeros(B, T, H, C)
    # True noise scale varies by horizon step; the constant fit cannot express that.
    true_sigma = torch.tensor([0.3, 1.0, 3.0]).view(1, 1, H, 1)
    y = mu[:, : T - H] + true_sigma * torch.randn(B, T - H, H, C)

    learned = compute_nll(mu, (2 * torch.log(true_sigma)).expand_as(mu), y, 4, H)
    const = fit_constant_sigma(mu, y, 4, H)
    const_nll = compute_nll(mu, torch.full_like(mu, 2 * math.log(float(const))), y, 4, H)
    assert float(learned["nll_total"].mean()) < float(const_nll["nll_total"].mean())


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_kernels_survive_an_empty_anchor_range():
    """``warmup >= T - H_d`` must yield zeros, not an exception."""
    mu = torch.zeros(2, 3, 4, 5)
    logvar = torch.zeros(2, 3, 4, 5)
    y = torch.zeros(2, 0, 4, 5)

    nll = compute_nll(mu, logvar, y, 10, 4)
    crps = compute_crps(mu, logvar, y, 10, 4)
    cov = compute_interval_coverage(mu, logvar, y, 10, 4)
    rel = compute_reliability_by_horizon(mu, logvar, y, 10, 4, n_bins=5)

    assert nll["nll_total"].shape == (2,) and float(nll["nll_total"].abs().sum()) == 0.0
    assert crps["crps_per_horizon"].shape == (2, 4)
    assert cov["coverage"].shape == (2, 4)
    assert torch.isnan(rel["ks_stat"]).all()
    assert float(fit_constant_sigma(mu, y, 10, 4)) == 0.0
