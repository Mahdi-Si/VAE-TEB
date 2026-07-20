r"""Tests for the predictive-calibration metric functions.

Calibration metrics are unusually easy to write in a self-consistently wrong way: a CRPS with a
dropped term, a coverage scored against $0.95$ instead of $0.9545$, or a PIT built from the
variance rather than the standard deviation all produce plausible numbers that move in the right
direction. So the checks here are against quantities known *independently* of the implementation
-- a closed form evaluated by hand at a point, the exact nominal from $\mathrm{erf}$, and the
statistical property the transform is defined by, on synthetic data drawn from the very
distribution the model claims.
"""
from __future__ import annotations

import math

import pytest
import torch
from scipy import stats

from teb_vae.lag_attn.eval import metrics


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------
def test_crps_matches_the_closed_form_at_a_known_point() -> None:
    r"""At $y = \mu$ and $\sigma = 1$ the closed form collapses to $2\varphi(0) - \pi^{-1/2}$.

    $$\mathrm{CRPS} = 1\cdot\left[0 + \tfrac{2}{\sqrt{2\pi}} - \tfrac{1}{\sqrt{\pi}}\right]
    \approx 0.2337$$
    """
    # float64, so the comparison tests the formula rather than fp32's seven digits.
    mu = torch.zeros(1, dtype=torch.float64)
    expected = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    value = metrics.crps_gaussian(mu, mu, torch.zeros(1, dtype=torch.float64))
    assert float(value) == pytest.approx(expected, rel=1e-12)
    assert float(value) == pytest.approx(0.23369, abs=1e-5)


def test_crps_scales_linearly_with_sigma_when_the_residual_does() -> None:
    r"""$\mathrm{CRPS}(\mu, y, \sigma)$ is homogeneous: doubling $\sigma$ and the residual
    doubles the score. A dropped $\sigma$ factor would break exactly this."""
    logvar = torch.zeros(1)
    single = metrics.crps_gaussian(torch.zeros(1), torch.ones(1), logvar)
    doubled = metrics.crps_gaussian(
        torch.zeros(1), torch.full((1,), 2.0), logvar + 2.0 * math.log(2.0)
    )
    assert float(doubled) == pytest.approx(2.0 * float(single), rel=1e-6)


def test_crps_is_non_negative_and_smallest_at_the_truth() -> None:
    """A proper score: no forecast beats predicting the observation itself."""
    logvar = torch.zeros(5)
    residuals = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    scores = metrics.crps_gaussian(torch.zeros(5), residuals, logvar)
    assert bool((scores >= 0.0).all())
    assert int(torch.argmin(scores)) == 2


def test_crps_agrees_with_a_numerical_integration_of_its_definition() -> None:
    r"""$\mathrm{CRPS} = \int \left(F(t) - \mathbb{1}[t \ge y]\right)^2 dt$, integrated directly.

    The definition, not the closed form -- which is what makes this an independent check rather
    than a restatement of the implementation.
    """
    mu, sigma, observation = 0.3, 1.7, -0.8

    def _segment(low: float, high: float, indicator: float) -> float:
        """Integrate one side of the observation, with the indicator supplied rather than derived.

        Split at ``observation`` because the indicator jumps there and a trapezoid rule straddling
        the jump carries an error far larger than its own step size. The indicator is passed in
        rather than evaluated on the grid because ``observation`` is an endpoint of *both*
        segments and ``grid >= observation`` is true at both copies of it -- which would apply the
        post-jump value on the pre-jump side and inject an error of half a step times the jump.
        """
        grid = torch.linspace(low, high, 200001, dtype=torch.float64)
        cdf = torch.tensor(stats.norm.cdf(grid.numpy(), loc=mu, scale=sigma))
        return float(torch.trapezoid((cdf - indicator) ** 2, grid))

    numerical = _segment(-40.0, observation, 0.0) + _segment(observation, 40.0, 1.0)

    closed = metrics.crps_gaussian(
        torch.tensor([mu], dtype=torch.float64),
        torch.tensor([observation], dtype=torch.float64),
        torch.tensor([2.0 * math.log(sigma)], dtype=torch.float64),
    )
    assert float(closed) == pytest.approx(numerical, rel=1e-5)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
def test_the_two_sigma_nominal_is_not_zero_point_ninety_five() -> None:
    r"""$0.95$ is $\pm 1.96\sigma$. Scoring a $2\sigma$ band against it reports a calibrated
    model as over-confident on every horizon."""
    assert metrics.nominal_central_coverage(2.0) == pytest.approx(0.9545, abs=5e-5)
    assert metrics.nominal_central_coverage(1.0) == pytest.approx(0.6827, abs=5e-5)
    assert metrics.nominal_central_coverage(3.0) == pytest.approx(0.9973, abs=5e-5)
    assert metrics.nominal_central_coverage(2.0) != 0.95


def test_the_two_sigma_nominal_matches_the_models_own_constant() -> None:
    """Pinned against ``plotting.py``, so a training figure and an eval table cannot disagree."""
    from teb_vae.lag_attn.plotting import _NOMINAL_2SIGMA

    assert metrics.nominal_central_coverage(2.0) == pytest.approx(_NOMINAL_2SIGMA, abs=5e-5)


def test_empirical_coverage_hits_the_nominal_on_calibrated_data() -> None:
    """Drawn from the very distribution the model claims, so the coverage is the nominal."""
    generator = torch.Generator().manual_seed(0)
    mu = torch.zeros(200000)
    logvar = torch.full((200000,), math.log(2.25))  # sigma = 1.5
    y = mu + 1.5 * torch.randn(200000, generator=generator)

    for k_sigma in (1.0, 2.0, 3.0):
        empirical = float(metrics.coverage_indicator(mu, y, logvar, k_sigma).mean())
        assert empirical == pytest.approx(metrics.nominal_central_coverage(k_sigma), abs=5e-3)


def test_coverage_uses_sigma_not_the_variance() -> None:
    r"""With $\sigma^2 = 4$, a residual of $3$ is inside $2\sigma = 4$ and outside $1\sigma$.

    Reading the variance as the scale is the single most common way to get this wrong, and it
    stays plausible: it merely reports the model as better calibrated than it is.
    """
    mu, y, logvar = torch.zeros(1), torch.full((1,), 3.0), torch.full((1,), math.log(4.0))
    assert float(metrics.coverage_indicator(mu, y, logvar, 2.0)) == 1.0
    assert float(metrics.coverage_indicator(mu, y, logvar, 1.0)) == 0.0


# ---------------------------------------------------------------------------
# PIT
# ---------------------------------------------------------------------------
def test_pit_is_uniform_on_well_calibrated_data_by_a_ks_test() -> None:
    """The property the transform is defined by, tested as such rather than by its formula."""
    generator = torch.Generator().manual_seed(1)
    mu = torch.randn(50000, generator=generator)
    logvar = torch.full((50000,), math.log(0.64))  # sigma = 0.8
    y = mu + 0.8 * torch.randn(50000, generator=generator)

    pit = metrics.pit_values(mu, y, logvar).numpy()
    assert float(stats.kstest(pit, "uniform").pvalue) > 0.01


def test_pit_departs_from_uniform_when_the_variance_is_wrong() -> None:
    """The negative control: a KS test that passes on everything proves nothing."""
    generator = torch.Generator().manual_seed(2)
    mu = torch.zeros(50000)
    y = 3.0 * torch.randn(50000, generator=generator)
    # Claimed sigma = 1 against a true sigma of 3: heavily over-confident.
    pit = metrics.pit_values(mu, y, torch.zeros(50000)).numpy()
    assert float(stats.kstest(pit, "uniform").pvalue) < 1e-10


def test_pit_is_bounded_and_centred() -> None:
    mu, logvar = torch.zeros(3), torch.zeros(3)
    pit = metrics.pit_values(mu, torch.tensor([-10.0, 0.0, 10.0]), logvar)
    assert float(pit[1]) == pytest.approx(0.5)
    assert bool(((pit >= 0.0) & (pit <= 1.0)).all())


# ---------------------------------------------------------------------------
# NLL and the homoscedastic reference
# ---------------------------------------------------------------------------
def test_the_log_density_carries_the_constant_the_training_loss_drops() -> None:
    r"""$-\log p$ at $y = \mu$, $\sigma = 1$ is exactly $\tfrac{1}{2}\log 2\pi$.

    The training-shaped ``per_element_loss`` drops that term, which makes it comparable with a
    training curve but not a density -- and an NLL *gain* over a reference is only meaningful
    between genuine densities.
    """
    zeros = torch.zeros(1)
    assert float(metrics.gaussian_log_density(zeros, zeros, zeros)) == pytest.approx(
        0.5 * math.log(2.0 * math.pi)
    )
    assert float(
        metrics.per_element_loss(zeros, zeros, likelihood="gaussian_nll", sigma_obs="learned")
    ) == pytest.approx(0.0)


def test_the_log_density_agrees_with_scipy() -> None:
    mu, sigma, y = 0.4, 1.3, -0.2
    ours = metrics.gaussian_log_density(
        torch.tensor([mu]), torch.tensor([y]), torch.tensor([2.0 * math.log(sigma)])
    )
    assert float(ours) == pytest.approx(-float(stats.norm.logpdf(y, mu, sigma)), rel=1e-6)


def test_the_homoscedastic_reference_is_the_maximum_likelihood_constant_variance() -> None:
    """The strongest constant-variance reference available, not a straw man."""
    generator = torch.Generator().manual_seed(3)
    mu = torch.zeros(4, 5, 3, 2)
    y = 2.0 * torch.randn(4, 5, 3, 2, generator=generator)
    mask = torch.ones(4, 5, 3, 1)

    fitted = float(torch.exp(metrics.homoscedastic_logvar(mu, y, mask)))
    assert fitted == pytest.approx(float((y**2).mean()), rel=1e-6)

    # No other constant variance achieves a lower NLL -- that is what "maximum likelihood" means.
    best = float(
        metrics.masked_pooled_mean(
            metrics.gaussian_log_density(
                mu, y, torch.full_like(y, math.log(fitted))
            ),
            mask,
        )
    )
    for factor in (0.5, 0.8, 1.25, 2.0):
        other = float(
            metrics.masked_pooled_mean(
                metrics.gaussian_log_density(
                    mu, y, torch.full_like(y, math.log(fitted * factor))
                ),
                mask,
            )
        )
        assert other > best


def test_the_homoscedastic_reference_honours_the_mask() -> None:
    """A masked-out region of wild residuals must not inflate the fitted variance."""
    mu = torch.zeros(1, 4, 2, 1)
    y = torch.ones(1, 4, 2, 1)
    y[:, 2:] = 100.0
    mask = torch.ones(1, 4, 2, 1)
    mask[:, 2:] = 0.0
    assert float(torch.exp(metrics.homoscedastic_logvar(mu, y, mask))) == pytest.approx(1.0)
