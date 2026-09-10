r"""The explicit sum, the prior-relative residual, and the divergence it collapses to.

The central quantity of the architecture, and the one whose simplified form is correct **only**
under this exact parameterisation. Four things are checked here and each of them fails as a wrong
number rather than as an exception:

* the residual divergence must equal the family's own diagonal-Gaussian formula evaluated on the
  final parameters, not on a re-derivation of them, which is why the comparison imports that
  function rather than restating it;
* a zero update must give an exactly zero divergence and parameters bitwise equal to the prior's, so
  every source control downstream is read against a real invariant;
* the bounds must hold for arbitrary raw proposals, because their whole purpose is to hold when the
  head is far from its initialisation;
* a **zero-sum reallocation** across lags must leave the total update and the divergence untouched
  while changing what suppressing a single lag reports. That is not a curiosity: it is why no
  nonnegative per-lag allocation of the divergence exists, and a test is the only thing that stops
  one being reintroduced later as an apparently obvious readout.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.nets.losses import kld_tensor
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    CANCELLATION_EPS,
    bound_update,
    cancellation_ratio,
    default_lag_scale,
    residual_kl,
    residual_parameters,
    sum_proposals,
)
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_BATCH,
    TINY_D_Z,
    TINY_N_LAGS,
)

#: FP64 tolerances for the algebraic identities. Numerical checks, not predictive margins.
ALGEBRA_ATOL = 1.0e-10
ALGEBRA_RTOL = 1.0e-8

#: Cases per randomised identity check. Enough to cover the prior log-variance range and the
#: interior of both bounds; more would re-measure the same identity.
N_CASES = 2000

#: The starting bounds the architecture is specified at.
A_MAX = 3.0
B_MAX = 1.0

#: The prior log-variance range the family's heads emit.
PRIOR_LOGVAR_RANGE = (-5.0, 3.0)


def random_prior(seed: int = 20260909, cases: int = N_CASES):
    """A randomised prior over the range the family's own head can emit, in FP64.

    Args:
        seed: Seed for the draw.
        cases: Rows to draw.

    Returns:
        ``(mu_prior, logvar_prior)``, each $(\\text{cases}, d_z)$ in ``float64``.
    """
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(cases, TINY_D_Z, generator=generator, dtype=torch.float64) * 2.0
    low, high = PRIOR_LOGVAR_RANGE
    logvar = (
        torch.rand(cases, TINY_D_Z, generator=generator, dtype=torch.float64)
        * (high - low)
        + low
    )
    return mu, logvar


def random_updates(seed: int = 20260910, cases: int = N_CASES):
    """Randomised raw proposals passed through the stated limiters, in FP64.

    Drawn wide enough that a good share of coordinates sit near saturation, which is where the
    bounded and unbounded forms disagree most.

    Args:
        seed: Seed for the draw.
        cases: Rows to draw.

    Returns:
        ``(a, b)``, each $(\\text{cases}, d_z)$ in ``float64``.
    """
    generator = torch.Generator().manual_seed(seed)
    raw_a = torch.randn(cases, TINY_D_Z, generator=generator, dtype=torch.float64) * 4.0
    raw_b = torch.randn(cases, TINY_D_Z, generator=generator, dtype=torch.float64) * 2.0
    return bound_update(raw_a, A_MAX), bound_update(raw_b, B_MAX)


# =================================================================================================
# The divergence
# =================================================================================================
def test_the_residual_divergence_equals_the_family_formula() -> None:
    """Against the shared closed form, on the final parameters, in FP64.

    The comparison runs against the function the rest of the family reports this quantity from
    rather than against a formula written again here: two copies of one identity are two identities,
    and the point of this check is that the simplified form is the same number.
    """
    mu_prior, logvar_prior = random_prior()
    a, b = random_updates()

    mu_full, logvar_full = residual_parameters(mu_prior, logvar_prior, a, b)
    general = kld_tensor(mu_prior, logvar_prior, mu_full, logvar_full)
    residual = residual_kl(a, b)

    assert residual.dtype == torch.float64
    assert torch.allclose(residual, general, atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL)


def test_the_residual_divergence_is_nonnegative() -> None:
    r"""Because $e^x \ge 1 + x$, with equality only at zero."""
    a, b = random_updates()
    assert bool((residual_kl(a, b) >= 0.0).all())


def test_a_zero_update_gives_an_exactly_zero_divergence_and_the_prior_back() -> None:
    """Bitwise, not approximately: every source control downstream is read against this."""
    mu_prior, logvar_prior = random_prior(cases=64)
    zero = torch.zeros_like(mu_prior)

    mu_full, logvar_full = residual_parameters(mu_prior, logvar_prior, zero, zero)
    assert torch.equal(mu_full, mu_prior)
    assert torch.equal(logvar_full, logvar_prior)
    assert torch.all(residual_kl(zero, zero) == 0.0)


def test_the_mean_only_arm_leaves_the_scale_untouched() -> None:
    r"""$\sigma^q = \sigma^p$ exactly, and the divergence is $\tfrac12\lVert a\rVert^2$."""
    mu_prior, logvar_prior = random_prior(cases=64)
    a, _ = random_updates(cases=64)

    mu_full, logvar_full = residual_parameters(mu_prior, logvar_prior, a, None)
    assert torch.equal(logvar_full, logvar_prior)
    # A copy rather than the same object: a caller writing into one must not corrupt the other.
    assert logvar_full is not logvar_prior

    assert torch.allclose(
        residual_kl(a, None), 0.5 * a**2, atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL
    )
    assert torch.allclose(
        residual_kl(a, None),
        kld_tensor(mu_prior, logvar_prior, mu_full, logvar_full),
        atol=ALGEBRA_ATOL,
        rtol=ALGEBRA_RTOL,
    )


def test_the_variance_term_survives_a_very_small_scale_update() -> None:
    r"""``expm1`` is why. Its series near zero is $2b^2 + \tfrac43 b^3 + \cdots$.

    Written as ``exp(2b) - 1`` the leading digits cancel, and the divergence of a nearly-converged
    scale head reads as noise or as a small negative number rather than as a small positive one.
    """
    b = torch.tensor([[1e-9, -1e-9, 1e-12, 0.0]], dtype=torch.float64)
    a = torch.zeros_like(b)

    value = residual_kl(a, b)
    assert bool((value >= 0.0).all())
    expected = 0.5 * (torch.expm1(2.0 * b) - 2.0 * b)
    assert torch.allclose(value, expected, atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL)
    # The naive form is what this avoids. At b = 1e-12 the true value is 2e-24 and the
    # cancellation leaves a *negative* number some seven orders of magnitude larger in
    # magnitude -- a divergence that cannot be negative, reported as negative.
    naive = 0.5 * (torch.exp(2.0 * b) - 1.0 - 2.0 * b)
    assert float(naive[0, 2]) < 0.0
    assert float(value[0, 2]) > 0.0
    assert float(value[0, 2]) == pytest.approx(2.0 * (1e-12**2), rel=1e-6)


def test_the_divergence_accumulates_in_at_least_single_precision() -> None:
    """A half-precision update must not reduce in half precision."""
    a = torch.full((4, TINY_D_Z), 0.01, dtype=torch.float16)
    assert residual_kl(a).dtype == torch.float32


# =================================================================================================
# The bounds
# =================================================================================================
def test_the_mean_correction_is_bounded_in_prior_standard_deviations() -> None:
    r"""$\lvert \mu^q_d - \mu^p_d \rvert \le a_{\max}\sigma^p_d$, for arbitrary raw proposals."""
    mu_prior, logvar_prior = random_prior()
    generator = torch.Generator().manual_seed(5)
    raw = torch.randn(N_CASES, TINY_D_Z, generator=generator, dtype=torch.float64) * 50.0

    a = bound_update(raw, A_MAX)
    mu_full, _ = residual_parameters(mu_prior, logvar_prior, a)
    sigma_prior = torch.exp(0.5 * logvar_prior)

    assert bool(((mu_full - mu_prior).abs() <= A_MAX * sigma_prior + 1e-12).all())


def test_the_scale_ratio_is_bounded_both_ways() -> None:
    r"""$e^{-b_{\max}} \le \sigma^q_d / \sigma^p_d \le e^{b_{\max}}$, and both signs are permitted.

    Extra evidence can raise or lower conditional uncertainty for a particular observation, so a
    one-sided bound would be an assumption about the physiology rather than a numerical guard.
    """
    mu_prior, logvar_prior = random_prior()
    generator = torch.Generator().manual_seed(6)
    raw = torch.randn(N_CASES, TINY_D_Z, generator=generator, dtype=torch.float64) * 50.0

    b = bound_update(raw, B_MAX)
    _, logvar_full = residual_parameters(mu_prior, logvar_prior, torch.zeros_like(b), b)
    ratio = torch.exp(0.5 * (logvar_full - logvar_prior))

    assert bool((ratio <= math.exp(B_MAX) + 1e-12).all())
    assert bool((ratio >= math.exp(-B_MAX) - 1e-12).all())
    assert bool((ratio > 1.0).any()) and bool((ratio < 1.0).any())


def test_the_implied_full_log_variance_range_widens_by_the_scale_bound() -> None:
    r"""The prior's $[-5, 3]$ becomes $[-7, 5]$, which is a different range from the sibling's.

    Stated as a measurement because it is the reason the full log-variance must **not** be passed
    through the prior head's own smooth bound a second time: that map is a sigmoid onto $[-5, 3]$,
    it is not idempotent, and re-applying it would silently re-parameterise the model and destroy
    the exact equality at a zero update.
    """
    low, high = PRIOR_LOGVAR_RANGE
    logvar_prior = torch.tensor([[low, high]], dtype=torch.float64)
    for sign, expected in ((-1.0, low - 2.0 * B_MAX), (1.0, high + 2.0 * B_MAX)):
        b = torch.full_like(logvar_prior, sign * B_MAX)
        _, logvar_full = residual_parameters(
            torch.zeros_like(logvar_prior), logvar_prior, torch.zeros_like(b), b
        )
        index = 0 if sign < 0 else 1
        assert float(logvar_full[0, index]) == pytest.approx(expected)


def test_the_bound_is_smooth_and_refuses_a_degenerate_limit() -> None:
    """A nonzero gradient everywhere, so a saturated coordinate can still recover."""
    raw = torch.tensor([-40.0, 0.0, 40.0], dtype=torch.float64, requires_grad=True)
    bound_update(raw, A_MAX).sum().backward()
    assert raw.grad is not None
    assert bool((raw.grad > 0.0).all())

    with pytest.raises(ValueError, match="must be > 0"):
        bound_update(torch.zeros(2), 0.0)


# =================================================================================================
# The sum
# =================================================================================================
def test_the_summation_scale_is_the_configured_lag_count() -> None:
    r"""$c_L = L^{-1/2}$, resolved once and passed in, never derived from the data."""
    assert default_lag_scale(4) == pytest.approx(0.5)
    assert default_lag_scale(TINY_N_LAGS) == pytest.approx(TINY_N_LAGS**-0.5)
    with pytest.raises(ValueError, match="n_lags must be >= 1"):
        default_lag_scale(0)


def test_masking_lags_does_not_change_the_summation_scale() -> None:
    """The failure this prevents: the same evidence weighing differently at two anchors.

    A version that renormalised by the number of currently available lags would make an anchor near
    the start of a record, where most lags are out of range, scale its surviving proposals up -- and
    the resulting drift in the coupling readout would look like a physiological gradient.
    """
    generator = torch.Generator().manual_seed(21)
    proposals = torch.randn(
        TINY_BATCH, 2, TINY_N_LAGS, TINY_D_Z, generator=generator, dtype=torch.float64
    )
    scale = default_lag_scale(TINY_N_LAGS)

    masked = proposals.clone()
    masked[:, :, 3:] = 0.0
    total = sum_proposals(masked, c_lag=scale)

    # The two surviving lags, scaled by the CONFIGURED count rather than by their own.
    expected = scale * proposals[:, :, :3].sum(dim=2)
    assert torch.allclose(total, expected, atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL)
    assert not torch.allclose(total, default_lag_scale(3) * proposals[:, :, :3].sum(dim=2))


def test_the_sum_refuses_a_mis_shaped_proposal_array() -> None:
    """The lag axis is positional, so a missing one would reduce the wrong axis."""
    with pytest.raises(ValueError, match="must be 4-D"):
        sum_proposals(torch.zeros(2, 3, 4), c_lag=0.5)


# =================================================================================================
# Reallocation, and why no per-lag allocation exists
# =================================================================================================
def test_a_zero_sum_reallocation_leaves_the_posterior_and_divergence_untouched() -> None:
    r"""Adding $k_\ell$ with $\sum_\ell k_\ell = 0$ changes every proposal and nothing observable.

    The counterexample that forecloses a per-lag decomposition. Two parameterisations agreeing on
    every prediction and every divergence disagree on what suppressing a single lag reports, so a
    suppression margin measures a property of the fitted parameterisation and not a unique
    functional contribution -- and a proposal norm is not a per-lag transfer of information.
    """
    generator = torch.Generator().manual_seed(31)
    proposals = torch.randn(
        TINY_BATCH, 2, TINY_N_LAGS, TINY_D_Z, generator=generator, dtype=torch.float64
    )
    scale = default_lag_scale(TINY_N_LAGS)

    # A zero-sum reallocation: move mass from lag 0 to lag 1 and change nothing else.
    shift = torch.randn(
        TINY_BATCH, 2, TINY_D_Z, generator=generator, dtype=torch.float64
    )
    realloc = proposals.clone()
    realloc[:, :, 0] += shift
    realloc[:, :, 1] -= shift

    a_before = bound_update(sum_proposals(proposals, c_lag=scale), A_MAX)
    a_after = bound_update(sum_proposals(realloc, c_lag=scale), A_MAX)
    assert torch.allclose(a_before, a_after, atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL)
    assert torch.allclose(
        residual_kl(a_before), residual_kl(a_after), atol=ALGEBRA_ATOL, rtol=ALGEBRA_RTOL
    )

    # And yet suppressing one lag reports two different things.
    def suppress(array: torch.Tensor, lag: int) -> torch.Tensor:
        """The bounded mean update with one lag's proposal set to zero."""
        kept = array.clone()
        kept[:, :, lag] = 0.0
        return bound_update(sum_proposals(kept, c_lag=scale), A_MAX)

    assert not torch.allclose(suppress(proposals, 0), suppress(realloc, 0))


def test_exact_cancellation_is_indistinguishable_from_silence_in_the_total() -> None:
    """Two opposed proposals sum to zero, so the divergence is zero while both are large.

    The reason the cancellation ratio is reported at all, and the reason a divergence penalty on the
    summed update cannot discipline the proposals that produced it.
    """
    opposed = torch.zeros(1, 1, 2, TINY_D_Z, dtype=torch.float64)
    opposed[0, 0, 0] = 1.0
    opposed[0, 0, 1] = -1.0

    a = bound_update(sum_proposals(opposed, c_lag=default_lag_scale(2)), A_MAX)
    assert torch.all(a == 0.0)
    assert torch.all(residual_kl(a) == 0.0)

    ratio, numerator, denominator = cancellation_ratio(opposed)
    assert float(numerator) == pytest.approx(0.0)
    assert float(denominator) > 0.0
    assert float(ratio) == pytest.approx(0.0)


def test_the_cancellation_ratio_reports_its_two_components() -> None:
    """A near-zero ratio from cancellation and from silence must be tellable apart.

    Both give a ratio of about zero. Only the denominator says which happened, which is why it comes
    back beside the ratio rather than being recoverable from it.
    """
    silent = torch.zeros(1, 1, 2, TINY_D_Z, dtype=torch.float64)
    ratio, numerator, denominator = cancellation_ratio(silent)
    assert float(ratio) == pytest.approx(0.0)
    assert float(numerator) == pytest.approx(0.0)
    assert float(denominator) == pytest.approx(0.0)

    aligned = torch.zeros(1, 1, 2, TINY_D_Z, dtype=torch.float64)
    aligned[0, 0, 0, 0] = 1.0
    aligned[0, 0, 1, 0] = 1.0
    ratio, numerator, denominator = cancellation_ratio(aligned)
    assert float(ratio) == pytest.approx(1.0, abs=CANCELLATION_EPS * 10)
    assert float(numerator) == pytest.approx(2.0)
    assert float(denominator) == pytest.approx(2.0)


def test_the_cancellation_ratio_ignores_the_summation_scale() -> None:
    """It is common to numerator and denominator, so the ratio measures shape, not magnitude."""
    generator = torch.Generator().manual_seed(41)
    proposals = torch.randn(2, 3, TINY_N_LAGS, TINY_D_Z, generator=generator)
    plain, _, _ = cancellation_ratio(proposals)
    scaled, _, _ = cancellation_ratio(proposals * default_lag_scale(TINY_N_LAGS))
    assert torch.allclose(plain, scaled, atol=1e-6)


def test_the_cancellation_ratio_refuses_a_mis_shaped_array() -> None:
    """Same reason the sum does: the lag axis is positional."""
    with pytest.raises(ValueError, match="must be 4-D"):
        cancellation_ratio(torch.zeros(2, 3, 4))


# =================================================================================================
# End to end, on the shapes the model will use
# =================================================================================================
def test_the_three_stages_compose_into_a_bounded_prior_relative_update() -> None:
    """Proposals in, full parameters and a divergence out, at the tensor contract's own shapes."""
    generator = torch.Generator().manual_seed(51)
    n_anchors = 4
    mean_proposals = torch.randn(
        TINY_BATCH, n_anchors, TINY_N_LAGS, TINY_D_Z, generator=generator
    )
    scale_proposals = torch.randn(
        TINY_BATCH, n_anchors, TINY_N_LAGS, TINY_D_Z, generator=generator
    )
    mu_prior = torch.randn(TINY_BATCH, n_anchors, TINY_D_Z, generator=generator)
    logvar_prior = torch.rand(
        TINY_BATCH, n_anchors, TINY_D_Z, generator=generator
    ) * 8.0 - 5.0

    scale = default_lag_scale(TINY_N_LAGS)
    a = bound_update(sum_proposals(mean_proposals, c_lag=scale), A_MAX)
    b = bound_update(sum_proposals(scale_proposals, c_lag=scale), B_MAX)
    mu_full, logvar_full = residual_parameters(mu_prior, logvar_prior, a, b)
    per_coordinate = residual_kl(a, b)

    assert mu_full.shape == (TINY_BATCH, n_anchors, TINY_D_Z)
    assert logvar_full.shape == mu_full.shape
    assert per_coordinate.shape == mu_full.shape
    assert per_coordinate.sum(dim=-1).shape == (TINY_BATCH, n_anchors)
    assert torch.allclose(
        per_coordinate,
        kld_tensor(mu_prior, logvar_prior, mu_full, logvar_full),
        atol=1e-5,
        rtol=1e-5,
    )
