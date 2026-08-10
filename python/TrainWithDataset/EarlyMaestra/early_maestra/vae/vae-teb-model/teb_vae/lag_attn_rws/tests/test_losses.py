r"""The per-anchor loss primitives: the raw likelihood, the prior rate, and the three shape terms.

The numbers are pinned against hand-computed constants -- the likelihood at the tiny block size
($H \cdot R = 4 \cdot 16 = 64$), the prior rate at the shipped latent width ($d_z = 64$) --
because "summed, not averaged, over its own axes" is exactly the property a silently
mean-reduced implementation would fake on random data.

The shape terms are pinned the same way and additionally against the mask, because each is an
easy thing to write in a form that *looks* masked and is not: the multiscale term pools, so a
mask applied after pooling would smear a gap sentinel across a whole window; the derivative term
consumes two samples per element, so a mask taken from one of them admits half of every gap edge;
and the boundary term reaches into a neighbouring anchor's block for the sample it compares
against, which is exactly where a validity decision can be taken from the wrong anchor.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.nets.losses import (
    MS_RATES,
    masked_boundary_gap,
    masked_derivative_huber,
    masked_multiscale_l1,
    masked_prior_rate,
    masked_raw_likelihood,
    masked_source_kl,
)

_B, _T_VALID, _H, _R = 2, 12, 4, 16
_BLOCK = _H * _R  # 64 raw samples per anchor at the tiny geometry


def _tensors(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(_B, _T_VALID, _H, _R, generator=generator)
    target = torch.randn(_B, _T_VALID, _H, _R, generator=generator)
    logvar = torch.randn(_B, _T_VALID, _H, _R, generator=generator)
    mask = torch.ones(_B, _T_VALID, _H)
    return mu, target, logvar, mask


def test_an_unknown_likelihood_is_rejected_listing_the_choices():
    mu, target, logvar, mask = _tensors()
    with pytest.raises(ValueError, match=r"mse.*gaussian_nll"):
        masked_raw_likelihood(mu, target, mask, likelihood="huber", logvar=logvar)


def test_gaussian_without_a_logvar_is_rejected():
    mu, target, _, mask = _tensors()
    with pytest.raises(ValueError, match="logvar"):
        masked_raw_likelihood(mu, target, mask, likelihood="gaussian_nll")


def test_block_and_sample_differ_by_exactly_the_block_size():
    mu, target, logvar, mask = _tensors()
    d_block, d_sample = masked_raw_likelihood(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar
    )
    assert torch.allclose(d_block, _BLOCK * d_sample, rtol=1e-6)


def test_mse_is_summed_over_the_block_not_averaged():
    """Constant error e = 0.5 everywhere: the block value must be 64 * 0.25, not 0.25."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    target = torch.full_like(mu, 0.5)
    mask = torch.ones(_B, _T_VALID, _H)
    d_block, d_sample = masked_raw_likelihood(mu, target, mask, likelihood="mse")
    assert torch.allclose(d_block, torch.tensor(_BLOCK * 0.25))
    assert torch.allclose(d_sample, torch.tensor(0.25))


def test_the_gaussian_carries_its_full_constant():
    """A perfect forecast at unit variance is not zero nats: each sample still costs
    0.5*log(2 pi), so the block value is 64 times that. A dropped constant would make every
    later ELBO claim quietly wrong."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    d_block, _ = masked_raw_likelihood(
        mu, mu.clone(), torch.ones(_B, _T_VALID, _H),
        likelihood="gaussian_nll", logvar=torch.zeros_like(mu),
    )
    assert torch.allclose(d_block, torch.tensor(_BLOCK * 0.5 * math.log(2.0 * math.pi)))


@pytest.mark.parametrize("likelihood", ["mse", "gaussian_nll"])
def test_a_masked_sample_contributes_exactly_zero(likelihood):
    """Planting a large (finite) error at a masked position leaves the loss bitwise
    unchanged -- which is the property that keeps the -11 sigma gap sentinels out."""
    mu, target, logvar, mask = _tensors()
    mask[:, 5, 2] = 0.0

    reference = masked_raw_likelihood(
        mu, target, mask, likelihood=likelihood, logvar=logvar
    )
    planted_target = target.clone()
    planted_target[:, 5, 2, :] = 1.0e6
    planted = masked_raw_likelihood(
        mu, planted_target, mask, likelihood=likelihood, logvar=logvar
    )
    assert torch.equal(reference[0], planted[0])
    assert torch.equal(reference[1], planted[1])


def test_fully_masked_anchors_leave_the_average_unchanged():
    """The denominator counts contributing anchors, so masking whole anchors out of uniform
    data must not move the per-anchor value -- an all-anchor denominator would dilute it."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    target = torch.full_like(mu, 2.0)
    full_mask = torch.ones(_B, _T_VALID, _H)
    partial_mask = full_mask.clone()
    partial_mask[:, :3] = 0.0

    d_full, _ = masked_raw_likelihood(mu, target, full_mask, likelihood="mse")
    d_partial, _ = masked_raw_likelihood(mu, target, partial_mask, likelihood="mse")
    assert torch.allclose(d_full, d_partial)


def test_an_all_masked_batch_returns_zero_not_nan():
    mu, target, logvar, _ = _tensors()
    zero_mask = torch.zeros(_B, _T_VALID, _H)
    d_block, d_sample = masked_raw_likelihood(
        mu, target, zero_mask, likelihood="gaussian_nll", logvar=logvar
    )
    assert float(d_block) == 0.0 and float(d_sample) == 0.0


def test_gradient_reaches_only_unmasked_positions():
    mu, target, logvar, mask = _tensors()
    mask[:, 5, 2] = 0.0
    mu.requires_grad_(True)
    d_block, _ = masked_raw_likelihood(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar
    )
    d_block.backward()
    assert mu.grad is not None
    assert float(mu.grad[:, 5, 2].abs().max()) == 0.0
    assert float(mu.grad[:, 5, 1].abs().max()) > 0.0


# --------------------------------------------------------------------------------------
# The masked prior rate
# --------------------------------------------------------------------------------------
_D_Z = 64  # the shipped latent width; the hand constants below are stated at this width

#: Per-dimension rate 0.5 * (e^lv - 1 - lv) at hand-picked log-variances: the optimum, the
#: clamp floor the reported production run pinned on, and the value that run started from.
_RATE_AT_LV0 = 0.0
_RATE_AT_FLOOR = 2.003369  # lv = -5
_RATE_AT_START = 1.063457  # lv = -3.081


def _prior_mask() -> torch.Tensor:
    return torch.ones(_B, _T_VALID)


def test_the_prior_rate_is_exactly_zero_at_unit_variance():
    """The anchor's optimum: at lv = 0 every dimension contributes exactly zero, so the total
    is zero rather than merely small."""
    rate = masked_prior_rate(torch.zeros(_B, _T_VALID, _D_Z), _prior_mask())
    assert float(rate) == _RATE_AT_LV0


@pytest.mark.parametrize(
    ("logvar", "per_dim"),
    [(-5.0, _RATE_AT_FLOOR), (-3.081, _RATE_AT_START)],
    ids=["clamp-floor", "reported-run-start"],
)
def test_the_prior_rate_matches_the_hand_computed_constants(logvar, per_dim):
    """Pinned against constants computed by hand, not against the implementation: at the clamp
    floor a fully-masked anchor costs 64 * 2.003369 = 128.22 nats, at the reported run's
    starting log-variance 64 * 1.063457 = 68.06."""
    rate = masked_prior_rate(
        torch.full((_B, _T_VALID, _D_Z), logvar), _prior_mask()
    )
    assert float(rate) == pytest.approx(_D_Z * per_dim, rel=1e-6)


@pytest.mark.parametrize("logvar", [-1.0, 1.0], ids=["narrow", "wide"])
def test_the_prior_rate_is_strictly_positive_off_the_optimum(logvar):
    """Both directions: a narrow prior and a wide one are both charged, which is what makes the
    term an anchor at sigma = 1 rather than a one-sided floor."""
    rate = masked_prior_rate(torch.full((_B, _T_VALID, _D_Z), logvar), _prior_mask())
    assert float(rate) > 0.0


def test_the_prior_rate_shares_the_source_kls_support_and_denominator():
    """The addability contract: reduced through ``masked_source_kl``'s own reduction, the same
    per-dimension tensor gives bitwise the same scalar -- same mask, same contributing-anchor
    count -- so the two terms are in one unit and summable without rescaling."""
    generator = torch.Generator().manual_seed(0)
    logvar = torch.randn(_B, _T_VALID, _D_Z, generator=generator)
    mask = _prior_mask()
    mask[:, :3] = 0.0  # fully-masked anchors must leave the denominator too

    rate = masked_prior_rate(logvar, mask)
    rate_btd = 0.5 * (logvar.exp() - 1.0 - logvar)
    through_source_reduction = masked_source_kl(rate_btd, mask)["source_conditioned_kl_raw"]

    assert torch.equal(rate, through_source_reduction)


def test_a_masked_anchors_log_variance_cannot_move_the_prior_rate():
    generator = torch.Generator().manual_seed(1)
    logvar = torch.randn(_B, _T_VALID, _D_Z, generator=generator)
    mask = _prior_mask()
    mask[:, 4] = 0.0

    reference = masked_prior_rate(logvar, mask)
    poisoned = logvar.clone()
    poisoned[:, 4] = -1.0e3
    assert torch.equal(reference, masked_prior_rate(poisoned, mask))


def test_an_empty_prior_mask_returns_zero_not_nan():
    assert float(masked_prior_rate(torch.randn(_B, _T_VALID, _D_Z), torch.zeros(_B, _T_VALID))) == 0.0


def test_the_prior_rate_carries_gradient():
    """Unlike the raw KL readout: weighted, this is an objective term, so it must sit in the
    graph -- and its gradient must respect the mask."""
    logvar = torch.full((_B, _T_VALID, _D_Z), -2.0, requires_grad=True)
    mask = _prior_mask()
    mask[:, 5] = 0.0

    rate = masked_prior_rate(logvar, mask)
    assert rate.requires_grad
    rate.backward()
    assert logvar.grad is not None
    assert float(logvar.grad[:, 5].abs().max()) == 0.0
    assert float(logvar.grad[:, 4].abs().max()) > 0.0


# --------------------------------------------------------------------------------------
# The multiscale L1 term
# --------------------------------------------------------------------------------------
#: Pooled-element count of one tiny anchor's block, summed over the three rates:
#: $64/1 + 64/4 + 64/16 = 64 + 16 + 4 = 84$. Every hand constant below is stated at this number,
#: which is why the rates themselves are pinned first -- changing them changes what these tests
#: mean, and a silently re-derived constant would hide that.
_MS_POOLED_ELEMENTS = 84


def test_the_pooling_rates_are_the_documented_ones():
    """The constants below are stated at these three rates."""
    assert MS_RATES == (1, 4, 16)
    assert sum(_BLOCK // rate for rate in MS_RATES) == _MS_POOLED_ELEMENTS


def test_the_multiscale_term_sums_over_every_scale():
    """A constant offset of $0.5$: every pooled element at every rate is off by exactly $0.5$
    (an average of a constant is that constant), so the per-anchor value is $84 \\cdot 0.5$.
    A term that pooled at one rate only would read $32$, and one that averaged over scales
    instead of summing would read $16$."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    target = torch.full_like(mu, 0.5)

    value = masked_multiscale_l1(mu, target, torch.ones(_B, _T_VALID, _H))

    assert float(value) == pytest.approx(_MS_POOLED_ELEMENTS * 0.5, rel=1e-6)


def test_a_masked_position_cannot_move_the_multiscale_term():
    """The reason both operands are masked *before* pooling: after pooling, a gap sentinel at a
    masked position would already have been averaged into every pool that covers it, and no mask
    applied afterwards could take it back out."""
    mu, target, _logvar, mask = _tensors()
    mask[:, 5, 2] = 0.0

    reference = masked_multiscale_l1(mu, target, mask)
    planted = target.clone()
    planted[:, 5, 2, :] = 1.0e6

    assert torch.equal(reference, masked_multiscale_l1(mu, planted, mask))


def test_the_multiscale_denominator_counts_contributing_anchors():
    """Uniform data, whole anchors masked out: the per-anchor value must not move. An
    all-anchor denominator would dilute it by the masked fraction."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    target = torch.full_like(mu, 2.0)
    full_mask = torch.ones(_B, _T_VALID, _H)
    partial_mask = full_mask.clone()
    partial_mask[:, :3] = 0.0

    assert torch.allclose(
        masked_multiscale_l1(mu, target, full_mask),
        masked_multiscale_l1(mu, target, partial_mask),
    )


def test_an_all_masked_batch_gives_a_finite_zero_multiscale():
    mu, target, _logvar, _mask = _tensors()
    value = masked_multiscale_l1(mu, target, torch.zeros(_B, _T_VALID, _H))
    assert float(value) == 0.0 and torch.isfinite(value)


def test_a_block_shorter_than_the_coarsest_rate_is_refused_naming_the_geometry():
    """A feature-target model with a narrow surviving channel set reaches this, and a silent
    fallback -- skipping the coarse scale, or padding the block -- would make the term mean
    something different at that geometry with nothing saying so."""
    short = torch.zeros(_B, _T_VALID, 2, 4)  # flattened block of 8 < max(MS_RATES)
    with pytest.raises(ValueError, match=r"8 elements.*H=2 x X=4.*16"):
        masked_multiscale_l1(short, short.clone(), torch.ones(_B, _T_VALID, 2))


# --------------------------------------------------------------------------------------
# The derivative Huber term
# --------------------------------------------------------------------------------------
#: Difference pairs inside one anchor's flattened block: $H \cdot R - 1$.
_PAIRS = _BLOCK - 1


def _ramped_target(slope: float) -> torch.Tensor:
    """A target whose flattened block rises by ``slope`` between consecutive samples."""
    positions = torch.arange(_BLOCK, dtype=torch.float32).view(1, 1, _H, _R)
    return (slope * positions).expand(_B, _T_VALID, _H, _R).contiguous()


@pytest.mark.parametrize(
    ("slope", "per_pair"),
    # Quadratic branch: 0.5 * s^2 at s = 0.5. Linear branch: delta * (|s| - delta/2) at s = 3,
    # delta = 1 -- the whole point of Huber over L2, and the only thing that pins delta.
    [(0.5, 0.125), (3.0, 2.5)],
    ids=["quadratic-branch", "linear-branch"],
)
def test_the_derivative_term_matches_the_hand_computed_huber(slope, per_pair):
    """A flat forecast against a constant-slope target: every one of the $63$ pairs inside an
    anchor is off by exactly ``slope``."""
    mu = torch.zeros(_B, _T_VALID, _H, _R)
    value = masked_derivative_huber(mu, _ramped_target(slope), torch.ones(_B, _T_VALID, _H))

    assert float(value) == pytest.approx(_PAIRS * per_pair, rel=1e-6)


def test_a_masked_position_cannot_move_the_derivative_term():
    """Both pairs that touch a masked sample are excluded, which is what the *product* of the
    pair's two mask entries buys over either one of them alone."""
    mu, target, _logvar, mask = _tensors()
    mask[:, 5, 2] = 0.0

    reference = masked_derivative_huber(mu, target, mask)
    planted = target.clone()
    planted[:, 5, 2, :] = 1.0e6

    assert torch.equal(reference, masked_derivative_huber(mu, planted, mask))


def test_an_all_masked_batch_gives_a_finite_zero_derivative():
    mu, target, _logvar, _mask = _tensors()
    value = masked_derivative_huber(mu, target, torch.zeros(_B, _T_VALID, _H))
    assert float(value) == 0.0 and torch.isfinite(value)


def test_both_shape_terms_carry_gradient_only_where_the_mask_admits_it():
    """Weighted, these are objective terms rather than diagnostics, so they must sit in the
    graph -- and a gradient at a masked position would train the model against a gap."""
    for term in (masked_multiscale_l1, masked_derivative_huber):
        mu, target, _logvar, mask = _tensors()
        mask[:, 5, 2] = 0.0
        mu.requires_grad_(True)

        value = term(mu, target, mask)
        assert value.requires_grad, term.__name__
        value.backward()

        assert mu.grad is not None
        assert float(mu.grad[:, 5, 2].abs().max()) == 0.0, term.__name__
        assert float(mu.grad[:, 5, 1].abs().max()) > 0.0, term.__name__


# --------------------------------------------------------------------------------------
# The boundary continuity term
# --------------------------------------------------------------------------------------
_BOUNDARY_T_VALID, _BOUNDARY_H, _BOUNDARY_X = 3, 2, 4


def _boundary_case():
    r"""A hand-set geometry where each anchor's boundary gap is a written-down number.

    Anchor $t$'s first forecast sample is ``mu[:, t, 0, 0]``; the sample it must continue is
    ``target[:, t-1, 0, -1]``, because anchor $t-1$'s horizon step $0$ *is* decimated step $t$.
    Gaps: $|5 - 1| = 4$ at $t = 1$ and $|2 - 4| = 2$ at $t = 2$; anchor $0$ has none.

    Returns:
        ``(mu, target, mask, weight)``.
    """
    shape = (1, _BOUNDARY_T_VALID, _BOUNDARY_H, _BOUNDARY_X)
    mu, target = torch.zeros(shape), torch.zeros(shape)
    mu[0, 0, 0, 0] = 99.0  # anchor 0 is excluded structurally; an absurd value must not show
    mu[0, 1, 0, 0], mu[0, 2, 0, 0] = 5.0, 2.0
    target[0, 0, 0, -1], target[0, 1, 0, -1] = 1.0, 4.0
    mask = torch.ones(1, _BOUNDARY_T_VALID, _BOUNDARY_H)
    weight = torch.ones(1, _BOUNDARY_T_VALID + _BOUNDARY_H)
    return mu, target, mask, weight


def test_the_boundary_term_is_the_hand_computed_gap_per_anchor():
    """$(4 + 2) / 3$: summed over the two anchors that have a boundary, divided by the three
    contributing ones -- the same denominator every other term uses."""
    mu, target, mask, weight = _boundary_case()

    assert float(masked_boundary_gap(mu, target, mask, weight)) == pytest.approx(6.0 / 3.0)


def test_the_boundary_term_excludes_anchor_zero_structurally():
    """Not by assuming a warm-up: the range starts at $t = 1$, so a ``warmup_period = 0``
    geometry still has no anchor reaching for a sample before its own window. Anchor $0$'s
    forecast carries $99$ above and contributes nothing."""
    mu, target, mask, weight = _boundary_case()
    reference = masked_boundary_gap(mu, target, mask, weight)

    moved = mu.clone()
    moved[0, 0, 0, 0] = -1.0e6

    assert torch.equal(reference, masked_boundary_gap(moved, target, mask, weight))


def test_a_below_threshold_boundary_sample_contributes_nothing():
    """The anchor's own ``weight`` decides. Dropping anchor $2$'s validity removes its gap of
    $2$ while the denominator -- the contributing-anchor count, which the mask decides -- stays
    at three."""
    mu, target, mask, weight = _boundary_case()
    weight[0, 2] = 0.0

    assert float(masked_boundary_gap(mu, target, mask, weight)) == pytest.approx(4.0 / 3.0)


def test_the_boundary_validity_is_the_anchors_own_not_its_predecessors():
    """The distinction the term is written around. Anchor $0$ is dropped from the forecast mask
    entirely -- a coverage-floor decision about *its* window -- and anchor $1$'s boundary must
    still count, because the sample it continues is anchor $1$'s own last observed one and its
    validity is anchor $1$'s ``weight``. Reading ``mask[:, 0, 0]`` instead would silently zero
    it. The denominator falls to the two contributing anchors: $(4 + 2) / 2$."""
    mu, target, mask, weight = _boundary_case()
    mask[0, 0] = 0.0

    assert float(masked_boundary_gap(mu, target, mask, weight)) == pytest.approx(6.0 / 2.0)


def test_a_single_anchor_geometry_gives_a_finite_zero_boundary():
    """$T_{\\mathrm{valid}} = 1$ leaves the $t \\in [1, T_{\\mathrm{valid}})$ range empty."""
    mu = torch.randn(1, 1, _BOUNDARY_H, _BOUNDARY_X)
    value = masked_boundary_gap(
        mu, torch.randn_like(mu), torch.ones(1, 1, _BOUNDARY_H), torch.ones(1, 1 + _BOUNDARY_H)
    )
    assert float(value) == 0.0 and torch.isfinite(value)


def test_an_all_masked_batch_gives_a_finite_zero_boundary():
    mu, target, mask, weight = _boundary_case()
    value = masked_boundary_gap(mu, target, torch.zeros_like(mask), weight)
    assert float(value) == 0.0 and torch.isfinite(value)


def test_the_boundary_term_carries_gradient_into_the_first_forecast_sample_only():
    """It is a one-sample term: everything else in the block must be untouched by it, or the
    level anchor would quietly become a second reconstruction loss."""
    mu, target, mask, weight = _boundary_case()
    mu.requires_grad_(True)

    value = masked_boundary_gap(mu, target, mask, weight)
    assert value.requires_grad
    value.backward()

    assert mu.grad is not None
    assert float(mu.grad[0, 1, 0, 0].abs()) > 0.0
    assert float(mu.grad[0, 1, 0, 1:].abs().max()) == 0.0
    assert float(mu.grad[0, 1, 1].abs().max()) == 0.0
    assert float(mu.grad[0, 0].abs().max()) == 0.0  # anchor 0 has no boundary term
