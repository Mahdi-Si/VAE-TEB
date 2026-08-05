r"""The masked raw likelihood and the masked prior rate: per-anchor reductions, in nats.

The numbers are pinned against hand-computed constants -- the likelihood at the tiny block size
($H \cdot R = 4 \cdot 16 = 64$), the prior rate at the shipped latent width ($d_z = 48$) --
because "summed, not averaged, over its own axes" is exactly the property a silently
mean-reduced implementation would fake on random data.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.nets.losses import (
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
_D_Z = 48  # the shipped latent width; the hand constants below are stated at this width

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
    floor a fully-masked anchor costs 48 * 2.003369 = 96.16 nats, at the reported run's
    starting log-variance 48 * 1.063457 = 51.05."""
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
