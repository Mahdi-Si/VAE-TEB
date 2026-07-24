r"""The masked raw likelihood: summed over the block, averaged over anchors, in nats.

The numbers are pinned against hand-computed constants at the tiny block size
($H \cdot R = 4 \cdot 16 = 64$), because "summed, not averaged, over its own axes" is exactly
the property a silently mean-reduced implementation would fake on random data.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_rws.nets.losses import masked_raw_likelihood

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
