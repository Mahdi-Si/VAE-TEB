"""Smooth log-variance bounding, and why it is not a hard clamp.

A log-variance has to be bounded -- unbounded, it drifts until the Gaussian likelihood
degenerates. The obvious bound is ``torch.clamp``, and it has a failure mode that does not show
up in a loss curve: outside the range its gradient is exactly zero, so a log-variance that
saturates can never come back. It is not slow to recover, it is unable to.

A scaled sigmoid maps into the same range with a gradient that is small but strictly positive
everywhere, which is the whole reason it is here. These tests pin that difference.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn.nets.blocks import smooth_bound

_LO, _HI = -5.0, 3.0


def test_smooth_maps_into_the_open_range():
    # Kept within a range where a float32 sigmoid does not round to exactly {0, 1}.
    r = torch.linspace(-15.0, 15.0, 61)
    out = smooth_bound(r, _LO, _HI)
    assert torch.all(out > _LO) and torch.all(out < _HI)


def test_smooth_is_monotone():
    r = torch.linspace(-15.0, 15.0, 61)
    out = smooth_bound(r, _LO, _HI)
    assert torch.all(out[1:] - out[:-1] >= 0)


def test_smooth_gradient_is_nonzero_where_clamp_is_zero():
    """The reason this bound exists, asserted as a contrast.

    float64 so ``sigmoid(20)`` is not rounded to exactly $1.0$ (it is $\\approx 1 - 2\\cdot
    10^{-9}$); the smooth-bound gradient there is tiny ($\\approx 1.6\\cdot 10^{-8}$) but
    strictly positive, whereas a hard clamp is exactly zero once $|r|$ leaves the range.
    """
    r = torch.tensor([20.0, -20.0], dtype=torch.float64, requires_grad=True)
    smooth_bound(r, _LO, _HI).sum().backward()
    assert r.grad is not None
    assert torch.all(r.grad.abs() > 0), "smooth bound has zero gradient at |r|=20"

    clamped = torch.tensor([20.0, -20.0], dtype=torch.float64, requires_grad=True)
    torch.clamp(clamped, _LO, _HI).sum().backward()
    assert clamped.grad is not None
    assert torch.all(clamped.grad == 0), "clamp should have zero gradient at |r|=20"


def test_smooth_is_not_idempotent():
    """Callers must bound a raw value, never an already-bounded one.

    The heads return their pre-bound raw log-variance precisely so this can be respected; if the
    map were idempotent that return value would be redundant.
    """
    r = torch.linspace(-4.0, 2.0, 25)
    once = smooth_bound(r, _LO, _HI)
    twice = smooth_bound(once, _LO, _HI)
    assert not torch.allclose(once, twice)


def test_range_endpoints_are_respected():
    r = torch.zeros(1)
    # sigmoid(0) = 0.5, so the midpoint of the range.
    assert torch.allclose(smooth_bound(r, _LO, _HI), torch.tensor([(_LO + _HI) / 2.0]))
