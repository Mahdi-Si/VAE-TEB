r"""The derangement must have no fixed points, structurally.

The permutation control is only a control if every target really is paired with a *different*
recording's source. A single fixed point silently turns one sample's "shuffled" reading back into
its true reading, weakening the contrast in the safe direction -- the model looks more
source-specific than it is, and nothing anywhere raises.

Sattolo's algorithm gives the guarantee structurally: it samples uniformly from the cyclic
permutations, and every cycle of length $B \ge 2$ is fixed-point-free.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.controls import make_derangement


@pytest.mark.parametrize("batch_size", [2, 3, 4, 5, 8, 16, 33, 128])
def test_derangement_has_no_fixed_points(batch_size):
    perm = make_derangement(batch_size)
    identity = torch.arange(batch_size)
    assert perm.shape == (batch_size,)
    assert perm.dtype == torch.long
    assert not bool((perm == identity).any()), f"fixed point at B={batch_size}"


@pytest.mark.parametrize("batch_size", [2, 3, 8, 33])
def test_derangement_is_a_permutation(batch_size):
    perm = make_derangement(batch_size)
    assert torch.equal(perm.sort().values, torch.arange(batch_size))


def test_derangement_is_deterministic_under_a_seeded_generator():
    first = make_derangement(16, generator=torch.Generator().manual_seed(0))
    second = make_derangement(16, generator=torch.Generator().manual_seed(0))
    assert torch.equal(first, second)


def test_derangement_draws_more_than_one_permutation():
    """A constant "derangement" would have no fixed points and still be useless."""
    generator = torch.Generator().manual_seed(0)
    seen = {tuple(make_derangement(8, generator=generator).tolist()) for _ in range(50)}
    assert len(seen) > 1


@pytest.mark.parametrize("batch_size", [0, 1])
def test_derangement_rejects_a_degenerate_batch(batch_size):
    """No derangement of fewer than two elements exists; callers must skip the control."""
    with pytest.raises(ValueError, match="batch_size >= 2"):
        make_derangement(batch_size)
