r"""S3-T02: ``make_derangement`` never leaks a fixed point into the permutation control.

If :math:`\pi(i) = i` for any :math:`i`, that sample's "shuffled" source is its *true*
source, and :math:`K_{\mathrm{shuffled}}` silently absorbs real transfer entropy -- the
negative control would understate itself. Sattolo's algorithm makes the guarantee structural
(it draws from the cyclic permutations, all of which are fixed-point-free) rather than
probabilistic, so these tests assert the invariant exhaustively over sizes and seeds.
"""
from __future__ import annotations

import pytest
import torch

from model.vae_teb_prediction.model.vae_teb_lag_attn_v3 import make_derangement


@pytest.mark.parametrize("batch_size", [2, 3, 4, 5, 8, 16, 33, 128])
def test_no_fixed_points_across_seeds(batch_size):
    for seed in range(40):
        g = torch.Generator().manual_seed(seed)
        perm = make_derangement(batch_size, generator=g)
        assert bool((perm != torch.arange(batch_size)).all()), (
            f"fixed point at batch_size={batch_size}, seed={seed}: {perm.tolist()}"
        )


@pytest.mark.parametrize("batch_size", [2, 7, 64])
def test_is_a_valid_permutation(batch_size):
    g = torch.Generator().manual_seed(1234)
    perm = make_derangement(batch_size, generator=g)
    assert perm.dtype == torch.long
    assert perm.shape == (batch_size,)
    assert torch.equal(torch.sort(perm).values, torch.arange(batch_size))


def test_deterministic_under_a_seeded_generator():
    a = make_derangement(16, generator=torch.Generator().manual_seed(99))
    b = make_derangement(16, generator=torch.Generator().manual_seed(99))
    c = make_derangement(16, generator=torch.Generator().manual_seed(100))
    assert torch.equal(a, b)
    assert not torch.equal(a, c), "different seeds produced the same derangement"


def test_draws_more_than_one_permutation():
    """Guard against a degenerate implementation that always returns the same shift."""
    seen = {
        tuple(make_derangement(6, generator=torch.Generator().manual_seed(s)).tolist())
        for s in range(50)
    }
    assert len(seen) > 1


@pytest.mark.parametrize("batch_size", [0, 1])
def test_raises_when_no_derangement_exists(batch_size):
    with pytest.raises(ValueError, match="batch_size >= 2"):
        make_derangement(batch_size)
