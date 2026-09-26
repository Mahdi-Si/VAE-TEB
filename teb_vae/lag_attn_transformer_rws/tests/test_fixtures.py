r"""The shared fixtures are themselves load-bearing, so they get their own tests.

Two failure modes here would make the rest of the suite lie silently. A stub batch whose
``weight`` stopped planting its gap inside the trained-anchor range would leave every mask test
green whether or not the masks work. And the causality probe is the shape of every structural
assertion in this package -- if its negative controls were dropped, a module reading the future or
a module returning zeros would pass all of them.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    BATCH,
    SEQ_LEN,
    STUB_GAP_STEP,
    TINY_KWARGS,
    assert_token_causal,
    make_stub_batch,
    resample_after,
)


def test_the_stub_batch_plants_its_gap():
    """A silently gap-free fixture would leave every mask test vacuous."""
    batch = make_stub_batch(BATCH, SEQ_LEN)
    assert (batch.weight[:, STUB_GAP_STEP] == 0.0).all()
    assert (batch.weight == 0.0).any()
    # The gap sits inside the tiny trained-anchor range, where every mask can see it.
    assert TINY_KWARGS["warmup_period"] <= STUB_GAP_STEP
    assert STUB_GAP_STEP < SEQ_LEN - int(TINY_KWARGS["horizon"])


def test_resample_after_leaves_the_prefix_bit_identical():
    """The probe would report causality that was never tested if it touched the prefix."""
    x = torch.randn(2, 8, 3)

    perturbed = resample_after(x, 3)

    assert torch.equal(perturbed[:, :4], x[:, :4])
    assert not torch.equal(perturbed[:, 4:], x[:, 4:])


def test_the_causality_probe_fails_on_a_non_causal_module():
    """A probe that cannot fail is not a probe.

    The failure it must catch is the one this architecture is actually exposed to: a statistic
    pooled over time. A cumulative *reverse* mean is the cheapest module with that property.
    """

    class _ReadsTheFuture(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flip(1).cumsum(1).flip(1)

    x = torch.randn(2, 8, 3)
    assert_token_causal(nn.Identity(), x, 3, label="identity")
    with pytest.raises(AssertionError, match="moved by"):
        assert_token_causal(_ReadsTheFuture(), x, 3, label="leaky")


def test_the_causality_probe_fails_on_a_dead_module():
    """The negative-control half: a module returning zeros is bit-stable everywhere."""

    class _Dead(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    with pytest.raises(AssertionError, match="never reached"):
        assert_token_causal(_Dead(), torch.randn(2, 8, 3), 3, label="dead")
