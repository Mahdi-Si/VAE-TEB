r"""The stored-timeline lag quantity, pinned.

The failure this guards against is not a crash: it is a figure axis or a reported number that
silently carries -- or silently drops -- the input-delay term $\delta$. The expected values are
literals rather than a re-derivation of the formula, because a test that recomputes the
arithmetic under test passes whatever the arithmetic happens to be.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds


@pytest.mark.parametrize(
    ("lag_step", "delay_steps", "expected"),
    [
        # $\delta = 0$: the default, and the whole ungated configuration.
        (0, 0, 0.0),
        (1, 0, 4.0),
        (5, 0, 20.0),
        (90, 0, 360.0),
        # A source memory read $\delta$ steps stale puts the true lag $\delta$ steps further back;
        # dropping the term would report a delayed channel as if it were prompt.
        (0, 30, 120.0),
        (5, 3, 32.0),
        (10, 30, 160.0),
    ],
)
def test_the_compensated_lag_is_four_seconds_a_step_plus_the_input_delay(
    lag_step, delay_steps, expected
):
    assert lag_compensated_seconds(lag_step, delay_steps=delay_steps) == pytest.approx(expected)


def test_a_whole_lag_axis_converts_elementwise():
    """Figures label an axis, not a scalar; the same call must serve both."""
    axis = lag_compensated_seconds(torch.arange(4), delay_steps=2)

    assert isinstance(axis, torch.Tensor)
    assert torch.allclose(axis, torch.tensor([8.0, 12.0, 16.0, 20.0]))


def test_a_negative_delay_is_refused():
    """A negative delay reads the source memory from the future and would *shorten* the reported
    lag; it can only come from a sign error upstream."""
    with pytest.raises(ValueError, match="delay_steps"):
        lag_compensated_seconds(3, delay_steps=-1)
