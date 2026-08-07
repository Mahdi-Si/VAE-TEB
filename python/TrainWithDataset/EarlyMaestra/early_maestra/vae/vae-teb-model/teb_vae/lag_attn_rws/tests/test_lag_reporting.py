r"""The two lag quantities, pinned.

The failure this guards against is not a crash: it is a figure axis or a reported number that
silently carries -- or silently drops -- one of the two corrections. Both arms are asserted with
literal expected values rather than by re-deriving the formula, because a test that recomputes
the arithmetic under test passes whatever the arithmetic happens to be.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.lag_report import (
    MECHANICAL_SHIFT_SECONDS,
    SECONDS_PER_STEP,
    lag_compensated_seconds,
    lag_original_sensor_seconds,
)


def test_the_step_and_shift_constants_are_the_pipelines():
    """$4$ s per decimated step and the $20$ s mechanical correction preprocessing removes."""
    assert SECONDS_PER_STEP == pytest.approx(4.0)
    assert MECHANICAL_SHIFT_SECONDS == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("lag_step", "expected"), [(0, 0.0), (1, 4.0), (5, 20.0), (90, 360.0)]
)
def test_with_no_input_delay_the_compensated_lag_is_four_seconds_a_step(lag_step, expected):
    """$\\delta = 0$: the default, and the whole ungated configuration."""
    assert lag_compensated_seconds(lag_step) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lag_step", "delay_steps", "expected"), [(0, 30, 120.0), (5, 3, 32.0), (10, 30, 160.0)]
)
def test_a_nonzero_input_delay_lengthens_the_reported_lag(lag_step, delay_steps, expected):
    """A source memory read $\\delta$ steps stale puts the true lag $\\delta$ steps further back;
    dropping the term would report a delayed channel as if it were prompt."""
    assert lag_compensated_seconds(lag_step, delay_steps=delay_steps) == pytest.approx(expected)


@pytest.mark.parametrize(("lag_step", "delay_steps"), [(0, 0), (5, 0), (5, 3), (90, 30)])
def test_the_sensor_timeline_is_the_compensated_one_plus_the_mechanical_shift(
    lag_step, delay_steps
):
    """The two quantities differ by exactly $20$ s, always -- which is the whole reason they must
    not be reported interchangeably."""
    compensated = lag_compensated_seconds(lag_step, delay_steps=delay_steps)
    sensor = lag_original_sensor_seconds(lag_step, delay_steps=delay_steps)

    assert sensor - compensated == pytest.approx(20.0)


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
