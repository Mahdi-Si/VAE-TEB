r"""The two lag quantities are two different numbers, and only one of them is a content lead time.

``lag_compensated_seconds`` answers "where does a peak at lag $\ell$ sit on a *stored-coefficient*
axis". ``physical_lag_seconds`` answers a different question -- how far apart are the two content
epochs the source and target coefficients summarise -- and it needs an input the other does not
take: one composed group-delay reference per stream, in seconds.

**The stored timeline is canonical.** No test here carries a sensor-timeline or dataset-shift
term, and the identity has none: a version that reintroduced a constant offset fails the
equal-reference case below, which pins the answer to the bare grid term.

The two regimes the identity has to reproduce, both asserted below:

* **Equal references.** Whatever delay both streams carry cancels out of a lag, so the answer
  collapses to the grid term. A run whose target and source are on the same clock therefore
  reports a lag that mentions no filter at all.
* **A zero target reference** -- the raw-target cells, where the target is a raw sample and passes
  through no filter. The source reference survives in full, which is the whole reason the physical
  lag is recoverable there and not in the feature-target cells.

Pure arithmetic over a module that imports no model, so nothing here builds a network.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets import lag_report
from teb_vae.lag_attn.nets.lag_report import (
    SECONDS_PER_STEP,
    lag_compensated_seconds,
    physical_lag_seconds,
)

#: The shipped source reference: filter $30$'s composed delay at $0.008240$ Hz, which is the
#: slowest kept target channel and therefore the clock both streams are shifted onto. Written out
#: rather than resolved from a shard, because this file tests arithmetic and not a dataset.
SHIPPED_REFERENCE_S = 402.1604


def test_the_module_carries_no_dataset_shift_term():
    """The builder's UP shift is part of the stored signal, and nothing downstream undoes it.

    Pinned by name so a helper that undoes it cannot come back under the old names: a reader who
    finds this test knows the absence is a decision rather than an omission.
    """
    for forbidden in ("MECHANICAL_SHIFT_SECONDS", "lag_original_sensor_seconds"):
        assert not hasattr(lag_report, forbidden), forbidden


def test_equal_references_leave_the_grid_term_alone():
    r"""$\tau^u_{\mathrm{ref}} = \tau^y_{\mathrm{ref}}$: a delay common to both streams cancels.

    Only the *difference* of the two references appears in the identity, because a delay shared by
    both coefficient sequences shifts them together and a lag between them does not see it. Asserted
    at a reference far from zero, so a version that dropped one of the two terms fails here -- and
    with no constant offset at all, so a version that reintroduced one fails here too.
    """
    for lag in (0, 1, 7, 90):
        for horizon in (0, 3, 29):
            expected = SECONDS_PER_STEP * (lag + 1 + horizon)
            assert physical_lag_seconds(
                lag,
                source_reference_s=SHIPPED_REFERENCE_S,
                target_reference_s=SHIPPED_REFERENCE_S,
                horizon_element=horizon,
            ) == pytest.approx(expected)


def test_a_zero_target_reference_keeps_the_whole_source_reference():
    r"""The raw-target case: $\tau^y \equiv 0$, so the answer is the grid term plus $\tau^u_{\mathrm{ref}}$.

    This is the regime the two raw-target cells are in, and the one that makes a lag in seconds
    reportable at all: a raw sample passes through no filter, so its epoch is the instant it is at
    and the bias is one known constant rather than a channel-pair-indexed range.
    """
    for lag in (0, 5, 90):
        for horizon in (0, 29):
            expected = SECONDS_PER_STEP * (lag + 1 + horizon) + SHIPPED_REFERENCE_S
            assert physical_lag_seconds(
                lag, source_reference_s=SHIPPED_REFERENCE_S, horizon_element=horizon
            ) == pytest.approx(expected)
    # The default is the raw-target value rather than a convention, so omitting it and stating it
    # must agree -- a default of anything else would make every crws caption quietly wrong.
    assert physical_lag_seconds(3, source_reference_s=SHIPPED_REFERENCE_S) == pytest.approx(
        physical_lag_seconds(3, source_reference_s=SHIPPED_REFERENCE_S, target_reference_s=0.0)
    )


def test_the_realised_delay_factor_scales_only_the_reference_difference():
    r"""$\kappa$ multiplies $(\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}})$ and nothing else.

    The grid term is exact stored-step arithmetic and must not move with a content-delay
    convention; the reference difference is the only approximate part of the identity.
    """
    source, target = 288.2672, 402.1604
    for kappa in (1.0, 0.875, 0.5):
        expected = SECONDS_PER_STEP * (10 + 1 + 4) + kappa * (source - target)
        assert physical_lag_seconds(
            10,
            source_reference_s=source,
            target_reference_s=target,
            horizon_element=4,
            realised_delay_factor=kappa,
        ) == pytest.approx(expected)


def test_it_is_not_the_compensated_lag_under_any_argument():
    r"""The two quantities differ by $\Delta(1 + h) + \tau^u_{\mathrm{ref}} - \Delta\delta$.

    Worth pinning because the failure mode is a caption, not an exception: a page that reached for
    ``lag_compensated_seconds`` where it meant this one would draw an axis short by the whole
    reference -- $402$ s at the shipped configuration -- with every tick still a plausible number.
    """
    compensated = lag_compensated_seconds(10, delay_steps=0)
    physical = physical_lag_seconds(10, source_reference_s=SHIPPED_REFERENCE_S)
    assert physical - compensated == pytest.approx(SECONDS_PER_STEP + SHIPPED_REFERENCE_S)


def test_a_whole_lag_axis_passes_through_as_a_tensor():
    """A tensor in, a tensor out -- the shape a figure labelling a whole axis needs."""
    axis = torch.arange(6)
    seconds = physical_lag_seconds(axis, source_reference_s=SHIPPED_REFERENCE_S)
    assert isinstance(seconds, torch.Tensor) and seconds.shape == axis.shape
    assert torch.allclose(
        seconds,
        torch.tensor(
            [
                float(physical_lag_seconds(int(lag), source_reference_s=SHIPPED_REFERENCE_S))
                for lag in axis
            ],
            dtype=seconds.dtype,
        ),
    )


@pytest.mark.parametrize(
    "kwargs, named",
    [
        ({"source_reference_s": -1.0}, "source_reference_s"),
        ({"source_reference_s": 0.0, "target_reference_s": -1e-6}, "target_reference_s"),
    ],
)
def test_a_negative_reference_is_refused_by_name(kwargs, named):
    """A composed one-sided group delay cannot be negative, and a negative one shortens the answer.

    Refused rather than tolerated for the reason the delay validator gives: the wrong direction here
    is silent, and it is the direction that makes a reported lead time look better than it is.
    """
    with pytest.raises(ValueError, match=named):
        physical_lag_seconds(0, **kwargs)
