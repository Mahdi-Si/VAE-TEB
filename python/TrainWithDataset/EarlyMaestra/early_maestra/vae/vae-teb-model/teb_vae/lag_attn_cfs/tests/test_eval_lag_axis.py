r"""The lag axis, and a per-lag vector read against it.

The conversion $\tau_\ell = 4(\ell + \delta)$ is the family's, computed here through the one shared
converter -- an axis assembled its own way is how a figure and the number quoted beside it come to
disagree, and that failure has happened in this repository before. The causal input delay $\delta$
is part of the axis rather than a correction applied to a reported peak afterwards.

A profile read against the axis is padded with ``NaN`` rather than zero, truncated rather than
wrapped, and a column an older run's table does not carry reads as unmeasured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from teb_vae.lag_attn.nets import lag_report
from teb_vae.lag_attn_cfs.eval import lag_axis


# =================================================================================================
# The axis itself
# =================================================================================================
def test_the_axis_is_the_shared_conversion_elementwise() -> None:
    axis = lag_axis.compensated_seconds_axis(5, delay_steps=0)

    assert axis.tolist() == [0.0, 4.0, 8.0, 12.0, 16.0]
    assert axis.dtype == np.float64


def test_a_causal_input_delay_shifts_the_whole_axis_by_its_own_amount() -> None:
    r"""A peak at lag $\ell$ refers to source content $\ell + \delta$ steps back, so the delay is
    part of the axis rather than a correction applied to a reported peak afterwards."""
    axis = lag_axis.compensated_seconds_axis(3, delay_steps=2)

    assert axis.tolist() == [8.0, 12.0, 16.0]
    assert axis.tolist() == [
        float(lag_report.lag_compensated_seconds(lag, delay_steps=2)) for lag in range(3)
    ]


# =================================================================================================
# Reading a per-lag vector against it
# =================================================================================================
def test_a_short_profile_is_padded_with_nan_rather_than_with_zero() -> None:
    """A lag whose value was never measured and a lag the source never attended to are different
    statements, and a zero would make a profile's argmax, its width and its total all read as if
    the missing bins had been measured and found empty."""
    padded = lag_axis.padded_profile([1.0, 2.0], 4)

    assert padded[:2].tolist() == [1.0, 2.0]
    assert np.isnan(padded[2:]).all()


def test_a_longer_profile_is_truncated_to_the_axis() -> None:
    assert lag_axis.padded_profile([1.0, 2.0, 3.0], 2).tolist() == [1.0, 2.0]


def test_a_column_an_older_runs_table_does_not_carry_draws_as_absent() -> None:
    """So re-running one analysis against a finished directory reports the profile as unmeasured
    rather than taking down the figure."""
    frame = pd.DataFrame({"kl_by_lag": [0.5, 0.25]})

    assert lag_axis.profile_column(frame, "kl_by_lag", 3)[:2].tolist() == [0.5, 0.25]
    assert np.isnan(lag_axis.profile_column(frame, "kl_by_lag", 3)[2])
    assert np.isnan(lag_axis.profile_column(frame, "attention_by_lag", 3)).all()
    assert np.isnan(lag_axis.profile_column(pd.DataFrame(), "kl_by_lag", 3)).all()
