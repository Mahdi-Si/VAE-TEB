r"""The lag axis: the arithmetic that is shared, and the claim that is not.

The conversion $\tau_\ell = 4(\ell + \delta)$ is the family's, computed here through the one shared
converter -- an axis assembled its own way is how a figure and the number quoted beside it come to
disagree, and that failure has happened in this repository before, by up to thirty steps.

What is *this* package's is what the axis is time **in**. The streams this cell reads are wavelet
moduli and phase harmonics out of a strictly one-sided bank, and a one-sided filter has a group
delay: a coefficient stored at step $t$ summarises signal content centred somewhere before $t$, by
a per-channel amount reaching $791$ s -- the same order as the whole lag search. So a peak's
position on this axis is **not** a physiological latency, and the two constants asserted below are
what stop a caption from saying that it is.

Both are pinned by their *content* rather than by their exact wording: what matters is that the
label names the domain, that the caveat carries both figures a reader needs to see the comparison,
and that neither word this domain cannot support survives anywhere in either.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.nets import lag_report
from teb_vae.lag_attn_cfs.eval import figures_seam, lag_axis

_MODULE_SOURCE = Path(lag_axis.__file__).read_text(encoding="utf-8")


# =================================================================================================
# The label and the caveat
# =================================================================================================
def test_the_axis_label_names_the_domain_the_coefficients_live_in() -> None:
    label = lag_axis.COEFFICIENT_LAG_AXIS_LABEL

    assert "stored-coefficient" in label
    assert "lag" in label and "s," in label


@pytest.mark.parametrize("forbidden", ["bpm", "physiological", "physiologic"])
def test_the_axis_label_claims_neither_a_unit_nor_a_latency_it_cannot_support(
    forbidden: str,
) -> None:
    """``bpm`` has no meaning over wavelet coefficients, and ``physiological`` is precisely the
    reading the group delay forbids. The sibling's label says neither, and it is still the wrong
    label here -- which is why this package binds its own rather than importing that one."""
    assert forbidden not in lag_axis.COEFFICIENT_LAG_AXIS_LABEL.lower()


def test_the_caveat_carries_both_figures_so_the_comparison_can_be_read() -> None:
    r"""The point is not either number: it is that a group delay of up to $791$ s sits against a
    lag search of $364$ s, so a reader handed only the lag figures could not know the axis is
    uncorrected for something of its own size."""
    caveat = lag_axis.GROUP_DELAY_CAVEAT

    assert "791" in caveat
    assert "364" in caveat
    assert lag_axis.MAX_MEASURED_GROUP_DELAY_SECONDS == 791.0
    assert lag_axis.SHIPPED_LAG_SPAN_SECONDS == 364.0


def test_the_caveat_refuses_the_name_the_readout_must_not_be_given() -> None:
    """The lag map is an attribution over stored-coefficient time. Calling it a transfer entropy
    would claim a correction nothing in this pipeline makes."""
    caveat = lag_axis.GROUP_DELAY_CAVEAT

    assert "not a physiological latency" in caveat
    assert "not a transfer entropy" in caveat


def test_the_lag_span_is_derived_from_the_shared_step_length_rather_than_restated() -> None:
    r"""$364 = 91 \times 4$, and the $4$ is the family's decimation constant. A restated literal is
    how two modules come to disagree about how long a step is."""
    assert lag_axis.SECONDS_PER_STEP is lag_report.SECONDS_PER_STEP
    assert lag_axis.SHIPPED_LAG_SPAN_SECONDS == (
        (lag_axis.SHIPPED_MAX_LAG_STEPS + 1) * lag_report.SECONDS_PER_STEP
    )

    imported = {
        alias.name
        for node in ast.walk(ast.parse(_MODULE_SOURCE))
        if isinstance(node, ast.ImportFrom) and node.module == "teb_vae.lag_attn.nets.lag_report"
        for alias in node.names
    }
    assert {"SECONDS_PER_STEP", "lag_compensated_seconds"} <= imported


def test_the_figure_seam_draws_this_label_rather_than_the_raw_cells_one() -> None:
    """The one edit that would put the wrong claim on every lag figure in this package while
    changing no arithmetic at all: binding ``lag_report.COMPENSATED_LAG_AXIS_LABEL`` here."""
    assert figures_seam.COEFFICIENT_LAG_AXIS_LABEL is lag_axis.COEFFICIENT_LAG_AXIS_LABEL
    assert figures_seam.GROUP_DELAY_CAVEAT is lag_axis.GROUP_DELAY_CAVEAT
    assert not hasattr(figures_seam, "COMPENSATED_LAG_AXIS_LABEL")
    assert (
        lag_axis.COEFFICIENT_LAG_AXIS_LABEL != lag_report.COMPENSATED_LAG_AXIS_LABEL
    )


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
