r"""Two lag distributions compared: the four properties that make a distance column readable.

``lag_hist`` exists so that a table can say how far apart two per-lag cells are -- two clinical
classes in one window, or one class in two windows. Each of its decisions is a way that column could
be silently wrong, and each wrong version still produces a smooth, plausible trajectory:

**The unit of the transport distance is seconds on the compensated axis.** A version that transported
over lag *indices* would report a quarter of the right number and would still move in the right
direction, at the right times, for the right cohorts. It is pinned against a rigid shift whose answer
is arithmetic a reader can check: shift a profile $k$ bins and the distance is $4k$ s.

**The two distances are not interchangeable, and the tests say which is which.** Jensen--Shannon is
blind to the axis and saturates once two supports separate; Wasserstein keeps measuring. A pair of
disjoint one-hot profiles pins both halves of that at once.

**A cell with no mass is ``NaN``, never zero.** Zero is the strongest possible claim of agreement, so
returning it where nothing could be compared would put the loudest finding in the table exactly
where there is no evidence. Every entry point is asserted to refuse rather than to agree.

**Normalisation is by finite positive mass.** A non-finite bin dropped from the total and a
non-finite bin read as zero give the same *shape* and different *totals*, so only a row that carries
one shows the difference.

The axis these run against is deliberately **offset**, for the reason ``test_eval_lag_shape.py``
gives: $\tau_\ell = 4(\ell + \delta)$ with $\delta > 0$, because the compensated axis never starts
at zero and a centroid built against a zero-based axis would pass a zero-based fixture.

Following the fixture rule this suite is bound by, everything below is evidence about arithmetic,
units, symmetry and refusals -- never about the sign or magnitude of any effect on real data.
"""
from __future__ import annotations

import numpy as np
import pytest

from teb_vae.lag_attn_cfs.eval import lag_hist

#: A short axis, and one that starts where a real one does: eleven bins at 4 s with a causal input
#: delay of three steps, so $\tau_0 = 12$ s rather than $0$.
_N_LAGS = 11
_DELAY_STEPS = 3
_SECONDS_PER_STEP = 4.0
_SECONDS = _SECONDS_PER_STEP * (np.arange(_N_LAGS, dtype=np.float64) + _DELAY_STEPS)


def _one_hot(index: int, height: float = 1.0) -> np.ndarray:
    """A profile with all its mass in one bin, at an arbitrary height."""
    profile = np.zeros(_N_LAGS, dtype=np.float64)
    profile[index] = height
    return profile


def _bump(centre: int, width: float = 1.2) -> np.ndarray:
    """A normalised Gaussian bump on the lag index axis, kept well inside the axis.

    Args:
        centre: The bin the bump is centred on.
        width: Its standard deviation in bins.

    Returns:
        The $(L,)$ profile, summing to one.
    """
    offsets = np.arange(_N_LAGS, dtype=np.float64) - float(centre)
    profile = np.exp(-0.5 * (offsets / float(width)) ** 2)
    return profile / profile.sum()


# =================================================================================================
# Normalisation
# =================================================================================================
def test_normalise_turns_every_row_into_a_distribution_over_the_lags() -> None:
    """Each row sums to one, whatever magnitude it arrived at."""
    rows = np.vstack([_one_hot(2, height=7.0), _bump(5), np.full(_N_LAGS, 0.25)])

    shares = lag_hist.normalise(rows)

    assert shares.shape == (3, _N_LAGS)
    assert shares.sum(axis=1) == pytest.approx(1.0)
    # The magnitude divides out: a one-hot at height seven is the same distribution as at height one.
    assert shares[0] == pytest.approx(lag_hist.normalise(_one_hot(2))[0])


def test_normalise_drops_a_non_finite_bin_from_the_total_rather_than_reading_it_as_zero() -> None:
    """The invariant every function in the lag vocabulary holds, and the row that shows it.

    A bin read as zero would leave the *shape* untouched and the *total* too large, so only the
    share of the surviving bins says which convention was used.
    """
    profile = np.full(_N_LAGS, 1.0)
    profile[4] = np.nan

    shares = lag_hist.normalise(profile)[0]

    assert np.isnan(shares[4])
    # Ten surviving bins of equal mass, not eleven.
    assert shares[0] == pytest.approx(1.0 / (_N_LAGS - 1))
    assert np.nansum(shares) == pytest.approx(1.0)


def test_a_row_with_no_mass_normalises_to_nan_rather_than_to_zero() -> None:
    """"Attended nowhere" and "never measured" are different statements."""
    shares = lag_hist.normalise(np.vstack([np.zeros(_N_LAGS), _bump(5)]))

    assert np.isnan(shares[0]).all()
    assert np.isfinite(shares[1]).all()


# =================================================================================================
# The transport distance, and its unit
# =================================================================================================
def test_the_transport_distance_of_a_rigid_shift_is_the_shift_in_seconds() -> None:
    """The known answer that pins the unit: $k$ bins apart is $4k$ s apart, not $k$.

    A version transporting over lag indices would report a quarter of this and would still trend
    correctly across every cohort and window.
    """
    for steps in (1, 3, 7):
        left, right = _one_hot(1), _one_hot(1 + steps)

        moved = lag_hist.wasserstein_seconds(left, right, _SECONDS)

        assert moved == pytest.approx(steps * _SECONDS_PER_STEP)


def test_the_transport_distance_is_symmetric_and_zero_on_identical_profiles() -> None:
    """Zero means agreement, and it is reserved for it."""
    left, right = _bump(3), _bump(7)

    assert lag_hist.wasserstein_seconds(left, left, _SECONDS) == pytest.approx(0.0)
    assert lag_hist.wasserstein_seconds(left, right, _SECONDS) == pytest.approx(
        lag_hist.wasserstein_seconds(right, left, _SECONDS)
    )


def test_the_transport_distance_ignores_magnitude() -> None:
    """It compares distributions, so ten times a profile is the same profile."""
    left, right = _bump(3), _bump(7)

    assert lag_hist.wasserstein_seconds(left, 10.0 * right, _SECONDS) == pytest.approx(
        lag_hist.wasserstein_seconds(left, right, _SECONDS)
    )


# =================================================================================================
# The overlap distance, and how it differs from the transport one
# =================================================================================================
def test_the_overlap_distance_is_bounded_symmetric_and_zero_on_identical_profiles() -> None:
    """Base two, so the ceiling is one and a reader compares against it without the axis width."""
    left, right = _bump(3), _bump(7)

    close = lag_hist.jensen_shannon(left, left)
    apart = lag_hist.jensen_shannon(left, right)

    assert close == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= apart <= 1.0
    assert apart == pytest.approx(lag_hist.jensen_shannon(right, left))


def test_disjoint_supports_saturate_the_overlap_distance_while_the_transport_one_keeps_measuring(
) -> None:
    """Why both travel: past the point the supports separate, only one of them still reports.

    Two one-hot profiles never overlap however close they are, so Jensen--Shannon is at its ceiling
    for a one-step separation and for a nine-step one alike, while Wasserstein reports each.
    """
    near = (_one_hot(1), _one_hot(2))
    far = (_one_hot(1), _one_hot(10))

    assert lag_hist.jensen_shannon(*near) == pytest.approx(1.0)
    assert lag_hist.jensen_shannon(*far) == pytest.approx(1.0)
    assert lag_hist.wasserstein_seconds(*far, _SECONDS) > lag_hist.wasserstein_seconds(
        *near, _SECONDS
    )


# =================================================================================================
# The signed difference, and the refusals
# =================================================================================================
def test_the_centroid_is_taken_on_the_offset_axis() -> None:
    """A centroid built against a zero-based axis would be short by the causal input delay."""
    centre = lag_hist.centroid_seconds(_one_hot(4), _SECONDS)

    assert centre == pytest.approx(_SECONDS[4])
    assert centre != pytest.approx(4 * _SECONDS_PER_STEP)


def test_the_centroid_difference_carries_the_sign_the_two_distances_cannot() -> None:
    """Oriented left minus right, so a positive value means the first cell sits at the longer lag."""
    record = lag_hist.cell_distances(_one_hot(8), _one_hot(2), _SECONDS)

    assert record["centroid_delta_s"] == pytest.approx(6 * _SECONDS_PER_STEP)
    assert record["jensen_shannon"] >= 0.0
    assert record["wasserstein_s"] >= 0.0
    # Reversing the pair flips the sign and leaves both distances alone.
    reversed_record = lag_hist.cell_distances(_one_hot(2), _one_hot(8), _SECONDS)
    assert reversed_record["centroid_delta_s"] == pytest.approx(-record["centroid_delta_s"])
    assert reversed_record["wasserstein_s"] == pytest.approx(record["wasserstein_s"])


@pytest.mark.parametrize(
    "left, right",
    [
        (np.zeros(_N_LAGS), _bump(5)),
        (_bump(5), np.zeros(_N_LAGS)),
        (np.full(_N_LAGS, np.nan), _bump(5)),
        (np.zeros(0), np.zeros(0)),
    ],
    ids=["left empty", "right empty", "left all non-finite", "both empty"],
)
def test_a_comparison_that_cannot_be_made_is_nan_throughout(left, right) -> None:
    """Never zero: zero is the strongest claim of agreement in the column."""
    record = lag_hist.cell_distances(left, right, _SECONDS)

    assert all(np.isnan(value) for value in record.values())


def test_a_length_mismatch_is_refused_rather_than_padded() -> None:
    """A cell of the wrong width is mis-assembled, not short."""
    record = lag_hist.cell_distances(_bump(5), np.ones(_N_LAGS - 2), _SECONDS)

    assert np.isnan(record["jensen_shannon"])
    assert np.isnan(record["wasserstein_s"])


def test_the_quantile_lags_are_the_first_bins_reaching_each_level() -> None:
    """The same first-bin rule the stacked reducer applies, on one cell: a distribution with a
    quarter of its mass in each of four bins reports those bins' lags for the quartiles."""
    seconds = np.arange(6, dtype=np.float64) * 4.0
    profile = np.array([0.0, 0.25, 0.25, 0.25, 0.25, 0.0])

    lags = lag_hist.quantile_seconds(profile, seconds, (0.25, 0.5, 0.75))

    assert lags.tolist() == [4.0, 8.0, 12.0]
    assert np.isnan(lag_hist.quantile_seconds(np.zeros(6), seconds, (0.5,))).all()
    assert np.isnan(lag_hist.quantile_seconds(profile[:-1], seconds, (0.5,))).all()
