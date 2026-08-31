r"""The shape of a lag profile, reduced to scalars: the equality that justifies one implementation.

Two things are pinned here, and the first is the reason this module exists at all.

**The vectorised reducer and the scalar helpers are the same arithmetic.** ``lag_kl`` describes a
pooled profile one at a time through :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.peak_width`,
:func:`~teb_vae.lag_attn_cfs.eval.lag_shape.mass_above` and
:func:`~teb_vae.lag_attn_cfs.eval.lag_shape.degeneracy`; ``lag_clocks`` describes every segment at
once through :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.profile_statistics`. The second is written
for a matrix rather than looping over the first, so nothing in the code makes them agree -- and if
they drift, two artifacts of one run report two different peaks and two different degeneracy
verdicts while both look entirely ordinary. They are asserted equal row by row, over profiles
chosen to hit every branch each of them has: bimodal, one-hot, flat, sparse, with a missing bin,
empty, and negative.

**Every statistic is a known answer.** A centroid can be off by a bin, an entropy can be computed
over the unnormalised profile, a quantile can land one bin early, a width can count non-contiguous
bins -- and each wrong version still produces a smooth, plausible trajectory. Each is asserted
against a profile whose answer is arithmetic a reader can check in their head.

The axis these run against is deliberately **offset**: $\tau_\ell = 4(\ell + \delta)$ with
$\delta > 0$, because the compensated axis never starts at zero and three of the statistics -- the
two mass shares and, through them, any reading of "near the anchor" -- would be wrong in a way no
zero-based fixture could show.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from teb_vae.lag_attn_cfs.eval import lag_shape
from teb_vae.lag_attn_cfs.eval.analyses import lag_kl

#: A short axis, and one that starts where a real one does: eleven bins at 4 s with a causal input
#: delay of three steps, so $\tau_0 = 12$ s rather than $0$.
_N_LAGS = 11
_DELAY_STEPS = 3
_SECONDS = 4.0 * (np.arange(_N_LAGS, dtype=np.float64) + _DELAY_STEPS)


def _profiles() -> List[np.ndarray]:
    """Every branch the reducer and the scalar helpers have, one row each.

    Returns:
        The rows, in the order the assertions below name them: bimodal, one-hot at the shortest
        lag, one-hot at the longest, flat, sparse, carrying a non-finite bin, all-zero, and
        carrying a negative bin.
    """
    return [
        np.array([0.0, 0.6, 1.0, 0.6, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0] + [0.0] * (_N_LAGS - 1)),
        np.array([0.0] * (_N_LAGS - 1) + [1.0]),
        np.full(_N_LAGS, 0.2),
        np.array([0.0] * (_N_LAGS - 1) + [3.0]),
        np.where(np.arange(_N_LAGS) == 5, np.nan, np.linspace(0.1, 1.0, _N_LAGS)),
        np.zeros(_N_LAGS),
        np.array([1.0, -0.5] + [0.1] * (_N_LAGS - 2)),
    ]


@pytest.fixture(scope="module")
def reduced():
    """The whole fixture stack through the reducer, once."""
    return lag_shape.profile_statistics(np.vstack(_profiles()), _SECONDS)


# =================================================================================================
# One implementation, two shapes of caller
# =================================================================================================
def test_the_vectorised_reducer_agrees_with_the_scalar_helpers_row_by_row(reduced) -> None:
    """The equality that makes moving the peak helpers a de-duplication rather than a fork.

    Only the rows the reducer calls usable are compared: it reports a row with no mass or a
    negative bin as ``NaN`` throughout, deliberately, while the scalar helpers -- which describe
    whatever they are handed -- still return a number for one. That divergence is the population
    rule, not a disagreement about arithmetic, and it is asserted separately below.
    """
    statistics, _ = reduced
    for index, profile in enumerate(_profiles()):
        if not np.isfinite(statistics["centroid"][index]):
            continue
        peak = lag_kl.peak_width(profile.tolist())
        concentration = lag_kl.mass_above(profile.tolist())
        verdict = lag_kl.degeneracy(profile.tolist())
        assert statistics["peak"][index] == pytest.approx(_SECONDS[peak["argmax"]]), index
        assert statistics["peak_width"][index] == pytest.approx(
            peak["width_bins"] * lag_shape.SECONDS_PER_LAG_STEP
        ), index
        assert statistics["peak_mass"][index] == pytest.approx(concentration["share"]), index
        assert bool(statistics["peak_degenerate"][index]) is verdict["degenerate"], index
        assert statistics["zero_fraction"][index] == pytest.approx(
            verdict["zero_fraction"]
        ), index
        # The scale-carrying pair, and the second assertion is the one that pins the name
        # collision: ``peak_width``'s "peak" is the peak's VALUE while the reducer's "peak" is its
        # POSITION in seconds. Until ``peak_nats`` existed there was nothing to compare the value
        # against, so the two meanings of one word were never checked against each other.
        assert statistics["total_nats"][index] == pytest.approx(
            profile[np.isfinite(profile)].sum()
        ), index
        assert statistics["peak_nats"][index] == pytest.approx(peak["peak"]), index
        assert statistics["peak"][index] != pytest.approx(statistics["peak_nats"][index]) or (
            _SECONDS[peak["argmax"]] == peak["peak"]
        ), index


def test_the_scale_carrying_pair_is_what_the_other_twelve_divide_out() -> None:
    r"""Two profiles of identical shape and tenfold different magnitude.

    Every statistic formed from $p_\ell = w_\ell / \sum_k w_k$ must agree between them -- that is
    the convention, and it is the right one for a question about *where* the mass sits. It is the
    wrong one for a trajectory: a window whose coupling collapsed and a window whose coupling
    merely moved would reduce identically. ``total_nats`` and ``peak_nats`` are what separate them,
    and this test is the statement that they and only they carry the scale.
    """
    small = np.array([[0.1, 0.9, 0.2, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    large = small * 10.0

    reduced_small, _ = lag_shape.profile_statistics(small, _SECONDS)
    reduced_large, _ = lag_shape.profile_statistics(large, _SECONDS)

    scale_free = [key for key in lag_shape.STATISTIC_KEYS if not key.endswith("_nats")]
    for key in scale_free:
        assert reduced_small[key][0] == pytest.approx(
            reduced_large[key][0], nan_ok=True
        ), key

    assert reduced_large["total_nats"][0] == pytest.approx(
        10.0 * reduced_small["total_nats"][0]
    )
    assert reduced_large["peak_nats"][0] == pytest.approx(10.0 * reduced_small["peak_nats"][0])


def test_a_row_the_reducer_refuses_is_one_the_scalar_helpers_would_still_describe(reduced) -> None:
    """The two unusable rows are ``NaN`` throughout, and the record says which was which."""
    statistics, record = reduced
    empty, negative = 6, 7
    for index in (empty, negative):
        for key in lag_shape.STATISTIC_KEYS:
            assert np.isnan(statistics[key][index]), (key, index)
    assert record == {"n_rows": 8, "n_usable": 6, "n_empty": 1, "n_negative": 1}
    # The scalar helper, handed the same all-zero row, still describes it -- which is why the
    # comparison above skips these rows rather than expecting them to agree.
    assert lag_kl.peak_width(_profiles()[empty].tolist())["argmax"] == 0


# =================================================================================================
# Known answers, one statistic at a time
# =================================================================================================
def test_the_moments_are_the_hand_computed_answer() -> None:
    """A two-bin profile splitting its mass three to one against the arithmetic that defines it."""
    profile = np.zeros(_N_LAGS)
    profile[2], profile[6] = 3.0, 1.0
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    shares = np.array([0.75, 0.25])
    lags = _SECONDS[[2, 6]]
    centroid = float(shares @ lags)
    spread = float(np.sqrt(shares @ (lags - centroid) ** 2))
    assert statistics["centroid"][0] == pytest.approx(centroid)
    assert statistics["spread"][0] == pytest.approx(spread)
    # Positive: the light quarter sits at the longer lag, so the tail runs away from the anchor.
    assert statistics["skewness"][0] > 0.0
    assert statistics["skewness"][0] == pytest.approx(
        float(shares @ (lags - centroid) ** 3) / spread ** 3
    )


def test_the_centroid_is_on_the_compensated_axis_rather_than_on_the_bin_index() -> None:
    """The offset is the failure this catches: a bin index reads as a lag of zero at bin zero."""
    profile = np.zeros(_N_LAGS)
    profile[0] = 1.0
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    assert statistics["centroid"][0] == pytest.approx(4.0 * _DELAY_STEPS)
    assert statistics["centroid"][0] != pytest.approx(0.0)


def test_the_quantiles_are_read_off_the_cumulative_mass() -> None:
    """A profile with a quarter of its mass in each of four bins has its quartiles at those bins."""
    profile = np.zeros(_N_LAGS)
    profile[[1, 3, 5, 9]] = 1.0
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    # Cumulative mass reaches 0.25 at bin 1, 0.5 at bin 3, 0.75 at bin 5.
    assert statistics["median"][0] == pytest.approx(_SECONDS[3])
    assert statistics["iqr"][0] == pytest.approx(_SECONDS[5] - _SECONDS[1])


def test_the_median_resists_a_far_bin_that_drags_the_centroid() -> None:
    """The reason a robust centre is reported beside the centroid at all."""
    profile = np.zeros(_N_LAGS)
    profile[1] = 0.9
    profile[10] = 0.1
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    assert statistics["median"][0] == pytest.approx(_SECONDS[1])
    assert statistics["centroid"][0] > statistics["median"][0]


def test_the_entropy_is_over_the_normalised_profile_and_survives_exact_zeros() -> None:
    """``entmax15`` emits exact zeros, so $p\\log p$ must read as its limit rather than warn."""
    one_hot = np.zeros(_N_LAGS)
    one_hot[4] = 7.0
    flat = np.full(_N_LAGS, 0.2)
    statistics, _ = lag_shape.profile_statistics(np.vstack([one_hot, flat]), _SECONDS)
    assert statistics["entropy"][0] == pytest.approx(0.0)
    # Zero rather than negative zero: the second reads in a CSV as though something had gone wrong.
    assert not np.signbit(statistics["entropy"][0])
    assert statistics["entropy"][1] == pytest.approx(np.log(_N_LAGS))
    # And the entropy is scale-free: the unnormalised one-hot is 7.0, not 1.0.
    assert statistics["effective_support"][0] == pytest.approx(lag_shape.SECONDS_PER_LAG_STEP)
    assert statistics["effective_support"][1] == pytest.approx(
        _N_LAGS * lag_shape.SECONDS_PER_LAG_STEP
    )


def test_the_mass_shares_are_measured_from_the_axis_start_rather_than_from_zero() -> None:
    """The delay-invariance that makes the two shares comparable across runs.

    The same profile on an axis with a larger causal delay must report the same shares. On an
    absolute threshold it would not: every bin would move further from zero and the near share
    would shrink toward nothing for a reason that has nothing to do with the model.
    """
    # A forty-bin axis, so the near window is a real cut: bins 0-15 sit within 60 s of the start.
    width = 40
    axis = 4.0 * (np.arange(width, dtype=np.float64) + _DELAY_STEPS)
    profile = np.zeros(width)
    profile[[0, 1, 30]] = [2.0, 1.0, 1.0]
    near = lag_shape.profile_statistics(profile[None, :], axis)[0]["near_mass"][0]
    assert near == pytest.approx(0.75)
    shifted = 4.0 * (np.arange(width, dtype=np.float64) + 100)
    assert lag_shape.profile_statistics(profile[None, :], shifted)[0]["near_mass"][0] == (
        pytest.approx(near)
    )


def test_the_far_share_starts_where_the_near_share_ends_and_they_do_not_overlap() -> None:
    """Stated as a property of the constants, so moving one cannot make a profile count twice."""
    assert lag_shape.FAR_SECONDS > lag_shape.NEAR_SECONDS
    wide = 4.0 * (np.arange(200, dtype=np.float64) + _DELAY_STEPS)
    profile = np.ones(200)
    statistics, _ = lag_shape.profile_statistics(profile[None, :], wide)
    assert statistics["near_mass"][0] + statistics["far_mass"][0] <= 1.0


def test_the_peak_width_counts_the_contiguous_run_and_not_every_tall_bin() -> None:
    """A bimodal profile is two peaks, not one very wide one -- the whole point of contiguity."""
    profile = np.zeros(_N_LAGS)
    profile[[1, 2, 3, 5]] = [0.6, 1.0, 0.6, 0.9]
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    assert statistics["peak_width"][0] == pytest.approx(3 * lag_shape.SECONDS_PER_LAG_STEP)
    # Bin 5 stands above half the peak and is excluded from the width but not from the mass.
    assert statistics["peak_mass"][0] == pytest.approx(3.1 / 3.1)


def test_a_missing_bin_breaks_the_peaks_run_rather_than_extending_it() -> None:
    """A lag never measured is not a lag the source attended to at half height."""
    profile = np.array([0.0, 0.6, 1.0, np.nan, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    statistics, _ = lag_shape.profile_statistics(profile[None, :], _SECONDS)
    assert statistics["peak_width"][0] == pytest.approx(2 * lag_shape.SECONDS_PER_LAG_STEP)


def test_a_missing_bin_is_dropped_from_the_mass_rather_than_read_as_zero() -> None:
    """Both are a bin with no attribution in it, and they mean different things."""
    with_gap = np.array([1.0, np.nan, 1.0] + [0.0] * (_N_LAGS - 3))
    as_zero = np.array([1.0, 0.0, 1.0] + [0.0] * (_N_LAGS - 3))
    statistics, _ = lag_shape.profile_statistics(np.vstack([with_gap, as_zero]), _SECONDS)
    # The centroid agrees -- the gap carried no mass either way -- but the zero fraction cannot:
    # one profile has nine measured empty bins and the other has ten.
    assert statistics["centroid"][0] == pytest.approx(statistics["centroid"][1])
    assert statistics["zero_fraction"][0] == pytest.approx(8 / 10)
    assert statistics["zero_fraction"][1] == pytest.approx(9 / 11)


# =================================================================================================
# The guard
# =================================================================================================
@pytest.mark.parametrize(
    "profile, degenerate, why",
    [
        (np.full(_N_LAGS, 0.2), True, "flat: the peak is not distinguishable from the bulk"),
        (
            np.array([0.0] * 8 + [1.0, 1.0, 1.0]), False,
            f"sparse at {8 / _N_LAGS:.0%}, which is below the threshold",
        ),
        (
            np.array([0.0] * (_N_LAGS - 1) + [1.0]), True,
            f"sparse at {(_N_LAGS - 1) / _N_LAGS:.0%}, which is just above it",
        ),
        (np.array([0.0] * 19 + [1.0]), True, "sparse at 95%, which is well above it"),
        (
            np.array([0.05, 0.1, 0.4, 1.0, 0.4, 0.1, 0.05, 0.02, 0.01]), False,
            "an ordinary peaked profile",
        ),
    ],
)
def test_the_degeneracy_flag_fires_on_the_two_shapes_that_have_no_readable_peak(
    profile: np.ndarray, degenerate: bool, why: str
) -> None:
    """Named case by case, so a threshold moved by one changes a named row rather than a count."""
    seconds = 4.0 * (np.arange(profile.size, dtype=np.float64) + _DELAY_STEPS)
    statistics, _ = lag_shape.profile_statistics(profile[None, :], seconds)
    assert bool(statistics["peak_degenerate"][0]) is degenerate, why
    assert lag_kl.degeneracy(profile.tolist())["degenerate"] is degenerate, why


# =================================================================================================
# The refusals
# =================================================================================================
def test_a_profile_of_the_wrong_width_is_refused_rather_than_reshaped() -> None:
    """A vector of another length is mis-assembled, and a plausible wrong answer is the danger."""
    statistics, record = lag_shape.profile_statistics(np.ones((3, _N_LAGS - 2)), _SECONDS)
    for key in lag_shape.STATISTIC_KEYS:
        assert np.isnan(statistics[key]).all(), key
    assert record["n_rows"] == 3 and record["n_usable"] == 0
    assert "mis-assembled" in record["note"]


@pytest.mark.parametrize(
    "rows, seconds",
    [
        (np.zeros((0, _N_LAGS)), _SECONDS),
        (np.zeros(_N_LAGS), _SECONDS),
        (np.zeros((2, 0)), np.zeros(0)),
    ],
    ids=["no rows", "not two-dimensional", "no axis"],
)
def test_an_input_with_nothing_to_reduce_returns_the_full_mapping_and_a_zeroed_record(
    rows: np.ndarray, seconds: np.ndarray
) -> None:
    """Every key present, so a caller building columns from the mapping cannot raise on an edge."""
    statistics, record = lag_shape.profile_statistics(rows, seconds)
    assert set(statistics) == set(lag_shape.STATISTIC_KEYS)
    assert record["n_usable"] == 0


# =================================================================================================
# Restricting a profile, and reading a signed one
# =================================================================================================
def test_a_band_restriction_carries_its_axis_with_it() -> None:
    """The profile and the seconds axis are cut together, because a statistic in seconds taken on
    a restricted profile against the full axis would be a centroid of the wrong window."""
    rows = np.arange(2 * _N_LAGS, dtype=np.float64).reshape(2, _N_LAGS)

    sliced, axis = lag_shape.restrict_to_band(rows, _SECONDS, (3, 6))

    assert sliced.shape == (2, 4)
    assert list(axis) == list(_SECONDS[3:7])
    assert list(sliced[0]) == list(rows[0, 3:7])


@pytest.mark.parametrize(
    "band, fragment",
    [
        ((6, 3), "empty"),
        ((-1, 4), "reaches past"),
        ((3, _N_LAGS), "reaches past"),
    ],
)
def test_a_band_that_names_more_than_it_measures_is_refused(band, fragment: str) -> None:
    """Clipped rather than refused, a band would be reported under a name that overstates the lags
    it covered; empty, it would report a row of zeros as a finding. Both are the refusals
    ``_validate_occlusion_bands`` already makes at config load, restated here because this function
    is also reachable from a band nobody validated."""
    with pytest.raises(ValueError, match=fragment):
        lag_shape.restrict_to_band(np.zeros((1, _N_LAGS)), _SECONDS, band)


def test_the_band_mass_is_one_definition_the_planted_gate_now_binds() -> None:
    """Known-answer, and then the delegation.

    The gate had its own copy of this arithmetic while it was the only consumer. The clock-excess
    band shares report the same quantity from an analysis now, so the copy is gone and the gate
    calls this -- which is what keeps a share quoted in a recovery record and a share quoted in a
    report from being two implementations that merely agree today.
    """
    from teb_vae.lag_attn_cfs import lag_recovery_check

    profile = [5.0, 1.0, 0.5, 0.6, 0.7, 0.6, 0.5, 0.2, 0.1, 0.1, 0.1]

    record = lag_shape.band_mass(profile, (3, 6))

    # 0.6 + 0.7 + 0.6 + 0.5 = 2.4 of a 9.4 total.
    assert record["n_bins"] == 4
    assert record["nats"] == pytest.approx(2.4)
    assert record["share"] == pytest.approx(2.4 / 9.4)
    # And the gate reports that number rather than one of its own.
    assert lag_recovery_check.band_share(profile, (3, 6)) == pytest.approx(2.4 / 9.4)


def test_a_band_of_an_empty_profile_reports_nan_rather_than_a_zero_share() -> None:
    """Zero would read as a band that was measured and found empty. Nothing was measured."""
    record = lag_shape.band_mass([0.0] * _N_LAGS, (3, 6))

    assert np.isnan(record["share"])
    assert record["nats"] == 0.0


def test_rectifying_a_signed_profile_reports_what_it_discarded() -> None:
    r"""$\Delta^+_\ell = \max(\Delta_\ell, 0)$, and the census that keeps it honest.

    The rectified total is an **upper bound** on $\sum_\ell \Delta_\ell$, never a partition of
    it, so a reader quoting it as ``coupling_minus_clock`` would overstate by exactly
    ``negative_nats``. Both totals are therefore returned, and ``rectified_frac`` says how large
    the gap is relative to what survived.
    """
    positive, record = lag_shape.rectified_profile(np.array([[-0.2, 0.5, -0.3, 1.0]]))

    assert list(positive[0]) == [0.0, 0.5, 0.0, 1.0]
    assert record["positive_nats"] == pytest.approx(1.5)
    assert record["negative_nats"] == pytest.approx(-0.5)
    # The signed sum survives rectification as a reported number even though the vector does not.
    assert record["net_nats"] == pytest.approx(1.0)
    assert record["rectified_frac"] == pytest.approx(0.5 / 1.5)
    assert record["n_rows_with_negative_bin"] == 1


def test_the_reducer_still_refuses_the_signed_profile_it_was_always_going_to_refuse() -> None:
    """The load-bearing pair, asserted together in one test because the whole design rests on the
    relationship between them: ``profile_statistics`` rejects a negative bin outright -- which is
    the precondition for its entropy, its quantiles and its mass shares, not a defensive check --
    and rectification is what makes a signed clock-excess profile reducible **without** touching
    that refusal. If a later change relaxed the reducer instead, this test is what fails."""
    signed = np.array([[-0.2, 0.5, -0.3, 1.0]])
    axis = np.arange(4, dtype=np.float64) * 4.0

    _, refused = lag_shape.profile_statistics(signed, axis)
    assert refused["n_negative"] == 1
    assert refused["n_usable"] == 0

    positive, _ = lag_shape.rectified_profile(signed)
    statistics, accepted = lag_shape.profile_statistics(positive, axis)
    assert accepted["n_negative"] == 0
    assert accepted["n_usable"] == 1
    # And the centroid it now yields is the mass-weighted mean of the surviving bins.
    assert statistics["centroid"][0] == pytest.approx((0.5 * 4.0 + 1.0 * 12.0) / 1.5)


def test_every_key_the_module_advertises_is_one_the_reducer_returns() -> None:
    """The tuple consumers lay their columns out from, pinned to what actually arrives."""
    statistics, _ = lag_shape.profile_statistics(np.ones((2, _N_LAGS)), _SECONDS)
    assert tuple(sorted(statistics)) == tuple(sorted(lag_shape.STATISTIC_KEYS))
    assert len(lag_shape.STATISTIC_KEYS) == len(set(lag_shape.STATISTIC_KEYS))


def test_the_peak_vocabulary_lag_kl_exposes_is_this_modules_own_object() -> None:
    """Compared by identity: a correct local copy would still be a second definition."""
    assert lag_kl.peak_width is lag_shape.peak_width
    assert lag_kl.mass_above is lag_shape.mass_above
    assert lag_kl.secondary_peaks is lag_shape.secondary_peaks
    assert lag_kl.degeneracy is lag_shape.degeneracy
    assert lag_kl.DEGENERATE_PEAK_TO_MEDIAN == lag_shape.DEGENERATE_PEAK_TO_MEDIAN
    assert lag_kl.DEGENERATE_ZERO_FRACTION == lag_shape.DEGENERATE_ZERO_FRACTION
