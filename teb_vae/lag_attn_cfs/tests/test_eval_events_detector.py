r"""The contraction detector, which is the only detector this package carries.

Its output becomes ``seconds_since_contraction`` on the per-anchor table and is what
contraction-conditioned coupling conditions on, so three of its properties decide whether that
readout means anything:

**The onset is a level crossing, not a gradient test, and that is a correction rather than a
preference.** The inherited walk-back was entered *at* the peak, where the smoothed gradient is
approximately zero and therefore already below any positive threshold, so the loop body never ran
and the reported onset was the peak index itself. Both obvious repairs fail too: a walk conditioned
on $g \ge 0$ stops at the apex whenever the noise put a negative gradient there, and a two-stage
walk that steps off the apex first stops mid-flank, because a smoothed corner's slope passes
continuously through any positive threshold on its way to zero. A crossing of a level set by the
peak's own prominence has neither failure, and the trace below is built so that a gradient walk
would demonstrably stop at the apex.

**Gaps are masked by ``weight``, never by value.** A missing raw sample is stored as $0.0$, which
is an extreme excursion rather than a detectable sentinel. Validity is carried separately, the gap
is interpolated across before smoothing so it contributes no edge, and any event whose span touches
the gap is then dropped -- because its shape was partly invented by that interpolation. Both halves
are needed and each is asserted, since either alone passes for the wrong reason.

**Nothing about decelerations survives the port.** A deceleration detector scores a bpm waveform
and this cell forecasts wavelet coefficients, so it is out of scope -- and a detector kept "in
case" is a detector nothing tests. The scan at the bottom is what makes that a property of the
module rather than of the day it was written.
"""
from __future__ import annotations

import numpy as np
import pytest

from teb_vae.lag_attn_cfs.eval import events

#: The synthetic trace: ten minutes at the raw rate, with contractions three minutes apart.
_DURATION_S = 600.0
_N_SAMPLES = int(_DURATION_S * events.FS_RAW)
_PEAKS_S = (120.0, 300.0, 480.0)
_RISE_S = 40.0
_FALL_S = 40.0
_AMPLITUDE = 25.0
_NOISE = 0.5

#: How far a recovered onset may sit from the injected one. The walk stops where the trace has
#: risen through the last tenth of the event's prominence, so on a linear ramp the recovered onset
#: is later than the injected one by that tenth of the rise -- a stated offset rather than a
#: tolerance absorbing an unknown.
_ONSET_TOLERANCE_S = (1.0 - events.FLANK_LEVEL_FRAC) * _RISE_S + 4.0


def _triangle(length: int, centre: int, rise: int, fall: int) -> np.ndarray:
    """A piecewise-linear bump: zero, up over ``rise``, down over ``fall``, zero.

    Linear rather than Gaussian on purpose: a ramp's onset is a single index rather than a tail
    asymptoting into noise, so "the recovered onset is within N seconds of the injected one" is a
    statement about the detector rather than about where one chooses to call a Gaussian's foot.

    Args:
        length: Samples in the output.
        centre: Index of the apex.
        rise: Samples the leading edge takes to reach the apex.
        fall: Samples the trailing edge takes to return to zero.

    Returns:
        The unit-amplitude bump $(length,)$.
    """
    positions = np.arange(length, dtype=np.float64)
    up = 1.0 - (centre - positions) / float(max(rise, 1))
    down = 1.0 - (positions - centre) / float(max(fall, 1))
    return np.clip(np.minimum(up, down), 0.0, 1.0)


def _injected() -> dict:
    """The known answer: onset and peak indices of every injected contraction."""
    peak = np.round(np.asarray(_PEAKS_S) * events.FS_RAW).astype(np.int64)
    return {"peak": peak, "onset": peak - int(round(_RISE_S * events.FS_RAW))}


def _up_trace(seed: int = 0, *, notch_apex: bool = False) -> np.ndarray:
    """A UP trace carrying the injected contractions in light noise.

    Args:
        seed: Noise seed, so the trace is reproducible.
        notch_apex: Subtract a one-sample dip at each apex, which makes the smoothed gradient
            there negative. A gradient-conditioned walk-back stops on it; a level crossing does
            not notice it at all.

    Returns:
        The trace $(R,)$.
    """
    rng = np.random.default_rng(seed)
    trace = 30.0 + _NOISE * rng.standard_normal(_N_SAMPLES)
    rise = int(round(_RISE_S * events.FS_RAW))
    fall = int(round(_FALL_S * events.FS_RAW))
    for peak in _injected()["peak"]:
        trace += _AMPLITUDE * _triangle(_N_SAMPLES, int(peak), rise, fall)
        if notch_apex:
            trace[int(peak)] -= 2.0
    return trace


# =================================================================================================
# The onset walk
# =================================================================================================
def test_the_injected_contractions_are_recovered_at_their_known_positions() -> None:
    found = events.detect_contractions(_up_trace())

    assert found["peak_raw"].size == len(_PEAKS_S)
    assert np.abs(found["peak_raw"] - _injected()["peak"]).max() / events.FS_RAW <= 2.0


def test_the_onset_is_the_rising_edge_rather_than_the_peak_index() -> None:
    r"""The defect this detector was corrected for, pinned in the form that would catch its return.

    A characterization of the inherited behaviour would read ``onset_raw == peak_raw``; what is
    asserted here is that the onset precedes the peak *and* lands where the ramp actually began,
    which is what makes it a correction rather than a shift.
    """
    found = events.detect_contractions(_up_trace())

    assert (found["onset_raw"] < found["peak_raw"]).all(), "the onset must precede the peak"
    error = (found["onset_raw"] - _injected()["onset"]) / events.FS_RAW
    assert np.abs(error).max() <= _ONSET_TOLERANCE_S, f"onset error {error} s"


def test_an_apex_a_gradient_walk_would_stop_at_does_not_move_the_onset() -> None:
    r"""The construction that separates the two rules.

    Each apex carries a downward notch, so the smoothed gradient there is *negative* -- a walk
    conditioned on "keep going while the trace is rising" halts on the first step and reports the
    peak as the onset. The level crossing is monotone in the thing being measured and cannot be
    ended by one sample, so the recovered onsets are the same ones the clean trace gives.
    """
    clean = events.detect_contractions(_up_trace())
    notched = events.detect_contractions(_up_trace(notch_apex=True))

    assert notched["onset_raw"].size == clean["onset_raw"].size
    assert (notched["peak_raw"] - notched["onset_raw"]).min() > 0.5 * _RISE_S * events.FS_RAW
    assert np.abs(notched["onset_raw"] - clean["onset_raw"]).max() <= 2


def test_the_gradient_at_a_found_peak_is_below_any_positive_threshold() -> None:
    """The premise behind the correction, measured rather than asserted from the story.

    A peak of the smoothed trace has an approximately zero gradient *by definition of being a
    peak*, so a walk-back conditioned on "the trace is still rising steeply" was already false at
    its first step. The flank's own gradient -- the slope the walk was supposed to be following --
    is nearly an order of magnitude larger on this trace, and the ratio is what the assertion
    pins: any threshold chosen to track the flank leaves the apex below it.
    """
    trace = _up_trace()
    smoothed = events._smooth(
        trace, window=int(round(events.CONTRACTION_SMOOTH_S * events.FS_RAW))
    )
    gradient = np.gradient(smoothed)
    found = events.detect_contractions(trace)

    at_peaks = np.abs(gradient[found["peak_raw"]])
    on_flanks = np.abs(gradient[found["peak_raw"] - int(round(20 * events.FS_RAW))])

    assert at_peaks.max() < 0.2 * on_flanks.min()


def test_the_prominence_threshold_is_sigma_relative_and_therefore_unit_free() -> None:
    """UP has no clinically absolute scale here, so a fixed threshold would mean different things
    on two recordings -- and the loader hands this detector a z-scored trace, not a raw one."""
    raw = _up_trace()
    z_scored = (raw - raw.mean()) / raw.std()

    assert (
        events.detect_contractions(z_scored)["peak_raw"].tolist()
        == events.detect_contractions(raw)["peak_raw"].tolist()
    )


def test_a_quiet_trace_carries_no_contractions() -> None:
    """Not vacuous in the other direction: a detector that fires on noise would satisfy every
    recovery test above while making the conditioned readout meaningless."""
    quiet = 30.0 + _NOISE * np.random.default_rng(1).standard_normal(_N_SAMPLES)

    assert events.detect_contractions(quiet)["peak_raw"].size == 0


def test_a_trace_shorter_than_the_detector_can_work_on_returns_no_events() -> None:
    short = _up_trace()[: int(0.5 * events.MIN_CONTRACTION_TRACE_S * events.FS_RAW)]

    found = events.detect_contractions(short)

    assert set(found) == {"onset_raw", "peak_raw", "end_raw"}
    assert all(array.size == 0 for array in found.values())


# =================================================================================================
# Gaps: masked by weight, never by value
# =================================================================================================
def _gapped():
    """The trace with a stored gap over the middle contraction's **rising flank**.

    On the flank rather than over the apex, deliberately: the peak itself survives, so the
    detector still finds the event and the question becomes whether an event whose *span* was
    partly invented is reported. Zeroed over the apex the event would simply cease to exist, and
    the drop rule would never be exercised at all.

    Returns:
        ``(trace, valid, lo, hi)``.
    """
    trace = _up_trace()
    peak = int(_injected()["peak"][1])
    lo, hi = peak - int(round(_RISE_S * events.FS_RAW)), peak - int(round(8 * events.FS_RAW))
    trace[lo:hi] = 0.0
    valid = np.ones(_N_SAMPLES, dtype=bool)
    valid[lo:hi] = False
    return trace, valid, lo, hi


def test_an_event_whose_span_touches_a_gap_is_dropped() -> None:
    """Its shape was partly invented by the interpolation that removed the gap's edges, so it is
    an event about the fill rather than about the recording."""
    trace, valid, lo, hi = _gapped()

    found = events.detect_contractions(trace, valid=valid)["peak_raw"]

    assert not ((found >= lo - int(_RISE_S * events.FS_RAW)) & (found < hi + 60)).any()
    # The two contractions clear of the gap are untouched: the rule drops an event, not a run.
    assert found.size == len(_PEAKS_S) - 1


def test_the_same_event_is_reported_when_nothing_says_the_region_is_invalid() -> None:
    """The other direction, without which the test above would pass on a detector that had simply
    lost its sensitivity: the peak is still there and still found. What removes it is the validity
    mask saying that half its rising edge is a fill, and nothing in the values says so."""
    trace, valid, _lo, _hi = _gapped()

    unmasked = events.detect_contractions(trace, valid=None)["peak_raw"]
    masked = events.detect_contractions(trace, valid=valid)["peak_raw"]

    assert unmasked.size == len(_PEAKS_S)
    assert masked.size == len(_PEAKS_S) - 1
    assert set(masked.tolist()) < set(unmasked.tolist())


def test_the_gap_is_interpolated_across_rather_than_held_or_zeroed() -> None:
    """A straight line across the gap contributes no local extremum. A hold or a sentinel would
    put an edge there for the smoother to turn into a slope and the peak finder into an event."""
    trace, valid, lo, hi = _gapped()

    filled = events.fill_gaps(trace, valid)

    assert filled[lo:hi].min() > 0.0
    assert np.allclose(filled[lo:hi], np.interp(
        np.arange(lo, hi, dtype=np.float64),
        [lo - 1, hi],
        [trace[lo - 1], trace[hi]],
    ), atol=1e-9)
    # Outside the gap nothing moved.
    assert np.array_equal(filled[:lo], trace[:lo])


def test_an_all_valid_or_all_invalid_trace_is_returned_unchanged() -> None:
    """An all-invalid trace has nothing to interpolate *from*, and every event on it is dropped
    anyway; inventing a fill there would be inventing the recording."""
    trace = _up_trace()

    assert np.array_equal(events.fill_gaps(trace, np.ones(_N_SAMPLES, dtype=bool)), trace)
    assert np.array_equal(events.fill_gaps(trace, np.zeros(_N_SAMPLES, dtype=bool)), trace)


# =================================================================================================
# The decimated-to-raw validity expansion
# =================================================================================================
def test_the_decimated_weight_expands_to_one_flag_per_raw_sample() -> None:
    valid = events.raw_validity([1.0, 0.0, 1.0], decimation=4)

    assert valid.tolist() == [True] * 4 + [False] * 4 + [True] * 4


def test_a_partially_covered_step_counts_as_valid_for_detection() -> None:
    """The looser of this repository's two conventions, deliberately: the model's own
    ``VALID_THRESHOLD`` demands a fully covered step, and detecting an event across a partially
    covered one is a weaker claim than scoring a forecast on it."""
    assert events.raw_validity([0.5], decimation=2).all()
    assert not events.raw_validity([0.0], decimation=2).any()


def test_a_validity_mask_may_be_truncated_but_never_padded() -> None:
    """Padding would mark uncovered samples valid, which is the one direction a validity mask must
    never fail in."""
    assert events.raw_validity([1.0, 1.0], decimation=4, raw_len=6).size == 6

    with pytest.raises(ValueError, match="uncovered samples valid"):
        events.raw_validity([1.0], decimation=4, raw_len=8)


# =================================================================================================
# What did not come across
# =================================================================================================
def test_no_symbol_relating_to_deceleration_detection_survives() -> None:
    """An attribute scan rather than an import check: the constants are what a copied analysis
    would reach for first, and a module that still exported them would let one back in."""
    surviving = sorted(
        name for name in vars(events)
        if "deceleration" in name.lower() or "bpm" in name.lower()
    )

    assert surviving == []
    assert not hasattr(events, "detect_decelerations")


def test_the_helpers_reachable_only_from_that_detector_are_gone_too() -> None:
    """``usable_interval`` and ``usable_horizon_steps`` fixed a horizon step so an anchor-level
    event rate was a rate per event; ``match_events`` paired a branch's detections against the
    truth's. All three exist for deceleration forecast skill, which this package does not have."""
    for name in ("usable_interval", "usable_horizon_steps", "match_events"):
        assert not hasattr(events, name), name


def test_the_module_exports_exactly_what_the_contraction_path_needs() -> None:
    """``__all__`` is the seam the collection pass reads through, and a name left in it that the
    module no longer defines is an ``ImportError`` at the worst moment of a multi-hour run."""
    assert set(events.__all__) == {
        "CONTRACTION_EDGE_S",
        "CONTRACTION_MIN_DISTANCE_S",
        "CONTRACTION_MIN_WIDTH_S",
        "CONTRACTION_PROMINENCE_SIGMA",
        "CONTRACTION_SMOOTH_S",
        "CONTRACTION_WALK_S",
        "FLANK_LEVEL_FRAC",
        "FS_RAW",
        "MIN_CONTRACTION_TRACE_S",
        "ONSET_WALK_BACK_NOTE",
        "ScipyRequired",
        "detect_contractions",
        "drop_events_overlapping_gaps",
        "fill_gaps",
        "raw_validity",
    }
    for name in events.__all__:
        assert hasattr(events, name), name


def test_the_divergence_from_the_siblings_published_numbers_travels_in_the_output() -> None:
    """The sibling's stage-2 probe is a published negative result calibrated against the
    uncorrected walk-back, so this module states the difference rather than leaving two sets of
    onsets to be compared silently."""
    assert "walk-back" in events.ONSET_WALK_BACK_NOTE
    assert "peak index as the onset" in events.ONSET_WALK_BACK_NOTE
