r"""Physiological event detection on the raw traces: contractions and decelerations.

The raw forecast target is what makes this pipeline able to ask a clinical question at all. An
$H \cdot R$-sample block is $4H$ seconds of fetal heart rate; it can be inverted to bpm, searched for
decelerations, and lined up against the uterine activity that preceded it. No earlier evaluation
in this repository could do any of that, because no earlier model forecast a raw signal.

Two detectors, both operating on the $4\,$Hz grid:

* :func:`detect_contractions` finds uterine-contraction events on the UP trace. Its prominence
  threshold is **$\sigma$-relative**, so it is unit-free and applies equally to the loader's
  z-scored UP and to a raw one. That is a property of the detector rather than a convenience: UP
  has no clinically absolute scale here, and a fixed threshold would mean different things on two
  recordings.
* :func:`detect_decelerations` finds downward dips on the FHR trace, and requires **bpm**. That
  is the deliberate change from the source these two are ported from.

**Why bpm, and what changed.** The detectors come from the sibling package's stage-2 probe
(``teb_vae/lag_attn/eval/stage2_contraction_deceleration_probe.py``), which is itself a verbatim
vendoring of the causal-TE validation suite's ``events.py``. There, the deceleration prominence
was chosen at run time by a heuristic -- ``max(|x|) > 30`` was taken to mean "this looks like
bpm", and anything else fell back to $0.3\,\sigma$. Handed a z-scored trace the detector therefore
switched, silently, to calling $0.3\,\sigma$ wiggles decelerations. Here the contract is stated
instead: the input **is** bpm, the threshold is
$\max(\mathrm{prominence\_bpm},\, \sigma\text{-relative})$ unconditionally, and a caller that has
not inverted the loader's normalisation gets no detections rather than a plausible number. The
conversion itself is :func:`~teb_vae.lag_attn_rws.eval.metrics.to_bpm`, which is the repository's
one supported z-to-bpm path.

**Gaps are masked by ``weight``, never by value.** A missing sample is stored as $0.0$ bpm, which
after z-scoring is roughly $-11\sigma$: it is not a detectable sentinel, it is the deepest
deceleration in the recording. So validity is carried separately, as a boolean array derived from
the decimated ``weight`` by :func:`raw_validity`, and it is used twice --

1. the invalid samples are **linearly interpolated across** before smoothing, so a gap contributes
   no edge for the peak finder to lock onto; and
2. any event whose span touches an invalid sample is **dropped**, because an event straddling a
   gap is an event whose shape was invented by step 1.

Both are necessary. Interpolating alone would report events built out of the interpolation;
dropping alone would leave the raw discontinuity in the smoothed trace and manufacture events on
either side of it.

**One divergence from the sibling is deliberate and is named here.**
:func:`detect_contractions`'s rising-edge walk-back is **fixed** in this module and is *not* fixed
in the sibling's copy. The original entered the walk-back loop at the peak, where the smoothed
gradient is approximately zero and therefore already below the positive-gradient threshold, so the
loop body never ran and the reported "onset" was the peak index itself. The sibling's stage-2
probe is a published negative result whose lag window and self-test tolerances were calibrated
against that behaviour, so correcting it there would move numbers that have already been read;
this module states the divergence instead. See :func:`detect_contractions` for the corrected
walk-back and :data:`ONSET_WALK_BACK_NOTE` for the sentence a run carries into its own output.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: Raw sampling rate of both traces, in Hz. The decimated grid is $16$ raw samples per $4\,$s step.
FS_RAW = 4.0

#: Contraction detector settings, in seconds except the prominence.
CONTRACTION_SMOOTH_S = 10.0
CONTRACTION_MIN_DISTANCE_S = 60.0
CONTRACTION_MIN_WIDTH_S = 20.0
CONTRACTION_PROMINENCE_SIGMA = 0.5
CONTRACTION_EDGE_S = 30.0
#: How far back and forward the onset and end walks may travel from the peak.
CONTRACTION_WALK_S = 60.0

#: Deceleration detector settings. The prominence is the **larger** of an absolute bpm threshold
#: and a $\sigma$-relative one, unconditionally -- see the module docstring.
DECELERATION_SMOOTH_S = 8.0
DECELERATION_MIN_DISTANCE_S = 15.0
DECELERATION_PROMINENCE_BPM = 10.0
DECELERATION_PROMINENCE_SIGMA = 0.3
DECELERATION_EDGE_S = 30.0
DECELERATION_WALK_S = 60.0

#: How far down an event's own prominence its onset and end sit. At $0.9$ the onset is the point
#: where the event has risen through the last tenth of its prominence -- unambiguously started,
#: and far enough above the baseline that noise cannot place it. It is a level rather than a
#: gradient because a gradient test cannot be made to work at either end: see :func:`_flank`.
FLANK_LEVEL_FRAC = 0.9

#: Shortest trace either detector will look at, in seconds. Below it ``find_peaks`` has nothing to
#: work with and the edge rule has already removed the whole trace; both return no events, which
#: is why a caller reports the length rather than the empty result.
MIN_CONTRACTION_TRACE_S = 60.0
MIN_DECELERATION_TRACE_S = 30.0

#: The sentence a run carries beside every contraction onset it reports, so the divergence from
#: the sibling's published stage-2 numbers is legible in the artifacts rather than only here.
ONSET_WALK_BACK_NOTE = (
    "contraction onsets are the corrected rising-edge walk-back; the sibling stage-2 probe's "
    "walk-back starts at the peak, where the smoothed gradient is already below threshold, and "
    "therefore reports the peak index as the onset"
)


class ScipyRequired(RuntimeError):
    """``scipy`` is needed for peak finding and is not importable.

    Raised rather than returning no events. "No events" is a measurement, and a detector that
    reports it because a dependency is missing produces a hit rate of zero that reads exactly like
    a model that forecasts nothing.
    """


def _require_scipy() -> Tuple[Any, Any]:
    """Return ``(find_peaks, savgol_filter)``, raising when scipy is absent.

    Imported at the call site rather than at module import, following this package's convention:
    ``scipy`` is the one extra dependency the evaluation has, and importing it should not be the
    price of importing a constant.

    Returns:
        The two scipy callables.

    Raises:
        ScipyRequired: If scipy cannot be imported.
    """
    try:
        from scipy.signal import find_peaks, savgol_filter
    except Exception as exc:  # noqa: BLE001 - re-raised as this module's own type
        raise ScipyRequired(
            "event detection needs scipy.signal.find_peaks and savgol_filter. Install scipy, or "
            "skip the event analyses; silently reporting zero events would be indistinguishable "
            "from a recording in which nothing happened."
        ) from exc
    return find_peaks, savgol_filter


# =============================================================================
# Validity, gaps, and the shared signal preparation
# =============================================================================
def raw_validity(
    weight: Any, *, decimation: int, raw_len: Optional[int] = None, threshold: float = 0.0
) -> np.ndarray:
    r"""Expand a decimated ``weight`` row into a per-raw-sample validity mask.

    Args:
        weight: The decimated validity $(T,)$, as the loader stores it -- fractional at partially
            covered steps and exactly $0$ where the recording is missing.
        decimation: Raw samples per decimated step $D$.
        raw_len: Length to produce. Defaults to $T \cdot D$; a shorter value truncates and a
            longer one is refused, because a validity mask silently padded with ``True`` marks a
            gap as usable.
        threshold: A step counts as valid when its weight is **strictly above** this. Zero by
            default, which is the looser of the two conventions in this repository: the model's
            own ``VALID_THRESHOLD = 1.0`` demands a fully covered step, and detecting an event
            across a partially covered one is a weaker claim than scoring a forecast on it.

    Returns:
        A boolean array of length ``raw_len``.

    Raises:
        ValueError: If ``raw_len`` exceeds what the weight covers.
    """
    values = np.asarray(weight, dtype=np.float64).reshape(-1)
    expanded = np.repeat(values > float(threshold), int(decimation))
    if raw_len is None:
        return expanded
    if int(raw_len) > expanded.size:
        raise ValueError(
            f"weight covers {expanded.size} raw sample(s) but {int(raw_len)} were asked for; "
            f"padding the difference would mark uncovered samples valid, which is the one "
            f"direction a validity mask must never fail in"
        )
    return expanded[: int(raw_len)]


def fill_gaps(signal: Any, valid: Optional[Any]) -> np.ndarray:
    """Linearly interpolate a trace across its invalid samples.

    Interpolation rather than a sentinel or a hold: a gap is stored as $0.0$ bpm, and both a step
    down to zero and a step to a held value are edges the smoother turns into a slope and the peak
    finder turns into an event. A straight line across the gap contributes no local extremum, so
    the only events that survive nearby are ones the surrounding real signal supports -- and even
    those are dropped by :func:`drop_events_overlapping_gaps`, because their shape partly came
    from this line.

    Args:
        signal: The trace $(R,)$.
        valid: Boolean validity of the same length, or ``None`` for "all valid".

    Returns:
        The filled trace as ``float64``. Returned unchanged when nothing is invalid, and returned
        as-is when *nothing* is valid -- an all-invalid trace has nothing to interpolate from, and
        every event on it is dropped anyway.
    """
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if valid is None:
        return array
    mask = np.asarray(valid, dtype=bool).reshape(-1)[: array.size]
    if mask.size < array.size:
        mask = np.concatenate([mask, np.zeros(array.size - mask.size, dtype=bool)])
    if mask.all() or not mask.any():
        return array
    filled = array.copy()
    positions = np.arange(array.size, dtype=np.float64)
    filled[~mask] = np.interp(positions[~mask], positions[mask], array[mask])
    return filled


def drop_events_overlapping_gaps(
    events: Dict[str, np.ndarray], valid: Optional[Any], *, span_keys: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Remove every event whose span touches an invalid sample.

    Args:
        events: A detector's output, all arrays the same length.
        valid: Boolean per-raw-sample validity, or ``None`` for "all valid".
        span_keys: The two keys bounding an event, as ``(first, last)``.

    Returns:
        The same mapping with the offending events removed from every array at once, so the
        arrays stay parallel.
    """
    first_key, last_key = span_keys
    if valid is None or not len(events.get(first_key, ())):
        return events
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    # Cumulative count of invalid samples, so "does [lo, hi] contain one" is two lookups rather
    # than a slice per event.
    invalid_before = np.concatenate([[0], np.cumsum(~mask)])
    lo = np.clip(np.asarray(events[first_key], dtype=np.int64), 0, mask.size - 1)
    hi = np.clip(np.asarray(events[last_key], dtype=np.int64), 0, mask.size - 1)
    keep = (invalid_before[hi + 1] - invalid_before[lo]) == 0
    return {name: np.asarray(values)[keep] for name, values in events.items()}


def _smooth(x: np.ndarray, *, window: int, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing, with the window coerced odd and clipped to the trace.

    Args:
        x: The trace $(R,)$.
        window: Smoothing window in samples.
        poly: Polynomial order.

    Returns:
        The smoothed trace, or the input unchanged when it is too short to filter.
    """
    _, savgol_filter = _require_scipy()
    array = np.asarray(x, dtype=np.float64)
    if array.size < max(window, poly + 2):
        return array
    width = int(window)
    if width % 2 == 0:
        width += 1
    width = max(width, poly + 2 + (1 if (poly + 2) % 2 == 0 else 0))
    if width >= array.size:
        return array
    return np.asarray(savgol_filter(array, width, poly), dtype=np.float64)


def _to_1d(x: Any) -> np.ndarray:
    """Coerce an arbitrary-shape array to 1-D by averaging any leading axes."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1]).mean(axis=0)


def _empty(*names: str) -> Dict[str, np.ndarray]:
    """An empty detector result carrying the named index arrays."""
    return {name: np.empty(0, dtype=np.int64) for name in names}


# =============================================================================
# Contractions
# =============================================================================
def detect_contractions(
    up_raw: Any,
    *,
    fs: float = FS_RAW,
    valid: Optional[Any] = None,
    smooth_seconds: float = CONTRACTION_SMOOTH_S,
    min_distance_seconds: float = CONTRACTION_MIN_DISTANCE_S,
    min_width_seconds: float = CONTRACTION_MIN_WIDTH_S,
    prominence_sigma: float = CONTRACTION_PROMINENCE_SIGMA,
    edge_seconds: float = CONTRACTION_EDGE_S,
) -> Dict[str, np.ndarray]:
    r"""Detect uterine-contraction events on the UP trace.

    The trace is smoothed, local maxima are found at ``distance``, ``width`` and
    $\sigma$-relative ``prominence``, and each event's rising-edge onset and falling-edge end are
    recovered by walking out from the peak along the smoothed gradient.

    **The walk-back is the corrected one.** The original walked back from the peak while
    $g \ge g^\star$, where $g^\star$ was the $80$th percentile of the trace's positive gradients
    -- "steeply rising". At the peak the smoothed gradient is approximately *zero*, which is below
    any positive $g^\star$, so the loop exited before its first step and the reported onset was
    the peak index. Both obvious repairs fail too, and both were measured on a fixture with a
    known answer: a walk conditioned on $g \ge 0$ stops at the apex whenever the noise put a
    negative gradient there, and a two-stage walk that steps off the apex first stops mid-flank,
    because a smoothed corner's slope passes continuously through any positive threshold on its
    way to zero. :func:`_flank` replaces the gradient test with a **level crossing** of the peak's
    own prominence, which has neither failure. The global percentile goes with it: a threshold
    calibrated on the whole trace's gradient distribution lands inside the flank on any recording
    whose contractions make up a large share of that distribution.

    Args:
        up_raw: The UP trace $(R,)$ at ``fs`` Hz, in any units -- the prominence is
            $\sigma$-relative, so the loader's z-scoring changes nothing.
        fs: Sampling rate in Hz.
        valid: Per-raw-sample validity, from :func:`raw_validity`, or ``None``.
        smooth_seconds: Savitzky-Golay window.
        min_distance_seconds: Minimum spacing between peaks.
        min_width_seconds: Minimum peak width at half-prominence.
        prominence_sigma: Required prominence, in units of $\sigma$ of the smoothed trace.
        edge_seconds: Drop peaks within this many seconds of either end.

    Returns:
        ``{'onset_raw', 'peak_raw', 'end_raw'}`` of ``int64`` raw-sample indices, parallel and
        possibly empty.

    Raises:
        ScipyRequired: If scipy is unavailable.
    """
    find_peaks, _ = _require_scipy()
    signal = _to_1d(up_raw)
    n = int(signal.size)
    empty = _empty("onset_raw", "peak_raw", "end_raw")
    if n < int(MIN_CONTRACTION_TRACE_S * fs):
        return empty

    smooth = _smooth(fill_gaps(signal, valid), window=max(int(round(smooth_seconds * fs)), 5))
    sigma = float(np.nanstd(smooth)) or 1.0
    peaks, properties = find_peaks(
        smooth,
        distance=max(int(round(min_distance_seconds * fs)), 1),
        prominence=float(prominence_sigma) * sigma,
        width=max(int(round(min_width_seconds * fs)), 1),
    )
    peaks = np.asarray(peaks, dtype=np.int64)
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    edge = int(round(edge_seconds * fs))
    kept = (peaks >= edge) & (peaks < (n - edge))
    peaks, prominences = peaks[kept], prominences[kept]
    if peaks.size == 0:
        return empty

    onsets, ends = _event_flanks(smooth, peaks, prominences, walk_seconds=CONTRACTION_WALK_S, fs=fs)
    return drop_events_overlapping_gaps(
        {"onset_raw": onsets, "peak_raw": peaks, "end_raw": ends},
        valid,
        span_keys=("onset_raw", "end_raw"),
    )


def _event_flanks(
    values: np.ndarray,
    peaks: np.ndarray,
    prominences: np.ndarray,
    *,
    walk_seconds: float,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return each peak's leading and trailing flank feet.

    Args:
        values: The smoothed trace, inverted when the extrema are dips.
        peaks: The peak indices.
        prominences: ``find_peaks``' prominence per peak, which sets each event's own level.
        walk_seconds: How far either walk may travel from the peak.
        fs: Sampling rate in Hz.

    Returns:
        ``(onsets, ends)``, parallel to ``peaks``.
    """
    walk = int(round(float(walk_seconds) * float(fs)))
    last = int(values.size) - 1
    onsets = np.empty_like(peaks)
    ends = np.empty_like(peaks)
    for position, peak in enumerate(peaks):
        level = float(values[int(peak)]) - FLANK_LEVEL_FRAC * float(prominences[position])
        onsets[position] = _flank(
            values, peak=int(peak), stop=max(0, int(peak) - walk), level=level, backwards=True
        )
        ends[position] = _flank(
            values, peak=int(peak), stop=min(last, int(peak) + walk), level=level,
            backwards=False,
        )
    return onsets, ends


def _flank(values: np.ndarray, *, peak: int, stop: int, level: float, backwards: bool) -> int:
    r"""Walk out from a peak to where the trace has fallen through ``level``, and return there.

    **A level crossing, not a gradient test, and that is the correction.** The original walked
    back while the gradient stayed above a threshold, which is unusable on either side of the
    obvious choices: above zero it stops at the apex, where the smoothed gradient is
    approximately zero and its sign is whichever way the noise fell; above a positive threshold
    it stops mid-flank, because a smoothed corner's slope passes continuously through that
    threshold on the way to zero. Both answers are "the onset is roughly the peak", which is the
    defect. A crossing of a level defined by the peak's own prominence has neither failure: it
    is monotone in the thing being measured, a single noisy sample cannot end it, and it is
    guaranteed to land strictly before the peak whenever the prominence is positive.

    Args:
        values: The smoothed trace, already inverted by the caller when the extremum is a dip.
        peak: The peak's index.
        stop: How far the walk may travel; also the answer when the level is never crossed.
        level: The value the flank is walked out to.
        backwards: Walk towards lower indices -- the leading flank -- rather than higher.

    Returns:
        The first index at or past ``level``. Bounded by ``stop``, which is a visibly degenerate
        span rather than a plausible onset when the walk runs its full length.
    """
    step = -1 if backwards else 1
    index = int(peak)
    while (index > stop if backwards else index < stop) and values[index] > level:
        index += step
    return index


# =============================================================================
# Decelerations
# =============================================================================
def detect_decelerations(
    fhr_bpm: Any,
    *,
    fs: float = FS_RAW,
    valid: Optional[Any] = None,
    smooth_seconds: float = DECELERATION_SMOOTH_S,
    min_distance_seconds: float = DECELERATION_MIN_DISTANCE_S,
    prominence_bpm: float = DECELERATION_PROMINENCE_BPM,
    prominence_sigma: float = DECELERATION_PROMINENCE_SIGMA,
    edge_seconds: float = DECELERATION_EDGE_S,
) -> Dict[str, np.ndarray]:
    r"""Detect fetal-heart-rate decelerations -- downward dips -- on an FHR trace **in bpm**.

    Decelerations are local minima, so the smoothed trace is inverted and the same peak finder is
    used. The prominence is
    $\max(\mathrm{prominence\_bpm},\; \mathrm{prominence\_sigma} \cdot \sigma)$ with no unit
    heuristic: a $10\,$bpm floor is what makes the detection clinical rather than relative, and it
    is also what makes a caller that forgot to invert the loader's z-scoring get nothing instead
    of getting noise.

    The onset and end walks are :func:`detect_contractions`' walks applied to the inverted trace,
    for which a dip is a peak. This detector never had the walk-back defect: its condition was
    already a plain sign test, which the nadir's own approximately-zero gradient satisfies.

    Args:
        fhr_bpm: The FHR trace $(R,)$ at ``fs`` Hz, in bpm.
        fs: Sampling rate in Hz.
        valid: Per-raw-sample validity, from :func:`raw_validity`, or ``None``.
        smooth_seconds: Savitzky-Golay window.
        min_distance_seconds: Minimum spacing between nadirs.
        prominence_bpm: Absolute prominence floor, in bpm.
        prominence_sigma: Prominence in units of $\sigma$ of the smoothed trace.
        edge_seconds: Drop nadirs within this many seconds of either end.

    Returns:
        ``{'onset_raw', 'nadir_raw', 'end_raw'}`` of ``int64`` raw-sample indices, parallel and
        possibly empty.

    Raises:
        ScipyRequired: If scipy is unavailable.
    """
    find_peaks, _ = _require_scipy()
    signal = _to_1d(fhr_bpm)
    n = int(signal.size)
    empty = _empty("onset_raw", "nadir_raw", "end_raw")
    if n < int(MIN_DECELERATION_TRACE_S * fs):
        return empty

    smooth = _smooth(fill_gaps(signal, valid), window=max(int(round(smooth_seconds * fs)), 5))
    sigma = float(np.nanstd(smooth)) or 1.0
    # On the *inverted* trace, so a dip is a peak and every step after this one -- the prominence,
    # the flank walks -- is the contraction detector's, unchanged.
    inverted = -smooth
    nadirs, properties = find_peaks(
        inverted,
        distance=max(int(round(min_distance_seconds * fs)), 1),
        prominence=max(float(prominence_bpm), float(prominence_sigma) * sigma),
    )
    nadirs = np.asarray(nadirs, dtype=np.int64)
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    edge = int(round(edge_seconds * fs))
    kept = (nadirs >= edge) & (nadirs < (n - edge))
    nadirs, prominences = nadirs[kept], prominences[kept]
    if nadirs.size == 0:
        return empty

    onsets, ends = _event_flanks(
        inverted, nadirs, prominences, walk_seconds=DECELERATION_WALK_S, fs=fs
    )
    return drop_events_overlapping_gaps(
        {"onset_raw": onsets, "nadir_raw": nadirs, "end_raw": ends},
        valid,
        span_keys=("onset_raw", "end_raw"),
    )


# =============================================================================
# Block mode: what an H*R-sample forecast can and cannot be searched for
# =============================================================================
def usable_interval(
    n_samples: int, *, fs: float = FS_RAW, edge_seconds: float = DECELERATION_EDGE_S
) -> Tuple[int, int]:
    r"""Return the half-open sample interval a detection may land in, inside one block.

    The detector drops events within ``edge_seconds`` of either end, so a block of $HR$
    samples ($4H$ s) leaves an interior two ``edge_seconds`` shorter than the forecast. That
    is a property of the ported detector rather than of the model, and it is what every rate
    computed on blocks is a rate *over*, so it is computed here and reported rather than being
    left implicit.

    Args:
        n_samples: Samples in the block.
        fs: Sampling rate in Hz.
        edge_seconds: The detector's edge rule.

    Returns:
        ``(lo, hi)``. Empty -- ``hi <= lo`` -- when the block is shorter than twice the edge, in
        which case no rate can be computed on blocks of that length at all.
    """
    edge = int(round(float(edge_seconds) * float(fs)))
    return edge, int(n_samples) - edge


def usable_horizon_steps(
    n_samples: int,
    *,
    raw_per_step: int,
    fs: float = FS_RAW,
    edge_seconds: float = DECELERATION_EDGE_S,
) -> np.ndarray:
    r"""Return the horizon steps lying wholly inside a block's usable interval.

    Fixing the horizon step $\tau$ is what removes the pseudo-replication that makes an
    anchor-level event rate meaningless. Consecutive anchors' forecast windows overlap in $H - 1$
    of their $H$ steps, so one physiological deceleration is re-detected once per anchor; but for
    a *fixed* $\tau$ there is exactly one anchor whose block places a given absolute raw sample at
    that step. A rate computed per $\tau$ is therefore a rate per event by construction rather
    than by a de-duplication rule applied afterwards.

    Args:
        n_samples: Samples in the block, $H \cdot R$.
        raw_per_step: Raw samples per horizon step $R$.
        fs: Sampling rate in Hz.
        edge_seconds: The detector's edge rule.

    Returns:
        The usable $\tau$ values, ascending; empty when the interval admits no whole step.
    """
    lo, hi = usable_interval(n_samples, fs=fs, edge_seconds=edge_seconds)
    if hi <= lo or int(raw_per_step) < 1:
        return np.empty(0, dtype=np.int64)
    first = int(np.ceil(lo / float(raw_per_step)))
    last = int(np.floor(hi / float(raw_per_step)))
    return np.arange(max(first, 0), max(last, 0), dtype=np.int64)


def match_events(
    left: Any, right: Any, *, tolerance: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily pair two sorted index sets within a tolerance.

    Greedy nearest-first rather than optimal assignment, and the two agree under a stated
    condition: detections from one detector run are at least its ``min_distance`` apart, so two of
    them can both be within ``tolerance`` of the same reference only if
    $2\\,\\mathrm{tolerance} > \\mathrm{min\\_distance}$. Callers keep the tolerance below half the
    distance, which makes at most one candidate ever in range. Stated rather than assumed, because
    a tolerance raised past that bound makes this pass order-dependent silently.

    Args:
        left: Reference indices, e.g. the truth's.
        right: Candidate indices, e.g. a branch's.
        tolerance: Largest absolute index difference that counts as the same event.

    Returns:
        ``(matched_left, matched_right)`` boolean masks, one per input, marking the paired
        entries.
    """
    a = np.asarray(list(left), dtype=np.int64)
    b = np.asarray(list(right), dtype=np.int64)
    matched_left = np.zeros(a.size, dtype=bool)
    matched_right = np.zeros(b.size, dtype=bool)
    if a.size == 0 or b.size == 0:
        return matched_left, matched_right
    for index, value in enumerate(a):
        distance = np.abs(b - value)
        distance[matched_right] = np.iinfo(np.int64).max
        best = int(np.argmin(distance))
        if distance[best] <= int(tolerance):
            matched_left[index] = True
            matched_right[best] = True
    return matched_left, matched_right


__all__ = [
    "CONTRACTION_EDGE_S",
    "CONTRACTION_MIN_DISTANCE_S",
    "CONTRACTION_MIN_WIDTH_S",
    "CONTRACTION_PROMINENCE_SIGMA",
    "CONTRACTION_SMOOTH_S",
    "DECELERATION_EDGE_S",
    "DECELERATION_MIN_DISTANCE_S",
    "DECELERATION_PROMINENCE_BPM",
    "DECELERATION_PROMINENCE_SIGMA",
    "DECELERATION_SMOOTH_S",
    "FLANK_LEVEL_FRAC",
    "FS_RAW",
    "ONSET_WALK_BACK_NOTE",
    "ScipyRequired",
    "detect_contractions",
    "detect_decelerations",
    "drop_events_overlapping_gaps",
    "fill_gaps",
    "match_events",
    "raw_validity",
    "usable_horizon_steps",
    "usable_interval",
]
