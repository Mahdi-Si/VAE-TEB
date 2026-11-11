r"""Session-7 event detectors: a UA reference, improved detectors, and a causal UA activity.

Pure ``numpy``/``scipy``. Every function takes one $4\,$Hz trace and a boolean validity of the
same length. Thresholds are in **stored units** (toco units for UA, bpm for FHR); a z-scored UA
input needs ``scale`` $= \sigma_u$.

Contents:

* :func:`reference_contractions`: the detector-free reference used to audit the shipped detector.
* :func:`detect_contractions_v7`: a drop-in replacement of
  ``teb_vae.lag_attn_cfs.eval.events.detect_contractions`` (same keys ``onset_raw``,
  ``peak_raw``, ``end_raw``) with absolute thresholds, a $120\,$s walk and a tolerant validity
  rule. In the training weight it is called as
  ``functools.partial(detect_contractions_v7, scale=sigma_up)``, because the batch's ``up`` is
  z-scored by the loader.
* :func:`ua_activity`: a causal per-step activity in $[0, 1]$ that can replace the binary onset
  window as the reward weight.
* :func:`detect_decelerations_v7`: a NICHD deceleration detector (depth $\ge 15\,$bpm below a
  robust baseline for $\ge 15\,$s), same keys as the shipped ``detect_decelerations``
  (``onset_raw``, ``nadir_raw``, ``end_raw``, ``depth_bpm``) plus ``prolonged``.

**The UA reference.** The shipped contraction detector sets its prominence threshold in units of
the segment's own $\sigma$, so what counts as a contraction depends on what else the segment
holds. The reference does not. It builds a band-limited UA trace $u$, a robust baseline $b$ (a
rolling low quantile), the rise $r = u - b$, and calls a contraction any local maximum of $u$ with

$$
r(t_p) \ge \theta, \qquad \operatorname{prom}(t_p) \ge \theta / 2,
$$

in absolute toco units. $\theta$ was set on recordings of split code 0 only: a two-component
Gaussian mixture fitted to $\log_{10} r(t_p)$ of every local maximum (prominence $\ge 2$ units,
spacing $\ge 30\,$s) gives a low component at 9.7 units and a high component at 39.9 units, and
the posterior of the high component crosses $0.5$ at $r = 20$ units. The onset is the start of
the sustained rise: the walk back from the peak stops where the trace stops falling (a local
minimum, with a small tolerance) or where $r$ drops to $10\%$ of $r(t_p)$, whichever comes first.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, medfilt

#: Sampling rate of the stored traces in Hz.
FS = 4.0
#: Reference rise threshold $\theta$ in toco units (code-0 mixture crossing).
REF_THETA = 20.0
#: Lowpass corner of the band-limited UA, in Hz.
REF_LOWPASS_HZ = 0.04
#: Median prefilter span in seconds (removes spikes before the lowpass).
REF_MEDIAN_S = 4.0
#: Window of the rolling low quantile, in seconds, and its quantile.
REF_BASE_WINDOW_S = 240.0
REF_BASE_Q = 10.0
#: Smoothing of the baseline, in seconds.
REF_BASE_SMOOTH_S = 60.0
#: Minimum spacing of reference peaks, in seconds.
REF_MIN_DISTANCE_S = 30.0
#: Minimum time above half of the rise, in seconds.
REF_MIN_HALF_WIDTH_S = 15.0
#: Fraction of the rise at which the onset and end walks stop.
REF_FOOT_FRAC = 0.1
#: Slope tolerance of the local-minimum stop, in toco units per sample.
REF_SLOPE_TOL = 0.01
#: Longest walk from the peak, in seconds.
REF_WALK_S = 120.0


def fill_invalid(x: np.ndarray, valid: Optional[np.ndarray]) -> np.ndarray:
    """Linearly interpolate a trace across its invalid samples.

    Args:
        x: The trace.
        valid: Boolean validity or ``None``.

    Returns:
        A ``float64`` copy with invalid samples interpolated; unchanged when all or none are valid.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    if valid is None:
        return x
    v = np.asarray(valid, dtype=bool)
    if v.all() or not v.any():
        return x
    pos = np.arange(x.size)
    x[~v] = np.interp(pos[~v], pos[v], x[v])
    return x


def band_limited(x: np.ndarray, valid: Optional[np.ndarray], *, fs: float = FS,
                 lowpass_hz: float = REF_LOWPASS_HZ, median_s: float = REF_MEDIAN_S) -> np.ndarray:
    r"""Zero-phase band-limited trace: gap fill, running median, 4th-order Butterworth lowpass.

    Args:
        x: The trace.
        valid: Its validity.
        fs: Sampling rate in Hz.
        lowpass_hz: Lowpass corner in Hz.
        median_s: Running-median span in seconds.

    Returns:
        The band-limited trace.
    """
    y = fill_invalid(x, valid)
    k = int(round(median_s * fs)) | 1
    y = medfilt(y, k)
    b, a = butter(4, lowpass_hz / (fs / 2.0), btype="low")
    return filtfilt(b, a, y, padtype="odd", padlen=min(200, y.size - 1))


def rolling_baseline(u: np.ndarray, *, fs: float = FS, window_s: float = REF_BASE_WINDOW_S,
                     q: float = REF_BASE_Q, smooth_s: float = REF_BASE_SMOOTH_S) -> np.ndarray:
    r"""Centred rolling low quantile of a band-limited trace, smoothed, on the $f_s$ grid.

    Computed at $1\,$Hz and interpolated back, for speed.

    Args:
        u: The band-limited trace.
        fs: Sampling rate in Hz.
        window_s: Quantile window in seconds.
        q: Percentile in $[0, 100]$.
        smooth_s: Moving-average span of the baseline in seconds.

    Returns:
        The baseline $b$, same length as ``u``.
    """
    step = max(int(round(fs)), 1)
    d = u[::step]
    base = percentile_filter(d, q, size=int(window_s) | 1, mode="nearest")
    base = uniform_filter1d(base, int(smooth_s) | 1, mode="nearest")
    return np.interp(np.arange(u.size) / step, np.arange(d.size), base)


def _foot(u: np.ndarray, r: np.ndarray, peak: int, level: float, stop: int, step: int,
          tol: float) -> int:
    """Walk from ``peak`` in direction ``step`` while the trace keeps falling and stays above ``level``."""
    i = peak
    while i != stop:
        j = i + step
        if r[j] <= level or u[j] > u[i] + tol:
            return j if r[j] <= level else i
        i = j
    return i


def reference_contractions(up: np.ndarray, valid: Optional[np.ndarray], *, fs: float = FS,
                           theta: float = REF_THETA) -> Dict[str, np.ndarray]:
    r"""Detector-free reference contractions on a UA trace in toco units.

    Args:
        up: The UA trace in stored units.
        valid: Its validity; invalid samples are interpolated before filtering.
        fs: Sampling rate in Hz.
        theta: Absolute rise threshold $\theta$ in toco units.

    Returns:
        ``{'onset_raw', 'peak_raw', 'end_raw', 'rise', 'invalid_in_span'}``; ``rise`` is
        $r(t_p)$ and ``invalid_in_span`` the count of invalid samples in $[\mathrm{onset},
        \mathrm{end}]$.
    """
    u = band_limited(up, valid, fs=fs)
    b = rolling_baseline(u, fs=fs)
    r = u - b
    peaks, props = find_peaks(u, prominence=theta / 2.0, distance=int(REF_MIN_DISTANCE_S * fs))
    peaks = peaks[r[peaks] >= theta]
    walk = int(REF_WALK_S * fs)
    on, en, keep = [], [], []
    for p in peaks.tolist():
        level = REF_FOOT_FRAC * r[p]
        o = _foot(u, r, p, level, max(0, p - walk), -1, REF_SLOPE_TOL)
        e = _foot(u, r, p, level, min(u.size - 1, p + walk), 1, REF_SLOPE_TOL)
        half = r[o:e + 1] >= 0.5 * r[p]
        if half.sum() < REF_MIN_HALF_WIDTH_S * fs:
            continue
        on.append(o)
        en.append(e)
        keep.append(p)
    peaks = np.asarray(keep, dtype=np.int64)
    on = np.asarray(on, dtype=np.int64)
    en = np.asarray(en, dtype=np.int64)
    if valid is None:
        inv = np.zeros(peaks.size, dtype=np.int64)
    else:
        cum = np.concatenate([[0], np.cumsum(~np.asarray(valid, dtype=bool))])
        inv = cum[en + 1] - cum[on] if peaks.size else np.zeros(0, dtype=np.int64)
    return {"onset_raw": on, "peak_raw": peaks, "end_raw": en, "rise": r[peaks] if peaks.size else np.zeros(0),
            "invalid_in_span": inv}


def ua_rise(up: np.ndarray, valid: Optional[np.ndarray], *, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    """The band-limited UA and its baseline, for plotting.

    Args:
        up: The UA trace.
        valid: Its validity.
        fs: Sampling rate.

    Returns:
        ``(u, b)``.
    """
    u = band_limited(up, valid, fs=fs)
    return u, rolling_baseline(u, fs=fs)


# =============================================================================
# Improved contraction detector (v7)
# =============================================================================
#: v7 settings. Thresholds are in toco units of the STORED trace; see ``scale``.
V7_SMOOTH_S = 10.0
V7_MIN_DISTANCE_S = 60.0
V7_MIN_WIDTH_S = 15.0
V7_PROMINENCE = 8.0
V7_RISE = 15.0
V7_EDGE_S = 30.0
V7_WALK_S = 120.0
V7_FLANK_LEVEL_FRAC = 0.9
V7_ONSET_MIN_RISE_FRAC = 0.5
#: Validity tolerance: the longest invalid run allowed inside the rise (onset to peak), the
#: largest invalid share of the whole span, and the half-width of the window around the onset
#: that must be fully valid. All in seconds except the share.
V7_MAX_RISE_GAP_S = 8.0
V7_MAX_SPAN_INVALID_SHARE = 0.25
V7_ONSET_GUARD_S = 4.0


def _savgol(x: np.ndarray, window: int) -> np.ndarray:
    """Savitzky-Golay smoothing of order 3 with an odd window, as the shipped detector smooths."""
    from scipy.signal import savgol_filter

    w = int(window) | 1
    if w >= x.size:
        return x
    return savgol_filter(x, w, 3)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array."""
    if not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return int((idx[1::2] - idx[::2]).max())


def detect_contractions_v7(
    up_raw: np.ndarray,
    *,
    fs: float = FS,
    valid: Optional[np.ndarray] = None,
    scale: float = 1.0,
    prominence: float = V7_PROMINENCE,
    rise: float = V7_RISE,
    min_width_seconds: float = V7_MIN_WIDTH_S,
    min_distance_seconds: float = V7_MIN_DISTANCE_S,
    edge_seconds: float = V7_EDGE_S,
    walk_seconds: float = V7_WALK_S,
    tolerant_validity: bool = True,
) -> Dict[str, np.ndarray]:
    r"""Contraction detector with absolute thresholds and a tolerant validity rule.

    It keeps the shipped detector's core (gap fill, Savitzky-Golay smoothing over $10\,$s,
    ``find_peaks``, the onset and end at the crossing of $90\%$ of the peak's own prominence, and
    the onset-rise rule) and changes five things:

    1. **Absolute thresholds.** A peak needs a prominence of at least ``prominence`` toco units
       AND a rise of at least ``rise`` units above a rolling baseline $b$ (the $10$th percentile
       over $240\,$s), instead of $0.5\sigma$ of the segment. The decision then no longer depends
       on what else the segment holds.
    2. **Walk bound $120\,$s** instead of $60\,$s, so a slow rise or a long relaxation is not cut.
    3. **Width $15\,$s** at half prominence instead of $20\,$s.
    4. **Tolerant validity.** An event is dropped only when its onset neighbourhood
       ($\pm 4\,$s) holds an invalid sample, its rise (onset to peak) holds an invalid run longer
       than $8\,$s, or more than $25\%$ of its span is invalid. A short flag on the apex no longer
       removes a contraction. ``tolerant_validity=False`` restores the shipped rule (any invalid
       sample in the span).
    5. **Units.** The thresholds are in toco units of the stored trace. The training batch holds
       the loader-normalised trace $z = (u - \mu)/\sigma_u$, so the caller must pass
       ``scale`` $= \sigma_u$ (the ``up`` std of the statistics file the loader used).

    Args:
        up_raw: The UA trace $(R,)$.
        fs: Sampling rate in Hz.
        valid: Per-sample validity, or ``None``.
        scale: Toco units per input unit ($1$ for stored units, $\sigma_u$ for z-scored input).
        prominence: Minimum prominence in toco units.
        rise: Minimum rise above the rolling baseline at the peak, in toco units.
        min_width_seconds: Minimum width at half prominence.
        min_distance_seconds: Minimum peak spacing.
        edge_seconds: Peaks this close to either end are dropped.
        walk_seconds: Longest onset or end walk from the peak.
        tolerant_validity: Use the tolerant rule (default) or the shipped strict rule.

    Returns:
        ``{'onset_raw', 'peak_raw', 'end_raw'}``, ``int64`` raw indices, parallel, possibly empty;
        the keys of the shipped detector, so it can be swapped in.
    """
    x = np.asarray(up_raw, dtype=np.float64).reshape(-1) * float(scale)
    n = x.size
    empty = {k: np.empty(0, dtype=np.int64) for k in ("onset_raw", "peak_raw", "end_raw")}
    if n < int(60 * fs):
        return empty
    v = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)[:n]
    if not v.any():
        return empty
    smooth = _savgol(fill_invalid(x, v), max(int(round(V7_SMOOTH_S * fs)), 5))
    base = rolling_baseline(smooth, fs=fs)
    # ``find_peaks`` applies ``distance`` before ``width``: in a $60\,$s neighbourhood it keeps
    # the highest sample, so a narrow apex spike can win and then fail the width test. Testing
    # width first and spacing by prominence afterwards was measured on code 0 and gained only
    # 0.4 recall points, so the shipped order is kept.
    peaks, props = find_peaks(smooth, distance=max(int(round(min_distance_seconds * fs)), 1),
                              prominence=float(prominence),
                              width=max(int(round(min_width_seconds * fs)), 1))
    proms = props["prominences"]
    edge = int(round(edge_seconds * fs))
    keep = (peaks >= edge) & (peaks < n - edge) & ((smooth[peaks] - base[peaks]) >= float(rise))
    peaks, proms = peaks[keep], proms[keep]
    if peaks.size == 0:
        return empty
    walk = int(round(walk_seconds * fs))
    on = np.empty_like(peaks)
    en = np.empty_like(peaks)
    for k, p in enumerate(peaks.tolist()):
        level = smooth[p] - V7_FLANK_LEVEL_FRAC * proms[k]
        i = p
        while i > max(0, p - walk) and smooth[i] > level:
            i -= 1
        on[k] = i
        j = p
        while j < min(n - 1, p + walk) and smooth[j] > level:
            j += 1
        en[k] = j
    rising = smooth[peaks] - smooth[on] >= V7_ONSET_MIN_RISE_FRAC * proms
    peaks, on, en = peaks[rising], on[rising], en[rising]
    inv = ~v
    cum = np.concatenate([[0], np.cumsum(inv)])
    ok = np.ones(peaks.size, dtype=bool)
    g = int(round(V7_ONSET_GUARD_S * fs))
    for k in range(peaks.size):
        o, p, e = int(on[k]), int(peaks[k]), int(en[k])
        if not tolerant_validity:
            ok[k] = cum[e + 1] - cum[o] == 0
        elif cum[min(n, o + g + 1)] - cum[max(0, o - g)] > 0:
            ok[k] = False
        elif _longest_run(inv[o:p + 1]) > V7_MAX_RISE_GAP_S * fs:
            ok[k] = False
        elif (cum[e + 1] - cum[o]) > V7_MAX_SPAN_INVALID_SHARE * (e - o + 1):
            ok[k] = False
    return {"onset_raw": on[ok].astype(np.int64), "peak_raw": peaks[ok].astype(np.int64),
            "end_raw": en[ok].astype(np.int64)}


# =============================================================================
# Causal UA activity
# =============================================================================
#: Activity mapping $a = \sigma((r - \theta_a)/s_a)$, with $r$ the causal rise in toco units.
ACT_THETA = 15.0
ACT_SLOPE = 3.0
ACT_LOWPASS_HZ = 0.04
ACT_BASE_WINDOW_S = 240


def ua_activity(up: np.ndarray, valid: Optional[np.ndarray], *, fs: float = FS, scale: float = 1.0,
                step: int = 16, theta: float = ACT_THETA, slope: float = ACT_SLOPE) -> np.ndarray:
    r"""Causal per-step UA activity in $[0, 1]$, a detector-free alternative to the onset window.

    Every operation uses only samples up to the current one:

    1. an invalid sample holds the last valid value (interpolation would read ahead);
    2. a trailing running median over $4\,$s, then a 2nd-order Butterworth lowpass at
       $0.04\,$Hz applied forward only;
    3. a trailing baseline $b_t$, the $10$th percentile of the lowpassed trace over the past
       $240\,$s at $1\,$Hz (a shorter window near the segment start);
    4. the rise $r_t = u_t - b_t$ and the activity
       $$a_t = \frac{1}{1 + \exp\!\big(-(r_t - \theta_a)/s_a\big)},$$
       with $\theta_a = 15$ and $s_a = 3$ toco units;
    5. step $k$ takes the value at its last raw sample $16k + 15$, so it reads nothing after its
       own step.

    Args:
        up: UA trace $(R,)$.
        valid: Its validity, or ``None``.
        fs: Sampling rate in Hz.
        scale: Toco units per input unit (see :func:`detect_contractions_v7`).
        step: Raw samples per step.
        theta: $\theta_a$ in toco units.
        slope: $s_a$ in toco units.

    Returns:
        The activity per step, shape $(\lfloor R / \mathrm{step} \rfloor,)$.
    """
    from scipy.signal import lfilter, lfilter_zi

    x = np.asarray(up, dtype=np.float64).reshape(-1) * float(scale)
    n = x.size
    v = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)[:n]
    if v.any():
        idx = np.maximum.accumulate(np.where(v, np.arange(n), -1))
        idx[idx < 0] = int(np.argmax(v))
        x = x[idx]
    k = int(round(4 * fs))
    padded = np.concatenate([np.full(k - 1, x[0]), x])
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)
    b, a = butter(2, ACT_LOWPASS_HZ / (fs / 2.0), btype="low")
    u = lfilter(b, a, med, zi=lfilter_zi(b, a) * med[0])[0]
    per = int(fs)
    d = u[per - 1::per]
    base1 = np.empty(d.size)
    for t in range(d.size):
        base1[t] = np.percentile(d[max(0, t - ACT_BASE_WINDOW_S + 1):t + 1], 10)
    # Raw sample r reads the baseline of the last completed second at or before it.
    sec = (np.arange(n) - (per - 1)) // per
    base = np.where(sec >= 0, base1[np.clip(sec, 0, d.size - 1)], u[0])
    r = u - base
    act = 1.0 / (1.0 + np.exp(-(r - theta) / slope))
    return act[step - 1::step][: n // step]


# =============================================================================
# Improved deceleration detector (v7)
# =============================================================================
#: NICHD thresholds of the v7 deceleration detector.
DEC_DEPTH_BPM = 15.0
DEC_MIN_S = 15.0
DEC_RETURN_BPM = 5.0
DEC_MERGE_S = 5.0
DEC_WALK_S = 120.0
DEC_EDGE_S = 30.0


def fhr_baseline(fhr: np.ndarray, valid: Optional[np.ndarray], *, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    r"""NICHD-style FHR baseline and a $1\,$Hz-smoothed trace, both on the input grid.

    The trace is averaged to $1\,$Hz over valid samples and smoothed over $5\,$s. The baseline
    is a centred $10\,$min rolling median, recomputed twice after excluding samples more than
    $15\,$bpm away from the current estimate, so accelerations and decelerations do not pull it.

    Args:
        fhr: FHR in bpm.
        valid: Its validity.
        fs: Sampling rate in Hz (an integer number of samples per second).

    Returns:
        ``(smooth, baseline)`` at ``fs``; gaps are interpolated.
    """
    import pandas as pd

    per = int(fs)
    n = fhr.size - fhr.size % per
    v = np.ones(fhr.size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    x = np.where(v[:n], fhr[:n], np.nan).reshape(-1, per)
    with np.errstate(all="ignore"):
        x1 = np.nanmean(x, axis=1)
    sm = pd.Series(x1).rolling(5, center=True, min_periods=1).mean()
    b = sm.rolling(600, center=True, min_periods=60).median()
    for _ in range(2):
        b = sm.where((sm - b).abs() <= 15).rolling(600, center=True, min_periods=60).median()
    grid = np.arange(fhr.size) / fs
    t1 = np.arange(x1.size) + (per - 1) / (2.0 * per)
    out = []
    for s in (sm.to_numpy(), b.to_numpy()):
        ok = np.isfinite(s)
        out.append(np.interp(grid, t1[ok], s[ok]) if ok.any() else np.full(fhr.size, np.nan))
    return out[0], out[1]


def detect_decelerations_v7(fhr_bpm: np.ndarray, *, fs: float = FS,
                            valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    r"""NICHD deceleration detector: at least $15\,$bpm below a robust baseline for $\ge 15\,$s.

    A candidate is a run where the smoothed trace lies at least $15\,$bpm below
    :func:`fhr_baseline` (runs closer than $5\,$s are merged) that lasts at least $15\,$s. The
    nadir is the run minimum. The onset and the end are found by walking out of the run while the
    trace is still more than $5\,$bpm below the baseline and still climbing away from the nadir,
    at most $120\,$s. A nadir within $30\,$s of either end is dropped, and so is an event whose
    span holds an invalid sample.

    This is the session-7 reference rule promoted to a detector, so its agreement with that
    reference is true by construction.

    Args:
        fhr_bpm: FHR in bpm.
        fs: Sampling rate in Hz.
        valid: Validity, or ``None``.

    Returns:
        ``{'onset_raw', 'nadir_raw', 'end_raw', 'depth_bpm', 'prolonged'}``; ``depth_bpm`` is
        the depth below the baseline at the nadir; ``prolonged`` marks onset-to-end $\ge 2\,$min
        (NICHD prolonged deceleration).
    """
    f = np.asarray(fhr_bpm, dtype=np.float64).reshape(-1)
    v = np.ones(f.size, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    smooth, base = fhr_baseline(f, v, fs=fs)
    below = (base - smooth) >= DEC_DEPTH_BPM
    idx = np.flatnonzero(np.diff(np.concatenate([[0], below.astype(np.int8), [0]])))
    runs: list = []
    for a, b in zip(idx[::2], idx[1::2]):
        if runs and a - runs[-1][1] < DEC_MERGE_S * fs:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    walk = int(DEC_WALK_S * fs)
    edge = int(DEC_EDGE_S * fs)
    on, nd, en, dp = [], [], [], []
    cum = np.concatenate([[0], np.cumsum(~v)])
    for a, b in runs:
        if b - a < DEC_MIN_S * fs:
            continue
        nadir = a + int(np.argmin(smooth[a:b]))
        o = a
        while o > max(0, a - walk) and base[o - 1] - smooth[o - 1] > DEC_RETURN_BPM and smooth[o - 1] >= smooth[o] - 0.05:
            o -= 1
        e = b - 1
        while e < min(f.size - 1, b - 1 + walk) and base[e + 1] - smooth[e + 1] > DEC_RETURN_BPM and smooth[e + 1] >= smooth[e] - 0.05:
            e += 1
        if nadir < edge or nadir >= f.size - edge or cum[e + 1] - cum[o] > 0:
            continue
        on.append(o)
        nd.append(nadir)
        en.append(e)
        dp.append(float(base[nadir] - smooth[nadir]))
    on_a, nd_a, en_a = (np.asarray(z, dtype=np.int64) for z in (on, nd, en))
    return {"onset_raw": on_a, "nadir_raw": nd_a, "end_raw": en_a, "depth_bpm": np.asarray(dp),
            "prolonged": (en_a - on_a) >= 120 * fs}
