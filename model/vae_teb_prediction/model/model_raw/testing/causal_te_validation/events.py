"""Contraction and deceleration event detection for causal-TE validation.

Tests 4 and 10 of the causal-TE validation suite need physiologically
meaningful events on the raw FHR/UP traces:

* **Contractions** — local maxima of the smoothed UP signal with a minimum
  inter-event distance of 60 s and minimum width of 20 s.
* **FHR decelerations** — local downward dips of the smoothed FHR signal
  with a prominence consistent with a clinically significant drop
  ($\\ge 10\\,$bpm in raw bpm units, $\\ge 0.3\\,\\sigma$ in z-units).

Both detectors operate on the **raw 4 Hz grid** first; the helper
:func:`raw_to_decim_index` then maps event indices to the decimated
$T_{\\mathrm{dec}} = 300$ grid that indexes the model's $K_t$,
$\\widetilde{\\mathrm{TE}}_{t,\\ell}$ and $\\bar\\alpha_{t,\\ell}$.

The decimation factor is $16$ ($4\\,$Hz $\\times$ $16$ = $64$-sample
window $\\Rightarrow$ one decimated step per $4\\,$s), matching
``analyses/changepoint.py:185``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from model.vae_teb_prediction.model.model_raw.geometry import CROP

try:
    from scipy.signal import find_peaks, savgol_filter
except Exception:  # pragma: no cover - scipy is a hard dep elsewhere
    find_peaks = None  # type: ignore[assignment]
    savgol_filter = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _smooth(x: np.ndarray, *, window: int, poly: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing with a robust fallback when scipy is missing.

    Args:
        x: 1-D signal.
        window: Smoothing window in samples (must be odd; clipped to length).
        poly: Polynomial order (must be < window).

    Returns:
        Smoothed signal of identical shape.
    """
    if savgol_filter is None or x.size < max(window, poly + 2):
        return np.asarray(x, dtype=np.float32)
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 2 + (1 if (poly + 2) % 2 == 0 else 0))
    if w >= x.size:
        return np.asarray(x, dtype=np.float32)
    smoothed = savgol_filter(np.asarray(x, dtype=np.float32), w, poly)
    return np.asarray(smoothed, dtype=np.float32)


def _drop_edge_events(
    indices: np.ndarray,
    *,
    n_raw: int,
    edge_lo: int,
    edge_hi: int,
) -> np.ndarray:
    """Remove events that fall in the warmup/horizon-exclusion zone."""
    if indices.size == 0:
        return indices
    keep = (indices >= edge_lo) & (indices < (n_raw - edge_hi))
    return indices[keep]


def _to_1d(x: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary-shape array to 1-D by averaging non-time axes.

    The dataloader stores ``fhr`` and ``up`` as 1-D ``(R,)`` traces in v1.
    Some downstream code may pass ``(C, R)`` accidentally; we average the
    channel axis to keep this helper robust.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0).astype(np.float32)
    return arr.reshape(-1, arr.shape[-1]).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Contraction detection (UP)
# ---------------------------------------------------------------------------


def detect_contractions(
    up_raw: np.ndarray,
    fs: float = 4.0,
    *,
    smooth_seconds: float = 10.0,
    min_distance_seconds: float = 60.0,
    min_width_seconds: float = 20.0,
    prominence_sigma: float = 0.5,
    edge_seconds: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Detect uterine-contraction events on the raw UP trace.

    Detection pipeline:

    1. Smooth the trace with a Savitzky-Golay filter
       (default $\\sim 10\\,$s window).
    2. Find local maxima with
       :func:`scipy.signal.find_peaks` using ``distance=60\\,$s,
       ``width=20\\,$s, ``prominence=0.5 \\cdot \\sigma_{up}``.
    3. Estimate the rising edge ``onset_raw`` by walking backward from
       each peak until the smoothed gradient drops below the
       80th-percentile of positive gradients within $[peak - 60\\,$s,
       peak].

    Args:
        up_raw: Raw UP signal of shape $(R,)$ at ``fs`` Hz.
        fs: Sampling rate in Hz (default 4.0).
        smooth_seconds: Savitzky-Golay window in seconds.
        min_distance_seconds: Minimum spacing between peaks.
        min_width_seconds: Minimum peak width at half-prominence.
        prominence_sigma: Required prominence in units of $\\sigma$ of
            the smoothed UP.
        edge_seconds: Drop events within this many seconds of the start
            or end of the trace.

    Returns:
        Dict with three integer arrays of raw-sample indices:
        ``{"onset_raw", "peak_raw", "end_raw"}``. Empty arrays when no
        events detected or when scipy is unavailable.
    """
    sig = _to_1d(up_raw)
    n = int(sig.size)
    empty = {"onset_raw": np.empty(0, dtype=np.int64),
             "peak_raw":  np.empty(0, dtype=np.int64),
             "end_raw":   np.empty(0, dtype=np.int64)}
    if n < int(60 * fs) or find_peaks is None:
        return empty

    smooth_w = max(int(round(smooth_seconds * fs)), 5)
    smooth = _smooth(sig, window=smooth_w, poly=3)
    sig_std = float(np.nanstd(smooth)) or 1.0

    distance = max(int(round(min_distance_seconds * fs)), 1)
    width = max(int(round(min_width_seconds * fs)), 1)
    prominence = float(prominence_sigma) * sig_std

    peaks, _ = find_peaks(
        smooth,
        distance=distance,
        prominence=prominence,
        width=width,
    )
    edge = int(round(edge_seconds * fs))
    peaks = _drop_edge_events(np.asarray(peaks, dtype=np.int64),
                              n_raw=n, edge_lo=edge, edge_hi=edge)
    if peaks.size == 0:
        return empty

    # Rising-edge onset estimate per peak.
    grad = np.gradient(smooth).astype(np.float32)
    pos_grad = grad[grad > 0]
    grad_thresh = (
        float(np.percentile(pos_grad, 80))
        if pos_grad.size > 0 else 0.0
    )
    onsets = np.empty_like(peaks)
    ends = np.empty_like(peaks)
    look_back = int(round(60.0 * fs))
    look_fwd = int(round(60.0 * fs))
    for i, p in enumerate(peaks):
        lo = max(0, int(p) - look_back)
        # Walk backward from peak until the gradient first drops below threshold.
        idx = int(p)
        while idx > lo and grad[idx] >= grad_thresh:
            idx -= 1
        onsets[i] = idx
        # Walk forward until the gradient becomes consistently non-negative
        # (post-peak relaxation back to baseline).
        hi = min(n - 1, int(p) + look_fwd)
        idx = int(p)
        while idx < hi and grad[idx] <= -grad_thresh:
            idx += 1
        ends[i] = idx

    return {"onset_raw": onsets, "peak_raw": peaks, "end_raw": ends}


# ---------------------------------------------------------------------------
# Deceleration detection (FHR)
# ---------------------------------------------------------------------------


def detect_decelerations(
    fhr_raw: np.ndarray,
    fs: float = 4.0,
    *,
    smooth_seconds: float = 8.0,
    min_distance_seconds: float = 15.0,
    prominence_bpm: float = 10.0,
    prominence_sigma: float = 0.3,
    edge_seconds: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Detect FHR-deceleration events (downward dips) on the raw FHR trace.

    Decelerations are local minima, so we invert the signal and call
    :func:`scipy.signal.find_peaks` on $-\\mathrm{FHR}_{\\mathrm{smooth}}$.
    The prominence threshold is the maximum of an absolute bpm threshold
    (typically $10\\,$bpm) and a $\\sigma$-relative threshold; this lets
    the detector work on raw bpm signals AND z-score-normalised signals
    without requiring the caller to know which it is.

    Args:
        fhr_raw: Raw FHR signal of shape $(R,)$ at ``fs`` Hz.
        fs: Sampling rate in Hz (default 4.0).
        smooth_seconds: Savitzky-Golay window in seconds.
        min_distance_seconds: Minimum spacing between deceleration nadirs.
        prominence_bpm: Prominence in bpm if signal looks like raw bpm.
        prominence_sigma: Prominence in units of $\\sigma$.
        edge_seconds: Drop events within this many seconds of the edges.

    Returns:
        Dict with three integer arrays: ``{"onset_raw", "nadir_raw",
        "end_raw"}``.
    """
    sig = _to_1d(fhr_raw)
    n = int(sig.size)
    empty = {"onset_raw": np.empty(0, dtype=np.int64),
             "nadir_raw": np.empty(0, dtype=np.int64),
             "end_raw":   np.empty(0, dtype=np.int64)}
    if n < int(30 * fs) or find_peaks is None:
        return empty

    smooth_w = max(int(round(smooth_seconds * fs)), 5)
    smooth = _smooth(sig, window=smooth_w, poly=3)
    sig_std = float(np.nanstd(smooth)) or 1.0
    sig_max = float(np.nanmax(np.abs(smooth))) if smooth.size else 0.0
    looks_like_bpm = sig_max > 30.0  # FHR in bpm is centred near 140

    prominence = max(
        float(prominence_bpm) if looks_like_bpm else 0.0,
        float(prominence_sigma) * sig_std,
    )
    distance = max(int(round(min_distance_seconds * fs)), 1)

    inv = -smooth
    nadirs, _ = find_peaks(inv, distance=distance, prominence=prominence)
    edge = int(round(edge_seconds * fs))
    nadirs = _drop_edge_events(np.asarray(nadirs, dtype=np.int64),
                               n_raw=n, edge_lo=edge, edge_hi=edge)
    if nadirs.size == 0:
        return empty

    # Onset = walking backward from nadir until the gradient turns positive.
    grad = np.gradient(smooth).astype(np.float32)
    look_back = int(round(60.0 * fs))
    look_fwd = int(round(60.0 * fs))
    onsets = np.empty_like(nadirs)
    ends = np.empty_like(nadirs)
    for i, t in enumerate(nadirs):
        lo = max(0, int(t) - look_back)
        idx = int(t)
        while idx > lo and grad[idx] <= 0:
            idx -= 1
        onsets[i] = idx
        hi = min(n - 1, int(t) + look_fwd)
        idx = int(t)
        while idx < hi and grad[idx] >= 0:
            idx += 1
        ends[i] = idx

    return {"onset_raw": onsets, "nadir_raw": nadirs, "end_raw": ends}


# ---------------------------------------------------------------------------
# Raw <-> decimated index helpers
# ---------------------------------------------------------------------------


def raw_to_decim_index(
    t_raw: np.ndarray, *, decim: int = 16, T_dec: int = 300, crop: int = CROP,
) -> np.ndarray:
    """Map raw 4 Hz indices onto the model's **cropped** decimated grid.

    A raw sample ``r`` belongs to the *uncropped* token ``r // decim``; the model then trims
    ``crop`` tokens off each side (``geometry.CROP``) before emitting its ``kld_per_t`` /
    ``te_lag_map`` / ``attn_weights``. A genuine event at raw sample ``r`` therefore aligns to the
    **cropped** anchor ``r // decim - crop`` (cropped anchor ``t`` == uncropped token ``t + crop``;
    see ``geometry.n_raw``). Omitting the ``- crop`` offset reads the bottleneck ``crop`` steps
    (1 min at production geometry) too late, misaligning every event-triggered / lag-alignment
    statistic. Out-of-range indices clip to ``[0, T_dec - 1]`` (edge-zone events land at the
    boundary, where the warmup / horizon trimming in :func:`label_event_windows` drops them).

    Args:
        t_raw: Integer raw-sample indices.
        decim: Decimation factor (default 16).
        T_dec: Decimated (cropped) sequence length (default 300).
        crop: Tokens the model trims off each side after the front end (default ``geometry.CROP``).

    Returns:
        Integer cropped-grid indices of the same shape as ``t_raw``.
    """
    arr = np.asarray(t_raw, dtype=np.int64)
    out = arr // int(decim) - int(crop)
    return np.clip(out, 0, int(T_dec) - 1)


def label_event_windows(
    events_dec: np.ndarray,
    T_dec: int,
    *,
    onset_pad: int = 2,
    peak_pad: int = 1,
    warmup: int = 0,
    horizon_exclude: int = 0,
) -> np.ndarray:
    """Build a $W^{+}$ mask of event-aligned decimated timesteps.

    For each event index $t \\in \\mathrm{events\\_dec}$, mark
    $[t - \\mathrm{onset\\_pad}, t + \\mathrm{peak\\_pad}]$ as ``True``
    (clipped to $[\\mathrm{warmup}, T_{\\mathrm{dec}} - \\mathrm{horizon\\_exclude})$).
    The complement (with the same warmup/horizon trimming) is the quiet
    mask $W^{-}$.

    Args:
        events_dec: Integer decimated-grid event indices.
        T_dec: Decimated sequence length.
        onset_pad: Steps before the event.
        peak_pad: Steps after the event.
        warmup: Initial steps to exclude (typically ``model.warmup_steps``).
        horizon_exclude: Trailing steps to exclude (typically
            ``model.horizon``) so events near the horizon edge are not
            counted twice.

    Returns:
        Boolean mask of shape $(T_{\\mathrm{dec}},)$. ``True`` indicates
        an event window.
    """
    mask = np.zeros(int(T_dec), dtype=bool)
    if events_dec.size == 0:
        return mask
    lo = int(max(0, warmup))
    hi = int(max(lo, T_dec - max(0, horizon_exclude)))
    for t in events_dec:
        a = int(max(lo, int(t) - int(onset_pad)))
        b = int(min(hi, int(t) + int(peak_pad) + 1))
        if b > a:
            mask[a:b] = True
    # Always exclude the warmup region (defensive).
    if lo > 0:
        mask[:lo] = False
    if hi < T_dec:
        mask[hi:] = False
    return mask


def quiet_mask(
    event_mask: np.ndarray,
    *,
    warmup: int = 0,
    horizon_exclude: int = 0,
) -> np.ndarray:
    """Complement of ``event_mask`` excluding warmup / trailing horizon.

    Args:
        event_mask: Boolean event-window mask.
        warmup: Initial steps to also exclude.
        horizon_exclude: Trailing steps to also exclude.

    Returns:
        Boolean quiet mask of the same shape as ``event_mask``.
    """
    T = int(event_mask.size)
    out = ~event_mask
    if warmup > 0:
        out[:warmup] = False
    if horizon_exclude > 0:
        out[T - horizon_exclude:] = False
    return out


def pair_events_by_window(
    up_events_dec: np.ndarray,
    fhr_events_dec: np.ndarray,
    *,
    max_lag_steps: int,
) -> Optional[np.ndarray]:
    """Pair UP events with the next FHR event within ``max_lag_steps``.

    For each UP-event index $u$, find the smallest FHR-event index $f$
    with $0 \\le f - u \\le \\mathrm{max\\_lag\\_steps}$. Each FHR event
    is matched at most once (one-shot greedy assignment in ascending
    order).

    Args:
        up_events_dec: UP-event decimated indices.
        fhr_events_dec: FHR-event decimated indices.
        max_lag_steps: Maximum positive lag in decimated steps.

    Returns:
        ``(N, 2)`` int64 array of paired ``(u_idx, f_idx)`` indices, or
        ``None`` if no pairs were found.
    """
    if up_events_dec.size == 0 or fhr_events_dec.size == 0:
        return None
    used = np.zeros(int(fhr_events_dec.size), dtype=bool)
    pairs = []
    fhr_sorted = np.argsort(fhr_events_dec)
    fhr_ord = fhr_events_dec[fhr_sorted]
    for u in np.sort(up_events_dec):
        # Find the first unused fhr event with index >= u.
        for j_ord, f in enumerate(fhr_ord):
            j = int(fhr_sorted[j_ord])
            if used[j]:
                continue
            if f < u:
                continue
            if (f - u) > max_lag_steps:
                break
            pairs.append((int(u), int(f)))
            used[j] = True
            break
    if not pairs:
        return None
    return np.asarray(pairs, dtype=np.int64)
