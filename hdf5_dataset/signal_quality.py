r"""Per-sample quality codes, cleaning and coefficient contamination for the CTG signals.

The dataset builder reads every record through the MIMO adaptor and cuts it into segments of
$N = 5280$ samples at $f_s = 4$ Hz. This module takes one segment's **pre-fill** signals -- the
signals before ``MimoSequence.interpolate_input`` has replaced anything -- and does four things:

1. It gives every sample a ``uint8`` quality code for UA (:class:`UaBit`) and for FHR
   (:class:`FhrBit`). A sample can carry several bits.
2. It corrects the UA level after a detected baseline step, so the level difference of a sensor
   repositioning does not stay in the signal.
3. It bridges every invalid span with a one-interval monotone cubic (PCHIP), so a span edge does
   not excite every wavelet channel with a broadband step.
4. It measures how much of each stored coefficient is built from invalid samples. For a causal
   channel with magnitude kernel $w_c$ and the invalid indicator $b_n$, the contamination at step
   $k$ is

   $$\rho_c[k] = \frac{\sum_{j \ge 0} w_c[j]\, b_{Dk - j}}{\sum_{j \ge 0} w_c[j]},$$

   with $D = 16$ raw samples per step and $b_{n < 0} = b_0$, because the transform pads its
   history with the first sample. A coefficient is valid when $\rho_c[k] \le \varepsilon$.

**What the codes mean.** The UA values $\ge 128$ (128, 255) are invalid codes: in the local
shards every such run starts with a jump of 30 units or more, most of them from a value of 5 or
less. A run at exactly 100 (or 127, the top of a 0--127 monitor range) entered smoothly is a
contraction crest that the device or the export clamped: it is *censored* -- the timing is right
and the true value is at least the stored one -- and it stays valid. A ceiling run entered by a
jump, or longer than :attr:`QualityConfig.censored_max_s`, is an artifact.

**Jumps are local.** A jump of :attr:`QualityConfig.jump_units` within one stored step marks only
the discontinuity, because a steep contraction rise can contain one. A wider span is marked only
when the evidence is specific: a sharp opposite jump back to the old level (a *box*), a level
that is settled before the jump and settled again right after it (a *step*), or a pulse shorter
than any contraction (a *spike*, :attr:`QualityConfig.spike_side_s`).

**Causality.** Most rules read only past samples. Six read ahead, and each read is bounded by
:attr:`QualityConfig.lookahead_s`:

* a box is closed by its opposite jump within :attr:`QualityConfig.box_max_s`;
* a step needs a settled window within :attr:`QualityConfig.step_search_s` and no return to the
  old level within :attr:`QualityConfig.step_confirm_s`;
* the transient after a gap ends at the start of a settled window of
  :attr:`QualityConfig.settle_window_s`;
* a spike is measured over :attr:`QualityConfig.spike_side_s` on both sides;
* a constant run is a hold once it reaches :attr:`QualityConfig.hold_long_s`, or earlier with an
  edge jump at its end;
* a floor run is marked when an invalid code follows it within
  :attr:`QualityConfig.floor_near_s`;
* the bridge uses the first valid sample after a span. Every sample it writes is invalid, so a
  model that honours the mask never reads the bridge before that sample.

The module is pure NumPy (plus ``scipy.signal`` for the FFT convolution). It does not import the
scattering code; :func:`magnitude_kernels` takes the bank's arrays as arguments.
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Version of the rules and of the stored fields. A builder writes it as a file attribute, and a
#: file with another version is not complete for a build of this one.
QUALITY_VERSION = 1


class UaBit(enum.IntFlag):
    """Quality bits of one UA sample.

    Attributes:
        MISSING: Outside a recorded run, or in a padding tail.
        CODE: An invalid code: a value above 127, or below 0.
        CENSORED: Clamped by the device, valid. At the ceiling (100 or 127) after a smooth rise,
            the true value is at least the stored one; in a constant rest at the floor the
            uterus has decayed into, it is at most the stored one.
        TRANSIENT: A jump, a box between two opposite jumps, a spike, or the settling after a
            gap.
        STEP: A jump from a settled level to a new settled level: a sensor repositioning.
        HOLD: A constant hold that a repair or a frozen sensor produced.
        FLOOR: A floor (loss of contact) next to an invalid code, or an exact-zero run.
        SHIFTED: Valid, but a level correction after a ``STEP`` changed the value.
    """

    MISSING = 1
    CODE = 2
    CENSORED = 4
    TRANSIENT = 8
    STEP = 16
    HOLD = 32
    FLOOR = 64
    SHIFTED = 128


class FhrBit(enum.IntFlag):
    """Quality bits of one FHR sample.

    Attributes:
        MISSING: Outside a recorded run, or in a padding tail.
        REPAIRED: The upstream repair wrote this value (from the raw channel, when it is read).
        JUMP: Within :attr:`QualityConfig.fhr_jump_pad_s` of a jump of at least
            :attr:`QualityConfig.fhr_jump_bpm` within one sample.
        HOLD: Exactly constant for at least :attr:`QualityConfig.fhr_hold_min_s`.
        LINEAR: A monotone straight-line ramp for at least :attr:`QualityConfig.fhr_linear_min_s`.
    """

    MISSING = 1
    REPAIRED = 2
    JUMP = 8
    HOLD = 32
    LINEAR = 64


#: The UA bits that make a sample invalid. ``CENSORED`` and ``SHIFTED`` samples stay valid.
UA_INVALID = int(
    UaBit.MISSING | UaBit.CODE | UaBit.TRANSIENT | UaBit.STEP | UaBit.HOLD | UaBit.FLOOR
)
#: The FHR bits that make a sample invalid when the upstream repair is not known: every FHR bit.
FHR_INVALID = int(FhrBit.MISSING | FhrBit.REPAIRED | FhrBit.JUMP | FhrBit.HOLD | FhrBit.LINEAR)
#: The FHR bits that make a sample invalid when the repair mask comes from the raw channel.
#: ``HOLD`` and ``LINEAR`` are then recorded but do not invalidate: against the exact mask they
#: flag smooth, low-variability FHR as often as real repairs, and low variability is itself a
#: sign of the outcome the model predicts.
FHR_INVALID_EXACT = int(FhrBit.MISSING | FhrBit.REPAIRED | FhrBit.JUMP)


@dataclass(frozen=True)
class QualityConfig:
    r"""Thresholds of the quality rules. Times are in seconds, UA levels in TOCO units.

    Attributes:
        fs: Sampling rate in Hz.
        code_threshold: A UA value above this is an invalid code.
        ceiling_values: UA values that are a device or export ceiling.
        jump_units: A UA change of at least this over :attr:`jump_lag_s` is a jump. A
            contraction rises at most about $3$ units/s, so $15$ units in $1$ s is at least
            $4\times$ that.
        jump_lag_s: The lag of the jump difference. UP is $1$ Hz data held over $4$ samples, so
            $1$ s spans exactly one stored step.
        pre_level_s: Window before a jump whose median is the level before it.
        return_units: The signal is back at a level when it is within this of it.
        step_units: A level change of at least this is a ``STEP``, and a box must come back to
            within this of the level before it.
        settle_window_s: Length of a settled window.
        settle_range_units: Largest range (max - min) of a settled window, and of the window
            before a step.
        max_transient_s: Longest transient after a gap; it ends here even if the signal never
            settles.
        box_max_s: Longest box: an opposite jump later than this does not close it.
        step_search_s: A step must reach a settled window within this after its jump. A
            contraction keeps rising or falling; a repositioned sensor holds its new level.
        step_confirm_s: A step must not come back to the old level within this after its jump,
            so a broad contraction crest is not a step.
        spike_units: Smallest height of a spike above the higher of its two sides.
        spike_side_s: Width of each side window of a spike. A contraction crest falls by about
            $4$ units over $3$ s, so $12$ units is $3\times$ that.
        censored_max_s: A ceiling run is censored for this long, and a transient after it.
        gap_level_units: After a gap, a level this far from the level before the gap starts a
            transient that runs until the signal returns or settles.
        hold_min_s: Shortest constant run that can be a hold or a clamped rest.
        hold_long_s: A constant run this long is a hold without an edge condition, unless it is
            a clamped rest.
        hold_edge_units: A jump of at least this at the run's start or end makes a constant run
            a hold.
        hold_fhr_share: A constant run is also a hold when the FHR holds over at least this share
            of its first :attr:`hold_min_s`.
        rest_context_s: A constant run is a clamped rest when its value is at most the minimum of
            the usable samples in this window before it, plus :attr:`rest_units`. The uterus has
            decayed into the device's floor.
        rest_units: Tolerance of the clamped-rest test.
        rest_max_s: A clamped rest is censored for this long, and a hold after it. The builder's
            flat-UA gate uses the same $300$ s.
        floor_units: A UA value at most this is a floor value.
        floor_min_s: Shortest floor run.
        floor_near_s: A floor run is invalid when an invalid code or a missing sample lies
            within this distance of it.
        fhr_hold_min_s: Shortest exactly constant FHR run that is a hold. Natural integer runs
            reach about $11$ s.
        fhr_linear_min_s: Shortest monotone FHR ramp that can be a linear repair.
        fhr_linear_tol_bpm: Largest 95th-percentile residual of a linear repair's line fit. A
            repair rounded to whole bpm has residuals of at most $0.5$.
        fhr_linear_min_rise_bpm: Smallest total change of a linear repair; a smaller one is a
            hold.
        fhr_jump_bpm: An FHR change of at least this within one sample is a jump.
        fhr_jump_pad_s: Half-width of the span a jump marks.
        bridge_slope_s: Window before and after a span that sets the bridge's end slopes.
    """

    fs: float = 4.0
    code_threshold: float = 127.5
    ceiling_values: Tuple[float, ...] = (100.0, 127.0)
    jump_units: float = 15.0
    jump_lag_s: float = 1.0
    pre_level_s: float = 5.0
    return_units: float = 5.0
    step_units: float = 10.0
    settle_window_s: float = 20.0
    settle_range_units: float = 6.0
    max_transient_s: float = 150.0
    box_max_s: float = 60.0
    step_search_s: float = 10.0
    step_confirm_s: float = 40.0
    spike_units: float = 12.0
    spike_side_s: float = 3.0
    censored_max_s: float = 120.0
    gap_level_units: float = 10.0
    hold_min_s: float = 30.0
    hold_long_s: float = 60.0
    hold_edge_units: float = 10.0
    hold_fhr_share: float = 0.8
    rest_context_s: float = 120.0
    rest_units: float = 1.0
    rest_max_s: float = 300.0
    floor_units: float = 2.0
    floor_min_s: float = 20.0
    floor_near_s: float = 30.0
    fhr_hold_min_s: float = 15.0
    fhr_linear_min_s: float = 15.0
    fhr_linear_tol_bpm: float = 0.75
    fhr_linear_min_rise_bpm: float = 2.0
    fhr_jump_bpm: float = 20.0
    fhr_jump_pad_s: float = 0.5
    bridge_slope_s: float = 4.0

    def samples(self, seconds: float) -> int:
        """Convert a duration to a whole number of samples, at least 1."""
        return max(int(round(seconds * self.fs)), 1)

    @property
    def lookahead_s(self) -> float:
        """The longest look into the future of any rule: see the module docstring."""
        return float(max(self.box_max_s + self.pre_level_s, self.step_search_s + self.settle_window_s,
                         self.step_confirm_s, self.settle_window_s, self.spike_side_s,
                         self.hold_long_s + 2.0 * self.pre_level_s, 2.0 * self.floor_near_s))

    def as_attrs(self) -> Dict[str, Any]:
        """Return the thresholds as flat HDF5-attribute values, prefixed ``quality_``."""
        attrs: Dict[str, Any] = {"quality_version": QUALITY_VERSION,
                                 "quality_lookahead_s": self.lookahead_s}
        for key, value in asdict(self).items():
            attrs[f"quality_{key}"] = np.asarray(value, dtype=np.float64)
        return attrs


DEFAULT_CONFIG = QualityConfig()


# =================================================================================================
# Run helpers
# =================================================================================================
def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return the maximal runs of ``True`` in a boolean array.

    Args:
        mask: 1-D boolean array.

    Returns:
        ``[(start, stop), ...]`` with ``stop`` exclusive, in order.
    """
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0:
        return []
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def constant_runs(x: np.ndarray, usable: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Return the runs of exactly equal consecutive usable samples.

    Args:
        x: 1-D signal.
        usable: Samples that may belong to a run.
        min_len: Shortest run to return, in samples.

    Returns:
        ``[(start, stop), ...]`` with ``stop`` exclusive.
    """
    same = np.zeros(x.size, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & usable[1:] & usable[:-1]
    out = []
    for start, stop in runs(same):
        # The run of equal *pairs* [start, stop) starts one sample earlier as a run of samples.
        first = start - 1
        if stop - first >= min_len:
            out.append((first, stop))
    return out


def _distance_to(mask: np.ndarray, forward: bool) -> np.ndarray:
    """Samples to the nearest ``True`` ahead (``forward``) or behind; ``inf`` when there is none.

    Args:
        mask: 1-D boolean array.
        forward: Measure to the next ``True`` at or after each sample, else at or before it.

    Returns:
        ``float64`` distances in samples.
    """
    n = mask.size
    out = np.full(n, np.inf)
    last = -np.inf
    order = range(n - 1, -1, -1) if forward else range(n)
    for i in order:
        if mask[i]:
            last = i
        out[i] = abs(i - last)
    return out


# =================================================================================================
# FHR codes
# =================================================================================================
def fhr_quality(
    fhr: np.ndarray,
    present: np.ndarray,
    repaired: Optional[np.ndarray] = None,
    cfg: QualityConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Return the FHR quality bits of one segment.

    Args:
        fhr: Pre-fill FHR in bpm, ``(N,)``.
        present: Samples inside a recorded run and outside the padding tail.
        repaired: Samples the upstream repair wrote, when the raw channel was read; ``None``
            leaves ``REPAIRED`` unset and relies on the hold and ramp rules.
        cfg: The thresholds.

    Returns:
        ``uint8`` bits, ``(N,)``.
    """
    y = np.asarray(fhr, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    q = np.zeros(y.size, dtype=np.uint8)
    q[~present] |= np.uint8(FhrBit.MISSING)
    if repaired is not None:
        q[np.asarray(repaired, dtype=bool) & present] |= np.uint8(FhrBit.REPAIRED)

    for start, stop in constant_runs(y, present, cfg.samples(cfg.fhr_hold_min_s)):
        q[start:stop] |= np.uint8(FhrBit.HOLD)

    hold = (q & np.uint8(FhrBit.HOLD)) != 0
    for start, stop in _linear_ramps(y, present, cfg):
        # A hold with a step at its edge is monotone too; it stays a hold only.
        if np.mean(hold[start:stop]) < 0.5:
            q[start:stop] |= np.uint8(FhrBit.LINEAR)

    pad = cfg.samples(cfg.fhr_jump_pad_s)
    jump = np.zeros(y.size, dtype=bool)
    jump[1:] = (np.abs(np.diff(y)) >= cfg.fhr_jump_bpm) & present[1:] & present[:-1]
    for index in np.flatnonzero(jump):
        q[max(0, index - pad):min(y.size, index + pad)] |= np.uint8(FhrBit.JUMP)
    return q


def _linear_ramps(y: np.ndarray, present: np.ndarray, cfg: QualityConfig) -> List[Tuple[int, int]]:
    """Return monotone runs that a straight line fits to within the repair's rounding.

    Natural FHR at $4$ Hz reverses direction within seconds, so a monotone run of
    :attr:`QualityConfig.fhr_linear_min_s` is already unusual; the line fit separates a rounded
    linear repair from a smooth acceleration.

    Args:
        y: FHR in bpm.
        present: Usable samples.
        cfg: The thresholds.

    Returns:
        ``[(start, stop), ...]`` with ``stop`` exclusive.
    """
    min_len = cfg.samples(cfg.fhr_linear_min_s)
    d = np.diff(y)
    out: List[Tuple[int, int]] = []
    for sign in (1.0, -1.0):
        # A pair is part of a monotone run in this direction if it does not move the other way.
        ok = np.zeros(y.size, dtype=bool)
        ok[1:] = (sign * d >= 0) & present[1:] & present[:-1]
        for start, stop in runs(ok):
            first = start - 1
            if stop - first < min_len:
                continue
            seg = y[first:stop]
            if abs(seg[-1] - seg[0]) < cfg.fhr_linear_min_rise_bpm:
                continue
            t = np.arange(seg.size, dtype=np.float64)
            slope, intercept = np.polyfit(t, seg, 1)
            residual = np.abs(seg - (slope * t + intercept))
            if np.percentile(residual, 95) <= cfg.fhr_linear_tol_bpm:
                out.append((first, stop))
    return out


# =================================================================================================
# UA codes
# =================================================================================================
@dataclass(frozen=True)
class StepEvent:
    """One detected baseline step.

    Attributes:
        start: First sample of the step's transient.
        settle: First sample of the settled window; the level correction starts here.
        delta: Settled level minus the level before the jump, in UA units.
    """

    start: int
    settle: int
    delta: float


def ua_quality(
    up: np.ndarray,
    present: np.ndarray,
    fhr_hold: Optional[np.ndarray] = None,
    cfg: QualityConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, List[StepEvent]]:
    """Return the UA quality bits of one segment and its baseline steps.

    Args:
        up: Pre-fill UA in TOCO units, ``(N,)``.
        present: Samples inside a recorded run and outside the padding tail.
        fhr_hold: Samples where the FHR is a constant hold (the ``HOLD`` bit of
            :func:`fhr_quality`); ``None`` disables the joint-hold rule.
        cfg: The thresholds.

    Returns:
        ``(bits, steps)``: ``uint8`` bits ``(N,)``, and the ``STEP`` events in order.
    """
    x = np.asarray(up, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    n = x.size
    q = np.zeros(n, dtype=np.uint8)
    q[~present] |= np.uint8(UaBit.MISSING)
    code = present & ((x > cfg.code_threshold) | (x < 0.0))
    q[code] |= np.uint8(UaBit.CODE)
    data = present & ~code
    ceiling = data & np.isin(x, np.asarray(cfg.ceiling_values, dtype=np.float64))

    steps = _mark_transients(x, data, ceiling, q, cfg)
    _mark_spikes(x, data, q, cfg)
    _mark_censored(ceiling, q, cfg)
    _mark_holds(x, data & ~ceiling, q, fhr_hold, cfg)
    _mark_floors(x, data, q, cfg)
    # A frozen run is exactly constant, so the jump out of it passes the step's settled-level
    # test. Its level is not a baseline: the jump is only a discontinuity.
    held = (q & UaBit.HOLD) != 0
    kept: List[StepEvent] = []
    for step in steps:
        if step.start > 0 and held[step.start - 1]:
            span = slice(step.start, step.settle)
            q[span] &= np.uint8(0xFF ^ int(UaBit.STEP))
            q[span] |= np.uint8(UaBit.TRANSIENT)
        else:
            kept.append(step)
    return q, kept


def _level_before(x: np.ndarray, data: np.ndarray, stop: int, cfg: QualityConfig) -> Optional[float]:
    """Median of the usable samples in the :attr:`QualityConfig.pre_level_s` before ``stop``."""
    start = max(0, stop - cfg.samples(cfg.pre_level_s))
    window = x[start:stop][data[start:stop]]
    return float(np.median(window)) if window.size else None


def _settled_window(
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, begin: int, stop: int, cfg: QualityConfig
) -> Optional[Tuple[int, float]]:
    """Return the first settled window that starts in ``[begin, stop)``, or ``None``.

    A settled window lasts :attr:`QualityConfig.settle_window_s`, holds only usable samples off
    the ceiling, and has a range of at most :attr:`QualityConfig.settle_range_units`.

    Returns:
        ``(start, median)`` of the window.
    """
    window = cfg.samples(cfg.settle_window_s)
    for j in range(begin, min(stop, x.size - window + 1)):
        span = slice(j, j + window)
        if data[span].all() and not ceiling[span].any():
            values = x[span]
            if values.max() - values.min() <= cfg.settle_range_units:
                return j, float(np.median(values))
    return None


def _settle(
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, begin: int, level: float,
    cfg: QualityConfig,
) -> int:
    """Return the end of the transient after a gap.

    The scan starts at ``begin`` and stops at the next gap, at a return to within
    :attr:`QualityConfig.return_units` of ``level``, at the start of a settled window, or after
    :attr:`QualityConfig.max_transient_s`. Ceiling samples neither end nor settle it.
    """
    limit = min(x.size, begin + cfg.samples(cfg.max_transient_s))
    for j in range(begin, limit):
        if not data[j]:
            return j
        if ceiling[j]:
            continue
        if abs(x[j] - level) <= cfg.return_units:
            return j
        if _settled_window(x, data, ceiling, j, j + 1, cfg) is not None:
            return j
    return limit


def _box_end(
    x: np.ndarray, data: np.ndarray, jump: np.ndarray, i: int, sign: float, level: float,
    cfg: QualityConfig,
) -> Optional[int]:
    """Return the end of a box opened by the jump at ``i``, or ``None`` if there is no box.

    A box closes at the first opposite jump within :attr:`QualityConfig.box_max_s` that lands
    within :attr:`QualityConfig.step_units` of ``level`` and stays there: the median of the
    :attr:`QualityConfig.pre_level_s` after it is also that close. There is no gap in between. A
    spike that only touches the old level does not close a box.
    """
    lag = cfg.samples(cfg.jump_lag_s)
    hold = cfg.samples(cfg.pre_level_s)
    stop = min(x.size, i + cfg.samples(cfg.box_max_s) + 1)
    gaps = np.flatnonzero(~data[i:stop])
    if gaps.size:
        stop = i + int(gaps[0])
    for j in np.flatnonzero(jump[i + lag:stop]) + i + lag:
        if np.sign(x[j] - x[j - lag]) != -sign or abs(x[j] - level) > cfg.step_units:
            continue
        after = x[j:j + hold][data[j:j + hold]]
        if after.size and abs(float(np.median(after)) - level) <= cfg.step_units:
            return int(j) + 1
    return None


def _step_after(
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, start: int, i: int, sign: float,
    level: float, cfg: QualityConfig,
) -> Optional[Tuple[int, float]]:
    """Return ``(settle, settled level)`` if the jump at ``i`` is a step, or ``None``.

    Three tests must pass. The :attr:`QualityConfig.pre_level_s` before the jump is settled. A
    settled window starts within :attr:`QualityConfig.step_search_s` after the jump, at least
    :attr:`QualityConfig.step_units` away from ``level`` in the jump's direction. The signal does
    not come back to within :attr:`QualityConfig.return_units` of ``level`` within
    :attr:`QualityConfig.step_confirm_s`. A jump inside a contraction rise fails the first test,
    and one near a broad crest fails the third.
    """
    pre = slice(max(0, start - cfg.samples(cfg.pre_level_s)), start)
    before = x[pre][data[pre]]
    if before.size < cfg.samples(cfg.pre_level_s) // 2:
        return None
    if before.max() - before.min() > cfg.settle_range_units:
        return None
    found = _settled_window(x, data, ceiling, i + 1, i + 1 + cfg.samples(cfg.step_search_s), cfg)
    if found is None or (found[1] - level) * sign < cfg.step_units:
        return None
    after = slice(i + 1, min(x.size, i + 1 + cfg.samples(cfg.step_confirm_s)))
    if np.any(np.abs(x[after][data[after]] - level) <= cfg.return_units):
        return None
    return found


def _mark_transients(
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, q: np.ndarray, cfg: QualityConfig
) -> List[StepEvent]:
    """Mark jumps, boxes, baseline steps and the settling after gaps.

    Two triggers open an event. The first is a jump of at least :attr:`QualityConfig.jump_units`
    within :attr:`QualityConfig.jump_lag_s` between two usable samples. It marks a box up to its
    opposite jump (:func:`_box_end`), a ``STEP`` up to its settled window (:func:`_step_after`),
    or else only the discontinuity and one stored step on each side. The second is the end of a
    gap whose next value lies at least :attr:`QualityConfig.gap_level_units` from the level before
    the gap; it marks until :func:`_settle` closes it. A level change across a gap is not
    corrected.
    """
    n = x.size
    lag = cfg.samples(cfg.jump_lag_s)
    steps: List[StepEvent] = []
    gap_end = np.zeros(n, dtype=bool)
    gap_end[1:] = data[1:] & ~data[:-1]
    jump = np.zeros(n, dtype=bool)
    jump[lag:] = data[lag:] & data[:-lag] & (np.abs(x[lag:] - x[:-lag]) >= cfg.jump_units)

    # Only the candidate samples are visited; an event's end skips every candidate inside it.
    resume = 0
    for i in np.flatnonzero(gap_end | jump):
        i = int(i)
        if i < resume:
            continue
        if gap_end[i]:
            stop = i
            while stop > 0 and not data[stop - 1]:
                stop -= 1
            level = _level_before(x, data, stop, cfg)
            if level is not None and abs(x[i] - level) >= cfg.gap_level_units:
                end = _settle(x, data, ceiling, i, level, cfg)
                q[i:end] |= np.uint8(UaBit.TRANSIENT)
                resume = max(end, i + 1)
                continue
        if jump[i]:
            start = i - lag + 1
            level = _level_before(x, data, start, cfg)
            if level is None:
                level = float(x[i - lag])
            sign = float(np.sign(x[i] - x[i - lag]))
            end = _box_end(x, data, jump, i, sign, level, cfg)
            if end is not None:
                q[start:end] |= np.uint8(UaBit.TRANSIENT)
                resume = end
                continue
            step = _step_after(x, data, ceiling, start, i, sign, level, cfg)
            if step is not None:
                settle, settled = step
                q[start:settle] |= np.uint8(UaBit.STEP)
                steps.append(StepEvent(start=start, settle=settle, delta=settled - level))
                resume = max(settle, i + lag)
                continue
            q[max(0, start - lag):min(n, i + 1 + lag)] |= np.uint8(UaBit.TRANSIENT)
            # One stored step shows as a jump at `lag` consecutive samples: skip the rest of it.
            resume = i + lag
    return steps


def _mark_spikes(x: np.ndarray, data: np.ndarray, q: np.ndarray, cfg: QualityConfig) -> None:
    r"""Mark pulses and dips shorter than any contraction.

    For each usable sample $n$ with side windows $L_n = x[n-s, n)$ and $R_n = x(n, n+s]$ of
    :attr:`QualityConfig.spike_side_s`, the prominence is $x_n - \max(\min L_n, \min R_n)$. A
    run with prominence at least :attr:`QualityConfig.spike_units` is a spike; its samples above
    half its height, and one stored step on each side, are ``TRANSIENT``. The same test on $-x$
    finds dips. A contraction crest is too broad to pass: it falls only a few units within
    $s$ on either side.
    """
    n = x.size
    side = cfg.samples(cfg.spike_side_s)
    lag = cfg.samples(cfg.jump_lag_s)
    if n <= 2 * side:
        return
    for sign in (1.0, -1.0):
        y = np.where(data, sign * x, np.inf)
        lowest = np.lib.stride_tricks.sliding_window_view(y, side).min(axis=1)
        left = np.full(n, np.inf)
        left[side:] = lowest[: n - side]
        right = np.full(n, np.inf)
        right[: n - side] = lowest[1:]
        base = np.maximum(left, right)
        peak = data & np.isfinite(base) & (sign * x - base >= cfg.spike_units)
        for start, stop in runs(peak):
            top = start + int(np.argmax(sign * x[start:stop]))
            half = base[top] + 0.5 * (sign * x[top] - base[top])
            lo, hi = top, top + 1
            while lo > 0 and data[lo - 1] and sign * x[lo - 1] >= half:
                lo -= 1
            while hi < n and data[hi] and sign * x[hi] >= half:
                hi += 1
            q[max(0, lo - lag):min(n, hi + lag)] |= np.uint8(UaBit.TRANSIENT)


def _mark_censored(ceiling: np.ndarray, q: np.ndarray, cfg: QualityConfig) -> None:
    """Mark ceiling samples outside a transient as censored, up to the censored limit."""
    limit = cfg.samples(cfg.censored_max_s)
    artifact = (q & (UaBit.TRANSIENT | UaBit.STEP)) != 0
    for start, stop in runs(ceiling):
        cut = min(stop, start + limit)
        clean = ~artifact[start:cut]
        q[start:cut][clean] |= np.uint8(UaBit.CENSORED)
        if stop > cut:
            q[cut:stop] |= np.uint8(UaBit.TRANSIENT)


def _mark_holds(
    x: np.ndarray,
    usable: np.ndarray,
    q: np.ndarray,
    fhr_hold: Optional[np.ndarray],
    cfg: QualityConfig,
) -> None:
    r"""Mark constant UA runs that a repair or a frozen sensor produced, and clamped rests.

    UP is $1$ Hz integer data, so quiet stretches are naturally constant for up to about $24$ s.
    A run of at least :attr:`QualityConfig.hold_min_s` is a hold if it has an edge jump or a
    joint FHR hold. Otherwise it is a clamped rest if its value is the minimum of the
    :attr:`QualityConfig.rest_context_s` before it: the uterus has decayed into the device's
    floor, which is valid and ``CENSORED`` up to :attr:`QualityConfig.rest_max_s`. Any other run
    of :attr:`QualityConfig.hold_long_s` is a hold.

    The start edge is a jump of :attr:`QualityConfig.hold_edge_units` within one stored step into
    the run. The end edge is such a jump out of a run shorter than
    :attr:`QualityConfig.hold_long_s` after which the signal stays away: the median of the
    $2\times$ :attr:`QualityConfig.pre_level_s` after the run is also that far. A spike that
    starts where a rest ends is not an edge; the spike rule marks it.
    """
    floor = x <= cfg.floor_units
    lag = cfg.samples(cfg.jump_lag_s)
    first = cfg.samples(cfg.hold_min_s)
    context = cfg.samples(cfg.rest_context_s)
    rest_max = cfg.samples(cfg.rest_max_s)
    settle = cfg.samples(2.0 * cfg.pre_level_s)
    for start, stop in constant_runs(x, usable & ~floor, first):
        long_run = stop - start >= cfg.samples(cfg.hold_long_s)
        edge = False
        before = start - lag
        if before >= 0 and usable[before]:
            edge |= abs(x[start] - x[before]) >= cfg.hold_edge_units
        after = stop - 1 + lag
        if not long_run and after < x.size and usable[after]:
            window = slice(stop, min(x.size, stop + settle))
            later = float(np.median(x[window][usable[window]]))
            edge |= (abs(x[after] - x[stop - 1]) >= cfg.hold_edge_units
                     and abs(later - x[stop - 1]) >= cfg.hold_edge_units)
        joint = False
        if fhr_hold is not None:
            head = slice(start, min(stop, start + first))
            joint = float(np.mean(fhr_hold[head])) >= cfg.hold_fhr_share
        if edge or joint:
            q[start:stop] |= np.uint8(UaBit.HOLD)
            continue
        past = slice(max(0, start - context), start)
        prior = x[past][usable[past]]
        if prior.size == 0 or x[start] <= prior.min() + cfg.rest_units:
            cut = min(stop, start + rest_max)
            q[start:cut] |= np.uint8(UaBit.CENSORED)
            q[cut:stop] |= np.uint8(UaBit.HOLD)
        elif long_run:
            q[start:stop] |= np.uint8(UaBit.HOLD)


def _mark_floors(x: np.ndarray, data: np.ndarray, q: np.ndarray, cfg: QualityConfig) -> None:
    r"""Mark floor runs that are a lost contact rather than a quiet uterus.

    A floor run (values at most :attr:`QualityConfig.floor_units` for at least
    :attr:`QualityConfig.floor_min_s`) is invalid if it is all exact zeros, if it starts within
    :attr:`QualityConfig.floor_near_s` after a code or a missing sample, or -- for its samples
    within $2\times$ that distance of the next one -- if it ends within that distance before one.
    """
    bad = (q & (UaBit.CODE | UaBit.MISSING)) != 0
    since = _distance_to(bad, forward=False)
    until = _distance_to(bad, forward=True)
    near = cfg.samples(cfg.floor_near_s)
    floor = data & (x <= cfg.floor_units)
    for start, stop in runs(floor):
        if stop - start < cfg.samples(cfg.floor_min_s):
            continue
        if np.all(x[start:stop] == 0.0) or since[start] <= near:
            q[start:stop] |= np.uint8(UaBit.FLOOR)
        elif until[stop - 1] <= near:
            span = np.arange(start, stop)
            q[span[until[start:stop] <= 2 * near]] |= np.uint8(UaBit.FLOOR)


# =================================================================================================
# Cleaning
# =================================================================================================
def correct_levels(up: np.ndarray, steps: Sequence[StepEvent], valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract each step's level change from the samples after it settles.

    The TOCO baseline is arbitrary (a monitor resets it on connection), so a level change at a
    repositioning carries no information. Each correction uses only the settled window after the
    step, which lies inside the step's masked span or just after it.

    Args:
        up: The UA signal.
        steps: Output of :func:`ua_quality`.
        valid: Valid samples; only these change.

    Returns:
        ``(corrected, shifted)``: the corrected signal, and the valid samples whose value changed.
    """
    out = np.asarray(up, dtype=np.float64).copy()
    offset = np.zeros(out.size)
    for step in steps:
        offset[step.settle:] += step.delta
    shifted = valid & (offset != 0.0)
    out[shifted] -= offset[shifted]
    return out, shifted


def bridge(x: np.ndarray, invalid: np.ndarray, cfg: QualityConfig = DEFAULT_CONFIG) -> np.ndarray:
    r"""Replace every invalid span by a one-interval monotone cubic between its valid neighbours.

    The end slopes are the mean slopes over :attr:`QualityConfig.bridge_slope_s` outside the
    span, limited as in Fritsch--Carlson: zero if their sign differs from the secant
    $\delta = (y_1 - y_0)/h$, and at most $3|\delta|$ in size. So the bridge never overshoots its
    two anchors. A span at the segment start holds the first valid value; a span at the end holds
    the last one. A segment with no valid sample is returned unchanged.

    Args:
        x: The signal.
        invalid: Samples to replace.
        cfg: The thresholds.

    Returns:
        The bridged signal (a new array).
    """
    y = np.asarray(x, dtype=np.float64).copy()
    invalid = np.asarray(invalid, dtype=bool)
    n = y.size
    if invalid.all() or not invalid.any():
        return y
    k = cfg.samples(cfg.bridge_slope_s)
    for start, stop in runs(invalid):
        left, right = start - 1, stop
        if left < 0:
            y[start:stop] = y[right]
            continue
        if right >= n:
            y[start:stop] = y[left]
            continue
        h = float(right - left)
        y0, y1 = y[left], y[right]
        delta = (y1 - y0) / h
        m0 = _outside_slope(y, invalid, left, -k)
        m1 = _outside_slope(y, invalid, right, k)
        if delta == 0.0:
            m0 = m1 = 0.0
        else:
            m0 = 0.0 if np.sign(m0) != np.sign(delta) else float(np.clip(m0, -3 * abs(delta), 3 * abs(delta)))
            m1 = 0.0 if np.sign(m1) != np.sign(delta) else float(np.clip(m1, -3 * abs(delta), 3 * abs(delta)))
        t = (np.arange(start, stop, dtype=np.float64) - left) / h
        t2, t3 = t * t, t * t * t
        y[start:stop] = ((2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * m0
                         + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * m1)
    return y


def _outside_slope(y: np.ndarray, invalid: np.ndarray, anchor: int, span: int) -> float:
    """Mean slope over ``span`` samples from ``anchor`` away from the bridged span.

    Args:
        y: The signal.
        invalid: Samples that are not data.
        anchor: The valid sample next to the span.
        span: Signed window: negative looks before the anchor, positive after it.

    Returns:
        The slope per sample, or ``0.0`` if the window leaves the segment or touches an invalid
        sample.
    """
    other = anchor + span
    if other < 0 or other >= y.size:
        return 0.0
    lo, hi = min(anchor, other), max(anchor, other)
    if invalid[lo:hi + 1].any():
        return 0.0
    return float((y[anchor] - y[other]) / (anchor - other))


@dataclass(frozen=True)
class CleanResult:
    """One segment's codes and cleaned signals.

    Attributes:
        up_bits: UA quality bits, ``uint8 (N,)``.
        fhr_bits: FHR quality bits, ``uint8 (N,)``.
        up_clean: Level-corrected, bridged UA.
        fhr_clean: Bridged FHR.
        steps: The UA baseline steps.
        fhr_invalid_bits: The FHR bits that invalidate a sample: :data:`FHR_INVALID_EXACT`
            when the repair mask was given, else :data:`FHR_INVALID`.
    """

    up_bits: np.ndarray
    fhr_bits: np.ndarray
    up_clean: np.ndarray
    fhr_clean: np.ndarray
    steps: Tuple[StepEvent, ...]
    fhr_invalid_bits: int = FHR_INVALID

    @property
    def up_invalid(self) -> np.ndarray:
        """Invalid UA samples."""
        return (self.up_bits & UA_INVALID) != 0

    @property
    def fhr_invalid(self) -> np.ndarray:
        """Invalid FHR samples."""
        return (self.fhr_bits & self.fhr_invalid_bits) != 0


def clean_segment(
    fhr: np.ndarray,
    up: np.ndarray,
    present: np.ndarray,
    repaired: Optional[np.ndarray] = None,
    cfg: QualityConfig = DEFAULT_CONFIG,
    *,
    correct_steps: bool = True,
) -> CleanResult:
    """Code and clean one segment.

    Args:
        fhr: Pre-fill FHR in bpm, ``(N,)``.
        up: Pre-fill UA in TOCO units, ``(N,)``.
        present: Samples inside a recorded run and outside the padding tail.
        repaired: FHR samples the upstream repair wrote, or ``None``. When it is given, only
            :data:`FHR_INVALID_EXACT` invalidates an FHR sample.
        cfg: The thresholds.
        correct_steps: Subtract the level change after each UA baseline step.

    Returns:
        The :class:`CleanResult`.
    """
    fhr_bits = fhr_quality(fhr, present, repaired, cfg)
    fhr_hold = (fhr_bits & FhrBit.HOLD) != 0
    up_bits, steps = ua_quality(up, present, fhr_hold, cfg)
    up_invalid = (up_bits & UA_INVALID) != 0
    up_level = np.asarray(up, dtype=np.float64)
    if correct_steps and steps:
        up_level, shifted = correct_levels(up_level, steps, ~up_invalid)
        up_bits = up_bits | np.where(shifted, np.uint8(UaBit.SHIFTED), np.uint8(0))
    up_clean = bridge(up_level, up_invalid, cfg)
    fhr_invalid_bits = FHR_INVALID if repaired is None else FHR_INVALID_EXACT
    fhr_clean = bridge(fhr, (fhr_bits & fhr_invalid_bits) != 0, cfg)
    return CleanResult(up_bits=up_bits.astype(np.uint8), fhr_bits=fhr_bits,
                       up_clean=up_clean, fhr_clean=fhr_clean, steps=tuple(steps),
                       fhr_invalid_bits=fhr_invalid_bits)


# =================================================================================================
# Coefficient contamination
# =================================================================================================
def magnitude_kernels(
    psi: np.ndarray,
    phi: np.ndarray,
    kept: Mapping[str, np.ndarray],
    pairs: Mapping[str, np.ndarray],
    pair_shift: Optional[Mapping[str, np.ndarray]] = None,
    mass: float = 0.999,
) -> Dict[str, List[np.ndarray]]:
    r"""Build the causal magnitude kernel of every stored channel.

    A scattering channel $c \ge 1$ is $|x * \psi_{c-1}| * \phi$ and channel $0$ is $x * \phi$, so
    raw sample $n - j$ can reach the coefficient at $n$ only through $w_c = |\psi_{c-1}| * |\phi|$
    (or $|\phi|$). A phase channel multiplies two legs at the same instant -- the faster leg $j$
    delayed by $s_{ij}$ under envelope alignment -- and then smooths with $\phi$, so its kernel is
    $(|\psi_i| + |\psi_j|(\cdot - s_{ij})) * |\phi|$. Each kernel is normalised to sum $1$ and
    truncated where its cumulative sum reaches ``mass``.

    Args:
        psi: ``(n_filters, n_taps)`` complex causal wavelets, index $0$ = now.
        phi: ``(n_taps,)`` causal low-pass.
        kept: Block name -> kept channel indices before the drop (``CausalChannelPlan.kept``).
            Scattering blocks are the names ending in ``_st``; phase blocks end in ``_ph``.
        pairs: Phase block name -> ``(n_pairs, 2)`` filter pairs, column 0 the slower leg.
        pair_shift: Phase block name -> per-pair delay of the faster leg in samples, or ``None``
            for no leg alignment.
        mass: Cumulative mass at which a kernel is truncated.

    Returns:
        Block name -> one ``float64`` kernel per kept channel, in stored order.
    """
    from scipy.signal import fftconvolve

    abs_phi = np.abs(np.asarray(phi))
    abs_psi = np.abs(np.asarray(psi))

    def finish(kernel: np.ndarray) -> np.ndarray:
        kernel = np.clip(kernel, 0.0, None)
        kernel = kernel / kernel.sum()
        cut = int(np.searchsorted(np.cumsum(kernel), mass)) + 1
        kernel = kernel[:cut]
        return kernel / kernel.sum()

    out: Dict[str, List[np.ndarray]] = {}
    for block, channels in kept.items():
        kernels = []
        for channel in np.asarray(channels, dtype=int):
            if block.endswith("_st"):
                if channel == 0:
                    kernel = abs_phi.copy()
                else:
                    kernel = fftconvolve(abs_psi[channel - 1], abs_phi)[: abs_phi.size]
            else:
                i, j = (int(v) for v in np.asarray(pairs[block])[channel])
                shift = 0 if pair_shift is None else int(np.asarray(pair_shift[block])[channel])
                legs = abs_psi[i].copy()
                legs[shift:] += abs_psi[j][: legs.size - shift]
                kernel = fftconvolve(legs, abs_phi)[: abs_phi.size]
            kernels.append(finish(kernel))
        out[block] = kernels
    return out


class ContaminationBank:
    r"""The kernels of one stored block, with their spectra computed once.

    Calling the bank returns $\rho_c[k]$ for every kernel and stored step. The history before
    sample $0$ is the edge pad of the transform, so $b_{n<0} = b_0$.

    Args:
        kernels: Normalised causal kernels from :func:`magnitude_kernels`.
        n_signal: Raw samples per segment $N$.
        decimation: Raw samples per stored step; step $k$ reads raw sample $Dk$.
    """

    def __init__(self, kernels: Sequence[np.ndarray], n_signal: int, decimation: int = 16) -> None:
        self.n_signal = int(n_signal)
        self.decimation = int(decimation)
        self.history = max(int(k.size) for k in kernels) - 1
        self.n_fft = 1 << int(np.ceil(np.log2(self.n_signal + 2 * self.history)))
        self.spectra = np.stack([np.fft.rfft(k, n=self.n_fft) for k in kernels])
        self.picks = self.history + np.arange(self.n_signal // self.decimation) * self.decimation

    def __call__(self, bad: np.ndarray) -> np.ndarray:
        """Return the contamination of ``(N,)`` or ``(B, N)`` invalid indicators.

        Args:
            bad: Invalid indicator on the raw grid.

        Returns:
            ``(C, T)`` or ``(B, C, T)`` ``float32`` values in $[0, 1]$.
        """
        b = np.asarray(bad, dtype=np.float64)
        single = b.ndim == 1
        if single:
            b = b[None, :]
        padded = np.concatenate([np.repeat(b[:, :1], self.history, axis=1), b], axis=1)
        spectrum = np.fft.rfft(padded, n=self.n_fft, axis=1)
        out = np.empty((b.shape[0], self.spectra.shape[0], self.picks.size), dtype=np.float32)
        for c in range(self.spectra.shape[0]):
            full = np.fft.irfft(spectrum * self.spectra[c][None, :], n=self.n_fft, axis=1)
            out[:, c, :] = np.clip(full[:, self.picks], 0.0, 1.0)
        return out[0] if single else out


def contamination(
    bad: np.ndarray, kernels: Sequence[np.ndarray], decimation: int = 16
) -> np.ndarray:
    r"""Return $\rho_c[k]$ for every kernel and stored step (one-off form of the bank).

    Args:
        bad: ``(N,)`` or ``(B, N)`` invalid indicator on the raw grid.
        kernels: Normalised causal kernels from :func:`magnitude_kernels`.
        decimation: Raw samples per stored step.

    Returns:
        ``(C, T)`` or ``(B, C, T)`` ``float32`` contamination in $[0, 1]$, $T = N / D$.
    """
    n = np.asarray(bad).shape[-1]
    return ContaminationBank(kernels, n, decimation)(bad)


# =================================================================================================
# Summaries
# =================================================================================================
def bit_shares(bits: np.ndarray, flags: Any, over: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Share of samples carrying each bit.

    Args:
        bits: ``uint8`` quality bits.
        flags: :class:`UaBit` or :class:`FhrBit`.
        over: Samples to average over; ``None`` uses all.

    Returns:
        ``{bit name: share}``.
    """
    sel = np.ones(bits.shape, dtype=bool) if over is None else np.asarray(over, dtype=bool)
    denominator = max(int(sel.sum()), 1)
    return {flag.name.lower(): float(((bits & int(flag)) != 0)[sel].sum() / denominator)
            for flag in flags}
