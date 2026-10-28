r"""Per-sample quality codes, cleaning and coefficient contamination for the CTG signals.

The dataset builder reads every record through the MIMO adaptor and cuts it into segments of
$N = 5280$ samples at $f_s = 4$ Hz. This module takes one segment's **pre-fill** signals -- the
signals before ``MimoSequence.interpolate_input`` has replaced anything -- and does three things:

1. It gives every sample a quality code: ``uint16`` for UA (:class:`UaBit`) and ``uint8`` for
   FHR (:class:`FhrBit`). A sample can carry several bits.
2. It bridges every invalid span with a one-interval monotone cubic (PCHIP), so a span edge does
   not excite every wavelet channel with a broadband step.
3. It measures how much of each stored coefficient is built from invalid samples. For a causal
   channel with magnitude kernel $w_c$ and the invalid indicator $b_n$, the contamination at step
   $k$ is

   $$\rho_c[k] = \frac{\sum_{j \ge 0} w_c[j]\, b_{Dk - j}}{\sum_{j \ge 0} w_c[j]},$$

   with $D = 16$ raw samples per step and $b_{n < 0} = b_0$, because the transform pads its
   history with the first sample. A coefficient is valid when $\rho_c[k] \le \varepsilon$.

**What the codes mean.** The UA values $\ge 128$ (128, 255) are invalid codes: in the local
shards every such run starts with a jump of 30 units or more, most of them from a value of 5 or
less. A run at exactly 100 (or 127, the top of a 0--127 monitor range) entered smoothly is a
contraction crest that the device or the export clamped: it is *censored* -- the timing is right
and the true value is at least the stored one -- and it stays valid. A ceiling run that a jump
enters or leaves, or a ceiling region longer than :attr:`QualityConfig.censored_max_s`, is an
artifact as a whole. A constant rest at the minimum of the recent past is the other clamp: the
uterus has decayed into the device's floor or reference level, the true value is at most the
stored one, and it is a valid ``REST``. ``CENSORED`` and ``REST`` carry information and stay
valid; a sample that carries none gets an invalid bit of its own.

**Flat regions.** Constant runs at one value with short excursions between them form one flat
region (:attr:`QualityConfig.flat_merge_gap_s`). Each rule that limits a duration applies to
the region, not to a single run: a monitor that adds a pulse to a flat line every $302$ s kept
every run below $300$ s, and so below every limit. A region carries no information after
:attr:`QualityConfig.rest_max_s`; its later samples are ``FLAT`` (invalid). A pulse of a long
flat region, which leaves the value and returns to it, is a ``TRANSIENT``, and it is not an edge
of the runs beside it.

**Pushing.** In the second stage of labour, bearing-down efforts add short peaks on top of each
contraction. The jump and spike rules mark them, and the mask then follows the delivery mode,
because a vaginal delivery spends its last hour in the second stage. With the second-stage onset
known, a ``TRANSIENT`` run that rides on a contraction, stays above the rest and touches no code,
gap or ceiling is a valid ``PUSH`` (:attr:`QualityConfig.push_on_units`). Without the onset the
rule is off.

**Level shifts.** A step leaves a new level in valid samples, and a coefficient whose kernel
spans the step responds to the jump although few of its samples are invalid. Every ``STEP``, and
every short invalid span between two settled levels that differ by
:attr:`QualityConfig.step_units` without a return, is a level shift (:class:`StepEvent`).
:meth:`ContaminationBank.step_term` gives the coefficients that straddle it a contamination of
their own. A floor run that a downward ``STEP`` settles into is a lost contact (``FLOOR``).

**Jumps are local.** A jump of :attr:`QualityConfig.jump_units` within one stored step marks only
the discontinuity, because a steep contraction rise can contain one. A wider span is marked only
when the evidence is specific: a sharp opposite jump back to the old level (a *box*), a level
that is settled before the jump and settled again right after it (a *step*), or a pulse shorter
than any contraction (a *spike*, :attr:`QualityConfig.spike_side_s`).

**Levels are not corrected.** A step is masked up to its settled window, and the bridge crosses
it; the level after it stays as recorded. A correction that subtracts each step's level change
adds up its errors: a contraction plateau or a clamp taken for a step moved whole rests by up to
$100$ units in the profile. Without a correction, a missed or a false step leaves the signal as
the device wrote it.

**Islands.** In a region where the sensor toggles between a level and a ceiling or a code, the
short runs between the toggles pass every rule. They are not a signal the bridge may anchor on,
so a short valid run in a mostly invalid neighbourhood is an ``ISLAND``
(:attr:`QualityConfig.island_max_s`).

**Oscillations.** Maternal breathing can add a regular wave of $5$--$20$ s period to the UA, on
the rest and on every contraction. Its sharp edges pass the jump test and its crests pass the
spike test, so the version 2 rules masked whole segments of real contractions. A train of at
least :attr:`QualityConfig.osc_min_peaks` regularly spaced peaks (or dips) is an
``OSCILLATION``: the jump and spike rules ignore it, and its samples stay valid. An artifact burst
is not regular, so it is still marked. A sensor that toggles to the ceiling can be regular, so a
wave swings at most :attr:`QualityConfig.osc_max_units`, and a jump to or from a ceiling value is
marked inside a wave too.

**Causality.** Most rules read only past samples. Twelve read ahead, and each read is bounded by
:attr:`QualityConfig.lookahead_s`:

* a box is closed by its opposite jump within :attr:`QualityConfig.box_max_s`;
* a step needs a settled window within :attr:`QualityConfig.step_search_s` and no return to the
  old level within :attr:`QualityConfig.step_confirm_s`;
* a level shift across an invalid span of at most :attr:`QualityConfig.shift_span_max_s` needs
  a settled window within :attr:`QualityConfig.shift_search_s` and no return within
  :attr:`QualityConfig.step_confirm_s`;
* a ceiling region is censored only if it ends within :attr:`QualityConfig.censored_max_s`
  without a jump;
* a pulse of a flat region is known once the run after it starts, within
  :attr:`QualityConfig.flat_merge_gap_s` plus :attr:`QualityConfig.flat_run_min_s`, and once
  its region reaches :attr:`QualityConfig.pulse_region_s`;
* a push is judged by the :attr:`QualityConfig.push_context_s` after it;
* the transient after a gap ends at the start of a settled window of
  :attr:`QualityConfig.settle_window_s`;
* a spike is measured over :attr:`QualityConfig.spike_side_s` on both sides;
* a constant run is a hold once it reaches :attr:`QualityConfig.hold_long_s`, or earlier with an
  edge jump at its end;
* a floor run is marked when an invalid code follows it within
  :attr:`QualityConfig.floor_near_s`;
* an island is judged by the :attr:`QualityConfig.island_context_s` after it;
* an oscillation starts once the next :attr:`QualityConfig.osc_min_peaks` peaks of its train
  are seen, within :attr:`QualityConfig.osc_min_peaks` $\times$
  :attr:`QualityConfig.osc_period_max_s`;
* the bridge uses the first valid sample after a span. Every sample it writes is invalid, so a
  model that honours the mask never reads the bridge before that sample.

The module is pure NumPy, plus ``scipy.signal`` for the peaks and the FFT convolution and
``scipy.io`` to read the raw FHR channel. It does not import the scattering code;
:func:`magnitude_kernels` takes the bank's arrays as arguments.
"""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Version of the rules and of the stored fields. A builder writes it as a file attribute, and a
#: file with another version is not complete for a build of this one.
QUALITY_VERSION = 4


class UaBit(enum.IntFlag):
    """Quality bits of one UA sample.

    Attributes:
        MISSING: Outside a recorded run, or in a padding tail.
        CODE: An invalid code: a value above 127, or below 0.
        CENSORED: A contraction crest that the device clamped at the ceiling (100 or 127) after
            a smooth rise. Valid; the true value is at least the stored one.
        TRANSIENT: A jump, a box between two opposite jumps, a spike, the settling after a gap,
            or a pulse of a flat region.
        STEP: A jump from a settled level to a new settled level: a sensor repositioning.
        HOLD: A constant hold that a repair or a frozen sensor produced.
        FLOOR: A floor (loss of contact): next to an invalid code, after a downward step, or an
            exact-zero run.
        ISLAND: A short run that passed every other rule inside a mostly invalid region.
        OSCILLATION: Inside a regular train of peaks or dips (maternal breathing). Valid: the
            bit only records that the jump and spike rules were not applied here. It is the
            ninth bit, so the UA bits are ``uint16``.
        REST: A constant rest at the minimum of the recent past: the uterus has decayed into the
            device's floor or reference level. Valid; the true value is at most the stored one.
        PUSH: In the second stage, a run the jump and spike rules marked that rides on a
            contraction: a bearing-down effort. Valid.
        FLAT: A sample of a flat region more than :attr:`QualityConfig.rest_max_s` after its
            start. Invalid: it carries no information.
    """

    MISSING = 1
    CODE = 2
    CENSORED = 4
    TRANSIENT = 8
    STEP = 16
    HOLD = 32
    FLOOR = 64
    ISLAND = 128
    OSCILLATION = 256
    REST = 512
    PUSH = 1024
    FLAT = 2048


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


#: Storage type of the UA bits: :class:`UaBit` has twelve bits.
UA_DTYPE = np.uint16
#: The UA bits that make a sample invalid. ``CENSORED``, ``OSCILLATION``, ``REST`` and ``PUSH``
#: samples stay valid.
UA_INVALID = int(
    UaBit.MISSING | UaBit.CODE | UaBit.TRANSIENT | UaBit.STEP | UaBit.HOLD | UaBit.FLOOR
    | UaBit.ISLAND | UaBit.FLAT
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
        step_confirm_s: A step must not come back to the old level within this after its jump.
            A contraction returns to its rest level within about $2$ minutes, so a flat-topped
            contraction is not a step; at $40$ s, plateaus of $40$--$60$ s passed as steps in
            the profile.
        spike_units: Smallest height of a spike above the higher of its two sides.
        spike_side_s: Width of each side window of a spike. A contraction crest falls by about
            $4$ units over $3$ s, so $12$ units is $3\times$ that.
        censored_max_s: A ceiling region longer than this is an artifact as a whole: no
            contraction crest stays clamped this long. Ceiling runs at one value with at most
            :attr:`flat_merge_gap_s` between them form one region.
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
            decayed into the device's floor. The rest level of the push rule is read over the
            same window.
        rest_units: Tolerance of the clamped-rest test.
        rest_max_s: A flat region carries information for this long from its start; its later
            samples are ``FLAT``. The builder's flat-UA gate uses the same $300$ s.
        flat_run_min_s: Shortest constant run of a flat region.
        flat_merge_gap_s: Longest excursion between two constant runs of one flat region. In the
            profile, monitor pulses on a flat line last $1$--$12$ s.
        pulse_units: Smallest height of a pulse of a flat region, above or below its value. The
            monitor pulses rise $8$--$16$ units; a quiet rest wiggles by $1$.
        pulse_region_s: A flat region at least this long has its pulses marked.
        floor_units: A UA value at most this is a floor value.
        floor_min_s: Shortest floor run.
        floor_near_s: A floor run is invalid when an invalid code or a missing sample lies
            within this distance of it.
        island_max_s: A valid run shorter than this, with invalid samples on both sides, can be
            an ``ISLAND``.
        island_context_s: Width of each window, before and after the run, that judges an island.
        island_share: A run is an island when at least this share of each of its two context
            windows is invalid. A run between two isolated spikes has mostly valid context and
            stays valid.
        osc_units: Smallest prominence of a peak (or dip) of an oscillation, measured within
            :attr:`osc_period_max_s` on each side. The rest of a quiet uterus varies by about
            $2$ units.
        osc_max_units: Largest prominence of a peak of an oscillation. Breathing swings the TOCO
            by about $10$--$20$ units; a toggle between a rest and the ceiling swings further.
        osc_period_min_s: Shortest spacing of two peaks of an oscillation.
        osc_period_max_s: Longest spacing of two peaks of an oscillation. Maternal breathing has
            a period of about $3$--$20$ s; contractions are at least $1$ minute apart.
        osc_min_peaks: Fewest peaks of a train. A random burst of $5$ spikes is rarely regular.
        osc_ratio: Largest ratio of two consecutive spacings in a train.
        push_on_units: In the second stage, a ``TRANSIENT`` run is a push when the level just
            before or just after it is at least this far above the rest level (the 10th
            percentile of the :attr:`rest_context_s` before it).
        push_max_s: Longest push run.
        push_context_s: Window before and after a run that gives the contraction level.
        shift_span_max_s: Longest invalid span that can carry a level shift.
        shift_search_s: A level shift needs a settled window that starts within this after its
            span.
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
    step_confirm_s: float = 120.0
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
    flat_run_min_s: float = 10.0
    flat_merge_gap_s: float = 15.0
    pulse_units: float = 3.0
    pulse_region_s: float = 60.0
    floor_units: float = 2.0
    floor_min_s: float = 20.0
    floor_near_s: float = 30.0
    island_max_s: float = 20.0
    island_context_s: float = 20.0
    island_share: float = 0.5
    osc_units: float = 6.0
    osc_max_units: float = 30.0
    osc_period_min_s: float = 5.0
    osc_period_max_s: float = 20.0
    osc_min_peaks: int = 5
    osc_ratio: float = 1.5
    push_on_units: float = 10.0
    push_max_s: float = 60.0
    push_context_s: float = 10.0
    shift_span_max_s: float = 60.0
    shift_search_s: float = 30.0
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
                         self.step_confirm_s + self.jump_lag_s,
                         self.censored_max_s + self.flat_merge_gap_s + 2.0 * self.jump_lag_s,
                         self.settle_window_s,
                         self.spike_side_s, self.hold_long_s + 2.0 * self.pre_level_s,
                         2.0 * self.floor_near_s, self.island_max_s + self.island_context_s,
                         self.osc_min_peaks * self.osc_period_max_s,
                         self.flat_merge_gap_s + self.flat_run_min_s,
                         # A pulse is marked only once its flat region reaches pulse_region_s.
                         self.pulse_region_s + self.flat_merge_gap_s,
                         self.push_max_s + self.push_context_s,
                         self.shift_span_max_s + max(self.step_confirm_s,
                                                     self.shift_search_s + self.settle_window_s)))

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
    """One level shift. It is recorded and masked; the level is not corrected.

    Attributes:
        start: First sample of the step's ``STEP`` span, or of the invalid span.
        settle: First sample of the settled window after it.
        delta: Settled level minus the level before, in UA units.
        at: The sample of the discontinuity: the jump of a step, the middle of a span.
            :meth:`ContaminationBank.step_term` reads it.
        kind: ``'step'`` for a ``STEP`` jump, ``'span'`` for a shift across an invalid span.
    """

    start: int
    settle: int
    delta: float
    at: int
    kind: str = "step"


@dataclass(frozen=True)
class FlatRegion:
    """One flat line: constant runs at one value, with short gaps between them.

    Attributes:
        start: First sample of the first run.
        stop: One past the last sample of the last run.
        value: The constant value, in TOCO units.
        gaps: ``[(start, stop), ...]`` between consecutive runs, ``stop`` exclusive.
    """

    start: int
    stop: int
    value: float
    gaps: Tuple[Tuple[int, int], ...]


def flat_regions(x: np.ndarray, usable: np.ndarray, cfg: QualityConfig) -> List[FlatRegion]:
    """Return the flat regions of one segment.

    A region joins the constant runs of at least :attr:`QualityConfig.flat_run_min_s` at one
    value whose gaps are at most :attr:`QualityConfig.flat_merge_gap_s`. A gap can hold a pulse,
    a one-unit wiggle or a few missing samples; the line on both sides is the same.

    Args:
        x: UA in TOCO units.
        usable: Samples that may belong to a run (data off the ceiling).
        cfg: The thresholds.

    Returns:
        The regions in order. A region of one run has no gaps.
    """
    gap = cfg.samples(cfg.flat_merge_gap_s)
    out: List[FlatRegion] = []
    current: Optional[List[Any]] = None
    for start, stop in constant_runs(x, usable, cfg.samples(cfg.flat_run_min_s)):
        value = float(x[start])
        if current is not None and value == current[2] and start - current[1] <= gap:
            current[3].append((current[1], start))
            current[1] = stop
            continue
        if current is not None:
            out.append(FlatRegion(current[0], current[1], current[2], tuple(current[3])))
        current = [start, stop, value, []]
    if current is not None:
        out.append(FlatRegion(current[0], current[1], current[2], tuple(current[3])))
    return out


def ua_quality(
    up: np.ndarray,
    present: np.ndarray,
    fhr_hold: Optional[np.ndarray] = None,
    cfg: QualityConfig = DEFAULT_CONFIG,
    stage2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[StepEvent]]:
    """Return the UA quality bits of one segment and its level shifts.

    Args:
        up: Pre-fill UA in TOCO units, ``(N,)``.
        present: Samples inside a recorded run and outside the padding tail.
        fhr_hold: Samples where the FHR is a constant hold (the ``HOLD`` bit of
            :func:`fhr_quality`); ``None`` disables the joint-hold rule.
        cfg: The thresholds.
        stage2: Samples at or after the second-stage onset; ``None`` (onset unknown) disables
            the push rule.

    Returns:
        ``(bits, shifts)``: ``uint16`` bits ``(N,)``, and the level shifts (``STEP`` jumps and
        shifts across invalid spans) in order of :attr:`StepEvent.at`.
    """
    x = np.asarray(up, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    n = x.size
    q = np.zeros(n, dtype=UA_DTYPE)
    q[~present] |= UA_DTYPE(UaBit.MISSING)
    code = present & ((x > cfg.code_threshold) | (x < 0.0))
    q[code] |= UA_DTYPE(UaBit.CODE)
    data = present & ~code
    ceiling = data & np.isin(x, np.asarray(cfg.ceiling_values, dtype=np.float64))

    # First, because the jump and spike rules must not read a breathing wave as artifacts.
    regular = _oscillations(x, data, cfg)
    q[regular] |= UA_DTYPE(UaBit.OSCILLATION)
    regions = flat_regions(x, data & ~ceiling, cfg)
    pulses = _mark_pulses(x, data, regions, q, cfg)
    steps = _mark_transients(x, data, ceiling, q, regular, cfg)
    _mark_spikes(x, data, q, regular & ~ceiling, cfg)
    _mark_censored(x, ceiling, q, cfg)
    _mark_holds(x, data & ~ceiling, q, fhr_hold, pulses, cfg)
    _mark_flat(regions, q, cfg)
    # A frozen run is exactly constant, so the jump out of it passes the step's settled-level
    # test. Its level is not a baseline: the jump is only a discontinuity. A flat line past the
    # rest limit is the same.
    held = (q & (UaBit.HOLD | UaBit.FLAT)) != 0
    kept: List[StepEvent] = []
    for step in steps:
        if step.start > 0 and held[step.start - 1]:
            span = slice(step.start, step.settle)
            q[span] &= UA_DTYPE(0xFFFF ^ int(UaBit.STEP))
            q[span] |= UA_DTYPE(UaBit.TRANSIENT)
        else:
            kept.append(step)
    _mark_floors(x, data, q, kept, cfg)
    if stage2 is not None:
        _mark_pushes(x, data, ceiling, q, np.asarray(stage2, dtype=bool), cfg)
    # Last, because an island is judged by every other invalid bit around it.
    _mark_islands(q, cfg)
    shifts = kept + _span_shifts(x, data & ~ceiling, q, kept, cfg)
    return q, sorted(shifts, key=lambda event: event.at)


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
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, q: np.ndarray, start: int, i: int,
    sign: float, level: float, cfg: QualityConfig,
) -> Optional[Tuple[int, float]]:
    """Return ``(settle, settled level)`` if the jump at ``i`` is a step, or ``None``.

    Three tests must pass. The :attr:`QualityConfig.pre_level_s` before the jump is settled, and
    every sample in it is usable, off the ceiling and outside an earlier transient: a level that
    a clamp or an artifact holds is not a baseline. A settled window starts within
    :attr:`QualityConfig.step_search_s` after the jump, at least :attr:`QualityConfig.step_units`
    away from ``level`` in the jump's direction. The signal does not come back to within
    :attr:`QualityConfig.return_units` of ``level`` within :attr:`QualityConfig.step_confirm_s`.
    A jump inside a contraction rise fails the first test, and a flat-topped contraction fails
    the third.
    """
    pre = slice(max(0, start - cfg.samples(cfg.pre_level_s)), start)
    clean = data[pre] & ~ceiling[pre] & ((q[pre] & UA_DTYPE(UaBit.TRANSIENT | UaBit.STEP)) == 0)
    if clean.size < cfg.samples(cfg.pre_level_s) // 2 or not clean.all():
        return None
    before = x[pre]
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
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, q: np.ndarray, regular: np.ndarray,
    cfg: QualityConfig,
) -> List[StepEvent]:
    """Mark jumps, boxes, baseline steps and the settling after gaps.

    Two triggers open an event. The first is a jump of at least :attr:`QualityConfig.jump_units`
    within :attr:`QualityConfig.jump_lag_s` between two usable samples, outside an oscillation
    (``regular``) unless it lands on or leaves a ceiling value. It marks a box up to its opposite jump (:func:`_box_end`), a ``STEP`` up to
    its settled window (:func:`_step_after`), or else only the discontinuity and one stored step
    on each side. The second is the end of a gap whose next value lies at least
    :attr:`QualityConfig.gap_level_units` from the level before the gap; it marks until
    :func:`_settle` closes it.
    """
    n = x.size
    lag = cfg.samples(cfg.jump_lag_s)
    steps: List[StepEvent] = []
    gap_end = np.zeros(n, dtype=bool)
    gap_end[1:] = data[1:] & ~data[:-1]
    jump = np.zeros(n, dtype=bool)
    jump[lag:] = data[lag:] & data[:-lag] & (np.abs(x[lag:] - x[:-lag]) >= cfg.jump_units)
    # An edge of a breathing wave is not a discontinuity; a jump to or from the ceiling still is.
    touch = ceiling.copy()
    touch[lag:] |= ceiling[:-lag]
    jump &= ~regular | touch

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
                q[i:end] |= UA_DTYPE(UaBit.TRANSIENT)
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
                q[start:end] |= UA_DTYPE(UaBit.TRANSIENT)
                resume = end
                continue
            step = _step_after(x, data, ceiling, q, start, i, sign, level, cfg)
            if step is not None:
                settle, settled = step
                q[start:settle] |= UA_DTYPE(UaBit.STEP)
                steps.append(StepEvent(start=start, settle=settle, delta=settled - level, at=i))
                resume = max(settle, i + lag)
                continue
            q[max(0, start - lag):min(n, i + 1 + lag)] |= UA_DTYPE(UaBit.TRANSIENT)
            # One stored step shows as a jump at `lag` consecutive samples: skip the rest of it.
            resume = i + lag
    return steps


def _mark_spikes(
    x: np.ndarray, data: np.ndarray, q: np.ndarray, regular: np.ndarray, cfg: QualityConfig
) -> None:
    r"""Mark pulses and dips shorter than any contraction, outside an oscillation.

    For each usable sample $n$ with side windows $L_n = x[n-s, n)$ and $R_n = x(n, n+s]$ of
    :attr:`QualityConfig.spike_side_s`, the prominence is $x_n - \max(\min L_n, \min R_n)$. A
    run with prominence at least :attr:`QualityConfig.spike_units` is a spike; its samples above
    half its height, and one stored step on each side, are ``TRANSIENT``. The same test on $-x$
    finds dips. A contraction crest is too broad to pass: it falls only a few units within
    $s$ on either side. A spike whose top lies in an oscillation and off the ceiling
    (``regular``) is a crest of the wave and is not marked.
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
            if regular[top]:
                continue
            half = base[top] + 0.5 * (sign * x[top] - base[top])
            lo, hi = top, top + 1
            while lo > 0 and data[lo - 1] and sign * x[lo - 1] >= half:
                lo -= 1
            while hi < n and data[hi] and sign * x[hi] >= half:
                hi += 1
            q[max(0, lo - lag):min(n, hi + lag)] |= UA_DTYPE(UaBit.TRANSIENT)


def _oscillations(x: np.ndarray, data: np.ndarray, cfg: QualityConfig) -> np.ndarray:
    r"""Return the samples inside a regular train of peaks, or of dips.

    In each run of usable samples, a peak is a local maximum whose prominence within
    :attr:`QualityConfig.osc_period_max_s` on each side lies between
    :attr:`QualityConfig.osc_units` and :attr:`QualityConfig.osc_max_units`; a dip is a peak of
    $-x$. A train is a maximal sequence of
    peaks whose spacings lie in [:attr:`QualityConfig.osc_period_min_s`,
    :attr:`QualityConfig.osc_period_max_s`] and whose consecutive spacings differ by at most the
    factor :attr:`QualityConfig.osc_ratio`. A train of at least
    :attr:`QualityConfig.osc_min_peaks` peaks covers its first to its last peak, plus half its
    median spacing on each side, so the edges before the first crest and after the last one are
    inside it.

    Args:
        x: UA in TOCO units.
        data: Usable samples: present and not a code.
        cfg: The thresholds.

    Returns:
        Boolean ``(N,)``, ``True`` inside an oscillation.
    """
    import warnings

    from scipy.signal import find_peaks

    n = x.size
    region = np.zeros(n, dtype=bool)
    shortest = cfg.samples(cfg.osc_period_min_s)
    longest = cfg.samples(cfg.osc_period_max_s)
    need = int(cfg.osc_min_peaks)
    for start, stop in runs(data):
        if stop - start < (need - 1) * shortest:
            continue
        for sign in (1.0, -1.0):
            # A flat top can have a prominence of 0 inside the window. The prominence floor
            # drops such a peak, so SciPy's warning about it carries nothing.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="some peaks have a prominence of 0")
                peaks, _ = find_peaks(sign * x[start:stop],
                                      prominence=(cfg.osc_units, cfg.osc_max_units),
                                      wlen=2 * longest + 1)
            for first, last, spacing in _regular_trains(peaks, shortest, longest, need,
                                                        cfg.osc_ratio):
                pad = spacing // 2
                lo = start + max(0, int(peaks[first]) - pad)
                hi = start + min(stop - start, int(peaks[last]) + pad + 1)
                region[lo:hi] = True
    return region


def _regular_trains(
    peaks: np.ndarray, shortest: int, longest: int, need: int, ratio: float
) -> List[Tuple[int, int, int]]:
    """Return the maximal trains of regularly spaced peaks.

    Args:
        peaks: Peak positions in ascending order.
        shortest: Shortest spacing in samples.
        longest: Longest spacing in samples.
        need: Fewest peaks of a train.
        ratio: Largest ratio of two consecutive spacings.

    Returns:
        ``[(first, last, spacing), ...]``: indices into ``peaks`` of the first and the last peak
        of each train, and its median spacing in samples.
    """
    trains: List[Tuple[int, int, int]] = []
    first, previous = 0, 0
    for k in range(1, len(peaks) + 1):
        fits, gap = False, 0
        if k < len(peaks):
            gap = int(peaks[k] - peaks[k - 1])
            fits = shortest <= gap <= longest
            steady = k - first < 2 or max(gap, previous) <= ratio * min(gap, previous)
            if fits and steady:
                previous = gap
                continue
        # The train peaks[first:k] ends here.
        if k - first >= need:
            trains.append((first, k - 1, int(np.median(np.diff(peaks[first:k])))))
        if k < len(peaks):
            # A spacing that fits but breaks the rhythm starts a new train at its first peak.
            first = k - 1 if fits else k
            previous = gap
    return trains


def _mark_censored(x: np.ndarray, ceiling: np.ndarray, q: np.ndarray, cfg: QualityConfig) -> None:
    """Decide each ceiling region as a whole: ``CENSORED`` crest or ``TRANSIENT`` artifact.

    Ceiling runs at one value with at most :attr:`QualityConfig.flat_merge_gap_s` between them
    form one region. A region is an artifact if it spans more than
    :attr:`QualityConfig.censored_max_s`, or if any sample of its runs is already a transient or
    a step: a jump entered or left it, or it lies in a box, a spike or the settling after a gap.
    Otherwise every sample of its runs is censored. A decision per run would keep a clamp that
    dips by one unit now and then censored for minutes; a decision per sample would keep the
    tail of a jump-entered run valid, and the bridge would anchor on it.
    """
    limit = cfg.samples(cfg.censored_max_s)
    gap = cfg.samples(cfg.flat_merge_gap_s)
    artifact = (q & (UaBit.TRANSIENT | UaBit.STEP)) != 0
    groups: List[List[Tuple[int, int]]] = []
    for start, stop in runs(ceiling):
        last = groups[-1][-1] if groups else None
        if last is not None and x[start] == x[last[0]] and start - last[1] <= gap:
            groups[-1].append((start, stop))
        else:
            groups.append([(start, stop)])
    for group in groups:
        bad = group[-1][1] - group[0][0] > limit or any(artifact[a:b].any() for a, b in group)
        bit = UA_DTYPE(UaBit.TRANSIENT if bad else UaBit.CENSORED)
        for start, stop in group:
            q[start:stop] |= bit


def _mark_holds(
    x: np.ndarray,
    usable: np.ndarray,
    q: np.ndarray,
    fhr_hold: Optional[np.ndarray],
    pulses: np.ndarray,
    cfg: QualityConfig,
) -> None:
    r"""Mark constant UA runs that a repair or a frozen sensor produced, and clamped rests.

    UP is $1$ Hz integer data, so quiet stretches are naturally constant for up to about $24$ s.
    A run of at least :attr:`QualityConfig.hold_min_s` is a hold if it has an edge jump or a
    joint FHR hold. Otherwise it is a clamped rest if its value is the minimum of the
    :attr:`QualityConfig.rest_context_s` before it: the uterus has decayed into the device's
    floor, which is a valid ``REST``; :func:`_mark_flat` limits it to
    :attr:`QualityConfig.rest_max_s`. Any other run of :attr:`QualityConfig.hold_long_s` is a
    hold.

    The start edge is a jump of :attr:`QualityConfig.hold_edge_units` within one stored step into
    the run. The end edge is such a jump out of a run shorter than
    :attr:`QualityConfig.hold_long_s` after which the signal stays away: the median of the
    $2\times$ :attr:`QualityConfig.pre_level_s` after the run is also that far. A spike that
    starts where a rest ends is not an edge; the spike rule marks it. Nor is a pulse of a flat
    region (*pulses*): the runs on either side of it are one flat line, and one of them must not
    become a hold while the other stays a rest.
    """
    floor = x <= cfg.floor_units
    lag = cfg.samples(cfg.jump_lag_s)
    first = cfg.samples(cfg.hold_min_s)
    context = cfg.samples(cfg.rest_context_s)
    settle = cfg.samples(2.0 * cfg.pre_level_s)
    for start, stop in constant_runs(x, usable & ~floor, first):
        long_run = stop - start >= cfg.samples(cfg.hold_long_s)
        edge = False
        before = start - lag
        if before >= 0 and usable[before] and not pulses[before]:
            edge |= abs(x[start] - x[before]) >= cfg.hold_edge_units
        after = stop - 1 + lag
        if not long_run and after < x.size and usable[after] and not pulses[after]:
            window = slice(stop, min(x.size, stop + settle))
            later = float(np.median(x[window][usable[window]]))
            edge |= (abs(x[after] - x[stop - 1]) >= cfg.hold_edge_units
                     and abs(later - x[stop - 1]) >= cfg.hold_edge_units)
        joint = False
        if fhr_hold is not None:
            head = slice(start, min(stop, start + first))
            joint = float(np.mean(fhr_hold[head])) >= cfg.hold_fhr_share
        if edge or joint:
            q[start:stop] |= UA_DTYPE(UaBit.HOLD)
            continue
        past = slice(max(0, start - context), start)
        prior = x[past][usable[past]]
        if prior.size == 0 or x[start] <= prior.min() + cfg.rest_units:
            q[start:stop] |= UA_DTYPE(UaBit.REST)
        elif long_run:
            q[start:stop] |= UA_DTYPE(UaBit.HOLD)


def _mark_floors(x: np.ndarray, data: np.ndarray, q: np.ndarray, steps: Sequence[StepEvent],
                 cfg: QualityConfig) -> None:
    r"""Mark floor runs that are a lost contact rather than a quiet uterus.

    A floor run (values at most :attr:`QualityConfig.floor_units` for at least
    :attr:`QualityConfig.floor_min_s`) is invalid if it is all exact zeros, if it starts within
    :attr:`QualityConfig.floor_near_s` after a code or a missing sample, if a downward ``STEP``
    of at least :attr:`QualityConfig.step_units` settles into it (the settled window starts at
    most that distance before the run, or inside it), or -- for its samples within $2\times$
    that distance of the next code or missing sample -- if it ends within that distance before
    one. A uterus relaxes over tens of seconds; a level that falls to the floor in one stored
    step is a sensor that lost contact.
    """
    bad = (q & (UaBit.CODE | UaBit.MISSING)) != 0
    since = _distance_to(bad, forward=False)
    until = _distance_to(bad, forward=True)
    near = cfg.samples(cfg.floor_near_s)
    drops = [step.settle for step in steps if step.delta <= -cfg.step_units]
    floor = data & (x <= cfg.floor_units)
    for start, stop in runs(floor):
        if stop - start < cfg.samples(cfg.floor_min_s):
            continue
        entered = any(start - near <= settle < stop for settle in drops)
        if np.all(x[start:stop] == 0.0) or since[start] <= near or entered:
            q[start:stop] |= UA_DTYPE(UaBit.FLOOR)
        elif until[stop - 1] <= near:
            span = np.arange(start, stop)
            q[span[until[start:stop] <= 2 * near]] |= UA_DTYPE(UaBit.FLOOR)


def _mark_pulses(
    x: np.ndarray, data: np.ndarray, regions: Sequence[FlatRegion], q: np.ndarray,
    cfg: QualityConfig,
) -> np.ndarray:
    """Mark the pulses of long flat regions as ``TRANSIENT``.

    A pulse is a gap of a flat region of at least :attr:`QualityConfig.pulse_region_s` in which
    a usable sample lies :attr:`QualityConfig.pulse_units` or more from the region's value. The
    spike rule misses the monitor pulses on a flat line, which rise only $8$ units; a quiet rest
    does not leave an exact value and come back to it within seconds.

    Args:
        x: UA in TOCO units.
        data: Usable samples: present and not a code.
        regions: Output of :func:`flat_regions`.
        q: UA bits, changed in place.
        cfg: The thresholds.

    Returns:
        Boolean ``(N,)``, ``True`` on a pulse.
    """
    pulse = np.zeros(x.size, dtype=bool)
    longest = cfg.samples(cfg.pulse_region_s)
    for region in regions:
        if region.stop - region.start < longest:
            continue
        for start, stop in region.gaps:
            values = x[start:stop][data[start:stop]]
            if values.size and float(np.max(np.abs(values - region.value))) >= cfg.pulse_units:
                pulse[start:stop] = True
    q[pulse] |= UA_DTYPE(UaBit.TRANSIENT)
    return pulse


def _mark_flat(regions: Sequence[FlatRegion], q: np.ndarray, cfg: QualityConfig) -> None:
    """Mark the valid samples of each flat region after :attr:`QualityConfig.rest_max_s` ``FLAT``.

    The limit counts from the start of the region, so pulses or wiggles between its runs do not
    reset it. A ``REST`` sample that becomes ``FLAT`` loses its ``REST`` bit: ``REST`` stays the
    name of a valid clamped rest.
    """
    cap = cfg.samples(cfg.rest_max_s)
    valid = (q & UA_DTYPE(UA_INVALID)) == 0
    keep = UA_DTYPE(0xFFFF ^ int(UaBit.REST))
    for region in regions:
        cut = region.start + cap
        if cut >= region.stop:
            continue
        index = np.arange(cut, region.stop)[valid[cut:region.stop]]
        q[index] &= keep
        q[index] |= UA_DTYPE(UaBit.FLAT)


def _median_in(x: np.ndarray, ok: np.ndarray, start: int, stop: int) -> Optional[float]:
    """Median of ``x[start:stop]`` over the samples where ``ok``; ``None`` if there is none."""
    start, stop = max(0, start), min(x.size, stop)
    values = x[start:stop][ok[start:stop]] if stop > start else x[:0]
    return float(np.median(values)) if values.size else None


def _mark_pushes(
    x: np.ndarray, data: np.ndarray, ceiling: np.ndarray, q: np.ndarray, stage2: np.ndarray,
    cfg: QualityConfig,
) -> None:
    r"""In the second stage, turn ``TRANSIENT`` runs that ride on a contraction into ``PUSH``.

    A run qualifies when every one of its samples carries ``TRANSIENT`` and no other invalid
    bit, every sample lies in the second stage, and:

    * it is at most :attr:`QualityConfig.push_max_s` long;
    * it and its two neighbours are usable data, its neighbours are valid, and it touches no
      ceiling value: a gap, a code or a ceiling toggle stays an artifact;
    * the median of the valid samples in the :attr:`QualityConfig.push_context_s` before it, or
      after it, lies at least :attr:`QualityConfig.push_on_units` above the rest level $r$, the
      10th percentile of the valid samples in the :attr:`QualityConfig.rest_context_s` before
      it: the run sits on a contraction;
    * no sample falls below $r -$ :attr:`QualityConfig.return_units`: a dropout is not a push.

    Args:
        x: UA in TOCO units.
        data: Usable samples: present and not a code.
        ceiling: Usable samples at a ceiling value.
        q: UA bits, changed in place.
        stage2: Samples at or after the second-stage onset.
        cfg: The thresholds.
    """
    transient = int(UaBit.TRANSIENT)
    only = (((q & UA_DTYPE(transient)) != 0) & ((q & UA_DTYPE(UA_INVALID ^ transient)) == 0)
            & stage2)
    if not only.any():
        return
    clean = data & ((q & UA_DTYPE(UA_INVALID)) == 0)
    n = x.size
    longest = cfg.samples(cfg.push_max_s)
    context = cfg.samples(cfg.push_context_s)
    past = cfg.samples(cfg.rest_context_s)
    for start, stop in runs(only):
        if stop - start > longest or start == 0 or stop >= n:
            continue
        if not (clean[start - 1] and clean[stop]) or not data[start:stop].all() \
                or ceiling[start:stop].any():
            continue
        prior = x[max(0, start - past):start][clean[max(0, start - past):start]]
        if prior.size == 0:
            continue
        rest = float(np.percentile(prior, 10))
        sides = [v for v in (_median_in(x, clean, start - context, start),
                             _median_in(x, clean, stop, stop + context)) if v is not None]
        if not sides or max(sides) < rest + cfg.push_on_units:
            continue
        if float(x[start:stop].min()) < rest - cfg.return_units:
            continue
        q[start:stop] &= UA_DTYPE(0xFFFF ^ transient)
        q[start:stop] |= UA_DTYPE(UaBit.PUSH)


def _span_shifts(
    x: np.ndarray, usable: np.ndarray, q: np.ndarray, steps: Sequence[StepEvent],
    cfg: QualityConfig,
) -> List[StepEvent]:
    """Find level shifts across short invalid spans.

    The ``STEP`` test, applied to a whole invalid span instead of one jump. A run of invalid
    samples of at most :attr:`QualityConfig.shift_span_max_s` that holds no ``STEP`` is a level
    shift when the :attr:`QualityConfig.pre_level_s` before it is valid and settled (range at
    most :attr:`QualityConfig.settle_range_units`), a settled window starts within
    :attr:`QualityConfig.shift_search_s` after it at least :attr:`QualityConfig.step_units` away
    from the level before, and no valid sample of the :attr:`QualityConfig.step_confirm_s` after
    the span comes back to within :attr:`QualityConfig.return_units` of that level. A
    contraction returns to its rest within about $2$ minutes, so it fails the last test. Nothing
    is marked; the shifts only feed :meth:`ContaminationBank.step_term`.

    Args:
        x: UA in TOCO units.
        usable: Samples that are data and off the ceiling.
        q: The final UA bits.
        steps: The ``STEP`` events.
        cfg: The thresholds.

    Returns:
        One :class:`StepEvent` of kind ``'span'`` per shift; ``at`` is the middle of the span.
    """
    n = x.size
    invalid = (q & UA_DTYPE(UA_INVALID)) != 0
    good = usable & ~invalid
    no_ceiling = np.zeros(n, dtype=bool)
    pre_n = cfg.samples(cfg.pre_level_s)
    longest = cfg.samples(cfg.shift_span_max_s)
    search = cfg.samples(cfg.shift_search_s)
    confirm = cfg.samples(cfg.step_confirm_s)
    out: List[StepEvent] = []
    for start, stop in runs(invalid):
        if stop - start > longest or start < pre_n or stop >= n:
            continue
        if any(step.start < stop and step.settle > start for step in steps):
            continue
        before = x[start - pre_n:start]
        if not good[start - pre_n:start].all() or before.max() - before.min() > cfg.settle_range_units:
            continue
        level = float(np.median(before))
        found = _settled_window(x, good, no_ceiling, stop, stop + search, cfg)
        if found is None or abs(found[1] - level) < cfg.step_units:
            continue
        after = slice(stop, min(n, stop + confirm))
        if np.any(np.abs(x[after][good[after]] - level) <= cfg.return_units):
            continue
        out.append(StepEvent(start=start, settle=found[0], delta=found[1] - level,
                             at=(start + stop) // 2, kind="span"))
    return out


def _mark_islands(q: np.ndarray, cfg: QualityConfig) -> None:
    r"""Mark short valid runs inside mostly invalid regions as ``ISLAND``.

    A valid run shorter than :attr:`QualityConfig.island_max_s`, with an invalid sample on each
    side, is an island when at least :attr:`QualityConfig.island_share` of the
    :attr:`QualityConfig.island_context_s` before it is invalid, and also of the same window
    after it. Both sides must pass: a run next to one long gap is the edge of a valid region,
    not an island. A run that touches the segment's start or end is never an island. Every
    decision reads the invalid mask from before this rule, so the result does not depend on the
    order of the runs.

    Args:
        q: UA bits, changed in place.
        cfg: The thresholds.
    """
    invalid = (q & UA_DTYPE(UA_INVALID)) != 0
    n = q.size
    longest = cfg.samples(cfg.island_max_s)
    context = cfg.samples(cfg.island_context_s)
    for start, stop in runs(~invalid):
        if start == 0 or stop == n or stop - start >= longest:
            continue
        before = float(np.mean(invalid[max(0, start - context):start]))
        after = float(np.mean(invalid[stop:stop + context]))
        if min(before, after) >= cfg.island_share:
            q[start:stop] |= UA_DTYPE(UaBit.ISLAND)


# =================================================================================================
# Cleaning
# =================================================================================================
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
        up_bits: UA quality bits, ``uint16 (N,)``.
        fhr_bits: FHR quality bits, ``uint8 (N,)``.
        up_clean: Bridged UA. Its level is not corrected after a step.
        fhr_clean: Bridged FHR.
        steps: The UA level shifts (``STEP`` jumps and shifts across invalid spans), recorded
            only; :meth:`ContaminationBank.step_term` turns them into contamination.
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
    stage2: Optional[np.ndarray] = None,
) -> CleanResult:
    """Code and clean one segment.

    Args:
        fhr: Pre-fill FHR in bpm, ``(N,)``.
        up: Pre-fill UA in TOCO units, ``(N,)``.
        present: Samples inside a recorded run and outside the padding tail.
        repaired: FHR samples the upstream repair wrote, or ``None``. When it is given, only
            :data:`FHR_INVALID_EXACT` invalidates an FHR sample.
        cfg: The thresholds.
        stage2: UA samples at or after the second-stage onset, or ``None`` when the onset is
            unknown; see :func:`ua_quality`.

    Returns:
        The :class:`CleanResult`.
    """
    fhr_bits = fhr_quality(fhr, present, repaired, cfg)
    fhr_hold = (fhr_bits & FhrBit.HOLD) != 0
    up_bits, steps = ua_quality(up, present, fhr_hold, cfg, stage2)
    up_invalid = (up_bits & UA_INVALID) != 0
    up_clean = bridge(np.asarray(up, dtype=np.float64), up_invalid, cfg)
    fhr_invalid_bits = FHR_INVALID if repaired is None else FHR_INVALID_EXACT
    fhr_clean = bridge(fhr, (fhr_bits & fhr_invalid_bits) != 0, cfg)
    return CleanResult(up_bits=up_bits.astype(UA_DTYPE), fhr_bits=fhr_bits,
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
        # cum[c, j] = mass of kernel c on lags 0..j; 1 past its end.
        self.cumulative = np.ones((len(kernels), self.history + 1), dtype=np.float64)
        for c, kernel in enumerate(kernels):
            self.cumulative[c, :kernel.size] = np.cumsum(kernel)

    def step_term(self, positions: Sequence[int]) -> np.ndarray:
        r"""Return the contamination that level shifts add, for one segment.

        A shift at raw sample $s$ splits the window of the coefficient at $n = Dk$: the mass
        $A_c = \sum_{j=0}^{n-s} w_c[j]$ lies on the new level and $1 - A_c$ on the old one. The
        coefficient mixes two baselines, which no valid-sample count shows, so the term is
        $$\rho^{\mathrm{step}}_c[k] = \max_s 2\min(A_c, 1 - A_c),$$
        which is $0$ before the shift and after the kernel has passed it, and $1$ when the
        window is split in half. It is combined with :meth:`__call__` by a maximum.

        Args:
            positions: Raw samples of the shifts (:attr:`StepEvent.at`).

        Returns:
            ``(C, T)`` ``float32`` values in $[0, 1]$; zeros when there is no shift.
        """
        out = np.zeros((self.cumulative.shape[0], self.picks.size), dtype=np.float32)
        now = np.arange(self.picks.size) * self.decimation
        for position in positions:
            lag = now - int(position)
            ahead = lag >= 0
            mass = np.zeros((self.cumulative.shape[0], now.size), dtype=np.float64)
            mass[:, ahead] = self.cumulative[:, np.minimum(lag[ahead], self.history)]
            np.maximum(out, 2.0 * np.minimum(mass, 1.0 - mass), out=out, casting="unsafe")
        return out

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
# The raw FHR channel
# =================================================================================================
def _series(segdat: Any) -> Tuple[int, np.ndarray]:
    """Lay one ``segdat`` channel on an absolute 4 Hz index grid.

    Args:
        segdat: A channel's ``segdat`` struct from ``efmStruct``.

    Returns:
        ``(first_index, values)``: the grid index of ``values[0]``, and the values with $-1$
        where the channel has no sample.
    """
    incr = float(segdat.domainIncr)
    starts = np.atleast_1d(np.asarray(segdat.domainStart, dtype=np.float64))
    onsets = np.atleast_1d(np.asarray(segdat.parameterSet.onsetPointer, dtype=np.int64)) - 1
    lengths = np.atleast_1d(np.asarray(segdat.parameterSet.segLength, dtype=np.int64))
    data = np.asarray(segdat.dataSet, dtype=np.float64).ravel()
    first = np.round(starts / incr).astype(np.int64)
    origin = int(first.min())
    values = np.full(int((first + lengths).max() - origin), -1.0)
    for g, onset, length in zip(first, onsets, lengths):
        values[g - origin:g - origin + length] = data[onset:onset + length]
    return origin, values


def _lookup(origin: int, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Read ``values`` at absolute grid indices, $-1$ outside the channel."""
    local = index - origin
    out = np.full(index.shape, -1.0)
    inside = (local >= 0) & (local < values.size)
    out[inside] = values[local[inside]]
    return out


def raw_fhr_repaired(record_path: str, domain_start: np.ndarray, fhr_pre: np.ndarray,
                     present: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    r"""Mark the FHR samples that the upstream repair wrote, from the raw channel of the record.

    The adaptor reads the last of ``Repaired HR1``, ``Repaired HR2`` and ``Repaired external
    FECG`` present. The raw channel is the same name without ``Repaired``. Each segment is first
    placed on the repaired channel's grid (the offset with the most exact matches, within
    $\pm 4$ samples), then the raw channel is read at one record-wide offset within $\pm 1$
    sample. A present sample is repaired if the raw value there is absent, 0, 255, or different.

    Args:
        record_path: The ``.mat`` file.
        domain_start: Segment start times in seconds, ``(n,)``.
        fhr_pre: Pre-fill FHR ``(n, N)``.
        present: Present samples ``(n, N)``.

    Returns:
        ``(repaired, info)``: a boolean ``(n, N)`` array, or ``None`` when the record has no raw
        channel; and the channel name, the offsets and the match rates.
    """
    from scipy.io import loadmat

    mat = loadmat(record_path, squeeze_me=True, struct_as_record=False)["efmStruct"]
    signals = {sig.measure: sig for sig in np.atleast_1d(mat.signals)}
    chosen = [k for k in ("Repaired HR1", "Repaired HR2", "Repaired external FECG") if k in signals]
    if not chosen:
        return None, {"raw_fhr": "no repaired channel"}
    raw_key = chosen[-1].replace("Repaired ", "")
    if raw_key not in signals:
        return None, {"raw_fhr": f"no {raw_key!r} channel"}
    rep_origin, rep_values = _series(signals[chosen[-1]].segdat)
    raw_origin, raw_values = _series(signals[raw_key].segdat)

    n_seg, n = fhr_pre.shape
    grids = []
    placed = 0
    for i in range(n_seg):
        base = int(np.floor(domain_start[i] / 0.25)) + np.arange(n)
        best, best_hits = 0, -1
        for offset in range(-4, 5):
            hits = int(np.sum(_lookup(rep_origin, rep_values, base + offset)[present[i]]
                              == fhr_pre[i][present[i]]))
            if hits > best_hits:
                best, best_hits = offset, hits
        grids.append(base + best)
        placed += best_hits
    raw_offset, raw_hits = 0, -1
    for offset in (-1, 0, 1):
        hits = sum(int(np.sum(_lookup(raw_origin, raw_values, grids[i] + offset)[present[i]]
                              == fhr_pre[i][present[i]])) for i in range(n_seg))
        if hits > raw_hits:
            raw_offset, raw_hits = offset, hits
    repaired = np.zeros((n_seg, n), dtype=bool)
    for i in range(n_seg):
        raw = _lookup(raw_origin, raw_values, grids[i] + raw_offset)
        repaired[i] = present[i] & ((raw <= 0) | (raw >= 255) | (raw != fhr_pre[i]))
    total = max(int(present.sum()), 1)
    return repaired, {"raw_fhr": raw_key, "raw_offset": raw_offset,
                      "repaired_grid_match": placed / total, "raw_match": raw_hits / total}


# =================================================================================================
# Summaries
# =================================================================================================
def bit_shares(bits: np.ndarray, flags: Any, over: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Share of samples carrying each bit.

    Args:
        bits: Quality bits.
        flags: :class:`UaBit` or :class:`FhrBit`.
        over: Samples to average over; ``None`` uses all.

    Returns:
        ``{bit name: share}``.
    """
    sel = np.ones(bits.shape, dtype=bool) if over is None else np.asarray(over, dtype=bool)
    denominator = max(int(sel.sum()), 1)
    return {flag.name.lower(): float(((bits & int(flag)) != 0)[sel].sum() / denominator)
            for flag in flags}
