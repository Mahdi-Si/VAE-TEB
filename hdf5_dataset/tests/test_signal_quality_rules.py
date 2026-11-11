r"""The UA rules of quality version 4 on synthetic segments of $5280$ samples at $4$ Hz.

Each test builds one trace that holds one pattern from the quality profiling and checks the bits
the rules give it:

* a flat line with a monitor pulse every $302$ s: the pulses are ``TRANSIENT``, the first
  $300$ s is a valid ``REST``, the rest of the line is ``FLAT``;
* a rest of $200$ s between contractions: ``REST``, valid, no ``FLAT``;
* a contraction clipped at $100$: ``CENSORED``, valid; a ceiling clamp of $200$ s: ``TRANSIENT``;
* a downward step to the floor: ``FLOOR``;
* bursts on contractions: ``TRANSIENT`` in the first stage, ``PUSH`` in the second;
* a level shift across a gap: one ``'span'`` shift;
* :meth:`ContaminationBank.step_term`: zero before a shift and after the kernel has passed it.
* the valid-box rule (``QualityConfig.box_valid``): off by default and then without effect; with
  it, a box of $30$ s that returns to the rest is a valid ``BOX`` and the bridge keeps its raw
  samples, while a short box, a box with a code and a level step stay as they were.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from hdf5_dataset import signal_quality as sq

B = sq.UaBit
FS = 4
N = 5280
T = np.arange(N) / FS
PRESENT = np.ones(N, dtype=bool)


def _s(seconds: float) -> int:
    """Seconds -> samples."""
    return int(seconds * FS)


def _share(q: np.ndarray, bit: int, start: int = 0, stop: int = N) -> float:
    """Share of ``q[start:stop]`` that carries ``bit``."""
    return float(np.mean((q[start:stop] & int(bit)) != 0))


def _contractions(base: float = 15.0, amp: float = 50.0, period: float = 180.0) -> np.ndarray:
    """Gaussian contractions on a rest level, held for $1$ s like the stored $1$ Hz UA."""
    x = base + amp * np.exp(-0.5 * (((T % period) - period / 2) / 20.0) ** 2)
    return np.round(np.repeat(x[::4], 4)[:N])


def test_a_flat_line_with_pulses_is_rest_then_flat() -> None:
    x = np.full(N, 20.0)
    pulses = np.arange(185, 1320, 302).tolist()
    for p in pulses:
        x[_s(p):_s(p) + 8] = 28.0
    q, _ = sq.ua_quality(x, PRESENT)
    for p in pulses:
        assert _share(q, B.TRANSIENT, _s(p), _s(p) + 8) == 1.0
    assert _share(q, B.FLAT, 0, _s(299)) == 0.0
    assert _share(q, B.FLAT, _s(301), N) > 0.98
    assert _share(q, sq.UA_INVALID, 0, _s(185)) == 0.0
    assert not np.any((q & B.FLAT) & (q & B.REST))


def test_a_short_rest_between_contractions_stays_valid() -> None:
    x = _contractions()
    x[_s(400):_s(600)] = x[_s(400):_s(600)].min()
    q, _ = sq.ua_quality(x, PRESENT)
    assert _share(q, B.REST, _s(400), _s(600)) == 1.0
    assert _share(q, B.FLAT) == 0.0
    assert _share(q, sq.UA_INVALID, _s(405), _s(595)) == 0.0


def test_a_clipped_crest_is_censored_and_a_long_clamp_is_transient() -> None:
    crest = np.minimum(_contractions(amp=100.0, period=240.0), 100.0)
    q, _ = sq.ua_quality(crest, PRESENT)
    ceiling = crest == 100.0
    assert ceiling.any() and np.all((q[ceiling] & B.CENSORED) != 0)
    assert _share(q, sq.UA_INVALID) == 0.0

    clamp = _contractions()
    clamp[_s(700):_s(900)] = 127.0
    q, _ = sq.ua_quality(clamp, PRESENT)
    assert _share(q, B.TRANSIENT, _s(700), _s(900)) == 1.0
    assert _share(q, B.CENSORED, _s(700), _s(900)) == 0.0


def test_a_drop_to_the_floor_is_lost_contact() -> None:
    x = np.round(40 + 2 * np.sin(2 * np.pi * T / 37))
    x[_s(500):] = 1.0 + (np.random.default_rng(0).random(N - _s(500)) < 0.2)
    q, steps = sq.ua_quality(x, PRESENT)
    assert any(step.delta <= -sq.DEFAULT_CONFIG.step_units for step in steps)
    assert _share(q, B.FLOOR, _s(530), N) == 1.0


def test_bursts_on_contractions_are_pushes_only_in_the_second_stage() -> None:
    x = _contractions(amp=60.0, period=120.0)
    for p in np.arange(60, 1320, 120).tolist():
        for k in (-8, 0, 8):
            x[_s(p + k):_s(p + k) + 6] += 30
    x = np.clip(x, 0, 99)
    stage2 = T >= 600
    q_off, _ = sq.ua_quality(x, PRESENT)
    q, _ = sq.ua_quality(x, PRESENT, stage2=stage2)
    assert _share(q_off, B.TRANSIENT, _s(600), N) > 0.05
    assert _share(q, B.PUSH, 0, _s(600)) == 0.0
    assert _share(q, B.TRANSIENT, 0, _s(600)) == _share(q_off, B.TRANSIENT, 0, _s(600))
    assert _share(q, B.PUSH, _s(600), N) == _share(q_off, B.TRANSIENT, _s(600), N)
    assert _share(q, B.TRANSIENT, _s(600), N) == 0.0


def test_a_level_shift_across_a_gap_is_recorded() -> None:
    x = np.round(30 + 1.5 * np.sin(2 * np.pi * T / 41))
    x[_s(600):] += 25
    present = PRESENT.copy()
    present[_s(590):_s(610)] = False
    _, steps = sq.ua_quality(x, present)
    assert [step.kind for step in steps] == ["span"]
    assert steps[0].at == _s(600) and abs(steps[0].delta - 25) <= 3


def test_the_step_term_covers_the_kernel_after_the_shift() -> None:
    kernel = np.ones(2048) / 2048
    bank = sq.ContaminationBank([kernel], N, 16)
    assert bank.step_term([]).max() == 0.0
    term = bank.step_term([_s(600)])[0]
    clock = np.arange(N // 16) * 16 / FS
    assert term.dtype == np.float32
    assert np.all(term[clock < 600] == 0.0)
    assert np.all(term[clock >= 600 + 512] == 0.0)
    assert term.max() > 0.99
    masked = clock[term > 0.1]
    assert abs((masked.max() - masked.min()) - 0.9 * 512) <= 8


# -------------------------------------------------------------------------------------------------
# The valid-box rule (``QualityConfig.box_valid``, off by default)

BOX_ON = dataclasses.replace(sq.DEFAULT_CONFIG, box_valid=True)


def _wiggling_contractions() -> np.ndarray:
    """Contractions on a rest that alternates by one unit each second, so no rest run is constant."""
    return _contractions() + (np.arange(N) // FS) % 2


def _top(level: float, seconds: float) -> np.ndarray:
    """A flat top that steps through three values, one per second, so it is not a frozen run."""
    return level + (np.arange(_s(seconds)) // FS) % 3


def _invalid(q: np.ndarray) -> np.ndarray:
    return (q & sq.UA_INVALID) != 0


def test_the_valid_box_rule_is_off_by_default_and_changes_nothing_then() -> None:
    x = _wiggling_contractions()
    x[_s(345):_s(375)] = _top(80, 30)
    q_default, shifts_default = sq.ua_quality(x, PRESENT)
    q_off, shifts_off = sq.ua_quality(x, PRESENT, cfg=dataclasses.replace(sq.DEFAULT_CONFIG, box_valid=False))
    assert sq.DEFAULT_CONFIG.box_valid is False
    assert np.array_equal(q_default, q_off) and shifts_default == shifts_off
    assert _share(q_default, B.BOX) == 0.0
    assert _share(q_default, B.TRANSIENT, _s(345), _s(375)) == 1.0
    # The look-ahead of the default rules does not move; the option's own look-ahead is longer.
    assert sq.DEFAULT_CONFIG.lookahead_s == 180.0 and BOX_ON.lookahead_s == 244.0
    assert int(B.BOX) & sq.UA_INVALID == 0


def test_a_box_that_returns_to_the_rest_is_valid_with_the_option() -> None:
    for level in (80.0, 100.0):                      # below the ceiling, and a clamp at it
        x = _wiggling_contractions()
        x[_s(345):_s(375)] = _top(80, 30) if level == 80.0 else 100.0
        q_off, _ = sq.ua_quality(x, PRESENT)
        q_on, _ = sq.ua_quality(x, PRESENT, cfg=BOX_ON)
        was = sq.runs(_invalid(q_off))
        assert len(was) == 1 and 16 * FS < was[0][1] - was[0][0] <= 64 * FS
        start, stop = was[0]
        assert not _invalid(q_on).any()
        assert _share(q_on, B.BOX, start, stop) == 1.0 and _share(q_on, B.TRANSIENT) == 0.0
        # Every bit outside the run is the bit of the default rules.
        outside = np.ones(N, dtype=bool)
        outside[start:stop] = False
        assert np.array_equal(q_on[outside], q_off[outside])
        # The bridge then leaves the raw box in the signal; without the option it removes it.
        assert np.array_equal(sq.bridge(x, _invalid(q_on)), x)
        assert sq.bridge(x, _invalid(q_off))[start + _s(5):stop - _s(5)].max() < 40.0


def test_a_short_box_a_low_box_and_a_box_with_a_code_stay_as_they_were() -> None:
    cases = {}
    x = _wiggling_contractions()
    x[_s(345):_s(357)] = _top(80, 12)                # 12 s: at most the minimum length
    cases["short"] = x
    x = _wiggling_contractions()
    x[_s(345):_s(375)] = _top(80, 30)
    x[_s(355)] = 255.0                               # an invalid code inside the box
    cases["code"] = x
    for name, x in cases.items():
        q_off, shifts_off = sq.ua_quality(x, PRESENT)
        q_on, shifts_on = sq.ua_quality(x, PRESENT, cfg=BOX_ON)
        assert _invalid(q_off).any(), name
        assert np.array_equal(q_on, q_off) and shifts_on == shifts_off, name


def test_a_level_step_is_not_a_valid_box() -> None:
    x = _wiggling_contractions()
    x[_s(345):] += 40.0                              # the level does not come back
    q_off, shifts_off = sq.ua_quality(x, PRESENT)
    q_on, shifts_on = sq.ua_quality(x, PRESENT, cfg=BOX_ON)
    assert len(shifts_off) == 1
    assert np.array_equal(q_on, q_off) and shifts_on == shifts_off
    assert _share(q_on, B.BOX) == 0.0


def test_the_option_is_recorded_in_the_file_attributes() -> None:
    off, on = sq.DEFAULT_CONFIG.as_attrs(), BOX_ON.as_attrs()
    assert float(off["quality_box_valid"]) == 0.0 and float(on["quality_box_valid"]) == 1.0
    assert float(on["quality_lookahead_s"]) == 244.0
    assert off["quality_version"] == on["quality_version"] == sq.QUALITY_VERSION
