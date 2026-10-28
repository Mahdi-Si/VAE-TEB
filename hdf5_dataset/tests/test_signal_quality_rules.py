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
"""
from __future__ import annotations

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
