r"""Shared pieces of the phase-1 probes: segment rows, kept UA channels, band groups, phases.

Every phase-1 script reads the same code-0/2 segments, names the 42 kept UA channels the same
way, assigns them to the same three band groups and labels raw samples with the same contraction
phase. This module holds those choices.

**Kept channels.** The model keeps 42 of the 46 stored UA channels: ``up_st`` channels 0--31 and
``up_ph`` channels 0--9, in that order (st first, then ph).

**Band groups.** The group of a channel follows its causal window $W_c$ (the stored attribute
``causal_warmup_steps``, in 4 s steps):

* ``slow``: ``st29``--``st31`` and ``ph6``--``ph9`` (windows of 107--149 steps, 7--10 min);
* ``fast``: every other kept channel with $W_c \le 16$ steps ($\le 64$ s);
* ``mid``: the rest.

**Contraction phases.** For one detected contraction with onset $o$, peak $p$ and end $e$ (raw
samples), a raw sample $n$ is in the ``peak`` phase when $|n - p| \le 10$ s, in ``rise`` when
$o \le n < p - 10$ s, in ``fall`` when $p + 10$ s $< n \le e$, and ``between`` otherwise. The
detector is :func:`teb_vae.lag_attn_cfs.eval.events.detect_contractions` on the bridged ``up``,
with validity from the recording coverage (``weight > 0``) only. The UA quality mask is NOT
passed: the detector drops every event that touches an invalid sample of ``valid``, and the
phase tables need exactly the contractions that hold invalid UA.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Output folder of phase 1.
OUT1 = S.OUT / "phase1"
#: Raw samples per segment.
N_RAW = 5280
#: Feature steps per segment.
N_STEPS = 330
#: Kept channel indices per block, in model order (st first).
KEPT_ST = list(range(32))
KEPT_PH = list(range(10))
#: Channel names of the 42 kept channels, in model order.
KEPT_NAMES = [f"st{i}" for i in KEPT_ST] + [f"ph{i}" for i in KEPT_PH]
#: Slow channels named by the prior notes.
SLOW = {"st29", "st30", "st31", "ph6", "ph7", "ph8", "ph9"}
#: Largest causal window (steps) of a ``fast`` channel.
FAST_MAX_WARMUP = 16
#: Contraction phase codes on the raw grid.
PHASES = ("between", "rise", "peak", "fall")
#: Half-width of the ``peak`` phase in seconds.
PEAK_HALF_S = 10.0
#: Gap length classes in seconds (upper bounds, inclusive).
LEN_EDGES = (4, 8, 16, 32, 64, 128)
#: The loader's contamination threshold of the strict rule.
RHO_EPS = 0.1


def split_rows(cohort: str) -> np.ndarray:
    """Rows of one shard whose recording has split code 0 or 2.

    Args:
        cohort: ``'healthy'``, ``'acidosis'`` or ``'hie'``.

    Returns:
        Sorted row indices.
    """
    with h5py.File(S.SHARDS[cohort], "r") as handle:
        guids = handle["guid"][:]
    codes = np.array([S.guid_hash(g) for g in guids])
    return np.flatnonzero(np.isin(codes, [S.FIT_CODE, S.SELECT_CODE]))


def channel_meta(cohort: str = "acidosis") -> Dict[str, np.ndarray]:
    """Causal window, delay and band group of the 42 kept channels.

    Args:
        cohort: Shard to read the attributes from (all shards share them).

    Returns:
        ``{'name', 'warmup_steps', 'delay_s', 'group'}``, each of length 42.
    """
    with h5py.File(S.SHARDS[cohort], "r") as handle:
        w_st = np.asarray(handle["up_st"].attrs["causal_warmup_steps"])[KEPT_ST]
        w_ph = np.asarray(handle["up_ph"].attrs["causal_warmup_steps"])[KEPT_PH]
        d_st = np.asarray(handle["up_st"].attrs["causal_delay_s"])[KEPT_ST]
        d_ph = np.asarray(handle["up_ph"].attrs["causal_delay_s"])[KEPT_PH]
    warmup = np.concatenate([w_st, w_ph]).astype(int)
    delay = np.concatenate([d_st, d_ph]).astype(float)
    group = np.array(["slow" if n in SLOW else ("fast" if w <= FAST_MAX_WARMUP else "mid")
                      for n, w in zip(KEPT_NAMES, warmup)])
    return {"name": np.array(KEPT_NAMES), "warmup_steps": warmup, "delay_s": delay, "group": group}


def len_class(seconds: float) -> str:
    """Name the length class of a gap.

    Args:
        seconds: Gap length in seconds.

    Returns:
        ``'<=4'`` ... ``'<=128'`` or ``'>128'``.
    """
    for edge in LEN_EDGES:
        if seconds <= edge:
            return f"<={edge}"
    return f">{LEN_EDGES[-1]}"


def detect(up: np.ndarray, weight: np.ndarray) -> Dict[str, np.ndarray]:
    """Run the contraction detector on one bridged UA trace.

    Args:
        up: Bridged UA ``(N,)``.
        weight: Step validity ``(T,)``; samples of a step with weight 0 are invalid.

    Returns:
        ``{'onset_raw', 'peak_raw', 'end_raw'}``.
    """
    from teb_vae.lag_attn_cfs.eval.events import detect_contractions, raw_validity

    valid = raw_validity(weight, decimation=S.RAW_PER_STEP, raw_len=up.size)
    return detect_contractions(up, fs=S.FS, valid=valid)


def phase_labels(n: int, events: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Label every raw sample with its contraction phase and contraction index.

    Args:
        n: Raw samples.
        events: Detector output.

    Returns:
        ``(phase, index)``: ``int8`` codes into :data:`PHASES` and the contraction index
        (``-1`` between contractions).
    """
    phase = np.zeros(n, dtype=np.int8)
    index = np.full(n, -1, dtype=np.int32)
    half = int(PEAK_HALF_S * S.FS)
    for i, (o, p, e) in enumerate(zip(events["onset_raw"], events["peak_raw"], events["end_raw"])):
        o, p, e = int(o), int(p), int(e)
        lo, hi = max(o, 0), min(e, n - 1)
        index[lo:hi + 1] = i
        phase[lo:hi + 1] = 1
        phase[max(p - half, lo):min(p + half, hi) + 1] = 2
        phase[min(p + half + 1, n):hi + 1] = 3
    return phase, index


def strict_step_bad(bits: np.ndarray, rho_st: np.ndarray, rho_ph: np.ndarray) -> np.ndarray:
    r"""The loader's strict rule on the full (untrimmed) step grid, per kept channel.

    Step $k$ of channel $c$ is bad when one of the raw samples $16k-15, \dots, 16k$ carries an
    invalid bit, or when $\rho_c[k] > 0.1$.

    Args:
        bits: ``up_quality`` ``(N,)``.
        rho_st: ``up_st_rho`` as stored ``(36, T)`` ``uint8`` or as floats in $[0, 1]$.
        rho_ph: ``up_ph_rho`` likewise ``(10, T)``.

    Returns:
        ``(T, 42)`` bool, ``True`` = bad.
    """
    steps = rho_st.shape[-1]
    bad = (bits.astype(np.int64) & S.UP_INVALID_BITS) != 0
    padded = np.concatenate([np.zeros(S.RAW_PER_STEP - 1, dtype=bool), bad])
    step_bad = padded[: steps * S.RAW_PER_STEP].reshape(steps, S.RAW_PER_STEP).any(axis=1)
    scale = 255.0 if rho_st.dtype == np.uint8 else 1.0
    rho = np.concatenate([rho_st[KEPT_ST], rho_ph[KEPT_PH]], axis=0).astype(np.float64) / scale
    return step_bad[:, None] | (rho.T > RHO_EPS)


def segment_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal runs of ``True`` (``stop`` exclusive), via the builder's own helper.

    Args:
        mask: 1-D bool.

    Returns:
        ``[(start, stop), ...]``.
    """
    from hdf5_dataset import signal_quality as sq

    return sq.runs(mask)


def recording_of(cohort: Sequence[str], guid: Sequence[str]) -> np.ndarray:
    """Recording key ``cohort/guid`` per row, for the bootstrap.

    Args:
        cohort: Cohort per row.
        guid: GUID per row.

    Returns:
        Object array of keys.
    """
    return np.array([f"{c}/{g}" for c, g in zip(cohort, guid)], dtype=object)
