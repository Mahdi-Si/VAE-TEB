r"""Shared loading, stitching and statistics of the phase-1 detector audit.

The audit reads the three local shards, keeps only recordings whose split code is 0 (fit) or 2
(select), and hands every script the same arrays. It also rebuilds the validity exactly as the
training contraction weight builds it, so the audited detector is the trained one.

**The training call.** The contraction weight runs
:func:`teb_vae.lag_attn_cfs.eval.events.detect_contractions` on the batch's ``up`` field, which is
the stored (bridged) ``up`` after the loader's z-scoring, at $f_s = 4\,$Hz, under the validity

$$
v_r = \mathbb 1[w_{\lfloor r/16 \rfloor} > 0] \wedge \mathbb 1[q_r \,\&\, 2299 = 0],
$$

with $w$ the decimated step weight and $q$ the ``up_quality`` bit code. The detector's prominence
is $\sigma$-relative, so an affine z-scoring changes nothing, and the stored units are used here.

**Segment geometry.** ``epoch`` is the segment START in seconds relative to delivery (checked:
the second segment of a pair equals the first one from the epoch difference onwards, sample by
sample). Consecutive segments start $1200\,$s apart and last $1320\,$s, so two neighbours share
$120\,$s of identical signal. A run of such neighbours stitches into one exact continuous trace.

**Bootstraps.** :func:`ratio_bootstrap` and :func:`stat_bootstrap` resample recordings, as
:func:`s7common.paired_bootstrap` does, with 2000 resamples and a 95% percentile interval.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Output folder of the detector audit.
OUT_DIR = S.OUT / "detectors"
#: Segment length in raw samples and seconds.
SEG_RAW, SEG_S = 5280, 1320.0
#: Raw samples per step.
DEC = S.RAW_PER_STEP
#: FHR invalid masks are attributes of ``fhr_quality``; these are their stored values.
FHR_INVALID_EXACT, FHR_INVALID_RULES = 11, 107
#: Hours-before-delivery bins of the segment midpoint.
HOUR_BINS: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 13.0)


@dataclass
class Segments:
    r"""Rows of the audit split, sorted by recording and start time.

    Attributes:
        cohort: Cohort name per row.
        guid: Recording GUID per row.
        code: Split code $h$ per row.
        epoch: Segment start in seconds relative to delivery (negative).
        up: Stored bridged UA $(N, 5280)$ in stored units.
        up_valid: Joint UA validity $v_r$ $(N, 5280)$, as the training weight reads it.
        step_valid: Raw expansion of $w > 0$ $(N, 5280)$.
        weight: Step weight $(N, 330)$.
        fhr: Stored bridged FHR in bpm $(N, 5280)$.
        fhr_valid: FHR validity $(N, 5280)$: step weight above 0 and no invalid quality bit.
        up_quality: UA bit codes $(N, 5280)$.
    """

    cohort: np.ndarray
    guid: np.ndarray
    code: np.ndarray
    epoch: np.ndarray
    up: np.ndarray
    up_valid: np.ndarray
    step_valid: np.ndarray
    weight: np.ndarray
    fhr: np.ndarray
    fhr_valid: np.ndarray
    up_quality: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.guid.size)

    def hours_before(self) -> np.ndarray:
        """Hours from the segment midpoint to delivery (positive)."""
        return -(self.epoch + SEG_S / 2.0) / 3600.0

    def subset(self, rows: np.ndarray) -> "Segments":
        """Return the rows ``rows`` as a new :class:`Segments`."""
        return Segments(**{k: getattr(self, k)[rows] for k in (
            "cohort", "guid", "code", "epoch", "up", "up_valid", "step_valid", "weight", "fhr",
            "fhr_valid", "up_quality")})


def load_segments(codes: Sequence[int] = (0, 2), with_fhr: bool = True) -> Segments:
    r"""Load every segment of recordings whose split code is in ``codes``.

    Args:
        codes: Allowed split codes. Never pass 1 or 3: they are the closed test set.
        with_fhr: Also load the FHR trace and its validity.

    Returns:
        The rows, sorted by (cohort, guid, epoch).

    Raises:
        ValueError: If a test code is requested.
    """
    import h5py

    if set(codes) & set(S.TEST_CODES):
        raise ValueError("codes 1 and 3 are the closed test set")
    parts: Dict[str, List[np.ndarray]] = {k: [] for k in (
        "cohort", "guid", "code", "epoch", "up", "up_valid", "step_valid", "weight", "fhr",
        "fhr_valid", "up_quality")}
    for cohort, path in S.SHARDS.items():
        with h5py.File(path, "r") as handle:
            guids = np.array([S.guid_text(g) for g in handle["guid"][:]])
            hcode = np.array([S.guid_hash(g) for g in guids])
            epoch = handle["epoch"][:].astype(np.float64)
            rows = np.where(np.isin(hcode, list(codes)))[0]
            rows = rows[np.lexsort((epoch[rows], guids[rows]))]
            read = np.sort(rows)
            back = np.searchsorted(read, rows)
            up = handle["up"][read][back].astype(np.float64)
            q = handle["up_quality"][read][back]
            w = handle["weight"][read][back].astype(np.float64)
            step_valid = np.repeat(w > 0, DEC, axis=1)[:, :SEG_RAW]
            parts["up_valid"].append(step_valid & ((q & S.UP_INVALID_BITS) == 0))
            parts["step_valid"].append(step_valid)
            parts["up"].append(up)
            parts["up_quality"].append(q)
            parts["weight"].append(w)
            if with_fhr:
                fhr = handle["fhr"][read][back].astype(np.float64)
                fq = handle["fhr_quality"][read][back].astype(np.int64)
                exact = handle["fhr_repair_exact"][read][back].astype(bool)
                bits = np.where(exact, FHR_INVALID_EXACT, FHR_INVALID_RULES)[:, None]
                parts["fhr"].append(fhr)
                parts["fhr_valid"].append(step_valid & ((fq & bits) == 0) & (fhr > 0))
            else:
                parts["fhr"].append(np.zeros((rows.size, 1)))
                parts["fhr_valid"].append(np.zeros((rows.size, 1), dtype=bool))
            parts["cohort"].append(np.full(rows.size, cohort))
            parts["guid"].append(guids[rows])
            parts["code"].append(hcode[rows])
            parts["epoch"].append(epoch[rows])
    return Segments(**{k: np.concatenate(v) for k, v in parts.items()})


def shipped_contractions(up: np.ndarray, valid: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run the shipped CFS contraction detector exactly as the training weight runs it.

    Args:
        up: One UA trace in stored units.
        valid: Its joint validity, or ``None``.

    Returns:
        ``{'onset_raw', 'peak_raw', 'end_raw'}``.
    """
    from teb_vae.lag_attn_cfs.eval.events import detect_contractions

    return detect_contractions(up, fs=float(S.FS), valid=valid)


def shipped_decelerations(fhr: np.ndarray, valid: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run the shipped deceleration detector on one bpm trace.

    Args:
        fhr: One FHR trace in bpm.
        valid: Its validity, or ``None``.

    Returns:
        ``{'onset_raw', 'nadir_raw', 'end_raw', 'depth_bpm'}``.
    """
    from teb_vae.lag_attn_cfs.eval.events import detect_decelerations

    return detect_decelerations(fhr, fs=float(S.FS), valid=valid)


def onset_weight(onsets_raw: np.ndarray, n_steps: int = 330, reach: int = 15) -> np.ndarray:
    r"""The binary training reward of one segment: $1$ on steps $o \le k \le o + \ell$.

    Args:
        onsets_raw: Onset raw indices.
        n_steps: Steps per segment.
        reach: $\ell$ in steps.

    Returns:
        A float array $(T,)$ of 0 and 1.
    """
    out = np.zeros(n_steps)
    for onset in (np.asarray(onsets_raw, dtype=np.int64) // DEC).tolist():
        out[onset: onset + reach + 1] = 1.0
    return out


# =============================================================================
# Stitching and re-cutting
# =============================================================================
def runs(seg: Segments) -> List[np.ndarray]:
    r"""Group rows into runs of consecutive segments that start exactly $1200\,$s apart.

    Args:
        seg: The loaded rows, sorted by recording and start.

    Returns:
        A list of row-index arrays, each a run of at least two rows.
    """
    out: List[np.ndarray] = []
    current = [0]
    for row in range(1, len(seg)):
        same = seg.guid[row] == seg.guid[row - 1] and abs(seg.epoch[row] - seg.epoch[row - 1] - 1200.0) < 1e-3
        if same:
            current.append(row)
        else:
            if len(current) > 1:
                out.append(np.array(current))
            current = [row]
    if len(current) > 1:
        out.append(np.array(current))
    return out


def stitch(arrays: Sequence[np.ndarray], rows: np.ndarray, per_row: int, hop: int) -> np.ndarray:
    """Stitch consecutive rows into one trace, keeping the first copy of every overlap.

    Args:
        arrays: A 2-D array whose rows are segments.
        rows: Row indices of one run.
        per_row: Length of a row.
        hop: Start-to-start distance in the row's own units.

    Returns:
        The stitched 1-D trace of length ``hop * (len(rows) - 1) + per_row``.
    """
    pieces = [arrays[r][:hop] for r in rows[:-1]] + [arrays[rows[-1]][:per_row]]
    return np.concatenate(pieces)


# =============================================================================
# Bootstraps (recording level)
# =============================================================================
def _groups(recording: Sequence[Any]) -> Tuple[np.ndarray, int]:
    _, inverse = np.unique(np.asarray(recording), return_inverse=True)
    return inverse, (int(inverse.max()) + 1 if inverse.size else 0)


def ratio_bootstrap(recording: Sequence[Any], num: Sequence[float], den: Sequence[float],
                    n_resamples: int = 2000, seed: int = 0) -> Dict[str, float]:
    r"""Recording-level bootstrap of a pooled ratio $\sum n_i / \sum d_i$.

    Args:
        recording: Recording id per row.
        num: Numerator per row.
        den: Denominator per row.
        n_resamples: Resamples.
        seed: Generator seed.

    Returns:
        ``{'mean', 'lo', 'hi', 'n_rec', 'n_rows'}``.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    inverse, n_rec = _groups(recording)
    if n_rec == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0, "n_rows": 0}
    a = np.bincount(inverse, weights=num, minlength=n_rec)
    b = np.bincount(inverse, weights=den, minlength=n_rec)
    draws = np.random.default_rng(seed).integers(0, n_rec, size=(n_resamples, n_rec))
    vals = a[draws].sum(1) / np.maximum(b[draws].sum(1), 1e-12)
    return {"mean": float(a.sum() / max(b.sum(), 1e-12)), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "n_rec": n_rec, "n_rows": int(num.size)}


def ratio_diff_bootstrap(recording: Sequence[Any], num_a: Sequence[float], den_a: Sequence[float],
                         recording_b: Sequence[Any], num_b: Sequence[float], den_b: Sequence[float],
                         n_resamples: int = 2000, seed: int = 0) -> Dict[str, float]:
    r"""Recording-level bootstrap of $\sum n^B / \sum d^B - \sum n^A / \sum d^A$, same recordings drawn.

    The two arms may have different rows (for example, the events of two detectors), but both are
    resampled with the same recording draw, so the interval is paired at the recording level.

    Args:
        recording: Recording id per row of arm A.
        num_a: Numerator per row of arm A.
        den_a: Denominator per row of arm A.
        recording_b: Recording id per row of arm B.
        num_b: Numerator per row of arm B.
        den_b: Denominator per row of arm B.
        n_resamples: Resamples.
        seed: Generator seed.

    Returns:
        ``{'mean', 'lo', 'hi', 'n_rec'}`` of the difference B minus A.
    """
    ids = np.unique(np.concatenate([np.asarray(recording, dtype=object), np.asarray(recording_b, dtype=object)]).astype(str))
    pos = {k: i for i, k in enumerate(ids)}
    sums = []
    for rec, num, den in ((recording, num_a, den_a), (recording_b, num_b, den_b)):
        g = np.array([pos[str(r)] for r in rec], dtype=np.int64)
        sums.append((np.bincount(g, weights=np.asarray(num, float), minlength=ids.size),
                     np.bincount(g, weights=np.asarray(den, float), minlength=ids.size)))
    draws = np.random.default_rng(seed).integers(0, ids.size, size=(n_resamples, ids.size))
    (na, da), (nb, db) = sums
    vals = nb[draws].sum(1) / np.maximum(db[draws].sum(1), 1e-12) - na[draws].sum(1) / np.maximum(da[draws].sum(1), 1e-12)
    return {"mean": float(nb.sum() / max(db.sum(), 1e-12) - na.sum() / max(da.sum(), 1e-12)),
            "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "n_rec": int(ids.size)}


def stat_bootstrap(recording: Sequence[Any], values: Sequence[float],
                   stat: Callable[[np.ndarray], float], n_resamples: int = 2000,
                   seed: int = 0) -> Dict[str, float]:
    r"""Recording-level bootstrap of any statistic of pooled values (a median, an IQR).

    Args:
        recording: Recording id per value.
        values: The values.
        stat: Statistic of a 1-D array.
        n_resamples: Resamples.
        seed: Generator seed.

    Returns:
        ``{'mean', 'lo', 'hi', 'n_rec', 'n_rows'}``; ``mean`` is the statistic on all values.
    """
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(values)
    values = values[keep]
    inverse, n_rec = _groups(np.asarray(recording)[keep])
    if n_rec == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_rec": 0, "n_rows": 0}
    order = np.argsort(inverse, kind="stable")
    sorted_vals = values[order]
    bounds = np.concatenate([[0], np.cumsum(np.bincount(inverse, minlength=n_rec))])
    chunks = [sorted_vals[bounds[i]:bounds[i + 1]] for i in range(n_rec)]
    rng = np.random.default_rng(seed)
    out = np.empty(n_resamples)
    for k in range(n_resamples):
        pick = rng.integers(0, n_rec, size=n_rec)
        out[k] = stat(np.concatenate([chunks[i] for i in pick]))
    return {"mean": float(stat(values)), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "n_rec": n_rec, "n_rows": int(values.size)}


def iqr(x: np.ndarray) -> float:
    """Interquartile range."""
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1)


def median(x: np.ndarray) -> float:
    """Median."""
    return float(np.median(x))


def share(pred: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], float]:
    """Make a statistic that is the share of values for which ``pred`` holds."""
    return lambda x: float(np.mean(pred(x))) if x.size else float("nan")


def match_peaks(a: np.ndarray, b: np.ndarray, tol: int) -> List[Tuple[int, int]]:
    r"""Greedy one-to-one matching of two sorted index sets by closest distance within ``tol``.

    Args:
        a: First set of indices.
        b: Second set of indices.
        tol: Largest allowed $|a_i - b_j|$.

    Returns:
        Matched ``(i, j)`` index pairs.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return []
    dist = np.abs(a[:, None] - b[None, :])
    pairs = []
    used_a, used_b = set(), set()
    for flat in np.argsort(dist, axis=None, kind="stable"):
        i, j = divmod(int(flat), b.size)
        if dist[i, j] > tol:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j))
    return pairs


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars and arrays inside a nested structure to plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
