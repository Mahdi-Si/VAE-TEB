r"""Per-anchor weights for the full branch, from the contraction timeline of the raw UP trace.

The coupling the source-conditioned branch exists to measure is sparse in time: a uterine
contraction can move the fetal heart rate, and between contractions the source has nothing to add
that the target's own past does not already carry. One $\beta$ applied at every anchor cannot tell
the two regimes apart, so this module builds a weight $w_t$ on the dense step grid that is $1$ at
every anchor a recent contraction could still be informing and a configured floor
$w_{\min} \in [0, 1)$ elsewhere,

$$
w_t = \begin{cases} 1 & \exists\, o \in \mathcal O : o \le t \le o + \ell \\
w_{\min} & \text{otherwise,} \end{cases}
$$

where $\mathcal O$ is the set of contraction onsets on the decimated grid and $\ell$ the reach in
steps. The objective applies it to the **full** branch's reconstruction alone
(:func:`~teb_vae.lag_attn_rws.nets.losses.compute_loss`), so between contractions the source
pathway has its KL cost and no reward, and its residual is pushed to zero there, while at active
anchors the reward is undiluted.

**The detector is the cell's own evaluation detector, run exactly as that evaluation runs it.**
The caller supplies it: each task hands over the ``detect_contractions`` its own collection pass
calls to write ``seconds_since_contraction`` on the per-anchor table (the raw-signal cells'
:mod:`teb_vae.lag_attn_rws.eval.events`; the causal-feature cells'
:mod:`teb_vae.lag_attn_cfs.eval.events`, which additionally drops a peak whose onset walk finds no
rising edge). It runs on the **raw** trace at :data:`~teb_vae.lag_attn_rws.eval.events.FS_RAW`,
under the raw validity :func:`~teb_vae.lag_attn_rws.eval.events.raw_validity` expands from the
decimated weight, and its onsets are floored onto the decimated grid. A weight here and a bin on
that cell's table therefore name one event by construction. (A pooled $4$-second trace fed to the
same detector was measured to disagree on about one event in six: at that rate the detector's
smoothing window collapses and the $\sigma$-relative prominence rule selects a different peak set.)
The stored UP timeline is read as it is stored.

**On a signal-quality build the detector can also read the UA mask** (``raw_valid``, the loader's
``up_valid``). An invalid UA sample is then a gap exactly as an uncovered step is: the detector
interpolates across it and drops every event whose span touches it, so a contraction drawn by the
bridge across removed UA, or one whose rise holds an artifact, weighs nothing. The collection pass
passes the same mask, so the weight and the table still name one event.

**Onsets are a dataset constant per segment**, so a caller may pass a per-row key and a cache:
the detector then runs once per segment per process and later steps pay a lookup. The weight is a
loss weight, not a model input: the model never sees it, and the evaluation never applies it, so
every ``nll_*`` the evaluation reports remains a log-density of the unweighted block.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, MutableMapping, Optional, Sequence

import numpy as np
import torch

from teb_vae.lag_attn_rws.eval.events import FS_RAW, detect_contractions, raw_validity

__all__ = [
    "contraction_anchor_weight",
    "contraction_onset_steps",
    "contraction_source_mask",
    "contraction_span_steps",
]


def _detect_events(
    up_raw: np.ndarray,
    weight: np.ndarray,
    *,
    decimation: int,
    detector: Callable[..., Dict[str, Any]],
    raw_valid: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Run the detector on one segment under the joint validity; see :func:`contraction_onset_steps`."""
    steps = int(weight.shape[0])
    raw_len = steps * int(decimation)
    trace = np.asarray(up_raw, dtype=np.float64)[:raw_len]
    if trace.shape[0] != raw_len:
        raise ValueError(
            f"up_raw has {trace.shape[0]} samples but the weight grid needs {raw_len} "
            f"({steps} steps x decimation {int(decimation)})"
        )
    if not np.isfinite(trace).all():
        raise ValueError(
            "up_raw holds a non-finite sample; the stored UP is finite by construction, so this is "
            "an upstream change rather than a row to score at the floor"
        )
    valid = raw_validity(weight, decimation=int(decimation), raw_len=raw_len)
    if raw_valid is not None:
        ua = np.asarray(raw_valid, dtype=bool).reshape(-1)[:raw_len]
        if ua.shape[0] != raw_len:
            raise ValueError(
                f"raw_valid has {ua.shape[0]} samples but the weight grid needs {raw_len}; a UA "
                f"mask padded with True would mark uncovered samples valid"
            )
        valid = valid & ua
    return detector(trace, fs=FS_RAW, valid=valid)


def contraction_span_steps(
    up_raw: np.ndarray,
    weight: np.ndarray,
    *,
    decimation: int,
    detector: Callable[..., Dict[str, Any]] = detect_contractions,
    raw_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Contraction spans of one segment on the decimated grid: ``(n, 2)`` onset and end steps.

    The same detection as :func:`contraction_onset_steps`; each raw index is floored onto the
    decimated grid, so step $k$ is in a span when $o \le k \le e$.

    Args:
        up_raw: The raw UP trace; see :func:`contraction_onset_steps`.
        weight: The decimated validity signal $(T,)$.
        decimation: Raw samples per decimated step $D$.
        detector: The ``detect_contractions`` to run.
        raw_valid: Optional per-raw-sample UA validity, ANDed into the detector's validity.

    Returns:
        ``int16`` ``(n, 2)`` array of ``(onset, end)`` steps, ascending by onset.
    """
    found = _detect_events(
        up_raw, weight, decimation=decimation, detector=detector, raw_valid=raw_valid
    )
    onsets = np.asarray(found["onset_raw"], dtype=np.int64) // int(decimation)
    ends = np.asarray(found["end_raw"], dtype=np.int64) // int(decimation)
    order = np.argsort(onsets, kind="stable")
    return np.stack([onsets[order], ends[order]], axis=1).astype(np.int16).reshape(-1, 2)


def contraction_onset_steps(
    up_raw: np.ndarray,
    weight: np.ndarray,
    *,
    decimation: int,
    detector: Callable[..., Dict[str, Any]] = detect_contractions,
    raw_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Contraction onsets of one segment on the decimated grid, from its raw UP trace.

    Args:
        up_raw: The raw UP trace $(L_{\mathrm{raw}},)$ or longer; the first $T \cdot D$ samples
            are read. Units do not matter: the detector's prominence rule is $\sigma$-relative.
        weight: The decimated validity signal $(T,)$. Steps at weight $0$ are gaps, expanded to the
            raw grid exactly as the evaluation expands them, so an event overlapping a gap is
            dropped as the evaluation drops it.
        decimation: Raw samples per decimated step $D$.
        detector: The ``detect_contractions`` to run -- the one the calling cell's evaluation uses,
            so a training weight and an evaluation bin name one event. Defaults to the raw-signal
            cells' detector.
        raw_valid: Optional per-raw-sample UA validity (the loader's ``up_valid``), ANDed into
            the validity the detector reads. ``None`` -- the default -- is the weight alone.

    Returns:
        The onsets as ``int16`` decimated-step indices, ascending, possibly empty.

    Raises:
        ValueError: If the trace holds a non-finite sample. The shards are finite by construction
            and the loader adds nothing, so a NaN here is an upstream change worth failing on
            rather than a row silently scored at the floor. Also if ``raw_valid`` is shorter than
            the grid.
    """
    onsets = _detect_events(
        up_raw, weight, decimation=decimation, detector=detector, raw_valid=raw_valid
    )["onset_raw"]
    return np.sort(np.asarray(onsets, dtype=np.int64) // int(decimation)).astype(np.int16)


def contraction_source_mask(
    up_raw: torch.Tensor,
    weight: torch.Tensor,
    *,
    decimation: int,
    raw_valid: Optional[torch.Tensor] = None,
    keys: Optional[Sequence[Hashable]] = None,
    cache: Optional[MutableMapping[Hashable, np.ndarray]] = None,
    detector: Callable[..., Dict[str, Any]] = detect_contractions,
) -> torch.Tensor:
    r"""The $(B, T)$ steps that lie inside a detected contraction, $o \le k \le e$.

    The source-side counterpart of :func:`contraction_anchor_weight`, for the ablation that keeps a
    lag key only while the uterus contracts (``source_event_mask``). Same detector, same validity,
    same key-and-cache contract; the cache holds spans, so it must not be shared with the onset
    cache of :func:`contraction_anchor_weight`.

    Args:
        up_raw: The raw UP trace $(B, L_{\mathrm{raw}})$.
        weight: The decimated validity signal $(B, T)$.
        decimation: Raw samples per decimated step $D$.
        raw_valid: Optional per-raw-sample UA validity $(B, L_{\mathrm{raw}})$.
        keys: One hashable key per row, or ``None``.
        cache: A span cache keyed by ``keys``, or ``None``.
        detector: The ``detect_contractions`` to run.

    Returns:
        Boolean $(B, T)$ on ``weight``'s device, ``True`` inside a contraction.
    """
    batch, steps = int(weight.shape[0]), int(weight.shape[1])
    spans = _cached_rows(
        up_raw, weight, raw_valid, keys, cache,
        lambda trace, row_weight, row_valid: contraction_span_steps(
            trace, row_weight, decimation=decimation, detector=detector, raw_valid=row_valid
        ),
    )
    out = np.zeros((batch, steps), dtype=bool)
    for index in range(batch):
        for onset, end in spans[index].tolist():
            out[index, onset : end + 1] = True
    return torch.from_numpy(out).to(device=weight.device)


def _cached_rows(
    up_raw: torch.Tensor,
    weight: torch.Tensor,
    raw_valid: Optional[torch.Tensor],
    keys: Optional[Sequence[Hashable]],
    cache: Optional[MutableMapping[Hashable, np.ndarray]],
    compute: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray],
) -> list:
    """Per-row detector results, read from *cache* where present and computed and stored otherwise.

    One device-to-host copy, of the rows the cache does not hold, and none once it holds every
    segment: that copy is the step's only forced synchronisation point.
    """
    batch = int(weight.shape[0])
    if int(up_raw.shape[0]) != batch:
        raise ValueError(
            f"up_raw has batch {int(up_raw.shape[0])} against the weight grid's {batch}"
        )
    if raw_valid is not None and int(raw_valid.shape[0]) != batch:
        raise ValueError(
            f"raw_valid has batch {int(raw_valid.shape[0])} against the weight grid's {batch}"
        )
    if keys is not None and len(keys) != batch:
        raise ValueError(f"keys has {len(keys)} entries for a batch of {batch}")
    rows: list = [None] * batch
    missing = []
    for index in range(batch):
        key = None if keys is None else keys[index]
        hit = None if (cache is None or key is None) else cache.get(key)
        if hit is None:
            missing.append(index)
        else:
            rows[index] = hit
    if missing:
        picked = torch.as_tensor(missing, device=up_raw.device)
        up_np = up_raw.index_select(0, picked).detach().to(device="cpu", dtype=torch.float64).numpy()
        weight_np = (
            weight.index_select(0, picked.to(weight.device)).detach()
            .to(device="cpu", dtype=torch.float64).numpy()
        )
        valid_np = (
            None if raw_valid is None
            else raw_valid.index_select(0, picked.to(raw_valid.device)).detach()
            .to(device="cpu", dtype=torch.bool).numpy()
        )
        for position, index in enumerate(missing):
            result = compute(
                up_np[position], weight_np[position],
                None if valid_np is None else valid_np[position],
            )
            rows[index] = result
            if cache is not None and keys is not None and keys[index] is not None:
                cache[keys[index]] = result
    return rows


def contraction_anchor_weight(
    up_raw: torch.Tensor,
    weight: torch.Tensor,
    *,
    decimation: int,
    reach_steps: int,
    floor: float,
    keys: Optional[Sequence[Hashable]] = None,
    cache: Optional[MutableMapping[Hashable, np.ndarray]] = None,
    detector: Callable[..., Dict[str, Any]] = detect_contractions,
    raw_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Build the $(B, T)$ full-branch anchor weight from the raw UP trace.

    Args:
        up_raw: The raw UP trace $(B, L_{\mathrm{raw}})$, loader-normalised or not. Only the first
            $T \cdot D$ samples of each row are read.
        weight: The decimated validity signal $(B, T)$.
        decimation: Raw samples per decimated step $D$ ($16$ at the production geometry).
        reach_steps: $\ell$, how many steps after an onset an anchor still counts as informed.
        floor: $w_{\min}$, the weight of every other anchor, in $[0, 1)$. $0$ removes those anchors
            from the full branch's reconstruction entirely.
        keys: One hashable key per row -- the segment's ``(guid, epoch)`` -- or ``None``. With a
            ``cache``, a row whose key is present skips the detector and reads its onsets back.
        cache: A mapping the onsets are read from and written to, keyed by ``keys``. ``None``
            runs the detector on every row. A cache must be fed one detector only; the tasks keep
            one per module.
        detector: The ``detect_contractions`` to run; see :func:`contraction_onset_steps`.
        raw_valid: Optional per-raw-sample UA validity $(B, L_{\mathrm{raw}})$, the loader's
            ``up_valid``. An event that touches an invalid UA sample is dropped. ``None`` -- the
            default -- is the weight alone, bitwise the earlier behaviour. A cache must be fed one
            choice only.

    Returns:
        The weight $(B, T)$ on ``weight``'s device and dtype, every entry $w_{\min}$ or $1$.

    Raises:
        ValueError: If ``floor`` is outside $[0, 1)$, ``reach_steps`` is negative, ``keys`` is
            given with a length other than $B$, or a trace is shorter than the $T \cdot D$ samples
            the grid needs or holds a non-finite sample.
    """
    floor = float(floor)
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}; 1.0 would be the unweighted branch")
    reach_steps = int(reach_steps)
    if reach_steps < 0:
        raise ValueError(f"reach_steps must be >= 0, got {reach_steps}")
    batch, steps = int(weight.shape[0]), int(weight.shape[1])
    onsets_per_row = _cached_rows(
        up_raw, weight, raw_valid, keys, cache,
        lambda trace, row_weight, row_valid: contraction_onset_steps(
            trace, row_weight, decimation=decimation, detector=detector, raw_valid=row_valid
        ),
    )
    out = np.full((batch, steps), floor, dtype=np.float32)
    for index in range(batch):
        for onset in onsets_per_row[index].tolist():
            out[index, onset : onset + reach_steps + 1] = 1.0
    return torch.from_numpy(out).to(device=weight.device, dtype=weight.dtype)
