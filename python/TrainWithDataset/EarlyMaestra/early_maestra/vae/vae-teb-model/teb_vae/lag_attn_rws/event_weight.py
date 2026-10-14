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

**The detector is the evaluation's, run exactly as the evaluation runs it.** Onsets come from
:func:`~teb_vae.lag_attn_rws.eval.events.detect_contractions` on the **raw** trace at
:data:`~teb_vae.lag_attn_rws.eval.events.FS_RAW`, under the raw validity
:func:`~teb_vae.lag_attn_rws.eval.events.raw_validity` expands from the decimated weight -- the
same call the collection pass makes to write ``seconds_since_contraction`` on its per-anchor table
-- and are then floored onto the decimated grid. A weight here and a bin there therefore name one
event by construction. (A pooled $4$-second trace fed to the same detector was measured to disagree
on about one event in six: at that rate the detector's smoothing window collapses and the
$\sigma$-relative prominence rule selects a different peak set.) The stored UP timeline is read as
it is stored.

**Onsets are a dataset constant per segment**, so a caller may pass a per-row key and a cache:
the detector then runs once per segment per process and later steps pay a lookup. The weight is a
loss weight, not a model input: the model never sees it, and the evaluation never applies it, so
every ``nll_*`` the evaluation reports remains a log-density of the unweighted block.
"""
from __future__ import annotations

from typing import Hashable, MutableMapping, Optional, Sequence

import numpy as np
import torch

from teb_vae.lag_attn_rws.eval.events import FS_RAW, detect_contractions, raw_validity

__all__ = ["contraction_anchor_weight", "contraction_onset_steps"]


def contraction_onset_steps(
    up_raw: np.ndarray, weight: np.ndarray, *, decimation: int
) -> np.ndarray:
    r"""Contraction onsets of one segment on the decimated grid, from its raw UP trace.

    Args:
        up_raw: The raw UP trace $(L_{\mathrm{raw}},)$ or longer; the first $T \cdot D$ samples
            are read. Units do not matter: the detector's prominence rule is $\sigma$-relative.
        weight: The decimated validity signal $(T,)$. Steps at weight $0$ are gaps, expanded to the
            raw grid exactly as the evaluation expands them, so an event overlapping a gap is
            dropped as the evaluation drops it.
        decimation: Raw samples per decimated step $D$.

    Returns:
        The onsets as ``int16`` decimated-step indices, ascending, possibly empty.

    Raises:
        ValueError: If the trace holds a non-finite sample. The shards are finite by construction
            and the loader adds nothing, so a NaN here is an upstream change worth failing on
            rather than a row silently scored at the floor.
    """
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
    onsets = detect_contractions(trace, fs=FS_RAW, valid=valid)["onset_raw"]
    return np.sort(np.asarray(onsets, dtype=np.int64) // int(decimation)).astype(np.int16)


def contraction_anchor_weight(
    up_raw: torch.Tensor,
    weight: torch.Tensor,
    *,
    decimation: int,
    reach_steps: int,
    floor: float,
    keys: Optional[Sequence[Hashable]] = None,
    cache: Optional[MutableMapping[Hashable, np.ndarray]] = None,
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
            runs the detector on every row.

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
    if int(up_raw.shape[0]) != batch:
        raise ValueError(
            f"up_raw has batch {int(up_raw.shape[0])} against the weight grid's {batch}"
        )
    if keys is not None and len(keys) != batch:
        raise ValueError(f"keys has {len(keys)} entries for a batch of {batch}")

    onsets_per_row: list = [None] * batch
    missing = []
    for index in range(batch):
        key = None if keys is None else keys[index]
        hit = None if (cache is None or key is None) else cache.get(key)
        if hit is None:
            missing.append(index)
        else:
            onsets_per_row[index] = hit
    if missing:
        # One device-to-host copy, of the rows the cache does not hold, and none once it holds
        # every segment: that copy is the step's only forced synchronisation point.
        rows = torch.as_tensor(missing, device=up_raw.device)
        up_np = up_raw.index_select(0, rows).detach().to(device="cpu", dtype=torch.float64).numpy()
        weight_np = (
            weight.index_select(0, rows).detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        for position, index in enumerate(missing):
            onsets = contraction_onset_steps(
                up_np[position], weight_np[position], decimation=decimation
            )
            onsets_per_row[index] = onsets
            if cache is not None and keys is not None and keys[index] is not None:
                cache[keys[index]] = onsets
    out = np.full((batch, steps), floor, dtype=np.float32)
    for index in range(batch):
        for onset in onsets_per_row[index].tolist():
            out[index, onset : onset + reach_steps + 1] = 1.0
    return torch.from_numpy(out).to(device=weight.device, dtype=weight.dtype)
