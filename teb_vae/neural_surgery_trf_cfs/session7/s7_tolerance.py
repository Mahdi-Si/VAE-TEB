r"""A tolerance rule for bridged UA: which invalid runs the flags and the reward may ignore.

The dataset builder fills every invalid UA run with a bridge and marks the run in the raw
validity mask. Two consumers read that mask, and at epoch 424 both treat a 4-second run like a
4-minute one:

* the per-channel validity flags $v_{s,c}$ of the UA adapter (a channel is flagged at step $s$
  when more than 10% of its filter window rests on invalid samples);
* the contraction reward $\omega_t$, which drops every contraction whose span touches one invalid
  sample.

This module builds tolerant versions of both from the raw mask alone.

**Short runs.** An invalid run of length $n \le T_{\mathrm{tol}}$ is *short*; every other run is
*long*.

**The causal guard.** The bridge across a run $[a, b]$ is drawn toward the samples after $b$, and
its end slope reads $g$ further samples (the builder's slope window). A coefficient at step $s$
that rests on bridged samples of a run with $b + g$ beyond the end of step $s$ therefore contains
UA from after step $s$. Such a run is *running* at $s$. A short run is tolerated at step $s$ only
after it has stopped running,

$$\mathrm{running}(s) = \exists\, \text{short run } [a, b]:\; a < D(s+1) \le b + g + 1 ,$$

with $D$ raw samples per step. While a run is running, the flags stay as the builder made them
and the anchor earns no reward.

**Tolerant flags.** With $W_c$ the causal window of channel $c$ in steps,

$$v'_{s,c} = v_{s,c} \;\lor\; \Bigl(\text{no long-run sample in } [D(s+1-W_c),\, D(s+1)) \;\land\;
  \lnot\,\mathrm{running}(s)\Bigr).$$

The rule only ever turns an invalid flag valid. It is stricter than a recomputed $\rho$, because
it asks for zero long-run samples in the window instead of at most 10% of the window weight.

**Tolerant reward.** The detector reads a raw validity in which short runs count as valid, so it
drops a contraction only when its span touches a long run. The anchor weight is then closed at
every anchor where a short run is running.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S


def declared_windows() -> np.ndarray:
    r"""Causal window $W_c$ of every DECLARED UA channel in feature steps, from the shard itself.

    The builder stores each channel's warm-up on the untrimmed grid (attribute
    ``causal_warmup_steps`` of ``up_st`` and ``up_ph``). That value is the number of steps a
    channel needs before its coefficient rests only on recorded samples, so it is also the reach
    of an invalid sample into later steps. The model's own warm-up vector is not used here: it
    is counted on the trimmed grid, where the fast channels read 0.

    Returns:
        ``(c_u,)`` integers, scattering channels first, each at least 1.
    """
    import h5py

    with h5py.File(S.SHARDS["acidosis"], "r") as handle:
        steps = np.concatenate([np.asarray(handle[name].attrs["causal_warmup_steps"], dtype=np.int64)
                                for name in ("up_st", "up_ph")])
    return np.maximum(steps, 1)


def run_table(invalid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Start and end (inclusive) of every invalid run of one row.

    Args:
        invalid: Boolean ``(L,)``, ``True`` = invalid raw sample.

    Returns:
        ``(starts, ends)`` as integer arrays, ascending.
    """
    padded = np.concatenate([[False], invalid, [False]]).astype(np.int8)
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1


def split_runs(up_valid: np.ndarray, tolerance_s: float, guard_s: float = 4.0,
               decimation: int = S.RAW_PER_STEP) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Split the invalid samples of a batch into long-run samples, short-run samples and running steps.

    Args:
        up_valid: Raw UA validity ``(B, L)``, ``True`` = valid.
        tolerance_s: $T_{\mathrm{tol}}$ in seconds; a run at most this long is short.
        guard_s: The bridge's slope window $g$ in seconds.
        decimation: Raw samples per feature step $D$.

    Returns:
        ``(long_bad, short_bad, running)``: boolean ``(B, L)``, ``(B, L)`` and ``(B, L // D)``.
        ``running[b, s]`` is ``True`` when a short run is running at step $s$.
    """
    invalid = ~np.asarray(up_valid, dtype=bool)
    batch, length = invalid.shape
    steps = length // decimation
    long_bad = np.zeros_like(invalid)
    short_bad = np.zeros_like(invalid)
    running = np.zeros((batch, steps), dtype=bool)
    limit = int(round(tolerance_s * S.FS))
    guard = int(round(guard_s * S.FS))
    for row in range(batch):
        starts, ends = run_table(invalid[row])
        for a, b in zip(starts.tolist(), ends.tolist()):
            if b - a + 1 > limit:
                long_bad[row, a:b + 1] = True
                continue
            short_bad[row, a:b + 1] = True
            # Step s is affected when a < D(s+1) and b + g + 1 >= D(s+1).
            first = a // decimation
            last = min(steps - 1, (b + guard + 1) // decimation - 1)
            if last >= first:
                running[row, first:last + 1] = True
    return long_bad, short_bad, running


def tolerant_flags(flags: torch.Tensor, up_valid: torch.Tensor, window_steps: torch.Tensor,
                   tolerance_s: float, guard_s: float = 4.0,
                   decimation: int = S.RAW_PER_STEP, guard_running: bool = True) -> torch.Tensor:
    r"""Apply the tolerance rule to the per-channel validity flags.

    Args:
        flags: The builder's flags ``(B, T, C)``, boolean, ``True`` = usable.
        up_valid: Raw UA validity ``(B, L)`` on the same trimmed grid, $L = D\,T$.
        window_steps: The causal window $W_c$ of each channel in steps, ``(C,)``.
        tolerance_s: $T_{\mathrm{tol}}$ in seconds. ``0`` returns ``flags`` unchanged.
        guard_s: The bridge's slope window in seconds.
        decimation: Raw samples per feature step.
        guard_running: Keep the builder's flags at steps where a short run is running. Set it to
            ``False`` when the look-ahead is handled per anchor by :func:`removed_keys`; the flags
            of the steps inside a short run then turn valid too.

    Returns:
        The tolerant flags ``(B, T, C)``, boolean.
    """
    if tolerance_s <= 0:
        return flags
    long_bad, _, running = split_runs(up_valid.cpu().numpy(), tolerance_s, guard_s, decimation)
    batch, steps, _ = flags.shape
    cumulative = np.concatenate([np.zeros((batch, 1), dtype=np.int64), np.cumsum(long_bad, axis=1)], axis=1)
    end = (np.arange(steps) + 1) * decimation                                           # (T,)
    window = np.asarray(window_steps.cpu().numpy(), dtype=np.int64)                    # (C,)
    start = np.clip(end[:, None] - window[None, :] * decimation, 0, None)              # (T, C)
    count = cumulative[:, end][:, :, None] - cumulative[np.arange(batch)[:, None, None], start[None]]
    clean = (count == 0) & ~running[:, :, None] if guard_running else (count == 0)
    return flags | torch.from_numpy(clean).to(flags.device)


def tolerant_reward_validity(up_valid: torch.Tensor, tolerance_s: float) -> torch.Tensor:
    """The raw validity the contraction detector reads under the tolerance rule.

    Args:
        up_valid: Raw UA validity ``(B, L)``.
        tolerance_s: $T_{\\mathrm{tol}}$ in seconds.

    Returns:
        Boolean ``(B, L)``: valid samples plus the samples of short runs.
    """
    if tolerance_s <= 0:
        return up_valid
    _, short_bad, _ = split_runs(up_valid.cpu().numpy(), tolerance_s)
    return up_valid | torch.from_numpy(short_bad).to(up_valid.device)


def running_steps(up_valid: torch.Tensor, tolerance_s: float, guard_s: float = 4.0,
                  decimation: int = S.RAW_PER_STEP) -> torch.Tensor:
    """Steps at which a short run is running; the reward is closed there.

    Args:
        up_valid: Raw UA validity ``(B, L)``.
        tolerance_s: $T_{\\mathrm{tol}}$ in seconds.
        guard_s: The bridge's slope window in seconds.
        decimation: Raw samples per feature step.

    Returns:
        Boolean ``(B, L // D)``.
    """
    _, _, running = split_runs(up_valid.cpu().numpy(), tolerance_s, guard_s, decimation)
    return torch.from_numpy(running).to(up_valid.device)


def removed_keys(up_valid: torch.Tensor, num_lags: int, guard_s: float = 4.0,
                 decimation: int = S.RAW_PER_STEP) -> torch.Tensor:
    r"""Keys that an anchor must not read, because their value rests on UA after the anchor.

    The bridge across an invalid run $[a, b]$ depends on samples up to $b + g$. Every coefficient
    at a step $s$ with $a < D(s+1)$ can rest on that bridge. For an anchor $t$ at which the run is
    still running, $b + g + 1 \ge D(t+1)$, those coefficients contain UA from after the anchor, so
    the key at lag $\ell = t - s$ is removed:

    $$R_{t\ell} = \exists\,[a, b]:\; a < D(t - \ell + 1) \;\land\; b + g + 1 \ge D(t+1).$$

    The rule covers runs of every length. Once a run has stopped running, its keys return.

    Args:
        up_valid: Raw UA validity ``(B, L)``, ``True`` = valid.
        num_lags: The number of lags $L$ of the attention.
        guard_s: The bridge's slope window $g$ in seconds.
        decimation: Raw samples per feature step $D$.

    Returns:
        Boolean ``(B, T, L)`` in lag order, ``True`` = removed.
    """
    invalid = ~np.asarray(up_valid.cpu().numpy(), dtype=bool)
    batch, length = invalid.shape
    steps = length // decimation
    guard = int(round(guard_s * S.FS))
    removed = np.zeros((batch, steps, num_lags), dtype=bool)
    for row in range(batch):
        starts, ends = run_table(invalid[row])
        for a, b in zip(starts.tolist(), ends.tolist()):
            first = a // decimation
            last = min(steps - 1, (b + guard + 1) // decimation - 1)
            for anchor in range(first, last + 1):
                reach = min(anchor - first, num_lags - 1)
                removed[row, anchor, :reach + 1] = True
    return torch.from_numpy(removed).to(up_valid.device)
