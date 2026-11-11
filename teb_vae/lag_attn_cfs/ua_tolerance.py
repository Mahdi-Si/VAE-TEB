r"""Which bridged UA the model may read: per-anchor key removal, tolerant flags, tolerant reward.

The dataset builder fills every invalid UA run with a bridge and marks the run in the raw
validity mask ``up_valid`` $(B, L)$, $L = D\,T$ raw samples on the trimmed grid, $D$ raw samples
per feature step. Three rules read that mask, and this module holds the pure functions behind
all three.

**Key removal** (model keyword ``source_running_gap_keys: remove``). The bridge across an invalid
run $[a, b]$ is drawn toward the samples after $b$, and its end slope reads $g$ further samples
(the builder's bridge slope window, ``ua_bridge_guard_s``). A coefficient at step $s$ with
$a < D(s+1)$ can therefore rest on UA after step $s$ while the run is still running. For an
anchor $t$ and the key at lag $\ell$, $s = t - \ell$,

$$R_{t\ell} = \exists\,[a, b]:\; a < D(t - \ell + 1) \;\land\; b + g + 1 \ge D(t + 1),$$

for runs of every length. A removed key leaves that anchor's attention; once the run has stopped
running, its keys return.

**Tolerant flags** (task key ``source_valid_tolerance_s``). An invalid run of
$n \le T_{\mathrm{tol}} f_s$ samples is *short*; every other run is *long*. With $W_c$ the causal
window of channel $c$ in feature steps on the untrimmed grid,

$$v'_{s,c} = v_{s,c} \;\lor\; \bigl(\text{no long-run sample in } [D(s+1-W_c),\, D(s+1))\bigr).$$

There is no exception for the steps inside a short run, because key removal already takes every
key whose value can read past its anchor out of the attention. The rule only turns a flag valid.

**Tolerant reward** (task key ``event_weight_ua_tolerance_s``). The contraction detector reads a
raw validity in which the samples of short runs count as valid, so a contraction is dropped only
when its span touches a long run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

#: Raw UA samples per second on the stored grid. It is the builder's quality sampling rate
#: (``QualityConfig.fs`` in ``hdf5_dataset/signal_quality.py``); the network knows only the
#: decimation $D$, not seconds, so the conversion of the three second-valued keys lives here.
RAW_FS_HZ: float = 4.0

#: The two values of the model keyword ``source_running_gap_keys``. ``'keep'`` is the model
#: before the keyword existed; ``'remove'`` applies :func:`removed_keys`.
RUNNING_GAP_MODES: Tuple[str, ...] = ("keep", "remove")


def seconds_to_samples(seconds: float, fs: float = RAW_FS_HZ) -> int:
    r"""Convert a duration to whole raw samples, $\mathrm{round}(t f_s)$.

    Args:
        seconds: The duration in seconds.
        fs: The raw sampling rate in Hz.

    Returns:
        The number of raw samples.
    """
    return int(round(float(seconds) * float(fs)))


def run_table(invalid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Row, start and end (inclusive) of every invalid run of a batch.

    Each row is padded with a valid sample on both sides before the change points are taken, so a
    run never crosses from one row into the next.

    Args:
        invalid: Boolean $(B, L)$, ``True`` = invalid raw sample.

    Returns:
        ``(rows, starts, ends)``: integer arrays of equal length, ascending within each row.
    """
    invalid = np.asarray(invalid, dtype=bool)
    if invalid.ndim == 1:
        invalid = invalid[None]
    batch = int(invalid.shape[0])
    padded = np.zeros((batch, int(invalid.shape[1]) + 2), dtype=np.int8)
    padded[:, 1:-1] = invalid
    change = np.diff(padded, axis=1)
    rows, starts = np.nonzero(change == 1)
    end_rows, ends = np.nonzero(change == -1)
    # np.nonzero walks row-major, so the k-th start and the k-th end belong to one run.
    assert np.array_equal(rows, end_rows)
    return rows, starts, ends - 1


def split_runs(
    up_valid: torch.Tensor, tolerance_s: float, fs: float = RAW_FS_HZ
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Split the invalid samples of a batch into long-run and short-run samples.

    A run of $n$ samples is short when $n \le \mathrm{round}(T_{\mathrm{tol}} f_s)$.

    Args:
        up_valid: Raw UA validity $(B, L)$, ``True`` = valid.
        tolerance_s: $T_{\mathrm{tol}}$ in seconds.
        fs: The raw sampling rate in Hz.

    Returns:
        ``(long_bad, short_bad)``: boolean $(B, L)$ on the device of ``up_valid``.
    """
    invalid = ~up_valid.detach().to(torch.bool).cpu().numpy()
    batch, length = invalid.shape
    rows, starts, ends = run_table(invalid)
    is_long = (ends - starts + 1) > seconds_to_samples(tolerance_s, fs)
    long_bad = np.zeros((batch, length), dtype=bool)
    short_bad = np.zeros((batch, length), dtype=bool)
    # A +1 at each run start and a -1 one past its end; the running sum is 1 exactly inside.
    for target, chosen in ((long_bad, is_long), (short_bad, ~is_long)):
        marks = np.zeros((batch, length + 1), dtype=np.int32)
        np.add.at(marks, (rows[chosen], starts[chosen]), 1)
        np.add.at(marks, (rows[chosen], ends[chosen] + 1), -1)
        target[:] = np.cumsum(marks, axis=1)[:, :length] > 0
    device = up_valid.device
    return torch.from_numpy(long_bad).to(device), torch.from_numpy(short_bad).to(device)


def tolerant_flags(
    flags: torch.Tensor,
    up_valid: torch.Tensor,
    window_steps: torch.Tensor,
    tolerance_s: float,
    decimation: int,
    fs: float = RAW_FS_HZ,
) -> torch.Tensor:
    r"""Apply the tolerance rule to the per-channel validity flags.

    $v'_{s,c} = v_{s,c} \lor (\text{no long-run sample in } [D(s+1-W_c), D(s+1)))$. A window that
    starts before the trimmed grid is cut at sample $0$.

    Args:
        flags: The dataset's flags $(B, T, C)$, boolean, ``True`` = usable, at the declared
            source width.
        up_valid: Raw UA validity $(B, L)$ on the same trimmed grid, $L = D\,T$.
        window_steps: $W_c$ per declared channel in feature steps, $(C,)$, each at least $1$.
            It must be counted on the **untrimmed** grid: on the trimmed grid the fast channels
            read $0$, and a window of $0$ steps would turn their flags valid whatever the UA did.
        tolerance_s: $T_{\mathrm{tol}}$ in seconds; a value $\le 0$ returns ``flags`` unchanged.
        decimation: $D$, raw samples per feature step.
        fs: The raw sampling rate in Hz.

    Returns:
        The tolerant flags $(B, T, C)$, boolean.

    Raises:
        ValueError: If the raw grid is not $D\,T$ samples long or the window vector is not $(C,)$.
    """
    if tolerance_s is None or float(tolerance_s) <= 0:
        return flags
    batch, steps, width = (int(n) for n in flags.shape)
    _check_raw_grid(up_valid, steps, decimation)
    window = torch.as_tensor(window_steps, dtype=torch.long, device=flags.device).reshape(-1)
    if int(window.numel()) != width:
        raise ValueError(
            f"window_steps has {int(window.numel())} entries against {width} flag channels; the "
            f"windows are positional against the declared source channels."
        )
    long_bad, _ = split_runs(up_valid, tolerance_s, fs)
    long_bad = long_bad.to(flags.device)
    cumulative = torch.cat(
        [long_bad.new_zeros((batch, 1), dtype=torch.long), long_bad.long().cumsum(dim=1)], dim=1
    )                                                                          # (B, L + 1)
    end = (torch.arange(steps, device=flags.device) + 1) * int(decimation)     # (T,)
    start = (end[:, None] - window.clamp_min(1)[None, :] * int(decimation)).clamp_min(0)
    count = cumulative[:, end][:, :, None] - cumulative[:, start]              # (B, T, C)
    return flags.to(torch.bool) | (count == 0)


def tolerant_reward_validity(
    up_valid: torch.Tensor, tolerance_s: Optional[float], fs: float = RAW_FS_HZ
) -> torch.Tensor:
    r"""The raw validity the contraction detector reads under the tolerance rule.

    Args:
        up_valid: Raw UA validity $(B, L)$.
        tolerance_s: $T_{\mathrm{tol}}$ in seconds; ``None`` or $\le 0$ returns ``up_valid``.
        fs: The raw sampling rate in Hz.

    Returns:
        Boolean $(B, L)$: the valid samples plus the samples of short runs.
    """
    if tolerance_s is None or float(tolerance_s) <= 0:
        return up_valid
    _, short_bad = split_runs(up_valid, tolerance_s, fs)
    return up_valid.to(torch.bool) | short_bad


def removed_keys(
    up_valid: torch.Tensor,
    num_steps: int,
    num_lags: int,
    decimation: int,
    guard_s: float = 4.0,
    fs: float = RAW_FS_HZ,
) -> torch.Tensor:
    r"""Keys that an anchor must not read, because their value rests on UA after the anchor.

    $$R_{t\ell} = \exists\,[a, b]:\; a < D(t - \ell + 1) \;\land\; b + g + 1 \ge D(t + 1).$$

    Each run $[a, b]$ covers the anchors $t \in [\lfloor a/D \rfloor,\,
    \lfloor (b + g + 1)/D \rfloor - 1]$, and at a covered anchor it removes the lags
    $0 \le \ell \le t - \lfloor a/D \rfloor$. With $m_t$ the smallest $\lfloor a/D \rfloor$ over
    the runs covering $t$, $R_{t\ell} = [\ell \le t - m_t]$. Within a row both interval ends are
    non-decreasing in the run index, so $m_t$ is the start of the first run whose end reaches $t$.

    Args:
        up_valid: Raw UA validity $(B, L)$, ``True`` = valid, $L = D\,T$.
        num_steps: $T$, the feature steps of the source stream.
        num_lags: $L_{\mathrm{lag}}$, the lags of the attention.
        decimation: $D$, raw samples per feature step.
        guard_s: $g$ in seconds, the bridge's slope window.
        fs: The raw sampling rate in Hz.

    Returns:
        Boolean $(B, T, L_{\mathrm{lag}})$ in lag order on the device of ``up_valid``,
        ``True`` = removed.

    Raises:
        ValueError: If the raw grid is not $D\,T$ samples long.
    """
    steps = int(num_steps)
    _check_raw_grid(up_valid, steps, decimation)
    invalid = ~up_valid.detach().to(torch.bool).cpu().numpy()
    batch = int(invalid.shape[0])
    guard = seconds_to_samples(guard_s, fs)
    rows, starts, ends = run_table(invalid)
    first = starts // int(decimation)
    last = np.minimum(steps - 1, (ends + guard + 1) // int(decimation) - 1)
    covers = last >= first
    anchor = np.arange(steps)
    # The default m_t = t + 1 makes t - m_t = -1, which removes no lag.
    reach = np.full((batch, steps), -1, dtype=np.int64)
    for row in np.unique(rows[covers]):
        chosen = covers & (rows == row)
        row_first, row_last = first[chosen], last[chosen]
        index = np.searchsorted(row_last, anchor, side="left")
        inside = index < row_first.size
        smallest = np.where(inside, row_first[np.minimum(index, row_first.size - 1)], steps)
        reach[row] = np.where(inside & (smallest <= anchor), anchor - smallest, -1)
    reach_t = torch.from_numpy(reach).to(up_valid.device)
    lags = torch.arange(int(num_lags), device=up_valid.device)
    return lags[None, None, :] <= reach_t[:, :, None]


def _check_raw_grid(up_valid: torch.Tensor, steps: int, decimation: int) -> None:
    """Refuse a raw mask that is not the trimmed grid of the feature steps.

    Args:
        up_valid: Raw UA validity $(B, L)$.
        steps: $T$.
        decimation: $D$.

    Raises:
        ValueError: If $L \ne D\,T$.
    """
    if up_valid.dim() != 2 or int(up_valid.shape[1]) != int(steps) * int(decimation):
        raise ValueError(
            f"up_valid has shape {tuple(up_valid.shape)}; expected (B, {int(steps) * int(decimation)})"
            f", the trimmed raw grid of {int(steps)} feature steps at {int(decimation)} samples "
            f"per step. The rule places runs against step boundaries, so another grid would "
            f"remove and admit the wrong keys."
        )


def declared_source_windows(paths: Sequence[str], use_up_st: bool = True) -> Tuple[int, ...]:
    r"""$W_c$ per DECLARED source channel in feature steps on the untrimmed grid, from the shards.

    The builder stores each channel's warm-up on the untrimmed grid as the attribute
    ``causal_warmup_steps`` of ``up_st`` and ``up_ph``. It is the number of steps a channel needs
    before its coefficient rests only on recorded samples, so it is also how far one invalid
    sample reaches into later steps. Every file is checked for agreement by
    :func:`hdf5_dataset.hdf5_dataset.read_causal_warmup`; the stored vector of the first file is
    returned. The trimmed vector that the network carries is not usable here, because the trim
    clips the fast channels to $0$.

    Args:
        paths: The configured shard files.
        use_up_st: Whether the source stream carries the ``up_st`` block before ``up_ph``.

    Returns:
        $\max(W_c, 1)$ per declared source channel, in stream order.

    Raises:
        ValueError: If no file is given, the files disagree, or a block is absent.
    """
    from hdf5_dataset.hdf5_dataset import _read_file_layout, read_causal_warmup

    paths = [str(Path(path)) for path in paths]
    read_causal_warmup(paths)   # refuses missing, two-sided or disagreeing files
    stored = _read_file_layout(paths[0]).warmup_steps or {}
    blocks = ("up_st", "up_ph") if use_up_st else ("up_ph",)
    missing = [block for block in blocks if block not in stored]
    if missing:
        raise ValueError(f"{paths[0]} stores no causal_warmup_steps for {missing}")
    vector = np.concatenate([np.asarray(stored[block], dtype=np.int64) for block in blocks])
    return tuple(int(step) for step in np.maximum(vector, 1))


__all__ = [
    "RAW_FS_HZ",
    "RUNNING_GAP_MODES",
    "declared_source_windows",
    "removed_keys",
    "run_table",
    "seconds_to_samples",
    "split_runs",
    "tolerant_flags",
    "tolerant_reward_validity",
]
