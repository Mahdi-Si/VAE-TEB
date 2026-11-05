r"""The four states of the posterior event gate, and the one rule that assigns an anchor to a state.

The posterior applies the effective gate $g^{\mathrm{eff}}_t = g_t\,o_t$, where $g_t \in (0, 1)$ is
the learned gate and $o_t \in \{0, 1\}$ says whether at least one UA lag key survived the
signal-quality mask. The effective gate is $0$ for two different reasons, and a reader must keep
them apart:

* **forced shut** -- $o_t = 0$. The data removed every key, so $q = p$ whatever the model wants.
* **model shut** -- $o_t = 1$ and $g_t < g_{\mathrm{shut}}$. The UA was available and the model
  chose not to use it.

The other two states split the anchors where the UA was available and the gate is not shut:
**partly open** for $g_{\mathrm{shut}} \le g_t < g_{\mathrm{open}}$ and **open** for
$g_t \ge g_{\mathrm{open}}$.

The sample page and the ``event_gate`` analysis both import this module, so a state on a page and
the same state in a table are one rule. The module needs ``numpy`` only, so the evaluation's
table-only layer can import it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

#: $g_{\mathrm{shut}}$: below it, an anchor whose UA keys are open is **model shut**. At $0.2$ the
#: posterior residual keeps at most a fifth of its size, so the source moves the belief very little.
GATE_SHUT_BELOW = 0.2

#: $g_{\mathrm{open}}$: from it up, an anchor is **open**. At $0.8$ the residual keeps at least four
#: fifths of its size.
GATE_OPEN_FROM = 0.8

#: State codes, in the order every table and legend lists them.
FORCED_SHUT, MODEL_SHUT, PARTLY_OPEN, OPEN = 0, 1, 2, 3

#: The state names, indexed by code.
GATE_STATE_NAMES: Tuple[str, ...] = ("forced_shut", "model_shut", "partly_open", "open")

#: How each state reads in a legend, indexed by code.
GATE_STATE_LABELS: Tuple[str, ...] = (
    "forced shut ($o_t = 0$)",
    f"model shut ($g_t < {GATE_SHUT_BELOW:g}$)",
    "partly open",
    f"open ($g_t \\geq {GATE_OPEN_FROM:g}$)",
)


def gate_states(learned: np.ndarray, source_open: Optional[np.ndarray] = None) -> np.ndarray:
    r"""Assign every anchor to one of the four gate states.

    Args:
        learned: The learned gate $g_t$, any shape. A ``NaN`` entry gets state $-1$.
        source_open: $o_t$ in the shape of ``learned``, or ``None`` when the model applies no UA
            mask. Then every key is open and no anchor is forced shut.

    Returns:
        An ``int8`` array of state codes in the shape of ``learned``: :data:`FORCED_SHUT`,
        :data:`MODEL_SHUT`, :data:`PARTLY_OPEN`, :data:`OPEN`, or $-1$ where $g_t$ is not finite.
    """
    gate = np.asarray(learned, dtype=np.float64)
    states = np.full(gate.shape, -1, dtype=np.int8)
    finite = np.isfinite(gate)
    states[finite & (gate < GATE_SHUT_BELOW)] = MODEL_SHUT
    states[finite & (gate >= GATE_SHUT_BELOW) & (gate < GATE_OPEN_FROM)] = PARTLY_OPEN
    states[finite & (gate >= GATE_OPEN_FROM)] = OPEN
    if source_open is not None:
        closed = np.asarray(source_open, dtype=np.float64) < 0.5
        # Forced shut wins over the learned value: with no key the gate has no effect at all.
        states[finite & closed] = FORCED_SHUT
    return states


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Return the length of every run of consecutive ``True`` entries in a 1-D boolean array.

    Args:
        flags: One segment's per-step flags, in time order.

    Returns:
        The run lengths in steps, in time order. Empty when no entry is ``True``.
    """
    values = np.asarray(flags, dtype=bool).ravel()
    if not values.any():
        return np.zeros(0, dtype=np.int64)
    padded = np.concatenate([[False], values, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (stops - starts).astype(np.int64)


__all__ = [
    "FORCED_SHUT",
    "GATE_OPEN_FROM",
    "GATE_SHUT_BELOW",
    "GATE_STATE_LABELS",
    "GATE_STATE_NAMES",
    "MODEL_SHUT",
    "OPEN",
    "PARTLY_OPEN",
    "gate_states",
    "run_lengths",
]
