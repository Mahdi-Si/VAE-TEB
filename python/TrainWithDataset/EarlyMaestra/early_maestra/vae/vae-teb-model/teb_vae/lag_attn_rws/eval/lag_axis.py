r"""The lag axis, and reading a per-lag vector against it. One implementation, two consumers.

A lag index $\ell$ is not seconds. The figure to report is the **compensated** one,
$\tau_\ell = 4(\ell + \delta)$, where $\delta$ is the causal input delay the source channels are
read with -- so a peak at lag $\ell$ refers to source content $\ell + \delta$ steps back. The
other seconds figure, which adds the $20$ s the preprocessing already removed, exists only to
locate a finding in the original sensor files and appears in no analysis.

This module exists because two analyses draw that axis and a third will. The historical failure
was exactly this shape: two consumers computed the same quantity their own way, one of them read
the delay under a name that did not exist, and the two reports of one run disagreed by up to $30$
steps -- two minutes -- with nothing raising. The conversion itself lives in ``nets/lag_report``
and is shared with the training figure; what lives here is the *axis* built from it, and the two
helpers for laying a per-lag vector alongside that axis.

``NaN`` rather than zero for a bin the pass did not produce, in both helpers. A lag whose value
was never measured and a lag the source never attended to are different statements, and a zero
would make a profile's argmax, its width and its total all read as if the missing bins had been
measured and found empty.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from teb_vae.lag_attn_rws.nets.lag_report import lag_compensated_seconds


def compensated_seconds_axis(n_lags: int, delay_steps: int) -> np.ndarray:
    r"""The lag axis in compensated seconds: $\tau_\ell = 4(\ell + \delta)$.

    Built elementwise through the shared converter rather than as ``4 * lags + offset``. The two
    are the same arithmetic, and that is the point: an axis assembled its own way is how a figure
    and the number quoted beside it come to disagree.

    Args:
        n_lags: Lag window width $L$.
        delay_steps: The causal input delay $\delta$ in decimated steps, read from the model's own
            accessor and from nowhere else.

    Returns:
        The axis, $(L,)$, in seconds.
    """
    return np.array(
        [float(lag_compensated_seconds(lag, delay_steps=delay_steps)) for lag in range(n_lags)],
        dtype=np.float64,
    )


def padded_profile(values: Any, size: int) -> np.ndarray:
    """Return a per-lag vector at the axis's length, ``NaN``-filled where it is shorter.

    Args:
        values: The values, possibly empty or of another length.
        size: The axis's length.

    Returns:
        The padded array as ``float64``.
    """
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values,
                       dtype=np.float64)
    if array.size == size:
        return array
    padded = np.full(size, np.nan, dtype=np.float64)
    padded[: min(array.size, size)] = array[:size]
    return padded


def profile_column(frame: pd.DataFrame, name: str, size: int) -> np.ndarray:
    """Read one per-lag column of a table at the axis's length.

    Args:
        frame: A per-lag table.
        name: The column to read.
        size: The axis's length.

    Returns:
        The column as ``float64``, all-``NaN`` when the frame does not carry it -- so a profile an
        older run's tables did not hold draws as absent rather than taking down the figure.
    """
    if name not in getattr(frame, "columns", []) or len(frame) == 0:
        return np.full(size, np.nan, dtype=np.float64)
    return padded_profile(np.asarray(frame[name], dtype=np.float64), size)
