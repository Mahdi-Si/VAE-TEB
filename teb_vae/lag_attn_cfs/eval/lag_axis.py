r"""The lag axis, and reading a per-lag vector against it. One implementation, three consumers.

A lag index $\ell$ is not seconds. The figure to report is the **compensated** one,
$\tau_\ell = 4(\ell + \delta)$, where $\delta$ is the causal input delay the source channels are
read with -- so a peak at lag $\ell$ refers to source content $\ell + \delta$ steps back. The
other seconds figure, which *subtracts* the $20$ s the preprocessing already removed -- it advanced
the source trace, so undoing that moves the figure down -- exists only to locate a finding in the
original sensor files and appears in no analysis.

**What that axis is time *in* is where this module differs from the raw cells', and the
arithmetic is not what differs.** $\tau_\ell$ is computed here by the identical shared converter.
But the streams this model reads are not signals: they are wavelet-modulus and phase-harmonic
coefficients produced by a strictly one-sided filter bank, and a one-sided filter has a group
delay. A coefficient stored at step $t$ therefore summarises signal content centred somewhere
*before* $t$, by a per-channel amount the shards record as ``causal_delay_s`` -- 13.3 s to 791.0 s
on the committed causal fixture. Nothing in this pipeline corrects for it, deliberately: the
correction is per channel *pair* while the lag map is per head over a pooled source state, so the
mapping would itself be an unvalidated construction (both ``DESIGN.md`` records carry the
``lean-limit:`` for it).

**This module needs the stale-step number and only that one.** A run may additionally align its
input channels onto a common reference $\tau_{\mathrm{ref}}$, at which point two quantities exist
where one did: ``source_delay_steps``, the largest shift applied to any channel and therefore how
far back in *stored steps* the source memory reaches, and ``reference_delay_s``, the physical
instant every aligned channel of that stream reports at a step -- one per stream, and under a dual
reference the two differ by a constant the record carries as ``inter_stream_offset_s``.
$\tau_\ell$ is built from the first, because it is
an axis in stored-coefficient time and a stored-step count is exactly what indexes it. The second
belongs to a lag stated in *physical* seconds, which this package does not emit -- it appears in the
run's causality disclosure and its summary as ``source_reference_delay_s``, beside the axis rather
than inside it. Alignment does not remove the caveat below; it collapses the correction it names
from a channel-pair-indexed quantity to a constant, and applying that constant is a decision made
where a physical lag is reported, not here.

So the axis is **stored-coefficient time**, :data:`COEFFICIENT_LAG_AXIS_LABEL` says so, and
:data:`GROUP_DELAY_CAVEAT` is the sentence that travels on every lag-resolved artifact this
package emits. The two constants live here rather than in each caption for the reason the axis
itself does: a caption assembled its own way is how a figure and the number quoted beside it come
to disagree, and a caveat written per artifact is one that goes missing from the artifact that
gets quoted.

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

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP, lag_compensated_seconds

#: Axis label for a lag axis in this target domain. Deliberately **not**
#: ``lag_report.COMPENSATED_LAG_AXIS_LABEL``: that one says "mechanically compensated", which is
#: true here too and is not the caveat that matters. What matters is that the quantity being
#: lagged is a stored coefficient rather than a signal sample, and no other label says so.
#:
#: It contains no "bpm" and no "physiological": the first has no meaning in a domain of wavelet
#: coefficients, and the second is the claim the group delay forbids.
COEFFICIENT_LAG_AXIS_LABEL = "lag (s, stored-coefficient time)"

#: The largest composed one-sided group delay measured on the committed causal fixture's four
#: stored blocks, in seconds -- the number the caveat quotes. Measured rather than derived, and
#: recorded here as the *fixture's* value: a run's own per-block minimum, median and maximum are
#: read off its configured shards into ``preflight.json`` and ``summary.json``, which is
#: where a reader of one run's artifacts gets the figure that applies to that run.
MAX_MEASURED_GROUP_DELAY_SECONDS = 791.0

#: The shipped lag window: ``max_lag = 90`` searched lags plus lag $0$, so $L = 91$ bins spanning
#: $L \cdot \Delta = 364$ s. Written as the model's ``max_lag`` rather than as $L$ because that is
#: the config key an arm moves, and multiplied by the shared :data:`SECONDS_PER_STEP` rather than
#: by a restated $4$.
SHIPPED_MAX_LAG_STEPS = 90
SHIPPED_LAG_SPAN_SECONDS = (SHIPPED_MAX_LAG_STEPS + 1) * SECONDS_PER_STEP

#: The sentence every lag-resolved artifact carries. The comparison is the point rather than
#: either number: a group delay of the same order as the lag search means a peak's *position* on
#: this axis cannot be read as a physiological delay at all, and a reader given only the lag
#: figures would have no way to know that.
GROUP_DELAY_CAVEAT = (
    f"this lag axis is stored-coefficient time, not physiological time: the coefficients are "
    f"produced by a strictly one-sided bank whose composed per-channel group delay reaches "
    f"{MAX_MEASURED_GROUP_DELAY_SECONDS:g} s on the committed causal fixture, which is the same "
    f"order as the {SHIPPED_LAG_SPAN_SECONDS:g} s lag search itself "
    f"({SHIPPED_MAX_LAG_STEPS + 1} lags at {SECONDS_PER_STEP:g} s per step). The lag map is an "
    f"attribution over the axis the coefficients are stored on, uncorrected for that delay, and "
    f"is therefore not a physiological latency and not a transfer entropy"
)


def compensated_seconds_axis(n_lags: int, delay_steps: int) -> np.ndarray:
    r"""The lag axis in compensated seconds: $\tau_\ell = 4(\ell + \delta)$.

    Built elementwise through the shared converter rather than as ``4 * lags + offset``. The two
    are the same arithmetic, and that is the point: an axis assembled its own way is how a figure
    and the number reported beside it come to disagree.

    The result is in **stored-coefficient** seconds; see :data:`GROUP_DELAY_CAVEAT`, which every
    artifact drawn against this axis carries.

    Args:
        n_lags: Lag window width $L$.
        delay_steps: The causal input delay $\delta$ in decimated steps, read from the model's own
            accessor and from nowhere else. The **stored-step** figure, not the alignment
            reference; see the module docstring for why this axis takes that one.

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


# =================================================================================================
# The measured lag support, read back rather than assumed
# =================================================================================================
#: The preflight artifact the measured lag support is read back from. ``preflight.py`` owns the
#: record and writes it; this is the reader's name for the same file, restated rather than imported
#: because that module rebuilds a checkpoint -- and the two analyses that read the number are layer
#: 2 and run offline, with no model and no GPU. ``tests/test_eval_lag_axis.py`` pins the two names
#: equal, so a rename cannot leave one side reading a file the other stopped writing.
PREFLIGHT_FILENAME = "preflight.json"

#: What :func:`read_lag_support` returns when the record is absent or carries no support block.
#: ``None`` rather than a default margin, and the flag ``None`` rather than ``False``: an analysis
#: must be able to tell "the geometry is truncated" from "nobody measured", and a ``False`` here
#: would make an unmeasured run report a truncation it never had.
UNMEASURED_LAG_SUPPORT = {
    "measured": False,
    "lag_support_margin_steps": None,
    "every_lag_valid_at_every_anchor": None,
}


def read_lag_support(results_dir: Any) -> dict:
    r"""Read the lag-support margin the preflight guard measured for this run.

    $$\texttt{lag\_support\_margin\_steps} = \min_t \mathcal A - (L - 1) - F_u,$$

    the earliest decoded anchor less the furthest searched lag less the lag floor. Everything the
    per-lag analyses simplify away -- the support correction, the untruncated recomputation and the
    entropy ceiling collapsing to $\log L$ -- holds exactly when it is $\ge 0$. The three
    quantities behind it move independently, so an analysis that assumed the shipped geometry would
    report a simplification a ``sweep_floor_*`` arm had quietly taken away.

    Read off the run's own ``preflight.json`` rather than recomputed, for the reason every other
    file-on-disk dependency in this package exists: the analyses layer holds no model on the path
    that matters, an offline re-run against a finished directory.

    Args:
        results_dir: The run's results directory, which is the ``output_dir`` an analysis is given.

    Returns:
        The support block with ``measured`` set, or :data:`UNMEASURED_LAG_SUPPORT` when there is no
        record to read. Absent rather than defaulted: a run whose margin nobody measured must not
        report one.
    """
    import json
    from pathlib import Path

    path = Path(results_dir) / PREFLIGHT_FILENAME
    if not path.is_file():
        return {**UNMEASURED_LAG_SUPPORT, "reason": f"{path} does not exist"}
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    support = ((record or {}).get("causality") or {}).get("lag_support")
    if not isinstance(support, dict) or "lag_support_margin_steps" not in support:
        return {
            **UNMEASURED_LAG_SUPPORT,
            "reason": f"{path} carries no causality.lag_support block",
        }
    return {"measured": True, **support}
