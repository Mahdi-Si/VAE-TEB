r"""The shape of a per-lag profile, reduced to scalars. One implementation, two consumers.

A per-lag vector -- the KL attribution, or the attention over the lags -- is a distribution over the
compensated axis :mod:`~teb_vae.lag_attn_cfs.eval.lag_axis` builds, and two analyses reduce it to
numbers. ``lag_kl`` describes the peak of a *pooled* profile, one per cohort; ``lag_clocks``
describes every *segment's own* profile against two clinical clocks. The two are the same vocabulary
at two population levels, and this module is where that vocabulary lives, so that "degenerate", "the
peak's width" and "the mass near the peak" have one definition rather than two.

**That is why the peak helpers moved here rather than being copied.** They were ``lag_kl``'s, and
``lag_clocks`` refused to report a peak at all on the stated ground that the guard which makes a
positional claim honest lived in an analysis it may not import. Copying the guard would have forked
what "degenerate" means -- two thresholds drifting apart while both files still read as though they
described one criterion. Moving it down a layer removes the obstacle instead of routing around it.

**Two shapes of caller, and both are served here.** ``lag_kl`` holds one profile at a time and wants
a description with its reasons spelled out, so :func:`peak_width`, :func:`mass_above`,
:func:`secondary_peaks` and :func:`degeneracy` each take a single ``(L,)`` sequence and return a
record. ``lag_clocks`` holds an ``(n, L)`` stack of every scored segment and wants columns, so
:func:`profile_statistics` returns one ``(n,)`` array per statistic from a single vectorised pass.
The second is **not** a loop over the first -- it is the same arithmetic written for a matrix -- and
``tests/test_eval_lag_shape.py`` pins the two equal row by row. That test is what makes the move a
de-duplication rather than a second implementation with a shared docstring.

**Non-finite bins are dropped from the mass, never treated as zero**, in every function here. A lag
whose value was never measured and a lag the source never attended to are different statements, and
the second would pull a centroid toward a bin carrying no evidence and would make a profile's width,
its argmax and its total all read as though the missing bins had been measured and found empty.

This module is numpy-and-stdlib only. It touches no model, no table and no figure, so an analysis
running offline against a finished run directory imports it without paying for ``torch``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: Fraction of the peak a bin must reach to count as part of it. Half, so the reported width is
#: the familiar full width at half maximum and a reader does not have to learn a local convention.
PEAK_FRACTION = 0.5

#: A profile is **degenerate** below this peak-to-median ratio: the peak is not distinguishable
#: from the bulk, so its argmax names a bin rather than a finding. Mechanical rather than
#: eyeballed, because "the profile looked flat" is not a criterion a second reader can apply.
DEGENERATE_PEAK_TO_MEDIAN = 1.1

#: A profile is also degenerate above this exact-zero fraction. ``entmax15`` assigns exact zeros,
#: so a profile can be sparse legitimately; one that is *this* sparse has a handful of live bins
#: and its shape is set by which of them survived rather than by where the source informed.
DEGENERATE_ZERO_FRACTION = 0.9

#: Seconds per lag step, for quoting a peak *width* in the units its axis is drawn in. A width is
#: a difference of lag indices, so the causal input delay cancels out of it -- which is why this
#: is the step size rather than a call to the compensated converter.
SECONDS_PER_LAG_STEP = SECONDS_PER_STEP

#: Lag offset, measured from the **shortest lag the axis carries**, inside which mass counts as
#: sitting near the anchor. Fifteen steps of the shipped 91-bin, 364 s search.
#:
#: Measured from the axis's own start rather than from zero, and that is not a detail: the
#: compensated axis begins at $\tau_0 = 4\delta$ rather than at $0$, so a threshold stated in
#: absolute compensated seconds would admit a different number of bins on every run whose causal
#: input delay differs -- and a cohort comparison would then be reading two different windows. The
#: offset is delay-invariant, so the same fraction means the same thing across runs and across the
#: two cells that share this pipeline.
NEAR_SECONDS = 60.0

#: ... and the offset beyond which mass counts as sitting far from the anchor. Sixty steps, so the
#: near and far windows do not touch: a profile can be diffuse without being counted as either.
FAR_SECONDS = 240.0

#: Every statistic :func:`profile_statistics` returns, in the order a table carries them. Written
#: out rather than derived from the returned mapping, so a consumer can lay out its columns before
#: calling and a statistic added here without a column fails in that consumer rather than silently
#: going unemitted.
STATISTIC_KEYS: Tuple[str, ...] = (
    "centroid",
    "spread",
    "median",
    "iqr",
    "effective_support",
    "peak",
    "peak_width",
    "entropy",
    "skewness",
    "near_mass",
    "far_mass",
    "peak_mass",
    "peak_degenerate",
    "zero_fraction",
)


# =================================================================================================
# One profile at a time: the peak, described rather than merely located
# =================================================================================================
def peak_width(profile: Sequence[float], *, fraction: float = PEAK_FRACTION) -> Dict[str, Any]:
    r"""Describe the peak by its extent rather than by its position alone.

    The contiguous run of bins around the argmax that stay at or above ``fraction`` of the peak
    -- the full width at half maximum when ``fraction`` is $0.5$. Contiguity is what makes it a
    *width*: counting every bin above the threshold anywhere in the profile would report a bimodal
    profile as one very wide peak, which is the opposite of what a second peak means.

    Args:
        profile: One value per lag.
        fraction: Height, as a fraction of the peak, defining the peak's edge.

    Returns:
        The argmax, the peak value, the inclusive bin bounds of the peak, and its width in bins.
        All ``None`` on an empty or non-finite profile.
    """
    values = np.asarray(list(profile), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).any():
        return {"argmax": None, "peak": None, "lo": None, "hi": None, "width_bins": None}
    finite = np.where(np.isfinite(values), values, -np.inf)
    argmax = int(np.argmax(finite))
    peak = float(finite[argmax])
    threshold = peak * float(fraction)
    lo = argmax
    while lo > 0 and finite[lo - 1] >= threshold:
        lo -= 1
    hi = argmax
    while hi + 1 < finite.size and finite[hi + 1] >= threshold:
        hi += 1
    return {
        "argmax": argmax,
        "peak": peak,
        "lo": int(lo),
        "hi": int(hi),
        "width_bins": int(hi - lo + 1),
    }


def mass_above(profile: Sequence[float], *, fraction: float = PEAK_FRACTION) -> Dict[str, Any]:
    """How much of the profile's total sits in bins at or above a fraction of the peak.

    The concentration the argmax does not report: a profile whose peak holds four fifths of the
    attribution and one whose peak holds a twentieth have the same argmax and are different
    findings.

    Args:
        profile: One value per lag.
        fraction: Height threshold, as a fraction of the peak.

    Returns:
        The share of the total in those bins, how many bins they are, and the threshold used.
    """
    values = np.asarray(list(profile), dtype=np.float64)
    finite = values[np.isfinite(values)]
    total = float(finite.sum()) if finite.size else 0.0
    if not finite.size or total <= 0.0:
        return {"share": float("nan"), "n_bins": 0, "threshold": float("nan")}
    threshold = float(finite.max()) * float(fraction)
    selected = finite[finite >= threshold]
    return {
        "share": float(selected.sum() / total),
        "n_bins": int(selected.size),
        "threshold": threshold,
    }


def secondary_peaks(
    profile: Sequence[float], *, fraction: float = PEAK_FRACTION, min_separation: int = 1
) -> List[Dict[str, Any]]:
    """Find local maxima other than the tallest, above a fraction of it.

    A second peak is a finding rather than noise: it says the source informs the forecast at two
    separated delays, which an argmax reports as one of them and a mean reports as neither.

    Args:
        profile: One value per lag.
        fraction: How tall, relative to the global peak, a local maximum must be to count.
        min_separation: How many bins a local maximum must stand clear of the global peak.

    Returns:
        One record per secondary peak -- its lag, its value and its share of the global peak --
        ordered by height, tallest first.
    """
    values = np.asarray(list(profile), dtype=np.float64)
    if values.size < 3 or not np.isfinite(values).any():
        return []
    finite = np.where(np.isfinite(values), values, -np.inf)
    argmax = int(np.argmax(finite))
    peak = float(finite[argmax])
    if not np.isfinite(peak) or peak <= 0.0:
        return []
    threshold = peak * float(fraction)
    found: List[Dict[str, Any]] = []
    for index in range(1, finite.size - 1):
        if abs(index - argmax) <= int(min_separation):
            continue
        value = float(finite[index])
        if value >= threshold and value >= finite[index - 1] and value >= finite[index + 1]:
            found.append({"lag_step": index, "value": value, "share_of_peak": value / peak})
    return sorted(found, key=lambda record: -record["value"])


def degeneracy(profile: Sequence[float]) -> Dict[str, Any]:
    """Decide mechanically whether a profile has a shape worth reading at all.

    Two ways it does not, and they are different failures. A **flat** profile -- peak barely above
    the median -- has an argmax that names whichever bin won a coin toss. A profile that is almost
    entirely **exact zeros** has a shape set by which handful of bins ``entmax15`` kept alive.

    Args:
        profile: One value per lag.

    Returns:
        The flag, the two measured statistics behind it, the thresholds they were judged against,
        and the reasons that fired. An empty profile is degenerate, with that as its reason.
    """
    values = np.asarray(list(profile), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "degenerate": True,
            "peak_to_median": float("nan"),
            "zero_fraction": float("nan"),
            "peak_to_median_threshold": DEGENERATE_PEAK_TO_MEDIAN,
            "zero_fraction_threshold": DEGENERATE_ZERO_FRACTION,
            "reasons": ["the profile carries no finite value"],
        }
    median = float(np.median(finite))
    peak = float(finite.max())
    # A zero median makes the ratio infinite rather than undefined, which is the right reading:
    # a peak standing over an all-zero bulk is as far from flat as a profile gets.
    ratio = float("inf") if median == 0.0 and peak > 0.0 else (
        peak / median if median != 0.0 else float("nan")
    )
    zero_fraction = float((finite == 0.0).sum() / finite.size)
    reasons: List[str] = []
    if np.isfinite(ratio) and ratio < DEGENERATE_PEAK_TO_MEDIAN:
        reasons.append(
            f"peak-to-median {ratio:.3g} is below {DEGENERATE_PEAK_TO_MEDIAN}, so the peak is not "
            f"distinguishable from the bulk and its argmax names a bin rather than a lag"
        )
    if zero_fraction > DEGENERATE_ZERO_FRACTION:
        reasons.append(
            f"{zero_fraction:.1%} of the bins are exactly zero, above "
            f"{DEGENERATE_ZERO_FRACTION:.0%}, so the shape is set by which bins survived "
            f"sparsification rather than by where the source informed"
        )
    return {
        "degenerate": bool(reasons),
        "peak_to_median": ratio,
        "zero_fraction": zero_fraction,
        "peak_to_median_threshold": DEGENERATE_PEAK_TO_MEDIAN,
        "zero_fraction_threshold": DEGENERATE_ZERO_FRACTION,
        "reasons": reasons,
    }


# =================================================================================================
# A stack of profiles at once: every statistic, one pass
# =================================================================================================
def _blank(count: int) -> Dict[str, np.ndarray]:
    """Return the full statistic mapping at ``count`` rows, every value ``NaN``."""
    return {key: np.full(count, np.nan, dtype=np.float64) for key in STATISTIC_KEYS}


def profile_statistics(
    rows: Any, seconds: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    r"""Reduce a stack of per-lag vectors to every scalar that describes their shape.

    With $p_{i\ell} = w_{i\ell} / \sum_k w_{ik}$ the normalised profile of row $i$ on the
    compensated axis $\tau_\ell$:

    $$\bar\tau_i = \sum_\ell p_{i\ell}\tau_\ell, \qquad
    \sigma_i = \sqrt{\sum_\ell p_{i\ell}\tau_\ell^2 - \bar\tau_i^2}, \qquad
    H_i = -\sum_\ell p_{i\ell}\log p_{i\ell}$$

    and, from the cumulative mass, the quantile lags $\tau_{i,q}$ with the median at $q = 0.5$ and
    the inter-quartile range $\tau_{i,0.75} - \tau_{i,0.25}$.

    **The moments come from the raw moments rather than from a second pass over the residuals** --
    the same numbers in exact arithmetic, three matrix products instead of six. Floating point can
    drive the variance slightly negative on a profile concentrated in a single bin, so it is clipped
    at zero rather than allowed to produce a ``NaN`` square root on a perfectly ordinary profile.

    **Three families, and they answer different questions.** The moments say where the mass sits and
    how far it is spread; the quantiles say the same robustly, which matters because these profiles
    are skewed and a single far bin moves $\bar\tau$ much more than it moves the median. The entropy
    and the mass fractions say how *concentrated* the profile is without reference to a centre at
    all -- a bimodal profile has an ordinary centroid and an ordinary spread, and only $H_i$ says it
    is not a single lump. The peak family says where the tallest bin is, and is reported only beside
    :data:`STATISTIC_KEYS`' ``peak_degenerate``, which says whether that position means anything.

    **Non-finite bins are dropped from the mass, not treated as zero.** A lag whose value was never
    measured and a lag the source never attended to are different statements, and the second would
    pull the centroid toward a bin that carries no evidence.

    A row is reported as ``NaN`` throughout rather than as a number when it carries no mass at all
    -- a segment that scored no anchors -- or when any bin is negative, which these profiles cannot
    be: the attribution is a non-negative KL times a non-negative attention, so a negative bin is a
    defect and is counted rather than averaged into a cohort.

    Args:
        rows: The per-segment vectors, $(n, L)$, in the caller's row order.
        seconds: The compensated lag axis, $(L,)$, ascending.

    Returns:
        ``(statistics, record)`` -- one $(n,)$ array per key of :data:`STATISTIC_KEYS`, and how many
        rows carried usable mass, how many were empty and how many were rejected as negative. Every
        array is all-``NaN`` when the matrix does not have the axis's width: a vector of another
        length is a mis-assembled profile rather than a short one, and reshaping it into a plausible
        wrong answer is what this refuses.
    """
    values = np.asarray(rows, dtype=np.float64)
    if seconds.size == 0 or values.ndim != 2 or values.shape[0] == 0:
        return _blank(0), {"n_rows": 0, "n_usable": 0, "n_empty": 0, "n_negative": 0}
    count = int(values.shape[0])
    n_lags = int(seconds.size)
    if values.shape[1] != n_lags:
        return _blank(count), {
            "n_rows": count, "n_usable": 0, "n_empty": 0, "n_negative": 0,
            "note": (
                f"the profile is {values.shape[1]} bins wide against a {n_lags}-bin lag "
                f"axis, so it is a mis-assembled vector rather than a short one"
            ),
        }

    finite = np.isfinite(values)
    weights = np.where(finite, values, 0.0)
    negative = (weights < 0.0).any(axis=1)
    total = weights.sum(axis=1)
    usable = (total > 0.0) & ~negative
    # Divided by one where the row is unusable, so the arithmetic below stays warning-free; every
    # row that took this branch is overwritten with NaN by the masking at the end.
    divisor = np.where(usable, total, 1.0)
    shares = weights / divisor[:, None]

    # --- The moments -----------------------------------------------------------------------------
    first = shares @ seconds
    second = shares @ (seconds ** 2)
    third = shares @ (seconds ** 3)
    variance = np.maximum(second - first ** 2, 0.0)
    spread = np.sqrt(variance)
    # The third *central* moment, expanded from the raw ones for the same reason the variance is.
    central_third = third - 3.0 * first * second + 2.0 * first ** 3
    skewness = np.divide(
        central_third, spread ** 3, out=np.full(count, np.nan), where=spread > 0.0
    )

    # --- The quantiles ---------------------------------------------------------------------------
    cumulative = np.cumsum(shares, axis=1)
    # Pinned rather than trusted: a cumulative sum that lands at 0.9999999 through floating point
    # would make ``argmax`` find no bin at all and silently return bin 0 -- the shortest lag, which
    # is exactly the wrong answer and an entirely plausible-looking one.
    cumulative[:, -1] = 1.0
    quantile_lag = {
        quantile: seconds[(cumulative >= quantile).argmax(axis=1)]
        for quantile in (0.25, 0.5, 0.75)
    }
    median = quantile_lag[0.5]
    iqr = quantile_lag[0.75] - quantile_lag[0.25]

    # --- The concentration -----------------------------------------------------------------------
    # ``xlogy`` convention, and it is load-bearing here rather than defensive: ``entmax15`` assigns
    # lags exactly zero, so a well-formed profile routinely carries bins at which $p\log p$ must be
    # read as its limit of zero rather than as a warning.
    positive = shares > 0.0
    # Subtracted from zero rather than negated, so a profile concentrated in a single bin reports
    # ``0.0`` rather than ``-0.0`` -- the same number, which reads in a CSV as though something had
    # gone slightly wrong.
    entropy = 0.0 - np.sum(
        np.where(positive, shares * np.log(np.where(positive, shares, 1.0)), 0.0), axis=1
    )
    # Perplexity, quoted as a width: $e^H$ is the number of bins a uniform profile of the same
    # entropy would occupy, and one bin is one lag step.
    effective_support = SECONDS_PER_LAG_STEP * np.exp(entropy)

    offset = seconds - float(seconds[0])
    near_mass = shares[:, offset <= NEAR_SECONDS].sum(axis=1)
    far_mass = shares[:, offset >= FAR_SECONDS].sum(axis=1)

    # --- The peak, and the guard that says whether to read it ------------------------------------
    # Filled with $-\infty$ rather than with zero, matching :func:`peak_width` bin for bin: a bin
    # that was never measured must not be able to win the argmax, and must break the contiguous run
    # rather than extend it.
    filled = np.where(finite, values, -np.inf)
    peak_index = filled.argmax(axis=1)
    peak_value = filled[np.arange(count), peak_index]
    peak_seconds = seconds[peak_index]
    above = filled >= (peak_value * PEAK_FRACTION)[:, None]
    # The contiguous run around the peak, found as the gap between the nearest sub-threshold bin on
    # each side -- the vectorised form of the two ``while`` loops in :func:`peak_width`.
    lag_index = np.arange(n_lags)[None, :]
    below = ~above
    left_edge = np.where(below & (lag_index < peak_index[:, None]), lag_index, -1).max(axis=1)
    right_edge = np.where(below & (lag_index > peak_index[:, None]), lag_index, n_lags).min(axis=1)
    peak_width_seconds = (right_edge - left_edge - 1).astype(np.float64) * SECONDS_PER_LAG_STEP
    # ``& finite`` because a non-finite bin sits at $-\infty$ and would otherwise count as above a
    # threshold that is itself $-\infty$ on a row with nothing measured.
    peak_mass = np.where(above & finite, weights, 0.0).sum(axis=1) / divisor

    # The degeneracy criterion reads the **raw** profile, exactly as :func:`degeneracy` does: both
    # of its statistics are scale-invariant, and reading the same array keeps the two provably equal
    # rather than equal-up-to-a-renormalisation.
    n_finite = finite.sum(axis=1)
    has_finite = n_finite > 0
    # Substituted with zeros where a row has nothing finite, so ``nanmedian`` and ``nanmax`` are
    # never handed an all-NaN row; those rows are overwritten immediately below.
    masked = np.where(has_finite[:, None], np.where(finite, values, np.nan), 0.0)
    median_bin = np.nanmedian(masked, axis=1)
    peak_bin = np.nanmax(masked, axis=1)
    ratio = np.where(
        median_bin == 0.0,
        np.where(peak_bin > 0.0, np.inf, np.nan),
        np.divide(peak_bin, median_bin, out=np.full(count, np.nan), where=median_bin != 0.0),
    )
    zero_fraction = np.divide(
        ((values == 0.0) & finite).sum(axis=1).astype(np.float64),
        n_finite.astype(np.float64),
        out=np.full(count, np.nan),
        where=has_finite,
    )
    degenerate = (
        (np.isfinite(ratio) & (ratio < DEGENERATE_PEAK_TO_MEDIAN))
        | (zero_fraction > DEGENERATE_ZERO_FRACTION)
    )
    degenerate = np.where(has_finite, degenerate, True).astype(np.float64)

    computed = {
        "centroid": first,
        "spread": spread,
        "median": median,
        "iqr": iqr,
        "effective_support": effective_support,
        "peak": peak_seconds,
        "peak_width": peak_width_seconds,
        "entropy": entropy,
        "skewness": skewness,
        "near_mass": near_mass,
        "far_mass": far_mass,
        "peak_mass": peak_mass,
        "peak_degenerate": degenerate,
        "zero_fraction": zero_fraction,
    }
    statistics = {
        key: np.where(usable, np.asarray(value, dtype=np.float64), np.nan)
        for key, value in computed.items()
    }
    return statistics, {
        "n_rows": count,
        "n_usable": int(usable.sum()),
        "n_empty": int((~usable & ~negative).sum()),
        "n_negative": int(negative.sum()),
    }
