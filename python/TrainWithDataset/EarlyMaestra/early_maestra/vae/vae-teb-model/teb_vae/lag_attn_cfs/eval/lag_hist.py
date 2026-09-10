r"""Two per-lag distributions compared, and the arithmetic that says how far apart they are.

:mod:`~teb_vae.lag_attn_cfs.eval.lag_shape` reduces **one** profile to scalars. This module answers
the question that reduction cannot: given two profiles -- two clinical classes in the same window,
or one class in two windows -- *how far apart are they, as distributions*. A pair of centroids
answers it only for the family of shifts; two profiles with the same centroid and different shapes
are a finding no positional statistic reports, and on this family the profiles are skewed monotone
decays whose centroid sits far from their peak, which is exactly the regime where a single scalar
is least informative.

**Two distances, because they fail differently and neither subsumes the other.**

* :func:`jensen_shannon` is a metric on distributions, bounded in $[0, 1]$ at base $2$, and reads
  purely as *overlap*: it is $0$ when two profiles coincide and $1$ when their supports are
  disjoint. It knows nothing about the axis -- two profiles differing by one lag step and two
  differing by ninety are equally far apart if their supports do not meet -- which is what makes it
  the right reading of "do these two look at the same lags at all".
* :func:`wasserstein_seconds` is the transport distance **on the compensated seconds axis**, so it
  is quoted in seconds and reads directly as "the distribution moved this far". It is the one that
  degrades gracefully: two narrow profiles a step apart are a step apart, where Jensen--Shannon
  would already be near its ceiling. It is also the one that can be near zero for two profiles that
  share a centre and differ entirely in width.

Reported together, they separate the three ways a lag distribution can change -- it moved (large
$W_1$, moderate $\mathrm{JS}$), it spread or narrowed in place (small $W_1$, non-zero $\mathrm{JS}$)
or it moved onto lags the other never touched (both large) -- and a table carrying only one of them
cannot tell those apart.

**Both are computed on the normalised profile.** A distance between un-normalised profiles would be
partly a statement about how much coupling each cell carried, which is a different question and one
:data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS`' ``total_nats`` already answers.

**Non-finite bins are dropped from the mass, never treated as zero**, which is the invariant every
function in ``lag_shape`` holds and holds for the same reason: a lag whose value was never measured
and a lag the source never attended to are different statements, and reading the first as the second
would pull a centroid toward a bin carrying no evidence and would make two cells measured over
different lag supports look like two cells that disagreed.

**A cell with no mass is ``NaN``, never zero.** Zero is the distance between two identical
distributions -- the strongest possible statement of agreement -- so returning it for a comparison
that could not be made would put the loudest finding in the table wherever there was no evidence at
all.

This module is numpy-and-stdlib only, and SciPy is imported lazily inside the two functions that
need it, which is this package's convention rather than a local choice: a box without SciPy loses
these two numbers and nothing else. It touches no model, no table and no figure, so an analysis
running offline against a finished run directory imports it without paying for ``torch``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

#: Logarithm base the Jensen--Shannon distance is taken in. Two, so the divergence it is the square
#: root of is bounded by $1$ bit and the distance itself by $1$ -- a ceiling a reader can compare a
#: value against without knowing the lag axis's width. Not a setting: a table whose rows were
#: produced under two bases would carry one column with two ceilings.
JENSEN_SHANNON_BASE = 2.0


def _mass(profile: Any) -> np.ndarray:
    """Return a profile's finite, non-negative mass as ``float64``, zeros where it has none.

    Args:
        profile: One value per lag, in any dtype.

    Returns:
        The $(L,)$ vector with non-finite bins replaced by zero and negative bins clipped to it.
        Negatives are clipped rather than rejected here because the caller has already been told
        about them: ``lag_shape.profile_statistics`` counts a row with any negative bin as
        ``n_negative`` and reports it as ``NaN`` throughout, so a negative reaching this module is
        a bin of a row that analysis has already disqualified.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def normalise(rows: Any) -> np.ndarray:
    r"""Row-normalise a stack of per-lag profiles into distributions over the lags.

    $$p_{i\ell} = \frac{w_{i\ell}}{\sum_k w_{ik}}$$

    the same normalisation ``lag_shape.profile_statistics`` performs internally, exposed because the
    drawn histogram and the compared feature must be the *same* object: a figure normalised one way
    beside a statistic normalised another is how a page and the number quoted from it come to
    disagree.

    Args:
        rows: The profiles, $(n, L)$ or a single $(L,)$ vector.

    Returns:
        The $(n, L)$ normalised stack. A row carrying no finite positive mass is all-``NaN`` rather
        than all-zero: "this cell attended nowhere" and "this cell was never measured" are different
        statements, and a zero row would enter a mean as evidence of the first.

        A **bin** that is not finite stays ``NaN`` for the same reason, and is excluded from the
        denominator rather than counted as empty -- so the surviving bins of a row with one missing
        lag share the whole of it, and a figure drawn from this shows a gap where nothing was
        measured rather than a bar at zero. This is where the emitted density column comes from,
        which is why it keeps the distinction that :func:`_mass` -- feeding a distance, which needs
        a real vector -- has to give up.
    """
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.size == 0:
        return matrix.astype(np.float64)
    finite = np.isfinite(matrix)
    mass = np.where(finite & (matrix > 0.0), matrix, 0.0)
    totals = mass.sum(axis=1)
    shares = np.divide(
        mass, totals[:, None], out=np.full(mass.shape, np.nan), where=totals[:, None] > 0.0
    )
    return np.where(finite, shares, np.nan)


def jensen_shannon(left: Any, right: Any) -> float:
    r"""The Jensen--Shannon **distance** between two per-lag profiles, in $[0, 1]$.

    $$\mathrm{JS}(p, q) = \sqrt{\tfrac{1}{2}D_{\mathrm{KL}}(p \,\|\, m)
    + \tfrac{1}{2}D_{\mathrm{KL}}(q \,\|\, m)}, \qquad m = \tfrac{1}{2}(p + q)$$

    at base :data:`JENSEN_SHANNON_BASE`. The square root -- the *distance* rather than the
    divergence -- because it is the form that obeys the triangle inequality, so a column of them is
    readable as a geometry over cells rather than only pairwise.

    Symmetric and bounded, which is what makes it the readout to quote when the question is whether
    two cells attend to the same lags at all. It is **not** the readout for "how far did it move":
    it is blind to the axis, so a one-step shift of a narrow profile and a ninety-step one are the
    same number. :func:`wasserstein_seconds` is that readout.

    Args:
        left: One value per lag.
        right: One value per lag, the same length.

    Returns:
        The distance, or ``NaN`` when either profile carries no finite positive mass or the two are
        not the same length -- a length mismatch is a mis-assembled cell rather than a short one,
        and padding it into a plausible wrong answer is what this refuses.
    """
    p, q = _mass(left), _mass(right)
    if p.size == 0 or p.size != q.size or p.sum() <= 0.0 or q.sum() <= 0.0:
        return float("nan")
    # Lazily, so importing this module costs nothing on a box without SciPy and the two analyses
    # that never reach here never pay for it. ``jensenshannon`` normalises both inputs itself.
    from scipy.spatial.distance import jensenshannon

    return float(jensenshannon(p, q, base=JENSEN_SHANNON_BASE))


def wasserstein_seconds(left: Any, right: Any, seconds: Any) -> float:
    r"""The $1$-Wasserstein distance between two per-lag profiles, **in seconds**.

    $$W_1(p, q) = \int \bigl| F_p(\tau) - F_q(\tau) \bigr| \, \mathrm{d}\tau$$

    with both profiles read as distributions over the compensated lag axis $\tau_\ell$. The unit is
    the axis's own, so the number reads directly as "the distribution moved this far" and a reader
    compares it against the $4$ s lag step and the axis's span without a conversion.

    This is the distance that degrades gracefully. Two narrow profiles one lag step apart are one
    step apart here, where :func:`jensen_shannon` -- whose supports barely meet -- would already sit
    near its ceiling. It is correspondingly blind to a change of *width* at a fixed centre, which is
    why the two travel together.

    The axis is **stored-coefficient time**; see
    :data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT`, which every artifact carrying
    this number also carries. A displacement in seconds on this axis is not a displacement in
    physiological latency.

    Args:
        left: One value per lag.
        right: One value per lag, the same length.
        seconds: The compensated lag axis, $(L,)$, ascending.

    Returns:
        The distance in seconds, or ``NaN`` when either profile carries no finite positive mass or
        the three arrays do not share a length.
    """
    p, q = _mass(left), _mass(right)
    axis = np.asarray(seconds, dtype=np.float64).ravel()
    if p.size == 0 or p.size != q.size or p.size != axis.size:
        return float("nan")
    if p.sum() <= 0.0 or q.sum() <= 0.0 or not np.isfinite(axis).all():
        return float("nan")
    # Both distributions live on the same grid, so the axis is passed as both value vectors and the
    # profiles as the weights; ``wasserstein_distance`` normalises the weights itself.
    from scipy.stats import wasserstein_distance

    return float(wasserstein_distance(axis, axis, u_weights=p, v_weights=q))


def centroid_seconds(profile: Any, seconds: Any) -> float:
    r"""One profile's mass-weighted centre on the compensated axis, $\bar\tau = \sum_\ell p_\ell
    \tau_\ell$.

    Duplicated arithmetic in appearance only: ``lag_shape.profile_statistics`` returns this per
    *row of a stack* and is what the per-recording feature columns are built from, while what is
    wanted here is the centre of an already-pooled cell so that a signed difference can travel in
    the distance table beside the two unsigned distances. Reading the pooled cell through the
    stacked reducer would mean assembling a one-row matrix to recover one float, and the sign is
    the point: the two distances are both non-negative and neither says which way the distribution
    moved.

    Args:
        profile: One value per lag.
        seconds: The compensated lag axis, $(L,)$.

    Returns:
        The centroid in seconds, or ``NaN`` when the profile carries no finite positive mass or the
        two arrays do not share a length.
    """
    mass = _mass(profile)
    axis = np.asarray(seconds, dtype=np.float64).ravel()
    total = mass.sum()
    if mass.size == 0 or mass.size != axis.size or total <= 0.0:
        return float("nan")
    return float(np.dot(mass, axis) / total)


def cell_distances(left: Any, right: Any, seconds: Any) -> Dict[str, float]:
    r"""Compare two pooled per-lag cells: both distances and the signed centroid difference.

    The two distances say *how far apart* the cells are and the difference says *which way*, which
    neither of them can: both are non-negative by construction, so a table of them alone cannot
    distinguish a cell whose mass moved toward the anchor from one whose mass moved away.

    The difference is oriented ``left`` minus ``right``, and every caller in this pipeline orders
    its pairs through :func:`~teb_vae.lag_attn_cfs.eval.cohort.ordered_groups` -- worst cohort first
    -- so a positive value means the more severe class sits at the longer lag. That is the same
    orientation Cliff's delta carries in every pairwise table of this package, deliberately: two
    sign conventions for "the worse cohort is higher" is one more than a reader can hold.

    Args:
        left: The first cell's per-lag profile.
        right: The second cell's.
        seconds: The compensated lag axis, $(L,)$.

    Returns:
        ``{jensen_shannon, wasserstein_s, centroid_delta_s}``. Every value is ``NaN`` when either
        cell carries no mass, rather than zero -- zero here is the strongest available claim of
        agreement, and a comparison that could not be made must not make it.
    """
    return {
        "jensen_shannon": jensen_shannon(left, right),
        "wasserstein_s": wasserstein_seconds(left, right, seconds),
        "centroid_delta_s": centroid_seconds(left, seconds) - centroid_seconds(right, seconds),
    }
