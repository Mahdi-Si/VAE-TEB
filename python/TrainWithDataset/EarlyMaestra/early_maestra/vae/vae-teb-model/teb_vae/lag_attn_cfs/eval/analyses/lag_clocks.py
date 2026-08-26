r"""Does the lag structure itself move as delivery -- or the second stage -- approaches?

Three analyses already stand next to this question and none of them answers it. ``lag_kl`` says
*where in the past* the source informed the future, pooled over labour; ``time_to_delivery`` and
``second_stage`` say how *much* coupling there is at each point of two clinical clocks. Nothing
says whether the informative past is closer at delivery than it was six hours earlier, or whether
it moves differently for a cohort that ends badly.

**What is resolved.** Fourteen attributes of a segment's own lag profile, on the compensated axis
$\tau_\ell = 4(\ell + \delta)$ that :mod:`~teb_vae.lag_attn_cfs.eval.lag_axis` builds, with
$p_\ell = w_\ell / \sum_k w_k$. They fall into four families, and the families exist because each
answers a question the others cannot:

* **The moments** -- $\bar\tau = \sum_\ell p_\ell \tau_\ell$ and
  $\sigma_\tau = \sqrt{\sum_\ell p_\ell (\tau_\ell - \bar\tau)^2}$, the centre of mass and its
  spread, plus the skewness that says which side the tail is on.
* **The quantiles** -- the median lag and the inter-quartile range, read off the cumulative mass.
  These profiles are skewed, and one distant bin moves $\bar\tau$ far more than it moves the
  median; where the two disagree, the disagreement is the skew.
* **The concentration** -- the entropy $H = -\sum_\ell p_\ell \log p_\ell$, the effective support
  $\Delta e^{H}$ it implies, and the share of mass near the anchor and far from it. A bimodal
  profile has an unremarkable centroid and an unremarkable spread; only these say it is not one
  lump.
* **The peak** -- where the tallest bin is, how wide it is at half height, how much mass it holds,
  and the two statistics of the guard below.

Every one is computed twice, over the KL attribution and over the attention profile, for the reason
:data:`~teb_vae.lag_attn_cfs.eval.cohort.CLOCK_READOUTS` carries two coupling readouts rather than
one: the two fail differently. The attribution is $K_t$ times the attention and therefore inherits
the prior-variance inflation; the attention profile is the model's own focus and does not. A shift
visible in one and absent from the other is a finding about which of the two is being read.

**The peak is reported, and it is reported with its guard.** ``entmax15`` assigns lags exactly
zero, so a profile that is flat or nearly empty still has a perfectly confident argmax, and a
position quoted without the mechanical criterion that says whether the profile has a shape at all
is not a reading. That criterion used to live in ``lag_kl`` -- which an analysis may not import --
and this analysis therefore reported no peak. It now lives in
:mod:`~teb_vae.lag_attn_cfs.eval.lag_shape`, one layer down, so ``lag_peak_*_s`` travels beside
``lag_peak_degenerate_*`` and ``lag_zero_fraction_*`` in the same row of the same table. A window
whose degenerate share is high has a peak trajectory that means nothing, and the number saying so
is on the page. ``lag_kl_stratified_peaks.csv`` remains where the *pooled* positional reading
lives; this is the per-segment one.

**Both clocks, one implementation.** The two differ in the column they bin on, the sign convention
of their axis, the orientation their figures are drawn in and whether an eligibility rule applies;
everything else -- the aggregation chain, the split, the inference and the marks -- is identical,
so it is written once and driven from :data:`CLOCKS`. The window grid is
:data:`~teb_vae.lag_attn_cfs.eval.cohort.TRAJECTORY_BIN_HOURS`, bound from the layer below, so a
window here is the same duration as a window on either coupling clock and the two pages can be read
against each other.

**Four Holm families, and none of them joint.** Two clocks times two tested readouts. Each
correction controls the family-wise error rate within its own clock and its own readout, and a
reader quoting a window from two of them is making two comparisons. That is the same rule
``second_stage`` states against ``time_to_delivery`` and it is stated again in every record here,
because a family a reader has to infer is one they will infer wrongly.

**The population is the recording, inside the window as well as across it**, and the second clock's
population is a subset: a recording with no recorded onset cannot be placed on that axis at all.
The eligibility rule is :func:`~teb_vae.lag_attn_cfs.eval.cohort.second_stage_eligibility`'s, the
shared one, and its per-recording table is already written by ``second_stage`` -- this analysis
carries the counts rather than a second copy of that table, and declares itself ``capped`` so the
coverage block reads it as a different population by design rather than as a disagreement.

**The axis is stored-coefficient time.** Every lag figure and every record here carries
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT`: a centroid that moves by ninety
seconds is a shift in the attribution over the axis the coefficients are stored on, not a
physiological latency.

.. note::

    lean-limit: the per-window cohort split below is a third copy in this package -- the two
    coupling clocks carry the other two -- and unlike theirs it is already parameterised over both
    clocks; move it into ``cohort.py`` and repoint all three when a fourth consumer needs it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
    read_lag_support,
)
from teb_vae.lag_attn_cfs.eval.lag_shape import (
    DEGENERATE_PEAK_TO_MEDIAN,
    DEGENERATE_ZERO_FRACTION,
    FAR_SECONDS,
    NEAR_SECONDS,
    PEAK_FRACTION,
    STATISTIC_KEYS,
    profile_statistics,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "lag_clocks"

#: What it writes. One set of tables for both clocks, each row naming the clock it belongs to: the
#: two are the same quantities against two landmarks, and splitting them across two file sets would
#: make the comparison a join rather than a filter.
PER_RECORDING_FILENAME = "lag_clocks_per_recording.csv"
TRAJECTORY_FILENAME = "lag_clocks_trajectory.csv"
PROFILE_FILENAME = "lag_clocks_profile.csv"
SIGNIFICANCE_FILENAME = "lag_clocks_significance.csv"
PAIRWISE_FILENAME = "lag_clocks_pairwise.csv"

#: Family-wise error rate each clock's Holm correction controls. Deliberately not an
#: ``eval_config`` key: an operator who could raise it could make any window significant.
DEFAULT_ALPHA = 0.05

#: Window width in hours, bound from the layer below rather than restated, so a window here is the
#: same duration as a window on either coupling clock.
TRAJECTORY_BIN_HOURS = cohort.TRAJECTORY_BIN_HOURS

#: A positional index onto the per-sample vector sidecar, attached to the frame this analysis
#: works on so a binned subset can still find its own rows in a matrix that is aligned with the
#: table's *original* row order. Never emitted: every table below is built from the key columns
#: and the feature columns, and this is neither.
ROW_COLUMN = "_source_row"


@dataclass(frozen=True)
class ProfileSource:
    """One per-lag vector this analysis reduces to scalars.

    Attributes:
        key: The short name the feature columns are built from.
        attribute: The vector readout's name on the sidecar.
        meaning: What the profile is, recorded beside every number derived from it.
    """

    key: str
    attribute: str
    meaning: str


#: The two profiles, and they are two for the reason the clocks carry two coupling readouts: the
#: attribution is $K_t$ times the attention, so it inherits the prior-variance inflation the
#: attention profile is immune to. The **untruncated** attribution rather than the raw one because
#: it is the profile a per-cohort positional claim rests on, and the **support-corrected**
#: attention for the same reason: each lag on its own contributing-anchor count.
PROFILE_SOURCES: Tuple[ProfileSource, ...] = (
    ProfileSource(
        "kl",
        "lag_profile_untruncated",
        "the per-lag KL attribution, over the anchors whose lag support is complete",
    ),
    ProfileSource(
        "attn",
        "attention_profile_support_corrected",
        "the attention over the lags, each on its own contributing-anchor count",
    ),
)


@dataclass(frozen=True)
class Statistic:
    """One scalar :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.profile_statistics` returns.

    A statistic is a *reduction*, not a column: it says nothing about which profile it was taken
    over. Crossing it with :data:`PROFILE_SOURCES` is what makes a column, and writing the two
    axes separately is what keeps them from drifting -- a statistic added here reaches both
    profiles, and there is no way to add it to one and forget the other.

    Attributes:
        key: The name the reducer returns it under, one of
            :data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS`.
        suffix: The unit, spelled into the column name -- ``"_s"`` for seconds, ``"_nats"`` for
            an entropy, empty for a dimensionless share or ratio. A column whose unit a reader
            has to infer is one they will infer wrongly.
        unit: The $y$-axis label of a panel drawing it. ``None`` takes the shared lag axis label,
            which is what every seconds-valued statistic is drawn on.
        drawn: Whether the features page carries a panel for it. Five do not. The centroid and the
            spread are already the two reduced panels of the *profile* page, and drawing them twice
            would put one quantity on two pages under two ribbons. The peak's width and mass and
            the zero fraction are each a second reading of a panel that is on this page -- the
            peak, and the degenerate share it is guarded by -- and the page is nine rows as it
            stands. All five reach both tables regardless.
        tested: Whether the three-layer inference runs on it. Only the centroid is, which keeps
            each clock's Holm family at two rather than at twenty-eight and the windows page at
            five rows. Promoting a statistic is this flag and nothing else.
        meaning: What the number is, written into every record.
    """

    key: str
    suffix: str
    unit: Optional[str]
    drawn: bool
    tested: bool
    meaning: str


#: The fourteen statistics, in the order every table carries them: the moments, the quantiles, the
#: concentration, then the peak and the guard that says whether to read it.
STATISTICS: Tuple[Statistic, ...] = (
    Statistic(
        "centroid", "_s", None, drawn=False, tested=True,
        meaning=(
            "centre of mass of the profile, in compensated seconds; smaller means the mass sits "
            "nearer the anchor"
        ),
    ),
    Statistic(
        "spread", "_s", None, drawn=False, tested=False,
        meaning=(
            "mass-weighted spread about the centre, in seconds; smaller means the mass is "
            "concentrated rather than diffuse"
        ),
    ),
    Statistic(
        "skewness", "", "skewness (dimensionless)", drawn=True, tested=False,
        meaning=(
            "third standardised moment; positive means the tail runs toward the longer "
            "lags, so the centroid sits further from the anchor than the median does"
        ),
    ),
    Statistic(
        "median", "_s", None, drawn=True, tested=False,
        meaning=(
            "the lag at which half the mass has accumulated, in seconds; the robust centre, "
            "which a single distant bin moves far less than it moves the centroid"
        ),
    ),
    Statistic(
        "iqr", "_s", None, drawn=True, tested=False,
        meaning=(
            "inter-quartile range of the mass, in seconds; the robust spread, and the width "
            "a skewed profile is honestly described by"
        ),
    ),
    Statistic(
        "entropy", "_nats", "entropy (nats)", drawn=True, tested=False,
        meaning=(
            "Shannon entropy of the profile over the lags, in nats; higher means the mass is "
            "spread over more lags, with no reference to where its centre sits"
        ),
    ),
    Statistic(
        "effective_support", "_s", None, drawn=True, tested=False,
        meaning=(
            "the width a uniform profile of the same entropy would occupy, in seconds; how "
            "many lags the mass effectively covers, independent of where its centre is"
        ),
    ),
    Statistic(
        "near_mass", "", "share of the mass", drawn=True, tested=False,
        meaning=(
            f"share of the mass within {NEAR_SECONDS:g} s of the shortest lag the axis carries; "
            "measured from the axis's own start, so it means the same thing at any causal delay"
        ),
    ),
    Statistic(
        "far_mass", "", "share of the mass", drawn=True, tested=False,
        meaning=(
            f"share of the mass beyond {FAR_SECONDS:g} s from the shortest lag the axis carries"
        ),
    ),
    Statistic(
        "peak", "_s", None, drawn=True, tested=False,
        meaning=(
            "the lag holding the most mass, in compensated seconds; read only where "
            "lag_peak_degenerate is low, and see that column's meaning for why"
        ),
    ),
    Statistic(
        "peak_degenerate", "", "share of segments", drawn=True, tested=False,
        meaning=(
            "1 where the profile has no shape worth reading -- peak-to-median below "
            f"{DEGENERATE_PEAK_TO_MEDIAN}, or more than {DEGENERATE_ZERO_FRACTION:.0%} of bins "
            "exactly zero -- so the per-recording mean is the share of that recording's segments "
            "whose peak names a bin rather than a lag"
        ),
    ),
    Statistic(
        "peak_width", "_s", None, drawn=False, tested=False,
        meaning=(
            f"full width of the peak at {PEAK_FRACTION:g} of its height, in seconds, over the "
            "contiguous run of bins around it"
        ),
    ),
    Statistic(
        "peak_mass", "", "share of the mass", drawn=False, tested=False,
        meaning=(
            f"share of the mass in bins at or above {PEAK_FRACTION:g} of the peak; the "
            "concentration the peak's position does not report"
        ),
    ),
    Statistic(
        "zero_fraction", "", "share of bins", drawn=False, tested=False,
        meaning=(
            "share of the profile's finite bins that are exactly zero; entmax15 sparsifies, "
            "so this is expected to be non-zero and is the second half of the criterion above"
        ),
    ),
)


@dataclass(frozen=True)
class Feature:
    """One scalar summary of one profile, per segment -- a :class:`Statistic` on a source.

    Attributes:
        column: The column it is carried under, which is also the name it is reported under --
            one vocabulary rather than a column name and a display name that can drift apart.
        source: Which of :data:`PROFILE_SOURCES` it is computed from.
        statistic: Which reduction it is.
        tested: Whether the three-layer inference runs on it; see :class:`Statistic`.
        meaning: What the number is, written into every record.
    """

    column: str
    source: str
    statistic: str
    tested: bool
    meaning: str


def _feature_column(source_key: str, statistic: Statistic) -> str:
    """Build the column name one statistic is carried under for one profile.

    Assembled in one place rather than formatted at each use, for the reason the lag axis is: a
    name built its own way in a second file is how a writer and a reader come to disagree about
    which column holds which number.

    Args:
        source_key: The profile's short name, ``"kl"`` or ``"attn"``.
        statistic: The reduction.

    Returns:
        ``lag_<statistic>_<source><suffix>`` -- for example ``lag_centroid_kl_s``, which is the
        name this analysis has always used for the first of them.
    """
    return f"lag_{statistic.key}_{source_key}{statistic.suffix}"


#: The features: every statistic on every profile, source-major inside each statistic so the two
#: readings of one quantity sit next to each other in every table.
FEATURES: Tuple[Feature, ...] = tuple(
    Feature(
        _feature_column(source.key, statistic),
        source.key,
        statistic.key,
        statistic.tested,
        f"{statistic.meaning} -- over {source.meaning}",
    )
    for statistic in STATISTICS
    for source in PROFILE_SOURCES
)

#: The columns, in table order.
FEATURE_COLUMNS: Tuple[str, ...] = tuple(feature.column for feature in FEATURES)

#: The features the inference runs on, in the order their panels are drawn.
READOUTS: Tuple[Feature, ...] = tuple(feature for feature in FEATURES if feature.tested)

#: The statistics the features page draws, in panel order.
DRAWN_STATISTICS: Tuple[Statistic, ...] = tuple(
    statistic for statistic in STATISTICS if statistic.drawn
)


@dataclass(frozen=True)
class Clock:
    """One clinical landmark and everything that differs about reading a quantity against it.

    Attributes:
        name: The clock's name, carried on every row of every table so the two are one file set.
        binner: The shared binning function for this landmark.
        bin_column: The window index column it adds.
        center_column: The window centre column travelling with it.
        axis_label: The x-axis label, naming the sign convention outright.
        inverted: Whether the axis is drawn with the landmark at the right. True for time before
            delivery, which decreases toward the event; false for the signed second-stage axis,
            which reads naturally left to right with the onset marked where it falls.
        figure: The profile page's filename.
        windows_figure: The tested page's filename.
        features_figure: The untested statistics' page's filename. Its own page rather than more
            rows on the profile one, because half of what it draws is not in seconds and a panel
            sharing an axis with a quantity in different units is a panel that will be read
            against it.
        eligible_only: Whether the second-stage eligibility rule applies before binning.
    """

    name: str
    binner: Callable[[pd.DataFrame], pd.DataFrame]
    bin_column: str
    center_column: str
    axis_label: str
    inverted: bool
    figure: str
    windows_figure: str
    features_figure: str
    eligible_only: bool


#: The two clocks. The axis labels are restated here rather than imported because an analysis may
#: not import another and ``cohort.py`` owns the arithmetic rather than the captions; the second
#: one names its sign convention outright, because a reader who takes a negative value for "after"
#: reads the whole trajectory backwards and nothing on the page would contradict them.
CLOCKS: Tuple[Clock, ...] = (
    Clock(
        name="time_to_delivery",
        binner=cohort.add_time_bins,
        bin_column=cohort.BIN_COLUMN,
        center_column=cohort.BIN_CENTER_COLUMN,
        axis_label="Time before delivery (hours)",
        inverted=True,
        figure="lag_time_to_delivery",
        windows_figure="lag_time_to_delivery_windows",
        features_figure="lag_time_to_delivery_features",
        eligible_only=False,
    ),
    Clock(
        name="second_stage",
        binner=cohort.add_second_stage_bins,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        axis_label="Hours from second-stage onset (negative = before onset, positive = after)",
        inverted=False,
        figure="lag_second_stage",
        windows_figure="lag_second_stage_windows",
        features_figure="lag_second_stage_features",
        eligible_only=True,
    ),
)

#: The method sentence written into every record, so a $p$-value here is readable without this
#: module -- including which family it belongs to and what the sign of an effect size means.
METHOD = (
    "Per window of the named clock: Kruskal-Wallis across clinical classes over one value per "
    "recording, Holm step-down correction across that clock's windows as one family, pairwise "
    "two-sided Mann-Whitney U with Cliff's delta for the windows significant after Holm only. "
    "Every pair is oriented from the more severe class to the less severe one, so a positive "
    "Cliff's delta means the more severe class's values run higher. Non-parametric "
    "throughout. Classes "
    f"with fewer than {shared_stats.MIN_GROUP_SIZE} recordings in a window are excluded from it "
    "and recorded. Each (clock, readout) is its own Holm family and the four are NOT corrected "
    "jointly: two clocks are different alignments of an overlapping population and the two "
    "profiles are two readings of the same recordings, so a reader quoting two of them is making "
    "two comparisons."
)

#: How the peak reported here is guarded, in the record rather than left for a reader to look for.
#: A position quoted without this sentence is the failure the sentence exists to prevent.
PEAK_REFERENCE = (
    f"lag_peak_*_s is an argmax and must be read beside lag_peak_degenerate_*, never alone: "
    f"entmax15 assigns lags exactly zero, so a flat or nearly empty profile still has a perfectly "
    f"confident argmax. A segment counts as degenerate when its peak-to-median ratio is below "
    f"{DEGENERATE_PEAK_TO_MEDIAN} or more than {DEGENERATE_ZERO_FRACTION:.0%} of its finite bins "
    f"are exactly zero, and the per-recording mean of that flag is the share of a window's "
    f"segments whose peak names a bin rather than a lag. The pooled positional reading, over whole "
    f"recordings rather than per segment, remains lag_kl/lag_kl_stratified_peaks.csv"
)


# =================================================================================================
# The scalars, from one segment's profile
# =================================================================================================
def add_feature_columns(
    per_sample: pd.DataFrame, vectors: Dict[str, np.ndarray], seconds: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Attach every per-segment feature to a copy of the per-sample table.

    The arithmetic itself is
    :func:`~teb_vae.lag_attn_cfs.eval.lag_shape.profile_statistics`, called once per profile
    rather than once per statistic: the fourteen share a normalisation, a cumulative sum and a
    mask, and computing them in one pass is what keeps the reduction over tens of thousands of
    segments a matter of a dozen matrix products.

    Args:
        per_sample: The collected per-sample table.
        vectors: The per-sample vector readouts, aligned with its row order.
        seconds: The compensated lag axis.

    Returns:
        ``(frame, record)`` -- a copy carrying :data:`FEATURE_COLUMNS` and :data:`ROW_COLUMN`, and
        one entry per profile saying how many segments it reduced. A profile absent from the
        sidecar yields ``NaN`` columns and a recorded reason rather than an exception: a run
        directory collected before that readout existed is a partial input, not a broken one.
    """
    frame = per_sample.copy()
    frame[ROW_COLUMN] = np.arange(len(frame), dtype=np.int64)
    record: Dict[str, Any] = {}
    for source in PROFILE_SOURCES:
        matrix = vectors.get(source.attribute)
        columns = {
            statistic.key: _feature_column(source.key, statistic) for statistic in STATISTICS
        }
        if matrix is None or np.asarray(matrix).ndim != 2:
            for column in columns.values():
                frame[column] = np.nan
            record[source.key] = {
                "attribute": source.attribute,
                "meaning": source.meaning,
                "n_usable": 0,
                "reason": f"the vector sidecar carries no 2-D {source.attribute!r}",
            }
            continue
        rows = np.asarray(matrix, dtype=np.float64)[: len(frame)]
        statistics, counts = profile_statistics(rows, seconds)
        for key, column in columns.items():
            frame[column] = _padded(statistics[key], len(frame))
        record[source.key] = {
            "attribute": source.attribute,
            "meaning": source.meaning,
            **counts,
        }
    return frame, record


def _padded(values: np.ndarray, size: int) -> np.ndarray:
    """Return ``values`` at the table's length, ``NaN``-filled where the sidecar was shorter."""
    if values.size == size:
        return values
    padded = np.full(size, np.nan, dtype=np.float64)
    padded[: min(values.size, size)] = values[:size]
    return padded


# =================================================================================================
# Placing the segments on a clock
# =================================================================================================
def clock_rows(clock: Clock, featured: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin the featured table on one clock, applying that clock's population rule first.

    Args:
        clock: The clock to place the segments on.
        featured: The per-sample table with the feature columns attached.

    Returns:
        ``(binned, population)`` -- the binned rows, and the record of who is on this clock.
        On the second-stage clock that record is the eligibility count: **one rule drops a
        recording, it has no onset**, and the two diagnostics
        :func:`~teb_vae.lag_attn_cfs.eval.cohort.second_stage_eligibility` reports beside it are
        counted and filter nothing. The per-recording eligibility table itself is written by the
        ``second_stage`` analysis and is not duplicated here.
    """
    population: Dict[str, Any] = {"clock": clock.name}
    rows = featured
    if clock.eligible_only:
        eligibility = cohort.second_stage_eligibility(featured)
        admitted = sorted(
            set(eligibility.loc[eligibility["eligible"].astype(bool), "guid"].astype(str))
        ) if len(eligibility) else []
        population.update(
            {
                "n_recordings": int(len(eligibility)),
                "n_eligible": len(admitted),
                "n_dropped_no_onset": int(len(eligibility) - len(admitted)),
                "n_onset_at_delivery": int(
                    eligibility.loc[
                        eligibility["eligible"].astype(bool), "onset_at_delivery"
                    ].astype(bool).sum()
                ) if len(eligibility) else 0,
                "n_inconsistent_onset": int(
                    eligibility.loc[
                        eligibility["eligible"].astype(bool), "inconsistent_onset"
                    ].astype(bool).sum()
                ) if len(eligibility) else 0,
                "onset_consistency_tolerance_s": float(cohort.ONSET_CONSISTENCY_TOLERANCE_S),
                "eligibility_table": "second_stage/second_stage_eligibility.csv",
            }
        )
        rows = (
            featured[featured["guid"].astype(str).isin(admitted)]
            if "guid" in featured.columns
            else featured.iloc[:0]
        )
    binned = clock.binner(rows)
    population["n_segments"] = int(len(binned))
    population["n_windows"] = (
        int(binned[clock.bin_column].nunique()) if len(binned) else 0
    )
    return binned, population


def per_recording_frames(clock: Clock, binned: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Reduce a binned table to one row per (cohort, window, recording), on both cohort axes.

    Both axes because the emitted table is what a reader recomputes any cell of the tables above
    it from, and the subgroup cut is legitimate to *read* even though only the class cut is
    tested: eight cohorts inside a half-hour window rarely clear the floor a test needs.

    Args:
        clock: The clock whose window columns to group on.
        binned: The binned per-sample table.

    Returns:
        Cohort axis to its per-recording-per-window frame.
    """
    return {
        axis: cohort.per_recording_in_bins(
            binned,
            FEATURE_COLUMNS,
            group_column=axis,
            bin_column=clock.bin_column,
            center_column=clock.center_column,
        )
        for axis in labels.GROUP_COLUMNS
    }


def trajectory_rows(clock: Clock, per_recording: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """Summarise every feature within each (cohort, window) cell, over its recordings.

    Args:
        clock: The clock the windows belong to.
        per_recording: The per-recording frames, by cohort axis.

    Returns:
        Long-form rows: the clock, the cohort axis, the feature, the window and its centre, the
        count of **recordings** behind it, and the mean with its quartiles.
    """
    rows: List[Dict[str, Any]] = []
    for axis, frame in per_recording.items():
        for feature in FEATURES:
            for row in cohort.trajectory_rows(
                frame,
                feature.column,
                metric=feature.column,
                bin_column=clock.bin_column,
                center_column=clock.center_column,
            ):
                rows.append({"clock": clock.name, "group_column": axis, **row})
    return rows


# =================================================================================================
# Significance across the classes, window by window
# =================================================================================================
def _class_values(frame: pd.DataFrame, column: str) -> Dict[str, np.ndarray]:
    """Split one window's rows into per-class finite vectors, dropping none of them.

    Args:
        frame: One window's per-recording rows.
        column: The feature column.

    Returns:
        Every class present, in the canonical HIE / acidosis / healthy order, mapped to its finite
        values. **Nothing is filtered here**: a class too small to test is still a class a figure
        must show. The order is load-bearing rather than presentational -- the pairwise sweep names
        each pair in the order it receives the classes, so this is what makes every comparison read
        more severe against less severe.
    """
    values_by_class: Dict[str, np.ndarray] = {}
    if frame.empty or "group" not in frame.columns or column not in frame.columns:
        return values_by_class
    for group in cohort.ordered_groups(
        sorted(set(frame["group"].astype(str))), labels.CLASS_COLUMN
    ):
        values = np.asarray(
            frame.loc[frame["group"].astype(str) == group, column], dtype=np.float64
        )
        values_by_class[group] = values[np.isfinite(values)]
    return values_by_class


def window_samples(
    clock: Clock, per_recording: pd.DataFrame, column: str
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
    """Split a per-recording frame into the cells one feature is tested and drawn from.

    One function rather than two because the figure and the test must describe the *same* cells: a
    violin drawn from one set of values under a $p$-value computed from another is a page that
    disagrees with itself and looks entirely ordinary.

    Args:
        clock: The clock whose windows to cut on.
        per_recording: The class-axis per-recording-per-window frame.
        column: The feature column.

    Returns:
        ``(samples, meta)`` keyed by window, in ascending window order. ``samples`` holds every
        class present, unfiltered -- :func:`testable_windows` applies the floor, because the figure
        must draw the cells the test may not use or a cohort thinning out toward the edge of the
        axis simply disappears. ``meta`` holds the window's centre, the classes excluded as too
        small, and the floor they were excluded at.
    """
    samples: Dict[int, Dict[str, np.ndarray]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    if per_recording.empty or clock.bin_column not in getattr(per_recording, "columns", []):
        return samples, meta
    for window in sorted(int(value) for value in per_recording[clock.bin_column].unique()):
        cell = per_recording[per_recording[clock.bin_column] == window]
        values_by_class = _class_values(cell, column)
        samples[int(window)] = values_by_class
        meta[int(window)] = {
            "bin_center_h": float(cell[clock.center_column].iloc[0]),
            "groups_excluded_as_too_small": {
                group: int(values.size)
                for group, values in values_by_class.items()
                if values.size < shared_stats.MIN_GROUP_SIZE
            },
            "min_group_size": shared_stats.MIN_GROUP_SIZE,
        }
    return samples, meta


def testable_windows(
    samples: Dict[int, Dict[str, np.ndarray]]
) -> Dict[int, Dict[str, np.ndarray]]:
    """Drop the classes too small to test, keeping every window.

    The window itself is kept even when nothing in it survives: a window that could not be tested
    has to reach the output as such, or a reader cannot tell it from a window nobody looked at.

    Args:
        samples: The unfiltered cells from :func:`window_samples`.

    Returns:
        The same windows, each holding only the classes with at least ``MIN_GROUP_SIZE``
        recordings.
    """
    return {
        window: {
            group: values
            for group, values in cell.items()
            if values.size >= shared_stats.MIN_GROUP_SIZE
        }
        for window, cell in samples.items()
    }


def analyse_windows(
    clock: Clock, per_recording: pd.DataFrame, column: str, *, alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Test whether the class trajectories of one feature differ, window by window.

    Args:
        clock: The clock the windows belong to.
        per_recording: The class-axis per-recording-per-window frame.
        column: The feature column to test.
        alpha: Family-wise error rate the Holm correction controls.

    Returns:
        The record: whether the test could run, the per-window omnibus results with their
        Holm-adjusted $p$-values, and the pairwise comparisons for the windows that survived.
        ``tested`` is ``False`` with a reason on a split carrying fewer than two classes, which is
        the ordinary outcome on the healthy-only pretraining split rather than a failure.
    """
    groups = (
        cohort.ordered_groups(
            sorted(set(per_recording["group"].astype(str))), labels.CLASS_COLUMN
        )
        if not per_recording.empty and "group" in per_recording.columns
        else []
    )
    base: Dict[str, Any] = {
        "clock": clock.name,
        "metric_column": column,
        "group_column": labels.CLASS_COLUMN,
        "alpha": float(alpha),
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "method": METHOD,
    }
    if len(groups) < 2:
        return {
            **base,
            "tested": False,
            "reason": f"fewer than two clinical classes present ({groups or 'none'})",
            "per_window": [],
            "pairwise": {},
        }

    # The cohort half of the job is this analysis's own: which classes are in a window, in which
    # order, and which were too small to enter the test. That order is load-bearing rather than
    # cosmetic: the pairwise sweep names each pair in the order it receives, so passing the
    # classes severity-descending is what makes every comparison read HIE against acidosis, HIE
    # against healthy, acidosis against healthy -- more severe against less severe, never the
    # reverse.
    # The arithmetic half -- the omnibus, Holm across the windows and the pairwise sweep on the
    # survivors -- is ``stats.windowed_group_comparisons``, shared with every other trajectory
    # analysis in the family, so that "significant" has one definition here rather than one per
    # clock.
    samples, meta = window_samples(clock, per_recording, column)
    outcome = shared_stats.windowed_group_comparisons(
        testable_windows(samples), meta_by_window=meta, alpha=alpha
    )
    significant = [record for record in outcome["per_window"] if record["significant"]]
    return {
        **base,
        "tested": True,
        "classes": groups,
        "n_windows": outcome["n_windows"],
        "n_windows_tested": outcome["n_windows_tested"],
        "n_significant_windows": len(significant),
        "significant_bin_centers_h": [record["bin_center_h"] for record in significant],
        "per_window": outcome["per_window"],
        "pairwise": outcome["pairwise"],
    }


# =================================================================================================
# The profile as a function of the clock
# =================================================================================================
@dataclass(frozen=True)
class ProfileField:
    """One cohort's mean lag profile across a clock's windows.

    Attributes:
        group: The cohort.
        mean: The mean profile, $(L, W)$ -- lag down, window across, over recordings.
        share: The same field with every window normalised to sum to one, which is what makes it
            a distribution over lags rather than a picture of the coupling magnitude the clocks
            already draw.
        counts: Recordings behind each window, $(W,)$.
    """

    group: str
    mean: np.ndarray
    share: np.ndarray
    counts: np.ndarray


def window_profiles(
    clock: Clock, binned: pd.DataFrame, matrix: Any, n_lags: int
) -> Tuple[List[int], List[float], List[ProfileField]]:
    """Average a per-lag vector within each recording, then within each (class, window) cell.

    The aggregation chain applied *inside* a window, exactly as the scalar features go through
    :func:`~teb_vae.lag_attn_cfs.eval.cohort.per_recording_in_bins`: a recording contributing
    eleven segments to a window must not outvote one contributing two.

    Args:
        clock: The clock whose windows to cut on.
        binned: The binned per-sample table, carrying :data:`ROW_COLUMN`.
        matrix: The per-sample vectors, $(n, L)$, in the **unbinned** table's row order.
        n_lags: The lag axis width, so an empty cohort still produces a field of the right shape.

    Returns:
        ``(windows, centres, fields)`` -- the window indices every cohort's field is laid out on,
        in ascending order, their centres in hours, and one :class:`ProfileField` per class in the
        canonical cohort order. Every cohort is laid out on the *same* window axis, with ``NaN``
        where it has no recordings there, so the panels drawn from them can be read against one
        another column by column.
    """
    values = np.asarray(matrix, dtype=np.float64) if matrix is not None else np.zeros((0, 0))
    needed = {labels.CLASS_COLUMN, "guid", clock.bin_column, clock.center_column, ROW_COLUMN}
    if (
        binned.empty
        or values.ndim != 2
        or values.shape[1] != int(n_lags)
        or not needed <= set(binned.columns)
    ):
        return [], [], []

    frame = binned[binned[labels.CLASS_COLUMN].notna()]
    if frame.empty:
        return [], [], []
    positions = np.asarray(frame[ROW_COLUMN], dtype=np.int64)
    positions = positions[positions < values.shape[0]]
    if positions.size != len(frame):
        return [], [], []

    columns = list(range(int(n_lags)))
    table = pd.DataFrame(values[positions], columns=columns)
    table["group"] = [str(value) for value in frame[labels.CLASS_COLUMN]]
    table["window"] = [int(value) for value in frame[clock.bin_column]]
    table["centre"] = [float(value) for value in frame[clock.center_column]]
    table["guid"] = list(frame["guid"].astype(str))

    per_guid = table.groupby(["group", "window", "guid"], sort=True)[columns].mean()
    per_cell = per_guid.groupby(["group", "window"], sort=True)[columns].mean()
    # Recordings that measured something, not rows: a recording whose every segment scored no
    # anchors is an all-NaN row the means correctly skip, so counting it here would label a cell
    # with evidence that did not go into it.
    counted = per_guid.notna().any(axis=1).groupby(["group", "window"], sort=True).sum()

    windows = sorted({int(value) for value in table["window"]})
    centres = [
        float(table.loc[table["window"] == window, "centre"].iloc[0]) for window in windows
    ]
    present = cohort.ordered_groups(
        sorted({str(value) for value in table["group"]}), labels.CLASS_COLUMN
    )
    fields: List[ProfileField] = []
    for group in present:
        mean = np.full((int(n_lags), len(windows)), np.nan)
        counts = np.zeros(len(windows), dtype=np.int64)
        for index, window in enumerate(windows):
            if (group, window) not in per_cell.index:
                continue
            mean[:, index] = np.asarray(per_cell.loc[(group, window)], dtype=np.float64)
            counts[index] = int(counted.loc[(group, window)])
        totals = np.nansum(mean, axis=0)
        share = np.divide(
            mean, np.where(totals > 0.0, totals, np.nan)[None, :],
            out=np.full_like(mean, np.nan), where=np.isfinite(mean),
        )
        fields.append(ProfileField(group=group, mean=mean, share=share, counts=counts))
    return windows, centres, fields


# =================================================================================================
# Emission
# =================================================================================================
def profile_frame(
    clock: Clock,
    source: ProfileSource,
    windows: Sequence[int],
    centres: Sequence[float],
    fields: Sequence[ProfileField],
    seconds: np.ndarray,
) -> pd.DataFrame:
    """Lay the per-cell mean profiles out long-form, one row per (cohort, window, lag)."""
    rows: List[Dict[str, Any]] = []
    for field in fields:
        for index, window in enumerate(windows):
            for lag in range(int(seconds.size)):
                rows.append(
                    {
                        "clock": clock.name,
                        "group_column": labels.CLASS_COLUMN,
                        "group": field.group,
                        "profile": source.key,
                        "time_bin": int(window),
                        "bin_center_h": float(centres[index]),
                        "n_recordings": int(field.counts[index]),
                        "lag_step": int(lag),
                        "compensated_seconds": float(seconds[lag]),
                        "mean_value": float(field.mean[lag, index]),
                        "share": float(field.share[lag, index]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "group_column", "group", "profile", "time_bin", "bin_center_h",
            "n_recordings", "lag_step", "compensated_seconds", "mean_value", "share",
        ],
    )


def significance_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the per-window omnibus results of every (clock, readout) into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        for window in record.get("per_window") or []:
            rows.append(
                {
                    "clock": record["clock"],
                    "metric_column": record["metric_column"],
                    "time_bin": window["time_bin"],
                    "bin_center_h": window["bin_center_h"],
                    "n_classes": window.get("n_groups"),
                    "n_recordings": sum((window.get("n_per_group") or {}).values()),
                    "statistic": window.get("statistic"),
                    "p_value": window.get("p_value"),
                    "p_holm": window.get("p_holm", float("nan")),
                    "significant": window.get("significant", False),
                    "alpha": window.get("alpha", float("nan")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "metric_column", "time_bin", "bin_center_h", "n_classes", "n_recordings",
            "statistic", "p_value", "p_holm", "significant", "alpha",
        ],
    )


def pairwise_frame(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten the surviving windows' pairwise comparisons into one table."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        centres = {
            int(window["time_bin"]): window["bin_center_h"]
            for window in record.get("per_window") or []
        }
        for key, comparisons in (record.get("pairwise") or {}).items():
            for item in comparisons:
                rows.append(
                    {
                        "clock": record["clock"],
                        "metric_column": record["metric_column"],
                        "time_bin": int(key),
                        "bin_center_h": centres.get(int(key), float("nan")),
                        "left": item["left"],
                        "right": item["right"],
                        "n_left": item["n_left"],
                        "n_right": item["n_right"],
                        "p_value": item["p_value"],
                        "cliffs_delta": item["cliffs_delta"],
                        "magnitude": item["magnitude"],
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "clock", "metric_column", "time_bin", "bin_center_h", "left", "right", "n_left",
            "n_right", "p_value", "cliffs_delta", "magnitude",
        ],
    )


# =================================================================================================
# The figures
# =================================================================================================
def build_profile_figure(
    clock: Clock,
    windows: Sequence[int],
    centres: Sequence[float],
    fields: Sequence[ProfileField],
    rows: Sequence[Dict[str, Any]],
    seconds: np.ndarray,
) -> Any:
    r"""Draw where the informative past sits, window by window, and how that moves.

    One heatmap per class -- lag down, clock across, colour the **share** of the attribution in
    that lag bin -- and then one panel per tested readout carrying the same thing reduced to a
    number: the median centroid across recordings with its inter-quartile ribbon, and the median
    spread as a dashed line beside it.

    **The class panels share one colour scale**, which is not a detail. Three panels each scaled to
    its own extremes paint the same colour for three different shares, and the comparison the three
    panels exist for is then the one thing they cannot support -- while every colourbar stays
    correct, so nothing in the numbers gives it away.

    **The share, not the magnitude.** Every window is normalised to sum to one, so the field
    answers *where* the attribution sits rather than *how much* of it there is; how much is what
    ``time_to_delivery`` and ``second_stage`` already draw, and a field carrying both would move
    for either reason with nothing saying which.

    Args:
        clock: The clock being drawn.
        windows: The window indices, ascending.
        centres: Their centres in hours.
        fields: One field per class, in the canonical order.
        rows: This clock's trajectory rows, for the two reduced panels.
        seconds: The compensated lag axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(max(len(fields), 1) + len(READOUTS), height_per_row=3.0)
    limit = max(
        (float(np.nanmax(field.share)) for field in fields if np.isfinite(field.share).any()),
        default=0.0,
    )
    half = float(TRAJECTORY_BIN_HOURS) / 2.0
    extent = (
        (float(centres[0]) - half, float(centres[-1]) + half,
         float(seconds[0]), float(seconds[-1]))
        if centres and seconds.size
        else None
    )
    for index, field in enumerate(fields):
        axis = axes[index, 0]
        figures.heatmap_with_colorbar(
            figure, axis,
            # Row-reversed: the shared panel draws with ``origin='upper'``, so data row 0 lands at
            # the top of the extent -- which with an increasing seconds extent would put lag 0 at
            # the largest label and silently invert the whole panel.
            field.share[::-1],
            title=(
                f"{field.group}: share of the KL attribution by lag and window "
                f"(n={int(field.counts.max()) if field.counts.size else 0} recordings at most)"
            ),
            ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
            symmetric=False,
            vlimits=(0.0, limit) if limit > 0.0 else None,
            colorbar_label="share of the attribution",
            extent=extent,
            interpolation="none",
        )
        if clock.inverted:
            axis.invert_xaxis()
        else:
            axis.axvline(
                0.0, color=figures.COLOR_LIGHT_GRAY, linestyle=":",
                linewidth=figures.LINE_REGULAR,
            )
        axis.set_xlabel(clock.axis_label)
    if not fields:
        axes[0, 0].text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=axes[0, 0].transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        figures.style_axes(axes[0, 0])

    for offset, feature in enumerate(READOUTS):
        _draw_trajectory_panel(
            axes[max(len(fields), 1) + offset, 0], clock, rows, feature,
            title=f"{feature.column} against the clock, by {labels.CLASS_COLUMN}",
        )
    figures.caveat_note(figure)
    return figure


def _draw_trajectory_panel(
    ax: Any, clock: Clock, rows: Sequence[Dict[str, Any]], feature: Feature, *, title: str
) -> int:
    """Draw one readout's class trajectories: median centroid with its ribbon, and the spread.

    The recording count is annotated per window rather than reported once, because it is what a
    trajectory hides: a window's median can move because the cohort changed rather than because the
    lag structure did, and the only thing that says which is the number behind the point.

    Args:
        ax: Target axes.
        clock: The clock, for the orientation and the axis label.
        rows: This clock's trajectory rows, both features and every cohort axis.
        feature: The centroid feature to draw; its cohort's spread is drawn beside it.
        title: Panel title.

    Returns:
        The number of cohorts drawn. Zero draws the empty note instead.
    """
    spread_column = f"lag_spread_{feature.source}_s"
    selected = [
        row for row in rows
        if row["group_column"] == labels.CLASS_COLUMN and row["clock"] == clock.name
    ]
    centroid_rows = [row for row in selected if row["metric"] == feature.column]
    spread_rows = [row for row in selected if row["metric"] == spread_column]
    groups = cohort.ordered_groups(
        [row["group"] for row in centroid_rows], labels.CLASS_COLUMN
    ) if centroid_rows else []
    if not groups:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return 0

    # From this package's one cohort palette, so a class is the same green / amber / red here as on
    # every other figure this evaluation draws of it.
    colours = figures.group_colors(groups)
    # The dashed convention is named once rather than per cohort: three more legend entries saying
    # the same thing would cover the lines the panel exists to show.
    spread_labelled = False
    for group in groups:
        cell = sorted(
            (row for row in centroid_rows if row["group"] == group),
            key=lambda row: row["bin_center_h"],
        )
        if not cell:
            continue
        x = np.array([row["bin_center_h"] for row in cell], dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x,
            np.array([row["q25"] for row in cell], dtype=np.float64),
            np.array([row["q75"] for row in cell], dtype=np.float64),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, np.array([row["median"] for row in cell], dtype=np.float64),
            marker="o", markersize=3, color=colour, linewidth=figures.LINE_EMPHASIS,
            label=f"{group} centroid (n={int(sum(row['n_recordings'] for row in cell))})",
        )
        for row in cell:
            ax.annotate(
                str(int(row["n_recordings"])),
                (float(row["bin_center_h"]), float(row["median"])),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontsize=figures.FONT_TINY, color=colour,
            )
        spread_cell = sorted(
            (row for row in spread_rows if row["group"] == group),
            key=lambda row: row["bin_center_h"],
        )
        if spread_cell:
            ax.plot(
                np.array([row["bin_center_h"] for row in spread_cell], dtype=np.float64),
                np.array([row["median"] for row in spread_cell], dtype=np.float64),
                linestyle="--", color=colour, linewidth=figures.LINE_THIN,
                label="median spread (dashed)" if not spread_labelled else "_nolegend_",
            )
            spread_labelled = True
    ax.set_title(title)
    ax.set_xlabel(clock.axis_label)
    ax.set_ylabel(figures.COEFFICIENT_LAG_AXIS_LABEL)
    if clock.inverted:
        # Delivery sits at the right, so the eye reads left to right toward it.
        ax.invert_xaxis()
    else:
        ax.axvline(
            0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR, zorder=0
        )
    ax.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
    figures.style_axes(ax)
    return len(groups)


def build_windows_figure(
    clock: Clock, class_frame: pd.DataFrame, records: Sequence[Dict[str, Any]]
) -> Any:
    """Draw what the centroid trajectory is made of: the distributions, their $p$ and the effects.

    Args:
        clock: The clock being drawn.
        class_frame: The class-axis per-recording-per-window frame.
        records: The significance records of this clock, in :data:`READOUTS` order.

    Returns:
        The figure; the caller renders and closes it.
    """
    present = (
        sorted(set(class_frame["group"].astype(str)))
        if len(class_frame) and "group" in class_frame.columns
        else []
    )
    readouts = []
    for feature, record in zip(READOUTS, records):
        samples, _ = window_samples(clock, class_frame, feature.column)
        # Aligned with the record's own window list by construction rather than by agreement, so a
        # window the test skipped cannot shift the cells drawn under the windows after it.
        order = [int(row["time_bin"]) for row in record.get("per_window") or []]
        readouts.append((feature.column, [samples.get(key, {}) for key in order], record))

    figure = figures.windowed_comparison_figure(
        readouts,
        groups=cohort.ordered_groups(present, labels.CLASS_COLUMN),
        bin_width=TRAJECTORY_BIN_HOURS,
        # The same floor the test excludes a cell at, passed rather than defaulted, so the page and
        # the p-values beneath it agree about which cells carry evidence.
        min_body_size=shared_stats.MIN_GROUP_SIZE,
        xlabel=clock.axis_label,
        ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        delivery_orientation=clock.inverted,
    )
    figures.caveat_note(figure)
    return figure


#: Statistics whose panel carries a second number annotated on each point, as
#: ``{drawn: annotated}``.
#: One entry, and it is the one that makes a positional claim readable: a peak trajectory with no
#: statement of how often that peak was degenerate is a line a reader will take at face value.
PANEL_ANNOTATIONS: Dict[str, str] = {"peak": "peak_degenerate"}


def build_features_figure(clock: Clock, rows: Sequence[Dict[str, Any]]) -> Any:
    r"""Draw the untested statistics against one clock, one panel each.

    **Its own page rather than more rows on the profile one**, and the reason is units. Half of
    what is drawn here is not in seconds -- an entropy in nats, three shares between zero and one,
    a dimensionless skewness -- and a panel that shares a figure with a quantity in different units
    is a panel that will be read against it. The profile page stays what it was: the share of the
    attribution by lag and window, and the two tested centroid trajectories beneath it.

    **The two profiles share each panel, solid for the attribution and dashed for the attention.**
    That is the pairing the whole analysis rests on: the attribution is $K_t$ times the attention
    and inherits the prior-variance inflation the attention is immune to, so a statistic that moves
    in one and not the other is a finding about which of the two is being read -- and it is only
    visible when the two are on one axis. The ribbon is the attribution's inter-quartile range
    alone; a second ribbon would cover the lines it sits behind.

    **Nothing on this page carries a $p$-value**, and that is not an omission. These statistics are
    tabled and drawn rather than tested, which is what keeps each clock's Holm family at two; a
    trajectory here that looks separated is a hypothesis, and ``lag_clocks_significance.csv``
    carries the only claims this analysis makes.

    Args:
        clock: The clock being drawn.
        rows: This clock's trajectory rows -- every feature and every cohort axis; the class axis
            is selected per panel.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(max(len(DRAWN_STATISTICS), 1), height_per_row=3.0)
    for index, statistic in enumerate(DRAWN_STATISTICS):
        _draw_feature_panel(
            axes[index, 0],
            clock,
            rows,
            statistic,
            annotate=PANEL_ANNOTATIONS.get(statistic.key),
            title=f"{statistic.key} against the clock, by {labels.CLASS_COLUMN}",
        )
    figures.caveat_note(figure)
    return figure


def _draw_feature_panel(
    ax: Any,
    clock: Clock,
    rows: Sequence[Dict[str, Any]],
    statistic: Statistic,
    *,
    annotate: Optional[str],
    title: str,
) -> int:
    """Draw one statistic's class trajectories, both profiles on one axis.

    Args:
        ax: Target axes.
        clock: The clock, for the orientation and the axis label.
        rows: This clock's trajectory rows, every feature and every cohort axis.
        statistic: The statistic to draw.
        annotate: Key of a second statistic whose per-window median is written above each point of
            the primary profile's line, or ``None``. Used for the peak, whose position means
            nothing in a window where most segments were degenerate.
        title: Panel title.

    Returns:
        The number of cohorts drawn. Zero draws the empty note instead.
    """
    selected = [
        row for row in rows
        if row["group_column"] == labels.CLASS_COLUMN and row["clock"] == clock.name
    ]
    primary, companion = PROFILE_SOURCES[0], PROFILE_SOURCES[1]
    primary_rows = [
        row for row in selected if row["metric"] == _feature_column(primary.key, statistic)
    ]
    companion_rows = [
        row for row in selected if row["metric"] == _feature_column(companion.key, statistic)
    ]
    annotated_rows = (
        [
            row for row in selected
            if row["metric"] == _feature_column(primary.key, _statistic_by_key(annotate))
        ]
        if annotate
        else []
    )
    groups = cohort.ordered_groups(
        [row["group"] for row in primary_rows], labels.CLASS_COLUMN
    ) if primary_rows else []
    if not groups:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return 0

    colours = figures.group_colors(groups)
    # The dashed convention is named once rather than per cohort: three more legend entries saying
    # the same thing would cover the lines the panel exists to show.
    companion_labelled = False
    for group in groups:
        cell = sorted(
            (row for row in primary_rows if row["group"] == group),
            key=lambda row: row["bin_center_h"],
        )
        if not cell:
            continue
        x = np.array([row["bin_center_h"] for row in cell], dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x,
            np.array([row["q25"] for row in cell], dtype=np.float64),
            np.array([row["q75"] for row in cell], dtype=np.float64),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, np.array([row["median"] for row in cell], dtype=np.float64),
            marker="o", markersize=3, color=colour, linewidth=figures.LINE_EMPHASIS,
            label=f"{group} {primary.key} (n={int(sum(row['n_recordings'] for row in cell))})",
        )
        companion_cell = sorted(
            (row for row in companion_rows if row["group"] == group),
            key=lambda row: row["bin_center_h"],
        )
        if companion_cell:
            ax.plot(
                np.array([row["bin_center_h"] for row in companion_cell], dtype=np.float64),
                np.array([row["median"] for row in companion_cell], dtype=np.float64),
                linestyle="--", color=colour, linewidth=figures.LINE_THIN,
                label=f"{companion.key} (dashed)" if not companion_labelled else "_nolegend_",
            )
            companion_labelled = True
        if annotated_rows:
            # Keyed on the window rather than zipped, so a window one statistic could not be
            # computed in cannot shift every annotation after it onto the wrong point.
            by_window = {
                int(row["time_bin"]): float(row["median"])
                for row in annotated_rows if row["group"] == group
            }
            for row in cell:
                value = by_window.get(int(row["time_bin"]))
                if value is None or not np.isfinite(value):
                    continue
                ax.annotate(
                    f"{value:.0%}", (float(row["bin_center_h"]), float(row["median"])),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    fontsize=figures.FONT_TINY, color=colour,
                )
    ax.set_title(title)
    ax.set_xlabel(clock.axis_label)
    ax.set_ylabel(statistic.unit or figures.COEFFICIENT_LAG_AXIS_LABEL)
    if clock.inverted:
        # Delivery sits at the right, so the eye reads left to right toward it.
        ax.invert_xaxis()
    else:
        ax.axvline(
            0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_REGULAR, zorder=0
        )
    ax.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
    figures.style_axes(ax)
    return len(groups)


def _statistic_by_key(key: str) -> Statistic:
    """Return the :class:`Statistic` registered under ``key``.

    Args:
        key: One of :data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS`.

    Returns:
        Its registration.

    Raises:
        KeyError: If nothing is registered under that key, which is a typo in
            :data:`PANEL_ANNOTATIONS` rather than a condition a run can reach.
    """
    for statistic in STATISTICS:
        if statistic.key == key:
            return statistic
    raise KeyError(f"no statistic is registered under {key!r}; registered: {STATISTIC_KEYS}")


# =================================================================================================
# The analysis
# =================================================================================================
def _skip(reason: str, n_segments: int) -> Dict[str, Any]:
    """Return the recorded skip, and log it.

    Args:
        reason: Why there is nothing to draw.
        n_segments: How many segments the table held.

    Returns:
        The protocol's keys with ``n_samples`` ``None`` -- this analysis scored no population, and
        a zero would enter the coverage block as a disagreement with every analysis that did.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": False},
        "skipped": True,
        "reason": reason,
        "n_segments": int(n_segments),
        "files": [],
    }


def run_lag_clocks_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the lag structure against both clinical clocks and test the class trajectories.

    Args:
        context: The analysis context, read for the per-sample table, the per-sample vector
            sidecar and the pass's lag block. No model is touched, so this runs against a finished
            run directory with no checkpoint and no GPU.
        eval_config: The validated block. Unused: neither the window width nor the significance
            level is configurable, for the reason stated on each constant.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused: the population is the set of segments carrying a
            finite clock coordinate and a lag profile, which only the tables know.

    Returns:
        The protocol's keys plus, per clock, the population it was drawn over and the significance
        of both tested readouts; the lag axis every number is stated on and the caveat that goes
        with it; and the paths written. A recorded skip -- naming its cause -- when the table is
        empty, when the vector sidecar carries no profile, or when neither clock places a segment.
    """
    collection = context.collection
    per_sample = collection.per_sample
    # The run's horizon, applied before anything is binned: it bounds the
    # population on the segment's own start, so every clock in the run answers
    # for the same segments. ``None`` leaves the frame untouched.
    per_sample = cohort.within_horizon(
        per_sample, eval_config.get("max_hours_before_delivery")
    )
    if per_sample.empty:
        return _skip("the collected per-sample table was empty", 0)

    lag = dict(dict(getattr(collection, "results", None) or {}).get("lag") or {})
    delay_steps = int(lag.get("delay_steps") or 0)
    n_lags = int(lag.get("n_lags") or 0)
    if n_lags <= 0:
        return _skip(
            "the collection record carries no lag geometry, so there is no axis to resolve a "
            "profile against",
            len(per_sample),
        )
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    vectors = dict(getattr(collection, "vectors", None) or {})

    featured, sources = add_feature_columns(per_sample, vectors, seconds)
    if not any(int(record.get("n_usable") or 0) for record in sources.values()):
        return _skip(
            "no segment carried a usable lag profile, so neither clock has anything to resolve "
            f"(sources: {sorted(sources)})",
            len(per_sample),
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    clocks: List[Dict[str, Any]] = []
    significance: List[Dict[str, Any]] = []
    per_recording_tables: List[pd.DataFrame] = []
    trajectory: List[Dict[str, Any]] = []
    profiles: List[pd.DataFrame] = []
    written: List[str] = []

    for clock in CLOCKS:
        binned, population = clock_rows(clock, featured)
        if binned.empty:
            population.update(
                {"drawn": False, "reason": f"no segment could be placed on the {clock.name} clock"}
            )
            clocks.append(population)
            continue

        frames = per_recording_frames(clock, binned)
        for axis, frame in frames.items():
            if len(frame):
                per_recording_tables.append(frame.assign(clock=clock.name, group_column=axis))
        rows = trajectory_rows(clock, frames)
        trajectory.extend(rows)

        class_frame = frames.get(labels.CLASS_COLUMN, pd.DataFrame())
        records = [
            analyse_windows(clock, class_frame, feature.column) for feature in READOUTS
        ]
        significance.extend(records)

        # Both profiles reach the CSV; the first is the one the heatmap draws, and it is the
        # attribution rather than the attention because that is the readout this architecture
        # exists to produce. The other profile's shape is on the table beside it.
        cells = {
            source.key: window_profiles(clock, binned, vectors.get(source.attribute), n_lags)
            for source in PROFILE_SOURCES
        }
        for source in PROFILE_SOURCES:
            windows, centres, fields = cells[source.key]
            profiles.append(profile_frame(clock, source, windows, centres, fields, seconds))
        drawn_windows, drawn_centres, drawn_fields = cells[PROFILE_SOURCES[0].key]

        written.append(
            str(
                figures.render_figure(
                    build_profile_figure(
                        clock, drawn_windows, drawn_centres, drawn_fields, rows, seconds
                    ),
                    directory / clock.figure,
                ).name
            )
        )
        written.append(
            str(
                figures.render_figure(
                    build_windows_figure(clock, class_frame, records),
                    directory / clock.windows_figure,
                ).name
            )
        )
        written.append(
            str(
                figures.render_figure(
                    build_features_figure(clock, rows),
                    directory / clock.features_figure,
                ).name
            )
        )
        population.update(
            {
                "drawn": True,
                "n_recordings": population.get(
                    "n_eligible",
                    int(class_frame["guid"].nunique()) if len(class_frame) else 0,
                ),
                "n_significant_windows": {
                    record["metric_column"]: record.get("n_significant_windows", 0)
                    for record in records
                },
            }
        )
        clocks.append(population)

    if not any(record.get("drawn") for record in clocks):
        return _skip(
            "neither clock placed a segment: the table carries no finite 'epoch' and no recording "
            f"with a '{cohort.SECOND_STAGE_COLUMN}' on every segment",
            len(per_sample),
        )

    # Written even when empty, with the key columns present: a table a reader can open and find
    # nothing in is a different statement from a table that was never written.
    tall = (
        pd.concat(per_recording_tables, ignore_index=True)
        if per_recording_tables
        else pd.DataFrame(columns=["clock", "group_column", "group", "guid", *FEATURE_COLUMNS])
    )
    tall.to_csv(directory / PER_RECORDING_FILENAME, index=False)
    pd.DataFrame(
        trajectory,
        columns=[
            "clock", "group_column", "metric", "group", "time_bin", "bin_center_h",
            "n_recordings", "mean", "q25", "median", "q75",
        ],
    ).to_csv(directory / TRAJECTORY_FILENAME, index=False)
    (
        pd.concat(profiles, ignore_index=True) if profiles else profile_frame(
            CLOCKS[0], PROFILE_SOURCES[0], [], [], [], seconds
        )
    ).to_csv(directory / PROFILE_FILENAME, index=False)
    significance_frame(significance).to_csv(directory / SIGNIFICANCE_FILENAME, index=False)
    pairwise_frame(significance).to_csv(directory / PAIRWISE_FILENAME, index=False)

    drawn = [record for record in clocks if record.get("drawn")]
    logger.info(
        f"{ANALYSIS_DIRNAME}: {len(drawn)} of {len(CLOCKS)} clock(s) drawn over "
        f"{sum(int(record.get('n_windows') or 0) for record in drawn)} window(s) of "
        f"{TRAJECTORY_BIN_HOURS:g} h; "
        f"{sum(record.get('n_significant_windows', 0) for record in significance)} significant "
        f"window(s) across {len(significance)} Holm family(ies)"
    )
    return {
        "n_samples": scored_sample_count(featured, READOUTS[0].column),
        "composition": {
            clock["clock"]: {
                "n_segments": clock.get("n_segments", 0),
                "n_recordings": clock.get("n_recordings", 0),
                "n_windows": clock.get("n_windows", 0),
            }
            for clock in clocks
        },
        # The second clock scores the recordings that carry an onset only, which is a subset of the
        # evaluated cohort -- so this analysis stays out of the population comparison rather than
        # entering it as a disagreement about who was evaluated.
        "plan": {
            "capped": True,
            "reason": (
                "the second-stage half is scored over the recordings that carry an onset only, "
                "which is a subset of the evaluated cohort"
            ),
        },
        "bin_width_hours": float(TRAJECTORY_BIN_HOURS),
        "delay_steps": delay_steps,
        "n_lags": n_lags,
        # The sentence every lag-resolved artifact carries. In the record as well as under the
        # figures, because ``summary.json`` is the artifact that gets quoted and a reader of it
        # would otherwise have the lag numbers and no statement of what they are lags *in*.
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "peak_reference": PEAK_REFERENCE,
        # Measured rather than assumed: what preflight computed from this run's own geometry.
        "lag_support": read_lag_support(output_dir),
        "profiles": sources,
        # The constants three of the statistics are defined against. In the record rather than only
        # in the module, because a share of the mass "near the anchor" is meaningless without the
        # number that decided what near is, and ``summary.json`` is the artifact that gets quoted.
        "statistic_thresholds": {
            "near_seconds": float(NEAR_SECONDS),
            "far_seconds": float(FAR_SECONDS),
            "near_far_measured_from": (
                "the shortest lag the axis carries, not zero, so the two shares mean the same "
                "thing at any causal input delay"
            ),
            "peak_fraction": float(PEAK_FRACTION),
            "degenerate_peak_to_median": float(DEGENERATE_PEAK_TO_MEDIAN),
            "degenerate_zero_fraction": float(DEGENERATE_ZERO_FRACTION),
        },
        "features": [
            {
                "column": feature.column,
                "profile": feature.source,
                "statistic": feature.statistic,
                "tested": feature.tested,
                "meaning": feature.meaning,
            }
            for feature in FEATURES
        ],
        "clocks": clocks,
        "significance": [
            {
                key: value
                for key, value in record.items()
                # The per-window and pairwise detail is on the two CSVs; what belongs in the
                # summary is the headline of each family.
                if key not in ("per_window", "pairwise")
            }
            for record in significance
        ],
        # No grouped variants are declared: this analysis is already cut by cohort, so fanning the
        # by-class and by-subgroup emitter over its frame would resolve a cut by a cut.
        "files": [
            PER_RECORDING_FILENAME, TRAJECTORY_FILENAME, PROFILE_FILENAME,
            SIGNIFICANCE_FILENAME, PAIRWISE_FILENAME, *written,
        ],
    }
