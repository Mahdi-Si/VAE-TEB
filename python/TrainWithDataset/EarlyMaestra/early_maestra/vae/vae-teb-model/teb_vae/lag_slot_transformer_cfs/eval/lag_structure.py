r"""The lag structure of this architecture, reduced from the sidecars the collection pass writes.

The lag-attentive cells resolve their lag axis through an attention distribution and a per-lag
allocation of the divergence. This architecture has neither; what it has is two per-lag readouts
of one fitted parameterisation, written by the collection pass per anchor and per segment:

* the **proposal norm** $\lVert r^\mu_{t,\ell} \rVert_2$, what the head emitted at lag $\ell$;
* the **signed divergence drop** $K_t - K_t^{\setminus \ell}$, what removing that lag alone does
  to the divergence.

This module is the one place the shape of those profiles is reduced -- centroid, spread,
quantiles, concentration, the guarded peak, the band masses -- and the one place they are cut on
the two clinical clocks. Three analyses read it (the pooled profile, the clock trajectories and
the high-divergence selection), and an analysis may not import another, so the arithmetic they
share lives one layer down here rather than sideways in any of them.

**Every reduction here is of a readout that is not an allocation.** The reallocation the
suppression qualification names -- $r_\ell \mapsto r_\ell + k_\ell(h_t)$ with
$\sum_\ell k_\ell \equiv 0$ -- leaves the update, the divergence and every prediction unchanged
while changing both profiles at every lag. So a centroid here says where the fitted head's
proposals sit, and a band mass says how much of them sit in a band; neither says the source at
that lag was necessary. The qualification travels on every table and figure built from this
module, and the shape statistics are the family's own
(:mod:`teb_vae.lag_attn_cfs.eval.lag_shape`) so a centroid means the same thing on both cells --
computed of a different profile, under a different reading.

**The signed profile is rectified before it is shaped.** The shape statistics are functions of
$p_\ell = w_\ell / \sum_k w_k$ and refuse a negative bin; the divergence drop can be negative at a
lag whose proposal was cancelling another's. So its shape is taken of $\max(\cdot, 0)$, and what
was discarded travels beside it per segment as ``negative_drop`` and ``net_drop`` rather than
being dropped -- exactly as the family rectifies its clock-excess profile.

The lag axis throughout is **stored-coefficient time**: seconds per stored step times the lag,
offset by the model's own input delay, and nothing else. Lag identities use only the feature
grid and the filter-delay terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval import lag_shape
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.frames import per_recording_means
from teb_vae.lag_slot_transformer_cfs.eval.figures import BAND_COLORS
from teb_vae.lag_slot_transformer_cfs.nets.controls import SUPPRESSION_QUALIFICATION

# =============================================================================
# The two profiles, and the axis they are read on
# =============================================================================
@dataclass(frozen=True)
class ProfileSource:
    """One per-lag readout of this architecture, as the sidecars carry it.

    Attributes:
        key: The suffix every statistic of this profile carries in a table column.
        vector: The per-sample vector in ``per_sample_vectors.npz``.
        anchor_map: The per-anchor map in ``per_anchor_vectors.npz``.
        label: What the readout is, for a panel title.
        unit: The unit of one bin.
        signed: Whether a bin may be negative, in which case the shape is taken of the rectified
            profile and the discarded negative mass is reported beside it.
    """

    key: str
    vector: str
    anchor_map: str
    label: str
    unit: str
    signed: bool


#: The two readouts, in the order every table carries them. The proposal norm is what the head
#: emitted; the divergence drop is what removing the lag does to the divergence. Both are read
#: under :data:`~teb_vae.lag_slot_transformer_cfs.nets.controls.SUPPRESSION_QUALIFICATION`.
PROFILE_SOURCES: Tuple[ProfileSource, ...] = (
    ProfileSource(
        key="proposal",
        vector="proposal_lag_profile",
        anchor_map="proposal_lag_map",
        label="proposal norm",
        unit="latent units",
        signed=False,
    ),
    ProfileSource(
        key="drop",
        vector="divergence_drop_lag_profile",
        anchor_map="divergence_drop_lag_map",
        label="divergence drop, lag removed alone",
        unit="nats per anchor",
        signed=True,
    ),
)

#: The shape statistics carried per profile, and the column suffix each takes. The family's
#: keys, renamed only where the family's suffix names a unit this profile is not in: the
#: proposal norm is in latent units, so its total is ``total`` rather than ``total_nats``.
STATISTIC_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("centroid", "centroid_s"),
    ("spread", "spread_s"),
    ("median", "median_s"),
    ("iqr", "iqr_s"),
    ("effective_support", "effective_support_s"),
    ("peak", "peak_s"),
    ("peak_width", "peak_width_s"),
    ("entropy", "entropy_nats"),
    ("skewness", "skewness"),
    ("near_mass", "near_mass"),
    ("far_mass", "far_mass"),
    ("peak_mass", "peak_mass"),
    ("peak_degenerate", "peak_degenerate"),
    ("zero_fraction", "zero_fraction"),
    ("total_nats", "total"),
    ("peak_nats", "peak_value"),
)

#: The identity columns a segment carries into every table here, so a row can be joined back to
#: the family's per-sample table and placed on either clock.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "sample_index",
    "guid",
    "epoch",
    labels.CLASS_COLUMN,
    labels.SUBGROUP_COLUMN,
    "time_from_labor_onset",
    cohort.SECOND_STAGE_COLUMN,
)


def statistic_column(statistic: str, source: ProfileSource) -> str:
    """The column one statistic of one profile is carried under.

    Args:
        statistic: A key of :data:`~teb_vae.lag_attn_cfs.eval.lag_shape.STATISTIC_KEYS`.
        source: The profile.

    Returns:
        ``<suffix>_<key>``, e.g. ``centroid_s_proposal``.
    """
    suffix = dict(STATISTIC_COLUMNS)[statistic]
    return f"{suffix}_{source.key}"


def band_column(band: str, source: ProfileSource) -> str:
    """The column one declared band's mass of one profile is carried under.

    Args:
        band: The band's declared name.
        source: The profile.

    Returns:
        ``band_<band>_<key>``.
    """
    return f"band_{band}_{source.key}"


@dataclass(frozen=True)
class LagAxis:
    """The lag axis of a run, read from its own results block.

    Attributes:
        n_lags: Candidate lags $L$.
        delay_steps: The model's input delay in stored steps.
        seconds_per_step: Seconds per stored step.
        seconds: The axis in stored-coefficient seconds, $(L,)$.
    """

    n_lags: int
    delay_steps: int
    seconds_per_step: float
    seconds: np.ndarray


def lag_axis_of(results: Mapping[str, Any]) -> Optional[LagAxis]:
    """Read the lag axis off the results block, or ``None`` on a run that read no lag.

    Args:
        results: The results block of the summary.

    Returns:
        The axis, or ``None`` when the block carries no lag count.
    """
    axis = (results.get("lag_readouts") or {}).get("lag_axis") or {}
    n_lags = axis.get("n_lags")
    if not n_lags:
        return None
    delay = int(axis.get("delay_steps", 0) or 0)
    step = float(axis.get("seconds_per_step", SECONDS_PER_STEP))
    seconds = step * (np.arange(int(n_lags), dtype=np.float64) + delay)
    return LagAxis(int(n_lags), delay, step, seconds)


def declared_bands(results: Mapping[str, Any]) -> Dict[str, Tuple[int, int]]:
    """The declared lag bands with their inclusive edges, in declaration order.

    Args:
        results: The results block of the summary.

    Returns:
        ``{band: (lo, hi)}`` without the two reference identities.
    """
    edges = (results.get("lag_readouts") or {}).get("band_edges") or {}
    return {
        str(name): (int(span[0]), int(span[1]))
        for name, span in edges.items()
        if name not in ("none", "all") and isinstance(span, (list, tuple)) and len(span) == 2
    }


def profile_matrix(collection: Any, source: ProfileSource) -> Optional[np.ndarray]:
    """One profile's per-sample matrix, aligned with the per-sample table's rows.

    Args:
        collection: The collection, read for its per-sample vectors.
        source: Which profile.

    Returns:
        The $(n, L)$ float64 matrix, or ``None`` when the sidecar does not carry it -- an arm that
        sums no per-lag updates, or a directory collected before the sidecar existed.
    """
    vectors = dict(getattr(collection, "vectors", None) or {})
    values = vectors.get(source.vector)
    if values is None:
        return None
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(collection.per_sample):
        return None
    return matrix


# =============================================================================
# The shape of a profile, per row
# =============================================================================
def shape_columns(
    matrix: np.ndarray, seconds: np.ndarray, source: ProfileSource
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Every shape statistic of one profile, one value per row, under this module's columns.

    A signed profile is rectified first and the discarded mass reported per row, because the
    family's reducer refuses a negative bin and a negative bin is a measurement here rather than
    a defect.

    Args:
        matrix: The profiles, $(n, L)$.
        seconds: The lag axis in seconds, $(L,)$.
        source: Which profile, for the column suffix and the signed rule.

    Returns:
        ``(columns, census)`` -- the per-row arrays keyed by column name, and the reducer's
        census of how many rows carried usable mass.
    """
    values = np.asarray(matrix, dtype=np.float64)
    columns: Dict[str, np.ndarray] = {}
    if source.signed:
        finite = np.where(np.isfinite(values), values, 0.0)
        positive = np.maximum(finite, 0.0)
        negative = np.minimum(finite, 0.0)
        # A row with no finite bin at all carries NaN rather than a zero net: nothing was read.
        any_finite = np.isfinite(values).any(axis=1)
        columns[f"net_{source.key}"] = np.where(any_finite, finite.sum(axis=1), np.nan)
        columns[f"positive_{source.key}"] = np.where(any_finite, positive.sum(axis=1), np.nan)
        columns[f"negative_{source.key}"] = np.where(any_finite, negative.sum(axis=1), np.nan)
        shaped = np.where(np.isfinite(values), positive, np.nan)
    else:
        shaped = values
    statistics, census = lag_shape.profile_statistics(shaped, seconds)
    for statistic, _suffix in STATISTIC_COLUMNS:
        columns[statistic_column(statistic, source)] = np.asarray(
            statistics[statistic], dtype=np.float64
        )
    return columns, census


def band_mass_columns(
    matrix: np.ndarray, bands: Mapping[str, Tuple[int, int]], source: ProfileSource
) -> Dict[str, np.ndarray]:
    """The mass of one profile inside each declared band, per row.

    The signed profile is summed **signed** here, not rectified: a band's net drop is the
    quantity a reader compares with the band's suppression margin, and rectifying it would
    overstate every band by its cancelled part.

    Args:
        matrix: The profiles, $(n, L)$.
        bands: ``{band: (lo, hi)}``, inclusive.
        source: Which profile, for the column name.

    Returns:
        ``{band_<band>_<key>: (n,)}``; NaN where a row carries no finite bin in the band.
    """
    values = np.asarray(matrix, dtype=np.float64)
    columns: Dict[str, np.ndarray] = {}
    for band, (lo, hi) in bands.items():
        window = values[:, lo : hi + 1] if values.shape[1] > lo else values[:, :0]
        any_finite = np.isfinite(window).any(axis=1) if window.shape[1] else np.zeros(
            values.shape[0], dtype=bool
        )
        columns[band_column(band, source)] = np.where(
            any_finite, np.nansum(np.where(np.isfinite(window), window, 0.0), axis=1), np.nan
        )
    return columns


def segment_table(
    collection: Any, results: Mapping[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One row per segment: its identity, the shape of both profiles, and their band masses.

    The table every lag-structure analysis starts from. Built from the per-sample vector sidecar
    and the per-sample table alone, so an offline re-run against a finished directory works
    with no checkpoint.

    Args:
        collection: The collection, read for its per-sample table and vectors.
        results: The results block, for the lag axis and the declared bands.

    Returns:
        ``(table, record)`` -- the table, and a record naming which profiles were present, the
        reducer's census per profile, and why the table is empty when it is.
    """
    per_sample = collection.per_sample
    axis = lag_axis_of(results)
    record: Dict[str, Any] = {"sources": {}, "n_lags": None if axis is None else axis.n_lags}
    identity = pd.DataFrame(
        {name: per_sample[name] for name in IDENTITY_COLUMNS if name in per_sample.columns}
    )
    if axis is None:
        record["reason"] = "the run reported no lag axis: this arm has no source pathway"
        return identity.iloc[:0], record
    bands = declared_bands(results)
    frames: List[pd.DataFrame] = [identity.reset_index(drop=True)]
    for source in PROFILE_SOURCES:
        matrix = profile_matrix(collection, source)
        if matrix is None:
            record["sources"][source.key] = {"present": False}
            continue
        columns, census = shape_columns(matrix, axis.seconds, source)
        columns.update(band_mass_columns(matrix, bands, source))
        record["sources"][source.key] = {"present": True, **census}
        frames.append(pd.DataFrame(columns))
    if len(frames) == 1:
        record["reason"] = (
            "the per-sample vector sidecar carries no lag profile: this arm sums no per-lag "
            "updates, or the directory was collected before the sidecar existed"
        )
        return identity.iloc[:0], record
    return pd.concat(frames, axis=1), record


def present_sources(record: Mapping[str, Any]) -> List[ProfileSource]:
    """The profiles a segment-table record says were present, in declaration order.

    Args:
        record: The record :func:`segment_table` returned.

    Returns:
        The sources.
    """
    sources = record.get("sources") or {}
    return [source for source in PROFILE_SOURCES if (sources.get(source.key) or {}).get("present")]


# =============================================================================
# Pooled and stratified profiles
# =============================================================================
def normalised_rows(matrix: np.ndarray) -> np.ndarray:
    """Each row as a share of its own finite, rectified mass; NaN rows where there is none.

    Args:
        matrix: The profiles, $(n, L)$.

    Returns:
        The shares, $(n, L)$.
    """
    values = np.asarray(matrix, dtype=np.float64)
    positive = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = positive.sum(axis=1)
    shares = np.divide(
        positive, total[:, None], out=np.full_like(positive, np.nan), where=total[:, None] > 0.0
    )
    return shares


def pooled_profile(matrix: np.ndarray, guids: Sequence[str]) -> Dict[str, Any]:
    """The profile pooled over recordings: each recording's segments averaged first.

    Recordings rather than segments carry equal weight, for the reason every interval in this
    pipeline is taken over recordings: a recording contributing eleven segments must not outvote
    one contributing two.

    Args:
        matrix: The per-segment profiles, $(n, L)$.
        guids: One recording identifier per row.

    Returns:
        ``mean``, ``q25``, ``median``, ``q75`` over recordings, each $(L,)$, and
        ``n_recordings``.
    """
    frame = pd.DataFrame(np.asarray(matrix, dtype=np.float64))
    frame["guid"] = [str(guid) for guid in guids]
    per_recording = frame.groupby("guid", sort=True).mean().to_numpy()
    if per_recording.shape[0] == 0:
        width = int(np.asarray(matrix).shape[1]) if np.asarray(matrix).ndim == 2 else 0
        blank = np.full(width, np.nan)
        return {"mean": blank, "q25": blank, "median": blank, "q75": blank, "n_recordings": 0}
    return {
        "mean": np.nanmean(per_recording, axis=0),
        "q25": np.nanpercentile(per_recording, 25, axis=0),
        "median": np.nanpercentile(per_recording, 50, axis=0),
        "q75": np.nanpercentile(per_recording, 75, axis=0),
        "n_recordings": int(per_recording.shape[0]),
    }


def peak_record(profile: np.ndarray, seconds: np.ndarray) -> Dict[str, Any]:
    """Locate and describe the peak of one pooled profile, with the family's degeneracy guard.

    Args:
        profile: The pooled profile, $(L,)$, rectified.
        seconds: The lag axis in seconds.

    Returns:
        The argmax lag and its seconds, the peak's share of the mass, its width at half height,
        the secondary-peak count and the degeneracy verdict -- NaN throughout on an empty
        profile.
    """
    values = np.asarray(profile, dtype=np.float64)
    finite = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(finite.sum())
    if values.size == 0 or total <= 0.0:
        return {
            "argmax_lag_step": None, "argmax_seconds": None, "peak_share": None,
            "peak_width_s": None, "n_secondary_peaks": None, "degenerate": None,
            "degeneracy_reason": "no mass",
        }
    argmax = int(np.argmax(finite))
    width = lag_shape.peak_width(finite.tolist())
    degeneracy = lag_shape.degeneracy(finite.tolist())
    secondary = lag_shape.secondary_peaks(finite.tolist())
    step = float(seconds[1] - seconds[0]) if seconds.size > 1 else 0.0
    width_bins = width.get("width_bins")
    return {
        "argmax_lag_step": argmax,
        "argmax_seconds": float(seconds[argmax]),
        "peak_share": float(finite[argmax] / total),
        "peak_width_s": None if width_bins is None else float(width_bins) * step,
        "n_secondary_peaks": int(len(secondary)),
        "degenerate": bool(degeneracy.get("degenerate", False)),
        "degeneracy_reason": "; ".join(str(reason) for reason in degeneracy.get("reasons") or []),
    }


# =============================================================================
# The two clinical clocks
# =============================================================================
@dataclass(frozen=True)
class Clock:
    """One clinical clock: how a segment table is binned on it and how its axis is drawn.

    Attributes:
        name: The clock's name, used in filenames and rows.
        bin: The binner from the cohort layer.
        bin_column: The window index column the binner adds.
        center_column: The window centre column the binner adds.
        axis_label: The x label of a figure on this clock.
        inverted: Whether delivery sits at the right, so the axis is drawn inverted.
        eligible_only: Whether only recordings the second-stage eligibility rule admits are
            placed on it.
    """

    name: str
    bin: Callable[..., pd.DataFrame]
    bin_column: str
    center_column: str
    axis_label: str
    inverted: bool
    eligible_only: bool


#: The two clocks, on the family's own grid of :data:`~teb_vae.lag_attn_cfs.eval.cohort.TRAJECTORY_BIN_HOURS`.
CLOCKS: Tuple[Clock, ...] = (
    Clock(
        name="time_to_delivery",
        bin=cohort.add_time_bins,
        bin_column=cohort.BIN_COLUMN,
        center_column=cohort.BIN_CENTER_COLUMN,
        axis_label="Time before delivery (hours)",
        inverted=True,
        eligible_only=False,
    ),
    Clock(
        name="second_stage",
        bin=cohort.add_second_stage_bins,
        bin_column=cohort.SECOND_STAGE_BIN_COLUMN,
        center_column=cohort.SECOND_STAGE_BIN_CENTER_COLUMN,
        axis_label="Time from second-stage onset (hours; negative before onset)",
        inverted=False,
        eligible_only=True,
    ),
)

#: Window width in hours, bound from the cohort layer so both clocks and every clock analysis
#: share one grid.
BIN_HOURS = float(cohort.TRAJECTORY_BIN_HOURS)

#: Family-wise error rate every Holm family here controls. Not configurable, for the reason the
#: bin width is not.
ALPHA = 0.05


def clock_rows(table: pd.DataFrame, clock: Clock) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Bin a segment table on one clock, admitting only the recordings the clock can place.

    Args:
        table: A segment table carrying ``epoch`` and the second-stage offset.
        clock: The clock.

    Returns:
        ``(binned, population)`` -- the binned rows, and how many recordings the clock admitted.
    """
    population: Dict[str, Any] = {"n_recordings": int(table["guid"].nunique()) if len(table) else 0}
    frame = table
    if clock.eligible_only and len(table):
        eligibility = cohort.second_stage_eligibility(table)
        admitted = set(
            eligibility.loc[eligibility["eligible"].astype(bool), "guid"].astype(str)
        )
        frame = table[table["guid"].astype(str).isin(admitted)]
        population["n_eligible"] = len(admitted)
        population["n_ineligible"] = int(population["n_recordings"] - len(admitted))
    binned = clock.bin(frame, width=BIN_HOURS) if len(frame) else clock.bin(frame)
    return binned, population


def per_recording_by_axis(
    binned: pd.DataFrame, columns: Sequence[str], clock: Clock
) -> Dict[str, pd.DataFrame]:
    """One row per (cohort, window, recording) on each cohort axis.

    Args:
        binned: The binned segment rows.
        columns: The value columns to reduce.
        clock: The clock, for its bin columns.

    Returns:
        Cohort axis to its frame. Both axes, because a reader wants both cuts; only the class
        axis is tested.
    """
    return {
        axis: cohort.per_recording_in_bins(
            binned, columns, group_column=axis,
            bin_column=clock.bin_column, center_column=clock.center_column,
        )
        for axis in labels.GROUP_COLUMNS
    }


def trajectory_rows(
    per_recording: Mapping[str, pd.DataFrame], columns: Sequence[str], clock: Clock
) -> List[Dict[str, Any]]:
    """Summarise every (axis, cohort, window) cell of every column, over its recordings.

    Args:
        per_recording: The per-axis frames from :func:`per_recording_by_axis`.
        columns: The value columns.
        clock: The clock, for its bin columns and its name on every row.

    Returns:
        Long-form rows carrying the clock, the axis, the metric, the cohort, the window and the
        quartiles.
    """
    rows: List[Dict[str, Any]] = []
    for axis, frame in per_recording.items():
        for column in columns:
            for row in cohort.trajectory_rows(
                frame, column, metric=column,
                bin_column=clock.bin_column, center_column=clock.center_column,
            ):
                rows.append({"clock": clock.name, "group_column": axis, **row})
    return rows


def window_samples(
    class_frame: pd.DataFrame, column: str, clock: Clock
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, Any]]]:
    """Split a class-axis frame into the per-window, per-class cells a test and a figure share.

    Args:
        class_frame: The class-axis per-recording-per-window frame.
        column: The value column.
        clock: The clock, for its bin columns.

    Returns:
        ``(samples, meta)`` keyed by window in ascending order; every class present is kept,
        unfiltered, so the figure can draw the classes the test excludes.
    """
    samples: Dict[int, Dict[str, np.ndarray]] = {}
    meta: Dict[int, Dict[str, Any]] = {}
    if class_frame.empty or clock.bin_column not in class_frame.columns:
        return samples, meta
    groups = cohort.ordered_groups(
        sorted(set(class_frame["group"].astype(str))), labels.CLASS_COLUMN
    )
    for window in sorted(int(value) for value in class_frame[clock.bin_column].unique()):
        cell = class_frame[class_frame[clock.bin_column] == window]
        values_by_class: Dict[str, np.ndarray] = {}
        for group in groups:
            values = np.asarray(
                cell.loc[cell["group"].astype(str) == group, column], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if values.size:
                values_by_class[group] = values
        samples[window] = values_by_class
        meta[window] = {
            "bin_center_h": float(cell[clock.center_column].iloc[0]),
            "groups_excluded_as_too_small": {
                group: int(values.size) for group, values in values_by_class.items()
                if values.size < shared_stats.MIN_GROUP_SIZE
            },
            "min_group_size": shared_stats.MIN_GROUP_SIZE,
        }
    return samples, meta


def windowed_tests(
    class_frame: pd.DataFrame, column: str, clock: Clock, *, alpha: float = ALPHA
) -> Dict[str, Any]:
    """Test one column across the classes window by window on one clock, Holm across windows.

    Args:
        class_frame: The class-axis per-recording-per-window frame.
        column: The value column.
        clock: The clock.
        alpha: The family-wise error rate.

    Returns:
        The record: ``tested`` with a reason when fewer than two classes are present, else the
        per-window omnibus results with their Holm-adjusted $p$, the pairwise comparisons of the
        survivors and the counts.
    """
    groups = (
        cohort.ordered_groups(sorted(set(class_frame["group"].astype(str))), labels.CLASS_COLUMN)
        if len(class_frame) and "group" in class_frame.columns
        else []
    )
    base = {
        "clock": clock.name,
        "metric_column": column,
        "group_column": labels.CLASS_COLUMN,
        "alpha": float(alpha),
        "bin_width_hours": BIN_HOURS,
        "method": METHOD,
    }
    if len(groups) < 2:
        return {
            **base, "tested": False,
            "reason": f"fewer than two clinical classes present ({groups or 'none'})",
            "per_window": [], "pairwise": {},
        }
    samples, meta = window_samples(class_frame, column, clock)
    testable = {
        window: {
            group: values for group, values in cell.items()
            if values.size >= shared_stats.MIN_GROUP_SIZE
        }
        for window, cell in samples.items()
    }
    outcome = shared_stats.windowed_group_comparisons(testable, meta_by_window=meta, alpha=alpha)
    return {
        **base, "tested": True, "classes": groups,
        "n_windows": outcome["n_windows"],
        "n_windows_tested": outcome["n_windows_tested"],
        "n_significant_windows": outcome["n_significant_windows"],
        "per_window": outcome["per_window"],
        "pairwise": outcome["pairwise"],
    }


#: The method sentence written into every test record, so a $p$-value is readable without this
#: module.
METHOD = (
    "Per window: Kruskal-Wallis across clinical classes over one value per recording, Holm "
    "step-down correction across the windows of one clock and one readout as one family, "
    "pairwise two-sided Mann-Whitney U with Cliff's delta for the windows significant after Holm "
    "only. Every pair is oriented from the more severe class to the less severe one. Classes with "
    f"fewer than {shared_stats.MIN_GROUP_SIZE} recordings in a window are excluded from it and "
    "recorded. Every readout tested here is a readout of one fitted parameterisation and not an "
    "allocation over lags."
)


def significance_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten per-window omnibus results of several test records into one table.

    Args:
        records: The records :func:`windowed_tests` returned.

    Returns:
        One row per (record, window).
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        for window in record.get("per_window") or []:
            rows.append({
                "clock": record.get("clock"),
                "metric_column": record.get("metric_column"),
                "time_bin": window.get("time_bin"),
                "bin_center_h": window.get("bin_center_h"),
                "n_classes": window.get("n_groups"),
                "n_recordings": sum((window.get("n_per_group") or {}).values()),
                "statistic": window.get("statistic"),
                "p_value": window.get("p_value"),
                "p_holm": window.get("p_holm", float("nan")),
                "significant": window.get("significant", False),
                "alpha": window.get("alpha", float("nan")),
            })
    return pd.DataFrame(rows, columns=[
        "clock", "metric_column", "time_bin", "bin_center_h", "n_classes", "n_recordings",
        "statistic", "p_value", "p_holm", "significant", "alpha",
    ])


def pairwise_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten the surviving windows' pairwise comparisons of several records into one table.

    Args:
        records: The records :func:`windowed_tests` returned.

    Returns:
        One row per (record, window, pair).
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        centres = {
            int(window["time_bin"]): window.get("bin_center_h")
            for window in record.get("per_window") or []
        }
        for key, comparisons in (record.get("pairwise") or {}).items():
            for item in comparisons:
                rows.append({
                    "clock": record.get("clock"),
                    "metric_column": record.get("metric_column"),
                    "time_bin": int(key),
                    "bin_center_h": centres.get(int(key), float("nan")),
                    "left": item["left"], "right": item["right"],
                    "n_left": item["n_left"], "n_right": item["n_right"],
                    "p_value": item["p_value"], "cliffs_delta": item["cliffs_delta"],
                    "magnitude": item["magnitude"],
                })
    return pd.DataFrame(rows, columns=[
        "clock", "metric_column", "time_bin", "bin_center_h", "left", "right", "n_left",
        "n_right", "p_value", "cliffs_delta", "magnitude",
    ])


def summary_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    """A test record without its per-window and pairwise detail, for the summary block.

    Args:
        record: The record :func:`windowed_tests` returned.

    Returns:
        The headline keys.
    """
    return {key: value for key, value in record.items() if key not in ("per_window", "pairwise")}


def windowed_shares(
    binned: pd.DataFrame, matrix: np.ndarray, clock: Clock
) -> Tuple[List[int], List[float], Dict[str, Dict[str, Any]]]:
    """The mean normalised profile per (class, window), each recording weighted once.

    Args:
        binned: The binned segment rows, carrying ``sample_index`` to index ``matrix``.
        matrix: The per-sample profiles, $(n, L)$, in per-sample row order.
        clock: The clock, for its bin columns.

    Returns:
        ``(windows, centres, fields)`` -- the window indices and centres in ascending order, and
        per class a ``share`` field $(L, W)$ with the recording count behind it.
    """
    if binned.empty or "sample_index" not in binned.columns:
        return [], [], {}
    shares = normalised_rows(matrix)
    labelled = binned[binned[labels.CLASS_COLUMN].notna()]
    windows = sorted(int(value) for value in labelled[clock.bin_column].unique())
    centres = [
        float(labelled.loc[labelled[clock.bin_column] == window, clock.center_column].iloc[0])
        for window in windows
    ]
    fields: Dict[str, Dict[str, Any]] = {}
    groups = cohort.ordered_groups(
        sorted(set(labelled[labels.CLASS_COLUMN].astype(str))), labels.CLASS_COLUMN
    )
    for group in groups:
        rows = labelled[labelled[labels.CLASS_COLUMN].astype(str) == group]
        field = np.full((shares.shape[1], len(windows)), np.nan)
        for column, window in enumerate(windows):
            cell = rows[rows[clock.bin_column] == window]
            if cell.empty:
                continue
            per_recording = (
                pd.DataFrame(shares[cell["sample_index"].to_numpy(dtype=np.int64)])
                .assign(guid=cell["guid"].astype(str).to_numpy())
                .groupby("guid").mean().to_numpy()
            )
            if per_recording.shape[0]:
                field[:, column] = np.nanmean(per_recording, axis=0)
        fields[group] = {"share": field, "n_recordings": int(rows["guid"].nunique())}
    return windows, centres, fields


# =============================================================================
# Drawing
# =============================================================================
def draw_trajectory_panel(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    clock: Clock,
    title: str,
    ylabel: str,
    zero: bool = False,
) -> int:
    """Draw one metric's per-cohort trajectory: median with inter-quartile ribbon, counts on it.

    Args:
        ax: Target axes.
        rows: The trajectory rows of one metric on one clock and one axis.
        axis: The cohort axis, for the colour and order conventions.
        clock: The clock, for the orientation and the x label.
        title: Panel title.
        ylabel: Y label.
        zero: Whether to draw a reference line at zero.

    Returns:
        The number of cohorts drawn; zero draws the empty note.
    """
    groups = cohort.ordered_groups([row["group"] for row in rows], axis) if rows else []
    ax.set_title(title)
    if not groups:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY, fontstyle="italic",
        )
        figures.style_axes(ax)
        return 0
    colours = figures.group_colors(groups)
    for group in groups:
        cell = sorted(
            (row for row in rows if row["group"] == group), key=lambda row: row["bin_center_h"]
        )
        x = np.array([row["bin_center_h"] for row in cell], dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x, np.array([row["q25"] for row in cell]), np.array([row["q75"] for row in cell]),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, np.array([row["median"] for row in cell]), marker="o",
            markersize=figures.MARKER_SMALL, color=colour, linewidth=figures.LINE_REGULAR,
            label=f"{group} (n={int(cell[0].get('n_recordings_total', 0))})",
        )
        for row in cell:
            ax.annotate(
                str(int(row["n_recordings"])), (float(row["bin_center_h"]), float(row["median"])),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=figures.FONT_TINY, color=colour,
            )
    if zero:
        ax.axhline(0.0, color=figures.COLOR_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
    if clock.inverted:
        ax.invert_xaxis()
    else:
        ax.axvline(0.0, color=figures.COLOR_LIGHT_GRAY, linestyle=":", linewidth=figures.LINE_THIN)
    ax.set_xlabel(clock.axis_label)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=figures.FONT_SMALL, loc="best")
    figures.style_axes(ax)
    return len(groups)


def windows_page(
    class_frame: pd.DataFrame,
    records: Sequence[Tuple[str, Mapping[str, Any], str]],
    clock: Clock,
) -> Any:
    """One clock's tested page: the per-window distributions, their significance, the effects.

    Args:
        class_frame: The class-axis per-recording-per-window frame.
        records: ``(name, record, column)`` per tested readout.
        clock: The clock.

    Returns:
        The figure; the caller renders and closes it.
    """
    present = (
        sorted(set(class_frame["group"].astype(str)))
        if len(class_frame) and "group" in class_frame.columns
        else []
    )
    readouts = []
    for name, record, column in records:
        samples, _meta = window_samples(class_frame, column, clock)
        order = [int(row["time_bin"]) for row in record.get("per_window") or []]
        readouts.append((name, [samples.get(key, {}) for key in order], record))
    return figures.windowed_comparison_figure(
        readouts,
        groups=cohort.ordered_groups(present, labels.CLASS_COLUMN),
        bin_width=BIN_HOURS,
        min_body_size=shared_stats.MIN_GROUP_SIZE,
        xlabel=clock.axis_label,
        ylabel="one value per recording",
        delivery_orientation=clock.inverted,
    )


def shade_bands(ax: Any, bands: Mapping[str, Tuple[int, int]], seconds: np.ndarray) -> None:
    """Shade the declared bands on a lag axis drawn in seconds, naming each along the top.

    Args:
        ax: Target axes whose x axis is the lag axis in seconds.
        bands: ``{band: (lo, hi)}`` in stored steps.
        seconds: The lag axis in seconds.
    """
    if seconds.size == 0:
        return
    half = float(seconds[1] - seconds[0]) / 2.0 if seconds.size > 1 else 0.5
    for index, (name, (lo, hi)) in enumerate(bands.items()):
        if lo >= seconds.size:
            continue
        left = float(seconds[lo]) - half
        right = float(seconds[min(hi, seconds.size - 1)]) + half
        # The declared bands take the same fixed colours in declaration order as on every other
        # lag figure of this cell, so a band is one colour across the whole run.
        colour = BAND_COLORS[index % len(BAND_COLORS)]
        ax.axvspan(left, right, color=colour, alpha=0.08, linewidth=0, zorder=0)
        # Just inside the bottom of the panel rather than the top, where a legend given headroom
        # would sit on the names.
        ax.annotate(
            name, xy=(0.5 * (left + right), 0.0), xycoords=ax.get_xaxis_transform(),
            xytext=(0.0, 2.0), textcoords="offset points", ha="center", va="bottom",
            fontsize=figures.FONT_TINY, color=colour,
        )


def per_recording_table(table: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Reduce a segment table to one row per recording over the named columns.

    Args:
        table: A segment table.
        columns: The value columns; names absent from the table are skipped.

    Returns:
        The family's per-recording frame, indexed by ``guid``, with cohort labels.
    """
    present = [name for name in columns if name in table.columns]
    return per_recording_means(table, present)


#: The qualification every artifact of the lag structure carries, bound from the controls
#: module so the artifact and the design say one thing.
QUALIFICATION = SUPPRESSION_QUALIFICATION

__all__ = [
    "ALPHA",
    "BIN_HOURS",
    "CLOCKS",
    "Clock",
    "IDENTITY_COLUMNS",
    "LagAxis",
    "METHOD",
    "PROFILE_SOURCES",
    "ProfileSource",
    "QUALIFICATION",
    "STATISTIC_COLUMNS",
    "band_column",
    "band_mass_columns",
    "clock_rows",
    "declared_bands",
    "draw_trajectory_panel",
    "lag_axis_of",
    "normalised_rows",
    "pairwise_frame",
    "peak_record",
    "per_recording_by_axis",
    "per_recording_table",
    "pooled_profile",
    "present_sources",
    "profile_matrix",
    "segment_table",
    "shade_bands",
    "shape_columns",
    "significance_frame",
    "statistic_column",
    "summary_of",
    "trajectory_rows",
    "window_samples",
    "windowed_shares",
    "windowed_tests",
    "windows_page",
]
