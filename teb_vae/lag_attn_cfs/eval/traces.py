r"""The per-recording trace: every segment of a recording, in time order, at anchor resolution.

Every other readout in this family reduces a recording to a number, a profile or a window mean.
This module keeps the recording whole: each of its segments is one forward pass, each forward
produces a latent, a divergence and a lag readout at every decoded anchor, and laid end to end on
the absolute time axis those are the recording's **evolution** -- what the model believed, how
far the source moved that belief, and from which lags, hour by hour up to delivery.

Two cells produce traces and both go through this module, because what differs between them is
*which* tensors a forward yields and not what a trace is. A cell hands over one
:class:`SegmentTrace` per segment -- the anchor axis, the scored-anchor indicator, a mapping of
per-anchor scalars and a mapping of per-anchor vectors, already as arrays -- and everything from
there is shared: the time axis, the derived latent scalars, the lag-shape statistics, the
per-segment summary, the arrays on disk and the two figures. Nothing here touches a model or a
loader; the cell's own analysis does the forward and the identity check.

**The two forms.** The *full* form is one row per decoded anchor, and it is what shows structure
inside a segment: a divergence that rises before a contraction, a lag centroid that moves within
twenty minutes. The *summary* form is one row per segment, every scalar averaged over the
segment's scored anchors, and it is what shows the recording across hours. Both carry the
identity a clinical question is asked in -- the class and the subgroup -- so a trace lifted out of
its directory still says whose it is.

**Three rules the reductions obey**, each with the failure it prevents:

* **Only scored anchors enter a mean.** The latent exists at every decoded anchor, including the
  ones the coverage floor rejected and the ones inside a signal-loss gap where the encoder reads a
  padded input; averaging those in produces a state excursion that reads as a physiological event
  and is a gap. The full table keeps every decoded anchor with the indicator beside it, so the
  choice is visible rather than made for the reader.
* **A segment with no scored anchor is ``NaN``, never ``0.0``.** A zeroed latent is the origin of
  the space, which on a centred figure is the population mean -- the most plausible-looking wrong
  answer available.
* **Means, never samples.** The latent stored is $\mu$, not $z$: the samples carry the
  reparameterisation noise, so two passes over one checkpoint would draw two different paths and
  the difference would be $\epsilon$.

**The absolute axis is** $t_{\mathrm{abs}} = \mathrm{epoch} + \Delta\, t$ **seconds**, negative
before delivery, with $\Delta$ the stored step -- the convention the whole-delivery trajectory
already uses, so the two cannot disagree by a constant about where a recording's points are.

**A lag axis here is stored-coefficient time**, and every lag-resolved panel prints the caveat.
The shape statistics of a lag profile are the shared vocabulary of :mod:`lag_shape`, so a centroid
on a trace and a centroid on a clock page are the same arithmetic.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort, lag_shape
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import sanitise_guid
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

#: The ``eval_config.caps`` name bounding how many recordings are traced **per clinical class**.
#: Per class, not in total: the point of the draw is that the two rare classes are traced as often
#: as the common one. A cap on figures and files rather than a retention cap; absent means the
#: default below.
TRACES_CAP = "traces_per_class"

#: Recordings traced per class when the cap says nothing. Ten, because that is the smallest number
#: at which a reader looking at one class can tell a recording from a pattern.
DEFAULT_TRACES_PER_CLASS = 10

#: Segments a recording must hold to be eligible. A recording contributing one segment has no
#: evolution to show, and it is counted rather than drawn as a one-point trace.
MIN_SEGMENTS_PER_TRACE = 2

#: Offset added to the run's seed for the recording draw, distinct from every other draw's so the
#: traced recordings are not the pages' recordings under another name.
TRACE_DRAW_SEED_OFFSET = 6

#: The directory every trace lands in, and the files it holds.
ANALYSIS_DIRNAME = "recording_traces"
MANIFEST_FILENAME = "recording_traces.csv"
SEGMENT_SUMMARY_FILENAME = "segment_summary.csv"
ANCHOR_TRACE_FILENAME = "anchor_trace.parquet"
SUMMARY_FIGURE = "recording_traces_summary"

#: Tails of the per-recording files, after ``<guid>_<subgroup>``. The subgroup travels in the
#: name, because a file named by GUID alone does not say which cohort it came from.
FULL_ARRAYS_SUFFIX = "_full"
TRACE_FIGURE_SUFFIX = "_trace"

#: The class directory a recording with no recoverable clinical class lands in. A word rather
#: than an empty component, so the path stays addressable and says why.
UNLABELLED_CLASS = "unlabelled"

#: The manifest's columns: whose trace a file is, how much it holds and where it is.
MANIFEST_COLUMNS: Tuple[str, ...] = (
    "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "n_segments", "n_segments_collected",
    "n_anchors", "n_contributing", "span_hours", "arrays_file", "figure_file",
)

#: Suffixes the lag-shape statistics are written under, keyed by :data:`lag_shape.STATISTIC_KEYS`.
#: The seconds-valued ones say so; the two scale-carrying ones are named for what they are on
#: *any* profile rather than in nats, because a profile here may be an attention weight or a
#: proposal norm as well as a divergence.
LAG_STATISTIC_SUFFIXES: Mapping[str, str] = {
    "centroid": "centroid_s",
    "spread": "spread_s",
    "median": "median_s",
    "iqr": "iqr_s",
    "effective_support": "effective_support_s",
    "peak": "peak_s",
    "peak_width": "peak_width_s",
    "entropy": "entropy_nats",
    "skewness": "skewness",
    "near_mass": "near_mass",
    "far_mass": "far_mass",
    "peak_mass": "peak_mass",
    "peak_degenerate": "peak_degenerate",
    "zero_fraction": "zero_fraction",
    "total_nats": "total",
    "peak_nats": "peak_value",
}

#: The identity columns every table leads with, in the order a reader wants them.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "epoch", "segment_order",
)

#: Decades below a lag panel's own maximum the logarithmic colour scale is floored at, so one
#: near-zero cell cannot stretch the colormap over decades that hold nothing.
LOG_PANEL_DECADES = 4.0


# =============================================================================
# What a cell hands over
# =============================================================================
@dataclass
class SegmentTrace:
    """One segment's forward, reduced to arrays on the decoded anchor axis.

    Attributes:
        guid: The recording identifier.
        epoch: The segment's start on the absolute axis, in seconds before delivery (negative).
        clinical_class: The recording's class, or ``None`` when the split carries none.
        subgroup: The recording's subgroup, or ``None``.
        anchor: The decoded decimated steps, $(A,)$ -- the forward's own ``anchor_index``, never
            a row position.
        contributing: Whether each anchor was scored, $(A,)$ bool.
        scalars: ``{name: (A,)}`` per-anchor scalars, in the cell's own units.
        vectors: ``{name: (A, K)}`` per-anchor vectors: the latent parameters, the per-coordinate
            divergence, the lag profiles.
        clocks: Per-segment clinical clocks the cell could read (labour onset, second-stage
            onset), ``NaN`` where the recording has none. Carried through to the summary.
    """

    guid: str
    epoch: float
    clinical_class: Optional[str]
    subgroup: Optional[str]
    anchor: np.ndarray
    contributing: np.ndarray
    scalars: Dict[str, np.ndarray] = field(default_factory=dict)
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    clocks: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecordingTrace:
    """One recording's segments in time order, with both forms assembled.

    Attributes:
        guid: The recording identifier.
        clinical_class: Its class, or ``None``.
        subgroup: Its subgroup, or ``None``.
        segments: The segment traces, sorted by ``epoch``.
        anchors: The full form: one row per decoded anchor of every segment.
        summary: The summary form: one row per segment.
        segment_vectors: ``{name: (S, K)}`` each vector averaged over each segment's scored
            anchors, ``NaN`` rows where a segment scored none. What the summary's lag-shape
            statistics were taken of, and what the arrays file carries beside the full vectors.
    """

    guid: str
    clinical_class: Optional[str]
    subgroup: Optional[str]
    segments: List[SegmentTrace]
    anchors: pd.DataFrame
    summary: pd.DataFrame
    segment_vectors: Dict[str, np.ndarray]


# =============================================================================
# Choosing which recordings to trace
# =============================================================================
def select_recordings(
    recordings: pd.DataFrame,
    *,
    per_class: int,
    seed: int,
    min_segments: int = MIN_SEGMENTS_PER_TRACE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""Draw up to ``per_class`` eligible recordings from every clinical class, seeded.

    Equal $N$ per class rather than a proportional quota, for the reason the class-balanced pages
    exist: on the shipped cohort the healthy class holds most of the recordings, and a proportional
    draw would trace the two rare classes once or twice against healthy's dozen. Recordings with
    fewer than ``min_segments`` segments are excluded and counted, never drawn: a one-segment
    trace shows no evolution.

    Args:
        recordings: One row per recording, carrying ``guid``, the class and subgroup columns, and
            ``n_segments`` -- how many segments the dataset holds for it.
        per_class: Recordings to draw from each class, as an upper bound.
        seed: The draw's seed.
        min_segments: The eligibility floor.

    Returns:
        ``(chosen, accounting)``: the drawn rows in class order, and per class how many
        recordings the split held, how many were eligible and how many were drawn -- beside the
        count of recordings with no class at all, which are never drawn under a class heading.
    """
    accounting: Dict[str, Any] = {
        "per_class_cap": int(per_class),
        "min_segments": int(min_segments),
        "seed": int(seed),
        "n_unlabelled_recordings": 0,
        "classes": {},
    }
    if recordings.empty or labels.CLASS_COLUMN not in recordings.columns:
        return recordings.head(0), accounting

    labelled = recordings[recordings[labels.CLASS_COLUMN].notna()]
    accounting["n_unlabelled_recordings"] = int(len(recordings) - len(labelled))
    known = [name for name in labels.CLASS_NAMES.values() if name in set(labelled[labels.CLASS_COLUMN])]
    ordered = known + sorted(
        (name for name in set(labelled[labels.CLASS_COLUMN]) if name not in known), key=str
    )
    # One generator walked in a fixed class order, so the draw is a function of the seed alone
    # and not of the order the table happened to arrive in.
    generator = np.random.default_rng(int(seed))
    pieces: List[pd.DataFrame] = []
    for name in ordered:
        members = labelled[labelled[labels.CLASS_COLUMN] == name].sort_values("guid")
        eligible = members[members["n_segments"] >= int(min_segments)]
        take = min(int(per_class), len(eligible))
        drawn = eligible.iloc[np.sort(generator.permutation(len(eligible))[:take])] if take else eligible.head(0)
        accounting["classes"][str(name)] = {
            "n_recordings": int(len(members)),
            "n_eligible": int(len(eligible)),
            "n_selected": int(len(drawn)),
        }
        pieces.append(drawn)
    chosen = pd.concat(pieces, ignore_index=True) if pieces else labelled.head(0)
    return chosen, accounting


# =============================================================================
# The full form
# =============================================================================
def lag_statistic_columns(prefix: str) -> Dict[str, str]:
    """Return ``{statistic key: column name}`` for one lag profile family."""
    return {key: f"{prefix}_{suffix}" for key, suffix in LAG_STATISTIC_SUFFIXES.items()}


def _lag_statistics(rows: np.ndarray, seconds: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    """The shape statistics of a stack of lag profiles, under this family's column names."""
    statistics, _record = lag_shape.profile_statistics(rows, seconds)
    columns = lag_statistic_columns(prefix)
    return {columns[key]: values for key, values in statistics.items()}


def _derived_latent_scalars(vectors: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    r"""Per-anchor scalars read off the latent vectors, wherever the vectors are present.

    * ``mu_prior_norm``, ``mu_post_norm`` -- $\lVert \mu \rVert_2$ over the latent coordinates;
      ``delta_mu_norm`` -- $\lVert \mu^q - \mu^p \rVert_2$, how far the source moved the belief.
    * ``mean_logvar_prior``, ``mean_logvar_post`` -- the mean log-variance over coordinates, the
      pinned-variance detector's per-anchor form; ``prior_rate`` -- the objective's own
      $\sum_d \tfrac12(e^{\ell^p_d} - 1 - \ell^p_d)$, zero exactly at unit prior variance.
    * ``n_active_dims`` -- coordinates whose divergence clears the activity epsilon;
      ``kld_top_dim_share`` -- the largest coordinate's share of the anchor's divergence, ``NaN``
      where there is none to share.

    Args:
        vectors: The segment's per-anchor vectors.

    Returns:
        The scalars, only those the vectors support.
    """
    derived: Dict[str, np.ndarray] = {}
    mu_prior, mu_post = vectors.get("mu_prior"), vectors.get("mu_post")
    if mu_prior is not None:
        derived["mu_prior_norm"] = np.linalg.norm(mu_prior, axis=-1)
    if mu_post is not None:
        derived["mu_post_norm"] = np.linalg.norm(mu_post, axis=-1)
    if mu_prior is not None and mu_post is not None:
        derived["delta_mu_norm"] = np.linalg.norm(mu_post - mu_prior, axis=-1)
    for name in ("logvar_prior", "logvar_post"):
        logvar = vectors.get(name)
        if logvar is not None:
            derived[f"mean_{name}"] = logvar.mean(axis=-1)
    logvar_prior = vectors.get("logvar_prior")
    if logvar_prior is not None:
        derived["prior_rate"] = (0.5 * (np.exp(logvar_prior) - 1.0 - logvar_prior)).sum(axis=-1)
    kld = vectors.get("kld_per_dim")
    if kld is not None:
        derived["n_active_dims"] = (kld > KLD_ACTIVE_EPS).sum(axis=-1).astype(np.float64)
        total = kld.sum(axis=-1)
        derived["kld_top_dim_share"] = np.divide(
            kld.max(axis=-1), total, out=np.full(total.shape, np.nan), where=total > 0.0
        )
    return derived


def anchor_rows(
    segment: SegmentTrace,
    *,
    segment_order: int,
    lag_profiles: Mapping[str, str],
    lag_seconds: np.ndarray,
) -> pd.DataFrame:
    """The full form of one segment: one row per decoded anchor.

    Args:
        segment: The segment's arrays.
        segment_order: The segment's position among its recording's segments, by ``epoch``.
        lag_profiles: ``{vector name: column prefix}`` naming which vectors are lag profiles and
            what their shape statistics are called.
        lag_seconds: The compensated lag axis those profiles are read on, $(L,)$.

    Returns:
        The rows: identity, the time axis, the indicator, the cell's scalars, the derived latent
        scalars and every lag family's shape statistics, in that order.
    """
    anchor = np.asarray(segment.anchor, dtype=np.int64).reshape(-1)
    t_abs = float(segment.epoch) + anchor.astype(np.float64) * float(SECONDS_PER_STEP)
    columns: Dict[str, Any] = {
        "guid": segment.guid,
        labels.CLASS_COLUMN: segment.clinical_class,
        labels.SUBGROUP_COLUMN: segment.subgroup,
        "epoch": float(segment.epoch),
        "segment_order": int(segment_order),
        "anchor": anchor,
        "t_abs_sec": t_abs,
        cohort.HOURS_COLUMN: -t_abs / cohort.SECONDS_PER_HOUR,
        "contributing": np.asarray(segment.contributing, dtype=bool).reshape(-1),
    }
    for name, values in segment.scalars.items():
        columns[name] = np.asarray(values, dtype=np.float64).reshape(-1)
    columns.update(_derived_latent_scalars(segment.vectors))
    for name, prefix in lag_profiles.items():
        if name in segment.vectors:
            columns.update(_lag_statistics(segment.vectors[name], lag_seconds, prefix))
    frame = pd.DataFrame(columns)
    for name in ("guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN):
        frame[name] = frame[name].astype(object)
    return frame


# =============================================================================
# The summary form
# =============================================================================
def _segment_vector_means(segment: SegmentTrace) -> Dict[str, np.ndarray]:
    """Each vector averaged over the segment's scored anchors, ``NaN`` when it scored none."""
    keep = np.asarray(segment.contributing, dtype=bool).reshape(-1)
    means: Dict[str, np.ndarray] = {}
    for name, values in segment.vectors.items():
        array = np.asarray(values, dtype=np.float64)
        if not keep.any():
            means[name] = np.full(array.shape[1:], np.nan)
            continue
        # Over the finite entries only: a lag map is NaN at the lags an anchor could not read,
        # and a lag live at some anchors has a mean over those rather than none at all. A column
        # finite nowhere stays NaN, which the warning would otherwise announce on every segment.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means[name] = np.nanmean(array[keep], axis=0)
    return means


def _latent_dispersion(segment: SegmentTrace, mean: Optional[np.ndarray]) -> float:
    r"""Within-segment spread of the posterior mean, $\sqrt{\operatorname{mean}_t \lVert \mu^q_t - \bar\mu^q \rVert^2}$.

    The number that says whether a segment mean is a summary at all: if it is comparable to the
    step between consecutive segments, the path is a walk inside a cloud and the connecting line
    is decoration.
    """
    mu_post = segment.vectors.get("mu_post")
    keep = np.asarray(segment.contributing, dtype=bool).reshape(-1)
    if mu_post is None or mean is None or not keep.any():
        return float("nan")
    residual = np.asarray(mu_post, dtype=np.float64)[keep] - mean[None, :]
    return float(np.sqrt((residual ** 2).sum(axis=-1).mean()))


def assemble_recording(
    segments: Sequence[SegmentTrace],
    *,
    lag_profiles: Mapping[str, str],
    lag_seconds: np.ndarray,
    break_after_s: float,
) -> RecordingTrace:
    """Order a recording's segments by ``epoch`` and build both forms.

    The summary averages every per-anchor scalar over the segment's scored anchors, **except** the
    lag-shape columns: those are recomputed on the segment's *mean profile*, because the mean of a
    centroid is not the centroid of the mean, and the profile-of-the-mean is what the clock
    analyses report per segment -- so a trace and a clock page describe the same object.

    Args:
        segments: The recording's segment traces, in any order. Every segment must carry the
            same ``guid``.
        lag_profiles: ``{vector name: column prefix}``, as for :func:`anchor_rows`.
        lag_seconds: The compensated lag axis, $(L,)$.
        break_after_s: Seconds an epoch gap must exceed to count as a missing stretch of the
            recording rather than the ordinary stride between stored segments.

    Returns:
        The assembled recording.

    Raises:
        ValueError: If the segments do not all belong to one recording.
    """
    guids = {str(segment.guid) for segment in segments}
    if len(guids) != 1:
        raise ValueError(
            f"a recording trace is assembled from one recording's segments, got {sorted(guids)}"
        )
    ordered = sorted(segments, key=lambda segment: float(segment.epoch))
    guid = ordered[0].guid
    clinical_class = ordered[0].clinical_class
    subgroup = ordered[0].subgroup

    frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []
    vector_means: Dict[str, List[np.ndarray]] = {}
    previous_mu: Optional[np.ndarray] = None
    previous_epoch: Optional[float] = None
    for order, segment in enumerate(ordered):
        rows = anchor_rows(
            segment, segment_order=order, lag_profiles=lag_profiles, lag_seconds=lag_seconds
        )
        frames.append(rows)
        means = _segment_vector_means(segment)
        for name, value in means.items():
            vector_means.setdefault(name, []).append(value)

        scored = rows[rows["contributing"]]
        lag_columns = {
            column
            for prefix in lag_profiles.values()
            for column in lag_statistic_columns(prefix).values()
        }
        averaged = {
            name: (float(scored[name].mean()) if len(scored) else float("nan"))
            for name in rows.columns
            if name not in IDENTITY_COLUMNS
            and name not in ("anchor", "contributing", "t_abs_sec", cohort.HOURS_COLUMN)
            and name not in lag_columns
            and pd.api.types.is_numeric_dtype(rows[name])
        }
        # The shape of the segment's mean profile, not the mean of the per-anchor shapes.
        lag_statistics: Dict[str, float] = {}
        for name, prefix in lag_profiles.items():
            if name in means:
                stacked = _lag_statistics(means[name][None, :], lag_seconds, prefix)
                lag_statistics.update({column: float(values[0]) for column, values in stacked.items()})

        mu_mean = means.get("mu_post")
        epoch = float(segment.epoch)
        gap = float("nan") if previous_epoch is None else epoch - previous_epoch
        step = (
            float("nan")
            if previous_mu is None or mu_mean is None
            else float(np.linalg.norm(mu_mean - previous_mu))
        )
        summary_rows.append(
            {
                "guid": guid,
                labels.CLASS_COLUMN: clinical_class,
                labels.SUBGROUP_COLUMN: subgroup,
                "epoch": epoch,
                "segment_order": order,
                cohort.HOURS_COLUMN: -epoch / cohort.SECONDS_PER_HOUR,
                **{name: float(value) for name, value in segment.clocks.items()},
                "n_anchors": int(len(rows)),
                "n_contributing": int(len(scored)),
                "t_abs_first_sec": float(rows["t_abs_sec"].min()) if len(rows) else float("nan"),
                "t_abs_last_sec": float(rows["t_abs_sec"].max()) if len(rows) else float("nan"),
                "epoch_gap_s": gap,
                "is_break": bool(np.isfinite(gap) and gap > float(break_after_s)),
                **averaged,
                **lag_statistics,
                "latent_dispersion": _latent_dispersion(segment, mu_mean),
                "latent_step": step,
            }
        )
        # A segment that scored nothing leaves the step undefined rather than carrying the last
        # scored mean forward across it.
        previous_mu = mu_mean if (mu_mean is not None and np.isfinite(mu_mean).all()) else None
        previous_epoch = epoch

    anchors = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return RecordingTrace(
        guid=guid,
        clinical_class=clinical_class,
        subgroup=subgroup,
        segments=ordered,
        anchors=anchors,
        summary=summary,
        segment_vectors={name: np.stack(values, axis=0) for name, values in vector_means.items()},
    )


# =============================================================================
# On disk
# =============================================================================
def recording_stem(guid: Any, subgroup: Any) -> str:
    """The filename stem a recording's artifacts share: the GUID and its subgroup."""
    return f"{sanitise_guid(guid)}_{sanitise_guid(subgroup if subgroup is not None else 'na')}"


def class_dirname(clinical_class: Any) -> str:
    """The class subdirectory a recording's files go in."""
    return str(clinical_class) if clinical_class is not None else UNLABELLED_CLASS


def write_recording_arrays(path: Any, recording: RecordingTrace, *, lag_seconds: np.ndarray) -> Path:
    """Write a recording's full per-anchor vectors and its per-segment means, compressed.

    The scalar tables are CSV and parquet elsewhere; this file carries what a table cannot -- the
    $(N, d_z)$ latent path, the $(N, d_z)$ per-coordinate divergence and the $(N, L)$ lag maps
    over every decoded anchor of every segment -- beside the axes that place each row.

    Args:
        path: Destination, including the extension.
        recording: The assembled recording.
        lag_seconds: The compensated lag axis, written beside the lag maps so they can be read
            without the run's delay record.

    Returns:
        The path written.
    """
    anchors = recording.anchors
    arrays: Dict[str, np.ndarray] = {
        "epoch": np.asarray(anchors["epoch"], dtype=np.float64),
        "segment_order": np.asarray(anchors["segment_order"], dtype=np.int64),
        "anchor": np.asarray(anchors["anchor"], dtype=np.int64),
        "t_abs_sec": np.asarray(anchors["t_abs_sec"], dtype=np.float64),
        "contributing": np.asarray(anchors["contributing"], dtype=bool),
        "lag_seconds": np.asarray(lag_seconds, dtype=np.float64),
        "segment_epoch": np.asarray(recording.summary["epoch"], dtype=np.float64),
    }
    names = sorted({name for segment in recording.segments for name in segment.vectors})
    for name in names:
        arrays[name] = np.concatenate(
            [np.asarray(segment.vectors[name], dtype=np.float32) for segment in recording.segments],
            axis=0,
        )
    for name, values in recording.segment_vectors.items():
        arrays[f"segment_{name}"] = np.asarray(values, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


# =============================================================================
# The figures
# =============================================================================
class HeatmapPanel(NamedTuple):
    r"""One heatmap row of the recording figure: a per-anchor vector against time.

    Attributes:
        vector: The vector's name in :attr:`SegmentTrace.vectors`.
        title: The panel title.
        ylabel: The vertical axis label.
        symmetric: Whether the colour scale is symmetric about zero (a signed quantity) or runs
            from zero (a non-negative one).
        log: Whether to draw on a logarithmic colour scale, floored :data:`LOG_PANEL_DECADES`
            below the panel's own maximum. For a lag map, where one dominant lag flattens the rest
            of the panel into the bottom colour on a linear scale.
        lag_axis: Whether the vertical axis is the compensated lag axis in seconds rather than a
            coordinate index; the caveat is printed under a figure carrying one.
        argmax_column: A per-anchor scalar column drawn over the panel as the lag the map peaks
            at, or ``None``.
        subtract: A second vector subtracted from the first before drawing, or ``None``. How the
            source shift $\mu^q - \mu^p$ is drawn without a third stored array that could
            disagree with the two it is the difference of.
    """

    vector: str
    title: str
    ylabel: str
    symmetric: bool = False
    log: bool = False
    lag_axis: bool = False
    argmax_column: Optional[str] = None
    subtract: Optional[str] = None


class LinePanel(NamedTuple):
    """One line row of the recording figure: per-anchor scalars against time.

    Attributes:
        columns: The scalar columns drawn, one line each, legend-labelled by name.
        title: The panel title.
        ylabel: The vertical axis label.
    """

    columns: Tuple[str, ...]
    title: str
    ylabel: str


def _segment_slices(frame: pd.DataFrame) -> List[pd.DataFrame]:
    """The full-form rows split per segment, each in anchor order."""
    return [
        cell.sort_values("anchor") for _, cell in frame.groupby("segment_order", sort=True)
    ]


def _draw_segment_boundaries(ax: Any, frame: pd.DataFrame) -> None:
    """Mark where each segment's first decoded anchor lands, so a step at a join reads as a join."""
    for cell in _segment_slices(frame):
        ax.axvline(
            float(cell[cohort.HOURS_COLUMN].iloc[0]), color=figures.COLOR_LIGHT_GRAY,
            linewidth=figures.LINE_HAIRLINE, zorder=0,
        )


def _panel_block(segment: SegmentTrace, panel: HeatmapPanel) -> Optional[np.ndarray]:
    """The array one heatmap panel draws for one segment, or ``None`` where it is absent."""
    block = segment.vectors.get(panel.vector)
    if block is None:
        return None
    if panel.subtract is not None:
        other = segment.vectors.get(panel.subtract)
        if other is None:
            return None
        return np.asarray(block, dtype=np.float64) - np.asarray(other, dtype=np.float64)
    return np.asarray(block, dtype=np.float64)


def _draw_heatmap(
    ax: Any,
    figure: Any,
    recording: RecordingTrace,
    panel: HeatmapPanel,
    *,
    lag_seconds: np.ndarray,
) -> None:
    """Draw one vector family of the whole recording, segment by segment, on the hours axis."""
    blocks = [_panel_block(segment, panel) for segment in recording.segments]
    present = [block for block in blocks if block is not None]
    if not present:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
        ax.set_title(panel.title)
        figures.style_axes(ax, grid="none")
        return
    stacked = np.concatenate([np.asarray(block, dtype=np.float64) for block in present], axis=0)
    finite = stacked[np.isfinite(stacked)]
    if panel.log:
        positive = finite[finite > 0.0]
        top = float(positive.max()) if positive.size else 1.0
        norm = mcolors.LogNorm(vmin=top * 10.0 ** (-LOG_PANEL_DECADES), vmax=top)
        cmap = "viridis"
    elif panel.symmetric:
        limit = float(np.abs(finite).max()) if finite.size else 1.0
        norm = mcolors.Normalize(vmin=-limit, vmax=limit)
        cmap = "RdBu_r"
    else:
        norm = mcolors.Normalize(
            vmin=0.0, vmax=float(finite.max()) if finite.size else 1.0
        )
        cmap = "viridis"

    width = float(SECONDS_PER_STEP) / cohort.SECONDS_PER_HOUR
    y_edges = (
        np.concatenate([lag_seconds - 0.5 * SECONDS_PER_STEP, [lag_seconds[-1] + 0.5 * SECONDS_PER_STEP]])
        if panel.lag_axis
        else np.arange(stacked.shape[1] + 1) - 0.5
    )
    mesh = None
    for segment, block in zip(recording.segments, blocks):
        if block is None or len(segment.anchor) == 0:
            continue
        hours = -(float(segment.epoch) + np.asarray(segment.anchor, dtype=np.float64) * SECONDS_PER_STEP) / cohort.SECONDS_PER_HOUR
        # Anchors are consecutive steps within a segment, so the cell edges are half a step
        # either side of each anchor; a gap inside a segment draws as missing cells rather than
        # as a stretched neighbour.
        x_edges = np.concatenate([hours + 0.5 * width, [hours[-1] - 0.5 * width]])
        values = np.asarray(block, dtype=np.float64).T
        if panel.log:
            values = np.where(values > 0.0, values, np.nan)
        mesh = ax.pcolormesh(x_edges, y_edges, values, cmap=cmap, norm=norm, rasterized=True, shading="flat")
        if panel.argmax_column is not None and panel.argmax_column in recording.anchors.columns:
            cell = recording.anchors[recording.anchors["epoch"] == float(segment.epoch)].sort_values("anchor")
            peak = np.asarray(cell[panel.argmax_column], dtype=np.float64)
            if panel.lag_axis:
                peak = np.where(np.isfinite(peak), lag_seconds[np.clip(peak, 0, len(lag_seconds) - 1).astype(int)], np.nan)
            ax.plot(
                hours, peak, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_THIN,
                label=panel.argmax_column,
            )
    if mesh is not None:
        figure.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03)
    if panel.argmax_column is not None:
        ax.legend(fontsize=figures.FONT_LABEL, loc="upper right")
    _draw_segment_boundaries(ax, recording.anchors)
    ax.set_title(panel.title)
    ax.set_ylabel(panel.ylabel if not panel.lag_axis else COEFFICIENT_LAG_AXIS_LABEL)
    ax.invert_xaxis()
    figures.style_axes(ax, grid="none")


def _draw_lines(ax: Any, recording: RecordingTrace, panel: LinePanel) -> None:
    """Draw per-anchor scalars of the whole recording, lifted at every unscored anchor."""
    frame = recording.anchors
    drawn = 0
    colours = [figures.COLOR_BLUE, figures.COLOR_ORANGE, figures.COLOR_GREEN, figures.COLOR_PURPLE]
    for colour, column in zip(colours, panel.columns):
        if column not in frame.columns:
            continue
        first = True
        for cell in _segment_slices(frame):
            hours = np.asarray(cell[cohort.HOURS_COLUMN], dtype=np.float64)
            values = np.asarray(cell[column], dtype=np.float64)
            # Unscored anchors are lifted rather than dropped, so the line breaks where nothing
            # was scored instead of interpolating across it.
            values = np.where(np.asarray(cell["contributing"], dtype=bool), values, np.nan)
            if np.isfinite(values).any():
                ax.plot(
                    hours, values, color=colour, linewidth=figures.LINE_REGULAR,
                    label=column if first else None,
                )
                first = False
                drawn += 1
    if drawn == 0:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
        )
    else:
        ax.legend(fontsize=figures.FONT_LABEL, loc="upper right")
        _draw_segment_boundaries(ax, frame)
        ax.invert_xaxis()
    ax.set_title(panel.title)
    ax.set_ylabel(panel.ylabel)
    figures.style_axes(ax)


def build_recording_figure(
    recording: RecordingTrace,
    *,
    panels: Sequence[Any],
    lag_seconds: np.ndarray,
    caveat: Optional[str] = None,
) -> Any:
    """Draw one recording's trace: every panel against hours before delivery, on a shared axis.

    Args:
        recording: The assembled recording.
        panels: :class:`HeatmapPanel` and :class:`LinePanel` entries, top to bottom.
        lag_seconds: The compensated lag axis, for the lag-resolved heatmaps.
        caveat: A sentence printed under the figure, or ``None``. A figure carrying a lag axis
            carries the group-delay caveat.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(panels), height_per_row=1.7, width=13.0)
    for row, panel in enumerate(panels):
        ax = axes[row, 0]
        if isinstance(panel, HeatmapPanel):
            _draw_heatmap(ax, figure, recording, panel, lag_seconds=lag_seconds)
        else:
            _draw_lines(ax, recording, panel)
        if row < len(panels) - 1:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time before delivery (hours)")
    n_segments = len(recording.segments)
    span = (
        float(recording.summary[cohort.HOURS_COLUMN].max() - recording.summary[cohort.HOURS_COLUMN].min())
        if len(recording.summary) else float("nan")
    )
    figure.suptitle(
        f"guid {recording.guid} — subgroup {recording.subgroup} — class {recording.clinical_class} "
        f"— {n_segments} segment(s) over {span:.1f} h",
        fontsize=figures.FONT_NOTE,
    )
    if caveat:
        figures.caveat_note(figure, caveat)
    return figure


class SummaryMetric(NamedTuple):
    """One row of the summary figure: a per-segment column against hours before delivery.

    Attributes:
        column: The summary column.
        ylabel: Its axis label.
    """

    column: str
    ylabel: str


def build_summary_figure(summary: pd.DataFrame, *, metrics: Sequence[SummaryMetric]) -> Any:
    """Draw every traced recording's per-segment summary, one line per recording, by class colour.

    Args:
        summary: The stacked segment summaries of every traced recording.
        metrics: The columns to draw, one panel each.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(metrics), height_per_row=1.9, width=11.0)
    classes = (
        labels.ordered_groups(
            [name for name in summary[labels.CLASS_COLUMN].dropna().unique()], labels.CLASS_COLUMN
        )
        if len(summary) and labels.CLASS_COLUMN in summary.columns else []
    )
    colours = figures.group_colors(classes)
    for row, metric in enumerate(metrics):
        ax = axes[row, 0]
        drawn = 0
        if metric.column in summary.columns:
            for guid, cell in summary.groupby("guid", sort=True):
                cell = cell.sort_values("epoch")
                name = cell[labels.CLASS_COLUMN].iloc[0]
                hours = np.asarray(cell[cohort.HOURS_COLUMN], dtype=np.float64)
                values = np.asarray(cell[metric.column], dtype=np.float64)
                # A break is a gap *before* a segment, so the line is lifted before it.
                breaks = np.flatnonzero(np.asarray(cell["is_break"], dtype=bool))
                hours = np.insert(hours, breaks, np.nan)
                values = np.insert(values, breaks, np.nan)
                if np.isfinite(values).any():
                    ax.plot(
                        hours, values, marker="o", markersize=2.0,
                        color=colours.get(str(name), figures.COLOR_GRAY),
                        linewidth=figures.LINE_THIN, alpha=0.85,
                    )
                    drawn += 1
        if drawn == 0:
            ax.text(
                0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
                ha="center", va="center", fontsize=figures.FONT_NOTE, color=figures.COLOR_GRAY,
            )
        else:
            ax.invert_xaxis()
        ax.set_title(metric.column)
        ax.set_ylabel(metric.ylabel)
        ax.set_xlabel("Time before delivery (hours)" if row == len(metrics) - 1 else "")
        figures.style_axes(ax)
    if classes:
        handles = [
            Line2D([0], [0], color=colours[name], linewidth=figures.LINE_EMPHASIS, label=name)
            for name in classes
        ]
        axes[0, 0].legend(handles=handles, fontsize=figures.FONT_LABEL, loc="upper right")
    figure.suptitle(
        "Per-segment summaries of every traced recording, one line per recording",
        fontsize=figures.FONT_NOTE,
    )
    return figure


__all__ = [
    "ANALYSIS_DIRNAME",
    "ANCHOR_TRACE_FILENAME",
    "DEFAULT_TRACES_PER_CLASS",
    "FULL_ARRAYS_SUFFIX",
    "HeatmapPanel",
    "IDENTITY_COLUMNS",
    "LinePanel",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILENAME",
    "MIN_SEGMENTS_PER_TRACE",
    "RecordingTrace",
    "SEGMENT_SUMMARY_FILENAME",
    "SUMMARY_FIGURE",
    "SegmentTrace",
    "SummaryMetric",
    "TRACES_CAP",
    "TRACE_DRAW_SEED_OFFSET",
    "TRACE_FIGURE_SUFFIX",
    "UNLABELLED_CLASS",
    "anchor_rows",
    "assemble_recording",
    "build_recording_figure",
    "build_summary_figure",
    "class_dirname",
    "lag_statistic_columns",
    "recording_stem",
    "select_recordings",
    "write_recording_arrays",
]
