r"""The attribution pass: which segments, which anchors, every Captum call, and what is written.

:mod:`~teb_vae.lag_attn_cfs.eval.attributions` is the arithmetic -- the wrapper, the baselines,
the Captum calls, the reductions and the figure builders. This module is the **pass** both cells
run over it: it draws the segments and the traced recordings, re-reads them through a strictly
sequential subset loader with the identity check, runs every attribution at every chosen anchor,
reduces the maps to the durable tables, and writes the files, the figures and the block a summary
carries. The lag-attentive cells reach it from their registered analysis, the lag-residual cell
from its post-pass stage, so the two directories hold the same tables under the same names and a
reader can lay one cell's attribution beside the other's.

**The selection is per recording.** Every summary here is over recordings, so a recording stays
one unit however many of its segments are attributed: its rows are averaged per segment, then per
recording. The recordings are drawn with the traces' own class-balanced, seeded draw, spread
round-robin over each class's subgroups, at an eligibility floor of one segment. The main pass
takes ``caps.attribution_segments_per_recording`` segments from each, spread evenly over its
stored timeline (one: the middle segment); the cohort pass always takes one, because its tests
count recordings and are better served by more recordings than by more segments of the same
ones. The cap is on segments, so each class draws ``ceil(cap / (n_classes * per_recording))``
recordings, and the draw stops at the cap.

**The anchors are each segment's own most coupled ones**: $K_t$ in the top $30\%$ of that
segment's scored anchors, with a clean forecast window and history, at most
:data:`~.attributions.ANCHORS_PER_SEGMENT` of them at least one horizon apart.

**What is attributed per anchor.** The main readouts (:data:`~.attributions.MAIN_READOUTS`: the
divergence, the mean-decoded forecast gap, the full block score and the full block error) under
both baselines; the model's own lag readout on every configured lag band under the source-null
baseline alone (the target streams are held there, so the question is what source content makes
the model read that band); the anchor's largest per-coordinate divergence under the source-null
baseline; the layer split of every main readout; and the source zeroed band by band, as feature
ablation, on every main readout. The structural checks are measured on every row
of a real run and land in the block beside the fixture-proved ones.

**One example anchor per class keeps its full maps**, for the example pages and the overview,
beside the raw FHR and UP of its segment. Its maps are the main calls' own rows at that anchor,
kept as they are made; only the variants no main call takes are integrated again for it.

**The class comparison has its own, larger draw** (:func:`run_cohort`). One example per class
cannot separate a class from a recording, and a main-pass segment is too costly to scale up. The
cohort pass draws ``caps.attribution_cohort_segments`` segments by the same rule, integrates only
the divergence and the forecast gap under the source-null baseline, adds Grad-CAM
(:mod:`~teb_vae.lag_attn_cfs.eval.gradcam`), and hands the result to
:mod:`~teb_vae.lag_attn_cfs.eval.class_contrast` for class means, class differences and tests.

**Every written table names its rows**: ``guid``, subgroup, class, ``epoch`` and ``anchor`` on
every per-row and per-anchor file (the row-aligned arrays carry them under a ``row_`` prefix), and
the ``unit`` of the readout, which is also the unit of its attributions.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import class_contrast, cohort, frames, gradcam, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import (
    check_batch_identity,
    epoch_stamp,
    subset_loader,
)
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, batch_field, model_inputs

#: The manifest of traced recordings, and its columns.
#: The manifest of the example pages -- one anchor per class with every example readout's maps --
#: and its columns: whose anchor, where in the cohort, when, and where the page is.
EXAMPLE_MANIFEST_FILENAME = "attribution_examples.csv"
EXAMPLE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "epoch", "anchor", "figure_file",
)

TRACE_MANIFEST_FILENAME = "attribution_traces.csv"
TRACE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "n_segments", "n_anchors", "span_hours",
    "coverage", "arrays_file", "figure_file",
)

#: The per-recording columns declared for the runner's by-class and by-subgroup fan-out: the
#: source and target totals and the two agreement statistics of the two main readouts under the
#: source-null baseline.
RECORDING_VALUE_COLUMNS: Tuple[str, ...] = tuple(
    f"{readout}_{name}"
    for readout in core.MAIN_READOUTS
    for name in ("source_total", "target_total", "lag_corr", "lag_js")
)

#: The files another analysis left behind that this pass joins where they exist. Names restated
#: here rather than imported: an analysis may not import another, and this module serves one.
OCCLUSION_SUMMARY_PATH = ("occlusion", "occlusion_summary.csv")
SPECTRAL_BANDS_PATH = ("spectral_skill", "spectral_skill_bands.csv")
CHANNEL_MAP_FILENAME = "band_channel_map.csv"

#: Below this share of a readout's magnitude, a completeness residual counts as converged.
COMPLETENESS_TOLERANCE = 1e-2


# =============================================================================
# Selection
# =============================================================================
def labelled_segments(rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce a per-segment table to the four identity columns, dropping duplicates.

    Args:
        rows: A frame carrying ``guid``, ``epoch`` and the two cohort columns.

    Returns:
        One row per ``(guid, epoch)``.
    """
    columns = ["guid", "epoch", *labels.GROUP_COLUMNS]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=columns)
    present = [name for name in columns if name in rows.columns]
    frame = rows[present].copy()
    for name in columns:
        if name not in frame.columns:
            frame[name] = None
    frame["guid"] = frame["guid"].astype(str)
    frame["epoch"] = frame["epoch"].astype(np.float64)
    return frame.drop_duplicates(subset=["guid", "epoch"]).reset_index(drop=True)[columns]


def recording_table(segments: pd.DataFrame, index_map: Mapping[Tuple[str, Optional[int]], int]) -> pd.DataFrame:
    """One row per labelled recording with its segment count in the dataset.

    Args:
        segments: From :func:`labelled_segments`.
        index_map: The dataset's ``{(guid, rounded epoch): index}`` listing.

    Returns:
        ``guid``, the cohort columns and ``n_segments``.
    """
    labelled = frames.per_recording_labels(segments)
    if labelled.empty:
        return pd.DataFrame(columns=["guid", *labels.GROUP_COLUMNS, "n_segments"])
    counts: Dict[str, int] = {}
    for guid, _stamp in index_map:
        counts[guid] = counts.get(guid, 0) + 1
    frame = labelled.reset_index()
    frame["n_segments"] = [int(counts.get(str(guid), 0)) for guid in frame["guid"]]
    return frame


def informative_anchors(
    per_anchor: Optional[pd.DataFrame],
    *,
    quantile: float = core.HIGH_KL_QUANTILE,
    coverage: float = core.CLEAN_COVERAGE,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    r"""The anchors worth explaining: high $K_t$ within their own segment, and a clean forecast window.

    The threshold is the ``quantile`` of $K_t$ over **each segment's** scored anchors, so every
    segment offers its own most coupled anchors. Under a threshold pooled over the cohort, a
    subgroup whose coupling is weak at every anchor offered none and vanished from every
    by-subgroup figure. The cost is that a class contrast now compares each segment's relatively
    most coupled moments, not anchors above one cohort-wide level of $K_t$; the record keeps the
    pooled level beside the per-segment ones so a reader can see how far apart the two are.

    Args:
        per_anchor: The collection pass's per-anchor table, with ``guid``, ``epoch``, ``anchor``,
            ``kld_per_t`` and ``coverage``; ``None`` or incomplete falls back to evenly spread
            anchors.
        quantile: The per-segment $K_t$ quantile an anchor must reach.
        coverage: The forecast coverage an anchor must reach.

    Returns:
        ``(candidates, record)``: ``guid``, ``epoch``, ``anchor`` and ``kld_per_t`` of every
        qualifying anchor, highest $K_t$ first, or ``None``; and the rule with its thresholds and
        counts.
    """
    needed = {"guid", "epoch", "anchor", "kld_per_t", "coverage"}
    if per_anchor is None or per_anchor.empty or not needed <= set(per_anchor.columns):
        return None, {"rule": "spread", "reason": "no per-anchor table with kld_per_t and coverage"}
    kl = per_anchor["kld_per_t"].to_numpy(dtype=np.float64)
    per_segment = (
        per_anchor.groupby(["guid", "epoch"], dropna=False)["kld_per_t"]
        .transform(lambda values: values.quantile(float(quantile)))
        .to_numpy(dtype=np.float64)
    )
    keep = (kl >= per_segment) & (per_anchor["coverage"].to_numpy(dtype=np.float64) >= float(coverage))
    candidates = per_anchor.loc[keep, ["guid", "epoch", "anchor", "kld_per_t"]].copy()
    candidates["guid"] = candidates["guid"].astype(str)
    candidates = candidates.sort_values("kld_per_t", ascending=False).reset_index(drop=True)
    thresholds = pd.Series(per_segment).groupby(
        [per_anchor["guid"].to_numpy(), per_anchor["epoch"].to_numpy()], dropna=False
    ).first()
    return candidates, {
        "rule": "high_kl_clean_per_segment", "kl_quantile": float(quantile),
        "kl_threshold_nats_median": float(np.nanmedian(thresholds)) if len(thresholds) else float("nan"),
        "kl_threshold_nats_pooled": float(np.nanquantile(kl, float(quantile))),
        "coverage_min": float(coverage), "history_validity_min": float(coverage),
        "n_scored_anchors": int(len(per_anchor)), "n_candidate_anchors": int(len(candidates)),
        "n_recordings_with_candidates": int(candidates["guid"].nunique()),
    }


def spread_positions(n_items: int, count: int) -> List[int]:
    r"""Pick ``count`` positions spread evenly over ``n_items`` ordered items.

    Position $i$ is $\lfloor (i + \tfrac12)\, n / k \rfloor$, the centre of the $i$-th of $k$ equal
    slices, so one item is the middle one, ``n // 2``, and $k \ge n$ takes every item.

    Args:
        n_items: How many ordered items there are, $n$.
        count: How many to pick, $k$.

    Returns:
        The distinct positions, ascending.
    """
    if count >= n_items:
        return list(range(n_items))
    return [int((index + 0.5) * n_items / count) for index in range(count)]


def select_segments(
    segments: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    *,
    cap: int,
    seed: int,
    candidates: Optional[pd.DataFrame] = None,
    examples_per_class: int = 0,
    segments_per_recording: int = core.DEFAULT_SEGMENTS_PER_RECORDING,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Draw the segments to attribute: class-balanced, spread over subgroups, seeded, capped.

    Each class draws ``ceil(cap / (n_classes * segments_per_recording))`` recordings, round-robin
    over its subgroups so that every subgroup a class holds is drawn before any is drawn twice.
    Each drawn recording then gives ``segments_per_recording`` segments spread evenly over its
    stored timeline by ``epoch`` -- its middle segment when that is one. With ``candidates`` only
    segments holding a candidate are eligible, and each row also carries ``anchors``, that
    segment's candidate steps highest $K_t$ first, and ``top_kld``. The best segment of each of
    the ``examples_per_class`` recordings of a class with the largest ``top_kld`` gets an
    ``example_rank`` $0, 1, \\ldots$ for the example pages, so no recording is paged twice.

    Args:
        segments: From :func:`labelled_segments`.
        index_map: The dataset listing.
        cap: The segment cap.
        seed: The draw's seed.
        candidates: From :func:`informative_anchors`, or ``None``.
        examples_per_class: Example pages per class; used with ``candidates`` only.
        segments_per_recording: Segments taken from each drawn recording, at least one.

    Returns:
        ``(rows, accounting)``: the chosen rows with ``dataset_index``, ascending in that index,
        and the draw's accounting with the cap recorded.
    """
    per_recording = max(int(segments_per_recording), 1)
    if candidates is not None:
        candidates = candidates.assign(stamp=[epoch_stamp(e) for e in candidates["epoch"]])
        held = set(zip(candidates["guid"].astype(str), candidates["stamp"]))
        keep = [(str(g), epoch_stamp(e)) in held for g, e in zip(segments["guid"], segments["epoch"])]
        segments = segments[np.asarray(keep, dtype=bool)] if len(segments) else segments
    recordings = recording_table(segments, index_map)
    classes = [name for name in recordings[labels.CLASS_COLUMN].dropna().unique()] if len(recordings) else []
    per_class = int(math.ceil(int(cap) / (max(len(classes), 1) * per_recording)))
    chosen, accounting = traces.select_recordings(
        recordings, per_class=per_class, seed=int(seed), min_segments=1,
        stratify_column=labels.SUBGROUP_COLUMN,
    )
    accounting["segment_cap"] = int(cap)
    accounting["segments_per_recording"] = per_recording
    picked: List[Dict[str, Any]] = []
    for _, choice in chosen.iterrows():
        guid = str(choice["guid"])
        own = segments[segments["guid"].astype(str) == guid].sort_values("epoch")
        mine = candidates[candidates["guid"] == guid] if candidates is not None else None
        for position in spread_positions(len(own), per_recording):
            if len(picked) >= int(cap):
                break
            segment = own.iloc[position]
            stamp = epoch_stamp(segment["epoch"])
            index = index_map.get((guid, stamp))
            if index is None:
                continue
            extra: Dict[str, Any] = {}
            if mine is not None:
                steps = mine[mine["stamp"] == stamp]
                extra = {"anchors": steps["anchor"].astype(int).tolist(), "top_kld": float(steps["kld_per_t"].max())}
            picked.append(
                {
                    "guid": guid, "epoch": float(segment["epoch"]),
                    labels.CLASS_COLUMN: choice[labels.CLASS_COLUMN],
                    labels.SUBGROUP_COLUMN: choice[labels.SUBGROUP_COLUMN],
                    "dataset_index": int(index), **extra,
                }
            )
    columns = ["guid", "epoch", *labels.GROUP_COLUMNS, "dataset_index"]
    if candidates is not None:
        columns += ["anchors", "top_kld"]
    rows = pd.DataFrame(picked, columns=columns)
    if candidates is not None and examples_per_class > 0 and len(rows):
        best = rows["top_kld"] == rows.groupby("guid")["top_kld"].transform("max")
        best &= ~rows.duplicated(subset=["guid", "top_kld"])
        rank = rows[best].groupby(labels.CLASS_COLUMN)["top_kld"].rank(method="first", ascending=False) - 1
        rows["example_rank"] = rank.where(rank < int(examples_per_class)).reindex(rows.index)
    accounting["n_segments_selected"] = int(len(rows))
    return rows.sort_values("dataset_index").reset_index(drop=True), accounting


def recording_completeness(
    index_map: Mapping[Tuple[str, Optional[int]], int],
    *,
    stride_s: float,
    window_hours: Optional[float],
) -> pd.DataFrame:
    r"""How completely each recording's stored segments fill the window a trace is read over.

    $$\texttt{coverage} = \frac{n_{\mathrm{window}}}{\lfloor S / \Delta_{\mathrm{seg}} \rfloor + 1},$$

    with $n_{\mathrm{window}}$ the segments the dataset holds inside the window, $S$ the window's
    span in seconds and $\Delta_{\mathrm{seg}}$ the stride between consecutive stored segments. The
    window is the last ``window_hours`` before delivery when the run bounds its clocks, and the
    recording's own span otherwise -- so a recording with a missing hour scores below one whose
    segments tile the span, and the trace lands on the recording with the fewest holes rather
    than on a random one.

    Args:
        index_map: The dataset's ``{(guid, rounded epoch): index}`` listing, which is what the
            trace re-reads and therefore what completeness is measured on.
        stride_s: $\Delta_{\mathrm{seg}}$.
        window_hours: The bound in hours, or ``None`` for each recording's own span.

    Returns:
        One row per recording: ``guid``, ``n_segments``, ``n_in_window``, ``n_expected``,
        ``span_hours`` and ``coverage`` in $[0, 1]$.
    """
    epochs: Dict[str, List[float]] = {}
    for (guid, stamp), _index in index_map.items():
        if stamp is not None:
            epochs.setdefault(str(guid), []).append(float(stamp))
    rows: List[Dict[str, Any]] = []
    for guid, values in epochs.items():
        stamps = np.sort(np.asarray(values, dtype=np.float64))
        if window_hours is not None:
            inside = stamps[stamps >= -float(window_hours) * cohort.SECONDS_PER_HOUR]
            span = float(window_hours) * cohort.SECONDS_PER_HOUR
        else:
            inside = stamps
            span = float(stamps.max() - stamps.min()) if stamps.size else 0.0
        expected = int(span // float(stride_s)) + 1
        rows.append(
            {
                "guid": guid, "n_segments": int(stamps.size), "n_in_window": int(inside.size),
                "n_expected": expected, "span_hours": span / cohort.SECONDS_PER_HOUR,
                "coverage": min(float(inside.size) / float(expected), 1.0),
            }
        )
    return pd.DataFrame(rows, columns=["guid", "n_segments", "n_in_window", "n_expected", "span_hours", "coverage"])


def select_trace_recordings(
    segments: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    *,
    stride_s: float,
    window_hours: Optional[float],
    per_class: int = core.TRACE_RECORDINGS_PER_CLASS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The recordings whose attribution is followed through every segment: the most complete per class.

    Ranked by :func:`recording_completeness` rather than drawn, because a trace over a recording
    with a missing hour shows the hole and not the evolution; ties break on the segment count and
    then on the identifier, so two runs choose the same recording. A recording with fewer than
    two segments inside the window has no evolution to show and is never chosen.

    Args:
        segments: From :func:`labelled_segments`, for the class of each recording.
        index_map: The dataset listing, for the segments each recording holds.
        stride_s: The stride between consecutive stored segments, in seconds.
        window_hours: The bound the run reads its clocks over, or ``None``.
        per_class: Recordings per class.

    Returns:
        ``(chosen, accounting)``: the chosen rows with the cohort columns and the completeness
        columns, and per class how many recordings were labelled, eligible and chosen, with each
        chosen recording's coverage.
    """
    labelled = frames.per_recording_labels(segments).reset_index() if not segments.empty else pd.DataFrame(columns=["guid", *labels.GROUP_COLUMNS])
    completeness = recording_completeness(index_map, stride_s=stride_s, window_hours=window_hours)
    table = labelled.merge(completeness, on="guid", how="inner") if len(labelled) else completeness.head(0)
    accounting: Dict[str, Any] = {
        "per_class": int(per_class), "min_segments_in_window": traces.MIN_SEGMENTS_PER_TRACE,
        "window_hours": None if window_hours is None else float(window_hours),
        "segment_stride_s": float(stride_s), "classes": {},
    }
    pieces: List[pd.DataFrame] = []
    if not table.empty and labels.CLASS_COLUMN in table.columns:
        known = table[table[labels.CLASS_COLUMN].notna()]
        for name in labels.ordered_groups(list(known[labels.CLASS_COLUMN].unique()), labels.CLASS_COLUMN):
            members = known[known[labels.CLASS_COLUMN].astype(object) == name]
            eligible = members[members["n_in_window"] >= traces.MIN_SEGMENTS_PER_TRACE]
            ranked = eligible.sort_values(["coverage", "n_in_window", "guid"], ascending=[False, False, True])
            chosen = ranked.head(int(per_class))
            accounting["classes"][str(name)] = {
                "n_recordings": int(len(members)), "n_eligible": int(len(eligible)), "n_selected": int(len(chosen)),
                "coverage": [float(value) for value in chosen["coverage"]],
            }
            pieces.append(chosen)
    columns = ["guid", *labels.GROUP_COLUMNS, "n_segments", "n_in_window", "n_expected", "span_hours", "coverage"]
    chosen_all = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    return chosen_all[[c for c in columns if c in chosen_all.columns]].reset_index(drop=True), accounting


def recording_rows(guid: str, index_map: Mapping[Tuple[str, Optional[int]], int]) -> pd.DataFrame:
    """Every dataset row of one recording, ascending in dataset index.

    Args:
        guid: The recording.
        index_map: The dataset listing.

    Returns:
        ``guid``, ``epoch`` and ``dataset_index`` per stored segment.
    """
    listed = sorted(
        (index, stamp) for (listed_guid, stamp), index in index_map.items()
        if listed_guid == str(guid) and stamp is not None
    )
    return pd.DataFrame(
        {"guid": [str(guid)] * len(listed), "epoch": [float(stamp) for _, stamp in listed],
         "dataset_index": [int(index) for index, _ in listed]}
    )


# =============================================================================
# Joins from other analyses' files
# =============================================================================
def read_channel_map(results_dir: Any) -> Optional[pd.DataFrame]:
    """The declared-axis channel map, or ``None`` when the run wrote none."""
    path = Path(results_dir) / CHANNEL_MAP_FILENAME
    return pd.read_csv(path) if path.is_file() else None


def read_occlusion_summary(results_dir: Any) -> Optional[pd.DataFrame]:
    """The occlusion analysis's per-band summary, or ``None`` when its pass never ran here."""
    path = Path(results_dir).joinpath(*OCCLUSION_SUMMARY_PATH)
    return pd.read_csv(path) if path.is_file() else None


def read_spectral_bands(results_dir: Any) -> Optional[pd.DataFrame]:
    """The band-resolved skill table, or ``None`` when that analysis never ran here."""
    path = Path(results_dir).joinpath(*SPECTRAL_BANDS_PATH)
    return pd.read_csv(path) if path.is_file() else None


# =============================================================================
# One batch
# =============================================================================
class BatchWork:
    """Everything one batch's attributions produce, before they are reduced into tables.

    Attributes:
        rows: One record per attribution row.
        vectors: Row-aligned arrays, appended per row: the two time profiles, the lag profile,
            the model profile, the two channel profiles and the layer split.
        examples: One example anchor per class, keyed by class name, as
            :func:`attribute_example` builds it: the inputs the encoders read and the full maps
            of every example readout under both baselines, for the map figures.
        lag_channel: Running sums of the maps re-indexed by offset from the anchor, keyed by
            ``(readout, baseline, stream)``: ``sum`` and ``abs_sum`` $(L, C)$ over the rows and
            ``count`` $(L, C)$ of the finite cells, for the population lag-by-channel maps.
            Accumulated rather than kept per row because a row's map is $L \\times C$ and the
            figure wants only the mean.
        lag_channel_by_class: The same running sums per clinical class, keyed
            ``(readout, baseline, stream, class)``, for the class-mean maps.
        gradcam_rows: One record per Grad-CAM row (anchor and readout): identity, readout value
            and the unnormalised total of each view.
        gradcam_vectors: Row-aligned normalised Grad-CAM profiles, one list per view.
        n_scattering: Width of the target scattering block, read off the first batch.
        forward_equivalents: How many single-row forward-and-backward passes the batch cost.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.rows: List[Dict[str, Any]] = []
        self.vectors: Dict[str, List[np.ndarray]] = {}
        self.examples: Dict[str, Dict[str, Any]] = {}
        self.lag_channel: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}
        self.lag_channel_by_class: Dict[Tuple[str, ...], Dict[str, np.ndarray]] = {}
        self.gradcam_rows: List[Dict[str, Any]] = []
        self.gradcam_vectors: Dict[str, List[np.ndarray]] = {}
        self.n_scattering: Optional[int] = None
        self.forward_equivalents = 0
        # Anchors the informative rule kept although their lag-window history failed the check.
        self.n_unclean_anchors = 0

    def accumulate_lag_channel(
        self, key: Tuple[str, ...], aligned: np.ndarray, store: Optional[Dict[Tuple[str, ...], Dict[str, np.ndarray]]] = None
    ) -> None:
        """Add one call's $(N, L, C)$ offset-aligned maps to the running sums under ``key``, in
        ``store`` (the pooled sums by default)."""
        store = self.lag_channel if store is None else store
        field = np.asarray(aligned, dtype=np.float64)
        finite = np.isfinite(field)
        entry = store.get(key)
        if entry is None:
            entry = {
                "sum": np.zeros(field.shape[1:]), "abs_sum": np.zeros(field.shape[1:]),
                "count": np.zeros(field.shape[1:]),
            }
            store[key] = entry
        entry["sum"] += np.where(finite, field, 0.0).sum(axis=0)
        entry["abs_sum"] += np.where(finite, np.abs(field), 0.0).sum(axis=0)
        entry["count"] += finite.sum(axis=0)

    def append_vector(self, name: str, values: np.ndarray) -> None:
        """Append one row-aligned array under a name."""
        self.vectors.setdefault(name, []).append(np.asarray(values))


def _identity(batch: Any, rows: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per sample of a batch: guid, epoch, class, subgroup, from the rows it was built from."""
    identity: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        identity.append(
            {
                "guid": str(row["guid"]), "epoch": float(row["epoch"]),
                labels.CLASS_COLUMN: row.get(labels.CLASS_COLUMN),
                labels.SUBGROUP_COLUMN: row.get(labels.SUBGROUP_COLUMN),
            }
        )
    return identity


def attribute_example(
    task: Any,
    inputs: Sequence[torch.Tensor],
    extra: Sequence[torch.Tensor],
    cell: core.CellBinding,
    *,
    sample: int,
    column: int,
    anchor: int,
    identity: Mapping[str, Any],
    model_profile: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
    n_steps: int,
    likelihood: str,
    live: Mapping[str, Optional[np.ndarray]],
    horizons: Optional[Mapping[str, int]] = None,
    latent: Optional[Mapping[str, np.ndarray]] = None,
    maps: Optional[Mapping[Tuple[str, str, str], Dict[str, Any]]] = None,
    layer: Optional[Mapping[Tuple[str, str], np.ndarray]] = None,
    raw: Optional[Mapping[str, np.ndarray]] = None,
    raw_units: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], int]:
    r"""Attribute every example readout at one anchor of one sample, keeping the full maps.

    The main pass keeps only reductions of its maps; this keeps the maps themselves, for one
    anchor per class, so the map figures can show which coefficients -- at which stored step and
    which channel -- each readout responded to, beside the coefficients the encoders read. Every
    readout of :data:`~teb_vae.lag_attn_cfs.eval.attributions.EXAMPLE_READOUTS` and the lag
    readout on every configured band is attributed under **both** baselines, because the target
    map exists only along the all-zero path and the source map is read along the source-null one.
    The maps and layer splits the main pass already took at this very row arrive in ``maps`` and
    ``layer`` and are not taken again: every attribution here is deterministic, so a second call
    would return the same arrays at the cost of a full integration.

    Args:
        task: The loaded task.
        inputs: The batch's ``(y_st, y_ph, u_stream)``, unexpanded.
        extra: The batch's ``(target_features, weight)``, unexpanded.
        cell: The cell binding.
        sample: Which batch element the example is.
        column: The anchor's position on the anchor axis.
        anchor: The anchor as a stored step.
        identity: The sample's identity columns.
        model_profile: The model's own lag readout at the anchor, $(L,)$.
        lag_bands: The configured lag bands.
        n_steps: Integration steps.
        likelihood: The objective's likelihood, for the block-score readouts.
        live: Per stream, each declared channel's first live step.
        horizons: The named horizon steps the per-step score is attributed at, or ``None``.
        latent: The latent at the anchor -- ``mu_prior``, ``shift`` ($\mu^q - \mu^p$) and
            ``kld_dim``, each $(d_z,)$ -- for the page's activation row, or ``None``.
        maps: The maps already taken at this row, keyed as the returned ``maps`` are, or ``None``.
        layer: The layer splits already taken at this row, keyed as the returned ``layer`` is, or
            ``None``.
        raw: The segment's raw signals in physical units (``fhr``, ``up``), or ``None``.
        raw_units: Their units, or ``None``.

    Returns:
        ``(example, forward_equivalents)``: the example record -- identity, ``anchor``, ``column``,
        ``horizon``, ``inputs`` (the declared target stream with its two blocks concatenated, and
        the source stream, each $(T, C)$), ``n_scattering``, the live steps, ``model_profile``,
        ``latent``, ``raw`` and ``raw_units``, ``maps`` keyed by ``(readout, baseline, band)`` with
        the two stream maps, the lag-aligned source profile and the readout at the input and at the
        exact baseline, and ``layer`` keyed by ``(readout, band)`` with the per-unit layer split
        under the source-null baseline -- and what the attributions taken here cost.
    """
    model = task.orig_model
    one_inputs = tuple(x[sample:sample + 1].detach() for x in inputs)
    one_extra = tuple(x[sample:sample + 1].detach() for x in extra)
    columns = torch.tensor([int(column)], dtype=torch.long, device=one_inputs[0].device)
    n_lags = int(np.asarray(model_profile).shape[-1])
    maps = dict(maps or {})
    layer = dict(layer or {})
    cost = 0
    for readout, tag, band, step in core.example_variants(lag_bands, horizons):
        wrapper = core.AnchorReadout(
            model, cell, readout=readout, likelihood=likelihood, lag_band=band, horizon=step
        ).eval()
        for baseline in core.BASELINES:
            if (readout, baseline, tag) in maps:
                continue
            result = core.integrated_gradients(wrapper, one_inputs, one_extra, columns, baseline=baseline, n_steps=n_steps)
            cost += int(n_steps) + 3
            maps[(readout, baseline, tag)] = example_map(result, 0, n_lags)
        # The activation split, under the source-null path along which it is complete.
        if (readout, tag) not in layer:
            split = core.layer_attribution(wrapper, one_inputs, one_extra, columns, baseline=core.BASELINE_SOURCE_NULL, n_steps=n_steps)
            if split["per_unit"].size:
                cost += int(n_steps) + 2
                layer[(readout, tag)] = np.asarray(split["per_unit"][0], dtype=np.float64)
    y_st, y_ph, u_stream = one_inputs
    example = {
        **dict(identity), "anchor": int(anchor), "column": int(column),
        "horizon": int(getattr(model, "horizon", 0) or 0),
        "inputs": {
            core.STREAM_TARGET: torch.cat([y_st, y_ph], dim=-1)[0].detach().cpu().to(torch.float32).numpy(),
            core.STREAM_SOURCE: u_stream[0].detach().cpu().to(torch.float32).numpy(),
        },
        "n_scattering": int(y_st.shape[-1]),
        "live_target": live.get(core.STREAM_TARGET), "live_source": live.get(core.STREAM_SOURCE),
        "model_profile": np.asarray(model_profile, dtype=np.float64),
        "latent": {key: np.asarray(value, dtype=np.float64) for key, value in dict(latent or {}).items()},
        "raw": dict(raw or {}),
        "raw_units": dict(raw_units or {}),
        "maps": maps,
        "layer": layer,
    }
    return example, cost


def example_map(result: core.AttributionBatch, offset: int, n_lags: int) -> Dict[str, Any]:
    """What an example page keeps of one row of one Captum call: both maps, the lag profile, the values.

    Args:
        result: The call's attributions.
        offset: The row.
        n_lags: $L$.

    Returns:
        ``{target, source, lag_profile, value_input, value_baseline}``.
    """
    source = result.source[offset:offset + 1]
    return {
        core.STREAM_TARGET: result.target[offset], core.STREAM_SOURCE: result.source[offset],
        "lag_profile": core.lag_profile(core.time_profile(source), [int(result.anchor[offset])], n_lags)[0],
        "value_input": float(result.value_input[offset]), "value_baseline": float(result.value_baseline[offset]),
    }


def attribute_batch(
    task: Any,
    batch: Any,
    rows: pd.DataFrame,
    cell: core.CellBinding,
    *,
    lag_bands: Mapping[str, Tuple[int, int]],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    n_steps: int,
    anchors_per_segment: int,
    readouts: Sequence[str] = core.MAIN_READOUTS,
    baselines: Sequence[str] = core.BASELINES,
    with_layer: bool = True,
    with_ablation: bool = True,
    with_lag_readout: bool = True,
    with_top_coordinate: bool = True,
    with_horizon: bool = True,
    keep_maps_for: Optional[Dict[str, Any]] = None,
    work: Optional[BatchWork] = None,
    raw_scales: Optional[Mapping[str, Tuple[float, float]]] = None,
    gradcam_readouts: Sequence[str] = (),
) -> BatchWork:
    """Run every attribution of one batch and reduce each row to its record and vectors.

    The example anchor of a class that still wants one is chosen before the calls -- the middle
    of its sample's chosen anchors -- so each call's maps at that row are kept as they are made;
    :func:`attribute_example` then integrates only the variants no call here took.

    Args:
        task: The loaded task.
        batch: The batch, already on the device.
        rows: The batch's rows, in batch order, carrying the identity columns.
        cell: The cell binding.
        lag_bands: The configured lag bands, possibly empty.
        channel_groups: ``{stream: {band: positions}}`` from the channel map, possibly empty.
        n_steps: Integration steps.
        anchors_per_segment: Anchors chosen per sample.
        readouts: The readouts attributed under every baseline.
        baselines: The baselines.
        with_layer: Whether to take the layer split.
        with_ablation: Whether to run the band ablation.
        with_lag_readout: Whether to attribute the lag readout per band.
        with_top_coordinate: Whether to attribute the anchor's top divergence coordinate.
        with_horizon: Whether to attribute the per-step score at the named horizon steps.
        keep_maps_for: ``{class: None}`` of the classes whose example anchor is still wanted;
            filled in place with the example record as each is attributed.
        work: The accumulator to extend, or ``None`` for a fresh one.
        raw_scales: From :func:`~teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales`, for the
            example's raw signals in physical units; ``None`` keeps loader units, labelled so.
        gradcam_readouts: The readouts to take Grad-CAM of at the same anchors
            (:mod:`~teb_vae.lag_attn_cfs.eval.gradcam`); lag-attentive cells only.

    Returns:
        The accumulator.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)
    inputs = (y_st, y_ph, u_stream)
    extra = (target_features, weight)
    kwargs: Dict[str, Any] = {"anchor_phase": DENSE_ANCHOR_GEOMETRY[0], "anchor_stride": DENSE_ANCHOR_GEOMETRY[1]}
    if not cell.dense_latent:
        kwargs["return_proposals"] = True
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, **kwargs)
    contributing = core.contributing_columns(model, weight, outputs)
    work = work if work is not None else BatchWork()
    informative = "anchors" in rows.columns
    if informative:
        # The segment's high-KL clean candidates, highest K_t first; see informative_columns.
        anchor_axis, validity = outputs["anchor_index"].detach().cpu().numpy(), weight.detach().cpu().numpy()
        lag_window = int(getattr(getattr(model, "lag_attn", None), "L", 0) or int(model.max_lag) + 1)
        columns_per_sample = []
        for element, steps_wanted in enumerate(rows["anchors"]):
            chosen, unclean = core.informative_columns(
                anchor_axis[element], contributing[element], steps_wanted, validity[element],
                n_lags=lag_window, per_segment=anchors_per_segment, spacing=int(model.horizon),
            )
            columns_per_sample.append(chosen)
            work.n_unclean_anchors += unclean
    else:
        columns_per_sample = core.spread_columns(contributing, anchors_per_segment)
    identity = _identity(batch, rows)
    if work.n_scattering is None:
        work.n_scattering = int(y_st.shape[-1])
    rows_inputs, rows_extra, columns, sample = core.expand_rows(inputs, extra, columns_per_sample)
    n_rows = int(columns.shape[0])
    if n_rows == 0:
        return work
    with torch.no_grad():
        model_profile = core.model_lag_readout(
            model, cell, outputs, torch.as_tensor(sample, dtype=torch.long, device=columns.device), columns
        )
        # The anchor's largest per-coordinate divergence, for the top-coordinate readout.
        kld_dim = (
            model.kld_tensor(mu_prior=outputs["mu_prior"], logvar_prior=outputs["logvar_prior"],
                             mu_post=outputs["mu_post"], logvar_post=outputs["logvar_post"])
            if cell.dense_latent else outputs["kld_per_anchor_dim"]
        )
    n_lags = int(model_profile.shape[1])
    live = {stream: core.warm_from_step(model, stream) for stream in core.STREAMS}
    anchors_all = outputs["anchor_index"].detach().cpu().numpy()
    horizons = core.horizon_steps(model)
    source_split = core.source_block_split(model)

    def anchor_steps() -> np.ndarray:
        """Each row's anchor as a stored step."""
        return np.asarray([int(anchors_all[s, c]) for s, c in zip(sample, columns.cpu().numpy())], dtype=np.int64)

    steps = anchor_steps()
    if cell.dense_latent:
        kld_rows = kld_dim.detach().cpu().numpy()[sample, steps]
    else:
        kld_rows = kld_dim.detach().cpu().numpy()[sample, columns.cpu().numpy()]
    top_coordinate = torch.as_tensor(np.argmax(kld_rows, axis=1), dtype=torch.long, device=columns.device)

    # The example anchor of a sample: its highest-KL chosen anchor under the informative rule, the
    # middle of its spread anchors otherwise.
    example_rows: Dict[str, int] = {}
    for element in range(len(identity)) if keep_maps_for is not None else ():
        key = example_key(rows.iloc[element])
        if key is None or key not in keep_maps_for or keep_maps_for[key] is not None or key in example_rows:
            continue
        of_sample = np.flatnonzero(sample == element)
        if of_sample.size:
            example_rows[key] = int(of_sample[0] if informative else of_sample[of_sample.size // 2])
    wanted = {
        (readout, baseline, tag)
        for readout, tag, _band, _step in core.example_variants(lag_bands, horizons) for baseline in core.BASELINES
    }
    kept_maps: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {key: {} for key in example_rows}
    kept_layer: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {key: {} for key in example_rows}

    calls: List[Tuple[str, str, Optional[Tuple[int, int]], Optional[torch.Tensor], str, int]] = []
    for readout in readouts:
        for baseline in baselines:
            calls.append((readout, baseline, None, None, "", 0))
    # The per-step score at the named horizon steps, under both baselines: the target's share
    # exists only along the all-zero path, the source's is read along the source-null one.
    for step in (horizons.values() if with_horizon else ()):
        for baseline in baselines:
            calls.append((core.READOUT_NLL_HORIZON, baseline, None, None, f"h{int(step)}", int(step)))
    if with_lag_readout:
        for name, band in lag_bands.items():
            calls.append((core.READOUT_LAG_BAND, core.BASELINE_SOURCE_NULL, tuple(band), None, str(name), 0))
    if with_top_coordinate:
        calls.append((core.READOUT_KLD_DIM, core.BASELINE_SOURCE_NULL, None, top_coordinate, "", 0))

    for readout, baseline, band, coordinates, band_name, horizon in calls:
        wrapper = core.AnchorReadout(
            model, cell, readout=readout, likelihood=likelihood, lag_band=band or (0, 0), horizon=horizon
        ).eval()
        result = core.integrated_gradients(
            wrapper, rows_inputs, rows_extra, columns, baseline=baseline,
            coordinates=coordinates, n_steps=n_steps,
        )
        work.forward_equivalents += n_rows * (int(n_steps) + 3)
        _reduce_rows(
            work, result, identity=identity, sample=sample, band_name=band_name, n_lags=n_lags,
            model_profile=model_profile, lag_bands=lag_bands, channel_groups=channel_groups,
            live=live, kld_rows=kld_rows, n_scattering=int(y_st.shape[-1]), source_split=source_split,
            unit=core.readout_unit(readout, cell),
        )
        if (readout, baseline, band_name) in wanted:
            for key, offset in example_rows.items():
                kept_maps[key][(readout, baseline, band_name)] = example_map(result, offset, n_lags)
        layer = None
        if with_layer and readout in core.MAIN_READOUTS and baseline == core.BASELINE_SOURCE_NULL:
            layer = core.layer_attribution(
                wrapper, rows_inputs, rows_extra, columns, baseline=baseline, n_steps=n_steps
            )
            work.forward_equivalents += n_rows * (int(n_steps) + 2)
            if layer["per_unit"].size:
                for key, offset in example_rows.items():
                    kept_layer[key][(readout, band_name)] = np.asarray(layer["per_unit"][offset], dtype=np.float64)
        ablation: Optional[Dict[str, np.ndarray]] = None
        if with_ablation and readout in core.MAIN_READOUTS and baseline == core.BASELINE_SOURCE_NULL and lag_bands:
            ablation = core.ablate_lag_bands(
                wrapper, rows_inputs, rows_extra, columns, steps, lag_bands
            )
            work.forward_equivalents += n_rows * (len(lag_bands) + 2)
        start = len(work.rows) - n_rows
        for offset in range(n_rows):
            record = work.rows[start + offset]
            if layer is not None and layer["per_unit"].size:
                record["layer_total"] = float(layer["total"][offset])
                per_unit = layer["per_unit"][offset]
            else:
                per_unit = np.full(0, np.nan)
            work.append_vector("layer_per_unit", per_unit)
            if ablation is not None:
                for name, values in ablation.items():
                    record[f"ablation_{name}"] = float(values[offset])
    # Grad-CAM at the same anchors: one forward and one backward per readout.
    for readout in (gradcam_readouts if cell.dense_latent else ()):
        wrapper = core.AnchorReadout(model, cell, readout=readout, likelihood=likelihood).eval()
        cams = gradcam.gradcam(wrapper, rows_inputs, rows_extra, columns)
        work.forward_equivalents += 2 * n_rows
        for offset in range(n_rows):
            work.gradcam_rows.append({
                **identity[int(sample[offset])], "anchor": int(steps[offset]), "readout": readout,
                "value": float(cams["value"][offset]),
                **{f"{view}_total": float(cams[f"{view}_total"][offset]) for view in gradcam.VIEWS},
            })
            for view in gradcam.VIEWS:
                work.gradcam_vectors.setdefault(view, []).append(cams[view][offset].astype(np.float32))
    if not example_rows:
        return work
    # The example anchors, their full maps kept for the map figures, beside the raw signals of
    # their segments in physical units -- attached through the traces' own path, so a gap is a
    # gap here exactly as it is on a trace.
    holders = [
        traces.SegmentTrace(
            guid=str(who["guid"]), epoch=float(who["epoch"]), clinical_class=None, subgroup=None,
            anchor=np.zeros(0, dtype=np.int64), contributing=np.zeros(0, dtype=bool),
        )
        for who in identity
    ]
    traces.attach_raw_signals(holders, batch, raw_scales)
    for key, offset in example_rows.items():
        element = int(sample[offset])
        where = (element, int(steps[offset])) if cell.dense_latent else (element, int(columns[offset].item()))
        with torch.no_grad():
            mu_prior = outputs["mu_prior"][where].detach().cpu().to(torch.float64).numpy()
            mu_post = outputs["mu_post"][where].detach().cpu().to(torch.float64).numpy()
        latent = {"mu_prior": mu_prior, "shift": mu_post - mu_prior, "kld_dim": np.asarray(kld_rows[offset], dtype=np.float64)}
        example, cost = attribute_example(
            task, inputs, extra, cell, sample=element, column=int(columns[offset].item()),
            anchor=int(steps[offset]), identity=identity[element], model_profile=model_profile[offset],
            lag_bands=lag_bands, n_steps=n_steps, likelihood=likelihood, live=live,
            horizons=horizons, latent=latent, maps=kept_maps[key], layer=kept_layer[key],
            raw=holders[element].raw, raw_units=holders[element].raw_units,
        )
        work.forward_equivalents += cost
        example["example_rank"] = int(rows.iloc[element].get("example_rank", 0) or 0)
        example["kld_rank_note"] = (
            "highest-KL clean anchor of the class" if informative else "middle spread anchor of the first segment of the class"
        )
        keep_maps_for[key] = example
        work.examples[key] = example
    return work


def example_key(row: pd.Series) -> Optional[str]:
    """The example-page key of a selected segment, or ``None`` when it gives no example page.

    Under the informative rule a segment is an example when it carries an ``example_rank``, and the
    key is ``<class>:<rank>``. Without it the key is the class, so the first segment of each class
    gives the one example.
    """
    if "example_rank" not in row.index:
        return str(row.get(labels.CLASS_COLUMN))
    rank = row["example_rank"]
    return None if rank is None or not np.isfinite(float(rank)) else f"{row[labels.CLASS_COLUMN]}:{int(rank)}"


def _reduce_rows(
    work: BatchWork,
    result: core.AttributionBatch,
    *,
    identity: Sequence[Dict[str, Any]],
    sample: np.ndarray,
    band_name: str,
    n_lags: int,
    model_profile: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    live: Mapping[str, Optional[np.ndarray]],
    kld_rows: np.ndarray,
    n_scattering: int = 0,
    source_split: int = 0,
    unit: str = "readout units",
) -> None:
    """Turn one Captum call's maps into records and row-aligned vectors.

    Every record carries the row's identity (``guid``, ``epoch``, class, subgroup, ``anchor``),
    what was attributed (``readout``, ``baseline``, ``band``) and the ``unit`` of the readout,
    which is also the unit of every attribution column beside it.
    """
    n_rows = int(result.anchor.shape[0])
    target_time = core.time_profile(result.target)
    source_time = core.time_profile(result.source)
    blocks = core.block_sums(result.target, result.source, n_scattering=n_scattering, source_split=source_split)
    target_channels = core.channel_profile(result.target)
    source_channels = core.channel_profile(result.source)
    lags = core.lag_profile(source_time, result.anchor, n_lags)
    target_lags = core.lag_profile(target_time, result.anchor, n_lags)
    fit = core.agreement(lags, model_profile)
    # The population lag-by-channel maps, for the main readouts under both baselines.
    if result.readout in core.MAIN_READOUTS and n_rows:
        row_class = np.asarray([str(identity[int(s)][labels.CLASS_COLUMN]) for s in sample])
        for stream, maps in ((core.STREAM_TARGET, result.target), (core.STREAM_SOURCE, result.source)):
            aligned = core.offset_channel_map(maps, result.anchor, n_lags)
            work.accumulate_lag_channel((result.readout, result.baseline, stream), aligned)
            for name in np.unique(row_class):
                work.accumulate_lag_channel(
                    (result.readout, result.baseline, stream, name), aligned[row_class == name],
                    store=work.lag_channel_by_class,
                )
    lag_groups = core.lag_band_groups(lag_bands, n_lags) if lag_bands else {}
    lag_band_values = core.band_sums(lags, lag_groups) if lag_groups else {}
    band_values = {
        stream: core.band_sums(target_channels if stream == core.STREAM_TARGET else source_channels, groups)
        for stream, groups in channel_groups.items()
    }
    steps = np.arange(result.target.shape[1])
    for offset in range(n_rows):
        anchor = int(result.anchor[offset])
        after = steps > anchor
        gated = 0.0
        source_live = live.get(core.STREAM_SOURCE)
        if source_live is not None:
            cold = steps[:, None] < source_live[None, :]
            gated = float(np.abs(result.source[offset])[cold].max()) if cold.any() else 0.0
        who = identity[int(sample[offset])]
        record: Dict[str, Any] = {
            **who,
            "anchor": anchor,
            "column": int(result.column[offset]),
            "readout": result.readout,
            "baseline": result.baseline,
            "band": band_name,
            "unit": unit,
            "coordinate": int(result.coordinate[offset]),
            "value_input": float(result.value_input[offset]),
            "value_baseline": float(result.value_baseline[offset]),
            "value_entry": float(result.value_entry[offset]),
            "entry_jump": float(result.value_entry[offset] - result.value_baseline[offset]),
            "attributed": float(target_time[offset].sum() + source_time[offset].sum()),
            "ig_delta": float(result.delta[offset]),
            # Against the larger of the readout's move along the path and the readout itself:
            # a source that barely moves a readout would otherwise report a residual of the
            # order of one over a difference of the order of nothing.
            "completeness_rel": float(
                abs(result.delta[offset]) / max(
                    abs(result.value_input[offset] - result.value_entry[offset]),
                    abs(result.value_input[offset]), 1e-12,
                )
            ),
            "target_total": float(target_time[offset].sum()),
            "source_total": float(source_time[offset].sum()),
            "target_abs_total": float(np.abs(result.target[offset]).sum()),
            "source_abs_total": float(np.abs(result.source[offset]).sum()),
            "after_anchor_max_abs": float(max(
                np.abs(result.target[offset][after]).max() if after.any() else 0.0,
                np.abs(result.source[offset][after]).max() if after.any() else 0.0,
            )),
            "gated_off_max_abs": gated,
            "lag_corr": float(fit["lag_corr"][offset]),
            "lag_js": float(fit["lag_js"][offset]),
            "kld_top_coordinate": int(np.argmax(kld_rows[offset])),
        }
        for name, values in lag_band_values.items():
            record[f"lagband_{name}"] = float(values[offset])
        for stream, bands in band_values.items():
            for name, values in bands.items():
                record[f"band_{stream}_{name}"] = float(values[offset])
        for name, (signed, unsigned) in blocks.items():
            record[f"block_{name}"] = float(signed[offset])
            record[f"block_abs_{name}"] = float(unsigned[offset])
        work.rows.append(record)
        work.append_vector("time_profile_target", target_time[offset].astype(np.float32))
        work.append_vector("time_profile_source", source_time[offset].astype(np.float32))
        work.append_vector("lag_profile", lags[offset].astype(np.float32))
        work.append_vector("target_lag_profile", target_lags[offset].astype(np.float32))
        work.append_vector("model_profile", model_profile[offset].astype(np.float32))
        work.append_vector("channel_profile_target", target_channels[offset].astype(np.float32))
        work.append_vector("channel_profile_source", source_channels[offset].astype(np.float32))
        # The unsigned channel profiles, which the signed ones cannot recover once a channel's
        # contributions have cancelled over time.
        work.append_vector("channel_abs_profile_target", np.abs(result.target[offset]).sum(axis=0).astype(np.float32))
        work.append_vector("channel_abs_profile_source", np.abs(result.source[offset]).sum(axis=0).astype(np.float32))


# =============================================================================
# The pass over the selected segments
# =============================================================================
def run_segments(
    task: Any,
    loader: Any,
    selected: pd.DataFrame,
    cell: core.CellBinding,
    *,
    lag_bands: Mapping[str, Tuple[int, int]],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    n_steps: int,
    anchors_per_segment: int,
    raw_scales: Optional[Mapping[str, Tuple[float, float]]] = None,
    examples: bool = True,
    **options: Any,
) -> Tuple[BatchWork, int]:
    """Re-read the selected segments in dataset order and attribute every one.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        selected: From :func:`select_segments`.
        cell: The cell binding.
        lag_bands: The configured lag bands.
        channel_groups: From the channel map.
        n_steps: Integration steps.
        anchors_per_segment: Anchors per segment.
        raw_scales: The raw signals' physical scales, for the example pages.
        examples: Whether to keep one example anchor per class for the map pages.
        **options: Passed to :func:`attribute_batch`: the readouts, the baselines, the ``with_*``
            switches and ``gradcam_readouts``.

    Returns:
        ``(work, n_batches)``.
    """
    work = BatchWork()
    if selected.empty:
        return work, 0
    classes = (
        {key: None for key in (example_key(row) for _, row in selected.iterrows()) if key is not None}
        if examples else None
    )
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))
    segments = subset_loader(loader, list(selected["dataset_index"]), batch_size=batch_size)
    position = 0
    n_batches = 0
    for batch in segments:
        guids = batch_field(batch, "guid")
        count = len(guids) if isinstance(guids, (list, tuple)) else 1
        batch_rows = selected.iloc[position:position + count]
        position += count
        check_batch_identity(batch, batch_rows)
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        attribute_batch(
            task, moved, batch_rows, cell, lag_bands=lag_bands, channel_groups=channel_groups,
            n_steps=n_steps, anchors_per_segment=anchors_per_segment, keep_maps_for=classes, work=work,
            raw_scales=raw_scales, **options,
        )
        n_batches += 1
    return work, n_batches


def target_only_check(task: Any, loader: Any, selected: pd.DataFrame, cell: core.CellBinding, *, n_steps: int) -> Dict[str, Any]:
    """Attribute a target-only readout on the first selected segment and report its source attribution.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        selected: From :func:`select_segments`.
        cell: The cell binding.
        n_steps: Integration steps.

    Returns:
        ``{'readout', 'source_attr_max_abs', 'n_rows'}``, or a recorded absence.
    """
    if selected.empty:
        return {"readout": core.READOUT_NLL_BASE, "source_attr_max_abs": None, "n_rows": 0}
    first = subset_loader(loader, [int(selected["dataset_index"].iloc[0])], batch_size=1)
    batch = next(iter(first))
    moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
    work = attribute_batch(
        task, moved, selected.iloc[:1], cell, lag_bands={}, channel_groups={}, n_steps=n_steps,
        anchors_per_segment=1, readouts=(core.READOUT_NLL_BASE,), baselines=(core.BASELINE_SOURCE_NULL,),
        with_layer=False, with_ablation=False, with_lag_readout=False, with_top_coordinate=False,
        with_horizon=False,
    )
    worst = max(
        (float(row["source_abs_total"]) for row in work.rows if row["readout"] == core.READOUT_NLL_BASE), default=None
    )
    return {"readout": core.READOUT_NLL_BASE, "source_attr_max_abs": worst, "n_rows": len(work.rows)}


# =============================================================================
# Tables
# =============================================================================
def rows_frame(work: BatchWork) -> pd.DataFrame:
    """The per-row table."""
    return pd.DataFrame(work.rows)


def stack_vectors(work: BatchWork) -> Dict[str, np.ndarray]:
    """Stack the row-aligned vectors, padding the layer split to its widest row."""
    stacked: Dict[str, np.ndarray] = {}
    for name, pieces in work.vectors.items():
        if not pieces:
            continue
        width = max(int(np.asarray(p).shape[-1]) if np.asarray(p).ndim else 0 for p in pieces)
        matrix = np.full((len(pieces), width), np.nan, dtype=np.float32)
        for row, piece in enumerate(pieces):
            values = np.asarray(piece, dtype=np.float32).reshape(-1)
            matrix[row, :values.size] = values
        stacked[name] = matrix
    return stacked


def recordings_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per recording: the main readouts' totals and agreement under the source-null baseline.

    The readouts are joined anchor by anchor, averaged over each segment's anchors and then over
    the recording's segments -- the family's aggregation chain -- so ``n_segments`` counts
    segments.

    Args:
        rows: The per-row table.

    Returns:
        Indexed by ``guid`` with the cohort columns, ``n_segments`` and the columns of
        :data:`RECORDING_VALUE_COLUMNS`.
    """
    if rows.empty:
        return pd.DataFrame(columns=[*RECORDING_VALUE_COLUMNS, *labels.GROUP_COLUMNS, "n_segments"])
    keys = ["guid", "epoch", "anchor", *labels.GROUP_COLUMNS]
    pieces = []
    for readout in core.MAIN_READOUTS:
        subset = rows[(rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)]
        if subset.empty:
            continue
        part = subset[[*keys, "source_total", "target_total", "lag_corr", "lag_js"]].copy()
        part = part.rename(columns={name: f"{readout}_{name}" for name in ("source_total", "target_total", "lag_corr", "lag_js")})
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=[*RECORDING_VALUE_COLUMNS, *labels.GROUP_COLUMNS, "n_segments"])
    merged = pieces[0]
    for part in pieces[1:]:
        merged = merged.merge(part, on=keys, how="outer")
    values = [c for c in RECORDING_VALUE_COLUMNS if c in merged.columns]
    per_segment = merged.groupby(["guid", "epoch"], as_index=False, dropna=False).agg(
        {**{name: "mean" for name in values}, **{name: "first" for name in labels.GROUP_COLUMNS}}
    )
    return frames.per_recording_means(per_segment, values)


def _mean_over_recordings(rows: pd.DataFrame, column: str) -> Tuple[float, int]:
    """Mean of a column per recording, then over recordings; and the recording count."""
    if rows.empty or column not in rows.columns:
        return float("nan"), 0
    per = rows.groupby("guid")[column].mean()
    finite = per[np.isfinite(per)]
    return (float(finite.mean()) if len(finite) else float("nan")), int(len(finite))


def summary_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (readout, baseline, band): recording-mean values, totals, agreement and the checks."""
    records: List[Dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame(records)
    for (readout, baseline, band), subset in rows.groupby(["readout", "baseline", "band"], sort=False):
        record: Dict[str, Any] = {
            "readout": readout, "baseline": baseline, "band": band,
            "unit": str(subset["unit"].iloc[0]) if "unit" in subset.columns else "readout units",
            "n_rows": int(len(subset)), "n_segments": int(subset[["guid", "epoch"]].drop_duplicates().shape[0]),
            "n_recordings": int(subset["guid"].nunique()),
        }
        for column in ("value_input", "value_baseline", "value_entry", "entry_jump", "attributed",
                       "target_total", "source_total", "target_abs_total", "source_abs_total", "lag_corr", "lag_js"):
            record[f"{column}_mean"], _ = _mean_over_recordings(subset, column)
        record["completeness_rel_max"] = float(np.nanmax(subset["completeness_rel"])) if subset["completeness_rel"].notna().any() else float("nan")
        record["completeness_rel_median"] = float(np.nanmedian(subset["completeness_rel"])) if subset["completeness_rel"].notna().any() else float("nan")
        record["after_anchor_max_abs"] = float(subset["after_anchor_max_abs"].max())
        record["gated_off_max_abs"] = float(subset["gated_off_max_abs"].max())
        records.append(record)
    return pd.DataFrame(records)


def bands_frame(rows: pd.DataFrame, spectral: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per (readout, stream, frequency band): the recording-mean attribution, with the skill gap joined."""
    records: List[Dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame(records)
    band_columns = [c for c in rows.columns if c.startswith("band_")]
    skill = {}
    if spectral is not None and "band" in spectral.columns and "pred_gap_nats" in spectral.columns:
        skill = {str(row["band"]): float(row["pred_gap_nats"]) for _, row in spectral.iterrows()}
    # Each stream on the path along which it moves: the target under the all-zero baseline, the
    # source under the source-null one. The target's attribution on the source-null path is zero
    # by construction and would be a column of zeros rather than a reading.
    stream_baseline = {core.STREAM_TARGET: core.BASELINE_ALL_ZERO, core.STREAM_SOURCE: core.BASELINE_SOURCE_NULL}
    for readout in core.MAIN_READOUTS:
        for column in band_columns:
            _, stream, name = column.split("_", 2)
            baseline = stream_baseline.get(stream, core.BASELINE_SOURCE_NULL)
            subset = rows[(rows["readout"] == readout) & (rows["baseline"] == baseline)]
            if subset.empty:
                continue
            mean, count = _mean_over_recordings(subset, column)
            records.append(
                {
                    "readout": readout, "stream": stream, "baseline": baseline, "band": name,
                    "attribution_mean": mean, "n_recordings": count,
                    "spectral_skill_pred_gap_nats": skill.get(name, float("nan")) if stream == core.STREAM_TARGET else float("nan"),
                    "unit": core.READOUT_UNITS[readout],
                }
            )
    return pd.DataFrame(records)


def lag_bands_frame(rows: pd.DataFrame, lag_bands: Mapping[str, Tuple[int, int]], occlusion: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per (readout, lag band): the IG sum, the ablation delta and the occlusion delta where present.

    The occlusion delta is that pass's mean over recordings of the horizon-summed change in the
    full-branch block score when the band is zeroed -- the same aggregation as the two columns
    beside it -- read from its pooled column only on a summary that carries no recording mean.
    It is the change of ``nll_full`` itself and minus the change of ``pred_gap``, whose base
    branch reads no source; the divergence and the squared error it does not score.
    """
    records: List[Dict[str, Any]] = []
    if rows.empty or not lag_bands:
        return pd.DataFrame(records)
    occluded: Dict[str, float] = {}
    if occlusion is not None and "band" in occlusion.columns:
        column = next(
            (name for name in ("delta_total_recording_mean_nats", "delta_total_nats") if name in occlusion.columns), None
        )
        if column is not None:
            occluded = {str(row["band"]): float(row[column]) for _, row in occlusion.iterrows()}
    sign = {core.READOUT_NLL_FULL: 1.0, core.READOUT_PRED_GAP: -1.0}
    for readout in core.MAIN_READOUTS:
        subset = rows[(rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)]
        if subset.empty:
            continue
        for name, (low, high) in lag_bands.items():
            ig_mean, count = _mean_over_recordings(subset, f"lagband_{name}")
            ablation_mean, _ = _mean_over_recordings(subset, f"ablation_{name}")
            delta = occluded.get(str(name), float("nan"))
            records.append(
                {
                    "readout": readout, "band": name, "lag_lo": int(low), "lag_hi": int(high),
                    "ig_attribution_mean": ig_mean, "ablation_delta_mean": ablation_mean,
                    "occlusion_delta_total_nats": sign[readout] * delta if readout in sign else float("nan"),
                    "n_recordings": count,
                    "unit": core.READOUT_UNITS[readout],
                }
            )
    return pd.DataFrame(records)


def blocks_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (readout, band, baseline, block): the recording-mean signed and unsigned sums and the unsigned share.

    Every readout but the per-coordinate and the lag-band ones, so the table answers which of
    the four input blocks the latent change, the gain, the score, the fidelity and the per-step
    score each responded to. The share is per row -- the block's unsigned sum over the four --
    then averaged per recording and over recordings, so a segment with a large readout does not
    decide the population's shares.
    """
    records: List[Dict[str, Any]] = []
    columns = [f"block_abs_{name}" for name in core.BLOCKS]
    if rows.empty or not all(column in rows.columns for column in columns):
        return pd.DataFrame(records)
    keep = ~rows["readout"].isin([core.READOUT_KLD_DIM, core.READOUT_LAG_BAND])
    total = rows.loc[keep, columns].sum(axis=1)
    for (readout, band, baseline), subset in rows[keep].groupby(["readout", "band", "baseline"], sort=False):
        shares = subset[columns].div(total.loc[subset.index].replace(0.0, np.nan), axis=0)
        for name in core.BLOCKS:
            signed, count = _mean_over_recordings(subset, f"block_{name}")
            unsigned, _ = _mean_over_recordings(subset, f"block_abs_{name}")
            share, _ = _mean_over_recordings(shares.assign(guid=subset["guid"]), f"block_abs_{name}")
            records.append(
                {
                    "readout": readout, "band": band, "baseline": baseline, "block": name,
                    "label": core.variant_label(str(readout), str(band)),
                    "signed_mean": signed, "unsigned_mean": unsigned, "share_mean": share, "n_recordings": count,
                }
            )
    return pd.DataFrame(records)


def layer_frame(rows: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Long-form: per (readout, unit, class or pooled) the recording-mean layer attribution."""
    records: List[Dict[str, Any]] = []
    if rows.empty or "layer_per_unit" not in vectors or vectors["layer_per_unit"].shape[1] == 0:
        return pd.DataFrame(records)
    matrix = vectors["layer_per_unit"]
    for readout in core.MAIN_READOUTS:
        keep = ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy()
        keep = keep & np.isfinite(matrix).any(axis=1)
        if not keep.any():
            continue
        subset = rows[keep]
        values = matrix[keep]
        groups = [("pooled", np.ones(len(subset), dtype=bool))]
        for name in labels.ordered_groups(list(subset[labels.CLASS_COLUMN].dropna().unique()), labels.CLASS_COLUMN):
            groups.append((name, (subset[labels.CLASS_COLUMN].astype(object) == name).to_numpy()))
        for name, mask in groups:
            guids = subset["guid"].to_numpy()[mask]
            per_recording = np.stack(
                [np.nanmean(values[mask][guids == guid], axis=0) for guid in np.unique(guids)], axis=0
            ) if mask.any() else np.zeros((0, values.shape[1]))
            mean = np.nanmean(per_recording, axis=0) if len(per_recording) else np.full(values.shape[1], np.nan)
            for unit in range(values.shape[1]):
                records.append({"readout": readout, "unit": int(unit), labels.CLASS_COLUMN: name,
                                "n_recordings": int(len(np.unique(guids))), "mean": float(mean[unit])})
    return pd.DataFrame(records)


def null_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (readout, baseline, class): the recording-mean values behind the null decomposition."""
    records: List[Dict[str, Any]] = []
    if rows.empty:
        return pd.DataFrame(records)
    for readout in core.MAIN_READOUTS:
        for baseline in core.BASELINES:
            subset = rows[(rows["readout"] == readout) & (rows["baseline"] == baseline)]
            if subset.empty:
                continue
            classes = labels.ordered_groups(list(subset[labels.CLASS_COLUMN].dropna().unique()), labels.CLASS_COLUMN)
            for name in ["pooled", *classes]:
                part = subset if name == "pooled" else subset[subset[labels.CLASS_COLUMN].astype(object) == name]
                record: Dict[str, Any] = {"readout": readout, "baseline": baseline, labels.CLASS_COLUMN: name,
                                          "n_recordings": int(part["guid"].nunique())}
                for column in ("value_input", "value_baseline", "value_entry", "entry_jump", "attributed", "target_total", "source_total"):
                    record[f"{column}_mean"], _ = _mean_over_recordings(part, column)
                records.append(record)
    return pd.DataFrame(records)


# =============================================================================
# The trace
# =============================================================================
def trace_recording(
    task: Any,
    loader: Any,
    rows: pd.DataFrame,
    cell: core.CellBinding,
    *,
    clinical_class: Optional[str],
    subgroup: Optional[str],
    n_steps: int,
    anchors_per_segment: int,
    raw_scales: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> List[traces.SegmentTrace]:
    """Attribute the trace readouts at the chosen anchors of every segment of one recording.

    Every readout of :data:`~teb_vae.lag_attn_cfs.eval.attributions.TRACE_READOUTS` is attributed
    at the same anchors under the source-null baseline, and each lands on the trace under its own
    prefix -- ``kld_value_input``, ``pred_gap_source_total`` -- beside one model lag map, so the
    latent change and the forecast gain are read against each other anchor by anchor. Each
    segment carries its raw signals, which the trace figure draws on its top rows.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        rows: From :func:`recording_rows`.
        cell: The cell binding.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.
        n_steps: Integration steps.
        anchors_per_segment: Anchors per segment.
        raw_scales: The raw signals' physical scales, or ``None`` for loader units.

    Returns:
        One trace per segment that scored an anchor, in dataset order.
    """
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))
    segments = subset_loader(loader, list(rows["dataset_index"]), batch_size=batch_size)
    gathered: List[traces.SegmentTrace] = []
    position = 0
    scalar_names = ("value_input", "value_baseline", "source_total", "target_total", "lag_corr", "lag_js")
    for batch in segments:
        guids = batch_field(batch, "guid")
        count = len(guids) if isinstance(guids, (list, tuple)) else 1
        batch_rows = rows.iloc[position:position + count].copy()
        position += count
        check_batch_identity(batch, batch_rows)
        batch_rows[labels.CLASS_COLUMN] = clinical_class
        batch_rows[labels.SUBGROUP_COLUMN] = subgroup
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        work = attribute_batch(
            task, moved, batch_rows, cell, lag_bands={}, channel_groups={}, n_steps=n_steps,
            anchors_per_segment=anchors_per_segment, readouts=core.TRACE_READOUTS,
            baselines=(core.BASELINE_SOURCE_NULL,), with_layer=False, with_ablation=False,
            with_lag_readout=False, with_top_coordinate=False, with_horizon=False,
        )
        table = rows_frame(work)
        vectors = stack_vectors(work)
        # One trace per batch sample, in batch order, so the raw signals attach by position; the
        # samples that scored no anchor are dropped after.
        batch_traces: List[traces.SegmentTrace] = []
        for _, row in batch_rows.iterrows():
            of_segment = (table["epoch"] == float(row["epoch"])).to_numpy() if len(table) else np.zeros(0, dtype=bool)
            parts = {
                readout: of_segment & (table["readout"].astype(str) == readout).to_numpy()
                for readout in core.TRACE_READOUTS
            }
            lead = parts[core.TRACE_READOUTS[0]]
            if not lead.any():
                batch_traces.append(traces.SegmentTrace(
                    guid=str(row["guid"]), epoch=float(row["epoch"]), clinical_class=clinical_class,
                    subgroup=subgroup, anchor=np.zeros(0, dtype=np.int64), contributing=np.zeros(0, dtype=bool),
                ))
                continue
            anchors = table[lead]["anchor"].to_numpy().astype(np.int64)
            scalars: Dict[str, np.ndarray] = {}
            trace_vectors: Dict[str, np.ndarray] = {
                "model_lag_map": vectors["model_profile"][lead].astype(np.float64),
            }
            for readout, keep in parts.items():
                part = table[keep]
                # Every readout was attributed at the same anchors in the same order; a
                # mismatch would put one readout's values under another's anchors.
                if not np.array_equal(part["anchor"].to_numpy().astype(np.int64), anchors):
                    raise RuntimeError(
                        f"the {readout!r} rows of segment {row['epoch']} do not share the anchors of "
                        f"{core.TRACE_READOUTS[0]!r}"
                    )
                for name in scalar_names:
                    scalars[f"{readout}_{name}"] = part[name].to_numpy().astype(np.float64)
                trace_vectors[f"{readout}_attribution_lag_map"] = np.abs(vectors["lag_profile"][keep]).astype(np.float64)
            batch_traces.append(
                traces.SegmentTrace(
                    guid=str(row["guid"]), epoch=float(row["epoch"]),
                    clinical_class=clinical_class, subgroup=subgroup,
                    anchor=anchors, contributing=np.ones(len(anchors), dtype=bool),
                    scalars=scalars, vectors=trace_vectors,
                )
            )
        traces.attach_raw_signals(batch_traces, moved, raw_scales)
        gathered.extend(segment for segment in batch_traces if len(segment.anchor))
    return gathered


def lag_channel_summary(
    work: BatchWork, store: Optional[Mapping[Tuple[str, ...], Mapping[str, np.ndarray]]] = None
) -> Dict[Tuple[str, ...], Dict[str, Any]]:
    """Reduce the running lag-by-channel sums to means, ``NaN`` where no row reached a cell.

    Args:
        work: The accumulator.
        store: Which running sums to reduce: ``work.lag_channel`` by default, or
            ``work.lag_channel_by_class``.

    Returns:
        ``{key: {'mean': (L, C), 'mean_abs': (L, C), 'n_rows': int}}`` with the store's keys:
        ``(readout, baseline, stream)``, and the class appended for the per-class store.
    """
    summary: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for key, entry in (work.lag_channel if store is None else store).items():
        count = np.asarray(entry["count"], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0.0, entry["sum"] / count, np.nan)
            mean_abs = np.where(count > 0.0, entry["abs_sum"] / count, np.nan)
        summary[key] = {"mean": mean, "mean_abs": mean_abs, "n_rows": int(count.max()) if count.size else 0}
    return summary


def lag_channel_arrays(summary: Mapping[Tuple[str, str, str], Mapping[str, Any]], lag_seconds: np.ndarray) -> Dict[str, np.ndarray]:
    """Flatten the lag-by-channel means into arrays for the ``npz``, one entry per key."""
    arrays: Dict[str, np.ndarray] = {"lag_seconds": np.asarray(lag_seconds, dtype=np.float64)}
    for (readout, baseline, stream), entry in summary.items():
        stem = f"{readout}__{baseline}__{stream}"
        arrays[f"{stem}__mean"] = np.asarray(entry["mean"], dtype=np.float32)
        arrays[f"{stem}__mean_abs"] = np.asarray(entry["mean_abs"], dtype=np.float32)
        arrays[f"{stem}__n_rows"] = np.asarray(int(entry["n_rows"]), dtype=np.int64)
    return arrays


def gradcam_frame(work: BatchWork, lag_seconds: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """The Grad-CAM rows with each view's centroid, and their row-aligned profiles.

    Args:
        work: The accumulator.
        lag_seconds: The lag axis of the ``source`` and ``attention`` views.

    Returns:
        ``(rows, vectors)``: one row per anchor and readout with ``<view>_total`` and
        ``<view>_centroid_s``, and ``{view: (N, W)}``.
    """
    rows = pd.DataFrame(work.gradcam_rows)
    vectors = {view: np.stack(work.gradcam_vectors[view]) for view in gradcam.VIEWS if work.gradcam_vectors.get(view)}
    for view, matrix in vectors.items():
        axis = gradcam.offset_seconds(matrix.shape[1]) if view == "target" else np.asarray(lag_seconds)[:matrix.shape[1]]
        rows[f"{view}_centroid_s"] = gradcam.centroid(matrix.astype(np.float64), axis)
    return rows, vectors


def run_cohort(
    task: Any,
    loader: Any,
    labelled: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    cell: core.CellBinding,
    *,
    cap: int,
    seed: int,
    directory: Path,
    lag_seconds: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    n_steps: int,
    caveat: str,
    candidates: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """The cohort pass: a larger class-balanced draw, attributed lightly, for the class comparison.

    The same selection rule as the main pass (one middle segment per recording, class-balanced,
    seeded) at its own cap. Per segment it runs only what the class comparison reads: the
    integrated gradients of :data:`~.class_contrast.COHORT_READOUTS` under ``source_null`` and the
    Grad-CAM of :data:`~.class_contrast.GRADCAM_READOUTS`, with no layer split, ablation, lag-band
    readout, horizon readout or example pages. That is about a fifth of a main-pass segment's cost.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        labelled: The labelled segments.
        index_map: The dataset listing.
        cell: The cell binding.
        cap: The cohort segment cap.
        seed: The draw and bootstrap seed.
        directory: The attribution directory.
        lag_seconds: The lag axis.
        lag_bands: The configured lag bands.
        channel_groups: From the channel map.
        n_steps: Integration steps.
        caveat: The one-line figure note.
        candidates: The high-KL clean anchors from :func:`informative_anchors`, or ``None`` for
            evenly spread anchors.

    Returns:
        The ``class_contrast`` block: the plan, the selection, the cost, the class counts, the
        significant metrics and the files.
    """
    selected, accounting = select_segments(labelled, index_map, cap=cap, seed=seed, candidates=candidates)
    gradcam_readouts = class_contrast.GRADCAM_READOUTS if cell.dense_latent else ()
    started = time.perf_counter()
    work, _ = run_segments(
        task, loader, selected, cell, lag_bands=lag_bands, channel_groups=channel_groups, n_steps=n_steps,
        anchors_per_segment=core.ANCHORS_PER_SEGMENT, examples=False,
        readouts=class_contrast.COHORT_READOUTS, baselines=(core.BASELINE_SOURCE_NULL,),
        with_layer=False, with_ablation=False, with_lag_readout=False, with_top_coordinate=False, with_horizon=False,
        gradcam_readouts=gradcam_readouts,
    )
    elapsed = time.perf_counter() - started
    rows, vectors = rows_frame(work), stack_vectors(work)
    rows.to_csv(directory / class_contrast.COHORT_ROWS_FILENAME, index=False)
    np.savez_compressed(
        directory / class_contrast.COHORT_VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64),
        **row_identity(rows), **vectors,
    )
    files = [class_contrast.COHORT_ROWS_FILENAME, class_contrast.COHORT_VECTORS_FILENAME]
    cam_rows, cam_vectors = gradcam_frame(work, lag_seconds)
    if len(cam_rows):
        cam_rows.to_csv(directory / class_contrast.GRADCAM_ROWS_FILENAME, index=False)
        np.savez_compressed(
            directory / class_contrast.GRADCAM_VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64),
            **row_identity(cam_rows), **cam_vectors,
        )
        files += [class_contrast.GRADCAM_ROWS_FILENAME, class_contrast.GRADCAM_VECTORS_FILENAME]
    block = class_contrast.run_class_contrast(
        rows, vectors, cam_rows, cam_vectors, lag_channel_summary(work, work.lag_channel_by_class),
        directory=directory, lag_seconds=lag_seconds, lag_bands=lag_bands, seed=seed, caveat=caveat,
    )
    block["files"] = files + block["files"]
    block["plan"] = {
        "cap": int(cap), "seed": int(seed), "anchors_per_segment": core.ANCHORS_PER_SEGMENT, "ig_steps": int(n_steps),
        "ig_readouts": list(class_contrast.COHORT_READOUTS), "ig_baselines": [core.BASELINE_SOURCE_NULL],
        "gradcam_readouts": list(gradcam_readouts), "gradcam_views": list(gradcam.VIEWS) if gradcam_readouts else [],
        "gradcam_target_layer": gradcam.target_layer(task.orig_model)[1] if gradcam_readouts else None,
        "anchor_rule": "high_kl_clean_per_segment" if candidates is not None else "spread",
        "n_unclean_anchors": int(work.n_unclean_anchors),
    }
    block["selection"] = accounting
    block["cost"] = core.cost_record(
        elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
        n_forward_equivalents=int(work.forward_equivalents), device=getattr(task, "device", None),
    )
    logger.info(
        f"{core.ANALYSIS_DIRNAME}: cohort pass attributed {len(selected)} segment(s) at {len(rows)} row(s) "
        f"in {elapsed:.1f} s; {len(block['significant_metrics'])} of {block['n_metrics_tested']} class "
        f"metric(s) differ after Holm"
    )
    return block


def example_arrays(examples: Sequence[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Flatten the example records into the arrays the maps file carries.

    Per example: identity, anchor, the two input streams, the block width and the live steps;
    per kept map: which example it belongs to, its readout, baseline and band, the two stream
    maps and the readout's values. Flat rather than nested because ``npz`` holds arrays, and one
    ``map_example`` index is enough to put every map back under its anchor.

    Args:
        examples: The example records, in class order.

    Returns:
        The arrays, ready for ``np.savez_compressed``.
    """
    arrays: Dict[str, np.ndarray] = {
        "example_guid": np.asarray([str(e["guid"]) for e in examples], dtype=object),
        "example_subgroup": np.asarray([str(e[labels.SUBGROUP_COLUMN]) for e in examples], dtype=object),
        "example_clinical_class": np.asarray([str(e[labels.CLASS_COLUMN]) for e in examples], dtype=object),
        "example_epoch": np.asarray([float(e["epoch"]) for e in examples], dtype=np.float64),
        "example_anchor": np.asarray([int(e["anchor"]) for e in examples], dtype=np.int64),
        "example_horizon": np.asarray([int(e.get("horizon", 0) or 0) for e in examples], dtype=np.int64),
        "example_n_scattering": np.asarray([int(e["n_scattering"]) for e in examples], dtype=np.int64),
        "input_target": np.stack([e["inputs"][core.STREAM_TARGET] for e in examples], axis=0),
        "input_source": np.stack([e["inputs"][core.STREAM_SOURCE] for e in examples], axis=0),
        "model_profile": np.stack([np.asarray(e["model_profile"], dtype=np.float64) for e in examples], axis=0),
    }
    for stream in core.STREAMS:
        live = [e.get(f"live_{stream}") for e in examples]
        if all(value is not None for value in live):
            arrays[f"live_{stream}"] = np.stack([np.asarray(value, dtype=np.int64) for value in live], axis=0)
    # The raw signals of each example's segment in the unit they were drawn in, NaN at gaps, on
    # the raw grid from the segment's start.
    for name in traces.RAW_SIGNAL_FIELDS:
        raw = [(e.get("raw") or {}).get(name) for e in examples]
        if all(value is not None for value in raw) and len({np.asarray(value).size for value in raw}) == 1:
            arrays[f"example_raw_{name}"] = np.stack([np.asarray(value, dtype=np.float32) for value in raw], axis=0)
            arrays[f"example_raw_{name}_unit"] = np.asarray(
                [str((e.get("raw_units") or {}).get(name, "normalised")) for e in examples], dtype=object
            )
    keys = [(index, key) for index, e in enumerate(examples) for key in e["maps"]]
    arrays["map_example"] = np.asarray([index for index, _ in keys], dtype=np.int64)
    arrays["map_readout"] = np.asarray([key[0] for _, key in keys], dtype=object)
    arrays["map_baseline"] = np.asarray([key[1] for _, key in keys], dtype=object)
    arrays["map_band"] = np.asarray([key[2] for _, key in keys], dtype=object)
    arrays["map_value_input"] = np.asarray([examples[i]["maps"][k]["value_input"] for i, k in keys], dtype=np.float64)
    arrays["map_value_baseline"] = np.asarray([examples[i]["maps"][k]["value_baseline"] for i, k in keys], dtype=np.float64)
    for stream in core.STREAMS:
        arrays[f"map_{stream}"] = np.stack([examples[i]["maps"][k][stream] for i, k in keys], axis=0)
    arrays["map_lag_profile"] = np.stack([examples[i]["maps"][k]["lag_profile"] for i, k in keys], axis=0)
    # The latent at the anchor and the layer split, where the examples carry them.
    for name in ("mu_prior", "shift", "kld_dim"):
        values = [np.asarray((e.get("latent") or {}).get(name, np.zeros(0)), dtype=np.float64) for e in examples]
        if all(v.size for v in values):
            arrays[f"example_latent_{name}"] = np.stack(values, axis=0)
    splits = [np.asarray((examples[i].get("layer") or {}).get((k[0], k[2]), np.zeros(0)), dtype=np.float64) for i, k in keys]
    width = max((int(v.size) for v in splits), default=0)
    if width:
        layer = np.full((len(splits), width), np.nan)
        for row, values in enumerate(splits):
            layer[row, :values.size] = values
        arrays["map_layer"] = layer
    return arrays


def write_example_pages(
    examples: Sequence[Mapping[str, Any]],
    *,
    directory: Path,
    lag_seconds: np.ndarray,
    cell: core.CellBinding,
    caveat: str,
    lag_bands: Mapping[str, Tuple[int, int]],
    horizons: Optional[Mapping[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Render one example page per class under ``maps/`` and return their manifest rows.

    Args:
        examples: The example records, in class order.
        directory: The analysis directory.
        lag_seconds: The compensated lag axis.
        cell: The cell binding.
        caveat: The one-line note printed under every page (:data:`~attributions.ATTRIBUTION_NOTE`).
        lag_bands: The configured lag bands, in the order their rows are drawn.
        horizons: The named horizon steps, in the order their rows are drawn, or ``None``.

    Returns:
        One row per page, in :data:`EXAMPLE_MANIFEST_COLUMNS` order.
    """
    manifest: List[Dict[str, Any]] = []
    root = Path(directory) / core.EXAMPLE_DIRNAME
    if examples:
        root.mkdir(parents=True, exist_ok=True)
    # One set of colour scales for every page, so a colour means one value on all of them.
    norms = core.example_norms(examples)
    for item in examples:
        stem = (
            f"{traces.class_dirname(item[labels.CLASS_COLUMN])}_"
            f"{traces.recording_stem(item['guid'], item[labels.SUBGROUP_COLUMN])}_anchor{int(item['anchor'])}"
        )
        figure = figures.render_figure(
            core.build_example_figure(
                item, lag_seconds=lag_seconds, cell=cell, caveat=caveat, lag_bands=lag_bands, horizons=horizons,
                norms=norms,
            ),
            root / f"{stem}{core.EXAMPLE_SUFFIX}",
        )
        manifest.append(
            {
                "guid": str(item["guid"]), labels.CLASS_COLUMN: item[labels.CLASS_COLUMN],
                labels.SUBGROUP_COLUMN: item[labels.SUBGROUP_COLUMN], "epoch": float(item["epoch"]),
                "anchor": int(item["anchor"]), "figure_file": Path(figure).relative_to(directory).as_posix(),
            }
        )
    return manifest


def run_traces(
    task: Any,
    loader: Any,
    chosen: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    cell: core.CellBinding,
    *,
    directory: Path,
    lag_seconds: np.ndarray,
    break_after_s: float,
    n_steps: int,
    anchors_per_segment: int,
    caveat: str,
    raw_scales: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """Trace every chosen recording and write its arrays and figure under its class directory.

    Returns:
        ``(manifest, failures, anchors)``: the manifest rows, the failures, and every traced
        anchor's row -- identity, clock, the trace readouts' scalars and lag statistics -- which
        the figure draws and the arrays file does not carry.
    """
    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    anchor_frames: List[pd.DataFrame] = []
    root = Path(directory) / core.TRACE_DIRNAME
    # Drawn after every recording is traced, so each panel shares one scale across the classes and
    # the pages compare by eye: (recording, destination, manifest row).
    pending: List[Tuple[Any, Path, int]] = []
    for _, choice in chosen.iterrows():
        guid = str(choice["guid"])
        clinical_class = choice[labels.CLASS_COLUMN]
        subgroup = choice[labels.SUBGROUP_COLUMN] if not pd.isna(choice[labels.SUBGROUP_COLUMN]) else None
        rows = recording_rows(guid, index_map)
        if rows.empty:
            failures.append({"guid": guid, "error": "no dataset row resolves to this recording"})
            continue
        try:
            segments = trace_recording(
                task, loader, rows, cell, clinical_class=clinical_class, subgroup=subgroup,
                n_steps=n_steps, anchors_per_segment=anchors_per_segment, raw_scales=raw_scales,
            )
            if not segments:
                failures.append({"guid": guid, "error": "no segment of this recording scored an anchor"})
                continue
            recording = traces.assemble_recording(
                segments, lag_profiles=core.TRACE_LAG_PROFILES, lag_seconds=lag_seconds, break_after_s=break_after_s
            )
            stem = traces.recording_stem(guid, subgroup)
            class_dir = root / traces.class_dirname(clinical_class)
            arrays = traces.write_recording_arrays(class_dir / f"{stem}{core.TRACE_SUFFIX}.npz", recording, lag_seconds=lag_seconds)
        except Exception as error:  # noqa: BLE001 - one recording is not worth the rest of them
            logger.warning(f"{core.ANALYSIS_DIRNAME}: trace of {guid} failed: {error}")
            failures.append({"guid": guid, "error": f"{type(error).__name__}: {error}"})
            continue
        anchor_frames.append(recording.anchors)
        span = recording.summary[cohort.HOURS_COLUMN]
        manifest.append(
            {
                "guid": guid, labels.CLASS_COLUMN: clinical_class, labels.SUBGROUP_COLUMN: subgroup,
                "n_segments": int(len(recording.segments)), "n_anchors": int(len(recording.anchors)),
                "span_hours": float(span.max() - span.min()) if len(span) else float("nan"),
                "coverage": float(choice["coverage"]) if "coverage" in choice.index else float("nan"),
                "arrays_file": Path(arrays).relative_to(directory).as_posix(),
                "figure_file": None,
            }
        )
        pending.append((recording, class_dir / f"{stem}{core.TRACE_SUFFIX}", len(manifest) - 1))
    scales = traces.shared_panel_scales([recording for recording, _, _ in pending], core.TRACE_PANELS)
    for recording, destination, row in pending:
        try:
            figure = figures.render_figure(
                traces.build_recording_figure(
                    recording, panels=core.TRACE_PANELS, lag_seconds=lag_seconds,
                    caveat=caveat,
                    scales=scales,
                ),
                destination,
            )
            manifest[row]["figure_file"] = Path(figure).relative_to(directory).as_posix()
        except Exception as error:  # noqa: BLE001 - one figure is not worth the rest of them
            logger.warning(f"{core.ANALYSIS_DIRNAME}: trace figure of {recording.guid} failed: {error}")
            failures.append({"guid": recording.guid, "error": f"{type(error).__name__}: {error}"})
    return manifest, failures, (pd.concat(anchor_frames, ignore_index=True) if anchor_frames else pd.DataFrame())


# =============================================================================
# The whole pass
# =============================================================================
#: Every traced anchor's row, all traced recordings in one table.
TRACE_ANCHORS_FILENAME = "attribution_trace_anchors.csv"

#: The identity every row of the per-row table carries, restated in the row-aligned arrays file
#: under a ``row_`` prefix so the arrays read without the table beside them.
ROW_IDENTITY_COLUMNS: Tuple[str, ...] = (
    "guid", labels.SUBGROUP_COLUMN, labels.CLASS_COLUMN, "epoch", "anchor", "readout", "baseline", "band",
)


def row_identity(rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    """The per-row identity arrays for the row-aligned arrays file, as strings and numbers (no pickle)."""
    arrays: Dict[str, np.ndarray] = {}
    for name in ROW_IDENTITY_COLUMNS:
        if name not in rows.columns:
            continue
        if name in ("epoch", "anchor"):
            arrays[f"row_{name}"] = rows[name].to_numpy(dtype=np.float64 if name == "epoch" else np.int64)
        else:
            arrays[f"row_{name}"] = rows[name].astype(str).to_numpy(dtype=str)
    return arrays


def source_channel_shifts(model: Any) -> List[int]:
    """The per-channel shift the source gate applies before the lag attention, in stored steps."""
    steps = getattr(getattr(getattr(model, "source_gate", None), "delay", None), "delay_steps", None)
    return [] if steps is None else [int(step) for step in torch.as_tensor(steps).reshape(-1).tolist()]
def run_pass(
    task: Any,
    loader: Any,
    segments: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    cell: core.CellBinding,
    *,
    eval_config: Mapping[str, Any],
    results_dir: Any,
    lag_seconds: np.ndarray,
    break_after_s: float,
    segment_stride_s: float,
    channel_map: Optional[pd.DataFrame],
    occlusion: Optional[pd.DataFrame],
    spectral: Optional[pd.DataFrame],
    delay_steps: int,
    per_anchor: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Select, attribute, reduce, write, and describe.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader.
        segments: The labelled segments the pass may draw from, from :func:`labelled_segments`.
        index_map: The dataset listing.
        cell: The cell binding.
        eval_config: The validated block, for the cap, the seed and the lag bands.
        results_dir: The run's results directory; the pass writes into its own subdirectory.
        lag_seconds: The compensated lag axis.
        break_after_s: The trace's break tolerance in seconds.
        segment_stride_s: The stride between consecutive stored segments, in seconds, for the
            completeness the traced recordings are ranked by.
        channel_map: The declared-axis channel map, or ``None``.
        occlusion: The occlusion summary, or ``None``.
        spectral: The band-resolved skill table, or ``None``.
        delay_steps: The stored-step delay the lag axis carries, recorded in the plan.
        per_anchor: The collection pass's per-anchor table. With it, both passes attribute the
            high-KL clean anchors (:func:`informative_anchors`) and the example pages show the
            :data:`~.attributions.EXAMPLES_PER_CLASS` highest-KL ones per class; without it, they
            fall back to evenly spread anchors and one example per class.

    Returns:
        The block: counts, plan, selection, cost, checks, the method record, the trace manifest,
        failures and files.
    """
    directory = Path(str(results_dir)) / core.ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    caps = dict(eval_config.get("caps") or {})
    cap = int(caps.get(core.CAP_NAME) or core.DEFAULT_SEGMENTS)
    per_recording = int(caps.get(core.SEGMENTS_PER_RECORDING_CAP_NAME) or core.DEFAULT_SEGMENTS_PER_RECORDING)
    seed = int(eval_config.get("seed", 0)) + core.DRAW_SEED_OFFSET
    configured = dict(eval_config.get("occlusion_bands") or {})
    lag_bands = {str(name): (int(span[0]), int(span[1])) for name, span in configured.items()}
    n_steps = core.IG_STEPS
    horizons = core.horizon_steps(task.orig_model)
    # ponytail: every lag here is a stored-step offset t - s, which is the attention's lag only
    # while the source gate shifts no channel (the shipped unaligned arms). An aligned arm needs
    # the per-channel shift added to the offset in lag_profile, offset_channel_map and
    # lag_band_feature_mask; until then it is recorded and warned about, not corrected.
    shifts = source_channel_shifts(task.orig_model)
    if any(shifts):
        logger.warning(
            f"{core.ANALYSIS_DIRNAME}: the source gate shifts channels by up to {max(shifts)} "
            f"step(s), so the stored-step lags of this analysis sit that far from the attention's "
            f"own lags and from the occlusion bands"
        )
    plan: Dict[str, Any] = {
        "capped": True, "cap": cap, "seed": seed, "segments_per_recording": per_recording,
        "anchors_per_segment": core.ANCHORS_PER_SEGMENT,
        "ig_steps": n_steps, "entry_fraction": core.BASELINE_ENTRY_FRACTION,
        "baselines": list(core.BASELINES), "readouts": list(core.MAIN_READOUTS),
        "horizon_steps": {name: int(step) for name, step in horizons.items()},
        "example_readouts": list(core.EXAMPLE_READOUTS),
        "lag_readout": cell.lag_readout, "lag_bands": {name: list(span) for name, span in lag_bands.items()},
        "lag_definition": "stored-step offset from the anchor, t - s",
        "source_channel_shift_steps_max": max(shifts, default=0),
        "layer": cell.layer_label, "delay_steps": int(delay_steps),
        # The anchor axis the attributed anchors are chosen on, and what the attributed forward
        # decodes: each row's own anchor alone, at the two latent means (attributed_forward).
        "anchor_phase": DENSE_ANCHOR_GEOMETRY[0], "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
        "attributed_forward": "one anchor per row, decoded at the latent means, no epsilon drawn",
        "trace_recordings_per_class": core.TRACE_RECORDINGS_PER_CLASS,
        # The bound is the window the traced recordings are ranked for completeness over, and,
        # when set, the only segments of each chosen recording that are traced.
        "max_hours_before_delivery_applied": eval_config.get("max_hours_before_delivery") is not None,
        "trace_completeness_window_hours": eval_config.get("max_hours_before_delivery"),
    }
    labelled = labelled_segments(segments)
    candidates, plan["anchor_selection"] = informative_anchors(per_anchor)
    selected, accounting = select_segments(
        labelled, index_map, cap=cap, seed=seed, candidates=candidates, examples_per_class=core.EXAMPLES_PER_CLASS,
        segments_per_recording=per_recording,
    )
    channel_groups = core.channel_groups_from_map(channel_map)
    raw_scales = traces.raw_signal_scales(None, loader)

    started = time.perf_counter()
    work, n_batches = run_segments(
        task, loader, selected, cell, lag_bands=lag_bands, channel_groups=channel_groups,
        n_steps=n_steps, anchors_per_segment=core.ANCHORS_PER_SEGMENT, raw_scales=raw_scales,
    )
    purity = target_only_check(task, loader, selected, cell, n_steps=n_steps)
    elapsed = time.perf_counter() - started

    rows = rows_frame(work)
    vectors = stack_vectors(work)
    rows.to_csv(directory / core.ROWS_FILENAME, index=False)
    np.savez_compressed(
        directory / core.VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64),
        **row_identity(rows), **vectors,
    )
    plan["anchor_selection"]["n_unclean_anchors"] = int(work.n_unclean_anchors)
    # The example anchors in class order, worst first, as every cohort figure orders them, then by
    # rank within a class; the overview shows the first of each class.
    order = labels.ordered_groups(sorted({str(e[labels.CLASS_COLUMN]) for e in work.examples.values()}), labels.CLASS_COLUMN)
    examples = sorted(work.examples.values(), key=lambda e: (order.index(str(e[labels.CLASS_COLUMN])), e.get("example_rank", 0)))
    overview = [e for e in examples if e.get("example_rank", 0) == 0]
    if examples:
        np.savez_compressed(directory / core.MAPS_FILENAME, **example_arrays(examples))
    lag_channel = lag_channel_summary(work)
    if lag_channel:
        np.savez_compressed(directory / core.LAG_CHANNEL_FILENAME, **lag_channel_arrays(lag_channel, lag_seconds))
    per_recording = recordings_frame(rows)
    per_recording.to_csv(directory / core.RECORDINGS_FILENAME)
    summary = summary_frame(rows)
    summary.to_csv(directory / core.SUMMARY_FILENAME, index=False)
    bands = bands_frame(rows, spectral)
    bands.to_csv(directory / core.BANDS_FILENAME, index=False)
    lag_band_table = lag_bands_frame(rows, lag_bands, occlusion)
    lag_band_table.to_csv(directory / core.LAG_BANDS_FILENAME, index=False)
    layer = layer_frame(rows, vectors)
    layer.to_csv(directory / core.LAYER_FILENAME, index=False)
    null = null_frame(rows)
    null.to_csv(directory / core.NULL_FILENAME, index=False)
    blocks = blocks_frame(rows)
    blocks.to_csv(directory / core.BLOCKS_FILENAME, index=False)

    # The long sentence travels in the record; the figures print its one-line form.
    caveat = core.ATTRIBUTION_CAVEAT
    note = core.ATTRIBUTION_NOTE
    files: List[str] = [
        core.ROWS_FILENAME, core.VECTORS_FILENAME, core.RECORDINGS_FILENAME, core.SUMMARY_FILENAME,
        core.BANDS_FILENAME, core.LAG_BANDS_FILENAME, core.LAYER_FILENAME, core.NULL_FILENAME,
        core.BLOCKS_FILENAME,
    ]
    if examples:
        files.append(core.MAPS_FILENAME)
    if lag_channel:
        files.append(core.LAG_CHANNEL_FILENAME)
    figure_paths = [
        figures.render_figure(core.build_map_figure(overview, lag_seconds=lag_seconds, cell=cell, caveat=note), directory / core.MAP_FIGURE),
        figures.render_figure(
            core.build_lag_profile_figure(
                rows, vectors, lag_seconds=lag_seconds, readouts=core.MAIN_READOUTS, cell=cell, caveat=note,
                lag_bands=lag_bands,
            ),
            directory / core.LAG_PROFILE_FIGURE,
        ),
        figures.render_figure(core.build_band_figure(bands, lag_band_table, readouts=core.MAIN_READOUTS, caveat=note), directory / core.BAND_FIGURE),
        figures.render_figure(
            core.build_layer_figure(layer, _top_coordinate_rows(rows, vectors), cell=cell, lag_seconds=lag_seconds, caveat=note),
            directory / core.LAYER_FIGURE,
        ),
        figures.render_figure(core.build_null_figure(null, caveat=note), directory / core.NULL_FIGURE),
        figures.render_figure(
            core.build_channel_figure(
                rows, vectors, readouts=core.MAIN_READOUTS, channel_groups=channel_groups,
                n_scattering=work.n_scattering, caveat=note,
            ),
            directory / core.CHANNEL_FIGURE,
        ),
        figures.render_figure(
            core.build_lag_channel_figure(
                lag_channel, lag_seconds=lag_seconds, readouts=core.MAIN_READOUTS,
                n_scattering=work.n_scattering, caveat=note,
            ),
            directory / core.LAG_CHANNEL_FIGURE,
        ),
        figures.render_figure(
            core.build_time_profile_figure(rows, vectors, lag_seconds=lag_seconds, readouts=core.MAIN_READOUTS, caveat=note),
            directory / core.TIME_PROFILE_FIGURE,
        ),
        figures.render_figure(
            core.build_checks_figure(rows, tolerance=COMPLETENESS_TOLERANCE, caveat=note),
            directory / core.CHECKS_FIGURE,
        ),
        figures.render_figure(
            core.build_delivery_figure(rows, vectors, lag_seconds=lag_seconds, readouts=core.MAIN_READOUTS, caveat=note),
            directory / core.DELIVERY_FIGURE,
        ),
        figures.render_figure(core.build_block_figure(blocks, caveat=note), directory / core.BLOCK_FIGURE),
        figures.render_figure(
            core.build_horizon_figure(rows, vectors, lag_seconds=lag_seconds, horizons=horizons, caveat=note),
            directory / core.HORIZON_FIGURE,
        ),
    ]
    files.extend(Path(path).name for path in figure_paths)
    example_manifest = write_example_pages(
        examples, directory=directory, lag_seconds=lag_seconds, cell=cell, caveat=note, lag_bands=lag_bands,
        horizons=horizons,
    )
    pd.DataFrame(example_manifest, columns=list(EXAMPLE_MANIFEST_COLUMNS)).to_csv(
        directory / EXAMPLE_MANIFEST_FILENAME, index=False
    )
    files.append(EXAMPLE_MANIFEST_FILENAME)

    # The class comparison reads its own, larger draw: one example anchor per class says nothing
    # about a class, and the main pass is too costly per segment to scale up.
    contrast = run_cohort(
        task, loader, labelled, index_map, cell,
        cap=int(caps.get(class_contrast.COHORT_CAP_NAME) or class_contrast.DEFAULT_COHORT_SEGMENTS),
        seed=seed, directory=directory, lag_seconds=lag_seconds, lag_bands=lag_bands,
        channel_groups=channel_groups, n_steps=n_steps, caveat=note, candidates=candidates,
    )
    files.extend(contrast["files"])

    window_hours = eval_config.get("max_hours_before_delivery")
    chosen, trace_accounting = select_trace_recordings(
        labelled, index_map, stride_s=segment_stride_s,
        window_hours=None if window_hours is None else float(window_hours),
    )
    trace_started = time.perf_counter()
    manifest, failures, trace_anchors = run_traces(
        task, loader, chosen, cohort.within_horizon_index(index_map, window_hours), cell,
        directory=directory, lag_seconds=lag_seconds,
        break_after_s=break_after_s, n_steps=n_steps, anchors_per_segment=core.ANCHORS_PER_SEGMENT, caveat=note,
        raw_scales=raw_scales,
    )
    trace_elapsed = time.perf_counter() - trace_started
    pd.DataFrame(manifest, columns=list(TRACE_MANIFEST_COLUMNS)).to_csv(directory / TRACE_MANIFEST_FILENAME, index=False)
    trace_anchors.to_csv(directory / TRACE_ANCHORS_FILENAME, index=False)
    files.extend([TRACE_MANIFEST_FILENAME, TRACE_ANCHORS_FILENAME])

    main_rows = rows[rows["readout"].isin(core.MAIN_READOUTS)] if len(rows) else rows
    checks = {
        "after_anchor_max_abs": float(rows["after_anchor_max_abs"].max()) if len(rows) else None,
        "gated_off_max_abs": float(rows["gated_off_max_abs"].max()) if len(rows) else None,
        "completeness_rel_max": float(np.nanmax(main_rows["completeness_rel"])) if len(main_rows) else None,
        "completeness_rel_median": float(np.nanmedian(main_rows["completeness_rel"])) if len(main_rows) else None,
        "completeness_tolerance": COMPLETENESS_TOLERANCE,
        "n_rows_over_tolerance": int((main_rows["completeness_rel"] > COMPLETENESS_TOLERANCE).sum()) if len(main_rows) else 0,
        "target_only": purity,
        "meaning": (
            "after_anchor_max_abs is the largest attribution to any stored step after the "
            "anchor, exactly zero on a causal model; gated_off_max_abs the largest attribution to "
            "a source step a channel had not warmed up at, exactly zero under the input gate; "
            "completeness_rel is |IG sum - (f(x) - f(x0))| over the larger of |f(x) - f(x0)| and "
            "|f(x)| per row, with x0 the entry point; target_only is the source attribution of the "
            "base block score, exactly zero on a model whose prior reads no source"
        ),
    }
    by_class = {str(name): int(count) for name, count in selected[labels.CLASS_COLUMN].value_counts().items()} if len(selected) else {}
    logger.info(
        f"{core.ANALYSIS_DIRNAME}: attributed {len(selected)} segment(s) at {len(rows)} row(s) in "
        f"{elapsed:.1f} s; traced {len(manifest)} recording(s) in {trace_elapsed:.1f} s, "
        f"{len(failures)} failed"
    )
    return {
        "n_samples": int(len(selected)),
        "composition": {
            "n_recordings": int(selected["guid"].nunique()) if len(selected) else 0,
            "n_segments_by_class": by_class,
            "n_rows": int(len(rows)),
        },
        "plan": plan,
        "selection": accounting,
        "trace_selection": trace_accounting,
        "cost": {
            **core.cost_record(
                elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
                n_forward_equivalents=int(work.forward_equivalents), device=getattr(task, "device", None),
            ),
            # The traces are a second attribution loop, over every segment of one recording per
            # class, and are timed apart from the drawn segments the rate above describes.
            "trace_elapsed_s": float(trace_elapsed),
            "trace_n_anchors": int(len(trace_anchors)),
        },
        "checks": checks,
        "class_contrast": contrast,
        "summary": summary.to_dict(orient="records"),
        "lag_bands": lag_band_table.to_dict(orient="records"),
        "blocks": blocks.to_dict(orient="records"),
        "joined": {
            "channel_map": channel_map is not None,
            "occlusion_summary": occlusion is not None,
            "spectral_skill_bands": spectral is not None,
        },
        "methods": core.METHOD_RECORD,
        "lag_qualification": cell.lag_qualification,
        "caveat": caveat,
        "traces": manifest,
        "examples": example_manifest,
        "failures": failures,
        "files": files,
    }


def _top_coordinate_rows(rows: pd.DataFrame, vectors: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """The top-coordinate readout's rows with their lag-aligned profiles attached, for the layer figure."""
    if rows.empty or "lag_profile" not in vectors:
        return pd.DataFrame()
    keep = (rows["readout"] == core.READOUT_KLD_DIM).to_numpy()
    if not keep.any():
        return pd.DataFrame()
    subset = rows[keep].copy()
    subset["lag_profile"] = list(vectors["lag_profile"][keep])
    return subset


__all__ = [
    "CHANNEL_MAP_FILENAME", "COMPLETENESS_TOLERANCE", "EXAMPLE_MANIFEST_COLUMNS",
    "EXAMPLE_MANIFEST_FILENAME", "OCCLUSION_SUMMARY_PATH",
    "RECORDING_VALUE_COLUMNS", "SPECTRAL_BANDS_PATH", "TRACE_MANIFEST_COLUMNS",
    "TRACE_MANIFEST_FILENAME", "BatchWork", "attribute_batch", "attribute_example", "bands_frame", "blocks_frame",
    "example_arrays", "labelled_segments", "lag_channel_arrays", "lag_channel_summary",
    "lag_bands_frame", "layer_frame", "null_frame", "read_channel_map", "read_occlusion_summary",
    "read_spectral_bands", "recording_completeness", "recording_rows", "recording_table",
    "recordings_frame", "rows_frame",
    "example_key", "gradcam_frame", "informative_anchors", "run_cohort", "run_pass", "run_segments", "run_traces", "select_segments", "select_trace_recordings",
    "stack_vectors", "summary_frame", "target_only_check", "trace_recording", "write_example_pages",
    "ROW_IDENTITY_COLUMNS", "TRACE_ANCHORS_FILENAME", "example_map", "row_identity", "source_channel_shifts",
]
