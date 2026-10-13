r"""Attribution by anchor stratum in the last hours before delivery: high against low $K_t$, and
anchors inside a uterine contraction against anchors between contractions, per clinical class.

The cohort pass (:func:`~teb_vae.lag_attn_cfs.eval.attribution_pass.run_cohort`) attributes only
each segment's **high-KL** anchors, from the **middle** segment of each recording. It therefore
cannot say what changes when the divergence $K_t$ is high, and it mixes early and late labour. This
pass asks two paired questions instead, inside one late window:

1. **High against low $K_t$.** In every chosen segment the anchors in the upper
   $1 - q_{\mathrm{hi}}$ of that segment's $K_t$ (``kl_high``, $q_{\mathrm{hi}} = 0.7$) are attributed
   beside the anchors in its lower $q_{\mathrm{lo}} = 0.3$ (``kl_low``).
2. **Inside against between contractions.** The collection pass detects contractions on the raw UP
   trace (:func:`~teb_vae.lag_attn_cfs.eval.events.detect_contractions`) and writes
   ``seconds_since_contraction`` per anchor. An anchor with $0 \le a < w$ seconds since the last
   onset is ``ua_contraction``; an anchor with $a \ge w$, or with no onset behind it, is
   ``ua_quiet``. $w$ is ``eval_config.event_lag_window_s`` ($120$ s by default). Only segments
   with at least one contraction anchor give UA strata, so a flat UP trace is not read as "between
   contractions".

**The classes are compared at one coupling level, not at one tier.** The two KL tiers are per
segment, so ``kl_high`` can mean $0.5$ nats in one class and $3$ nats in another, and integrated
gradients of $K_t$ sum to roughly $K_t - K_{\mathrm{null}}$: a class difference read at a tier is
largely a class difference in $K_t$. Tiers stay right **within** a recording, where A and B are
paired. **Between** classes the pass adds a fifth stratum, ``kl_matched``: the anchors inside one
band of the pooled $K_t$ distribution (:func:`matched_band`), so every class is read at the same
nats. Its class tests (:func:`matched_class_tests`) and :data:`MATCHED_FIGURE` are the class
comparison this pass stands behind; the column titles of the contrast figures print each class's
median $K_t$ per tier, so the gap between tiers and levels stays visible.

**The late window.** Segments that start at most :data:`LATE_WINDOW_HOURS` ($2$ h) before delivery
are eligible, with ``epoch`` the segment start in seconds relative to delivery (negative before it).
The recordings are drawn class-balanced and seeded, as in the cohort pass, with
:data:`SEGMENTS_PER_RECORDING` segments each, up to ``caps.attribution_strata_segments`` segments.

**What is attributed.** Per stratum, at up to
:data:`~teb_vae.lag_attn_cfs.eval.attributions.ANCHORS_PER_SEGMENT` anchors per segment, one
horizon apart: integrated gradients of $K_t$ and of the forecast gap under both baselines, and
Grad-CAM of the same two readouts in the three views of
:mod:`~teb_vae.lag_attn_cfs.eval.gradcam` (dense-latent cells only). The source lag profile is
read under ``source_null``, which holds the target streams at the input, so the target
attribution there is exactly zero. The target profile and the source share of the attribution
are read under ``all_zero``, the only baseline under which the FHR history moves. Each row also
carries the collection pass's own $K_t$, forecast gain and contraction age at that anchor.

**How strata are compared.** One recording is one unit. Each recording's anchors are averaged per
stratum. A contrast is then the **paired** difference, stratum A minus stratum B, over the
recordings that hold both. Per class it reports the mean difference with a $95\%$ percentile
bootstrap interval and a Wilcoxon signed-rank test, Holm-adjusted within one family (one contrast,
one class, one method and readout). Across classes it tests whether the paired difference itself
differs, with Kruskal--Wallis, Holm-adjusted within one contrast, method and readout.

**A forward-only summary uses every late anchor.** :func:`build_late_behaviour_figure` needs no
gradient. It reads the per-anchor table and its lag-map sidecar for the whole late window: $K_t$
and the forecast gain against minutes before delivery and against seconds since contraction onset,
and the attention and $K_t$ lag profiles of high against low $K_t$ anchors.

**Limits.** The model trained on healthy recordings only, so a class difference describes the
fitted model's response, not a physiological mechanism. The strata are defined by $K_t$ itself, so
a high-against-low difference in a readout that grows with $K_t$ is partly built in; read shares and
centroids, which are scale-free, before totals. Minutes before delivery is the segment start plus
the anchor's stored step, so it inherits the segment's trimming offset (under one minute).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import attribution_pass, class_contrast, gradcam
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import figures as shared_figures, labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.dataset_rows import epoch_stamp
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL

# =============================================================================
# Settings
# =============================================================================
#: The late window, in hours before delivery, that a segment's start must fall in.
LATE_WINDOW_HOURS = 2.0

#: The cap on the strata draw, under ``eval_config.caps``, and its default. Every segment is
#: attributed once per stratum (five) under both baselines, so the cost is about seven cohort segments per
#: segment: 72 segments take about 40 minutes on one laptop GPU for the shipped transformer.
STRATA_CAP_NAME = "attribution_strata_segments"
DEFAULT_STRATA_SEGMENTS = 72

#: Segments per drawn recording: one, its middle late segment, because the tests count recordings
#: and are better served by more recordings than by more segments of the same ones.
SEGMENTS_PER_RECORDING = 1

#: The per-segment $K_t$ quantiles of the two KL strata.
KL_HIGH_QUANTILE = core.HIGH_KL_QUANTILE
KL_LOW_QUANTILE = 0.3

#: The contraction window $w$ when ``eval_config`` carries no ``event_lag_window_s``.
DEFAULT_CONTRACTION_WINDOW_S = 120.0

#: The per-anchor columns this pass reads. The forecast gain is the first gain column present, in
#: the order the ``lag_high_kl`` analysis uses.
CONTRACTION_AGE_COLUMN = "seconds_since_contraction"
GAIN_COLUMNS: Tuple[str, ...] = ("mean_pred_gap", "mc_pred_gap", "pred_gap")

#: The readouts this pass attributes, by both methods.
READOUTS: Tuple[str, ...] = (core.READOUT_KLD, core.READOUT_PRED_GAP)

KL_HIGH, KL_LOW = "kl_high", "kl_low"
UA_CONTRACTION, UA_QUIET = "ua_contraction", "ua_quiet"
#: Anchors inside one band of the **pooled** $K_t$ distribution, the same nats for every class.
KL_MATCHED = "kl_matched"
STRATA: Tuple[str, ...] = (KL_HIGH, KL_LOW, UA_CONTRACTION, UA_QUIET, KL_MATCHED)

#: The pooled $K_t$ of the clean late anchors is cut into this many equal-count bands; the matched
#: stratum is the band whose worst-covered class holds the most recordings (:func:`matched_band`).
MATCHED_BANDS = 5

#: Each contrast as ``(A, B)``; a paired difference reads A minus B.
CONTRASTS: Dict[str, Tuple[str, str]] = {"kl": (KL_HIGH, KL_LOW), "ua": (UA_CONTRACTION, UA_QUIET)}
CONTRAST_TITLES: Dict[str, str] = {"kl": "high vs low $K_t$", "ua": "in vs between contractions"}
STRATUM_LABELS: Dict[str, str] = {
    KL_HIGH: "high $K_t$", KL_LOW: "low $K_t$", UA_CONTRACTION: "in contraction", UA_QUIET: "between contractions",
    KL_MATCHED: "matched $K_t$",
}

#: Stratum A in black, stratum B in blue: neither is a class colour.
STRATUM_COLOURS: Tuple[str, str] = (figures.COLOR_BLACK, figures.COLOR_BLUE)

#: The bins of the forward-only figure: minutes before delivery and seconds since onset.
DELIVERY_BIN_MINUTES = 10.0
ONSET_BIN_SECONDS = 15.0
ONSET_MAX_SECONDS = 300.0

#: The scalar metrics drawn on the paired figure, with their axis labels. Each name is
#: ``<method>:<readout>:<column>`` or ``anchor:<column>``.
PAIRED_METRICS: Tuple[Tuple[str, str], ...] = (
    ("anchor:kld_per_t", "$K_t$ (nats)"),
    ("anchor:gain", "forecast gain (nats)"),
    (f"ig:{core.READOUT_KLD}:source_share", "source share of |IG|, $K_t$"),
    (f"ig:{core.READOUT_KLD}:lag_centroid_s", "IG lag centroid of $K_t$ (s)"),
    (f"gradcam:{core.READOUT_KLD}:attention_centroid_s", "Grad-CAM attention centroid (s)"),
    (f"ig:{core.READOUT_PRED_GAP}:lag_centroid_s", "IG lag centroid of gap (s)"),
)

#: The profiles and scalars of the KL-matched class comparison (:func:`build_matched_figure`).
MATCHED_METRICS: Tuple[Tuple[str, str], ...] = (
    ("anchor:kld_per_t", "$K_t$ at anchor (nats): the match"),
    (f"ig:{core.READOUT_KLD}:lag_centroid_s", "IG lag centroid of $K_t$ (s)"),
    (f"ig:{core.READOUT_KLD}:source_share", "source share of |IG|, $K_t$"),
    (f"gradcam:{core.READOUT_KLD}:attention_centroid_s", "Grad-CAM attention centroid (s)"),
)

BOOTSTRAP_RESAMPLES = class_contrast.BOOTSTRAP_RESAMPLES
CONFIDENCE = class_contrast.CONFIDENCE
ALPHA = class_contrast.ALPHA

ROWS_FILENAME = "strata_rows.csv"
VECTORS_FILENAME = "strata_vectors.npz"
GRADCAM_ROWS_FILENAME = "strata_gradcam_rows.csv"
GRADCAM_VECTORS_FILENAME = "strata_gradcam_vectors.npz"
RECORDINGS_FILENAME = "strata_recordings.csv"
STATS_FILENAME = "strata_stats.csv"
IG_FIGURE = "strata_{contrast}_ig"
GRADCAM_FIGURE = "strata_{contrast}_gradcam"
PAIRED_FIGURE = "strata_paired"
LATE_FIGURE = "late_behaviour"
MATCHED_FIGURE = "strata_matched"
ONSET_FIGURE = "strata_onset_locked"

#: Seed offset of the onset-shuffle null of :func:`build_onset_figure`.
ONSET_NULL_SEED_OFFSET = 7
MATCHED_STATS_FILENAME = "strata_matched_stats.csv"
MATCHED_PAIRWISE_FILENAME = "strata_matched_pairwise.csv"

#: The two IG vectors this pass keeps, both on the lag axis: the source and the target attribution
#: re-indexed by offset from the anchor.
IG_VECTORS: Tuple[Tuple[str, str], ...] = (("lag_profile", "source"), ("target_lag_profile", "target"))

#: The baseline each IG vector is drawn under. The target attribution is exactly zero under
#: ``source_null``, which holds the target streams at the input, so it is drawn under ``all_zero``.
IG_VECTOR_BASELINE: Dict[str, str] = {"source": core.BASELINE_SOURCE_NULL, "target": core.BASELINE_ALL_ZERO}


# =============================================================================
# Anchor strata
# =============================================================================
def hours_before_delivery(epoch: Any) -> np.ndarray:
    """Hours before delivery of a segment start, from ``epoch`` in seconds relative to delivery."""
    return -np.asarray(epoch, dtype=np.float64) / 3600.0


def gain_column(frame: pd.DataFrame) -> Optional[str]:
    """The first of :data:`GAIN_COLUMNS` the frame carries, or ``None``."""
    return next((name for name in GAIN_COLUMNS if name in frame.columns), None)


def late_anchors(per_anchor: Optional[pd.DataFrame], *, late_hours: float = LATE_WINDOW_HOURS) -> pd.DataFrame:
    """The scored anchors of the segments that start within ``late_hours`` of delivery.

    Args:
        per_anchor: The collection pass's per-anchor table, or ``None``.
        late_hours: The late window.

    Returns:
        The late rows with ``guid`` as a string, ``stamp`` (the rounded epoch),
        ``minutes_before_delivery`` of each anchor and ``row`` (the position in ``per_anchor``,
        which the lag-map sidecar is aligned to) added; empty when the table lacks ``epoch``,
        ``anchor`` or ``kld_per_t``.
    """
    needed = {"guid", "epoch", "anchor", "kld_per_t"}
    if per_anchor is None or per_anchor.empty or not needed <= set(per_anchor.columns):
        return pd.DataFrame()
    hours = hours_before_delivery(per_anchor["epoch"])
    keep = np.isfinite(hours) & (hours >= 0.0) & (hours <= float(late_hours))
    late = per_anchor.loc[keep].reset_index(drop=True)
    late["row"] = np.flatnonzero(keep)
    late["guid"] = late["guid"].astype(str)
    late["stamp"] = [epoch_stamp(e) for e in late["epoch"]]
    seconds = late["epoch"].to_numpy(dtype=np.float64) + late["anchor"].to_numpy(dtype=np.float64) * SECONDS_PER_STEP
    late["minutes_before_delivery"] = np.clip(-seconds / 60.0, 0.0, None)
    return late


def attach_class(late: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Give every late anchor its recording's class, from ``labelled``, and drop the unlabelled.

    Args:
        late: From :func:`late_anchors`.
        labelled: The labelled segments, with ``guid`` and the class column.

    Returns:
        The labelled late rows, re-indexed from zero.
    """
    if late.empty:
        return late
    if labels.CLASS_COLUMN not in late.columns:
        named = labelled.assign(guid=labelled["guid"].astype(str)).drop_duplicates("guid") if len(labelled) else labelled
        mapping = named.set_index("guid")[labels.CLASS_COLUMN] if len(named) else pd.Series(dtype=object)
        late = late.assign(**{labels.CLASS_COLUMN: late["guid"].map(mapping)})
    return late[late[labels.CLASS_COLUMN].notna()].reset_index(drop=True)


def kl_tier(late: pd.DataFrame) -> np.ndarray:
    """Each late anchor's per-segment KL tier: ``kl_high``, ``kl_low`` or ``''``.

    Args:
        late: From :func:`late_anchors`.

    Returns:
        $(N,)$ strings.
    """
    if late.empty:
        return np.zeros(0, dtype=object)
    grouped = late.groupby(["guid", "stamp"], dropna=False)["kld_per_t"]
    high = grouped.transform("quantile", KL_HIGH_QUANTILE).to_numpy(dtype=np.float64)
    low = grouped.transform("quantile", KL_LOW_QUANTILE).to_numpy(dtype=np.float64)
    kl = late["kld_per_t"].to_numpy(dtype=np.float64)
    tier = np.full(len(late), "", dtype=object)
    tier[kl <= low] = KL_LOW
    # High after low, so a segment with one distinct value is all high rather than all low.
    tier[kl >= high] = KL_HIGH
    return tier


def ua_phase(late: pd.DataFrame, *, window_s: float) -> np.ndarray:
    """Each late anchor's contraction phase: ``ua_contraction``, ``ua_quiet`` or ``''``.

    A segment with no anchor inside a contraction gets ``''`` throughout, because its quiet anchors
    may sit on a flat or missing UP trace rather than between contractions.

    Args:
        late: From :func:`late_anchors`.
        window_s: $w$, the seconds after an onset that count as inside the contraction.

    Returns:
        $(N,)$ strings; all ``''`` when the table has no contraction column.
    """
    phase = np.full(len(late), "", dtype=object)
    if late.empty or CONTRACTION_AGE_COLUMN not in late.columns:
        return phase
    age = late[CONTRACTION_AGE_COLUMN].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(age) & (age >= 0.0) & (age < float(window_s))
    # The age counts onsets inside the segment only, so an anchor with no onset behind it is
    # "between contractions" only if the segment holds at least w seconds of history before it;
    # otherwise a contraction that began before the segment could still be running.
    history_s = late["anchor"].to_numpy(dtype=np.float64) * SECONDS_PER_STEP
    measured = np.isfinite(age) | (history_s >= float(window_s))
    has_event = pd.Series(inside).groupby([late["guid"].to_numpy(), late["stamp"].to_numpy()]).transform("any").to_numpy()
    phase[inside] = UA_CONTRACTION
    phase[has_event & ~inside & measured] = UA_QUIET
    return phase


def matched_band(late: pd.DataFrame, clean: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
    r"""The pooled $K_t$ band in which every class can be compared at the same coupling level.

    The clean late anchors of every class are pooled and cut at the
    $0, \frac{1}{B}, \ldots, 1$ quantiles of $K_t$, with $B$ = :data:`MATCHED_BANDS`. For each
    band, each class counts its recordings with at least one anchor in it. The chosen band is the
    one whose smallest class count is largest; a tie goes to the narrowest band in nats, because a
    wide band does not match the classes inside it. One threshold in nats then holds for every
    class, which is what a per-segment tier cannot give. Whether the match held is read off the
    $K_t$ test of :func:`matched_class_tests`.

    Args:
        late: The labelled late anchors (:func:`attach_class`).
        clean: $(N,)$ which anchors reach the clean coverage.

    Returns:
        ``(low, high, record)``: the inclusive band edges in nats, and the edges and the per-band
        recording counts by class. ``NaN`` edges when there is nothing to cut.
    """
    kl = late["kld_per_t"].to_numpy(dtype=np.float64)
    usable = clean & np.isfinite(kl)
    if not usable.any():
        return float("nan"), float("nan"), {"bands": MATCHED_BANDS, "reason": "no clean late anchor"}
    edges = np.quantile(kl[usable], np.linspace(0.0, 1.0, MATCHED_BANDS + 1))
    band = np.clip(np.searchsorted(edges, kl, side="right") - 1, 0, MATCHED_BANDS - 1)
    table = pd.DataFrame({"guid": late["guid"].to_numpy()[usable], "cls": late[labels.CLASS_COLUMN].astype(str).to_numpy()[usable],
                          "band": band[usable]})
    counts = table.drop_duplicates(["guid", "band"]).groupby(["band", "cls"]).size().unstack(fill_value=0)
    counts = counts.reindex(range(MATCHED_BANDS), fill_value=0)
    worst = counts.min(axis=1).to_numpy()
    # Among the bands that cover the classes best, the narrowest in nats: the top band of a skewed
    # K_t distribution can span an order of magnitude, inside which the classes are not matched.
    tied = np.flatnonzero(worst == worst.max())
    widths = np.diff(edges)
    # A zero-width band is a tie of many equal K_t (e.g. exactly zero), not a narrow match.
    open_bands = tied[widths[tied] > 0.0]
    tied = open_bands if open_bands.size else tied
    chosen = int(tied[np.argmin(widths[tied])])
    return float(edges[chosen]), float(edges[chosen + 1]), {
        "bands": MATCHED_BANDS, "edges_nats": [float(e) for e in edges], "chosen_band": chosen,
        "low_nats": float(edges[chosen]), "high_nats": float(edges[chosen + 1]), "width_nats": float(widths[chosen]),
        "recordings_per_band_by_class": {str(c): [int(v) for v in counts[c]] for c in counts.columns},
    }


def strata_candidates(late: pd.DataFrame, *, window_s: float, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Every clean late anchor once per stratum it belongs to, in a seeded random order.

    Random rather than ranked: an extreme-first order would compare each segment's most extreme
    anchors with each other, not the two tiers.

    Args:
        late: From :func:`late_anchors`.
        window_s: The contraction window $w$.
        seed: The shuffle seed.

    Returns:
        ``(candidates, record)``: ``guid``, ``stamp``, ``anchor``, ``kld_per_t`` and ``stratum``,
        and the rule with its counts.
    """
    record: Dict[str, Any] = {
        "late_window_hours": LATE_WINDOW_HOURS, "kl_high_quantile": KL_HIGH_QUANTILE,
        "kl_low_quantile": KL_LOW_QUANTILE, "contraction_window_s": float(window_s),
        "coverage_min": core.CLEAN_COVERAGE, "contraction_column": CONTRACTION_AGE_COLUMN in late.columns,
    }
    if late.empty:
        record["n_late_anchors"] = 0
        return pd.DataFrame(columns=["guid", "stamp", "anchor", "kld_per_t", "stratum"]), record
    clean = late["coverage"].to_numpy(dtype=np.float64) >= core.CLEAN_COVERAGE if "coverage" in late.columns else np.ones(len(late), bool)
    tiers = {"kl": kl_tier(late), "ua": ua_phase(late, window_s=window_s)}
    low, high, record["matched"] = matched_band(late, clean)
    kl = late["kld_per_t"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        tiers["matched"] = np.where((kl >= low) & (kl <= high), KL_MATCHED, "").astype(object)
    pieces = []
    for strata in tiers.values():
        for name in np.unique(strata[strata != ""]):
            keep = clean & (strata == name)
            pieces.append(late.loc[keep, ["guid", "stamp", "anchor", "kld_per_t"]].assign(stratum=str(name)))
    candidates = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["guid", "stamp", "anchor", "kld_per_t", "stratum"])
    candidates = candidates.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    record.update({
        "n_late_anchors": int(len(late)),
        "n_candidates_by_stratum": {str(k): int(v) for k, v in candidates["stratum"].value_counts().items()},
        "n_late_segments": int(late.groupby(["guid", "stamp"]).ngroups),
    })
    return candidates, record


def select_strata_segments(
    labelled: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    candidates: pd.DataFrame,
    *,
    cap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Draw the late segments that hold both KL strata, class-balanced and seeded.

    Among a recording's eligible segments, only those holding the most strata stay eligible, so a
    recording's one drawn segment carries its contraction and matched anchors whenever any of its
    late segments does; the middle of those is then drawn.

    Args:
        labelled: The labelled segments (``attribution_pass.labelled_segments``).
        index_map: The dataset listing.
        candidates: From :func:`strata_candidates`.
        cap: The segment cap.
        seed: The draw seed.

    Returns:
        ``(rows, accounting)`` as :func:`~.attribution_pass.select_segments` returns them.
    """
    held = candidates.groupby(["guid", "stamp"])["stratum"].agg(set) if len(candidates) else pd.Series(dtype=object)
    coverage = {key: len(names) for key, names in held.items() if {KL_HIGH, KL_LOW} <= names}
    best: Dict[str, int] = {}
    for (guid, _stamp), count in coverage.items():
        best[guid] = max(best.get(guid, 0), count)
    eligible = {key for key, count in coverage.items() if count == best[key[0]]}
    keep = np.asarray([(str(g), epoch_stamp(e)) in eligible for g, e in zip(labelled["guid"], labelled["epoch"])], dtype=bool)
    return attribution_pass.select_segments(
        labelled[keep] if len(labelled) else labelled, index_map, cap=cap, seed=seed,
        segments_per_recording=SEGMENTS_PER_RECORDING,
    )


# =============================================================================
# The pass
# =============================================================================
def _anchor_lookup(late: pd.DataFrame) -> Dict[Tuple[str, Optional[int], int], int]:
    """``{(guid, stamp, anchor): row position}`` over the late per-anchor rows."""
    return {key: position for position, key in enumerate(zip(late["guid"], late["stamp"], late["anchor"].astype(int)))}


def join_anchor_values(frame: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """Add the collection pass's $K_t$, forecast gain and contraction age at each row's anchor."""
    if frame.empty:
        return frame
    lookup = _anchor_lookup(late)
    position = np.asarray([lookup.get((str(g), epoch_stamp(e), int(a)), -1)
                           for g, e, a in zip(frame["guid"], frame["epoch"], frame["anchor"])])
    found = position >= 0
    gain = gain_column(late)
    for target, source in (("anchor_kld_per_t", "kld_per_t"), ("anchor_gain", gain), ("anchor_contraction_age_s", CONTRACTION_AGE_COLUMN)):
        values = np.full(len(frame), np.nan)
        if source is not None and source in late.columns:
            values[found] = late[source].to_numpy(dtype=np.float64)[position[found]]
        frame[target] = values
    return frame


def run_strata(
    task: Any,
    loader: Any,
    labelled: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    cell: core.CellBinding,
    *,
    per_anchor: Optional[pd.DataFrame],
    anchor_vectors: Optional[Mapping[str, np.ndarray]],
    cap: int,
    seed: int,
    window_s: float,
    directory: Path,
    lag_seconds: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
    channel_groups: Mapping[str, Mapping[str, np.ndarray]],
    n_steps: int,
    caveat: str,
) -> Dict[str, Any]:
    """Select the late strata, attribute them, compare them, and write the tables and figures.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        labelled: The labelled segments.
        index_map: The dataset listing.
        cell: The cell binding.
        per_anchor: The collection pass's per-anchor table, or ``None``.
        anchor_vectors: Its row-aligned lag maps (``kl_lag_map``, ``attention_lag_map``), or
            ``None``.
        cap: The segment cap.
        seed: The draw, shuffle and bootstrap seed.
        window_s: The contraction window $w$ in seconds.
        directory: The attribution directory.
        lag_seconds: The lag axis.
        lag_bands: The configured lag bands.
        channel_groups: From the channel map.
        n_steps: Integration steps.
        caveat: The one-line figure note.

    Returns:
        The ``strata`` block: the rule, the selection, the cost, the paired results that pass Holm,
        and the files. A run without the per-anchor columns records a skip.
    """
    late = attach_class(late_anchors(per_anchor), labelled)
    if late.empty:
        reason = "no labelled per-anchor rows with guid, epoch, anchor and kld_per_t in the late window"
        logger.warning(f"{core.ANALYSIS_DIRNAME}: strata pass skipped -- {reason}")
        return {"skipped": True, "reason": reason, "files": []}
    candidates, rule = strata_candidates(late, window_s=window_s, seed=seed)
    files: List[str] = []
    figure_paths = [figures.render_figure(
        build_late_behaviour_figure(late, _late_vectors(per_anchor, anchor_vectors, late), lag_seconds=lag_seconds,
                                    window_s=window_s, seed=seed, caveat=caveat),
        directory / LATE_FIGURE,
    )]
    selected, accounting = select_strata_segments(labelled, index_map, candidates, cap=cap, seed=seed)
    gradcam_readouts = READOUTS if cell.dense_latent else ()
    started = time.perf_counter()
    ig_frames, cam_frames = [], []
    ig_vectors: Dict[str, List[np.ndarray]] = {name: [] for name, _ in IG_VECTORS}
    cam_vectors: Dict[str, List[np.ndarray]] = {view: [] for view in gradcam.VIEWS}
    forward_equivalents = 0
    for stratum in STRATA:
        mine = candidates[candidates["stratum"] == stratum]
        wanted = mine.groupby(["guid", "stamp"])["anchor"].agg(lambda s: [int(a) for a in s]).to_dict() if len(mine) else {}
        rows = selected.assign(anchors=[wanted.get((str(g), epoch_stamp(e)), []) for g, e in zip(selected["guid"], selected["epoch"])])
        rows = rows[rows["anchors"].map(len) > 0].reset_index(drop=True)
        if rows.empty:
            continue
        work, _ = attribution_pass.run_segments(
            task, loader, rows, cell, lag_bands=lag_bands, channel_groups=channel_groups, n_steps=n_steps,
            anchors_per_segment=core.ANCHORS_PER_SEGMENT, examples=False,
            readouts=READOUTS, baselines=core.BASELINES,
            with_layer=False, with_ablation=False, with_lag_readout=False, with_top_coordinate=False, with_horizon=False,
            gradcam_readouts=gradcam_readouts,
        )
        forward_equivalents += int(work.forward_equivalents)
        frame = attribution_pass.rows_frame(work)
        if len(frame):
            stacked = attribution_pass.stack_vectors(work)
            ig_frames.append(frame.assign(stratum=stratum))
            for name, _ in IG_VECTORS:
                ig_vectors[name].append(stacked[name])
        cams, profiles = attribution_pass.gradcam_frame(work, lag_seconds)
        if len(cams):
            cam_frames.append(cams.assign(stratum=stratum))
            for view in gradcam.VIEWS:
                cam_vectors[view].append(profiles[view])
    elapsed = time.perf_counter() - started
    rows = join_anchor_values(pd.concat(ig_frames, ignore_index=True), late) if ig_frames else pd.DataFrame()
    vectors = {name: np.concatenate(parts) for name, parts in ig_vectors.items() if parts}
    cam_rows = join_anchor_values(pd.concat(cam_frames, ignore_index=True), late) if cam_frames else pd.DataFrame()
    cam_profiles = {view: np.concatenate(parts) for view, parts in cam_vectors.items() if parts}
    if len(rows):
        rows.to_csv(directory / ROWS_FILENAME, index=False)
        np.savez_compressed(directory / VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64),
                            row_stratum=rows["stratum"].astype(str).to_numpy(dtype=str), **attribution_pass.row_identity(rows), **vectors)
        files += [ROWS_FILENAME, VECTORS_FILENAME]
    if len(cam_rows):
        cam_rows.to_csv(directory / GRADCAM_ROWS_FILENAME, index=False)
        np.savez_compressed(directory / GRADCAM_VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64),
                            row_stratum=cam_rows["stratum"].astype(str).to_numpy(dtype=str), **attribution_pass.row_identity(cam_rows), **cam_profiles)
        files += [GRADCAM_ROWS_FILENAME, GRADCAM_VECTORS_FILENAME]
    metrics = recording_metrics(rows, vectors, cam_rows, lag_seconds=lag_seconds)
    stats = paired_tests(metrics, seed=seed)
    metrics.to_csv(directory / RECORDINGS_FILENAME, index=False)
    stats.to_csv(directory / STATS_FILENAME, index=False)
    files += [RECORDINGS_FILENAME, STATS_FILENAME]
    for contrast in CONTRASTS:
        figure_paths.append(figures.render_figure(
            build_contrast_figure(contrast, "ig", rows, vectors, lag_seconds=lag_seconds, seed=seed, caveat=caveat),
            directory / IG_FIGURE.format(contrast=contrast),
        ))
        if len(cam_rows):
            figure_paths.append(figures.render_figure(
                build_contrast_figure(contrast, "gradcam", cam_rows, cam_profiles, lag_seconds=lag_seconds, seed=seed, caveat=caveat),
                directory / GRADCAM_FIGURE.format(contrast=contrast),
            ))
    figure_paths.append(figures.render_figure(build_paired_figure(metrics, stats, caveat=caveat), directory / PAIRED_FIGURE))
    matched_stats, matched_pairwise = matched_class_tests(metrics, seed=seed)
    matched_stats.to_csv(directory / MATCHED_STATS_FILENAME, index=False)
    matched_pairwise.to_csv(directory / MATCHED_PAIRWISE_FILENAME, index=False)
    files += [MATCHED_STATS_FILENAME, MATCHED_PAIRWISE_FILENAME]
    figure_paths.append(figures.render_figure(
        build_matched_figure(rows, vectors, cam_rows, cam_profiles, metrics, matched_stats, band=rule.get("matched", {}),
                             lag_seconds=lag_seconds, seed=seed, caveat=caveat),
        directory / MATCHED_FIGURE,
    ))
    figure_paths.append(figures.render_figure(
        build_onset_figure(rows, vectors, cam_rows, cam_profiles, seed=seed, caveat=caveat,
                           delay_steps=int(round(float(np.asarray(lag_seconds)[0]) / float(SECONDS_PER_STEP))) if len(lag_seconds) else 0),
        directory / ONSET_FIGURE,
    ))
    files += [Path(path).name for path in figure_paths]
    # A result the strata produce by construction is a check, kept in the CSV and out of the list.
    findings = stats["significant"].astype(bool) & ~stats["by_construction"].astype(bool) if len(stats) else None
    significant = stats.loc[findings, ["contrast", "class", "metric"]] if len(stats) else pd.DataFrame()
    level_check = "anchor:kld_per_t"
    matched_significant = (
        matched_stats.loc[matched_stats["significant"].astype(bool) & (matched_stats["metric"] != level_check), "metric"].tolist()
        if len(matched_stats) else []
    )
    # The matched band was cut over every late recording but is tested on the drawn ones only.
    matched_rule = dict(rule.get("matched", {}))
    matched_rule["n_drawn_recordings_by_class"] = _recording_counts(metrics[metrics["stratum"] == KL_MATCHED]) if len(metrics) else {}
    matched_rule["level_check_p_holm"] = (
        float(matched_stats.loc[matched_stats["metric"] == level_check, "p_holm"].iloc[0])
        if len(matched_stats) and (matched_stats["metric"] == level_check).any() else None
    )
    if matched_rule.get("low_nats") is not None and not matched_rule["high_nats"] > matched_rule["low_nats"]:
        logger.warning(f"{core.ANALYSIS_DIRNAME}: the matched K_t band collapsed to one value; most late K_t are tied")
    rule["matched"] = matched_rule
    logger.info(
        f"{core.ANALYSIS_DIRNAME}: strata pass attributed {len(selected)} late segment(s) at {len(rows)} IG row(s) "
        f"in {elapsed:.1f} s; {len(significant)} paired result(s) pass Holm"
    )
    return {
        "rule": rule,
        "selection": accounting,
        "plan": {
            "cap": int(cap), "seed": int(seed), "segments_per_recording": SEGMENTS_PER_RECORDING,
            "anchors_per_segment_per_stratum": core.ANCHORS_PER_SEGMENT, "ig_steps": int(n_steps),
            "readouts": list(READOUTS), "baselines": list(core.BASELINES),
            "gradcam_readouts": list(gradcam_readouts), "contrasts": {k: list(v) for k, v in CONTRASTS.items()},
        },
        "n_recordings_by_class": _recording_counts(metrics),
        "significant": significant.to_dict(orient="records"),
        "matched_significant": matched_significant,
        "test": "paired: Wilcoxon signed-rank of A - B per class, Holm within contrast x class x method:readout; "
                "across classes: Kruskal-Wallis of the paired differences, Holm within contrast x method:readout; "
                "matched: Kruskal-Wallis across classes at one pooled K_t band, Holm within method:readout, "
                "Mann-Whitney U and Cliff's delta per class pair",
        "cost": core.cost_record(
            elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
            n_forward_equivalents=forward_equivalents, device=getattr(task, "device", None),
        ),
        "files": files,
    }


def _recording_counts(metrics: pd.DataFrame) -> Dict[str, int]:
    """Recordings per class in the per-recording table."""
    if metrics.empty:
        return {}
    return {str(k): int(v) for k, v in metrics.drop_duplicates("guid")[labels.CLASS_COLUMN].value_counts().items()}


def _late_vectors(
    per_anchor: Optional[pd.DataFrame], anchor_vectors: Optional[Mapping[str, np.ndarray]], late: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """The per-anchor lag maps of the late rows, row-aligned with ``late``; empty when absent."""
    if per_anchor is None or not anchor_vectors:
        return {}
    out = {}
    for name in ("attention_lag_map", "kl_lag_map"):
        matrix = anchor_vectors.get(name)
        if matrix is not None and len(matrix) == len(per_anchor):
            out[name] = np.asarray(matrix)[late["row"].to_numpy()].astype(np.float64)
    return out


# =============================================================================
# Per-recording metrics and paired tests
# =============================================================================
def anchor_metrics(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    cam_rows: pd.DataFrame,
    *,
    lag_seconds: np.ndarray,
) -> pd.DataFrame:
    r"""One row per attributed anchor and stratum, one column per scalar metric.

    Columns are named ``ig:<readout>:<metric>``, ``gradcam:<readout>:<metric>`` and
    ``anchor:<metric>``. ``source_total``, ``source_abs_total`` and ``lag_centroid_s`` come from
    the ``source_null`` rows. ``source_share`` comes from the ``all_zero`` rows, the only baseline
    under which the target streams move:
    $\sum|a_{\mathrm{src}}| / (\sum|a_{\mathrm{src}}| + \sum|a_{\mathrm{tgt}}|)$, the share of the
    unsigned attribution that lands on the source (UA) rather than on the FHR history. Under
    ``source_null`` it would be $1$ by construction.

    Returns:
        ``guid``, class, ``stratum``, ``epoch``, ``anchor`` and the metric columns.
    """
    key = ["guid", labels.CLASS_COLUMN, "stratum", "epoch", "anchor"]
    pieces: List[pd.DataFrame] = []
    if len(rows):
        anchor = rows.drop_duplicates(key)[key + ["anchor_kld_per_t", "anchor_gain"]].rename(
            columns={"anchor_kld_per_t": "anchor:kld_per_t", "anchor_gain": "anchor:gain"})
        pieces.append(anchor)
        for readout in READOUTS:
            keep = ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy()
            if keep.any():
                part = rows.loc[keep, key].copy()
                part[f"ig:{readout}:source_total"] = rows.loc[keep, "source_total"].to_numpy(dtype=np.float64)
                part[f"ig:{readout}:source_abs_total"] = rows.loc[keep, "source_abs_total"].to_numpy(dtype=np.float64)
                profile, _ = gradcam.normalise(np.abs(np.asarray(vectors["lag_profile"], dtype=np.float64)[keep]))
                part[f"ig:{readout}:lag_centroid_s"] = gradcam.centroid(profile, lag_seconds)
                pieces.append(part)
            keep = ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_ALL_ZERO)).to_numpy()
            if keep.any():
                part = rows.loc[keep, key].copy()
                source_abs = rows.loc[keep, "source_abs_total"].to_numpy(dtype=np.float64)
                target_abs = rows.loc[keep, "target_abs_total"].to_numpy(dtype=np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    part[f"ig:{readout}:source_share"] = source_abs / (source_abs + target_abs)
                pieces.append(part)
    if len(cam_rows):
        for readout in READOUTS:
            keep = (cam_rows["readout"] == readout).to_numpy()
            if not keep.any():
                continue
            names = [f"{view}_{stat}" for view in gradcam.VIEWS for stat in ("centroid_s", "total")]
            pieces.append(cam_rows.loc[keep, key + names].rename(columns={n: f"gradcam:{readout}:{n}" for n in names}))
    if not pieces:
        return pd.DataFrame(columns=key)
    merged = pieces[0]
    for part in pieces[1:]:
        merged = merged.merge(part, on=key, how="outer")
    return merged


def recording_metrics(
    rows: pd.DataFrame, vectors: Mapping[str, np.ndarray], cam_rows: pd.DataFrame, *, lag_seconds: np.ndarray
) -> pd.DataFrame:
    """The anchor metrics averaged per recording and stratum: one row per ``(guid, stratum)``."""
    anchors = anchor_metrics(rows, vectors, cam_rows, lag_seconds=lag_seconds)
    if anchors.empty:
        return pd.DataFrame(columns=["guid", labels.CLASS_COLUMN, "stratum"])
    value_columns = [c for c in anchors.columns if c not in ("guid", labels.CLASS_COLUMN, "stratum", "epoch", "anchor")]
    grouped = anchors.groupby(["guid", "stratum"], as_index=False)
    means = grouped[value_columns].mean()
    means[labels.CLASS_COLUMN] = grouped[labels.CLASS_COLUMN].first()[labels.CLASS_COLUMN].to_numpy()
    means["n_anchors"] = grouped.size()["size"].to_numpy()
    return means


def paired_frame(metrics: pd.DataFrame, contrast: str) -> pd.DataFrame:
    """One row per recording holding both strata of ``contrast``: ``<metric>__a``, ``<metric>__b``."""
    first, second = CONTRASTS[contrast]
    a = metrics[metrics["stratum"] == first].set_index("guid")
    b = metrics[metrics["stratum"] == second].set_index("guid")
    shared = a.index.intersection(b.index)
    columns = [c for c in metrics.columns if c not in ("guid", "stratum", labels.CLASS_COLUMN, "n_anchors")]
    out = pd.DataFrame({labels.CLASS_COLUMN: a.loc[shared, labels.CLASS_COLUMN]}, index=shared)
    for column in columns:
        out[f"{column}__a"] = a.loc[shared, column].to_numpy(dtype=np.float64)
        out[f"{column}__b"] = b.loc[shared, column].to_numpy(dtype=np.float64)
    return out


def _family(metric: str) -> str:
    """The Holm family of a metric name: its method and readout, or the ``anchor`` metric itself.

    Each ``anchor:`` metric is a family of its own, because ``anchor:kld_per_t`` is a check (the
    KL tiers separate it by construction; the matched band must not), and a check sharing a Holm
    family would have its $p$ moved by an unrelated readout.
    """
    parts = metric.split(":")
    return metric if parts[0] == "anchor" else ":".join(parts[:2])


def _by_construction(contrast: str, metric: str) -> bool:
    r"""Whether a paired result follows from how the strata are cut, and so is a check, not a finding.

    The ``kl`` tiers are cut on $K_t$, so $K_t$ itself separates them, and so do the IG totals of
    $K_t$, which sum to about $K_t - K_{\mathrm{null}}$.
    """
    return contrast == "kl" and metric in (
        "anchor:kld_per_t", f"ig:{core.READOUT_KLD}:source_total", f"ig:{core.READOUT_KLD}:source_abs_total",
    )


def _bootstrap_mean(values: np.ndarray, seed: int) -> Tuple[float, float]:
    """The :data:`CONFIDENCE` percentile interval of the mean of ``values``."""
    interval = shared_stats.bootstrap_ci(values, confidence=CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=seed)
    return float(interval["lo"]), float(interval["hi"])


def paired_tests(metrics: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """The paired stratum contrasts per class, and whether the contrast differs across classes.

    Args:
        metrics: From :func:`recording_metrics`.
        seed: The bootstrap seed.

    Returns:
        One row per contrast, metric and class (``test = 'wilcoxon'``), with the per-class means of
        A and B, the mean paired difference A $-$ B with its bootstrap interval, $p$ and Holm $p$;
        and one row per contrast and metric with ``class = 'across_classes'`` and
        ``test = 'kruskal-wallis'`` on the paired differences.
    """
    records: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=["contrast", "metric", "family", "class", "test", "p_value", "p_holm", "significant", "by_construction"])
    order = labels.ordered_groups(sorted(metrics[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN)
    names = [c for c in metrics.columns if c not in ("guid", "stratum", labels.CLASS_COLUMN, "n_anchors")]
    for contrast, (first, second) in CONTRASTS.items():
        paired = paired_frame(metrics, contrast)
        if paired.empty:
            continue
        for metric in names:
            differences: Dict[str, np.ndarray] = {}
            for name in order:
                own = paired[paired[labels.CLASS_COLUMN] == name]
                a, b = own[f"{metric}__a"].to_numpy(), own[f"{metric}__b"].to_numpy()
                test = shared_stats.wilcoxon_paired(a, b, label_left=first, label_right=second)
                finite = np.isfinite(a) & np.isfinite(b)
                diff = a[finite] - b[finite]
                differences[name] = diff
                lo, hi = _bootstrap_mean(diff, seed) if diff.size else (np.nan, np.nan)
                records.append({
                    "contrast": contrast, "metric": metric, "family": _family(metric), "class": name, "test": "wilcoxon",
                    "stratum_a": first, "stratum_b": second, "n_pairs": int(diff.size),
                    "mean_a": float(np.mean(a[finite])) if diff.size else np.nan,
                    "mean_b": float(np.mean(b[finite])) if diff.size else np.nan,
                    "mean_diff": float(np.mean(diff)) if diff.size else np.nan, "diff_ci_lo": lo, "diff_ci_hi": hi,
                    "statistic": test["statistic"], "p_value": test["p_value"],
                })
            testable = {n: d for n, d in differences.items() if d.size >= shared_stats.MIN_GROUP_SIZE}
            omnibus = shared_stats.kruskal_across_groups(testable)
            records.append({
                "contrast": contrast, "metric": metric, "family": _family(metric), "class": "across_classes",
                "test": "kruskal-wallis", "stratum_a": first, "stratum_b": second,
                "n_pairs": int(sum(d.size for d in testable.values())),
                "statistic": omnibus["statistic"], "p_value": omnibus["p_value"],
            })
    if not records:
        return pd.DataFrame(columns=["contrast", "metric", "family", "class", "test", "p_value", "p_holm", "significant", "by_construction"])
    frame = pd.DataFrame(records)
    frame["by_construction"] = [_by_construction(c, m) for c, m in zip(frame["contrast"], frame["metric"])]
    frame["p_holm"] = np.nan
    for _, index in frame.groupby(["contrast", "class", "family"]).groups.items():
        frame.loc[index, "p_holm"] = shared_stats.holm_adjust(frame.loc[index, "p_value"].tolist())
    frame["significant"] = np.isfinite(frame["p_holm"]) & (frame["p_holm"] < ALPHA)
    return frame


def matched_class_tests(metrics: pd.DataFrame, *, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The class tests of the KL-matched stratum, with the cohort's own rule.

    Every per-recording metric of the ``kl_matched`` stratum goes through
    :func:`~.class_contrast.class_tests`: Kruskal--Wallis across classes, Holm within one method
    and readout, then Mann--Whitney $U$ and Cliff's $\\delta$ per class pair. ``anchor:kld_per_t``
    is its own family and is the check on the match: it should **not** differ.

    Args:
        metrics: From :func:`recording_metrics`.
        seed: The bootstrap seed.

    Returns:
        ``(stats, pairwise)`` as :func:`~.class_contrast.class_tests` returns them.
    """
    matched = metrics[metrics["stratum"] == KL_MATCHED].set_index("guid") if len(metrics) else pd.DataFrame()
    if matched.empty:
        return pd.DataFrame(columns=["family", "metric", "p_holm", "significant"]), pd.DataFrame()
    names = [c for c in matched.columns if c not in ("stratum", labels.CLASS_COLUMN, "n_anchors") and matched[c].notna().any()]
    families: Dict[str, List[str]] = {}
    for name in names:
        families.setdefault(_family(name), []).append(name)
    return class_contrast.class_tests(matched, families, seed=seed)


# =============================================================================
# Figures
# =============================================================================
def _curves(frame: pd.DataFrame, matrix: np.ndarray) -> Tuple[List[str], List[str], np.ndarray]:
    """Per-recording mean curves: ``(guids, classes, (R, W))``, ``NaN`` ignored.

    The arithmetic of :func:`~.class_contrast.per_recording` through one ``groupby``, because the
    late window can hold hundreds of thousands of anchors.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    guids = frame["guid"].astype(str).to_numpy()
    if not len(guids):
        return [], [], np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0))
    table = pd.DataFrame(matrix)
    table["guid"] = guids
    means = table.groupby("guid", sort=False).mean()
    classes = pd.Series(frame[labels.CLASS_COLUMN].astype(str).to_numpy(), index=guids).groupby(level=0, sort=False).first()
    return list(means.index), classes.reindex(means.index).tolist(), means.to_numpy()


def _paired_curves(frame: pd.DataFrame, matrix: np.ndarray, first: str, second: str) -> Dict[str, np.ndarray]:
    """Per class, the per-recording curve of stratum A minus stratum B, over recordings with both."""
    strata = frame["stratum"].to_numpy()
    guids_a, classes_a, curves_a = _curves(frame[strata == first], matrix[strata == first])
    guids_b, _, curves_b = _curves(frame[strata == second], matrix[strata == second])
    where_b = {guid: i for i, guid in enumerate(guids_b)}
    shared = [(i, where_b[g]) for i, g in enumerate(guids_a) if g in where_b]
    if not shared:
        return {}
    diffs = np.stack([curves_a[i] - curves_b[j] for i, j in shared])
    return class_contrast.by_class([classes_a[i] for i, _ in shared], diffs)


def _panels(method: str, rows: pd.DataFrame, vectors: Mapping[str, np.ndarray], lag_seconds: np.ndarray):
    """The rows of a contrast figure: ``(label, ylabel, keep mask, matrix, x, xlabel)``."""
    out = []
    for readout in READOUTS:
        keep = (rows["readout"] == readout).to_numpy() if len(rows) else np.zeros(0, bool)
        short = core.READOUT_SHORT.get(readout, readout)
        if method == "ig":
            unit = str(rows.loc[keep, "unit"].iloc[0]) if keep.any() else ""
            for name, stream in IG_VECTORS:
                baseline = IG_VECTOR_BASELINE[stream]
                matrix = np.asarray(vectors.get(name, np.zeros((len(rows), len(lag_seconds)))), dtype=np.float64)
                out.append((f"{short}, IG {stream} ({baseline})", f"attribution ({unit})", keep & (rows["baseline"] == baseline).to_numpy(),
                            matrix, np.asarray(lag_seconds)[: matrix.shape[1]], COEFFICIENT_LAG_AXIS_LABEL))
        else:
            for view in gradcam.VIEWS:
                matrix, x = class_contrast.view_axis(view, np.asarray(vectors[view], dtype=np.float64), lag_seconds)
                xlabel = "offset before the anchor (min)" if view == "target" else COEFFICIENT_LAG_AXIS_LABEL
                out.append((f"{short}, {gradcam.VIEW_TITLES[view]}", "Grad-CAM share", keep, matrix, x, xlabel))
    return out


def build_contrast_figure(
    contrast: str,
    method: str,
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    seed: int,
    caveat: str,
) -> Any:
    r"""One contrast, one method: the two strata per class, and their paired difference by class.

    One row per readout and profile: for ``ig`` the source and target attribution by lag, for
    ``gradcam`` the three views. One column per class draws the mean over recordings of stratum A
    (black) and stratum B (blue) with $95\%$ bootstrap bands. The last column draws, per class, the
    mean of the per-recording difference A $-$ B with its band; the rug under it marks where a
    band excludes zero.

    Args:
        contrast: A key of :data:`CONTRASTS`.
        method: ``'ig'`` or ``'gradcam'``.
        rows: The method's rows, with ``stratum`` and the class column.
        vectors: Their row-aligned profiles.
        lag_seconds: The lag axis.
        seed: The bootstrap seed.
        caveat: The one-line figure note.

    Returns:
        The figure.
    """
    first, second = CONTRASTS[contrast]
    order = labels.ordered_groups(sorted(rows[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN) if len(rows) else []
    panels = _panels(method, rows, vectors, lag_seconds) if len(rows) else []
    n_cols = len(order) + 1
    figure, axes = figures.new_figure(max(len(panels), 1), max(n_cols, 2), height_per_row=1.7, width=shared_figures.FIGURE_WIDTH * 1.25)
    if not panels:
        core._empty(axes[0, 0], f"{CONTRAST_TITLES[contrast]}: no strata rows")
        return figure
    class_colours = figures.group_colors(order)
    # The K_t level each class column was drawn at, per stratum: the tiers are per segment, so the
    # same tier name can mean different nats in different classes.
    levels = rows[rows["stratum"].isin([first, second])].groupby([labels.CLASS_COLUMN, "stratum"])["anchor_kld_per_t"].median()
    titles = {
        name: f"{name}\nmedian $K_t$ A / B = {levels.get((name, first), np.nan):.2g} / {levels.get((name, second), np.nan):.2g}"
        for name in order
    }
    for index, (label, ylabel, keep, matrix, x, xlabel) in enumerate(panels):
        # An x label where the axis changes below this row, as well as on the bottom row.
        if index == len(panels) - 1 or panels[index + 1][5] != xlabel:
            row_xlabel = xlabel
        else:
            row_xlabel = ""
        frame = rows[keep].reset_index(drop=True)
        sub = matrix[keep]
        in_contrast = frame["stratum"].isin([first, second]).to_numpy()
        frame, sub = frame[in_contrast].reset_index(drop=True), sub[in_contrast]
        shared_axis = axes[index, 0]
        for col, name in enumerate(order):
            ax = axes[index, col]
            of_class = (frame[labels.CLASS_COLUMN].astype(str) == name).to_numpy()
            drawn = False
            for stratum, colour in zip((first, second), STRATUM_COLOURS):
                mask = of_class & (frame["stratum"] == stratum).to_numpy()
                if not mask.any():
                    continue
                _guids, _classes, curves = _curves(frame[mask], sub[mask])
                mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
                ax.plot(x, mean, color=colour, linewidth=figures.LINE_EMPHASIS, label=f"{STRATUM_LABELS[stratum]} (n={len(curves)})")
                ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0.0)
                drawn = True
            if col:
                ax.sharey(shared_axis)
            _finish(ax, drawn, title=titles[name] if index == 0 else "", xlabel=row_xlabel,
                    ylabel=f"{label}\n{ylabel}" if col == 0 else "", legend=index == 0)
        ax = axes[index, len(order)]
        groups = _paired_curves(frame, sub, first, second)
        for slot, (name, curves) in enumerate(groups.items()):
            mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
            colour = class_colours.get(name, figures.COLOR_GRAY)
            ax.plot(x, mean, color=colour, linewidth=figures.LINE_EMPHASIS, label=f"{name} (n={len(curves)})")
            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0.0)
            class_contrast.mark_excluding_zero(ax, x, lo, hi, colour=colour, slot=slot)
        ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN)
        class_contrast.reserve_rug(ax, len(groups))
        _finish(ax, bool(groups), title="paired difference, A − B" if index == 0 else "",
                xlabel=row_xlabel, ylabel="", legend=index == 0)
    figure.suptitle(
        f"{'Integrated gradients' if method == 'ig' else 'Grad-CAM'} by anchor stratum, last {LATE_WINDOW_HOURS:g} h: "
        f"{CONTRAST_TITLES[contrast]} (A = {STRATUM_LABELS[first]}, B = {STRATUM_LABELS[second]})",
        fontsize=figures.FONT_NOTE,
    )
    figures.caveat_note(figure, caveat)
    return figure


def _finish(ax: Any, drawn: bool, *, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    """Title, labels and a compact legend, or the empty note."""
    if not drawn:
        core._empty(ax, title)
        return
    figures.style_axes(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", fontsize=figures.FONT_TINY, handlelength=1.0)


def _p_text(stats: pd.DataFrame, contrast: str, metric: str, name: str) -> str:
    """``p=0.012*`` for one class's paired test, starred when it passes Holm, or ``''``."""
    if stats.empty:
        return ""
    hit = stats[(stats["contrast"] == contrast) & (stats["metric"] == metric) & (stats["class"] == name)]
    if hit.empty or not np.isfinite(hit["p_value"].iloc[0]):
        return ""
    return f"p={hit['p_value'].iloc[0]:.2g}" + ("*" if bool(hit["significant"].iloc[0]) else "")


def build_paired_figure(metrics: pd.DataFrame, stats: pd.DataFrame, *, caveat: str) -> Any:
    r"""Per recording, stratum A against stratum B on the scalar metrics of :data:`PAIRED_METRICS`.

    One row per contrast, one panel per metric. Per class, each recording is a thin line from its
    stratum-A value (left) to its stratum-B value (right); the bold markers are the class means
    with $95\%$ bootstrap intervals. The tick label under each class carries the Wilcoxon
    signed-rank $p$, starred when it passes Holm in its family. The panel title carries the
    Kruskal--Wallis $p$ of the paired differences across classes.

    Returns:
        The figure.
    """
    n_cols = len(PAIRED_METRICS)
    figure, axes = figures.new_figure(len(CONTRASTS), n_cols, height_per_row=2.0, width=figures.WINDOWS_FIGURE_WIDTH)
    order = labels.ordered_groups(sorted(metrics[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN) if len(metrics) else []
    colours = figures.group_colors(order)
    for row, (contrast, (first, second)) in enumerate(CONTRASTS.items()):
        paired = paired_frame(metrics, contrast) if len(metrics) else pd.DataFrame()
        for col, (metric, ylabel) in enumerate(PAIRED_METRICS):
            ax = axes[row, col]
            if paired.empty or f"{metric}__a" not in paired.columns:
                core._empty(ax, f"{CONTRAST_TITLES[contrast]}")
                continue
            for position, name in enumerate(order):
                own = paired[paired[labels.CLASS_COLUMN] == name]
                a, b = own[f"{metric}__a"].to_numpy(), own[f"{metric}__b"].to_numpy()
                finite = np.isfinite(a) & np.isfinite(b)
                left, right = position - 0.18, position + 0.18
                for va, vb in zip(a[finite], b[finite]):
                    ax.plot((left, right), (va, vb), color=colours[name], alpha=0.3, linewidth=figures.LINE_HAIRLINE)
                for x, values in ((left, a[finite]), (right, b[finite])):
                    if values.size:
                        lo, hi = _bootstrap_mean(values, 0)
                        ax.errorbar(x, float(values.mean()), yerr=[[max(values.mean() - lo, 0)], [max(hi - values.mean(), 0)]],
                                    fmt="o", color=colours[name], markersize=3.0, elinewidth=figures.LINE_REGULAR,
                                    markeredgecolor=figures.COLOR_BLACK, markeredgewidth=0.3)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([f"{name}\n{_p_text(stats, contrast, metric, name)}" for name in order])
            ax.set_xlim(-0.6, len(order) - 0.4)
            figures.style_axes(ax)
            across = stats[(stats["contrast"] == contrast) & (stats["metric"] == metric) & (stats["class"] == "across_classes")] if len(stats) else stats
            kw = f"KW p={across['p_value'].iloc[0]:.2g}" if len(across) and np.isfinite(across["p_value"].iloc[0]) else "KW: n/a"
            ax.set_title(kw)
            ax.set_ylabel(f"{CONTRAST_TITLES[contrast]}\n{ylabel}" if col == 0 else ylabel)
    figure.suptitle(
        f"Paired strata per recording, last {LATE_WINDOW_HOURS:g} h: per class, left marker A "
        f"(high $K_t$ / in contraction), right marker B (low $K_t$ / between contractions)",
        fontsize=figures.FONT_NOTE,
    )
    figures.caveat_note(figure, caveat)
    return figure


def build_matched_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    cam_rows: pd.DataFrame,
    cam_profiles: Mapping[str, np.ndarray],
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    band: Mapping[str, Any],
    lag_seconds: np.ndarray,
    seed: int,
    caveat: str,
) -> Any:
    r"""The classes compared at one coupling level: the ``kl_matched`` anchors only.

    Every anchor here has $K_t$ inside one band of the pooled distribution, so a class difference is
    not a difference of $K_t$ level. Profile rows, each a class-mean panel and a class-difference
    panel with $95\%$ bootstrap bands over recordings and the significance rug: the unsigned IG lag
    share of $K_t$ and of the gap under ``source_null``, and the Grad-CAM share of $K_t$ in the three
    views. The last two rows are the per-recording scalars of :data:`MATCHED_METRICS`, titled with
    the Holm-adjusted Kruskal--Wallis $p$; the first of them, $K_t$ itself, checks the match.

    Returns:
        The figure.
    """
    specs: List[Tuple[str, pd.DataFrame, np.ndarray, np.ndarray, str]] = []
    for readout in READOUTS:
        keep = ((rows["stratum"] == KL_MATCHED) & (rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy() if len(rows) else np.zeros(0, bool)
        if keep.any():
            shares = gradcam.normalise(np.abs(np.asarray(vectors["lag_profile"], dtype=np.float64)[keep]))[0]
            specs.append((f"{core.READOUT_SHORT.get(readout, readout)}: IG unsigned lag share", rows[keep], shares,
                          np.asarray(lag_seconds)[: shares.shape[1]], COEFFICIENT_LAG_AXIS_LABEL))
    keep = ((cam_rows["stratum"] == KL_MATCHED) & (cam_rows["readout"] == core.READOUT_KLD)).to_numpy() if len(cam_rows) else np.zeros(0, bool)
    for view in (gradcam.VIEWS if keep.any() else ()):
        matrix, x = class_contrast.view_axis(view, np.asarray(cam_profiles[view], dtype=np.float64)[keep], lag_seconds)
        specs.append((f"{core.READOUT_SHORT[core.READOUT_KLD]}: Grad-CAM {gradcam.VIEW_TITLES[view]}", cam_rows[keep], matrix, x,
                      "offset before the anchor (min)" if view == "target" else COEFFICIENT_LAG_AXIS_LABEL))
    n_scalar_rows = -(-len(MATCHED_METRICS) // 2)
    figure, axes = figures.new_figure(max(len(specs), 1) + n_scalar_rows, 2, height_per_row=1.6, width=shared_figures.FIGURE_WIDTH)
    matched = metrics[metrics["stratum"] == KL_MATCHED] if len(metrics) else metrics
    notes = {}
    if len(matched) and "anchor:kld_per_t" in matched.columns:
        notes = {str(k): f"median $K_t$={v:.2g}" for k, v in matched.groupby(labels.CLASS_COLUMN)["anchor:kld_per_t"].median().items() if np.isfinite(v)}
    if not specs:
        core._empty(axes[0, 0], "no KL-matched rows")
    for index, (title, frame, matrix, x, xlabel) in enumerate(specs):
        _guids, classes, curves = _curves(frame, matrix)
        groups = class_contrast.by_class(classes, curves)
        class_contrast._class_lines(axes[index, 0], x, groups, seed=seed, title=title, xlabel=xlabel, ylabel="share",
                                    legend=index == 0, notes=notes)
        class_contrast._difference_lines(axes[index, 1], x, groups, seed=seed, title="class differences", xlabel=xlabel,
                                         ylabel="difference of shares", legend=index == 0)
    order = labels.ordered_groups(sorted(matched[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN) if len(matched) else []
    base = max(len(specs), 1)
    for position, (metric, ylabel) in enumerate(MATCHED_METRICS):
        ax = axes[base + position // 2, position % 2]
        if metric not in matched.columns:
            core._empty(ax, ylabel)
            continue
        samples = {name: matched.loc[matched[labels.CLASS_COLUMN] == name, metric].to_numpy(dtype=np.float64) for name in order}
        hit = stats[stats["metric"] == metric] if len(stats) else stats
        p = f"KW p_Holm = {hit['p_holm'].iloc[0]:.2g}" if len(hit) and np.isfinite(hit["p_holm"].iloc[0]) else "KW: not testable"
        figures.violin_panel(ax, samples, title=p, ylabel=ylabel, colors=figures.group_colors(order))
    low, high = band.get("low_nats"), band.get("high_nats")
    span = f"$K_t$ in [{low:.3g}, {high:.3g}] nats" if low is not None and high is not None else "no band"
    figure.suptitle(f"Classes at matched coupling, last {LATE_WINDOW_HOURS:g} h: {span}", fontsize=figures.FONT_NOTE)
    figures.caveat_note(figure, caveat)
    return figure


def onset_locked(profiles: np.ndarray, age_s: np.ndarray, offset_steps: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    r"""Re-index lag profiles on the most recent contraction onset.

    Column $\ell$ of a profile is the source step $\ell + \delta$ stored steps before the anchor,
    and the onset is $a$ seconds before the anchor's endpoint, on the same stored timeline. The
    step at column $\ell$ therefore sits $x = a - 4(\ell + \delta)$ seconds after the onset:
    $x = 0$ is the onset, $x > 0$ is inside or after the contraction, $x < 0$ is before it. Rows
    whose onset lies outside the window the profile covers, or that have no onset, are dropped.

    Args:
        profiles: $(N, L)$ shares by lag.
        age_s: $(N,)$ seconds since the most recent onset.
        offset_steps: $\delta$. Zero for an IG profile, which is indexed by the stored-step offset
            $t_a - s$ itself; the model's causal input delay for a view on the lag attention's own
            axis (the Grad-CAM ``source`` and ``attention`` views), which reads the source
            $\delta$ steps further back.

    Returns:
        ``(aligned, x)``: $(N, 2L - 1)$ with ``NaN`` where a row does not reach, and the axis in
        seconds; rows outside the window are all ``NaN``.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    n_rows, n_lags = profiles.shape
    x = (np.arange(2 * n_lags - 1) - (n_lags - 1)) * float(SECONDS_PER_STEP)
    aligned = np.full((n_rows, 2 * n_lags - 1), np.nan)
    with np.errstate(invalid="ignore"):
        onset_step = np.round(np.asarray(age_s, dtype=np.float64) / float(SECONDS_PER_STEP)) - int(offset_steps)
    lags = np.arange(n_lags)
    for row in np.flatnonzero(np.isfinite(onset_step) & (onset_step >= 0) & (onset_step < n_lags)):
        aligned[row, int(onset_step[row]) - lags + n_lags - 1] = profiles[row]
    return aligned, x


def build_onset_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    cam_rows: pd.DataFrame,
    cam_profiles: Mapping[str, np.ndarray],
    *,
    delay_steps: int = 0,
    seed: int,
    caveat: str,
) -> Any:
    r"""Does the attribution lock onto the contraction? Profiles aligned on the last onset, per class.

    Every attributed anchor whose most recent contraction onset lies inside the lag window, from any
    stratum and counted once, has its unsigned IG lag share and its Grad-CAM shares re-indexed by
    :func:`onset_locked`. One row per profile, one column per class: the class mean over
    recordings with a $95\%$ bootstrap band, and in grey the same mean after the onset ages are
    shuffled among the class's anchors (the onset-shuffle null). A peak near $x = 0$ that the null
    does not show is attribution locked to the contraction onset; a curve that follows the null is
    attribution the lag geometry alone explains.

    The IG profiles are indexed by the stored-step offset itself; the two Grad-CAM views are on the
    lag attention's own axis, which reads the source ``delay_steps`` further back, so they are
    aligned with that offset (:func:`onset_locked`).

    Returns:
        The figure.
    """
    specs: List[Tuple[str, pd.DataFrame, np.ndarray, int]] = []
    key = ["guid", "epoch", "anchor", "readout", "baseline"]
    if len(rows) and "anchor_contraction_age_s" in rows.columns:
        once = ~rows.duplicated(subset=key).to_numpy()
        for readout in READOUTS:
            keep = once & ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy()
            if keep.any():
                shares = gradcam.normalise(np.abs(np.asarray(vectors["lag_profile"], dtype=np.float64)[keep]))[0]
                specs.append((f"{core.READOUT_SHORT.get(readout, readout)}, IG", rows[keep], shares, 0))
    if len(cam_rows) and "anchor_contraction_age_s" in cam_rows.columns:
        once = ~cam_rows.duplicated(subset=["guid", "epoch", "anchor", "readout"]).to_numpy()
        keep = once & (cam_rows["readout"] == core.READOUT_KLD).to_numpy()
        for view in (("attention", "source") if keep.any() else ()):
            specs.append((f"{core.READOUT_SHORT[core.READOUT_KLD]}, Grad-CAM {view}", cam_rows[keep],
                          np.asarray(cam_profiles[view], dtype=np.float64)[keep], int(delay_steps)))
    frames = [frame for _, frame, _, _ in specs]
    order = labels.ordered_groups(sorted({str(c) for f in frames for c in f[labels.CLASS_COLUMN].dropna()}), labels.CLASS_COLUMN)
    figure, axes = figures.new_figure(max(len(specs), 1), max(len(order), 1), height_per_row=1.7, width=shared_figures.FIGURE_WIDTH * 1.1)
    if not specs or not order:
        core._empty(axes[0, 0], "no attributed anchor with a contraction onset in the lag window")
        return figure
    colours = figures.group_colors(order)
    rng = np.random.default_rng(int(seed) + ONSET_NULL_SEED_OFFSET)
    for index, (title, frame, matrix, offset) in enumerate(specs):
        frame = frame.reset_index(drop=True)
        age = frame["anchor_contraction_age_s"].to_numpy(dtype=np.float64)
        aligned, x = onset_locked(matrix, age, offset)
        inside = np.isfinite(aligned).any(axis=1)
        for col, name in enumerate(order):
            ax = axes[index, col]
            mine = inside & (frame[labels.CLASS_COLUMN].astype(str) == name).to_numpy()
            if not mine.any():
                core._empty(ax, name if index == 0 else "")
                continue
            shuffled = age[mine].copy()
            rng.shuffle(shuffled)
            null, _ = onset_locked(matrix[mine], shuffled, offset)
            for curves_of, colour, label, style in ((aligned[mine], colours[name], "aligned", "-"), (null, figures.COLOR_GRAY, "onset-shuffled", "--")):
                _guids, _classes, curves = _curves(frame[mine], curves_of)
                mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
                ax.plot(x, mean, color=colour, linestyle=style, linewidth=figures.LINE_EMPHASIS, label=f"{label} (n={len(curves)})")
                ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0.0)
            ax.axvline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
            _finish(ax, True, title=name if index == 0 else "", xlabel="seconds after the onset" if index == len(specs) - 1 else "",
                    ylabel=f"{title}\nshare" if col == 0 else "", legend=index == 0)
    figure.suptitle(
        f"Attribution aligned on the most recent contraction onset, last {LATE_WINDOW_HOURS:g} h (x = 0: onset; x < 0: before it)",
        fontsize=figures.FONT_NOTE,
    )
    figures.caveat_note(figure, caveat)
    return figure


def _binned_curves(frame: pd.DataFrame, value: np.ndarray, position: np.ndarray, edges: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Per recording, the mean of ``value`` in each bin of ``position``: ``(classes, (R, n_bins))``."""
    bins = np.digitize(position, edges) - 1
    keep = (bins >= 0) & (bins < len(edges) - 1) & np.isfinite(value)
    table = pd.DataFrame({"guid": frame["guid"].to_numpy()[keep], "cls": frame[labels.CLASS_COLUMN].to_numpy()[keep],
                          "bin": bins[keep], "value": value[keep]})
    if table.empty:
        return [], np.zeros((0, len(edges) - 1))
    pivot = table.pivot_table(index="guid", columns="bin", values="value", aggfunc="mean").reindex(columns=range(len(edges) - 1))
    classes = table.groupby("guid")["cls"].first().reindex(pivot.index).astype(str).tolist()
    return classes, pivot.to_numpy(dtype=np.float64)


def _class_band_panel(ax: Any, x: np.ndarray, groups: Mapping[str, np.ndarray], *, seed: int, rug: bool) -> bool:
    """Class means with bootstrap bands, and a rug where a band excludes zero when ``rug``."""
    colours = figures.group_colors(list(groups))
    drawn = False
    for slot, (name, curves) in enumerate(groups.items()):
        if not len(curves):
            continue
        mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
        ax.plot(x, mean, color=colours[name], linewidth=figures.LINE_EMPHASIS, label=f"{name} (n={len(curves)})")
        ax.fill_between(x, lo, hi, color=colours[name], alpha=0.18, linewidth=0.0)
        if rug:
            class_contrast.mark_excluding_zero(ax, x, lo, hi, colour=colours[name], slot=slot)
        drawn = True
    if rug:
        ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN)
        class_contrast.reserve_rug(ax, len(groups))
    return drawn


def build_late_behaviour_figure(
    late: pd.DataFrame,
    lag_maps: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    window_s: float,
    seed: int,
    caveat: str,
) -> Any:
    r"""What the network does over every anchor of the late window, by class, with no gradient.

    **a, b** $K_t$ and the forecast gain against minutes before delivery, in
    :data:`DELIVERY_BIN_MINUTES` bins, delivery on the right. **c, d** The same two readouts
    against seconds since the last contraction onset, in :data:`ONSET_BIN_SECONDS` bins; the shaded
    span is the contraction window $w$. **e** The head-averaged attention by lag at high-$K_t$
    anchors minus low-$K_t$ anchors (per-segment tiers), per recording, then by class. **f** The same
    for the $K_t$ lag map, each anchor's map divided by its sum first, so the difference is one of
    location, not size. All curves are class means over recordings with $95\%$ bootstrap bands.

    Args:
        late: The labelled late anchors (:func:`attach_class`).
        lag_maps: ``attention_lag_map`` and ``kl_lag_map``, row-aligned with ``late``; a missing
            map leaves its panel empty.
        lag_seconds: The lag axis.
        window_s: The contraction window $w$.
        seed: The bootstrap seed.
        caveat: The one-line figure note.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(3, 2, height_per_row=1.8, width=shared_figures.FIGURE_WIDTH)
    frame = late
    gain = gain_column(frame)
    edges_t = np.arange(0.0, LATE_WINDOW_HOURS * 60.0 + DELIVERY_BIN_MINUTES, DELIVERY_BIN_MINUTES)
    edges_c = np.arange(0.0, ONSET_MAX_SECONDS + ONSET_BIN_SECONDS, ONSET_BIN_SECONDS)
    age = frame[CONTRACTION_AGE_COLUMN].to_numpy(dtype=np.float64) if CONTRACTION_AGE_COLUMN in frame.columns else np.full(len(frame), np.nan)
    specs = [
        (axes[0, 0], "kld_per_t", frame["minutes_before_delivery"].to_numpy(), edges_t, "$K_t$ (nats)", "minutes before delivery", "$K_t$ toward delivery"),
        (axes[0, 1], gain, frame["minutes_before_delivery"].to_numpy(), edges_t, "forecast gain (nats)", "minutes before delivery", "Forecast gain toward delivery"),
        (axes[1, 0], "kld_per_t", age, edges_c, "$K_t$ (nats)", "seconds since contraction onset", "$K_t$ around contractions"),
        (axes[1, 1], gain, age, edges_c, "forecast gain (nats)", "seconds since contraction onset", "Forecast gain around contractions"),
    ]
    for index, (ax, column, position, edges, ylabel, xlabel, title) in enumerate(specs):
        if column is None or frame.empty:
            core._empty(ax, title)
            continue
        classes, curves = _binned_curves(frame, frame[column].to_numpy(dtype=np.float64), np.asarray(position, dtype=np.float64), edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        drawn = _class_band_panel(ax, centres, class_contrast.by_class(classes, curves), seed=seed, rug=False)
        _finish(ax, drawn, title=title, xlabel=xlabel, ylabel=ylabel, legend=index == 0)
        if drawn and index < 2:
            ax.invert_xaxis()
        if drawn and index >= 2:
            ax.axvspan(0.0, float(window_s), color=figures.COLOR_LIGHT_GRAY, alpha=0.5, linewidth=0.0, zorder=0)
    tiers = kl_tier(frame)
    for ax, name, title in ((axes[2, 0], "attention_lag_map", "Attention, high − low $K_t$"), (axes[2, 1], "kl_lag_map", "$K_t$ lag share, high − low $K_t$")):
        matrix = lag_maps.get(name)
        if matrix is None or frame.empty:
            core._empty(ax, title)
            continue
        matrix = np.asarray(matrix, dtype=np.float64)
        if name == "kl_lag_map":
            total = matrix.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                matrix = np.where(total > 0.0, matrix / total, np.nan)
        groups = _paired_curves(frame.assign(stratum=tiers), matrix, KL_HIGH, KL_LOW)
        drawn = _class_band_panel(ax, np.asarray(lag_seconds)[: matrix.shape[1]], groups, seed=seed, rug=True)
        _finish(ax, drawn, title=title, xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel="difference of shares", legend=False)
    figure.suptitle(f"Network behaviour over every anchor of the last {LATE_WINDOW_HOURS:g} h before delivery", fontsize=figures.FONT_NOTE)
    figures.caveat_note(figure, caveat)
    return figure


__all__ = [
    "CONTRASTS", "DEFAULT_STRATA_SEGMENTS", "LATE_WINDOW_HOURS", "READOUTS", "STRATA", "STRATA_CAP_NAME",
    "KL_MATCHED", "MATCHED_BANDS", "anchor_metrics", "attach_class", "build_contrast_figure", "build_late_behaviour_figure",
    "build_matched_figure", "build_onset_figure", "build_paired_figure", "matched_band", "matched_class_tests", "onset_locked",
    "join_anchor_values", "kl_tier", "late_anchors", "paired_frame", "paired_tests", "recording_metrics",
    "run_strata", "select_strata_segments", "strata_candidates", "ua_phase",
]
