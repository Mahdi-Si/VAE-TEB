r"""Contraction-conditioned coupling: does the source matter more when the uterus is contracting?

One readout, where the raw pipeline's ``events`` has three, and the two that are gone are gone for
one reason: **they score a bpm waveform**. Deceleration forecasting skill runs a deceleration
detector over each branch's forecast *block*, and the contraction-triggered response averages that
block around a trigger. This model's forecast block is $H \times C_{\mathrm{keep}}$ wavelet-modulus and
phase-harmonic coefficients in the loader's $z$ units, and "a deceleration in coefficient space" is
a new construction rather than a port -- one that would have to define what a deceleration *is* on
a channel axis with no order and no clinical unit, and then defend the definition. Neither is
attempted, and :data:`REMOVED_READOUTS` says so in the emitted record, so a reader of
``summary.json`` meets the absence rather than inferring it from a missing key.

What survives is the half that conditions on **timing** rather than on the forecast's shape, and it
ports unchanged. ``pred_gap`` and $K_t$ are restricted to anchors within ``event_lag_window_s`` of
a detected contraction and compared against count-matched control anchors drawn from the same
recordings.

**This is the per-anchor table's load-bearing consumer, and the reason that table exists.** A
per-segment mean cannot express "the anchors near a contraction", so the contraction timing is
computed in the collection pass -- which is the only pass holding the raw UP trace, since the model
reads the source as decimated coefficients and a contraction exists nowhere in the tables unless
that pass puts it there -- and lands on the per-anchor table as
:data:`~teb_vae.lag_attn_cfs.eval.collect.CONTRACTION_AGE_COLUMN`. The consequence is that this
analysis runs over **every anchor of the split** rather than over the retained samples alone: it
needs no forecast block, no waveform retention and no cap, which is exactly why the two readouts
that did need them are the two that are gone.

**The events themselves are shown, not only counted.** Where the pass retained raw UP, the same
detector is re-run over those traces and every contraction it finds is listed with the segment's
identity and drawn on the trace, beside the window after each onset in which an anchor counts as an
event anchor -- the one place a reader can check that the conditioning is on contractions at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort, events, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval.collect import CONTRACTION_AGE_COLUMN
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_labels,
)

#: This analysis's own subdirectory inside the results directory. The sibling's name, kept, because
#: the readout it holds is the sibling's readout: a directory renamed to match what it now contains
#: would make the two runs' output trees stop lining up for a reader comparing them.
ANALYSIS_DIRNAME = "events"

#: What it writes. The per-recording table is **wide** -- one row per recording, one
#: ``<metric>_{event,control,difference}`` triple per readout -- so the runner's cohort fan-out
#: resolves each readout on its own rather than pooling two metrics' differences into one violin.
CONDITIONED_FILENAME = "conditioned_coupling.csv"
CONDITIONED_PER_RECORDING_FILENAME = "conditioned_coupling_per_recording.csv"

#: The detector's output on the retained raw UP traces: one row per contraction found, and the
#: figure that shows them on the trace, so the events every conditioned number rests on can be
#: checked by eye rather than trusted.
DETECTIONS_FILENAME = "contraction_detections.csv"
DETECTION_FIGURE = "contraction_detection"

#: Retained segments the detection figure draws, as many from every clinical class.
DETECTION_SAMPLES_PER_CLASS = 2

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
CONDITIONED_FIGURE = "conditioned_coupling"

#: Guards. Below either of these the conditioned comparison is a recorded skip rather than a
#: number: a rate over a handful of anchors from one or two recordings is dominated by which
#: recordings they were.
MIN_EVENT_ANCHORS = 200
MIN_EVENT_RECORDINGS = 4

#: The two coupling readouts the conditioned comparison cuts, as ``(reported name, column)``.
CONDITIONED_READOUTS: Tuple[Tuple[str, str], ...] = (
    ("pred_gap_mc_nats", "mc_pred_gap"),
    ("source_conditioned_kl_raw", "kld_per_t"),
)

#: The sibling's two readouts this package does not have, each with the reason. Emitted in the
#: analysis's own record rather than only documented: a reader meets ``events`` in ``summary.json``
#: expecting the three the raw pipeline reports, and a key that is simply missing reads as a step
#: that failed rather than as one that was never defined here.
REMOVED_READOUTS: Tuple[Dict[str, str], ...] = (
    {
        "readout": "deceleration_skill",
        "reason": (
            "it runs a deceleration detector over each branch's forecast block in bpm; this "
            "model forecasts the kept wavelet-modulus and phase-harmonic coefficients in the "
            "loader's z units, and defining a deceleration on that axis is a new construction "
            "rather than a port of this one"
        ),
    },
    {
        "readout": "contraction_triggered_response",
        "reason": (
            "it averages the forecast, the truth and their difference as bpm waveforms around a "
            "contraction onset; the same objection applies, and the timing half of the question "
            "it asks is answered by the conditioned coupling below, which conditions on when a "
            "contraction happened rather than on what the forecast looked like"
        ),
    },
)


# =============================================================================
# Contraction-conditioned coupling
# =============================================================================
def conditioned_anchors(
    per_anchor: pd.DataFrame, *, window_s: float, seed: int
) -> pd.DataFrame:
    r"""Split the anchor table into event anchors and count-matched control anchors.

    Args:
        per_anchor: The per-anchor table, carrying :data:`~teb_vae.lag_attn_cfs.eval.collect.
            CONTRACTION_AGE_COLUMN`.
        window_s: Seconds after a contraction onset an anchor still counts as conditioned.
        seed: Seed for the control draw, so the comparison is reproducible from the summary.

    Returns:
        The event and control rows with a ``condition`` column. Controls are drawn **within each
        recording** and matched to that recording's event count, so the comparison cannot be a
        comparison between recordings wearing two labels. Empty when the column is absent.
    """
    if per_anchor.empty or CONTRACTION_AGE_COLUMN not in per_anchor.columns:
        return per_anchor.head(0).assign(condition=pd.Series(dtype=object))
    age = np.asarray(per_anchor[CONTRACTION_AGE_COLUMN], dtype=np.float64)
    near = np.isfinite(age) & (age >= 0.0) & (age <= float(window_s))
    events_frame = per_anchor[near]
    # "Not near" is both far and never: NaN means no contraction preceded this anchor at all,
    # which is as much a control as one an hour past its contraction.
    far_frame = per_anchor[~near]

    rng = np.random.default_rng(int(seed))
    picked: List[pd.DataFrame] = []
    counts = events_frame.groupby("guid").size() if len(events_frame) else pd.Series(dtype=int)
    for guid, cell in far_frame.groupby("guid", sort=True):
        wanted = int(counts.get(guid, 0))
        if wanted <= 0:
            continue
        take = min(wanted, len(cell))
        chosen = rng.choice(len(cell), size=take, replace=False)
        picked.append(cell.iloc[np.sort(chosen)])
    controls = pd.concat(picked, ignore_index=True) if picked else far_frame.head(0)
    return pd.concat(
        [events_frame.assign(condition="event"), controls.assign(condition="control")],
        ignore_index=True,
    )


def conditioned_rows(
    split: pd.DataFrame, labels_by_guid: pd.Series, *, resamples: int, seed: int
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Compare each readout between event and control anchors, per recording and per class.

    Args:
        split: The event/control anchors.
        labels_by_guid: Each recording's clinical class, for the per-class cut.
        resamples: Bootstrap resamples over recordings.
        seed: Bootstrap seed.

    Returns:
        ``(rows, per_recording)``. Each row carries the paired difference with its interval and a
        signed-rank test; the frame carries one row per recording per readout, which is what the
        grouped fan-out and the figure both read.
    """
    rows: List[Dict[str, Any]] = []
    pieces: List[pd.DataFrame] = []
    for name, column in CONDITIONED_READOUTS:
        if column not in split.columns or not len(split):
            continue
        # Per recording *within* each condition first: a recording contributing forty event
        # anchors and one contributing four then count the same, which is the unit every other
        # comparison in this pipeline is made on.
        pivot = split.groupby(["guid", "condition"])[column].mean().unstack("condition")
        for side in ("event", "control"):
            if side not in pivot.columns:
                pivot[side] = np.nan
        cut = pd.DataFrame(
            {
                "guid": [str(value) for value in pivot.index],
                "metric": name,
                labels.CLASS_COLUMN: [labels_by_guid.get(value) for value in pivot.index],
                "event": np.asarray(pivot["event"], dtype=np.float64),
                "control": np.asarray(pivot["control"], dtype=np.float64),
            }
        )
        cut["difference"] = cut["event"] - cut["control"]
        pieces.append(cut)
        rows.append(_conditioned_row(name, column, cut, resamples=resamples, seed=seed))
        # Named ``class_name`` rather than ``cohort``, which is the module holding the canonical
        # order two lines below. The per-class rows follow that order rather than the alphabetical
        # one ``groupby`` produces, so the CSV reads HIE, acidosis, healthy like every figure.
        by_class = {
            str(class_name): cell
            for class_name, cell in cut.groupby(labels.CLASS_COLUMN, dropna=True)
        }
        for class_name in cohort.ordered_groups(list(by_class), labels.CLASS_COLUMN):
            rows.append(
                _conditioned_row(
                    name, column, by_class[class_name],
                    resamples=resamples, seed=seed, cohort=class_name,
                )
            )
    per_recording = (
        pd.concat(pieces, ignore_index=True) if pieces
        else pd.DataFrame(columns=["guid", "metric", labels.CLASS_COLUMN, "event", "control",
                                   "difference"])
    )
    return rows, per_recording


def _conditioned_row(
    name: str,
    column: str,
    frame: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
    cohort: str = "pooled",
) -> Dict[str, Any]:
    """Summarise one readout's event-minus-control difference over recordings."""
    differences = finite_column(frame, "difference")
    interval = shared_stats.bootstrap_ci(differences, resamples=resamples, seed=seed)
    paired = shared_stats.wilcoxon_paired(
        finite_column(frame, "event"),
        finite_column(frame, "control"),
        label_left="anchors near a contraction",
        label_right="count-matched control anchors",
    )
    return {
        "metric": name,
        "source_column": column,
        "cohort": cohort,
        **{key: value for key, value in describe(differences).items() if key != "metric"},
        "event_mean": float(np.nanmean(finite_column(frame, "event")))
        if len(frame) else float("nan"),
        "control_mean": float(np.nanmean(finite_column(frame, "control")))
        if len(frame) else float("nan"),
        "ci_lo": interval["lo"],
        "ci_hi": interval["hi"],
        "wilcoxon_p_value": paired["p_value"],
        "wilcoxon_n_pairs": paired["n_pairs"],
    }


def per_recording_table(
    per_recording: pd.DataFrame, split: pd.DataFrame, per_sample: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Lay the per-recording comparison out wide, one row per recording, identity first.

    Args:
        per_recording: The long frame :func:`conditioned_rows` returns.
        split: The event and control anchors, for each recording's two anchor counts.
        per_sample: The per-sample table, for each recording's class and subgroup.

    Returns:
        ``guid``, the class and subgroup columns, both anchor counts, then one
        ``<metric>_{event,control,difference}`` triple per readout.
    """
    identity = ["guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN]
    if per_recording.empty:
        return pd.DataFrame(columns=identity)
    wide = per_recording.pivot(index="guid", columns="metric", values=["event", "control", "difference"])
    wide.columns = [f"{metric}_{side}" for side, metric in wide.columns]
    wide = wide[[
        f"{name}_{side}" for name, _ in CONDITIONED_READOUTS
        for side in ("event", "control", "difference") if f"{name}_{side}" in wide.columns
    ]]
    counts = split.assign(guid=split["guid"].astype(str)).groupby(["guid", "condition"]).size().unstack(fill_value=0)
    for side in ("control", "event"):
        wide.insert(0, f"n_{side}_anchors", counts.get(side, pd.Series(dtype=int)).reindex(wide.index).fillna(0).astype(int))
    known = (
        per_recording_labels(per_sample).rename(index=str)
        if per_sample is not None and not per_sample.empty else pd.DataFrame()
    )
    for name in (labels.SUBGROUP_COLUMN, labels.CLASS_COLUMN):
        values = known[name].reindex(wide.index) if name in known.columns else None
        wide.insert(0, name, values if values is not None else None)
    return wide.rename_axis("guid").reset_index()


def detection_rows(
    retained: Dict[str, np.ndarray],
    per_sample: Optional[pd.DataFrame],
    *,
    scales: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Run the detector over every retained raw UP trace and list what it found.

    The same detector, on the same trace and validity, that put ``seconds_since_contraction`` on
    the per-anchor table -- so these are the events every conditioned number is measured from.

    Args:
        retained: The collection's retained arrays: ``up_raw``, ``weight`` and the sample index.
        per_sample: The per-sample table, for each segment's identity.
        scales: From :func:`~teb_vae.lag_attn_cfs.eval.traces.raw_signal_scales`, for the peak's
            physical amplitude.

    Returns:
        One row per contraction: the segment's identity, the onset, peak and end in seconds from
        the start of the trimmed segment, and the peak pressure with its unit. Empty when no raw
        UP was retained.
    """
    up = retained.get("up_raw")
    if up is None or len(up) == 0:
        return pd.DataFrame()
    weight = retained.get("weight")
    index = np.asarray(retained.get("waveforms_sample_index", np.arange(len(up))), dtype=np.int64)
    table = (
        per_sample.set_index("sample_index", drop=False)
        if per_sample is not None and "sample_index" in per_sample.columns else None
    )
    rows: List[Dict[str, Any]] = []
    for row in range(len(up)):
        raw = np.asarray(up[row], dtype=np.float64).reshape(-1)
        valid = None if weight is None else traces.raw_validity_of(weight[row], raw.size)
        found = events.detect_contractions(raw, valid=valid)
        physical, unit = traces.physical_raw(raw, "up", scales, valid)
        key = int(index[row]) if row < index.size else row
        identity = table.loc[key] if table is not None and key in table.index else None
        for onset, peak, end in zip(found["onset_raw"], found["peak_raw"], found["end_raw"]):
            rows.append(
                {
                    **{
                        name: None if identity is None or name not in identity.index else identity[name]
                        for name in ("guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "epoch")
                    },
                    "sample_index": key,
                    "retained_row": row,
                    "onset_s": float(onset) / events.FS_RAW,
                    "peak_s": float(peak) / events.FS_RAW,
                    "end_s": float(end) / events.FS_RAW,
                    "peak_value": float(physical[int(peak)]),
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def build_detection_figure(
    retained: Dict[str, np.ndarray],
    per_sample: Optional[pd.DataFrame],
    detections: pd.DataFrame,
    *,
    window_s: float,
    first_anchor_s: float,
    scales: Dict[str, Tuple[float, float]],
) -> Optional[Any]:
    """Draw the detector's contractions on raw UP, a few retained segments per clinical class.

    Each row is one segment on the fixed pressure scale: onset, peak and end marked, the span
    after each onset in which an anchor counts as an event anchor shaded, and the stretch before
    the first decoded anchor greyed, since no anchor there can be conditioned on anything. The
    page carries one short key line; the event window length and the onset convention are in the
    record (``event_lag_window_s``, ``onset_convention``) and ``FIGURE_GUIDE.md``.

    Args:
        retained: The collection's retained arrays.
        per_sample: The per-sample table, for identity.
        detections: From :func:`detection_rows`.
        window_s: ``event_lag_window_s``, the width of the shaded span after each onset.
        first_anchor_s: Seconds into the segment at which the first decoded anchor's step ends.
        scales: The raw-signal scales.

    Returns:
        The figure, or ``None`` when no raw UP was retained.
    """
    up = retained.get("up_raw")
    if up is None or len(up) == 0:
        return None
    weight = retained.get("weight")
    index = np.asarray(retained.get("waveforms_sample_index", np.arange(len(up))), dtype=np.int64)
    table = (
        per_sample.set_index("sample_index", drop=False)
        if per_sample is not None and "sample_index" in per_sample.columns else None
    )

    def _label(row: int, name: str) -> Any:
        key = int(index[row]) if row < index.size else row
        found = table.loc[key] if table is not None and key in table.index else None
        return None if found is None or name not in found.index else found[name]

    classes = [str(_label(row, labels.CLASS_COLUMN)) for row in range(len(up))]
    chosen: List[int] = []
    for name in cohort.ordered_groups(list(dict.fromkeys(classes)), labels.CLASS_COLUMN):
        chosen += [row for row in range(len(up)) if classes[row] == name][:DETECTION_SAMPLES_PER_CLASS]

    figure, axes = figures.new_figure(len(chosen), height_per_row=1.3)
    for axis, row in zip(axes[:, 0], chosen):
        raw = np.asarray(up[row], dtype=np.float64).reshape(-1)
        valid = None if weight is None else traces.raw_validity_of(weight[row], raw.size)
        physical, unit = traces.physical_raw(raw, "up", scales, valid)
        seconds = (np.arange(raw.size) + 1.0) / events.FS_RAW
        axis.axvspan(0.0, first_anchor_s, color=figures.COLOR_LIGHT_GRAY, alpha=0.6, linewidth=0, zorder=0)
        traces.draw_raw_signal(axis, seconds, physical, "up", unit)
        found = detections[detections["retained_row"] == row] if len(detections) else detections
        for _, event in found.iterrows():
            axis.axvspan(event["onset_s"], event["onset_s"] + window_s, color=figures.COLOR_GREEN,
                         alpha=0.12, linewidth=0, zorder=0)
            for key, marker in (("onset_s", "^"), ("peak_s", "o"), ("end_s", "v")):
                position = int(round(float(event[key]) * events.FS_RAW))
                axis.plot(event[key], physical[min(position, physical.size - 1)], marker=marker,
                          color=figures.COLOR_VERMILLION, markersize=figures.MARKER_SMALL + 1.0, linestyle="none")
        axis.set_title(
            f"guid {_label(row, 'guid')}, subgroup {_label(row, labels.SUBGROUP_COLUMN)}"
        )
        axis.set_xlim(0.0, raw.size / events.FS_RAW)
        axis.set_xlabel("seconds into the segment")
        figures.style_axes(axis)
    figures.caveat_note(
        figure,
        "^ onset, o peak, v end. Green: event window. Grey: before first anchor.",
    )
    return figure


# =============================================================================
# The figure
# =============================================================================
def build_conditioned_figure(per_recording: pd.DataFrame, *, window_s: Optional[float] = None) -> Any:
    """Draw the event-versus-control comparison per readout, split by clinical class.

    The classes run HIE, acidosis, healthy and are coloured red, amber, green -- the same order
    and palette every other cohort figure in this evaluation uses, so two of them can be read
    side by side. ``groupby`` alone would order them alphabetically, putting acidosis first. Each
    violin names its recording count on its tick label; the event window it was cut at is in the
    record and ``FIGURE_GUIDE.md``.

    Args:
        per_recording: The long per-recording frame, one row per recording and readout.
        window_s: ``event_lag_window_s``. Unused by the drawing; kept for the caller's signature.

    Returns:
        The figure; the caller renders and closes it.
    """
    metrics = [name for name, _ in CONDITIONED_READOUTS]
    figure, axes = figures.new_figure(len(metrics))
    for index, metric in enumerate(metrics):
        axis = axes[index, 0]
        cut = per_recording[per_recording["metric"] == metric] if len(per_recording) else (
            per_recording
        )
        values: Dict[str, np.ndarray] = {}
        if len(cut):
            # ``dropna=False`` keeps the unlabelled recordings as their own violin rather than
            # dropping them silently; the canonical order does not know that label, so it lands
            # after the three classes it does know.
            by_class = {
                f"{name}": finite_column(cell, "difference")
                for name, cell in cut.groupby(labels.CLASS_COLUMN, dropna=False)
            }
            values = {
                name: by_class[name]
                for name in cohort.ordered_groups(list(by_class), labels.CLASS_COLUMN)
            }
        figures.violin_panel(
            axis, {f"{name} (n={values[name].size})": values[name] for name in values} or {"all": np.zeros(0)},
            title=metric,
            ylabel="event minus control (nats)",
            colors={f"{name} (n={values[name].size})": colour
                    for name, colour in figures.group_colors(list(values)).items()},
            reference=0.0, reference_label="no effect",
        )
    return figure


# =============================================================================
# The registry entry point
# =============================================================================
def run_events_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Restrict both coupling readouts to anchors near a contraction, against matched controls.

    Args:
        context: The analysis context, read for the per-anchor table the contraction timing rides
            on, the per-sample table the clinical labels come from, and the retained raw UP the
            detection figure is drawn on. The model is not touched.
        eval_config: The validated block, for the event window, the bootstrap settings and the
            seed the control draw follows from.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the conditioned comparison, carrying its own skip reason when the
        timing column is absent or the guards did not clear, and :data:`REMOVED_READOUTS` either
        way.
    """
    collection = context.collection
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    window_s = float(eval_config.get("event_lag_window_s", 120.0))

    conditioned = _conditioned_readout(
        collection, directory, window_s=window_s, resamples=resamples, seed=seed
    )
    pd.DataFrame(conditioned["rows"]).to_csv(directory / CONDITIONED_FILENAME, index=False)
    figure_name = str(figures.render_figure(
        build_conditioned_figure(conditioned["per_recording"], window_s=window_s),
        directory / CONDITIONED_FIGURE,
    ).name)
    written = [CONDITIONED_FILENAME, figure_name]
    if len(conditioned["per_recording"]):
        written.append(CONDITIONED_PER_RECORDING_FILENAME)
    detection = _detection_readout(context, directory, window_s=window_s)
    written += detection.pop("files")

    result: Dict[str, Any] = {
        # The anchors this analysis actually compared, not the segments behind them: the readout is
        # per anchor, and reporting a segment count would put it in the population comparison
        # against analyses whose ``n_samples`` counts something else.
        "n_samples": None,
        "composition": {
            "n_event_anchors": conditioned["n_event_anchors"],
            "n_control_anchors": conditioned["n_control_anchors"],
            "n_recordings": conditioned["n_recordings"],
        },
        # Uncapped, and that is the structural difference from the sibling's ``events``: this
        # readout reads the per-anchor table rather than a retained forecast block, so it runs over
        # every anchor of the split and ``caps.waveforms`` does not reach it.
        "plan": {
            "capped": False,
            "bootstrap_resamples": resamples,
            "seed": seed,
            "event_lag_window_s": window_s,
        },
        "conditioned": conditioned["record"],
        "detection": detection,
        "removed_readouts": [dict(entry) for entry in REMOVED_READOUTS],
        "files": written,
    }
    if len(conditioned["per_recording"]):
        result["grouped_frames"] = [
            grouped_frame_entry(
                ANALYSIS_DIRNAME, CONDITIONED_PER_RECORDING_FILENAME,
                [f"{name}_difference" for name, _ in CONDITIONED_READOUTS],
            )
        ]
    return result


def _detection_readout(context: Any, directory: Path, *, window_s: float) -> Dict[str, Any]:
    """List and draw the detector's contractions on the retained raw UP, or record why not."""
    collection = context.collection
    retained = dict(getattr(collection, "retained", None) or {})
    if retained.get("up_raw") is None:
        return {"skipped": True, "reason": "no raw UP was retained (eval_config.caps.waveforms)", "files": []}
    per_sample = getattr(collection, "per_sample", None)
    scales = traces.raw_signal_scales(getattr(context, "config", None))
    detections = detection_rows(retained, per_sample, scales=scales)
    detections.to_csv(directory / DETECTIONS_FILENAME, index=False)
    geometry = dict((getattr(collection, "record", None) or {}).get("geometry") or {})
    first = int(geometry.get("anchor_first", geometry.get("anchor_floor", 0)) or 0)
    figure = build_detection_figure(
        retained, per_sample, detections, window_s=window_s,
        first_anchor_s=(first + 1) * SECONDS_PER_STEP,
        scales=scales,
    )
    files = [DETECTIONS_FILENAME]
    if figure is not None:
        files.append(str(figures.render_figure(figure, directory / DETECTION_FIGURE).name))
    n_segments = int(len(retained["up_raw"]))
    return {
        "skipped": False,
        "n_retained_segments": n_segments,
        "n_contractions": int(len(detections)),
        "contractions_per_segment": float(len(detections)) / n_segments if n_segments else None,
        "onset_convention": events.ONSET_WALK_BACK_NOTE,
        "files": files,
    }


def _conditioned_readout(
    collection: Any, directory: Path, *, window_s: float, resamples: int, seed: int
) -> Dict[str, Any]:
    """Restrict both coupling readouts to anchors near a contraction, or record why not."""
    per_anchor = getattr(collection, "per_anchor", None)
    per_sample = getattr(collection, "per_sample", None)
    blank = pd.DataFrame(columns=["guid", "metric", labels.CLASS_COLUMN, "event", "control",
                                  "difference"])
    if per_anchor is None or per_anchor.empty or CONTRACTION_AGE_COLUMN not in per_anchor.columns:
        reason = (
            "the per-anchor table carries no contraction timing, so there are no anchors to "
            "condition on"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: conditioned coupling skipped -- {reason}")
        return {"rows": [], "per_recording": blank, "n_event_anchors": 0, "n_control_anchors": 0,
                "n_recordings": 0,
                "record": {"skipped": True, "reason": reason}}

    split = conditioned_anchors(per_anchor, window_s=window_s, seed=seed)
    events_only = split[split["condition"] == "event"] if len(split) else split
    n_event_anchors = int(len(events_only))
    n_control_anchors = int((split["condition"] == "control").sum()) if len(split) else 0
    n_recordings = int(events_only["guid"].nunique()) if n_event_anchors else 0
    if n_event_anchors < MIN_EVENT_ANCHORS or n_recordings < MIN_EVENT_RECORDINGS:
        reason = (
            f"{n_event_anchors} anchor(s) within {window_s:g} s of a contraction over "
            f"{n_recordings} recording(s); the comparison needs at least {MIN_EVENT_ANCHORS} over "
            f"{MIN_EVENT_RECORDINGS}"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: conditioned coupling skipped -- {reason}")
        return {"rows": [], "per_recording": blank, "n_event_anchors": n_event_anchors,
                "n_control_anchors": n_control_anchors, "n_recordings": n_recordings,
                "record": {"skipped": True, "reason": reason}}

    labels_by_guid = (
        per_sample.groupby("guid")[labels.CLASS_COLUMN].first()
        if per_sample is not None and labels.CLASS_COLUMN in per_sample.columns
        else pd.Series(dtype=object)
    )
    rows, per_recording = conditioned_rows(
        split, labels_by_guid, resamples=resamples, seed=seed
    )
    per_recording_table(per_recording, split, per_sample).to_csv(
        directory / CONDITIONED_PER_RECORDING_FILENAME, index=False
    )
    logger.info(
        f"{ANALYSIS_DIRNAME}: {n_event_anchors} event anchor(s) against {n_control_anchors} "
        f"count-matched control(s) over {n_recordings} recording(s)"
    )
    return {
        "rows": rows,
        "per_recording": per_recording,
        "n_event_anchors": n_event_anchors,
        "n_control_anchors": n_control_anchors,
        "n_recordings": n_recordings,
        "record": {
            "window_s": float(window_s),
            "n_event_anchors": n_event_anchors,
            "n_control_anchors": n_control_anchors,
            "n_recordings": n_recordings,
            # What "within the window of a contraction" is measured from. The convention has to
            # travel with the number: the detector's onset is a level crossing of the peak's own
            # prominence, not the first sample of the rise, so a reader comparing this window
            # against a clinical one is otherwise comparing two different zeros.
            "onset_convention": events.ONSET_WALK_BACK_NOTE,
            "rows": rows,
        },
    }
