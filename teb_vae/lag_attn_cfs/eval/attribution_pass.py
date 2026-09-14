r"""The attribution pass: which segments, which anchors, every Captum call, and what is written.

:mod:`~teb_vae.lag_attn_cfs.eval.attributions` is the arithmetic -- the wrapper, the baselines,
the Captum calls, the reductions and the figure builders. This module is the **pass** both cells
run over it: it draws the segments and the traced recordings, re-reads them through a strictly
sequential subset loader with the identity check, runs every attribution at every chosen anchor,
reduces the maps to the durable tables, and writes the files, the figures and the block a summary
carries. The lag-attentive cells reach it from their registered analysis, the lag-residual cell
from its post-pass stage, so the two directories hold the same tables under the same names and a
reader can lay one cell's attribution beside the other's.

**The selection is per recording, one segment each.** Every summary here is over recordings, so
one attributed segment per drawn recording keeps every recording one unit and the class balance
exact; the recordings are drawn with the traces' own class-balanced, seeded draw, at an
eligibility floor of one segment rather than two, and the segment is the recording's middle one
by ``epoch``. The cap is on segments, so the per-class draw is the cap divided by the number of
classes the split carries, rounded up, and the last class drawn stops at the cap.

**What is attributed per anchor.** The divergence and the mean-decoded forecast gap under both
baselines; the model's own lag readout on every configured lag band under the source-null
baseline alone (the target streams are held there, so the question is what source content makes
the model read that band); the anchor's largest per-coordinate divergence under the source-null
baseline; the layer split of the divergence and the gap; and the source zeroed band by band, as
feature ablation, on the divergence and the gap. The structural checks are measured on every row
of a real run and land in the block beside the fixture-proved ones.
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
from teb_vae.lag_attn_cfs.eval import cohort, frames, lag_axis, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import (
    check_batch_identity,
    epoch_stamp,
    subset_loader,
)
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, batch_field, model_inputs

#: The manifest of traced recordings, and its columns.
TRACE_MANIFEST_FILENAME = "attribution_traces.csv"
TRACE_MANIFEST_COLUMNS: Tuple[str, ...] = (
    "guid", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN, "n_segments", "n_anchors", "span_hours",
    "arrays_file", "figure_file",
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


def select_segments(
    segments: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    *,
    cap: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Draw the segments to attribute: one per recording, class-balanced, seeded, capped.

    Args:
        segments: From :func:`labelled_segments`.
        index_map: The dataset listing.
        cap: The segment cap.
        seed: The draw's seed.

    Returns:
        ``(rows, accounting)``: the chosen rows with ``dataset_index``, ascending in that index,
        and the draw's accounting with the cap recorded.
    """
    recordings = recording_table(segments, index_map)
    classes = [name for name in recordings[labels.CLASS_COLUMN].dropna().unique()] if len(recordings) else []
    per_class = int(math.ceil(int(cap) / max(len(classes), 1)))
    chosen, accounting = traces.select_recordings(
        recordings, per_class=per_class, seed=int(seed), min_segments=1
    )
    accounting["segment_cap"] = int(cap)
    picked: List[Dict[str, Any]] = []
    for _, choice in chosen.iterrows():
        if len(picked) >= int(cap):
            break
        guid = str(choice["guid"])
        own = segments[segments["guid"].astype(str) == guid].sort_values("epoch")
        if own.empty:
            continue
        middle = own.iloc[len(own) // 2]
        index = index_map.get((guid, epoch_stamp(middle["epoch"])))
        if index is None:
            continue
        picked.append(
            {
                "guid": guid, "epoch": float(middle["epoch"]),
                labels.CLASS_COLUMN: choice[labels.CLASS_COLUMN],
                labels.SUBGROUP_COLUMN: choice[labels.SUBGROUP_COLUMN],
                "dataset_index": int(index),
            }
        )
    columns = ["guid", "epoch", *labels.GROUP_COLUMNS, "dataset_index"]
    rows = pd.DataFrame(picked, columns=columns)
    accounting["n_segments_selected"] = int(len(rows))
    return rows.sort_values("dataset_index").reset_index(drop=True), accounting


def select_trace_recordings(
    segments: pd.DataFrame,
    index_map: Mapping[Tuple[str, Optional[int]], int],
    *,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """The recordings whose attribution is followed through every segment, per class.

    Drawn under the trace analysis's own seed offset and eligibility floor, so the first
    recording per class is the first one that analysis traced and the two figures describe the
    same recording.

    Args:
        segments: From :func:`labelled_segments`.
        index_map: The dataset listing.
        seed: The run's seed, before the trace offset.

    Returns:
        ``(chosen, accounting)`` as :func:`traces.select_recordings` returns them.
    """
    recordings = recording_table(segments, index_map)
    return traces.select_recordings(
        recordings, per_class=core.TRACE_RECORDINGS_PER_CLASS,
        seed=int(seed) + traces.TRACE_DRAW_SEED_OFFSET,
    )


def recording_rows(guid: str, index_map: Mapping[Tuple[str, Optional[int]], int]) -> pd.DataFrame:
    """Every dataset row of one recording, ascending in dataset index."""
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
        maps: The first full map per class, for the map figure.
        forward_equivalents: How many single-row forward-and-backward passes the batch cost.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.rows: List[Dict[str, Any]] = []
        self.vectors: Dict[str, List[np.ndarray]] = {}
        self.maps: List[Dict[str, Any]] = []
        self.forward_equivalents = 0

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
    keep_maps_for: Optional[Dict[str, Any]] = None,
    work: Optional[BatchWork] = None,
) -> BatchWork:
    """Run every attribution of one batch and reduce each row to its record and vectors.

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
        keep_maps_for: ``{class: None}`` of the classes whose first map is still wanted; filled
            in place as maps are kept.
        work: The accumulator to extend, or ``None`` for a fresh one.

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
    columns_per_sample = core.spread_columns(contributing, anchors_per_segment)
    identity = _identity(batch, rows)
    work = work if work is not None else BatchWork()
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

    def anchor_steps() -> np.ndarray:
        """Each row's anchor as a stored step."""
        return np.asarray([int(anchors_all[s, c]) for s, c in zip(sample, columns.cpu().numpy())], dtype=np.int64)

    steps = anchor_steps()
    if cell.dense_latent:
        kld_rows = kld_dim.detach().cpu().numpy()[sample, steps]
    else:
        kld_rows = kld_dim.detach().cpu().numpy()[sample, columns.cpu().numpy()]
    top_coordinate = torch.as_tensor(np.argmax(kld_rows, axis=1), dtype=torch.long, device=columns.device)

    calls: List[Tuple[str, str, Optional[Tuple[int, int]], Optional[torch.Tensor], str]] = []
    for readout in readouts:
        for baseline in baselines:
            calls.append((readout, baseline, None, None, ""))
    if with_lag_readout:
        for name, band in lag_bands.items():
            calls.append((core.READOUT_LAG_BAND, core.BASELINE_SOURCE_NULL, tuple(band), None, str(name)))
    if with_top_coordinate:
        calls.append((core.READOUT_KLD_DIM, core.BASELINE_SOURCE_NULL, None, top_coordinate, ""))

    for readout, baseline, band, coordinates, band_name in calls:
        wrapper = core.AnchorReadout(
            model, cell, readout=readout, likelihood=likelihood, lag_band=band or (0, 0)
        ).eval()
        result = core.integrated_gradients(
            wrapper, rows_inputs, rows_extra, columns, baseline=baseline,
            coordinates=coordinates, n_steps=n_steps,
        )
        work.forward_equivalents += n_rows * (int(n_steps) + 3)
        _reduce_rows(
            work, result, identity=identity, sample=sample, band_name=band_name, n_lags=n_lags,
            model_profile=model_profile, lag_bands=lag_bands, channel_groups=channel_groups,
            live=live, kld_rows=kld_rows,
        )
        layer = None
        if with_layer and readout in core.MAIN_READOUTS and baseline == core.BASELINE_SOURCE_NULL:
            layer = core.layer_attribution(
                wrapper, rows_inputs, rows_extra, columns, baseline=baseline, n_steps=n_steps
            )
            work.forward_equivalents += n_rows * (int(n_steps) + 2)
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
                record["layer_off_axis_total"] = float(layer["off_axis_total"][offset])
                per_unit = layer["per_unit"][offset]
            else:
                per_unit = np.full(0, np.nan)
            work.append_vector("layer_per_unit", per_unit)
            if ablation is not None:
                for name, values in ablation.items():
                    record[f"ablation_{name}"] = float(values[offset])
        # The first full map per class, from the divergence under the source-null baseline.
        if keep_maps_for is not None and readout == core.READOUT_KLD and baseline == core.BASELINE_SOURCE_NULL:
            for offset in range(n_rows):
                clinical_class = identity[int(sample[offset])][labels.CLASS_COLUMN]
                key = str(clinical_class)
                if key in keep_maps_for and keep_maps_for[key] is None:
                    keep_maps_for[key] = {
                        **identity[int(sample[offset])], "anchor": int(result.anchor[offset]),
                        "readout": readout, "baseline": baseline,
                        core.STREAM_TARGET: result.target[offset], core.STREAM_SOURCE: result.source[offset],
                        "lag_profile": core.lag_profile(core.time_profile(result.source[offset:offset + 1]), [int(result.anchor[offset])], n_lags)[0],
                        "model_profile": model_profile[offset],
                        "live_target": live[core.STREAM_TARGET], "live_source": live[core.STREAM_SOURCE],
                    }
                    work.maps.append(keep_maps_for[key])
    return work


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
) -> None:
    """Turn one Captum call's maps into records and row-aligned vectors."""
    n_rows = int(result.anchor.shape[0])
    target_time = core.time_profile(result.target)
    source_time = core.time_profile(result.source)
    target_channels = core.channel_profile(result.target)
    source_channels = core.channel_profile(result.source)
    lags = core.lag_profile(source_time, result.anchor, n_lags)
    target_lags = core.lag_profile(target_time, result.anchor, n_lags)
    fit = core.agreement(lags, model_profile)
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
        work.rows.append(record)
        work.append_vector("time_profile_target", target_time[offset].astype(np.float32))
        work.append_vector("time_profile_source", source_time[offset].astype(np.float32))
        work.append_vector("lag_profile", lags[offset].astype(np.float32))
        work.append_vector("target_lag_profile", target_lags[offset].astype(np.float32))
        work.append_vector("model_profile", model_profile[offset].astype(np.float32))
        work.append_vector("channel_profile_target", target_channels[offset].astype(np.float32))
        work.append_vector("channel_profile_source", source_channels[offset].astype(np.float32))


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

    Returns:
        ``(work, n_batches)``.
    """
    work = BatchWork()
    if selected.empty:
        return work, 0
    classes = {str(name): None for name in selected[labels.CLASS_COLUMN].dropna().unique()}
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
    )
    worst = max((float(row["source_abs_total"]) for row in work.rows), default=None)
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

    Args:
        rows: The per-row table.

    Returns:
        Indexed by ``guid`` with the cohort columns, ``n_segments`` and the columns of
        :data:`RECORDING_VALUE_COLUMNS`.
    """
    if rows.empty:
        return pd.DataFrame(columns=[*RECORDING_VALUE_COLUMNS, *labels.GROUP_COLUMNS, "n_segments"])
    pieces = []
    for readout in core.MAIN_READOUTS:
        subset = rows[(rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)]
        if subset.empty:
            continue
        part = subset[["guid", "epoch", *labels.GROUP_COLUMNS, "source_total", "target_total", "lag_corr", "lag_js"]].copy()
        part = part.rename(columns={name: f"{readout}_{name}" for name in ("source_total", "target_total", "lag_corr", "lag_js")})
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=[*RECORDING_VALUE_COLUMNS, *labels.GROUP_COLUMNS, "n_segments"])
    merged = pieces[0]
    for part in pieces[1:]:
        merged = merged.merge(part, on=["guid", "epoch", *labels.GROUP_COLUMNS], how="outer")
    return frames.per_recording_means(merged, [c for c in RECORDING_VALUE_COLUMNS if c in merged.columns])


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
    for readout in core.MAIN_READOUTS:
        subset = rows[(rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)]
        if subset.empty:
            continue
        for column in band_columns:
            _, stream, name = column.split("_", 2)
            mean, count = _mean_over_recordings(subset, column)
            records.append(
                {
                    "readout": readout, "stream": stream, "band": name, "attribution_mean": mean,
                    "n_recordings": count,
                    "spectral_skill_pred_gap_nats": skill.get(name, float("nan")) if stream == core.STREAM_TARGET else float("nan"),
                    "unit": "readout units (nats per anchor for both readouts)",
                }
            )
    return pd.DataFrame(records)


def lag_bands_frame(rows: pd.DataFrame, lag_bands: Mapping[str, Tuple[int, int]], occlusion: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per (readout, lag band): the IG sum, the ablation delta and the occlusion delta where present."""
    records: List[Dict[str, Any]] = []
    if rows.empty or not lag_bands:
        return pd.DataFrame(records)
    occluded = {}
    if occlusion is not None and "band" in occlusion.columns and "delta_total_nats" in occlusion.columns:
        occluded = {str(row["band"]): float(row["delta_total_nats"]) for _, row in occlusion.iterrows()}
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
                    # The occlusion pass reports the forecast COST of removing the band; on the gap
                    # readout that cost is minus the gap's own change, so the sign is flipped here
                    # and left absent on the divergence, which that pass does not score.
                    "occlusion_delta_total_nats": (-delta if readout == core.READOUT_PRED_GAP else float("nan")),
                    "n_recordings": count,
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
        keep &= np.isfinite(matrix).any(axis=1)
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
) -> List[traces.SegmentTrace]:
    """Attribute the divergence at the chosen anchors of every segment of one recording.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader.
        rows: From :func:`recording_rows`.
        cell: The cell binding.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.
        n_steps: Integration steps.
        anchors_per_segment: Anchors per segment.

    Returns:
        One trace per segment, in dataset order.
    """
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))
    segments = subset_loader(loader, list(rows["dataset_index"]), batch_size=batch_size)
    gathered: List[traces.SegmentTrace] = []
    position = 0
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
            anchors_per_segment=anchors_per_segment, readouts=(core.READOUT_KLD,),
            baselines=(core.BASELINE_SOURCE_NULL,), with_layer=False, with_ablation=False,
            with_lag_readout=False, with_top_coordinate=False,
        )
        table = rows_frame(work)
        vectors = stack_vectors(work)
        for offset, (_, row) in enumerate(batch_rows.iterrows()):
            keep = (table["epoch"] == float(row["epoch"])).to_numpy() if len(table) else np.zeros(0, dtype=bool)
            if not keep.any():
                continue
            part = table[keep]
            gathered.append(
                traces.SegmentTrace(
                    guid=str(row["guid"]), epoch=float(row["epoch"]),
                    clinical_class=clinical_class, subgroup=subgroup,
                    anchor=part["anchor"].to_numpy().astype(np.int64),
                    contributing=np.ones(len(part), dtype=bool),
                    scalars={name: part[name].to_numpy().astype(np.float64)
                             for name in ("value_input", "value_baseline", "source_total", "target_total", "lag_corr", "lag_js")},
                    vectors={"attribution_lag_map": np.abs(vectors["lag_profile"][keep]).astype(np.float64),
                             "model_lag_map": vectors["model_profile"][keep].astype(np.float64)},
                )
            )
    return gathered


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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Trace every chosen recording and write its arrays and figure under its class directory.

    Returns:
        ``(manifest, failures)``.
    """
    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    root = Path(directory) / core.TRACE_DIRNAME
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
                n_steps=n_steps, anchors_per_segment=anchors_per_segment,
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
            figure = figures.render_figure(
                traces.build_recording_figure(
                    recording, panels=core.TRACE_PANELS, lag_seconds=lag_seconds,
                    caveat=f"{cell.lag_qualification}. {caveat}. {lag_axis.GROUP_DELAY_CAVEAT}",
                ),
                class_dir / f"{stem}{core.TRACE_SUFFIX}",
            )
        except Exception as error:  # noqa: BLE001 - one recording is not worth the rest of them
            logger.warning(f"{core.ANALYSIS_DIRNAME}: trace of {guid} failed: {error}")
            failures.append({"guid": guid, "error": f"{type(error).__name__}: {error}"})
            continue
        span = recording.summary[cohort.HOURS_COLUMN]
        manifest.append(
            {
                "guid": guid, labels.CLASS_COLUMN: clinical_class, labels.SUBGROUP_COLUMN: subgroup,
                "n_segments": int(len(recording.segments)), "n_anchors": int(len(recording.anchors)),
                "span_hours": float(span.max() - span.min()) if len(span) else float("nan"),
                "arrays_file": Path(arrays).relative_to(directory).as_posix(),
                "figure_file": Path(figure).relative_to(directory).as_posix(),
            }
        )
    return manifest, failures


# =============================================================================
# The whole pass
# =============================================================================
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
    channel_map: Optional[pd.DataFrame],
    occlusion: Optional[pd.DataFrame],
    spectral: Optional[pd.DataFrame],
    delay_steps: int,
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
        channel_map: The declared-axis channel map, or ``None``.
        occlusion: The occlusion summary, or ``None``.
        spectral: The band-resolved skill table, or ``None``.
        delay_steps: The stored-step delay the lag axis carries, recorded in the plan.

    Returns:
        The block: counts, plan, selection, cost, checks, the method record, the trace manifest,
        failures and files.
    """
    directory = Path(str(results_dir)) / core.ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    caps = dict(eval_config.get("caps") or {})
    cap = int(caps.get(core.CAP_NAME) or core.DEFAULT_SEGMENTS)
    seed = int(eval_config.get("seed", 0)) + core.DRAW_SEED_OFFSET
    configured = dict(eval_config.get("occlusion_bands") or {})
    lag_bands = {str(name): (int(span[0]), int(span[1])) for name, span in configured.items()}
    n_steps = core.IG_STEPS
    plan: Dict[str, Any] = {
        "capped": True, "cap": cap, "seed": seed, "anchors_per_segment": core.ANCHORS_PER_SEGMENT,
        "ig_steps": n_steps, "entry_fraction": core.BASELINE_ENTRY_FRACTION,
        "baselines": list(core.BASELINES), "readouts": list(core.MAIN_READOUTS),
        "lag_readout": cell.lag_readout, "lag_bands": {name: list(span) for name, span in lag_bands.items()},
        "layer": cell.layer_label, "delay_steps": int(delay_steps),
        "anchor_phase": DENSE_ANCHOR_GEOMETRY[0], "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
        "trace_recordings_per_class": core.TRACE_RECORDINGS_PER_CLASS,
        "max_hours_before_delivery_applied": False,
    }
    labelled = labelled_segments(segments)
    selected, accounting = select_segments(labelled, index_map, cap=cap, seed=seed)
    channel_groups = core.channel_groups_from_map(channel_map)

    started = time.perf_counter()
    work, n_batches = run_segments(
        task, loader, selected, cell, lag_bands=lag_bands, channel_groups=channel_groups,
        n_steps=n_steps, anchors_per_segment=core.ANCHORS_PER_SEGMENT,
    )
    purity = target_only_check(task, loader, selected, cell, n_steps=n_steps)
    elapsed = time.perf_counter() - started

    rows = rows_frame(work)
    vectors = stack_vectors(work)
    rows.to_csv(directory / core.ROWS_FILENAME, index=False)
    np.savez_compressed(directory / core.VECTORS_FILENAME, lag_seconds=np.asarray(lag_seconds, dtype=np.float64), **vectors)
    if work.maps:
        np.savez_compressed(
            directory / core.MAPS_FILENAME,
            guid=np.asarray([m["guid"] for m in work.maps], dtype=object),
            subgroup=np.asarray([str(m[labels.SUBGROUP_COLUMN]) for m in work.maps], dtype=object),
            clinical_class=np.asarray([str(m[labels.CLASS_COLUMN]) for m in work.maps], dtype=object),
            anchor=np.asarray([m["anchor"] for m in work.maps], dtype=np.int64),
            target=np.stack([m[core.STREAM_TARGET] for m in work.maps], axis=0),
            source=np.stack([m[core.STREAM_SOURCE] for m in work.maps], axis=0),
        )
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

    caveat = core.ATTRIBUTION_CAVEAT
    files: List[str] = [
        core.ROWS_FILENAME, core.VECTORS_FILENAME, core.RECORDINGS_FILENAME, core.SUMMARY_FILENAME,
        core.BANDS_FILENAME, core.LAG_BANDS_FILENAME, core.LAYER_FILENAME, core.NULL_FILENAME,
    ]
    if work.maps:
        files.append(core.MAPS_FILENAME)
    figure_paths = [
        figures.render_figure(core.build_map_figure(work.maps, lag_seconds=lag_seconds, cell=cell, caveat=caveat), directory / core.MAP_FIGURE),
        figures.render_figure(
            core.build_lag_profile_figure(rows, vectors, lag_seconds=lag_seconds, readouts=core.MAIN_READOUTS, cell=cell, caveat=caveat),
            directory / core.LAG_PROFILE_FIGURE,
        ),
        figures.render_figure(core.build_band_figure(bands, lag_band_table, readouts=core.MAIN_READOUTS, caveat=caveat), directory / core.BAND_FIGURE),
        figures.render_figure(
            core.build_layer_figure(layer, _top_coordinate_rows(rows, vectors), cell=cell, lag_seconds=lag_seconds, caveat=caveat),
            directory / core.LAYER_FIGURE,
        ),
        figures.render_figure(core.build_null_figure(null, caveat=caveat), directory / core.NULL_FIGURE),
    ]
    files.extend(Path(path).name for path in figure_paths)

    chosen, trace_accounting = select_trace_recordings(labelled, index_map, seed=int(eval_config.get("seed", 0)))
    manifest, failures = run_traces(
        task, loader, chosen, index_map, cell, directory=directory, lag_seconds=lag_seconds,
        break_after_s=break_after_s, n_steps=n_steps, anchors_per_segment=core.ANCHORS_PER_SEGMENT, caveat=caveat,
    )
    pd.DataFrame(manifest, columns=list(TRACE_MANIFEST_COLUMNS)).to_csv(directory / TRACE_MANIFEST_FILENAME, index=False)
    files.append(TRACE_MANIFEST_FILENAME)

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
        f"{elapsed:.1f} s; traced {len(manifest)} recording(s), {len(failures)} failed"
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
        "cost": core.cost_record(
            elapsed_s=elapsed, n_segments=int(len(selected)), n_rows=int(len(rows)),
            n_forward_equivalents=int(work.forward_equivalents), device=getattr(task, "device", None),
        ),
        "checks": checks,
        "summary": summary.to_dict(orient="records"),
        "lag_bands": lag_band_table.to_dict(orient="records"),
        "joined": {
            "channel_map": channel_map is not None,
            "occlusion_summary": occlusion is not None,
            "spectral_skill_bands": spectral is not None,
        },
        "methods": core.METHOD_RECORD,
        "lag_qualification": cell.lag_qualification,
        "caveat": caveat,
        "traces": manifest,
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
    subset["target_lag_profile"] = list(vectors["target_lag_profile"][keep])
    return subset


__all__ = [
    "CHANNEL_MAP_FILENAME", "COMPLETENESS_TOLERANCE", "OCCLUSION_SUMMARY_PATH",
    "RECORDING_VALUE_COLUMNS", "SPECTRAL_BANDS_PATH", "TRACE_MANIFEST_COLUMNS",
    "TRACE_MANIFEST_FILENAME", "BatchWork", "attribute_batch", "bands_frame", "labelled_segments",
    "lag_bands_frame", "layer_frame", "null_frame", "read_channel_map", "read_occlusion_summary",
    "read_spectral_bands", "recording_rows", "recording_table", "recordings_frame", "rows_frame",
    "run_pass", "run_segments", "run_traces", "select_segments", "select_trace_recordings",
    "stack_vectors", "summary_frame", "target_only_check", "trace_recording",
]
