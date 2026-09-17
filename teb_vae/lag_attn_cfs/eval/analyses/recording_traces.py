r"""Per-recording traces: every segment of a chosen recording, in time order, at anchor resolution.

Every other analysis reduces the split to a distribution and a recording to a number. This one
follows individual recordings through labour: for at least a handful of recordings per clinical
class it re-runs the forward over **every** segment the dataset holds for them, orders the segments
by ``epoch``, and writes what the model did at every decoded anchor -- the latent and its
parameters, the divergence and its per-coordinate split, the lag attribution and the attention
over lags -- beside the forecast scores the collection pass already took at those same anchors.
Two forms come out of it, both under one identity: the **full** form at anchor resolution, and the
**summary** form at segment resolution. See :mod:`~teb_vae.lag_attn_cfs.eval.traces` for the
reductions, the file layout and the figures, which are shared with the lag-slot cell.

**This is one of the analyses that touch the model**, for the reason the diagnostic pages do: the
latent means and log-variances are on no durable table, the collection pass would have to retain
$(A_{\max}, d_z)$ per segment for *every* segment to serve a selection it cannot know before the
tables exist, and a recording's trace is a few hundred forwards against a pass that decodes four
branches at $K$ draws over the whole split. So the segments are re-read from a strictly sequential
loader over a ``Subset`` of the evaluation dataset, and every batch is checked against the rows it
was built from -- the collection pass runs under a seeded shuffle, and a trace of the wrong
recording is a plausible picture that nothing downstream would notice.

**What is joined rather than recomputed.** The per-anchor forecast scores -- both ``pred_gap``
estimators and the block scores -- come off ``per_anchor.parquet`` by ``(guid, epoch, anchor)``,
because recomputing them would repeat the Monte Carlo decode for numbers the pass already holds.
The divergence is *both* recomputed and joined: the trace's own ``kld_per_t`` against the table's
is the check that the forward re-read here is the forward the tables describe, and its worst
disagreement is recorded rather than assumed away.

**The selection is a seeded draw of eligible recordings, equal per class**, from the recordings
``per_sample.csv`` labels -- so it is not the population the pass's caps bound: a trace is the
recording as the dataset holds it, including segments a stratified cap left uncollected, which are
traced and marked as such rather than dropped to match. The one bound that *is* applied is
``max_hours_before_delivery``: when set, only the segments recorded within it are traced, counted
for eligibility, or drawn from, so a bounded run's traces describe the population its clocks are
read over. A recording with one segment (inside the window) has no evolution to show and is
counted, never drawn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort, frames, lag_axis, traces, traces_html
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import (
    check_batch_identity,
    dataset_index_map,
    epoch_stamp,
    subset_loader,
)
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    anchor_support,
    attention_entropy,
    batch_field,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = traces.ANALYSIS_DIRNAME

#: The two lag profiles a forward of this cell yields per anchor, and the column prefix each
#: one's shape statistics are written under. Both, for the reason every lag readout in this
#: package carries both: the attribution is $K_t$ times the attention and inherits the
#: prior-variance inflation the attention is immune to.
LAG_PROFILES: Dict[str, str] = {"kl_lag_map": "kl_lag", "attention_lag_map": "attn_lag"}

#: Per-anchor columns joined from the collection pass's table, wherever the table carries them.
#: The scores are joined rather than recomputed; the divergence is joined **beside** the
#: recomputed one, under its own name, so the two can be compared.
JOINED_COLUMNS: Tuple[str, ...] = (
    "mc_pred_gap", "mean_pred_gap", "pred_gap", "mc_nll_base_block", "mc_nll_full_block",
    "nll_base_block", "nll_full_block", "seconds_since_contraction",
)

#: The name the table's divergence travels under on the trace, beside the recomputed
#: ``kld_per_t``. The worst absolute disagreement between the two is the run's check that the
#: re-read forward is the collected one.
COLLECTED_KL_COLUMN = "kld_per_t_collected"

#: Above this worst disagreement, in nats, the check is logged as a warning rather than passed
#: silently. Loose enough for a device's non-associative reduction order, tight enough that a
#: different checkpoint or a different anchor geometry cannot hide under it.
KL_AGREEMENT_TOLERANCE = 1e-3

#: The per-segment clocks carried from ``per_sample.csv`` onto the summary, where the segment was
#: collected. A segment the cap left uncollected carries ``NaN`` on both.
CLOCK_COLUMNS: Tuple[str, ...] = ("time_from_labor_onset", cohort.SECOND_STAGE_COLUMN)

#: The rows of every recording's figure, top to bottom. Every model quantity is at the anchor
#: step against hours before delivery, so a column is the same anchor on every row; the joined
#: forecast gap is the only row read off the collection pass rather than off the re-read forward.
PANELS: Tuple[Any, ...] = (
    traces.LinePanel(("kld_per_t",), "Divergence $K_t$ per anchor", "nats", labels=("$K_t$",),
                     segment_mean=True),
    traces.LinePanel(
        ("mean_pred_gap", "mc_pred_gap"), "Forecast gap per anchor, joined from the collection pass",
        "nats", labels=("mean decode", "Monte Carlo"), segment_mean=True,
    ),
    traces.HeatmapPanel("mu_post", "Posterior mean $\\mu^q$ over the latent coordinates", "coordinate",
                        symmetric=True),
    traces.HeatmapPanel("mu_post", "Source shift of the mean, $\\mu^q - \\mu^p$", "coordinate",
                        symmetric=True, subtract="mu_prior"),
    traces.HeatmapPanel("kld_per_dim", "Divergence per latent coordinate, $K_{t,d}$", "coordinate"),
    traces.HeatmapPanel("kl_lag_map", "KL attribution over lags (log colour scale)", "",
                        log=True, lag_axis=True, argmax_column="argmax_lag"),
    traces.LinePanel(("mu_prior_norm", "delta_mu_norm"), "Latent norms", "latent units",
                     labels=("$\\|\\mu^p\\|_2$", "$\\|\\mu^q - \\mu^p\\|_2$")),
    traces.LinePanel(("mean_logvar_prior", "mean_logvar_post"), "Mean log-variance over coordinates", "",
                     labels=("prior", "posterior")),
    traces.LinePanel(("kl_lag_centroid_s", "kl_lag_median_s"), "Lag centre of the KL attribution",
                     "s (stored-coefficient time)", labels=("centroid", "median")),
    traces.LinePanel(("kl_lag_entropy_nats", "attn_lag_entropy_nats"), "Lag entropy", "nats",
                     labels=("KL attribution", "attention")),
)

#: The rows of the cross-recording summary figure: one per-segment column each, one line per
#: traced recording in its class colour under the class median.
SUMMARY_METRICS: Tuple[traces.SummaryMetric, ...] = (
    traces.SummaryMetric("kld_per_t", "nats per anchor", "Divergence $K_t$"),
    traces.SummaryMetric("mean_pred_gap", "nats per anchor", "Mean-decoded forecast gap"),
    traces.SummaryMetric("delta_mu_norm", "latent units",
                         "Source shift of the latent mean, $\\|\\mu^q - \\mu^p\\|_2$"),
    traces.SummaryMetric("n_active_dims", "coordinates", "Active latent coordinates"),
    traces.SummaryMetric("kl_lag_centroid_s", "s (stored-coefficient time)",
                         "Lag centroid of the KL attribution"),
    traces.SummaryMetric("kl_lag_entropy_nats", "nats", "Lag entropy of the KL attribution"),
)


# =============================================================================
# Choosing the recordings, and finding their segments
# =============================================================================
def recording_frame(
    per_sample: pd.DataFrame, index_map: Dict[Tuple[str, Optional[int]], int]
) -> pd.DataFrame:
    """One row per labelled recording: its cohort labels and how many segments the dataset holds.

    The labels come from the collected table, resolved per recording by the one-value-or-none
    rule; the segment count comes from the dataset's own listing, because the trace covers every
    segment the dataset holds and not only the ones the pass collected.

    Args:
        per_sample: The per-sample table.
        index_map: From :func:`dataset_index_map`.

    Returns:
        The recordings, with ``guid``, the two label columns and ``n_segments``.
    """
    labelled = frames.per_recording_labels(per_sample)
    if labelled.empty:
        return pd.DataFrame(columns=["guid", *labels.GROUP_COLUMNS, "n_segments"])
    counts: Dict[str, int] = {}
    for guid, _stamp in index_map:
        counts[guid] = counts.get(guid, 0) + 1
    frame = labelled.reset_index()
    frame["n_segments"] = [int(counts.get(str(guid), 0)) for guid in frame["guid"]]
    return frame


def recording_rows(
    guid: str,
    index_map: Dict[Tuple[str, Optional[int]], int],
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    """Every dataset row of one recording, in dataset order, with what the pass knows of each.

    Args:
        guid: The recording.
        index_map: From :func:`dataset_index_map`.
        per_sample: The per-sample table, for the clocks of the segments the pass collected.

    Returns:
        Rows carrying ``guid``, ``epoch``, ``dataset_index``, ``collected`` and the clock
        columns, sorted by ``dataset_index`` -- the order a sequential ``Subset`` loader visits.
    """
    listed = [
        (stamp, index) for (listed_guid, stamp), index in index_map.items()
        if listed_guid == str(guid) and stamp is not None
    ]
    frame = pd.DataFrame(
        {
            "guid": [str(guid)] * len(listed),
            "epoch": [float(stamp) for stamp, _ in listed],
            "dataset_index": [int(index) for _, index in listed],
        }
    )
    collected = per_sample[per_sample["guid"].astype(str) == str(guid)] if len(per_sample) else per_sample
    stamps = {
        epoch_stamp(row["epoch"]): row for _, row in collected.iterrows()
    } if len(collected) else {}
    frame["collected"] = [epoch_stamp(epoch) in stamps for epoch in frame["epoch"]]
    for column in CLOCK_COLUMNS:
        frame[column] = [
            float(stamps[epoch_stamp(epoch)][column])
            if epoch_stamp(epoch) in stamps and column in stamps[epoch_stamp(epoch)].index
            else float("nan")
            for epoch in frame["epoch"]
        ]
    return frame.sort_values("dataset_index").reset_index(drop=True)


# =============================================================================
# The forward, reduced to arrays
# =============================================================================
@torch.no_grad()
def gather_segment_traces(
    model: Any,
    outputs: Dict[str, torch.Tensor],
    weight: torch.Tensor,
    rows: pd.DataFrame,
    *,
    clinical_class: Optional[str],
    subgroup: Optional[str],
) -> List[traces.SegmentTrace]:
    r"""Reduce one dense forward to one :class:`~teb_vae.lag_attn_cfs.eval.traces.SegmentTrace` per sample.

    Everything is gathered at the forward's own ``anchor_index``, never at a row position: the
    latent tensors are dense over $T$ while the decoded anchors are a gathered set, so a trace
    indexed by position would place every anchor at the wrong time with no shape error.

    Args:
        model: The rebuilt net.
        outputs: The forward's dict.
        weight: The validity signal the task's target builder returned, for the masks.
        rows: The batch's rows, in batch order, carrying ``guid``, ``epoch`` and the clocks.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.

    Returns:
        One trace per sample of the batch.
    """
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    mask, coverage, _kl_support = anchor_support(model, weight, outputs)
    contributing = contributing_anchors(mask) > 0.0
    kld_btd = model.kld_tensor(
        mu_prior=outputs["mu_prior"], logvar_prior=outputs["logvar_prior"],
        mu_post=outputs["mu_post"], logvar_post=outputs["logvar_post"],
    )
    head_averaged = outputs["attn_weights"].mean(dim=2)
    entropy = attention_entropy(outputs["attn_weights"]).mean(dim=-1)

    def _at(tensor: torch.Tensor, sample: int, index: torch.Tensor) -> np.ndarray:
        """Gather one sample's dense tensor at its decoded anchors, as float64 on the host."""
        return tensor[sample].index_select(0, index).detach().cpu().to(torch.float64).numpy()

    gathered: List[traces.SegmentTrace] = []
    for position, (_, row) in enumerate(rows.iterrows()):
        valid = anchor_valid[position].to(torch.bool)
        index = anchors[position][valid].to(torch.long)
        kl_lag_map = _at(outputs["source_kl_lag_map"], position, index)
        vectors = {
            "mu_prior": _at(outputs["mu_prior"], position, index),
            "mu_post": _at(outputs["mu_post"], position, index),
            "logvar_prior": _at(outputs["logvar_prior"], position, index),
            "logvar_post": _at(outputs["logvar_post"], position, index),
            "kld_per_dim": _at(kld_btd, position, index),
            "kld_per_head": _at(outputs["kld_per_t_per_head"], position, index),
            "kl_lag_map": kl_lag_map,
            "attention_lag_map": _at(head_averaged, position, index),
        }
        scalars = {
            "kld_per_t": _at(outputs["kld_per_t"], position, index),
            "coverage": coverage[position][valid].detach().cpu().to(torch.float64).numpy(),
            "argmax_lag": kl_lag_map.argmax(axis=-1).astype(np.float64),
            "attention_entropy_nats": _at(entropy, position, index),
        }
        gathered.append(
            traces.SegmentTrace(
                guid=str(row["guid"]),
                epoch=float(row["epoch"]),
                clinical_class=clinical_class,
                subgroup=subgroup,
                anchor=index.detach().cpu().numpy(),
                contributing=contributing[position][valid].detach().cpu().numpy(),
                scalars=scalars,
                vectors=vectors,
                clocks={column: float(row[column]) for column in CLOCK_COLUMNS if column in row.index},
            )
        )
    return gathered


@torch.no_grad()
def trace_recording(
    task: Any,
    loader: Any,
    rows: pd.DataFrame,
    *,
    clinical_class: Optional[str],
    subgroup: Optional[str],
) -> List[traces.SegmentTrace]:
    """Re-read every segment of one recording and reduce each forward to a trace.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, for its dataset, its collation and its batch size.
        rows: The recording's rows from :func:`recording_rows`, ascending in ``dataset_index``.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.

    Returns:
        The segment traces, in dataset order; the caller sorts them by ``epoch``.
    """
    model = task.orig_model
    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    batch_size = max(1, int(getattr(loader, "batch_size", None) or 1))
    segments = subset_loader(loader, list(rows["dataset_index"]), batch_size=batch_size)
    gathered: List[traces.SegmentTrace] = []
    position = 0
    for batch in segments:
        guids = batch_field(batch, "guid")
        count = len(guids) if isinstance(guids, (list, tuple)) else 1
        batch_rows = rows.iloc[position:position + count]
        position += count
        check_batch_identity(batch, batch_rows)
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        y_st, y_ph, u_stream, _target_features, weight = model_inputs(task, moved)
        outputs = model(
            y_st, y_ph, u_stream, anchor_phase=anchor_phase, anchor_stride=anchor_stride
        )
        segment_traces = gather_segment_traces(
            model, outputs, weight, batch_rows,
            clinical_class=clinical_class, subgroup=subgroup,
        )
        traces.attach_raw_signals(segment_traces, batch)
        gathered.extend(segment_traces)
    return gathered


# =============================================================================
# Joining what the pass already measured
# =============================================================================
def join_collected_anchors(anchors: pd.DataFrame, per_anchor: pd.DataFrame) -> pd.DataFrame:
    """Attach the collection pass's per-anchor scores to the trace's rows, by ``(guid, epoch, anchor)``.

    A left join: an anchor the pass never scored -- a segment a cap left uncollected -- keeps its
    re-read values and carries ``NaN`` in every joined column. The epoch is matched on its
    rounded value, because the two sides read it through different casts.

    Args:
        anchors: The trace's full-form rows.
        per_anchor: The collection pass's per-anchor table.

    Returns:
        The rows with :data:`JOINED_COLUMNS` and :data:`COLLECTED_KL_COLUMN` appended.
    """
    if anchors.empty or per_anchor is None or per_anchor.empty:
        return anchors
    wanted = [name for name in JOINED_COLUMNS if name in per_anchor.columns]
    keep = ["guid", "epoch", "anchor", *wanted]
    if "kld_per_t" in per_anchor.columns:
        keep.append("kld_per_t")
    table = per_anchor[keep].copy()
    table["guid"] = table["guid"].astype(str)
    table["_stamp"] = [epoch_stamp(value) for value in table["epoch"]]
    table["anchor"] = table["anchor"].astype(np.int64)
    table = table.drop(columns=["epoch"]).rename(columns={"kld_per_t": COLLECTED_KL_COLUMN})
    left = anchors.copy()
    left["_stamp"] = [epoch_stamp(value) for value in left["epoch"]]
    left["guid"] = left["guid"].astype(str)
    merged = left.merge(table, on=["guid", "_stamp", "anchor"], how="left")
    return merged.drop(columns=["_stamp"])


def kl_agreement(anchors: pd.DataFrame) -> Optional[float]:
    """The worst absolute disagreement between the re-read and the collected divergence, or ``None``."""
    if COLLECTED_KL_COLUMN not in anchors.columns or "kld_per_t" not in anchors.columns:
        return None
    both = anchors[[COLLECTED_KL_COLUMN, "kld_per_t"]].dropna()
    if both.empty:
        return None
    return float((both[COLLECTED_KL_COLUMN] - both["kld_per_t"]).abs().max())


# =============================================================================
# The registry entry point
# =============================================================================
def run_recording_traces_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Trace a seeded, class-balanced draw of recordings through every one of their segments.

    Args:
        context: The analysis context, read for the per-sample and per-anchor tables and -- with
            the pages, the probe and the occlusion readout -- for the task and the loader the
            segments are re-read through.
        eval_config: The validated block, for ``caps.traces_per_class`` and the draw's seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the selection accounting, the per-recording manifest, the
        divergence agreement and what failed. A pass with no model records a skip.
    """
    collection = context.collection
    per_sample = getattr(collection, "per_sample", None)
    per_anchor = getattr(collection, "per_anchor", None)
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        reason = (
            "a recording trace is the forward output of every segment of a recording, so it is "
            "re-read from the loader rather than read off a table; this pass built no model and "
            "no loader, which is what an offline re-run against a finished directory is"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"n_samples": None, "composition": {}, "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    seed = int(eval_config.get("seed", 0)) + traces.TRACE_DRAW_SEED_OFFSET
    caps = eval_config.get("caps") or {}
    per_class = int(caps.get(traces.TRACES_CAP) or traces.DEFAULT_TRACES_PER_CLASS)
    delay_steps = int(((collection.results or {}).get("lag") or {}).get("delay_steps") or 0)
    record = dict(getattr(collection, "record", None) or {})
    break_after_s = lag_axis.break_tolerance_s(record)

    window_hours = eval_config.get("max_hours_before_delivery")
    # The bound the run reads its clocks over: only the segments recorded within it are traced,
    # counted for eligibility, or drawn from, so a bounded run's traces describe the population
    # its trajectories do.
    index_map = cohort.within_horizon_index(dataset_index_map(loader), window_hours)
    recordings = recording_frame(per_sample, index_map)
    chosen, accounting = traces.select_recordings(recordings, per_class=per_class, seed=seed)

    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    anchor_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    lag_seconds: Optional[np.ndarray] = None
    for _, choice in chosen.iterrows():
        guid = str(choice["guid"])
        clinical_class = choice[labels.CLASS_COLUMN]
        subgroup = choice[labels.SUBGROUP_COLUMN] if not pd.isna(choice[labels.SUBGROUP_COLUMN]) else None
        rows = recording_rows(guid, index_map, per_sample)
        if rows.empty:
            # Counted by name rather than silently absent: the table labelled a recording the
            # dataset cannot place, which is a fault in the index mapping and not a short trace.
            failures.append({"guid": guid, "error": "no dataset row resolves to this recording"})
            continue
        try:
            segments = trace_recording(
                task, loader, rows, clinical_class=clinical_class, subgroup=subgroup
            )
            if lag_seconds is None:
                n_lags = int(segments[0].vectors["kl_lag_map"].shape[-1])
                lag_seconds = lag_axis.compensated_seconds_axis(n_lags, delay_steps)
            recording = traces.assemble_recording(
                segments, lag_profiles=LAG_PROFILES, lag_seconds=lag_seconds,
                break_after_s=break_after_s,
            )
            recording.anchors = join_collected_anchors(recording.anchors, per_anchor)
            # The joined scores reach the summary form too, averaged over each segment's scored
            # anchors, so the summary figure can draw the forecast gap beside the divergence.
            traces.add_segment_means(recording, JOINED_COLUMNS)
            stem = traces.recording_stem(guid, subgroup)
            class_dir = directory / traces.class_dirname(clinical_class)
            arrays = traces.write_recording_arrays(
                class_dir / f"{stem}{traces.FULL_ARRAYS_SUFFIX}.npz", recording,
                lag_seconds=lag_seconds,
            )
            figure = figures.render_figure(
                traces.build_recording_figure(
                    recording, panels=PANELS, lag_seconds=lag_seconds, caveat=lag_axis.GROUP_DELAY_CAVEAT
                ),
                class_dir / f"{stem}{traces.TRACE_FIGURE_SUFFIX}",
            )
            dashboard = traces_html.write_recording_dashboard(
                traces_html.build_recording_dashboard(
                    recording, panels=PANELS, lag_seconds=lag_seconds, caveat=lag_axis.GROUP_DELAY_CAVEAT
                ),
                class_dir / f"{stem}{traces.TRACE_FIGURE_SUFFIX}",
            )
        except Exception as error:  # noqa: BLE001 - one recording is not worth the rest of them
            logger.warning(f"{ANALYSIS_DIRNAME}: recording {guid} failed: {error}")
            failures.append({"guid": guid, "error": f"{type(error).__name__}: {error}"})
            continue
        anchor_frames.append(recording.anchors)
        summary_frames.append(recording.summary)
        span = recording.summary[cohort.HOURS_COLUMN]
        manifest.append(
            {
                "guid": guid,
                labels.CLASS_COLUMN: clinical_class,
                labels.SUBGROUP_COLUMN: subgroup,
                "n_segments": int(len(recording.segments)),
                "n_segments_collected": int(rows["collected"].sum()),
                "n_anchors": int(len(recording.anchors)),
                "n_contributing": int(recording.anchors["contributing"].sum()),
                "span_hours": float(span.max() - span.min()) if len(span) else float("nan"),
                "arrays_file": Path(arrays).relative_to(directory).as_posix(),
                "figure_file": Path(figure).relative_to(directory).as_posix(),
                "dashboard_file": Path(dashboard).relative_to(directory).as_posix(),
            }
        )

    anchors = pd.concat(anchor_frames, ignore_index=True) if anchor_frames else pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not anchors.empty:
        anchors.to_parquet(directory / traces.ANCHOR_TRACE_FILENAME, index=False)
    summary.to_csv(directory / traces.SEGMENT_SUMMARY_FILENAME, index=False)
    pd.DataFrame(manifest, columns=list(traces.MANIFEST_COLUMNS)).to_csv(
        directory / traces.MANIFEST_FILENAME, index=False
    )
    summary_figure = figures.render_figure(
        traces.build_summary_figure(
            summary, metrics=SUMMARY_METRICS, window_hours=window_hours,
        ),
        directory / traces.SUMMARY_FIGURE,
    )
    agreement = kl_agreement(anchors)
    if agreement is not None and agreement > KL_AGREEMENT_TOLERANCE:
        logger.warning(
            f"{ANALYSIS_DIRNAME}: the re-read divergence disagrees with the collected one by up "
            f"to {agreement:.3g} nats; the traces may not describe the forward the tables do"
        )
    logger.info(
        f"{ANALYSIS_DIRNAME}: traced {len(manifest)} recording(s) over "
        f"{int(summary['n_anchors'].sum()) if len(summary) else 0} anchor(s), {len(failures)} failed"
    )
    by_class = (
        {str(name): int(count) for name, count in
         pd.DataFrame(manifest)[labels.CLASS_COLUMN].value_counts().items()}
        if manifest else {}
    )
    return {
        # Segments, so this analysis's population is comparable with every other analysis's.
        "n_samples": int(len(summary)),
        "composition": {
            "n_recordings": int(len(manifest)),
            "n_recordings_by_class": by_class,
            "n_segments": int(len(summary)),
            "n_segments_collected": int(sum(entry["n_segments_collected"] for entry in manifest)),
        },
        "plan": {
            "capped": True, "traces_per_class": per_class, "seed": seed,
            "min_segments": traces.MIN_SEGMENTS_PER_TRACE,
            "anchor_phase": DENSE_ANCHOR_GEOMETRY[0], "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
            "delay_steps": delay_steps,
            "break_tolerance_s": float(break_after_s),
            "max_hours_before_delivery": None if window_hours is None else float(window_hours),
            "max_hours_before_delivery_applied": window_hours is not None,
        },
        "selection": accounting,
        "recordings": manifest,
        # The check that the re-read forward is the collected one: the worst per-anchor
        # disagreement of the two divergences, ``None`` where no anchor was collected.
        "kl_agreement_max_abs": agreement,
        "kl_agreement_tolerance": KL_AGREEMENT_TOLERANCE,
        "failures": failures,
        "files": [
            traces.MANIFEST_FILENAME, traces.SEGMENT_SUMMARY_FILENAME,
            *([traces.ANCHOR_TRACE_FILENAME] if not anchors.empty else []),
            str(Path(summary_figure).name),
        ],
    }
