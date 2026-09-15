r"""Per-recording traces for the lag-residual cell: chosen recordings followed segment by segment.

The scoring pass reduces every recording to one row of the per-recording table, and the batch
records it reduces from are discarded once it has. This stage runs **after** the pass, on the
identities the pass recorded, and follows a class-balanced draw of recordings through every segment
the dataset holds for them, in ``epoch`` order: one dense forward per segment with the proposals
retained, reduced to the latent parameters, the divergence and its per-coordinate split, the
bounded update, and the per-lag proposal norm at every decoded anchor. What comes out is the
family's shared trace -- the same two forms, the same file layout and the same figures the
lag-attentive cells write -- so a reader can put one recording's evolution under this
architecture beside the same recording's under another.

**What the lag readout is here.** This architecture poses no query and computes no distribution
over lags; its per-lag quantity is a signed proposal $r^\mu_{t,\ell}$, an update rather than an
allocation. The lag map on a trace is therefore the proposal **norm** $\lVert r^\mu_{t,\ell}
\rVert_2$ at every anchor and lag -- what the head emitted before the sum and the limiter --
masked to the lags that carried any available channel, and its shape statistics read where the
proposals are large rather than where a divergence was attributed. That is stated on every
artifact rather than left to be inferred from the column names, and none of the names is one of
the attention-shaped keys this package refuses.

**Which arms trace what.** The proposal map exists on the recommended explicit-sum fusion with a
source pathway; the comparator's normalised fusion retains no per-lag array and the target-only
arm has no source pathway at all, and on both the trace carries the latent and the divergence
alone, with the lag family recorded as absent rather than zero-filled.

**The per-anchor scores are single-draw.** Both branches' block scores come off the forward's own
decoded forecasts under the one shared $\epsilon$, exactly as the training objective sees them;
the pass's Monte Carlo scores are per recording, not per anchor, so there is nothing to join.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import cohort, frames, lag_axis, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import (
    check_batch_identity,
    dataset_index_map,
    subset_loader,
)
from teb_vae.lag_attn_cfs.eval.metrics import (
    anchor_support,
    batch_field,
    batch_guids,
    model_inputs,
)
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors

#: The anchor geometry every trace forward runs at: dense, as the pass itself.
DENSE_ANCHOR_GEOMETRY: Tuple[int, int] = (0, 1)

#: The one lag family a forward of this cell can yield per anchor, and the prefix its shape
#: statistics are written under. The proposal norm, not an allocation; see the module docstring.
LAG_PROFILES: Dict[str, str] = {"proposal_lag_map": "proposal_lag"}

#: Status values the summary block carries, so a reader can tell "not run" from "ran and found
#: nothing" from "ran".
STATUS_SKIPPED = "SKIPPED"
STATUS_EMPTY = "EMPTY"
STATUS_TRACED = "TRACED"

#: The qualification every artifact of this stage carries about its lag family.
PROPOSAL_QUALIFICATION = (
    "The lag map on these traces is the proposal norm ||r_{t,l}||_2 at every anchor and lag: an "
    "update magnitude before the sum and the limiter, not a distribution over lags and not an "
    "allocation of the divergence. Its shape statistics read where proposals are large. The axis "
    "is stored-coefficient time."
)

#: The rows of every recording's figure, top to bottom.
PANELS: Tuple[Any, ...] = (
    traces.LinePanel(("kld_per_t",), "Divergence $K_t$ per anchor", "nats", labels=("$K_t$",),
                     segment_mean=True),
    traces.LinePanel(("pred_gap",), "Forecast gap per anchor, single draw", "nats",
                     labels=("base $-$ full",), segment_mean=True),
    traces.HeatmapPanel("mu_post", "Full-branch mean $\\mu^q$ over the latent coordinates", "coordinate",
                        symmetric=True),
    traces.HeatmapPanel("update_mean", "Bounded mean update $a_t$", "coordinate", symmetric=True),
    traces.HeatmapPanel("kld_per_dim", "Divergence per latent coordinate, $K_{t,d}$", "coordinate"),
    traces.HeatmapPanel("proposal_lag_map", "Proposal norm $\\| r_{t,\\ell}\\|_2$ over lags", "",
                        lag_axis=True, argmax_column="proposal_argmax_lag"),
    traces.LinePanel(("mu_prior_norm", "delta_mu_norm"), "Latent norms", "latent units",
                     labels=("$\\|\\mu^p\\|_2$", "$\\|\\mu^q - \\mu^p\\|_2$")),
    traces.LinePanel(("mean_logvar_prior", "mean_logvar_post"), "Mean log-variance over coordinates", "",
                     labels=("prior", "posterior")),
    traces.LinePanel(("proposal_lag_centroid_s", "proposal_lag_median_s"),
                     "Lag centre of the proposal norm", "s (stored-coefficient time)",
                     labels=("centroid", "median")),
    traces.LinePanel(("cancellation_ratio_mean",), "Cancellation ratio of the mean update", "",
                     labels=("cancellation ratio",)),
)

#: The rows of the cross-recording summary figure.
SUMMARY_METRICS: Tuple[traces.SummaryMetric, ...] = (
    traces.SummaryMetric("kld_per_t", "nats per anchor", "Divergence $K_t$"),
    traces.SummaryMetric("pred_gap", "nats per anchor", "Single-draw forecast gap"),
    traces.SummaryMetric("delta_mu_norm", "latent units",
                         "Source shift of the latent mean, $\\|\\mu^q - \\mu^p\\|_2$"),
    traces.SummaryMetric("n_active_dims", "coordinates", "Active latent coordinates"),
    traces.SummaryMetric("proposal_lag_centroid_s", "s (stored-coefficient time)",
                         "Lag centroid of the proposal norm"),
    traces.SummaryMetric("cancellation_ratio_mean", "ratio", "Cancellation ratio of the mean update"),
)


# =============================================================================
# The identities the pass records
# =============================================================================
def batch_identity(batch: Any, batch_size: int) -> Dict[str, List[Any]]:
    """Read one batch's per-sample identity: recording, epoch, class and subgroup.

    The class is recovered from the weight-scaled ``target`` exactly as the family's labelling
    does, and is ``None`` for every sample when the loader was not asked for that field; the
    subgroup comes off the shard basename, which the dataset always stamps.

    Args:
        batch: A batch from the data module.
        batch_size: Its sample count.

    Returns:
        ``{'guid', 'epoch', clinical_class, subgroup}``, each a list of ``batch_size`` entries.
    """
    epoch = batch_field(batch, "epoch")
    epochs = (
        [float("nan")] * batch_size if epoch is None
        else [float(value) for value in np.asarray(
            epoch.detach().cpu() if isinstance(epoch, torch.Tensor) else epoch, dtype=np.float64
        ).reshape(-1)]
    )
    target, weight = batch_field(batch, "target"), batch_field(batch, "weight")
    classes: List[Optional[str]] = [
        None if target is None or weight is None
        else labels.class_name(labels.clinical_class_code(target[index], weight[index]))
        for index in range(batch_size)
    ]
    basenames = batch_field(batch, "source_file_basename")
    subgroups = [
        labels.subgroup_of(
            basenames[index] if isinstance(basenames, (list, tuple)) else basenames
        )
        for index in range(batch_size)
    ]
    return {
        "guid": batch_guids(batch, batch_size),
        "epoch": epochs,
        labels.CLASS_COLUMN: classes,
        labels.SUBGROUP_COLUMN: subgroups,
    }


def identity_frame(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Stack the pass's per-batch identities into one row per scored segment."""
    columns = ["guid", "epoch", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN]
    rows = {name: [] for name in columns}
    for record in records:
        identity = record.get("identity") or {}
        for name in columns:
            rows[name].extend(identity.get(name, []))
    return pd.DataFrame(rows)


# =============================================================================
# The forward, reduced to arrays
# =============================================================================
@torch.no_grad()
def gather_segment_traces(
    model: Any,
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    weight: torch.Tensor,
    rows: pd.DataFrame,
    *,
    likelihood: str,
    clinical_class: Optional[str],
    subgroup: Optional[str],
) -> List[traces.SegmentTrace]:
    r"""Reduce one dense forward to one trace per sample of the batch.

    Every latent tensor of this architecture already lives on the decoded anchor axis, so the
    reduction is a selection of the valid slots rather than a gather; the anchor written beside
    each row is still the forward's own ``anchor_index``.

    Args:
        model: The rebuilt net.
        outputs: The forward's dict, taken with ``return_proposals=True``.
        target: The gathered forecast target the forward's own forecasts are scored against.
        weight: The validity signal the task's target builder returned, for the masks.
        rows: The batch's rows, in batch order, carrying ``guid`` and ``epoch``.
        likelihood: The objective's likelihood, for the single-draw block scores.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.

    Returns:
        One trace per sample.
    """
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    mask, coverage, _kl_support = anchor_support(model, weight, outputs)
    contributing = contributing_anchors(mask) > 0.0
    nll_full, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood=likelihood, logvar=outputs["logvar_full"]
    )
    nll_base, _ = masked_raw_block_per_anchor(
        outputs["mu_base"], target, mask, likelihood=likelihood, logvar=outputs["logvar_base"]
    )
    # The proposal norm exists on the explicit-sum fusion with a source pathway only; elsewhere
    # the lag family is absent from the trace rather than zero-filled.
    proposals = outputs.get("mean_proposals")
    lag_valid = outputs.get("lag_valid")
    proposal_norm = None
    if proposals is not None and lag_valid is not None:
        proposal_norm = torch.where(
            lag_valid.to(torch.bool), proposals.norm(dim=-1),
            torch.full_like(proposals[..., 0], float("nan")),
        )

    def _rows(tensor: torch.Tensor, sample: int, valid: torch.Tensor) -> np.ndarray:
        """One sample's anchor-axis tensor at its valid slots, as float64 on the host."""
        return tensor[sample][valid].detach().cpu().to(torch.float64).numpy()

    gathered: List[traces.SegmentTrace] = []
    for position, (_, row) in enumerate(rows.iterrows()):
        valid = anchor_valid[position].to(torch.bool)
        vectors = {
            "mu_prior": _rows(outputs["mu_prior"], position, valid),
            "mu_post": _rows(outputs["mu_post"], position, valid),
            "logvar_prior": _rows(outputs["logvar_prior"], position, valid),
            "logvar_post": _rows(outputs["logvar_post"], position, valid),
            "kld_per_dim": _rows(outputs["kld_per_anchor_dim"], position, valid),
            "update_mean": _rows(outputs["update_mean"], position, valid),
        }
        scalars = {
            "kld_per_t": _rows(outputs["kld_per_anchor"], position, valid),
            "coverage": _rows(coverage, position, valid),
            "nll_base_block": _rows(nll_base, position, valid),
            "nll_full_block": _rows(nll_full, position, valid),
        }
        scalars["pred_gap"] = scalars["nll_base_block"] - scalars["nll_full_block"]
        if proposal_norm is not None:
            lag_map = _rows(proposal_norm, position, valid)
            vectors["proposal_lag_map"] = lag_map
            # The argmax over the lags that carried a channel; NaN where none did.
            finite = np.isfinite(lag_map)
            filled = np.where(finite, lag_map, -np.inf)
            scalars["proposal_argmax_lag"] = np.where(
                finite.any(axis=-1), filled.argmax(axis=-1).astype(np.float64), np.nan
            )
        if "cancellation_ratio_mean" in outputs:
            scalars["cancellation_ratio_mean"] = _rows(outputs["cancellation_ratio_mean"], position, valid)
        gathered.append(
            traces.SegmentTrace(
                guid=str(row["guid"]),
                epoch=float(row["epoch"]),
                clinical_class=clinical_class,
                subgroup=subgroup,
                anchor=anchors[position][valid].detach().cpu().numpy().astype(np.int64),
                contributing=contributing[position][valid].detach().cpu().numpy(),
                scalars=scalars,
                vectors=vectors,
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
        rows: The recording's rows, ascending in ``dataset_index``.
        clinical_class: The recording's class.
        subgroup: The recording's subgroup.

    Returns:
        The segment traces, in dataset order; the assembly sorts them by ``epoch``.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    phase, stride = DENSE_ANCHOR_GEOMETRY
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
        y_st, y_ph, u_stream, target_features, weight = model_inputs(task, moved)
        outputs = model(
            y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride, return_proposals=True
        )
        target = model._build_forecast_target(target_features, outputs["anchor_index"])
        gathered.extend(
            gather_segment_traces(
                model, outputs, target, weight, batch_rows, likelihood=likelihood,
                clinical_class=clinical_class, subgroup=subgroup,
            )
        )
    return gathered


# =============================================================================
# The stage
# =============================================================================
def _recording_rows(guid: str, index_map: Dict[Tuple[str, Optional[int]], int]) -> pd.DataFrame:
    """Every dataset row of one recording, in dataset order."""
    listed = sorted(
        (index, stamp) for (listed_guid, stamp), index in index_map.items()
        if listed_guid == str(guid) and stamp is not None
    )
    return pd.DataFrame(
        {
            "guid": [str(guid)] * len(listed),
            "epoch": [float(stamp) for _, stamp in listed],
            "dataset_index": [int(index) for index, _ in listed],
        }
    )


def run_recording_traces(
    task: Any,
    loader: Any,
    identities: pd.DataFrame,
    *,
    eval_config: Dict[str, Any],
    results_dir: Any,
    geometry_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Trace a seeded, class-balanced draw of recordings and write the trace directory.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader the pass walked.
        identities: One row per scored segment -- ``guid``, ``epoch``, class, subgroup -- as the
            pass recorded them.
        eval_config: The validated settings, for ``caps.traces_per_class`` and the seed.
        results_dir: The run's results directory; the stage writes into its own subdirectory.
        geometry_record: The collection-style geometry record the break tolerance is read from,
            or ``None`` for the family's default stride.

    Returns:
        The block the summary carries: a status, the selection accounting, the per-recording
        manifest, what failed and the files written. Never raises for a recording that failed --
        each is recorded by name -- but a stage that could not run at all says why.
    """
    directory = Path(str(results_dir)) / traces.ANALYSIS_DIRNAME
    caps = dict(eval_config.get("caps") or {})
    per_class = int(caps.get(traces.TRACES_CAP) or traces.DEFAULT_TRACES_PER_CLASS)
    seed = int(eval_config.get("seed", 0)) + traces.TRACE_DRAW_SEED_OFFSET
    plan = {
        "capped": True, "traces_per_class": per_class, "seed": seed,
        "min_segments": traces.MIN_SEGMENTS_PER_TRACE,
        "anchor_phase": DENSE_ANCHOR_GEOMETRY[0], "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
        "lag_qualification": PROPOSAL_QUALIFICATION,
    }
    if identities.empty or identities[labels.CLASS_COLUMN].isna().all():
        reason = (
            "no scored segment carries a clinical class, so no class-balanced draw can be made; "
            "the class is recovered from the weight-scaled target, which reaches a batch only "
            "when the override delta's load_fields names it"
        )
        logger.warning(f"{traces.ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"status": STATUS_SKIPPED, "reason": reason, "plan": plan, "files": []}

    index_map = dataset_index_map(loader)
    labelled = frames.per_recording_labels(identities)
    counts: Dict[str, int] = {}
    for guid, _stamp in index_map:
        counts[guid] = counts.get(guid, 0) + 1
    recordings = labelled.reset_index()
    recordings["n_segments"] = [int(counts.get(str(guid), 0)) for guid in recordings["guid"]]
    chosen, accounting = traces.select_recordings(recordings, per_class=per_class, seed=seed)
    if chosen.empty:
        return {"status": STATUS_EMPTY, "plan": plan, "selection": accounting,
                "reason": "no labelled recording holds enough segments to trace", "files": []}

    directory.mkdir(parents=True, exist_ok=True)
    break_after_s = lag_axis.break_tolerance_s({"geometry": dict(geometry_record or {})})
    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    anchor_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    lag_family_present = False
    lag_seconds = lag_axis.compensated_seconds_axis(int(task.orig_model.n_lags), 0)
    for _, choice in chosen.iterrows():
        guid = str(choice["guid"])
        clinical_class = choice[labels.CLASS_COLUMN]
        subgroup = choice[labels.SUBGROUP_COLUMN] if not pd.isna(choice[labels.SUBGROUP_COLUMN]) else None
        rows = _recording_rows(guid, index_map)
        if rows.empty:
            failures.append({"guid": guid, "error": "no dataset row resolves to this recording"})
            continue
        try:
            segments = trace_recording(task, loader, rows, clinical_class=clinical_class, subgroup=subgroup)
            recording = traces.assemble_recording(
                segments, lag_profiles=LAG_PROFILES, lag_seconds=lag_seconds,
                break_after_s=break_after_s,
            )
            stem = traces.recording_stem(guid, subgroup)
            class_dir = directory / traces.class_dirname(clinical_class)
            arrays = traces.write_recording_arrays(
                class_dir / f"{stem}{traces.FULL_ARRAYS_SUFFIX}.npz", recording, lag_seconds=lag_seconds
            )
            figure = figures.render_figure(
                traces.build_recording_figure(
                    recording, panels=PANELS, lag_seconds=lag_seconds,
                    caveat=f"{PROPOSAL_QUALIFICATION} {lag_axis.GROUP_DELAY_CAVEAT}",
                ),
                class_dir / f"{stem}{traces.TRACE_FIGURE_SUFFIX}",
            )
        except Exception as error:  # noqa: BLE001 - one recording is not worth the rest of them
            logger.warning(f"{traces.ANALYSIS_DIRNAME}: recording {guid} failed: {error}")
            failures.append({"guid": guid, "error": f"{type(error).__name__}: {error}"})
            continue
        anchor_frames.append(recording.anchors)
        summary_frames.append(recording.summary)
        lag_family_present = lag_family_present or any(
            name in recording.segment_vectors for name in LAG_PROFILES
        )
        span = recording.summary[cohort.HOURS_COLUMN]
        manifest.append(
            {
                "guid": guid,
                labels.CLASS_COLUMN: clinical_class,
                labels.SUBGROUP_COLUMN: subgroup,
                "n_segments": int(len(recording.segments)),
                "n_segments_collected": int(
                    (identities["guid"].astype(str) == guid).sum()
                ),
                "n_anchors": int(len(recording.anchors)),
                "n_contributing": int(recording.anchors["contributing"].sum()),
                "span_hours": float(span.max() - span.min()) if len(span) else float("nan"),
                "arrays_file": Path(arrays).relative_to(directory).as_posix(),
                "figure_file": Path(figure).relative_to(directory).as_posix(),
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
            summary, metrics=SUMMARY_METRICS, window_hours=eval_config.get("max_hours_before_delivery"),
        ),
        directory / traces.SUMMARY_FIGURE,
    )
    by_class = (
        {str(name): int(count) for name, count in
         pd.DataFrame(manifest)[labels.CLASS_COLUMN].value_counts().items()}
        if manifest else {}
    )
    logger.info(
        f"{traces.ANALYSIS_DIRNAME}: traced {len(manifest)} recording(s) over "
        f"{int(len(anchors))} anchor(s), {len(failures)} failed"
    )
    return {
        "status": STATUS_TRACED if manifest else STATUS_EMPTY,
        "plan": plan,
        "selection": accounting,
        "n_recordings": int(len(manifest)),
        "n_recordings_by_class": by_class,
        "n_segments": int(len(summary)),
        # Whether the lag family was on the traces at all, which is an arm property: the
        # explicit-sum fusion with a source pathway carries it, the others do not. Presence of
        # the vectors, not finiteness of a statistic: a head that proposes nothing yet has a lag
        # family whose shape is undefined, which is a different statement from having none.
        "lag_family_present": bool(lag_family_present),
        "recordings": manifest,
        "failures": failures,
        "directory": traces.ANALYSIS_DIRNAME,
        "files": [
            traces.MANIFEST_FILENAME, traces.SEGMENT_SUMMARY_FILENAME,
            *([traces.ANCHOR_TRACE_FILENAME] if not anchors.empty else []),
            str(Path(summary_figure).name),
        ],
    }


__all__ = [
    "LAG_PROFILES",
    "PANELS",
    "PROPOSAL_QUALIFICATION",
    "STATUS_EMPTY",
    "STATUS_SKIPPED",
    "STATUS_TRACED",
    "SUMMARY_METRICS",
    "batch_identity",
    "gather_segment_traces",
    "identity_frame",
    "run_recording_traces",
    "trace_recording",
]
