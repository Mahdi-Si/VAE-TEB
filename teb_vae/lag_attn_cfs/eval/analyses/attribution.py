r"""Captum attributions of the divergence, the forecast gain and the lag readout to the inputs.

Every other readout of the lag structure here is observational -- the attention over lags, the
KL attribution built from it -- or interventional on one axis, the lag band ``occlusion`` zeroes.
This analysis asks the mechanistic question directly: **which input coefficients, at which stored
steps and which channels, drove** the divergence $K_t$, the mean-decoded forecast gap and the
attention mass on each lag band, at a chosen anchor of a chosen segment. The arithmetic -- the
wrapper that turns the dense forward into one scalar per anchor, the two baselines, the
integrated gradients with their entry point, the layer split on the posterior head's per-head
fusion outputs, the band ablation, the reductions and the figures -- lives one layer down in
:mod:`~teb_vae.lag_attn_cfs.eval.attributions` and the pass in
:mod:`~teb_vae.lag_attn_cfs.eval.attribution_pass`, shared with the lag-residual cell so the two
directories hold the same tables under the same names.

**This is one of the analyses that touch the model**, for the reason ``occlusion`` does: an
attribution is a gradient of a forward that has to be taken, and no table carries it. The
segments are re-read through the sequential subset loader with the identity check, exactly as the
traces are, and a pass with no model records a skip.

**The selection is a seeded, class-balanced draw of recordings, one segment each** -- the
recording-level draw of :func:`~teb_vae.lag_attn_cfs.eval.traces.select_recordings` at an
eligibility floor of one segment, rather than the per-segment class-balanced draw the pages use.
Recording level, because every summary this analysis reports is over recordings and one segment
per recording keeps every recording one unit; the pages' segment-level draw would let one
recording contribute several segments to a class's mean. The cap is
``eval_config.caps.attribution_segments``.

**A late-window strata pass follows the main pass**
(:mod:`~teb_vae.lag_attn_cfs.eval.attribution_strata`): in the last two hours before delivery it
attributes high- against low-$K_t$ anchors, and anchors inside against between contractions, and
compares them as paired differences per recording, by class. Its cap is
``eval_config.caps.attribution_strata_segments`` and its block is ``strata``.

**What is joined rather than recomputed.** The frequency band of every declared input channel
comes off the channel map ``band_partition`` persisted; the occlusion analysis's per-band delta
and the band-resolved skill come off their own tables where those passes ran in this directory,
and are recorded as absent otherwise -- the dependency is on files, so ``--only attribution``
against a finished directory still works with a checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from teb_vae.lag_attn_cfs.eval import attribution_pass, attribution_strata, lag_axis
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval.dataset_rows import dataset_index_map
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = core.ANALYSIS_DIRNAME


def run_attribution_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attribute the divergence, the gap and the lag readout over the inputs of a drawn set of segments.

    Args:
        context: The analysis context, read for the per-sample table's labels and -- with the
            pages, the traces and the occlusion readout -- for the task and the loader the
            segments are re-read through.
        eval_config: The validated block, for ``caps.attribution_segments``, the seed and the
            lag bands.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the selection accounting, the measured structural checks, the
        cost, the method record, the trace manifest and the files. A pass with no model records
        a skip.
    """
    collection = context.collection
    per_sample = getattr(collection, "per_sample", None)
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        reason = (
            "an attribution is a gradient of a forward that has to be taken, so it is re-read from "
            "the loader rather than read off a table; this pass built no model and no loader, "
            "which is what an offline re-run against a finished directory is"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"n_samples": None, "composition": {}, "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    results_dir = Path(output_dir)
    delay_steps = int(((collection.results or {}).get("lag") or {}).get("delay_steps") or 0)
    record = dict(getattr(collection, "record", None) or {})
    model = task.orig_model
    n_lags = int(model.lag_attn.L)
    index_map = dataset_index_map(loader)
    lag_seconds = lag_axis.compensated_seconds_axis(n_lags, delay_steps)
    channel_map = attribution_pass.read_channel_map(results_dir)
    block = attribution_pass.run_pass(
        task, loader, per_sample, index_map, core.ATTENTION_CELL,
        eval_config=eval_config, results_dir=results_dir,
        lag_seconds=lag_seconds,
        break_after_s=lag_axis.break_tolerance_s(record),
        segment_stride_s=lag_axis.segment_stride_s(record),
        channel_map=channel_map,
        occlusion=attribution_pass.read_occlusion_summary(results_dir),
        spectral=attribution_pass.read_spectral_bands(results_dir),
        delay_steps=delay_steps,
        per_anchor=getattr(collection, "per_anchor", None),
    )
    # The late-window strata: high against low K_t and inside against between contractions.
    caps = dict(eval_config.get("caps") or {})
    configured = dict(eval_config.get("occlusion_bands") or {})
    block["strata"] = attribution_strata.run_strata(
        task, loader, attribution_pass.labelled_segments(per_sample), index_map, core.ATTENTION_CELL,
        per_anchor=getattr(collection, "per_anchor", None),
        anchor_vectors=getattr(collection, "anchor_vectors", None),
        cap=int(caps.get(attribution_strata.STRATA_CAP_NAME) or attribution_strata.DEFAULT_STRATA_SEGMENTS),
        seed=int(eval_config.get("seed", 0)) + core.DRAW_SEED_OFFSET,
        window_s=float(eval_config.get("event_lag_window_s") or attribution_strata.DEFAULT_CONTRACTION_WINDOW_S),
        directory=results_dir / core.ANALYSIS_DIRNAME, lag_seconds=lag_seconds,
        lag_bands={str(name): (int(span[0]), int(span[1])) for name, span in configured.items()},
        channel_groups=core.channel_groups_from_map(channel_map), n_steps=core.IG_STEPS,
        caveat=core.ATTRIBUTION_NOTE,
    )
    block["files"] = list(block.get("files", [])) + list(block["strata"]["files"])
    block["grouped_frames"] = [
        grouped_frame_entry(
            ANALYSIS_DIRNAME, core.RECORDINGS_FILENAME, attribution_pass.RECORDING_VALUE_COLUMNS
        )
    ]
    return block
