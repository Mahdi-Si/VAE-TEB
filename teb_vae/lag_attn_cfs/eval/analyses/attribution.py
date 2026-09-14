r"""Captum attributions of the divergence, the forecast gain and the lag readout to the inputs.

Every other readout of the lag structure here is observational -- the attention over lags, the
KL attribution built from it -- or interventional on one axis, the lag band ``occlusion`` zeroes.
This analysis asks the mechanistic question directly: **which input coefficients, at which stored
steps and which channels, drove** the divergence $K_t$, the mean-decoded forecast gap and the
attention mass on each lag band, at a chosen anchor of a chosen segment. The arithmetic -- the
wrapper that turns the dense forward into one scalar per anchor, the two baselines, the
integrated gradients with their entry point, the layer split on the posterior head's inputs, the
band ablation, the reductions and the figures -- lives one layer down in
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

from teb_vae.lag_attn_cfs.eval import attribution_pass, lag_axis
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
    block = attribution_pass.run_pass(
        task, loader, per_sample, dataset_index_map(loader), core.ATTENTION_CELL,
        eval_config=eval_config, results_dir=results_dir,
        lag_seconds=lag_axis.compensated_seconds_axis(n_lags, delay_steps),
        break_after_s=lag_axis.break_tolerance_s(record),
        channel_map=attribution_pass.read_channel_map(results_dir),
        occlusion=attribution_pass.read_occlusion_summary(results_dir),
        spectral=attribution_pass.read_spectral_bands(results_dir),
        delay_steps=delay_steps,
    )
    block["grouped_frames"] = [
        grouped_frame_entry(
            ANALYSIS_DIRNAME, core.RECORDINGS_FILENAME, attribution_pass.RECORDING_VALUE_COLUMNS
        )
    ]
    return block
