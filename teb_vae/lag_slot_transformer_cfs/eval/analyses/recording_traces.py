r"""The per-recording traces as a registered analysis: a class-balanced draw followed segment by segment.

A thin wrapper over :mod:`teb_vae.lag_slot_transformer_cfs.eval.recording_traces`, which owns the
gather on the anchor axis, the selection, the files and the figures. What this module adds is the
family's protocol around it: the analysis is selected by name, runs under the failure-isolating
wrapper, reads the identities off the shared per-sample table rather than off a record the pass
kept, and records a skip when the pass built no model -- which is what an offline re-run against a
finished directory is.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.traces import ANALYSIS_DIRNAME
from teb_vae.lag_slot_transformer_cfs.eval import recording_traces as stage

#: The identity columns the stage draws its selection from, off the shared per-sample table.
IDENTITY_COLUMNS = ("guid", "epoch", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN)


def run_recording_traces_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Trace a seeded, class-balanced draw of recordings through every one of their segments.

    Args:
        context: The analysis context, read for the per-sample table's identities and -- with
            the pages and the attributions -- for the task and the loader the segments are
            re-read through.
        eval_config: The validated block, for ``caps.traces_per_class`` and the seed.
        output_dir: The results directory; the stage writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the stage's own block. A pass with no model records a skip.
    """
    del probe
    collection = context.collection
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        reason = (
            "a trace is the forward output of every segment of a recording, so it is re-read "
            "through the model rather than off a table; this pass built no model and no loader, "
            "which is what an offline re-run against a finished directory is"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"n_samples": None, "composition": {}, "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    per_sample = collection.per_sample
    identities = per_sample[[name for name in IDENTITY_COLUMNS if name in per_sample.columns]]
    block = stage.run_recording_traces(
        task, loader, identities,
        eval_config=dict(eval_config), results_dir=output_dir,
        geometry_record=dict((getattr(collection, "record", None) or {}).get("geometry") or {}),
    )
    return {
        # Segments traced, or None when the stage drew nothing: a capped analysis reports what it
        # actually re-read, and a skip reports no population rather than a population of zero.
        "n_samples": int(block["n_segments"]) if block.get("n_segments") else None,
        "composition": dict(block.get("n_recordings_by_class") or {}),
        **block,
    }


__all__ = ["IDENTITY_COLUMNS", "run_recording_traces_analysis"]
