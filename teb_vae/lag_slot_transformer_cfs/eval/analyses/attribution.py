r"""The Captum attributions as a registered analysis, over the family's shared core.

A thin wrapper over :mod:`teb_vae.lag_slot_transformer_cfs.eval.attribution`, which owns this
cell's wrapper on the anchor axis, its baselines and its layer split. What this module adds is the
family's protocol: selected by name, run under the failure-isolating wrapper, identities read off
the shared per-sample table, and a recorded skip when the pass built no model.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_slot_transformer_cfs.eval import attribution as stage
from teb_vae.lag_slot_transformer_cfs.eval.analyses.recording_traces import IDENTITY_COLUMNS


def run_attribution_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attribute a class-balanced draw of segments back over the three input streams.

    Args:
        context: The analysis context, read for the per-sample table's identities, the merged
            configuration the channel map is built from, and the task and loader the segments
            are re-read and differentiated through.
        eval_config: The validated block, for the cap, the seed and the lag bands.
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
            "an attribution is a gradient of the forward, which no table can hold; this pass "
            "built no model and no loader, which is what an offline re-run against a finished "
            "directory is"
        )
        logger.warning(f"{core.ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"n_samples": None, "composition": {}, "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    per_sample = collection.per_sample
    identities = per_sample[[name for name in IDENTITY_COLUMNS if name in per_sample.columns]]
    block = stage.run_attribution(
        task, loader, identities,
        config=dict(getattr(context, "config", None) or {}),
        eval_config=dict(eval_config), results_dir=output_dir,
        geometry_record=dict((getattr(collection, "record", None) or {}).get("geometry") or {}),
    )
    # The pass's own block already carries the protocol's keys where it ran; a skip carries
    # none, and reports no population rather than a population of zero. ``capped`` is stated
    # whichever it was: the attributed segments are a draw, never the split.
    record: Dict[str, Any] = {"n_samples": None, "composition": {}, **block}
    record["plan"] = {"capped": True, **dict(block.get("plan") or {})}
    return record


__all__ = ["run_attribution_analysis"]
