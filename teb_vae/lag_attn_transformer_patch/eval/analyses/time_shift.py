r"""``time_shift`` -- port; owner E1-O (``notes/EVAL_PLAN.md`` §2).

**Question.** Is the coupling about the UP **at the right moment**, or about this patient's UP at
any moment? The source is swapped for the nearest non-overlapping segment of the same recording.
Everything target-side stays the segment's own. The readouts are ΔK and the mean-decoded Δgap; the
verdict is ``coupling_is_time_specific``.

**No port was needed.** The shared implementation runs unchanged on the eval view (plan D3/D4):
* ``metrics.model_inputs`` returns ``(y_patch, empty, u_patch, summaries, weight)``;
* the forward is the CFS five-argument form;
* ``model._build_forecast_target`` is the loss's own ``a + 1 + τ`` gather;
* the partner's ``u_patch`` is patchified from its raw ``up`` under ``model.source_validity``;
* ``source_gate`` is ``None`` and there is no persistence input.

So ``score_pairs`` (``teb_vae/lag_attn_cfs/eval/analyses/time_shift.py``) is the patch scorer, and
ΔK, Δg and the verdict are computed exactly as in CFS. The partner is at least one stored segment
away (``T·4 + 2·trim`` s), so this is a recording-level timing control, not a lag probe. The
sub-token delay sweep is ``raw_shift``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from teb_vae.lag_attn_cfs.eval.analyses import time_shift as _shared

#: ``(headline_name, key)`` pairs registered as ``("time_shift", "headline", key)``. None: the
#: shared entries already resolve on this module's (identical) headline block.
HEADLINE: Tuple[Tuple[str, str], ...] = ()


def run_time_shift_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The shared swap control on the eval view's patch streams (``caps.time_shift`` segments)."""
    result = _shared.run_time_shift_analysis(
        context, eval_config=eval_config, output_dir=output_dir, probe=probe
    )
    result.setdefault("plan", {})["implementation"] = (
        "shared score_pairs on the eval view: u_patch swapped for the partner segment's u_patch"
    )
    return result
