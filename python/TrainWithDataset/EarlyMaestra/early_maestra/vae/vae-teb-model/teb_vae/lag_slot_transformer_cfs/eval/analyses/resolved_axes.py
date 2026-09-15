r"""The two resolved axes: the gap and every margin by horizon step and by stored target block.

Every scored arm is resolved by horizon step and by stored target block under the shared draws,
so a source that informs the first predicted step and not the last, or the scattering block and
not the phase-harmonic one, is a statement the run can make. Each position is the marginal mixture
of that subset's own likelihood factors, so the positions do not sum to the joint block score and
are read for their shape rather than their total.

One table and two figures, from the results block the pass assembled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from teb_vae.lag_attn_cfs.eval import figures_seam as figure_seam
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import figures
from teb_vae.lag_slot_transformer_cfs.eval.analyses.lag_suppression import write_rows
from teb_vae.lag_slot_transformer_cfs.eval.lag_metrics import SUPPRESSION_PREFIX

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "resolved_axes"

#: The horizon-resolved table: one row per scored arm and horizon step, with the interval.
HORIZON_FILENAME = "horizon_resolved.csv"

#: The figures, by stem.
HORIZON_FIGURE = "horizon_resolved"
BLOCK_FIGURE = "block_resolved"


def horizon_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per arm and horizon step, assembled from the results block's horizon axis.

    Args:
        results: The assembled results.

    Returns:
        The rows; empty when the run resolved no horizon axis.
    """
    block = results.get("horizon_resolved") or {}
    positions = list(block.get("positions") or [])
    rows: List[Dict[str, Any]] = []
    series: List[Tuple[str, Mapping[str, Any]]] = [
        (f"nll_{arm}", record) for arm, record in (block.get("nll") or {}).items()
    ]
    if block.get("pred_gap"):
        series.append(("pred_gap", block["pred_gap"]))
    series += [
        (f"margin_{SUPPRESSION_PREFIX}{band}", record)
        for band, record in (block.get("band_margins") or {}).items()
    ]
    series += [
        (f"margin_{name}", record)
        for name, record in (block.get("control_margins") or {}).items()
    ]
    for name, record in series:
        for position, step in enumerate(positions):
            rows.append(
                {
                    "series": name,
                    "horizon_step": step,
                    "point": record["point"][position],
                    "lo": record["lo"][position],
                    "hi": record["hi"][position],
                    "n": record.get("n"),
                }
            )
    return rows


def run_resolved_axes_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the horizon table and draw the two resolved-axis figures.

    Args:
        context: The analysis context, read for the results block.
        eval_config: The validated block. Unused beyond the protocol.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys and the files written.
    """
    del eval_config, probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    per_sample = collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    rows = horizon_rows(results)
    write_rows(directory / HORIZON_FILENAME, rows)
    written: List[str] = [HORIZON_FILENAME] if rows else []
    for stem, figure in (
        (HORIZON_FIGURE, figures.build_horizon_figure(results)),
        (BLOCK_FIGURE, figures.build_block_figure(results)),
    ):
        written.append(str(figure_seam.render_figure(figure, directory / stem, tight=False).name))

    horizon = results.get("horizon_resolved") or {}
    block = results.get("block_resolved") or {}
    return {
        "n_samples": scored_sample_count(per_sample, "mc_nll_full_block"),
        "composition": {
            "n_horizon_steps": int(len(horizon.get("positions") or [])),
            "blocks": list(block.get("positions") or []),
        },
        "plan": {"capped": False},
        "files": written,
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "BLOCK_FIGURE",
    "HORIZON_FIGURE",
    "HORIZON_FILENAME",
    "horizon_rows",
    "run_resolved_axes_analysis",
]
