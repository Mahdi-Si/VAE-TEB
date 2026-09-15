r"""The lag axis: band suppression, then the per-lag profile, in the order the design reads them.

Two figures and one table from the results block the pass assembled. The band figure is read
first -- whole-band and joint removals -- and the per-lag figure after it, because a single-lag
peak read off a window whose joint removal does nothing is noise. The table carries every lag's
exposure, latent profile and predictive margin in one row per lag, so the figures redraw from it.

Every artifact here carries the qualification the lag readouts must be read with: suppression
measures reliance on a fitted parameterisation, not a unique contribution and not a delay.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam as figure_seam
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import figures

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "lag_suppression"

#: The per-lag table: one row per candidate lag, carrying the exposure, the latent profile and
#: the predictive margin with its interval. What the lag figures are drawn from.
LAG_PROFILE_FILENAME = "lag_profile.csv"

#: The figures, by stem.
BAND_FIGURE = "band_suppression"
PROFILE_FIGURE = "lag_profile"


def lag_profile_rows(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per candidate lag, assembled from the results block's own lag readouts.

    Built from the results rather than from the tensors, so the table and the figure drawn from
    it carry exactly the numbers the summary does.

    Args:
        results: The assembled results.

    Returns:
        The rows, in lag order; empty when the run read no lag.
    """
    readouts = results.get("lag_readouts") or {}
    axis = readouts.get("lag_axis") or {}
    exposure = readouts.get("exposure") or {}
    profile = readouts.get("lag_profile") or {}
    latent = profile.get("latent") or {}
    predictive = profile.get("predictive") or {}
    margin = predictive.get("margin_nats") or {}
    anchors = list(exposure.get("per_lag_anchors") or [])
    if not anchors:
        return []
    channels = list(exposure.get("per_lag_channels") or [])
    step = float(axis.get("seconds_per_step", SECONDS_PER_STEP))
    delay = int(axis.get("delay_steps", 0) or 0)
    rows: List[Dict[str, Any]] = []
    for lag in range(len(anchors)):
        row: Dict[str, Any] = {
            "lag": lag,
            "seconds": step * (lag + delay),
            "anchors": float(anchors[lag]),
            "channels": float(channels[lag]) if lag < len(channels) else "",
        }
        for name in ("proposal_norm", "update_shift", "divergence_drop", "scale_proposal_norm"):
            values = latent.get(name)
            row[name] = "" if not values or values[lag] is None else float(values[lag])
        for part in ("point", "lo", "hi"):
            values = margin.get(part)
            row[f"predictive_margin_{part}"] = (
                "" if not values or lag >= len(values) else float(values[lag])
            )
        rows.append(row)
    return rows


def write_rows(path: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of homogeneous rows as a CSV, or nothing when there are none.

    Args:
        path: The file to write.
        rows: The rows, every one carrying the same keys.
    """
    if not rows:
        return
    columns = list(rows[0])
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_lag_suppression_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the per-lag table and draw the band and profile figures.

    Args:
        context: The analysis context, read for the results block.
        eval_config: The validated block, for the profile cap echoed into the plan.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the per-lag row count and the files written.
    """
    del probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    per_sample = collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    rows = lag_profile_rows(results)
    write_rows(directory / LAG_PROFILE_FILENAME, rows)
    written: List[str] = [LAG_PROFILE_FILENAME] if rows else []
    for stem, figure in (
        (BAND_FIGURE, figures.build_band_figure(results)),
        (PROFILE_FIGURE, figures.build_lag_profile_figure(results)),
    ):
        written.append(str(figure_seam.render_figure(figure, directory / stem, tight=False).name))

    profile = ((results.get("lag_readouts") or {}).get("lag_profile") or {}).get("predictive") or {}
    return {
        # The band margins are read over every scored segment; only the predictive single-lag
        # profile is capped, and its own block says on how many segments.
        "n_samples": scored_sample_count(per_sample, "mc_nll_full_block"),
        "composition": {"n_lags": int(len(rows))},
        "plan": {
            "capped": False,
            "lag_profile_cap": (eval_config.get("caps") or {}).get("lag_profile"),
            "predictive_profile_status": profile.get("status"),
        },
        "files": written,
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "BAND_FIGURE",
    "LAG_PROFILE_FILENAME",
    "PROFILE_FIGURE",
    "lag_profile_rows",
    "run_lag_suppression_analysis",
    "write_rows",
]
