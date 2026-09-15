r"""The scored arms: every arm's score, every margin paired over recordings, and the per-recording table.

Three figures and one table, all drawn from the results block the pass assembled and never from a
tensor, so a figure and the number it illustrates cannot disagree and the whole set redraws from a
finished directory on a box with no checkpoint.

The **table** is this cell's protocol table: one row per recording under this cell's own column
names, which is what the acceptance pass reads several runs of. It is written here rather than by
the pass because it is a reading of the results rather than a product of the forward, and because
this is the analysis a reader of the arms opens. The ``scored_split`` record beside it says which
recordings the pass scored and out of which files, which is the block the reserved-partition
question is answered from.
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from teb_vae.lag_attn_cfs.eval import figures_seam as figure_seam
from teb_vae.lag_attn_cfs.eval.frames import scored_sample_count
from teb_vae.lag_slot_transformer_cfs.eval import figures

#: Where this analysis writes, inside the results directory.
ANALYSIS_DIRNAME = "arms"

#: The per-recording table, and its path relative to the results directory -- the form the
#: acceptance pass reads it by, restated there as a literal so that module stays stdlib-only.
PER_RECORDING_FILENAME = "per_recording.csv"
PER_RECORDING_TABLE = f"{ANALYSIS_DIRNAME}/{PER_RECORDING_FILENAME}"

#: The figures, by stem; the format is the run's.
HEADLINE_FIGURE = "headline_arms"
GAP_FIGURE = "pred_gap_recordings"
CALIBRATION_FIGURE = "mixture_calibration"


def write_per_recording_table(path: Any, table: Mapping[str, Mapping[str, float]]) -> None:
    """Write one row per recording, in a stable column and row order.

    Sorted by recording rather than by the order the loader happened to hand the batches out, so
    two runs of the same split produce two files a reader can diff line by line.

    Args:
        path: The file to write.
        table: ``{recording: {column: value}}``.
    """
    columns = sorted({name for row in table.values() for name in row})
    with open(str(path), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["guid", *columns])
        for guid in sorted(table):
            row = table[guid]
            writer.writerow([guid, *(row.get(name, "") for name in columns)])


def scored_split_record(
    config: Mapping[str, Any], recordings: Sequence[str]
) -> Dict[str, Any]:
    """Which recordings this pass scored, and which files they came out of.

    **This is the block the reserved-partition question is answered from.** A confirmation run has
    to be shown to have scored a partition that no run used to choose an architecture, a
    hyperparameter or a threshold ever touched, and neither a checkpoint nor a metric can say that:
    the only evidence is which files were opened and which recordings came back. Both are recorded
    here so the question is settled from the artifacts rather than from a memory of which shards
    were pointed at in which week.

    The label is the common parent of the scored files rather than a parsed fold name. A parse
    would encode one dataset layout into a gate that has to keep working when the next build names
    its directories differently, and the common parent separates two partitions exactly as well.

    Args:
        config: The merged run configuration.
        recordings: The recordings the pass actually scored.

    Returns:
        The split block: the files, the statistics they were standardised with, the common parent,
        a digest of the scored recording set, and where the per-recording table is.
    """
    shards = [str(path) for path in config.get("dataset_config", {}).get("vae_test_datasets", [])]
    try:
        common = os.path.commonpath([os.path.dirname(os.path.abspath(path)) for path in shards])
    except ValueError:
        # Paths on different drives have no common parent. A Windows-only condition, and a
        # split assembled from two drives is a legitimate arrangement rather than an error.
        common = ""
    ordered = sorted(str(guid) for guid in recordings)
    return {
        "shards": shards,
        "stat_path": str(config.get("dataset_config", {}).get("stat_path", "")),
        "label": common,
        "n_recordings": len(ordered),
        # A cheap identity for the scored recording set: two runs whose digests agree scored the
        # same recordings, and two whose digests differ need the tables themselves to say how far
        # apart they are. It is an identity check, not a privacy measure -- the table beside it
        # carries the recordings in the clear, as every per-recording export in this repository
        # does.
        "recording_digest": hashlib.sha256(
            "\n".join(ordered).encode("utf-8")
        ).hexdigest()[:16],
        "per_recording_table": PER_RECORDING_TABLE,
    }


def run_arms_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the per-recording table and draw the three arm figures.

    Args:
        context: The analysis context, read for the results block and the merged configuration.
        eval_config: The validated block, echoed into the plan.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys, the scored-split record and the files written.
    """
    del probe
    collection = context.collection
    results = dict(getattr(collection, "results", None) or {})
    per_sample = collection.per_sample
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    table = dict(results.get("per_recording") or {})
    write_per_recording_table(directory / PER_RECORDING_FILENAME, table)

    written: List[str] = [PER_RECORDING_FILENAME]
    for stem, figure in (
        (HEADLINE_FIGURE, figures.build_headline_figure(results)),
        (GAP_FIGURE, figures.build_gap_distribution_figure(results, table)),
        (CALIBRATION_FIGURE, figures.build_calibration_figure(results)),
    ):
        written.append(str(figure_seam.render_figure(figure, directory / stem, tight=False).name))

    return {
        "n_samples": scored_sample_count(per_sample, "mc_nll_full_block"),
        "composition": {"n_recordings": int(len(table))},
        "plan": {
            "capped": False,
            "bootstrap_resamples": int(eval_config.get("bootstrap_resamples", 0)),
            "seed": int(eval_config.get("seed", 0)),
        },
        "scored_split": scored_split_record(
            dict(getattr(context, "config", None) or {}), list(table)
        ),
        "files": written,
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "CALIBRATION_FIGURE",
    "GAP_FIGURE",
    "HEADLINE_FIGURE",
    "PER_RECORDING_FILENAME",
    "PER_RECORDING_TABLE",
    "run_arms_analysis",
    "scored_split_record",
    "write_per_recording_table",
]
