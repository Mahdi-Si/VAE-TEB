r"""Per-sample diagnostic pages of this cell: the family's selections, this architecture's page.

The family's ``samples`` analysis draws the lag-attentive page through the lag-attentive task's
seams; this one draws the same selections -- a shard-stratified draw, a class-balanced draw, and
the two tails of each headline metric -- through this cell's own page,
:func:`~teb_vae.lag_slot_transformer_cfs.sample_page.build_residual_page`, whose latent rows live
on the anchor axis and whose lag rows are the per-lag proposals rather than an attention. What is
this cell's own is therefore the forward and the drawing; the selection rules, the manifest, the
filename pattern and the page seams are the family's and are imported rather than restated, so a
directory of pages under this cell is read exactly as one under a lag-attentive cell.

Every selected segment gets **one** page, the full one. The family's reduced page exists because
its full page is fifteen rows deep and leads with the raw context; this cell's page is already
the shorter one and leads with its forecast and lag rows, so a second variant would be a second
file of the same rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses.samples import (
    ANALYSIS_DIRNAME,
    CLASS_DIRNAME,
    CLASS_DRAW_SEED_OFFSET,
    DEFAULT_PAGES_PER_CLASS,
    DEFAULT_STRATIFIED_PAGES,
    EXTREME_METRICS,
    EXTREME_PAGES_PER_TAIL,
    FULL_VARIANT,
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    PAGE_SEAMS,
    STRATIFIED_DIRNAME,
    cohort_label,
    extreme_rows,
    input_stream_rows,
    page_filename,
    page_identity,
    page_scalars,
    page_seams,
    per_class_rows,
    raw_trace_normalization,
    stratified_rows,
)
from teb_vae.lag_attn_cfs.eval.dataset_rows import (
    check_identity,
    dataset_index_map,
    resolve_rows,
    subset_loader,
)
from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, batch_field, model_inputs
from teb_vae.lag_slot_transformer_cfs.sample_page import (
    build_residual_page,
    residual_lag_panels,
)

#: The one variant this cell renders per selected segment; see the module docstring.
PAGE_VARIANTS: Tuple[str, ...] = (FULL_VARIANT,)


@torch.no_grad()
def render_pages(
    task: Any,
    loader: Any,
    rows: pd.DataFrame,
    directory: Path,
    *,
    normalization: Optional[Dict[str, Any]],
    seams: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[int]]:
    """Render one page per resolved row, recording any that fail rather than losing the rest.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, for its dataset and collation.
        rows: Resolved rows carrying ``dataset_index``, ascending.
        directory: Where the pages go; created if absent.
        normalization: The loader's statistics, so the raw-context row renders in physical units.
        seams: The task's page seams, from the family's ``page_seams``.

    Returns:
        ``(written, failures, input rows drawn)``. Each written entry names its variant, its file
        and the segment's identity; a failure carries its dataset index and the error, so a page
        that could not be drawn is a recorded absence rather than a gap nobody notices. The third
        element is how many gated-input rows the seam produced, ``None`` when no page rendered.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    n_input_rows: Optional[int] = None
    if not len(rows):
        return written, failures, n_input_rows

    model = task.orig_model
    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    pages = subset_loader(loader, list(rows["dataset_index"]))
    for position, batch in enumerate(pages):
        row = rows.iloc[position]
        index = int(row["dataset_index"])
        try:
            check_identity(batch, row)
            moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            y_st, y_ph, u_stream, target_features, _weight = model_inputs(task, moved)
            # With the proposals retained: the lag rows are drawn from them, and they are the
            # largest tensor this architecture holds, which is why they are off by default.
            outs = model(
                y_st, y_ph, u_stream,
                anchor_phase=anchor_phase, anchor_stride=anchor_stride, return_proposals=True,
            )
            panels = input_stream_rows(
                model, seams["input_stream_panels"], (y_st, y_ph, u_stream), 0
            )
        except Exception as error:  # noqa: BLE001 - one segment is not worth the rest of them
            logger.warning(f"{ANALYSIS_DIRNAME}: forward for dataset index {index} failed: {error}")
            failures.append({"dataset_index": index, "guid": str(row["guid"]),
                             "variant": "forward", "error": f"{type(error).__name__}: {error}"})
            continue
        n_input_rows = len(panels)

        name = page_filename(index, row["guid"], row["epoch"])
        try:
            figure = build_residual_page(
                outs=outs,
                target_features=target_features,
                geometry=model.geometry,
                sample_index=0,
                epoch=(
                    int(round(float(row["epoch"])))
                    if np.isfinite(float(row["epoch"])) else 0
                ),
                guid=str(row["guid"]),
                # The subgroup beside the GUID, so a page lifted out of its directory still says
                # which cohort the recording came from.
                cohort=cohort_label(row),
                beta=float(task.hparams.get("kld_beta", 1.0)),
                scalars=page_scalars(row),
                # Read off the batch rather than from ``model_inputs``, which returns only what
                # the net is fed: the raw source trace is never one of the model's inputs.
                up_raw=batch_field(moved, "up"),
                normalization_stats=normalization or None,
                delay_steps=int(getattr(model, "source_delay_steps", 0) or 0),
                forecast_rows=seams["forecast_rows"],
                batch=moved,
                input_streams=panels,
                forecast_extra_rows=seams["forecast_extra_rows"],
                lag_panels=residual_lag_panels(model, outs, sample_index=0),
            )
            page = figures.render_figure(figure, directory / name)
            written.append({"variant": FULL_VARIANT, "file": page.name, **page_identity(row)})
        except Exception as error:  # noqa: BLE001 - one page is not worth the rest of them
            logger.warning(f"{ANALYSIS_DIRNAME}: page for dataset index {index} failed: {error}")
            failures.append({"dataset_index": index, "guid": str(row["guid"]),
                             "variant": FULL_VARIANT, "error": f"{type(error).__name__}: {error}"})
    return written, failures, n_input_rows


def run_samples_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render the stratified and class-balanced draws, plus the extremes of each headline metric.

    Args:
        context: The analysis context, read for the per-sample table and -- with the traces and
            the attributions -- for the task and the loader a page has to be re-rendered from.
        eval_config: The validated block, for ``caps.pages``, ``caps.pages_per_class`` and the
            draw's seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the manifest of what was rendered, what failed and by which
        index. A pass with no model records a skip.
    """
    del probe
    collection = context.collection
    per_sample = getattr(collection, "per_sample", None)
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None or per_sample is None or per_sample.empty:
        reason = (
            "a diagnostic page is the whole forward output of one segment, so it is rendered "
            "rather than read off a table; this pass built no model and no loader, which is what "
            "an offline re-run against a finished directory is"
        )
        logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"n_samples": None, "n_files": None, "composition": {},
                "plan": {"capped": True}, "skipped": True, "reason": reason, "files": []}

    seed = int(eval_config.get("seed", 0))
    caps = eval_config.get("caps") or {}
    cap = int(caps.get("pages") or DEFAULT_STRATIFIED_PAGES)
    per_class = int(caps.get("pages_per_class") or DEFAULT_PAGES_PER_CLASS)
    seams = page_seams(task)
    normalization = raw_trace_normalization(loader)
    index_map = dataset_index_map(loader)

    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    written = 0
    n_input_rows: Optional[int] = None

    def _render(rows: pd.DataFrame, selection: str) -> int:
        """Render one selection into its own directory and fold its outcome into the manifest."""
        nonlocal n_input_rows
        records, failed, observed = render_pages(
            task, loader, rows, directory / selection, normalization=normalization, seams=seams,
        )
        manifest.extend({"selection": selection, **record} for record in records)
        failures.extend({"selection": selection, **entry} for entry in failed)
        if observed is not None:
            n_input_rows = observed
        return len(records)

    drawn_rows = stratified_rows(per_sample, cap=cap, seed=seed)
    drawn = resolve_rows(drawn_rows, index_map)
    n_unlocatable = int(len(drawn_rows) - len(drawn))
    written += _render(drawn, STRATIFIED_DIRNAME)

    class_rows = per_class_rows(per_sample, per_class=per_class, seed=seed + CLASS_DRAW_SEED_OFFSET)
    by_class = resolve_rows(class_rows, index_map)
    n_unlocatable += int(len(class_rows) - len(by_class))
    written += _render(by_class, CLASS_DIRNAME)

    missing: List[str] = []
    for stem, column in EXTREME_METRICS:
        tails = extreme_rows(per_sample, column, per_tail=EXTREME_PAGES_PER_TAIL)
        if not len(tails["low"]) and not len(tails["high"]):
            missing.append(column)
            continue
        for side, frame in tails.items():
            rows = resolve_rows(frame, index_map)
            n_unlocatable += int(len(frame) - len(rows))
            written += _render(rows, f"{stem}_{side}")

    pd.DataFrame(manifest, columns=list(MANIFEST_COLUMNS)).to_csv(
        directory / MANIFEST_FILENAME, index=False
    )
    logger.info(
        f"{ANALYSIS_DIRNAME}: rendered {written} page(s), {len(failures)} failed, "
        f"{len(index_map)} dataset row(s) locatable"
    )
    return {
        "n_samples": int(written),
        "n_files": int(len(manifest)),
        "composition": {
            "n_stratified": int(len(drawn)),
            "n_shards_reached": int(drawn["source_file_basename"].nunique())
            if "source_file_basename" in drawn.columns and len(drawn) else 0,
            "n_by_class": int(len(by_class)),
            "n_classes_reached": int(by_class[labels.CLASS_COLUMN].nunique())
            if labels.CLASS_COLUMN in by_class.columns and len(by_class) else 0,
        },
        "plan": {
            "capped": True, "cap": int(cap), "seed": seed,
            "pages_per_class": int(per_class),
            "page_variants": list(PAGE_VARIANTS),
            "extreme_pages_per_tail": int(EXTREME_PAGES_PER_TAIL),
            "anchor_phase": DENSE_ANCHOR_GEOMETRY[0],
            "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
            # Which of the task's seams resolved, and how many gated-input rows the page drew:
            # a seam the task stopped declaring costs rows on every page, and nothing in a
            # rendered PDF says so.
            "page_seams": {
                name: bool(
                    seams[name] if name != "forecast_extra_rows" else len(seams[name])
                )
                for name in PAGE_SEAMS
            },
            "n_input_rows": n_input_rows,
        },
        "failures": failures,
        "missing_metrics": missing,
        "n_unlocatable_rows": n_unlocatable if index_map else None,
        "files": [MANIFEST_FILENAME],
    }


__all__ = ["PAGE_VARIANTS", "render_pages", "run_samples_analysis"]
