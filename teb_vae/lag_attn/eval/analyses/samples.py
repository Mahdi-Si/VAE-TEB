r"""One diagnostic page per recording, for the handful worth looking at individually.

Every other analysis reduces the split to a distribution. This one does the opposite: it keeps a
small, seeded, stratified draw of recordings and renders each as a full page, because a
distribution cannot show *why* a recording forecasts badly and a page can -- the attention
collapsing onto lag $0$, the residual concentrated in the phase-harmonic block, the latent going
quiet halfway through.

Two things it does deliberately.

**The draw is stratified, not a prefix.** The test loader is built ``shuffle=False`` over eight
concatenated per-subgroup files, so the first eight samples are one subgroup and one clinical
class -- and eight pages of the same subgroup is the least useful draw available. The cap is a
seeded subsample over the whole index space, stratified by source file, so a cap of at least the
file count reaches every shard.

**One page failing does not lose the rest.** The pages are the last thing a multi-hour run
produces, and a single recording with a degenerate field -- an all-zero weight, a raw trace of
the wrong length -- must not discard the seven pages already on disk. Each page is rendered
inside its own guard and its failure is recorded per sample rather than raised.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import figures, masks, metrics, preflight, sample_figure
from teb_vae.lag_attn.eval.collectors import CollectionPlan
from teb_vae.lag_attn.eval.runner import (
    EvalRunner,
    batch_size_of,
    get_field,
    guid_of,
    to_numpy,
)

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "samples"

#: The ``eval_config.caps`` key this analysis reads.
CAP_NAME = "samples"

#: Cap applied when the config sets none. Every page is a full-size PDF, so an uncapped run over
#: a real test split would emit thousands of them and take longer than every other analysis
#: combined -- an unset cap here means "a few", not "all of them".
DEFAULT_CAP = 8

#: Forward keys the diagnostic page reads, sliced per sample. An explicit list rather than a
#: shape heuristic -- see :func:`_render_one`.
FIGURE_OUTPUT_KEYS: tuple = (
    "mu_full",
    "z",
    "mu_prior",
    "logvar_prior",
    "mu_post",
    "logvar_post",
    "kld_per_t",
    "attn_weights",
    "te_lag_map",
)

#: Characters kept in the GUID portion of a filename. A GUID is an opaque record identifier and
#: is not guaranteed to be path-safe.
_SAFE_GUID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def sample_filename(guid: str, epoch: Optional[float], index: int) -> str:
    """Return the PDF filename for one sample.

    The GUID and the epoch are both in the name so a page can be matched back to its row in any
    emitted CSV without opening it, and the global index leads so a directory listing sorts into
    loader order.

    Args:
        guid: The record identifier.
        epoch: The recording's epoch in hours, or ``None`` when the batch carries none.
        index: The sample's global index in the loader.

    Returns:
        The filename, including the ``.pdf`` suffix.
    """
    safe = "".join(char for char in str(guid) if char in _SAFE_GUID_CHARS)[:32] or "unknown"
    stamp = "na" if epoch is None or not np.isfinite(epoch) else f"{float(epoch):.2f}"
    return f"sample{int(index):04d}_{safe}_epoch{stamp}.pdf"


def _epoch_of(batch: Any, offset: int) -> Optional[float]:
    """Return one sample's epoch as a float, or ``None`` when the batch carries no ``epoch``."""
    epoch = get_field(batch, "epoch")
    if epoch is None:
        return None
    values = np.atleast_1d(to_numpy(epoch)).ravel()
    if offset >= values.size:
        return None
    return float(values[offset])


def _per_sample_metrics(runner: EvalRunner, batch: Any, outputs: Dict[str, Any]) -> Dict[str, Any]:
    r"""Compute the per-sample scalars that accompany the pages.

    Deliberately the same quantities the pages draw -- the forecast error, the KL the latent
    carried, the lag the attention selected -- so the CSV is a legible index of the pages rather
    than a second, differently-defined metric table. The authoritative distributions live in the
    forecast, latent and attention analyses.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.
        outputs: That batch's completed forward, reused rather than recomputed.

    Returns:
        Column name to a per-sample array of length $B$.
    """
    view = runner.forecast_view(batch, outputs)
    columns: Dict[str, Any] = dict(
        metrics.forecast_metrics(view.mu_full, view.y_plus, view.mask, view.n_scattering)
    )

    seq_len = int(outputs["kld_per_t"].shape[1])
    batch_size = int(outputs["kld_per_t"].shape[0])
    mask_bt = masks.kld_mask(
        runner.model, get_field(batch, "weight"), batch_size, seq_len,
        device=outputs["kld_per_t"].device,
    )
    columns.update(metrics.kld_aggregates(metrics.kld_per_dim(outputs, runner.model), mask_bt))
    # (B, d_z) rather than (B,); it belongs in the latent analysis's per-dimension table, not in
    # a one-row-per-sample index.
    columns.pop("kld_per_dim_mean", None)

    # Mean over heads, then argmax: the same reduction the attention row of the page draws, so
    # the number in the CSV is the red line in the picture.
    alpha = outputs["attn_weights"].mean(dim=2)
    support = masks.kld_support(runner.model, seq_len, device=alpha.device) > 0
    in_support = alpha[:, support, :]
    columns["mean_argmax_lag"] = (
        in_support.argmax(dim=-1).float().mean(dim=1)
        if bool(support.any())
        else torch.full((batch_size,), float("nan"))
    )
    return columns


def _render_one(
    directory: Path,
    outputs: Dict[str, Any],
    batch: Any,
    offset: int,
    index: int,
    *,
    runner: EvalRunner,
    up_shift_secs: float,
    te_lag_label: str,
) -> str:
    """Render and save one sample's page, returning the filename written.

    Args:
        directory: The analysis directory.
        outputs: The batch's forward outputs.
        batch: The batch the forward was run on.
        offset: Position within the batch.
        index: The sample's global index in the loader.
        runner: The loaded runner, for the geometry.
        up_shift_secs: Dataset UP shift, for the lag second-axis.
        te_lag_label: Whether the TE lag map is an attribution or a diagnostic.

    Returns:
        The filename written.
    """
    y_st, y_ph = runner.build_target_streams(batch)
    seq_len = int(y_st.shape[1])
    raw_fhr, raw_up = get_field(batch, "fhr"), get_field(batch, "up")
    guid = guid_of(batch, offset)
    epoch = _epoch_of(batch, offset)

    figure = None
    try:
        figure = sample_figure.build_sample_figure(
            outputs={
                # An explicit key list, not "every tensor with a batch dimension". ``forward``
                # also returns ``warmup_mask``, which is $(T,)$ -- indexing it by a *batch*
                # offset silently yields the mask value at time step ``offset``, which is a
                # wrong number rather than an error. Naming the keys the figure reads makes
                # that unrepresentable.
                key: outputs[key][offset]
                for key in FIGURE_OUTPUT_KEYS
                if isinstance(outputs.get(key), torch.Tensor)
            },
            y_st=y_st[offset],
            y_ph=y_ph[offset],
            fhr_raw=None if raw_fhr is None else raw_fhr[offset],
            up_raw=None if raw_up is None else raw_up[offset],
            warmup=int(runner.model._warmup_steps(seq_len)),
            horizon=int(runner.model.horizon),
            guid=guid,
            epoch=epoch,
            step_seconds=metrics.STEP_SECONDS,
            up_shift_secs=up_shift_secs,
            te_lag_label=te_lag_label,
        )
        name = sample_filename(guid, epoch, index)
        # tight=False: the page sets its own GridSpec margins, and tight_layout both warns that
        # it cannot handle the twinned and colorbar axes and would undo the alignment that is
        # the whole point of the reserved colorbar column.
        figures.render_to_pdf(figure, directory / name, tight=False)
        return name
    finally:
        # The builder is inside the ``try`` because a failure *there* is exactly what the
        # caller's per-sample guard absorbs -- and a figure left open by it would never be
        # reclaimed, one per selected sample. ``render_to_pdf`` closes on success; closing an
        # already-closed figure is a no-op.
        if figure is not None:
            figures.plt.close(figure)


def run_samples_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render a diagnostic page for a capped, stratified draw of recordings.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, supplying the sample count and the per-file grouping
            the draw stratifies over.

    Returns:
        The headline summary for ``summary.json``.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    caps = eval_config.get("caps") or {}
    cap = caps.get(CAP_NAME, DEFAULT_CAP)
    seed = int(eval_config.get("seed", 0))
    max_samples = eval_config.get("max_samples")
    up_shift_secs = float(eval_config.get("up_shift_secs", 0.0))
    n_total = int((probe or {}).get("n_samples") or 0)
    groups = (probe or {}).get("source_files")

    plan = (
        CollectionPlan.build(n_total, cap, seed, groups=groups)
        if n_total
        else None
    )
    # Recorded rather than enforced: a run against a checkpoint without head_structured_latent
    # still gets its pages, with the TE row honestly labelled a diagnostic.
    te_lag_label = preflight.te_lag_map_label(runner)

    rows: List[Dict[str, Any]] = []
    written: List[str] = []
    failures: Dict[str, str] = {}
    composition: Counter = Counter()
    global_index = 0

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        batch_size = batch_size_of(batch)
        selected = [
            offset
            for offset in range(batch_size)
            if plan is None or plan.keeps(global_index + offset)
        ]
        if selected:
            outputs = runner.forward(batch)
            columns = _per_sample_metrics(runner, batch, outputs)
            for offset in selected:
                index = global_index + offset
                source = _source_of(batch, offset)
                row: Dict[str, Any] = {
                    "sample_index": index,
                    "guid": guid_of(batch, offset),
                    "source_file": source,
                    "epoch": _epoch_of(batch, offset),
                }
                for name, values in columns.items():
                    array = np.asarray(to_numpy(values)).ravel()
                    row[name] = float(array[offset]) if offset < array.size else float("nan")

                try:
                    row["figure"] = _render_one(
                        directory, outputs, batch, offset, index,
                        runner=runner, up_shift_secs=up_shift_secs,
                        te_lag_label=te_lag_label,
                    )
                    written.append(row["figure"])
                except Exception as exc:  # noqa: BLE001 - one page must not lose the others
                    row["figure"] = None
                    failures[str(index)] = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        f"[samples] sample {index} (guid {row['guid']}) failed to render: "
                        f"{type(exc).__name__}: {exc}"
                    )
                rows.append(row)
                composition[source] += 1
        global_index += batch_size

    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "per_sample.csv", index=False)

    logger.info(
        f"samples: wrote {len(written)} page(s) for {len(frame)} selected sample(s); "
        f"composition {dict(composition)}"
    )
    return {
        "n_samples": int(len(frame)),
        "n_figures": len(written),
        "composition": dict(sorted(composition.items())),
        "plan": None if plan is None else plan.describe(),
        "cap": None if cap is None else int(cap),
        "te_lag_map_label": te_lag_label,
        "failures": failures,
        "figures": [str(directory / name) for name in written],
    }


def _source_of(batch: Any, offset: int) -> str:
    """Return the shard basename a sample came from, or ``'unknown'``."""
    names = get_field(batch, "source_file_basename")
    if names is None:
        return "unknown"
    if isinstance(names, (list, tuple)):
        return str(names[offset])
    return str(names)
