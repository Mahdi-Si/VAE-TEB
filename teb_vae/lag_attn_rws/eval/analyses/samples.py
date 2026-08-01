r"""Per-recording diagnostic pages, and the triage that picks which recordings to look at.

Every other analysis reduces the split to a distribution. This one does the opposite: it renders
individual segments, in full, so that a number nobody believes can be looked at. The page is
:func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` -- the same seven rows the
training callback writes every validation epoch, drawn from the same builder rather than from a
second one that could disagree with it.

Two draws, and they answer different questions:

* **A stratified sample.** Seeded, over the whole index space, quota-allocated per shard with a
  floor of one, so a cap at or above the shard count reaches every shard. A prefix would not: the
  loader is unshuffled over eight concatenated per-subgroup files, so the first $n$ samples are
  one subgroup and one clinical class -- the predecessor's documented "only 1 class found".
* **The extremes.** ``per_sample.csv`` sorted by each headline metric, head and tail. This is what
  turns an outlier in a distribution into a recording somebody can inspect, and it can only be
  done after the pass: the rows are chosen by a table that did not exist while the pass ran.

**This is the one analysis that touches the model**, and it is worth being explicit about why the
alternative does not work. A page needs the entire forward output of one segment -- eleven tensors,
about three megabytes -- so retaining pages for the extremes would mean retaining them for
*everything*, since which segments are extreme is not known until the table exists. So the pages
are re-rendered from a fresh, strictly sequential loader over a ``Subset`` of the evaluation
dataset, and the identity of every batch is **checked against the row it was selected from**
rather than trusted: the collection pass runs under a seeded shuffle, so a row's position in the
table is not its position in the dataset, and an index mapping that is off by one produces pages
that are perfectly plausible and about the wrong recordings.

A pass with no checkpoint has no model to render with, and records a skip.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Subset

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import subsample_indices
from teb_vae.lag_attn_rws.eval.metrics import batch_field, model_inputs
from teb_vae.lag_attn_rws.sample_page import build_diagnostic_figure

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "samples"

#: Where the stratified pages go, relative to that directory. The extremes go into
#: ``<metric>_low/`` and ``<metric>_high/`` beside it.
STRATIFIED_DIRNAME = "stratified"

#: The manifest of what was rendered and what failed.
MANIFEST_FILENAME = "sample_pages.csv"

#: How many pages the stratified draw renders when ``eval_config.caps.pages`` says nothing, and
#: how many rows each extreme takes. Constants rather than settings: a page is a picture, and no
#: number in any table depends on how many were drawn.
#:
#: Ten per selection, so every directory under ``samples/`` holds the same number and a reader
#: comparing two of them is comparing like with like. ``EXTREME_PAGES_PER_TAIL`` is an **upper**
#: bound rather than a promise -- see :func:`extreme_rows`, which lowers it rather than let the two
#: tails of one metric overlap.
DEFAULT_STRATIFIED_PAGES = 10
EXTREME_PAGES_PER_TAIL = 10

#: The metrics the extremes are taken on, as ``(directory stem, column, ascending is "low")``.
#: One per axis a page could be worth opening for: the coupling readout, the forecast score and
#: the KL. A metric absent from the table is skipped and recorded, never guessed at.
EXTREME_METRICS: Tuple[Tuple[str, str], ...] = (
    ("pred_gap", "mc_pred_gap"),
    ("nll_full_block", "nll_full_block"),
    ("source_conditioned_kl_raw", "source_conditioned_kl_raw"),
)

#: What a rendered page is called. The pattern is asserted by test, because these filenames are
#: the only index a reader has: a GUID carrying a path separator, a space or a non-ASCII character
#: must not be able to write outside the directory or produce a name a shell cannot address.
FILENAME_PATTERN = re.compile(r"sample\d{4}_[A-Za-z0-9_-]{1,32}_epoch(-?\d+|na)\.pdf")

#: Characters kept from a GUID; everything else becomes ``-``.
_SAFE_GUID = re.compile(r"[^A-Za-z0-9_-]")


def sanitise_guid(guid: Any) -> str:
    """Return a GUID reduced to filename-safe characters, truncated to 32.

    Args:
        guid: The recording identifier, or anything printable.

    Returns:
        The sanitised stem. Never empty -- an unnamed recording becomes ``na``, because an empty
        component would collapse two underscores and break the pattern the manifest is read by.
    """
    text = _SAFE_GUID.sub("-", str(guid))[:32]
    return text or "na"


def epoch_stamp(epoch: Any) -> Optional[int]:
    """Return an ``epoch`` as a whole number of seconds, or ``None`` when it is not one.

    Args:
        epoch: The segment's ``epoch``, which is NaN for a segment that carries none and may be
            absent entirely from an older table.

    Returns:
        The rounded value, or ``None``.
    """
    try:
        value = float(epoch)
    except (TypeError, ValueError):
        return None
    return int(round(value)) if np.isfinite(value) else None


def page_filename(index: int, guid: Any, epoch: Any) -> str:
    """Return the filename one page is written as.

    Args:
        index: The sample's index in the evaluation dataset.
        guid: Its recording identifier.
        epoch: Its ``epoch``, or anything non-finite for a segment that carries none.

    Returns:
        ``sample<index>_<guid>_epoch<epoch|na>.pdf``.
    """
    stamp = epoch_stamp(epoch)
    return (
        f"sample{int(index):04d}_{sanitise_guid(guid)}_"
        f"epoch{'na' if stamp is None else stamp}.pdf"
    )


# =============================================================================
# Mapping a table row back to the dataset it came from
# =============================================================================
def dataset_index_map(loader: Any) -> Dict[Tuple[str, Optional[int]], int]:
    """Return ``{(guid, rounded epoch): dataset index}`` for the evaluation dataset.

    Built from the dataset's own index listing rather than from the pass's row order. The
    collection pass runs under a seeded shuffle, so a row's ``sample_index`` is its position in
    *that* pass and not in the dataset; and batches the derangement could not control are skipped
    entirely, so even an unshuffled pass would drift. The identity a row carries is the only thing
    that survives both.

    Args:
        loader: The evaluation dataloader.

    Returns:
        The mapping, empty when the dataset cannot list its own recordings -- which a caller reads
        as "no page can be located", not as "no page is needed".
    """
    dataset = getattr(loader, "dataset", None)
    lister = getattr(dataset, "get_the_lists", None)
    if not callable(lister):
        return {}
    guids, epochs, _targets = lister()
    return {
        (str(guid), epoch_stamp(epoch)): index
        for index, (guid, epoch) in enumerate(zip(guids, epochs))
        if epoch_stamp(epoch) is not None
    }


def resolve_rows(
    rows: pd.DataFrame, index_map: Dict[Tuple[str, Optional[int]], int]
) -> pd.DataFrame:
    """Attach each row's dataset index, dropping and counting the rows that do not resolve.

    Args:
        rows: Rows of the per-sample table.
        index_map: From :func:`dataset_index_map`.

    Returns:
        The resolvable rows with a ``dataset_index`` column, in dataset order -- which is the
        order a sequential loader over a ``Subset`` visits them in, and is what makes the identity
        check below a check rather than a coincidence.
    """
    if not len(rows) or not index_map:
        return rows.head(0).assign(dataset_index=pd.Series(dtype=np.int64))
    resolved = [
        index_map.get((str(row["guid"]), epoch_stamp(row["epoch"])))
        for _, row in rows.iterrows()
    ]
    frame = rows.copy()
    frame["dataset_index"] = pd.Series(resolved, index=frame.index, dtype="Int64")
    frame = frame[frame["dataset_index"].notna()].copy()
    frame["dataset_index"] = frame["dataset_index"].astype(np.int64)
    return frame.sort_values("dataset_index").reset_index(drop=True)


def page_loader(loader: Any, indices: Sequence[int]) -> DataLoader:
    """Build a strictly sequential single-sample loader over the chosen dataset rows.

    Args:
        loader: The evaluation dataloader, read for its dataset and its collation.
        indices: Dataset indices, ascending.

    Returns:
        A ``DataLoader`` over a ``Subset``, one sample per batch, no sampler and no shuffle. One
        sample per batch because a page is one sample and a partly-rendered batch would be paid
        for in full.

    Raises:
        ValueError: If the indices are not ascending. The identity check downstream assumes the
            loader visits them in the order they were resolved in, and a caller reordering them is
            the one way that assumption breaks silently.
    """
    order = [int(value) for value in indices]
    if any(later <= earlier for earlier, later in zip(order, order[1:])):
        raise ValueError(
            f"page indices must be strictly ascending, got {order}. A Subset is visited in the "
            f"order it was built, so an unordered index list pairs each rendered page with "
            f"another row's guid and epoch and every page is plausible."
        )
    return DataLoader(
        Subset(loader.dataset, order),
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=loader.collate_fn,
    )


def check_identity(batch: Any, row: Any) -> None:
    """Raise unless the batch is the segment the row says it is.

    Args:
        batch: A one-sample batch from :func:`page_loader`.
        row: The per-sample row it was selected from.

    Raises:
        ValueError: On any disagreement in ``guid`` or ``epoch``. Asserted rather than assumed:
            an off-by-one in the index mapping renders a complete, plausible page of the wrong
            recording, and nothing else in the run would notice.
    """
    guid = batch.guid if not isinstance(batch, dict) else batch.get("guid")
    epoch = batch.epoch if not isinstance(batch, dict) else batch.get("epoch")
    found_guid = str(guid[0]) if isinstance(guid, (list, tuple)) else str(guid)
    found_epoch = float(np.asarray(epoch).reshape(-1)[0]) if epoch is not None else float("nan")
    wanted_guid, wanted_epoch = str(row["guid"]), float(row["epoch"])
    same_epoch = (
        abs(found_epoch - wanted_epoch) < 1.0
        or (not np.isfinite(found_epoch) and not np.isfinite(wanted_epoch))
    )
    if found_guid != wanted_guid or not same_epoch:
        raise ValueError(
            f"the dataset row rendered as a page is not the row it was selected from: the loader "
            f"yielded guid={found_guid!r} epoch={found_epoch} where the table row says "
            f"guid={wanted_guid!r} epoch={wanted_epoch}. The index mapping is wrong, and every "
            f"page it produced is a plausible picture of the wrong recording."
        )


# =============================================================================
# Rendering
# =============================================================================
@torch.no_grad()
def render_pages(
    task: Any,
    loader: Any,
    rows: pd.DataFrame,
    directory: Path,
    *,
    delay_steps: int,
    normalization: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Render one page per row, recording any that fail rather than losing the rest.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, for its dataset and collation.
        rows: Resolved rows carrying ``dataset_index``, ascending.
        directory: Where the pages go; created if absent.
        delay_steps: The model's causal input delay, for the compensated lag axes.
        normalization: The loader's FHR statistics, so the page renders in bpm.

    Returns:
        ``(written filenames, failures)``. A failure carries its dataset index and the error, so a
        page that could not be drawn is a recorded absence rather than a gap in the directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    failures: List[Dict[str, Any]] = []
    if not len(rows):
        return written, failures

    model = task.orig_model
    pages = page_loader(loader, list(rows["dataset_index"]))
    for position, batch in enumerate(pages):
        row = rows.iloc[position]
        index = int(row["dataset_index"])
        try:
            check_identity(batch, row)
            moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            y_st, y_ph, u_stream, fhr_raw, _weight = model_inputs(task, moved)
            outs = model(y_st, y_ph, u_stream)
            figure = build_diagnostic_figure(
                outs=outs,
                kld_per_dim=model.kld_tensor(
                    mu_prior=outs["mu_prior"], logvar_prior=outs["logvar_prior"],
                    mu_post=outs["mu_post"], logvar_post=outs["logvar_post"],
                ),
                fhr_raw=fhr_raw,
                geometry=model.geometry,
                sample_index=0,
                epoch=int(round(float(row["epoch"]))) if np.isfinite(float(row["epoch"])) else 0,
                guid=str(row["guid"]),
                beta=float(task.hparams.get("kld_beta", 1.0)),
                scalars=_page_scalars(row),
                # Read off the batch rather than from `model_inputs`, which returns only what the
                # net is fed: the raw source trace is never one of the model's inputs. `None` for
                # a batch that does not carry it, which the page renders as an FHR-only first row.
                up_raw=batch_field(moved, "up"),
                normalization_stats=normalization or None,
                delay_steps=int(delay_steps),
            )
            name = page_filename(index, row["guid"], row["epoch"])
            figures.render_to_pdf(figure, directory / name)
            written.append(name)
        except Exception as error:  # noqa: BLE001 - one page is not worth the rest of them
            logger.warning(f"{ANALYSIS_DIRNAME}: page for dataset index {index} failed: {error}")
            failures.append({"dataset_index": index, "guid": str(row["guid"]),
                             "error": f"{type(error).__name__}: {error}"})
    return written, failures


def _page_scalars(row: Any) -> Dict[str, float]:
    """Return the readouts the page's title carries, from the row rather than from a re-scoring."""
    names = ("nll_base_block", "nll_full_block", "pred_gap", "source_conditioned_kl_raw")
    return {
        name: float(row[name])
        for name in names
        if name in row.index and np.isfinite(float(row[name]))
    }


# =============================================================================
# Choosing which rows to render
# =============================================================================
def stratified_rows(per_sample: pd.DataFrame, *, cap: int, seed: int) -> pd.DataFrame:
    """Draw a seeded, shard-stratified sample of rows over the whole table.

    Args:
        per_sample: The per-sample table.
        cap: How many rows to draw.
        seed: The draw's seed.

    Returns:
        The drawn rows. Stratified on ``source_file_basename`` with a floor of one per shard, so a
        cap at or above the shard count reaches every shard rather than merely being likely to.
    """
    if per_sample.empty:
        return per_sample
    groups = (
        list(per_sample["source_file_basename"])
        if "source_file_basename" in per_sample.columns else None
    )
    drawn = subsample_indices(len(per_sample), int(cap), int(seed), groups=groups)
    if drawn is None:
        return per_sample
    return per_sample.iloc[[int(value) for value in drawn.tolist()]]


def extreme_rows(per_sample: pd.DataFrame, column: str, *, per_tail: int) -> Dict[str, pd.DataFrame]:
    """Return the rows carrying the smallest and largest finite values of one column.

    **The two tails are disjoint**, and on a small split that costs pages rather than correctness:
    ``per_tail`` is lowered to half the finite rows when there are not enough to fill both. Taking
    the head and the tail of fewer than $2 \\times$ ``per_tail`` rows would put the same segment in
    ``<metric>_low/`` *and* ``<metric>_high/``, where it reads as simultaneously the best and the
    worst case the model produced -- and the segments that double up are the ones nearest the
    median, which are not extreme in either direction.

    Args:
        per_sample: The per-sample table.
        column: The metric to sort on.
        per_tail: Rows per tail, as an upper bound.

    Returns:
        ``{'low': frame, 'high': frame}``, disjoint, both empty when the column is absent or
        carries no finite value.
    """
    if per_sample.empty or column not in per_sample.columns:
        return {"low": per_sample.head(0), "high": per_sample.head(0)}
    finite = per_sample[per_sample[column].notna()].sort_values(column)
    per_tail = min(int(per_tail), len(finite) // 2)
    if per_tail < 1:
        # One finite value is not two extremes. Reporting it as both would be the same claim the
        # disjointness rule above exists to prevent, made on the thinnest possible evidence.
        return {"low": finite.head(0), "high": finite.head(0)}
    return {"low": finite.head(per_tail), "high": finite.tail(per_tail)}


# =============================================================================
# The registry entry point
# =============================================================================
def run_samples_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render a stratified draw of diagnostic pages, plus the extremes of each headline metric.

    Args:
        context: The analysis context, read for the per-sample table and -- uniquely among the
            analyses -- for the task and the loader a page has to be re-rendered from.
        eval_config: The validated block, for ``caps.pages`` and the draw's seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the manifest of what was rendered, what failed and by which
        index. A pass with no model records a skip.
    """
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
        return {"n_samples": None, "composition": {}, "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    seed = int(eval_config.get("seed", 0))
    cap = int((eval_config.get("caps") or {}).get("pages") or DEFAULT_STRATIFIED_PAGES)
    record = dict(getattr(collection, "record", None) or {})
    normalization = dict(record.get("normalization") or {})
    delay_steps = int(((collection.results or {}).get("lag") or {}).get("delay_steps") or 0)
    index_map = dataset_index_map(loader)

    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    written = 0

    # Counted where the drop happens, over every frame that is actually resolved. A table-wide
    # subtraction cannot express this: the two row counts describe different populations -- scored
    # segments against locatable dataset rows -- so their difference is negative whenever the
    # collection pass skipped a batch, and zero whenever a genuine drop happens to offset one.
    drawn_rows = stratified_rows(per_sample, cap=cap, seed=seed)
    drawn = resolve_rows(drawn_rows, index_map)
    n_unlocatable = int(len(drawn_rows) - len(drawn))
    names, failed = render_pages(
        task, loader, drawn, directory / STRATIFIED_DIRNAME,
        delay_steps=delay_steps, normalization=normalization,
    )
    manifest.extend({"selection": STRATIFIED_DIRNAME, "file": name} for name in names)
    failures.extend({"selection": STRATIFIED_DIRNAME, **entry} for entry in failed)
    written += len(names)

    missing: List[str] = []
    for stem, column in EXTREME_METRICS:
        tails = extreme_rows(per_sample, column, per_tail=EXTREME_PAGES_PER_TAIL)
        if not len(tails["low"]) and not len(tails["high"]):
            missing.append(column)
            continue
        for side, frame in tails.items():
            selection = f"{stem}_{side}"
            rows = resolve_rows(frame, index_map)
            n_unlocatable += int(len(frame) - len(rows))
            names, failed = render_pages(
                task, loader, rows, directory / selection,
                delay_steps=delay_steps, normalization=normalization,
            )
            manifest.extend({"selection": selection, "file": name} for name in names)
            failures.extend({"selection": selection, **entry} for entry in failed)
            written += len(names)

    pd.DataFrame(manifest, columns=["selection", "file"]).to_csv(
        directory / MANIFEST_FILENAME, index=False
    )
    logger.info(
        f"{ANALYSIS_DIRNAME}: rendered {written} page(s), {len(failures)} failed, "
        f"{len(index_map)} dataset row(s) locatable"
    )
    return {
        "n_samples": int(written),
        "composition": {
            "n_stratified": int(len(drawn)),
            "n_shards_reached": int(drawn["source_file_basename"].nunique())
            if "source_file_basename" in drawn.columns and len(drawn) else 0,
        },
        "plan": {"capped": True, "cap": int(cap), "seed": seed,
                 "extreme_pages_per_tail": int(EXTREME_PAGES_PER_TAIL)},
        # By index, so a failure is recoverable rather than merely counted.
        "failures": failures,
        "missing_metrics": missing,
        # Rows the pages wanted and the dataset could not place, summed over the stratified draw
        # and every extreme tail. ``None`` when there is no index to locate against at all, which
        # is a different statement from "nothing failed to resolve".
        "n_unlocatable_rows": n_unlocatable if index_map else None,
        "files": [MANIFEST_FILENAME],
    }
