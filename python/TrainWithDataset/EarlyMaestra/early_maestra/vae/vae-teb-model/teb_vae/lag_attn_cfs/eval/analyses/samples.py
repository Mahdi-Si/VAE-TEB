r"""Per-recording diagnostic pages, and the triage that picks which recordings to look at.

Every other analysis reduces the split to a distribution. This one does the opposite: it renders
individual segments, in full, so that a number nobody believes can be looked at. The page is
:func:`~teb_vae.lag_attn_rws.sample_page.build_diagnostic_figure` -- the same layout owner the
training callback draws through -- reached through **the task's own three seams**
(``forecast_rows``, ``forecast_extra_rows``, ``input_stream_panels``) rather than through a second
builder that could disagree with the one the fit was watched through. On this cell those seams
produce a **fifteen-row** page: the raw-context row, the forecast lanes, the six causal extra rows
(truth, both branch means, the signed skill difference, $\sigma^q$, and the per-window score), two
gated-input rows, and the five latent and lag rows the layout owns.

Two draws, and they answer different questions:

* **A stratified sample.** Seeded, over the whole index space, quota-allocated per shard with a
  floor of one, so a cap at or above the shard count reaches every shard. A prefix would not: the
  loader is unshuffled over eight concatenated per-subgroup files, so the first $n$ samples are
  one subgroup and one clinical class.
* **The extremes.** ``per_sample.csv`` sorted by each headline metric, head and tail. This is what
  turns an outlier in a distribution into a recording somebody can inspect, and it can only be
  done after the pass: the rows are chosen by a table that did not exist while the pass ran.

**This is one of the two analyses that touch the model**, and it is worth being explicit about why
the alternative does not work. A page needs the entire forward output of one segment -- and here
that is four $(A_{\max}, H, C_{\mathrm{keep}})$ forecast tensors beside the latent and attention
maps -- so retaining pages for the extremes would mean retaining them for *everything*, since which
segments are extreme is not known until the table exists. So the pages are re-rendered from a
fresh, strictly sequential loader over a ``Subset`` of the evaluation dataset, and the identity of
every batch is **checked against the row it was selected from** rather than trusted: the collection
pass runs under a seeded shuffle, so a row's position in the table is not its position in the
dataset, and an index mapping that is off by one produces pages that are perfectly plausible and
about the wrong recordings.

**The forward is dense**, at :data:`~teb_vae.lag_attn_cfs.eval.metrics.DENSE_ANCHOR_GEOMETRY`, the
geometry the collection pass scored at -- so the anchors a page draws are the anchors the scalars
in its own title were computed over. The alternative is not a stylistic one: this cell's forecast
tensors are indexed by *position in the decoded set*, and a page drawn at the training tiling would
place every window at the wrong time with no shape error anywhere in it.

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

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import subsample_indices
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    batch_field,
    model_inputs,
)
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

#: The metrics the extremes are taken on, as ``(directory stem, column)``. One per axis a page
#: could be worth opening for: the coupling readout, the forecast score and the KL. A metric absent
#: from the table is skipped and recorded, never guessed at.
EXTREME_METRICS: Tuple[Tuple[str, str], ...] = (
    ("pred_gap", "mc_pred_gap"),
    ("nll_full_block", "nll_full_block"),
    ("source_conditioned_kl_raw", "source_conditioned_kl_raw"),
)

#: The three page seams this cell's task supplies, resolved off the task object by name. Named as
#: a tuple rather than read inline so that "the page is the task's" is checkable: a seam the task
#: stops declaring costs rows on every page of a run, and nothing in a rendered PDF says so.
PAGE_SEAMS: Tuple[str, ...] = ("forecast_rows", "forecast_extra_rows", "input_stream_panels")

#: How many rows the page has when every seam resolves: two the layout always reserves, the six
#: :data:`~teb_vae.lag_attn_cfs.sample_page.CAUSAL_EXTRA_ROWS`, the two gated input streams, and
#: the five latent and lag rows. Recorded in the analysis's ``plan`` rather than asserted here --
#: this module owns none of those counts -- so a page that quietly lost its input rows is visible
#: in ``summary.json`` instead of only in a PDF nobody opened.
EXPECTED_PAGE_ROWS = 15

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
# The task's page seams
# =============================================================================
def page_seams(task: Any) -> Dict[str, Any]:
    """Resolve the three seams the task supplies, by the same names the training callback uses.

    Off the task object rather than by importing this package's ``sample_page``: the seams are the
    *task's* to declare -- it is the task that decides what ``_build_raw_target`` returns and what
    the encoders are fed -- and resolving them here by name is what keeps one page definition
    behind both the fit's diagnostics and the evaluation's.

    Args:
        task: The loaded task.

    Returns:
        ``{'forecast_rows': ..., 'forecast_extra_rows': (...), 'input_stream_panels': ...}``, each
        ``None`` or empty where the task declares nothing -- which costs rows on the page and is
        reported in the analysis's ``plan`` rather than raising.
    """
    return {
        "forecast_rows": getattr(task, "forecast_rows", None),
        "forecast_extra_rows": tuple(getattr(task, "forecast_extra_rows", ()) or ()),
        "input_stream_panels": getattr(task, "input_stream_panels", None),
    }


def input_stream_rows(model: Any, builder: Any, inputs: Sequence[Any], index: int) -> Tuple[Any, ...]:
    """Build the page's gated-input rows for one sample.

    **Unlike the training callback's wrapper, a failure here is not swallowed.** That wrapper warns
    and returns no panels, which is right during a fit -- the seven rows below them do not depend
    on the input rows and a diagnostic figure is not worth a failed epoch. In an evaluation the
    same behaviour would write a silently shortened page into a results directory somebody reads
    months later, with the only trace a log line from a process that has exited. So the exception
    travels, the per-page handler records it by dataset index, and the run reports a page that
    failed rather than a page that is missing two rows.

    Args:
        model: The net, for its gates, its adapters and its resolved warm-up vectors.
        builder: The task's ``input_stream_panels`` seam, or ``None``.
        inputs: The tensors the forward consumed, ``(y_st, y_ph, u_stream)``.
        index: Which sample of the batch the page is being drawn for.

    Returns:
        The panels, empty when the task declares no builder.
    """
    if builder is None:
        return ()
    return tuple(builder(model, inputs, sample_index=index))


def raw_trace_normalization(loader: Any) -> Optional[Dict[str, Any]]:
    """Return the loader's own statistics dict, for the page's raw-context row.

    **Not** ``collection.record['normalization']``, and the difference is the point. That record is
    deliberately narrowed to the four stored feature blocks (section 5.5: nothing in this pipeline
    converts a coefficient into anything, and the record exists to say what scale a number is on).
    Row 1 of the page draws the raw FHR and UP traces, which are a different pair of fields
    entirely, and hands them to :func:`~teb_vae.lag_attn_rws.sample_page.raw_context_row` -- which
    labels its axis ``normalised`` unless it finds those fields' constants. Passing the narrowed
    record would therefore mislabel a bpm trace rather than merely lose a conversion.

    Args:
        loader: The evaluation dataloader.

    Returns:
        The statistics dict keyed by field name, or ``None`` when the dataset reports none. Absent
        statistics are not an error: the row falls back to loader units and says so on its axis.
    """
    dataset = getattr(loader, "dataset", None)
    getter = getattr(dataset, "get_normalization_stats", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception as error:  # noqa: BLE001 - a page's axis label is not worth a failed run
        logger.debug(f"{ANALYSIS_DIRNAME}: normalization statistics unavailable: {error}")
        return None


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
    seams: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], Optional[int]]:
    """Render one page per row, recording any that fail rather than losing the rest.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, for its dataset and collation.
        rows: Resolved rows carrying ``dataset_index``, ascending.
        directory: Where the pages go; created if absent.
        delay_steps: The model's causal input delay, for the compensated lag axes.
        normalization: The loader's statistics, so the raw-context row renders in physical units.
        seams: The task's three page seams, from :func:`page_seams`.

    Returns:
        ``(written filenames, failures, input rows drawn)``. A failure carries its dataset index
        and the error, so a page that could not be drawn is a recorded absence rather than a gap in
        the directory. The third element is how many gated-input rows the seam actually produced --
        ``None`` when no page was rendered, and **measured rather than assumed**, because it is
        what decides whether the page has the rows a reader expects.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    failures: List[Dict[str, Any]] = []
    n_input_rows: Optional[int] = None
    if not len(rows):
        return written, failures, n_input_rows

    model = task.orig_model
    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    pages = page_loader(loader, list(rows["dataset_index"]))
    for position, batch in enumerate(pages):
        row = rows.iloc[position]
        index = int(row["dataset_index"])
        try:
            check_identity(batch, row)
            moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            y_st, y_ph, u_stream, target_features, _weight = model_inputs(task, moved)
            outs = model(
                y_st, y_ph, u_stream,
                anchor_phase=anchor_phase, anchor_stride=anchor_stride,
            )
            panels = input_stream_rows(
                model, seams["input_stream_panels"], (y_st, y_ph, u_stream), 0
            )
            figure = build_diagnostic_figure(
                outs=outs,
                kld_per_dim=model.kld_tensor(
                    mu_prior=outs["mu_prior"], logvar_prior=outs["logvar_prior"],
                    mu_post=outs["mu_post"], logvar_post=outs["logvar_post"],
                ),
                # The **feature** target, which is what this cell's forecast rows read as
                # ``rows.target``. The argument keeps the family's name; the tensor is
                # $(B, T, c_y)$ rather than a raw trace, and the raw traces reach the page
                # through ``batch`` and ``up_raw`` below.
                fhr_raw=target_features,
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
                forecast_rows=seams["forecast_rows"],
                # The batch itself, for the same reason: a task whose target is not the raw
                # signal cannot recover the raw traces from what it returned.
                batch=moved,
                input_streams=panels,
                forecast_extra_rows=seams["forecast_extra_rows"],
            )
            name = page_filename(index, row["guid"], row["epoch"])
            figures.render_to_pdf(figure, directory / name)
            written.append(name)
            n_input_rows = len(panels)
        except Exception as error:  # noqa: BLE001 - one page is not worth the rest of them
            logger.warning(f"{ANALYSIS_DIRNAME}: page for dataset index {index} failed: {error}")
            failures.append({"dataset_index": index, "guid": str(row["guid"]),
                             "error": f"{type(error).__name__}: {error}"})
    return written, failures, n_input_rows


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
        context: The analysis context, read for the per-sample table and -- with the sufficiency
            probe, uniquely -- for the task and the loader a page has to be re-rendered from.
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
    delay_steps = int(((collection.results or {}).get("lag") or {}).get("delay_steps") or 0)
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
        names, failed, observed = render_pages(
            task, loader, rows, directory / selection,
            delay_steps=delay_steps, normalization=normalization, seams=seams,
        )
        manifest.extend({"selection": selection, "file": name} for name in names)
        failures.extend({"selection": selection, **entry} for entry in failed)
        if observed is not None:
            n_input_rows = observed
        return len(names)

    # Counted where the drop happens, over every frame that is actually resolved. A table-wide
    # subtraction cannot express this: the two row counts describe different populations -- scored
    # segments against locatable dataset rows -- so their difference is negative whenever the
    # collection pass skipped a batch, and zero whenever a genuine drop happens to offset one.
    drawn_rows = stratified_rows(per_sample, cap=cap, seed=seed)
    drawn = resolve_rows(drawn_rows, index_map)
    n_unlocatable = int(len(drawn_rows) - len(drawn))
    written += _render(drawn, STRATIFIED_DIRNAME)

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
        "plan": {
            "capped": True, "cap": int(cap), "seed": seed,
            "extreme_pages_per_tail": int(EXTREME_PAGES_PER_TAIL),
            "anchor_phase": DENSE_ANCHOR_GEOMETRY[0],
            "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
            # Which of the task's seams resolved, and how many rows the page therefore has. A seam
            # the task stopped declaring costs rows on every page of the run and nothing in a
            # rendered PDF says so, which is why the count is a number in the summary.
            "page_seams": {
                name: bool(
                    seams[name] if name != "forecast_extra_rows" else len(seams[name])
                )
                for name in PAGE_SEAMS
            },
            "page_rows": _page_row_count(seams, n_input_rows),
            "expected_page_rows": EXPECTED_PAGE_ROWS,
        },
        # By index, so a failure is recoverable rather than merely counted.
        "failures": failures,
        "missing_metrics": missing,
        # Rows the pages wanted and the dataset could not place, summed over the stratified draw
        # and every extreme tail. ``None`` when there is no index to locate against at all, which
        # is a different statement from "nothing failed to resolve".
        "n_unlocatable_rows": n_unlocatable if index_map else None,
        "files": [MANIFEST_FILENAME],
    }


def _page_row_count(seams: Dict[str, Any], n_input_rows: Optional[int]) -> Optional[int]:
    """Return how many rows the pages this run rendered actually have.

    Derived from the layout's own rule rather than from a literal: two rows the builder always
    reserves (raw context and forecast), the extra rows the forecast seam asked for, one per gated
    input row the seam **produced**, and the five latent and lag rows the builder owns.

    The input count is measured on a rendered page rather than inferred from the presence of a
    builder, because that is exactly the number that can be wrong: the shipped builder is welded to
    the production two-sided filter bank and refuses these widths, and this cell supplies its own
    replacement -- so "a builder is declared" and "two rows were drawn" are different claims, and
    only the second is worth putting in a summary.

    Args:
        seams: The resolved seams.
        n_input_rows: Input rows observed on a rendered page, or ``None`` when none was rendered.

    Returns:
        The row count, which a caller compares against :data:`EXPECTED_PAGE_ROWS`, or ``None`` when
        no page was rendered and there is therefore nothing measured to report.
    """
    if n_input_rows is None:
        return None
    return 2 + len(seams["forecast_extra_rows"]) + int(n_input_rows) + 5
