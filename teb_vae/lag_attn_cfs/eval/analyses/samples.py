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

**Every selected segment is drawn twice**, from one forward pass -- see :data:`PAGE_VARIANTS`.
The full page is the fifteen-row one above. The reduced page beside it, named with a ``_compact``
tail, keeps :data:`~teb_vae.lag_attn_cfs.sample_page.COMPACT_PAGE_ROWS`: the raw context, the
target block as the encoder receives it, the latent state, $K_t$, and the lag attention on a
logarithmic colour scale. It answers what this recording's latent and attention did, which is a
different question from what the model predicted, and eight rows of forecast between the two is
what makes the full page slow to read for it.

Three draws, and they answer different questions:

* **A stratified sample.** Seeded, over the whole index space, quota-allocated per shard with a
  floor of one, so a cap at or above the shard count reaches every shard. A prefix would not: the
  loader is unshuffled over eight concatenated per-subgroup files, so the first $n$ samples are
  one subgroup and one clinical class. The quota follows shard size, so what this draw renders is
  what the split mostly **contains**.
* **A class-balanced sample.** The same number of segments from every clinical class, which the
  stratified draw cannot give: its quota is proportional, and on the shipped cohort that leaves
  the two rare classes with a page or two against healthy's dozen. Balanced pages are what two
  classes can be **compared** across; the stratified ones are not.
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
from teb_vae.lag_attn_cfs.eval._reuse import labels, subsample_indices
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    batch_field,
    model_inputs,
)
from teb_vae.lag_attn_cfs.sample_page import COMPACT_PAGE_ROWS
from teb_vae.lag_attn_rws.sample_page import build_diagnostic_figure

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "samples"

#: Where the stratified pages go, relative to that directory. The extremes go into
#: ``<metric>_low/`` and ``<metric>_high/`` beside it.
STRATIFIED_DIRNAME = "stratified"

#: Where the class-balanced pages go. Separate from :data:`STRATIFIED_DIRNAME` rather than
#: replacing it, because the two draws answer different questions and neither substitutes for
#: the other: the stratified draw is **representative** -- its quota follows shard size, so what
#: it renders is what the split mostly contains -- and this one is **balanced**, so the two rare
#: clinical classes get the same number of pages as the common one and can be compared against
#: it. Read the first to see the split; read the second to see a class.
CLASS_DIRNAME = "by_class"

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

#: How many pages the class-balanced draw renders **per clinical class** when
#: ``eval_config.caps.pages_per_class`` says nothing. Per class, not in total: the whole point
#: of this draw is that the number does not depend on how many segments the class has.
DEFAULT_PAGES_PER_CLASS = 10

#: Offset added to the run's seed for the class-balanced draw. Two draws over the same table
#: from one seed would walk the same permutation stream, so the class draw's pages would be
#: correlated with the stratified draw's rather than independent of them -- and a reader
#: comparing the two directories would be looking at the same segments twice without being
#: told. Any fixed non-zero value does the job; this one is the run's own convention.
_CLASS_DRAW_SEED_OFFSET = 5

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
#:
#: The optional ``_compact`` tail is the **reduced** page of the same segment, written beside the
#: full one. A suffix rather than a directory on purpose: the two pages are one segment's, and a
#: reader who has found the segment has found both.
FILENAME_PATTERN = re.compile(
    r"sample\d{4}_[A-Za-z0-9_-]{1,32}_epoch(-?\d+|na)(_compact)?\.pdf"
)

#: The ``variant`` values the manifest records, and the tail that distinguishes the reduced
#: page's filename from the full one's. One source for both, so a directory listing and
#: ``sample_pages.csv`` cannot come to disagree about which file is which.
FULL_VARIANT = "full"
COMPACT_VARIANT = "compact"
COMPACT_SUFFIX = f"_{COMPACT_VARIANT}"

#: The two pages rendered for every selected segment, as
#: ``(variant, rows or None for the whole page, log lag attention)``. Both come off **one**
#: forward pass -- see :func:`render_pages` -- so the reduced page costs drawing and nothing
#: else, and the two cannot describe different states of the model.
#:
#: The reduced page is the one to open when the question is what the latent and the attention
#: did; the full one carries the eight forecast rows as well. Its lag attention is drawn on a
#: **logarithmic** colour scale, which the full page's is not: on a page that leads with the lag
#: panel the small weights are the content, and a linear scale flattens them under whichever lag
#: happens to dominate.
PAGE_VARIANTS: Tuple[Tuple[str, Optional[Tuple[str, ...]], bool], ...] = (
    (FULL_VARIANT, None, False),
    (COMPACT_VARIANT, COMPACT_PAGE_ROWS, True),
)

#: How many rows the reduced page has. Recorded in the analysis's ``plan`` for the same reason
#: :data:`EXPECTED_PAGE_ROWS` is: a row silently lost from a PDF is visible nowhere else.
EXPECTED_COMPACT_PAGE_ROWS = len(COMPACT_PAGE_ROWS)

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


def page_filename(index: int, guid: Any, epoch: Any, *, compact: bool = False) -> str:
    """Return the filename one page is written as.

    Args:
        index: The sample's index in the evaluation dataset.
        guid: Its recording identifier.
        epoch: Its ``epoch``, or anything non-finite for a segment that carries none.
        compact: Whether this is the reduced page. The two variants of one segment differ by
            :data:`COMPACT_SUFFIX` alone, so they sort together in a directory listing and a
            reader who has found one has found the other.

    Returns:
        ``sample<index>_<guid>_epoch<epoch|na>[_compact].pdf``.
    """
    stamp = epoch_stamp(epoch)
    return (
        f"sample{int(index):04d}_{sanitise_guid(guid)}_"
        f"epoch{'na' if stamp is None else stamp}"
        f"{COMPACT_SUFFIX if compact else ''}.pdf"
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
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Optional[int]]:
    """Render every page variant of every row, recording any that fail rather than losing the rest.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, for its dataset and collation.
        rows: Resolved rows carrying ``dataset_index``, ascending.
        directory: Where the pages go; created if absent.
        delay_steps: The model's causal input delay, for the compensated lag axes.
        normalization: The loader's statistics, so the raw-context row renders in physical units.
        seams: The task's three page seams, from :func:`page_seams`.

    Returns:
        ``(written, failures, input rows drawn)``. ``written`` carries one
        ``{'variant', 'file'}`` entry per **file** -- every one of :data:`PAGE_VARIANTS` for
        every row that rendered -- so the manifest indexes the whole directory rather than half
        of it. A failure carries its dataset index, its variant and the error, so a page that
        could not be drawn is a recorded absence rather than a gap nobody notices. The third
        element is how many gated-input rows the seam actually produced -- ``None`` when no page
        was rendered, and **measured rather than assumed**, because it is what decides whether
        the page has the rows a reader expects.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Dict[str, str]] = []
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
        # The forward is separated from the drawing so that both variants come off **one** dict. A
        # second forward would double this analysis's model time, and -- the real reason -- would
        # let two pages of one segment disagree about what the model did.
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
            kld_per_dim = model.kld_tensor(
                mu_prior=outs["mu_prior"], logvar_prior=outs["logvar_prior"],
                mu_post=outs["mu_post"], logvar_post=outs["logvar_post"],
            )
        except Exception as error:  # noqa: BLE001 - one segment is not worth the rest of them
            logger.warning(
                f"{ANALYSIS_DIRNAME}: forward for dataset index {index} failed: {error}"
            )
            failures.append({"dataset_index": index, "guid": str(row["guid"]),
                             "variant": "forward",
                             "error": f"{type(error).__name__}: {error}"})
            continue
        n_input_rows = len(panels)

        # One try per variant rather than one around both: a reduced page that fails must not turn
        # a full page that already wrote into a recorded failure. They are independent drawings of
        # the same forward, and a run is better off with one of them than with neither.
        for variant, page_rows, log_lag_attention in PAGE_VARIANTS:
            name = page_filename(
                index, row["guid"], row["epoch"], compact=variant == COMPACT_VARIANT
            )
            try:
                figure = build_diagnostic_figure(
                    outs=outs,
                    kld_per_dim=kld_per_dim,
                    # The **feature** target, which is what this cell's forecast rows read as
                    # ``rows.target``. The argument keeps the family's name; the tensor is
                    # $(B, T, c_y)$ rather than a raw trace, and the raw traces reach the page
                    # through ``batch`` and ``up_raw`` below.
                    fhr_raw=target_features,
                    geometry=model.geometry,
                    sample_index=0,
                    epoch=(
                        int(round(float(row["epoch"])))
                        if np.isfinite(float(row["epoch"])) else 0
                    ),
                    guid=str(row["guid"]),
                    beta=float(task.hparams.get("kld_beta", 1.0)),
                    scalars=_page_scalars(row),
                    # Read off the batch rather than from `model_inputs`, which returns only what
                    # the net is fed: the raw source trace is never one of the model's inputs.
                    # `None` for a batch that does not carry it, which the page renders as an
                    # FHR-only first row.
                    up_raw=batch_field(moved, "up"),
                    normalization_stats=normalization or None,
                    delay_steps=int(delay_steps),
                    forecast_rows=seams["forecast_rows"],
                    # The batch itself, for the same reason: a task whose target is not the raw
                    # signal cannot recover the raw traces from what it returned.
                    batch=moved,
                    input_streams=panels,
                    forecast_extra_rows=seams["forecast_extra_rows"],
                    rows=page_rows,
                    log_lag_attention=log_lag_attention,
                )
                figures.render_to_pdf(figure, directory / name)
                written.append({"variant": variant, "file": name})
            except Exception as error:  # noqa: BLE001 - one page is not worth the rest of them
                logger.warning(
                    f"{ANALYSIS_DIRNAME}: {variant} page for dataset index {index} failed: "
                    f"{error}"
                )
                failures.append({"dataset_index": index, "guid": str(row["guid"]),
                                 "variant": variant,
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


def per_class_rows(per_sample: pd.DataFrame, *, per_class: int, seed: int) -> pd.DataFrame:
    r"""Draw up to ``per_class`` rows uniformly at random from **every** clinical class.

    Equal $N$, not a proportional quota, and that is the whole difference from
    :func:`stratified_rows`. :func:`subsample_indices` cannot express it: its ``groups``
    argument splits a *total* cap in proportion to group size, which is the right rule for a
    draw meant to represent the split and the wrong one for a draw meant to let two classes be
    compared. On the shipped cohort the healthy class carries most of the segments, so a
    proportional draw gives ``hie`` one or two pages against healthy's dozen.

    Args:
        per_sample: The per-sample table.
        per_class: How many rows to draw from each class, as an upper bound -- a class with
            fewer segments than that contributes all of them.
        seed: The draw's seed. Offset from the run's by :data:`_CLASS_DRAW_SEED_OFFSET` by the
            caller, so this draw and the stratified one are independent rather than two reads
            of one permutation stream.

    Returns:
        The drawn rows, in table order, from every class the column names -- segments whose
        class is missing are skipped. **Empty** when the table carries no
        ``labels.CLASS_COLUMN`` at all, which is a recorded absence rather than a silent
        fallback to some other grouping: a directory named ``by_class`` whose pages were not
        drawn by class is worse than one that is empty and says why.
    """
    if per_sample.empty or labels.CLASS_COLUMN not in per_sample.columns:
        return per_sample.head(0)

    positions: Dict[Any, List[int]] = {}
    for position, value in enumerate(per_sample[labels.CLASS_COLUMN]):
        # A missing class is an absence, not a class. Drawn as one it would fill a tenth of a
        # directory called ``by_class`` with segments that have none, under a heading a reader
        # would take for a clinical group. An unrecognised *name* is kept, because that is a
        # labelling this cell does not know about rather than one that is not there.
        if pd.isna(value):
            continue
        positions.setdefault(value, []).append(position)

    # Classes are visited in the labelling's own order, with anything unrecognised after it in
    # sorted order. One generator walked in a fixed order is what makes the draw reproducible:
    # dict insertion order here is the table's row order, which the collection pass shuffled.
    known = [name for name in labels.CLASS_NAMES.values() if name in positions]
    # The keys themselves, sorted by their text -- not their text. A table can carry a class
    # value that is not one of the three names, including a missing one, and rebuilding the key
    # from `str` would then look up something the mapping does not hold.
    ordered = known + sorted((key for key in positions if key not in known), key=str)

    generator = torch.Generator().manual_seed(int(seed))
    picked: List[int] = []
    for name in ordered:
        members = positions[name]
        take = min(int(per_class), len(members))
        if take <= 0:
            continue
        member_index = torch.tensor(members, dtype=torch.long)
        chosen = member_index[torch.randperm(len(members), generator=generator)[:take]]
        picked.extend(int(value) for value in chosen.tolist())
    # Sorted back into table order, so the pages of one class are not a contiguous block and the
    # resolved dataset indices are ascending, which `page_loader` requires.
    return per_sample.iloc[sorted(picked)]


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
    """Render the stratified and class-balanced draws, plus the extremes of each headline metric.

    Every selected segment gets **two** pages -- see :data:`PAGE_VARIANTS` -- off one forward.

    Args:
        context: The analysis context, read for the per-sample table and -- with the sufficiency
            probe, uniquely -- for the task and the loader a page has to be re-rendered from.
        eval_config: The validated block, for ``caps.pages``, ``caps.pages_per_class`` and the
            draw's seed.
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
        return {"n_samples": None, "n_files": None, "composition": {},
                "plan": {"capped": True},
                "skipped": True, "reason": reason, "files": []}

    seed = int(eval_config.get("seed", 0))
    caps = eval_config.get("caps") or {}
    cap = int(caps.get("pages") or DEFAULT_STRATIFIED_PAGES)
    per_class = int(caps.get("pages_per_class") or DEFAULT_PAGES_PER_CLASS)
    delay_steps = int(((collection.results or {}).get("lag") or {}).get("delay_steps") or 0)
    seams = page_seams(task)
    normalization = raw_trace_normalization(loader)
    index_map = dataset_index_map(loader)

    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    written = 0
    n_input_rows: Optional[int] = None

    def _render(rows: pd.DataFrame, selection: str) -> int:
        """Render one selection into its own directory and fold its outcome into the manifest.

        Args:
            rows: The resolved rows to draw, carrying ``dataset_index``.
            selection: The subdirectory name, which is also the manifest's ``selection``.

        Returns:
            How many **segments** rendered, not how many files: a segment is what was selected,
            and counting its variants would make the number depend on how many pages the run
            happens to draw per segment. The file count is the manifest's length.
        """
        nonlocal n_input_rows
        records, failed, observed = render_pages(
            task, loader, rows, directory / selection,
            delay_steps=delay_steps, normalization=normalization, seams=seams,
        )
        manifest.extend({"selection": selection, **record} for record in records)
        failures.extend({"selection": selection, **entry} for entry in failed)
        if observed is not None:
            n_input_rows = observed
        return sum(1 for record in records if record["variant"] == FULL_VARIANT)

    # Counted where the drop happens, over every frame that is actually resolved. A table-wide
    # subtraction cannot express this: the two row counts describe different populations -- scored
    # segments against locatable dataset rows -- so their difference is negative whenever the
    # collection pass skipped a batch, and zero whenever a genuine drop happens to offset one.
    drawn_rows = stratified_rows(per_sample, cap=cap, seed=seed)
    drawn = resolve_rows(drawn_rows, index_map)
    n_unlocatable = int(len(drawn_rows) - len(drawn))
    written += _render(drawn, STRATIFIED_DIRNAME)

    # The balanced draw, beside the representative one rather than instead of it. Its own seed
    # offset, so the two are independent rather than two reads of one permutation stream.
    class_rows = per_class_rows(
        per_sample, per_class=per_class, seed=seed + _CLASS_DRAW_SEED_OFFSET
    )
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

    pd.DataFrame(manifest, columns=["selection", "variant", "file"]).to_csv(
        directory / MANIFEST_FILENAME, index=False
    )
    logger.info(
        f"{ANALYSIS_DIRNAME}: rendered {written} segment(s) as {len(manifest)} page(s), "
        f"{len(failures)} failed, {len(index_map)} dataset row(s) locatable"
    )
    return {
        "n_samples": int(written),
        # Files, not segments: with two variants per page the two numbers differ, and a run
        # that lost one variant everywhere would still report the full segment count.
        "n_files": int(len(manifest)),
        "composition": {
            "n_stratified": int(len(drawn)),
            "n_shards_reached": int(drawn["source_file_basename"].nunique())
            if "source_file_basename" in drawn.columns and len(drawn) else 0,
            "n_by_class": int(len(by_class)),
            # How many classes the balanced draw actually reached. Zero means the table carried
            # no clinical class at all, which is why the directory is empty -- a statement the
            # page count alone cannot make.
            "n_classes_reached": int(by_class[labels.CLASS_COLUMN].nunique())
            if labels.CLASS_COLUMN in by_class.columns and len(by_class) else 0,
        },
        "plan": {
            "capped": True, "cap": int(cap), "seed": seed,
            "pages_per_class": int(per_class),
            "page_variants": [variant for variant, _, _ in PAGE_VARIANTS],
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
            "compact_page_rows": len(COMPACT_PAGE_ROWS),
            "expected_compact_page_rows": EXPECTED_COMPACT_PAGE_ROWS,
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
