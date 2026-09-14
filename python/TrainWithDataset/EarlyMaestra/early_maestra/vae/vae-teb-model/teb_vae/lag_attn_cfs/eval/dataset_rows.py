r"""Mapping a collected table row back to the dataset row it came from, and reading it again.

Two analyses re-run inference on segments the collection pass has already scored: the diagnostic
pages, which need the whole forward output of one segment, and the recording traces, which need
the whole forward output of *every* segment of a chosen recording. Both start from rows of
``per_sample.csv`` and both have the same invisible failure mode -- a forward taken over the wrong
dataset row is a complete, plausible, correctly formatted result about a recording nobody asked
for, and no downstream number moves. So the mapping lives here once, at layer 1 (it builds a
``DataLoader``), and every consumer goes through the same three steps:

1. :func:`dataset_index_map` -- ``{(guid, rounded epoch): dataset index}`` from the dataset's own
   listing, never from the pass's row order. The collection pass runs under a seeded shuffle, so
   a row's ``sample_index`` is its position in *that* pass and not in the dataset; and batches the
   derangement could not control are skipped entirely, so even an unshuffled pass would drift.
2. :func:`resolve_rows` -- attach each row's dataset index, dropping and counting the rows the
   dataset cannot place, in **ascending** index order.
3. :func:`subset_loader` -- a strictly sequential loader over exactly those indices, whose every
   batch is then checked against the rows it was built from by :func:`check_batch_identity`.

An analysis may not import another analysis, which is why these are not left on the pages module
they were written for: anything two analyses share moves one layer down.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

#: Characters kept from a GUID in a filename; everything else becomes ``-``.
_SAFE_GUID = re.compile(r"[^A-Za-z0-9_-]")


def sanitise_guid(guid: Any) -> str:
    """Return a GUID reduced to filename-safe characters, truncated to 32.

    Args:
        guid: The recording identifier, or anything printable.

    Returns:
        The sanitised stem. Never empty -- an unnamed recording becomes ``na``, because an empty
        component would collapse two underscores and break the pattern a manifest is read by.
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


def dataset_index_map(loader: Any) -> Dict[Tuple[str, Optional[int]], int]:
    """Return ``{(guid, rounded epoch): dataset index}`` for the evaluation dataset.

    Built from the dataset's own index listing rather than from the pass's row order, for the
    reason the module docstring gives: the identity a row carries is the only thing that survives
    both the seeded shuffle and a skipped batch.

    Args:
        loader: The evaluation dataloader.

    Returns:
        The mapping, empty when the dataset cannot list its own recordings -- which a caller reads
        as "no row can be located", not as "no row is needed".
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
        rows: Rows of the per-sample table, carrying ``guid`` and ``epoch``.
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


def subset_loader(loader: Any, indices: Sequence[int], *, batch_size: int = 1) -> DataLoader:
    """Build a strictly sequential loader over the chosen dataset rows.

    Args:
        loader: The evaluation dataloader, read for its dataset and its collation.
        indices: Dataset indices, strictly ascending.
        batch_size: Samples per batch. One for a page, which is one sample and would pay for a
            partly rendered batch in full; a recording's segments may travel together.

    Returns:
        A ``DataLoader`` over a ``Subset``, no sampler and no shuffle, visiting ``indices`` in the
        order given.

    Raises:
        ValueError: If the indices are not strictly ascending. The identity check downstream
            assumes the loader visits them in the order they were resolved in, and a caller
            reordering them is the one way that assumption breaks silently.
    """
    order = [int(value) for value in indices]
    if any(later <= earlier for earlier, later in zip(order, order[1:])):
        raise ValueError(
            f"subset indices must be strictly ascending, got {order}. A Subset is visited in the "
            f"order it was built, so an unordered index list pairs each forward with another "
            f"row's guid and epoch and every result is plausible."
        )
    return DataLoader(
        Subset(loader.dataset, order),
        batch_size=int(batch_size),
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=loader.collate_fn,
    )


def _batch_identity(batch: Any, position: int) -> Tuple[str, float]:
    """Read ``(guid, epoch)`` of one sample of a batch, tolerating mapping and attribute access."""
    guid = batch.guid if not isinstance(batch, dict) else batch.get("guid")
    epoch = batch.epoch if not isinstance(batch, dict) else batch.get("epoch")
    found_guid = (
        str(guid[position]) if isinstance(guid, (list, tuple)) else str(guid)
    )
    found_epoch = (
        float(np.asarray(epoch).reshape(-1)[position]) if epoch is not None else float("nan")
    )
    return found_guid, found_epoch


def _same_identity(found: Tuple[str, float], wanted: Tuple[str, float]) -> bool:
    """Whether a batch sample's identity is the row's, to the second on the epoch."""
    same_epoch = (
        abs(found[1] - wanted[1]) < 1.0
        or (not np.isfinite(found[1]) and not np.isfinite(wanted[1]))
    )
    return found[0] == wanted[0] and same_epoch


def check_identity(batch: Any, row: Any, *, position: int = 0) -> None:
    """Raise unless sample ``position`` of the batch is the segment the row says it is.

    Args:
        batch: A batch from :func:`subset_loader`.
        row: The per-sample row it was selected from, carrying ``guid`` and ``epoch``.
        position: Which sample of the batch to compare.

    Raises:
        ValueError: On any disagreement in ``guid`` or ``epoch``. Asserted rather than assumed:
            an off-by-one in the index mapping produces a complete, plausible result about the
            wrong recording, and nothing else in the run would notice.
    """
    found = _batch_identity(batch, int(position))
    wanted = (str(row["guid"]), float(row["epoch"]))
    if not _same_identity(found, wanted):
        raise ValueError(
            f"the dataset row re-read is not the row it was selected from: the loader yielded "
            f"guid={found[0]!r} epoch={found[1]} where the table row says guid={wanted[0]!r} "
            f"epoch={wanted[1]}. The index mapping is wrong, and every result it produced is a "
            f"plausible picture of the wrong recording."
        )


def check_batch_identity(batch: Any, rows: pd.DataFrame) -> None:
    """Raise unless every sample of a batch is the row it was built from, position by position.

    Args:
        batch: A batch from :func:`subset_loader`, holding ``len(rows)`` samples.
        rows: The rows the batch's indices were taken from, in the same order.

    Raises:
        ValueError: On a batch of the wrong size, or on any positional disagreement.
    """
    guid = batch.guid if not isinstance(batch, dict) else batch.get("guid")
    found_size = len(guid) if isinstance(guid, (list, tuple)) else 1
    if found_size != len(rows):
        raise ValueError(
            f"a batch of {found_size} sample(s) was built from {len(rows)} row(s); the loader and "
            f"the rows it was built from have come apart, so no position can be trusted."
        )
    for position, (_, row) in enumerate(rows.iterrows()):
        check_identity(batch, row, position=position)


__all__: List[str] = [
    "check_batch_identity",
    "check_identity",
    "dataset_index_map",
    "epoch_stamp",
    "resolve_rows",
    "sanitise_guid",
    "subset_loader",
]
