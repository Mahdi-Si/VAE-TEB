r"""Loader-iteration patterns: run a per-batch computation, accumulate, return a DataFrame.

Every collector runs under ``runner.inference_mode()``, which is both ``no_grad`` and ``eval()``.
The second half is not redundant: under ``train()`` dropout is live inside the attention, so the
attention rows do not sum to $1$ and the ``te_lag_map`` identity quietly stops holding.

**A cap is a seeded subsample over the whole index space, never a prefix.** The test loader is
built ``shuffle=False`` over eight concatenated per-subgroup files, so the first $N$ samples are
file $0$ alone -- one subgroup, one clinical class. That is the predecessor's "only 1 class
found" failure arriving by a second route, and its only recorded workaround was "do not use a
cap". :class:`CollectionPlan` draws instead over the full index, stratified by source file when
the probe supplied one, and every capped collection returns the per-file composition it
*actually* drew, so a skewed draw is visible in the run's output rather than invisible.

**The heavy collectors are heavy.** Per retained sample, at the production geometry
($T = 300$, $c_y = 109$, $H_d = 15$, $M = 4$, $L = 91$, fp32):

* :func:`collect_predictions` -- the overlap-averaged forecast and its target, $2 \times (T,
  c_y)$, about $262$ KB. Averaged rather than the raw $(A, H_d, c_y)$ per-anchor tensor, which
  is $1.9$ MB per sample per field and would need gigabytes at the shipped cap of $2000$.
* :func:`collect_attention` -- $(T, M, L)$, about $436$ KB, so the shipped cap of $2000$ costs
  roughly $0.9$ GB. It is the largest retention in the pipeline and the one to lower first if a
  run is tight on host memory.

:func:`collect_metrics` retains only scalars and is not capped by default.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval.labels import batch_labels
from teb_vae.lag_attn.eval.masks import subsample_indices
from teb_vae.lag_attn.eval.runner import (
    EvalRunner,
    batch_size_of,
    get_field,
    guid_of,
    to_numpy,
)

#: Signature of a per-batch computation: given the runner and one batch, return a mapping from
#: column name to a per-sample value array of length $B$.
PerBatchFn = Callable[[EvalRunner, Any], Dict[str, Any]]


@dataclass
class CollectionPlan:
    """Which global sample indices a capped collection retains.

    Built once from the loader probe's totals, so the draw is decided before iteration starts
    and every collector in a run can be given the same plan or its own seeded one.
    """

    n_total: int
    cap: Optional[int] = None
    seed: int = 0
    retained: Optional[set] = None

    @classmethod
    def build(
        cls,
        n_total: int,
        cap: Optional[int],
        seed: int,
        *,
        groups: Optional[Sequence[Any]] = None,
    ) -> "CollectionPlan":
        """Draw the retained index set.

        Args:
            n_total: Number of samples the loader will yield.
            cap: Maximum to retain, or ``None`` for all of them.
            seed: Seed, so a rerun retains the same samples.
            groups: Per-index group key -- the source file basename -- to stratify over. Passing
                it is what upgrades "very probably covers every file" into "covers every file
                whenever the cap is at least the file count".

        Returns:
            The plan. ``retained is None`` means everything is kept.
        """
        indices = subsample_indices(int(n_total), cap, int(seed), groups=groups)
        return cls(
            n_total=int(n_total),
            cap=None if cap is None else int(cap),
            seed=int(seed),
            retained=None if indices is None else {int(value) for value in indices.tolist()},
        )

    def keeps(self, index: int) -> bool:
        """Whether the sample at global index ``index`` is retained."""
        return self.retained is None or int(index) in self.retained

    def describe(self) -> Dict[str, Any]:
        """Return the plan as a record for ``summary.json``."""
        return {
            "n_total": self.n_total,
            "cap": self.cap,
            "seed": self.seed,
            "n_retained": self.n_total if self.retained is None else len(self.retained),
            "capped": self.retained is not None,
        }


@dataclass
class Collected:
    """A collector's output: the frame, what was drawn, and from where."""

    frame: pd.DataFrame
    composition: Dict[str, int] = field(default_factory=dict)
    n_seen: int = 0
    plan: Optional[Dict[str, Any]] = None
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-safe record of the collection, for the run summary."""
        return {
            "n_rows": int(len(self.frame)),
            "n_seen": int(self.n_seen),
            "composition": dict(self.composition),
            "plan": self.plan,
        }


def _source_of(batch: Any, index: int) -> str:
    """Return the shard basename a sample came from, or ``'unknown'``.

    ``CombinedHDF5Dataset`` stamps ``source_file_basename`` on every sample, so per-file
    provenance is recoverable without a new field.

    Args:
        batch: A batch from the data module.
        index: Position within the batch.

    Returns:
        The basename.
    """
    names = get_field(batch, "source_file_basename")
    if names is None:
        return "unknown"
    if isinstance(names, (list, tuple)):
        return str(names[index])
    return str(names)


def collect_metrics(
    runner: EvalRunner,
    loader: Any,
    per_batch: PerBatchFn,
    *,
    max_samples: Optional[int] = None,
    plan: Optional[CollectionPlan] = None,
    progress_label: str = "collect",
) -> Collected:
    """Run ``per_batch`` over the loader and stack its per-sample outputs into a DataFrame.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        per_batch: Given ``(runner, batch)``, returns ``{column: array of length B}``. Values
            may be tensors, arrays or lists.
        max_samples: Stop after this many samples. A *prefix* cap, appropriate only for a smoke
            run; use ``plan`` for a representative subsample.
        plan: Which global indices to retain. ``None`` retains everything.
        progress_label: Prefix for the progress log line.

    Returns:
        The collected rows, always carrying ``guid`` and ``source_file`` columns alongside
        whatever ``per_batch`` produced.

    Raises:
        ValueError: If ``per_batch`` returns a column whose length is not the batch size --
            otherwise the columns silently misalign and every row after the first short batch
            describes the wrong sample.
    """
    rows: List[Dict[str, Any]] = []
    composition: Counter = Counter()
    global_index = 0
    n_seen = 0

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        # The *batch* is the authority on how many samples there are, never the returned columns.
        # Taking the size from the columns would let a short column silently redefine the batch,
        # dropping a sample per batch and misaligning `guid` and `source_file` -- which come from
        # the batch -- against every metric column.
        batch_size = batch_size_of(batch)
        columns = per_batch(runner, batch)
        n_seen += batch_size

        # Attached here rather than by each analysis: the class and the subgroup are properties of
        # the *sample*, not of the question being asked of it, and every analysis that collects a
        # per-sample frame wants both. Doing it once is also what makes a by-group variant a
        # ``groupby`` on an existing column rather than a second pass over the loader.
        group_columns = batch_labels(batch, batch_size)

        for offset in range(batch_size):
            index = global_index + offset
            if not (plan is None or plan.keeps(index)):
                continue
            source = _source_of(batch, offset)
            row: Dict[str, Any] = {
                "sample_index": index,
                "guid": guid_of(batch, offset),
                "source_file": source,
            }
            row.update({name: values[offset] for name, values in group_columns.items()})
            for name, values in columns.items():
                array = _as_column(name, values, batch_size)
                row[name] = array[offset]
            rows.append(row)
            composition[source] += 1
        global_index += batch_size

    frame = pd.DataFrame(rows)
    logger.info(
        f"[{progress_label}] collected {len(frame)} row(s) from {n_seen} sample(s); "
        f"composition {dict(composition)}"
    )
    return Collected(
        frame=frame,
        composition=dict(sorted(composition.items())),
        n_seen=n_seen,
        plan=None if plan is None else plan.describe(),
    )


def _as_column(name: str, values: Any, batch_size: int) -> np.ndarray:
    """Coerce one collected column to a length-$B$ array, raising on a length mismatch.

    Args:
        name: Column name, for the error message.
        values: The collected values.
        batch_size: Expected length.

    Returns:
        The values as a 1-D array of length ``batch_size``.

    Raises:
        ValueError: If the length disagrees with the batch size.
    """
    if isinstance(values, (list, tuple)):
        array = np.asarray(values, dtype=object if _is_stringy(values) else None)
    else:
        array = np.asarray(to_numpy(values))
    if array.ndim == 0:
        array = np.repeat(array[None], batch_size)
    if array.shape[0] != batch_size:
        raise ValueError(
            f"collected column {name!r} has length {array.shape[0]} but the batch holds "
            f"{batch_size} sample(s). A per-batch function must return one value per sample; "
            f"a mismatch silently misaligns every column of every later row."
        )
    return array


def _is_stringy(values: Sequence[Any]) -> bool:
    """Whether a sequence holds strings, which must not be cast to float."""
    return bool(values) and isinstance(values[0], str)


def collect_predictions(
    runner: EvalRunner,
    loader: Any,
    *,
    plan: Optional[CollectionPlan] = None,
    max_samples: Optional[int] = None,
) -> Collected:
    r"""Retain the overlap-averaged forecast and its target for the planned samples.

    The averaged $(T, c_y)$ rendering rather than the raw $(A, H_d, c_y)$ per-anchor tensor: the
    figures that consume this draw a channel-by-time heatmap, and retaining the per-anchor form
    would cost $1.9$ MB per sample per field instead of $131$ KB. See the module docstring.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        plan: Which global indices to retain.
        max_samples: Prefix cap on iteration.

    Returns:
        A :class:`Collected` whose ``arrays`` carries ``forecast`` and ``target``, each
        $(N, T, c_y)$, and whose ``frame`` carries the matching GUIDs and source files.
    """
    from teb_vae.lag_attn.figure_primitives import average_forecast_per_channel

    horizon = int(runner.model.horizon)
    forecasts: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    composition: Counter = Counter()
    global_index = 0
    n_seen = 0

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        outputs = runner.forward(batch)
        y_st, y_ph = runner.build_target_streams(batch)
        target = torch.cat([y_st, y_ph], dim=-1)
        mu_full = outputs["mu_full"]
        batch_size, seq_len = int(mu_full.shape[0]), int(target.shape[1])
        warmup = int(runner.model._warmup_steps(seq_len))
        n_seen += batch_size

        for offset in range(batch_size):
            index = global_index + offset
            if not (plan is None or plan.keeps(index)):
                continue
            forecasts.append(
                average_forecast_per_channel(
                    to_numpy(mu_full[offset]), seq_len, horizon, warmup
                )
            )
            targets.append(to_numpy(target[offset]))
            source = _source_of(batch, offset)
            rows.append(
                {"sample_index": index, "guid": guid_of(batch, offset), "source_file": source}
            )
            composition[source] += 1
        global_index += batch_size

    arrays = {
        "forecast": np.stack(forecasts) if forecasts else np.zeros((0, 0, 0), dtype=np.float32),
        "target": np.stack(targets) if targets else np.zeros((0, 0, 0), dtype=np.float32),
    }
    logger.info(f"[predictions] retained {len(rows)} of {n_seen} sample(s)")
    return Collected(
        frame=pd.DataFrame(rows),
        composition=dict(sorted(composition.items())),
        n_seen=n_seen,
        plan=None if plan is None else plan.describe(),
        arrays=arrays,
    )


def collect_attention(
    runner: EvalRunner,
    loader: Any,
    *,
    plan: Optional[CollectionPlan] = None,
    max_samples: Optional[int] = None,
) -> Collected:
    r"""Retain the per-sample attention weights $(T, M, L)$ for the planned samples.

    The heaviest retention in the pipeline -- see the module docstring for the arithmetic. The
    weights come back in lag order with index $0$ the current step, matching
    ``LagAttention.build_lag_mask``, so a lag index needs no reindexing before it is converted to
    seconds.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        plan: Which global indices to retain.
        max_samples: Prefix cap on iteration.

    Returns:
        A :class:`Collected` whose ``arrays['attn_weights']`` is $(N, T, M, L)$.
    """
    weights: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    composition: Counter = Counter()
    global_index = 0
    n_seen = 0

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        outputs = runner.forward(batch)
        alpha = outputs["attn_weights"]
        batch_size = int(alpha.shape[0])
        n_seen += batch_size

        for offset in range(batch_size):
            index = global_index + offset
            if not (plan is None or plan.keeps(index)):
                continue
            weights.append(to_numpy(alpha[offset]))
            source = _source_of(batch, offset)
            rows.append(
                {"sample_index": index, "guid": guid_of(batch, offset), "source_file": source}
            )
            composition[source] += 1
        global_index += batch_size

    stacked = np.stack(weights) if weights else np.zeros((0, 0, 0, 0), dtype=np.float32)
    logger.info(f"[attention] retained {len(rows)} of {n_seen} sample(s)")
    return Collected(
        frame=pd.DataFrame(rows),
        composition=dict(sorted(composition.items())),
        n_seen=n_seen,
        plan=None if plan is None else plan.describe(),
        arrays={"attn_weights": stacked},
    )
