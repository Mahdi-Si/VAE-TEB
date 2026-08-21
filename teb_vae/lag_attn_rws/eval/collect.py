r"""The shared collection pass, and the durable tables every later analysis reads.

One forward over the split is the dominant cost of an evaluation -- four latent branches decoded
over $H \cdot R = 480$ raw samples per anchor at $K$ Monte Carlo draws -- and almost every
analysis wants the same forward. So the pass runs **once**, and what it produces is written down:

* ``per_sample.csv`` -- one row per segment, carrying every scalar readout beside the identity a
  clinical question is asked in: ``guid``, ``epoch``, ``clinical_class``, ``subgroup``,
  ``cs_label``, ``bg_label``, ``time_from_labor_onset``, ``source_file_basename``, ``n_anchors``
  and ``n_segments_in_guid``. The labels are attached **here**, once, rather than by each
  analysis: the class and the subgroup are properties of the sample, not of the question, so a
  by-class number is a ``groupby`` on an existing column rather than a second pass.
* ``per_anchor.parquet`` -- one row per *contributing* anchor, keyed
  ``(guid, epoch, anchor)``, carrying $K_t$, both per-anchor score pairs and their gaps, the
  argmax lag and the forecast coverage. Its load-bearing consumer is contraction-conditioned
  coupling, which cannot be recovered from a per-segment mean.
* ``per_sample_vectors.npz`` -- the per-sample vector readouts (the per-dimension KL and the four
  $L$-wide lag vectors), in ``per_sample.csv``'s row order. A sidecar rather than $4L + d_z$
  extra CSV columns, which at the shipped geometry would be $412$ of them.
* ``collection.json`` -- the provenance sidecar: which checkpoint, which seed, which
  ``eval_config``, how many rows, what was excluded and why, the readouts the pass produced, and
  the three facts an offline analysis cannot recover for itself -- the model's anchor-to-raw
  geometry, the loader's FHR z-scoring, and the model's own bound conventions, without which
  nothing downstream can shade a warm-up prefix, quote a number in bpm, or draw a clamp margin
  where the readouts were measured against it.

**A segment that scored no anchors measured nothing, and its columns are NaN rather than zero.**
The per-sample mean divides by a denominator clamped to $1$, so an empty numerator reads as
exactly ``0.0`` -- a fabricated score, not a small one. Averaged into a summed-$480$-sample block
figure of hundreds of nats it drags the headline toward zero and shrinks ``pred_gap`` with no
other symptom. NaN is the representation that makes every downstream ``mean()`` skip it by
default, and the exclusions are counted per recording and per subgroup rather than merely
dropped.

**Heavy quantities, one decision each.** Three things several later analyses want are on neither
table, and each gets a different treatment rather than a blanket one. Per retained sample at the
shipped geometry ($T = 300$, $T_{\mathrm{valid}} = 270$, $H = 30$, $R = 16$, $L = 91$, $M = 4$,
fp32):

* **Per-raw-sample residuals and log-variances** ($T_{\mathrm{valid}} \times H \times R$,
  $506$ KiB per tensor) -- **streamed as an exact accumulator**, resolved by horizon step. What
  calibration and horizon-resolved skill need from them are sums, and a sum of $10^9$ raw samples
  costs $H$ floats instead of terabytes. Resolved by $\tau$ because that axis exists on neither
  table and cannot be recovered from either.
* **The per-anchor forecast waveform** (the same $506$ KiB tensor, four of them for truth, base,
  full and the full branch's log-variance: $\approx 2.0$ MiB per sample) -- **retained under a
  seeded cap**, ``caps.waveforms``. The overlay figures and the event analysis need actual
  waveforms, but for a handful of recordings, not for all of them.
* **The attention weights** ($T \times M \times L$, $427$ KiB per sample) -- **retained under
  ``caps.attention``**, the same reasoning and the largest single retention here.
* **The per-anchor lag map** ($T \times L$, $107$ KiB per sample) -- **not retained**. The
  per-anchor table carries its argmax and the vectors sidecar carries its per-sample profile;
  lean-limit: the per-anchor $\times$ lag heatmap needs a second pass, which is worth adding when
  an analysis actually draws one.

A cap is **opt-in**: a quantity absent from ``eval_config.caps`` is retained for *no* samples.
The alternative default -- retain everything unless capped -- is $2.4$ MiB per sample, which is
several gigabytes over a real split, held for a figure nobody asked for.

**A finished run is re-runnable without a model.** :func:`load_or_collect` reads the tables back
when the sidecar agrees with the run being asked for, and **refuses** when it does not: a
directory holding another checkpoint's tables, or a table shorter than the sidecar says, is a
silent mixing of two runs, and a mismatch that reads as success is worse than a re-collection.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn_rws.eval import events, spectra
from teb_vae.lag_attn_rws.eval._reuse import labels, subsample_indices
from teb_vae.lag_attn_rws.eval.metrics import (
    VECTOR_READOUTS,
    BatchReadout,
    batch_field,
    check_cached_verdicts,
    evaluate,
)
from teb_vae.lag_attn_rws.eval.report_seam import json_safe
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC, SATURATION_FRAC

#: How ``read_csv`` parses the per-sample table's floats. ``pandas`` writes a float in its shortest
#: exactly-round-tripping form but **reads** it back with a fast parser that is not exact, so the
#: default round trip loses the last bits of every value.
#:
#: That is not cosmetic. A fresh run's analyses read the frame in memory while a re-run's read it
#: off disk, so an inexact round trip makes the *same* run report different numbers depending on
#: which it was: a per-recording mean amplifies a last-bit disagreement through the cancellation in
#: quantities like the mean signed error, and a re-run's summary then fails to compare equal to the
#: summary it re-ran -- which is the property the offline path exists to have.
PER_SAMPLE_FLOAT_PRECISION = "round_trip"

#: The two durable tables, their sidecars, and the provenance record.
PER_SAMPLE_FILENAME = "per_sample.csv"
PER_ANCHOR_FILENAME = "per_anchor.parquet"
VECTORS_FILENAME = "per_sample_vectors.npz"
RETAINED_FILENAME = "retained_arrays.npz"
COLLECTION_FILENAME = "collection.json"

#: The pooled cross-spectral maps, at full frequency resolution and resolved by lead time. A third
#: sidecar rather than a block on the record, because $14$ arrays of $(H, F)$ per cohort is about a
#: megabyte of JSON that ``record_summary_view`` would then embed into ``summary.json``; and not a
#: fourth family inside ``per_sample_vectors.npz``, which is documented as row-aligned with
#: ``per_sample.csv`` and whose consumers index it that way.
COHERENCE_FILENAME = "coherence_spectra.npz"

#: Prefix under which the per-segment cross-spectral sums travel in the vectors sidecar. They are
#: row-aligned like every other vector readout, but they are **sums** rather than means, so they
#: are deliberately not members of ``metrics.VECTOR_READOUTS`` -- that tuple drives
#: ``aggregate_by_recording``, which takes per-recording *means*, and a mean of two segments' sums
#: is not a quantity anything downstream could use.
SPECTRAL_VECTOR_PREFIX = "coherence_"

#: Cohort key under which the whole split's pooled spectra are accumulated, beside one key per
#: clinical class. Named rather than ``None`` so the sidecar reads without a legend.
SPECTRA_POOLED_KEY = "all"

#: Columns that identify a segment, as opposed to scoring it. Ordered so the head of
#: ``per_sample.csv`` reads as "which recording, when, which cohort" before any number.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "sample_index",
    "guid",
    "epoch",
    labels.CLASS_COLUMN,
    labels.SUBGROUP_COLUMN,
    "cs_label",
    "bg_label",
    "time_from_labor_onset",
    "second_stage_onset",
    "source_file_basename",
    "n_anchors",
    "n_segments_in_guid",
)

#: The per-anchor table's key. Unique by construction -- one recording's segments are disjoint in
#: time, so they carry distinct ``epoch`` values -- and checked rather than assumed, because a
#: duplicate key silently breaks every join built on it.
PER_ANCHOR_KEY: Tuple[str, ...] = ("guid", "epoch", "anchor")

#: ``eval_config.caps`` name -> the model tensors that cap retains. The names are the model's own
#: forward-output keys, plus ``target`` for the raw future the forecast is scored against, so a
#: retained array cannot be a differently assembled version of what was scored.
#:
#: ``up_raw`` and ``weight`` ride with the waveforms rather than forming a third quantity, and
#: that is a correctness choice rather than a saving. The event analysis triggers on contractions
#: found in ``up_raw`` and averages the forecast blocks around them; a separate cap would draw a
#: separate sample set, and the two halves of that average would then describe different
#: recordings. They cost about $1\%$ of what they travel with -- $4800$ and $300$ floats against
#: four $(T_{\mathrm{valid}}, H, R)$ blocks.
RETAINED_QUANTITIES: Dict[str, Tuple[str, ...]] = {
    "waveforms": ("target", "mu_base", "mu_full", "logvar_full", "up_raw", "weight"),
    "attention": ("attn_weights",),
}

#: Per-anchor column naming how long ago the most recent contraction started, in seconds. NaN
#: where the anchor has no contraction behind it in this segment -- never a large number, which
#: every threshold downstream would read as "long ago" rather than as "never".
CONTRACTION_AGE_COLUMN = "seconds_since_contraction"

#: How often the collection pass reports where it is, in batches. The pass is the multi-hour step
#: of a production run and every other step is seconds, so silence here is silence for the whole
#: run; and an operator who cannot tell a slow pass from a hung one will restart a healthy one.
#: A constant rather than a setting: it changes nothing about the numbers, and ``eval_config`` is
#: reserved for values that do.
PROGRESS_EVERY_BATCHES = 25

#: The per-horizon-step accumulator's fields, per branch. ``count`` is the number of raw samples
#: behind each $\tau$ bin -- $\sum_{b,t} m_{b,t,\tau} \cdot R$ -- and ``n_anchors`` the anchors
#: behind it; both are emitted so every sum can be turned into a mean by a reader who has only
#: this record, and so the horizon-resolved score divides by its own per-$\tau$ denominator rather
#: than by a per-anchor indicator that would count masked steps as scored zeros.
HORIZON_STATISTICS: Tuple[str, ...] = (
    "sum_sq", "sum_standardised_sq", "sum_logvar", "count", "sum_block", "n_anchors",
)


class TablesProvenanceMismatch(RuntimeError):
    """A finished run's tables do not describe the run being asked for.

    Distinct from a crash: the tables are intact and readable, they simply belong to a different
    checkpoint, seed or evaluation configuration. Continuing would mix two runs' numbers under one
    summary.
    """


# =============================================================================
# Retention
# =============================================================================
@dataclass(frozen=True)
class RetentionPlan:
    """Which samples' heavy arrays are kept, decided before the pass starts.

    Attributes:
        seed: Seed for the draw, so a re-run keeps the same samples.
        n_total: Samples the pass will see, needed to draw over the whole index space rather than
            over a prefix. ``None`` is legal only when no quantity is capped.
        caps: The validated ``eval_config.caps`` mapping. A quantity **absent** from it is
            retained for no samples; a cap of ``None`` retains every sample; an integer retains
            that many.
        retained: Quantity -> the sample positions kept, or ``None`` for "all of them".
    """

    seed: int
    n_total: Optional[int]
    caps: Dict[str, Optional[int]]
    retained: Dict[str, Optional[Set[int]]]

    @classmethod
    def build(
        cls, caps: Optional[Dict[str, Optional[int]]], *, n_total: Optional[int], seed: int
    ) -> "RetentionPlan":
        """Draw the retained positions for every capped quantity.

        The draw is over the whole index space, never a prefix: the pass runs over a fixed-seed
        *shuffled* loader, so a position is already a uniform draw over the population, but a
        prefix would still land wherever the shuffle happened to put the first batches.

        Args:
            caps: The validated caps mapping, or ``None`` for an empty one.
            n_total: Samples the pass will see.
            seed: Seed for the draw.

        Returns:
            The plan.

        Raises:
            ValueError: If a quantity is capped but ``n_total`` is unknown -- the draw would then
                have to become a prefix, which is the one thing a cap must never be.
        """
        caps = dict(caps or {})
        retained: Dict[str, Optional[Set[int]]] = {}
        for quantity in RETAINED_QUANTITIES:
            if quantity not in caps:
                retained[quantity] = set()
                continue
            cap = caps[quantity]
            if cap is None:
                retained[quantity] = None
                continue
            if n_total is None:
                raise ValueError(
                    f"eval_config.caps.{quantity} = {cap} needs the sample count of the pass to "
                    f"draw over, and the loader did not report one. Either drop the cap or hand "
                    f"the collection pass an explicit n_total; a cap resolved without it could "
                    f"only be a prefix, which draws one subgroup out of eight concatenated "
                    f"per-subgroup shards."
                )
            drawn = subsample_indices(int(n_total), int(cap), int(seed))
            retained[quantity] = (
                None if drawn is None else {int(value) for value in drawn.tolist()}
            )
        return cls(seed=int(seed), n_total=n_total, caps=caps, retained=retained)

    def keeps(self, quantity: str, index: int) -> bool:
        """Whether the sample at global position ``index`` is retained for ``quantity``."""
        planned = self.retained.get(quantity, set())
        return planned is None or int(index) in planned

    def tensor_names(self) -> Tuple[str, ...]:
        """Return the model tensors any active quantity needs, in a stable order.

        Empty when nothing is retained, which is what lets the pass skip carrying them at all.
        """
        names: List[str] = []
        for quantity, tensors in RETAINED_QUANTITIES.items():
            if self.retained.get(quantity) == set():
                continue
            names.extend(name for name in tensors if name not in names)
        return tuple(names)

    def describe(
        self,
        retained_bytes: Optional[Dict[str, int]] = None,
        kept_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Return the plan as a record for the summary.

        Args:
            retained_bytes: Bytes actually held per quantity, measured after the pass. A cap that
                did not reduce anything is then visible as a number rather than as an intention.
            kept_counts: Samples actually retained per quantity, measured after the pass. Reported
                beside ``n_planned`` because the two can disagree -- the plan is drawn over the
                index space the pass was expected to reach, and a pass that stopped early reached
                less of it. A plan that intended eight and kept none says so here rather than
                leaving a figure mysteriously absent.

        Returns:
            One entry per quantity: its cap, how many samples it planned and kept, and its size.
        """
        record: Dict[str, Any] = {"seed": self.seed, "n_total": self.n_total, "quantities": {}}
        for quantity in RETAINED_QUANTITIES:
            planned = self.retained.get(quantity, set())
            record["quantities"][quantity] = {
                # "absent" is the third state, and it is the default one: not capped, and not
                # uncapped either -- retained for nobody until a cap asks.
                "cap": self.caps.get(quantity, "absent"),
                "n_planned": self.n_total if planned is None else len(planned),
                "n_kept": (
                    None if kept_counts is None else int(kept_counts.get(quantity, 0))
                ),
                "tensors": list(RETAINED_QUANTITIES[quantity]),
                "n_bytes": (
                    None if retained_bytes is None else int(retained_bytes.get(quantity, 0))
                ),
            }
        return record


# =============================================================================
# Batch plumbing
# =============================================================================
def _numeric_column(batch: Any, name: str, batch_size: int) -> np.ndarray:
    """Read one per-sample numeric field as a length-$B$ float array.

    Args:
        batch: A batch from the data module.
        name: The field name.
        batch_size: The batch's sample count, taken from a tensor field rather than from this
            column -- see :func:`_check_length`.

    Returns:
        The values, or an all-NaN column when the batch does not carry the field. NaN rather than
        zero: ``time_from_labor_onset`` is genuinely NaN wherever the recording is absent from the
        labour-onset table, and a zero there would read as "at labour onset".

    Raises:
        ValueError: If the field is present but does not hold one value per sample.
    """
    values = batch_field(batch, name)
    if values is None:
        return np.full(batch_size, np.nan, dtype=np.float64)
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().to(torch.float64).numpy().reshape(-1)
    else:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    _check_length(name, array.shape[0], batch_size)
    return array


def _string_column(batch: Any, name: str, batch_size: int) -> List[Optional[str]]:
    """Read one per-sample string field, tolerating collation into a list or a bare value.

    Args:
        batch: A batch from the data module.
        name: The field name.
        batch_size: The batch's sample count.

    Returns:
        One value per sample, ``None`` where the batch carries the field for nobody.

    Raises:
        ValueError: If the field is a sequence of the wrong length.
    """
    values = batch_field(batch, name)
    if values is None:
        return [None] * batch_size
    if isinstance(values, (list, tuple)):
        _check_length(name, len(values), batch_size)
        return [str(value) for value in values]
    return [str(values)] * batch_size


def _check_length(name: str, found: int, batch_size: int) -> None:
    """Raise when a per-sample column is not one value per sample.

    The batch is the authority on how many samples there are, always. Taking the count from a
    column instead would let a short one redefine the batch, and every row after it would carry
    one sample's ``guid`` beside another sample's numbers -- a misalignment that produces
    perfectly plausible output.

    Args:
        name: The column name, for the message.
        found: The length observed.
        batch_size: The length required.

    Raises:
        ValueError: On any disagreement.
    """
    if found != batch_size:
        raise ValueError(
            f"batch field {name!r} holds {found} value(s) but the batch holds {batch_size} "
            f"sample(s). A per-sample field must carry one value per sample; a short column "
            f"silently misaligns guid, class and subgroup against every readout on the row."
        )


def _class_column(batch: Any, batch_size: int) -> List[Optional[str]]:
    r"""Recover each sample's clinical class from its weight-scaled ``target``.

    ``target(t) = class\_id \cdot weight(t)``, so an acidosis step (code $2$) at
    $\mathrm{weight} = 0.5$ stores $1.0$ -- indistinguishable from a fully valid healthy step.
    The recovery divides the weight back out; a pad-only window and a uniformly zero target both
    yield *no class* rather than a phantom class $0$.

    Args:
        batch: A batch from the data module.
        batch_size: The batch's sample count.

    Returns:
        One class name per sample, ``None`` where the sample carries no class.
    """
    target = batch_field(batch, "target")
    weight = batch_field(batch, "weight")
    if target is None or weight is None:
        return [None] * batch_size
    return [
        labels.class_name(labels.clinical_class_code(target[index], weight[index]))
        for index in range(batch_size)
    ]


# =============================================================================
# The collection
# =============================================================================
@dataclass
class Collection:
    """Everything one pass over the split produced.

    Attributes:
        per_sample: One row per segment.
        per_anchor: One row per contributing anchor.
        vectors: Per-sample vector readouts, aligned with ``per_sample``'s row order.
        retained: Heavy arrays kept under the retention plan, each with a matching
            ``<name>_sample_index`` array so a row can be traced back to its segment.
        spectra: The pooled cross-spectral maps, keyed ``<cohort>_<statistic>`` and shaped
            $(H, F)$ -- **not** row-aligned with ``per_sample``, which is why they live in their
            own sidecar rather than in ``vectors``. Empty on a geometry too short to hold one Welch
            window, and on a run collected before the coherence analysis existed.
        results: The readouts, verdicts and control accounting the pass computed.
        record: The provenance sidecar -- see :func:`write_collection`.
        from_cache: Whether this came off disk rather than out of a forward pass.
    """

    per_sample: pd.DataFrame
    per_anchor: pd.DataFrame
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    retained: Dict[str, np.ndarray] = field(default_factory=dict)
    spectra: Dict[str, np.ndarray] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    record: Dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False


class _Collector:
    """Accumulates the two tables while :func:`~teb_vae.lag_attn_rws.eval.metrics.evaluate` runs.

    A sink rather than a second loop: the readouts, the tables and the retained arrays all come
    off the same forward, and a second pass over the split would double the only expensive part of
    a run. The per-anchor rows are accumulated as per-batch arrays and concatenated once, because
    a real split produces millions of them and a list of dicts would not fit.
    """

    def __init__(self, plan: RetentionPlan, *, geometry: Any = None) -> None:
        self._plan = plan
        self._geometry = geometry
        self._rows: Dict[str, List[Any]] = {}
        self._vectors: Dict[str, List[np.ndarray]] = {name: [] for name in VECTOR_READOUTS}
        self._anchor_blocks: List[Dict[str, np.ndarray]] = []
        self._retained: Dict[str, List[np.ndarray]] = {}
        self._retained_index: Dict[str, List[int]] = {}
        self._horizon: Dict[str, np.ndarray] = {}
        self._spectral: Dict[str, List[np.ndarray]] = {}
        self._pooled: Dict[str, np.ndarray] = {}
        self._band_index: Optional[np.ndarray] = None
        self._sample_index = 0
        self._n_batches = 0
        self._started = time.perf_counter()

    # -- the sink -----------------------------------------------------------------
    def observe(self, batch: Any, readout: BatchReadout) -> None:
        """Record one scored batch.

        Args:
            batch: The batch, on the model's device, for the identity columns.
            readout: Its per-sample readouts, per-anchor tensors and horizon sums.
        """
        batch_size = len(readout.guids)
        n_anchors = readout.n_anchors.detach().cpu().to(torch.float64).numpy()
        scored = n_anchors > 0.0

        self._append_identity(batch, readout, batch_size, n_anchors)
        self._append_readouts(readout, scored)
        self._append_vectors(readout, scored)
        self._append_anchors(batch, readout, batch_size)
        self._append_retained(readout, batch_size)
        self._accumulate_horizon(readout)
        self._accumulate_spectral(batch, readout, batch_size, scored)
        self._sample_index += batch_size
        self._n_batches += 1
        self._log_progress()

    def _log_progress(self) -> None:
        """Report throughput and a remaining-time estimate every so many batches.

        The estimate is a straight extrapolation of the average rate so far, and it is stated as
        one rather than dressed up: the pass is uniform work per sample, so the average is a good
        estimate, and a wrong one is still incomparably better than silence.

        Omitted entirely when the loader could not say how many samples it holds -- an estimate
        against an unknown total would be a number with no meaning, and the throughput alone is
        still worth logging.
        """
        if self._n_batches % PROGRESS_EVERY_BATCHES:
            return
        elapsed = max(time.perf_counter() - self._started, 1e-9)
        rate = self._sample_index / elapsed
        total = self._plan.n_total
        if total and rate > 0.0:
            remaining = max(int(total) - self._sample_index, 0) / rate
            logger.info(
                f"collection: {self._sample_index}/{int(total)} sample(s) over "
                f"{self._n_batches} batch(es), {rate:.1f} samples/s, "
                f"~{remaining / 60.0:.1f} min remaining"
            )
            return
        logger.info(
            f"collection: {self._sample_index} sample(s) over {self._n_batches} batch(es), "
            f"{rate:.1f} samples/s (the loader reported no total, so there is no estimate)"
        )

    # -- per-sample ----------------------------------------------------------------
    def _append_identity(
        self, batch: Any, readout: BatchReadout, batch_size: int, n_anchors: np.ndarray
    ) -> None:
        """Append the columns that say which segment a row is, rather than how it scored."""
        basenames = _string_column(batch, "source_file_basename", batch_size)
        first = self._sample_index
        self._extend("sample_index", list(range(first, first + batch_size)))
        self._extend("guid", list(readout.guids))
        self._extend("epoch", _numeric_column(batch, "epoch", batch_size).tolist())
        self._extend(labels.CLASS_COLUMN, _class_column(batch, batch_size))
        self._extend(
            labels.SUBGROUP_COLUMN, [labels.subgroup_of(name) for name in basenames]
        )
        self._extend("cs_label", _numeric_column(batch, "cs_label", batch_size).tolist())
        self._extend("bg_label", _numeric_column(batch, "bg_label", batch_size).tolist())
        self._extend(
            "time_from_labor_onset",
            _numeric_column(batch, "time_from_labor_onset", batch_size).tolist(),
        )
        self._extend(
            "second_stage_onset",
            _numeric_column(batch, "second_stage_onset", batch_size).tolist(),
        )
        self._extend("source_file_basename", basenames)
        self._extend("n_anchors", [int(value) for value in n_anchors])

    def _append_readouts(self, readout: BatchReadout, scored: np.ndarray) -> None:
        """Append the scalar readouts, NaN wherever the segment scored no anchors."""
        for name, values in readout.columns.items():
            column = values.detach().cpu().to(torch.float64).numpy()
            # Not zero. The per-sample mean clamps its denominator to 1, so an unscored segment's
            # numerator of nothing reads as exactly 0.0 -- which averages in as a real score.
            self._extend(name, np.where(scored, column, np.nan).tolist())

    def _append_vectors(self, readout: BatchReadout, scored: np.ndarray) -> None:
        """Append the per-sample vector readouts, blanked on the same rule as the scalars."""
        for name in VECTOR_READOUTS:
            rows = getattr(readout, name).detach().cpu().to(torch.float64).numpy()
            self._vectors[name].append(np.where(scored[:, None], rows, np.nan))

    def _extend(self, name: str, values: Sequence[Any]) -> None:
        """Append one column's per-sample values, keeping every column the same length."""
        self._rows.setdefault(name, []).extend(values)

    # -- per-anchor ----------------------------------------------------------------
    def _append_anchors(self, batch: Any, readout: BatchReadout, batch_size: int) -> None:
        """Append one row per contributing anchor of this batch.

        Only contributing anchors: an anchor the reconstruction does not score has no forecast
        term and no KL support, so a row for it would carry a column of structural zeros that
        every later mean would average in. That is also why a segment scoring nothing leaves no
        row here at all, which is this table's form of the same exclusion the sample table writes
        as NaN.

        Args:
            batch: The batch, for the raw source trace the contraction column is derived from.
            readout: The batch's readouts, carrying the per-anchor tensors.
            batch_size: The batch's sample count -- also how far back this batch's epochs reach
                in the per-sample column, which was appended immediately before this call.
        """
        if not readout.per_anchor:
            return
        contributing = readout.per_anchor["contributing"].detach().cpu().numpy() > 0.0
        if not contributing.any():
            return

        sample_positions, anchor_positions = np.nonzero(contributing)
        block: Dict[str, np.ndarray] = {
            "sample_index": sample_positions.astype(np.int64) + self._sample_index,
            "guid": np.asarray(readout.guids, dtype=object)[sample_positions],
            # Read back off the per-sample column rather than off the batch a second time, so the
            # two tables cannot disagree about which epoch a segment carries.
            "epoch": np.asarray(self._rows["epoch"][-batch_size:], dtype=np.float64)[
                sample_positions
            ],
            "anchor": anchor_positions.astype(np.int64),
        }
        for name, values in readout.per_anchor.items():
            if name == "contributing":
                # Implicit: every row of this table is a contributing anchor.
                continue
            array = values.detach().cpu().numpy()
            block[name] = array[sample_positions, anchor_positions]
        ages = self._contraction_ages(batch, batch_size, contributing.shape[1])
        block[CONTRACTION_AGE_COLUMN] = ages[sample_positions, anchor_positions]
        self._anchor_blocks.append(block)

    def _contraction_ages(self, batch: Any, batch_size: int, n_anchors: int) -> np.ndarray:
        r"""Return, per sample and anchor, the seconds since the most recent contraction onset.

        Computed here rather than in an analysis because this is the only pass that holds the raw
        UP trace: the model reads the source as scattering and phase channels, so a contraction
        exists nowhere in the tables unless it is put there. It is the per-anchor table's
        load-bearing column -- restricting the coupling readout to anchors a contraction could
        still be influencing is a comparison a per-segment mean cannot express.

        The onset is taken on the **stored** timeline, which the preprocessing has already
        advanced UP by $20\,$s on. Adding that back would double-count a correction that was made
        once, deliberately, upstream.

        Args:
            batch: The batch, for its raw ``up`` and its ``weight``.
            batch_size: Samples in the batch.
            n_anchors: Anchors the readout scored, $T_{\mathrm{valid}}$.

        Returns:
            A $(B, T_{\mathrm{valid}})$ float array, NaN where an anchor has no contraction
            behind it -- and NaN throughout when the batch carries no ``up`` or the geometry is
            unknown, so a missing input reads as "not measured" rather than as "never".
        """
        ages = np.full((batch_size, n_anchors), np.nan, dtype=np.float64)
        up = batch_field(batch, "up")
        if self._geometry is None or not isinstance(up, torch.Tensor):
            return ages
        decimation = int(self._geometry.decimation)
        raw_len = int(self._geometry.raw_len)
        up_np = up.detach().cpu().to(torch.float64).numpy().reshape(batch_size, -1)[:, :raw_len]
        weight = batch_field(batch, "weight")
        weights = (
            None if not isinstance(weight, torch.Tensor)
            else weight.detach().cpu().to(torch.float64).numpy().reshape(batch_size, -1)
        )
        # Each anchor's causal endpoint on the raw grid; a contraction at or before it is one the
        # anchor could have read.
        endpoints = decimation * (np.arange(n_anchors, dtype=np.float64) + 1.0) - 1.0
        for index in range(batch_size):
            valid = (
                None if weights is None
                else events.raw_validity(weights[index], decimation=decimation, raw_len=raw_len)
            )
            onsets = np.sort(events.detect_contractions(up_np[index], valid=valid)["onset_raw"])
            if onsets.size == 0:
                continue
            # The most recent onset at or before each endpoint; -1 marks an anchor with none.
            previous = np.searchsorted(onsets, endpoints, side="right") - 1
            recent = np.where(previous >= 0, onsets[np.clip(previous, 0, None)], np.nan)
            ages[index] = (endpoints - recent) / events.FS_RAW
        return ages

    # -- heavy arrays --------------------------------------------------------------
    def _append_retained(self, readout: BatchReadout, batch_size: int) -> None:
        """Keep the planned samples' heavy arrays, and nothing else."""
        if not readout.retained:
            return
        for quantity, tensors in RETAINED_QUANTITIES.items():
            keep = [
                offset
                for offset in range(batch_size)
                if self._plan.keeps(quantity, self._sample_index + offset)
            ]
            if not keep or not all(name in readout.retained for name in tensors):
                continue
            self._retained_index.setdefault(quantity, []).extend(
                self._sample_index + offset for offset in keep
            )
            for name in tensors:
                rows = readout.retained[name].detach().cpu().numpy()
                self._retained.setdefault(name, []).append(rows[keep])

    # -- accumulators --------------------------------------------------------------
    def _accumulate_horizon(self, readout: BatchReadout) -> None:
        """Add this batch's per-horizon-step residual and log-variance sums.

        Kept in float64 rather than in the model's float32: the sums run over $10^9$ raw samples
        on a real split, and a float32 accumulator loses its low bits long before that.
        """
        for name, values in readout.horizon_sums.items():
            array = values.detach().cpu().to(torch.float64).numpy()
            if name in self._horizon:
                self._horizon[name] = self._horizon[name] + array
            else:
                self._horizon[name] = array

    def _accumulate_spectral(
        self, batch: Any, readout: BatchReadout, batch_size: int, scored: np.ndarray
    ) -> None:
        r"""Reduce this batch's cross-spectral sums two ways, and store both.

        The full $(B, H, F)$ product is roughly $430$ KB per segment -- most of a gigabyte over a
        real split -- so neither marginal is stored whole. What travels instead is:

        * **band-collapsed, lead-resolved, per segment** into the vectors sidecar, which is the
          form every interval, cohort cut and per-recording statistic is computed from; and
        * **frequency-resolved, lead-resolved, pooled** over the split and over each clinical
          class into its own sidecar, which is the form the frequency $\times$ lead-time map is
          drawn from.

        Both are sums, and nothing here ratios. Summing two segments' statistics and ratioing once
        is exactly the ratio of their pooled windows, which is what lets the per-recording
        reduction downstream be a plain ``groupby`` sum -- and what keeps the coherence's
        small-sample bias out of every number.

        The band collapse happens here rather than in ``metrics`` because the pooling needs each
        sample's clinical class, which is a property of the batch rather than of the model; keeping
        both reductions in one place is what stops them disagreeing about which windows were kept.

        Args:
            batch: The batch, for the clinical-class labels the pooled maps are keyed on.
            readout: Its readouts, carrying ``spectral_sums``.
            batch_size: The batch's sample count.
            scored: Which samples scored at least one anchor.
        """
        sums = readout.spectral_sums
        if not sums:
            return
        classes = _class_column(batch, batch_size)

        # Any statistic seen on an earlier batch but absent from this one is padded with NaN rows
        # rather than skipped. Skipping would leave that key short by a batch while every other key
        # grew, and the vectors sidecar is aligned to ``per_sample.csv`` **by position** -- so from
        # the first such batch onward, one sample's spectra would sit on another sample's row and
        # every cohort cut downstream would be reading the wrong recording. The source statistics
        # are the ones this can happen to: they exist only where the batch carried a raw UP trace.
        for name in self._spectral_names(sums):
            values = sums.get(name)
            if values is None:
                template = self._spectral[f"{SPECTRAL_VECTOR_PREFIX}{name}"][0]
                self._spectral[f"{SPECTRAL_VECTOR_PREFIX}{name}"].append(
                    np.full((batch_size,) + template.shape[1:], np.nan, dtype=np.float64)
                )
                continue
            array = values.detach().cpu().to(torch.float64).numpy()
            key = f"{SPECTRAL_VECTOR_PREFIX}{name}"
            if array.ndim == 3:
                # (B, H, F). Pooled first, at full resolution, before the bands are applied.
                self._accumulate_pooled(name, array, classes)
                collapsed = spectra.collapse_to_bands(
                    array, self._spectral_band_index(array.shape[-1]), len(spectra.band_names())
                )
                # Flattened tau-major to one trailing axis, the convention every vector readout
                # follows -- see ``spectra.reshape_band_horizon`` for the one place it is undone.
                rows = collapsed.reshape(array.shape[0], -1)
            else:
                # (B, H): the window counts and the time-domain references. Pooled as well, so the
                # sidecar carries its own denominator rather than borrowing the vectors'.
                self._accumulate_pooled(name, array, classes)
                rows = array
            # Blanked on the same rule as every other vector readout: a segment that scored no
            # anchors measured nothing, and a row of exact zeros would average in as a measurement.
            self._spectral.setdefault(key, []).append(np.where(scored[:, None], rows, np.nan))

    def _accumulate_pooled(
        self, name: str, array: np.ndarray, classes: Sequence[Optional[str]]
    ) -> None:
        """Add one statistic's batch to the pooled map and to each clinical class's own.

        Args:
            name: The statistic's name.
            array: $(B, H, F)$ or $(B, H)$.
            classes: One clinical class per sample, ``None`` where the segment carries none.
        """
        cohorts: Dict[str, Any] = {SPECTRA_POOLED_KEY: slice(None)}
        for label in {value for value in classes if value}:
            cohorts[str(label)] = np.array(
                [value == label for value in classes], dtype=bool
            )
        for cohort, selector in cohorts.items():
            block = array[selector].sum(axis=0)
            key = f"{cohort}_{name}"
            self._pooled[key] = (
                block if key not in self._pooled else self._pooled[key] + block
            )

    def _spectral_names(self, sums: Dict[str, Any]) -> List[str]:
        """Every statistic seen so far, plus this batch's, so no key is ever skipped for a batch.

        The union rather than this batch's own keys. The vectors sidecar is positionally aligned
        with ``per_sample.csv``, so a key that misses one batch is short by that batch's rows for
        ever after -- and every row below the gap then belongs to a different segment. Returning
        the union lets the caller pad the absentees instead.

        Args:
            sums: This batch's cross-spectral statistics.

        Returns:
            The statistic names, in a stable order.
        """
        known = [
            name[len(SPECTRAL_VECTOR_PREFIX) :]
            for name in self._spectral
            if name.startswith(SPECTRAL_VECTOR_PREFIX)
        ]
        return known + [name for name in sums if name not in known]

    def _spectral_band_index(self, n_freq: int) -> np.ndarray:
        r"""The bin-to-band map for this run's spectra, built once.

        Derived from the readout's own bin count rather than from the geometry, so it is correct
        even where the collector was handed no geometry -- and so the map can never describe a
        different ``nperseg`` than the sums it is applied to.

        Args:
            n_freq: One-sided bin count $F$, from which $\mathrm{nperseg} = 2(F - 1)$.

        Returns:
            $(F,)$ of band positions.
        """
        cached = self._band_index
        if cached is None or cached.size != int(n_freq):
            cached = spectra.band_index(spectra.frequency_axis(2 * (int(n_freq) - 1)))
            self._band_index = cached
        return cached

    # -- assembly ------------------------------------------------------------------
    def finish(self) -> "Collection":
        """Build the tables, the sidecars and the accounting.

        Returns:
            The collection, without the ``results`` the pass's caller assembles separately.
        """
        per_sample = pd.DataFrame(self._rows)
        if not per_sample.empty:
            # Counted over every segment the recording contributed, scored or not: it is the
            # denominator of "how much of this recording was measurable", and dropping the
            # unscored ones from it would make that fraction always 1.
            counts = per_sample.groupby("guid")["sample_index"].transform("size")
            per_sample["n_segments_in_guid"] = counts.astype(np.int64)
            # Identity first, then the readouts: the head of the file should read as "which
            # recording, when, which cohort" before it reads as a number.
            per_sample = per_sample.reindex(
                columns=[name for name in IDENTITY_COLUMNS if name in per_sample.columns]
                + [name for name in per_sample.columns if name not in IDENTITY_COLUMNS]
            )

        vectors = {
            name: (
                np.concatenate(blocks, axis=0)
                if blocks
                else np.zeros((0, 0), dtype=np.float64)
            )
            for name, blocks in self._vectors.items()
        }
        # The cross-spectral sums ride the same sidecar and the same row order as every other
        # vector readout, under their own prefix. They are kept out of ``VECTOR_READOUTS`` because
        # that tuple drives the per-recording *mean*, and these are sums.
        vectors.update(
            {name: np.concatenate(blocks, axis=0) for name, blocks in self._spectral.items()}
        )
        per_anchor = _concatenate_blocks(self._anchor_blocks)
        retained = self._finish_retained()
        return Collection(
            per_sample=per_sample,
            per_anchor=per_anchor,
            vectors=vectors,
            retained=retained,
            spectra=dict(self._pooled),
            record=self._build_record(per_sample, per_anchor, retained),
        )

    def _finish_retained(self) -> Dict[str, np.ndarray]:
        """Stack the retained arrays, each beside the sample indices it came from."""
        stacked: Dict[str, np.ndarray] = {
            name: np.concatenate(blocks, axis=0) for name, blocks in self._retained.items()
        }
        for quantity, indices in self._retained_index.items():
            stacked[f"{quantity}_sample_index"] = np.asarray(indices, dtype=np.int64)
        return stacked

    def _build_record(
        self, per_sample: pd.DataFrame, per_anchor: pd.DataFrame, retained: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Assemble the accounting: exclusions, denominators, retention and the accumulators."""
        readout_columns = [
            name for name in per_sample.columns if name not in IDENTITY_COLUMNS
        ]
        excluded = per_sample[per_sample["n_anchors"] <= 0] if not per_sample.empty else per_sample
        retained_bytes = {
            quantity: int(
                sum(
                    int(retained[name].nbytes)
                    for name in tensors
                    if name in retained
                )
            )
            for quantity, tensors in RETAINED_QUANTITIES.items()
        }
        return {
            "n_per_sample_rows": int(len(per_sample)),
            "n_per_anchor_rows": int(len(per_anchor)),
            "n_recordings": int(per_sample["guid"].nunique()) if not per_sample.empty else 0,
            # Excluded from every average and counted here. A run where this is large measured
            # far less than its segment count suggests, and no other number says so.
            "n_segments_excluded_zero_anchors": int(len(excluded)),
            # Every fraction a later analysis reports has to carry the count it was computed
            # over; these are those counts, per column, taken after the zero-anchor blanking.
            "denominators": {
                name: int(per_sample[name].notna().sum()) for name in readout_columns
            },
            # Per recording and per subgroup, because zero-anchor segments are not spread
            # evenly: they cluster in the recordings with the worst signal, so a pooled count
            # hides which cohort the missing measurements came from.
            "excluded_zero_anchors": {
                "per_guid": _counts_of(excluded, "guid"),
                "per_subgroup": _counts_of(excluded, labels.SUBGROUP_COLUMN),
            },
            "retention": self._plan.describe(
                retained_bytes,
                {name: len(indices) for name, indices in self._retained_index.items()},
            ),
            # The residual and log-variance sums, resolved by horizon step. On neither table by
            # construction: both are per anchor, and this axis is inside an anchor.
            "horizon": {name: values.tolist() for name, values in sorted(self._horizon.items())},
            # What the cross-spectral estimator did, so a reader of the numbers can check the
            # convention rather than infer it. Small and flat: the spectra themselves are two
            # sidecars away.
            "coherence": self._coherence_record(),
        }

    def _coherence_record(self) -> Dict[str, Any]:
        """Describe the cross-spectral estimator and how much of the split it could measure.

        The window-drop accounting is the part that matters. Whole-window validity is *stricter*
        than the per-step mask every other analysis scores on, so this analysis legitimately
        describes a smaller population -- and a reader who does not see the drop fraction beside
        the numbers has no way to tell a well-covered run from one where most windows fell in gaps.

        Returns:
            The layout, the band table with its bin counts, the seam bins, and the drop accounting.
            Empty when the pass accumulated nothing.
        """
        if not self._pooled:
            return {}
        record: Dict[str, Any] = {"bands": spectra.band_edges()}
        geometry = self._geometry
        if geometry is not None:
            layout = spectra.slice_geometry(
                t_valid=geometry.t_valid,
                warmup=geometry.warmup,
                horizon=geometry.horizon,
                raw_per_step=geometry.r,
            )
            record.update(layout.describe())
            axis = spectra.frequency_axis(layout.nperseg)
            record["band_bin_counts"] = spectra.band_bin_counts(axis)
            record["seam_bins"] = spectra.seam_bins(layout.nperseg, layout.raw_per_step).tolist()
            record["seam_frequencies_hz"] = [
                float(axis[index])
                for index in spectra.seam_bins(layout.nperseg, layout.raw_per_step)
            ]

        kept = self._pooled.get(f"{SPECTRA_POOLED_KEY}_n_windows")
        possible = self._pooled.get(f"{SPECTRA_POOLED_KEY}_n_windows_possible")
        if kept is not None and possible is not None:
            total_kept = float(np.asarray(kept).sum())
            total_possible = float(np.asarray(possible).sum())
            record["n_windows_kept"] = total_kept
            record["n_windows_possible"] = total_possible
            record["window_drop_fraction"] = (
                1.0 - total_kept / total_possible if total_possible > 0.0 else float("nan")
            )
            # Per lead time as well: a gap costs the same windows at every $\tau$, so a profile
            # that varies with $\tau$ means the slice construction is misaligned.
            record["n_windows_kept_per_lead"] = np.asarray(kept).tolist()
        return record


def geometry_record(geometry: Any) -> Dict[str, int]:
    r"""Return the anchor-to-raw geometry as plain numbers, for the analyses that read it offline.

    Recorded rather than re-derived from the configuration. The shaded warm-up prefix on an
    anchor profile, the seconds axis of a horizon curve and the raw-sample denominators all come
    from here, and every one of them is a statement about the model that was evaluated -- so it is
    read from that model once, in the only pass that has it, instead of being rebuilt from a
    config file by each consumer.

    Args:
        geometry: The model's :class:`~teb_vae.lag_attn_rws.nets.geometry.TrimmedRawGeometry`.

    Returns:
        The four constructor fields and the three derived counts.
    """
    return {
        "raw_len": int(geometry.raw_len),
        "decimation": int(geometry.decimation),
        "horizon": int(geometry.horizon),
        "warmup": int(geometry.warmup),
        "t": int(geometry.t),
        "t_valid": int(geometry.t_valid),
        "raw_per_step": int(geometry.r),
    }


def bounds_record(model: Any) -> Dict[str, Any]:
    r"""Return the model's own bound conventions, for the analyses that read them offline.

    A clamp is a property of the checkpoint, not of a config file: nothing reconciles
    ``logvar_clamp`` against the checkpoint's ``model_kwargs``, so a summary that read it from the
    merged configuration would be quoting whatever that file currently says. Read from the model
    once, in the only pass that has one, and every downstream margin -- the two lines on the
    log-variance figure, the bins its histogram was laid out over -- resolves to the same numbers
    the readouts were computed against.

    Args:
        model: The loaded :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.

    Returns:
        The log-variance clamp, the margin fraction that decides "on the clamp" and the absolute
        margin it works out to, and the two tanh bounds the saturation fractions are measured
        against.
    """
    lo, hi = float(model.logvar_clamp[0]), float(model.logvar_clamp[1])
    return {
        "logvar_clamp": [lo, hi],
        "logvar_margin_frac": float(LOGVAR_FLOOR_MARGIN_FRAC),
        "logvar_margin": float(LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)),
        "mu_scale": float(model.mu_scale),
        "delta_mu_scale": float(model.delta_mu_scale),
        "saturation_frac": float(SATURATION_FRAC),
    }


#: The raw signals whose z-scoring is recorded: the target, so a score can be reported in bpm, and
#: the source, so the per-sample pages draw the UP trace in mmHg. Both are entries the *figures*
#: need; nothing in this file reads either.
_NORMALIZED_SIGNALS = ("fhr", "up")


def normalization_record(loader: Any) -> Dict[str, Any]:
    """Return the loader's raw-signal z-scoring, so a later analysis needs no loader.

    Only the two raw signals, and only their two scalars each. The full statistics dict carries
    per-channel arrays for seven fields and torch tensors beside them, none of which any analysis
    reads and none of which survives a JSON round trip.

    ``up`` is here for one reason: the per-sample page draws the source trace beside the target,
    and the training callback -- which reaches the loader's statistics directly -- draws it in
    mmHg. Without this entry the same builder would render the same segment in z-units on the
    evaluation's copy of the page, and the two would silently disagree.

    Args:
        loader: The evaluation dataloader.

    Returns:
        ``{field: {'mean': ..., 'std': ...}}`` for each raw signal the dataset reports, or ``{}``
        when it reports none. A partial record is the honest outcome rather than a failure: an
        evaluation without a field's statistics still produces every number, in the loader's own
        units and labelled as such.
    """
    dataset = getattr(loader, "dataset", None)
    getter = getattr(dataset, "get_normalization_stats", None)
    stats = getter() if callable(getter) else None
    record: Dict[str, Any] = {}
    for field in _NORMALIZED_SIGNALS:
        block = (stats or {}).get(field)
        if isinstance(block, dict) and "mean" in block and "std" in block:
            record[field] = {"mean": float(block["mean"]), "std": float(block["std"])}
    return record


def _counts_of(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    """Return a ``{value: count}`` mapping for one column, skipping unlabelled rows."""
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].dropna().value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def _concatenate_blocks(blocks: Sequence[Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Concatenate per-batch column blocks into one frame, in key order.

    Args:
        blocks: Per-batch mappings from column name to array, all carrying the same names.

    Returns:
        The frame, empty with the key columns present when there were no blocks -- an empty table
        that a consumer can still ``groupby`` is friendlier than one with no schema at all.
    """
    if not blocks:
        return pd.DataFrame({name: [] for name in PER_ANCHOR_KEY})
    names = list(blocks[0])
    ordered = list(PER_ANCHOR_KEY) + [name for name in names if name not in PER_ANCHOR_KEY]
    return pd.DataFrame(
        {name: np.concatenate([block[name] for block in blocks], axis=0) for name in ordered}
    )


def check_per_anchor_key(per_anchor: pd.DataFrame) -> None:
    """Raise when the per-anchor table's key is not unique.

    Args:
        per_anchor: The assembled table.

    Raises:
        ValueError: Naming the duplicated keys. One recording's segments are disjoint in time, so
            two rows sharing ``(guid, epoch, anchor)`` means either two segments of one recording
            carry the same ``epoch``, or the batches carried no ``epoch`` at all -- and every join
            and every ``groupby`` built on the key would then silently double-count.
    """
    if per_anchor.empty:
        return
    duplicated = per_anchor.duplicated(subset=list(PER_ANCHOR_KEY), keep=False)
    if not bool(duplicated.any()):
        return
    offending = per_anchor.loc[duplicated, list(PER_ANCHOR_KEY)].head(5)
    raise ValueError(
        f"the per-anchor key {list(PER_ANCHOR_KEY)} is not unique: {int(duplicated.sum())} row(s) "
        f"collide, e.g.\n{offending.to_string(index=False)}\nThe key assumes each segment of a "
        f"recording carries its own epoch; a batch reaching this table without an epoch field "
        f"produces exactly this collision, and preflight requires the field for that reason."
    )


def collect_tables(
    task: Any,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    num_samples: int,
    n_total: Optional[int] = None,
    max_batches: Optional[int] = None,
    perm_generator: Optional[torch.Generator] = None,
    mc_generator: Optional[torch.Generator] = None,
    delay_steps: int = 0,
) -> Collection:
    """Run the one shared forward pass and assemble both tables.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader.
        eval_config: The validated ``eval_config`` block, for the caps, the seed and the two
            verdict thresholds.
        num_samples: Monte Carlo draws $K$.
        n_total: Samples the pass will see, for the retention draw. Read from the loader's
            dataset when omitted.
        max_batches: Stop after this many scored batches, for a smoke run.
        perm_generator: Generator seeding the derangements.
        mc_generator: Generator for the Monte Carlo $\\epsilon$.
        delay_steps: The causal input delay, for the lag report.

    Returns:
        The collection, with ``results`` carrying what :func:`metrics.evaluate` reports.
    """
    plan = RetentionPlan.build(
        eval_config.get("caps"),
        n_total=_loader_length(loader) if n_total is None else int(n_total),
        seed=int(eval_config.get("seed", 0)),
    )
    collector = _Collector(plan, geometry=task.orig_model.geometry)
    results = evaluate(
        task,
        loader,
        num_samples=num_samples,
        max_batches=max_batches,
        perm_generator=perm_generator,
        mc_generator=mc_generator,
        delay_steps=delay_steps,
        prior_shuffle_min_nats=float(eval_config["prior_shuffle_min_nats"]),
        min_active_dims=int(eval_config["min_active_dims"]),
        retain=plan.tensor_names(),
        on_batch=collector.observe,
    )

    collection = collector.finish()
    collection.results = results
    # Two facts about *this* pass that no later analysis can recover for itself, because both
    # belong to things it deliberately does not hold: the model's anchor-to-raw geometry, and the
    # loader's FHR z-scoring. Without them an offline re-run cannot shade a warm-up prefix or say
    # a number in bpm.
    collection.record["geometry"] = geometry_record(task.orig_model.geometry)
    collection.record["normalization"] = normalization_record(loader)
    collection.record["bounds"] = bounds_record(task.orig_model)
    collection.record["likelihood"] = str(results.get("likelihood", ""))
    check_per_anchor_key(collection.per_anchor)
    if len(collection.per_sample) != int(results["n_samples"]):
        raise ValueError(
            f"the per-sample table holds {len(collection.per_sample)} row(s) but the pass scored "
            f"{results['n_samples']} sample(s). One row per scored segment is what makes every "
            f"table-driven analysis agree with the headline readouts."
        )
    logger.info(
        f"collected {len(collection.per_sample)} sample row(s) and "
        f"{len(collection.per_anchor)} anchor row(s) from "
        f"{collection.record['n_recordings']} recording(s); "
        f"{collection.record['n_segments_excluded_zero_anchors']} segment(s) scored no anchors"
    )
    return collection


def _loader_length(loader: Any) -> Optional[int]:
    """Return how many samples a loader will yield, or ``None`` when it cannot say.

    Args:
        loader: The evaluation dataloader, or any iterable standing in for one.

    Returns:
        The dataset length, or ``None`` -- which is only a problem when a cap needs it.
    """
    dataset = getattr(loader, "dataset", None)
    try:
        return None if dataset is None else int(len(dataset))
    except TypeError:
        return None


# =============================================================================
# Provenance, writing and reading back
# =============================================================================
def file_digest(path: Any, *, chunk_bytes: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks.

    Args:
        path: The file to hash.
        chunk_bytes: Read size, so a multi-gigabyte checkpoint is not held in memory.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(eval_config: Dict[str, Any]) -> str:
    """Return a digest of the resolved ``eval_config`` block.

    Sorted keys and a canonical separator, so the digest depends on the settings rather than on
    the order a merge happened to produce.

    Args:
        eval_config: The validated block.

    Returns:
        The hex digest.
    """
    canonical = json.dumps(eval_config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance(
    *, checkpoint_path: Optional[Any], eval_config: Dict[str, Any], num_samples: int
) -> Dict[str, Any]:
    """Return the identity of the run whose tables these are.

    Args:
        checkpoint_path: The evaluated checkpoint, or ``None`` for a pass that had none.
        eval_config: The validated ``eval_config`` block.
        num_samples: The Monte Carlo draw count actually used, which may differ from the
            configured one on a smoke run and is what the numbers depend on.

    Returns:
        The provenance record.
    """
    checkpoint: Dict[str, Any] = {"path": None, "sha256": None}
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": file_digest(checkpoint_path),
        }
    return {
        "checkpoint": checkpoint,
        "seed": int(eval_config.get("seed", 0)),
        "num_mc_samples": int(num_samples),
        "eval_config_digest": config_digest(eval_config),
    }


def write_collection(collection: Collection, results_dir: Any) -> Path:
    """Write both tables, both sidecars and the provenance record.

    Args:
        collection: What the pass produced.
        results_dir: The run's results directory.

    Returns:
        The path of the provenance record.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    collection.per_sample.to_csv(results_dir / PER_SAMPLE_FILENAME, index=False)
    # Parquet only: pyarrow is pinned in requirements.txt and present in the environment, so a
    # format-negotiation branch would be an untested second path for no gain.
    collection.per_anchor.to_parquet(results_dir / PER_ANCHOR_FILENAME, index=False)
    np.savez_compressed(results_dir / VECTORS_FILENAME, **collection.vectors)
    if collection.retained:
        np.savez_compressed(results_dir / RETAINED_FILENAME, **collection.retained)
    if collection.spectra:
        np.savez_compressed(results_dir / COHERENCE_FILENAME, **collection.spectra)

    path = results_dir / COLLECTION_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(collection.record), handle, indent=2, allow_nan=False)
    logger.info(f"wrote {path}")
    return path


def record_summary_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the provenance record without what ``summary.json`` already carries.

    ``results`` lives in the record so that a directory whose forward pass is skipped still
    answers every question the pass answered; the summary carries the same block at its top
    level, and repeating it there would double the largest object in the file.

    Args:
        record: The provenance record.

    Returns:
        The subset the summary embeds.
    """
    return {name: value for name, value in record.items() if name != "results"}


def has_collection(results_dir: Any) -> bool:
    """Whether a finished run's tables are already in this directory.

    Asked *before* the loader is built rather than inferred from a failed read: a run reusing a
    finished directory must not construct a dataloader or run a probe pass over shards it is not
    going to score.

    Args:
        results_dir: The run's results directory.

    Returns:
        ``True`` when the provenance record is present. The tables themselves are checked against
        it by :func:`load_collection`, which is where a truncated one is refused.
    """
    return (Path(results_dir) / COLLECTION_FILENAME).is_file()


def read_record(results_dir: Any) -> Optional[Dict[str, Any]]:
    """Read the provenance sidecar without reading the tables it describes.

    A pass with no model has nothing to collect *with*, so what the tables were collected under is
    a fact to be adopted rather than a setting to be compared against -- and adopting it needs the
    record before the far more expensive read of the tables themselves.

    Args:
        results_dir: The run's results directory.

    Returns:
        The record, or ``None`` when the directory holds none.
    """
    path = Path(results_dir) / COLLECTION_FILENAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_collection(results_dir: Any) -> Collection:
    """Read a finished run's tables back.

    Args:
        results_dir: The run's results directory.

    Returns:
        The collection, with ``from_cache`` set.

    Raises:
        FileNotFoundError: If either table or the provenance record is absent.
        TablesProvenanceMismatch: If a table is shorter than the record says -- a run killed
            mid-write leaves exactly that, and it reads as a smaller population rather than as a
            broken file.
    """
    results_dir = Path(results_dir)
    record_path = results_dir / COLLECTION_FILENAME
    if not record_path.is_file():
        raise FileNotFoundError(f"no {COLLECTION_FILENAME} in {results_dir}")
    with open(record_path, encoding="utf-8") as handle:
        record = json.load(handle)

    per_sample = pd.read_csv(
        results_dir / PER_SAMPLE_FILENAME, float_precision=PER_SAMPLE_FLOAT_PRECISION
    )
    per_anchor = pd.read_parquet(results_dir / PER_ANCHOR_FILENAME)
    vectors: Dict[str, np.ndarray] = {}
    vectors_path = results_dir / VECTORS_FILENAME
    if vectors_path.is_file():
        with np.load(vectors_path) as handle:
            vectors = {name: handle[name] for name in handle.files}
    retained: Dict[str, np.ndarray] = {}
    retained_path = results_dir / RETAINED_FILENAME
    if retained_path.is_file():
        with np.load(retained_path) as handle:
            retained = {name: handle[name] for name in handle.files}
    # Absent on a directory collected before the coherence analysis existed, which the analysis
    # reports as a skip naming the re-collection rather than as an empty result.
    pooled_spectra: Dict[str, np.ndarray] = {}
    coherence_path = results_dir / COHERENCE_FILENAME
    if coherence_path.is_file():
        with np.load(coherence_path) as handle:
            pooled_spectra = {name: handle[name] for name in handle.files}

    _check_row_counts(record, per_sample, per_anchor, results_dir)
    return Collection(
        per_sample=per_sample,
        per_anchor=per_anchor,
        vectors=vectors,
        retained=retained,
        spectra=pooled_spectra,
        results=dict(record.get("results") or {}),
        record=record,
        from_cache=True,
    )


def _check_row_counts(
    record: Dict[str, Any], per_sample: pd.DataFrame, per_anchor: pd.DataFrame, results_dir: Path
) -> None:
    """Raise when a table on disk is not the length its record claims.

    Args:
        record: The provenance record.
        per_sample: The per-sample table as read.
        per_anchor: The per-anchor table as read.
        results_dir: The directory, for the message.

    Raises:
        TablesProvenanceMismatch: Naming the table, the recorded count and the observed one.
    """
    for name, frame, key in (
        (PER_SAMPLE_FILENAME, per_sample, "n_per_sample_rows"),
        (PER_ANCHOR_FILENAME, per_anchor, "n_per_anchor_rows"),
    ):
        expected = record.get(key)
        if expected is not None and int(expected) != len(frame):
            raise TablesProvenanceMismatch(
                f"{results_dir / name} holds {len(frame)} row(s) but {COLLECTION_FILENAME} "
                f"records {int(expected)}. The table is truncated -- a run killed mid-write "
                f"leaves this -- and reading it would report a silently narrower population."
            )


def check_provenance(record: Dict[str, Any], expected: Dict[str, Any]) -> None:
    """Raise when a finished run's tables describe a different run.

    Args:
        record: The provenance record read off disk.
        expected: What :func:`provenance` says about the run being asked for.

    Raises:
        TablesProvenanceMismatch: Naming both values of whatever disagrees. Refusing rather than
            re-collecting on top: a directory holding two checkpoints' rows under one summary is
            not something a reader can unpick afterwards.
    """
    found = record.get("provenance") or {}
    checkpoint_found = (found.get("checkpoint") or {}).get("sha256")
    checkpoint_expected = (expected.get("checkpoint") or {}).get("sha256")
    # Only when both are known: a table-only re-run legitimately has no checkpoint to compare.
    if checkpoint_found and checkpoint_expected and checkpoint_found != checkpoint_expected:
        raise TablesProvenanceMismatch(
            f"the tables in this directory were collected from checkpoint "
            f"{(found.get('checkpoint') or {}).get('path')!r} (sha256 {checkpoint_found[:12]}), "
            f"not from {(expected.get('checkpoint') or {}).get('path')!r} (sha256 "
            f"{checkpoint_expected[:12]}). Evaluate into a fresh output directory."
        )
    for key, label in (
        ("seed", "eval_config.seed"),
        ("num_mc_samples", "the Monte Carlo draw count"),
        ("eval_config_digest", "the eval_config block"),
    ):
        if key in found and key in expected and found[key] != expected[key]:
            raise TablesProvenanceMismatch(
                f"the tables in this directory were collected with {label} = {found[key]!r}, "
                f"not {expected[key]!r}. Every number in them depends on it, so they cannot be "
                f"reused for this run; evaluate into a fresh output directory."
            )


def load_or_collect(
    results_dir: Any,
    collect: Callable[[], Collection],
    *,
    checkpoint_path: Optional[Any],
    eval_config: Dict[str, Any],
    num_samples: int,
) -> Collection:
    """Reuse a finished run's tables when they describe this run, otherwise collect them.

    This is what makes the offline promise real: the forward pass is the expensive part, and an
    analysis re-run against a finished directory must not pay it again. The refusals are what
    makes it safe -- reuse is only ever silent when it is also correct.

    Args:
        results_dir: The run's results directory.
        collect: Called to run the pass when there is nothing valid to reuse.
        checkpoint_path: The checkpoint being evaluated, or ``None``.
        eval_config: The validated ``eval_config`` block.
        num_samples: The Monte Carlo draw count in force.

    Returns:
        The collection, either read back or freshly collected and written.

    Raises:
        TablesProvenanceMismatch: If tables exist but belong to another run, or are truncated.
    """
    results_dir = Path(results_dir)
    expected = provenance(
        checkpoint_path=checkpoint_path, eval_config=eval_config, num_samples=num_samples
    )
    if (results_dir / COLLECTION_FILENAME).is_file():
        collection = load_collection(results_dir)
        check_provenance(collection.record, expected)
        # Beside the provenance check because it is the same question asked of the other half of
        # the record: that one says the tables describe *this* run, this one says they were
        # written under *this* set of acceptance criteria. Both are refusals rather than repairs,
        # and both belong here rather than at the point of use -- reuse is the only path on which
        # either can be wrong.
        check_cached_verdicts((collection.results or {}).get("verdicts"))
        logger.info(
            f"reusing {len(collection.per_sample)} collected row(s) from {results_dir}; "
            f"the forward pass is skipped"
        )
        return collection

    collection = collect()
    collection.record["provenance"] = expected
    # The readouts travel with the tables rather than only into summary.json, so a directory the
    # forward pass is skipped for still answers every question the pass answered.
    collection.record["results"] = collection.results
    write_collection(collection, results_dir)
    return collection
