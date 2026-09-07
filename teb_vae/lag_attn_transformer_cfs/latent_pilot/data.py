r"""Cohort, labels, anchor timestamps, support masks and recording bags.

Both halves are written: the cohort side (LP-03) -- loader assembly, the one identifying pass
over a split, recording-level label consolidation, split and grouping checks, exposure provenance,
coverage and the two serialised tables -- and the temporal side (LP-04) -- anchor timestamps,
support masks, window and bin boundaries, deduplication, late eligibility and the bag reductions.

This module records which existing symbols are reused, and -- more importantly -- the three places
where the existing pipeline answers a *different* question and this package must add its own path
rather than repoint one of theirs.

Reused, by import, never by copy
--------------------------------

``teb_vae.lag_attn.eval.labels``
    ``clinical_class_code(target_row, weight_row)`` recovers the class code as the modal
    $\mathrm{round}(\mathrm{target}_t / \mathrm{weight}_t)$ over steps with $\mathrm{weight}_t > 0$,
    and returns ``None`` for a pad-only window or a uniformly zero target. That last case is why the
    stored ``target`` must never be averaged into a class: it is the code scaled by per-step
    validity, so a partially valid acidosis step is numerically indistinguishable from a fully valid
    healthy one, and a zero means *no class* rather than healthy. ``CLASS_NAMES``
    (``1: healthy, 2: acidosis, 3: hie``), ``class_name``, ``subgroup_of``, ``CANONICAL_SUBGROUPS``
    (the eight ``healthy_{no_,}bg_{no_,}cs`` / ``acidosis_{no_,}cs`` / ``hie_{no_,}cs`` shard names),
    ``CLASS_COLUMN`` and ``SUBGROUP_COLUMN`` supply the rest of the cohort identity. This package
    adds one thing on top: a **recording-level consistency check**, because a per-sample modal code
    says nothing about whether a GUID's segments agree with each other.
``hdf5_dataset.hdf5_dataset``
    ``CombinedHDF5Dataset`` and ``create_optimized_dataloader`` carry the filters this pilot needs
    -- ``allowed_guids``, ``cs_label``, ``bg_label``, ``epoch_min`` / ``epoch_max``, ``label``,
    ``load_fields``, ``trim_minutes``. ``decimated_trim_steps(trim_minutes)`` returns the
    ``(raw, decimated)`` samples discarded from **each** end, and is the only correct source of that
    conversion. ``RAW_SAMPLING_HZ = 4`` with ``DECIMATION = 16`` fixes the decimated step at four
    seconds, which ``teb_vae.lag_attn.nets.lag_report.SECONDS_PER_STEP`` states as $4.0$; the pilot
    reads the contract rather than restating the number.
``train.data_module.GraphDataModule``
    Builds the split loaders from a resolved ``dataset_config``. The pilot points its three splits
    at the fold-1 shard lists and keeps the checkpoint's ``stat_path``, ``normalize_fields`` and
    ``trim_minutes`` exactly as the run trained under them.
``teb_vae.lag_attn_rws.nets.raw_masks``
    ``forecast_mask(weight, geometry, coverage_floor=, anchors=, anchor_valid=)`` returns the
    $(B, A, H)$ validity mask and the per-anchor coverage; ``contributing_anchors(mask)`` reduces it
    to the $(B, A)$ indicator of anchors that carry a reconstruction term at all; ``kl_mask``
    scatters that same support back to $(B, T)$. The pilot conservatively adopts
    *forecast-contributing support* as its anchor policy, which is what makes the latent analysis
    inherit the forecast's availability exclusions -- a limitation recorded in the run's coverage
    table rather than argued away.
``teb_vae.lag_attn_cfs.eval.collect``
    ``PER_ANCHOR_KEY = ("guid", "epoch", "anchor")`` and ``check_per_anchor_key`` define the
    per-anchor identity every join in this package is built on.
``teb_vae.lag_attn_cfs.eval.cohort``
    ``cohort_counts`` and the class/subgroup ordering are reused for reporting.

Where the existing pipeline answers a different question
--------------------------------------------------------

*The time axis.* ``cohort.add_time_bins`` and ``cohort.within_horizon`` bin on the **segment start**:
$h = -\mathrm{epoch}/3600$. The loader trims the signals but leaves ``epoch`` as the *untrimmed*
segment start in seconds relative to delivery, so a segment-start bin is not the time an anchor
observes. This package computes the anchor's own timestamp instead,

$$t_{isa} = \mathrm{epoch}_{is} + \Delta \cdot k(m) + \Delta \cdot a,
\qquad r_{isa} = -t_{isa} / 3600,$$

with $\Delta = 4$ s from the decimation contract above and $k(m)$ the **decimated** steps the
loader's trim discards, through ``decimated_trim_steps`` rather than as $60m$. The two agree
exactly at the shipped ``trim_minutes: 1.0`` -- $240$ raw samples, $15$ steps, $60$ s -- and
diverge only where the trim is not a whole number of steps, where the loader's own conversion is
by definition the right one. The worked case the tests pin: $\mathrm{epoch} = -3600$, $m = 1$,
$a = 150$ gives $-2940$ s, i.e. **49 minutes** before delivery, not 60. This lands in a new column
beside the existing ones; the established segment-start reports keep their meaning.

*The coarse filter.* Anchors are kept for $0 < r \le 3$ h, but a segment that *crosses* the
three-hour boundary carries eligible anchors and must not be dropped by a loader filter set at
``epoch_min = -10800``. The coarse filter is widened by a whole stored segment -- derived from the
checkpoint's ``sequence_length`` and the loader's trim rather than from the 22 minutes the current
shards happen to hold -- and the exact anchor filter runs afterwards.

*Post-delivery endpoints.* $\mathrm{epoch} < 0$ does not prove that a segment, or a scored forecast
window, lies wholly before delivery. Each scored coefficient endpoint is checked per channel,
including the checkpoint's ``target_forecast_shift``, and the exclusion reason is serialized. The
stored UP timeline is canonical: no downstream timing correction is applied anywhere in this package.

Deduplication, bags and bins
----------------------------

``(guid, epoch, anchor)`` is deduplicated, and duplicate *absolute* anchor times within a GUID are
inspected as well: overlapping segments can supply the same instant twice, and the survivor is
chosen deterministically -- greater valid history first, then a fixed segment order. They are never
counted as independent observations. Warm-up and stride leave real gaps between segment readouts,
which the trajectory outputs mark rather than interpolate across.

The supervised bag is the final hour, $0 < r \le 1$: anchors are averaged within a segment,

$$e_{is} = \frac{1}{|A^{\rm late}_{is}|} \sum_{a \in A^{\rm late}_{is}} \mu^q_{isa},$$

and segments are pooled into one recording vector with a 30-minute half-life inside that hour,

$$\omega_{is} = 2^{-\widetilde r_{is} / 0.5}, \qquad
v_i = \frac{\sum_s \omega_{is} e_{is}}{\sum_s \omega_{is}},$$

where $\widetilde r_{is}$ is the median time-before-delivery of that segment's included anchors.
Averaging within segments first is what stops a densely sampled segment from dominating, and the
weighting is identical for both classes -- it is a recency preference, not a severity target. Each
recording supplies exactly one supervised loss.

Eligibility for that bag, fixed before any outcome comparison and applied identically to both
classes: at least two contributing segments within the final hour, and a valid anchor inside the
final 30 minutes. Exclusions are counted and reported. The trajectory axis uses six fixed bins --
$(2.5, 3]$, $(2, 2.5]$, $(1.5, 2]$, $(1, 1.5]$, $(0.5, 1]$, $(0, 0.5]$ hours -- averaged anchors
within segments, then segments within recordings. A recording without late eligibility may appear in
a separately labelled coverage appendix and never enters the paired primary comparison.

Nothing here derives a severity score, interpolates a missing time, or imposes a monotonic
progression: the data supply a recording's outcome $y_i$, not its state $s_i(t)$, and anchors one to
three hours before delivery receive preservation supervision only -- they are not relabelled
healthy either. Clinical metadata (outcome, CS status, blood-gas status, delivery time) selects and
describes windows; none of it is ever a model or classifier input.
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_cfs.eval.config_schema import force_single_process_loader
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

# =============================================================================
# What the pilot loads, and what the model is allowed to see
# =============================================================================
#: Fields the pilot loads **beyond** the training contract, and the one consumer each has here.
#:
#: The training configuration does not load ``target``, so a run built from it alone has no class
#: code at all -- which is the gap this list closes. ``target`` is the code scaled by per-step
#: validity and is read only through :func:`teb_vae.lag_attn.eval.labels.clinical_class_code`;
#: ``cs_label`` and ``bg_label`` are the descriptive strata; ``time_from_labor_onset`` and
#: ``second_stage_onset`` are recorded for the coverage table and are NaN wherever the recording is
#: absent from the labour-onset table, which is a fact about the cohort rather than a defect.
#:
#: **None of these ever reaches the model.** The forward takes the three coefficient streams named
#: in :data:`MODEL_INPUT_FIELDS` and nothing else; these ride on the batch as identity and are used
#: to select, describe and group recordings. Feeding delivery time, outcome, CS or blood-gas status
#: to the encoder or the classifier would make the whole comparison circular.
CLINICAL_FIELDS: Tuple[str, ...] = (
    "target",
    "cs_label",
    "bg_label",
    "time_from_labor_onset",
    "second_stage_onset",
)

#: The coefficient streams the forward consumes, in the order it takes them.
MODEL_INPUT_FIELDS: Tuple[str, ...] = ("fhr_st", "fhr_ph", "up_st", "up_ph")

#: Fields the checkpoint's own loader contract must already carry. ``weight`` gates every mask,
#: ``guid`` and ``epoch`` key every aggregation and the anchor tiling's phase, and the four streams
#: are the inputs. A configuration missing one of them is refused rather than silently repaired:
#: the loader skips an unknown field without a word, so the failure would otherwise appear as an
#: empty tensor much later.
REQUIRED_LOAD_FIELDS: Tuple[str, ...] = MODEL_INPUT_FIELDS + ("weight", "guid", "epoch")

# =============================================================================
# Table schema
# =============================================================================
SPLIT_COLUMN = "split"
GUID_COLUMN = "guid"
EPOCH_COLUMN = "epoch"
SAMPLE_INDEX_COLUMN = "sample_index"
SOURCE_FILE_COLUMN = "source_file_basename"
CLASS_CODE_COLUMN = "class_code"
#: The binary outcome the pilot supervises: healthy 0, acidosis or HIE 1.
OUTCOME_COLUMN = "outcome"
PATIENT_COLUMN = "patient"
EXCLUSION_COLUMN = "exclusion_reason"

#: The three fixed splits, in the order they are reported.
SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: Stored class code -> binary outcome. Acidosis and HIE pool into one positive class, and the two
#: keep their own subgroup label for reporting. The codes are categories: there is deliberately no
#: mapping to an ordinal severity scale, and no regression target 1 < 2 < 3 anywhere in this
#: package.
BINARY_OUTCOME: Dict[int, int] = {1: 0, 2: 1, 3: 1}

#: Why a recording is excluded from the supervised comparison. LP-04 adds the coverage reasons; the
#: ones here are all about identity and labelling, and every one of them is counted and reported
#: rather than silently dropped.
EXCLUDED_NO_CLASS = "no_valid_class_code"
EXCLUDED_CONFLICTING_CLASS = "conflicting_class_codes"
EXCLUDED_UNKNOWN_CLASS = "unknown_class_code"
EXCLUDED_CONFLICTING_METADATA = "conflicting_recording_metadata"

#: Serialised outputs of this stage.
MANIFEST_FILENAME = "split_manifest.csv"
COVERAGE_FILENAME = "coverage.csv"

#: The two cohort tables as the stages hand them to each other. Parquet beside the human-readable
#: CSV manifest rather than instead of it: a stage that read the CSV back would have to re-infer
#: every dtype the writer flattened -- a missing outcome becomes a float ``nan``, a boolean flag
#: becomes the string ``'True'`` -- and one stage disagreeing with another about whether a recording
#: has an outcome is exactly the failure the cohort code exists to prevent.
RECORDINGS_FILENAME = "recordings.parquet"
SEGMENTS_FILENAME = "segments.parquet"


# =============================================================================
# Loader assembly
# =============================================================================
def pilot_loader_config(
    resolved_config: Mapping[str, Any],
    *,
    shards: Sequence[str],
    statistics: str,
    epoch_min: Optional[float] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one split's loader configuration from the checkpoint's own resolved config.

    The checkpoint's configuration is the authority and is copied rather than rebuilt: the trim,
    the normalisation fields, the channel contract and the geometry all stay exactly as the run
    trained under them. Four things change, and nothing else does -- the shard lists, the
    statistics file, the loaded-field list, and the coarse epoch filter.

    Both shard lists are pointed at the same split on purpose. Only one of them is read here, but a
    configuration that still carried the pretraining list in the other would be one keystroke away
    from fitting on it.

    Args:
        resolved_config: The configuration written beside the checkpoint, as a mapping. Not
            mutated: the returned config shares no nested object with it.
        shards: This split's HDF5 shards.
        statistics: The statistics file the checkpoint was trained under.
        epoch_min: Coarse lower bound on the segment start, in seconds relative to delivery.
            ``None`` keeps the configuration's own value. The exact anchor-time filter runs
            afterwards, so this bound only has to be generous enough not to drop a segment that
            crosses the window boundary.
        batch_size: Loader batch size. ``None`` keeps the configuration's own.

    Returns:
        The loader configuration, single-process and with the clinical identity fields added.

    Raises:
        PilotConfigError: If the checkpoint's configuration does not load one of
            :data:`REQUIRED_LOAD_FIELDS`. The loader skips a field a shard does not carry without a
            word, so an absent one surfaces much later as a missing tensor.
    """
    config = deepcopy(dict(resolved_config))
    dataset = config.setdefault("dataset_config", {})
    dataloader = dataset.setdefault("dataloader_config", {})
    kwargs = dataloader.setdefault("dataset_kwargs", {})

    load_fields = list(kwargs.get("load_fields") or [])
    missing = [name for name in REQUIRED_LOAD_FIELDS if name not in load_fields]
    if missing:
        raise PilotConfigError(
            f"the checkpoint's resolved config does not load {missing}, so the pilot cannot build "
            f"its inputs, its masks or its per-segment identity from it. This is a property of the "
            f"run that produced the checkpoint, not something a pilot setting may paper over."
        )
    # Appended rather than replaced, so the run's own contract survives intact and the addition is
    # visibly an addition. Order-preserving and idempotent: re-resolving a config already extended
    # by this function returns the same list.
    for name in CLINICAL_FIELDS:
        if name not in load_fields:
            load_fields.append(name)
    kwargs["load_fields"] = load_fields

    dataset["vae_test_datasets"] = list(shards)
    dataset["vae_train_datasets"] = list(shards)
    dataset["stat_path"] = str(statistics)

    # Explicitly off. The loader's own ``label`` filter tests exact float equality against the
    # weight-scaled target, so it matches only fully valid steps and silently drops every
    # partially masked segment; the class is recovered from the target/weight ratio instead.
    kwargs["label"] = None
    if epoch_min is not None:
        kwargs["epoch_min"] = float(epoch_min)
    if batch_size is not None:
        config.setdefault("general_config", {}).setdefault("batch_size", {})["test"] = int(
            batch_size
        )
    # Spawned workers over a multi-file HDF5 dataset silently truncate every pass after the first,
    # which on this pipeline means an extraction that quietly covers part of the split.
    force_single_process_loader(config)
    return config


# =============================================================================
# One pass over a split
# =============================================================================
def _field(batch: Any, name: str) -> Any:
    """Read a batch field by name, tolerating both mapping and attribute access.

    A local copy of the evaluation pipeline's reader rather than an import of it: that module
    reaches the model class through its binding, and this one must stay importable by the synthetic
    logic tests, which build no model and have no GPU.

    Args:
        batch: A batch from the data module, or a stub.
        name: The field name.

    Returns:
        The value, or ``None`` when the batch does not carry it.
    """
    if isinstance(batch, Mapping):
        return batch.get(name)
    return getattr(batch, name, None)


def _to_numpy(values: Any) -> np.ndarray:
    """Return ``values`` as a float64 array, accepting a tensor, an array or a sequence."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _strings_of(batch: Any, name: str, batch_size: int) -> List[Optional[str]]:
    """Return one string per sample for a field that survives collation as a list."""
    values = _field(batch, name)
    if values is None:
        return [None] * batch_size
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return [str(values)] * batch_size


def _batch_size(batch: Any) -> int:
    """Return how many samples one batch holds.

    Read from a batched tensor rather than a collated list, so the count does not depend on which
    optional identifier fields a shard happened to carry.

    Args:
        batch: A batch from the data module.

    Returns:
        The sample count.

    Raises:
        PilotConfigError: If nothing in the batch has a readable leading dimension.
    """
    for name in ("weight", "target", "fhr_st", "up_st"):
        value = _field(batch, name)
        if value is not None and getattr(value, "ndim", 0) >= 1:
            return int(value.shape[0])
    for name in ("guid", SOURCE_FILE_COLUMN):
        value = _field(batch, name)
        if isinstance(value, (list, tuple)):
            return len(value)
    raise PilotConfigError(
        "cannot determine the batch size: the batch carries none of weight, target, fhr_st, "
        "up_st, guid or source_file_basename."
    )


def _optional_bool(value: Any) -> Optional[bool]:
    """Return a stored flag as a boolean, or ``None`` when it is absent or not finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return bool(number)


def segment_frame(loader: Any, *, split: str, max_batches: Optional[int] = None) -> pd.DataFrame:
    """Iterate one split once and record what identifies each segment.

    No forward pass and no model: this is the pass that establishes which recordings exist, what
    they are labelled, and where in time they sit, before anything is fitted against them.

    Args:
        loader: The split's dataloader.
        split: The split name, stored on every row.
        max_batches: Stop after this many batches. For a smoke run only -- the loader is
            unshuffled over concatenated per-subgroup shards, so a batch cap is a *prefix* and
            yields one subgroup and one class rather than a sample of the split.

    Returns:
        One row per segment, keyed by ``(guid, epoch)``, carrying the class code recovered from the
        target/weight ratio, the canonical subgroup, the CS and blood-gas flags and both clinical
        clocks. A segment whose class cannot be recovered keeps a null code and is counted, not
        dropped -- the exclusion belongs in the manifest.

    Raises:
        PilotConfigError: If the split yields no samples at all, which means the shard list
            resolves to nothing or a loader filter excluded everything.
    """
    rows: List[Dict[str, Any]] = []
    n_batches = 0
    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1
        size = _batch_size(batch)
        guids = _strings_of(batch, GUID_COLUMN, size)
        sources = _strings_of(batch, SOURCE_FILE_COLUMN, size)
        epochs = _to_numpy(_field(batch, EPOCH_COLUMN)) if _field(batch, EPOCH_COLUMN) is not None \
            else np.full(size, np.nan)
        target = _field(batch, "target")
        weight = _field(batch, "weight")
        targets = None if target is None else _to_numpy(target)
        weights = None if weight is None else _to_numpy(weight)
        cs = _field(batch, "cs_label")
        bg = _field(batch, "bg_label")
        onset = _field(batch, "time_from_labor_onset")
        stage = _field(batch, "second_stage_onset")
        cs_values = None if cs is None else _to_numpy(cs).ravel()
        bg_values = None if bg is None else _to_numpy(bg).ravel()
        onset_values = None if onset is None else _to_numpy(onset).ravel()
        stage_values = None if stage is None else _to_numpy(stage).ravel()

        for index in range(size):
            code: Optional[int] = None
            if targets is not None:
                target_row = np.atleast_1d(targets[index]).ravel()
                weight_row = (
                    np.atleast_1d(weights[index]).ravel()
                    if weights is not None and index < len(weights)
                    else np.ones_like(target_row)
                )
                # The ratio, never the raw target: a partially valid acidosis step stores 1.0 and
                # is otherwise indistinguishable from a fully valid healthy one, and a zero means
                # "no class" rather than healthy.
                code = labels.clinical_class_code(target_row, weight_row)
            rows.append({
                SPLIT_COLUMN: split,
                SAMPLE_INDEX_COLUMN: len(rows),
                GUID_COLUMN: guids[index],
                EPOCH_COLUMN: float(epochs[index]) if index < len(epochs) else float("nan"),
                SOURCE_FILE_COLUMN: sources[index],
                labels.SUBGROUP_COLUMN: labels.subgroup_of(sources[index]),
                CLASS_CODE_COLUMN: code,
                labels.CLASS_COLUMN: labels.class_name(code),
                "cs_label": _optional_bool(
                    cs_values[index] if cs_values is not None and index < len(cs_values) else None
                ),
                "bg_label": _optional_bool(
                    bg_values[index] if bg_values is not None and index < len(bg_values) else None
                ),
                "time_from_labor_onset": (
                    float(onset_values[index])
                    if onset_values is not None and index < len(onset_values) else float("nan")
                ),
                "second_stage_onset": (
                    float(stage_values[index])
                    if stage_values is not None and index < len(stage_values) else float("nan")
                ),
            })

    if not rows:
        raise PilotConfigError(
            f"the {split!r} loader yielded no samples. Either the shard list resolves to nothing, "
            f"or a loader filter (epoch_min / epoch_max / cs_label / bg_label / label) excluded "
            f"every segment. The resolved loader configuration is recorded with the run."
        )
    frame = pd.DataFrame(rows)
    logger.info(
        f"{split}: {len(frame)} segment(s) over {frame[GUID_COLUMN].nunique()} recording(s) from "
        f"{frame[SOURCE_FILE_COLUMN].nunique()} shard(s); classes "
        f"{dict(Counter(frame[labels.CLASS_COLUMN].tolist()))}"
    )
    return frame


# =============================================================================
# Recording-level identity
# =============================================================================
def binary_outcome(code: Optional[int]) -> Optional[int]:
    """Map a stored class code to the pilot's binary outcome.

    Args:
        code: The stored code, or ``None``.

    Returns:
        ``0`` for healthy, ``1`` for acidosis or HIE, and ``None`` for an absent or unrecognised
        code -- which is an exclusion to be counted, never a third class and never a zero.
    """
    if code is None:
        return None
    return BINARY_OUTCOME.get(int(code))


def _unique_or_none(values: Iterable[Any]) -> Tuple[Optional[Any], bool]:
    """Collapse a recording's per-segment values into one.

    Args:
        values: The values seen across a recording's segments, nulls included.

    Returns:
        ``(value, consistent)``. ``value`` is the single non-null value, or ``None`` when there was
        none; ``consistent`` is ``False`` when the segments disagreed, which is a conflict to be
        excluded rather than a majority to be taken.
    """
    seen = {
        value for value in values
        if value is not None and not (isinstance(value, float) and np.isnan(value))
    }
    if not seen:
        return None, True
    if len(seen) > 1:
        return None, False
    return seen.pop(), True


def recording_frame(segments: pd.DataFrame) -> pd.DataFrame:
    """Consolidate a segment table into one row per recording, with exclusions named.

    A per-segment class code says nothing about whether a recording's segments agree with each
    other, and the modal code inside one segment is not a consistency check across a GUID. This is
    where that check happens: a recording whose segments carry two different codes is **excluded**
    and counted, never resolved by majority vote. The same rule applies to the CS and blood-gas
    flags and to the subgroup, all of which are properties of a delivery rather than of a window.

    Args:
        segments: One or more splits' segment tables from :func:`segment_frame`.

    Returns:
        One row per GUID: its split, class code and binary outcome, subgroup, flags, segment count,
        the span of its segment starts, the last observed segment start, and an exclusion reason
        (empty for a usable recording). Eligibility columns are added later, by the temporal
        support code, and a recording excluded here never acquires them.
    """
    rows: List[Dict[str, Any]] = []
    for guid, group in segments.groupby(GUID_COLUMN, sort=True):
        codes = [code for code in group[CLASS_CODE_COLUMN].tolist() if code is not None
                 and not (isinstance(code, float) and np.isnan(code))]
        distinct = sorted({int(code) for code in codes})
        subgroup, subgroup_ok = _unique_or_none(group[labels.SUBGROUP_COLUMN].tolist())
        cs_label, cs_ok = _unique_or_none(group["cs_label"].tolist())
        bg_label, bg_ok = _unique_or_none(group["bg_label"].tolist())

        code: Optional[int] = None
        reason = ""
        if not distinct:
            reason = EXCLUDED_NO_CLASS
        elif len(distinct) > 1:
            reason = EXCLUDED_CONFLICTING_CLASS
        else:
            code = distinct[0]
            if binary_outcome(code) is None:
                reason = EXCLUDED_UNKNOWN_CLASS
        if not reason and not (subgroup_ok and cs_ok and bg_ok):
            reason = EXCLUDED_CONFLICTING_METADATA

        epochs = np.asarray(group[EPOCH_COLUMN], dtype=np.float64)
        finite = epochs[np.isfinite(epochs)]
        splits = sorted({str(value) for value in group[SPLIT_COLUMN].tolist()})
        rows.append({
            GUID_COLUMN: str(guid),
            SPLIT_COLUMN: splits[0] if len(splits) == 1 else "|".join(splits),
            "n_splits": len(splits),
            CLASS_CODE_COLUMN: code,
            labels.CLASS_COLUMN: labels.class_name(code),
            OUTCOME_COLUMN: binary_outcome(code),
            "class_codes_seen": ",".join(str(value) for value in distinct),
            labels.SUBGROUP_COLUMN: subgroup,
            "cs_label": cs_label,
            "bg_label": bg_label,
            "n_segments": int(len(group)),
            "source_files": ",".join(sorted({str(v) for v in group[SOURCE_FILE_COLUMN].tolist()})),
            # Epochs are negative seconds before delivery, so the maximum is the segment closest to
            # it. Reported as a real observation: a recording's last observed time is never moved
            # to the delivery landmark, and no missing terminal trace is extrapolated.
            "first_epoch": float(finite.min()) if finite.size else float("nan"),
            "last_epoch": float(finite.max()) if finite.size else float("nan"),
            EXCLUSION_COLUMN: reason,
        })
    frame = pd.DataFrame(rows)
    counts = exclusion_counts(frame)
    if counts:
        logger.info(f"recording-level exclusions before any coverage rule: {counts}")
    return frame


def exclusion_counts(recordings: pd.DataFrame) -> Dict[str, int]:
    """Count recordings by exclusion reason.

    Args:
        recordings: The recording table.

    Returns:
        Reason -> count, excluding the empty reason. Empty when nothing was excluded.
    """
    if recordings.empty or EXCLUSION_COLUMN not in recordings.columns:
        return {}
    reasons = [str(value) for value in recordings[EXCLUSION_COLUMN].tolist() if str(value)]
    return dict(sorted(Counter(reasons).items()))


# =============================================================================
# Split, grouping and exposure
# =============================================================================
def check_split_disjoint(
    recordings: pd.DataFrame, *, group_column: str = GUID_COLUMN
) -> None:
    """Refuse a cohort whose splits share a recording, or a patient.

    Repeated measurements of one individual on both sides of a split let a model recognise the
    individual rather than generalise to a new one, and the resulting held-out number is not a
    held-out number. The splits are fixed and this check is what proves they were honoured; it is
    never repaired by dropping the duplicate, because a split that overlaps was not built the way
    it was believed to be built.

    Args:
        recordings: The recording table, one row per GUID.
        group_column: What must not span splits. :data:`GUID_COLUMN` always; :data:`PATIENT_COLUMN`
            as well, where a patient mapping exists.

    Raises:
        PilotConfigError: Naming the offending groups and the splits they span.
    """
    if recordings.empty or group_column not in recordings.columns:
        return
    spanning: Dict[str, List[str]] = {}
    for value, group in recordings.groupby(group_column, sort=True):
        splits = sorted({str(item) for item in group[SPLIT_COLUMN].tolist()})
        # A single row can already record a span: a GUID seen in two splits is consolidated into
        # one row whose split reads "train|val".
        expanded = sorted({part for item in splits for part in item.split("|")})
        if len(expanded) > 1:
            spanning[str(value)] = expanded
    if not spanning:
        return
    shown = dict(sorted(spanning.items())[:5])
    raise PilotConfigError(
        f"{len(spanning)} {group_column}(s) appear in more than one split, e.g. {shown}. The fold's "
        f"splits are fixed and must be disjoint: a recording on both sides is fitted on and scored "
        f"on, and the held-out metric it contributes to is not held out. Check the configured shard "
        f"lists against the fold the dataset was built as."
    )


def load_guid_list(path: Optional[Any]) -> Optional[Set[str]]:
    """Read a GUID list, or report that there is none.

    Args:
        path: A text file with one GUID per line; blank lines and ``#`` comments are ignored.
            ``None`` means the list was not supplied.

    Returns:
        The GUIDs, or ``None`` when no path was given. ``None`` and the empty set are different
        answers and are kept apart: "not supplied" is unknown provenance, while an empty file is a
        positive claim that the population is empty.

    Raises:
        FileNotFoundError: If a path was given and is not there. A provenance file that silently
            resolves to nothing would turn "unknown exposure" into "no exposure".
    """
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"GUID list {str(resolved)!r} does not exist. Leave the setting null to record the "
            f"provenance as unknown; a missing file must not be read as an empty one."
        )
    guids: Set[str] = set()
    for line in resolved.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            guids.add(text)
    return guids


def attach_patient_groups(
    recordings: pd.DataFrame, *, patient_map: Optional[Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add the grouping unit every resampling and every split check is performed over.

    Args:
        recordings: The recording table.
        patient_map: A JSON object mapping GUID to a patient or delivery identifier, or ``None``.

    Returns:
        ``(frame, record)``. The frame gains :data:`PATIENT_COLUMN`; a GUID absent from the mapping
        is its own group, which is the honest fallback rather than a guess. The record says whether
        a mapping was supplied and how many GUIDs it covered, so the report can disclose GUID-only
        grouping rather than implying patient-level independence that was never established.

    Raises:
        FileNotFoundError: If a mapping path was given and is not there.
        PilotConfigError: If the mapping is not a JSON object of GUID to identifier.
    """
    frame = recordings.copy()
    mapping: Dict[str, str] = {}
    if patient_map is not None:
        path = Path(patient_map)
        if not path.is_file():
            raise FileNotFoundError(f"patient map {str(path)!r} does not exist.")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise PilotConfigError(
                f"patient map {str(path)!r} must be a JSON object mapping GUID to a patient or "
                f"delivery identifier, got {type(loaded).__name__}."
            )
        mapping = {str(key): str(value) for key, value in loaded.items()}

    guids = [str(value) for value in frame.get(GUID_COLUMN, pd.Series(dtype=str)).tolist()]
    frame[PATIENT_COLUMN] = [mapping.get(guid, guid) for guid in guids]
    covered = sum(1 for guid in guids if guid in mapping)
    record = {
        "patient_map_supplied": patient_map is not None,
        "n_recordings": len(guids),
        "n_recordings_with_patient_id": covered,
        "n_distinct_groups": int(frame[PATIENT_COLUMN].nunique()) if len(guids) else 0,
        "grouping": "patient" if covered else "guid",
        "note": (
            "no patient mapping was supplied, so grouping is GUID-only: repeated recordings of one "
            "individual, if any exist, are treated as independent by every split check and every "
            "bootstrap resample"
            if not covered else
            "grouping is by the supplied patient/delivery identifier; GUIDs absent from the "
            "mapping form their own group"
        ),
    }
    return frame, record


def exposure_record(
    recordings: pd.DataFrame,
    *,
    pretraining_guids: Optional[Set[str]] = None,
    selection_guids: Optional[Set[str]] = None,
    statistics_population: Optional[str] = None,
    held_out_splits: Sequence[str] = ("val", "test"),
) -> Dict[str, Any]:
    """Record what is known -- and what is not -- about this cohort's exposure to the checkpoint.

    Intended folder roles do not prove an actual run was disjoint, so this reports three separate
    states rather than a verdict: exposure *measured* against a supplied population, exposure
    *unknown* because none was supplied, and the pretraining statistics' own fitting population.
    Unknown stays unknown. It is never reported as "no overlap", and it is what makes
    ``clean_holdout_supported`` false.

    Args:
        recordings: The recording table, after grouping.
        pretraining_guids: The GUIDs the checkpoint was pretrained on, or ``None`` for unsupplied.
        selection_guids: The GUIDs its checkpoint selection used, or ``None``.
        statistics_population: Free text naming the population the input statistics were fitted on.
        held_out_splits: The splits whose exposure matters.

    Returns:
        The record, listing the exposed GUIDs where they can be established.
    """
    held_out = {
        str(row[GUID_COLUMN])
        for _, row in recordings.iterrows()
        if any(part in held_out_splits for part in str(row[SPLIT_COLUMN]).split("|"))
    }

    def _overlap(population: Optional[Set[str]], name: str) -> Dict[str, Any]:
        if population is None:
            return {
                "known": False,
                "n_exposed": None,
                "exposed_guids": None,
                "note": (
                    f"no {name} GUID list was supplied, so exposure of the held-out recordings is "
                    f"UNKNOWN. This is not a statement that they are disjoint."
                ),
            }
        exposed = sorted(held_out & population)
        return {
            "known": True,
            "n_population": len(population),
            "n_exposed": len(exposed),
            "exposed_guids": exposed,
            "note": (
                f"{len(exposed)} held-out recording(s) appear in the {name} population; any "
                f"held-out claim must be described as exploratory reuse"
                if exposed else
                f"no held-out recording appears in the supplied {name} population"
            ),
        }

    pretraining = _overlap(pretraining_guids, "pretraining")
    selection = _overlap(selection_guids, "checkpoint-selection")
    return {
        "n_held_out_recordings": len(held_out),
        "pretraining": pretraining,
        "selection": selection,
        "statistics": {
            "population": statistics_population,
            "known": statistics_population is not None,
            "note": (
                "the population the input statistics were fitted on was not recorded, so their "
                "exposure to the held-out recordings is unknown"
                if statistics_population is None else
                "as supplied by the operator; recorded verbatim and not inferred"
            ),
        },
        # True only when both populations were supplied AND neither touches the held-out split.
        # Any other combination -- including "no list was given" -- leaves it false.
        "clean_holdout_supported": bool(
            pretraining["known"] and selection["known"]
            and not pretraining["n_exposed"] and not selection["n_exposed"]
        ),
    }


def require_both_classes(
    recordings: pd.DataFrame,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    eligible_only: bool = True,
) -> None:
    """Refuse a split that cannot answer the question it is there to answer.

    Discrimination is undefined on a split carrying one binary class, and average precision has no
    chance level there either. This raises rather than reporting an empty metric -- and it is
    deliberately not a reason to try another fold, whose pictures would then have been chosen by
    looking at them.

    Args:
        recordings: The recording table.
        splits: The splits that must carry both classes.
        eligible_only: Count only recordings that survived every rule so far, when an eligibility
            column exists. The counts that matter are the ones after exclusions, not the shard's.

    Raises:
        PilotConfigError: Naming the split and the counts it has.
    """
    frame = recordings
    if eligible_only:
        frame = frame[frame[EXCLUSION_COLUMN].astype(str) == ""]
        if "eligible" in frame.columns:
            frame = frame[frame["eligible"].astype(bool)]
    for split in splits:
        rows = frame[frame[SPLIT_COLUMN].astype(str) == split]
        present = {
            int(value) for value in rows[OUTCOME_COLUMN].tolist()
            if value is not None and not (isinstance(value, float) and np.isnan(value))
        }
        if present != {0, 1}:
            counts = {
                outcome: int((rows[OUTCOME_COLUMN] == outcome).sum()) for outcome in (0, 1)
            }
            raise PilotConfigError(
                f"split {split!r} carries {counts} usable recording(s) by binary outcome, so the "
                f"specified experiment cannot estimate discrimination on it. Fix the cohort or "
                f"report that the pilot cannot run on this fold -- do not select another fold "
                f"because its results look better."
            )


# =============================================================================
# Coverage and serialisation
# =============================================================================
def _flag_is(value: Any, flag: bool) -> bool:
    """Whether a stored CS/blood-gas flag equals ``flag``.

    Compared by value rather than by identity: pandas stores a fully populated boolean column as
    ``numpy.bool_``, which is not the ``True`` singleton, so an identity test would put every
    recording in the ``False`` stratum and report it as a measurement.

    Args:
        value: The stored flag, possibly ``None`` or NaN.
        flag: The stratum being counted.

    Returns:
        ``True`` when the flag is present and equal. A missing flag belongs to neither stratum.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(value) == flag


def coverage_summary(segments: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Summarise each split by stratum, including the strata that are empty.

    An absent row and a zero row read very differently, and only one of them is honest: a subgroup
    with no recordings is a fact about this fold that a reader must see before reading any
    subgroup contrast. Every canonical subgroup therefore gets a row on every split, whether or not
    the cohort has one.

    Args:
        segments: The segment table for all splits.
        recordings: The recording table for all splits.

    Returns:
        One row per ``(split, stratum kind, stratum)``, with recording and segment counts, the
        eligible count where eligibility has been computed, and the distribution of last observed
        segment starts in hours before delivery.
    """
    rows: List[Dict[str, Any]] = []

    def _emit(split: str, kind: str, name: str, guids: Sequence[str]) -> None:
        chosen = recordings[recordings[GUID_COLUMN].isin(list(guids))]
        usable = chosen[chosen[EXCLUSION_COLUMN].astype(str) == ""]
        if "eligible" in usable.columns:
            usable = usable[usable["eligible"].astype(bool)]
        last = np.asarray(chosen.get("last_epoch", pd.Series(dtype=float)), dtype=np.float64)
        last = -last[np.isfinite(last)] / 3600.0
        rows.append({
            SPLIT_COLUMN: split,
            "stratum_kind": kind,
            "stratum": name,
            "n_recordings": int(len(chosen)),
            "n_usable_recordings": int(len(usable)),
            "n_segments": int(
                segments[segments[GUID_COLUMN].isin(list(guids))].shape[0]
            ),
            "last_observed_hours_before_delivery_median": (
                float(np.median(last)) if last.size else float("nan")
            ),
            "last_observed_hours_before_delivery_min": (
                float(last.min()) if last.size else float("nan")
            ),
            "last_observed_hours_before_delivery_max": (
                float(last.max()) if last.size else float("nan")
            ),
        })

    for split in SPLITS:
        in_split = recordings[
            recordings[SPLIT_COLUMN].astype(str).apply(lambda value: split in value.split("|"))
        ] if not recordings.empty else recordings
        if in_split.empty:
            continue
        guids = [str(value) for value in in_split[GUID_COLUMN].tolist()]
        _emit(split, "all", "all", guids)
        for name in ("healthy", "acidosis", "hie"):
            _emit(split, labels.CLASS_COLUMN, name, [
                str(row[GUID_COLUMN]) for _, row in in_split.iterrows()
                if row[labels.CLASS_COLUMN] == name
            ])
        for outcome, name in ((0, "healthy"), (1, "adverse")):
            _emit(split, OUTCOME_COLUMN, name, [
                str(row[GUID_COLUMN]) for _, row in in_split.iterrows()
                if row[OUTCOME_COLUMN] == outcome
            ])
        # Every canonical subgroup, present or not.
        for name in labels.CANONICAL_SUBGROUPS:
            _emit(split, labels.SUBGROUP_COLUMN, name, [
                str(row[GUID_COLUMN]) for _, row in in_split.iterrows()
                if row[labels.SUBGROUP_COLUMN] == name
            ])
        for column in ("cs_label", "bg_label"):
            for flag in (True, False):
                _emit(split, column, str(flag), [
                    str(row[GUID_COLUMN]) for _, row in in_split.iterrows()
                    if _flag_is(row[column], flag)
                ])
    return pd.DataFrame(rows)


def write_manifest(recordings: pd.DataFrame, directory: Any) -> Path:
    """Write the recording manifest into a run directory.

    Args:
        recordings: The recording table.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / MANIFEST_FILENAME
    recordings.to_csv(target, index=False)
    logger.info(f"wrote {len(recordings)} recording(s) to {target}")
    return target


def write_coverage(coverage: pd.DataFrame, directory: Any) -> Path:
    """Write the coverage table into a run directory.

    Args:
        coverage: The coverage table.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / COVERAGE_FILENAME
    coverage.to_csv(target, index=False)
    logger.info(f"wrote {len(coverage)} coverage row(s) to {target}")
    return target


def write_cohort_table(frame: pd.DataFrame, directory: Any, *, name: str) -> Path:
    """Persist one cohort table for the stages that follow.

    The cohort is established once, by the stage that reads the shards, and every later stage reads
    it back rather than rebuilding it: two derivations of "which recordings are in" are two chances
    for one of them to keep a recording the other dropped.

    Args:
        frame: :data:`RECORDINGS_FILENAME`'s table or :data:`SEGMENTS_FILENAME`'s.
        directory: The run directory. Created if absent.
        name: Which of the two, as a filename constant.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / str(name)
    frame.to_parquet(target, index=False)
    logger.info(f"wrote {len(frame)} row(s) to {target}")
    return target


def read_cohort_table(directory: Any, *, name: str) -> pd.DataFrame:
    """Read one cohort table a previous stage established.

    Args:
        directory: The run directory.
        name: Which table, as a filename constant.

    Returns:
        The table, with its dtypes as written.

    Raises:
        FileNotFoundError: If no stage has established the cohort yet. Rebuilding it here would
            silently answer a question the operator asked a different stage.
    """
    target = Path(directory) / str(name)
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing: the cohort has not been established in this run. Run the "
            f"preflight stage, which reads the shards and writes it, before any stage that "
            f"consumes it."
        )
    return pd.read_parquet(target)


def split_loader(config: Mapping[str, Any]) -> Any:
    """Build one split's dataloader from a pilot loader configuration.

    The repository's own data module, reached the way the evaluation pipeline reaches it, so the
    batching, the collation and the single-process policy are the ones a real run uses. Imported
    lazily: importing this module must cost no data module, and the minimal logic subset imports it.

    Args:
        config: The configuration from :func:`pilot_loader_config`, whose ``vae_test_datasets``
            names the split to read.

    Returns:
        A re-iterable dataloader over that split.
    """
    from train.data_module import GraphDataModule

    return GraphDataModule(dict(config)).test_dataloader()


# =============================================================================
# The time axis: anchors, not segment starts
# =============================================================================
#: Seconds per hour, so the conversion appears once.
SECONDS_PER_HOUR = 3600.0

ANCHOR_COLUMN = "anchor"
#: The anchor's own time relative to delivery, in seconds and negative before it.
ANCHOR_SECONDS_COLUMN = "anchor_seconds"
#: The same instant as a positive number of hours **before** delivery, which is the axis every
#: window, bin and figure in this pilot is defined on.
HOURS_COLUMN = "hours_before_delivery"
#: The latest stored coefficient any scored channel reads for this anchor, in seconds relative to
#: delivery. It is the endpoint the post-delivery check is made against.
LATEST_SCORED_COLUMN = "latest_scored_seconds"
#: Row index into the aligned latent matrix, so a reduction can gather without a join.
ROW_COLUMN = "row"
BIN_COLUMN = "time_bin"
BIN_LABEL_COLUMN = "time_bin_label"

#: Why an anchor is not retained. Every one of these is recorded on the row rather than removed, so
#: the exclusion counts survive into the coverage table.
EXCLUDED_NOT_CONTRIBUTING = "no_forecast_contribution"
EXCLUDED_OUTSIDE_WINDOW = "outside_preservation_window"
EXCLUDED_POST_DELIVERY = "scored_endpoint_at_or_after_delivery"
EXCLUDED_DUPLICATE_KEY = "duplicate_segment_anchor_key"
EXCLUDED_DUPLICATE_TIME = "duplicate_absolute_anchor_time"

#: Why a recording cannot supply a supervised bag. Coverage rules, not clinical criteria, and
#: applied identically to both classes.
INELIGIBLE_NO_LATE_ANCHOR = "no_retained_anchor_in_supervised_window"
INELIGIBLE_FEW_LATE_SEGMENTS = "fewer_late_segments_than_required"
INELIGIBLE_NO_FINAL_ANCHOR = "no_retained_anchor_within_final_minutes"

#: Comparisons on the hour axis are made to this tolerance, so an anchor landing exactly on a bin
#: or window edge falls on the closed side rather than wherever float rounding sends it. Four
#: seconds is one decimated step, and this is nine orders of magnitude below that: it separates
#: representation error from any real time difference.
_HOUR_TOLERANCE = 1e-9


def step_seconds() -> float:
    """Seconds per decimated step, read from the dataset's own decimation contract.

    Returns:
        The step, $\\mathrm{DECIMATION} / \\mathrm{RAW\\_SAMPLING\\_HZ} = 16 / 4 = 4$ seconds.
        Derived rather than written down, so a dataset rebuilt at another decimation moves every
        timestamp in this package with it instead of silently disagreeing by a constant.
    """
    from hdf5_dataset.hdf5_dataset import DECIMATION, RAW_SAMPLING_HZ

    return float(DECIMATION) / float(RAW_SAMPLING_HZ)


def trim_seconds(trim_minutes: Optional[float]) -> float:
    """Seconds the loader's trim discards from the **start** of a stored segment.

    Taken through the loader's own ``decimated_trim_steps`` rather than as $60m$, because the trim
    drops a whole number of *decimated steps*: at the shipped ``trim_minutes: 1.0`` the two agree
    exactly ($240$ raw samples, $15$ steps, $60$ s), and at a trim whose sample count is not a
    multiple of the decimation they do not. Using the loader's own conversion is what keeps this
    timestamp on the same grid as the array the loader actually returned.

    Args:
        trim_minutes: The configured symmetric trim, or ``None`` for no trim.

    Returns:
        The offset from the untrimmed segment start to the first retained step, in seconds.
    """
    from hdf5_dataset.hdf5_dataset import decimated_trim_steps

    _raw, steps = decimated_trim_steps(trim_minutes)
    return float(steps) * step_seconds()


def anchor_seconds(
    epoch: Any, anchor: Any, *, trim_minutes: Optional[float]
) -> np.ndarray:
    r"""The absolute time an anchor observes, relative to delivery.

    $$t_{isa} = \mathrm{epoch}_{is} + \mathrm{trim\_seconds} + \Delta \cdot a,$$

    with $\Delta$ the decimated step. **``epoch`` is the untrimmed segment start**, which is the
    whole reason this function exists: the loader trims the signals but leaves ``epoch`` alone, so
    an anchor's time is neither the segment's start nor an offset from it that anyone can guess.

    The worked case: $\mathrm{epoch} = -3600$, ``trim_minutes`` $= 1$, $a = 150$ gives $-2940$ s --
    **49 minutes** before delivery, not 60.

    Args:
        epoch: The segment's untrimmed start, in seconds relative to delivery.
        anchor: The anchor's index into the **trimmed** decimated array.
        trim_minutes: The loader's trim.

    Returns:
        Seconds relative to delivery, negative before it.
    """
    return (
        np.asarray(epoch, dtype=np.float64)
        + trim_seconds(trim_minutes)
        + step_seconds() * np.asarray(anchor, dtype=np.float64)
    )


def hours_before_delivery(seconds: Any) -> np.ndarray:
    """Convert seconds relative to delivery into positive hours before it.

    Args:
        seconds: Times relative to delivery, negative before it.

    Returns:
        Hours before delivery. A time at or after delivery comes out at or below zero and is
        excluded by the window rule rather than clipped into the first bin.
    """
    return -np.asarray(seconds, dtype=np.float64) / SECONDS_PER_HOUR


def segment_span_seconds(sequence_length: int, trim_minutes: Optional[float]) -> float:
    """How much wall-clock time one stored segment spans, before trimming.

    Derived from the checkpoint's own ``sequence_length`` and the loader's trim rather than from
    the "22 minutes" the current shards happen to hold, so a dataset built at another segment
    length widens the coarse filter by the right amount without an edit here.

    Args:
        sequence_length: Trimmed decimated steps per segment, $T$.
        trim_minutes: The loader's symmetric trim.

    Returns:
        The untrimmed span in seconds: $(T + 2 \\cdot \\mathrm{trim\\_steps}) \\cdot \\Delta$.
    """
    return (
        float(sequence_length) * step_seconds() + 2.0 * trim_seconds(trim_minutes)
    )


def coarse_epoch_min(preservation_hours: float, span_seconds: float) -> float:
    r"""The loader's coarse lower bound on the segment start.

    A segment whose *start* is earlier than the window still carries anchors inside it, so the
    bound is widened by a whole segment:

    $$\mathrm{epoch\_min} = -\left(3600 \cdot \mathrm{preservation\_hours} + \mathrm{span}\right).$$

    Setting it at $-3600 \cdot \mathrm{preservation\_hours}$ instead would drop every segment that
    crosses the boundary, taking their eligible anchors with them -- silently, because the loader
    reports a filter as a smaller dataset and not as an exclusion. The exact per-anchor filter runs
    afterwards; this bound only has to be generous.

    Args:
        preservation_hours: The window anchors are kept in, in hours before delivery.
        span_seconds: The untrimmed span of one stored segment.

    Returns:
        The bound, in seconds relative to delivery.
    """
    return -(SECONDS_PER_HOUR * float(preservation_hours) + float(span_seconds))


def in_window(hours: Any, low: float, high: float) -> np.ndarray:
    r"""Membership of the half-open window $(\mathrm{low}, \mathrm{high}]$ in hours before delivery.

    Half-open at the early edge and closed at the late one, so adjacent windows and adjacent bins
    tile the axis without sharing an anchor. An anchor exactly on an edge lands in the window whose
    upper bound it is.

    Args:
        hours: Hours before delivery.
        low: Exclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        A boolean array.
    """
    values = np.asarray(hours, dtype=np.float64)
    return (values > float(low) + _HOUR_TOLERANCE) & (values <= float(high) + _HOUR_TOLERANCE)


# =============================================================================
# Anchor support: one policy, taken from the model that was trained
# =============================================================================
def contributing_support(
    model: Any, *, weight: Any, anchor_index: Any, anchor_valid: Any
) -> Tuple[Any, Any]:
    """Which decoded anchors carry a forecast term, under the checkpoint's own rules.

    One support policy for every model version, and it is the objective's rather than a second
    definition: the validity signal is pooled by the model's own ``scored_weight`` (the identity
    under the stored clock, and a conservative minimum over the shift span under a re-indexed
    one), and the mask is then the same ``forecast_mask`` the training loss and the evaluation
    pipeline build -- warm-up prefix, the anchor's own validity, each forecast step's validity, the
    padding flag and the checkpoint's coverage floor, all inherited rather than restated.

    Adopting *forecast-contributing* support is a deliberate conservatism with a consequence worth
    recording: the latent analysis inherits the forecast's availability exclusions, so it describes
    the anchors the model could be scored at rather than every anchor a state could be inferred at.

    Args:
        model: The rebuilt checkpoint. Read for ``scored_weight``, ``geometry`` and
            ``coverage_floor`` only -- no forward is run here.
        weight: The loader's decimated validity signal $(B, T)$.
        anchor_index: The anchors the forward decoded, $(B, A)$.
        anchor_valid: Which of those are real rather than padding, $(B, A)$.

    Returns:
        ``(contributing, coverage_frac)``: a boolean $(B, A)$ indicator and the per-anchor valid
        fraction of the forecast window, before the warm-up, padding and floor factors.
    """
    from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

    mask, coverage = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchor_index,
        anchor_valid=anchor_valid,
    )
    return contributing_anchors(mask).bool(), coverage


def max_forecast_shift(model: Any) -> int:
    """The furthest-forward per-channel re-indexing of a scored coefficient.

    Args:
        model: The rebuilt checkpoint.

    Returns:
        $\\max_c s_c$, or $0$ under the stored clock where the model carries no shift. The
        *maximum* rather than the magnitude: the post-delivery check asks how far forward the
        latest scored element reaches, and a delaying clock (all $s_c \\le 0$) reaches less far
        than the unshifted one rather than further.
    """
    shift = getattr(model, "target_forecast_shift", None)
    if shift is None or not len(shift):
        return 0
    return int(max(int(value) for value in shift))


def latest_scored_seconds(
    frame: pd.DataFrame, *, horizon: int, forecast_shift: int, trim_minutes: Optional[float]
) -> np.ndarray:
    r"""When the last coefficient an anchor is scored against was recorded.

    Anchor $t$, horizon step $\tau$ and channel $c$ are scored against stored step
    $t + 1 + \tau + s_c$, so the furthest element any channel reaches is
    $t + H + \max_c s_c$.

    Args:
        frame: An anchor table carrying :data:`ANCHOR_COLUMN` and :data:`EPOCH_COLUMN`.
        horizon: The checkpoint's $H$.
        forecast_shift: $\max_c s_c$, from :func:`max_forecast_shift`.
        trim_minutes: The loader's trim.

    Returns:
        Seconds relative to delivery.
    """
    return anchor_seconds(
        frame[EPOCH_COLUMN],
        np.asarray(frame[ANCHOR_COLUMN], dtype=np.float64) + int(horizon) + int(forecast_shift),
        trim_minutes=trim_minutes,
    )


def add_anchor_times(
    frame: pd.DataFrame,
    *,
    trim_minutes: Optional[float],
    horizon: int,
    forecast_shift: int = 0,
) -> pd.DataFrame:
    """Add the anchor's own time, its hour coordinate and its latest scored endpoint.

    Args:
        frame: An anchor table carrying :data:`EPOCH_COLUMN` and :data:`ANCHOR_COLUMN`.
        trim_minutes: The loader's trim.
        horizon: The checkpoint's forecast horizon.
        forecast_shift: The furthest-forward per-channel shift.

    Returns:
        A copy carrying :data:`ANCHOR_SECONDS_COLUMN`, :data:`HOURS_COLUMN` and
        :data:`LATEST_SCORED_COLUMN`. The columns are added beside the existing ones; nothing that
        reads segment-start time is redefined.
    """
    out = frame.copy()
    out[ANCHOR_SECONDS_COLUMN] = anchor_seconds(
        out[EPOCH_COLUMN], out[ANCHOR_COLUMN], trim_minutes=trim_minutes
    )
    out[HOURS_COLUMN] = hours_before_delivery(out[ANCHOR_SECONDS_COLUMN])
    out[LATEST_SCORED_COLUMN] = latest_scored_seconds(
        out, horizon=horizon, forecast_shift=forecast_shift, trim_minutes=trim_minutes
    )
    return out


def mark_window_and_delivery(
    frame: pd.DataFrame, *, preservation_hours: float
) -> pd.DataFrame:
    r"""Record which anchors fall outside the window or read past delivery.

    Two separate rules, and the second is not implied by the first: $\mathrm{epoch} < 0$ says a
    segment *started* before delivery, and an anchor inside $(0, 3]$ hours can still be scored
    against a coefficient recorded after it. Both reasons are written onto the row rather than
    removing it, so the coverage table can report what was lost and why.

    Args:
        frame: An anchor table after :func:`add_anchor_times`.
        preservation_hours: The upper edge of the preserved window.

    Returns:
        A copy carrying :data:`EXCLUSION_COLUMN`, empty where the anchor is retained. An existing
        reason -- support, typically -- is never overwritten: the first reason an anchor failed is
        the one reported.
    """
    out = frame.copy()
    if EXCLUSION_COLUMN not in out.columns:
        out[EXCLUSION_COLUMN] = ""
    reasons = out[EXCLUSION_COLUMN].astype(str).to_numpy(dtype=object)
    hours = np.asarray(out[HOURS_COLUMN], dtype=np.float64)

    outside = ~in_window(hours, 0.0, float(preservation_hours))
    reasons = np.where((reasons == "") & outside, EXCLUDED_OUTSIDE_WINDOW, reasons)
    # Strictly before delivery: a coefficient recorded at the landmark itself is not a forecast of
    # anything this cohort observed.
    post = np.asarray(out[LATEST_SCORED_COLUMN], dtype=np.float64) >= 0.0
    reasons = np.where((reasons == "") & post, EXCLUDED_POST_DELIVERY, reasons)
    out[EXCLUSION_COLUMN] = reasons
    return out


def deduplicate_anchors(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per instant per recording, deterministically.

    Two distinct duplications, and both are counted rather than dropped:

    * the same ``(guid, epoch, anchor)`` key twice, which is a collection defect;
    * two overlapping segments of one recording supplying the **same absolute time**, which is a
      property of the shards. The survivor is the row with more valid history behind it inside its
      own segment -- the larger anchor index -- and, where that ties, the earlier segment start.
      Both criteria are fixed in advance, so the choice cannot depend on which segment happened to
      be read first.

    Never counted as independent observations: a recording that contributed the same instant twice
    would otherwise weigh double in its own segment mean and in every bin it lands in.

    Args:
        frame: An anchor table after :func:`mark_window_and_delivery`.

    Returns:
        A copy whose duplicate rows carry a duplication reason. Rows already excluded for another
        reason keep it and take no part in the contest.
    """
    out = frame.copy()
    if EXCLUSION_COLUMN not in out.columns:
        out[EXCLUSION_COLUMN] = ""
    reasons = out[EXCLUSION_COLUMN].astype(str).to_numpy(dtype=object)

    live = [index for index, reason in enumerate(reasons) if reason == ""]
    seen_keys: Set[Tuple[str, float, int]] = set()
    for position in live:
        row = out.iloc[position]
        key = (str(row[GUID_COLUMN]), float(row[EPOCH_COLUMN]), int(row[ANCHOR_COLUMN]))
        if key in seen_keys:
            reasons[position] = EXCLUDED_DUPLICATE_KEY
        else:
            seen_keys.add(key)

    # Sorted so the winner of each absolute-time contest is met first: larger anchor index -- more
    # valid history inside its own segment -- then the earlier segment start.
    contenders = sorted(
        (position for position in live if reasons[position] == ""),
        key=lambda position: (
            str(out.iloc[position][GUID_COLUMN]),
            float(out.iloc[position][ANCHOR_SECONDS_COLUMN]),
            -int(out.iloc[position][ANCHOR_COLUMN]),
            float(out.iloc[position][EPOCH_COLUMN]),
        ),
    )
    seen_times: Set[Tuple[str, float]] = set()
    for position in contenders:
        row = out.iloc[position]
        stamp = (str(row[GUID_COLUMN]), float(row[ANCHOR_SECONDS_COLUMN]))
        if stamp in seen_times:
            reasons[position] = EXCLUDED_DUPLICATE_TIME
        else:
            seen_times.add(stamp)

    out[EXCLUSION_COLUMN] = reasons
    return out


def retained(frame: pd.DataFrame) -> pd.DataFrame:
    """The anchors that survived every support, window, delivery and duplication rule.

    This is also the **teacher support**: the preservation term is applied over exactly these
    anchors across the whole preserved window, so the anchors the student is held to are the
    anchors the analysis reads, with no third definition in between.

    Args:
        frame: An anchor table carrying :data:`EXCLUSION_COLUMN`.

    Returns:
        The retained rows, in their original order.
    """
    if frame.empty or EXCLUSION_COLUMN not in frame.columns:
        return frame
    return frame[frame[EXCLUSION_COLUMN].astype(str) == ""]


def anchor_exclusion_counts(frame: pd.DataFrame) -> Dict[str, int]:
    """Count anchors by exclusion reason.

    Args:
        frame: An anchor table.

    Returns:
        Reason -> count, excluding the retained rows.
    """
    if frame.empty or EXCLUSION_COLUMN not in frame.columns:
        return {}
    return dict(sorted(Counter(
        reason for reason in frame[EXCLUSION_COLUMN].astype(str).tolist() if reason
    ).items()))


# =============================================================================
# Bins
# =============================================================================
def bin_edges(*, bin_hours: float, preservation_hours: float) -> List[Tuple[float, float]]:
    r"""The fixed trajectory bins, nearest delivery first.

    Six half-hour bins over three hours at the shipped settings: $(0, 0.5]$, $(0.5, 1]$,
    $(1, 1.5]$, $(1.5, 2]$, $(2, 2.5]$, $(2.5, 3]$.

    Args:
        bin_hours: Bin width in hours.
        preservation_hours: The window's upper edge.

    Returns:
        ``(low, high]`` pairs in hours before delivery, index 0 nearest delivery.
    """
    count = int(round(float(preservation_hours) / float(bin_hours)))
    return [(index * bin_hours, (index + 1) * bin_hours) for index in range(count)]


def assign_time_bins(
    frame: pd.DataFrame, *, bin_hours: float, preservation_hours: float
) -> pd.DataFrame:
    r"""Add the fixed trajectory bin each anchor falls in.

    $$\mathrm{bin} = \left\lceil r / w \right\rceil - 1,$$

    which places an anchor exactly on an edge in the bin that edge closes -- the same half-open
    rule :func:`in_window` uses, so a bin and a window never disagree about one anchor.

    Args:
        frame: An anchor table after :func:`add_anchor_times`.
        bin_hours: Bin width in hours.
        preservation_hours: The window's upper edge.

    Returns:
        A copy carrying :data:`BIN_COLUMN` (an integer index, $-1$ outside the window) and
        :data:`BIN_LABEL_COLUMN`.
    """
    out = frame.copy()
    edges = bin_edges(bin_hours=bin_hours, preservation_hours=preservation_hours)
    hours = np.asarray(out[HOURS_COLUMN], dtype=np.float64)
    index = np.ceil(hours / float(bin_hours) - _HOUR_TOLERANCE).astype(np.int64) - 1
    inside = in_window(hours, 0.0, float(preservation_hours))
    index = np.where(inside, np.clip(index, 0, len(edges) - 1), -1)
    out[BIN_COLUMN] = index
    out[BIN_LABEL_COLUMN] = [
        f"({edges[value][0]:g}, {edges[value][1]:g}]" if value >= 0 else ""
        for value in index.tolist()
    ]
    return out


# =============================================================================
# Reductions: anchors -> segments -> recordings
# =============================================================================
def _gathered_mean(values: np.ndarray, rows: Sequence[int]) -> np.ndarray:
    """Mean of the selected rows of a value matrix."""
    return np.asarray(values, dtype=np.float64)[list(rows)].mean(axis=0)


def segment_means(
    frame: pd.DataFrame, values: np.ndarray, *, group_columns: Sequence[str] = ()
) -> Tuple[pd.DataFrame, np.ndarray]:
    r"""Average anchors within each segment.

    $$e_{is} = \frac{1}{|A_{is}|} \sum_{a \in A_{is}} \mu_{isa}.$$

    The first half of the reduction hierarchy, and the half that stops a densely sampled segment
    from outvoting a sparse one: every later step works on segment vectors, so an anchor count
    never reaches a recording's weight.

    Args:
        frame: Retained anchors, carrying :data:`ROW_COLUMN` and the identity columns.
        values: The aligned value matrix, one row per anchor.
        group_columns: Extra columns to group within, such as the time bin.

    Returns:
        ``(frame, matrix)``: one row per segment with its median time before delivery and anchor
        count, and the matching means. The frame's :data:`ROW_COLUMN` indexes the returned matrix,
        so the next reduction chains without a join.
    """
    keys = [GUID_COLUMN, EPOCH_COLUMN, *group_columns]
    rows: List[Dict[str, Any]] = []
    means: List[np.ndarray] = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(keys, key)),
            # The median of the anchors actually included, which is what the recency weight is
            # computed from -- not the segment's start, and not the midpoint of a nominal window.
            HOURS_COLUMN: float(np.median(np.asarray(group[HOURS_COLUMN], dtype=np.float64))),
            "n_anchors": int(len(group)),
            ROW_COLUMN: len(means),
        })
        means.append(_gathered_mean(values, group[ROW_COLUMN].tolist()))
    matrix = np.stack(means) if means else np.zeros((0, np.asarray(values).shape[-1]))
    return pd.DataFrame(rows), matrix


def recency_weights(hours: Any, *, halflife_hours: float) -> np.ndarray:
    r"""The recency weight each segment carries inside the supervised window.

    $$\omega_{is} = 2^{-\widetilde r_{is} / h}.$$

    A weighting preference and nothing more: it is identical for healthy and adverse recordings,
    it encodes no severity, and it is applied only within the supervised hour, where every
    contributing segment is already close to delivery.

    Args:
        hours: Each segment's median time before delivery.
        halflife_hours: The half-life $h$.

    Returns:
        The weights, all positive.
    """
    return np.exp2(-np.asarray(hours, dtype=np.float64) / float(halflife_hours))


def recording_means(
    segments: pd.DataFrame,
    values: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    group_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, np.ndarray]:
    r"""Average segments within each recording, optionally weighted.

    $$v_i = \frac{\sum_s \omega_{is} e_{is}}{\sum_s \omega_{is}},$$

    with $\omega \equiv 1$ when no weights are given. This is the step that makes every recording
    one observation, which is what the classification loss, the bootstrap and every reported count
    are defined over.

    Args:
        segments: Segment rows from :func:`segment_means`.
        values: The aligned segment-mean matrix.
        weights: Per-segment weights, or ``None`` for a plain mean.
        group_columns: Extra columns to group within, such as the time bin.

    Returns:
        ``(frame, matrix)``: one row per recording (per group), carrying its segment and anchor
        counts and its median time before delivery, with :data:`ROW_COLUMN` indexing the matrix.
    """
    keys = [GUID_COLUMN, *group_columns]
    matrix = np.asarray(values, dtype=np.float64)
    weight_vector = None if weights is None else np.asarray(weights, dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    means: List[np.ndarray] = []
    for key, group in segments.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        positions = group[ROW_COLUMN].tolist()
        block = matrix[positions]
        if weight_vector is None:
            mean = block.mean(axis=0)
        else:
            omega = weight_vector[positions]
            mean = (omega[:, None] * block).sum(axis=0) / omega.sum()
        rows.append({
            **dict(zip(keys, key)),
            HOURS_COLUMN: float(np.median(np.asarray(group[HOURS_COLUMN], dtype=np.float64))),
            "n_segments": int(len(group)),
            "n_anchors": int(group["n_anchors"].sum()),
            ROW_COLUMN: len(means),
        })
        means.append(mean)
    stacked = np.stack(means) if means else np.zeros((0, matrix.shape[-1]))
    return pd.DataFrame(rows), stacked


def recording_bags(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    supervised_hours: float,
    halflife_hours: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """The supervised bag: one vector per recording from its final-hour anchors.

    Anchors are averaged within segments first, then segments are pooled with the recency weight,
    so duplicating a segment's anchors does not multiply that recording's supervised weight and
    each recording supplies exactly one classification loss.

    Args:
        frame: Retained anchors, after :func:`add_anchor_times`.
        values: The aligned per-anchor value matrix.
        supervised_hours: The supervised window's upper edge.
        halflife_hours: The recency half-life inside that window.

    Returns:
        ``(frame, matrix)``: one row per recording with a bag, and the bags themselves.
    """
    late = frame[in_window(frame[HOURS_COLUMN], 0.0, float(supervised_hours))]
    segments, segment_values = segment_means(late, values)
    weights = recency_weights(segments[HOURS_COLUMN], halflife_hours=halflife_hours)
    return recording_means(segments, segment_values, weights=weights)


def recording_bin_means(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    bin_hours: float,
    preservation_hours: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """One vector per recording per occupied trajectory bin.

    Unweighted inside a bin -- the recency weight belongs to the supervised bag and would tilt a
    trajectory point towards its own bin edge -- and unoccupied bins are simply absent rather than
    filled: a recording with no anchors in a bin has no observation there, and drawing one would
    invent a measurement.

    Args:
        frame: Retained anchors, after :func:`add_anchor_times`.
        values: The aligned per-anchor value matrix.
        bin_hours: Bin width in hours.
        preservation_hours: The window's upper edge.

    Returns:
        ``(frame, matrix)``: one row per ``(guid, bin)`` with counts and the bin label.
    """
    binned = assign_time_bins(
        frame, bin_hours=bin_hours, preservation_hours=preservation_hours
    )
    binned = binned[binned[BIN_COLUMN] >= 0]
    segments, segment_values = segment_means(binned, values, group_columns=(BIN_COLUMN,))
    recordings, matrix = recording_means(
        segments, segment_values, group_columns=(BIN_COLUMN,)
    )
    edges = bin_edges(bin_hours=bin_hours, preservation_hours=preservation_hours)
    if not recordings.empty:
        recordings[BIN_LABEL_COLUMN] = [
            f"({edges[int(value)][0]:g}, {edges[int(value)][1]:g}]"
            for value in recordings[BIN_COLUMN].tolist()
        ]
    return recordings, matrix


def window_means(
    frame: pd.DataFrame, values: np.ndarray, *, low: float, high: float
) -> Tuple[pd.DataFrame, np.ndarray]:
    r"""One vector per recording over an arbitrary window $(\mathrm{low}, \mathrm{high}]$.

    Used for the paired early/late temporal comparison, through the same anchors-then-segments-then-
    recordings order as everything else, so an early and a late summary of one recording are
    reduced identically and their difference is not an artefact of two aggregation rules.

    Args:
        frame: Retained anchors, after :func:`add_anchor_times`.
        values: The aligned per-anchor value matrix.
        low: Exclusive lower bound in hours before delivery.
        high: Inclusive upper bound.

    Returns:
        ``(frame, matrix)``: one row per recording observed in the window. A recording with no
        anchors there is absent, which is what makes the paired comparison paired.
    """
    inside = frame[in_window(frame[HOURS_COLUMN], low, high)]
    segments, segment_values = segment_means(inside, values)
    return recording_means(segments, segment_values)


# =============================================================================
# Who is in: the one filter every stage applies
# =============================================================================
def eligible_outcomes(recordings: pd.DataFrame, *, split: str) -> Dict[str, int]:
    """GUID -> binary outcome for the recordings of one split that survived every rule.

    One definition of "in", read by the bags, the plans and every analysis: a recording excluded at
    the labelling stage or by the late-coverage rules is out of the fit and out of every count
    reported from it. Written once because three copies of a filter are three chances for one of
    them to keep a recording the others dropped.

    Args:
        recordings: The recording table, with eligibility attached where it exists.
        split: The split being selected.

    Returns:
        The mapping, empty when nothing survives.
    """
    frame = recordings[recordings[SPLIT_COLUMN].astype(str) == str(split)]
    frame = frame[frame[EXCLUSION_COLUMN].astype(str) == ""]
    if "eligible" in frame.columns:
        frame = frame[frame["eligible"].astype(bool)]
    return {
        str(row[GUID_COLUMN]): int(row[OUTCOME_COLUMN])
        for _index, row in frame.iterrows()
        if row[OUTCOME_COLUMN] is not None and not pd.isna(row[OUTCOME_COLUMN])
    }


def eligible_anchors(
    frame: pd.DataFrame, recordings: pd.DataFrame, *, split: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Restrict an anchor table to the eligible, labelled recordings of one split.

    Args:
        frame: Retained anchors. Its :data:`ROW_COLUMN` is preserved, so the returned view still
            indexes the extraction's own arrays.
        recordings: The recording table.
        split: The split being selected.

    Returns:
        ``(anchors, outcomes)``.
    """
    outcomes = eligible_outcomes(recordings, split=split)
    return frame[frame[GUID_COLUMN].astype(str).isin(outcomes)], outcomes


# =============================================================================
# Late eligibility
# =============================================================================
def late_eligibility(
    frame: pd.DataFrame,
    recordings: pd.DataFrame,
    *,
    supervised_hours: float,
    min_late_segments: int,
    final_anchor_within_minutes: float,
) -> pd.DataFrame:
    """Decide which recordings can supply a supervised bag, and say why the others cannot.

    Two coverage rules, fixed before any outcome comparison and applied identically to both
    classes: at least ``min_late_segments`` contributing segments inside the supervised window, and
    a retained anchor within the final ``final_anchor_within_minutes``. They are pragmatic rules
    about what was observed, not clinical onset criteria, and the recording's *actual* last
    observed time is recorded beside the verdict -- never moved to the delivery landmark, and never
    extrapolated towards it.

    A recording already excluded for a labelling reason keeps that reason and is not re-judged.

    Args:
        frame: Retained anchors, after :func:`add_anchor_times`.
        recordings: The recording table from :func:`recording_frame`.
        supervised_hours: The supervised window's upper edge.
        min_late_segments: Contributing segments required inside it.
        final_anchor_within_minutes: How close to delivery a retained anchor must reach.

    Returns:
        A copy of ``recordings`` carrying ``eligible``, ``eligibility_reason``,
        ``n_late_segments``, ``n_late_anchors`` and ``last_anchor_hours``.
    """
    late = frame[in_window(frame[HOURS_COLUMN], 0.0, float(supervised_hours))]
    final_hours = float(final_anchor_within_minutes) / 60.0

    segments_by_guid: Dict[str, int] = {}
    anchors_by_guid: Dict[str, int] = {}
    nearest_by_guid: Dict[str, float] = {}
    for guid, group in late.groupby(GUID_COLUMN, sort=True):
        segments_by_guid[str(guid)] = int(group[EPOCH_COLUMN].nunique())
        anchors_by_guid[str(guid)] = int(len(group))
        nearest_by_guid[str(guid)] = float(
            np.asarray(group[HOURS_COLUMN], dtype=np.float64).min()
        )

    out = recordings.copy()
    eligible: List[bool] = []
    reasons: List[str] = []
    n_segments: List[int] = []
    n_anchors: List[int] = []
    nearest: List[float] = []
    for _, row in out.iterrows():
        guid = str(row[GUID_COLUMN])
        segments = segments_by_guid.get(guid, 0)
        anchors = anchors_by_guid.get(guid, 0)
        closest = nearest_by_guid.get(guid, float("nan"))
        n_segments.append(segments)
        n_anchors.append(anchors)
        nearest.append(closest)

        prior = str(row.get(EXCLUSION_COLUMN, ""))
        if prior:
            eligible.append(False)
            reasons.append(prior)
        elif anchors == 0:
            eligible.append(False)
            reasons.append(INELIGIBLE_NO_LATE_ANCHOR)
        elif segments < int(min_late_segments):
            eligible.append(False)
            reasons.append(INELIGIBLE_FEW_LATE_SEGMENTS)
        elif not (closest <= final_hours + _HOUR_TOLERANCE):
            eligible.append(False)
            reasons.append(INELIGIBLE_NO_FINAL_ANCHOR)
        else:
            eligible.append(True)
            reasons.append("")

    out["n_late_segments"] = n_segments
    out["n_late_anchors"] = n_anchors
    # The real last observation, in hours before delivery. NaN where the recording contributed no
    # retained anchor to the supervised window at all.
    out["last_anchor_hours"] = nearest
    out["eligible"] = eligible
    out["eligibility_reason"] = reasons
    logger.info(
        f"late eligibility: {int(sum(eligible))} of {len(out)} recording(s) eligible; "
        f"reasons {dict(sorted(Counter(r for r in reasons if r).items()))}"
    )
    return out
