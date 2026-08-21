r"""One pass over the loader that says what the evaluation split actually yielded.

This is the pipeline's only real input validator, it runs before anything else, and it needs no
checkpoint, no model and no GPU -- so it also runs on its own:

.. code-block:: bash

    python -m teb_vae.lag_attn_rws.eval.probe --config <run>/model_checkpoints/resolved_config.yaml

It is the artifact that would have caught the predecessor's hardest bug: a loader that silently
truncated its second pass to the first file's index range, which was invisible in every other
output and presented only as "only 1 class found". Nothing else in a run reports per-file
coverage, so nothing else can see that failure.

Four conditions raise rather than warn, because each one silently narrows the population every
headline number is then computed over:

1. **The split yielded nothing.** A path that resolves to no file, or a filter that excluded
   everything.
2. **A configured shard contributed no samples** on an uncapped pass. Under a batch cap this is
   expected instead -- the loader is unshuffled over concatenated per-subgroup files, so a prefix
   of the batches reaches only the first shards -- and is a warning naming the shards it missed.
3. **A required field is absent from the batch.** The loader *skips* a field a shard does not
   carry, with no error, so a missing ``target`` would present as "no classes found" and a
   missing ``epoch`` as "no trajectory data" rather than as a data problem.
4. **A GUID appears in more than one shard.** The holdout split is one pool with no fold loop, so
   a recording in two subgroup files would be counted twice and would put the same delivery on
   both sides of a between-subgroup comparison.

The pass is pure bookkeeping: it reads the batches and reduces them, and it does no forward of
its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Repository root: ``teb_vae/lag_attn_rws/eval/probe.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead
# of the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval import launch  # noqa: E402
from teb_vae.lag_attn_rws.eval._reuse import labels  # noqa: E402
from teb_vae.lag_attn_rws.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
    validate_eval_config,
)

#: File written into the run directory.
PROBE_FILENAME = "loader_probe.json"

#: Fields every evaluation batch must carry. The first eight are the model's own data contract;
#: the last six are what the clinical questions are asked in, and each is silently absent from a
#: batch whose shard does not hold it. ``source_file_basename`` is stamped by the dataset itself
#: rather than requested, and is checked here because every by-subgroup table is keyed on it.
REQUIRED_BATCH_FIELDS: Tuple[str, ...] = (
    "fhr",
    "up",
    "fhr_st",
    "fhr_ph",
    "up_st",
    "up_ph",
    "weight",
    "guid",
    "target",
    "epoch",
    "cs_label",
    "bg_label",
    "time_from_labor_onset",
    "second_stage_onset",
    "source_file_basename",
)

#: Probe keys that are per-sample rather than summary, and therefore never serialised. They are
#: returned in memory for the consumers that need them -- the GUID and source-file vectors every
#: stratified draw and every later ``groupby`` is built from. Written out they would repeat every
#: GUID in the split inside a file meant to be read at a glance.
IN_MEMORY_KEYS: Tuple[str, ...] = ("guids", "source_files")


# =============================================================================
# Reading one batch
# =============================================================================
def _field(batch: Any, name: str) -> Any:
    """Read a batch field by name, tolerating both mapping and attribute access.

    Args:
        batch: A batch from the data module, or a stub.
        name: The field name.

    Returns:
        The field value, or ``None`` when the batch does not carry it.
    """
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def _field_names(batch: Any) -> List[str]:
    """Return the names one batch carries, sorted."""
    if isinstance(batch, dict):
        return sorted(str(key) for key in batch.keys())
    return sorted(name for name in vars(batch) if not name.startswith("_"))


def _to_numpy(values: Any) -> np.ndarray:
    """Return ``values`` as a float64 array, accepting a tensor, an array or a sequence."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64)
    return np.asarray(values, dtype=np.float64)


def _batch_size(batch: Any) -> int:
    """Return how many samples one batch holds.

    Taken from a batched tensor rather than from a collated list, so the count does not depend on
    which optional identifier fields the shard happened to carry.

    Args:
        batch: A batch from the data module.

    Returns:
        The sample count.

    Raises:
        RuntimeError: If the batch carries nothing whose leading dimension can be read.
    """
    for name in ("weight", "target", "fhr", "fhr_st", "up"):
        value = _field(batch, name)
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            return int(value.shape[0])
    for name in ("guid", "source_file_basename"):
        value = _field(batch, name)
        if isinstance(value, (list, tuple)):
            return len(value)
    raise RuntimeError(
        f"cannot determine the batch size: none of the expected fields is present. Fields seen: "
        f"{_field_names(batch)}."
    )


def _basenames_of(batch: Any, batch_size: int) -> List[str]:
    """Return one source-file basename per sample.

    Args:
        batch: A batch from the data module.
        batch_size: The batch's sample count.

    Returns:
        The basenames, ``'unknown'`` where the batch carries none.
    """
    values = _field(batch, "source_file_basename")
    if values is None:
        return ["unknown"] * batch_size
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return [str(values)] * batch_size


def _guids_of(batch: Any, batch_size: int) -> List[str]:
    """Return one GUID per sample. ``guid`` survives collation as a ``list[str]``."""
    values = _field(batch, "guid")
    if values is None:
        return ["unknown"] * batch_size
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return [str(values)] * batch_size


def weight_distribution(weight: Any) -> Dict[str, Any]:
    r"""Summarise a per-step validity tensor.

    ``binary`` is the load-bearing field. If ``weight`` is strictly $\{0, 1\}$ the mask arithmetic
    downstream is an intersection of indicator functions; if it is fractional, every masked mean
    is a weighted mean and must be read as one -- and the model's own
    ``raw_masks.VALID_THRESHOLD`` treats a partially valid step as *invalid*, which is a different
    convention from the one the class recovery uses.

    Args:
        weight: Per-step validity, $(B, T)$.

    Returns:
        Summary statistics, including whether the values are strictly binary.
    """
    values = _to_numpy(weight).ravel()
    unique = np.unique(values)
    return {
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "mean": float(values.mean()) if values.size else float("nan"),
        "zero_frac": float((values == 0.0).mean()) if values.size else float("nan"),
        "n_unique": int(unique.size),
        # Capped: a fractional weight field has as many distinct values as it has entries.
        "unique_head": [float(value) for value in unique[:8]],
        "binary": bool(unique.size <= 2 and np.isin(unique, (0.0, 1.0)).all()),
    }


def target_value_summary(target: Any) -> Dict[str, Any]:
    r"""Summarise the *raw* target values, as distinct from the class histogram.

    The class histogram records one code per recording, which is the right reduction for "which
    classes are present" and the wrong one for "is this field what I think it is". ``target``
    stores the class code scaled by the per-step ``weight``, so a fractional value appears exactly
    at the partially valid boundaries of a segment -- and it is those boundaries that make reading
    ``target`` directly wrong. Counting them here is what says whether the fixture, or the
    production data, actually exercises that case.

    Args:
        target: Per-step target, $(B, T)$.

    Returns:
        The counts behind "any fractional" and "any non-finite".
    """
    values = _to_numpy(target).ravel()
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0.0]
    return {
        "n_values": int(values.size),
        "n_non_finite": int(values.size - finite.size),
        "n_nonzero": int(nonzero.size),
        # NaN != round(NaN), so a non-finite value left in here would masquerade as fractional.
        "n_fractional": int(np.sum(nonzero != np.round(nonzero))),
    }


# =============================================================================
# The pass
# =============================================================================
def run_probe(
    loader: Any,
    *,
    configured_files: Optional[Sequence[str]] = None,
    required_fields: Sequence[str] = REQUIRED_BATCH_FIELDS,
    max_batches: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Iterate the loader once, recording what the split actually yielded.

    Args:
        loader: The evaluation dataloader.
        configured_files: Paths the config asked for. Every basename here must appear in the
            counts with a nonzero count. ``None`` skips that check, which is appropriate only
            when the caller has no config to compare against.
        required_fields: Fields the first batch must carry. Pass an empty sequence to skip.
        max_batches: Stop after this many batches, for a smoke pass. A batch cap is a *prefix*
            over the concatenated per-subgroup index, so a capped pass sees a biased draw and the
            missing-shard check downgrades to a warning.
        output_dir: Where to write ``loader_probe.json``. ``None`` skips the write.

    Returns:
        The probe record. :data:`IN_MEMORY_KEYS` are returned but never serialised.

    Raises:
        RuntimeError: If the split yields nothing, if a required field is absent from the batch,
            if a configured file yields no samples on an uncapped pass, or if a GUID appears in
            more than one shard.
    """
    per_file: Counter = Counter()
    per_cs_label: Counter = Counter()
    per_bg_label: Counter = Counter()
    per_target_class: Counter = Counter()
    weight_summaries: List[Dict[str, Any]] = []
    target_summaries: List[Dict[str, Any]] = []
    shards_by_guid: Dict[str, Set[str]] = defaultdict(set)

    guids: List[str] = []
    sources: List[str] = []
    epochs: List[float] = []
    onsets: List[float] = []
    second_stage: List[float] = []
    fields_seen: List[str] = []
    n_samples = 0
    n_batches = 0

    for batch in loader:
        if max_batches is not None and n_batches >= max_batches:
            break
        n_batches += 1

        if not fields_seen:
            fields_seen = _field_names(batch)
            _check_required_fields(fields_seen, required_fields)

        batch_size = _batch_size(batch)
        basenames = _basenames_of(batch, batch_size)
        batch_guids = _guids_of(batch, batch_size)

        for guid, basename in zip(batch_guids, basenames):
            per_file[basename] += 1
            sources.append(basename)
            guids.append(guid)
            shards_by_guid[guid].add(basename)

        cs_label = _field(batch, "cs_label")
        if cs_label is not None:
            for value in _to_numpy(cs_label).ravel().tolist():
                per_cs_label[str(bool(value))] += 1
        bg_label = _field(batch, "bg_label")
        if bg_label is not None:
            for value in _to_numpy(bg_label).ravel().tolist():
                per_bg_label[str(bool(value))] += 1

        weight = _field(batch, "weight")
        target = _field(batch, "target")
        if target is not None:
            # ``target`` is the class code scaled by the per-step weight, so a partially valid
            # acidosis step stores 1.0 -- indistinguishable from a fully valid healthy step.
            # ``clinical_class_code`` divides the weight back out and returns None for a pad-only
            # or uniformly zero row, so there is no phantom class 0.
            target_rows = _to_numpy(target)
            weight_rows = _to_numpy(weight) if weight is not None else None
            for index, row in enumerate(target_rows):
                row = np.atleast_1d(row).ravel()
                weight_row = (
                    weight_rows[index]
                    if weight_rows is not None and index < len(weight_rows)
                    else np.ones_like(row)
                )
                code = labels.clinical_class_code(row, weight_row)
                per_target_class[str(labels.class_name(code))] += 1
            target_summaries.append(target_value_summary(target))

        if weight is not None:
            weight_summaries.append(weight_distribution(weight))

        epoch = _field(batch, "epoch")
        if epoch is not None:
            epochs.extend(_to_numpy(epoch).ravel().tolist())
        onset = _field(batch, "time_from_labor_onset")
        if onset is not None:
            onsets.extend(_to_numpy(onset).ravel().tolist())
        stage = _field(batch, "second_stage_onset")
        if stage is not None:
            second_stage.extend(_to_numpy(stage).ravel().tolist())

        n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError(
            "the test loader yielded no samples at all. Either "
            "dataset_config.vae_test_datasets resolves to nothing, or a dataset_kwargs filter "
            "(epoch_min / epoch_max / label / cs_label / bg_label) excluded every sample. Check "
            "the resolved config dumped into this run directory."
        )

    _check_no_guid_spans_two_shards(shards_by_guid)

    record: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "n_unique_guids": len(set(guids)),
        "batch_fields": fields_seen,
        "per_file": dict(sorted(per_file.items())),
        "per_cs_label": dict(sorted(per_cs_label.items())),
        "per_bg_label": dict(sorted(per_bg_label.items())),
        "per_target_class": dict(sorted(per_target_class.items())),
        # Distinct from per_target_class: that is one code per recording, this is every value.
        "target_values": _merge_target_summaries(target_summaries),
        "weight": _merge_weight_summaries(weight_summaries),
        "epoch": _epoch_summary(epochs),
        # Only the count: NaN means the recording is absent from the labour-onset table, and a
        # reader needs the denominator before reading any labour-onset number.
        "time_from_labor_onset": {
            "n_values": len(onsets),
            "n_nan": int(np.sum(~np.isfinite(np.asarray(onsets, dtype=np.float64)))) if onsets else 0,
        },
        # The same counted-not-dropped rule, for the second clock. A run whose shards
        # carry the field but never populated it is a run whose second-stage analysis
        # records a skip, and this is where that is visible before the analysis says so.
        "second_stage_onset": {
            "n_values": len(second_stage),
            "n_nan": (
                int(np.sum(~np.isfinite(np.asarray(second_stage, dtype=np.float64))))
                if second_stage else 0
            ),
        },
        "max_batches": max_batches,
    }

    if configured_files is not None:
        _check_every_file_contributed(configured_files, per_file, capped=max_batches is not None)

    logger.info(
        f"loader probe: {n_samples} samples over {n_batches} batches from {len(per_file)} "
        f"file(s), {record['n_unique_guids']} recordings; classes {dict(per_target_class)}"
    )

    if output_dir is not None:
        write_probe(record, output_dir)

    # Returned, not serialised: one row per sample does not belong in a JSON summary.
    record["guids"] = guids
    record["source_files"] = sources
    return record


def _epoch_summary(epochs: Sequence[float]) -> Dict[str, Any]:
    r"""Summarise the per-sample ``epoch``, which is negative seconds before delivery.

    Args:
        epochs: One value per sample.

    Returns:
        The range in seconds and in hours, or an empty dict when the batches carried no ``epoch``.
    """
    if not epochs:
        return {}
    values = np.asarray(epochs, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n_values": int(values.size), "n_non_finite": int(values.size)}
    return {
        "n_values": int(values.size),
        "n_non_finite": int(values.size - finite.size),
        "min_seconds": float(finite.min()),
        "max_seconds": float(finite.max()),
        "mean_seconds": float(finite.mean()),
        # Negative: epoch counts backwards from delivery.
        "min_hours": float(finite.min() / 3600.0),
        "max_hours": float(finite.max() / 3600.0),
    }


def _merge_target_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the per-batch raw-target summaries into one.

    Args:
        summaries: Per-batch summaries from :func:`target_value_summary`.

    Returns:
        The merged summary, or an empty dict when the batches carried no ``target``.
    """
    if not summaries:
        return {}
    totals: Dict[str, Any] = {
        key: int(sum(item[key] for item in summaries))
        for key in ("n_values", "n_non_finite", "n_nonzero", "n_fractional")
    }
    totals["any_fractional"] = bool(totals["n_fractional"] > 0)
    totals["any_non_finite"] = bool(totals["n_non_finite"] > 0)
    return totals


def _merge_weight_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the per-batch weight summaries into one.

    Args:
        summaries: Per-batch summaries from :func:`weight_distribution`.

    Returns:
        The merged summary, or an empty dict when the batches carried no ``weight``.
    """
    if not summaries:
        return {}
    return {
        "min": min(item["min"] for item in summaries),
        "max": max(item["max"] for item in summaries),
        "mean": float(np.mean([item["mean"] for item in summaries])),
        "zero_frac": float(np.mean([item["zero_frac"] for item in summaries])),
        # The question this exists to answer: is `weight` ever fractional in this population?
        "binary": all(item["binary"] for item in summaries),
    }


# =============================================================================
# The four refusals
# =============================================================================
def _check_required_fields(seen: Sequence[str], required: Sequence[str]) -> None:
    """Raise when the batch is missing a field the pipeline reads.

    Args:
        seen: The field names the first batch carried.
        required: The names that must be present.

    Raises:
        RuntimeError: Naming the missing fields.
    """
    missing = [name for name in required if name not in set(seen)]
    if not missing:
        return
    raise RuntimeError(
        f"the batch is missing required field(s): {missing}. The loader SKIPS a field a shard "
        f"does not carry, silently, so this is either an absent "
        f"dataloader_config.dataset_kwargs.load_fields entry or a shard written without the "
        f"field. Fields seen: {list(seen)}. A shard set built before a field existed has two "
        f"recoveries and no third: rebuild it with hdf5_dataset/new_pipeline/create_new_pipeline.py, "
        f"which creates every field named here in every build, or drop the field from load_fields "
        f"and lose the readouts asked in it."
    )


def _check_every_file_contributed(
    configured_files: Sequence[str], per_file: Counter, *, capped: bool
) -> None:
    """Raise when a configured shard contributed no samples.

    Args:
        configured_files: The paths from ``dataset_config.vae_test_datasets``.
        per_file: Observed counts, keyed by basename.
        capped: Whether the pass was capped, in which case a missing file is expected rather than
            an error -- a batch cap reads a prefix of the concatenated index.

    Raises:
        RuntimeError: If an uncapped pass left a configured file at zero.
    """
    expected = [Path(str(path)).name for path in configured_files]
    missing = [name for name in expected if per_file.get(name, 0) == 0]
    if not missing:
        return
    if capped:
        logger.warning(
            f"{len(missing)} configured file(s) contributed no samples under the active batch "
            f"cap: {missing}. A capped pass reads a prefix of the concatenated index, so this is "
            f"expected -- but it does mean the pass saw a biased draw."
        )
        return
    raise RuntimeError(
        f"configured shard(s) yielded zero samples: {missing}. Observed counts: {dict(per_file)}. "
        f"Either the path does not resolve, or a dataset_kwargs filter excluded the whole file. "
        f"Every headline number would otherwise be computed over a silently narrowed population."
    )


def _check_no_guid_spans_two_shards(shards_by_guid: Dict[str, Set[str]]) -> None:
    """Raise when one recording appears in more than one holdout shard.

    Args:
        shards_by_guid: GUID to the set of shard basenames it was seen in.

    Raises:
        RuntimeError: Naming the offending GUIDs and their shards.
    """
    spanning = {
        guid: sorted(shards) for guid, shards in shards_by_guid.items() if len(shards) > 1
    }
    if not spanning:
        return
    shown = dict(sorted(spanning.items())[:5])
    raise RuntimeError(
        f"{len(spanning)} recording(s) appear in more than one shard, e.g. {shown}. The holdout "
        f"split is one pool with no fold loop, so a GUID in two subgroup files is counted twice "
        f"and lands on both sides of every between-subgroup comparison. Either "
        f"dataset_config.vae_test_datasets lists overlapping files, or the split was built in "
        f"the pipeline's per-fold 'augmented' mode rather than as a single holdout."
    )


# =============================================================================
# Output
# =============================================================================
def summary_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the probe record without its per-sample vectors.

    Args:
        record: The full probe record.

    Returns:
        The summary-safe subset.
    """
    return {key: value for key, value in record.items() if key not in IN_MEMORY_KEYS}


def write_probe(record: Dict[str, Any], output_dir: Any) -> Path:
    """Write ``loader_probe.json``.

    Args:
        record: The probe record. :data:`IN_MEMORY_KEYS` are skipped.
        output_dir: The directory to write into, created if absent.

    Returns:
        The path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROBE_FILENAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary_view(record), handle, indent=2, default=str)
    logger.info(f"wrote {path}")
    return path


def read_probe(output_dir: Any) -> Optional[Dict[str, Any]]:
    """Read back a probe record a previous pass over this directory wrote.

    What makes an offline re-run answer the population questions at all. The pass itself is one
    full iteration of the loader, and a run that is reusing a finished directory's tables has
    deliberately not built a loader; the record it left behind carries everything the sanity
    checks read.

    The per-sample vectors of :data:`IN_MEMORY_KEYS` are **not** in the file and do not come back.
    A consumer that needs them needs the pass, not the record.

    Args:
        output_dir: The results directory.

    Returns:
        The record, or ``None`` when the directory holds none.
    """
    path = Path(output_dir) / PROBE_FILENAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def format_cohort_table(record: Dict[str, Any]) -> str:
    """Render the probe record as the table the command line prints.

    Args:
        record: The probe record.

    Returns:
        A multi-line string. Every count is shown beside the total it is a part of, so a reader
        never has to divide two numbers from different blocks to learn the coverage.
    """
    total = int(record.get("n_samples", 0))
    lines = [
        "cohort",
        f"  samples          {total}",
        f"  batches          {record.get('n_batches', 0)}",
        f"  recordings       {record.get('n_unique_guids', 0)}",
    ]

    for heading, key in (
        ("per shard", "per_file"),
        ("per class", "per_target_class"),
        ("per cs_label", "per_cs_label"),
        ("per bg_label", "per_bg_label"),
    ):
        counts = record.get(key) or {}
        lines.append(f"{heading}")
        if not counts:
            lines.append("  (not reported)")
            continue
        for name, count in counts.items():
            share = 100.0 * float(count) / float(total) if total else float("nan")
            lines.append(f"  {str(name):<32} {int(count):>6}  ({share:5.1f}%)")

    weight = record.get("weight") or {}
    if weight:
        lines.append("weight")
        lines.append(
            f"  binary {weight['binary']}   zero_frac {weight['zero_frac']:.4f}   "
            f"range [{weight['min']:.3f}, {weight['max']:.3f}]"
        )
    epoch = record.get("epoch") or {}
    if epoch and "min_hours" in epoch:
        lines.append("epoch (hours before delivery)")
        lines.append(f"  [{epoch['min_hours']:.2f}, {epoch['max_hours']:.2f}]")
    onset = record.get("time_from_labor_onset") or {}
    if onset:
        lines.append(
            f"time_from_labor_onset: {onset['n_nan']} of {onset['n_values']} absent (NaN)"
        )
    stage = record.get("second_stage_onset") or {}
    if stage:
        lines.append(
            f"second_stage_onset: {stage['n_nan']} of {stage['n_values']} absent (NaN)"
        )
    return "\n".join(lines)


# =============================================================================
# Entry point
# =============================================================================
def probe_config(
    config: Dict[str, Any],
    *,
    max_batches: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the evaluation loader from a merged config and probe it.

    Args:
        config: A merged, validated run config.
        max_batches: Batch cap for a smoke pass.
        output_dir: Where to write ``loader_probe.json``.

    Returns:
        The probe record.
    """
    from train.data_module import GraphDataModule

    loader = GraphDataModule(config).test_dataloader()
    return run_probe(
        loader,
        configured_files=(config.get("dataset_config") or {}).get("vae_test_datasets"),
        max_batches=max_batches,
        output_dir=output_dir,
    )


def main(
    config_path: Any,
    *,
    overrides: Optional[Any] = None,
    max_batches: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a config, merge the evaluation overrides over it, and probe the split.

    Args:
        config_path: A run config -- normally the ``resolved_config.yaml`` a training run wrote
            beside its checkpoints.
        overrides: The override delta. Defaults to the committed one.
        max_batches: Batch cap for a smoke pass.
        output_dir: Where to write ``loader_probe.json``.

    Returns:
        The probe record.
    """
    config = merge_eval_overrides(load_config(str(config_path)), overrides)
    force_single_process_loader(config)
    # Validated even though the probe reads none of it: a misspelled eval_config key must fail on
    # the cheapest command that touches it, not four analyses into a full run.
    config["eval_config"] = validate_eval_config(config)
    return probe_config(config, max_batches=max_batches, output_dir=output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_rws.eval.probe",
        description="Report what the evaluation split actually yields. No model, no checkpoint.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Run config to probe; normally the resolved_config.yaml beside the checkpoints. "
             "Required, but not by argparse: it is enforced after the launch dict is merged, "
             "because required=True fires before RUN_ARGS is ever consulted.",
    )
    parser.add_argument(
        "--overrides", default=None,
        help="Evaluation override delta. Default: the committed eval_overrides.yaml.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help=f"Directory to write {PROBE_FILENAME} into. Default: no file, table only.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop after this many batches. A prefix over the concatenated shards; smoke only.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    values, sources = launch.resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = launch.missing_required(values, ("config",))
    if refusal:
        raise SystemExit(refusal)
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        # Shard paths inside a config are repo-root-relative for the tiny variant, and a relative
        # path resolved against an arbitrary working directory surfaces as "no samples match the
        # specified filters" with no mention of the real cause.
        logger.info(f"changing working directory to the repo root: {_REPO_ROOT}")
        os.chdir(_REPO_ROOT)
    logger.info(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    record = main(
        values["config"],
        overrides=values["overrides"],
        max_batches=values["max_batches"],
        output_dir=values["output_dir"],
    )
    print(format_cohort_table(record))
    return 0


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so a flag overrides one value and leaves the
#: rest standing, and a key that is not an argparse ``dest`` raises at startup.
#:
#: **Running this file directly needs ``config`` filled in below**: the probe reads a run config
#: and reports what its evaluation split actually yields, and there is no default worth guessing.
#: Point it at the ``resolved_config.yaml`` beside the checkpoints. Everything else is optional --
#: ``overrides`` defaults to the committed delta, and omitting ``output_dir`` reports without
#: writing.
RUN_ARGS: Dict[str, Any] = {
    "config": None,
    "overrides": None,
    "output_dir": None,
    "max_batches": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
