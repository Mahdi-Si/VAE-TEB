r"""What the evaluation split yields, and what the forward actually returns when handed it.

Two passes, one module, and they answer two different questions. Both run before anything is
scored, and neither writes a number any analysis reads.

**The population pass** is the sibling's, unchanged in behaviour: one iteration of the loader,
recording per-shard, per-class and per-label counts into ``loader_probe.json``. It is the artifact
that would have caught the predecessor's hardest bug -- a loader that silently truncated its second
pass to the first file's index range, invisible in every other output and presenting only as "only 1
class found". Nothing else in a run reports per-file coverage, so nothing else can see that failure.
It needs no checkpoint, no model and no GPU:

.. code-block:: bash

    python -m teb_vae.lag_attn_cfs.eval.probe --config <run>/model_checkpoints/resolved_config.yaml

Four conditions raise rather than warn, because each one silently narrows the population every
headline number is then computed over:

1. **The split yielded nothing.** A path that resolves to no file, or a filter that excluded
   everything.
2. **A configured shard contributed no samples** on an uncapped pass. Under a batch cap this is
   expected instead -- the loader is unshuffled over concatenated per-subgroup files, so a prefix of
   the batches reaches only the first shards -- and is a warning naming the shards it missed.
3. **A required field is absent from the batch.** The loader *skips* a field a shard does not carry,
   with no error, so a missing ``target`` would present as "no classes found" and a missing ``epoch``
   as "no trajectory data" rather than as a data problem.
4. **A GUID appears in more than one shard.** The holdout split is one pool with no fold loop, so a
   recording in two subgroup files would be counted twice and would put the same delivery on both
   sides of a between-subgroup comparison.

**The forward-contract pass is this cell's own, and is why a ``--checkpoint`` flag exists here where
the sibling's probe has none.** This model's forward takes **five** arguments rather than three and
*raises* without a phase once the configured stride exceeds $1$; it returns two keys the family's
does not; and its four forecast tensors are $(B, A_{\max}, H, C_{\mathrm{keep}})$ rather than
$(B, T_{\mathrm{valid}}, H, R)$. Every one of those is a fact the readout module has to be written
against, so it is *measured* here first and read from a printout rather than from a source file:

.. code-block:: bash

    python -m teb_vae.lag_attn_cfs.eval.probe --checkpoint <run>/model_checkpoints/<name>.ckpt

The batch is assembled through the task's own ``_build_forward_inputs``, which is the seam the
training step and the diagnostic callback both use -- so what is probed is the call the model is
actually given, not a hand-built approximation of it. Outside a step that seam resolves the **dense**
geometry, $(\varphi, S) = (0, 1)$, which is exactly the geometry the evaluation decodes at; the pass
refuses anything else rather than reporting a contract at a tiling no evaluation uses.

This module deliberately imports no ``metrics``: it is what the edits to that module are made
against, so it has to run before that module exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/eval/probe.py`` -> up four.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Launched as a script (an IDE's Run button) this file's own directory goes on sys.path instead of
# the repository root, and every absolute import below fails before __main__ is reached.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval import launch, preflight  # noqa: E402
from teb_vae.lag_attn_cfs.eval._reuse import labels  # noqa: E402
from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING, ModelBinding  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import (  # noqa: E402
    force_single_process_loader,
    merge_eval_overrides,
    validate_eval_config,
)
from teb_vae.lag_attn_rws.trainer import RESOLVED_CONFIG_FILENAME  # noqa: E402
from train.graph_models_utils import check_model_class, load_checkpoint_strict  # noqa: E402

#: File written into the run directory by the population pass.
PROBE_FILENAME = "loader_probe.json"

#: Fields every evaluation batch must carry. The first eight are the model's own data contract; the
#: next five are what the clinical questions are asked in, and each is silently absent from a batch
#: whose shard does not hold it. ``source_file_basename`` is stamped by the dataset itself rather
#: than requested, and is checked here because every by-subgroup table is keyed on it.
#:
#: ``fhr`` and ``up`` are on the list although this model reads neither: the target is
#: ``fhr_st`` / ``fhr_ph`` and the source reaches it as ``up_st`` / ``up_ph``. They are here because
#: the contraction detector reads the raw ``up`` trace and the diagnostic page's first row draws the
#: raw ``fhr``, and a contraction exists nowhere in the tables unless the one pass holding the raw
#: trace puts it there.
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
    "source_file_basename",
)

#: Probe keys that are per-sample rather than summary, and therefore never serialised. They are
#: returned in memory for the consumers that need them -- the GUID and source-file vectors every
#: stratified draw and every later ``groupby`` is built from. Written out they would repeat every
#: GUID in the split inside a file meant to be read at a glance.
IN_MEMORY_KEYS: Tuple[str, ...] = ("guids", "source_files")

#: The anchor geometry the evaluation decodes at, and the only one this probe reports a contract for.
#: ``SeqVaeLagAttnCfsTask.resolve_anchor_geometry`` returns exactly this pair on ``val`` and
#: ``test``; the training tiling exists for gradient decorrelation and activation memory, neither of
#: which applies where there is no backward pass.
DENSE_ANCHOR_GEOMETRY: Tuple[int, int] = (0, 1)


# =================================================================================================
# Reading one batch
# =================================================================================================
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
    downstream is an intersection of indicator functions; if it is fractional, every masked mean is a
    weighted mean and must be read as one -- and the model's own ``raw_masks.VALID_THRESHOLD`` treats
    a partially valid step as *invalid*, which is a different convention from the one the class
    recovery uses.

    It matters more here than in the raw cells: these coefficients carry no gap sentinel of their
    own, so ``weight`` is the only trustworthy validity signal and it gates every mask, every
    baseline and every event.

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
    classes are present" and the wrong one for "is this field what I think it is". ``target`` stores
    the class code scaled by the per-step ``weight``, so a fractional value appears exactly at the
    partially valid boundaries of a segment -- and it is those boundaries that make reading
    ``target`` directly wrong. Counting them here is what says whether the fixture, or the production
    data, actually exercises that case.

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


# =================================================================================================
# The population pass
# =================================================================================================
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
        configured_files: Paths the config asked for. Every basename here must appear in the counts
            with a nonzero count. ``None`` skips that check, which is appropriate only when the
            caller has no config to compare against.
        required_fields: Fields the first batch must carry. Pass an empty sequence to skip.
        max_batches: Stop after this many batches, for a smoke pass. A batch cap is a *prefix* over
            the concatenated per-subgroup index, so a capped pass sees a biased draw and the
            missing-shard check downgrades to a warning.
        output_dir: Where to write ``loader_probe.json``. ``None`` skips the write.

    Returns:
        The probe record. :data:`IN_MEMORY_KEYS` are returned but never serialised.

    Raises:
        RuntimeError: If the split yields nothing, if a required field is absent from the batch, if a
            configured file yields no samples on an uncapped pass, or if a GUID appears in more than
            one shard.
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
            # ``clinical_class_code`` divides the weight back out and returns None for a pad-only or
            # uniformly zero row, so there is no phantom class 0.
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

        n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError(
            "the test loader yielded no samples at all. Either dataset_config.vae_test_datasets "
            "resolves to nothing, or a dataset_kwargs filter (epoch_min / epoch_max / label / "
            "cs_label / bg_label) excluded every sample. Check the resolved config dumped into this "
            "run directory."
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
        # Only the count: NaN means the recording is absent from the labour-onset table, and a reader
        # needs the denominator before reading any labour-onset number.
        "time_from_labor_onset": {
            "n_values": len(onsets),
            "n_nan": (
                int(np.sum(~np.isfinite(np.asarray(onsets, dtype=np.float64)))) if onsets else 0
            ),
        },
        "max_batches": max_batches,
    }

    if configured_files is not None:
        _check_every_file_contributed(configured_files, per_file, capped=max_batches is not None)

    logger.info(
        f"loader probe: {n_samples} samples over {n_batches} batches from {len(per_file)} file(s), "
        f"{record['n_unique_guids']} recordings; classes {dict(per_target_class)}"
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


# =================================================================================================
# The four refusals
# =================================================================================================
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
        f"the batch is missing required field(s): {missing}. The loader SKIPS a field a shard does "
        f"not carry, silently, so this is either an absent "
        f"dataloader_config.dataset_kwargs.load_fields entry or a shard written without the field. "
        f"Fields seen: {list(seen)}."
    )


def _check_every_file_contributed(
    configured_files: Sequence[str], per_file: Counter, *, capped: bool
) -> None:
    """Raise when a configured shard contributed no samples.

    Args:
        configured_files: The paths from ``dataset_config.vae_test_datasets``.
        per_file: Observed counts, keyed by basename.
        capped: Whether the pass was capped, in which case a missing file is expected rather than an
            error -- a batch cap reads a prefix of the concatenated index.

    Raises:
        RuntimeError: If an uncapped pass left a configured file at zero.
    """
    expected = [Path(str(path)).name for path in configured_files]
    missing = [name for name in expected if per_file.get(name, 0) == 0]
    if not missing:
        return
    if capped:
        logger.warning(
            f"{len(missing)} configured file(s) contributed no samples under the active batch cap: "
            f"{missing}. A capped pass reads a prefix of the concatenated index, so this is "
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
        f"split is one pool with no fold loop, so a GUID in two subgroup files is counted twice and "
        f"lands on both sides of every between-subgroup comparison. Either "
        f"dataset_config.vae_test_datasets lists overlapping files, or the split was built in the "
        f"pipeline's per-fold 'augmented' mode rather than as a single holdout."
    )


# =================================================================================================
# The forward-contract pass
# =================================================================================================
def resolved_config_for(checkpoint: Any) -> Path:
    """Find the resolved config the training run wrote for this checkpoint.

    A run's layout is ``<out_dir_base>/<stamp>-<tag>/model_checkpoints/<name>.ckpt``, and the config
    is written into that checkpoint directory. The run root and its ``train_results`` are searched as
    well, so a config placed there by hand is still found.

    Args:
        checkpoint: Path to the checkpoint being probed.

    Returns:
        Path to the resolved config.

    Raises:
        FileNotFoundError: Naming every location tried. A checkpoint moved out of its run directory
            has lost the record of what it was trained on, and probing it against a guessed
            configuration is worse than not probing it.
    """
    checkpoint = Path(checkpoint)
    run_root = checkpoint.parent.parent
    candidates = [
        checkpoint.parent / RESOLVED_CONFIG_FILENAME,
        run_root / "train_results" / RESOLVED_CONFIG_FILENAME,
        run_root / RESOLVED_CONFIG_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no {RESOLVED_CONFIG_FILENAME} found for checkpoint {checkpoint}. Tried: "
        + ", ".join(str(path) for path in candidates)
        + f". The training entry point writes it beside the checkpoints; a checkpoint copied out of "
        f"its run directory must be copied together with that file."
    )


def read_checkpoint(checkpoint_path: Any) -> Dict[str, Any]:
    """Read a checkpoint blob off disk.

    Args:
        checkpoint_path: Path to the ``.ckpt``.

    Returns:
        The blob.

    Raises:
        FileNotFoundError: If the checkpoint is not there.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def resolve_device(requested: Optional[str]) -> torch.device:
    """Resolve the device a forward is run on.

    Args:
        requested: An explicit device string, or ``None`` to choose automatically.

    Returns:
        The device: what was asked for, else CUDA when available, else CPU.
    """
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_task(
    checkpoint_path: Any,
    device: torch.device,
    *,
    blob: Optional[Dict[str, Any]] = None,
    binding: ModelBinding = CFS_BINDING,
) -> Any:
    """Rebuild the net and its task from a checkpoint, and load the weights.

    The order is load-bearing: the class guard runs before construction, because the net's
    constructor is keyword-only and another model's ``model_kwargs`` would otherwise surface as a
    ``TypeError`` naming a parameter rather than as a message naming both classes.

    The **task** rather than the net alone, because the batch is assembled through
    ``task._build_forward_inputs``: this model's forward takes five arguments, two of which are the
    anchor geometry, and re-deriving them here would be a second implementation of the seam that
    exists to stop exactly that.

    Args:
        checkpoint_path: Path to the checkpoint.
        device: Device to place the model on.
        blob: An already-read checkpoint, so a caller that needed its ``model_kwargs`` before
            construction does not pay a second read. Read from ``checkpoint_path`` when omitted.
        binding: The model to rebuild. Every refusal below names ``binding.model_cls``, so a probe of
            another architecture reports the class it was actually asked for rather than this one's.

    Returns:
        The task, in evaluation mode on ``device``.

    Raises:
        FileNotFoundError: If the checkpoint is not there.
        RuntimeError: If it carries no ``model_kwargs``, no ``hyper_parameters``, or if its state
            dict does not align into the rebuilt model. ``load_checkpoint_strict`` returns ``None``
            rather than raising, so an unchecked call would probe randomly initialised weights.
    """
    checkpoint_path = Path(checkpoint_path)
    model_name = binding.model_cls.__name__
    if blob is None:
        blob = read_checkpoint(checkpoint_path)
    check_model_class(blob, model_name)

    model_kwargs = blob.get("model_kwargs") if isinstance(blob, dict) else None
    if not model_kwargs:
        raise RuntimeError(
            f"checkpoint {str(checkpoint_path)!r} carries no 'model_kwargs', so the architecture "
            f"cannot be rebuilt. {model_name}() with no arguments builds the production geometry "
            f"rather than raising -- and it builds it UNGATED, with no keep-index and no warm-up "
            f"mask -- so guessing would silently probe the wrong model."
        )
    hparams = blob.get("hyper_parameters") if isinstance(blob, dict) else None
    if not hparams:
        raise RuntimeError(
            f"checkpoint {str(checkpoint_path)!r} carries no 'hyper_parameters', so the likelihood "
            f"and loss weights the run trained under are unknown; scoring it under assumed defaults "
            f"would report a different objective's numbers."
        )

    model = binding.model_cls(**model_kwargs)
    task = binding.task_cls(
        model,
        model_kwargs=model_kwargs,
        beta_schedule=hparams.get("beta_schedule"),
        kld_beta=hparams.get("kld_beta", 1.0),
        # Defaulted for checkpoints that predate the prior anchor; a checkpoint trained at a
        # non-zero weight must be scored under that same objective.
        beta_prior=hparams.get("beta_prior", 0.0),
        lambda_full=hparams.get("lambda_full", 1.0),
        lambda_base=hparams.get("lambda_base", 1.0),
        likelihood=hparams.get("likelihood", "gaussian_nll"),
        free_bits=hparams.get("free_bits", 0.0),
    )
    if load_checkpoint_strict(model=task.orig_model, checkpoint=blob) is None:
        raise RuntimeError(
            f"could not align checkpoint {str(checkpoint_path)!r} into {model_name}: no module "
            f"matched its state dict. Probing would otherwise proceed on randomly initialised "
            f"weights and report the shapes as though they came from the trained model."
        )
    task.to(device)
    task.eval()
    return task


def _tensor_record(value: Any) -> Dict[str, Any]:
    """Describe one forward output: its shape and dtype, or its scalar value.

    Args:
        value: A tensor from the forward dict.

    Returns:
        ``{'shape', 'dtype'}``, with ``'value'`` added for the zero-dimensional diagnostics.
    """
    if not isinstance(value, torch.Tensor):
        return {"shape": None, "dtype": type(value).__name__}
    record: Dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
    }
    if value.dim() == 0:
        record["value"] = float(value)
    return record


def forward_contract(task: Any, batch: Any) -> Dict[str, Any]:
    r"""Run one batch through the forward and report what it returned.

    Measured rather than read: the readout module is built against this contract, and a contract read
    off a source file is one that can be read wrong. Every number here comes from the
    tensors the model produced on a real batch.

    The batch is assembled through ``task._build_forward_inputs``, the seam the training step and the
    diagnostic callback both use. Outside a step that seam resolves the dense geometry
    $(\varphi, S) = (0, 1)$, which is what the evaluation decodes at; anything else is refused rather
    than reported, because a contract measured at a training tiling would name an $A_{\max}$ no
    evaluation ever produces.

    Args:
        task: The rebuilt task, in evaluation mode.
        batch: One batch from the evaluation loader.

    Returns:
        The contract record: the returned keys with their shapes, the anchor set, the block geometry,
        the resolved warm-up widths and the measured lag support.

    Raises:
        RuntimeError: If the seam resolved anything but the dense anchor geometry.
    """
    model = task.orig_model
    inputs = task._build_forward_inputs(batch)
    phase, stride = inputs[3], inputs[4]
    dense_phase, dense_stride = DENSE_ANCHOR_GEOMETRY
    # The phase is a scalar on the dense stages and a per-sample ``(B,)`` tensor at the training
    # tiling, so it is flattened rather than cast: ``int()`` on the tensor form raises a ValueError
    # about element counts, which is a refusal nobody could act on.
    phases = (
        [int(value) for value in phase.reshape(-1).tolist()]
        if isinstance(phase, torch.Tensor)
        else [int(phase)]
    )
    if int(stride) != dense_stride or any(value != dense_phase for value in phases):
        raise RuntimeError(
            f"_build_forward_inputs resolved (anchor_phase, anchor_stride) = "
            f"({sorted(set(phases))}, {int(stride)}), not the dense {DENSE_ANCHOR_GEOMETRY}. The "
            f"evaluation decodes densely and this probe reports the contract it decodes at; a "
            f"contract measured at the training tiling would name an A_max no evaluation run "
            f"produces. The task's stage defaults to one of DENSE_STAGES outside a step, so this "
            f"means the stage on the task was left set by a step that did not finish."
        )

    with torch.no_grad():
        outputs = model(*inputs)

    anchors, valid = outputs["anchor_index"], outputs["anchor_valid"]
    per_row_valid = valid.sum(dim=1)
    return {
        "batch_size": int(anchors.shape[0]),
        "anchor_phase": dense_phase,
        "anchor_stride": dense_stride,
        "n_output_keys": len(outputs),
        "outputs": {name: _tensor_record(value) for name, value in sorted(outputs.items())},
        "anchor_index": {
            "a_max": int(anchors.shape[1]),
            "first": int(anchors[0, 0]),
            "last": int(anchors[0, -1]),
            "dtype": str(anchors.dtype).replace("torch.", ""),
            "n_valid_min": int(per_row_valid.min()),
            "n_valid_max": int(per_row_valid.max()),
            # Padded slots repeat the row's last real anchor and are False in `anchor_valid`, so a
            # duplicate among the *valid* entries would be a real defect rather than the convention.
            "n_distinct_valid_first_row": int(torch.unique(anchors[0][valid[0]]).numel()),
        },
        "block": {
            "horizon": int(model.horizon),
            "decoder_out_channels": int(model.decoder_out_channels),
            "block_width": int(model.horizon) * int(model.decoder_out_channels),
        },
        "warmup_budget": {
            "target_declared_width": int(model.c_y),
            "target_kept_width": int(model.decoder_out_channels),
            "target_max_warmup_steps": max(model.target_warmup_steps or (0,)),
            "source_declared_width": int(model.c_u),
            "source_max_warmup_steps": max(model.source_warmup_steps or (0,)),
            "target_warm_frac": float(model.target_warm_frac),
        },
        "anchor_geometry": preflight.anchor_geometry(model),
        "lag_support": preflight.lag_support(model),
    }


def format_forward_contract(record: Dict[str, Any]) -> str:
    """Render the forward-contract record as the block the command line prints.

    Args:
        record: The record from :func:`forward_contract`.

    Returns:
        A multi-line string. Every shape is printed beside its key, in sorted order, so the
        twenty-two-key dict can be diffed against a later run's by eye.
    """
    anchors = record["anchor_index"]
    block = record["block"]
    budget = record["warmup_budget"]
    support = record["lag_support"]
    geometry = record["anchor_geometry"]

    lines = [
        "forward contract",
        f"  called at         anchor_phase={record['anchor_phase']}, "
        f"anchor_stride={record['anchor_stride']} (dense)",
        f"  batch size        {record['batch_size']}",
        f"  returned keys     {record['n_output_keys']}",
        "outputs",
    ]
    for name, entry in record["outputs"].items():
        shape = "scalar" if entry["shape"] == [] else str(entry["shape"])
        value = f"  = {entry['value']:.6g}" if "value" in entry else ""
        lines.append(f"  {name:<24} {shape:<28} {entry['dtype']}{value}")

    lines += [
        "anchor set",
        f"  A_max             {anchors['a_max']}",
        f"  first / last      {anchors['first']} / {anchors['last']}",
        f"  valid per row     {anchors['n_valid_min']}-{anchors['n_valid_max']} "
        f"({anchors['n_distinct_valid_first_row']} distinct in row 0)",
        f"  floor F           {geometry['anchor_floor']}   T_valid {geometry['t_valid']}   "
        f"training stride {geometry['training_stride']}",
        "forecast block",
        f"  H x C_keep        {block['horizon']} x {block['decoder_out_channels']} = "
        f"{block['block_width']}",
        "warm-up budget",
        f"  target            {budget['target_kept_width']}/{budget['target_declared_width']} "
        f"channels, slowest survivor waits {budget['target_max_warmup_steps']} steps",
        f"  source            {budget['source_declared_width']} channels (ungated), slowest waits "
        f"{budget['source_max_warmup_steps']} steps",
        f"  target_warm_frac  {budget['target_warm_frac']:.6g}",
        "lag support (measured)",
        f"  min anchor - (L-1) - lag_floor = {support['min_decoded_anchor']} - "
        f"{support['max_lag']} - {support['lag_floor']} = "
        f"{support['lag_support_margin_steps']}",
        f"  every lag valid at every anchor: {support['every_lag_valid_at_every_anchor']}",
    ]
    return "\n".join(lines)


def format_causality(record: Dict[str, Any]) -> str:
    """Render the causality disclosure as the block the command line prints.

    Args:
        record: The record from
            :func:`~teb_vae.lag_attn_cfs.eval.preflight.causality_disclosure`.

    Returns:
        A multi-line string: the statement, then the per-block group delays it points at.
    """
    lines = ["causality", "  " + record["statement"], "group delay (s, from the shards)"]
    delays = record.get("group_delay_seconds") or {}
    for name, entry in delays.items():
        if name == "source":
            lines.append(f"  read from         {entry}")
            continue
        lines.append(
            f"  {name:<17} min {entry['min']:.1f}   median {entry['median']:.1f}   "
            f"max {entry['max']:.1f}   ({entry['n_channels']} channels)"
        )
    return "\n".join(lines)


# =================================================================================================
# Output
# =================================================================================================
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

    What makes an offline re-run answer the population questions at all. The pass itself is one full
    iteration of the loader, and a run that is reusing a finished directory's tables has deliberately
    not built a loader; the record it left behind carries everything the sanity checks read.

    The per-sample vectors of :data:`IN_MEMORY_KEYS` are **not** in the file and do not come back. A
    consumer that needs them needs the pass, not the record.

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
    """Render the population record as the table the command line prints.

    Args:
        record: The probe record.

    Returns:
        A multi-line string. Every count is shown beside the total it is a part of, so a reader never
        has to divide two numbers from different blocks to learn the coverage.
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
    return "\n".join(lines)


# =================================================================================================
# Entry point
# =================================================================================================
def _test_loader(config: Dict[str, Any]) -> Any:
    """Build the evaluation dataloader.

    Imported lazily so that importing this module -- which the layering walk does, and which a test
    of the forward contract does -- costs no data module.

    Args:
        config: A merged, validated run config.

    Returns:
        The test dataloader.
    """
    from train.data_module import GraphDataModule

    return GraphDataModule(config).test_dataloader()


def main(
    config: Optional[Any] = None,
    *,
    checkpoint: Optional[Any] = None,
    overrides: Optional[Any] = None,
    max_batches: Optional[int] = None,
    output_dir: Optional[Any] = None,
    device: Optional[str] = None,
    binding: ModelBinding = CFS_BINDING,
) -> Dict[str, Any]:
    """Probe a split, and -- given a checkpoint -- the forward contract it will be read through.

    Args:
        config: A run config. Defaults to the ``resolved_config.yaml`` beside *checkpoint*, which is
            the record of what that run was trained on; a guessed configuration is worse than none.
        checkpoint: A trained checkpoint. Given one, the forward-contract pass runs as well.
        overrides: The override delta. Defaults to the binding's committed one.
        max_batches: Batch cap for a smoke pass over the population.
        output_dir: Where to write ``loader_probe.json``.
        device: Device for the forward pass. ``None`` picks CUDA when available.
        binding: The model to rebuild, for a probe of the second cfs cell.

    Returns:
        ``{'population': ..., 'forward': ..., 'causality': ...}``. The last two are ``None`` when no
        checkpoint was given.

    Raises:
        ValueError: If neither a config nor a checkpoint was given.
    """
    if config is None and checkpoint is None:
        raise ValueError(
            "probe needs either --config (a run config to probe the split of) or --checkpoint (a "
            "trained checkpoint, whose resolved_config.yaml is read from beside it)."
        )
    config_path = Path(config) if config is not None else resolved_config_for(checkpoint)

    merged = merge_eval_overrides(load_config(str(config_path)), overrides or binding.overrides_path)
    force_single_process_loader(merged)
    # Validated even though the population pass reads none of it: a misspelled eval_config key must
    # fail on the cheapest command that touches it, not four analyses into a full run.
    merged["eval_config"] = validate_eval_config(merged)

    loader = _test_loader(merged)
    population = run_probe(
        loader,
        configured_files=(merged.get("dataset_config") or {}).get("vae_test_datasets"),
        max_batches=max_batches,
        output_dir=output_dir,
    )

    forward: Optional[Dict[str, Any]] = None
    causality: Optional[Dict[str, Any]] = None
    if checkpoint is not None:
        task = load_task(checkpoint, resolve_device(device), binding=binding)
        # One batch, taken from the same loader the population pass just walked. A second iteration
        # of an unshuffled loader yields the same first batch, so the contract is reported for a
        # sample the counts above describe.
        forward = forward_contract(task, next(iter(loader)))
        causality = preflight.causality_disclosure(
            merged, task.orig_model, binding.encoder_disclosure
        )

    return {"population": population, "forward": forward, "causality": causality}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m teb_vae.lag_attn_cfs.eval.probe",
        description=(
            "Report what the evaluation split yields, and -- with --checkpoint -- what this model's "
            "forward returns when handed one of its batches."
        ),
    )
    parser.add_argument(
        "--config", default=None,
        help="Run config to probe; normally the resolved_config.yaml beside the checkpoints. "
             "Defaults to the one beside --checkpoint. Required only when neither is given, and "
             "enforced after the launch dict is merged, because required=True fires before RUN_ARGS "
             "is ever consulted.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Trained checkpoint. Given one, the forward contract and the causality disclosure are "
             "reported as well as the population.",
    )
    parser.add_argument(
        "--overrides", default=None,
        help="Evaluation override delta. Default: the committed eval_overrides.yaml.",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help=f"Directory to write {PROBE_FILENAME} into. Default: no file, tables only.",
    )
    parser.add_argument(
        "--max-batches", dest="max_batches", type=int, default=None,
        help="Stop the population pass after this many batches. A prefix over the concatenated "
             "shards; smoke only.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for the forward pass. Default: cuda:0 when available, else cpu.",
    )
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run. Returns the process exit code."""
    values, sources = launch.resolve_launch_args(build_parser(), RUN_ARGS, argv)
    if values["config"] is None and values["checkpoint"] is None:
        # Not `launch.missing_required`, which requires every name it is given: exactly one of these
        # two is needed, and a message demanding both would send an operator to supply a config they
        # do not have.
        raise SystemExit(
            "one of --checkpoint or --config is required. Pass it on the command line, or -- to "
            "launch this file from an IDE's Run button -- set 'checkpoint' or 'config' in RUN_ARGS "
            "near the bottom of this module. --checkpoint reads the resolved_config.yaml beside it, "
            "so it is the one to fill in for the full report."
        )
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
        checkpoint=values["checkpoint"],
        overrides=values["overrides"],
        max_batches=values["max_batches"],
        output_dir=values["output_dir"],
        device=values["device"],
    )
    print(format_cohort_table(record["population"]))
    if record["forward"] is not None:
        print(format_forward_contract(record["forward"]))
    if record["causality"] is not None:
        print(format_causality(record["causality"]))
    return 0


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button.
#:
#: Keyed by argparse ``dest``. Resolution is per key, so a flag overrides one value and leaves the
#: rest standing, and a key that is not an argparse ``dest`` raises at startup.
#:
#: **Running this file directly needs ``checkpoint`` OR ``config`` filled in below.** With
#: ``checkpoint`` the report is complete -- the population, the forward contract and the causality
#: disclosure -- and the run config is read from beside it, which is the record of what that
#: checkpoint was trained on. With ``config`` alone the population half runs and needs no model, no
#: GPU and no checkpoint. Everything else is optional: ``overrides`` defaults to the committed delta,
#: omitting ``output_dir`` reports without writing, and ``device`` picks CUDA when it is there.
RUN_ARGS: Dict[str, Any] = {
    "checkpoint": None,
    "config": None,
    "overrides": None,
    "output_dir": None,
    "max_batches": None,
    "device": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
