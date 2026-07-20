r"""One pass over the loader that says what the loader actually yielded.

This is the pipeline's only real input validator, and it is the artifact that would have
caught the predecessor's hardest bug -- a loader that silently truncated its second pass to the
first file's index range, which was invisible in every other output and presented only as
"only 1 class found". Nothing else in a run reports per-file coverage, so nothing else can see
that failure.

Two departures from the predecessor, both deliberate.

**A configured file yielding zero samples raises.** The predecessor logged it and was ignored.
A zero-count file means either a path that resolves to nothing or a filter that excluded the
whole shard, and both silently narrow the population every headline number is computed over.

**One pass, and no forward.** The probe reads the loader, not the model: it records what the
split actually yielded and supplies the per-file stratification every capped draw uses. An
earlier form also cached a per-sample ``z_mean`` through ``encode_only``, on the reasoning that
the latent analyses could then skip a pass -- but nothing ever read it. ``latent`` takes its own
full pass and needs the per-step posterior, not a support-averaged coordinate, so the cache cost
an extra encode over the whole split on every run and saved nothing. It is gone; the probe is
still unskippable, for the two reasons that were always the real ones.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.runner import EvalRunner, field_names, get_field, guid_of, to_numpy

#: File written into the run directory.
PROBE_FILENAME = "loader_probe.json"

#: Probe keys that are per-sample rather than summary, and therefore never serialised. They are
#: returned in memory for the analyses that consume them -- the GUID and source-file vectors a
#: later ``groupby`` and every stratified draw need. Written out they would repeat every GUID in
#: the split inside a summary meant to be read at a glance.
IN_MEMORY_KEYS = ("guids", "source_files")


def _weight_distribution(weight: torch.Tensor) -> Dict[str, Any]:
    r"""Summarise a per-step validity tensor.

    ``binary`` is the load-bearing field. If ``weight`` is strictly $\{0, 1\}$ the mask
    arithmetic downstream is an intersection of indicator functions and the ``label`` filter's
    exact-float-equality hazard is moot; if it is fractional, every masked mean is a weighted
    mean and must be read as one.

    Args:
        weight: Per-step validity, $(B, T)$.

    Returns:
        Summary statistics, including whether the values are strictly binary.
    """
    values = to_numpy(weight).ravel()
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


def _target_value_summary(target: torch.Tensor) -> Dict[str, Any]:
    r"""Summarise the *raw* target values, as distinct from the class histogram.

    The class histogram records one value per recording -- its first nonzero step -- which is the
    right reduction for "which classes are present" and the wrong one for "was this field
    truncated to integers". ``target`` is the class label scaled by the per-step ``weight``, so a
    fractional value appears only at the partially-valid boundaries of a segment, and a
    recording's first nonzero step is almost always in a full-weight region. Asking the class
    histogram about truncation therefore answers "every value is an integer" on healthy
    fractional-weight data.

    Args:
        target: Per-step target, $(B, T)$ or $(B, T, C)$.

    Returns:
        Whether any finite value is fractional, whether any is non-finite, and the counts behind
        both.
    """
    values = to_numpy(target).ravel()
    finite = values[np.isfinite(values)]
    nonzero = finite[finite != 0.0]
    return {
        "n_values": int(values.size),
        "n_non_finite": int(values.size - finite.size),
        "n_nonzero": int(nonzero.size),
        # The question the truncation check needs answered, over every step rather than one.
        "n_fractional": int(np.sum(nonzero != np.round(nonzero))),
    }


def _merge_target_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the per-batch raw-target summaries into one.

    Args:
        summaries: Per-batch summaries from :func:`_target_value_summary`.

    Returns:
        The merged summary, or an empty dict when the batches carried no ``target``.
    """
    if not summaries:
        return {}
    totals = {
        key: int(sum(item[key] for item in summaries))
        for key in ("n_values", "n_non_finite", "n_nonzero", "n_fractional")
    }
    totals["any_fractional"] = bool(totals["n_fractional"] > 0)
    totals["any_non_finite"] = bool(totals["n_non_finite"] > 0)
    return totals




def run_probe(
    runner: EvalRunner,
    loader: Any,
    *,
    configured_files: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    output_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Iterate the loader once, recording what the split actually yielded.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        configured_files: Paths the config asked for. Every basename here must appear in the
            counts with a nonzero count, or the probe raises. ``None`` skips that check, which
            is only appropriate when the caller has no config to compare against.
        max_samples: Sample cap for the pass. ``None`` covers the whole split.
        output_dir: Where to write ``loader_probe.json``. ``None`` skips the write.

    Returns:
        The probe record. ``guids`` and ``source_files`` are returned in memory and not written
        to JSON; everything else is serialisable.

    Raises:
        RuntimeError: If the split yields no samples at all, or if a configured file yields
            none.
    """
    per_file: Counter = Counter()
    per_cs_label: Counter = Counter()
    per_bg_label: Counter = Counter()
    per_target_class: Counter = Counter()
    weight_summaries: List[Dict[str, Any]] = []
    target_summaries: List[Dict[str, Any]] = []

    guids: List[str] = []
    sources: List[str] = []
    n_samples = 0
    n_batches = 0
    fields_seen: List[str] = []

    for batch in runner.iter_batches(loader, max_samples=max_samples):
        n_batches += 1
        # Assembled for the batch size and for the width checks build_target_streams performs;
        # the probe does no forward of its own.
        y_st, _ = runner.build_target_streams(batch)
        batch_size = int(y_st.shape[0])

        if not fields_seen:
            fields_seen = list(field_names(batch))

        basenames = get_field(batch, "source_file_basename")
        for index in range(batch_size):
            name = str(basenames[index]) if basenames is not None else "unknown"
            per_file[name] += 1
            sources.append(name)
            guids.append(guid_of(batch, index))

        cs_label = get_field(batch, "cs_label")
        if cs_label is not None:
            for value in to_numpy(cs_label).ravel().tolist():
                per_cs_label[str(bool(value))] += 1
        bg_label = get_field(batch, "bg_label")
        if bg_label is not None:
            for value in to_numpy(bg_label).ravel().tolist():
                per_bg_label[str(bool(value))] += 1

        weight = get_field(batch, "weight")

        target = get_field(batch, "target")
        if target is not None:
            # ``target`` stores the class code *scaled by* the per-step weight, so a partially
            # valid step of an acidosis recording at weight 0.5 stores 1.0 -- indistinguishable
            # from a fully valid healthy step. Reading the raw value keyed this counter on
            # quantities like "0.75", which report.check_classes_present then counted as a
            # distinct class: one such recording was enough to make a genuinely single-class
            # split report two, permanently defeating the very coverage check that counter
            # exists to feed. labels.clinical_class_code divides the weight back out and
            # already returns None for a pad-only or uniformly zero row.
            target_rows = to_numpy(target)
            weight_rows = to_numpy(weight) if weight is not None else None
            for index, row in enumerate(target_rows):
                weight_row = (
                    weight_rows[index]
                    if weight_rows is not None and index < len(weight_rows)
                    else np.ones_like(np.atleast_1d(row).ravel())
                )
                code = labels.clinical_class_code(np.atleast_1d(row).ravel(), weight_row)
                per_target_class[str(labels.class_name(code))] += 1
            target_summaries.append(_target_value_summary(target))

        if weight is not None:
            weight_summaries.append(_weight_distribution(weight))

        n_samples += batch_size

    if n_samples == 0:
        raise RuntimeError(
            "the test loader yielded no samples at all. Either dataset_config.vae_test_datasets "
            "resolves to nothing, or a dataset_kwargs filter (epoch_min / epoch_max / label / "
            "cs_label / bg_label) excluded every sample. Check the resolved config dumped into "
            "this run directory."
        )

    record: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "n_unique_guids": len(set(guids)),
        "batch_fields": fields_seen,
        "per_file": dict(sorted(per_file.items())),
        "per_cs_label": dict(sorted(per_cs_label.items())),
        "per_bg_label": dict(sorted(per_bg_label.items())),
        "per_target_class": dict(sorted(per_target_class.items())),
        # Distinct from per_target_class: that is one value per recording, this is every value.
        # The truncation check needs the second and cannot be answered by the first.
        "target_values": _merge_target_summaries(target_summaries),
        "weight": _merge_weight_summaries(weight_summaries),
        "max_samples": max_samples,
    }

    if configured_files is not None:
        _check_every_file_contributed(configured_files, per_file, capped=max_samples is not None)

    logger.info(
        f"loader probe: {n_samples} samples over {n_batches} batches from "
        f"{len(per_file)} file(s); classes {dict(per_target_class)}"
    )

    if output_dir is not None:
        write_probe(record, output_dir)

    # Returned, not serialised: one row per sample does not belong in a JSON summary.
    record["guids"] = guids
    record["source_files"] = sources
    return record


def _merge_weight_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the per-batch weight summaries into one.

    Args:
        summaries: Per-batch summaries from :func:`_weight_distribution`.

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
        # The question this exists to answer: is `weight` ever fractional in production data?
        "binary": all(item["binary"] for item in summaries),
    }


def _check_every_file_contributed(
    configured_files: List[str], per_file: Counter, *, capped: bool
) -> None:
    """Raise when a configured shard contributed no samples.

    Args:
        configured_files: The paths from ``dataset_config.vae_test_datasets``.
        per_file: Observed counts, keyed by basename.
        capped: Whether the pass was capped, in which case a missing file is expected rather
            than an error -- a prefix cap over concatenated shards reaches only the first ones.

    Raises:
        RuntimeError: If an uncapped pass left a configured file at zero.
    """
    expected = [Path(str(path)).name for path in configured_files]
    missing = [name for name in expected if per_file.get(name, 0) == 0]
    if not missing:
        return
    if capped:
        logger.warning(
            f"{len(missing)} configured file(s) contributed no samples under the active cap: "
            f"{missing}. A capped pass reads a prefix of the concatenated index, so this is "
            f"expected -- but it does mean the capped analyses saw a biased draw."
        )
        return
    raise RuntimeError(
        f"configured shard(s) yielded zero samples: {missing}. Observed counts: "
        f"{dict(per_file)}. Either the path does not resolve, or a dataset_kwargs filter "
        f"excluded the whole file. Every headline number would otherwise be computed over a "
        f"silently narrowed population."
    )


def summary_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the probe record without its per-sample vectors.

    Used both by :func:`write_probe` and by the orchestrator when it copies the probe's result
    into ``summary.json``, so neither can accidentally serialise one GUID per sample.

    Args:
        record: The full probe record.

    Returns:
        The summary-safe subset.
    """
    return {key: value for key, value in record.items() if key not in IN_MEMORY_KEYS}


def write_probe(record: Dict[str, Any], output_dir: Any) -> Path:
    """Write ``loader_probe.json``.

    Args:
        record: The probe record. The in-memory entries in ``IN_MEMORY_KEYS`` are skipped.
        output_dir: The run directory.

    Returns:
        The path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / PROBE_FILENAME
    serialisable = summary_view(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2, default=str)
    logger.info(f"wrote {path}")
    return path
