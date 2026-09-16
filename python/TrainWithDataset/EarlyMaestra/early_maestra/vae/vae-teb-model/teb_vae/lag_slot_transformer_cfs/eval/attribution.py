r"""Captum attributions for the lag-residual cell: the post-pass stage over the family's shared core.

The scoring pass reduces every recording to one row and discards the batches it reduced them
from. This stage runs **after** the pass, on the identities the pass recorded, and attributes
three per-anchor scalars of a class-balanced draw of segments back over the three input streams:
the divergence, the mean-decoded forecast gap, and the proposal norm on each configured lag band.
The wrapper, the two baselines, the integrated gradients, the band ablation, the reductions and
the figures are the family's -- the same tables under the same names the lag-attentive cells
write -- so an attribution under this architecture reads beside one under another.

**What the layer split is here.** This architecture poses no query and computes no distribution
over lags; what the fusion sums is one signed proposal per lag. The layer attribution is taken on
the proposal head's **output** -- the per-lag mean proposals -- summed over the latent coordinates
at the anchor, so it is a per-lag split of the readout *through* the summation and the limiter.
It is an attribution, not an allocation: the proposals admit a zero-sum reallocation that leaves
every prediction identical while moving what each lag is attributed, and the qualification the
pass records says so on every artifact.

**The lag readout attributed under the band name is the proposal norm on that band**, which is an
update magnitude before the sum and the limiter and neither a distribution over lags nor an
allocation of the divergence. None of the names this stage writes is one of the attention-shaped
keys the acceptance gate refuses.

**The channel map is built here when the directory holds none.** The lag-attentive cells' runner
emits it as an unskippable step; this cell's pass does not, so the frequency-band tables would be
empty for a reason about the runner rather than about the model. The stage builds it through the
same function, from the same shards and the same resolved budget, and records a skip by name when
the shards carry no channel provenance.

**An ablated input coordinate is marked, and its attribution is asserted to be zero.** The model
applies its input ablation as the first step of its forward, so an integrated gradient with
respect to an ablated coordinate is exactly zero by construction; the stage reads that back off
the written example maps and refuses the run if it is not, because an ablated coordinate that
attracted attribution would mean the ablation is not at the input boundary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval.band_partition import KIND_ORDER0
from teb_vae.lag_attn_cfs.eval import attribution_pass, lag_axis
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.dataset_rows import dataset_index_map

#: Status values the summary block carries.
STATUS_SKIPPED = "SKIPPED"
STATUS_ATTRIBUTED = "ATTRIBUTED"

#: The per-coordinate marker table the stage writes beside the family's tables: one row per
#: ablated input coordinate, with the largest attribution any example map assigned to it.
ABLATED_INPUTS_FILENAME = "attribution_ablated_inputs.csv"

#: The status an ablated coordinate carries in that table and in the summary block.
STATUS_ABLATED = "ablated"


def ablated_input_check(model: Any, directory: Path) -> Dict[str, Any]:
    """Mark every ablated input coordinate and read its attribution off the written maps.

    Args:
        model: The rebuilt net, for its input policy.
        directory: The attribution stage's own directory, where the example maps were written.

    Returns:
        ``{'coordinates': [...], 'max_abs_attribution': float | None, 'measured_on': ...}``. The
        list is empty on a model with no ablation. The maximum is ``None`` when no example map
        was written, which is a measurement the stage did not make rather than a zero.

    Raises:
        ValueError: If any example map assigns a nonzero attribution to an ablated coordinate.
    """
    # The model's own record of what it ablates, with the partition's coefficient kind added so
    # the row can be found in the channel map. Read off the model rather than off the binding's
    # disclosure, which this module must not import: the binding registers this stage.
    coordinates: List[Dict[str, Any]] = [
        {**entry, "kind": KIND_ORDER0} for entry in model.input_ablation_record()["ablated_inputs"]
    ]
    if not coordinates:
        return {"coordinates": [], "max_abs_attribution": None, "measured_on": "no ablation"}

    maps_path = directory / core.MAPS_FILENAME
    if not maps_path.is_file():
        for entry in coordinates:
            entry["max_abs_attribution"] = None
            entry["status"] = STATUS_ABLATED
        return {
            "coordinates": coordinates,
            "max_abs_attribution": None,
            "measured_on": "no example map was written, so nothing was read back",
        }

    largest = 0.0
    with np.load(maps_path) as handle:
        for entry in coordinates:
            key = f"map_{entry['stream']}"
            if key not in handle:
                entry["max_abs_attribution"] = None
                entry["status"] = STATUS_ABLATED
                continue
            field = np.asarray(handle[key], dtype=np.float64)
            # The maps are indexed by the declared input axis of each stream, on which the
            # ablated coordinate sits at the channel the policy names.
            value = float(np.nanmax(np.abs(field[..., int(entry["channel"])]))) if field.size else 0.0
            entry["max_abs_attribution"] = value
            entry["status"] = STATUS_ABLATED
            largest = max(largest, value)
    if largest > 0.0:
        offending = [
            f"{entry['field']}[{entry['channel']}]"
            for entry in coordinates
            if (entry.get("max_abs_attribution") or 0.0) > 0.0
        ]
        raise ValueError(
            f"an ablated input coordinate received a nonzero attribution: {offending}, largest "
            f"{largest:.3g}. The ablation is applied as the first step of the forward, so the "
            f"gradient with respect to an ablated coordinate is exactly zero by construction; a "
            f"nonzero value here means some path read the original stream."
        )
    return {
        "coordinates": coordinates,
        "max_abs_attribution": largest,
        "measured_on": f"every example map in {core.MAPS_FILENAME}",
    }


def ensure_channel_map(config: Mapping[str, Any], model: Any, results_dir: Any) -> Dict[str, Any]:
    """Write the declared-axis channel map into the results directory when it is absent.

    Args:
        config: The merged run configuration, for the shard list, the source block flag and the
            loader's trim.
        model: The rebuilt net, for its target keep-index and kept width.
        results_dir: The run's results directory.

    Returns:
        The channel-map step's own record, or ``{'skipped': True, 'reason': ...}``.
    """
    from teb_vae.lag_attn_cfs.eval.analyses.band_partition import emit_partition, trim_steps_of

    path = Path(str(results_dir)) / attribution_pass.CHANNEL_MAP_FILENAME
    if path.is_file():
        return {"skipped": False, "reason": "already present"}
    dataset = dict(config.get("dataset_config") or {})
    vae = dict((config.get("model_config") or {}).get("VAE_model") or {})
    gate = getattr(model, "target_gate", None)
    keep_index = None if gate is None else [int(v) for v in gate.keep_index.tolist()]
    try:
        return emit_partition(
            list(dataset.get("vae_test_datasets") or []), results_dir,
            use_up_st=bool(vae.get("use_up_st", True)),
            declared={"target": vae.get("c_y"), "source": vae.get("c_u")},
            trim_steps=trim_steps_of(dict(config)),
            keep_index=keep_index,
            kept_width=int(getattr(model, "decoder_out_channels", 0) or 0) or None,
        )
    except Exception as error:  # noqa: BLE001 - a missing map costs the band tables, not the stage
        return {"skipped": True, "reason": f"{type(error).__name__}: {error}"}


def run_attribution(
    task: Any,
    loader: Any,
    identities: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    results_dir: Any,
    geometry_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attribute a class-balanced draw of segments and write the attribution directory.

    Args:
        task: The loaded task.
        loader: The evaluation dataloader the pass walked.
        identities: One row per scored segment -- ``guid``, ``epoch``, class, subgroup -- as the
            pass recorded them.
        config: The merged run configuration, for the channel map.
        eval_config: The validated settings, for the cap, the seed and the lag bands.
        results_dir: The run's results directory; the stage writes into its own subdirectory.
        geometry_record: The collection-style geometry record the trace's break tolerance is read
            from, or ``None`` for the family's default stride.

    Returns:
        The block the summary carries: a status, the pass's own block, and the channel-map record.
        A stage that could not run at all says why.
    """
    if identities.empty or identities[labels.CLASS_COLUMN].isna().all():
        reason = (
            "no scored segment carries a clinical class, so no class-balanced draw can be made; "
            "the class is recovered from the weight-scaled target, which reaches a batch only "
            "when the override delta's load_fields names it"
        )
        logger.warning(f"{core.ANALYSIS_DIRNAME}: skipped -- {reason}")
        return {"status": STATUS_SKIPPED, "reason": reason, "files": []}
    model = task.orig_model
    channel_record = ensure_channel_map(config, model, results_dir)
    block = attribution_pass.run_pass(
        task, loader, identities, dataset_index_map(loader), core.SLOT_CELL,
        eval_config=eval_config, results_dir=results_dir,
        lag_seconds=lag_axis.compensated_seconds_axis(int(model.n_lags), 0),
        break_after_s=lag_axis.break_tolerance_s({"geometry": dict(geometry_record or {})}),
        segment_stride_s=lag_axis.segment_stride_s({"geometry": dict(geometry_record or {})}),
        channel_map=attribution_pass.read_channel_map(results_dir),
        occlusion=None,
        spectral=None,
        delay_steps=0,
    )
    directory = Path(str(results_dir)) / core.ANALYSIS_DIRNAME
    ablated = ablated_input_check(model, directory)
    if ablated["coordinates"]:
        pd.DataFrame(ablated["coordinates"]).to_csv(directory / ABLATED_INPUTS_FILENAME, index=False)
        block["files"] = [*block.get("files", []), ABLATED_INPUTS_FILENAME]
    block["checks"] = {
        **block.get("checks", {}),
        "ablated_input_max_abs": ablated["max_abs_attribution"],
    }
    return {
        "status": STATUS_ATTRIBUTED,
        "channel_map": channel_record,
        "directory": core.ANALYSIS_DIRNAME,
        "ablated_inputs": ablated,
        **block,
    }


__all__ = [
    "ABLATED_INPUTS_FILENAME",
    "STATUS_ABLATED",
    "STATUS_ATTRIBUTED",
    "STATUS_SKIPPED",
    "ablated_input_check",
    "ensure_channel_map",
    "run_attribution",
]
