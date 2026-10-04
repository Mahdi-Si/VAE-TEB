r"""What the shared CFS evaluation cannot derive about the patch model: ``TRF_PATCH_BINDING``.

The pipeline is ``teb_vae/lag_attn_cfs/eval``, reached through one
:class:`~teb_vae.lag_attn_cfs.eval.binding.ModelBinding` (plan D1). This module is the only place the
patch eval's registry is written, so no analysis author edits it: a new analysis module under
``analyses/`` is registered here once, and declares its own headline scalars as ``HEADLINE``.

Registry (in run order): the shared table analyses, with ``calibration`` replaced in place by the
patch port; then the CFS cell's extras with the patch ports of ``samples``, ``occlusion`` and
``time_shift`` substituted in place (``samples``, ``recording_traces`` and ``attribution`` see
unlabelled recordings under ``raw.UNLABELLED``); then the ten new raw-signal analyses; then
``cross_subgroup`` over :data:`METRIC_SOURCES` (``channel_skill`` in place of ``spectral_skill``).
Excluded: ``warmup`` and ``spectral_skill`` (a filter-bank warm-up and frequency channels a raw patch
does not have) and the unskippable ``band_partition`` (an ST/PH channel map the model never reads).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import preflight
from teb_vae.lag_attn_cfs.eval.binding import EXTRA_ANALYSES as CFS_EXTRA_ANALYSES
from teb_vae.lag_attn_cfs.eval.binding import HEADLINE_SCALARS as CFS_HEADLINE_SCALARS
from teb_vae.lag_attn_cfs.eval.analyses import cross_subgroup
from teb_vae.lag_attn_cfs.eval.binding import ModelBinding
from teb_vae.lag_attn_cfs.eval.probe import REQUIRED_BATCH_FIELDS
from teb_vae.lag_attn_transformer_cfs.eval.binding import GEOMETRY_KEYS as TRF_CFS_GEOMETRY_KEYS
from teb_vae.lag_attn_transformer_cfs.eval.binding import trf_cfs_encoder_disclosure
from teb_vae.lag_attn_transformer_e2e.trainer import _check_raw_length_against_shard
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import (
    calibration,
    channel_skill,
    decelerations,
    delay_map,
    event_locked,
    fhr_drivers,
    impulse_response,
    latent_descriptors,
    occlusion,
    raw_attribution,
    raw_shift,
    samples,
    signal_loss,
    time_shift,
)
from teb_vae.lag_attn_transformer_patch.eval.view import (
    SeqVaeLagAttnTrfPatch,
    SeqVaeLagAttnTrfPatchEvalTask,
)

#: The committed override delta, merged over a checkpoint's own resolved config.
DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "configs" / "eval_overrides.yaml"

#: The transformer-CFS keys less the three the patch constructor does not take (``c_y``, ``c_u``,
#: ``use_up_st``), plus the four that define this cell's target and source. Each is a constructor
#: keyword **and** a ``VAE_model`` key of ``configs/default.yaml`` (``reconcile`` skips any other).
GEOMETRY_KEYS: Tuple[str, ...] = tuple(
    key for key in TRF_CFS_GEOMETRY_KEYS if key not in ("c_y", "c_u", "use_up_st")
) + ("target_summary_loc", "target_summary_scale", "variability_eps", "source_validity")


def _run(module: Any) -> Any:
    """A module's entry point, ``run_<name>_analysis``."""
    return getattr(module, f"run_{module.__name__.rsplit('.', 1)[-1]}_analysis")


#: ``cross_subgroup``'s per-recording sources for this registry: the shared list less the three
#: ``warmup`` tertiles, with ``spectral_skill``'s band entry replaced by ``channel_skill``'s two channels.
METRIC_SOURCES: Tuple[cross_subgroup.MetricSource, ...] = tuple(
    source for source in cross_subgroup.METRIC_SOURCES if source.analysis not in ("warmup", "spectral_skill")
) + tuple(
    cross_subgroup.MetricSource("channel_skill", channel_skill.PER_RECORDING_FILENAME, column, higher_is_better=True)
    for column in ("pred_gap_level", "pred_gap_variability")
)


def run_cross_subgroup_analysis(
    context: Any, *, eval_config: Dict[str, Any], output_dir: Any, probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The shared cross-cohort tests over :data:`METRIC_SOURCES` (passed on the context)."""
    fields = {field.name: getattr(context, field.name) for field in dataclasses.fields(context)}
    return cross_subgroup.run_cross_subgroup_analysis(
        SimpleNamespace(**fields, metric_sources=METRIC_SOURCES), eval_config=eval_config,
        output_dir=output_dir, probe=probe,
    )


#: Shared-registry names whose implementation the patch replaces in place.
REPLACED_ANALYSES: Dict[str, Any] = {
    "calibration": _run(calibration),
    "cross_subgroup": run_cross_subgroup_analysis,
}

#: CFS-extra names whose implementation the patch port replaces in place (same position).
_PORTED_EXTRAS = {"samples": samples, "occlusion": occlusion, "time_shift": time_shift}

#: Class-balanced selections that would skip an unlabelled recording (the planted instrument): run
#: them on per-sample classes filled with ``raw.UNLABELLED``.
_UNLABELLED_FILLED = ("samples", "recording_traces", "attribution")

#: The ten new analyses, in run order: table readers first, then the re-forwarding passes, then
#: the IG passes (the most expensive last, so an earlier failure costs least).
NEW_ANALYSES = (
    channel_skill,
    latent_descriptors,
    event_locked,
    decelerations,
    signal_loss,
    delay_map,
    raw_shift,
    impulse_response,
    raw_attribution,
    fhr_drivers,
)



def _extra(name: str, function: Any) -> Any:
    """A CFS extra as this binding runs it: the patch port where there is one, unlabelled-filled where listed."""
    function = _run(_PORTED_EXTRAS[name]) if name in _PORTED_EXTRAS else function
    return raw.with_unlabelled_classes(function) if name in _UNLABELLED_FILLED else function


EXTRA_ANALYSES: Dict[str, Any] = {
    **{name: _extra(name, function) for name, function in CFS_EXTRA_ANALYSES.items()},
    **{module.__name__.rsplit(".", 1)[-1]: _run(module) for module in NEW_ANALYSES},
}

EXCLUDED_ANALYSES: Tuple[str, ...] = ("warmup", "spectral_skill", "band_partition")

#: The CFS extras' headline entries less the excluded analyses', the two geometry guards re-homed
#: on the collection readouts, and every patch module's own ``HEADLINE`` declarations.
HEADLINE_SCALARS: Tuple[Tuple[str, Tuple[str, ...]], ...] = tuple(
    entry for entry in CFS_HEADLINE_SCALARS if entry[1][0] not in EXCLUDED_ANALYSES
) + (
    ("anchors_per_sample", ("readouts", "anchors_per_sample")),
    ("target_warm_frac", ("readouts", "target_warm_frac")),
) + tuple(
    (headline_name, (module.__name__.rsplit(".", 1)[-1], "headline", key))
    for module in (samples, occlusion, time_shift, calibration, *NEW_ANALYSES)
    for headline_name, key in getattr(module, "HEADLINE", ())
)

#: The loader probe's required fields: the shared list less the four ST/PH blocks nothing reads.
REQUIRED_FIELDS: Tuple[str, ...] = tuple(
    name for name in REQUIRED_BATCH_FIELDS if name not in preflight.STORED_BLOCKS
)

#: The causality record's wording, true of raw patches (the shared text describes a filter bank).
CAUSALITY_TEXT: Dict[str, Any] = {
    "statement": (
        "Patch token t reads the loader-normalized raw FHR and UP samples [16t, 16t+15] at 4 Hz and "
        "nothing later, so a forecast of tokens t+1..t+H from anchor t is a genuine forecast: the "
        "inputs do not contain their own future, and there is no filter and no group delay. The "
        "coupling readout is still named source_conditioned_kl_raw and no number in this run may be "
        "labelled a transfer entropy."
    ),
    "lag_axis": {
        "label": "lag (s): the UP patch l tokens before the anchor patch, 4 s quantised; no filter delay",
        "caveat": (
            f"Lag l is the UP patch whose last sample lies {SECONDS_PER_STEP:g}*l s before the anchor "
            "patch's last sample. A UP->FHR delay d informs lags [d-H, d-1], so the delay a lag "
            f"explains is {SECONDS_PER_STEP:g}*(l+1+tau) s and needs the horizon axis."
        ),
    },
    "group_delay_seconds": {},
}


def check_patch_shards(config: Dict[str, Any], model: Any) -> None:
    """The shard guard for raw patches: the trimmed raw length must be ``T * R`` of the **model**.

    Replaces the ST/PH declared-width guard. Runs the training driver's own check on a view whose
    training list is the evaluation shards and whose geometry is the rebuilt model's.

    Raises:
        EvalPreconditionUnmet: Carrying the trainer guard's message.
    """
    dataset = dict(config.get("dataset_config") or {})
    dataset["vae_train_datasets"] = list(dataset.get("vae_test_datasets") or [])
    view = {
        "dataset_config": dataset,
        "model_config": {"VAE_model": {
            "sequence_length": int(model.sequence_length), "raw_per_step": int(model.raw_per_step),
        }},
    }
    try:
        _check_raw_length_against_shard(view)
    except ValueError as exc:
        raise preflight.EvalPreconditionUnmet(str(exc)) from exc


def patch_encoder_disclosure(model: Any) -> Dict[str, Any]:
    """The transformer-CFS encoder disclosure, plus the two facts a raw patch adds."""
    return {
        **trf_cfs_encoder_disclosure(model),
        "raw_per_step": int(preflight.disclosed_attribute(model, "raw_per_step")),
        "source_validity": str(preflight.disclosed_attribute(model, "source_validity")),
    }


TRF_PATCH_BINDING = ModelBinding(
    model_cls=SeqVaeLagAttnTrfPatch,
    task_cls=SeqVaeLagAttnTrfPatchEvalTask,
    tag="lag_attn_trf_patch",
    geometry_keys=GEOMETRY_KEYS,
    encoder_disclosure=patch_encoder_disclosure,
    overrides_path=DEFAULT_OVERRIDES_PATH,
    extra_analyses=EXTRA_ANALYSES,
    headline_scalars=HEADLINE_SCALARS,
    excluded_analyses=EXCLUDED_ANALYSES,
    replaced_analyses=REPLACED_ANALYSES,
    target_fields=("fhr", "up"),
    shard_guard=check_patch_shards,
    required_batch_fields=REQUIRED_FIELDS,
    causality_text=CAUSALITY_TEXT,
)

__all__ = [
    "CAUSALITY_TEXT",
    "DEFAULT_OVERRIDES_PATH",
    "EXCLUDED_ANALYSES",
    "EXTRA_ANALYSES",
    "GEOMETRY_KEYS",
    "HEADLINE_SCALARS",
    "NEW_ANALYSES",
    "REPLACED_ANALYSES",
    "METRIC_SOURCES",
    "TRF_PATCH_BINDING",
    "check_patch_shards",
    "patch_encoder_disclosure",
    "run_cross_subgroup_analysis",
]
