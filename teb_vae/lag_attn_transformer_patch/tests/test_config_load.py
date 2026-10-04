"""P4-06: every shipped config reaches the model or the task, the pre-flight refuses the removed
keys by name, and no config drifts from its reference beyond its declared delta (plan B.5, B.8,
B.9).
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.trainer import _NON_CONSTRUCTOR_KEYS
from teb_vae.lag_attn_transformer_e2e.tests.test_config_load import TASK_LEVEL_KEYS, _leaves
from teb_vae.lag_attn_transformer_patch.trainer import LagAttnTrfPatchTrainer
from teb_vae.lag_attn_transformer_rws.trainer import NULLABLE_MODEL_KEYS

from .conftest import absolutize_dataset_paths

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CFS_DEFAULT = (
    Path(__file__).resolve().parents[2] / "lag_attn_transformer_cfs" / "configs" / "default.yaml"
)
_VAE = "model_config.VAE_model"
_ABSENT = "<absent>"

#: B.5's removed constructor keys, plus one key of each causal-feature family the CFS config
#: carries. Written out, not imported from the trainer: the trainer's list is what is under test.
REFUSED_KEYS = (
    "c_y", "c_u", "use_up_st",
    "target_keep_index", "target_warmup_steps", "source_keep_index", "source_warmup_steps",
    "target_align_delays", "source_align_delays",
    "causal_warmup_budget_steps", "target_weight_st", "target_scored_horizon",
    "target_phase_fast_cutoff_hz",
)

#: Leaves every arm must rename, so its run is not filed under the baseline's name.
ARM_IDENTITY = {
    "general_config.tag",
    "advanced_config.tracking.mlflow.run_name",
    "advanced_config.tracking.mlflow.tags.variant",
}

#: B.9 arms: file -> the one leaf it moves.
ARMS = {
    "sweep_warmup_134.yaml": {f"{_VAE}.warmup_period": 134},
    "sweep_lag_kv_encoder.yaml": {f"{_VAE}.lag_kv_source": "encoder"},
    "sweep_persistence.yaml": {f"{_VAE}.persistence_residual": True},
    "sweep_source_validity_fhr.yaml": {f"{_VAE}.source_validity": "fhr_weight"},
}

CFS_IDENTITY = ARM_IDENTITY | {
    "general_config.folders_config.out_dir_base",
    "advanced_config.tracking.mlflow.experiment_name",
}

#: The B.9 delta against the CFS default: leaf -> this package's value. Anything else differing is
#: drift, and the two cells would no longer differ in the representation alone.
CFS_DELTA = {
    f"{_VAE}.warmup_period": 30,
    f"{_VAE}.persistence_residual": False,
    f"{_VAE}.prior_availability_input": False,
    f"{_VAE}.horizon_weight_halflife_steps": None,
    f"{_VAE}.source_validity": "finite",
    # C12: identity placeholders until summary_stats.py runs on the production shards.
    f"{_VAE}.target_summary_loc": [0.0, 0.0],
    f"{_VAE}.target_summary_scale": [1.0, 1.0],
    f"{_VAE}.variability_eps": 0.01,
    "dataset_config.dataloader_config.dataset_kwargs.load_fields": [
        "fhr", "up", "weight", "guid", "epoch",
    ],
    "dataset_config.dataloader_config.normalize_fields": ["fhr", "up"],
    "advanced_config.trainer.gradient_clip_val": 280.0,
    "advanced_config.spike_breaker.additive_margin": 175.0,
    **{
        f"{_VAE}.{key}": _ABSENT
        for key in (
            "c_y", "c_u", "use_up_st",
            "causal_align_reference", "causal_align_reference_source", "causal_leg_alignment",
            "causal_phase_operator", "causal_reach_budget_s", "causal_target_forecast_clock",
            "causal_warmup_budget_steps",
            "target_phase_fast_cutoff_hz", "target_phase_fast_horizon",
            "target_weight_ph", "target_weight_st",
        )
    },
}


def _delta(config: dict, reference: dict) -> dict:
    """``{leaf path: config's value}`` wherever the two resolved configs differ."""
    mine, theirs = dict(_leaves(config)), dict(_leaves(reference))
    return {
        path: mine.get(path, _ABSENT)
        for path in mine.keys() | theirs.keys()
        if mine.get(path, _ABSENT) != theirs.get(path, _ABSENT)
    }


def test_every_config_key_reaches_the_model_or_task_and_passes_preflight() -> None:
    """The signature sweep drops a key naming no constructor argument, and a ``null`` whose
    constructor default is not ``null``, both in silence."""
    params = inspect.signature(LagAttnTrfPatchTrainer.MODEL_CLS.__init__).parameters
    sweepable = set(params) - set(_NON_CONSTRUCTOR_KEYS)
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        config = absolutize_dataset_paths(load_config(str(path)))
        vae = config["model_config"]["VAE_model"]

        orphans = [k for k in vae if k not in sweepable and k not in TASK_LEVEL_KEYS]
        assert orphans == [], f"{path.name}: {orphans} reach neither the constructor nor the task"
        dropped_nulls = [
            k for k, v in vae.items()
            if v is None and k in params and k not in NULLABLE_MODEL_KEYS
            and params[k].default is not None
        ]
        assert dropped_nulls == [], f"{path.name}: null {dropped_nulls} rebuilds the default"
        LagAttnTrfPatchTrainer.preflight(config)


def test_preflight_refuses_every_removed_key_by_name() -> None:
    base = load_config(str(_CONFIG_DIR / "default.yaml"))
    unrefused = []
    for key in REFUSED_KEYS:
        config = copy.deepcopy(base)
        config["model_config"]["VAE_model"][key] = 1
        try:
            LagAttnTrfPatchTrainer.preflight(config)
        except ValueError as error:
            if key in str(error):
                continue
        unrefused.append(key)
    assert unrefused == []


def test_arms_and_default_differ_from_their_reference_only_in_the_declared_leaves() -> None:
    default = load_config(str(_CONFIG_DIR / "default.yaml"))
    assert {p.name for p in _CONFIG_DIR.glob("sweep_*.yaml")} == set(ARMS)
    for name, leaf in ARMS.items():
        delta = _delta(load_config(str(_CONFIG_DIR / name)), default)
        assert ARM_IDENTITY <= delta.keys(), f"{name} keeps a baseline identity key"
        assert {p: v for p, v in delta.items() if p not in ARM_IDENTITY} == leaf, name

    delta = _delta(default, load_config(str(_CFS_DEFAULT)))
    assert CFS_IDENTITY <= delta.keys(), "default.yaml keeps a CFS identity key"
    assert {p: v for p, v in delta.items() if p not in CFS_IDENTITY} == CFS_DELTA
