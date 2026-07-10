r"""S1-T04: class-aware checkpoint guard and rebuild.

A checkpoint must always be graded with the architecture it was trained under. These tests
pin :func:`pl_module_v2.rebuild_model_from_checkpoint`, the shared helper both
``eval_v2.run_eval`` and ``run_pipeline_v2.run_test_plots`` now use:

* a saved ``v3_prod`` checkpoint rebuilds as ``SeqVaeLagAttnV3`` with ``causal_norm is True``
  (keyed on the stored ``model_class``, NOT the v1 alias a bare ``build_model(model_kwargs)``
  would fall back to);
* a ``parity`` checkpoint rebuilds as ``SeqVaeLagAttnV3`` (not v1) with
  ``posterior_logvar == 'independent'`` -- from its OWN kwargs, so an arm is never graded as
  another arm;
* grading a checkpoint whose ``model_class`` differs from the configured class raises;
* a legacy blob with no ``model_class`` still loads through the committed alias.

Fast (tiny CPU models); no cache required.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import (  # noqa: E402
    pl_module_v2 as plm,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E402,E501
    resolve_arm,
)

_CPU = torch.device("cpu")

_BASE_MODEL = {
    "sequence_length": 16, "d_model": 16, "d_z": 8, "horizon": 4, "warmup_period": 2,
    "c_y": 87, "c_u": 101, "use_up_st": True, "max_lag": 8, "num_heads": 4, "d_head": 4,
    "lstm_layers": 1, "logvar_clamp": [-5.0, 3.0], "head_structured_latent": True,
}

# The v3 overlay + per-arm deltas, mirroring config_synth_v3.yaml (arms override model.v3).
_V3_CFG = {
    "model": {
        **_BASE_MODEL, "class": "SeqVaeLagAttnV3",
        "v3": {"causal_norm": True, "posterior_logvar": "residual",
               "logvar_bound": "smooth", "kld_support": "anchor",
               "lag_bias_init": "alibi_decay", "use_entmax": True,
               "freeze_unused_attn_proj": True},
    },
    "arms": {
        "parity": {"model": {"v3": {"causal_norm": False, "posterior_logvar": "independent",
                                    "logvar_bound": "clamp", "kld_support": "full",
                                    "lag_bias_init": "normal", "use_entmax": False}}},
        "v3_prod": {"model": {"v3": {"causal_norm": True}}},
    },
}


def _save_arm_checkpoint(tmp_path, arm):
    """Build + checkpoint an arm's tiny model; return the ckpt path."""
    model_cfg = resolve_arm(_V3_CFG, arm)["model"]
    model, kwargs = plm.build_model(model_cfg, _CPU)
    path = tmp_path / f"{arm}.ckpt"
    plm.save_checkpoint_v2(
        path, model=model, model_kwargs=kwargs, config={"experiment": {}},
        data_meta={}, epoch=1, val_loss=float("nan"),
        loss_settings={}, latent_stats_fitted=False, arm=arm,
    )
    return path


def test_v3_prod_checkpoint_rebuilds_as_v3(tmp_path) -> None:
    path = _save_arm_checkpoint(tmp_path, "v3_prod")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttnV3"
    model, _ = plm.rebuild_model_from_checkpoint(
        blob, _CPU, expected_class="SeqVaeLagAttnV3")
    assert type(model).__name__ == "SeqVaeLagAttnV3"
    assert model.causal_norm is True


def test_parity_checkpoint_rebuilds_as_v3_not_v1(tmp_path) -> None:
    """The parity arm is a v3 with the independent head -- NOT a literal v1."""
    path = _save_arm_checkpoint(tmp_path, "parity")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttnV3"
    model, _ = plm.rebuild_model_from_checkpoint(
        blob, _CPU, expected_class="SeqVaeLagAttnV3")
    assert type(model).__name__ == "SeqVaeLagAttnV3"
    assert model.posterior_logvar == "independent"
    assert model.causal_norm is False


def test_class_mismatch_raises(tmp_path) -> None:
    """Grading a v3 checkpoint under a config that expects v1 raises."""
    path = _save_arm_checkpoint(tmp_path, "v3_prod")
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    with pytest.raises(ValueError):
        plm.rebuild_model_from_checkpoint(blob, _CPU, expected_class="SeqVaeLagAttnV1")


def test_legacy_blob_without_model_class_uses_alias() -> None:
    """A legacy blob (no ``model_class``) warns and rebuilds through the committed alias."""
    model, kwargs = plm.build_model(_BASE_MODEL, _CPU)  # a v1 tiny model
    blob = {"model_kwargs": kwargs, "model_state_dict": model.state_dict()}  # no model_class
    with pytest.warns(RuntimeWarning):
        rebuilt, _ = plm.rebuild_model_from_checkpoint(
            blob, _CPU, expected_class="SeqVaeLagAttnV1")
    assert type(rebuilt).__name__ == "SeqVaeLagAttnV1"


def test_resolved_model_class_name_helper() -> None:
    assert plm.resolved_model_class_name({"class": "SeqVaeLagAttnV3"}) == "SeqVaeLagAttnV3"
    assert plm.resolved_model_class_name({}) == "SeqVaeLagAttnV1"  # absent -> alias
