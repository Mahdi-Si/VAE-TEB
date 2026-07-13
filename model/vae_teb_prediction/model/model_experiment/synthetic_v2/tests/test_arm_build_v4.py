r"""S4-T04 tests: every arm resolves AND constructs a valid model (leaky arm -> leaky class)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import pytest

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.arms_v4 import (
    arm_uses_leaky_frontend,
    list_arms,
    resolve_arm_v4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
    LeakyRawFrontendSeqVaeRawV4,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import (
    SeqVaeRawV4,
    load_config,
)

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_synth_v4.yaml"


@pytest.fixture(scope="module")
def base_config() -> Dict[str, Any]:
    return load_config(str(_CONFIG_PATH))


def _small_model_kwargs(arm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    r"""Build small-but-prod-geometry model kwargs, honouring the arm's frontend/model deltas.

    Mirrors the trainer's ``_build_model_kwargs`` selection (nested ``frontend`` + raw geometry +
    flat SeqVaeRawV4 args) but at shrunk widths so construction is cheap.
    """
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_FRONTEND,
        SMALL_PROD_V3_KWARGS,
    )

    vae = arm_cfg["model_config"]["VAE_model"]
    # Start from the small_prod widths, then overlay the arm's frontend deltas (e.g. stages [16],
    # antialias/gated flags) and flat model deltas (disable_source).
    frontend = dict(SMALL_PROD_FRONTEND)
    for key, val in (vae.get("frontend") or {}).items():
        if key in ("stages", "antialias", "gated", "norm_kind"):
            frontend[key] = val
    # single_stride collapses to one stage at d_raw; drop the multi-stage channel list so the
    # frontend's len(stages)==1 branch takes over.
    if frontend.get("stages") == [16]:
        frontend.pop("channels", None)

    kwargs: Dict[str, Any] = dict(
        frontend=frontend, raw_len=5280, decimation=16, **SMALL_PROD_V3_KWARGS,
    )
    for flat in ("disable_source", "fhr_mean", "fhr_std", "up_mean", "up_std"):
        if flat in vae:
            kwargs[flat] = vae[flat]
    return kwargs


def test_all_arms_listed(base_config):
    r"""The config declares the expected arm ladder."""
    arms = set(list_arms(base_config))
    assert {"prod", "frontend_noncausal", "single_stride", "no_antialias", "no_gated",
            "disable_source", "am_carrier_prod"} <= arms


@pytest.mark.parametrize("arm", [
    "prod", "frontend_noncausal", "single_stride", "no_antialias", "no_gated",
    "disable_source", "am_carrier_prod",
])
def test_arm_resolves_and_constructs(base_config, arm):
    r"""Each arm resolves and constructs a `SeqVaeRawV4` (or the leaky subclass)."""
    cfg = resolve_arm_v4(base_config, arm)
    kwargs = _small_model_kwargs(cfg)
    cls = LeakyRawFrontendSeqVaeRawV4 if arm_uses_leaky_frontend(cfg, arm) else SeqVaeRawV4
    model = cls(**kwargs)
    assert model is not None

    if arm == "single_stride":
        assert cfg["model_config"]["VAE_model"]["frontend"]["stages"] == [16]
    if arm == "disable_source":
        assert cfg["model_config"]["VAE_model"]["disable_source"] is True
        assert model.disable_source is True
    if arm == "frontend_noncausal":
        assert isinstance(model, LeakyRawFrontendSeqVaeRawV4)
    if arm == "am_carrier_prod":
        assert cfg["experiment"]["data_tag"] == "G1_raw_v4_am"


@pytest.mark.parametrize("arm,key,expected", [
    ("single_stride", "stages", [16]),
    ("no_antialias", "antialias", False),
    ("no_gated", "gated", False),
])
def test_pilotize_preserves_frontend_ablation(base_config, arm, key, expected):
    r"""``_pilotize_config`` must carry each frontend-ablation arm's delta into the shrunk pilot config.

    Drives the REAL trainer pilot path (not the test-local ``_small_model_kwargs`` copy of the overlay)
    so a regression that wholesale-replaced the frontend with the small-prod widths -- silently
    de-ablating ``single_stride`` / ``no_antialias`` / ``no_gated`` in a ``--pilot`` sweep -- is caught.
    """
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.trainer_v4 import (
        _pilotize_config,
    )

    cfg = _pilotize_config(resolve_arm_v4(base_config, arm))
    frontend = cfg["model_config"]["VAE_model"]["frontend"]
    assert frontend[key] == expected
    if arm == "single_stride":
        # channels dropped so the frontend's len(stages)==1 branch picks its own channel count.
        assert "channels" not in frontend
