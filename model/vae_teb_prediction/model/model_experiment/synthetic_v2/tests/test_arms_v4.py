r"""S0-T04: arm resolution (deltas author under the ``model_raw`` paths; leaky-class marker)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model.vae_teb_prediction.model.model_experiment.synthetic_v2 import arms_v4

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_single_stride_sets_frontend_stages(config: dict) -> None:
    r"""``single_stride`` overrides the front-end stride schedule to a single stride-16."""
    resolved = arms_v4.resolve_arm_v4(config, "single_stride")
    assert resolved["model_config"]["VAE_model"]["frontend"]["stages"] == [16]


def test_disable_source_sets_flag(config: dict) -> None:
    r"""``disable_source`` flips the model's no-UP flag."""
    resolved = arms_v4.resolve_arm_v4(config, "disable_source")
    assert resolved["model_config"]["VAE_model"]["disable_source"] is True


def test_am_carrier_arm_switches_data_tag_and_render(config: dict) -> None:
    r"""``am_carrier_prod`` points the cache at the am ``data_tag`` and the am render mode."""
    resolved = arms_v4.resolve_arm_v4(config, "am_carrier_prod")
    assert resolved["experiment"]["data_tag"] == "G1_raw_v4_am"
    assert resolved["benchmarks"]["G1_raw_v4"]["raw"]["render_mode"] == "am_carrier"


def test_none_arm_returns_config_unchanged(config: dict) -> None:
    r"""The arm-less path returns the input object unchanged."""
    assert arms_v4.resolve_arm_v4(config, None) is config


def test_unknown_arm_raises(config: dict) -> None:
    r"""An unknown arm name raises ``ValueError``."""
    with pytest.raises(ValueError):
        arms_v4.resolve_arm_v4(config, "does_not_exist")


def test_deep_merge_preserves_sibling_frontend_keys(config: dict) -> None:
    r"""An arm delta merges into the front-end block without dropping its sibling keys."""
    resolved = arms_v4.resolve_arm_v4(config, "no_antialias")
    frontend = resolved["model_config"]["VAE_model"]["frontend"]
    assert frontend["antialias"] is False
    # The rest of the front-end block survives the merge.
    assert frontend["gated"] is True
    assert frontend["channels"] == [32, 64, 96, 128]


def test_frontend_noncausal_is_leaky_class(config: dict) -> None:
    r"""``frontend_noncausal`` is selected by class (the leaky-front-end marker), others are not."""
    assert arms_v4.arm_uses_leaky_frontend(config, "frontend_noncausal") is True
    for arm in ("prod", "single_stride", "disable_source", "am_carrier_prod", None):
        assert arms_v4.arm_uses_leaky_frontend(config, arm) is False


def test_leaky_marker_check_rejects_unknown_arm(config: dict) -> None:
    r"""The leaky-marker check raises on an unknown arm, mirroring ``resolve_arm``."""
    with pytest.raises(ValueError):
        arms_v4.arm_uses_leaky_frontend(config, "nope")


def test_list_arms(config: dict) -> None:
    r"""``list_arms`` enumerates every configured arm."""
    arms = set(arms_v4.list_arms(config))
    assert {"prod", "frontend_noncausal", "single_stride", "no_antialias",
            "no_gated", "disable_source", "am_carrier_prod"} <= arms
