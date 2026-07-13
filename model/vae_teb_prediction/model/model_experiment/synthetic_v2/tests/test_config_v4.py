r"""S0-T01: contract tests for ``config_synth_v4.yaml``.

Assert the hybrid config carries a valid ``model_raw`` 4-key model schema (so the reused
:class:`GraphModelVaeTebRawV4Trainer` can read it) AND the synthetic-only top-level blocks
(``experiment`` / ``benchmarks.G1_raw_v4`` / ``arms``) the v4 data/eval stages consume.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.v4

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_synth_v4.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    r"""Parse ``config_synth_v4.yaml`` once for the module."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_yaml_parses_and_is_a_mapping(config: dict) -> None:
    r"""The file parses to a top-level mapping."""
    assert isinstance(config, dict)
    assert config, "config parsed empty"


def test_model_raw_four_key_schema(config: dict) -> None:
    r"""The four ``model_raw`` top-level keys are present."""
    for key in ("general_config", "model_config", "dataset_config", "advanced_config"):
        assert key in config, f"missing model_raw schema key {key!r}"


def test_frontend_block_present(config: dict) -> None:
    r"""``model_config.VAE_model.frontend`` exists with the raw geometry kwargs."""
    vae = config["model_config"]["VAE_model"]
    assert "frontend" in vae
    frontend = vae["frontend"]
    for key in ("stages", "channels", "antialias", "gated", "norm_kind"):
        assert key in frontend, f"frontend missing {key!r}"
    assert vae["raw_len"] == 5280
    assert vae["decimation"] == 16


def test_trim_minutes_is_null(config: dict) -> None:
    r"""The loader stays untrimmed (raw_len 5280 / decimated 330 requires ``trim_minutes`` null)."""
    dl = config["dataset_config"]["dataloader_config"]
    assert dl["dataset_kwargs"]["trim_minutes"] is None
    assert dl["normalize_fields"] == ["fhr", "up"]


def test_mlflow_nested_under_tracking(config: dict) -> None:
    r"""mlflow lives at ``advanced_config.tracking.mlflow`` (read by ``graph_model_base``)."""
    tracking = config["advanced_config"]["tracking"]
    assert "mlflow" in tracking
    assert tracking["mlflow"]["enabled"] is True


def test_beta_schedule_anneals_up(config: dict) -> None:
    r"""$\beta$ anneals UP so $K$ grows from its zero-init (``start < end``)."""
    beta = config["model_config"]["VAE_model"]["beta_schedule"]
    assert float(beta["start"]) < float(beta["end"])


def test_experiment_block_and_distinct_data_tag(config: dict) -> None:
    r"""``experiment`` selects the benchmark and keys a cache distinct from ``tag``."""
    experiment = config["experiment"]
    assert experiment["benchmark"] == "G1_raw_v4"
    assert experiment["data_tag"] != experiment["tag"]


def test_benchmark_has_data_raw_mix(config: dict) -> None:
    r"""``benchmarks.G1_raw_v4`` carries non-empty ``data`` / ``raw`` / ``mix`` sub-blocks."""
    bench = config["benchmarks"]["G1_raw_v4"]
    for key in ("data", "raw", "mix"):
        assert bench.get(key), f"benchmark.{key} is empty/missing"


def test_direct_bipolar_render(config: dict) -> None:
    r"""The primary render is ``direct`` bipolar (``one_sided`` false) -- no carrier, no AM."""
    raw = config["benchmarks"]["G1_raw_v4"]["raw"]
    assert raw["render_mode"] == "direct"
    assert raw["direct"]["one_sided"] is False


def test_concentrated_grid_knobs(config: dict) -> None:
    r"""Concentrated recipe: single fixed short lag $D=8$, a TE ladder with a null anchor."""
    mix = config["benchmarks"]["G1_raw_v4"]["mix"]
    assert mix["lag_grid"] == [8]
    assert mix["lag_mode"] == "fixed"
    assert 0.0 in mix["target_te_grid"]
    assert mix["target_te_grid"] == sorted(mix["target_te_grid"])


def test_all_arms_present(config: dict) -> None:
    r"""All seven trained arms + ``am_carrier_prod`` are configured."""
    arms = config["arms"]
    expected = {
        "prod", "frontend_noncausal", "single_stride", "no_antialias",
        "no_gated", "disable_source", "am_carrier_prod",
    }
    assert expected <= set(arms), f"missing arms: {expected - set(arms)}"


def test_arm_deltas_author_under_model_raw_paths(config: dict) -> None:
    r"""Arm deltas live under the ``model_raw`` config paths (no ``model.v4`` overlay)."""
    arms = config["arms"]
    assert arms["single_stride"]["model_config"]["VAE_model"]["frontend"]["stages"] == [16]
    assert arms["disable_source"]["model_config"]["VAE_model"]["disable_source"] is True
    assert arms["no_antialias"]["model_config"]["VAE_model"]["frontend"]["antialias"] is False
    assert arms["no_gated"]["model_config"]["VAE_model"]["frontend"]["gated"] is False


def test_frontend_noncausal_is_leaky_class_marker(config: dict) -> None:
    r"""The G0 negative control is selected by class, not by a (forbidden) config norm."""
    assert config["arms"]["frontend_noncausal"].get("_leaky_class") is True


def test_am_carrier_arm_points_at_am_data_tag(config: dict) -> None:
    r"""The ``am_carrier_prod`` arm switches the cache ``data_tag`` and the render mode."""
    arm = config["arms"]["am_carrier_prod"]
    assert arm["experiment"]["data_tag"] == "G1_raw_v4_am"
    assert arm["benchmarks"]["G1_raw_v4"]["raw"]["render_mode"] == "am_carrier"
