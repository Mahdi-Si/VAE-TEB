r"""The shipped configurations, against the constructor they build.

One failure this file exists to catch, and it is silent in both directions. The experiment driver
forwards a ``model_config.VAE_model`` key only when ``inspect.signature`` names it, so a key that
names no constructor parameter is **dropped without a word** and the run trains a different model
from the one the file describes. And a constructor parameter no configuration sets takes its
default, which may be a value no arm intends.

The second half is this architecture's own: every keyword it refuses is a parameter of its
signature, so a configuration setting one would be forwarded and would raise at startup. The
configurations must therefore not set any of them -- which is easy to violate by copying a block
from a sibling.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Set

import pytest
import yaml

from teb_vae.lag_slot_transformer_cfs.nets.core import REFUSED_KEYWORDS
from teb_vae.lag_slot_transformer_cfs.nets.model import SeqVaeLagResidualTrfCfs

#: The configuration directory this package ships.
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"

#: Keys under ``model_config.VAE_model`` that are the *task's* or the *resolver's* rather than the
#: constructor's. Each is consumed elsewhere by name, so it is not an orphan even though the
#: signature sweep drops it.
NON_CONSTRUCTOR_KEYS: Set[str] = {
    # The objective weights and the likelihood, read by the task from its hyperparameters.
    "beta_schedule",
    "kld_beta",
    "beta_prior",
    "lambda_full",
    "lambda_base",
    "likelihood",
    "free_bits",
    # The causal representation, resolved into the four channel tuples and the novelty vector.
    "causal_reach_budget_s",
    "causal_warmup_budget_steps",
    "causal_align_reference",
    "causal_align_reference_source",
    "causal_target_forecast_clock",
    "causal_leg_alignment",
    "causal_phase_operator",
}

#: Constructor parameters no configuration is expected to set, with the reason.
UNSET_BY_DESIGN: Set[str] = {
    # Resolved from the warm-up budget, never written by hand.
    "target_keep_index",
    "target_warmup_steps",
    "source_keep_index",
    "source_warmup_steps",
    "target_align_delays",
    "source_align_delays",
    "target_novelty_frac",
    "target_forecast_shift",
    # Weight initialisation is not a configuration decision.
    "init_weights",
} | set(REFUSED_KEYWORDS)


def load(name: str) -> Dict[str, Any]:
    """Load one configuration, resolving its ``base:`` chain by a shallow merge.

    Shallow per top-level block, which is what the shipped loader does and all this file needs:
    every key it inspects lives one level under ``model_config`` or ``general_config``.

    Args:
        name: The configuration filename.

    Returns:
        The merged mapping.
    """
    raw = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    base_name = raw.pop("base", None)
    if base_name is None:
        return raw
    merged = load(base_name)
    for block, value in raw.items():
        if isinstance(value, dict) and isinstance(merged.get(block), dict):
            for key, inner in value.items():
                if isinstance(inner, dict) and isinstance(merged[block].get(key), dict):
                    merged[block][key].update(inner)
                else:
                    merged[block][key] = inner
        else:
            merged[block] = value
    return merged


def vae_block(name: str) -> Dict[str, Any]:
    """The model block of one configuration.

    Args:
        name: The configuration filename.

    Returns:
        The mapping under ``model_config.VAE_model``.
    """
    return load(name)["model_config"]["VAE_model"]


#: Every configuration this package ships, discovered rather than written out.
#:
#: Discovered here and written out nowhere, which is the opposite of the entry-point tuple's rule
#: and for the opposite reason: an entry point that forgot the launch convention must fail rather
#: than go unseen, while an arm added to the directory and forgotten here would be an arm nothing
#: checked. The two silent failures below -- a key naming no constructor parameter, and a key the
#: constructor refuses -- are exactly the ones a new arm is most likely to introduce, because an arm
#: is written by copying a sibling.
CONFIGS = tuple(sorted(path.name for path in CONFIG_ROOT.glob("*.yaml")))


@pytest.mark.parametrize("name", CONFIGS)
def test_every_model_key_is_a_constructor_parameter_or_a_named_exception(name: str) -> None:
    """The silent half: a key naming nothing is dropped by the sweep and changes no model.

    Args:
        name: The configuration to check.
    """
    parameters = set(inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters)
    orphans = sorted(set(vae_block(name)) - parameters - NON_CONSTRUCTOR_KEYS)
    assert orphans == [], orphans


@pytest.mark.parametrize("name", CONFIGS)
def test_no_configuration_sets_a_refused_keyword(name: str) -> None:
    """Every one of them is a parameter, so setting one would refuse the run at startup.

    Easy to violate by copying a block from the lag-attentive sibling, and the failure would arrive
    only once someone launched a real run.

    Args:
        name: The configuration to check.
    """
    offending = sorted(set(vae_block(name)) & set(REFUSED_KEYWORDS))
    assert offending == []


def test_the_production_config_sets_every_parameter_that_shapes_a_run() -> None:
    """The other half: a parameter no configuration sets takes a default no arm chose.

    The exceptions are named and reasoned in :data:`UNSET_BY_DESIGN` rather than left implicit.
    """
    parameters = inspect.signature(SeqVaeLagResidualTrfCfs.__init__).parameters
    expected = {
        name
        for name in parameters
        if name != "self" and name not in UNSET_BY_DESIGN
    }
    missing = sorted(expected - set(vae_block("default.yaml")))
    assert missing == [], missing


def test_the_production_geometry_is_the_arm_the_design_fixes() -> None:
    """The values every reported number is over, read from the file rather than assumed."""
    block = vae_block("default.yaml")
    assert block["sequence_length"] == 300
    assert block["horizon"] == 10
    assert block["warmup_period"] == 134
    assert block["causal_warmup_budget_steps"] == 134
    assert block["max_lag"] == 90
    assert block["c_y"] == 80
    assert block["c_u"] == 46
    assert block["d_model"] == 128
    assert block["d_z"] == 64
    assert block["anchor_stride"] == 5
    assert block["causal_target_forecast_clock"] == "stored"
    assert block["causal_phase_operator"] == "integer_harmonic_v1"
    assert block["causal_align_reference"] is None
    assert block["causal_align_reference_source"] is None


def test_the_stride_and_the_model_width_satisfy_the_constructor() -> None:
    """Two refusals the configuration could trip, checked without building the whole model."""
    block = vae_block("default.yaml")
    assert 1 <= block["anchor_stride"] <= block["horizon"]
    assert block["d_model"] % 2 == 0


def test_the_diagnostic_page_is_not_enabled() -> None:
    """It draws an attention matrix this architecture does not compute.

    Enabled, it raises inside a handler that warns and continues -- so the run would keep going,
    the figure would never appear, and the suite would stay green.
    """
    callbacks = load("default.yaml")["advanced_config"]["callbacks"]
    assert "lag_attn_rws_plotting" not in callbacks


def test_the_target_blocks_are_normalised() -> None:
    """Without them the target arrives at its stored scale and the Gaussian score is meaningless.

    A run configured this way trains to completion with nothing raising, which is why the shared
    entry point guards it and why the configuration is checked here as well.
    """
    fields = load("default.yaml")["dataset_config"]["dataloader_config"]["normalize_fields"]
    assert {"fhr_st", "fhr_ph"} <= set(fields)


def test_the_tiling_phase_key_fields_are_loaded() -> None:
    """Without either, every segment sits on one tile grid forever and no count differs."""
    kwargs = load("default.yaml")["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    assert {"guid", "epoch"} <= set(kwargs["load_fields"])


def test_the_tiny_config_points_at_the_committed_fixture_shard() -> None:
    """A smoke run that needed the production shards would not be a smoke run."""
    dataset = load("tiny.yaml")["dataset_config"]
    for key in ("vae_train_datasets", "vae_test_datasets"):
        assert all(path.startswith("teb_vae/") for path in dataset[key])
    assert dataset["stat_path"].startswith("teb_vae/")


def test_the_tiny_config_exercises_the_chunked_path() -> None:
    """An unchunked-only smoke run would leave the chunked accumulation untried on the real path."""
    block = vae_block("tiny.yaml")
    assert block["anchor_chunk"] is not None
    assert block["lag_chunk"] is not None
    assert block["lag_chunk"] < block["max_lag"] + 1


def test_the_tiny_config_inherits_the_data_geometry_from_the_parent() -> None:
    """Capacity is shrunk; the geometry describes the data and must not be."""
    tiny, production = vae_block("tiny.yaml"), vae_block("default.yaml")
    for key in (
        "sequence_length",
        "horizon",
        "warmup_period",
        "causal_warmup_budget_steps",
        "c_y",
        "c_u",
        "anchor_stride",
    ):
        assert tiny[key] == production[key], key
    assert tiny["d_model"] < production["d_model"]
