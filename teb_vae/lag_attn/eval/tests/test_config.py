"""The eval configs merge to what the pipeline expects, and bad ``eval_config`` is refused.

Two concerns, deliberately in one file because they answer the same question from two sides:
what the shipped config resolves to relative to the training config it chains off, and what the
validator refuses. A merged config that looks right but silently keeps a training-time filter is
as broken as one that fails to parse, and only the first half of this file can catch it.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn.eval.config_schema import VALID_KEYS, validate_eval_config
from teb_vae.lag_attn.eval.tests.conftest import EVAL_TINY_CONFIG

#: Repo-root-relative, matching every documented invocation.
EVAL_CONFIG = "teb_vae/lag_attn/eval/configs/eval.yaml"

#: The training config ``eval.yaml`` chains off through ``base:``.
TRAINING_CONFIG = "teb_vae/lag_attn/configs/default.yaml"


@pytest.fixture(scope="module")
def eval_config(repo_root):
    """The merged production eval config."""
    return load_config(str(repo_root / EVAL_CONFIG))


@pytest.fixture(scope="module")
def training_config(repo_root):
    """The merged training config the eval config inherits from."""
    return load_config(str(repo_root / TRAINING_CONFIG))


@pytest.fixture(scope="module")
def eval_tiny_config(repo_root):
    """The merged test-suite eval config."""
    return load_config(str(repo_root / EVAL_TINY_CONFIG))


# ---------------------------------------------------------------------------
# The merged configs
# ---------------------------------------------------------------------------
def test_eval_config_inherits_the_training_geometry_and_objective(eval_config, training_config):
    """The ``base:`` chain is what keeps the eval geometry from drifting from the run's.

    The objective is inherited too, and is what preflight reconciles against the checkpoint.
    """
    eval_vae = eval_config["model_config"]["VAE_model"]
    training_vae = training_config["model_config"]["VAE_model"]
    for key in ("c_y", "c_u", "sequence_length", "likelihood", "sigma_obs"):
        assert eval_vae[key] == training_vae[key], key


def test_eval_config_points_at_one_shard_per_canonical_subgroup(eval_config):
    """Every canonical subgroup, not the healthy-only pretraining pair."""
    shards = eval_config["dataset_config"]["vae_test_datasets"]
    assert len(shards) == len(labels.CANONICAL_SUBGROUPS)
    assert {labels.subgroup_of(path) for path in shards} == set(labels.CANONICAL_SUBGROUPS)


def test_eval_config_extends_load_fields(eval_config):
    """The model's five fields survive the wholesale list replacement, plus the four new ones."""
    fields = eval_config["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    for name in ("fhr_st", "fhr_ph", "up_st", "up_ph", "weight", "guid"):
        assert name in fields, f"{name} is consumed by the model and must survive the merge"
    for name in ("target", "cs_label", "bg_label", "epoch"):
        assert name in fields


def test_eval_config_clears_inherited_training_filters(eval_config, training_config):
    """Each of these silently *drops* eval samples rather than failing, which is the hazard."""
    kwargs = eval_config["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    assert kwargs["epoch_min"] is None, "inherited epoch_min would couple coverage to the extraction floor"
    assert kwargs["epoch_max"] is None
    assert kwargs["label"] is None, "the label filter compares floats for equality"
    # trim_minutes must NOT be cleared: it must keep matching the stats file.
    training_kwargs = training_config["dataset_config"]["dataloader_config"]["dataset_kwargs"]
    assert kwargs["trim_minutes"] == training_kwargs["trim_minutes"]


def test_eval_config_block_validates(eval_config):
    """The shipped block passes its own validator, and carries every documented key."""
    resolved = validate_eval_config(eval_config)
    assert set(resolved) == set(VALID_KEYS)
    assert resolved["bands"], "the lag ablation needs at least one band"
    max_lag = eval_config["model_config"]["VAE_model"]["max_lag"]
    assert all(high <= max_lag for _, high in resolved["bands"].values())


def test_eval_tiny_config_matches_the_suite_checkpoint_objective(eval_tiny_config):
    """The tiny checkpoint is built with the task's PROD_HPARAMS; the config must agree.

    Preflight raises on a disagreement between the config and the checkpoint's own
    ``hyper_parameters`` rather than silently preferring either, so this is the setting that
    makes the suite's end-to-end run possible at all.
    """
    from teb_vae.lag_attn.tests.conftest import PROD_HPARAMS

    vae = eval_tiny_config["model_config"]["VAE_model"]
    assert vae["beta_schedule"] == PROD_HPARAMS["beta_schedule"]
    assert vae["kld_beta"] == PROD_HPARAMS["kld_beta"]


# ---------------------------------------------------------------------------
# Validation: one case per rejection path
# ---------------------------------------------------------------------------
def _config_with(block, max_lag: int = 90):
    """A minimal config carrying ``block`` as its ``eval_config``."""
    return {"model_config": {"VAE_model": {"max_lag": max_lag}}, "eval_config": block}


def test_unknown_key_raises_and_names_the_valid_set():
    """The whole point: ``max_sample`` would otherwise parse and silently mean 'no cap'."""
    with pytest.raises(ValueError, match="unknown eval_config key") as excinfo:
        validate_eval_config(_config_with({"max_sample": 10}))
    message = str(excinfo.value)
    assert "'max_sample'" in message
    assert "max_samples" in message, "the message must list the valid set"


def test_band_beyond_max_lag_raises():
    with pytest.raises(ValueError, match="exceeds the model's max_lag"):
        validate_eval_config(_config_with({"bands": {"too_long": [80, 95]}}))


def test_empty_band_raises():
    """lo > hi masks every lag, which entmax15 cannot evaluate at all."""
    with pytest.raises(ValueError, match="is empty"):
        validate_eval_config(_config_with({"bands": {"backwards": [30, 10]}}))


def test_malformed_band_raises():
    with pytest.raises(ValueError, match="inclusive \\[lo, hi\\] lag pair"):
        validate_eval_config(_config_with({"bands": {"scalar": 5}}))


@pytest.mark.parametrize("cap", [-1, 0, 2.5, "many", True])
def test_bad_cap_raises(cap):
    """Negative, zero, fractional, non-numeric -- and ``True``, which is an int in Python."""
    with pytest.raises(ValueError):
        validate_eval_config(_config_with({"caps": {"predictions": cap}}))


def test_cap_of_none_means_uncapped():
    resolved = validate_eval_config(_config_with({"caps": {"predictions": None}}))
    assert resolved["caps"]["predictions"] is None


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5])
def test_out_of_range_seed_raises(seed):
    with pytest.raises(ValueError):
        validate_eval_config(_config_with({"seed": seed}))


@pytest.mark.parametrize("floor", [-0.1, float("nan"), float("inf"), "low"])
def test_bad_health_probe_floor_raises(floor):
    with pytest.raises(ValueError):
        validate_eval_config(_config_with({"health_probe_floor": floor}))


def test_the_removed_up_shift_key_is_refused_as_unknown():
    """The stored timeline is canonical, so a config still naming the removed ``up_shift_secs``
    key must fail loudly rather than be honoured or ignored."""
    with pytest.raises(ValueError, match="unknown eval_config key"):
        validate_eval_config(_config_with({"up_shift_secs": -20.0}))


def test_missing_block_falls_back_to_defaults():
    """A partial block is legitimate; only a *misspelled* key is not."""
    resolved = validate_eval_config({"model_config": {"VAE_model": {"max_lag": 90}}})
    assert set(resolved) == set(VALID_KEYS)
    assert resolved["max_samples"] is None
    assert resolved["bands"] == {}
