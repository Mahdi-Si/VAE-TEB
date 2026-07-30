r"""The ``eval_config`` block is validated at load, and a misspelling raises rather than defaults.

YAML absorbs whatever it is given. ``max_sample`` instead of ``max_samples`` parses cleanly, is
never read, and means "no cap" -- a run that took hours instead of minutes and reported nothing
about why. Every key is therefore checked against a closed set before a model, a loader or an
output directory exists.

Two of the checks are subtler than they look, and both are pinned below. ``bool`` is an ``int``
subclass in Python, so ``caps: {predictions: true}`` would validate and then retain one sample; a
cap of ``0`` would retain nothing while still reporting success, which is indistinguishable in
the output from an analysis that found nothing.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_rws.eval.config_schema import DEFAULTS, VALID_KEYS, validate_eval_config


def _config(**block) -> dict:
    return {"eval_config": dict(block)}


# ---------------------------------------------------------------------------
# The key set
# ---------------------------------------------------------------------------
def test_an_unknown_key_raises_and_names_the_valid_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_eval_config(_config(max_sample=100))

    message = str(excinfo.value)
    assert "'max_sample'" in message
    # The valid set is in the message, so the fix does not require finding this module.
    for key in sorted(VALID_KEYS):
        assert key in message


def test_absent_keys_are_filled_from_the_defaults() -> None:
    resolved = validate_eval_config({})
    assert set(resolved) == set(VALID_KEYS)
    assert resolved == DEFAULTS

    partial = validate_eval_config(_config(seed=7))
    assert partial["seed"] == 7
    assert partial["num_mc_samples"] == DEFAULTS["num_mc_samples"]


def test_the_two_knobs_that_would_let_a_config_decide_a_finding_are_absent() -> None:
    """An operator who could widen the significance level or the trajectory bin width could make
    a difference appear or disappear from a config file. Neither is a setting."""
    assert "alpha" not in VALID_KEYS
    assert "trajectory_bin_hours" not in VALID_KEYS


def test_the_defaults_match_the_readout_module_they_restate() -> None:
    """``config_schema`` must stay a stdlib parse, so three defaults are written out rather than
    imported from the module that owns them. This is the pin that keeps the two equal."""
    from teb_vae.lag_attn_rws.eval import metrics

    assert DEFAULTS["num_mc_samples"] == metrics.DEFAULT_NUM_SAMPLES
    assert DEFAULTS["prior_shuffle_min_nats"] == metrics.DEFAULT_PRIOR_SHUFFLE_MIN_NATS
    assert DEFAULTS["min_active_dims"] == metrics.DEFAULT_MIN_ACTIVE_DIMS


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------
def test_a_boolean_cap_is_rejected_rather_than_silently_capping_at_one() -> None:
    with pytest.raises(ValueError, match="caps.predictions must be an integer"):
        validate_eval_config(_config(caps={"predictions": True}))


def test_a_zero_cap_is_rejected() -> None:
    """It would retain nothing while reporting success, which reads as "found nothing"."""
    with pytest.raises(ValueError, match="caps.samples must be >= 1"):
        validate_eval_config(_config(caps={"samples": 0}))


def test_a_null_cap_means_no_cap_and_a_positive_one_survives() -> None:
    resolved = validate_eval_config(_config(caps={"samples": None, "predictions": 20}))
    assert resolved["caps"] == {"samples": None, "predictions": 20}


def test_caps_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="caps must be a mapping"):
        validate_eval_config(_config(caps=[1, 2]))


# ---------------------------------------------------------------------------
# The scalar knobs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "block, pattern",
    [
        ({"seed": -1}, "seed must be >= 0"),
        ({"seed": 2**32}, "numpy's bound"),
        ({"seed": True}, "seed must be an integer"),
        ({"num_mc_samples": 0}, "num_mc_samples must be >= 1"),
        ({"max_samples": 0}, "max_samples must be >= 1"),
        ({"max_samples": True}, "max_samples must be an integer"),
        ({"prior_shuffle_min_nats": -0.5}, "prior_shuffle_min_nats must be >= 0"),
        ({"prior_shuffle_min_nats": float("nan")}, "prior_shuffle_min_nats must be finite"),
        ({"prior_shuffle_min_nats": True}, "prior_shuffle_min_nats must be a number"),
        ({"min_active_dims": 0}, "min_active_dims must be >= 1"),
        ({"event_lag_window_s": 0.0}, "event_lag_window_s must be >= 1"),
        ({"bootstrap_resamples": 10}, "bootstrap_resamples must be >= 100"),
    ],
)
def test_each_out_of_range_value_raises_naming_its_key(block: dict, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        validate_eval_config(_config(**block))


def test_max_samples_may_be_null_and_a_positive_integer() -> None:
    assert validate_eval_config(_config(max_samples=None))["max_samples"] is None
    assert validate_eval_config(_config(max_samples=250))["max_samples"] == 250


def test_an_integer_is_accepted_where_a_float_is_expected() -> None:
    """YAML writes ``120`` for ``120.0``; refusing that would be a formatting rule, not a check."""
    resolved = validate_eval_config(_config(event_lag_window_s=120, prior_shuffle_min_nats=1))
    assert resolved["event_lag_window_s"] == pytest.approx(120.0)
    assert isinstance(resolved["event_lag_window_s"], float)
    assert resolved["prior_shuffle_min_nats"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The block itself
# ---------------------------------------------------------------------------
def test_an_absent_block_validates_to_the_defaults() -> None:
    assert validate_eval_config({"general_config": {}}) == DEFAULTS


def test_a_non_mapping_block_raises() -> None:
    with pytest.raises(ValueError, match="eval_config must be a mapping"):
        validate_eval_config({"eval_config": [1, 2, 3]})


def test_validation_does_not_mutate_the_caller_s_block() -> None:
    """The merged config is dumped into the run directory; validation must not edit it."""
    config = _config(caps={"samples": 5})
    validate_eval_config(config)
    assert config["eval_config"]["caps"] == {"samples": 5}
