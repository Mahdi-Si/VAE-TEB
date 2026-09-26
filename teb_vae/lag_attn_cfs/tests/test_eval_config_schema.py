r"""The ``eval_config`` block is validated at load, and a misspelling raises rather than defaults.

YAML absorbs whatever it is given. ``max_sample`` instead of ``max_samples`` parses cleanly, is
never read, and means "no cap" -- a run that took hours instead of minutes and reported nothing
about why. Every key is therefore checked against a closed set before a model, a loader or an
output directory exists.

``bool`` is an ``int`` subclass in Python, so every numeric key is asserted to refuse ``true``
rather than read it as $1$; a nullable key is validated only when set, so both halves of that are
asserted too. The key set may differ from the sibling's only by this cell's own causal readouts,
and the restated defaults are pinned against the modules that own them.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from teb_vae.lag_attn_cfs.eval.config_schema import DEFAULTS, VALID_KEYS, validate_eval_config

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_the_key_set_diverges_from_the_siblings_by_exactly_its_two_causal_readouts() -> None:
    """The fork's key set is not free to drift. Both additions belong to readouts only a causal
    cell has (the availability-clock margin and the occlusion bands), and nothing is *dropped*: a
    key the sibling validates and this fork does not would be refused on a config merged from
    theirs."""
    from teb_vae.lag_attn_rws.eval.config_schema import VALID_KEYS as SIBLING_KEYS

    assert VALID_KEYS - SIBLING_KEYS == {"clock_margin_min_nats", "occlusion_bands"}
    assert SIBLING_KEYS - VALID_KEYS == set()


def test_the_defaults_match_the_readout_module_they_restate() -> None:
    """``config_schema`` must stay a stdlib parse, so three defaults are written out rather than
    imported from the module that owns them. This is the pin that keeps the two equal."""
    from teb_vae.lag_attn_cfs.eval import metrics

    assert DEFAULTS["num_mc_samples"] == metrics.DEFAULT_NUM_SAMPLES
    assert DEFAULTS["prior_shuffle_min_nats"] == metrics.DEFAULT_PRIOR_SHUFFLE_MIN_NATS
    assert DEFAULTS["min_active_dims"] == metrics.DEFAULT_MIN_ACTIVE_DIMS


def test_the_horizon_floor_is_one_trajectory_bin() -> None:
    """The floor restates ``cohort``'s bin width rather than importing it -- this module stays a
    stdlib parse -- so the two are pinned together here instead."""
    from teb_vae.lag_attn_cfs.eval.cohort import TRAJECTORY_BIN_HOURS
    from teb_vae.lag_attn_cfs.eval.config_schema import _MIN_HORIZON_HOURS

    assert _MIN_HORIZON_HOURS == TRAJECTORY_BIN_HOURS


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
_NAN = float("nan")


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
        ({"prior_shuffle_min_nats": _NAN}, "prior_shuffle_min_nats must be finite"),
        ({"prior_shuffle_min_nats": True}, "prior_shuffle_min_nats must be a number"),
        ({"min_active_dims": 0}, "min_active_dims must be >= 1"),
        ({"event_lag_window_s": 0.0}, "event_lag_window_s must be >= 1"),
        ({"bootstrap_resamples": 10}, "bootstrap_resamples must be >= 100"),
        # A margin of zero passes on any non-negative difference and a negative one never fails,
        # so either would leave the availability-clock verdict inert while printing PASS.
        ({"clock_margin_min_nats": 0}, "clock_margin_min_nats must be >= 0.001"),
        ({"clock_margin_min_nats": -0.5}, "clock_margin_min_nats must be >= 0.001"),
        ({"clock_margin_min_nats": True}, "clock_margin_min_nats must be a number"),
        ({"clock_margin_min_nats": _NAN}, "clock_margin_min_nats must be finite"),
        # Below one trajectory bin the bound empties every clock rather than narrowing it.
        ({"max_hours_before_delivery": 0.25}, "max_hours_before_delivery"),
        ({"max_hours_before_delivery": -1.0}, "max_hours_before_delivery"),
        ({"max_hours_before_delivery": float("inf")}, "max_hours_before_delivery"),
        ({"max_hours_before_delivery": True}, "max_hours_before_delivery"),
        ({"max_hours_before_delivery": "four"}, "max_hours_before_delivery"),
        ({"figure_format": "docx"}, "figure_format must be one of"),
        ({"figure_format": 3}, "figure_format must be a string"),
        ({"caps": [1, 2]}, "caps must be a mapping"),
    ],
)
def test_each_out_of_range_value_raises_naming_its_key(block: dict, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        validate_eval_config(_config(**block))


@pytest.mark.parametrize(
    "key", ["max_samples", "clock_margin_min_nats", "figure_format", "max_hours_before_delivery"]
)
def test_a_nullable_key_accepts_null(key: str) -> None:
    """``null`` written out and an absent key mean the same thing: validated only when set."""
    assert validate_eval_config(_config(**{key: None}))[key] is None


@pytest.mark.parametrize(
    "key, value",
    [
        ("event_lag_window_s", 120),
        ("prior_shuffle_min_nats", 1),
        ("clock_margin_min_nats", 1),
        ("max_hours_before_delivery", 4),
    ],
)
def test_an_integer_is_accepted_where_a_float_is_expected(key: str, value: int) -> None:
    """YAML writes ``120`` for ``120.0``; refusing that would be a formatting rule, not a check, and
    an int must not stay an int downstream."""
    resolved = validate_eval_config(_config(**{key: value}))[key]

    assert resolved == pytest.approx(float(value))
    assert isinstance(resolved, float)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("svg", "svg"), ("SVG", "svg"), (".png", "png"), ("  pdf  ", "pdf")],
    ids=["plain", "upper", "dotted", "padded"],
)
def test_an_operator_typed_format_is_normalised(given: str, expected: str) -> None:
    """A config value is hand-typed, so the shapes a hand produces all have to resolve."""
    resolved = validate_eval_config({"eval_config": {"figure_format": given}})

    assert resolved["figure_format"] == expected


# ---------------------------------------------------------------------------
# The block itself
# ---------------------------------------------------------------------------
def test_a_non_mapping_block_raises() -> None:
    with pytest.raises(ValueError, match="eval_config must be a mapping"):
        validate_eval_config({"eval_config": [1, 2, 3]})


def test_validation_does_not_mutate_the_caller_s_block() -> None:
    """The merged config is dumped into the run directory; validation must not edit it."""
    config = _config(caps={"samples": 5})
    validate_eval_config(config)
    assert config["eval_config"]["caps"] == {"samples": 5}


# ---------------------------------------------------------------------------
# What importing it costs
# ---------------------------------------------------------------------------
def test_importing_the_module_pulls_in_no_numeric_stack() -> None:
    """A misconfigured run must cost a parse, not a checkpoint load.

    Asserted on a **fresh interpreter's** ``sys.modules`` rather than by walking this module's own
    import statements: the expensive imports that matter are transitive, and a source scan would
    pass while ``teb_vae.lag_attn.config`` quietly grew a ``pandas`` dependency two levels down.
    Inside this test session ``torch`` is already imported by other tests, so the question can only
    be asked in a process that has imported nothing else.
    """
    probe = (
        "import sys;"
        "import teb_vae.lag_attn_cfs.eval.config_schema;"
        "print(','.join(sorted(n for n in ('torch', 'matplotlib', 'pandas') if n in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "", (
        f"importing config_schema pulled in {result.stdout.strip()}; it validates a run's settings "
        f"before a model, a loader or an output directory exists and must stay a stdlib parse"
    )


# ---------------------------------------------------------------------------
# The width form of the occlusion bands
# ---------------------------------------------------------------------------
def _with_geometry(max_lag: int, **bands) -> dict:
    return {
        "eval_config": {"occlusion_bands": dict(bands)},
        "model_config": {"VAE_model": {"max_lag": max_lag}},
    }


def test_a_partition_width_resolves_against_the_model_window_and_keeps_declared_bands() -> None:
    """The derived bands come first and cover the window exactly once; a band declared beside the
    width is kept as written, which is how the fixed cross-bank bands stay a declaration."""
    resolved = validate_eval_config(_with_geometry(11, partition_width=4, head=[0, 5]))
    bands = resolved["occlusion_bands"]
    assert list(bands) == ["lags_000_003", "lags_004_007", "lags_008_011", "head"]
    assert bands["lags_008_011"] == (8, 11)
    assert bands["head"] == (0, 5)
    covered = sorted(
        lag
        for name, (lo, hi) in bands.items()
        if name.startswith("lags_")
        for lag in range(lo, hi + 1)
    )
    assert covered == list(range(12))


def test_the_remainder_folds_into_the_last_derived_band() -> None:
    """A lag left out of the partition is a lag no suppression arm removes."""
    bands = validate_eval_config(_with_geometry(9, partition_width=4))["occlusion_bands"]
    assert list(bands) == ["lags_000_003", "lags_004_009"]


def test_a_partition_width_without_geometry_is_refused() -> None:
    """There is no window to cut, and a silent pass-through would hand an integer to the mask
    builder under a band's name."""
    with pytest.raises(ValueError, match="no model geometry"):
        validate_eval_config(_config(occlusion_bands={"partition_width": 4}))


def test_a_partition_width_covering_the_window_is_refused() -> None:
    with pytest.raises(ValueError, match="whole axis"):
        validate_eval_config(_with_geometry(3, partition_width=4))


def test_a_declared_band_named_like_a_derived_one_is_refused() -> None:
    with pytest.raises(ValueError, match="also the name"):
        validate_eval_config(_with_geometry(7, partition_width=4, lags_000_003=[0, 3]))


def test_a_non_positive_width_is_refused() -> None:
    with pytest.raises(ValueError, match="partition_width"):
        validate_eval_config(_with_geometry(7, partition_width=0))
