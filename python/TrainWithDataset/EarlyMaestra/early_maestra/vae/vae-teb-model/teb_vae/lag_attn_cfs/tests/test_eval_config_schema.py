r"""The ``eval_config`` block is validated at load, and a misspelling raises rather than defaults.

YAML absorbs whatever it is given. ``max_sample`` instead of ``max_samples`` parses cleanly, is
never read, and means "no cap" -- a run that took hours instead of minutes and reported nothing
about why. Every key is therefore checked against a closed set before a model, a loader or an
output directory exists.

Two of the checks are subtler than they look, and both are pinned below. ``bool`` is an ``int``
subclass in Python, so ``caps: {predictions: true}`` would validate and then retain one sample; a
cap of ``0`` would retain nothing while still reporting success, which is indistinguishable in
the output from an analysis that found nothing.

This package's own key is ``clock_margin_min_nats``, and what it is pinned on here is that it
ships **unset**. A default threshold would decide the headline availability-clock verdict on the
very first production runs -- which are the runs that were supposed to measure where the threshold
belongs -- and neither a FAIL nor a PASS from a guessed number is readable afterwards.
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


def test_the_key_set_diverges_from_the_siblings_by_exactly_the_clock_margin() -> None:
    """The fork's key set is not free to drift. Anything else added or dropped here is a divergence
    in the run's configuration surface that nothing else in the output would record."""
    from teb_vae.lag_attn_rws.eval.config_schema import VALID_KEYS as SIBLING_KEYS

    assert VALID_KEYS - SIBLING_KEYS == {"clock_margin_min_nats"}
    assert SIBLING_KEYS - VALID_KEYS == set()


def test_the_three_knobs_that_would_let_a_config_decide_a_finding_are_absent() -> None:
    """An operator who could widen the significance level or the trajectory bin width could make a
    difference appear or disappear from a config file. ``anchor_stride`` is this cell's own
    addition to that list and is the worst of the three: the evaluation decodes the dense anchor
    set, and a stride would change the population every number is computed over rather than only
    how it is tested."""
    assert "alpha" not in VALID_KEYS
    assert "trajectory_bin_hours" not in VALID_KEYS
    assert "anchor_stride" not in VALID_KEYS


def test_the_defaults_match_the_readout_module_they_restate() -> None:
    """``config_schema`` must stay a stdlib parse, so three defaults are written out rather than
    imported from the module that owns them. This is the pin that keeps the two equal, and it
    activates by itself once ``metrics.py`` exists."""
    metrics = pytest.importorskip(
        "teb_vae.lag_attn_cfs.eval.metrics",
        reason="the readout module does not exist yet; there is nothing to pin against",
    )

    assert DEFAULTS["num_mc_samples"] == metrics.DEFAULT_NUM_SAMPLES
    assert DEFAULTS["prior_shuffle_min_nats"] == metrics.DEFAULT_PRIOR_SHUFFLE_MIN_NATS
    assert DEFAULTS["min_active_dims"] == metrics.DEFAULT_MIN_ACTIVE_DIMS


# ---------------------------------------------------------------------------
# The clock margin, which ships unset
# ---------------------------------------------------------------------------
def test_the_clock_margin_defaults_to_none_so_an_omitted_key_is_inconclusive() -> None:
    """The whole point of the key. A config that never mentions it must get an explicit
    INCONCLUSIVE verdict with the measured difference beside it, not a threshold nobody chose."""
    assert DEFAULTS["clock_margin_min_nats"] is None
    assert validate_eval_config({})["clock_margin_min_nats"] is None


def test_an_explicit_null_clock_margin_resolves_to_none() -> None:
    """``null`` in the shipped delta and an absent key must mean the same thing; the delta writes
    it out so a reader sees the setting rather than inferring it from silence."""
    assert validate_eval_config(_config(clock_margin_min_nats=None))["clock_margin_min_nats"] is None


def test_a_set_clock_margin_survives_as_a_float() -> None:
    resolved = validate_eval_config(_config(clock_margin_min_nats=0.5))
    assert resolved["clock_margin_min_nats"] == pytest.approx(0.5)
    assert isinstance(resolved["clock_margin_min_nats"], float)


@pytest.mark.parametrize("value", [0, 0.0, -0.5])
def test_a_clock_margin_that_could_never_fail_is_refused(value) -> None:
    """Zero passes on any non-negative difference and a negative value can never fail at all, so
    either would render an active verdict inert while still printing PASS."""
    with pytest.raises(ValueError, match="clock_margin_min_nats must be >= 0.001"):
        validate_eval_config(_config(clock_margin_min_nats=value))


def test_a_boolean_clock_margin_is_refused() -> None:
    r"""``bool`` is a numeric subtype, so ``true`` would validate and then mean a margin of
    $1$ nat -- a threshold nobody typed, gating the headline verdict."""
    with pytest.raises(ValueError, match="clock_margin_min_nats must be a number"):
        validate_eval_config(_config(clock_margin_min_nats=True))


def test_a_non_finite_clock_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="clock_margin_min_nats must be finite"):
        validate_eval_config(_config(clock_margin_min_nats=float("nan")))


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


# =================================================================================================
# figure_format
# =================================================================================================
def test_the_figure_format_defaults_to_none_so_a_run_keeps_the_pdf_default() -> None:
    """``None`` rather than ``"pdf"``: the default lives in ``figures``, and only there."""
    assert DEFAULTS["figure_format"] is None
    assert validate_eval_config({"eval_config": {}})["figure_format"] is None


@pytest.mark.parametrize(
    ("given", "expected"),
    [("svg", "svg"), ("SVG", "svg"), (".png", "png"), ("  pdf  ", "pdf")],
    ids=["plain", "upper", "dotted", "padded"],
)
def test_an_operator_typed_format_is_normalised(given: str, expected: str) -> None:
    """A config value is hand-typed, so the shapes a hand produces all have to resolve."""
    resolved = validate_eval_config({"eval_config": {"figure_format": given}})

    assert resolved["figure_format"] == expected


def test_a_format_matplotlib_cannot_write_is_refused_and_names_the_supported_set() -> None:
    """The failure this key exists to prevent is a typo that would otherwise reach ``savefig``."""
    with pytest.raises(ValueError, match=r"figure_format must be one of"):
        validate_eval_config({"eval_config": {"figure_format": "docx"}})


def test_a_non_string_format_is_refused_before_it_reaches_matplotlib() -> None:
    with pytest.raises(ValueError, match=r"figure_format must be a string"):
        validate_eval_config({"eval_config": {"figure_format": 3}})


def test_the_supported_set_is_the_installed_matplotlib_s_own() -> None:
    """Read from the live build rather than restated here, so it cannot go stale against it."""
    from teb_vae.lag_attn.eval.figures import SUPPORTED_FIGURE_FORMATS

    for accepted in ("pdf", "svg", "png"):
        assert accepted in SUPPORTED_FIGURE_FORMATS
        assert validate_eval_config(
            {"eval_config": {"figure_format": accepted}}
        )["figure_format"] == accepted


# =================================================================================================
# max_hours_before_delivery
# =================================================================================================
def test_the_horizon_defaults_to_none_so_a_run_evaluates_every_segment() -> None:
    """``None`` is not a missing value here but the shipped setting: no bound."""
    assert DEFAULTS["max_hours_before_delivery"] is None
    assert validate_eval_config({"eval_config": {}})["max_hours_before_delivery"] is None


def test_a_horizon_survives_as_a_float() -> None:
    """An operator writing ``4`` means four hours, and an int must not stay an int downstream."""
    resolved = validate_eval_config({"eval_config": {"max_hours_before_delivery": 4}})

    assert resolved["max_hours_before_delivery"] == pytest.approx(4.0)
    assert isinstance(resolved["max_hours_before_delivery"], float)


@pytest.mark.parametrize("value", [0.25, 0.0, -1.0], ids=["under-a-bin", "zero", "negative"])
def test_a_horizon_with_no_whole_window_in_it_is_refused(value: float) -> None:
    """Below one trajectory bin the bound empties every clock rather than narrowing it, which
    reads as "the analysis found nothing" instead of "you asked for less than one window"."""
    with pytest.raises(ValueError, match=r"max_hours_before_delivery"):
        validate_eval_config({"eval_config": {"max_hours_before_delivery": value}})


def test_a_non_finite_or_boolean_horizon_is_refused() -> None:
    """``True`` is an ``int`` in Python and would otherwise validate as a one-hour bound."""
    for value in (float("nan"), float("inf"), True, "four"):
        with pytest.raises(ValueError, match=r"max_hours_before_delivery"):
            validate_eval_config({"eval_config": {"max_hours_before_delivery": value}})


def test_the_horizon_floor_is_one_trajectory_bin() -> None:
    """The floor restates ``cohort``'s bin width rather than importing it -- this module stays a
    stdlib parse -- so the two are pinned together here instead."""
    from teb_vae.lag_attn_cfs.eval.cohort import TRAJECTORY_BIN_HOURS
    from teb_vae.lag_attn_cfs.eval.config_schema import _MIN_HORIZON_HOURS

    assert _MIN_HORIZON_HOURS == TRAJECTORY_BIN_HOURS
