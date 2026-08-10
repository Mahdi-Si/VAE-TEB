r"""Lint for the KL-weight arms: one axis, two keys, and a ratio that must not drift.

The reconstruction here is summed over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients against the
raw-signal comparison model's $H \cdot R = 480$ samples, while the KL is summed over the same
$d_z$ either way -- the two models move that width together, so it cancels out of the comparison
whatever it is. At a shared $\beta$ the rate term therefore applies roughly $4.9\times$ less pressure
than it did, and the **direction is the part that is easy to get backwards**: a larger
reconstruction at fixed $\beta$ makes $\beta\,\mathrm{KL}$ relatively *weaker*, so the latent opens
**wider**. The scale-matched value is $2340 / 480 = 4.875$, and the four arms bracket it at
$\{1.0,\, 2.5,\, 5.0,\, 10.0\}$ rather than sitting at or below the inherited $1.0$, which would
have put three of them on the same side of the point they exist to bracket.

Each arm moves **two** keys, not one, and that is the axis rather than a second delta.
$\beta_{\mathrm{prior}}$ anchors the conditional prior's scale through a restoring force that
*saturates* at $\beta_{\mathrm{prior}} / 2$ per latent dimension, while the reconstruction's
opposing pressure grows as the decoder sharpens -- so it behaves as a threshold rather than a dial,
and the thing that grew is exactly the pressure it is measured against. Holding
$\beta_{\mathrm{prior}} / \beta$ fixed keeps the anchor's standing constant across the sweep;
freezing $\beta_{\mathrm{prior}}$ while $\beta$ moved would attack that threshold from both sides at
once and leave a pinning arm with two candidate explanations.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box --
a key that does not resolve, a stray second delta, a value outside the declared set, a ratio typo'd
in one file out of four -- rather than days into a production run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.collapse import KL_COLLAPSE_PATIENCE_EPOCHS

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

_VAE = "model_config.VAE_model"
_BETA_END = f"{_VAE}.beta_schedule.end"
_BETA_PRIOR = f"{_VAE}.beta_prior"

#: The ratio $\beta_{\mathrm{prior}} / \beta$ held fixed across the file and every arm. Its value is
#: the raw-signal comparison model's own ($0.1 / 1.0$), which is what "the anchor keeps the same
#: standing against the KL it is measured beside" means concretely.
ANCHOR_RATIO = 0.1

#: Every arm: file name -> (converged KL weight, prior anchor weight). The anchor is written out
#: rather than computed from the ratio, so a file whose two keys disagree fails against a stated
#: number instead of against an expression that would derive the same mistake twice.
_ARMS: Dict[str, Any] = {
    "sweep_beta_1p0.yaml": (1.0, 0.1),
    "sweep_beta_2p5.yaml": (2.5, 0.25),
    "sweep_beta_5p0.yaml": (5.0, 0.5),
    "sweep_beta_10p0.yaml": (10.0, 1.0),
}

#: The declared value set, compared against the values read back from the resolved files so the
#: files -- not this module's table alone -- carry the burden of proof.
STATED_BETA_SET = {1.0, 2.5, 5.0, 10.0}

#: The scale-matched point the bracket is built around: the block cardinality ratio $2340 / 480$.
SCALE_MATCHED_BETA = 2340.0 / 480.0


def _flatten(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a config mapping to ``{dotted path: leaf value}``.

    Dicts recurse; everything else -- scalars, lists, ``None`` -- is a leaf, matching the loader's
    own merge semantics, where a list replaces wholesale and is therefore a value.

    Args:
        node: The mapping to walk.
        prefix: Dotted prefix accumulated so far.

    Returns:
        One entry per leaf.
    """
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _resolved(name: str) -> Dict[str, Any]:
    return load_config(str(_CONFIG_DIR / name))


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


# --------------------------------------------------------------------------------------
# The inventory is closed
# --------------------------------------------------------------------------------------
def test_the_config_directory_holds_exactly_the_declared_arms():
    """Both directions: a declared arm whose file is missing, and a stray ``sweep_*.yaml`` nobody
    declared -- which would be launchable, and would run outside every assertion below."""
    present = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")}

    assert present == set(_ARMS)


def test_the_arm_values_are_exactly_the_stated_set():
    """Read back from the resolved files rather than from this module's own table."""
    observed = {_flatten(_resolved(name))[_BETA_END] for name in _ARMS}

    assert observed == STATED_BETA_SET


def test_the_bracket_straddles_the_scale_matched_value():
    """The reason for *these* four numbers rather than the inherited bracket. Two arms below the
    scale-matched point and two at or above it, so the sweep can find it rather than only bounding
    it from one side -- which is what three arms at or under the comparison model's $1.0$ would
    have done."""
    below = {beta for beta in STATED_BETA_SET if beta < SCALE_MATCHED_BETA}
    at_or_above = {beta for beta in STATED_BETA_SET if beta >= SCALE_MATCHED_BETA}

    assert len(below) == 2 and len(at_or_above) == 2
    assert min(at_or_above) == pytest.approx(5.0)  # the shipped weight is the one just above it


# --------------------------------------------------------------------------------------
# Every arm is the default plus exactly its declared delta
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_resolves_with_its_base_consumed(name):
    """``load_config`` must both succeed and eat the ``base:`` directive; a leftover ``base`` key
    would reach the validator as an unknown key and the MLflow param dump as noise."""
    assert "base" not in _resolved(name)


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_differs_from_the_default_in_exactly_its_two_swept_keys(name, default_flat):
    """The one-axis property. Key sets must match exactly -- a typo'd override *adds* a path rather
    than moving one, and a config key that reaches nothing raises nothing -- and the differing
    values must be exactly the two swept keys. An arm restating the shipped weights legitimately
    differs in nothing, which is why the expectation is filtered against the default rather than
    asserted to be non-empty."""
    beta, beta_prior = _ARMS[name]
    arm_flat = _flatten(_resolved(name))
    intended = {_BETA_END: beta, _BETA_PRIOR: beta_prior}

    assert set(arm_flat) == set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    expected = {path for path, value in intended.items() if default_flat[path] != value}
    assert differing == expected

    for path, value in intended.items():
        assert arm_flat[path] == value


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_holds_the_anchor_ratio_the_design_fixes(name):
    """The second key is not free. The anchor's restoring force saturates at
    $\\beta_{\\mathrm{prior}} / 2$ per dimension while the reconstruction it opposes is what this
    target domain multiplied, so an arm that moved $\\beta$ alone would be sweeping the anchor's
    standing at the same time and a pinning prior would have two explanations."""
    arm_flat = _flatten(_resolved(name))

    assert arm_flat[_BETA_PRIOR] / arm_flat[_BETA_END] == pytest.approx(ANCHOR_RATIO)


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_ramps_beta_from_exactly_zero(name):
    """Only the endpoint is swept. $z$ is the only route to the decoder here, so a nonzero weight
    before the decoder can use the latent at all is the standard route to posterior collapse -- and
    an arm that lost the warm-up would collapse for a reason that has nothing to do with its
    endpoint."""
    schedule = _resolved(name)["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_runs_long_enough_to_be_judged_by_the_collapse_criterion(name):
    """The criterion reads the tail: a run is collapsed when its source-conditioned KL is below the
    threshold at every one of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs. An arm shorter
    than that window cannot be judged at all, and the patience is imported rather than restated so a
    revision there reaches this bound instead of leaving a stale copy of it here."""
    config = _resolved(name)

    assert config["general_config"]["epochs"] > KL_COLLAPSE_PATIENCE_EPOCHS
    # And by a margin that leaves a tail to read, rather than only clearing the boundary: the beta
    # ramp alone occupies the first `warmup_epochs`, during which the KL is not yet paying its rate.
    assert (
        config["general_config"]["epochs"]
        > config["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
        + KL_COLLAPSE_PATIENCE_EPOCHS
    )
