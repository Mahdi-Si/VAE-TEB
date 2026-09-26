r"""Lint for the KL-weight arms: one axis, two keys, and a ratio that must not drift.

The reconstruction here is summed over $H \cdot C_{\mathrm{keep}}$ coefficients against the
raw-signal comparison model's $H \cdot R$ samples, so at a shared $\beta$ the rate term applies
less pressure than it did, and the arms bracket the scale-matched weight.

Each arm moves **two** keys, not one, and that is the axis rather than a second delta.
$\beta_{\mathrm{prior}}$ anchors the conditional prior's scale through a restoring force that
*saturates* at $\beta_{\mathrm{prior}} / 2$ per latent dimension, while the reconstruction's
opposing pressure grows as the decoder sharpens. Holding $\beta_{\mathrm{prior}} / \beta$ at the
default's ratio keeps the anchor's standing constant across the sweep; freezing
$\beta_{\mathrm{prior}}$ while $\beta$ moved would leave a pinning arm with two candidate
explanations.

These tests are a lint, not a fit: every ``sweep_*.yaml`` on disk is checked, so a stray arm cannot
run outside them, and nothing about an arm is transcribed -- its $\beta$ is read off its file name
and its anchor off the default's ratio.
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

#: Every arm on disk, so a new file is linted the moment it exists.
_ARMS = sorted(path.name for path in _CONFIG_DIR.glob("sweep_beta_*.yaml"))


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


def _beta_from_name(name: str) -> float:
    """``sweep_beta_2p5.yaml`` -> ``2.5``."""
    return float(name[len("sweep_beta_") : -len(".yaml")].replace("p", "."))


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


def test_there_are_arms_to_check():
    """A silently-empty glob would make the parametrised lint below vacuous."""
    assert _ARMS


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_differs_from_the_default_in_exactly_its_two_swept_keys(name, default_flat):
    """The one-axis property. Key sets must match exactly -- a typo'd override *adds* a path rather
    than moving one, and a leftover ``base`` key would add one too -- and nothing but the two swept
    keys may differ. The arm's $\\beta$ is the one its file name states, and its anchor holds the
    default's $\\beta_{\\mathrm{prior}} / \\beta$."""
    arm_flat = _flatten(load_config(str(_CONFIG_DIR / name)))

    assert set(arm_flat) == set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    assert differing <= {_BETA_END, _BETA_PRIOR}
    assert arm_flat[_BETA_END] == pytest.approx(_beta_from_name(name))
    assert arm_flat[_BETA_PRIOR] / arm_flat[_BETA_END] == pytest.approx(
        default_flat[_BETA_PRIOR] / default_flat[_BETA_END]
    )


def test_a_run_is_long_enough_to_be_judged_by_the_collapse_criterion():
    """The criterion reads the tail: a run is collapsed when its source-conditioned KL is below the
    threshold at every one of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs, and the beta
    ramp occupies the first ``warmup_epochs``. Checked on the default, which every arm matches
    outside the two swept keys."""
    config = load_config(str(_DEFAULT))

    assert (
        config["general_config"]["epochs"]
        > config["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
        + KL_COLLAPSE_PATIENCE_EPOCHS
    )
