r"""Lint for the sweep arms: one axis each, a closed inventory, and parity with the conv-LSTM twin.

This package ships a single arm, the tiling ablation: ``anchor_stride: 1`` is the inert stride --
the dense range every other cell of the grid decodes -- so the arm is the control that says what the
tiling itself costs or buys. There is no horizon arm (that comparison exists one package over, on
this same encoder), no floor arm (the floor is a declared input-warmth policy the constructor and the
pre-flight both refuse to lower) and no encoder arm (the encoder is the conv-Transformer raw-signal
package's own object and was swept there).

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box --
a key that does not resolve, a stray second delta, a file nobody declared -- rather than days into a
production run. Every arm also passes the validator in ``tests/test_config_load.py``, which globs the
whole directory.
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

#: Every arm: file name -> the exact leaf delta it declares. Written out rather than derived, so a
#: file whose keys disagree fails against a stated intention instead of against an expression that
#: would derive the same mistake twice.
_ARMS: Dict[str, Dict[str, Any]] = {
    "sweep_anchor_stride_1.yaml": {f"{_VAE}.anchor_stride": 1},
}


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


def test_the_config_directory_holds_exactly_the_declared_arms() -> None:
    """Both directions: a declared arm whose file is missing, and a stray ``sweep_*.yaml`` nobody
    declared -- which would be launchable, and would run outside the one-axis check below."""
    present = {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")}

    assert present == set(_ARMS)


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_differs_from_the_default_in_exactly_its_declared_keys(name) -> None:
    """The one-axis property. Key sets must match exactly -- a typo'd override *adds* a path rather
    than moving one, and a config key that reaches nothing raises nothing."""
    default_flat = _flatten(load_config(str(_DEFAULT)))
    intended = _ARMS[name]
    arm_flat = _flatten(_resolved(name))

    assert set(arm_flat) == set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    assert differing == set(intended)

    for path, value in intended.items():
        assert arm_flat[path] == value


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_runs_long_enough_to_be_judged_by_the_collapse_criterion(name) -> None:
    """The criterion reads the tail: a run is collapsed when its source-conditioned KL is below the
    threshold at every one of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs. An arm shorter
    than the beta ramp plus that window cannot be judged at all."""
    config = _resolved(name)

    assert (
        config["general_config"]["epochs"]
        > config["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
        + KL_COLLAPSE_PATIENCE_EPOCHS
    )


def test_the_stride_arm_matches_the_conv_lstm_cells_own_arm() -> None:
    """The two cells of this row ship the same ablation, and it has to be the same ablation: a
    difference in what the arm moves would make the encoder comparison unreadable on that arm."""
    theirs = load_config(
        str(
            Path(__file__).resolve().parents[3]
            / "teb_vae"
            / "lag_attn_crws"
            / "configs"
            / "sweep_anchor_stride_1.yaml"
        )
    )
    mine = _resolved("sweep_anchor_stride_1.yaml")

    assert (
        mine["model_config"]["VAE_model"]["anchor_stride"]
        == theirs["model_config"]["VAE_model"]["anchor_stride"]
    )
