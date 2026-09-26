r"""Lint for the sweep arms: each is the default plus only the keys it sets, and each can run.

Every ``sweep_*.yaml`` in the config directory is linted, found by glob rather than listed, so an arm
added later is checked without anyone remembering to register it. The delta each arm declares is
read off its own file (everything but ``base:``) rather than restated here, so the lint compares the
resolved config against what the file says and not against a second copy of it.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box --
a key that does not resolve, an arm that changes nothing, a run name and variant tag that disagree,
a stride or floor the geometry cannot tile, a run too short for the collapse criterion -- rather
than days into a production run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.collapse import KL_COLLAPSE_PATIENCE_EPOCHS

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

#: Where a run says which arm it is. The name is what an operator reads and the tag is what an arm
#: table groups on, and a run whose two disagree is the drift the identity check makes visible.
_RUN_NAME = "advanced_config.tracking.mlflow.run_name"
_VARIANT = "advanced_config.tracking.mlflow.tags.variant"

#: Every arm on disk.
_ARMS = sorted(path.name for path in _CONFIG_DIR.glob("sweep_*.yaml"))


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


def _declared(name: str) -> Dict[str, Any]:
    """The leaves an arm's own file sets, without its ``base:`` directive."""
    raw = yaml.safe_load((_CONFIG_DIR / name).read_text(encoding="utf-8"))
    raw.pop("base", None)
    return _flatten(raw)


@pytest.fixture(scope="module")
def default_flat() -> Dict[str, Any]:
    return _flatten(load_config(str(_DEFAULT)))


def test_there_are_arms_to_lint():
    """A silently-empty glob would make every parametrised test below vacuous."""
    assert _ARMS


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_is_the_default_plus_only_the_keys_it_sets(name, default_flat):
    """The one-axis property, computed. Key sets must match exactly -- a typo'd override *adds* a
    path rather than moving one, and a config key that reaches nothing raises nothing -- the arm
    must actually move something, and everything it moves is a key its own file sets.

    An arm that names itself does so on both identity keys, and the two agree and differ from the
    default's: a run whose artifacts cannot name its arm gets attributed to the default by whoever
    reads it later.
    """
    declared = _declared(name)
    arm_flat = _flatten(load_config(str(_CONFIG_DIR / name)))

    assert set(arm_flat) == set(default_flat), set(arm_flat) ^ set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    assert differing, f"{name} changes nothing"
    assert differing <= set(declared)

    if _RUN_NAME in declared or _VARIANT in declared:
        assert arm_flat[_RUN_NAME] == arm_flat[_VARIANT], "the name and the tag name different arms"
        assert arm_flat[_RUN_NAME] != default_flat[_RUN_NAME]


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_can_tile_its_geometry_and_be_judged(name):
    r"""The feasibility the constructor and the resolver refuse on, checked per arm because several
    move exactly the quantities it is stated in.

    The stride must lie in $[1, H]$. At the last phase the first anchor is $F + S - 1$; if it does
    not exist, a sample drawn at that phase contributes no forecast at all. The floor must pair with
    the budget, $F \ge B - 1$, or the objective scores assumed pre-recording history as signal. And
    the run must outlast the $\beta$ warm-up by the collapse criterion's window
    (:data:`KL_COLLAPSE_PATIENCE_EPOCHS`), or it cannot be judged collapsed or not at all.
    """
    config = load_config(str(_CONFIG_DIR / name))
    vae = config["model_config"]["VAE_model"]
    floor, stride, horizon = (
        int(vae["warmup_period"]), int(vae["anchor_stride"]), int(vae["horizon"])
    )
    t_valid = int(vae["sequence_length"]) - horizon

    assert 1 <= stride <= horizon
    assert floor + stride <= t_valid
    assert floor >= int(vae["causal_warmup_budget_steps"]) - 1
    assert config["general_config"]["epochs"] > (
        vae["beta_schedule"]["warmup_epochs"] + KL_COLLAPSE_PATIENCE_EPOCHS
    )
