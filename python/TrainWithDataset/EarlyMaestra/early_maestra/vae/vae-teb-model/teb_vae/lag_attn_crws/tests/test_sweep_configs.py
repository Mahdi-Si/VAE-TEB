r"""Every sweep arm passes this driver's pre-flight, moves the stride with the horizon, and runs long
enough to be judged.

The arms bracket the anchor tiling and the horizon. These tests are a lint, not a fit: they exist so
a malformed arm is caught on the development box -- a floor that no longer pairs with the budget, a
stride that leaves a phase with no anchor, a boundary term switched on, a horizon change that left
the stride behind -- rather than days into a production run.

The pre-flight is run against the arm's resolved config with the tiny variant's dataset block
swapped in: the shipped shard paths are placeholders, and the pre-flight's budget resolver reads the
shards. The tiny variant points at the committed causal fixture at the real geometry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws.collapse import KL_COLLAPSE_PATIENCE_EPOCHS

from .conftest import absolutize_dataset_paths

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

#: Every shipped arm, found on disk rather than listed, so a new arm is linted without an edit here.
_ARMS = sorted(path.name for path in _CONFIG_DIR.glob("sweep_*.yaml"))


def _resolved(name: str) -> Dict[str, Any]:
    return load_config(str(_CONFIG_DIR / name))


def _vae(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["model_config"]["VAE_model"]


def test_there_are_arms_to_check():
    """A silently-empty glob would make every test below vacuous."""
    assert _ARMS, f"no sweep_*.yaml under {_CONFIG_DIR}"


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_passes_the_preflight(name):
    """The driver's own refusals, over the arm's resolved geometry: the floor pairs with the resolved
    budget, every tile phase keeps at least one anchor, the boundary term is off, and the loader
    fields the tile phase and the validity mask are read from are present."""
    config = _resolved(name)
    config["dataset_config"] = _resolved("tiny.yaml")["dataset_config"]

    LagAttnCrwsTrainer.preflight(absolutize_dataset_paths(config))


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_that_moves_the_horizon_moves_the_stride_with_it(name):
    """The two are one decision everywhere except in an arm whose content is decoupling them: a
    horizon change that left the stride behind would either overlap the windows again or leave
    target steps no phase ever covers, and neither would change a shape."""
    default, arm = _vae(_resolved("default.yaml")), _vae(_resolved(name))

    if arm["horizon"] != default["horizon"]:
        assert arm["anchor_stride"] == arm["horizon"]


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_runs_long_enough_to_be_judged_by_the_collapse_criterion(name):
    """The criterion reads the tail: a run is collapsed when its source-conditioned KL is below the
    threshold at every one of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs, after the
    $\\beta$ warm-up. An arm shorter than that window cannot be judged at all."""
    config = _resolved(name)

    assert (
        config["general_config"]["epochs"]
        > _vae(config)["beta_schedule"]["warmup_epochs"] + KL_COLLAPSE_PATIENCE_EPOCHS
    )
