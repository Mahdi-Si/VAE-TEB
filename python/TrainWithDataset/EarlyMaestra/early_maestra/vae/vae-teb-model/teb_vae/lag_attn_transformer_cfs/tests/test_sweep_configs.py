r"""Lint for the arms: each is a well-formed variant of the default, and each can be trained.

The **tiling** arm is this package's own long-standing one, and the arms it does *not* carry are a
decision rather than an oversight. The floor, horizon and depth arms answer questions about the
target domain, which ``teb_vae/lag_attn_cfs/configs`` already asks; the encoder arms answer
questions about the encoder, which ``teb_vae/lag_attn_transformer_rws/configs`` already asks. The
tiling arm is the only one of those whose answer could differ between the two encoders, because
what it moves is the per-step gradient noise, and a pre-normalised attention stack is exactly the
architecture whose stability is sensitive to that.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box --
a key that does not resolve, an arm that moves nothing, a name and a tag that disagree, a tiling or
a floor the constructor would refuse, a warm-up the arm lost -- rather than days into a production
run. Every ``sweep_*.yaml`` present is checked, discovered rather than listed; that every arm also
validates with no dead key is checked by ``test_config_load.py`` over the whole directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.collapse import KL_COLLAPSE_PATIENCE_EPOCHS

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_DEFAULT = _CONFIG_DIR / "default.yaml"

#: Where a run says which arm it is. Both paths, because either alone is recoverable from a
#: half-configured run: the name is what an operator reads and the tag is what an arm table groups
#: on, and a run whose two disagree is the drift this guard exists to make visible.
_RUN_NAME = "advanced_config.tracking.mlflow.run_name"
_VARIANT = "advanced_config.tracking.mlflow.tags.variant"

_ARMS = sorted(path.name for path in _CONFIG_DIR.glob("sweep_*.yaml"))


def _flatten(node: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a config mapping to ``{dotted path: leaf value}``."""
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def test_there_are_arms_to_lint():
    """A silently-empty glob would make every parametrised case below vanish rather than fail."""
    assert _ARMS


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_is_a_well_formed_variant_of_the_default(name):
    """Key sets must match the default's exactly -- a typo'd override *adds* a path rather than
    moving one, and a config key that reaches nothing raises nothing -- and the arm must move at
    least one leaf besides its identity, or it is the default under another name.

    An arm that declares an identity declares it on both keys, and the two agree: a config is a file
    on one machine and a run is an artifact directory on another, and a run whose artifacts cannot
    name its arm gets attributed to the default by whoever reads it later.
    """
    default = _flatten(load_config(str(_DEFAULT)))
    arm = _flatten(load_config(str(_CONFIG_DIR / name)))

    assert set(arm) == set(default)
    moved = {path for path in default if arm[path] != default[path]} - {_RUN_NAME, _VARIANT}
    assert moved, f"{name} moves nothing but its identity"
    if arm[_RUN_NAME] != default[_RUN_NAME] or arm[_VARIANT] != default[_VARIANT]:
        assert arm[_RUN_NAME] == arm[_VARIANT], "the name and the tag name different arms"
        assert arm[_RUN_NAME] != default[_RUN_NAME]


@pytest.mark.parametrize("name", _ARMS)
def test_an_arm_can_be_trained_and_judged(name):
    r"""The feasibility and schedule rules every arm needs whatever its own axis is.

    * At the last phase the first anchor is $F + S - 1$; if it does not exist, a sample drawn at
      that phase contributes no forecast at all and its share of the epoch is silently dropped.
    * $F \ge B - 1$ over the survivors: an arm that lowered the floor would score assumed
      pre-recording history as signal with every shape correct.
    * $\beta$ ramps from exactly zero: $z$ is the only route to the decoder, so an arm that lost the
      warm-up would collapse for a reason that has nothing to do with its own axis.
    * The step-granular learning-rate ramp stays on; zeroed, the arm would fall back to the
      framework's epoch-granularity path and change the optimisation alongside its own axis.
    * The run outlasts the $\beta$ warm-up by the collapse criterion's window, which reads the tail:
      a run shorter than that cannot be judged at all.
    """
    config = load_config(str(_CONFIG_DIR / name))
    vae = config["model_config"]["VAE_model"]
    floor, stride = int(vae["warmup_period"]), int(vae["anchor_stride"])
    t_valid = int(vae["sequence_length"]) - int(vae["horizon"])
    schedule = vae["beta_schedule"]
    epochs = config["general_config"]["epochs"]

    assert 1 <= stride
    assert floor + stride <= t_valid
    assert floor >= vae["causal_warmup_budget_steps"] - 1
    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert config["general_config"]["lr_warmup_steps"] > 0
    assert epochs > schedule["warmup_epochs"] + KL_COLLAPSE_PATIENCE_EPOCHS
