r"""Lint for the four arms: one axis each, a closed inventory, and a stride that follows the horizon.

The four arms bracket the two decisions this cell makes that no other cell of the grid does -- the
anchor tiling and the anchor floor -- plus one that every cell makes and this one makes differently
(the horizon), plus one parity choice inherited rather than measured (the decoder depth).

Each is the shipped configuration with the smallest key set that expresses the change, and one of
them moves **two** keys rather than one: ``sweep_horizon_30.yaml`` moves the stride with the
horizon, because below the horizon the forecast windows overlap again and above it there are target
steps no phase ever covers. That is the axis rather than a second delta, and the test below states
it as such.

These tests are a lint, not a fit. They exist so a malformed arm is caught on the development box --
a key that does not resolve, a stray second delta, a stride left behind by a horizon change, a file
nobody declared -- rather than days into a production run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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
    # The tiling ablation. 1 is the INERT stride -- the dense range every other cell of the grid
    # decodes -- so this arm is the control that says what the tiling costs or buys.
    "sweep_anchor_stride_1.yaml": {f"{_VAE}.anchor_stride": 1},
    # The horizon arm, and the one that moves two keys: the stride is not free.
    "sweep_horizon_30.yaml": {f"{_VAE}.horizon": 30, f"{_VAE}.anchor_stride": 30},
    # The ten-minute policy floor. The budget stays at 134, because the pairing requires
    # F >= B - 1 rather than equality.
    "sweep_floor_150.yaml": {f"{_VAE}.warmup_period": 150},
    # Parameter economy against parity: the receptive-field argument is moot in both directions,
    # because the horizon attention blocks mix every token unmasked after the refine stack.
    "sweep_horizon_depth_3.yaml": {f"{_VAE}.horizon_depth": 3},
}

#: The arms whose stride is expected to equal their horizon. ``sweep_anchor_stride_1.yaml`` is the
#: exception BY CONSTRUCTION -- decoupling the two is the whole content of that arm -- which is why
#: the invariant is stated as a set rather than as a rule over every file.
_STRIDE_FOLLOWS_HORIZON = tuple(
    name for name in _ARMS if name != "sweep_anchor_stride_1.yaml"
)


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


def _geometry(config: Dict[str, Any]) -> Tuple[int, int, int, int]:
    r"""$(F, S, H, T_{\mathrm{valid}})$ read off a resolved config."""
    vae = config["model_config"]["VAE_model"]
    return (
        int(vae["warmup_period"]),
        int(vae["anchor_stride"]),
        int(vae["horizon"]),
        int(vae["sequence_length"]) - int(vae["horizon"]),
    )


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


def test_the_config_directory_holds_exactly_the_declared_files():
    """The directory lint, over every YAML rather than only the arms: a file that arrived undeclared
    is one nothing here checks, and configs are launchable by path."""
    present = {path.name for path in _CONFIG_DIR.glob("*.yaml")}

    assert present == set(_ARMS) | {"default.yaml", "tiny.yaml", "smoke_hie.yaml"}


# --------------------------------------------------------------------------------------
# Every arm is the default plus exactly its declared delta
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_resolves_with_its_base_consumed(name):
    """``load_config`` must both succeed and eat the ``base:`` directive; a leftover ``base`` key
    would reach the validator as an unknown key and the MLflow param dump as noise."""
    assert "base" not in _resolved(name)


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_differs_from_the_default_in_exactly_its_declared_keys(name, default_flat):
    """The one-axis property. Key sets must match exactly -- a typo'd override *adds* a path rather
    than moving one, and a config key that reaches nothing raises nothing."""
    intended = _ARMS[name]
    arm_flat = _flatten(_resolved(name))

    assert set(arm_flat) == set(default_flat)
    differing = {path for path in default_flat if arm_flat[path] != default_flat[path]}
    assert differing == set(intended)

    for path, value in intended.items():
        assert arm_flat[path] == value


@pytest.mark.parametrize("name", _STRIDE_FOLLOWS_HORIZON)
def test_an_arm_moves_the_stride_with_the_horizon(name):
    """The two are one decision everywhere except in the arm whose content is decoupling them. A
    horizon change that left the stride behind would either overlap the windows again or leave
    target steps no phase ever covers, and neither would change a shape."""
    _floor, stride, horizon, _t_valid = _geometry(_resolved(name))

    assert stride == horizon


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_leaves_at_least_one_tile_at_every_phase(name):
    """The feasibility the resolver refuses on, checked per arm because two of them move exactly the
    quantities it is stated in. At the last phase the first anchor is $F + S - 1$; if it does not
    exist, a sample drawn at that phase contributes no forecast at all and its share of the epoch is
    silently dropped."""
    floor, stride, _horizon, t_valid = _geometry(_resolved(name))

    assert floor + stride <= t_valid
    assert 1 <= stride


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_pairs_its_floor_with_the_shipped_budget(name):
    r"""$F \ge B - 1$ over the survivors, which at the shipped budget is $F \ge 133$. Every arm keeps
    the budget, so an arm that lowered the floor would score assumed pre-recording history as signal
    with every shape correct."""
    floor, _stride, _horizon, _t_valid = _geometry(_resolved(name))
    vae = _resolved(name)["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert floor >= vae["causal_warmup_budget_steps"] - 1


def test_the_horizon_arm_resolves_to_the_two_sided_block_width_and_its_own_tile_count():
    """The arm is a horizon change, not a silent half-change. What it restores is the *question* --
    how far ahead the model is asked to forecast -- and what it does not restore is the block, which
    is $30 \\times 98$ against the two-sided sibling's $30 \\times 78$ because $C_\\mathrm{keep}$ is
    what the warm-up budget decides."""
    floor, stride, horizon, t_valid = _geometry(_resolved("sweep_horizon_30.yaml"))
    kept = 98

    assert horizon == 30
    assert horizon * kept == 2940
    assert t_valid == 270  # against the shipped 285: the anchors the longer horizon costs
    assert -(-(t_valid - floor) // stride) == 5  # A_max, from the geometry rather than a literal


def test_the_stride_arm_restores_the_dense_anchor_set():
    """1 is the value every other cell of the grid runs at, which is what makes this the control:
    at stride 1 the decoded set is the dense range $[F, T_\\mathrm{valid})$ and each target
    coefficient is scored by up to $H$ anchors again."""
    floor, stride, _horizon, t_valid = _geometry(_resolved("sweep_anchor_stride_1.yaml"))

    assert stride == 1
    assert -(-(t_valid - floor) // stride) == t_valid - floor == 152


def test_the_floor_arm_keeps_the_identical_channels_and_costs_exactly_two_tiles():
    """The cost of the policy, stated as a number rather than as an argument: the same $98$ channels
    (the budget did not move), two fewer tiles at phase $0$, and $17$ fewer covered target steps."""
    shipped_floor, stride, _horizon, t_valid = _geometry(load_config(str(_DEFAULT)))
    arm_floor, arm_stride, _h, arm_t_valid = _geometry(_resolved("sweep_floor_150.yaml"))

    assert (arm_stride, arm_t_valid) == (stride, t_valid)
    assert arm_floor - shipped_floor == 17
    assert -(-(t_valid - shipped_floor) // stride) - -(-(t_valid - arm_floor) // stride) == 2


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_ramps_beta_from_exactly_zero(name):
    """No arm sweeps beta, and every arm needs the ramp: $z$ is the only route to the decoder, so an
    arm that lost the warm-up would collapse for a reason that has nothing to do with its own
    axis."""
    schedule = _resolved(name)["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_runs_long_enough_to_be_judged_by_the_collapse_criterion(name):
    """The criterion reads the tail: a run is collapsed when its source-conditioned KL is below the
    threshold at every one of its final :data:`KL_COLLAPSE_PATIENCE_EPOCHS` epochs. An arm shorter
    than that window cannot be judged at all."""
    config = _resolved(name)

    assert config["general_config"]["epochs"] > KL_COLLAPSE_PATIENCE_EPOCHS
    assert (
        config["general_config"]["epochs"]
        > config["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
        + KL_COLLAPSE_PATIENCE_EPOCHS
    )


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_keeps_the_boundary_shape_term_off(name):
    """It is a slicing identity over adjacent anchors, and every arm here decodes a set whose
    entries are at least one stride apart -- including the dense one, where the identity is legal
    but the driver refuses it anyway rather than making the refusal conditional on a value."""
    assert _resolved(name)["model_config"]["VAE_model"]["lambda_boundary"] == 0.0
