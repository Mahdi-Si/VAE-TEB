r"""Lint for the two arms: one axis each, a closed inventory, and a stride that follows the horizon.

The two arms bracket the one decision this cell makes that the model it is compared against does not
-- the anchor tiling -- plus the decision it makes *differently*, the horizon. Each is the shipped
configuration with the smallest key set that expresses the change, and one of them moves **two** keys
rather than one: ``sweep_horizon_30.yaml`` moves the stride with the horizon, because below the
horizon the forecast windows overlap again and above it there are target steps no phase ever covers.
That is the axis rather than a second delta, and the test below states it as such.

**There is deliberately no floor arm**, and the absence is the record rather than an omission. The
anchor floor here is a declared input-warmth policy rather than a validity requirement -- the raw
target is honest at every step -- so the interesting move would be *downward*, to the model's own
$30$-step warm-up, which buys $255$ anchors against $152$. Both the constructor and the pre-flight
refuse it, from one function, and lifting that refusal changes what a run **claims** rather than what
the data supports. It belongs in the design record, not in a launchable file.

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
    # decodes -- so this arm is the control that says what the tiling itself costs or buys.
    "sweep_anchor_stride_1.yaml": {f"{_VAE}.anchor_stride": 1},
    # The horizon arm, and the one that moves two keys: the stride is not free.
    "sweep_horizon_30.yaml": {f"{_VAE}.horizon": 30, f"{_VAE}.anchor_stride": 30},
}

#: Every non-arm file the directory may hold, with what each is for. Named so a file arriving
#: undeclared fails the lint below rather than becoming launchable and unchecked.
_NON_ARM_CONFIGS = {
    "default.yaml",      # the production configuration
    "tiny.yaml",         # the CPU smoke variant, at the real geometry
    "smoke_causal.yaml",  # the instrumented run the two pending constants are measured from
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

    assert present == set(_ARMS) | _NON_ARM_CONFIGS


def test_there_is_no_floor_arm():
    """Stated as an assertion rather than left as an absence. A floor below the pairing is refused by
    the constructor and again by the pre-flight, from one function, so a file that moved it would be
    a launchable config that cannot launch -- and the move worth making is downward, which changes
    what the run claims rather than what the data supports."""
    for name in _ARMS:
        assert f"{_VAE}.warmup_period" not in _ARMS[name], name

    # Over the raw files rather than the resolved configs, and over the settings rather than the
    # prose: ``default.yaml`` is the one file allowed to declare the floor, and a variant that
    # re-stated it -- at any value -- would be a second declaration of a policy stated in one place.
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        if path.name == _DEFAULT.name:
            continue
        settings = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any("warmup_period" in line for line in settings), path.name


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
    """The feasibility the resolver refuses on, checked per arm because both of them move exactly the
    quantities it is stated in. At the last phase the first anchor is $F + S - 1$; if it does not
    exist, a sample drawn at that phase contributes no forecast at all and its share of the epoch is
    silently dropped."""
    floor, stride, _horizon, t_valid = _geometry(_resolved(name))

    assert floor + stride <= t_valid
    assert 1 <= stride


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_pairs_its_floor_with_the_shipped_budget(name):
    r"""$F \ge B - 1$ over the survivors, which at the shipped budget is $F \ge 133$. Both arms keep
    the budget and the floor, so this is a check that neither moved one by accident: the floor is the
    declared input-warmth policy, and an arm that lowered it would decode anchors whose inputs are
    still partly assumed pre-recording history with every number finite."""
    floor, _stride, _horizon, _t_valid = _geometry(_resolved(name))
    vae = _resolved(name)["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert floor >= vae["causal_warmup_budget_steps"] - 1


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_keeps_the_boundary_shape_term_off(name):
    """It is a slicing identity over adjacent anchors, and both arms decode a set whose entries are
    at least one stride apart -- including the dense one, where the identity is legal but the driver
    refuses it anyway rather than making the refusal conditional on a value."""
    assert _resolved(name)["model_config"]["VAE_model"]["lambda_boundary"] == 0.0


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_ramps_beta_from_exactly_zero(name):
    """No arm sweeps beta, and every arm needs the ramp: $z$ is the only route to the decoder, so an
    arm that lost the warm-up would collapse for a reason that has nothing to do with its own axis."""
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


# --------------------------------------------------------------------------------------
# What each arm buys, as a number rather than as an argument
# --------------------------------------------------------------------------------------
def test_the_horizon_arm_restores_the_comparison_models_block_exactly():
    r"""The property that makes this arm worth more here than in any other causal cell. The block is
    $H \cdot R$ and $R$ is a property of the raw grid rather than of a channel budget, so at
    $H = 30$ it is $480$ raw samples -- **exactly** the comparison model's. This is the one arm in
    the family under which a nat crosses the two-sided/one-sided input boundary unchanged."""
    floor, stride, horizon, t_valid = _geometry(_resolved("sweep_horizon_30.yaml"))
    raw_per_step = _resolved("sweep_horizon_30.yaml")["model_config"]["VAE_model"]["raw_per_step"]
    sibling = load_config(
        str(Path(__file__).resolve().parents[3] / "teb_vae" / "lag_attn_rws" / "configs"
            / "default.yaml")
    )["model_config"]["VAE_model"]

    assert horizon == sibling["horizon"] == 30
    assert horizon * raw_per_step == sibling["horizon"] * sibling["raw_per_step"] == 480
    assert t_valid == 270  # against the shipped 285: the anchors the longer horizon costs
    assert -(-(t_valid - floor) // stride) == 5  # A_max, from the geometry rather than a literal


def test_the_stride_arm_restores_the_dense_anchor_set():
    r"""1 is the value every other cell of the grid runs at, which is what makes this the control: at
    stride 1 the decoded set is the dense range $[F, T_{\mathrm{valid}})$ and each raw sample is
    scored by up to $H$ anchors again."""
    floor, stride, _horizon, t_valid = _geometry(_resolved("sweep_anchor_stride_1.yaml"))

    assert stride == 1
    assert -(-(t_valid - floor) // stride) == t_valid - floor == 152
