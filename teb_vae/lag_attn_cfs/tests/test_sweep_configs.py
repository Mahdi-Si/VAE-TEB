r"""Lint for the four arms: one axis each, a closed inventory, and a stride that follows the horizon.

The four arms bracket the two decisions this cell makes that no other cell of the grid does -- the
anchor tiling and the anchor floor -- plus one that every cell makes and this one makes differently
(the horizon), plus one parity choice inherited rather than measured (the decoder depth).

Each is the shipped configuration with the smallest key set that expresses the change, and one of
them moves **two** keys rather than one: ``sweep_horizon_15.yaml`` moves the stride with the
horizon, because below the horizon the forecast windows overlap again and above it there are target
steps no phase ever covers. That is the axis rather than a second delta, and the test below states
it as such.

The horizon arm points **up**: the shipped horizon is $10$ (since 2026-09-05, down from $30$), and
the arm restores the one minute this cell originally shipped at, so the two runs bracket the
horizon question from both sides. It replaced a ``sweep_horizon_30.yaml`` that restored the
two-minute horizon back when the default was one minute. Its stride follows the default's
$S = H / 2$ rule rather than partitioning, so the arm moves the horizon and nothing about the overlap.

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

#: Where a run says which arm it is. Both paths, because either alone is recoverable from a
#: half-configured run: the name is what an operator reads and the tag is what an arm table groups
#: on, and a run whose two disagree is the drift this guard exists to make visible.
_RUN_NAME = "advanced_config.tracking.mlflow.run_name"
_VARIANT = "advanced_config.tracking.mlflow.tags.variant"

#: Every arm: file name -> the exact leaf delta it declares. Written out rather than derived, so a
#: file whose keys disagree fails against a stated intention instead of against an expression that
#: would derive the same mistake twice.
#:
#: The four arms declared last carry **three** paths each rather than one: their own axis, and the
#: two identity keys. The identity pair is not a second delta -- it is the same delta, written
#: where a finished run can still be asked which side of the axis it trained on. The four arms
#: above them predate that convention and inherit the default's name; they are left as they are
#: rather than renamed under a task about the inventory.
_ARMS: Dict[str, Dict[str, Any]] = {
    # The tiling ablation. 1 is the INERT stride -- the dense range every other cell of the grid
    # decodes -- so this arm is the control that says what the tiling costs or buys.
    "sweep_anchor_stride_1.yaml": {f"{_VAE}.anchor_stride": 1},
    # The horizon arm, and the one that moves two keys: the stride is not free. It lengthens the
    # shipped horizon to one minute, following the default's S = H / 2 tiling rule up with it,
    # which costs anchors and buys forecast reach.
    "sweep_horizon_15.yaml": {f"{_VAE}.horizon": 15, f"{_VAE}.anchor_stride": 7},
    # The ten-minute policy floor. The budget stays at 134, because the pairing requires
    # F >= B - 1 rather than equality.
    "sweep_floor_150.yaml": {f"{_VAE}.warmup_period": 150},
    # Parameter economy against parity: the receptive-field argument is moot in both directions,
    # because the horizon attention blocks mix every token unmasked after the refine stack.
    "sweep_horizon_depth_3.yaml": {f"{_VAE}.horizon_depth": 3},
    # The lag-bias seed. The default seeds the learnable (num_heads, L) bias FLAT; this restores
    # the decaying seed, which predicts a lag-0 peak on its own. `lag_bias_init` deliberately does
    # not move with it -- `normal` builds no bias parameter at all, so it is a different object.
    "sweep_lag_bias_decay.yaml": {
        f"{_VAE}.alibi_slope_scale": 1.0,
        _RUN_NAME: "lag_attn_cfs_alibi_decay",
        _VARIANT: "lag_attn_cfs_alibi_decay",
    },
    # One clock rather than two: the source key null restores the single-reference resolution, so
    # both streams read on the target's 402.1604 s and the lag axis carries no inter-stream offset.
    "sweep_align_target_max.yaml": {
        f"{_VAE}.causal_align_reference": "target_max",
        _RUN_NAME: "lag_attn_cfs_align_target_max",
        _VARIANT: "lag_attn_cfs_align_target_max",
    },
    # The sharp lag memory: keys and values from the adapter output, one step of reach, against the
    # default's conv stem. The smallest of the three K/V models.
    "sweep_lag_kv_adapter.yaml": {
        f"{_VAE}.lag_kv_source": "adapter",
        _RUN_NAME: "lag_attn_cfs_kv_adapter",
        _VARIANT: "lag_attn_cfs_kv_adapter",
    },
    "sweep_target_clock_input.yaml": {
        f"{_VAE}.causal_align_reference": "target_max",
        f"{_VAE}.causal_target_forecast_clock": "input",
        _RUN_NAME: "lag_attn_cfs_clock_input",
        _VARIANT: "lag_attn_cfs_clock_input",
    },
    # The configuration this cell shipped before 2026-09-05, kept as the comparator the promoted
    # default replaced: legacy fractional-phase representation, the dual input reference, the
    # approximate physical clock and its stride-5 tiling, on the LEGACY shards.
    "sweep_legacy_dualref_physclock.yaml": {
        f"{_VAE}.c_y": 102,
        f"{_VAE}.c_u": 51,
        f"{_VAE}.causal_phase_operator": "ratio_power_v0",
        f"{_VAE}.causal_align_reference": "target_max",
        f"{_VAE}.causal_align_reference_source": 288.2672,
        f"{_VAE}.causal_target_forecast_clock": "physical",
        f"{_VAE}.anchor_stride": 5,
        "dataset_config.vae_train_datasets": [
            "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/pre_training_dataset/train_dataset_cs.hdf5",
            "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/pre_training_dataset/train_dataset_no_cs.hdf5",
        ],
        "dataset_config.vae_test_datasets": [
            "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/pre_training_dataset/test_dataset_cs.hdf5",
            "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/pre_training_dataset/test_dataset_no_cs.hdf5",
        ],
        "dataset_config.stat_path": "/data1/fetal-heart-tracing/HDF5_Datasets/REPOINT_ME_causal/stats.hdf5",
        _RUN_NAME: "lag_attn_cfs_dualref288_physclock",
        _VARIANT: "lag_attn_cfs_dualref288_physclock",
    },
}

#: The arms whose delta is one *model* axis plus the identity pair, and the value each names. The
#: identity keys are excluded here on purpose: what this checks is that the arm's own axis is a key
#: the constructor or the task actually reads, which the tracking block is not.
_IDENTITY_PATHS = (_RUN_NAME, _VARIANT)


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

    assert present == set(_ARMS) | {
        "default.yaml",
        "tiny.yaml",
        "smoke_hie.yaml",
        # The identifiability instrument's own delta. Not an arm and deliberately not linted as
        # one: it pins a geometry (the production lag window at tiny widths, a single clock) that
        # is held fixed against the default so a later default flip cannot move the instrument,
        # which is exactly the property the one-axis lint would refuse.
        "planted.yaml",
    }


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_that_declares_an_identity_names_itself_on_both_keys(name):
    """The drift guard, checked where it can be: a run's own records must say which arm trained.

    A config is a file on one machine and a run is an artifact directory on another; the two have
    been separated before, and a run whose artifacts cannot name its arm gets attributed to the
    default by whoever reads it later. So the arm goes in the name AND in the variant tag, and the
    two must agree -- a mismatch is the one failure that would otherwise survive both halves.
    """
    intended = _ARMS[name]
    if not any(path in intended for path in _IDENTITY_PATHS):
        pytest.skip(f"{name} predates the identity convention and inherits the default's name")

    flat = _flatten(_resolved(name))
    assert flat[_RUN_NAME] == flat[_VARIANT], "the name and the tag name different arms"
    assert flat[_RUN_NAME] == intended[_RUN_NAME]
    assert flat[_RUN_NAME] != _flatten(load_config(str(_DEFAULT)))[_RUN_NAME]


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


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_keeps_the_default_tiling_unless_its_delta_moves_it(name, default_flat):
    """The tiling travels with the horizon, so the invariant is the PAIRING rather than
    stride-equals-horizon: an arm that does not declare `anchor_stride` inherits the shipped
    S = H / 2 tiling with the shipped horizon, and an arm that declares it states its own
    pairing in its delta -- where the one-axis test already pins it. Either way the stride must
    stay inside $[1, H]$, where the constructor admits it."""
    intended = _ARMS[name]
    _floor, stride, horizon, _t_valid = _geometry(_resolved(name))

    assert 1 <= stride <= horizon
    if f"{_VAE}.anchor_stride" not in intended:
        assert stride == default_flat[f"{_VAE}.anchor_stride"] == 5 == horizon // 2


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


def test_the_horizon_arm_lengthens_the_block_and_pays_for_it_in_anchors():
    """The arm is a horizon change and nothing else: the default forecasts $10$ steps, the arm
    $15$, and the stride follows the default's $S = H / 2$ rule up so the half-horizon overlap is
    kept. What it buys is forecast reach and what it costs is anchors -- $T_\\mathrm{valid}$
    shortens and the tile count falls. The block grows with the horizon, to $15 \\times 76$, so no
    nat from this arm is comparable to a shipped one."""
    floor, stride, horizon, t_valid = _geometry(_resolved("sweep_horizon_15.yaml"))
    _shipped_floor, shipped_stride, shipped_horizon, shipped_t_valid = _geometry(
        load_config(str(_DEFAULT))
    )
    kept = 76

    assert horizon == 15 > shipped_horizon
    assert stride == 7 == horizon // 2
    assert horizon * kept == 1140 > shipped_horizon * kept  # against the shipped 760
    assert t_valid == 285 < shipped_t_valid  # the anchors the longer horizon costs
    # A_max, from the geometry rather than a literal, and fewer of them than the default tiles.
    assert -(-(t_valid - floor) // stride) == 22 < -(-(shipped_t_valid - floor) // shipped_stride)


def test_the_stride_arm_restores_the_dense_anchor_set():
    """1 is the value every other cell of the grid runs at, which is what makes this the control:
    at stride 1 the decoded set is the dense range $[F, T_\\mathrm{valid})$ and each target
    coefficient is scored by up to $H$ anchors again."""
    floor, stride, _horizon, t_valid = _geometry(_resolved("sweep_anchor_stride_1.yaml"))

    assert stride == 1
    assert -(-(t_valid - floor) // stride) == t_valid - floor == 156


def test_the_floor_arm_keeps_the_identical_channels_and_costs_the_tiles_the_stride_prices():
    """The cost of the policy, stated as a number rather than as an argument: the same channel set
    (neither the budget nor the alignment reference moved), $16$ fewer anchors, and the tiles that
    span costs at the shipped stride.

    The **withheld anchors** are the invariant and the **tiles** are not: the same $16$-step span
    costs $\\lceil 16 / S \\rceil$ tiles at phase $0$ -- one at the horizon-partitioning stride of
    $30$, two at $S = 10$, four at the shipped $S = 5$ (which the legacy physical-clock tiling
    also used). Both numbers
    are asserted, the tile count derived from the shipped stride rather than written as a literal,
    so that a future tiling change fails here rather than quietly re-pricing the arm -- and the
    tile count is taken over the EFFECTIVE ceiling, resolved against the committed shards, because
    a physical clock's trailing anchors do not exist to be withheld.

    The withheld span is $16$ rather than $17$ because the shipped floor moved with the alignment,
    from $133$ to $134$; the arm's own floor is a round policy number and did not.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    from .conftest import causal_config

    shipped_floor, stride, _horizon, t_valid = _geometry(load_config(str(_DEFAULT)))
    arm_floor, arm_stride, _h, arm_t_valid = _geometry(_resolved("sweep_floor_150.yaml"))

    budget = resolve_warmup_budget(
        causal_config(causal_target_forecast_clock="stored", anchor_stride=stride)
    )
    assert budget is not None
    ceiling = t_valid - budget.max_forecast_advance

    assert (arm_stride, arm_t_valid) == (stride, t_valid)
    assert arm_floor - shipped_floor == 16
    tiles_cost = -(-(ceiling - shipped_floor) // stride) - -(-(ceiling - arm_floor) // stride)
    assert tiles_cost == -(-16 // stride) == 4


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
