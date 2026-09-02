r"""Lint for the arms: a single axis each, a closed inventory, and a stride that follows the horizon.

The **tiling** arm is this package's own long-standing one, and the arms it does *not* carry are a
decision rather than an oversight. The floor, horizon and depth arms answer questions about the
target domain, which ``teb_vae/lag_attn_cfs/configs`` already asks; the encoder arms answer
questions about the encoder, which ``teb_vae/lag_attn_transformer_rws/configs`` already asks. The
tiling arm is the only one of those whose answer could differ between the two encoders, because
what it moves is the per-step gradient noise -- $136$ decoded anchors against $\approx 4.53$ -- and
a pre-normalised attention stack is exactly the architecture whose stability is sensitive to that.

The **lag** arms beside it exist in both feature-target cells, because every gate that produced
them was read per parent: the two encoders answer the same question differently and a result on one
is not a result on the other. The **source-dropout** pair exists here alone, which is the mirror of
the rule above -- it regularises the source map, and this is the encoder that map is built from.

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
#: The six declared after the tiling arm carry **three** paths each rather than one: their own
#: axis, and the two identity keys. The identity pair is not a second delta -- it is the same
#: delta, written where a finished run can still be asked which side of the axis it trained on.
#: Two of them exist in this cell alone: ``source_dropout`` is the one seam that regularises the
#: source map without touching the target pathway, and this is the encoder whose attention stack
#: is what it regularises.
_ARMS: Dict[str, Dict[str, Any]] = {
    # The tiling ablation. 1 is the INERT stride -- the dense range every other cell of the grid
    # decodes -- so this arm is the control that says what the tiling costs or buys.
    "sweep_anchor_stride_1.yaml": {f"{_VAE}.anchor_stride": 1},
    # The lag-bias seed. The default seeds the learnable (num_heads, L) bias FLAT; this restores
    # the decaying seed, which predicts a lag-0 peak on its own. `lag_bias_init` deliberately does
    # not move with it -- `normal` builds no bias parameter at all, so it is a different object.
    "sweep_lag_bias_decay.yaml": {
        f"{_VAE}.alibi_slope_scale": 1.0,
        _RUN_NAME: "lag_attn_trf_cfs_alibi_decay",
        _VARIANT: "lag_attn_trf_cfs_alibi_decay",
    },
    # One clock rather than two: the source key null restores the single-reference resolution, so
    # both streams read on the target's 402.1604 s and the lag axis carries no inter-stream offset.
    "sweep_align_target_max.yaml": {
        f"{_VAE}.causal_align_reference_source": None,
        _RUN_NAME: "lag_attn_trf_cfs_align_target_max",
        _VARIANT: "lag_attn_trf_cfs_align_target_max",
    },
    # No clock at all -- the unaligned end of the pre-registered pair. The SECOND model key is
    # forced rather than a second axis: a source reference is one half of a pair of clocks, and the
    # resolver refuses it against an unaligned target by name, so an arm moving only the first
    # would not launch.
    "sweep_align_unaligned.yaml": {
        f"{_VAE}.causal_align_reference": None,
        f"{_VAE}.causal_align_reference_source": None,
        _RUN_NAME: "lag_attn_trf_cfs_unaligned",
        _VARIANT: "lag_attn_trf_cfs_unaligned",
    },
    # The sharp lag memory: keys and values from the adapter output, one step of reach, against the
    # default's conv stem. The smallest of the three K/V models.
    "sweep_lag_kv_adapter.yaml": {
        f"{_VAE}.lag_kv_source": "adapter",
        _RUN_NAME: "lag_attn_trf_cfs_kv_adapter",
        _VARIANT: "lag_attn_trf_cfs_kv_adapter",
    },
    # The source-regularisation pair. Two points on one axis rather than two axes, which is why
    # they are two files and not one with two keys.
    "sweep_source_dropout_02.yaml": {
        f"{_VAE}.source_dropout": 0.2,
        _RUN_NAME: "lag_attn_trf_cfs_source_dropout_02",
        _VARIANT: "lag_attn_trf_cfs_source_dropout_02",
    },
    "sweep_source_dropout_03.yaml": {
        f"{_VAE}.source_dropout": 0.3,
        _RUN_NAME: "lag_attn_trf_cfs_source_dropout_03",
        _VARIANT: "lag_attn_trf_cfs_source_dropout_03",
    },
    # The forecast-clock pair. Each moves TWO model keys, and the second travels with the first:
    # the shipped stride of 5 exists to recover tiles under the physical clock's shortened
    # ceiling, so an arm that restored the stored or input clock at stride 5 would compare two
    # tilings as well as two clocks. 30 restores the horizon-partitioning tiling every historical
    # run trained at.
    "sweep_target_clock_stored.yaml": {
        f"{_VAE}.causal_target_forecast_clock": "stored",
        f"{_VAE}.anchor_stride": 30,
        _RUN_NAME: "lag_attn_trf_cfs_clock_stored",
        _VARIANT: "lag_attn_trf_cfs_clock_stored",
    },
    "sweep_target_clock_input.yaml": {
        f"{_VAE}.causal_target_forecast_clock": "input",
        f"{_VAE}.anchor_stride": 30,
        _RUN_NAME: "lag_attn_trf_cfs_clock_input",
        _VARIANT: "lag_attn_trf_cfs_clock_input",
    },
}

#: The two paths excluded from the "one axis" reading above: what they move is how a finished run
#: is identified, not what it computes.
_IDENTITY_PATHS = (_RUN_NAME, _VARIANT)


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


def test_the_config_directory_holds_exactly_the_declared_arms():
    """Both directions: a declared arm whose file is missing, and a stray ``sweep_*.yaml`` nobody
    declared -- which would be launchable, and would run outside every assertion below."""
    assert {path.name for path in _CONFIG_DIR.glob("sweep_*.yaml")} == set(_ARMS)


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
def test_an_arm_leaves_at_least_one_tile_at_every_phase(name):
    """The feasibility the constructor refuses on. At the last phase the first anchor is
    $F + S - 1$; if it does not exist, a sample drawn at that phase contributes no forecast at all
    and its share of the epoch is silently dropped."""
    floor, stride, _horizon, t_valid = _geometry(_resolved(name))

    assert 1 <= stride
    assert floor + stride <= t_valid


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_pairs_its_floor_with_the_shipped_budget(name):
    r"""$F \ge B - 1$ over the survivors, which at the shipped budget is $F \ge 133$. The arm keeps
    the budget, so an arm that lowered the floor would score assumed pre-recording history as signal
    with every shape correct."""
    floor, _stride, _horizon, _t_valid = _geometry(_resolved(name))
    vae = _resolved(name)["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert floor >= vae["causal_warmup_budget_steps"] - 1


def test_the_stride_arm_restores_the_dense_anchor_set():
    """1 is the value every other cell of the grid runs at, which is what makes this the control:
    at stride 1 the decoded set is the dense range $[F, T_\\mathrm{valid})$ and each target
    coefficient is scored by up to $H$ anchors again."""
    floor, stride, _horizon, t_valid = _geometry(_resolved("sweep_anchor_stride_1.yaml"))

    assert stride == 1
    assert -(-(t_valid - floor) // stride) == t_valid - floor == 136


def test_the_default_pairs_the_stride_with_the_forecast_clock():
    """The tiling travels with the forecast clock: the physical clock's ceiling leaves a 51-anchor
    span, and stride 5 is what keeps ~10 training tiles per sample there (A_max = 11). The
    stored-clock arm restores the horizon-partitioning 30 with its clock, where its own delta
    test pins the pairing."""
    default = load_config(str(_DEFAULT))
    _floor, stride, horizon, _t_valid = _geometry(default)

    assert default["model_config"]["VAE_model"]["causal_target_forecast_clock"] == "physical"
    assert stride == 5
    assert horizon == 30


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_ramps_beta_from_exactly_zero(name):
    """No arm sweeps beta, and every arm needs the ramp: $z$ is the only route to the decoder, so an
    arm that lost the warm-up would collapse for a reason that has nothing to do with its own
    axis."""
    schedule = _resolved(name)["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0


@pytest.mark.parametrize("name", sorted(_ARMS))
def test_an_arm_keeps_the_step_granular_ramp(name):
    """The encoder half of the diamond. An arm that zeroed it would fall back to the framework's
    epoch-granularity path and change the architecture's optimisation alongside its own axis."""
    assert _resolved(name)["general_config"]["lr_warmup_steps"] > 0


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
    """It is a slicing identity over adjacent anchors, and the driver refuses any other value
    unconditionally rather than making the refusal conditional on the stride."""
    assert _resolved(name)["model_config"]["VAE_model"]["lambda_boundary"] == 0.0
