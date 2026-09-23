r"""Lint for the arms: a single axis each, a closed inventory, and a stride that follows the horizon.

The **tiling** arm is this package's own long-standing one, and the arms it does *not* carry are a
decision rather than an oversight. The floor, horizon and depth arms answer questions about the
target domain, which ``teb_vae/lag_attn_cfs/configs`` already asks; the encoder arms answer
questions about the encoder, which ``teb_vae/lag_attn_transformer_rws/configs`` already asks. The
tiling arm is the only one of those whose answer could differ between the two encoders, because
what it moves is the per-step gradient noise -- all $T_\mathrm{valid} - F$ decoded anchors against
$A_{\max}$ -- and a pre-normalised attention stack is exactly the architecture whose stability is
sensitive to that.

The **lag** arms beside it exist in both feature-target cells, because every gate that produced
them was read per parent: the two encoders answer the same question differently and a result on one
is not a result on the other. The **source-dropout** arm exists here alone, which is the mirror of
the rule above -- it regularises the source map, and this is the encoder that map is built from.
The two **likelihood** arms ablate the 2026-09-23 final revision, which only this cell carries.

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
#: Every arm declared after the tiling arm carries its own axis plus the two identity keys. The identity pair is not a second delta -- it is the same
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
        f"{_VAE}.causal_align_reference": "target_max",
        _RUN_NAME: "lag_attn_trf_cfs_align_target_max",
        _VARIANT: "lag_attn_trf_cfs_align_target_max",
    },
    # The previous K/V memory: the conv stem the default shipped until the 2026-09-23 final revision
    # moved it to the one-step adapter. The comparator for the old key/value choice. (The adapter
    # and dropout-0.2 arms were deleted that day: both now equal the default.)
    "sweep_lag_kv_conv_stem.yaml": {
        f"{_VAE}.lag_kv_source": "conv_stem",
        _RUN_NAME: "lag_attn_trf_cfs_kv_conv_stem",
        _VARIANT: "lag_attn_trf_cfs_kv_conv_stem",
    },
    # The two likelihood ablations of the final revision, one mechanism each: the factorised
    # Gaussian along the horizon, and every cell scored.
    "sweep_factorised_likelihood.yaml": {
        f"{_VAE}.forecast_ar_residual": False,
        _RUN_NAME: "lag_attn_trf_cfs_factorised_likelihood",
        _VARIANT: "lag_attn_trf_cfs_factorised_likelihood",
    },
    "sweep_all_cells_scored.yaml": {
        f"{_VAE}.target_phase_fast_cutoff_hz": None,
        f"{_VAE}.target_phase_fast_horizon": None,
        _RUN_NAME: "lag_attn_trf_cfs_all_cells_scored",
        _VARIANT: "lag_attn_trf_cfs_all_cells_scored",
    },
    # The source-regularisation axis: the default ships 0.2 since 2026-09-23, so this is the next
    # point on it.
    "sweep_source_dropout_03.yaml": {
        f"{_VAE}.source_dropout": 0.3,
        _RUN_NAME: "lag_attn_trf_cfs_source_dropout_03",
        _VARIANT: "lag_attn_trf_cfs_source_dropout_03",
    },
    "sweep_target_clock_input.yaml": {
        f"{_VAE}.causal_align_reference": "target_max",
        f"{_VAE}.causal_target_forecast_clock": "input",
        _RUN_NAME: "lag_attn_trf_cfs_clock_input",
        _VARIANT: "lag_attn_trf_cfs_clock_input",
    },
    # The configuration this cell shipped before 2026-09-05, kept as the comparator the promoted
    # default replaced: legacy fractional-phase representation, the dual input reference, the
    # approximate physical clock and its stride-5 tiling, on the LEGACY shards. The horizon and the
    # other final-revision leaves are NOT part of its delta: they follow the default, so the
    # comparison holds them equal.
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
        _RUN_NAME: "lag_attn_trf_cfs_dualref288_physclock",
        _VARIANT: "lag_attn_trf_cfs_dualref288_physclock",
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
    default_floor, _s, _h, default_t_valid = _geometry(load_config(str(_DEFAULT)))

    assert stride == 1
    assert -(-(t_valid - floor) // stride) == t_valid - floor == default_t_valid - default_floor


def test_the_default_pairs_the_stride_with_the_forecast_clock():
    r"""The tiling travels with the horizon: on the stored clock the ceiling is
    $T_\mathrm{valid} = T - H$, and $S = H / 2$ tiles the span $[F, T_\mathrm{valid})$ into
    $\lceil (T_\mathrm{valid} - F) / S \rceil$ tiles at phase 0 and one fewer at the last phase
    whenever the span is not a multiple of $S$. The tile counts are read off the net's own
    :meth:`_build_anchor_index` rather than restated, so the arithmetic here and the one that runs
    cannot disagree. Pinned here as well as in test_config_load.py because this file is where a
    stride left behind by a horizon change is meant to be caught."""
    import torch

    from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs

    default = load_config(str(_DEFAULT))
    floor, stride, horizon, t_valid = _geometry(default)
    vae = default["model_config"]["VAE_model"]

    assert vae["causal_target_forecast_clock"] == "stored"
    assert stride == horizon // 2
    assert vae["horizon_weight_halflife_steps"] == float(horizon)

    geometry = type(
        "_Geometry",
        (),
        {"anchor_stride": stride, "horizon": horizon, "warmup_period": floor,
         "anchor_ceiling": t_valid},
    )()
    counts = []
    for phase in range(stride):
        _index, valid = CausalWarmupInputs._build_anchor_index(
            geometry, 1, torch.device("cpu"), anchor_phase=phase
        )
        counts.append(int(valid.sum()))
    a_max = -(-(t_valid - floor) // stride)
    assert counts[0] == a_max == int(valid.shape[1])
    assert min(counts) == -(-(t_valid - floor - (stride - 1)) // stride)


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
