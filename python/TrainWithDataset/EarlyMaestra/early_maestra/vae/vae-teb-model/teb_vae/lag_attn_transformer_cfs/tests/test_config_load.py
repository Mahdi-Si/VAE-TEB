r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting either comparison model's, and
the price of that is drift -- drift in a key that has nothing to do with the encoder or with the
transform is exactly what destroys the two comparisons this package exists to make. A difference in
``seed``, ``lr``, ``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to
an architecture by every reading of the runs.

So parity is a tested property on **both** edges of the square:

* against ``teb_vae/lag_attn_cfs/configs/default.yaml`` every leaf must agree outside
  :data:`ENCODER_EDGE_EXEMPT_PATHS` -- the identity keys, the seven encoder keys, the five conv-LSTM
  keys they replace, ``lr_warmup_steps`` and the re-derived gradient clip;
* against ``teb_vae/lag_attn_transformer_fs/configs/default.yaml`` every leaf must agree outside
  :data:`TARGET_EDGE_EXEMPT_PATHS` -- the identity keys, the dataset, the one-sided geometry, the
  three keys that have no two-sided counterpart, and the two loss-scale constants this target domain
  re-derived.

The two exemption sets are disjoint except for the identity keys, which is what makes the square
close: a key that had to be exempted on both edges would be one neither comparison could read.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from train.test_utils import make_graph_model

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CONV_LSTM_ONLY_KEYS,
    absolutize_dataset_paths,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"

#: The two comparison configs, one per edge of the square.
_ENCODER_SIBLING = _REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "default.yaml"
_TARGET_SIBLING = (
    _REPO_ROOT / "teb_vae" / "lag_attn_transformer_fs" / "configs" / "default.yaml"
)

#: Every YAML this package may hold. A file that arrived undeclared is one nothing here checks, and
#: configs are launchable by path.
DECLARED_CONFIG_FILES = frozenset(
    {
        "default.yaml",
        "tiny.yaml",
        "smoke_hie.yaml",
        # The identifiability instrument's own delta: tiny widths at the PRODUCTION lag window and
        # a single clock, both pinned so that a later default flip cannot move the instrument.
        "planted.yaml",
        "sweep_anchor_stride_1.yaml",
        "sweep_lag_bias_decay.yaml",
        "sweep_align_target_max.yaml",
        "sweep_align_unaligned.yaml",
        "sweep_lag_kv_adapter.yaml",
        "sweep_source_dropout_02.yaml",
        "sweep_source_dropout_03.yaml",
    }
)

#: Keys that must exist. Some are checked by ``validate_config``; the rest are read with a bare
#: index in ``GraphModelBase.__init__``, which runs *before* the validator -- so a missing one raises
#: a bare ``KeyError`` rather than the friendly ``ValueError``.
_REQUIRED_PATHS = (
    "general_config.tag",
    "general_config.cuda_devices",
    "general_config.epochs",
    "general_config.lr",
    "general_config.batch_size",
    "general_config.folders_config",
    "advanced_config",
    "advanced_config.trainer",
    "advanced_config.logging",
    "dataset_config.vae_train_datasets",
    "dataset_config.vae_test_datasets",
    "dataset_config.stat_path",
    "model_config.VAE_model",
)

#: ``VAE_model`` keys the *task* consumes rather than the constructor. Each names a term of the
#: objective or its schedule, so a name here that reached nothing would train a different loss.
TASK_LEVEL_KEYS = (
    "beta_schedule",
    "free_bits",
    "beta_prior",
    "likelihood",
    "lambda_full",
    "lambda_base",
    "lambda_ms",
    "lambda_deriv",
    "lambda_boundary",
    "causal_warmup_budget_steps",
    # All three alignment keys are resolved against the SHARDS by the trainer and reach the
    # constructor only as the shift tuples, so none of them names a constructor argument.
    "causal_align_reference",
    # The source stream's own clock, snapped against the SOURCE's stored delays rather than the
    # target's. Task-level for the same reason as the key above and for one more: what it produces
    # is a second keep-index and a second shift tuple on one stream, which is a resolution result.
    "causal_align_reference_source",
    "causal_leg_alignment",
    "causal_reach_budget_s",
)

#: The seven keys this architecture adds, all of them describing the encoders being swapped in.
ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

_IDENTITY_PATHS = (
    "general_config.tag",
    "general_config.folders_config.out_dir_base",
    "advanced_config.tracking.mlflow.experiment_name",
    "advanced_config.tracking.mlflow.run_name",
    "advanced_config.tracking.mlflow.tags.variant",
)

_VAE = "model_config.VAE_model"

#: The ENCODER edge: what may differ from the conv-LSTM causal cell, and why. Anything not here must
#: be identical, because a difference outside this list would be attributed to the encoder by every
#: reading of the two runs.
ENCODER_EDGE_EXEMPT_PATHS: Dict[str, str] = {
    **{path: "identity: a shared name or tree makes the two runs indistinguishable" for path in _IDENTITY_PATHS},
    **{f"{_VAE}.{key}": "the encoder being swapped in" for key in ENCODER_KEYS},
    **{f"{_VAE}.{key}": "the conv-LSTM encoder being swapped out" for key in CONV_LSTM_ONLY_KEYS},
    "general_config.lr_warmup_steps": (
        "a property of the encoder's optimisation, not of the objective: a pre-norm attention "
        "stack is fragile in exactly the first few hundred updates"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "re-derived for this encoder at the same block and anchor count; the measurement is "
        "recorded in the config"
    ),
}

#: The TARGET edge: what may differ from the conv-Transformer two-sided cell.
TARGET_EDGE_EXEMPT_PATHS: Dict[str, str] = {
    **{path: "identity: a shared name or tree makes the two runs indistinguishable" for path in _IDENTITY_PATHS},
    "dataset_config.vae_train_datasets": "causal shards",
    "dataset_config.vae_test_datasets": "causal shards",
    "dataset_config.stat_path": "statistics accumulated excluding the warm-up region",
    "dataset_config.dataloader_config.dataset_kwargs.load_fields": (
        "'epoch' is the per-segment key the tile phase is derived from"
    ),
    f"{_VAE}.c_y": "the one-sided cascade keeps 36 + 66 rather than 43 + 66",
    f"{_VAE}.c_u": "the one-sided cascade keeps 36 + 15 rather than 43 + 15",
    # `horizon` is deliberately ABSENT: both cells forecast two minutes, so an exemption here would
    # be a permission with no divergence behind it and the stale-exemption test would report it.
    # The BLOCK still differs (30 * 98 against 30 * 78) because C_keep is what the budget decides,
    # which is why `additive_margin` below stays exempt while the horizon no longer is.
    f"{_VAE}.warmup_period": "the anchor floor the warm-up budget pairs with",
    f"{_VAE}.causal_reach_budget_s": "undefined on this dataset; the resolver refuses a value",
    f"{_VAE}.causal_warmup_budget_steps": "no two-sided counterpart",
    f"{_VAE}.causal_align_reference": (
        "no two-sided counterpart: the symmetric bank's channels already report one "
        "instant, so there is no clock to move them onto"
    ),
    f"{_VAE}.causal_leg_alignment": "only a causal shard records a phase-harmonic operator",
    f"{_VAE}.anchor_stride": "no two-sided counterpart",
    f"{_VAE}.lag_floor": "no two-sided counterpart",
    "advanced_config.spike_breaker.additive_margin": (
        "stated in nats of the summed block, and this target domain changes both the block and "
        "the anchor count"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "measured at this block and anchor count, which the target domain moves"
    ),
    # The training controls. Only these two appear here: the comparison config has no
    # `secondary_monitor` key at all and the six architecture switches are absent from it too, and
    # this edge compares the paths the two files SHARE -- so a key one side does not have is not a
    # divergence it could declare.
    "advanced_config.callbacks.early_stopping.enabled": (
        "stops on val/total_loss where the comparison model runs its epoch budget out; enabled "
        "because a run of this cell reaches its composite optimum well before its budget ends"
    ),
    "advanced_config.callbacks.early_stopping.patience": (
        "the second half of the control above, in validation epochs; inheriting the comparison "
        "model's value would make the flag above inert rather than merely different"
    ),
}

#: The tiny variant's declared delta, and the local variant's. Written out so a stray override is a
#: failure rather than a surprise on the box.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.plot_frequency",
        "general_config.lr_warmup_steps",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        f"{_VAE}.d_model",
        f"{_VAE}.d_z",
        f"{_VAE}.d_head",
        f"{_VAE}.max_lag",
        f"{_VAE}.dropout",
        f"{_VAE}.encoder_d_ff",
        f"{_VAE}.target_attention_blocks",
        f"{_VAE}.source_attention_blocks",
        f"{_VAE}.source_attention_window",
        f"{_VAE}.likelihood",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "advanced_config.tracking.mlflow.enabled",
    }
)

SMOKE_HIE_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.lr_warmup_steps",
        "general_config.plot_frequency",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        f"{_VAE}.beta_schedule.warmup_epochs",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "dataset_config.dataloader_config.prefetch_factor",
        "advanced_config.tracking.mlflow.enabled",
    }
)

#: Surviving target channels at the shipped warm-up budget.
KEPT_TARGET_CHANNELS = 98

#: Surviving source channels at the shipped ALIGNMENT, which is the only rule that gates this
#: stream: the warm-up budget keeps all $51$, and the reference drops every channel whose composed
#: delay is above it.
#:
#: **The number moved when the source gained a clock of its own.** Under one reference at the
#: target's $402.1604$ s it was $47$ -- four ``up_st`` casualties and every one of the fifteen
#: ``up_ph``. The shipped source reference is $288.2672$ s, a hundred and fourteen seconds faster,
#: and it costs eight more: $30$ of $36$ ``up_st`` and $9$ of $15$ ``up_ph``. That is the trade the
#: reference was pinned on -- a physiological delay lands mid-window on the lag axis instead of
#: below its near edge -- and it is priced here rather than argued, because a source width is what
#: every attended summary in this cell is built from.
KEPT_SOURCE_CHANNELS = 39


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


def _has(config: dict, dotted: str) -> bool:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def _get(config: dict, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = node[part]
    return node


def _model_kwargs_from(config: dict, tmp_path) -> dict:
    """Run a config through the real driver's signature sweep and return the kwargs.

    The paths are absolutised first, because this driver's sweep **reads the shards**: the warm-up
    boundary is a property of the data and there is nothing to read it from otherwise.
    """
    path = Path(tmp_path) / "config.yaml"
    path.write_text(
        yaml.safe_dump(absolutize_dataset_paths(config), sort_keys=False), encoding="utf-8"
    )
    return LagAttnTrfCfsTrainer(config_file_path=str(path))._build_model_kwargs()


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib ``warnings``
    module, so a ``pytest.warns`` assertion against it would pass no matter what the config held.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def shipped() -> dict:
    return load_config(str(_CONFIG))


@pytest.fixture
def tiny() -> dict:
    return load_config(str(_TINY))


@pytest.fixture
def smoke_hie() -> dict:
    return load_config(str(_SMOKE_HIE))


@pytest.fixture
def encoder_sibling() -> dict:
    return load_config(str(_ENCODER_SIBLING))


@pytest.fixture
def target_sibling() -> dict:
    return load_config(str(_TARGET_SIBLING))


# --------------------------------------------------------------------------------------
# The shipped config loads and everything in it reaches something
# --------------------------------------------------------------------------------------
def test_every_effectively_required_key_is_present(shipped):
    assert [path for path in _REQUIRED_PATHS if not _has(shipped, path)] == []


def test_the_shipped_config_validates_with_no_unknown_or_dead_key_warnings(
    tmp_path, loguru_warnings
):
    """Drives the framework's real validator, not a copy of its rules."""
    graph_model = make_graph_model(
        _CONFIG, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []


def test_every_vae_model_key_reaches_the_constructor_or_the_task(shipped):
    """A key that reaches nothing does not raise -- the constructor has a default for everything --
    so the run trains a *different architecture* than its config describes, and only a checkpoint
    that will not reload months later reveals it."""
    accepted = set(inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters)

    orphans = [
        key
        for key in _get(shipped, _VAE)
        if key not in accepted and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == []


def test_the_seven_encoder_keys_are_present_and_reach_the_constructor(shipped):
    """The whole of what this cell changes against its encoder sibling. A key that failed to reach
    the constructor would leave the encoder at its own default with the config still describing the
    one the run was meant to have."""
    accepted = set(inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters)
    block = _get(shipped, _VAE)

    for key in ENCODER_KEYS:
        assert key in block, key
        assert key in accepted, key


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_no_replaced_encoder_key_survives_in_any_config(key, shipped, tiny, smoke_hie):
    """Each names a component this architecture does not have. The driver's sweep forwards by name
    against the real signature, so a copied key would not crash a launch -- it would simply reach
    nothing, which is why its absence is asserted rather than its rejection."""
    for config in (shipped, tiny, smoke_hie):
        assert key not in _get(config, _VAE), key


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_budget_keeps(tmp_path):
    """The binding this model's whole unit convention rests on, resolved through the real driver:
    the warm-up budget decides the surviving channels, the survivors decide the decoder width, and
    the width decides what every reported nat is summed over.

    Driven on the tiny variant because the shipped config's shard paths are deliberately
    non-existent placeholders and this resolution **reads the shards**; the tiny variant carries the
    identical geometry, which is exactly why it does.
    """
    kwargs = _model_kwargs_from(load_config(str(_TINY)), tmp_path)

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == KEPT_TARGET_CHANNELS
    assert len(kwargs["target_warmup_steps"]) == KEPT_TARGET_CHANNELS
    assert len(kwargs["source_keep_index"]) == KEPT_SOURCE_CHANNELS
    assert len(kwargs["target_align_delays"]) == KEPT_TARGET_CHANNELS
    assert len(kwargs["source_align_delays"]) == KEPT_SOURCE_CHANNELS
    # The reach guard's keywords, which name a different mechanism and stay refused.
    assert "target_delays" not in kwargs and "source_delays" not in kwargs

    model = SeqVaeLagAttnTrfCfs(**kwargs)
    assert model.decoder.mean_head.out_features == KEPT_TARGET_CHANNELS
    assert model.raw_per_step == 16  # untouched by the width
    assert model.horizon * model.decoder_out_channels == 2940


def test_the_shipped_geometry_pairs_the_floor_with_the_budget(shipped):
    r"""$F \ge \max(B - 1,\ \max_c(W'_c + d_c))$, and $B$ is the *survivors'* maximum rather than
    the configured threshold. The alignment makes the second term bind, at exactly $B$."""
    vae = _get(shipped, _VAE)

    assert vae["warmup_period"] == 134
    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["causal_align_reference"] == "target_max"
    assert vae["warmup_period"] >= vae["causal_warmup_budget_steps"] - 1
    assert vae["c_y"] == CAUSAL_C_Y
    assert vae["c_u"] == CAUSAL_C_U


def test_the_anchor_stride_equals_the_configured_horizon(shipped):
    """At $S = H$ the windows partition the timeline. Below it they overlap again; above it the
    constructor refuses, because target steps between two tiles would never be scored."""
    vae = _get(shipped, _VAE)

    assert vae["anchor_stride"] == vae["horizon"] == 30
    assert vae["lag_floor"] == 0


# --------------------------------------------------------------------------------------
# Pin one: the encoder edge
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", _IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(
    shipped, encoder_sibling, target_sibling, path
):
    """Inheriting any of these mixes this model's runs into a comparison model's experiment, which
    is unrecoverable afterwards because the two are then indistinguishable by every field anything
    indexes on."""
    assert _get(shipped, path) != _get(encoder_sibling, path)
    assert _get(shipped, path) != _get(target_sibling, path)


def test_every_comparable_leaf_equals_the_encoder_siblings_value(shipped, encoder_sibling):
    """The encoder edge, total. Both files describe the same target domain at the same budget, so
    every leaf outside the declared list must agree or the encoder comparison is confounded."""
    mine, theirs = _flatten(shipped), _flatten(encoder_sibling)
    shared = set(mine) & set(theirs)

    differing = {path for path in shared if mine[path] != theirs[path]}
    assert differing <= set(ENCODER_EDGE_EXEMPT_PATHS), sorted(
        differing - set(ENCODER_EDGE_EXEMPT_PATHS)
    )

    # And the key sets differ only where the encoder does.
    assert set(mine) - set(theirs) == {
        "general_config.lr_warmup_steps",
        *(f"{_VAE}.{key}" for key in ENCODER_KEYS),
    }
    assert set(theirs) - set(mine) == {f"{_VAE}.{key}" for key in CONV_LSTM_ONLY_KEYS}


def test_the_encoder_edge_declares_no_exemption_that_is_no_longer_a_divergence(
    shipped, encoder_sibling
):
    """An exemption that is not a divergence is a claim nothing tests. It is the more dangerous half
    of the pair: a stale entry silently widens the allow-list against a future edit."""
    mine, theirs = _flatten(shipped), _flatten(encoder_sibling)

    stale = [
        path
        for path in ENCODER_EDGE_EXEMPT_PATHS
        if path in mine and path in theirs and mine[path] == theirs[path]
    ]

    assert stale == []


# --------------------------------------------------------------------------------------
# Pin two: the target edge
# --------------------------------------------------------------------------------------
def test_every_comparable_leaf_equals_the_target_siblings_value(shipped, target_sibling):
    """The target edge, total. Both files describe the same encoder, so every leaf outside the
    declared list must agree or the transform comparison is confounded."""
    mine, theirs = _flatten(shipped), _flatten(target_sibling)
    shared = set(mine) & set(theirs)

    differing = {path for path in shared if mine[path] != theirs[path]}
    assert differing <= set(TARGET_EDGE_EXEMPT_PATHS), sorted(
        differing - set(TARGET_EDGE_EXEMPT_PATHS)
    )

    # The keys with no two-sided counterpart; nothing on the two-sided side is missing here.
    assert set(mine) - set(theirs) == {
        f"{_VAE}.causal_warmup_budget_steps",
        f"{_VAE}.causal_align_reference",
        f"{_VAE}.causal_leg_alignment",
        f"{_VAE}.anchor_stride",
        f"{_VAE}.lag_floor",
        # The per-block reconstruction weights. The two-sided cell scores its channels uniformly,
        # so it has no counterpart to compare against rather than a differing value.
        f"{_VAE}.target_weight_st",
        f"{_VAE}.target_weight_ph",
        # The second half of the dual clock. Present here and nowhere else for the same reason
        # `causal_align_reference` is: a symmetric bank's channels already report one instant.
        f"{_VAE}.causal_align_reference_source",
        # The four architecture switches whose off-state is bitwise the two-sided model. They are
        # ABSENT from that config rather than set to their off-values, which is the stronger
        # statement: the two-sided cells never take these keys, so no config of theirs can drift
        # onto an arm, and the comparison stays a transform comparison.
        f"{_VAE}.lag_kv_source",
        f"{_VAE}.prior_availability_input",
        f"{_VAE}.persistence_residual",
        f"{_VAE}.horizon_weight_halflife_steps",
        # The lag-bias seed's slope multiplier. Shipped FLAT here, where a decaying seed would
        # predict the lag-0 peak this cell exists to measure; the two-sided cell reads no
        # physiological delay off its lag axis and leaves the constructor default standing.
        f"{_VAE}.alibi_slope_scale",
        # The second checkpoint criterion. Absent there, and absence builds no second callback.
        "advanced_config.callbacks.model_checkpoint.secondary_monitor",
    }
    assert set(theirs) - set(mine) == set()


def test_the_target_edge_declares_no_exemption_that_is_no_longer_a_divergence(
    shipped, target_sibling
):
    mine, theirs = _flatten(shipped), _flatten(target_sibling)

    stale = [
        path
        for path in TARGET_EDGE_EXEMPT_PATHS
        if path in mine and path in theirs and mine[path] == theirs[path]
    ]

    assert stale == []


def test_the_two_edges_overlap_only_where_they_must(shipped):
    """The square closes iff the two allow-lists are disjoint outside the identity keys and the one
    constant both edges move. A key exempt on both edges is one neither comparison can read, and it
    would be exempt for two different reasons that nothing forces to agree."""
    both = set(ENCODER_EDGE_EXEMPT_PATHS) & set(TARGET_EDGE_EXEMPT_PATHS)

    assert both == set(_IDENTITY_PATHS) | {"advanced_config.trainer.gradient_clip_val"}


def test_the_clip_moved_on_both_edges_and_the_margin_on_only_one(
    shipped, encoder_sibling, target_sibling
):
    """The one asymmetry worth stating as a test. ``additive_margin`` is in nats of the summed block
    and the encoder edge changes neither the block nor the anchor count, so it must equal the
    conv-LSTM causal cell's exactly -- and it is the two-sided cell's that differs.
    ``gradient_clip_val`` is a gradient statistic, which both edges move.

    The value moved from $3.0 \\times 10^{3}$ to $9.0 \\times 10^{3}$ when both causal cells went to
    the two-minute horizon, and the equality is what the move had to preserve: the two cells were
    re-measured separately (worst excursions $3928$ here against $5090$ there) and the shared margin
    clears the larger, because one bar across the edge is the point.
    """
    breaker = "advanced_config.spike_breaker.additive_margin"
    clip = "advanced_config.trainer.gradient_clip_val"

    assert _get(shipped, breaker) == _get(encoder_sibling, breaker) == 9.0e3
    assert _get(shipped, breaker) != _get(target_sibling, breaker)
    assert _get(shipped, clip) != _get(encoder_sibling, clip)
    assert _get(shipped, clip) != _get(target_sibling, clip)
    # The relative test stays off at the same value on every cell: it is a switch, not a scale.
    floor = "advanced_config.spike_breaker.ema_floor"
    assert _get(shipped, floor) == _get(encoder_sibling, floor) == _get(target_sibling, floor)


# --------------------------------------------------------------------------------------
# The rest of the shipped block
# --------------------------------------------------------------------------------------
def test_the_config_directory_holds_exactly_the_declared_files():
    """A file that arrived undeclared is one nothing here checks, and configs are launchable by
    path."""
    assert {path.name for path in _CONFIG_DIR.glob("*.yaml")} == set(DECLARED_CONFIG_FILES)


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient and the causal variant does not store it.
    In ``load_fields`` the loader would raise -- after every rank had initialised -- and in
    ``normalize_fields`` it is silently ignored and reads as though the block were handled."""
    for path in _CONFIG_DIR.glob("*.yaml"):
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_the_two_phase_key_fields_are_loaded(shipped):
    """``load_fields`` is honoured literally, with no forced additions. Without ``guid`` and
    ``epoch`` the tile phase has nothing per-segment to key on, and $A_{\\max}$ is a geometry
    constant either way -- so no shape, no count and no metric would differ."""
    load_fields = _get(shipped, "dataset_config.dataloader_config.dataset_kwargs.load_fields")

    assert "guid" in load_fields and "epoch" in load_fields


def test_the_target_blocks_are_loaded_and_normalized(shipped):
    """Both, in both lists. An unnormalised target makes the Gaussian NLL meaningless against a
    unit-scale variance model with the loader raising nothing."""
    loader = _get(shipped, "dataset_config.dataloader_config")

    for field in LagAttnTrfCfsTrainer.TARGET_FIELDS:
        assert field in loader["normalize_fields"], field
        assert field in loader["dataset_kwargs"]["load_fields"], field


def test_the_boundary_shape_weight_is_zero(shipped):
    """A slicing identity over *adjacent* anchors, against a set whose entries are $S$ apart. The
    shared objective raises on the combination and the driver's pre-flight refuses it before a run
    directory exists."""
    assert _get(shipped, f"{_VAE}.lambda_boundary") == 0.0


def test_precision_is_float32(shipped):
    assert _get(shipped, "advanced_config.trainer.precision") == "32-true"


def test_compile_ships_off_but_is_live_for_this_driver(shipped, tmp_path):
    """Different from the conv-LSTM causal cell's, where the key is inert: the LSTM that made the
    raw base refuse compilation outright is gone here, so the key becomes live and shipping it off
    is a decision rather than a formality."""
    assert _get(shipped, "advanced_config.trainer.compile") is False

    config = dict(shipped)
    config["advanced_config"]["trainer"]["compile"] = True
    path = Path(tmp_path) / "compile.yaml"
    path.write_text(
        yaml.safe_dump(absolutize_dataset_paths(config), sort_keys=False), encoding="utf-8"
    )

    assert LagAttnTrfCfsTrainer(config_file_path=str(path)).compile_model_requested() is True


def test_num_sanity_val_steps_is_zero(shipped):
    """REQUIRED: the metrics callback has no sanity guard, so a sanity pass would shift every epoch
    number against MLflow and the checkpoint filenames."""
    assert _get(shipped, "advanced_config.trainer.num_sanity_val_steps") == 0


def test_the_step_warmup_is_configured_and_positive(shipped):
    """The one non-encoder key on the encoder edge. At $0$ the task delegates to the framework's
    epoch-granularity path, which cannot express a ramp completing inside a fraction of one epoch."""
    assert _get(shipped, "general_config.lr_warmup_steps") > 0


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """$z$ is the only route to the decoder, so a nonzero $\\beta$ before the decoder can use the
    latent at all is the standard route to posterior collapse."""
    schedule = _get(shipped, f"{_VAE}.beta_schedule")

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["end"] == 1.0


def test_the_plotting_block_keeps_the_shared_drivers_spelling(shipped):
    """The callback assembly is inherited whole and reads this literal; renaming the block to match
    this package would leave the figure permanently off, with ``enabled: true`` still reading
    correct and nothing in the log saying why."""
    assert LagAttnTrfCfsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert _has(shipped, f"advanced_config.callbacks.{LagAttnTrfCfsTrainer.PLOT_CONFIG_KEY}")


# --------------------------------------------------------------------------------------
# The two derived variants
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, declared",
    [("tiny.yaml", TINY_DELTA_PATHS), ("smoke_hie.yaml", SMOKE_HIE_DELTA_PATHS)],
)
def test_a_variant_names_only_its_declared_deltas(name, declared, shipped):
    """Key sets must match exactly -- a typo'd override *adds* a path rather than moving one, and a
    config key that reaches nothing raises nothing."""
    variant = load_config(str(_CONFIG_DIR / name))
    assert "base" not in variant

    variant_flat, shipped_flat = _flatten(variant), _flatten(shipped)
    assert set(variant_flat) == set(shipped_flat)

    differing = {path for path in shipped_flat if variant_flat[path] != shipped_flat[path]}
    assert differing == set(declared), sorted(differing ^ set(declared))


def test_the_tiny_variant_inherits_the_geometry_that_decides_what_it_exercises(tiny, shipped):
    """The decoder's width IS the resolved budget's surviving-channel count and the decoded anchor
    count IS the floor, stride and horizon -- so a smoke run at a shrunken budget or a shorter
    window would exercise a decoder and an anchor set the production run does not have."""
    for key in (
        "sequence_length",
        "c_y",
        "c_u",
        "horizon",
        "warmup_period",
        "anchor_stride",
        "causal_warmup_budget_steps",
        "coverage_floor",
        "raw_per_step",
    ):
        assert _get(tiny, f"{_VAE}.{key}") == _get(shipped, f"{_VAE}.{key}"), key
    assert (
        _get(tiny, "dataset_config.dataloader_config.dataset_kwargs.trim_minutes")
        == _get(shipped, "dataset_config.dataloader_config.dataset_kwargs.trim_minutes")
    )


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny):
    """Both head constraints at once, and they are independent: ``num_heads * d_head == d_model``
    for the lag attention, and ``d_model / encoder_num_heads`` even for rotary position encoding."""
    vae = _get(tiny, _VAE)

    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0
    assert (vae["d_model"] // vae["encoder_num_heads"]) % 2 == 0


def test_the_tiny_variant_exercises_the_step_warmup_inside_a_smoke_fit(tiny):
    """A ramp longer than the run is a ramp the smoke never leaves, and one of zero is a path the
    smoke never enters. Both would leave the conv-Transformer half of the diamond untested."""
    assert 0 < _get(tiny, "general_config.lr_warmup_steps") <= 8


def test_the_tiny_variant_points_at_the_committed_causal_shard(tiny):
    """And at the causal statistics beside it: the two-sided pair carries no ``transform``
    attribute and no warm-up vectors, and the pre-flight refuses it by name."""
    for key in ("vae_train_datasets", "vae_test_datasets"):
        paths = _get(tiny, f"dataset_config.{key}")
        assert paths == ["teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5"], key
        assert (_REPO_ROOT / paths[0]).exists()
    stats = _get(tiny, "dataset_config.stat_path")
    assert stats.endswith("tiny_stats_causal.hdf5")
    assert (_REPO_ROOT / stats).exists()


def test_the_resolved_tiny_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver."""
    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnTrfCfs(**_model_kwargs_from(load_config(str(_TINY)), tmp_path))
    # The smoke model is small everywhere except where it must not be: the decoder still emits the
    # production width and the forward still decodes the production tile count.
    assert model.d_model == 32
    assert model.decoder_out_channels == KEPT_TARGET_CHANNELS
    assert model.anchor_stride == 30


def test_the_local_variant_names_a_built_and_leg_aligned_causal_shard(smoke_hie):
    """The tripwire this replaces asserted the shard's *absence*, so that the day one was built it
    would fail and someone would re-read the header. That day came; the header now records what was
    built, and this asserts the config names it.

    ``output/`` is gitignored, so the shard is a dev-box artefact rather than a committed fixture:
    the config contract is checked everywhere, the file's own attributes only where it is present.
    The two-sided ``output/hie_cs.hdf5`` still cannot stand in for one."""
    for key in ("vae_train_datasets", "vae_test_datasets"):
        for path in _get(smoke_hie, f"dataset_config.{key}"):
            assert "causal" in path, path
    assert "PREREQUISITE, AND IT IS NOW SATISFIED" in _SMOKE_HIE.read_text(encoding="utf-8")

    shard = _REPO_ROOT / _get(smoke_hie, "dataset_config.vae_train_datasets")[0]
    if not shard.is_file():
        pytest.skip(f"{shard} is a gitignored dev-box artefact and is absent here")

    import h5py

    expected = _get(smoke_hie, "model_config.VAE_model.causal_leg_alignment")
    with h5py.File(shard, "r") as handle:
        assert handle.attrs["transform"] == "causal"
        # An aligned shard and an unaligned one share every width, warm-up and stored delay, so
        # this attribute is the only thing on the file that could disagree with the config.
        assert handle.attrs["causal_leg_alignment"] == expected


def test_the_local_variant_scales_the_step_warmup_into_its_own_budget(smoke_hie, shipped):
    """A few hundred windows at batch 32 is on the order of ten steps an epoch, so the shipped
    2,000-step ramp would still be climbing when the run ended and every column would be read off a
    model that had never trained at its own learning rate."""
    assert _get(smoke_hie, "general_config.lr_warmup_steps") < _get(
        shipped, "general_config.lr_warmup_steps"
    )
    assert _get(smoke_hie, "general_config.lr_warmup_steps") > 0


def test_the_local_variant_runs_on_one_device_with_tracking_off(smoke_hie):
    assert _get(smoke_hie, "general_config.cuda_devices") == [0]
    assert _get(smoke_hie, "advanced_config.tracking.mlflow.enabled") is False


def test_the_local_variant_reads_the_same_shard_as_its_encoder_sibling(smoke_hie):
    """The encoder edge is only readable if both cells are read on the same data."""
    sibling = load_config(
        str(_REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "smoke_hie.yaml")
    )

    for key in ("vae_train_datasets", "vae_test_datasets", "stat_path"):
        assert _get(smoke_hie, f"dataset_config.{key}") == _get(sibling, f"dataset_config.{key}")
