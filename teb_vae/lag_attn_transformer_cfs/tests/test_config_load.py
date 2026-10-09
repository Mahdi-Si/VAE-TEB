r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting either comparison model's, and
the price of that is drift -- drift in a key that has nothing to do with the encoder or with the
transform is exactly what destroys the two comparisons this package exists to make. A difference in
``seed``, ``lr``, ``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to
an architecture by every reading of the runs.

So parity is a tested property on **both** edges of the square:

* against ``teb_vae/lag_attn_cfs/configs/default.yaml`` every leaf must agree outside
  :data:`ENCODER_EDGE_EXEMPT_PATHS` -- the identity keys, the seven encoder keys, the five conv-LSTM
  keys they replace, ``lr_warmup_steps``, the re-derived gradient clip, and the leaves of the
  2026-09-23 final revision (:data:`FINAL_REVISION_PATHS`);
* against ``teb_vae/lag_attn_transformer_fs/configs/default.yaml`` every leaf must agree outside
  :data:`TARGET_EDGE_EXEMPT_PATHS` -- the identity keys, the dataset, the one-sided geometry, the
  keys that have no two-sided counterpart, the two loss-scale constants this target domain
  re-derived, and the final-revision leaves the two-sided cell shares.

Every exemption must also still *be* a divergence, so a stale entry cannot silently widen either
allow-list. Beside the two edges: every YAML in ``configs/`` validates with no unknown or dead key,
the derived geometry of the shipped config is self-consistent, the two variants move no key the
shipped config lacks, and the tiny variant resolves through the real driver into a buildable model.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from train.test_utils import make_graph_model

from teb_vae.lag_attn_cfs.tests.conftest import INT_C_U, INT_C_Y

from .conftest import CONV_LSTM_ONLY_KEYS, absolutize_dataset_paths

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
    # The source channel choice: resolved by the same resolver into `source_keep_index`, which
    # the constructor takes; the key itself names no constructor argument.
    "causal_source_channels",
    # The forecast clock: resolved against the shards' delay vectors into the signed
    # `target_forecast_shift` tuple the constructor takes, exactly as the alignment references
    # resolve into theirs.
    "causal_target_forecast_clock",
    "causal_leg_alignment",
    # The phase-harmonic operator VERSION the shards were built with, checked against their root
    # attribute exactly as the leg alignment is; a resolver expectation, not a constructor key.
    "causal_phase_operator",
    "causal_reach_budget_s",
    # The per-channel scored-horizon rule: resolved by the trainer against the shards' phase-leg
    # frequencies into the `target_scored_horizon` tuple the constructor takes.
    "target_phase_fast_cutoff_hz",
    "target_phase_fast_horizon",
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

#: The leaves the owner's 2026-09-23 FINAL REVISION moved in this cell and in no sibling. The conv-LSTM
#: causal cell is deliberately not revised, so each of these is a declared divergence on the encoder
#: edge; the ones the two-sided cell also carries are declared on the target edge too.
FINAL_REVISION_PATHS: Dict[str, str] = {
    f"{_VAE}.horizon": "final revision: H back to 30 in this cell only",
    f"{_VAE}.anchor_stride": "final revision: S = H / 2 follows the horizon",
    f"{_VAE}.horizon_weight_halflife_steps": "final revision: half-life = H",
    f"{_VAE}.max_lag": "final revision: 37 in this cell only",
    f"{_VAE}.lag_kv_source": "final revision: the one-step adapter memory",
    f"{_VAE}.source_dropout": "final revision: 0.2 on the source pathway alone",
    "advanced_config.spike_breaker.additive_margin": (
        "final revision: scaled by the block ratio with the horizon, not re-measured"
    ),
}

#: The keys the final revision ADDED, present in this config and in neither sibling's.
FINAL_REVISION_KEYS = (
    "forecast_ar_residual",
    "target_phase_fast_cutoff_hz",
    "target_phase_fast_horizon",
)

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
    # The transformer cell carries the 2026-09-23 final revision and the conv-LSTM cell does not,
    # so the encoder edge no longer differs in the encoder alone: every final-revision leaf is
    # declared here rather than the conv-LSTM config being edited to follow.
    **FINAL_REVISION_PATHS,
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
    # `horizon` is OFF this edge again since 2026-09-23: both cells forecast 30 steps. The BLOCK
    # differs as it always did because C_keep is what the budget decides.
    f"{_VAE}.warmup_period": "the anchor floor the warm-up budget pairs with",
    f"{_VAE}.causal_reach_budget_s": "undefined on this dataset; the resolver refuses a value",
    f"{_VAE}.causal_warmup_budget_steps": "no two-sided counterpart",
    f"{_VAE}.causal_align_reference": (
        "no two-sided counterpart: the symmetric bank's channels already report one "
        "instant, so there is no clock to move them onto"
    ),
    f"{_VAE}.causal_leg_alignment": "only a causal shard records a phase-harmonic operator",
    f"{_VAE}.causal_phase_operator": "only a causal shard records a phase-operator version",
    f"{_VAE}.anchor_stride": "no two-sided counterpart",
    f"{_VAE}.lag_floor": "no two-sided counterpart",
    "advanced_config.spike_breaker.additive_margin": (
        "stated in nats of the summed block, and this target domain changes both the block and "
        "the anchor count"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "measured at this block and anchor count, which the target domain moves"
    ),
    # The final-revision leaves the two-sided cell also carries (the rest are absent there).
    f"{_VAE}.max_lag": FINAL_REVISION_PATHS[f"{_VAE}.max_lag"],
    f"{_VAE}.source_dropout": FINAL_REVISION_PATHS[f"{_VAE}.source_dropout"],
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

#: Surviving target channels at the shipped warm-up budget, on the committed integer-operator
#: fixture the tiny variant reads.
KEPT_TARGET_CHANNELS = 76

#: Every YAML this package holds, discovered rather than listed: a config that arrives is checked
#: by the first test below without anyone having to register it.
_ALL_CONFIGS = sorted(path.name for path in _CONFIG_DIR.glob("*.yaml"))


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
# Every config loads, validates, and everything in it reaches something
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_every_config_validates_and_every_vae_key_reaches_the_constructor_or_the_task(
    name, tmp_path, loguru_warnings
):
    """Drives the framework's real validator, not a copy of its rules, on each file resolved
    through its ``base:`` chain -- the only way it ever reaches the experiment driver.

    A ``VAE_model`` key that reaches nothing does not raise -- the constructor has a default for
    everything -- so the run trains a *different architecture* than its config describes, and only
    a checkpoint that will not reload months later reveals it. A copied conv-LSTM key is the
    standing case.
    """
    resolved = resolve_config_file(str(_CONFIG_DIR / name), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    config = load_config(str(_CONFIG_DIR / name))
    assert "base" not in config
    accepted = set(inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters)
    orphans = [
        key for key in _get(config, _VAE) if key not in accepted and key not in TASK_LEVEL_KEYS
    ]
    assert orphans == []


# --------------------------------------------------------------------------------------
# The shipped geometry is self-consistent
# --------------------------------------------------------------------------------------
def test_the_shipped_geometry_pairs_the_floor_with_the_budget_on_the_integer_shards(shipped):
    r"""$F \ge B - 1$, with $B$ the survivors' maximum rather than the configured threshold, and
    the declared widths are the committed integer-operator fixture's."""
    vae = _get(shipped, _VAE)

    assert vae["warmup_period"] >= vae["causal_warmup_budget_steps"] - 1
    assert vae["c_y"] == INT_C_Y
    assert vae["c_u"] == INT_C_U


def test_the_anchor_stride_pairs_with_the_forecast_clock(shipped):
    r"""The three travel together: the stored clock's ceiling is $T_\mathrm{valid} = T - H$, and
    the final revision's rules are $S = H / 2$ and half-life $= H$. Asserted as rules rather than
    literals, so a horizon change that left the stride or the half-life behind -- a different tile
    count and a different weight profile, silently -- fails here rather than training a different
    objective. The fast scored horizon must lie inside the horizon, or the resolver refuses it."""
    vae = _get(shipped, _VAE)

    # The span below is the stored clock's; any other clock moves the ceiling.
    assert vae["causal_target_forecast_clock"] == "stored"
    assert vae["anchor_stride"] == vae["horizon"] // 2
    assert vae["horizon_weight_halflife_steps"] == float(vae["horizon"])
    assert 1 <= vae["target_phase_fast_horizon"] <= vae["horizon"]
    # The span and the tile count the config comments state, derived the way the net derives them.
    span = vae["sequence_length"] - vae["horizon"] - vae["warmup_period"]
    assert span > vae["anchor_stride"]
    assert -(-span // vae["anchor_stride"]) >= 2


def test_the_decoder_depth_still_covers_the_horizon(shipped):
    r"""The family's receptive-field criterion $\mathrm{RF} = 1 + (k-1)(2^d - 1) \ge H + 1$ over the
    horizon axis. Nothing in the net ties the depth to the horizon, so a longer horizon that left
    the depth behind would decode its far tokens from a core that cannot see the near ones."""
    vae = _get(shipped, _VAE)
    receptive_field = 1 + (vae["horizon_kernel"] - 1) * (2 ** vae["horizon_depth"] - 1)

    assert receptive_field >= vae["horizon"] + 1


# --------------------------------------------------------------------------------------
# Pin one: the encoder edge
# --------------------------------------------------------------------------------------
def test_the_identity_keys_are_this_models_own(shipped, encoder_sibling, target_sibling):
    """Inheriting any of these mixes this model's runs into a comparison model's experiment, which
    is unrecoverable afterwards because the two are then indistinguishable by every field anything
    indexes on."""
    for path in _IDENTITY_PATHS:
        assert _get(shipped, path) != _get(encoder_sibling, path), path
        assert _get(shipped, path) != _get(target_sibling, path), path


def test_every_comparable_leaf_equals_the_encoder_siblings_value(shipped, encoder_sibling):
    """The encoder edge, total. Both files describe the same target domain at the same budget, so
    every leaf outside the declared list must agree or the encoder comparison is confounded."""
    mine, theirs = _flatten(shipped), _flatten(encoder_sibling)
    shared = set(mine) & set(theirs)

    differing = {path for path in shared if mine[path] != theirs[path]}
    assert differing <= set(ENCODER_EDGE_EXEMPT_PATHS), sorted(
        differing - set(ENCODER_EDGE_EXEMPT_PATHS)
    )

    # And the key sets differ only where the encoder does, plus the keys the final revision added.
    assert set(mine) - set(theirs) == {
        "general_config.lr_warmup_steps",
        *(f"{_VAE}.{key}" for key in ENCODER_KEYS),
        *(f"{_VAE}.{key}" for key in FINAL_REVISION_KEYS),
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
        f"{_VAE}.causal_phase_operator",
        f"{_VAE}.anchor_stride",
        f"{_VAE}.lag_floor",
        # The per-block reconstruction weights. The two-sided cell scores its channels uniformly,
        # so it has no counterpart to compare against rather than a differing value.
        f"{_VAE}.target_weight_st",
        f"{_VAE}.target_weight_ph",
        # The second half of the dual clock. Present here and nowhere else for the same reason
        # `causal_align_reference` is: a symmetric bank's channels already report one instant.
        f"{_VAE}.causal_align_reference_source",
        # The forecast clock, absent from the two-sided cell for the same reason: its stored
        # coefficients already describe the instant they are stored at, so there is no per-channel
        # staleness for a clock to correct on the question side.
        f"{_VAE}.causal_target_forecast_clock",
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
        # The final revision's likelihood keys, which no sibling carries.
        *(f"{_VAE}.{key}" for key in FINAL_REVISION_KEYS),
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


# --------------------------------------------------------------------------------------
# The rest of the shipped block
# --------------------------------------------------------------------------------------
def test_the_target_blocks_and_the_two_phase_key_fields_are_loaded(shipped):
    """``load_fields`` is honoured literally, with no forced additions. Both target blocks must be
    loaded and normalised -- an unnormalised target makes the Gaussian NLL meaningless against a
    unit-scale variance model with the loader raising nothing -- and without ``guid`` and ``epoch``
    the tile phase has nothing per-segment to key on, with no shape, count or metric differing."""
    loader = _get(shipped, "dataset_config.dataloader_config")
    load_fields = loader["dataset_kwargs"]["load_fields"]

    for field in LagAttnTrfCfsTrainer.TARGET_FIELDS:
        assert field in loader["normalize_fields"], field
        assert field in load_fields, field
    assert "guid" in load_fields and "epoch" in load_fields


def test_compile_is_live_for_this_driver(shipped, tmp_path):
    """Different from the conv-LSTM causal cell's, where the key is inert: the LSTM that made the
    raw base refuse compilation outright is gone here, so the key reaches the driver's decision."""
    config = dict(shipped)
    config["advanced_config"]["trainer"]["compile"] = True
    path = Path(tmp_path) / "compile.yaml"
    path.write_text(
        yaml.safe_dump(absolutize_dataset_paths(config), sort_keys=False), encoding="utf-8"
    )

    assert LagAttnTrfCfsTrainer(config_file_path=str(path)).compile_model_requested() is True


def test_the_plotting_block_sits_under_the_key_the_driver_reads(shipped):
    """The callback assembly is inherited whole and reads the driver's key; a block renamed to match
    this package would leave the figure permanently off, with ``enabled: true`` still reading
    correct and nothing in the log saying why."""
    assert _has(shipped, f"advanced_config.callbacks.{LagAttnTrfCfsTrainer.PLOT_CONFIG_KEY}")


# --------------------------------------------------------------------------------------
# The two derived variants
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["tiny.yaml", "smoke_hie.yaml"])
def test_a_variant_moves_no_key_the_shipped_config_lacks(name, shipped):
    """Key sets must match exactly -- a typo'd override *adds* a path rather than moving one, and a
    config key that reaches nothing raises nothing."""
    variant = load_config(str(_CONFIG_DIR / name))

    assert set(_flatten(variant)) == set(_flatten(shipped))


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


def test_the_step_warmup_ramp_is_live_in_every_variant(shipped, tiny, smoke_hie):
    """At $0$ the task delegates to the framework's epoch-granularity path, which cannot express a
    ramp completing inside a fraction of one epoch. A smoke ramp longer than the smoke run is one it
    never leaves, so each variant scales the ramp into its own budget."""
    ramp = "general_config.lr_warmup_steps"

    assert _get(shipped, ramp) > 0
    assert 0 < _get(tiny, ramp) <= 8
    assert 0 < _get(smoke_hie, ramp) < _get(shipped, ramp)


def test_the_tiny_variant_resolves_through_the_driver_into_the_shipped_decoder(tmp_path):
    """The smoke model is small everywhere except where it must not be: the budget resolved against
    the committed shard decides the surviving channels, the survivors decide the decoder width, and
    the forward still decodes the production tiling -- with the final revision's two likelihood
    mechanisms reaching the model through the real driver."""
    kwargs = _model_kwargs_from(load_config(str(_TINY)), tmp_path)

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"])
    # Unaligned: the mapping emits no shift vectors at all rather than zeros, and the reach guard's
    # keywords, which name a different mechanism, stay refused.
    for absent in ("target_align_delays", "source_align_delays", "target_delays", "source_delays"):
        assert absent not in kwargs, absent

    model = SeqVaeLagAttnTrfCfs(**kwargs)
    shipped_vae = _get(load_config(str(_CONFIG)), _VAE)
    assert model.decoder.mean_head.out_features == KEPT_TARGET_CHANNELS
    assert model.decoder_out_channels == KEPT_TARGET_CHANNELS
    assert model.anchor_stride == shipped_vae["anchor_stride"]
    assert model.horizon == shipped_vae["horizon"]
    assert model.forecast_ar_residual is True
    assert model.target_scored_horizon is not None
    assert set(model.target_scored_horizon) == {
        shipped_vae["target_phase_fast_horizon"], shipped_vae["horizon"]
    }
    # The stored clock advances nothing: every anchor up to T_valid is decoded.
    assert model.target_forecast_shift is None
    assert model.anchor_ceiling == model.geometry.t_valid


def test_the_ua_s0_variant_reads_one_source_channel_on_the_physical_clock(shipped, tmp_path):
    r"""``ua_s0_only.yaml`` moves three ``VAE_model`` leaves and its identity, and nothing else;
    resolved through the real driver on the committed fixture those leaves give a source gate of
    width $1$ (declared index $0$, $S_0$), a one-channel adapter, and the physical clock's ceiling
    $T_{\mathrm{valid}} - \max_c s_c$ -- and the model forwards at that geometry."""
    import torch

    variant = load_config(str(_CONFIG_DIR / "ua_s0_only.yaml"))
    mine, theirs = _flatten(variant), _flatten(shipped)
    assert set(mine) - set(theirs) == {f"{_VAE}.causal_source_channels"}
    moved = {path for path in theirs if mine[path] != theirs[path]} - set(_IDENTITY_PATHS)
    assert moved == {f"{_VAE}.causal_target_forecast_clock", f"{_VAE}.anchor_stride"}
    assert _get(variant, f"{_VAE}.causal_target_forecast_clock") == "physical"

    # The same three leaves over the tiny config, which is what points at the committed shard.
    tiny = load_config(str(_TINY))
    for key in ("causal_source_channels", "causal_target_forecast_clock", "anchor_stride"):
        tiny["model_config"]["VAE_model"][key] = _get(variant, f"{_VAE}.{key}")
    kwargs = _model_kwargs_from(tiny, tmp_path)
    assert kwargs["source_keep_index"] == (0,)
    # $S_0$'s stored warm-up of 5 steps sits inside the loader's 15-step trim, so it rebases to 0.
    assert kwargs["source_warmup_steps"] == (0,)
    assert "source_align_delays" not in kwargs

    model = SeqVaeLagAttnTrfCfs(**kwargs)
    assert model.source_gate is not None and model.source_gate.out_channels == 1
    assert model.source_adapter.in_dim == 1
    assert model.anchor_ceiling == model.geometry.t_valid - max(kwargs["target_forecast_shift"])

    vae = _get(tiny, _VAE)
    length, c_y, c_u = vae["sequence_length"], vae["c_y"], vae["c_u"]
    y_st, y_ph = torch.randn(2, length, 36), torch.randn(2, length, c_y - 36)
    with torch.no_grad():
        out = model(y_st, y_ph, torch.randn(2, length, c_u), anchor_phase=0)
    assert out["source_state"].shape == (2, length, model.d_model)
    assert out["mu_full"].shape[-1] == model.decoder_out_channels


def test_the_local_variant_reads_the_same_shard_as_its_encoder_sibling(smoke_hie):
    """The encoder edge is only readable if both cells are read on the same data."""
    sibling = load_config(
        str(_REPO_ROOT / "teb_vae" / "lag_attn_cfs" / "configs" / "smoke_hie.yaml")
    )

    for key in ("vae_train_datasets", "vae_test_datasets", "stat_path"):
        assert _get(smoke_hie, f"dataset_config.{key}") == _get(sibling, f"dataset_config.{key}")
