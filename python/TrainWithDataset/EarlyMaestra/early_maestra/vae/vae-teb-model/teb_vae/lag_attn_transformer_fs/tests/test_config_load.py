r"""The shipped configs load, validate, build this model, and do not drift from their siblings.

``default.yaml`` here is written out in full rather than inheriting either comparison model's, and
the price of that is drift -- drift in a key that has nothing to do with the axis under study is
exactly what destroys the two comparisons this package exists to make.

**Every shipped config** is resolved, validated by the framework's own validator, swept into
constructor kwargs by the real driver and built, and every ``VAE_model`` key it carries must reach
the constructor or the task. The built decoder must be exactly as wide as the resolved target
keep-index, which is the binding every reported nat is summed over.

**Three pins**, each catching something the other two cannot:

**Pin 1, against the conv-Transformer sibling: total.** Both models are built by the same
constructor, so every leaf is comparable, with an ``"<absent>"`` sentinel so a key present in only
one file is drift.

**Pin 2, against the feature-domain sibling: schema-limited.** The encoder keys are excluded by name
-- the ones this constructor has and that one does not, and the ones describing the encoder being
replaced. What Pin 2 adds over Pin 1 plus the two sibling pins is that no *further* leaf appears on
the target edge.

**Pin 3, the square closes -- on keys, not values.** The set of paths differing on each target edge
is equal, and likewise for each encoder edge, outside the declared leaves. What Pin 3 catches is a
leaf entering or leaving one edge without the other following.

Each pin's allow-list has a reverse guard, so an exemption that no longer marks a divergence fails.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.tests.test_config_load import (
    ENCODER_KEYS,
    REPLACED_ENCODER_KEYS,
)
from train.test_utils import make_graph_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"

#: The three other cells of the grid. ``_ENCODER_SIBLING`` is this model at the other encoder and
#: ``_TARGET_SIBLING`` this encoder at the other target; ``_ROOT_CONFIG`` is the cell both siblings
#: descend from, and it is read only by Pin 3, which needs all four nodes of the square.
_ENCODER_SIBLING = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"
_TARGET_SIBLING = (
    _REPO_ROOT / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "default.yaml"
)
_ROOT_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"

#: The feature-domain sibling's local validation config, which ``smoke_hie.yaml`` is pinned against.
_SIBLING_SMOKE_HIE = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "smoke_hie.yaml"

#: The *config* keys the target edge is blind to: the encoder schema this constructor adds and the
#: one it replaces. ``decoder_out_channels`` is not among them -- it is named in no YAML, because the
#: width follows the target gate.
SCHEMA_ONLY_PATHS = tuple(
    f"model_config.VAE_model.{key}" for key in ENCODER_KEYS + REPLACED_ENCODER_KEYS
)

#: ``VAE_model`` keys that name no constructor argument and are still real: the experiment driver and
#: the task read each of them by name. ``causal_reach_budget_s`` is translated rather than forwarded
#: -- it resolves into the four concrete channel tuples the net takes, and here those tuples also
#: decide the decoder's width.
TASK_LEVEL_KEYS = (
    "beta_schedule",
    "kld_beta",
    "beta_prior",
    "lambda_full",
    "lambda_base",
    "lambda_ms",
    "lambda_deriv",
    "lambda_boundary",
    "likelihood",
    "free_bits",
    "causal_reach_budget_s",
)

#: Why the three auxiliary shape weights are the one *model* divergence on the target edge.
AUX_OFF_REASON = (
    "the shape terms read the forecast block's last axis as consecutive raw samples, and this block's "
    "last axis is an unordered index of surviving target channels, so the weights ship at 0.0"
)

#: **Pin 1's** allow-list against the conv-Transformer sibling. No wildcards, so adding an entry is a
#: decision rather than an omission.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture and the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.spike_breaker.additive_margin": (
        "stated in nats of the summed block, which is $H \\cdot C_{keep}$ coefficients here against "
        "$H \\cdot R$ samples there"
    ),
    "model_config.VAE_model.lambda_ms": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_deriv": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_boundary": AUX_OFF_REASON,
    "advanced_config.trainer.gradient_clip_val": (
        "MEASURED: re-derived from this stack's own pre-clip gradient-norm quantile"
    ),
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: **Pin 2's** allow-list against the feature-domain sibling, on top of :data:`SCHEMA_ONLY_PATHS`.
TARGET_EDGE_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture and the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "general_config.lr_warmup_steps": (
        "the step-granular ramp both conv-Transformer models add; absent from both conv-LSTM ones"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "MEASURED: re-derived from this stack's own pre-clip gradient-norm quantile"
    ),
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: Every leaf either pin declares. **Pin 3** subtracts this from both sides of each comparison, so it
#: reads the *undeclared* key sets.
DECLARED_PATHS = frozenset(PARITY_EXEMPT_PATHS) | frozenset(TARGET_EDGE_EXEMPT_PATHS)


def _leaves(node: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dict leaf of a config mapping.

    Lists are leaves: a config list is a value (device ids, shard paths, kernels), never a namespace,
    so descending into one would compare positions rather than settings.

    Args:
        node: The mapping to walk.
        prefix: Dotted prefix accumulated so far.

    Yields:
        One ``(path, value)`` pair per leaf.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), node


def _differing_paths(mine: dict, theirs: dict) -> set:
    """Return the dotted paths at which two flattened configs disagree, absence included."""
    return {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }


def _drift(mine: dict, theirs: dict, excluded) -> dict:
    """Return ``{path: (mine, theirs)}`` for every differing leaf outside ``excluded``."""
    return {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(_differing_paths(mine, theirs) - set(excluded))
    }


def _model_kwargs_from(config: dict) -> dict:
    """Run a config through the real driver's signature sweep and return the kwargs.

    Args:
        config: A loaded config mapping.

    Returns:
        The constructor kwargs a launch on this config would build the net from.
    """
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return LagAttnTrfFsTrainer(config_file_path=str(path))._build_model_kwargs()


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib ``warnings``
    module, so a ``pytest.warns`` assertion against it would pass no matter what the config
    contained.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def shipped() -> dict:
    return load_config(str(_CONFIG))


@pytest.fixture
def smoke_hie() -> dict:
    return load_config(str(_SMOKE_HIE))


@pytest.fixture
def target_sibling() -> dict:
    """The conv-Transformer raw-target config: this model at the other target domain."""
    return load_config(str(_TARGET_SIBLING))


@pytest.fixture
def encoder_sibling() -> dict:
    """The conv-LSTM feature-target config: this target domain at the other encoder."""
    return load_config(str(_ENCODER_SIBLING))


# --------------------------------------------------------------------------------------
# Every shipped config loads, validates, reaches something and builds
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("config_path", [_CONFIG, _TINY, _SMOKE_HIE], ids=lambda p: p.name)
def test_each_shipped_config_validates_and_builds_a_decoder_as_wide_as_the_budget_keeps(
    config_path, tmp_path, loguru_warnings
):
    """Resolved first, which is the only way a config ever reaches the experiment driver.

    A ``VAE_model`` key that reaches nothing does not raise -- the constructor has a default for
    everything -- so the run would train a *different architecture* than its config describes. The
    decoder width follows the resolved target keep-index, and the target blocks and the plot block
    must be where the inherited guard and the inherited callback assembly look for them.
    """
    resolved = resolve_config_file(str(config_path), str(tmp_path))
    make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    ).validate_config()
    assert [message for message in loguru_warnings if "config:" in message] == []

    config = load_config(str(config_path))
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)
    orphans = [
        key
        for key in config["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]
    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"

    kwargs = _model_kwargs_from(config)
    model = SeqVaeLagAttnTrfFs(**kwargs)
    assert model.decoder_out_channels == len(kwargs["target_keep_index"])
    assert model.decoder.mean_head.out_features == model.decoder_out_channels

    dataloader = config["dataset_config"]["dataloader_config"]
    for field in LagAttnTrfFsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field
    assert config["advanced_config"]["callbacks"][LagAttnTrfFsTrainer.PLOT_CONFIG_KEY]["enabled"]
    # ``fhr_up_ph`` mixes both signals in one coefficient: anywhere in a config it would break the
    # target-only / source-conditioned separation and put the source into the forecast target.
    assert not [path for path, value in _leaves(config) if "fhr_up_ph" in str(value)]


@pytest.mark.parametrize("config_path", [_TINY, _SMOKE_HIE], ids=lambda p: p.name)
def test_each_local_config_points_at_files_that_exist(config_path):
    """The two configs a dev box runs name repo-relative shards and statistics; a moved file would
    otherwise surface only as a loader error deep inside a launch."""
    dataset = load_config(str(config_path))["dataset_config"]

    for path in (*dataset["vae_train_datasets"], *dataset["vae_test_datasets"], dataset["stat_path"]):
        assert (_REPO_ROOT / path).is_file(), f"{path} is missing"


# --------------------------------------------------------------------------------------
# Pin 1: total, against the conv-Transformer sibling
# --------------------------------------------------------------------------------------
def test_every_leaf_equals_the_target_siblings_value(shipped, target_sibling):
    """This model's net is that one's plus a target-domain mixin, so every leaf is comparable."""
    drift = _drift(dict(_leaves(shipped)), dict(_leaves(target_sibling)), PARITY_EXEMPT_PATHS)

    assert drift == {}, (
        f"these leaves differ from the conv-Transformer sibling's config and are not declared in "
        f"PARITY_EXEMPT_PATHS: {drift}"
    )


def test_pin_one_declares_no_exemption_that_is_no_longer_a_divergence(shipped, target_sibling):
    """The reverse guard: an exemption for a key that no longer differs is a permission that outlived
    its reason, and the next accidental divergence there would go unreported."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(target_sibling))

    stale = [
        path
        for path in PARITY_EXEMPT_PATHS
        if mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


# --------------------------------------------------------------------------------------
# Pin 2: schema-limited, against the feature-domain sibling
# --------------------------------------------------------------------------------------
def test_every_comparable_leaf_equals_the_encoder_siblings_value(shipped, encoder_sibling):
    """No leaf outside the encoder schema and the declared allow-list differs on the target edge."""
    drift = _drift(
        dict(_leaves(shipped)),
        dict(_leaves(encoder_sibling)),
        set(SCHEMA_ONLY_PATHS) | set(TARGET_EDGE_EXEMPT_PATHS),
    )

    assert drift == {}, (
        f"these leaves differ from the feature-domain sibling's config outside the encoder keys and "
        f"are not declared in TARGET_EDGE_EXEMPT_PATHS: {drift}"
    )


def test_pin_two_declares_no_exemption_that_is_no_longer_a_divergence(shipped, encoder_sibling):
    """The reverse guard again."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(encoder_sibling))

    stale = [
        path
        for path in TARGET_EDGE_EXEMPT_PATHS
        if mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


# --------------------------------------------------------------------------------------
# Pin 3: the square closes, on key sets
# --------------------------------------------------------------------------------------
def test_the_two_target_edges_of_the_square_carry_the_same_key_set(
    shipped, target_sibling, encoder_sibling
):
    """Sets, not values: a leaf entering or leaving one target edge without the other following is a
    change to the shared ancestor that one descendant tracks and the other does not, which every
    pairwise pin passes because each pin only ever sees one edge. Declared leaves are excluded: each
    carries a stated reason and is covered by its own pin's reverse guard."""
    root = load_config(str(_ROOT_CONFIG))

    mine = _differing_paths(dict(_leaves(shipped)), dict(_leaves(target_sibling)))
    theirs = _differing_paths(dict(_leaves(encoder_sibling)), dict(_leaves(root)))

    assert mine - DECLARED_PATHS == theirs - DECLARED_PATHS


def test_the_two_encoder_edges_of_the_square_carry_the_same_key_set(
    shipped, target_sibling, encoder_sibling
):
    """The other axis, where the undeclared set is exactly the encoder schema: an encoder key added
    to one row of the grid and not the other is the drift that makes the two rows non-comparable."""
    root = load_config(str(_ROOT_CONFIG))

    mine = _differing_paths(dict(_leaves(shipped)), dict(_leaves(encoder_sibling)))
    theirs = _differing_paths(dict(_leaves(target_sibling)), dict(_leaves(root)))

    assert mine - DECLARED_PATHS == theirs - DECLARED_PATHS == set(SCHEMA_ONLY_PATHS)


# --------------------------------------------------------------------------------------
# The dev-box validation variant
# --------------------------------------------------------------------------------------
def test_the_local_variant_inherits_everything_the_reading_depends_on(smoke_hie, shipped):
    """Only the run's *scale* is local. The block the NLL sums over, the weights it is balanced
    against, the clamp the log-variances sit inside, the encoders and the breaker are inherited, or
    the reading is of a different model."""
    vae = smoke_hie["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_reach_budget_s", "horizon", "beta_prior", "likelihood", "logvar_clamp", *ENCODER_KEYS
    ):
        assert vae[key] == shipped_vae[key], key
    assert vae["beta_schedule"]["end"] == shipped_vae["beta_schedule"]["end"]
    assert (
        smoke_hie["advanced_config"]["trainer"]["gradient_clip_val"]
        == shipped["advanced_config"]["trainer"]["gradient_clip_val"]
    )
    assert smoke_hie["advanced_config"]["spike_breaker"] == shipped["advanced_config"][
        "spike_breaker"
    ]


def test_the_local_variant_matches_the_encoder_siblings_local_variant_outside_the_encoder(
    smoke_hie,
):
    """The production pins' argument, applied where the comparison is actually run: the two dev-box
    configs differ in the encoder keys, the step ramp and identity, and nothing else."""
    sibling = load_config(str(_SIBLING_SMOKE_HIE))

    drift = _drift(
        dict(_leaves(smoke_hie)),
        dict(_leaves(sibling)),
        set(SCHEMA_ONLY_PATHS) | set(TARGET_EDGE_EXEMPT_PATHS),
    )

    assert drift == {}, (
        f"the two dev-box configs differ outside the encoder keys and the declared allow-list, so "
        f"the local encoder comparison is confounded: {drift}"
    )
