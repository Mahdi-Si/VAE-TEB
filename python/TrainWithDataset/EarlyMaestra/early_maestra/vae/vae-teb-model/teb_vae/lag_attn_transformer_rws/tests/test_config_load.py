r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the comparison model's,
because the two ``VAE_model`` blocks do not share a schema: five of that model's keys name nothing
in this constructor and seven of this one's name nothing in that one, and a merge would leave the
dead half silently dropped by the signature sweep. The price of writing it out is drift, and drift
in a non-encoder key is exactly what destroys the comparison the package exists to make -- a
difference in ``seed``, ``lr``, ``free_bits`` or the spike breaker would be attributed to the
encoder.

So parity is a tested property, in both directions: every leaf outside the encoder schema must
equal the comparison config's value, against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence
means declaring it there. That one comparison is also what pins every correctness setting the
two models share -- precision, compile, the beta warm-up, the spike breaker, the reach budget, the
loaded and normalised fields. Four of the exemptions are mandatory rather than permitted -- the
output directory, the MLflow experiment, the run name and the variant tag are *identity*, and
inheriting them would write these runs into the other model's tree.

Beside parity: both shipped configs resolve, validate with no dead-key warning and build the
encoders they declare through the real driver, every ``VAE_model`` key reaches the constructor
or the task, and the smoke variant's fixture shards exist.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from train.test_utils import make_graph_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"

#: The seven encoder keys. Every one of them varies across a planned architecture arm, which is
#: what makes each a key rather than a constant in the net.
ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)

#: Comparison-model keys that must NOT survive into this config. Each names a piece of the encoder
#: being replaced -- recurrent depth, an extra dilation schedule and its kernel, the conv pre-norm
#: grouping, and the time-pooling normaliser's causalisation switch. None of them means anything
#: here, and the signature sweep would drop each without a word.
REPLACED_ENCODER_KEYS = (
    "lstm_layers",
    "encoder_extra_dilations",
    "encoder_extra_kernel",
    "conv_norm_groups",
    "causal_norm",
)

#: ``VAE_model`` keys that name no constructor argument and are still real: the experiment driver
#: and the task read each of them by name. ``causal_reach_budget_s`` is translated rather than
#: forwarded -- it resolves into the four concrete channel tuples the net takes.
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

#: Leaves outside ``model_config.VAE_model`` that are allowed to differ from the comparison
#: config, with the reason. Anything else differing is drift, and drift is a confound.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "general_config.lr_warmup_steps": "the step-granular ramp this model adds; absent there",
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: The four exemptions that are mandatory rather than merely permitted: copying any of them writes
#: this model's runs into the comparison model's output tree and MLflow experiment.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)


def _leaves(node: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dict leaf of a config mapping.

    Lists are leaves: a config list is a value (device ids, shard paths, kernels), never a
    namespace, so descending into one would compare positions rather than settings.

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


def _get(config: dict, dotted: str) -> Any:
    """Return the value at a dotted path; raises ``KeyError`` if it is not there."""
    node: Any = config
    for part in dotted.split("."):
        node = node[part]
    return node


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
def tiny() -> dict:
    return load_config(str(_TINY))


@pytest.fixture
def sibling() -> dict:
    return load_config(str(_SIBLING_CONFIG))


# --------------------------------------------------------------------------------------
# The shipped config loads and everything in it reaches something
# --------------------------------------------------------------------------------------
def test_every_vae_model_key_reaches_the_constructor_or_the_task(shipped):
    """A key that reaches nothing does not raise -- the constructor has a default for everything --
    so the run trains a *different architecture* than its config describes and only a checkpoint
    that will not reload months later reveals it."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


@pytest.mark.parametrize("config_path", [_CONFIG, _TINY], ids=["default", "tiny"])
def test_a_shipped_config_validates_and_builds_the_encoders_it_declares(
    config_path, tmp_path, loguru_warnings
):
    """Resolved first, which is the only way a config reaches the experiment driver, then driven
    through the framework's real validator and the real driver's signature sweep. The encoders
    built are the ones the config declares: the target reads the full causal prefix, the source a
    bounded window."""
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    resolved = resolve_config_file(str(config_path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    config = load_config(str(config_path))
    vae = config["model_config"]["VAE_model"]
    model = SeqVaeLagAttnTrfRws(**_model_kwargs_from(config, LagAttnTrfRwsTrainer))
    assert len(model.target_encoder.attention_blocks) == vae["target_attention_blocks"]
    assert len(model.source_encoder.attention_blocks) == vae["source_attention_blocks"]
    assert model.source_encoder.attention_window == vae["source_attention_window"]
    assert model.target_encoder.receptive_field is None


def _model_kwargs_from(config: dict, trainer_cls) -> dict:
    """Run a config through the real driver's signature sweep and return the kwargs.

    Args:
        config: A loaded config mapping.
        trainer_cls: The driver class whose sweep is used.

    Returns:
        The constructor kwargs a launch on this config would build the net from.
    """
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return trainer_cls(config_file_path=str(path))._build_model_kwargs()


# --------------------------------------------------------------------------------------
# Identity, and parity with the model this one is compared against
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(shipped, sibling, path):
    """Copying any of these writes this model's runs into the comparison model's output tree and
    mixes them into its MLflow experiment -- which is unrecoverable after the fact, because both
    runs are then indistinguishable by the only fields anything indexes on."""
    value = str(_get(shipped, path))

    assert value != str(_get(sibling, path))
    # Either spelling of this architecture: the identifiers abbreviate it (`lag_attn_trf_rws`)
    # while the output tree carries the package directory name in full.
    assert "trf" in value or "transformer" in value, (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


def test_every_non_encoder_leaf_equals_the_comparison_configs_value(shipped, sibling):
    """The whole comparison rests on this. A difference in ``seed``, ``lr``, ``free_bits``, the
    coverage floor, the likelihood or the spike breaker would be attributed to the encoder."""
    encoder_paths = {f"model_config.VAE_model.{key}" for key in ENCODER_KEYS}
    encoder_paths |= {f"model_config.VAE_model.{key}" for key in REPLACED_ENCODER_KEYS}

    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))
    compared = (set(mine) | set(theirs)) - encoder_paths - set(PARITY_EXEMPT_PATHS)

    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(compared)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert drift == {}, (
        f"these leaves differ from the comparison config and are not declared in "
        f"PARITY_EXEMPT_PATHS: {drift}"
    )


def test_every_declared_parity_exemption_is_a_real_divergence(shipped, sibling):
    """The other direction: an exemption for a key that no longer differs is a permission that
    outlived its reason, and the next accidental divergence there would go unreported."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))

    stale = [
        path
        for path in PARITY_EXEMPT_PATHS
        if mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    """This package commits no binary fixtures; both models read the same shards."""
    for path in (
        *tiny["dataset_config"]["vae_train_datasets"],
        *tiny["dataset_config"]["vae_test_datasets"],
        tiny["dataset_config"]["stat_path"],
    ):
        assert (_REPO_ROOT / path).is_file(), (
            f"{path} is missing; the committed fixture shards moved or were deleted"
        )
