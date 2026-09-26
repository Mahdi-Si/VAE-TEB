r"""The shipped configs load, validate, build, reach nothing dead, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the comparison model's, because
the two ``VAE_model`` blocks do not share a schema: several of that model's keys describe the stored
feature blocks this model does not read, and a merge would leave them behind, dropped in silence by
the signature sweep. The price of writing it out is drift, and drift in a key outside the input
block is exactly what destroys the comparison the package exists to make: a difference in ``seed``,
``lr``, ``free_bits``, the coverage floor, the encoder schedule or the spike breaker would be
attributed to the input representation.

So parity is a tested property, in both directions: every leaf outside :data:`INPUT_PATHS` must
equal the comparison config's value, against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence
means declaring it there. The identity exemptions -- output directory, MLflow experiment, run name,
variant tag -- are mandatory rather than permitted, and are asserted to name this model. The smoke
variant is held to the same discipline against ``default.yaml`` through :data:`TINY_DELTA_PATHS`.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from train.test_utils import make_graph_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
#: The model this one is compared against, leaf for leaf. Not the raw-signal model: the encoder is
#: shared with the conv-Transformer one, so that is the config a difference here would be a
#: difference *from*.
_SIBLING_CONFIG = (
    _REPO_ROOT / "teb_vae" / "lag_attn_transformer_rws" / "configs" / "default.yaml"
)

#: The leaves that **are** the change this package makes, and are therefore outside the parity
#: comparison rather than exempted from it: the ``VAE_model`` keys describing stored feature blocks
#: that no longer exist here, the two reach budgets, and the two loader lists that decide what is
#: read off the shard at all. Everything else must match.
INPUT_PATHS = (
    "model_config.VAE_model.c_y",
    "model_config.VAE_model.c_u",
    "model_config.VAE_model.use_up_st",
    "model_config.VAE_model.causal_reach_budget_s",
    # The two reach keys are a pair: the comparison config bounds how far a stored feature reads
    # FORWARD (and this package refuses that key outright, having no stored features), while this
    # one bounds how far the learned front end reaches BACK. Neither is the other's value under a
    # different name, so both belong here rather than in PARITY_EXEMPT_PATHS.
    "model_config.VAE_model.frontend_reach_budget_s",
    "dataset_config.dataloader_config.dataset_kwargs.load_fields",
    "dataset_config.dataloader_config.normalize_fields",
)

#: ``VAE_model`` keys that name no constructor argument and are still real: the experiment driver
#: and the task read each of them by name.
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
)

#: Leaves outside :data:`INPUT_PATHS` that are allowed to differ from the comparison config, with
#: the reason. Anything else differing is drift, and drift is a confound.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: The exemptions that are mandatory rather than merely permitted: copying any of them writes this
#: model's runs into the comparison model's output tree and MLflow experiment.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
#: Declared, so a smoke variant cannot quietly acquire a second delta and stop being a smoke
#: variant of the config it claims to be one of. Note what is *not* here: ``sequence_length``,
#: ``raw_per_step`` and ``warmup_period``, so the smoke fit runs the front ends at their production
#: reach against the production budget rather than at some shrunken schedule.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.lr_warmup_steps",
        # Pinned back to every-epoch in tiny: the train-smoke tests count one figure set per epoch.
        "general_config.plot_frequency",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "model_config.VAE_model.d_model",
        "model_config.VAE_model.d_z",
        "model_config.VAE_model.d_head",
        "model_config.VAE_model.max_lag",
        "model_config.VAE_model.encoder_d_ff",
        "model_config.VAE_model.dropout",
        "model_config.VAE_model.likelihood",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "advanced_config.tracking.mlflow.enabled",
    }
)


def _leaves(node: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dict leaf of a config mapping.

    Lists are leaves: a config list is a value (device ids, shard paths, kernels, field names),
    never a namespace, so descending into one would compare positions rather than settings.

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
# Every shipped config loads, validates and builds, and everything in it reaches something
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("config_path", [_CONFIG, _TINY], ids=["default", "tiny"])
def test_the_config_validates_without_warnings_and_builds_the_model(
    tmp_path, loguru_warnings, config_path
):
    """Resolved first, which is the only way a config reaches the experiment driver, then through
    the framework's real validator and the real driver's signature sweep into the constructor."""
    from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer

    resolved = resolve_config_file(str(config_path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    SeqVaeLagAttnTrfE2E(**_model_kwargs_from(load_config(str(config_path)), LagAttnTrfE2ETrainer))


def test_every_vae_model_key_reaches_the_constructor_or_the_task(shipped):
    """A key that reaches nothing does not raise -- the constructor has a default for everything --
    so the run trains a *different architecture* than its config describes and only a checkpoint
    that will not reload months later reveals it."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfE2E.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


# --------------------------------------------------------------------------------------
# Identity, and parity with the model this one is compared against
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(shipped, sibling, path):
    """Copying any of these writes this model's runs into the comparison model's output tree and
    mixes them into its MLflow experiment -- which is unrecoverable after the fact."""
    value = str(_get(shipped, path))

    assert value != str(_get(sibling, path))
    # Either spelling of this architecture: the identifiers abbreviate it (`lag_attn_trf_e2e`)
    # while the output tree carries the package directory name in full.
    assert "e2e" in value, (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


def test_every_non_input_leaf_equals_the_comparison_configs_value(shipped, sibling):
    """The whole comparison rests on this. A difference in ``seed``, ``lr``, ``free_bits``, the
    coverage floor, the likelihood, the encoder schedule or the spike breaker would be attributed
    to the input representation."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))
    compared = (set(mine) | set(theirs)) - set(INPUT_PATHS) - set(PARITY_EXEMPT_PATHS)

    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(compared)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert drift == {}, (
        f"these leaves differ from the comparison config and are not declared in INPUT_PATHS or "
        f"PARITY_EXEMPT_PATHS: {drift}"
    )


@pytest.mark.parametrize(
    "declared",
    [tuple(PARITY_EXEMPT_PATHS), INPUT_PATHS],
    ids=["parity-exemptions", "input-paths"],
)
def test_every_declared_divergence_is_a_real_one(shipped, sibling, declared):
    """The other direction: a declared divergence for a key that no longer differs is a permission
    that outlived its reason, silently widening the hole the parity check looks through."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))

    stale = [
        path for path in declared if mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


# --------------------------------------------------------------------------------------
# The smoke variant
# --------------------------------------------------------------------------------------
def test_the_tiny_delta_is_exactly_the_declared_key_list(tiny, shipped):
    """Both directions: an undeclared delta is a smoke run that silently stops resembling the
    production one, and a declared delta that is not there is a stale declaration."""
    mine = dict(_leaves(tiny))
    theirs = dict(_leaves(shipped))
    differing = {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert differing == set(TINY_DELTA_PATHS)
