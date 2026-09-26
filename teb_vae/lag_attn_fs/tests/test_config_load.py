r"""The shipped configs load, validate, build, reach something, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the comparison model's, and the
price of that is drift -- drift in a key that has nothing to do with the target domain is exactly
what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``causal_reach_budget_s``, the coverage floor or the spike breaker would be
attributed to the target domain by every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison
config's value against :data:`PARITY_EXEMPT_PATHS`, and every declared exemption must be a real
divergence. The comparison is total rather than schema-limited because this model's net is a
*subclass* whose constructor schema is unchanged: every key means the same thing in both files.

Beside parity: every config in the directory resolves, validates without warnings and builds a
model whose decoder is as wide as the reach budget keeps; every config loads and normalises both
target blocks and never names the cross-channel block; the dev-box variant differs from the
shipped config only where it declares; and every ``VAE_model`` key reaches the constructor or the
task.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from train.test_utils import make_graph_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"

#: Every config this package ships, sweep arms included. Globbed rather than listed: a config added
#: later is exactly the one nobody would think to add to a list here.
_SHIPPED_CONFIGS = sorted(_CONFIG_DIR.glob("*.yaml"))

#: ``VAE_model`` keys that name no constructor argument and are still real: the experiment driver
#: and the task read each of them by name. ``causal_reach_budget_s`` is translated rather than
#: forwarded -- it resolves into the four concrete channel tuples the net takes, and here those
#: tuples also decide the decoder's width.
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

#: Why the three auxiliary shape weights are the one *model* divergence from the comparison config.
#: Written once and shared by the three entries below, because it is one reason rather than three.
AUX_OFF_REASON = (
    "the shape terms read the forecast block's last axis as consecutive raw samples -- pooling it, "
    "differencing it, and joining its first entry to the anchor's last observed sample -- and this "
    "block's last axis is the surviving target channels, an unordered index with no metric; the "
    "weights ship at 0.0 so the columns are honest zeros rather than raw-domain formulas"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else
#: differing is drift, and drift is a confound. No wildcards: the list names its contents so that
#: adding an entry is a decision rather than an omission.
#:
#: Five are identity, one is a retuned number, and three are a term that does not exist here.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.spike_breaker.additive_margin": (
        "re-derived at this loss scale: the margin is stated in nats of the summed block, and this "
        "block is wider than the raw model's"
    ),
    "model_config.VAE_model.lambda_ms": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_deriv": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_boundary": AUX_OFF_REASON,
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: The exemptions that exist because a *number* had to move, as opposed to a name. The margin is
#: stated in nats of the summed block, so a larger block moves it upwards.
RETUNED_PATHS = ("advanced_config.spike_breaker.additive_margin",)

#: Every leaf of ``smoke_hie.yaml`` that resolves to something other than ``default.yaml``'s value.
#: This variant is the one whose numbers get quoted, so a delta it acquired quietly would be a
#: difference between the model that was read and the model that ships. Only the run's *scale* is
#: local; the one model-block entry is the beta ramp's length, which cannot be inherited because the
#: variant's epoch budget is a small fraction of a production run's.
SMOKE_HIE_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.plot_frequency",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "model_config.VAE_model.beta_schedule.warmup_epochs",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "dataset_config.dataloader_config.prefetch_factor",
        "advanced_config.tracking.mlflow.enabled",
    }
)


def _leaves(node: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every non-dict leaf of a config mapping.

    Lists are leaves: a config list is a value (device ids, shard paths, dilations), never a
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
def smoke_hie() -> dict:
    return load_config(str(_SMOKE_HIE))


@pytest.fixture
def sibling() -> dict:
    return load_config(str(_SIBLING_CONFIG))


# --------------------------------------------------------------------------------------
# Every shipped config loads, validates, builds, and everything in it reaches something
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", _SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_config_validates_and_builds_a_decoder_as_wide_as_the_budget_keeps(
    path, tmp_path, loguru_warnings
):
    """Resolved first, which is the only way a config ever reaches the experiment driver, then
    validated by the framework's real validator and built through the real driver's sweep. The
    reach budget decides the surviving channels and the survivors decide the decoder width, so the
    width is asserted against the resolved keep-index rather than against a number."""
    resolved = resolve_config_file(str(path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    config = load_config(str(path))
    kwargs = _model_kwargs_from(config, LagAttnFsTrainer)
    model = SeqVaeLagAttnFs(**kwargs)
    # The sweep forwarded the config rather than building an all-defaults model.
    assert model.d_model == config["model_config"]["VAE_model"]["d_model"]
    expected_width = len(kwargs.get("target_keep_index") or range(model.c_y))
    assert model.decoder_out_channels == model.decoder.out_channels == expected_width


def test_every_vae_model_key_reaches_the_constructor_or_the_task(shipped):
    """A key that reaches nothing does not raise -- the constructor has a default for everything --
    so the run trains a *different architecture* than its config describes, and only a checkpoint
    that will not reload months later reveals it."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnFs.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


@pytest.mark.parametrize("path", _SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_config_loads_and_normalizes_both_target_blocks_and_no_cross_channel_block(path):
    """The target is the concatenation of both blocks, so a config carrying one of them is a target
    with a hole in it, and an unnormalised block makes the Gaussian NLL meaningless with nothing
    else raising. ``fhr_up_ph`` mixes both signals in one coefficient: loading it would break the
    target-only / source-conditioned separation and put the source's own signal into the target."""
    dataloader = load_config(str(path))["dataset_config"]["dataloader_config"]
    load_fields = dataloader["dataset_kwargs"]["load_fields"]
    normalize_fields = dataloader["normalize_fields"]

    for field in LagAttnFsTrainer.TARGET_FIELDS:
        assert field in normalize_fields, field
        assert field in load_fields, field
    assert "fhr_up_ph" not in load_fields
    assert "fhr_up_ph" not in normalize_fields


# --------------------------------------------------------------------------------------
# Parity with the model this one is compared against
# --------------------------------------------------------------------------------------
def test_every_leaf_equals_the_comparison_configs_value(shipped, sibling):
    """The whole comparison rests on this, and here it is total rather than schema-limited: the net
    is a subclass whose constructor schema is unchanged, so every key means the same thing in both
    files and every key is comparable."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))
    compared = (set(mine) | set(theirs)) - set(PARITY_EXEMPT_PATHS)

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


@pytest.mark.parametrize("path", RETUNED_PATHS)
def test_each_retuned_value_is_larger_than_the_one_it_replaces(shipped, sibling, path):
    """It moved because this objective sums a larger block, so it moved *upwards*. A retune that
    landed below the comparison model's value would be a sign the scale argument had been applied in
    the wrong direction."""
    assert float(_get(shipped, path)) > float(_get(sibling, path))


# --------------------------------------------------------------------------------------
# The committed-shard variants
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", [_TINY, _SMOKE_HIE], ids=lambda path: path.name)
def test_the_committed_shard_variants_point_at_files_that_exist(path):
    """Both variants read committed shards and statistics; a moved or deleted file would otherwise
    surface only minutes into a smoke run."""
    dataset = load_config(str(path))["dataset_config"]
    for relative in (
        *dataset["vae_train_datasets"],
        *dataset["vae_test_datasets"],
        dataset["stat_path"],
    ):
        assert (_REPO_ROOT / relative).is_file(), f"{relative} is missing"


def test_the_local_delta_is_exactly_the_declared_key_list(smoke_hie, shipped):
    """Both directions: this is the variant whose numbers get quoted, so an undeclared delta is a
    difference between the model that was read and the model that ships, attributed to neither."""
    mine = dict(_leaves(smoke_hie))
    theirs = dict(_leaves(shipped))
    differing = {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert differing == set(SMOKE_HIE_DELTA_PATHS)


def test_the_local_variant_ramps_beta_inside_its_own_epoch_budget(smoke_hie):
    """The one model-block delta. Beta must reach its endpoint early enough that the columns are
    read off a model that has been paying its rate for most of the run."""
    warmup = smoke_hie["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
    epochs = smoke_hie["general_config"]["epochs"]

    assert warmup * 10 <= epochs
