r"""The shipped configs load, validate, and contain nothing that reaches nothing.

A config key is only real if some code reads it. The tree this model was ported from accumulated
keys that nothing consumed -- a plotting block whose ``enabled: false`` disabled nothing, a
``checkpoint_frequency`` no module read, a spike-breaker ``warn_on_skip`` with no target -- and each
one reads to a maintainer as a control that exists. These tests hold the line: every key is checked
against the framework's own validator, and the keys deliberately dropped in the port are asserted
absent so they cannot drift back in.

``validate_config`` is not a schema (only nine keys are required, unknown keys merely warn), so a
green validator is necessary and not sufficient. The unknown-key assertion below is what makes it
close to sufficient for ``advanced_config``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from train.test_utils import make_graph_model

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_V3 = _CONFIG_DIR / "v3.yaml"
_V3_TINY = _CONFIG_DIR / "v3_tiny.yaml"

#: The fourteen keys that must exist. Nine are checked by ``validate_config``; the other five are
#: read with a bare index in ``GraphModelBase.__init__``, which runs *before* the validator -- so a
#: missing one raises a bare ``KeyError`` from the constructor rather than the friendly ``ValueError``
#: naming it. Both sets are equally required; only the error message differs.
_REQUIRED_PATHS = (
    "general_config.tag",
    "general_config.cuda_devices",
    "general_config.epochs",
    "general_config.lr",
    "general_config.batch_size",
    "general_config.folders_config",
    "advanced_config",
    "advanced_config.trainer",
    "general_config.lr_milestone",
    "general_config.plot_frequency",
    "general_config.folders_config.out_dir_base",
    "general_config.batch_size.train",
    "general_config.batch_size.test",
    "general_config.seed",
)

#: Keys dropped in the port, each because nothing read it. Asserted absent so a copy-paste from the
#: old config cannot quietly restore a control that does not control anything.
_DROPPED_PATHS = (
    "general_config.checkpoint_frequency",         # nothing under train/ reads it
    "advanced_config.memory",                      # denylisted dead block
    "advanced_config.callbacks.loss_plotting",     # max_history_size reaches nothing
    "advanced_config.callbacks.reconstruction_plotting",
    "advanced_config.callbacks.comprehensive_plotting",
    "advanced_config.spike_breaker.warn_on_skip",  # no framework target; silently ignored
    "advanced_config.spike_breaker.ema_momentum",  # renamed to ema_decay by the framework
    "model_config.VAE_model.loss_spike_skip",      # moved to advanced_config.spike_breaker
    "model_config.VAE_model.logvar_bound",         # smooth bounding is unconditional
    "model_config.VAE_model.posterior_logvar",     # a residual posterior logvar is unconditional
    "model_config.VAE_model.latent_stats_momentum",  # the latent-stats mechanism no longer exists
    "model_config.VAE_model.horizon_refine.attn",  # reaches nothing
    "model_config.VAE_model.encoder.temporal_backbone",
    "model_config.warm_start_from",                # the v1 weight bridge does not exist here
    "model_config.classifier",                     # a different model's config
    "dataset_config.kfold_base_path",
    "dataset_config.num_folds",
    "dataset_config.classifier_train_datasets",
)


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib ``warnings``
    module, so a ``pytest.warns`` or ``caplog`` assertion against it would pass no matter what the
    config contained.
    """
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink_id)


def _has(config, dotted) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


@pytest.fixture
def v3():
    return load_config(str(_V3))


@pytest.fixture
def tiny():
    return load_config(str(_V3_TINY))


# --------------------------------------------------------------------------------------
# The shipped config
# --------------------------------------------------------------------------------------
def test_every_effectively_required_key_is_present(v3):
    missing = [path for path in _REQUIRED_PATHS if not _has(v3, path)]
    assert missing == []


def test_the_shipped_config_validates_with_no_unknown_or_dead_key_warnings(tmp_path, loguru_warnings):
    """Drives the framework's real validator, not a copy of its rules.

    ``validate_config`` warns rather than raises on an unknown or dead key, so the warning list --
    not the absence of an exception -- is the assertion that matters. This is what makes "no key in
    this file reaches nothing" enforceable for ``advanced_config`` rather than aspirational.
    """
    graph_model = make_graph_model(_V3, **{"general_config.folders_config.out_dir_base": str(tmp_path)})

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []


def test_no_dropped_key_has_crept_back(v3):
    present = [path for path in _DROPPED_PATHS if _has(v3, path)]
    assert present == []


def test_compile_is_off_and_stays_off(v3):
    """Not a preference. LSTM, checkpointed attention and data-dependent mask indexing each break
    inductor independently, so this is a correctness setting with a comment to match."""
    assert v3["advanced_config"]["trainer"]["compile"] is False


def test_causal_norm_is_on(v3):
    """With it off the prior conditions on the future and the reported KL is not a TE surrogate."""
    assert v3["model_config"]["VAE_model"]["causal_norm"] is True


def test_the_breaker_is_configured_as_a_non_finite_guard_only(v3):
    r"""The relative test cannot work on a loss that goes negative, so it is switched off.

    ``max(EMA, ema_floor)`` collapses to the floor once the EMA is negative, and at a $0.0$ floor
    that makes every positive batch a spike -- discarding its gradient and logging the EMA in place
    of the real loss. A floor far above any reachable loss disables the relative test outright,
    while the non-finite guard, which never consults the threshold, keeps a NaN from reaching the
    weights.
    """
    breaker = v3["advanced_config"]["spike_breaker"]

    assert breaker["enabled"] is True  # the non-finite guard is the point
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9
    assert breaker["ema_decay"] == 0.02  # the framework's spelling of the old ema_momentum


def test_the_shipped_config_earns_plain_ddp(v3):
    """The three flags whose combination decides the strategy.

    Learned observation variance keeps the decoder logvar heads in DDP's expectation set, and
    freezing the starved attention projection removes the one parameter that head-structured
    latents leave without a gradient. Together they are what make find_unused_parameters
    unnecessary; the strategy selector asserts the consequence.
    """
    vae = v3["model_config"]["VAE_model"]
    assert vae["likelihood"] == "gaussian_nll"
    assert vae["sigma_obs"] == "learned"
    assert vae["head_structured_latent"] is True
    assert vae["freeze_unused_attn_proj"] is True


def test_mlflow_registers_the_final_model_without_double_storing_weights(v3):
    """``log_model`` is not dead and is not a synonym for ``log_checkpoints``.

    ``log_model`` drives the run-logging callback the trainer builder attaches, which logs and
    registers the final eager model. ``log_checkpoints`` drives Lightning's own per-checkpoint
    weight artifacts and falls back to ``log_model`` when absent -- so omitting it while
    ``log_model: true`` silently turns checkpoint uploading on and stores the weights twice.
    """
    mlflow = v3["advanced_config"]["tracking"]["mlflow"]
    assert mlflow["log_model"] is True
    assert mlflow["log_checkpoints"] is False


def test_load_fields_covers_what_the_model_and_plots_read_and_nothing_else(v3):
    load_fields = set(
        v3["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    )
    # The five the model consumes: two target streams, two source streams, and the validity mask.
    assert {"fhr_st", "fhr_ph", "up_st", "up_ph", "weight"} <= load_fields
    # The three the diagnostic plots need.
    assert {"fhr", "up", "guid"} <= load_fields
    # Classifier-era fields this model never reads.
    assert load_fields.isdisjoint({"target", "epoch", "cs_label", "bg_label"})


def test_the_source_stream_width_agrees_with_the_ablation_toggle(v3):
    """The constructor rejects a mismatch, but it does so after a config has already shipped."""
    vae = v3["model_config"]["VAE_model"]
    assert vae["c_u"] == (101 if vae["use_up_st"] else 58)


def test_lr_milestone_is_singular(v3):
    """The experiment driver reads the singular key with a bare index.

    The Lightning module's constructor argument is the plural ``lr_milestones``; the config key is
    singular. A config that spelled it plural would raise ``KeyError`` from the constructor.
    """
    assert "lr_milestone" in v3["general_config"]
    assert "lr_milestones" not in v3["general_config"]


# --------------------------------------------------------------------------------------
# The smoke variant
# --------------------------------------------------------------------------------------
def test_the_tiny_variant_names_only_its_deltas():
    """The whole point of the base: mechanism.

    Counted on the raw file, not the resolved config: a variant that re-stated the inherited
    settings would be a copy that drifts, which is what a subclass-per-variant tree already proved.
    """
    lines = [
        line for line in _V3_TINY.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) < 30, f"v3_tiny.yaml carries {len(lines)} non-comment lines; it should be deltas only"


def test_the_tiny_variant_inherits_the_settings_it_does_not_name(tiny, v3):
    """Including every correctness requirement, which is why it must not re-state them."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["model_config"]["VAE_model"]["causal_norm"] is True
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    # The real geometry: the committed shard carries the real field shapes.
    assert tiny["model_config"]["VAE_model"]["sequence_length"] == 300
    assert tiny["model_config"]["VAE_model"]["c_y"] == v3["model_config"]["VAE_model"]["c_y"]
    assert tiny["model_config"]["VAE_model"]["c_u"] == v3["model_config"]["VAE_model"]["c_u"]
    assert (
        tiny["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
        == v3["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    )


def test_the_tiny_variant_overrides_what_a_local_smoke_run_needs(tiny):
    assert tiny["general_config"]["epochs"] == 1
    assert tiny["general_config"]["cuda_devices"] == [0]
    assert tiny["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
    assert tiny["dataset_config"]["dataloader_config"]["num_workers"] == 0


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny):
    """Caught here rather than as a ValueError three steps into the smoke run."""
    vae = tiny["model_config"]["VAE_model"]
    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by head_structured_latent


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    for path in (
        *tiny["dataset_config"]["vae_train_datasets"],
        *tiny["dataset_config"]["vae_test_datasets"],
        tiny["dataset_config"]["stat_path"],
    ):
        assert (Path(__file__).resolve().parents[3] / path).is_file(), (
            f"{path} is missing; regenerate it with scripts/make_tiny_shard.py"
        )


def test_the_resolved_tiny_variant_validates(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver.

    The driver reads a config path and does not know about ``base:``; handed the raw file it would
    see a config missing almost every required key. Resolve-then-write is the seam.
    """
    resolved = resolve_config_file(str(_V3_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []
