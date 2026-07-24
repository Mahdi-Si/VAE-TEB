r"""The shipped configs load, validate, and contain nothing that reaches nothing.

A config key is only real if some code reads it. Two families of keys are asserted *absent*
here, for different reasons. The sibling-model keys this architecture has no code for
(``lambda_perm`` as a loss weight, ``lag_smoothness_lambda``, ``detach_baseline_in_full``,
``kld_support``) must not drift back in from the config this one was derived from. And the knobs
this net made unconditional (``sigma_obs``, ``head_structured_latent``,
``freeze_unused_attn_proj``) must not reappear as keys either -- each would read to a maintainer
as a control that exists, when the learned observation variance, the head-structured posterior
and the frozen attention projection are structural facts of the model, not choices.

``validate_config`` is not a schema (only nine keys are required, unknown keys merely warn), so
a green validator is necessary and not sufficient; the warning-list assertion is what closes the
gap for ``advanced_config``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from train.test_utils import make_graph_model

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"

#: The fourteen keys that must exist. Nine are checked by ``validate_config``; the other five
#: are read with a bare index in ``GraphModelBase.__init__``, which runs *before* the validator
#: -- so a missing one raises a bare ``KeyError`` rather than the friendly ``ValueError``.
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

#: Keys that must stay out. The first group is the sibling's mechanisms this architecture
#: deliberately does not have; the second is knobs this net's constructor made unconditional.
_FORBIDDEN_PATHS = (
    "model_config.VAE_model.lambda_perm",
    "model_config.VAE_model.perm_every_n_batches",
    "model_config.VAE_model.lag_smoothness_lambda",
    "model_config.VAE_model.detach_baseline_in_full",
    "model_config.VAE_model.kld_support",
    "model_config.VAE_model.sigma_obs",
    "model_config.VAE_model.head_structured_latent",
    "model_config.VAE_model.freeze_unused_attn_proj",
    "model_config.VAE_model.lambda_lag",
    "model_config.VAE_model.horizon_refine",
    "model_config.VAE_model.encoder",
)


@pytest.fixture
def loguru_warnings():
    """Collect the validator's warnings.

    ``validate_config`` reports an unknown or dead key through loguru, not the stdlib
    ``warnings`` module, so a ``pytest.warns`` assertion against it would pass no matter what
    the config contained.
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
def shipped():
    return load_config(str(_CONFIG))


@pytest.fixture
def tiny():
    return load_config(str(_TINY))


# --------------------------------------------------------------------------------------
# The shipped config
# --------------------------------------------------------------------------------------
def test_every_effectively_required_key_is_present(shipped):
    missing = [path for path in _REQUIRED_PATHS if not _has(shipped, path)]
    assert missing == []


def test_the_shipped_config_validates_with_no_unknown_or_dead_key_warnings(
    tmp_path, loguru_warnings
):
    """Drives the framework's real validator, not a copy of its rules."""
    graph_model = make_graph_model(
        _CONFIG, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []


def test_no_sibling_only_or_unconditional_key_is_present(shipped, tiny):
    """Both directions of the key-hygiene rule; see the module docstring."""
    for config in (shipped, tiny):
        present = [path for path in _FORBIDDEN_PATHS if _has(config, path)]
        assert present == []


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient; a single appearance anywhere in a
    config -- load_fields, normalize_fields, a comment someone uncommented -- is a defect."""
    for path in (_CONFIG, _TINY):
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_compile_is_off_and_stays_off(shipped):
    """Not a preference: LSTM, checkpointed attention and data-dependent mask indexing each
    break inductor independently."""
    assert shipped["advanced_config"]["trainer"]["compile"] is False


def test_causal_norm_is_on(shipped):
    """With it off the prior conditions on the future and the KL is not a coupling readout."""
    assert shipped["model_config"]["VAE_model"]["causal_norm"] is True


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch
    number against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_the_sampler_is_lightnings(shipped):
    assert shipped["advanced_config"]["trainer"]["use_distributed_sampler"] is True


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """z is the only route to the decoder; a nonzero beta before the decoder can use the
    latent is the standard route to posterior collapse."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]
    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["end"] == 1.0


def test_the_breaker_is_a_non_finite_guard_plus_the_additive_test(shipped):
    r"""The relative test stays off; the additive test is what detects a finite blow-up.

    ``max(EMA, ema_floor)`` collapses to the floor once the EMA is negative -- and a summed
    480-sample learned-variance NLL goes negative harder than the sibling's loss ever did -- so
    the floor sits far above any reachable loss, disabling the relative test outright, while
    ``additive_margin`` compares against the *raw* EMA and keeps working there.
    """
    breaker = shipped["advanced_config"]["spike_breaker"]

    assert breaker["enabled"] is True  # the non-finite guard is the point
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9
    assert breaker["additive_margin"] > 0.0  # finite-spike detection; 0 would disable it
    assert breaker["max_consecutive_skips"] > 0  # the deadlock escape hatch


def test_the_reach_budget_key_exists_and_ships_null(shipped):
    """The config axis for the causal input delay: null = all channels, no delay -- the clean
    architecture comparison against the sibling. Presence is asserted (not merely non-error)
    because ``null`` and *absent* look identical to a ``.get``."""
    vae = shipped["model_config"]["VAE_model"]
    assert "causal_reach_budget_s" in vae
    assert vae["causal_reach_budget_s"] is None


def test_load_fields_covers_what_the_model_and_plots_read(shipped):
    load_fields = set(
        shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]
    )
    # The six the model consumes: the raw target, two target streams, two source streams, and
    # the validity mask.
    assert {"fhr", "fhr_st", "fhr_ph", "up_st", "up_ph", "weight"} <= load_fields
    # The two the diagnostic plots need.
    assert {"up", "guid"} <= load_fields
    # Classifier-era fields this model never reads.
    assert load_fields.isdisjoint({"target", "epoch", "cs_label", "bg_label"})


def test_the_raw_target_is_normalized(shipped):
    """'fhr' in normalize_fields is a correctness requirement: the raw target must be z-scored
    or the Gaussian NLL is meaningless, and nothing raises on its own."""
    assert "fhr" in shipped["dataset_config"]["dataloader_config"]["normalize_fields"]


def test_lr_milestone_is_singular(shipped):
    """The experiment driver reads the singular key with a bare index."""
    assert "lr_milestone" in shipped["general_config"]
    assert "lr_milestones" not in shipped["general_config"]


# --------------------------------------------------------------------------------------
# The smoke variant
# --------------------------------------------------------------------------------------
def test_the_tiny_variant_names_only_its_deltas():
    """The whole point of the ``base:`` mechanism, counted on the raw file: a variant that
    re-stated the inherited settings would be a copy that drifts."""
    lines = [
        line
        for line in _TINY.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) < 32, (
        f"tiny.yaml carries {len(lines)} non-comment lines; it should be deltas only"
    )


def test_the_base_key_never_reaches_the_validator(tiny):
    """``load_config`` consumes ``base:``; were it left in, ``validate_config`` would warn on
    an unknown key and the MLflow param dump would carry a loader directive."""
    assert "base" not in tiny


def test_the_tiny_variant_inherits_the_settings_it_does_not_name(tiny, shipped):
    """Including every correctness requirement, which is why it must not re-state them."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["model_config"]["VAE_model"]["causal_norm"] is True
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    # The real geometry: the committed shard carries the real field shapes.
    assert tiny["model_config"]["VAE_model"]["sequence_length"] == 300
    assert tiny["model_config"]["VAE_model"]["raw_per_step"] == 16
    assert tiny["model_config"]["VAE_model"]["c_y"] == shipped["model_config"]["VAE_model"]["c_y"]
    assert tiny["model_config"]["VAE_model"]["c_u"] == shipped["model_config"]["VAE_model"]["c_u"]
    assert (
        tiny["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
        == shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    )


def test_the_tiny_variant_overrides_what_a_local_smoke_run_needs(tiny):
    assert tiny["general_config"]["epochs"] == 1
    assert tiny["general_config"]["cuda_devices"] == [0]
    assert tiny["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
    assert tiny["dataset_config"]["dataloader_config"]["num_workers"] == 0
    # The deliberate delta: mse starves the decoder logvar heads, so the smoke run exercises
    # the configuration whose DDP strategy is the fallback.
    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny):
    """Caught here rather than as a ValueError three steps into the smoke run."""
    vae = tiny["model_config"]["VAE_model"]
    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    for path in (
        *tiny["dataset_config"]["vae_train_datasets"],
        *tiny["dataset_config"]["vae_test_datasets"],
        tiny["dataset_config"]["stat_path"],
    ):
        assert (Path(__file__).resolve().parents[3] / path).is_file(), (
            f"{path} is missing; the committed fixture shards moved or were deleted"
        )


def test_the_resolved_tiny_variant_validates(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver."""
    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [m for m in loguru_warnings if "config:" in m] == []
