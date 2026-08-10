r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the comparison model's, because
the two ``VAE_model`` blocks do not share a schema: four of that model's keys name nothing in this
constructor -- they describe the stored feature blocks this model does not read -- and a merge would
leave them behind, dropped in silence by the signature sweep. The price of writing it out is drift,
and drift in a key outside the input block is exactly what destroys the comparison the package
exists to make: a difference in ``seed``, ``lr``, ``free_bits``, the coverage floor or the spike
breaker would be attributed to the input representation.

So parity is a tested property, in both directions: every leaf outside :data:`INPUT_PATHS` must
equal the comparison config's value, against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence
means declaring it there. Four of those exemptions are mandatory rather than permitted -- the output
directory, the MLflow experiment, the run name and the variant tag are *identity*, and inheriting
them would write these runs into the other model's tree.

The one thing this file cannot check by comparison is what is **absent**: the front end has no
configuration surface at all, so a front-end key appearing here would reach nothing and the run
would build a different front end from the one its config appears to describe. That gets its own
assertion.
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

#: The fourteen keys that must exist. Nine are checked by ``validate_config``; the other five are
#: read with a bare index in ``GraphModelBase.__init__``, which runs *before* the validator -- so a
#: missing one raises a bare ``KeyError`` rather than the friendly ``ValueError``.
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

#: The leaves that **are** the change this package makes, and are therefore outside the parity
#: comparison rather than exempted from it. Four ``VAE_model`` keys describing stored feature
#: blocks that no longer exist here, and the two loader lists that decide what is read off the
#: shard at all. Everything else must match.
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
#: and the task read each of them by name. Shorter than the comparison config's list by
#: ``causal_reach_budget_s``, which this package refuses outright.
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

#: The four exemptions that are mandatory rather than merely permitted: copying any of them writes
#: this model's runs into the comparison model's output tree and MLflow experiment.
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
        # Pinned back to every-epoch in tiny: the shipped config plots every 5th epoch (a
        # render-cost decision for multi-day runs), while the train-smoke tests count one
        # figure set per epoch of a 1-3 epoch run.
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


def _has(config: dict, dotted: str) -> bool:
    """Whether a dotted path is present, distinguishing an explicit ``None`` from absence."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


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
# The shipped config loads and everything in it reaches something
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

    assert [message for message in loguru_warnings if "config:" in message] == []


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


def test_no_replaced_input_key_survives(shipped, tiny):
    """The four keys that describe the input representation being replaced. The signature sweep
    would drop each without a word, so the entry point refuses them by name -- and they must not be
    here to be refused in the first place."""
    for config in (shipped, tiny):
        present = [
            key
            for key in ("c_y", "c_u", "use_up_st", "causal_reach_budget_s")
            if key in config["model_config"]["VAE_model"]
        ]
        assert present == []


#: The only front-end key there is. Its stage widths are derived from ``d_model`` and its kernels
#: are a module constant, so the *shape* has no configuration surface; what is configurable is the
#: backward reach the shape is checked against, because that is the bound a depth or kernel arm
#: has to move together with the stack.
FRONTEND_KEYS = ("frontend_reach_budget_s",)


def test_the_front_ends_only_configuration_surface_is_its_reach_budget(shipped, tiny):
    """A second front-end key would reach nothing -- the signature sweep drops what names no
    constructor argument without a word -- leaving the run building a front end its config appears
    to describe and does not. Both directions, so the budget key cannot vanish either."""
    for config in (shipped, tiny):
        vae = config["model_config"]["VAE_model"]
        found = [key for key in vae if "frontend" in key or "front_end" in key]
        assert sorted(found) == sorted(FRONTEND_KEYS)


def test_the_front_end_reach_budget_ships_at_the_warmup_ceiling(shipped):
    """$120$ s $= 480$ raw samples $=$ ``warmup_period * raw_per_step``, so the shipped value is
    exactly what the model derived before the key existed -- explicit and sweepable rather than
    implied. A budget may only tighten that ceiling; the constructor refuses a larger one."""
    vae = shipped["model_config"]["VAE_model"]
    assert vae["frontend_reach_budget_s"] == 120.0
    ceiling_samples = vae["warmup_period"] * vae["raw_per_step"]
    assert vae["frontend_reach_budget_s"] * vae["raw_per_step"] / 4.0 == ceiling_samples


def test_the_shipped_config_builds_the_shipped_architecture(shipped):
    """Resolved through the real driver, so the assertion is about what a launch produces."""
    from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer

    model = SeqVaeLagAttnTrfE2E(**_model_kwargs_from(shipped, LagAttnTrfE2ETrainer))

    assert len(model.target_encoder.attention_blocks) == 6
    assert len(model.source_encoder.attention_blocks) == 3
    assert model.source_encoder.attention_window == 16
    # The stem reaches 21 steps = 84 s; the source bound 21 + 3*15 = 66 steps = 264 s, inside the
    # 360 s lag search range, which is what keeps the encoder a local summariser.
    assert model.source_encoder.receptive_field == 66
    assert model.target_encoder.receptive_field is None  # the full causal prefix
    # And the front ends, which is what the config does NOT say: production kernels, production
    # reach, against the budget warmup_period * raw_per_step derives.
    assert model.target_frontend.reach_samples == 322
    assert model.frontend_reach_budget == 480


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


def test_every_declared_input_divergence_is_a_real_one(shipped, sibling):
    """Same argument, for the set that is excluded rather than exempted. An input path that no
    longer differs would be silently widening the hole the parity check looks through."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))

    stale = [
        path
        for path in INPUT_PATHS
        if mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


def test_the_encoder_block_is_pinned_equal_leaf_for_leaf(shipped, sibling):
    """Spelled out rather than left to the sweep above, because it is the sharpest form of what
    this package claims: the experiment is SAME ENCODER, DIFFERENT INPUT, so an encoder key that
    drifted would let an encoder difference masquerade as a result about the input."""
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    for key in (
        "encoder_conv_kernels", "encoder_conv_dilations", "encoder_num_heads", "encoder_d_ff",
        "target_attention_blocks", "source_attention_blocks", "source_attention_window",
    ):
        assert mine[key] == theirs[key], f"{key} drifted from the comparison config"


def test_the_plotting_block_keeps_the_inherited_drivers_spelling(shipped):
    """The callback assembly is inherited and reads this literal. Renaming the block to match this
    package would disable the per-epoch diagnostic figure with no error anywhere."""
    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_precision_is_float32(shipped):
    """Mixed precision would need float32 islands around the log-variances, the closed-form KL and
    the NLL reduction; any of those moving is a difference the comparison would misattribute."""
    assert shipped["advanced_config"]["trainer"]["precision"] == "32-true"


def test_compile_ships_off(shipped):
    """Off in the shipped config, but **live** rather than inert here: the driver honours the key
    (``compile_model_requested``), because only the net's forward is compiled and the objective --
    with the ``kld_active_frac`` indexing that genuinely defeats inductor -- runs eager through
    ``orig_model``. It ships off because inductor may reassociate float arithmetic and ``pred_gap``
    is a $10^{-4}$-relative difference of two block NLLs, not because it cannot be turned on."""
    assert shipped["advanced_config"]["trainer"]["compile"] is False


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_accumulate_grad_batches_is_present_and_the_warm_start_is_off(shipped):
    """``accumulate_grad_batches`` is the first step of the memory escalation, so it must exist as
    a key rather than as a framework default. ``core_model_checkpoint`` stays null: a checkpoint
    from either sibling carries a different ``model_class`` stamp and holds input-adapter tensors
    this architecture does not have, and the front ends are the thing being measured."""
    assert "accumulate_grad_batches" in shipped["general_config"]
    assert shipped["model_config"]["core_model_checkpoint"] is None


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """z is the only route to the decoder; a nonzero beta before the decoder can use the latent is
    the standard route to posterior collapse."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]
    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["end"] == 1.0
    assert schedule["warmup_epochs"] == 50


def test_the_breaker_is_pinned_by_value_rather_than_by_presence(shipped):
    r"""``max(EMA, ema_floor)`` collapses to the floor once the EMA is negative -- and a summed
    480-sample learned-variance NLL goes negative -- so the floor sits far above any reachable loss,
    disabling the relative test outright, while ``additive_margin`` compares against the *raw* EMA
    and keeps working there. This exact configuration has already cost this repository a run."""
    breaker = shipped["advanced_config"]["spike_breaker"]

    assert breaker["enabled"] is True  # the non-finite guard is the point
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9
    assert breaker["additive_margin"] > 0.0  # finite-spike detection; 0 would disable it
    assert breaker["max_consecutive_skips"] > 0  # the deadlock escape hatch


def test_the_clip_threshold_is_marked_provisional_for_this_architecture(shipped):
    """It is inherited from a run of a model with no front end, and the front ends are a new
    gradient path. The value is pinned equal by the parity check; what this asserts is that the
    file says where it came from and what it is re-derived from, since a threshold far below the
    typical norm rescales every step while completing normally."""
    assert shipped["advanced_config"]["trainer"]["gradient_clip_val"] == 5000.0
    text = _CONFIG.read_text(encoding="utf-8")
    assert "PROVISIONAL FOR THIS ARCHITECTURE" in text
    assert "train/grad_norm" in text
    assert "train/grad_clip_frac" in text


def test_both_raw_signals_are_loaded_and_normalized(shipped):
    """The two-line summary of this package's data contract, and both halves are silent when wrong:
    an unnormalized ``fhr`` makes the Gaussian NLL meaningless, an unnormalized ``up`` shifts every
    coupling number. Neither front end standardises anything itself."""
    dataloader = shipped["dataset_config"]["dataloader_config"]
    for field in ("fhr", "up"):
        assert field in dataloader["normalize_fields"]
        assert field in dataloader["dataset_kwargs"]["load_fields"]


def test_no_stored_feature_block_is_loaded(shipped):
    """The read this package exists to stop making. ``fhr_up_ph`` stays absent for the reason it
    always was: a coefficient mixing both signals would destroy the target-only /
    source-conditioned separation the design rests on."""
    dataloader = shipped["dataset_config"]["dataloader_config"]
    for field in ("fhr_st", "fhr_ph", "up_st", "up_ph", "fhr_up_ph"):
        assert field not in dataloader["dataset_kwargs"]["load_fields"]
        assert field not in dataloader["normalize_fields"]


def test_the_cross_channel_block_appears_in_no_config():
    """A single appearance anywhere in a config -- load_fields, normalize_fields, a comment someone
    uncommented -- is a defect."""
    for path in (_CONFIG, _TINY):
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


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
    assert len(lines) < 36, (
        f"tiny.yaml carries {len(lines)} non-comment lines; it should be deltas only"
    )


def test_the_base_key_never_reaches_the_validator(tiny):
    """``load_config`` consumes ``base:``; were it left in, ``validate_config`` would warn on an
    unknown key and the MLflow param dump would carry a loader directive."""
    assert "base" not in tiny


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


def test_the_tiny_variant_inherits_the_settings_it_does_not_name(tiny, shipped):
    """Including every correctness requirement, which is why it must not re-state them."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["advanced_config"]["trainer"]["precision"] == "32-true"
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    assert (
        tiny["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
        == shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    )


def test_the_tiny_geometry_stays_real_so_the_front_end_runs_at_production_reach(tiny):
    """The committed shard carries the real field shapes, and two things are only exercised at the
    real trim: the raw-index geometry (the forecast of anchor $t$ starts at $16(t+1)$) and the front
    ends' own 16-samples-per-token stride. ``warmup_period`` in particular is *not* shrunk, because
    it is also the front ends' reach budget -- a smaller one would build a different, narrower front
    end than the production run's and the smoke fit would exercise a stack nobody ships."""
    vae = tiny["model_config"]["VAE_model"]

    assert vae["sequence_length"] == 300
    assert vae["raw_per_step"] == 16
    assert vae["warmup_period"] == 30
    assert vae["horizon"] == 30


def test_the_tiny_variant_overrides_what_a_local_smoke_run_needs(tiny):
    assert tiny["general_config"]["epochs"] == 1
    assert tiny["general_config"]["cuda_devices"] == [0]
    assert tiny["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
    assert tiny["dataset_config"]["dataloader_config"]["num_workers"] == 0
    # The deliberate delta: mse starves the decoder logvar heads, so the smoke run exercises the
    # configuration whose DDP strategy is the fallback.
    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"


def test_the_tiny_variant_exercises_the_step_warmup_inside_a_smoke_fit(tiny):
    """A ramp longer than the fit would leave the step-granular path configured and unexercised,
    which is exactly the failure a smoke run is for. Two optimizer steps per epoch against the
    four-sample shard, so a handful of steps completes within a couple of epochs."""
    warmup_steps = tiny["general_config"]["lr_warmup_steps"]

    assert warmup_steps > 0
    assert warmup_steps <= 8


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny):
    """Caught here rather than as a ``ValueError`` three steps into the smoke run. The last two are
    this architecture's own: rotary position encoding rotates coordinate pairs, so the derived
    encoder head width must be even; and the front end's $(d/4, d/2, 3d/4, d)$ stage widths need
    ``d_model`` divisible by four."""
    vae = tiny["model_config"]["VAE_model"]
    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior
    assert (vae["d_model"] // vae["encoder_num_heads"]) % 2 == 0
    assert vae["d_model"] % 4 == 0


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    """This package commits no binary fixtures; all three models read the same shards."""
    for path in (
        *tiny["dataset_config"]["vae_train_datasets"],
        *tiny["dataset_config"]["vae_test_datasets"],
        tiny["dataset_config"]["stat_path"],
    ):
        assert (_REPO_ROOT / path).is_file(), (
            f"{path} is missing; the committed fixture shards moved or were deleted"
        )


def test_the_resolved_tiny_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver. In the fast
    tier deliberately: a broken smoke config would otherwise stay invisible in the default gate."""
    from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnTrfE2E(
        **_model_kwargs_from(load_config(str(_TINY)), LagAttnTrfE2ETrainer)
    )
    # The smoke model's front ends are the production ones at a narrower width: the kernels and the
    # reach are the shipped constants, only the (d/4, d/2, 3d/4, d) widths shrink with d_model.
    assert model.target_frontend.reach_samples == 322
    assert model.target_frontend.stage_modules[-1].out_channels == 32
