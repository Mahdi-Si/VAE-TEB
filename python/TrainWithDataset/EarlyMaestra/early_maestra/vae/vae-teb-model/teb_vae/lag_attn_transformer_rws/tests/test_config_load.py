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
means declaring it there. Four of those exemptions are mandatory rather than permitted -- the
output directory, the MLflow experiment, the run name and the variant tag are *identity*, and
inheriting them would write these runs into the other model's tree.
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

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
#: Declared, so a smoke variant cannot quietly acquire a second delta and stop being a smoke
#: variant of the config it claims to be one of.
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
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


def test_the_seven_encoder_keys_are_present_and_reach_the_constructor(shipped):
    """Each varies across a planned arm, so an arm that could not change it would not be an arm."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfRws.__init__).parameters)
    vae = shipped["model_config"]["VAE_model"]

    for key in ENCODER_KEYS:
        assert key in vae, f"{key} is missing from the shipped config"
        assert key in constructor_keys, f"{key} names no constructor argument"


def test_no_replaced_encoder_key_survives(shipped, tiny):
    """The five keys that describe the encoder being replaced. Each would be dropped in silence."""
    for config in (shipped, tiny):
        present = [key for key in REPLACED_ENCODER_KEYS if key in config["model_config"]["VAE_model"]]
        assert present == []


def test_the_shipped_config_builds_the_shipped_architecture(shipped):
    """Resolved through the real driver, so the assertion is about what a launch produces."""
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    model = SeqVaeLagAttnTrfRws(**_model_kwargs_from(shipped, LagAttnTrfRwsTrainer))

    assert len(model.target_encoder.attention_blocks) == 4
    assert len(model.source_encoder.attention_blocks) == 3
    assert model.source_encoder.attention_window == 16
    # The stem reaches 21 steps = 84 s; the source bound 21 + 3*15 = 66 steps = 264 s, inside the
    # 360 s lag search range, which is what keeps the encoder a local summariser.
    assert model.source_encoder.receptive_field == 66
    assert model.target_encoder.receptive_field is None  # the full causal prefix


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
    """Off in the shipped config, but **live** rather than inert: with the recurrence gone this
    driver honours the key (``compile_model_requested``), because only the net's forward is
    compiled and the objective -- with the ``kld_active_frac`` indexing that genuinely defeats
    inductor -- runs eager through ``orig_model``. It ships off because inductor may reassociate
    float arithmetic and ``pred_gap`` is a $10^{-4}$-relative difference of two block NLLs."""
    assert shipped["advanced_config"]["trainer"]["compile"] is False


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_accumulate_grad_batches_is_present_and_the_warm_start_is_off(shipped):
    """``accumulate_grad_batches`` is the first step of the memory escalation, so it must exist as
    a key rather than as a framework default. ``core_model_checkpoint`` stays null: a checkpoint
    from the comparison model carries a different ``model_class`` stamp and different encoder
    tensors, and the guard would refuse it -- correctly."""
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


def test_the_reach_budget_key_exists_and_ships_the_guard_on(shipped):
    """The config axis for the causal input delay. It ships at $120$ s rather than ``null``:
    the stored features are two-sided, so at ``null`` the target branch reads up to $974$ s of its
    own future and $D_{\\mathrm{base}}$ is measured through that leak. Presence is asserted (not
    merely non-error) because ``null`` and *absent* look identical to a ``.get``, and the value is
    pinned because a silent revert to ``null`` would put the leak back with nothing failing."""
    vae = shipped["model_config"]["VAE_model"]
    assert "causal_reach_budget_s" in vae
    assert vae["causal_reach_budget_s"] == 120


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient; a single appearance anywhere in a
    config -- load_fields, normalize_fields, a comment someone uncommented -- is a defect."""
    for path in (_CONFIG, _TINY):
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_the_raw_target_is_loaded_and_normalized(shipped):
    """'fhr' in normalize_fields is a correctness requirement: the raw target must be z-scored or
    the Gaussian NLL is meaningless, and nothing raises on its own."""
    dataloader = shipped["dataset_config"]["dataloader_config"]
    assert "fhr" in dataloader["normalize_fields"]
    assert "fhr" in dataloader["dataset_kwargs"]["load_fields"]


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
    # The real geometry: the committed shard carries the real field shapes.
    assert tiny["model_config"]["VAE_model"]["sequence_length"] == 300
    assert tiny["model_config"]["VAE_model"]["raw_per_step"] == 16
    for key in ENCODER_KEYS:
        if key != "encoder_d_ff":  # the one width the smoke variant shrinks
            assert tiny["model_config"]["VAE_model"][key] == shipped["model_config"]["VAE_model"][key]
    assert (
        tiny["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
        == shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    )


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
    """Caught here rather than as a ``ValueError`` three steps into the smoke run. The last one is
    this architecture's own: rotary position encoding rotates coordinate pairs, so the derived
    encoder head width must be even."""
    vae = tiny["model_config"]["VAE_model"]
    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior
    assert (vae["d_model"] // vae["encoder_num_heads"]) % 2 == 0


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


def test_the_resolved_tiny_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver."""
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    SeqVaeLagAttnTrfRws(**_model_kwargs_from(load_config(str(_TINY)), LagAttnTrfRwsTrainer))
