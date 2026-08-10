r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the comparison model's, and the
price of that is drift -- drift in a key that has nothing to do with the target domain is exactly
what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``causal_reach_budget_s``, the coverage floor or the spike breaker would be
attributed to the target domain by every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison
config's value against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence means declaring it
there. The comparison is total rather than schema-limited, and that is the difference from the
conv-Transformer sibling's version of this file: that model replaced the encoders and so has seven
keys of its own and five of the comparison model's that name nothing in it, while this model's net
is a *subclass* whose constructor schema is unchanged. Every key means the same thing in both
files, so every key is comparable.

Five of the six exemptions are mandatory rather than permitted. Four are *identity* -- the output
directory, the MLflow experiment, the run name and the variant tag -- and inheriting them writes
these runs into the other model's tree. The fifth is ``general_config.tag``. **Only one is a
number**, and that is the outcome of measuring rather than of drafting: three loss-scale constants
were drafted as divergences on the argument that this objective sums a $4.9\times$ larger block, and
when each was measured at that scale only ``additive_margin`` had actually moved. The other two live
in :data:`MEASURED_TO_MATCH_PATHS`, whose test asserts the equality so that it reads as a
measurement.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from train.test_utils import make_graph_model

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"
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
    "block's last axis is 78 surviving target channels, an unordered index with no metric; the "
    "weights ship at 0.0 so the columns are honest zeros rather than raw-domain formulas"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else
#: differing is drift, and drift is a confound. Nine entries and no wildcards: the list names its
#: contents so that adding a tenth is a decision rather than an omission.
#:
#: **Five are identity, one is a retuned number, and three are a term that does not exist here.**
#: Three loss-scale constants were drafted as exemptions on the argument that this objective sums a
#: $4.9\times$ larger block; all three were then measured at that scale, and only one of them moved.
#: ``gradient_clip_val`` came out equal because the block's extra terms land on disjoint rows of a
#: per-channel output head rather than accumulating onto one shared parameter, so the loss scales and
#: the norm of its gradient does not. ``beta_schedule.end`` and ``beta_prior`` came out equal because
#: the sweep that bracketed the scale-matched point chose its lower edge -- see ``RESULTS.md``. Each
#: of the three has a test below saying so, because an exemption for a key that no longer differs is a
#: permission that outlived its reason and would wave through the next accidental divergence there.
#:
#: The three auxiliary shape weights are a different kind of entry: not a constant re-derived at this
#: scale but a term whose *formula* has no meaning here, so it is switched off rather than retuned.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.spike_breaker.additive_margin": (
        "re-derived at this loss scale: the margin is stated in nats of the summed block, and this "
        "block is 2340 coefficients against 480 samples"
    ),
    "model_config.VAE_model.lambda_ms": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_deriv": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_boundary": AUX_OFF_REASON,
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: The loss-scale constants that were re-derived at this objective's own scale and landed on the
#: comparison model's value. Named so that equality stays visible as a *measurement* rather than
#: reading as an oversight, and so a future retune has to remove a key from here deliberately.
MEASURED_TO_MATCH_PATHS = (
    "advanced_config.trainer.gradient_clip_val",
    "model_config.VAE_model.beta_schedule.end",
    "model_config.VAE_model.beta_prior",
)

#: The exemptions that are mandatory rather than merely permitted: copying any of them writes this
#: model's runs into the comparison model's output tree and MLflow experiment.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)

#: The exemptions that exist because a *number* had to move, as opposed to a name. Named separately
#: so the reason it carries stays a claim about this objective's scale rather than a general licence
#: to differ -- and because the direction is what an argument applied backwards would reverse.
#:
#: One entry, not the three first drafted. The margin is stated in nats of the summed block and is
#: the only one of the three that a $4.9\times$ larger block genuinely moved.
RETUNED_PATHS = ("advanced_config.spike_breaker.additive_margin",)

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
#: Declared, so a smoke variant cannot quietly acquire a second delta and stop being a smoke
#: variant of the config it claims to be one of.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        # Pinned back to every-epoch in tiny: the shipped config plots every 5th epoch (a
        # render-cost decision for multi-day runs), while the train-smoke tests count one figure
        # set per epoch of a 1-3 epoch run.
        "general_config.plot_frequency",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "model_config.VAE_model.d_model",
        "model_config.VAE_model.d_z",
        "model_config.VAE_model.d_head",
        "model_config.VAE_model.max_lag",
        "model_config.VAE_model.lstm_layers",
        "model_config.VAE_model.dropout",
        "model_config.VAE_model.likelihood",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "advanced_config.tracking.mlflow.enabled",
    }
)

#: Every leaf of ``smoke_hie.yaml`` that resolves to something other than ``default.yaml``'s value.
#: Declared for the same reason ``tiny.yaml``'s list is, and it carries more weight here: this
#: variant is the one whose numbers get quoted, so a delta it acquired quietly would be a difference
#: between the model that was read and the model that ships, attributed to neither.
#:
#: Note what is **not** in it. Every model width, the reach budget, the clip, the spike breaker, the
#: likelihood and ``beta_schedule.end`` are all inherited: only the run's *scale* is local. The one
#: model-block entry is the beta ramp's length, which cannot be inherited because 50 epochs is a
#: hundredth of a production run and a quarter of this one.
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
def smoke_hie() -> dict:
    return load_config(str(_SMOKE_HIE))


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
    so the run trains a *different architecture* than its config describes, and only a checkpoint
    that will not reload months later reveals it."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnFs.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


def test_raw_per_step_is_present_and_is_no_longer_the_decoder_width(shipped):
    """The one key whose *meaning* the target-domain change alters without changing its value.

    ``TrimmedRawGeometry`` validates the raw index identities against it and the diagnostic page's
    first row is drawn on the raw grid, so it stays a geometry input; what each horizon token emits
    is the surviving target-channel count. Deleting it would break the geometry rather than narrow
    the decoder, and setting it would not widen the decoder by one channel."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["raw_per_step"] == 16
    assert "decoder_out_channels" not in vae


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_budget_keeps(shipped):
    """The binding this model's whole unit convention rests on, resolved through the real driver:
    the reach budget decides the surviving channels, the survivors decide the decoder width, and
    the width decides what every reported nat is summed over."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    kwargs = _model_kwargs_from(shipped, LagAttnFsTrainer)
    model = SeqVaeLagAttnFs(**kwargs)

    assert len(kwargs["target_keep_index"]) == 78
    assert model.decoder_out_channels == 78
    assert model.decoder.out_channels == 78
    assert model.raw_per_step == 16  # untouched by the width
    assert model.horizon * model.decoder_out_channels == 2340


def test_loss_only_keys_do_not_reach_the_constructor(shipped):
    """The net takes tensors and computes a loss on request; it owns none of these. The constructor
    is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on the production
    config -- a poor place to find out."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    kwargs = _model_kwargs_from(shipped, LagAttnFsTrainer)

    for name in TASK_LEVEL_KEYS:
        assert name not in kwargs, f"{name} is not the net's"


# --------------------------------------------------------------------------------------
# Identity, and parity with the model this one is compared against
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(shipped, sibling, path):
    """Copying any of these writes this model's runs into the comparison model's output tree and
    mixes them into its MLflow experiment -- unrecoverable after the fact, because both runs are
    then indistinguishable by the only fields anything indexes on."""
    value = str(_get(shipped, path))

    assert value != str(_get(sibling, path))
    assert "fs" in value, (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


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


def test_the_two_weights_hold_the_anchor_ratio_the_design_fixes(shipped, sibling):
    r"""$\beta_p$ moves with $\beta$, whatever $\beta$ is.

    The ratio is the invariant, not either value: the anchor's restoring force saturates at
    $\beta_p / 2$ per latent dimension while the reconstruction it opposes is the thing this target
    domain multiplied, so an arm that moved one key alone would sweep the anchor's standing at the
    same time and a pinning prior would have two explanations. Asserted against the comparison
    model's ratio rather than against a literal, so the pair can be retuned together without this
    becoming a second place the number lives.
    """
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    assert mine["beta_prior"] / mine["beta_schedule"]["end"] == pytest.approx(
        theirs["beta_prior"] / theirs["beta_schedule"]["end"]
    )
    # The warm-up is the posterior-collapse guard and is not part of the swept axis, so only the
    # endpoint was ever in question.
    assert mine["beta_schedule"]["start"] == 0.0


@pytest.mark.parametrize("path", RETUNED_PATHS)
def test_each_retuned_value_is_larger_than_the_one_it_replaces(shipped, sibling, path):
    """It moved because this objective sums a larger block, so it moved *upwards*. A retune that
    landed below the comparison model's value would be a sign the scale argument had been applied in
    the wrong direction, which is the mistake the whole family of these constants shares."""
    assert float(_get(shipped, path)) > float(_get(sibling, path))


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_PATHS)
def test_each_measured_constant_landed_on_the_comparison_models_value(shipped, sibling, path):
    """The three loss-scale constants that were drafted as divergences and measured back to parity.
    The assertion exists so the equality stays visible as a **measurement** rather than reading as an
    oversight, and so a future retune has to leave :data:`MEASURED_TO_MATCH_PATHS` deliberately.

    ``gradient_clip_val``: a clipping threshold does not transfer across loss scales, so it was
    re-derived from a 120-epoch instrumented run at this objective's own scale (the config comment
    records the run and the percentiles). The pre-clip gradient-norm distribution came out close to
    the comparison model's -- q99 4421 against 4681 -- and the same rule, the smallest round value
    above q99, returns the same number. The reason belongs beside it: the reconstruction sums over 78
    channels but the decoder's output head is per-channel, so the extra terms land on disjoint rows
    of two ``Linear`` layers rather than accumulating onto one shared parameter. The loss scales with
    the block; the norm of its gradient does not.

    ``beta_schedule.end`` and ``beta_prior``: this file shipped the scale-matched 5.0 / 0.5 first,
    and the four-arm sweep that bracketed that point came out monotone in $\\beta$ with its lower
    edge winning every column of the selection rule. See ``RESULTS.md``.
    """
    assert float(_get(shipped, path)) == float(_get(sibling, path))
    assert path not in PARITY_EXEMPT_PATHS


def test_no_path_is_both_exempt_and_measured_to_match():
    """The two lists make opposite claims about the same key, so an overlap is a contradiction the
    parity assertions above would resolve arbitrarily."""
    assert set(MEASURED_TO_MATCH_PATHS).isdisjoint(PARITY_EXEMPT_PATHS)
    assert set(RETUNED_PATHS) <= set(PARITY_EXEMPT_PATHS)


def test_the_plotting_block_keeps_the_inherited_drivers_spelling(shipped):
    """The callback assembly is inherited and reads this literal. Renaming the block to match this
    package would disable the per-epoch diagnostic figure with no error anywhere."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True
    assert LagAttnFsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_precision_is_float32(shipped):
    """Mixed precision would need float32 islands around the log-variances, the closed-form KL and
    the NLL reduction -- and this model's NLL reduction is over 2340 terms, the largest in the
    family."""
    assert shipped["advanced_config"]["trainer"]["precision"] == "32-true"


def test_compile_is_off_and_stays_inert_for_this_net(shipped):
    """Inherited whole from the raw-signal sibling, along with the reason: this is that net, LSTM
    encoders included, so the driver does not read the key at all."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    assert shipped["advanced_config"]["trainer"]["compile"] is False
    assert LagAttnFsTrainer.compile_model_requested(object()) is False


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_accumulate_grad_batches_is_present_and_the_warm_start_is_off(shipped):
    """``accumulate_grad_batches`` is the first step of the memory escalation, so it must exist as
    a key rather than as a framework default -- and the output activations here are $1.68$ GiB at
    batch $128$ against the raw model's $0.25$. ``core_model_checkpoint`` stays null: a blob from
    the comparison model carries a different ``model_class`` stamp and a decoder head of a
    different width, and the guard would refuse it -- correctly."""
    assert "accumulate_grad_batches" in shipped["general_config"]
    assert shipped["model_config"]["core_model_checkpoint"] is None


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """z is the only route to the decoder; a nonzero beta before the decoder can use the latent is
    the standard route to posterior collapse. Only the endpoint was retuned."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["warmup_epochs"] == 50


def test_the_breaker_is_pinned_by_value_rather_than_by_presence(shipped):
    r"""``max(EMA, ema_floor)`` collapses to the floor once the EMA is negative -- and a summed
    2340-coefficient learned-variance NLL goes negative sooner than anything else in this family --
    so the floor sits far above any reachable loss, disabling the relative test outright, while
    ``additive_margin`` compares against the *raw* EMA and keeps working there. This exact
    configuration has already cost this repository a run."""
    breaker = shipped["advanced_config"]["spike_breaker"]

    assert breaker["enabled"] is True  # the non-finite guard is the point
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9
    assert breaker["additive_margin"] > 0.0  # finite-spike detection; 0 would disable it
    assert breaker["max_consecutive_skips"] > 0  # the deadlock escape hatch


def test_the_reach_budget_ships_the_guard_on_and_decides_the_block(shipped):
    r"""It does one more thing here than in the comparison model: the survivors are also what the
    decoder emits, so this key sets the block cardinality and therefore the units of every reported
    nat. Presence is asserted (not merely non-error) because ``null`` and *absent* look identical to
    a ``.get``, and the value is pinned because a silent revert to ``null`` would both restore the
    input-side leak and change what every number means."""
    vae = shipped["model_config"]["VAE_model"]

    assert "causal_reach_budget_s" in vae
    assert vae["causal_reach_budget_s"] == 120


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient. A single appearance anywhere in a config
    -- load_fields, normalize_fields, a comment someone uncommented -- is a defect here twice over:
    it would break the target-only / source-conditioned separation *and* put the source's own
    signal into the forecast target.

    Globbed rather than listed: a config added later is exactly the one nobody would think to add
    to a list here, and this guard costs nothing to run over the whole directory."""
    configs = sorted(_CONFIG_DIR.glob("*.yaml"))

    assert configs, "the config directory is empty; the glob is checking nothing"
    for path in configs:
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_the_target_blocks_are_loaded_and_normalized(shipped):
    """The one guard the target domain moves. Both blocks, field by field: a config carrying one of
    them is a target with a hole in it, and an unnormalised block makes the Gaussian NLL
    meaningless with nothing else raising."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    dataloader = shipped["dataset_config"]["dataloader_config"]

    for field in LagAttnFsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


# --------------------------------------------------------------------------------------
# The smoke variant
# --------------------------------------------------------------------------------------
def _non_comment_lines(path: Path) -> int:
    """Count the lines of a YAML file that are neither blank nor a comment."""
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def test_the_tiny_variant_names_only_its_deltas():
    """The whole point of the ``base:`` mechanism, counted on the raw file: a variant that re-stated
    the inherited settings would be a copy that drifts.

    Budgeted against the comparison model's own smoke variant rather than against a written-down
    number, because the two are smoke variants of configs that are themselves pinned leaf-for-leaf:
    a delta this one has and that one does not is the same drift, one level down."""
    sibling_tiny = _SIBLING_CONFIG.parent / "tiny.yaml"

    mine = _non_comment_lines(_TINY)

    assert mine <= _non_comment_lines(sibling_tiny), (
        f"tiny.yaml carries {mine} non-comment lines against the comparison variant's "
        f"{_non_comment_lines(sibling_tiny)}; it should be deltas only"
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
    """Including every correctness requirement, which is why it must not re-state them -- and
    including the reach budget, which here is not merely a guard but the decoder's width."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["advanced_config"]["trainer"]["precision"] == "32-true"
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    vae = tiny["model_config"]["VAE_model"]
    assert vae["sequence_length"] == 300
    assert vae["raw_per_step"] == 16
    assert vae["horizon"] == 30
    assert vae["causal_reach_budget_s"] == shipped["model_config"]["VAE_model"][
        "causal_reach_budget_s"
    ]
    assert vae["causal_norm"] is True
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


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny):
    """Caught here rather than as a ``ValueError`` three steps into the smoke run."""
    vae = tiny["model_config"]["VAE_model"]

    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior
    assert vae["warmup_period"] < vae["sequence_length"] - vae["horizon"]


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    """This package commits no binary fixtures; every model in the family reads the same shards."""
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
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnFs(**_model_kwargs_from(load_config(str(_TINY)), LagAttnFsTrainer))
    # The smoke model is small everywhere except where it must not be: the decoder still emits the
    # production width, because that follows the budget rather than any width the variant shrinks.
    assert model.d_model == 32
    assert model.decoder_out_channels == 78


# --------------------------------------------------------------------------------------
# The dev-box validation variant
# --------------------------------------------------------------------------------------
def test_the_local_variant_resolves_with_its_base_consumed(smoke_hie):
    """A leftover ``base`` key would reach ``validate_config`` as an unknown key and the resolved
    config written beside the run's checkpoints -- its provenance record -- would carry a loader
    directive rather than a setting."""
    assert "base" not in smoke_hie


def test_the_local_delta_is_exactly_the_declared_key_list(smoke_hie, shipped):
    """Both directions, and it matters more here than for ``tiny.yaml``: this is the variant whose
    numbers get quoted, so an undeclared delta is a difference between the model that was read and
    the model that ships, attributed to neither."""
    mine = dict(_leaves(smoke_hie))
    theirs = dict(_leaves(shipped))
    differing = {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert differing == set(SMOKE_HIE_DELTA_PATHS)


def test_the_local_variant_inherits_everything_the_reading_depends_on(smoke_hie, shipped):
    """Only the run's *scale* is local. Every quantity the pre-registered criteria are read from --
    the block the NLL sums over, the weights it is balanced against, the clamp the log-variances sit
    inside, and the breaker that could silently replace the loss with its own EMA -- is inherited,
    or the reading is of a different model."""
    vae = smoke_hie["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    # The block cardinality, hence the units of every reported nat.
    assert vae["causal_reach_budget_s"] == shipped_vae["causal_reach_budget_s"]
    assert vae["horizon"] == shipped_vae["horizon"]
    # The weights whose balance the run is a reading of, and the ramp's start.
    assert vae["beta_schedule"]["end"] == shipped_vae["beta_schedule"]["end"]
    assert vae["beta_schedule"]["start"] == 0.0
    assert vae["beta_prior"] == shipped_vae["beta_prior"]
    assert vae["likelihood"] == "gaussian_nll"  # the learned-variance heads, not the debug path
    assert vae["logvar_clamp"] == shipped_vae["logvar_clamp"]
    assert vae["causal_norm"] is True
    assert (
        smoke_hie["advanced_config"]["trainer"]["gradient_clip_val"]
        == shipped["advanced_config"]["trainer"]["gradient_clip_val"]
    )
    assert smoke_hie["advanced_config"]["spike_breaker"] == shipped["advanced_config"][
        "spike_breaker"
    ]


def test_the_local_variant_ramps_beta_inside_its_own_epoch_budget(smoke_hie):
    """The one model-block delta, and the one that cannot be inherited: 50 epochs is a hundredth of
    a production run and a quarter of this one. Beta must reach its endpoint early enough that the
    columns are read off a model that has been paying its rate for most of the run."""
    warmup = smoke_hie["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
    epochs = smoke_hie["general_config"]["epochs"]

    assert warmup * 10 <= epochs


def test_the_local_variant_points_at_the_committed_shard_and_its_own_statistics(smoke_hie):
    """Both files, and the statistics file is the one that matters. It is generated from this shard;
    inheriting the four-sample fixture's would pass every existing guard, because the channel counts
    match, and silently z-score the target against another recording's mean."""
    dataset = smoke_hie["dataset_config"]

    assert dataset["vae_train_datasets"] == dataset["vae_test_datasets"]  # in-sample, deliberately
    for path in (*dataset["vae_train_datasets"], dataset["stat_path"]):
        assert (_REPO_ROOT / path).is_file(), (
            f"{path} is missing; regenerate it from the committed shard with "
            f"hdf5_dataset/calculate_dataset_stats.py at trim_minutes=1.0"
        )
    assert "hie_cs" in dataset["stat_path"]


def test_the_local_variant_runs_on_one_device_with_tracking_off(smoke_hie):
    """A dev-box run: one GPU, and no MLflow server to log to. The run directory is the record."""
    assert smoke_hie["general_config"]["cuda_devices"] == [0]
    assert smoke_hie["advanced_config"]["tracking"]["mlflow"]["enabled"] is False


def test_the_resolved_local_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver -- and built at
    the production decoder width, which is the whole reason this variant does not shrink the model."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    resolved = resolve_config_file(str(_SMOKE_HIE), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnFs(**_model_kwargs_from(load_config(str(_SMOKE_HIE)), LagAttnFsTrainer))
    assert model.d_model == 128
    assert model.horizon * model.decoder_out_channels == 2340


def test_the_local_variant_normalizes_and_loads_both_target_blocks(smoke_hie):
    """Inherited, and asserted anyway: the entry point's guard runs against
    ``LagAttnFsTrainer.TARGET_FIELDS``, and this is the config that reaches it on the one shard
    whose statistics are not the fixture's."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    dataloader = smoke_hie["dataset_config"]["dataloader_config"]

    for field in LagAttnFsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field
