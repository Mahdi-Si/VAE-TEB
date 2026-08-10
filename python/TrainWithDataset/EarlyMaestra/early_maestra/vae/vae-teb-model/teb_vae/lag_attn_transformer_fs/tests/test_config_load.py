r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting either comparison model's, and
the price of that is drift -- drift in a key that has nothing to do with the axis under study is
exactly what destroys the two comparisons this package exists to make. A difference in ``seed``,
``lr``, ``free_bits``, ``causal_reach_budget_s``, the coverage floor or the spike breaker would be
attributed to an encoder by one reading and to a target domain by the other.

So this file carries **three** pins rather than one sibling's one, and it matters that each catches
something the other two cannot:

**Pin 1, against the conv-Transformer sibling: total.** Both models are built by the same
constructor -- this one's net is that one's plus a target-domain mixin -- so every key means the same
thing in both files and every leaf is comparable, with an ``"<absent>"`` sentinel so a key present in
only one file is drift. This is the strongest of the three, and stronger than anything that sibling
can say about *its* pin, which must exclude twelve encoder keys.

**Pin 2, against the feature-domain sibling: schema-limited.** Twelve keys are excluded by name --
the seven this constructor has and that one does not, and the five that describe the encoder being
replaced. Most of what is left is *implied*: for any leaf outside every allow-list, Pin 1 plus the
two green sibling pins give equality by transitivity, because the four configs are four nodes and
those three pins are a spanning tree of them. What Pin 2 adds is one thing the tree cannot give --
that no *thirteenth* leaf appears on the target edge. The feature suite pins its config against the
raw one and Pin 1 pins this against the conv-Transformer one, so nothing otherwise stops a leaf
differing on **both** target edges in the same way and passing every pin.

**Pin 3, the square closes -- on keys, not values.** The set of paths differing on each target edge
is equal, and likewise for each encoder edge, both outside identity. Sets deliberately: asserting the
*deltas* equal would pre-commit the re-derived ``additive_margin`` to landing on exactly
$5 \times 10^{3}$, which is the number the measurement exists to determine. What Pin 3 catches is a
leaf entering or leaving one edge without the other following -- a change to the shared ancestor that
one descendant tracks and the other does not.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
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
#: The production pins compare the configs that will run on the prod box; this one compares the two
#: files the *encoder axis is actually read from* on this box.
_SIBLING_SMOKE_HIE = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "smoke_hie.yaml"

#: Exactly the files this package ships. No encoder sweep belongs here -- their
#: values were swept in the raw domain and the encoders are reached by import, not by copy -- and
#: this is what stops one arriving undeclared.
DECLARED_CONFIG_FILES = frozenset({"default.yaml", "tiny.yaml", "smoke_hie.yaml"})

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

#: The seven encoder keys this constructor has and the feature-domain sibling's has never heard of.
#: Taken from the conv-Transformer sibling's own test module rather than restated: the two lists
#: describe one constructor schema, and a second copy could only drift from it.
from teb_vae.lag_attn_transformer_rws.tests.test_config_load import (  # noqa: E402
    ENCODER_KEYS,
    REPLACED_ENCODER_KEYS,
)

#: The twelve *config* keys the target edge is blind to. Seven plus five, and it is exactly twelve
#: rather than the thirteen keywords the two constructors differ by, because ``decoder_out_channels``
#: is named in no YAML in the repository -- the width follows the target gate. Confusing the thirteen
#: with the twelve is how an exclusion list ends up one name too long.
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

#: Why the three auxiliary shape weights are the one *model* divergence on the target edge. Written
#: once and shared by the three entries below, because it is one reason rather than three, and worded
#: exactly as the feature-domain sibling's copy: the two target edges of the square carry the same
#: divergence for the same reason.
AUX_OFF_REASON = (
    "the shape terms read the forecast block's last axis as consecutive raw samples -- pooling it, "
    "differencing it, and joining its first entry to the anchor's last observed sample -- and this "
    "block's last axis is 78 surviving target channels, an unordered index with no metric; the "
    "weights ship at 0.0 so the columns are honest zeros rather than raw-domain formulas"
)

#: **Pin 1's** allow-list: every leaf of both files is compared against the conv-Transformer sibling
#: except these. Nine entries and no wildcards, so adding a tenth is a decision rather than an
#: omission.
#:
#: Five are *identity* and are mandatory rather than merely permitted -- copying any of them writes
#: this model's runs into that model's output tree and MLflow experiment, which is unrecoverable
#: after the fact because both runs are then indistinguishable by the only fields anything indexes
#: on. Two are the loss-scale constants, and both are measurements at this encoder. The last three
#: are the auxiliary shape weights, which are not a constant retuned for this block but a term whose
#: *formula* has no meaning over it, so they are switched off rather than rescaled.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture and the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.spike_breaker.additive_margin": (
        "the margin is stated in nats of the summed block, and this block is 2340 coefficients "
        "against 480 samples -- the one leaf on the TARGET edge of the square, and re-measured at "
        "this encoder rather than inherited"
    ),
    "model_config.VAE_model.lambda_ms": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_deriv": AUX_OFF_REASON,
    "model_config.VAE_model.lambda_boundary": AUX_OFF_REASON,
    "advanced_config.trainer.gradient_clip_val": (
        "MEASURED: this stack's pre-clip gradient norms are half the conv-Transformer raw model's "
        "at the same shard (q99 3048 against 4683), so the rule both models applied returns 4000 "
        "here and 5000 there -- a divergence produced by a measurement, not by drift"
    ),
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: **Pin 2's** allow-list, against the feature-domain sibling, on top of the twelve excluded schema
#: keys. ``lr_warmup_steps`` is the one non-encoder leaf on the encoder edge that is a property of
#: the architecture: a pre-norm attention stack's optimisation needs it and neither conv-LSTM model
#: has it at all. ``gradient_clip_val`` is the one that is a property of the *measurement*.
#:
#: ``additive_margin`` is deliberately **not** here any more. It was exempt while its value at this
#: encoder was unknown; the instrumented run put it on the feature sibling's number, so the pin now
#: compares it and the equality is asserted rather than permitted -- see
#: :data:`MEASURED_TO_MATCH_ENCODER_EDGE`.
TARGET_EDGE_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the architecture and the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "general_config.lr_warmup_steps": (
        "the step-granular ramp both conv-Transformer models add; absent from both conv-LSTM ones"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "MEASURED: the conv-LSTM feature model's q99 on this shard is 4421 and this stack's is "
        "3048, so the same rule returns 5000 there and 4000 here -- the encoder moves this "
        "constant, which is exactly what the derivation existed to find out"
    ),
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
}

#: The exemptions that are mandatory rather than merely permitted, shared by both pins.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)

#: Every leaf either pin declares, whatever the declaration says. **Pin 3** subtracts this from both
#: sides of each comparison, so it reads the *undeclared* key sets: a declared leaf carries a stated
#: reason and is already covered by its own pin's reverse guard, and one of them differs precisely
#: because a measurement at this encoder put it there.
DECLARED_PATHS = frozenset(PARITY_EXEMPT_PATHS) | frozenset(TARGET_EDGE_EXEMPT_PATHS)

#: Leaves that are neither pinned equal nor asserted different, because no run had measured them at
#: this encoder yet. **Empty**: the instrumented run landed and both loss-scale constants moved into
#: a list that makes a claim.
#:
#: It held ``additive_margin`` while its value here was unknown -- equal to the feature-domain
#: sibling and different from the conv-Transformer one, so *both* a declared divergence and an
#: asserted equality would have pre-committed the experiment's result. The run put it on the feature
#: sibling's number, so it left this tuple for :data:`MEASURED_TO_MATCH_ENCODER_EDGE` and Pin 2 now
#: compares it. ``gradient_clip_val`` was deliberately never here: it was equal to both comparison
#: models, both pins compared it, and the measurement moving it failed a pin and forced the
#: divergence to be declared with its reason -- which is what happened.
#:
#: Kept as an empty tuple rather than deleted: it is the seam a future re-derivation at a new scale
#: parks a key in, and the two reverse guards already read it.
PENDING_MEASUREMENT_PATHS: Tuple[str, ...] = ()

#: Leaves whose equality with **both** comparison models is a measurement rather than an oversight,
#: carried across from the feature suite. The four-arm $\beta$ sweep bracketed the scale-matched
#: point and chose its lower edge, which happens to be the raw models' own pair, so an equality here
#: reads as a coincidence unless something says otherwise.
#:
#: Two entries. The instrumented run decided the other two candidates and neither joined this tuple:
#: ``gradient_clip_val`` came out *different* from both comparison models and is declared in both
#: allow-lists above, and ``additive_margin`` matches one model rather than both -- see
#: :data:`MEASURED_TO_MATCH_ENCODER_EDGE`.
MEASURED_TO_MATCH_PATHS = (
    "model_config.VAE_model.beta_schedule.end",
    "model_config.VAE_model.beta_prior",
)

#: Leaves whose equality with the **encoder-axis** comparison model alone is a measurement. Separate
#: from :data:`MEASURED_TO_MATCH_PATHS` because that tuple asserts equality with *both* comparison
#: models, and a key that equals one and differs from the other cannot be recorded there without
#: weakening what the tuple claims about its own entries.
#:
#: One entry. ``additive_margin`` is stated in nats of the summed block, so the conv-Transformer raw
#: model's $1 \times 10^{3}$ could not transfer across a $4.9\times$ block; what was open was whether
#: the conv-LSTM feature model's $5 \times 10^{3}$ transfers across an *encoder*. The instrumented run
#: says it does -- this stack's post-warm-up ``main_loss`` fluctuation puts the shipped margin at
#: $4.0\times$ the largest epoch-to-epoch movement and $11.5\times$ the standard deviation, against
#: $3.3\times$ and $12\times$ for the value's own derivation -- so the equality is a result and is
#: asserted rather than exempted.
MEASURED_TO_MATCH_ENCODER_EDGE = ("advanced_config.spike_breaker.additive_margin",)

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
#: Declared as a frozenset so a smoke variant cannot quietly acquire a second delta and stop being a
#: smoke variant of the config it claims to be one of.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.lr_warmup_steps",
        # Pinned back to every-epoch in tiny: the shipped config plots every 5th epoch (a render-cost
        # decision for multi-day runs), while the train-smoke tests count one figure set per epoch of
        # a 1-3 epoch run.
        "general_config.plot_frequency",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "model_config.VAE_model.d_model",
        "model_config.VAE_model.d_z",
        "model_config.VAE_model.d_head",
        "model_config.VAE_model.max_lag",
        "model_config.VAE_model.encoder_d_ff",
        "model_config.VAE_model.target_attention_blocks",
        "model_config.VAE_model.source_attention_blocks",
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
#: Declared for the same reason ``tiny.yaml``'s list is, and it carries more weight here: this is the
#: variant whose numbers get quoted and the one the loss-scale constants are derived from, so a delta
#: it acquired quietly would be a difference between the model that was read and the model that ships,
#: attributed to neither.
#:
#: Note what is **not** in it. Every model width, the reach budget, the clip, the spike breaker, the
#: likelihood and ``beta_schedule.end`` are all inherited: only the run's *scale* is local. The two
#: schedule-length entries are the exceptions, and both are lengths that cannot be inherited because
#: a fraction of a production run is a large fraction of this one.
SMOKE_HIE_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.lr_warmup_steps",
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

#: What the shipped reach budget resolves the two streams to, and the decoder width that follows.
SHIPPED_TARGET_CHANNELS = 78
SHIPPED_SOURCE_CHANNELS = 29


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


def _differing_paths(mine: dict, theirs: dict) -> set:
    """Return the dotted paths at which two flattened configs disagree, absence included."""
    return {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
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

    from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer

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
def tiny() -> dict:
    return load_config(str(_TINY))


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
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


def test_no_task_level_key_is_secretly_a_constructor_argument():
    """The other direction, and the one that keeps the check above honest: a constructor keyword
    wrongly listed as task-level would let a genuinely orphaned key hide behind the exemption."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)

    assert [key for key in TASK_LEVEL_KEYS if key in constructor_keys] == []


def test_the_seven_encoder_keys_are_present_and_reach_the_constructor(shipped):
    """Each varies across an arm that was measured in the raw domain and is inherited here; a key
    the constructor did not take would be one the signature sweep drops without a word."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfFs.__init__).parameters)
    vae = shipped["model_config"]["VAE_model"]

    assert len(ENCODER_KEYS) == 7
    for key in ENCODER_KEYS:
        assert key in vae, f"{key} is missing from the shipped config"
        assert key in constructor_keys, f"{key} names no constructor argument"


def test_no_replaced_encoder_key_survives_in_any_config(shipped, tiny, smoke_hie):
    """The five keys describing the encoder being replaced. Each would be dropped in silence, so a
    copy-pasted block from the feature-domain sibling would leave a config that reads correct and
    builds a different model."""
    assert len(REPLACED_ENCODER_KEYS) == 5
    for config in (shipped, tiny, smoke_hie):
        present = [
            key for key in REPLACED_ENCODER_KEYS if key in config["model_config"]["VAE_model"]
        ]
        assert present == []


def test_raw_per_step_is_present_and_is_not_the_decoder_width(shipped):
    """The one key whose *meaning* the target domain alters without changing its value.

    ``TrimmedRawGeometry`` validates the raw index identities against it and the diagnostic page's
    first row is drawn on the raw grid, so it stays a geometry input; what each horizon token emits
    is the surviving target-channel count. Deleting it would break the geometry rather than narrow
    the decoder, and setting ``decoder_out_channels`` is not even possible -- it is not a keyword of
    this constructor at all, which is what stops a second field disagreeing with the gate."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["raw_per_step"] == 16
    assert "decoder_out_channels" not in vae
    assert "decoder_out_channels" not in inspect.signature(
        SeqVaeLagAttnTrfFs.__init__
    ).parameters


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_budget_keeps(shipped):
    """The binding this model's whole unit convention rests on, resolved through the real driver: the
    reach budget decides the surviving channels, the survivors decide the decoder width, and the
    width decides what every reported nat is summed over."""
    kwargs = _model_kwargs_from(shipped)
    model = SeqVaeLagAttnTrfFs(**kwargs)

    assert "causal_reach_budget_s" not in kwargs
    assert len(kwargs["target_keep_index"]) == SHIPPED_TARGET_CHANNELS
    assert len(kwargs["source_keep_index"]) == SHIPPED_SOURCE_CHANNELS
    assert model.decoder_out_channels == SHIPPED_TARGET_CHANNELS
    assert model.decoder.out_channels == SHIPPED_TARGET_CHANNELS
    assert model.raw_per_step == 16  # untouched by the width
    assert model.horizon * model.decoder_out_channels == 2340


def test_the_shipped_config_builds_the_shipped_encoders(shipped):
    """The other half of what a launch produces: the encoder block reaches the two encoders, and the
    bounded-source / full-prefix-target asymmetry is the architecture rather than a setting."""
    model = SeqVaeLagAttnTrfFs(**_model_kwargs_from(shipped))

    assert len(model.target_encoder.attention_blocks) == 6
    assert len(model.source_encoder.attention_blocks) == 3
    assert model.source_encoder.attention_window == 16
    # The stem reaches 21 steps = 84 s; the source bound 21 + 3*15 = 66 steps = 264 s, inside the
    # 360 s lag search range, which is what keeps the encoder a local summariser.
    assert model.source_encoder.receptive_field == 66
    assert model.target_encoder.receptive_field is None  # the full causal prefix


def test_loss_only_keys_do_not_reach_the_constructor(shipped):
    """The net takes tensors and computes a loss on request; it owns none of these. The constructor
    is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on the production
    config -- a poor place to find out."""
    kwargs = _model_kwargs_from(shipped)

    for name in TASK_LEVEL_KEYS:
        assert name not in kwargs, f"{name} is not the net's"


# --------------------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(shipped, target_sibling, encoder_sibling, path):
    """Each must differ from **both** comparison configs and name this architecture *and* this
    target domain, so a value copy-pasted from either one fails. Copying any of them writes this
    model's runs into that model's output tree and mixes them into its MLflow experiment --
    unrecoverable after the fact, because both runs are then indistinguishable by the only fields
    anything indexes on."""
    value = str(_get(shipped, path))

    assert value != str(_get(target_sibling, path))
    assert value != str(_get(encoder_sibling, path))
    # Either spelling of the architecture: the identifiers abbreviate it (`lag_attn_trf_fs`) while
    # the output tree carries the package directory name in full.
    assert "trf" in value or "transformer" in value, (
        f"{path} is {value!r}, which does not name this architecture"
    )
    assert "fs" in value, f"{path} is {value!r}, which does not name this target domain"


# --------------------------------------------------------------------------------------
# Pin 1: total, against the conv-Transformer sibling
# --------------------------------------------------------------------------------------
def test_every_leaf_equals_the_target_siblings_value(shipped, target_sibling):
    """The strongest of the three pins. This model's net is that one's plus a target-domain mixin, so
    the constructor schema is unchanged, every key means the same thing in both files, and every leaf
    is comparable -- with an ``"<absent>"`` sentinel, so a key present in only one file is drift."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(target_sibling))
    compared = (set(mine) | set(theirs)) - set(PARITY_EXEMPT_PATHS)

    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(compared)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

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
        if path not in PENDING_MEASUREMENT_PATHS
        and mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


# --------------------------------------------------------------------------------------
# Pin 2: schema-limited, against the feature-domain sibling
# --------------------------------------------------------------------------------------
def test_the_exclusion_list_is_the_twelve_config_keys_and_not_the_thirteen_keywords():
    """``decoder_out_channels`` is the thirteenth constructor keyword the two nets differ by and is
    named in no YAML in the repository, so an exclusion list built from the *schema* difference would
    be one name too long -- and would then be silently blind to a key that could appear in a config
    later."""
    assert len(SCHEMA_ONLY_PATHS) == 12
    assert len(set(SCHEMA_ONLY_PATHS)) == 12
    assert "model_config.VAE_model.decoder_out_channels" not in SCHEMA_ONLY_PATHS


def test_every_comparable_leaf_equals_the_encoder_siblings_value(shipped, encoder_sibling):
    """Most of this is implied by Pin 1 plus the two green sibling pins -- the four configs are four
    nodes and those three pins are a spanning tree of them -- and that is fine. What Pin 2 adds is
    the one thing the tree cannot give: no *thirteenth* leaf on the target edge. Nothing otherwise
    stops a leaf differing on both target edges in the same way and passing every pin.

    The two sibling tests it leans on are
    ``teb_vae/lag_attn_fs/tests/test_config_load.py::test_every_leaf_equals_the_comparison_configs_value``
    and
    ``teb_vae/lag_attn_transformer_rws/tests/test_config_load.py::test_every_non_encoder_leaf_equals_the_comparison_configs_value``.
    Delete either and this pin stops being mostly-implied and starts being the only thing checking a
    large part of the edge -- which it would then do, at this file's own granularity.
    """
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(encoder_sibling))
    compared = (
        (set(mine) | set(theirs))
        - set(SCHEMA_ONLY_PATHS)
        - set(TARGET_EDGE_EXEMPT_PATHS)
    )

    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(compared)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert drift == {}, (
        f"these leaves differ from the feature-domain sibling's config outside the twelve encoder "
        f"keys and are not declared in TARGET_EDGE_EXEMPT_PATHS: {drift}"
    )


def test_pin_two_declares_no_exemption_that_is_no_longer_a_divergence(shipped, encoder_sibling):
    """The reverse guard again, with the pending-measurement key excused by name rather than by an
    exception the reader has to infer."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(encoder_sibling))

    stale = [
        path
        for path in TARGET_EDGE_EXEMPT_PATHS
        if path not in PENDING_MEASUREMENT_PATHS
        and mine.get(path, "<absent>") == theirs.get(path, "<absent>")
    ]

    assert stale == []


def test_nothing_is_still_awaiting_a_measurement():
    """The instrumented run landed, so every leaf that differs from a comparison model differs for a
    stated reason and every leaf that matches one does so as a result.

    The three assertions keep the seam honest for the next re-derivation rather than merely asserting
    emptiness: a key parked here must be exempt from *both* reverse guards, since a pending key is by
    definition one whose value neither pin may claim, and it must not simultaneously be recorded as a
    measured equality -- the two lists make opposite claims and the guards would resolve the
    contradiction arbitrarily.
    """
    assert PENDING_MEASUREMENT_PATHS == ()
    assert set(PENDING_MEASUREMENT_PATHS) <= set(PARITY_EXEMPT_PATHS) & set(
        TARGET_EDGE_EXEMPT_PATHS
    )
    assert set(PENDING_MEASUREMENT_PATHS).isdisjoint(
        set(MEASURED_TO_MATCH_PATHS) | set(MEASURED_TO_MATCH_ENCODER_EDGE)
    )


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_ENCODER_EDGE)
def test_each_encoder_edge_match_is_equal_there_and_different_on_the_target_edge(
    shipped, target_sibling, encoder_sibling, path
):
    """The asymmetry is the whole content of the entry, so both halves are asserted.

    Equal to the conv-LSTM feature model because the block cardinality decides this constant and the
    encoder does not -- which the instrumented run measured rather than assumed. Different from the
    conv-Transformer raw model because that block is $480$ samples against $2340$ coefficients. A
    future edit that made it equal to both, or different from both, would be saying something new
    about which of the two axes this constant lives on, and should have to say so here.
    """
    assert float(_get(shipped, path)) == float(_get(encoder_sibling, path))
    assert float(_get(shipped, path)) != float(_get(target_sibling, path))
    assert path not in TARGET_EDGE_EXEMPT_PATHS, "an asserted equality must not also be exempt"
    assert path in PARITY_EXEMPT_PATHS, "the target-edge divergence still needs its declaration"


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_PATHS)
def test_each_measured_constant_equals_both_comparison_models(
    shipped, target_sibling, encoder_sibling, path
):
    r"""The two weights whose equality with every other cell of the grid is a *measurement*.

    The feature-domain sibling shipped the scale-matched $\beta = 5.0$, $\beta_p = 0.5$ first, on the
    argument that its block is $4.9\times$ larger; the four-arm sweep that bracketed that point came
    out monotone in $\beta$ with its lower edge winning every column of the selection rule. The block
    cardinality is unchanged here, so the same answer applies and no sweep ships. The assertion exists
    so the equality reads as a result rather than as an oversight, and so a future retune has to leave
    this tuple deliberately.
    """
    assert float(_get(shipped, path)) == float(_get(target_sibling, path))
    assert float(_get(shipped, path)) == float(_get(encoder_sibling, path))
    assert path not in PARITY_EXEMPT_PATHS
    assert path not in TARGET_EDGE_EXEMPT_PATHS


def test_the_two_weights_hold_the_anchor_ratio_the_design_fixes(shipped, encoder_sibling):
    r"""$\beta_p$ moves with $\beta$, whatever $\beta$ is.

    The ratio is the invariant, not either value: the anchor's restoring force saturates at
    $\beta_p / 2$ per latent dimension while the reconstruction it opposes is the thing the target
    domain multiplied, so an arm that moved one key alone would sweep the anchor's standing at the
    same time and a pinning prior would have two explanations. Asserted against a sibling's ratio
    rather than against a literal, so the pair can be retuned together without this becoming a second
    place the number lives.
    """
    mine = shipped["model_config"]["VAE_model"]
    theirs = encoder_sibling["model_config"]["VAE_model"]

    assert mine["beta_prior"] / mine["beta_schedule"]["end"] == pytest.approx(
        theirs["beta_prior"] / theirs["beta_schedule"]["end"]
    )
    assert mine["beta_schedule"]["start"] == 0.0


# --------------------------------------------------------------------------------------
# Pin 3: the square closes, on key sets
# --------------------------------------------------------------------------------------
def test_the_two_target_edges_of_the_square_carry_the_same_key_set(
    shipped, target_sibling, encoder_sibling
):
    """Sets, not values -- deliberately. Asserting the *deltas* equal would pre-commit the re-derived
    ``additive_margin`` to landing on exactly the feature sibling's value, which is the number the
    measurement exists to determine.

    What this catches is a leaf entering or leaving one target edge without the other following: a
    change to the shared ancestor that one descendant tracks and the other does not, which every
    pairwise pin passes because each pin only ever sees one edge.

    **Undeclared** key sets, for the same reason the comparison is on keys at all. A leaf declared in
    an allow-list carries a stated reason and is already covered by that pin's reverse guard, and one
    of them -- ``gradient_clip_val`` -- is on this edge *because a measurement put it there*: this
    stack's gradient norms are half the conv-Transformer raw model's, so the shared derivation rule
    returns a different value here. Comparing declared leaves too would make this test fail for the
    one reason the square is meant to accommodate.
    """
    root = load_config(str(_ROOT_CONFIG))

    mine = _differing_paths(dict(_leaves(shipped)), dict(_leaves(target_sibling)))
    theirs = _differing_paths(dict(_leaves(encoder_sibling)), dict(_leaves(root)))

    assert mine - DECLARED_PATHS == theirs - DECLARED_PATHS


def test_the_two_encoder_edges_of_the_square_carry_the_same_key_set(
    shipped, target_sibling, encoder_sibling
):
    """The other axis, and the one where the set is large: twelve leaves outside identity and outside
    the declarations -- the twelve encoder keys. An encoder key added to one row of the grid and not
    the other is exactly the drift that makes the two rows non-comparable.

    ``lr_warmup_steps`` and ``gradient_clip_val`` sit outside the comparison as declarations, the
    first because both conv-Transformer models add it and neither conv-LSTM model has it, the second
    because it was re-derived here.
    """
    root = load_config(str(_ROOT_CONFIG))

    mine = _differing_paths(dict(_leaves(shipped)), dict(_leaves(encoder_sibling)))
    theirs = _differing_paths(dict(_leaves(target_sibling)), dict(_leaves(root)))

    assert mine - DECLARED_PATHS == theirs - DECLARED_PATHS
    assert len(mine - DECLARED_PATHS) == 12


# --------------------------------------------------------------------------------------
# The config directory
# --------------------------------------------------------------------------------------
def test_the_config_directory_holds_exactly_the_declared_files():
    """No encoder sweep and no $\\beta$ sweep belongs here -- the encoder values were swept in the raw
    domain and the encoders are reached by import, and the block cardinality is unchanged so the
    feature-domain sweep answered $\\beta$. This is what stops a sweep arriving undeclared and quietly
    becoming a result nobody registered."""
    present = {path.name for path in _CONFIG_DIR.glob("*.yaml")}

    assert present == DECLARED_CONFIG_FILES


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient. A single appearance anywhere in a config
    -- load_fields, normalize_fields, a comment someone uncommented -- is a defect here twice over:
    it would break the target-only / source-conditioned separation *and* put the source's own signal
    into the forecast target.

    Globbed rather than listed: a config added later is exactly the one nobody would think to add to
    a list here, and this guard costs nothing to run over the whole directory."""
    configs = sorted(_CONFIG_DIR.glob("*.yaml"))

    assert configs, "the config directory is empty; the glob is checking nothing"
    for path in configs:
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_precision_is_float32(shipped):
    """Mixed precision would need float32 islands around the log-variances, the closed-form KL and
    the NLL reduction -- and this model's NLL reduction is over 2340 terms, the largest in the
    family. Any of those moving is a difference the two comparisons would misattribute."""
    assert shipped["advanced_config"]["trainer"]["precision"] == "32-true"


def test_compile_ships_off_but_is_live_for_this_driver(shipped):
    """Off in the shipped config, but **live** rather than inert -- and that is the one behavioural
    difference the diamond's resolution order introduces. The feature-domain sibling's driver does not
    read the key at all (its net's LSTM encoders defeat inductor unconditionally); this driver
    inherits the conv-Transformer sibling's reading, because the encoders that made the refusal true
    are gone. It ships off because inductor may reassociate float arithmetic and ``pred_gap`` is a
    small difference of two large block NLLs."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
    from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
    from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

    assert shipped["advanced_config"]["trainer"]["compile"] is False
    assert (
        LagAttnTrfFsTrainer.compile_model_requested
        is LagAttnTrfRwsTrainer.compile_model_requested
    )
    assert LagAttnFsTrainer.compile_model_requested(object()) is False


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_accumulate_grad_batches_is_present_and_the_warm_start_is_off(shipped):
    """``accumulate_grad_batches`` is the first step of the memory escalation, so it must exist as a
    key rather than as a framework default -- and this is the model whose decoder head emits $78$
    channels on encoders whose target stream runs unwindowed causal attention over $T = 300$.
    ``core_model_checkpoint`` stays null: a blob from any sibling carries a different ``model_class``
    stamp, and either different encoder tensors or a decoder head of a different width, and the guard
    would refuse it -- correctly."""
    assert "accumulate_grad_batches" in shipped["general_config"]
    assert shipped["model_config"]["core_model_checkpoint"] is None


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """z is the only route to the decoder; a nonzero beta before the decoder can use the latent is
    the standard route to posterior collapse."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["warmup_epochs"] == 50


def test_the_step_warmup_is_configured_and_positive(shipped):
    """The one non-encoder key on the encoder edge. At $0$ the task delegates to the framework's
    epoch-granularity schedule, so a silent revert to zero would leave a pre-norm attention stack
    with no ramp at all and nothing in the log saying so."""
    assert shipped["general_config"]["lr_warmup_steps"] == 2000


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
    r"""It does one more thing in this target domain than in the raw one: the survivors are also what
    the decoder emits, so this key sets the block cardinality and therefore the units of every
    reported nat. Presence is asserted (not merely non-error) because ``null`` and *absent* look
    identical to a ``.get``, and the value is pinned because a silent revert to ``null`` would both
    restore the input-side leak and change what every number means."""
    vae = shipped["model_config"]["VAE_model"]

    assert "causal_reach_budget_s" in vae
    assert vae["causal_reach_budget_s"] == 120


def test_the_target_blocks_are_loaded_and_normalized(shipped):
    """The one guard the target domain moves, and it reaches this driver from its *feature* parent:
    without both blocks in ``normalize_fields`` the target arrives at its stored scale, the Gaussian
    NLL is computed against a z-scale variance model, and nothing else raises."""
    from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer

    dataloader = shipped["dataset_config"]["dataloader_config"]

    assert LagAttnTrfFsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")
    for field in LagAttnTrfFsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


def test_the_plotting_block_keeps_the_shared_drivers_spelling(shipped):
    """The callback assembly is inherited whole and reads this literal. Renaming the block to match
    this package would disable the per-epoch diagnostic figure with no error anywhere."""
    from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer

    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True
    assert LagAttnTrfFsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


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

    Budgeted against the conv-Transformer sibling's own smoke variant plus the two extra widths this
    one shrinks, rather than against a written-down number: the two are smoke variants of configs
    that are themselves pinned leaf-for-leaf, so a delta this one has and that one does not is the
    same drift, one level down."""
    sibling_tiny = _TARGET_SIBLING.parent / "tiny.yaml"

    mine = _non_comment_lines(_TINY)

    assert mine <= _non_comment_lines(sibling_tiny) + 2, (
        f"tiny.yaml carries {mine} non-comment lines against the conv-Transformer variant's "
        f"{_non_comment_lines(sibling_tiny)} plus the two attention block counts; it should be "
        f"deltas only"
    )


def test_the_base_key_never_reaches_the_validator(tiny, smoke_hie):
    """``load_config`` consumes ``base:``; were it left in, ``validate_config`` would warn on an
    unknown key and the resolved config written beside the run's checkpoints -- its provenance record
    -- would carry a loader directive rather than a setting."""
    assert "base" not in tiny
    assert "base" not in smoke_hie


def test_the_tiny_delta_is_exactly_the_declared_key_list(tiny, shipped):
    """Both directions: an undeclared delta is a smoke run that silently stops resembling the
    production one, and a declared delta that is not there is a stale declaration."""
    differing = _differing_paths(dict(_leaves(tiny)), dict(_leaves(shipped)))

    assert differing == set(TINY_DELTA_PATHS)


def test_the_tiny_variant_inherits_the_settings_it_does_not_name(tiny, shipped):
    """Including every correctness requirement, which is why it must not re-state them -- and
    including the reach budget, which here is not merely a guard but the decoder's width."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["advanced_config"]["trainer"]["precision"] == "32-true"
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    vae = tiny["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]
    assert vae["sequence_length"] == 300
    assert vae["raw_per_step"] == 16
    assert vae["horizon"] == 30
    assert vae["causal_reach_budget_s"] == shipped_vae["causal_reach_budget_s"]
    # The conv stem and the source window stay real: the stem's reach and the source encoder's
    # receptive-field bound are the architecture, not a size.
    assert vae["encoder_conv_kernels"] == shipped_vae["encoder_conv_kernels"]
    assert vae["encoder_conv_dilations"] == shipped_vae["encoder_conv_dilations"]
    assert vae["encoder_num_heads"] == shipped_vae["encoder_num_heads"]
    assert vae["source_attention_window"] == shipped_vae["source_attention_window"]
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


def test_the_tiny_geometry_satisfies_both_independent_head_constraints(tiny):
    """Caught here rather than as a ``ValueError`` three steps into the smoke run, and the two
    constraints are genuinely independent: the constructor validates
    ``num_heads * d_head == d_model`` for the **lag-attention** heads, while ``encoder_num_heads`` is
    documented as unrelated to ``num_heads`` and carries its own requirement that
    ``d_model / encoder_num_heads`` be even, because rotary position encoding rotates coordinate
    pairs. The two products coincide in the shipped config only because both head counts happen to be
    $4$; a variant that shrank ``d_head`` while treating them as one constraint raises at
    construction."""
    vae = tiny["model_config"]["VAE_model"]

    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior
    assert (vae["d_model"] // vae["encoder_num_heads"]) % 2 == 0
    assert vae["warmup_period"] < vae["sequence_length"] - vae["horizon"]


def test_the_tiny_variant_points_at_the_committed_shard(tiny):
    """This package commits no binary fixtures; every model in the grid reads the same shards."""
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
    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnTrfFs(**_model_kwargs_from(load_config(str(_TINY))))
    # The smoke model is small everywhere except where it must not be: the decoder still emits the
    # production width, because that follows the budget rather than any width the variant shrinks.
    assert model.d_model == 32
    assert model.decoder_out_channels == SHIPPED_TARGET_CHANNELS
    # And the source encoder's bound is still strictly inside the sequence at the shrunken depth, so
    # the bounded-source asymmetry is exercised rather than clamped away.
    assert model.source_encoder.receptive_field == 21 + 2 * (16 - 1)
    assert model.source_encoder.receptive_field < model.sequence_length


# --------------------------------------------------------------------------------------
# The dev-box validation variant
# --------------------------------------------------------------------------------------
def test_the_local_delta_is_exactly_the_declared_key_list(smoke_hie, shipped):
    """Both directions, and it matters more here than for ``tiny.yaml``: this is the variant whose
    numbers get quoted and the one the two loss-scale constants are derived from, so an undeclared
    delta is a difference between the model that was read and the model that ships, attributed to
    neither."""
    differing = _differing_paths(dict(_leaves(smoke_hie)), dict(_leaves(shipped)))

    assert differing == set(SMOKE_HIE_DELTA_PATHS)


def test_the_local_variant_inherits_everything_the_reading_depends_on(smoke_hie, shipped):
    """Only the run's *scale* is local. Every quantity the pre-registered criteria are read from --
    the block the NLL sums over, the weights it is balanced against, the clamp the log-variances sit
    inside, the encoder that produces the history states, and the breaker that could silently replace
    the loss with its own EMA -- is inherited, or the reading is of a different model."""
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
    # The encoders, in full: this is the file the encoder axis is read from locally.
    for key in ENCODER_KEYS:
        assert vae[key] == shipped_vae[key], key
    assert (
        smoke_hie["advanced_config"]["trainer"]["gradient_clip_val"]
        == shipped["advanced_config"]["trainer"]["gradient_clip_val"]
    )
    assert smoke_hie["advanced_config"]["spike_breaker"] == shipped["advanced_config"][
        "spike_breaker"
    ]


def test_the_local_variant_scales_both_schedule_lengths_into_its_own_budget(smoke_hie):
    """The two model-scale deltas, and the only two that cannot be inherited: a length that is a
    fraction of a production run is a large fraction of this one.

    The beta ramp must reach its endpoint early enough that the coupling columns are read off a model
    that has been paying its rate for most of the run. The step ramp matters for a second reason: the
    shipped $2000$ steps is $91\\%$ of this run's $\\approx 2{,}200$, so inherited it would leave the
    learning rate still ramping when the criteria are read -- and the pre-clip gradient norms the clip
    is re-derived from would be drawn almost entirely from a model frozen near its initialisation."""
    warmup_epochs = smoke_hie["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
    epochs = smoke_hie["general_config"]["epochs"]
    batch = smoke_hie["general_config"]["batch_size"]["train"]
    warmup_steps = smoke_hie["general_config"]["lr_warmup_steps"]

    assert warmup_epochs * 10 <= epochs
    # 339 windows in the committed shard, so the optimizer-step total is epochs * ceil(339 / batch).
    total_steps = epochs * -(-339 // batch)
    assert 0 < warmup_steps <= total_steps // 5


def test_the_local_variant_matches_the_encoder_siblings_local_variant_outside_the_encoder(
    smoke_hie,
):
    """The production pins' argument, applied where the comparison is actually run. The two dev-box
    configs differ in the twelve encoder keys, the step ramp neither conv-LSTM model has, and
    identity -- so the two local runs the encoder axis is read from differ in the encoder and nothing
    else, at the same epoch and step budget.

    Written against the same allow-list Pin 2 uses, including the pending-measurement key: the
    instrumented run this file configures is what re-derives it, so pinning it here would be the same
    circularity one level down."""
    sibling = load_config(str(_SIBLING_SMOKE_HIE))

    mine = dict(_leaves(smoke_hie))
    theirs = dict(_leaves(sibling))
    compared = (
        (set(mine) | set(theirs))
        - set(SCHEMA_ONLY_PATHS)
        - set(TARGET_EDGE_EXEMPT_PATHS)
    )

    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(compared)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert drift == {}, (
        f"the two dev-box configs differ outside the encoder keys and the declared allow-list, so "
        f"the local encoder comparison is confounded: {drift}"
    )


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


def test_the_local_variant_normalizes_and_loads_both_target_blocks(smoke_hie):
    """Inherited, and asserted anyway: the entry point's guard runs against this driver's
    ``TARGET_FIELDS``, and this is the config that reaches it on the one shard whose statistics are
    not the fixture's."""
    from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer

    dataloader = smoke_hie["dataset_config"]["dataloader_config"]

    for field in LagAttnTrfFsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


def test_the_resolved_local_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver -- and built at the
    production widths and the production decoder width, which is the whole reason this variant does
    not shrink the model."""
    resolved = resolve_config_file(str(_SMOKE_HIE), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnTrfFs(**_model_kwargs_from(load_config(str(_SMOKE_HIE))))
    assert model.d_model == 128
    assert len(model.target_encoder.attention_blocks) == 6
    assert model.horizon * model.decoder_out_channels == 2340
