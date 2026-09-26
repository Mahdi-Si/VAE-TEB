r"""The shipped configs load, validate, build, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the conv-LSTM cell of this row's,
and the price of that is drift -- drift in a key that has nothing to do with the encoder is exactly
what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to the encoder by
every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison config's
value against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence means declaring it there. The
comparison is total rather than schema-limited: the two models compose the *same* input mixin over
two architectures, so every key present in both files means the same thing in both.

The exemptions fall into four kinds, and the split is the record rather than bookkeeping: the
**identity** keys (run tag, output directory, MLflow experiment, run name and variant tag), whose
inheritance would write these runs into the other model's tree; the **encoder** keys this
architecture removes and adds, which are the whole declared content of the edge; the encoder's
optimisation key ``lr_warmup_steps``; and one **measurement**, ``gradient_clip_val``, which is stated
in units of the summed block and was re-measured on this encoder rather than argued
(``tests/test_spike_breaker.py`` brackets it against the distribution it came from).

The two variants that inherit ``default.yaml`` are held to it the same way: each may differ only in
the leaves its delta list declares, so a smoke run cannot silently stop resembling the production one
and the instrumented run cannot measure a model other than the one its constant will guard.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from train.test_utils import make_graph_model

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_CAUSAL = _CONFIG_DIR / "smoke_causal.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_crws" / "configs" / "default.yaml"

#: ``VAE_model`` keys that name no constructor argument and are still real: the experiment driver
#: and the task read each of them by name. ``causal_warmup_budget_steps`` is *translated* rather
#: than forwarded -- it resolves against the configured shards into the four channel tuples the net
#: takes.
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
    "causal_warmup_budget_steps",
    # Both alignment keys are resolved against the SHARDS by the trainer and reach the
    # constructor only as the two shift tuples, so neither names a constructor argument.
    "causal_align_reference",
    "causal_leg_alignment",
)

#: The seven keys this architecture adds, with one reason: they are the encoder, and the encoder is
#: the entire declared content of this edge of the grid.
ENCODER_ADDED_REASON = (
    "ENCODER: the causal conv-Transformer stack -- gated causal depthwise conv stem, pre-normalised "
    "causal self-attention with RoPE -- which the conv-LSTM cell has no analogue of"
)

#: And the five it removes, with the reason each reaches nothing here.
ENCODER_REMOVED_REASON = (
    "ENCODER: no recurrent branch, no appended dilation schedule and no time-pooling normaliser "
    "left to causalise, so the key is not a constructor argument of this model at all"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else differing
#: is drift, and drift is a confound. No wildcards: the list names its contents so that adding an
#: entry is a decision rather than an omission.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the encoder",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
    "model_config.VAE_model.encoder_conv_kernels": ENCODER_ADDED_REASON,
    "model_config.VAE_model.encoder_conv_dilations": ENCODER_ADDED_REASON,
    "model_config.VAE_model.encoder_num_heads": ENCODER_ADDED_REASON,
    "model_config.VAE_model.encoder_d_ff": ENCODER_ADDED_REASON,
    "model_config.VAE_model.target_attention_blocks": ENCODER_ADDED_REASON,
    "model_config.VAE_model.source_attention_blocks": ENCODER_ADDED_REASON,
    "model_config.VAE_model.source_attention_window": ENCODER_ADDED_REASON,
    "model_config.VAE_model.lstm_layers": ENCODER_REMOVED_REASON,
    "model_config.VAE_model.encoder_extra_dilations": ENCODER_REMOVED_REASON,
    "model_config.VAE_model.encoder_extra_kernel": ENCODER_REMOVED_REASON,
    "model_config.VAE_model.conv_norm_groups": ENCODER_REMOVED_REASON,
    "model_config.VAE_model.causal_norm": (
        ENCODER_REMOVED_REASON
        + " -- and its absence is the one architectural claim this cell can make that the "
        "conv-LSTM one cannot: step-wise causality holds unconditionally, with no flag to get wrong"
    ),
    "general_config.lr_warmup_steps": (
        "ENCODER: the step-granular learning-rate ramp a pre-normalised attention stack needs in "
        "exactly its first few hundred updates, which an epoch-granularity schedule cannot express "
        "at all. It exists in every conv-Transformer sibling and in no conv-LSTM one"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "RETUNED: the block and the anchor count are unchanged across this edge, so nothing "
        "predicts this value and only the instrumented run says where it lands -- the smallest "
        "round value above the pre-clip norm's q99 on that run"
    ),
}

#: The exemptions that are mandatory rather than merely permitted: copying any of them writes this
#: model's runs into a comparison model's output tree and MLflow experiment.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.plot_frequency",
        "general_config.lr_warmup_steps",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "model_config.VAE_model.d_model",
        "model_config.VAE_model.d_z",
        "model_config.VAE_model.d_head",
        "model_config.VAE_model.max_lag",
        "model_config.VAE_model.dropout",
        "model_config.VAE_model.encoder_d_ff",
        "model_config.VAE_model.target_attention_blocks",
        "model_config.VAE_model.source_attention_blocks",
        "model_config.VAE_model.source_attention_window",
        "model_config.VAE_model.likelihood",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "advanced_config.tracking.mlflow.enabled",
    }
)

#: Every leaf of ``smoke_causal.yaml`` that resolves to something other than ``default.yaml``'s
#: value. Note what is **not** in it: every model width, the encoder block, the warm-up budget, the
#: anchor floor, the stride, the horizon, the beta pair and the whole spike-breaker block are
#: inherited -- the run has to be the shipped objective at the shipped widths or the distribution it
#: measures is another model's. Only the run's *scale* is local, plus the parked clip, which is the
#: reason it exists, plus the ramp length, which the run's own step budget forces.
SMOKE_CAUSAL_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.plot_frequency",
        "general_config.lr_warmup_steps",
        "general_config.batch_size.train",
        "general_config.batch_size.test",
        "general_config.folders_config.out_dir_base",
        "dataset_config.vae_train_datasets",
        "dataset_config.vae_test_datasets",
        "dataset_config.stat_path",
        "dataset_config.dataloader_config.num_workers",
        "advanced_config.trainer.gradient_clip_val",
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


def _differing(mine: dict, theirs: dict) -> set:
    """The dotted leaf paths at which two configs differ, a path absent on one side included."""
    mine, theirs = dict(_leaves(mine)), dict(_leaves(theirs))
    return {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }


def _model_kwargs_from(config: dict, trainer_cls, tmp_path) -> dict:
    """Run a config through the real driver's signature sweep and return the kwargs.

    The paths are absolutised first, because this driver's sweep **reads the shards**: the warm-up
    boundary is a property of the data and there is nothing to read it from otherwise.
    """
    path = Path(tmp_path) / "config.yaml"
    path.write_text(
        yaml.safe_dump(absolutize_dataset_paths(config), sort_keys=False), encoding="utf-8"
    )
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
def smoke_causal() -> dict:
    return load_config(str(_SMOKE_CAUSAL))


@pytest.fixture
def sibling() -> dict:
    return load_config(str(_SIBLING_CONFIG))


# --------------------------------------------------------------------------------------
# Every config loads, validates and builds, and everything in it reaches something
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path", sorted(_CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.name
)
def test_every_shipped_config_validates_with_no_unknown_or_dead_key_warnings(
    path, tmp_path, loguru_warnings
) -> None:
    """Drives the framework's real validator, not a copy of its rules. Globbed, so a config added
    later is checked without anyone having to remember it; resolved first, which is the only way a
    ``base:`` variant ever reaches the experiment driver -- and constructing the driver is also what
    reads the keys ``GraphModelBase.__init__`` indexes before the validator runs."""
    resolved = resolve_config_file(str(path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []


@pytest.mark.parametrize("path", [_TINY, _SMOKE_CAUSAL], ids=lambda path: path.name)
def test_a_config_on_the_committed_shards_builds_through_the_driver(path, tmp_path) -> None:
    """Config -> signature sweep -> budget resolved against the shards -> constructor.

    The shipped config's shard paths are deliberately non-existent placeholders and this resolution
    **reads the shards**, so the two variants on the committed fixture stand in; each inherits the
    shipped geometry and budget. The warm-up budget decides the input adapters' widths and nothing
    else: the decoder emits ``raw_per_step`` raw samples per horizon token, so no configuration and
    no budget can put the decoder and the target on different widths. And no loss-only key reaches
    the constructor, which is keyword-only with no ``**kwargs`` -- a leaked key would be a
    ``TypeError`` on the production config.
    """
    kwargs = _model_kwargs_from(load_config(str(path)), LagAttnTrfCrwsTrainer, tmp_path)
    model = SeqVaeLagAttnTrfCrws(**kwargs)

    assert not set(TASK_LEVEL_KEYS) & set(kwargs)
    assert model.target_adapter.linear.in_features == len(kwargs["target_keep_index"])
    assert model.source_adapter.linear.in_features == len(kwargs["source_keep_index"])
    assert model.decoder_out_channels == model.raw_per_step
    assert len(kwargs["target_keep_index"]) != model.raw_per_step


def test_every_vae_model_key_reaches_the_constructor_or_the_task(shipped) -> None:
    """A key that reaches nothing does not raise -- the constructor has a default for everything --
    so the run trains a *different architecture* than its config describes, and only a checkpoint
    that will not reload months later reveals it."""
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


def test_the_anchor_stride_equals_the_configured_horizon(shipped) -> None:
    """The two are one decision: below the horizon the forecast windows overlap again, above it
    there are target steps no phase ever covers. Asserted rather than defaulted, so a horizon change
    that left the stride behind fails here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"]


def test_the_plotting_block_is_the_one_the_inherited_driver_reads(shipped) -> None:
    """The callback assembly is inherited and reads the driver's ``PLOT_CONFIG_KEY``. Renaming the
    block to match this package would disable the per-epoch diagnostic figure with no error
    anywhere."""
    assert LagAttnTrfCrwsTrainer.PLOT_CONFIG_KEY in shipped["advanced_config"]["callbacks"]


def test_the_driver_loads_and_normalizes_every_field_it_reads(shipped) -> None:
    """``load_fields`` is honoured literally with no forced additions, so each field the driver
    reads is load-bearing. The raw target must also be normalized: an unnormalised one arrives at
    ~140 bpm and makes the Gaussian NLL meaningless with nothing else raising. ``weight`` is the only
    trustworthy gap signal for a raw target -- gaps are stored as 0 bpm, about -11 sigma after
    z-scoring -- and ``guid``/``epoch`` are what the tile phase is keyed on."""
    from teb_vae.lag_attn_crws.trainer import PHASE_KEY_FIELDS, WEIGHT_FIELD

    dataloader = shipped["dataset_config"]["dataloader_config"]
    load_fields = dataloader["dataset_kwargs"]["load_fields"]

    for field in LagAttnTrfCrwsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in load_fields, field
    for field in (WEIGHT_FIELD, *PHASE_KEY_FIELDS):
        assert field in load_fields, field


def test_the_driver_refuses_a_weighted_boundary_term(shipped) -> None:
    """The one shape weight that does not transfer from the raw-signal siblings: the term is a
    slicing identity over ADJACENT anchors, and this cell always decodes a tiled set. Refused by the
    causal parent's pre-flight, which is the half of the driver diamond this asserts resolves first
    -- the conv-Transformer parent's own pre-flight would accept it."""
    shipped["model_config"]["VAE_model"]["lambda_boundary"] = 0.5

    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnTrfCrwsTrainer.preflight(shipped)


# --------------------------------------------------------------------------------------
# Identity, and parity with the model this one is compared against
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", IDENTITY_PATHS)
def test_the_identity_keys_are_this_models_own(shipped, sibling, path) -> None:
    """Copying any of these writes this model's runs into the comparison model's output tree and
    mixes them into its MLflow experiment -- unrecoverable after the fact, because both runs are
    then indistinguishable by the only fields anything indexes on."""
    value = str(_get(shipped, path))

    assert value != str(_get(sibling, path))
    # Two spellings are accepted because two are shipped and both are this model's: the output tree
    # carries the package name, everything MLflow indexes on carries the abbreviated stem the
    # checkpoint filenames use. What is not accepted is a value naming neither.
    assert any(name in value for name in ("transformer_crws", "trf_crws", "trf-crws")), (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


def test_every_leaf_equals_the_comparison_configs_value(shipped, sibling) -> None:
    """The whole comparison rests on this, and here it is total rather than schema-limited: both
    models compose the same input mixin, so every key present in both means the same thing in
    both."""
    mine = dict(_leaves(shipped))
    theirs = dict(_leaves(sibling))
    drift = {
        path: (mine.get(path, "<absent>"), theirs.get(path, "<absent>"))
        for path in sorted(_differing(shipped, sibling) - set(PARITY_EXEMPT_PATHS))
    }

    assert drift == {}, (
        f"these leaves differ from the comparison config and are not declared in "
        f"PARITY_EXEMPT_PATHS: {drift}"
    )


def test_every_declared_parity_exemption_is_a_real_divergence(shipped, sibling) -> None:
    """The other direction: an exemption for a key that no longer differs is a permission that
    outlived its reason, and the next accidental divergence there would go unreported."""
    stale = sorted(set(PARITY_EXEMPT_PATHS) - _differing(shipped, sibling))

    assert stale == []


# --------------------------------------------------------------------------------------
# The two variants differ from the shipped config only where they declare
# --------------------------------------------------------------------------------------
def test_the_tiny_delta_is_exactly_the_declared_key_list(tiny, shipped) -> None:
    """Both directions: an undeclared delta is a smoke run that silently stops resembling the
    production one, and a declared delta that is not there is a stale declaration. A leftover
    ``base`` directive would show up here as an undeclared leaf."""
    assert _differing(tiny, shipped) == set(TINY_DELTA_PATHS)


def test_the_instrumented_delta_is_exactly_the_declared_key_list(smoke_causal, shipped) -> None:
    """Both directions, and it matters more here than for ``tiny.yaml``: this is the variant whose
    numbers become a shipped constant, so an undeclared delta is a threshold derived from a model
    that is not the one it will guard."""
    assert _differing(smoke_causal, shipped) == set(SMOKE_CAUSAL_DELTA_PATHS)


def test_the_instrumented_variant_measures_the_objective_rather_than_its_guards(
    smoke_causal, shipped
) -> None:
    """Three properties the recorded distribution is only valid under, each stated relative to the
    config it describes rather than as a number.

    The clip is parked far above anything reachable: a run measured under an active clip reports
    the gradient-norm distribution of an already-clipped optimizer, which is not the distribution a
    threshold should be set from. The step-granular ramp is scaled into the run's own step budget:
    the shipped ramp would outlast the run, and a distribution measured under a ramp is the ramp's.
    And the run outlasts the inherited beta ramp by a wide margin, or it describes that ramp.
    """
    general = smoke_causal["general_config"]
    ramp = general["lr_warmup_steps"]

    assert smoke_causal["advanced_config"]["trainer"]["gradient_clip_val"] >= 1.0e9
    assert 0 < ramp < shipped["general_config"]["lr_warmup_steps"]
    assert ramp <= general["epochs"] // 4
    assert general["epochs"] >= 10 * smoke_causal["model_config"]["VAE_model"][
        "beta_schedule"
    ]["warmup_epochs"]


def test_the_instrumented_variant_reads_the_conv_lstm_cells_measurement_data(smoke_causal) -> None:
    """The same file the conv-LSTM cell of this row measured its own constants on: the encoder edge
    is only readable if both cells are read on the same data. In-sample, deliberately."""
    dataset = smoke_causal["dataset_config"]
    sibling_smoke = load_config(str(_SIBLING_CONFIG.parent / "smoke_causal.yaml"))

    assert dataset["vae_train_datasets"] == dataset["vae_test_datasets"]
    assert (
        dataset["vae_train_datasets"] == sibling_smoke["dataset_config"]["vae_train_datasets"]
    )
