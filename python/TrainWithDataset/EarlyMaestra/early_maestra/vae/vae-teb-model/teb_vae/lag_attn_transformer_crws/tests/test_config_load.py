r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the conv-LSTM cell of this row's,
and the price of that is drift -- drift in a key that has nothing to do with the encoder is exactly
what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to the encoder by
every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison config's
value against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence means declaring it there. The
comparison is total rather than schema-limited: the two models compose the *same* input mixin over
two architectures, so every key present in both files means the same thing in both.

The nineteen exemptions fall into four kinds, and the split is the record rather than bookkeeping.
**Five are identity** -- the run tag, the output directory, the MLflow experiment, the run name and
the variant tag -- and inheriting any of them writes these runs into the other model's tree.
**Twelve are the encoder**: the five conv-LSTM keys this architecture does not have and the seven it
adds, which is the whole declared content of the edge. **One is the encoder's optimisation**:
``lr_warmup_steps``, which exists in every conv-Transformer sibling and in no conv-LSTM one.

**One is a measurement**, and it is the one worth reading twice. ``gradient_clip_val`` is stated in
units of the summed block, and the encoder edge leaves the block and the anchor count untouched -- so
unlike across the target axis there is no arithmetic that would predict it, and the only way to know
whether it moves is to measure it. It did, and it is declared in :data:`PARITY_EXEMPT_PATHS` with a
reason beginning ``RETUNED`` and listed again in :data:`RETUNED_PATHS`.

The loss-scale constants that were re-derived and came back to parity live in
:data:`MEASURED_TO_MATCH_PATHS`, whose test asserts the equality so that it reads as a measurement
rather than as an oversight. ``additive_margin`` is the one that matters there: a faster-fitting
encoder could in principle produce larger excursions above the breaker's own EMA, and only the
instrumented run says it does not.
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
from train.test_utils import make_graph_model

from .conftest import CAUSAL_C_U, CAUSAL_C_Y, CONV_LSTM_ONLY_KEYS, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_CAUSAL = _CONFIG_DIR / "smoke_causal.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_crws" / "configs" / "default.yaml"

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
#: is drift, and drift is a confound. Nineteen entries and no wildcards: the list names its contents
#: so that adding a twentieth is a decision rather than an omission.
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

#: The exemptions that were **measured** rather than merely permitted. Named as a separate list
#: because the numbers behind them live in one place, ``tests/test_spike_breaker.py``, where each is
#: bracketed against the distribution it came from.
RETUNED_PATHS = ("advanced_config.trainer.gradient_clip_val",)

#: The loss-scale constants that were re-derived on this encoder and landed on the comparison
#: model's value. Named so that equality stays visible as a *measurement* rather than reading as an
#: oversight, and so a future retune has to remove a key from here deliberately.
#:
#: ``additive_margin`` is the one that had to be measured rather than argued: the margin is stated
#: in nats of the summed block and this edge moves neither the block nor the anchor count, but a
#: faster-fitting encoder could still produce larger excursions above the breaker's own EMA, and
#: only a run says whether it does.
MEASURED_TO_MATCH_PATHS = (
    "advanced_config.spike_breaker.additive_margin",
    "advanced_config.spike_breaker.ema_floor",
    "model_config.VAE_model.horizon_embed_std",
)

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
# The shipped config loads and everything in it reaches something
# --------------------------------------------------------------------------------------
def test_every_effectively_required_key_is_present(shipped) -> None:
    missing = [path for path in _REQUIRED_PATHS if not _has(shipped, path)]
    assert missing == []


def test_the_shipped_config_validates_with_no_unknown_or_dead_key_warnings(
    tmp_path, loguru_warnings
) -> None:
    """Drives the framework's real validator, not a copy of its rules."""
    graph_model = make_graph_model(
        _CONFIG, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []


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


@pytest.mark.parametrize("key", CONV_LSTM_ONLY_KEYS)
def test_no_config_names_a_conv_lstm_key(key) -> None:
    """Each names a component this architecture does not have, so each would be dropped by the
    signature sweep and reach nothing at all -- silently. Globbed over every YAML rather than checked
    on the shipped file alone: a config added later is exactly the one nobody would think to check.
    """
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        settings = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert not any(key in line for line in settings), f"{path.name} names {key}"


def test_a_conv_lstm_key_is_refused_by_the_constructor_it_would_reach(tmp_path) -> None:
    """The other direction, and the one that makes the absence above a guard rather than a habit: a
    key that *did* survive the sweep would raise at the constructor, naming itself."""
    import torch

    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnTrfCrwsTrainer, tmp_path)
    torch.manual_seed(0)

    with pytest.raises(TypeError, match="lstm_layers"):
        SeqVaeLagAttnTrfCrws(**dict(kwargs, lstm_layers=2))


def test_loss_only_keys_do_not_reach_the_constructor(tmp_path) -> None:
    """The net takes tensors and computes a loss on request; it owns none of these. The constructor
    is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on the production
    config -- a poor place to find out."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnTrfCrwsTrainer, tmp_path)

    for name in TASK_LEVEL_KEYS:
        assert name not in kwargs, f"{name} is not the net's"


def test_the_shipped_geometry_pairs_the_floor_with_the_budget(shipped) -> None:
    r"""One decision, two keys. $F \ge B - 1$ is the declared input-warmth policy -- every kept input
    channel warm by the first forecast step -- and the configuration exposes the pair so the floor
    may exceed the minimum rather than being derived from the threshold."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["warmup_period"] == 133 == vae["causal_warmup_budget_steps"] - 1
    assert vae["c_y"] == CAUSAL_C_Y
    assert vae["c_u"] == CAUSAL_C_U


def test_the_anchor_stride_equals_the_configured_horizon(shipped) -> None:
    """The two are one decision: below the horizon the forecast windows overlap again, above it
    there are target steps no phase ever covers. Asserted rather than defaulted, so a horizon change
    that left the stride behind fails here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"] == 15


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_raw_grid(tmp_path) -> None:
    """The one binding this cell does **not** make, asserted so its absence is a passing test rather
    than a comment: the warm-up budget decides the input adapters' widths and nothing else. The
    decoder emits ``raw_per_step`` raw samples per horizon token, so the block is H * R and no
    configuration -- and no budget -- can put the decoder and the target on different widths.

    Driven on the tiny variant because the shipped config's shard paths are deliberately
    non-existent placeholders and this resolution **reads the shards**; the tiny variant carries the
    identical geometry, which is exactly why it does."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnTrfCrwsTrainer, tmp_path)
    model = SeqVaeLagAttnTrfCrws(**kwargs)

    assert len(kwargs["target_keep_index"]) == 98  # the budget's reach: the INPUTS, not the target
    assert model.decoder_out_channels == model.raw_per_step == 16
    assert model.horizon * model.geometry.r == 240


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


def test_every_declared_parity_exemption_is_a_real_divergence(shipped, sibling) -> None:
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


def test_the_encoder_edge_is_exactly_the_twelve_keys_it_claims(shipped, sibling) -> None:
    """The square's own premise, as a set rather than as prose: the seven this architecture adds are
    present here and absent there, and the five it removes are the other way round."""
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    added = {
        "encoder_conv_kernels", "encoder_conv_dilations", "encoder_num_heads", "encoder_d_ff",
        "target_attention_blocks", "source_attention_blocks", "source_attention_window",
    }
    assert added <= set(mine)
    assert added.isdisjoint(theirs)
    assert set(CONV_LSTM_ONLY_KEYS) <= set(theirs)
    assert set(CONV_LSTM_ONLY_KEYS).isdisjoint(mine)
    # And nothing else in the model block differs, which is what makes the edge readable.
    assert {key for key in set(mine) & set(theirs) if mine[key] != theirs[key]} == set()


@pytest.mark.parametrize("path", RETUNED_PATHS)
def test_each_retuned_constant_is_declared_and_really_moved(shipped, sibling, path) -> None:
    """The direction is not asserted here, and the omission is the record. Across the *target* axis
    the block halves, so a threshold that came back larger would be describing a model with more to
    clip; across the ENCODER edge the block is unchanged and there is no arithmetic that predicts a
    direction at all. What the measurement said is in the config comment and in
    ``tests/test_spike_breaker.py``, where the value is bracketed against the distribution it came
    from."""
    assert path in PARITY_EXEMPT_PATHS
    assert PARITY_EXEMPT_PATHS[path].startswith("RETUNED")
    assert float(_get(shipped, path)) != float(_get(sibling, path))


def test_the_instrumented_config_is_the_record_of_where_that_constant_came_from() -> None:
    """The re-derivation names a measurement, and the measurement names a config. Asserted together
    so a file deleted as unused takes the only reproducible route to that number with it."""
    assert _SMOKE_CAUSAL.is_file()
    text = _SMOKE_CAUSAL.read_text(encoding="utf-8")

    assert "gradient_clip_val" in text and "additive_margin" in text


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_PATHS)
def test_each_measured_constant_landed_on_the_comparison_models_value(
    shipped, sibling, path
) -> None:
    r"""The three loss-scale constants whose value is demonstrably insensitive to what this edge
    moves. The assertion exists so the equality stays visible as a **measurement** rather than
    reading as an oversight, and so a future retune has to leave
    :data:`MEASURED_TO_MATCH_PATHS` deliberately.

    ``additive_margin`` is stated in nats of the summed block, and the encoder edge changes neither
    the block nor the anchor count -- but a faster-fitting encoder could still produce larger
    excursions above the breaker's own EMA, so this one was re-measured rather than argued.

    ``ema_floor`` disables the relative spike test outright, and the property it needs is that it
    sits far above any *reachable* loss, which is a property of the clamp and the block rather than
    of an encoder. ``horizon_embed_std`` is chosen against the post-initialisation correlation
    between two horizon tokens inside a decoder core both cells reach by import.
    """
    assert float(_get(shipped, path)) == float(_get(sibling, path))
    assert path not in PARITY_EXEMPT_PATHS


def test_no_path_is_both_exempt_and_measured_to_match() -> None:
    """The two lists make opposite claims about the same key, so an overlap is a contradiction the
    parity assertions above would resolve arbitrarily."""
    assert set(MEASURED_TO_MATCH_PATHS).isdisjoint(PARITY_EXEMPT_PATHS)
    assert set(RETUNED_PATHS) <= set(PARITY_EXEMPT_PATHS)
    assert set(IDENTITY_PATHS) <= set(PARITY_EXEMPT_PATHS) | {"general_config.tag"}


def test_the_two_weights_hold_the_anchor_ratio_the_design_fixes(shipped, sibling) -> None:
    r"""$\beta_p$ moves with $\beta$, whatever $\beta$ is. The ratio is the invariant, not either
    value: the anchor's restoring force saturates at $\beta_p / 2$ per latent dimension while the
    reconstruction it opposes is what an encoder change moves."""
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    assert mine["beta_prior"] / mine["beta_schedule"]["end"] == pytest.approx(
        theirs["beta_prior"] / theirs["beta_schedule"]["end"]
    )
    assert mine["beta_schedule"]["start"] == 0.0


def test_the_plotting_block_keeps_the_inherited_drivers_spelling(shipped) -> None:
    """The callback assembly is inherited and reads this literal. Renaming the block to match this
    package would disable the per-epoch diagnostic figure with no error anywhere."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True
    assert LagAttnTrfCrwsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_precision_is_float32(shipped) -> None:
    """Mixed precision would need float32 islands around the log-variances, the closed-form KL and
    the NLL reduction."""
    assert shipped["advanced_config"]["trainer"]["precision"] == "32-true"


def test_compile_is_off_and_is_live_rather_than_inert_for_this_net(shipped) -> None:
    """The one place a config key means something different here than on the conv-LSTM cell of this
    row. There the driver refuses compilation outright; here it is permitted and the shipped value
    is a decision rather than a constraint, so the key has to be read as one."""
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    assert shipped["advanced_config"]["trainer"]["compile"] is False
    assert LagAttnCrwsTrainer.compile_model_requested(object()) is False
    assert (
        LagAttnTrfCrwsTrainer.compile_model_requested
        is not LagAttnCrwsTrainer.compile_model_requested
    )


def test_num_sanity_val_steps_is_zero(shipped) -> None:
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_the_beta_warmup_starts_at_exactly_zero(shipped) -> None:
    """z is the only route to the decoder; a nonzero beta before the decoder can use the latent is
    the standard route to posterior collapse."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["warmup_epochs"] == 50


def test_the_step_granular_ramp_is_configured_and_positive(shipped) -> None:
    """The one non-encoder key on the encoder edge, and the reason it exists: a pre-normalised
    attention stack is fragile in exactly its first few hundred updates, and the framework's
    epoch-granularity path cannot address that at all -- it additionally rejects ``start_factor=0``,
    so it cannot ramp from zero."""
    assert shipped["general_config"]["lr_warmup_steps"] > 0


def test_the_breaker_is_pinned_by_value_rather_than_by_presence(shipped) -> None:
    r"""``max(EMA, ema_floor)`` collapses to the floor once the EMA is negative -- and this
    objective, averaged over roughly ten anchors per step, goes negative early -- so the floor sits
    far above any reachable loss, disabling the relative test outright, while ``additive_margin``
    compares against the *raw* EMA and keeps working there."""
    breaker = shipped["advanced_config"]["spike_breaker"]

    assert breaker["enabled"] is True  # the non-finite guard is the point
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9
    assert breaker["additive_margin"] > 0.0  # finite-spike detection; 0 would disable it
    assert breaker["max_consecutive_skips"] > 0  # the deadlock escape hatch


def test_the_boundary_shape_weight_is_zero_and_the_driver_refuses_any_other_value(shipped) -> None:
    """The one shape weight that does not transfer from the raw-signal siblings, and the refusal is
    specific to this input domain: the term is a slicing identity over ADJACENT anchors, and this
    cell always decodes a tiled set."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    assert shipped["model_config"]["VAE_model"]["lambda_boundary"] == 0.0

    broken = load_config(str(_CONFIG))
    broken["model_config"]["VAE_model"]["lambda_boundary"] = 0.5
    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnTrfCrwsTrainer.preflight(broken)


def test_the_two_within_block_shape_weights_transfer_unchanged(shipped, sibling) -> None:
    """The other two shape terms are raw-waveform quantities over one anchor's own block, so neither
    the anchor axis nor the encoder touches them."""
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    assert mine["lambda_ms"] == theirs["lambda_ms"] == 0.1
    assert mine["lambda_deriv"] == theirs["lambda_deriv"] == 0.1


def test_the_cross_channel_block_appears_in_no_config() -> None:
    """``fhr_up_ph`` mixes both signals in one coefficient, and the causal variant does not store it
    at all. Globbed rather than listed: a config added later is exactly the one nobody would think
    to add to a list here."""
    configs = sorted(_CONFIG_DIR.glob("*.yaml"))

    assert configs, "the config directory is empty; the glob is checking nothing"
    for path in configs:
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_the_validity_mask_emitter_appears_in_no_config() -> None:
    """``emit_validity_mask`` would pay for a per-sample copy of a filter-bank constant, per worker,
    per collate and per host-to-device copy -- buying nothing a single resolved vector in
    ``model_kwargs`` does not already give, and that vector is also what reaches the checkpoint."""
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        assert "emit_validity_mask" not in path.read_text(encoding="utf-8"), path.name


def test_the_raw_target_is_loaded_and_normalized(shipped) -> None:
    """An unnormalised raw target arrives at ~140 bpm and makes the Gaussian NLL meaningless with
    nothing else raising."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    dataloader = shipped["dataset_config"]["dataloader_config"]

    for field in LagAttnTrfCrwsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


def test_the_validity_signal_and_the_two_phase_key_fields_are_loaded(shipped) -> None:
    """``weight`` is the only trustworthy gap signal for a raw target -- gaps are stored as 0 bpm,
    about -11 sigma after z-scoring -- and ``guid``/``epoch`` are what the tile phase is keyed on.
    ``load_fields`` is honoured literally with no forced additions, so each is load-bearing."""
    from teb_vae.lag_attn_crws.trainer import PHASE_KEY_FIELDS, WEIGHT_FIELD

    load_fields = shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]

    assert WEIGHT_FIELD in load_fields
    for field in PHASE_KEY_FIELDS:
        assert field in load_fields, field


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


def test_the_tiny_variant_names_only_its_deltas() -> None:
    """The whole point of the ``base:`` mechanism, counted on the raw file: a variant that re-stated
    the inherited settings would be a copy that drifts. Budgeted against the conv-Transformer
    causal-feature cell's own tiny variant, which shrinks the same encoder."""
    sibling_tiny = (
        _REPO_ROOT / "teb_vae" / "lag_attn_transformer_cfs" / "configs" / "tiny.yaml"
    )

    mine = _non_comment_lines(_TINY)

    assert mine <= _non_comment_lines(sibling_tiny) + 1, (
        f"tiny.yaml carries {mine} non-comment lines against the comparison variant's "
        f"{_non_comment_lines(sibling_tiny)}; it should be deltas only"
    )


def test_the_base_key_never_reaches_the_validator(tiny) -> None:
    """``load_config`` consumes ``base:``; were it left in, ``validate_config`` would warn on an
    unknown key and the MLflow param dump would carry a loader directive."""
    assert "base" not in tiny


def test_the_tiny_delta_is_exactly_the_declared_key_list(tiny, shipped) -> None:
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


def test_the_tiny_variant_inherits_the_geometry_that_decides_what_it_exercises(
    tiny, shipped
) -> None:
    """Including every correctness requirement, and -- unlike the two-sided cells' smoke variants --
    including the warm-up budget, the anchor floor and the stride: the decoded anchor count IS the
    resolved floor, stride and horizon, and the input adapters' widths ARE the resolved budget's
    survivor counts, so a shrunken variant would exercise an anchor set and an input stream the
    production run does not have."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["advanced_config"]["trainer"]["precision"] == "32-true"
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    vae = tiny["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]
    for key in (
        "sequence_length", "raw_per_step", "horizon", "warmup_period", "anchor_stride",
        "lag_floor", "c_y", "c_u", "causal_warmup_budget_steps", "causal_reach_budget_s",
        "encoder_conv_kernels", "encoder_conv_dilations", "encoder_num_heads",
        "lambda_boundary",
    ):
        assert vae[key] == shipped_vae[key], key
    assert (
        tiny["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
        == shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["trim_minutes"]
    )


def test_the_tiny_variant_overrides_what_a_local_smoke_run_needs(tiny) -> None:
    assert tiny["general_config"]["epochs"] == 1
    assert tiny["general_config"]["cuda_devices"] == [0]
    assert tiny["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
    assert tiny["dataset_config"]["dataloader_config"]["num_workers"] == 0
    # The deliberate delta: mse starves the decoder logvar heads, so the smoke run exercises the
    # configuration whose DDP strategy is the fallback.
    assert tiny["model_config"]["VAE_model"]["likelihood"] == "mse"
    # And the ramp is shortened rather than switched off: a one-epoch run over four windows is a
    # handful of optimizer steps, and the step-granular schedule is exactly what this architecture's
    # smoke run should exercise rather than skip.
    assert 0 < tiny["general_config"]["lr_warmup_steps"] < 10


def test_the_tiny_geometry_satisfies_the_constructor_invariants(tiny) -> None:
    """Caught here rather than as a ``ValueError`` three steps into the smoke run."""
    vae = tiny["model_config"]["VAE_model"]

    assert vae["num_heads"] * vae["d_head"] == vae["d_model"]
    assert vae["d_z"] % vae["num_heads"] == 0  # required by the head-structured posterior
    assert vae["d_model"] % vae["encoder_num_heads"] == 0
    assert (vae["d_model"] // vae["encoder_num_heads"]) % 2 == 0  # rotary position encoding
    assert vae["warmup_period"] < vae["sequence_length"] - vae["horizon"]
    # And the tiling's own feasibility: the last phase's first anchor must exist.
    assert vae["warmup_period"] + vae["anchor_stride"] <= (
        vae["sequence_length"] - vae["horizon"]
    )


def test_the_tiny_variant_points_at_the_committed_causal_shard(tiny) -> None:
    """This package commits no binary fixtures; every model in the family reads shards owned by
    ``lag_attn``. The CAUSAL pair, not the two-sided one beside it."""
    for path in (
        *tiny["dataset_config"]["vae_train_datasets"],
        *tiny["dataset_config"]["vae_test_datasets"],
        tiny["dataset_config"]["stat_path"],
    ):
        assert "causal" in path, f"{path} is not the causal fixture"
        assert (_REPO_ROOT / path).is_file(), (
            f"{path} is missing; the committed fixture shards moved or were deleted"
        )


def test_the_resolved_tiny_variant_validates_and_builds(tmp_path, loguru_warnings) -> None:
    """Resolved first, which is the only way it ever reaches the experiment driver."""
    from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnTrfCrws(
        **_model_kwargs_from(load_config(str(_TINY)), LagAttnTrfCrwsTrainer, tmp_path)
    )
    # The smoke model is small everywhere except where it must not be: the input adapters still read
    # the production channel set and the forward still decodes the production tile count.
    assert model.d_model == 32
    assert model.target_adapter.linear.in_features == 98
    assert model.anchor_stride == 15


# --------------------------------------------------------------------------------------
# The instrumented variant the retuned constant is measured from
# --------------------------------------------------------------------------------------
def test_the_instrumented_variant_resolves_with_its_base_consumed(smoke_causal) -> None:
    """A leftover ``base`` key would reach ``validate_config`` as an unknown key and the resolved
    config written beside the run's checkpoints -- its provenance record -- would carry a loader
    directive rather than a setting."""
    assert "base" not in smoke_causal


def test_the_instrumented_delta_is_exactly_the_declared_key_list(smoke_causal, shipped) -> None:
    """Both directions, and it matters more here than for ``tiny.yaml``: this is the variant whose
    numbers become a shipped constant, so an undeclared delta is a threshold derived from a model
    that is not the one it will guard."""
    mine = dict(_leaves(smoke_causal))
    theirs = dict(_leaves(shipped))
    differing = {
        path
        for path in set(mine) | set(theirs)
        if mine.get(path, "<absent>") != theirs.get(path, "<absent>")
    }

    assert differing == set(SMOKE_CAUSAL_DELTA_PATHS)


def test_the_instrumented_variant_inherits_everything_the_measurement_depends_on(
    smoke_causal, shipped
) -> None:
    """Only the run's *scale*, the parked clip and the ramp length are local. Every quantity the
    constants are stated in -- the block the NLL sums over, the anchor set it is averaged over, the
    encoder that produces the gradients, the weights it is balanced against, the clamp the
    log-variances sit inside, and the breaker whose own EMA is the statistic -- is inherited, or the
    measurement is of a different model."""
    vae = smoke_causal["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_warmup_budget_steps", "warmup_period", "anchor_stride", "horizon", "raw_per_step",
        "beta_prior", "logvar_clamp", "d_z", "d_model", "lambda_ms", "lambda_deriv",
        "encoder_conv_kernels", "encoder_conv_dilations", "encoder_num_heads", "encoder_d_ff",
        "target_attention_blocks", "source_attention_blocks", "source_attention_window",
    ):
        assert vae[key] == shipped_vae[key], key
    assert vae["beta_schedule"] == shipped_vae["beta_schedule"]
    assert vae["likelihood"] == "gaussian_nll"  # the learned-variance heads, not the debug path
    assert (
        smoke_causal["advanced_config"]["spike_breaker"]
        == shipped["advanced_config"]["spike_breaker"]
    )


def test_the_instrumented_variant_parks_the_clip_far_above_anything_reachable(
    smoke_causal,
) -> None:
    """The reason the file exists. A run measured under an active clip reports the gradient-norm
    distribution of an already-clipped optimizer, which is not the distribution a threshold should be
    set from -- and the value being replaced is precisely the one doing the clipping."""
    assert smoke_causal["advanced_config"]["trainer"]["gradient_clip_val"] >= 1.0e9


def test_the_instrumented_variant_scales_the_ramp_into_its_own_step_budget(
    smoke_causal, shipped
) -> None:
    """The shipped ramp outlasts this whole run, and a gradient-norm distribution measured under a
    ramp is the ramp's distribution rather than the objective's."""
    ramp = smoke_causal["general_config"]["lr_warmup_steps"]

    assert 0 < ramp < shipped["general_config"]["lr_warmup_steps"]
    assert ramp <= smoke_causal["general_config"]["epochs"] // 4


def test_the_instrumented_variant_states_its_step_count_and_reaches_it(smoke_causal) -> None:
    """The recorded percentiles are meaningless without the sample size behind them, and here the
    sample size is the epoch count: the committed fixture holds four windows and the batch takes all
    four, so one epoch is exactly one optimizer step."""
    general = smoke_causal["general_config"]

    assert general["epochs"] == 600
    assert general["batch_size"]["train"] == 4
    # The beta ramp is inherited, so the measurement has to outlast it by a wide margin or it
    # describes the ramp rather than the objective.
    assert general["epochs"] >= 10 * smoke_causal["model_config"]["VAE_model"][
        "beta_schedule"
    ]["warmup_epochs"]
    assert "600 optimizer steps" in _SMOKE_CAUSAL.read_text(encoding="utf-8")


def test_the_instrumented_variant_points_at_the_committed_causal_shard(smoke_causal) -> None:
    """The same file the conv-LSTM cell of this row measured its own constants on: the encoder edge
    is only readable if both cells are read on the same data."""
    dataset = smoke_causal["dataset_config"]
    sibling_smoke = load_config(str(_SIBLING_CONFIG.parent / "smoke_causal.yaml"))

    assert dataset["vae_train_datasets"] == dataset["vae_test_datasets"]  # in-sample, deliberately
    assert (
        dataset["vae_train_datasets"] == sibling_smoke["dataset_config"]["vae_train_datasets"]
    )
    for path in (*dataset["vae_train_datasets"], dataset["stat_path"]):
        assert "causal" in path, path
        assert (_REPO_ROOT / path).is_file(), path


def test_the_instrumented_variant_runs_on_one_device_with_tracking_off(smoke_causal) -> None:
    """A dev-box run: one GPU, and no MLflow server to log to. The run directory is the record."""
    assert smoke_causal["general_config"]["cuda_devices"] == [0]
    assert smoke_causal["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
