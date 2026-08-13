r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the two-sided feature model's,
and the price of that is drift -- drift in a key that has nothing to do with the transform is
exactly what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to one-sidedness by
every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison
config's value against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence means declaring it
there. The comparison is total rather than schema-limited: this model's net is a *subclass* of the
comparison model's whose constructor schema gains three keys and loses none, so every key means the
same thing in both files and every key is comparable.

The twenty exemptions fall into four kinds, and the split is the record rather than bookkeeping.
**Five are identity** -- the run tag, the output directory, the MLflow experiment, the run name and
the variant tag -- and inheriting any of them writes these runs into the other model's tree. **Four
are the dataset**: causal shards and the statistics accumulated from them, plus the two loader keys
the tile phase is derived from. **Eight are the geometry the transform forces**: the channel widths
the one-sided cascade leaves, the horizon, the anchor floor the warm-up budget pairs with, the three
keys that have no two-sided counterpart, and the reach budget this dataset makes undefined. **Three
are numbers re-derived at this objective's own scale**, and they are the ones worth reading twice:
of the four loss-scale constants measured, only two moved. ``ema_floor`` and ``horizon_embed_std``
landed on the comparison model's values and live in :data:`MEASURED_TO_MATCH_PATHS`, whose test
asserts the equality so that it reads as a measurement rather than as an oversight.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from train.test_utils import make_graph_model

from .conftest import CAUSAL_C_U, CAUSAL_C_Y, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"

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
#: takes, and here those tuples also decide the decoder's width.
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

#: Shared by the four geometry entries below, because it is one reason rather than four: the
#: one-sided cascade drops seven scattering channels per block at write time, the warm-up budget
#: drops four more from the target, and the anchor floor is what the surviving warm-ups force.
GEOMETRY_REASON = (
    "the one-sided cascade's own geometry: 36 + 66 target and 36 + 15 source channels against the "
    "two-sided 43 + 66 and 43 + 15, and an anchor floor of B - 1 over the surviving warm-ups "
    "rather than the model's own 30-step one"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else differing
#: is drift, and drift is a confound. Twenty entries and no wildcards: the list names its contents
#: so that adding a twenty-first is a decision rather than an omission.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the target domain",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
    "dataset_config.vae_train_datasets": "DATASET: the causal shards, not the two-sided ones",
    "dataset_config.vae_test_datasets": "DATASET: the causal shards, not the two-sided ones",
    "dataset_config.stat_path": (
        "DATASET: statistics accumulated from the causal shards EXCLUDING the warm-up region, "
        "which is what makes zero the channel mean over the region the model reads"
    ),
    "dataset_config.dataloader_config.dataset_kwargs.load_fields": (
        "DATASET: 'epoch' is added because the anchor tiling's per-segment phase is keyed on the "
        "segment's own start time as well as the recording identifier"
    ),
    "model_config.VAE_model.c_y": GEOMETRY_REASON,
    "model_config.VAE_model.c_u": GEOMETRY_REASON,
    "model_config.VAE_model.warmup_period": GEOMETRY_REASON,
    "model_config.VAE_model.horizon": (
        "one minute rather than two: a shorter horizon lengthens T_valid and buys back anchors, "
        "which is one of the two levers on the warm-up cost"
    ),
    "model_config.VAE_model.causal_reach_budget_s": (
        "null and REQUIRED null: the forward reach L95 is an energy quantile of a two-sided kernel, "
        "measured on a bank that did not produce these coefficients, and a delay is a shift"
    ),
    "model_config.VAE_model.causal_warmup_budget_steps": (
        "the guard this dataset needs, which has no two-sided counterpart at all"
    ),
    "model_config.VAE_model.anchor_stride": (
        "the anchor tiling, which has no two-sided counterpart: the shipped four decode every "
        "anchor, which is this key's inert value"
    ),
    "model_config.VAE_model.lag_floor": (
        "the lag validity floor, which has no two-sided counterpart; it ships at 0, where the lag "
        "mask is bitwise the comparison model's"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "RETUNED: re-derived from a 600-step instrumented run at this geometry, where q99 of the "
        "pre-clip norm is 5742 against the comparison model's 4421 -- this cell averages the same "
        "summed block over ~10.1 anchors per step rather than ~240"
    ),
    "advanced_config.spike_breaker.additive_margin": (
        "RETUNED: the margin is stated in nats of the summed block, and this block is 1470 "
        "coefficients against 2340; measured against the breaker's own excursion-above-EMA "
        "statistic rather than against the epoch-to-epoch movement"
    ),
}

#: The loss-scale constants that were re-derived at this objective's own scale and landed on the
#: comparison model's value. Named so that equality stays visible as a *measurement* rather than
#: reading as an oversight, and so a future retune has to remove a key from here deliberately.
MEASURED_TO_MATCH_PATHS = (
    "advanced_config.spike_breaker.ema_floor",
    "model_config.VAE_model.horizon_embed_std",
)

#: The exemptions that are mandatory rather than merely permitted: copying any of them writes this
#: model's runs into the comparison model's output tree and MLflow experiment.
IDENTITY_PATHS = tuple(
    path for path, reason in PARITY_EXEMPT_PATHS.items() if reason.startswith("IDENTITY")
) + ("general_config.tag",)

#: The exemptions that exist because a *number* had to move, as opposed to a name or a width.
RETUNED_PATHS = (
    "advanced_config.trainer.gradient_clip_val",
    "advanced_config.spike_breaker.additive_margin",
)

#: Every leaf of ``tiny.yaml`` that resolves to something other than ``default.yaml``'s value.
TINY_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
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
#: Note what is **not** in it: every model width, the warm-up budget, the anchor floor, the stride,
#: the clip, the spike breaker and ``beta_schedule.end`` are all inherited -- only the run's *scale*
#: is local. The one model-block entry is the beta ramp's length, which cannot be inherited because
#: 50 epochs is a hundredth of a production run and a quarter of this one.
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
    constructor_keys = set(inspect.signature(SeqVaeLagAttnCfs.__init__).parameters)
    orphans = [
        key
        for key in shipped["model_config"]["VAE_model"]
        if key not in constructor_keys and key not in TASK_LEVEL_KEYS
    ]

    assert orphans == [], f"{orphans} name neither a constructor argument nor a task-level key"


def test_loss_only_keys_do_not_reach_the_constructor(shipped, tmp_path):
    """The net takes tensors and computes a loss on request; it owns none of these. The constructor
    is keyword-only with no ``**kwargs``, so a leaked key would be a ``TypeError`` on the production
    config -- a poor place to find out."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnCfsTrainer, tmp_path)

    for name in TASK_LEVEL_KEYS:
        assert name not in kwargs, f"{name} is not the net's"


def test_the_shipped_geometry_pairs_the_floor_with_the_budget(shipped):
    r"""One decision, two keys. $F \ge B - 1$ is what makes every scored target coefficient honest,
    and the configuration exposes the pair so the floor may exceed the minimum -- which is what makes
    the ten-minute policy arm a config change."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["warmup_period"] == 133 == vae["causal_warmup_budget_steps"] - 1
    assert vae["c_y"] == CAUSAL_C_Y
    assert vae["c_u"] == CAUSAL_C_U


def test_the_anchor_stride_equals_the_configured_horizon(shipped):
    """The two are one decision: below the horizon the forecast windows overlap again, above it
    there are target steps no phase ever covers. Asserted rather than defaulted, so a horizon change
    that left the stride behind fails here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"] == 15


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_budget_keeps(tmp_path):
    """The binding this model's whole unit convention rests on, resolved through the real driver:
    the warm-up budget decides the surviving channels, the survivors decide the decoder width, and
    the width decides what every reported nat is summed over.

    Driven on the tiny variant because the shipped config's shard paths are deliberately
    non-existent placeholders and this resolution **reads the shards**; the tiny variant carries the
    identical geometry, which is exactly why it does."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnCfsTrainer, tmp_path)
    model = SeqVaeLagAttnCfs(**kwargs)

    assert len(kwargs["target_keep_index"]) == 98
    assert model.decoder_out_channels == 98
    assert model.raw_per_step == 16  # untouched by the width
    assert model.horizon * model.decoder_out_channels == 1470


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
    assert "cfs" in value, (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


def test_every_leaf_equals_the_comparison_configs_value(shipped, sibling):
    """The whole comparison rests on this, and here it is total rather than schema-limited: this
    net is a *subclass* of the comparison model's whose schema gains three keys and loses none, so
    every key means the same thing in both files."""
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
    $\beta_p / 2$ per latent dimension while the reconstruction it opposes is what this cell
    changed, so an arm that moved one key alone would sweep the anchor's standing at the same time
    and a pinning prior would have two explanations.
    """
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    assert mine["beta_prior"] / mine["beta_schedule"]["end"] == pytest.approx(
        theirs["beta_prior"] / theirs["beta_schedule"]["end"]
    )
    assert mine["beta_schedule"]["start"] == 0.0


@pytest.mark.parametrize("path", RETUNED_PATHS)
def test_each_retuned_value_is_a_real_move_with_a_measured_reason(shipped, sibling, path):
    """Both moved because this cell averages the same objective over roughly a twenty-fourth of the
    anchors per step, and the two moved in **opposite** directions -- the clip up, because the
    per-step gradient is noisier, and the margin down, because it is stated in nats of a block that
    shrank by a third. A retune that moved both the same way would be a sign the scale argument had
    been applied to one axis only."""
    assert float(_get(shipped, path)) != float(_get(sibling, path))
    assert "RETUNED" in PARITY_EXEMPT_PATHS[path]

    if path.endswith("gradient_clip_val"):
        assert float(_get(shipped, path)) > float(_get(sibling, path))
    else:
        assert float(_get(shipped, path)) < float(_get(sibling, path))


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_PATHS)
def test_each_measured_constant_landed_on_the_comparison_models_value(shipped, sibling, path):
    """The two loss-scale constants that were re-derived at this scale and came back to parity. The
    assertion exists so the equality stays visible as a **measurement** rather than reading as an
    oversight, and so a future retune has to leave :data:`MEASURED_TO_MATCH_PATHS` deliberately.

    ``ema_floor`` disables the relative spike test outright, and the property it needs is that it
    sits far above any *reachable* loss. The per-coefficient Gaussian NLL is bounded below by
    $0.5(\\log 2\\pi + \\texttt{logvar\\_clamp\\_lo}) \\approx -1.58$, so two reconstruction terms
    over $1470$ coefficients cannot reach beyond $\\approx 4.6 \\times 10^{3}$ in magnitude -- five
    orders of magnitude of headroom, exactly as at the larger block.

    ``horizon_embed_std`` is chosen against the post-initialisation *correlation* between two
    horizon tokens, which is a function of the embedding's scale against the broadcast projected
    latent's and does not depend on how many tokens there are. Measured on this model at $0.8$:
    $0.445915$ at $H = 15$ against $0.447476$ at $H = 30$. The halved token count moved the third
    decimal place.
    """
    assert float(_get(shipped, path)) == float(_get(sibling, path))
    assert path not in PARITY_EXEMPT_PATHS


def test_no_path_is_both_exempt_and_measured_to_match():
    """The two lists make opposite claims about the same key, so an overlap is a contradiction the
    parity assertions above would resolve arbitrarily."""
    assert set(MEASURED_TO_MATCH_PATHS).isdisjoint(PARITY_EXEMPT_PATHS)
    assert set(RETUNED_PATHS) <= set(PARITY_EXEMPT_PATHS)
    assert set(IDENTITY_PATHS) <= set(PARITY_EXEMPT_PATHS) | {"general_config.tag"}


def test_the_plotting_block_keeps_the_inherited_drivers_spelling(shipped):
    """The callback assembly is inherited and reads this literal. Renaming the block to match this
    package would disable the per-epoch diagnostic figure with no error anywhere."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True
    assert LagAttnCfsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_precision_is_float32(shipped):
    """Mixed precision would need float32 islands around the log-variances, the closed-form KL and
    the NLL reduction."""
    assert shipped["advanced_config"]["trainer"]["precision"] == "32-true"


def test_compile_is_off_and_stays_inert_for_this_net(shipped):
    """Inherited whole, along with the reason: this is that net, LSTM encoders included, so the
    driver does not read the key at all."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    assert shipped["advanced_config"]["trainer"]["compile"] is False
    assert LagAttnCfsTrainer.compile_model_requested(object()) is False


def test_num_sanity_val_steps_is_zero(shipped):
    """``MetricsLoggingCallback`` has no sanity guard; a nonzero value shifts every epoch number
    against MLflow and the checkpoint filenames."""
    assert shipped["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0


def test_the_beta_warmup_starts_at_exactly_zero(shipped):
    """z is the only route to the decoder; a nonzero beta before the decoder can use the latent is
    the standard route to posterior collapse."""
    schedule = shipped["model_config"]["VAE_model"]["beta_schedule"]

    assert schedule["kind"] == "linear_warmup"
    assert schedule["start"] == 0.0
    assert schedule["warmup_epochs"] == 50


def test_the_breaker_is_pinned_by_value_rather_than_by_presence(shipped):
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


def test_the_boundary_shape_weight_is_zero_and_the_driver_refuses_any_other_value(shipped):
    """The one shape term whose refusal is specific to this cell: it is a slicing identity over
    ADJACENT anchors, and this family always decodes a tiled set."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    assert shipped["model_config"]["VAE_model"]["lambda_boundary"] == 0.0

    broken = load_config(str(_CONFIG))
    broken["model_config"]["VAE_model"]["lambda_boundary"] = 0.5
    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCfsTrainer.preflight(broken)


def test_the_cross_channel_block_appears_in_no_config():
    """``fhr_up_ph`` mixes both signals in one coefficient, and the causal variant does not store it
    at all. Globbed rather than listed: a config added later is exactly the one nobody would think
    to add to a list here."""
    configs = sorted(_CONFIG_DIR.glob("*.yaml"))

    assert configs, "the config directory is empty; the glob is checking nothing"
    for path in configs:
        assert "fhr_up_ph" not in path.read_text(encoding="utf-8"), path.name


def test_the_validity_mask_emitter_appears_in_no_config():
    """``emit_validity_mask`` would pay for a per-sample copy of a filter-bank constant, per worker,
    per collate and per host-to-device copy -- buying nothing a single resolved vector in
    ``model_kwargs`` does not already give, and that vector is also what reaches the checkpoint."""
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        assert "emit_validity_mask" not in path.read_text(encoding="utf-8"), path.name


def test_the_target_blocks_are_loaded_and_normalized(shipped):
    """The one guard the target domain moves. Both blocks, field by field: an unnormalised block
    makes the Gaussian NLL meaningless with nothing else raising."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    dataloader = shipped["dataset_config"]["dataloader_config"]

    for field in LagAttnCfsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


def test_the_two_phase_key_fields_are_loaded(shipped):
    """Load-bearing rather than incidental, and the one difference from the two-sided cells' list:
    the tile phase is keyed on the recording identifier and the segment's own start time, and
    ``load_fields`` is honoured literally with no forced additions."""
    from teb_vae.lag_attn_cfs.trainer import PHASE_KEY_FIELDS

    load_fields = shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]

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


def test_the_tiny_variant_names_only_its_deltas():
    """The whole point of the ``base:`` mechanism, counted on the raw file: a variant that re-stated
    the inherited settings would be a copy that drifts. Budgeted against the comparison model's own
    smoke variant, because the two are smoke variants of configs that are themselves pinned
    leaf-for-leaf."""
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


def test_the_tiny_variant_inherits_the_geometry_that_decides_what_it_exercises(tiny, shipped):
    """Including every correctness requirement, and -- unlike the two-sided cells' smoke variants --
    including the warm-up budget, the anchor floor and the stride: the decoder's width IS the
    resolved budget's survivor count and the decoded anchor count IS the floor, stride and horizon,
    so a shrunken variant would exercise a decoder and an anchor set the production run does not
    have."""
    assert tiny["advanced_config"]["trainer"]["compile"] is False
    assert tiny["advanced_config"]["trainer"]["precision"] == "32-true"
    assert tiny["advanced_config"]["spike_breaker"]["ema_floor"] >= 1.0e9
    assert tiny["advanced_config"]["trainer"]["num_sanity_val_steps"] == 0
    vae = tiny["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]
    for key in (
        "sequence_length", "raw_per_step", "horizon", "warmup_period", "anchor_stride",
        "lag_floor", "c_y", "c_u", "causal_warmup_budget_steps", "causal_reach_budget_s",
        "causal_norm",
    ):
        assert vae[key] == shipped_vae[key], key
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
    # And the tiling's own feasibility: the last phase's first anchor must exist.
    assert vae["warmup_period"] + vae["anchor_stride"] <= (
        vae["sequence_length"] - vae["horizon"]
    )


def test_the_tiny_variant_points_at_the_committed_causal_shard(tiny):
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


def test_the_resolved_tiny_variant_validates_and_builds(tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnCfs(
        **_model_kwargs_from(load_config(str(_TINY)), LagAttnCfsTrainer, tmp_path)
    )
    # The smoke model is small everywhere except where it must not be: the decoder still emits the
    # production width and the forward still decodes the production tile count.
    assert model.d_model == 32
    assert model.decoder_out_channels == 98
    assert model.anchor_stride == 15


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
    the block the NLL sums over, the anchor set it is averaged over, the weights it is balanced
    against, the clamp the log-variances sit inside, and the breaker that could silently replace the
    loss with its own EMA -- is inherited, or the reading is of a different model."""
    vae = smoke_hie["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_warmup_budget_steps", "warmup_period", "anchor_stride", "horizon", "beta_prior",
        "logvar_clamp", "d_z", "d_model",
    ):
        assert vae[key] == shipped_vae[key], key
    assert vae["beta_schedule"]["end"] == shipped_vae["beta_schedule"]["end"]
    assert vae["beta_schedule"]["start"] == 0.0
    assert vae["likelihood"] == "gaussian_nll"  # the learned-variance heads, not the debug path
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
    a production run and a quarter of this one."""
    warmup = smoke_hie["model_config"]["VAE_model"]["beta_schedule"]["warmup_epochs"]
    epochs = smoke_hie["general_config"]["epochs"]

    assert warmup * 10 <= epochs


def test_the_local_variant_names_a_causal_shard_that_does_not_yet_exist(smoke_hie):
    """Deliberately a placeholder, and the reason is stated rather than left as an omission: there
    is no committed causal HIE shard, and the two-sided one beside it cannot stand in -- it carries
    no ``transform`` attribute, no per-block warm-up vectors and the two-sided channel counts, and
    the pre-flight refuses it by name. The header says what to build and how."""
    dataset = smoke_hie["dataset_config"]

    assert dataset["vae_train_datasets"] == dataset["vae_test_datasets"]  # in-sample, deliberately
    for path in (*dataset["vae_train_datasets"], dataset["stat_path"]):
        assert "causal" in path, path
        assert not (_REPO_ROOT / path).is_file(), (
            f"{path} now exists -- if a causal HIE shard has been built, this test should assert "
            f"its presence instead, and the config header's build instructions can go"
        )
    assert "PREREQUISITE" in _SMOKE_HIE.read_text(encoding="utf-8")


def test_the_local_variant_runs_on_one_device_with_tracking_off(smoke_hie):
    """A dev-box run: one GPU, and no MLflow server to log to. The run directory is the record."""
    assert smoke_hie["general_config"]["cuda_devices"] == [0]
    assert smoke_hie["advanced_config"]["tracking"]["mlflow"]["enabled"] is False


def test_the_local_variant_normalizes_and_loads_both_target_blocks(smoke_hie):
    """Inherited, and asserted anyway: the entry point's guard runs against
    ``LagAttnCfsTrainer.TARGET_FIELDS``."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    dataloader = smoke_hie["dataset_config"]["dataloader_config"]

    for field in LagAttnCfsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field
