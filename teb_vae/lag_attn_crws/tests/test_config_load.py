r"""The shipped configs load, validate, contain nothing that reaches nothing, and do not drift.

``default.yaml`` here is written out in full rather than inheriting the raw-signal model's, and the
price of that is drift -- drift in a key that has nothing to do with the input representation is
exactly what destroys the comparison the package exists to make. A difference in ``seed``, ``lr``,
``free_bits``, ``d_z``, the coverage floor or the beta pair would be attributed to one-sidedness by
every reading of the two runs.

So parity is a tested property, in both directions: **every** leaf must equal the comparison
config's value against :data:`PARITY_EXEMPT_PATHS`, and adding a divergence means declaring it
there. The comparison is total rather than schema-limited: this model's net is a *subclass* of the
comparison model's whose constructor schema gains four keys, loses two delay keys nothing here
configures, and re-points none -- so every key present in both means the same thing in both files.

The twenty exemptions fall into five kinds, and the split is the record rather than bookkeeping.
**Five are identity** -- the run tag, the output directory, the MLflow experiment, the run name and
the variant tag -- and inheriting any of them writes these runs into the other model's tree. **Four
are the dataset**: causal shards and the statistics accumulated from them, plus the loader key the
tile phase is derived from. **Eight are the geometry the transform forces or the tiling adds**: the
channel widths the one-sided cascade leaves, the horizon, the anchor floor the warm-up budget pairs
with, the three keys that have no two-sided counterpart, and the reach budget this dataset makes
undefined. **One is the objective**: ``lambda_boundary``, which is a meaningful raw-waveform term on
the comparison model and a slicing identity with no meaning over a tiled anchor set here.

**Two are measurements**, and they are the ones worth reading twice. ``gradient_clip_val`` and
``spike_breaker.additive_margin`` are both stated in units of the summed block, and this cell moves
the block *and* the anchor count in opposite directions -- so neither transfers by arithmetic and
neither was scaled. Both were re-derived from the instrumented run through
``configs/smoke_causal.yaml``, both moved **down**, and both are declared in
:data:`PARITY_EXEMPT_PATHS` with a reason beginning ``RETUNED`` and listed again in
:data:`RETUNED_PATHS`, whose test asserts the direction as well as the divergence. The distribution
they were derived from is recorded in ``tests/test_spike_breaker.py``, where the brackets they have
to sit inside are checked against it rather than against each other.

The two loss-scale constants that were re-derived and came back to parity live in
:data:`MEASURED_TO_MATCH_PATHS`, whose test asserts the equality so that it reads as a measurement
rather than as an oversight.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import pytest
import yaml
from loguru import logger

from teb_vae.lag_attn.config import load_config, resolve_config_file
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from train.test_utils import make_graph_model

from .conftest import CAUSAL_C_U, CAUSAL_C_Y, absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_CAUSAL = _CONFIG_DIR / "smoke_causal.yaml"
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

#: Shared by the geometry entries below, because it is one reason rather than several: the one-sided
#: cascade drops seven scattering channels per block at write time, the warm-up budget drops four
#: more from the target-feature stream, and the anchor floor is what the surviving warm-ups make the
#: declared input-warmth policy cost.
GEOMETRY_REASON = (
    "the one-sided cascade's own geometry: 36 + 66 target-feature and 36 + 15 source channels "
    "against the two-sided 43 + 66 and 43 + 15, and an anchor floor of B - 1 over the surviving "
    "warm-ups rather than the model's own 30-step one"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else differing
#: is drift, and drift is a confound. Twenty entries and no wildcards: the list names its contents
#: so that adding a twenty-first is a decision rather than an omission.
PARITY_EXEMPT_PATHS: Dict[str, str] = {
    "general_config.tag": "the run tag names the input representation",
    "general_config.folders_config.out_dir_base": "IDENTITY: a shared output tree mixes the runs",
    "advanced_config.tracking.mlflow.experiment_name": "IDENTITY: MLflow experiment",
    "advanced_config.tracking.mlflow.run_name": "IDENTITY: MLflow run name",
    "advanced_config.tracking.mlflow.tags.variant": "IDENTITY: MLflow variant tag",
    "dataset_config.vae_train_datasets": "DATASET: the causal shards, not the two-sided ones",
    "dataset_config.vae_test_datasets": "DATASET: the causal shards, not the two-sided ones",
    "dataset_config.stat_path": (
        "DATASET: statistics accumulated from the causal shards EXCLUDING each channel's warm-up "
        "region, which is what makes zero the channel mean over the region the model reads"
    ),
    "dataset_config.dataloader_config.dataset_kwargs.load_fields": (
        "DATASET: 'epoch' is added because the anchor tiling's per-segment phase is keyed on the "
        "segment's own start time as well as the recording identifier"
    ),
    "model_config.VAE_model.c_y": GEOMETRY_REASON,
    "model_config.VAE_model.c_u": GEOMETRY_REASON,
    "model_config.VAE_model.warmup_period": GEOMETRY_REASON,
    # `horizon` is deliberately ABSENT: this cell now forecasts the comparison model's two minutes,
    # so an exemption here would be a permission with no divergence behind it. The BLOCK matches
    # with it -- H * R = 480 on both sides, because R is a property of the raw grid rather than of a
    # channel budget -- which is what makes a nat comparable across the input-representation edge
    # and is the whole of what the horizon move bought this row.
    "model_config.VAE_model.causal_reach_budget_s": (
        "null and REQUIRED null: the forward reach L95 is an energy quantile of a two-sided kernel, "
        "measured on a bank that did not produce these coefficients, and a delay is a shift"
    ),
    "model_config.VAE_model.causal_align_reference": (
        "the channel alignment, which has no two-sided counterpart: the comparison model's bank "
        "is symmetric and its channels already report one instant, so there is no clock to move"
    ),
    "model_config.VAE_model.causal_leg_alignment": (
        "which phase-harmonic operator built the configured shards, which only a causal shard "
        "records at all"
    ),
    "model_config.VAE_model.causal_warmup_budget_steps": (
        "the guard this dataset needs, which has no two-sided counterpart at all"
    ),
    "model_config.VAE_model.anchor_stride": (
        "the anchor tiling, which has no two-sided counterpart: the comparison model decodes every "
        "anchor, which is this key's inert value"
    ),
    "model_config.VAE_model.lag_floor": (
        "the lag validity floor, which has no two-sided counterpart; it ships at 0, where the lag "
        "mask is bitwise the comparison model's"
    ),
    "model_config.VAE_model.lambda_boundary": (
        "REQUIRED 0.0 rather than the comparison model's 0.05: the term is a slicing identity over "
        "ADJACENT anchors, and this cell always decodes a set whose entries are a stride apart. It "
        "is the only one of the three shape weights that does not transfer -- the multiscale L1 and "
        "the derivative Huber are within-block quantities and are inherited unchanged"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "RETUNED: the block halves (240 raw samples against 480) while the decoded anchors per step "
        "fall by roughly 24x, and the two move the gradient distribution in opposite directions, so "
        "the value is measured rather than scaled -- the smallest round value above the pre-clip "
        "norm's q99 on the instrumented run"
    ),
    "advanced_config.spike_breaker.additive_margin": (
        "RETUNED: the margin is stated in nats of the summed block and transfers across neither the "
        "block size nor the anchor count, both of which this cell moves; measured against the "
        "breaker's own excursion-above-EMA statistic on the instrumented run"
    ),
}

#: The exemptions that were **measured** rather than merely permitted. Named as a separate list for
#: two reasons: the direction is part of the claim -- a smaller block cannot want a larger threshold
#: -- and the numbers behind them live in one place, ``tests/test_spike_breaker.py``, where each is
#: bracketed against the distribution it came from rather than against the other.
RETUNED_PATHS = (
    "advanced_config.trainer.gradient_clip_val",
    "advanced_config.spike_breaker.additive_margin",
)

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

#: Every leaf of ``smoke_causal.yaml`` that resolves to something other than ``default.yaml``'s
#: value. Note what is **not** in it: every model width, the warm-up budget, the anchor floor, the
#: stride, the horizon, the beta pair and the whole spike-breaker block are inherited -- the run has
#: to be the shipped objective at the shipped widths or the distribution it measures is another
#: model's. Only the run's *scale* is local, plus the parked clip, which is the reason it exists.
SMOKE_CAUSAL_DELTA_PATHS = frozenset(
    {
        "general_config.tag",
        "general_config.cuda_devices",
        "general_config.epochs",
        "general_config.plot_frequency",
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
    constructor_keys = set(inspect.signature(SeqVaeLagAttnCrws.__init__).parameters)
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
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnCrwsTrainer, tmp_path)

    for name in TASK_LEVEL_KEYS:
        assert name not in kwargs, f"{name} is not the net's"


def test_the_shipped_geometry_pairs_the_floor_with_the_budget(shipped):
    r"""One decision, three keys. The floor is the maximum of $B - 1$, the declared input-warmth
    policy -- every kept input channel warm by the first forecast step -- and $\max_c(W'_c + d_c)$,
    which the alignment makes bind at exactly $B$. The configuration exposes all three so the floor
    may exceed the minimum rather than being derived from the threshold."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["causal_align_reference"] == 42.21
    assert vae["warmup_period"] == 134 == vae["causal_warmup_budget_steps"]
    assert vae["c_y"] == CAUSAL_C_Y
    assert vae["c_u"] == CAUSAL_C_U


def test_the_anchor_stride_equals_the_configured_horizon(shipped):
    """The two are one decision: below the horizon the forecast windows overlap again, above it
    there are target steps no phase ever covers. Asserted rather than defaulted, so a horizon change
    that left the stride behind fails here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"] == 30


def test_the_shipped_config_builds_a_decoder_as_wide_as_the_raw_grid(tmp_path):
    """The one binding this cell does **not** make, asserted so its absence is a passing test rather
    than a comment: the warm-up budget decides the input adapters' widths and nothing else. The
    decoder emits ``raw_per_step`` raw samples per horizon token, so the block is H * R and no
    configuration -- and no budget -- can put the decoder and the target on different widths.

    Driven on the tiny variant because the shipped config's shard paths are deliberately
    non-existent placeholders and this resolution **reads the shards**; the tiny variant carries the
    identical geometry, which is exactly why it does."""
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnCrwsTrainer, tmp_path)
    model = SeqVaeLagAttnCrws(**kwargs)

    # 38, not the budget's 98: this cell forecasts the RAW signal, so nothing on the target side
    # cancels the source reference out of the physical-lag identity and both streams drop to the
    # 42.21 s reference that puts the 20-120 s coupling band inside the lag axis.
    assert len(kwargs["target_keep_index"]) == 38
    assert len(kwargs["source_keep_index"]) == 17
    assert model.decoder_out_channels == model.raw_per_step == 16
    assert model.horizon * model.geometry.r == 480


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
    assert "crws" in value, (
        f"{path} is {value!r}, which does not name this model -- a reader cannot tell whose run "
        f"it is"
    )


def test_every_leaf_equals_the_comparison_configs_value(shipped, sibling):
    """The whole comparison rests on this, and here it is total rather than schema-limited: this net
    is a *subclass* of the comparison model's whose schema gains four keys and re-points none, so
    every key present in both means the same thing in both files."""
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
def test_each_retuned_constant_moved_up_from_the_comparison_models_value(shipped, sibling, path):
    """Both are stated in units of the summed block, and neither transfers by arithmetic.

    **The direction inverted when the horizon moved**, and that is what this asserts. At $H = 15$
    this cell's block was half the comparison model's and both constants came back smaller. At
    $H = 30$ the block is *equal* to it -- $30 \times 16 = 480$ raw samples on both sides, because
    $R$ is a property of the raw grid -- while the anchor count is still roughly a fiftieth, so the
    per-step gradient and the per-step excursion are both larger and both constants come back
    bigger. The block stopped being the axis that separates these two cells; the anchor count is
    now the whole of it."""
    assert path in PARITY_EXEMPT_PATHS
    assert PARITY_EXEMPT_PATHS[path].startswith("RETUNED")
    assert float(_get(shipped, path)) > float(_get(sibling, path))


def test_the_instrumented_config_is_the_record_of_where_those_two_constants_came_from():
    """The re-derivations name a measurement, and the measurement names a config. Asserted together
    so a file deleted as unused takes the only reproducible route to those two numbers with it."""
    assert _SMOKE_CAUSAL.is_file()
    text = _SMOKE_CAUSAL.read_text(encoding="utf-8")

    assert "gradient_clip_val" in text and "additive_margin" in text


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


@pytest.mark.parametrize("path", MEASURED_TO_MATCH_PATHS)
def test_each_measured_constant_landed_on_the_comparison_models_value(shipped, sibling, path):
    r"""The two loss-scale constants whose value is demonstrably insensitive to what this cell moves.
    The assertion exists so the equality stays visible as a **measurement** rather than reading as an
    oversight, and so a future retune has to leave :data:`MEASURED_TO_MATCH_PATHS` deliberately.

    ``ema_floor`` disables the relative spike test outright, and the property it needs is that it
    sits far above any *reachable* loss. The per-sample Gaussian NLL is bounded below by
    $0.5(\log 2\pi + \texttt{logvar\_clamp\_lo}) \approx -1.58$, so two reconstruction terms over
    $240$ raw samples cannot reach beyond $\approx 7.6 \times 10^{2}$ in magnitude -- six orders of
    magnitude of headroom, more than at the larger block rather than less.

    ``horizon_embed_std`` is chosen against the post-initialisation *correlation* between two
    horizon tokens, which is a function of the embedding's scale against the broadcast projected
    latent's and does not depend on how many tokens there are: $0.445915$ at $H = 15$ against
    $0.447476$ at $H = 30$ on this architecture. The halved token count moved the third decimal
    place.
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
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    assert shipped["advanced_config"]["callbacks"]["lag_attn_rws_plotting"]["enabled"] is True
    assert LagAttnCrwsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


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
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    assert shipped["advanced_config"]["trainer"]["compile"] is False
    assert LagAttnCrwsTrainer.compile_model_requested(object()) is False


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
    """The one shape weight that does not transfer from the comparison model, and the refusal is
    specific to this cell: the term is a slicing identity over ADJACENT anchors, and this cell always
    decodes a tiled set."""
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    assert shipped["model_config"]["VAE_model"]["lambda_boundary"] == 0.0

    broken = load_config(str(_CONFIG))
    broken["model_config"]["VAE_model"]["lambda_boundary"] = 0.5
    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCrwsTrainer.preflight(broken)


def test_the_two_within_block_shape_weights_transfer_unchanged(shipped, sibling):
    """The other two shape terms are raw-waveform quantities over one anchor's own block, so the
    anchor axis does not touch them and the target is the same signal on the same grid. They are
    inherited at the comparison model's weights, and their equality is what makes the boundary term's
    exemption a statement about the anchor axis rather than about shape terms in general."""
    mine = shipped["model_config"]["VAE_model"]
    theirs = sibling["model_config"]["VAE_model"]

    assert mine["lambda_ms"] == theirs["lambda_ms"] == 0.1
    assert mine["lambda_deriv"] == theirs["lambda_deriv"] == 0.1


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


def test_the_raw_target_is_loaded_and_normalized(shipped):
    """An unnormalised raw target arrives at ~140 bpm and makes the Gaussian NLL meaningless with
    nothing else raising."""
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    dataloader = shipped["dataset_config"]["dataloader_config"]

    for field in LagAttnCrwsTrainer.TARGET_FIELDS:
        assert field in dataloader["normalize_fields"], field
        assert field in dataloader["dataset_kwargs"]["load_fields"], field


def test_the_validity_signal_is_loaded(shipped):
    """``weight`` is the only trustworthy gap signal for a raw target: gaps are stored as 0 bpm,
    which after z-scoring is about -11 sigma rather than a detectable sentinel."""
    from teb_vae.lag_attn_crws.trainer import WEIGHT_FIELD

    load_fields = shipped["dataset_config"]["dataloader_config"]["dataset_kwargs"]["load_fields"]

    assert WEIGHT_FIELD in load_fields


def test_the_two_phase_key_fields_are_loaded(shipped):
    """Load-bearing rather than incidental, and the one difference from the comparison model's list:
    the tile phase is keyed on the recording identifier and the segment's own start time, and
    ``load_fields`` is honoured literally with no forced additions."""
    from teb_vae.lag_attn_crws.trainer import PHASE_KEY_FIELDS

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
        "causal_norm", "lambda_boundary",
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
    from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer

    resolved = resolve_config_file(str(_TINY), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    model = SeqVaeLagAttnCrws(
        **_model_kwargs_from(load_config(str(_TINY)), LagAttnCrwsTrainer, tmp_path)
    )
    # The smoke model is small everywhere except where it must not be: the input adapters still read
    # the production channel set and the forward still decodes the production tile count.
    assert model.d_model == 32
    assert model.target_adapter.linear.in_features == 38
    assert model.anchor_stride == 30


# --------------------------------------------------------------------------------------
# The instrumented variant the two pending constants are measured from
# --------------------------------------------------------------------------------------
def test_the_instrumented_variant_resolves_with_its_base_consumed(smoke_causal):
    """A leftover ``base`` key would reach ``validate_config`` as an unknown key and the resolved
    config written beside the run's checkpoints -- its provenance record -- would carry a loader
    directive rather than a setting."""
    assert "base" not in smoke_causal


def test_the_instrumented_delta_is_exactly_the_declared_key_list(smoke_causal, shipped):
    """Both directions, and it matters more here than for ``tiny.yaml``: this is the variant whose
    numbers become two shipped constants, so an undeclared delta is a threshold derived from a model
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
):
    """Only the run's *scale* and the parked clip are local. Every quantity the two constants are
    stated in -- the block the NLL sums over, the anchor set it is averaged over, the weights it is
    balanced against, the clamp the log-variances sit inside, and the breaker whose own EMA is the
    statistic -- is inherited, or the measurement is of a different model."""
    vae = smoke_causal["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_warmup_budget_steps", "warmup_period", "anchor_stride", "horizon", "raw_per_step",
        "beta_prior", "logvar_clamp", "d_z", "d_model", "lambda_ms", "lambda_deriv",
    ):
        assert vae[key] == shipped_vae[key], key
    assert vae["beta_schedule"] == shipped_vae["beta_schedule"]
    assert vae["likelihood"] == "gaussian_nll"  # the learned-variance heads, not the debug path
    assert vae["causal_norm"] is True
    assert (
        smoke_causal["advanced_config"]["spike_breaker"]
        == shipped["advanced_config"]["spike_breaker"]
    )


def test_the_instrumented_variant_parks_the_clip_far_above_anything_reachable(smoke_causal):
    """The reason the file exists. A run measured under an active clip reports the gradient-norm
    distribution of an already-clipped optimizer, which is not the distribution a threshold should be
    set from -- and the value being replaced is precisely the one doing the clipping."""
    assert smoke_causal["advanced_config"]["trainer"]["gradient_clip_val"] >= 1.0e9


def test_the_instrumented_variant_states_its_step_count_and_reaches_it(smoke_causal):
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


def test_the_instrumented_variant_points_at_the_committed_causal_shard(smoke_causal):
    """Unlike the causal-feature cell's dev-box variant, this one runs today: the constants it
    measures block a shipped config, so it cannot wait on a shard that does not exist."""
    dataset = smoke_causal["dataset_config"]

    assert dataset["vae_train_datasets"] == dataset["vae_test_datasets"]  # in-sample, deliberately
    for path in (*dataset["vae_train_datasets"], dataset["stat_path"]):
        assert "causal" in path, path
        assert (_REPO_ROOT / path).is_file(), path


def test_the_instrumented_variant_runs_on_one_device_with_tracking_off(smoke_causal):
    """A dev-box run: one GPU, and no MLflow server to log to. The run directory is the record."""
    assert smoke_causal["general_config"]["cuda_devices"] == [0]
    assert smoke_causal["advanced_config"]["tracking"]["mlflow"]["enabled"] is False
