r"""The shipped configs load, validate, reach something, and do not drift from the comparison.

``default.yaml`` here is written out in full rather than inheriting the two-sided feature model's,
and the price of that is drift -- drift in a key that has nothing to do with the transform is
exactly what destroys the comparison the package exists to make. So parity is a tested property, in
both directions: **every** leaf must equal the comparison config's value unless it is declared in
:data:`PARITY_EXEMPT_PATHS`, and every declared exemption must be a real divergence. The comparison
is total rather than schema-limited, because this model's net is a *subclass* of the comparison
model's and every key means the same thing in both files.

Beside parity: the shipped, smoke and dev-box configs each resolve and pass the framework's own
validator with no unknown or dead key; every ``VAE_model`` key reaches the constructor or the task;
the driver refuses the one shape term this cell cannot score; the fields the target domain and the
tile phase need are loaded; the two variants inherit the geometry and the objective they exist to
exercise; and the tiny variant resolves through the real driver into a model whose decoder is as
wide as the warm-up budget keeps.
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

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_HIE = _CONFIG_DIR / "smoke_hie.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"

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
    # All three alignment keys are resolved against the SHARDS by the trainer and reach the
    # constructor only as the shift tuples, so none of them names a constructor argument.
    "causal_align_reference",
    # The source stream's own clock, snapped against the SOURCE's stored delays rather than the
    # target's. It is task-level for the same reason as the key above and for one more: what it
    # produces is a second keep-index and a second shift tuple on one stream only, which is a
    # resolution result and not an architecture choice.
    "causal_align_reference_source",
    # The forecast clock: resolved against the shards' delay vectors into the signed
    # `target_forecast_shift` tuple the constructor takes, exactly as the alignment references
    # resolve into theirs.
    "causal_target_forecast_clock",
    "causal_leg_alignment",
    # The phase-harmonic operator VERSION the shards were built with, checked against their root
    # attribute exactly as the leg alignment is; a resolver expectation, not a constructor key.
    "causal_phase_operator",
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
#: is drift, and drift is a confound. No wildcards: the list names its contents so that adding an
#: entry is a decision rather than an omission.
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
    "model_config.VAE_model.causal_phase_operator": (
        "which phase-harmonic operator VERSION built the configured shards, which only a causal "
        "shard records at all"
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
    "model_config.VAE_model.target_weight_st": (
        "the per-block reconstruction weights, which have no two-sided counterpart: the comparison "
        "model scores its channels uniformly, and its objective is a log-density because of it"
    ),
    "model_config.VAE_model.target_weight_ph": (
        "the second of the pair; see target_weight_st"
    ),
    "model_config.VAE_model.horizon": (
        "this cell forecasts 10 steps since 2026-09-05 (with the conv-Transformer causal cell); "
        "the two-sided cell still forecasts 30"
    ),
    "advanced_config.trainer.gradient_clip_val": (
        "RETUNED: measured on a 600-step instrumented run at the H = 30 geometry (q99 of the "
        "pre-clip norm 14181 against the comparison model's 4421 -- this cell averages its block "
        "over far fewer anchors per step than the dense ~240) and scaled by the block ratio to the "
        "H = 10 block"
    ),
    "advanced_config.spike_breaker.additive_margin": (
        "RETUNED: the margin is stated in nats of the summed block, and this block is 760 "
        "coefficients against 2340; measured against the breaker's own excursion-above-EMA "
        "statistic at H = 30 and scaled by the block ratio, held under the reachable magnitude"
    ),
    # The six architecture switches and the two training controls this revision added. Every one is
    # a key the comparison model's constructor does not have and its config never carries, so the
    # divergence is "this mechanism exists here" rather than "this number was chosen differently" --
    # and each ships at a value whose OFF-state is bitwise the comparison model's behaviour, which
    # is what keeps the two runs comparable at all.
    "model_config.VAE_model.lag_kv_source": (
        "the lag attention's key/value memory, which has no two-sided counterpart: it selects "
        "between the deep source encoder and a local source representation, and only a stream "
        "read through an availability gate has the second"
    ),
    "model_config.VAE_model.prior_availability_input": (
        "the prior's availability clock, which has no two-sided counterpart: it announces which "
        "source channels have arrived, and a two-sided stream announces nothing because every "
        "channel is present from the first step"
    ),
    "model_config.VAE_model.persistence_residual": (
        "the target-only persistence term in the decoder mean, which the comparison model's "
        "constructor does not take; off there, and off is bitwise its current decoder"
    ),
    "model_config.VAE_model.horizon_weight_halflife_steps": (
        "the decaying horizon weighting of the reconstruction, which the comparison model's "
        "constructor does not take; null there, and null is the uniform sum it already computes"
    ),
    "model_config.VAE_model.alibi_slope_scale": (
        "the lag-bias seed's slope multiplier. Shipped at 0.0 here -- a FLAT learnable per-lag "
        "bias -- because a decaying seed predicts a lag-0 peak before the model has read anything, "
        "which is a hazard only a cell reading a physiological delay off the lag axis has"
    ),
    "model_config.VAE_model.causal_align_reference_source": (
        "the SOURCE stream's own clock, the second half of a dual reference. It has no two-sided "
        "counterpart for the same reason causal_align_reference does not, and it is the one "
        "divergence of this pair that is a chosen number: 288.2672 s, snapped to a stored source "
        "delay, at a known -113.8932 s offset against the target's clock"
    ),
    "model_config.VAE_model.causal_target_forecast_clock": (
        "which clock the forecast target is SCORED on, which has no two-sided counterpart: a "
        "symmetric bank's coefficients already describe the instant they are stored at, so there "
        "is no per-channel staleness for a clock to correct on the question side"
    ),
    "advanced_config.callbacks.early_stopping.enabled": (
        "the training controls: this row stops on val/total_loss where the comparison model runs "
        "its epoch budget out. Enabled here because a run of this cell has been observed to reach "
        "its composite optimum hundreds of epochs before its budget ends"
    ),
    "advanced_config.callbacks.early_stopping.patience": (
        "the second half of the control above, in validation epochs; inheriting the comparison "
        "model's value would make the flag above inert rather than merely different"
    ),
    "advanced_config.callbacks.model_checkpoint.secondary_monitor": (
        "the second checkpoint criterion, on val/nll_full_block. Absent in the comparison config, "
        "where absence builds no second callback: the composite optimum and the best conditioned "
        "forecast are different epochs, and only one of them is recoverable without this"
    ),
}


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
# Every config loads, and everything in it reaches something
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", (_CONFIG, _TINY, _SMOKE_HIE), ids=lambda path: path.name)
def test_the_resolved_config_validates_with_no_unknown_or_dead_key_warnings(
    path, tmp_path, loguru_warnings
):
    """Resolved first, which is the only way a variant ever reaches the experiment driver, and
    driven through the framework's real validator rather than a copy of its rules. A leftover
    ``base:`` key would surface here as an unknown key."""
    resolved = resolve_config_file(str(path), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
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


# --------------------------------------------------------------------------------------
# Parity with the model this one is compared against
# --------------------------------------------------------------------------------------
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


def test_the_plotting_block_is_the_one_the_inherited_driver_reads(shipped):
    """The callback assembly is inherited and reads the block by the driver's key. A block renamed
    to match this package would disable the per-epoch diagnostic figure with no error anywhere."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    callbacks = shipped["advanced_config"]["callbacks"]
    assert callbacks[LagAttnCfsTrainer.PLOT_CONFIG_KEY]["enabled"] is True


# --------------------------------------------------------------------------------------
# The settings that are correctness requirements
# --------------------------------------------------------------------------------------
def test_the_driver_refuses_a_nonzero_boundary_shape_weight():
    """The one shape term whose refusal is specific to this cell: it is a slicing identity over
    ADJACENT anchors, and this family always decodes a tiled set."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    broken = load_config(str(_CONFIG))
    broken["model_config"]["VAE_model"]["lambda_boundary"] = 0.5
    with pytest.raises(ValueError, match="lambda_boundary"):
        LagAttnCfsTrainer.preflight(broken)


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


def test_the_anchor_stride_and_weight_half_life_track_the_horizon(shipped):
    """The three travel together: the stride is $S = H/2$ and the horizon-weight half-life is
    $H/2$. Asserted as relations rather than values, so a horizon change that left the stride or
    the half-life behind (a different tile count and a different weight profile, silently) fails
    here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"] // 2
    assert vae["horizon_weight_halflife_steps"] == vae["horizon"] / 2


# --------------------------------------------------------------------------------------
# The smoke variant
# --------------------------------------------------------------------------------------
def test_the_tiny_variant_inherits_the_geometry_that_decides_what_it_exercises(tiny, shipped):
    """Including every correctness requirement, and -- unlike the two-sided cells' smoke variants --
    including the warm-up budget, the anchor floor and the stride: the decoder's width IS the
    resolved budget's survivor count and the decoded anchor count IS the floor, stride and horizon,
    so a shrunken variant would exercise a decoder and an anchor set the production run does not
    have."""
    for key in ("compile", "precision", "num_sanity_val_steps"):
        assert tiny["advanced_config"]["trainer"][key] == shipped["advanced_config"]["trainer"][
            key
        ], key
    assert (
        tiny["advanced_config"]["spike_breaker"]["ema_floor"]
        == shipped["advanced_config"]["spike_breaker"]["ema_floor"]
    )
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


def test_the_tiny_variant_builds_a_decoder_as_wide_as_the_budget_keeps(tiny, tmp_path):
    """The binding this model's whole unit convention rests on, resolved through the real driver:
    the warm-up budget decides the surviving channels, the survivors decide the decoder width, and
    the width decides what every reported nat is summed over. The smoke model is small everywhere
    except there and in the tiling it decodes."""
    from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer

    kwargs = _model_kwargs_from(load_config(str(_TINY)), LagAttnCfsTrainer, tmp_path)
    model = SeqVaeLagAttnCfs(**kwargs)
    vae = tiny["model_config"]["VAE_model"]

    assert model.decoder_out_channels == len(kwargs["target_keep_index"])
    assert model.anchor_stride == vae["anchor_stride"]
    assert model.horizon == vae["horizon"]


# --------------------------------------------------------------------------------------
# The dev-box validation variant
# --------------------------------------------------------------------------------------
def test_the_local_variant_inherits_everything_the_reading_depends_on(smoke_hie, shipped):
    """Only the run's *scale* is local. Every quantity the pre-registered criteria are read from --
    the block the NLL sums over, the anchor set it is averaged over, the weights it is balanced
    against, the clamp the log-variances sit inside, and the breaker that could silently replace the
    loss with its own EMA -- is inherited, or the reading is of a different model."""
    vae = smoke_hie["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_warmup_budget_steps", "warmup_period", "anchor_stride", "horizon", "beta_prior",
        "logvar_clamp", "d_z", "d_model", "likelihood", "causal_norm",
    ):
        assert vae[key] == shipped_vae[key], key
    for key in ("start", "end"):
        assert vae["beta_schedule"][key] == shipped_vae["beta_schedule"][key], key
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
