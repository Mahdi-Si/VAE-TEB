r"""The shipped configs load, validate, reach the model, and do not drift from the comparison model.

``default.yaml`` here is written out in full rather than inheriting the raw-signal model's, and the
price of that is drift -- drift in a key that has nothing to do with the input representation is
exactly what destroys the comparison the package exists to make. So parity is a tested property, in
both directions: **every** leaf must equal the comparison config's value unless it is declared in
:data:`PARITY_EXEMPT_PATHS`, and every declared exemption must be a real divergence.

Beside that: the shipped config passes the framework's own validator; every ``VAE_model`` key names
a constructor argument or a task-level key; the stride equals the horizon; the plotting block keeps
the spelling the inherited callback assembly reads; and the two runnable variants -- ``tiny.yaml``
and the instrumented ``smoke_causal.yaml`` -- inherit the geometry and objective they exist to
exercise, validate once resolved, and build a model against the committed causal shard. The
pre-flight refusals (boundary weight, load fields, cross-channel block) are ``test_preflight.py``'s.
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
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from train.test_utils import make_graph_model

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_SMOKE_CAUSAL = _CONFIG_DIR / "smoke_causal.yaml"
_SIBLING_CONFIG = _REPO_ROOT / "teb_vae" / "lag_attn_rws" / "configs" / "default.yaml"

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

#: Shared by the geometry entries below, because it is one reason rather than several: the
#: one-sided cascade stores fewer scattering channels per block, and the anchor floor is what the
#: surviving warm-ups make the declared input-warmth policy cost.
GEOMETRY_REASON = (
    "the one-sided cascade's own channel widths, and an anchor floor of B - 1 over the surviving "
    "warm-ups rather than the model's own one"
)

#: Every leaf allowed to differ from the comparison config, with the reason. Anything else differing
#: is drift, and drift is a confound. No wildcards: the list names its contents so that adding an
#: entry is a decision rather than an omission.
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
    # The four architecture switches and the three training controls this row takes. Each names a
    # key the comparison model's constructor does not have and its config never carries, so the
    # divergence is "this mechanism exists here" rather than "this number was chosen differently".
    # Two of the six causal switches are DELIBERATELY ABSENT from this list because they are
    # absent from this row: `persistence_residual` persists a stored target coefficient and this
    # row's target is the raw signal, and `causal_align_reference_source` is the second half of a
    # pair of clocks that a zero-delay raw target does not need -- one reference is already
    # source-only here. Both declines are stated in the design record rather than defaulted off.
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
    "model_config.VAE_model.horizon_weight_halflife_steps": (
        "the decaying horizon weighting of the reconstruction, which the comparison model's "
        "constructor does not take; null there, and null is the uniform sum it already computes"
    ),
    "model_config.VAE_model.alibi_slope_scale": (
        "the lag-bias seed's slope multiplier. Shipped at 0.0 here -- a FLAT learnable per-lag "
        "bias -- because a decaying seed predicts a lag-0 peak before the model has read anything, "
        "which is a hazard only a cell reading a physiological delay off the lag axis has"
    ),
    "advanced_config.callbacks.early_stopping.enabled": (
        "the training controls: this row stops on val/total_loss where the comparison model runs "
        "its epoch budget out"
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
# The shipped config validates and everything in it reaches something
# --------------------------------------------------------------------------------------
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


def test_the_anchor_stride_equals_the_configured_horizon(shipped):
    """The two are one decision: below the horizon the forecast windows overlap again, above it
    there are target steps no phase ever covers. Asserted rather than defaulted, so a horizon change
    that left the stride behind fails here rather than training a different objective."""
    vae = shipped["model_config"]["VAE_model"]

    assert vae["anchor_stride"] == vae["horizon"]


def test_the_plotting_block_is_the_one_the_inherited_driver_reads(shipped):
    """The callback assembly is inherited and reads this key by name. Renaming the block to match
    this package would disable the per-epoch diagnostic figure with no error anywhere."""
    assert LagAttnCrwsTrainer.PLOT_CONFIG_KEY in shipped["advanced_config"]["callbacks"]


# --------------------------------------------------------------------------------------
# Parity with the model this one is compared against
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# The two runnable variants
# --------------------------------------------------------------------------------------
def test_the_tiny_variant_inherits_the_geometry_that_decides_what_it_exercises(tiny, shipped):
    """Unlike the two-sided cells' smoke variants, including the warm-up budget, the anchor floor
    and the stride: the decoded anchor count IS the resolved floor, stride and horizon, and the input
    adapters' widths ARE the resolved budget's survivor counts, so a shrunken variant would exercise
    an anchor set and an input stream the production run does not have."""
    vae = tiny["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]
    for key in (
        "sequence_length", "raw_per_step", "horizon", "warmup_period", "anchor_stride",
        "lag_floor", "c_y", "c_u", "causal_warmup_budget_steps", "causal_reach_budget_s",
        "causal_align_reference", "causal_norm", "lambda_boundary",
    ):
        assert vae[key] == shipped_vae[key], key
    for path in (
        "advanced_config.trainer.precision",
        "advanced_config.trainer.num_sanity_val_steps",
        "advanced_config.spike_breaker.ema_floor",
        "dataset_config.dataloader_config.dataset_kwargs.trim_minutes",
    ):
        assert _get(tiny, path) == _get(shipped, path), path


def test_the_instrumented_variant_inherits_everything_the_measurement_depends_on(
    smoke_causal, shipped
):
    """Only the run's *scale* and the parked clip are local. Every quantity the two retuned
    constants are stated in -- the block the NLL sums over, the anchor set it is averaged over, the
    weights it is balanced against, the clamp the log-variances sit inside, and the breaker whose
    own EMA is the statistic -- is inherited, or the measurement is of a different model."""
    vae = smoke_causal["model_config"]["VAE_model"]
    shipped_vae = shipped["model_config"]["VAE_model"]

    for key in (
        "causal_warmup_budget_steps", "causal_align_reference", "warmup_period", "anchor_stride",
        "horizon", "raw_per_step", "beta_prior", "beta_schedule", "logvar_clamp", "d_z", "d_model",
        "lambda_ms", "lambda_deriv", "likelihood", "causal_norm",
    ):
        assert vae[key] == shipped_vae[key], key
    assert (
        smoke_causal["advanced_config"]["spike_breaker"]
        == shipped["advanced_config"]["spike_breaker"]
    )


@pytest.mark.parametrize("variant", [_TINY, _SMOKE_CAUSAL], ids=["tiny", "smoke_causal"])
def test_the_resolved_variant_validates_and_builds(variant, tmp_path, loguru_warnings):
    """Resolved first, which is the only way it ever reaches the experiment driver, and then built
    through the driver's own signature sweep -- which reads the committed causal shard the variant
    points at, so a moved or deleted fixture fails here. The decoder emits ``raw_per_step`` raw
    samples per horizon token whatever the resolved budget keeps."""
    resolved = resolve_config_file(str(variant), str(tmp_path))
    graph_model = make_graph_model(
        resolved, **{"general_config.folders_config.out_dir_base": str(tmp_path)}
    )

    graph_model.validate_config()

    assert [message for message in loguru_warnings if "config:" in message] == []
    kwargs = _model_kwargs_from(load_config(str(variant)), LagAttnCrwsTrainer, tmp_path)
    model = SeqVaeLagAttnCrws(**kwargs)
    assert model.target_adapter.linear.in_features == len(kwargs["target_keep_index"])
    assert model.decoder.mean_head.out_features == model.raw_per_step
