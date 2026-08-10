r"""The driver turns config into this model, and the entry point runs the guards before anything.

Every line of behaviour here is inherited from two parents at once, which is the design and also the
risk: **three class attributes are the entire difference** between building this architecture and
building either model it is compared against, and each failure is silent. Both parents set all three,
so resolution order alone would take the feature side -- a conv-LSTM model, built, trained and
reported under this package's config, tag and MLflow experiment, with nothing anywhere raising.

The rest of the diamond is asserted by *where each name resolves*, because that is the only thing a
diamond can get wrong. Two resolutions carry consequences nothing else records:

* ``TARGET_FIELDS`` and ``TRACKED_METRICS`` come from the feature parent. Without ``fhr_st`` and
  ``fhr_ph`` in ``normalize_fields`` the target arrives at its stored scale, the Gaussian NLL is
  computed against a z-scale variance model, and a multi-day run trains a meaningless objective to
  completion. The guard that catches it lives in the shared entry point and reads the driver it was
  handed, so a ``trainer_cls=`` wiring mistake would leave it checking the *raw* model's ``fhr`` --
  which these configs satisfy.
* ``compile_model_requested`` comes from the conv-Transformer parent, so ``torch.compile`` is **live**
  on a model whose feature-domain ancestor never exercised it. That is the right outcome -- it is the
  transformer encoder that makes compilation worth having -- but it arrives by resolution order rather
  than by anything written down, which is why it is asserted here rather than left to be discovered.

The pre-flight guards get their own assertions for the same reason they exist: their whole value is
failing *before* the run directory, the log sinks and the MLflow run exist on every rank of a
multi-rank launch, so a guard that stopped running would only be noticed by the run it failed to
protect.

There is deliberately no ``test_main.py``. The entry point is a one-line delegation and every guard
belongs to the shared module; both subclassing precedents in this family carry these assertions here
and neither has such a file.
"""
from __future__ import annotations

import inspect
import os
import runpy
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import yaml
from lightning.pytorch.callbacks import Callback, LearningRateMonitor

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from teb_vae.lag_attn_transformer_fs import trainer as trainer_module
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer
from train.graph_model_base import GraphModelBase

from .conftest import absolutize_dataset_paths

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_CONFIG = _CONFIG_DIR / "default.yaml"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_transformer_fs.trainer"

#: Class attributes this driver may declare. A set rather than a count, following both parents'
#: version of this assertion: a count passes a subclass that overrode ``train_model`` in 75 lines
#: while dropping the plot callback.
_OWN_ATTRIBUTES = {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}

#: The linearisation the design measured. The two branches here are **not** disjoint -- both parents
#: set all three class attributes above -- so the diamond is well-formed because both descend from the
#: shared driver, not because their overrides do not collide.
_EXPECTED_MRO = [
    "LagAttnTrfFsTrainer",
    "LagAttnFsTrainer",
    "LagAttnTrfRwsTrainer",
    "LagAttnRwsTrainer",
    "GraphModelBase",
]

#: Every driver in the tree, so the checkpoint stem is checked for distinctness against all of them
#: rather than against the two it is compared with. The stem is the checkpoint *filename*: a
#: copy-pasted one interleaves two models' blobs in whichever output tree they share, and the
#: end-to-end package is as capable of sharing one as the grid's other three.
_FAMILY_DRIVERS = (
    LagAttnRwsTrainer,
    LagAttnFsTrainer,
    LagAttnTrfRwsTrainer,
    LagAttnTrfE2ETrainer,
    LagAttnTrfFsTrainer,
)


class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads off ``self.trainer``, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the whole run, not per epoch, and
    is typed ``float`` because an unlimited run is reported as infinity.
    """

    def __init__(self, estimated_stepping_batches: float = 100.0, max_epochs: int = 10) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def __init__(self, config=None):
        self.config = config

    def train_dataloader(self):
        return object()

    def val_dataloader(self):
        return object()


@pytest.fixture
def driver(tmp_path):
    """A driver on the shipped config, with its output directories redirected under ``tmp_path``.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    instance = LagAttnTrfFsTrainer(config_file_path=str(_CONFIG))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    instance.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return instance


def _tiny_config_at(tmp_path, mutate=None) -> str:
    """Write a resolved, path-absolutised copy of the tiny config into ``tmp_path``.

    Args:
        tmp_path: Directory to write into.
        mutate: Optional callable applied to the config before it is written.

    Returns:
        The written path.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "runs")
    if mutate is not None:
        mutate(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_declares_three_attributes_and_overrides_no_method():
    """Every method it could override is a piece of machinery both comparisons rest on: the kwarg
    sweep, ``create_model``, the callback assembly, the DDP selection, the learning-rate monitor
    swap. Redefining any of them here would be a second copy free to drift from the one a comparison
    model runs under."""
    own = {name for name in vars(LagAttnTrfFsTrainer) if not name.startswith("_")}
    # ``isroutine`` rather than ``callable``: two of the three declared attributes are *classes*, and
    # a class is callable, so a plain callability filter would report the re-pointings as methods.
    methods = {
        name
        for name, value in vars(LagAttnTrfFsTrainer).items()
        if inspect.isroutine(value) or isinstance(value, (classmethod, staticmethod, property))
    }

    assert own == _OWN_ATTRIBUTES
    assert methods == set()


def test_the_diamond_linearises_the_way_the_design_measured_it():
    """The two branches are not disjoint -- both parents set all three class attributes -- so the
    diamond is legal because both descend from the shared driver. A reorder would take the other
    parent's attributes and build the other model."""
    names = [cls.__name__ for cls in LagAttnTrfFsTrainer.__mro__]

    assert names[: len(_EXPECTED_MRO)] == _EXPECTED_MRO


def test_all_three_colliding_attributes_are_re_pointed():
    """All three, because all three collide. Omit ``MODEL_CLS`` and the driver builds a conv-LSTM
    model with no error anywhere -- a run that looks like this package and is not; omit ``TASK_CLS``
    and the same, one layer up; omit ``CHECKPOINT_STEM`` and it writes ``lag-attn-fs-*.ckpt``."""
    assert LagAttnTrfFsTrainer.MODEL_CLS is SeqVaeLagAttnTrfFs
    assert LagAttnTrfFsTrainer.TASK_CLS is SeqVaeLagAttnTrfFsTask
    assert LagAttnTrfFsTrainer.CHECKPOINT_STEM == "lag-attn-trf-fs"
    # Each parent really does set all three: this is what makes the re-pointing necessary rather
    # than defensive.
    for parent in (LagAttnFsTrainer, LagAttnTrfRwsTrainer):
        for attribute in _OWN_ATTRIBUTES:
            assert attribute in vars(parent), f"{parent.__name__} does not set {attribute}"
    assert LagAttnTrfFsTrainer.MODEL_CLS is not SeqVaeLagAttnFs
    assert LagAttnTrfFsTrainer.TASK_CLS is not SeqVaeLagAttnFsTask


def test_every_drivers_checkpoint_stem_is_distinct():
    """The stem is the checkpoint filename. Two models writing under one stem into a shared output
    tree are indistinguishable by name, and the blobs' ``model_class`` stamp is only discoverable
    after loading one."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


def test_the_inherited_drivers_still_build_the_models_they_always_did():
    """The attributes exist so this package can reuse two drivers; the reuse is worthless if it
    changed what a launch of either comparison model produces."""
    assert LagAttnFsTrainer.MODEL_CLS is SeqVaeLagAttnFs
    assert LagAttnFsTrainer.CHECKPOINT_STEM == "lag-attn-fs"
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-rws"
    assert LagAttnRwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS


# --------------------------------------------------------------------------------------
# Where the rest of the diamond resolves
# --------------------------------------------------------------------------------------
def test_the_target_domain_attributes_come_from_the_feature_parent():
    """The one guard a target-domain change moves, and the four columns that make forecasting
    distinguishable from reconstruction. Both directions on the metric list: a name the framework
    never emits is a CSV column that is NaN in every row, and a metric the task emits that is not
    here never reaches the CSV at all."""
    assert LagAttnTrfFsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")
    assert LagAttnTrfFsTrainer.TARGET_FIELDS is LagAttnFsTrainer.TARGET_FIELDS
    assert LagAttnTrfFsTrainer.TRACKED_METRICS is LagAttnFsTrainer.TRACKED_METRICS
    assert len(LagAttnTrfFsTrainer.TRACKED_METRICS) == 78
    added = set(LagAttnTrfFsTrainer.TRACKED_METRICS) - set(_TRACKED_METRICS)
    assert added == {
        f"{stage}/{name}"
        for stage in ("train", "val")
        for name in ("pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph")
    }
    assert set(_TRACKED_METRICS) - set(LagAttnTrfFsTrainer.TRACKED_METRICS) == set()
    # No duplicates: the collector keys on the name, and a repeat would silently write one column.
    assert len(set(LagAttnTrfFsTrainer.TRACKED_METRICS)) == 78


@pytest.mark.parametrize(
    "method",
    ["compile_model_requested", "_build_model_kwargs", "create_model", "_build_trainer_kwargs"],
)
def test_the_encoder_machinery_comes_from_the_conv_transformer_parent(method):
    """Four pieces the feature parent does not define at all, so lookup passes through. The
    consequences differ but the failure mode is the same in each: resolving to the shared driver
    instead would drop the nullable-key re-admission (an unbounded source encoder *is*
    ``source_attention_window: null``), the step-granular learning-rate monitor, the warm-up
    forwarding, and the live compile decision -- each silently."""
    assert method not in vars(LagAttnFsTrainer), f"{method} is defined on the feature parent too"
    assert getattr(LagAttnTrfFsTrainer, method) is getattr(LagAttnTrfRwsTrainer, method)


def test_the_shared_driver_still_owns_the_plot_key_and_the_preflight_hook():
    """``PLOT_CONFIG_KEY`` is deliberately not derived from the package name: a sibling that renames
    it to match its own package gets no figure, no error and nothing in the log saying why.
    ``preflight`` is a documented no-op, and its being absent from ``vars`` here is what keeps that
    property true -- there is nothing behind the hook to inherit, so an override that forgets
    ``super()`` cannot drop a check."""
    assert LagAttnTrfFsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnFsTrainer)
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnTrfRwsTrainer)
    assert "preflight" not in vars(LagAttnTrfFsTrainer)
    assert LagAttnTrfFsTrainer.preflight({}) is None


def test_the_ddp_selection_and_the_hook_name_are_the_shared_drivers():
    """The framework calls ``select_ddp_strategy`` and nothing else, so an underscore-prefixed copy
    would never run. Inherited unchanged through both parents, which is what
    ``tests/test_ddp_strategy.py`` then re-earns as evidence rather than as a claim."""
    assert "select_ddp_strategy" in vars(LagAttnRwsTrainer)
    assert "_select_ddp_strategy" not in vars(LagAttnTrfFsTrainer)
    assert LagAttnTrfFsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy


# --------------------------------------------------------------------------------------
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_geometry_and_the_encoder_block_reach_the_constructor(driver):
    kwargs = driver._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["d_model"] == 128
    assert kwargs["d_z"] == 64
    assert kwargs["horizon"] == 30
    assert kwargs["raw_per_step"] == 16
    assert kwargs["warmup_period"] == 30
    assert kwargs["c_y"] == 109
    assert kwargs["c_u"] == 58
    assert kwargs["max_lag"] == 90
    assert kwargs["coverage_floor"] == 0.9
    assert kwargs["target_attention_blocks"] == 6
    assert kwargs["source_attention_blocks"] == 3
    assert kwargs["source_attention_window"] == 16
    assert kwargs["encoder_num_heads"] == 4
    assert kwargs["encoder_d_ff"] == 512


def test_the_reach_budget_reaches_the_constructor_as_the_four_channel_tuples(driver):
    """Translated rather than forwarded, and here the translation also fixes the decoder's width --
    so a checkpoint recording only ``causal_reach_budget_s`` could not be rebuilt without re-running
    the filter bank."""
    kwargs = driver._build_model_kwargs()

    assert "causal_reach_budget_s" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_delays"]) == 78
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_delays"]) == 29


def test_no_decoder_width_key_reaches_the_constructor(driver):
    """``decoder_out_channels`` is not a keyword of this constructor at all: the width follows the
    gate through the mixin's hook, and a second field naming it could disagree with the target the
    run is actually scored on."""
    assert "decoder_out_channels" not in driver._build_model_kwargs()


def test_a_replaced_encoder_key_is_dropped_rather_than_forwarded(driver):
    """The sweep forwards by name against the real signature, so a copy-pasted key from the
    feature-domain sibling's config cannot crash a launch -- but it also cannot reach anything, which
    is why ``test_config_load.py`` asserts none of them is present in the first place."""
    driver.config["model_config"]["VAE_model"]["lstm_layers"] = 2
    driver.config["model_config"]["VAE_model"]["causal_norm"] = True

    kwargs = driver._build_model_kwargs()

    assert "lstm_layers" not in kwargs
    assert "causal_norm" not in kwargs


def test_a_null_source_window_reaches_the_constructor_as_the_unbounded_encoder(driver):
    """Inherited from the conv-Transformer parent rather than restated, so it is asserted here: the
    shared sweep drops every ``null``, reading it as "leave the constructor default". That is right
    for a key whose null means *unset* and wrong for the one key of this architecture whose null is a
    value -- an unbounded source encoder **is** ``source_attention_window: null``. Dropped, the sweep
    would rebuild the shipped 16-step window while the arm still reported under the unbounded arm's
    name."""
    driver.config["model_config"]["VAE_model"]["source_attention_window"] = None

    kwargs = driver._build_model_kwargs()

    assert "source_attention_window" in kwargs
    assert kwargs["source_attention_window"] is None
    assert SeqVaeLagAttnTrfFs(**kwargs).source_encoder.receptive_field is None


def test_the_inherited_sweep_still_drops_every_other_null(driver):
    """The other direction, and the reason the re-admission is a declared key set rather than a
    blanket change: a ``null`` anywhere else still means "use the constructor's own default", and
    forwarding one would hand the net a ``None`` where it expects a number."""
    from teb_vae.lag_attn_transformer_rws import trainer as sibling_trainer

    driver.config["model_config"]["VAE_model"]["d_head"] = None

    kwargs = driver._build_model_kwargs()

    assert "d_head" not in kwargs
    assert sibling_trainer.NULLABLE_MODEL_KEYS == frozenset({"source_attention_window"})


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the depthwise correction, the post-init delta-head
    zeroing the zero-KL start depends on, and the output-head calibration that keeps the init NLL of
    78 channels near the trivial predictor's."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


def test_the_resolved_kwargs_build_the_model_the_config_describes(driver):
    """The sweep's output is only correct if the constructor accepts it -- and, here, if the model it
    produces decodes the width the reach budget kept while keeping the encoders the config names."""
    model = SeqVaeLagAttnTrfFs(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnTrfFs)
    assert model.decoder_out_channels == 78
    assert model.raw_per_step == 16
    assert len(model.target_encoder.attention_blocks) == 6
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())
    # The depthwise correction ran over the stem: the generic pass Xavier-fills a (C, 1, k) weight at
    # roughly an eighth of the variance-preserving scale, and a count of zero would mean the repair
    # was a no-op on a model that has a stem.
    assert model.n_depthwise_init == 4


# --------------------------------------------------------------------------------------
# create_model
# --------------------------------------------------------------------------------------
def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfFs)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfFsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_create_model_does_not_trip_on_the_absent_causal_norm_flag(driver):
    """The shared ``create_model`` reads ``causal_norm`` off the net through a ``getattr`` with a safe
    default. There is no time-pooling normaliser in this architecture to causalise, so the attribute
    does not exist -- which is also why the causality claim here is unconditional."""
    driver.create_model()

    assert not hasattr(driver.pytorch_model, "causal_norm")


def test_create_model_passes_the_spike_breaker_block_to_the_task(driver):
    """The block is validated by the framework and read by the module -- but nothing forwards it
    automatically, so a driver that forgot would leave a fully configured ``enabled: true`` block
    doing nothing at all."""
    driver.create_model()

    breaker = driver.pl_model.hparams["spike_breaker"]
    assert breaker["enabled"] is True
    assert breaker["comparison_metric"] == "main_loss"
    assert breaker["ema_floor"] >= 1.0e9


def test_create_model_passes_the_loss_hyperparameters_to_the_task(driver):
    """The objective is shared with both comparison models, so these values *are* the comparisons.

    Read from the shipped config rather than restated. What this asserts is the *forwarding*, and a
    literal here would turn every retune of the pair into a failure in an unrelated driver test --
    which is a second place for the number to live and a second place for it to go stale."""
    driver.create_model()

    shipped_vae = load_config(str(_CONFIG))["model_config"]["VAE_model"]
    hparams = driver.pl_model.hparams
    assert hparams["likelihood"] == "gaussian_nll"
    assert hparams["lambda_full"] == 1.0
    assert hparams["lambda_base"] == 1.0
    assert hparams["free_bits"] == 0.0
    assert hparams["beta_schedule"] == shipped_vae["beta_schedule"]
    assert hparams["beta_prior"] == shipped_vae["beta_prior"]
    # The two are a pair: the anchor's restoring force saturates at beta_prior/2 per latent
    # dimension, so what the design fixes is their ratio and not either value.
    assert hparams["beta_prior"] / hparams["beta_schedule"]["end"] == pytest.approx(0.1)


def test_create_model_forwards_the_step_warmup(driver):
    """Reached through the conv-Transformer parent's ``create_model``. It lands in ``hparams`` and
    therefore in every checkpoint, which is where the task reads it back from -- and a driver that did
    not forward it would leave the step path configured and dead."""
    driver.create_model()

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == 2000


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture at this budget's width, and not into the
    constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


@pytest.mark.parametrize(
    "model_class", ["SeqVaeLagAttnFs", "SeqVaeLagAttnTrfRws", "SeqVaeLagAttnRws"]
)
def test_a_core_checkpoint_from_any_sibling_is_refused_before_it_is_loaded(
    driver, tmp_path, model_class
):
    """All three would partially align by accident: the feature sibling shares every tensor name but
    the encoders', the conv-Transformer sibling every name but the decoder head's. The class stamp is
    what turns a partial-alignment failure into a message naming the model that wrote the blob."""
    foreign = tmp_path / f"{model_class}.ckpt"
    torch.save({"state_dict": {}, "model_class": model_class}, foreign)
    driver.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        driver.create_model()


def test_an_unalignable_core_checkpoint_raises_rather_than_training_from_scratch(
    driver, tmp_path
):
    """``load_checkpoint_strict`` returns ``None`` when nothing lines up; it does not raise."""
    unrelated = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"nothing.like.this": torch.zeros(2)}}, unrelated)
    driver.config["model_config"]["core_model_checkpoint"] = str(unrelated)

    with pytest.raises(RuntimeError, match="could not align"):
        driver.create_model()


# --------------------------------------------------------------------------------------
# The trainer kwargs: the learning-rate monitor, and the ramp it exists to show
# --------------------------------------------------------------------------------------
def test_exactly_one_learning_rate_monitor_is_attached_and_it_is_step_granular(driver):
    """A warm-up measured in optimizer steps completes well inside the first epochs, so the
    framework's epoch-granular monitor cannot show it at all -- and a schedule nobody can observe is
    a schedule nobody can tell has silently done nothing. Replaced rather than supplemented, so
    exactly one of them logs, under one name, at one resolution."""
    monitors = [
        callback
        for callback in driver._build_trainer_kwargs([])["callbacks"]
        if isinstance(callback, LearningRateMonitor)
    ]

    assert len(monitors) == 1
    assert monitors[0].logging_interval == "step"


def test_the_frameworks_monitor_is_still_epoch_granular_for_the_feature_sibling(tmp_path):
    """The replacement reaches this driver through one parent, not through the framework: the
    conv-LSTM feature model must keep logging at the granularity its own schedule operates on."""
    sibling_config = _REPO_ROOT / "teb_vae" / "lag_attn_fs" / "configs" / "default.yaml"
    instance = LagAttnFsTrainer(config_file_path=str(sibling_config))
    instance.train_results_dir = str(tmp_path)

    monitors = [
        callback
        for callback in instance._build_trainer_kwargs([])["callbacks"]
        if isinstance(callback, LearningRateMonitor)
    ]

    assert [monitor.logging_interval for monitor in monitors] == ["epoch"]


def test_the_callbacks_the_shared_assembly_supplies_survive_the_replacement(driver):
    """The monitor swap rebuilds the callback list, so it must not drop what ``train_model`` handed
    in -- which is the metrics collector, the CSV writer, the loss plot, the hyperparameter log, the
    checkpointer and the diagnostic page."""
    sentinel = Callback()

    callbacks = driver._build_trainer_kwargs([sentinel])["callbacks"]

    assert sentinel in callbacks


def _trace_lr(module, steps: int, base_lr: float = 1e-3) -> List[float]:
    """Step the module's schedule ``steps`` times and return the rate each step trains at.

    Args:
        module: A task with ``lr_warmup_steps`` on its hparams and a trainer attached.
        steps: How many scheduler steps to record.
        base_lr: The optimizer's configured rate, which the factor multiplies.

    Returns:
        The learning rate *before* each step, so index $s$ is the rate step $s$ trains at.
    """
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    schedule = module.build_lr_scheduler(optimizer)
    scheduler = schedule["scheduler"] if isinstance(schedule, dict) else schedule
    values = []
    for _ in range(steps):
        values.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
    return values


def test_the_configured_warmup_reaches_the_scheduler_and_the_rate_moves_per_step(tmp_path):
    """The seam that fails silently, traced end to end rather than described: the config names a
    ramp, the conv-Transformer parent's ``create_model`` forwards it onto ``hparams``, and the feature
    parent's task -- through the diamond -- has to build a schedule that actually moves the rate on
    every optimizer step.

    One test here rather than a ported module: the sibling suite owns the ramp's arithmetic, and what
    is new is that it survives two levels of multiple inheritance. A ramp attached at
    ``interval: "epoch"`` would take ``lr_warmup_steps`` *epochs* and this trace would be flat.
    """
    driver = LagAttnTrfFsTrainer(config_file_path=_tiny_config_at(tmp_path))
    driver.create_model()
    warmup_steps = load_config(str(_TINY))["general_config"]["lr_warmup_steps"]
    driver.pl_model.trainer = _FakeTrainer()

    # The optimizer is built at the rate the *driver* resolved from the config, so the trace is a
    # statement about the run rather than about a rate this test chose.
    base_lr = float(driver.lr)
    trace = _trace_lr(driver.pl_model, warmup_steps + 2, base_lr=base_lr)

    assert int(driver.pl_model.hparams["lr_warmup_steps"]) == warmup_steps > 0
    # Linear from lr/N, reaching the full rate on the last ramp step and holding after it.
    assert trace == pytest.approx(
        [base_lr * min(1.0, (step + 1) / warmup_steps) for step in range(warmup_steps + 2)]
    )
    assert len(set(trace)) > 1, "the rate never moved; the ramp is attached at epoch granularity"


# --------------------------------------------------------------------------------------
# The entry point and the pre-flight ordering
# --------------------------------------------------------------------------------------
@pytest.fixture
def recording_main(monkeypatch):
    """Run ``main`` with every expensive step recorded rather than done.

    Patched on **this** driver rather than on the shared base, unlike the feature sibling's version of
    this fixture. ``create_model`` here resolves to the conv-Transformer parent's, which calls
    ``super().create_model()`` and then reaches for ``self.pl_model``; stubbing only the base would
    leave that wrapper running against a model that was never built.
    """
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(LagAttnTrfFsTrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(LagAttnTrfFsTrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(LagAttnTrfFsTrainer, "train_model", _record("train_model"))
    monkeypatch.setattr(shared_trainer, "GraphDataModule", _StubDataModule)
    return calls


def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the four guards, the temporary
    resolved-config file and the resolved-config write. What this package supplies is which driver it
    constructs -- and that is the one thing a delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnTrfFsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnTrfFsTrainer


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all: building the model
    first means no seeding, no log sinks, nowhere to write, and ``mlflow_logger is None`` -- which
    silently drops the MLflow callback from the fit."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "create_model", "train_model"]


def test_all_four_pre_flight_guards_and_the_driver_hook_run_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of a
    multi-rank launch. The driver's own ``preflight`` runs last of the five and still before
    ``setup_config``; here it is the shared no-op, so what is asserted is the *position* it would
    occupy."""
    order = []
    monkeypatch.setattr(
        LagAttnTrfFsTrainer, "setup_config", lambda self: order.append("setup_config")
    )
    for attribute, label in (
        ("_check_stat_path", "stat_path"),
        ("_check_declared_widths_against_shard", "widths"),
        ("_check_raw_target_normalized", "target_normalized"),
        ("_check_causal_budget_resolves", "causal_budget"),
    ):
        monkeypatch.setattr(
            # ``**_`` because the normalisation guard is handed the driver's TARGET_FIELDS; these
            # stubs record the order and have no opinion about any guard's arguments.
            shared_trainer, attribute, lambda config, _label=label, **_: order.append(_label)
        )
    monkeypatch.setattr(
        LagAttnTrfFsTrainer,
        "preflight",
        classmethod(lambda cls, config: order.append("preflight")),
    )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnTrfFsTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:6] == [
        "stat_path", "widths", "target_normalized", "causal_budget", "preflight", "setup_config",
    ], order


def test_the_normalisation_guard_is_handed_this_drivers_target_fields(tmp_path, monkeypatch):
    """The plumbing a ``trainer_cls=`` wiring mistake breaks, and the plumbing the *diamond* can
    break: checked by value rather than by behaviour, because a guard wired to the raw driver would
    still run and still refuse *something*, and this config satisfies its check."""
    seen = {}
    monkeypatch.setattr(
        shared_trainer,
        "_check_raw_target_normalized",
        lambda config, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(LagAttnTrfFsTrainer, "setup_config", lambda self: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)

    with pytest.raises(AttributeError):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen == {"fields": ("fhr_st", "fhr_ph")}


@pytest.mark.parametrize("field", ["fhr_st", "fhr_ph"])
@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_a_missing_target_block_raises_naming_the_field_and_the_list(
    recording_main, tmp_path, list_key, field
):
    """Without both blocks in ``normalize_fields`` the target arrives at its stored scale, the
    Gaussian NLL is meaningless, and nothing else raises -- a full run trained on nothing. Both
    fields, because "one of them is missing" is not a fix anybody can act on."""

    def _drop(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        fields = (
            dataloader["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else dataloader[list_key]
        )
        fields.remove(field)

    with pytest.raises(ValueError, match=rf"'{field}' in .*{list_key}"):
        trainer_module.main(_tiny_config_at(tmp_path, _drop))

    assert "create_model" not in recording_main


def test_an_unsatisfiable_reach_budget_raises_before_any_training_happens(
    recording_main, tmp_path
):
    """A budget whose delay outruns ``warmup_period`` would zero-fill trained anchors -- and here it
    would also silently re-size the decoder, since the width follows the survivors."""

    def _too_deep(config):
        config["model_config"]["VAE_model"]["causal_reach_budget_s"] = 240.0

    with pytest.raises(ValueError, match="warmup_period"):
        trainer_module.main(_tiny_config_at(tmp_path, _too_deep))

    assert "create_model" not in recording_main


def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive. It matters more here than for the raw models:
    the recorded budget is what says how wide the decoder was, and therefore what the run's nats were
    summed over."""
    monkeypatch.setattr(LagAttnTrfFsTrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnTrfFsTrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        # setup_config is what creates the run directories, so the write must follow it; the real one
        # runs, and only the directory it chose is recorded.
        GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(LagAttnTrfFsTrainer, "setup_config", _remember)

    trainer_module.main(_tiny_config_at(tmp_path))

    written = Path(captured["checkpoint_dir"]) / shared_trainer.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    # Fully resolved: the encoder block the tiny variant does not name is present.
    assert reloaded["model_config"]["VAE_model"]["source_attention_window"] == 16
    record = reloaded["model_config"][shared_trainer.RESOLVED_BUDGET_KEY]
    assert record is not None
    assert record["causal_reach_budget_s"] == 120
    assert len(record["target_keep_index"]) == 78


# --------------------------------------------------------------------------------------
# torch.compile: live here, and it arrives by resolution order
# --------------------------------------------------------------------------------------
def test_the_shipped_config_leaves_compilation_off(driver):
    """Off is the shipped value, so the baseline is an eager run and a compiled run is a deliberate
    act. What this asserts is the *driver's* reading of the key; the config test owns the key."""
    assert driver.compile_model_requested() is False


def test_the_key_is_live_here_and_inert_for_the_feature_sibling(driver):
    """The one behavioural difference the diamond's resolution order introduces, and the reason it is
    asserted rather than left to be discovered. The feature parent does not read the key at all -- its
    net's LSTM encoders defeat inductor unconditionally -- and this model replaced those encoders, so
    lookup passes through to the parent that honours it. If the key stayed inert an operator could set
    ``compile: true``, see nothing in the log, and believe they had measured a compiled run."""
    driver.config["advanced_config"]["trainer"]["compile"] = True

    assert driver.compile_model_requested() is True
    assert LagAttnFsTrainer.compile_model_requested(object()) is False
    assert "compile_model_requested" in vars(LagAttnRwsTrainer)


def test_compilation_and_attention_checkpointing_are_refused_together(driver):
    """The one genuine inductor blocker still reachable from this config surface, and the guard comes
    across with the decision. Silently dropping either would give a run that is neither the compiled
    one nor the checkpointed one."""
    driver.config["advanced_config"]["trainer"]["compile"] = True
    driver.config["model_config"]["VAE_model"]["attention_grad_checkpoint"] = True

    with pytest.raises(ValueError) as excinfo:
        driver.compile_model_requested()

    message = str(excinfo.value)
    assert "attention_grad_checkpoint" in message and "compile" in message


def test_the_objective_is_never_the_thing_compiled():
    """Only the forward is compiled. The task reaches ``compute_loss`` through ``orig_model``, which
    keeps the data-dependent ``kld_active_frac`` indexing out of the graph -- so this is the line that
    makes the key safe to honour at all, and it is asserted rather than trusted."""
    import inspect

    from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask

    source = inspect.getsource(SeqVaeLagAttnRwsTask.compute_loss_and_metrics)

    assert "self.orig_model.compute_loss(" in source
    assert "self.model.compute_loss(" not in source


# --------------------------------------------------------------------------------------
# The command line and the Run-button convention
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative, so
    the resolver must anchor there and leave absolute paths alone."""
    resolved = trainer_module._resolve_cli_config_path(
        "teb_vae/lag_attn_transformer_fs/configs/tiny.yaml"
    )

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


def test_the_argument_is_optional_so_the_run_button_works_at_all():
    """``required=True`` fires before ``RUN_CONFIG`` is ever consulted, which would make the Run
    button unusable no matter what the constant said, and a non-``None`` argparse default would make
    the constant unreachable. The refusal, when there is nothing to fall back on, has to name the
    constant rather than only the flag."""
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "required=True" not in source
    assert "default=None" in source
    assert "RUN_CONFIG" in source.split("parser.error(")[1]


def test_the_module_docstring_carries_both_launch_lines():
    """It is the only place an operator finds the multi-rank invocation, and the only place the
    Run-button trap is written down: a Run-button launch of ``default.yaml`` is a *single* process
    whose seven ``cuda_devices`` make the framework spawn DDP workers underneath it."""
    docstring = trainer_module.__doc__ or ""

    assert "python -m teb_vae.lag_attn_transformer_fs.trainer" in docstring
    assert "torchrun --nproc_per_node=7" in docstring
    assert "TEB_RUN_STAMP" in docstring
    assert "cuda_devices" in docstring and "RUN_CONFIG" in docstring
    for name in ("tiny.yaml", "smoke_hie.yaml", "default.yaml"):
        assert name in docstring, name


def _launched_config(monkeypatch, argv) -> str:
    """Execute the module's ``__main__`` block under ``argv`` and return the config it launched.

    The block is re-executed in a fresh namespace, so the ``main`` it calls is the one this package's
    module imports from the shared entry point -- patching that name is what makes the launch
    observable without running a fit.

    Args:
        monkeypatch: The pytest fixture.
        argv: The command line, including the program name.

    Returns:
        The config path the entry point was called with.
    """
    launched = {}
    monkeypatch.setattr(
        shared_trainer, "main", lambda path, trainer_cls=None: launched.setdefault("path", path)
    )
    monkeypatch.setattr(sys, "argv", list(argv))
    # The block chdirs to the repo root when it is not already there; entering it under monkeypatch
    # is what restores the working directory afterwards.
    monkeypatch.chdir(os.getcwd())

    runpy.run_module(_MODULE_NAME, run_name="__main__", alter_sys=True)

    return launched["path"]


def test_the_command_line_config_wins_over_run_config(monkeypatch, tmp_path):
    """``RUN_CONFIG`` exists and is valid, so an entry point that ignored ``--config`` would launch a
    full production run when a smoke run was asked for."""
    requested = str(tmp_path / "requested.yaml")

    assert _launched_config(monkeypatch, ["trainer", "--config", requested]) == requested


def test_a_launch_with_no_command_line_falls_back_to_run_config(monkeypatch):
    """Which is what makes an IDE Run button work at all."""
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG


# --------------------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------------------
def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route; a stray global seed would silently override it while looking like diligence."""
    package_dir = Path(__file__).resolve().parents[1]
    offenders = []
    for path in package_dir.rglob("*.py"):
        if "tests" in path.parts:
            continue  # tests seed themselves for reproducibility, legitimately
        source = path.read_text(encoding="utf-8")
        for pattern in ("torch.manual_seed", "seed_everything", "np.random.seed"):
            if pattern in source:
                offenders.append(f"{path.name}: {pattern}")

    assert offenders == []
