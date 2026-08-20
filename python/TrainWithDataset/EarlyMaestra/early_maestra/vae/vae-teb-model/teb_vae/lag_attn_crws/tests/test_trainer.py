r"""The driver turns config into this model: five class attributes, one translation, two hooks.

Almost all of this module's behaviour is inherited, which is the design and also the risk: a stale
attribute produces a run that trains the wrong model, wraps it in the wrong task, writes its
checkpoints under the wrong stem, or checks the wrong loader fields for normalisation -- and none of
those raises.

One attribute is asserted **absent**, and it is the one a reader would expect to be present.
``TARGET_FIELDS`` is inherited as ``('fhr',)`` because this cell forecasts the same raw target as
the model it is compared against; the causal-feature driver overrides it to the two stored feature
blocks, and copying that override here would point the shared entry point's normalisation guard at
fields this objective never scores while leaving the field it does score unchecked.

Two of the additions are specific to this cell and neither has a downstream symptom.
``causal_warmup_budget_steps`` names no constructor argument, so a driver that failed to translate
it would build an **ungated** model that trains to completion having read the region where a
one-sided filter's output is a function of assumed pre-recording history -- on coefficients whose
normalisation constants excluded exactly that region, so those values are on no defined scale. And
the inherited ``causal_standing_message`` is *false* here: it states that the stored features let
step $t$ read up to $974$ s into its own future, which is the property this dataset variant removes.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_crws import trainer as trainer_module
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from train.graph_model_base import GraphModelBase

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    absolutize_dataset_paths,
    hand_seeding_offenders,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_crws.trainer"

#: What the shipped guard resolves to against the committed causal fixture. Pinned so a "guarded"
#: run cannot silently be the unguarded one -- which here changes the input adapters' widths and
#: therefore what the encoders are shown, with no shape anywhere else moving.
#:
#: The two numbers come from two different rules. The **warm-up budget** takes four `fhr_st`
#: channels off the target and never touches the source; the **alignment reference** takes the four
#: `up_st` channels slower than it off the source and never touches the target, because the
#: reference is the target's own maximum.
GUARDED_TARGET_CHANNELS = 98
GUARDED_SOURCE_CHANNELS = 47

#: The metric surface: the raw-signal sibling's, plus three suffixes on both stages, plus the
#: source-null KL on validation. Pinned as a literal the way each sibling package pins its own.
TRACKED_METRIC_COUNT = 77

#: Callables and class attributes this driver may define. A set rather than a count: a count passes
#: a subclass that overrode ``train_model`` in 75 lines while dropping the plot callback.
#:
#: ``create_model`` is on it and is a short wrapper around ``super()``. It exists for two things a
#: run cannot recover afterwards: the run seed, which the tile phase is derived from and which
#: reaches no task in the family by any other route, and the resolved anchor geometry, which nothing
#: in the shipped code ties to the horizon.
_OWN_ATTRIBUTES = {
    "MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM", "TRACKED_METRICS",
    "resolved_warmup", "causal_standing_message", "create_model", "preflight",
    "_build_model_kwargs",
    # Written by ``abc``, not by this class.
    "_abc_impl",
}


@pytest.fixture
def driver(tmp_path):
    """A driver on the tiny config, with its output directories redirected under ``tmp_path``.

    The tiny config rather than the shipped one, and that is not a shortcut: the shipped config's
    shard paths are deliberately non-existent placeholders, and this driver's ``_build_model_kwargs``
    **reads the shards** -- the warm-up boundary is a property of the data and there is nothing to
    read it from otherwise. The tiny variant carries the real geometry and points at the committed
    causal fixture, so every resolved number below is the production one.

    ``setup_config`` is never called -- it would seed, open log sinks and probe MLflow -- so the
    directories are assigned directly.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    instance = LagAttnCrwsTrainer(config_file_path=str(path))
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


class _StubDataModule:
    """A data module that hands out loaders nothing iterates."""

    def __init__(self, config=None):
        self.config = config

    def train_dataloader(self):
        return object()

    def val_dataloader(self):
        return object()


# --------------------------------------------------------------------------------------
# The class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_names_this_model_and_not_the_ones_it_is_compared_against():
    """A stale attribute here trains another model under this model's config, tag and MLflow
    experiment, and nothing anywhere raises."""
    assert LagAttnCrwsTrainer.MODEL_CLS is SeqVaeLagAttnCrws
    assert LagAttnCrwsTrainer.TASK_CLS is SeqVaeLagAttnCrwsTask
    assert LagAttnCrwsTrainer.CHECKPOINT_STEM == "lag-attn-crws"
    for sibling in (LagAttnRwsTrainer, LagAttnCfsTrainer):
        assert LagAttnCrwsTrainer.MODEL_CLS is not sibling.MODEL_CLS
        assert LagAttnCrwsTrainer.CHECKPOINT_STEM != sibling.CHECKPOINT_STEM


def test_the_inherited_drivers_still_build_the_models_they_always_did():
    """The attributes exist so this package can reuse the driver; the reuse is worthless if it
    changed what a launch of either comparison model produces."""
    assert LagAttnRwsTrainer.MODEL_CLS is SeqVaeLagAttnRws
    assert LagAttnRwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS
    assert LagAttnCfsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")


def test_the_driver_declares_exactly_the_named_attributes():
    """Every method it could otherwise override is machinery the comparison rests on: the kwarg
    sweep, the callback assembly, the DDP selection. Redefining any of them here would be a second
    copy free to drift from the one the comparison model runs under."""
    own = {name for name in vars(LagAttnCrwsTrainer) if not name.startswith("__")}

    assert own == _OWN_ATTRIBUTES


def test_the_target_fields_are_inherited_rather_than_re_pointed():
    """The one guard a target-domain change moves -- and this cell does not move the target. The
    causal-feature driver overrides this tuple to the two stored feature blocks; copying that here
    would point the shared entry point's normalisation guard at fields this objective never scores
    while leaving ``fhr``, which it does, unchecked. Absence is the assertion, not the value."""
    assert "TARGET_FIELDS" not in vars(LagAttnCrwsTrainer)
    assert LagAttnCrwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnCrwsTrainer.TARGET_FIELDS is LagAttnRwsTrainer.TARGET_FIELDS


def test_the_metric_surface_is_the_parents_plus_this_cells_seven_entries():
    """Pinned as a literal count beside the decomposition that produces it, so an accidental
    addition fails against a stated number rather than growing the CSV in silence."""
    assert len(LagAttnCrwsTrainer.TRACKED_METRICS) == TRACKED_METRIC_COUNT
    assert len(_TRACKED_METRICS) == TRACKED_METRIC_COUNT - 7
    assert len(set(LagAttnCrwsTrainer.TRACKED_METRICS)) == TRACKED_METRIC_COUNT


def test_the_plot_config_key_is_absent_and_resolves_to_the_inherited_spelling():
    """The callback assembly is inherited whole and reads that literal. Renaming the block to match
    this package would leave the per-epoch diagnostic figure permanently off, with `enabled: true`
    still reading correct to anyone looking at the config and nothing in the log saying why."""
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnCrwsTrainer)
    assert LagAttnCrwsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"


# --------------------------------------------------------------------------------------
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_warmup_budget_reaches_the_constructor_as_the_four_channel_tuples(driver):
    """Translated rather than forwarded: the threshold names no constructor argument, and what the
    network takes is the concrete channel set it resolves to against the shards. A checkpoint
    recording only the threshold could not be rebuilt without re-reading the data."""
    kwargs = driver._build_model_kwargs()

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"])
    assert len(kwargs["target_keep_index"]) == GUARDED_TARGET_CHANNELS
    assert len(kwargs["source_keep_index"]) == GUARDED_SOURCE_CHANNELS
    # The budget's own boundary, read back off the resolved vector rather than declared: the slowest
    # survivor is what the anchor floor has to clear.
    assert max(kwargs["target_warmup_steps"]) == 134


def test_the_delay_keywords_are_absent_because_a_delay_is_not_a_warm_up(driver):
    r"""``target_delays`` and ``source_delays`` reach ``ChannelDelay``, which SHIFTS --
    $\mathrm{out}[t,c] = x[t - \delta_c, c]$ -- leaving content permanently late. A warm-up masks a
    leading region and leaves the rest at its own index, so routing one under the other's name would
    train a different model with every shape intact."""
    import inspect

    kwargs = driver._build_model_kwargs()

    assert "target_delays" not in kwargs
    assert "source_delays" not in kwargs
    assert "target_delays" not in set(
        inspect.signature(SeqVaeLagAttnCrws.__init__).parameters
    )


def test_the_two_new_geometry_keys_reach_the_constructor(driver):
    """``anchor_stride`` and ``lag_floor`` are real constructor arguments rather than translations,
    so the sweep forwards them -- and a config that lost either would build a model at the inert
    defaults with nothing raising."""
    kwargs = driver._build_model_kwargs()

    assert kwargs["anchor_stride"] == 30
    assert kwargs["lag_floor"] == 0


def test_no_decoder_width_key_reaches_the_constructor(driver):
    """``decoder_out_channels`` is deliberately absent from the config, and here it is absent for a
    stronger reason than in the feature-target cells: the raw block's width follows ``raw_per_step``,
    so no configuration can put the decoder and the target on different widths unless a config
    invents one."""
    kwargs = driver._build_model_kwargs()

    assert "decoder_out_channels" not in kwargs
    assert SeqVaeLagAttnCrws(**kwargs).decoder_out_channels == kwargs["raw_per_step"] == 16


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the post-init delta-head zeroing the zero-KL start
    depends on, and the output-head calibration that keeps the init NLL near the trivial
    predictor's."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


def test_the_resolved_kwargs_actually_build_the_model_the_config_describes(driver):
    """The sweep's output is only correct if the constructor accepts it -- and, here, if the model
    it produces reads the widths the budget kept and tiles at the stride the config states."""
    model = SeqVaeLagAttnCrws(**driver._build_model_kwargs())

    assert model.target_adapter.linear.in_features == GUARDED_TARGET_CHANNELS
    assert model.source_adapter.linear.in_features == GUARDED_SOURCE_CHANNELS
    assert model.raw_per_step == 16
    assert model.anchor_stride == model.horizon == 30
    # The declared widths are untouched, which is what the data boundary checks against.
    assert (model.c_y, model.c_u) == (CAUSAL_C_Y, CAUSAL_C_U)
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())


def test_a_config_without_a_budget_builds_the_ungated_model(driver):
    """The arm every "the guard did something" comparison is made against, and the one whose
    silence is the hazard: no gate, no warm-up mask, and both adapters at their declared widths."""
    driver.config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None

    kwargs = driver._build_model_kwargs()

    assert "target_keep_index" not in kwargs
    assert driver.resolved_warmup is None
    assert SeqVaeLagAttnCrws(**kwargs).target_adapter.linear.in_features == CAUSAL_C_Y


# --------------------------------------------------------------------------------------
# What the run's log says about its own standing
# --------------------------------------------------------------------------------------
def test_the_causal_standing_message_is_not_the_inherited_one(driver):
    """The inherited sentence is the most misleading line a run of this cell could carry: it says
    the stored features let step $t$ read up to $974$ s into its own future, which is the property
    this dataset variant removes. What is true here is the pair no other cell can claim together --
    the inputs carry no future and the target has no warm-up -- plus what the budget did about the
    region where one-sidedness costs something."""
    driver._build_model_kwargs()

    message = driver.causal_standing_message()

    assert "974" not in message
    assert "one-sided" in message
    assert "98/102" in message and "47/51" in message


def test_the_ungated_standing_message_says_what_is_being_read_unguarded(driver):
    """The other branch, and it must not read as an absence of a problem: with no budget the model
    reads the leading region where the coefficients are a function of assumed pre-recording
    history, normalised with constants that excluded exactly it."""
    driver.config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None
    driver._build_model_kwargs()

    message = driver.causal_standing_message()

    assert "none" in message
    assert "pre-recording history" in message


def test_create_model_hands_the_task_the_run_seed(driver):
    """The tile phase is derived from it, and it reaches no task in the family by any other route --
    the framework reads it once for ``seed_everything`` and the inherited ``create_model``
    enumerates every ``TASK_CLS`` keyword explicitly. Without this a resumed run would silently
    re-tile every segment."""
    driver.create_model()

    assert driver.pl_model.hparams["seed"] == driver.config["general_config"]["seed"] == 42


def test_create_model_hands_the_task_the_resolved_budget(driver):
    """The run-level warm-up figure is about the channels the budget **dropped**, and a dropped
    channel's own $W'_c$ is exactly what the checkpoint does not carry -- ``model_kwargs`` stamps the
    survivors' vector, because that is what the constructor needs. So the figure cannot be built from
    the network alone, and this is the only route the budget has to the task that draws it.

    A plain attribute rather than a hyperparameter: the four channel tuples are already in the
    checkpoint's ``model_kwargs``, and a second copy of them under another name would be a second
    thing to keep true across a resume."""
    driver.create_model()

    assert driver.pl_model.warmup_budget is driver.resolved_warmup is not None
    assert "warmup_budget" not in driver.pl_model.hparams


def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnCrws)
    assert isinstance(driver.pl_model, SeqVaeLagAttnCrwsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_create_model_logs_the_resolved_anchor_geometry(driver):
    """Nothing in the shipped code ties the anchor stride to the horizon, so a config that shortened
    the horizon and left the stride behind would train a different model with every shape correct.
    The run's own first lines are where that is visible -- and the raw block width is logged beside
    it because nats from this configuration are comparable to no sibling.

    Collected through a loguru sink rather than ``caplog``: the driver logs through loguru, which
    does not propagate to the standard library's logging tree, so a ``caplog`` assertion here would
    pass on a driver that logged nothing at all.
    """
    from loguru import logger

    messages = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        driver.create_model()
    finally:
        logger.remove(sink)

    line = next(m for m in messages if "resolved anchor geometry" in m)
    assert "H=30" in line and "S=30" in line and "F=134" in line
    assert "A_max=5" in line and "T_valid=270" in line
    assert "tiles per sample 4-5" in line
    assert "raw block width H*R=480" in line
    # The refine stack's span, beside the horizon it has to cover. At the shipped horizon_depth=4
    # and horizon_kernel=3 it is 1 + 2*(2^4 - 1) = 31 tokens against H + 1 = 16, so the family's
    # RF >= H + 1 criterion is slack here -- and exactly binding on the horizon arm, at 31 against
    # 31, which is why it is printed rather than assumed: nothing in the shipped code ties
    # horizon_depth to horizon, so an arm that moved one and left the other would be silent.
    assert "horizon receptive field=31 tokens against H+1=31" in line


def test_the_logged_receptive_field_is_read_off_the_built_stack(driver):
    """Not recomputed from the config, and not this package's own arithmetic.

    The depth and the kernel are constructor arguments the net does not keep, so a config leaving
    either at its default would otherwise be reported as whatever the config happened to say. The
    helper is the causal-feature driver's, reached by import, so the family's $\\mathrm{RF} \\ge
    H + 1$ criterion has one implementation rather than one per cell.
    """
    from teb_vae.lag_attn_cfs.trainer import _horizon_receptive_field

    driver.create_model()
    model = driver.pytorch_model
    blocks = model.decoder.core.refine.blocks

    assert _horizon_receptive_field(model) == 1 + (3 - 1) * (2 ** len(blocks) - 1)


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture at this budget's widths and this file's tiling,
    and not into the constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


def test_a_core_checkpoint_from_a_comparison_model_is_refused_before_it_is_loaded(
    driver, tmp_path
):
    """The models share every tensor name -- the decoder head's included, since both emit
    ``raw_per_step`` -- so a foreign blob would align by accident rather than fail. The class stamp
    is the only thing that turns that into a message naming the model that wrote it."""
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeLagAttnRws"}, foreign)
    driver.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        driver.create_model()


# --------------------------------------------------------------------------------------
# The entry point and its guards
# --------------------------------------------------------------------------------------
@pytest.fixture
def recording_main(monkeypatch):
    """Run ``main`` with every expensive step recorded rather than done.

    Patched on the **base** driver, so a test that drives the shared entry point with a comparison
    model's driver is stubbed too.
    """
    calls = []

    def _record(name, result=None):
        def _recorder(self, *args, **kwargs):
            calls.append(name)
            return result

        return _recorder

    monkeypatch.setattr(LagAttnRwsTrainer, "setup_config", _record("setup_config"))
    monkeypatch.setattr(LagAttnRwsTrainer, "train_model", _record("train_model"))
    # On THIS driver rather than on the base for ``create_model`` alone: this one is overridden, and
    # the override reads ``self.pytorch_model`` -- so stubbing only the base would leave the wrapper
    # running against a model the stub never built.
    monkeypatch.setattr(LagAttnCrwsTrainer, "create_model", _record("create_model"))
    monkeypatch.setattr(shared_trainer, "GraphDataModule", _StubDataModule)
    return calls


def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the four guards and the
    resolved-config write. What this package supplies is which driver it constructs -- and that is
    the one thing a delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnCrwsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnCrwsTrainer


def test_setup_config_runs_before_the_model_is_built(recording_main, tmp_path):
    """The order that decides whether a run is seeded, logged and tracked at all."""
    trainer_module.main(_tiny_config_at(tmp_path))

    assert recording_main == ["setup_config", "create_model", "train_model"]


def test_the_four_shared_guards_and_this_drivers_hook_run_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch -- and this driver's own refusals must be in that window too, not after it."""
    order = []
    monkeypatch.setattr(
        LagAttnCrwsTrainer, "setup_config", lambda self: order.append("setup_config")
    )
    for attribute, label in (
        ("_check_stat_path", "stat_path"),
        ("_check_declared_widths_against_shard", "widths"),
        ("_check_raw_target_normalized", "target_normalized"),
        ("_check_causal_budget_resolves", "causal_budget"),
    ):
        monkeypatch.setattr(
            shared_trainer, attribute, lambda config, _label=label, **_: order.append(_label)
        )
    monkeypatch.setattr(
        LagAttnCrwsTrainer, "preflight", classmethod(lambda cls, config: order.append("preflight"))
    )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnCrwsTrainer, "create_model", lambda self: order.append("create_model")
    )

    with pytest.raises(AttributeError):
        # GraphDataModule is stubbed to None, so main dies at train_dataloader() -- after the part
        # under test. The order up to that point is the assertion.
        trainer_module.main(_tiny_config_at(tmp_path))

    assert order[:6] == [
        "stat_path", "widths", "target_normalized", "causal_budget", "preflight", "setup_config",
    ], order


def test_the_normalisation_guard_is_handed_this_drivers_target_fields(tmp_path, monkeypatch):
    """The plumbing a ``trainer_cls=`` wiring mistake breaks, checked by value rather than by
    behaviour. Here the value is the *inherited* tuple, which is exactly why it is worth asserting:
    a guard wired to the causal-feature driver would still run, still refuse something, and would
    ask about two feature blocks this objective never scores."""
    seen = {}
    monkeypatch.setattr(
        shared_trainer,
        "_check_raw_target_normalized",
        lambda config, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(LagAttnCrwsTrainer, "setup_config", lambda self: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)

    with pytest.raises(AttributeError):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen == {"fields": ("fhr",)}


@pytest.mark.parametrize("list_key", ["load_fields", "normalize_fields"])
def test_a_missing_raw_target_field_raises_naming_the_field_and_the_list(
    recording_main, tmp_path, list_key
):
    """Without ``fhr`` in ``normalize_fields`` the target arrives at ~140 bpm, the Gaussian NLL is
    computed against a z-scale variance model, and nothing else raises -- a full run trained on
    nothing."""

    def _drop(config):
        dataloader = config["dataset_config"]["dataloader_config"]
        fields = (
            dataloader["dataset_kwargs"]["load_fields"]
            if list_key == "load_fields"
            else dataloader[list_key]
        )
        fields.remove("fhr")

    with pytest.raises(ValueError, match=rf"'fhr' in .*{list_key}"):
        trainer_module.main(_tiny_config_at(tmp_path, _drop))

    assert "create_model" not in recording_main


def test_the_resolved_config_is_written_beside_the_checkpoints(tmp_path, monkeypatch):
    """A run's own config is otherwise recoverable only from the text of its log or from an MLflow
    artifact whose on-disk location nothing can derive."""
    monkeypatch.setattr(LagAttnCrwsTrainer, "create_model", lambda self: None)
    monkeypatch.setattr(LagAttnCrwsTrainer, "train_model", lambda self, *args: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: _StubDataModule())

    captured = {}

    def _remember(self):
        GraphModelBase.setup_config(self)
        captured["checkpoint_dir"] = self.model_checkpoint_dir

    monkeypatch.setattr(LagAttnCrwsTrainer, "setup_config", _remember)

    trainer_module.main(_tiny_config_at(tmp_path))

    written = Path(captured["checkpoint_dir"]) / shared_trainer.RESOLVED_CONFIG_FILENAME
    assert written.is_file()
    reloaded = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert "base" not in reloaded
    vae = reloaded["model_config"]["VAE_model"]
    assert vae["causal_warmup_budget_steps"] == 134
    assert vae["warmup_period"] == 134
    # Both independently-toggleable mechanisms, so a run is reconstructable from its own artifacts:
    # which clock its channels were put on, and which shard variant it expected to read.
    assert vae["causal_align_reference"] == "target_max"
    assert vae["causal_leg_alignment"] == "envelope"
    # The two-sided guard's record is null here, which is the correct record of "no reach budget
    # was resolved" rather than an omission.
    assert reloaded["model_config"][shared_trainer.RESOLVED_BUDGET_KEY] is None


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative."""
    resolved = trainer_module._resolve_cli_config_path("teb_vae/lag_attn_crws/configs/tiny.yaml")

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


def test_the_argument_is_optional_so_the_run_button_works_at_all():
    """``required=True`` fires before ``RUN_CONFIG`` is ever consulted, which would make the Run
    button unusable no matter what the constant said -- and the refusal, when there is nothing to
    fall back on, has to name the constant rather than only the flag."""
    source = Path(trainer_module.__file__).read_text(encoding="utf-8")

    assert "required=True" not in source
    assert "RUN_CONFIG" in source.split("parser.error(")[1]


def _launched_config(monkeypatch, argv) -> str:
    """Execute the module's ``__main__`` block under ``argv`` and return the config it launched."""
    launched = {}
    monkeypatch.setattr(
        shared_trainer, "main", lambda path, trainer_cls=None: launched.setdefault("path", path)
    )
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.chdir(os.getcwd())

    runpy.run_module(_MODULE_NAME, run_name="__main__", alter_sys=True)

    return launched["path"]


def test_the_command_line_config_wins_over_run_config(monkeypatch, tmp_path):
    requested = str(tmp_path / "requested.yaml")

    assert _launched_config(monkeypatch, ["trainer", "--config", requested]) == requested


def test_a_launch_with_no_command_line_falls_back_to_run_config(monkeypatch):
    launched = _launched_config(monkeypatch, ["trainer"])

    assert Path(launched) == _REPO_ROOT / trainer_module.RUN_CONFIG


# --------------------------------------------------------------------------------------
# Hygiene
# --------------------------------------------------------------------------------------
def test_no_module_in_the_package_seeds_by_hand():
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route; a stray global seed would silently override it while looking like diligence -- and here
    it would additionally move every tile phase, since the seed is one of the four halves of the
    phase key.

    The scan itself is the causal-feature suite's, reached by reference so the two views of this
    package check one rule -- and exempting a seed inside ``torch.random.fork_rng``, which restores
    the stream it found and therefore cannot override anything.
    """
    assert hand_seeding_offenders(Path(__file__).resolve().parents[1]) == []
