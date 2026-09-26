r"""The driver turns config into this model: six class attributes, one translation, one hook.

Almost all of this module's behaviour is inherited, which is the design and also the risk: a stale
attribute produces a run that trains the wrong model, wraps it in the wrong task, writes its
checkpoints under the wrong stem, or checks the wrong loader fields for normalisation -- and none
of those raises.

Two of the additions are specific to this cell and neither has a downstream symptom.
``causal_warmup_budget_steps`` names no constructor argument, so a driver that failed to translate
it would build an **ungated** model that trains to completion having read the region where a
one-sided filter's output is a function of assumed pre-recording history -- on coefficients whose
normalisation constants excluded exactly that region, so those values are on no defined scale. And
the inherited ``causal_standing_message`` is *false* here: it states that the stored features let
step $t$ read into its own future, which is the property this dataset variant removes.

Also checked: the entry point constructs this driver and runs this driver's pre-flight hook before
``setup_config``, the command line falls back to ``RUN_CONFIG`` for the IDE Run button, no module
seeds by hand, and the shipped early-stopping and second-checkpoint controls are live.
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
from teb_vae.lag_attn_cfs import trainer as trainer_module
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_rws import trainer as shared_trainer
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer

from .conftest import absolutize_dataset_paths, hand_seeding_offenders

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"
_MODULE_NAME = "teb_vae.lag_attn_cfs.trainer"


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
    instance = LagAttnCfsTrainer(config_file_path=str(path))
    instance.output_base_dir = str(tmp_path)
    instance.train_results_dir = str(tmp_path / "train_results")
    instance.model_checkpoint_dir = str(tmp_path / "model_checkpoints")
    return instance


def _tiny_config_at(tmp_path) -> str:
    """Write a resolved, path-absolutised copy of the tiny config into ``tmp_path``.

    Args:
        tmp_path: Directory to write into.

    Returns:
        The written path.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    config["general_config"]["folders_config"]["out_dir_base"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------------------
# Config to constructor
# --------------------------------------------------------------------------------------
def test_the_config_reaches_the_constructor_translated_and_forwarded(driver):
    """The budget is **translated** rather than forwarded: the threshold names no constructor
    argument, and what the network takes is the concrete channel set it resolves to against the
    shards -- which here also fixes the decoder's width, so a checkpoint recording only the
    threshold could not be rebuilt without re-reading the data. Under the delay names the resolved
    vectors would reach ``ChannelDelay``, which SHIFTS, and train a different model with every shape
    intact. The two tiling keys are real constructor arguments and are **forwarded**, since a config
    that lost either would build a model at the inert defaults with nothing raising. And no decoder
    width key reaches the constructor: the width follows the budget."""
    kwargs = driver._build_model_kwargs()
    vae = driver.config["model_config"]["VAE_model"]
    budget = driver.resolved_warmup

    assert budget is not None
    assert "causal_warmup_budget_steps" not in kwargs
    for stream in ("target", "source"):
        resolved = getattr(budget, stream)
        assert list(kwargs[f"{stream}_keep_index"]) == list(resolved.keep_index)
        assert list(kwargs[f"{stream}_warmup_steps"]) == list(resolved.warmup_steps)
    # The slowest survivor is what the anchor floor has to clear, and it is inside the budget.
    assert max(kwargs["target_warmup_steps"]) <= vae["causal_warmup_budget_steps"]
    assert "target_delays" not in kwargs and "source_delays" not in kwargs
    for key in ("anchor_stride", "lag_floor"):
        assert kwargs[key] == vae[key], key
    assert "decoder_out_channels" not in kwargs


def test_init_weights_is_never_a_config_decision(driver):
    """Skipping initialisation would also skip the post-init delta-head zeroing the zero-KL start
    depends on, and the output-head calibration that keeps the init NLL near the trivial
    predictor's."""
    driver.config["model_config"]["VAE_model"]["init_weights"] = False

    assert "init_weights" not in driver._build_model_kwargs()


def test_the_resolved_kwargs_actually_build_the_model_the_config_describes(driver):
    """The sweep's output is only correct if the constructor accepts it -- and, here, if the model
    it produces decodes the width the budget kept and tiles at the stride the config states."""
    kwargs = driver._build_model_kwargs()
    vae = driver.config["model_config"]["VAE_model"]
    model = SeqVaeLagAttnCfs(**kwargs)

    kept_target = len(kwargs["target_keep_index"])
    assert model.decoder_out_channels == kept_target
    assert model.target_adapter.linear.in_features == kept_target
    assert model.source_adapter.linear.in_features == len(kwargs["source_keep_index"])
    for key in ("anchor_stride", "horizon", "c_y", "c_u"):
        assert getattr(model, key) == vae[key], key
    # The stored clock advances nothing: every anchor up to T_valid is decoded.
    assert model.target_forecast_shift is None
    assert model.anchor_ceiling == model.geometry.t_valid
    # The unconditional freeze the DDP strategy relies on.
    assert not any(parameter.requires_grad for parameter in model.lag_attn.W_o.parameters())


def test_a_config_without_a_budget_builds_the_ungated_model(driver):
    """The arm every "the guard did something" comparison is made against, and the one whose
    silence is the hazard: no gate, no warm-up mask, and a decoder emitting every declared
    channel."""
    driver.config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None

    kwargs = driver._build_model_kwargs()

    assert "target_keep_index" not in kwargs
    assert driver.resolved_warmup is None
    assert SeqVaeLagAttnCfs(**kwargs).decoder_out_channels == (
        driver.config["model_config"]["VAE_model"]["c_y"]
    )


# --------------------------------------------------------------------------------------
# What the run's log says about its own standing
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("budgeted", (True, False), ids=("budgeted", "unbudgeted"))
def test_the_causal_standing_message_is_not_the_inherited_one(driver, budgeted):
    """The inherited sentence is the most misleading line a causal run could carry: it says the
    stored features let step $t$ read into its own future, which is the property this dataset
    variant removes. So on both branches -- with a budget and without one -- the line this driver
    logs must be its own rather than the one it inherits."""
    if not budgeted:
        driver.config["model_config"]["VAE_model"]["causal_warmup_budget_steps"] = None
    driver._build_model_kwargs()

    assert driver.causal_standing_message() != LagAttnRwsTrainer.causal_standing_message(driver)


def test_create_model_hands_the_task_the_run_seed(driver):
    """The tile phase is derived from it, and it reaches no task in the family by any other route --
    the framework reads it once for ``seed_everything`` and the inherited ``create_model``
    enumerates every ``TASK_CLS`` keyword explicitly. Without this a resumed run would silently
    re-tile every segment."""
    driver.create_model()

    assert driver.pl_model.hparams["seed"] == driver.config["general_config"]["seed"]


def test_create_model_builds_this_net_and_wraps_it_in_this_task(driver):
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnCfs)
    assert isinstance(driver.pl_model, SeqVaeLagAttnCfsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


def test_the_checkpoint_kwargs_are_the_ones_the_model_was_built_from(driver):
    """So the blob rebuilds into this architecture at this budget's width and this file's tiling,
    and not into the constructor's defaults."""
    driver.create_model()

    assert driver.pl_model._model_kwargs == driver._build_model_kwargs()


def test_a_core_checkpoint_from_a_comparison_model_is_refused_before_it_is_loaded(
    driver, tmp_path
):
    """The models share every tensor name but the decoder head's, so the class stamp is what turns
    a partial-alignment failure into a message naming the model that wrote the blob."""
    foreign = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {}, "model_class": "SeqVaeLagAttnFs"}, foreign)
    driver.config["model_config"]["core_model_checkpoint"] = str(foreign)

    with pytest.raises(ValueError, match="does not match the active model class"):
        driver.create_model()


# --------------------------------------------------------------------------------------
# The entry point and its guards
# --------------------------------------------------------------------------------------
def test_the_entry_point_constructs_this_packages_driver(monkeypatch, tmp_path):
    """``main`` delegates to the shared entry point, which owns the four guards and the
    resolved-config write. What this package supplies is which driver it constructs -- and that is
    the one thing a delegation can get wrong while still running."""
    seen = {}

    def _capture_init(self, config_file_path=None):
        seen["cls"] = type(self)
        raise RuntimeError("stop here")

    monkeypatch.setattr(LagAttnCfsTrainer, "__init__", _capture_init)

    with pytest.raises(RuntimeError, match="stop here"):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen["cls"] is LagAttnCfsTrainer


def test_the_four_shared_guards_and_this_drivers_hook_run_before_setup_config(
    tmp_path, monkeypatch
):
    """Their whole value is failing before the run directory and MLflow run exist on every rank of
    a multi-rank launch -- and this driver's own refusals must be in that window too, not after it."""
    order = []
    monkeypatch.setattr(
        LagAttnCfsTrainer, "setup_config", lambda self: order.append("setup_config")
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
        LagAttnCfsTrainer, "preflight", classmethod(lambda cls, config: order.append("preflight"))
    )
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)
    monkeypatch.setattr(
        LagAttnCfsTrainer, "create_model", lambda self: order.append("create_model")
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
    behaviour: a guard wired to the raw model's driver would still run and still refuse
    *something*, and this config satisfies that model's field."""
    seen = {}
    monkeypatch.setattr(
        shared_trainer,
        "_check_raw_target_normalized",
        lambda config, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(LagAttnCfsTrainer, "setup_config", lambda self: None)
    monkeypatch.setattr(shared_trainer, "GraphDataModule", lambda config: None)

    with pytest.raises(AttributeError):
        trainer_module.main(_tiny_config_at(tmp_path))

    assert seen == {"fields": ("fhr_st", "fhr_ph")}


# --------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------
def test_relative_config_paths_resolve_against_the_repository_root():
    """An IDE's working directory is arbitrary; every documented invocation is repo-root relative."""
    resolved = trainer_module._resolve_cli_config_path("teb_vae/lag_attn_cfs/configs/tiny.yaml")

    assert Path(resolved) == _TINY
    absolute = str(_TINY)
    assert trainer_module._resolve_cli_config_path(absolute) == absolute


def test_run_config_points_at_a_config_that_exists():
    """The IDE Run button resolves through ``RUN_CONFIG``; a stale path breaks it silently."""
    assert trainer_module.RUN_CONFIG is not None
    assert (_REPO_ROOT / trainer_module.RUN_CONFIG).is_file()


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

    The scan itself is :func:`~teb_vae.lag_attn_cfs.tests.conftest.hand_seeding_offenders`, which
    exempts a seed inside ``torch.random.fork_rng``: that restores the stream it found and therefore
    cannot override anything.
    """
    assert hand_seeding_offenders(Path(__file__).resolve().parents[1]) == []


def test_the_seeding_scan_still_reports_an_unforked_seed(tmp_path):
    """Non-vacuity for the exemption above, in both directions: a bare global seed is reported and
    the same call inside a forked RNG is not. Without this, loosening the rule to admit the
    oracle's probe initialisation would silently admit every hand seed in the package."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "bare.py").write_text("import torch\ntorch.manual_seed(0)\n", encoding="utf-8")
    (package / "forked.py").write_text(
        "import torch\nwith torch.random.fork_rng():\n    torch.manual_seed(0)\n",
        encoding="utf-8",
    )

    offenders = hand_seeding_offenders(package)

    assert offenders == ["bare.py: torch.manual_seed("]


# =================================================================================================
# The training controls
#
# Two decisions about when a run stops and which epochs it keeps, and both are configuration rather
# than code -- which is exactly why they need a test. A flag that reaches nothing raises nothing:
# `enabled: false` and a monitor the framework never emits are indistinguishable from the outside,
# and the run simply trains its whole budget out with no line saying the control was inert.
#
# The shipped config is read here rather than the tiny one, because these are production settings:
# the tiny variant runs two epochs and a patience of fifty would be inert there for a reason that
# says nothing about the arm.
# =================================================================================================
def _shipped_callbacks_block():
    """The shipped config's ``advanced_config.callbacks`` block, read off the committed file."""
    from teb_vae.lag_attn.config import load_config

    shipped = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    return load_config(str(shipped))["advanced_config"]["callbacks"]


def test_early_stopping_monitors_a_metric_this_task_emits(
    task, stub_batch, perturb_posterior
) -> None:
    """A run of this family has been observed to reach its composite optimum hundreds of epochs
    before its budget ends, so early stopping is on -- and it has to monitor a name the task
    actually emits, because Lightning treats a monitor it cannot find as nothing to stop on: the
    flag would read as enabled in every artifact and the run would train to its budget anyway.

    The patience and the minimum improvement must both be positive; the exact values are a choice,
    and pinning them would make a retune a test edit rather than a decision.
    """
    block = _shipped_callbacks_block()["early_stopping"]
    module = task()
    perturb_posterior(module.orig_model)
    _loss, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert block["monitor"].split("/", 1)[1] in val_metrics
    assert int(block["patience"]) > 0
    assert float(block["min_delta"]) > 0.0, (
        "a zero min_delta stops on any improvement at all, which on a noisy validation curve is "
        "never"
    )


def test_a_config_naming_no_second_monitor_builds_no_second_callback(driver) -> None:
    """The key is absent-by-default on the shared driver, which is what keeps the two-sided cells at
    one criterion until their own configs opt in. Exercised by removing it rather than by reading
    another cell's config, so what is under test is this driver's branch."""
    from lightning.pytorch.callbacks import ModelCheckpoint

    driver.config["advanced_config"]["callbacks"]["model_checkpoint"].pop(
        "secondary_monitor", None
    )
    captured = {}

    def _capture(callbacks, model=None):
        captured["callbacks"] = callbacks

        class _StubTrainer:
            def fit(self, *args, **kwargs):
                pass

        return _StubTrainer()

    driver.build_trainer = staticmethod(_capture)  # type: ignore[method-assign]
    driver.create_model()
    driver.train_model(object(), object())

    checkpoints = [cb for cb in captured["callbacks"] if isinstance(cb, ModelCheckpoint)]
    assert len(checkpoints) == 1
    assert driver.secondary_checkpoint_callback is None
