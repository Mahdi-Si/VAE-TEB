r"""The experiment driver: three class attributes, and where the rest of the diamond resolves.

Every method this driver could override is a piece of machinery a comparison rests on -- the kwarg
sweep, ``create_model``, the callback assembly, the DDP selection, the learning-rate monitor swap,
the five pre-flight refusals. Redefining any of them here would be a second copy free to drift from
the one a comparison model runs under, so the class body is three attributes and nothing else.

The three are re-pointed because all three **collide**: both parents set ``MODEL_CLS``, ``TASK_CLS``
and ``CHECKPOINT_STEM``, so resolution order alone would take the causal side, and each omission
fails silently -- a conv-LSTM model, a conv-LSTM task, or two models' checkpoints interleaved under
one stem in whichever output tree they share.

Two members are defined on **both** parents and both must still run: ``_build_model_kwargs`` and
``create_model``. Each calls ``super()``, so the linearisation threads the conv-Transformer's
contributions underneath the causal one's. That is asserted by behaviour rather than by identity,
because identity alone would report only the outermost half.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import absolutize_dataset_paths

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"

#: The three attributes this class declares, and nothing else.
_OWN_ATTRIBUTES = {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}

#: The linearisation the design names.
_EXPECTED_MRO = [
    "LagAttnTrfCfsTrainer",
    "LagAttnCfsTrainer",
    "LagAttnTrfRwsTrainer",
    "LagAttnRwsTrainer",
]

#: Every driver in the family, for the distinct-stem check.
_FAMILY_DRIVERS = (
    LagAttnRwsTrainer,
    LagAttnTrfRwsTrainer,
    LagAttnFsTrainer,
    LagAttnTrfFsTrainer,
    LagAttnCfsTrainer,
    LagAttnTrfCfsTrainer,
)


@pytest.fixture
def driver(tmp_path):
    """A driver on the tiny config, with its shard paths absolutised.

    The tiny config rather than the shipped one because this driver's kwarg sweep **reads the
    shards**: the warm-up boundary is a property of the data. The tiny variant carries the identical
    geometry and budget, which is exactly why it can stand in.
    """
    config = absolutize_dataset_paths(load_config(str(_TINY)))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    built = LagAttnTrfCfsTrainer(config_file_path=str(path))
    built.output_base_dir = str(tmp_path)
    return built


# --------------------------------------------------------------------------------------
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_declares_three_attributes_and_overrides_no_method():
    """``isroutine`` rather than ``callable``: two of the three declared attributes are *classes*,
    and a class is callable, so a plain callability filter would report the re-pointings as
    methods."""
    own = {name for name in vars(LagAttnTrfCfsTrainer) if not name.startswith("_")}
    methods = {
        name
        for name, value in vars(LagAttnTrfCfsTrainer).items()
        if inspect.isroutine(value) or isinstance(value, (classmethod, staticmethod, property))
    }

    assert own == _OWN_ATTRIBUTES
    assert methods == set()


def test_the_diamond_linearises_the_way_the_design_measured_it():
    """The two branches are not disjoint -- both parents set all three class attributes -- so the
    diamond is legal because both descend from the shared driver. A reorder would take the other
    parent's attributes and build the other model."""
    names = [cls.__name__ for cls in LagAttnTrfCfsTrainer.__mro__]

    assert names[: len(_EXPECTED_MRO)] == _EXPECTED_MRO


def test_all_three_colliding_attributes_are_re_pointed():
    assert LagAttnTrfCfsTrainer.MODEL_CLS is SeqVaeLagAttnTrfCfs
    assert LagAttnTrfCfsTrainer.TASK_CLS is SeqVaeLagAttnTrfCfsTask
    assert LagAttnTrfCfsTrainer.CHECKPOINT_STEM == "lag-attn-trf-cfs"
    # Each parent really does set all three: this is what makes the re-pointing necessary rather
    # than defensive.
    for parent in (LagAttnCfsTrainer, LagAttnTrfRwsTrainer):
        for attribute in _OWN_ATTRIBUTES:
            assert attribute in vars(parent), f"{parent.__name__} does not set {attribute}"
    assert LagAttnTrfCfsTrainer.MODEL_CLS is not SeqVaeLagAttnCfs
    assert LagAttnTrfCfsTrainer.TASK_CLS is not SeqVaeLagAttnCfsTask


def test_every_drivers_checkpoint_stem_is_distinct():
    """The stem is the checkpoint filename. Two models writing under one stem into a shared output
    tree are indistinguishable by name, and the blob's ``model_class`` stamp is only discoverable
    after loading one."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


def test_the_inherited_drivers_still_build_the_models_they_always_did():
    """The attributes exist so this package can reuse two drivers; the reuse is worthless if it
    changed what a launch of either comparison model produces."""
    assert LagAttnCfsTrainer.MODEL_CLS is SeqVaeLagAttnCfs
    assert LagAttnCfsTrainer.CHECKPOINT_STEM == "lag-attn-cfs"
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-rws"
    assert LagAttnRwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS


# --------------------------------------------------------------------------------------
# Where the rest of the diamond resolves
# --------------------------------------------------------------------------------------
def test_the_target_domain_attributes_come_from_the_causal_parent():
    """The one guard a target-domain change moves, and the metric surface a target domain decides.
    Both directions on the metric list: a name the framework never emits is a CSV column that is NaN
    in every row, and a metric the task emits that is not here never reaches the CSV at all."""
    assert LagAttnTrfCfsTrainer.TARGET_FIELDS == ("fhr_st", "fhr_ph")
    assert LagAttnTrfCfsTrainer.TARGET_FIELDS is LagAttnCfsTrainer.TARGET_FIELDS
    assert LagAttnTrfCfsTrainer.TRACKED_METRICS is LagAttnCfsTrainer.TRACKED_METRICS

    added = set(LagAttnTrfCfsTrainer.TRACKED_METRICS) - set(_TRACKED_METRICS)
    assert added == {
        f"{stage}/{name}"
        for stage in ("train", "val")
        for name in (
            "pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph",
            "target_warm_frac", "anchors_per_sample",
            "source_lag_warmth_frac_st", "source_lag_warmth_frac_ph",
            "pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi",
            "pred_gap_novel_lo", "pred_gap_novel_mid", "pred_gap_novel_hi",
        )
    } | {"val/kld_source_null"}
    assert set(_TRACKED_METRICS) - set(LagAttnTrfCfsTrainer.TRACKED_METRICS) == set()
    # No duplicates: the collector keys on the name, and a repeat would silently write one column.
    assert len(set(LagAttnTrfCfsTrainer.TRACKED_METRICS)) == len(
        LagAttnTrfCfsTrainer.TRACKED_METRICS
    )


def test_the_preflight_refusals_come_from_the_causal_parent():
    """Five refusals, every one guarding a failure whose symptom is a *number*: a two-sided shard
    the objective would happily score, a floor that admits anchors whose target is pre-recording
    history, a boundary term with no meaning over a tiled set, a source stream missing the block
    that decides whether the start indicator is built, and the two loader fields the tile phase is
    keyed on."""
    assert "preflight" not in vars(LagAttnTrfCfsTrainer)
    # ``__func__``, not the bound object: ``preflight`` is a classmethod, so each attribute access
    # builds a fresh binding and an identity check on the binding would fail for every class.
    assert LagAttnTrfCfsTrainer.preflight.__func__ is LagAttnCfsTrainer.preflight.__func__
    assert "preflight" in vars(LagAttnCfsTrainer)
    assert LagAttnTrfCfsTrainer.preflight.__func__ is not LagAttnRwsTrainer.preflight.__func__


@pytest.mark.parametrize("method", ["compile_model_requested", "_build_trainer_kwargs"])
def test_the_encoder_machinery_comes_from_the_conv_transformer_parent(method):
    """Two pieces the causal parent does not define at all, so lookup passes through. Resolving to
    the shared driver instead would drop the step-granular learning-rate monitor and the live
    compile decision -- each silently."""
    assert method not in vars(LagAttnCfsTrainer), f"{method} is defined on the causal parent too"
    assert getattr(LagAttnTrfCfsTrainer, method) is getattr(LagAttnTrfRwsTrainer, method)


@pytest.mark.parametrize("method", ["_build_model_kwargs", "create_model"])
def test_the_two_colliding_methods_resolve_to_the_causal_parent(method):
    """And both parents define them, which is why the *behaviour* tests below exist: identity alone
    would report the outermost half of a cooperative chain and say nothing about the other."""
    assert method in vars(LagAttnCfsTrainer), method
    assert method in vars(LagAttnTrfRwsTrainer), method
    assert getattr(LagAttnTrfCfsTrainer, method) is getattr(LagAttnCfsTrainer, method)


def test_the_shared_driver_still_owns_the_plot_key_and_the_ddp_hook():
    """``PLOT_CONFIG_KEY`` is deliberately not derived from the package name: a sibling that renames
    it to match its own package gets no figure, no error and nothing in the log saying why."""
    assert LagAttnTrfCfsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnCfsTrainer)
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnTrfRwsTrainer)
    assert LagAttnTrfCfsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy


# --------------------------------------------------------------------------------------
# Config to constructor: both halves of the cooperative chain fire
# --------------------------------------------------------------------------------------
def test_the_geometry_and_the_encoder_block_reach_the_constructor(driver):
    kwargs = driver._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["horizon"] == 30
    assert kwargs["warmup_period"] == 134
    assert kwargs["anchor_stride"] == 13
    assert kwargs["c_y"] == 80
    assert kwargs["c_u"] == 46
    assert kwargs["target_attention_blocks"] == 2  # the tiny variant's own override
    assert kwargs["source_attention_window"] == 8
    assert kwargs["encoder_conv_kernels"] == [5, 9]


def test_the_warm_up_budget_reaches_the_constructor_as_the_four_channel_tuples(driver):
    """The causal half of the chain. ``causal_warmup_budget_steps`` names no constructor argument at
    all: the driver resolves it against the configured shards into the four tuples the network
    takes, and those are what land in every checkpoint.

    The two widths are decided by different rules and, now, by different clocks: the target's by
    the warm-up budget, the source's by an alignment reference of its own that is faster than the
    target's. $39$ rather than $47$ is what that costs, and it is a width every module on the
    source side is built at rather than a preference -- which is why it is pinned here, where the
    resolution that produces it is what is under test.
    """
    kwargs = driver._build_model_kwargs()

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"]) == 76
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_warmup_steps"]) == 46
    assert "target_delays" not in kwargs and "source_delays" not in kwargs
    assert driver.resolved_warmup is not None


def test_the_nullable_encoder_key_survives_the_sweep(driver):
    """The conv-Transformer half of the same chain, and the one a reader assuming "it resolves to
    the causal parent" would lose. An unbounded source encoder *is* ``source_attention_window:
    null``; the inherited sweep drops every null, and the transformer parent re-admits this one."""
    driver.config["model_config"]["VAE_model"]["source_attention_window"] = None

    kwargs = driver._build_model_kwargs()

    assert "source_attention_window" in kwargs
    assert kwargs["source_attention_window"] is None


def test_no_decoder_width_key_reaches_the_constructor(driver):
    """``decoder_out_channels`` is not a keyword of this constructor at all: the width follows the
    gate through the mixin's hook, and a second field naming it could disagree with the target the
    run is actually scored on."""
    assert "decoder_out_channels" not in driver._build_model_kwargs()


def test_a_replaced_encoder_key_is_dropped_rather_than_forwarded(driver):
    """The sweep forwards by name against the real signature, so a copy-pasted key from the
    conv-LSTM causal cell's config cannot crash a launch -- but it also cannot reach anything, which
    is why ``test_config_load.py`` asserts none of them is present in the first place."""
    driver.config["model_config"]["VAE_model"]["lstm_layers"] = 2
    driver.config["model_config"]["VAE_model"]["causal_norm"] = True

    kwargs = driver._build_model_kwargs()

    assert "lstm_layers" not in kwargs
    assert "causal_norm" not in kwargs


def test_the_built_model_is_this_packages_and_carries_both_halves(driver):
    """The end-to-end check that the two class attributes and the two cooperative methods agree:
    the driver's own kwargs build this architecture at this budget."""
    torch.manual_seed(0)
    model = driver.MODEL_CLS(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnTrfCfs)
    assert model.decoder_out_channels == 76
    assert model.anchor_stride == 13
    # The stored clock advances nothing: every anchor up to T_valid is decoded.
    assert model.target_forecast_shift is None
    assert model.anchor_ceiling == model.geometry.t_valid
    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())


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


def test_early_stopping_is_on_and_monitors_a_metric_this_task_emits(
    task, stub_batch, perturb_posterior
) -> None:
    """Both halves of one guarantee, and the second is what makes the first worth having.

    On, because a run of this family has been observed to reach its composite optimum hundreds of
    epochs before its budget ends, and every epoch after that is compute spent on a checkpoint
    nothing will select. And monitoring a name the task actually emits, because Lightning treats a
    monitor it cannot find as nothing to stop on: the flag would read as enabled in every artifact
    and the run would train to its budget anyway.

    The patience is asserted as a *band* rather than a value. What it has to be is long enough that
    a plateau in a noisy validation curve is not read as convergence and short enough to save real
    time; the exact number inside that band is a choice, and pinning it would make a retune a test
    edit rather than a decision.
    """
    block = _shipped_callbacks_block()["early_stopping"]
    module = task()
    perturb_posterior(module.orig_model)
    _loss, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert block["enabled"] is True
    assert block["monitor"] == "val/total_loss"
    assert block["monitor"].split("/", 1)[1] in val_metrics
    assert 40 <= int(block["patience"]) <= 60
    assert float(block["min_delta"]) > 0.0, (
        "a zero min_delta stops on any improvement at all, which on a noisy validation curve is "
        "never"
    )


def test_the_second_checkpoint_criterion_names_a_metric_this_task_emits(
    task, stub_batch, perturb_posterior
) -> None:
    """The composite optimum and the best conditioned forecast are different epochs -- fifty-odd
    apart on a measured run of this family -- and one criterion makes the other epoch's weights
    unrecoverable afterwards.

    The monitor has to be a name the task emits for the same reason early stopping's does: a
    ``ModelCheckpoint`` whose monitor never appears in ``callback_metrics`` saves nothing and says
    nothing, so the second criterion would be a config line and an empty directory.
    """
    block = _shipped_callbacks_block()["model_checkpoint"]
    module = task()
    perturb_posterior(module.orig_model)
    _loss, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert block["secondary_monitor"] == "val/nll_full_block"
    assert block["secondary_monitor"].split("/", 1)[1] in val_metrics
    # A different criterion from the primary one, which is the whole point: two callbacks on one
    # monitor would keep the same epochs twice and cost twice the disk for nothing.
    assert block["secondary_monitor"] != block["monitor"]


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
