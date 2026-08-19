r"""The experiment driver: three class attributes, and where the rest of the diamond resolves.

Every method this driver could override is a piece of machinery a comparison rests on -- the kwarg
sweep, ``create_model``, the callback assembly, the DDP selection, the learning-rate monitor swap,
the six pre-flight refusals. Redefining any of them here would be a second copy free to drift from
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
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import _TRACKED_METRICS, LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import absolutize_dataset_paths

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: The three attributes this class declares, and nothing else.
_OWN_ATTRIBUTES = {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}

#: The linearisation the design names.
_EXPECTED_MRO = [
    "LagAttnTrfCrwsTrainer",
    "LagAttnCrwsTrainer",
    "LagAttnTrfRwsTrainer",
    "LagAttnRwsTrainer",
]

#: Every driver in the family, for the distinct-stem check. All nine rather than the six the grid
#: names: the checkpoint stem is a filename, and a filename collides with whatever else is written
#: beside it.
_FAMILY_DRIVERS = (
    LagAttnRwsTrainer,
    LagAttnTrfRwsTrainer,
    LagAttnTrfE2ETrainer,
    LagAttnFsTrainer,
    LagAttnTrfFsTrainer,
    LagAttnCfsTrainer,
    LagAttnTrfCfsTrainer,
    LagAttnCrwsTrainer,
    LagAttnTrfCrwsTrainer,
)

#: The three metric suffixes this input domain adds on both stages, and the one it adds on
#: validation alone.
_ADDED_SUFFIXES = (
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
)
_ADDED_VAL_ONLY = ("kld_source_null",)


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
    built = LagAttnTrfCrwsTrainer(config_file_path=str(path))
    built.output_base_dir = str(tmp_path)
    return built


# --------------------------------------------------------------------------------------
# The three class attributes, and what they decide
# --------------------------------------------------------------------------------------
def test_the_driver_declares_three_attributes_and_overrides_no_method() -> None:
    """``isroutine`` rather than ``callable``: two of the three declared attributes are *classes*,
    and a class is callable, so a plain callability filter would report the re-pointings as
    methods."""
    own = {name for name in vars(LagAttnTrfCrwsTrainer) if not name.startswith("_")}
    methods = {
        name
        for name, value in vars(LagAttnTrfCrwsTrainer).items()
        if inspect.isroutine(value) or isinstance(value, (classmethod, staticmethod, property))
    }

    assert own == _OWN_ATTRIBUTES
    assert methods == set()


def test_the_diamond_linearises_the_way_the_design_measured_it() -> None:
    """The two branches are not disjoint -- both parents set all three class attributes -- so the
    diamond is legal because both descend from the shared driver. A reorder would take the other
    parent's attributes and build the other model."""
    names = [cls.__name__ for cls in LagAttnTrfCrwsTrainer.__mro__]

    assert names[: len(_EXPECTED_MRO)] == _EXPECTED_MRO


def test_all_three_colliding_attributes_are_re_pointed() -> None:
    assert LagAttnTrfCrwsTrainer.MODEL_CLS is SeqVaeLagAttnTrfCrws
    assert LagAttnTrfCrwsTrainer.TASK_CLS is SeqVaeLagAttnTrfCrwsTask
    assert LagAttnTrfCrwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-crws"
    # Each parent really does set all three: this is what makes the re-pointing necessary rather
    # than defensive.
    for parent in (LagAttnCrwsTrainer, LagAttnTrfRwsTrainer):
        for attribute in _OWN_ATTRIBUTES:
            assert attribute in vars(parent), f"{parent.__name__} does not set {attribute}"
    assert LagAttnTrfCrwsTrainer.MODEL_CLS is not SeqVaeLagAttnCrws
    assert LagAttnTrfCrwsTrainer.TASK_CLS is not SeqVaeLagAttnCrwsTask


def test_every_drivers_checkpoint_stem_is_distinct() -> None:
    """The stem is the checkpoint filename. Two models writing under one stem into a shared output
    tree are indistinguishable by name, and the blob's ``model_class`` stamp is only discoverable
    after loading one."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


def test_the_inherited_drivers_still_build_the_models_they_always_did() -> None:
    """The attributes exist so this package can reuse two drivers; the reuse is worthless if it
    changed what a launch of either comparison model produces."""
    assert LagAttnCrwsTrainer.MODEL_CLS is SeqVaeLagAttnCrws
    assert LagAttnCrwsTrainer.CHECKPOINT_STEM == "lag-attn-crws"
    assert LagAttnTrfRwsTrainer.CHECKPOINT_STEM == "lag-attn-trf-rws"
    assert LagAttnRwsTrainer.TARGET_FIELDS == ("fhr",)
    assert LagAttnRwsTrainer.TRACKED_METRICS == _TRACKED_METRICS


# --------------------------------------------------------------------------------------
# Where the rest of the diamond resolves
# --------------------------------------------------------------------------------------
def test_the_target_fields_are_the_shared_ancestors_and_are_not_re_pointed() -> None:
    """``("fhr",)`` on every driver of the raw-target row, and re-pointing it is the one edit this
    driver must not make: the shared entry point guards ``normalize_fields`` field by field from
    whatever this tuple says, so a wrong value makes the guard check another model's field and the
    Gaussian NLL is computed against a target at its stored scale with nothing raising."""
    assert LagAttnTrfCrwsTrainer.TARGET_FIELDS == ("fhr",)
    assert "TARGET_FIELDS" not in vars(LagAttnTrfCrwsTrainer)
    assert "TARGET_FIELDS" not in vars(LagAttnCrwsTrainer)
    assert LagAttnTrfCrwsTrainer.TARGET_FIELDS is LagAttnRwsTrainer.TARGET_FIELDS


def test_the_metric_surface_comes_from_the_causal_parent() -> None:
    """The metric surface an input domain decides. Both directions on the list: a name the framework
    never emits is a CSV column that is NaN in every row, and a metric the task emits that is not
    here never reaches the CSV at all."""
    assert LagAttnTrfCrwsTrainer.TRACKED_METRICS is LagAttnCrwsTrainer.TRACKED_METRICS
    assert len(LagAttnTrfCrwsTrainer.TRACKED_METRICS) == 77

    added = set(LagAttnTrfCrwsTrainer.TRACKED_METRICS) - set(_TRACKED_METRICS)
    assert added == {
        f"{stage}/{name}" for stage in ("train", "val") for name in _ADDED_SUFFIXES
    } | {f"val/{name}" for name in _ADDED_VAL_ONLY}
    assert set(_TRACKED_METRICS) - set(LagAttnTrfCrwsTrainer.TRACKED_METRICS) == set()
    # No duplicates: the collector keys on the name, and a repeat would silently write one column.
    assert len(set(LagAttnTrfCrwsTrainer.TRACKED_METRICS)) == len(
        LagAttnTrfCrwsTrainer.TRACKED_METRICS
    )


def test_the_preflight_refusals_come_from_the_causal_parent() -> None:
    """Six refusals, every one guarding a failure whose symptom is a *number*: a two-sided shard the
    objective would happily score, a floor that admits anchors whose inputs are still pre-recording
    history, a boundary term with no meaning over a tiled set, a source stream missing the block that
    decides whether the start indicator is built, the two loader fields the tile phase is keyed on,
    and the validity signal the raw target is masked with."""
    assert "preflight" not in vars(LagAttnTrfCrwsTrainer)
    # ``__func__``, not the bound object: ``preflight`` is a classmethod, so each attribute access
    # builds a fresh binding and an identity check on the binding would fail for every class.
    assert LagAttnTrfCrwsTrainer.preflight.__func__ is LagAttnCrwsTrainer.preflight.__func__
    assert "preflight" in vars(LagAttnCrwsTrainer)
    assert LagAttnTrfCrwsTrainer.preflight.__func__ is not LagAttnRwsTrainer.preflight.__func__


@pytest.mark.parametrize("method", ["compile_model_requested", "_build_trainer_kwargs"])
def test_the_encoder_machinery_comes_from_the_conv_transformer_parent(method) -> None:
    """Two pieces the causal parent does not define at all, so lookup passes through. Resolving to
    the shared driver instead would drop the step-granular learning-rate monitor and the live
    compile decision -- each silently."""
    assert method not in vars(LagAttnCrwsTrainer), f"{method} is defined on the causal parent too"
    assert getattr(LagAttnTrfCrwsTrainer, method) is getattr(LagAttnTrfRwsTrainer, method)


@pytest.mark.parametrize("method", ["_build_model_kwargs", "create_model"])
def test_the_two_colliding_methods_resolve_to_the_causal_parent(method) -> None:
    """And both parents define them, which is why the *behaviour* tests below exist: identity alone
    would report the outermost half of a cooperative chain and say nothing about the other."""
    assert method in vars(LagAttnCrwsTrainer), method
    assert method in vars(LagAttnTrfRwsTrainer), method
    assert getattr(LagAttnTrfCrwsTrainer, method) is getattr(LagAttnCrwsTrainer, method)


def test_the_diagnostic_page_seam_comes_from_the_causal_parent() -> None:
    """The page's forecast rows are the conv-LSTM cell of this row's, reached through the task. This
    package ships no page of its own, and the driver's plot key is what routes a run to one."""
    assert LagAttnTrfCrwsTrainer.PLOT_CONFIG_KEY == "lag_attn_rws_plotting"
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnCrwsTrainer)
    assert "PLOT_CONFIG_KEY" not in vars(LagAttnTrfRwsTrainer)
    assert LagAttnTrfCrwsTrainer.select_ddp_strategy is LagAttnRwsTrainer.select_ddp_strategy


# --------------------------------------------------------------------------------------
# Config to constructor: both halves of the cooperative chain fire
# --------------------------------------------------------------------------------------
def test_the_geometry_and_the_encoder_block_reach_the_constructor(driver) -> None:
    kwargs = driver._build_model_kwargs()

    assert kwargs["sequence_length"] == 300
    assert kwargs["horizon"] == 30
    assert kwargs["warmup_period"] == 133
    assert kwargs["anchor_stride"] == 30
    assert kwargs["c_y"] == 102
    assert kwargs["c_u"] == 51
    assert kwargs["target_attention_blocks"] == 2  # the tiny variant's own override
    assert kwargs["source_attention_window"] == 8
    assert kwargs["encoder_conv_kernels"] == [5, 9]


def test_the_warm_up_budget_reaches_the_constructor_as_the_four_channel_tuples(driver) -> None:
    """The causal half of the chain. ``causal_warmup_budget_steps`` names no constructor argument at
    all: the driver resolves it against the configured shards into the four tuples the network
    takes, and those are what land in every checkpoint."""
    kwargs = driver._build_model_kwargs()

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"]) == 98
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_warmup_steps"]) == 51
    assert "target_delays" not in kwargs and "source_delays" not in kwargs
    assert driver.resolved_warmup is not None


def test_the_nullable_encoder_key_survives_the_sweep(driver) -> None:
    """The conv-Transformer half of the same chain, and the one a reader assuming "it resolves to
    the causal parent" would lose. An unbounded source encoder *is* ``source_attention_window:
    null``; the inherited sweep drops every null, and the transformer parent re-admits this one."""
    driver.config["model_config"]["VAE_model"]["source_attention_window"] = None

    kwargs = driver._build_model_kwargs()

    assert "source_attention_window" in kwargs
    assert kwargs["source_attention_window"] is None


def test_no_decoder_width_key_reaches_the_constructor(driver) -> None:
    """``decoder_out_channels`` is not a keyword of this constructor at all: the raw block is $R$
    samples per horizon token, so no configuration can put the decoder and the target on different
    widths."""
    assert "decoder_out_channels" not in driver._build_model_kwargs()


def test_a_replaced_encoder_key_is_dropped_rather_than_forwarded(driver) -> None:
    """The sweep forwards by name against the real signature, so a copy-pasted key from the
    conv-LSTM cell of this row's config cannot crash a launch -- but it also cannot reach anything,
    which is why ``test_config_load.py`` asserts none of them is present in the first place."""
    driver.config["model_config"]["VAE_model"]["lstm_layers"] = 2
    driver.config["model_config"]["VAE_model"]["causal_norm"] = True

    kwargs = driver._build_model_kwargs()

    assert "lstm_layers" not in kwargs
    assert "causal_norm" not in kwargs


def test_the_built_model_is_this_packages_and_carries_both_halves(driver) -> None:
    """The end-to-end check that the two class attributes and the two cooperative methods agree:
    the driver's own kwargs build this architecture at this budget."""
    torch.manual_seed(0)
    model = driver.MODEL_CLS(**driver._build_model_kwargs())

    assert isinstance(model, SeqVaeLagAttnTrfCrws)
    assert model.decoder_out_channels == 16
    assert model.anchor_stride == 30
    assert not any(isinstance(module, torch.nn.LSTM) for module in model.modules())


def test_the_startup_log_states_the_resolved_anchor_geometry(driver, caplog) -> None:
    """Inherited from the causal parent, and reached through the diamond. Nothing in the shipped code
    ties the stride to the horizon, so a config that shortened one and left the other would train a
    different model with every shape correct; the run's own first lines are where that is
    recoverable months later."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        driver.create_model()
    finally:
        logger.remove(sink_id)

    geometry = [line for line in messages if "resolved anchor geometry" in line]
    assert geometry, messages[-5:]
    for field in ("H=30", "S=30", "F=133", "T_valid=270", "A_max=5", "H*R=480"):
        assert field in geometry[0], field


# --------------------------------------------------------------------------------------
# The Run-button convention
# --------------------------------------------------------------------------------------
def test_the_run_config_constant_names_a_real_file() -> None:
    """The module runs from an IDE's Run button with no command line, so ``RUN_CONFIG`` is the only
    thing standing between the operator and a ``--config is required`` error."""
    from teb_vae.lag_attn_transformer_crws import trainer as trainer_module

    assert trainer_module.RUN_CONFIG is not None
    resolved = trainer_module._resolve_cli_config_path(trainer_module.RUN_CONFIG)
    assert Path(resolved).is_file(), resolved


def test_the_entry_point_states_no_required_argument_and_names_the_constant() -> None:
    """``required=True`` fires before ``RUN_CONFIG`` is ever read, so it makes the Run button
    unusable no matter what the constant says -- and the refusal message has to name both ways to
    supply the value or an operator has no route from the error to the fix."""
    source = (_PACKAGE_DIR / "trainer.py").read_text(encoding="utf-8")

    assert "required=True" not in source
    assert "RUN_CONFIG" in source
    assert "set RUN_CONFIG" in source


def test_no_module_in_the_package_seeds_by_hand() -> None:
    """``general_config.seed`` through the framework's ``configure_determinism`` is the only seeding
    route, and here a stray global seed would additionally move every tile phase, since the seed is
    one of the four halves of the phase key."""
    from .conftest import hand_seeding_offenders

    assert hand_seeding_offenders(_PACKAGE_DIR) == []
