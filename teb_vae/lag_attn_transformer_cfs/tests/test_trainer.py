r"""The experiment driver: three class attributes, and where the rest of the diamond resolves.

Every method this driver could override is a piece of machinery a comparison rests on -- the kwarg
sweep, ``create_model``, the callback assembly, the DDP selection, the learning-rate monitor swap,
the pre-flight refusals. Redefining any of them here would be a second copy free to drift from the
one a comparison model runs under, so the class body is three attributes and nothing else.

The three are re-pointed because all three **collide**: both parents set ``MODEL_CLS``, ``TASK_CLS``
and ``CHECKPOINT_STEM``, so resolution order alone would take the causal side, and each omission
fails silently -- a conv-LSTM model, a conv-LSTM task, or two models' checkpoints interleaved under
one stem in whichever output tree they share.

Two members are defined on **both** parents and both must still run: ``_build_model_kwargs`` and
``create_model``. Each calls ``super()``, so the linearisation threads the conv-Transformer's
contributions underneath the causal one's. That is asserted by behaviour rather than by identity,
because identity alone would report only the outermost half. The two training controls are then
checked to monitor metrics the task actually emits.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import absolutize_dataset_paths

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_TINY = _CONFIG_DIR / "tiny.yaml"

#: The tiny variant's model block, whose geometry is the shipped one (test_config_load.py pins
#: that): the values the driver must forward are read here rather than restated as literals.
_TINY_VAE = load_config(str(_TINY))["model_config"]["VAE_model"]

#: The three attributes this class declares, and nothing else.
_OWN_ATTRIBUTES = {"MODEL_CLS", "TASK_CLS", "CHECKPOINT_STEM"}

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


def test_the_colliding_classes_are_re_pointed_at_this_package():
    """Omit either and the driver builds the conv-LSTM model or task with no error anywhere."""
    assert LagAttnTrfCfsTrainer.MODEL_CLS is SeqVaeLagAttnTrfCfs
    assert LagAttnTrfCfsTrainer.TASK_CLS is SeqVaeLagAttnTrfCfsTask


def test_every_drivers_checkpoint_stem_is_distinct():
    """The stem is the checkpoint filename. Two models writing under one stem into a shared output
    tree are indistinguishable by name, and the blob's ``model_class`` stamp is only discoverable
    after loading one."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


# --------------------------------------------------------------------------------------
# Where the rest of the diamond resolves
# --------------------------------------------------------------------------------------
def test_the_preflight_refusals_come_from_the_causal_parent():
    """Every one guards a failure whose symptom is a *number*: a two-sided shard the objective would
    happily score, a floor that admits anchors whose target is pre-recording history, a boundary
    term with no meaning over a tiled set, and the loader fields the tile phase is keyed on."""
    # ``__func__``, not the bound object: ``preflight`` is a classmethod, so each attribute access
    # builds a fresh binding and an identity check on the binding would fail for every class.
    assert LagAttnTrfCfsTrainer.preflight.__func__ is LagAttnCfsTrainer.preflight.__func__
    assert LagAttnTrfCfsTrainer.preflight.__func__ is not LagAttnRwsTrainer.preflight.__func__


@pytest.mark.parametrize("method", ["compile_model_requested", "_build_trainer_kwargs"])
def test_the_encoder_machinery_comes_from_the_conv_transformer_parent(method):
    """Two pieces the causal parent does not define at all, so lookup passes through. Resolving to
    the shared driver instead would drop the step-granular learning-rate monitor and the live
    compile decision -- each silently."""
    assert method not in vars(LagAttnCfsTrainer), f"{method} is defined on the causal parent too"
    assert getattr(LagAttnTrfCfsTrainer, method) is getattr(LagAttnTrfRwsTrainer, method)


# --------------------------------------------------------------------------------------
# Config to constructor: both halves of the cooperative chain fire
# --------------------------------------------------------------------------------------
def test_every_configured_constructor_key_reaches_the_constructor_unchanged(driver):
    """Every non-null ``VAE_model`` leaf that names a constructor parameter arrives at its
    configured value -- the geometry, the encoder block and the revision's switches alike -- while
    the task-level rule pair is resolved against the shard rather than forwarded by name."""
    kwargs = driver._build_model_kwargs()
    parameters = set(inspect.signature(SeqVaeLagAttnTrfCfs.__init__).parameters)

    forwarded = {key for key, value in _TINY_VAE.items() if key in parameters and value is not None}
    assert forwarded >= {"sequence_length", "horizon", "anchor_stride", "encoder_conv_kernels"}
    for key in sorted(forwarded):
        assert kwargs[key] == _TINY_VAE[key], key
    assert "target_phase_fast_cutoff_hz" not in kwargs
    assert len(kwargs["target_scored_horizon"]) == _TINY_VAE["c_y"]


def test_the_warm_up_budget_reaches_the_constructor_as_the_four_channel_tuples(driver):
    """The causal half of the chain. ``causal_warmup_budget_steps`` names no constructor argument at
    all: the driver resolves it against the configured shards into the four tuples the network
    takes, and those are what land in every checkpoint. The widths are the committed
    integer-operator fixture's survivors: the budget takes target channels off, and on the promoted
    unaligned default every source channel survives."""
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


# =================================================================================================
# The training controls
#
# Two decisions about when a run stops and which epochs it keeps, and both are configuration rather
# than code -- which is exactly why they need a test. A monitor the framework never emits is
# indistinguishable from the outside from a disabled control: Lightning treats a monitor it cannot
# find as nothing to stop on, and a ``ModelCheckpoint`` whose monitor never appears saves nothing.
#
# The shipped config is read here rather than the tiny one, because these are production settings.
# =================================================================================================
def test_both_training_controls_monitor_metrics_this_task_emits(
    task, stub_batch, perturb_posterior
) -> None:
    """The early-stopping monitor and both checkpoint criteria name validation metrics the task
    emits, the two checkpoint criteria differ (two callbacks on one monitor would keep the same
    epochs twice), and a zero ``min_delta`` would stop on any improvement at all, which on a noisy
    validation curve is never."""
    shipped = load_config(str(_CONFIG_DIR / "default.yaml"))["advanced_config"]["callbacks"]
    module = task()
    perturb_posterior(module.orig_model)
    _loss, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    early = shipped["early_stopping"]
    checkpoint = shipped["model_checkpoint"]
    for monitor in (early["monitor"], checkpoint["monitor"], checkpoint["secondary_monitor"]):
        stage, name = monitor.split("/", 1)
        assert stage == "val" and name in val_metrics, monitor
    assert checkpoint["secondary_monitor"] != checkpoint["monitor"]
    assert float(early["min_delta"]) > 0.0
