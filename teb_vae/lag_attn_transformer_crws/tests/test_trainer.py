r"""The experiment driver: three class attributes, and both halves of the cooperative kwarg chain.

The class body is three attributes, re-pointed because all three **collide**: both parents set
``MODEL_CLS``, ``TASK_CLS`` and ``CHECKPOINT_STEM``, so resolution order alone would take the causal
side, and each omission fails silently -- a conv-LSTM model, a conv-LSTM task, or two models'
checkpoints interleaved under one stem in whichever output tree they share.

Two members are defined on **both** parents and both must still run: ``_build_model_kwargs`` and
``create_model``. Each calls ``super()``, so the linearisation threads the conv-Transformer's
contributions underneath the causal one's. That is asserted by behaviour rather than by identity,
because identity alone would report only the outermost half.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_cfs.trainer import LagAttnCfsTrainer
from teb_vae.lag_attn_crws.trainer import LagAttnCrwsTrainer
from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer
from teb_vae.lag_attn_rws.trainer import LagAttnRwsTrainer
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from teb_vae.lag_attn_transformer_crws.trainer import LagAttnTrfCrwsTrainer
from teb_vae.lag_attn_transformer_e2e.trainer import LagAttnTrfE2ETrainer
from teb_vae.lag_attn_transformer_fs.trainer import LagAttnTrfFsTrainer
from teb_vae.lag_attn_transformer_rws.trainer import LagAttnTrfRwsTrainer

from .conftest import absolutize_dataset_paths

_TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny.yaml"

#: Every driver in the family, for the distinct-stem check. All of them rather than the ones the
#: grid names: the checkpoint stem is a filename, and a filename collides with whatever else is
#: written beside it.
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
def test_every_drivers_checkpoint_stem_is_distinct() -> None:
    """The stem is the checkpoint filename. Two models writing under one stem into a shared output
    tree are indistinguishable by name, and the blob's ``model_class`` stamp is only discoverable
    after loading one -- and a stem this driver forgot to re-point would collide with a parent's."""
    stems = [cls.CHECKPOINT_STEM for cls in _FAMILY_DRIVERS]

    assert len(set(stems)) == len(stems), stems


def test_the_driver_builds_this_packages_model_and_task(driver) -> None:
    """The end-to-end check that the two re-pointed classes and the two cooperative methods agree:
    the driver's own ``create_model`` builds this architecture wrapped in this task, at this
    budget."""
    driver.create_model()

    assert isinstance(driver.pytorch_model, SeqVaeLagAttnTrfCrws)
    assert isinstance(driver.pl_model, SeqVaeLagAttnTrfCrwsTask)
    assert driver.pl_model.orig_model is driver.pytorch_model


# --------------------------------------------------------------------------------------
# Config to constructor: both halves of the cooperative chain fire
# --------------------------------------------------------------------------------------
def test_every_configured_constructor_key_reaches_the_constructor_unchanged(driver) -> None:
    """The geometry, the encoder block and every other configured keyword the constructor takes
    arrive at the value the config states, so the sweep neither drops nor rewrites one."""
    vae = driver.config["model_config"]["VAE_model"]
    constructor_keys = set(inspect.signature(SeqVaeLagAttnTrfCrws.__init__).parameters)

    kwargs = driver._build_model_kwargs()

    forwarded = {key for key in vae if key in constructor_keys and vae[key] is not None}
    assert forwarded, "the config names no constructor keyword; the check is vacuous"
    for key in forwarded:
        assert kwargs[key] == vae[key], key


def test_the_warm_up_budget_reaches_the_constructor_as_the_four_channel_tuples(driver) -> None:
    """The causal half of the chain. ``causal_warmup_budget_steps`` names no constructor argument at
    all: the driver resolves it against the configured shards into the four tuples the network
    takes, and those are what land in every checkpoint."""
    kwargs = driver._build_model_kwargs()

    assert "causal_warmup_budget_steps" not in kwargs
    assert len(kwargs["target_keep_index"]) == len(kwargs["target_warmup_steps"]) > 0
    assert len(kwargs["source_keep_index"]) == len(kwargs["source_warmup_steps"]) > 0
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
