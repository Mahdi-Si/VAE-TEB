r"""Checkpoints are self-describing: they carry ``model_class`` and ``model_kwargs``.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no
config file -- the config that produced a run is a mutable file that may not exist by the time
anyone loads the weights -- and ``check_model_class`` can refuse a blob written by a different
model *before* the rebuild is attempted, when the error can still say what is wrong.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import TINY_KWARGS
from train.graph_models_utils import check_model_class, load_checkpoint_strict


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def test_the_checkpoint_carries_the_model_class_and_kwargs(task):
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == "SeqVaeLagAttnRws"
    assert checkpoint["model_kwargs"] == TINY_KWARGS


def test_the_flags_that_change_the_architecture_survive(task):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then
    align nothing and return ``None`` -- which a caller that does not check reads as success."""
    checkpoint = _lightning_style_checkpoint(task())

    for flag in ("sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "max_lag"):
        assert flag in checkpoint["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_base_stamp_survives_the_override(task):
    """``on_save_checkpoint`` here adds a field; it must not replace the base's work. An
    override that skipped ``super()`` would drop ``model_class`` and every guard that reads it
    would silently degrade to its warn-and-continue path."""
    checkpoint = _lightning_style_checkpoint(task())

    assert "model_class" in checkpoint
    assert "model_kwargs" in checkpoint
    assert checkpoint["epoch"] == 3  # and it must not have clobbered Lightning's own fields


def test_the_class_guard_accepts_this_model_and_rejects_another(task):
    checkpoint = _lightning_style_checkpoint(task())

    check_model_class(checkpoint, "SeqVaeLagAttnRws")  # must not raise
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(checkpoint, "SeqVaeLagAttn")


def test_a_checkpoint_round_trips_into_a_fresh_model(task, inputs, tmp_path):
    """The whole contract, end to end: save, reload, rebuild from the blob, same forward out."""
    module = task()
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttnRws")
    rebuilt = SeqVaeLagAttnRws(**blob["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    # Seeded before each forward: the model samples z, so without this the two differ by noise.
    torch.manual_seed(5)
    reference = module.orig_model(*inputs)
    torch.manual_seed(5)
    got = rebuilt(*inputs)

    for key in ("mu_prior", "logvar_post", "mu_full", "logvar_full", "source_kl_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"


def test_the_loss_hyperparameters_reach_the_checkpoint(task):
    """So a run's objective is recoverable from its checkpoint, not only from a mutable
    config file."""
    module = task()

    for name in ("likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule"):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"
