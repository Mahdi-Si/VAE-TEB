r"""Checkpoints are self-describing: they carry ``model_class`` and ``model_kwargs``.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no config
file -- the config that produced a run is a mutable file that may not exist by the time anyone
loads the weights -- and the class guard (tested where it is defined) can refuse a blob written by
a different model *before* the rebuild is attempted.

Checked here: the stamp this model writes, without clobbering Lightning's own fields; a bitwise
save / reload / rebuild round trip, unguarded and under a real channel gate; and that the loss
hyperparameters the evaluation reconciles against reach the checkpoint.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import TINY_KWARGS
from .test_forward_contract import guarded_kwargs


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def test_the_checkpoint_carries_the_model_class_and_kwargs(task):
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfRws"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    # The override adds fields; it must not clobber Lightning's own.
    assert checkpoint["epoch"] == 3


def test_a_checkpoint_round_trips_into_a_fresh_model(task, inputs, tmp_path):
    """The whole contract, end to end: save, reload, rebuild from the blob, same forward out.

    Bitwise rather than within a tolerance: this is the same computation on the same weights on the
    same device, and anything less than exact equality would be evidence that a tensor did not make
    it across. Seeded before each forward because the model samples $z$; without that the two would
    differ by noise and the comparison would say nothing.
    """
    module = task()
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(blob, "SeqVaeLagAttnTrfRws")
    rebuilt = SeqVaeLagAttnTrfRws(**blob["model_kwargs"])
    assert set(rebuilt.state_dict()) == set(module.orig_model.state_dict())
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    torch.manual_seed(5)
    reference = module.orig_model(*inputs)
    torch.manual_seed(5)
    got = rebuilt(*inputs)

    for key in reference:
        assert torch.equal(reference[key], got[key]), f"drift on {key}"


def test_the_loss_hyperparameters_reach_the_checkpoint(task):
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config
    file -- and the objective is what makes the comparison a comparison."""
    module = task()

    for name in ("likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule"):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"


def test_a_guarded_checkpoint_records_the_channel_tuples_it_was_built_at(task, tiny_kwargs):
    """The adapters' input widths depend on the resolved reach budget, so a checkpoint recording
    only the budget in seconds could not be rebuilt without re-running the resolution -- which
    depends on a filter bank, not on the config. The four tuples are therefore in ``model_kwargs``,
    and the rebuilt adapters must come out at the surviving widths rather than the declared ones."""
    kwargs = guarded_kwargs(tiny_kwargs)
    module = task(model_kwargs=kwargs)
    blob = _lightning_style_checkpoint(module)

    for name in ("target_keep_index", "target_delays", "source_keep_index", "source_delays"):
        assert name in blob["model_kwargs"], f"{name} missing; the guard would not be rebuildable"

    rebuilt = SeqVaeLagAttnTrfRws(**blob["model_kwargs"])
    assert rebuilt.target_adapter.linear.in_features == len(kwargs["target_keep_index"])
    assert rebuilt.source_adapter.linear.in_features == len(kwargs["source_keep_index"])
    assert rebuilt.target_adapter.linear.in_features < int(kwargs["c_y"])
    assert load_checkpoint_strict(rebuilt, blob) is not None
