r"""Checkpoints are self-describing: they carry ``model_class`` and ``model_kwargs``.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no config
file -- the config that produced a run is a mutable file that may not exist by the time anyone
loads the weights -- and the class guard can refuse a blob written by a different model *before*
the rebuild is attempted, when the error can still say what is wrong.

The class guard matters more here than it does for a model with no siblings. This architecture and
the one it is compared against share every tensor below the encoders: the prior head, the lag
attention, the posterior head, the horizon core and the decoder are the same modules under the same
names. A blob from the other model would therefore *partially* align, and a loader that trusted a
non-``None`` return would train from a mixture of loaded and random weights and report success.
"""
from __future__ import annotations

import pytest
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


def test_the_flags_that_change_the_architecture_survive(task):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success. The four
    encoder-shape keys are the ones this architecture adds: each changes a parameter count, so a
    blob missing one cannot be rebuilt at all."""
    checkpoint = _lightning_style_checkpoint(task())

    for flag in (
        "sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "max_lag",
        "encoder_conv_kernels", "encoder_conv_dilations", "encoder_d_ff",
        "target_attention_blocks", "source_attention_blocks", "source_attention_window",
    ):
        assert flag in checkpoint["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_base_stamp_survives_the_override(task):
    """``on_save_checkpoint`` adds a field; it must not replace the base's work. An override that
    skipped ``super()`` would drop ``model_class`` and every guard that reads it would silently
    degrade to its warn-and-continue path."""
    checkpoint = _lightning_style_checkpoint(task())

    assert "model_class" in checkpoint
    assert "model_kwargs" in checkpoint
    assert checkpoint["epoch"] == 3  # and it must not have clobbered Lightning's own fields


def test_the_class_guard_refuses_a_checkpoint_from_the_comparison_model(task):
    """The two models share every tensor below the encoders, so a foreign blob aligns in part."""
    checkpoint = _lightning_style_checkpoint(task())

    check_model_class(checkpoint, "SeqVaeLagAttnTrfRws")  # must not raise
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(checkpoint, "SeqVaeLagAttnRws")

    foreign = dict(checkpoint, model_class="SeqVaeLagAttnRws")
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(foreign, "SeqVaeLagAttnTrfRws")


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


def test_a_rebuild_from_the_wrong_kwargs_fails_rather_than_partially_aligning(task):
    """The negative control for the round trip above. Dropping one attention block leaves most of
    the state dict alignable -- everything below the encoders is untouched -- so a rebuild that
    ignored the recorded kwargs would load a partly random model and report nothing."""
    module = task()
    blob = _lightning_style_checkpoint(module)

    shallower = SeqVaeLagAttnTrfRws(
        **dict(blob["model_kwargs"], target_attention_blocks=1)
    )

    assert set(shallower.state_dict()) != set(module.orig_model.state_dict())


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
