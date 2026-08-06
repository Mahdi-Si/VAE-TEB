r"""Checkpoints are self-describing: they carry ``model_class`` and ``model_kwargs``.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no config
file -- the config that produced a run is a mutable file that may not exist by the time anyone loads
the weights -- and the class guard can refuse a blob written by a different model *before* the
rebuild is attempted, when the error can still say what is wrong.

The class guard matters more here than for a model with no siblings, and more than for either
sibling. All three architectures share every tensor below the *encoder inputs*: the encoders
themselves, the prior head, the lag attention, the posterior head, the horizon core and the decoder
are the same modules under the same names, and this model differs from the conv-Transformer one only
by two front ends standing where two input adapters stood. A blob from that model would therefore
align on the large majority of its tensors, and a loader that trusted a non-``None`` return would
train from a mixture of loaded and random weights and report success.

One thing about the round trip is this package's own: the front ends hold eight fixed anti-alias
filters as **non-persistent** buffers, so they are absent from the saved ``state_dict`` and are
rebuilt by the constructor. That is what lets a strict load align at all, and it is asserted rather
than assumed -- a persistent buffer would make every checkpoint carry a constant of the architecture
and fail to load the moment the tap count changed, reported as a missing key rather than as what it
is.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import TINY_KWARGS


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def test_the_checkpoint_carries_the_model_class_and_kwargs(task):
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfE2E"
    assert checkpoint["model_kwargs"] == TINY_KWARGS


def test_the_flags_that_change_the_architecture_survive(task):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success.

    ``frontend_kernels`` is this architecture's own addition to the list, and it behaves unlike the
    others: it is a constructor default rather than a config key, so a *production* run's
    ``model_kwargs`` does not carry it at all and the checkpoint is silent about what the front ends
    were built at -- the module constant is the record, and a rebuild inherits whatever that
    constant says today. What this asserts is the case that does arise: whenever the kwargs *do*
    carry it, dropping it must break the rebuild rather than quietly aligning a different stack.
    """
    checkpoint = _lightning_style_checkpoint(task())

    for flag in (
        "sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "warmup_period", "max_lag",
        "frontend_kernels",
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


@pytest.mark.parametrize("sibling", ["SeqVaeLagAttnTrfRws", "SeqVaeLagAttnRws"])
def test_the_class_guard_refuses_a_sibling_blob_in_both_directions(task, sibling):
    """Both directions, and it matters more here than anywhere: this model and the conv-Transformer
    one share every tensor below the encoder inputs, so a foreign blob aligns on most of its
    parameters and a partial alignment reports success."""
    checkpoint = _lightning_style_checkpoint(task())

    check_model_class(checkpoint, "SeqVaeLagAttnTrfE2E")  # must not raise
    # This model's blob, offered to a sibling's loader.
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(checkpoint, sibling)
    # A sibling's blob, offered to this model's loader.
    foreign = dict(checkpoint, model_class=sibling)
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(foreign, "SeqVaeLagAttnTrfE2E")


def test_the_fixed_anti_alias_filters_are_absent_from_the_saved_state_dict(task):
    """Eight of them, one per stage per stream. Non-persistent because they are constants of the
    architecture rather than learned state -- and because a persistent buffer would make the round
    trip below fail the first time the tap count moved, reported as a missing key."""
    module = task()
    checkpoint = _lightning_style_checkpoint(module)

    saved = [name for name in checkpoint["state_dict"] if name.endswith("decimate.fir")]
    assert saved == []
    buffers = [
        name for name, _ in module.orig_model.named_buffers() if name.endswith("decimate.fir")
    ]
    assert len(buffers) == 8, buffers


def test_a_checkpoint_round_trips_into_a_fresh_model(task, raw_inputs, tmp_path):
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
    check_model_class(blob, "SeqVaeLagAttnTrfE2E")
    rebuilt = SeqVaeLagAttnTrfE2E(**blob["model_kwargs"])
    assert set(rebuilt.state_dict()) == set(module.orig_model.state_dict())
    assert load_checkpoint_strict(rebuilt, blob) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    torch.manual_seed(5)
    reference = module.orig_model(*raw_inputs)
    torch.manual_seed(5)
    got = rebuilt(*raw_inputs)

    for key in reference:
        assert torch.equal(reference[key], got[key]), f"drift on {key}"


def test_a_rebuild_from_the_wrong_kwargs_fails_rather_than_partially_aligning(task):
    """The negative control for the round trip above. Dropping one attention block leaves most of
    the state dict alignable -- the front ends and everything below the encoders are untouched -- so
    a rebuild that ignored the recorded kwargs would load a partly random model and report
    nothing."""
    module = task()
    blob = _lightning_style_checkpoint(module)

    shallower = SeqVaeLagAttnTrfE2E(
        **dict(blob["model_kwargs"], target_attention_blocks=1)
    )

    assert set(shallower.state_dict()) != set(module.orig_model.state_dict())


def test_a_rebuild_at_different_front_end_kernels_fails_the_same_way(task):
    """The front-end half of the same argument, and the reason ``frontend_kernels`` has to be in
    ``model_kwargs``: the depthwise weights are $(C, 1, k)$, so a rebuild at another kernel schedule
    produces the same key names at different shapes -- which a strict load rejects and a lenient one
    would quietly skip."""
    module = task()
    blob = _lightning_style_checkpoint(module)

    narrower = SeqVaeLagAttnTrfE2E(**dict(blob["model_kwargs"], frontend_kernels=(3, 3, 3, 3)))

    saved = module.orig_model.state_dict()
    rebuilt = narrower.state_dict()
    assert set(saved) == set(rebuilt)  # same names...
    mismatched = [name for name in saved if saved[name].shape != rebuilt[name].shape]
    assert mismatched, "a different kernel schedule produced identical shapes"
    assert all("frontend" in name for name in mismatched), mismatched


def test_the_loss_hyperparameters_reach_the_checkpoint(task):
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config
    file -- and the objective is what makes the comparison a comparison."""
    module = task()

    for name in ("likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
                 "beta_prior"):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"
