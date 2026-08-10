r"""Checkpoints are self-describing, and this model's width is one of the things they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file -- the config that produced a run is a
mutable file that may not exist when anyone loads the weights -- and ``check_model_class`` can
refuse a foreign blob *before* the rebuild is attempted, while the error can still say what is
wrong.

This model raises the stakes on the second field. Its decoder width is not a configuration key: it
is $C_{\mathrm{keep}}$, resolved from the reach budget. Nothing writes that number into the blob,
so the **only** thing that makes a checkpoint rebuildable is ``target_keep_index`` -- and a
checkpoint that lost it would rebuild a model whose decoder emits $c_y$ values per token instead
of $78$, and refuse its own weights.

The machinery under test is the sibling's, reached here through the sibling's task: this model's
own task overrides how its target is built and nothing about checkpointing, so ``model_class`` is
stamped from ``type(orig_model).__name__`` and the contract is a property of the model class and
its kwargs rather than of the wrapper.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    TASK_HPARAMS,
    TINY_KEEP_INDEX,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from train.graph_models_utils import check_model_class, load_checkpoint_strict

#: A second reach budget, so a cross-budget load can be exercised. Two channels rather than three:
#: the decoder width follows, and it must not accidentally match.
_OTHER_KEEP_INDEX = (1, 4)
_OTHER_DELAYS = (0, 1)


def _wrapped(cls, kwargs):
    """Wrap a freshly-built model in the shared task, as a run does."""
    torch.manual_seed(0)
    return SeqVaeLagAttnRwsTask(
        cls(**kwargs), lr=1e-3, model_kwargs=dict(kwargs), **TASK_HPARAMS
    )


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


@pytest.fixture
def blob():
    """A checkpoint written by this model at the small guarded geometry."""
    return _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnFs, tiny_gated_kwargs()))


# ---------------------------------------------------------------------------------------
# What the blob carries
# ---------------------------------------------------------------------------------------
def test_the_checkpoint_names_this_model_class(blob):
    """Stamped from the eager model, so it says ``SeqVaeLagAttnFs`` even though the wrapper is
    the shared task."""
    assert blob["model_class"] == "SeqVaeLagAttnFs"
    assert blob["model_kwargs"] == tiny_gated_kwargs()


def test_the_base_stamp_survives_the_override(blob):
    """The task's ``on_save_checkpoint`` adds a field; it must not replace the base's work. An
    override that skipped ``super()`` would drop ``model_class`` and every guard reading it would
    silently degrade to its warn-and-continue path."""
    assert "model_class" in blob and "model_kwargs" in blob
    assert blob["epoch"] == 3  # and Lightning's own fields are untouched


def test_the_keep_index_is_the_only_record_of_the_decoder_width(blob):
    """The field this model cannot be rebuilt without. ``decoder_out_channels`` is deliberately
    absent: the width is derived from the gate, so a second field recording it could disagree
    with the gate and one of the two would be believed."""
    kwargs = blob["model_kwargs"]

    assert kwargs["target_keep_index"] == TINY_KEEP_INDEX
    assert "decoder_out_channels" not in kwargs
    rebuilt = SeqVaeLagAttnFs(**kwargs)
    assert rebuilt.decoder_out_channels == len(TINY_KEEP_INDEX)


def test_raw_per_step_is_present_because_it_is_still_geometry(blob):
    """It stopped being the decoder width; it did not stop being a geometry input. The trimmed
    grid validates its raw index identities against it, so a blob without it rebuilds a model
    whose geometry cannot be constructed at all."""
    kwargs = blob["model_kwargs"]

    assert kwargs["raw_per_step"] == 16
    rebuilt = SeqVaeLagAttnFs(**kwargs)
    assert rebuilt.geometry.r == 16
    assert rebuilt.decoder_out_channels != rebuilt.raw_per_step


def test_the_flags_that_change_the_architecture_survive(blob):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success."""
    for flag in ("sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "max_lag",
                 "c_y", "target_keep_index", "target_delays"):
        assert flag in blob["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_loss_hyperparameters_reach_the_checkpoint():
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config."""
    module = _wrapped(SeqVaeLagAttnFs, tiny_gated_kwargs())

    for name in ("likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
                 "beta_prior"):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"


# ---------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_round_trips_into_a_fresh_model(inputs, tmp_path):
    """The whole contract end to end: save, reload, rebuild from the blob alone, same forward."""
    module = _wrapped(SeqVaeLagAttnFs, tiny_gated_kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(saved, "SeqVaeLagAttnFs")
    rebuilt = SeqVaeLagAttnFs(**saved["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, saved) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    torch.manual_seed(5)
    reference = module.orig_model(*inputs)
    torch.manual_seed(5)
    got = rebuilt(*inputs)

    for key in ("mu_prior", "logvar_post", "mu_full", "logvar_full", "source_kl_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"


# ---------------------------------------------------------------------------------------
# Foreign blobs
# ---------------------------------------------------------------------------------------
def test_the_class_guard_accepts_this_model_and_rejects_the_sibling(blob):
    check_model_class(blob, "SeqVaeLagAttnFs")  # must not raise

    with pytest.raises(ValueError, match="does not match the active model class") as excinfo:
        check_model_class(blob, "SeqVaeLagAttnRws")
    assert "SeqVaeLagAttnFs" in str(excinfo.value)


def test_a_raw_model_blob_is_refused_by_the_class_check():
    """The load a shared ``core_model_checkpoint`` key makes easy to attempt. Every tensor but the
    decoder's two heads has the same name and shape in both models, so the *interesting* half of
    this is what happens when the guard is skipped -- see the next test."""
    raw_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnRws, tiny_gated_kwargs()))

    assert raw_blob["model_class"] == "SeqVaeLagAttnRws"
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(raw_blob, "SeqVaeLagAttnFs")


def test_with_the_guard_skipped_the_load_still_refuses_rather_than_partly_succeeding():
    """The all-or-nothing property the class guard is layered on top of, pinned so a change to
    the loader cannot quietly turn a refusal into a partial warm start.

    ``load_checkpoint_strict`` evaluates a candidate module's alignment *before* loading anything
    and skips it on any missing key, unexpected key or shape mismatch. The four decoder-head
    tensors mismatch, so it returns ``None`` and no weight is written -- and the driver raises on
    ``None`` rather than training a model it thought it had warm-started. What the class guard
    buys is therefore the **message**: without it the failure names misaligned keys instead of
    naming the model that wrote the blob.
    """
    raw_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnRws, tiny_gated_kwargs()))
    feature_model = SeqVaeLagAttnFs(**tiny_gated_kwargs())
    before = feature_model.decoder.mean_head.weight.clone()

    assert load_checkpoint_strict(feature_model, raw_blob) is None
    assert torch.equal(feature_model.decoder.mean_head.weight, before)


def test_a_checkpoint_from_another_reach_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their nats are not comparable, their decoders are different
    widths, and their checkpoints are mutually unloadable. The refusal comes from the width the
    stamped keep-index implies, which is exactly why that field has to travel."""
    other_kwargs = dict(
        tiny_gated_kwargs(), target_keep_index=_OTHER_KEEP_INDEX, target_delays=_OTHER_DELAYS
    )
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnFs, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnFs")  # same class: the guard cannot help
    assert other_blob["model_kwargs"]["target_keep_index"] == _OTHER_KEEP_INDEX

    shipped_width_model = SeqVaeLagAttnFs(**tiny_gated_kwargs())
    assert shipped_width_model.decoder_out_channels == len(TINY_KEEP_INDEX)
    assert load_checkpoint_strict(shipped_width_model, other_blob) is None

    # And rebuilt from its own kwargs it loads, which is what makes the refusal above a statement
    # about the budget rather than about the blob being broken.
    rebuilt = SeqVaeLagAttnFs(**other_blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == len(_OTHER_KEEP_INDEX)
    assert load_checkpoint_strict(rebuilt, other_blob) is not None
