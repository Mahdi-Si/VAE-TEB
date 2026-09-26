r"""Checkpoints are self-describing, and for this model the *width* is one of the things they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file, and ``check_model_class`` can refuse a
foreign blob *before* the rebuild is attempted, while the error can still say what is wrong.

**The decoder width is recorded nowhere directly.** It is $C_{\mathrm{keep}}$, resolved from the
reach budget, and ``decoder_out_channels`` is not a keyword of this constructor at all -- so the
stamped ``target_keep_index`` is the only thing that makes a checkpoint rebuildable at its width.

**Three foreign models can write a blob that partly aligns.** This architecture shares the encoders
with ``SeqVaeLagAttnTrfRws`` and the decoder width with ``SeqVaeLagAttnFs``, and everything below the
encoders with both plus ``SeqVaeLagAttnRws``. So the class guard buys the **message**, and
``load_checkpoint_strict`` must still write no weight when the guard is skipped. Two arms of *this*
model at different budgets stamp the same class, so there only the width can refuse the load.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import tiny_gated_kwargs as conv_lstm_tiny_gated_kwargs
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_fs.nets.model import SeqVaeLagAttnTrfFs
from teb_vae.lag_attn_transformer_fs.tests.conftest import (
    TASK_HPARAMS,
    TINY_KEEP_INDEX,
    tiny_gated_kwargs,
)
from teb_vae.lag_attn_transformer_rws.nets.model import SeqVaeLagAttnTrfRws
from train.graph_models_utils import check_model_class, load_checkpoint_strict

#: A second reach budget, so a cross-budget load can be exercised. Its keep-index length differs from
#: the tiny guard's, so the two decoder widths cannot coincide by accident.
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


# ---------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_round_trips_into_a_fresh_model(task, inputs, tmp_path):
    """The whole contract end to end, through this package's own task: save, reload, check the stamp,
    rebuild from the blob alone at the stamped width, same forward.

    Bitwise rather than within a tolerance: the same computation on the same weights on the same
    device, seeded before each forward because the model samples $z$.
    """
    kwargs = tiny_gated_kwargs()
    module = task(model_kwargs=kwargs)
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["model_class"] == "SeqVaeLagAttnTrfFs"
    assert saved["model_kwargs"] == kwargs
    check_model_class(saved, "SeqVaeLagAttnTrfFs")
    rebuilt = SeqVaeLagAttnTrfFs(**saved["model_kwargs"])
    assert rebuilt.decoder_out_channels == len(TINY_KEEP_INDEX)
    assert set(rebuilt.state_dict()) == set(module.orig_model.state_dict())
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

    for key in reference:
        assert torch.equal(reference[key], got[key]), f"drift on {key}"


# ---------------------------------------------------------------------------------------
# Three foreign blobs
# ---------------------------------------------------------------------------------------
def _foreign_blob(name: str) -> dict:
    """A checkpoint written by one of the three models whose tensors partly align with this one.

    Each is built from *its own* suite's keyword set, because the three constructors' schemas differ.
    All four models are at the same tiny guarded geometry, so the tensors that align do align.

    Args:
        name: ``'trf_rws'``, ``'fs'`` or ``'rws'``.

    Returns:
        The Lightning-style checkpoint dict.
    """
    if name == "trf_rws":
        return _lightning_style_checkpoint(
            _wrapped(SeqVaeLagAttnTrfRws, tiny_gated_kwargs())
        )
    cls = SeqVaeLagAttnFs if name == "fs" else SeqVaeLagAttnRws
    return _lightning_style_checkpoint(_wrapped(cls, conv_lstm_tiny_gated_kwargs()))


@pytest.mark.parametrize(
    "name, stamped",
    [("trf_rws", "SeqVaeLagAttnTrfRws"), ("fs", "SeqVaeLagAttnFs"), ("rws", "SeqVaeLagAttnRws")],
)
def test_the_class_guard_fires_first_and_names_the_model_that_wrote_the_blob(name, stamped):
    """The guard's whole product is the message. Each of these blobs aligns in part, so a loader that
    trusted a non-``None`` return would warm-start from a mixture of loaded and random weights and
    report success -- and the failure would name misaligned keys rather than naming the model."""
    blob = _foreign_blob(name)

    assert blob["model_class"] == stamped
    with pytest.raises(ValueError, match="does not match the active model class") as excinfo:
        check_model_class(blob, "SeqVaeLagAttnTrfFs")
    assert stamped in str(excinfo.value)


@pytest.mark.parametrize("name", ["trf_rws", "fs", "rws"])
def test_with_the_guard_skipped_the_load_still_writes_no_weight(name):
    """The all-or-nothing property the class guard is layered on top of: ``load_checkpoint_strict``
    returns ``None`` on a partly aligning blob and leaves every tensor bitwise what it was, so a
    change to the loader cannot quietly turn a refusal into a partial warm start."""
    blob = _foreign_blob(name)
    model = SeqVaeLagAttnTrfFs(**tiny_gated_kwargs())
    before = {
        key: tensor.detach().clone() for key, tensor in model.state_dict().items()
    }

    assert load_checkpoint_strict(model, blob) is None
    moved = [
        key for key, tensor in model.state_dict().items() if not torch.equal(tensor, before[key])
    ]
    assert moved == [], moved


def test_the_feature_siblings_blob_differs_only_in_the_encoder_tensors():
    """Why the message matters most for that one blob: everything but the encoders has the same name
    *and the same shape*, including the decoder's two widened heads -- the two feature models are the
    same target at two encoders and nothing else."""
    blob = _foreign_blob("fs")
    mine = SeqVaeLagAttnTrfFs(**tiny_gated_kwargs()).state_dict()
    theirs = blob["state_dict"]
    stripped = {
        key.split("orig_model.", 1)[-1]: tensor for key, tensor in theirs.items()
    }

    shared = set(mine) & set(stripped)
    assert shared, "the two models share no tensor name; the premise of this file is wrong"
    assert all(mine[key].shape == stripped[key].shape for key in shared)
    for name in ("decoder.mean_head.weight", "decoder.logvar_head.weight"):
        assert name in shared and mine[name].shape[0] == len(TINY_KEEP_INDEX)
    assert all(
        "encoder" in key for key in set(mine) - set(stripped)
    ), sorted(set(mine) - set(stripped))[:5]


# ---------------------------------------------------------------------------------------
# A cross-budget blob, which the class check cannot help with at all
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_from_another_reach_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their decoders are different widths and their checkpoints are
    mutually unloadable. The refusal comes from the width the stamped keep-index implies, which is
    exactly why that field has to travel."""
    other_kwargs = dict(
        tiny_gated_kwargs(), target_keep_index=_OTHER_KEEP_INDEX, target_delays=_OTHER_DELAYS
    )
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnTrfFs, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnTrfFs")  # same class: the guard cannot help
    assert len(_OTHER_KEEP_INDEX) != len(TINY_KEEP_INDEX)

    tiny_width_model = SeqVaeLagAttnTrfFs(**tiny_gated_kwargs())
    assert load_checkpoint_strict(tiny_width_model, other_blob) is None

    # The positive control: rebuilt from its own kwargs the same blob loads, which is what makes the
    # refusal a statement about the budget rather than about a broken blob.
    rebuilt = SeqVaeLagAttnTrfFs(**other_blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == len(_OTHER_KEEP_INDEX)
    assert load_checkpoint_strict(rebuilt, other_blob) is not None
