r"""Checkpoints are self-describing -- ``model_class`` and ``model_kwargs`` -- and round-trip.

A stock Lightning ``.ckpt`` carries neither. With both, a checkpoint can be rebuilt with no config
file, and the class guard can refuse a blob written by a different model *before* the rebuild is
attempted. The guard matters more here than for a model with no siblings: this model and the
conv-Transformer one share every tensor below the encoder inputs, so a foreign blob would align on
most of its tensors and a partial load would report success.

The round trip is this package's own in one respect: the front ends hold their fixed anti-alias
filters as **non-persistent** buffers, so they are absent from the saved ``state_dict`` and rebuilt
by the constructor, which is what lets a strict load align at all.
"""
from __future__ import annotations

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
    """The stamp is what lets a loader refuse a sibling's blob before trying to align it, and the
    kwargs are every architectural flag the rebuild needs. The override must add to Lightning's own
    fields, not replace them."""
    checkpoint = _lightning_style_checkpoint(task())

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfE2E"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    assert checkpoint["epoch"] == 3


def test_a_checkpoint_round_trips_into_a_fresh_model(task, raw_inputs, tmp_path):
    """The whole contract, end to end: save, reload, rebuild from the blob, same forward out.

    Bitwise rather than within a tolerance: this is the same computation on the same weights on the
    same device, and anything less than exact equality would be evidence that a tensor did not make
    it across. Seeded before each forward because the model samples $z$.
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
