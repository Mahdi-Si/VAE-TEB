r"""Checkpoints are self-describing, and this model's input set and tiling are two of the things
they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file -- and here with no shard either, since
the keep-index and warm-up vectors are resolved from the data once and then travel in
``model_kwargs``. Checked: the stamp is the class and the exact constructor kwargs; a save/reload
round trip rebuilds the same forward; the rebuild needs no shard on disk; and two foreign blobs are
refused whole rather than partly loaded -- a causal-feature blob, whose tensors differ from this
model's only in the decoder head, and a blob of this model at another warm-up budget, which the
class stamp cannot separate and whose input adapter width is what misaligns.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    TASK_HPARAMS,
    TINY_STRIDE,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    make_streams,
    tiny_warmup_kwargs,
)

#: A second warm-up budget, so a cross-budget load can be exercised. The keep-index is shorter, and
#: here that moves the **input adapters** rather than the decoder -- which is the whole reason this
#: cell needs its own version of that test.
_OTHER_KEEP_INDEX = TINY_TARGET_KEEP_INDEX[:20]
_OTHER_WARMUP_STEPS = TINY_TARGET_WARMUP_STEPS[:20]

#: Raw samples a horizon token emits, and therefore the decoder's width at every budget.
RAW_PER_STEP = 16


def _kwargs(**overrides) -> dict:
    """The guarded tiny keyword set, carrying both keys the shipped config states explicitly.

    ``lag_floor`` is written out at its own default rather than left off, because a production blob
    carries it: the config declares it and the driver's signature sweep forwards every declared key.
    A fixture that omitted it would make the assertion below a statement about this helper rather
    than about what a run stamps.
    """
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, lag_floor=0, **overrides)


def _wrapped(cls, kwargs, task_cls=SeqVaeLagAttnCrwsTask):
    """Wrap a freshly-built model in this package's task, as a run does."""
    torch.manual_seed(0)
    return task_cls(cls(**kwargs), lr=1e-3, model_kwargs=dict(kwargs), **TASK_HPARAMS)


def _lightning_style_checkpoint(module) -> dict:
    """Mimic what Lightning hands to ``on_save_checkpoint``."""
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3, "global_step": 42}
    module.on_save_checkpoint(checkpoint)
    return checkpoint


@pytest.fixture
def blob():
    """A checkpoint written by this model at the small guarded geometry."""
    return _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCrws, _kwargs()))


# ---------------------------------------------------------------------------------------
# What the blob carries
# ---------------------------------------------------------------------------------------
def test_the_checkpoint_names_this_model_class(blob):
    """Stamped from the eager model, so it says ``SeqVaeLagAttnCrws`` even though the wrapper adds
    only a phase and a stride to the shared task."""
    assert blob["model_class"] == "SeqVaeLagAttnCrws"
    assert blob["model_kwargs"] == _kwargs()


# ---------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_round_trips_into_a_fresh_model(tmp_path):
    """The whole contract end to end: save, reload, rebuild from the blob alone, same forward."""
    module = _wrapped(SeqVaeLagAttnCrws, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(saved, "SeqVaeLagAttnCrws")
    rebuilt = SeqVaeLagAttnCrws(**saved["model_kwargs"])
    assert load_checkpoint_strict(rebuilt, saved) is not None, (
        "load_checkpoint_strict could not align the saved state dict; the wrapper's "
        "double-prefixed state_dict is no longer being cleaned"
    )

    module.orig_model.eval()
    rebuilt.eval()
    inputs = make_streams(_kwargs())
    torch.manual_seed(5)
    reference = module.orig_model(*inputs, 0, TINY_STRIDE)
    torch.manual_seed(5)
    got = rebuilt(*inputs, 0, TINY_STRIDE)

    for key in ("mu_prior", "logvar_post", "mu_full", "logvar_full", "source_kl_lag_map"):
        assert torch.allclose(reference[key], got[key], atol=1e-6), f"drift on {key}"
    assert torch.equal(reference["anchor_index"], got["anchor_index"])


def test_a_checkpoint_reloads_with_no_shard_present(tmp_path, monkeypatch):
    """The property the whole ``model_kwargs`` stamp exists for, and the one this cell needs more
    than the raw-target sibling it is compared against: the budget is resolved **against the
    shards**, so a blob recording only the threshold could not be rebuilt anywhere the data is not.

    Driven from a directory containing no HDF5 at all, with the working directory moved there, so a
    rebuild that reached for a shard by a relative path would fail rather than quietly find one."""
    module = _wrapped(SeqVaeLagAttnCrws, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = SeqVaeLagAttnCrws(**saved["model_kwargs"])

    assert not list(Path(empty).glob("*.hdf5"))
    assert load_checkpoint_strict(rebuilt, saved) is not None
    assert rebuilt.target_adapter.linear.in_features == len(TINY_TARGET_KEEP_INDEX)
    assert rebuilt.target_warmup_steps == TINY_TARGET_WARMUP_STEPS
    assert rebuilt.decoder_out_channels == RAW_PER_STEP


# ---------------------------------------------------------------------------------------
# Foreign blobs
# ---------------------------------------------------------------------------------------
def test_a_foreign_blob_with_the_guard_skipped_refuses_rather_than_partly_succeeding():
    """The all-or-nothing property the class guard is layered on top of, pinned so a change to the
    loader cannot quietly turn a refusal into a partial warm start.

    ``load_checkpoint_strict`` evaluates a candidate module's alignment *before* loading anything
    and skips it on any missing key, unexpected key or shape mismatch. Against a causal-feature blob
    the misalignment is the decoder head alone -- every encoder tensor matches, which is exactly the
    partial warm start this refusal must not become. What the class guard buys is therefore the
    **message**: without it the failure names two misaligned tensors instead of naming the model
    that wrote the blob."""
    foreign_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCfs, _kwargs()))
    raw_model = SeqVaeLagAttnCrws(**_kwargs())
    before = raw_model.decoder.mean_head.weight.clone()

    assert load_checkpoint_strict(raw_model, foreign_blob) is None
    assert torch.equal(raw_model.decoder.mean_head.weight, before)


def test_a_checkpoint_from_another_warm_up_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their nats are not comparable and their checkpoints are
    mutually unloadable.

    The refusal comes from a different tensor than in the causal-feature cell, and that is the point:
    there the budget moves the decoder head, here it cannot, so what misaligns is the **input
    adapter** whose width the stamped keep-index implies. Both decoders are $R$ wide and align
    perfectly, which is precisely why the adapter has to be the thing that refuses."""
    other_kwargs = _kwargs(
        target_keep_index=_OTHER_KEEP_INDEX, target_warmup_steps=_OTHER_WARMUP_STEPS
    )
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCrws, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnCrws")  # same class: the guard cannot help
    assert other_blob["model_kwargs"]["target_keep_index"] == _OTHER_KEEP_INDEX

    shipped_width_model = SeqVaeLagAttnCrws(**_kwargs())
    assert shipped_width_model.target_adapter.linear.in_features == len(TINY_TARGET_KEEP_INDEX)
    # The decoders agree, so nothing about the forecast's shape says the budgets differ.
    assert shipped_width_model.decoder_out_channels == RAW_PER_STEP
    assert load_checkpoint_strict(shipped_width_model, other_blob) is None

    # And rebuilt from its own kwargs it loads, which is what makes the refusal above a statement
    # about the budget rather than about the blob being broken.
    rebuilt = SeqVaeLagAttnCrws(**other_blob["model_kwargs"])
    assert rebuilt.target_adapter.linear.in_features == len(_OTHER_KEEP_INDEX)
    assert rebuilt.decoder_out_channels == RAW_PER_STEP
    assert load_checkpoint_strict(rebuilt, other_blob) is not None
