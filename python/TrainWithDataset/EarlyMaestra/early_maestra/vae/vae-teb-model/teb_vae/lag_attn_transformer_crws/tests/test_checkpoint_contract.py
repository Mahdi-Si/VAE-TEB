r"""What a checkpoint carries, and what a load refuses.

A run's channel set must be recoverable from its checkpoint **alone**, with no shard present. That
is not a convenience here: the warm-up budget is resolved *against the shards*, the input adapters'
widths and availability patterns depend on the four tuples it resolves to, and a blob recording only
the threshold could not be rebuilt anywhere the data is not.

Two refusals are layered on top of it, and they catch different mistakes. The class guard separates
this model from its siblings -- exercised on the conv-LSTM twin, the nearest miss, which shares
every geometry key and most tensor names -- and buys the *message*: without it the failure names
misaligned keys instead of naming the model that wrote the blob. The width the stamped keep-index
implies separates two arms of *this* model at two budgets, which the class guard cannot, because
both stamp the same class name.

The second refusal lands on a different tensor than it does in the causal-feature cells, and that is
worth stating rather than discovering: there the budget moves the decoder head, here it cannot -- the
raw block is $R$ wide at every budget -- so what misaligns is the **input adapter**.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_crws.nets.model import SeqVaeLagAttnCrws
from teb_vae.lag_attn_crws.tests.conftest import (
    tiny_warmup_kwargs as conv_lstm_tiny_warmup_kwargs,
)
from teb_vae.lag_attn_transformer_crws.nets.model import SeqVaeLagAttnTrfCrws
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from train.graph_models_utils import check_model_class, load_checkpoint_strict

from .conftest import (
    TASK_HPARAMS,
    TINY_STRIDE,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    make_streams,
    tiny_warmup_kwargs,
)

#: A second warm-up budget, so a cross-budget load can be exercised. The keep-index is shorter, so
#: the *input adapter's* width follows it -- the decoder's does not, and cannot.
_OTHER_KEEP_INDEX = TINY_TARGET_KEEP_INDEX[:20]
_OTHER_WARMUP_STEPS = TINY_TARGET_WARMUP_STEPS[:20]


def _kwargs(**overrides) -> dict:
    """The guarded tiny keyword set, carrying both keys the shipped config states explicitly.

    ``lag_floor`` is written out at its own default rather than left off, because a production blob
    carries it: the config declares it and the driver's signature sweep forwards every declared key.
    """
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, lag_floor=0, **overrides)


def _wrapped(cls, kwargs, task_cls=SeqVaeLagAttnTrfCrwsTask):
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
    return _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnTrfCrws, _kwargs()))


# ---------------------------------------------------------------------------------------
# What the blob carries
# ---------------------------------------------------------------------------------------
def test_the_checkpoint_names_this_model_class(blob) -> None:
    """Stamped from the eager model, so it says ``SeqVaeLagAttnTrfCrws`` even though the wrapper is
    an empty diamond of two tasks."""
    assert blob["model_class"] == "SeqVaeLagAttnTrfCrws"
    assert blob["model_kwargs"] == _kwargs()


# ---------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_round_trips_into_a_fresh_model(tmp_path) -> None:
    """The whole contract end to end: save, reload, rebuild from the blob alone, same forward."""
    module = _wrapped(SeqVaeLagAttnTrfCrws, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(saved, "SeqVaeLagAttnTrfCrws")
    rebuilt = SeqVaeLagAttnTrfCrws(**saved["model_kwargs"])
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


def test_a_checkpoint_reloads_with_no_shard_present(tmp_path, monkeypatch) -> None:
    """The property the whole ``model_kwargs`` stamp exists for, and the one this input domain
    needs more than the two-sided cells: the budget is resolved **against the shards**, so a blob
    recording only the threshold could not be rebuilt anywhere the data is not.

    Driven from a directory containing no HDF5 at all, with the working directory moved there, so a
    rebuild that reached for a shard by a relative path would fail rather than quietly find one."""
    module = _wrapped(SeqVaeLagAttnTrfCrws, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = SeqVaeLagAttnTrfCrws(**saved["model_kwargs"])

    assert not list(Path(empty).glob("*.hdf5"))
    assert load_checkpoint_strict(rebuilt, saved) is not None
    assert rebuilt.target_adapter.linear.in_features == len(TINY_TARGET_KEEP_INDEX)
    assert rebuilt.target_warmup_steps == TINY_TARGET_WARMUP_STEPS


# ---------------------------------------------------------------------------------------
# Foreign blobs
# ---------------------------------------------------------------------------------------
def test_the_conv_lstm_cells_blob_is_refused_whole_with_or_without_the_class_guard() -> None:
    """The nearest miss in the whole family: the two cells of this row share every geometry key, the
    same target, the same budget, the same tiling and most tensor names. Only the encoder differs.

    The class guard refuses it by name. Skipped, ``load_checkpoint_strict`` must still refuse rather
    than partly succeed: it evaluates a candidate module's alignment *before* loading anything and
    skips it on any missing key, unexpected key or shape mismatch, so what the guard buys is the
    **message** -- the model that wrote the blob rather than a list of misaligned keys.
    """
    foreign_blob = _lightning_style_checkpoint(
        _wrapped(
            SeqVaeLagAttnCrws,
            conv_lstm_tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, lag_floor=0),
        )
    )

    assert foreign_blob["model_class"] == "SeqVaeLagAttnCrws"
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(foreign_blob, "SeqVaeLagAttnTrfCrws")

    transformer = SeqVaeLagAttnTrfCrws(**_kwargs())
    before = transformer.decoder.mean_head.weight.clone()

    assert load_checkpoint_strict(transformer, foreign_blob) is None
    assert torch.equal(transformer.decoder.mean_head.weight, before)


def test_a_checkpoint_from_another_warm_up_budget_is_refused() -> None:
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
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnTrfCrws, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnTrfCrws")  # same class: the guard cannot help
    assert other_blob["model_kwargs"]["target_keep_index"] == _OTHER_KEEP_INDEX

    shipped_width_model = SeqVaeLagAttnTrfCrws(**_kwargs())
    assert shipped_width_model.target_adapter.linear.in_features == len(TINY_TARGET_KEEP_INDEX)
    # The decoders agree, so nothing about the forecast's shape says the budgets differ.
    assert shipped_width_model.decoder_out_channels == shipped_width_model.raw_per_step
    assert load_checkpoint_strict(shipped_width_model, other_blob) is None

    # And rebuilt from its own kwargs it loads, which is what makes the refusal above a statement
    # about the budget rather than about the blob being broken.
    rebuilt = SeqVaeLagAttnTrfCrws(**other_blob["model_kwargs"])
    assert rebuilt.target_adapter.linear.in_features == len(_OTHER_KEEP_INDEX)
    assert rebuilt.decoder_out_channels == rebuilt.raw_per_step
    assert load_checkpoint_strict(rebuilt, other_blob) is not None
