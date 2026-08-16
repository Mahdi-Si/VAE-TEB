r"""Checkpoints are self-describing, and this model's input set and tiling are two of the things
they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file -- the config that produced a run is a
mutable file that may not exist when anyone loads the weights -- and ``check_model_class`` can refuse
a foreign blob *before* the rebuild is attempted, while the error can still say what is wrong.

What this cell adds to that contract is not a decoder width. The decoder emits ``raw_per_step`` raw
samples per horizon token and no budget, gate or config can move it, which is exactly the difference
from the causal-feature cell it shares every input tensor with: **two blobs of these two models are
the same tensors under the same names, and only the decoder head's width tells them apart.** The
class stamp is what separates them, and the test below drives that case rather than describing it.

Two fields are load-bearing here in their own right.

``target_keep_index`` records *which stored input channels the encoders were fed*. It is resolved
from the **warm-up budget against the shards**, so a blob recording only the threshold could not be
rebuilt anywhere the data is not -- and the data may be on another machine.

``target_warmup_steps`` is what the input adapters mask and announce with. A blob that lost it would
rebuild a model with the right widths and **no availability terms at all**, reading the region where
a one-sided filter's output is a function of assumed pre-recording history -- with every tensor
aligning and every shape correct.
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


def test_the_base_stamp_survives_the_override(blob):
    """The task's ``on_save_checkpoint`` adds a field; it must not replace the base's work."""
    assert "model_class" in blob and "model_kwargs" in blob
    assert blob["epoch"] == 3  # and Lightning's own fields are untouched


def test_the_keep_index_records_the_inputs_and_the_decoder_width_is_not_a_budget_decision(blob):
    """Both halves of the one binding this cell does **not** make.

    ``target_keep_index`` is the record of which stored channels the encoders were fed, and the
    model cannot be rebuilt without it: the adapter's input width follows it, and the budget that
    produced it is a property of the shards.

    The decoder is the other half and its *independence* is the point: it emits ``raw_per_step`` raw
    samples per horizon token at every budget, so ``decoder_out_channels`` is a constant of the
    architecture rather than something the blob has to carry. A field recording it separately could
    disagree with ``raw_per_step`` and one of the two would be believed."""
    kwargs = blob["model_kwargs"]

    assert kwargs["target_keep_index"] == TINY_TARGET_KEEP_INDEX
    assert "decoder_out_channels" not in kwargs
    rebuilt = SeqVaeLagAttnCrws(**kwargs)
    assert rebuilt.target_adapter.linear.in_features == len(TINY_TARGET_KEEP_INDEX)
    assert rebuilt.decoder_out_channels == rebuilt.raw_per_step == RAW_PER_STEP


def test_the_warm_up_vectors_travel_and_the_delay_names_do_not(blob):
    """Two halves of one refusal. The warm-up vectors are what the adapters mask and announce with,
    so a blob without them rebuilds a model with the right widths and no availability terms at all.
    And they must travel under **these** names: ``target_delays`` reaches ``ChannelDelay``, which
    shifts rather than masks, so a blob carrying one under the other's name would be ambiguous
    between two families under one key."""
    kwargs = blob["model_kwargs"]

    assert kwargs["target_warmup_steps"] == TINY_TARGET_WARMUP_STEPS
    assert "source_warmup_steps" in kwargs
    assert "target_delays" not in kwargs and "source_delays" not in kwargs

    rebuilt = SeqVaeLagAttnCrws(**kwargs)
    assert rebuilt.target_adapter.availability is not None
    assert rebuilt.source_adapter.availability is not None


def test_the_tiling_travels_because_it_decides_what_a_row_of_the_csv_means(blob):
    """``anchor_stride`` is a real constructor argument rather than a translation, so it lands in
    the blob with everything else -- and it has to, because it decides how many anchors a training
    step decodes and therefore what ``anchors_per_sample``, ``nll_base_block`` and every per-anchor
    number were averaged over."""
    kwargs = blob["model_kwargs"]

    assert kwargs["anchor_stride"] == TINY_STRIDE
    assert kwargs["lag_floor"] == 0
    assert SeqVaeLagAttnCrws(**kwargs).anchor_stride == TINY_STRIDE


def test_the_flags_that_change_the_architecture_survive(blob):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success."""
    for flag in (
        "sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "max_lag", "c_y",
        "warmup_period", "target_keep_index", "target_warmup_steps", "anchor_stride", "lag_floor",
    ):
        assert flag in blob["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_loss_hyperparameters_and_the_run_seed_reach_the_checkpoint():
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config --
    and so is its **tiling**: the per-segment phase is a hash of the seed among other things, so a
    resumed run that did not know it would re-tile every segment from the epoch it resumed at."""
    module = _wrapped(SeqVaeLagAttnCrws, _kwargs())

    for name in (
        "likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule", "beta_prior",
        "seed",
    ):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"


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
def test_the_class_guard_accepts_this_model_and_rejects_the_siblings(blob):
    check_model_class(blob, "SeqVaeLagAttnCrws")  # must not raise

    for foreign in ("SeqVaeLagAttnRws", "SeqVaeLagAttnCfs"):
        with pytest.raises(ValueError, match="does not match the active model class") as excinfo:
            check_model_class(blob, foreign)
        assert "SeqVaeLagAttnCrws" in str(excinfo.value)


def test_a_causal_feature_blob_is_refused_by_the_class_check():
    """The nearest miss in the whole family, and the one a shared ``core_model_checkpoint`` key
    makes easy to attempt. That model reads the **identical three input tensors** at the identical
    widths under the identical tensor names, and differs in one thing: what its decoder emits. So
    the class stamp is the only thing that separates the two, which is why it is exercised rather
    than argued about."""
    foreign_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCfs, _kwargs()))

    assert foreign_blob["model_class"] == "SeqVaeLagAttnCfs"
    # Built from the identical keyword set, which is what makes "the same inputs" a fact here rather
    # than a claim about two configurations that happen to look alike.
    assert foreign_blob["model_kwargs"] == _kwargs()
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(foreign_blob, "SeqVaeLagAttnCrws")


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
