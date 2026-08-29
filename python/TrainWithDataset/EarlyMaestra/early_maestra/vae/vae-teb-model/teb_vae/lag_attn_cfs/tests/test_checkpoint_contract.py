r"""Checkpoints are self-describing, and this model's width and tiling are two of the things they
describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file -- the config that produced a run is a
mutable file that may not exist when anyone loads the weights -- and ``check_model_class`` can refuse
a foreign blob *before* the rebuild is attempted, while the error can still say what is wrong.

This model raises the stakes twice over.

Its decoder width is not a configuration key: it is $C_{\mathrm{keep}}$, resolved from the **warm-up
budget against the shards**. Nothing writes that number into the blob, so the only thing that makes a
checkpoint rebuildable is ``target_keep_index`` -- and unlike the two-sided cell, where the same is
true, the resolution here needs the *data*: a threshold-only record could not be re-resolved without
the shards that produced the run, which may be on another machine.

And its warm-up vectors are what the input adapters mask and announce with. A blob that lost
``target_warmup_steps`` would rebuild a model with the right widths and **no availability terms at
all**, reading the region where a one-sided filter's output is a function of assumed pre-recording
history -- with every tensor aligning and every shape correct.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.model import SeqVaeLagAttnCfs
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
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
#: the decoder width follows and cannot accidentally match.
_OTHER_KEEP_INDEX = TINY_TARGET_KEEP_INDEX[:20]
_OTHER_WARMUP_STEPS = TINY_TARGET_WARMUP_STEPS[:20]


def _kwargs(**overrides) -> dict:
    """The guarded tiny keyword set, carrying both keys the shipped config states explicitly.

    ``lag_floor`` is written out at its own default rather than left off, because a production blob
    carries it: the config declares it and the driver's signature sweep forwards every declared key.
    A fixture that omitted it would make the assertion below a statement about this helper rather
    than about what a run stamps.
    """
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, lag_floor=0, **overrides)


def _wrapped(cls, kwargs, task_cls=SeqVaeLagAttnCfsTask):
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
    return _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCfs, _kwargs()))


@pytest.fixture
def blob_with_switches():
    """A checkpoint written at the shipped switches rather than at their off-states.

    A second fixture rather than a change to the first: the tests above are about what a checkpoint
    carries at *any* configuration, and moving the base arm under them would make each of them a
    test of this arm instead.
    """
    return _lightning_style_checkpoint(
        _wrapped(SeqVaeLagAttnCfs, _kwargs(**_SHIPPED_SWITCHES))
    )


#: The architecture switches this cell ships, at the values its config carries.
_SHIPPED_SWITCHES = dict(
    lag_kv_source="conv_stem",
    prior_availability_input=True,
    persistence_residual=True,
    horizon_weight_halflife_steps=5.0,
    alibi_slope_scale=0.0,
)


# ---------------------------------------------------------------------------------------
# What the blob carries
# ---------------------------------------------------------------------------------------
def test_the_checkpoint_names_this_model_class(blob):
    """Stamped from the eager model, so it says ``SeqVaeLagAttnCfs`` even though the wrapper adds
    only a phase and a stride to the shared task."""
    assert blob["model_class"] == "SeqVaeLagAttnCfs"
    assert blob["model_kwargs"] == _kwargs()


def test_the_base_stamp_survives_the_override(blob):
    """The task's ``on_save_checkpoint`` adds a field; it must not replace the base's work."""
    assert "model_class" in blob and "model_kwargs" in blob
    assert blob["epoch"] == 3  # and Lightning's own fields are untouched


def test_the_keep_index_is_the_only_record_of_the_decoder_width(blob):
    """The field this model cannot be rebuilt without. ``decoder_out_channels`` is deliberately
    absent: the width is derived from the gate, so a second field recording it could disagree with
    the gate and one of the two would be believed."""
    kwargs = blob["model_kwargs"]

    assert kwargs["target_keep_index"] == TINY_TARGET_KEEP_INDEX
    assert "decoder_out_channels" not in kwargs
    rebuilt = SeqVaeLagAttnCfs(**kwargs)
    assert rebuilt.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)


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

    rebuilt = SeqVaeLagAttnCfs(**kwargs)
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
    assert SeqVaeLagAttnCfs(**kwargs).anchor_stride == TINY_STRIDE


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
    module = _wrapped(SeqVaeLagAttnCfs, _kwargs())

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
    module = _wrapped(SeqVaeLagAttnCfs, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(saved, "SeqVaeLagAttnCfs")
    rebuilt = SeqVaeLagAttnCfs(**saved["model_kwargs"])
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
    than its siblings: the budget is resolved **against the shards**, so a blob recording only the
    threshold could not be rebuilt anywhere the data is not.

    Driven from a directory containing no HDF5 at all, with the working directory moved there, so a
    rebuild that reached for a shard by a relative path would fail rather than quietly find one."""
    module = _wrapped(SeqVaeLagAttnCfs, _kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = SeqVaeLagAttnCfs(**saved["model_kwargs"])

    assert not list(Path(empty).glob("*.hdf5"))
    assert load_checkpoint_strict(rebuilt, saved) is not None
    assert rebuilt.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)
    assert rebuilt.target_warmup_steps == TINY_TARGET_WARMUP_STEPS


# ---------------------------------------------------------------------------------------
# Foreign blobs
# ---------------------------------------------------------------------------------------
def test_the_class_guard_accepts_this_model_and_rejects_the_siblings(blob):
    check_model_class(blob, "SeqVaeLagAttnCfs")  # must not raise

    for foreign in ("SeqVaeLagAttnRws", "SeqVaeLagAttnFs"):
        with pytest.raises(ValueError, match="does not match the active model class") as excinfo:
            check_model_class(blob, foreign)
        assert "SeqVaeLagAttnCfs" in str(excinfo.value)


def test_a_two_sided_feature_model_blob_is_refused_by_the_class_check():
    """The load a shared ``core_model_checkpoint`` key makes easy to attempt. The two models share
    every tensor name; only the widths differ, and only because the two budgets keep different
    channel counts -- so the *interesting* half is what happens when the guard is skipped."""
    from teb_vae.lag_attn_fs.tests.conftest import tiny_gated_kwargs

    foreign_blob = _lightning_style_checkpoint(
        _wrapped(SeqVaeLagAttnFs, tiny_gated_kwargs(), task_cls=SeqVaeLagAttnCfsTask)
    )

    assert foreign_blob["model_class"] == "SeqVaeLagAttnFs"
    with pytest.raises(ValueError, match="does not match the active model class"):
        check_model_class(foreign_blob, "SeqVaeLagAttnCfs")


def test_a_foreign_blob_with_the_guard_skipped_refuses_rather_than_partly_succeeding():
    """The all-or-nothing property the class guard is layered on top of, pinned so a change to the
    loader cannot quietly turn a refusal into a partial warm start.

    ``load_checkpoint_strict`` evaluates a candidate module's alignment *before* loading anything
    and skips it on any missing key, unexpected key or shape mismatch. What the class guard buys is
    therefore the **message**: without it the failure names misaligned keys instead of naming the
    model that wrote the blob."""
    from teb_vae.lag_attn_fs.tests.conftest import tiny_gated_kwargs

    foreign_blob = _lightning_style_checkpoint(
        _wrapped(SeqVaeLagAttnFs, tiny_gated_kwargs(), task_cls=SeqVaeLagAttnCfsTask)
    )
    causal_model = SeqVaeLagAttnCfs(**_kwargs())
    before = causal_model.decoder.mean_head.weight.clone()

    assert load_checkpoint_strict(causal_model, foreign_blob) is None
    assert torch.equal(causal_model.decoder.mean_head.weight, before)


def test_a_checkpoint_from_another_warm_up_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their nats are not comparable, their decoders are different
    widths, and their checkpoints are mutually unloadable. The refusal comes from the width the
    stamped keep-index implies, which is exactly why that field has to travel."""
    other_kwargs = _kwargs(
        target_keep_index=_OTHER_KEEP_INDEX, target_warmup_steps=_OTHER_WARMUP_STEPS
    )
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnCfs, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnCfs")  # same class: the guard cannot help
    assert other_blob["model_kwargs"]["target_keep_index"] == _OTHER_KEEP_INDEX

    shipped_width_model = SeqVaeLagAttnCfs(**_kwargs())
    assert shipped_width_model.decoder_out_channels == len(TINY_TARGET_KEEP_INDEX)
    assert load_checkpoint_strict(shipped_width_model, other_blob) is None

    # And rebuilt from its own kwargs it loads, which is what makes the refusal above a statement
    # about the budget rather than about the blob being broken.
    rebuilt = SeqVaeLagAttnCfs(**other_blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == len(_OTHER_KEEP_INDEX)
    assert load_checkpoint_strict(rebuilt, other_blob) is not None


def test_the_switches_the_revision_added_survive_and_carry_their_values(blob_with_switches):
    """The same rule as above for the six architecture switches, and one they need more.

    A missing width shows up as a shape mismatch on load. A missing **switch** does not: the
    constructor has a default for every one of them, so a checkpoint that dropped one would rebuild
    at the OFF value -- a different architecture, loading cleanly, reporting numbers under the arm's
    name. The evaluation reconciles three of them against the config for exactly this reason, and
    that reconciliation can only work if they are on the blob.

    The values are asserted, not only the keys: a stamp carrying ``lag_kv_source`` at whatever the
    default is would satisfy a presence check and still describe the wrong model.
    """
    stamped = blob_with_switches["model_kwargs"]

    for flag, value in _SHIPPED_SWITCHES.items():
        assert flag in stamped, f"{flag} missing from model_kwargs"
        assert stamped[flag] == value, flag


def test_the_horizon_weight_is_absent_from_the_state_dict_and_present_in_the_kwargs(
    blob_with_switches,
):
    r"""The one switch whose tensor deliberately does not travel, and why both halves matter.

    ``horizon_weight_halflife_steps`` is a **number** the constructor turns into an $(H,)$ buffer,
    and the buffer is non-persistent. So the half-life is recoverable from the checkpoint -- an
    arm's objective is part of what it was -- while the tensor is not, which is what keeps a
    checkpoint loadable at another horizon. A persistent buffer would put $H$ into the state dict
    and make every cross-horizon reload a shape error for a tensor that is a pure function of two
    numbers already on the blob.
    """
    assert "horizon_weight_halflife_steps" in blob_with_switches["model_kwargs"]
    assert not any("horizon_weight" in key for key in blob_with_switches["state_dict"])
