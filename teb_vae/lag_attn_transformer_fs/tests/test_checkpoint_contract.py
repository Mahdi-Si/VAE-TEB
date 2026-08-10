r"""Checkpoints are self-describing, and for this model the *width* is one of the things they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file -- the config that produced a run is a
mutable file that may not exist when anyone loads the weights -- and ``check_model_class`` can refuse
a foreign blob *before* the rebuild is attempted, while the error can still say what is wrong.

This model raises the stakes on the second field twice over.

**The decoder width is recorded nowhere directly.** It is $C_{\mathrm{keep}}$, resolved from the
reach budget, and ``decoder_out_channels`` is not a keyword of this constructor at all -- so no
second field can disagree with the gate, and the **only** thing that makes a checkpoint rebuildable
is ``target_keep_index``. A checkpoint that lost it would rebuild a model whose decoder emits $c_y$
values per token instead of $78$ and refuse its own weights.

**Three foreign models can write a blob that partly aligns.** This architecture shares the encoders
with ``SeqVaeLagAttnTrfRws`` and the decoder width with ``SeqVaeLagAttnFs``, and everything below the
encoders with both plus ``SeqVaeLagAttnRws``. So the class guard buys the **message**, not the
refusal: ``load_checkpoint_strict`` evaluates alignment before loading anything and writes no weight
either way. The feature sibling is where the message matters most, since the two differ only in the
encoder tensors.

The machinery under test is the shared task's, which is where checkpointing lives; this package's own
task overrides nothing about it, so ``model_class`` is stamped from ``type(orig_model).__name__`` and
the contract is a property of the model class and its kwargs rather than of the wrapper.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import TINY_KWARGS as CONV_LSTM_TINY_KWARGS
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

#: A second reach budget, so a cross-budget load can be exercised. Two channels rather than three:
#: the decoder width follows the keep-index's *length*, and it must not accidentally match.
_OTHER_KEEP_INDEX = (1, 4)
_OTHER_DELAYS = (0, 1)

#: The seven keys this architecture adds. Each changes a parameter count, so a blob missing one
#: cannot be rebuilt at all -- and a blob that recorded none of them would describe an architecture
#: rather than *this* architecture.
_ENCODER_KEYS = (
    "encoder_conv_kernels",
    "encoder_conv_dilations",
    "encoder_num_heads",
    "encoder_d_ff",
    "target_attention_blocks",
    "source_attention_blocks",
    "source_attention_window",
)


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
    return _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnTrfFs, tiny_gated_kwargs()))


# ---------------------------------------------------------------------------------------
# What the blob carries
# ---------------------------------------------------------------------------------------
def test_the_checkpoint_names_this_model_class(blob):
    """Stamped from the eager model, so it says ``SeqVaeLagAttnTrfFs`` even though the wrapper is the
    shared task."""
    assert blob["model_class"] == "SeqVaeLagAttnTrfFs"
    assert blob["model_kwargs"] == tiny_gated_kwargs()


def test_the_base_stamp_survives_the_override(blob):
    """``on_save_checkpoint`` adds a field; it must not replace the base's work. An override that
    skipped ``super()`` would drop ``model_class`` and every guard reading it would silently degrade
    to its warn-and-continue path."""
    assert "model_class" in blob and "model_kwargs" in blob
    assert blob["epoch"] == 3  # and Lightning's own fields are untouched


def test_the_keep_index_is_the_only_record_of_the_decoder_width(blob):
    """The field this model cannot be rebuilt without.

    ``decoder_out_channels`` is not a keyword of this constructor at all -- unlike the conv-LSTM
    feature sibling, where it exists and is deliberately left unset -- so there is no second field
    that *could* record the width and disagree with the gate.
    """
    kwargs = blob["model_kwargs"]

    assert kwargs["target_keep_index"] == TINY_KEEP_INDEX
    assert "decoder_out_channels" not in kwargs
    rebuilt = SeqVaeLagAttnTrfFs(**kwargs)
    assert rebuilt.decoder_out_channels == len(TINY_KEEP_INDEX) == 3


def test_raw_per_step_is_present_because_it_is_still_geometry(blob):
    """It stopped being the decoder width; it did not stop being a geometry input. The trimmed grid
    validates its raw index identities against it, so a blob without it rebuilds a model whose
    geometry cannot be constructed at all."""
    kwargs = blob["model_kwargs"]

    assert kwargs["raw_per_step"] == 16
    rebuilt = SeqVaeLagAttnTrfFs(**kwargs)
    assert rebuilt.geometry.r == 16
    assert rebuilt.decoder_out_channels != rebuilt.raw_per_step


def test_the_seven_encoder_keys_are_stamped(blob):
    """So a checkpoint records the architecture that wrote it. Each changes a parameter count, so a
    blob missing one cannot be rebuilt -- and a rebuild that ignored the recorded kwargs would load a
    partly random model and report nothing."""
    for key in _ENCODER_KEYS:
        assert key in blob["model_kwargs"], f"{key} missing from model_kwargs"

    shallower = SeqVaeLagAttnTrfFs(**dict(blob["model_kwargs"], target_attention_blocks=1))
    reference = SeqVaeLagAttnTrfFs(**blob["model_kwargs"])
    assert set(shallower.state_dict()) != set(reference.state_dict())


def test_the_flags_that_change_the_architecture_survive(blob):
    """A missing flag rebuilds a different model, and ``load_checkpoint_strict`` would then align
    nothing and return ``None`` -- which a caller that does not check reads as success."""
    for flag in ("sequence_length", "d_model", "d_z", "horizon", "raw_per_step", "max_lag",
                 "c_y", "target_keep_index", "target_delays", "source_keep_index", "source_delays"):
        assert flag in blob["model_kwargs"], f"{flag} missing from model_kwargs"


def test_the_loss_hyperparameters_reach_the_checkpoint():
    """So a run's objective is recoverable from its checkpoint, not only from a mutable config."""
    module = _wrapped(SeqVaeLagAttnTrfFs, tiny_gated_kwargs())

    for name in ("likelihood", "free_bits", "lambda_full", "lambda_base", "beta_schedule",
                 "beta_prior"):
        assert name in module.hparams, f"{name} is not in hparams and will not be checkpointed"


# ---------------------------------------------------------------------------------------
# The round trip
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_round_trips_into_a_fresh_model(inputs, tmp_path):
    """The whole contract end to end: save, reload, rebuild from the blob alone, same forward.

    Bitwise rather than within a tolerance: the same computation on the same weights on the same
    device, seeded before each forward because the model samples $z$.
    """
    module = _wrapped(SeqVaeLagAttnTrfFs, tiny_gated_kwargs())
    path = tmp_path / "model.ckpt"
    torch.save(_lightning_style_checkpoint(module), path)

    saved = torch.load(path, map_location="cpu", weights_only=False)
    check_model_class(saved, "SeqVaeLagAttnTrfFs")
    rebuilt = SeqVaeLagAttnTrfFs(**saved["model_kwargs"])
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

    Each is built from *its own* suite's keyword set, because the three constructors' schemas differ
    -- the conv-Transformer pair take the seven encoder keys and the conv-LSTM pair take five others.
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
    "name, stamped, what_aligns",
    [
        ("trf_rws", "SeqVaeLagAttnTrfRws", "the same encoders at the wrong decoder width"),
        ("fs", "SeqVaeLagAttnFs", "the same decoder width behind the wrong encoders"),
        ("rws", "SeqVaeLagAttnRws", "neither"),
    ],
)
def test_the_class_guard_fires_first_and_names_the_model_that_wrote_the_blob(
    name, stamped, what_aligns
):
    """The guard's whole product is the message. Each of these blobs aligns in part, so a loader that
    trusted a non-``None`` return would warm-start from a mixture of loaded and random weights and
    report success -- and the failure would name misaligned keys rather than naming the model."""
    blob = _foreign_blob(name)

    assert blob["model_class"] == stamped, what_aligns
    with pytest.raises(ValueError, match="does not match the active model class") as excinfo:
        check_model_class(blob, "SeqVaeLagAttnTrfFs")
    assert stamped in str(excinfo.value)


@pytest.mark.parametrize("name", ["trf_rws", "fs", "rws"])
def test_with_the_guard_skipped_the_load_still_writes_no_weight(name):
    """The all-or-nothing property the class guard is layered on top of, pinned so a change to the
    loader cannot quietly turn a refusal into a partial warm start.

    ``load_checkpoint_strict`` evaluates a candidate module's alignment *before* loading anything and
    skips it on any missing key, unexpected key or shape mismatch. So it returns ``None`` and the
    decoder's mean head is bitwise what it was -- and the driver raises on ``None`` rather than
    training a model it thought it had warm-started.
    """
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
    *and the same shape*, including the decoder's two widened heads, so the misalignment a bare loader
    would report is a list of encoder keys rather than a statement about the model."""
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
        assert name in shared and mine[name].shape[0] == 3
    # And what does not align is encoders on both sides, which is the whole difference.
    assert all(
        "encoder" in key for key in set(mine) - set(stripped)
    ), sorted(set(mine) - set(stripped))[:5]


# ---------------------------------------------------------------------------------------
# A cross-budget blob, which the class check cannot help with at all
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_from_another_reach_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their nats are not comparable, their decoders are different
    widths, and their checkpoints are mutually unloadable. The refusal comes from the width the
    stamped keep-index implies, which is exactly why that field has to travel.

    The keep-index lengths are chosen to differ ($2$ against $3$), so the widths cannot coincide by
    accident and the refusal is about the budget rather than about two arms that happened to match.
    """
    other_kwargs = dict(
        tiny_gated_kwargs(), target_keep_index=_OTHER_KEEP_INDEX, target_delays=_OTHER_DELAYS
    )
    other_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnTrfFs, other_kwargs))

    check_model_class(other_blob, "SeqVaeLagAttnTrfFs")  # same class: the guard cannot help
    assert other_blob["model_kwargs"]["target_keep_index"] == _OTHER_KEEP_INDEX
    assert len(_OTHER_KEEP_INDEX) != len(TINY_KEEP_INDEX)

    shipped_width_model = SeqVaeLagAttnTrfFs(**tiny_gated_kwargs())
    assert shipped_width_model.decoder_out_channels == len(TINY_KEEP_INDEX)
    assert load_checkpoint_strict(shipped_width_model, other_blob) is None

    # The positive control: rebuilt from its own kwargs the same blob loads, which is what makes the
    # refusal a statement about the budget rather than about a broken blob.
    rebuilt = SeqVaeLagAttnTrfFs(**other_blob["model_kwargs"])
    assert rebuilt.decoder_out_channels == len(_OTHER_KEEP_INDEX) == 2
    assert load_checkpoint_strict(rebuilt, other_blob) is not None


def test_the_four_models_stamp_four_distinct_class_names():
    """A guard on the premise of every refusal above: the stamp is ``type(orig_model).__name__``, so
    two models sharing a class name would make the guard silently useless."""
    stamps = {
        _lightning_style_checkpoint(_wrapped(cls, kwargs))["model_class"]
        for cls, kwargs in (
            (SeqVaeLagAttnTrfFs, tiny_gated_kwargs()),
            (SeqVaeLagAttnTrfRws, tiny_gated_kwargs()),
            (SeqVaeLagAttnFs, conv_lstm_tiny_gated_kwargs()),
            (SeqVaeLagAttnRws, CONV_LSTM_TINY_KWARGS),
        )
    }

    assert stamps == {
        "SeqVaeLagAttnTrfFs",
        "SeqVaeLagAttnTrfRws",
        "SeqVaeLagAttnFs",
        "SeqVaeLagAttnRws",
    }
