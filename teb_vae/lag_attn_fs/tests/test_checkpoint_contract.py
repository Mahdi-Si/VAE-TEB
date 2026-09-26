r"""Checkpoints are self-describing, and this model's width is one of the things they describe.

A stock Lightning ``.ckpt`` carries neither the class that wrote it nor the kwargs that built it.
With both, a checkpoint can be rebuilt with no config file, and ``check_model_class`` can refuse a
foreign blob before the rebuild is attempted.

This model raises the stakes on the kwargs. Its decoder width is not a configuration key: it is
$C_{\mathrm{keep}}$, resolved from the reach budget, and the only thing that carries it into the
blob is ``target_keep_index``. What is checked here is the round trip, the refusal of the raw
sibling's blob (whose tensors match this model's in name everywhere but the decoder's two output
heads), and the refusal of a blob from another reach budget of this same model.

The machinery under test is the sibling's, reached through the sibling's task: this model's own
task overrides how its target is built and nothing about checkpointing.
"""
from __future__ import annotations

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


def test_with_the_guard_skipped_the_load_still_refuses_rather_than_partly_succeeding():
    """The all-or-nothing property the class guard is layered on top of.

    ``load_checkpoint_strict`` evaluates a candidate module's alignment *before* loading anything
    and skips it on any missing key, unexpected key or shape mismatch. The decoder-head tensors of
    the raw sibling mismatch, so it returns ``None`` and no weight is written -- and the driver
    raises on ``None`` rather than training a model it thought it had warm-started.
    """
    raw_blob = _lightning_style_checkpoint(_wrapped(SeqVaeLagAttnRws, tiny_gated_kwargs()))
    feature_model = SeqVaeLagAttnFs(**tiny_gated_kwargs())
    before = feature_model.decoder.mean_head.weight.clone()

    assert load_checkpoint_strict(feature_model, raw_blob) is None
    assert torch.equal(feature_model.decoder.mean_head.weight, before)


def test_a_checkpoint_from_another_reach_budget_is_refused():
    """Two arms of *this* model at different budgets stamp the same ``model_class``, so the class
    guard cannot separate them -- and their decoders are different widths. The refusal comes from
    the width the stamped keep-index implies, which is exactly why that field has to travel."""
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
