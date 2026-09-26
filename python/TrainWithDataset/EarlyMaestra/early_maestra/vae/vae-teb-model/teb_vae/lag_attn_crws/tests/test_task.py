r"""The task is the raw-signal sibling's, plus the tiling seam bound from the causal-feature task.

Everything about turning a batch into a loss is inherited from ``SeqVaeLagAttnRwsTask`` -- the raw
target included -- and the tiling members (the phase, the anchor geometry, the five-tuple input
builder, the two re-pointed readouts) are bound by reference from ``SeqVaeLagAttnCfsTask``, whose
own suite pins them. What this cell writes itself is ``compute_loss_and_metrics``, which makes the
step's stage reachable from the bound input builder, so what is checked here is:

* the forward inputs are the five-tuple at this model's widths and the target is the raw signal,
  not a feature block -- which is what the wrong base class would silently change;
* the stage reaches the input builder, so training decodes a tile and validation every valid anchor;
* the stage is put back even when the step raises;
* the ungated arm runs through the same task with the decoder still $R$ samples wide.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_crws.task import DENSE_STAGES

from .conftest import CAUSAL_PH_WIDTH, CAUSAL_ST_WIDTH, TINY_STRIDE


def test_the_forward_inputs_are_five_and_the_target_is_the_raw_signal(task, stub_batch):
    """Everything downstream reads ``inputs[0]`` for the batch size and the device rather than a
    named tensor, so the arity change reaches nothing else. The raw target is NOT among the inputs:
    it stays behind the inherited builder, and it is the raw signal itself."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)

    assert len(inputs) == 5
    assert inputs[0].shape[-1] == CAUSAL_ST_WIDTH
    assert inputs[1].shape[-1] == CAUSAL_PH_WIDTH
    assert torch.equal(inputs[2][..., :CAUSAL_ST_WIDTH], stub_batch.up_st)
    target, _weight = module._build_raw_target(stub_batch)
    assert torch.equal(target, stub_batch.fhr)


def test_the_decoded_anchor_count_follows_the_resolved_stage(task, stub_batch):
    """The property the whole five-tuple exists for, read off the forward rather than off the
    arguments: training decodes a tile and validation decodes every valid anchor."""
    module = task()
    model = module.orig_model
    dense = model.geometry.t_valid - model.warmup_period

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    tiles = -(-dense // TINY_STRIDE)
    assert tiles - 1 <= float(train_metrics["anchors_per_sample"]) <= tiles
    assert float(val_metrics["anchors_per_sample"]) == pytest.approx(float(dense))
    assert module._stage in DENSE_STAGES


def test_the_stage_is_restored_even_when_the_step_raises(task, stub_batch, monkeypatch):
    """The ``finally`` rather than a trailing assignment. A step that raises would otherwise leave
    ``_stage`` on ``'train'``, and the diagnostic callback's next out-of-step call would draw a figure
    at a tile grid that depends on the epoch."""
    module = task()

    def _explode(self, batch):
        raise RuntimeError("planted")

    monkeypatch.setattr(type(module), "_build_raw_target", _explode)

    with pytest.raises(RuntimeError, match="planted"):
        module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert module._stage in DENSE_STAGES


def test_the_ungated_model_still_runs_through_this_task(task, stub_batch, tiny_kwargs):
    """No budget means no gate, no warm-up mask and no dropped channel -- and the task must not
    assume otherwise, because that arm is what every "the guard did something" comparison is made
    against. The decoder's width does not move with it: it is ``raw_per_step`` either way."""
    module = task(model_kwargs=dict(tiny_kwargs, anchor_stride=TINY_STRIDE))

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert torch.isfinite(loss)
    assert module.orig_model.decoder_out_channels == int(tiny_kwargs["raw_per_step"])
    assert torch.isfinite(metrics["anchors_per_sample"])
