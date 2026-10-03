"""P4-07: the task's two own members, ``_build_forward_inputs`` and ``_added_metrics``.

Everything else is the CRWS task's and is pinned in its suite.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.tests.conftest import TASK_HPARAMS
from teb_vae.lag_attn_transformer_patch.nets.patching import patchify
from teb_vae.lag_attn_transformer_patch.task import SeqVaeLagAttnTrfPatchTask

from .conftest import TINY_KWARGS, build, make_stub_batch


def _task() -> SeqVaeLagAttnTrfPatchTask:
    task = SeqVaeLagAttnTrfPatchTask(
        build(), lr=1e-3, model_kwargs=dict(TINY_KWARGS), **TASK_HPARAMS
    )
    task.setup("fit")
    return task


def test_each_stage_gets_its_patch_streams_and_anchor_geometry():
    """Outside a step: FHR masked by ``weight``, UP by ``source_validity``, dense ``(0, 1)``. Train
    tiles at stride 15 (16 anchors) with a reproducible phase; val/test decode all 240 and report
    the source-null floor and the permutation control, which train does not."""
    task, batch = _task(), make_stub_batch(seq_len=300)

    def patches(raw, validity):
        return patchify(raw, batch.weight, raw_per_step=16, validity=validity)

    y, u, phase, stride = task._build_forward_inputs(batch)
    assert (phase, stride) == (0, 1)
    assert torch.equal(y, patches(batch.fhr, "fhr_weight"))
    assert torch.equal(u, patches(batch.up, "finite"))  # differs at the stub's gap step
    task.orig_model.source_validity = "fhr_weight"  # the ablation arm reaches the source stream
    assert torch.equal(task._build_forward_inputs(batch)[1], patches(batch.up, "fhr_weight"))
    task.orig_model.source_validity = "finite"

    task._stage = "train"
    _, _, phase, stride = task._build_forward_inputs(batch)
    assert stride == 15 and torch.equal(task._build_forward_inputs(batch)[2], phase)

    for stage, anchors in (("train", 16), ("val", 240), ("test", 240)):
        loss, metrics = task.compute_loss_and_metrics(batch, 0, stage)
        assert torch.isfinite(loss) and float(metrics["anchors_per_sample"]) == anchors, stage
        for name in ("kld_source_null", "nll_shuffled_block"):
            assert (name in metrics) == (stage != "train"), (stage, name)
            assert stage == "train" or torch.isfinite(metrics[name]), (stage, name)


def test_kld_source_null_is_the_control_over_a_valid_flat_null(perturb_posterior):
    """The val readout is ``controls.source_null_kld`` on the model's own forward and ``u_patch``,
    and its all-zero null is a valid, flat stream: moving the source ``missing`` embedding leaves
    it bitwise unchanged. Perturbed first, since at init every KL is 0."""
    task, batch = _task(), make_stub_batch(seq_len=300)
    model = task.orig_model
    perturb_posterior(model)
    _, metrics = task.compute_loss_and_metrics(batch, 0, "val")

    inputs = task._build_forward_inputs(batch)
    with torch.no_grad():
        fo = model(*inputs)
        floor = controls.source_null_kld(model, fo, inputs[1], batch.weight)
        assert floor > 0 and torch.equal(metrics["kld_source_null"], floor)
        # Random, not a constant shift, which the adapter's norm would cancel.
        model.source_adapter.missing.add_(torch.randn(model.source_adapter.missing.shape))
        assert torch.equal(controls.source_null_kld(model, fo, inputs[1], batch.weight), floor)
