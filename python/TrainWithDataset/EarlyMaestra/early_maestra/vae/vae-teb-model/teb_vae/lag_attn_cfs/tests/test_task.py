r"""The task is the feature-target sibling's, plus the tiling phase and the source-null arm.

Two models are only comparable if they optimise the same thing, and here that is not a claim about
two copies of an objective agreeing -- it is the same code: the loss assembly, the $\beta$ schedule,
the metric surface, the permutation control, the spike-breaker wiring, the gradient-norm logging,
the checkpoint contract and the concatenated feature target are all inherited unmodified. One
absence is asserted directly, because nothing else would notice it: a re-added ``training_step``
silently disables the config-gated loss-spike breaker.

What is added is everything that follows from the anchor tiling, and each addition guards a failure
whose symptom is a number rather than an exception:

* ``_build_forward_inputs`` returns **five** tensors. Resolving the stride from ``self.training``
  instead would make ``total_loss`` a function of the dropout switch, because the diagnostic
  callback calls ``eval()`` *during* training and then the objective.
* the phase is a ``blake2b`` hash and not Python's ``hash()``, whose ``str`` salt is per process.
  That failure is invisible to any same-process test, so the test below crosses a process boundary.
* ``_mu_gap_rms`` is re-pointed at the decoded anchors, restoring a property its own inherited
  docstring already promises.
* ``kld_source_null`` is validation-only and is the one readout the permutation control cannot
  stand in for.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn_cfs.task import DENSE_STAGES, SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

from .conftest import (
    BATCH,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    TINY_STRIDE,
    make_stub_batch,
    tiny_warmup_kwargs,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["training_step", "validation_step", "test_step"])
def test_the_framework_steps_are_not_taken_back(method):
    """The framework's steps run the config-gated spike breaker and the stage bookkeeping, and a
    subclass defining its own silently disables them -- with every metric this file checks still
    produced through ``compute_loss_and_metrics``."""
    assert method not in vars(SeqVaeLagAttnCfsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_seed_is_a_hyperparameter_and_therefore_survives_a_resume(task):
    """It reaches no task in the family today; the tile phase is derived from it, so a resumed run
    that did not know it would silently re-tile every segment. ``save_hyperparameters`` is what puts
    it in the checkpoint."""
    assert task().hparams["seed"] == 0
    assert task(seed=7).hparams["seed"] == 7


# ---------------------------------------------------------------------------------------
# The five-tuple, and the stage it is resolved from
# ---------------------------------------------------------------------------------------
def test_the_forward_inputs_are_five_and_the_first_three_are_the_siblings(task, stub_batch):
    """Everything downstream reads ``inputs[0]`` for the batch size and the device rather than a
    named tensor, so the arity change reaches nothing else."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)
    target, _weight = module._build_raw_target(stub_batch)

    assert len(inputs) == 5
    assert torch.equal(torch.cat([inputs[0], inputs[1]], dim=-1), target)
    assert inputs[0].shape[-1] == CAUSAL_ST_WIDTH
    assert inputs[1].shape[-1] == CAUSAL_PH_WIDTH
    assert torch.equal(inputs[2][..., :CAUSAL_ST_WIDTH], stub_batch.up_st)


@pytest.mark.parametrize("stage", ["val", "test"])
def test_both_evaluation_stages_decode_the_dense_range_at_phase_zero(task, stub_batch, stage):
    """A single phase is deterministic but PHASE-BIASED: it would sample the same tile set at a
    fixed offset from the segment start forever, so any structure varying with position in the
    segment would never be seen at another offset. There is no gradient at either stage, so neither
    the redundancy argument nor the memory argument that motivates the tiling applies."""
    module = task()

    phase, stride = module.resolve_anchor_geometry(stage, stub_batch)

    assert (phase, stride) == (0, 1)
    assert stage in DENSE_STAGES


def test_training_tiles_at_the_models_own_stride_with_a_per_sample_phase(task, stub_batch):
    module = task()

    phase, stride = module.resolve_anchor_geometry("train", stub_batch)

    assert stride == module.orig_model.anchor_stride == TINY_STRIDE
    assert isinstance(phase, torch.Tensor)
    assert phase.shape == (BATCH,)
    assert bool(((phase >= 0) & (phase < stride)).all())


def test_the_stage_reaches_the_input_builder_and_is_put_back(task, stub_batch):
    """The attribute lives for exactly the length of one step. Its default is a dense stage, so the
    diagnostic callback -- which calls ``_build_forward_inputs`` outside any step -- draws the dense,
    epoch-independent anchor set rather than a training tile grid."""
    module = task()
    seen = []
    original = module.resolve_anchor_geometry
    module.resolve_anchor_geometry = lambda stage, batch: (
        seen.append(stage) or original(stage, batch)
    )

    module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert seen == ["train"]
    assert module._stage in DENSE_STAGES
    assert module._build_forward_inputs(stub_batch)[4] == 1


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


def test_the_geometry_is_not_a_function_of_the_dropout_switch(task, stub_batch):
    """The rejected alternative, asserted rather than argued. The diagnostic callback calls
    ``eval()`` *during* training and then the objective, so a stride read off ``self.training``
    would compute ``total_loss``, ``nll_base_block`` and ``anchors_per_sample`` at a different
    anchor set than the objective saw."""
    module = task()

    module.train()
    train_mode = module.resolve_anchor_geometry("val", stub_batch)
    module.eval()
    eval_mode = module.resolve_anchor_geometry("val", stub_batch)

    assert train_mode == eval_mode == (0, 1)


# ---------------------------------------------------------------------------------------
# The phase
# ---------------------------------------------------------------------------------------
def test_the_phase_is_a_function_of_the_segment_and_not_only_of_the_recording(task):
    """``guid`` identifies the RECORDING, not the segment, and an unshuffled loader over
    per-recording shards puts consecutive segments of one recording in one batch. Keyed on the GUID
    alone every segment of a recording would share a tile grid within an epoch, leaving in place
    exactly the within-batch gradient correlation the tiling exists to break."""
    module = task()
    batch = make_stub_batch(8)
    batch.guid = ["ONE_RECORDING"] * 8

    phases = module.anchor_phase(batch)

    assert phases.shape == (8,)
    assert len(set(phases.tolist())) > 1, (
        "every segment of one recording got the same tile grid; the phase is keyed on the GUID "
        "alone"
    )


def test_the_grid_rotates_with_the_epoch_and_covers_the_stride(task):
    """The claim that every anchor in $[F, T_\\mathrm{valid})$ is eventually decoded rests on this:
    a phase that did not move with the epoch would leave the same anchors undecoded forever."""
    module = task()
    batch = make_stub_batch(BATCH)

    seen = set()
    for epoch in range(64):
        module.__class__.current_epoch = property(lambda self, _e=epoch: _e)
        seen.update(module.anchor_phase(batch).tolist())
    del module.__class__.current_epoch

    assert seen == set(range(TINY_STRIDE))


def test_the_seed_moves_the_phase(task):
    """It is one of the four halves of the key, so two runs of one config at different seeds tile
    differently -- which is what makes the tiling a draw over epochs rather than a fixed grid."""
    batch = make_stub_batch(16)

    default = task(seed=0).anchor_phase(batch)
    reseeded = task(seed=99).anchor_phase(batch)

    assert not torch.equal(default, reseeded)


def test_the_forward_draws_no_random_number(task, stub_batch):
    """The reason the phase is hashed rather than sampled. A draw inside the step would consume the
    global RNG stream, move the reparameterisation $\\epsilon$ for every subsequent step and break
    every bitwise comparison in the suite -- and would not survive a checkpoint resume."""
    module = task()
    torch.manual_seed(0)
    before = torch.random.get_rng_state()

    module._build_forward_inputs(stub_batch)
    module.anchor_phase(stub_batch)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_phase_is_stable_across_processes_with_different_hash_seeds(tmp_path):
    """The failure a same-process test cannot see. Python's ``hash()`` on a ``str`` is salted per
    process by ``PYTHONHASHSEED``, which is random by default, so a phase derived from it is stable
    neither across DDP ranks nor across a resume -- and nothing raises, because $A_{\\max}$ is a
    geometry constant either way. So this crosses a process boundary with two different salts.
    """
    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import torch
        from teb_vae.lag_attn_cfs.tests.conftest import make_stub_batch, make_task
        task = make_task(seed=42)
        print(",".join(str(value) for value in task.anchor_phase(make_stub_batch(8)).tolist()))
        """
    )
    path = tmp_path / "phase.py"
    path.write_text(script, encoding="utf-8")

    outputs = []
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, str(path), str(_REPO_ROOT)],
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "PYTHONHASHSEED": hash_seed},
            cwd=str(_REPO_ROOT),
            check=True,
        )
        outputs.append(result.stdout.strip().splitlines()[-1])

    assert len(set(outputs)) == 1, f"the phase moved with PYTHONHASHSEED: {outputs}"


@pytest.mark.parametrize("field", ["guid", "epoch"])
def test_a_missing_phase_key_field_names_the_config_list_that_carries_it(task, stub_batch, field):
    """``load_fields`` is honoured literally with no forced additions, so dropping either key leaves
    every segment on one tile grid forever with no shape, no count and no metric differing."""
    module = task()
    delattr(stub_batch, field)

    with pytest.raises(RuntimeError, match=r"load_fields"):
        module.anchor_phase(stub_batch)


# ---------------------------------------------------------------------------------------
# The two re-pointed readouts
# ---------------------------------------------------------------------------------------
def test_the_latent_gap_uses_the_same_anchor_support_as_the_kl(task, stub_batch):
    """The inherited version rebuilds both masks with no anchor set, and its own docstring promises
    it uses "the KL's own anchor support ... so the two cannot drift". Under tiling that promise
    fails: the gap would average the belief shift over every anchor while the
    ``source_conditioned_kl_raw`` printed beside it averages over the roughly ten the objective saw.
    """
    module = task()
    model = module.orig_model
    # The TRAINING geometry, explicitly. Outside a step ``_build_forward_inputs`` resolves the dense
    # one by design, and at stride 1 the tiled support and the dense support coincide -- so this
    # test would hold on the inherited implementation.
    phase, stride = module.resolve_anchor_geometry("train", stub_batch)
    y_st, y_ph = module._build_target_streams(stub_batch)
    outs = model(y_st, y_ph, module._build_source_stream(stub_batch), phase, stride)
    _target, weight = module._build_raw_target(stub_batch)

    forecast, _coverage = forecast_mask(
        weight,
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=outs["anchor_index"],
        anchor_valid=outs["anchor_valid"],
    )
    tiled_support = kl_mask(
        forecast, model.geometry, anchors=outs["anchor_index"], anchor_valid=outs["anchor_valid"]
    )
    dense_forecast, _ = forecast_mask(weight, model.geometry, coverage_floor=model.coverage_floor)
    dense_support = kl_mask(dense_forecast, model.geometry)

    # The two supports really do differ, or the assertion below would hold on the inherited version.
    assert not torch.equal(tiled_support, dense_support)
    gap_sq = ((outs["mu_post"] - outs["mu_prior"]) ** 2).sum(dim=-1)
    expected = torch.sqrt(
        (gap_sq * tiled_support).sum() / tiled_support.sum().clamp_min(1.0)
    )
    assert torch.equal(module._mu_gap_rms(outs, weight), expected)


def test_the_source_null_floor_is_reported_on_validation_and_not_on_training(
    task, stub_batch, perturb_posterior
):
    """It is a readout that never enters the objective and costs a source encode per step. Absent,
    never zero-filled, on the steps that did not run it: the framework's epoch value is the mean over
    the steps that reported a metric, so a zero placeholder would scale the aggregate toward
    nothing."""
    module = task()
    perturb_posterior(module.orig_model)

    _, train_metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    _, val_metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert "kld_source_null" not in train_metrics
    assert "kld_source_null" in val_metrics
    assert torch.isfinite(val_metrics["kld_source_null"])


def test_the_source_null_floor_is_the_controls_own_function_at_the_forwards_anchors(
    task, stub_batch, perturb_posterior
):
    """Driven against the free function rather than against a number, so the task is asserted to be
    a call site rather than a second implementation -- and at the anchors the forward decoded, which
    is what makes the difference from ``source_conditioned_kl_raw`` a subtraction of two things
    measured the same way."""
    module = task()
    perturb_posterior(module.orig_model)
    inputs = module._build_forward_inputs(stub_batch)
    outs = module.orig_model(*inputs)
    _target, weight = module._build_raw_target(stub_batch)

    expected = controls.source_null_kld(module.orig_model, outs, inputs[2], weight)
    added = module._added_metrics(inputs, outs, weight, "val")

    assert torch.equal(added["kld_source_null"], expected)


def test_a_readout_reusing_an_objective_metric_name_is_refused(
    task, stub_batch, monkeypatch
):
    """The hook merges last, so a plain update would let a readout replace an objective metric
    *under the objective's own name*.

    ``pred_gap`` in ``metrics_history.csv`` would then stop meaning what it means in every other
    cell of the grid, with no error and nothing in the log -- and the column is read across models.
    Refused instead, naming the key.
    """
    module = task()
    monkeypatch.setattr(
        type(module),
        "_added_metrics",
        lambda self, inputs, outs, weight, stage: {"pred_gap": torch.zeros(())},
    )

    with pytest.raises(ValueError, match=r"pred_gap"):
        module.compute_loss_and_metrics(stub_batch, 0, "val")

    # Not vacuous: a name of its own still merges, which is what the hook is for.
    monkeypatch.setattr(
        type(module),
        "_added_metrics",
        lambda self, inputs, outs, weight, stage: {"a_name_no_objective_uses": torch.zeros(())},
    )
    _loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")
    assert "a_name_no_objective_uses" in metrics


def test_a_source_derangement_leaves_the_null_floor_unchanged(
    task, stub_batch, perturb_posterior
):
    """The property that makes it a control the shuffle is not. The availability pattern is
    identical in every row of the batch, so no permutation of rows can remove it -- which is exactly
    why a second arm was needed."""
    module = task()
    perturb_posterior(module.orig_model)
    inputs = module._build_forward_inputs(stub_batch)
    outs = module.orig_model(*inputs)
    _target, weight = module._build_raw_target(stub_batch)

    matched = controls.source_null_kld(module.orig_model, outs, inputs[2], weight)
    deranged = controls.source_null_kld(
        module.orig_model, outs, inputs[2][torch.tensor([1, 0])], weight
    )

    assert torch.equal(matched, deranged)


# ---------------------------------------------------------------------------------------
# The objective still runs, and still starts where it must
# ---------------------------------------------------------------------------------------
def test_every_metric_is_numeric_and_unprefixed(task, stub_batch, perturb_posterior):
    """A name carrying a '/' bypasses stage framing and can poison a ``ModelCheckpoint`` monitor."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}"
        assert "/" not in name


def test_the_loss_is_finite_and_carries_gradient(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert torch.equal(metrics["main_loss"], loss.detach())


def test_the_zero_kl_start_survives_the_task(task, stub_batch):
    """At initialisation the posterior *is* the prior, so the coupling readout starts at exactly
    zero and every nat it later reports had to be earned."""
    module = task()

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["source_conditioned_kl_raw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)


def test_a_batch_of_one_still_trains(task, perturb_posterior):
    """A rank can receive a batch of one under DDP. The permutation control refuses it; the step
    must not, and neither must the per-sample phase."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1), 0, "train")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics


def test_the_ungated_model_still_runs_through_this_task(task, stub_batch, tiny_kwargs):
    """No budget means no gate, no warm-up mask and no dropped channel -- and the task must not
    assume otherwise, because that arm is what every "the guard did something" comparison is made
    against."""
    module = task(model_kwargs=tiny_kwargs)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "val")

    assert torch.isfinite(loss)
    assert float(metrics["target_warm_frac"]) == 1.0
    # Every declared channel, in order: no gate, so the decoder emits the whole stream and the
    # guarded arm's dropped channels are back.
    assert module.orig_model.decoder_out_channels == int(tiny_kwargs["c_y"])
    assert module.orig_model.decoder_out_channels > len(
        tiny_warmup_kwargs()["target_keep_index"]
    )
