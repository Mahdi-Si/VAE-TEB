r"""The task is the sibling's task plus one re-pointed builder, and that is the whole point.

Two models are only comparable if they optimise the same thing. Here that is not a claim about two
copies of an objective agreeing -- it is the same code: the loss assembly, the $\beta$ schedule,
the metric surface, the permutation control, the spike-breaker wiring, the gradient-norm logging
and the checkpoint contract are all inherited unmodified. So most of the assertions below are about
*absence*: a re-added ``training_step`` silently disables the config-gated loss-spike breaker, a
re-added ``compute_loss_and_metrics`` takes back the permutation control and the ``main_loss`` name
the breaker watches, and neither fails anything on its own.

The one override is checked from both sides. It must return the concatenated target stream -- the
tensor the loss is computed against -- and it must return *only* that: the same batch, run through
the sibling's task and this one, must give two losses that differ solely because the target domain
differs, with the source pathway, the masks and the KL identical.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from train.pl_model_base import LightningModelBase

from .conftest import TASK_HPARAMS, TINY_KWARGS, make_stub_batch

#: Callables the subclass may define. A set rather than a count, following the sibling suite: a
#: count passes a subclass that took back ``training_step`` in 140 lines while dropping something
#: else. ``forecast_rows`` is deliberately **not** here and does not belong here: the shared
#: plotting callback resolves it with ``getattr(..., None)``, so an attribute of that name
#: overrides nothing and adding one is not a second override of anything. It is also a
#: ``property``, which is not itself callable, so the set below excludes it without a special
#: case -- and the tests at the bottom of this file assert what it resolves to.
_OWN_CALLABLES = {"_build_raw_target"}


def _sibling_task():
    """The raw-target model wrapped in its own task, at the same tiny geometry.

    Built here rather than taken from a fixture because the point of every comparison below is
    that the two tasks differ in exactly one place; a shared factory would have to choose a class
    and would hide which one it chose.

    Returns:
        A ``SeqVaeLagAttnRwsTask`` over ``TINY_KWARGS``.
    """
    from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**TINY_KWARGS)
    task = SeqVaeLagAttnRwsTask(model, lr=1e-3, model_kwargs=dict(TINY_KWARGS), **TASK_HPARAMS)
    task.setup("fit")
    return task


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_subclass_adds_exactly_one_method():
    """A second override is a second thing that can diverge from the objective being shared."""
    own = {
        name
        for name, value in vars(SeqVaeLagAttnFsTask).items()
        if callable(value) and not name.startswith("__")
    }

    assert own == _OWN_CALLABLES


@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers",
     "compute_loss_and_metrics", "on_save_checkpoint", "setup", "_mu_gap_rms",
     "_build_forward_inputs", "_build_target_streams", "_build_source_stream",
     "_resolve_beta", "__init__"],
)
def test_the_inherited_machinery_is_not_taken_back(method):
    """``training_step`` matters most: the framework's version runs the config-gated spike
    breaker, and a subclass defining its own silently disables it. ``_mu_gap_rms`` is the one that
    looks target-shaped and is not -- it reads the geometry, the coverage floor, the two masks and
    the two latent means, all of which are domain-neutral."""
    assert method not in vars(SeqVaeLagAttnFsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_constructor_goes_through_the_base(task):
    """Not through a grandparent ``LightningModule.__init__`` bypass, which would silently drop
    ``save_hyperparameters``, ``_orig_model``, ``self.model`` and the breaker counters."""
    module = task()

    assert isinstance(module, LightningModelBase)
    assert isinstance(module, SeqVaeLagAttnRwsTask)
    assert module.orig_model is module._orig_model
    assert hasattr(module, "_spike_ema_loss")
    assert module.hparams.get("lr") == 1e-3


def test_compilation_stays_off_and_the_eager_module_is_what_runs(task):
    """This net is the sibling's net, LSTM encoders included, so the refusal is inherited whole:
    the driver does not read ``advanced_config.trainer.compile`` at all."""
    from teb_vae.lag_attn_fs.trainer import LagAttnFsTrainer

    module = task()

    assert module.model is module.orig_model
    assert module.hparams.get("compile_model") is False
    assert LagAttnFsTrainer.compile_model_requested(object()) is False
    assert "compile_model_requested" not in vars(LagAttnFsTrainer)


# ---------------------------------------------------------------------------------------
# The one override
# ---------------------------------------------------------------------------------------
def test_the_target_is_the_two_blocks_concatenated_in_the_declared_order(task, patterned_batch):
    """Asserted against the planted pattern rather than against a shape: at these widths a
    transposed or reversed concatenation produces the same $(B, T, 109)$ tensor, and only a value
    check says which channel ended up where. The declared order is what the reach budget's
    keep-index is positional into."""
    module = task()

    target, weight = module._build_raw_target(patterned_batch)

    assert target.shape == (patterned_batch.fhr_st.shape[0], patterned_batch.fhr_st.shape[1], 109)
    assert torch.equal(target[..., :43], patterned_batch.fhr_st)
    assert torch.equal(target[..., 43:], patterned_batch.fhr_ph)
    assert weight is patterned_batch.weight


def test_the_target_is_the_undelayed_stream(task, patterned_batch, tiny_gated):
    """The sharpest correctness trap in the build, checked at the task boundary as well as at the
    net's: a target run through the input gate would ask anchor $t$ for the future of anchor
    $t - \\delta_c$, per channel, with every shape downstream unchanged."""
    module = task(model_kwargs=tiny_gated)

    target, _ = module._build_raw_target(patterned_batch)

    gated = module.orig_model.target_gate(target)
    assert target.shape[-1] == 109 != gated.shape[-1]
    assert not torch.equal(target[..., module.orig_model.target_gate.keep_index], gated)


def test_the_target_is_what_the_loss_is_actually_computed_against(
    task, stub_batch, perturb_posterior
):
    """The builder is only correct if the step uses it. Driven through the real
    ``compute_loss_and_metrics`` and compared against the net's objective called by hand on the
    builder's output -- a step that rebuilt the target another way would agree here only by
    coincidence."""
    module = task()
    perturb_posterior(module.orig_model)
    target, weight = module._build_raw_target(stub_batch)

    torch.manual_seed(5)
    _loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")
    torch.manual_seed(5)
    outs = module.orig_model(*module._build_forward_inputs(stub_batch))
    expected = module.orig_model.compute_loss(
        outs, target, weight=weight, beta=module._resolve_beta(0), **{
            key: TASK_HPARAMS[key]
            for key in ("lambda_full", "lambda_base", "likelihood", "free_bits")
        }
    )["metrics"]

    assert torch.equal(metrics["nll_full_block"], expected["nll_full_block"])
    assert torch.equal(metrics["nll_base_block"], expected["nll_base_block"])


def test_a_missing_weight_names_the_config_key_that_fixes_it(task, stub_batch):
    """The stored coefficients carry no detectable gap sentinel of their own, so the decimated
    weight is the only trustworthy validity signal for this target too."""
    module = task()
    del stub_batch.weight

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_raw_target(stub_batch)


def test_a_target_width_mismatch_is_caught_by_the_inherited_check(task, stub_batch):
    """Reusing ``_build_target_streams`` rather than reading the batch again is what puts this
    check on the target path: the target *is* the input stream, so one rule covers both."""
    module = task()
    stub_batch.fhr_ph = torch.randn(stub_batch.fhr_st.shape[0], stub_batch.fhr_st.shape[1], 44)

    with pytest.raises(RuntimeError, match="target stream is 87 channels"):
        module._build_raw_target(stub_batch)


def test_a_shard_whose_blocks_split_elsewhere_is_refused_naming_the_declared_split(
    task, stub_batch
):
    """The check that keeps the two per-block gap columns honest.

    A shard whose blocks are $42$ and $67$ still totals the declared $c_y = 109$, so the joint
    width check above passes and every shape downstream is correct -- and the net, which splits at
    a number it cannot derive from a *sum*, would report one channel of the second block inside
    ``pred_gap_st``. Nothing else in the run depends on the split, which is exactly why it has to be
    refused here rather than left to be noticed.
    """
    module = task()
    batch_size, seq_len = stub_batch.fhr_st.shape[0], stub_batch.fhr_st.shape[1]
    stub_batch.fhr_st = torch.randn(batch_size, seq_len, 42)
    stub_batch.fhr_ph = torch.randn(batch_size, seq_len, 67)

    with pytest.raises(RuntimeError, match=r"TARGET_BLOCK_SPLIT"):
        module._build_raw_target(stub_batch)


def test_the_declared_split_matches_the_committed_shards_blocks(task, stub_batch):
    """The other direction: the value shipped in the net is the one the data actually has, so the
    guard above is not simply refusing everything."""
    module = task()

    assert module.orig_model.TARGET_BLOCK_SPLIT == int(stub_batch.fhr_st.shape[-1]) == 43
    assert (
        int(stub_batch.fhr_st.shape[-1]) + int(stub_batch.fhr_ph.shape[-1])
        == module.orig_model.c_y
    )


def test_the_forward_inputs_are_unchanged_by_the_override(task, stub_batch):
    """The net is fed the two blocks separately and scored against their concatenation. Both come
    off the same builder, so they cannot disagree about which tensor the model saw."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)
    target, _ = module._build_raw_target(stub_batch)

    assert len(inputs) == 3
    assert torch.equal(torch.cat([inputs[0], inputs[1]], dim=-1), target)


# ---------------------------------------------------------------------------------------
# The metric surface, against the model this one is compared with
# ---------------------------------------------------------------------------------------
def test_every_metric_is_numeric_and_unprefixed(task, stub_batch, perturb_posterior):
    """A name carrying a '/' bypasses stage framing and can poison a ``ModelCheckpoint``
    monitor."""
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}"
        assert "/" not in name


def test_the_metric_set_is_the_siblings_plus_the_declared_observability_columns(
    task, stub_batch, perturb_posterior
):
    """Driven from both tasks' real metrics dicts rather than from a list, so the claim is about
    the two tasks. The four additions are the horizon- and block-resolved forecast gaps, which the
    sibling has no reason to report -- its block's last axis counts raw samples of one signal, and
    the whole point of resolving the gap here is that this block's channels are not one thing."""
    mine = task()
    theirs = _sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, "train")
    _, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, "train")

    assert set(my_metrics) - set(their_metrics) == {
        "pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph"
    }
    assert set(their_metrics) - set(my_metrics) == set()


def test_the_validation_metric_set_adds_the_same_three_readouts(
    task, stub_batch, perturb_posterior
):
    """Validation adds the permutation control's three, which only a source-conditioned forward
    can produce -- so this is also where a control that silently stopped running shows."""
    mine = task()
    theirs = _sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, "val")
    _, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, "val")

    control = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}
    assert control <= set(my_metrics)
    assert set(my_metrics) - set(their_metrics) == {
        "pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph"
    }


def test_main_loss_is_emitted_unprefixed_and_is_what_the_breaker_consumes(task):
    """Emission is not consumption: the framework falls back to the returned loss when
    ``metrics['main_loss']`` is missing, silently. Drive the real breaker with a ``main_loss`` far
    below the returned loss and check which one seeded the EMA."""
    module = task(
        spike_breaker={"enabled": True, "warmup_batches": 0, "comparison_metric": "main_loss"}
    )

    returned = torch.tensor(100.0, requires_grad=True)
    metrics = {"total_loss": returned, "main_loss": torch.tensor(1.0)}
    module._apply_spike_breaker(returned, metrics, module.hparams["spike_breaker"])

    assert float(module._spike_ema_loss) == pytest.approx(1.0), (
        "the breaker seeded its EMA from the returned loss, so it is not watching main_loss"
    )


def test_the_loss_is_finite_and_carries_gradient(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert torch.equal(metrics["main_loss"], loss.detach())


def test_the_zero_kl_start_survives_the_task(task, stub_batch):
    """At initialisation the posterior *is* the prior, so the coupling readout starts at exactly
    zero and every nat it later reports had to be earned. Seen here through the task's own
    diagnostic, after the batch assembly and the loss have both run."""
    module = task()

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["source_conditioned_kl_raw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["pred_gap"]) == pytest.approx(0.0, abs=1e-6)


def test_the_scheduled_beta_is_what_weights_the_kl_and_what_is_reported(
    task, stub_batch, perturb_posterior
):
    """The schedule is inherited, and what this checks is that the inherited resolution still
    reaches the objective through a re-pointed target builder."""
    module = task(
        hparams={
            "beta_schedule": {
                "kind": "linear_warmup", "start": 0.0, "end": 5.0, "warmup_epochs": 10
            },
            "kld_beta": 0.01,
        }
    )
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["kld_beta"]) == pytest.approx(module._resolve_beta(module.current_epoch))
    assert float(metrics["kld_beta"]) != pytest.approx(0.01)


def test_the_configured_beta_prior_weights_the_objective_and_is_echoed(
    task, stub_batch, perturb_posterior
):
    """The anchor term reaches the loss by value through this model's delegating ``compute_loss``
    -- a wrapper that accepts the keyword and drops it passes any signature check."""
    anchored = task(hparams={"beta_prior": 0.5})
    unanchored = task(hparams={"beta_prior": 0.0})
    perturb_posterior(anchored.orig_model)
    perturb_posterior(unanchored.orig_model)  # same seed in the factory -> identical weights

    torch.manual_seed(2)
    loss_anchored, metrics = anchored.compute_loss_and_metrics(stub_batch, 1, "train")
    torch.manual_seed(2)
    loss_unanchored, _ = unanchored.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["beta_prior"]) == pytest.approx(0.5)
    assert float(metrics["prior_rate"]) > 0.0
    assert float(loss_anchored - loss_unanchored) == pytest.approx(
        0.5 * float(metrics["prior_rate"]), rel=1e-4
    )


def test_the_validity_mask_changes_the_loss(task, make_stub_batch_fn, perturb_posterior):
    """A weight the loss ignored would let gaps pollute every term, silently."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch_fn()

    torch.manual_seed(1)
    _, all_valid = module.compute_loss_and_metrics(batch, 1, "train")
    batch.weight[:, : batch.weight.shape[1] // 2] = 0.0
    torch.manual_seed(1)
    _, half_masked = module.compute_loss_and_metrics(batch, 1, "train")

    assert float(all_valid["nll_full_block"]) != pytest.approx(
        float(half_masked["nll_full_block"]), rel=1e-6
    )


# ---------------------------------------------------------------------------------------
# The checkpoint stamp
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_written_by_this_task_stamps_this_model(task):
    """The stamp is what lets a loader refuse a blob from the raw-target sibling before trying to
    align it -- the two share every tensor name but the decoder head's."""
    module = task()
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["model_class"] == "SeqVaeLagAttnFs"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    assert checkpoint["epoch"] == 3  # the override must not clobber Lightning's own fields


# ---------------------------------------------------------------------------------------
# The seam the diagnostic page reaches through
# ---------------------------------------------------------------------------------------
def test_the_shared_plot_callback_reaches_the_target_through_this_builder():
    """The callback calls ``pl_module._build_raw_target(batch)`` and hands the result to the net's
    own ``compute_loss``, which is why the concatenation belongs in the task: a figure cannot then
    disagree with the objective it illustrates about what was scored.

    Checked in the source because a *surviving* second route produces a correct-looking figure
    drawn against the wrong tensor rather than a failure."""
    from teb_vae.lag_attn_rws import plotting

    source = inspect.getsource(plotting.LagAttnRwsPlotCallback._generate_plots)

    assert "pl_module._build_raw_target(batch)" in source
    assert "batch.fhr" not in source


def test_the_page_seam_is_resolved_off_the_task_and_this_task_fills_it(task):
    """The callback resolves ``forecast_rows`` with ``getattr(..., None)`` and the shared builder
    turns a ``None`` back into its own raw rows, so this attribute is the entire route by which a
    model in another target domain replaces two rows of a seven-row page.

    Three halves, because any one alone would be misleading: the callback really does look for the
    name, this task really does answer it, and what it answers with is a one-argument callable --
    the seam's contract. A plain class attribute would satisfy the first two and fail the third,
    because a function assigned to a class binds ``self`` into the row builder's first parameter.
    """
    from teb_vae.lag_attn_fs.sample_page import feature_forecast_rows
    from teb_vae.lag_attn_rws import plotting

    rows = task().forecast_rows

    assert "forecast_rows" in inspect.getsource(plotting.LagAttnRwsPlotCallback._generate_plots)
    assert rows.func is feature_forecast_rows
    assert set(rows.keywords) == {"keep_index", "block_split"}
    required = [
        name
        for name, parameter in inspect.signature(rows).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    assert required == ["rows"], "the callback calls this with exactly one positional argument"


def test_the_seam_carries_the_channel_facts_the_page_cannot_recover(task, shipped_gated):
    """The keep-index and the block split are the two things the page is handed rather than
    derives. Without the first it would draw every lane's forecast against the wrong channel's
    truth and label it with a positional index that changes meaning at the next reach budget;
    without the second the error map's two stored blocks would run together."""
    from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs

    guarded = task(model_kwargs=shipped_gated).forecast_rows
    unguarded = task().forecast_rows

    assert list(guarded.keywords["keep_index"]) == list(shipped_gated["target_keep_index"])
    assert guarded.keywords["block_split"] == SeqVaeLagAttnFs.TARGET_BLOCK_SPLIT
    # And the ungated arm, where the decoder emits every declared channel in order and there is no
    # gate to read an index off at all.
    assert unguarded.keywords["keep_index"] is None


def test_the_task_never_reaches_the_net_around_the_builders(task, stub_batch):
    """A remaining direct read of a batch field inside the inherited step would make the override
    advisory: it would run, and the loss would go on being computed against the raw trace."""
    source = inspect.getsource(SeqVaeLagAttnRwsTask.compute_loss_and_metrics)

    assert "self._build_raw_target(batch)" in source
    assert "batch.fhr" not in source
    # And the module under test genuinely routes through it, at both call sites in one step.
    module = task()
    calls = []
    original = module._build_raw_target
    module._build_raw_target = lambda batch: (calls.append(batch) or original(batch))
    module.compute_loss_and_metrics(stub_batch, 0, "val")
    assert len(calls) == 1


def test_the_source_stream_is_still_the_siblings(task, stub_batch):
    """The source pathway is untouched by the target-domain change, which is the premise the
    comparison between the two models rests on."""
    module = task()

    u_stream = module._build_source_stream(stub_batch)

    assert u_stream.shape[-1] == module.orig_model.c_u == 58
    assert torch.equal(u_stream[..., :43], stub_batch.up_st)
    assert torch.equal(u_stream[..., 43:], stub_batch.up_ph)


def test_a_batch_of_one_still_trains(task, perturb_posterior):
    """A rank can receive a batch of one under DDP. The permutation control refuses it; the step
    must not."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1), 0, "train")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics
