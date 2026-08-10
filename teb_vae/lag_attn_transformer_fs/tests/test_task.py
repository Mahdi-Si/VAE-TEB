r"""The task is two parents and nothing of its own, and the diamond is what has to be asserted.

Every behaviour this class has is inherited, so the only defects it can carry are *resolution*
defects, and none of them raises. ``_build_raw_target`` resolving to the shared ancestor scores the
net against the raw FHR trace while every column keeps its name; ``build_lr_scheduler`` resolving
there leaves ``lr_warmup_steps: 2000`` in the config with no ramp attached anywhere, the CSV column
merely looking flat. So the linearisation is asserted as a list of class names, and each of the three
inherited behaviours against the class the design names -- not merely that the class is a subclass of
both, which a reordered diamond satisfies too.

The rest is about *absence*. A re-added ``training_step`` silently disables the config-gated
loss-spike breaker; a re-added ``compute_loss_and_metrics`` takes back the permutation control and
the ``main_loss`` name the breaker watches; a constructor of its own is a second keyword schema for
one shared objective. None of those fails anything on its own, which is why each is checked by name.

What is genuinely new here -- as opposed to inherited from a suite that already ran it -- is the
permutation control's *gating through this task*: the control is duck-typed on six model attributes
and both sibling suites already exercise it, one against these encoders and one against this target,
so what nobody has yet run is the pair. One test, not a ported module.
"""
from __future__ import annotations

import inspect

import pytest
import torch

from teb_vae.lag_attn_fs.sample_page import feature_forecast_rows
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from train.pl_model_base import LightningModelBase

from .conftest import SEQ_LEN, TASK_HPARAMS, TINY_KWARGS, make_stub_batch

#: The linearisation the design names, down to the shared framework base. A list rather than two
#: ``issubclass`` checks: both orders of the bases are legal Python and only one of them puts the
#: feature target's ``_build_raw_target`` ahead of the shared ancestor's. Truncated at
#: ``LightningModelBase`` because everything below it is the framework's own hierarchy and is not
#: this package's statement to make.
_EXPECTED_MRO = [
    "SeqVaeLagAttnTrfFsTask",
    "SeqVaeLagAttnFsTask",
    "SeqVaeLagAttnTrfRwsTask",
    "SeqVaeLagAttnRwsTask",
    "LightningModelBase",
]

#: The three permutation-control readouts, which only a source-conditioned validation forward can
#: produce.
_CONTROL_METRICS = {"nll_shuffled_block", "kld_shuffled", "shuffle_penalty"}


class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads off ``self.trainer``, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the whole run, not per epoch, and
    is typed ``float`` because an unlimited run is reported as infinity.
    """

    def __init__(self, estimated_stepping_batches: float = 100.0, max_epochs: int = 10) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_task_defines_nothing_at_all():
    """Asserted by set equality over ``vars(...)`` rather than by a line count, which passes a class
    that took back ``compute_loss_and_metrics`` in 90 lines. ``forecast_rows`` is a ``property`` and
    so is not itself callable, and it is not defined here in any case -- the tests below assert what
    it resolves to.

    ``_abc_impl`` is excluded because it is not the class body's: ``LightningModule`` derives from
    ``ABC``, and ``ABCMeta`` writes that cache into the ``__dict__`` of every subclass it creates.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfFsTask).items()
        if callable(value) and not name.startswith("__")
    }

    assert own == set()
    assert [
        name
        for name in vars(SeqVaeLagAttnTrfFsTask)
        if not name.startswith("__") and name != "_abc_impl"
    ] == []


def test_the_diamond_linearises_the_way_the_design_measured_it():
    """A diamond that silently reorders changes what the model trains on and what schedule it trains
    under, and nothing else would notice. Both parents descend from the shared task, which is why
    the linearisation exists at all -- not because the two branches happen to be disjoint."""
    names = [cls.__name__ for cls in SeqVaeLagAttnTrfFsTask.__mro__]

    assert names[: len(_EXPECTED_MRO)] == _EXPECTED_MRO
    assert issubclass(SeqVaeLagAttnTrfFsTask, LightningModelBase)


def test_the_target_builder_and_the_page_seam_come_from_the_feature_parent():
    """``_build_raw_target`` resolving to the shared ancestor would score this net against the raw
    FHR trace -- a running, reporting model optimising a different objective under this config. The
    page seam is the other half of the same parent's contribution: the shared callback resolves
    ``forecast_rows`` with ``getattr(..., None)`` and reads a ``None`` as "use the raw page", so a
    task that lost it would draw a plausible figure of the wrong rows."""
    assert SeqVaeLagAttnTrfFsTask._build_raw_target is SeqVaeLagAttnFsTask._build_raw_target
    assert (
        SeqVaeLagAttnTrfFsTask._build_raw_target is not SeqVaeLagAttnRwsTask._build_raw_target
    )
    assert SeqVaeLagAttnTrfFsTask.forecast_rows is SeqVaeLagAttnFsTask.forecast_rows


def test_the_step_granular_ramp_comes_from_the_conv_transformer_parent():
    """The ramp exists only on that parent. Resolving to the shared ancestor instead would leave
    ``lr_warmup_steps: 2000`` in the config, the learning-rate monitor logging per step, and no ramp
    attached anywhere -- with nothing raising and the CSV column merely looking flat."""
    assert (
        SeqVaeLagAttnTrfFsTask.build_lr_scheduler
        is SeqVaeLagAttnTrfRwsTask.build_lr_scheduler
    )
    assert (
        SeqVaeLagAttnTrfFsTask.build_lr_scheduler
        is not SeqVaeLagAttnRwsTask.build_lr_scheduler
    )


@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers",
     "compute_loss_and_metrics", "on_save_checkpoint", "setup", "_mu_gap_rms",
     "_build_forward_inputs", "_build_target_streams", "_build_source_stream",
     "_resolve_beta", "_should_run_perm", "__init__"],
)
def test_the_inherited_machinery_is_not_taken_back(method):
    """``training_step`` matters most: the framework's version runs the config-gated spike breaker,
    and a subclass defining its own silently disables it. ``configure_optimizers`` is the second --
    it is what calls ``build_lr_scheduler`` at all."""
    assert method not in vars(SeqVaeLagAttnTrfFsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


def test_the_constructor_keyword_set_is_the_shared_objectives():
    """Compared as an ordered mapping against both parents, because a reordered signature is still a
    different call and this is what a later offline pass would rebuild the task from."""
    mine = list(inspect.signature(SeqVaeLagAttnTrfFsTask.__init__).parameters)

    assert mine == list(inspect.signature(SeqVaeLagAttnFsTask.__init__).parameters)
    assert mine == list(inspect.signature(SeqVaeLagAttnTrfRwsTask.__init__).parameters)


def test_the_constructor_goes_through_the_base(task):
    """Not through a grandparent ``LightningModule.__init__`` bypass, which would silently drop
    ``save_hyperparameters``, ``_orig_model``, ``self.model`` and the breaker counters."""
    module = task()

    assert isinstance(module, SeqVaeLagAttnTrfFsTask)
    assert module.orig_model is module._orig_model
    assert hasattr(module, "_spike_ema_loss")
    assert module.hparams.get("lr") == 1e-3


# ---------------------------------------------------------------------------------------
# The inherited target builder, reached through this task
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


def test_a_shard_whose_blocks_split_elsewhere_is_refused_with_the_feature_parents_message(
    task, stub_batch
):
    """The check that keeps the two per-block gap columns honest, and it has to arrive through the
    diamond rather than be restated.

    A shard whose blocks are $42$ and $67$ still totals the declared $c_y = 109$, so the joint-width
    check passes and every shape downstream is correct -- and the net, which splits at a number it
    cannot derive from a *sum*, would report one channel of the second block inside ``pred_gap_st``.
    Nothing else in the run depends on the split, which is why it is refused here rather than left
    to be noticed.
    """
    module = task()
    batch_size, seq_len = stub_batch.fhr_st.shape[0], stub_batch.fhr_st.shape[1]
    stub_batch.fhr_st = torch.randn(batch_size, seq_len, 42)
    stub_batch.fhr_ph = torch.randn(batch_size, seq_len, 67)

    with pytest.raises(RuntimeError, match=r"TARGET_BLOCK_SPLIT"):
        module._build_raw_target(stub_batch)


def test_the_declared_split_matches_the_committed_shards_blocks(task, stub_batch):
    """The other direction: the value the mixin ships is the one the data actually has, so the guard
    above is not simply refusing everything."""
    module = task()

    assert module.orig_model.TARGET_BLOCK_SPLIT == int(stub_batch.fhr_st.shape[-1]) == 43
    assert (
        int(stub_batch.fhr_st.shape[-1]) + int(stub_batch.fhr_ph.shape[-1])
        == module.orig_model.c_y
    )


def test_a_missing_weight_names_the_config_key_that_fixes_it(task, stub_batch):
    """The stored coefficients carry no detectable gap sentinel of their own, so the decimated
    weight is the only trustworthy validity signal for this target."""
    module = task()
    del stub_batch.weight

    with pytest.raises(RuntimeError, match="load_fields"):
        module._build_raw_target(stub_batch)


def test_the_forward_inputs_are_the_two_blocks_the_target_concatenates(task, stub_batch):
    """The net is fed the two blocks separately and scored against their concatenation. Both come
    off the same builders, so they cannot disagree about which tensor the model saw -- and the
    signature is the one this architecture's ``forward`` takes, unchanged by the target domain."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)
    target, _weight = module._build_raw_target(stub_batch)

    assert len(inputs) == 3
    assert torch.equal(torch.cat([inputs[0], inputs[1]], dim=-1), target)
    assert inputs[2].shape[-1] == module.orig_model.c_u == 58


# ---------------------------------------------------------------------------------------
# The page seam, resolved off this task
# ---------------------------------------------------------------------------------------
def test_the_page_seam_binds_this_nets_channel_facts(task, shipped_gated):
    """The keep-index and the block split are the two things the page is handed rather than derives,
    and both are read off *this* net's own gate -- which the conv-Transformer base builds at its own
    construction site. Read back off the partial rather than inferred from a successful render."""
    guarded = task(model_kwargs=shipped_gated).forecast_rows
    unguarded = task().forecast_rows

    assert guarded.func is feature_forecast_rows
    assert set(guarded.keywords) == {"keep_index", "block_split"}
    assert list(guarded.keywords["keep_index"]) == list(shipped_gated["target_keep_index"])
    assert guarded.keywords["block_split"] == 43
    # The ungated arm, where the decoder emits every declared channel in order and there is no gate
    # to read an index off at all.
    assert unguarded.keywords["keep_index"] is None
    required = [
        name
        for name, parameter in inspect.signature(guarded).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    assert required == ["rows"], "the callback calls this with exactly one positional argument"


# ---------------------------------------------------------------------------------------
# The learning-rate schedule, reached through the diamond
# ---------------------------------------------------------------------------------------
def test_the_step_warmup_is_attached_at_step_granularity(task):
    """The single assertion that catches the wrong-branch resolution above at run time rather than
    by identity. A ramp measured in optimizer steps but attached at ``interval: "epoch"`` takes
    ``lr_warmup_steps`` *epochs* -- silently. The ramp's arithmetic is inherited code and is pinned
    in the sibling's own suite; ``tests/test_trainer.py`` traces the rate moving per step."""
    module = task()
    setattr(module.hparams, "lr_warmup_steps", 4)
    module.trainer = _FakeTrainer()

    schedule = module.build_lr_scheduler(
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    )

    assert isinstance(schedule, dict)
    assert schedule["interval"] == "step"
    assert schedule["frequency"] == 1


# ---------------------------------------------------------------------------------------
# The metric surface
# ---------------------------------------------------------------------------------------
def test_every_metric_is_numeric_and_unprefixed(task, stub_batch, perturb_posterior):
    """A name carrying a '/' bypasses stage framing and can poison a ``ModelCheckpoint`` monitor."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    for name, value in metrics.items():
        assert isinstance(value, torch.Tensor), f"{name} is a {type(value).__name__}"
        assert "/" not in name


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


def test_the_metric_set_is_the_feature_siblings_exactly(task, stub_batch, perturb_posterior):
    """Driven from both tasks' real metrics dicts rather than from a list, so the claim is about the
    two tasks. This model differs from the feature sibling in its encoders alone, so a metric either
    of them emitted and the other did not would be a column the shared tracked list cannot collect
    -- and the four forecast-gap splits, which are the whole observability addition, arrive through
    the same mixin in both."""
    mine = task()
    theirs = _feature_sibling_task()
    perturb_posterior(mine.orig_model)
    perturb_posterior(theirs.orig_model)

    _loss, my_metrics = mine.compute_loss_and_metrics(stub_batch, 0, "train")
    _loss, their_metrics = theirs.compute_loss_and_metrics(stub_batch, 0, "train")

    assert set(my_metrics) == set(their_metrics)
    assert {
        "pred_gap_tau_first", "pred_gap_tau_last", "pred_gap_st", "pred_gap_ph"
    } <= set(my_metrics)


def _feature_sibling_task():
    """The conv-LSTM feature model wrapped in its own task, at that suite's tiny geometry.

    Built from the *feature* suite's keyword set rather than from this one's: the two constructors'
    schemas differ by six keywords, so one set cannot build both models. What the two share is the
    objective and the metric surface, which is exactly what the comparison above is about.

    Returns:
        A ``SeqVaeLagAttnFsTask`` at the feature suite's tiny geometry.
    """
    from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
    from teb_vae.lag_attn_fs.tests.conftest import TINY_KWARGS as SIBLING_TINY_KWARGS

    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**SIBLING_TINY_KWARGS)
    task = SeqVaeLagAttnFsTask(
        model, lr=1e-3, model_kwargs=dict(SIBLING_TINY_KWARGS), **TASK_HPARAMS
    )
    task.setup("fit")
    return task


def test_the_loss_is_finite_and_carries_gradient(task, stub_batch, perturb_posterior):
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert torch.equal(metrics["main_loss"], loss.detach())


def test_the_zero_kl_start_survives_the_task(task, stub_batch):
    """At initialisation the posterior *is* the prior, so the coupling readout starts at exactly zero
    and every nat it later reports had to be earned. Seen here through the task's own diagnostic,
    after the batch assembly and the loss have both run."""
    module = task()

    _loss, metrics = module.compute_loss_and_metrics(stub_batch, 1, "train")

    assert float(metrics["source_conditioned_kl_raw"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["mu_post_prior_gap_rms"]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------
# The permutation control, gated through this task
# ---------------------------------------------------------------------------------------
def test_the_permutation_control_runs_on_a_validation_batch_of_two_or_more(
    task, perturb_posterior
):
    """The control is duck-typed on ``query_uses_logvar``, ``query_proj``, ``lag_attn``,
    ``posterior_head``, ``decoder`` and ``geometry`` -- attributes, not a model class -- and both
    sibling suites already run it, one against these encoders and one against this target. What is
    new is the pair, gated through this task: a model missing one of those names would simply stop
    producing the three specificity readouts while every other column looked healthy.

    Perturbed first, because at initialisation a deranged source moves nothing and
    ``shuffle_penalty`` is $0$ for a reason that has nothing to do with being correct.
    """
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(4, SEQ_LEN), 0, "val")

    assert _CONTROL_METRICS <= set(metrics)
    assert float(metrics["shuffle_penalty"]) != pytest.approx(0.0, abs=1e-6)
    assert float(metrics["nll_shuffled_block"]) == pytest.approx(
        float(metrics["nll_full_block"]) + float(metrics["shuffle_penalty"]), rel=1e-5
    )


def test_a_batch_of_one_skips_the_control_and_still_trains(task, perturb_posterior):
    """A rank can receive a batch of one under DDP, and a single sample cannot be deranged. The
    metrics are *absent* rather than zero-filled: the framework aggregates an epoch metric as the
    mean over the steps that reported it, so a zero placeholder would scale the aggregate down and
    invert the $D_{\\mathrm{full}} < D_{\\mathrm{base}} < D_{\\mathrm{shuffled}}$ reading the control
    exists to give."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1, SEQ_LEN), 0, "val")

    assert torch.isfinite(loss)
    assert _CONTROL_METRICS.isdisjoint(set(metrics))


def test_the_control_never_runs_on_a_training_batch(task, perturb_posterior):
    """It is a readout and never enters the objective, on any stage."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(4, SEQ_LEN), 0, "train")

    assert _CONTROL_METRICS.isdisjoint(set(metrics))


# ---------------------------------------------------------------------------------------
# The checkpoint stamp
# ---------------------------------------------------------------------------------------
def test_a_checkpoint_written_by_this_task_stamps_this_model(task):
    """The stamp is what lets a loader refuse a blob from any of the three siblings before trying to
    align it -- the feature one shares every tensor name but the encoders', and the encoder one
    every tensor name but the decoder head's."""
    module = task()
    checkpoint = {"state_dict": module.state_dict(), "epoch": 3}

    module.on_save_checkpoint(checkpoint)

    assert checkpoint["model_class"] == "SeqVaeLagAttnTrfFs"
    assert checkpoint["model_kwargs"] == TINY_KWARGS
    assert checkpoint["epoch"] == 3  # the override must not clobber Lightning's own fields
