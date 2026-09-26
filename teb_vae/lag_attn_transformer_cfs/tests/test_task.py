r"""The task is one diamond: the causal cell on one side, the step-granular ramp on the other.

Two models are only comparable if they optimise the same thing, and here that is not a claim about
two copies of an objective agreeing -- it is the same code, reached through both parents at once.
So the structural assertions are about *absence* and about *resolution*: a re-added
``training_step`` silently disables the config-gated loss-spike breaker, a re-added
``_build_raw_target`` gives this model a second target builder that could drift from the one the
comparison models are scored through, and neither fails anything on its own.

**The diamond is legal because both parents descend from the shared task**, not because their added
members happen to be disjoint. They are today, but that is a fact about today's code, so every
member either parent defines is asserted to resolve to that parent. A future member defined on
both sides would resolve to the causal side by order alone, silently.

Through the assembled task: every metric the driver tracks is emitted on its stage (validation-only
readouts absent, never zero-filled, on training), and the two geometry guards read their derived
values -- tiled on training, dense on validation.
"""
from __future__ import annotations

from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_transformer_cfs.task import SeqVaeLagAttnTrfCfsTask
from teb_vae.lag_attn_transformer_cfs.trainer import LagAttnTrfCfsTrainer
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

from .conftest import TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

#: Tracked names logged by the framework's step hooks and callbacks rather than returned by
#: ``compute_loss_and_metrics``: the spike breaker's two, the gradient-norm pair and the learning
#: rate.
_LOGGED_OUTSIDE_THE_STEP = {"spike_skipped", "spike_ema_loss", "grad_norm", "grad_clip_frac", "lr"}


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_class_body_is_empty():
    """Set equality over ``vars``, not a line count.

    With nothing defined here, the objective, its $\\beta$ schedule, the metric surface, the
    permutation control, the spike-breaker wiring and the checkpoint contract cannot have moved:
    they are the shared task's own code objects, reached through two parents.

    ``_abc_impl`` is written by ``abc`` on every concrete subclass and is not this class's.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfCfsTask).items()
        if callable(value) or isinstance(value, (property, classmethod, staticmethod))
    }

    assert own == set()
    assert {
        name for name in vars(SeqVaeLagAttnTrfCfsTask) if not name.startswith("__")
    } <= {"_abc_impl"}
    assert "__init__" not in vars(SeqVaeLagAttnTrfCfsTask)


def test_the_two_branches_are_disjoint_today_and_each_member_resolves_to_its_parent():
    """The diamond's one silent hazard, made explicit. If a member ever appears on both sides it
    resolves to the causal parent, and this test is what turns that from an accident into a
    decision someone had to make. The tiling, the page seams and the constructor that carries the
    run seed come from the causal side, the target builder from the feature grandparent that
    precedes the conv-Transformer parent in the linearisation; the step-granular learning-rate ramp
    from the other."""
    # ``_abc_impl`` is written by ``abc`` onto every concrete subclass, so it is on both sides for
    # a reason that has nothing to do with either design.
    causal = {
        name
        for cls in (SeqVaeLagAttnCfsTask, SeqVaeLagAttnFsTask)
        for name in vars(cls)
        if not name.startswith("__") and name != "_abc_impl"
    }
    transformer = {
        name
        for name in vars(SeqVaeLagAttnTrfRwsTask)
        if not name.startswith("__") and name != "_abc_impl"
    }

    assert causal and transformer
    assert causal & transformer == set()
    for name in causal | transformer | {"__init__"}:
        owner = SeqVaeLagAttnTrfRwsTask if name in transformer else SeqVaeLagAttnCfsTask
        assert getattr(SeqVaeLagAttnTrfCfsTask, name) is getattr(owner, name), name


# ---------------------------------------------------------------------------------------
# The metric surface and the anchor geometry the stage decides
# ---------------------------------------------------------------------------------------
def test_every_tracked_metric_is_emitted_on_its_stage_and_nowhere_else(task, perturb_posterior):
    """Against the driver's own tracked list rather than a restated one. A tracked name the task
    never emits is a CSV column that is NaN in every row; a validation-only readout emitted on
    training -- even as a zero placeholder -- would scale the framework's epoch mean toward
    nothing, since that mean is over the steps that reported it."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch()

    emitted = {
        stage: set(module.compute_loss_and_metrics(batch, 0, stage)[1])
        for stage in ("train", "val")
    }
    tracked = {
        stage: {
            name.split("/", 1)[1]
            for name in LagAttnTrfCfsTrainer.TRACKED_METRICS
            if name.startswith(f"{stage}/")
        } - _LOGGED_OUTSIDE_THE_STEP
        for stage in ("train", "val")
    }

    for stage in ("train", "val"):
        missing = sorted(tracked[stage] - emitted[stage])
        assert missing == [], (stage, missing)
    validation_only = tracked["val"] - tracked["train"]
    assert validation_only, "no validation-only readout is tracked; the check below is vacuous"
    assert validation_only.isdisjoint(emitted["train"]), sorted(validation_only & emitted["train"])


def test_the_two_geometry_guards_read_their_derived_values(task, perturb_posterior):
    """Two of this domain's readouts are guards rather than results. ``target_warm_frac`` is exactly
    $1.0$ under the constructor's pairing refusal; ``anchors_per_sample`` sits in the band the tile
    phase can produce on training and at the dense span on validation, where a single phase would be
    phase-biased and there is no gradient to save. A row outside either band means the geometry
    broke rather than that the model learned."""
    module = task(model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE))
    perturb_posterior(module.orig_model)
    model = module.orig_model
    span = model.geometry.t_valid - model.warmup_period

    _loss, train_metrics = module.compute_loss_and_metrics(make_stub_batch(), 0, "train")
    _loss, val_metrics = module.compute_loss_and_metrics(make_stub_batch(), 0, "val")

    assert float(train_metrics["target_warm_frac"]) == 1.0
    assert float(val_metrics["target_warm_frac"]) == 1.0
    fewest = -(-(span - (TINY_STRIDE - 1)) // TINY_STRIDE)
    most = -(-span // TINY_STRIDE)
    assert fewest <= float(train_metrics["anchors_per_sample"]) <= most
    assert float(val_metrics["anchors_per_sample"]) == float(span)
