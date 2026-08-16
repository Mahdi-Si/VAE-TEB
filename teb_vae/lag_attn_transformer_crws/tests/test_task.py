r"""The task is one diamond: the causal-input cell on one side, the step-granular ramp on the other.

Two models are only comparable if they optimise the same thing, and here that is not a claim about
two copies of an objective agreeing -- it is the same code, reached through both parents at once. So
most of the assertions below are about *absence* and about *identity*: a re-added ``training_step``
silently disables the config-gated loss-spike breaker, a re-added ``_build_raw_target`` gives this
model a second target builder that could drift from the one the comparison models are scored
through, and neither fails anything on its own.

**The diamond is legal because both parents descend from the shared task**, not because their added
members happen to be disjoint. They are today -- everything the causal-input cell adds against
$\{$``build_lr_scheduler``$\}$ -- but that is a fact about today's code, so the linearisation is
pinned as a list of class names and each behaviour is asserted against the class the design names.
A future member defined on both sides would resolve to the causal side by order alone, silently.
"""
from __future__ import annotations

import inspect

import pytest
import torch
from torch.optim.lr_scheduler import LambdaLR

from teb_vae.lag_attn_cfs.task import DENSE_STAGES, SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask
from teb_vae.lag_attn_transformer_crws.task import SeqVaeLagAttnTrfCrwsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from train.pl_model_base import LightningModelBase

from .conftest import TINY_STRIDE, make_stub_batch, tiny_warmup_kwargs

#: The linearisation the design names, pinned as a list of class names. Both parents derive from the
#: shared task, which is what makes the diamond legal; the *order* is what decides where a member
#: defined on both sides would resolve.
_EXPECTED_MRO = [
    "SeqVaeLagAttnTrfCrwsTask",
    "SeqVaeLagAttnCrwsTask",
    "SeqVaeLagAttnTrfRwsTask",
    "SeqVaeLagAttnRwsTask",
    "LightningModelBase",
]

#: The three readouts this cell adds to the shared metric surface on both stages, and the one that
#: runs on the evaluation stages alone. Three rather than the causal-feature cells' seven: five of
#: theirs partition kept *target* channels, and a raw target has none.
_CAUSAL_METRICS = (
    "anchors_per_sample",
    "source_lag_warmth_frac_st",
    "source_lag_warmth_frac_ph",
)
_CAUSAL_VAL_ONLY_METRICS = ("kld_source_null",)

#: The five the causal-feature cells report and this one must not, because each partitions target
#: channels this target does not have.
_FEATURE_ONLY_METRICS = (
    "target_warm_frac",
    "pred_gap_st",
    "pred_gap_ph",
    "pred_gap_warm_lo",
    "pred_gap_warm_mid",
    "pred_gap_warm_hi",
)


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_class_body_is_empty() -> None:
    """Set equality over ``vars``, not a line count.

    With nothing defined here, the objective, its $\\beta$ schedule, the metric surface, the
    permutation control, the spike-breaker wiring and the checkpoint contract cannot have moved:
    they are the shared task's own code objects, reached through two parents.

    ``_abc_impl`` is written by ``abc`` on every concrete subclass and is not this class's.
    """
    own = {
        name
        for name, value in vars(SeqVaeLagAttnTrfCrwsTask).items()
        if callable(value) or isinstance(value, (property, classmethod, staticmethod))
    }

    assert own == set()
    assert {
        name for name in vars(SeqVaeLagAttnTrfCrwsTask) if not name.startswith("__")
    } <= {"_abc_impl"}
    assert "__init__" not in vars(SeqVaeLagAttnTrfCrwsTask)


def test_the_diamond_linearises_the_way_the_design_names_it() -> None:
    names = [cls.__name__ for cls in SeqVaeLagAttnTrfCrwsTask.__mro__]

    assert names[: len(_EXPECTED_MRO)] == _EXPECTED_MRO


def test_the_two_branches_are_disjoint_today_and_the_test_says_which_way_a_collision_falls() -> None:
    """The diamond's one silent hazard, made explicit. If a member ever appears on both sides it
    resolves to the causal parent, and this test is what turns that from an accident into a decision
    someone had to make."""
    # ``_abc_impl`` is written by ``abc`` onto every concrete subclass, so it is on both sides for a
    # reason that has nothing to do with either design.
    causal = {
        name
        for name in vars(SeqVaeLagAttnCrwsTask)
        if not name.startswith("__") and name != "_abc_impl"
    }
    transformer = {
        name
        for name in vars(SeqVaeLagAttnTrfRwsTask)
        if not name.startswith("__") and name != "_abc_impl"
    }

    assert causal & transformer == set()
    for name in causal | transformer:
        owner = SeqVaeLagAttnCrwsTask if name in causal else SeqVaeLagAttnTrfRwsTask
        assert getattr(SeqVaeLagAttnTrfCrwsTask, name) is getattr(owner, name), name


@pytest.mark.parametrize(
    "method",
    ["training_step", "validation_step", "test_step", "forward", "configure_optimizers",
     "on_save_checkpoint", "setup", "_build_raw_target", "_resolve_beta", "_should_run_perm",
     "_sync_perm_decision"],
)
def test_the_inherited_machinery_is_not_taken_back(method) -> None:
    """``training_step`` matters most: the framework's version runs the config-gated spike breaker,
    and a subclass defining its own silently disables it."""
    assert method not in vars(SeqVaeLagAttnTrfCrwsTask), (
        f"{method} is overridden; the inherited implementation is the seam this model uses"
    )


# ---------------------------------------------------------------------------------------
# Where each half comes from
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "member",
    ["anchor_phase", "_phase_field", "resolve_anchor_geometry", "_build_forward_inputs",
     "_mu_gap_rms", "_added_metrics", "compute_loss_and_metrics", "forecast_rows",
     "input_stream_panels", "input_budget_figure", "_stage", "warmup_budget", "__init__"],
)
def test_the_tiling_and_the_page_seams_come_from_the_causal_parent(member) -> None:
    """Identity, not equality. ``__init__`` is on the list because the run seed the tile phase is
    derived from reaches ``save_hyperparameters`` through it, and a resumed run that did not know the
    seed would silently re-tile every segment. ``forecast_rows`` is on it because the page's rows
    walk this cell's ``anchor_index`` and name no encoder, so both cells of this row must draw one
    picture rather than two.

    Membership of ``vars`` rather than a truthiness check, because two of the thirteen are class
    attributes whose declared value is falsy -- ``warmup_budget`` is ``None`` until a driver hands
    the task a resolved budget -- and a truthy guard would silently stop asserting them.
    """
    assert member in vars(SeqVaeLagAttnCrwsTask), f"{member} moved off the parent"
    assert getattr(SeqVaeLagAttnTrfCrwsTask, member) is getattr(SeqVaeLagAttnCrwsTask, member)


def test_the_raw_target_builder_comes_from_the_shared_ancestor() -> None:
    """The target is the raw FHR future, exactly the raw-signal siblings'. Neither parent redefines
    the builder, so a second copy anywhere in this chain could only drift from the one every
    comparison model is scored through."""
    assert "_build_raw_target" not in vars(SeqVaeLagAttnCrwsTask)
    assert "_build_raw_target" not in vars(SeqVaeLagAttnTrfRwsTask)
    assert (
        SeqVaeLagAttnTrfCrwsTask._build_raw_target
        is SeqVaeLagAttnRwsTask.__dict__["_build_raw_target"]
    )


def test_the_phase_machinery_is_the_causal_feature_cells_object_two_hops_up() -> None:
    """The binding chain, end to end. The conv-LSTM cell of this row binds the phase derivation from
    the causal-feature cell rather than copying it, so the two-subprocess hash stability that suite
    proved covers this model too -- and this test is what says the chain did not quietly grow a
    second implementation in the middle."""
    for member in ("anchor_phase", "resolve_anchor_geometry", "_build_forward_inputs"):
        assert getattr(SeqVaeLagAttnTrfCrwsTask, member) is getattr(
            SeqVaeLagAttnCfsTask, member
        ), member


def test_the_step_granular_ramp_comes_from_the_conv_transformer_parent() -> None:
    """The one member that half of the diamond exists for. The causal parent does not define it at
    all, so lookup passes through -- and resolving to the shared task instead would silently drop the
    ramp a pre-normalised attention stack needs in its first few hundred updates."""
    assert "build_lr_scheduler" not in vars(SeqVaeLagAttnCrwsTask)
    assert "build_lr_scheduler" in vars(SeqVaeLagAttnTrfRwsTask)
    assert (
        SeqVaeLagAttnTrfCrwsTask.build_lr_scheduler is SeqVaeLagAttnTrfRwsTask.build_lr_scheduler
    )
    assert (
        SeqVaeLagAttnTrfCrwsTask.build_lr_scheduler is not SeqVaeLagAttnRwsTask.build_lr_scheduler
    )


def test_the_objective_is_the_shared_ancestors(task) -> None:
    """The reason the whole chain exists: what a run optimises is one code object, reached from eight
    model classes."""
    assert issubclass(SeqVaeLagAttnTrfCrwsTask, LightningModelBase)
    module = task()

    assert type(module).__mro__[: len(_EXPECTED_MRO)] == tuple(
        cls for cls in SeqVaeLagAttnTrfCrwsTask.__mro__[: len(_EXPECTED_MRO)]
    )


def test_the_constructor_signature_is_the_causal_parents() -> None:
    """No ``__init__`` here, so this is the causal parent's -- which takes the run seed the tile
    phase is keyed on and is what makes a *resumed* run reproduce the phases it was drawing."""
    signature = inspect.signature(SeqVaeLagAttnTrfCrwsTask.__init__)

    assert signature == inspect.signature(SeqVaeLagAttnCrwsTask.__init__)
    assert "seed" in signature.parameters


# ---------------------------------------------------------------------------------------
# The learning-rate ramp, reached through the diamond
# ---------------------------------------------------------------------------------------
class _FakeTrainer:
    """The two properties ``build_lr_scheduler`` reads, and nothing else.

    ``estimated_stepping_batches`` is the optimizer-step total for the **whole run**, not per epoch.
    It is typed ``float`` because an unlimited run is reported as infinity.
    """

    def __init__(self, estimated_stepping_batches: float, max_epochs: int) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs


def test_the_ramp_delegates_when_no_step_warmup_is_configured(task) -> None:
    """At $0$ -- what a config that never sets the key resolves to -- the inherited
    epoch-granularity path stays reachable and costs nothing."""
    module = task()
    setattr(module.hparams, "lr_warmup_steps", 0)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    assert module.build_lr_scheduler(optimizer) is None


def test_the_ramp_is_a_step_granular_lambda_when_configured(task) -> None:
    """One ``LambdaLR`` at ``interval: 'step'``, carrying both the ramp and the milestone decay: a
    ramp measured in steps but stepped once per epoch would take ``lr_warmup_steps`` *epochs* to
    complete, silently.

    Reached here through the diamond rather than through the conv-Transformer task directly, which
    is the whole point: the causal parent comes first in the linearisation, so a member it grew would
    shadow this one.
    """
    module = task()
    setattr(module.hparams, "lr_warmup_steps", 100)
    setattr(module.hparams, "lr_milestones", [])
    module.trainer = _FakeTrainer(1000.0, 10)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)

    schedule = module.build_lr_scheduler(optimizer)

    assert isinstance(schedule, dict)
    assert isinstance(schedule["scheduler"], LambdaLR)
    assert schedule["interval"] == "step"
    factor = schedule["scheduler"].lr_lambdas[0]
    assert factor(0) == pytest.approx(0.01)
    assert factor(99) == pytest.approx(1.0)
    assert factor(500) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------
# The anchor geometry the stage decides
# ---------------------------------------------------------------------------------------
def test_the_forward_takes_five_arguments(task, stub_batch) -> None:
    """Five, not the family's three: the two target-stream blocks, the source stream, the per-sample
    tile phase and the stride."""
    module = task()
    module._stage = "train"

    inputs = module._build_forward_inputs(stub_batch)

    assert len(inputs) == 5
    phase, stride = inputs[3], inputs[4]
    assert stride == TINY_STRIDE
    assert isinstance(phase, torch.Tensor) and phase.shape == (stub_batch.fhr_st.shape[0],)
    assert bool(((phase >= 0) & (phase < TINY_STRIDE)).all())


@pytest.mark.parametrize("stage", DENSE_STAGES)
def test_both_evaluation_stages_decode_densely(task, stub_batch, stage) -> None:
    """A single phase is deterministic but *phase-biased*, and there is no gradient at either stage,
    so neither the redundancy argument nor the memory argument that motivates the tiling applies."""
    module = task()

    phase, stride = module.resolve_anchor_geometry(stage, stub_batch)

    assert (phase, stride) == (0, 1)


def test_the_metric_surface_carries_the_three_causal_readouts(task, perturb_posterior) -> None:
    """Emitted through the diamond exactly as they are on the conv-LSTM cell of this row: two from
    the input mixin's own merge, plus the geometry guard, plus ``kld_source_null`` from the task's
    ``_added_metrics``, which needs the source stream the forward dict does not carry."""
    module = task()
    perturb_posterior(module.orig_model)
    batch = make_stub_batch()

    _loss, train_metrics = module.compute_loss_and_metrics(batch, 0, "train")
    _loss, val_metrics = module.compute_loss_and_metrics(batch, 0, "val")

    for name in _CAUSAL_METRICS:
        assert name in train_metrics, name
        assert name in val_metrics, name
    for name in _CAUSAL_VAL_ONLY_METRICS:
        assert name in val_metrics, name
        # Absent, never zero-filled: the framework's epoch value is the mean over the steps that
        # reported it, so a zero placeholder would scale the aggregate toward nothing.
        assert name not in train_metrics, name


@pytest.mark.parametrize("name", _FEATURE_ONLY_METRICS)
def test_no_target_channel_readout_is_emitted(task, perturb_posterior, name) -> None:
    """Each of the five partitions kept target channels, and this target's last axis counts raw
    samples -- which have no warm-up, no filter and no order to rank by. A column emitted here would
    be a number with no referent rather than a wrong one."""
    module = task()
    perturb_posterior(module.orig_model)

    _loss, metrics = module.compute_loss_and_metrics(make_stub_batch(), 0, "val")

    assert name not in metrics


def test_the_geometry_guard_reads_its_derived_value(task, perturb_posterior) -> None:
    """One of the three is a guard rather than a result: ``anchors_per_sample`` sits at its
    geometry-derived value, and a row outside that band means the geometry broke rather than that the
    model learned something."""
    module = task(model_kwargs=tiny_warmup_kwargs(anchor_stride=TINY_STRIDE))
    perturb_posterior(module.orig_model)
    model = module.orig_model
    span = model.geometry.t_valid - model.warmup_period

    _loss, train_metrics = module.compute_loss_and_metrics(make_stub_batch(), 0, "train")
    _loss, val_metrics = module.compute_loss_and_metrics(make_stub_batch(), 0, "val")

    fewest = -(-(span - (TINY_STRIDE - 1)) // TINY_STRIDE)
    most = -(-span // TINY_STRIDE)
    assert fewest <= float(train_metrics["anchors_per_sample"]) <= most
    assert float(val_metrics["anchors_per_sample"]) == float(span)
