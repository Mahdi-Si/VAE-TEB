r"""The task is two parents and nothing of its own, and the diamond is what has to be asserted.

Every behaviour this class has is inherited, so the only defects it can carry are *resolution*
defects, and none of them raises. ``_build_raw_target`` resolving to the shared ancestor scores the
net against the raw FHR trace while every column keeps its name; ``build_lr_scheduler`` resolving
there leaves the configured step warm-up with no ramp attached anywhere; ``forecast_rows`` resolving
there draws the raw page. Each is asserted by *behaviour* through this task, so a reordered diamond
fails rather than trains something else, and the empty class body is asserted directly.

What is genuinely new here -- as opposed to inherited from a suite that already ran it -- is the
pairing: the metric surface against the conv-LSTM feature model's, the breaker's comparison metric
against what this task emits, and the permutation control, duck-typed on model attributes, running
through this task on this net.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_fs.sample_page import feature_forecast_rows
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_transformer_fs.task import SeqVaeLagAttnTrfFsTask

from .conftest import SEQ_LEN, TASK_HPARAMS, make_stub_batch

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

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


def _feature_sibling_task():
    """The conv-LSTM feature model wrapped in its own task, at that suite's tiny geometry.

    Built from the *feature* suite's keyword set rather than from this one's: the two constructors'
    schemas differ, so one set cannot build both models.

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


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_task_defines_nothing_at_all():
    """A re-added ``training_step`` would silently disable the config-gated spike breaker, a re-added
    ``compute_loss_and_metrics`` would take back the permutation control and the ``main_loss`` name,
    and a constructor of its own would be a second keyword schema for one shared objective.

    ``_abc_impl`` is excluded because it is not the class body's: ``ABCMeta`` writes that cache into
    the ``__dict__`` of every subclass it creates.
    """
    assert [
        name
        for name in vars(SeqVaeLagAttnTrfFsTask)
        if not name.startswith("__") and name != "_abc_impl"
    ] == []
    assert "__init__" not in vars(SeqVaeLagAttnTrfFsTask)


# ---------------------------------------------------------------------------------------
# The feature parent's half of the diamond, by behaviour
# ---------------------------------------------------------------------------------------
def test_the_target_is_the_two_blocks_concatenated_in_the_declared_order(task, patterned_batch):
    """Asserted against the planted pattern rather than against a shape: a transposed or reversed
    concatenation produces a correctly shaped tensor, and only a value check says which channel ended
    up where. The declared order is what the reach budget's keep-index is positional into."""
    module = task()
    split = patterned_batch.fhr_st.shape[-1]

    target, weight = module._build_raw_target(patterned_batch)

    assert target.shape[-1] == module.orig_model.c_y
    assert torch.equal(target[..., :split], patterned_batch.fhr_st)
    assert torch.equal(target[..., split:], patterned_batch.fhr_ph)
    assert weight is patterned_batch.weight


def test_the_page_seam_binds_this_nets_channel_facts(task, shipped_gated):
    """The keep-index and the block split are the two things the page is handed rather than derives,
    and both are read off *this* net's own gate -- which the conv-Transformer base builds at its own
    construction site. The ungated arm has no gate to read an index off, so it binds ``None``."""
    module = task(model_kwargs=shipped_gated)
    guarded = module.forecast_rows
    unguarded = task().forecast_rows

    assert guarded.func is feature_forecast_rows
    assert set(guarded.keywords) == {"keep_index", "block_split"}
    assert list(guarded.keywords["keep_index"]) == list(shipped_gated["target_keep_index"])
    assert guarded.keywords["block_split"] == module.orig_model.TARGET_BLOCK_SPLIT
    assert unguarded.keywords["keep_index"] is None


# ---------------------------------------------------------------------------------------
# The conv-Transformer parent's half of the diamond, by behaviour
# ---------------------------------------------------------------------------------------
def test_the_step_warmup_is_attached_at_step_granularity(task):
    """A ramp measured in optimizer steps but attached at ``interval: "epoch"`` takes
    ``lr_warmup_steps`` *epochs* -- silently. The ramp's arithmetic is inherited code and is pinned in
    the sibling's own suite."""
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
def test_the_metric_set_is_the_feature_siblings_exactly(task, stub_batch, perturb_posterior):
    """Driven from both tasks' real metrics dicts rather than from a list. This model differs from the
    feature sibling in its encoders alone, so a metric either of them emitted and the other did not
    would be a column the shared tracked list cannot collect."""
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


def test_the_configured_comparison_metric_is_one_the_task_emits(
    task, stub_batch, perturb_posterior
):
    """``comparison_metric`` falls back to the returned loss silently when the named metric is
    missing, so the shipped breaker block must name something this task genuinely emits."""
    breaker = load_config(str(_CONFIG))["advanced_config"]["spike_breaker"]
    module = task()
    perturb_posterior(module.orig_model)

    _, metrics = module.compute_loss_and_metrics(stub_batch, 0, "train")

    assert breaker["comparison_metric"] in metrics


# ---------------------------------------------------------------------------------------
# The permutation control, gated through this task
# ---------------------------------------------------------------------------------------
def test_the_permutation_control_runs_on_a_validation_batch_of_two_or_more(
    task, perturb_posterior
):
    """The control is duck-typed on model attributes rather than on a model class, so a model missing
    one of them would simply stop producing the three specificity readouts while every other column
    looked healthy.

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
