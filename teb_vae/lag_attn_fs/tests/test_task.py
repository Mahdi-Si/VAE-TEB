r"""The task is the sibling's task plus one re-pointed builder, and that is the whole point.

Two models are only comparable if they optimise the same thing. Here that is the same code: the
loss assembly, the $\beta$ schedule, the metric surface, the permutation control, the spike-breaker
wiring and the checkpoint contract are all inherited unmodified and pinned in the sibling suites.
So the first assertion is about *absence*: a re-added ``training_step`` silently disables the
config-gated loss-spike breaker, a re-added ``compute_loss_and_metrics`` takes back the permutation
control and the ``main_loss`` name the breaker watches, and neither fails anything on its own.

The one override is checked for what it returns -- the two stored target blocks concatenated in the
declared order, the tensor the loss is actually computed against -- and for the input checks it
routes through: a missing weight, a wrong joint width, and a shard whose blocks split somewhere
other than where the net's per-block gap columns assume. A batch of one, which a DDP rank can
receive and the permutation control refuses, must still train.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask

from .conftest import TASK_HPARAMS, make_stub_batch

#: Callables the subclass may define. A set rather than a count, following the sibling suite: a
#: count passes a subclass that took back ``training_step`` in 140 lines while dropping something
#: else. ``forecast_rows`` is deliberately **not** here and does not belong here: the shared
#: plotting callback resolves it with ``getattr(..., None)``, so an attribute of that name
#: overrides nothing and adding one is not a second override of anything. It is also a
#: ``property``, which is not itself callable, so the set below excludes it without a special
#: case.
_OWN_CALLABLES = {"_build_raw_target"}


# ---------------------------------------------------------------------------------------
# What the subclass is
# ---------------------------------------------------------------------------------------
def test_the_subclass_adds_exactly_one_method():
    """A second override is a second thing that can diverge from the objective being shared. The
    constructor is checked too: a narrowed ``__init__`` would bypass the base's
    ``save_hyperparameters`` and the breaker counters."""
    own = {
        name
        for name, value in vars(SeqVaeLagAttnFsTask).items()
        if callable(value) and not name.startswith("__")
    }

    assert own == _OWN_CALLABLES
    assert "__init__" not in vars(SeqVaeLagAttnFsTask)


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


def test_the_forward_inputs_are_unchanged_by_the_override(task, stub_batch):
    """The net is fed the two blocks separately and scored against their concatenation. Both come
    off the same builder, so they cannot disagree about which tensor the model saw."""
    module = task()

    inputs = module._build_forward_inputs(stub_batch)
    target, _ = module._build_raw_target(stub_batch)

    assert len(inputs) == 3
    assert torch.equal(torch.cat([inputs[0], inputs[1]], dim=-1), target)


def test_a_batch_of_one_still_trains(task, perturb_posterior):
    """A rank can receive a batch of one under DDP. The permutation control refuses it; the step
    must not."""
    module = task()
    perturb_posterior(module.orig_model)

    loss, metrics = module.compute_loss_and_metrics(make_stub_batch(1), 0, "train")

    assert torch.isfinite(loss)
    assert "nll_shuffled_block" not in metrics
