r"""The evaluation readouts, and the parity that keeps them tied to the training objective.

The load-bearing test here is the parity one. Everything else in the evaluation package is
plumbing around two numbers -- $D_{\mathrm{base}}$ and $D_{\mathrm{full}}$ -- and the whole
exercise is worthless if those are not the quantities the training loop optimised. So the
per-sample readouts are recombined into the exact anchor-weighted total the loss reduces to, and
compared against what the task itself reports on the same batch.

The second theme is aggregation. Anchors are not independent samples: consecutive anchors'
forecast windows overlap in $29$ of their $30$ horizon steps, and one long recording holds
hundreds of them. Averaging per recording and then across recordings is what stops a headline
number from being dominated by whichever recording happens to be longest.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.eval.metrics import (
    Aggregate,
    BatchReadout,
    aggregate_by_recording,
    batch_guids,
    batch_size_of,
    evaluate,
    evaluate_batch,
    latent_health,
    lag_summary,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS

from .conftest import BATCH, TINY_KWARGS, make_stub_batch


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior.

    Load-bearing: at initialisation the delta heads are zero, so the posterior *is* the prior,
    every KL is exactly zero, base and full are bitwise identical, and every assertion below
    would pass on a model that is completely wrong.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    return module


class _OneBatchLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


# =============================================================================
# Parity with the training objective
# =============================================================================
def test_the_per_sample_readouts_recombine_into_the_training_loss(trained_task, stub_batch):
    r"""The anchor-weighted total of the per-sample values *is* the loss's own reduction,
    $\sum_b \sum_t / \sum_b n_b$. If these drift, the evaluation is scoring something the
    objective never saw.

    Both sides are seeded identically because each runs its own forward, and the reparameterised
    latent is stochastic: the reconstruction terms are a function of the draw, so an unseeded
    comparison would be comparing two different samples of the same quantity and could only ever
    be asserted to a loose tolerance -- which is exactly the tolerance a real drift would hide
    inside.
    """
    torch.manual_seed(4)
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    torch.manual_seed(4)
    _loss, metrics = trained_task.compute_loss_and_metrics(stub_batch, 0, "val")

    weights = readout.n_anchors
    for name in ("nll_base_block", "nll_full_block"):
        recombined = float(
            (readout.columns[name] * weights).sum() / weights.sum().clamp_min(1.0)
        )
        assert recombined == pytest.approx(float(metrics[name]), rel=1e-5)


def test_the_kl_readout_recombines_into_the_training_kl(trained_task, stub_batch):
    """Same check on the other term, whose mask is a different one -- the KL's anchor support is
    not the reconstruction's."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    _loss, metrics = trained_task.compute_loss_and_metrics(stub_batch, 0, "val")

    support = kl_support_counts(trained_task, stub_batch)
    recombined = float(
        (readout.columns["source_conditioned_kl_raw"] * support).sum() / support.sum()
    )
    assert recombined == pytest.approx(float(metrics["source_conditioned_kl_raw"]), rel=1e-5)


def kl_support_counts(module, batch) -> torch.Tensor:
    """Masked anchor counts per sample, from the same masks the loss uses."""
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

    model = module.orig_model
    forecast, _ = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    return kl_mask(forecast, model.geometry).sum(dim=1)


def test_the_predictive_gap_is_the_difference_of_the_two_scores(trained_task, stub_batch):
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert torch.allclose(
        readout.columns["pred_gap"],
        readout.columns["nll_base_block"] - readout.columns["nll_full_block"],
    )


# =============================================================================
# What a batch produces
# =============================================================================
def test_every_branch_is_scored_when_the_batch_can_be_deranged(trained_task, stub_batch):
    readout = evaluate_batch(trained_task, stub_batch, num_samples=2)

    for name in ("base", "full", "shuffled", "base_shuffled_mu"):
        assert f"mc_nll_{name}_block" in readout.columns
        assert readout.columns[f"mc_nll_{name}_block"].shape == (BATCH,)


def test_a_batch_too_small_to_derange_scores_only_the_two_real_branches(trained_task):
    """A one-sample batch has no stranger to borrow a source from. Producing the columns anyway,
    filled with the sample's own values, would report the negative control as passing."""
    readout = evaluate_batch(trained_task, make_stub_batch(batch_size=1), num_samples=1)

    assert "mc_nll_base_block" in readout.columns
    assert "mc_nll_shuffled_block" not in readout.columns


def test_the_per_dimension_kl_and_the_lag_profiles_have_the_model_shapes(
    trained_task, stub_batch
):
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    model = trained_task.orig_model

    assert readout.kld_per_dim.shape == (model.d_z,)
    assert readout.lag_profile.shape == (model.max_lag + 1,)
    assert readout.attention_profile.shape == (model.max_lag + 1,)


def test_the_lag_map_profile_sums_to_the_total_kl(trained_task, stub_batch):
    r"""The identity the head-structured latent buys: $\sum_\ell \widetilde K_{t,\ell} = K_t$,
    exactly rather than in expectation, because the attention probabilities carry no dropout."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    support = kl_support_counts(trained_task, stub_batch)
    kl_total = float(
        (readout.columns["source_conditioned_kl_raw"] * support).sum() / support.sum()
    )

    assert float(readout.lag_profile.sum()) == pytest.approx(kl_total, rel=1e-5)


# =============================================================================
# Aggregation
# =============================================================================
def _readout(guids, values, anchors) -> BatchReadout:
    """A hand-built readout carrying one column, for the aggregation arithmetic."""
    width = 3
    return BatchReadout(
        guids=list(guids),
        columns={"score": torch.tensor(values, dtype=torch.float32)},
        n_anchors=torch.tensor(anchors, dtype=torch.float32),
        kld_per_dim=torch.zeros(width),
        lag_profile=torch.zeros(width),
        attention_profile=torch.zeros(width),
    )


def test_a_segment_that_scored_no_anchors_is_excluded_rather_than_counted_as_zero():
    """Its per-sample mean divides by a denominator clamped to 1, so an empty numerator reads
    as exactly 0.0 -- a fabricated score, not a small one. Averaged into a summed-480-sample
    block figure it would drag the headline toward zero and shrink pred_gap silently.
    """
    aggregate = aggregate_by_recording(
        [_readout(["a", "a"], [4.0, 0.0], [10, 0]), _readout(["b"], [6.0], [10])]
    )

    assert aggregate.per_recording["a"]["score"] == pytest.approx(4.0)
    assert aggregate.overall["score"] == pytest.approx(5.0)
    assert aggregate.n_samples == 3, "every segment seen is still counted"
    assert aggregate.n_samples_without_anchors == 1, "and the excluded one is reported"


def test_a_recording_with_no_scored_segments_does_not_reach_the_headline():
    """Excluding the samples must also drop the recording, not leave it as an empty bucket
    that contributes a zero row to the across-recording mean."""
    aggregate = aggregate_by_recording(
        [_readout(["a", "b"], [4.0, 0.0], [10, 0])]
    )

    assert set(aggregate.per_recording) == {"a"}
    assert aggregate.overall["score"] == pytest.approx(4.0)
    assert aggregate.n_samples_without_anchors == 1


def test_segments_of_one_recording_count_once(trained_task):
    """Three segments of recording A and one of B: A must not outvote B three to one."""
    aggregate = aggregate_by_recording(
        [_readout(["a", "a"], [1.0, 3.0], [10, 10]), _readout(["a", "b"], [2.0, 10.0], [10, 10])]
    )

    assert aggregate.n_recordings == 2
    assert aggregate.per_recording["a"]["score"] == pytest.approx(2.0)
    assert aggregate.per_recording["b"]["score"] == pytest.approx(10.0)
    # The recording mean, not the segment mean, which would be 4.0.
    assert aggregate.overall["score"] == pytest.approx(6.0)


def test_an_absent_identifier_still_aggregates(trained_task, stub_batch):
    """The stub batch carries no ``guid``; every sample lands in one 'unknown' recording rather
    than being dropped."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    aggregate = aggregate_by_recording([readout])

    assert list(aggregate.per_recording) == ["unknown"]
    assert aggregate.n_samples == BATCH


def test_inconsistent_columns_are_refused_rather_than_averaged(trained_task):
    """A last batch too small to derange produces a different column set; averaging it in would
    quietly drop the negative control from the headline."""
    full = evaluate_batch(trained_task, make_stub_batch(batch_size=2), num_samples=1)
    partial = evaluate_batch(trained_task, make_stub_batch(batch_size=1), num_samples=1)

    with pytest.raises(ValueError, match="different readout columns"):
        aggregate_by_recording([full, partial])


def test_no_batches_aggregates_to_nothing_rather_than_raising():
    aggregate = aggregate_by_recording([])

    assert aggregate.n_recordings == 0 and aggregate.overall == {}


# =============================================================================
# Latent health and the lag report
# =============================================================================
def test_latent_health_counts_dimensions_against_the_training_threshold():
    """The same threshold the training metric ``kld_active_frac`` reports against; a second copy
    would be a second threshold."""
    aggregate = Aggregate(kld_per_dim=[1.0, 0.5, KLD_ACTIVE_EPS / 2.0, 0.0])

    health = latent_health(aggregate)

    assert health["d_z"] == 4
    assert health["active_dims"] == 2
    assert health["active_frac"] == pytest.approx(0.5)
    assert health["activity_threshold_nats"] == KLD_ACTIVE_EPS


def test_latent_health_reports_the_top_dimensions_share():
    """"Collapsed into one dimension" stated as a number rather than left to be eyeballed off
    the distribution."""
    health = latent_health(Aggregate(kld_per_dim=[9.0, 0.5, 0.5]))

    assert health["top_dimension_share"] == pytest.approx(0.9)
    assert health["kld_per_dimension"] == [9.0, 0.5, 0.5]


def test_the_lag_report_carries_both_seconds_quantities():
    """Both, and named: the compensated one is the residual physiological lag, the other maps
    back to the uncorrected sensor files, and they differ by the 20 s already removed."""
    aggregate = Aggregate(lag_profile=[0.1, 0.9, 0.2], attention_profile=[0.5, 0.2, 0.3])

    summary = lag_summary(aggregate)

    assert summary["kl_argmax_lag_step"] == 1
    assert summary["kl_lag_compensated_seconds"] == pytest.approx(4.0)
    assert summary["kl_lag_original_sensor_seconds"] == pytest.approx(24.0)
    assert summary["attention_argmax_lag_step"] == 0


def test_the_lag_report_folds_in_a_configured_input_delay():
    summary = lag_summary(Aggregate(lag_profile=[0.1, 0.9], attention_profile=[1.0, 0.0]),
                          delay_steps=30)

    assert summary["delay_steps"] == 30
    assert summary["kl_lag_compensated_seconds"] == pytest.approx(4.0 * (1 + 30))


def test_the_lag_report_is_empty_when_nothing_was_collected():
    assert lag_summary(Aggregate()) == {}


# =============================================================================
# Batch helpers
# =============================================================================
def test_the_batch_size_comes_from_a_tensor_field(stub_batch):
    assert batch_size_of(stub_batch) == BATCH
    assert batch_size_of({"fhr_st": torch.zeros(5, 4, 43)}) == 5
    assert batch_size_of(object()) == 0


def test_guids_are_read_as_strings_and_default_to_unknown(stub_batch):
    """``guid`` survives collation as a ``list[str]``, never a tensor."""
    assert batch_guids(stub_batch, BATCH) == ["unknown"] * BATCH
    assert batch_guids({"fhr_st": None, "guid": ["a", "b"]}, 2) == ["a", "b"]
    assert batch_guids({"guid": torch.tensor([7, 8])}, 2) == ["7", "8"]


# =============================================================================
# The whole loop
# =============================================================================
def test_the_evaluation_loop_assembles_every_reported_section(trained_task):
    loader = _OneBatchLoader([make_stub_batch(seed=0), make_stub_batch(seed=1)])

    results = evaluate(trained_task, loader, num_samples=2)

    assert results["n_batches"] == 2
    assert results["n_samples"] == 2 * BATCH
    assert results["likelihood"] == TINY_KWARGS.get("likelihood", "gaussian_nll")
    assert set(results) >= {
        "readouts", "latent_health", "lag", "per_recording", "verdicts", "num_mc_samples"
    }
    assert results["readouts"]["pred_gap"] == pytest.approx(
        results["readouts"]["nll_base_block"] - results["readouts"]["nll_full_block"], rel=1e-5
    )


def test_the_evaluation_loop_leaves_the_task_as_it_found_it(trained_task):
    """It flips the module into evaluation mode; a training loop that borrowed it back in
    ``train`` mode would silently start running dropout-free."""
    trained_task.train()

    evaluate(trained_task, _OneBatchLoader([make_stub_batch()]), num_samples=1)

    assert trained_task.training is True


def test_batches_too_small_to_derange_are_skipped_and_counted(trained_task):
    """Skipped rather than partially scored, and *counted*, so a run whose loader hands out
    mostly degenerate batches cannot look like a full evaluation."""
    loader = _OneBatchLoader([make_stub_batch(batch_size=1), make_stub_batch(batch_size=2)])

    results = evaluate(trained_task, loader, num_samples=1)

    assert results["n_batches"] == 1
    assert results["n_batches_skipped_too_small"] == 1
