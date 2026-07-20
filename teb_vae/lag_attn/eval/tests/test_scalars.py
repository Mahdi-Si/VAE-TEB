r"""Tests for the test-split scalar pass and the joint collapse verdict.

The key assertion is :func:`test_the_metric_set_matches_the_tasks_own`. The scalar table exists
to be lined up against a row of a training run's ``metrics.csv``, so a metric added to the task
and not here would leave a gap in a table that still looks complete. Pinning the two tuples
equal turns that into a test failure naming the metric.

The verdict cases are driven by synthetic values rather than by a model, because the point is
the *combination rule* -- collapsed requires both components, and one alone is inconclusive --
and reaching each of the four corners through a real checkpoint would need four checkpoints.

The two parity groups at the bottom exist because "reconciles with training" is this module's
whole claim, and both were false in ways the rest of the suite could not see:

* the $\beta$ cases run against a ``linear_warmup`` checkpoint, because every other fixture pins
  ``beta_schedule: None`` and under that pin the configured constant *is* the effective value --
  so a pipeline reading ``objective.kld_beta`` looked right everywhere;
* the residual cases run at $B > 1$ with **non-binary** weights, because at $B = 1$ a pooled root
  and a mean of per-sample roots are the same number, and with binary weights a mask built
  without the validity factor is the same mask.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn.eval.analyses import scalars as scalars_analysis


# ---------------------------------------------------------------------------
# The metric set
# ---------------------------------------------------------------------------
def test_the_metric_set_matches_the_tasks_own() -> None:
    """Mirrored rather than imported, and pinned equal here.

    ``trainer.py`` pulls in Lightning, and an eval run has no business standing up a training
    framework to read a tuple of strings -- but the two must not drift, so this is where a
    metric added to the task and not to eval fails.
    """
    from teb_vae.lag_attn.trainer import _METRIC_SUFFIXES

    assert scalars_analysis.METRIC_SUFFIXES == tuple(_METRIC_SUFFIXES), (
        "eval's metric list has drifted from trainer.py::_METRIC_SUFFIXES; add the new metric to "
        "analyses/scalars.py, either as a produced value or as a NOT_APPLICABLE entry"
    )


def test_every_suffix_is_either_produced_or_explicitly_not_applicable(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """A suffix must never simply vanish -- an absence nobody notices is the failure mode."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )

    covered = set(summary["metrics"]) | set(summary["not_applicable"])
    missing = set(scalars_analysis.METRIC_SUFFIXES) - covered
    assert not missing, f"{sorted(missing)} appear in neither the table nor the exclusions"


def test_the_permutation_keys_are_omitted_not_zero_filled(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """Zero-filling would scale their means and invert the ordering the control exists to check.

    ``task.py`` records the same reasoning for why it omits rather than zero-fills them on the
    training steps where the control did not run.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )

    for name in ("perm_loss", "kld_shuffled", "feat_loss_shuffled", "shuffle_penalty"):
        assert name not in summary["metrics"], f"{name} was zero-filled rather than omitted"
        assert name in summary["not_applicable"]


def test_the_likelihood_string_never_reaches_the_table(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """A metric consumer coerces a non-numeric value to a clean $0.0$ rather than raising."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )
    assert "likelihood" not in summary["metrics"]
    assert all(isinstance(value, float) for value in summary["metrics"].values())


def test_the_csv_and_the_exclusion_map_are_written(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )
    directory = Path(tmp_path) / "results" / scalars_analysis.ANALYSIS_DIRNAME

    frame = pd.read_csv(directory / "test_metrics.csv")
    assert set(frame.columns) == {"metric", "value"}
    assert set(frame["metric"]) == set(summary["metrics"])

    excluded = json.loads((directory / "not_applicable.json").read_text(encoding="utf-8"))
    assert excluded == summary["not_applicable"]
    assert all(reason for reason in excluded.values()), "every exclusion needs a reason"


def test_pred_gap_is_the_difference_the_task_reports(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    r"""$\mathrm{pred\_gap} = L_{\mathrm{base}} - L_{\mathrm{feat}}$, and the sign matters."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )
    table = summary["metrics"]
    assert table["pred_gap"] == pytest.approx(
        table["base_loss"] - table["feat_loss"], rel=1e-6
    )


# ---------------------------------------------------------------------------
# Pooled reduction
# ---------------------------------------------------------------------------
def test_the_pooled_accumulator_is_not_a_mean_of_means() -> None:
    """A short final batch would otherwise count as much as a full one.

    The split-level number must equal what ``compute_loss`` would report had the whole split
    arrived as a single batch.
    """
    accumulator = scalars_analysis._PooledAccumulator()
    accumulator.add(1.0, weight=90.0)
    accumulator.add(11.0, weight=10.0)
    assert accumulator.mean() == pytest.approx(2.0)  # not 6.0, the mean of the two means


def test_the_pooled_accumulator_ignores_a_non_finite_contribution() -> None:
    """One ``NaN`` batch must not turn the whole split's metric into ``NaN``."""
    accumulator = scalars_analysis._PooledAccumulator()
    accumulator.add(float("nan"), weight=10.0)
    accumulator.add(4.0, weight=10.0)
    assert accumulator.mean() == pytest.approx(4.0)


def test_the_pooled_accumulator_reports_nan_when_nothing_contributed() -> None:
    """Zero would read as a measured value rather than as an absent one."""
    assert np.isnan(scalars_analysis._PooledAccumulator().mean())


# ---------------------------------------------------------------------------
# The joint collapse verdict
# ---------------------------------------------------------------------------
THRESHOLDS = {"pred_gap": 1e-4, "kld_raw": 1e-3}


def test_both_components_near_zero_is_collapsed() -> None:
    """The condition the shipped config actually documents."""
    verdict = scalars_analysis.collapse_verdict(1e-9, 1e-9, THRESHOLDS)
    assert verdict["verdict"] == "collapsed"
    assert verdict["pred_gap_near_zero"] and verdict["kld_raw_near_zero"]


def test_neither_component_near_zero_is_healthy() -> None:
    verdict = scalars_analysis.collapse_verdict(0.5, 0.5, THRESHOLDS)
    assert verdict["verdict"] == "healthy"


def test_a_flat_pred_gap_alone_is_inconclusive_not_collapsed() -> None:
    """A posterior carrying information the decoder does not act on -- a different finding."""
    verdict = scalars_analysis.collapse_verdict(1e-9, 0.5, THRESHOLDS)
    assert verdict["verdict"] == "inconclusive"
    assert "does not act on" in verdict["detail"]


def test_a_flat_kld_alone_is_inconclusive_not_collapsed() -> None:
    """A decoder using a latent the KL under-reports -- likewise worth chasing, not collapse."""
    verdict = scalars_analysis.collapse_verdict(0.5, 1e-9, THRESHOLDS)
    assert verdict["verdict"] == "inconclusive"
    assert "under-reports" in verdict["detail"]


def test_the_verdict_carries_its_components_and_thresholds() -> None:
    """Reported as one verdict *with* the two numbers beside it, not as two loose numbers."""
    verdict = scalars_analysis.collapse_verdict(0.25, 0.75, THRESHOLDS)
    assert verdict["pred_gap"] == pytest.approx(0.25)
    assert verdict["kld_raw"] == pytest.approx(0.75)
    assert verdict["thresholds"] == THRESHOLDS


def test_a_nan_component_is_not_read_as_near_zero() -> None:
    """An unmeasured component must not be mistaken for a measured zero."""
    verdict = scalars_analysis.collapse_verdict(float("nan"), 1e-9, THRESHOLDS)
    assert verdict["pred_gap_near_zero"] is False
    assert verdict["verdict"] == "inconclusive"


def test_the_verdict_is_reachable_from_the_analysis_summary(
    make_eval_runner, tiny_loader, tiny_eval_config, tmp_path
) -> None:
    """It is the run's headline conclusion, so it must be a field rather than a log line."""
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    torch.manual_seed(3)
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config=tiny_eval_config["eval_config"],
        output_dir=tmp_path / "results",
    )
    assert summary["collapse"]["verdict"] in {"collapsed", "inconclusive", "healthy"}
    assert set(summary["collapse"]) >= {"verdict", "detail", "pred_gap", "kld_raw", "thresholds"}


# ---------------------------------------------------------------------------
# The beta schedule
#
# `task.py::_resolve_beta` is the only definition that matters: training logs its result under
# the name `kld_beta` and multiplies `kld_loss` by it inside `total_loss`. These cases pin
# `Objective.effective_beta` to it directly rather than restating the ramp, so a change to the
# schedule's semantics fails here instead of producing two plausible numbers.
# ---------------------------------------------------------------------------
def _objective_and_task(schedule, kld_beta=0.001):
    """An :class:`Objective` and a task carrying the *same* schedule, for a term-by-term compare."""
    from teb_vae.lag_attn.eval.runner import Objective
    from teb_vae.lag_attn.tests.conftest import PROD_HPARAMS

    hparams = dict(PROD_HPARAMS, beta_schedule=schedule, kld_beta=kld_beta)
    return Objective(**hparams), _make_task_with(hparams)


def _make_task_with(hparams):
    """Build a task at the tiny geometry under ``hparams``, for its ``_resolve_beta``."""
    from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask
    from teb_vae.lag_attn.tests.conftest import SHIPPED_KWARGS

    torch.manual_seed(0)
    kwargs = dict(SHIPPED_KWARGS)
    return SeqVaeLagAttnTask(SeqVaeLagAttn(**kwargs), lr=1e-3, model_kwargs=kwargs, **hparams)


#: Epochs spanning the ramp: before it, inside it, at the join, and long past it. The join is the
#: case a ``<`` / ``<=`` slip would move, and "long past" is where the shipped run actually sits.
SCHEDULE_EPOCHS = (0, 1, 17, 49, 50, 51, 412)


@pytest.mark.parametrize("epoch", SCHEDULE_EPOCHS)
def test_effective_beta_matches_the_tasks_resolver_under_a_warmup_schedule(epoch) -> None:
    r"""The parity that makes the reported $\beta$ meaningful, epoch by epoch."""
    from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_BETA_SCHEDULE

    objective, task = _objective_and_task(dict(SHIPPED_BETA_SCHEDULE))
    assert objective.effective_beta(epoch) == pytest.approx(task._resolve_beta(epoch))


def test_a_warmup_schedule_is_not_the_configured_constant() -> None:
    r"""The gap the old code shipped: the shipped ramp ends $100\times$ above the fallback."""
    from teb_vae.lag_attn.eval.tests.conftest import (
        SHIPPED_BETA_SCHEDULE,
        SHIPPED_FALLBACK_KLD_BETA,
    )

    objective, _ = _objective_and_task(dict(SHIPPED_BETA_SCHEDULE))
    assert objective.effective_beta(60) == pytest.approx(0.1)
    assert objective.effective_beta(60) == pytest.approx(100.0 * SHIPPED_FALLBACK_KLD_BETA)
    assert objective.kld_beta == pytest.approx(SHIPPED_FALLBACK_KLD_BETA)


@pytest.mark.parametrize(
    "schedule",
    [
        None,
        {"kind": "constant"},
        {"kind": "constant", "value": 0.25},
        {"kind": "linear_warmup", "start": 0.5, "end": 0.9, "warmup_epochs": 0},
    ],
)
@pytest.mark.parametrize("epoch", (0, 50))
def test_effective_beta_matches_the_tasks_resolver_for_every_supported_kind(schedule, epoch) -> None:
    """Including the fallbacks: no schedule, a ``constant`` with and without a ``value``.

    The ``warmup_epochs: 0`` case is the division the ramp would otherwise perform by zero; both
    sides must short-circuit to ``end`` rather than one raising.
    """
    objective, task = _objective_and_task(schedule)
    assert objective.effective_beta(epoch) == pytest.approx(task._resolve_beta(epoch))


def test_an_unknown_schedule_kind_raises_rather_than_falling_back() -> None:
    """As it does in training. A quiet fallback would report a $\\beta$ the run never used."""
    objective, task = _objective_and_task({"kind": "cosine"})
    with pytest.raises(ValueError, match="cosine"):
        objective.effective_beta(10)
    with pytest.raises(ValueError, match="cosine"):
        task._resolve_beta(10)


def test_the_epoch_comes_from_the_checkpoint_not_from_a_config(warmup_checkpoint, tmp_path) -> None:
    """``Objective.from_checkpoint`` reads the blob's own top-level ``epoch``."""
    from teb_vae.lag_attn.eval.runner import EvalRunner

    runner = EvalRunner.from_checkpoint(warmup_checkpoint(60), tmp_path / "run")
    assert runner.objective.train_epoch == 60
    assert runner.objective.effective_beta() == pytest.approx(0.1)


def test_a_scheduled_beta_with_no_epoch_refuses_to_guess() -> None:
    """Reading a missing epoch as $0$ would report the ramp's start for any checkpoint."""
    from teb_vae.lag_attn.eval.tests.conftest import SHIPPED_BETA_SCHEDULE

    objective, _ = _objective_and_task(dict(SHIPPED_BETA_SCHEDULE))
    assert objective.train_epoch is None
    with pytest.raises(RuntimeError, match="epoch"):
        objective.effective_beta()


def test_the_reported_kld_beta_is_the_schedules_value(
    warmup_checkpoint, tiny_loader, tmp_path
) -> None:
    r"""The table's ``kld_beta`` must be what ``task.py`` logs under that name.

    Against the old code this reads $0.001$ -- the configured fallback -- rather than the
    schedule's $0.1$, and nothing else in the table changes to reveal it.
    """
    from teb_vae.lag_attn.eval.runner import EvalRunner

    runner = EvalRunner.from_checkpoint(warmup_checkpoint(120), tmp_path / "run")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config={}, output_dir=tmp_path / "results"
    )

    expected = _make_task_with(
        dict(runner.objective.as_dict())
    )._resolve_beta(runner.objective.train_epoch)
    assert summary["metrics"]["kld_beta"] == pytest.approx(expected)
    assert summary["metrics"]["kld_beta"] == pytest.approx(0.1)


def test_the_beta_provenance_names_the_effective_and_the_configured_value_apart(
    warmup_checkpoint, tiny_loader, tmp_path
) -> None:
    """Both are reported, under names neither of which can be read as the other."""
    from teb_vae.lag_attn.eval.runner import EvalRunner

    runner = EvalRunner.from_checkpoint(warmup_checkpoint(120), tmp_path / "run")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config={}, output_dir=tmp_path / "results"
    )

    provenance = summary["beta"]
    assert provenance["kld_beta_effective"] == pytest.approx(0.1)
    assert provenance["kld_beta_configured"] == pytest.approx(0.001)
    assert provenance["checkpoint_epoch"] == 120
    assert provenance["beta_schedule"]["kind"] == "linear_warmup"


def test_total_loss_is_reassembled_with_the_effective_beta(
    warmup_checkpoint, tiny_loader, tmp_path
) -> None:
    r"""$L = \lambda_f L_{\mathrm{feat}} + \lambda_b L_{\mathrm{base}} + \beta L_{KL}
    + \lambda_\ell L_{\mathrm{lag}}$, with $\beta$ the schedule's.

    The KL term is the whole difference, so the case is only meaningful when it is large enough
    to move the total -- asserted rather than assumed.
    """
    from teb_vae.lag_attn.eval.runner import EvalRunner

    runner = EvalRunner.from_checkpoint(warmup_checkpoint(120), tmp_path / "run")
    summary = scalars_analysis.run_scalar_analysis(
        runner, tiny_loader, eval_config={}, output_dir=tmp_path / "results"
    )
    table = summary["metrics"]
    objective = runner.objective
    beta = objective.effective_beta()

    expected = (
        objective.lambda_full * table["feat_loss"]
        + objective.lambda_base * table["base_loss"]
        + beta * table["kld_loss"]
        + objective.lambda_lag * table["lag_smoothness"]
    )
    assert table["total_loss"] == pytest.approx(expected, rel=1e-9)
    assert table["main_loss"] == pytest.approx(expected, rel=1e-9)

    # What the old code produced. Kept as an explicit contrast so the case cannot silently
    # degenerate into one where the two agree.
    under_constant = expected + (objective.kld_beta - beta) * table["kld_loss"]
    assert abs(under_constant - expected) > 1e-9, "kld_loss is too small to separate the two betas"
    assert table["total_loss"] != pytest.approx(under_constant, rel=1e-9)


# ---------------------------------------------------------------------------
# The residual diagnostics
#
# Both deviations from `task.py` biased the same way -- low -- and neither is visible at B = 1
# with binary weights, which is what the pre-existing coverage used.
# ---------------------------------------------------------------------------
def _weighted_batch(batch_size=3, seed=1):
    """A stub batch whose ``weight`` is fractional and differs per sample.

    Non-binary and non-uniform on purpose. With a binary weight, folding it into the mask changes
    which cells count but not the *shape* of the error, and with one identical weight per sample
    the per-sample and pooled reductions coincide -- so either simplification hides one of the two
    defects.
    """
    from teb_vae.lag_attn.tests.conftest import SEQ_LEN, make_stub_batch

    batch = make_stub_batch(batch_size=batch_size, seq_len=SEQ_LEN, seed=seed)
    weight = torch.ones(batch_size, SEQ_LEN)
    for index in range(batch_size):
        weight[index, : 10 * (index + 1)] = 0.0        # a different gap per sample
        weight[index, -6:] = 0.25 * (index + 1)        # a fractional edge, as the real shards carry
    batch.weight = weight
    return batch


def _task_over(runner):
    """Wrap the runner's own model in the task, so both sides score one set of weights."""
    from teb_vae.lag_attn.task import SeqVaeLagAttnTask
    from teb_vae.lag_attn.tests.conftest import PROD_HPARAMS

    return SeqVaeLagAttnTask(
        runner.model, lr=1e-3, model_kwargs=dict(runner.model_kwargs), **dict(PROD_HPARAMS)
    )


def test_the_residual_diagnostics_pool_exactly_as_the_task_does(make_eval_runner, tmp_path) -> None:
    r"""Numerical parity with ``task.py::_compute_residual_diagnostics`` on one forward.

    Both sides consume the *same* ``forward_outputs``, so the sampled $z$ is shared and any
    difference is a difference of masking or of reduction -- which is precisely what was wrong.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    task = _task_over(runner)
    batch = _weighted_batch()

    with torch.no_grad():
        outputs = runner.forward(batch)
    expected = task._compute_residual_diagnostics(
        forward_outputs=outputs, weight=batch.weight
    )

    produced = scalars_analysis._residual_diagnostics(runner, outputs, batch.weight)
    for name in ("delta_mu_rms", "mu_post_prior_gap_rms"):
        sum_squares, mask_sum = produced[name]
        assert mask_sum > 0.0
        pooled_rms = float(np.sqrt(sum_squares / mask_sum))
        assert pooled_rms == pytest.approx(float(expected[name]), rel=1e-5)


def test_dropping_the_weight_from_the_masks_would_change_the_answer(
    make_eval_runner, tmp_path
) -> None:
    """The guard on the fix: with ``weight=None`` both masks skip the validity factor entirely.

    Without this, a later edit could quietly drop the argument again and every assertion above
    would still pass on a batch whose weight happened to be all ones.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    batch = _weighted_batch()
    with torch.no_grad():
        outputs = runner.forward(batch)

    weighted = scalars_analysis._residual_diagnostics(runner, outputs, batch.weight)
    unweighted = scalars_analysis._residual_diagnostics(runner, outputs, None)
    for name in ("delta_mu_rms", "mu_post_prior_gap_rms"):
        assert weighted[name][1] < unweighted[name][1], (
            f"{name}: the unweighted mask must count the gap steps the loss never scored"
        )


def test_the_pooled_rms_is_not_the_mean_of_the_per_sample_rms(make_eval_runner, tmp_path) -> None:
    r"""Jensen: $\mathrm{mean}\sqrt{x} < \sqrt{\mathrm{mean}\,x}$ whenever the samples differ.

    This is the reduction half of the defect, and it is invisible at $B = 1$ where the two
    coincide exactly. The per-sample form is still correct *as such* -- ``analyses/residual.py``
    reports it deliberately -- so this asserts the difference rather than deprecating it.
    """
    from teb_vae.lag_attn.eval import masks, metrics

    runner = make_eval_runner(output_dir=tmp_path / "runner")
    batch = _weighted_batch()
    with torch.no_grad():
        outputs = runner.forward(batch)

    batch_size, seq_len = int(outputs["mu_prior"].shape[0]), int(outputs["mu_prior"].shape[1])
    latent_mask = masks.kld_mask(runner.model, batch.weight, batch_size, seq_len)
    per_sample = metrics.posterior_drift(outputs, latent_mask).cpu().numpy()

    sum_squares, mask_sum = scalars_analysis._residual_diagnostics(
        runner, outputs, batch.weight
    )["mu_post_prior_gap_rms"]
    pooled = float(np.sqrt(sum_squares / mask_sum))

    assert float(np.nanmean(per_sample)) < pooled, "the mean of roots must be the smaller number"
    assert float(np.nanmean(per_sample)) != pytest.approx(pooled, rel=1e-4)


def test_the_split_level_rms_roots_the_pooled_ratio_once(make_eval_runner, tmp_path) -> None:
    r"""Across batches too: $\sqrt{\sum_b \mathrm{SS}_b / \sum_b m_b}$, not a mean of roots.

    ``mu_post_prior_gap_rms`` is the deterministic one of the pair -- it reads ``mu_post`` and
    ``mu_prior`` only, never the sampled $z$ -- so the reference can be recomputed from a second
    forward without seeding the sampler.
    """
    runner = make_eval_runner(output_dir=tmp_path / "runner")
    batches = [_weighted_batch(batch_size=3, seed=1), _weighted_batch(batch_size=2, seed=2)]

    summary = scalars_analysis.run_scalar_analysis(
        runner, batches, eval_config={}, output_dir=tmp_path / "results"
    )

    total_squares, total_mask, per_batch_roots = 0.0, 0.0, []
    with torch.no_grad():
        for batch in batches:
            pair = scalars_analysis._residual_diagnostics(
                runner, runner.forward(batch), batch.weight
            )["mu_post_prior_gap_rms"]
            total_squares += pair[0]
            total_mask += pair[1]
            per_batch_roots.append(float(np.sqrt(pair[0] / pair[1])))

    pooled = float(np.sqrt(total_squares / total_mask))
    assert summary["metrics"]["mu_post_prior_gap_rms"] == pytest.approx(pooled, rel=1e-6)
    # The two batches have different mask densities, so a mean of the per-batch roots is a
    # different number -- which is what makes the assertion above load-bearing.
    assert float(np.mean(per_batch_roots)) != pytest.approx(pooled, rel=1e-9)
