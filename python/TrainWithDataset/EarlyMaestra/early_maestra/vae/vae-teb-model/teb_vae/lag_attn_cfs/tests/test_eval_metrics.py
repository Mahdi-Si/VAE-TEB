r"""The evaluation readouts in the feature domain, and the three identities that make them readable.

Two things separate this file from the sibling's, and both are the same thing seen twice: the
target is $98$ wavelet-modulus and phase-harmonic coefficients rather than $16$ raw samples, and
the forward decodes a **gathered anchor set** rather than a contiguous prefix.

So the load-bearing assertions here are the ones a mechanical copy of the raw pipeline would have
got wrong *silently*:

* the forward is called densely, in exactly one place, and the target and both masks are built
  from the anchor set it returned rather than from a second derivation of it;
* ``mc_predictive_block`` **gathers** the latent at those anchors instead of slicing a prefix;
* every trivial baseline is built on the **gathered kept channels**, so its channel axis is
  positionally the target's -- the four channels the warm-up budget drops are interior to the
  declared order, so a baseline built on the declared width would be a silent mis-pairing of
  channels rather than a shape error;
* the per-channel gap vector sums to ``pred_gap``, over each stored block to ``pred_gap_st`` and
  ``pred_gap_ph``, and over each warm-up tertile to the three ``pred_gap_warm_*`` -- so the vector
  and those five scalars cannot disagree about the same decomposition.

The parity against the training objective lives in ``test_eval_parity.py``; the two controls in
``test_eval_controls.py``; the registry in ``test_eval_verdicts.py``.

**One tolerance convention, stated once.** ``pred_gap`` is a difference of order $10^{1}$ between
two block scores of order $10^{3}$, so two float32 reductions of the same terms in different orders
agree on the *scores* to $10^{-10}$ relative and on their difference to only $10^{-6}$ -- the
cancellation amplifies it by the ratio of the two magnitudes. Every recomposition below is
therefore asserted at an absolute tolerance scaled by the **block score**, which is the same
convention ``report_seam.check_per_anchor_recombines`` already uses and the only one that
distinguishes a rounding difference from a real one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.eval import metrics
from teb_vae.lag_attn_cfs.eval.metrics import (
    DENSE_ANCHOR_GEOMETRY,
    VECTOR_READOUTS,
    Aggregate,
    BatchReadout,
    aggregate_by_recording,
    baseline_forecasts,
    batch_guids,
    batch_size_of,
    build_verdicts,
    evaluate,
    evaluate_batch,
    horizon_residual_sums,
    lag_anchor_counts,
    lag_profiles,
    lag_summary,
    latent_health,
    masked_raw_error_sums,
    mc_predictive_block,
    model_inputs,
    source_lag_warmth_per_sample,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS, masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

from .conftest import (
    BATCH,
    TINY_KWARGS,
    build,
    make_stub_batch,
    tiny_warmup_kwargs,
)

#: Relative tolerance on a quantity that is not a cancelling difference.
RTOL = 1e-6


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior.

    Load-bearing: at initialisation the delta heads are zero, so the posterior *is* the prior,
    every KL is exactly zero, base and full are bitwise identical, ``pred_gap`` is zero, and every
    assertion below would pass on a model that is completely wrong.
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


def _dense_pieces(module, batch):
    """The forward, the anchor set, the gathered target and both masks, as ``evaluate_batch``
    builds them. Rebuilt rather than reached into, so a test comparing the two is comparing two
    derivations."""
    model = module.orig_model
    y_st, y_ph, u_stream, target_features, weight = model_inputs(module, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    mask, _coverage = forecast_mask(
        weight, model.geometry, coverage_floor=model.coverage_floor,
        anchors=anchors, anchor_valid=anchor_valid,
    )
    support = kl_mask(mask, model.geometry, anchors=anchors, anchor_valid=anchor_valid)
    return outputs, target_features, weight, target, mask, support


def _block_scale(readout: BatchReadout) -> float:
    """The magnitude every ``pred_gap`` recomposition tolerance is set by; see the docstring."""
    return float(readout.columns["nll_base_block"].abs().max())


# =================================================================================================
# The dense forward, the anchored target and the anchored masks
# =================================================================================================
def test_the_forward_is_called_in_exactly_one_place_and_at_the_dense_geometry() -> None:
    """An AST walk rather than a grep: a second call site is how a run would decode two anchor
    sets and score one against the other, and neither shape nor count would say so."""
    tree = ast.parse(Path(metrics.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "model"
    ]

    assert len(calls) == 1, f"{len(calls)} calls to the net's forward; there must be exactly one"
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "anchor_phase", "anchor_stride"
    }, "both halves of the anchor geometry must be named, never positional"


def test_the_dense_geometry_is_the_pair_the_task_resolves_on_the_evaluation_stages(
    trained_task, stub_batch
) -> None:
    """Measured against the task's own seam rather than asserted from a constant: the evaluation
    decodes what ``resolve_anchor_geometry('test', batch)`` returns, and a constant that had
    drifted from it would name an $A_{\\max}$ no evaluation run produces."""
    resolved = trained_task.resolve_anchor_geometry("test", stub_batch)

    assert (int(resolved[0]), int(resolved[1])) == DENSE_ANCHOR_GEOMETRY


def test_the_forward_decodes_every_valid_anchor_at_that_geometry(trained_task, stub_batch) -> None:
    outputs, _tf, _weight, _target, _mask, _support = _dense_pieces(trained_task, stub_batch)
    model = trained_task.orig_model
    expected = model.geometry.t_valid - model.warmup_period

    assert outputs["anchor_index"].shape == (BATCH, expected)
    assert int(outputs["anchor_index"][0, 0]) == model.warmup_period
    assert bool(outputs["anchor_valid"].all()), "the dense set has no padding by construction"


def test_the_target_and_the_masks_carry_the_anchor_axis(trained_task, stub_batch) -> None:
    outputs, _tf, _weight, target, mask, support = _dense_pieces(trained_task, stub_batch)
    model = trained_task.orig_model
    a_max = int(outputs["anchor_index"].shape[1])

    assert target.shape == (BATCH, a_max, model.horizon, model.decoder_out_channels)
    assert mask.shape == (BATCH, a_max, model.horizon)
    # The one place the anchor axis is scattered back to dense, because the latent tensors the KL
    # gates are produced at every step whatever was decoded.
    assert support.shape == (BATCH, model.geometry.t)


def test_the_target_is_bitwise_the_one_the_objective_builds(trained_task, stub_batch) -> None:
    """Not merely the same shape. The builder and the anchor set are the two things that could
    differ, and the objective reaches both through one line of ``compute_loss``: a second gather
    here -- of a different channel set, or at a second anchor derivation -- would produce a target
    of the right shape scored against the wrong coefficients."""
    outputs, target_features, _weight, expected, _mask, _support = _dense_pieces(
        trained_task, stub_batch
    )
    model = trained_task.orig_model

    readout = evaluate_batch(trained_task, stub_batch, num_samples=1, retain=("target",))

    # The exact expression ``FeatureForecastTarget.compute_loss`` runs.
    objective_target = model._build_forecast_target(
        target_features, outputs.get("anchor_index")
    )
    assert torch.equal(readout.retained["target"], objective_target)
    assert torch.equal(readout.retained["target"], expected)


def test_the_monte_carlo_estimator_gathers_the_latent_at_the_anchors(
    trained_task, stub_batch
) -> None:
    r"""The property that breaks first if the gather were a slice.

    Driven at $K = 1$ with a branch whose log-variance is $-\infty$ in effect, so the "sample" is
    the mean exactly and the estimator's output must be the training path's own block score --
    which is only true if the latent it decoded is the one the forward decoded. A slice of the
    contiguous prefix would produce $T_{\mathrm{valid}}$ anchors against a $A_{\max}$ target: not
    broadcastable, so it fails loudly rather than silently, but four functions deep and $K$ draws
    into a multi-hour pass.
    """
    outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)
    model = trained_task.orig_model
    deterministic = torch.full_like(outputs["z_post"], -1.0e30)

    scores, _contributing = mc_predictive_block(
        model,
        {"full": (outputs["z_post"], deterministic)},
        target,
        mask,
        anchors=outputs["anchor_index"],
        likelihood="gaussian_nll",
        num_samples=1,
    )
    expected, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood="gaussian_nll",
        logvar=outputs["logvar_full"],
    )

    assert torch.equal(scores["full"], expected)


def test_the_estimator_refuses_to_run_without_an_anchor_set(
    trained_task, stub_batch
) -> None:
    """``anchors`` is a required keyword rather than an optional one, so the slice the sibling
    takes is not reachable here by omission."""
    outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)

    with pytest.raises(TypeError, match="anchors"):
        mc_predictive_block(
            trained_task.orig_model,
            {"full": (outputs["mu_post"], outputs["logvar_post"])},
            target,
            mask,
            likelihood="gaussian_nll",
        )


def test_calling_the_forward_without_a_phase_is_refused_at_a_real_stride(
    trained_task, stub_batch
) -> None:
    """The negative control that proves the dense call is *necessary* rather than merely
    sufficient. This task is built at the tiling stride, where a forward with no phase would decode
    every sample of every epoch at one grid -- and $A_{\\max}$ is a geometry constant either way,
    so nothing about the shapes would say so."""
    model = trained_task.orig_model
    y_st, y_ph, u_stream, _tf, _weight = model_inputs(trained_task, stub_batch)

    assert model.anchor_stride > 1, "this control needs a model that actually tiles"
    with pytest.raises(ValueError, match="anchor_phase is required"):
        model(y_st, y_ph, u_stream)


def test_the_per_anchor_table_carries_the_decimated_step_each_row_scores(
    trained_task, stub_batch
) -> None:
    """The anchor axis is a gathered *set*, so a row's position in it is not the step it scores.
    A table keyed on position alone could not be joined against the trajectory axis or the event
    table, and the join would look plausible."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    outputs, _tf, _weight, _target, _mask, _support = _dense_pieces(trained_task, stub_batch)

    assert torch.equal(readout.per_anchor["anchor_index"], outputs["anchor_index"])


# =================================================================================================
# Baselines in feature space
# =================================================================================================
def _constant_channel_batch(seed: int = 0):
    """A batch whose target stream is the constant $c$ in declared channel $c$, at every step.

    The known-answer construction for the positional-identity assertion: any baseline built on the
    gathered stream carries the keep-index's own values, in the keep-index's order, and one built
    on the declared width would carry $0, 1, 2, \\ldots$ instead.
    """
    batch = make_stub_batch(seed=seed)
    channels = batch.fhr_st.shape[-1]
    steps = batch.fhr_st.shape[1]
    declared = torch.arange(batch.fhr_st.shape[-1] + batch.fhr_ph.shape[-1], dtype=torch.float32)
    grid = declared[None, None, :].expand(batch.fhr_st.shape[0], steps, -1).clone()
    batch.fhr_st = grid[:, :, :channels].contiguous()
    batch.fhr_ph = grid[:, :, channels:].contiguous()
    return batch


def test_every_baseline_is_built_on_the_gathered_kept_channels(trained_task) -> None:
    """The single most likely way a raw-domain assumption survives into this analysis. The four
    channels the warm-up budget drops are interior to the declared order, so a baseline built on
    the declared width would be a silent mis-pairing of channels wherever the shapes happened to
    survive -- not a shape error."""
    batch = _constant_channel_batch()
    outputs, target_features, weight, _target, _mask, _support = _dense_pieces(
        trained_task, batch
    )
    model = trained_task.orig_model
    keep = model.target_gate.keep_index.to(torch.float32)

    baselines = baseline_forecasts(
        target_features, weight, model, outputs["anchor_index"]
    )

    assert keep.numel() < model.c_y, "an ungated model would make this assertion vacuous"
    for name in ("persistence", "segment_mean"):
        block = baselines[name]
        assert block.shape[-1] == int(model.decoder_out_channels)
        assert torch.allclose(block.reshape(-1, keep.numel())[0], keep)


def test_persistence_carries_the_last_observed_step_rather_than_the_last(trained_task) -> None:
    """"Last observed" rather than "last": ``weight`` is the only trustworthy validity signal
    here, because unlike the raw trace the coefficients carry no gap sentinel at all -- a
    carried-forward invalid step is an ordinary-looking number that would quietly measure the gap.
    """
    batch = make_stub_batch(seed=5)
    # A ramp per step, identical in every channel, so the value carried forward names the step.
    steps = torch.arange(batch.fhr_st.shape[1], dtype=torch.float32)
    batch.fhr_st = steps[None, :, None].expand_as(batch.fhr_st).clone()
    batch.fhr_ph = steps[None, :, None].expand_as(batch.fhr_ph).clone()
    gap_start, gap_end = 8, 12
    batch.weight[:, gap_start : gap_end + 1] = 0.0

    outputs, target_features, weight, _target, _mask, _support = _dense_pieces(
        trained_task, batch
    )
    baselines = baseline_forecasts(
        target_features, weight, trained_task.orig_model, outputs["anchor_index"]
    )
    anchors = outputs["anchor_index"][0].tolist()
    carried = baselines["persistence"][0, :, 0, 0].tolist()

    for position, anchor in enumerate(anchors):
        expected = gap_start - 1 if gap_start <= anchor <= gap_end else anchor
        assert carried[position] == pytest.approx(float(expected)), (
            f"anchor {anchor} carried step {carried[position]}, not {expected}"
        )


def test_climatology_is_exactly_zero(trained_task, stub_batch) -> None:
    """Exactly, not approximately: it is the z-scored population mean, which the loader's own
    statistics put at zero by construction."""
    outputs, target_features, weight, _target, _mask, _support = _dense_pieces(
        trained_task, stub_batch
    )

    baselines = baseline_forecasts(
        target_features, weight, trained_task.orig_model, outputs["anchor_index"]
    )

    assert float(baselines["climatology"]) == 0.0
    assert baselines["climatology"].numel() == 1, "a scalar broadcasts; an expanded grid costs"


def test_zero_is_the_channel_mean_over_the_region_the_model_reads(
    cohort_config, cohort_loader
) -> None:
    r"""What makes the climatology baseline a baseline rather than an arbitrary constant.

    The statistics were accumulated **excluding** each channel's warm-up region, so zero is the
    population mean over exactly the region the model reads -- and it is that property, not the
    number, that the source-null control's premise also rests on. Measured over the whole generated
    split rather than asserted, restricted to the valid steps at or after each kept channel's own
    rebased warm-up.

    The tolerance is a property of the fixture's size, not of the claim: eight real segments reused
    across eight cohort shards is $48$ segments, and a channel mean over that many is a sample mean
    rather than the population's.
    """
    from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget

    budget = resolve_warmup_budget(cohort_config)
    assert budget is not None
    keep = np.asarray(budget.target.keep_index, dtype=np.int64)
    waits = np.asarray(budget.target.warmup_steps, dtype=np.int64)

    totals = np.zeros(keep.size, dtype=np.float64)
    counts = np.zeros(keep.size, dtype=np.float64)
    for batch in cohort_loader:
        stream = torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1).numpy()[:, :, keep]
        valid = batch.weight.numpy() >= 1.0
        warm = np.arange(stream.shape[1])[None, :, None] >= waits[None, None, :]
        weights = (valid[:, :, None] & warm).astype(np.float64)
        totals += (stream * weights).sum(axis=(0, 1))
        counts += weights.sum(axis=(0, 1))

    means = totals / np.maximum(counts, 1.0)
    assert float(counts.min()) > 0.0, "every kept channel must contribute or this proves nothing"
    assert abs(float(totals.sum() / counts.sum())) < 0.05
    assert float(np.max(np.abs(means))) < 0.25


def test_the_segment_mean_reads_only_the_segments_valid_steps(trained_task) -> None:
    """Deliberately the stronger baseline -- it is **not** causal, since it reads the segment's
    whole future -- so a model that fails to beat it has learned nothing recording-specific that a
    constant could not say. Averaging the gap in would make it a weaker one for the wrong reason.
    """
    batch = make_stub_batch(seed=6)
    batch.fhr_st = torch.ones_like(batch.fhr_st)
    batch.fhr_ph = torch.ones_like(batch.fhr_ph)
    # An invalid stretch carrying a value nothing should average in.
    batch.fhr_st[:, 3:7] = 100.0
    batch.fhr_ph[:, 3:7] = 100.0
    batch.weight[:, 3:7] = 0.0

    outputs, target_features, weight, _target, _mask, _support = _dense_pieces(
        trained_task, batch
    )
    baselines = baseline_forecasts(
        target_features, weight, trained_task.orig_model, outputs["anchor_index"]
    )

    assert torch.allclose(
        baselines["segment_mean"], torch.ones_like(baselines["segment_mean"]), rtol=RTOL
    )


def test_a_segment_with_no_valid_step_yields_nan_rather_than_a_fabricated_zero(
    trained_task
) -> None:
    """Zero is the *climatology* here, so a fabricated zero would silently report the population
    mean as this segment's own and score a second identical baseline under a different name. Such a
    segment scores no anchors, so the NaN leaves the aggregation with the rest of its row."""
    batch = make_stub_batch(seed=8)
    batch.weight[1] = 0.0

    outputs, target_features, weight, _target, _mask, _support = _dense_pieces(
        trained_task, batch
    )
    baselines = baseline_forecasts(
        target_features, weight, trained_task.orig_model, outputs["anchor_index"]
    )

    assert bool(torch.isfinite(baselines["segment_mean"][0]).all())
    assert bool(torch.isnan(baselines["segment_mean"][1]).all())

    readout = evaluate_batch(trained_task, batch, num_samples=1)
    assert float(readout.n_anchors[1]) == 0.0, "the NaN row must leave the aggregation whole"


def test_every_baseline_is_scored_by_the_same_masked_scorer_and_denominator(
    trained_task, stub_batch
) -> None:
    """All three shapes broadcast against the model branches', so the same scorer accepts them
    unchanged and all three are reduced by the same contributing-anchor count -- which is what
    makes a skill score a comparison of predictors rather than of scoring conventions."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    for name in metrics.BASELINE_NAMES:
        assert readout.columns[f"nll_{name}_block"].shape == (BATCH,)
        assert readout.columns[f"sq_error_{name}"].shape == (BATCH,)
    assert set(metrics.FORECAST_BRANCHES) == {"base", "full", *metrics.BASELINE_NAMES}
    assert metrics.BASELINE_LOGVAR == 0.0


def test_a_perfect_forecast_scores_a_baseline_skill_of_exactly_one(
    trained_task, stub_batch
) -> None:
    """The composition the ``forecast`` analysis reads: a forecast equal to the truth has zero
    squared error, and ``skill_against`` then returns exactly $1$ -- not $0.999$, and not a
    division by a baseline error that happened to be zero too."""
    from teb_vae.lag_attn_cfs.eval.frames import skill_against

    _outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    perfect = masked_raw_error_sums(target, target, mask)
    model_error = float(perfect["sum_sq"].max())

    assert model_error == 0.0
    for name in metrics.BASELINE_NAMES:
        baseline_error = np.asarray(readout.columns[f"sq_error_{name}"], dtype=np.float64)
        assert baseline_error.min() > 0.0, "a zero-error baseline would make this vacuous"
        assert skill_against(np.zeros_like(baseline_error), baseline_error) == pytest.approx(1.0)


# =================================================================================================
# The warm-up tertiles, the source warmth and the two geometry guards
# =================================================================================================
def test_the_three_warm_up_tertile_gaps_recompose_into_the_forecast_gap(
    trained_task, stub_batch
) -> None:
    """The property that makes them a decomposition rather than three unrelated numbers."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    recomposed = (
        readout.columns["pred_gap_warm_lo"]
        + readout.columns["pred_gap_warm_mid"]
        + readout.columns["pred_gap_warm_hi"]
    )

    assert torch.allclose(
        recomposed, readout.columns["pred_gap"], rtol=0.0, atol=RTOL * _block_scale(readout)
    )


def test_the_two_stored_block_gaps_recompose_into_the_same_number(
    trained_task, stub_batch
) -> None:
    """The other cut of the same axis. The two splits are not restatements of each other: the
    tertiles cut by filter speed and run *across* the stored block boundary."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    recomposed = readout.columns["pred_gap_st"] + readout.columns["pred_gap_ph"]

    assert torch.allclose(
        recomposed, readout.columns["pred_gap"], rtol=0.0, atol=RTOL * _block_scale(readout)
    )


def test_the_two_channel_splits_are_not_the_same_partition(trained_task) -> None:
    """Non-vacuity for the pair above: on a model whose tertiles happened to coincide with the
    stored blocks the two recompositions would be one assertion written twice."""
    model = trained_task.orig_model
    keep = model.target_gate.keep_index
    first_block = keep < model.TARGET_BLOCK_SPLIT

    for group in range(3):
        members = model.warm_tertile_id == group
        assert bool(members.any())
        assert not bool(torch.equal(members, first_block))


def test_the_decoded_anchor_count_is_the_dense_set(trained_task, stub_batch) -> None:
    """Counted off ``anchor_valid`` rather than off the mask, deliberately: it must report the
    **decoded** set, so a batch whose weight is entirely zero still says which anchors the forward
    built rather than reporting the geometry as having collapsed."""
    model = trained_task.orig_model
    expected = model.geometry.t_valid - model.warmup_period

    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert [float(value) for value in readout.columns["anchors_per_sample"]] == [
        float(expected)
    ] * BATCH
    assert float(readout.n_anchors.max()) < float(expected), (
        "the stub's gap must drop some anchors, or the decoded and scored counts coincide and "
        "this assertion could not tell them apart"
    )


def test_the_warm_target_fraction_is_exactly_one(trained_task, stub_batch) -> None:
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert [float(value) for value in readout.columns["target_warm_frac"]] == [1.0] * BATCH


def test_a_floor_below_the_kept_channels_budget_is_refused_at_construction() -> None:
    r"""Which is why no column can ever report ``target_warm_frac`` below $1.0$ on a model this
    constructor built. A forecast at anchor $t$ reads target step $t + 1$ at the earliest, so the
    floor must be at least $B - 1$; below it the objective scores assumed pre-recording history as
    signal, with every shape correct and every warm-fraction readout still reporting $1.0$."""
    kwargs = tiny_warmup_kwargs()
    budget = max(kwargs["target_warmup_steps"])

    build(dict(kwargs, warmup_period=budget - 1))
    with pytest.raises(ValueError, match="below the anchor floor"):
        build(dict(kwargs, warmup_period=budget - 2))


def test_the_source_warmth_columns_recombine_into_the_models_own_reduction(
    trained_task, stub_batch
) -> None:
    """The model's ``_source_lag_warmth`` is the definition and reduces a whole batch to two
    scalars; the columns here open the batch axis and nothing else. Recombined by **attention
    mass** -- which is the weight each sample's fraction enters the model's ratio with -- the two
    must agree."""
    outputs, _tf, _weight, target, _mask, _support = _dense_pieces(trained_task, stub_batch)
    model = trained_task.orig_model

    per_sample = source_lag_warmth_per_sample(model, outputs, target.dtype)
    shared = model._source_lag_warmth(outputs, target)

    alpha = outputs["attn_weights"]
    index = outputs["anchor_index"][:, :, None, None].expand(
        -1, -1, alpha.shape[2], alpha.shape[3]
    )
    mass = (
        alpha.gather(1, index) * outputs["anchor_valid"].to(alpha.dtype)[:, :, None, None]
    ).sum(dim=(1, 2, 3))

    for name in ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph"):
        recombined = float((per_sample[name] * mass).sum() / mass.sum())
        assert recombined == pytest.approx(float(shared[name]), rel=1e-5)


def test_the_source_warmth_fractions_are_proportions(trained_task, stub_batch) -> None:
    """In $[0, 1]$ by construction, because the denominator is the attention mass actually present
    rather than a row count: a row with no admissible lag normalises to zero, and zero over zero
    would otherwise be the answer.

    Nothing here asserts the value is small. A small value is the **expected** finding on this
    cell, and it is a finding about a model and a population -- neither of which this fixture is.
    """
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    for name in ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph"):
        values = readout.columns[name]
        assert bool(((values >= 0.0) & (values <= 1.0)).all())


# =================================================================================================
# The horizon and calibration accumulators, on the coefficient axis
# =================================================================================================
def test_the_per_horizon_counts_sum_to_the_scored_coefficient_count(
    trained_task, stub_batch
) -> None:
    r"""The denominator is per $\tau$, not the per-anchor contributing indicator -- that indicator
    is an ``amax`` over $\tau$, so using it would divide a late horizon's numerator by a count that
    includes the steps the mask already zeroed, and the late horizons would read artificially good
    exactly where the signal is worst."""
    outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)

    sums = horizon_residual_sums(
        outputs["mu_full"], outputs["logvar_full"], target, mask
    )

    channels = float(target.shape[-1])
    assert sums["count"].shape == (trained_task.orig_model.horizon,)
    assert float(sums["count"].sum()) == pytest.approx(float(mask.sum()) * channels)
    assert sums["sum_sq"].dtype == torch.float64, "a real split reaches 1e9 terms"


def test_the_horizon_block_scores_sum_to_the_per_anchor_block_score(
    trained_task, stub_batch
) -> None:
    r"""$\sum_\tau D_{b,a,\tau} = D_{b,a}$, by construction rather than by coincidence: the two
    reduce the same elementwise term over different axes."""
    outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)

    per_tau = metrics.masked_raw_block_per_horizon_step(
        outputs["mu_full"], target, mask, likelihood="gaussian_nll",
        logvar=outputs["logvar_full"],
    )
    per_anchor, _contributing = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood="gaussian_nll",
        logvar=outputs["logvar_full"],
    )

    assert torch.allclose(per_tau.sum(dim=2), per_anchor, rtol=1e-5)


def test_the_calibration_census_counts_coefficients(trained_task, stub_batch) -> None:
    outputs, _tf, _weight, target, mask, _support = _dense_pieces(trained_task, stub_batch)

    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert float(readout.calibration_sums["count"]) == pytest.approx(
        float(mask.sum()) * float(target.shape[-1])
    )
    assert float(outputs["logvar_full"].shape[-1]) == float(target.shape[-1])


def test_no_spectral_symbol_survives_in_the_readout_module() -> None:
    """``coherence`` is not ported at all: a stored coefficient is a **modulus**, so the analysing
    filter's phase was discarded before the value was written and the cross-spectral sufficient
    statistics have no analogue here at any window length. Accumulating them anyway would be paying
    for an estimator nothing can read."""
    forbidden = (
        "tau_slices",
        "tau_slice_window_validity",
        "source_tau_slices",
        "cross_spectral_sums",
        "_welch_segments",
        "spectral_sums",
    )
    tree = ast.parse(Path(metrics.__file__).read_text(encoding="utf-8"))

    assert [name for name in forbidden if hasattr(metrics, name)] == []
    # Walked rather than grepped: the module docstring names ``coherence`` once, in the sentence
    # saying it is not ported, and a substring scan cannot tell that from a lazy import inside a
    # function -- which is exactly how the estimator would come back.
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module or ''}.{alias.name}" for alias in node.names]
    assert [name for name in imported if "spectra" in name or "coherence" in name] == []
    defined = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert [name for name in defined if name in forbidden] == []
    # And the dataclass field the sibling carries them on is gone with them, so a sink written
    # against the sibling's readout fails rather than silently writing an empty sidecar.
    assert not hasattr(metrics.BatchReadout, "spectral_sums")


# =================================================================================================
# The per-channel vector readouts
# =================================================================================================
def test_the_per_channel_gap_vector_sums_to_the_samples_own_forecast_gap(
    trained_task, stub_batch
) -> None:
    """Per sample, not only in the batch mean: a compensating pair of per-sample errors would hide
    inside a batch-level equality."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert torch.allclose(
        readout.gap_per_channel.sum(dim=1),
        readout.columns["pred_gap"],
        rtol=0.0,
        atol=RTOL * _block_scale(readout),
    )


@pytest.mark.parametrize(
    "column,selector",
    [
        ("pred_gap_st", "block"),
        ("pred_gap_ph", "block_complement"),
        ("pred_gap_warm_lo", 0),
        ("pred_gap_warm_mid", 1),
        ("pred_gap_warm_hi", 2),
    ],
)
def test_the_vector_and_the_scalars_agree_about_the_same_decomposition(
    trained_task, stub_batch, column, selector
) -> None:
    """The new vector and the five existing scalars are two reductions of one tensor, and this is
    what says they cannot come apart."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    model = trained_task.orig_model

    if selector == "block":
        mask = model.target_gate.keep_index < model.TARGET_BLOCK_SPLIT
    elif selector == "block_complement":
        mask = model.target_gate.keep_index >= model.TARGET_BLOCK_SPLIT
    else:
        mask = model.warm_tertile_id == selector

    partial = (readout.gap_per_channel * mask.to(readout.gap_per_channel.dtype)).sum(dim=1)

    assert torch.allclose(
        partial, readout.columns[column], rtol=0.0, atol=RTOL * _block_scale(readout)
    )


def test_the_vectors_are_positional_against_the_kept_channel_axis(
    trained_task, stub_batch
) -> None:
    """Length asserted against the gate's own width rather than a literal, and row-aligned with
    the per-sample columns -- which is what lets the band-resolved analysis join them against a
    channel map without a key."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)
    kept = int(trained_task.orig_model.target_gate.out_channels)

    for name in ("gap_per_channel", "sq_error_per_channel_base", "sq_error_per_channel_full"):
        vector = getattr(readout, name)
        assert vector.shape == (BATCH, kept)
        assert name in VECTOR_READOUTS
    assert kept == int(trained_task.orig_model.decoder_out_channels)


@pytest.mark.parametrize("branch", ["base", "full"])
def test_the_per_channel_squared_error_averages_to_the_pooled_column(
    trained_task, stub_batch, branch: str
) -> None:
    """A band-level skill needs a natural zero, which a pooled squared error does not have. The
    mean over channels being the pooled column is what keeps the resolved form comparable with it.
    """
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    resolved = getattr(readout, f"sq_error_per_channel_{branch}").mean(dim=1)

    assert torch.allclose(resolved, readout.columns[f"sq_error_{branch}"], rtol=1e-5)


def test_a_masked_sample_contributes_exactly_zero_to_every_channel(trained_task) -> None:
    """Zero rather than NaN: a NaN in one row of a vector readout propagates through the
    per-recording mean and takes the whole channel out of the band analysis above it."""
    batch = make_stub_batch(seed=9)
    batch.weight[1] = 0.0

    readout = evaluate_batch(trained_task, batch, num_samples=1)

    assert bool(torch.equal(readout.gap_per_channel[1], torch.zeros_like(
        readout.gap_per_channel[1]
    )))
    assert bool(torch.isfinite(readout.gap_per_channel).all())


# =================================================================================================
# Aggregation
# =================================================================================================
def _readout(guids, values, anchors, vectors=None) -> BatchReadout:
    """A hand-built readout carrying one column, for the aggregation arithmetic.

    Args:
        guids: Recording identifier per sample.
        values: The single column's per-sample values.
        anchors: Contributing anchors per sample; a zero excludes that sample.
        vectors: Optional per-sample vector values, one row per sample, shared by every vector
            readout. Zeros when omitted.
    """
    rows = torch.zeros(len(guids), 3) if vectors is None else torch.tensor(
        vectors, dtype=torch.float32
    )
    return BatchReadout(
        guids=list(guids),
        columns={"score": torch.tensor(values, dtype=torch.float32)},
        n_anchors=torch.tensor(anchors, dtype=torch.float32),
        # Driven from the readout set rather than listed: every vector travels the identical
        # chain, so this helper tests the chain rather than a particular vector, and a readout
        # added later reaches these assertions without an edit here.
        **{name: rows for name in VECTOR_READOUTS},
    )


def test_a_segment_that_scored_no_anchors_is_excluded_rather_than_counted_as_zero() -> None:
    """Its per-sample mean divides by a denominator clamped to 1, so an empty numerator reads
    as exactly 0.0 -- a fabricated score, not a small one. Averaged into a summed-1470-coefficient
    block figure it would drag the headline toward zero and shrink pred_gap silently."""
    aggregate = aggregate_by_recording(
        [_readout(["a", "a"], [4.0, 0.0], [10, 0]), _readout(["b"], [6.0], [10])]
    )

    assert aggregate.per_recording["a"]["score"] == pytest.approx(4.0)
    assert aggregate.overall["score"] == pytest.approx(5.0)
    assert aggregate.n_samples == 3, "every segment seen is still counted"
    assert aggregate.n_samples_without_anchors == 1, "and the excluded one is reported"


def test_a_pass_that_scored_nothing_reports_no_headline_rather_than_zeros() -> None:
    """The across-recording denominator must not be clamped to 1 when no recording survived."""
    aggregate = aggregate_by_recording([_readout(["a", "b"], [4.0, 6.0], [0, 0])])

    assert aggregate.overall == {}
    assert aggregate.n_recordings == 0
    assert aggregate.n_samples_without_anchors == 2
    assert aggregate.gap_per_channel == [], "the vectors are equally absent"

    statuses = {verdict.name: verdict.status for verdict in build_verdicts(aggregate)}
    assert set(statuses.values()) == {"INCONCLUSIVE"}, (
        f"a pass that measured nothing must not diagnose anything, got {statuses}"
    )


def test_segments_of_one_recording_count_once() -> None:
    """Three segments of recording A and one of B: A must not outvote B three to one."""
    aggregate = aggregate_by_recording(
        [_readout(["a", "a"], [1.0, 3.0], [10, 10]), _readout(["a", "b"], [2.0, 10.0], [10, 10])]
    )

    assert aggregate.n_recordings == 2
    assert aggregate.per_recording["a"]["score"] == pytest.approx(2.0)
    # The recording mean, not the segment mean, which would be 4.0.
    assert aggregate.overall["score"] == pytest.approx(6.0)


def test_the_vector_readouts_take_the_same_route_as_the_scalars() -> None:
    """Same per-recording denominator, same zero-anchor exclusion -- which is what keeps a
    decomposition equal to the scalar it decomposes."""
    aggregate = aggregate_by_recording(
        [
            _readout(
                ["a", "a", "b"],
                [1.0, 3.0, 10.0],
                [10, 10, 10],
                vectors=[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            ),
            _readout(["b"], [0.0], [0], vectors=[[99.0, 0.0, 0.0]]),
        ]
    )

    assert aggregate.overall["score"] == pytest.approx(6.0)
    for name in VECTOR_READOUTS:
        assert getattr(aggregate, name)[0] == pytest.approx(6.0)


def test_inconsistent_columns_are_refused_rather_than_averaged(trained_task) -> None:
    """A last batch too small to derange produces a different column set; averaging it in would
    quietly drop the negative control from the headline."""
    full = evaluate_batch(trained_task, make_stub_batch(batch=2), num_samples=1)
    partial = evaluate_batch(trained_task, make_stub_batch(batch=1), num_samples=1)

    with pytest.raises(ValueError, match="different readout columns"):
        aggregate_by_recording([full, partial])


def test_no_batches_aggregates_to_nothing_rather_than_raising() -> None:
    aggregate = aggregate_by_recording([])

    assert aggregate.n_recordings == 0 and aggregate.overall == {}


# =================================================================================================
# Latent health and the lag report
# =================================================================================================
def test_latent_health_counts_dimensions_against_the_training_threshold() -> None:
    """The same threshold the training metric ``kld_active_frac`` reports against; a second copy
    would be a second threshold."""
    health = latent_health(Aggregate(kld_per_dim=[1.0, 0.5, KLD_ACTIVE_EPS / 2.0, 0.0]))

    assert health["d_z"] == 4
    assert health["active_dims"] == 2
    assert health["activity_threshold_nats"] == KLD_ACTIVE_EPS


def test_the_lag_report_carries_the_group_delay_caveat() -> None:
    """It travels in the record rather than only in a document beside it: a peak's position on this
    axis is not a physiological latency, and a reader given only the lag figures would have no way
    to know that."""
    from teb_vae.lag_attn_cfs.eval.lag_axis import GROUP_DELAY_CAVEAT

    summary = lag_summary(Aggregate(lag_profile=[0.1, 0.9, 0.2], attention_profile=[0.5, 0.2, 0.3]))

    assert summary["axis_caveat"] == GROUP_DELAY_CAVEAT
    assert summary["kl_argmax_lag_step"] == 1
    assert summary["attention_argmax_lag_step"] == 0


def test_the_lag_report_is_empty_when_nothing_was_collected() -> None:
    assert lag_summary(Aggregate()) == {}


def test_the_per_lag_anchor_counts_use_the_models_own_floored_mask(
    trained_task, stub_batch
) -> None:
    """The model's ``build_lag_mask``, not the attention module's: this cell floors it with
    ``lag_floor``, so reading the module's would describe a support the attention was never
    computed over."""
    model = trained_task.orig_model
    _outputs, _tf, _weight, _target, _mask, support = _dense_pieces(trained_task, stub_batch)

    validity = model.build_lag_mask(support.shape[1], device=support.device)
    counts = lag_anchor_counts(support, validity)

    assert counts.shape == (BATCH, model.max_lag + 1)
    # Short lags exist at more anchors than long ones wherever the floor does not already cover
    # the whole lag window, which is the case at this tiny geometry.
    assert float(counts[0, 0]) >= float(counts[0, -1])


def test_the_raw_attribution_sums_to_the_samples_own_kl(trained_task, stub_batch) -> None:
    r"""$\sum_\ell \widetilde K_{t,\ell} = K_t$, exactly rather than in expectation, because the
    attention probabilities carry no dropout. Asserted per sample, which is the stronger claim."""
    readout = evaluate_batch(trained_task, stub_batch, num_samples=1)

    assert torch.allclose(
        readout.lag_profile.sum(dim=1),
        readout.columns["source_conditioned_kl_raw"],
        rtol=1e-5,
    )
    assert float(readout.columns["source_conditioned_kl_raw"].min()) > 0.0, (
        "a zero KL would make the identity hold vacuously"
    )


def test_the_support_correction_recovers_an_argmax_the_raw_profile_gets_wrong() -> None:
    """The known-answer case, written on the arithmetic rather than on a model. Every lag carries
    the same attribution *per contributing anchor* except the last, which carries 5% more -- so the
    truth peaks there. Dividing every bin by the common anchor total scales each one by its own
    support instead, which moves the peak short."""
    seq_len, n_lags = 120, 91
    support = torch.zeros(1, seq_len)
    support[:, 30:] = 1.0
    validity = (
        torch.arange(seq_len)[:, None] - torch.arange(n_lags)[None, :] >= 0
    )
    per_anchor = torch.ones(n_lags)
    per_anchor[-1] = 1.05
    lag_map = validity.to(torch.float32)[None, :, :] * per_anchor[None, None, :]

    raw, corrected, counts = lag_profiles(lag_map, support, validity)

    assert int(raw.argmax(dim=1)[0]) < n_lags - 1, "the raw profile peaks short"
    assert int(corrected.argmax(dim=1)[0]) == n_lags - 1
    assert torch.allclose(corrected[0], per_anchor, atol=1e-5)
    assert float(counts[0, -1]) < float(counts[0, 0])


# =================================================================================================
# Batch helpers
# =================================================================================================
def test_the_batch_size_comes_from_a_tensor_field(stub_batch) -> None:
    assert batch_size_of(stub_batch) == BATCH
    assert batch_size_of({"fhr_st": torch.zeros(5, 4, 36)}) == 5
    assert batch_size_of(object()) == 0


def test_guids_are_read_as_strings_and_default_to_unknown() -> None:
    """``guid`` survives collation as a ``list[str]``, never a tensor."""
    assert batch_guids({"fhr_st": None, "guid": ["a", "b"]}, 2) == ["a", "b"]
    assert batch_guids({"guid": torch.tensor([7, 8])}, 2) == ["7", "8"]
    assert batch_guids({"fhr_st": None}, 2) == ["unknown", "unknown"]


# =================================================================================================
# The whole loop
# =================================================================================================
def test_the_evaluation_loop_assembles_every_reported_section(trained_task) -> None:
    loader = _OneBatchLoader([make_stub_batch(seed=0), make_stub_batch(seed=1, guid_prefix="REC")])

    results = evaluate(trained_task, loader, num_samples=2)

    assert results["n_batches"] == 2
    assert results["n_samples"] == 2 * BATCH
    assert results["units"] == metrics.NORMALISED_UNIT
    assert set(results) >= {
        "readouts", "latent_health", "lag", "per_recording", "verdicts", "num_mc_samples",
        "anchor_geometry", "calibration", "controls",
    }
    assert results["readouts"]["pred_gap"] == pytest.approx(
        results["readouts"]["nll_base_block"] - results["readouts"]["nll_full_block"], rel=1e-5
    )
    assert len(results["verdicts"]) == len(metrics.VERDICT_ORDER)


def test_the_run_records_the_geometry_its_numbers_were_produced_at(trained_task) -> None:
    """A figure or a table that did not say which geometry it was produced at would be unreadable
    against the training CSV: $A_{\\max}$ differs by a factor of the stride between them."""
    model = trained_task.orig_model
    loader = _OneBatchLoader([make_stub_batch()])

    record = evaluate(trained_task, loader, num_samples=1)["anchor_geometry"]

    assert (record["anchor_phase"], record["anchor_stride"]) == DENSE_ANCHOR_GEOMETRY
    assert record["training_stride"] == int(model.anchor_stride)
    assert record["training_stride"] != record["anchor_stride"]
    assert record["anchors_per_sample_expected"] == (
        model.geometry.t_valid - model.warmup_period
    )
    assert record["block_width"] == model.horizon * model.decoder_out_channels


def test_the_geometry_guard_passes_on_a_real_pass(trained_task) -> None:
    """End to end, through the aggregation and the registry: the guard has to be decidable from
    what a pass actually reports, not only from hand-built numbers."""
    results = evaluate(trained_task, _OneBatchLoader([make_stub_batch()]), num_samples=1)

    statuses = {entry["name"]: entry["status"] for entry in results["verdicts"]}
    assert statuses["anchor_geometry_intact"] == "PASS"


def test_the_clock_criterion_is_inconclusive_under_the_shipped_unset_threshold(
    trained_task
) -> None:
    """And still reports the measurement, which is the whole point of shipping it unset."""
    results = evaluate(trained_task, _OneBatchLoader([make_stub_batch()]), num_samples=1)

    entry = next(
        record for record in results["verdicts"]
        if record["name"] == "coupling_exceeds_availability_clock"
    )
    assert entry["status"] == "INCONCLUSIVE"
    assert "coupling_minus_clock_nats" in entry["values"]
    assert "coupling_minus_clock" in results["readouts"]


def test_the_evaluation_loop_leaves_the_task_as_it_found_it(trained_task) -> None:
    """It flips the module into evaluation mode; a training loop that borrowed it back in
    ``train`` mode would silently start running dropout-free."""
    trained_task.train()

    evaluate(trained_task, _OneBatchLoader([make_stub_batch()]), num_samples=1)

    assert trained_task.training is True


def test_batches_too_small_to_derange_are_skipped_and_counted(trained_task) -> None:
    loader = _OneBatchLoader([make_stub_batch(batch=1), make_stub_batch(batch=2)])

    results = evaluate(trained_task, loader, num_samples=1)

    assert results["n_batches"] == 1
    assert results["n_batches_skipped_too_small"] == 1


def test_the_tiny_geometry_exercises_the_gated_model(trained_task) -> None:
    """Non-vacuity for the whole file: an ungated model has no keep-index, no tertiles and no
    source warmth, so most of the surface asserted above would be trivial."""
    model = trained_task.orig_model

    assert model.target_gate is not None
    assert int(model.decoder_out_channels) < int(TINY_KWARGS["c_y"])
