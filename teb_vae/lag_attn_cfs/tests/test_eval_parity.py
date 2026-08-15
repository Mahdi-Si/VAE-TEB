r"""The evaluation's readouts *are* the objective's numbers, recombined.

Everything else in this package is plumbing around three quantities -- $D_{\mathrm{base}}$,
$D_{\mathrm{full}}$ and $\bar K$ -- and the entire exercise is worthless if those are not what the
training loop optimised. The collect-and-emit split makes that easy to lose quietly: the readouts
are reduced per sample inside ``evaluate_batch``, written to a table, and read back by analyses
that never see the loss function. Nothing downstream would notice a drift.

So this is the cheapest guard against it. The per-sample values are recombined with their
**anchor-count weights** -- $\sum_b n_b v_b / \sum_b n_b$, which is exactly the reduction the loss
takes -- and compared against what ``compute_loss`` reports on the same batch, for all three
quantities:

* the two reconstruction terms, over the forecast mask's contributing anchors;
* the KL, over the KL support, which is built on a **different grid**: the forecast mask is sparse
  at $(B, A_{\max}, H)$ while the KL support is dense at $(B, T)$, and the second is *derived*
  from the first by a scatter rather than restated.

**The obvious control for the paragraph above does not work here, and its substitute is stronger.**
Swapping the two weightings cannot fail at this cell's geometry: ``kl_mask`` scatters the
contributing-anchor indicator into the dense grid, so
the two weight *vectors* are numerically identical however the anchor axis is laid out -- a swap is
a no-op and a test asserting otherwise would be asserting a falsehood. What *is* discriminating,
and is what the criterion is protecting against, is a **restated** support: an implementation that
wrote the KL's support as $\mathbb{1}[F \le t < T - H]\,v_t$ instead of deriving it from the
forecast mask keeps every anchor the coverage floor dropped, charges $\beta \cdot \mathrm{KL}$ on
anchors with no reconstruction term pulling the posterior off the prior, and those anchors cluster
immediately before every gap. That control is below, and it disagrees.

Both sides seed the global RNG identically because each runs its own forward and the
reparameterised latent is stochastic: unseeded, the two would be different draws of the same
quantity and could only be compared at a tolerance loose enough to hide a real drift.

The last test is what makes the rest non-vacuous: a deliberately **mis-weighted** recombination --
the unweighted mean, which is what an implementation that forgot the anchor counts would compute
-- must disagree. Under the dense evaluation geometry every sample decodes the same anchors, so a
**gap** is the only thing that makes the contributing counts differ at all, and a batch without one
would make this whole file pass vacuously.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.eval.metrics import DENSE_ANCHOR_GEOMETRY, evaluate_batch, model_inputs

from .conftest import BATCH, STUB_GAP_STEP, make_stub_batch

#: Seeded identically on both sides of every comparison; see the module docstring.
_SEED = 4


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior.

    Load-bearing: at initialisation the delta heads are zero, so the posterior *is* the prior,
    every KL is exactly zero, and a KL parity assertion would hold on any implementation at all.
    """
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    return module


@pytest.fixture
def uneven_batch():
    """A batch whose two samples have *different* contributing-anchor counts.

    The whole point of the weighting is that a long segment counts for more than a short one, so a
    fixture where every sample contributes equally cannot tell a weighted recombination from an
    unweighted one. Under the dense geometry both samples decode the same anchor set, so the
    difference has to come from validity: the second sample is gapped over a stretch inside the
    trained anchor range, which the coverage floor turns into several dropped anchors.
    """
    batch = make_stub_batch(seed=11)
    batch.weight[1, STUB_GAP_STEP - 3 : STUB_GAP_STEP + 1] = 0.0
    return batch


def kl_support_counts(module, batch) -> torch.Tensor:
    """Masked anchor counts per sample, from the same two masks the loss builds.

    Rebuilt through ``forecast_mask`` and ``kl_mask`` at the anchor set the dense forward returns,
    rather than reusing the reconstruction's counts: the KL's support is *derived* from the
    forecast mask and lives on the dense $(B, T)$ grid, and deriving it here is what makes this a
    second derivation rather than a second reading of one number.
    """
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

    model = module.orig_model
    y_st, y_ph, u_stream, _target_features, weight = model_inputs(module, batch)
    phase, stride = DENSE_ANCHOR_GEOMETRY
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=phase, anchor_stride=stride)
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    forecast, _coverage = forecast_mask(
        weight, model.geometry, coverage_floor=model.coverage_floor,
        anchors=anchors, anchor_valid=anchor_valid,
    )
    return kl_mask(
        forecast, model.geometry, anchors=anchors, anchor_valid=anchor_valid
    ).sum(dim=1)


def restated_support_counts(module, batch) -> torch.Tensor:
    r"""The support a *restated* rule would produce: $\mathbb{1}[F \le t < T - H]\,v_t$.

    Not what the pipeline computes, and deliberately so -- it is the discriminating negative
    control of the module docstring. It keeps every anchor the coverage floor dropped, so it
    exceeds the derived support wherever a gap reaches into a forecast window.
    """
    model = module.orig_model
    _y_st, _y_ph, _u, _target_features, weight = model_inputs(module, batch)
    valid = (weight >= 1.0).to(weight.dtype)
    steps = torch.arange(weight.shape[1], device=weight.device)
    window = (
        (steps >= model.warmup_period) & (steps < model.geometry.t_valid)
    ).to(weight.dtype)
    return (valid * window[None, :]).sum(dim=1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    """The loss's own reduction: $\\sum_b n_b v_b / \\sum_b n_b$."""
    return float((values * weights).sum() / weights.sum().clamp_min(1.0))


def _both_sides(module, batch):
    """Return ``(readout, metrics)`` from two forwards seeded identically.

    ``'val'`` rather than ``'train'``: it is one of ``DENSE_STAGES``, so the task's own
    ``resolve_anchor_geometry`` returns exactly the $(0, 1)$ the evaluation decodes at, and the two
    sides are then comparing one anchor set rather than a tile grid against a dense range.
    """
    torch.manual_seed(_SEED)
    readout = evaluate_batch(module, batch, num_samples=1)
    torch.manual_seed(_SEED)
    _loss, metrics = module.compute_loss_and_metrics(batch, 0, "val")
    return readout, metrics


# =================================================================================================
# The three totals
# =================================================================================================
@pytest.mark.parametrize("name", ["nll_base_block", "nll_full_block"])
def test_the_reconstruction_readouts_recombine_into_the_objectives_totals(
    trained_task, uneven_batch, name: str
) -> None:
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(readout.columns[name], readout.n_anchors)

    assert recombined == pytest.approx(float(metrics[name]), rel=1e-5)


def test_the_kl_readout_recombines_into_the_objectives_kl(trained_task, uneven_batch) -> None:
    """The KL's support is derived on the dense grid from the sparse forecast mask, so this is a
    second derivation and not a repeat of the test above."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(
        readout.columns["source_conditioned_kl_raw"],
        kl_support_counts(trained_task, uneven_batch),
    )

    assert recombined == pytest.approx(float(metrics["source_conditioned_kl_raw"]), rel=1e-5)


def test_the_derived_and_restated_supports_are_not_the_same_set(
    trained_task, uneven_batch
) -> None:
    """Non-vacuity for the control below, and the fact the whole KL-support argument rests on: the
    coverage floor drops anchors a restated rule keeps, and it drops them exactly where a gap
    reaches into a forecast window."""
    derived = kl_support_counts(trained_task, uneven_batch)
    restated = restated_support_counts(trained_task, uneven_batch)

    assert bool((derived < restated).all()), (derived.tolist(), restated.tolist())


def test_a_restated_kl_support_disagrees_with_the_objective(
    trained_task, uneven_batch
) -> None:
    """The discriminating control. An implementation that wrote the KL's support as
    $\\mathbb{1}[F \\le t < T - H]\\,v_t$ rather than deriving it from the forecast mask would
    charge the KL on anchors carrying no reconstruction term, and those anchors cluster immediately
    before every gap -- so the artifact reads as coupling fading exactly where the signal degrades.
    """
    readout, metrics = _both_sides(trained_task, uneven_batch)

    mis_supported = weighted_mean(
        readout.columns["source_conditioned_kl_raw"],
        restated_support_counts(trained_task, uneven_batch),
    )

    assert mis_supported != pytest.approx(
        float(metrics["source_conditioned_kl_raw"]), rel=1e-5
    )


def test_the_fixture_actually_has_unequal_anchor_counts(trained_task, uneven_batch) -> None:
    """Non-vacuity for the mis-weighting test below: on equal counts the weighted and unweighted
    reductions coincide and nothing could distinguish them."""
    readout = evaluate_batch(trained_task, uneven_batch, num_samples=1)

    counts = [float(value) for value in readout.n_anchors]
    assert len(counts) == BATCH
    assert counts[0] != counts[1]


@pytest.mark.parametrize(
    "name", ["nll_base_block", "nll_full_block", "source_conditioned_kl_raw"]
)
def test_a_mis_weighted_recombination_disagrees_with_the_objective(
    trained_task, uneven_batch, name: str
) -> None:
    """The unweighted mean is what an implementation that dropped the anchor counts would compute.
    It must **not** reproduce the objective's total -- otherwise every assertion above would pass
    on a broken aggregation."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    unweighted = float(readout.columns[name].mean())

    assert unweighted != pytest.approx(float(metrics[name]), rel=1e-5)


# =================================================================================================
# This target domain's own columns, against the model's own reductions
# =================================================================================================
@pytest.mark.parametrize(
    "name",
    ["pred_gap_st", "pred_gap_ph", "pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi"],
)
def test_the_channel_resolved_gaps_recombine_into_the_models_own(
    trained_task, uneven_batch, name: str
) -> None:
    r"""The five channel-axis gaps are the model's own ``_resolved_forecast_gaps`` reductions with
    the batch sum opened up, so each must recombine into the scalar ``compute_loss`` reports.

    They are the readouts most likely to drift, because each is a partial sum of ``pred_gap`` and a
    second copy of the summation is a second opportunity for one of them to stop being one.
    """
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(readout.columns[name], readout.n_anchors)

    assert recombined == pytest.approx(float(metrics[name]), rel=1e-4)


def test_the_two_geometry_guards_recombine_into_the_models_own(
    trained_task, uneven_batch
) -> None:
    """Both are constants of the geometry rather than statistics, so they recombine exactly rather
    than to a tolerance -- which is what makes a value off either one a structural finding."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    assert float(readout.columns["target_warm_frac"][0]) == float(metrics["target_warm_frac"])
    assert float(readout.columns["anchors_per_sample"].mean()) == float(
        metrics["anchors_per_sample"]
    )


# =================================================================================================
# The bound diagnostics, against the model's own
# =================================================================================================
@pytest.mark.parametrize(
    "name", ["mean_logvar_prior", "mean_logvar_post", "logvar_prior_floor_frac"]
)
def test_the_prior_variance_diagnostics_recombine_into_the_models_own(
    trained_task, uneven_batch, name: str
) -> None:
    """These are the readouts a coupling number's validity rests on, and the evaluation recomputes
    them per sample rather than reading the model's batch-level scalars. Recombined over the KL
    support they must be the same numbers the trainer logs."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(
        readout.columns[name], kl_support_counts(trained_task, uneven_batch)
    )

    assert recombined == pytest.approx(float(metrics[name]), rel=1e-4)


def test_the_prior_rate_readout_recombines_into_the_objectives_own_term(
    trained_task, uneven_batch
) -> None:
    """The per-sample rate column against the batch-level reduction the objective computes. The
    evaluation recomputes the expression rather than importing the loss function, because the loss
    reduces a whole batch to one scalar and this table needs the quantity per sample -- and two
    copies of one formula is exactly where a sign or a factor drifts."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(
        readout.columns["prior_rate"], kl_support_counts(trained_task, uneven_batch)
    )

    assert recombined == pytest.approx(float(metrics["prior_rate"]), rel=1e-4)


def test_the_saturation_fractions_recombine_into_the_models_flat_means(
    trained_task, uneven_batch
) -> None:
    """The ``_raw`` framing is the model's own: a flat mean over *every* element, warm-up prefix
    and untrained tail included. The ``_masked`` one beside it is this package's, and the two may
    legitimately differ -- which is why both are emitted."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    for column, reported in (
        ("mu_prior_sat_frac_raw", "mu_prior_sat_frac"),
        ("delta_mu_sat_frac_raw", "delta_mu_sat_frac"),
    ):
        # Every sample carries the same element count in the raw framing, so the flat mean over
        # samples is the model's own scalar.
        assert float(readout.columns[column].mean()) == pytest.approx(
            float(metrics[reported]), rel=1e-5
        )
