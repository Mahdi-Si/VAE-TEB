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
* the KL, over the KL support, which is a **different** anchor set and therefore a different
  weight vector -- a parity test using one weighting for both would pass with the KL's aggregation
  broken.

Both sides seed the global RNG identically because each runs its own forward and the
reparameterised latent is stochastic: unseeded, the two would be different draws of the same
quantity and could only be compared at a tolerance loose enough to hide a real drift.

The last test is what makes the rest non-vacuous: a deliberately **mis-weighted** recombination --
the unweighted mean, which is what an implementation that forgot the anchor counts would compute
-- must disagree. On a fixture where every sample has the same anchor count it would not, so the
batch is built with a gap in one sample and not the other.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.eval.metrics import evaluate_batch

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
    """A batch whose two samples have *different* anchor counts.

    The whole point of the weighting is that a long segment counts for more than a short one, so a
    fixture where every sample contributes equally cannot tell a weighted recombination from an
    unweighted one. The second sample is gapped over a stretch inside the trained anchor range.
    """
    batch = make_stub_batch(seed=11)
    batch.weight[1, STUB_GAP_STEP - 3 : STUB_GAP_STEP + 1] = 0.0
    return batch


def kl_support_counts(module, batch) -> torch.Tensor:
    """Masked anchor counts per sample, from the same two masks the loss builds.

    Rebuilt through ``forecast_mask`` and ``kl_mask`` rather than reusing the reconstruction's
    counts: the KL's support is derived from the forecast mask but is not the same set, and using
    one for both is precisely the mistake this file exists to catch.
    """
    from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask

    model = module.orig_model
    forecast, _coverage = forecast_mask(
        batch.weight, model.geometry, coverage_floor=model.coverage_floor
    )
    return kl_mask(forecast, model.geometry).sum(dim=1)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    """The loss's own reduction: $\\sum_b n_b v_b / \\sum_b n_b$."""
    return float((values * weights).sum() / weights.sum().clamp_min(1.0))


def _both_sides(module, batch):
    """Return ``(readout, metrics)`` from two forwards seeded identically."""
    torch.manual_seed(_SEED)
    readout = evaluate_batch(module, batch, num_samples=1)
    torch.manual_seed(_SEED)
    _loss, metrics = module.compute_loss_and_metrics(batch, 0, "val")
    return readout, metrics


# =============================================================================
# The three totals
# =============================================================================
@pytest.mark.parametrize("name", ["nll_base_block", "nll_full_block"])
def test_the_reconstruction_readouts_recombine_into_the_objectives_totals(
    trained_task, uneven_batch, name: str
) -> None:
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(readout.columns[name], readout.n_anchors)

    assert recombined == pytest.approx(float(metrics[name]), rel=1e-5)


def test_the_kl_readout_recombines_into_the_objectives_kl(trained_task, uneven_batch) -> None:
    """The KL's anchor support is not the reconstruction's, so this is a second weight vector and
    not a repeat of the test above."""
    readout, metrics = _both_sides(trained_task, uneven_batch)

    recombined = weighted_mean(
        readout.columns["source_conditioned_kl_raw"],
        kl_support_counts(trained_task, uneven_batch),
    )

    assert recombined == pytest.approx(float(metrics["source_conditioned_kl_raw"]), rel=1e-5)


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


# =============================================================================
# The bound diagnostics, against the model's own
# =============================================================================
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
