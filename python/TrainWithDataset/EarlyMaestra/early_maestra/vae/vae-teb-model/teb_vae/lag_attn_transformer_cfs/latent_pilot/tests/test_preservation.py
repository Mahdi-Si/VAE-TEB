r"""The preservation pass, on the real net.

**Execution-machine tests**, like the extraction and model-contract suites they sit beside: every
one of these decodes a real forecast. They need no checkpoint file, no clinical data and no GPU --
the model is the committed tiny geometry and the loader is a list of the suite's synthetic stub
batches::

    python -m pytest teb_vae/lag_attn_transformer_cfs/latent_pilot/tests/test_preservation.py -q

What they establish: a model that did not change measures the same forecast twice; a model whose
mean heads did change moves ``mse_full`` while leaving the prior branch bit-identical; the Monte
Carlo draws are common random numbers keyed to the examples; and an anchor whose scored coefficient
reaches past delivery is excluded here for exactly the reason it is excluded from the latent tables.

The gate's own arithmetic -- tolerances, the zero-baseline rule, the support check -- is hand-checked
without a model in ``tests/logic/test_gates.py``.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, evaluate
from teb_vae.lag_attn_transformer_cfs.latent_pilot import model as pilot_model
from teb_vae.lag_attn_transformer_cfs.tests.conftest import (
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_warmup_kwargs,
)

#: Trim and window, as in the extraction suite. The stub epochs below are chosen against them.
TRIM_MINUTES = 1.0
PRESERVATION_HOURS = 3.0

#: Segment starts an hour before delivery: every decoded anchor lands inside the window and every
#: scored coefficient stays well before the landmark.
INSIDE_EPOCHS = (-3600.0, -3000.0)

#: A segment whose anchors are inside the window but whose forecast reaches the landmark. At
#: ``trim_minutes=1`` the anchor sits at ``epoch + 60 + 4a`` seconds and its furthest scored
#: coefficient at ``epoch + 60 + 4(a + H)``, so with ``epoch = -100`` and $H = 4$ the anchors from
#: $a = 6$ on are scored against a coefficient recorded at or after delivery.
POST_DELIVERY_EPOCH = -100.0


@pytest.fixture
def loaded():
    """A loaded-checkpoint bundle around the tiny model, without a checkpoint file."""
    kwargs = tiny_warmup_kwargs(anchor_stride=TINY_STRIDE)
    task = make_task(model_kwargs=kwargs)
    pilot_model.freeze_for_pilot(task.orig_model)
    blob = {
        "model_kwargs": dict(kwargs),
        "model_class": type(task.orig_model).__name__,
        "epoch": 0,
    }
    config = {
        "dataset_config": {
            "stat_path": "/synthetic/stats.hdf5",
            "dataloader_config": {"dataset_kwargs": {"trim_minutes": TRIM_MINUTES}},
        }
    }
    return types.SimpleNamespace(
        task=task,
        model=task.orig_model,
        blob=blob,
        config=config,
        checkpoint_path="/synthetic/tiny.ckpt",
        digest="synthetic-digest",
        geometry=pilot_model.geometry_record(task.orig_model, blob, config),
    )


def _loader(seed: int = 0, epochs=INSIDE_EPOCHS):
    """A one-batch loader whose segments sit inside the preserved window."""
    batch = make_stub_batch(batch=len(epochs), seed=seed)
    batch.epoch = torch.tensor(list(epochs), dtype=torch.float32)
    batch.guid = [f"SYNTH-{index}" for index in range(len(epochs))]
    return [batch]


def _guids(epochs=INSIDE_EPOCHS):
    """The recordings a loader built from ``epochs`` yields."""
    return [f"SYNTH-{index}" for index in range(len(epochs))]


def _outcomes(epochs=INSIDE_EPOCHS):
    """One healthy and one adverse recording, so the healthy-only readout has a stratum."""
    return {guid: index % 2 for index, guid in enumerate(_guids(epochs))}


def _measure(loaded, *, epochs=INSIDE_EPOCHS, **overrides):
    """Run one preservation pass at this file's window settings."""
    return evaluate.preservation_pass(
        loaded,
        _loader(epochs=epochs),
        guids=_guids(epochs),
        outcomes=_outcomes(epochs),
        preservation_hours=PRESERVATION_HOURS,
        **overrides,
    )


def _perturb(model, amount: float) -> None:
    """Move every posterior mean-output weight, which is the whole trainable set."""
    with torch.no_grad():
        for _name, parameter in pilot_model.mean_head_parameters(model):
            parameter.add_(amount * torch.ones_like(parameter))


# =============================================================================
# The measurement itself
# =============================================================================
def test_the_pass_is_deterministic_without_the_monte_carlo_columns(loaded):
    """Two readings of one model must agree exactly, or a gate is decided by noise."""
    first = _measure(loaded)
    second = _measure(loaded)

    # equal_nan, because the Monte Carlo columns are absent-as-NaN in this pass by design and a
    # plain comparison would report a bit-identical reading as a difference.
    assert np.array_equal(first.values, second.values, equal_nan=True)
    assert first.record["support_digest"] == second.record["support_digest"]
    assert first.record["mse_full"] == second.record["mse_full"]


def test_the_monte_carlo_columns_are_absent_rather_than_zero_when_not_requested(loaded):
    reading = _measure(loaded)

    for name in ("nll_full", "nll_base"):
        assert np.isnan(reading.record[name])
    assert reading.record["mc_draws"] is None
    # The gated columns are finite whether or not the draws were taken.
    assert reading.record["finite"]
    assert np.isfinite(reading.record["mse_full"])


def test_the_forecast_is_scored_per_coefficient_over_the_objective_s_own_support(loaded):
    """Every retained anchor carries a forecast term, and only those are reduced."""
    reading = _measure(loaded)
    retained = data.retained(reading.frame)

    assert retained["contributing"].all()
    assert len(retained) < len(reading.frame), "the stub weight gap must exclude something"
    assert reading.record["n_retained_anchors"] == len(retained)
    assert reading.record["n_recordings"] == retained[data.GUID_COLUMN].nunique()
    assert reading.record["mse_full"] > 0.0


def test_recordings_weigh_the_same_however_many_anchors_they_brought(loaded):
    """The reduction is anchors, then segments, then recordings, as everywhere else."""
    reading = _measure(loaded)
    pooled = float(reading.recordings["mse_full"].mean())

    assert reading.record["mse_full"] == pytest.approx(pooled)
    assert set(reading.recordings[data.GUID_COLUMN]) <= set(_guids())


# =============================================================================
# Identity and deliberate degradation
# =============================================================================
def test_an_unchanged_model_passes_its_own_gate(loaded):
    baseline = _measure(loaded)

    result = evaluate.gate_decision(
        baseline.record,
        dict(baseline.record),
        forecast_mse_max_increase=0.10,
        saturation_max_increase_pp=5.0,
    )

    assert result.passed
    assert result.reasons == []


def test_moving_the_mean_heads_moves_the_full_branch_and_not_the_prior_branch(loaded):
    """The invariant and the measurement, from one pair of real readings."""
    before = _measure(loaded)
    _perturb(loaded.model, 0.05)
    after = _measure(loaded)

    assert before.record["support_digest"] == after.record["support_digest"]
    assert after.record["mse_full"] != before.record["mse_full"]
    # mu_prior is untouched by this adaptation, so its forecast is bit-identical.
    assert after.record["mse_base"] == before.record["mse_base"]


def test_a_saturating_perturbation_fails_the_declared_saturation_gate(loaded):
    """The delta head is zero-initialised, so the baseline sits at no saturation at all and a
    large enough update puts every coordinate on the bound."""
    before = _measure(loaded)
    assert before.record["delta_mu_sat_pp"] == pytest.approx(0.0)

    _perturb(loaded.model, 50.0)
    after = _measure(loaded)

    assert after.record["delta_mu_sat_pp"] > 50.0
    result = evaluate.gate_decision(
        before.record,
        after.record,
        forecast_mse_max_increase=0.10,
        saturation_max_increase_pp=5.0,
    )
    assert not result.passed
    assert any("saturation" in reason for reason in result.reasons)


def test_two_readings_of_different_supports_are_refused_rather_than_compared(loaded):
    """A candidate measured over other anchors is a broken measurement, not a failed gate."""
    inside = _measure(loaded)
    shifted = _measure(loaded, epochs=(-3600.0, -2400.0))

    assert inside.record["support_digest"] != shifted.record["support_digest"]
    with pytest.raises(evaluate.GateEvaluationError, match="different anchors"):
        evaluate.gate_decision(
            inside.record,
            shifted.record,
            forecast_mse_max_increase=0.10,
            saturation_max_increase_pp=5.0,
        )


# =============================================================================
# The support rules, at the same definitions the latent tables use
# =============================================================================
def test_an_anchor_scored_past_delivery_is_excluded_for_that_reason(loaded):
    """The anchor is inside the window; its furthest scored coefficient is not."""
    epochs = (-3600.0, POST_DELIVERY_EPOCH)
    reading = _measure(loaded, epochs=epochs)
    exclusions = data.anchor_exclusion_counts(reading.frame)

    assert exclusions.get(data.EXCLUDED_POST_DELIVERY, 0) > 0
    excluded = reading.frame[
        reading.frame[data.EXCLUSION_COLUMN] == data.EXCLUDED_POST_DELIVERY
    ]
    assert (excluded[data.HOURS_COLUMN] > 0.0).all()
    assert (excluded[data.LATEST_SCORED_COLUMN] >= 0.0).all()
    assert data.retained(reading.frame)[data.LATEST_SCORED_COLUMN].max() < 0.0


def test_only_the_subset_s_recordings_reach_the_measurement(loaded):
    """A gate measured on more recordings than it named is a gate on another population."""
    reading = evaluate.preservation_pass(
        loaded,
        _loader(),
        guids=[_guids()[0]],
        outcomes=_outcomes(),
        preservation_hours=PRESERVATION_HOURS,
    )

    assert set(reading.frame[data.GUID_COLUMN]) == {_guids()[0]}
    assert reading.record["n_requested_recordings"] == 1
    assert reading.record["n_recordings"] == 1


def test_a_subset_the_loader_never_yields_refuses_rather_than_measuring_nothing(loaded):
    with pytest.raises(evaluate.GateEvaluationError, match="no anchor inside"):
        evaluate.preservation_pass(
            loaded,
            _loader(),
            guids=["NOT-IN-THIS-SPLIT"],
            outcomes={},
            preservation_hours=PRESERVATION_HOURS,
        )


# =============================================================================
# Common random numbers
# =============================================================================
def test_the_monte_carlo_draws_are_shared_between_two_passes_of_one_seed(loaded):
    """Before and after must see identical noise, or the NLL difference is partly the draw."""
    first = _measure(loaded, mc_draws=2, mc_seed=7)
    second = _measure(loaded, mc_draws=2, mc_seed=7)
    other = _measure(loaded, mc_draws=2, mc_seed=8)

    assert np.array_equal(first.values, second.values)
    assert first.record["nll_full"] == second.record["nll_full"]
    assert other.record["nll_full"] != first.record["nll_full"]
    assert np.isfinite(first.record["nll_full"])
    assert np.isfinite(first.record["nll_base"])


def test_the_kl_diagnostics_are_reported_beside_the_gates(loaded):
    reading = _measure(loaded)

    assert reading.record["n_kl_anchors"] > 0
    assert np.isfinite(reading.record["source_conditioned_kl"])
    assert np.isfinite(reading.record["prior_rate"])
    assert 0.0 <= reading.record["kld_active_frac"] <= 1.0


# =============================================================================
# Persistence
# =============================================================================
def test_a_reading_round_trips_through_a_run_directory(loaded, tmp_path):
    reading = _measure(loaded)
    evaluate.save_preservation(reading, tmp_path, name="frozen")

    record = evaluate.load_preservation_record(tmp_path, name="frozen")

    assert record["support_digest"] == reading.record["support_digest"]
    assert record["mse_full"] == pytest.approx(reading.record["mse_full"])


def test_a_missing_reading_names_the_stage_that_produces_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="before the first candidate"):
        evaluate.load_preservation_record(tmp_path, name="frozen")
