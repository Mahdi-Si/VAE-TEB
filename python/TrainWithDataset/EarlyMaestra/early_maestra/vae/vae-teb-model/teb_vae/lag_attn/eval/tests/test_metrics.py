r"""Tests for the pure metric functions.

Two styles, matched to what each can catch. The forecast and uplift functions are checked
against expectations computed **by hand** on tensors small enough to reason about, because that
is the only way to catch a reduction that is self-consistently wrong. The KL functions are
checked against the *model's own* readouts, because a hand-computed KL would just be a second
copy of the formula under test.

The KL cases run on a ``perturb_posterior`` model. On an untouched one the posterior equals the
prior exactly and every KL is $0$, so every assertion passes on a model that is entirely wrong.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import masks, metrics
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import SEQ_LEN, SHIPPED_KWARGS, make_stub_batch


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
def test_pooled_mean_counts_entries_not_mask_cells() -> None:
    r"""The channel factor is what makes the denominator match ``compute_loss``.

    Dropping it would multiply every reported loss by $C$ -- large enough to be obviously wrong
    and small enough to be mistaken for a scale convention.
    """
    values = torch.full((2, 3, 4, 5), 2.0)
    mask = torch.ones(2, 3, 4, 1)
    # sum = 2 * 2*3*4*5 = 240; denom = (2*3*4) * 5 = 120 -> mean 2.0, the value itself.
    assert float(metrics.masked_pooled_mean(values, mask)) == pytest.approx(2.0)


def test_pooled_mean_is_a_mask_weighted_mean() -> None:
    """Half the anchors masked out must leave the mean of a constant field unchanged."""
    values = torch.full((1, 4, 2, 3), 7.0)
    mask = torch.zeros(1, 4, 2, 1)
    mask[:, :2] = 1.0
    assert float(metrics.masked_pooled_mean(values, mask)) == pytest.approx(7.0)


def test_per_sample_mean_divides_each_sample_by_its_own_mask() -> None:
    """The per-sample form is a *different* number from the pooled one when densities differ."""
    values = torch.zeros(2, 2, 1, 1)
    values[0] = 4.0
    values[1] = 10.0
    mask = torch.ones(2, 2, 1, 1)
    mask[1, 1] = 0.0  # sample 1 has half the entries

    per_sample = metrics.masked_per_sample_mean(values, mask)
    assert per_sample.tolist() == pytest.approx([4.0, 10.0])
    # Pooled weights sample 0 twice as heavily: (4+4+10)/3.
    assert float(metrics.masked_pooled_mean(values, mask)) == pytest.approx(18.0 / 3.0)


def test_a_fully_masked_sample_yields_nan_not_zero() -> None:
    """Zero is a legitimate metric value, so it must not double as "no data".

    A zero here would be indistinguishable from a perfect forecast and would drag every
    downstream mean toward it.
    """
    values = torch.full((2, 3, 2, 4), 5.0)
    mask = torch.ones(2, 3, 2, 1)
    mask[1] = 0.0
    result = metrics.masked_per_sample_mean(values, mask)
    assert float(result[0]) == pytest.approx(5.0)
    assert math.isnan(float(result[1]))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def test_mse_per_element_loss_is_the_squared_error_itself() -> None:
    """Under ``'mse'`` the log-variance head must not enter the loss at all."""
    squared = torch.tensor([1.0, 4.0])
    logvar = torch.tensor([100.0, -100.0])  # would dominate if it leaked in
    assert torch.equal(
        metrics.per_element_loss(squared, logvar, likelihood="mse", sigma_obs=1.0), squared
    )


def test_gaussian_nll_matches_the_closed_form_by_hand() -> None:
    r"""$\ell = \tfrac{1}{2} e^{2} \sigma^{-2} + \tfrac{1}{2}\log\sigma^2$, constant dropped."""
    squared = torch.tensor([4.0])
    logvar = torch.tensor([math.log(2.0)])
    expected = 0.5 * 4.0 / 2.0 + 0.5 * math.log(2.0)
    got = metrics.per_element_loss(
        squared, logvar, likelihood="gaussian_nll", sigma_obs="learned"
    )
    assert float(got[0]) == pytest.approx(expected)


def test_a_scalar_sigma_obs_overrides_the_learned_head() -> None:
    """A scalar ``sigma_obs`` must ignore ``logvar`` entirely, as ``compute_loss`` does."""
    squared = torch.tensor([4.0])
    logvar = torch.tensor([999.0])
    expected = 0.5 * 4.0 / (0.5**2) + 0.5 * math.log(0.5**2)
    got = metrics.per_element_loss(
        squared, logvar, likelihood="gaussian_nll", sigma_obs=0.5
    )
    assert float(got[0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "likelihood,sigma_obs,match",
    [
        ("poisson", 1.0, "unknown likelihood"),
        ("gaussian_nll", "estimated", "must be 'learned'"),
        ("gaussian_nll", -1.0, "must be positive"),
    ],
)
def test_an_unusable_objective_raises(likelihood: str, sigma_obs, match: str) -> None:
    """Mirrors ``compute_loss``'s own validation, so a bad objective fails the same way."""
    with pytest.raises(ValueError, match=match):
        metrics.per_element_loss(
            torch.ones(2), torch.zeros(2), likelihood=likelihood, sigma_obs=sigma_obs
        )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def test_forecast_metrics_split_the_channel_axis_at_the_supplied_index() -> None:
    """The split is a parameter derived from the batch, never a hardcoded $43$.

    A perfect scattering block and a wrong phase block must show up as exactly that.
    """
    batch, anchors, horizon, split, n_phase = 1, 2, 2, 3, 2
    target = torch.zeros(batch, anchors, horizon, split + n_phase)
    forecast = target.clone()
    forecast[..., split:] = 2.0  # phase block off by 2 everywhere

    mask = torch.ones(batch, anchors, horizon, 1)
    result = metrics.forecast_metrics(forecast, target, mask, n_scattering=split)

    assert float(result["feat_mse_scattering"][0]) == pytest.approx(0.0)
    assert float(result["feat_mse_phase"][0]) == pytest.approx(4.0)
    # Total is the entry-weighted blend: (3*0 + 2*4) / 5.
    assert float(result["feat_mse_total"][0]) == pytest.approx(8.0 / 5.0)


def test_r2_is_computed_against_the_masked_per_channel_mean() -> None:
    r"""A model predicting each channel's own mean scores $R^2 = 0$, not something higher.

    Against a single scalar mean over all channels, $SS_{\mathrm{tot}}$ would be dominated by
    the offsets *between* channels and this would score near $1$.
    """
    # Two channels with very different offsets and small within-channel variance.
    target = torch.zeros(1, 4, 1, 2)
    target[0, :, 0, 0] = torch.tensor([100.0, 101.0, 99.0, 100.0])
    target[0, :, 0, 1] = torch.tensor([-50.0, -49.0, -51.0, -50.0])
    forecast = torch.zeros_like(target)
    forecast[..., 0] = 100.0
    forecast[..., 1] = -50.0

    mask = torch.ones(1, 4, 1, 1)
    result = metrics.forecast_metrics(forecast, target, mask, n_scattering=1)
    assert float(result["feat_r2_total"][0]) == pytest.approx(0.0, abs=1e-5)


def test_r2_is_one_for_an_exact_forecast() -> None:
    """The other endpoint, so the sign convention is pinned."""
    torch.manual_seed(0)
    target = torch.randn(2, 3, 2, 4)
    mask = torch.ones(2, 3, 2, 1)
    result = metrics.forecast_metrics(target.clone(), target, mask, n_scattering=2)
    assert result["feat_r2_total"].tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_r2_is_nan_when_the_target_has_no_variance_to_explain() -> None:
    """Undefined, not zero: there is no variance to have explained a fraction of."""
    target = torch.full((1, 3, 2, 2), 5.0)
    mask = torch.ones(1, 3, 2, 1)
    result = metrics.forecast_metrics(target.clone(), target, mask, n_scattering=1)
    assert math.isnan(float(result["feat_r2_total"][0]))


def test_horizon_and_anchor_profiles_have_the_geometry_they_claim() -> None:
    """Lengths asserted against the tensor's own shape, never against a literal."""
    batch, anchors, horizon, channels = 2, 5, 3, 4
    torch.manual_seed(0)
    forecast = torch.randn(batch, anchors, horizon, channels)
    target = torch.randn(batch, anchors, horizon, channels)
    mask = torch.ones(batch, anchors, horizon, 1)

    assert metrics.horizon_error_profile(forecast, target, mask).shape == (batch, horizon)
    assert metrics.anchor_error_profile(forecast, target, mask).shape == (batch, anchors)


def test_the_horizon_profile_reports_a_horizon_that_is_actually_worse() -> None:
    """The non-vacuity check: sabotage one horizon step and the profile must name it."""
    forecast = torch.zeros(1, 4, 3, 2)
    target = torch.zeros(1, 4, 3, 2)
    target[:, :, 2, :] = 3.0  # the last horizon step is badly forecast
    mask = torch.ones(1, 4, 3, 1)

    profile = metrics.horizon_error_profile(forecast, target, mask)[0]
    assert profile.tolist() == pytest.approx([0.0, 0.0, 9.0])
    assert int(torch.argmax(profile)) == 2


def test_a_masked_anchor_is_nan_in_the_anchor_profile() -> None:
    """The warm-up anchors are masked by construction and must render as gaps, not zeros."""
    forecast = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.ones(1, 3, 2, 1)
    mask[:, 0] = 0.0
    profile = metrics.anchor_error_profile(forecast, target, mask)[0]
    assert math.isnan(float(profile[0]))
    assert float(profile[1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Band and per-channel resolution
# ---------------------------------------------------------------------------
def _band_inputs(seed: int = 0):
    """A forecast, target and mask with an uneven mask density, plus a tiling partition.

    Uneven density on purpose: with every cell unmasked the weighted-mean identity below holds
    for a wider class of wrong reductions, so a uniform mask would make the test weaker than it
    looks.
    """
    torch.manual_seed(seed)
    forecast = torch.randn(3, 5, 2, 7)
    target = torch.randn(3, 5, 2, 7)
    mask = torch.ones(3, 5, 2, 1)
    mask[0, :2] = 0.0
    mask[2, 4] = 0.0
    partition = {"low": [0, 3, 6], "mid": [1, 4], "high": [2, 5]}
    return forecast, target, mask, partition


def _channel_weighted_blend(bands) -> torch.Tensor:
    """Blend per-band MSEs back together weighted by each band's channel count."""
    first = next(iter(bands.values()))["feat_mse"]
    weighted = torch.zeros_like(first)
    total_channels = 0
    for record in bands.values():
        weighted = weighted + float(record["n_channels"]) * torch.nan_to_num(record["feat_mse"])
        total_channels += int(record["n_channels"])
    return weighted / float(total_channels)


def test_the_channel_weighted_mean_over_bands_reproduces_the_overall_mse() -> None:
    r"""The identity that catches a partition with a gap or an overlap.

    Each band divides by its own channel count against the *same* mask sum, so weighting by
    $C_g$ and dividing by $\sum_g C_g$ must give the total back exactly. A partition that dropped
    a channel, or counted one twice, cannot satisfy it.
    """
    forecast, target, mask, partition = _band_inputs()
    bands = metrics.band_forecast_metrics(forecast, target, mask, partition)
    overall = metrics.forecast_metrics(forecast, target, mask, n_scattering=3)["feat_mse_total"]

    assert _channel_weighted_blend(bands).tolist() == pytest.approx(
        torch.nan_to_num(overall).tolist(), rel=1e-6
    )


def test_a_partition_with_an_overlap_breaks_the_weighted_mean_identity() -> None:
    """Non-vacuity: the identity above must actually be able to fail."""
    forecast, target, mask, _ = _band_inputs()
    overlapping = {"a": [0, 1, 2, 3], "b": [3, 4, 5, 6]}  # channel 3 counted twice
    bands = metrics.band_forecast_metrics(forecast, target, mask, overlapping)
    overall = metrics.forecast_metrics(forecast, target, mask, n_scattering=3)["feat_mse_total"]

    assert _channel_weighted_blend(bands).tolist() != pytest.approx(
        torch.nan_to_num(overall).tolist(), rel=1e-6
    )


def test_a_band_reports_the_error_of_its_own_channels_only() -> None:
    """Sabotage one band's channels and only that band must move."""
    target = torch.zeros(1, 3, 2, 6)
    forecast = target.clone()
    forecast[..., [2, 3]] = 4.0
    mask = torch.ones(1, 3, 2, 1)

    bands = metrics.band_forecast_metrics(
        forecast, target, mask, {"clean": [0, 1], "dirty": [2, 3], "also_clean": [4, 5]}
    )
    assert float(bands["dirty"]["feat_mse"][0]) == pytest.approx(16.0)
    assert float(bands["clean"]["feat_mse"][0]) == pytest.approx(0.0)
    assert float(bands["also_clean"]["feat_mse"][0]) == pytest.approx(0.0)


def test_an_empty_band_is_dropped_rather_than_reported_as_zero() -> None:
    """Zero is a legitimate MSE, so an empty band reported as zero reads as the best band."""
    forecast, target, mask, partition = _band_inputs()
    partition = dict(partition, empty=[])
    bands = metrics.band_forecast_metrics(forecast, target, mask, partition)
    assert "empty" not in bands
    assert set(bands) == {"low", "mid", "high"}


def test_the_band_profiles_carry_the_geometry_they_claim() -> None:
    forecast, target, mask, partition = _band_inputs()
    bands = metrics.band_forecast_metrics(forecast, target, mask, partition)
    for record in bands.values():
        assert record["feat_mse"].shape == (3,)
        assert record["horizon"].shape == (3, 2)
        assert record["anchor"].shape == (3, 5)


def test_a_band_naming_a_channel_outside_the_forecast_raises() -> None:
    """Clipping would place real error in the wrong band and nothing downstream would notice."""
    forecast, target, mask, _ = _band_inputs()
    with pytest.raises(IndexError, match="outside the forecast"):
        metrics.band_forecast_metrics(forecast, target, mask, {"bad": [0, 99]})


#: Tolerance for the accumulator's identities. It is ``float32`` precision, not slack: the
#: per-batch reduction runs on the tensors' own dtype before being folded into a ``float64``
#: total, so the accumulator and a straight ``masked_pooled_mean`` -- and two different
#: batchings of the same data -- agree to about seven digits rather than to the last bit.
ACCUMULATOR_TOL = 1e-6


def test_the_channel_mean_of_the_accumulator_reproduces_the_pooled_total() -> None:
    """The identity that catches a mis-sliced channel axis or a double-counted horizon."""
    forecast, target, mask, _ = _band_inputs()
    accumulator = metrics.ChannelErrorAccumulator(n_channels=7, horizon=2)
    accumulator.update(forecast, target, mask)

    pooled = float(metrics.masked_pooled_mean((forecast - target) ** 2, mask))
    assert accumulator.total_mse() == pytest.approx(pooled, rel=ACCUMULATOR_TOL)
    assert float(np.mean(accumulator.per_channel_mse())) == pytest.approx(
        pooled, rel=ACCUMULATOR_TOL
    )


def test_the_accumulator_streams_batches_to_the_same_answer_as_one_pass() -> None:
    """Streaming is the whole point: it must not depend on how the split is batched."""
    forecast, target, mask, _ = _band_inputs()
    one_pass = metrics.ChannelErrorAccumulator(n_channels=7, horizon=2)
    one_pass.update(forecast, target, mask)

    streamed = metrics.ChannelErrorAccumulator(n_channels=7, horizon=2)
    for index in range(int(forecast.shape[0])):
        streamed.update(forecast[index: index + 1], target[index: index + 1], mask[index: index + 1])

    assert streamed.total_mse() == pytest.approx(one_pass.total_mse(), rel=ACCUMULATOR_TOL)
    assert streamed.per_channel_mse() == pytest.approx(
        one_pass.per_channel_mse(), rel=ACCUMULATOR_TOL
    )
    assert streamed.n_samples == int(forecast.shape[0])


def test_the_accumulator_names_the_channel_that_was_sabotaged() -> None:
    """Non-vacuity: a per-channel readout that did not resolve channels would fail here."""
    target = torch.zeros(2, 3, 2, 5)
    forecast = target.clone()
    forecast[..., 3] = 6.0

    accumulator = metrics.ChannelErrorAccumulator(n_channels=5, horizon=2)
    accumulator.update(forecast, target, torch.ones(2, 3, 2, 1))
    per_channel = accumulator.per_channel_mse()

    assert int(np.argmax(per_channel)) == 3
    assert per_channel[3] == pytest.approx(36.0)


def test_the_accumulator_resolves_a_horizon_step_within_a_channel() -> None:
    target = torch.zeros(1, 4, 3, 2)
    forecast = target.clone()
    forecast[:, :, 2, 1] = 5.0  # last horizon step of channel 1 only

    accumulator = metrics.ChannelErrorAccumulator(n_channels=2, horizon=3)
    accumulator.update(forecast, target, torch.ones(1, 4, 3, 1))
    field = accumulator.per_channel_horizon_mse()

    assert field.shape == (2, 3)
    assert field[1, 2] == pytest.approx(25.0)
    assert field[0, 2] == pytest.approx(0.0)
    assert field[1, 0] == pytest.approx(0.0)


def test_an_accumulator_that_saw_nothing_reports_nan_rather_than_zero() -> None:
    """Zero is a perfect forecast; "no data" must not be reported as one."""
    accumulator = metrics.ChannelErrorAccumulator(n_channels=4, horizon=2)
    assert math.isnan(accumulator.total_mse())
    assert np.isnan(accumulator.per_channel_mse()).all()


def test_the_accumulator_refuses_a_batch_of_the_wrong_geometry() -> None:
    """Broadcasting one horizon's error across another's would be silent otherwise."""
    accumulator = metrics.ChannelErrorAccumulator(n_channels=4, horizon=2)
    wrong = torch.zeros(1, 2, 3, 4)
    with pytest.raises(ValueError, match="was built for"):
        accumulator.update(wrong, wrong.clone(), torch.ones(1, 2, 3, 1))


# ---------------------------------------------------------------------------
# Uplift and residual
# ---------------------------------------------------------------------------
def test_uplift_is_positive_when_the_full_forecast_is_better() -> None:
    """Sign convention: positive uplift means the source pathway helped."""
    target = torch.zeros(1, 2, 2, 2)
    mu_full = torch.full_like(target, 1.0)
    mu_base = torch.full_like(target, 3.0)
    logvar = torch.zeros_like(target)

    result = metrics.uplift_metrics(
        mu_full, mu_base, target, logvar, logvar, torch.ones(1, 2, 2, 1),
        likelihood="mse", sigma_obs=1.0,
    )
    assert float(result["l_full"][0]) == pytest.approx(1.0)
    assert float(result["l_base"][0]) == pytest.approx(9.0)
    assert float(result["uplift_abs"][0]) == pytest.approx(8.0)
    assert float(result["uplift_rel"][0]) == pytest.approx(8.0 / 9.0)


def test_relative_uplift_divides_by_the_magnitude_so_a_negative_loss_does_not_flip_it() -> None:
    """Under ``gaussian_nll`` a well-calibrated baseline loss is routinely negative.

    Dividing by the signed value would invert the uplift's sign on exactly the healthy runs.
    """
    target = torch.zeros(1, 2, 2, 2)
    # A tiny predictive variance makes both NLLs negative; full is still the better of the two.
    logvar = torch.full_like(target, -6.0)
    mu_full = torch.zeros_like(target)
    mu_base = torch.full_like(target, 0.01)

    result = metrics.uplift_metrics(
        mu_full, mu_base, target, logvar, logvar, torch.ones(1, 2, 2, 1),
        likelihood="gaussian_nll", sigma_obs="learned",
    )
    assert float(result["l_base"][0]) < 0.0, "this case is only meaningful with a negative loss"
    assert float(result["uplift_abs"][0]) > 0.0
    assert float(result["uplift_rel"][0]) > 0.0


def test_residual_usage_is_zero_for_a_dead_pathway_and_positive_otherwise() -> None:
    """``residual_ratio`` is the scale-free collapse signal."""
    mu_full = torch.full((1, 2, 2, 3), 2.0)
    dead = torch.zeros_like(mu_full)
    alive = torch.full_like(mu_full, 1.0)
    mask = torch.ones(1, 2, 2, 1)

    assert float(metrics.residual_usage(dead, mu_full, mask)["residual_ratio"][0]) == 0.0
    assert float(metrics.residual_usage(alive, mu_full, mask)["residual_ratio"][0]) == pytest.approx(0.5)


def test_residual_rms_sums_over_channels_as_the_training_diagnostic_does() -> None:
    r"""Matches ``task.py``'s ``delta_mu_rms``: $\sqrt{\mathrm{mean}_t \sum_c \delta_c^2}$."""
    delta = torch.ones(1, 2, 2, 4)  # sum over 4 channels = 4 at every cell
    mask = torch.ones(1, 2, 2, 1)
    result = metrics.residual_usage(delta, delta, mask)
    assert float(result["residual_rms"][0]) == pytest.approx(2.0)


def test_residual_per_anchor_localises_a_pathway_that_dies_partway() -> None:
    """A residual active early and flat later is a different finding from one never active."""
    delta = torch.zeros(1, 4, 2, 3)
    delta[:, :2] = 1.0
    trace = metrics.residual_per_anchor(delta, torch.ones(1, 4, 2, 1))[0]
    assert trace[:2].tolist() == pytest.approx([math.sqrt(3.0)] * 2)
    assert trace[2:].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------------------
# KL
# ---------------------------------------------------------------------------
@pytest.fixture
def perturbed_forward(perturb_posterior):
    """A perturbed model, a batch, and its forward -- the precondition for every KL assertion."""
    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(SHIPPED_KWARGS, kld_support="anchor"))
    perturb_posterior(model)
    model.eval()
    batch = make_stub_batch(batch_size=3, seq_len=SEQ_LEN, seed=1)
    torch.manual_seed(5)
    with torch.no_grad():
        outputs = model(
            batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
        )
    return model, batch, outputs


def test_kld_per_dim_agrees_with_kld_per_t_inside_the_support(perturbed_forward) -> None:
    r"""``kld_per_t`` is the *unmasked* full-$T$ sum over $d_z$.

    Compared inside the support only, as the presentation-level ``NaN`` masking that analyses
    apply outside it could never equal a finite number there.
    """
    model, _, outputs = perturbed_forward
    kld_btd = metrics.kld_per_dim(outputs, model)
    support = masks.kld_support(model, SEQ_LEN) > 0

    assert float(kld_btd.abs().sum()) > 0.0, "a perturbed posterior must give a nonzero KL"
    assert torch.allclose(
        kld_btd.sum(dim=-1)[:, support], outputs["kld_per_t"][:, support], atol=1e-5
    )


def test_kld_aggregates_are_nan_safe_on_an_empty_support(perturbed_forward) -> None:
    """A sample whose every step is invalid reports ``NaN``, not a zero that reads as no coupling."""
    model, _, outputs = perturbed_forward
    kld_btd = metrics.kld_per_dim(outputs, model)
    mask = masks.kld_mask(model, torch.ones(3, SEQ_LEN), 3, SEQ_LEN)
    mask = mask.clone()
    mask[2] = 0.0

    aggregates = metrics.kld_aggregates(kld_btd, mask)
    assert float(aggregates["kld_mean"][0]) > 0.0
    assert math.isnan(float(aggregates["kld_mean"][2]))
    assert math.isnan(float(aggregates["kld_sum"][2]))
    # All three keys or the contract is worse than useless. A norm is non-negative, so an empty
    # support that returned $0$ here would land at the extreme of the scale rather than off it:
    # ``latent.py`` filters on ``np.isfinite``, which keeps that zero, biasing the pooled mean
    # down and putting a by-subgroup spike at exactly $0$ that reads as latent collapse.
    assert float(aggregates["kld_dim_l2"][0]) > 0.0
    assert math.isnan(float(aggregates["kld_dim_l2"][2]))
    # The premise of that fix: the row is wholly NaN, so there is no partial-NaN case a
    # ``nan_to_num`` inside the norm could have been repairing.
    assert bool(torch.isnan(aggregates["kld_per_dim_mean"][2]).all())


def test_kld_aggregates_relate_to_each_other_as_their_names_claim(perturbed_forward) -> None:
    r"""$\mathrm{mean} = \mathrm{sum} / (n_{\mathrm{steps}} \cdot d_z)$, and the per-dim means average to it."""
    model, batch, outputs = perturbed_forward
    kld_btd = metrics.kld_per_dim(outputs, model)
    mask = masks.kld_mask(model, batch.weight, 3, SEQ_LEN)
    aggregates = metrics.kld_aggregates(kld_btd, mask)

    steps = float(mask[0].sum())
    d_z = int(model.d_z)
    assert float(aggregates["kld_mean"][0]) == pytest.approx(
        float(aggregates["kld_sum"][0]) / (steps * d_z), rel=1e-5
    )
    assert float(aggregates["kld_per_dim_mean"][0].mean()) == pytest.approx(
        float(aggregates["kld_mean"][0]), rel=1e-5
    )


def test_posterior_drift_uses_the_kl_support_not_the_warmup_alone(perturbed_forward) -> None:
    """Under ``'anchor'`` the final $H_d$ steps are outside the support and must not contribute.

    Rebuilding the window as warm-up-only is how this reads systematically low against a
    ``kld_raw`` computed elsewhere.
    """
    model, batch, outputs = perturbed_forward
    mask = masks.kld_mask(model, batch.weight, 3, SEQ_LEN)

    corrupted = dict(outputs)
    corrupted["mu_post"] = outputs["mu_post"].clone()
    corrupted["mu_post"][:, -int(model.horizon):] += 1000.0

    assert torch.allclose(
        metrics.posterior_drift(outputs, mask), metrics.posterior_drift(corrupted, mask)
    )


def test_latent_health_passes_the_models_own_diagnostics_through(perturbed_forward) -> None:
    """A passthrough, so eval cannot invent a second definition of "active"."""
    _, _, outputs = perturbed_forward
    health = metrics.latent_health(outputs)
    assert set(health) == {"kld_active_frac", "mu_prior_sat_frac", "delta_mu_sat_frac"}
    assert health["kld_active_frac"] == pytest.approx(float(outputs["kld_active_frac"]))
    assert 0.0 <= health["mu_prior_sat_frac"] <= 1.0


def test_latent_health_reports_nan_for_a_readout_the_forward_did_not_carry() -> None:
    """``encode_only`` returns a subset, so a missing key is expected rather than exceptional."""
    assert math.isnan(metrics.latent_health({})["kld_active_frac"])


# ---------------------------------------------------------------------------
# Lag conversion
# ---------------------------------------------------------------------------
def test_lag_to_seconds_defaults_to_the_axis_the_training_plots_draw() -> None:
    r"""$4\ell$ with no offset -- what ``plotting.py`` has always produced."""
    assert metrics.lag_to_seconds(0) == pytest.approx(0.0)
    assert metrics.lag_to_seconds(10) == pytest.approx(40.0)


def test_lag_to_seconds_carries_the_datasets_up_advance() -> None:
    r"""With $\Delta_{UP} = -20$ s a peak at lag $\ell$ reads as $4(\ell - 5)$ s.

    The dataset ADVANCED the uterine-pressure trace by $20$ s -- ``mimo_adaptor.py`` runs
    ``up_shifted[:-80] = up_signal[80:]`` -- so stored position $g$ holds what the sensor recorded
    at $g + 20$ s. Attention at anchor $t$ reading source $t - \ell$ therefore compares fetal
    heart rate at $4t$ against uterine activity recorded at $4(t-\ell) + 20$: a lead of
    $4\ell - 20$ s, which is **negative** for the first five lags.

    This asserted $+20 / +60 / 4(90+5)$ until the sign was corrected. That is the same arithmetic
    the function had, so the test could not have caught it -- it is pinned here in the corrected
    direction precisely so a revert fails.
    """
    assert metrics.lag_to_seconds(0, up_shift_secs=-20.0) == pytest.approx(-20.0)
    assert metrics.lag_to_seconds(5, up_shift_secs=-20.0) == pytest.approx(0.0)
    assert metrics.lag_to_seconds(10, up_shift_secs=-20.0) == pytest.approx(20.0)
    assert metrics.lag_to_seconds(90, up_shift_secs=-20.0) == pytest.approx(4.0 * (90 - 5))


def test_lag_to_seconds_maps_an_array_elementwise() -> None:
    """A figure converts a whole axis at once."""
    lags = np.arange(4)
    seconds = metrics.lag_to_seconds(lags, up_shift_secs=-20.0)
    assert seconds.tolist() == pytest.approx([-20.0, -16.0, -12.0, -8.0])


def test_lag_seconds_physical_returns_a_float_column_from_a_tensor() -> None:
    """Its consumers are a DataFrame column and a matplotlib axis; neither takes a tensor."""
    seconds = metrics.lag_seconds_physical(torch.arange(3), up_shift_secs=-20.0)
    assert isinstance(seconds, np.ndarray) and seconds.dtype == np.float64
    assert seconds.tolist() == pytest.approx([-20.0, -16.0, -12.0])


# ---------------------------------------------------------------------------
# Attention diagnostics
# ---------------------------------------------------------------------------
def test_attention_rows_sum_to_one_under_eval_with_dropout_live() -> None:
    r"""The dropout hazard, tested on a model that actually has dropout.

    ``attn_dropout`` is applied *after* the normaliser and before the ``alpha`` handed back, so a
    ``train()``-mode pass returns rows that do not sum to $1$ and the identity
    $\sum_\ell \widetilde{TE}_{t,\ell} = K_t$ quietly stops holding. Every fixture in this
    repository is built at ``dropout=0.0``, where a ``train()`` pass is indistinguishable from an
    ``eval()`` one -- so a test on the default fixture would pass whatever the mode and prove
    nothing at all about the thing it is named for.
    """
    model = SeqVaeLagAttn(**dict(SHIPPED_KWARGS, dropout=0.5))
    batch = make_stub_batch(batch_size=3)
    streams = (
        batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1)
    )

    model.train()
    torch.manual_seed(0)
    with torch.no_grad():
        training_rows = model(*streams)["attn_weights"].sum(dim=-1)

    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        eval_rows = model(*streams)["attn_weights"].sum(dim=-1)

    assert torch.allclose(eval_rows, torch.ones_like(eval_rows), atol=1e-5)
    assert not torch.allclose(training_rows, torch.ones_like(training_rows), atol=1e-3), (
        "dropout did not perturb the rows, so this test cannot see the hazard it exists for"
    )


def test_attention_diagnostics_are_computed_only_over_the_support() -> None:
    """Anchors outside the support must not move a single reported number."""
    torch.manual_seed(0)
    weights = torch.rand(2, 6, 3, 5)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    support = torch.zeros(2, 6)
    support[:, 2:5] = 1.0

    before = metrics.attention_diagnostics(weights, support)
    corrupted = weights.clone()
    corrupted[:, 5] = torch.nn.functional.one_hot(torch.tensor(4), 5).float()
    after = metrics.attention_diagnostics(corrupted, support)

    assert torch.allclose(before["alpha_bar"], after["alpha_bar"])
    assert torch.allclose(before["mass_by_lag"], after["mass_by_lag"])
    assert torch.equal(before["argmax_lag"], after["argmax_lag"])


def test_attention_diagnostics_shapes_follow_the_geometry() -> None:
    r"""Per-head entropy is $(B, T, M)$ -- one number per head per anchor, not a head average."""
    weights = torch.full((2, 6, 3, 5), 0.2)
    diagnostics = metrics.attention_diagnostics(weights, torch.ones(2, 6))

    assert diagnostics["alpha_bar"].shape == (2, 3, 5)
    assert diagnostics["mass_by_lag"].shape == (2, 5)
    assert diagnostics["entropy"].shape == (2, 6, 3)
    assert diagnostics["entropy_mean"].shape == (2, 3)
    assert diagnostics["argmax_lag"].shape == (2,)
    assert diagnostics["head_diversity"].shape == (2,)


def test_a_uniform_row_has_maximal_entropy_and_a_peaked_row_has_none() -> None:
    r"""Entropy in nats, bounded above by $\log L$. This is what separates a peak from a smear."""
    uniform = metrics.attention_diagnostics(
        torch.full((1, 2, 1, 4), 0.25), torch.ones(1, 2)
    )
    assert float(uniform["entropy_mean"][0, 0]) == pytest.approx(math.log(4.0))

    peaked = torch.zeros(1, 2, 1, 4)
    peaked[..., 2] = 1.0
    result = metrics.attention_diagnostics(peaked, torch.ones(1, 2))
    # entmax15 produces exact zeros, so 0*log(0) must be 0 rather than -inf or NaN.
    assert float(result["entropy_mean"][0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert int(result["argmax_lag"][0]) == 2


def test_head_diversity_is_zero_for_identical_heads_and_one_for_disjoint_ones() -> None:
    """Four heads that all found the same lag are one head with four times the parameters."""
    identical = torch.zeros(1, 2, 2, 4)
    identical[..., 1] = 1.0
    assert float(metrics.attention_diagnostics(identical, torch.ones(1, 2))["head_diversity"]) == (
        pytest.approx(0.0, abs=1e-9)
    )

    disjoint = torch.zeros(1, 2, 2, 4)
    disjoint[:, :, 0, 0] = 1.0
    disjoint[:, :, 1, 3] = 1.0
    assert float(metrics.attention_diagnostics(disjoint, torch.ones(1, 2))["head_diversity"]) == (
        pytest.approx(1.0)
    )


def test_a_sample_with_no_supported_anchor_yields_nan_and_a_negative_argmax() -> None:
    """$-1$ rather than $0$: lag $0$ is a real answer and the most commonly reported one."""
    weights = torch.full((2, 4, 2, 3), 1.0 / 3.0)
    support = torch.ones(2, 4)
    support[1] = 0.0

    diagnostics = metrics.attention_diagnostics(weights, support)
    assert int(diagnostics["argmax_lag"][0]) >= 0
    assert int(diagnostics["argmax_lag"][1]) == -1
    assert bool(torch.isnan(diagnostics["mass_by_lag"][1]).all())
    assert bool(torch.isnan(diagnostics["entropy_mean"][1]).all())


def test_dead_anchors_are_excluded_by_the_live_anchor_mask() -> None:
    r"""A dead row sums to $0$ and is not a distribution; averaging it in drags the profile down.

    ``_ablate_dead_anchors`` zeroes the row deliberately and does not renormalise, so this is the
    mask every lag-resolved readout must intersect with before it averages.
    """
    weights = torch.full((2, 5, 2, 4), 0.25)
    weights[:, 1] = 0.0

    live = masks.live_anchor_mask(weights)
    assert live.shape == (2, 5)
    assert not bool(live[:, 1].any())
    assert bool(live[:, [0, 2, 3, 4]].all())

    # With the dead anchor averaged in, the profile no longer sums to 1 -- which is exactly the
    # corruption the mask prevents.
    naive = metrics.attention_diagnostics(weights, torch.ones(2, 5))
    assert float(naive["mass_by_lag"][0].sum()) == pytest.approx(0.8)
    masked = metrics.attention_diagnostics(weights, live.float())
    assert float(masked["mass_by_lag"][0].sum()) == pytest.approx(1.0)
