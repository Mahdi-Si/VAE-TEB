r"""The assembled objective: the three-term total, its masks, and its return contract.

Two of these tests restate structural invariants on the loss path deliberately. The zero-KL
start is asserted again *through* ``compute_loss`` -- a wiring mistake between forward and loss
(a stale key, a wrong branch fed to the wrong term) would leave the forward-path test green and
this one red. And the mask test runs end to end: a value planted at a gapped raw position must
be invisible to the total, which exercises target gather, mask construction and reduction
together.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.tests.conftest import STUB_GAP_STEP, make_stub_batch


def _model(tiny_kwargs, **overrides) -> SeqVaeLagAttnRws:
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(**dict(tiny_kwargs, **overrides)).eval()


def _loss(model, batch, perturb=None, **loss_kwargs):
    if perturb is not None:
        perturb(model)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))
    return out, model.compute_loss(out, batch.fhr, weight=batch.weight, **loss_kwargs)


def test_the_dict_separates_tensors_from_the_likelihood_string(tiny_kwargs):
    _, result = _loss(_model(tiny_kwargs), make_stub_batch())
    assert set(result) == {"metrics", "likelihood"}
    assert result["likelihood"] == "gaussian_nll"
    assert all(
        isinstance(value, torch.Tensor) for value in result["metrics"].values()
    ), "a non-tensor inside metrics would poison a splatted logger"


def test_the_total_is_the_documented_four_term_sum(tiny_kwargs, perturb_posterior):
    """Distinct coefficients, perturbed model: the total must recompose from the returned
    parts under exactly the documented weights.

    Four terms, not the original three: nothing in the three-term objective penalises a
    *narrow* prior, and the first production run pinned 99.2% of prior log-variances on the
    clamp floor within one epoch -- so the prior's scale rate joined the objective, weighted
    by ``beta_prior``. Exercised at a non-zero weight so the term is covered rather than
    multiplied by zero."""
    _, result = _loss(
        _model(tiny_kwargs),
        make_stub_batch(),
        perturb=perturb_posterior,
        beta=0.7,
        beta_prior=0.11,
        lambda_full=1.0,
        lambda_base=0.3,
    )
    metrics = result["metrics"]
    recomposed = (
        1.0 * metrics["nll_full_block"]
        + 0.3 * metrics["nll_base_block"]
        + 0.7 * metrics["source_conditioned_kl_train"]
        + 0.11 * metrics["prior_rate"]
    )
    assert torch.allclose(metrics["total_loss"], recomposed, rtol=1e-6)
    assert float(metrics["source_conditioned_kl_train"]) > 0.0
    assert float(metrics["prior_rate"]) > 0.0
    assert torch.allclose(metrics["kld_beta"], torch.tensor(0.7))
    assert torch.allclose(metrics["beta_prior"], torch.tensor(0.11))

    # The negative control: the historical three-term recomposition must now fall short by
    # exactly the weighted prior rate, so a silently dropped fourth term cannot pass.
    three_term = recomposed - 0.11 * metrics["prior_rate"]
    assert not torch.allclose(metrics["total_loss"], three_term, rtol=1e-6)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_zero_beta_prior_the_rate_is_reported_but_inert(tiny_kwargs, likelihood):
    """The opt-in contract: at the default ``beta_prior=0.0`` the total is bitwise the
    three-term sum, while ``prior_rate`` is still emitted -- under every likelihood, ``mse``
    included, so the smoke configuration reports it too."""
    _, result = _loss(_model(tiny_kwargs), make_stub_batch(), likelihood=likelihood)
    metrics = result["metrics"]

    three_term = (
        metrics["nll_full_block"]
        + metrics["nll_base_block"]
        + 1.0 * metrics["source_conditioned_kl_train"]
    )
    assert torch.equal(metrics["total_loss"], three_term)
    assert torch.isfinite(metrics["prior_rate"])
    assert float(metrics["prior_rate"]) > 0.0  # the uncalibrated prior is off unit scale
    assert float(metrics["beta_prior"]) == 0.0


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_init_the_two_reconstruction_terms_are_bitwise_equal(tiny_kwargs, likelihood):
    """The zero-KL start, restated on the loss path: D_0 == D_1 bitwise and the KL term is
    exactly zero, under both likelihoods."""
    _, result = _loss(_model(tiny_kwargs), make_stub_batch(), likelihood=likelihood)
    metrics = result["metrics"]
    assert torch.equal(metrics["nll_full_block"], metrics["nll_base_block"])
    assert torch.equal(metrics["nll_full_sample"], metrics["nll_base_sample"])
    assert float(metrics["source_conditioned_kl_train"]) == 0.0
    assert float(metrics["source_conditioned_kl_raw"]) == 0.0
    assert float(metrics["pred_gap"]) == 0.0


def test_an_unknown_likelihood_is_rejected_listing_the_choices(tiny_kwargs):
    model = _model(tiny_kwargs)
    batch = make_stub_batch()
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))
    with pytest.raises(ValueError, match=r"mse.*gaussian_nll"):
        model.compute_loss(out, batch.fhr, weight=batch.weight, likelihood="huber")


def test_a_gapped_raw_position_is_invisible_end_to_end(tiny_kwargs):
    """Planting an absurd value inside the gapped step's raw block (the only route by which
    fhr_raw enters the loss is the future-target gather) must leave every reconstruction
    number bitwise unchanged."""
    model = _model(tiny_kwargs)
    batch = make_stub_batch()
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))

    reference = model.compute_loss(out, batch.fhr, weight=batch.weight)
    planted_fhr = batch.fhr.clone()
    start = model.geometry.decimation * STUB_GAP_STEP
    planted_fhr[:, start : start + model.geometry.decimation] = 1.0e6
    planted = model.compute_loss(out, planted_fhr, weight=batch.weight)

    for key in ("total_loss", "nll_full_block", "nll_base_block"):
        assert torch.equal(reference["metrics"][key], planted["metrics"][key]), key


def test_anchor_coverage_frac_reports_the_pre_floor_distribution_mean(tiny_kwargs):
    """With the stub gap, anchors 6..9 keep 3/4 of their window: the mean over the ten
    post-warmup anchors is (6 + 4*0.75)/10 = 0.9. A full weight reads 1.0."""
    model = _model(tiny_kwargs)
    batch = make_stub_batch()
    _, gapped = _loss(model, batch)
    assert torch.allclose(gapped["metrics"]["anchor_coverage_frac"], torch.tensor(0.9))

    full = make_stub_batch()
    full.weight = torch.ones_like(full.weight)
    _, clean = _loss(model, full)
    assert torch.allclose(clean["metrics"]["anchor_coverage_frac"], torch.tensor(1.0))


def test_the_coverage_floor_is_honoured(tiny_kwargs):
    """Same forward outputs, two floors: the 0.9-floor model drops the 0.75-coverage anchors
    entirely, so its reconstruction numbers must differ from the 0.0-floor model's on a
    gapped batch and agree on a clean one."""
    batch = make_stub_batch()
    strict = _model(tiny_kwargs, coverage_floor=0.9)
    lax = _model(tiny_kwargs, coverage_floor=0.0)  # same seed -> identical weights

    with torch.no_grad():
        out = strict(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))

    gapped_strict = strict.compute_loss(out, batch.fhr, weight=batch.weight)
    gapped_lax = lax.compute_loss(out, batch.fhr, weight=batch.weight)
    assert not torch.equal(
        gapped_strict["metrics"]["nll_full_block"], gapped_lax["metrics"]["nll_full_block"]
    )

    clean_weight = torch.ones_like(batch.weight)
    clean_strict = strict.compute_loss(out, batch.fhr, weight=clean_weight)
    clean_lax = lax.compute_loss(out, batch.fhr, weight=clean_weight)
    assert torch.equal(
        clean_strict["metrics"]["nll_full_block"], clean_lax["metrics"]["nll_full_block"]
    )


def test_free_bits_raises_the_trained_kl_above_the_raw_one(tiny_kwargs, perturb_posterior):
    _, result = _loss(
        _model(tiny_kwargs), make_stub_batch(), perturb=perturb_posterior, free_bits=0.5
    )
    metrics = result["metrics"]
    assert float(metrics["source_conditioned_kl_train"]) > float(
        metrics["source_conditioned_kl_raw"]
    )


def test_the_objective_carries_gradient(tiny_kwargs, perturb_posterior):
    """A smoke check that the assembled total is trainable: backward runs and reaches the
    decoder and the posterior pathway."""
    model = _model(tiny_kwargs).train()
    perturb_posterior(model)
    batch = make_stub_batch()
    out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))
    result = model.compute_loss(out, batch.fhr, weight=batch.weight)
    result["metrics"]["total_loss"].backward()
    assert model.decoder.mean_head.weight.grad is not None
    assert float(model.decoder.mean_head.weight.grad.abs().max()) > 0.0


def test_the_latent_diagnostics_are_present_and_finite(tiny_kwargs):
    _, result = _loss(_model(tiny_kwargs), make_stub_batch())
    metrics = result["metrics"]
    for key in (
        "mean_logvar_full",
        "mean_logvar_base",
        "mean_logvar_prior",
        "mean_logvar_post",
        "logvar_prior_floor_frac",
        "delta_mu_rms",
        "kld_active_frac",
    ):
        assert key in metrics and torch.isfinite(metrics[key]), key
    # At init the posterior sits on the prior exactly.
    assert float(metrics["delta_mu_rms"]) == 0.0
    assert torch.equal(metrics["mean_logvar_prior"], metrics["mean_logvar_post"])
