r"""The assembled objective: the seven-term total, its masks, and its return contract.

Two of these tests restate structural invariants on the loss path deliberately. The zero-KL
start is asserted again *through* ``compute_loss`` -- a wiring mistake between forward and loss
(a stale key, a wrong branch fed to the wrong term) would leave the forward-path test green and
this one red. And the mask test runs end to end: a value planted at a gapped raw position must
be invisible to the total, which exercises target gather, mask construction and reduction
together.

The last two tests are the **equivalence harness**, and they answer a different question from the
rest: not "is the objective right" but "did the objective change". They are the evidence a later
generalisation of ``nets/losses.py`` is inert, so both are computed in-process with both sides in
one run -- never against committed decimal constants, which do not survive a move between the dev
box and the production one and which no rule here could legally regenerate.

They divide the work:

* :func:`test_the_method_and_the_free_function_agree_on_every_metric` pins the model's method
  against a direct call to the free function with every value the method reads off ``self``
  supplied explicitly, so a method quietly passing a different geometry, coverage floor or bound
  is caught.
* :func:`test_every_metric_reassembles_from_the_primitives` pins the *assembly* -- which target,
  which masks, which denominators -- against an independent one built straight from
  ``build_future_target``, ``forecast_mask`` and ``kl_mask``. It calls the assembled objective
  nowhere, so it holds whatever shape that function's signature takes, and it is the one that
  would catch a per-element denominator built from the wrong width: that mistake changes no loss,
  fails no shape check, and rescales the four log-variance diagnostics alone.

The second of those is **shared**. :func:`reassembled_metrics` and
:func:`assert_objective_reassembles` are free functions the feature-domain packages import and
drive with their own model, their own target and their own block width, because the objective
they are asking about is the same one. The dependency runs one way only: nothing here imports a
package downstream of this one, so each of the four models is pinned by the suite that owns it.
"""
from __future__ import annotations

from typing import Iterable

import pytest
import torch

from teb_vae.lag_attn_rws.nets.losses import (
    LOGVAR_FLOOR_MARGIN_FRAC,
    compute_loss as free_objective,
    kld_tensor,
    masked_boundary_gap,
    masked_derivative_huber,
    masked_multiscale_l1,
    masked_prior_rate,
    masked_raw_likelihood,
    masked_source_kl,
)
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_index, build_future_target
from teb_vae.lag_attn_rws.tests.conftest import (
    STUB_GAP_STEP,
    TINY_KWARGS,
    make_stub_batch,
    tiny_gated_kwargs,
)

#: Coefficients the equivalence harness runs at. Mutually distinct and none of them a default:
#: at equal weights a term swapped for another passes, at ``beta_prior=0`` the fourth term is
#: multiplied away, and at ``free_bits=0`` the raw and trained KL are one tensor rather than two.
#: The three shape weights are nonzero for the same reason and one more -- at zero the objective
#: does not compute those terms at all, so the harness would be comparing three exact zeros.
_HARNESS_COEFFICIENTS = dict(
    beta=0.7,
    beta_prior=0.11,
    lambda_full=1.0,
    lambda_base=0.3,
    free_bits=0.05,
    lambda_ms=0.13,
    lambda_deriv=0.17,
    lambda_boundary=0.19,
)


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


def _free_objective_explicitly(model, outs, batch, *, likelihood):
    """Call the free objective with every value the method supplies from ``self`` written out.

    The one place in this file that names the free function's argument list, so a change to that
    signature touches these three lines and leaves every assertion below standing.
    """
    return free_objective(
        outs,
        # Gathered here rather than read off anything the model built, so "supplied explicitly"
        # is literal: the index grid is rebuilt from the geometry too.
        build_future_target(
            batch.fhr, model.geometry, future_index=build_future_index(model.geometry)
        ),
        weight=batch.weight,
        geometry=model.geometry,
        block_width=model.geometry.r,
        coverage_floor=model.coverage_floor,
        logvar_clamp=model.logvar_clamp,
        likelihood=likelihood,
        **_HARNESS_COEFFICIENTS,
    )


def test_the_dict_separates_tensors_from_the_likelihood_string(tiny_kwargs):
    _, result = _loss(_model(tiny_kwargs), make_stub_batch())
    assert set(result) == {"metrics", "likelihood"}
    assert result["likelihood"] == "gaussian_nll"
    assert all(
        isinstance(value, torch.Tensor) for value in result["metrics"].values()
    ), "a non-tensor inside metrics would poison a splatted logger"


def _recompose(metrics, coefficients) -> torch.Tensor:
    """The documented seven-term sum, written out from the reported parts.

    Args:
        metrics: The objective's metric dict.
        coefficients: The weights it was called at, keyed as the objective's kwargs.

    Returns:
        The recomposed total.
    """
    return (
        coefficients["lambda_full"] * metrics["nll_full_block"]
        + coefficients["lambda_base"] * metrics["nll_base_block"]
        + coefficients["beta"] * metrics["source_conditioned_kl_train"]
        + coefficients["beta_prior"] * metrics["prior_rate"]
        + coefficients["lambda_ms"] * metrics["aux_multiscale"]
        + coefficients["lambda_deriv"] * metrics["aux_derivative"]
        + coefficients["lambda_boundary"] * metrics["aux_boundary"]
    )


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_total_is_the_documented_seven_term_sum(
    tiny_kwargs, perturb_posterior, likelihood
):
    """Distinct coefficients, perturbed model: the total must recompose from the returned
    parts under exactly the documented weights.

    Seven terms, not the original four: the three shape terms regularise the forecast *mean*,
    which the factorized Gaussian leaves free to be the over-smoothed conditional average. Both
    likelihoods, because the shape terms read means only and so are identical under either --
    the tiny smoke configuration runs ``mse`` and must report the same numbers.

    Every weight is nonzero, so each term is covered rather than multiplied away, and each is
    subtracted back one at a time below: a total that silently dropped one would still pass an
    equality against a recomposition that had dropped it too."""
    _, result = _loss(
        _model(tiny_kwargs),
        make_stub_batch(),
        perturb=perturb_posterior,
        likelihood=likelihood,
        **_HARNESS_COEFFICIENTS,
    )
    metrics = result["metrics"]
    recomposed = _recompose(metrics, _HARNESS_COEFFICIENTS)

    assert torch.allclose(metrics["total_loss"], recomposed, rtol=1e-6)
    assert float(metrics["source_conditioned_kl_train"]) > 0.0
    assert float(metrics["prior_rate"]) > 0.0
    assert torch.allclose(metrics["kld_beta"], torch.tensor(_HARNESS_COEFFICIENTS["beta"]))
    assert torch.allclose(
        metrics["beta_prior"], torch.tensor(_HARNESS_COEFFICIENTS["beta_prior"])
    )

    # The negative controls, one per weighted term: drop it from the recomposition and the
    # equality must fail. Vacuous if the term is zero, so each is checked to be positive first.
    for weight_key, metric_key in (
        ("beta_prior", "prior_rate"),
        ("lambda_ms", "aux_multiscale"),
        ("lambda_deriv", "aux_derivative"),
        ("lambda_boundary", "aux_boundary"),
    ):
        assert float(metrics[metric_key]) > 0.0, metric_key
        short = recomposed - _HARNESS_COEFFICIENTS[weight_key] * metrics[metric_key]
        assert not torch.allclose(metrics["total_loss"], short, rtol=1e-6), metric_key


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_at_zero_shape_weights_the_total_is_bitwise_the_four_term_sum(
    tiny_kwargs, perturb_posterior, likelihood
):
    """The inertness contract the shape terms shipped under: at the default weights of $0.0$ the
    objective is *bitwise* what it was before they existed, and the three metrics are exact
    zeros rather than small numbers.

    ``torch.equal``, not ``allclose``: every model, config and checkpoint that predates these
    terms must be unaffected to the last bit, and a tolerance is exactly what would hide a term
    that was computed and then multiplied by zero -- which is not what happens (it is not
    computed at all) and is the difference the ``aux_*`` metrics report."""
    four_term_coefficients = dict(
        _HARNESS_COEFFICIENTS, lambda_ms=0.0, lambda_deriv=0.0, lambda_boundary=0.0
    )
    _, result = _loss(
        _model(tiny_kwargs),
        make_stub_batch(),
        perturb=perturb_posterior,
        likelihood=likelihood,
        **four_term_coefficients,
    )
    metrics = result["metrics"]

    four_term = (
        four_term_coefficients["lambda_full"] * metrics["nll_full_block"]
        + four_term_coefficients["lambda_base"] * metrics["nll_base_block"]
        + four_term_coefficients["beta"] * metrics["source_conditioned_kl_train"]
        + four_term_coefficients["beta_prior"] * metrics["prior_rate"]
    )
    assert torch.equal(metrics["total_loss"], four_term)

    for key in ("aux_multiscale", "aux_derivative", "aux_boundary"):
        assert float(metrics[key]) == 0.0, key
        assert torch.isfinite(metrics[key]), key
    for key in ("lambda_ms", "lambda_deriv", "lambda_boundary"):
        assert float(metrics[key]) == 0.0, key


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


@pytest.mark.parametrize(
    "coefficients", [{}, _HARNESS_COEFFICIENTS], ids=["shape-terms-off", "shape-terms-on"]
)
def test_a_gapped_raw_position_is_invisible_end_to_end(tiny_kwargs, coefficients):
    """Planting an absurd value inside the gapped step's raw block (the only route by which
    fhr_raw enters the loss is the future-target gather) must leave every reconstruction
    number bitwise unchanged.

    Run with the shape terms on as well, because each of them reads the target through a
    different route the reconstruction does not use -- a pooled neighbourhood, a difference
    pair, and a sample taken from the *previous* anchor's block -- so masking that holds for the
    NLL says nothing about any of them."""
    model = _model(tiny_kwargs)
    batch = make_stub_batch()
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], -1))

    reference = model.compute_loss(out, batch.fhr, weight=batch.weight, **coefficients)
    planted_fhr = batch.fhr.clone()
    start = model.geometry.decimation * STUB_GAP_STEP
    planted_fhr[:, start : start + model.geometry.decimation] = 1.0e6
    planted = model.compute_loss(out, planted_fhr, weight=batch.weight, **coefficients)

    for key in (
        "total_loss",
        "nll_full_block",
        "nll_base_block",
        "aux_multiscale",
        "aux_derivative",
        "aux_boundary",
    ):
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


# =============================================================================
# The reassembly harness, shared with the feature-domain packages
#
# Two free functions rather than a test, because the same question has to be asked of four models
# that live in four packages: this one owns the arithmetic and each package's own
# ``test_objective.py`` supplies its model, its target and its block width. The dependency runs
# one way -- this module imports nothing downstream of itself.
# =============================================================================
def reassembled_metrics(
    model,
    forward_outputs,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    likelihood: str,
    coefficients: dict,
    block_width: int,
) -> dict:
    r"""Rebuild every metric the shared objective reports, from the objective's own primitives.

    Calls :func:`~teb_vae.lag_attn_rws.nets.losses.compute_loss` nowhere. The masks come from
    ``forecast_mask`` and ``kl_mask``, each term from the primitive that defines it, and the
    weighted sum and the six per-element diagnostics are written out by hand -- exactly what the
    objective itself writes out by hand.

    A second copy of arithmetic is normally the thing to avoid; here it is the point, and it is
    what makes this harness survive a change to the objective's *signature*. The per-element
    denominator is the sharpest case: no shape check and no loss value depends on it, so a
    denominator built from the wrong width rescales four reported numbers by a constant and
    nothing else notices.

    Args:
        model: The net, read for its geometry, coverage floor and log-variance clamp.
        forward_outputs: Its forward dict.
        target: The forecast block $(B, T_{\mathrm{valid}}, H, X)$, built by the caller from
            whatever its target domain is.
        weight: Decimated validity signal $(B, T)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        coefficients: The objective weights the model was scored at, keyed as its kwargs. Absent
            keys default to $0.0$, which is what the shipped feature-domain callers do with the
            three shape weights.
        block_width: $X$ -- what the target's last axis counts. Written out by the caller rather
            than read off the model, because this is the one quantity a self-consistent objective
            cannot be checked against itself on.

    Returns:
        The expected metric dictionary, in the objective's own key set.
    """
    geometry, (lo, hi) = model.geometry, model.logvar_clamp
    dtype = target.dtype

    def _weight_of(name: str) -> float:
        return float(coefficients.get(name, 0.0))

    mask, coverage_frac = forecast_mask(
        weight, geometry, coverage_floor=model.coverage_floor
    )
    kl_support = kl_mask(mask, geometry)

    expected: dict = {}
    expected["nll_full_block"], expected["nll_full_sample"] = masked_raw_likelihood(
        forward_outputs["mu_full"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_full"],
    )
    expected["nll_base_block"], expected["nll_base_sample"] = masked_raw_likelihood(
        forward_outputs["mu_base"],
        target,
        mask,
        likelihood=likelihood,
        logvar=forward_outputs["logvar_base"],
    )
    expected.update(
        masked_source_kl(
            kld_tensor(
                mu_prior=forward_outputs["mu_prior"],
                logvar_prior=forward_outputs["logvar_prior"],
                mu_post=forward_outputs["mu_post"],
                logvar_post=forward_outputs["logvar_post"],
            ),
            kl_support,
            free_bits=_weight_of("free_bits"),
        )
    )
    expected["prior_rate"] = masked_prior_rate(forward_outputs["logvar_prior"], kl_support)

    # The shape terms, each over both branches in the objective's own order, and each an exact
    # zero when its weight is zero -- the objective does not compute an unweighted term at all,
    # so its metric is a real zero rather than the value it would have taken.
    zero = torch.zeros((), device=target.device, dtype=dtype)
    for name, weight_key, term in (
        ("aux_multiscale", "lambda_ms", masked_multiscale_l1),
        ("aux_derivative", "lambda_deriv", masked_derivative_huber),
    ):
        expected[name] = (
            term(forward_outputs["mu_full"], target, mask)
            + term(forward_outputs["mu_base"], target, mask)
            if _weight_of(weight_key) != 0.0
            else zero.clone()
        )
    expected["aux_boundary"] = (
        masked_boundary_gap(forward_outputs["mu_full"], target, mask, weight)
        + masked_boundary_gap(forward_outputs["mu_base"], target, mask, weight)
        if _weight_of("lambda_boundary") != 0.0
        else zero.clone()
    )

    expected["total_loss"] = (
        _weight_of("lambda_full") * expected["nll_full_block"]
        + _weight_of("lambda_base") * expected["nll_base_block"]
        + _weight_of("beta") * expected["source_conditioned_kl_train"]
        + _weight_of("beta_prior") * expected["prior_rate"]
        + _weight_of("lambda_ms") * expected["aux_multiscale"]
        + _weight_of("lambda_deriv") * expected["aux_derivative"]
        + _weight_of("lambda_boundary") * expected["aux_boundary"]
    )
    expected["pred_gap"] = expected["nll_base_block"] - expected["nll_full_block"]
    expected["kld_beta"] = torch.tensor(_weight_of("beta"), dtype=dtype)
    expected["beta_prior"] = torch.tensor(_weight_of("beta_prior"), dtype=dtype)
    for name in ("lambda_ms", "lambda_deriv", "lambda_boundary"):
        expected[name] = torch.tensor(_weight_of(name), dtype=dtype)
    expected["anchor_coverage_frac"] = coverage_frac[:, geometry.warmup :].mean()

    # The per-element reductions, over the mask broadcast across the block's last axis. The width
    # is the caller's constant: this denominator is the whole reason the harness exists.
    elem_mask = mask[..., None]
    elem_denom = (elem_mask.sum() * float(block_width)).clamp_min(1.0)
    margin = LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    for branch in ("full", "base"):
        expected[f"mean_logvar_{branch}"] = (
            forward_outputs[f"logvar_{branch}"] * elem_mask
        ).sum() / elem_denom
    logvar_full = forward_outputs["logvar_full"]
    expected["logvar_full_floor_frac"] = (
        (logvar_full <= lo + margin).to(dtype) * elem_mask
    ).sum() / elem_denom
    expected["logvar_full_ceil_frac"] = (
        (logvar_full >= hi - margin).to(dtype) * elem_mask
    ).sum() / elem_denom

    support = kl_support > 0
    assert bool(support.any()), "the batch must leave some anchor inside the KL support"
    logvar_prior_masked = forward_outputs["logvar_prior"][support]
    expected["logvar_prior_floor_frac"] = (
        (logvar_prior_masked <= lo + margin).to(dtype).mean()
    )
    expected["mean_logvar_prior"] = logvar_prior_masked.mean()
    expected["mean_logvar_post"] = forward_outputs["logvar_post"][support].mean()
    expected["delta_mu_rms"] = (
        (forward_outputs["mu_post"] - forward_outputs["mu_prior"])[support]
        .pow(2)
        .mean()
        .sqrt()
    )
    return expected


def assert_objective_reassembles(
    model,
    forward_outputs,
    target: torch.Tensor,
    weight: torch.Tensor,
    produced: dict,
    *,
    likelihood: str,
    coefficients: dict,
    block_width: int,
    package_owned: Iterable[str] = (),
) -> None:
    r"""Assert a model's reported metrics are exactly :func:`reassembled_metrics`.

    Key-set equality first, in both directions: a metric that appeared without being reassembled
    would otherwise slip through the value comparison entirely. A package that adds readouts of
    its own -- the feature-domain models' four resolved forecast gaps -- declares them in
    ``package_owned``, so an unannounced *fifth* addition still fails here.

    ``torch.equal``, not ``allclose``: the question is whether the number moved, and a tolerance
    is exactly what would hide a term computed over a slightly different anchor set.

    Args:
        model: The net.
        forward_outputs: Its forward dict.
        target: The forecast block, built by the caller.
        weight: Decimated validity signal $(B, T)$.
        produced: The ``metrics`` dict the model returned.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        coefficients: The objective weights the model was scored at.
        block_width: $X$, written out by the caller.
        package_owned: Metric names this package adds on top of the shared objective's.
    """
    expected = reassembled_metrics(
        model,
        forward_outputs,
        target,
        weight,
        likelihood=likelihood,
        coefficients=coefficients,
        block_width=block_width,
    )
    owned = set(package_owned)
    assert set(produced) - owned == set(expected), (
        "the reassembly must cover every reported metric: "
        f"missing={set(produced) - owned - set(expected)} "
        f"extra={set(expected) - set(produced)}"
    )
    assert owned <= set(produced), f"declared but absent: {sorted(owned - set(produced))}"

    differing = [key for key, value in expected.items() if not torch.equal(value, produced[key])]
    assert not differing, differing


# =============================================================================
# The equivalence harness
# =============================================================================
@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_the_method_and_the_free_function_agree_on_every_metric(
    tiny_kwargs, perturb_posterior, likelihood
):
    """The method supplies four things from ``self``; supplied by hand instead, the objective must
    return the identical dict.

    Every key, not a chosen subset, and ``torch.equal`` rather than ``allclose``: a method that
    passed its own geometry but a stale coverage floor would move only the anchor set, and a
    scalar comparison at a tolerance is exactly what would miss it.
    """
    model = _model(tiny_kwargs)
    batch = make_stub_batch()
    outs, through_method = _loss(
        model, batch, perturb=perturb_posterior, likelihood=likelihood, **_HARNESS_COEFFICIENTS
    )
    directly = _free_objective_explicitly(model, outs, batch, likelihood=likelihood)

    assert set(directly["metrics"]) == set(through_method["metrics"])
    differing = [
        key
        for key, value in through_method["metrics"].items()
        if not torch.equal(value, directly["metrics"][key])
    ]
    assert not differing, differing
    assert directly["likelihood"] == through_method["likelihood"] == likelihood
    # Not vacuous: the perturbation is what puts the KL, the prior rate and the posterior
    # displacement on non-zero values, so all four terms are compared at something.
    assert float(through_method["metrics"]["source_conditioned_kl_raw"]) > 0.0
    assert float(through_method["metrics"]["delta_mu_rms"]) > 0.0


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
@pytest.mark.parametrize("guard", ["ungated", "gated"], ids=["ungated", "gated"])
def test_every_metric_reassembles_from_the_primitives(perturb_posterior, likelihood, guard):
    """The assembly, pinned independently: which target, which masks, which denominators.

    Driven through :func:`assert_objective_reassembles`, which calls the assembled objective
    nowhere -- so it holds whatever shape that function's signature takes, and it is what a
    later generalisation of ``nets/losses.py`` has to leave standing.

    Both guard states, because they are different code paths into the same objective: a gated
    model reads its streams through a channel gate and an input adapter carrying availability
    terms, an ungated one through neither, and only the gated arm can catch a change that moves
    the masked prefix.
    """
    kwargs = TINY_KWARGS if guard == "ungated" else tiny_gated_kwargs()
    model = _model(dict(kwargs))
    batch = make_stub_batch()
    outs, result = _loss(
        model, batch, perturb=perturb_posterior, likelihood=likelihood, **_HARNESS_COEFFICIENTS
    )
    geometry = model.geometry
    target = build_future_target(
        batch.fhr, geometry, future_index=build_future_index(geometry)
    )

    assert_objective_reassembles(
        model,
        outs,
        target,
        batch.weight,
        result["metrics"],
        likelihood=likelihood,
        coefficients=_HARNESS_COEFFICIENTS,
        # The raw grid's R, written out here rather than taken from the objective: this
        # denominator is the whole reason the harness exists.
        block_width=geometry.r,
    )

    # The denominator claim, stated as the ratio it is, so the assertion above cannot be read as
    # having compared two copies of the same mistake.
    metrics = result["metrics"]
    assert float(metrics["nll_full_sample"]) == pytest.approx(
        float(metrics["nll_full_block"]) / (geometry.horizon * geometry.r), rel=1e-6
    )
    # Not vacuous: the perturbation is what puts the KL and the posterior displacement on
    # non-zero values, so the reassembly compared them at something.
    assert float(metrics["source_conditioned_kl_raw"]) > 0.0
    assert float(metrics["delta_mu_rms"]) > 0.0
