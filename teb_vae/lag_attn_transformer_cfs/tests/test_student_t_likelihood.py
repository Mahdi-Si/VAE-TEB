r"""The Student-$t$ innovation density behind ``forecast_student_nu``.

The keyword replaces the Gaussian density of each forecast innovation $\varepsilon$ by a scaled
Student-$t$ density with $\nu$ degrees of freedom and the decoder's
$\sigma = \exp(\log\sigma^2 / 2)$ as its scale:

$$-\log p(\varepsilon) = c_\nu + \tfrac{1}{2}\log\sigma^2
+ \tfrac{\nu + 1}{2}\operatorname{log1p}\!\left(\varepsilon^2 e^{-\log\sigma^2} / \nu\right).$$

The claims pinned here:

* **The score function.** The per-cell term equals ``torch.distributions.StudentT``'s negative
  log-probability, on the plain residual, on the AR(1) innovation and across a masked step, where
  the carried scale is the marginal one. ``student_nu=None`` is the Gaussian term bitwise, and
  ``'mse'`` ignores $\nu$. A value $\nu \le 2$ is refused, because the variance is then infinite.
* **The model.** Off by default: no parameter, no buffer and the two-key likelihood dict. On, the
  state dict is unchanged and the objective scores both branches under $\nu$.
* **The evaluation.** Coverage keeps the nominal masses and reads them inside the Student-$t$
  intervals, so draws from $t_\nu$ read calibrated under $\nu$ and fail the 3-sigma level under
  the Gaussian reading.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_cfs.eval import metrics as eval_metrics
from teb_vae.lag_attn_rws.nets.losses import (
    masked_raw_block_per_anchor,
    raw_sample_score,
    validate_student_nu,
)
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask

from .conftest import TINY_STRIDE, build, make_streams, tiny_warmup_kwargs

#: The degrees of freedom the pilot fitted, and the value every test here uses.
NU = 8.0

_LOG_2PI = math.log(2.0 * math.pi)


def _kwargs(**overrides):
    """The guarded tiny set at the tiling stride, with the named leaves moved."""
    return tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, **overrides)


def _random_block(seed: int = 0, shape=(3, 4, 6, 5)):
    """A float64 forecast block: mean, target and log-variance, all of one shape."""
    generator = torch.Generator().manual_seed(seed)
    mu = torch.randn(shape, generator=generator, dtype=torch.float64)
    target = mu + 1.5 * torch.randn(shape, generator=generator, dtype=torch.float64)
    logvar = 0.5 * torch.randn(shape, generator=generator, dtype=torch.float64)
    return mu, target, logvar


def _t_nll(value: torch.Tensor, logvar: torch.Tensor, nu: float = NU) -> torch.Tensor:
    """The reference: ``torch.distributions.StudentT`` at location 0 and scale $\\sigma$."""
    density = torch.distributions.StudentT(
        torch.tensor(nu, dtype=value.dtype), torch.zeros_like(value), torch.exp(0.5 * logvar)
    )
    return -density.log_prob(value)


# =================================================================================================
# The score function
# =================================================================================================
def test_the_term_is_the_student_t_negative_log_density_of_the_residual() -> None:
    mu, target, logvar = _random_block()

    score = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar, student_nu=NU)

    torch.testing.assert_close(score, _t_nll(target - mu, logvar), rtol=1e-10, atol=1e-10)


def test_the_term_is_the_student_t_density_of_the_ar1_innovation() -> None:
    r"""$e_\tau = r_\tau - \phi_c r_{\tau-1}$ with $r_{-1} = 0$ and no masked step: the scale is
    the decoder's own at every step."""
    mu, target, logvar = _random_block(seed=1)
    phi = torch.tensor([0.0, 0.3, -0.5, 0.9, 0.99], dtype=torch.float64)
    residual = target - mu
    previous = torch.cat([torch.zeros_like(residual[..., :1, :]), residual[..., :-1, :]], dim=-2)
    innovation = residual - phi * previous

    score = raw_sample_score(
        mu, target, likelihood="gaussian_nll", logvar=logvar, ar_coef=phi, student_nu=NU
    )

    torch.testing.assert_close(score, _t_nll(innovation, logvar), rtol=1e-10, atol=1e-10)


def test_a_masked_step_is_marginalised_under_the_carried_scale() -> None:
    r"""With step $2$ masked, step $3$ is scored on $e_3 = r_3 - \phi^2 r_1$ under the scale
    $\sqrt{s_3^2 + \phi^2 s_2^2}$; every other valid step keeps its own scale. Only the valid steps
    are compared: the caller multiplies the masked step by zero."""
    mu, target, logvar = _random_block(seed=2, shape=(1, 1, 5, 1))
    phi = torch.tensor([0.7], dtype=torch.float64)
    step_mask = torch.tensor([[[1.0, 1.0, 0.0, 1.0, 1.0]]], dtype=torch.float64)
    r = (target - mu)[0, 0, :, 0]
    s2 = torch.exp(logvar)[0, 0, :, 0]
    p = float(phi)
    innovation = torch.stack([r[0], r[1] - p * r[0], r[3] - p * p * r[1], r[4] - p * r[3]])
    variance = torch.stack([s2[0], s2[1], s2[3] + p * p * s2[2], s2[4]])

    score = raw_sample_score(
        mu, target, likelihood="gaussian_nll", logvar=logvar, ar_coef=phi,
        step_mask=step_mask, student_nu=NU,
    )[0, 0, :, 0]

    expected = _t_nll(innovation, torch.log(variance))
    torch.testing.assert_close(score[[0, 1, 3, 4]], expected, rtol=1e-10, atol=1e-10)


def test_the_block_reducer_forwards_nu_to_the_term() -> None:
    mu, target, logvar = _random_block(seed=3)
    mask = (torch.rand(mu.shape[:-1], generator=torch.Generator().manual_seed(3)) > 0.3).double()
    phi = torch.full((mu.shape[-1],), 0.5, dtype=torch.float64)

    block, _contributing = masked_raw_block_per_anchor(
        mu, target, mask, likelihood="gaussian_nll", logvar=logvar, ar_coef=phi, student_nu=NU
    )
    per_sample = raw_sample_score(
        mu, target, likelihood="gaussian_nll", logvar=logvar, ar_coef=phi, step_mask=mask,
        student_nu=NU,
    )

    torch.testing.assert_close(block, (per_sample * mask[..., None]).sum(dim=(2, 3)))


def test_none_is_the_gaussian_term_bitwise_and_mse_ignores_nu() -> None:
    mu, target, logvar = _random_block(seed=4)
    mu, target, logvar = mu.float(), target.float(), logvar.float()
    diff2 = (target - mu) ** 2

    gaussian = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar, student_nu=None)
    assert torch.equal(gaussian, 0.5 * (_LOG_2PI + logvar + diff2 * torch.exp(-logvar)))
    assert torch.equal(gaussian, raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar))

    squared = raw_sample_score(mu, target, likelihood="mse", student_nu=NU)
    assert torch.equal(squared, raw_sample_score(mu, target, likelihood="mse"))


@pytest.mark.parametrize("bad", [2.0, 1.5, 0.0, -3.0, float("nan"), float("inf")])
def test_a_nu_without_finite_variance_is_refused(bad: float) -> None:
    mu, target, logvar = _random_block(seed=5)
    with pytest.raises(ValueError, match="student_nu"):
        raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar, student_nu=bad)
    with pytest.raises(ValueError, match="student_nu"):
        validate_student_nu(bad)


def test_a_large_nu_tends_to_the_gaussian_term() -> None:
    mu, target, logvar = _random_block(seed=6)
    gaussian = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar)
    heavy = raw_sample_score(mu, target, likelihood="gaussian_nll", logvar=logvar, student_nu=1e8)
    torch.testing.assert_close(heavy, gaussian, rtol=1e-6, atol=1e-6)


# =================================================================================================
# The model
# =================================================================================================
def test_the_gaussian_model_is_unchanged() -> None:
    """No attribute value, the two-key likelihood dict, and the same state dict as a Student-t
    model built from the same seed -- which therefore loads it strictly."""
    torch.manual_seed(0)
    gaussian = build(_kwargs())
    torch.manual_seed(0)
    student = build(_kwargs(forecast_student_nu=NU))

    assert gaussian.forecast_student_nu is None
    assert gaussian.forecast_likelihood_kwargs() == {"cell_mask": None, "ar_coef": None}
    terms = student.forecast_likelihood_kwargs()
    assert terms["student_nu"] == NU and isinstance(terms["student_nu"], float)

    left, right = gaussian.state_dict(), student.state_dict()
    assert list(left) == list(right)
    assert all(torch.equal(left[name], right[name]) for name in left)
    student.load_state_dict(left, strict=True)


@pytest.mark.parametrize("bad", [2.0, 1.0])
def test_the_model_refuses_a_nu_without_finite_variance(bad: float) -> None:
    with pytest.raises(ValueError, match="student_nu"):
        build(_kwargs(forecast_student_nu=bad))


def test_the_objective_scores_both_branches_under_nu() -> None:
    r"""Each branch's ``nll_*_block`` is the block recomputed with the model's own likelihood
    kwargs, $\nu$ included, and it differs from the Gaussian block on the same forecast."""
    kwargs = _kwargs(forecast_student_nu=NU, forecast_ar_residual=True)
    model = build(kwargs)
    y_st, y_ph, u_stream = make_streams(kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(y_st, y_ph, u_stream, anchor_phase=0)
    features = torch.cat([y_st, y_ph], dim=-1)
    weight = torch.ones(y_st.shape[0], model.geometry.t)
    metrics = model.compute_loss(out, features, weight=weight)["metrics"]

    mask, _coverage = forecast_mask(
        model.scored_weight(weight), model.geometry, coverage_floor=model.coverage_floor,
        anchors=out["anchor_index"], anchor_valid=out["anchor_valid"],
    )
    target = model._build_forecast_target(features, out["anchor_index"])
    terms = eval_metrics.forecast_likelihood_terms(model)
    assert terms["student_nu"] == NU

    def _block(branch, **density):
        per_anchor, contributing = masked_raw_block_per_anchor(
            out[f"mu_{branch}"], target, mask, likelihood="gaussian_nll",
            logvar=out[f"logvar_{branch}"], channel_weight=model.target_channel_weight,
            horizon_weight=getattr(model, "horizon_weight", None), **density,
        )
        return per_anchor.sum() / contributing.sum().clamp_min(1.0)

    torch.testing.assert_close(metrics["nll_full_block"], _block("full", **terms), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(metrics["nll_base_block"], _block("base", **terms), rtol=1e-6, atol=1e-6)
    gaussian_terms = {k: v for k, v in terms.items() if k != "student_nu"}
    assert not torch.allclose(_block("full", **gaussian_terms), _block("full", **terms))


# =================================================================================================
# The evaluation
# =================================================================================================
def test_the_half_widths_hold_the_nominal_masses() -> None:
    from scipy import stats

    assert eval_metrics.coverage_half_widths(None) == (1.0, 2.0, 3.0)
    widths = eval_metrics.coverage_half_widths(NU)
    for width, nominal in zip(widths, eval_metrics.COVERAGE_NOMINALS):
        assert stats.t.cdf(width, NU) - stats.t.cdf(-width, NU) == pytest.approx(nominal, abs=1e-12)


def _t_sums(student_nu, n: int = 400_000, seed: int = 0):
    r"""Calibration sums of draws $\varepsilon = \sigma T$, $T \sim t_\nu$, read under ``student_nu``."""
    torch.manual_seed(seed)
    logvar = 0.4 * torch.randn(n // 50, 5, 10).double()
    draws = torch.distributions.StudentT(torch.tensor(NU, dtype=torch.float64)).sample(logvar.shape)
    innovation = torch.exp(0.5 * logvar) * draws
    mask = torch.ones(innovation.shape[:-1], dtype=torch.float64)
    sums = eval_metrics.calibration_sums(
        torch.zeros_like(innovation), logvar, innovation, mask, logvar_clamp=(-6.0, 3.0),
        student_nu=student_nu,
    )
    return eval_metrics.calibration_report(sums, logvar_clamp=(-6.0, 3.0), student_nu=student_nu)


def test_student_t_draws_read_calibrated_under_nu_and_not_under_the_gaussian() -> None:
    student = _t_sums(NU)
    gaussian = _t_sums(None)

    assert student["student_nu"] == NU and student["density"] == "student_t"
    assert gaussian["student_nu"] is None and gaussian["density"] == "gaussian"
    for record in student["coverage"]:
        assert abs(record["observed"] - record["nominal"]) <= 0.01
    assert student["pit"]["max_cdf_deviation"] < 0.01
    assert student["mean_standardised_sq"] == pytest.approx(1.0, abs=0.03)

    tolerance = eval_metrics.DEFAULT_COVERAGE_TAIL_TOLERANCE
    passed = eval_metrics._calibration_verdict(student, tail_tolerance=tolerance)
    failed = eval_metrics._calibration_verdict(gaussian, tail_tolerance=tolerance)
    assert passed.status == "PASS" and passed.values["student_nu"] == NU
    assert failed.status == "FAIL" and failed.values["worst_level_sigma"] == 3.0
    assert "student_nu" not in failed.values


def test_the_student_t_crps_matches_its_expectation_and_the_gaussian_limit() -> None:
    r"""For $X \sim t_\nu$ the mean CRPS is $\tfrac12\mathbb E|X - X'|$; at a very large $\nu$ the
    closed form is the Gaussian one."""
    torch.manual_seed(0)
    density = torch.distributions.StudentT(torch.tensor(NU, dtype=torch.float64))
    x, x_prime = density.sample((400_000,)), density.sample((400_000,))
    closed = float(eval_metrics.student_t_crps(x, NU).mean())
    assert closed == pytest.approx(0.5 * float((x - x_prime).abs().mean()), abs=5e-3)

    z = torch.linspace(-4.0, 4.0, 81, dtype=torch.float64)
    normal_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    normal_pdf = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    gaussian = z * (2.0 * normal_cdf - 1.0) + 2.0 * normal_pdf - 1.0 / math.sqrt(math.pi)
    torch.testing.assert_close(eval_metrics.student_t_crps(z, 1e7), gaussian, rtol=1e-5, atol=1e-5)
