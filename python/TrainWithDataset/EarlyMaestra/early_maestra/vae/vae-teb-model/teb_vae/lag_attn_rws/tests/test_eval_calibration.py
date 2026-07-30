r"""One known answer pins every calibration statistic at once.

Draw the target from exactly the distribution the decoder claims -- $x \sim \mathcal{N}(\mu,
\sigma^2)$ with the model's own $\mu$ and $\sigma$ -- and every statistic here has an analytic
value simultaneously:

* the PIT is uniform, so its density is $1.0$ in every bin and its worst CDF deviation is
  sampling noise;
* central coverage is $\operatorname{erf}(k/\sqrt{2})$ at each of the three levels;
* the standardised residual variance is $1$;
* and the gain over the homoscedastic MLE is $\approx 0$ when the true $\sigma$ is constant, since
  a constant variance is then the right model too.

That last one is why the miscalibrated cases matter as much as the calibrated one. A statistic
that is right on a calibrated sample and unmoved by a variance that is twice too small is not
measuring calibration, and every case below is paired with its miscalibrated counterpart.

The **nominals** are the second trap. $2\sigma$ central coverage is $0.9545$, not the $0.95$ it is
universally quoted as; a model checked against $0.95$ looks half a point miscalibrated while being
exactly right. They are computed from ``erf`` rather than written down, and that is asserted.
"""
from __future__ import annotations

import math
import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval import metrics as metrics_module
from teb_vae.lag_attn_rws.eval.analyses import calibration as calibration_analysis
from teb_vae.lag_attn_rws.eval.metrics import (
    COVERAGE_LEVELS,
    COVERAGE_NOMINALS,
    FAIL,
    INCONCLUSIVE,
    PASS,
    Aggregate,
    build_verdicts,
    calibration_report,
    calibration_sums,
)

#: The clamp every synthetic case is laid out over -- the shipped one.
CLAMP = (-5.0, 3.0)

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


def _gaussian_case(
    *, sigma_true: float, sigma_model: float, batch: int = 24, seed: int = 0
) -> Dict[str, Any]:
    r"""Accumulate the calibration sums for a target drawn at $\sigma_{\rm true}$ and modelled at
    $\sigma_{\rm model}$.

    Args:
        sigma_true: The standard deviation the target is actually drawn with.
        sigma_model: The standard deviation the decoder claims.
        batch: Samples. Sized so the case holds $\\approx 6 \\times 10^4$ raw samples: a
            twenty-bin PIT then has $\\approx 3000$ per bin, whose sampling noise is under $2\\%$
            of the flat density, and the three-sigma tail still holds $\\approx 170$ exceedances.
            Every tolerance below is set against those two numbers rather than by taste.
        seed: Draw seed.

    Returns:
        The report, so every assertion below reads the finished statistics.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, 40, 4, 16)
    mu = torch.zeros(shape, dtype=torch.float64)
    target = mu + torch.randn(shape, generator=generator, dtype=torch.float64) * sigma_true
    logvar = torch.full(shape, 2.0 * math.log(sigma_model), dtype=torch.float64)
    mask = torch.ones(shape[:3], dtype=torch.float64)
    return calibration_report(
        calibration_sums(mu, logvar, target, mask, logvar_clamp=CLAMP), logvar_clamp=CLAMP
    )


# =============================================================================
# The nominals
# =============================================================================
def test_the_nominals_are_the_erf_values_and_not_the_ones_people_quote() -> None:
    assert COVERAGE_LEVELS == (1, 2, 3)
    assert COVERAGE_NOMINALS == pytest.approx((0.682689492, 0.954499736, 0.997300204), abs=1e-9)
    assert COVERAGE_NOMINALS[1] != pytest.approx(0.95, abs=1e-3), (
        "two-sigma coverage is 0.9545; a model checked against 0.95 reads as miscalibrated while "
        "being exactly right"
    )


# =============================================================================
# The exactly calibrated case
# =============================================================================
def test_an_exactly_calibrated_gaussian_gives_a_flat_pit_and_nominal_coverage() -> None:
    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)

    density = np.asarray(report["pit"]["density"], dtype=np.float64)
    assert density == pytest.approx(np.ones(density.size), abs=0.05)
    assert report["pit"]["max_cdf_deviation"] < 0.02
    for record, nominal in zip(report["coverage"], COVERAGE_NOMINALS):
        assert record["observed"] == pytest.approx(nominal, abs=0.01)
        assert record["nominal"] == pytest.approx(nominal)
    assert report["mean_standardised_sq"] == pytest.approx(1.0, rel=0.02)


def test_an_exactly_calibrated_case_gains_nothing_over_one_constant_variance() -> None:
    """The homoscedastic MLE is fitted to the very residuals being scored, so on a genuinely
    homoscedastic sample it *is* the right model and the learned variance earns nothing. A gain
    that appeared here would be an arithmetic error, not a finding."""
    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)

    assert report["nll"]["gain_per_raw_sample"] == pytest.approx(0.0, abs=0.01)
    assert report["nll"]["homoscedastic_sigma"] == pytest.approx(1.0, rel=0.02)


def test_the_crps_of_a_calibrated_standard_normal_is_its_analytic_value() -> None:
    r"""$\mathbb{E}\,\mathrm{CRPS}(\mathcal{N}(0,1), X) = 1/\sqrt{\pi}$ for $X \sim
    \mathcal{N}(0,1)$.

    From the kernel form $\mathrm{CRPS} = \mathbb{E}|X - x| - \tfrac{1}{2}\mathbb{E}|X - X'|$:
    with $x$ itself standard normal both expectations are over a difference of two independent
    standard normals, which is $\mathcal{N}(0, 2)$ with mean absolute value $2/\sqrt{\pi}$ -- so
    the first term is $2/\sqrt{\pi}$ and the second $1/\sqrt{\pi}$. A closed form, so this pins
    the coefficients of the closed-form expression rather than its sign.
    """
    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)

    assert report["crps_normalised"] == pytest.approx(1.0 / math.sqrt(math.pi), rel=0.02)


def test_the_crps_scales_with_the_spread_it_is_measured_in() -> None:
    """CRPS has the units of the signal, which is why it can be quoted in bpm at all -- and why
    the conversion to bpm is the spread one rather than the affine one."""
    narrow = _gaussian_case(sigma_true=1.0, sigma_model=1.0)
    wide = _gaussian_case(sigma_true=2.0, sigma_model=2.0)

    assert wide["crps_normalised"] == pytest.approx(2.0 * narrow["crps_normalised"], rel=0.05)


# =============================================================================
# The miscalibrated cases, which are what make the above non-vacuous
# =============================================================================
def test_an_over_confident_variance_shows_up_in_the_tails_and_the_pit() -> None:
    """Half the true spread claimed: the standardised residuals are twice too large, so coverage
    falls short at every level and the PIT piles up at both ends."""
    report = _gaussian_case(sigma_true=1.0, sigma_model=0.5)

    density = np.asarray(report["pit"]["density"], dtype=np.float64)
    assert report["mean_standardised_sq"] == pytest.approx(4.0, rel=0.05)
    assert all(
        record["observed"] < record["nominal"] - 0.01 for record in report["coverage"]
    )
    assert density[0] > 1.5 and density[-1] > 1.5, "a U-shaped PIT is what too-small looks like"


def test_an_under_confident_variance_fails_in_the_other_direction() -> None:
    report = _gaussian_case(sigma_true=1.0, sigma_model=2.0)

    density = np.asarray(report["pit"]["density"], dtype=np.float64)
    assert report["mean_standardised_sq"] == pytest.approx(0.25, rel=0.05)
    assert report["coverage"][0]["observed"] > report["coverage"][0]["nominal"]
    assert density[density.size // 2] > 1.2, "a peaked PIT is what too-large looks like"


def test_a_homoscedastic_model_is_beaten_by_nothing_and_says_so() -> None:
    """The gain is against the *best constant* variance, so a model that is itself constant but
    wrong still gains nothing: the MLE would simply pick the right constant."""
    report = _gaussian_case(sigma_true=1.0, sigma_model=0.5)

    assert report["nll"]["gain_per_raw_sample"] < 0.0, (
        "an over-confident constant variance is worse than the fitted constant, so the gain is "
        "negative rather than zero"
    )


def test_an_empty_census_is_a_skip_rather_than_a_number() -> None:
    assert calibration_report({}) == {}
    assert calibration_report({"count": 0.0}) == {}


# =============================================================================
# The verdict
# =============================================================================
def _by_name(verdicts) -> Dict[str, Any]:
    return {verdict.name: verdict for verdict in verdicts}


def _verdict_for(report: Dict[str, Any]):
    return _by_name(
        build_verdicts(Aggregate(overall={}, kld_per_dim=[0.4, 0.3]), calibration=report)
    )["calibration_near_nominal"]


def test_the_calibration_verdict_passes_on_the_calibrated_case_and_fails_the_other() -> None:
    assert _verdict_for(_gaussian_case(sigma_true=1.0, sigma_model=1.0)).status == PASS
    assert _verdict_for(_gaussian_case(sigma_true=1.0, sigma_model=0.5)).status == FAIL


def test_the_verdict_carries_every_level_it_judged() -> None:
    verdict = _verdict_for(_gaussian_case(sigma_true=1.0, sigma_model=1.0))

    for level, nominal in zip(COVERAGE_LEVELS, COVERAGE_NOMINALS):
        assert verdict.values[f"nominal_{level}_sigma"] == pytest.approx(nominal)
        assert 0.0 <= verdict.values[f"observed_{level}_sigma"] <= 1.0


def test_a_run_with_no_predictive_distribution_is_inconclusive() -> None:
    verdict = _verdict_for({})

    assert verdict.status == INCONCLUSIVE
    assert "mse" in verdict.detail


# =============================================================================
# The analysis, its skip, and its recommendation
# =============================================================================
def _context(
    *, likelihood: str, report: Optional[Dict[str, Any]] = None, floor: float = 0.0,
    ceil: float = 0.0,
) -> Any:
    """An analysis context carrying a finished calibration census and the clamp fractions."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b"],
            "mean_logvar_full": [-0.5, -0.4, -0.6, -0.5],
            "logvar_full_floor_frac": [floor] * 4,
            "logvar_full_ceil_frac": [ceil] * 4,
        }
    )
    collection = types.SimpleNamespace(
        per_sample=per_sample,
        per_anchor=pd.DataFrame(),
        record={
            "likelihood": likelihood,
            "bounds": {"logvar_clamp": list(CLAMP), "logvar_margin": 0.4},
            "normalization": {"fhr": {"mean": 140.0, "std": 10.0}},
        },
        retained={},
        results={"likelihood": likelihood, "calibration": report or {}, "verdicts": []},
    )
    return AnalysisContext(collection=collection, config={})


def test_an_mse_checkpoint_records_a_skip_rather_than_a_number(tmp_path) -> None:
    result = calibration_analysis.run_calibration_analysis(
        _context(likelihood="mse"), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    assert result["skipped"] is True
    assert "mse" in result["reason"]
    # None rather than zero: this analysis scored no population, and a zero would enter the
    # coverage block as a disagreement with every analysis that did.
    assert result["n_samples"] is None
    assert result["files"] == []
    assert not (tmp_path / calibration_analysis.ANALYSIS_DIRNAME).exists()


def test_the_analysis_writes_its_tables_and_both_figures(tmp_path) -> None:
    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)

    result = calibration_analysis.run_calibration_analysis(
        _context(likelihood="gaussian_nll", report=report),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / calibration_analysis.ANALYSIS_DIRNAME
    for name in (
        calibration_analysis.COVERAGE_FILENAME,
        calibration_analysis.PIT_FILENAME,
        calibration_analysis.LOGVAR_FILENAME,
        calibration_analysis.PER_RECORDING_FILENAME,
        calibration_analysis.PIT_FIGURE,
        calibration_analysis.LOGVAR_FIGURE,
    ):
        assert (directory / name).is_file(), name
    coverage = pd.read_csv(directory / calibration_analysis.COVERAGE_FILENAME)
    assert list(coverage["level_sigma"]) == list(COVERAGE_LEVELS)
    assert result["crps_unit"] == "bpm"
    assert result["crps"] == pytest.approx(result["crps_normalised"] * 10.0, rel=1e-6)


def test_the_recommendation_names_the_config_key_it_would_change(tmp_path) -> None:
    result = calibration_analysis.run_calibration_analysis(
        _context(likelihood="gaussian_nll", report=_gaussian_case(sigma_true=1.0, sigma_model=1.0)),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    recommendation = result["recommendation"]
    assert recommendation["config_key"] == "model_config.VAE_model.logvar_clamp"
    # Nothing is binding on a calibrated case, so the recommendation is explicitly *no change*.
    assert recommendation["change_recommended"] is False
    assert recommendation["proposed"] == list(CLAMP)


def test_a_binding_floor_produces_a_wider_proposed_clamp(tmp_path) -> None:
    """The other branch: a recommendation that is emitted unconditionally is one that gets
    applied unconditionally.

    The census and the floor fraction are two readings of the *same* log-variance, so the fixture
    makes them agree: a decoder whose variance genuinely sits within the margin of the clamp
    floor. A binding fraction injected beside a census concentrated four nats away is a state that
    cannot occur, and it lets the assertion be satisfied by a clamp that was never moved.
    """
    # $2\log\sigma = -4.9$, inside the 0.4-nat margin of the $-5.0$ floor.
    on_the_floor = math.exp(-4.9 / 2.0)
    report = _gaussian_case(sigma_true=1.0, sigma_model=on_the_floor)

    result = calibration_analysis.run_calibration_analysis(
        _context(likelihood="gaussian_nll", report=report, floor=1.0),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    recommendation = result["recommendation"]
    assert recommendation["change_recommended"] is True
    assert recommendation["floor_binds"] is True
    # The property this test is named for, with no disjunct that another state could satisfy:
    # `proposed[0] < q0.001` holds for a clamp left exactly where it was on any case whose mass
    # sits inside the bound, which is every case but this one.
    assert recommendation["proposed"][0] < CLAMP[0]
    assert "logvar_clamp" in recommendation["detail"]


def test_the_logvar_figure_marks_both_clamp_margins_where_the_model_measures_them() -> None:
    r"""At exactly $\mathrm{lo} + 0.4$ and $\mathrm{hi} - 0.4$, read from the clamp the checkpoint
    carries. The bound is a sigmoid, so the asymptote is never reached and mass *at* it would be
    invisible -- the margin is what "pinned" means in every number this pipeline reports."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)
    histogram = calibration_analysis.logvar_frame(report)
    bounds = {"logvar_clamp": list(CLAMP), "logvar_margin": 0.4}

    figure = calibration_analysis.build_logvar_figure(histogram, bounds)
    try:
        axis = figure.axes[0]
        margins = sorted(
            float(line.get_xdata()[0]) for line in axis.lines
            if line.get_linestyle() == "--"
        )
    finally:
        shared_figures.plt.close(figure)

    assert margins == pytest.approx([CLAMP[0] + 0.4, CLAMP[1] - 0.4])


def test_the_pit_figure_draws_the_uniform_reference_and_the_reliability_diagonal() -> None:
    from teb_vae.lag_attn.eval import figures as shared_figures

    report = _gaussian_case(sigma_true=1.0, sigma_model=1.0)
    pit = calibration_analysis.pit_frame(report)
    coverage = calibration_analysis.coverage_frame(report)

    figure = calibration_analysis.build_pit_figure(pit, coverage)
    try:
        histogram_axis, reliability_axis = figure.axes[0], figure.axes[1]
        references = [
            float(line.get_ydata()[0]) for line in histogram_axis.lines
            if len(set(np.asarray(line.get_ydata(), dtype=np.float64))) == 1
        ]
        diagonal = [
            (np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
            for line in reliability_axis.lines
            if line.get_label() == "calibrated"
        ]
    finally:
        shared_figures.plt.close(figure)

    assert 1.0 in references, "the uniform density a calibrated PIT sits at must be drawn"
    assert len(diagonal) == 1
    assert diagonal[0][0] == pytest.approx([0.0, 1.0])
    assert diagonal[0][1] == pytest.approx([0.0, 1.0])


# =============================================================================
# Through the real readout path
# =============================================================================
def test_the_census_is_not_accumulated_under_a_likelihood_that_has_no_variance(
    task, perturb_posterior
) -> None:
    """Under ``mse`` the decoder's log-variance head is never fitted, so a PIT of its output would
    be arithmetic over an untrained tensor -- a number, and a meaningless one."""
    from .conftest import make_stub_batch

    module = task(hparams={"likelihood": "mse"})
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = metrics_module.evaluate_batch(module, make_stub_batch(seed=2), num_samples=1)

    assert readout.calibration_sums == {}


def test_the_census_is_accumulated_under_gaussian_nll(task, perturb_posterior) -> None:
    from .conftest import make_stub_batch

    module = task(hparams={"likelihood": "gaussian_nll"})
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = metrics_module.evaluate_batch(module, make_stub_batch(seed=2), num_samples=1)
    report = calibration_report(
        readout.calibration_sums, logvar_clamp=module.orig_model.logvar_clamp
    )

    assert report["n_raw_samples"] > 0
    assert len(report["coverage"]) == len(COVERAGE_LEVELS)
    assert report["pit"]["n_bins"] == metrics_module.PIT_BINS
    assert report["logvar"]["n_bins"] == metrics_module.LOGVAR_BINS


@pytest.mark.slow
def test_a_fitted_gaussian_checkpoint_produces_the_whole_block(fitted_evaluated) -> None:
    """End to end on the only fixture that has an observation model at all.

    Everything above drives the arithmetic directly; this is what says the census reaches the
    summary, the analysis reads it, and the verdict resolves to something other than the skip the
    ``mse`` fixture produces -- three seams a synthetic report cannot exercise.
    """
    results = fitted_evaluated["summary"]["results"]
    block = results["calibration"]

    assert block.get("skipped") is not True
    assert block["likelihood"] == "gaussian_nll"
    assert block["n_raw_samples"] > 0
    assert [row["level_sigma"] for row in block["coverage"]] == list(COVERAGE_LEVELS)
    assert block["calibration_verdict"]["status"] in {"PASS", "FAIL"}
    assert block["recommendation"]["config_key"] == "model_config.VAE_model.logvar_clamp"

    directory = fitted_evaluated["results_dir"] / calibration_analysis.ANALYSIS_DIRNAME
    for name in (
        calibration_analysis.COVERAGE_FILENAME,
        calibration_analysis.PIT_FIGURE,
        calibration_analysis.LOGVAR_FIGURE,
    ):
        assert (directory / name).is_file(), name


@pytest.mark.slow
def test_the_calibration_headline_scalars_resolve_under_gaussian_nll(fitted_evaluated) -> None:
    """The conditional half of the headline guard: these three are ``None`` under ``mse`` by
    construction, and a registry entry that never resolved *anywhere* would be a number the
    acceptance gate silently reads as absent."""
    headline = fitted_evaluated["summary"]["results"]["headline"]

    for name in (
        "calibration_mean_standardised_sq",
        "calibration_pit_max_cdf_deviation",
        "calibration_nll_gain_per_raw_sample",
    ):
        assert headline[name] is not None, name
        assert np.isfinite(float(headline[name])), name


def test_the_census_sums_over_batches_exactly(task, perturb_posterior) -> None:
    """The accumulator is exact against a single pass over the same data, because addition is --
    which is what lets a real split's $10^9$ raw samples be summarised in $H$ floats."""
    from .conftest import make_stub_batch

    module = task(hparams={"likelihood": "gaussian_nll"})
    perturb_posterior(module.orig_model)
    module.eval()
    batch = make_stub_batch(seed=2)

    torch.manual_seed(0)
    once = metrics_module.evaluate_batch(module, batch, num_samples=1).calibration_sums
    torch.manual_seed(0)
    again = metrics_module.evaluate_batch(module, batch, num_samples=1).calibration_sums

    for name, value in once.items():
        assert torch.allclose(value + again[name], 2.0 * value), name
