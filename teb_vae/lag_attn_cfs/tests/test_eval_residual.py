r"""Three quantities that are routinely conflated, and the bias that hides in the fourth.

**The two latent quantities are not the same number.** ``delta_mu_rms`` is the RMS of
$\mu^q - \mu^p$ per **element**; ``mu_post_prior_gap_rms`` sums over $d_z$ first, so it is the size
of the belief shift per step. At equal support they differ by exactly $\sqrt{d_z}$ -- which is
what makes the conflation invisible on a $d_z = 1$ fixture and a factor of eight wrong at the
shipped $d_z = 64$. The test builds them from real forward outputs and asserts the ratio.

**Both are on the KL's own support, not the forecast mask's.** In this cell the two differ: the
reconstruction reduces over a sparse $(B, A_{\max}, H)$ anchor mask while the KL reduces over a
dense $(B, T)$ one, so a belief shift averaged over the forecast's support would be a mean over a
different set of steps than the KL it is read beside.

**The forecast difference is not ``pred_gap``.** One is a distance between two forecasts, the
other a difference between two *scores*. Two forecasts can differ everywhere and score identically,
and a source that moves the forecast without improving it is a different finding from one that
does neither.

**Every RMS roots once, at the end.** By Jensen $\operatorname{mean}(\sqrt{x}) \le
\sqrt{\operatorname{mean}(x)}$, so averaging finished per-segment roots is biased **low** -- in
the direction that flatters the model. The direction is asserted rather than assumed: the analysis
reports both numbers, and the biased one must sit at or below the rooted-once one on a frame where
the per-recording spread is real.

**And nothing is converted.** A wavelet modulus has no clinical unit, so every reported RMS is in
the loader's $z$ units. The sibling reports the forecast difference in bpm; the absence of that
column is asserted rather than assumed, because a conversion reintroduced through a copied helper
would produce a plausible-looking number over a channel axis whose statistics span orders of
magnitude.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext
from teb_vae.lag_attn_cfs.eval.analyses import residual as residual_analysis
from teb_vae.lag_attn_cfs.eval.metrics import NORMALISED_UNIT, evaluate_batch

from .conftest import make_stub_batch

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}


@pytest.fixture
def readout(task, perturb_posterior):
    """One real forward's readouts, shared by the three tests that need a genuine drift."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)
    return {
        "columns": evaluate_batch(module, make_stub_batch(seed=3), num_samples=1).columns,
        "d_z": float(module.orig_model.d_z),
    }


# =================================================================================================
# The two latent quantities
# =================================================================================================
def test_the_per_element_and_per_step_drifts_differ_by_the_square_root_of_d_z(readout) -> None:
    r"""Built from a real forward rather than from synthetic columns: what is under test is that
    the collection pass computes two different reductions of $\mu^q - \mu^p$, and a fixture that
    supplied both would be testing the test."""
    per_element = readout["columns"]["delta_mu_sq"]
    per_step = readout["columns"]["mu_post_prior_gap_sq"]
    d_z = readout["d_z"]

    assert torch.allclose(per_step, per_element * d_z, rtol=1e-5)
    # And the rooted figures differ by sqrt(d_z), which is the factor a conflation gets wrong.
    assert float(per_step.mean().sqrt()) == pytest.approx(
        float(per_element.mean().sqrt()) * d_z**0.5, rel=1e-5
    )
    assert float(per_element.mean()) > 0.0, "a zero drift would satisfy this vacuously"


def test_the_squares_are_carried_unrooted_and_the_rooted_column_is_their_root(readout) -> None:
    """The aggregation chain must carry the square; the rooted column stays because it is the
    figure the trainer logs and the headline quotes."""
    columns = readout["columns"]

    assert torch.allclose(columns["delta_mu_rms"], columns["delta_mu_sq"].sqrt())


def test_the_forecast_difference_is_a_distance_and_not_a_difference_of_scores(readout) -> None:
    """Two forecasts can differ everywhere and score identically, so this is a separate readout
    from ``pred_gap`` rather than a rescaling of it."""
    columns = readout["columns"]
    difference = columns["forecast_difference_sq"]

    assert float(difference.min()) >= 0.0, "a squared distance cannot be negative"
    assert float(difference.mean()) > 0.0
    # Unlike pred_gap, which is signed and can be either.
    assert difference.shape == columns["pred_gap"].shape


# =================================================================================================
# The Jensen direction
# =================================================================================================
def _per_guid(
    mean_squares: Sequence[float], per_segment_roots: Optional[Sequence[float]] = None
) -> pd.DataFrame:
    """A per-recording frame carrying the three mean squares and the biased rooted column."""
    frame = pd.DataFrame(
        {
            "forecast_difference_sq": list(mean_squares),
            "delta_mu_sq": list(mean_squares),
            "mu_post_prior_gap_sq": [value * 8.0 for value in mean_squares],
        }
    )
    frame["delta_mu_rms"] = (
        list(per_segment_roots)
        if per_segment_roots is not None
        else [float(np.sqrt(value)) for value in mean_squares]
    )
    return frame


def test_rooting_once_at_the_end_differs_from_averaging_finished_roots() -> None:
    r"""The frame spreads the per-recording squares widely, which is where the two reductions
    differ: $\sqrt{\operatorname{mean}(1, 9)} = 2.236$, while $\operatorname{mean}(1, 3) = 2$."""
    rows = residual_analysis.build_rows(
        _per_guid([1.0, 9.0, 1.0, 9.0]), resamples=200, seed=0
    )
    by_name = {row["metric"]: row for row in rows}

    assert by_name["delta_mu_rms"]["rms_normalised"] == pytest.approx(float(np.sqrt(5.0)))
    assert by_name["delta_mu_rms"]["mean_of_per_segment_rms"] == pytest.approx(2.0)


def test_the_bias_of_averaging_roots_runs_in_the_direction_that_flatters_the_model() -> None:
    """Asserted as a sign, not as a magnitude: Jensen gives the inequality on any input, and a
    single hand-checked example would not say that."""
    rows = residual_analysis.build_rows(
        _per_guid([0.25, 4.0, 1.0, 16.0]), resamples=200, seed=0
    )
    biased = {row["metric"]: row for row in rows}["delta_mu_rms"]

    assert biased["jensen_bias"] < 0.0, (
        "the mean of per-segment roots must sit *below* the rooted-once value; a positive bias "
        "means the two are the wrong way round"
    )
    assert biased["mean_of_per_segment_rms"] < biased["rms_normalised"]


def test_a_frame_with_no_spread_makes_the_two_reductions_agree() -> None:
    """The boundary of the inequality above, which is what says the test discriminates rather
    than always finding a bias."""
    rows = residual_analysis.build_rows(_per_guid([4.0] * 4), resamples=200, seed=0)
    biased = {row["metric"]: row for row in rows}["delta_mu_rms"]

    assert biased["jensen_bias"] == pytest.approx(0.0, abs=1e-12)


# =================================================================================================
# The unit, and the conversion that is not here
# =================================================================================================
def test_every_metric_is_reported_in_z_units_and_nothing_converts() -> None:
    """The sibling puts the forecast difference through a spread conversion into bpm. There is no
    clinical unit for a wavelet modulus, and one pooled RMS cannot survive being pushed back
    through 98 per-channel statistics that span orders of magnitude."""
    rows = residual_analysis.build_rows(_per_guid([1.0] * 4), resamples=200, seed=0)

    for row in rows:
        assert row["unit"] == NORMALISED_UNIT == "normalised"
        assert "rms" not in row, "the converted column was removed rather than repointed"
        assert not [name for name in row if "bpm" in str(name).lower()]
        # The only rooted column is the z-unit one, and it is the root of its own mean square.
        assert row["rms_normalised"] == pytest.approx(float(np.sqrt(row["mean_square"])))


def test_the_summary_states_one_unit_rather_than_deriving_it_from_a_row() -> None:
    """Derived from whichever row converted, the label would silently become ``normalised`` the day
    a conversion was added back and applied to only one metric."""
    result_unit = residual_analysis.NORMALISED_UNIT

    assert result_unit == "normalised"
    assert not hasattr(residual_analysis, "sigma_to_bpm")


# =================================================================================================
# The analysis
# =================================================================================================
def _context(per_sample: pd.DataFrame, record: Optional[Dict[str, Any]] = None) -> AnalysisContext:
    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record=record or {}, retained={},
        results={},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_writes_its_tables_and_states_the_shared_variance_caveat(tmp_path) -> None:
    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b"],
            "forecast_difference_sq": [1.0, 1.0, 4.0, 4.0],
            "delta_mu_sq": [0.25, 0.25, 1.0, 1.0],
            "mu_post_prior_gap_sq": [2.0, 2.0, 8.0, 8.0],
            "delta_mu_rms": [0.5, 0.5, 1.0, 1.0],
        }
    )

    result = residual_analysis.run_residual_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / residual_analysis.ANALYSIS_DIRNAME
    assert (directory / residual_analysis.PER_RECORDING_FILENAME).is_file()
    assert (directory / residual_analysis.SUMMARY_FILENAME).is_file()
    assert [row["metric"] for row in result["metrics"]] == [
        name for _, name, _ in residual_analysis.RMS_METRICS
    ]
    # The caveat travels in the output, not only in the docstring: it weakens the reading in the
    # model's favour, which is the kind that has to be written where the number is.
    assert "shared" in result["caveat"] and "one shared decoder" in result["caveat"]
    assert result["unit"] == NORMALISED_UNIT


def test_the_meaning_of_each_metric_names_the_support_it_was_reduced_over() -> None:
    """The two supports genuinely differ here, so an emitted row that did not say which one it used
    would be a number a reader could only guess the denominator of."""
    meanings = {name: meaning for _, name, meaning in residual_analysis.RMS_METRICS}

    assert "coefficient" in meanings["forecast_difference_rms"]
    assert "KL" in meanings["delta_mu_rms"]
    assert "KL" in meanings["mu_post_prior_gap_rms"]


@pytest.mark.slow
def test_the_real_run_reports_all_three_metrics_with_their_denominators(collected_run) -> None:
    block = collected_run["summary"]["results"]["residual"]

    by_name = {row["metric"]: row for row in block["metrics"]}
    assert set(by_name) == {name for _, name, _ in residual_analysis.RMS_METRICS}
    for row in by_name.values():
        assert row["n"] == collected_run["summary"]["results"]["n_recordings"]
        assert row["rms_normalised"] >= 0.0
    # The relation the two latent quantities must keep, end to end.
    d_z = float(collected_run["summary"]["results"]["latent_health"]["d_z"])
    assert by_name["mu_post_prior_gap_rms"]["rms_normalised"] == pytest.approx(
        by_name["delta_mu_rms"]["rms_normalised"] * d_z**0.5, rel=1e-5
    )
