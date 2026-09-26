r"""The residual readouts: two latent drifts that are routinely conflated, and the RMS bias.

**The two latent quantities are not the same number.** ``delta_mu_rms`` is the RMS of
$\mu^q - \mu^p$ per **element**; ``mu_post_prior_gap_rms`` sums over $d_z$ first, so it is the size
of the belief shift per step. At equal support they differ by exactly $\sqrt{d_z}$ -- which is
what makes the conflation invisible on a $d_z = 1$ fixture and badly wrong at the shipped width.
The test builds them from real forward outputs and asserts the ratio.

**Every RMS roots once, at the end.** By Jensen $\operatorname{mean}(\sqrt{x}) \le
\sqrt{\operatorname{mean}(x)}$, so averaging finished per-segment roots is biased **low** -- in
the direction that flatters the model. The analysis reports both numbers and their difference,
asserted against a hand-computed frame and against the no-spread boundary where they agree.

**Units.** The forecast difference is a spread and converts to bpm through the standard deviation
alone; the latent drifts have no bpm and stay normalised.
"""
from __future__ import annotations

import types
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval.analyses import residual as residual_analysis
from teb_vae.lag_attn_rws.eval.metrics import evaluate_batch

from .conftest import make_stub_batch

#: Bootstrap settings: instant, and seeded.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}

#: The loader statistics the bpm column is converted with, in the tests that assert on it.
NORMALIZATION = {"fhr": {"mean": 140.0, "std": 10.0}}


# =============================================================================
# The two latent quantities
# =============================================================================
def test_the_per_element_and_per_step_drifts_differ_by_the_square_root_of_d_z(
    task, perturb_posterior
) -> None:
    r"""Built from a real forward rather than from synthetic columns: what is under test is that
    the collection pass computes two different reductions of $\mu^q - \mu^p$, and a fixture that
    supplied both would be testing the test."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    torch.manual_seed(0)

    readout = evaluate_batch(module, make_stub_batch(seed=3), num_samples=1)

    per_element = readout.columns["delta_mu_sq"]
    per_step = readout.columns["mu_post_prior_gap_sq"]
    d_z = float(module.orig_model.d_z)
    assert torch.allclose(per_step, per_element * d_z, rtol=1e-5)
    # And the rooted figures differ by sqrt(d_z), which is the factor a conflation gets wrong.
    assert float(per_step.mean().sqrt()) == pytest.approx(
        float(per_element.mean().sqrt()) * d_z**0.5, rel=1e-5
    )
    assert float(per_element.mean()) > 0.0, "a zero drift would satisfy this vacuously"
    # The chain carries the square; the rooted column beside it is the one the trainer logs.
    assert torch.allclose(readout.columns["delta_mu_rms"], per_element.sqrt())


# =============================================================================
# The Jensen direction
# =============================================================================
def _per_guid(mean_squares, per_segment_roots=None) -> pd.DataFrame:
    """A per-recording frame carrying the three mean squares and the biased rooted column."""
    frame = pd.DataFrame(
        {
            "forecast_difference_sq": mean_squares,
            "delta_mu_sq": mean_squares,
            "mu_post_prior_gap_sq": [value * 8.0 for value in mean_squares],
        }
    )
    frame["delta_mu_rms"] = (
        per_segment_roots
        if per_segment_roots is not None
        else [float(np.sqrt(value)) for value in mean_squares]
    )
    return frame


def test_rooting_once_at_the_end_differs_from_averaging_finished_roots() -> None:
    r"""The frame spreads the per-recording squares widely, which is where the two reductions
    differ: $\sqrt{\operatorname{mean}(1, 9)} = 2.236$, while $\operatorname{mean}(1, 3) = 2$."""
    rows = residual_analysis.build_rows(
        _per_guid([1.0, 9.0, 1.0, 9.0]), None, resamples=200, seed=0
    )
    by_name = {row["metric"]: row for row in rows}

    assert by_name["delta_mu_rms"]["rms_normalised"] == pytest.approx(float(np.sqrt(5.0)))
    assert by_name["delta_mu_rms"]["mean_of_per_segment_rms"] == pytest.approx(2.0)
    # Negative: the mean of finished roots sits below, in the direction that flatters the model.
    assert by_name["delta_mu_rms"]["jensen_bias"] == pytest.approx(2.0 - float(np.sqrt(5.0)))


def test_a_frame_with_no_spread_makes_the_two_reductions_agree() -> None:
    """The boundary of the inequality above, which is what says the test discriminates rather
    than always finding a bias."""
    rows = residual_analysis.build_rows(_per_guid([4.0] * 4), None, resamples=200, seed=0)
    biased = {row["metric"]: row for row in rows}["delta_mu_rms"]

    assert biased["jensen_bias"] == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# The unit conversion
# =============================================================================
def test_the_forecast_difference_converts_through_the_spread_path_not_the_affine_one() -> None:
    r"""An RMS of $1$ z-unit is $10$ bpm at $\mathrm{std} = 10$, not $150$. The affine map is the
    plausible-looking wrong answer: it returns a physiologically reasonable number, which is why
    nobody questions it."""
    rows = residual_analysis.build_rows(
        _per_guid([1.0] * 4), NORMALIZATION, resamples=200, seed=0
    )
    by_name = {row["metric"]: row for row in rows}

    assert by_name["forecast_difference_rms"]["rms"] == pytest.approx(10.0)
    assert by_name["forecast_difference_rms"]["unit"] == "bpm"


def test_the_latent_quantities_stay_in_latent_units() -> None:
    """A latent drift has no bpm: it is a distance in the latent's own coordinates, and labelling
    it bpm would be the same category error the conversion above exists to prevent."""
    rows = residual_analysis.build_rows(
        _per_guid([1.0] * 4), NORMALIZATION, resamples=200, seed=0
    )
    by_name = {row["metric"]: row for row in rows}

    for name in ("delta_mu_rms", "mu_post_prior_gap_rms"):
        assert by_name[name]["unit"] == "normalised"
        assert by_name[name]["rms"] == pytest.approx(by_name[name]["rms_normalised"])


def test_absent_statistics_leave_the_label_normalised() -> None:
    rows = residual_analysis.build_rows(_per_guid([1.0] * 4), None, resamples=200, seed=0)

    assert {row["metric"]: row for row in rows}["forecast_difference_rms"]["unit"] == (
        "normalised"
    )


# =============================================================================
# The analysis
# =============================================================================
def _context(per_sample: pd.DataFrame, record: Optional[Dict[str, Any]] = None) -> Any:
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record=record or {}, retained={},
        results={},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_writes_its_tables(tmp_path) -> None:
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
        _context(per_sample, {"normalization": NORMALIZATION}),
        eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None,
    )

    directory = tmp_path / residual_analysis.ANALYSIS_DIRNAME
    assert (directory / residual_analysis.PER_RECORDING_FILENAME).is_file()
    assert (directory / residual_analysis.SUMMARY_FILENAME).is_file()
    assert [row["metric"] for row in result["metrics"]] == [
        name for _, name, _, _ in residual_analysis.RMS_METRICS
    ]
    assert result["unit"] == "bpm"


def test_the_real_run_reports_all_three_metrics_with_their_denominators(evaluated) -> None:
    block = evaluated["summary"]["results"]["residual"]

    by_name = {row["metric"]: row for row in block["metrics"]}
    assert set(by_name) == {name for _, name, _, _ in residual_analysis.RMS_METRICS}
    for row in by_name.values():
        assert row["n"] == evaluated["summary"]["results"]["n_recordings"]
        assert row["rms_normalised"] >= 0.0
    # The relation the two latent quantities must keep, end to end.
    d_z = float(evaluated["summary"]["results"]["latent_health"]["d_z"])
    assert by_name["mu_post_prior_gap_rms"]["rms_normalised"] == pytest.approx(
        by_name["delta_mu_rms"]["rms_normalised"] * d_z**0.5, rel=1e-5
    )
