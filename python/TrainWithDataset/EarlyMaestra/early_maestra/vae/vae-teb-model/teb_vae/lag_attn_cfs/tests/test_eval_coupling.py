r"""The coupling readout, its denominator, and the interval on it.

Three things are pinned here, and each catches a different way a coupling number is reported
wrongly.

**Known answers on synthetic per-recording vectors.** A vector that is positive on every recording
must report a positive fraction of exactly $1.0$ with a finite interval; one that is positive on
three of four must report $0.75$. Nothing about the model is involved -- this is the arithmetic,
and it is the arithmetic that would be wrong if the fraction were computed over segments, or over
the wrong axis, or against the wrong sign.

**The denominator is visible.** ``np.nan > 0`` is ``False``, so a recording that scored no anchors
-- and therefore measured *nothing* -- silently counts as evidence against a positive gap. The
seeded-with-NaN case is what distinguishes a fraction that excluded and counted them from one that
quietly voted them down: the same data reports $2/2$ under the first and $2/4$ under the second.

**The two estimators are labelled by path.** ``pred_gap`` is one subtraction on two different
estimators of the same quantity -- the Monte Carlo marginalised score and the single-draw training
path -- and a table carrying both under one name is unreadable. Every row here names its column
and its path.

**The percentages have known answers and refuse to invent one.** A quartered mean-square error is
a $75\%$ MSE reduction and a $50\%$ RMSE reduction -- two different numbers off one ratio, so an
implementation that rooted the wrong quantity is caught by asserting both. Beside them, the three
ways a percentage lies rather than fails: a zero-error denominator reporting ``inf`` instead of
``NaN``; a source that made the forecast *worse* being clipped to zero instead of reported
negative; and a missing $H \cdot C_{\mathrm{keep}}$ being replaced by a guessed block width, which
would rescale the likelihood-space answer with nothing raising.

**The block width is this cell's own, and it is not a constant of the architecture.** The
denominator is $H \cdot C_{\mathrm{keep}} = 15 \times 98 = 1470$ under the shipped warm-up budget,
where $C_{\mathrm{keep}}$ is whatever that budget left standing -- so the percentage is
*budget-local*, the record says so in words, and the value is read off the collection record's
``block_width`` rather than multiplied out from a horizon and a per-step constant this target
domain does not have.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval.analyses import coupling as coupling_analysis
from teb_vae.lag_attn_cfs.eval.frames import positive_fraction

#: Bootstrap settings for the tests: enough resamples for a stable interval, few enough to be
#: instant. The seed is what makes every interval below reproducible.
EVAL_CONFIG = {"bootstrap_resamples": 200, "seed": 0}

#: The shipped block width, $H \cdot C_{\mathrm{keep}} = 15 \times 98$. Written out rather than
#: imported from the model, because a test that read the constant from the implementation it is
#: checking would assert only that the line equals itself.
BLOCK_WIDTH = 1470.0


def _per_guid(**columns: List[float]) -> pd.DataFrame:
    """A per-recording frame carrying the named columns and NaN for everything else it needs."""
    length = len(next(iter(columns.values())))
    frame = pd.DataFrame(columns)
    for name in coupling_analysis.VALUE_COLUMNS:
        if name not in frame.columns:
            frame[name] = np.full(length, np.nan)
    return frame


# =============================================================================
# The positive fraction, and its denominator
# =============================================================================
def test_an_all_positive_gap_reports_a_fraction_of_one_with_a_finite_interval() -> None:
    per_guid = _per_guid(
        mc_pred_gap=[0.5, 1.5, 2.5, 3.5],
        mc_nll_base_block=[10.5, 11.5, 12.5, 13.5],
        mc_nll_full_block=[10.0, 10.0, 10.0, 10.0],
    )

    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)
    headline = rows[0]

    assert headline["metric"] == "pred_gap_mc_nats"
    assert headline["positive_fraction"] == pytest.approx(1.0)
    assert headline["n_recordings_scored"] == 4
    assert np.isfinite(headline["ci_lo"]) and np.isfinite(headline["ci_hi"])
    assert headline["ci_lo"] <= headline["mean"] <= headline["ci_hi"]


def test_a_mixed_gap_reports_the_fraction_that_is_actually_positive() -> None:
    """Non-vacuity for the case above: an implementation returning 1.0 unconditionally passes it."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 1.0, 1.0, -1.0]), resamples=200, seed=0
    )

    assert rows[0]["positive_fraction"] == pytest.approx(0.75)
    assert rows[0]["n_positive"] == 3


def test_unscored_recordings_are_excluded_and_counted_rather_than_voted_down() -> None:
    r"""The one that matters. ``np.nan > 0`` is ``False``, so a recording that measured nothing
    would otherwise be counted as evidence *against* a positive gap -- and a run whose coverage
    collapsed would report a falling positive fraction rather than a falling $n$."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 2.0, np.nan, np.nan]), resamples=200, seed=0
    )

    assert rows[0]["positive_fraction"] == pytest.approx(1.0), (
        "the two NaN recordings must be excluded, not counted as non-positive"
    )
    assert rows[0]["n_recordings_scored"] == 2
    assert rows[0]["n_recordings_dropped_not_finite"] == 2


def test_the_denominator_helper_reports_both_counts_on_an_all_nan_input() -> None:
    record = positive_fraction([np.nan, np.nan])

    assert np.isnan(record["fraction"])
    assert record["n"] == 0 and record["n_dropped_not_finite"] == 2


# =============================================================================
# The two estimators
# =============================================================================
def test_both_pred_gap_estimators_are_reported_and_labelled_by_path() -> None:
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[1.0, 2.0, 3.0], pred_gap=[1.1, 2.1, 3.1]), resamples=200, seed=0
    )

    assert [row["metric"] for row in rows] == [
        "pred_gap_mc_nats", "pred_gap_train_path_nats"
    ]
    assert [row["source_column"] for row in rows] == ["mc_pred_gap", "pred_gap"]
    assert "marginalised" in rows[0]["score_path"]
    assert "single-draw" in rows[1]["score_path"]
    # Each estimator's own values, not one standing for both.
    assert rows[0]["mean"] == pytest.approx(2.0)
    assert rows[1]["mean"] == pytest.approx(2.1)


def test_the_paired_test_runs_on_the_two_block_scores_the_gap_is_the_difference_of() -> None:
    """Paired rather than unpaired: each recording contributes both scores, and pairing removes
    the between-recording variance that dominates every readout here."""
    per_guid = _per_guid(
        mc_pred_gap=[0.5] * 6,
        mc_nll_base_block=[10.5] * 6,
        mc_nll_full_block=[10.0] * 6,
    )

    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)

    assert rows[0]["wilcoxon_n_pairs"] == 6
    assert rows[0]["wilcoxon_median_difference"] == pytest.approx(0.5)


def test_quantiles_travel_beside_the_mean() -> None:
    """A mean and an interval describe a symmetric distribution; these are routinely skewed."""
    rows = coupling_analysis.build_gap_rows(
        _per_guid(mc_pred_gap=[0.0, 1.0, 2.0, 3.0, 100.0]), resamples=200, seed=0
    )

    assert rows[0]["q50"] == pytest.approx(2.0)
    assert rows[0]["mean"] > rows[0]["q75"], "the skew is what the quantiles exist to show"


# =============================================================================
# The KL, reported as a description
# =============================================================================
def test_the_kl_rows_name_the_unfloored_value_and_the_control_beside_it() -> None:
    rows = coupling_analysis.build_kl_rows(
        _per_guid(
            source_conditioned_kl_raw=[3.0, 4.0, 5.0],
            source_conditioned_kl_shuffled_raw=[3.5, 4.5, 5.5],
        ),
        resamples=200, seed=0,
    )

    assert [row["metric"] for row in rows] == list(coupling_analysis.KL_COLUMNS)
    assert "unfloored" in rows[0]["score_path"]
    assert rows[0]["mean"] == pytest.approx(4.0)
    assert rows[1]["mean"] == pytest.approx(4.5)


def test_no_headline_kl_column_is_the_floored_one() -> None:
    """Free bits are applied per dimension per step before summing, so the floored value exceeds
    the raw one by construction and hides a collapsed source pathway."""
    assert all("train" not in column for column in coupling_analysis.KL_COLUMNS)


def test_the_availability_clock_control_is_not_this_analysis_to_report() -> None:
    """``kld_source_null`` is on the same per-sample table and is deliberately **not** reduced
    here: it is the source-null analysis's subject, and a second reduction of it under a coupling
    heading would put two differently-aggregated copies of one control in one summary."""
    assert "kld_source_null" not in coupling_analysis.VALUE_COLUMNS
    assert "coupling_minus_clock" not in coupling_analysis.VALUE_COLUMNS


# =============================================================================
# The figure
# =============================================================================
def test_the_distribution_figure_marks_zero_and_the_interval_on_the_mean() -> None:
    """Zero is the finding: a distribution straddling it is a different result from one sitting
    above it, and a histogram without that line invites a reader to eyeball the sign."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    per_guid = _per_guid(mc_pred_gap=[1.0, 2.0, 3.0, 4.0], pred_gap=[1.0, 2.0, 3.0, 4.0])
    rows = coupling_analysis.build_gap_rows(per_guid, resamples=200, seed=0)

    figure = coupling_analysis.build_distribution_figure(per_guid, rows)
    try:
        axis = figure.axes[0]
        # ``axvline`` draws under a blended transform, so its x data are in data coordinates.
        vertical = [
            float(line.get_xdata()[0]) for line in axis.lines
            if len(set(np.asarray(line.get_xdata(), dtype=np.float64))) == 1
        ]
        spans = [
            (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
            for patch in axis.patches
            if float(patch.get_width()) > 0.0 and patch.get_label().startswith("95%")
        ]
        n_panels = len(figure.axes)
    finally:
        shared_figures.plt.close(figure)

    assert 0.0 in vertical, "the no-improvement reference must be drawn"
    assert spans == [(pytest.approx(rows[0]["ci_lo"]), pytest.approx(rows[0]["ci_hi"]))]
    assert n_panels == 2, "the second panel is the two estimators side by side"


# =============================================================================
# The percentages
# =============================================================================
#: A collection record the likelihood-space percentage is defined under: the shipped block width
#: $H \cdot C_{\mathrm{keep}} = 15 \times 98 = 1470$, and a likelihood whose block score is a
#: log-density.
GEOMETRY = {
    "geometry": {"horizon": 15, "target_kept_width": 98, "block_width": 1470},
    "likelihood": coupling_analysis.LIKELIHOOD_PERCENT_REQUIRES,
}


def test_the_error_space_percentages_are_the_documented_ratios() -> None:
    r"""A quartered mean-square error is a $75\%$ MSE reduction and a $50\%$ RMSE one.

    The two are not interchangeable and the factor between them is not $2$ in general, so a
    known answer on both is what catches an implementation that rooted the wrong quantity.
    """
    per_guid = _per_guid(sq_error_base=[4.0, 4.0], sq_error_full=[1.0, 1.0])

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)

    assert columns["pred_gap_mse_pct"] == pytest.approx([75.0, 75.0])
    assert columns["pred_gap_rmse_pct"] == pytest.approx([50.0, 50.0])


def test_the_two_error_space_percentages_never_disagree_about_the_sign() -> None:
    """They are one ratio under a root, so a recording the source helped in one must be helped in
    the other. An implementation dividing twice could break this and look plausible."""
    per_guid = _per_guid(
        sq_error_base=[4.0, 1.0, 2.0, 9.0], sq_error_full=[1.0, 4.0, 2.0, 3.0]
    )

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)

    assert np.array_equal(
        np.sign(columns["pred_gap_mse_pct"]), np.sign(columns["pred_gap_rmse_pct"])
    )


def test_a_zero_error_denominator_reports_nothing_rather_than_an_infinite_improvement() -> None:
    """A recording whose target-only branch is exactly right is degenerate. ``NaN`` says so;
    ``inf`` would be reported as a spectacular result, and would fail the headline's own
    finiteness check on the way out."""
    per_guid = _per_guid(sq_error_base=[0.0, 4.0], sq_error_full=[1.0, 1.0])

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)
    rows = coupling_analysis.build_percent_rows(
        per_guid.assign(**columns), resamples=200, seed=0
    )
    row = next(row for row in rows if row["metric"] == "pred_gap_mse_pct")

    assert np.isnan(columns["pred_gap_mse_pct"][0])
    assert not np.isinf(columns["pred_gap_mse_pct"]).any()
    # Excluded and counted, never averaged in.
    assert row["n_recordings_scored"] == 1
    assert row["n_recordings_dropped_not_finite"] == 1


def test_the_likelihood_percentage_is_the_per_coefficient_density_ratio() -> None:
    r"""$100(e^{\Delta/(H C_{\mathrm{keep}})} - 1)$, and exactly $0$ at $\Delta = 0$ -- the natural
    zero that makes this a percentage at all."""
    per_guid = _per_guid(mc_pred_gap=[0.0, 14.7])

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)

    # $14.7 / 1470 = 0.01$ exactly, and the second value is hand-computed rather than restated as
    # the implementation's own expression, which would assert only that the line equals itself.
    assert columns["pred_gap_mc_likelihood_pct"] == pytest.approx([0.0, 1.0050167084168058])


def test_a_source_that_made_the_forecast_worse_reports_a_negative_percentage() -> None:
    """Not clipped at zero, and counted against in the positive fraction: a source that hurt is a
    finding, and a percentage floored at zero would report it as neutral."""
    per_guid = _per_guid(
        sq_error_base=[1.0, 1.0, 1.0, 1.0], sq_error_full=[0.25, 0.25, 0.25, 4.0]
    )

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)
    rows = coupling_analysis.build_percent_rows(
        per_guid.assign(**columns), resamples=200, seed=0
    )
    row = next(row for row in rows if row["metric"] == "pred_gap_rmse_pct")

    assert columns["pred_gap_rmse_pct"][3] == pytest.approx(-100.0)
    assert row["positive_fraction"] == pytest.approx(0.75)
    assert row["n_positive"] == 3


def test_an_mse_checkpoint_gets_no_likelihood_percentage() -> None:
    """Under ``'mse'`` a block score is a sum of squared errors, not a log-density -- the model's
    own marginaliser says so and averages rather than ``logsumexp``-ing for the same reason. So
    ``exp(gap / (H*C_keep))`` is not a density ratio, and reporting it as "extra probability
    density" would be a false statement about the run rather than an imprecise one."""
    support = coupling_analysis.likelihood_percent_support(
        {"geometry": {"horizon": 15, "block_width": 1470}, "likelihood": "mse"}
    )

    assert support["skipped"] is True
    assert support["coefficients_per_anchor"] is None
    assert "mse" in support["reason"] and "log-density" in support["reason"]
    # The error-space percentages do not care: a squared error is a squared error either way.
    columns = coupling_analysis.percent_columns(
        _per_guid(sq_error_base=[4.0], sq_error_full=[1.0], mc_pred_gap=[14.7]),
        samples_per_anchor=support["coefficients_per_anchor"],
    )
    assert set(columns) == set(coupling_analysis.ERROR_SPACE_PERCENTS)
    assert columns["pred_gap_mse_pct"] == pytest.approx([75.0])


def test_an_unknown_likelihood_is_a_skip_rather_than_a_pass() -> None:
    """A record that does not say how it was scored cannot be assumed to have been scored the one
    way this readout is defined under. The key predates the percentage, so its absence means the
    tables are wrong rather than old."""
    support = coupling_analysis.likelihood_percent_support(
        {"geometry": {"horizon": 15, "block_width": 1470}}
    )

    assert support["skipped"] is True
    assert support["coefficients_per_anchor"] is None


def test_a_degenerate_block_width_is_refused_like_an_absent_one() -> None:
    """A geometry present but naming a zero-width forecast block would divide by zero."""
    assert coupling_analysis.coefficients_per_anchor(
        {"geometry": {"horizon": 15, "block_width": 0}}
    ) is None


def test_the_block_width_is_read_rather_than_multiplied_out_from_the_raw_grid() -> None:
    r"""The one edit this fork had to make, and the one that would be invisible if it were wrong.

    In the raw cells the denominator is $H \cdot R$ with $R$ a constant of the sampling grid. Here
    it is $H \cdot C_{\mathrm{keep}}$, $C_{\mathrm{keep}}$ is whatever the warm-up budget left
    standing, and the collection record carries it already resolved. A geometry carrying the raw
    cells' key and not this one must therefore skip rather than silently score against $R$.
    """
    assert coupling_analysis.coefficients_per_anchor(GEOMETRY) == BLOCK_WIDTH
    assert coupling_analysis.coefficients_per_anchor(
        {"geometry": {"horizon": 15, "raw_per_step": 16}}
    ) is None


def test_the_likelihood_percentage_records_that_it_is_budget_local() -> None:
    """$C_{\\mathrm{keep}}$ is a consequence of ``causal_warmup_budget_steps``, so the same
    checkpoint re-evaluated under a looser budget reports a different percentage from the same
    nats. The record says so whether or not the percentage was computed -- a reader who finds it
    present needs the caveat more than one who finds it absent."""
    computed = coupling_analysis.likelihood_percent_support(GEOMETRY)
    skipped = coupling_analysis.likelihood_percent_support({})

    for support in (computed, skipped):
        assert "budget" in support["budget_local"]
        assert "H*C_keep" in support["budget_local"]
    assert computed["skipped"] is False
    assert computed["coefficients_per_anchor"] == BLOCK_WIDTH


def test_the_run_records_why_the_likelihood_percentage_was_skipped(tmp_path) -> None:
    """A skip nobody can read afterwards is half a skip. The reason reaches ``results`` in the
    package's usual ``skipped``/``reason`` shape."""
    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b"],
            "mc_pred_gap": [1.0, 1.2, 0.8, 0.9],
            "sq_error_base": [4.0] * 4,
            "sq_error_full": [1.0] * 4,
        }
    )

    result = coupling_analysis.run_coupling_analysis(
        _context(
            per_sample,
            record={"geometry": {"horizon": 15, "block_width": 1470}, "likelihood": "mse"},
        ),
        eval_config=EVAL_CONFIG,
        output_dir=tmp_path,
        probe=None,
    )

    percent = result["pred_gap_percent"]
    assert percent["likelihood_space"]["skipped"] is True
    assert percent["likelihood_space"]["likelihood"] == "mse"
    assert percent["likelihood_space"]["reason"]
    # Absent from the headline, the rows and the table -- not present and NaN.
    assert coupling_analysis.LIKELIHOOD_SPACE_PERCENT not in percent["headline"]
    assert coupling_analysis.LIKELIHOOD_SPACE_PERCENT not in {
        row["metric"] for row in percent["rows"]
    }
    written = pd.read_csv(
        tmp_path / coupling_analysis.ANALYSIS_DIRNAME / coupling_analysis.PER_RECORDING_FILENAME
    )
    assert coupling_analysis.LIKELIHOOD_SPACE_PERCENT not in written.columns
    # The error-space pair is still reported in full.
    assert set(coupling_analysis.ERROR_SPACE_PERCENTS) <= set(percent["headline"])


def test_absent_geometry_skips_the_likelihood_percentage_rather_than_guessing_a_block_width() -> None:
    r"""$H \cdot C_{\mathrm{keep}}$ is the denominator that turns a block score into a
    per-coefficient one. A guessed constant there would rescale the answer with nothing raising,
    so an unavailable geometry produces no column, no row and no headline key."""
    per_guid = _per_guid(mc_pred_gap=[1.0, 2.0], sq_error_base=[4.0, 4.0], sq_error_full=[1.0, 1.0])

    assert coupling_analysis.coefficients_per_anchor({}) is None
    assert coupling_analysis.coefficients_per_anchor(GEOMETRY) == BLOCK_WIDTH
    assert coupling_analysis.likelihood_percent_support({})["coefficients_per_anchor"] is None

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=None)
    rows = coupling_analysis.build_percent_rows(
        per_guid.assign(**columns), resamples=200, seed=0
    )

    assert "pred_gap_mc_likelihood_pct" not in columns
    assert "pred_gap_mc_likelihood_pct" not in {row["metric"] for row in rows}
    assert "pred_gap_mc_likelihood_pct" not in coupling_analysis.percent_headline(rows)
    # The two that do not need the geometry are unaffected.
    assert {row["metric"] for row in rows} == {"pred_gap_rmse_pct", "pred_gap_mse_pct"}


def test_the_headline_block_omits_a_percentage_it_could_not_measure() -> None:
    """Omitted, not carried as ``NaN``. The headline's finiteness check fails on a non-finite
    number and passes on an absent one, and the two say different things."""
    per_guid = _per_guid(sq_error_base=[np.nan, np.nan], sq_error_full=[1.0, 1.0])

    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)
    rows = coupling_analysis.build_percent_rows(
        per_guid.assign(**columns), resamples=200, seed=0
    )
    headline = coupling_analysis.percent_headline(rows)

    assert "pred_gap_rmse_pct" not in headline and "pred_gap_mse_pct" not in headline
    assert all(np.isfinite(value) for value in headline.values())


def test_every_registered_percentage_can_be_produced() -> None:
    """The registry and the arithmetic are two lists that must agree: a name registered here and
    never computed would be a permanent ``None`` in the headline."""
    produced = set(coupling_analysis.percent_columns(
        _per_guid(sq_error_base=[4.0], sq_error_full=[1.0], mc_pred_gap=[1.0]),
        samples_per_anchor=BLOCK_WIDTH,
    ))

    assert produced == {name for name, _ in coupling_analysis.PERCENT_COLUMNS}


def test_the_percent_figure_separates_the_two_spaces() -> None:
    """A per-coefficient density ratio and an error reduction differ by an order of magnitude, so
    one shared axis would flatten whichever is smaller into a line at zero."""
    from teb_vae.lag_attn.eval import figures as shared_figures

    per_guid = _per_guid(
        sq_error_base=[4.0, 4.0, 4.0], sq_error_full=[1.0, 2.0, 3.0], mc_pred_gap=[1.0, 2.0, 3.0]
    )
    columns = coupling_analysis.percent_columns(per_guid, samples_per_anchor=BLOCK_WIDTH)
    per_guid = per_guid.assign(**columns)
    rows = coupling_analysis.build_percent_rows(per_guid, resamples=200, seed=0)

    figure = coupling_analysis.build_percent_figure(per_guid, rows)
    try:
        n_panels = len(figure.axes)
        vertical = [
            float(line.get_xdata()[0]) for line in figure.axes[0].lines
            if len(set(np.asarray(line.get_xdata(), dtype=np.float64))) == 1
        ]
    finally:
        shared_figures.plt.close(figure)

    assert n_panels == 3, "error space and likelihood space do not share an axis"
    assert 0.0 in vertical, "the no-improvement reference must be drawn"


# =============================================================================
# End to end, on a real run
# =============================================================================
def _context(
    per_sample: pd.DataFrame,
    results: Optional[Dict[str, Any]] = None,
    record: Optional[Dict[str, Any]] = None,
) -> Any:
    """An analysis context built by hand, with no model and no collection pass."""
    from teb_vae.lag_attn_cfs.eval.analyses import AnalysisContext

    collection = types.SimpleNamespace(
        per_sample=per_sample, per_anchor=pd.DataFrame(), record=record or {}, retained={},
        results=results or {},
    )
    return AnalysisContext(collection=collection, config={})


def test_the_analysis_writes_its_tables_and_figure(tmp_path) -> None:
    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b", "c", "c"],
            "mc_pred_gap": [1.0, 1.2, 0.8, 0.9, -0.1, 0.2],
            "pred_gap": [1.1, 1.3, 0.7, 1.0, -0.2, 0.3],
            "mc_nll_base_block": [11.0] * 6,
            "mc_nll_full_block": [10.0] * 6,
            "nll_base_block": [11.0] * 6,
            "nll_full_block": [10.0] * 6,
            "source_conditioned_kl_raw": [2.0] * 6,
            "source_conditioned_kl_shuffled_raw": [2.5] * 6,
        }
    )

    result = coupling_analysis.run_coupling_analysis(
        _context(per_sample), eval_config=EVAL_CONFIG, output_dir=tmp_path, probe=None
    )

    directory = tmp_path / coupling_analysis.ANALYSIS_DIRNAME
    for name in (
        coupling_analysis.PER_RECORDING_FILENAME,
        coupling_analysis.SUMMARY_FILENAME,
        figure_filename(coupling_analysis.DISTRIBUTION_FIGURE),
    ):
        assert (directory / name).is_file(), name
    # One row per recording, not per segment.
    assert len(pd.read_csv(directory / coupling_analysis.PER_RECORDING_FILENAME)) == 3
    assert result["composition"]["n_recordings"] == 3
    assert result["n_samples"] == 6


def test_the_analysis_writes_the_percentages_it_can_compute(tmp_path) -> None:
    """The whole path, on a frame carrying the squared errors and a record carrying the geometry:
    the percentage columns reach the per-recording table -- which is what the by-cohort fan-out
    reduces -- and the summary carries a row for each."""
    per_sample = pd.DataFrame(
        {
            "guid": ["a", "a", "b", "b", "c", "c"],
            "mc_pred_gap": [1.0, 1.2, 0.8, 0.9, -0.1, 0.2],
            "pred_gap": [1.1, 1.3, 0.7, 1.0, -0.2, 0.3],
            "sq_error_base": [4.0] * 6,
            "sq_error_full": [1.0, 1.0, 2.0, 2.0, 9.0, 9.0],
        }
    )

    result = coupling_analysis.run_coupling_analysis(
        _context(per_sample, record=GEOMETRY),
        eval_config=EVAL_CONFIG,
        output_dir=tmp_path,
        probe=None,
    )

    directory = tmp_path / coupling_analysis.ANALYSIS_DIRNAME
    assert (directory / figure_filename(coupling_analysis.PERCENT_FIGURE)).is_file()

    per_recording = pd.read_csv(directory / coupling_analysis.PER_RECORDING_FILENAME)
    names = [name for name, _ in coupling_analysis.PERCENT_COLUMNS]
    assert set(names) <= set(per_recording.columns)
    # The cohort fan-out can only resolve a column the frame actually carries, and the percentage
    # it names is the one that has to reach the table. (The emitter skips a name the frame lacks,
    # which is how the KL metrics behave on a fixture that carries no KL.)
    assert "pred_gap_rmse_pct" in result["grouped_frames"][0]["value_columns"]
    assert "pred_gap_rmse_pct" in set(per_recording.columns)

    summary = pd.read_csv(directory / coupling_analysis.SUMMARY_FILENAME)
    assert set(names) <= set(summary["metric"])
    # The nats rows are untouched by the percentages sharing their table.
    assert {"pred_gap_mc_nats", "pred_gap_train_path_nats"} <= set(summary["metric"])

    percent = result["pred_gap_percent"]
    assert percent["likelihood_space"]["skipped"] is False
    assert percent["likelihood_space"]["coefficients_per_anchor"] == BLOCK_WIDTH
    assert set(percent["headline"]) == set(names)
    assert all(np.isfinite(value) for value in percent["headline"].values())
    # Known answer end to end: recording 'a' has its squared error quartered.
    assert per_recording.set_index("guid").loc["a", "pred_gap_mse_pct"] == pytest.approx(75.0)


@pytest.mark.slow
def test_the_real_run_reports_both_estimators_with_their_denominators(collected_run) -> None:
    block = collected_run["summary"]["results"]["coupling"]

    metrics = {row["metric"]: row for row in block["pred_gap"]}
    assert set(metrics) == {"pred_gap_mc_nats", "pred_gap_train_path_nats"}
    for row in metrics.values():
        assert row["n_recordings_scored"] == collected_run["summary"]["results"]["n_recordings"]
        assert 0.0 <= row["positive_fraction"] <= 1.0
        assert row["n_positive"] <= row["n_recordings_scored"]


@pytest.mark.slow
def test_the_real_run_divides_the_likelihood_percentage_by_this_cells_block_width(
    collected_run,
) -> None:
    r"""The fixture's model is at the tiny geometry rather than the shipped one, so the assertion
    is the *identity* $H \cdot C_{\mathrm{keep}}$ against the run's own recorded geometry rather
    than the literal $1470$ -- which is what would still hold if the budget changed, and what
    would fail if the denominator came from anywhere but the model that was scored."""
    summary = collected_run["summary"]
    geometry = summary["collection"]["geometry"]
    support = summary["results"]["coupling"]["pred_gap_percent"]["likelihood_space"]

    assert geometry["block_width"] == geometry["horizon"] * geometry["target_kept_width"]
    if not support["skipped"]:
        assert support["coefficients_per_anchor"] == float(geometry["block_width"])
    assert "budget" in support["budget_local"]
