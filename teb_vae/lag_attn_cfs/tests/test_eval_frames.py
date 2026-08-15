r"""The aggregation chain every analysis reads the per-sample table through.

The chain is *per anchor* $\to$ support-weighted mean within a segment $\to$ unweighted mean over a
recording's segments $\to$ across recordings, and this module owns the middle arrow. Four of its
properties are choices rather than consequences, and each is one edit away from being lost without
any shape or count changing:

**A missing measurement is skipped, never imputed.** A segment that scored no anchors measured
nothing, and the collection pass writes that as ``NaN`` rather than as the ``0.0`` an empty
numerator over a clamped denominator produces. Averaged in as zero it would drag every mean toward
zero in proportion to how much coverage the run lost -- which is exactly the direction that makes a
failing run look like a modest one.

**The denominator travels with the number.** Every statistic reports the count of recordings that
actually contributed, so a run whose coverage collapsed reports a falling $n$ rather than a moving
mean.

**A degenerate baseline yields ``NaN``, not $\infty$ and not $0$.** ``skill_against`` is a ratio,
and a baseline with zero error is a recording the baseline reproduced exactly. Infinity would be
reported as evidence; $0.0$ would read as "no improvement", which is a measurement.

**The chain reduces *unrooted* quantities.** An RMS is the square root of a mean, and by Jensen the
average of per-segment roots is biased **low** -- in the direction that flatters the model. So the
squares travel through this module and the root is taken once, at the end, by whichever analysis
reports it. Nothing here roots anything, and the test below is what says that difference is large
enough to matter rather than a rounding argument.

Everything on the chain is in the loader's $z$ units. There is no bpm conversion anywhere in this
package: the forecast target is 98 wavelet coefficients, and inverting the per-channel statistics
would put them on scales spanning orders of magnitude, which destroys every pooled statistic
computed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval import frames
from teb_vae.lag_attn_cfs.eval._reuse import labels


@pytest.fixture
def per_sample() -> pd.DataFrame:
    """Two recordings: one with two scored segments, one whose second segment scored nothing."""
    return pd.DataFrame(
        {
            "guid": ["A", "A", "B", "B"],
            "sample_index": [0, 1, 2, 3],
            labels.CLASS_COLUMN: ["healthy", "healthy", "acidosis", "acidosis"],
            labels.SUBGROUP_COLUMN: ["healthy_bg_cs"] * 2 + ["acidosis_cs"] * 2,
            "pred_gap": [1.0, 3.0, 10.0, np.nan],
            "n_anchors": [152, 152, 152, 0],
        }
    )


# =================================================================================================
# The skill formula
# =================================================================================================
@pytest.mark.parametrize("baseline", [0.0, -1.0, -1e-12])
def test_a_degenerate_baseline_yields_nan_rather_than_infinity_or_zero(baseline: float) -> None:
    """``NaN`` is the only honest answer: it propagates as unmeasured through the mean and through
    the bootstrap, where ``inf`` would poison both and ``0.0`` would be read as a finding."""
    skill = frames.skill_against(np.array([1.0]), np.array([baseline]))

    assert np.isnan(skill).all()


def test_the_two_answers_the_criteria_are_stated_in_are_exact() -> None:
    """A forecast equal to the truth scores exactly $1$ on every recording and a forecast equal to
    the baseline exactly $0$ on every recording, so the mean carries both answers unchanged."""
    baseline = np.array([2.0, 5.0, 0.25])

    assert frames.skill_against(np.zeros(3), baseline) == pytest.approx(np.ones(3))
    assert frames.skill_against(baseline, baseline) == pytest.approx(np.zeros(3))


def test_the_skill_is_per_recording_rather_than_a_ratio_of_two_pooled_means() -> None:
    """The two differ whenever the recordings are not identically scaled, and this is the form the
    acceptance criteria are stated in -- which is also the form a bootstrap over recordings has a
    per-recording quantity to resample."""
    model = np.array([1.0, 8.0])
    baseline = np.array([2.0, 10.0])

    per_recording = float(np.mean(frames.skill_against(model, baseline)))
    pooled = 1.0 - float(model.mean() / baseline.mean())

    assert per_recording == pytest.approx(0.35)
    assert pooled != pytest.approx(per_recording)


# =================================================================================================
# The per-recording reduction
# =================================================================================================
def test_a_recording_is_one_row_however_many_segments_it_contributed(per_sample) -> None:
    reduced = frames.per_recording_means(per_sample, ["pred_gap"])

    assert list(reduced.index) == ["A", "B"]
    assert reduced.loc["A", "pred_gap"] == pytest.approx(2.0)
    # Both segments counted, including the one that measured nothing: the count is the recording's
    # size rather than the metric's denominator, and the two answer different questions.
    assert list(reduced["n_segments"]) == [2, 2]


def test_a_segment_that_scored_no_anchors_is_skipped_rather_than_averaged_in_as_zero(
    per_sample,
) -> None:
    """Recording B measured $10.0$ on one segment and nothing on the other. Imputing $0.0$ would
    report $5.0$ -- half the measurement, from a segment that measured nothing."""
    reduced = frames.per_recording_means(per_sample, ["pred_gap"])

    assert reduced.loc["B", "pred_gap"] == pytest.approx(10.0)


def test_the_honest_denominator_travels_with_every_statistic(per_sample) -> None:
    """A run whose coverage collapsed must report a falling $n$ rather than a moving mean."""
    reduced = frames.per_recording_means(per_sample, ["pred_gap"])
    described = frames.describe(frames.finite_column(reduced, "pred_gap"), name="pred_gap")

    assert (described["n"], described["n_dropped"]) == (2, 0)
    assert frames.scored_sample_count(per_sample, "pred_gap") == 3
    assert frames.scored_sample_count(per_sample, "not_a_column") is None


def test_the_cohort_labels_come_along_so_a_grouped_variant_is_a_groupby(per_sample) -> None:
    """A by-class variant must be a ``groupby`` on a column this frame already carries, rather than
    a second reduction with its own chance of using a different unit."""
    reduced = frames.per_recording_means(per_sample, ["pred_gap"])

    assert list(reduced[labels.CLASS_COLUMN]) == ["healthy", "acidosis"]
    assert list(reduced[labels.SUBGROUP_COLUMN]) == ["healthy_bg_cs", "acidosis_cs"]


def test_a_recording_whose_segments_disagree_about_its_cohort_carries_none() -> None:
    """It means the recording appears in two shards, which is a fault the loader probe raises on.
    Choosing the first or the commonest label would replace that fault with a plausible answer."""
    frame = pd.DataFrame(
        {
            "guid": ["A", "A"],
            labels.CLASS_COLUMN: ["healthy", "acidosis"],
            "pred_gap": [1.0, 2.0],
        }
    )

    resolved = frames.per_recording_labels(frame)

    assert resolved.loc["A", labels.CLASS_COLUMN] is None


def test_an_empty_frame_still_carries_the_columns_a_caller_will_index(per_sample) -> None:
    """An empty frame a caller can still index is friendlier than one with no schema, and an
    analysis reached with an empty table must skip rather than raise."""
    empty = frames.per_recording_means(per_sample.iloc[:0], ["pred_gap"])

    assert "pred_gap" in empty.columns and "n_segments" in empty.columns
    assert len(empty) == 0


def test_a_column_the_pass_did_not_produce_reads_as_unmeasured_rather_than_raising(
    per_sample,
) -> None:
    """A readout absent from an older run's tables must report as unmeasured, so re-running one
    analysis against a finished directory does not take down the analysis that wanted it."""
    missing = frames.finite_column(per_sample, "spectral_skill")

    assert missing.shape == (4,)
    assert np.isnan(missing).all()
    assert frames.describe(missing)["n"] == 0
    assert np.isnan(frames.describe(missing)["mean"])


# =================================================================================================
# The unrooted reduction, and the Jensen gap it keeps measurable
# =================================================================================================
def test_rooting_after_the_reduction_and_rooting_before_it_are_different_numbers() -> None:
    r"""Why this module never roots anything.

    Two segments of one recording with mean squares $1$ and $9$. Rooted **once**, after the
    reduction, the RMS is $\sqrt{5} \approx 2.236$; averaging the per-segment roots gives $2$, a
    $10.6\%$ under-report. Both numbers are legitimate quantities and only one of them is an RMS,
    so the chain carries the squares and the analysis that reports an RMS roots at the end of it.
    """
    frame = pd.DataFrame({"guid": ["A", "A"], "sq_error_full": [1.0, 9.0]})

    reduced = frames.per_recording_means(frame, ["sq_error_full"])
    rooted_once = float(np.sqrt(reduced.loc["A", "sq_error_full"]))
    average_of_roots = float(np.sqrt(frame["sq_error_full"]).mean())

    assert reduced.loc["A", "sq_error_full"] == pytest.approx(5.0)
    assert rooted_once == pytest.approx(np.sqrt(5.0))
    assert average_of_roots == pytest.approx(2.0)
    # Jensen's direction, which is the one that matters: the biased estimator flatters the model.
    assert average_of_roots < rooted_once


# =================================================================================================
# The two summary readouts
# =================================================================================================
def test_the_positive_fraction_counts_its_own_denominator() -> None:
    """``np.nan > 0`` is ``False``, so a recording that measured nothing would otherwise be counted
    silently as evidence *against* -- and a run whose coverage collapsed would report a falling
    positive fraction rather than a falling $n$."""
    record = frames.positive_fraction([1.0, -1.0, np.nan, 2.0])

    assert record["fraction"] == pytest.approx(2.0 / 3.0)
    assert (record["n_positive"], record["n"], record["n_dropped_not_finite"]) == (2, 3, 1)


def test_nothing_finite_summarises_as_nan_rather_than_as_zero() -> None:
    """``0.0`` reads as a measurement; every statistic of an empty set has to read as absent."""
    described = frames.describe([np.nan, np.inf, -np.inf])

    assert described["n"] == 0 and described["n_dropped"] == 3
    assert all(np.isnan(described[key]) for key in ("mean", "min", "max", "q25", "q50", "q75"))
    assert np.isnan(frames.positive_fraction([])["fraction"])


def test_the_quartiles_travel_beside_the_mean_because_these_readouts_are_skewed() -> None:
    described = frames.describe(np.arange(101, dtype=np.float64), name="pred_gap")

    assert described["metric"] == "pred_gap"
    assert (described["q25"], described["q50"], described["q75"]) == (25.0, 50.0, 75.0)
    assert (described["min"], described["max"]) == (0.0, 100.0)


# =================================================================================================
# The grouped-frame declaration
# =================================================================================================
def test_a_declared_frame_is_named_by_a_path_relative_to_the_results_directory() -> None:
    """An absolute path in the summary is a machine-specific string in a block two runs of one
    checkpoint must compare equal, and it stops resolving the moment the directory is copied."""
    entry = frames.grouped_frame_entry("warmup", "per_recording.csv", ["pred_gap_warm_hi"])

    assert entry == {
        "directory": "warmup",
        "path": "warmup/per_recording.csv",
        "stem": "per_recording",
        "value_columns": ["pred_gap_warm_hi"],
    }
    assert not entry["path"].startswith("/")


# =================================================================================================
# The recomposition guard the channel-axis splits share
# =================================================================================================
def _split(residual: float = 0.0, block_score: float = 1.0) -> pd.DataFrame:
    """A per-recording frame whose three parts miss their total by ``residual``."""
    return pd.DataFrame(
        {
            "part_a": [0.1] * 3,
            "part_b": [0.2] * 3,
            "part_c": [0.3] * 3,
            "pred_gap": [0.6 - residual] * 3,
            frames.RECOMPOSITION_SCALE_COLUMN: [block_score] * 3,
        }
    )


def test_a_split_that_recomposes_is_reported_as_holding() -> None:
    record = frames.recomposition_check(
        _split(), ["part_a", "part_b", "part_c"], "pred_gap", identity="a + b + c == total"
    )

    assert record["holds"] is True
    assert record["n_recordings"] == 3
    assert record["max_abs_residual"] == pytest.approx(0.0, abs=1e-12)
    assert record["identity"] == "a + b + c == total"


def test_the_tolerance_is_relative_to_the_block_score_and_not_to_the_quantity_split() -> None:
    r"""``pred_gap`` is a *difference* of two block scores of order $10^{3}$, so the float32 error
    it inherits belongs to those rather than to the small number between them. A tolerance relative
    to the difference would tighten without limit as a model improved, refusing a healthy
    decomposition on exactly the runs that matter."""
    parts = ["part_a", "part_b", "part_c"]

    accepted = frames.recomposition_check(
        _split(residual=2.5e-4, block_score=1360.0), parts, "pred_gap", identity="x"
    )
    refused = frames.recomposition_check(
        _split(residual=2.5e-4, block_score=1.0), parts, "pred_gap", identity="x"
    )

    assert accepted["holds"] is True
    assert refused["holds"] is False
    assert accepted["max_abs_residual"] == pytest.approx(refused["max_abs_residual"])
    assert accepted["scale_column"] == frames.RECOMPOSITION_SCALE_COLUMN


def test_a_frame_carrying_no_block_score_falls_back_to_the_quantity_split() -> None:
    """An older run's tables may not carry the column, and the total is the only other magnitude
    available -- reported as a tighter check rather than as no check."""
    frame = _split(residual=2.5e-4).drop(columns=[frames.RECOMPOSITION_SCALE_COLUMN])

    record = frames.recomposition_check(
        frame, ["part_a", "part_b", "part_c"], "pred_gap", identity="x"
    )

    assert record["holds"] is False


def test_a_split_nothing_could_be_checked_on_is_unchecked_rather_than_holding() -> None:
    """``None`` rather than ``True``: an identity nothing was available to check is not one that
    held, and a summary cannot tell the two apart unless the record says which."""
    frame = _split()
    frame["pred_gap"] = [np.nan] * 3

    record = frames.recomposition_check(
        frame, ["part_a", "part_b", "part_c"], "pred_gap", identity="x"
    )

    assert record["holds"] is None
    assert record["n_recordings"] == 0


def test_the_worst_recording_decides_rather_than_the_mean_of_the_residuals() -> None:
    """The mechanism that would break a channel-axis split moves value *between* the parts, so the
    per-recording error is zero-mean by construction and a mean would report nothing."""
    frame = _split()
    frame.loc[0, "part_a"] = 0.1 + 0.5
    frame.loc[1, "part_a"] = 0.1 - 0.5

    record = frames.recomposition_check(
        frame, ["part_a", "part_b", "part_c"], "pred_gap", identity="x"
    )

    assert record["holds"] is False
    assert record["max_abs_residual"] == pytest.approx(0.5)
