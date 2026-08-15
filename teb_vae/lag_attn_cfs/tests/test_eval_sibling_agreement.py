r"""The fork's first anti-drift measure: the shared arithmetic still agrees with the sibling's.

This package is a **copy** of ``teb_vae/lag_attn_rws/eval``, edited for a different target domain.
A copy is how two things that must stay comparable stop being comparable: the first fix to an
aggregation lands on one side, the two ``summary.json`` files quietly stop meaning the same thing,
and the encoder-by-target grid they exist to fill becomes unreadable long before anyone notices.

So the quantities that are supposed to be *identical* are re-derived here through both packages'
implementations on identical stub inputs, and asserted equal.

**What a failure here means.** Not that this package is wrong. It means one of the two moved and
the other did not, and the next question is which -- and whether the move was deliberate. If it
was, the module's entry in ``divergences.json`` becomes ``divergent`` with a reason, and the
assertion below is deleted in the same commit. What must never happen is that the assertion is
deleted and the entry is not: ``tests/test_eval_divergences.py`` reads the entries' named
assertions out of *this file*, so an ``equivalent`` claim with nothing behind it fails there.

**This is the only file in this package that imports the sibling evaluation package**, apart from
``test_eval_reuse.py``, which pins the shared primitives by identity. The layering test forbids the
import everywhere else, in every form, and asserts that the exemption reaches exactly these two
files -- a package that copied the analyses and then reached back for one helper would have two
implementations *and* a dependency.

The name matters too: ``test_eval_parity.py`` is taken in the sibling suite and means something
else entirely -- the evaluation's readouts recombining to ``compute_loss``'s own numbers, which is
ported separately. Two files named ``parity`` asserting different things is exactly the confusion
this suite exists to avoid.
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.eval import analyses, cohort, frames, launch, report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.analyses import latent as latent_analysis
from teb_vae.lag_attn_cfs.eval.analyses import perm_control as perm_control_analysis
from teb_vae.lag_attn_cfs.eval.analyses import sufficiency as sufficiency_analysis
from teb_vae.lag_attn_cfs.eval.analyses import time_to_delivery as ttd_analysis
from teb_vae.lag_attn_rws.eval import analyses as sibling_analyses
from teb_vae.lag_attn_rws.eval import cohort as sibling_cohort
from teb_vae.lag_attn_rws.eval import frames as sibling_frames
from teb_vae.lag_attn_rws.eval import launch as sibling_launch
from teb_vae.lag_attn_rws.eval import report_seam as sibling_report_seam
from teb_vae.lag_attn_rws.eval import run as sibling_run
from teb_vae.lag_attn_rws.eval.analyses import latent as sibling_latent_analysis
from teb_vae.lag_attn_rws.eval.analyses import (  # noqa: F401
    sufficiency as sibling_sufficiency_analysis,
)
from teb_vae.lag_attn_rws.eval.analyses import (
    perm_control as sibling_perm_control_analysis,
)
from teb_vae.lag_attn_rws.eval.analyses import time_to_delivery as sibling_ttd_analysis


@pytest.fixture
def per_sample() -> pd.DataFrame:
    """One frame both packages reduce, carrying every case a plausible edit would move.

    A recording with two segments and one with a single segment (so the aggregation is not an
    identity on both), a segment that measured nothing, a cohort neither canonical order knows,
    and an ``epoch`` spanning three trajectory bins.
    """
    return pd.DataFrame(
        {
            "guid": ["A", "A", "B", "C"],
            "clinical_class": ["healthy", "healthy", "hie", "not_a_class"],
            "subgroup": ["healthy_bg_cs", "healthy_bg_cs", "hie_cs", "not_a_subgroup"],
            "epoch": [-3600.0, -5400.0, -900.0, -12600.0],
            "pred_gap": [1.0, 3.0, np.nan, 7.0],
            "sq_error_full": [1.0, 9.0, 4.0, 16.0],
        }
    )


# =================================================================================================
# frames.py
# =================================================================================================
def test_the_skill_formula_is_one_formula_in_both_packages() -> None:
    r"""$1 - \mathrm{MSE}_{\rm model}/\mathrm{MSE}_{\rm ref}$, per recording, with the degenerate
    cases included: a zero denominator, a negative one and a missing value all have to fail to the
    same thing, and ``NaN`` is not a value ``==`` compares equal, so they are compared as masks."""
    model = np.array([1.0, 8.0, 0.0, 2.0, np.nan])
    baseline = np.array([2.0, 10.0, 0.0, -1.0, 4.0])

    mine = frames.skill_against(model, baseline)
    theirs = sibling_frames.skill_against(model, baseline)

    assert np.array_equal(np.isnan(mine), np.isnan(theirs))
    assert mine[~np.isnan(mine)].tolist() == theirs[~np.isnan(theirs)].tolist()
    # Not vacuous: the finite half carries real values rather than being empty.
    assert np.isfinite(mine).sum() == 2


def test_the_per_recording_aggregation_chain_agrees(per_sample) -> None:
    """The middle arrow of the chain -- per segment to per recording -- including the two cases an
    edit would move first: the single-segment recording (where the reduction is an identity) and
    the recording whose only segment measured nothing."""
    mine = frames.per_recording_means(per_sample, ["pred_gap"])
    theirs = sibling_frames.per_recording_means(per_sample, ["pred_gap"])

    pd.testing.assert_frame_equal(mine, theirs)
    assert list(mine.index) == ["A", "B", "C"]
    assert np.isnan(mine.loc["B", "pred_gap"])


def test_the_unrooted_reduction_that_keeps_the_jensen_pair_measurable_agrees(per_sample) -> None:
    """Both packages reduce *squares* and root once at the end, so the average-of-roots bias stays
    measurable rather than being baked in. Asserted on both halves of the pair, because a fork that
    rooted inside the reduction would still agree on the second half alone."""
    mine = frames.per_recording_means(per_sample, ["sq_error_full"])["sq_error_full"]
    theirs = sibling_frames.per_recording_means(per_sample, ["sq_error_full"])["sq_error_full"]

    assert mine.tolist() == theirs.tolist()
    rooted_once = float(np.sqrt(mine.loc["A"]))
    average_of_roots = float(np.sqrt(per_sample.loc[:1, "sq_error_full"]).mean())
    assert rooted_once == pytest.approx(np.sqrt(5.0))
    assert average_of_roots < rooted_once


def test_the_summary_statistics_and_the_positive_fraction_agree() -> None:
    """The two readouts every analysis reports a per-recording vector through, on a vector carrying
    a negative, a zero and a non-finite value -- which is where the two could differ at all."""
    values = [1.0, -1.0, 0.0, np.nan, 2.0]

    assert frames.describe(values, name="pred_gap") == sibling_frames.describe(
        values, name="pred_gap"
    )
    assert frames.positive_fraction(values) == sibling_frames.positive_fraction(values)
    assert frames.describe(values)["n"] == 4


# =================================================================================================
# cohort.py
# =================================================================================================
def test_the_cohort_order_is_one_ordering_in_both_packages() -> None:
    """A fork of the cohort order is a fork of which recordings are compared against which. Both
    axes, plus the two cases an edit moves first: a partial cohort and an unknown label."""
    classes = ["hie", "not_a_class", "healthy", "acidosis"]
    subgroups = sorted(("hie_cs", "healthy_bg_cs", "acidosis_no_cs", "not_a_subgroup"))

    assert cohort.ordered_groups(classes, "clinical_class") == sibling_cohort.ordered_groups(
        classes, "clinical_class"
    ) == ["healthy", "acidosis", "hie", "not_a_class"]
    assert cohort.ordered_groups(subgroups, "subgroup") == sibling_cohort.ordered_groups(
        subgroups, "subgroup"
    )
    # An empty cohort is the other end of the same function and must not raise in either.
    assert cohort.ordered_groups([], "clinical_class") == sibling_cohort.ordered_groups(
        [], "clinical_class"
    ) == []


def test_the_time_axis_bins_identically_in_both_packages(per_sample) -> None:
    """The trajectory axis is a constant in both, not a setting. A bin width that drifted would
    make two runs' trajectory figures describe different windows under the same labels."""
    assert cohort.TRAJECTORY_BIN_HOURS == sibling_cohort.TRAJECTORY_BIN_HOURS

    mine = cohort.add_time_bins(per_sample)
    theirs = sibling_cohort.add_time_bins(per_sample)

    pd.testing.assert_frame_equal(mine, theirs)
    assert mine[cohort.BIN_COLUMN].nunique() == 4


def test_the_population_record_is_assembled_the_same_way(per_sample) -> None:
    """The counts, the disjointness computation and the two sentences: a package that quietly
    disagreed about who was evaluated would disagree about every cohort statement above it."""
    config = {
        "dataset_config": {
            "vae_train_datasets": ["/data/train/healthy_bg_cs.hdf5"],
            "vae_test_datasets": ["/data/test/hie_cs.hdf5"],
        }
    }

    assert cohort.build_cohort_block(per_sample, config) == sibling_cohort.build_cohort_block(
        per_sample, config
    )


# =================================================================================================
# report_seam.py -- the mechanism, not the content
# =================================================================================================
def test_the_headline_path_resolver_is_one_walker() -> None:
    """The registries differ deliberately -- this cell has two more verdicts and no frequency
    entries -- but the *walk* that turns a path into a value must not: a path that resolves in one
    package has to resolve in the other, or two summaries disagree about what "absent" means."""
    results = {"readouts": {"mc_pred_gap": 1.5}, "coupling": {"headline": {}}}

    assert report_seam._dig is sibling_report_seam._dig
    for path in (("readouts", "mc_pred_gap"), ("coupling", "headline", "absent"), ("nothing",)):
        assert report_seam._dig(results, path) == sibling_report_seam._dig(results, path)


def test_the_identity_tolerances_are_the_same_numbers() -> None:
    """Both packages judge the same structural identities, so a tolerance that moved on one side
    would make one of them report a failure the other calls rounding."""
    assert report_seam.IDENTITY_RTOL == sibling_report_seam.IDENTITY_RTOL
    assert report_seam.IDENTITY_ATOL == sibling_report_seam.IDENTITY_ATOL
    assert report_seam.IDENTITY_TOLERANCE == sibling_report_seam.IDENTITY_TOLERANCE
    assert report_seam.identity_tolerance_for(1e5) == sibling_report_seam.identity_tolerance_for(
        1e5
    )
    assert report_seam.identity_tolerance_for(None) == sibling_report_seam.identity_tolerance_for(
        None
    )


def test_the_cross_table_recombination_check_agrees_where_both_carry_the_column() -> None:
    """The check itself is shared arithmetic; only which columns it looks for differs. Run on a
    frame carrying a column both registries hold, the verdict, the residual and the tolerance must
    be the same numbers."""
    per_sample = pd.DataFrame({"sample_index": [0, 1], "nll_full_block": [10.0, 20.0]})
    per_anchor = pd.DataFrame(
        {"sample_index": [0, 0, 1, 1], "nll_full_block": [9.0, 11.0, 10.0, 20.0]}
    )

    mine = report_seam.check_per_anchor_recombines(per_sample, per_anchor)
    theirs = sibling_report_seam.check_per_anchor_recombines(per_sample, per_anchor)

    assert mine == theirs
    assert mine["verdict"] == "fail"


def test_the_analysis_registry_keeps_the_shared_order_of_the_analyses_both_carry() -> None:
    """The registries differ deliberately -- this cell has two analyses the raw cells cannot have,
    drops ``coherence`` outright, and lands the rest a few at a time. What must **not** differ is
    the relative order of the analyses both packages carry: the two ``summary.json`` files are
    read side by side in the arm comparison, and a registry that reordered what it kept would make
    that comparison an exercise in re-sorting.

    Asserted as a subsequence rather than as a list, which is the form that stays true both while
    the remaining analyses land and after they have.
    """
    shared_order = [name for name in sibling_run.ANALYSES if name in run_module.ANALYSIS_FUNCTIONS]
    mine = [name for name in run_module.ANALYSES if name in sibling_run.ANALYSIS_FUNCTIONS]

    assert mine == shared_order
    # Not vacuous: the filter above must actually have kept something, and dropped something.
    assert len(shared_order) >= 6
    assert "coherence" not in run_module.ANALYSIS_FUNCTIONS


def test_the_verdict_ordering_is_the_shared_one_extended_rather_than_reordered() -> None:
    """This cell promotes two verdicts the raw cells cannot have. They are **appended**: a reorder
    would change the order every console table and every arm table reads them in, for both."""
    shared = sibling_report_seam.HEADLINE_VERDICTS

    assert report_seam.HEADLINE_VERDICTS[: len(shared)] == shared
    assert len(report_seam.HEADLINE_VERDICTS) == len(shared) + 2


# =================================================================================================
# launch.py and the analysis protocol
# =================================================================================================
def test_the_launch_merge_resolves_identically_in_both_packages() -> None:
    """The Run-button convention is a repository rule rather than a package's, so the two copies
    must resolve a command line, a launch dict and an absent value to the same three sources."""
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="agreement")
        parser.add_argument("--alpha", default=None)
        parser.add_argument("--beta", default=None)
        parser.add_argument("--gamma", default=None)
        return parser

    run_args = {"alpha": "from_dict", "beta": "from_dict"}
    argv = ["--alpha", "from_cli"]

    assert launch.resolve_launch_args(_parser(), run_args, argv) == (
        sibling_launch.resolve_launch_args(_parser(), run_args, argv)
    )
    assert launch.missing_required({"config": None}, ("config",)) == (
        sibling_launch.missing_required({"config": None}, ("config",))
    )
    assert launch.CLI_SOURCE == sibling_launch.CLI_SOURCE


def test_the_analysis_protocol_is_the_same_contract_in_both_packages() -> None:
    """An analysis is written against this protocol, and the runner reads its result through it.
    Two packages with two contracts is two registries whose entries are not interchangeable -- and
    the fourth argument is the one a reader reconstructs wrongly: ``probe`` is the *population*
    record the collection pass produced, not a model probe."""
    assert analyses.REQUIRED_RESULT_KEYS == sibling_analyses.REQUIRED_RESULT_KEYS
    assert analyses.GROUPED_FRAMES_KEY == sibling_analyses.GROUPED_FRAMES_KEY

    mine = [field.name for field in dataclasses.fields(analyses.AnalysisContext)]
    theirs = [field.name for field in dataclasses.fields(sibling_analyses.AnalysisContext)]
    assert mine == theirs


# =================================================================================================
# analyses/perm_control.py -- the specificity control, whose arithmetic is domain-neutral
# =================================================================================================
def _perm_control_frame() -> pd.DataFrame:
    """Per-recording block scores across the four branches, spread widely enough to discriminate.

    The four columns are deliberately *not* in the ordering a healthy model produces on every row:
    two recordings are given a shuffled score below base, so an implementation that classified the
    pooled means rather than the per-recording ones would still be compared on inputs where the two
    disagree.
    """
    return pd.DataFrame(
        {
            "mc_nll_base_block": [10.0, 11.0, 12.0, 13.0, 14.0],
            "mc_nll_full_block": [9.0, 10.5, 11.0, 12.5, 13.0],
            "mc_nll_shuffled_block": [12.0, 10.0, 13.0, 12.0, 16.0],
            "mc_nll_base_shuffled_mu_block": [10.5, 11.5, 12.5, 13.5, 14.5],
            "source_conditioned_kl_raw": [1.0, 1.2, 0.8, 1.1, 0.9],
            "source_conditioned_kl_shuffled_raw": [1.4, 1.5, 1.0, 1.3, 1.2],
        }
    )


@pytest.mark.parametrize(
    "scores",
    [
        (10.0, 9.0, 12.0),      # specific: the full ordering holds
        (10.0, 9.0, 9.5),       # influential but not specific: a stranger's source helps too
        (10.0, 10.5, 12.0),     # no improvement: the source cost something
        (10.0, float("nan"), 12.0),   # inconclusive: a branch is missing
    ],
)
def test_the_specificity_outcome_is_named_the_same_way_in_both_packages(scores) -> None:
    """Four losses arranged into four different findings, and both packages must reach the same
    name for each. The classification is where a plausible edit goes wrong silently: swapping the
    two comparisons turns ``influential_not_specific`` -- a real finding -- into ``specific``."""
    base, full, shuffled = scores

    mine = perm_control_analysis.classify_outcome(base, full, shuffled)
    theirs = sibling_perm_control_analysis.classify_outcome(base, full, shuffled)

    assert mine == theirs
    assert mine in perm_control_analysis.OUTCOMES
    assert perm_control_analysis.OUTCOMES == sibling_perm_control_analysis.OUTCOMES


def test_the_three_paired_controls_keep_the_same_sign_convention_in_both_packages() -> None:
    """All three are ``control - reference``, so positive means the control cost something. A sign
    that flipped in one package would make ``source_margin`` -- which the headline promotes -- read
    backwards, and the number itself would stay perfectly plausible."""
    frame = _perm_control_frame()

    mine = perm_control_analysis.build_penalty_rows(frame, resamples=200, seed=0)
    theirs = sibling_perm_control_analysis.build_penalty_rows(frame, resamples=200, seed=0)

    assert [row["penalty"] for row in mine] == [row["penalty"] for row in theirs]
    assert mine == theirs
    # Non-vacuity: the fixture puts a stranger's source both above and below base, so the positive
    # fraction is strictly between 0 and 1 and two implementations cannot agree by both saturating.
    shuffle = {row["penalty"]: row for row in mine}["shuffle_penalty"]
    assert 0.0 < shuffle["positive_fraction"] < 1.0


def test_the_branch_summary_and_the_kl_description_agree_in_both_packages() -> None:
    """The four branch rows and the KL-space reading, compared whole. ``shuffled_exceeds_true`` is
    a *description* consumed by nothing, and the two packages must describe it identically or one
    run's summary says a healthy model is broken."""
    frame = _perm_control_frame()

    assert perm_control_analysis.build_branch_rows(
        frame, resamples=200, seed=0
    ) == sibling_perm_control_analysis.build_branch_rows(frame, resamples=200, seed=0)
    assert perm_control_analysis.build_kl_description(
        frame
    ) == sibling_perm_control_analysis.build_kl_description(frame)
    assert (
        perm_control_analysis.SOURCE_MARGIN_SCALAR
        == sibling_perm_control_analysis.SOURCE_MARGIN_SCALAR
    )


# =================================================================================================
# analyses/latent.py -- the one analysis whose subject is domain-neutral
# =================================================================================================
def test_the_latent_spectrum_is_laid_out_the_same_way_in_both_packages() -> None:
    """A latent is $d_z$ numbers per step whether the decoder emits raw samples or wavelet
    coefficients, so this analysis is a copy with nothing edited -- and this is what says so.

    Asserted on the *frame*, not on a scalar: the sort order, the share denominator and the
    activity threshold are three separate decisions and a divergence in any one of them would
    change which dimensions a reader is told are active.
    """
    spectrum = [0.1, 2.0, 0.001, 0.5]

    pd.testing.assert_frame_equal(
        latent_analysis.spectrum_frame(spectrum),
        sibling_latent_analysis.spectrum_frame(spectrum),
    )
    # The degenerate ends of the same function: an empty latent, and one carrying no KL at all.
    assert latent_analysis.spectrum_frame([]).empty
    assert sibling_latent_analysis.spectrum_frame([]).empty
    pd.testing.assert_frame_equal(
        latent_analysis.spectrum_frame([0.0, 0.0]),
        sibling_latent_analysis.spectrum_frame([0.0, 0.0]),
    )


def test_the_latent_diagnostics_are_the_same_thirteen_reductions_in_both_packages() -> None:
    """The column set, the order, and the summary row each reduces to. The order matters as much
    as the set: it is the order of the emitted table, and the two cells' tables are read side by
    side in the arm comparison.

    Every field of every row is compared **except** ``meaning``, which is the one-line English
    gloss written into the emitted CSV. That field is excluded rather than ignored: the next test
    asserts exactly which glosses differ and why, so an unreviewed edit to one still fails.
    """
    assert [name for name, _ in latent_analysis.DIAGNOSTIC_COLUMNS] == [
        name for name, _ in sibling_latent_analysis.DIAGNOSTIC_COLUMNS
    ]
    assert latent_analysis.GROUPED_METRICS == sibling_latent_analysis.GROUPED_METRICS

    # A per-recording frame carrying every reduced column, with the case a divergence in the
    # reduction would move first: a recording that measured nothing, which must be dropped and
    # counted rather than averaged in. Enough finite recordings beside it that the bootstrap
    # returns real bounds -- an interval of NaN on both sides would compare unequal here and would
    # make the assertion vacuous in the other direction.
    per_guid = pd.DataFrame(
        {
            name: [0.1, 0.4, np.nan, 0.9, 0.6]
            for name, _gloss in latent_analysis.DIAGNOSTIC_COLUMNS
        }
    )

    def _numbers(rows):
        return [
            {key: value for key, value in row.items() if key != "meaning"} for row in rows
        ]

    mine = latent_analysis.build_diagnostic_rows(per_guid, resamples=64, seed=0)
    theirs = sibling_latent_analysis.build_diagnostic_rows(per_guid, resamples=64, seed=0)

    assert _numbers(mine) == _numbers(theirs)
    assert all(np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"]) for row in mine)
    assert all(row["n"] == 4 and row["n_dropped"] == 1 for row in mine)


def test_the_only_gloss_that_differs_is_the_one_naming_this_target_domains_unit() -> None:
    """The exclusion above, made accountable. Exactly one of the thirteen column descriptions is
    reworded here, and it is the one that would otherwise write "raw samples" into
    ``latent_diagnostics.csv`` -- a unit this pipeline does not have, on a figure whose
    denominator is the scored target coefficients rather than raw samples of a heart rate."""
    mine = dict(latent_analysis.DIAGNOSTIC_COLUMNS)
    theirs = dict(sibling_latent_analysis.DIAGNOSTIC_COLUMNS)

    differing = sorted(name for name in mine if mine[name] != theirs[name])

    assert differing == ["mean_logvar_full"]
    assert "raw sample" in theirs["mean_logvar_full"]
    assert "coefficient" in mine["mean_logvar_full"]
    assert "raw sample" not in mine["mean_logvar_full"]


# =================================================================================================
# analyses/time_to_delivery.py -- the clinical question, asked identically in both cells
# =================================================================================================
@pytest.fixture
def time_axis_frame() -> pd.DataFrame:
    """Two classes over two windows, with enough recordings per class for the tests to run.

    The values are separated between the classes on purpose: an implementation that binned or
    reduced differently would still produce *a* trajectory on an undifferentiated frame, and the
    two packages would agree on nothing meaningful.
    """
    rows = []
    for offset, name in ((0.0, "healthy"), (50.0, "acidosis")):
        for recording in range(5):
            for segment, hours in enumerate((1.0, 3.0)):
                rows.append(
                    {
                        "guid": f"{name}_{recording:02d}",
                        "epoch": -hours * 3600.0,
                        labels.CLASS_COLUMN: name,
                        labels.SUBGROUP_COLUMN: f"{name}_shard",
                        "mc_pred_gap": offset + float(recording) + 0.1 * segment,
                        "source_conditioned_kl_raw": offset + float(recording),
                    }
                )
    return pd.DataFrame(rows)


def test_the_time_before_delivery_grid_and_its_readouts_are_the_same_in_both_packages(
    time_axis_frame,
) -> None:
    """The window width, the family-wise error rate and the two readouts tracked. A width that
    drifted would make two runs' trajectory tables describe different windows under one set of
    labels, which is unreadable rather than merely different."""
    assert ttd_analysis.TRAJECTORY_BIN_HOURS == sibling_ttd_analysis.TRAJECTORY_BIN_HOURS
    assert ttd_analysis.DEFAULT_ALPHA == sibling_ttd_analysis.DEFAULT_ALPHA
    assert ttd_analysis.READOUTS == sibling_ttd_analysis.READOUTS
    assert ttd_analysis.METHOD == sibling_ttd_analysis.METHOD

    mine = ttd_analysis.build_per_recording(time_axis_frame)
    theirs = sibling_ttd_analysis.build_per_recording(time_axis_frame)

    assert sorted(mine) == sorted(theirs)
    for axis in mine:
        pd.testing.assert_frame_equal(mine[axis], theirs[axis])
    # Not vacuous: the reduction actually put the ten recordings into two windows each.
    assert len(mine[labels.CLASS_COLUMN]) == 20


def test_the_three_layers_of_inference_reach_the_same_verdicts(time_axis_frame) -> None:
    """Kruskal per window, Holm across the windows, pairwise on the survivors. Compared as whole
    records so a divergence in any layer -- the omnibus, the correction, or which windows the
    pairwise tests were allowed to run on -- fails here rather than in one run's summary."""
    mine = ttd_analysis.build_per_recording(time_axis_frame)[labels.CLASS_COLUMN]
    theirs = sibling_ttd_analysis.build_per_recording(time_axis_frame)[labels.CLASS_COLUMN]

    my_record = ttd_analysis.analyse_windows(mine, "mc_pred_gap")
    their_record = sibling_ttd_analysis.analyse_windows(theirs, "mc_pred_gap")

    assert my_record == their_record
    # Non-vacuity: the classes are fifty nats apart, so both windows must survive Holm. Two
    # implementations that both refused to test anything would otherwise compare equal.
    assert my_record["n_windows_tested"] == 2
    assert my_record["n_significant_windows"] == 2


# =================================================================================================
# analyses/sufficiency.py -- the same three scores, the same two gaps, the same join key
# =================================================================================================
@pytest.fixture
def oracle_scored_frame() -> pd.DataFrame:
    """A per-sample table carrying the three block scores the sufficiency analysis compares.

    Four recordings of two segments each, with the three scores ordered oracle < full < base so
    both gaps are non-zero and signed: an implementation that subtracted the other way round, or
    that averaged over segments rather than over recordings, produces a different number on this
    frame rather than the same one twice. Four rather than two because ``bootstrap_ci`` refuses
    below three recordings and returns ``NaN`` with a note -- on which two implementations that
    both refused would compare equal and prove nothing.
    """
    rows = []
    for recording in range(4):
        for segment in range(2):
            base = 100.0 + 10.0 * recording + float(segment)
            rows.append(
                {
                    "guid": f"REC{recording:02d}",
                    "epoch": -3600.0 * (segment + 1),
                    labels.CLASS_COLUMN: "healthy",
                    labels.SUBGROUP_COLUMN: "healthy_shard",
                    "mc_nll_base_block": base,
                    "mc_nll_full_block": base - 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_the_sufficiency_scores_and_both_gaps_are_defined_identically_in_both_packages() -> None:
    """The three columns compared, the two differences taken, and which of them are resolved by
    cohort. A package that renamed a column or reversed a subtraction would put a differently
    signed ``delta_suff_nats`` under one name in two summaries the arm table reads side by side."""
    assert sufficiency_analysis.SCORE_COLUMNS == sibling_sufficiency_analysis.SCORE_COLUMNS
    assert sufficiency_analysis.GAP_METRICS == sibling_sufficiency_analysis.GAP_METRICS
    assert sufficiency_analysis.GROUPED_METRICS == sibling_sufficiency_analysis.GROUPED_METRICS
    # Non-vacuity: both gaps are ``left - right`` over columns that are themselves in the score
    # list, which is what makes them differences over one denominator rather than three
    # populations.
    scored = {column for column, _label in sufficiency_analysis.SCORE_COLUMNS}
    assert all(
        left in scored and right in scored
        for _name, left, right, _meaning in sufficiency_analysis.GAP_METRICS
    )


def test_the_oracle_score_join_and_its_summary_rows_agree(oracle_scored_frame) -> None:
    """The join key and the per-recording reduction behind $\\Delta_{\\mathrm{suff}}$, run through
    both implementations on one frame. The key is ``(guid, rounded epoch)`` because one side of it
    has been through a CSV round trip on an offline re-run, and a join that silently matched
    nothing would report an empty analysis rather than a broken key."""
    per_segment = {
        "guid": [f"REC{index // 2:02d}" for index in range(8)],
        # Deliberately off by a fraction of a second, so the rounding in the key is what makes the
        # join succeed: a float comparison would match nothing here.
        "epoch": [-3600.4 if index % 2 == 0 else -7200.3 for index in range(8)],
        "nll_oracle_block": [90.0 + float(index) for index in range(8)],
        "oracle_n_anchors": [152.0] * 8,
    }

    mine = sufficiency_analysis.join_oracle_scores(oracle_scored_frame, per_segment)
    theirs = sibling_sufficiency_analysis.join_oracle_scores(oracle_scored_frame, per_segment)

    pd.testing.assert_frame_equal(mine, theirs)
    # Non-vacuity: every row joined, so the rows below are a reduction over real data rather than
    # two empty frames comparing equal.
    assert len(mine) == len(oracle_scored_frame)

    my_guids = frames.per_recording_means(
        mine, [column for column, _label in sufficiency_analysis.SCORE_COLUMNS]
    )
    their_guids = sibling_frames.per_recording_means(
        theirs, [column for column, _label in sibling_sufficiency_analysis.SCORE_COLUMNS]
    )
    for name, left, right, _meaning in sufficiency_analysis.GAP_METRICS:
        my_guids[name] = frames.finite_column(my_guids, left) - frames.finite_column(
            my_guids, right
        )
        their_guids[name] = sibling_frames.finite_column(
            their_guids, left
        ) - sibling_frames.finite_column(their_guids, right)

    my_rows = sufficiency_analysis.build_rows(my_guids, resamples=64, seed=3)
    their_rows = sibling_sufficiency_analysis.build_rows(their_guids, resamples=64, seed=3)

    # Compared as frames rather than as lists of dicts: a Wilcoxon on four pairs legitimately
    # reports a NaN p-value, and ``nan != nan`` would fail on two records that agree exactly.
    pd.testing.assert_frame_equal(pd.DataFrame(my_rows), pd.DataFrame(their_rows))
    # Non-vacuity again: the gap is a real, signed number over the four recordings rather than the
    # NaN two implementations that both refused to estimate would agree on.
    by_metric = {str(row["metric"]): row for row in my_rows}
    assert by_metric["delta_suff_nats"]["n"] == 4
    assert np.isfinite(float(by_metric["delta_suff_nats"]["value"]))
    assert float(by_metric["delta_suff_nats"]["value"]) > 0.0
