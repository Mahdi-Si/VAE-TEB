r"""The ten-verdict registry, and the two criteria only this cell can have.

Eight of the ten are the shared pipeline's and keep its positions, so two summaries from two cells
of the encoder-by-target grid line up row for row. The two additions are appended rather than
interleaved, and each answers a question no other cell in the grid can ask:

``coupling_exceeds_availability_clock``
    The source availability pattern $m^u_{t,c}$ is a deterministic function of $t$, identical in
    every row of a batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$ -- so the posterior
    can be pushed off the prior by the availability *clock* alone, with no source information in
    it. The permutation control deranges rows and no permutation of rows removes what every row
    shares, which is why this is a separate criterion rather than a tightening of
    ``source_specificity``. Its threshold **ships unset**: a provisional number would decide a FAIL
    on the first production runs, which is the run that is supposed to measure it.

``anchor_geometry_intact``
    Two exact numbers rather than a statistic. The dense anchor set and the fully warm target axis
    are what every other number in a run is computed over, so a count off by one anchor means the
    population moved and nothing else in the summary would say so.

Three properties bind the whole registry and are asserted first: the reporting order is *derived*
from it rather than restated; a criterion whose inputs are absent is ``INCONCLUSIVE`` and never
``PASS``, because an unevaluated criterion reported as satisfied is worse than one not evaluated;
and the promotion list is pinned equal to the one ``report_seam`` restates without ``torch``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

from teb_vae.lag_attn_cfs.eval import metrics, report_seam
from teb_vae.lag_attn_cfs.eval import run as run_module
from teb_vae.lag_attn_cfs.eval.metrics import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    Aggregate,
    StaleCachedVerdicts,
    Verdict,
    anchor_geometry_verdict,
    availability_clock_verdict,
    build_verdicts,
    check_cached_verdicts,
    order_verdicts,
)

#: The two criteria this cell adds to the shared eight.
CELL_SPECIFIC = ("coupling_exceeds_availability_clock", "anchor_geometry_intact")


def _aggregate(**overall) -> Aggregate:
    """An aggregate carrying one recording and the named headline readouts."""
    return Aggregate(per_recording={"a": dict(overall)}, overall=dict(overall))


# =================================================================================================
# The registry is one declaration
# =================================================================================================
def test_the_reporting_order_is_derived_from_the_registry() -> None:
    """Derived, never restated: two tuples maintained by hand are two tuples that disagree."""
    assert metrics.VERDICT_ORDER == tuple(name for name, _ in metrics.VERDICT_REGISTRY)


def test_the_registry_carries_ten_criteria_and_the_two_this_cell_adds() -> None:
    assert len(metrics.VERDICT_REGISTRY) == 10
    for name in CELL_SPECIFIC:
        assert name in metrics.VERDICT_ORDER


def test_the_eight_shared_criteria_keep_the_siblings_positions() -> None:
    """So an arm table or a reader diffing two cells' summaries lines the shared rows up rather
    than comparing a raw cell's fourth criterion against this one's fifth."""
    sibling = pytest.importorskip("teb_vae.lag_attn_rws.eval.metrics")

    assert metrics.VERDICT_ORDER[: len(sibling.VERDICT_ORDER)] == sibling.VERDICT_ORDER
    assert metrics.VERDICT_ORDER[len(sibling.VERDICT_ORDER) :] == CELL_SPECIFIC


def test_the_promotion_list_matches_what_the_reporting_seam_restates() -> None:
    """``report_seam`` restates the names rather than importing them -- it must stay importable
    without ``torch`` -- so the two are pinned equal here instead of drifting apart quietly."""
    assert metrics.PROMOTED_VERDICTS == report_seam.HEADLINE_VERDICTS


def test_every_registered_criterion_is_promoted_today() -> None:
    """Not the same as promotion being redundant: a later diagnostic criterion may be worth
    reporting without being one an acceptance gate reads, and the column is what keeps that
    decision in the registry rather than in the reporting layer."""
    assert metrics.PROMOTED_VERDICTS == metrics.VERDICT_ORDER


# =================================================================================================
# A criterion with no inputs is never a pass
# =================================================================================================
def test_a_pass_that_measured_nothing_reports_ten_inconclusive_verdicts() -> None:
    """A run that scored no anchor at all must not diagnose anything. Fabricated zeros are not
    neutral: the loss criteria would FAIL on ``0.0 == 0.0``, the clamp criteria would PASS on a
    log-variance nothing ever wrote, and the geometry guard would FAIL on a geometry no forward
    ever ran."""
    statuses = {
        verdict.name: verdict.status for verdict in build_verdicts(Aggregate())
    }

    assert list(statuses) == list(metrics.VERDICT_ORDER)
    assert set(statuses.values()) == {INCONCLUSIVE}


@pytest.mark.parametrize("name", CELL_SPECIFIC)
def test_the_two_new_criteria_are_inconclusive_rather_than_absent(name: str) -> None:
    """Never omitted: the summary's verdict list is read by name and by position, so a silent gap
    in it reads exactly like a criterion that passed."""
    produced = {verdict.name for verdict in build_verdicts(Aggregate())}

    assert name in produced


def test_a_verdict_the_registry_does_not_know_is_refused() -> None:
    with pytest.raises(ValueError, match="VERDICT_REGISTRY"):
        order_verdicts([Verdict("invented_criterion", PASS, "c", "d", {})])


def test_a_registered_verdict_this_run_did_not_produce_is_refused() -> None:
    """The failure a silent gap would otherwise be: reported as missing rather than dropped."""
    produced = [
        verdict for verdict in build_verdicts(Aggregate())
        if verdict.name != "anchor_geometry_intact"
    ]

    with pytest.raises(ValueError, match="anchor_geometry_intact"):
        order_verdicts(produced)


def test_a_reused_verdict_block_from_an_older_registry_is_refused() -> None:
    """The offline re-run path reports a collected directory's verdicts verbatim, and
    ``order_verdicts`` never runs on it -- so a directory collected under nine criteria would be
    re-reported as a nine-criterion run under a pipeline that declares ten."""
    cached = [
        {"name": name} for name in metrics.VERDICT_ORDER
        if name != "coupling_exceeds_availability_clock"
    ]

    with pytest.raises(StaleCachedVerdicts, match="coupling_exceeds_availability_clock"):
        check_cached_verdicts(cached)


def test_a_verdict_block_the_registry_still_describes_is_accepted() -> None:
    check_cached_verdicts([{"name": name} for name in metrics.VERDICT_ORDER])
    check_cached_verdicts(None)


# =================================================================================================
# The availability-clock criterion
# =================================================================================================
def test_the_unset_threshold_is_inconclusive_and_still_reports_the_measurement() -> None:
    """The shipped state, and the whole point of shipping it: the number reaches the arm tables
    from the first run so the threshold can be set from the observed spread rather than guessed."""
    verdict = availability_clock_verdict(3.0, 1.25, margin_min_nats=None)

    assert verdict.status == INCONCLUSIVE
    assert verdict.values["coupling_minus_clock_nats"] == pytest.approx(1.75)
    assert "margin_min_nats" not in verdict.values


def test_the_values_block_carries_the_difference_both_interval_ends_and_the_threshold() -> None:
    """A reader is never handed only a status: the four numbers the status was decided from travel
    with it."""
    verdict = availability_clock_verdict(
        3.0, 1.25, margin_min_nats=0.5, interval=(1.1, 2.4)
    )

    assert verdict.values["source_conditioned_kl_raw"] == pytest.approx(3.0)
    assert verdict.values["kld_source_null"] == pytest.approx(1.25)
    assert verdict.values["coupling_minus_clock_nats"] == pytest.approx(1.75)
    assert verdict.values["interval_lo"] == pytest.approx(1.1)
    assert verdict.values["interval_hi"] == pytest.approx(2.4)
    assert verdict.values["margin_min_nats"] == pytest.approx(0.5)


def test_the_decision_is_on_the_intervals_lower_end_rather_than_the_point_estimate() -> None:
    """A difference measured over fourteen recordings can clear any margin on its mean while its
    interval crosses zero, so the mean is precisely the statistic that cannot decide this."""
    clears = availability_clock_verdict(3.0, 1.25, margin_min_nats=0.5, interval=(0.9, 2.4))
    misses = availability_clock_verdict(3.0, 1.25, margin_min_nats=0.5, interval=(0.1, 3.2))

    assert clears.status == PASS
    assert misses.status == FAIL
    assert misses.values["coupling_minus_clock_nats"] == pytest.approx(1.75), (
        "a FAIL must still report the measurement it failed on"
    )


def test_a_threshold_with_no_interval_is_inconclusive_rather_than_decided() -> None:
    """The interval is produced by the ``source_null`` analysis, and a verdict decided on the point
    estimate in its absence would be a different criterion under the same name."""
    verdict = availability_clock_verdict(3.0, 1.25, margin_min_nats=0.5)

    assert verdict.status == INCONCLUSIVE
    assert verdict.values["coupling_minus_clock_nats"] == pytest.approx(1.75)


def test_an_unmeasured_clock_is_inconclusive_rather_than_a_small_one() -> None:
    verdict = availability_clock_verdict(3.0, None, margin_min_nats=0.5)

    assert verdict.status == INCONCLUSIVE
    assert "coupling_minus_clock_nats" not in verdict.values


def test_the_record_states_what_the_null_actually_floors() -> None:
    """It weakens the claim in the model's favour and nothing else would surface it: zeroing floors
    no source *variation*, and the encoder's response to a flat trajectory is not literally the
    availability pattern's response."""
    verdict = availability_clock_verdict(3.0, 1.25, margin_min_nats=None)

    assert "DESIGN.md" in verdict.detail
    assert "weaker" in verdict.detail


def test_the_clock_criterion_reaches_the_registry_from_the_aggregate() -> None:
    """Threaded rather than merely available: the two columns are on the per-sample table, so the
    verdict has to read them off the aggregate's headline block."""
    verdicts = {
        verdict.name: verdict
        for verdict in build_verdicts(
            _aggregate(source_conditioned_kl_raw=3.0, kld_source_null=1.25),
            clock_margin_min_nats=0.5,
            clock_interval=(0.9, 2.4),
        )
    }

    assert verdicts["coupling_exceeds_availability_clock"].status == PASS


# =================================================================================================
# The anchor-geometry guard
# =================================================================================================
def test_the_geometry_guard_passes_only_on_the_exact_pair() -> None:
    verdict = anchor_geometry_verdict(152.0, 1.0, expected_anchors_per_sample=152)

    assert verdict.status == PASS
    assert verdict.values["anchors_per_sample"] == pytest.approx(152.0)
    assert verdict.values["expected_anchors_per_sample"] == pytest.approx(152.0)


@pytest.mark.parametrize(
    "anchors,warm", [(151.0, 1.0), (152.0, 0.999), (10.13, 1.0)]
)
def test_a_geometry_off_either_number_fails(anchors: float, warm: float) -> None:
    """The third case is the one that motivates the guard: $152/15$ is what the *training* tiling
    decodes per sample, so a run that resolved its anchor geometry from the wrong stage would
    report it here rather than nowhere."""
    verdict = anchor_geometry_verdict(anchors, warm, expected_anchors_per_sample=152)

    assert verdict.status == FAIL
    assert verdict.values["anchors_per_sample"] == pytest.approx(anchors)


def test_the_expectation_comes_from_the_checkpoint_rather_than_a_shipped_literal() -> None:
    """So a legitimate arm -- ``sweep_horizon_15``, ``sweep_floor_150`` -- moves the expectation
    with the model instead of failing a guard written against the shipped geometry."""
    verdict = anchor_geometry_verdict(120.0, 1.0, expected_anchors_per_sample=120)

    assert verdict.status == PASS


def test_an_offline_rerun_with_no_checkpoint_cannot_decide_the_count() -> None:
    """``--only <analysis> --output-dir <a finished run>`` reaches here with no model, and an
    unevaluated criterion reported as satisfied is worse than one not evaluated."""
    verdict = anchor_geometry_verdict(152.0, 1.0, expected_anchors_per_sample=None)

    assert verdict.status == INCONCLUSIVE
    assert verdict.values["anchors_per_sample"] == pytest.approx(152.0)


def test_the_guard_reaches_the_registry_from_the_aggregate() -> None:
    verdicts = {
        verdict.name: verdict
        for verdict in build_verdicts(
            _aggregate(anchors_per_sample=152.0, target_warm_frac=1.0),
            expected_anchors_per_sample=152,
        )
    }

    assert verdicts["anchor_geometry_intact"].status == PASS


def test_a_failing_geometry_guard_names_both_numbers_in_its_detail() -> None:
    """A status alone would send a reader back to the tables to find out *which* population the
    run measured; the sentence carries the decoded count and the required one."""
    verdict = anchor_geometry_verdict(11.0, 1.0, expected_anchors_per_sample=152)

    assert verdict.status == FAIL
    assert "11" in verdict.detail and "152" in verdict.detail
    assert "different population" in verdict.detail


# =================================================================================================
# The wiring: the one criterion an analysis has to finish
# =================================================================================================
def _results(
    *,
    coupling: float = 3.0,
    clock: float = 1.25,
    interval: Tuple[float, float] = (0.9, 2.4),
    positive_fraction: float = 1.0,
    n_recordings: int = 14,
    with_analysis: bool = True,
) -> Dict[str, Any]:
    """A run's results block: the collected verdicts, plus what ``source_null`` reported.

    Args:
        coupling: The matched coupling readout.
        clock: The source-null readout.
        interval: The bootstrap interval over recordings for the difference.
        positive_fraction: The fraction of recordings on which the difference is positive.
        n_recordings: The denominator behind it.
        with_analysis: Whether the analysis ran at all. ``False`` is the ``--only`` re-run of
            something else, and the skipped-analysis case.

    Returns:
        The results mapping, shaped as a run's is.
    """
    collected = build_verdicts(
        _aggregate(
            source_conditioned_kl_raw=coupling,
            kld_source_null=clock,
            anchors_per_sample=152.0,
            target_warm_frac=1.0,
        ),
        expected_anchors_per_sample=152,
    )
    results: Dict[str, Any] = {"verdicts": [verdict.as_dict() for verdict in collected]}
    if with_analysis:
        results["source_null"] = {
            "difference": {
                "source_conditioned_kl_raw_nats": coupling,
                "kld_source_null_nats": clock,
                "coupling_minus_clock_nats": coupling - clock,
                "ci_lo": interval[0],
                "ci_hi": interval[1],
                "positive_fraction": positive_fraction,
                "n_recordings": n_recordings,
            }
        }
    return results


def _by_name(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The run's verdict block, keyed by criterion."""
    return {str(verdict["name"]): verdict for verdict in results["verdicts"]}


def test_the_shipped_unset_threshold_emits_the_measurement_and_decides_nothing() -> None:
    """The state the threshold is meant to be *set from*, and the failure mode this shape exists
    to prevent: a provisional number deciding a FAIL on the very run that is supposed to measure
    it, with nothing in the output saying which of the two mistakes it made."""
    results = _results()

    revised = run_module.revise_clock_verdict(results, eval_config={})

    assert revised is not None
    assert revised["status"] == INCONCLUSIVE
    assert _by_name(results)["coupling_exceeds_availability_clock"] is revised
    values = revised["values"]
    assert values["coupling_minus_clock_nats"] == pytest.approx(1.75)
    assert values["interval_lo"] == pytest.approx(0.9)
    assert values["interval_hi"] == pytest.approx(2.4)
    # The denominators, beside a status that says nothing: with no threshold this is all a reader
    # gets, and a difference without the recording count behind it is one nobody can weigh.
    assert values["positive_fraction"] == pytest.approx(1.0)
    assert values["n_recordings"] == pytest.approx(14.0)
    assert "margin_min_nats" not in values


def test_a_set_threshold_decides_on_the_intervals_lower_end() -> None:
    """The interval, not the point estimate: a difference over fourteen recordings can clear any
    margin on its mean while its interval crosses zero."""
    clears = _results(interval=(0.9, 2.4))
    misses = _results(interval=(0.1, 3.2))
    config = {"clock_margin_min_nats": 0.5}

    assert run_module.revise_clock_verdict(clears, eval_config=config)["status"] == PASS
    assert run_module.revise_clock_verdict(misses, eval_config=config)["status"] == FAIL


def test_a_run_whose_two_readouts_are_equal_fails_and_says_it_is_measuring_a_clock() -> None:
    """The hazard in its pure form: the whole coupling readout is the availability pattern, and
    the run must say so rather than reporting a healthy-looking coupling."""
    results = _results(coupling=2.0, clock=2.0, interval=(-0.1, 0.1))

    revised = run_module.revise_clock_verdict(
        results, eval_config={"clock_margin_min_nats": 0.5}
    )

    assert revised["status"] == FAIL
    assert revised["values"]["coupling_minus_clock_nats"] == pytest.approx(0.0)
    assert "deterministic function of time" in revised["detail"]


@pytest.mark.parametrize(
    "results",
    [
        _results(with_analysis=False),
        _results(interval=(float("nan"), float("nan"))),
    ],
    ids=["analysis_skipped", "no_interval"],
)
def test_a_missing_interval_leaves_the_collected_inconclusive_standing(results) -> None:
    """Never a pass by default: the criterion is stated on an interval, so its absence leaves the
    criterion unevaluated rather than satisfied -- and an unevaluated criterion reported as
    satisfied is worse than one not evaluated."""
    before = _by_name(results)["coupling_exceeds_availability_clock"]["status"]

    revised = run_module.revise_clock_verdict(
        results, eval_config={"clock_margin_min_nats": 0.5}
    )

    assert revised is None
    assert before == INCONCLUSIVE
    assert _by_name(results)["coupling_exceeds_availability_clock"]["status"] == INCONCLUSIVE


def test_the_revision_replaces_one_criterion_and_leaves_the_registry_intact() -> None:
    """The verdict block is read by name *and* by position, so a revision that reordered it or
    dropped a criterion would be read as a criterion that passed."""
    results = _results()
    before = [verdict["name"] for verdict in results["verdicts"]]

    run_module.revise_clock_verdict(results, eval_config={"clock_margin_min_nats": 0.5})

    assert [verdict["name"] for verdict in results["verdicts"]] == before
    assert before == list(metrics.VERDICT_ORDER)
    # Only the one criterion moved.
    assert _by_name(results)["anchor_geometry_intact"]["status"] == PASS


def test_the_measured_difference_reaches_the_headline_whatever_the_verdict_said() -> None:
    """The number is not conditional on the threshold, which is what lets the threshold be set from
    the observed spread across the first production runs rather than guessed."""
    from teb_vae.lag_attn_cfs.eval.binding import CFS_BINDING

    results = _results()
    run_module.revise_clock_verdict(results, eval_config={})

    headline = report_seam.build_headline(results, CFS_BINDING.headline_scalars)

    assert headline["verdict_coupling_exceeds_availability_clock"] == INCONCLUSIVE
    assert headline["coupling_minus_clock_nats"] == pytest.approx(1.75)
    assert headline["coupling_minus_clock_ci_lo"] == pytest.approx(0.9)
    assert headline["kld_source_null_nats"] == pytest.approx(1.25)
