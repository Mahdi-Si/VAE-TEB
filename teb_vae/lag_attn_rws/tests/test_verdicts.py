r"""The acceptance verdicts: three-valued, and never a label without its numbers.

The two predictive criteria are the model's own,
$D_{\mathrm{full}} < D_{\mathrm{base}}$ and
$D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$. The two representation criteria
check that what is being measured sits where the design claims it does: that the prior latent
carries the target's predictive state -- shuffling it must hurt, because a prior that can be
swapped between recordings at no cost was never carrying one -- and that the KL has not collapsed
onto one or two dimensions.

``INCONCLUSIVE`` is a real outcome here, not a hedge. A control that could not run and a criterion
that failed are different facts, and reporting the first as ``FAIL`` would make a small last batch
look like a broken model.
"""
from __future__ import annotations

import pytest

from teb_vae.lag_attn_rws.eval.metrics import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    VERDICT_ORDER,
    Aggregate,
    build_verdicts,
)


def _aggregate(**scores) -> Aggregate:
    """An aggregate carrying the named branch scores and a healthy latent."""
    overall = {f"mc_nll_{name}_block": value for name, value in scores.items()}
    return Aggregate(overall=overall, kld_per_dim=[0.4, 0.3, 0.2, 0.001])


def _by_name(verdicts) -> dict:
    """Index verdicts by name."""
    return {verdict.name: verdict for verdict in verdicts}


def test_every_verdict_is_reported_every_time():
    """A criterion that silently disappears when it cannot be evaluated reads as a criterion
    that passed.

    Driven from the registry rather than from a literal list, deliberately. The list here was the
    four original criteria; the three variance criteria that joined them made this the *third*
    place a verdict name had to be written down, and a test that has to be edited to add a
    criterion stops being a guard against one going missing. The four originals are pinned
    separately below, so relaxing to the registry does not relax what they are.
    """
    names = [verdict.name for verdict in build_verdicts(_aggregate())]

    assert names == list(VERDICT_ORDER)
    assert len(names) == len(set(names))


def test_the_four_original_criteria_are_still_reported_in_order():
    """The registry may grow; it may not quietly drop one of these."""
    names = [verdict.name for verdict in build_verdicts(_aggregate())]
    originals = [
        "predictive_improvement",
        "source_specificity",
        "prior_carries_target_state",
        "latent_not_collapsed",
    ]

    assert [name for name in names if name in originals] == originals


def test_no_verdict_is_a_bare_boolean_and_each_carries_its_numbers():
    verdicts = build_verdicts(_aggregate(base=10.0, full=8.0, shuffled=14.0,
                                         base_shuffled_mu=60.0))

    for verdict in verdicts:
        assert verdict.status in {PASS, FAIL, INCONCLUSIVE}
        assert verdict.criterion and verdict.detail
        assert verdict.values, f"{verdict.name} reports a status with nothing behind it"
        assert verdict.as_dict()["status"] == verdict.status


# =============================================================================
# The predictive criteria
# =============================================================================
@pytest.mark.parametrize(
    ("base", "full", "expected"), [(10.0, 8.0, PASS), (10.0, 10.0, FAIL), (8.0, 10.0, FAIL)]
)
def test_the_source_must_actually_improve_the_forecast(base, full, expected):
    verdict = _by_name(build_verdicts(_aggregate(base=base, full=full)))[
        "predictive_improvement"
    ]

    assert verdict.status == expected
    assert verdict.values["pred_gap"] == pytest.approx(base - full)


@pytest.mark.parametrize(
    ("base", "full", "shuffled", "expected"),
    [
        (10.0, 8.0, 14.0, PASS),
        # A wrong source that is no worse than no source: the posterior reacts to any source,
        # which is precisely what a nonzero KL alone cannot distinguish.
        (10.0, 8.0, 9.0, FAIL),
        # The full forecast is not better than the base one, so the ordering fails at its head.
        (10.0, 11.0, 14.0, FAIL),
    ],
)
def test_a_strangers_source_must_be_worse_than_no_source(base, full, shuffled, expected):
    verdict = _by_name(
        build_verdicts(_aggregate(base=base, full=full, shuffled=shuffled))
    )["source_specificity"]

    assert verdict.status == expected
    assert verdict.values["shuffle_penalty"] == pytest.approx(shuffled - base)


@pytest.mark.parametrize(
    ("full", "shuffled", "expected"),
    [
        (8.0, 14.0, PASS),
        # A stranger's source forecasts exactly as well as this recording's: nothing the source
        # pathway carries is specific to this recording, whatever the KL says.
        (8.0, 8.0, FAIL),
        (8.0, 7.0, FAIL),
    ],
)
def test_a_strangers_source_must_be_worse_than_this_recordings_own(full, shuffled, expected):
    """The one predictive criterion referenced against ``full`` rather than against ``base``."""
    verdict = _by_name(
        build_verdicts(_aggregate(base=10.0, full=full, shuffled=shuffled))
    )["source_margin_positive"]

    assert verdict.status == expected
    assert verdict.values["source_margin"] == pytest.approx(shuffled - full)


def test_a_negative_gain_with_a_positive_margin_is_reported_as_both():
    r"""The state the margin criterion exists to make sayable, and the reason it is not redundant.

    A model whose latent geometry charges more for the source than the source delivers scores
    $D_{\rm full} > D_{\rm base}$ -- no predictive gain -- while a derangement-shuffled stranger
    still scores worse than the matched source. Every criterion referenced against $D_{\rm base}$
    fails, and the run is nonetheless using *this* recording's source. Before the margin criterion
    the summary could only say ``FAIL``; it must now say ``FAIL / PASS / FAIL``, and the middle
    one is the finding.

    Asserted against the registry rather than a literal count, because the emitted list carries a
    synthetic entry beyond it and a count would pin the wrong number.
    """
    verdicts = build_verdicts(_aggregate(base=10.0, full=12.0, shuffled=14.0))
    by_name = _by_name(verdicts)

    assert by_name["predictive_improvement"].status == FAIL
    assert by_name["source_margin_positive"].status == PASS
    assert by_name["source_specificity"].status == FAIL
    # And the three are adjacent and in this order, because they are read as a triple.
    order = [verdict.name for verdict in verdicts]
    triple = ["predictive_improvement", "source_margin_positive", "source_specificity"]
    assert order[: len(triple)] == triple


def test_the_stronger_criterion_implies_the_weaker_one():
    """``source_specificity`` asks D_full < D_base < D_shuffled, which entails D_shuffled >
    D_full. A run cannot pass the ordering and fail the margin, and a future edit that let it
    would mean one of the two had stopped measuring what its name says."""
    for base, full, shuffled in [
        (10.0, 8.0, 14.0), (10.0, 8.0, 9.0), (10.0, 12.0, 14.0), (10.0, 9.9, 10.1),
    ]:
        by_name = _by_name(build_verdicts(_aggregate(base=base, full=full, shuffled=shuffled)))
        if by_name["source_specificity"].status == PASS:
            assert by_name["source_margin_positive"].status == PASS


def test_a_control_that_could_not_run_is_inconclusive_not_failed():
    """A batch of one sample cannot be deranged. That is a missing measurement, not a failed
    criterion, and conflating them would report a healthy model as broken."""
    verdicts = _by_name(build_verdicts(_aggregate(base=10.0, full=8.0)))

    assert verdicts["predictive_improvement"].status == PASS
    assert verdicts["source_margin_positive"].status == INCONCLUSIVE
    assert verdicts["source_specificity"].status == INCONCLUSIVE
    assert verdicts["prior_carries_target_state"].status == INCONCLUSIVE


# =============================================================================
# The prior-shuffle control
# =============================================================================
def test_a_prior_latent_that_can_be_swapped_for_free_fails():
    """The check that the whole readout rests on: with no ``decoder_state`` bypass the prior
    latent is the only route target history has to the decoder, so a stranger's prior must
    forecast this recording badly. If it does not, $\\mu^p$ is not the target state and every
    interpretation built on that reading is unsupported."""
    verdict = _by_name(
        build_verdicts(_aggregate(base=10.0, full=8.0, base_shuffled_mu=10.0))
    )["prior_carries_target_state"]

    assert verdict.status == FAIL
    assert verdict.values["degradation"] == pytest.approx(0.0)


def test_a_badly_damaged_baseline_passes():
    verdict = _by_name(
        build_verdicts(_aggregate(base=10.0, full=8.0, base_shuffled_mu=500.0))
    )["prior_carries_target_state"]

    assert verdict.status == PASS
    assert verdict.values["degradation"] == pytest.approx(490.0)


def test_a_degradation_below_the_stated_margin_is_inconclusive():
    """The margin is provisional until a converged run revises it, so a value under it is
    reported as unresolved rather than as a verdict the number cannot support."""
    verdict = _by_name(
        build_verdicts(
            _aggregate(base=10.0, full=8.0, base_shuffled_mu=10.4),
            prior_shuffle_min_nats=1.0,
        )
    )["prior_carries_target_state"]

    assert verdict.status == INCONCLUSIVE
    assert verdict.values["margin"] == pytest.approx(1.0)


def test_the_margin_is_configurable_and_moves_the_verdict():
    scores = dict(base=10.0, full=8.0, base_shuffled_mu=10.4)

    loose = _by_name(build_verdicts(_aggregate(**scores), prior_shuffle_min_nats=0.1))
    strict = _by_name(build_verdicts(_aggregate(**scores), prior_shuffle_min_nats=5.0))

    assert loose["prior_carries_target_state"].status == PASS
    assert strict["prior_carries_target_state"].status == INCONCLUSIVE


# =============================================================================
# Latent collapse
# =============================================================================
def test_a_latent_spread_over_several_dimensions_passes():
    verdict = _by_name(build_verdicts(_aggregate()))["latent_not_collapsed"]

    assert verdict.status == PASS
    assert verdict.values["active_dims"] == pytest.approx(3.0)


def test_a_latent_collapsed_onto_one_dimension_fails():
    aggregate = Aggregate(overall={}, kld_per_dim=[5.0, 1e-6, 1e-6, 1e-6])

    verdict = _by_name(build_verdicts(aggregate))["latent_not_collapsed"]

    assert verdict.status == FAIL
    assert verdict.values["active_dims"] == pytest.approx(1.0)
    assert verdict.values["top_dimension_share"] > 0.99


def test_a_latent_with_no_kl_at_all_is_inconclusive_rather_than_collapsed():
    """An untrained or fully collapsed source pathway has no information to distribute, which is
    a different diagnosis from a badly shaped latent and points at a different fix."""
    verdict = _by_name(build_verdicts(Aggregate(kld_per_dim=[0.0, 0.0, 0.0])))[
        "latent_not_collapsed"
    ]

    assert verdict.status == INCONCLUSIVE


def test_the_minimum_active_dimension_count_is_configurable():
    aggregate = Aggregate(kld_per_dim=[0.4, 0.3, 0.0, 0.0])

    assert _by_name(build_verdicts(aggregate, min_active_dims=2))[
        "latent_not_collapsed"
    ].status == PASS
    assert _by_name(build_verdicts(aggregate, min_active_dims=3))[
        "latent_not_collapsed"
    ].status == FAIL
