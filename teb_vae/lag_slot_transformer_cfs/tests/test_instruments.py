r"""The synthetic instruments: that they measure what they declare, and how they can stop doing it.

An instrument is a measuring device, and the failures that matter are the ones where it still
produces a number. Every check here is one of those:

* a generator whose planted dependence is not actually in the tensors it hands over, so a low power
  figure is a property of the generator and reads as a property of the readout;
* a band derived at one geometry and applied at another, so a recovery verdict scores the readout
  against a plant that is somewhere else;
* a lag partition that leaves lags out, so a plant sitting in one of them is unreachable and nothing
  says so;
* a criterion read from the run instead of from the declaration, which is what turns a campaign into
  a search for a threshold that passes;
* a rate whose denominator drops the runs that found nothing, which is the difference between a
  measurement and a display of successful examples.

**What is deliberately not asserted here: the outcome.** No test requires an instrument to detect
its plant. Power is a *measurement*, and a suite that failed when it came out low would be a suite
that pressures the campaign toward a flattering number -- which is the one thing an instrument must
never be tuned for. The campaign's recorded evidence carries the rates; this file establishes that
the machinery producing them is wired to the truth each generator declared.
"""
from __future__ import annotations

import math
from typing import Dict, List

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.instruments import campaign, criteria
from teb_vae.lag_slot_transformer_cfs.instruments.generators import (
    PLANTED_DELAY,
    InstrumentGeometry,
    direct_support,
    stored_feature_generators,
)

#: A geometry small enough for a suite and structurally faithful in the two things the instruments
#: read: a lag window that holds the planted band strictly inside it, and an anchor floor above the
#: plant so the earliest anchor can still reach the source it refers to.
TINY = InstrumentGeometry(
    sequence_length=64, warmup_period=28, segments=6, selection=4, holdout=6
)

#: The declared settings a suite run uses. Small in every count and identical in every rule to the
#: shipped ones, so what is exercised here is the campaign rather than a second set of criteria.
TINY_CRITERIA = criteria.Criteria(resamples=60, num_mc_samples=2)


@pytest.fixture(scope="module")
def generators() -> Dict[str, object]:
    """Every stored-feature generator at the suite geometry.

    Returns:
        ``{name: Generator}``.
    """
    return stored_feature_generators(TINY)


# =================================================================================================
# The generators carry what they declare
# =================================================================================================
def test_every_generator_produces_the_declared_shapes_and_finite_values(generators) -> None:
    """A generator whose tensors are the wrong shape fails loudly; one with a stray non-finite does
    not, and would poison a fit several frames later.

    Args:
        generators: The instruments.
    """
    for name, generator in generators.items():
        batch = generator.build(TINY, 3)
        assert batch.y_st.shape == (TINY.rows, TINY.sequence_length, TINY.n_target_scattering), name
        assert batch.y_ph.shape == (TINY.rows, TINY.sequence_length, TINY.n_target_phase), name
        assert batch.u_stream.shape == (TINY.rows, TINY.sequence_length, TINY.n_source), name
        assert batch.weight.shape == (TINY.rows, TINY.sequence_length), name
        for tensor in batch:
            assert bool(torch.isfinite(tensor).all()), name


def test_a_build_is_a_function_of_its_seed(generators) -> None:
    """Two seeds must give different segments and one seed the same ones twice.

    Without the first the repeats are one run counted three times, and the rate is a rate over one
    realisation. Without the second nothing in a campaign is reproducible from its own record.

    Args:
        generators: The instruments.
    """
    for name, generator in generators.items():
        first, again, other = (
            generator.build(TINY, 3),
            generator.build(TINY, 3),
            generator.build(TINY, 4),
        )
        assert torch.equal(first.y_st, again.y_st), name
        assert not torch.equal(first.y_st, other.y_st), name


def test_the_planted_generators_actually_carry_their_plant(generators) -> None:
    r"""Measured on the tensors, by a linear fit, before any model is involved.

    This is the check that keeps a low power figure honest. A generator whose plant never reached
    the streams would report no detections at any budget, and the campaign would read as a finding
    about the architecture. Here the source at the referenced lag is required to explain variance
    the target's own recent history does not, and a **wrong** lag is required to explain none.
    """
    generator = generators["single_delay"]
    batch = generator.build(TINY, 3)
    target = torch.cat([batch.y_st, batch.y_ph], dim=-1)[:, :, 0]
    source = batch.u_stream[:, :, :3].mean(dim=-1)

    labels: List[torch.Tensor] = []
    history: List[torch.Tensor] = []
    planted: List[torch.Tensor] = []
    wrong: List[torch.Tensor] = []
    for step in range(TINY.warmup_period, TINY.sequence_length - TINY.horizon):
        for horizon in range(1, TINY.horizon + 1):
            labels.append(target[:, step + horizon])
            history.append(target[:, step])
            planted.append(source[:, step + horizon - PLANTED_DELAY])
            wrong.append(source[:, step])

    outcome = torch.cat(labels)

    def explained(*columns: torch.Tensor) -> float:
        """Fraction of the label variance a least-squares fit on these columns explains."""
        design = torch.stack(list(columns) + [torch.ones_like(outcome)], dim=-1)
        solution = torch.linalg.lstsq(design, outcome.unsqueeze(-1)).solution
        residual = outcome - (design @ solution).squeeze(-1)
        return 1.0 - float(residual.var() / outcome.var())

    history_only = explained(torch.cat(history))
    with_plant = explained(torch.cat(history), torch.cat(planted))
    with_wrong = explained(torch.cat(history), torch.cat(wrong))

    assert with_plant > history_only + 0.05
    assert with_wrong < history_only + 0.01


def test_the_redundant_source_is_recoverable_from_target_history_and_the_planted_one_is_not(
    generators,
) -> None:
    """The property that makes one generator a control and the other an instrument.

    Asserted as a **contrast**, because either half alone is satisfied by a broken generator. A
    source that predicted nothing at all would look like a control while failing to be the one
    declared -- which is a source that a readout could mistake for novelty. So the redundant
    source has to be nearly perfectly recoverable from target channels the model already holds,
    and the planted source has to be nearly unrecoverable from them.

    Args:
        generators: The instruments.
    """

    def recoverable(name: str) -> float:
        """Fraction of the source's variance a fit on two steps of target channels explains."""
        batch = generators[name].build(TINY, 3)
        target = torch.cat([batch.y_st, batch.y_ph], dim=-1)
        source = batch.u_stream[:, 1:, 0].reshape(-1)
        design = torch.cat(
            [target[:, 1:, :].reshape(source.shape[0], -1),
             target[:, :-1, :].reshape(source.shape[0], -1),
             torch.ones(source.shape[0], 1)],
            dim=-1,
        )
        solution = torch.linalg.lstsq(design, source.unsqueeze(-1)).solution
        residual = source - (design @ solution).squeeze(-1)
        return 1.0 - float(residual.var() / source.var())

    assert recoverable("redundant_source") > 0.9
    assert recoverable("single_delay") < 0.2


# =================================================================================================
# The declared bands and the lag partition
# =================================================================================================
@pytest.mark.parametrize(
    ("offsets", "horizon", "n_lags", "expected"),
    (
        ((12,), 4, 24, (8, 11)),
        ((12, 20), 4, 24, (8, 19)),
        # Clipped at the near edge rather than reaching below lag zero.
        ((3,), 4, 24, (0, 2)),
        # Entirely past the searched window: not a band with no support, but a question the
        # geometry cannot be asked.
        ((60,), 4, 24, None),
    ),
)
def test_the_direct_support_follows_the_horizon_identity(
    offsets, horizon, n_lags, expected
) -> None:
    r"""$\ell = d - h$ pooled over the offsets and the horizon, clipped to the searched window.

    Derived rather than written down, because a fixture rebuilt at another delay or a run at another
    horizon moves it -- and a band that did not move would score a readout against a plant that is
    somewhere else.

    Args:
        offsets: The referenced source offsets.
        horizon: The forecast length.
        n_lags: The candidate lag count.
        expected: The band, or ``None`` where it falls outside the window.
    """
    assert direct_support(offsets, horizon, n_lags) == expected


def test_a_band_over_no_referenced_time_is_refused() -> None:
    """It is a generator that plants no delay and should declare none, not an empty band."""
    with pytest.raises(ValueError, match="at least one referenced source offset"):
        direct_support([], 4, 24)


def test_the_lag_partition_covers_every_lag_exactly_once() -> None:
    """A lag left out is a lag no suppression arm removes, and a plant sitting in it is unreachable.

    Checked over several widths, including ones that do not divide the axis, because the remainder
    is where a partition usually loses a lag.
    """
    for n_lags in (16, 24, 31):
        for width in (3, 4, 5, 7):
            windows = criteria.lag_windows(n_lags, width)
            covered: List[int] = []
            for low, high in windows.values():
                covered.extend(range(low, high + 1))
            assert sorted(covered) == list(range(n_lags)), (n_lags, width)


def test_a_partition_of_one_window_is_refused() -> None:
    """Suppressing every lag at once is the silence arm, which measures the equality invariant."""
    with pytest.raises(ValueError, match="whole\n?\\s*axis"):
        criteria.lag_windows(8, 8)


# =================================================================================================
# The verdicts read the declaration, not the run
# =================================================================================================
def test_a_gap_whose_interval_spans_zero_is_not_a_detection() -> None:
    """The rule that makes a false-positive rate a measurement rather than the sign of noise.

    A point estimate is positive about half the time under no effect, so a rule reading it alone
    would report a false-positive rate near one half whatever the readout did.
    """
    settings = criteria.Criteria(resamples=200)
    spanning = [0.4, -0.3, 0.2, -0.5, 0.35, -0.25, 0.1, -0.15]
    clearly_positive = [0.4, 0.35, 0.5, 0.45, 0.38, 0.42, 0.47, 0.41]

    assert criteria.relevance_verdict(spanning, settings)["detected"] is False
    assert criteria.relevance_verdict(clearly_positive, settings)["detected"] is True


def test_a_recovery_needs_both_the_right_window_and_an_interval_that_clears_zero() -> None:
    """Either clause alone admits a verdict the readout did not earn.

    A peak in the right window whose margin is indistinguishable from zero is a readout that found
    nothing and happened to rank one window first; a confident margin in the wrong window is a
    readout that found something somewhere else.
    """
    settings = criteria.Criteria(resamples=200)
    windows = {"000_003": (0, 3), "004_007": (4, 7), "008_011": (8, 11)}
    strong = [0.5, 0.45, 0.55, 0.48, 0.52, 0.5]
    weak = [0.02, -0.03, 0.05, -0.04, 0.01, 0.0]
    nothing = [-0.4, -0.3, -0.5, -0.35, -0.45, -0.4]

    inside = criteria.recovery_verdict(
        {"000_003": nothing, "004_007": weak, "008_011": strong}, windows, (8, 11), settings
    )
    assert inside["recovered"] is True
    assert inside["peak_window"]["lags"] == [8, 11]
    assert inside["peak_distance_lags"] == 0

    elsewhere = criteria.recovery_verdict(
        {"000_003": strong, "004_007": weak, "008_011": nothing}, windows, (8, 11), settings
    )
    assert elsewhere["recovered"] is False
    assert elsewhere["peak_distance_lags"] > 0

    faint = criteria.recovery_verdict(
        {"000_003": nothing, "004_007": nothing, "008_011": weak}, windows, (8, 11), settings
    )
    assert faint["scored"] is True
    assert faint["recovered"] is False


def test_a_generator_with_no_declared_support_is_not_scored_for_recovery() -> None:
    """Recorded as unscored rather than as a failure: there is nothing to have found."""
    settings = criteria.Criteria(resamples=60)
    windows = {"000_003": (0, 3), "004_007": (4, 7)}
    values = [0.5, 0.4, 0.6, 0.45, 0.5, 0.55]

    verdict = criteria.recovery_verdict(
        {"000_003": values, "004_007": values}, windows, None, settings
    )
    assert verdict["scored"] is False
    assert verdict["recovered"] is False


# =================================================================================================
# The rates
# =================================================================================================
def _record(name: str, informative: bool, detected: bool, recovered: bool = False) -> Dict:
    """One synthetic campaign record, for the aggregation checks.

    Args:
        name: The generator's name.
        informative: Whether its source carries conditional information.
        detected: Whether the relevance rule fired.
        recovered: Whether the recovery rule fired.

    Returns:
        The record.
    """
    return {
        "generator": name,
        "truth": {"source_informative": informative, "causal": True, "note": "synthetic"},
        "relevance": {"detected": detected},
        "recovery": {"scored": True, "recovered": recovered},
    }


def test_power_and_the_false_positive_rate_never_share_a_denominator() -> None:
    """Averaging them would hide which family a firing came from, which is the whole distinction."""
    records = [
        _record("planted", True, True),
        _record("planted", True, False),
        _record("control", False, True),
        _record("control", False, False),
        _record("control", False, False),
    ]
    rates = criteria.campaign_rates(records)

    assert rates["power"]["runs"] == 2
    assert rates["false_positive_rate"]["runs"] == 3
    assert rates["power"]["rate"] == pytest.approx(0.5)
    assert rates["false_positive_rate"]["rate"] == pytest.approx(1.0 / 3.0)
    assert rates["per_generator"]["planted"]["rate_is"] == "power"
    assert rates["per_generator"]["control"]["rate_is"] == "false_positive_rate"


def test_a_run_that_found_nothing_stays_in_the_denominator() -> None:
    """A campaign that dropped its failures would report a rate over the runs that worked."""
    records = [_record("planted", True, False) for _ in range(4)]
    rates = criteria.campaign_rates(records)

    assert rates["per_generator"]["planted"]["runs"] == 4
    assert rates["per_generator"]["planted"]["detection_rate"] == 0.0
    assert rates["power"]["rate"] == 0.0


# =================================================================================================
# One run, end to end
# =================================================================================================
def test_one_run_produces_a_complete_record_and_reads_its_truth_from_the_declaration() -> None:
    """The integration, at a budget small enough for a suite.

    It asserts the record's **shape** and its provenance, never its verdict: at four optimizer steps
    nothing has been learned and a test that demanded a detection would be demanding one from noise.
    """
    generators = stored_feature_generators(TINY)
    record = campaign.run_once(
        generators["single_delay"], 5, TINY_CRITERIA, geometry=TINY, steps=4
    )

    assert record["generator"] == "single_delay"
    assert record["truth"]["direct_support"] == list(
        direct_support([PLANTED_DELAY], TINY.horizon, TINY.n_lags)
    )
    assert record["truth"]["source_informative"] is True
    assert record["geometry"]["fit_segments"] == TINY.segments
    assert record["geometry"]["scored_segments"] == TINY.holdout
    assert set(record["relevance"]) == {"detected", "gap"}
    assert record["recovery"]["scored"] is True
    assert set(record["windows"]) == set(
        criteria.lag_windows(TINY.n_lags, TINY_CRITERIA.window_width)
    )
    assert math.isfinite(float(record["kld_per_anchor"]))


def test_the_three_splits_are_disjoint_and_the_scored_one_is_never_fitted() -> None:
    """The property every rate in a campaign rests on.

    Checked on the slices themselves rather than on a downstream number, because the failure is
    silent: a run that scored the segments it fitted would report a plausible gap on every
    generator, and the controls would be the ones that showed it.
    """
    batch = stored_feature_generators(TINY)["single_delay"].build(TINY, 5)
    fitted = campaign._slice(batch, 0, TINY.segments)
    selected = campaign._slice(batch, TINY.segments, TINY.segments + TINY.selection)
    scored = campaign._slice(
        batch, TINY.segments + TINY.selection, TINY.rows
    )

    assert fitted.y_st.shape[0] == TINY.segments
    assert selected.y_st.shape[0] == TINY.selection
    assert scored.y_st.shape[0] == TINY.holdout
    assert TINY.segments + TINY.selection + TINY.holdout == TINY.rows
    # No segment appears in two splits, asserted on the values rather than on the arithmetic above.
    for left, right in ((fitted, selected), (fitted, scored), (selected, scored)):
        for row in range(min(left.y_st.shape[0], right.y_st.shape[0])):
            assert not torch.equal(left.y_st[row], right.y_st[row])


def test_a_campaign_selection_naming_no_instrument_is_refused() -> None:
    """It would otherwise run a smaller campaign and report its rates as the campaign's."""
    with pytest.raises(ValueError, match="no instrument named"):
        campaign.run_campaign(
            TINY, TINY_CRITERIA, seeds=(1,), steps=1, include_raw=False, only=("nonesuch",)
        )


def test_the_campaign_record_carries_the_criteria_it_was_decided_by() -> None:
    """A verdict whose rule is not in the artifact cannot be told from one chosen afterwards."""
    record = campaign.run_campaign(
        TINY, TINY_CRITERIA, seeds=(5,), steps=2, include_raw=False, only=("constant_source",)
    )

    assert record["criteria"]["window_width"] == TINY_CRITERIA.window_width
    assert record["criteria"]["confidence"] == TINY_CRITERIA.confidence
    assert "interval on the per-segment predictive gap" in record["criteria"]["relevance_rule"]
    assert record["rates"]["per_generator"]["constant_source"]["rate_is"] == "false_positive_rate"
    assert record["fit"]["steps"] == 2
