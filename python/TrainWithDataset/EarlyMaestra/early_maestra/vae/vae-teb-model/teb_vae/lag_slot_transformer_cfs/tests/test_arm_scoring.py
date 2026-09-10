r"""The comparator arms through the real fit and the real scoring pass, at fixture scale.

The structural file beside this one proves what each arm *is*. This one proves that each of them
survives the pipeline it is about to be run through for days, which is a different question with
different failure modes and none of them visible from a constructor:

* a fit that raises on its first validation step because a metric surface reads a column the arm
  does not emit;
* a scoring pass that raises because an intervention path assumes the recommended arm's cached
  per-lag updates;
* a control reported as a margin of exactly zero on an arm where it was never an intervention at
  all, which reads as a finding and is a defect;
* a summary that records the numbers and not which arm produced them, leaving a directory of runs
  that cannot be assembled into a comparison afterwards.

**What this file does not do.** It trains nothing to convergence, uses no production shard and
makes no claim about which mechanism matters. It establishes that the comparison an operator is
about to spend days on is wired to answer its question.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from teb_vae.lag_slot_transformer_cfs.eval import verify as eval_verify

from .test_arms import fit_arm, score_arm

pytestmark = pytest.mark.slow

#: The arms fitted and scored here, and the leaf each one moves against the recommended candidate.
#:
#: Three rather than five, and the three are chosen by which of them exercise machinery no other
#: arm does. The attention fusion brings a second suppression path and a forward that publishes
#: neither per-lag updates nor a cancellation ratio. The convolution stem brings an adapter and a
#: state gather. The capacity control brings two interventions that must be skipped by name rather
#: than reported as zeros. The mean-only arm's evaluation path is the candidate's with one absent
#: key, which the structural file and the smoke fit already cover.
SCORED_ARMS: Dict[str, Dict[str, Any]] = {
    "pointwise_attention": {
        "model_config.VAE_model.lag_fusion": "attention",
        # Refused together with an attention fusion, which normalises over the whole lag axis: a
        # chunk would renormalise inside each chunk and compute a different model. The fixture
        # configuration sets one, so an arm built on it has to clear it.
        "model_config.VAE_model.lag_chunk": None,
    },
    "attention_reference": {
        "model_config.VAE_model.lag_fusion": "attention",
        "model_config.VAE_model.lag_chunk": None,
        "model_config.VAE_model.source_stem": "conv",
    },
    "capacity_control": {"model_config.VAE_model.source_values_withheld": True},
}


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    """Fit and score every comparator arm once.

    Module-scoped because the fits are the expensive part of this file and every assertion below
    reads the same runs.

    Args:
        tmp_path_factory: pytest's directory factory.

    Returns:
        A mapping from arm name to its parsed summary.
    """
    # A short root: the fits write deeply nested run directories, and a long temporary prefix
    # pushes the checkpoint paths past what the platform will rename.
    work = Path(tmp_path_factory.mktemp("m"))
    return {
        name: score_arm(work, name, fit_arm(work, name, overrides))
        for name, overrides in SCORED_ARMS.items()
    }


@pytest.mark.parametrize("arm", sorted(SCORED_ARMS))
def test_every_comparator_arm_fits_and_scores_end_to_end(arm: str, scored) -> None:
    """The claim that costs the most to discover late.

    A comparator that trains for a week and then cannot be scored through the candidate's estimator
    is not a comparator: the one comparison it exists for is between two models scored the same
    way.

    Args:
        arm: The arm to read.
        scored: The fitted and scored arms.
    """
    summary = scored[arm]

    assert summary["arm"]["source_disabled"] is False
    assert summary["headline"]["pred_gap"]["point"] is not None
    assert summary["n_recordings"] > 0


@pytest.mark.parametrize("arm", sorted(SCORED_ARMS))
def test_every_summary_says_which_arm_produced_it(arm: str, scored) -> None:
    """Otherwise a directory of runs is a set of numbers with no subjects.

    The parameter split travels with it for the same reason: two arms differing by a fusion differ
    by however many weights that fusion holds, and a predictive difference is not attributable to a
    mechanism until the budgets are on the table beside it.

    Args:
        arm: The arm to read.
        scored: The fitted and scored arms.
    """
    block = scored[arm]["arm"]
    expected_fusion = (
        "attention"
        if SCORED_ARMS[arm].get("model_config.VAE_model.lag_fusion") == "attention"
        else "local"
    )
    expected_stem = SCORED_ARMS[arm].get("model_config.VAE_model.source_stem", "pointwise")

    assert block["lag_fusion"] == expected_fusion
    assert block["source_stem"] == expected_stem
    assert block["parameters"]["source"] > 0
    assert (
        block["parameters"]["source"] + block["parameters"]["target"]
        == block["parameters"]["total"]
    )


def test_the_attention_arms_record_what_their_band_margins_mean(scored) -> None:
    """The one readout that does not compare across the pair, said in the artifact.

    Removing a lag from an explicit sum leaves every other term standing; removing it from a
    normalised distribution grows the survivors. Both answer the same question and the two numbers
    are not on one scale, so a reader who differenced them would be measuring the aggregation.

    Args:
        scored: The fitted and scored arms.
    """
    note = scored["pointwise_attention"]["arm"]["suppression_semantics"]

    assert "normalised distribution" in note
    assert "never against the other" in note
    # And the local arm says the opposite, so the distinction is legible from either file alone.
    assert scored["capacity_control"]["arm"]["suppression_semantics"].startswith(
        "a band's proposals are removed from the explicit sum"
    )


def test_the_attention_arms_report_no_distribution_over_lags_anywhere(scored) -> None:
    """The gate that would catch a distribution being published under an attention name.

    Run against the arm that actually has one, which is the only place it could ever fire.

    Args:
        scored: The fitted and scored arms.
    """
    for arm in ("pointwise_attention", "attention_reference"):
        result = eval_verify.verify(scored[arm])
        assert result["failed"] == [], (arm, result["failed"])


def test_the_attention_arms_still_reproduce_their_reference_identities(scored) -> None:
    """On real trained weights, through the selector path rather than the subtractive one.

    The empty-band arm must reproduce the matched forward and the silence arm the prior, or every
    margin in the file carries the difference between two code paths as well as its intervention.
    This is the check that the second suppression path is the same computation as the forward.

    Args:
        scored: The fitted and scored arms.
    """
    for arm in ("pointwise_attention", "attention_reference"):
        verdict = next(
            record
            for record in eval_verify.verify(scored[arm])["verdicts"]
            if record["name"] == "reference_arms_are_exact"
        )
        assert verdict["status"] == "PASS", (arm, verdict)


def test_the_capacity_control_skips_the_controls_that_are_not_interventions_on_it(
    scored,
) -> None:
    """Named with a reason, never reported as margins of zero.

    A zero margin is the finding that replacing the source values changed nothing. On this arm they
    were never read, so the two must not read the same in an artifact. Band suppression is a
    different case and still runs: lag identity and availability are real inputs to this arm, and
    removing a band removes them.

    Args:
        scored: The fitted and scored arms.
    """
    controls = scored["capacity_control"]["source_controls"]

    assert {"replace", "permute"} <= set(controls["skipped"])
    for arm in ("replace", "permute"):
        assert "withhold" in controls["skipped"][arm]
    for margin in (
        "replace_zeros_margin_nats",
        "replace_constant_margin_nats",
        "permute_margin_nats",
    ):
        assert controls[margin] is None, margin
    # The silence arm still runs and is still the equality invariant, on this arm as on every other.
    assert controls["silence_margin_nats"] is not None
    assert scored["capacity_control"]["lag_readouts"]["band_suppression"] != {}


def test_the_convolution_arm_discloses_a_reach_the_pointwise_one_does_not(scored) -> None:
    """The quantity the comparison between the two exists to move.

    It is the resolution floor of every band margin taken over that representation: two lags closer
    together than this reach are summaries of overlapping windows. Disclosed by the run, read off
    the module, so a comparator at another schedule reports its own.

    Args:
        scored: The fitted and scored arms.
    """
    conv = scored["attention_reference"]["encoder_disclosure"]
    pointwise = scored["pointwise_attention"]["encoder_disclosure"]

    assert conv["source_stem"] == "conv"
    assert pointwise["source_receptive_field_steps"] == 1
    assert conv["source_receptive_field_steps"] > 1
    assert conv["searched_lag_steps"] == pointwise["searched_lag_steps"]


def test_both_fusions_report_the_same_exposure_axes(scored) -> None:
    """So two arms' exposure tables are comparable rather than differently shaped.

    Availability is a property of the channel and the stored step. An encoder that mixed the values
    together did not make the announcement disappear, and an exposure table that changed with the
    representation would not be an exposure table.

    Args:
        scored: The fitted and scored arms.
    """
    shapes = {
        arm: {
            key: len(scored[arm]["lag_readouts"]["exposure"][key])
            for key in ("per_lag_anchors", "per_lag_channels", "per_source_channel")
        }
        for arm in SCORED_ARMS
    }
    assert len(set(map(str, shapes.values()))) == 1, shapes
    assert all(count > 0 for count in shapes["attention_reference"].values())
