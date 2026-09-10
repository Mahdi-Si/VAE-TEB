r"""The lag readouts, and the three ways one of them can look like a finding without being one.

The failures this file exists to catch are all of the same shape: a number that is arithmetically
correct and reads as something it is not.

* **An unsupported bin reported as a zero.** A band deep in the warm-up staircase has almost no
  available source to remove, so its margin is small for a reason that is about the schedule. Read
  as a measured zero it says "the source did not matter there", which is the sentence the evidence
  motivating this architecture was full of and could not distinguish from an absent effect.
* **A cancellation ratio without its denominator.** Near zero it means either that the proposals
  cancel or that they are all near zero -- a source pathway arguing with itself, or one that has
  switched off. Both tests below produce the same ratio from those two states, which is the point.
* **A qualification that lives only in the code.** It has to be in the artifact a reader opens, so
  it is asserted on what the readout returns rather than on the constant that produced it.
"""
from __future__ import annotations

import pytest
import torch
import yaml

from teb_vae.lag_slot_transformer_cfs.eval import lag_metrics
from teb_vae.lag_slot_transformer_cfs.eval.binding import DEFAULT_OVERRIDES_PATH


@pytest.fixture(scope="module")
def shipped_bands():
    """The lag bands the committed override delta declares.

    Returns:
        ``{name: [lo, hi]}``.
    """
    raw = yaml.safe_load(DEFAULT_OVERRIDES_PATH.read_text(encoding="utf-8"))
    return raw["eval_config"]["occlusion_bands"]


# =============================================================================
# Bands and their reference arms
# =============================================================================
def test_the_shipped_bands_partition_the_searched_window(shipped_bands) -> None:
    """A reader comparing two arms compares the same intervals, so the partition is pinned.

    Read from the committed delta rather than restated, because the delta is what a run actually
    uses; a second copy here would be free to describe a partition no run scores.
    """
    edges = sorted((int(lo), int(hi)) for lo, hi in shipped_bands.values())
    assert edges[0][0] == 0
    for (_, previous_hi), (next_lo, _) in zip(edges, edges[1:]):
        assert next_lo == previous_hi + 1, edges


def test_the_two_reference_arms_bracket_the_declared_bands(shipped_bands) -> None:
    """Whole-band and joint removals come before any single-lag result, which is what this order
    encodes: a single-lag peak read off a window whose joint removal does nothing is noise."""
    n_lags = max(int(hi) for _, hi in shipped_bands.values()) + 1
    masks = lag_metrics.band_masks(shipped_bands, n_lags)

    assert list(masks)[0] == "none"
    assert list(masks)[-1] == "all"
    assert not bool(masks["none"].any())
    # The declared partition covers the searched window, so the joint arm removes every lag.
    assert bool(masks["all"].all())


def test_an_empty_band_refuses_rather_than_reporting_a_row_of_zeros() -> None:
    """It removes nothing, so its margin would be identically zero and read as an absent effect."""
    with pytest.raises(ValueError, match="is empty"):
        lag_metrics.band_masks({"backwards": [5, 2]}, 10)


def test_a_band_wider_than_the_searched_window_refuses() -> None:
    """It would be reported under a name that overstates what was removed."""
    with pytest.raises(ValueError, match="past the model's last candidate lag"):
        lag_metrics.band_masks({"far": [0, 20]}, 10)


# =============================================================================
# Exposure
# =============================================================================
def test_exposure_counts_anchors_and_channels_separately() -> None:
    """Index support and feature warm-up are different conditions that fail at different anchors.

    A lag can be in range at every anchor and carry two channels of forty-six, and an anchor count
    alone would report it as fully exposed.
    """
    batch, anchors, n_lags, channels = 2, 3, 4, 5
    channel_mask = torch.ones(batch, anchors, n_lags, channels, dtype=torch.bool)
    # One lag carries a single channel: in range everywhere, almost entirely cold.
    channel_mask[:, :, 2, 1:] = False
    lag_valid = channel_mask.any(dim=-1)
    contributing = torch.ones(batch, anchors)

    counts = lag_metrics.lag_exposure(lag_valid, channel_mask, contributing)

    scored = float(batch * anchors)
    assert counts["scored_anchors"] == scored
    assert counts["anchors_per_lag"].tolist() == [scored] * n_lags
    assert counts["channels_per_lag"].tolist() == [
        scored * channels,
        scored * channels,
        scored * 1.0,
        scored * channels,
    ]


def test_exposure_ignores_anchors_the_run_did_not_score() -> None:
    """A per-lag readout computed over unscored anchors measures the availability schedule."""
    channel_mask = torch.ones(1, 4, 2, 3, dtype=torch.bool)
    lag_valid = channel_mask.any(dim=-1)
    contributing = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    counts = lag_metrics.lag_exposure(lag_valid, channel_mask, contributing)

    assert counts["scored_anchors"] == 2.0
    assert counts["anchors_per_lag"].tolist() == [2.0, 2.0]


def test_exposure_refuses_masks_that_describe_different_geometries() -> None:
    """It would count one geometry's availability against another's anchors."""
    with pytest.raises(ValueError, match="first three axes must agree"):
        lag_metrics.lag_exposure(
            torch.ones(1, 3, 4, dtype=torch.bool),
            torch.ones(1, 3, 5, 2, dtype=torch.bool),
            torch.ones(1, 3),
        )


def test_channel_exposure_finds_the_channel_that_never_warmed() -> None:
    """A channel whose warm-up outlasts the anchor floor is unavailable however well the lag index
    behaves, and a per-lag profile pooled over channels hides it entirely."""
    channel_mask = torch.ones(1, 3, 2, 4, dtype=torch.bool)
    channel_mask[..., 3] = False

    counts = lag_metrics.channel_exposure(channel_mask, torch.ones(1, 3))

    assert counts.tolist() == [6.0, 6.0, 6.0, 0.0]


def test_the_counts_accumulate_across_batches() -> None:
    """A pass sums them, so a mean of per-batch means never enters the readout."""
    first = {"anchors_per_lag": torch.tensor([1.0, 2.0])}
    second = {"anchors_per_lag": torch.tensor([3.0, 4.0])}

    running = lag_metrics.merge_counts(None, first)
    running = lag_metrics.merge_counts(running, second)

    assert running["anchors_per_lag"].tolist() == [4.0, 6.0]
    # The first batch's tensor is not written into, which a running sum built by mutation would do.
    assert first["anchors_per_lag"].tolist() == [1.0, 2.0]


# =============================================================================
# Cancellation
# =============================================================================
def _cancellation_outputs(ratio: float, denominator: float, anchors: int = 2):
    """A forward-shaped stub carrying one cancellation state.

    Args:
        ratio: The ratio to report.
        denominator: The summed per-lag norm behind it.
        anchors: Anchors in the stub batch.

    Returns:
        A mapping shaped like the forward's cancellation keys.
    """
    shape = (1, anchors)
    return {
        "cancellation_ratio_mean": torch.full(shape, ratio),
        "cancellation_numerator_mean": torch.full(shape, ratio * denominator),
        "cancellation_denominator_mean": torch.full(shape, denominator),
    }


def test_a_small_ratio_is_read_differently_depending_on_its_denominator() -> None:
    """The whole reason both parts travel.

    Two states give the same near-zero ratio: proposals that cancel -- a large denominator with a
    small numerator -- and proposals that are all near zero, where both are small. A summary
    carrying the ratio alone cannot tell a source pathway arguing with itself from one that has
    switched off.
    """
    contributing = torch.ones(1, 2)
    cancelling = lag_metrics.cancellation_summary(
        lag_metrics.cancellation_totals(_cancellation_outputs(0.01, 5.0), contributing)
    )["mean"]
    inert = lag_metrics.cancellation_summary(
        lag_metrics.cancellation_totals(_cancellation_outputs(0.01, 1e-7), contributing)
    )["mean"]

    assert cancelling["ratio"] == pytest.approx(inert["ratio"])
    assert cancelling["denominator"] == pytest.approx(5.0)
    assert inert["denominator"] == pytest.approx(1e-7)


def test_the_cancellation_sums_accumulate_across_batches() -> None:
    """Averaged per batch instead, a batch holding one segment would weigh as much as a full one."""
    wide = lag_metrics.cancellation_totals(
        _cancellation_outputs(0.4, 2.0, anchors=8), torch.ones(1, 8)
    )
    narrow = lag_metrics.cancellation_totals(
        _cancellation_outputs(0.9, 2.0, anchors=2), torch.ones(1, 2)
    )

    merged = lag_metrics.merge_cancellation(None, wide)
    merged = lag_metrics.merge_cancellation(merged, narrow)
    summary = lag_metrics.cancellation_summary(merged)["mean"]

    assert summary["scored_anchors"] == 10.0
    assert summary["ratio"] == pytest.approx((0.4 * 8 + 0.9 * 2) / 10.0)


def test_a_pass_that_scored_nothing_reports_absence_rather_than_zero() -> None:
    """A zero cancellation ratio is a measurement; an empty pass is not one."""
    empty = lag_metrics.cancellation_totals(
        _cancellation_outputs(0.5, 1.0), torch.zeros(1, 2)
    )
    summary = lag_metrics.cancellation_summary(empty)["mean"]

    assert summary["scored_anchors"] == 0.0
    assert summary["ratio"] is lag_metrics.MISSING
    assert summary["denominator"] is lag_metrics.MISSING


# =============================================================================
# The margin block
# =============================================================================
def _headline(**points):
    """Bootstrap-shaped records carrying only the point estimates a margin needs.

    Args:
        **points: ``column=value``.

    Returns:
        The headline block.
    """
    return {name: {"point": value, "lo": value, "hi": value} for name, value in points.items()}


def test_a_band_with_no_available_source_is_recorded_as_missing() -> None:
    """It had nothing to remove, and a zero there is indistinguishable afterwards from a fully
    available band that did not matter."""
    headline = _headline(
        nll_full=100.0,
        **{"nll_suppress:none": 100.0, "nll_suppress:cold": 100.0, "nll_suppress:live": 103.0},
    )
    exposure = {
        "none": {"anchors": 0.0, "channels": 0.0},
        "cold": {"anchors": 40.0, "channels": 0.0},
        "live": {"anchors": 40.0, "channels": 900.0},
    }

    block = lag_metrics.band_suppression_block(headline, exposure)

    assert block["cold"]["margin_nats"] is lag_metrics.MISSING
    assert block["cold"]["band_channels"] == 0.0
    assert block["live"]["margin_nats"] == pytest.approx(3.0)
    # The reference arm removes nothing, so its zero exposure is not an absence of support.
    assert block["none"]["margin_nats"] == pytest.approx(0.0)


def test_every_band_carries_the_counts_that_say_whether_it_measured_anything() -> None:
    """A margin without its exposure is a number a reader cannot weigh."""
    headline = _headline(nll_full=10.0, **{"nll_suppress:near": 11.0})
    block = lag_metrics.band_suppression_block(
        headline, {"near": {"anchors": 512.0, "channels": 20000.0}}
    )

    assert block["near"]["band_anchors"] == 512.0
    assert block["near"]["band_channels"] == 20000.0
    # And the arm's own interval, so both ends of both arms are readable.
    assert set(block["near"]["suppressed_nll"]) >= {"point", "lo", "hi"}


def test_the_margins_are_not_renormalised_to_sum_to_the_joint_removal() -> None:
    r"""They do not decompose anything and are not made to.

    The limiter is applied after the summation, so the bounded update is not linear in the
    proposals: two bands' margins need not add to the margin of removing both, and presenting a set
    of numbers that add to something as a decomposition of that something is the misreading this
    check exists to prevent.
    """
    headline = _headline(
        nll_full=100.0,
        **{
            "nll_suppress:near": 101.0,
            "nll_suppress:far": 101.0,
            "nll_suppress:all": 105.0,
        },
    )
    exposure = {name: {"anchors": 10.0, "channels": 10.0} for name in ("near", "far", "all")}

    block = lag_metrics.band_suppression_block(headline, exposure)

    parts = block["near"]["margin_nats"] + block["far"]["margin_nats"]
    assert parts == pytest.approx(2.0)
    assert block["all"]["margin_nats"] == pytest.approx(5.0)
    # Reported as measured; nothing rescales the parts onto the whole.
    assert parts != pytest.approx(block["all"]["margin_nats"])


def test_a_run_with_no_matched_arm_reports_missing_rather_than_a_margin() -> None:
    """A margin against nothing is not a small margin."""
    block = lag_metrics.band_suppression_block(
        _headline(**{"nll_suppress:near": 11.0}), {"near": {"anchors": 1.0, "channels": 1.0}}
    )
    assert block["near"]["margin_nats"] is lag_metrics.MISSING


# =============================================================================
# The qualification
# =============================================================================
def test_the_qualification_travels_in_the_readout_rather_than_only_in_the_code() -> None:
    """A caveat that lives only in a docstring is one tidy-up away from being absent from the
    artifact a reader opens."""
    report = lag_metrics.qualified_report({"band_suppression": {}})

    text = report["qualification"]
    assert "which stored source time" in text
    assert "fitted computation" in text
    assert "physiological delay" in text
    # And it is added beside the blocks rather than replacing them.
    assert "band_suppression" in report
