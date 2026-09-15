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


def test_band_exposure_sums_each_band_over_its_own_lags() -> None:
    """The reference arms come out too: ``none`` removes nothing and ``all`` covers the window."""
    masks = lag_metrics.band_masks({"near": [0, 1], "far": [2, 3]}, 4)
    counts = {
        "anchors_per_lag": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        "channels_per_lag": torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float64),
    }

    exposure = lag_metrics.band_exposure(masks, counts)

    assert exposure == {
        "none": {"anchors": 0.0, "channels": 0.0},
        "near": {"anchors": 3.0, "channels": 30.0},
        "far": {"anchors": 7.0, "channels": 70.0},
        "all": {"anchors": 10.0, "channels": 100.0},
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a second device")
def test_band_exposure_reads_device_masks_against_cpu_counts() -> None:
    """The pass builds the masks on the model device and accumulates the counts on the CPU.

    The reduction is the one place the two meet, and on a CPU-only run they coincide, so the
    mismatch only shows on a GPU checkpoint: this is the test that would have caught it.
    """
    masks = lag_metrics.band_masks({"near": [0, 1]}, 3, device=torch.device("cuda"))
    counts = {
        "anchors_per_lag": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        "channels_per_lag": torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
    }

    exposure = lag_metrics.band_exposure(masks, counts)

    assert exposure["near"] == {"anchors": 3.0, "channels": 9.0}


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


# =============================================================================
# Paired intervals
# =============================================================================
def _per_recording(**columns):
    """A per-recording table from ``column=[values]``, recordings named by position.

    Args:
        **columns: One list per column, all the same length.

    Returns:
        ``{recording: {column: value}}``.
    """
    length = len(next(iter(columns.values())))
    return {
        f"rec{index}": {name: values[index] for name, values in columns.items()}
        for index in range(length)
    }


def test_a_paired_margin_is_an_interval_of_per_recording_differences() -> None:
    """The interval that belongs on a margin is the interval of the paired differences.

    Two arms whose scores swing together across recordings have wide intervals of their own and a
    narrow interval on their difference, and reading the two arm intervals as overlapping would
    call a consistent margin no difference.
    """
    matched = [100.0, 250.0, 40.0, 180.0, 90.0, 300.0]
    intervened = [value + 2.0 for value in matched]
    table = _per_recording(nll_full=matched, **{"nll_suppress:near": intervened})

    record = lag_metrics.paired_margin(
        table, "nll_suppress:near", "nll_full", resamples=200, seed=0
    )

    assert record["n_paired"] == 6
    assert record["point"] == pytest.approx(2.0)
    # Every difference is exactly two, so the interval has no width at all.
    assert record["lo"] == pytest.approx(2.0)
    assert record["hi"] == pytest.approx(2.0)


def test_a_paired_margin_is_missing_when_an_arm_did_not_run() -> None:
    """Missing, not zero: an arm that did not run measured nothing."""
    table = _per_recording(nll_full=[1.0, 2.0, 3.0])

    assert lag_metrics.paired_margin(table, "nll_permute", "nll_full", resamples=100, seed=0) is (
        lag_metrics.MISSING
    )


def test_the_suppression_block_carries_the_paired_interval_when_given() -> None:
    """Beside the point margin, under its own key, and absent where no pairing was possible."""
    headline = _headline(nll_full=10.0, **{"nll_suppress:near": 11.0, "nll_suppress:cold": 10.5})
    exposure = {
        "near": {"anchors": 4.0, "channels": 8.0},
        "cold": {"anchors": 4.0, "channels": 0.0},
    }
    intervals = {"near": {"point": 1.0, "lo": 0.5, "hi": 1.5}, "cold": {"point": 0.5}}

    block = lag_metrics.band_suppression_block(headline, exposure, intervals=intervals)

    assert block["near"]["margin_interval"] == intervals["near"]
    # An unsupported band has no margin and therefore no interval either.
    assert block["cold"]["margin_interval"] is lag_metrics.MISSING
    assert lag_metrics.band_suppression_block(headline, exposure)["near"]["margin_interval"] is (
        lag_metrics.MISSING
    )


# =============================================================================
# The curve bootstrap
# =============================================================================
def test_the_curve_bootstrap_takes_one_resampling_for_every_position() -> None:
    """A per-position resampling would give adjacent steps intervals over different recordings.

    Checked by the one property a shared resampling has that independent ones do not: a curve
    that is a constant multiple of another gets an interval that is the same multiple, position
    for position.
    """
    rows = {f"rec{index}": [float(index), 2.0 * float(index)] for index in range(8)}

    record = lag_metrics.bootstrap_curve(rows, resamples=300, seed=3)

    assert record["n"] == 8
    assert record["point"] == pytest.approx([3.5, 7.0])
    assert record["lo"][1] == pytest.approx(2.0 * record["lo"][0])
    assert record["hi"][1] == pytest.approx(2.0 * record["hi"][0])
    assert record["lo"][0] <= record["point"][0] <= record["hi"][0]


def test_the_curve_bootstrap_refuses_vectors_of_two_lengths() -> None:
    """Two lengths are two axes, and a mean across them is a mean of nothing in particular."""
    with pytest.raises(ValueError, match="one length"):
        lag_metrics.bootstrap_curve({"a": [1.0, 2.0], "b": [1.0]}, resamples=100, seed=0)


def test_the_curve_bootstrap_reports_absence_below_the_recording_minimum() -> None:
    """Two recordings reproduce themselves rather than estimating a spread."""
    record = lag_metrics.bootstrap_curve({"a": [1.0], "b": [2.0]}, resamples=100, seed=0)

    assert record["n"] == 2
    assert "note" in record
    assert all(value != value for value in record["point"])  # NaN throughout


# =============================================================================
# The per-lag latent profile
# =============================================================================
def _awake_forward():
    """A tiny model with a non-zero source pathway, and its forward with the proposals kept.

    The constructed head is zero at its output, so every proposal is zero and every per-lag
    readout is trivially zero; the projection is given random weights first so the profile has
    something to measure.

    Returns:
        ``(model, outputs, contributing)``.
    """
    from .conftest import build_tiny_model, tiny_streams

    model = build_tiny_model()
    torch.manual_seed(1)
    torch.nn.init.normal_(model.proposal_head.output_proj.weight, std=0.3)
    torch.nn.init.normal_(model.proposal_head.output_proj.bias, std=0.3)
    model.eval()
    y_st, y_ph, u_stream = tiny_streams()
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True)
    contributing = outputs["anchor_valid"].to(torch.float64)
    return model, outputs, contributing


def test_the_latent_profile_agrees_with_removing_one_lag_at_a_time() -> None:
    """The vectorised, chunked profile is the band suppression at the finest partition.

    Checked against the one-lag-at-a-time path the band arms already use, so the two readouts
    cannot come to describe different removals.
    """
    from teb_vae.lag_slot_transformer_cfs.nets.controls import suppressed_parameters

    model, outputs, contributing = _awake_forward()
    live = outputs["lag_valid"].to(torch.float64) * contributing[:, :, None]

    totals = lag_metrics.per_lag_latent_totals(model, outputs, contributing)

    n_lags = int(model.n_lags)
    for lag in range(n_lags):
        removed = torch.zeros(n_lags, dtype=torch.bool)
        removed[lag] = True
        single = suppressed_parameters(model, outputs, removed)
        drop = (outputs["kld_per_anchor"] - single["kld_per_anchor"]).to(torch.float64)
        shift = (outputs["update_mean"] - single["update_mean"]).norm(dim=-1).to(torch.float64)
        assert float(totals["divergence_drop_sum"][lag]) == pytest.approx(
            float((drop * live[:, :, lag]).sum()), abs=1e-6
        )
        assert float(totals["update_shift_sum"][lag]) == pytest.approx(
            float((shift * live[:, :, lag]).sum()), abs=1e-6
        )
    # The proposal norm is what the head emitted, and it is not zero on a woken pathway.
    assert float(totals["proposal_norm_sum"].sum()) > 0.0
    assert "scale_proposal_norm_sum" in totals


def test_the_latent_profile_has_no_scale_channel_on_the_mean_only_arm() -> None:
    """No scale head, no scale profile: absent rather than a row of zeros."""
    from .conftest import build_tiny_model, tiny_streams

    model = build_tiny_model(mean_only_residual=True)
    model.eval()
    y_st, y_ph, u_stream = tiny_streams()
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream, anchor_phase=0, anchor_stride=1, return_proposals=True)

    totals = lag_metrics.per_lag_latent_totals(
        model, outputs, outputs["anchor_valid"].to(torch.float64)
    )

    assert "scale_proposal_norm_sum" not in totals
    assert set(totals) == {"proposal_norm_sum", "update_shift_sum", "divergence_drop_sum"}


def test_the_latent_profile_needs_the_proposals() -> None:
    """Without them there is nothing to remove, and the refusal names the flag."""
    from .conftest import build_tiny_model

    with pytest.raises(KeyError, match="return_proposals"):
        lag_metrics.per_lag_latent_totals(build_tiny_model(), {}, torch.ones(1, 1))


def test_the_lag_profile_summary_records_missing_where_a_lag_had_no_support() -> None:
    """A lag no scored anchor was live at has no mean, and a zero there would read as inert."""
    totals = {"proposal_norm_sum": torch.tensor([4.0, 0.0, 6.0])}
    anchors = torch.tensor([2.0, 0.0, 3.0])

    summary = lag_metrics.lag_profile_summary(totals, anchors)

    assert summary["proposal_norm"] == [2.0, lag_metrics.MISSING, 2.0]
    assert summary["anchors_per_lag"] == [2.0, 0.0, 3.0]
    assert lag_metrics.lag_profile_summary(None, anchors) == {}
