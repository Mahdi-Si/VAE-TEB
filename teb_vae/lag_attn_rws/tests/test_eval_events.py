r"""The event detectors, and the three readouts built on them.

Three properties carry this suite, and each is a way the event analyses could produce a complete
set of plausible numbers while measuring something else.

**A detector must find what is there and nothing else.** The generated shards gained a
``with_events=True`` variant that injects contractions and decelerations at indices this suite can
state exactly, so "the detector works" is a comparison against a known answer rather than against
a plot. One injected deceleration is placed **on the weight gap** deliberately: a gap is stored as
$0.0$ bpm, which after z-scoring is roughly $-11\sigma$ and is therefore the deepest dip in the
recording, so a detector masking by value rather than by ``weight`` finds it. It must not be
recovered, and the detector must find it the moment the validity mask is withheld -- both
directions, because either alone passes vacuously.

**The contraction onset must be the onset.** The walk-back this module inherited reported the
*peak* as the onset, and did so silently. That defect and its correction are pinned below in the
form that distinguishes them: ``onset_index < peak_index``, and the recovered onset within a
stated tolerance of the injected one.

**A rate over events must not be a rate over anchors.** Consecutive anchors' forecast blocks
overlap in $H - 1$ of their $H$ steps, so an anchor-level hit rate counts one physiological
deceleration once per anchor. The per-event and per-anchor counts are both asserted, and their
ratio is asserted to be the number of usable horizon steps rather than assumed to be anything.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval import events
from teb_vae.lag_attn_rws.eval.analyses import events as events_analysis
from teb_vae.lag_attn_rws.eval.collect import CONTRACTION_AGE_COLUMN, Collection

from .conftest import (
    EVENT_CONTRACTION_RISE_S,
    EVENT_FHR_NOISE_BPM,
    EVENT_UP_NOISE,
    MULTI_CLASS_GAP_STEPS,
    MULTI_CLASS_SEQ_LEN,
    MULTI_CLASS_TRIM_STEPS,
    inject_events,
    injected_event_indices,
)

#: The trimmed geometry the fixture shards load at: $330 - 2 \times 15$ decimated steps of $16$.
_DECIMATION = 16
_TRIMMED_STEPS = MULTI_CLASS_SEQ_LEN - 2 * MULTI_CLASS_TRIM_STEPS
_TRIMMED_RAW = _TRIMMED_STEPS * _DECIMATION

#: How far a recovered onset may sit from the injected one. The flank walk stops where the trace
#: has risen through the last tenth of the event's prominence, so on a linear ramp the recovered
#: onset is later than the injected one by that tenth of the rise -- a stated offset rather than a
#: tolerance absorbing an unknown.
_ONSET_TOLERANCE_S = (1.0 - events.FLANK_LEVEL_FRAC) * EVENT_CONTRACTION_RISE_S + 2.0


def _synthetic_traces(seed: int = 0):
    """Return one trimmed-length ``(fhr_bpm, up, validity)`` triple carrying the injected events."""
    rng = np.random.default_rng(seed)
    fhr = (140.0 + EVENT_FHR_NOISE_BPM * rng.standard_normal(_TRIMMED_RAW))[None, :]
    up = (30.0 + EVENT_UP_NOISE * rng.standard_normal(_TRIMMED_RAW))[None, :]
    inject_events(fhr, up, trim_raw=0)
    return fhr[0], up[0], np.ones(_TRIMMED_RAW, dtype=bool)


# =============================================================================
# The detectors, against the injected answer
# =============================================================================
def test_the_injected_decelerations_are_recovered_at_their_known_positions() -> None:
    fhr, _up, valid = _synthetic_traces()

    found = events.detect_decelerations(fhr, valid=valid)["nadir_raw"]
    expected = injected_event_indices()["deceleration_nadir"]

    assert found.size == expected.size
    assert np.abs(found - expected).max() / events.FS_RAW <= 1.0


def test_the_injected_contractions_are_recovered_and_the_onset_precedes_the_peak() -> None:
    r"""The defect this module diverges from the sibling on, pinned in both of its forms.

    The inherited walk-back was entered **at** the peak, where the smoothed gradient is
    approximately zero and was therefore already below its positive threshold, so the loop body
    never ran and the reported onset *was* the peak. A characterization of that behaviour would
    read ``onset_index == peak_index``; what is asserted here is the corrected form, and the
    second assertion is what makes it a correction rather than a shift -- the onset lands where
    the contraction actually began.
    """
    _fhr, up, valid = _synthetic_traces()
    truth = injected_event_indices()

    found = events.detect_contractions(up, valid=valid)

    assert found["peak_raw"].size == truth["contraction_peak"].size
    assert (found["onset_raw"] < found["peak_raw"]).all(), "the onset must precede the peak"
    error = (found["onset_raw"] - truth["contraction_onset"]) / events.FS_RAW
    assert np.abs(error).max() <= _ONSET_TOLERANCE_S, f"onset error {error} s"


def test_a_gapped_region_produces_no_detection_and_would_without_the_mask() -> None:
    r"""Both directions, because either alone passes for the wrong reason.

    A missing sample is stored as $0.0$ bpm -- roughly $-11\sigma$ after z-scoring, the deepest
    dip in the recording -- so a detector masking by value finds it. Masking by ``weight`` is what
    removes it, and the test that only checks the masked path would pass on a detector that had
    simply lost its sensitivity.
    """
    fhr, _up, valid = _synthetic_traces()
    gap_lo = (_TRIMMED_STEPS // 2) * _DECIMATION
    gap_hi = gap_lo + MULTI_CLASS_GAP_STEPS * _DECIMATION
    fhr[gap_lo:gap_hi] = 0.0
    valid[gap_lo:gap_hi] = False

    masked = events.detect_decelerations(fhr, valid=valid)["nadir_raw"]
    by_value = events.detect_decelerations(fhr, valid=None)["nadir_raw"]

    assert not ((masked >= gap_lo - 60) & (masked < gap_hi + 60)).any()
    assert ((by_value >= gap_lo) & (by_value < gap_hi)).any(), (
        "without the mask the 0 bpm gap is itself the deepest deceleration, which is the failure "
        "the mask exists to prevent"
    )


def test_the_deceleration_detector_needs_bpm_and_finds_nothing_in_z_units() -> None:
    r"""The one behavioural change from the source these detectors were ported from.

    There, the prominence threshold was chosen at run time by a ``max(|x|) > 30`` heuristic, so a
    z-scored trace silently fell back to a $0.3\sigma$ threshold and its noise was reported as
    decelerations. Here the $10\,$bpm floor applies unconditionally: a caller that forgot to
    invert the loader's normalisation gets nothing, which is a visible failure.
    """
    fhr, _up, valid = _synthetic_traces()
    z_scored = (fhr - fhr.mean()) / fhr.std()

    assert events.detect_decelerations(z_scored, valid=valid)["nadir_raw"].size == 0
    assert events.detect_decelerations(fhr, valid=valid)["nadir_raw"].size > 0


def test_a_quiet_trace_carries_no_decelerations() -> None:
    r"""The non-hallucination direction, at a physiological noise level.

    Asserted for decelerations only, and that asymmetry is a property of the two detectors rather
    than an omission. A deceleration has an **absolute** floor -- $10\,$bpm -- so "nothing on a
    quiet trace" is a real guarantee. A contraction's prominence is $\sigma$-relative, because UP
    has no clinically absolute scale here and a fixed threshold would mean different things on two
    recordings; on pure noise that detector therefore finds the largest excursions *of the noise*,
    which is correct behaviour and not a property that can be pinned at zero. What is pinned for
    contractions is the positional agreement above and below.
    """
    rng = np.random.default_rng(3)
    quiet = 140.0 + 1.0 * rng.standard_normal(_TRIMMED_RAW)

    assert events.detect_decelerations(quiet)["nadir_raw"].size == 0


def test_the_usable_interior_is_half_the_block_and_is_computed_not_assumed() -> None:
    r"""A $480$-sample block leaves $240$ samples the detector will look at, and $14$ whole
    horizon steps -- so a per-anchor detection count is roughly fourteen times a per-event one,
    not thirty. The roadmap's ``1/30`` is the overlap of the horizon; the detector's own edge rule
    removes half of it before anything is counted."""
    lo, hi = events.usable_interval(480)
    steps = events.usable_horizon_steps(480, raw_per_step=16)

    assert (lo, hi) == (120, 360)
    assert steps.tolist() == list(range(8, 22))
    assert events.usable_horizon_steps(64, raw_per_step=16).size == 0, "a short block has none"


# =============================================================================
# The detectors, through the real loader
# =============================================================================
@pytest.mark.parametrize("field", ["fhr", "up"])
def test_the_event_shards_load_through_the_real_loader(event_loader, field) -> None:
    batch = next(iter(event_loader))

    values = getattr(batch, field)
    assert values.shape[1] == _TRIMMED_RAW
    assert torch.isfinite(values).all()


def test_every_loaded_event_segment_carries_exactly_the_injected_events(event_loader) -> None:
    """Exactly the injected events, and no others -- which is the non-hallucination property in
    the form that survives the loader, the trim and the weight gap."""
    truth = injected_event_indices()
    gap_lo = (_TRIMMED_STEPS // 2) * _DECIMATION
    gap_hi = gap_lo + MULTI_CLASS_GAP_STEPS * _DECIMATION
    expected = truth["deceleration_nadir"][
        (truth["deceleration_nadir"] < gap_lo - 60) | (truth["deceleration_nadir"] >= gap_hi + 60)
    ]

    seen = 0
    for batch in event_loader:
        for index in range(batch.fhr.shape[0]):
            valid = events.raw_validity(
                batch.weight[index].numpy(), decimation=_DECIMATION, raw_len=_TRIMMED_RAW
            )
            found = events.detect_decelerations(batch.fhr[index].numpy(), valid=valid)["nadir_raw"]
            assert found.size == expected.size, f"sample {seen}: {found} against {expected}"
            assert np.abs(found - expected).max() / events.FS_RAW <= 1.5
            seen += 1
    assert seen > 0


def test_the_plain_shards_carry_none_of_the_injected_events(multi_class_loader) -> None:
    r"""The default variant is unchanged white noise. Its smoothed FHR does swing past the
    $10\,$bpm clinical prominence -- $10\,$bpm of per-sample noise at $4\,$Hz is not a
    physiological trace -- so what is asserted is the thing that matters: none of the injected
    positions is recovered from it, so every event this suite finds came from the injection rather
    than from the fixture generator."""
    truth = injected_event_indices()
    batch = next(iter(multi_class_loader))

    for name, key, signal in (
        ("deceleration", "deceleration_nadir", batch.fhr[0].numpy()),
        ("contraction", "contraction_peak", batch.up[0].numpy()),
    ):
        found = (
            events.detect_decelerations(signal)["nadir_raw"] if name == "deceleration"
            else events.detect_contractions(signal)["peak_raw"]
        )
        if not found.size:
            continue
        distance = np.abs(found[:, None] - truth[key][None, :]).min(axis=1) / events.FS_RAW
        assert distance.min() > 5.0, f"a {name} landed on an injected position in a plain shard"


# =============================================================================
# Matching, and the per-event rate
# =============================================================================
def test_matching_pairs_within_the_tolerance_and_leaves_the_rest() -> None:
    left, right = np.array([100, 400]), np.array([104, 900])

    matched_left, matched_right = events.match_events(left, right, tolerance=8)

    assert matched_left.tolist() == [True, False]
    assert matched_right.tolist() == [True, False]


def _detection_entries(shift: int = 0, guids: int = 4) -> List[Dict[str, Any]]:
    """Detections in which every branch reproduces the truth, optionally shifted."""
    taus = [8, 9, 10]
    entries: List[Dict[str, Any]] = []
    for index in range(guids):
        truth = {tau: [1000 + 100 * tau] for tau in taus}
        moved = {tau: [value + shift for value in values] for tau, values in truth.items()}
        entries.append({"guid": f"g{index}", "truth": truth, "base": moved, "full": moved})
    return entries


def test_a_forecast_equal_to_the_truth_scores_a_perfect_hit_rate() -> None:
    """The known answer that pins the whole matching arithmetic at once."""
    rows = events_analysis.skill_rows(
        _detection_entries(), usable_tau=np.array([8, 9, 10]), raw_per_step=16,
        resamples=64, seed=0,
    )

    assert rows and all(row["hit_rate"] == pytest.approx(1.0) for row in rows)
    assert all(row["false_alarm_rate"] == pytest.approx(0.0) for row in rows)
    assert all(row["lead_time_abs_error_s"] == pytest.approx(0.0) for row in rows)


def test_a_shifted_forecast_loses_the_hits_and_gains_the_timing_error() -> None:
    """Inside the tolerance the events still match but late; past it they stop matching at all."""
    inside = events_analysis.skill_rows(
        _detection_entries(shift=16), usable_tau=np.array([8, 9, 10]), raw_per_step=16,
        resamples=64, seed=0,
    )
    outside = events_analysis.skill_rows(
        _detection_entries(shift=200), usable_tau=np.array([8, 9, 10]), raw_per_step=16,
        resamples=64, seed=0,
    )

    assert all(row["hit_rate"] == pytest.approx(1.0) for row in inside)
    assert all(row["lead_time_abs_error_s"] == pytest.approx(4.0) for row in inside)
    assert all(row["hit_rate"] == pytest.approx(0.0) for row in outside)
    assert all(row["false_alarm_rate"] == pytest.approx(1.0) for row in outside)


def test_the_lead_time_axis_is_seconds_ahead_of_the_anchor() -> None:
    rows = events_analysis.skill_rows(
        _detection_entries(), usable_tau=np.array([8, 21]), raw_per_step=16, resamples=32, seed=0
    )

    by_step = {row["horizon_step"]: row["lead_time_s"] for row in rows}
    assert by_step == {8: 36.0, 21: 88.0}


# =============================================================================
# The conditioned coupling, off the per-anchor table
# =============================================================================
def _anchor_frame(*, guids: int, anchors: int, near_every: int, gap: float) -> pd.DataFrame:
    """A per-anchor table whose near-contraction anchors carry a larger gap by construction.

    The noise is zero-mean and independent of the anchor index on purpose. An earlier draft used a
    ramp in the anchor, which made the near-contraction anchors -- being a regular subgrid --
    systematically earlier than the controls, and the "no coupling" case then reported a small but
    entirely one-sided difference that was a property of the fixture rather than of the code.
    """
    rng = np.random.default_rng(11)
    rows = []
    for guid in range(guids):
        for anchor in range(anchors):
            near = (anchor % near_every) == 0
            rows.append(
                {
                    "guid": f"g{guid}",
                    "epoch": -1000.0 * guid,
                    "anchor": anchor,
                    "mc_pred_gap": (gap if near else 0.0) + 0.05 * float(rng.standard_normal()),
                    "kld_per_t": 1.0,
                    CONTRACTION_AGE_COLUMN: 10.0 if near else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_the_control_anchors_are_count_matched_inside_each_recording() -> None:
    frame = _anchor_frame(guids=4, anchors=100, near_every=5, gap=1.0)

    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    counts = split.groupby(["guid", "condition"]).size().unstack("condition")
    assert (counts["event"] == counts["control"]).all()
    assert (counts["event"] == 20).all()


def test_a_conditioned_difference_is_recovered_with_its_interval() -> None:
    """The synthetic coupled case: the gap is larger near a contraction by a known amount."""
    frame = _anchor_frame(guids=6, anchors=100, near_every=5, gap=1.0)
    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    rows, per_recording = events_analysis.conditioned_rows(
        split, pd.Series(dtype=object), resamples=200, seed=0
    )

    pooled = next(row for row in rows if row["metric"] == "pred_gap_mc_nats")
    assert pooled["mean"] == pytest.approx(1.0, abs=0.15)
    assert pooled["ci_lo"] < pooled["mean"] < pooled["ci_hi"]
    assert set(per_recording["metric"]) == {"pred_gap_mc_nats", "source_conditioned_kl_raw"}
    # The second readout is constant, so its conditioned difference must come out at exactly zero:
    # a pipeline that reported a difference there would be reporting the control draw.
    flat = next(row for row in rows if row["metric"] == "source_conditioned_kl_raw")
    assert flat["mean"] == pytest.approx(0.0)


def test_a_no_coupling_case_is_indistinguishable_from_its_control() -> None:
    frame = _anchor_frame(guids=6, anchors=100, near_every=5, gap=0.0)
    split = events_analysis.conditioned_anchors(frame, window_s=120.0, seed=0)

    rows, _frame = events_analysis.conditioned_rows(
        split, pd.Series(dtype=object), resamples=200, seed=0
    )

    pooled = next(row for row in rows if row["metric"] == "pred_gap_mc_nats")
    assert abs(pooled["mean"]) < 0.05
    assert pooled["ci_lo"] < 0.0 < pooled["ci_hi"]


def test_the_guards_fire_on_a_small_population_and_record_a_skip(tmp_path) -> None:
    """A rate over a handful of anchors from two recordings is a description of those two."""
    frame = _anchor_frame(guids=2, anchors=20, near_every=5, gap=1.0)
    collection = Collection(per_sample=pd.DataFrame(), per_anchor=frame)

    record = events_analysis._conditioned_readout(
        collection, tmp_path, window_s=120.0, resamples=64, seed=0
    )

    assert record["record"]["skipped"] is True
    assert str(events_analysis.MIN_EVENT_ANCHORS) in record["record"]["reason"]
    assert record["rows"] == []


# =============================================================================
# The triggered average and its null
# =============================================================================
def _trigger_samples(*, coupled: bool, guids: int = 6, onsets_per: int = 6):
    """Retained-sample entries whose truth blocks dip after every trigger, or do not."""
    decimation, horizon = 16, 480
    anchors = 200
    entries = []
    rng = np.random.default_rng(0)
    for guid in range(guids):
        blocks = {name: rng.normal(0.0, 0.5, size=(anchors, horizon))
                  for name in ("truth", "base", "full")}
        onsets = np.array(
            [decimation * (40 + 20 * index) for index in range(onsets_per)], dtype=np.int64
        )
        if coupled:
            # A dip 60 s into every block that follows a trigger, which is where the response
            # window looks.
            for onset in onsets:
                anchor = int(np.ceil(onset / decimation)) - 1
                blocks["truth"][anchor, 200:280] -= 20.0
        blocks["difference"] = blocks["full"] - blocks["base"]
        entries.append(
            {
                "guid": f"g{guid}",
                "blocks": blocks,
                "truth_trace": blocks["truth"][:, :decimation].reshape(-1),
                "decimation": decimation,
                "onsets": onsets,
                "n_onsets_detected": int(onsets.size),
                # The same bounds `_trigger_entry` computes: a full baseline behind the trigger,
                # and an anchor whose block is carried -- which the last anchor's is.
                "first_trigger": decimation + 120,
                "last_trigger": decimation * anchors,
            }
        )
    return entries


def test_a_coupled_response_leaves_its_null_band_and_an_uncoupled_one_does_not() -> None:
    """The null is the measurement: a minimum over a window is negative on any data at all, so
    the same operator has to be applied to random triggers before a dip means anything."""
    coupled = events_analysis.triggered_average(
        _trigger_samples(coupled=True), rng=np.random.default_rng(0)
    )
    null = events_analysis.triggered_average(
        _trigger_samples(coupled=False), rng=np.random.default_rng(0)
    )

    assert coupled["curves"]["truth"]["dip_z"] < -3.0
    assert abs(null["curves"]["truth"]["dip_z"]) < 3.0
    # The null's own dip is decidedly negative even with no response, which is the selection bias
    # the band exists to measure rather than to assume away.
    assert null["curves"]["truth"]["null_dip_mean"] < 0.0


def test_the_null_is_count_matched_per_recording() -> None:
    samples = _trigger_samples(coupled=True, guids=3, onsets_per=4)
    rng = np.random.default_rng(0)

    drawn = [events_analysis._random_triggers(entry, rng, len(entry["onsets"]))
             for entry in samples]

    assert [len(values) for values in drawn] == [len(entry["onsets"]) for entry in samples]
    assert all(
        (values >= entry["first_trigger"]).all() and (values <= entry["last_trigger"]).all()
        for values, entry in zip(drawn, samples)
    )


def _entry_from_detected(monkeypatch, onsets, *, anchors: int = 200, decimation: int = 16):
    """Build one trigger entry through the real ``_trigger_entry`` with the detector pinned.

    The onset set is the thing under test, so it is supplied rather than detected; everything
    else -- the usable range, the filtering, the counts -- is the shipped code.
    """
    monkeypatch.setattr(
        events_analysis.events,
        "detect_contractions",
        lambda *args, **kwargs: {"onset_raw": np.asarray(onsets, dtype=np.int64)},
    )
    horizon = decimation * 30
    blocks = {name: np.zeros((anchors, horizon)) for name in ("truth", "base", "full")}
    blocks["difference"] = np.zeros((anchors, horizon))
    raw_len = decimation * anchors
    return events_analysis._trigger_entry(
        {"up_raw": [np.zeros(raw_len)]},
        0,
        {"guid": "g0"},
        blocks,
        np.ones(raw_len, dtype=bool),
        decimation=decimation,
    )


def test_only_the_onsets_that_contribute_a_snippet_reach_the_entry(monkeypatch) -> None:
    """The null draws its triggers inside the usable range, so every drawn trigger contributes.

    If the entry carried onsets the observation then discards, count-matching against them would
    average the null over more triggers than the observation -- tightening its band and inflating
    ``dip_z``, the one number that says whether a dip is a response. So the entry carries the
    usable set, and the detected count travels beside it as the honest denominator.
    """
    decimation, anchors = 16, 200
    too_early = decimation  # inside the baseline, so no pre-window
    too_late = decimation * anchors + 1  # past the last anchor's block
    usable = [decimation * 40, decimation * 90]
    entry = _entry_from_detected(
        monkeypatch, [too_early, *usable, too_late], anchors=anchors, decimation=decimation
    )

    assert list(entry["onsets"]) == usable, "an unusable onset must not reach the average"
    assert entry["n_onsets_detected"] == 4, "but the detected count is still reported"
    # The invariant that makes the count-match exact: every trigger on the entry contributes.
    assert events_analysis._snippets(entry, "truth", entry["onsets"]).shape[0] == len(usable)


def test_an_onset_in_the_final_block_is_scored_rather_than_discarded(monkeypatch) -> None:
    """The last anchor's block is carried whole, so a contraction that late has a full block to
    average over. Reserving the block width at the end of the trace instead silently drops every
    contraction in the final two minutes -- while the null, drawn in the same range, keeps them."""
    decimation, anchors = 16, 200
    last = decimation * anchors
    entry = _entry_from_detected(monkeypatch, [last], anchors=anchors, decimation=decimation)

    assert entry["last_trigger"] == last, "the range ends at the last anchor's own block start"
    assert list(entry["onsets"]) == [last]
    assert events_analysis._snippets(entry, "truth", entry["onsets"]).shape[0] == 1


def test_the_triggered_guard_fires_below_the_stated_minimum() -> None:
    record = events_analysis._triggered_or_skip(
        _trigger_samples(coupled=True, guids=2, onsets_per=1), seed=0
    )

    assert record["skipped"] is True
    assert str(events_analysis.MIN_TRIGGERS) in record["reason"]


# =============================================================================
# The whole analysis, and its skip
# =============================================================================
def test_the_waveform_readouts_skip_and_name_the_cap_when_nothing_was_retained(
    tmp_path, evaluated
) -> None:
    """Retention is opt-in, so the ordinary run has no waveform to run a detector on -- and the
    skip has to name the key that would change that rather than merely reporting nothing."""
    from teb_vae.lag_attn_rws.eval.collect import load_collection

    collection = load_collection(evaluated["results_dir"])
    readouts = events_analysis._waveform_readouts(
        collection, collection.record, resamples=64, seed=0
    )

    assert readouts["deceleration"]["skipped"] is True
    assert "caps.waveforms" in readouts["deceleration"]["reason"]
    assert readouts["skill"] == []


def _retained_collection(*, shift: int = 0, guids: int = 4):
    r"""A collection whose retained blocks carry known decelerations, at production geometry.

    ``mu_full`` is the truth itself, so the identity case is exact rather than approximate;
    ``mu_base`` is a flat line, which is what a model that has learned nothing forecasts and what
    makes the false-alarm denominator visible. ``shift`` moves ``mu_full``'s copy of the truth,
    which is how the "a shifted forecast loses hits" direction is exercised through the real
    detector rather than through hand-written detections.
    """
    decimation, horizon, t_valid = 16, 30, 40
    raw_len = decimation * (t_valid + horizon)
    rng = np.random.default_rng(5)
    nadirs = [420, 760]
    per_sample_rows, targets, fulls, bases, ups, weights = [], [], [], [], [], []
    for guid in range(guids):
        trace = 140.0 + 1.0 * rng.standard_normal(raw_len)
        shifted = trace.copy()
        for nadir in nadirs:
            dip = np.clip(1.0 - np.abs(np.arange(raw_len) - nadir) / 60.0, 0.0, 1.0)
            trace -= 25.0 * dip
            moved = np.clip(1.0 - np.abs(np.arange(raw_len) - (nadir + shift)) / 60.0, 0.0, 1.0)
            shifted -= 25.0 * moved
        index = (
            decimation * (np.arange(t_valid)[:, None, None] + 1)
            + decimation * np.arange(horizon)[None, :, None]
            + np.arange(decimation)[None, None, :]
        )
        targets.append(trace[index])
        fulls.append(shifted[index])
        bases.append(np.full_like(trace[index], 140.0))
        ups.append(30.0 + 0.1 * rng.standard_normal(raw_len))
        weights.append(np.ones(raw_len // decimation))
        per_sample_rows.append({"sample_index": guid, "guid": f"g{guid}", "epoch": -1000.0 * guid})

    collection = Collection(
        per_sample=pd.DataFrame(per_sample_rows),
        per_anchor=pd.DataFrame(),
        retained={
            "target": np.stack(targets),
            "mu_full": np.stack(fulls),
            "mu_base": np.stack(bases),
            "up_raw": np.stack(ups),
            "weight": np.stack(weights),
            "waveforms_sample_index": np.arange(guids, dtype=np.int64),
        },
    )
    collection.record = {
        "geometry": {"decimation": decimation, "raw_per_step": decimation, "raw_len": raw_len,
                     "horizon": horizon, "t_valid": t_valid},
        # An identity affine, so `to_bpm` labels the unit honestly without changing the values the
        # detector's absolute 10 bpm threshold is measured against.
        "normalization": {"fhr": {"mean": 0.0, "std": 1.0}},
    }
    return collection


def test_a_forecast_equal_to_the_truth_is_recovered_through_the_real_detector() -> None:
    """The end-to-end known answer: identical blocks, so every true event is found and nothing
    else is -- **within the usable interval**, which the row's own counts are over."""
    collection = _retained_collection(shift=0)

    readouts = events_analysis._waveform_readouts(
        collection, collection.record, resamples=64, seed=0
    )

    full = [row for row in readouts["skill"] if row["branch"] == "full"]
    scored = [row for row in full if row["n_true_events"] > 0]
    assert scored, "the injected decelerations must land inside the usable horizon steps"
    assert all(row["hit_rate"] == pytest.approx(1.0) for row in scored)
    assert all(row["false_alarm_rate"] == pytest.approx(0.0) for row in scored)
    assert all(row["lead_time_abs_error_s"] == pytest.approx(0.0) for row in scored)
    # The flat baseline forecasts no event at all: no hits, and no false alarms either, which is
    # what distinguishes "forecasts nothing" from "forecasts the wrong thing".
    base = [row for row in readouts["skill"] if row["branch"] == "base"]
    assert all(row["n_forecast_events"] == 0 for row in base)


def test_the_per_event_count_is_the_per_anchor_count_divided_by_the_usable_steps() -> None:
    r"""The pseudo-replication assertion. One physiological deceleration is re-detected once per
    anchor whose block contains it, so a per-anchor rate over-counts by that factor; fixing the
    horizon step reduces it to one detection per event per step, exactly."""
    collection = _retained_collection(shift=0)

    record = events_analysis._waveform_readouts(
        collection, collection.record, resamples=32, seed=0
    )["deceleration"]

    factor = record["pseudo_replication_factor"]
    assert factor == len(record["usable_horizon_steps"]) == 14
    assert record["n_true_detections_per_anchor"] == factor * record["n_true_events_per_step"]
    assert record["n_true_detections_per_anchor"] > record["n_true_events_per_step"] * 5


def test_a_forecast_shifted_past_the_tolerance_loses_its_hits() -> None:
    """Through the real detector, so the matching and the detection are exercised together."""
    near = _retained_collection(shift=0)
    far = _retained_collection(shift=200)

    def hits(collection):
        rows = events_analysis._waveform_readouts(
            collection, collection.record, resamples=32, seed=0
        )["skill"]
        scored = [row for row in rows if row["branch"] == "full" and row["n_true_events"] > 0]
        return float(np.mean([row["hit_rate"] for row in scored]))

    assert hits(near) == pytest.approx(1.0)
    assert hits(far) < 0.2


def test_the_analysis_runs_end_to_end_against_a_finished_run(tmp_path, evaluated) -> None:
    """Both halves, through the registry's own entry point, on the real run's tables."""
    from teb_vae.lag_attn_rws.eval.analyses import AnalysisContext
    from teb_vae.lag_attn_rws.eval.collect import load_collection

    collection = load_collection(evaluated["results_dir"])
    result = events_analysis.run_events_analysis(
        AnalysisContext(collection=collection),
        eval_config={"seed": 0, "bootstrap_resamples": 64, "event_lag_window_s": 120.0},
        output_dir=tmp_path,
    )

    for key in ("n_samples", "composition", "plan"):
        assert key in result
    directory = tmp_path / events_analysis.ANALYSIS_DIRNAME
    assert (directory / figure_filename(events_analysis.DECELERATION_FIGURE)).is_file()
    assert (directory / figure_filename(events_analysis.TRIGGERED_FIGURE)).is_file()
    assert (directory / figure_filename(events_analysis.CONDITIONED_FIGURE)).is_file()


def test_a_real_run_over_the_event_shards_recovers_the_injected_events(event_evaluated) -> None:
    """The whole chain, through the shipped entry point: shards with known events, the loader,
    the collection pass's contraction column, the retention cap, and both readouts."""
    result = event_evaluated["summary"]["results"]["events"]
    truth = injected_event_indices()["deceleration_nadir"]
    gap_lo = (_TRIMMED_STEPS // 2) * _DECIMATION
    gap_hi = gap_lo + MULTI_CLASS_GAP_STEPS * _DECIMATION
    recoverable = int(((truth < gap_lo - 60) | (truth >= gap_hi + 60)).sum())

    deceleration = result["deceleration"]
    assert "skipped" not in deceleration
    assert deceleration["unit"] == "bpm"
    assert deceleration["n_true_events_per_step"] == recoverable * result["n_samples"]
    assert deceleration["pseudo_replication_factor"] == len(
        deceleration["usable_horizon_steps"]
    )
    # The contraction column reached the anchor table, and both event guards cleared on it.
    assert result["triggered"].get("skipped") is not True
    assert result["conditioned"].get("skipped") is not True
    assert result["composition"]["n_event_anchors"] >= events_analysis.MIN_EVENT_ANCHORS


def test_the_null_band_brackets_the_null_mean_at_every_point() -> None:
    """The figure's one load-bearing property: the band is what says whether a dip is a finding."""
    from matplotlib.collections import PolyCollection

    from teb_vae.lag_attn.eval import figures as shared_figures

    record = events_analysis.triggered_average(
        _trigger_samples(coupled=True, guids=4, onsets_per=4), rng=np.random.default_rng(0)
    )
    figure = events_analysis.build_triggered_figure(record)
    try:
        bands = [
            artist for axis in figure.axes for artist in axis.collections
            if isinstance(artist, PolyCollection)
        ]
        assert bands, "every panel draws a fill_between band"
        for name, block in record["curves"].items():
            low = np.asarray(block["null_lo"], dtype=np.float64)
            high = np.asarray(block["null_hi"], dtype=np.float64)
            mean = np.asarray(block["null_mean"], dtype=np.float64)
            assert np.all(low <= mean + 1e-12) and np.all(mean <= high + 1e-12), name
        # Seconds, not horizon steps: the axis spans the block's own 120 s.
        assert figure.axes[0].get_xlabel().endswith("(s)")
    finally:
        shared_figures.plt.close(figure)
