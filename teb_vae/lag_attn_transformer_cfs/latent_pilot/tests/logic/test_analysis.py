r"""Trajectory bins, the paired windows, the geometry summaries and the one shared projection.

Hand-checkable arrays throughout. Nothing here builds a model, runs a forward or opens a shard: a
bin summary is a regrouping of an anchor frame, a centroid is a mean, an effective rank is an
entropy, and the projection is an eigendecomposition of a small weighted covariance. Only the
classifier-scoring helper needs a module, and the two tests that exercise it use a two-line stand-in
rather than the real one.

Importing the analysis module pulls the model-facing half of this package with it, which is the one
weight this file carries; it constructs nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze, data
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError


class _Extraction:
    """The two attributes the analyses read off a real extraction."""

    def __init__(self, frame, arrays):
        self.frame, self.arrays = frame, arrays

    @property
    def retained(self):
        return data.retained(self.frame)


class _Head:
    """A stand-in classifier: the first coordinate, unchanged.

    The real one is an ``nn.Module`` and constructing one is outside this subset; what these tests
    need from it is only that the *same* head is applied to every bin, which a callable makes just
    as visible.
    """

    def eval(self):
        return self

    def __call__(self, values):
        import torch

        return torch.as_tensor(values)[:, 0]


def _extraction(rows, *, split="train", d_z=2):
    """An anchor frame from ``(guid, epoch, anchor, hours, value)`` tuples."""
    frame = pd.DataFrame([
        {
            data.SPLIT_COLUMN: split,
            data.GUID_COLUMN: guid,
            data.EPOCH_COLUMN: float(epoch),
            data.ANCHOR_COLUMN: int(anchor),
            data.HOURS_COLUMN: float(hours),
            data.EXCLUSION_COLUMN: "",
            data.ROW_COLUMN: index,
        }
        for index, (guid, epoch, anchor, hours, _value) in enumerate(rows)
    ])
    values = np.stack([
        np.full(d_z, float(value)) for (_g, _e, _a, _h, value) in rows
    ])
    return _Extraction(frame, {"mu_post": values, "mu_prior": values * -1.0})


def _recordings(outcomes, *, split="train", classes=None):
    """A recording table carrying what the analyses read."""
    return pd.DataFrame([
        {
            data.GUID_COLUMN: guid,
            data.SPLIT_COLUMN: split,
            data.OUTCOME_COLUMN: outcome,
            labels.CLASS_COLUMN: (classes or {}).get(guid, "healthy" if outcome == 0 else "acidosis"),
            data.EXCLUSION_COLUMN: "",
            "eligible": True,
        }
        for guid, outcome in outcomes.items()
    ])


# =============================================================================
# Trajectory bins
# =============================================================================
def test_a_recording_gets_one_row_per_occupied_bin_and_none_for_the_rest():
    """Absence is preserved: a bin with no anchor has no observation, and none is invented."""
    extraction = _extraction([
        ("ONE", -3000.0, 0, 0.25, 1.0),
        ("ONE", -3000.0, 1, 0.75, 3.0),
        ("ONE", -9000.0, 0, 2.75, 5.0),
    ])

    bins, values = analyze.bin_summaries(
        extraction,
        _recordings({"ONE": 1}),
        split="train",
        bin_hours=0.5,
        preservation_hours=3.0,
    )

    assert sorted(bins[data.BIN_COLUMN].tolist()) == [0, 1, 5]
    assert len(values) == 3
    assert bins[data.OUTCOME_COLUMN].tolist() == [1, 1, 1]
    assert set(bins[labels.CLASS_COLUMN]) == {"acidosis"}


def test_bin_membership_follows_the_same_half_open_rule_as_every_window():
    extraction = _extraction([
        ("ONE", -3000.0, 0, 0.5, 1.0),   # closes bin 0
        ("ONE", -3000.0, 1, 1.0, 2.0),   # closes bin 1
        ("ONE", -3000.0, 2, 3.0, 3.0),   # closes bin 5
    ])

    bins, _values = analyze.bin_summaries(
        extraction, _recordings({"ONE": 0}), split="train", bin_hours=0.5,
        preservation_hours=3.0,
    )

    assert sorted(bins[data.BIN_COLUMN].tolist()) == [0, 1, 5]


def test_a_bin_summary_averages_anchors_then_segments():
    """Three anchors in one segment and one in another: the segments weigh the same."""
    extraction = _extraction([
        ("ONE", -3000.0, 0, 0.25, 0.0),
        ("ONE", -3000.0, 1, 0.25, 0.0),
        ("ONE", -3000.0, 2, 0.25, 0.0),
        ("ONE", -3600.0, 0, 0.25, 4.0),
    ])

    _bins, values = analyze.bin_summaries(
        extraction, _recordings({"ONE": 1}), split="train", bin_hours=0.5,
        preservation_hours=3.0,
    )

    assert float(values[0, 0]) == pytest.approx(2.0)


def test_bins_are_refused_for_a_split_the_extraction_does_not_hold():
    with pytest.raises(PilotConfigError, match="requested for 'val'"):
        analyze.bin_summaries(
            _extraction([("ONE", -3000.0, 0, 0.5, 1.0)], split="train"),
            _recordings({"ONE": 1}, split="val"),
            split="val",
            bin_hours=0.5,
            preservation_hours=3.0,
        )


def test_the_supervised_bins_are_the_ones_inside_the_final_hour():
    """Everything else is the same head applied outside the window it was fitted on."""
    assert analyze.supervised_bins(bin_hours=0.5, supervised_hours=1.0) == [0, 1]
    assert analyze.supervised_bins(bin_hours=0.5, supervised_hours=3.0) == [0, 1, 2, 3, 4, 5]


def test_one_frozen_head_scores_every_bin():
    extraction = _extraction([
        ("A", -3000.0, 0, 0.25, 1.0),
        ("A", -9000.0, 0, 2.75, 9.0),
    ])
    bins, values = analyze.bin_summaries(
        extraction, _recordings({"A": 1}), split="train", bin_hours=0.5,
        preservation_hours=3.0,
    )

    scored = analyze.score_frame(bins, values, _Head())

    assert sorted(scored[analyze.SCORE_COLUMN].tolist()) == [1.0, 9.0]


# =============================================================================
# Group bands
# =============================================================================
def _scored_bins(n=6):
    """A scored bin table with two groups and a bin each side of the supervised edge."""
    rows = []
    for index in range(n):
        for outcome in (0, 1):
            for bin_index in (0, 3):
                rows.append({
                    data.GUID_COLUMN: f"{outcome}-{index}",
                    data.OUTCOME_COLUMN: outcome,
                    data.BIN_COLUMN: bin_index,
                    data.BIN_LABEL_COLUMN: f"bin{bin_index}",
                    analyze.SCORE_COLUMN: float(outcome) + 0.1 * index,
                })
    return pd.DataFrame(rows)


def test_a_band_is_reported_per_group_and_bin_with_the_count_it_rests_on():
    bands = analyze.group_bands(
        _scored_bins(), resamples=200, seed=42, supervised=[0, 1]
    )

    assert len(bands) == 4
    assert set(bands["n_recordings"]) == {6}
    # Text, so the outcome grouping and the class grouping can share one table -- and one parquet
    # file, which is what the evaluate stage writes them to.
    assert bands["group"].map(type).eq(str).all()
    assert bands[bands[data.BIN_COLUMN] == 0]["supervised_window"].all()
    assert not bands[bands[data.BIN_COLUMN] == 3]["supervised_window"].any()
    for _index, row in bands.iterrows():
        assert row["lo"] <= row["mean"] <= row["hi"]


def test_a_bin_too_small_to_resample_reports_its_count_and_no_band():
    bands = analyze.group_bands(_scored_bins(n=2), resamples=200, seed=42)

    assert (bands["n_recordings"] == 2).all()
    assert bands["lo"].isna().all()
    assert bands["band_note"].str.len().gt(0).all()


def test_the_bands_are_reproducible_from_their_seed():
    first = analyze.group_bands(_scored_bins(), resamples=200, seed=42)
    again = analyze.group_bands(_scored_bins(), resamples=200, seed=42)

    assert first.equals(again)


# =============================================================================
# The paired early/late comparison
# =============================================================================
def _paired_extraction():
    """Four recordings, two observed in both windows and two in the late window only."""
    rows = []
    for index, guid in enumerate(("BOTH-0", "BOTH-1")):
        rows.append((guid, -9000.0, 0, 2.5, float(index)))
        rows.append((guid, -3000.0, 0, 0.5, float(index) + 2.0))
    for index, guid in enumerate(("LATE-0", "LATE-1")):
        rows.append((guid, -3000.0, 0, 0.5, float(index)))
    return _extraction(rows)


def test_only_recordings_observed_in_both_windows_are_paired():
    extraction = _paired_extraction()
    recordings = _recordings({"BOTH-0": 0, "BOTH-1": 1, "LATE-0": 0, "LATE-1": 1})

    paired = analyze.window_scores(
        extraction, recordings, _Head(), split="train", early=(2.0, 3.0), late=(0.0, 1.0)
    )

    assert sorted(paired[data.GUID_COLUMN]) == ["BOTH-0", "BOTH-1"]
    assert paired["delta"].tolist() == [2.0, 2.0]


def test_the_two_windows_are_reduced_by_the_same_order():
    """Their difference must not be an artefact of two aggregation rules."""
    extraction = _extraction([
        ("ONE", -9000.0, 0, 2.5, 0.0),
        ("ONE", -9000.0, 1, 2.5, 4.0),
        ("ONE", -3000.0, 0, 0.5, 10.0),
    ])

    paired = analyze.window_scores(
        extraction, _recordings({"ONE": 1}), _Head(), split="train",
        early=(2.0, 3.0), late=(0.0, 1.0),
    )

    # The early window's two anchors are one segment, so its score is their mean.
    assert float(paired[f"{analyze.SCORE_COLUMN}_early"].iloc[0]) == pytest.approx(2.0)
    assert float(paired["delta"].iloc[0]) == pytest.approx(8.0)


def test_the_contrast_reports_each_group_and_their_difference_with_counts():
    paired = pd.DataFrame({
        data.GUID_COLUMN: [f"R-{index}" for index in range(8)],
        data.OUTCOME_COLUMN: [0, 0, 0, 0, 1, 1, 1, 1],
        "delta": [0.0, 0.1, -0.1, 0.0, 1.0, 1.1, 0.9, 1.0],
    })

    record = analyze.paired_contrast(paired, resamples=200, seed=42)

    assert record["groups"]["healthy"]["point"] == pytest.approx(0.0)
    assert record["groups"]["adverse"]["point"] == pytest.approx(1.0)
    assert record["difference"]["point"] == pytest.approx(1.0)
    assert record["difference"]["n_left"] == 4
    assert record["n_paired_recordings"] == 8


def test_the_contrast_states_what_an_increase_is_and_is_not_evidence_of():
    paired = pd.DataFrame({
        data.GUID_COLUMN: ["R-0", "R-1"],
        data.OUTCOME_COLUMN: [0, 1],
        "delta": [0.0, 1.0],
    })

    record = analyze.paired_contrast(paired, resamples=100, seed=42)

    assert "not evidence of physiological worsening" in record["interpretation"]
    assert "neither coverage selection nor confounding" in record["limitation"]
    # Two recordings cannot support an interval, and none is reported.
    assert np.isnan(record["difference"]["lo"])


# =============================================================================
# Geometry
# =============================================================================
def test_centroids_are_the_class_means_and_a_missing_class_is_refused():
    values = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 8.0]])

    centroids = analyze.class_centroids(values, [0, 0, 1, 1])

    assert centroids[0].tolist() == [1.0, 0.0]
    assert centroids[1].tolist() == [0.0, 6.0]
    with pytest.raises(PilotConfigError, match="centroid is undefined"):
        analyze.class_centroids(values, [0, 0, 0, 0])


def test_the_nearest_centroid_score_is_positive_toward_the_adverse_centroid():
    centroids = {0: np.array([0.0, 0.0]), 1: np.array([4.0, 0.0])}

    scores = analyze.nearest_centroid_scores(
        np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 0.0]]), centroids
    )

    assert scores[0] == pytest.approx(-4.0)
    assert scores[1] == pytest.approx(4.0)
    assert scores[2] == pytest.approx(0.0)


def test_the_effective_rank_is_one_for_a_single_direction_and_d_for_a_flat_spectrum():
    assert analyze.effective_rank([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert analyze.effective_rank([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert analyze.effective_rank([2.0, 2.0]) == pytest.approx(2.0)


def test_the_zero_variance_convention_is_zero_and_not_one():
    """Nothing varying is no direction carrying anything, not one direction carrying everything."""
    assert analyze.effective_rank([0.0, 0.0, 0.0]) == 0.0
    assert analyze.effective_rank([]) == 0.0
    # Numerical negatives from an eigendecomposition are not directions.
    assert analyze.effective_rank([-1e-18, -2e-18]) == 0.0


def test_the_covariance_summary_reads_the_full_space_and_says_when_it_cannot():
    values = np.array([[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0], [-3.0, 0.0]])

    summary = analyze.covariance_summary(values)

    assert summary["d_z"] == 2
    assert summary["effective_rank"] == pytest.approx(1.0)
    assert summary["leading_share"] == pytest.approx(1.0)
    assert "note" in analyze.covariance_summary(values[:1])


def test_movement_is_measured_in_training_standard_deviations():
    before = np.array([[0.0, 0.0], [0.0, 0.0]])
    after = np.array([[3.0, 4.0], [0.0, 0.0]])

    summary = analyze.movement_summary(before, after, scale=np.array([1.0, 1.0]))
    scaled = analyze.movement_summary(before, after, scale=np.array([2.0, 2.0]))

    assert summary["max"] == pytest.approx(5.0)
    assert summary["median"] == pytest.approx(2.5)
    assert scaled["max"] == pytest.approx(2.5)
    assert "training standard deviations" in summary["units"]


def test_movement_between_two_populations_is_refused():
    with pytest.raises(PilotConfigError, match="same recordings"):
        analyze.movement_summary(
            np.zeros((3, 2)), np.zeros((2, 2)), scale=np.ones(2)
        )


# =============================================================================
# The shared projection
# =============================================================================
def _bin_frame(occupancy, *, split="train"):
    """A training bin table with the given number of bins per recording."""
    rows = []
    for guid, count in occupancy.items():
        for index in range(count):
            rows.append({
                data.SPLIT_COLUMN: split,
                data.GUID_COLUMN: guid,
                data.BIN_COLUMN: index,
            })
    return pd.DataFrame(rows)


def test_every_recording_weighs_the_same_however_many_bins_it_occupied():
    frame = _bin_frame({"DENSE": 6, "SPARSE": 1})

    weights = analyze.projection_weights(frame)

    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[:6].sum()) == pytest.approx(0.5)
    assert float(weights[6]) == pytest.approx(0.5)


def test_the_map_is_fitted_once_on_both_versions_and_applied_unchanged():
    frame = _bin_frame({"A": 2, "B": 2, "C": 2})
    before = np.array([[float(index), 0.0] for index in range(6)])
    after = before + np.array([0.0, 0.5])

    projection = analyze.fit_projection(
        {"pretrained": (frame, before), "adapted": (frame, after)}
    )

    assert projection.components.shape == (2, 2)
    assert projection.record["versions"] == ["adapted", "pretrained"]
    assert projection.record["labels_used"] is False
    # The leading direction is the one the data actually spreads along.
    assert abs(projection.components[0][0]) == pytest.approx(1.0, abs=1e-6)
    assert projection.explained_variance_ratio[0] > projection.explained_variance_ratio[1]


def test_the_axes_are_orthonormal_and_deterministically_oriented():
    """A flipped panel between two runs would read as movement rather than as a convention."""
    frame = _bin_frame({"A": 2, "B": 2, "C": 2})
    values = np.array([[-float(index), -0.5 * float(index)] for index in range(6)])

    first = analyze.fit_projection({"only": (frame, values)})
    again = analyze.fit_projection({"only": (frame, values)})

    assert np.allclose(first.components, again.components)
    assert np.allclose(first.components @ first.components.T, np.eye(2), atol=1e-9)
    for axis in first.components:
        assert axis[int(np.argmax(np.abs(axis)))] > 0.0


def test_the_projection_centres_on_its_own_weighted_mean():
    frame = _bin_frame({"A": 1, "B": 1, "C": 1, "D": 1})
    values = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

    projection = analyze.fit_projection({"only": (frame, values)})

    assert np.allclose(projection.mean, [1.0, 1.0])
    assert np.allclose(projection.transform(projection.mean[None, :]), 0.0, atol=1e-9)


def test_a_projection_fitted_on_anything_but_training_data_is_refused():
    frame = _bin_frame({"A": 2, "B": 2, "C": 2}, split="test")
    values = np.zeros((6, 2))

    with pytest.raises(PilotConfigError, match="fitted on training data alone"):
        analyze.fit_projection({"only": (frame, values)})


def test_two_versions_of_different_widths_cannot_share_one_map():
    frame = _bin_frame({"A": 2, "B": 2, "C": 2})

    with pytest.raises(PilotConfigError, match="two latent widths"):
        analyze.fit_projection(
            {"a": (frame, np.zeros((6, 2))), "b": (frame, np.zeros((6, 3)))}
        )


def test_a_projection_needs_something_to_fit_on():
    with pytest.raises(PilotConfigError, match="at least one model version"):
        analyze.fit_projection({})
    with pytest.raises(PilotConfigError, match="cannot support"):
        analyze.fit_projection({"only": (_bin_frame({"A": 2}), np.zeros((2, 2)))})


def test_the_map_round_trips_through_a_run_directory(tmp_path):
    frame = _bin_frame({"A": 2, "B": 2, "C": 2})
    values = np.array([[float(index), float(index) ** 2] for index in range(6)])
    projection = analyze.fit_projection({"only": (frame, values)})

    analyze.save_projection(projection, tmp_path)
    reloaded = analyze.load_projection(tmp_path)

    assert np.allclose(reloaded.transform(values), projection.transform(values))
    assert reloaded.record == projection.record


def test_a_missing_map_is_never_silently_refitted(tmp_path):
    with pytest.raises(FileNotFoundError, match="fitted once"):
        analyze.load_projection(tmp_path)
