r"""The per-recording aggregation when a column is present on some batches only.

The permute control runs per batch, and only where a cross-recording pairing exists: a batch that
one recording holds more than half of -- typically the trailing partial batch of a split -- records
the arm as skipped and carries no ``nll_permute`` column. Every other column is on every batch.
The aggregation must average such a column over the segments that carried it, and the summaries
built from it must read only the recordings that hold it, rather than crash on the first recording
whose batches disagree or quietly divide the column by segments that never scored it.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.eval import collect


def _record(guids: List[str], columns: Dict[str, List[float]], anchors: List[float]) -> Dict[str, Any]:
    """One per-batch record with only the fields the aggregation reads."""
    return {
        "guids": list(guids),
        "columns": {name: torch.tensor(values, dtype=torch.float64) for name, values in columns.items()},
        "curves": {},
        "n_anchors": torch.tensor(anchors, dtype=torch.float64),
    }


@pytest.fixture
def records() -> List[Dict[str, Any]]:
    """Recording ``a`` first appears in a batch without a partner, then in one with.

    That order is the one that raised: the bucket was built from the first batch's column names
    and the second batch's permute value had nowhere to go. Recording ``b`` appears only in the
    paired batch, and ``c`` only in the unpaired one, so it has no permute value at all.
    """
    return [
        # The unpaired batch: no ``nll_permute`` column.
        _record(
            ["a", "c"],
            {"nll_full": [1.0, 5.0], "nll_base": [2.0, 6.0]},
            anchors=[10.0, 30.0],
        ),
        # The paired batch: every column, including the permute arm.
        _record(
            ["a", "b"],
            {"nll_full": [3.0, 7.0], "nll_base": [4.0, 8.0], "nll_permute": [9.0, 11.0]},
            anchors=[20.0, 40.0],
        ),
    ]


def test_a_column_is_averaged_over_the_segments_that_carried_it(records) -> None:
    """A partial column divides by the segments that carried it; a recording with none lacks it."""
    per_recording, exposure, _ = collect.aggregate_by_recording(records)
    # Both segments of ``a`` carried the full column; only the second carried the permute column.
    assert per_recording["a"]["nll_full"] == pytest.approx(2.0)
    assert per_recording["a"]["nll_permute"] == pytest.approx(9.0)
    # A recording none of whose segments carried the column is absent from it, not zero.
    assert "nll_permute" not in per_recording["c"]
    assert per_recording["b"]["nll_permute"] == pytest.approx(11.0)
    # The exposure counts are the all-columns population.
    assert exposure["a"] == {"n_segments": 2.0, "n_scored_anchors": 30.0}
    assert exposure["c"] == {"n_segments": 1.0, "n_scored_anchors": 30.0}


def test_the_headline_bootstraps_a_partial_column_over_the_recordings_that_hold_it(records) -> None:
    """The headline takes the union of columns and reads each over the recordings holding it."""
    per_recording, _, _ = collect.aggregate_by_recording(records)
    headline = collect.arm_scores_block(per_recording, resamples=50, seed=0)
    # Every column is present; the partial one counts only the recordings that hold it. The
    # point itself is NaN here by the bootstrap's own minimum-group rule, so only ``n`` is read.
    assert set(headline) == {"nll_full", "nll_base", "nll_permute"}
    assert headline["nll_full"]["n"] == 3
    assert headline["nll_permute"]["n"] == 2


def test_the_anchor_weighted_mean_divides_each_column_by_its_own_anchors(records) -> None:
    """The anchor-weighted permute mean uses only the anchors of the batches that scored it."""
    weighted = collect.anchor_weighted(records)
    # The full column is weighted over every segment of both batches.
    assert weighted["nll_full"] == pytest.approx(5.0)
    # The permute column is weighted over the paired batch only, not over both batches.
    assert weighted["nll_permute"] == pytest.approx(620.0 / 60.0)


def test_the_anchor_weighted_estimand_weights_a_recordings_anchors_and_not_its_segments(records) -> None:
    """Recording ``a`` scored 10 anchors at 1.0 and 20 at 3.0: the equal-segment mean is 2.0 and
    the anchor-weighted one (10 + 60) / 30; a partial column divides by the anchors of the
    segments that carried it."""
    equal, _, _ = collect.aggregate_by_recording(records)
    weighted = collect.aggregate_by_recording_anchor_weighted(records)
    assert equal["a"]["nll_full"] == pytest.approx(2.0)
    assert weighted["a"]["nll_full"] == pytest.approx(70.0 / 30.0)
    assert weighted["a"]["nll_permute"] == pytest.approx(9.0)
    assert "nll_permute" not in weighted["c"]
    # A recording whose segments score equal anchor counts agrees under both estimands.
    assert weighted["b"]["nll_full"] == pytest.approx(equal["b"]["nll_full"])


def test_a_curve_row_blanked_to_nan_is_a_segment_that_did_not_carry_the_curve() -> None:
    """The single-lag profile blanks the rows its cohort did not admit; the aggregation reads
    the admitted rows only, rather than averaging a NaN in."""
    record = _record(["a", "a"], {"nll_full": [1.0, 3.0]}, anchors=[10.0, 10.0])
    record["curves"] = {
        "lag_margin": torch.tensor([[1.0, 2.0], [float("nan"), float("nan")]], dtype=torch.float64)
    }
    _, _, curves = collect.aggregate_by_recording([record])
    assert curves["lag_margin"]["a"].tolist() == [1.0, 2.0]


class _Batch(dict):
    """A batch that also answers attribute access, as the data module's does."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _labelled_batch(guids, classes):
    """A batch whose weight-scaled target encodes one class code per sample."""
    codes = {"healthy": 1, "acidosis": 2, "hie": 3, None: 0}
    steps = 4
    weight = torch.ones(len(guids), steps)
    target = torch.tensor([[float(codes[name])] * steps for name in classes])
    # ``fhr_st`` is the field the batch size is read from, as on a real batch.
    return _Batch(
        guid=list(guids), weight=weight, target=target, fhr_st=torch.zeros(len(guids), steps, 2),
        epoch=torch.tensor([-1200.0 * (i + 1) for i in range(len(guids))]),
        source_file_basename=["healthy_no_bg_no_cs.hdf5"] * len(guids),
    )


def test_the_profile_cohort_admits_under_the_cap_and_a_per_class_quota() -> None:
    """Cap 4 over three declared classes gives each class two slots: a batch of four healthy
    segments fills two, the rest wait for the other classes, and the identities are kept."""
    cohort = collect.LagProfileCohort(4, ["healthy", "acidosis", "hie"])
    first = cohort.admit(_labelled_batch(["g1", "g2", "g3", "g4"], ["healthy"] * 4))
    assert first.tolist() == [True, True, False, False]
    second = cohort.admit(_labelled_batch(["g5", "g6", "g7"], ["acidosis", "healthy", "hie"]))
    assert second.tolist() == [True, False, True]
    # Full: nothing more is admitted, and the caller is told to skip the single-lag arms.
    assert cohort.admit(_labelled_batch(["g8"], ["acidosis"])) is None
    record = cohort.record()
    assert record["composition"] == {"healthy": 2, "acidosis": 1, "hie": 1}
    assert record["quota"] == {"healthy": 2, "acidosis": 2, "hie": 2}
    assert [row["guid"] for row in record["segments"]] == ["g1", "g2", "g5", "g7"]
    assert record["segments"][0]["epoch"] == -1200.0
    assert record["segments"][0]["clinical_class"] == "healthy"


def test_the_profile_cohort_falls_back_to_the_cap_alone_on_an_unlabelled_split() -> None:
    """No declared class means no quota: the cap is the only rule, and an unset cap admits
    nothing at all."""
    cohort = collect.LagProfileCohort(3, [])
    admitted = cohort.admit(_labelled_batch(["g1", "g2", "g3", "g4"], [None] * 4))
    assert admitted.tolist() == [True, True, True, False]
    assert cohort.record()["composition"] == {"unlabelled": 3}
    assert "no class quota" in cohort.record()["selection"]
    assert collect.LagProfileCohort(None, ["healthy"]).admit(_labelled_batch(["g"], ["healthy"])) is None


def test_shard_classes_reads_the_canonical_names_and_ignores_the_rest() -> None:
    """Classes come from the canonical subgroup names; a pretraining shard declares none."""

    class _Loader:
        class dataset:
            paths = [
                "/x/healthy_no_bg_cs.hdf5", "/x/hie_cs.hdf5", "/x/healthy_bg_cs.hdf5",
                "/x/train_dataset_cs.hdf5",
            ]

    assert collect.shard_classes(_Loader()) == ["healthy", "hie"]
    assert collect.shard_classes(object()) == []
