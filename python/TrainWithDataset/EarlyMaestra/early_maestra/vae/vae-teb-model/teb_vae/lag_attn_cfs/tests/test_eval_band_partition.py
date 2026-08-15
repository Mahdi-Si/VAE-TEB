r"""The channel map: what each declared channel is, and what the kept axis a readout is on holds.

Two axes, and conflating them is the whole correctness risk this analysis carries. The band map is
over the $102$ **declared** target channels; every per-channel readout the collection pass writes is
over the $98$ **kept** ones. On this dataset the four the budget drops happen to be the trailing
four, so a positional join would look right on the first 98 rows and be wrong on any dataset whose
survivors are not a prefix -- which is exactly the failure no test would catch. The kept-axis map is
therefore emitted as its own file, carrying each kept channel's declared index, and the tests below
assert that the projection goes through it rather than through a slice.

Per the fixture rule in ``test_eval_fixtures.py``: everything asserted here is schema, shape,
counts, identities and coverage. Nothing asserts what any band's skill *is*.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget
from teb_vae.lag_attn_cfs.eval._reuse import band_partition as shared
from teb_vae.lag_attn_cfs.eval.analyses import band_partition as analysis

from .conftest import (
    CAUSAL_C_U,
    CAUSAL_C_Y,
    CAUSAL_PH_WIDTH,
    CAUSAL_ST_WIDTH,
    causal_config,
)

#: What the shipped budget resolves to on causal shards: 98 of the 102 declared target channels.
_KEPT_TARGET_CHANNELS = 98

#: The four the budget drops, in rebased coordinates, and the declared indices they sit at.
_DROPPED_WARMUPS = (162.0, 194.0, 233.0, 278.0)
_DROPPED_CHANNELS = (32, 33, 34, 35)

#: The five coverage counts of the band-resolved readout, measured on this fixture. Emitted as five
#: numbers rather than one ratio because the declared and scored numerators coincide at 95 by
#: arithmetic accident -- $102 - 7 = 98 - 3$ -- and "95 of 102" would imply the analysis banded
#: channels the decoder never emitted.
_COVERAGE = {
    "declared_total": 102,
    "dropped_declared": 4,
    "kept_total": 98,
    "known_kept": 95,
    "unknown_kept": 3,
}

#: The target scattering block's own breakdown: one order-0 lowpass, 28 order-1 filters some
#: selected phase pair named, and 7 that none did.
_TARGET_SCATTERING_WITHOUT_FREQUENCY = 7

#: The decimated steps ``trim_minutes: 1.0`` discards from each end. Restated rather than imported
#: so a rebase that silently stopped happening cannot be masked by the same expression on both
#: sides of the assertion.
_TRIM_STEPS = 15


# =================================================================================================
# Stubs: an analysis context with no model, which is the path this must work on
# =================================================================================================
class _StubCollection:
    """The two attributes this analysis reads off a collection, and nothing else."""

    def __init__(self, record: Dict[str, Any]) -> None:
        self.record = record


class _StubContext:
    """An :class:`AnalysisContext` with no task and no loader, as an offline re-run has."""

    def __init__(self, config: Dict[str, Any], record: Dict[str, Any]) -> None:
        self.config = config
        self.collection = _StubCollection(record)
        self.task = None
        self.loader = None


def _resolved_target(cohort_shards: Sequence[str]):
    """Resolve the shipped budget against every generated shard and return the target stream."""
    config = causal_config()
    config["dataset_config"]["vae_train_datasets"] = list(cohort_shards)
    config["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    resolved = resolve_warmup_budget(config)
    assert resolved is not None
    return resolved.target


def _context(
    cohort_shards: Sequence[str], keep_index: Optional[Sequence[int]], kept_width: Optional[int]
) -> _StubContext:
    """Build the analysis context a run would pass, over the generated shards."""
    config = causal_config()
    config["dataset_config"]["vae_test_datasets"] = list(cohort_shards)
    geometry: Dict[str, Any] = {
        "target_declared_width": CAUSAL_C_Y,
        "target_kept_width": kept_width,
        "target_keep_index": None if keep_index is None else [int(v) for v in keep_index],
    }
    return _StubContext(config, {"geometry": geometry})


@pytest.fixture(scope="module")
def emitted(cohort_shards, tmp_path_factory):
    """One run of the analysis at the shipped budget; every assertion below reads its output."""
    target = _resolved_target(cohort_shards)
    output_dir = tmp_path_factory.mktemp("band_partition")
    context = _context(cohort_shards, target.keep_index, target.kept_width)
    record = analysis.run_band_partition_analysis(
        context, eval_config={}, output_dir=output_dir, probe=None
    )
    assert record["skipped"] is False, record
    return {
        "record": record,
        "dir": Path(output_dir),
        "declared": pd.read_csv(Path(output_dir) / analysis.CHANNEL_MAP_FILENAME),
        "kept": pd.read_csv(Path(output_dir) / analysis.KEPT_CHANNEL_MAP_FILENAME),
    }


# =================================================================================================
# The declared axis
# =================================================================================================
def test_the_channel_map_is_one_row_per_channel_of_both_streams(emitted) -> None:
    r"""$102 + 51 = 153$ rows, laid out as the model receives them, with a ``stream`` column.

    The column rather than a remembered offset: a reader holding a channel index is asking which of
    the two model inputs it belongs to, and the two streams' indices both start at zero.
    """
    frame = emitted["declared"]

    assert len(frame) == CAUSAL_C_Y + CAUSAL_C_U == 153
    assert list(frame.columns[:2]) == ["stream", "channel"]
    counts = frame["stream"].value_counts().to_dict()
    assert counts == {"target": CAUSAL_C_Y, "source": CAUSAL_C_U}
    for stream, width in (("target", CAUSAL_C_Y), ("source", CAUSAL_C_U)):
        rows = frame[frame["stream"] == stream]
        assert list(rows["channel"]) == list(range(width))
        assert list(rows["block"]) == ["scattering"] * CAUSAL_ST_WIDTH + ["phase"] * (
            width - CAUSAL_ST_WIDTH
        )


def test_the_four_dropped_channels_are_present_with_their_rebased_warm_ups(emitted) -> None:
    r"""$W' = 162, 194, 233, 278$ at ``kept = False``, and every other target channel kept.

    Present rather than filtered out: a map that carried only the survivors could not say what the
    budget removed, and the four are the whole reason the two axes differ.
    """
    target = emitted["declared"][emitted["declared"]["stream"] == "target"]
    dropped = target[~target["kept"]]

    assert tuple(dropped["channel"]) == _DROPPED_CHANNELS
    assert tuple(dropped["causal_warmup_steps"]) == _DROPPED_WARMUPS
    assert int(target["kept"].sum()) == _KEPT_TARGET_CHANNELS


def test_the_warm_up_column_is_rebased_by_the_loaders_own_trim(emitted, cohort_shards) -> None:
    """Stored minus ``trim_steps``, floored at zero -- not the stored vector.

    The un-rebased number would misplace every channel's validity boundary by exactly the trim, and
    the four dropped channels would read 177, 209, 248 and 293 instead.
    """
    import h5py

    with h5py.File(cohort_shards[0], "r") as handle:
        stored = np.asarray(handle["fhr_st"].attrs["causal_warmup_steps"], dtype=np.float64)

    target = emitted["declared"][emitted["declared"]["stream"] == "target"]
    emitted_st = np.asarray(target["causal_warmup_steps"])[:CAUSAL_ST_WIDTH]

    assert np.array_equal(emitted_st, np.maximum(stored - _TRIM_STEPS, 0.0))
    assert emitted_st.min() >= 0.0


def test_every_channel_carries_a_finite_group_delay(emitted) -> None:
    """``causal_delay_s`` is on all four stored blocks, so no row of either stream is missing it.

    It is what makes the lag axis stored-coefficient time rather than physical time, so a run whose
    map could not say it would have the lag numbers and no statement of what they are lags in.
    """
    delays = np.asarray(emitted["declared"]["causal_delay_s"], dtype=np.float64)

    assert delays.size == 153
    assert np.isfinite(delays).all()
    assert (delays > 0.0).all()


def test_the_source_stream_is_never_gated(emitted) -> None:
    """Its keep-index is the identity by construction, not by arithmetic that happens to keep all.

    The source channels a budget would drop are the ones carrying the contraction envelope, against
    a lag search that exists to find the contraction-to-deceleration delay.
    """
    source = emitted["declared"][emitted["declared"]["stream"] == "source"]

    assert bool(source["kept"].all())
    assert len(source) == CAUSAL_C_U


# =================================================================================================
# Provenance: the phase blocks alone carry it, and the gaps are recorded rather than bucketed
# =================================================================================================
def test_a_block_with_no_selection_provenance_is_not_a_shard_level_skip(emitted) -> None:
    """``sel_*`` is on the two phase blocks only, which is the common case on this dataset.

    An analysis that read a block of missing provenance as a shard-level skip would emit no channel
    map at all on every causal shard the pipeline produces.
    """
    record = emitted["record"]

    assert record["skipped"] is False
    for stream in ("target", "source"):
        assert record["streams"][stream]["coverage"]["up_ph_attrs_present"] is True
    assert record["files"]["channel_map"] == analysis.CHANNEL_MAP_FILENAME


def test_the_scattering_channels_are_banded_through_the_phase_filter_map(emitted) -> None:
    r"""One order-0 lowpass, 28 order-1 filters a selected pair named, 7 that none did.

    The seven have no recoverable centre frequency for the same reason their warm-up is longest or
    shortest: they sit outside the phase selection's own band, at both extremes of the axis. They
    are recorded as ``unknown`` and never bucketed into a neighbour, whose skill they do not share.
    """
    target = emitted["declared"][emitted["declared"]["stream"] == "target"]
    scattering = target[target["block"] == "scattering"]

    assert len(scattering) == CAUSAL_ST_WIDTH
    unknown = scattering[scattering["band"] == shared.UNKNOWN_BAND]
    assert len(unknown) == _TARGET_SCATTERING_WITHOUT_FREQUENCY
    assert (
        emitted["record"]["streams"]["target"]["n_scattering_without_frequency"]
        == _TARGET_SCATTERING_WITHOUT_FREQUENCY
    )

    # Exactly one order-0 lowpass, and it is banded on merit rather than as a fallback.
    assert int((scattering["band"] == "slow_baseline").sum()) == 1
    known = scattering[scattering["band"].isin(["deceleration", "variability", "beat_to_beat"])]
    assert len(known) == CAUSAL_ST_WIDTH - _TARGET_SCATTERING_WITHOUT_FREQUENCY - 1
    frequencies = np.asarray(known["freq_hz_primary"], dtype=np.float64)
    assert np.isfinite(frequencies).all()
    assert 0.008 < frequencies.min() and frequencies.max() < 1.0


def test_every_phase_channel_is_banded(emitted) -> None:
    """The phase blocks carry their own provenance, so none of their channels is unknown."""
    frame = emitted["declared"]
    phase = frame[frame["block"] == "phase"]

    assert len(phase) == CAUSAL_PH_WIDTH + (CAUSAL_C_U - CAUSAL_ST_WIDTH)
    assert not (phase["band"] == shared.UNKNOWN_BAND).any()


# =================================================================================================
# The kept axis, which is what a per-channel readout is actually indexed on
# =================================================================================================
def test_the_kept_axis_map_is_the_gathered_axis_and_carries_its_declared_index(emitted) -> None:
    """98 rows in gather order, each naming the declared channel it came from.

    Named rather than implied: the projection is what stops a band label from shifting across the
    axis, and a reader must be able to reconcile the two maps without trusting this analysis.
    """
    kept = emitted["kept"]
    target = emitted["declared"][emitted["declared"]["stream"] == "target"]
    survivors = list(target[target["kept"]]["channel"])

    assert len(kept) == _KEPT_TARGET_CHANNELS
    assert list(kept["kept_channel"]) == list(range(_KEPT_TARGET_CHANNELS))
    assert list(kept["channel"]) == survivors
    assert list(kept["band"]) == list(target[target["kept"]]["band"])
    assert bool(kept["kept"].all())


def test_the_kept_axis_map_is_not_a_prefix_slice_of_the_declared_one() -> None:
    """A keep-index whose survivors are not a prefix must project by index, never by position.

    Constructed rather than taken from the fixture, whose survivors *are* a prefix -- which is
    exactly the arrangement that would let a positional join pass here and be wrong elsewhere.
    """
    rows = [
        {"stream": "target", "channel": index, "band": f"b{index}", "kept": index in (0, 5, 9)}
        for index in range(10)
    ]
    rows.append({"stream": "source", "channel": 0, "band": "other", "kept": True})

    projected = analysis.kept_channel_rows(rows, (0, 5, 9))

    assert [row["channel"] for row in projected] == [0, 5, 9]
    assert [row["band"] for row in projected] == ["b0", "b5", "b9"]
    assert [row["kept_channel"] for row in projected] == [0, 1, 2]


def test_a_keep_index_naming_an_absent_channel_raises() -> None:
    """A keep-index and a shard describing different channel axes is not a shorter map.

    Emitting one silently would give every band-resolved statement a wrong denominator, which is
    unrecoverable from the output.
    """
    rows = [{"stream": "target", "channel": 0, "band": "b0", "kept": True}]

    with pytest.raises(KeyError, match="different channel axes"):
        analysis.kept_channel_rows(rows, (0, 7))


def test_the_kept_map_is_written_even_when_every_channel_survives(
    cohort_shards, tmp_path_factory
) -> None:
    """The ungated model's kept axis is the declared one, and the file still exists.

    The join that reads it must not have to branch on whether the file is there, and "the two axes
    coincide here" is itself a fact worth recording.
    """
    output_dir = tmp_path_factory.mktemp("band_partition_ungated")
    record = analysis.run_band_partition_analysis(
        _context(cohort_shards, None, CAUSAL_C_Y),
        eval_config={},
        output_dir=output_dir,
        probe=None,
    )

    kept = pd.read_csv(Path(output_dir) / analysis.KEPT_CHANNEL_MAP_FILENAME)
    assert record["kept_axis"]["from_keep_index"] is False
    assert len(kept) == CAUSAL_C_Y
    assert list(kept["channel"]) == list(range(CAUSAL_C_Y))
    assert record["kept_axis"]["dropped_declared"] == 0


# =================================================================================================
# Coverage: five counts, not one ratio
# =================================================================================================
def test_the_five_coverage_counts_are_emitted(emitted) -> None:
    r"""102 declared, 4 dropped, 98 scored, 95 of them banded, 3 not.

    Five numbers rather than a ratio because the declared and scored numerators coincide at 95 by
    arithmetic accident, and quoting "95 of 102" would imply channels the decoder never emitted.
    """
    kept_axis = emitted["record"]["kept_axis"]

    for name, expected in _COVERAGE.items():
        assert kept_axis[name] == expected, name
    assert kept_axis["known_kept"] + kept_axis["unknown_kept"] == kept_axis["kept_total"]
    assert kept_axis["dropped_declared"] + kept_axis["kept_total"] == kept_axis["declared_total"]


def test_all_four_dropped_channels_are_unknown_band(emitted) -> None:
    """They sit below the phase selection's floor, which is the same property that makes their
    warm-up longest -- so the budget removes exactly the low end of the frequency axis and takes no
    banded channel with it."""
    assert emitted["record"]["kept_axis"]["dropped_bands"] == {shared.UNKNOWN_BAND: 4}


def test_the_kept_width_the_collection_recorded_is_cross_checked(emitted) -> None:
    """The record says whether the two agree rather than assuming they do: a keep-index and a
    decoder width that disagreed would make every per-channel join off by the difference."""
    kept_axis = emitted["record"]["kept_axis"]

    assert kept_axis["reported_kept_width"] == _KEPT_TARGET_CHANNELS
    assert kept_axis["width_agrees"] is True


# =================================================================================================
# The protocol, and the artifacts
# =================================================================================================
def test_the_analysis_returns_the_protocols_keys_and_scores_no_population(emitted) -> None:
    """``n_samples`` is ``None`` rather than zero: this step describes the data, not a population,
    and a zero would enter the coverage block as a disagreement with every analysis that does."""
    record = emitted["record"]

    assert record["n_samples"] is None
    assert record["composition"] == {}
    assert record["plan"]["capped"] is False


def test_the_written_paths_are_relative_and_all_three_exist(emitted) -> None:
    """Relative because this record is compared across runs and across arms: an absolute path makes
    two identical findings differ, and stops being true the moment the directory is copied."""
    files = emitted["record"]["files"]

    assert set(files) == {"partition", "channel_map", "kept_channel_map"}
    for name in files.values():
        assert not Path(name).is_absolute()
        assert (emitted["dir"] / name).is_file()
    with open(emitted["dir"] / analysis.PARTITION_FILENAME, encoding="utf-8") as handle:
        partition = json.load(handle)
    assert set(partition) == {"target", "source"}


def test_no_configured_shard_is_a_recorded_skip_rather_than_a_raise(tmp_path) -> None:
    """A run whose readouts all succeeded must not be marked failed because its shards are an older
    vintage or because a re-run was pointed at a directory with no dataset behind it."""
    record = analysis.emit_partition([], tmp_path)

    assert record["skipped"] is True
    assert record["reason"] == "no test shards were configured"


# =================================================================================================
# Against a real run: the kept axis is the axis the readouts are indexed on
# =================================================================================================
@pytest.mark.slow
def test_the_kept_map_length_equals_the_per_channel_vectors_width(collected_run) -> None:
    """The one assertion that closes the join: the map ``spectral_skill`` will read has exactly as
    many rows as the gap vector it will be joined to has entries. Anything else is a silent
    misattribution of one channel's skill to another channel's band."""
    results_dir = Path(collected_run["results_dir"])
    kept = pd.read_csv(results_dir / analysis.KEPT_CHANNEL_MAP_FILENAME)
    with np.load(results_dir / "per_sample_vectors.npz") as vectors:
        gap = vectors["gap_per_channel"]

    assert gap.ndim == 2
    assert len(kept) == gap.shape[1]
