r"""The input channel map: one row per channel the model consumes, read off the shards.

The numbers here are the **production** ones, not a fixture's arithmetic. The generated shards
carry the measured ``sel_*`` provenance of the real selector at the real geometry, so the map they
produce is the map a real run produces: $109$ target channels and $58$ source channels, with the
clinical occupancy $1 / 22 / 40 / 32$ over the target's bands and fourteen channels the attributes
cannot place at all.

Those fourteen are what this analysis exists to make visible. Both endpoints of a selected phase
pair must lie inside the phase band, which leaves fourteen order-1 filters unreferenced -- the
three fastest and the eleven slowest -- so the scattering channels above them have no recoverable
centre frequency. Placed in a band by a guess they would silently inflate whichever band absorbed
them; dropped from the map they would silently shrink the denominator of every band-resolved
statement. They are counted instead, and the count is asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from teb_vae.lag_attn.eval.tests import real_selection
from teb_vae.lag_attn_rws.eval.analyses import band_partition as analysis

#: The two streams the model is given, and the widths the shipped config declares for them.
TARGET_WIDTH = real_selection.N_SCATTERING + real_selection.N_FHR_PHASE  # 43 + 66
SOURCE_WIDTH = real_selection.N_SCATTERING + real_selection.N_UP_PHASE  # 43 + 15


@pytest.fixture(scope="module")
def emitted(multi_class_shards, tmp_path_factory):
    """One emission against the generated shards, read back off disk."""
    output_dir = tmp_path_factory.mktemp("band_partition")
    record = analysis.emit_partition(
        multi_class_shards,
        output_dir,
        use_up_st=True,
        declared={"target": TARGET_WIDTH, "source": SOURCE_WIDTH},
    )
    return {"record": record, "output_dir": output_dir}


# =============================================================================
# The map covers every input channel
# =============================================================================
def test_the_map_has_one_row_per_input_channel(emitted) -> None:
    frame = pd.read_csv(emitted["output_dir"] / analysis.CHANNEL_MAP_FILENAME)

    assert emitted["record"]["skipped"] is False
    assert len(frame) == TARGET_WIDTH + SOURCE_WIDTH == 167
    assert frame["stream"].value_counts().to_dict() == {
        "target": TARGET_WIDTH, "source": SOURCE_WIDTH
    }


def test_each_stream_is_laid_out_as_the_model_receives_it(emitted) -> None:
    """``cat([st, ph])``: the scattering block first, then the phase block, per stream. A channel
    index in this map is the index the model's own tensors carry."""
    frame = pd.read_csv(emitted["output_dir"] / analysis.CHANNEL_MAP_FILENAME)

    for stream, n_scattering in (
        ("target", real_selection.N_SCATTERING), ("source", real_selection.N_SCATTERING)
    ):
        block = frame[frame["stream"] == stream].sort_values("channel")
        assert list(block["block"])[:n_scattering] == ["scattering"] * n_scattering
        assert set(block["block"][n_scattering:]) == {"phase"}
        assert list(block["channel"]) == list(range(len(block)))


def test_the_target_bands_are_the_measured_production_occupancy(emitted) -> None:
    """A regression guard on the selection itself: if the pipeline's channel choice changes,
    this number moves and says so, rather than every band-resolved figure quietly shifting."""
    target = emitted["record"]["streams"]["target"]

    assert target["band_counts"] == real_selection.CLINICAL_BAND_COUNTS
    assert target["kind_counts"] == {
        **real_selection.FHR_KIND_COUNTS,
        "st_S0": 1,
        "st_S1": real_selection.N_SCATTERING - 1,
    }


# =============================================================================
# The channels the attributes cannot place
# =============================================================================
def test_the_unplaceable_scattering_channels_are_recorded_rather_than_omitted(emitted) -> None:
    frame = pd.read_csv(emitted["output_dir"] / analysis.CHANNEL_MAP_FILENAME)
    unknown = frame[frame["band"] == "unknown"]

    expected = len(real_selection.UNREFERENCED_FILTERS)
    assert expected == 14
    for stream in ("target", "source"):
        assert emitted["record"]["streams"][stream]["n_scattering_without_frequency"] == expected
        assert len(unknown[unknown["stream"] == stream]) == expected
    # Present in the map with no frequency, rather than absent from it: a band-resolved statement
    # about this model's inputs is incomplete by exactly these channels.
    assert unknown["freq_hz_primary"].isna().all()
    assert set(unknown["block"]) == {"scattering"}


def test_the_unplaceable_channels_are_the_filters_no_selected_pair_referenced(emitted) -> None:
    """Scattering channel $c \\ge 1$ is filter $c - 1$, and channel $0$ is the order-0 lowpass."""
    frame = pd.read_csv(emitted["output_dir"] / analysis.CHANNEL_MAP_FILENAME)
    target = frame[(frame["stream"] == "target") & (frame["band"] == "unknown")]

    assert sorted(int(channel) - 1 for channel in target["channel"]) == sorted(
        real_selection.UNREFERENCED_FILTERS
    )


def test_the_frequencies_are_already_in_hz(emitted) -> None:
    """The writer multiplies kymatio's normalised $\\xi$ by $f_s$ before storing, so a consumer
    that multiplied again would land a factor of four high and misband every channel."""
    frame = pd.read_csv(emitted["output_dir"] / analysis.CHANNEL_MAP_FILENAME)
    phase = frame[frame["block"] == "phase"]

    assert float(phase["freq_hz_primary"].max()) == pytest.approx(
        max(real_selection.FILTER_HZ[index] for index in real_selection.FHR_J), rel=1e-5
    )


# =============================================================================
# The source stream is the model's, not the signal's
# =============================================================================
def test_dropping_the_source_scattering_block_leaves_the_phase_block_alone(
    multi_class_shards, tmp_path
) -> None:
    """``use_up_st: false`` makes $c_u = 15$. Those channels are not merely unread -- the model is
    built without them, so a map of what it consumes must not list them."""
    record = analysis.emit_partition(multi_class_shards, tmp_path, use_up_st=False)

    assert record["streams"]["source"]["n_channels"] == real_selection.N_UP_PHASE
    assert record["streams"]["source"]["n_scattering"] == 0
    assert record["streams"]["target"]["n_channels"] == TARGET_WIDTH


def test_a_width_the_config_disagrees_with_is_recorded_rather_than_raised(
    multi_class_shards, tmp_path
) -> None:
    """Preflight already refuses a width mismatch against the data the model is fed. A second
    refusal from a step that only describes would cost a run its channel map and add nothing."""
    record = analysis.emit_partition(
        multi_class_shards, tmp_path, declared={"target": 42, "source": SOURCE_WIDTH}
    )

    assert record["skipped"] is False
    assert record["width_disagreements"] == {
        "target": {"declared": 42, "described_by_shard": TARGET_WIDTH}
    }


# =============================================================================
# Absent provenance
# =============================================================================
def test_a_shard_without_selection_attributes_is_a_recorded_skip(tmp_path) -> None:
    """A run whose readouts all succeeded must not be marked failed because its shards are an
    older vintage than ``_write_selection_attrs``."""
    import h5py

    shard = tmp_path / "no_attrs.hdf5"
    with h5py.File(str(shard), "w") as handle:
        for name, width in (("fhr_ph", 66), ("up_ph", 15), ("fhr_st", 43), ("up_st", 43)):
            handle.create_dataset(name, shape=(2, width, 5), dtype="f4")

    record = analysis.emit_partition([str(shard)], tmp_path / "out")

    assert record["skipped"] is True
    assert "sel_" in record["attempts"][0], "the skip must carry the actionable reason"
    assert not (tmp_path / "out" / analysis.CHANNEL_MAP_FILENAME).exists()


def test_no_configured_shard_at_all_is_distinguished_from_an_unusable_one(tmp_path) -> None:
    record = analysis.emit_partition([], tmp_path)

    assert record["skipped"] is True
    assert record["attempts"] == []
    assert "no test shards were configured" in record["reason"]


# =============================================================================
# The run emits it
# =============================================================================
def test_the_run_emits_the_channel_map_unskippably(evaluated) -> None:
    """It is not on the selectable registry: a run whose channel map could be skipped is a run
    whose frequency-resolved statements have no definition of a band behind them."""
    block = evaluated["summary"]["results"]["band_partition"]

    assert "band_partition" in evaluated["summary"]["analyses_unskippable"]
    assert block["skipped"] is False
    assert block["n_channels"] == 167
    for name in (analysis.PARTITION_FILENAME, analysis.CHANNEL_MAP_FILENAME):
        assert (Path(evaluated["results_dir"]) / name).is_file()


def test_the_emitted_paths_are_relative_to_the_run_directory(evaluated) -> None:
    """``results`` is compared across runs and across sweep arms; an absolute path makes two
    identical findings differ and stops being true the moment the directory is copied."""
    files = evaluated["summary"]["results"]["band_partition"]["files"]

    assert files == {
        "partition": analysis.PARTITION_FILENAME,
        "channel_map": analysis.CHANNEL_MAP_FILENAME,
    }


def test_the_written_partition_round_trips_as_json(emitted) -> None:
    blob = json.loads(
        (emitted["output_dir"] / analysis.PARTITION_FILENAME).read_text(encoding="utf-8")
    )

    assert sorted(blob) == ["source", "target"]
    assert blob["target"]["n_channels"] == TARGET_WIDTH
    assert len(blob["target"]["channels"]) == TARGET_WIDTH
