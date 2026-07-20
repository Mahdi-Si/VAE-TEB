r"""The channel map built from a shard's own provenance.

Driven by a synthetic shard written into ``tmp_path`` -- five lines of ``h5py`` -- rather than by
the committed binary fixtures. The committed ``tiny_shard.hdf5`` carries no ``sel_*`` attrs at
all (it is synthesised by ``scripts/make_tiny_shard.py``, not produced by the real pipeline), and
regenerating it to add them would perturb every other test in the suite for the benefit of this
one file.

The synthetic selection reproduces the production geometry's structure at a smaller size: powers
drawn from the $2^{k/Q}$ grid for $k \in \{4, 6, 8\}$, filter indices into an order-1 bank, and
centre frequencies already in Hz -- which is the property the whole map depends on and the one a
consumer is most likely to get wrong by multiplying by $f_s$ a second time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from teb_vae.lag_attn.eval import band_partition
from teb_vae.lag_attn.eval.tests import real_selection

#: Small stand-in for the production $43$.
N_SCATTERING = 8

#: The harmonic steps the current selection admits. $k = 0$ -- the diagonal -- is deliberately
#: not among them, which is why there is no ``ph_diag`` kind.
K_STEPS = (4, 6, 8)


def _make_shard(
    path: Path,
    *,
    n_phase: int = 9,
    n_up_phase: int = 3,
    attrs: bool = True,
    drop: tuple = (),
    stored_width: Optional[int] = None,
) -> Path:
    r"""Write a synthetic shard carrying ``sel_*`` provenance.

    Filter indices are laid out so the recovered filter-to-Hz map covers most but not all of the
    scattering bank, which is exactly the production situation: a filter outside the phase
    selection's band is never referenced and its channel has no recoverable frequency.

    Args:
        path: Destination file.
        n_phase: Number of ``fhr_ph`` channels.
        n_up_phase: Number of ``up_ph`` channels.
        attrs: Whether to write the selection attributes at all.
        drop: Attribute names to omit, for the missing-provenance cases.
        stored_width: Override the stored channel axis, to desynchronise it from the attrs.

    Returns:
        ``path``.
    """
    import h5py

    # Centre frequencies on a geometric grid, in Hz -- as the writer stores them.
    filter_hz = 2.0 * (2.0 ** (-np.arange(N_SCATTERING - 1) / 4.0))

    def _selection(count: int, offset: int) -> dict:
        # Pair filter i with filter j = i + k/... ; the exact pairing does not matter, only that
        # power = xi_j / xi_i lands on the 2^(k/Q) grid, which the geometric spacing guarantees.
        steps = np.array([K_STEPS[index % len(K_STEPS)] for index in range(count)])
        i_idx = np.array([(index + offset) % (N_SCATTERING - 2) + 1 for index in range(count)])
        j_idx = i_idx - (steps // 2)
        j_idx = np.clip(j_idx, 0, N_SCATTERING - 2)
        xi_i, xi_j = filter_hz[i_idx], filter_hz[j_idx]
        return {
            "sel_i": i_idx.astype(np.int32),
            "sel_j": j_idx.astype(np.int32),
            "sel_xi_i_hz": xi_i.astype(np.float32),
            "sel_xi_j_hz": xi_j.astype(np.float32),
            "sel_power": (xi_j / xi_i).astype(np.float32),
            "sel_band_hz": np.asarray([0.008, 1.0], dtype=np.float32),
            "sel_k_steps": np.asarray(K_STEPS, dtype=np.int32),
        }

    with h5py.File(str(path), "w") as handle:
        for name, count, offset in (("fhr_ph", n_phase, 0), ("up_ph", n_up_phase, 2)):
            width = count if stored_width is None or name != "fhr_ph" else stored_width
            node = handle.create_dataset(name, shape=(2, width, 5), dtype="f4")
            if not attrs:
                continue
            for key, value in _selection(count, offset).items():
                if key not in drop:
                    node.attrs[key] = value
        handle.create_dataset("fhr_st", shape=(2, N_SCATTERING, 5), dtype="f4")
    return path


@pytest.fixture
def shard(tmp_path) -> Path:
    return _make_shard(tmp_path / "synthetic.hdf5")


@pytest.fixture
def partition(shard):
    return band_partition.build_partition(shard, n_scattering=N_SCATTERING)


# ---------------------------------------------------------------------------
# Kind derivation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", K_STEPS)
def test_the_kind_inverts_the_harmonic_grid(k):
    r"""$p = 2^{k/Q}$ must come back as $k$."""
    assert band_partition.kind_of_power(2.0 ** (k / 4.0)) == f"ph_k{k}"


def test_a_power_slightly_off_the_grid_still_rounds_to_its_step():
    """The selection admits a 5% relative tolerance, so exact powers are not what arrives."""
    assert band_partition.kind_of_power(2.0 ** (6 / 4.0) * 1.02) == "ph_k6"


def test_a_degenerate_power_is_labelled_rather_than_silently_binned():
    assert band_partition.kind_of_power(0.0) == "ph_unknown"
    assert band_partition.kind_of_power(float("nan")) == "ph_unknown"


def test_no_diagonal_kind_is_produced_by_the_current_selection(partition):
    """Documented as expected, not an error: k = 0 is excluded by the shipped k_steps."""
    assert "ph_k0" not in partition.kind_counts()
    assert "ph_diag" not in partition.kind_counts()


# ---------------------------------------------------------------------------
# Band assignment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "hz, expected",
    [
        (0.004, "slow_baseline"),
        (0.008, "deceleration"),
        (0.039, "deceleration"),
        (0.04, "variability"),
        # Inclusive at the top, so the clinical [0.04, 0.25] Hz convention holds exactly.
        (0.25, "variability"),
        (0.3, "beat_to_beat"),
        (float("nan"), band_partition.UNKNOWN_BAND),
    ],
)
def test_clinical_band_edges(hz, expected):
    assert band_partition.band_of_hz(hz) == expected


def test_the_frequencies_are_used_as_stored_with_no_further_fs_multiplication(partition):
    r"""The writer already multiplied by $f_s$; doing it again lands every channel a factor of
    four high and moves most of them a whole band."""
    phase = [record for record in partition.channels if record.block == "phase"]
    stored_max = 2.0  # the synthetic bank's top centre frequency, in Hz
    assert max(record.freq_hz_primary for record in phase) <= stored_max + 1e-6


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------
def test_every_channel_is_mapped_exactly_once(partition):
    assert partition.n_channels == N_SCATTERING + partition.n_phase
    assert [record.channel for record in partition.channels] == list(range(partition.n_channels))


@pytest.mark.parametrize("name", ["clinical", "by_kind"])
def test_each_partition_tiles_the_channel_space_with_no_gaps_or_duplicates(partition, name):
    groups = partition.partition(name)
    covered = [index for indices in groups.values() for index in indices]
    assert sorted(covered) == list(range(partition.n_channels)), "gap or duplicate"
    assert len(covered) == len(set(covered))


def test_the_scattering_block_precedes_the_phase_block(partition):
    """The order the model is given: ``cat([y_st, y_ph])``."""
    blocks = [record.block for record in partition.channels]
    assert blocks[:N_SCATTERING] == ["scattering"] * N_SCATTERING
    assert set(blocks[N_SCATTERING:]) == {"phase"}


def test_channel_zero_is_the_order_zero_lowpass(partition):
    first = partition.channels[0]
    assert first.kind == band_partition.KIND_ORDER0
    assert not np.isfinite(first.freq_hz_primary)
    # On merit rather than as a fallback: the lowpass carries the slowest content there is.
    assert first.band == "slow_baseline"


def test_an_order_one_channel_maps_to_the_filter_one_index_below_it(partition):
    """``check_phase_diagonal_redundancy`` pins this ``i -> i+1`` layout."""
    for record in partition.channels[1:N_SCATTERING]:
        assert record.filter_i == record.channel - 1


def test_a_phase_channel_carries_both_frequencies_and_its_ratio(partition):
    phase = [record for record in partition.channels if record.block == "phase"]
    for record in phase:
        assert record.freq_hz_secondary <= record.freq_hz_primary + 1e-6, "xi_i <= xi_j"
        assert record.harmonic_ratio == pytest.approx(
            record.freq_hz_primary / record.freq_hz_secondary, rel=1e-4
        )


def test_a_scattering_channel_outside_the_phase_band_is_marked_unknown_not_guessed(tmp_path):
    """The one real limit of the attrs-only route, and it must be visible rather than filled in."""
    # A single phase channel references only two filters, so most scattering channels have no
    # recoverable frequency.
    shard = _make_shard(tmp_path / "narrow.hdf5", n_phase=1, n_up_phase=1)
    partition = band_partition.build_partition(shard, n_scattering=N_SCATTERING)

    assert partition.coverage["n_scattering_without_frequency"] > 0
    unknown = partition.partition("clinical")[band_partition.UNKNOWN_BAND]
    assert unknown, "unrecoverable channels must land in their own band"
    for channel in unknown:
        assert not np.isfinite(partition.channels[channel].freq_hz_primary)


def test_the_coverage_block_records_the_selection_the_shard_was_built_with(partition):
    assert partition.coverage["phase_k_steps"] == list(K_STEPS)
    assert partition.coverage["phase_band_hz"] == pytest.approx([0.008, 1.0])
    assert partition.coverage["up_ph_attrs_present"] is True


def test_the_partition_builds_without_up_ph_provenance(tmp_path):
    """``up_ph`` only widens the filter map; the target channels are all described without it."""
    shard = _make_shard(tmp_path / "no_up.hdf5")
    import h5py

    with h5py.File(str(shard), "a") as handle:
        for key in list(handle["up_ph"].attrs):
            del handle["up_ph"].attrs[key]

    partition = band_partition.build_partition(shard, n_scattering=N_SCATTERING)
    assert partition.coverage["up_ph_attrs_present"] is False
    assert partition.n_channels == N_SCATTERING + partition.n_phase


# ---------------------------------------------------------------------------
# Missing or inconsistent provenance
# ---------------------------------------------------------------------------
def test_a_shard_without_the_attributes_raises_naming_the_fallback(tmp_path):
    """The spike's alternative must be in the message: it is what the operator has to do next."""
    shard = _make_shard(tmp_path / "bare.hdf5", attrs=False)
    with pytest.raises(RuntimeError) as excinfo:
        band_partition.build_partition(shard, n_scattering=N_SCATTERING)
    message = str(excinfo.value)
    assert "sel_i" in message
    assert "compute_scattering_masks" in message, "the message must name the fallback"
    assert "_write_selection_attrs" in message, "and where the attrs come from"


def test_a_partially_written_selection_raises_naming_the_missing_attribute(tmp_path):
    shard = _make_shard(tmp_path / "partial.hdf5", drop=("sel_power",))
    with pytest.raises(RuntimeError, match="sel_power"):
        band_partition.build_partition(shard, n_scattering=N_SCATTERING)


def test_provenance_that_does_not_match_the_stored_width_raises(tmp_path):
    """A map built from the wrong provenance is off by an unknown offset and looks fine."""
    shard = _make_shard(tmp_path / "desync.hdf5", n_phase=9, stored_width=11)
    with pytest.raises(RuntimeError, match="does not belong to this data"):
        band_partition.build_partition(shard, n_scattering=N_SCATTERING)


def test_an_unknown_partition_name_lists_the_available_ones(partition):
    with pytest.raises(ValueError, match="by_kind"):
        partition.partition("by_octave")


# ---------------------------------------------------------------------------
# The real production selection
#
# The synthetic shard above proves the mechanics; these prove the builder against the actual
# 109-channel geometry, using provenance measured off the real selector rather than invented.
# See tests/real_selection.py for where the numbers come from.
# ---------------------------------------------------------------------------
@pytest.fixture
def real_partition(tmp_path):
    """The partition built from the production channel selection."""
    shard = real_selection.write_shard(tmp_path / "production.hdf5")
    return band_partition.build_partition(shard, n_scattering=real_selection.N_SCATTERING)


def test_the_real_geometry_yields_109_channels(real_partition):
    assert real_partition.n_scattering == 43
    assert real_partition.n_phase == 66
    assert real_partition.n_channels == 109


def test_the_measured_harmonic_kind_distribution_is_reproduced(real_partition):
    """24 / 22 / 20 for $k \\in \\{4, 6, 8\\}$ -- the figure the selection actually produces."""
    counts = real_partition.kind_counts()
    for kind, expected in real_selection.FHR_KIND_COUNTS.items():
        assert counts[kind] == expected, f"{kind}: expected {expected}, got {counts.get(kind)}"
    # The scattering side: one order-0 lowpass and 42 order-1 filters.
    assert counts["st_S0"] == 1 and counts["st_S1"] == 42
    assert sum(counts.values()) == 109


def test_no_diagonal_kind_appears_at_the_real_geometry(real_partition):
    """$k = 0$ is excluded by the shipped ``k_steps``, so there is no ``ph_diag``."""
    assert not any(kind in real_partition.kind_counts() for kind in ("ph_k0", "ph_diag"))


def test_the_real_clinical_band_occupancy_is_pinned(real_partition):
    """A pipeline change that moved channels between bands would otherwise be invisible."""
    counts = {
        name: len(indices) for name, indices in real_partition.partition("clinical").items()
    }
    assert counts == real_selection.CLINICAL_BAND_COUNTS
    assert sum(counts.values()) == 109


@pytest.mark.parametrize("name", ["clinical", "by_kind"])
def test_the_real_partitions_tile_all_109_channels(real_partition, name):
    covered = sorted(i for indices in real_partition.partition(name).values() for i in indices)
    assert covered == list(range(109))


def test_the_unrecoverable_scattering_channels_are_exactly_the_unreferenced_filters(
    real_partition,
):
    """The attrs-only route's one real limit, measured rather than estimated.

    Both endpoints of a selected pair must lie inside the phase band, so the three fastest and
    eleven slowest order-1 filters are never referenced -- and scattering channel $c = f + 1$
    above them has no centre frequency to band it by.
    """
    unknown = set(real_partition.partition("clinical")[band_partition.UNKNOWN_BAND])
    expected = {index + 1 for index in real_selection.UNREFERENCED_FILTERS}
    assert unknown == expected
    assert len(unknown) == 14
    assert real_partition.coverage["n_scattering_without_frequency"] == 14
    assert real_partition.coverage["n_filters_with_frequency"] == 28


def test_the_real_frequencies_land_inside_the_selections_own_band(real_partition):
    r"""If the Hz were multiplied by $f_s$ a second time they would land four times too high."""
    phase = [record for record in real_partition.channels if record.block == "phase"]
    low, high = real_selection.FHR_BAND_HZ
    for record in phase:
        assert low <= record.freq_hz_secondary <= high + 1e-6
        assert low <= record.freq_hz_primary <= high + 1e-6
    # And the top of the band is genuinely approached, so this is not vacuous.
    assert max(record.freq_hz_primary for record in phase) > 0.8


def test_the_higher_frequency_of_each_real_pair_is_the_lower_filter_index(real_partition):
    """``FILTER_HZ`` descends with index, which is the trap ``sel_i``/``sel_j`` exist to avoid."""
    phase = [record for record in real_partition.channels if record.block == "phase"]
    for record in phase:
        assert record.filter_j < record.filter_i
        assert record.freq_hz_primary > record.freq_hz_secondary


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_the_json_reloads_into_an_equivalent_partition(partition, tmp_path):
    written = band_partition.write_partition(partition, tmp_path / "out")
    reloaded = band_partition.load_partition(written["partition"])

    assert reloaded.n_channels == partition.n_channels
    assert reloaded.n_scattering == partition.n_scattering
    assert reloaded.kind_counts() == partition.kind_counts()
    for name in ("clinical", "by_kind"):
        assert reloaded.partition(name) == partition.partition(name)
    # The unbounded top band survives the round trip, which JSON cannot represent directly.
    assert reloaded.band_hz_ranges["beat_to_beat"][1] == float("inf")


def test_the_json_carries_the_labels_ranges_and_index_lists(partition, tmp_path):
    written = band_partition.write_partition(partition, tmp_path / "out")
    blob = json.loads(Path(written["partition"]).read_text(encoding="utf-8"))

    assert set(blob["partitions"]) == {"clinical", "by_kind"}
    assert blob["band_hz_ranges"]["deceleration"] == [0.008, 0.04]
    assert blob["band_hz_ranges"]["beat_to_beat"][1] is None, "infinity serialises as null"
    assert sum(len(indices) for indices in blob["partitions"]["clinical"].values()) == (
        blob["n_channels"]
    )


def test_emit_writes_both_files_and_summarises_the_partition(shard, tmp_path):
    record = band_partition.emit_partition([shard], N_SCATTERING, tmp_path / "run")

    assert record["skipped"] is False
    assert Path(record["files"]["partition"]).is_file()
    assert Path(record["files"]["channel_map"]).is_file()
    assert record["n_channels"] == record["n_scattering"] + record["n_phase"]
    assert sum(record["band_counts"].values()) == record["n_channels"]


def test_emit_falls_through_to_the_next_shard_when_the_first_has_no_provenance(tmp_path):
    """A run's shards need not all be the same vintage; one usable file is enough."""
    bare = _make_shard(tmp_path / "bare.hdf5", attrs=False)
    good = _make_shard(tmp_path / "good.hdf5")
    record = band_partition.emit_partition([bare, good], N_SCATTERING, tmp_path / "run")

    assert record["skipped"] is False and record["shard"] == str(good)


def test_emit_records_a_skip_rather_than_raising_when_no_shard_has_provenance(tmp_path):
    """A skip, not a failure: the rest of the run is unaffected and nothing consumes this yet."""
    bare = _make_shard(tmp_path / "bare.hdf5", attrs=False)
    record = band_partition.emit_partition([bare], N_SCATTERING, tmp_path / "run")

    assert record["skipped"] is True
    assert len(record["attempts"]) == 1
    assert "sel_i" in record["attempts"][0], "the skip must carry the actionable reason"
    assert not (tmp_path / "run" / band_partition.PARTITION_FILENAME).exists()


def test_emit_with_no_configured_shards_says_so(tmp_path):
    record = band_partition.emit_partition([], N_SCATTERING, tmp_path / "run")
    assert record["skipped"] is True and "no test shards" in record["reason"]


def test_the_csv_has_one_row_per_channel_with_its_kind_band_and_frequencies(
    partition, tmp_path
):
    """The CSV exists so a downstream plot can be redrawn with pandas and no import from here."""
    import pandas as pd

    written = band_partition.write_partition(partition, tmp_path / "out")
    frame = pd.read_csv(written["channel_map"])

    assert len(frame) == partition.n_channels
    for column in ("channel", "block", "kind", "band", "freq_hz_primary", "freq_hz_secondary"):
        assert column in frame.columns
    assert frame["channel"].tolist() == list(range(partition.n_channels))
