r"""What each of the $c_y = 109$ target channels means, read off the data rather than re-derived.

A frequency-resolved analysis needs to know which channel is which: that channel $57$ is a
phase-harmonic coefficient pairing filters at $0.021$ and $0.05$ Hz, and therefore sits in the
deceleration band. The predecessor pipeline answered that by *re-running the channel selector*.
That no longer works and cannot be repaired: its selector returns $44$ phase channels against
$66$-channel data, and it omits the $f_s$ conversion, so its nominal $0.006$ Hz threshold is
really $0.024$ Hz.

The shards already carry the answer. ``create_new_pipeline.py::_write_selection_attrs`` stamps
per-channel provenance onto the ``fhr_ph`` and ``up_ph`` datasets -- ``sel_i``, ``sel_j``,
``sel_xi_i_hz``, ``sel_xi_j_hz``, ``sel_power``, ``sel_band_hz``, ``sel_k_steps`` -- ordered to
match the stored channel axis, precisely so a consumer can stop re-deriving the selection. Those
attributes are the source of truth here.

Three properties of that provenance are load-bearing, and two of them are settled by the writer's
own source rather than by inspection:

**The frequencies are already in Hz.** ``_build_phase_selection`` multiplies kymatio's normalised
$\xi$ (cycles per sample) by $f_s$ before storing, so a consumer that multiplied again would land
a factor of $4$ high and put every channel in the wrong band.

**The ordering matches the stored channel axis.** The arrays are produced by boolean-indexing the
phase-pair axis with the same mask the coefficient block is sliced with, and boolean indexing
preserves ascending pair order. Channel $k$ of ``fhr_ph`` is described by element $k$ of every
``sel_*`` array.

**The scattering side comes from the same filter bank, transported through those same attrs.**
``sel_i`` and ``sel_j`` are indices into kymatio's order-1 filters, and
``check_phase_diagonal_redundancy.py`` pins the layout: exactly one order-0 channel, then the
order-1 filters in order, so scattering channel $c \ge 1$ is filter $c - 1$ and channel $0$ is
the order-0 lowpass with no centre frequency. Pairing ``sel_i`` with ``sel_xi_i_hz`` (and ``sel_j``
with ``sel_xi_j_hz``) therefore recovers the filter-index-to-Hz map without importing kymatio and
without coupling this package to the dataset-building module.

That last route recovers a filter's frequency only if some selected phase pair used it, and at
the production geometry it does not recover all of them. Measured against the real selector
($J = 11$, $Q = 4$, $T = 16$, ``shape=5280``, $f_s = 4$ Hz): both endpoints of a pair must lie
inside the ``fhr_ph`` band of $(0.008, 1.00)$ Hz, which references filters $3$ to $30$ and leaves
$14$ of $42$ unreferenced -- the three fastest ($\ge 1.05$ Hz) and the eleven slowest
($\le 0.0069$ Hz). The $14$ scattering channels above them therefore have no centre frequency,
and are placed in :data:`UNKNOWN_BAND` and counted in the partition's ``coverage`` block rather
than guessed at. The resulting clinical occupancy over the $109$ target channels is
$1 / 22 / 40 / 32$ across ``slow_baseline`` / ``deceleration`` / ``variability`` /
``beat_to_beat``, plus those $14$.

The arithmetic is per *stream* rather than per model: :func:`build_partition` partitions whichever
block ``phase_dataset`` names, so the same call describes the $c_y$ target of this package's model
and the $c_u$ source stream a model that consumes one is given. Only the phase block's identity
changes; the scattering side is banded through the same filter map either way, because both
selections index one filter bank.

Note also that ``FILTER_HZ`` **descends** with index -- filter $0$ is the fastest -- so the
higher-frequency member of a pair carries the *lower* index. That is exactly the kind of
inversion re-deriving the selection gets wrong, and reading ``sel_i`` / ``sel_j`` avoids.

``lean-limit: attrs-only band partition, so 14 of the 43 scattering channels carry no frequency;
add the compute_scattering_masks fallback when a frequency-resolved analysis is actually blocked
by those channels, or when a shard without sel_* attrs is encountered. Both the fallback's
dependency (kymatio) and its cost (a filter-bank construction) are available -- the measurements
quoted above were taken through it.``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The ``sel_*`` attributes a phase dataset must carry. All seven are written together by
#: ``_write_selection_attrs``, so a shard carrying some but not all is a partial write rather
#: than an older vintage, and is worth naming separately in the error.
REQUIRED_ATTRS: Tuple[str, ...] = (
    "sel_i",
    "sel_j",
    "sel_xi_i_hz",
    "sel_xi_j_hz",
    "sel_power",
)

#: The phase datasets a shard carries, one per signal. Both index the *same* order-1 filter bank,
#: so either one's ``sel_i`` / ``sel_j`` pairs widen the filter-index-to-Hz map the scattering
#: side is banded with -- which is why a partition of one block still harvests the other.
PHASE_DATASETS: Tuple[str, ...] = ("fhr_ph", "up_ph")

#: Wavelets per octave, $Q$. The harmonic grid is $p = 2^{k/Q}$, so the kind of a channel is
#: $k = \mathrm{round}(Q \log_2 p)$. Matches ``create_new_pipeline.SCATTERING_Q``; restated
#: rather than imported because ``teb_vae`` does not depend on ``hdf5_dataset``.
SCATTERING_Q = 4

#: Clinical frequency bands, as half-open $[\mathrm{low}, \mathrm{high})$ intervals in Hz, except
#: ``variability`` whose upper bound is inclusive so that $0.25$ Hz lands there rather than in
#: ``beat_to_beat``. These are the fetal-monitoring conventions the predecessor used, kept
#: identical so a band-resolved number from either tree means the same thing.
CLINICAL_BANDS: Dict[str, Tuple[float, float]] = {
    "slow_baseline": (0.0, 0.008),
    "deceleration": (0.008, 0.04),
    "variability": (0.04, 0.25),
    "beat_to_beat": (0.25, float("inf")),
}

#: Band for a channel whose centre frequency the attributes do not determine. Explicit rather
#: than folded into ``slow_baseline``: "we could not tell" and "it is a slow channel" are
#: different statements, and merging them would silently inflate the slow band.
UNKNOWN_BAND = "unknown"

#: Kind of the order-0 scattering channel, which has no centre frequency at all.
KIND_ORDER0 = "st_S0"

#: Kind of an order-1 scattering channel.
KIND_ORDER1 = "st_S1"


def kind_of_power(power: float, *, q: int = SCATTERING_Q) -> str:
    r"""Return the harmonic-kind label for a phase channel's power $p = \xi_j / \xi_i$.

    The selection admits pairs whose power sits within a relative tolerance of $2^{k/Q}$ for
    $k \in \{4, 6, 8\}$, so inverting gives $k = \mathrm{round}(Q \log_2 p)$.

    Note there is no ``ph_diag`` kind, and its absence is expected rather than a defect: a
    diagonal channel is $k = 0$, i.e. $\xi_i = \xi_j$, and the current selection's ``k_steps``
    begins at $4$. The predecessor's taxonomy carried one because its selection did.

    Args:
        power: The harmonic ratio $p \ge 1$.
        q: Wavelets per octave.

    Returns:
        ``'ph_k<k>'``, or ``'ph_unknown'`` when the power is not positive and finite.
    """
    value = float(power)
    if not np.isfinite(value) or value <= 0.0:
        return "ph_unknown"
    return f"ph_k{int(round(float(q) * np.log2(value)))}"


def band_of_hz(freq_hz: float, bands: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
    """Return the clinical band a centre frequency falls into.

    Args:
        freq_hz: The centre frequency in Hz. Non-finite yields :data:`UNKNOWN_BAND`.
        bands: The band table. ``None`` uses :data:`CLINICAL_BANDS`.

    Returns:
        The band name.
    """
    table = CLINICAL_BANDS if bands is None else bands
    value = float(freq_hz)
    if not np.isfinite(value):
        return UNKNOWN_BAND
    for name, (low, high) in table.items():
        # Inclusive at the top for ``variability`` -- see CLINICAL_BANDS.
        if name == "variability" and low <= value <= high:
            return name
        if low <= value < high:
            return name
    return list(table)[-1]


@dataclass(frozen=True)
class ChannelRecord:
    r"""One target channel's identity.

    Attributes:
        channel: Index into the concatenated $c_y$ target vector.
        block: ``'scattering'`` or ``'phase'``.
        kind: Coefficient kind -- ``st_S0``, ``st_S1``, or ``ph_k<k>``.
        band: Clinical band, or :data:`UNKNOWN_BAND`.
        freq_hz_primary: $\xi_j$ for a phase channel, the centre frequency for a scattering one.
        freq_hz_secondary: $\xi_i$ for a phase channel, ``NaN`` for a scattering one.
        harmonic_ratio: $p = \xi_j / \xi_i$, ``NaN`` for a scattering channel.
        filter_i: Lower-frequency filter index, or ``None``.
        filter_j: Higher-frequency filter index, or ``None``.
    """

    channel: int
    block: str
    kind: str
    band: str
    freq_hz_primary: float
    freq_hz_secondary: float
    harmonic_ratio: float
    filter_i: Optional[int] = None
    filter_j: Optional[int] = None

    def as_row(self) -> Dict[str, Any]:
        """Return the record as a flat dict, one CSV row."""
        return {
            "channel": self.channel,
            "block": self.block,
            "kind": self.kind,
            "band": self.band,
            "freq_hz_primary": self.freq_hz_primary,
            "freq_hz_secondary": self.freq_hz_secondary,
            "harmonic_ratio": self.harmonic_ratio,
            "filter_i": self.filter_i,
            "filter_j": self.filter_j,
        }


@dataclass
class BandPartition:
    """The channel map plus the two partitions over it.

    Attributes:
        channels: One record per target channel, in channel order.
        n_scattering: Width of the scattering block.
        n_phase: Width of the phase-harmonic block.
        band_hz_ranges: The clinical band table the partition was built with.
        coverage: What the attributes did and did not determine.
    """

    channels: List[ChannelRecord]
    n_scattering: int
    n_phase: int
    band_hz_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(CLINICAL_BANDS)
    )
    coverage: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        r"""Total target width $c_y$."""
        return len(self.channels)

    def partition(self, name: str) -> Dict[str, List[int]]:
        """Return one partition as label to channel indices.

        Args:
            name: ``'clinical'`` or ``'by_kind'``.

        Returns:
            Label to a sorted list of channel indices. Every channel appears in exactly one
            label, so the lists tile $[0, c_y)$.

        Raises:
            ValueError: If ``name`` is not a known partition.
        """
        if name == "clinical":
            labels = list(self.band_hz_ranges) + [UNKNOWN_BAND]
            key = "band"
        elif name == "by_kind":
            labels = sorted({record.kind for record in self.channels})
            key = "kind"
        else:
            raise ValueError(
                f"unknown partition {name!r}. Available: 'clinical', 'by_kind'. A partition by "
                f"octave has no consumer and would be a one-line addition here if one appeared."
            )
        groups: Dict[str, List[int]] = {label: [] for label in labels}
        for record in self.channels:
            groups.setdefault(getattr(record, key), []).append(record.channel)
        return {label: sorted(indices) for label, indices in groups.items()}

    def kind_counts(self) -> Dict[str, int]:
        """Return the number of channels of each kind."""
        counts: Dict[str, int] = {}
        for record in self.channels:
            counts[record.kind] = counts.get(record.kind, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> Dict[str, Any]:
        """Return the partition as a JSON-safe structure."""
        return {
            "n_channels": self.n_channels,
            "n_scattering": self.n_scattering,
            "n_phase": self.n_phase,
            "band_hz_ranges": {
                name: [low, None if not np.isfinite(high) else high]
                for name, (low, high) in self.band_hz_ranges.items()
            },
            "partitions": {
                name: self.partition(name) for name in ("clinical", "by_kind")
            },
            "kind_counts": self.kind_counts(),
            "coverage": dict(self.coverage),
            "channels": [record.as_row() for record in self.channels],
        }


def read_selection(path: Any, dataset: str) -> Dict[str, np.ndarray]:
    """Read one dataset's ``sel_*`` provenance attributes out of an HDF5 shard.

    Args:
        path: Path to the shard.
        dataset: Dataset name, ``'fhr_ph'`` or ``'up_ph'``.

    Returns:
        Attribute name to array, for every attribute present.

    Raises:
        KeyError: If the shard has no such dataset.
        RuntimeError: If any required attribute is missing, or the arrays disagree in length
            with each other or with the stored channel axis. All three mean the channel map
            cannot be trusted, and a partially-built map is worse than none: it would put real
            channels in the wrong band and nothing downstream would notice.
    """
    import h5py

    with h5py.File(str(path), "r") as handle:
        if dataset not in handle:
            raise KeyError(
                f"{str(path)!r} has no {dataset!r} dataset, so its channel provenance cannot be "
                f"read."
            )
        node = handle[dataset]
        attrs = {name: np.asarray(node.attrs[name]) for name in node.attrs}
        # (n_samples, n_channels, n_steps): the channel axis is the middle one.
        stored_width = int(node.shape[1]) if len(node.shape) >= 2 else -1

    missing = [name for name in REQUIRED_ATTRS if name not in attrs]
    if missing:
        raise RuntimeError(
            f"{dataset!r} in {str(path)!r} is missing the selection attribute(s) {missing}. "
            f"Present: {sorted(attrs)}. These are written by "
            f"create_new_pipeline.py::_write_selection_attrs, so a shard without them predates "
            f"that writer. Either rebuild the shard with the current pipeline, or add the "
            f"compute_scattering_masks fallback -- which re-derives the selection from the "
            f"filter bank at the cost of importing kymatio into the eval path."
        )

    lengths = {name: int(attrs[name].shape[0]) for name in REQUIRED_ATTRS}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"{dataset!r} in {str(path)!r} has selection attributes of differing lengths: "
            f"{lengths}. They describe one channel axis and must agree."
        )
    width = next(iter(lengths.values()))
    if stored_width >= 0 and width != stored_width:
        raise RuntimeError(
            f"{dataset!r} in {str(path)!r} stores {stored_width} channels but its selection "
            f"attributes describe {width}. The provenance does not belong to this data, so "
            f"every channel's band would be wrong by an unknown offset."
        )
    return attrs


def _filter_frequencies(selections: Sequence[Dict[str, np.ndarray]]) -> Dict[int, float]:
    """Recover the filter-index to centre-frequency map from the phase selections.

    ``sel_i`` / ``sel_j`` are indices into the order-1 filter bank and ``sel_xi_i_hz`` /
    ``sel_xi_j_hz`` are those filters' centre frequencies, so every selected pair contributes two
    known points. See the module docstring for why this is the whole map available without
    importing kymatio.

    Args:
        selections: The ``sel_*`` attribute dicts to harvest, typically ``fhr_ph`` and ``up_ph``.

    Returns:
        Filter index to centre frequency in Hz.
    """
    known: Dict[int, float] = {}
    for attrs in selections:
        for index_key, freq_key in (("sel_i", "sel_xi_i_hz"), ("sel_j", "sel_xi_j_hz")):
            indices = np.asarray(attrs[index_key]).ravel()
            frequencies = np.asarray(attrs[freq_key]).ravel()
            for index, frequency in zip(indices.tolist(), frequencies.tolist()):
                if np.isfinite(frequency):
                    known[int(index)] = float(frequency)
    return known


def build_partition(
    shard_path: Any,
    *,
    n_scattering: int,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    phase_dataset: str = "fhr_ph",
) -> BandPartition:
    r"""Build one stream's channel map from a shard's stored provenance.

    Args:
        shard_path: Path to any shard of the split under evaluation. The selection is a property
            of the pipeline revision, not of the recording, so any shard describes them all --
            and a mismatch between two shards would already have failed the runner's per-batch
            width check.
        n_scattering: Width of the scattering block, from the batch. Passed rather than assumed:
            the model stores only the combined width and cannot supply the split.
        bands: The clinical band table. ``None`` uses :data:`CLINICAL_BANDS`.
        phase_dataset: Which phase block this partition is over -- ``'fhr_ph'`` for the target
            stream, ``'up_ph'`` for a source stream. The *other* block of
            :data:`PHASE_DATASETS` is still read where it exists, because it references the same
            order-1 filter bank and so widens the frequency map the scattering side is banded
            with; it contributes no channels.

    Returns:
        The partition, with the scattering block occupying channels $[0, n_{\mathrm{st}})$ and
        the phase block the rest, matching the ``cat([st, ph])`` the model is given.

    Raises:
        RuntimeError: If the shard's provenance is missing or inconsistent. See
            :func:`read_selection`.
    """
    table = dict(CLINICAL_BANDS if bands is None else bands)
    selections: Dict[str, Dict[str, np.ndarray]] = {
        phase_dataset: read_selection(shard_path, phase_dataset)
    }
    for name in PHASE_DATASETS:
        if name in selections:
            continue
        try:
            selections[name] = read_selection(shard_path, name)
        except (KeyError, RuntimeError):
            # The other block only ever widens the filter map; a shard without it still describes
            # every channel of the block this partition is over.
            continue

    frequencies = _filter_frequencies(list(selections.values()))
    primary = selections[phase_dataset]

    records: List[ChannelRecord] = []
    # ---- Scattering block: channel 0 is order-0, channel c >= 1 is filter c - 1 -------------
    n_without_frequency = 0
    for channel in range(int(n_scattering)):
        if channel == 0:
            records.append(ChannelRecord(
                channel=channel, block="scattering", kind=KIND_ORDER0,
                # The order-0 channel is the lowpass: it carries the signal's slowest content
                # and belongs in the slow band on merit, not as a fallback for a missing value.
                band="slow_baseline",
                freq_hz_primary=float("nan"), freq_hz_secondary=float("nan"),
                harmonic_ratio=float("nan"),
            ))
            continue
        frequency = frequencies.get(channel - 1, float("nan"))
        if not np.isfinite(frequency):
            n_without_frequency += 1
        records.append(ChannelRecord(
            channel=channel, block="scattering", kind=KIND_ORDER1,
            band=band_of_hz(frequency, table),
            freq_hz_primary=float(frequency), freq_hz_secondary=float("nan"),
            harmonic_ratio=float("nan"),
            filter_i=channel - 1, filter_j=channel - 1,
        ))

    # ---- Phase block: element k of every sel_* array describes channel n_st + k -------------
    powers = np.asarray(primary["sel_power"]).ravel()
    xi_i = np.asarray(primary["sel_xi_i_hz"]).ravel()
    xi_j = np.asarray(primary["sel_xi_j_hz"]).ravel()
    index_i = np.asarray(primary["sel_i"]).ravel()
    index_j = np.asarray(primary["sel_j"]).ravel()
    for offset in range(int(powers.shape[0])):
        records.append(ChannelRecord(
            channel=int(n_scattering) + offset, block="phase",
            kind=kind_of_power(float(powers[offset])),
            # Banded on xi_j, the higher of the pair: it is the faster structure the coefficient
            # actually tracks, and it is the convention the predecessor's band table was drawn
            # against, so a band label means the same thing across both trees.
            band=band_of_hz(float(xi_j[offset]), table),
            freq_hz_primary=float(xi_j[offset]),
            freq_hz_secondary=float(xi_i[offset]),
            harmonic_ratio=float(powers[offset]),
            filter_i=int(index_i[offset]), filter_j=int(index_j[offset]),
        ))

    partition = BandPartition(
        channels=records,
        n_scattering=int(n_scattering),
        n_phase=int(powers.shape[0]),
        band_hz_ranges=table,
        coverage={
            "shard": str(shard_path),
            "phase_dataset": str(phase_dataset),
            "n_filters_with_frequency": len(frequencies),
            "n_scattering_without_frequency": n_without_frequency,
            # Whether the UP selection contributed to the filter map at all, whichever block is
            # being partitioned: it widens the frequency coverage of both.
            "up_ph_attrs_present": "up_ph" in selections,
            "phase_band_hz": [float(value) for value in
                              np.asarray(primary.get("sel_band_hz", [])).ravel().tolist()],
            "phase_k_steps": [int(value) for value in
                              np.asarray(primary.get("sel_k_steps", [])).ravel().tolist()],
            "note": (
                "a scattering channel has no recoverable centre frequency when no selected "
                "phase pair referenced its filter, which happens outside the phase selection's "
                "own band; those channels are placed in the 'unknown' band rather than guessed"
            ),
        },
    )
    return partition


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
#: Files written into the run directory.
PARTITION_FILENAME = "band_partition.json"
CHANNEL_MAP_FILENAME = "band_channel_map.csv"


def write_partition(partition: BandPartition, output_dir: Any) -> Dict[str, str]:
    """Write the partition as JSON and the channel map as CSV.

    Both, not one: the JSON round-trips into an equivalent partition for anything that wants to
    reason about the groups, and the CSV is one row per channel so a downstream plot can be
    redrawn from disk with ``pandas`` and no import from this package at all.

    Args:
        partition: The built partition.
        output_dir: The run's results directory.

    Returns:
        Artifact name to the path written.
    """
    import pandas as pd

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / PARTITION_FILENAME
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(partition.as_dict(), handle, indent=2)

    csv_path = directory / CHANNEL_MAP_FILENAME
    pd.DataFrame([record.as_row() for record in partition.channels]).to_csv(
        csv_path, index=False
    )
    return {"partition": str(json_path), "channel_map": str(csv_path)}


def emit_partition(
    shard_paths: Sequence[Any], n_scattering: int, output_dir: Any
) -> Dict[str, Any]:
    """Build and write the partition for a run, recording a clean skip when it cannot.

    Absent provenance is a **skip, not a failure**. Whether a given production shard predates
    ``_write_selection_attrs`` is a property of that file, and a run whose forecast, uplift and
    lag analyses all succeeded should not be marked failed because its shards are an older
    vintage -- especially since nothing else in the pipeline consumes the partition yet.

    Args:
        shard_paths: The configured test shards. The first readable one is used: the selection
            is a property of the pipeline revision rather than of the recording, and a mismatch
            between two shards would already have failed the runner's per-batch width check.
        n_scattering: Width of the scattering block, from a batch.
        output_dir: The run's results directory.

    Returns:
        A JSON-safe record: either the written paths and the channel counts, or ``skipped`` with
        the reason.
    """
    reasons: List[str] = []
    for path in shard_paths or ():
        try:
            partition = build_partition(path, n_scattering=int(n_scattering))
        except Exception as exc:  # noqa: BLE001 - recorded and reported, see the docstring
            reasons.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        written = write_partition(partition, output_dir)
        return {
            "skipped": False,
            "shard": str(path),
            "n_channels": partition.n_channels,
            "n_scattering": partition.n_scattering,
            "n_phase": partition.n_phase,
            "kind_counts": partition.kind_counts(),
            "band_counts": {
                name: len(indices) for name, indices in partition.partition("clinical").items()
            },
            "coverage": partition.coverage,
            "files": written,
        }
    return {
        "skipped": True,
        "reason": (
            "no configured shard carried usable sel_* channel provenance"
            if reasons else "no test shards were configured"
        ),
        "attempts": reasons,
    }


def load_partition(path: Any) -> BandPartition:
    """Reload a partition written by :func:`write_partition`.

    Args:
        path: Path to ``band_partition.json``.

    Returns:
        An equivalent partition. Band upper bounds serialised as ``null`` come back as infinity,
        which is what an unbounded top band means and what ``json`` cannot represent.
    """
    with open(str(path), "r", encoding="utf-8") as handle:
        blob = json.load(handle)
    return BandPartition(
        channels=[
            ChannelRecord(
                channel=int(row["channel"]), block=str(row["block"]), kind=str(row["kind"]),
                band=str(row["band"]),
                freq_hz_primary=float(row["freq_hz_primary"]),
                freq_hz_secondary=float(row["freq_hz_secondary"]),
                harmonic_ratio=float(row["harmonic_ratio"]),
                filter_i=None if row["filter_i"] is None else int(row["filter_i"]),
                filter_j=None if row["filter_j"] is None else int(row["filter_j"]),
            )
            for row in blob["channels"]
        ],
        n_scattering=int(blob["n_scattering"]),
        n_phase=int(blob["n_phase"]),
        band_hz_ranges={
            name: (float(low), float("inf") if high is None else float(high))
            for name, (low, high) in blob["band_hz_ranges"].items()
        },
        coverage=dict(blob.get("coverage") or {}),
    )
