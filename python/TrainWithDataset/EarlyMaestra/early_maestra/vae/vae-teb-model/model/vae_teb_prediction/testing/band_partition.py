"""Channel-to-band partition for the 87-channel FHR forecast target.

The lag-attentive VAE-TEB v1 model forecasts an 87-channel FHR feature
trajectory built as ``cat(fhr_st, fhr_ph)`` along the channel axis.
``fhr_st`` (43 channels) holds wavelet scattering coefficients and
``fhr_ph`` (44 channels) holds within-channel phase-harmonic
correlations. Each channel is tied to one or two physical centre
frequencies (in Hz). This module partitions the 87 channels into the
clinically meaningful frequency bands documented in
``knowledge/dataset/scattering_phase_pipeline.md`` Section 8:

- ``slow_baseline`` (f < 0.008 Hz, period > 125 s)
- ``deceleration`` (0.008 ≤ f < 0.04 Hz, 25 s – 2 min)
- ``variability`` (0.04 ≤ f ≤ 0.25 Hz, 4 – 25 s)
- ``beat_to_beat`` (f > 0.25 Hz, < 4 s)

The mapping is reconstructed deterministically at runtime by
instantiating a temporary :class:`KymatioPhaseScattering1D` with the
same parameters the dataset was built with (``J=11, Q=4, T=16,
shape=5280``), reading its scattering meta and the phase-coefficient
selection rule (``select_fhr_phase_coefficients(min_freq=0.006)``).
Results are cached per ``(J, Q, T, shape, fhr_phase_min_freq)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D


# Canonical band order. Must match the legends and column order of the
# downstream visualizers and the per-class overlay plots.
BAND_NAMES: Tuple[str, ...] = (
    "slow_baseline",
    "deceleration",
    "variability",
    "beat_to_beat",
)

# Half-open frequency intervals [low, high) in Hz, except ``variability``
# which uses the inclusive upper bound 0.25 Hz to match Section 8 of
# scattering_phase_pipeline.md.
BAND_HZ_RANGES: Dict[str, Tuple[float, float]] = {
    "slow_baseline": (0.0, 0.008),
    "deceleration": (0.008, 0.04),
    "variability": (0.04, 0.25),
    "beat_to_beat": (0.25, float("inf")),
}

# Coefficient-kind partition: groups the 87 channels by *what they
# encode* rather than by frequency. Useful for asking "is the model
# better at predicting envelope (st_S1) than rhythm phase stability
# (ph_diag)?". S0 is always a singleton, ph_h3 may be empty for the
# canonical lag-attn v1 config.
KIND_NAMES: Tuple[str, ...] = (
    "st_S0", "st_S1", "ph_diag", "ph_h2", "ph_h3", "ph_other",
)

# Finer 7-band partition derived from the underlying scattering frequencies.
# All intervals are half-open ``[low, high)`` (the last band is the
# unbounded catch-all). Boundaries chosen from the actual ``xi*fs`` content
# of the lag-attn v1 forecast target:
#   * baseline / early / late split at 0.013 Hz separates ~early decel
#     (>75 s period) from late decel (25 - 75 s).
#   * variability split at 0.15 Hz follows fetal HRV LF / MF convention.
#   * 1 Hz cutoff isolates the highest octave near Nyquist (fs=4Hz).
REFINED7_BAND_NAMES: Tuple[str, ...] = (
    "baseline",
    "early_decel",
    "late_decel",
    "lf_var",
    "mf_var",
    "beat_to_beat",
    "nyquist_edge",
)

REFINED7_HZ_RANGES: Dict[str, Tuple[float, float]] = {
    "baseline":     (0.0,   0.008),
    "early_decel":  (0.008, 0.013),
    "late_decel":   (0.013, 0.04),
    "lf_var":       (0.04,  0.15),
    "mf_var":       (0.15,  0.25),
    "beat_to_beat": (0.25,  1.0),
    "nyquist_edge": (1.0,   float("inf")),
}

# Special label for the S0 (DC) scattering channel which has no centre
# frequency and therefore no octave bin.
OCTAVE_DC_LABEL: str = "octave_dc"


def _build_octave_ranges(fs: float, J: int) -> Dict[str, Tuple[float, float]]:
    """Build the ``octave_k -> (low_hz, high_hz)`` mapping.

    ``octave_k`` covers ``xi*fs`` in ``[fs * 2^-(k+1), fs * 2^-k)`` Hz,
    matching the structure of the kymatio J-octave wavelet bank.

    For ``fs=4, J=11`` the lowest octave (``octave_10``) covers roughly
    ``[0.00195, 0.00391)`` Hz. Channels with centre frequencies *below*
    this floor (rare) are assigned to ``octave_dc``; the S0 channel is
    also placed there because it carries no centre frequency.

    Args:
        fs: Sampling frequency in Hz.
        J: Number of octaves in the wavelet bank.

    Returns:
        ``OrderedDict``-like ``dict`` from octave label to ``(low, high)``
        in Hz, sorted from highest octave (highest frequency) downwards.
    """
    ranges: Dict[str, Tuple[float, float]] = {}
    for k in range(J):
        lo = float(fs) * (2.0 ** (-(k + 1)))
        hi = float(fs) * (2.0 ** (-k))
        ranges[f"octave_{k}"] = (lo, hi)
    return ranges


def _band_for_hz(freq_hz: float) -> str:
    """Return the band name a given centre frequency falls into.

    The variability band is inclusive on both ends to match the clinical
    convention ``[0.04, 0.25] Hz``; the boundary 0.25 itself maps to
    ``variability`` rather than ``beat_to_beat``.
    """
    if not np.isfinite(freq_hz):
        return "slow_baseline"
    if freq_hz < BAND_HZ_RANGES["slow_baseline"][1]:
        return "slow_baseline"
    if freq_hz < BAND_HZ_RANGES["deceleration"][1]:
        return "deceleration"
    if freq_hz <= BAND_HZ_RANGES["variability"][1]:
        return "variability"
    return "beat_to_beat"


def _refined7_for_hz(freq_hz: float) -> str:
    """Return the refined-7 band name for a given centre frequency.

    Uses purely half-open intervals ``[low, high)`` from
    :data:`REFINED7_HZ_RANGES`. Non-finite frequencies (S0 channel) map
    to ``baseline``.
    """
    if not np.isfinite(freq_hz):
        return "baseline"
    for band in REFINED7_BAND_NAMES:
        lo, hi = REFINED7_HZ_RANGES[band]
        if lo <= freq_hz < hi:
            return band
    # Catch-all: anything at or above the last band's lower bound.
    return REFINED7_BAND_NAMES[-1]


def _octave_for_hz(
    freq_hz: float,
    octave_ranges: Dict[str, Tuple[float, float]],
) -> str:
    """Return the octave label for a given centre frequency.

    Returns :data:`OCTAVE_DC_LABEL` for non-finite frequencies (S0) and
    for any frequency strictly below the lowest octave's lower bound.
    """
    if not np.isfinite(freq_hz):
        return OCTAVE_DC_LABEL
    for label, (lo, hi) in octave_ranges.items():
        if lo <= freq_hz < hi:
            return label
    # Frequencies above the topmost octave (xi*fs >= fs/2 = Nyquist) or
    # below the lowest octave fall outside the J-octave bank. Highest
    # case is impossible by construction (xi < 1.0); lowest case maps
    # to octave_dc so the partition stays well-defined.
    return OCTAVE_DC_LABEL


@dataclass
class BandPartition:
    """Channel-to-band mapping for an 87-channel FHR forecast target.

    Attributes:
        band_names: Fixed canonical order of bands.
        band_hz_ranges: ``band -> (low_hz, high_hz)``; high is exclusive
            for all bands except ``variability``.
        band_period_ranges_s: ``band -> (low_period_s, high_period_s)``,
            derived from ``band_hz_ranges`` (period = 1 / freq).
        n_st_channels: Number of scattering channels (typically 43).
        n_ph_channels: Number of within-FHR phase channels (typically 44).
        n_total: ``n_st_channels + n_ph_channels`` (87 in the v1 default).
        st_idx: ``band -> int array`` of indices into channels [0, n_st).
        ph_idx: ``band -> int array`` of indices into channels
            [n_st, n_st + n_ph) (already shifted into the 87-channel
            space; subtract ``n_st_channels`` to recover the position
            inside ``fhr_ph``).
        combined_idx: Union of ``st_idx`` and ``ph_idx`` per band, into
            channels [0, n_total).
        channel_metadata: One row per 87-channel index with columns
            ``[channel, kind, band, freq_hz_primary, freq_hz_secondary,
            harmonic_ratio]``.
    """

    band_names: Tuple[str, ...]
    band_hz_ranges: Dict[str, Tuple[float, float]]
    band_period_ranges_s: Dict[str, Tuple[float, float]]
    n_st_channels: int
    n_ph_channels: int
    n_total: int
    st_idx: Dict[str, np.ndarray]
    ph_idx: Dict[str, np.ndarray]
    combined_idx: Dict[str, np.ndarray]
    channel_metadata: pd.DataFrame = field(repr=False)

    # Coefficient-kind partition (st_S0, st_S1, ph_diag, ph_h2, ph_h3,
    # ph_other -> indices in [0, n_total)).
    kind_names: Tuple[str, ...] = field(default_factory=lambda: KIND_NAMES)
    kind_idx: Dict[str, np.ndarray] = field(default_factory=dict)

    # Refined 7-band partition: same channels, finer frequency tiles.
    refined7_band_names: Tuple[str, ...] = field(
        default_factory=lambda: REFINED7_BAND_NAMES
    )
    refined7_hz_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    refined7_idx: Dict[str, np.ndarray] = field(default_factory=dict)

    # Per-octave partition derived from the kymatio J-octave wavelet bank.
    octave_names: Tuple[str, ...] = field(default_factory=tuple)
    octave_hz_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    octave_idx: Dict[str, np.ndarray] = field(default_factory=dict)

    def nonempty_bands(self) -> Tuple[str, ...]:
        """Return the clinical-4 bands that contain at least one channel."""
        return tuple(b for b in self.band_names if self.combined_idx[b].size > 0)

    def nonempty_partition(self, partition: str) -> Tuple[str, ...]:
        """Return the labels of a named partition that have at least one channel.

        Args:
            partition: One of ``"clinical_4band"``, ``"clinical_7band"``,
                ``"by_kind"``, ``"by_octave"``.

        Returns:
            Tuple of label names in canonical order, filtered to those
            with at least one channel.
        """
        names, idx = self._partition(partition)
        return tuple(n for n in names if idx[n].size > 0)

    def _partition(
        self, partition: str,
    ) -> Tuple[Tuple[str, ...], Dict[str, np.ndarray]]:
        """Return ``(names, idx_dict)`` for a named partition."""
        if partition == "clinical_4band":
            return self.band_names, self.combined_idx
        if partition == "clinical_7band":
            return self.refined7_band_names, self.refined7_idx
        if partition == "by_kind":
            return self.kind_names, self.kind_idx
        if partition == "by_octave":
            return self.octave_names, self.octave_idx
        raise KeyError(
            f"Unknown partition {partition!r}. Use one of "
            f"'clinical_4band', 'clinical_7band', 'by_kind', 'by_octave'."
        )

    def partition_idx(self, partition: str) -> Dict[str, np.ndarray]:
        """Return the channel-index dict for a named partition."""
        return self._partition(partition)[1]

    def partition_names(self, partition: str) -> Tuple[str, ...]:
        """Return the canonical label tuple for a named partition."""
        return self._partition(partition)[0]

    def to_json(self) -> Dict[str, object]:
        """Return a JSON-serializable summary (without ``channel_metadata``)."""

        def _ranges_to_json(ranges: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
            return {
                k: [
                    float(v[0]),
                    float(v[1]) if np.isfinite(v[1]) else None,
                ]
                for k, v in ranges.items()
            }

        def _idx_to_lists(idx: Dict[str, np.ndarray]) -> Dict[str, list]:
            return {k: np.asarray(v, dtype=int).tolist() for k, v in idx.items()}

        out: Dict[str, object] = {
            "band_names": list(self.band_names),
            "band_hz_ranges": _ranges_to_json(self.band_hz_ranges),
            "band_period_ranges_s": {
                k: [
                    float(v[0]) if np.isfinite(v[0]) else None,
                    float(v[1]) if np.isfinite(v[1]) else None,
                ]
                for k, v in self.band_period_ranges_s.items()
            },
            "n_st_channels": int(self.n_st_channels),
            "n_ph_channels": int(self.n_ph_channels),
            "n_total": int(self.n_total),
            "channel_counts_per_band": {
                b: {
                    "st": int(self.st_idx[b].size),
                    "ph": int(self.ph_idx[b].size),
                    "combined": int(self.combined_idx[b].size),
                }
                for b in self.band_names
            },
            "st_indices": {b: self.st_idx[b].tolist() for b in self.band_names},
            "ph_indices_in_87_space": {
                b: self.ph_idx[b].tolist() for b in self.band_names
            },
            "combined_indices": {
                b: self.combined_idx[b].tolist() for b in self.band_names
            },
            # New: collect every available partition under a single key so
            # downstream tools can iterate uniformly without knowing which
            # partitions exist.
            "partitions": {
                "clinical_4band": {
                    "names": list(self.band_names),
                    "hz_ranges": _ranges_to_json(self.band_hz_ranges),
                    "idx": _idx_to_lists(self.combined_idx),
                },
                "clinical_7band": {
                    "names": list(self.refined7_band_names),
                    "hz_ranges": _ranges_to_json(self.refined7_hz_ranges),
                    "idx": _idx_to_lists(self.refined7_idx),
                },
                "by_kind": {
                    "names": list(self.kind_names),
                    "idx": _idx_to_lists(self.kind_idx),
                },
                "by_octave": {
                    "names": list(self.octave_names),
                    "hz_ranges": _ranges_to_json(self.octave_hz_ranges),
                    "idx": _idx_to_lists(self.octave_idx),
                },
            },
        }
        return out

    def write(self, output_dir: Path) -> Tuple[Path, Path]:
        """Persist the partition to ``band_partition.json`` and ``band_channel_map.csv``.

        Args:
            output_dir: Directory to write into (created if missing).

        Returns:
            Tuple ``(json_path, csv_path)``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "band_partition.json"
        csv_path = output_dir / "band_channel_map.csv"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2)
        self.channel_metadata.to_csv(csv_path, index=False)
        return json_path, csv_path


def _scattering_st_xi_array(
    *, J: int, Q: int, T: int, signal_length: int, device: torch.device,
) -> np.ndarray:
    """Return the per-channel normalised xi values for the fhr_st block.

    The first entry is ``np.nan`` (the order-0 / S_0 channel has no
    centre frequency), followed by the order-1 wavelets in the order
    Kymatio emits them in the scattering output array.
    """
    scattering_module = KymatioPhaseScattering1D(
        J=J,
        Q=Q,
        T=T,
        shape=signal_length,
        device=device,
        tukey_alpha=None,
        max_order=1,
    )
    meta = scattering_module.scattering.meta()
    order_arr = np.asarray(meta["order"]).reshape(-1)
    xi_meta = np.asarray(meta["xi"], dtype=float)
    # Kymatio's meta returns xi as (n_coeffs, max_order); for max_order=1
    # we want the first column. For older releases xi is 1D.
    if xi_meta.ndim == 2:
        xi_first = xi_meta[:, 0]
    else:
        xi_first = xi_meta
    xis = np.full_like(order_arr, fill_value=np.nan, dtype=float)
    for idx, ord_val in enumerate(order_arr):
        if int(ord_val) == 0:
            xis[idx] = float("nan")  # S_0 has no centre frequency
        else:
            xis[idx] = float(xi_first[idx])
    return xis


def _phase_channel_metadata(
    *, J: int, Q: int, T: int, signal_length: int,
    fhr_phase_min_freq: float, device: torch.device,
) -> pd.DataFrame:
    """Build the (i, j, p, ξ_i, ξ_j, kind) record for every fhr_ph channel.

    Iteration order matches ``select_fhr_phase_coefficients`` exactly,
    which is what the dataset was built with at creation time.
    """
    scattering_module = KymatioPhaseScattering1D(
        J=J,
        Q=Q,
        T=T,
        shape=signal_length,
        device=device,
        tukey_alpha=None,
        max_order=1,
    )
    sel = scattering_module.select_fhr_phase_coefficients(
        min_freq=fhr_phase_min_freq
    )

    optimal_mask = sel["optimal_mask"].cpu().numpy().astype(bool)
    autocorr_mask = sel["masks"].get("autocorr")
    h2_mask = sel["masks"].get("harmonic_2")
    h3_mask = sel["masks"].get("harmonic_3")

    autocorr_mask = (
        autocorr_mask.cpu().numpy().astype(bool)
        if autocorr_mask is not None else np.zeros_like(optimal_mask)
    )
    h2_mask = (
        h2_mask.cpu().numpy().astype(bool)
        if h2_mask is not None else np.zeros_like(optimal_mask)
    )
    h3_mask = (
        h3_mask.cpu().numpy().astype(bool)
        if h3_mask is not None else np.zeros_like(optimal_mask)
    )

    i_idx_all = scattering_module.i_idx.cpu().numpy()
    j_idx_all = scattering_module.j_idx.cpu().numpy()
    powers_all = scattering_module.powers.cpu().numpy()
    centre_freqs = scattering_module.center_freqs.cpu().numpy()

    rows = []
    for k_pair in np.where(optimal_mask)[0]:
        i = int(i_idx_all[k_pair])
        j = int(j_idx_all[k_pair])
        p = float(powers_all[k_pair])
        xi_i = float(centre_freqs[i])
        xi_j = float(centre_freqs[j])
        if autocorr_mask[k_pair]:
            kind = "ph_diag"
        elif h2_mask[k_pair]:
            kind = "ph_h2"
        elif h3_mask[k_pair]:
            kind = "ph_h3"
        else:
            kind = "ph_other"
        rows.append({
            "i": i,
            "j": j,
            "harmonic_ratio": p,
            "xi_i": xi_i,
            "xi_j": xi_j,
            "kind": kind,
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def _build_band_partition_cached(
    J: int,
    Q: int,
    T: int,
    signal_length: int,
    fhr_phase_min_freq: float,
    n_st_channels: int,
    n_ph_channels: int,
    fs: float,
) -> BandPartition:
    device = torch.device("cpu")

    # ---- fhr_st channels --------------------------------------------------
    xis_full = _scattering_st_xi_array(
        J=J, Q=Q, T=T, signal_length=signal_length, device=device,
    )
    n_meta = int(xis_full.size)
    if n_meta < n_st_channels:
        raise RuntimeError(
            f"Scattering meta returned {n_meta} channels but "
            f"n_st_channels={n_st_channels}: cannot build the band "
            f"mapping. Check J/Q/T/shape match the dataset config."
        )
    xis_st = xis_full[:n_st_channels]
    hz_st = xis_st * float(fs)

    st_band_per_channel = []
    for ch in range(n_st_channels):
        if not np.isfinite(hz_st[ch]):
            # S_0 is the local-mean channel and carries the lowest-frequency
            # information of the segment by construction.
            st_band_per_channel.append("slow_baseline")
        else:
            st_band_per_channel.append(_band_for_hz(float(hz_st[ch])))

    st_idx = {b: [] for b in BAND_NAMES}
    for ch, band in enumerate(st_band_per_channel):
        st_idx[band].append(ch)
    st_idx = {b: np.asarray(idx, dtype=int) for b, idx in st_idx.items()}

    # ---- fhr_ph channels -------------------------------------------------
    ph_meta_df = _phase_channel_metadata(
        J=J, Q=Q, T=T, signal_length=signal_length,
        fhr_phase_min_freq=fhr_phase_min_freq, device=device,
    )
    if len(ph_meta_df) < n_ph_channels:
        raise RuntimeError(
            f"select_fhr_phase_coefficients returned "
            f"{len(ph_meta_df)} channels but n_ph_channels="
            f"{n_ph_channels}; check fhr_phase_min_freq."
        )
    ph_meta_df = ph_meta_df.iloc[:n_ph_channels].reset_index(drop=True).copy()
    ph_meta_df["xi_j_hz"] = ph_meta_df["xi_j"].astype(float) * float(fs)
    ph_meta_df["xi_i_hz"] = ph_meta_df["xi_i"].astype(float) * float(fs)
    ph_meta_df["band"] = ph_meta_df["xi_j_hz"].apply(_band_for_hz)

    n_st = int(n_st_channels)
    ph_idx = {b: [] for b in BAND_NAMES}
    for k, band in enumerate(ph_meta_df["band"].tolist()):
        ph_idx[band].append(n_st + k)  # store as 87-channel-space index
    ph_idx = {b: np.asarray(idx, dtype=int) for b, idx in ph_idx.items()}

    combined_idx = {
        b: np.sort(np.concatenate([st_idx[b], ph_idx[b]])).astype(int)
        for b in BAND_NAMES
    }

    # ---- secondary partitions (kind / refined7 / octave) -----------------
    # Build per-channel labels first; aggregate into per-label index lists
    # afterwards so the loop logic stays O(n_total).
    octave_ranges = _build_octave_ranges(fs=float(fs), J=int(J))
    octave_names: Tuple[str, ...] = tuple(octave_ranges.keys()) + (OCTAVE_DC_LABEL,)
    octave_full_ranges: Dict[str, Tuple[float, float]] = dict(octave_ranges)
    octave_full_ranges[OCTAVE_DC_LABEL] = (0.0, float("inf"))

    kind_per_channel: list = [None] * (n_st_channels + n_ph_channels)
    refined7_per_channel: list = [None] * (n_st_channels + n_ph_channels)
    octave_per_channel: list = [None] * (n_st_channels + n_ph_channels)

    # st channels
    for ch in range(n_st_channels):
        if not np.isfinite(hz_st[ch]):
            kind_per_channel[ch] = "st_S0"
            refined7_per_channel[ch] = "baseline"
            octave_per_channel[ch] = OCTAVE_DC_LABEL
        else:
            kind_per_channel[ch] = "st_S1"
            refined7_per_channel[ch] = _refined7_for_hz(float(hz_st[ch]))
            octave_per_channel[ch] = _octave_for_hz(
                float(hz_st[ch]), octave_ranges,
            )

    # ph channels (channel index = n_st + k in 87-channel space)
    for k, row in ph_meta_df.iterrows():
        ch = int(n_st + k)
        kind_per_channel[ch] = str(row["kind"])
        refined7_per_channel[ch] = _refined7_for_hz(float(row["xi_j_hz"]))
        octave_per_channel[ch] = _octave_for_hz(
            float(row["xi_j_hz"]), octave_ranges,
        )

    kind_idx_lists: Dict[str, list] = {k: [] for k in KIND_NAMES}
    for ch in range(n_st_channels + n_ph_channels):
        kind_idx_lists[str(kind_per_channel[ch])].append(ch)
    kind_idx: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=int) for k, v in kind_idx_lists.items()
    }

    refined7_idx_lists: Dict[str, list] = {b: [] for b in REFINED7_BAND_NAMES}
    for ch in range(n_st_channels + n_ph_channels):
        refined7_idx_lists[str(refined7_per_channel[ch])].append(ch)
    refined7_idx: Dict[str, np.ndarray] = {
        b: np.asarray(v, dtype=int) for b, v in refined7_idx_lists.items()
    }

    octave_idx_lists: Dict[str, list] = {name: [] for name in octave_names}
    for ch in range(n_st_channels + n_ph_channels):
        octave_idx_lists[str(octave_per_channel[ch])].append(ch)
    octave_idx: Dict[str, np.ndarray] = {
        name: np.asarray(v, dtype=int) for name, v in octave_idx_lists.items()
    }

    # ---- channel_metadata DataFrame --------------------------------------
    rows = []
    # st channels
    for ch in range(n_st_channels):
        is_s0 = not np.isfinite(hz_st[ch])
        rows.append({
            "channel": ch,
            "kind": "st_S0" if is_s0 else "st_S1",
            "band": st_band_per_channel[ch],
            "refined_band": refined7_per_channel[ch],
            "octave": octave_per_channel[ch],
            "freq_hz_primary": float("nan") if is_s0 else float(hz_st[ch]),
            "freq_hz_secondary": float("nan"),
            "harmonic_ratio": float("nan"),
        })
    # ph channels
    for k, row in ph_meta_df.iterrows():
        ch_idx = int(n_st + k)
        rows.append({
            "channel": ch_idx,
            "kind": str(row["kind"]),
            "band": str(row["band"]),
            "refined_band": refined7_per_channel[ch_idx],
            "octave": octave_per_channel[ch_idx],
            "freq_hz_primary": float(row["xi_j_hz"]),
            "freq_hz_secondary": float(row["xi_i_hz"]),
            "harmonic_ratio": float(row["harmonic_ratio"]),
        })
    channel_metadata = pd.DataFrame(rows)

    # ---- period ranges ---------------------------------------------------
    band_period_ranges_s: Dict[str, Tuple[float, float]] = {}
    for band, (lo, hi) in BAND_HZ_RANGES.items():
        # period = 1 / freq; map (lo, hi) → (1/hi, 1/lo)
        low_p = (1.0 / hi) if (np.isfinite(hi) and hi > 0) else 0.0
        high_p = (1.0 / lo) if (lo > 0) else float("inf")
        band_period_ranges_s[band] = (low_p, high_p)

    return BandPartition(
        band_names=BAND_NAMES,
        band_hz_ranges=dict(BAND_HZ_RANGES),
        band_period_ranges_s=band_period_ranges_s,
        n_st_channels=int(n_st_channels),
        n_ph_channels=int(n_ph_channels),
        n_total=int(n_st_channels + n_ph_channels),
        st_idx=st_idx,
        ph_idx=ph_idx,
        combined_idx=combined_idx,
        channel_metadata=channel_metadata,
        kind_names=KIND_NAMES,
        kind_idx=kind_idx,
        refined7_band_names=REFINED7_BAND_NAMES,
        refined7_hz_ranges=dict(REFINED7_HZ_RANGES),
        refined7_idx=refined7_idx,
        octave_names=octave_names,
        octave_hz_ranges=octave_full_ranges,
        octave_idx=octave_idx,
    )


def build_band_partition(
    *,
    J: int = 11,
    Q: int = 4,
    T: int = 16,
    signal_length: int = 5280,
    fhr_phase_min_freq: float = 0.006,
    n_st_channels: int = 43,
    n_ph_channels: int = 44,
    fs: float = 4.0,
) -> BandPartition:
    """Construct the channel-to-band mapping for the 87-channel FHR target.

    Args:
        J: Wavelet bank octaves; matches the dataset config.
        Q: Wavelets per octave.
        T: Low-pass support / decimation factor.
        signal_length: Raw segment length in samples (5280 for v1).
        fhr_phase_min_freq: Frequency floor used by
            ``select_fhr_phase_coefficients`` at dataset creation time
            (0.006 Hz for v1).
        n_st_channels: Number of scattering channels stored in
            ``fhr_st`` (43 for v1).
        n_ph_channels: Number of within-channel phase channels stored
            in ``fhr_ph`` (44 for v1).
        fs: Sampling frequency in Hz (4 Hz for CTG).

    Returns:
        :class:`BandPartition` describing the per-band channel index
        lists and per-channel metadata.
    """
    return _build_band_partition_cached(
        int(J),
        int(Q),
        int(T),
        int(signal_length),
        float(fhr_phase_min_freq),
        int(n_st_channels),
        int(n_ph_channels),
        float(fs),
    )


__all__ = [
    "BAND_NAMES",
    "BAND_HZ_RANGES",
    "KIND_NAMES",
    "OCTAVE_DC_LABEL",
    "REFINED7_BAND_NAMES",
    "REFINED7_HZ_RANGES",
    "BandPartition",
    "build_band_partition",
]
