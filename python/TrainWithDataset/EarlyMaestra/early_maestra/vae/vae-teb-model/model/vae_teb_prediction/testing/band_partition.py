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

    def nonempty_bands(self) -> Tuple[str, ...]:
        """Return the bands that contain at least one channel."""
        return tuple(b for b in self.band_names if self.combined_idx[b].size > 0)

    def to_json(self) -> Dict[str, object]:
        """Return a JSON-serializable summary (without ``channel_metadata``)."""
        return {
            "band_names": list(self.band_names),
            "band_hz_ranges": {
                k: [float(v[0]), float(v[1]) if np.isfinite(v[1]) else None]
                for k, v in self.band_hz_ranges.items()
            },
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
        }

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

    # ---- channel_metadata DataFrame --------------------------------------
    rows = []
    # st channels
    for ch in range(n_st_channels):
        is_s0 = not np.isfinite(hz_st[ch])
        rows.append({
            "channel": ch,
            "kind": "st_S0" if is_s0 else "st_S1",
            "band": st_band_per_channel[ch],
            "freq_hz_primary": float("nan") if is_s0 else float(hz_st[ch]),
            "freq_hz_secondary": float("nan"),
            "harmonic_ratio": float("nan"),
        })
    # ph channels
    for k, row in ph_meta_df.iterrows():
        rows.append({
            "channel": int(n_st + k),
            "kind": str(row["kind"]),
            "band": str(row["band"]),
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
    "BandPartition",
    "build_band_partition",
]
