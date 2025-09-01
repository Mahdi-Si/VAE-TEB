# Dataset Documentation

This document describes the end-to-end dataset pipeline for fetal heart rate (FHR) and uterine pressure (UP) signals: scattering features, phase harmonics, cross-channel phase harmonics, HDF5 storage, statistics, and normalized PyTorch loading.

## Overview

- **Signals**: FHR (bpm) and UP (mmHg), sampled at 4 Hz
- **Block length**: 5760 samples (≈24 minutes at 4 Hz)
- **Decimation**: 16× for feature sequences (5760 → 360)
- **Transforms**: Scattering (FHR), phase harmonics (FHR), cross-channel phase harmonics (UP→FHR)
- **Storage**: HDF5 with per-sample chunking and LZF compression
- **Normalization**: Statistics with transformation-aware normalization

## Signal Processing (`hdf5_dataset/kymatio_phase_scattering.py`)

We use `KymatioPhaseScattering1D` to compute scattering and phase features with carefully selected coefficients for FHR analysis.

- Configuration (defaults used by dataset creation):
  - `J=11`, `Q=4`, `T=16`, `max_order=1`, `oversampling=0`, `border_mode='reflect'`, `shape=5760`
  - Optional `tukey_alpha=None` (set to 0.1–0.25 to taper edges if desired)
- Boundary handling: iterative reflection padding consistent with Kymatio; exact unpadding via precomputed border indices
- Filters: Kymatio’s filter factory with correct `J_support`; low-pass `phi` and first-order wavelets `psi1`
- Phase correlations:
  - Build valid filter-pair indices with `xi_j ≥ xi_i`
  - Harmonic “acceleration” uses ratio `power = xi_j/xi_i`
  - Within-channel phase corr: accelerate `i`, conjugate-multiply with `j`, low-pass and decimate
  - Cross-channel phase corr: same computation between UP and FHR channels
  - Outputs are real-valued after smoothing

### Coefficient selection (automatic, FHR-focused)

- Within-channel (FHR):
  - Frequency threshold: `min_freq = 0.006 Hz`
  - Include auto-correlations (stability)
  - Include harmonic ratios `[2, 3]` with tolerance and `power ≤ 8`
  - Selected channels: 44
- Cross-channel (UP→FHR):
  - UP band: `< 0.02 Hz`
  - FHR band: `0.04–0.5 Hz`
  - Harmonic powers `1–32`
  - Selected channels: 130
- Scattering (FHR): first-order only
  - Channels: 43 (channel 0 is order-0 / low-pass; 1–42 are wavelet bands)

Notes:
- Selection counts are produced by `get_optimal_coefficients_for_fhr()` and used as masks during dataset creation.
- If `J/Q/T` change, counts may differ; the masks are recomputed accordingly.

## Dataset Creation (`hdf5_dataset/create_hdf5_dataset.py` → `append_sample`)

High-level steps for each record/segment:

1. Read and prepare signals via `EarlyMaestraMimoAdaptor` (transpose, padding with reflect, alignment, optional overlap, equalization). Blocks are constructed with `base_block_size=3840` and `shape=5760` and decimated by 16 for features.
2. Compute scattering and phase features on GPU/CPU:
   - Scattering: `compute_phase=True`, `phase_channels=[0]` (FHR)
   - Cross-channel: `compute_cross_phase=True`, `phase_channels=[0,1]` (FHR, UP)
3. Apply masks from the selection step:
   - Keep 44 FHR phase channels and 130 UP→FHR cross-phase channels
4. Filter out segments with excessive flat regions (strict flatness detection on FHR/UP)
5. Persist one sample per valid segment using `append_sample(...)`.

### HDF5 schema (shapes, dtypes)

Each sample uses per-sample chunking and LZF compression. Defaults: `len_signal=5760`, `len_sequence=360`.

- `fhr`: float32, `(len_signal,)`
- `up`: float32, `(len_signal,)`
- `fhr_st`: float32, `(43, len_sequence)`
- `fhr_ph`: float32, `(44, len_sequence)`
- `fhr_up_ph`: float32, `(130, len_sequence)`
- `target`: float32, `(len_sequence,)` (one-hot encoded classes per decimated step)
- `weight`: float32, `(len_sequence,)` (sample weights from adaptor)
- `epoch`: float32, `()` (scalar)
- `cs_label`: uint8, `()` (0/1)
- `bg_label`: uint8, `()` (0/1)
- `guid`: UTF-8 string, `()`

## Statistics and Normalization (`hdf5_dataset/calculate_dataset_stats.py`)

We compute statistics with transformation-aware accumulation to match the runtime normalization.

- Fields: `['fhr', 'up', 'fhr_st', 'fhr_ph', 'fhr_up_ph']`
- Optional trimming: `trim_minutes` (raw trimmed by `4*60*trim`, features by `/16`)
- Transform strategy during stats:
  - `fhr_st` (43 ch): channel 0 regular, channels 1–42 use `log(x + 1e-6)`
  - `fhr_ph` (44 ch): all channels `asinh(x)`
  - `fhr_up_ph` (130 ch): all channels `asinh(x)`
- Outputs: HDF5 stats file with per-field `mean`, `variance`, `std`, shape metadata, and channel lists (`regular_channels`, `log_channels`, `asinh_channels`) plus `log_epsilon`.
- Optional diagnostics: histogram plots (5–95th percentile) for raw vs normalized distributions per field/channel.

## PyTorch Dataset and Dataloader (`hdf5_dataset/hdf5_dataset.py`)

`CombinedHDF5Dataset` provides efficient loading with filtering and normalization:

- Multi-file loading with an index map; filter by GUIDs, `cs_label`, `bg_label`, epoch ranges, and by class presence in `target`
- Optional cache (default 2000 samples), pinned memory, device-friendly tensors
- Normalization (when `stats_path` is provided):
  - Reuses transformation-aware stats; applies `log` (scattering) and `asinh` (phase) then standardizes
  - Only transforms fields present in stats and optionally restricted via `normalize_fields`
- Shape adaptation for models: multi-channel features are transposed from `(channels, sequence)` to `(sequence, channels)` once in the loader
- Distributed training: optional `DistributedSampler` via `create_optimized_dataloader`

### Minimal usage

```python
from hdf5_dataset import CombinedHDF5Dataset, create_optimized_dataloader

dataset = CombinedHDF5Dataset(
    paths=["/path/to/data1.hdf5", "/path/to/data2.hdf5"],
    stats_path="/path/to/stats.hdf5",
    normalize_fields=["fhr_st", "fhr_ph", "fhr_up_ph"],
    cache_size=2000,
    pin_memory=True,
)

dataloader = create_optimized_dataloader(
    hdf5_files=["/path/to/data1.hdf5", "/path/to/data2.hdf5"],
    batch_size=32,
    num_workers=4,
    stats_path="/path/to/stats.hdf5",
)
```

## Key implementation details

- Reflect padding is applied iteratively to support large pad sizes; unpadding follows Kymatio’s border indices
- Low-pass smoothing and decimation are performed in the frequency domain; indices are clamped to avoid zero-length outputs
- Phase outputs are returned as real values post-smoothing
- Flat-region filtering removes segments with sustained constant values (conservative thresholds) to improve data quality
- All numeric arrays are stored as float32; labels are uint8; strings are UTF-8

## Exact selected channels (default J=11, Q=4, T=16, shape=5760)

The following snippet prints and saves the exact channel selections for both within-channel phase harmonics (FHR) and cross-channel phase harmonics (UP→FHR). It enumerates the output-channel index in the masked arrays (0-based), and the underlying filter-pair indices `(i_idx, j_idx)` with their center frequencies and harmonic power.

```python
import json
import torch
from hdf5_dataset.kymatio_phase_scattering import KymatioPhaseScattering1D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KymatioPhaseScattering1D(J=11, Q=4, T=16, shape=5760, device=device, max_order=1)

sel = model.get_optimal_coefficients_for_fhr(11, 4, 16)

# Phase selection (FHR)
phase_mask = sel['phase_selection']['optimal_mask']
phase_idx = torch.nonzero(phase_mask, as_tuple=False).squeeze(1)

phase_list = []
for out_ch, k in enumerate(phase_idx.tolist()):
    i = int(model.i_idx[k]); j = int(model.j_idx[k])
    phase_list.append({
        'out_channel': out_ch,              # 0..43 in fhr_ph
        'pair_index': k,                    # index in full pair list
        'i_idx': i,
        'j_idx': j,
        'freq_i': float(model.center_freqs[i].item()),
        'freq_j': float(model.center_freqs[j].item()),
        'power': float(model.powers[k].item())
    })

# Cross-channel selection (UP→FHR)
cross_mask = sel['cross_selection']['cross_mask']
cross_idx = torch.nonzero(cross_mask, as_tuple=False).squeeze(1)

cross_list = []
for out_ch, k in enumerate(cross_idx.tolist()):
    i = int(model.i_idx[k]); j = int(model.j_idx[k])
    cross_list.append({
        'out_channel': out_ch,              # 0..129 in fhr_up_ph
        'pair_index': k,
        'i_idx': i,                         # UP-filter index
        'j_idx': j,                         # FHR-filter index
        'freq_i': float(model.center_freqs[i].item()),
        'freq_j': float(model.center_freqs[j].item()),
        'power': float(model.powers[k].item())
    })

print(f"Phase channels selected: {len(phase_list)} (expected 44)")
print(f"Cross channels selected: {len(cross_list)} (expected 130)")

# Optional: save to JSON
with open('selected_phase_channels.json', 'w') as f:
    json.dump(phase_list, f, indent=2)
with open('selected_cross_channels.json', 'w') as f:
    json.dump(cross_list, f, indent=2)
```

Notes:
- The order of `out_channel` corresponds exactly to the channel axis in the saved HDF5 fields `fhr_ph` (44) and `fhr_up_ph` (130).
- The underlying pair indices `pair_index` correspond to the full pair list built from `(i_idx, j_idx)` where `xi_j ≥ xi_i`.
- If you change `J/Q/T` or `shape`, re-run the snippet to regenerate the lists.

## Data flow

```
Raw (FHR, UP @ 4 Hz, 5760)
  → Kymatio Phase Scattering (J=11, Q=4, T=16)
  → fhr_st (43, 360), fhr_ph (44, 360), fhr_up_ph (130, 360)
  → HDF5 (per-sample chunks, LZF)
  → Stats (log/asinh-aware, saved to HDF5)
  → CombinedHDF5Dataset (normalized tensors, ready for training)
```

## Notes and defaults

- Default counts: 43 scattering (FHR), 44 phase (FHR), 130 cross-phase (UP→FHR)
- If `J/Q/T` change, masks and counts are recomputed automatically
- Decimation is 16× by design (`T=16`, out length ≈ 360 for 5760 raw)
- Use `tukey_alpha` to reduce edge effects if required

## File organization

```
hdf5_dataset/
├── kymatio_phase_scattering.py   # Kymatio + phase harmonics + coefficient selection
├── create_hdf5_dataset.py        # Record reading, masking, filtering, HDF5 writing
├── hdf5_dataset.py               # HDF5 schema, PyTorch dataset, normalization
├── calculate_dataset_stats.py    # Stats computation + histogram visualization
└── test_hdf5_dataset.py          # Validation utilities
```

This implementation produces compact, physiologically-informed features with clear normalization, optimized I/O, and straightforward training integration.