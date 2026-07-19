r"""Generate the committed tiny HDF5 shard and stats file used by the lag-attention smoke run.

The training entry point wires a real dataloader to real HDF5. Without a committed fixture the
advertised smoke command cannot run at all on a fresh clone, and the entry point's load-bearing
call order is never exercised end to end. This writes a 4-sample shard carrying the real field
names, channel counts and decimation geometry -- only the sample count is small.

The shard is deliberately NOT built through ``hdf5_dataset/new_pipeline/create_new_pipeline.py``:
that module imports ``early_maestra`` and a kymatio scattering package which are not importable
here, and running the real scattering transform to produce four samples of noise would be a lot of
machinery for no extra fidelity. The *stats* file, by contrast, IS produced by the real calculator:
the dataset reader silently disables normalization on any stats-schema mismatch, so hand-rolling
that half is the one shortcut with a genuinely bad failure mode.

Run from the repo root:

    python scripts/make_tiny_shard.py

Geometry. Trimming removes ``int(4*60*trim_minutes)`` raw samples and that ``// 16`` decimated
samples from *each* end, so with ``trim_minutes=1.0``:

    on-disk len_sequence = 300 + 2 * 15 = 330      -> batch T   = 300
    on-disk len_signal   = 16 * 330    = 5280      -> batch raw = 4800

which matches the shipped config's ``sequence_length: 300``.
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

# Run from the repo root; this makes the script work when invoked as `python scripts/...` too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdf5_dataset.calculate_dataset_stats import calculate_and_save_dataset_stats  # noqa: E402

#: Minutes trimmed from each end. Matches the shipped config, so the fixture exercises the real
#: trim path rather than a geometry no production run uses.
TRIM_MINUTES = 1.0

#: Decimation factor from the 4 Hz raw grid to the feature grid.
DECIMATION = 16

#: Channel counts, all real: FHR scattering, FHR phase-harmonic, UP scattering, UP self-phase, and
#: the FHR-UP cross-phase block. The model reads the first four; the fifth exists because the stats
#: calculator computes over it and an absent field would just be skipped, leaving the fixture unable
#: to catch a cross-phase regression later.
#:
#: **Source of truth: `hdf5_dataset/new_pipeline/create_new_pipeline.py`.** The two self-phase
#: widths follow `PHASE_HARMONIC_K_STEPS` there (currently `(4, 6, 8)` -> fhr_ph 66, up_ph 15;
#: `(0, 4, 6, 8)` would give 94 and 26). Nothing in the test suite checks this fixture against the
#: production pipeline, so if that constant changes and this dict does not, the suite stays green
#: against a shard certifying the wrong geometry and the first real symptom is a width error on a
#: prod box. Re-run this script whenever that selection changes.
CHANNELS = {"fhr_st": 43, "fhr_ph": 66, "up_st": 43, "up_ph": 15, "fhr_up_ph": 79}

#: Scattering fields whose channels 1..C-1 are log-transformed at normalization time. Their samples
#: must be strictly positive: a negative value is clamped to 0 and becomes log(1e-6) ~ -13.8, which
#: is finite but nothing like real data.
LOG_FIELDS = ("fhr_st", "up_st")


def write_shard(path: str, *, n_samples: int, seq_len: int, seed: int) -> None:
    """Write the shard.

    Every dataset the reader touches is written, including four the model never reads:
    ``guid``, ``epoch``, ``cs_label`` and ``bg_label`` are read unconditionally by the dataset's
    index builder to derive the sample count and apply its filters. Because that builder catches its
    own exceptions and only warns, omitting one of them does not raise -- it yields an empty index
    and then the misleading ``ValueError("No samples match the specified filters.")``.

    Args:
        path: Destination ``.hdf5`` path. Overwritten if present.
        n_samples: Number of samples to write.
        seq_len: On-disk (untrimmed) feature-grid length.
        seed: Seed for the sample values, so the fixture is byte-reproducible.
    """
    rng = np.random.default_rng(seed)
    signal_len = seq_len * DECIMATION
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with h5py.File(path, "w", libver="latest") as handle:
        # Raw 4 Hz signals. Scaled to plausible FHR/UP magnitudes so the stats are not degenerate.
        handle.create_dataset(
            "fhr", data=(140.0 + 10.0 * rng.standard_normal((n_samples, signal_len))).astype("f4"),
            compression="lzf",
        )
        handle.create_dataset(
            "up", data=(30.0 + 10.0 * rng.standard_normal((n_samples, signal_len))).astype("f4"),
            compression="lzf",
        )

        # Feature grids, stored (N, C, T); the dataset transposes to (T, C) per sample on read.
        for field, n_channels in CHANNELS.items():
            values = rng.standard_normal((n_samples, n_channels, seq_len))
            if field in LOG_FIELDS:
                values = np.abs(values) + 0.1
            handle.create_dataset(field, data=values.astype("f4"), compression="lzf")

        # Per-step validity. All-valid keeps the fixture's masking trivial, so a masking bug shows
        # up as a shape or dtype error rather than as a plausible-looking number.
        handle.create_dataset(
            "weight", data=np.ones((n_samples, seq_len), dtype="f4"), compression="lzf"
        )
        handle.create_dataset(
            "target", data=np.zeros((n_samples, seq_len), dtype="f4"), compression="lzf"
        )

        # Index-builder fields. `epoch` is seconds before delivery, and must clear whatever
        # epoch_min/epoch_max the config sets or the index comes back empty.
        #
        # This was -50000.0, chosen to clear the old `epoch_max: -48000`. That filter was itself
        # a bug -- it selected zero segments from any real dataset, whose extraction floor is
        # MIN_DOMAIN_START_DATASET = -44640 -- so the fixture had been shaped around it and was
        # unreachable in production. -20000 s (~5.6 h before delivery) sits inside the real
        # window, which is what a fixture that means to exercise the real filters needs.
        handle.create_dataset(
            "epoch", data=np.full((n_samples,), -20000.0, dtype="f4"), compression="lzf"
        )
        handle.create_dataset(
            "cs_label", data=np.zeros((n_samples,), dtype="u1"), compression="lzf"
        )
        handle.create_dataset(
            "bg_label", data=np.zeros((n_samples,), dtype="u1"), compression="lzf"
        )
        # Variable-length UTF-8, uncompressed: h5py cannot chunk-compress a vlen string dataset.
        handle.create_dataset(
            "guid",
            data=[f"TINY_{i:03d}" for i in range(n_samples)],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def main() -> None:
    """Write the shard and its stats file."""
    default_dir = os.path.join(_REPO_ROOT, "teb_vae", "lag_attn", "tests", "fixtures")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=default_dir, help="Directory receiving both files.")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to write.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=330,
        help="On-disk feature length before trimming (330 -> 300 after trim_minutes=1.0).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed, so the fixture is reproducible.")
    args = parser.parse_args()

    shard_path = os.path.join(args.out_dir, "tiny_shard.hdf5")
    stats_path = os.path.join(args.out_dir, "tiny_stats.hdf5")

    write_shard(shard_path, n_samples=args.samples, seq_len=args.seq_len, seed=args.seed)
    print(f"wrote {shard_path}")

    # The real calculator, so the stats schema cannot drift from what the reader expects.
    calculate_and_save_dataset_stats(
        [shard_path], stats_path, trim_minutes=TRIM_MINUTES, plot_histograms=False, device="cpu"
    )
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
