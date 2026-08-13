r"""Generate the committed tiny HDF5 shards and stats files used by the lag-attention smoke runs.

The training entry point wires a real dataloader to real HDF5. Without a committed fixture the
advertised smoke command cannot run at all on a fresh clone, and the entry point's load-bearing
call order is never exercised end to end. This writes 4-sample shards carrying the real field
names, channel counts and decimation geometry -- only the sample count is small.

Two variants, and they are built very differently on purpose.

**Two-sided** (``tiny_shard.hdf5``) is deliberately NOT built through
``hdf5_dataset/new_pipeline/create_new_pipeline.py``: that module imports ``early_maestra`` and a
kymatio scattering package, and running the real two-sided transform to produce four samples of
noise would be a lot of machinery for no extra fidelity. Every block is synthesised from
``rng.standard_normal``, so its ``fhr``/``up`` and its coefficient blocks are unrelated.

**Causal** (``tiny_shard_causal.hdf5``) cannot be built that way, and the reason is the whole point
of the variant. What a causal shard claims about itself -- that a coefficient at step $t$ is a
function of $\{x(s) : s \le t\}$ and of nothing else, valid only past a per-channel warm-up -- is a
property of the *transform*, and synthesised blocks satisfy none of it. So the causal mode takes
real raw segments from the committed ``hdf5_dataset/tests/data/causal_fixture.hdf5`` and runs the
real :class:`~hdf5_dataset.causal_scattering_torch.CausalTorchBank` over them on the CPU. The
schema, the warm-up attributes and the channel plan all come from the production writer, reached
through the same import shim the dataset tests use, so nothing about the file's self-description is
restated here.

The *stats* file is produced by the real calculator in both modes: the dataset reader silently
disables normalization on any stats-schema mismatch, so hand-rolling that half is the one shortcut
with a genuinely bad failure mode. On the causal variant the calculator additionally excludes each
channel's warm-up region from its accumulation, which is what makes zero the channel mean over the
region a model actually reads.

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
from typing import Any, Dict, List, Optional

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
#:
#: The causal variant declares none of this: its widths come from the channel plan the production
#: resolver builds, which is the only thing that knows which scattering channels the warm-up rule
#: dropped.
CHANNELS = {"fhr_st": 43, "fhr_ph": 66, "up_st": 43, "up_ph": 15, "fhr_up_ph": 79}

#: Scattering fields whose channels 1..C-1 are log-transformed at normalization time. Their samples
#: must be strictly positive: a negative value is clamped to 0 and becomes log(1e-6) ~ -13.8, which
#: is finite but nothing like real data.
LOG_FIELDS = ("fhr_st", "up_st")

#: The committed raw segments the causal shard is transformed from, and how many of its eight rows
#: are used. Real production signal, so the coefficients the fixture stores are real coefficients.
CAUSAL_SOURCE = os.path.join(_REPO_ROOT, "hdf5_dataset", "tests", "data", "causal_fixture.hdf5")

#: Seconds before delivery every sample records. Inside the real extraction window
#: (``MIN_DOMAIN_START_DATASET = -44640``), so the fixture clears the shipped epoch filters rather
#: than being shaped around a filter no production run can satisfy.
EPOCH_SECONDS = -20000.0


def write_shard(path: str, *, n_samples: int, seq_len: int, seed: int) -> None:
    """Write the two-sided shard.

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
            "epoch", data=np.full((n_samples,), EPOCH_SECONDS, dtype="f4"), compression="lzf"
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


def read_causal_source(n_samples: int, signal_len: int) -> Dict[str, np.ndarray]:
    """Load the first ``n_samples`` real raw segments the causal shard is built from.

    Args:
        n_samples: Rows to take, from the top of the committed fixture.
        signal_len: Raw length the bank will be sized for; the fixture must already be it.

    Returns:
        ``{'fhr': (n, L) float32, 'up': (n, L) float32}``, exactly as stored.

    Raises:
        FileNotFoundError: If the committed raw fixture is missing.
        ValueError: If the fixture holds too few rows, or rows of the wrong length -- either would
            otherwise surface as a bank size mismatch several frames later.
    """
    if not os.path.exists(CAUSAL_SOURCE):
        raise FileNotFoundError(
            f"the committed raw-signal fixture is missing from {CAUSAL_SOURCE}; it is tracked in "
            f"git, and the causal shard cannot be synthesised without it -- synthesised blocks "
            f"satisfy none of the one-sidedness the variant exists to carry"
        )
    with h5py.File(CAUSAL_SOURCE, "r") as handle:
        stored = int(handle["fhr"].shape[0])
        if stored < n_samples:
            raise ValueError(
                f"{CAUSAL_SOURCE} holds {stored} segments but {n_samples} were asked for"
            )
        if int(handle["fhr"].shape[1]) != signal_len:
            raise ValueError(
                f"{CAUSAL_SOURCE} stores {int(handle['fhr'].shape[1])}-sample segments but the "
                f"requested geometry needs {signal_len}; the causal bank is sized from the signal "
                f"length, so the two cannot differ"
            )
        return {
            "fhr": np.asarray(handle["fhr"][:n_samples], dtype=np.float32),
            "up": np.asarray(handle["up"][:n_samples], dtype=np.float32),
        }


def write_causal_shard(path: str, *, n_samples: int, seq_len: int) -> Dict[str, Any]:
    """Write the causal shard, transforming real raw segments with the real causal bank.

    The schema, the root constants and the per-block warm-up and delay attributes are all the
    production writer's, reached through the import shim that stubs the prod-only adaptor. Nothing
    about the file's self-description is restated here, because a second description of a warm-up
    boundary is a second boundary.

    There is no seed: the file is a deterministic function of the committed raw segments and the
    filter bank, which is what makes it re-derivable rather than merely reproducible.

    Args:
        path: Destination ``.hdf5`` path. Overwritten if present.
        n_samples: Segments to take from the committed raw fixture.
        seq_len: On-disk (untrimmed) feature-grid length.

    Returns:
        The resolved per-block widths, so a caller can report what it wrote.
    """
    import torch

    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy
    from hdf5_dataset.smoke_check_channel_selection import _import_pipeline

    pipeline = _import_pipeline()
    signal_len = seq_len * DECIMATION
    raw = read_causal_source(n_samples, signal_len)

    device = torch.device("cpu")
    masks = pipeline.compute_scattering_masks(
        signal_len, scattering_T=DECIMATION, device=device, transform="causal"
    )
    plan = masks["channel_plan"]
    widths = pipeline.resolve_channel_layout(masks)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pipeline.create_initial_hdf5(
        path,
        len_signal=signal_len,
        len_sequence=seq_len,
        fhr_ph_selection=masks["fhr_ph_selection"],
        n_fhr_st_channels=widths["fhr_st"],
        # None, not 0: the causal variant produces no cross-phase block, and the writer refuses a
        # width here rather than creating a dataset that would stay empty for the whole build.
        n_cross_phase_channels=None,
        n_up_st_channels=widths["up_st"],
        up_ph_selection=masks["up_ph_selection"],
        transform="causal",
        channel_plan=plan,
    )

    blocks = transform_batch_numpy(
        CausalTorchBank(masks["causal_bank"], device, n_signal=signal_len),
        raw["fhr"],
        raw["up"],
        pipeline._selection_pairs(masks["fhr_ph_selection"]),
        pipeline._selection_pairs(masks["up_ph_selection"]),
        plan=plan,
    )

    ones = np.ones((n_samples, seq_len), dtype="f4")
    pipeline.append_samples_batch(
        path,
        fhr_batch=raw["fhr"],
        up_batch=raw["up"],
        fhr_st_batch=blocks["fhr_st"].astype("f4"),
        fhr_ph_batch=blocks["fhr_ph"].astype("f4"),
        # All-valid, like the two-sided fixture: a masking bug then shows up as a shape or dtype
        # error rather than as a plausible-looking number.
        target_batch=np.zeros((n_samples, seq_len), dtype="f4"),
        weight_batch=ones,
        guid_batch=[f"TINYC_{index:03d}" for index in range(n_samples)],
        epoch_batch=np.full((n_samples,), EPOCH_SECONDS, dtype="f4"),
        cs_label_batch=np.zeros((n_samples,), dtype="u1"),
        bg_label_batch=np.zeros((n_samples,), dtype="u1"),
        tlo_batch=np.full((n_samples,), np.nan, dtype="f4"),
        second_stage_batch=np.full((n_samples,), np.nan, dtype="f4"),
        up_st_batch=blocks["up_st"].astype("f4"),
        up_ph_batch=blocks["up_ph"].astype("f4"),
    )
    return widths


#: Everything the two entry points below can be pointed at, with the two-sided defaults applied
#: after the merge rather than by argparse -- an argparse default would make the dict entry
#: unreachable and the operator's edit silent. Nothing here has to be filled in: the file runs as
#: it stands and writes both variants into the committed fixtures directory.
RUN_ARGS: Dict[str, Any] = {
    # Directory receiving all four files; None -> teb_vae/lag_attn/tests/fixtures.
    "out_dir": None,
    # Which variants to write: any of 'two_sided', 'causal'; None -> both.
    "variants": None,
    # Samples per shard; None -> 4.
    "samples": None,
    # On-disk feature length before trimming; None -> 330, which trims to 300.
    "seq_len": None,
    # Seed for the two-sided shard's synthesised values; None -> 0. The causal shard has no seed:
    # it is a deterministic function of the committed raw segments and the filter bank.
    "seed": None,
}

#: Defaults applied after the merge, so every key above stays reachable from the dict.
_DEFAULTS: Dict[str, Any] = {
    "out_dir": os.path.join(_REPO_ROOT, "teb_vae", "lag_attn", "tests", "fixtures"),
    "variants": ["two_sided", "causal"],
    "samples": 4,
    "seq_len": 330,
    "seed": 0,
}

VARIANT_CHOICES = ("two_sided", "causal")


def build_parser() -> argparse.ArgumentParser:
    """The command line, with every default left at ``None``.

    A non-``None`` argparse default would be indistinguishable from a value the operator typed,
    which would make the matching :data:`RUN_ARGS` entry unreachable: the dict would be edited,
    nothing would change, and nothing would say why.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", dest="out_dir", help="Directory receiving the files.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        help="Which shards to write; both by default.",
    )
    parser.add_argument("--samples", type=int, help="Number of samples to write.")
    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        type=int,
        help="On-disk feature length before trimming (330 -> 300 after trim_minutes=1.0).",
    )
    parser.add_argument("--seed", type=int, help="Seed for the two-sided shard's values.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Write the requested shards and their stats files.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    parsed = vars(build_parser().parse_args(argv))
    # Per key: a value on the command line wins, then the dict, then the real default. Merging per
    # key rather than wholesale is what lets one flag override one value and leave the rest of the
    # dict standing.
    values = {
        key: parsed[key] if parsed[key] is not None else RUN_ARGS.get(key)
        for key in _DEFAULTS
    }
    values = {
        key: _DEFAULTS[key] if value is None else value for key, value in values.items()
    }

    for variant in values["variants"]:
        if variant == "two_sided":
            shard = os.path.join(values["out_dir"], "tiny_shard.hdf5")
            stats = os.path.join(values["out_dir"], "tiny_stats.hdf5")
            write_shard(
                shard,
                n_samples=values["samples"],
                seq_len=values["seq_len"],
                seed=values["seed"],
            )
        else:
            shard = os.path.join(values["out_dir"], "tiny_shard_causal.hdf5")
            stats = os.path.join(values["out_dir"], "tiny_stats_causal.hdf5")
            widths = write_causal_shard(
                shard, n_samples=values["samples"], seq_len=values["seq_len"]
            )
            print(f"  causal widths: {widths}")
        print(f"wrote {shard}")

        # The real calculator, so the stats schema cannot drift from what the reader expects. On
        # the causal variant it also applies the warm-up exclusion itself, which is what puts the
        # channel means on the region a model actually reads.
        calculate_and_save_dataset_stats(
            [shard], stats, trim_minutes=TRIM_MINUTES, plot_histograms=False, device="cpu"
        )
        print(f"wrote {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
