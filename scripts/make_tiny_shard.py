r"""Generate the committed tiny HDF5 shards and stats files used by the lag-attention smoke runs.

The training entry point wires a real dataloader to real HDF5. Without a committed fixture the
advertised smoke command cannot run at all on a fresh clone, and the entry point's load-bearing
call order is never exercised end to end. This writes 4-sample shards carrying the real field
names, channel counts and decimation geometry -- only the sample count is small.

Four variants, and the first two are built very differently on purpose.

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

The causal shard is written with the phase-harmonic legs **envelope-aligned**, which is what the
shipped model configuration expects and what its ``causal_leg_alignment`` root attribute records.
The exact invocation that produced the two committed causal binaries is

    python scripts/make_tiny_shard.py --variants causal --leg-alignment envelope

and the unaligned variant stays one flag away (``--leg-alignment none``) so a comparison arm can be
built without editing anything.

**Causal multi-cohort** (``causal_cohort``) is the causal mode again, written into one shard per
canonical subgroup with real clinical class codes, the two label axes and a labour-onset column --
the cohort structure the evaluation pipeline's every by-class and by-subgroup readout needs, and
which the single-file causal shard (one all-zero ``target``, one GUID per row) cannot supply.
The coefficients are the *same* real ones: the bank runs once and each shard selects rows from it,
so nothing about the file's self-description changes and no block is synthesised. What it adds is
the cohort assignment, a validity profile with fractional edges and a gap, and distinct identities.

Its ceiling is stated rather than discovered: the committed raw fixture holds **eight** segments,
so eight subgroup shards at three recordings of two segments each re-use segments under distinct
identities. That is legitimate for a fixture whose assertions are about counts, cohorts, shapes,
denominators and refusals -- and it is why no test may assert the sign, magnitude or significance
of any clinical effect on it. These shards are generated per test session rather than committed,
so ``causal_cohort`` is deliberately absent from the default variant list.

**Planted** (``tiny_shard_causal_planted.hdf5``) is the causal mode over segments this script
synthesises rather than reads, in which the FHR modulation is a deterministic function of the UP
envelope a known number of steps earlier. It exists because real signal cannot answer one question:
whether a lag readout can recover a delay it is known to be looking at. The plant is at the RAW
level and nowhere else -- the same real bank, the same production writer, the same statistics
calculator -- because a hand-written coefficient block would carry a fabricated warm-up boundary,
which is the one thing no mode here does. The build re-measures its own coupling from the written
coefficients and refuses to report success without it, so an instrument the bank did not carry is
found while the file is being written rather than after a model has been trained against it. The
invocation that produced the committed binaries is

    python scripts/make_tiny_shard.py --variants planted

and ``--check-planted <path>`` re-runs that measurement alone against a shard already on disk.

The *stats* file is produced by the real calculator in every mode: the dataset reader silently
disables normalization on any stats-schema mismatch, so hand-rolling that half is the one shortcut
with a genuinely bad failure mode. On the causal variants the calculator additionally excludes each
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

# Run from the repo root; this makes the script work when invoked as `python scripts/...` too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdf5_dataset.calculate_dataset_stats import calculate_and_save_dataset_stats  # noqa: E402
from hdf5_dataset.causal_scattering import LEG_ALIGNMENT_MODES  # noqa: E402
from hdf5_dataset.hdf5_dataset import RAW_SAMPLING_HZ  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args  # noqa: E402

#: Minutes trimmed from each end. Matches the shipped config, so the fixture exercises the real
#: trim path rather than a geometry no production run uses.
TRIM_MINUTES = 1.0

#: Decimation factor from the 4 Hz raw grid to the feature grid. The grid itself is
#: ``RAW_SAMPLING_HZ``, imported rather than restated as ``4.0``: the synthesised segments below are
#: written in seconds and a hardcoded rate would keep placing their frequencies against a grid the
#: shards no longer have.
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


def read_causal_source_count() -> int:
    """Return how many real raw segments the committed fixture holds.

    Read rather than written down, because it is the ceiling on every causal fixture built here and
    a stale constant would silently under- or over-draw. It is currently eight.

    Returns:
        The stored segment count.

    Raises:
        FileNotFoundError: If the committed raw fixture is missing.
    """
    if not os.path.exists(CAUSAL_SOURCE):
        raise FileNotFoundError(
            f"the committed raw-signal fixture is missing from {CAUSAL_SOURCE}; it is tracked in "
            f"git, and no causal shard can be synthesised without it"
        )
    with h5py.File(CAUSAL_SOURCE, "r") as handle:
        return int(handle["fhr"].shape[0])


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


def causal_transform(
    raw: Dict[str, np.ndarray], seq_len: int, leg_alignment: str = LEG_ALIGNMENT_MODES[0]
) -> Dict[str, Any]:
    """Run the real causal bank over a batch of raw segments, once.

    Shared by every causal mode rather than written once each, and the sharing is the point: what a
    causal shard claims about itself is a property of *this* bank, so a second description of the
    filter bank would be a second warm-up boundary. Every caller therefore gets its coefficients,
    its channel plan and its widths from one place.

    **The segments are an argument rather than a read**, which is the one difference from how this
    began. Two of the three modes below transform the committed raw fixture and one synthesises its
    own pair, and the two halves are separable: where the signal came from decides nothing about
    the transform, while the transform decides everything about what the file may claim. Fusing
    them would have made the synthesised mode either re-implement the bank or write its segments to
    disk first.

    There is no seed: the result is a deterministic function of the segments handed in, the filter
    bank and the leg alignment, which is what makes it re-derivable rather than merely reproducible.

    The alignment is threaded into the mask resolution *and* the transform from one argument, so
    the coefficients a caller writes and the ``causal_leg_alignment`` the writer stamps beside them
    cannot describe two different operators.

    Args:
        raw: ``{'fhr': (n, L) float32, 'up': (n, L) float32}`` on the $4$ Hz grid, from
            :func:`read_causal_source` or :func:`planted_raw_pair`.
        seq_len: On-disk (untrimmed) feature-grid length. The bank is sized from
            ``seq_len * DECIMATION``, so the segments must already be that long.
        leg_alignment: The phase-harmonic leg alignment, one of
            :data:`~hdf5_dataset.causal_scattering.LEG_ALIGNMENT_MODES`.

    Returns:
        ``{'pipeline', 'masks', 'widths', 'raw', 'blocks', 'signal_len'}`` -- the production
        pipeline module reached through the import shim, the resolved masks and channel plan, the
        per-block widths, the raw segments and their four coefficient blocks.

    Raises:
        ValueError: If the segments are not ``seq_len * DECIMATION`` samples long. The bank is
            sized from the signal length, so the two cannot differ -- and the mismatch would
            otherwise surface as a shape error several frames deeper.
    """
    import torch

    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy
    from hdf5_dataset.smoke_check_channel_selection import _import_pipeline

    pipeline = _import_pipeline()
    signal_len = seq_len * DECIMATION
    for name in ("fhr", "up"):
        if int(raw[name].shape[1]) != signal_len:
            raise ValueError(
                f"the {name!r} segments are {int(raw[name].shape[1])} samples long but the "
                f"requested geometry needs {signal_len}; the causal bank is sized from the signal "
                f"length, so the two cannot differ"
            )

    device = torch.device("cpu")
    masks = pipeline.compute_scattering_masks(
        signal_len,
        scattering_T=DECIMATION,
        device=device,
        transform="causal",
        leg_alignment=leg_alignment,
    )
    blocks = transform_batch_numpy(
        CausalTorchBank(masks["causal_bank"], device, n_signal=signal_len),
        raw["fhr"],
        raw["up"],
        pipeline._selection_pairs(masks["fhr_ph_selection"]),
        pipeline._selection_pairs(masks["up_ph_selection"]),
        plan=masks["channel_plan"],
        leg_alignment=leg_alignment,
    )
    return {
        "pipeline": pipeline,
        "masks": masks,
        "widths": pipeline.resolve_channel_layout(masks),
        "raw": raw,
        "blocks": blocks,
        "signal_len": signal_len,
    }


def create_causal_file(path: str, transformed: Dict[str, Any], seq_len: int) -> None:
    """Create one empty causal HDF5 through the production writer.

    The schema, the root constants and the per-block warm-up, delay and novelty attributes are all
    the production writer's, reached through the import shim that stubs the prod-only adaptor.
    Nothing about the file's self-description is restated here, because a second description of a
    warm-up boundary is a second boundary. The leg alignment and the novelty vectors travel in the
    resolved masks for the same reason: they come from the bank that produced the coefficients.

    Args:
        path: Destination ``.hdf5`` path. Overwritten if present.
        transformed: The return value of :func:`causal_transform`.
        seq_len: On-disk (untrimmed) feature-grid length.
    """
    pipeline, masks, widths = (
        transformed["pipeline"], transformed["masks"], transformed["widths"]
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pipeline.create_initial_hdf5(
        path,
        len_signal=transformed["signal_len"],
        len_sequence=seq_len,
        fhr_ph_selection=masks["fhr_ph_selection"],
        n_fhr_st_channels=widths["fhr_st"],
        # None, not 0: the causal variant produces no cross-phase block, and the writer refuses a
        # width here rather than creating a dataset that would stay empty for the whole build.
        n_cross_phase_channels=None,
        n_up_st_channels=widths["up_st"],
        up_ph_selection=masks["up_ph_selection"],
        transform="causal",
        channel_plan=masks["channel_plan"],
        leg_alignment=masks["causal_leg_alignment"],
        novelty_frac=masks["causal_novelty_frac"],
    )


def write_causal_shard(
    path: str, *, n_samples: int, seq_len: int, leg_alignment: str = LEG_ALIGNMENT_MODES[0]
) -> Dict[str, Any]:
    """Write the single-file causal shard, transforming real raw segments with the real causal bank.

    Args:
        path: Destination ``.hdf5`` path. Overwritten if present.
        n_samples: Segments to take from the committed raw fixture.
        seq_len: On-disk (untrimmed) feature-grid length.
        leg_alignment: The phase-harmonic leg alignment the phase blocks are built with, recorded
            on the file as ``causal_leg_alignment``.

    Returns:
        The resolved per-block widths, so a caller can report what it wrote.
    """
    transformed = causal_transform(
        read_causal_source(n_samples, seq_len * DECIMATION), seq_len, leg_alignment
    )
    blocks, raw = transformed["blocks"], transformed["raw"]
    create_causal_file(path, transformed, seq_len)

    ones = np.ones((n_samples, seq_len), dtype="f4")
    transformed["pipeline"].append_samples_batch(
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
    return transformed["widths"]


# =================================================================================================
# The causal multi-cohort mode
#
# The single causal shard above is one file, one GUID per row, an all-zero `target` and an all-ones
# `weight`. Every class-, subgroup- and cohort-aware path in the evaluation pipeline self-skips
# against it, so only the fallback branches would ever be exercised. This mode writes the same real
# coefficients into one shard per canonical subgroup, carrying real class codes and the clinical
# fields the cohort questions are asked in.
#
# It is NOT modelled on `teb_vae/lag_attn_rws/tests/conftest.py::write_multi_class_shards`, which
# synthesises its blocks from `rng.standard_normal`. That is right for a two-sided cell where only
# the shape matters and wrong here, for the reason this module's own docstring gives: a synthesised
# block would carry a fabricated `causal_warmup_steps`, make `target_warm_frac == 1.0` vacuous, and
# break the source-null control's premise that zero is the channel mean over the region the model
# reads. So the coefficients come from the real bank over real segments, and what this mode adds is
# the cohort assignment and nothing else.
#
# THE CEILING IS STATED RATHER THAN DISCOVERED. `hdf5_dataset/tests/data/causal_fixture.hdf5` holds
# EIGHT real raw segments, taken from one production shard. Eight subgroup shards at three GUIDs of
# two segments each need 48 rows, so segments are re-used under distinct identities. That is
# legitimate for a fixture whose assertions are about counts, cohorts, shapes, denominators and
# refusals -- and it is exactly why no test may assert the sign, magnitude or significance of any
# clinical effect on these shards: they are not a population.
# =================================================================================================

#: The eight canonical subgroup shards, each with the clinical class code and the two label axes it
#: carries. Written out as a table rather than parsed from the names, because the obvious substring
#: rules are both wrong: ``'healthy_no_bg_no_cs'.endswith('_cs')`` is ``True`` and
#: ``'_bg_' in 'healthy_no_bg_no_cs'`` is ``True``, so a generator built on them labels the doubly
#: negative subgroup positive on both axes -- and every by-label table then has one group.
#:
#: All eight rather than a subset: the class axis needs all three codes and the subgroup axis needs
#: more shards than classes, or the two groupings coincide and a bug that swapped them would be
#: invisible. The names are ``teb_vae.lag_attn.eval.labels.CANONICAL_SUBGROUPS``, restated here
#: because a data-generation script must not depend on the evaluation package.
COHORT_SUBGROUPS: Dict[str, Dict[str, int]] = {
    "healthy_no_bg_no_cs": {"code": 1, "cs": 0, "bg": 0},
    "healthy_no_bg_cs": {"code": 1, "cs": 1, "bg": 0},
    "healthy_bg_no_cs": {"code": 1, "cs": 0, "bg": 1},
    "healthy_bg_cs": {"code": 1, "cs": 1, "bg": 1},
    "acidosis_no_cs": {"code": 2, "cs": 0, "bg": 0},
    "acidosis_cs": {"code": 2, "cs": 1, "bg": 0},
    "hie_no_cs": {"code": 3, "cs": 0, "bg": 0},
    "hie_cs": {"code": 3, "cs": 1, "bg": 1},
}

#: Recordings per shard, and segments each recording contributes. Three rather than two, because
#: the shared rank tests exclude any group with fewer than ``stats.MIN_GROUP_SIZE = 3`` finite
#: values -- at two per shard every cohort is excluded and the by-subgroup and by-class tests could
#: only ever be exercised as skips. Two segments per GUID rather than one, because a GUID
#: contributing one segment aggregates to itself and the per-recording reduction is an identity.
COHORT_GUIDS_PER_SHARD = 3
COHORT_SEGMENTS_PER_GUID = 2

#: Decimated steps ``trim_minutes: 1.0`` removes from each end, and the weight profile written
#: inside what is left. The fractional steps sit INSIDE the trimmed window on purpose: at the
#: stored edges the trim would remove them and the class-recovery test they exist for would run on
#: uniformly valid data. The gap is what stops every mask assertion from holding vacuously.
#:
#: The profile is a property of the fixture and not of the recording -- the coefficients were
#: transformed from the full raw signal, which has no gap in it. That is the same compromise the
#: two-sided cells' generator makes, and it is safe for the same reason: `weight` is the validity
#: channel every mask reads, and nothing reads it as a claim about the signal.
COHORT_TRIM_STEPS = 15
COHORT_FRACTIONAL_STEPS = 4
COHORT_GAP_STEPS = 2

#: The fractional validity itself. At code 2 this stores ``target = 1.0``, which is exactly what a
#: fully valid healthy step stores -- the case that makes reading ``target`` directly wrong, and
#: the case the dataset's own ``label`` filter (exact float equality) silently drops.
COHORT_EDGE_WEIGHT = 0.5

#: Seconds before delivery the first written segment records, and the stride between consecutive
#: ones. Chosen so all 48 stay inside the real extraction window (the dataset floor is $-44640$ s)
#: and inside the shipped ``epoch_min: -48000`` filter, while spanning about ten hours -- a
#: trajectory binned by hour needs more than one bin to put things in, and a constant ``epoch``
#: would additionally give every segment of the split one tile grid.
COHORT_EPOCH_START = -44000.0
COHORT_EPOCH_STRIDE = 800.0

#: Seconds from a segment's own start to labour onset, where it is known at all.
COHORT_ONSET_OFFSET = 5400.0

#: The shard whose later recordings have no labour-onset time. NaN is what the real pipeline stores
#: wherever a GUID is absent from the onset table, and it must be preserved rather than dropped --
#: an all-finite column would make every onset test vacuous, and an all-NaN one would make every
#: trajectory test a skip.
COHORT_ONSET_MISSING_SHARD = "hie_no_cs"

#: Which of a shard's written segments the second stage begins at, as an index into its own epochs.
#: Placed INSIDE the span rather than before it so each shard's segments land on **both** sides of
#: onset: the positive half of that axis -- the part the second clock exists to show -- is
#: otherwise never exercised, and every assertion about it would hold vacuously.
#:
#: Every shard uses the same index, so the eight of them put their recordings in the same few
#: second-stage windows. That is deliberate: at three recordings per shard, per-class windows would
#: otherwise never reach the `MIN_GROUP_SIZE` floor and the run-level significance path would be a
#: skip on every window.
COHORT_SECOND_STAGE_INDEX = 2

#: The shard whose first recording carries an onset equal to delivery itself. That is what a
#: pipeline writes when it substitutes zero for a missing time, it passes a NaN filter, and it is
#: counted rather than excluded -- so the count has to be non-zero somewhere or the diagnostic that
#: reports it is never exercised.
COHORT_SECOND_STAGE_SENTINEL_SHARD = "hie_cs"


def cohort_weight_profile(seq_len: int) -> np.ndarray:
    """Build the per-step validity profile every cohort sample shares.

    Args:
        seq_len: On-disk (untrimmed) feature-grid length.

    Returns:
        ``(seq_len,)`` float32: fractional at both ends of the *trimmed* window, an all-zero gap in
        the middle, fully valid elsewhere.
    """
    weight = np.ones(seq_len, dtype="f4")
    low = COHORT_TRIM_STEPS
    high = seq_len - COHORT_TRIM_STEPS
    weight[low : low + COHORT_FRACTIONAL_STEPS] = COHORT_EDGE_WEIGHT
    weight[high - COHORT_FRACTIONAL_STEPS : high] = COHORT_EDGE_WEIGHT
    middle = seq_len // 2
    weight[middle : middle + COHORT_GAP_STEPS] = 0.0
    return weight


def write_causal_cohort_shards(
    directory: str, *, seq_len: int, leg_alignment: str = LEG_ALIGNMENT_MODES[0]
) -> List[str]:
    """Write one causal shard per canonical subgroup, from the real bank over real segments.

    The bank runs ONCE over the committed raw fixture and every shard selects rows from its output,
    so all eight files carry the same real coefficients, the same channel plan and the same warm-up
    attributes -- and re-using a raw segment across shards costs nothing beyond the identity it is
    written under.

    **Segments are re-used, deliberately and unavoidably.** The committed raw fixture holds eight
    segments; eight subgroup shards at :data:`COHORT_GUIDS_PER_SHARD` recordings of
    :data:`COHORT_SEGMENTS_PER_GUID` segments each need forty-eight rows. Row selection is rotated
    per shard so no two shards carry the same six in the same order, and within one shard the rows
    stay distinct. What this fixture is therefore evidence about is counts, cohorts, shapes,
    denominators, identities and refusals -- never the sign, magnitude or significance of any
    clinical effect.

    Args:
        directory: Destination directory, created if absent. One ``<subgroup>.hdf5`` per entry of
            :data:`COHORT_SUBGROUPS` is written into it, overwriting any already there.
        seq_len: On-disk (untrimmed) feature-grid length.
        leg_alignment: The phase-harmonic leg alignment every written shard is built at and
            records; the eight files must agree, or the loader refuses the mixed list.

    Returns:
        The written shard paths, in :data:`COHORT_SUBGROUPS` order.
    """
    per_shard = COHORT_GUIDS_PER_SHARD * COHORT_SEGMENTS_PER_GUID
    available = read_causal_source_count()
    if per_shard > available:
        raise ValueError(
            f"COHORT_GUIDS_PER_SHARD * COHORT_SEGMENTS_PER_GUID = {per_shard} exceeds the "
            f"{available} real segments in {CAUSAL_SOURCE}, so a shard would repeat a raw segment "
            f"under two GUIDs of its own. Re-use ACROSS shards is deliberate and stated; re-use "
            f"WITHIN one is not, because it would make that shard's per-recording aggregation an "
            f"average of identical rows."
        )
    transformed = causal_transform(
        read_causal_source(available, seq_len * DECIMATION), seq_len, leg_alignment
    )
    blocks, raw = transformed["blocks"], transformed["raw"]
    pipeline = transformed["pipeline"]

    weight_row = cohort_weight_profile(seq_len)
    weight = np.tile(weight_row, (per_shard, 1))

    written: List[str] = []
    for offset, (subgroup, meta) in enumerate(COHORT_SUBGROUPS.items()):
        path = os.path.join(directory, f"{subgroup}.hdf5")
        create_causal_file(path, transformed, seq_len)

        # Rotated rather than fixed, so the eight shards do not all carry the same six segments in
        # the same order. Within one shard the rows stay distinct as long as `per_shard` does not
        # exceed the committed segment count, which the refusal above guarantees.
        rows = [(offset + index) % available for index in range(per_shard)]
        # Distinct across the whole set because the subgroup stem is: two shards cannot name the
        # same recording, which is what every per-recording aggregation groups on.
        guids = [
            f"{subgroup.upper()}_{index // COHORT_SEGMENTS_PER_GUID:03d}"
            for index in range(per_shard)
        ]
        # Distinct across the whole set too, and spanning about ten hours before delivery, so a
        # trajectory binned by hour has more than one bin to put things in.
        first = offset * per_shard
        epochs = np.array(
            [
                COHORT_EPOCH_START + COHORT_EPOCH_STRIDE * (first + index)
                for index in range(per_shard)
            ],
            dtype="f4",
        )

        onset = epochs + COHORT_ONSET_OFFSET
        if subgroup == COHORT_ONSET_MISSING_SHARD:
            onset[COHORT_SEGMENTS_PER_GUID:] = np.nan

        # The second clinical clock, stored exactly as the real pipeline stores it:
        # `domain_start - t_sso`, signed, negative before second-stage onset and positive after.
        second_stage = epochs - epochs[COHORT_SECOND_STAGE_INDEX]
        if subgroup == COHORT_ONSET_MISSING_SHARD:
            # The same recordings the labour-onset table has never heard of, so one shard tells one
            # story about a cohort with no clinical times rather than two.
            second_stage = np.full((per_shard,), np.nan, dtype="f4")
        elif subgroup == COHORT_SECOND_STAGE_SENTINEL_SHARD:
            # Implied onset = epoch - offset = 0: second stage recorded at delivery.
            second_stage[:COHORT_SEGMENTS_PER_GUID] = epochs[:COHORT_SEGMENTS_PER_GUID]

        pipeline.append_samples_batch(
            path,
            fhr_batch=raw["fhr"][rows],
            up_batch=raw["up"][rows],
            fhr_st_batch=blocks["fhr_st"][rows].astype("f4"),
            fhr_ph_batch=blocks["fhr_ph"][rows].astype("f4"),
            # The class code scaled by validity, exactly as the real pipeline stores it.
            target_batch=(float(meta["code"]) * weight).astype("f4"),
            weight_batch=weight,
            guid_batch=guids,
            epoch_batch=epochs,
            cs_label_batch=np.full((per_shard,), meta["cs"], dtype="u1"),
            bg_label_batch=np.full((per_shard,), meta["bg"], dtype="u1"),
            tlo_batch=onset.astype("f4"),
            second_stage_batch=second_stage.astype("f4"),
            up_st_batch=blocks["up_st"][rows].astype("f4"),
            up_ph_batch=blocks["up_ph"][rows].astype("f4"),
        )
        written.append(path)
    return written


# =================================================================================================
# The planted-delay mode
#
# The two causal modes above transform REAL signal, which is what makes their coefficients honest --
# and it is exactly why neither can answer "can this architecture recover a delay it is known to be
# looking at". Real FHR and UP have no ground-truth lag written down anywhere, so a lag readout run
# over them has nothing to be scored against and a null result is unattributable: the architecture
# may be blind, or the delay may not be there.
#
# This mode plants one. The FHR modulation is a deterministic function of the UP envelope
# PLANTED_DELAY_STEPS * DECIMATION raw samples earlier, and nothing else about the build changes:
# the same real bank produces the coefficients, the same production writer stamps the schema and the
# warm-up attributes, and the same statistics calculator excludes the same warm-up region. The plant
# is at the RAW level and only there. A coefficient block written by hand would carry a fabricated
# warm-up boundary -- a second boundary, which is the one thing this generator refuses everywhere.
#
# WHAT THE DELAY MEANS ON THE LAG AXIS, since the two numbers differ and both are load-bearing.
# Target content at stored step s is a function of source content at s - delta, so the SERIES lag
# between the two stored streams is delta, and that is what the cross-correlation self-check below
# measures. A model at anchor t forecasts target step t + 1 + h, whose source content sits at
# t + 1 + h - delta, i.e. at attention lag l = delta - 1 - h. Across h in [0, H) that is the band
# [delta - H, delta - 1], which is what a lag profile's peak must fall inside. Choosing
# delta in (H, L - 1) is exactly what makes that band non-empty, strictly inside the searched
# window, and clear of lag 0 -- so a profile pinned at the near edge is a failure rather than an
# ambiguity.
#
# THE PLANT IS NOT A SIMULATION. One delay, one direction, linear in the envelope. It is an
# identifiability instrument, and no test may read a physiological claim off it.
# =================================================================================================
#: The planted source-to-target delay, in decimated steps. $45$ against the shipped $H = 30$ and
#: $L - 1 = 90$ puts the readable band at lags $[15, 44]$: non-empty, clear of both censoring edges,
#: and clear of lag $0$ by fifteen bins.
PLANTED_DELAY_STEPS = 45

#: Segments the planted shard carries. Its own constant rather than the ``samples`` argument, like
#: the cohort mode's counts and for the same reason: the recovery check is stated over this
#: population, so a shard built at another count would answer a different question under the same
#: filename. Eight matches the committed raw fixture's count, so the two causal fixtures put the
#: same number of recordings in front of a model.
PLANTED_SAMPLES = 8

#: The slow envelope's components, in Hz. Four incommensurate frequencies rather than one, so the
#: envelope is APERIODIC: a periodic drive would put a cross-correlation peak at $\delta$ and
#: another at every $\delta \pm kP$, and a lag search wide enough to contain two of them could not
#: say which one it found. They sit in $0.004$-$0.013$ Hz, i.e. $75$-$270$ s, which is slow enough
#: to be an envelope and fast enough that its own smearing through a channel's group delay is small
#: against the planted $180$ s.
PLANTED_ENVELOPE_HZ: Tuple[float, ...] = (0.00375, 0.00611, 0.00893, 0.01249)

#: Frequency the envelope additionally amplitude-modulates, and how deeply. The envelope enters the
#: stored coefficients twice over: through the signal *level*, which the order-0 low-pass carries,
#: and through the modulus of this carrier's band, which the order-1 channels around it carry. Two
#: routes rather than one because which stored channel a scattering bank puts a given content in is
#: the bank's decision and not this generator's -- and the check below reports which channels
#: actually received it rather than assuming.
PLANTED_CARRIER_HZ = 0.09
PLANTED_CARRIER_DEPTH = 0.25

#: The uncoupled components of the target, in Hz, and the amplitude each carries in bpm. They exist
#: so the shard carries CONTROLS: a stored target channel in one of these bands is a function of
#: nothing the source has, so a cross-correlation that peaked there would be measuring the check
#: rather than the plant.
#:
#: Five of them, spread across the bank's fast half rather than one tone, and the spread is the
#: point twice over. A single tone leaves every other fast channel with almost no content at all,
#: and a correlation taken on a channel carrying nothing is noise reported at three decimal places;
#: and one control channel is a control that could itself be the accident. None of the five is a
#: harmonic of :data:`PLANTED_CARRIER_HZ` or of an envelope component, so a channel that hears one
#: of them hears nothing the source is modulating.
PLANTED_CONTROL_HZ: Tuple[float, ...] = (0.047, 0.076, 0.123, 0.199, 0.322)
PLANTED_CONTROL_AMPLITUDE = 4.0

#: Signal levels, in the units of the real fixture: UP in mmHg over a resting tone, FHR in bpm about
#: a baseline that the envelope *decelerates* -- so the coupling is negative, which is why the check
#: below ranks by absolute correlation.
PLANTED_UP_BASE = 5.0
PLANTED_UP_GAIN = 70.0
PLANTED_FHR_BASE = 145.0
PLANTED_FHR_GAIN = 35.0

#: Golden-ratio conjugate, used to spread the per-segment phases. A deterministic low-discrepancy
#: sequence rather than an RNG draw: :func:`causal_transform` has no seed because it is a function
#: of its inputs, and a generator feeding it would be the one part of the fixture that could not be
#: re-derived from the constants in this file.
_GOLDEN = 0.6180339887498949

#: On-disk steps skipped before the self-check correlates anything. The stored leading region is a
#: function of assumed pre-recording history on the slow channels, so a correlation taken across it
#: would be a correlation with the bank's own transient. $150$ clears the shipped anchor floor of
#: $134$ in trimmed coordinates plus the $15$-step trim itself.
PLANTED_SKIP_STEPS = 150

#: How far a measured series lag may sit from :data:`PLANTED_DELAY_STEPS` and still count as
#: recovery, in steps. Non-zero because a channel's modulus is smoothed by its own group delay, so
#: the peak of a smeared envelope moves by a fraction of it.
PLANTED_LAG_TOLERANCE_STEPS = 4

#: How strong the correlation at that lag must be for a channel to be called coupled, and how weak
#: it must be for one to be called a control. The gap between them is deliberate: a channel in
#: neither class is reported and belongs to neither, which is the honest reading of a partial
#: coupling.
PLANTED_MIN_ABS_CORRELATION = 0.5
PLANTED_CONTROL_MAX_ABS_CORRELATION = 0.2

#: The two stored blocks the self-check correlates. They are produced by the SAME one-sided bank
#: applied to the two signals, so channel $c$ of one has exactly channel $c$ of the other's composed
#: group delay -- which is what lets a matched-index pair measure the planted lag directly, with the
#: two group delays cancelling instead of biasing the peak by their difference.
PLANTED_SOURCE_BLOCK = "up_st"
PLANTED_TARGET_BLOCK = "fhr_st"


def planted_raw_pair(n_samples: int, signal_len: int, delay_steps: int) -> Dict[str, np.ndarray]:
    r"""Synthesise raw FHR/UP segments in which FHR follows UP at a known delay.

    $$e_n(t) = \frac{1}{K}\sum_k \tfrac12\bigl(1 + \sin(2\pi f_k t + \varphi_{k,n})\bigr)
      \in [0, 1],$$

    $$u_n(t) = u_0 + g_u\, e_n(t)\bigl(1 + m\cos(2\pi f_{\mathrm{car}} t)\bigr), \qquad
      y_n(t) = y_0 - g_y\, e_n(t - \Delta_{\mathrm{raw}})
             + a\cos(2\pi f_{\mathrm{ctl}} t + \psi_n).$$

    The envelope is evaluated at $t - \Delta_{\mathrm{raw}}$ **as a function**, not shifted out of a
    buffer, so the coupling holds from the first sample and the record carries no seam. A shifted
    buffer would need a fill for its leading region, and that fill would be a boundary the shard
    does not declare -- the same objection that keeps every coefficient here coming from the bank.

    There is no random number anywhere: the per-segment phases are a golden-ratio sequence in the
    segment index, so the whole fixture is a function of the constants above and re-derivable from
    them rather than from a seed.

    Args:
        n_samples: How many segments to synthesise.
        signal_len: Raw samples per segment, on the $4$ Hz grid.
        delay_steps: $\delta$, the planted delay in **decimated** steps; the raw shift is
            ``delay_steps * DECIMATION``.

    Returns:
        ``{'fhr': (n_samples, signal_len) float32, 'up': (n_samples, signal_len) float32}``.

    Raises:
        ValueError: If the delay is not positive, or if it reaches beyond the segment -- a plant
            longer than the record is one no lag search inside the record could find.
    """
    if int(delay_steps) <= 0:
        raise ValueError(
            f"delay_steps={delay_steps} must be positive: the plant is a source-to-target delay, "
            f"and a non-positive one asks the target to lead the source."
        )
    raw_shift = int(delay_steps) * DECIMATION
    if raw_shift >= signal_len:
        raise ValueError(
            f"delay_steps={delay_steps} is {raw_shift} raw samples against a {signal_len}-sample "
            f"segment, so the coupled source content lies outside every segment the model sees."
        )

    seconds = np.arange(signal_len, dtype=np.float64) / RAW_SAMPLING_HZ
    delayed = seconds - float(raw_shift) / RAW_SAMPLING_HZ

    fhr = np.empty((n_samples, signal_len), dtype=np.float64)
    up = np.empty((n_samples, signal_len), dtype=np.float64)
    for index in range(n_samples):
        envelope_now = _planted_envelope(seconds, index)
        envelope_then = _planted_envelope(delayed, index)
        carrier = 1.0 + PLANTED_CARRIER_DEPTH * np.cos(
            2.0 * np.pi * PLANTED_CARRIER_HZ * seconds
        )
        control = np.zeros_like(seconds)
        for order, frequency in enumerate(PLANTED_CONTROL_HZ):
            phase = 2.0 * np.pi * (((order + 2) * (index + 1) * _GOLDEN) % 1.0)
            control += PLANTED_CONTROL_AMPLITUDE * np.cos(
                2.0 * np.pi * frequency * seconds + phase
            )
        up[index] = PLANTED_UP_BASE + PLANTED_UP_GAIN * envelope_now * carrier
        fhr[index] = PLANTED_FHR_BASE - PLANTED_FHR_GAIN * envelope_then + control
    return {"fhr": fhr.astype("f4"), "up": up.astype("f4")}


def _planted_envelope(seconds: np.ndarray, index: int) -> np.ndarray:
    """The slow drive of one segment, in $[0, 1]$, at the given times.

    Args:
        seconds: Times to evaluate at. May be negative, which is what lets the delayed copy be
            evaluated rather than shifted.
        index: The segment index, which sets the component phases.

    Returns:
        The envelope, same shape as ``seconds``.
    """
    total = np.zeros_like(seconds, dtype=np.float64)
    for order, frequency in enumerate(PLANTED_ENVELOPE_HZ):
        phase = 2.0 * np.pi * (((order + 1) * (index + 1) * _GOLDEN) % 1.0)
        total += 0.5 * (1.0 + np.sin(2.0 * np.pi * frequency * seconds + phase))
    return total / float(len(PLANTED_ENVELOPE_HZ))


def _lag_correlations(source: np.ndarray, target: np.ndarray, max_lag: int) -> np.ndarray:
    r"""Pearson correlation of ``target`` against ``source`` shifted back, one value per lag.

    $$\rho_\ell = \operatorname{mean}_n \operatorname{corr}
      \bigl(y_n[\ell:],\; x_n[:T-\ell]\bigr), \qquad \ell = 0 \dots L-1 .$$

    Averaged over segments rather than pooled across them, so one long segment cannot decide the
    peak on its own; the segments are the same length here, so the two agree, and the mean is what
    generalises if they ever do not.

    Args:
        source: $(n, T)$ source series.
        target: $(n, T)$ target series, same shape.
        max_lag: The furthest lag to evaluate, inclusive.

    Returns:
        $(max\_lag + 1,)$ correlations. A lag whose overlap has zero variance in either series --
        a constant channel -- contributes ``NaN`` and is skipped by the mean.
    """
    length = int(source.shape[1])
    values = np.full(int(max_lag) + 1, np.nan, dtype=np.float64)
    for lag in range(int(max_lag) + 1):
        left, right = source[:, : length - lag], target[:, lag:]
        if left.shape[1] < 2:
            continue
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        scale = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_sample = np.where(scale > 0.0, (left * right).sum(axis=1) / scale, np.nan)
        if np.isfinite(per_sample).any():
            values[lag] = float(np.nanmean(per_sample))
    return values


def planted_lag_report(
    blocks: Dict[str, np.ndarray],
    *,
    delay_steps: int = PLANTED_DELAY_STEPS,
    max_lag: int = 2 * PLANTED_DELAY_STEPS,
    skip_steps: int = PLANTED_SKIP_STEPS,
) -> List[Dict[str, Any]]:
    r"""Measure, per matched channel pair, where the stored coefficients say the delay is.

    This is the check that separates *the instrument is broken* from *the architecture cannot
    recover it*, and it runs before any model exists. The plant is at the raw level; whether it
    survives a strictly one-sided bank -- whose composed group delays reach the same order as the
    delay itself -- is a property of the transform rather than of the plant, and asserting it
    without measuring it would make every later use of the fixture unfalsifiable.

    The pairing is by **matched index**, not by search: :data:`PLANTED_SOURCE_BLOCK` and
    :data:`PLANTED_TARGET_BLOCK` are the same bank over the two signals, so channel $c$ of each has
    the same composed delay and the pair's group delays cancel out of the measured lag. A pair drawn
    across blocks would measure $\delta + \kappa(\tau^y_{c'} - \tau^u_c)/\Delta$ and report the
    difference as though it were the plant.

    Args:
        blocks: The bank's output, as :func:`causal_transform` returns it.
        delay_steps: The delay that was planted, for the in-band verdict.
        max_lag: Furthest lag searched. Twice the plant by default, so a peak at the plant is
            interior rather than pinned at an edge of the *search*.
        skip_steps: Leading stored steps excluded; see :data:`PLANTED_SKIP_STEPS`.

    Returns:
        One record per channel: its index, the lag of its strongest absolute correlation, that
        correlation with its sign, and whether the channel is coupled, a control, or neither.

    Raises:
        ValueError: If the two blocks do not have the same channel count -- matched-index pairing
            would then compare two different filters and the cancellation argument would not hold.
    """
    source = np.asarray(blocks[PLANTED_SOURCE_BLOCK], dtype=np.float64)[:, :, skip_steps:]
    target = np.asarray(blocks[PLANTED_TARGET_BLOCK], dtype=np.float64)[:, :, skip_steps:]
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"{PLANTED_SOURCE_BLOCK} has {source.shape[1]} channels against "
            f"{PLANTED_TARGET_BLOCK}'s {target.shape[1]}. The self-check pairs them by index on "
            f"the ground that they are one bank over two signals, and unequal widths mean they "
            f"are not."
        )

    records: List[Dict[str, Any]] = []
    for channel in range(int(source.shape[1])):
        correlations = _lag_correlations(source[:, channel], target[:, channel], max_lag)
        if not np.isfinite(correlations).any():
            records.append(
                {
                    "channel": channel, "lag": None, "correlation": float("nan"),
                    "coupled": False, "control": False,
                }
            )
            continue
        lag = int(np.nanargmax(np.abs(correlations)))
        peak = float(correlations[lag])
        records.append(
            {
                "channel": channel,
                "lag": lag,
                "correlation": peak,
                "coupled": bool(
                    abs(lag - int(delay_steps)) <= PLANTED_LAG_TOLERANCE_STEPS
                    and abs(peak) >= PLANTED_MIN_ABS_CORRELATION
                ),
                "control": bool(abs(peak) <= PLANTED_CONTROL_MAX_ABS_CORRELATION),
            }
        )
    return records


def planted_geometry(records: Sequence[Dict[str, Any]], delay_steps: int) -> Dict[str, Any]:
    """Reduce the per-channel report to the geometry a shard stamps and a check script reads.

    Args:
        records: The report from :func:`planted_lag_report`.
        delay_steps: The planted delay in decimated steps.

    Returns:
        The stamped geometry: the delay, the matched channel indices found coupled and found
        control, and the strongest coupled correlation with the lag it sat at. Every entry is
        **measured** on the written coefficients rather than declared, so a bank change that stopped
        carrying the plant would move these numbers instead of leaving a stale claim on the file.
    """
    coupled = [record for record in records if record["coupled"]]
    controls = [record for record in records if record["control"]]
    best = max(coupled, key=lambda record: abs(record["correlation"]), default=None)
    return {
        "planted_delay_steps": int(delay_steps),
        "planted_delay_seconds": float(delay_steps) * DECIMATION / RAW_SAMPLING_HZ,
        "planted_coupled_channels": np.asarray(
            [record["channel"] for record in coupled], dtype="i4"
        ),
        "planted_control_channels": np.asarray(
            [record["channel"] for record in controls], dtype="i4"
        ),
        "planted_best_channel": -1 if best is None else int(best["channel"]),
        "planted_best_lag_steps": -1 if best is None else int(best["lag"]),
        "planted_best_correlation": float("nan") if best is None else float(best["correlation"]),
        "planted_source_block": PLANTED_SOURCE_BLOCK,
        "planted_target_block": PLANTED_TARGET_BLOCK,
    }


def format_planted_report(
    records: Sequence[Dict[str, Any]], geometry: Dict[str, Any]
) -> str:
    """Render the self-check as the lines an operator reads, with its pass/fail verdict.

    Args:
        records: The report from :func:`planted_lag_report`.
        geometry: The stamped geometry from :func:`planted_geometry`.

    Returns:
        One line per channel that is coupled or a control, then the verdict. Channels in neither
        class are counted rather than listed: they are the partially coupled middle, and a line
        each would bury the two classes the check is about.
    """
    delay = int(geometry["planted_delay_steps"])
    coupled = list(geometry["planted_coupled_channels"])
    controls = list(geometry["planted_control_channels"])
    lines = [
        f"planted-delay self-check: delta = {delay} steps "
        f"({geometry['planted_delay_seconds']:g} s), "
        f"{PLANTED_TARGET_BLOCK} against {PLANTED_SOURCE_BLOCK} at matched channel index, "
        f"tolerance +/-{PLANTED_LAG_TOLERANCE_STEPS} steps",
        f"  {'channel':>8}  {'lag':>5}  {'corr':>7}  class",
    ]
    for record in records:
        if not (record["coupled"] or record["control"]):
            continue
        lines.append(
            f"  {record['channel']:>8}  {str(record['lag']):>5}  {record['correlation']:>+7.3f}  "
            + ("coupled" if record["coupled"] else "control")
        )
    neither = len(records) - len(coupled) - len(controls)
    lines.append(
        f"  ({neither} channel(s) in neither class: correlation between "
        f"{PLANTED_CONTROL_MAX_ABS_CORRELATION} and {PLANTED_MIN_ABS_CORRELATION}, or peaking "
        f"outside the planted band)"
    )
    passed = bool(coupled) and bool(controls)
    lines.append(
        f"VERDICT: {'PASS' if passed else 'FAIL'} -- {len(coupled)} coupled channel(s) peaking "
        f"within {PLANTED_LAG_TOLERANCE_STEPS} steps of {delay}, {len(controls)} flat control "
        f"channel(s). "
        + (
            f"Strongest: channel {geometry['planted_best_channel']} at lag "
            f"{geometry['planted_best_lag_steps']}, r = {geometry['planted_best_correlation']:+.3f}."
            if passed
            else "The instrument does not carry the plant; nothing may be concluded from a model "
            "run against it."
        )
    )
    return "\n".join(lines)


def write_planted_shard(
    path: str,
    *,
    seq_len: int,
    n_samples: int = PLANTED_SAMPLES,
    delay_steps: int = PLANTED_DELAY_STEPS,
    leg_alignment: str = LEG_ALIGNMENT_MODES[0],
) -> Dict[str, Any]:
    """Write the planted-delay causal shard and stamp what was planted on it.

    Args:
        path: Destination ``.hdf5`` path. Overwritten if present.
        seq_len: On-disk (untrimmed) feature-grid length.
        n_samples: Segments to synthesise.
        delay_steps: The planted delay, in decimated steps.
        leg_alignment: The phase-harmonic leg alignment the phase blocks are built with.

    Returns:
        ``{'widths', 'records', 'geometry'}`` -- the resolved per-block widths, the per-channel
        self-check report and the geometry stamped on the file.
    """
    transformed = causal_transform(
        planted_raw_pair(n_samples, seq_len * DECIMATION, delay_steps), seq_len, leg_alignment
    )
    blocks, raw = transformed["blocks"], transformed["raw"]
    create_causal_file(path, transformed, seq_len)

    ones = np.ones((n_samples, seq_len), dtype="f4")
    transformed["pipeline"].append_samples_batch(
        path,
        fhr_batch=raw["fhr"],
        up_batch=raw["up"],
        fhr_st_batch=blocks["fhr_st"].astype("f4"),
        fhr_ph_batch=blocks["fhr_ph"].astype("f4"),
        # All-valid, like the other causal fixtures: a masking bug then shows up as a shape or
        # dtype error rather than as a plausible-looking number.
        target_batch=np.zeros((n_samples, seq_len), dtype="f4"),
        weight_batch=ones,
        # Distinct per segment, and they must be: the evaluation's controls pair a sample with a
        # STRANGER's source, and a shard whose rows all name one recording offers no stranger.
        guid_batch=[f"PLANTED_{index:03d}" for index in range(n_samples)],
        epoch_batch=np.full((n_samples,), EPOCH_SECONDS, dtype="f4"),
        cs_label_batch=np.zeros((n_samples,), dtype="u1"),
        bg_label_batch=np.zeros((n_samples,), dtype="u1"),
        tlo_batch=np.full((n_samples,), np.nan, dtype="f4"),
        second_stage_batch=np.full((n_samples,), np.nan, dtype="f4"),
        up_st_batch=blocks["up_st"].astype("f4"),
        up_ph_batch=blocks["up_ph"].astype("f4"),
    )

    records = planted_lag_report(blocks, delay_steps=delay_steps)
    geometry = planted_geometry(records, delay_steps)
    # Stamped after the writer has finished, as extra ROOT attributes beside the ones it wrote.
    # Nothing the pipeline stamps is touched: the schema, the warm-up vectors and the leg alignment
    # stay exactly what the bank and the writer said, and these sit alongside them so a check script
    # reads the planted geometry off the file rather than assuming it.
    with h5py.File(path, "r+") as handle:
        for name, value in geometry.items():
            handle.attrs[name] = value
    return {"widths": transformed["widths"], "records": records, "geometry": geometry}


def read_planted_geometry(path: str) -> Dict[str, Any]:
    """Read back the planted geometry a shard was stamped with.

    Args:
        path: The planted shard.

    Returns:
        The stamped attributes, with the two channel lists as ``numpy`` arrays.

    Raises:
        ValueError: If the file carries no planted geometry, which means it is one of the other
            causal variants and no delay was planted in it at all.
    """
    with h5py.File(path, "r") as handle:
        if "planted_delay_steps" not in handle.attrs:
            raise ValueError(
                f"{path} carries no 'planted_delay_steps' root attribute, so no delay was planted "
                f"in it. The recovery check is scored against a stamped plant; run this script "
                f"with --variants planted to build one."
            )
        return {name: handle.attrs[name] for name in handle.attrs if name.startswith("planted_")}


def self_check_planted_shard(path: str) -> Tuple[str, bool]:
    """Re-measure a written planted shard's coupling from its stored coefficients alone.

    Reads the blocks back off disk rather than re-transforming, so what is checked is the file a
    model will actually be given -- including the ``float32`` round trip the writer applies.

    Args:
        path: The planted shard.

    Returns:
        ``(report, passed)``.
    """
    geometry = read_planted_geometry(path)
    with h5py.File(path, "r") as handle:
        blocks = {
            name: np.asarray(handle[name][:], dtype=np.float64)
            for name in (PLANTED_SOURCE_BLOCK, PLANTED_TARGET_BLOCK)
        }
    records = planted_lag_report(blocks, delay_steps=int(geometry["planted_delay_steps"]))
    measured = planted_geometry(records, int(geometry["planted_delay_steps"]))
    passed = bool(len(measured["planted_coupled_channels"])) and bool(
        len(measured["planted_control_channels"])
    )
    return format_planted_report(records, measured), passed


#: Everything the entry point below can be pointed at, with the real defaults applied after the
#: merge rather than by argparse -- an argparse default would make the dict entry unreachable and
#: the operator's edit silent. Nothing here has to be filled in: the file runs as it stands and
#: writes the three COMMITTED variants into the committed fixtures directory.
RUN_ARGS: Dict[str, Any] = {
    # Directory receiving the files; None -> teb_vae/lag_attn/tests/fixtures. Point this at a
    # temporary directory before asking for 'causal_cohort': those eight shards are generated per
    # test session and are deliberately not committed.
    "out_dir": None,
    # Which variants to write: any of VARIANT_CHOICES; None -> the three committed ones.
    # 'causal_cohort' is not in that default: it writes eight files that no test reads from disk,
    # because the evaluation suite generates them into tmp_path_factory.
    "variants": None,
    # Samples per shard; None -> 4. Read by 'two_sided' and 'causal' only: 'causal_cohort' and
    # 'planted' derive their own counts from their own constants, which is what their assertions
    # are stated in.
    "samples": None,
    # On-disk feature length before trimming; None -> 330, which trims to 300.
    "seq_len": None,
    # Seed for the two-sided shard's synthesised values; None -> 0. No causal mode has a seed: all
    # three are deterministic functions of their segments and the filter bank.
    "seed": None,
    # Phase-harmonic leg alignment for the causal variants: 'envelope' or 'none'; None ->
    # 'envelope', which is what the committed binaries carry and what the shipped model configs
    # expect. Ignored by 'two_sided', which has no causal phase block at all.
    "leg_alignment": None,
    # Re-measure an already-written planted shard's coupling instead of writing anything, given its
    # path. This is the self-check run on its own, for a fixture already on disk.
    "check_planted": None,
}

#: Defaults applied after the merge, so every key above stays reachable from the dict.
_DEFAULTS: Dict[str, Any] = {
    "out_dir": os.path.join(_REPO_ROOT, "teb_vae", "lag_attn", "tests", "fixtures"),
    "variants": ["two_sided", "causal", "planted"],
    "samples": 4,
    "seq_len": 330,
    "seed": 0,
    "leg_alignment": "envelope",
}

VARIANT_CHOICES = ("two_sided", "causal", "causal_cohort", "planted")

#: Where each variant's shards and statistics file land, relative to ``out_dir``. The cohort mode
#: names no shard stem: it writes one file per entry of :data:`COHORT_SUBGROUPS`.
_STATS_STEM: Dict[str, str] = {
    "two_sided": "tiny_stats.hdf5",
    "causal": "tiny_stats_causal.hdf5",
    "causal_cohort": "tiny_stats_causal_cohort.hdf5",
    "planted": "tiny_stats_causal_planted.hdf5",
}

#: The planted variant's shard stem, written out because the recovery check's configs name it.
PLANTED_SHARD_STEM = "tiny_shard_causal_planted.hdf5"


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
        help=(
            "Which shards to write; the three committed ones by default. 'causal_cohort' writes "
            "eight subgroup shards and is meant for a temporary --out-dir."
        ),
    )
    parser.add_argument("--samples", type=int, help="Number of samples to write.")
    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        type=int,
        help="On-disk feature length before trimming (330 -> 300 after trim_minutes=1.0).",
    )
    parser.add_argument("--seed", type=int, help="Seed for the two-sided shard's values.")
    parser.add_argument(
        "--leg-alignment",
        dest="leg_alignment",
        choices=LEG_ALIGNMENT_MODES,
        help=(
            "Phase-harmonic leg alignment for the causal variants; 'envelope' by default, which "
            "is what the committed binaries carry. 'none' builds the legacy comparison arm."
        ),
    )
    parser.add_argument(
        "--check-planted",
        dest="check_planted",
        help=(
            "Re-measure the coupling of an already-written planted shard and exit, writing "
            "nothing. Takes the shard's path."
        ),
    )
    return parser


def main(
    *,
    out_dir: str,
    variants: Sequence[str],
    samples: int,
    seq_len: int,
    seed: int,
    leg_alignment: str,
    check_planted: Optional[str] = None,
) -> int:
    """Write the requested shards and their stats files, or run the planted self-check alone.

    Args:
        out_dir: Directory receiving the files.
        variants: Which variants to write.
        samples: Samples per shard, for the two variants that take one.
        seq_len: On-disk feature length before trimming.
        seed: Seed for the two-sided shard's synthesised values.
        leg_alignment: Phase-harmonic leg alignment for the causal variants.
        check_planted: An existing planted shard to re-measure instead of writing anything.

    Returns:
        The process exit code. Non-zero only when a planted shard fails its own coupling check,
        which is the one outcome here that means a fixture must not be used.
    """
    if check_planted is not None:
        report, passed = self_check_planted_shard(check_planted)
        print(report)
        return 0 if passed else 1

    failed = False
    for variant in variants:
        stats = os.path.join(out_dir, _STATS_STEM[variant])
        if variant == "two_sided":
            shards = [os.path.join(out_dir, "tiny_shard.hdf5")]
            write_shard(shards[0], n_samples=samples, seq_len=seq_len, seed=seed)
        elif variant == "causal":
            shards = [os.path.join(out_dir, "tiny_shard_causal.hdf5")]
            widths = write_causal_shard(
                shards[0],
                n_samples=samples,
                seq_len=seq_len,
                leg_alignment=leg_alignment,
            )
            print(f"  causal widths: {widths}, leg alignment: {leg_alignment}")
        elif variant == "planted":
            shards = [os.path.join(out_dir, PLANTED_SHARD_STEM)]
            written = write_planted_shard(
                shards[0], seq_len=seq_len, leg_alignment=leg_alignment
            )
            print(f"  planted widths: {written['widths']}, leg alignment: {leg_alignment}")
            # Reported at build time, not only on demand: a shard whose plant the bank did not
            # carry is an instrument that would make every later model result unattributable, and
            # the moment to find that out is while the file is being written.
            print(format_planted_report(written["records"], written["geometry"]))
            failed = failed or not (
                len(written["geometry"]["planted_coupled_channels"])
                and len(written["geometry"]["planted_control_channels"])
            )
        else:
            shards = write_causal_cohort_shards(
                out_dir, seq_len=seq_len, leg_alignment=leg_alignment
            )
        for path in shards:
            print(f"wrote {path}")

        # The real calculator, so the stats schema cannot drift from what the reader expects. On
        # the causal variants it also applies the warm-up exclusion itself, which is what puts the
        # channel means on the region a model actually reads.
        #
        # ONE statistics file over the WHOLE cohort set, not one per shard: normalization constants
        # describe a dataset, and per-subgroup constants would z-score each cohort against its own
        # mean -- which is exactly the between-cohort difference every clinical contrast is about.
        calculate_and_save_dataset_stats(
            shards, stats, trim_minutes=TRIM_MINUTES, plot_histograms=False, device="cpu"
        )
        print(f"wrote {stats}")
    return 1 if failed else 0


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS`, then write what was asked for.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    # Per key, and only where nothing supplied a value: the merge above already preferred the
    # command line over the dict, and this is the third and last layer.
    values = {key: (_DEFAULTS.get(key) if value is None else value) for key, value in values.items()}
    # Relative paths -- an --out-dir above all -- resolve against the repository root rather than
    # whatever working directory an IDE chose, which is where every other path in this tree is
    # rooted.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    print(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(**values)


if __name__ == "__main__":
    sys.exit(_cli())
