r"""Generate the committed tiny HDF5 shards and stats files used by the lag-attention smoke runs.

The training entry point wires a real dataloader to real HDF5. Without a committed fixture the
advertised smoke command cannot run at all on a fresh clone, and the entry point's load-bearing
call order is never exercised end to end. This writes 4-sample shards carrying the real field
names, channel counts and decimation geometry -- only the sample count is small.

Three variants, and the first two are built very differently on purpose.

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
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

# Run from the repo root; this makes the script work when invoked as `python scripts/...` too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hdf5_dataset.calculate_dataset_stats import calculate_and_save_dataset_stats  # noqa: E402
from hdf5_dataset.causal_scattering import LEG_ALIGNMENT_MODES  # noqa: E402

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
    n_samples: int, seq_len: int, leg_alignment: str = LEG_ALIGNMENT_MODES[0]
) -> Dict[str, Any]:
    """Run the real causal bank over the committed raw segments, once.

    Shared by both causal modes rather than written twice, and the sharing is the point: what a
    causal shard claims about itself is a property of *this* bank, so a second description of the
    filter bank would be a second warm-up boundary. Both callers therefore get their coefficients,
    their channel plan and their widths from one place.

    There is no seed: the result is a deterministic function of the committed raw segments, the
    filter bank and the leg alignment, which is what makes it re-derivable rather than merely
    reproducible.

    The alignment is threaded into the mask resolution *and* the transform from one argument, so
    the coefficients a caller writes and the ``causal_leg_alignment`` the writer stamps beside them
    cannot describe two different operators.

    Args:
        n_samples: Segments to take from the committed raw fixture.
        seq_len: On-disk (untrimmed) feature-grid length.
        leg_alignment: The phase-harmonic leg alignment, one of
            :data:`~hdf5_dataset.causal_scattering.LEG_ALIGNMENT_MODES`.

    Returns:
        ``{'pipeline', 'masks', 'widths', 'raw', 'blocks', 'signal_len'}`` -- the production
        pipeline module reached through the import shim, the resolved masks and channel plan, the
        per-block widths, the raw segments and their four coefficient blocks.
    """
    import torch

    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy
    from hdf5_dataset.smoke_check_channel_selection import _import_pipeline

    pipeline = _import_pipeline()
    signal_len = seq_len * DECIMATION
    raw = read_causal_source(n_samples, signal_len)

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
    transformed = causal_transform(n_samples, seq_len, leg_alignment)
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
    transformed = causal_transform(available, seq_len, leg_alignment)
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


#: Everything the two entry points below can be pointed at, with the two-sided defaults applied
#: after the merge rather than by argparse -- an argparse default would make the dict entry
#: unreachable and the operator's edit silent. Nothing here has to be filled in: the file runs as
#: it stands and writes the two COMMITTED variants into the committed fixtures directory.
RUN_ARGS: Dict[str, Any] = {
    # Directory receiving the files; None -> teb_vae/lag_attn/tests/fixtures. Point this at a
    # temporary directory before asking for 'causal_cohort': those eight shards are generated per
    # test session and are deliberately not committed.
    "out_dir": None,
    # Which variants to write: any of 'two_sided', 'causal', 'causal_cohort'; None -> the two
    # committed ones. 'causal_cohort' is not in that default: it writes eight files that no test
    # reads from disk, because the evaluation suite generates them into tmp_path_factory.
    "variants": None,
    # Samples per shard; None -> 4. Read by 'two_sided' and 'causal' only: 'causal_cohort' derives
    # its own count from COHORT_GUIDS_PER_SHARD * COHORT_SEGMENTS_PER_GUID, which is what its
    # cohort assertions are stated in.
    "samples": None,
    # On-disk feature length before trimming; None -> 330, which trims to 300.
    "seq_len": None,
    # Seed for the two-sided shard's synthesised values; None -> 0. Neither causal mode has a seed:
    # both are deterministic functions of the committed raw segments and the filter bank.
    "seed": None,
    # Phase-harmonic leg alignment for the two causal variants: 'envelope' or 'none'; None ->
    # 'envelope', which is what the committed binaries carry and what the shipped model configs
    # expect. Ignored by 'two_sided', which has no causal phase block at all.
    "leg_alignment": None,
}

#: Defaults applied after the merge, so every key above stays reachable from the dict.
_DEFAULTS: Dict[str, Any] = {
    "out_dir": os.path.join(_REPO_ROOT, "teb_vae", "lag_attn", "tests", "fixtures"),
    "variants": ["two_sided", "causal"],
    "samples": 4,
    "seq_len": 330,
    "seed": 0,
    "leg_alignment": "envelope",
}

VARIANT_CHOICES = ("two_sided", "causal", "causal_cohort")

#: Where each variant's shards and statistics file land, relative to ``out_dir``. The cohort mode
#: names no shard stem: it writes one file per entry of :data:`COHORT_SUBGROUPS`.
_STATS_STEM: Dict[str, str] = {
    "two_sided": "tiny_stats.hdf5",
    "causal": "tiny_stats_causal.hdf5",
    "causal_cohort": "tiny_stats_causal_cohort.hdf5",
}


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
            "Which shards to write; the two committed ones by default. 'causal_cohort' writes "
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
        stats = os.path.join(values["out_dir"], _STATS_STEM[variant])
        if variant == "two_sided":
            shards = [os.path.join(values["out_dir"], "tiny_shard.hdf5")]
            write_shard(
                shards[0],
                n_samples=values["samples"],
                seq_len=values["seq_len"],
                seed=values["seed"],
            )
        elif variant == "causal":
            shards = [os.path.join(values["out_dir"], "tiny_shard_causal.hdf5")]
            widths = write_causal_shard(
                shards[0],
                n_samples=values["samples"],
                seq_len=values["seq_len"],
                leg_alignment=values["leg_alignment"],
            )
            print(f"  causal widths: {widths}, leg alignment: {values['leg_alignment']}")
        else:
            shards = write_causal_cohort_shards(
                values["out_dir"],
                seq_len=values["seq_len"],
                leg_alignment=values["leg_alignment"],
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
