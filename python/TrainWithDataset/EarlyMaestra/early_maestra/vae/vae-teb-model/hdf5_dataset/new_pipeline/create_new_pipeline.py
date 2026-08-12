"""New HDF5 dataset creation pipeline for CTG classification and VAE pretraining.

Self-contained script that:
  1. Prescreens all GUIDs for valid signal in the last 6 hours.
  2. Selects GUIDs for classification with class balancing + TLO constraints.
  3. Creates 10-fold stratified CV splits (80/10/10).
  4. Builds classification HDF5 datasets (12.4h range, v3 scattering).
  5. Builds pretraining HDF5 datasets from BG subgroup leftovers.
"""

import os
import sys
import math
import json
import pickle
import random
import hashlib
import logging
import traceback
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

from early_maestra.adaptor.mimo_adaptor import EarlyMaestraMimoAdaptor
from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.kymatio_phase_scattering import (
    KymatioPhaseScattering1D,
)
# Absolute, with the production prefix, unlike every other module in this package: this file is
# launched as a script and so has no ``__package__`` for a relative import to resolve against.
from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.causal_scattering import (
    CAUSAL_KERNEL_TAPS,
    CAUSAL_WARMUP_QUANTILE,
    GAMMATONE_ORDER,
    CausalChannelPlan,
    build_causal_bank,
    build_channel_plan,
    build_filter_bank,
)
from Variational_AutoEncoder.seqvae_teb.hdf5_dataset.causal_scattering_torch import (
    CausalTorchBank,
    transform_batch_numpy,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
matplotlib.use("Agg")
torch.backends.cudnn.enabled = False

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Mappings
# ---------------------------------------------------------------------------
FOLDER_TO_SUBGROUP = {
    "ACIDOSIS_NO_HIE_CS": "acidosis_cs",
    "ACIDOSIS_NO_HIE_NoCS": "acidosis_no_cs",
    "HEALTHY_NO_ACIDOSIS_CS": "healthy_bg_cs",
    "HEALTHY_NO_ACIDOSIS_NoCS": "healthy_bg_no_cs",
    "HEALTHY_NO_BG_CS": "healthy_no_bg_cs",
    "HEALTHY_NO_BG_NoCS": "healthy_no_bg_no_cs",
    "HIE_CS": "hie_cs",
    "HIE_NoCS": "hie_no_cs",
}

CSV_NAME_TO_SUBGROUP = {
    "HIE_NoCS": "hie_no_cs",
    "HIE_CS": "hie_cs",
    "ACIDOSIS_NO_HIE_CS": "acidosis_cs",
    "ACIDOSIS_NO_HIE_NoCS": "acidosis_no_cs",
    "HEALTHY_NO_ACIDOSIS_CS": "healthy_bg_cs",
    "HEALTHY_NO_ACIDOSIS_NoCS": "healthy_bg_no_cs",
    "HEALTHY_NO_BG_CS": "healthy_no_bg_cs",
    "HEALTHY_NO_BG_NoCS": "healthy_no_bg_no_cs",
}

SUBGROUP_TO_FOLDER = {v: k for k, v in FOLDER_TO_SUBGROUP.items()}

UNHEALTHY_SUBGROUPS = {"acidosis_cs", "acidosis_no_cs", "hie_cs", "hie_no_cs"}
HEALTHY_SUBGROUPS = {
    "healthy_bg_cs",
    "healthy_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_no_bg_no_cs",
}
BG_SUBGROUPS = {"healthy_bg_cs", "healthy_bg_no_cs"}

# (pre_defined_target, cs_label, bg_label)
SUBGROUP_META = {
    "healthy_no_bg_no_cs": (1, False, False),
    "healthy_no_bg_cs": (1, True, False),
    "healthy_bg_cs": (1, True, True),
    "healthy_bg_no_cs": (1, False, True),
    "acidosis_cs": (2, True, True),
    "acidosis_no_cs": (2, False, True),
    "hie_cs": (3, True, True),
    "hie_no_cs": (3, False, True),
}

BASE_BLOCK_SIZE = 3520
OVERLAP_PERCENTAGE = 1 / 11
SIGNAL_LENGTH = int(BASE_BLOCK_SIZE * 1.5)  # 5280
SEQUENCE_LENGTH = SIGNAL_LENGTH // 16  # 330
SEGMENT_DURATION_SEC = SIGNAL_LENGTH / 4  # 1320 s = 22 min
STEP_SIZE = int(SIGNAL_LENGTH * (1 - OVERLAP_PERCENTAGE))  # 4800
STEP_DURATION_SEC = STEP_SIZE / 4  # 1200 s = 20 min

SIX_HOURS_SEC = 21600
MIN_DOMAIN_START_SCREENING = -(SIX_HOURS_SEC + SEGMENT_DURATION_SEC)  # -22920
MIN_DOMAIN_START_DATASET = -44640  # ~12.4 hours

MIN_VALID_HOURS_UNHEALTHY = 2.0
MIN_VALID_HOURS_HEALTHY = 3.0
HEALTHY_BG_CLS_FRACTION = 0.10
TEST_HOLDOUT_FRACTION = 0.10
HEALTHY_NO_BG_TLO_RATIO = 0.75
WEIGHT_THRESHOLD = 0.90
FLAT_TOLERANCE = 1e-9

N_FOLDS = 10
VAL_RATIO = 1 / 9  # used in augmented mode: 80 / 10 / 10
RANDOM_STATE = 42
TLO_WITH_RATIO = 0.75
N_DURATION_BINS = 3  # quantile tertiles: short / medium / long labour

NO_BG_SUBGROUPS = {"healthy_no_bg_cs", "healthy_no_bg_no_cs"}

# ---------------------------------------------------------------------------
# Phase-harmonic channel selection
# ---------------------------------------------------------------------------
# Full rationale, measurements and rejected alternatives:
#   hdf5_dataset/PHASE_HARMONIC_CHANNEL_SELECTION.md
#
# A phase-harmonic coefficient pairs two wavelets $(\psi_i, \psi_j)$ and is
# large only when their phases are locked at ratio $p = \xi_j / \xi_i$. The
# selection is therefore a band in true Hz (which frequencies may participate)
# crossed with a set of harmonic steps $k$ on the $p = 2^{k/Q}$ power grid
# (which phase relationships to keep).
SCATTERING_Q = 4
SAMPLING_RATE_HZ = 4.0

# Band edges $(\min, \max)$ in true Hz, inclusive at both ends.
#
# Lower edge 0.008 Hz is the analysis-window floor (doc §3.1): a Gabor wavelet
# at $\xi$ has time envelope $\sigma_t = 1/(2\pi\sigma)$, and below ~0.008 Hz
# its $\pm 3\sigma_t$ support exceeds the 1200 s trimmed segment, so the
# coefficient measures reflection padding rather than signal.
#
# FHR upper edge 1.0 Hz is the beat-series Nyquist limit (doc §3.2): FHR is a
# rate derived from beat detection at 110-160 bpm and resampled to 4 Hz, so
# content above ~1 Hz is interpolation artifact, not physiology.
#
# UP upper edge 0.05 Hz is the contraction band (doc §3.2, §7): a contraction
# is a 45-90 s pulse recurring every 2-3 min, so essentially all uterine energy
# is below 0.05 Hz. This matches the cap already used by the cross-channel
# selector `select_fhr_up_cross_coefficients_v2`.
FHR_PHASE_BAND_HZ = (0.008, 1.00)
UP_PHASE_BAND_HZ = (0.008, 0.05)

# Harmonic steps $k$, giving powers $p = 2^{k/Q}$: k=4 -> p=2 (one octave,
# waveform asymmetry), k=6 -> p=2.83, k=8 -> p=4 (two octaves, coupling between
# well-separated rhythms).
#
# $k = 0$ (the diagonal, $p = 1$) is excluded: it reduces to
# $C_{i,i,1} = \phi * |z_i|^2$, which is near-collinear with the
# $\phi * |z_i|$ scattering channel already stored alongside it (doc §5,
# median $|r| = 0.967$ measured on synthetic FHR). The phase block earns its
# place through the off-diagonal terms, which encode what the scattering
# modulus discards.
#
# Yields fhr_ph = 66 and up_ph = 15 at the production geometry.
# To retain the diagonal, set this to (0, 4, 6, 8) -> fhr_ph = 94, up_ph = 26.
PHASE_HARMONIC_K_STEPS = (4, 6, 8)

# Relative tolerance on the power match $|p - 2^{k/Q}| < \text{tol} \cdot 2^{k/Q}$.
# Relative rather than absolute because the power grid is geometric: a fixed
# absolute window is far too permissive at large $p$ and too strict at small
# $p$. (An absolute tolerance is why the legacy selector's
# `harmonic_ratios=[2, 3]` silently matches zero harmonic-3 coefficients —
# $\log_2 3 = 1.585$ needs $k = 6.34$, which is off-grid. See doc §3.3.)
PHASE_POWER_REL_TOL = 0.05


# ---------------------------------------------------------------------------
# Transform variant
# ---------------------------------------------------------------------------
# The stored features are wavelet transforms, and the two-sided ones read raw samples on both
# sides of the step they are stored at -- so a model told it is conditioning on "the past up to
# $t$" reads part of the interval it is asked to forecast. The causal variant replaces the bank
# with a strictly one-sided one, drops the channels whose warm-up outruns the stored segment, and
# records the per-channel warm-up so every consumer can honour it.
#
# The variant is a root attribute of every file this pipeline writes from now on, which makes a
# file self-describing. A file **without** that attribute is a legacy two-sided file: absence is
# the normal, expected state of every dataset already on disk and is never a defect.
TWO_SIDED = "two_sided"
CAUSAL = "causal"
TRANSFORMS = (TWO_SIDED, CAUSAL)

# Blocks a file may store, and the two the causal variant does not produce. ``fhr_up_ph`` mixes
# both signals into one coefficient, no model loads it, and it is the one block with no ``sel_*``
# provenance to verify channel identity against.
COEFFICIENT_BLOCKS = ("fhr_st", "fhr_ph", "fhr_up_ph", "up_st", "up_ph")
CAUSAL_BLOCKS = ("fhr_st", "fhr_ph", "up_st", "up_ph")


def validate_transform(transform: str) -> str:
    """Refuse an unknown transform variant, naming both valid values.

    Args:
        transform: The requested variant.

    Returns:
        The variant, unchanged, so this can wrap an assignment.

    Raises:
        ValueError: If it is not one of :data:`TRANSFORMS`.
    """
    if transform not in TRANSFORMS:
        raise ValueError(
            f"unknown transform {transform!r}; use {TWO_SIDED!r} for the two-sided kymatio "
            f"bank or {CAUSAL!r} for the one-sided gammatone bank"
        )
    return transform


@dataclass(frozen=True)
class PhaseChannelSelection:
    r"""A phase-harmonic channel selection together with its provenance.

    Bundles the boolean pair mask with per-channel metadata describing every
    selected coefficient, so the width written into the HDF5 and the metadata
    describing those channels cannot drift apart. Previously the ``fhr_ph``
    width was a literal ``44`` in :func:`create_initial_hdf5` while the data
    written was ``phase_mask.sum()`` wide — this type makes that desync
    unrepresentable.

    All arrays are ordered identically to the channel axis of the stored
    coefficient block, i.e. the order produced by boolean-indexing the
    phase-pair axis with :attr:`mask` (ascending pair index).

    Index convention follows
    ``KymatioPhaseScattering1D._build_coupling_indices``, which keeps pairs
    satisfying $\xi_i \le \xi_j$. So ``i`` is the **lower**-frequency filter of
    the pair, ``j`` the higher, and ``power`` $= \xi_j / \xi_i \ge 1$. This
    matches the ``freq_hz_secondary`` $= \xi_i$ / ``freq_hz_primary`` $= \xi_j$
    convention used in ``band_partition.py``.

    Attributes:
        mask: Bool tensor over the phase-pair axis, shape ``(n_pairs,)``.
        i: Lower-frequency filter index per channel, shape ``(n_channels,)``.
        j: Higher-frequency filter index per channel, shape ``(n_channels,)``.
        xi_i_hz: $\xi_i$ in Hz, shape ``(n_channels,)``.
        xi_j_hz: $\xi_j$ in Hz, shape ``(n_channels,)``.
        power: Harmonic ratio $p = \xi_j / \xi_i$, shape ``(n_channels,)``.
        band_hz: The $(\min, \max)$ Hz band the mask was built from.
        k_steps: The harmonic steps $k$ the mask was built from.
    """

    mask: torch.Tensor
    i: np.ndarray
    j: np.ndarray
    xi_i_hz: np.ndarray
    xi_j_hz: np.ndarray
    power: np.ndarray
    band_hz: Tuple[float, float]
    k_steps: Tuple[int, ...]

    @property
    def n_channels(self) -> int:
        """Number of selected channels — the width of the stored block."""
        return int(self.i.shape[0])


# ============================================================================
# Verbosity helpers
# ============================================================================
@contextmanager
def suppress_stdout_stderr():
    """Redirect stdout and stderr to devnull temporarily."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def setup_verbosity(verbose: bool):
    """Configure logging level.  Progress bars are always shown.

    Args:
        verbose: If True show INFO logs, else ERROR only.
            Progress bars (tqdm) are always enabled regardless of this flag.
    """
    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.setLevel(level)


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass
class GuidTrackingEntry:
    """Per-GUID tracking data collected during HDF5 creation."""

    all_domain_starts: List[float] = field(default_factory=list)
    included_domain_starts: List[float] = field(default_factory=list)
    skipped_low_weight: List[float] = field(default_factory=list)
    skipped_flat_region: List[float] = field(default_factory=list)
    skipped_scatter_failed: List[float] = field(default_factory=list)
    skipped_duplicate: List[float] = field(default_factory=list)
    skipped_post_delivery: List[float] = field(default_factory=list)
    error: bool = False
    error_msg: Optional[str] = None


# ============================================================================
# Signal utility functions
# ============================================================================
def _normalize_guid(guid_str: str) -> str:
    """Normalize a GUID string for matching: uppercase, remove hyphens.

    Args:
        guid_str: Raw GUID string from filename or CSV.

    Returns:
        Uppercased GUID with hyphens stripped.
    """
    return guid_str.strip().upper().replace("-", "")


def interpolate_bad_values(signal_2d: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values with linear interpolation, per row.

    For each row, valid samples serve as interpolation knots. Edge NaNs are
    extrapolated flat.  Rows that are entirely bad are filled with 0.
    Operates in-place.

    Args:
        signal_2d: 2-D array of shape ``(n_segments, n_samples)``.

    Returns:
        The same array, modified in-place.
    """
    bad = ~np.isfinite(signal_2d)
    if not bad.any():
        return signal_2d
    indices = np.arange(signal_2d.shape[1])
    for row_idx in range(signal_2d.shape[0]):
        row_bad = bad[row_idx]
        if not row_bad.any():
            continue
        row_good = ~row_bad
        if not row_good.any():
            signal_2d[row_idx] = 0.0
            continue
        signal_2d[row_idx, row_bad] = np.interp(
            indices[row_bad], indices[row_good], signal_2d[row_idx, row_good]
        )
    return signal_2d


def find_flat_regions(
    signal: np.ndarray, tolerance: float = 1e-3, min_length: int = 20
) -> List[Tuple[int, int]]:
    """Find flat regions in a 1-D signal.

    Args:
        signal: 1-D array.
        tolerance: Max abs difference between consecutive samples to be
            considered flat.
        min_length: Minimum number of consecutive flat samples to qualify.

    Returns:
        List of ``(start_idx, end_idx)`` tuples for qualifying flat regions.
    """
    flat_regions: List[Tuple[int, int]] = []
    start_idx = None
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i - 1]) <= tolerance:
            if start_idx is None:
                start_idx = i - 1
        else:
            if start_idx is not None:
                end_idx = i - 1
                if (end_idx - start_idx + 1) >= min_length:
                    flat_regions.append((start_idx, end_idx))
                start_idx = None
    if start_idx is not None:
        end_idx = len(signal) - 1
        if (end_idx - start_idx + 1) >= min_length:
            flat_regions.append((start_idx, end_idx))
    return flat_regions


def deduplicate_segments(
    domain_starts: list, sample_weights: np.ndarray
) -> Tuple[List[int], List[int], Dict]:
    """Remove duplicate segments sharing the same domain_start value.

    Keeps only the segment with the highest mean sample weight for each
    unique domain_start.

    Args:
        domain_starts: Per-segment domain_start values.
        sample_weights: 2-D array ``(n_segments, n_samples)``.

    Returns:
        ``(keep_indices, removed_indices, duplicate_groups)``
    """
    groups: Dict[float, List[int]] = defaultdict(list)
    for idx, ds in enumerate(domain_starts):
        groups[ds].append(idx)

    keep_indices: List[int] = []
    removed_indices: List[int] = []
    duplicate_groups: Dict[float, List[int]] = {}

    for ds, indices in groups.items():
        if len(indices) == 1:
            keep_indices.append(indices[0])
        else:
            duplicate_groups[ds] = indices
            best_idx = max(
                indices, key=lambda i: float(np.mean(sample_weights[i, :]))
            )
            keep_indices.append(best_idx)
            for i in indices:
                if i != best_idx:
                    removed_indices.append(i)

    keep_indices.sort()
    removed_indices.sort()
    return keep_indices, removed_indices, duplicate_groups


# ============================================================================
# HDF5 I/O
# ============================================================================
#: ``source_pickle_path`` when a build did not resume from a fold pickle. An explicit sentinel
#: rather than an empty string: ``""`` is indistinguishable from an attribute written out of an
#: unset variable, and this is the record that says whether two datasets are comparable at all.
NO_SOURCE_PICKLE = "<fresh run>"


def guid_set_digest(records_list: Optional[List[str]]) -> str:
    """A stable digest of the GUID set a shard is built from.

    Comparability with an existing dataset is the entire justification for the resumed-run path:
    a causal build is only comparable to the two-sided one segment for segment if both consumed
    the same fold pickle and therefore the same GUIDs. Nothing records that today, which makes the
    claim unverifiable once the run is over. Sorting before hashing makes the digest a property of
    the *set* rather than of the order the records happened to be discovered in, so two builds of
    the same shard agree whatever their variant.

    Args:
        records_list: The ``.mat`` paths this shard is built from; ``None`` is treated as empty.

    Returns:
        The SHA-256 hex digest of the sorted GUID stems.
    """
    stems = sorted({os.path.splitext(os.path.basename(r))[0] for r in (records_list or [])})
    return hashlib.sha256("\n".join(stems).encode("utf-8")).hexdigest()


def _write_selection_attrs(dataset, selection: PhaseChannelSelection) -> None:
    r"""Attach per-channel selection provenance to a coefficient dataset.

    Lets any consumer recover what each channel means — which wavelet pair, at
    which frequencies, at which harmonic ratio — without re-deriving the
    selection from the filter bank.

    Note:
        These attrs are currently **write-only**: nothing in the repository
        reads them yet. They exist so the re-derivation in
        ``band_partition.py`` (which calls the legacy
        ``select_fhr_phase_coefficients`` and would build a 44-channel map
        against 66-channel data) can be replaced by reading provenance off the
        dataset. That migration is outstanding and lives outside
        ``hdf5_dataset/``; until it lands, the stale-channel-map problem is
        still live — it just fails loudly on the count mismatch rather than
        silently.

    Per-channel arrays are ordered to match the channel axis. ``sel_i`` is the
    **lower**-frequency filter of the pair and ``sel_j`` the higher, so
    ``sel_power`` $= \xi_j / \xi_i \ge 1$.

    Args:
        dataset: Target HDF5 dataset (``fhr_ph`` or ``up_ph``).
        selection: The selection the dataset was sized from.
    """
    dataset.attrs["sel_i"] = selection.i
    dataset.attrs["sel_j"] = selection.j
    dataset.attrs["sel_xi_i_hz"] = selection.xi_i_hz
    dataset.attrs["sel_xi_j_hz"] = selection.xi_j_hz
    dataset.attrs["sel_power"] = selection.power
    dataset.attrs["sel_band_hz"] = np.asarray(selection.band_hz, dtype=np.float32)
    dataset.attrs["sel_k_steps"] = np.asarray(selection.k_steps, dtype=np.int32)


def _write_causal_attrs(dataset, plan: CausalChannelPlan) -> None:
    r"""Attach per-channel warm-up and delay to a causal coefficient dataset.

    ``causal_warmup_steps`` is the leading region in which a channel's output
    is a function of the assumed pre-recording history rather than of the
    recording, in **untrimmed** decimated steps — the storage geometry every
    other stored field uses, so a consumer reading the file at any trim rebases
    it itself. Storing it trimmed would make the attribute silently wrong for
    every other trim.

    It is an attribute and not a stored mask because it is a property of the
    filter bank and is therefore identical for every segment in every file: a
    per-sample $(C, T)$ boolean mask would replicate one constant array tens of
    thousands of times, about 76 KB per sample against about 600 bytes per file.

    Note:
        ``causal_delay_s`` — the composed group delay each channel is stale by —
        has **no reader inside this pipeline**, which does not compensate for
        it. It is written ahead of its consumer, exactly as the ``sel_*`` attrs
        were: the future reader is whatever replaces the two-sided $L_{95}$
        reach guard in ``teb_vae/lag_attn/channel_reach.py``, which is
        meaningless on causal data because future energy there is exactly zero.
        That guard needs a staleness number per channel, and this is it.

    Args:
        dataset: Target HDF5 coefficient dataset.
        plan: The block's channel plan, which the dataset was sized from.
    """
    dataset.attrs["causal_warmup_steps"] = plan.warmup_steps.astype(np.int32)
    dataset.attrs["causal_delay_s"] = plan.delay_s.astype(np.float32)


def create_initial_hdf5(
    path: str,
    len_signal: int,
    len_sequence: int,
    fhr_ph_selection: PhaseChannelSelection,
    n_fhr_st_channels: int,
    n_cross_phase_channels: Optional[int],
    n_up_st_channels: int = 0,
    up_ph_selection: Optional[PhaseChannelSelection] = None,
    transform: str = TWO_SIDED,
    channel_plan: Optional[Dict[str, CausalChannelPlan]] = None,
    source_pickle_path: Optional[str] = None,
    source_guid_digest: Optional[str] = None,
) -> None:
    """Create a new empty HDF5 file with the full dataset schema.

    ``fhr_up_ph`` contains only the FHR↔UP cross-channel phase coefficients;
    the UP self-phase harmonics live in a separate first-class ``up_ph``
    dataset with their own per-channel asinh stats.

    Every block is sized from what will actually be written into it — the two
    self-phase blocks from their selections, the scattering blocks from the
    resolved width — so the stored width and the data written at write time
    cannot disagree. Both self-phase blocks carry ``sel_*`` provenance attrs
    describing every channel (see :func:`_write_selection_attrs`), identically
    on both variants: the causal build changes which *scattering* channels
    survive and changes no phase selection at all.

    The file is self-describing: ``transform`` is written on both variants, and
    a causal file additionally records the filter-bank constants its warm-up
    vectors were derived under. A file **without** ``transform`` is a legacy
    two-sided file — the normal state of every dataset already on disk — and
    needs no migration.

    Args:
        path: Output HDF5 file path (overwrites if exists).
        len_signal: Raw signal length (e.g. 5760).
        len_sequence: Sequence dimension length.
        fhr_ph_selection: Selection for ``fhr_ph``; supplies its width and
            provenance attrs.
        n_fhr_st_channels: Channels for ``fhr_st``; 43 two-sided, 36 causal.
        n_cross_phase_channels: Channels for ``fhr_up_ph`` (pure cross-phase,
            equals ``masks["n_cross"]``); ``None`` = do not create the dataset,
            which is what the causal variant does.
        n_up_st_channels: Number of UP scattering channels (0 = do not create
            ``up_st`` dataset).
        up_ph_selection: Selection for ``up_ph``; ``None`` = do not create the
            ``up_ph`` dataset.
        transform: ``'two_sided'`` or ``'causal'``.
        channel_plan: Required for ``'causal'``: supplies the per-channel
            warm-up and delay attrs, and is cross-checked against the widths.
        source_pickle_path: The fold pickle this build resumed from, recorded
            so comparability with the dataset that produced it is checkable
            afterwards. ``None`` records :data:`NO_SOURCE_PICKLE`.
        source_guid_digest: Digest of the GUID set written here, from
            :func:`guid_set_digest`. ``None`` records the digest of the empty
            set, which is what a file created without a records list contains.

    Raises:
        ValueError: On an unknown *transform*; on a causal file with no channel
            plan, with a cross-phase width, or whose widths disagree with the
            plan.
    """
    validate_transform(transform)
    if transform == CAUSAL:
        if channel_plan is None:
            raise ValueError(
                "a causal file needs its channel_plan: the stored warm-up vectors and the "
                "surviving channel set both come from it, and neither is recoverable afterwards"
            )
        if n_cross_phase_channels is not None:
            raise ValueError(
                f"the causal variant does not produce fhr_up_ph, so n_cross_phase_channels must "
                f"be None, not {n_cross_phase_channels}. Creating the dataset would leave it "
                f"empty for the whole build."
            )
        if up_ph_selection is None:
            raise ValueError("a causal file needs up_ph_selection; up_ph is not optional there")
        # The widths and the plan come from one resolver, so a disagreement here means the
        # parameters and the plan were resolved at different times or from different banks —
        # which would store a warm-up vector describing channels the data does not contain.
        for name, width in (
            ("fhr_st", n_fhr_st_channels),
            ("fhr_ph", fhr_ph_selection.n_channels),
            ("up_st", n_up_st_channels),
            ("up_ph", up_ph_selection.n_channels),
        ):
            if channel_plan[name].n_channels != width:
                raise ValueError(
                    f"channel plan for '{name}' has {channel_plan[name].n_channels} channels but "
                    f"the dataset is being created {width} wide"
                )

    try:
        os.remove(path)
    except OSError:
        pass

    chunk_n = 32
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w", libver="latest") as h5f:
        h5f.attrs["transform"] = transform
        # Provenance, on both variants: which fold pickle the run resumed from and which GUIDs
        # landed here. A causal dataset is comparable to a two-sided one segment for segment only
        # if both were built from the same pickle, and after the run these two attributes are the
        # only surviving evidence either way. Absent on every file predating them, which means
        # unknown provenance rather than an error.
        h5f.attrs["source_pickle_path"] = (
            NO_SOURCE_PICKLE if source_pickle_path is None else str(source_pickle_path)
        )
        h5f.attrs["source_guid_digest"] = (
            guid_set_digest(None) if source_guid_digest is None else str(source_guid_digest)
        )
        if transform == CAUSAL:
            # What the warm-up vectors below mean: the bank they were measured on, and the energy
            # quantile they enclose. Recorded so the valid region is recoverable from the file
            # rather than only from the code that wrote it.
            h5f.attrs["causal_kernel_taps"] = np.int32(CAUSAL_KERNEL_TAPS)
            h5f.attrs["gammatone_order"] = np.int32(GAMMATONE_ORDER)
            h5f.attrs["causal_warmup_quantile"] = np.float32(CAUSAL_WARMUP_QUANTILE)
        h5f.create_dataset(
            "fhr",
            shape=(0, len_signal),
            maxshape=(None, len_signal),
            dtype="f4",
            chunks=(chunk_n, len_signal),
            compression="lzf",
        )
        h5f.create_dataset(
            "up",
            shape=(0, len_signal),
            maxshape=(None, len_signal),
            dtype="f4",
            chunks=(chunk_n, len_signal),
            compression="lzf",
        )
        # fhr_st width is fixed by the filter bank, not by a selection: one
        # order-0 channel plus one per first-order wavelet, less whichever the
        # causal drop rule removed. The whole block is stored unmasked.
        fhr_st_ds = h5f.create_dataset(
            "fhr_st",
            shape=(0, n_fhr_st_channels, len_sequence),
            maxshape=(None, n_fhr_st_channels, len_sequence),
            dtype="f4",
            chunks=(chunk_n, n_fhr_st_channels, len_sequence),
            compression="lzf",
        )
        # fhr_ph width comes from the selection, never a literal. A hardcoded
        # width here is what previously pinned this dataset to 44 channels
        # regardless of the mask actually applied at write time.
        n_fhr_ph = fhr_ph_selection.n_channels
        fhr_ph_ds = h5f.create_dataset(
            "fhr_ph",
            shape=(0, n_fhr_ph, len_sequence),
            maxshape=(None, n_fhr_ph, len_sequence),
            dtype="f4",
            chunks=(chunk_n, n_fhr_ph, len_sequence),
            compression="lzf",
        )
        _write_selection_attrs(fhr_ph_ds, fhr_ph_selection)
        # fhr_up_ph: cross-channel phase, two-sided only.
        if n_cross_phase_channels is not None:
            h5f.create_dataset(
                "fhr_up_ph",
                shape=(0, n_cross_phase_channels, len_sequence),
                maxshape=(None, n_cross_phase_channels, len_sequence),
                dtype="f4",
                chunks=(chunk_n, n_cross_phase_channels, len_sequence),
                compression="lzf",
            )
        # up_st: UP scattering coefficients (optional, same structure as fhr_st)
        up_st_ds = None
        if n_up_st_channels > 0:
            up_st_ds = h5f.create_dataset(
                "up_st",
                shape=(0, n_up_st_channels, len_sequence),
                maxshape=(None, n_up_st_channels, len_sequence),
                dtype="f4",
                chunks=(chunk_n, n_up_st_channels, len_sequence),
                compression="lzf",
            )
        # up_ph: UP self-phase harmonics (optional). First-class field with its
        # own per-channel asinh stats — no longer concatenated into fhr_up_ph.
        up_ph_ds = None
        if up_ph_selection is not None:
            n_up_ph = up_ph_selection.n_channels
            up_ph_ds = h5f.create_dataset(
                "up_ph",
                shape=(0, n_up_ph, len_sequence),
                maxshape=(None, n_up_ph, len_sequence),
                dtype="f4",
                chunks=(chunk_n, n_up_ph, len_sequence),
                compression="lzf",
            )
            _write_selection_attrs(up_ph_ds, up_ph_selection)
        if transform == CAUSAL:
            for name, dataset in (
                ("fhr_st", fhr_st_ds), ("fhr_ph", fhr_ph_ds),
                ("up_st", up_st_ds), ("up_ph", up_ph_ds),
            ):
                _write_causal_attrs(dataset, channel_plan[name])
        h5f.create_dataset(
            "target",
            shape=(0, len_sequence),
            maxshape=(None, len_sequence),
            dtype="f4",
            chunks=(chunk_n, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "weight",
            shape=(0, len_sequence),
            maxshape=(None, len_sequence),
            dtype="f4",
            chunks=(chunk_n, len_sequence),
            compression="lzf",
        )
        h5f.create_dataset(
            "epoch",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "cs_label",
            shape=(0,),
            maxshape=(None,),
            dtype="u1",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "bg_label",
            shape=(0,),
            maxshape=(None,),
            dtype="u1",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "time_from_labor_onset",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "second_stage_onset",
            shape=(0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_n,),
            compression="lzf",
        )
        h5f.create_dataset(
            "guid",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dt,
            chunks=(chunk_n,),
        )


def create_hdf5_for_masks(
    path: str,
    masks: Dict[str, Any],
    len_sequence: int,
    len_signal: int = SIGNAL_LENGTH,
    records_list: Optional[List[str]] = None,
    source_pickle_path: Optional[str] = None,
) -> None:
    """Create an empty file whose every width and provenance attr comes from one place.

    Both write paths — the per-partition classification files and the
    pre-training files, which bypass ``_build_hdf5_for_partition`` entirely —
    go through here, so a variant threaded into one of them cannot be missing
    from the other. That half-threaded state is the real failure mode: it
    produces a directory of files that disagree with each other about what they
    contain. The same argument applies to the provenance attrs, which is why
    the records list is taken here rather than written by whoever fills the
    file afterwards.

    Args:
        path: Output HDF5 file path.
        masks: Output of :func:`compute_scattering_masks`.
        len_sequence: Sequence dimension length.
        len_signal: Raw signal length.
        records_list: The ``.mat`` paths this shard will be built from; the
            stored GUID digest is taken from it.
        source_pickle_path: The fold pickle the run resumed from, or ``None``
            for a fresh run.
    """
    widths = resolve_channel_layout(masks)
    create_initial_hdf5(
        path=path,
        len_signal=len_signal,
        len_sequence=len_sequence,
        fhr_ph_selection=masks["fhr_ph_selection"],
        n_fhr_st_channels=widths["fhr_st"],
        # None on the causal variant, which does not produce the cross-phase block.
        n_cross_phase_channels=widths["fhr_up_ph"],
        n_up_st_channels=widths["up_st"],
        up_ph_selection=masks["up_ph_selection"],
        transform=masks.get("transform", TWO_SIDED),
        channel_plan=masks.get("channel_plan"),
        source_pickle_path=source_pickle_path,
        source_guid_digest=guid_set_digest(records_list),
    )


def append_samples_batch(
    path: str,
    fhr_batch: np.ndarray,
    up_batch: np.ndarray,
    fhr_st_batch: np.ndarray,
    fhr_ph_batch: np.ndarray,
    target_batch: np.ndarray,
    weight_batch: np.ndarray,
    guid_batch: list,
    epoch_batch: np.ndarray,
    cs_label_batch: np.ndarray,
    bg_label_batch: np.ndarray,
    tlo_batch: np.ndarray,
    second_stage_batch: np.ndarray,
    fhr_up_ph_batch: Optional[np.ndarray] = None,
    up_st_batch: Optional[np.ndarray] = None,
    up_ph_batch: Optional[np.ndarray] = None,
) -> None:
    """Append K samples to an existing HDF5 file in a single open/close.

    ``fhr_up_ph`` is guarded in **both** directions, which the other optional
    blocks are not. Writing it unconditionally raises ``KeyError`` on a causal
    file, which has no such dataset; but skipping it when the dataset is absent
    — the pattern ``up_st`` and ``up_ph`` use — would silently discard a
    computed block on a two-sided file, which is the failure the geometry guard
    exists to prevent. So a batch without a dataset and a dataset without a
    batch both raise.

    Args:
        path: Path to existing HDF5 file.
        fhr_batch: Shape ``(K, len_signal)``.
        up_batch: Shape ``(K, len_signal)``.
        fhr_st_batch: Shape ``(K, n_fhr_st, len_seq)``.
        fhr_ph_batch: Shape ``(K, n_ph, len_seq)``.
        target_batch: Shape ``(K, len_seq)``.
        weight_batch: Shape ``(K, len_seq)``.
        guid_batch: List of GUID strings, length K.
        epoch_batch: Shape ``(K,)``, float32.
        cs_label_batch: Shape ``(K,)``, uint8.
        bg_label_batch: Shape ``(K,)``, uint8.
        tlo_batch: Shape ``(K,)``, float32.
        second_stage_batch: Shape ``(K,)``, float32.
        fhr_up_ph_batch: Shape ``(K, n_cross, len_seq)`` — cross-channel phase
            coefficients only; UP self-phase harmonics go in ``up_ph_batch``.
            ``None`` only for a causal file, which stores no such dataset.
        up_st_batch: Shape ``(K, n_up_st, len_seq)``. Optional — only written if
            the target HDF5 has the ``up_st`` dataset.
        up_ph_batch: Shape ``(K, n_up_phase, len_seq)``. Optional — only written
            if the target HDF5 has the ``up_ph`` dataset.

    Raises:
        ValueError: If ``fhr_up_ph`` exists in the file but no batch was given,
            or a batch was given but the dataset does not exist.
    """
    k = fhr_batch.shape[0]
    if k == 0:
        return
    with h5py.File(path, "a", libver="latest") as h5f:
        has_cross = "fhr_up_ph" in h5f
        if has_cross and fhr_up_ph_batch is None:
            raise ValueError(
                f"{path} has an fhr_up_ph dataset but no fhr_up_ph_batch was given; it would be "
                f"left empty for every sample written. Create the file with "
                f"n_cross_phase_channels=None if this build does not produce it."
            )
        if fhr_up_ph_batch is not None and not has_cross:
            raise ValueError(
                f"{path} has no fhr_up_ph dataset but an fhr_up_ph_batch was given; the block "
                f"would be computed and dropped on the floor with no error"
            )

        idx = h5f["fhr"].shape[0]
        new_size = idx + k
        # Iterates the datasets that exist, so an absent fhr_up_ph is simply not resized.
        for _name, ds in h5f.items():
            ds.resize((new_size,) + ds.shape[1:])
        h5f["fhr"][idx:new_size] = fhr_batch
        h5f["up"][idx:new_size] = up_batch
        h5f["fhr_st"][idx:new_size] = fhr_st_batch
        h5f["fhr_ph"][idx:new_size] = fhr_ph_batch
        if fhr_up_ph_batch is not None:
            h5f["fhr_up_ph"][idx:new_size] = fhr_up_ph_batch
        if up_st_batch is not None and "up_st" in h5f:
            h5f["up_st"][idx:new_size] = up_st_batch
        if up_ph_batch is not None and "up_ph" in h5f:
            h5f["up_ph"][idx:new_size] = up_ph_batch
        h5f["target"][idx:new_size] = target_batch
        h5f["weight"][idx:new_size] = weight_batch
        h5f["epoch"][idx:new_size] = epoch_batch
        h5f["cs_label"][idx:new_size] = cs_label_batch.astype(np.uint8)
        h5f["bg_label"][idx:new_size] = bg_label_batch.astype(np.uint8)
        if "time_from_labor_onset" in h5f:
            h5f["time_from_labor_onset"][idx:new_size] = tlo_batch
        if "second_stage_onset" in h5f:
            h5f["second_stage_onset"][idx:new_size] = second_stage_batch
        for i, g in enumerate(guid_batch):
            h5f["guid"][idx + i] = g


# ============================================================================
# Scattering masks
# ============================================================================
def _phase_pair_mask(
    model: KymatioPhaseScattering1D,
    min_hz: float,
    max_hz: Optional[float],
    k_steps: Tuple[int, ...],
    fs: Optional[float] = None,
    Q: Optional[int] = None,
    tol: Optional[float] = None,
) -> torch.Tensor:
    r"""Build a phase-pair mask over a true-Hz band and the $2^{k/Q}$ power grid.

    A pair $(i, j)$ is kept when both centre frequencies lie inside
    ``[min_hz, max_hz]`` and its power $p = \xi_j / \xi_i$ falls within a
    *relative* tolerance of $2^{k/Q}$ for some $k$ in ``k_steps``.

    ``model.center_freqs`` holds kymatio's normalised $\xi$ in cycles per
    sample, so the Hz thresholds are divided by ``fs`` here.

    This deliberately does **not** call
    ``KymatioPhaseScattering1D.select_fhr_phase_coefficients``: that selector
    omits the $f_s$ conversion (its ``min_freq`` behaves as a raw $\xi$
    threshold, so the nominal 0.006 Hz is really 0.024 Hz), has no upper band
    edge, and uses an absolute power tolerance. Correcting it in place would
    silently change behaviour at its four other call sites
    (``create_hdf5_dataset.py``, ``scattering_adapter.py``,
    ``band_partition.py``), so the selection is rebuilt from the filter bank
    here instead.

    Because pairs always satisfy $\xi_i \le \xi_j$, requiring both endpoints in
    band is equivalent to $\xi_i \ge$ ``min_hz`` and $\xi_j \le$ ``max_hz`` —
    the predicate form used in PHASE_HARMONIC_CHANNEL_SELECTION.md §10.

    Args:
        model: Constructed transform, supplying ``center_freqs``, ``i_idx``,
            ``j_idx`` and ``powers``.
        min_hz: Inclusive lower band edge in Hz.
        max_hz: Inclusive upper band edge in Hz; ``None`` means unbounded.
        k_steps: Harmonic steps $k$ to admit.
        fs: Sampling rate in Hz; ``None`` reads ``SAMPLING_RATE_HZ``.
        Q: Wavelets per octave, setting the power-grid spacing; ``None`` reads
            ``SCATTERING_Q``.
        tol: Relative tolerance on the power match; ``None`` reads
            ``PHASE_POWER_REL_TOL``.

    Returns:
        Bool tensor of shape ``(n_pairs,)``, ``True`` for selected pairs.
    """
    # Resolved here rather than bound as default arguments, which Python
    # evaluates once at import: that would make these three constants
    # un-overridable at runtime while PHASE_HARMONIC_K_STEPS (read at call
    # time in compute_scattering_masks) stayed patchable — an inconsistency
    # that silently produces a selection nobody asked for.
    fs = SAMPLING_RATE_HZ if fs is None else fs
    Q = SCATTERING_Q if Q is None else Q
    tol = PHASE_POWER_REL_TOL if tol is None else tol

    cf = model.center_freqs
    hi = (max_hz / fs) if max_hz is not None else float("inf")
    in_band = (cf >= min_hz / fs) & (cf <= hi)
    # Both endpoints of the pair must sit inside the band.
    pair_ok = in_band[model.i_idx] & in_band[model.j_idx]

    mask = torch.zeros_like(pair_ok)
    for k in k_steps:
        target = 2.0 ** (k / Q)
        # Relative tolerance: the grid is geometric, so a fixed absolute
        # window would be far too permissive at the large-$p$ end.
        mask |= pair_ok & (torch.abs(model.powers - target) < tol * target)
    return mask


def _build_phase_selection(
    model: KymatioPhaseScattering1D,
    band_hz: Tuple[float, float],
    k_steps: Tuple[int, ...],
    fs: Optional[float] = None,
    label: str = "phase",
) -> PhaseChannelSelection:
    r"""Build a phase mask and its per-channel metadata in one pass.

    Args:
        model: Constructed transform.
        band_hz: $(\min, \max)$ band edges in Hz.
        k_steps: Harmonic steps $k$ to admit.
        fs: Sampling rate in Hz, used to express the metadata in Hz; ``None``
            reads ``SAMPLING_RATE_HZ``.
        label: Field name used in the empty-selection error message.

    Returns:
        The selection, with metadata ordered to match the stored channel axis.

    Raises:
        ValueError: If the band and $k$-steps admit no pair at all.
    """
    fs = SAMPLING_RATE_HZ if fs is None else fs
    mask = _phase_pair_mask(model, band_hz[0], band_hz[1], k_steps, fs=fs)

    # An empty selection is always a misconfiguration, and it fails badly if
    # allowed through: a zero-width HDF5 dataset makes h5py reject the chunk
    # shape with "All chunk dimensions must be positive", naming neither the
    # field nor the band. Catch it here, where the band is in scope.
    if not bool(mask.any()):
        raise ValueError(
            f"Phase selection for '{label}' is empty: the band "
            f"{band_hz} Hz with k_steps={k_steps} matches no wavelet pair. "
            f"The band must span at least the widest harmonic step "
            f"(k={max(k_steps)} needs a factor of {2.0 ** (max(k_steps) / SCATTERING_Q):.2f} "
            f"between its edges)."
        )

    # Boolean indexing preserves ascending pair order, which is exactly the
    # order the coefficient block is sliced with at write time — so these
    # arrays line up channel-for-channel with the stored data.
    i = model.i_idx[mask].cpu().numpy().astype(np.int32)
    j = model.j_idx[mask].cpu().numpy().astype(np.int32)
    cf = model.center_freqs.cpu().numpy()

    return PhaseChannelSelection(
        mask=mask,
        i=i,
        j=j,
        xi_i_hz=(cf[i] * fs).astype(np.float32),
        xi_j_hz=(cf[j] * fs).astype(np.float32),
        power=model.powers[mask].cpu().numpy().astype(np.float32),
        band_hz=band_hz,
        k_steps=k_steps,
    )


def compute_scattering_masks(
    signal_length: int,
    scattering_T: int = 16,
    device=None,
    transform: str = TWO_SIDED,
) -> Dict[str, Any]:
    r"""Compute every coefficient selection once, up front.

    The two self-phase blocks (``fhr_ph``, ``up_ph``) are selected by true-Hz
    band crossed with the $2^{k/Q}$ harmonic grid — see the constants at the
    top of this module and PHASE_HARMONIC_CHANNEL_SELECTION.md for the
    rationale. The cross-channel block (``fhr_up_ph``) is unchanged: it still
    uses the two-band selector, whose $k = 0$ term is a genuine UP-to-FHR
    coupling rather than the redundant self-energy diagonal.

    At the production geometry ($J=11$, $Q=4$, $T=16$, ``shape=5280``,
    $f_s = 4$ Hz) this yields ``fhr_ph`` = 66, ``fhr_up_ph`` = 79 and
    ``up_ph`` = 15.

    For ``transform='causal'`` it additionally builds the causal bank and its
    channel plan **here**, so that the widths written into the file, the
    warm-up vectors stored beside them, the channels the transform gathers and
    the operator's log all come from one object computed once. The plan's phase
    pairs are taken from the :class:`PhaseChannelSelection` objects above and
    never from a second selector, which is what makes channel $c$ of the data
    and channel $c$ of the ``sel_*`` provenance the same channel by
    construction rather than by agreement.

    Args:
        signal_length: Raw signal length (e.g. 5280).
        scattering_T: Decimation factor.
        device: Torch device.
        transform: ``'two_sided'`` or ``'causal'``.

    Returns:
        Dict with two :class:`PhaseChannelSelection` objects
        (``fhr_ph_selection``, ``up_ph_selection``), the cross-phase mask
        (``cross_mask``), its channel count (``n_cross``), its selector
        metadata (``cross_metadata``), the scattering-block width
        (``n_scattering``) and the resolved ``transform``; plus
        ``causal_bank`` and ``channel_plan`` on the causal variant.
    """
    validate_transform(transform)
    tmp_model = KymatioPhaseScattering1D(
        J=11,
        Q=SCATTERING_Q,
        T=scattering_T,
        shape=signal_length,
        device=device,
        tukey_alpha=None,
        max_order=1,
    )
    fhr_ph_selection = _build_phase_selection(
        tmp_model, FHR_PHASE_BAND_HZ, PHASE_HARMONIC_K_STEPS, label="fhr_ph"
    )
    up_ph_selection = _build_phase_selection(
        tmp_model, UP_PHASE_BAND_HZ, PHASE_HARMONIC_K_STEPS, label="up_ph"
    )

    # fhr_up_ph is unchanged: still the two-band cross-channel selector with
    # the UP cap at 0.05 Hz. Its i/j semantics differ from the self-phase
    # blocks (UP filter vs FHR filter, not low vs high), so it is intentionally
    # not wrapped in a PhaseChannelSelection.
    cross_sel = tmp_model.select_fhr_up_cross_coefficients_v2(
        band_a_up_max_hz=0.05, band_b_up_max_hz=0.05
    )
    cross_mask = cross_sel["cross_mask"]

    masks: Dict[str, Any] = {
        "fhr_ph_selection": fhr_ph_selection,
        "up_ph_selection": up_ph_selection,
        "cross_mask": cross_mask,
        "n_cross": int(cross_mask.sum().item()),
        "cross_metadata": cross_sel.get("metadata", {}),
        # One order-0 channel plus one per first-order wavelet. Derived from the
        # bank rather than written as 43, so a J/Q change moves every width that
        # depends on it instead of leaving a literal behind.
        "n_scattering": 1 + len(tmp_model.center_freqs),
        "transform": transform,
    }

    if transform == CAUSAL:
        causal_bank = build_causal_bank(build_filter_bank(signal_length))
        masks["causal_bank"] = causal_bank
        masks["channel_plan"] = build_channel_plan(
            causal_bank,
            _selection_pairs(fhr_ph_selection),
            _selection_pairs(up_ph_selection),
            sequence_length=signal_length // scattering_T,
            decimation=scattering_T,
        )
    return masks


def _selection_pairs(selection: PhaseChannelSelection) -> np.ndarray:
    """The $(i, j)$ pair array of a stored selection, in stored channel order.

    The causal channel plan takes its pairs from here rather than from
    ``causal_scattering.selected_pairs``, which rebuilds the same rule
    independently. Two implementations that agree today could stop agreeing,
    and the failure would be a warm-up vector describing a different channel
    than the data and the ``sel_*`` attrs do — silently wrong rather than
    loudly broken. ``selected_pairs`` keeps its role as the independent rebuild
    used to *verify* a shard.

    Args:
        selection: The selection the block is sized from.

    Returns:
        ``(n_channels, 2)`` of filter indices, column 0 the lower frequency.
    """
    return np.stack(
        [np.asarray(selection.i, dtype=int), np.asarray(selection.j, dtype=int)], axis=1
    )


def resolve_channel_layout(masks: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """Stored width of every coefficient block, or ``None`` for a block this variant omits.

    The one place a width is decided. Every consumer — the schema that creates
    the datasets, the guard that checks them, the operator log — reads it from
    here, so a file cannot be created at one width and filled at another.

    Args:
        masks: Output of :func:`compute_scattering_masks`.

    Returns:
        ``{block: width or None}`` for all five blocks.
    """
    if masks.get("transform", TWO_SIDED) == CAUSAL:
        plan = masks["channel_plan"]
        widths: Dict[str, Optional[int]] = {
            name: plan[name].n_channels for name in CAUSAL_BLOCKS
        }
        widths["fhr_up_ph"] = None
        return widths

    n_scattering = int(masks["n_scattering"])
    return {
        "fhr_st": n_scattering,
        "fhr_ph": masks["fhr_ph_selection"].n_channels,
        "fhr_up_ph": int(masks["n_cross"]),
        "up_st": n_scattering,
        "up_ph": masks["up_ph_selection"].n_channels,
    }


def describe_layout(masks: Dict[str, Any], device: Optional[Any] = None) -> Dict[str, Any]:
    r"""Everything an operator needs to confirm before a multi-hour build commits.

    Returned as a dict rather than formatted here, for two reasons. A test can assert the *layout*
    against the channel plan instead of asserting the wording of an f-string; and the no-data smoke
    check can print the same numbers the pipeline logs without a second copy of the derivation.
    :func:`format_layout` turns it into lines.

    Every value is derived at run time. The previous log carried ``43`` as a literal inside its
    f-string, twice, which a change to $J$ or $Q$ would have left silently stale.

    Args:
        masks: Output of :func:`compute_scattering_masks`.
        device: The torch device the transform will run on; ``None`` means the writer's own default.

    Returns:
        ``transform``, ``device``, per-block ``widths``, ``c_y``/``c_u``, and — on the causal
        variant — the bank constants (``gammatone_order``, ``causal_kernel_taps``,
        ``causal_warmup_quantile``), the ``dropped`` channels per block as
        ``{'count', 'first', 'last'}``, and the per-block ``warmup_steps`` and ``delay_s`` ranges
        as ``(min, max)``.
    """
    transform = masks.get("transform", TWO_SIDED)
    widths = resolve_channel_layout(masks)
    layout: Dict[str, Any] = {
        "transform": transform,
        "device": "default" if device is None else str(device),
        "widths": widths,
        "c_y": int(widths["fhr_st"] or 0) + int(widths["fhr_ph"] or 0),
        "c_u": int(widths["up_st"] or 0) + int(widths["up_ph"] or 0),
    }
    if transform != CAUSAL:
        return layout

    plan = masks["channel_plan"]
    # The undropped width per block: what the transform produces before the drop rule runs. The
    # plan holds only the survivors, so the dropped set is the complement against these.
    n_scattering = int(masks["n_scattering"])
    full_widths = {
        "fhr_st": n_scattering,
        "up_st": n_scattering,
        "fhr_ph": masks["fhr_ph_selection"].n_channels,
        "up_ph": masks["up_ph_selection"].n_channels,
    }
    layout["gammatone_order"] = int(GAMMATONE_ORDER)
    layout["causal_kernel_taps"] = int(CAUSAL_KERNEL_TAPS)
    layout["causal_warmup_quantile"] = float(CAUSAL_WARMUP_QUANTILE)
    dropped: Dict[str, Dict[str, Optional[int]]] = {}
    warmup_range: Dict[str, Tuple[int, int]] = {}
    delay_range: Dict[str, Tuple[float, float]] = {}
    for name in CAUSAL_BLOCKS:
        block = plan[name]
        gone = sorted(set(range(full_widths[name])) - set(int(c) for c in block.kept))
        dropped[name] = {
            "count": len(gone),
            "first": gone[0] if gone else None,
            "last": gone[-1] if gone else None,
        }
        warmup_range[name] = (
            int(block.warmup_steps.min()), int(block.warmup_steps.max())
        )
        delay_range[name] = (float(block.delay_s.min()), float(block.delay_s.max()))
    layout["dropped"] = dropped
    layout["warmup_steps"] = warmup_range
    layout["delay_s"] = delay_range
    return layout


def format_layout(layout: Dict[str, Any]) -> List[str]:
    """Render :func:`describe_layout` as the lines the operator reads before a build starts.

    Args:
        layout: Output of :func:`describe_layout`.

    Returns:
        One string per line, in the order they should be emitted.
    """
    widths = layout["widths"]
    lines = []
    if layout["transform"] == CAUSAL:
        lines.append(
            f"Transform: {layout['transform']} (gammatone n={layout['gammatone_order']}, "
            f"{layout['causal_kernel_taps']} taps, "
            f"warm-up quantile {layout['causal_warmup_quantile']:g})"
        )
    else:
        lines.append(f"Transform: {layout['transform']}")
    lines.append(
        f"Channel layout: fhr_st={widths['fhr_st']} + fhr_ph={widths['fhr_ph']} "
        f"(c_y={layout['c_y']}), up_st={widths['up_st']} + up_ph={widths['up_ph']} "
        f"(c_u={layout['c_u']}), "
        f"fhr_up_ph={widths['fhr_up_ph'] if widths['fhr_up_ph'] is not None else 'absent'}"
    )
    if layout["transform"] == CAUSAL:
        for name in CAUSAL_BLOCKS:
            gone = layout["dropped"][name]
            if gone["count"]:
                lines.append(
                    f"Dropped {gone['count']} never-valid channels from {name} "
                    f"(channels {gone['first']}..{gone['last']})"
                )
        lines.append(
            "Warm-up range: "
            + ", ".join(
                f"{name} {layout['warmup_steps'][name][0]}..{layout['warmup_steps'][name][1]}"
                for name in CAUSAL_BLOCKS
            )
            + " steps"
        )
        # Recorded, never compensated: a consumer that forecasts from these channels is reading
        # each one as of this many seconds ago.
        lines.append(
            "Group delay: "
            + ", ".join(
                f"{name} {layout['delay_s'][name][0]:.1f}..{layout['delay_s'][name][1]:.1f}"
                for name in CAUSAL_BLOCKS
            )
            + " s"
        )
    lines.append(f"Device: {layout['device']}")
    return lines


def _validate_geometry(
    hdf5_path: str,
    expected_widths: Dict[str, Optional[int]],
    pair_masks: Optional[Dict[str, torch.Tensor]] = None,
    n_pairs: Optional[int] = None,
) -> None:
    """Fail fast on a geometry mismatch, before any transform work is done.

    Two independent things can disagree, and both are silent-until-late
    without this check:

    1. **Pair axis.** The masks are built against the filter bank inside
       :func:`compute_scattering_masks`; they are applied to the one built by
       the writer. If those banks differ (``J``, ``Q``, or ``shape``), the mask
       has the wrong length and ``phase_corr[mask, :]`` raises ``IndexError``.
       That raise lands in the per-record ``except Exception`` below, so
       *every* record fails identically, the run reports success, and a full
       set of empty HDF5 files ships as if validated. The causal path indexes
       responses by pair *index* rather than by a mask over a pair axis, so it
       supplies no model and this check is skipped rather than faked.
    2. **Stored widths.** Every dataset this writer fills must exist and be
       exactly as wide as the block written into it. A dataset that is absent
       is worse than one that is mis-sized: ``append_samples_batch`` skips
       missing optional fields silently, so the coefficients are computed and
       then dropped on the floor with no error and no warning.

    The required-block set is the *mapping*, not a branch: a width means the
    block must exist at that width, and ``None`` means it must be absent. That
    is how a missing ``fhr_up_ph`` stays fatal for a two-sided build and is
    correct for a causal one, with one implementation and no variant test
    inside.

    Args:
        hdf5_path: The file about to be filled.
        expected_widths: ``{block: width or None}`` from
            :func:`resolve_channel_layout`.
        pair_masks: Masks whose length must match the transform's pair axis.
        n_pairs: Pair-axis length of the transform the masks will index;
            ``None`` skips the pair-axis check.

    Raises:
        ValueError: On a pair-axis mismatch, a missing dataset, an unexpected
            dataset, or a width mismatch, naming the field and both numbers.
    """
    if n_pairs is not None:
        for field_name, mask in (pair_masks or {}).items():
            if int(mask.shape[0]) != n_pairs:
                raise ValueError(
                    f"Phase-pair axis mismatch for '{field_name}': the mask spans "
                    f"{int(mask.shape[0])} pairs but this transform produces "
                    f"{n_pairs}. The masks were built against a different filter "
                    f"bank — check that J, Q (SCATTERING_Q) and the signal length "
                    f"match between compute_scattering_masks and this writer."
                )

    with h5py.File(hdf5_path, "r") as h5f:
        for field_name, n_expected in expected_widths.items():
            if n_expected is None:
                if field_name in h5f:
                    raise ValueError(
                        f"Dataset '{field_name}' exists in {hdf5_path} but this build does not "
                        f"produce it, so it would stay empty for every sample. The file was "
                        f"created for a different transform variant."
                    )
                continue
            if field_name not in h5f:
                raise ValueError(
                    f"Dataset '{field_name}' is missing from {hdf5_path}, but "
                    f"this writer computes it for every segment. It would be "
                    f"silently discarded by append_samples_batch. Check that "
                    f"create_initial_hdf5 was given n_up_st_channels and "
                    f"up_ph_selection."
                )
            n_on_disk = int(h5f[field_name].shape[1])
            if n_on_disk != n_expected:
                raise ValueError(
                    f"Channel-count mismatch for '{field_name}' in "
                    f"{hdf5_path}: the dataset is {n_on_disk} channels wide "
                    f"but {n_expected} channels will be written. The HDF5 was "
                    f"created with a different selection or filter bank than "
                    f"the one being applied."
                )


# ============================================================================
# CSV metadata loading
# ============================================================================
def load_csv_metadata(
    csv_path: str, verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load labor onset and second stage data from the complete CSV.

    Args:
        csv_path: Path to CSV with columns ``trace_guid``,
            ``labor_onset_hours``, ``second_stage_onset_hours``.
        verbose: Whether to print summary info.

    Returns:
        Tuple of ``(labor_onset_map, second_stage_map)`` where each maps
        normalized GUID to seconds relative to delivery.
    """
    df = pd.read_csv(csv_path)
    labor_onset_map: Dict[str, float] = {}
    second_stage_map: Dict[str, float] = {}
    n_tlo_missing = 0
    n_ss_missing = 0

    for _, row in df.iterrows():
        guid = _normalize_guid(str(row["trace_guid"]))

        hours = row.get("labor_onset_hours")
        if pd.notna(hours) and str(hours).strip() != "":
            labor_onset_map[guid] = float(hours) * 3600.0
        else:
            n_tlo_missing += 1

        ss_hours = row.get("second_stage_onset_hours")
        if pd.notna(ss_hours) and str(ss_hours).strip() != "":
            second_stage_map[guid] = float(ss_hours) * 3600.0
        else:
            n_ss_missing += 1

    if verbose:
        logger.info(
            f"Loaded TLO for {len(labor_onset_map)} GUIDs "
            f"({n_tlo_missing} missing) from {csv_path}"
        )
        logger.info(
            f"Loaded second-stage for {len(second_stage_map)} GUIDs "
            f"({n_ss_missing} missing)"
        )
    return labor_onset_map, second_stage_map


# ============================================================================
# Step 1: GUID prescreening
# ============================================================================
def _run_mimo_pipeline(
    record_path: str,
    min_domain_start: float,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Load a .mat file through MIMO and return signals, domain_starts, weights.

    Args:
        record_path: Path to .mat file.
        min_domain_start: min_domain_start for prepare_data (both channels).
        verbose: If False, suppress MIMO stdout.

    Returns:
        ``(fhr, up, domain_starts, sample_weights)``
    """
    mimo_adaptor = EarlyMaestraMimoAdaptor(
        do_transpose=True,
        process_targets=True,
        n_aux_labels=None,
        signal_indices=range(0, 2),
        n_input_chan=2,
        labels=["HIE", "ACIDOSIS", "HEALTHY"],
        up_shift_secs=-20,
        default_target_index=0,
    )
    if verbose:
        mimo_adaptor.read_single_input(
            record_path,
            out_dec_factor=16,
            out_dec_factor_offset=0,
            target_is_onehot=True,
            dtype=np.float32,
        )
        mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
            batch_size=1,
            do_evaluate=True,
            align_left=True,
            do_split=True,
            do_pad=True,
            do_reflect=True,
            base_length=BASE_BLOCK_SIZE,
            do_equalize=True,
            do_merge=True,
            min_domain_start=[min_domain_start, min_domain_start],
            max_domain_start=[np.inf, np.inf],
            overlap_percentage=OVERLAP_PERCENTAGE,
        )
    else:
        with suppress_stdout_stderr():
            mimo_adaptor.read_single_input(
                record_path,
                out_dec_factor=16,
                out_dec_factor_offset=0,
                target_is_onehot=True,
                dtype=np.float32,
            )
            mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                batch_size=1,
                do_evaluate=True,
                align_left=True,
                do_split=True,
                do_pad=True,
                do_reflect=True,
                base_length=BASE_BLOCK_SIZE,
                do_equalize=True,
                do_merge=True,
                min_domain_start=[min_domain_start, min_domain_start],
                max_domain_start=[np.inf, np.inf],
                overlap_percentage=OVERLAP_PERCENTAGE,
            )

    fhr = mimo_prepared.block_input[:, :, 1].copy()
    up = mimo_prepared.block_input[:, :, 0].copy()
    domain_starts = list(mimo_prepared.domain_start)
    sample_weights = mimo_prepared.sample_weights
    return fhr, up, domain_starts, sample_weights


def _sanitize_signals(
    fhr: np.ndarray, up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate bad values, clamp range, flush denormals. In-place.

    Args:
        fhr: FHR array ``(N, samples)``.
        up: UP array ``(N, samples)``.

    Returns:
        ``(fhr, up)`` sanitized in-place.
    """
    interpolate_bad_values(fhr)
    interpolate_bad_values(up)
    fhr = np.clip(fhr, 0, 500).astype(np.float32)
    up = np.clip(up, -50, 500).astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    fhr[(fhr != 0) & (np.abs(fhr) < tiny)] = 0.0
    up[(up != 0) & (np.abs(up) < tiny)] = 0.0
    return fhr, up


def _quality_filter_segments(
    fhr: np.ndarray,
    up: np.ndarray,
    sample_weights: np.ndarray,
    domain_starts: list,
) -> Tuple[List[int], int, int]:
    """Apply weight threshold and flat region detection.

    Args:
        fhr: ``(N, samples)``
        up: ``(N, samples)``
        sample_weights: ``(N, n_dec)``
        domain_starts: Per-segment domain_start values.

    Returns:
        ``(valid_indices, n_low_weight, n_flat_region)``
    """
    n_low_weight = 0
    n_flat_region = 0
    valid_indices: List[int] = []

    for i in range(fhr.shape[0]):
        if np.mean(sample_weights[i, :]) < WEIGHT_THRESHOLD:
            n_low_weight += 1
            continue

        fhr_flat = find_flat_regions(fhr[i, :], tolerance=FLAT_TOLERANCE)
        up_flat = find_flat_regions(up[i, :], tolerance=FLAT_TOLERANCE)
        fhr_lens = [end - start + 1 for start, end in fhr_flat]
        up_lens = [end - start + 1 for start, end in up_flat]
        max_flat_fhr = max(fhr_lens, default=0)
        max_flat_up = max(up_lens, default=0)
        total_flat_fhr = sum(l for l in fhr_lens if l >= 240)

        if max_flat_fhr > 480 or max_flat_up > 1200 or total_flat_fhr > 1200:
            n_flat_region += 1
        else:
            valid_indices.append(i)

    return valid_indices, n_low_weight, n_flat_region


def prescreen_guid_6h(
    record_path: str,
    subgroup: str,
    tlo_hours: float,
    second_stage_hours: float,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Prescreen a single GUID for valid signal in the last 6 hours.

    Runs full MIMO + quality pipeline with a restricted domain range, then
    filters to the 6 h window and removes post-delivery segments.

    Args:
        record_path: Full path to .mat file.
        subgroup: Internal subgroup name.
        tlo_hours: Labor onset in hours (NaN if missing).
        second_stage_hours: Second stage onset in hours (NaN if missing).
        verbose: Suppress MIMO output when False.

    Returns:
        Dict with screening results (see output CSV schema in the plan).
    """
    guid_key = os.path.splitext(os.path.basename(record_path))[0]

    def _error(msg):
        return {
            "guid": guid_key,
            "subgroup": subgroup,
            "record_path": record_path,
            "n_total_segments": 0,
            "n_after_dedup": 0,
            "n_valid_segments_6h": 0,
            "n_low_weight": 0,
            "n_flat_region": 0,
            "n_duplicate": 0,
            "estimated_valid_hours_6h": 0.0,
            "has_tlo": not math.isnan(tlo_hours),
            "tlo_hours": tlo_hours,
            "has_second_stage": not math.isnan(second_stage_hours),
            "second_stage_hours": second_stage_hours,
            "domain_start_min": float("nan"),
            "domain_start_max": float("nan"),
            "n_post_delivery": 0,
            "eligible_2h": False,
            "eligible_3h": False,
            "error": True,
            "error_msg": msg,
        }

    try:
        fhr, up, domain_starts, sample_weights = _run_mimo_pipeline(
            record_path, MIN_DOMAIN_START_SCREENING, verbose
        )
        n_total = fhr.shape[0]

        fhr, up = _sanitize_signals(fhr, up)

        keep_idx, removed_idx, _ = deduplicate_segments(
            domain_starts, sample_weights
        )
        n_duplicate = len(removed_idx)
        if removed_idx:
            fhr = fhr[keep_idx]
            up = up[keep_idx]
            sample_weights = sample_weights[keep_idx]
            domain_starts = [domain_starts[i] for i in keep_idx]
        n_after_dedup = fhr.shape[0]

        valid_idx, n_low_weight, n_flat_region = _quality_filter_segments(
            fhr, up, sample_weights, domain_starts
        )

        # Filter to 6 h window and exclude post-delivery
        valid_6h_ds: List[float] = []
        n_post_delivery = 0
        for i in valid_idx:
            ds = domain_starts[i]
            if ds >= 0:
                n_post_delivery += 1
                continue
            if ds > MIN_DOMAIN_START_SCREENING:
                valid_6h_ds.append(ds)

        n_valid_6h = len(valid_6h_ds)
        if n_valid_6h > 0:
            est_hours = (
                (n_valid_6h - 1) * STEP_DURATION_SEC + SEGMENT_DURATION_SEC
            ) / 3600.0
            ds_min = min(valid_6h_ds)
            ds_max = max(valid_6h_ds)
        else:
            est_hours = 0.0
            ds_min = float("nan")
            ds_max = float("nan")

        return {
            "guid": guid_key,
            "subgroup": subgroup,
            "record_path": record_path,
            "n_total_segments": n_total,
            "n_after_dedup": n_after_dedup,
            "n_valid_segments_6h": n_valid_6h,
            "n_low_weight": n_low_weight,
            "n_flat_region": n_flat_region,
            "n_duplicate": n_duplicate,
            "estimated_valid_hours_6h": est_hours,
            "has_tlo": not math.isnan(tlo_hours),
            "tlo_hours": tlo_hours,
            "has_second_stage": not math.isnan(second_stage_hours),
            "second_stage_hours": second_stage_hours,
            "domain_start_min": ds_min,
            "domain_start_max": ds_max,
            "n_post_delivery": n_post_delivery,
            "eligible_2h": est_hours >= MIN_VALID_HOURS_UNHEALTHY,
            "eligible_3h": est_hours >= MIN_VALID_HOURS_HEALTHY,
            "error": False,
            "error_msg": "",
        }
    except Exception as e:
        return _error(str(e))


def prescreen_all_guids(
    records_base_path: str,
    tlo_csv_path: str,
    output_csv_path: str,
    verbose: bool = True,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Discover all GUIDs and prescreen each for 6 h valid signal.

    Args:
        records_base_path: Root directory with StudyGroup subfolders.
        tlo_csv_path: Complete CSV with TLO + second stage data.
        output_csv_path: Where to save the screening results CSV.
        verbose: Verbosity flag.
        num_workers: Parallel workers (default ``min(cpu_count, 8)``).

    Returns:
        DataFrame with one row per GUID, full screening details.
    """
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    # Load TLO/second-stage lookup
    labor_map, ss_map = load_csv_metadata(tlo_csv_path, verbose)

    # Discover .mat files
    all_jobs: List[Tuple[str, str, float, float]] = []
    for folder_name, subgroup in FOLDER_TO_SUBGROUP.items():
        efm_dir = os.path.join(records_base_path, folder_name, "EFMOut")
        if not os.path.isdir(efm_dir):
            logger.warning(f"Folder not found, skipping: {efm_dir}")
            continue
        for fname in sorted(os.listdir(efm_dir)):
            if not fname.endswith(".mat"):
                continue
            fpath = os.path.join(efm_dir, fname)
            guid_key = _normalize_guid(os.path.splitext(fname)[0])
            tlo_h = labor_map.get(guid_key, float("nan"))
            ss_h = ss_map.get(guid_key, float("nan"))
            if not math.isnan(tlo_h):
                tlo_h = tlo_h / 3600.0  # back to hours for storage
            if not math.isnan(ss_h):
                ss_h = ss_h / 3600.0
            all_jobs.append((fpath, subgroup, tlo_h, ss_h))

    logger.info(
        f"Prescreening {len(all_jobs)} GUIDs across "
        f"{len(FOLDER_TO_SUBGROUP)} subgroups (workers={num_workers})"
    )

    results: List[Dict[str, Any]] = []
    fn = partial(
        _prescreen_worker,
        verbose=verbose,
    )

    if num_workers <= 1:
        for job in tqdm(all_jobs, desc="Prescreening"):
            results.append(fn(job))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(fn, job): job for job in all_jobs}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Prescreening",
                disable=False,
            ):
                results.append(future.result())

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Screening results saved to {output_csv_path}")

    # Summary
    n_eligible_2h = df["eligible_2h"].sum()
    n_eligible_3h = df["eligible_3h"].sum()
    n_error = df["error"].sum()
    logger.info(
        f"Prescreening done: {len(df)} total, "
        f"{n_eligible_2h} eligible(>=2h), {n_eligible_3h} eligible(>=3h), "
        f"{n_error} errors"
    )
    for sg in sorted(df["subgroup"].unique()):
        sg_df = df[df["subgroup"] == sg]
        sg_2h = sg_df["eligible_2h"].sum()
        sg_3h = sg_df["eligible_3h"].sum()
        sg_err = sg_df["error"].sum()
        logger.info(
            f"  {sg:<28} {len(sg_df):>5} total, "
            f"{sg_2h:>5} elig(2h), {sg_3h:>5} elig(3h), {sg_err:>3} errors"
        )

    return df


def _prescreen_worker(
    job: Tuple[str, str, float, float], verbose: bool = True
) -> Dict[str, Any]:
    """Worker wrapper for prescreen_guid_6h (unpacks tuple).

    Args:
        job: ``(record_path, subgroup, tlo_hours, second_stage_hours)``
        verbose: Verbosity flag.

    Returns:
        Screening result dict.
    """
    record_path, subgroup, tlo_h, ss_h = job
    return prescreen_guid_6h(record_path, subgroup, tlo_h, ss_h, verbose)


# ============================================================================
# Step 2: GUID selection for classification
# ============================================================================
def _sample_with_tlo_constraint(
    pool_df: pd.DataFrame,
    target_n: int,
    tlo_ratio: float,
    rng: random.Random,
) -> List[str]:
    """Sample GUIDs from a pool enforcing a target TLO-present ratio.

    Args:
        pool_df: DataFrame slice with ``has_tlo`` and ``record_path`` columns.
        target_n: Number of GUIDs to select.
        tlo_ratio: Desired fraction with TLO (e.g. 0.75).
        rng: Seeded random instance.

    Returns:
        List of selected record paths (length <= *target_n*).
    """
    if target_n <= 0 or pool_df.empty:
        return []
    pool_with = pool_df[pool_df["has_tlo"] == True]["record_path"].tolist()
    pool_without = pool_df[pool_df["has_tlo"] == False]["record_path"].tolist()

    n_with = round(target_n * tlo_ratio)
    n_without = target_n - n_with

    # Relax if insufficient TLO GUIDs
    if len(pool_with) < n_with:
        n_with = len(pool_with)
        n_without = min(target_n - n_with, len(pool_without))
    if len(pool_without) < n_without:
        n_without = len(pool_without)
        n_with = min(target_n - n_without, len(pool_with))

    rng.shuffle(pool_with)
    rng.shuffle(pool_without)
    return pool_with[:n_with] + pool_without[:n_without]


def select_classification_guids(
    screening_df: pd.DataFrame,
    test_mode: str = "augmented",
    bg_cls_fraction: float = HEALTHY_BG_CLS_FRACTION,
    test_holdout_fraction: float = TEST_HOLDOUT_FRACTION,
    no_bg_tlo_ratio: float = HEALTHY_NO_BG_TLO_RATIO,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Select GUIDs for classification with balanced train/val and representative test.

    Unhealthy: all eligible (>=2 h).  Healthy BG: 10 % of eligible (>=3 h) per
    CS/NoCS, natural TLO ratio.  Healthy no-BG: added to balance train/val
    (75/25 TLO).  Test: population-proportional healthy subgroup distribution.

    Two test modes:
        * **holdout** — fixed test holdout (~10 %) before 10-fold CV.
        * **augmented** — standard 10-fold 80/10/10 on a balanced core pool;
          each fold's test partition augmented with extra healthy no-BG GUIDs
          that never appear in train/val.

    Args:
        screening_df: DataFrame from prescreening (all GUIDs).
        test_mode: ``"augmented"`` (default) or ``"holdout"``.
        bg_cls_fraction: Fraction of eligible BG to select for classification.
        test_holdout_fraction: Fraction held out for test per subgroup.
        no_bg_tlo_ratio: Enforced TLO ratio for no-BG subgroups.
        random_state: Random seed.
        verbose: Verbosity flag.

    Returns:
        Dict with keys ``test_mode``, ``trainval``, ``test``, ``tlo_map``,
        ``pretraining_bg_cs``, ``pretraining_bg_no_cs``, ``stats``.
    """
    rng = random.Random(random_state)

    # ------------------------------------------------------------------
    # A. Filter eligible pools with per-subgroup thresholds
    # ------------------------------------------------------------------
    no_err = screening_df["error"] == False

    unhealthy_eligible = screening_df[
        no_err
        & (screening_df["eligible_2h"] == True)
        & screening_df["subgroup"].isin(UNHEALTHY_SUBGROUPS)
    ].copy()

    healthy_bg_eligible = screening_df[
        no_err
        & (screening_df["eligible_3h"] == True)
        & screening_df["subgroup"].isin(BG_SUBGROUPS)
    ].copy()

    healthy_no_bg_eligible = screening_df[
        no_err
        & (screening_df["eligible_3h"] == True)
        & screening_df["subgroup"].isin(NO_BG_SUBGROUPS)
    ].copy()

    # ------------------------------------------------------------------
    # B. Unhealthy: ALL eligible
    # ------------------------------------------------------------------
    unhealthy_by_sg: Dict[str, List[str]] = {}
    for sg, grp in unhealthy_eligible.groupby("subgroup"):
        paths = grp["record_path"].tolist()
        rng.shuffle(paths)
        unhealthy_by_sg[sg] = paths
    n_unhealthy = sum(len(v) for v in unhealthy_by_sg.values())

    # ------------------------------------------------------------------
    # C. Healthy BG: 10 % per CS/NoCS, natural TLO ratio
    # ------------------------------------------------------------------
    bg_cls: Dict[str, List[str]] = {}
    bg_pretrain: Dict[str, List[str]] = {}
    for sg in ["healthy_bg_cs", "healthy_bg_no_cs"]:
        pool = healthy_bg_eligible[healthy_bg_eligible["subgroup"] == sg]
        pool_paths = pool["record_path"].tolist()
        rng.shuffle(pool_paths)
        n_cls = max(round(len(pool_paths) * bg_cls_fraction), 1) if pool_paths else 0
        bg_cls[sg] = pool_paths[:n_cls]
        bg_pretrain[sg] = pool_paths[n_cls:]

    n_bg_cls = sum(len(v) for v in bg_cls.values())

    # ------------------------------------------------------------------
    # D. Test holdout / core pool
    # ------------------------------------------------------------------
    trainval: Dict[str, List[str]] = {}
    test_guids: Dict[str, List[str]] = {}

    if test_mode == "holdout":
        # Hold out ~10 % per subgroup for test
        for sg, paths in unhealthy_by_sg.items():
            n_test = max(round(len(paths) * test_holdout_fraction), 1) if paths else 0
            test_guids[sg] = paths[:n_test]
            trainval[sg] = paths[n_test:]
        for sg in ["healthy_bg_cs", "healthy_bg_no_cs"]:
            paths = bg_cls[sg]
            n_test = max(round(len(paths) * test_holdout_fraction), 1) if paths else 0
            test_guids[sg] = paths[:n_test]
            trainval[sg] = paths[n_test:]
    else:
        # Augmented: all go to core pool
        for sg, paths in unhealthy_by_sg.items():
            trainval[sg] = list(paths)
        for sg in ["healthy_bg_cs", "healthy_bg_no_cs"]:
            trainval[sg] = list(bg_cls[sg])

    n_unhealthy_trainval = sum(
        len(v) for sg, v in trainval.items() if sg in UNHEALTHY_SUBGROUPS
    )
    n_bg_trainval = sum(
        len(v) for sg, v in trainval.items() if sg in BG_SUBGROUPS
    )

    # ------------------------------------------------------------------
    # E. Balance train/val with healthy_no_bg
    # ------------------------------------------------------------------
    deficit = n_unhealthy_trainval - n_bg_trainval

    # Eligible pools for no-BG subgroups
    no_bg_pools: Dict[str, pd.DataFrame] = {}
    for sg in NO_BG_SUBGROUPS:
        no_bg_pools[sg] = healthy_no_bg_eligible[
            healthy_no_bg_eligible["subgroup"] == sg
        ]

    total_no_bg_eligible = sum(len(p) for p in no_bg_pools.values())

    if deficit > 0 and total_no_bg_eligible > 0:
        frac_cs = len(no_bg_pools["healthy_no_bg_cs"]) / total_no_bg_eligible
        n_no_bg_cs_tv = min(
            round(deficit * frac_cs), len(no_bg_pools["healthy_no_bg_cs"])
        )
        n_no_bg_no_cs_tv = min(
            deficit - n_no_bg_cs_tv, len(no_bg_pools["healthy_no_bg_no_cs"])
        )
        # If capped, try to compensate
        if n_no_bg_cs_tv + n_no_bg_no_cs_tv < deficit:
            extra_cs = min(
                deficit - n_no_bg_cs_tv - n_no_bg_no_cs_tv,
                len(no_bg_pools["healthy_no_bg_cs"]) - n_no_bg_cs_tv,
            )
            n_no_bg_cs_tv += max(extra_cs, 0)
            extra_no_cs = min(
                deficit - n_no_bg_cs_tv - n_no_bg_no_cs_tv,
                len(no_bg_pools["healthy_no_bg_no_cs"]) - n_no_bg_no_cs_tv,
            )
            n_no_bg_no_cs_tv += max(extra_no_cs, 0)

        for sg, n_tv in [
            ("healthy_no_bg_cs", n_no_bg_cs_tv),
            ("healthy_no_bg_no_cs", n_no_bg_no_cs_tv),
        ]:
            trainval[sg] = _sample_with_tlo_constraint(
                no_bg_pools[sg], n_tv, no_bg_tlo_ratio, rng
            )
    else:
        for sg in NO_BG_SUBGROUPS:
            trainval[sg] = []

    # Track which no_bg paths are used for train/val (for exclusion later)
    used_no_bg_paths: set = set()
    for sg in NO_BG_SUBGROUPS:
        used_no_bg_paths.update(trainval.get(sg, []))

    # ------------------------------------------------------------------
    # F. Healthy no-BG for test (population-proportional)
    # ------------------------------------------------------------------
    eligible_healthy_counts = {
        "healthy_bg_cs": len(
            healthy_bg_eligible[healthy_bg_eligible["subgroup"] == "healthy_bg_cs"]
        ),
        "healthy_bg_no_cs": len(
            healthy_bg_eligible[healthy_bg_eligible["subgroup"] == "healthy_bg_no_cs"]
        ),
        "healthy_no_bg_cs": len(no_bg_pools["healthy_no_bg_cs"]),
        "healthy_no_bg_no_cs": len(no_bg_pools["healthy_no_bg_no_cs"]),
    }
    total_eligible_healthy = sum(eligible_healthy_counts.values())

    if total_eligible_healthy > 0:
        p = {
            sg: eligible_healthy_counts[sg] / total_eligible_healthy
            for sg in eligible_healthy_counts
        }
    else:
        p = {sg: 0.25 for sg in eligible_healthy_counts}

    if test_mode == "holdout":
        # BG test counts are known from the holdout split in Step D
        n_bg_test = sum(len(test_guids.get(sg, [])) for sg in BG_SUBGROUPS)
    else:
        # Augmented: estimate per-fold BG test count (~1/N_FOLDS of core BG)
        n_bg_in_core = sum(len(trainval.get(sg, [])) for sg in BG_SUBGROUPS)
        n_bg_test = max(round(n_bg_in_core / N_FOLDS), 1) if n_bg_in_core else 0

    p_bg = p["healthy_bg_cs"] + p["healthy_bg_no_cs"]

    if p_bg > 0 and n_bg_test > 0:
        total_healthy_test = round(n_bg_test / p_bg)
    else:
        total_healthy_test = n_bg_test

    for sg in NO_BG_SUBGROUPS:
        n_need = round(total_healthy_test * p[sg])
        remaining = no_bg_pools[sg][
            ~no_bg_pools[sg]["record_path"].isin(used_no_bg_paths)
        ]
        n_need = min(n_need, len(remaining))
        test_guids[sg] = _sample_with_tlo_constraint(
            remaining, n_need, no_bg_tlo_ratio, rng
        )

    # ------------------------------------------------------------------
    # G. Build stats & log
    # ------------------------------------------------------------------
    n_trainval_healthy = sum(
        len(v) for sg, v in trainval.items() if sg in HEALTHY_SUBGROUPS
    )
    stats = {
        "n_unhealthy": n_unhealthy,
        "n_bg_cls": n_bg_cls,
        "n_unhealthy_trainval": n_unhealthy_trainval,
        "n_bg_trainval": n_bg_trainval,
        "n_trainval_healthy": n_trainval_healthy,
        "deficit_filled": deficit,
        "test_mode": test_mode,
    }

    if verbose:
        logger.info("=" * 80)
        logger.info("GUID SELECTION SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"{'Subgroup':<28} {'Eligible':>8} {'TrainVal':>8} "
            f"{'Test':>6} {'Pretrain':>8}"
        )
        logger.info("-" * 80)
        all_sgs = sorted(
            set(list(trainval.keys()) + list(test_guids.keys()))
        )
        for sg in all_sgs:
            el = eligible_healthy_counts.get(sg, 0)
            if sg in UNHEALTHY_SUBGROUPS:
                el = len(
                    unhealthy_eligible[unhealthy_eligible["subgroup"] == sg]
                )
            tv = len(trainval.get(sg, []))
            ts = len(test_guids.get(sg, []))
            pt = len(bg_pretrain.get(sg, []))
            logger.info(
                f"  {sg:<28} {el:>8} {tv:>8} {ts:>6} {pt:>8}"
            )
        logger.info("-" * 80)
        logger.info(
            f"  Total unhealthy trainval: {n_unhealthy_trainval}"
        )
        logger.info(f"  Total healthy   trainval: {n_trainval_healthy}")
        logger.info(
            f"  Total test GUIDs: "
            f"{sum(len(v) for v in test_guids.values())}"
        )
        logger.info(
            f"  Pretraining BG leftovers: "
            f"{sum(len(v) for v in bg_pretrain.values())}"
        )
        logger.info(f"  Test mode: {test_mode}")
        logger.info("=" * 80)

    # Build path -> tlo_hours lookup for stratification in create_cv_splits
    all_trainval_paths = set()
    for sg_paths in trainval.values():
        all_trainval_paths.update(sg_paths)
    path_to_tlo = screening_df.set_index("record_path")["tlo_hours"].to_dict()
    tlo_map = {p: path_to_tlo[p] for p in all_trainval_paths if p in path_to_tlo}

    return {
        "test_mode": test_mode,
        "trainval": trainval,
        "test": test_guids,
        "tlo_map": tlo_map,
        "pretraining_bg_cs": bg_pretrain.get("healthy_bg_cs", []),
        "pretraining_bg_no_cs": bg_pretrain.get("healthy_bg_no_cs", []),
        "stats": stats,
    }


def _compute_duration_bins(
    paths: List[str],
    tlo_map: Dict[str, float],
    n_bins: int = N_DURATION_BINS,
) -> np.ndarray:
    """Bin labour durations into quantile-based groups for stratification.

    Args:
        paths: List of record file paths.
        tlo_map: Mapping ``record_path -> tlo_hours`` (negative hours
            before delivery, NaN if unknown).
        n_bins: Number of quantile bins for known durations.

    Returns:
        Integer array of bin labels aligned with *paths*.  Known durations
        are assigned to bins ``0 .. n_bins-1`` (quantile tertiles);
        unknown (NaN / missing) durations are assigned to bin ``n_bins``.
        If no known durations exist, returns all zeros so that
        stratification degenerates to plain KFold.
    """
    durations = np.array([
        abs(tlo_map.get(p, float("nan"))) for p in paths
    ])
    known_mask = ~np.isnan(durations)
    n_known = known_mask.sum()

    if n_known == 0:
        return np.zeros(len(paths), dtype=int)

    # Quantile boundaries from known durations
    known_durations = durations[known_mask]
    quantiles = np.linspace(1 / n_bins, 1 - 1 / n_bins, n_bins - 1)
    boundaries = np.quantile(known_durations, quantiles)

    bins = np.empty(len(paths), dtype=int)
    bins[known_mask] = np.digitize(known_durations, boundaries)  # 0..n_bins-1
    bins[~known_mask] = n_bins  # "unknown" bin
    return bins


# ============================================================================
# Step 3: K-Fold CV splits
# ============================================================================
def create_cv_splits(
    selection_result: Dict[str, Any],
    n_splits: int = N_FOLDS,
    val_ratio: float = VAL_RATIO,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Create stratified-by-subgroup K-fold CV splits.

    Within each subgroup, folds are stratified by **labour duration** (derived
    from TLO) using quantile-based bins.  If stratification is not feasible
    (too few GUIDs in a duration bin), falls back to plain ``KFold``.

    Supports two test modes:

    * **holdout** — ``KFold`` on the train/val pool gives 90/10 train/val per
      fold. The fixed test holdout is returned separately.
    * **augmented** (default) — Standard 80/10/10 via ``KFold`` + inner
      ``train_test_split``. Each fold's test partition is augmented with extra
      healthy no-BG GUIDs that never appear in train/val.

    Args:
        selection_result: Dict returned by ``select_classification_guids()``.
            Must contain ``tlo_map`` (path → tlo_hours) for duration
            stratification; if absent, falls back to plain KFold.
        n_splits: Number of CV folds.
        val_ratio: Inner val fraction (augmented mode only).
        random_state: Seed for reproducibility.

    Returns:
        Dict with ``test_mode``, ``folds``, and ``test`` (holdout) or
        ``test_augmentation`` (augmented).
    """
    test_mode = selection_result["test_mode"]
    trainval_data = selection_result["trainval"]
    test_data = selection_result["test"]
    tlo_map: Dict[str, float] = selection_result.get("tlo_map", {})

    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )
    kf_fallback = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )

    # Only split subgroups that actually have GUIDs
    active_groups = {
        sg: paths for sg, paths in trainval_data.items() if paths
    }

    # Build per-subgroup splits, handling small subgroups (< n_splits)
    # that would crash KFold.  For those, use leave-one-out cycling:
    # first n folds each hold out one sample; remaining folds put
    # everything in train (empty held-out set).
    rng_state = np.random.RandomState(random_state)
    splits_per_group: Dict[str, List[Tuple]] = {}
    for sg, paths in active_groups.items():
        n = len(paths)
        if n >= n_splits:
            duration_bins = _compute_duration_bins(paths, tlo_map)
            bin_counts = np.bincount(duration_bins)
            if np.all(bin_counts[bin_counts > 0] >= n_splits):
                try:
                    splits_per_group[sg] = list(
                        skf.split(paths, duration_bins)
                    )
                except ValueError:
                    logger.warning(
                        f"StratifiedKFold failed for {sg}; "
                        f"falling back to KFold"
                    )
                    splits_per_group[sg] = list(kf_fallback.split(paths))
            else:
                logger.warning(
                    f"Subgroup {sg}: some duration bins too small for "
                    f"StratifiedKFold (bin counts: {bin_counts.tolist()}); "
                    f"using KFold"
                )
                splits_per_group[sg] = list(kf_fallback.split(paths))
        else:
            indices = np.arange(n)
            rng_state.shuffle(indices)
            sg_splits: List[Tuple] = []
            for fi in range(n_splits):
                if fi < n:
                    held = np.array([indices[fi]])
                    rest = np.concatenate(
                        [indices[:fi], indices[fi + 1:]]
                    )
                else:
                    held = np.array([], dtype=int)
                    rest = indices.copy()
                sg_splits.append((rest, held))
            splits_per_group[sg] = sg_splits
            logger.warning(
                f"Subgroup {sg} has only {n} GUIDs (< {n_splits} folds); "
                f"using leave-one-out cycling"
            )

    if test_mode == "holdout":
        # KFold directly gives 90 % train / 10 % val — no inner split
        folds: Dict[str, Dict] = {}
        for fold_idx in range(n_splits):
            fold_name = f"fold_{fold_idx + 1}"
            fold_data: Dict[str, Dict[str, List[str]]] = {
                "train": {}, "val": {},
            }
            for sg, splits in splits_per_group.items():
                train_idx, val_idx = splits[fold_idx]
                fold_data["train"][sg] = [
                    active_groups[sg][i] for i in train_idx
                ]
                fold_data["val"][sg] = [
                    active_groups[sg][i] for i in val_idx
                ]
            folds[fold_name] = fold_data

        return {
            "test_mode": "holdout",
            "folds": folds,
            "test": test_data,
        }

    else:  # augmented
        folds = {}
        for fold_idx in range(n_splits):
            fold_name = f"fold_{fold_idx + 1}"
            fold_data = {"train": {}, "val": {}, "test": {}}

            for sg, splits in splits_per_group.items():
                train_val_idx, test_idx = splits[fold_idx]
                core_test = [active_groups[sg][i] for i in test_idx]

                # Guard: inner split needs >= 2 samples in train_val
                if len(train_val_idx) >= 2:
                    # Stratify inner split by duration bins when feasible
                    tv_paths = [
                        active_groups[sg][i] for i in train_val_idx
                    ]
                    tv_bins = _compute_duration_bins(tv_paths, tlo_map)
                    tv_bin_counts = np.bincount(tv_bins)
                    try:
                        if tv_bin_counts[tv_bin_counts > 0].min() >= 2:
                            train_idx, val_idx = train_test_split(
                                train_val_idx,
                                test_size=val_ratio,
                                shuffle=True,
                                random_state=random_state,
                                stratify=tv_bins,
                            )
                        else:
                            raise ValueError("bin too small")
                    except ValueError:
                        train_idx, val_idx = train_test_split(
                            train_val_idx,
                            test_size=val_ratio,
                            shuffle=True,
                            random_state=random_state,
                        )
                else:
                    train_idx = train_val_idx
                    val_idx = np.array([], dtype=int)

                fold_data["train"][sg] = [
                    active_groups[sg][i] for i in train_idx
                ]
                fold_data["val"][sg] = [
                    active_groups[sg][i] for i in val_idx
                ]
                fold_data["test"][sg] = core_test

            # Append test augmentation GUIDs (always in test, never in
            # train/val)
            for sg, aug_paths in test_data.items():
                if sg in fold_data["test"]:
                    fold_data["test"][sg].extend(aug_paths)
                else:
                    fold_data["test"][sg] = list(aug_paths)

            folds[fold_name] = fold_data

        return {
            "test_mode": "augmented",
            "folds": folds,
            "test_augmentation": test_data,
        }


# ============================================================================
# Step 4: HDF5 dataset creation from records list
# ============================================================================
def _transform_causal_record(
    torch_bank: CausalTorchBank,
    fhr: np.ndarray,
    up: np.ndarray,
    target_pairs: np.ndarray,
    source_pairs: np.ndarray,
    channel_plan: Dict[str, CausalChannelPlan],
    scatter_batch_size: int,
) -> Tuple[List[Optional[Dict[str, np.ndarray]]], Dict[int, str]]:
    """Transform one record's segments causally, in batches, isolating any that fail.

    The whole causal chain is a single forward call, so where the two-sided path runs four passes
    and explodes each into four per-segment dicts, this is one call and a slice.

    A ``RuntimeError`` from the transform is an out-of-memory guard: it is a property of the batch,
    not of any one segment, so the batch is retried a segment at a time and only a segment that
    fails **alone** is given up on. Storing a retried segment beside its peers is only sound
    because the chain is batch-invariant — every operation in it is elementwise or an FFT along
    time — which is why that invariance is asserted rather than assumed.

    Args:
        torch_bank: The realised causal bank, already on the build device.
        fhr: Raw fetal heart rate for the record's kept segments, ``(n_valid, len_signal)``.
        up: Raw uterine pressure, same shape.
        target_pairs: ``(n, 2)`` phase pairs for ``fhr_ph``, in stored channel order.
        source_pairs: ``(n, 2)`` phase pairs for ``up_ph``.
        channel_plan: The stored plan; the transform gathers its channels.
        scatter_batch_size: Segments per forward pass.

    Returns:
        ``(blocks, failures)``: one ``{block: (C, T)}`` dict per segment — ``None`` where the
        segment failed — and ``{segment_index: error message}`` for those that did.
    """
    n_valid = int(fhr.shape[0])
    blocks: List[Optional[Dict[str, np.ndarray]]] = [None] * n_valid
    failures: Dict[int, str] = {}

    def _run(start: int, stop: int) -> Dict[str, np.ndarray]:
        return transform_batch_numpy(
            torch_bank, fhr[start:stop], up[start:stop], target_pairs, source_pairs,
            plan=channel_plan,
        )

    for batch_start in range(0, n_valid, scatter_batch_size):
        batch_end = min(batch_start + scatter_batch_size, n_valid)
        try:
            batched = _run(batch_start, batch_end)
            for local_j, global_j in enumerate(range(batch_start, batch_end)):
                blocks[global_j] = {
                    name: batched[name][local_j] for name in CAUSAL_BLOCKS
                }
        except RuntimeError:
            for global_j in range(batch_start, batch_end):
                try:
                    alone = _run(global_j, global_j + 1)
                    blocks[global_j] = {name: alone[name][0] for name in CAUSAL_BLOCKS}
                except RuntimeError as segment_error:
                    failures[global_j] = str(segment_error)
    return blocks, failures


def create_hdf5_dataset_from_records_list(
    hdf5_path: str,
    records_list: List[str],
    cs_label: bool,
    bg_label: bool,
    pre_defined_target: int,
    precomputed_masks: Dict[str, Any],
    labor_onset_map: Dict[str, float],
    second_stage_map: Dict[str, float],
    base_block_size: int = BASE_BLOCK_SIZE,
    overlap_percentage: float = OVERLAP_PERCENTAGE,
    device: Optional[torch.device] = None,
    run_guid_analysis: bool = False,
    scatter_batch_size: int = 16,
    verbose: bool = True,
    transform: str = TWO_SIDED,
) -> List[str]:
    """Process a list of .mat files and write segments to an HDF5 file.

    Runs the full MIMO + sanitize + dedup + quality + post-delivery-skip +
    scattering v3 pipeline for each record, then batch-writes valid segments.

    Args:
        hdf5_path: Output HDF5 file (must already be created via
            ``create_initial_hdf5``).
        records_list: List of .mat file paths.
        cs_label: Caesarean section flag for all records.
        bg_label: Blood gas flag for all records.
        pre_defined_target: Class target (1=HEALTHY, 2=ACIDOSIS, 3=HIE).
        precomputed_masks: Dict from ``compute_scattering_masks``.
        labor_onset_map: Normalized GUID -> TLO in seconds.
        second_stage_map: Normalized GUID -> second stage in seconds.
        base_block_size: Base block size for MIMO.
        overlap_percentage: Overlap fraction.
        device: Torch device for scattering.
        run_guid_analysis: Collect per-GUID tracking data.
        scatter_batch_size: Scattering batch size.
        verbose: Verbosity flag.
        transform: ``'two_sided'`` or ``'causal'``; must match the variant the
            file was created for and the masks were computed for.

    Returns:
        List of record paths that errored.

    Raises:
        ValueError: On an unknown *transform*, or if it disagrees with the one
            the masks were computed for.
    """
    validate_transform(transform)
    if precomputed_masks.get("transform", TWO_SIDED) != transform:
        raise ValueError(
            f"transform={transform!r} but the masks were computed for "
            f"{precomputed_masks.get('transform', TWO_SIDED)!r}; the causal variant needs its "
            f"channel plan, which only compute_scattering_masks(transform='causal') builds"
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scattering_T = 16
    signal_length = int(base_block_size * 1.5)
    causal = transform == CAUSAL

    if causal:
        # No KymatioPhaseScattering1D on this path: the causal chain is a different filter bank,
        # and building the two-sided one anyway would hold a second bank's worth of device memory
        # for the whole build. The channel plan gathers the surviving channels inside the
        # transform, so the blocks come back at their stored widths.
        channel_plan = precomputed_masks["channel_plan"]
        torch_bank = CausalTorchBank(
            precomputed_masks["causal_bank"], device, n_signal=signal_length
        )
        # From the PhaseChannelSelection the sel_* provenance is written from, never from a second
        # selector: channel c of the data and channel c of the provenance are then the same channel
        # by construction rather than by two implementations agreeing.
        target_pairs = _selection_pairs(precomputed_masks["fhr_ph_selection"])
        source_pairs = _selection_pairs(precomputed_masks["up_ph_selection"])
        # No pair-axis check: this path indexes responses by pair *index* rather than masking a
        # pair axis, so there is no mask length that could disagree with a transform.
        _validate_geometry(hdf5_path, resolve_channel_layout(precomputed_masks))
    else:
        # Q must match the bank the masks were built against in
        # compute_scattering_masks; the pair-axis check below enforces that, but
        # sharing the constant keeps the two from drifting in the first place.
        st_model = KymatioPhaseScattering1D(
            J=11, Q=SCATTERING_Q, T=scattering_T, shape=signal_length,
            device=device, tukey_alpha=None, max_order=1,
        )

        phase_mask = precomputed_masks["fhr_ph_selection"].mask.to(device)
        cross_mask = precomputed_masks["cross_mask"].to(device)
        up_phase_mask = precomputed_masks["up_ph_selection"].mask.to(device)

        _validate_geometry(
            hdf5_path,
            resolve_channel_layout(precomputed_masks),
            pair_masks={
                "fhr_ph": phase_mask, "fhr_up_ph": cross_mask, "up_ph": up_phase_mask,
            },
            n_pairs=len(st_model.i_idx),
        )

    errors_list: List[str] = []
    guid_tracking: Optional[Dict[str, GuidTrackingEntry]] = (
        {} if run_guid_analysis else None
    )

    for record in tqdm(records_list, desc=os.path.basename(hdf5_path)):
        try:
            default_ti = (
                (pre_defined_target - 1)
                if pre_defined_target is not None
                else None
            )
            mimo_adaptor = EarlyMaestraMimoAdaptor(
                do_transpose=True,
                process_targets=True,
                n_aux_labels=None,
                signal_indices=range(0, 2),
                n_input_chan=2,
                labels=["HIE", "ACIDOSIS", "HEALTHY"],
                up_shift_secs=-20,
                default_target_index=default_ti,
            )
            if verbose:
                mimo_adaptor.read_single_input(
                    record, out_dec_factor=16, out_dec_factor_offset=0,
                    target_is_onehot=True, dtype=np.float32,
                )
                mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                    batch_size=1, do_evaluate=True, align_left=True,
                    do_split=True, do_pad=True, do_reflect=True,
                    base_length=base_block_size, do_equalize=True,
                    do_merge=True,
                    min_domain_start=[MIN_DOMAIN_START_DATASET, MIN_DOMAIN_START_DATASET],
                    max_domain_start=[np.inf, np.inf],
                    overlap_percentage=overlap_percentage,
                )
            else:
                with suppress_stdout_stderr():
                    mimo_adaptor.read_single_input(
                        record, out_dec_factor=16, out_dec_factor_offset=0,
                        target_is_onehot=True, dtype=np.float32,
                    )
                    mimo_prepared, _ = mimo_adaptor.mimo.prepare_data(
                        batch_size=1, do_evaluate=True, align_left=True,
                        do_split=True, do_pad=True, do_reflect=True,
                        base_length=base_block_size, do_equalize=True,
                        do_merge=True,
                        min_domain_start=[MIN_DOMAIN_START_DATASET, MIN_DOMAIN_START_DATASET],
                        max_domain_start=[np.inf, np.inf],
                        overlap_percentage=overlap_percentage,
                    )

            fhr = mimo_prepared.block_input[:, :, 1].copy()
            up = mimo_prepared.block_input[:, :, 0].copy()
            domain_starts = list(mimo_prepared.domain_start)
            sample_weights = mimo_prepared.sample_weights

            fhr, up = _sanitize_signals(fhr, up)

            guid_key = os.path.splitext(os.path.basename(record))[0]
            normalized_key = _normalize_guid(guid_key)
            labor_onset_sec = labor_onset_map.get(normalized_key, float("nan"))
            ss_sec = second_stage_map.get(normalized_key, float("nan"))

            if guid_tracking is not None:
                guid_tracking[guid_key] = GuidTrackingEntry(
                    all_domain_starts=[float(ds) for ds in domain_starts],
                )

            # Dedup
            keep_idx, removed_idx, _ = deduplicate_segments(
                domain_starts, sample_weights
            )
            if removed_idx:
                if guid_tracking is not None:
                    guid_tracking[guid_key].skipped_duplicate.extend(
                        float(domain_starts[i]) for i in removed_idx
                    )
                fhr = fhr[keep_idx]
                up = up[keep_idx]
                sample_weights = sample_weights[keep_idx]
                domain_starts = [domain_starts[i] for i in keep_idx]

            # Quality filter
            valid_indices: List[int] = []
            for i in range(fhr.shape[0]):
                if np.mean(sample_weights[i, :]) < WEIGHT_THRESHOLD:
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_low_weight.append(
                            float(domain_starts[i])
                        )
                    continue
                fhr_flat = find_flat_regions(fhr[i, :], tolerance=FLAT_TOLERANCE)
                up_flat = find_flat_regions(up[i, :], tolerance=FLAT_TOLERANCE)
                fhr_lens = [end - start + 1 for start, end in fhr_flat]
                up_lens = [end - start + 1 for start, end in up_flat]
                max_flat_fhr = max(fhr_lens, default=0)
                max_flat_up = max(up_lens, default=0)
                total_flat_fhr = sum(l for l in fhr_lens if l >= 240)
                if (
                    max_flat_fhr > 480
                    or max_flat_up > 1200
                    or total_flat_fhr > 1200
                ):
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_flat_region.append(
                            float(domain_starts[i])
                        )
                    continue
                # Skip post-delivery segments
                if domain_starts[i] >= 0:
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_post_delivery.append(
                            float(domain_starts[i])
                        )
                    continue
                valid_indices.append(i)

            if not valid_indices:
                continue

            # Batched scattering
            valid_fhr = fhr[valid_indices]
            valid_up = up[valid_indices]
            n_valid = len(valid_indices)
            scatter_failed: set = set()
            causal_blocks: List[Optional[Dict[str, np.ndarray]]] = []

            st_phase_list = [None] * n_valid
            st_cross_list = [None] * n_valid
            st_up_phase_list = [None] * n_valid
            st_up_scatter_list = [None] * n_valid

            if causal:
                causal_blocks, causal_failures = _transform_causal_record(
                    torch_bank, valid_fhr, valid_up, target_pairs, source_pairs,
                    channel_plan, scatter_batch_size,
                )
                for seg_j, message in causal_failures.items():
                    orig_idx = valid_indices[seg_j]
                    logger.error(
                        f"{guid_key} seg {orig_idx} "
                        f"(epoch={domain_starts[orig_idx]}): "
                        f"scattering failed: {message}"
                    )
                    scatter_failed.add(seg_j)
                    if guid_tracking is not None:
                        guid_tracking[guid_key].skipped_scatter_failed.append(
                            float(domain_starts[orig_idx])
                        )
            else:
                st_input = torch.from_numpy(
                    np.stack([valid_fhr, valid_up], axis=1)
                ).float().to(device)

                for batch_start in range(0, n_valid, scatter_batch_size):
                    batch_end = min(batch_start + scatter_batch_size, n_valid)
                    batch = st_input[batch_start:batch_end]
                    try:
                        bp = st_model(
                            x=batch, compute_phase=True,
                            compute_cross_phase=False,
                            scattering_channel=0, phase_channels=[0],
                        )
                        bc = st_model(
                            x=batch, compute_phase=False,
                            compute_cross_phase=True,
                            scattering_channel=0, phase_channels=[0, 1],
                        )
                        bup = st_model(
                            x=batch, compute_phase=True,
                            compute_cross_phase=False,
                            scattering_channel=0, phase_channels=[1],
                        )
                        bus = st_model(
                            x=batch, compute_phase=False,
                            compute_cross_phase=False,
                            scattering_channel=1,
                        )
                        bs = batch.shape[0]
                        for lj in range(bs):
                            gj = batch_start + lj
                            st_phase_list[gj] = {
                                k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                                for k, v in bp.items()
                            }
                            st_cross_list[gj] = {
                                k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                                for k, v in bc.items()
                            }
                            st_up_phase_list[gj] = {
                                k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                                for k, v in bup.items()
                            }
                            st_up_scatter_list[gj] = {
                                k: (v[lj:lj+1] if isinstance(v, torch.Tensor) and v.shape[0] == bs else v)
                                for k, v in bus.items()
                            }
                    except RuntimeError:
                        for lj in range(batch.shape[0]):
                            gj = batch_start + lj
                            seg = st_input[gj:gj+1]
                            try:
                                sp = st_model(x=seg, compute_phase=True, compute_cross_phase=False, scattering_channel=0, phase_channels=[0])
                                sc = st_model(x=seg, compute_phase=False, compute_cross_phase=True, scattering_channel=0, phase_channels=[0, 1])
                                su = st_model(x=seg, compute_phase=True, compute_cross_phase=False, scattering_channel=0, phase_channels=[1])
                                sus = st_model(x=seg, compute_phase=False, compute_cross_phase=False, scattering_channel=1)
                                st_phase_list[gj] = sp
                                st_cross_list[gj] = sc
                                st_up_phase_list[gj] = su
                                st_up_scatter_list[gj] = sus
                            except RuntimeError as seg_err:
                                orig_idx = valid_indices[gj]
                                logger.error(
                                    f"{guid_key} seg {orig_idx} "
                                    f"(epoch={domain_starts[orig_idx]}): "
                                    f"scattering failed: {seg_err}"
                                )
                                scatter_failed.add(gj)
                                if guid_tracking is not None:
                                    guid_tracking[guid_key].skipped_scatter_failed.append(
                                        float(domain_starts[orig_idx])
                                    )

            # Collect valid scattered segments
            b_fhr, b_up = [], []
            b_fhr_st, b_fhr_ph, b_fhr_up_ph = [], [], []
            b_up_st, b_up_ph = [], []
            b_target, b_weight = [], []
            b_guid, b_epoch = [], []
            b_cs, b_bg, b_tlo, b_ss = [], [], [], []

            record_name = os.path.splitext(os.path.basename(record))[0]

            for seg_j in range(n_valid):
                if seg_j in scatter_failed:
                    continue
                orig_idx = valid_indices[seg_j]

                # Both variants end here holding the same thing: one ``(C, T)`` numpy array per
                # stored block, already at its stored width. The causal path has no cross-phase
                # entry at all, which is what keeps ``b_fhr_up_ph`` empty and the append guard
                # below meaningful.
                if causal:
                    coefficients = causal_blocks[seg_j]
                else:
                    fhr_pass = st_phase_list[seg_j]
                    reduced = {
                        "fhr_st": fhr_pass["scattering"][0],
                        "fhr_ph": fhr_pass["phase_corr"][0][phase_mask, :],
                        # fhr_up_ph now carries ONLY the cross-channel phase
                        # coefficients. UP self-phase harmonics live in up_ph as a
                        # first-class field with their own per-channel asinh stats.
                        "fhr_up_ph": st_cross_list[seg_j]["cross_phase_corr"][0][cross_mask, :],
                        "up_st": st_up_scatter_list[seg_j]["scattering"][0],
                        "up_ph": st_up_phase_list[seg_j]["phase_corr"][0][up_phase_mask, :],
                    }
                    coefficients = {
                        name: value.detach().cpu().numpy() for name, value in reduced.items()
                    }

                if guid_tracking is not None:
                    guid_tracking[guid_key].included_domain_starts.append(
                        float(domain_starts[orig_idx])
                    )

                tflo = float(domain_starts[orig_idx]) - labor_onset_sec
                tss = float(domain_starts[orig_idx]) - ss_sec

                b_fhr.append(fhr[orig_idx, :])
                b_up.append(up[orig_idx, :])
                b_fhr_st.append(coefficients["fhr_st"])
                b_up_st.append(coefficients["up_st"])
                b_fhr_ph.append(coefficients["fhr_ph"])
                b_up_ph.append(coefficients["up_ph"])
                if not causal:
                    b_fhr_up_ph.append(coefficients["fhr_up_ph"])
                b_target.append(pre_defined_target * sample_weights[orig_idx, :])
                b_weight.append(sample_weights[orig_idx, :])
                b_guid.append(record_name)
                b_epoch.append(domain_starts[orig_idx])
                b_cs.append(cs_label)
                b_bg.append(bg_label)
                b_tlo.append(tflo)
                b_ss.append(tss)

            if b_fhr:
                append_samples_batch(
                    path=hdf5_path,
                    fhr_batch=np.stack(b_fhr),
                    up_batch=np.stack(b_up),
                    fhr_st_batch=np.stack(b_fhr_st),
                    fhr_ph_batch=np.stack(b_fhr_ph),
                    # Explicitly None rather than an empty stack: the append guard raises when a
                    # file has the dataset and the batch is missing, so this must say "this build
                    # does not produce it" rather than "there happened to be nothing to write".
                    fhr_up_ph_batch=None if causal else np.stack(b_fhr_up_ph),
                    target_batch=np.stack(b_target),
                    weight_batch=np.stack(b_weight),
                    guid_batch=b_guid,
                    epoch_batch=np.array(b_epoch, dtype=np.float32),
                    cs_label_batch=np.array(b_cs, dtype=np.uint8),
                    bg_label_batch=np.array(b_bg, dtype=np.uint8),
                    tlo_batch=np.array(b_tlo, dtype=np.float32),
                    second_stage_batch=np.array(b_ss, dtype=np.float32),
                    up_st_batch=np.stack(b_up_st),
                    up_ph_batch=np.stack(b_up_ph),
                )

        except Exception as e:
            errors_list.append(record)
            logger.error(f"Failed processing {record}:\n{traceback.format_exc()}")
            if guid_tracking is not None:
                err_guid = os.path.splitext(os.path.basename(record))[0]
                guid_tracking[err_guid] = GuidTrackingEntry(
                    error=True, error_msg=str(e)
                )

    # GUID analysis
    if run_guid_analysis and hdf5_path and guid_tracking:
        try:
            from guid_analysis import run_guid_analysis as _run_analysis
            segment_dur = signal_length / 4
            _run_analysis(hdf5_path, guid_tracking, segment_duration_sec=segment_dur)
        except Exception as e:
            logger.error(f"GUID analysis failed: {e}")

    return errors_list


# ============================================================================
# Step 5: Main orchestrator
# ============================================================================
def _build_hdf5_for_partition(
    part_dir: str,
    subgroups: Dict[str, List[str]],
    masks: Dict[str, Any],
    labor_onset_map: Dict[str, float],
    second_stage_map: Dict[str, float],
    sequence_length: int,
    run_guid_analysis: bool,
    scatter_batch_size: int,
    verbose: bool,
    transform: str = TWO_SIDED,
    device: Optional[torch.device] = None,
    source_pickle_path: Optional[str] = None,
) -> None:
    """Build HDF5 files for one partition (train, val, or test).

    Args:
        part_dir: Output directory for this partition.
        subgroups: ``{subgroup_name: [record_paths]}``.
        masks: Scattering masks dict.
        labor_onset_map: GUID -> TLO seconds.
        second_stage_map: GUID -> second stage seconds.
        sequence_length: Sequence dimension length.
        run_guid_analysis: Whether to run GUID analysis.
        scatter_batch_size: Scattering batch size.
        verbose: Verbosity flag.
        transform: ``'two_sided'`` or ``'causal'``.
        device: Torch device for the transform; ``None`` lets the writer pick.
        source_pickle_path: The fold pickle this run resumed from, recorded in
            every shard it writes.
    """
    os.makedirs(part_dir, exist_ok=True)
    for sg, records in subgroups.items():
        if not records:
            continue
        target, cs, bg = SUBGROUP_META[sg]
        hdf5_file = os.path.join(part_dir, f"{sg}.hdf5")
        create_hdf5_for_masks(
            hdf5_file, masks, len_sequence=sequence_length,
            records_list=records, source_pickle_path=source_pickle_path,
        )
        create_hdf5_dataset_from_records_list(
            hdf5_path=hdf5_file,
            records_list=records,
            cs_label=cs,
            bg_label=bg,
            pre_defined_target=target,
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map,
            second_stage_map=second_stage_map,
            base_block_size=BASE_BLOCK_SIZE,
            overlap_percentage=OVERLAP_PERCENTAGE,
            device=device,
            run_guid_analysis=run_guid_analysis,
            scatter_batch_size=scatter_batch_size,
            verbose=verbose,
            transform=transform,
        )


def create_new_pipeline(
    records_base_path: str,
    output_base_path: str,
    tlo_csv_path: str,
    test_mode: str = "augmented",
    verbose: bool = True,
    scatter_batch_size: int = 16,
    num_workers: Optional[int] = None,
    screening_csv_path: Optional[str] = None,
    classification_pickle_path: Optional[str] = None,
    transform: str = TWO_SIDED,
    device: Optional[str] = None,
):
    """Run the complete new dataset creation pipeline.

    Steps:
        1. Prescreen all GUIDs for valid signal in last 6 hours.
        2. Select GUIDs for classification (balanced train/val,
           population-proportional test).
        3. Create 10-fold stratified CV splits (stratified by labour
           duration within each subgroup).
        4. Build classification HDF5 datasets.
        5. Build pretraining HDF5 datasets from BG subgroup leftovers.

    Args:
        records_base_path: Root dir with StudyGroup subfolders.
        output_base_path: Output directory for all generated files.
        tlo_csv_path: Path to complete CSV with TLO + second stage data.
        test_mode: ``"augmented"`` (test in each fold + augmentation,
            default) or ``"holdout"`` (fixed test set).
        verbose: If False, suppress all output except errors.
        scatter_batch_size: Scattering batch size.
        num_workers: Parallel prescreening workers.
        screening_csv_path: Skip Step 1, load this pre-computed CSV.
        classification_pickle_path: Skip Steps 1-3, load this pickle. Recorded
            in every shard this run writes, together with a digest of that
            shard's GUID set: a causal dataset is comparable to a two-sided one
            segment for segment only if both resumed from the same pickle, and
            afterwards those two attributes are the only evidence of it.
        transform: ``'two_sided'`` (the shipped kymatio bank) or ``'causal'``
            (the one-sided gammatone bank, narrower scattering blocks, no
            ``fhr_up_ph``). Written into every file as a root attribute.
        device: Torch device for the transform, e.g. ``'cuda:3'`` to pin one
            GPU of eight. ``None`` keeps today's behaviour: the first CUDA
            device if one exists, else the CPU.

    Raises:
        ValueError: On an unknown *transform*, before anything is created.
    """
    # First statement, before os.makedirs and before the CSV is read: a refusal that arrives
    # later leaves an output directory behind and reports a CSV problem instead of the real one.
    validate_transform(transform)

    setup_verbosity(verbose)
    os.makedirs(output_base_path, exist_ok=True)
    torch_device = None if device is None else torch.device(device)

    # Load CSV metadata (needed for HDF5 creation in all paths)
    labor_onset_map, second_stage_map = load_csv_metadata(tlo_csv_path, verbose)

    # Compute scattering masks once
    logger.info("Computing scattering masks (v3)...")
    masks = compute_scattering_masks(
        SIGNAL_LENGTH, scattering_T=16, device=torch_device, transform=transform
    )
    # Log the resolved layout and the active selection parameters: this is the
    # operator's confirmation that the intended variant is running, and it
    # surfaces c_y / c_u so a stale model config is caught before the run ends.
    # Every number is derived, so a geometry change moves the log rather than
    # leaving a stale literal in it.
    for line in format_layout(describe_layout(masks, torch_device)):
        logger.info(line)
    logger.info(
        f"Phase selection: k_steps={PHASE_HARMONIC_K_STEPS}, "
        f"fhr_band={FHR_PHASE_BAND_HZ} Hz, up_band={UP_PHASE_BAND_HZ} Hz"
    )

    sequence_length = SIGNAL_LENGTH // 16

    # Shared kwargs for _build_hdf5_for_partition
    hdf5_kw = dict(
        masks=masks,
        labor_onset_map=labor_onset_map,
        second_stage_map=second_stage_map,
        sequence_length=sequence_length,
        scatter_batch_size=scatter_batch_size,
        verbose=verbose,
        transform=transform,
        device=torch_device,
        source_pickle_path=classification_pickle_path,
    )

    # ------------------------------------------------------------------
    # Resolve starting point based on skip flags
    # ------------------------------------------------------------------
    if classification_pickle_path is not None:
        # Skip Steps 1-3: load pre-computed CV result
        logger.info(
            f"Loading pre-computed CV result from: "
            f"{classification_pickle_path}"
        )
        with open(classification_pickle_path, "rb") as f:
            cv_result = pickle.load(f)

        # Backward compatibility: old pickle is a flat dict of folds
        if "test_mode" not in cv_result:
            logger.warning(
                "Legacy pickle format detected — treating as augmented mode"
            )
            cv_result = {
                "test_mode": "augmented",
                "folds": cv_result,
                "test_augmentation": {},
            }

        logger.info(
            f"Loaded {len(cv_result['folds'])} folds "
            f"(mode={cv_result['test_mode']})"
        )

        # Need pretraining BG leftovers — derive from all BG files minus
        # those used in classification (union across all folds/partitions)
        all_bg_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_CS"
        )
        all_bg_no_cs_files = _discover_mat_files(
            records_base_path, "HEALTHY_NO_ACIDOSIS_NoCS"
        )
        cls_bg_cs: set = set()
        cls_bg_no_cs: set = set()
        for fold_data in cv_result["folds"].values():
            for part in fold_data.values():
                cls_bg_cs.update(part.get("healthy_bg_cs", []))
                cls_bg_no_cs.update(part.get("healthy_bg_no_cs", []))
        # Also include test holdout if present
        test_dict = cv_result.get("test", cv_result.get("test_augmentation", {}))
        cls_bg_cs.update(test_dict.get("healthy_bg_cs", []))
        cls_bg_no_cs.update(test_dict.get("healthy_bg_no_cs", []))

        pretrain_bg_cs = [f for f in all_bg_cs_files if f not in cls_bg_cs]
        pretrain_bg_no_cs = [
            f for f in all_bg_no_cs_files if f not in cls_bg_no_cs
        ]

    else:
        # --- Step 1: Prescreening ---
        if screening_csv_path is not None:
            logger.info(f"Loading screening CSV: {screening_csv_path}")
            screening_df = pd.read_csv(screening_csv_path)
        else:
            screening_csv_out = os.path.join(
                output_base_path, "guid_screening_results.csv"
            )
            screening_df = prescreen_all_guids(
                records_base_path, tlo_csv_path, screening_csv_out,
                verbose=verbose, num_workers=num_workers,
            )

        # --- Step 2: GUID selection ---
        selection_result = select_classification_guids(
            screening_df, test_mode=test_mode, verbose=verbose
        )

        # Save selection summary
        summary_path = os.path.join(
            output_base_path, "classification_guid_selection_summary.json"
        )
        summary: Dict[str, Any] = {"test_mode": test_mode, "stats": selection_result["stats"]}
        for pool_name in ["trainval", "test"]:
            summary[pool_name] = {
                sg: {"count": len(paths), "paths_sample": paths[:3]}
                for sg, paths in selection_result[pool_name].items()
            }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # --- Step 3: Fold creation ---
        cv_result = create_cv_splits(
            selection_result, n_splits=N_FOLDS,
            val_ratio=VAL_RATIO, random_state=RANDOM_STATE,
        )
        pickle_path = os.path.join(
            output_base_path, "classification_dataset_records.pickle"
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(cv_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"CV result saved to {pickle_path}")

        pretrain_bg_cs = selection_result["pretraining_bg_cs"]
        pretrain_bg_no_cs = selection_result["pretraining_bg_no_cs"]

    # ------------------------------------------------------------------
    # Step 4: Classification HDF5 creation
    # ------------------------------------------------------------------
    actual_mode = cv_result["test_mode"]
    kfold_path = os.path.join(output_base_path, "k_fold_cross_validation_dataset")
    os.makedirs(kfold_path, exist_ok=True)

    if actual_mode == "holdout":
        # --- Build shared test HDF5 once ---
        test_dir = os.path.join(kfold_path, "test")
        test_data = cv_result["test"]
        logger.info(
            f"Creating shared test HDF5 "
            f"({sum(len(v) for v in test_data.values())} GUIDs)..."
        )
        _build_hdf5_for_partition(
            part_dir=test_dir,
            subgroups=test_data,
            run_guid_analysis=False,
            **hdf5_kw,
        )

        # --- Build fold train/val ---
        run_ga = True
        run_eda = True
        for fold_name, fold_data in cv_result["folds"].items():
            logger.info(f"Processing {fold_name}...")
            fold_dir = os.path.join(kfold_path, fold_name)
            for partition_name in ["train", "val"]:
                _build_hdf5_for_partition(
                    part_dir=os.path.join(fold_dir, partition_name),
                    subgroups=fold_data[partition_name],
                    run_guid_analysis=(run_ga and partition_name == "train"),
                    **hdf5_kw,
                )
            if run_eda:
                try:
                    from fold_eda_analysis import run_fold_eda
                    eda_dir = os.path.join(fold_dir, "fold_eda")
                    run_fold_eda(fold_dir, eda_dir, test_dir=test_dir)
                    logger.info(f"Fold EDA saved to {eda_dir}")
                except Exception as e:
                    logger.error(f"Fold EDA failed: {e}")
                run_eda = False
            run_ga = False

    else:  # augmented
        run_ga = True
        run_eda = True
        for fold_name, fold_data in cv_result["folds"].items():
            logger.info(f"Processing {fold_name}...")
            fold_dir = os.path.join(kfold_path, fold_name)
            for partition_name in ["train", "val", "test"]:
                _build_hdf5_for_partition(
                    part_dir=os.path.join(fold_dir, partition_name),
                    subgroups=fold_data[partition_name],
                    run_guid_analysis=(run_ga and partition_name == "train"),
                    **hdf5_kw,
                )
            if run_eda:
                try:
                    from fold_eda_analysis import run_fold_eda
                    eda_dir = os.path.join(fold_dir, "fold_eda")
                    run_fold_eda(fold_dir, eda_dir)
                    logger.info(f"Fold EDA saved to {eda_dir}")
                except Exception as e:
                    logger.error(f"Fold EDA failed: {e}")
                run_eda = False
            run_ga = False

    logger.info("Classification datasets complete.")

    # ------------------------------------------------------------------
    # Step 5: Pretraining HDF5 creation
    # ------------------------------------------------------------------
    pretrain_path = os.path.join(output_base_path, "pre_training_dataset")
    os.makedirs(pretrain_path, exist_ok=True)

    logger.info(
        f"Pretraining leftovers: BG_CS={len(pretrain_bg_cs)}, "
        f"BG_NoCS={len(pretrain_bg_no_cs)}"
    )

    random.shuffle(pretrain_bg_cs)
    split_cs = int(len(pretrain_bg_cs) * 0.9)
    train_cs = pretrain_bg_cs[:split_cs]
    test_cs = pretrain_bg_cs[split_cs:]

    random.shuffle(pretrain_bg_no_cs)
    split_no_cs = int(len(pretrain_bg_no_cs) * 0.9)
    train_no_cs = pretrain_bg_no_cs[:split_no_cs]
    test_no_cs = pretrain_bg_no_cs[split_no_cs:]

    pretrain_sets = [
        ("train_dataset_cs.hdf5", train_cs, True, True),
        ("train_dataset_no_cs.hdf5", train_no_cs, False, True),
        ("test_dataset_cs.hdf5", test_cs, True, True),
        ("test_dataset_no_cs.hdf5", test_no_cs, False, True),
    ]

    # These four bypass _build_hdf5_for_partition entirely, which is exactly why they go through
    # the same create_hdf5_for_masks: a variant threaded into the partition path alone would
    # produce a directory whose classification and pre-training files disagree.
    for fname, records, cs, bg in pretrain_sets:
        hdf5_file = os.path.join(pretrain_path, fname)
        logger.info(f"Creating {fname} ({len(records)} GUIDs)...")
        create_hdf5_for_masks(
            hdf5_file, masks, len_sequence=sequence_length,
            records_list=records, source_pickle_path=classification_pickle_path,
        )
        create_hdf5_dataset_from_records_list(
            hdf5_path=hdf5_file,
            records_list=records,
            cs_label=cs,
            bg_label=bg,
            pre_defined_target=1,  # all healthy
            precomputed_masks=masks,
            labor_onset_map=labor_onset_map,
            second_stage_map=second_stage_map,
            base_block_size=BASE_BLOCK_SIZE,
            overlap_percentage=OVERLAP_PERCENTAGE,
            device=torch_device,
            run_guid_analysis=False,
            scatter_batch_size=scatter_batch_size,
            verbose=verbose,
            transform=transform,
        )

    logger.info("Pretraining datasets complete.")
    logger.info("Pipeline finished.")


def _discover_mat_files(records_base_path: str, folder_name: str) -> List[str]:
    """List all .mat files in a StudyGroup subfolder.

    Args:
        records_base_path: Root dir with StudyGroup subfolders.
        folder_name: e.g. ``"HEALTHY_NO_ACIDOSIS_CS"``.

    Returns:
        Sorted list of full .mat file paths.
    """
    efm_dir = os.path.join(records_base_path, folder_name, "EFMOut")
    if not os.path.isdir(efm_dir):
        return []
    return sorted(
        os.path.join(efm_dir, f) for f in os.listdir(efm_dir) if f.endswith(".mat")
    )


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    # ---- Configure paths here ----
    records_base_path = r"/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/"
    output_base_path = r"/data1/fetal-heart-tracing/HDF5_Datasets/new_pipeline_6h"
    tlo_csv_path = r"/path/to/complete_labor_onset.csv"

    # ---- Options ----
    test_mode = "augmented"  # "augmented" (default) or "holdout"
    verbose = False
    scatter_batch_size = 128
    num_workers = None  # defaults to min(cpu_count, 8)

    # "two_sided" = the shipped kymatio bank. "causal" = the one-sided gammatone bank: scattering
    # blocks narrow to 36 channels, fhr_up_ph is not produced, and every block carries its
    # per-channel warm-up. Write it to a SEPARATE output_base_path — nothing here modifies an
    # existing dataset, and a causal build is only comparable to the two-sided one if it resumes
    # from that run's classification_pickle_path, which pins the same GUIDs, folds and segments.
    transform = "two_sided"
    # Torch device for the transform, e.g. "cuda:3" to pin one GPU of eight on the production box.
    # None keeps today's behaviour: the first CUDA device if there is one, else the CPU.
    device = None

    # ---- Resume / skip flags (set to None for full pipeline) ----
    screening_csv_path = None  # e.g. r"/path/to/guid_screening_results.csv"
    classification_pickle_path = None  # e.g. r"/path/to/classification_dataset_records.pickle"

    create_new_pipeline(
        records_base_path=records_base_path,
        output_base_path=output_base_path,
        tlo_csv_path=tlo_csv_path,
        test_mode=test_mode,
        verbose=verbose,
        scatter_batch_size=scatter_batch_size,
        num_workers=num_workers,
        screening_csv_path=screening_csv_path,
        classification_pickle_path=classification_pickle_path,
        transform=transform,
        device=device,
    )
