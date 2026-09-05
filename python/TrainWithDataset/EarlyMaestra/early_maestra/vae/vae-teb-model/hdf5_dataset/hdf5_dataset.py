import os
import h5py
import numpy as np
from typing import Union, Sequence, List, Tuple, Dict, Any, Optional, Iterable
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pickle
import atexit
import threading
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
import gc
from torch.utils.data.dataloader import default_collate
import yaml
import random
from torch.utils.data import DataLoader, Sampler
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Stored field groups
# ---------------------------------------------------------------------------
# One definition of what a coefficient block is. Every membership test below —
# which fields get trimmed on the channel axis, which get normalised, which get
# transposed into the model's (time, channel) layout, and which carry a default
# per-channel transform — reads it from here. Written out separately at each
# site, they drifted: a block added to one list and not another is normalised
# but not transposed, or transposed but never trimmed, and neither shows up as
# an error.
#
# Nothing here says a file must contain all of them. ``__getitem__`` skips any
# field the file does not have, which is what makes an absent ``fhr_up_ph`` on
# a causal file need no special case anywhere.
_SCATTERING_FIELDS = ('fhr_st', 'up_st')
_PHASE_FIELDS = ('fhr_ph', 'fhr_up_ph', 'up_ph')
_COEFFICIENT_FIELDS = frozenset(_SCATTERING_FIELDS + _PHASE_FIELDS)
_RAW_FIELDS = ('fhr', 'up')
#: Fields normalisation applies to: the coefficient blocks plus the raw signals.
_NORMALISED_FIELDS = _COEFFICIENT_FIELDS | frozenset(_RAW_FIELDS)


# ---------------------------------------------------------------------------
# Transform variants
# ---------------------------------------------------------------------------
# A file written by the current pipeline records which wavelet bank produced
# it. A file **without** that attribute is a legacy two-sided file: absence is
# the normal, expected state of every dataset already on disk and is never a
# defect, so it is resolved silently and warns about nothing.
TWO_SIDED = 'two_sided'
CAUSAL = 'causal'


def resolve_transform(attrs: Any) -> str:
    """Which wavelet bank produced a file, with the legacy default applied.

    Args:
        attrs: The file's root HDF5 attributes.

    Returns:
        ``'two_sided'`` or ``'causal'``. A missing ``transform`` attribute
        resolves to ``'two_sided'`` — every dataset built before the attribute
        existed is one, and reading one is the common case, not a degraded one.
    """
    value = attrs.get('transform', TWO_SIDED)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


#: Which phase-harmonic operator built a causal file's two phase blocks. ``'none'`` multiplied the
#: two legs of every pair at one stored index; ``'envelope'`` put them on one clock first. The
#: alignment changes no width, no warm-up and no stored delay, so unlike every other property of a
#: causal file it is **only** knowable from this attribute — which is why absence has to mean
#: something definite rather than "unknown".
LEG_ALIGNMENT_NONE = 'none'


def resolve_leg_alignment(attrs: Any) -> str:
    """Which phase-harmonic leg alignment produced a causal file, with the legacy default applied.

    Args:
        attrs: The file's root HDF5 attributes.

    Returns:
        The stored ``causal_leg_alignment``, or :data:`LEG_ALIGNMENT_NONE` when
        the attribute is absent — the state of every causal shard written
        before it existed, all of which multiplied the legs unaligned. Absence
        is therefore an answer rather than a gap, exactly as a missing
        ``transform`` is.
    """
    value = attrs.get('causal_leg_alignment', LEG_ALIGNMENT_NONE)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


#: Which phase-harmonic OPERATOR VERSION built a causal file's phase blocks. ``'ratio_power_v0'``
#: raised the accelerated leg to the floating-point ratio $\xi_j/\xi_i$ (fractional $2^{3/2}$
#: family included, principal-angle branch discontinuity included); ``'integer_harmonic_v1'``
#: applies a stored integer harmonic and admits only $k \in \{2, 4\}$. Restated rather than
#: imported from ``hdf5_dataset.causal_scattering``, which builds a kymatio filter bank at import;
#: ``tests/test_phase_operator.py`` asserts the two strings agree. Absence means the legacy
#: operator: every causal shard written before the version existed was built with it.
PHASE_OPERATOR_LEGACY = 'ratio_power_v0'


def resolve_phase_operator(attrs: Any) -> str:
    """Which phase-harmonic operator version produced a causal file, with the legacy default.

    Args:
        attrs: The file's root HDF5 attributes.

    Returns:
        The stored ``causal_phase_operator``, or :data:`PHASE_OPERATOR_LEGACY` when the
        attribute is absent -- the state of every causal shard written before the integer
        operator existed. Absence is an answer rather than a gap, exactly as it is for the leg
        alignment.
    """
    value = attrs.get('causal_phase_operator', PHASE_OPERATOR_LEGACY)
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


@dataclass(frozen=True)
class _FileLayout:
    """What one shard says about itself, read from attributes and shapes alone.

    Costs one file open per path and reads no coefficient data. It exists
    because a list of files that disagree with each other is accepted silently
    today and fails much later, in two different ways: ``default_collate``
    raises something opaque on the first mixed batch, while
    ``SignalSequenceDataset`` iterates the first segment's keys and instead
    **drops** whichever field the other file did not have.

    Attributes:
        path: The file this describes.
        transform: Resolved variant; see :func:`resolve_transform`.
        widths: Channels per stored coefficient block.
        lengths: Stored (untrimmed) steps per coefficient block.
        warmup_steps: Per-channel warm-up in untrimmed decimated steps, per
            block, or ``None`` for a two-sided file. Untrimmed because that is
            the storage geometry every other stored field uses; the loader
            rebases it for its own trim.
        delay_s: Per-channel composed group delay in **seconds**, per block, or
            ``None`` for a two-sided file. Carries only the blocks that
            actually have the attribute — like ``quantile`` and unlike
            ``warmup_steps``, it is read tolerantly here and enforced by
            :func:`read_causal_warmup`, because nothing in the loading path
            compensates for a delay and refusing it here would take a shard
            offline for a field no sample read uses.
        novelty_frac: Per-channel LEGACY novelty scalar (``causal_novelty_frac``,
            written at one fixed horizon by older builders), per block, or
            ``None`` for a two-sided file. Carries only the blocks that have the
            attribute, for the same reason ``delay_s`` does.
        novelty_curve: Per-channel horizon-free novelty curve
            (``causal_novelty_curve``, ``(C, W + 1)``), per block, or ``None``
            for a two-sided file. Carries only the blocks that have it; a current
            build writes it on every block and no legacy scalar.
        leg_alignment: Which phase-harmonic operator built the two phase
            blocks, or ``None`` for a two-sided file. **Absent is read as**
            :data:`LEG_ALIGNMENT_NONE`, so every shard written before the
            attribute existed stays loadable and reads as what it is. Read here
            rather than where it is compared because an aligned file has
            exactly the widths, warm-ups and delays of an unaligned one: this
            is the only thing that tells them apart, so a file list and a stats
            pairing must both be able to see it.
        phase_operator: Which phase-harmonic operator version built the two
            phase blocks, or ``None`` for a two-sided file. **Absent is read
            as** :data:`PHASE_OPERATOR_LEGACY`. The integer version also
            changes the phase widths, but a width is not a version: two
            selections could coincide in count and differ in exponent, so the
            string is what is compared.
        quantile: The ``causal_warmup_quantile`` the warm-up was measured at, or
            ``None`` for a two-sided file. Read here rather than where it is
            compared because it describes the same boundary
            ``warmup_steps`` does, and two files built at different quantiles
            carry different *channel counts* as well as different vectors — so a
            consumer that resolves a channel budget must be able to see it
            without opening every file a second time.
    """

    path: str
    transform: str
    widths: Dict[str, int]
    lengths: Dict[str, int]
    warmup_steps: Optional[Dict[str, np.ndarray]]
    delay_s: Optional[Dict[str, np.ndarray]]
    novelty_frac: Optional[Dict[str, np.ndarray]]
    leg_alignment: Optional[str]
    quantile: Optional[float]
    phase_operator: Optional[str] = None
    novelty_curve: Optional[Dict[str, np.ndarray]] = None


def _read_file_layout(path: str) -> _FileLayout:
    """Read one shard's self-description.

    Args:
        path: HDF5 file to describe.

    Returns:
        The file's :class:`_FileLayout`.

    Raises:
        ValueError: If the file claims to be causal but its warm-up attributes
            are missing or disagree with the channel axis they describe. Those
            attributes are all-or-nothing: a partially-attributed causal file
            would silently give some blocks a valid region and others none.
    """
    with h5py.File(path, 'r', libver='latest') as f:
        transform = resolve_transform(f.attrs)
        widths: Dict[str, int] = {}
        lengths: Dict[str, int] = {}
        for name in sorted(_COEFFICIENT_FIELDS):
            if name in f:
                widths[name] = int(f[name].shape[1])
                lengths[name] = int(f[name].shape[2])

        quantile = f.attrs.get('causal_warmup_quantile') if transform == CAUSAL else None
        warmup: Optional[Dict[str, np.ndarray]] = None
        delay: Optional[Dict[str, np.ndarray]] = None
        novelty: Optional[Dict[str, np.ndarray]] = None
        curve: Optional[Dict[str, np.ndarray]] = None
        leg_alignment: Optional[str] = None
        phase_operator: Optional[str] = None
        if transform == CAUSAL:
            leg_alignment = resolve_leg_alignment(f.attrs)
            phase_operator = resolve_phase_operator(f.attrs)
            warmup = {}
            delay = {}
            novelty = {}
            curve = {}
            for name, width in widths.items():
                if 'causal_delay_s' in f[name].attrs:
                    delay[name] = np.asarray(
                        f[name].attrs['causal_delay_s'], dtype=np.float64
                    )
                if 'causal_novelty_frac' in f[name].attrs:
                    novelty[name] = np.asarray(
                        f[name].attrs['causal_novelty_frac'], dtype=np.float64
                    )
                if 'causal_novelty_curve' in f[name].attrs:
                    table = np.asarray(f[name].attrs['causal_novelty_curve'], dtype=np.float64)
                    if table.ndim != 2 or table.shape[0] != width:
                        raise ValueError(
                            f"{path}: '{name}' stores {width} channels but its "
                            f"causal_novelty_curve has shape {table.shape}. The curve is read "
                            f"back as a parallel array to the warm-up, so a row-count "
                            f"disagreement would attribute one channel's curve to another."
                        )
                    curve[name] = table
                if 'causal_warmup_steps' not in f[name].attrs:
                    raise ValueError(
                        f"{path} declares transform='causal' but its '{name}' block carries no "
                        f"causal_warmup_steps attribute. The valid region of that block is then "
                        f"unknowable, and reading it as fully valid would feed a model the "
                        f"assumed pre-recording history as if it were signal."
                    )
                vector = np.asarray(f[name].attrs['causal_warmup_steps'], dtype=np.int64)
                if vector.shape != (width,):
                    raise ValueError(
                        f"{path}: '{name}' stores {width} channels but its causal_warmup_steps "
                        f"has {vector.size} entries. The attribute describes a different channel "
                        f"axis than the data, so no channel's validity can be trusted."
                    )
                warmup[name] = vector
    return _FileLayout(
        path=path,
        transform=transform,
        widths=widths,
        lengths=lengths,
        warmup_steps=warmup,
        delay_s=delay,
        novelty_frac=novelty,
        leg_alignment=leg_alignment,
        quantile=None if quantile is None else float(quantile),
        phase_operator=phase_operator,
        novelty_curve=curve,
    )


def _check_layouts_agree(layout: _FileLayout, reference: _FileLayout) -> None:
    """Refuse two shards that cannot be read as one dataset.

    Extracted so the loader's tolerant scan and the strict causal reader below
    apply the *same* comparison: the two differ in how they treat a missing or
    unreadable file, not in what "these files disagree" means, and a second copy
    of the comparison could only drift from this one.

    Args:
        layout: The shard being checked.
        reference: The shard every other one must agree with.

    Raises:
        ValueError: On a disagreement about variant, leg alignment, block set
            or a block width, naming both files and the disagreement.
    """
    if layout.transform != reference.transform:
        raise ValueError(
            f"Mixed transform variants in one dataset: {layout.path} is "
            f"'{layout.transform}' but {reference.path} is '{reference.transform}'. "
            f"Their channel axes mean different things and their statistics are not "
            f"interchangeable; build one dataset per variant."
        )
    if layout.leg_alignment != reference.leg_alignment:
        raise ValueError(
            f"Mixed causal leg alignments in one dataset: {layout.path} is "
            f"'{layout.leg_alignment}' but {reference.path} is '{reference.leg_alignment}'. "
            f"An aligned shard has exactly the widths, warm-ups and delays of an unaligned one, "
            f"so nothing else here can tell them apart -- and the two hold different phase "
            f"coefficients under the same channel names."
        )
    if layout.phase_operator != reference.phase_operator:
        raise ValueError(
            f"Mixed causal phase operators in one dataset: {layout.path} is "
            f"'{layout.phase_operator}' but {reference.path} is '{reference.phase_operator}'. "
            f"The two apply different exponents to the accelerated leg and select different "
            f"pair families, so their phase channels are different representations under the "
            f"same block names; build one dataset per operator version."
        )
    if set(layout.widths) != set(reference.widths):
        missing = sorted(set(reference.widths) - set(layout.widths))
        extra = sorted(set(layout.widths) - set(reference.widths))
        raise ValueError(
            f"Mismatched coefficient blocks: {layout.path} is missing {missing} and adds "
            f"{extra} relative to {reference.path}. A sample would carry different keys "
            f"depending on which file it came from, which collates into an opaque failure "
            f"or silently drops the field."
        )
    for name, width in layout.widths.items():
        if width != reference.widths[name]:
            raise ValueError(
                f"Mismatched '{name}' width: {layout.path} stores {width} channels but "
                f"{reference.path} stores {reference.widths[name]}."
            )


def _resolve_dataset_layout(paths: Sequence[str]) -> Optional[_FileLayout]:
    """Refuse an incoherent file list, and return what the coherent ones agree on.

    Variants are compared **resolved**, so a list mixing a legacy shard (no
    ``transform`` attribute) with a newly-written two-sided shard is accepted:
    they are the same variant, and comparing the raw attribute would refuse a
    combination that works perfectly well today.

    A file that cannot be opened is warned about and skipped, exactly as the
    index build already does — a coherence check is not the place to start
    failing on a broken file that is tolerated everywhere else.

    Args:
        paths: The dataset's files, in the order given.

    Returns:
        The first readable file's layout, which every other readable file
        agrees with, or ``None`` if none could be read.

    Raises:
        ValueError: On the first file that disagrees about variant, block set
            or a block width, naming the file and the disagreement.
    """
    reference: Optional[_FileLayout] = None
    for path in paths:
        if not os.path.exists(path):
            continue  # _build_index warns about this; one message is enough.
        try:
            layout = _read_file_layout(path)
        except ValueError:
            raise
        except Exception as error:
            warnings.warn(f"Could not read the layout of {path}: {error}")
            continue

        if reference is None:
            reference = layout
            continue

        _check_layouts_agree(layout, reference)
    return reference


def rebase_causal_warmup(
    layout: _FileLayout, trim_steps: int, trim_minutes: Optional[float] = None
) -> Optional[Dict[str, np.ndarray]]:
    r"""Move the stored warm-up into the coordinates of a trimmed window.

    Rebasing is $W' = \max(W - \text{trim},\ 0)$: the stored vector counts from
    stored step $0$, while a consumer that trims ``trim_steps`` from each end
    starts reading at stored step ``trim_steps``, so a channel whose warm-up
    ended at stored step $20$ has $20 - 15 = 5$ invalid steps left in a window
    trimmed by $15$.

    Shared by every consumer that reads the valid region — the sample loader and
    the statistics calculator — because they must agree on it exactly: a
    statistic accumulated over a wider region than the loader serves would
    normalise the data with constants drawn partly from the pre-recording
    history.

    Args:
        layout: The dataset's resolved layout.
        trim_steps: Decimated steps discarded from **each** end.
        trim_minutes: Only used to make the refusal below name the setting the
            caller actually configured.

    Returns:
        ``{block: (C,) int64}`` of rebased warm-ups, or ``None`` for a two-sided
        or legacy layout, which has no warm-up and is not a degraded case.

    Raises:
        ValueError: If any channel has no valid step left. Such a column
            normalises to zeros indistinguishable from real coefficients, so it
            must not be served or accumulated over.
    """
    if layout.warmup_steps is None:
        return None

    rebased: Dict[str, np.ndarray] = {}
    for name in sorted(layout.warmup_steps):
        stored = layout.warmup_steps[name]
        kept_steps = layout.lengths[name] - 2 * trim_steps
        vector = np.maximum(stored - trim_steps, 0).astype(np.int64)

        dead = np.flatnonzero(kept_steps - vector <= 0)
        if dead.size:
            raise ValueError(
                f"'{name}' channel {int(dead[0])} has no valid step: its warm-up is "
                f"{int(stored[dead[0]])} stored steps, which rebases to {int(vector[dead[0]])} "
                f"against a {kept_steps}-step window at trim_minutes={trim_minutes}. "
                f"An all-invalid channel normalises to zeros indistinguishable from real "
                f"coefficients, so it must not be served."
            )
        rebased[name] = vector
    return rebased


#: Raw sampling rate of the stored signals in Hz, and raw samples per decimated
#: step. The two constants the trim arithmetic is built from, named once because
#: every consumer that rebases anything against a trimmed window needs both.
RAW_SAMPLING_HZ = 4
DECIMATION = 16


def decimated_trim_steps(trim_minutes: Optional[float]) -> Tuple[int, int]:
    """Raw samples and decimated steps one ``trim_minutes`` setting discards per end.

    One function rather than the arithmetic written out at each site: a consumer
    that rebases a warm-up against a trimmed window must use the *loader's* trim
    exactly, and a copy of the conversion that rounded differently would move the
    valid region without moving anything that reports it.

    Args:
        trim_minutes: The configured symmetric trim in minutes, or ``None`` for
            no trim at all.

    Returns:
        ``(raw_samples, decimated_steps)`` discarded from **each** end.
    """
    if trim_minutes is None:
        return 0, 0
    raw = int(RAW_SAMPLING_HZ * 60 * trim_minutes)
    return raw, raw // DECIMATION


@dataclass(frozen=True)
class CausalWarmup:
    r"""A causal dataset's valid-region boundary, rebased for one consumer's trim.

    What :meth:`CombinedHDF5Dataset.causal_warmup_steps` reports, available
    *without* building a loader — so a model can resolve a channel budget from the
    shards it is about to train on before any sample is read, and can be refused
    if those shards do not agree with each other.

    Attributes:
        paths: The shards this was read from, in the order given.
        trim_minutes: The trim the vectors below are expressed against.
        trim_steps: Decimated steps that trim discards from **each** end.
        quantile: The ``causal_warmup_quantile`` every shard agrees on. Part of
            the record because the quantile sets both the warm-up vectors and the
            stored channel count, so two datasets built at different quantiles
            describe different channel axes.
        warmup_steps: ``{block: (C,) int64}``, rebased: the first step of the
            trimmed window at which each channel is a function of the recording
            rather than of assumed pre-recording history.
        delay_s: ``{block: (C,) float64}``, the composed one-sided group delay
            each channel is stale by, in **seconds** and **unrebased**. A delay
            is not a step index: it says how far back in physical time a
            coefficient's content sits, which no trim of the window moves. Kept
            in seconds for the same reason — it is a constant of the filter
            bank, and expressing it in steps would bake one decimation into a
            number that does not depend on one.
        novelty_frac: ``{block: (C,) float64}``, the LEGACY scalar novelty proxy
            ``causal_novelty_frac`` -- the share of each channel's composed
            envelope mass within the writer's one fixed horizon on the stored
            clock. Empty for a shard written before that attribute existed or
            after it was superseded by the curve; a consumer that needs a novelty
            vector must say so rather than assume a default, because there is no
            value a missing one could safely stand in as.
        novelty_curve: ``{block: (C, W + 1) float64}``, the horizon-free
            ``causal_novelty_curve`` of every current build: the share of each
            channel's composed envelope mass within $w$ stored steps, for
            $w = 0, \ldots, W$. A consumer looks its own horizon and per-channel
            label advance up in it. Empty on a legacy shard.
        leg_alignment: Which phase-harmonic leg alignment built the phase
            blocks of every shard in the list, all of which agree on it.
        kept_steps: ``{block: T}``, the steps the trimmed window leaves.
        phase_operator: Which phase-harmonic operator version built the phase
            blocks of every shard in the list, all of which agree on it;
            :data:`PHASE_OPERATOR_LEGACY` for every shard predating the version.
    """

    paths: Tuple[str, ...]
    trim_minutes: Optional[float]
    trim_steps: int
    quantile: Optional[float]
    warmup_steps: Dict[str, np.ndarray]
    delay_s: Dict[str, np.ndarray]
    novelty_frac: Dict[str, np.ndarray]
    leg_alignment: str
    kept_steps: Dict[str, int]
    phase_operator: str = PHASE_OPERATOR_LEGACY
    novelty_curve: Dict[str, np.ndarray] = field(default_factory=dict)


def read_causal_warmup(
    paths: Sequence[str], trim_minutes: Optional[float] = None
) -> CausalWarmup:
    r"""Read the causal warm-up off a file list and rebase it for ``trim_minutes``.

    The public entry point for a consumer that needs the boundary but not the
    data: it opens each file's attributes, refuses a list that does not describe
    one dataset, and rebases through :func:`rebase_causal_warmup` — the same
    function the loader and the statistics calculator share, so a third consumer
    cannot arrive at a third valid region.

    **Every** file is read, not the first. A two-sided test shard beside causal
    training shards, or a shard rebuilt at another ``causal_warmup_quantile``,
    would otherwise resolve cleanly against the first file and be evaluated
    against a boundary its own coefficients do not have.

    Stricter than the loader's own scan in two ways. A missing file raises here
    rather than being skipped: the loader tolerates a short file list because it
    still has samples to serve, while a boundary resolved from a subset of the
    configured shards is simply the wrong boundary. And ``causal_delay_s`` is
    required here and merely read where present in the layout, because a
    consumer that aligns channels resolves its shifts from that vector alone and
    a shard without it is not a shard it can be configured against.

    Args:
        paths: The dataset's files. All of them — training and held-out.
        trim_minutes: The trim the consumer will read these files at. Must be
            the loader's own setting, or the returned vectors describe a window
            nothing serves.

    Returns:
        The rebased boundary, with the geometry it was rebased against and the
        per-channel group delay beside it.

    Raises:
        ValueError: If ``paths`` is empty, if a file is missing, if any file is
            not causal, if a causal file carries no ``causal_delay_s`` or one of
            the wrong width, if the files disagree about variant, block set,
            block width, quantile, warm-up or delay, or if the rebase leaves a
            channel with no valid step.
    """
    if not paths:
        raise ValueError(
            "read_causal_warmup was given no files. The warm-up boundary is a property of the "
            "shards, so there is nothing to read it from."
        )

    layouts: List[_FileLayout] = []
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(
                f"{path} does not exist. A warm-up boundary resolved from the shards that "
                f"happen to be present is the wrong boundary for the dataset that was asked "
                f"for, and nothing downstream would say so."
            )
        layouts.append(_read_file_layout(path))

    for layout in layouts:
        if layout.transform != CAUSAL:
            raise ValueError(
                f"{layout.path} resolves to transform='{layout.transform}', not '{CAUSAL}'. Its "
                f"coefficients are two-sided: the value at step t is a weighted average over raw "
                f"samples on both sides of t, so it has no warm-up and no valid region to resolve."
            )
        # The delay is required *here* rather than in the layout: nothing on the loading path
        # compensates for it, but a consumer that aligns channels onto one clock resolves its
        # per-channel shifts from this vector and nothing else, so an absent one would be
        # discovered as a missing key deep inside that resolution rather than named here.
        assert layout.delay_s is not None  # every layout is causal, checked immediately above
        for name, width in sorted(layout.widths.items()):
            if name not in layout.delay_s:
                raise ValueError(
                    f"{layout.path} declares transform='{CAUSAL}' but its '{name}' block carries "
                    f"no causal_delay_s attribute. A one-sided channel is stale by its composed "
                    f"group delay, and a consumer that aligns channels in time has no other "
                    f"source for that number."
                )
            if layout.delay_s[name].shape != (width,):
                raise ValueError(
                    f"{layout.path}: '{name}' stores {width} channels but its causal_delay_s has "
                    f"{layout.delay_s[name].size} entries. The attribute describes a different "
                    f"channel axis than the data, so no channel's staleness can be trusted."
                )

    reference = layouts[0]
    for layout in layouts[1:]:
        _check_layouts_agree(layout, reference)
        if layout.quantile != reference.quantile:
            raise ValueError(
                f"Mismatched causal_warmup_quantile: {layout.path} was built at "
                f"{layout.quantile} but {reference.path} at {reference.quantile}. The quantile "
                f"sets both the per-channel warm-up and which channels survive the build, so the "
                f"two files describe different channel axes."
            )
        # Stored length, which `_check_layouts_agree` does not compare -- it is not a property a
        # loader serving one sample at a time needs the files to share. This reader does: it
        # rebases against `reference` alone and reports one `kept_steps` per block, so a shorter
        # shard beside a longer one would have the entire budget, the dead-channel refusal and the
        # consumer's own trim cross-check computed against a window it does not have.
        for name, length in sorted(layout.lengths.items()):
            if length != reference.lengths[name]:
                raise ValueError(
                    f"Mismatched '{name}' stored length: {layout.path} stores {length} steps but "
                    f"{reference.path} stores {reference.lengths[name]}. The warm-up is rebased "
                    f"against one window and reported once, so the shorter shard would be served "
                    f"channels whose warm-up outruns it with every readout still reporting a full "
                    f"valid region."
                )
        assert layout.warmup_steps is not None and reference.warmup_steps is not None
        for name, vector in sorted(layout.warmup_steps.items()):
            if not np.array_equal(vector, reference.warmup_steps[name]):
                raise ValueError(
                    f"Mismatched '{name}' causal_warmup_steps: {layout.path} disagrees with "
                    f"{reference.path}. The warm-up is a constant of the filter bank, so two "
                    f"shards that disagree were not built by the same bank and a budget resolved "
                    f"from one of them scores pad on the other."
                )
        assert layout.delay_s is not None and reference.delay_s is not None
        for name, vector in sorted(layout.delay_s.items()):
            if not np.array_equal(vector, reference.delay_s[name]):
                raise ValueError(
                    f"Mismatched '{name}' causal_delay_s: {layout.path} disagrees with "
                    f"{reference.path}. The delay is a constant of the filter bank, exactly as "
                    f"the warm-up is, so two shards that disagree were not built by the same "
                    f"bank and a channel alignment resolved from one of them is the wrong shift "
                    f"on the other."
                )
        # The curve is a constant of the bank too, and a novelty vector resolved from one shard's
        # curve must describe every shard in the list; a shard that carries none where the
        # reference does is a legacy build mixed into a current one.
        layout_curve = layout.novelty_curve or {}
        reference_curve = reference.novelty_curve or {}
        if set(layout_curve) != set(reference_curve) or any(
            not np.array_equal(layout_curve[name], reference_curve[name])
            for name in reference_curve
        ):
            raise ValueError(
                f"Mismatched causal_novelty_curve: {layout.path} disagrees with "
                f"{reference.path} (blocks carrying it: {sorted(layout_curve)} against "
                f"{sorted(reference_curve)}). The curve is a constant of the filter bank, so two "
                f"shards that disagree were not built by the same bank, or one predates the "
                f"attribute; a novelty split resolved from one of them mislabels the other."
            )

    _, trim_steps = decimated_trim_steps(trim_minutes)
    rebased = rebase_causal_warmup(reference, trim_steps, trim_minutes)
    assert rebased is not None  # every layout is causal, checked above
    assert reference.delay_s is not None and reference.novelty_frac is not None
    assert reference.leg_alignment is not None
    return CausalWarmup(
        paths=tuple(str(path) for path in paths),
        trim_minutes=trim_minutes,
        trim_steps=trim_steps,
        quantile=reference.quantile,
        warmup_steps=rebased,
        # Verbatim, not rebased: the trim moves where a window starts, not how far back in
        # physical time a coefficient's content sits.
        delay_s={name: reference.delay_s[name] for name in sorted(rebased)},
        # Only the blocks that carry it. A shard predating the attribute yields an empty mapping,
        # which a consumer must handle by name rather than by a stand-in value.
        novelty_frac={
            name: reference.novelty_frac[name]
            for name in sorted(rebased)
            if name in reference.novelty_frac
        },
        leg_alignment=reference.leg_alignment,
        kept_steps={
            name: reference.lengths[name] - 2 * trim_steps for name in sorted(rebased)
        },
        phase_operator=(
            PHASE_OPERATOR_LEGACY if reference.phase_operator is None else reference.phase_operator
        ),
        # Verbatim, like the delay: the trim moves where a window starts, not how a coefficient's
        # envelope mass accumulates behind it. Empty on a legacy shard.
        novelty_curve={
            name: reference.novelty_curve[name]
            for name in sorted(rebased)
            if reference.novelty_curve is not None and name in reference.novelty_curve
        },
    )


def normalize_tensor_data(
    data: torch.Tensor,
    field_name: str,
    normalization_stats: Dict[str, Dict[str, Any]],
    log_norm_channels_config: Dict[str, Any],
    asinh_norm_channels_config: Dict[str, Any],
    log_epsilon: float,
    pin_memory: bool = False,
    normalize_fields: Optional[set] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Normalizes a tensor using precomputed statistics from a stats dictionary.

    This function can be used independently of the CombinedHDF5Dataset class.
    It prepares the statistics (mean/std tensors) and then applies normalization.

    Args:
        data: The data tensor to normalize.
        field_name: The name of the field (e.g., 'fhr_st').
        normalization_stats: A dictionary of statistics, like one from DatasetStatsCalculator.
        log_norm_channels_config: A dictionary defining which channels use log normalization.
        asinh_norm_channels_config: A dictionary defining which channels use asinh normalization.
        log_epsilon: A small value to add before taking the logarithm.
        pin_memory: If True, pin the created mean/std tensors.
        normalize_fields: An optional set of fields to normalize. If None, all are attempted.
        dtype: The torch dtype for the tensors.

    Returns:
        The normalized data tensor.
    """
    if field_name not in normalization_stats:
        return data

    if normalize_fields is not None and field_name not in normalize_fields:
        return data

    stats = normalization_stats[field_name]

    # Prepare mean and std tensors from numpy arrays in stats
    if 'mean_tensor' not in stats or 'std_tensor' not in stats:
        if field_name in ['fhr', 'up']:
            mean_tensor = torch.tensor(stats['mean'], dtype=dtype)
            std_tensor = torch.tensor(np.sqrt(stats['variance']), dtype=dtype)
        else:
            mean_array = np.array(stats['mean'], dtype=np.float32)
            std_array = np.array(np.sqrt(stats['variance']), dtype=np.float32)
            mean_tensor = torch.from_numpy(mean_array).to(dtype).unsqueeze(-1)
            std_tensor = torch.from_numpy(std_array).to(dtype).unsqueeze(-1)
    else:
        mean_tensor = stats['mean_tensor']
        std_tensor = stats['std_tensor']

    if pin_memory and data.is_pinned():
        mean_tensor = mean_tensor.pin_memory()
        std_tensor = std_tensor.pin_memory()

    is_batch = data.dim() == 3

    # Apply transformation based on field type
    if field_name in ['fhr', 'up']:
        epsilon = 1e-8
        normalized_data = (data - mean_tensor) / (std_tensor + epsilon)
    else:
        # Multi-channel scattering data: apply log or standard normalization per channel.
        n_channels = data.shape[1] if is_batch else data.shape[0]

        # Determine which channels use log normalization from the config.
        log_config = log_norm_channels_config.get(field_name, [])
        log_channels = []
        if log_config == 'all_except_0':
            log_channels = [c for c in range(n_channels) if c != 0] if n_channels > 0 else []
        elif isinstance(log_config, list):
            log_channels = log_config

        # Determine which channels use asinh normalization from the config.
        asinh_config = asinh_norm_channels_config.get(field_name, [])
        asinh_channels = []
        if asinh_config == 'all':
            asinh_channels = [c for c in range(n_channels)]
        elif isinstance(asinh_config, list):
            asinh_channels = asinh_config

        # Start with a clone of the data; we will transform it in-place.
        data_transformed = data.clone()

        # Apply log transform to the specified channels.
        if log_channels:
            log_channels_tensor = torch.tensor(log_channels, device=data.device, dtype=torch.long)
            if is_batch:
                selected_data = data_transformed[:, log_channels_tensor, :]
                log_transformed_data = torch.log(torch.clamp(selected_data, min=0.0) + log_epsilon)
                data_transformed[:, log_channels_tensor, :] = log_transformed_data
            else:
                selected_data = data_transformed[log_channels_tensor, :]
                log_transformed_data = torch.log(torch.clamp(selected_data, min=0.0) + log_epsilon)
                data_transformed[log_channels_tensor, :] = log_transformed_data

        # Apply asinh transform to the specified channels.
        if asinh_channels:
            asinh_channels_tensor = torch.tensor(asinh_channels, device=data.device, dtype=torch.long)
            if is_batch:
                selected_data = data_transformed[:, asinh_channels_tensor, :]
                asinh_transformed_data = torch.asinh(selected_data)
                data_transformed[:, asinh_channels_tensor, :] = asinh_transformed_data
            else:
                selected_data = data_transformed[asinh_channels_tensor, :]
                asinh_transformed_data = torch.asinh(selected_data)
                data_transformed[asinh_channels_tensor, :] = asinh_transformed_data
        
        # Reshape stats tensors for broadcasting across all dimensions.
        if is_batch:
            mean_tensor = mean_tensor.view(1, -1, 1)
            std_tensor = std_tensor.view(1, -1, 1)
        
        # Apply standard normalization to the (potentially log-transformed) data.
        epsilon = 1e-8
        normalized_data = (data_transformed - mean_tensor) / (std_tensor + epsilon)

    return normalized_data


def create_initial_hdf5(
    path: str,
    len_signal: int,
    n_channels: int,
    len_sequence: int = 300,
    n_cross_phase_channels: int = 62,
    n_up_st_channels: int = 0,
) -> None:
    """
    Create a new HDF5 file with empty, resizable datasets for signal storage.

    DEPRECATED — this is the pre-split legacy schema, used only by
    ``create_hdf5_dataset.py`` and the synthetic fixtures in
    ``guid_hdf5_dataset.py``. It concatenates the UP self-phase into
    ``fhr_up_ph`` and emits no ``up_ph`` dataset, and it hardcodes the
    ``fhr_ph`` width at 44. The current writer is
    ``new_pipeline/create_new_pipeline.py``, which sizes ``fhr_ph`` and
    ``up_ph`` from their selections and records per-channel provenance in
    ``sel_*`` attrs.

    Legacy layout (J=11, Q=4, T=16):
        - FHR scattering: 43 coefficients (first order only)
        - FHR phase: 44 coefficients (legacy selector)
        - FHR-UP cross-phase + UP self-phase: dynamic count, concatenated
        - UP scattering (optional): 43 coefficients (first order only)

    Datasets created (first dim unlimited):
        - "fhr"       : float32, shape (N, len_signal)
        - "up"        : float32, shape (N, len_signal)
        - "fhr_st"    : float32, shape (N, 43, len_sequence) - FHR scattering coefficients
        - "fhr_ph"    : float32, shape (N, 44, len_sequence) - Selected phase coefficients
        - "fhr_up_ph" : float32, shape (N, n_cross_phase_channels, len_sequence) - Cross-phase + UP self-phase
        - "up_st"     : float32, shape (N, n_up_st_channels, len_sequence) - UP scattering coefficients (optional)
        - "target"    : float32, shape (N, len_sequence)
        - "weight"    : float32, shape (N, len_sequence)
        - "epoch"     : float32, shape (N,)
        - "cs_label"  : uint8 (0 or 1), shape (N,)
        - "bg_label"  : uint8 (0 or 1), shape (N,)
        - "time_from_labor_onset" : float32, shape (N,) - seconds since labor onset (NaN if unavailable)
        - "guid"      : variable-length UTF-8 strings, shape (N,)

    All datasets use chunked storage and LZF compression.

    Args:
        path:                    Path to output HDF5 file (overwrites if exists).
        len_signal:              Length of raw signal arrays (e.g. 5760).
        n_channels:              Total number of phase + cross-phase channels (fhr_ph + fhr_up_ph).
        len_sequence:            Length of sequence dimension (default: 300).
        n_cross_phase_channels:  Number of channels for fhr_up_ph (cross-phase + UP self-phase).
        n_up_st_channels:        Number of UP scattering channels (0 = do not create up_st dataset).
    """
    try:
        os.remove(path)
    except OSError:
        pass

    # Chunk size along sample axis — balances batch write speed and random read
    # overhead.  32 samples per chunk means batch writes (typical ~50 segments
    # per .mat file) only create 1-2 new chunks instead of 50.
    chunk_n = 32

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w", libver="latest") as h5f:
        h5f.create_dataset(
            "fhr", shape=(0, len_signal), maxshape=(None, len_signal),
            dtype="f4", chunks=(chunk_n, len_signal), compression="lzf"
        )
        h5f.create_dataset(
            "up", shape=(0, len_signal), maxshape=(None, len_signal),
            dtype="f4", chunks=(chunk_n, len_signal), compression="lzf"
        )
        # Create datasets with optimal channel counts
        # fhr_st: 43 scattering coefficients (first order)
        h5f.create_dataset(
            "fhr_st", shape=(0, 43, len_sequence), maxshape=(None, 43, len_sequence),
            dtype="f4", chunks=(chunk_n, 43, len_sequence), compression="lzf"
        )
        # fhr_ph: 44 selected phase coefficients
        h5f.create_dataset(
            "fhr_ph", shape=(0, 44, len_sequence), maxshape=(None, 44, len_sequence),
            dtype="f4", chunks=(chunk_n, 44, len_sequence), compression="lzf"
        )
        # fhr_up_ph: selected cross-phase + UP self-phase coefficients (v3)
        h5f.create_dataset(
            "fhr_up_ph", shape=(0, n_cross_phase_channels, len_sequence),
            maxshape=(None, n_cross_phase_channels, len_sequence),
            dtype="f4", chunks=(chunk_n, n_cross_phase_channels, len_sequence), compression="lzf"
        )
        # up_st: UP scattering coefficients (optional, same structure as fhr_st)
        if n_up_st_channels > 0:
            h5f.create_dataset(
                "up_st", shape=(0, n_up_st_channels, len_sequence),
                maxshape=(None, n_up_st_channels, len_sequence),
                dtype="f4", chunks=(chunk_n, n_up_st_channels, len_sequence),
                compression="lzf"
            )
        h5f.create_dataset(
            "target", shape=(0, len_sequence), maxshape=(None, len_sequence),
            dtype="f4", chunks=(chunk_n, len_sequence), compression="lzf"
        )
        h5f.create_dataset(
            "weight", shape=(0, len_sequence), maxshape=(None, len_sequence),
            dtype="f4", chunks=(chunk_n, len_sequence), compression="lzf"
        )
        h5f.create_dataset(
            "epoch", shape=(0,), maxshape=(None,),
            dtype="f4", chunks=(chunk_n,), compression="lzf"
        )
        h5f.create_dataset(
            "cs_label", shape=(0,), maxshape=(None,),
            dtype="u1", chunks=(chunk_n,), compression="lzf"
        )
        h5f.create_dataset(
            "bg_label", shape=(0,), maxshape=(None,),
            dtype="u1", chunks=(chunk_n,), compression="lzf"
        )
        h5f.create_dataset(
            "time_from_labor_onset", shape=(0,), maxshape=(None,),
            dtype="f4", chunks=(chunk_n,), compression="lzf"
        )
        h5f.create_dataset(
            "guid", shape=(0,), maxshape=(None,),
            dtype=str_dt, chunks=(chunk_n,)
        )


def append_sample(
    path: str,
    fhr: np.ndarray,
    up: np.ndarray,
    fhr_st: np.ndarray,
    fhr_ph: np.ndarray,
    fhr_up_ph: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    guid: str,
    epoch: float,
    cs_label: bool,
    bg_label: bool,
    time_from_labor_onset: float = float('nan'),
    up_st: Optional[np.ndarray] = None,
) -> None:
    """
    Append a single sample to an existing HDF5 dataset.

    Datasets are resized by +1 along axis=0, and new values written in-place.

    Args:
        path:      Path to existing HDF5 file.
        fhr:       Raw FHR array, shape (len_signal,).
        up:        Raw UP array, shape (len_signal,).
        fhr_st:    Scattering array, shape (n_channels, len_sequence).
        fhr_ph:    Phase array, shape (n_channels, len_sequence).
        fhr_up_ph: Cross-phase array, shape (n_channels, len_sequence).
        target:    Target array, shape (len_sequence,).
        weight:    Weight array, shape (len_sequence,).
        guid:      Unique identifier string.
        epoch:     Epoch as float.
        cs_label:  Case label flag.
        bg_label:  Background label flag.
        time_from_labor_onset: Seconds since labor onset (NaN if unavailable).
        up_st:     UP scattering array, shape (n_channels, len_sequence). Optional.
    """
    with h5py.File(path, "a", libver="latest") as h5f:
        idx = h5f["fhr"].shape[0]
        new_size = idx + 1
        for name, ds in h5f.items():
            ds.resize((new_size,) + ds.shape[1:])
        h5f["fhr"][idx]       = fhr
        h5f["up"][idx]        = up
        h5f["fhr_st"][idx]    = fhr_st
        h5f["fhr_ph"][idx]    = fhr_ph
        h5f["fhr_up_ph"][idx] = fhr_up_ph
        if up_st is not None and "up_st" in h5f:
            h5f["up_st"][idx] = up_st
        h5f["target"][idx]    = target
        h5f["weight"][idx]    = weight
        h5f["epoch"][idx]     = epoch
        h5f["cs_label"][idx]  = np.uint8(cs_label)
        h5f["bg_label"][idx]  = np.uint8(bg_label)
        if "time_from_labor_onset" in h5f:
            h5f["time_from_labor_onset"][idx] = time_from_labor_onset
        h5f["guid"][idx]      = guid


def append_samples_batch(
    path: str,
    fhr_batch: np.ndarray,
    up_batch: np.ndarray,
    fhr_st_batch: np.ndarray,
    fhr_ph_batch: np.ndarray,
    fhr_up_ph_batch: np.ndarray,
    target_batch: np.ndarray,
    weight_batch: np.ndarray,
    guid_batch: list,
    epoch_batch: np.ndarray,
    cs_label_batch: np.ndarray,
    bg_label_batch: np.ndarray,
    tlo_batch: np.ndarray,
    up_st_batch: Optional[np.ndarray] = None,
) -> None:
    """Append K samples to an existing HDF5 dataset in a single file open/close.

    This is a batched version of ``append_sample`` that avoids the overhead of
    opening the file, resizing every dataset by +1, and closing for each
    individual sample.  All datasets are resized once by +K and written via
    slice assignment.

    Args:
        path:            Path to existing HDF5 file.
        fhr_batch:       Raw FHR array, shape (K, len_signal).
        up_batch:        Raw UP array, shape (K, len_signal).
        fhr_st_batch:    Scattering array, shape (K, n_ch, len_seq).
        fhr_ph_batch:    Phase array, shape (K, n_ch, len_seq).
        fhr_up_ph_batch: Cross-phase array, shape (K, n_ch, len_seq).
        target_batch:    Target array, shape (K, len_seq).
        weight_batch:    Weight array, shape (K, len_seq).
        guid_batch:      List of GUID strings, length K.
        epoch_batch:     Epoch array, shape (K,).
        cs_label_batch:  CS label array, shape (K,), dtype uint8.
        bg_label_batch:  BG label array, shape (K,), dtype uint8.
        tlo_batch:       Time-from-labor-onset array, shape (K,).
        up_st_batch:     UP scattering array, shape (K, n_ch, len_seq). Optional.
    """
    k = fhr_batch.shape[0]
    if k == 0:
        return
    with h5py.File(path, "a", libver="latest") as h5f:
        idx = h5f["fhr"].shape[0]
        new_size = idx + k
        for _name, ds in h5f.items():
            ds.resize((new_size,) + ds.shape[1:])
        h5f["fhr"][idx:new_size]       = fhr_batch
        h5f["up"][idx:new_size]        = up_batch
        h5f["fhr_st"][idx:new_size]    = fhr_st_batch
        h5f["fhr_ph"][idx:new_size]    = fhr_ph_batch
        h5f["fhr_up_ph"][idx:new_size] = fhr_up_ph_batch
        if up_st_batch is not None and "up_st" in h5f:
            h5f["up_st"][idx:new_size] = up_st_batch
        h5f["target"][idx:new_size]    = target_batch
        h5f["weight"][idx:new_size]    = weight_batch
        h5f["epoch"][idx:new_size]     = epoch_batch
        h5f["cs_label"][idx:new_size]  = cs_label_batch.astype(np.uint8)
        h5f["bg_label"][idx:new_size]  = bg_label_batch.astype(np.uint8)
        if "time_from_labor_onset" in h5f:
            h5f["time_from_labor_onset"][idx:new_size] = tlo_batch
        for i, g in enumerate(guid_batch):
            h5f["guid"][idx + i] = g


class AttributeDict(dict):
    """A dictionary that allows attribute-style access."""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


class CombinedHDF5Dataset(Dataset):
    """
    High-performance PyTorch Dataset for one or more HDF5 files with identical structure.

    Nothing in this class depends on a particular channel count: widths come
    from the data and the stats file, and the per-channel transform is chosen
    by field name. Two layouts are in use, and a file says which it is through
    its root ``transform`` attribute (absent = legacy two-sided):

    ==================  ==========  ========
    Block               two-sided   causal
    ==================  ==========  ========
    ``fhr_st``          43          36
    ``fhr_ph``          66          66
    ``fhr_up_ph``       79          absent
    ``up_st``           43          36
    ``up_ph``           15          15
    ==================  ==========  ========

    All multi-channel scattering/phase fields are first-class HDF5 datasets
    with their own per-channel statistics. ``up_ph`` is no longer virtually
    sliced from ``fhr_up_ph`` — each field flows through the same
    normalisation code path. The two self-phase fields also carry ``sel_*``
    attrs describing which wavelet pair each channel came from; see
    ``PHASE_HARMONIC_CHANNEL_SELECTION.md``.

    **The causal valid region.** A one-sided filter reads only the past, so
    before its warm-up has passed a channel's output is a function of the
    assumed pre-recording history rather than of the recording. That boundary
    is a property of the *filter bank*: it is identical for every segment in
    every file, which is why it arrives as a per-block attribute rather than a
    stored per-sample mask — a $(C, T)$ boolean array per sample would
    replicate one constant tens of thousands of times, about 76 KB per sample
    against about 600 bytes per file.

    The stored vector is in **untrimmed** step coordinates, matching every
    other stored field, so this class rebases it for its own ``trim_minutes``:
    a channel with stored warm-up 20 read at ``trim_minutes=1.0`` reports 5,
    because the loader's own slice already discarded the first 15 steps.
    :attr:`causal_warmup_steps` is the rebased vector and
    :meth:`channel_valid_mask` is the ``(T, C)`` boolean form of it.

    **The group delay is not compensated.** Beyond its warm-up a causal
    channel is still *stale* by its composed group delay — hundreds of seconds
    on the slow channels — and nothing here shifts it back. A consumer that
    aligns channels in time reads that delay from
    :attr:`CausalWarmup.delay_s`, which :func:`read_causal_warmup` populates
    from the same ``causal_delay_s`` attribute beside the warm-up it already
    reports, so the two arrive together and agree across every configured
    shard.

    Optimized for:
    - Multi-GPU training with DistributedDataParallel
    - Multi-worker data loading
    - Memory efficiency and fast I/O
    - Advanced filtering and selective loading
    - Data normalization using precomputed statistics with optimal coefficient selection

    Args:
        paths: Path(s) to HDF5 file(s).
        load_fields: Specific fields to load (None loads all). Target and weight are always included.
        allowed_guids: Only samples with these GUIDs are included.
        cs_label: Filter by cs_label value (True/False/None for no filtering).
        bg_label: Filter by bg_label value (True/False/None for no filtering).
        epoch_min: Minimum epoch value (inclusive).
        epoch_max: Maximum epoch value (inclusive).
        label: Target class to filter by. Only samples where the one-hot encoded target
               has this class as 1 in at least one valid timestep are included.
        cache_size: Number of samples to keep in memory cache (0 disables caching).
        pin_memory: Pre-allocate tensors in pinned memory for faster GPU transfer.
        dtype: Data type for tensors (torch.float32 or torch.float16 for mixed precision).
        stats_path: Path to HDF5 statistics file for data normalization (None disables normalization).
        normalize_fields: List of fields to normalize (None normalizes all available fields with stats).
        trim_minutes: Optional trimming time in minutes for signal data
        emit_validity_mask: Add a ``<block>_valid`` boolean field per
            coefficient block to every sample. Default off: the mask is a
            dataset constant, so paying for it per sample, per worker, per
            collate and per host-to-device copy is only worth it for a model
            that consumes it batched. A model that does not can read
            :meth:`channel_valid_mask` once instead. Causal datasets only.
    """
    def __init__(
        self,
        paths: Union[str, Sequence[str]],
        load_fields: Optional[Sequence[str]] = None,
        allowed_guids: Optional[Sequence[str]] = None,
        cs_label: Optional[bool] = None,
        bg_label: Optional[bool] = None,
        epoch_min: Optional[float] = None,
        epoch_max: Optional[float] = None,
        label: Optional[int] = None,
        cache_size: int = 2000,
        pin_memory: bool = True,
        dtype: torch.dtype = torch.float32,
        stats_path: Optional[str] = None,
        normalize_fields: Optional[Sequence[str]] = None,
        trim_minutes: Optional[float] = None,
        emit_validity_mask: bool = False,
    ):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.load_fields = None if load_fields is None else set(load_fields)
        self.allowed_guids = set(allowed_guids) if allowed_guids is not None else None
        self.cs_label = cs_label
        self.bg_label = bg_label
        self.epoch_min = epoch_min
        self.epoch_max = epoch_max
        self.label = label
        self.cache_size = cache_size
        self.pin_memory = pin_memory
        self.dtype = dtype
        self.stats_path = stats_path
        self.normalize_fields = set(normalize_fields) if normalize_fields is not None else None
        self.trim_minutes = trim_minutes
        self.emit_validity_mask = emit_validity_mask
        self.trim_samples_raw, self.trim_samples_decimated = decimated_trim_steps(
            self.trim_minutes
        )

        # Thread-safe file handle management
        self.file_handles: List[Any] = [None] * len(self.paths)
        self._handle_locks = [threading.Lock() for _ in self.paths]
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, sample_idx)
        
        # Performance optimizations
        self._cache: Dict[int, "AttributeDict"] = {}
        self._cache_lock = threading.Lock()
        self._access_count = 0
        
        # Normalization statistics
        self.normalization_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self.normalization_enabled = False
        
        # Define which channels should use LOG normalization for optimal coefficients.
        # Derived from the field groups rather than listing block names a third
        # time: a scattering block keeps its order-0 channel linear and log-
        # transforms the rest, a phase block is signed and takes asinh
        # throughout. These configs are overwritten at load time by whatever
        # the stats file says (see `_load_normalization_stats`); what is kept
        # here is a sensible fallback used only when no stats file is provided.
        self.log_norm_channels_config = {
            name: 'all_except_0' for name in _SCATTERING_FIELDS
        }
        self.asinh_norm_channels_config = {name: 'all' for name in _PHASE_FIELDS}

        # This will be populated from the stats file, but the config above provides a fallback.
        self.order0_channels: Dict[str, List[int]] = {}
        self.log_epsilon = 1e-6  # For log transformation

        # Register cleanup for proper file handle management
        atexit.register(self._cleanup_handles)

        # What these files are, resolved before anything reads them: an
        # incoherent list is refused here rather than at the first mixed batch,
        # and the causal warm-up is needed before a single sample is served.
        self._layout = _resolve_dataset_layout(self.paths)
        self.transform = TWO_SIDED if self._layout is None else self._layout.transform
        self._rebased_warmup: Optional[Dict[str, np.ndarray]] = None
        self._valid_mask_cache: Dict[str, torch.Tensor] = {}
        self._resolve_validity()

        if self.emit_validity_mask and self._rebased_warmup is None:
            raise ValueError(
                f"emit_validity_mask=True needs a causal dataset, but these files resolve to "
                f"'{self.transform}'. A two-sided block has no warm-up, so the only mask this "
                f"could emit is all-True — which would assert that every step of every channel "
                f"is honest about the past, the very thing the causal variant exists to fix."
            )

        # Load normalization statistics if provided
        if self.stats_path is not None:
            self._load_normalization_stats()

        # Build index with optimized filtering
        self._build_index()

        # Validate dataset
        if not self.index_map:
            raise ValueError("No samples match the specified filters.")

        print(f"Initialized HDF5Dataset: {len(self.index_map)} samples from {len(self.paths)} files")
        if self.cache_size > 0:
            print(f"Caching enabled: {min(self.cache_size, len(self.index_map))} samples")
        if self.normalization_enabled:
            normalized_fields = list(self.normalization_stats.keys())
            print(f"Normalization enabled for fields: {normalized_fields}")

    def __getstate__(self):
        """Exclude unpicklable threading locks and open file handles for multiprocessing."""
        state = self.__dict__.copy()
        # Remove threading locks (cannot be pickled)
        state['_handle_locks'] = None
        state['_cache_lock'] = None
        # Remove open HDF5 file handles (cannot be shared across processes)
        state['file_handles'] = [None] * len(self.paths)
        # Clear cache (each worker builds its own)
        state['_cache'] = {}
        # The validity masks are KEPT, unlike the sample cache. They are a
        # filter-bank constant — a few tens of KB for the whole dataset, the
        # same in every worker — so shipping them costs less than having every
        # worker rebuild them, and they cannot go stale because nothing mutates
        # them. Stated explicitly because the default for a cache here is to
        # drop it.
        return state

    def __setstate__(self, state):
        """Recreate threading locks and file handles after unpickling in worker process."""
        self.__dict__.update(state)
        # Recreate locks
        self._handle_locks = [threading.Lock() for _ in self.paths]
        self._cache_lock = threading.Lock()
        # File handles remain None — will be lazily opened on first access

    # ------------------------------------------------------------------
    # The causal valid region
    # ------------------------------------------------------------------
    def _resolve_validity(self) -> None:
        r"""Rebase the stored warm-up for this trim, report it, and refuse a dead channel.

        Rebasing is $W' = \max(W - \text{trim},\ 0)$, because the loader's own
        slice has already discarded the first ``trim`` steps: a channel whose
        warm-up ended at stored step 20 has $20 - 15 = 5$ invalid steps left in
        a window that starts at stored step 15.

        The per-channel counts are **printed**, not merely available. At the
        production geometry the slowest surviving channel rebases to 278 and
        keeps 22 valid steps of 300 — $7\%$ of the window, with that channel's
        statistics resting on it. A number that thin should be visible before a
        training run, not inferred from its results afterwards.

        Raises:
            ValueError: If any channel has no valid step at all. Returning it
                would hand a model a column that normalises to zeros
                indistinguishable from real coefficients.
        """
        if self._layout is None:
            return

        trim = self.trim_samples_decimated
        rebased = rebase_causal_warmup(self._layout, trim, self.trim_minutes)
        if rebased is None:
            return

        report: List[str] = []
        for name, vector in rebased.items():
            kept_steps = self._layout.lengths[name] - 2 * trim
            valid = kept_steps - vector
            report.append(
                f"  {name}: {int(valid.min())}..{int(valid.max())} valid of {kept_steps} steps; "
                f"per channel {valid.tolist()}"
            )

        self._rebased_warmup = rebased
        print(f"Causal validity (warm-up rebased by {trim} trimmed steps):")
        for line in report:
            print(line)

    @property
    def causal_warmup_steps(self) -> Optional[Dict[str, np.ndarray]]:
        r"""Per-channel warm-up in **this dataset's** step coordinates, or ``None``.

        The stored attribute is untrimmed, matching the storage geometry of
        every other field; what is returned here is rebased for
        ``trim_minutes``, because that is the window a sample actually spans. A
        stored warm-up of 20 at ``trim_minutes=1.0`` reports 5.

        Returns:
            ``{block: (C,) int64}`` for a causal dataset, ``None`` for a
            two-sided or legacy one — which has no warm-up to report and is not
            a degraded case.
        """
        if self._rebased_warmup is None:
            return None
        return {name: vector.copy() for name, vector in self._rebased_warmup.items()}

    def channel_valid_mask(self, field: str) -> torch.Tensor:
        r"""The valid region of one block as a $(T, C)$ boolean tensor.

        $(T, C)$ rather than $(C, T)$: it matches the transposed layout
        ``__getitem__`` hands out, so a model can apply it to the data without
        transposing either. Built once per block and cached — it is a filter-bank
        constant, identical for every sample in every file of this dataset.

        Args:
            field: A coefficient block name, e.g. ``'fhr_st'``.

        Returns:
            ``(T, C)`` bool, ``True`` where the channel has left its warm-up.

        Raises:
            ValueError: On a two-sided dataset, which has no warm-up, or on a
                block this dataset does not store.
        """
        if self._rebased_warmup is None:
            raise ValueError(
                f"channel_valid_mask('{field}') needs a causal dataset; these files resolve to "
                f"'{self.transform}', whose blocks carry no warm-up."
            )
        if field not in self._rebased_warmup:
            raise ValueError(
                f"'{field}' is not a coefficient block of this dataset; it stores "
                f"{sorted(self._rebased_warmup)}"
            )
        cached = self._valid_mask_cache.get(field)
        if cached is not None:
            return cached

        assert self._layout is not None  # guaranteed by _rebased_warmup being set
        kept_steps = self._layout.lengths[field] - 2 * self.trim_samples_decimated
        warmup = torch.from_numpy(self._rebased_warmup[field])
        steps = torch.arange(kept_steps, dtype=warmup.dtype).unsqueeze(1)
        mask = steps >= warmup.unsqueeze(0)
        self._valid_mask_cache[field] = mask
        return mask

    def _check_stats_pairing(self) -> None:
        """Refuse a stats file that describes a different dataset than this one.

        Deliberately called **outside** the broad ``except Exception`` in
        :meth:`_load_normalization_stats`: that handler turns any failure into a
        warning and disables normalisation, so a genuinely mispaired stats file
        caught in there would degrade a run to *unnormalised training data*
        rather than stopping it. Both checks below are about which dataset the
        constants were computed over, which is not a loading problem to recover
        from.

        Variants are compared **resolved**, so a legacy stats file (written
        before the attribute existed) paired with a legacy or two-sided dataset
        is a match and says nothing. The mismatch that matters is a causal
        dataset normalised with two-sided constants, computed over a different
        channel set and over the invalid region as well.

        The same argument extends to the leg alignment, and there it is the
        *only* check that can fire: an aligned shard and an unaligned one have
        identical widths, so a stats file built on one and paired with the
        other passes every other test here.

        Raises:
            ValueError: On a variant or leg-alignment mismatch, or on a block
                whose channel count disagrees with the stats file's.
        """
        try:
            with h5py.File(self.stats_path, 'r') as f:
                stats_transform = resolve_transform(f.attrs)
                stats_alignment = resolve_leg_alignment(f.attrs)
                stats_operator = resolve_phase_operator(f.attrs)
                stats_widths = {
                    field: int(f[field].attrs['n_channels'])
                    for field in f.keys()
                    if field in _COEFFICIENT_FIELDS and 'n_channels' in f[field].attrs
                }
        except OSError:
            # A stats file that cannot be opened at all is not a mispairing, and
            # how this class answers one is not this check's business: the load
            # below hits the same error and warns exactly as it always has.
            return

        if stats_transform != self.transform:
            raise ValueError(
                f"Statistics/dataset variant mismatch: {self.stats_path} was computed over a "
                f"'{stats_transform}' dataset but these files are '{self.transform}'. The two "
                f"variants have different channels and different valid regions, so normalising "
                f"one with the other's constants is silently wrong."
            )

        if self._layout is not None and self._layout.leg_alignment is not None:
            if stats_alignment != self._layout.leg_alignment:
                raise ValueError(
                    f"Statistics/dataset leg-alignment mismatch: {self.stats_path} was computed "
                    f"over a '{stats_alignment}' dataset but these files are "
                    f"'{self._layout.leg_alignment}'. The two hold different phase coefficients "
                    f"at identical widths, so the width check below cannot see this and the "
                    f"phase blocks would be normalised with another transform's mean and scale."
                )

        if self._layout is not None and self._layout.phase_operator is not None:
            if stats_operator != self._layout.phase_operator:
                raise ValueError(
                    f"Statistics/dataset phase-operator mismatch: {self.stats_path} was computed "
                    f"over a '{stats_operator}' dataset but these files are "
                    f"'{self._layout.phase_operator}'. The two operators apply different "
                    f"exponents and select different pair families, so the phase blocks would be "
                    f"normalised with constants accumulated over another representation."
                )

        if self._layout is not None:
            for field, width in stats_widths.items():
                stored = self._layout.widths.get(field)
                if stored is not None and stored != width:
                    raise ValueError(
                        f"Statistics/dataset width mismatch on '{field}': {self.stats_path} holds "
                        f"{width} channels but {self._layout.path} stores {stored}. The stats file "
                        f"is keyed to a different channel selection."
                    )

    def _load_normalization_stats(self):
        """
        Load normalization statistics from HDF5 file.

        The stats file should be created by DatasetStatsCalculator.save_stats().
        """
        if not os.path.exists(self.stats_path):
            warnings.warn(f"Statistics file not found: {self.stats_path}. Normalization disabled.")
            return

        # Before the recovering handler below, never inside it — see the method's docstring.
        self._check_stats_pairing()

        try:
            stats = {}
            with h5py.File(self.stats_path, 'r') as f:
                # Load global metadata
                self.log_epsilon = f.attrs.get('log_epsilon', 1e-6)
                stats_trim_minutes = f.attrs.get('trim_minutes', -1.0)
                if self.trim_minutes is not None and stats_trim_minutes != self.trim_minutes:
                    warnings.warn(f"Dataset trim_minutes ({self.trim_minutes}) does not match stats file trim_minutes ({stats_trim_minutes}). This may lead to incorrect normalization.")
                elif self.trim_minutes is None and stats_trim_minutes > 0:
                     warnings.warn(f"Stats file was created with trim_minutes={stats_trim_minutes}, but dataset is not using trimming. Normalization might be incorrect.")

                for field in f.keys():
                    if field == 'metadata':
                        continue
                    
                    field_group = f[field]
                    field_stats = {
                        'shape': tuple(field_group.attrs['shape']),
                        'count': field_group.attrs['count']
                    }
                    
                    if field in ['fhr', 'up']:
                        # Single-channel data - scalar values
                        field_stats['mean'] = field_group.attrs['mean_scalar']
                        field_stats['std'] = field_group.attrs['std_scalar']
                        
                        # Convert to tensors for efficient computation
                        field_stats['mean_tensor'] = torch.tensor(
                            field_stats['mean'], dtype=self.dtype
                        )
                        field_stats['std_tensor'] = torch.tensor(
                            field_stats['std'], dtype=self.dtype
                        )
                        
                    else:
                        # Multi-channel data - per-channel arrays
                        field_stats['mean'] = field_group['mean'][()]
                        field_stats['std'] = field_group['std'][()]
                        
                        # Load transformation metadata if available
                        if 'regular_channels' in field_group.attrs:
                            field_stats['uses_log_transform'] = field_group.attrs.get('uses_log_transform', False)
                            field_stats['uses_asinh_transform'] = field_group.attrs.get('uses_asinh_transform', False)
                            field_stats['regular_channels'] = list(field_group.attrs.get('regular_channels', []))
                            field_stats['log_channels'] = list(field_group.attrs.get('log_channels', []))
                            field_stats['asinh_channels'] = list(field_group.attrs.get('asinh_channels', []))
                        # Backward compatibility for 'order0_channels' from old stats files
                        elif 'order0_channels' in field_group.attrs:
                            order0_channels = list(field_group.attrs.get('order0_channels', []))
                            n_channels = len(field_stats['mean'])
                            log_channels = [i for i in range(n_channels) if i not in order0_channels]
                            field_stats['uses_log_transform'] = True
                            field_stats['uses_asinh_transform'] = False
                            field_stats['regular_channels'] = order0_channels
                            field_stats['log_channels'] = log_channels
                            field_stats['asinh_channels'] = []
                        else:
                            field_stats['uses_log_transform'] = False
                            field_stats['uses_asinh_transform'] = False
                            field_stats['regular_channels'] = []
                            field_stats['log_channels'] = []
                            field_stats['asinh_channels'] = []
                        
                        # Convert to tensors with proper shape for broadcasting
                        # Shape will be (n_channels, 1) for broadcasting over sequence dimension
                        mean_array = np.array(field_stats['mean'], dtype=np.float32)
                        std_array = np.array(field_stats['std'], dtype=np.float32)
                        
                        field_stats['mean_tensor'] = torch.from_numpy(mean_array).to(self.dtype).unsqueeze(-1)
                        field_stats['std_tensor'] = torch.from_numpy(std_array).to(self.dtype).unsqueeze(-1)
                    
                    stats[field] = field_stats
            
            self.normalization_stats = stats
            self.normalization_enabled = True
            
            # Overwrite the default transformation configs with what was loaded from the stats file.
            log_config = {}
            asinh_config = {}
            if self.normalization_stats:
                for field, stats_dict in self.normalization_stats.items():
                    if stats_dict.get('uses_log_transform') and 'log_channels' in stats_dict:
                        log_config[field] = stats_dict['log_channels']
                    if stats_dict.get('uses_asinh_transform') and 'asinh_channels' in stats_dict:
                        asinh_config[field] = stats_dict['asinh_channels']
            
            self.log_norm_channels_config = log_config
            self.asinh_norm_channels_config = asinh_config

            # Report the actual transformations that will be used.
            log_transformed_fields = list(self.log_norm_channels_config.keys())
            asinh_transformed_fields = list(self.asinh_norm_channels_config.keys())

            if log_transformed_fields:
                print(f"Log transformation enabled for fields: {log_transformed_fields}")
                for field in log_transformed_fields:
                    log_channels = self.log_norm_channels_config.get(field, [])
                    # Infer regular channels from the full channel list
                    try:
                        # shape is (channels, sequence_len) for these fields
                        n_channels = self.normalization_stats[field]['shape'][0]
                        regular_channels = [c for c in range(n_channels) if c not in log_channels]
                        if regular_channels or log_channels:
                            print(f"  {field}: regular channels {regular_channels}, log channels {log_channels}")
                    except (KeyError, IndexError):
                         print(f"  {field}: log channels {log_channels}")

            if asinh_transformed_fields:
                print(f"Asinh transformation enabled for fields: {asinh_transformed_fields}")
                for field in asinh_transformed_fields:
                    asinh_channels = self.asinh_norm_channels_config.get(field, [])
                    if asinh_channels:
                        print(f"  {field}: asinh channels {asinh_channels}")
            
        except Exception as e:
            warnings.warn(f"Failed to load statistics from {self.stats_path}: {e}. Normalization disabled.")
            self.normalization_stats = None
            self.normalization_enabled = False

    def _normalize_data(self, field_name: str, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data using precomputed statistics.
        Applies log transformation to scattering coefficients (except order 0).
        
        Args:
            field_name: Name of the field being normalized
            data: Data tensor to normalize
            
        Returns:
            Normalized data tensor
        """
        if not self.normalization_enabled or field_name not in self.normalization_stats:
            return data
        
        # Check if this field should be normalized
        if self.normalize_fields is not None and field_name not in self.normalize_fields:
            return data
        
        return normalize_tensor_data(
            data=data,
            field_name=field_name,
            normalization_stats=self.normalization_stats,
            log_norm_channels_config=self.log_norm_channels_config,
            asinh_norm_channels_config=self.asinh_norm_channels_config,
            log_epsilon=self.log_epsilon,
            pin_memory=self.pin_memory,
            normalize_fields=self.normalize_fields,
            dtype=self.dtype
        )

    def get_normalization_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get the loaded normalization statistics.
        
        Returns:
            Dictionary containing normalization statistics for each field, or None if not loaded
        """
        return self.normalization_stats

    def is_normalization_enabled(self) -> bool:
        """
        Check if normalization is enabled.
        
        Returns:
            True if normalization is enabled, False otherwise
        """
        return self.normalization_enabled

    def _build_index(self):
        """Build sample index with optimized filtering."""
        for fidx, path in enumerate(self.paths):
            if not os.path.exists(path):
                warnings.warn(f"HDF5 file not found: {path}")
                continue
                
            try:
                with h5py.File(path, 'r', libver='latest') as f:
                    # Load all metadata at once for efficiency
                    guids = f['guid'][()]
                    epochs = f['epoch'][()]
                    cs_lbl = f['cs_label'][()]
                    bg_lbl = f['bg_label'][()]
                    n_samples = len(guids)
                    
                    # Vectorized filtering where possible
                    valid_mask = np.ones(n_samples, dtype=bool)
                    
                    # Apply epoch filtering
                    if self.epoch_min is not None:
                        valid_mask &= (epochs >= self.epoch_min)
                    if self.epoch_max is not None:
                        valid_mask &= (epochs <= self.epoch_max)
                    
                    # Apply label filtering
                    if self.cs_label is not None:
                        valid_mask &= (cs_lbl == self.cs_label)
                    if self.bg_label is not None:
                        valid_mask &= (bg_lbl == self.bg_label)
                    
                    # Process remaining samples
                    for i in np.where(valid_mask)[0]:
                        # GUID filtering
                        guid = guids[i].decode('utf-8') if isinstance(guids[i], bytes) else str(guids[i])
                        if self.allowed_guids and guid not in self.allowed_guids:
                            continue
                        
                        # Target label filtering
                        if self.label is not None:
                            target_data = f['target'][i]  # shape: (len_sequence,)
                            # Check if any timestep has the target label value
                            has_label = np.any(target_data == self.label)
                            if not has_label:
                                continue
                        
                        self.index_map.append((fidx, i))
                        
            except Exception as e:
                warnings.warn(f"Error processing {path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.index_map)

    def _open_handle(self, file_idx: int):
        """Thread-safe file handle opening with optimizations."""
        with self._handle_locks[file_idx]:
            if self.file_handles[file_idx] is None:
                try:
                    # Optimal HDF5 settings for performance
                    self.file_handles[file_idx] = h5py.File(
                        self.paths[file_idx], 'r',
                        libver='latest',
                        swmr=True,
                        rdcc_nbytes=1024**2 * 128,    # 128MB cache per file
                        rdcc_nslots=10007,            # Prime number for hash table
                        rdcc_w0=0.75,                 # Cache write policy
                        driver='sec2'                 # System call driver for better performance
                    )
                except Exception as e:
                    # Fallback to default settings
                    warnings.warn(f"Using default HDF5 settings for {self.paths[file_idx]}: {e}")
                    self.file_handles[file_idx] = h5py.File(
                        self.paths[file_idx], 'r', libver='latest', swmr=True
                    )
            return self.file_handles[file_idx]
    
    def _cleanup_handles(self):
        """Thread-safe cleanup of file handles."""
        for i, (handle, lock) in enumerate(zip(self.file_handles, self._handle_locks)):
            with lock:
                if handle is not None:
                    try:
                        handle.close()
                        self.file_handles[i] = None
                    except:
                        pass

    @lru_cache(maxsize=128)
    def _get_sample_fields(self, file_idx: int) -> Tuple[str, ...]:
        """Cache available fields for each file."""
        f = self._open_handle(file_idx)
        return tuple(f.keys())

    def _create_tensor(self, data: np.ndarray, pin_memory: bool = None) -> torch.Tensor:
        """Optimized tensor creation with optional memory pinning."""
        if pin_memory is None:
            pin_memory = self.pin_memory
            
        # Convert to tensor with specified dtype
        if data.dtype == np.float32 and self.dtype == torch.float32:
            # Direct conversion without copy for matching dtypes
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.from_numpy(data.astype(np.float32 if self.dtype == torch.float32 else np.float16))
        
        # Pin memory for faster GPU transfer
        if pin_memory and tensor.is_floating_point():
            tensor = tensor.pin_memory()
            
        return tensor.to(dtype=self.dtype)

    def __getitem__(self, idx: int) -> "AttributeDict":
        """Optimized sample loading with caching, memory management, and normalization."""
        # Check cache first
        if self.cache_size > 0:
            with self._cache_lock:
                if idx in self._cache:
                    return self._cache[idx]
        
        file_idx, sample_idx = self.index_map[idx]
        f = self._open_handle(file_idx)
        out: Dict[str, Any] = {}

        # Determine fields to load
        available_fields = self._get_sample_fields(file_idx)
        if self.load_fields is None:
            fields = list(available_fields)
        else:
            fields = list(self.load_fields)

        # Load data efficiently. All scattering/phase fields are first-class
        # HDF5 datasets — ``up_ph`` is no longer virtually sliced from
        # ``fhr_up_ph``.
        try:
            for name in fields:
                if name not in available_fields:
                    continue

                data = f[name][sample_idx]

                if self.trim_minutes is not None:
                    if name in _RAW_FIELDS:
                        start_trim = self.trim_samples_raw
                        end_trim = -self.trim_samples_raw if self.trim_samples_raw > 0 else None
                        data = data[start_trim:end_trim]
                    elif name in _COEFFICIENT_FIELDS:
                        start_trim = self.trim_samples_decimated
                        end_trim = -self.trim_samples_decimated if self.trim_samples_decimated > 0 else None
                        data = data[:, start_trim:end_trim]
                    elif name in ['target', 'weight']:
                        start_trim = self.trim_samples_decimated
                        end_trim = -self.trim_samples_decimated if self.trim_samples_decimated > 0 else None
                        data = data[start_trim:end_trim]

                if name in ('guid',):
                    out[name] = data.decode('utf-8') if isinstance(data, bytes) else str(data)
                elif name in ('cs_label', 'bg_label'):
                    out[name] = bool(data)
                else:
                    # Optimized tensor creation
                    tensor = self._create_tensor(np.asarray(data))

                    # Apply normalization if enabled and applicable
                    if self.normalization_enabled and name in _NORMALISED_FIELDS:
                        tensor = self._normalize_data(name, tensor)

                    # SPEED OPTIMIZATION: Apply permutation here once instead of multiple times in training
                    # Convert from HDF5 format (channels, sequence) to model format (sequence, channels)
                    if name in _COEFFICIENT_FIELDS and tensor.dim() == 2:
                        tensor = tensor.transpose(0, 1)  # (channels, seq) -> (seq, channels)

                    out[name] = tensor

        except Exception as e:
            warnings.warn(f"Error loading sample {idx} from {self.paths[file_idx]}: {e}")
            raise

        if self.emit_validity_mask:
            # Deliberately outside the loop above: a mask must bypass
            # ``_create_tensor`` (which casts through float32 and would turn a
            # boolean mask into floats), normalisation (there is nothing to
            # standardise) and the transpose (it is already in (T, C) layout).
            # The tensor is the shared cached constant, not a per-sample copy —
            # collating and stacking both copy it, so a consumer sees its own
            # batched array; treat the per-sample one as read-only.
            for name in sorted(_COEFFICIENT_FIELDS):
                if name in out:
                    out[f"{name}_valid"] = self.channel_valid_mask(name)

        out.setdefault('source_file', os.path.normpath(self.paths[file_idx]))
        out.setdefault('source_file_basename', os.path.basename(self.paths[file_idx]))
        out.setdefault('source_file_index', file_idx)

        sample = AttributeDict(out)

        # Cache management
        if self.cache_size > 0:
            with self._cache_lock:
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[idx] = sample
        
        self._access_count += 1
        return sample
    
    def get_the_lists(self):
        """
        Retrieves lists of GUIDs, epochs, and targets for all samples in the dataset index.
        Note: This can be slow for large datasets as it iterates through all samples.
        """
        guids, epochs, targets = [], [], []
        
        indices_by_file = {}
        for f_idx, s_idx in self.index_map:
            if f_idx not in indices_by_file:
                indices_by_file[f_idx] = []
            indices_by_file[f_idx].append(s_idx)

        for f_idx, s_indices in indices_by_file.items():
            handle = self._open_handle(f_idx)
            # Sort indices to improve read performance, h5py recommends this
            s_indices.sort()
            
            # Use fancy indexing to read all required samples from this file at once
            guids.extend([g.decode('utf-8') for g in handle['guid'][s_indices]])
            epochs.extend(handle['epoch'][s_indices])
            targets.extend(handle['target'][s_indices])
            
        return guids, epochs, targets

    def clear_cache(self):
        """Clear the sample cache to free memory."""
        with self._cache_lock:
            self._cache.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for monitoring."""
        return {
            'total_samples': len(self.index_map),
            'num_files': len(self.paths),
            'cache_size': len(self._cache),
            'access_count': self._access_count,
            'dtype': str(self.dtype),
            'pin_memory': self.pin_memory,
            'normalization_enabled': self.normalization_enabled,
            'stats_path': self.stats_path,
        }
    
    def __del__(self):
        """Cleanup when dataset is garbage collected."""
        if hasattr(self, "file_handles"):
            self._cleanup_handles()
            self.clear_cache()


def attribute_dict_collate(batch):
    """
    Collate a batch of AttributeDicts into a single AttributeDict of batched tensors.
    """
    collated = default_collate(batch)
    return AttributeDict(collated)


def create_optimized_dataloader(
    hdf5_files: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    rank: int = 0,
    world_size: int = 1,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    prefetch_factor: int = 2,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader for multi-GPU training.
    
    Args:
        hdf5_files: List of HDF5 file paths
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        rank: Current GPU rank for distributed training
        world_size: Total number of GPUs
        stats_path: Path to HDF5 statistics file for data normalization
        normalize_fields: List of fields to normalize (None normalizes all available fields with stats)
        **dataset_kwargs: Additional arguments for CombinedHDF5Dataset
    
    Returns:
        Optimized DataLoader instance
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Create dataset
    dataset = CombinedHDF5Dataset(
        paths=hdf5_files,
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        **dataset_kwargs
    )
    
    # Setup distributed sampler if multi-GPU
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True  # Ensures consistent batch sizes across GPUs
        )
        default_shuffle = False
    else:
        default_shuffle = True

    if shuffle is None:
        shuffle = default_shuffle
    
    # Optimal DataLoader settings
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        # Loader-level pinning: the collated batch is copied into page-locked memory by the
        # loader's pin thread in the MAIN process, which is what lets the trainer's
        # non_blocking host-to-device copy overlap the transfer with compute. This is NOT the
        # dataset-level `pin_memory` kwarg (worker-side per-tensor pinning): pinned status does
        # not survive the worker->main IPC handoff, so that one buys nothing under workers and
        # should stay off. Gated on CUDA so CPU-only runs skip the pointless copy (and the
        # torch warning it would emit).
        pin_memory=torch.cuda.is_available(),
        collate_fn=attribute_dict_collate
    )


# ---------------------------------------
# Get per guid batches
# ---------------------------------------

class GuidBatchSampler(Sampler[List[int]]):
    """Yield one batch—that GUID’s samples—at a time."""
    def __init__(self, guid_to_indices: Dict[str, List[int]], shuffle: bool = False) -> None:
        super().__init__(None)
        self._guid_to_indices = {g: idxs[:] for g, idxs in guid_to_indices.items() if idxs}
        self._guid_order = sorted(self._guid_to_indices)
        self._shuffle = shuffle

    def __iter__(self) -> Iterable[List[int]]:
        order = self._guid_order[:]
        if self._shuffle:
            random.shuffle(order)
        for guid in order:
            yield self._guid_to_indices[guid]

    def __len__(self) -> int:
        return len(self._guid_to_indices)


def build_guid_filtered_dataloader(
    dataset_paths: Sequence[str],
    min_samples: int = 5,
    max_guids: Optional[int] = None,
    sampler_shuffle: bool = False,
    stats_path: Optional[str] = None,
    normalize_fields: Optional[Sequence[str]] = None,
    dataloader_overrides: Optional[Dict[str, Any]] = None,
    **dataset_kwargs: Any,
) -> Tuple[List[str], DataLoader]:
    """
    Create a DataLoader that yields one batch per GUID for GUIDs with > min_samples.
    Returns (eligible_guid_list, dataloader).

    Args:
        dataset_paths: Sequence of paths to HDF5 dataset files.
        min_samples: Minimum number of samples required for a GUID to be eligible.
        max_guids: Maximum number of GUIDs to include (None for no limit).
        sampler_shuffle: Whether to shuffle the GUID order in the sampler.
        stats_path: Path to normalization statistics file.
        normalize_fields: Fields to normalize.
        dataloader_overrides: Override default DataLoader kwargs.
        **dataset_kwargs: Additional kwargs for CombinedHDF5Dataset.
    """
    dataset = CombinedHDF5Dataset(
        paths=list(dataset_paths),
        stats_path=stats_path,
        normalize_fields=normalize_fields,
        **dataset_kwargs,
    )

    guids, _, _ = dataset.get_the_lists()
    counts = Counter(guids)
    eligible_guids = [guid for guid, count in counts.items() if count > min_samples]

    # Limit to max_guids if specified
    if max_guids is not None and len(eligible_guids) > max_guids:
        eligible_guids = eligible_guids[:max_guids]

    guid_to_indices: Dict[str, List[int]] = defaultdict(list)
    eligible_guids_set = set(eligible_guids)
    for idx, guid in enumerate(guids):
        if guid in eligible_guids_set:
            guid_to_indices[guid].append(idx)

    batch_sampler = GuidBatchSampler(guid_to_indices, shuffle=sampler_shuffle)

    loader_kwargs: Dict[str, Any] = {
        "batch_sampler": batch_sampler,
        "collate_fn": attribute_dict_collate,
        "pin_memory": getattr(dataset, "pin_memory", False),
        "num_workers": 0,
    }
    if dataloader_overrides:
        loader_kwargs.update(dataloader_overrides)

    dataloader = DataLoader(dataset, **loader_kwargs)
    return eligible_guids, dataloader
