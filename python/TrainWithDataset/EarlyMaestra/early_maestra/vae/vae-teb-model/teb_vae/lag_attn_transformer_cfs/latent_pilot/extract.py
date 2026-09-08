r"""Keyed latent extraction and the frozen training-only scaler.

Written in LP-06: what is extracted, from which forward outputs, and why the scaler is fitted
once and then never refitted.

Extraction
----------

The forward is called densely -- ``anchor_phase=0``, ``anchor_stride=1`` -- regardless of the tiling
the run trained at, which is the same policy the surrounding evaluation pipeline uses, so the anchor
set is deterministic and comparable across models. From the forward's returned dictionary the pilot
keeps ``mu_post``, ``mu_prior``, ``logvar_post``, ``logvar_prior``, ``anchor_index`` and
``anchor_valid``; the support mask comes from the forecast-contributing rule in ``data.py``.

**Raw means are preserved, not only the pooled summaries.** Every derived quantity in this package --
bags, bins, geometry, movement -- is recomputable from the stored per-anchor coordinates, and a run
that saved only the reductions could not answer a question that was not asked in advance. Arrays are
row-aligned with an identity index carrying ``(guid, epoch, anchor)``, the split, the anchor's
absolute time before delivery and its support flags. That index is what makes the before/after
comparison exact: the frozen and adapted models are compared on **the same exported keys**, and a
mismatch is refused rather than joined around.

Every extraction is fingerprinted with the checkpoint digest, the resolved config, the support policy
and the anchor geometry. An incompatible cache is rejected with the reason named; two extractions
whose keys differ are never compared.

Extraction of the **test** split is reachable only from the final evaluation path, after the models,
the threshold, the controls and the analysis settings are locked. ``extract`` itself collects train
and validation only.

Scaling
-------

Per-coordinate moments $m_d, s_d$ are computed hierarchically on **pretrained training anchors**,
giving each recording equal weight -- anchors within segments, segments within recordings, then
recordings -- so a long recording does not set the scale for the cohort. Scales are floored at
$\max(10^{-3},\, 0.1 \operatorname{median}\{s_d : s_d > 0\})$; if no coordinate varies at all the run
reports latent collapse instead of fitting anything.

The resulting $S(x) = (x - m)/s$ is **frozen** and applied unchanged to validation and test, to the
adapted model's latents, to the teacher's, and to the control's. Refitting it per model or per split
would silently rescale the very movement the comparison is measuring. It is a separate object from,
and no substitute for, the checkpoint's own input normalization, whose statistics file and fitting
population are recorded in the protocol.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from teb_vae.lag_attn_transformer_cfs.latent_pilot import data, model as pilot_model
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: The four per-anchor latent quantities every stage reads, in a fixed order.
#:
#: Both means and both log-variances: the means are what the pilot supervises and analyses, and the
#: log-variances are what the invariant check and the collapse diagnostic are read from. Stored raw,
#: per anchor -- every bag, bin, window and geometry summary in this package is recomputable from
#: them, and a run that had saved only its reductions could not answer a question nobody asked in
#: advance.
LATENT_KEYS: Tuple[str, ...] = ("mu_post", "mu_prior", "logvar_post", "logvar_prior")

#: On-disk names inside a run directory.
LATENT_ARRAYS_FILENAME = "latents.npz"
LATENT_INDEX_FILENAME = "latent_index.parquet"
LATENT_RECORD_FILENAME = "latents.json"
SCALER_FILENAME = "latent_scaler.json"

#: Stored as float32. The forward runs in float32 and these are read back for linear algebra on
#: bags of at most a few thousand recordings, so nothing here needs float64 -- and a half-precision
#: copy, which the evaluation pipeline uses for its lag maps, would put rounding into the
#: standardized movement this pilot measures.
STORAGE_DTYPE = np.float32


class LatentCollapse(PilotConfigError):
    """No latent coordinate varies across the training recordings.

    A separate failure from a bad setting: the checkpoint's posterior is constant, so there is
    nothing to standardize, nothing to discriminate along, and no adaptation worth fitting. Reported
    rather than worked around, because a scaler built on a floor alone would turn a degenerate
    representation into plausible-looking numbers.
    """


@dataclass(frozen=True)
class LatentExtraction:
    """One split's per-anchor latents, with the identity they are keyed by.

    Attributes:
        frame: One row per decoded, in-window anchor: split, GUID, segment epoch, anchor index,
            its absolute time and hour coordinate, its latest scored endpoint, its trajectory bin,
            the support flag, an exclusion reason (empty when retained) and ``row`` -- the index
            into every array below.
        arrays: :data:`LATENT_KEYS` to ``(N, d_z)`` matrices, row-aligned with ``frame``.
        fingerprint: What these were extracted under. Two extractions are comparable only if their
            fingerprints agree.
        record: Counts and coverage for the run's own report.
    """

    frame: pd.DataFrame
    arrays: Dict[str, np.ndarray]
    fingerprint: Dict[str, Any]
    record: Dict[str, Any]

    @property
    def retained(self) -> pd.DataFrame:
        """The rows that survived support, delivery and duplication."""
        return data.retained(self.frame)

    def values(self, key: str = "mu_post", *, rows: Optional[Sequence[int]] = None) -> np.ndarray:
        """Return one latent quantity, optionally gathered.

        Args:
            key: One of :data:`LATENT_KEYS`.
            rows: Row indices to gather, typically ``frame['row']`` of a filtered view.

        Returns:
            The matrix, ``(N, d_z)`` or ``(len(rows), d_z)``.

        Raises:
            KeyError: If the quantity was not extracted.
        """
        matrix = self.arrays[key]
        return matrix if rows is None else matrix[list(rows)]


# =============================================================================
# Fingerprints
# =============================================================================
def support_fingerprint(loaded: Any, *, preservation_hours: float) -> Dict[str, Any]:
    """Describe the support policy an extraction was collected under.

    Everything that decides *which anchors exist* and *what they mean*: the checkpoint's identity,
    the geometry it resolved, the coverage floor and forecast clock its mask is built from, the
    decoding geometry, and the window. Two extractions that disagree on any of these are not two
    readings of one population, and joining them would compare different anchor sets under one name.

    Args:
        loaded: The loaded checkpoint bundle.
        preservation_hours: The window anchors were kept in.

    Returns:
        The fingerprint.
    """
    geometry = dict(loaded.geometry)
    return {
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_digest": loaded.digest,
        "model_class": geometry.get("model_class"),
        "d_z": geometry.get("d_z"),
        "sequence_length": geometry.get("sequence_length"),
        "horizon": geometry.get("horizon"),
        "warmup": geometry.get("warmup"),
        "trim_minutes": geometry.get("trim_minutes"),
        "coverage_floor": geometry.get("coverage_floor"),
        "target_forecast_shift": geometry.get("target_forecast_shift"),
        # Always dense here, and recorded so a later reader does not have to trust that it was.
        "anchor_phase": 0,
        "anchor_stride": 1,
        "support_policy": "forecast_contributing",
        "preservation_hours": float(preservation_hours),
        "latent_keys": list(LATENT_KEYS),
    }


def check_compatible(left: Mapping[str, Any], right: Mapping[str, Any], *, what: str) -> None:
    """Refuse two artifacts that were not produced under the same contract.

    Args:
        left: The fingerprint already on disk, or the earlier one.
        right: The fingerprint being compared against it.
        what: What is being compared, for the message.

    Raises:
        PilotConfigError: Naming every field that differs, with both values.
    """
    differences = {
        key: (left.get(key), right.get(key))
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    if differences:
        raise PilotConfigError(
            f"{what} was produced under a different contract: {differences}. An extraction is a "
            f"reading of one anchor set under one checkpoint; mixing two would report a comparison "
            f"between populations as a comparison between models."
        )


def assert_same_keys(before: LatentExtraction, after: LatentExtraction) -> None:
    """Refuse a before/after pair that does not describe the same anchors, in the same order.

    The paired comparison is only paired if both models were read at identical
    ``(guid, epoch, anchor)`` rows. A join would hide a mismatch by dropping the rows that differ,
    which is exactly the case where the difference matters.

    Args:
        before: The pretrained extraction.
        after: The adapted extraction.

    Raises:
        PilotConfigError: On a length or key mismatch, naming the first offending row.
    """
    check_compatible(before.fingerprint, after.fingerprint, what="the second extraction")
    keys = [data.GUID_COLUMN, data.EPOCH_COLUMN, data.ANCHOR_COLUMN]
    left, right = before.frame[keys], after.frame[keys]
    if len(left) != len(right):
        raise PilotConfigError(
            f"the two extractions hold {len(left)} and {len(right)} anchors. They must be read at "
            f"exactly the same support, or the before/after difference includes a change of "
            f"population."
        )
    mismatch = np.flatnonzero(
        (left.to_numpy() != right.to_numpy()).any(axis=1)
    ) if len(left) else np.empty(0, dtype=int)
    if mismatch.size:
        row = int(mismatch[0])
        raise PilotConfigError(
            f"the two extractions disagree at row {row}: {left.iloc[row].tolist()} against "
            f"{right.iloc[row].tolist()} ({mismatch.size} row(s) differ)."
        )


# =============================================================================
# Extraction
# =============================================================================
def check_split_allowed(split: str, *, allow_test: bool) -> None:
    """Refuse a test-split extraction outside the final evaluation path.

    The test split stays unread until the baseline, the adapted checkpoint, the scaler, the
    threshold, the controls and the analysis settings are locked. Enforced in code rather than by
    convention, because "we only looked at it at the end" is not something a run directory can
    demonstrate afterwards.

    Args:
        split: The split being extracted.
        allow_test: Set only by the evaluation stage, after selection is locked.

    Raises:
        PilotConfigError: If the test split is requested from anywhere else.
    """
    if str(split) == "test" and not allow_test:
        raise PilotConfigError(
            "the test split may only be extracted from the evaluation stage, after selection is "
            "locked. Fitting, early stopping and threshold selection read train and validation "
            "only; an extraction here would put the held-out split inside the loop that chose the "
            "model."
        )


def extract_split(
    loaded: Any,
    loader: Any,
    *,
    split: str,
    preservation_hours: float,
    bin_hours: float,
    allow_test: bool = False,
    max_batches: Optional[int] = None,
) -> LatentExtraction:
    """Read one split's per-anchor latents at the dense anchor set.

    No sampling is involved and none is needed: the quantities kept are the posterior and prior
    **means** and log-variances, all deterministic functions of the inputs, so this pass is
    reproducible without pinning a draw. The model stays in evaluation mode and the whole pass runs
    under ``no_grad``.

    Anchors outside the preserved window are filtered **before** their vectors are gathered -- they
    are the window's definition rather than an exclusion of interest -- and counted. Everything
    else is marked on its row and kept: the support rule, the post-delivery endpoint check and both
    deduplication rules, in that order, so the first reason an anchor failed is the one reported.

    Args:
        loaded: The loaded checkpoint bundle.
        loader: This split's dataloader.
        split: The split name, stored on every row.
        preservation_hours: The window anchors are kept in.
        bin_hours: Trajectory bin width, assigned here so every consumer reads one binning.
        allow_test: Passed to :func:`check_split_allowed`.
        max_batches: Stop after this many batches. Smoke runs only -- the loader is unshuffled over
            concatenated per-subgroup shards, so a cap is a prefix rather than a sample.

    Returns:
        The extraction.

    Raises:
        PilotConfigError: If the test split is requested early, if the forward does not resolve the
            dense anchor geometry, or if the split yields no in-window anchor at all.
    """
    check_split_allowed(split, allow_test=allow_test)
    task, model = loaded.task, loaded.model
    geometry = dict(loaded.geometry)
    trim_minutes = geometry.get("trim_minutes")
    horizon = int(geometry["horizon"])
    shift = data.max_forecast_shift(model)

    rows: List[Dict[str, Any]] = []
    collected: Dict[str, List[np.ndarray]] = {key: [] for key in LATENT_KEYS}
    n_batches = 0
    n_decoded = 0
    n_outside_window = 0
    n_padded = 0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            # tqdm rather than a log line every N batches: this is the longest silent stretch of a
            # run, and a percentage is what an operator actually wants from it. It reads the
            # loader's length itself and falls back to a plain counter when there is none.
            for batch in tqdm(loader, desc=f"extract:{split}", unit="batch"):
                if max_batches is not None and n_batches >= max_batches:
                    break
                n_batches += 1
                batch = pilot_model.to_device(task, batch)
                outputs = model(*pilot_model.forward_inputs(task, batch))
                anchor_index = outputs["anchor_index"]
                anchor_valid = outputs["anchor_valid"]
                support, coverage = data.contributing_support(
                    model,
                    weight=data._field(batch, "weight"),
                    anchor_index=anchor_index,
                    anchor_valid=anchor_valid,
                )

                # The cohort module's own batch readers, so a field is read here exactly as it is
                # read by the pass that built the manifest -- one reader, one set of collation
                # quirks handled in one place.
                guids = data._strings_of(batch, data.GUID_COLUMN, int(anchor_index.shape[0]))
                epochs = data._to_numpy(data._field(batch, data.EPOCH_COLUMN))
                anchors = anchor_index.detach().cpu().numpy()
                valid = anchor_valid.detach().cpu().numpy().astype(bool)
                supported = support.detach().cpu().numpy().astype(bool)
                coverage_values = coverage.detach().cpu().numpy()

                hours = data.hours_before_delivery(
                    data.anchor_seconds(
                        epochs[:, None], anchors, trim_minutes=trim_minutes
                    )
                )
                inside = data.in_window(hours, 0.0, float(preservation_hours))
                keep = valid & inside
                n_decoded += int(valid.sum())
                n_padded += int((~valid).sum())
                n_outside_window += int((valid & ~inside).sum())
                if not keep.any():
                    continue

                # Gathered once per batch rather than per anchor: the latent tensors are
                # (B, T, d_z) and the anchors index the time axis.
                # The forward's own gather idiom: an index expanded over the latent axis,
                # applied to the (B, T, d_z) tensors at the anchors it decoded.
                gather_index = anchor_index[:, :, None].expand(-1, -1, int(model.d_z))
                gathered = {
                    key: outputs[key].gather(1, gather_index).detach().cpu().numpy()
                    for key in LATENT_KEYS
                }
                sample_rows, anchor_rows = np.nonzero(keep)
                for sample, anchor_position in zip(sample_rows.tolist(), anchor_rows.tolist()):
                    rows.append({
                        data.SPLIT_COLUMN: split,
                        data.GUID_COLUMN: guids[sample],
                        data.EPOCH_COLUMN: float(epochs[sample]),
                        data.ANCHOR_COLUMN: int(anchors[sample, anchor_position]),
                        "coverage_frac": float(coverage_values[sample, anchor_position]),
                        "contributing": bool(supported[sample, anchor_position]),
                        data.ROW_COLUMN: len(rows),
                    })
                for key in LATENT_KEYS:
                    collected[key].append(
                        gathered[key][sample_rows, anchor_rows].astype(STORAGE_DTYPE)
                    )
    finally:
        if was_training:
            model.train()

    if not rows:
        raise PilotConfigError(
            f"the {split!r} split yielded no anchor inside the last {preservation_hours} hour(s). "
            f"Either the coarse epoch filter is too tight -- it must be widened by a whole stored "
            f"segment so a crossing segment survives -- or these shards hold nothing near delivery."
        )

    frame = pd.DataFrame(rows)
    arrays = {key: np.concatenate(collected[key], axis=0) for key in LATENT_KEYS}
    frame = data.add_anchor_times(
        frame, trim_minutes=trim_minutes, horizon=horizon, forecast_shift=shift
    )
    # Support first: an anchor the objective never scores is excluded for that reason rather than
    # for whatever it would have failed next.
    frame[data.EXCLUSION_COLUMN] = np.where(
        frame["contributing"].to_numpy(dtype=bool), "", data.EXCLUDED_NOT_CONTRIBUTING
    )
    frame = data.mark_window_and_delivery(frame, preservation_hours=preservation_hours)
    frame = data.deduplicate_anchors(frame)
    frame = data.assign_time_bins(
        frame, bin_hours=bin_hours, preservation_hours=preservation_hours
    )

    exclusions = data.anchor_exclusion_counts(frame)
    record = {
        "split": split,
        "n_batches": n_batches,
        "n_segments": int(frame[[data.GUID_COLUMN, data.EPOCH_COLUMN]].drop_duplicates().shape[0]),
        "n_recordings": int(frame[data.GUID_COLUMN].nunique()),
        "n_decoded_anchors": n_decoded,
        "n_padded_slots": n_padded,
        "n_outside_window": n_outside_window,
        "n_in_window_anchors": int(len(frame)),
        "n_retained_anchors": int(len(data.retained(frame))),
        "exclusions": exclusions,
        "max_batches": max_batches,
        "support_note": (
            "support is the objective's own forecast-contributing rule, so these anchors are the "
            "ones the model could be scored at rather than every anchor a state could be inferred "
            "at; the difference is the forecast's availability exclusions"
        ),
    }
    logger.info(
        f"{split}: {record['n_retained_anchors']} retained of {record['n_in_window_anchors']} "
        f"in-window anchors over {record['n_segments']} segment(s) and "
        f"{record['n_recordings']} recording(s); exclusions {exclusions}"
    )
    return LatentExtraction(
        frame=frame,
        arrays=arrays,
        fingerprint=support_fingerprint(loaded, preservation_hours=preservation_hours),
        record=record,
    )


# =============================================================================
# Persistence
# =============================================================================
def save_extraction(extraction: LatentExtraction, directory: Any, *, name: str) -> Path:
    """Write one extraction into a run directory.

    Args:
        extraction: The extraction.
        directory: The run directory. Created if absent.
        name: A prefix distinguishing, for example, the pretrained reading from the adapted one.

    Returns:
        The directory the three files were written into.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"{name}_{LATENT_ARRAYS_FILENAME}", **extraction.arrays)
    extraction.frame.to_parquet(path / f"{name}_{LATENT_INDEX_FILENAME}", index=False)
    (path / f"{name}_{LATENT_RECORD_FILENAME}").write_text(
        json.dumps(
            {"fingerprint": extraction.fingerprint, "record": extraction.record},
            indent=2, sort_keys=True, default=str,
        ),
        encoding="utf-8",
    )
    logger.info(f"wrote {name} extraction ({len(extraction.frame)} anchors) to {path}")
    return path


def load_extraction(
    directory: Any, *, name: str, expected: Optional[Mapping[str, Any]] = None
) -> LatentExtraction:
    """Read an extraction back, refusing one produced under another contract.

    Args:
        directory: The run directory.
        name: The prefix used when it was written.
        expected: The fingerprint it must match, or ``None`` to accept whatever is there. A stage
            that is about to compare two readings passes it; a stage merely reporting does not.

    Returns:
        The extraction.

    Raises:
        FileNotFoundError: If any of the three files is missing.
        PilotConfigError: If the stored fingerprint does not match ``expected``.
    """
    path = Path(directory)
    arrays_path = path / f"{name}_{LATENT_ARRAYS_FILENAME}"
    index_path = path / f"{name}_{LATENT_INDEX_FILENAME}"
    record_path = path / f"{name}_{LATENT_RECORD_FILENAME}"
    for candidate in (arrays_path, index_path, record_path):
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{candidate} is missing, so the {name!r} extraction cannot be read back. Run the "
                f"extraction stage before the stage that consumes it."
            )
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    fingerprint = dict(payload.get("fingerprint") or {})
    if expected is not None:
        check_compatible(fingerprint, dict(expected), what=f"the stored {name!r} extraction")
    with np.load(arrays_path) as handle:
        arrays = {key: handle[key] for key in LATENT_KEYS if key in handle}
    return LatentExtraction(
        frame=pd.read_parquet(index_path),
        arrays=arrays,
        fingerprint=fingerprint,
        record=dict(payload.get("record") or {}),
    )


# =============================================================================
# The frozen training scaler
# =============================================================================
@dataclass(frozen=True)
class LatentScaler:
    r"""The standardizing constants, fitted once and then immutable.

    $$S(x) = (x - m) / s.$$

    Frozen in the strong sense: the dataclass is immutable, the arrays it holds are marked
    read-only, and nothing in this package refits it. Refitting per model or per split would
    silently rescale the very movement the comparison exists to measure.

    Attributes:
        center: $m$, one per latent coordinate.
        scale: $s$, one per latent coordinate, after the floor.
        record: How it was fitted -- the population, the hierarchy, the floor and what it changed.
    """

    center: np.ndarray
    scale: np.ndarray
    record: Dict[str, Any]

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Standardize a value matrix.

        Args:
            values: ``(N, d_z)``.

        Returns:
            A new matrix; the input is not modified.
        """
        return (np.asarray(values, dtype=np.float64) - self.center) / self.scale


def _hierarchical_moments(
    frame: pd.DataFrame, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    r"""First and second moments with **every recording weighted equally**.

    Anchors are averaged within segments, segments within recordings, and recordings within the
    population -- the same three-level order every reduction in this pilot uses. A pooled moment
    would instead weight each recording by how many anchors it happened to contribute, so a long,
    well-covered recording would set the scale for the cohort and the standardization would differ
    between two folds that hold the same patients.

    Args:
        frame: Retained training anchors.
        values: The aligned per-anchor matrix.

    Returns:
        ``(mean, second moment, counts)``, both moments over the recording-weighted distribution.
    """
    matrix = np.asarray(values, dtype=np.float64)
    squares = matrix ** 2
    recording_first: List[np.ndarray] = []
    recording_second: List[np.ndarray] = []
    n_segments = 0
    for _guid, recording in frame.groupby(data.GUID_COLUMN, sort=True):
        segment_first: List[np.ndarray] = []
        segment_second: List[np.ndarray] = []
        for _key, segment in recording.groupby(data.EPOCH_COLUMN, sort=True):
            positions = segment[data.ROW_COLUMN].tolist()
            segment_first.append(matrix[positions].mean(axis=0))
            segment_second.append(squares[positions].mean(axis=0))
        n_segments += len(segment_first)
        recording_first.append(np.mean(np.stack(segment_first), axis=0))
        recording_second.append(np.mean(np.stack(segment_second), axis=0))
    counts = {
        "n_recordings": len(recording_first),
        "n_segments": n_segments,
        "n_anchors": int(len(frame)),
    }
    return (
        np.mean(np.stack(recording_first), axis=0),
        np.mean(np.stack(recording_second), axis=0),
        counts,
    )


def fit_scaler(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    key: str = "mu_post",
    minimum_scale: float = 1.0e-3,
    relative_floor: float = 0.1,
) -> LatentScaler:
    r"""Fit the standardizing constants on **pretrained training anchors only**.

    Scales are floored at $\max(10^{-3},\, 0.1 \operatorname{median}\{s_d : s_d > 0\})$: a
    coordinate that barely varies would otherwise be amplified until its noise dominated the
    standardized geometry, and the classification direction would be fitted to it.

    Args:
        frame: Retained anchors. Must carry the training split and nothing else.
        values: The aligned per-anchor matrix.
        key: Which latent quantity is being scaled, recorded only.
        minimum_scale: The absolute floor.
        relative_floor: The fraction of the median positive scale used as the relative floor.

    Returns:
        The scaler, immutable and with its fitting population recorded.

    Raises:
        PilotConfigError: If the frame is empty, if it carries any split other than training, or
            if the fitted moments are not finite. The constants must be a property of the training
            population alone -- fitting them on anything else leaks the evaluation distribution
            into every standardized number the run reports -- and they must be finite, since a NaN
            constant propagates silently into every later number rather than failing anywhere.
        LatentCollapse: If no coordinate varies at all.
    """
    # Emptiness first: an empty frame's split set is ``[]``, which is not ``["train"]``, so the
    # split check below would fire on it and blame contamination for a cohort that retained
    # nothing at all.
    if frame.empty:
        raise PilotConfigError(
            "no retained training anchor to fit the scaler on: every in-window training anchor "
            "was excluded by the support, delivery or duplication rules. The extraction record's "
            "'exclusions' counts say which rule removed them."
        )
    splits = sorted({str(value) for value in frame[data.SPLIT_COLUMN].tolist()})
    if splits != ["train"]:
        raise PilotConfigError(
            f"the scaler must be fitted on the training split alone, but the frame carries "
            f"{splits}. Standardizing constants fitted on validation or test would carry those "
            f"populations into every comparison this run makes."
        )

    first, second, counts = _hierarchical_moments(frame, values)
    # Finiteness before anything derived from these moments. A NaN reaches every later number in
    # the run: it survives ``maximum(..., 0.0)``, compares False against the collapse guard and
    # against the floor, and lands in latent_scaler.json, after which every standardized value in
    # that coordinate is NaN for both models and all three splits. The classifier logits go NaN,
    # the validation AUROC goes NaN, and a non-finite AUROC counts as "not better", so the fit
    # quietly retains epoch zero and the run finishes with no stage naming the cause.
    unusable = ~np.isfinite(first) | ~np.isfinite(second)
    if bool(unusable.any()):
        raise PilotConfigError(
            f"{key} is not finite at coordinate(s) {np.flatnonzero(unusable).tolist()} over the "
            f"{counts['n_recordings']} training recording(s), so the standardizing constants "
            f"would be NaN and so would every number standardized by them. The extraction is what "
            f"has to be fixed, not the scaler: check the checkpoint's input statistics and the "
            f"shards behind those recordings."
        )
    variance = np.maximum(second - first ** 2, 0.0)
    raw_scale = np.sqrt(variance)

    positive = raw_scale[raw_scale > 0.0]
    if positive.size == 0:
        raise LatentCollapse(
            f"no {key} coordinate varies across the {counts['n_recordings']} training "
            f"recording(s): the pretrained posterior mean is constant, so there is nothing to "
            f"standardize and nothing to discriminate along. Report the collapse rather than "
            f"fitting against a floor."
        )
    floor = max(float(minimum_scale), float(relative_floor) * float(np.median(positive)))
    scale = np.maximum(raw_scale, floor)

    center = np.array(first, dtype=np.float64)
    scale = np.array(scale, dtype=np.float64)
    # Read-only, so an accidental in-place edit downstream fails loudly instead of quietly
    # rescaling every number computed after it.
    center.setflags(write=False)
    scale.setflags(write=False)
    record = {
        "key": key,
        "population": "train",
        "hierarchy": "anchors within segments, segments within recordings, recordings equally",
        **counts,
        "d_z": int(center.size),
        "floor": floor,
        "minimum_scale": float(minimum_scale),
        "relative_floor": float(relative_floor),
        "n_coordinates_at_floor": int(np.sum(raw_scale < floor)),
        "n_coordinates_with_zero_variance": int(np.sum(raw_scale <= 0.0)),
        "median_positive_scale": float(np.median(positive)),
    }
    logger.info(
        f"scaler fitted on {counts['n_recordings']} training recording(s): floor {floor:.4g}, "
        f"{record['n_coordinates_at_floor']} of {record['d_z']} coordinate(s) raised to it"
    )
    return LatentScaler(center=center, scale=scale, record=record)


def save_scaler(scaler: LatentScaler, directory: Any) -> Path:
    """Write the scaler beside the run's other artifacts.

    Args:
        scaler: The fitted scaler.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / SCALER_FILENAME
    target.write_text(
        json.dumps(
            {
                "center": scaler.center.tolist(),
                "scale": scaler.scale.tolist(),
                "record": scaler.record,
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


def load_scaler(directory: Any) -> LatentScaler:
    """Read a fitted scaler back.

    Args:
        directory: The run directory.

    Returns:
        The scaler, with its arrays read-only again.

    Raises:
        FileNotFoundError: If it was never written.
    """
    target = Path(directory) / SCALER_FILENAME
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} is missing. The scaler is fitted once, during extraction, and every later "
            f"stage reads that one file rather than refitting."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    center = np.asarray(payload["center"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    center.setflags(write=False)
    scale.setflags(write=False)
    return LatentScaler(center=center, scale=scale, record=dict(payload.get("record") or {}))
