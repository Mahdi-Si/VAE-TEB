r"""Cohort tables, tasks, labeling strategies, context columns and fold checks (SPEC §6, §7.1, §12).

One pass per fold x split over ``CombinedHDF5Dataset`` with metadata-only fields, through the
pilot's :func:`segment_frame` (class code from target/weight, flags, clocks), extended with the §6.1
columns. GUID consolidation is the pilot's :func:`recording_frame` (conflicts exclude, never a
majority vote). Split disjointness, patient grouping, class presence and exposure records are the
pilot's too; the second-stage sentinel is detected by ``lag_attn_cfs`` and, deliberately unlike
there, turned into ``stage = unknown`` here (L7).

Clocks, all in seconds relative to delivery, with ``lead = 4 * trim_steps`` and ``end = lead + 4T``
(60 s and 1260 s at the shipped 1-minute trim)::

    t_end_s   = epoch_s + end                        tlo_end_s = time_from_labor_onset + end
    stage     = first if ss + end <= 0, second if ss + lead >= 0, straddle otherwise, unknown if NaN

``hours_to_delivery``, ``epoch_s``, ``t_end_s``, ``cs``, ``bg``, ``source_file`` and ``class_code``
are for labels, evaluation and strata only; :func:`context_features` is the §7.1 allow-list.
Clinical covariates (§7.3) join on ``guid_norm`` (:func:`join_covariates`: static, and timed by a strictly causal
as-of join) into raw ``cov:<name>`` / ``cov_age_h:<name>`` columns of the segment table; ``data.py`` encodes them
with train-fold statistics (L4). Their GUID-level availability feeds the L10 confound check.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from loguru import logger
from hdf5_dataset.hdf5_dataset import (
    DECIMATION, RAW_SAMPLING_HZ, AttributeDict, CombinedHDF5Dataset, attribute_dict_collate,
)
from teb_vae.classifier.config import (
    TASKS, Classifier, CohortCfg, CovariatesCfg, DataCfg, LabelsCfg, SourceCfg, resolve_path,
)
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn.eval.report import json_safe
from teb_vae.lag_attn_cfs.eval.cohort import second_stage_eligibility
from teb_vae.lag_attn_transformer_cfs.latent_pilot.data import (
    EXCLUDED_CONFLICTING_CLASS, EXCLUDED_CONFLICTING_METADATA, EXCLUDED_NO_CLASS,
    EXCLUDED_UNKNOWN_CLASS, PATIENT_COLUMN, PilotConfigError, attach_patient_groups,
    check_split_disjoint, exposure_record, recording_frame, require_both_classes, segment_frame,
)

SPLITS = ("train", "val", "test")
STAGES = ("first", "straddle", "second", "unknown")
#: Stage flags of the context vector (§7.1): first and unknown are the all-zero level, since an ``unknown`` flag
#: would equal ``~has_ss`` from segment 0 (future information).
CONTEXT_STAGES = ("straddle", "second")
SECONDS_PER_STEP = DECIMATION / RAW_SAMPLING_HZ
META_FIELDS = ("target", "weight", "epoch", "guid", "time_from_labor_onset",
               "second_stage_onset", "cs_label", "bg_label")
KEYS = ["fold", "split", "guid"]
#: recording_frame reasons that make a GUID's segments ``label_conflict``.
LABEL_REASONS = (EXCLUDED_NO_CLASS, EXCLUDED_CONFLICTING_CLASS, EXCLUDED_UNKNOWN_CLASS,
                 EXCLUDED_CONFLICTING_METADATA)
SEGMENT_COLUMNS = [
    "fold", "split", "ds_index", "source_file", "guid", "guid_norm", "epoch_s", "t_end_s", "slot",
    "seg_pos", "class_code", "clinical_class", "cs", "bg", "tlo_end_s", "ss_rel_s", "stage",
    "valid_frac", "hours_to_delivery", "y", "label_weight", "in_eval_window", "excluded",
    "exclusion_reason",
]
GUID_COLUMNS = [
    "fold", "split", "guid", "guid_norm", "source_file", "class_code", "clinical_class", "y", "cs",
    "bg", "n_segments", "first_epoch_s", "last_t_end_s", "has_tlo", "has_ss", "shared_test",
    "excluded", "exclusion_reason", "patient",
]


def normalize_guid(guid: str) -> str:
    """Uppercase, hyphens removed: ``create_new_pipeline._normalize_guid`` (not importable here)."""
    return str(guid).strip().upper().replace("-", "")


# ---- segment table (§6.1) ----------------------------------------------------------------------
def _samples(dataset: CombinedHDF5Dataset) -> Iterator[AttributeDict]:
    """``dataset[i]`` for every i in order, from one h5py read per column per file. The loader reads every field of
    every row separately, and a scalar read decompresses a whole lzf chunk: ~3 ms a segment, ~40 min over a real
    10-fold cohort (each GUID sits in every fold's shards). Same fields, trim and tensor conversion as
    ``CombinedHDF5Dataset.__getitem__`` (unnormalised metadata; no coefficient field)."""
    lo = dataset.trim_samples_decimated if dataset.trim_minutes is not None else 0
    for file_idx, group in itertools.groupby(dataset.index_map, key=lambda pair: pair[0]):
        rows, path = np.fromiter((i for _, i in group), dtype=np.int64), dataset.paths[file_idx]
        with h5py.File(path, "r") as h5:
            cols = {name: h5[name][()][rows] for name in dataset.load_fields if name in h5}
        where = {"source_file": os.path.normpath(path), "source_file_basename": os.path.basename(path),
                 "source_file_index": file_idx}
        for i in range(rows.size):
            out = {}
            for name, col in cols.items():
                value = col[i]
                if name == "guid":
                    out[name] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                elif name in ("cs_label", "bg_label"):
                    out[name] = bool(value)
                else:
                    out[name] = dataset._create_tensor(np.asarray(value[lo:-lo] if lo and name in ("target", "weight")
                                                                  else value))
            yield AttributeDict(out | where)


def _read_split(paths: Sequence[str], *, fold: int, split: str,
                trim_minutes: float) -> Tuple[pd.DataFrame, float, float]:
    """One metadata pass over a split; returns the frame and the observed window ``(lead, end)``."""
    dataset = CombinedHDF5Dataset(list(paths), load_fields=META_FIELDS, cache_size=0,
                                  pin_memory=False, trim_minutes=trim_minutes)
    valid = []

    def batches():
        samples = _samples(dataset)
        while chunk := list(itertools.islice(samples, 256)):
            batch = attribute_dict_collate(chunk)
            valid.append(batch["weight"].double().mean(1).numpy())
            yield batch

    frame = segment_frame(batches(), split=split)
    frame["valid_frac"] = np.concatenate(valid)
    frame["fold"] = fold
    lead = SECONDS_PER_STEP * dataset.trim_samples_decimated
    return frame, lead, lead + SECONDS_PER_STEP * len(dataset[0]["weight"])


def infer_stride(segments: pd.DataFrame) -> float:
    """Mode of positive within-GUID epoch differences; every difference must be a multiple of it."""
    diffs = segments.sort_values("epoch_s").groupby(KEYS)["epoch_s"].diff().round()
    diffs = diffs[diffs > 0]
    modes = diffs.mode()
    if len(modes) != 1 or (diffs % modes.iloc[0]).any():
        raise ValueError(
            f"segment stride is not unimodal: within-GUID epoch differences "
            f"{diffs.value_counts().head(8).to_dict()}. Set data.stride_s explicitly if intended."
        )
    return float(modes.iloc[0])


def stage_of(ss_rel_s: Any, lead_s: float, end_s: float) -> np.ndarray:
    """Labour stage over the observed window ``[ss + lead, ss + end]`` (§2.4, trim-aware)."""
    ss = np.asarray(ss_rel_s, dtype=float)
    return np.select([np.isnan(ss), ss + end_s <= 0, ss + lead_s >= 0],
                     ["unknown", "first", "second"], "straddle")


def segment_table(shards: Mapping[Tuple[int, str], Sequence[str]], *, trim_minutes: float = 1.0,
                  stride_s: Any = "auto", epoch_min_s: float = -44640.0,
                  min_valid_frac: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the segment table with segment-level exclusions (§6.1), no fold validation.

    Args:
        shards: ``{(fold, split): [shard paths]}``; any labels, e.g. ``{(0, "test"): [path]}``.
        trim_minutes: Loader trim; sets the observed window.
        stride_s: ``"auto"`` (inferred) or seconds.
        epoch_min_s: Segments starting earlier are ``outside_window`` (one window, L6).
        min_valid_frac: Segments with a lower mean weight are ``low_valid_frac``.

    Returns:
        ``(segments, info)``; GUID-level columns (``y``, ``seg_pos`` ...) come from
        :func:`guid_table`. ``info`` holds stride, window, sentinel and stage counts.
    """
    parts, windows = [], set()
    for (fold, split), paths in shards.items():
        frame, lead, end = _read_split(paths, fold=fold, split=split, trim_minutes=trim_minutes)
        parts.append(frame)
        windows.add((lead, end))
    if len(windows) != 1:
        raise ValueError(f"shards disagree on the observed window {windows}; mixed builds?")
    (lead, end), = windows

    seg = pd.concat(parts, ignore_index=True).rename(
        columns={"sample_index": "ds_index", "epoch": "epoch_s", "cs_label": "cs", "bg_label": "bg"})
    seg["guid"] = seg["guid"].astype(str)
    seg["guid_norm"] = seg["guid"].map(normalize_guid).astype(str)
    seg["source_file"] = seg.pop("source_file_basename").map(lambda n: Path(n).stem).astype(str)
    seg["class_code"] = seg["class_code"].astype("Int64")
    seg[["cs", "bg"]] = seg[["cs", "bg"]].astype(bool)
    stride = infer_stride(seg) if stride_s == "auto" else float(stride_s)
    first = seg.groupby(KEYS)["epoch_s"].transform("min")
    seg["slot"] = ((seg["epoch_s"] - first) / stride).round().astype(int)
    seg["t_end_s"] = seg["epoch_s"] + end
    seg["tlo_end_s"] = seg.pop("time_from_labor_onset") + end

    ss = seg.pop("second_stage_onset")
    onset = second_stage_eligibility(
        pd.DataFrame({"guid": seg["guid"], "epoch": seg["epoch_s"], "second_stage_onset": ss}))
    sentinel = sorted(onset["guid"][onset["onset_at_delivery"].astype(bool)])
    is_sentinel = seg["guid"].isin(sentinel)
    seg["ss_rel_s"] = ss.where(~is_sentinel)  # L7: onset at delivery is unknown, not "second"
    seg["stage"] = pd.Series(stage_of(seg["ss_rel_s"], lead, end), index=seg.index, dtype=str)
    seg["hours_to_delivery"] = -seg["t_end_s"] / 3600.0

    duplicate = seg.assign(_e=seg["epoch_s"].round()).duplicated(KEYS + ["_e"])  # L9, keep first
    reason = np.select(
        [duplicate, seg["epoch_s"] < epoch_min_s, seg["t_end_s"] > 0,  # L6, L8
         seg["valid_frac"] < min_valid_frac],
        ["duplicate_epoch", "outside_window", "crosses_delivery", "low_valid_frac"], "")
    seg["exclusion_reason"] = pd.Series(reason, index=seg.index, dtype=str)
    seg["excluded"] = seg["exclusion_reason"] != ""
    info = {
        "stride_s": stride, "window_s": [lead, end], "sentinel_guids": sentinel,
        "n_sentinel_segments": int(is_sentinel.sum()),
        "stage_counts": {s: int((seg["stage"] == s).sum()) for s in STAGES},
        "tlo_nan_rate": float(seg["tlo_end_s"].isna().mean()),
    }
    logger.info(f"segment table: {len(seg)} segments, {seg['guid'].nunique()} GUIDs, stride "
                f"{stride:g} s, stages {info['stage_counts']}, {len(sentinel)} sentinel GUID(s)")
    return seg, info


# ---- tasks and labeling strategies (§6.3, §6.5, §6.6) ------------------------------------------
def strategy_weights(delta_h: Any, y: Any, is_last: Any, *, strategy: str, horizon_h: float,
                     decay_halflife_h: float) -> np.ndarray:
    """Per-segment loss weight ω_n (§6.5); the target is always the GUID's ``y``.

    Negatives (``y == 0``) get 1 under ``propagate``/``horizon*``. ``final_only`` weights only
    the last position (both classes); ``mil`` has no per-segment term (all 0). ``k_warm`` is not here: it drops
    positions from the per-position loss only (``data.GuidDataset``'s ``w_pos``), never the segment-local term.
    """
    delta, positive = np.asarray(delta_h, float), np.asarray(y) > 0
    if strategy == "final_only":
        return np.asarray(is_last, float)
    if strategy == "mil":
        return np.zeros_like(delta)
    if strategy == "horizon":
        w = np.where(positive, (delta <= horizon_h).astype(float), 1.0)
    elif strategy == "horizon_decay":
        w = np.where(positive, 2.0 ** (-np.maximum(0.0, delta - horizon_h) / decay_halflife_h), 1.0)
    else:  # propagate
        w = np.ones_like(delta)
    return w


def warm_positions(labels: LabelsCfg, scope: str) -> int:
    """The effective ``labels.k_warm`` (§6.5): ``auto`` is 3 for ``propagate`` and 0 for the others; always 0 in
    segment scope, which has no sequence context to warm up (the config refuses an explicit k there)."""
    if scope == "segment":
        return 0
    return (3 if labels.strategy == "propagate" else 0) if labels.k_warm == "auto" else int(labels.k_warm)


def guid_table(seg: pd.DataFrame, labels: LabelsCfg, cohort: CohortCfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """GUID table (§6.2) and the task/label columns of the segment table.

    GUID exclusion reasons, first match wins: ``recording_frame``'s (conflicting codes or metadata,
    no class), ``healthy_no_bg`` (``cohort.include_healthy_no_bg: false``), ``task_exclude`` (code
    mapped to exclude), ``min_segments`` (retained segments < ``min_segments_per_guid``). Segments
    of an excluded GUID inherit its reason (``label_conflict`` for the label reasons), as does a retained
    GUID's segment without a class code.
    """
    seg = seg.copy()
    pilot = seg.rename(columns={"epoch_s": "epoch", "cs": "cs_label", "bg": "bg_label",
                                "source_file": "source_file_basename"})
    pilot["class_code"] = pilot["class_code"].astype(float)  # pd.NA -> NaN, recording_frame's "no class"
    rec = pd.concat([recording_frame(g).assign(fold=fold)
                     for (fold, _), g in pilot.groupby(["fold", "split"])], ignore_index=True)
    rec = rec[KEYS + ["class_code", "clinical_class", "exclusion_reason"]]

    by = [seg[k] for k in KEYS]  # a segment without a class code is excluded below: it never counts
    grouped, kept = seg.groupby(KEYS), seg[~seg["excluded"] & seg["class_code"].notna()].groupby(KEYS)
    stats = pd.DataFrame({
        "guid_norm": grouped["guid_norm"].first(), "source_file": grouped["source_file"].first(),
        "cs": grouped["cs"].first(), "bg": grouped["bg"].first(),
        "has_tlo": seg["tlo_end_s"].notna().groupby(by).any(),
        "has_ss": seg["ss_rel_s"].notna().groupby(by).any(),
        "n_segments": kept.size(), "first_epoch_s": kept["epoch_s"].min(),
        "last_t_end_s": kept["t_end_s"].max(),
    })
    guids = rec.merge(stats.reset_index(), on=KEYS)
    guids["guid"] = guids["guid"].astype(str)
    guids["class_code"] = guids["class_code"].astype("Int64")
    guids["n_segments"] = guids["n_segments"].fillna(0).astype(int)
    y = (guids["cs"].astype(float) if labels.task == "cs_outcome"
         else guids["class_code"].astype(float).map(TASKS[labels.task]))
    reason = guids["exclusion_reason"].astype(str)
    for name, hit in (
        ("healthy_no_bg", (guids["class_code"] == 1).fillna(False) & ~guids["bg"]
         & (not cohort.include_healthy_no_bg)),
        ("task_exclude", y.isna()),
        ("min_segments", guids["n_segments"] < cohort.min_segments_per_guid),
    ):
        reason = reason.mask((reason == "") & hit, name)
    guids["exclusion_reason"] = reason.astype(str)
    guids["excluded"] = reason != ""
    guids["y"] = y.astype("Int64")
    logger.info(f"GUID exclusions ({labels.task}): "
                f"{guids['exclusion_reason'][guids['excluded']].value_counts().to_dict()}")

    joined = seg[KEYS].merge(guids[KEYS + ["exclusion_reason", "y"]], on=KEYS, how="left")
    inherited = joined["exclusion_reason"].where(
        ~joined["exclusion_reason"].isin(LABEL_REASONS), "label_conflict").to_numpy()
    inherited = np.where((inherited == "") & seg["class_code"].isna().to_numpy(), "label_conflict",
                         inherited)  # a retained GUID's segment without a class code of its own
    seg["exclusion_reason"] = seg["exclusion_reason"].where(
        seg["exclusion_reason"] != "", inherited).astype(str)
    seg["excluded"] = seg["exclusion_reason"] != ""
    seg["y"] = joined["y"].astype("Int64").set_axis(seg.index)
    retained = ~seg["excluded"]
    seg["seg_pos"] = -1
    seg.loc[retained, "seg_pos"] = (
        seg[retained].groupby(KEYS)["epoch_s"].rank(method="first").astype(int) - 1)
    is_last = retained & (seg["seg_pos"] == seg.groupby(KEYS)["seg_pos"].transform("max"))
    weight = strategy_weights(
        seg["hours_to_delivery"], seg["y"].fillna(0).to_numpy(int), is_last, strategy=labels.strategy,
        horizon_h=labels.horizon_h, decay_halflife_h=labels.decay_halflife_h)
    seg["label_weight"] = np.where(retained, weight, 0.0)
    # bins: every retained segment; the per-bin segment metrics come from the M6 time-resolved analysis.
    window = {"all": True, "bins": True, "horizon": seg["hours_to_delivery"] <= labels.horizon_h,
              "stage:first": seg["stage"] == "first", "stage:second": seg["stage"] == "second"}
    seg["in_eval_window"] = retained & window[labels.eval_window]
    return seg, guids


# ---- fold validation (§6.7) --------------------------------------------------------------------
def fold_shards(data: DataCfg, fold: int) -> Dict[str, list]:
    """``{split: sorted shard paths}`` of one fold, after the ``data.subgroups`` allow-list."""
    root = resolve_path(data.kfold_root) / data.fold_dir.format(k=fold)
    out = {}
    for split in SPLITS:
        paths = sorted((root / getattr(data.split_dirs, split)).glob("*.hdf5"))
        if data.subgroups != "all":
            paths = [p for p in paths if p.stem in data.subgroups]
        if not paths:
            raise FileNotFoundError(f"no shards for fold {fold} / {split} under {root}")
        out[split] = [str(p) for p in paths]
    return out


def patient_groups(guids: pd.DataFrame, patient_map: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """``attach_patient_groups`` joined on ``guid_norm``, like every external table (§6.7): the map's keys are
    normalised (:func:`normalize_guid`), so a map in either spelling groups the shard GUIDs. A GUID the map does not
    cover is its own group (its raw GUID); two keys that normalise alike must name the same patient."""
    if patient_map is None:
        return attach_patient_groups(guids)
    loaded = json.loads(Path(patient_map).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PilotConfigError(f"patient map {patient_map} must be a JSON object mapping GUID to a patient id")
    norm: Dict[str, str] = {}
    for key, value in loaded.items():
        if norm.setdefault(normalize_guid(str(key)), str(value)) != str(value):
            raise PilotConfigError(f"patient map {patient_map}: {key!r} normalises to a GUID mapped to another patient")
    hit = guids["guid_norm"].astype(str).map(norm)
    frame = guids.assign(**{PATIENT_COLUMN: hit.fillna(guids["guid"]).astype(str)})
    covered = int(hit.notna().sum())
    return frame, {"patient_map_supplied": True, "n_recordings": len(frame), "n_recordings_with_patient_id": covered,
                   "n_distinct_groups": int(frame[PATIENT_COLUMN].nunique()),
                   "grouping": "patient" if covered else "guid"}


def validate_folds(guids: pd.DataFrame, *, task: str,
                   patient_map: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """L2 disjointness (GUID, and patient when mapped), class presence, ``shared_test`` flag and the
    ``patient`` column (the mapped id, else the GUID: the bootstrap's cluster unit)."""
    grouping: Dict[str, Any] = {}
    path = None if patient_map is None else resolve_path(patient_map)
    guids = patient_groups(guids, path)[0].astype({PATIENT_COLUMN: str})
    for fold, g in guids.groupby("fold"):
        try:
            check_split_disjoint(g)
            if path is not None:
                grouping[str(fold)] = patient_groups(g.drop(columns=PATIENT_COLUMN), path)[1]
                check_split_disjoint(g, group_column=PATIENT_COLUMN)
        except PilotConfigError:
            logger.error(f"L2: fold {fold} splits are not disjoint")
            raise
        kept = g[~g["excluded"]]
        for k in (range(3) if task == "three_class" else [1]):
            require_both_classes(kept.assign(outcome=(kept["y"] == k).astype(int)),
                                 splits=SPLITS, eligible_only=False)
    test = guids[guids["split"] == "test"]
    folds_per_guid = test.groupby("guid")["fold"].nunique()
    guids = guids.assign(shared_test=guids["guid"].isin(folds_per_guid.index[folds_per_guid > 1]))
    return guids, {"patient_grouping": grouping or "guid",
                   "n_shared_test_guids": int((folds_per_guid > 1).sum())}


def _shard_guids(paths: Sequence[str]) -> set:
    """Normalised GUIDs stored in ``paths``."""
    out: set = set()
    for path in paths:
        with h5py.File(resolve_path(path), "r") as handle:
            out |= {normalize_guid(g.decode() if isinstance(g, bytes) else g)
                    for g in handle["guid"][()]}
    return out


def resolved_config_for(checkpoint: Path) -> Path:
    """The training run's ``resolved_config.yaml`` for ``checkpoint``.

    Port of ``lag_attn_cfs.eval.probe.resolved_config_for``: that module does not import at HEAD
    (``eval/binding.py`` imports a missing ``analyses.time_shift``).
    """
    run_root = checkpoint.parent.parent
    for candidate in (checkpoint.parent, run_root / "train_results", run_root):
        if (candidate / "resolved_config.yaml").is_file():
            return candidate / "resolved_config.yaml"
    raise FileNotFoundError(f"L3: no resolved_config.yaml beside {checkpoint}; pretraining "
                            f"exposure cannot be verified")


def pretrain_exposure(guids: pd.DataFrame, source: SourceCfg, *,
                      allow_overlap: bool) -> Dict[str, Any]:
    """L3: the VAE's train/val shards vs every split; overlap with test raises unless allowed. A shard list the
    checkpoint's resolved config does not name (absent, null or empty) is an UNKNOWN population (``exposure_record``
    ``known: false``), never a known empty one: the C12 check is then inconclusive, not clean."""
    if source.kind != "vae":
        return {"applicable": False,
                "note": "source.kind is hdf5: no pretrained encoder, so no pretraining exposure"}
    checkpoint = resolve_path(source.vae.checkpoint)
    dataset = load_config(str(resolved_config_for(checkpoint))).get("dataset_config", {})
    pre, sel = (_shard_guids(dataset[k]) if dataset.get(k) else None for k in ("vae_train_datasets", "vae_test_datasets"))
    frame, folds = guids.assign(guid=guids["guid_norm"]), {}
    for fold, g in frame.groupby("fold"):
        overlap = sorted(set(g["guid"][g["split"] == "test"]) & ((pre or set()) | (sel or set())))
        if overlap and not allow_overlap:
            logger.error(f"L3: fold {fold}: {len(overlap)} test GUID(s) seen by the VAE")
            raise ValueError(
                f"L3: fold {fold} test split shares {len(overlap)} GUID(s) with the VAE's "
                f"pretraining/selection shards (e.g. {overlap[:5]}). Set "
                f"data.allow_pretrain_overlap: true only for a deliberate exploratory run.")
        folds[str(fold)] = {**exposure_record(g, pretraining_guids=pre, selection_guids=sel),
                            "test_overlap": overlap}
    return {"applicable": True, "checkpoint": str(checkpoint),
            "allow_pretrain_overlap": allow_overlap, "folds": folds}


# ---- clinical covariates (§7.3) and the missingness confound (L10) ------------------------------
#: Raw covariate columns of the segment table: the joined value, and a timed variable's age in hours.
COV, COV_AGE = "cov:", "cov_age_h:"
AVAILABILITY_COLUMNS = KEYS + ["variable", "available"]


def _read_table(path: str, need: Sequence[str], **kw: Any) -> pd.DataFrame:
    """A covariate CSV with its ``guid`` replaced by ``guid_norm``; missing columns raise."""
    frame = pd.read_csv(resolve_path(path), **kw)
    missing = sorted(set(need) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    return frame.assign(guid_norm=frame.pop("guid").map(normalize_guid).astype(str))


def join_covariates(seg: pd.DataFrame, cov: CovariatesCfg) -> pd.DataFrame:
    """Raw covariates of every ``seg`` row (§7.3), indexed like ``seg``: ``cov:<name>`` (float for ``numeric``, str
    for ``categorical``; NaN = missing) and, for a timed variable, ``cov_age_h:<name>``.

    Each variable lives in exactly one table. ``static_csv``: ``guid`` plus one column per variable, one row per
    GUID, joined on ``guid_norm``. ``timed_csv``: long ``guid, time_s, variable, value``, ``time_s`` in seconds
    relative to delivery, joined strictly causally: the latest observation with ``time_s <= t_end_s`` and age
    ``t_end_s - time_s <= max_age_h``. ``time_s`` only aligns; it never leaves this function.
    """
    out = pd.DataFrame(index=seg.index)
    if not cov.variables:
        return out
    static = _read_table(cov.static_csv, ["guid"], dtype=str).drop_duplicates() if cov.static_csv else None
    if static is not None and static["guid_norm"].duplicated().any():
        raise ValueError(f"{cov.static_csv}: static covariates are one row per GUID; repeated: "
                         f"{sorted(set(static.loc[static['guid_norm'].duplicated(), 'guid_norm']))[:5]}")
    timed = (_read_table(cov.timed_csv, ["guid", "time_s", "variable", "value"],
                         dtype={"guid": str, "variable": str, "value": str}) if cov.timed_csv else None)
    left = pd.DataFrame({"guid_norm": seg["guid_norm"].astype(str).to_numpy(),
                         "t_end_s": seg["t_end_s"].to_numpy(float), "_i": np.arange(len(seg))}
                        ).sort_values("t_end_s", kind="stable")
    for v in cov.variables:
        found = [name for name, hit in (("static_csv", static is not None and v.name in static),
                                        ("timed_csv", timed is not None and (timed["variable"] == v.name).any()))
                 if hit]
        if len(found) != 1:
            raise ValueError(f"covariate {v.name!r} must be in exactly one covariate table; found in {found}")
        parse = pd.to_numeric if v.kind == "numeric" else (lambda x: x)  # read as str: categorical stays str
        if found == ["static_csv"]:
            out[COV + v.name] = seg["guid_norm"].map(parse(static.set_index("guid_norm")[v.name]))
            continue
        obs = timed.loc[timed["variable"] == v.name, ["guid_norm", "time_s", "value"]].dropna()
        obs = obs.assign(time_s=obs["time_s"].astype(float), value=parse(obs["value"])).sort_values(
            "time_s", kind="stable")
        hit = pd.merge_asof(left, obs, left_on="t_end_s", right_on="time_s", by="guid_norm", direction="backward",
                            tolerance=3600.0 * cov.max_age_h).sort_values("_i")
        out[COV + v.name] = hit["value"].set_axis(seg.index)
        out[COV_AGE + v.name] = ((hit["t_end_s"] - hit["time_s"]) / 3600.0).set_axis(seg.index)
    return out


def covariate_availability(seg: pd.DataFrame, guids: pd.DataFrame, cov: CovariatesCfg) -> pd.DataFrame:
    """:data:`AVAILABILITY_COLUMNS` per GUID of ``guids`` and covariate: ``available`` iff a retained segment of the
    GUID has a joined value (the ``covariate:<name>`` subgroups, §11.6.1, and the L10 check)."""
    kept, parts = seg[~seg["excluded"]], []
    for v in cov.variables:
        has = kept[COV + v.name].notna().groupby([kept[k] for k in KEYS]).any().rename("available").reset_index()
        parts.append(guids[KEYS].merge(has, on=KEYS, how="left").assign(variable=v.name))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=AVAILABILITY_COLUMNS)
    return out.assign(available=out["available"].eq(True))[AVAILABILITY_COLUMNS]


def confound_check(guids: pd.DataFrame, max_delta: float,
                   availability: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """§7.3 step 4 (L10): per fold, the train missing rate per task class of ``has_tlo`` and of every covariate
    (not ``available`` in ``availability``, :func:`covariate_availability`), keyed by name; a gap over
    ``max_delta`` warns and is ``flagged``."""
    out: Dict[str, Any] = {}
    train = guids[(guids["split"] == "train") & ~guids["excluded"]]
    missing = {"has_tlo": ~train["has_tlo"].astype(bool)}
    for name, a in (() if availability is None else availability.groupby("variable", sort=False)):
        missing[name] = ~train[KEYS].merge(a, on=KEYS, how="left")["available"].eq(True).set_axis(train.index)
    for fold, g in train.groupby("fold"):
        out[str(fold)] = {}
        for name, miss in missing.items():
            rates = miss.loc[g.index].groupby(g["y"].astype(int)).mean()
            delta = float(rates.max() - rates.min())
            if delta > max_delta:
                logger.warning(f"L10: fold {fold} {name} missing rate differs across classes by "
                               f"{delta:.3f} > {max_delta}: {rates.round(3).to_dict()}")
            out[str(fold)][name] = {"missing_rate_by_class": {str(k): float(v) for k, v in rates.items()},
                                    "delta": delta, "flagged": delta > max_delta}
    return out


def confound_flagged(confound: Optional[Mapping[str, Any]]) -> bool:
    """Any fold's ``has_tlo`` or covariate is flagged in a :func:`confound_check` record."""
    return any(rec.get("flagged") for per_fold in (confound or {}).values() for rec in per_fold.values())


def fold_summary(seg: pd.DataFrame, guids: pd.DataFrame) -> pd.DataFrame:
    """Long counts per fold x split x class x subgroup x level x status (retained or reason)."""
    def counts(frame: pd.DataFrame, level: str) -> pd.DataFrame:
        status = frame["exclusion_reason"].replace("", "retained")
        return (frame.assign(level=level, status=status)
                .groupby(["fold", "split", "class_code", "source_file", "level", "status"],
                         dropna=False).size().rename("n").reset_index())
    return pd.concat([counts(guids, "guid"), counts(seg, "segment")], ignore_index=True)


# ---- context columns (§7.1) --------------------------------------------------------------------
def psi(x: Any) -> Any:
    """``sign(x) * log1p(|x|)``."""
    return np.sign(x) * np.log1p(np.abs(x))


def context_features(seg: pd.DataFrame, *, tlo_pre_onset: str = "clip") -> pd.DataFrame:
    """The §7.1 allow-listed per-segment context columns; no forbidden clock ever enters, and first and
    unknown stage share the all-zero flags (:data:`CONTEXT_STAGES`). ``tlo_pre_onset: clip`` (``context.tlo``)
    floors TLO at 0: a segment ending before labour onset says nothing about how long until onset, which is
    future information like the time until second stage (§2.4); ``signed`` keeps it (ablation)."""
    tlo_h = seg["tlo_end_s"] / 3600.0
    if tlo_pre_onset == "clip":
        tlo_h = tlo_h.clip(lower=0.0)
    in_ss_h = np.maximum(seg["ss_rel_s"] + seg["t_end_s"] - seg["epoch_s"], 0.0) / 3600.0
    return pd.DataFrame({
        "tlo_psi": psi(tlo_h.fillna(0.0)), "tlo_missing": tlo_h.isna().astype(float),
        **{f"stage_{s}": (seg["stage"] == s).astype(float) for s in CONTEXT_STAGES},
        "time_in_ss": psi(in_ss_h.fillna(0.0)), "valid_frac": seg["valid_frac"],
    }, index=seg.index)


# ---- the stage ---------------------------------------------------------------------------------
def build_cohort(cfg: Classifier, out_dir: Any) -> Dict[str, Any]:
    """Build, validate and write ``cohort/`` for ``cfg.run.folds``; return the manifest record."""
    shards = {(k, s): p for k in cfg.run.folds for s, p in fold_shards(cfg.data, k).items()}
    seg, info = segment_table(shards, trim_minutes=cfg.source.hdf5.trim_minutes,
                              stride_s=cfg.data.stride_s, epoch_min_s=cfg.data.epoch_min_s,
                              min_valid_frac=cfg.data.min_valid_frac)
    seg, guids = guid_table(seg, cfg.labels, cfg.cohort)
    guids, checks = validate_folds(guids, task=cfg.labels.task, patient_map=cfg.data.patient_map)
    exposure = pretrain_exposure(guids, cfg.source, allow_overlap=cfg.data.allow_pretrain_overlap)
    cov = cfg.context.covariates
    joined = join_covariates(seg, cov)
    seg = pd.concat([seg, joined], axis=1)
    availability = covariate_availability(seg, guids, cov)
    confound = confound_check(guids, cfg.context.missing_confound_max, availability)
    tables = {k: resolve_path(p) for k in ("static_csv", "timed_csv") if cov.variables and (p := getattr(cov, k))}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg[SEGMENT_COLUMNS + list(joined)].to_parquet(out / "segments.parquet", index=False)
    guids[GUID_COLUMNS].to_parquet(out / "guids.parquet", index=False)
    availability.to_parquet(out / "covariate_availability.parquet", index=False)
    fold_summary(seg, guids).to_csv(out / "fold_summary.csv", index=False)
    for name, payload in (("exposure", exposure), ("confound", confound)):
        (out / f"{name}.json").write_text(json.dumps(json_safe(payload), indent=2))
    return {
        "shards": sorted({path for paths in shards.values() for path in paths}), **info, **checks,
        "counts": {
            "segments": int(len(seg)), "segments_retained": int((~seg["excluded"]).sum()),
            "guids": int(len(guids)), "guids_retained": int((~guids["excluded"]).sum()),
            "segment_exclusions": seg["exclusion_reason"][seg["excluded"]].value_counts().to_dict(),
            "guid_exclusions": guids["exclusion_reason"][guids["excluded"]].value_counts().to_dict(),
        },
        "covariates": {"columns": list(joined), "tables": {k: {"path": str(p), "sha256": hashlib.sha256(
            p.read_bytes()).hexdigest()} for k, p in tables.items()}},
        "exposure": exposure, "confound": confound,
    }
