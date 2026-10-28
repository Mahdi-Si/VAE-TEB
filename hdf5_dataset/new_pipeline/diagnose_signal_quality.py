r"""Collect the data that the next revision of the UA and FHR quality rules needs.

``quality_report.py`` gives one mean per file. This module reads the same files back, segment by
segment, and measures five things that the report cannot show:

1. **Flat UA with blips.** A constant UA run that short pulses interrupt: how long the merged flat
   region is, how much of it the rules keep valid, and the spacing and height of the pulses.
   Each constant run of such a region can stay below ``rest_max_s`` and below the builder's flat
   gate, while the region as a whole does not.
2. **Transients against delivery.** The ``TRANSIENT`` share per time to delivery and per delivery
   mode, and before and after the second-stage onset.
3. **FHR repair mask.** The share of segments that are mostly ``REPAIRED``, and for a sample of
   records a second match of the pre-fill FHR against every 4 Hz channel of the ``.mat`` file.
4. **Level changes.** For each ``STEP`` run, and each ``TRANSIENT`` run after which the level
   stays moved, the level before, just after and $60$--$120$ s after; and the low run that
   follows a drop to near zero.
5. **Dense transient regions.** Long, mostly invalid regions whose slow trend still has the range
   of a contraction.

It also lists the field names of a sample of ``.mat`` files, and the values of the fields whose
name suggests a device, so that a flat pattern can be tied to a monitor.

The files are the ones ``build_fold_datasets._run_quality_report`` reads: every pretraining shard,
the partition files of one fold, and the holdout test files. Each file contributes a random sample
of segments; a segment carries the weight $n_\text{file} / n_\text{sampled}$, and every share in
``SUMMARY.md`` is weighted.

Outputs in ``output_dir``:

* ``SUMMARY.md`` and ``summary.json``: the tables;
* ``segments.csv``: one row per sampled segment;
* ``blips.csv``: one row per pulse inside a flat region of at least :data:`BLIP_REGION_MIN_S`;
* ``level_events.csv``: one row per ``STEP`` run and per level-moving ``TRANSIENT`` run;
* ``fhr_raw_records.csv`` and ``fhr_raw_channels.csv``: the second FHR match;
* ``mat_metadata.json``: the ``.mat`` field names and the device-like values. **Read it before you
  share it**: a value can name a site;
* ``pages/``: example pages per new category, drawn by ``quality_report.plot_example``. They hold
  clinical traces.

Thresholds of this module are diagnostic only; nothing here changes a built file.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd

#: Repository root: ``hdf5_dataset/new_pipeline/diagnose_signal_quality.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the quality rules can be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:  # Repository layout, and a box with the teb_vae package.
    from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402
except ModuleNotFoundError as _exc:
    if not str(_exc.name).startswith("teb_vae"):
        raise
    from launch import missing_required, resolve_launch_args  # noqa: E402

try:  # Production: every file in one folder.
    import signal_quality as sq  # noqa: E402
except ModuleNotFoundError as _exc:
    if _exc.name != "signal_quality":
        raise
    from hdf5_dataset import signal_quality as sq  # noqa: E402

logger = logging.getLogger(__name__)

#: Raw samples per second.
FS = 4.0
#: Directory names of a build (as in ``build_fold_datasets.py``).
PRETRAIN_DIR = "pre_training_dataset"
KFOLD_DIR = "k_fold_cross_validation_dataset"
#: Segments read at once from a file.
BATCH_ROWS = 256

#: Shortest constant run that can belong to a flat region.
MIN_RUN_S = 10.0
#: Two constant runs at the same value with at most this between them form one flat region; the
#: samples between them are a pulse ("blip").
MERGE_GAP_S = 15.0
#: Flat-region lengths that the summary counts. $300$ s is ``rest_max_s`` and the builder's gate.
FLAT_THRESHOLDS_S = (300.0, 600.0)
#: Pulses are listed only inside flat regions at least this long.
BLIP_REGION_MIN_S = 120.0
#: The builder drops a segment whose longest exact run of pre-fill UA exceeds this many samples.
BUILDER_FLAT_SAMPLES = 1200

#: Level windows of a level event: before it, just after it, and late after it.
LEVEL_PRE_S = 5.0
LEVEL_POST_S = 20.0
LEVEL_LATE_S = (60.0, 120.0)
#: Runs longer than this are not level events.
LEVEL_EVENT_MAX_S = 60.0
#: A level change of at least this many UA units is counted (``step_units`` of the rules).
LEVEL_UNITS = 10.0
#: A UA value at most this is "near zero" after a drop.
LOW_UNITS = 5.0

#: ``TRANSIENT`` / ``ISLAND`` runs with at most this between them form one dense region.
DENSE_GAP_S = 10.0
#: Shortest dense region.
DENSE_MIN_S = 120.0
#: Width of the running median that gives the slow trend of a dense region.
SLOW_MEDIAN_S = 30.0
#: A slow-trend range of at least this suggests contractions under the noise.
SLOW_RANGE_UNITS = 20.0

#: UA values that are a device or export ceiling (``ceiling_values`` of the rules).
CEILING = (100.0, 127.0)
#: Bin edges of the time to delivery, in hours, at the segment centre.
HOURS_EDGES = (0.0, 1.0, 2.0, 4.0, 8.0, math.inf)

#: Raw-channel offsets searched for the second FHR match, in samples.
RAW_OFFSETS = tuple(range(-8, 9))
#: ``.mat`` fields whose value is listed when their name matches. ``name`` is left out on purpose.
MAT_VALUE_PATTERN = re.compile(
    r"device|monitor|model|manufact|vendor|source|site|hosp|serial|version|type|unit", re.I)
#: Depth of the ``.mat`` field walk, and the most field names listed per record.
MAT_MAX_DEPTH = 3
MAT_MAX_FIELDS = 400


def _sibling(name: str) -> Any:
    """Import a module of this folder: flat first (production), then from the repository package.

    Args:
        name: Module name, e.g. ``'quality_report'``.

    Returns:
        The module.
    """
    import importlib

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
    return importlib.import_module(f"hdf5_dataset.new_pipeline.{name}")


def _n(seconds: float) -> int:
    """Convert seconds to a whole number of raw samples, at least 1."""
    return max(int(round(seconds * FS)), 1)


# =============================================================================
# Files
# =============================================================================
@dataclass(frozen=True)
class FileTask:
    """One HDF5 file of the build and how to sample it.

    Attributes:
        path: The file.
        label: Its path below the build directory.
        subgroup: The subgroup of every segment in it.
        split: ``pretrain_train``, ``pretrain_test``, ``train``, ``val``, ``test`` or
            ``holdout_test``.
        rows_per_file: Segments to sample; 0 reads every segment.
        seed: Base seed of the sample.
    """

    path: str
    label: str
    subgroup: str
    split: str
    rows_per_file: int
    seed: int


def plan_files(build_dir: Path, fold: int, rows_per_file: int, seed: int) -> List[FileTask]:
    """List the files that hold each segment of the build once.

    Every pool GUID sits in exactly one partition of a fold, so the files of one fold, the
    pretraining shards and the holdout test files cover the build without repeats.

    Args:
        build_dir: The ``output_base_path`` of ``build_fold_datasets.py``.
        fold: The fold whose partition files are read.
        rows_per_file: Segments to sample per file; 0 reads every segment.
        seed: Base seed of the samples.

    Returns:
        One task per file, sorted by label.
    """
    tasks = []
    for path in sorted((build_dir / PRETRAIN_DIR).glob("*/*/shard_*.hdf5")):
        tasks.append((path, path.parent.name, f"pretrain_{path.parent.parent.name}"))
    for path in sorted((build_dir / KFOLD_DIR / f"fold_{fold}").glob("*/*.hdf5")):
        tasks.append((path, path.stem, path.parent.name))
    for path in sorted((build_dir / KFOLD_DIR / "test").glob("*.hdf5")):
        tasks.append((path, path.stem, "holdout_test"))
    return [FileTask(str(p), p.relative_to(build_dir).as_posix(), sg, split, rows_per_file, seed)
            for p, sg, split in tasks]


def _bits(dataset: Any) -> Dict[str, int]:
    """Bit name (upper case) -> value, from the ``bits`` attribute of a quality dataset."""
    return {name.upper(): int(value) for name, value in json.loads(dataset.attrs["bits"]).items()}


def _cs_from_subgroup(subgroup: str) -> float:
    """Return 1 for a ``*_cs`` subgroup, 0 for a ``*_no_cs`` one, NaN otherwise."""
    if subgroup.endswith("_no_cs"):
        return 0.0
    return 1.0 if subgroup.endswith("_cs") else math.nan


# =============================================================================
# One segment
# =============================================================================
def _median(x: np.ndarray, ok: np.ndarray, start: int, stop: int) -> Optional[float]:
    """Median of ``x[start:stop]`` over the samples where ``ok``; ``None`` if there is none."""
    start, stop = max(0, start), min(x.size, stop)
    if stop <= start:
        return None
    values = x[start:stop][ok[start:stop]]
    return float(np.median(values)) if values.size else None


def merge_runs(runs: Sequence[Tuple[int, int]], gap: int) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    """Merge runs whose distance is at most *gap* samples.

    Args:
        runs: ``[(start, stop), ...]`` in order, ``stop`` exclusive.
        gap: Largest distance between two runs of one region, in samples.

    Returns:
        ``[(start, stop, gaps), ...]``, where *gaps* lists the ``(start, stop)`` of each space
        between two merged runs.
    """
    out: List[Tuple[int, int, List[Tuple[int, int]]]] = []
    for start, stop in runs:
        if out and start - out[-1][1] <= gap:
            first, last, gaps = out[-1]
            gaps.append((last, start))
            out[-1] = (first, stop, gaps)
        else:
            out.append((start, stop, []))
    return out


def flat_regions(x: np.ndarray, usable: np.ndarray) -> List[Dict[str, Any]]:
    r"""Find the flat regions of one UA segment.

    A flat region is a sequence of exactly constant runs of at least :data:`MIN_RUN_S`, all at
    the same value, where each run starts at most :data:`MERGE_GAP_S` after the one before. For
    example, UA at exactly 20 for $22$ min with a $2$ s pulse every $302$ s is one region of
    $1320$ s with 4 gaps, although no single run reaches $300$ s.

    Args:
        x: Pre-fill UA.
        usable: Samples that are present and not a code.

    Returns:
        One dict per region: ``start``, ``stop``, ``value``, ``gaps`` and ``const`` (the
        samples inside its constant runs).
    """
    regions: List[Dict[str, Any]] = []
    gap = _n(MERGE_GAP_S)
    for start, stop in sq.constant_runs(x, usable, _n(MIN_RUN_S)):
        value = float(x[start])
        last = regions[-1] if regions else None
        if last is not None and last["value"] == value and start - last["stop"] <= gap:
            last["gaps"].append((last["stop"], start))
            last["stop"] = stop
            last["const"] += stop - start
        else:
            regions.append({"start": start, "stop": stop, "value": value, "gaps": [],
                            "const": stop - start})
    return regions


def analyse_segment(
    up: np.ndarray, uq: np.ndarray, fq: np.ndarray, tss: float, ua_bits: Mapping[str, int],
    ua_invalid: int, fhr_bits: Mapping[str, int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Measure one segment.

    Args:
        up: Pre-fill UA (``up_raw``), ``(N,)``.
        uq: UA quality bits, ``(N,)``.
        fq: FHR quality bits, ``(N,)``.
        tss: ``second_stage_onset`` of the segment: its start minus the second-stage onset, in
            seconds; NaN when the onset is unknown.
        ua_bits: UA bit name -> value.
        ua_invalid: The UA bits that make a sample invalid.
        fhr_bits: FHR bit name -> value.

    Returns:
        ``(metrics, blips, levels)``: one dict of segment metrics, and the rows of
        ``blips.csv`` and ``level_events.csv``.
    """
    n = up.size
    has = {name: (uq & value) != 0 for name, value in ua_bits.items()}
    present = ~has["MISSING"]
    usable = present & ~has["CODE"]
    invalid = (uq & ua_invalid) != 0
    clean = usable & ~invalid
    transient, step = has["TRANSIENT"], has["STEP"]
    out: Dict[str, Any] = {f"ua_{name.lower()}": float(mask.mean()) for name, mask in has.items()}
    out["ua_invalid"] = float(invalid.mean())
    ceiling = np.isin(up, CEILING)
    out["censored_ceiling"] = float(np.mean(has["CENSORED"] & ceiling))
    out["censored_rest"] = float(np.mean(has["CENSORED"] & ~ceiling))

    # 2. Transients before and after the second-stage onset.
    transient_runs = sq.runs(transient)
    out["n_transient_runs"] = len(transient_runs)
    out["stage_known"] = bool(np.isfinite(tss))
    if out["stage_known"]:
        stage2 = (tss + np.arange(n) / FS) >= 0.0
        for label, mask in (("s1", ~stage2), ("s2", stage2)):
            out[f"n_{label}"] = int(mask.sum())
            out[f"transient_n_{label}"] = int(np.count_nonzero(transient & mask))
            out[f"invalid_n_{label}"] = int(np.count_nonzero(invalid & mask))
            out[f"transient_runs_{label}"] = sum(1 for start, _ in transient_runs if mask[start])
    else:
        for label in ("s1", "s2"):
            for key in ("n", "transient_n", "invalid_n", "transient_runs"):
                out[f"{key}_{label}"] = math.nan

    # 1. Flat regions and their pulses.
    single = sq.constant_runs(up, usable, _n(MIN_RUN_S))
    out["longest_const_run_s"] = max((b - a for a, b in single), default=0) / FS
    exact = sq.constant_runs(up, np.ones(n, dtype=bool), 2)
    out["longest_exact_run_all_s"] = max((b - a for a, b in exact), default=0) / FS
    regions = flat_regions(up, usable)
    out["flat_total_300_s"] = sum(r["stop"] - r["start"] for r in regions
                                  if r["stop"] - r["start"] >= _n(FLAT_THRESHOLDS_S[0])) / FS
    blips: List[Dict[str, Any]] = []
    best = max(regions, key=lambda r: r["stop"] - r["start"], default=None)
    if best is None:
        out.update(flat_region_s=0.0, flat_value=math.nan, flat_gaps=0, flat_const_share=math.nan,
                   flat_valid_share=math.nan, flat_censored_share=math.nan,
                   flat_hold_share=math.nan)
    else:
        span = slice(best["start"], best["stop"])
        out.update(flat_region_s=(best["stop"] - best["start"]) / FS, flat_value=best["value"],
                   flat_gaps=len(best["gaps"]),
                   flat_const_share=best["const"] / (best["stop"] - best["start"]),
                   flat_valid_share=float(np.mean(~invalid[span])),
                   flat_censored_share=float(np.mean(has["CENSORED"][span])),
                   flat_hold_share=float(np.mean(has["HOLD"][span])))
    for region in regions:
        length = region["stop"] - region["start"]
        if length < _n(BLIP_REGION_MIN_S):
            continue
        previous = None
        for a, b in region["gaps"]:
            values = up[a:b][usable[a:b]]
            blips.append({
                "region_value": region["value"], "region_s": length / FS, "t_s": a / FS,
                "length_s": (b - a) / FS,
                "height": float(np.max(np.abs(values - region["value"]))) if values.size
                else math.nan,
                "interval_s": math.nan if previous is None else (a - previous) / FS,
                "transient": bool(transient[a:b].any()), "invalid": bool(invalid[a:b].any()),
                "missing": bool((~present[a:b]).any()),
                "region_valid_share": float(np.mean(~invalid[region["start"]:region["stop"]])),
            })
            previous = a

    # 4. Level events.
    levels: List[Dict[str, Any]] = []
    out.update(n_step=0, n_level_change=0, max_abs_delta_late=0.0, n_down_to_low=0)
    low = (up <= LOW_UNITS) & usable
    for kind, mask in (("step", step), ("transient", transient & ~step)):
        for a, b in sq.runs(mask):
            if b - a > _n(LEVEL_EVENT_MAX_S):
                continue
            if kind == "step":
                out["n_step"] += 1
            # The levels are those the transform reads, valid or not; the valid shares say how
            # much of each window the mask already covers.
            windows = {"pre": (a - _n(LEVEL_PRE_S), a), "post": (b, b + _n(LEVEL_POST_S)),
                       "late": (b + _n(LEVEL_LATE_S[0]), b + _n(LEVEL_LATE_S[1]))}
            level = {name: _median(up, usable, lo_, hi_) for name, (lo_, hi_) in windows.items()}
            pre, post, late = level["pre"], level["post"], level["late"]
            if pre is None or post is None:
                continue
            valid_share = {f"{name}_valid_share": float(np.mean(clean[max(0, lo_):hi_]))
                           if min(hi_, n) > max(0, lo_) else math.nan
                           for name, (lo_, hi_) in windows.items()}
            delta_late = math.nan if late is None else late - pre
            moved = np.isfinite(delta_late) and abs(delta_late) >= LEVEL_UNITS
            low_s, low_valid = math.nan, math.nan
            if post <= LOW_UNITS and post - pre <= -LEVEL_UNITS:
                j = b
                while j < n and low[j]:
                    j += 1
                low_s = (j - b) / FS
                low_valid = float(np.mean(~invalid[b:j])) if j > b else math.nan
                out["n_down_to_low"] += 1
            if moved:
                out["n_level_change"] += 1
                out["max_abs_delta_late"] = max(out["max_abs_delta_late"], abs(delta_late))
            if kind == "step" or moved:
                levels.append({"kind": kind, "t_s": a / FS, "length_s": (b - a) / FS, "pre": pre,
                               "post": post, "late": late, "delta_post": post - pre,
                               "delta_late": delta_late, **valid_share, "low_after_s": low_s,
                               "low_after_valid_share": low_valid})

    # 5. Dense transient regions.
    noisy = transient | has["ISLAND"]
    dense = [(a, b) for a, b, _ in merge_runs(sq.runs(noisy), _n(DENSE_GAP_S))
             if b - a >= _n(DENSE_MIN_S)]
    out["dense_transient_s"] = sum(b - a for a, b in dense) / FS
    out["dense_slow_range"] = 0.0
    if dense:
        from scipy.ndimage import median_filter

        for a, b in dense:
            values = up[a:b][usable[a:b]]
            if values.size >= _n(SLOW_MEDIAN_S):
                slow = median_filter(values, size=_n(SLOW_MEDIAN_S), mode="nearest")
                spread = float(np.subtract(*np.percentile(slow, [95, 5])))
                out["dense_slow_range"] = max(out["dense_slow_range"], spread)

    # 3. FHR repair share.
    repaired = (fq & fhr_bits["REPAIRED"]) != 0
    out["fhr_repaired"] = float(repaired.mean())
    out["fhr_longest_repaired_s"] = max((b - a for a, b in sq.runs(repaired)), default=0) / FS
    return out, blips, levels


# =============================================================================
# Pass 1: every sampled segment
# =============================================================================
def analyse_file(task: FileTask) -> Dict[str, Any]:
    """Measure the sampled segments of one file.

    Args:
        task: The file.

    Returns:
        ``{"segments", "blips", "levels"}`` as data frames, and ``"info"`` about the file.
    """
    start = time.time()
    rng = np.random.default_rng([task.seed, zlib.crc32(task.label.encode())])
    segments, blips, levels = [], [], []
    with h5py.File(task.path, "r") as f:
        info = {"file": task.label, "subgroup": task.subgroup, "split": task.split,
                "segments": int(f["fhr"].shape[0]),
                "quality_version": int(f.attrs.get("quality_version", -1))}
        if "up_quality" not in f or info["segments"] == 0:
            info["sampled"] = 0
            return {"segments": pd.DataFrame(), "blips": pd.DataFrame(),
                    "levels": pd.DataFrame(), "info": info}
        n = info["segments"]
        take = n if task.rows_per_file <= 0 else min(n, task.rows_per_file)
        rows = np.arange(n) if take == n else np.sort(rng.choice(n, size=take, replace=False))
        info["sampled"] = int(rows.size)
        weight = n / rows.size
        ua_bits, fhr_bits = _bits(f["up_quality"]), _bits(f["fhr_quality"])
        ua_invalid = int(f["up_quality"].attrs["invalid_bits"])
        cs_default = _cs_from_subgroup(task.subgroup)
        for a in range(0, rows.size, BATCH_ROWS):
            batch = rows[a:a + BATCH_ROWS]
            up = f["up_raw"][batch].astype(np.float64)
            uq = f["up_quality"][batch].astype(np.int64)
            fq = f["fhr_quality"][batch].astype(np.int64)
            guid = f["guid"].asstr()[batch]
            epoch = f["epoch"][batch].astype(np.float64)
            tss = (f["second_stage_onset"][batch].astype(np.float64)
                   if "second_stage_onset" in f else np.full(batch.size, np.nan))
            cs = (f["cs_label"][batch].astype(np.float64)
                  if "cs_label" in f else np.full(batch.size, cs_default))
            up_valid = f["up_valid_frac"][batch]
            fhr_valid = f["fhr_valid_frac"][batch]
            for i, row in enumerate(batch):
                metrics, seg_blips, seg_levels = analyse_segment(
                    up[i], uq[i], fq[i], float(tss[i]), ua_bits, ua_invalid, fhr_bits)
                key = {"file": task.label, "row": int(row), "guid": str(guid[i])}
                segments.append({
                    **key, "path": task.path, "subgroup": task.subgroup, "split": task.split,
                    "cs": float(cs[i]), "epoch": float(epoch[i]),
                    "hours_to_delivery": -(float(epoch[i]) + up.shape[1] / FS / 2.0) / 3600.0,
                    "second_stage_onset": float(tss[i]), "weight": weight,
                    "up_valid_frac": float(up_valid[i]), "fhr_valid_frac": float(fhr_valid[i]),
                    **metrics})
                blips.extend({**key, "subgroup": task.subgroup, **b} for b in seg_blips)
                levels.extend({**key, "subgroup": task.subgroup, **e} for e in seg_levels)
    info["seconds"] = round(time.time() - start, 1)
    return {"segments": pd.DataFrame(segments), "blips": pd.DataFrame(blips),
            "levels": pd.DataFrame(levels), "info": info}


def run_files(tasks: Sequence[FileTask], workers: int) -> Dict[str, Any]:
    """Run :func:`analyse_file` over every file, in *workers* processes.

    Returns:
        The concatenated frames and the list of file infos.
    """
    results = []
    if workers <= 1:
        for i, task in enumerate(tasks, 1):
            results.append(analyse_file(task))
            logger.info(f"[{i}/{len(tasks)}] {task.label}")
    else:
        import multiprocessing as mp

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(analyse_file, task): task for task in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                logger.info(f"[{i}/{len(tasks)}] {futures[future].label}")
    frames = {key: [r[key] for r in results if len(r[key])] for key in ("segments", "blips",
                                                                         "levels")}
    out = {key: pd.concat(value, ignore_index=True) if value else pd.DataFrame()
           for key, value in frames.items()}
    out["files"] = sorted((r["info"] for r in results), key=lambda i: i["file"])
    return out


# =============================================================================
# Records
# =============================================================================
def record_paths(build_dir: Path) -> Dict[str, str]:
    """Map every GUID of the build to its ``.mat`` path.

    Reads ``pretraining_guids.csv`` and every path in ``classification_dataset_records.pickle``.

    Returns:
        ``{guid: path}``, keyed by the file stem and by its upper-case form.
    """
    paths: List[str] = []
    csv = build_dir / "pretraining_guids.csv"
    if csv.exists():
        paths += pd.read_csv(csv, dtype={"record_path": str})["record_path"].tolist()
    pkl = build_dir / "classification_dataset_records.pickle"
    if pkl.exists():
        with open(pkl, "rb") as fh:
            stack = [pickle.load(fh)]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
            elif isinstance(item, str) and item.lower().endswith(".mat"):
                paths.append(item)
    out: Dict[str, str] = {}
    for path in paths:
        stem = Path(str(path).replace("\\", "/")).stem
        out[stem] = str(path)
        out[stem.upper()] = str(path)
    return out


def _pick(frame: pd.DataFrame, flag: pd.Series, score: pd.Series, n: int,
          rng: np.random.Generator) -> pd.DataFrame:
    """Pick up to *n* rows: half the highest *score* where *flag*, half at random elsewhere.

    Returns:
        The rows, with a column ``group`` (``'flagged'`` or ``'control'``).
    """
    flagged = frame[flag].assign(_score=score[flag]).sort_values("_score", ascending=False)
    top = flagged.head((n + 1) // 2).drop(columns="_score").assign(group="flagged")
    rest = frame[~flag]
    control = rest.iloc[rng.permutation(len(rest))[: n - len(top)]].assign(group="control")
    return pd.concat([top, control], ignore_index=True)


def _guid_rows(path: str, guid: str) -> np.ndarray:
    """Row indices of one GUID in one file."""
    with h5py.File(path, "r") as f:
        return np.flatnonzero(f["guid"].asstr()[:] == guid)


# =============================================================================
# Pass 2: the raw FHR channels
# =============================================================================
def _grids(rep_origin: int, rep_values: np.ndarray, epoch: np.ndarray, fhr: np.ndarray,
           present: np.ndarray) -> List[np.ndarray]:
    """Place each segment on the repaired channel's grid, as ``sq.raw_fhr_repaired`` does."""
    grids = []
    for i in range(fhr.shape[0]):
        base = int(np.floor(epoch[i] / 0.25)) + np.arange(fhr.shape[1])
        hits = [int(np.sum(sq._lookup(rep_origin, rep_values, base + o)[present[i]]
                           == fhr[i][present[i]])) for o in range(-4, 5)]
        grids.append(base + int(np.argmax(hits)) - 4)
    return grids


def check_raw_fhr(path: str, label: str, guid: str, record: str,
                  group: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Match the pre-fill FHR of one record against every 4 Hz channel of its ``.mat`` file.

    For the channel the rules read (``Repaired ...`` without ``Repaired``) the result shows why
    a sample is ``REPAIRED``: the raw value is absent, 0, 255 or more, or a different number.
    For every other channel it shows whether that channel matches better, which would point to
    a repaired channel built from two sources.

    Args:
        path: The HDF5 file.
        label: Its label.
        guid: The record's GUID.
        record: The ``.mat`` path.
        group: ``'flagged'`` or ``'control'``.

    Returns:
        ``(record_row, channel_rows)``.
    """
    from scipy.io import loadmat

    rows = _guid_rows(path, guid)
    with h5py.File(path, "r") as f:
        fhr_bits = _bits(f["fhr_quality"])
        epoch = f["epoch"][rows].astype(np.float64)
        fhr = f["fhr_raw"][rows].astype(np.float64)
        fq = f["fhr_quality"][rows].astype(np.int64)
    present = (fq & fhr_bits["MISSING"]) == 0
    stored = (fq & fhr_bits["REPAIRED"]) != 0
    out: Dict[str, Any] = {"file": label, "guid": guid, "group": group, "record": record,
                           "n_segments": int(rows.size),
                           "stored_repaired_share": float(stored[present].mean())}
    channels: List[Dict[str, Any]] = []
    try:
        repaired, info = sq.raw_fhr_repaired(record, epoch, fhr, present)
        out.update({f"rules_{key}": value for key, value in info.items()})
        if repaired is not None:
            out["recomputed_repaired_share"] = float(repaired[present].mean())
            out["agreement_with_stored"] = float(np.mean(repaired[present] == stored[present]))
        mat = loadmat(record, squeeze_me=True, struct_as_record=False)["efmStruct"]
        signals = {str(sig.measure): sig for sig in np.atleast_1d(mat.signals)}
        chosen = [k for k in ("Repaired HR1", "Repaired HR2", "Repaired external FECG")
                  if k in signals]
        if not chosen:
            out["error"] = "no repaired channel"
            return out, channels
        rep_origin, rep_values = sq._series(signals[chosen[-1]].segdat)
        grids = _grids(rep_origin, rep_values, epoch, fhr, present)
        n_present = max(int(present.sum()), 1)
        for measure, sig in signals.items():
            row: Dict[str, Any] = {"guid": guid, "group": group, "measure": measure,
                                   "rules_raw_channel": measure == chosen[-1].replace(
                                       "Repaired ", ""),
                                   "repaired_source": measure == chosen[-1]}
            try:
                row["domain_incr"] = float(sig.segdat.domainIncr)
                if abs(row["domain_incr"] - 0.25) > 1e-9:
                    channels.append(row)
                    continue
                origin, values = sq._series(sig.segdat)
            except Exception as exc:  # noqa: BLE001 -- a channel of another layout
                row["error"] = repr(exc)[:120]
                channels.append(row)
                continue
            best, best_hits = 0, -1
            for offset in RAW_OFFSETS:
                hits = sum(int(np.sum(sq._lookup(origin, values, grids[i] + offset)[present[i]]
                                      == fhr[i][present[i]])) for i in range(rows.size))
                if hits > best_hits:
                    best, best_hits = offset, hits
            raw = np.stack([sq._lookup(origin, values, g + best) for g in grids])[present]
            pre = fhr[present]
            absent, zero, code = raw < 0, raw == 0, raw >= 255
            differ = ~absent & ~zero & ~code & (raw != pre)
            diff = np.abs(raw - pre)[differ]
            row.update(best_offset=best, match=best_hits / n_present,
                       absent=float(absent.mean()), zero=float(zero.mean()),
                       code_255=float(code.mean()), differ=float(differ.mean()),
                       differ_median_bpm=float(np.median(diff)) if diff.size else math.nan,
                       differ_within_1bpm=float(np.mean(diff <= 1.0)) if diff.size else math.nan)
            channels.append(row)
    except Exception as exc:  # noqa: BLE001 -- one record must not cost the others
        out["error"] = repr(exc)[:200]
    return out, channels


# =============================================================================
# Pass 3: .mat metadata
# =============================================================================
def _short(value: Any) -> Optional[str]:
    """A short text form of a scalar or a string, or ``None`` for anything larger."""
    if isinstance(value, (str, bytes)):
        text = value.decode(errors="replace") if isinstance(value, bytes) else value
        return text[:80]
    if np.isscalar(value):
        return str(value)[:80]
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype != object:
        return str(value.item())[:80]
    return None


def mat_metadata(record: str) -> Dict[str, Any]:
    """List the field names of one ``.mat`` file and the values of device-like fields.

    Only the values of fields whose name matches :data:`MAT_VALUE_PATTERN` are kept, and only
    when they are a scalar or a short string. The channels are listed by name, sampling step and
    sample count.

    Args:
        record: The ``.mat`` path.

    Returns:
        ``{"fields", "values", "signals"}``, or ``{"error"}``.
    """
    from scipy.io import loadmat

    try:
        mat = loadmat(record, squeeze_me=True, struct_as_record=False)["efmStruct"]
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)[:200]}
    fields: List[str] = []
    values: Dict[str, str] = {}

    def walk(obj: Any, prefix: str, depth: int) -> None:
        names = getattr(obj, "_fieldnames", None)
        if names is None or len(fields) >= MAT_MAX_FIELDS:
            return
        for name in names:
            value = getattr(obj, name)
            key = f"{prefix}.{name}" if prefix else name
            fields.append(key)
            text = _short(value)
            if text is not None and MAT_VALUE_PATTERN.search(name):
                values[key] = text
            if depth < MAT_MAX_DEPTH:
                child = value
                if isinstance(value, np.ndarray) and value.dtype == object and value.size:
                    child = value.flat[0]
                walk(child, key, depth + 1)

    walk(mat, "", 0)
    signals = []
    for sig in np.atleast_1d(getattr(mat, "signals", [])):
        try:
            signals.append({"measure": str(sig.measure),
                            "domain_incr": float(sig.segdat.domainIncr),
                            "n_samples": int(np.asarray(sig.segdat.dataSet).size)})
        except Exception:  # noqa: BLE001
            signals.append({"measure": str(getattr(sig, "measure", "?"))})
    return {"fields": fields, "values": values, "signals": signals}


# =============================================================================
# Summary
# =============================================================================
def _wmean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean over the finite values; NaN when there is none."""
    v, w = values.to_numpy(dtype=float), weights.to_numpy(dtype=float)
    ok = np.isfinite(v)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() and w[ok].sum() > 0 \
        else math.nan


def _wshare(condition: pd.Series, weights: pd.Series) -> float:
    """Weighted share of the rows where *condition* holds."""
    total = float(weights.sum())
    return float(weights[condition].sum() / total) if total > 0 else math.nan


def _wquantiles(values: pd.Series, weights: pd.Series, qs: Sequence[float]) -> List[float]:
    """Weighted quantiles over the finite values."""
    v, w = values.to_numpy(dtype=float), weights.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if not ok.any():
        return [math.nan] * len(qs)
    order = np.argsort(v[ok])
    v, w = v[ok][order], w[ok][order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return [float(np.interp(q, cum, v)) for q in qs]


def _by_subgroup(segments: pd.DataFrame, fn: Any) -> pd.DataFrame:
    """Apply *fn* (a frame -> dict) per subgroup and to all segments; one row each."""
    rows = [{"subgroup": sg, **fn(g)} for sg, g in segments.groupby("subgroup", sort=True)]
    rows.append({"subgroup": "ALL", **fn(segments)})
    return pd.DataFrame(rows)


def summarise(seg: pd.DataFrame, blips: pd.DataFrame, levels: pd.DataFrame,
              fhr_records: pd.DataFrame, fhr_channels: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build every table of ``SUMMARY.md``.

    Args:
        seg: ``segments.csv``.
        blips: ``blips.csv``.
        levels: ``level_events.csv``.
        fhr_records: ``fhr_raw_records.csv``.
        fhr_channels: ``fhr_raw_channels.csv``.

    Returns:
        Table name -> frame, in the order of the summary.
    """
    t: Dict[str, pd.DataFrame] = {}
    lo, hi = FLAT_THRESHOLDS_S

    def flat(g: pd.DataFrame) -> Dict[str, Any]:
        w, long = g["weight"], g["flat_region_s"] >= lo
        return {
            "sampled": len(g), "segments": round(float(w.sum())),
            f"flat_ge_{lo:.0f}s": _wshare(long, w),
            f"flat_ge_{hi:.0f}s": _wshare(g["flat_region_s"] >= hi, w),
            f"flat_ge_{lo:.0f}s_with_blips": _wshare(long & (g["flat_gaps"] >= 1), w),
            f"flat_ge_{lo:.0f}s_mostly_valid": _wshare(long & (g["flat_valid_share"] >= 0.5), w),
            "valid_share_in_long_flat": _wmean(g.loc[long, "flat_valid_share"], w[long]),
            "censored_share_in_long_flat": _wmean(g.loc[long, "flat_censored_share"], w[long]),
            "guids_with_long_flat": int(g.loc[long, "guid"].nunique()),
            "censored_ceiling": _wmean(g["censored_ceiling"], w),
            "censored_rest": _wmean(g["censored_rest"], w),
            "max_longest_exact_run_s": float(g["longest_exact_run_all_s"].max()),
        }

    t["1a_flat_regions"] = _by_subgroup(seg, flat)
    long = seg[seg["flat_region_s"] >= lo]
    if len(long):
        values = long.groupby("flat_value")["weight"].sum().sort_values(ascending=False)
        t["1b_flat_values"] = pd.DataFrame({"flat_value": values.index[:12],
                                            "segments": values.to_numpy()[:12].round(),
                                            "share": (values / values.sum()).to_numpy()[:12]})
    if len(blips):
        spaced = blips.dropna(subset=["interval_s"])
        hist = (2.0 * np.round(spaced["interval_s"] / 2.0)).value_counts().head(15)
        t["1c_blip_intervals"] = pd.DataFrame({"interval_s": hist.index, "blips": hist.to_numpy(),
                                               "share": hist.to_numpy() / max(len(spaced), 1)})
        q = (0.1, 0.5, 0.9)
        t["1d_blip_shape"] = pd.DataFrame([{
            "blips": len(blips), "guids": blips["guid"].nunique(),
            **{f"height_p{int(100 * p)}": v for p, v in zip(q, blips["height"].quantile(q))},
            **{f"length_s_p{int(100 * p)}": v for p, v in zip(q, blips["length_s"].quantile(q))},
            "marked_transient": float(blips["transient"].mean()),
            "marked_invalid": float(blips["invalid"].mean()),
            "with_missing": float(blips["missing"].mean()),
        }])

    def by_hours(g: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        bins = pd.cut(g["hours_to_delivery"], HOURS_EDGES, right=False)
        for (interval, cs), part in g.groupby([bins, "cs"], observed=True):
            name = f"{interval.left:g}-{interval.right:g}h_cs{int(cs)}"
            out[f"{name}_n"] = len(part)
            out[f"{name}_transient"] = _wmean(part["ua_transient"], part["weight"])
        return out

    hours_rows = []
    bins = pd.cut(seg["hours_to_delivery"], HOURS_EDGES, right=False)
    for (interval, cs), part in seg.groupby([bins, "cs"], observed=True):
        hours_rows.append({"hours_to_delivery": f"{interval.left:g}-{interval.right:g}",
                           "cs": int(cs), "sampled": len(part),
                           "ua_transient": _wmean(part["ua_transient"], part["weight"]),
                           "ua_invalid": _wmean(part["ua_invalid"], part["weight"]),
                           "ua_oscillation": _wmean(part["ua_oscillation"], part["weight"])})
    t["2a_transient_by_hours"] = pd.DataFrame(hours_rows)
    t["2b_transient_by_hours_subgroup"] = _by_subgroup(seg, by_hours)

    def stage(g: pd.DataFrame) -> Dict[str, Any]:
        w, known = g["weight"], g["stage_known"].astype(bool)
        k = g[known]
        out = {"sampled": len(g), "stage_known": _wshare(known, w),
               "with_stage2_samples": _wshare(known & (g["n_s2"] > 0), w)}
        for label in ("s1", "s2"):
            samples = float((k[f"n_{label}"] * k["weight"]).sum())
            hours = samples / FS / 3600.0
            out[f"transient_{label}"] = float((k[f"transient_n_{label}"] * k["weight"]).sum()
                                              / samples) if samples else math.nan
            out[f"invalid_{label}"] = float((k[f"invalid_n_{label}"] * k["weight"]).sum()
                                            / samples) if samples else math.nan
            out[f"transient_runs_per_h_{label}"] = float(
                (k[f"transient_runs_{label}"] * k["weight"]).sum() / hours) if hours else math.nan
        return out

    t["2c_stage_split"] = _by_subgroup(seg, stage)
    tss = seg["second_stage_onset"]
    t["2d_second_stage_sanity"] = pd.DataFrame([{
        "cs": int(cs), "sampled": len(g), "onset_known": float(np.isfinite(g["second_stage_onset"]).mean()),
        **{f"second_stage_onset_s_p{int(100 * p)}": v
           for p, v in zip((0.01, 0.5, 0.99), g["second_stage_onset"].quantile([0.01, 0.5, 0.99]))},
    } for cs, g in seg.assign(second_stage_onset=tss).groupby("cs")])

    def fhr(g: pd.DataFrame) -> Dict[str, Any]:
        w = g["weight"]
        p1, p5, p50 = _wquantiles(g["fhr_valid_frac"], w, (0.01, 0.05, 0.5))
        return {"sampled": len(g), "repaired_gt_0.5": _wshare(g["fhr_repaired"] > 0.5, w),
                "repaired_gt_0.9": _wshare(g["fhr_repaired"] > 0.9, w),
                "guids_repaired_gt_0.9": int(g.loc[g["fhr_repaired"] > 0.9, "guid"].nunique()),
                "fhr_valid_p1": p1, "fhr_valid_p5": p5, "fhr_valid_p50": p50}

    t["3a_fhr_repaired"] = _by_subgroup(seg, fhr)
    if len(fhr_records):
        keep = [c for c in ("group", "guid", "n_segments", "stored_repaired_share",
                            "recomputed_repaired_share", "agreement_with_stored", "rules_raw_fhr",
                            "rules_raw_offset", "rules_repaired_grid_match", "rules_raw_match",
                            "error") if c in fhr_records.columns]
        t["3b_fhr_raw_records"] = fhr_records[keep]
    if len(fhr_channels) and "match" in fhr_channels.columns:
        t["3c_fhr_raw_channels"] = (
            fhr_channels.dropna(subset=["match"]).groupby(["group", "measure"])
            .agg(records=("guid", "nunique"), match_median=("match", "median"),
                 match_max=("match", "max"), absent=("absent", "median"),
                 zero=("zero", "median"), code_255=("code_255", "median"),
                 differ=("differ", "median"), differ_median_bpm=("differ_median_bpm", "median"),
                 differ_within_1bpm=("differ_within_1bpm", "median"))
            .reset_index())

    def level(g: pd.DataFrame) -> Dict[str, Any]:
        w = g["weight"]
        return {"sampled": len(g), "steps_per_segment": _wmean(g["n_step"], w),
                "with_step": _wshare(g["n_step"] >= 1, w),
                "with_level_change": _wshare(g["n_level_change"] >= 1, w),
                "with_drop_to_low": _wshare(g["n_down_to_low"] >= 1, w)}

    t["4a_levels"] = _by_subgroup(seg, level)
    if len(levels):
        rows = []
        for kind, g in levels.groupby("kind"):
            q = (0.5, 0.9, 0.99)
            post = g["delta_post"].abs().quantile(q)
            late = g["delta_late"].abs().quantile(q)
            drop = g.dropna(subset=["low_after_s"])
            rows.append({"kind": kind, "events": len(g), "guids": g["guid"].nunique(),
                         "moved_late_share": float((g["delta_late"].abs() >= LEVEL_UNITS).mean()),
                         "post_valid_share": float(g["post_valid_share"].mean()),
                         "late_valid_share": float(g["late_valid_share"].mean()),
                         **{f"abs_delta_post_p{int(100 * p)}": v for p, v in zip(q, post)},
                         **{f"abs_delta_late_p{int(100 * p)}": v for p, v in zip(q, late)},
                         "drops_to_low": len(drop),
                         "low_after_s_p50": float(drop["low_after_s"].median()) if len(drop)
                         else math.nan,
                         "low_after_s_p90": float(drop["low_after_s"].quantile(0.9)) if len(drop)
                         else math.nan,
                         "low_after_valid_share": float(drop["low_after_valid_share"].mean())
                         if len(drop) else math.nan})
        t["4b_level_events"] = pd.DataFrame(rows)

    def dense(g: pd.DataFrame) -> Dict[str, Any]:
        w, has = g["weight"], g["dense_transient_s"] >= DENSE_MIN_S
        return {"sampled": len(g), "with_dense_region": _wshare(has, w),
                "with_dense_and_slow_range": _wshare(has & (g["dense_slow_range"]
                                                            >= SLOW_RANGE_UNITS), w),
                "dense_s_per_segment": _wmean(g["dense_transient_s"], w)}

    t["5a_dense_transient"] = _by_subgroup(seg, dense)
    return t


TABLE_NOTES = {
    "1a_flat_regions": f"Flat regions (constant runs at one value, gaps up to {MERGE_GAP_S:g} s). "
                       "Shares are weighted shares of segments. `censored_ceiling` / "
                       "`censored_rest`: share of samples that are CENSORED at 100 or 127, and "
                       "elsewhere.",
    "1b_flat_values": "The value of the longest flat region, for segments whose region is at "
                      f"least {FLAT_THRESHOLDS_S[0]:g} s.",
    "1c_blip_intervals": "Spacing of consecutive pulses inside one flat region, rounded to 2 s "
                         "(unweighted counts).",
    "1d_blip_shape": "Pulse height above the flat value, pulse length, and how the rules marked "
                     "the pulse (unweighted).",
    "2a_transient_by_hours": "Weighted sample shares per time to delivery (segment centre) and "
                             "delivery mode (cs = 1: caesarean).",
    "2b_transient_by_hours_subgroup": "The same, per subgroup.",
    "2c_stage_split": "Samples before (s1) and after (s2) the second-stage onset; segments with "
                      "an unknown onset are left out.",
    "2d_second_stage_sanity": "The stored `second_stage_onset` (segment start minus onset, s), "
                              "to check its sign and units.",
    "3a_fhr_repaired": "Weighted share of segments whose FHR is mostly REPAIRED, and quantiles "
                       "of `fhr_valid_frac`.",
    "3b_fhr_raw_records": "The rules' raw-channel match, recomputed per record.",
    "3c_fhr_raw_channels": "The pre-fill FHR against every 4 Hz channel at its best offset.",
    "4a_levels": f"STEP runs per segment, and segments with a level change of at least "
                 f"{LEVEL_UNITS:g} units that lasts {LEVEL_LATE_S[0]:g}-{LEVEL_LATE_S[1]:g} s.",
    "4b_level_events": "Level events: |delta| just after and late after; drops to near zero "
                       f"(<= {LOW_UNITS:g}) and the low run that follows.",
    "5a_dense_transient": f"Regions of TRANSIENT/ISLAND at least {DENSE_MIN_S:g} s; "
                          f"`slow_range` >= {SLOW_RANGE_UNITS:g} suggests contractions under "
                          "the noise.",
}


def write_summary(out: Path, tables: Mapping[str, pd.DataFrame], files: Sequence[Mapping],
                  run: Mapping[str, Any]) -> None:
    """Write ``SUMMARY.md`` and ``summary.json``."""
    markdown = _sibling("quality_report")._markdown
    lines = ["# Signal-quality diagnosis", "",
             f"Created {time.strftime('%Y-%m-%d %H:%M')}. Files: {len(files)}. Sampled "
             f"segments: {sum(int(f.get('sampled', 0)) for f in files)} of "
             f"{sum(int(f['segments']) for f in files)}.", "",
             "Send back: `SUMMARY.md`, `summary.json`, `blips.csv`, `level_events.csv`, "
             "`fhr_raw_records.csv`, `fhr_raw_channels.csv`, `mat_metadata.json` (read it "
             "first) and `pages/`.", ""]
    for name, table in tables.items():
        lines += [f"## {name}", "", TABLE_NOTES.get(name, ""), "",
                  markdown(table) if len(table) else "No rows.", ""]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"run": dict(run), "files": list(files),
               "tables": {name: json.loads(table.to_json(orient="records"))
                          for name, table in tables.items()}}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# Pages
# =============================================================================
PAGE_CATEGORIES = {
    "flat_blips": "UA flat region with pulses",
    "level_change": "UA level change that lasts",
    "drop_to_low": "UA drop to near zero",
    "dense_transient": "UA dense transients over a slow trend",
    "stage2_transient": "UA transients in the second stage",
}


def choose_pages(seg: pd.DataFrame, n: int, rng: np.random.Generator) -> List[Tuple[str, Dict]]:
    """Pick example segments per category: half the strongest, half at random, one per GUID.

    Returns:
        ``[(category, row), ...]``.
    """
    stage_share = seg["transient_n_s2"] / seg["n_s2"].where(seg["n_s2"] > 0)
    pools = {
        "flat_blips": (seg[(seg["flat_gaps"] >= 1) & (seg["flat_valid_share"] >= 0.5)
                           & (seg["flat_region_s"] >= FLAT_THRESHOLDS_S[0])], "flat_region_s"),
        "level_change": (seg[seg["n_level_change"] >= 1], "max_abs_delta_late"),
        "drop_to_low": (seg[seg["n_down_to_low"] >= 1], "max_abs_delta_late"),
        "dense_transient": (seg[(seg["dense_transient_s"] >= DENSE_MIN_S)
                                & (seg["dense_slow_range"] >= SLOW_RANGE_UNITS)],
                            "dense_transient_s"),
        "stage2_transient": (seg.assign(stage2_share=stage_share)[
            (seg["n_s2"] >= _n(300.0)) & (stage_share > 0)], "stage2_share"),
    }
    picks = []
    for category, (pool, column) in pools.items():
        if pool.empty:
            continue
        ranked = pool.sort_values(column, ascending=False).drop_duplicates("guid")
        top = ranked.head((n + 1) // 2)
        rest = ranked.drop(top.index)
        chosen = pd.concat([top, rest.iloc[rng.permutation(len(rest))[: n - len(top)]]])
        picks.extend((category, row.to_dict()) for _, row in chosen.iterrows())
    return picks


def draw_pages(out: Path, seg: pd.DataFrame, n: int, seed: int) -> List[str]:
    """Draw the example pages with ``quality_report.plot_example``.

    Returns:
        The page file names.
    """
    if n <= 0 or seg.empty:
        return []
    qr = _sibling("quality_report")
    plt = qr._style()
    (out / "pages").mkdir(exist_ok=True)
    for old in (out / "pages").glob("*.png"):  # pages of an earlier run
        old.unlink()
    names = []
    for number, (category, row) in enumerate(choose_pages(seg, n, np.random.default_rng(seed)), 1):
        try:
            fig = qr.plot_example(plt, category, row, 0.1, title=PAGE_CATEGORIES[category])
        except Exception:  # noqa: BLE001 -- one bad page must not cost the others
            logger.exception(f"page {category} {row['guid']} failed")
            continue
        name = f"{number:02d}_{category}_{row['guid']}_{int(row['row'])}.png"
        fig.savefig(out / "pages" / name, dpi=110)
        plt.close(fig)
        names.append(name)
    return names


# =============================================================================
# Entry point
# =============================================================================
def main(build_dir: Optional[str] = None, output_dir: Optional[str] = None,
         fold: Optional[int] = None, rows_per_file: Optional[int] = None,
         workers: Optional[int] = None, raw_fhr_records: Optional[int] = None,
         metadata_records: Optional[int] = None, examples: Optional[int] = None,
         seed: Optional[int] = None) -> int:
    """Apply the defaults and run the five measurements, the record checks and the pages.

    Args:
        build_dir: The ``output_base_path`` of ``build_fold_datasets.py``, built with
            ``signal_quality=True``.
        output_dir: Where the diagnosis goes. Default: ``<build_dir>/quality_diagnosis``.
        fold: The fold whose partition files are read. Default: 1.
        rows_per_file: Segments sampled per file; 0 reads every segment. Default: 3000.
        workers: Processes for the segment pass. Default: 8.
        raw_fhr_records: Records whose FHR is matched again against the ``.mat`` file; half of
            them mostly ``REPAIRED``. 0 skips the pass. Default: 40.
        metadata_records: Records whose ``.mat`` fields are listed; half of them with a long
            flat region. 0 skips the pass. Default: 30.
        examples: Pages per category; 0 draws none. Default: 4.
        seed: Seed of every sample. Default: 0.

    Returns:
        The process exit code.
    """
    build = Path(build_dir)
    out = Path(output_dir) if output_dir else build / "quality_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    fold = 1 if fold is None else int(fold)
    rows_per_file = 3000 if rows_per_file is None else int(rows_per_file)
    workers = 8 if workers is None else int(workers)
    raw_fhr_records = 40 if raw_fhr_records is None else int(raw_fhr_records)
    metadata_records = 30 if metadata_records is None else int(metadata_records)
    examples = 4 if examples is None else int(examples)
    seed = 0 if seed is None else int(seed)
    run = {"build_dir": str(build), "fold": fold, "rows_per_file": rows_per_file,
           "seed": seed, "quality_version_rules": sq.QUALITY_VERSION,
           "thresholds": {"MIN_RUN_S": MIN_RUN_S, "MERGE_GAP_S": MERGE_GAP_S,
                          "LEVEL_UNITS": LEVEL_UNITS, "LOW_UNITS": LOW_UNITS,
                          "DENSE_MIN_S": DENSE_MIN_S, "SLOW_RANGE_UNITS": SLOW_RANGE_UNITS}}

    tasks = plan_files(build, fold, rows_per_file, seed)
    if not tasks:
        raise FileNotFoundError(f"no HDF5 files of a build under {build}")
    logger.info(f"{len(tasks)} files, up to {rows_per_file or 'all'} segments each")
    result = run_files(tasks, workers)
    seg, blips, levels = result["segments"], result["blips"], result["levels"]
    if seg.empty:
        raise ValueError("no segment with quality datasets; was the build made with "
                         "signal_quality=True?")
    seg.to_csv(out / "segments.csv", index=False)
    blips.to_csv(out / "blips.csv", index=False)
    levels.to_csv(out / "level_events.csv", index=False)

    rng = np.random.default_rng(seed)
    paths = record_paths(build)
    per_guid = (seg.groupby(["path", "file", "guid"])
                .agg(fhr_repaired=("fhr_repaired", "mean"), flat=("flat_region_s", "max"))
                .reset_index())
    per_guid["record"] = per_guid["guid"].map(lambda g: paths.get(g, paths.get(str(g).upper())))
    per_guid = per_guid.dropna(subset=["record"]).drop_duplicates("guid")
    logger.info(f"{len(per_guid)} sampled GUIDs have a .mat path")

    fhr_rows, channel_rows = [], []
    if raw_fhr_records > 0 and len(per_guid):
        chosen = _pick(per_guid, per_guid["fhr_repaired"] > 0.5, per_guid["fhr_repaired"],
                       raw_fhr_records, rng)
        for i, r in enumerate(chosen.itertuples(index=False), 1):
            record_row, channels = check_raw_fhr(r.path, r.file, r.guid, r.record, r.group)
            fhr_rows.append(record_row)
            channel_rows.extend(channels)
            logger.info(f"FHR raw check [{i}/{len(chosen)}] {r.guid}")
    fhr_records, fhr_channels = pd.DataFrame(fhr_rows), pd.DataFrame(channel_rows)
    fhr_records.to_csv(out / "fhr_raw_records.csv", index=False)
    fhr_channels.to_csv(out / "fhr_raw_channels.csv", index=False)

    metadata: List[Dict[str, Any]] = []
    if metadata_records > 0 and len(per_guid):
        chosen = _pick(per_guid, per_guid["flat"] >= FLAT_THRESHOLDS_S[0], per_guid["flat"],
                       metadata_records, rng)
        for r in chosen.itertuples(index=False):
            metadata.append({"guid": r.guid, "group": r.group, "longest_flat_region_s": r.flat,
                             **mat_metadata(r.record)})
    (out / "mat_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    tables = summarise(seg, blips, levels, fhr_records, fhr_channels)
    run["pages"] = draw_pages(out, seg, examples, seed)
    write_summary(out, tables, result["files"], run)
    logger.info(f"diagnosis written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    No argument is ``required`` and none carries a default: :data:`RUN_ARGS` fills every gap,
    and :func:`main` applies the real defaults after the merge.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/new_pipeline/diagnose_signal_quality.py",
        description="Collect the data for the next revision of the signal-quality rules.")
    p.add_argument("--build-dir", dest="build_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--fold", dest="fold", type=int)
    p.add_argument("--rows-per-file", dest="rows_per_file", type=int)
    p.add_argument("--workers", dest="workers", type=int)
    p.add_argument("--raw-fhr-records", dest="raw_fhr_records", type=int)
    p.add_argument("--metadata-records", dest="metadata_records", type=int)
    p.add_argument("--examples", dest="examples", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If ``build_dir`` was supplied by neither route.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("build_dir",))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values)))
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # str -- the output_base_path of build_fold_datasets.py, built with signal_quality=True.
    #        Required.
    "build_dir": r"/data/deid/isilon/MS_model/new_ph_dataset/minimization_folds_quality_v3",
    # str | None -- where the diagnosis goes.  (None -> <build_dir>/quality_diagnosis)
    "output_dir": None,
    # int | None -- the fold whose partition files are read.  (None -> 1)
    "fold": None,
    # int | None -- segments sampled per file; 0 reads every segment.  (None -> 3000)
    "rows_per_file": None,
    # int | None -- processes for the segment pass.  (None -> 8)
    "workers": None,
    # int | None -- records matched again against the raw FHR channels; 0 skips.  (None -> 40)
    "raw_fhr_records": None,
    # int | None -- records whose .mat field names are listed; 0 skips.  (None -> 30)
    "metadata_records": None,
    # int | None -- example pages per category; 0 draws none.  (None -> 4)
    "examples": None,
    # int | None -- seed of every sample.  (None -> 0)
    "seed": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
