r"""Collect the cohort numbers of one dataset build for the paper (Section III-A and Table I).

The script reads one output directory of ``build_fold_datasets.py`` and the inputs that its
``build_manifest.json`` names. It writes aggregate numbers only. No GUID, no per-recording row and
no signal sample goes into the output, and a final pass replaces every string that looks like a
GUID with ``<guid>``. The output can therefore leave the production box.

The cohort forms in this order, and the script reads one source per stage:

1. ``guid_screening_results.csv``: every ``.mat`` recording of the eight subgroup folders, its
   screening result and its eligibility flags (``eligible_2h``, ``eligible_3h``).
2. ``classification_guid_selection_summary.json`` beside the screening CSV (optional): the
   counts of the selection step. The sample paths in it are dropped.
3. The fold folder of ``build_minimization_folds.py``: ``fold_partitions.csv``,
   ``fold_assignment.csv``, ``run_config.json`` and ``cohort_covariates.csv`` (the clinical
   covariates of the classification pool).
4. The build itself: ``pretraining_guids.csv``, the pretraining shards and the partition files.
5. The labour-onset CSV: labour onset, second-stage onset, admission time and contraction count
   per recording, all in hours relative to delivery.
6. Optional: the normalization statistics file and ``quality_report/quality_summary.csv``.

The paper uses three sets of recordings. The **pretraining training set** is
``pre_training_dataset/train``. The **pretraining validation set** is
``pre_training_dataset/test``; it selects the checkpoint. The **evaluation set** is the test
partition of one fold (``fold_<k>/test``) in augmented mode, or the fixed
``k_fold_cross_validation_dataset/test`` in holdout mode. The script reads every segment of these
sets and of the train and validation partitions of fold $k$. For the other folds it reads only the
``guid`` dataset, for the fold counts.

Each stored segment holds $N$ samples at $f_s = 4$ Hz, so it lasts $N / f_s$ seconds. Two
consecutive segments of one recording start a stride $\Delta$ apart and overlap when
$\Delta < N / f_s$. The stored signal of a recording with sorted segment starts
$s_1 < \dots < s_n$ and segment duration $L$ is the length of the union of the segments,

$$
D = L + \sum_{i=2}^{n} \min(s_i - s_{i-1}, L).
$$

Outputs in ``output_dir``:

* ``cohort_stats.json``: every number below, nested by topic;
* ``COHORT_STATS.md``: the same numbers as readable tables;
* ``tables/*.csv``: one CSV per table;
* ``table_cohort_draft.tex``: a draft of Table I (recordings, segments and hours per subgroup).

Edit :data:`RUN_ARGS` at the bottom and press Run, or pass the same values as command-line flags.
The script needs ``numpy``, ``pandas`` and ``h5py`` only; it imports nothing from the pipeline,
so it runs on the production box as a single file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Constants (copied from create_new_pipeline.py; this file imports nothing from it)
# =============================================================================
#: Sampling rate of the stored ``fhr`` and ``up`` signals.
FS_HZ = 4.0
PRETRAIN_DIR = "pre_training_dataset"
KFOLD_DIR = "k_fold_cross_validation_dataset"
MANIFEST_NAME = "build_manifest.json"
SELECTION_SUMMARY_NAME = "classification_guid_selection_summary.json"

#: The eight subgroups in table order: class, then blood gas, then caesarean section.
SUBGROUPS = ("healthy_bg_no_cs", "healthy_bg_cs", "healthy_no_bg_no_cs", "healthy_no_bg_cs",
             "acidosis_no_cs", "acidosis_cs", "hie_no_cs", "hie_cs")
#: Subgroup -> (outcome class, caesarean section, blood gas). Matches ``SUBGROUP_META``.
SUBGROUP_META = {
    "healthy_no_bg_no_cs": ("healthy", False, False),
    "healthy_no_bg_cs": ("healthy", True, False),
    "healthy_bg_cs": ("healthy", True, True),
    "healthy_bg_no_cs": ("healthy", False, True),
    "acidosis_cs": ("acidosis", True, True),
    "acidosis_no_cs": ("acidosis", False, True),
    "hie_cs": ("HIE", True, True),
    "hie_no_cs": ("HIE", False, True),
}
CLASSES = ("healthy", "acidosis", "HIE")
#: Subgroup -> source folder, for the text of the paper.
SUBGROUP_TO_FOLDER = {
    "acidosis_cs": "ACIDOSIS_NO_HIE_CS", "acidosis_no_cs": "ACIDOSIS_NO_HIE_NoCS",
    "healthy_bg_cs": "HEALTHY_NO_ACIDOSIS_CS", "healthy_bg_no_cs": "HEALTHY_NO_ACIDOSIS_NoCS",
    "healthy_no_bg_cs": "HEALTHY_NO_BG_CS", "healthy_no_bg_no_cs": "HEALTHY_NO_BG_NoCS",
    "hie_cs": "HIE_CS", "hie_no_cs": "HIE_NoCS",
}

#: Per-segment scalar datasets read from every file of the detailed sets. A missing dataset
#: (a build without the signal-quality step) gives NaN.
SEGMENT_FIELDS = ("epoch", "cs_label", "bg_label", "time_from_labor_onset", "second_stage_onset",
                  "up_valid_frac", "up_censored_frac", "fhr_valid_frac", "fhr_repair_exact",
                  "fhr_raw_match")
#: Coefficient blocks whose width and warm-up the layout section reports.
BLOCKS = ("fhr_st", "fhr_ph", "up_st", "up_ph", "fhr_up_ph")

#: A GUID with or without hyphens. Every output string passes through :func:`scrub`.
GUID_RE = re.compile(r"[0-9A-Fa-f]{8}-?[0-9A-Fa-f]{4}-?[0-9A-Fa-f]{4}-?[0-9A-Fa-f]{4}-?"
                     r"[0-9A-Fa-f]{12}")

#: The labour-onset CSV columns that the timing table reads, if present.
TLO_COLUMNS = ("labor_onset_hours", "second_stage_onset_hours", "admission_time_hours",
               "totalContractions", "labor_onset_reason_id")


# =============================================================================
# Small helpers
# =============================================================================
def _stem(path: Any) -> str:
    """Return the GUID of a ``.mat`` path: its file name without the extension."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _norm_guid(guid: Any) -> str:
    """Normalize a GUID as ``create_new_pipeline._normalize_guid`` does: upper case, no hyphens."""
    return str(guid).strip().upper().replace("-", "")


def _as_bool(values: pd.Series) -> pd.Series:
    """Parse a CSV column of ``True``/``False`` or ``1``/``0`` into booleans; others are False."""
    mapped = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
    return mapped.fillna(False).astype(bool)


def _class_of(subgroup: str) -> str:
    """Return the outcome class of a subgroup, or ``'unknown'``."""
    meta = SUBGROUP_META.get(subgroup)
    return meta[0] if meta else "unknown"


def describe(values: Iterable[float]) -> Dict[str, float]:
    """Return count, mean, standard deviation, quartiles and range of the finite values.

    Args:
        values: Numbers; NaN and infinite values are left out.

    Returns:
        ``{"n", "mean", "std", "min", "q1", "median", "q3", "max"}``, or ``{"n": 0}``.
    """
    x = np.asarray(list(values) if not isinstance(values, (np.ndarray, pd.Series)) else values,
                   dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0, "min": float(x.min()),
            "q1": float(q1), "median": float(med), "q3": float(q3), "max": float(x.max())}


def _flat(prefix: str, stats: Mapping[str, Any]) -> Dict[str, Any]:
    """Prefix the keys of a :func:`describe` result, for one flat table row."""
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _share(mask: pd.Series) -> float:
    """Return the share of True values among the non-missing ones, or NaN when none is known."""
    mask = mask.dropna()
    return float(mask.astype(bool).mean()) if len(mask) else float("nan")


def jsonable(obj: Any) -> Any:
    """Convert numpy, pandas and path objects to plain JSON types, recursively."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    return obj


def scrub(obj: Any) -> Any:
    """Replace every GUID-like string inside *obj* with ``<guid>``, recursively."""
    if isinstance(obj, str):
        return GUID_RE.sub("<guid>", obj)
    if isinstance(obj, Mapping):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def _attr_value(value: Any) -> Any:
    """Return an HDF5 attribute as a JSON-ready value; long arrays become a summary."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size > 16:
        if np.issubdtype(value.dtype, np.number):
            return {"array_len": int(value.size), **describe(value.ravel())}
        return {"array_len": int(value.size)}
    return jsonable(value)


def _optional_path(value: Optional[str]) -> Optional[Path]:
    """Return an existing path, or ``None`` for an empty or missing one."""
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        logger.warning(f"not found, skipped: {path}")
        return None
    return path


# =============================================================================
# Discovery and reading
# =============================================================================
def discover_files(build_dir: Path) -> pd.DataFrame:
    """List every finished HDF5 file of the build with its set, fold, partition and subgroup.

    Set names: ``pretrain_train``, ``pretrain_val`` (the ``test`` directory of the pretraining
    dataset), ``fold<k>_<partition>`` and ``holdout_test``. A ``.partial`` file is not listed.

    Args:
        build_dir: The ``output_base_path`` of ``build_fold_datasets.py``.

    Returns:
        One row per file with ``path``, ``rel``, ``set``, ``fold``, ``partition``, ``subgroup``.
    """
    rows: List[Dict[str, Any]] = []
    pre = build_dir / PRETRAIN_DIR
    for split, set_name in (("train", "pretrain_train"), ("test", "pretrain_val")):
        for path in sorted((pre / split).glob("*/*.hdf5")):
            rows.append({"path": path, "set": set_name, "fold": 0, "partition": split,
                         "subgroup": path.parent.name})
    kfold = build_dir / KFOLD_DIR
    for fold_dir in sorted(kfold.glob("fold_*")):
        try:
            k = int(fold_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        for part in ("train", "val", "test"):
            for path in sorted((fold_dir / part).glob("*.hdf5")):
                rows.append({"path": path, "set": f"fold{k}_{part}", "fold": k,
                             "partition": part, "subgroup": path.stem})
    for path in sorted((kfold / "test").glob("*.hdf5")):
        rows.append({"path": path, "set": "holdout_test", "fold": 0, "partition": "test",
                     "subgroup": path.stem})
    files = pd.DataFrame(rows, columns=["path", "set", "fold", "partition", "subgroup"])
    files["rel"] = [p.relative_to(build_dir).as_posix() for p in files["path"]]
    files["path"] = files["path"].astype(str)
    return files


def read_segments(path: str, detail: bool) -> Tuple[pd.DataFrame, int]:
    """Read the per-segment scalars of one file.

    Args:
        path: The HDF5 file.
        detail: Read every field of :data:`SEGMENT_FIELDS`; otherwise read ``guid`` only.

    Returns:
        ``(segments, n_rows)``: one row per segment with ``guid`` and, with *detail*, the fields.
    """
    import h5py

    with h5py.File(path, "r") as f:
        n = int(f["guid"].shape[0])
        columns: Dict[str, Any] = {"guid": f["guid"].asstr()[:] if n else np.array([], dtype=str)}
        if detail:
            for name in SEGMENT_FIELDS:
                columns[name] = (f[name][:].astype(np.float64) if name in f
                                 else np.full(n, np.nan))
    return pd.DataFrame(columns), n


def read_layout(path: str) -> Dict[str, Any]:
    """Return the root attributes, the dataset shapes and the causal warm-up of one file.

    Args:
        path: Any finished file of the build; every file carries the same layout.

    Returns:
        ``{"root_attrs", "datasets", "blocks"}``. ``blocks`` holds, per coefficient block, its
        channel count, its warm-up in stored steps and its delay in seconds.
    """
    import h5py

    with h5py.File(path, "r") as f:
        root = {k: _attr_value(v) for k, v in f.attrs.items()
                if k not in ("source_guid_digest", "source_pickle_path")}
        datasets = {name: {"per_segment_shape": list(ds.shape[1:]), "dtype": str(ds.dtype)}
                    for name, ds in f.items()}
        blocks: Dict[str, Any] = {}
        for name in BLOCKS:
            if name not in f:
                continue
            ds = f[name]
            entry: Dict[str, Any] = {"channels": int(ds.shape[1]), "steps": int(ds.shape[2])}
            if "causal_warmup_steps" in ds.attrs:
                entry["warmup_steps"] = describe(np.asarray(ds.attrs["causal_warmup_steps"]))
            if "causal_delay_s" in ds.attrs:
                entry["delay_s"] = describe(np.asarray(ds.attrs["causal_delay_s"]))
            if "sel_band_hz" in ds.attrs:
                entry["band_hz"] = jsonable(np.asarray(ds.attrs["sel_band_hz"]))
            if "sel_k_steps" in ds.attrs:
                entry["k_steps"] = jsonable(np.asarray(ds.attrs["sel_k_steps"]))
            blocks[name] = entry
        bits = {}
        for name in ("up_quality", "fhr_quality"):
            if name in f and "bits" in f[name].attrs:
                bits[name] = sorted(json.loads(f[name].attrs["bits"]).keys())
    return {"root_attrs": root, "datasets": datasets, "blocks": blocks, "quality_bits": bits}


# =============================================================================
# Per-recording table and set summaries
# =============================================================================
def recording_table(segments: pd.DataFrame, segment_s: float) -> pd.DataFrame:
    r"""Collapse the segments to one row per (set, subgroup, recording).

    The stored duration uses the union rule of the module docstring. The labour-onset and
    second-stage times come back from the stored offsets: ``time_from_labor_onset`` is
    $s - t_{\mathrm{lo}}$ and ``second_stage_onset`` is $s - t_{\mathrm{ss}}$, where $s$ is the
    segment start relative to delivery.

    Args:
        segments: Output of :func:`read_segments` with ``set`` and ``subgroup`` added.
        segment_s: Segment duration $L$ in seconds.

    Returns:
        One row per recording.
    """
    seg = segments.sort_values(["set", "subgroup", "guid", "epoch"]).copy()
    keys = ["set", "subgroup", "guid"]
    diff = seg.groupby(keys, sort=False)["epoch"].diff()
    seg["_covered"] = diff.clip(upper=segment_s).fillna(segment_s)
    seg["_tlo_h"] = (seg["epoch"] - seg["time_from_labor_onset"]) / 3600.0
    seg["_sso_h"] = (seg["epoch"] - seg["second_stage_onset"]) / 3600.0
    seg["_has_tlo"] = np.isfinite(seg["time_from_labor_onset"])
    seg["_has_sso"] = np.isfinite(seg["second_stage_onset"])
    rec = seg.groupby(keys, sort=False).agg(
        n_segments=("epoch", "size"), first_epoch=("epoch", "min"),
        last_epoch=("epoch", "max"), covered_s=("_covered", "sum"),
        cs=("cs_label", "max"), bg=("bg_label", "max"),
        has_tlo=("_has_tlo", "max"), has_sso=("_has_sso", "max"),
        tlo_hours=("_tlo_h", "mean"), sso_hours=("_sso_h", "mean"),
        up_valid=("up_valid_frac", "mean"), fhr_valid=("fhr_valid_frac", "mean"),
    ).reset_index()
    rec["class"] = rec["subgroup"].map(_class_of)
    return rec


def summarise_group(seg: pd.DataFrame, rec: pd.DataFrame) -> Dict[str, Any]:
    """Return the numbers of one group of recordings (a subgroup, a class or a whole set).

    Args:
        seg: The group's segments.
        rec: The group's rows of :func:`recording_table`.

    Returns:
        A flat dict; the keys are the column names of the summary tables.
    """
    out: Dict[str, Any] = {"recordings": int(len(rec)), "segments": int(len(seg)),
                           "stored_hours": float(rec["covered_s"].sum() / 3600.0)}
    out.update(_flat("segments_per_recording", describe(rec["n_segments"])))
    out.update(_flat("hours_per_recording", describe(rec["covered_s"] / 3600.0)))
    out.update(_flat("first_segment_h_before_delivery", describe(-rec["first_epoch"] / 3600.0)))
    out.update(_flat("last_segment_min_before_delivery", describe(-rec["last_epoch"] / 60.0)))
    out["cs_share"] = _share((rec["cs"] > 0).where(np.isfinite(rec["cs"])))
    out["bg_share"] = _share((rec["bg"] > 0).where(np.isfinite(rec["bg"])))
    out["labor_onset_known_share"] = _share(rec["has_tlo"])
    out["second_stage_known_share"] = _share(rec["has_sso"])
    out.update(_flat("labor_onset_h_before_delivery", describe(-rec["tlo_hours"])))
    out.update(_flat("second_stage_onset_h_before_delivery", describe(-rec["sso_hours"])))
    sso = seg["second_stage_onset"]
    out["segments_after_second_stage_onset_share"] = _share((sso >= 0).where(np.isfinite(sso)))
    tlo = seg["time_from_labor_onset"]
    out["segments_before_labor_onset_share"] = _share((tlo < 0).where(np.isfinite(tlo)))
    up = seg["up_valid_frac"]
    out["ua_valid_frac_mean"] = float(np.nanmean(up)) if np.isfinite(up).any() else float("nan")
    out["segments_ua_valid_below_half_share"] = _share((up < 0.5).where(np.isfinite(up)))
    for col, key in (("up_censored_frac", "ua_censored_frac_mean"),
                     ("fhr_valid_frac", "fhr_valid_frac_mean"),
                     ("fhr_repair_exact", "fhr_repair_exact_share"),
                     ("fhr_raw_match", "fhr_raw_match_mean")):
        x = seg[col]
        out[key] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
    out.update(_flat("ua_valid_frac_per_recording", describe(rec["up_valid"])))
    out.update(_flat("fhr_valid_frac_per_recording", describe(rec["fhr_valid"])))
    return out


def summarise_sets(segments: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Summarise every detailed set per subgroup, per class and in total.

    Returns:
        One row per (set, level, group); ``level`` is ``subgroup``, ``class`` or ``total``.
    """
    rows: List[Dict[str, Any]] = []
    segments = segments.assign(**{"class": segments["subgroup"].map(_class_of)})
    for set_name in sorted(segments["set"].unique(), key=_set_order):
        s_seg = segments[segments["set"] == set_name]
        s_rec = recordings[recordings["set"] == set_name]
        for level, column, groups in (("subgroup", "subgroup", SUBGROUPS),
                                      ("class", "class", CLASSES)):
            for group in groups:
                g_seg = s_seg[s_seg[column] == group]
                if g_seg.empty:
                    continue
                rows.append({"set": set_name, "level": level, "group": group,
                             **summarise_group(g_seg, s_rec[s_rec[column] == group])})
        rows.append({"set": set_name, "level": "total", "group": "all",
                     **summarise_group(s_seg, s_rec)})
    return pd.DataFrame(rows)


def _set_order(name: str) -> Tuple[int, str]:
    """Sort key of the set names: pretraining first, then the folds, then the holdout test."""
    order = {"pretrain_train": 0, "pretrain_val": 1, "holdout_test": 9}
    return order.get(name, 5), name


def count_all_files(files: pd.DataFrame, segments_by_file: Mapping[str, pd.DataFrame]
                    ) -> pd.DataFrame:
    """Count files, segments and recordings per (set, subgroup) for every file of the build."""
    rows = []
    for (set_name, subgroup), grp in files.groupby(["set", "subgroup"]):
        guids: set = set()
        n_seg = 0
        for path in grp["path"]:
            seg = segments_by_file[path]
            n_seg += len(seg)
            guids |= set(seg["guid"])
        rows.append({"set": set_name, "subgroup": subgroup, "class": _class_of(subgroup),
                     "files": int(len(grp)), "segments": int(n_seg),
                     "recordings": int(len(guids))})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["_o"] = table["set"].map(_set_order)
    table["_s"] = table["subgroup"].map({sg: i for i, sg in enumerate(SUBGROUPS)})
    return table.sort_values(["_o", "_s"]).drop(columns=["_o", "_s"]).reset_index(drop=True)


# =============================================================================
# Screening, selection, folds, covariates and labour timing
# =============================================================================
def screening_tables(screen: pd.DataFrame, healthy_gate: str) -> pd.DataFrame:
    """Summarise the screening CSV per subgroup.

    Args:
        screen: ``guid_screening_results.csv``.
        healthy_gate: The eligibility column of the healthy subgroups (``eligible_3h`` by
            default); the compromised subgroups always use ``eligible_2h``.

    Returns:
        One row per subgroup plus one per class and a total row.
    """
    df = screen.copy()
    for column in ("error", "eligible_2h", "eligible_3h", "has_tlo", "has_second_stage"):
        if column in df:
            df[column] = _as_bool(df[column])
    df["eligible_gate"] = np.where(df["subgroup"].map(_class_of) == "healthy",
                                   df.get(healthy_gate, False), df.get("eligible_2h", False))
    df["eligible_gate"] = df["eligible_gate"].astype(bool) & ~df["error"]
    df["class"] = df["subgroup"].map(_class_of)

    def one(grp: pd.DataFrame) -> Dict[str, Any]:
        good = grp[~grp["error"]]
        row = {"recordings_found": int(len(grp)), "read_errors": int(grp["error"].sum()),
               "eligible_2h": int(grp["eligible_2h"].sum()),
               "eligible_3h": int(grp["eligible_3h"].sum()),
               "eligible_by_gate": int(grp["eligible_gate"].sum()),
               "labor_onset_known": int(grp["has_tlo"].sum()) if "has_tlo" in grp else None,
               "second_stage_known": (int(grp["has_second_stage"].sum())
                                      if "has_second_stage" in grp else None)}
        for col in ("n_total_segments", "n_valid_segments_6h", "n_low_weight", "n_flat_region",
                    "n_duplicate", "n_post_delivery"):
            if col in good:
                row[f"{col}_sum"] = int(pd.to_numeric(good[col], errors="coerce").sum())
        if "estimated_valid_hours_6h" in good:
            row.update(_flat("valid_hours_6h", describe(
                pd.to_numeric(good["estimated_valid_hours_6h"], errors="coerce"))))
            row.update(_flat("valid_hours_6h_eligible", describe(pd.to_numeric(
                good.loc[good["eligible_gate"], "estimated_valid_hours_6h"], errors="coerce"))))
        return row

    rows = [{"level": "subgroup", "group": sg, "folder": SUBGROUP_TO_FOLDER.get(sg, ""),
             **one(df[df["subgroup"] == sg])} for sg in SUBGROUPS if (df["subgroup"] == sg).any()]
    rows += [{"level": "class", "group": c, "folder": "", **one(df[df["class"] == c])}
             for c in CLASSES if (df["class"] == c).any()]
    rows.append({"level": "total", "group": "all", "folder": "", **one(df)})
    return pd.DataFrame(rows)


def cohort_flow(screen: Optional[pd.DataFrame], healthy_gate: str,
                partitions: Optional[pd.DataFrame], pretrain_csv: Optional[pd.DataFrame],
                written: Mapping[str, set], eval_set: str) -> pd.DataFrame:
    """Follow the recordings of each subgroup from the folders to the written files.

    Columns: recordings found; read without error; eligible by the subgroup's gate; in the
    classification pool; extra test; listed for pretraining (train, validation); eligible but in
    no set; and the recordings with at least one segment in the pretraining files and in the
    evaluation set.

    Args:
        screen: The screening CSV, or ``None``.
        healthy_gate: As in :func:`screening_tables`.
        partitions: ``fold_partitions.csv``, or ``None``.
        pretrain_csv: ``pretraining_guids.csv`` of the build, or ``None``.
        written: Set name -> GUIDs with at least one segment in that set's files.
        eval_set: The name of the evaluation set.

    Returns:
        One row per subgroup and a total row.
    """
    rows = []
    for sg in SUBGROUPS:
        row: Dict[str, Any] = {"subgroup": sg, "class": _class_of(sg)}
        eligible: set = set()
        if screen is not None:
            grp = screen[screen["subgroup"] == sg]
            err = _as_bool(grp["error"])
            gate = healthy_gate if _class_of(sg) == "healthy" else "eligible_2h"
            elig = _as_bool(grp[gate]) & ~err
            eligible = set(grp.loc[elig, "guid"])
            row.update(found=int(len(grp)), read_ok=int((~err).sum()), eligible=len(eligible))
        pool: set = set()
        extra: set = set()
        if partitions is not None:
            grp = partitions[partitions["subgroup"] == sg]
            pool = set(grp.loc[~grp["extra_test"], "tracing_guid"])
            extra = set(grp.loc[grp["extra_test"], "tracing_guid"])
            row.update(classification_pool=len(pool), extra_test=len(extra))
        pre: set = set()
        if pretrain_csv is not None:
            grp = pretrain_csv[pretrain_csv["subgroup"] == sg]
            pre = set(grp["guid"])
            row.update(pretrain_listed_train=int((grp["split"] == "train").sum()),
                       pretrain_listed_val=int((grp["split"] == "test").sum()))
        if screen is not None and (partitions is not None or pretrain_csv is not None):
            row["eligible_in_no_set"] = len(eligible - pool - extra - pre)
            row["in_a_set_not_eligible"] = len((pool | extra | pre) - eligible)
        for set_name, label in (("pretrain_train", "written_pretrain_train"),
                                ("pretrain_val", "written_pretrain_val"),
                                (eval_set, "written_eval")):
            if set_name in written:
                row[label] = len({g for g in written[set_name]
                                  if g in written.get(f"_sg_{sg}", set())})
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        total = {"subgroup": "total", "class": "all",
                 **{c: int(table[c].sum()) for c in table.columns
                    if c not in ("subgroup", "class")}}
        table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    return table


def fold_tables(partitions: pd.DataFrame) -> pd.DataFrame:
    """GUID counts per fold, partition and subgroup from ``fold_partitions.csv``.

    Returns:
        One row per (fold, partition) with one column per subgroup, a ``total`` and the extra
        test count.
    """
    counts = partitions.groupby(["fold", "partition", "subgroup"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[c for c in SUBGROUPS if c in counts.columns])
    counts["total"] = counts.sum(axis=1)
    extra = partitions[partitions["extra_test"]].groupby(["fold", "partition"]).size()
    counts["of_which_extra_test"] = extra.reindex(counts.index).fillna(0).astype(int)
    return counts.reset_index()


def covariate_tables(covariates: pd.DataFrame, assignment: pd.DataFrame, eval_fold: int,
                     test_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise the clinical covariates of the classification pool per class.

    Columns with ``guid`` in the name or ending in ``_id`` are identifiers: only their number of
    distinct values is reported. A numeric column with more than 10 distinct values is
    continuous (median, quartiles, missing count). Every other column is categorical (counts of
    each level, at most 30 levels).

    Args:
        covariates: ``cohort_covariates.csv`` indexed by ``tracing_guid``.
        assignment: ``fold_assignment.csv``.
        eval_fold: The fold $k$ whose test partition is the evaluation set.
        test_mode: ``'augmented'`` or ``'holdout'``.

    Returns:
        ``(continuous, categorical)``: long tables with ``scope``, ``class``, ``covariate``.
    """
    meta = assignment.drop_duplicates("tracing_guid").set_index("tracing_guid")
    df = covariates.join(meta[["subgroup", "fold_id"]], how="inner")
    df["class"] = df["subgroup"].map(_class_of)
    scopes = {"classification_pool": df}
    if test_mode == "augmented":
        scopes["eval_set_pool_part"] = df[df["fold_id"] == eval_fold - 1]
    skip = {"subgroup", "fold_id", "class"}
    cont_rows, cat_rows = [], []
    for scope, frame in scopes.items():
        for cls in ("all",) + CLASSES:
            sub = frame if cls == "all" else frame[frame["class"] == cls]
            if sub.empty:
                continue
            for col in frame.columns:
                if col in skip:
                    continue
                values = sub[col]
                if "guid" in col.lower() or col.lower().endswith("_id"):
                    cat_rows.append({"scope": scope, "class": cls, "covariate": col,
                                     "level": "<distinct values>",
                                     "count": int(values.nunique(dropna=True)),
                                     "n": int(len(sub))})
                    continue
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.notna().sum() > 0 and frame[col].nunique(dropna=True) > 10:
                    cont_rows.append({"scope": scope, "class": cls, "covariate": col,
                                      "missing": int(numeric.isna().sum()),
                                      **describe(numeric)})
                    continue
                counts = values.astype(str).where(values.notna(), "<missing>").value_counts()
                for level, count in counts.head(30).items():
                    cat_rows.append({"scope": scope, "class": cls, "covariate": col,
                                     "level": level, "count": int(count), "n": int(len(sub)),
                                     "share": float(count / len(sub))})
    return pd.DataFrame(cont_rows), pd.DataFrame(cat_rows)


def labour_timing(tlo: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    r"""Summarise the labour-onset CSV per set and class, for the recordings of each set.

    All times are hours relative to delivery (negative before it). Derived columns: the first
    stage lasts $t_{\mathrm{ss}} - t_{\mathrm{lo}}$ hours and the second stage $-t_{\mathrm{ss}}$
    hours.

    Args:
        tlo: The labour-onset CSV.
        recordings: Output of :func:`recording_table`.

    Returns:
        A long table with ``set``, ``class``, ``variable`` and the :func:`describe` columns.
    """
    df = tlo.copy()
    if "trace_guid" not in df.columns:
        logger.warning("the labour-onset CSV has no trace_guid column; timing table skipped")
        return pd.DataFrame()
    df["_key"] = df["trace_guid"].map(_norm_guid)
    df = df.drop_duplicates("_key").set_index("_key")
    cols = [c for c in TLO_COLUMNS if c in df.columns and c != "labor_onset_reason_id"]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    if {"labor_onset_hours", "second_stage_onset_hours"} <= set(num.columns):
        num["first_stage_duration_h"] = num["second_stage_onset_hours"] - num["labor_onset_hours"]
        num["second_stage_duration_h"] = -num["second_stage_onset_hours"]
    rows = []
    rec = recordings.assign(_key=recordings["guid"].map(_norm_guid))
    for set_name in sorted(rec["set"].unique(), key=_set_order):
        s = rec[rec["set"] == set_name]
        for cls in ("all",) + CLASSES:
            keys = s["_key"] if cls == "all" else s.loc[s["class"] == cls, "_key"]
            if keys.empty:
                continue
            sub = num.reindex(keys.unique())
            in_csv = int(sub.notna().any(axis=1).sum())
            for col in num.columns:
                rows.append({"set": set_name, "class": cls, "variable": col,
                             "recordings": int(len(sub)), "recordings_in_csv": in_csv,
                             **describe(sub[col])})
    return pd.DataFrame(rows)


def overlap_matrix(guid_sets: Mapping[str, set]) -> pd.DataFrame:
    """Return the size of every pairwise intersection of the GUID sets."""
    names = list(guid_sets)
    data = [[len(guid_sets[a] & guid_sets[b]) for b in names] for a in names]
    return pd.DataFrame(data, index=names, columns=names)


def quality_summary(path: Path, files: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ``quality_report/quality_summary.csv`` per set and subgroup and per set and class.

    Every numeric column is a mean over segments, so the aggregate weights each file by its
    segment count.

    Args:
        path: The CSV.
        files: Output of :func:`discover_files`, to map each file label to its set.

    Returns:
        One row per (set, level, group).
    """
    table = pd.read_csv(path)
    meta = files.set_index("rel")[["set", "subgroup"]]
    table = table.join(meta, on="file", how="inner")
    table["class"] = table["subgroup"].map(_class_of)
    numeric = [c for c in table.columns
               if c not in ("file", "set", "subgroup", "class", "segments", "guids",
                            "quality_version") and pd.api.types.is_numeric_dtype(table[c])]
    rows = []
    for set_name in sorted(table["set"].unique(), key=_set_order):
        s = table[table["set"] == set_name]
        for level, column, groups in (("subgroup", "subgroup", SUBGROUPS),
                                      ("class", "class", CLASSES), ("total", None, ("all",))):
            for group in groups:
                g = s if column is None else s[s[column] == group]
                if g.empty:
                    continue
                w = g["segments"].to_numpy(dtype=np.float64)
                row = {"set": set_name, "level": level, "group": group,
                       "segments": int(w.sum())}
                for c in numeric:
                    x = g[c].to_numpy(dtype=np.float64)
                    ok = np.isfinite(x) & (w > 0)
                    row[c] = float(np.average(x[ok], weights=w[ok])) if ok.any() else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def stats_file_summary(path: Path) -> Dict[str, Any]:
    """Return the root attributes and per-field counts of the normalization statistics file."""
    import h5py

    out: Dict[str, Any] = {"path": str(path)}
    with h5py.File(path, "r") as f:
        out["root_attrs"] = {k: _attr_value(v) for k, v in f.attrs.items()}
        fields = {}
        for name, item in f.items():
            if not hasattr(item, "attrs"):
                continue
            entry = {k: _attr_value(item.attrs[k]) for k in
                     ("count", "n_channels", "shape") if k in item.attrs}
            if "quality_excluded" in item.attrs:
                entry["quality_excluded_total"] = float(np.sum(item.attrs["quality_excluded"]))
            fields[name] = entry
        out["fields"] = fields
    return out


# =============================================================================
# Report writers
# =============================================================================
def _fmt(value: Any, digits: int = 2) -> str:
    """Format one table cell."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        if float(value).is_integer() and abs(value) >= 100:
            return f"{int(value):,}"
        return f"{value:,.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: Optional[Sequence[str]] = None,
                   digits: int = 2) -> str:
    """Render a DataFrame as a GitHub Markdown table without the ``tabulate`` package."""
    if df is None or df.empty:
        return "_No data._\n"
    columns = [c for c in (columns or df.columns) if c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], digits) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _miqr(row: Mapping[str, Any], prefix: str, digits: int = 1) -> str:
    """Return ``median [q1, q3]`` of a flattened :func:`describe` result."""
    med = row.get(f"{prefix}_median")
    if med is None or (isinstance(med, float) and not math.isfinite(med)):
        return "--"
    return (f"{med:.{digits}f} [{row.get(f'{prefix}_q1'):.{digits}f}, "
            f"{row.get(f'{prefix}_q3'):.{digits}f}]")


def readable_set_table(summary: pd.DataFrame, sets: Sequence[str]) -> pd.DataFrame:
    """The compact table of the paper's sets: counts, hours and the main shares."""
    rows = []
    for _, r in summary[summary["set"].isin(sets)].iterrows():
        rows.append({
            "set": r["set"], "group": r["group"], "recordings": r["recordings"],
            "segments": r["segments"], "stored_h": r["stored_hours"],
            "h/rec median [IQR]": _miqr(r, "hours_per_recording"),
            "seg/rec median [IQR]": _miqr(r, "segments_per_recording", 0),
            "first seg h before delivery": _miqr(r, "first_segment_h_before_delivery"),
            "CS %": 100 * r["cs_share"], "BG %": 100 * r["bg_share"],
            "labor onset known %": 100 * r["labor_onset_known_share"],
            "2nd stage known %": 100 * r["second_stage_known_share"],
            "seg after 2nd-stage onset %": 100 * r["segments_after_second_stage_onset_share"],
            "UA valid %": 100 * r["ua_valid_frac_mean"],
            "seg UA valid<50% %": 100 * r["segments_ua_valid_below_half_share"],
            "FHR valid %": 100 * r["fhr_valid_frac_mean"],
        })
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame, eval_set: str) -> str:
    """Draft Table I: recordings, segments and stored hours per subgroup and set."""
    sets = [("pretrain_train", "Pretraining (train)"), ("pretrain_val", "Pretraining (val.)"),
            (eval_set, "Evaluation")]
    sub = summary[summary["level"].isin(["subgroup", "class", "total"])]

    def cell(set_name: str, group: str, key: str) -> str:
        row = sub[(sub["set"] == set_name) & (sub["group"] == group)]
        if row.empty:
            return "--"
        value = row.iloc[0][key]
        if value is None or not np.isfinite(value):
            return "--"
        return f"{value:,.0f}".replace(",", "{,}")

    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for _, label in sets)
    lines = [
        "% Draft of Table I, written by collect_cohort_stats.py. Check every number against the",
        "% build before use. rec. = recordings, seg. = 22-min segments, h = stored hours.",
        "\\begin{table*}[!t]",
        "\\caption{Cohort per subgroup. \\TBD{caption}}",
        "\\label{tab:cohort}",
        "\\centering",
        "\\begin{tabular}{l" + "rrr" * len(sets) + "}",
        "\\toprule",
        f"Subgroup & {head} \\\\",
        " & " + " & ".join(["rec.", "seg.", "h"] * len(sets)) + " \\\\",
        "\\midrule",
    ]
    names = {sg: sg.replace("_", "\\_") for sg in SUBGROUPS}
    names.update({c: f"\\textit{{{c}, all}}" for c in CLASSES})
    names["all"] = "\\textbf{Total}"
    order = list(SUBGROUPS[:4]) + ["healthy"] + list(SUBGROUPS[4:6]) + ["acidosis"] + \
        list(SUBGROUPS[6:]) + ["HIE", "all"]
    for group in order:
        cells = []
        for set_name, _ in sets:
            cells += [cell(set_name, group, "recordings"), cell(set_name, group, "segments"),
                      cell(set_name, group, "stored_hours")]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{names[group]} & " + " & ".join(cells) + " \\\\")
        if group in ("healthy", "acidosis", "HIE"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


# =============================================================================
# Entry point
# =============================================================================
def main(build_dir: str, output_dir: Optional[str] = None,
         screening_csv_path: Optional[str] = None, fold_dir: Optional[str] = None,
         tlo_csv_path: Optional[str] = None, stats_path: Optional[str] = None,
         eval_fold: int = 1, healthy_eligibility: str = "eligible_3h",
         all_folds: bool = True) -> int:
    """Collect every cohort number of one build and write the report.

    Args:
        build_dir: The ``output_base_path`` of ``build_fold_datasets.py``.
        output_dir: Where the report goes; ``None`` writes ``<build_dir>/paper_cohort_stats``.
        screening_csv_path: ``None`` takes the path from ``build_manifest.json``.
        fold_dir: ``None`` takes the path from ``build_manifest.json``.
        tlo_csv_path: ``None`` takes the path from ``build_manifest.json``.
        stats_path: The normalization statistics file; ``None`` tries ``<build_dir>/stats.hdf5``.
        eval_fold: The fold $k$ whose test partition is the evaluation set (augmented mode).
        healthy_eligibility: The screening column that admitted healthy recordings to the
            classification pool (``eligible_3h`` unless the selection run changed it).
        all_folds: Also count the files of every other fold (reads their ``guid`` only).

    Returns:
        The process exit code.
    """
    build = Path(build_dir)
    out = Path(output_dir) if output_dir else build / "paper_cohort_stats"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"created_at": _dt.datetime.now().isoformat(timespec="seconds"),
                              "build_dir": str(build), "eval_fold": eval_fold}
    warnings: List[str] = []

    # ---------------------------------------------------------------- manifest and inputs
    manifest: Dict[str, Any] = {}
    if (build / MANIFEST_NAME).exists():
        with open(build / MANIFEST_NAME, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        warnings.append(f"{MANIFEST_NAME} not found in the build directory")
    settings = manifest.get("settings", {})
    report["build_settings"] = {
        "transform": settings.get("transform"), "leg_alignment": settings.get("leg_alignment"),
        "phase_operator": settings.get("phase_operator"),
        "signal_quality": settings.get("signal_quality"),
        "quality_version": settings.get("quality_version"),
        "pretrain": manifest.get("pretrain"), "steps": manifest.get("steps"),
        "fold_run_config": {k: v for k, v in manifest.get("fold_run_config", {}).items()
                            if k != "unstratified_guids"},
        "guids_per_subgroup_manifest": manifest.get("guids_per_subgroup"),
    }
    screening_p = _optional_path(screening_csv_path or manifest.get("screening_csv_path"))
    fold_p = _optional_path(fold_dir or manifest.get("fold_dir"))
    tlo_p = _optional_path(tlo_csv_path or settings.get("tlo_csv_path"))
    stats_p = _optional_path(stats_path or (str(build / "stats.hdf5")
                                            if (build / "stats.hdf5").exists() else None))
    report["inputs"] = {"screening_csv": str(screening_p) if screening_p else None,
                        "fold_dir": str(fold_p) if fold_p else None,
                        "tlo_csv": str(tlo_p) if tlo_p else None,
                        "stats_file": str(stats_p) if stats_p else None}

    run_config: Dict[str, Any] = {}
    partitions = assignment = None
    if fold_p is not None:
        partitions = pd.read_csv(fold_p / "fold_partitions.csv",
                                 dtype={"tracing_guid": str, "subgroup": str})
        partitions["extra_test"] = _as_bool(partitions["extra_test"])
        partitions["fold"] = partitions["fold"].astype(int)
        assignment = pd.read_csv(fold_p / "fold_assignment.csv",
                                 dtype={"tracing_guid": str, "subgroup": str})
        if (fold_p / "run_config.json").exists():
            with open(fold_p / "run_config.json", encoding="utf-8") as f:
                run_config = json.load(f)
    run_config = run_config or manifest.get("fold_run_config", {})
    test_mode = run_config.get("test_mode", "augmented")
    eval_set = f"fold{eval_fold}_test" if test_mode == "augmented" else "holdout_test"
    report["test_mode"] = test_mode
    report["eval_set"] = eval_set

    # ---------------------------------------------------------------- the HDF5 files
    files = discover_files(build)
    if files.empty:
        raise SystemExit(f"no finished HDF5 file under {build}")
    detail_sets = {"pretrain_train", "pretrain_val", eval_set, f"fold{eval_fold}_train",
                   f"fold{eval_fold}_val"}
    files["detail"] = files["set"].isin(detail_sets)
    if not all_folds:
        files = files[files["detail"]].reset_index(drop=True)
    logger.info(f"{len(files)} files, {int(files['detail'].sum())} read in detail")

    segments_by_file: Dict[str, pd.DataFrame] = {}
    detailed: List[pd.DataFrame] = []
    for i, row in enumerate(files.itertuples(index=False), 1):
        seg, _ = read_segments(row.path, bool(row.detail))
        segments_by_file[row.path] = seg
        if row.detail:
            detailed.append(seg.assign(set=row.set, subgroup=row.subgroup))
        if i % 25 == 0 or i == len(files):
            logger.info(f"read {i}/{len(files)} files")
    if not detailed:
        raise SystemExit(f"none of the sets {sorted(detail_sets)} has a file under {build}")
    segments = pd.concat(detailed, ignore_index=True)
    if eval_set not in set(segments["set"]):
        warnings.append(f"the evaluation set {eval_set} has no file in the build")

    layout_path = files.loc[files["detail"], "path"].iloc[0]
    layout = read_layout(layout_path)
    report["layout"] = layout
    n_samples = layout["datasets"]["fhr"]["per_segment_shape"][0]
    n_steps = (layout["datasets"]["fhr_st"]["per_segment_shape"][-1]
               if "fhr_st" in layout["datasets"] else None)
    segment_s = n_samples / FS_HZ
    diffs = segments.sort_values(["set", "guid", "epoch"]).groupby(["set", "guid"])["epoch"].diff()
    diffs = diffs[diffs > 0]
    stride = float(diffs.mode().iloc[0]) if len(diffs) else float("nan")
    report["geometry"] = {
        "sampling_rate_hz": FS_HZ, "samples_per_segment": n_samples,
        "segment_duration_s": segment_s, "segment_duration_min": segment_s / 60.0,
        "stride_s_mode": stride, "stride_min_mode": stride / 60.0,
        "overlap_s": segment_s - stride if math.isfinite(stride) else None,
        "steps_per_segment": n_steps,
        "step_duration_s": (segment_s / n_steps) if n_steps else None,
        "epoch_s": describe(segments["epoch"]),
        "segments_with_epoch_at_or_after_delivery": int((segments["epoch"] >= 0).sum()),
    }

    # ---------------------------------------------------------------- per set summaries
    recordings = recording_table(segments, segment_s)
    summary = summarise_sets(segments, recordings)
    summary.to_csv(out / "tables" / "set_summary.csv", index=False)
    report["set_summary"] = summary
    counts = count_all_files(files, segments_by_file)
    counts.to_csv(out / "tables" / "file_counts_all_sets.csv", index=False)
    report["file_counts_all_sets"] = counts

    written = {name: set(grp["guid"]) for name, grp in segments.groupby("set")}
    for sg, grp in segments.groupby("subgroup"):
        written[f"_sg_{sg}"] = set(grp["guid"])

    # ---------------------------------------------------------------- screening and flow
    screen = None
    if screening_p is not None:
        screen = pd.read_csv(screening_p, dtype={"guid": str, "record_path": str,
                                                 "subgroup": str})
        screen["guid"] = screen["record_path"].map(_stem)
        sel_path = screening_p.parent / SELECTION_SUMMARY_NAME
        if sel_path.exists():
            with open(sel_path, encoding="utf-8") as f:
                sel = json.load(f)
            report["selection_summary"] = {
                "test_mode": sel.get("test_mode"), "stats": sel.get("stats"),
                **{pool: {sg: v.get("count") for sg, v in sel.get(pool, {}).items()}
                   for pool in ("trainval", "test", "pretraining")}}
            # The selection run records the healthy gate it applied; it overrides RUN_ARGS.
            recorded = (sel.get("stats") or {}).get("healthy_eligibility")
            if recorded and recorded != healthy_eligibility:
                warnings.append(f"healthy_eligibility={healthy_eligibility!r} replaced by "
                                f"{recorded!r}, the gate the selection run recorded")
                healthy_eligibility = recorded
        report["healthy_eligibility"] = healthy_eligibility
        screen_table = screening_tables(screen, healthy_eligibility)
        screen_table.to_csv(out / "tables" / "screening.csv", index=False)
        report["screening"] = screen_table
    pretrain_csv = None
    if (build / "pretraining_guids.csv").exists():
        pretrain_csv = pd.read_csv(build / "pretraining_guids.csv",
                                   dtype={"guid": str, "subgroup": str})
    flow = cohort_flow(screen, healthy_eligibility, partitions, pretrain_csv, written, eval_set)
    flow.to_csv(out / "tables" / "cohort_flow.csv", index=False)
    report["cohort_flow"] = flow

    # ---------------------------------------------------------------- folds
    if partitions is not None:
        folds = fold_tables(partitions)
        folds.to_csv(out / "tables" / "fold_counts.csv", index=False)
        report["fold_counts"] = folds
        report["fold_run_config"] = {k: v for k, v in run_config.items()
                                     if k != "unstratified_guids"}
        report["fold_run_config"]["n_unstratified_guids"] = len(
            run_config.get("unstratified_guids", []))
        if fold_p is not None and (fold_p / "minimization_report.json").exists():
            with open(fold_p / "minimization_report.json", encoding="utf-8") as f:
                report["minimization_report"] = json.load(f)
        if eval_set in written:
            extra = set(partitions.loc[partitions["extra_test"], "tracing_guid"])
            report["eval_set_composition"] = {
                "recordings": len(written[eval_set]),
                "from_classification_pool": len(written[eval_set] - extra),
                "extra_test": len(written[eval_set] & extra)}

    # ---------------------------------------------------------------- disjointness
    guid_sets: Dict[str, set] = {k: v for k, v in written.items() if not k.startswith("_sg_")}
    if partitions is not None:
        guid_sets["listed_pool"] = set(partitions.loc[~partitions["extra_test"], "tracing_guid"])
        guid_sets["listed_extra_test"] = set(
            partitions.loc[partitions["extra_test"], "tracing_guid"])
    if pretrain_csv is not None:
        guid_sets["listed_pretraining"] = set(pretrain_csv["guid"])
    overlaps = overlap_matrix(guid_sets)
    overlaps.to_csv(out / "tables" / "guid_overlaps.csv")
    report["guid_overlaps"] = overlaps.to_dict()
    must_be_empty = [("pretrain_train", "pretrain_val"), ("pretrain_train", eval_set),
                     ("pretrain_val", eval_set), ("listed_pretraining", "listed_pool"),
                     ("listed_pretraining", "listed_extra_test"),
                     (f"fold{eval_fold}_train", f"fold{eval_fold}_val"),
                     (f"fold{eval_fold}_train", eval_set), (f"fold{eval_fold}_val", eval_set)]
    checks = {}
    for a, b in must_be_empty:
        if a in guid_sets and b in guid_sets:
            n = len(guid_sets[a] & guid_sets[b])
            checks[f"{a} & {b}"] = n
            if n:
                warnings.append(f"{n} recordings are in both {a} and {b}")
    report["disjointness_checks"] = checks

    # ---------------------------------------------------------------- covariates and timing
    if fold_p is not None and (fold_p / "cohort_covariates.csv").exists():
        cov = pd.read_csv(fold_p / "cohort_covariates.csv", index_col=0)
        cov.index = cov.index.astype(str)
        cont, cat = covariate_tables(cov, assignment, eval_fold, test_mode)
        cont.to_csv(out / "tables" / "covariates_continuous.csv", index=False)
        cat.to_csv(out / "tables" / "covariates_categorical.csv", index=False)
        report["covariates_continuous"] = cont
        report["covariates_categorical"] = cat
    if tlo_p is not None:
        tlo = pd.read_csv(tlo_p, dtype={"trace_guid": str})
        timing = labour_timing(tlo, recordings)
        timing.to_csv(out / "tables" / "labour_timing.csv", index=False)
        report["labour_timing"] = timing
        if "labor_onset_reason_id" in tlo.columns:
            keys = {s: set(recordings.loc[recordings["set"] == s, "guid"].map(_norm_guid))
                    for s in recordings["set"].unique()}
            tlo_keys = tlo.assign(_key=tlo["trace_guid"].map(_norm_guid)).drop_duplicates("_key")
            report["labor_onset_reason_counts"] = {
                s: tlo_keys[tlo_keys["_key"].isin(k)]["labor_onset_reason_id"]
                .value_counts(dropna=False).to_dict() for s, k in keys.items()}

    # ---------------------------------------------------------------- quality and statistics
    q_path = build / "quality_report" / "quality_summary.csv"
    if q_path.exists():
        quality = quality_summary(q_path, files)
        quality.to_csv(out / "tables" / "quality_report_summary.csv", index=False)
        report["quality_report_summary"] = quality
        if eval_set.startswith("fold") and eval_set != "fold1_test":
            warnings.append("quality_report/ covers fold_1 only; its numbers for the "
                            "evaluation set describe fold_1/test, not " + eval_set)
    if stats_p is not None:
        report["stats_file"] = stats_file_summary(stats_p)

    report["warnings"] = warnings
    report = scrub(jsonable(report))
    with open(out / "cohort_stats.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    (out / "table_cohort_draft.tex").write_text(latex_table(summary, eval_set), encoding="utf-8")
    (out / "COHORT_STATS.md").write_text(
        scrub(render_markdown(report, summary, counts, flow, eval_set, eval_fold, screen is not None)),
        encoding="utf-8")
    for message in warnings:
        logger.warning(message)
    logger.info(f"report written to {out}")
    return 0


def render_markdown(report: Mapping[str, Any], summary: pd.DataFrame, counts: pd.DataFrame,
                    flow: pd.DataFrame, eval_set: str, eval_fold: int,
                    has_screening: bool) -> str:
    """Write ``COHORT_STATS.md``: the numbers of the report as readable tables."""
    paper_sets = ["pretrain_train", "pretrain_val", eval_set]
    md: List[str] = ["# Cohort statistics for Section III-A", "",
                     f"Build: `{report['build_dir']}`  ", f"Created: {report['created_at']}  ",
                     f"Test mode: {report['test_mode']}; evaluation set: `{eval_set}`", ""]
    geo = report["geometry"]
    md += ["## 1. Recording geometry", "",
           f"- Sampling rate: {geo['sampling_rate_hz']} Hz; segment: "
           f"{geo['samples_per_segment']} samples = {_fmt(geo['segment_duration_min'], 1)} min.",
           f"- Stride between segment starts (mode): {_fmt(geo['stride_min_mode'], 1)} min; "
           f"overlap {_fmt(geo['overlap_s'], 0)} s.",
           f"- Stored steps per segment: {geo['steps_per_segment']} of "
           f"{geo['step_duration_s']} s each.",
           f"- Segment start (epoch, s before delivery is negative): median "
           f"{geo['epoch_s'].get('median')}, min {geo['epoch_s'].get('min')}, "
           f"max {geo['epoch_s'].get('max')}; segments at or after delivery: "
           f"{geo['segments_with_epoch_at_or_after_delivery']}.", ""]
    blocks = report["layout"]["blocks"]
    rows = [{"block": k, "channels": v.get("channels"), "steps": v.get("steps"),
             "warm-up steps median": (v.get("warmup_steps") or {}).get("median"),
             "warm-up steps max": (v.get("warmup_steps") or {}).get("max"),
             "delay s min": (v.get("delay_s") or {}).get("min"),
             "delay s max": (v.get("delay_s") or {}).get("max")} for k, v in blocks.items()]
    md += ["Coefficient blocks:", "", markdown_table(pd.DataFrame(rows)), "",
           "Root attributes: `" + json.dumps(report["layout"]["root_attrs"])[:1500] + "`", ""]
    if has_screening:
        md += ["## 2. Screening (all recordings in the eight folders)", "",
               markdown_table(pd.DataFrame(report["screening"]),
                              ["level", "group", "folder", "recordings_found", "read_errors",
                               "eligible_2h", "eligible_3h", "eligible_by_gate",
                               "labor_onset_known", "second_stage_known",
                               "valid_hours_6h_median", "valid_hours_6h_q1",
                               "valid_hours_6h_q3"]), ""]
        if "selection_summary" in report:
            md += ["Selection summary (`stats` of the selection step):", "", "```json",
                   json.dumps(report["selection_summary"], indent=2)[:4000], "```", ""]
    md += ["## 3. Cohort flow per subgroup", "",
           "Found -> read without error -> eligible by gate -> classification pool / extra test "
           "/ pretraining list -> recordings with at least one written segment.", "",
           markdown_table(flow), ""]
    md += ["## 4. The paper's sets (pretraining train, pretraining validation, evaluation)", "",
           markdown_table(readable_set_table(summary, paper_sets), digits=1), ""]
    if "eval_set_composition" in report:
        md += [f"Evaluation set composition: `{json.dumps(report['eval_set_composition'])}`", ""]
    md += [f"## 5. Fold {eval_fold} train and validation partitions (classification pool)", "",
           markdown_table(readable_set_table(
               summary, [f"fold{eval_fold}_train", f"fold{eval_fold}_val"]), digits=1), ""]
    if report.get("labour_timing"):
        timing = pd.DataFrame(report["labour_timing"])
        timing = timing[timing["set"].isin(paper_sets)]
        md += ["## 6. Labour timing (labour-onset CSV, hours relative to delivery)", "",
               markdown_table(timing, ["set", "class", "variable", "recordings",
                                       "recordings_in_csv", "n", "median", "q1", "q3", "min",
                                       "max"]), ""]
    if "quality_report_summary" in report:
        q = pd.DataFrame(report["quality_report_summary"])
        q = q[q["level"].isin(["class", "total"])]
        md += ["## 7. Signal quality (quality_report/quality_summary.csv, segment-weighted)", "",
               markdown_table(q, digits=3), ""]
    if "covariates_continuous" in report:
        cont = pd.DataFrame(report["covariates_continuous"])
        md += ["## 8. Clinical covariates of the classification pool", "",
               "Continuous:", "",
               markdown_table(cont, ["scope", "class", "covariate", "n", "missing", "median",
                                     "q1", "q3", "min", "max"]), "",
               "Categorical: see `tables/covariates_categorical.csv`.", ""]
    if "fold_counts" in report:
        md += ["## 9. Folds (fold_partitions.csv, recordings)", "",
               markdown_table(pd.DataFrame(report["fold_counts"])), "",
               "Run config: `" + json.dumps(report.get("fold_run_config", {}))[:2000] + "`", ""]
    md += ["## 10. Written files: segments and recordings per set and subgroup", "",
           markdown_table(counts), ""]
    md += ["## 11. Disjointness checks (shared recordings; every value must be 0)", "",
           "```json", json.dumps(report["disjointness_checks"], indent=2), "```", ""]
    if "stats_file" in report:
        md += ["## 12. Normalization statistics file", "", "```json",
               json.dumps(report["stats_file"], indent=2)[:4000], "```", ""]
    md += ["## Warnings", ""] + [f"- {w}" for w in report["warnings"]] + \
          (["- none"] if not report["warnings"] else []) + [""]
    return "\n".join(md)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. No flag has a default: :data:`RUN_ARGS` fills the gaps."""
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/new_pipeline/collect_cohort_stats.py",
        description="Collect the cohort numbers of one dataset build for the paper.")
    p.add_argument("--build-dir", dest="build_dir")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--screening-csv-path", dest="screening_csv_path")
    p.add_argument("--fold-dir", dest="fold_dir")
    p.add_argument("--tlo-csv-path", dest="tlo_csv_path")
    p.add_argument("--stats-path", dest="stats_path")
    p.add_argument("--eval-fold", dest="eval_fold", type=int)
    p.add_argument("--healthy-eligibility", dest="healthy_eligibility",
                   choices=("eligible_2h", "eligible_3h"))
    p.add_argument("--all-folds", dest="all_folds", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run."""
    parsed = vars(build_parser().parse_args(argv))
    values = {**RUN_ARGS, **{k: v for k, v in parsed.items() if v is not None}}
    if not values.get("build_dir") or "/path/to/" in str(values["build_dir"]):
        raise SystemExit("Set build_dir in RUN_ARGS or pass --build-dir.")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # str -- the output_base_path of build_fold_datasets.py. Required.
    "build_dir": r"/data1/fetal-heart-tracing/HDF5_Datasets/last-12-hours-causal-p50-ua-masked-2026-10-27",

    # str | None -- where the report goes.  (None -> <build_dir>/paper_cohort_stats)
    "output_dir": None,

    # str | None -- the inputs; None takes each path from build_manifest.json. Set one when
    #        the build was moved and the manifest path no longer exists.
    "screening_csv_path": None,
    "fold_dir": None,
    "tlo_csv_path": None,

    # str | None -- the normalization statistics file of the training config (stat_path).
    #        (None -> <build_dir>/stats.hdf5 if it exists)
    "stats_path": None,

    # int -- the fold whose test partition is the evaluation set (augmented mode).
    "eval_fold": 1,

    # "eligible_3h" | "eligible_2h" -- the gate that admitted healthy recordings to the
    #        classification pool in the selection run.
    "healthy_eligibility": "eligible_3h",

    # bool -- also count the files of folds 2..K (reads only their guid dataset).
    "all_folds": True,
}


if __name__ == "__main__":
    sys.exit(_cli())
