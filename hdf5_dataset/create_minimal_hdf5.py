"""Build small, raw-signal-only HDF5 files: one for acidosis, one for HIE, one for healthy.

Every acidosis and every HIE GUID is read. Healthy GUIDs are drawn at random (seeded) from the
four healthy subgroups pooled, until the healthy file holds as many GUIDs as acidosis + HIE
together. A GUID counts only once it gives at least one segment.

Each record goes through the same reading and segment filtering as ``create_new_pipeline.py``
(MIMO read with the -20 s UP shift, sanitising, de-duplication, weight and flat-region gates,
post-delivery segments dropped; the 12.4 h range). Nothing else is computed: no scattering, no
phase harmonics, no masks, no signal-quality step.

Each record is also read with the MIMO fill off (``create_new_pipeline.mimo_fill``), and the files
store what the signal-quality step needs, so ``minimal_to_split_hdf5.py --quality v4|v5`` can run it
later without the ``.mat`` files: the pre-fill signals and the FHR repair mask from the record's raw
channel. The builder gates the filled and the pre-fill signals separately, so a row is kept when
either gate keeps it, and ``kept_filled`` / ``kept_prefill`` say which.

Per segment, each file stores:
    fhr, up                 (N, 5280) float32  -- 4 Hz, 22 min
    weight                  (N, 330)  float32  -- MIMO sample weights (1/16 rate)
    epoch                   (N,)      float32  -- segment start, seconds relative to delivery
    guid                    (N,)      str
    class_label             (N,)      uint8    -- 1 healthy, 2 acidosis, 3 HIE
    cs_label, bg_label      (N,)      uint8
    time_from_labor_onset   (N,)      float32  -- epoch - labour onset (s); NaN if unknown
    second_stage_onset      (N,)      float32  -- epoch - second-stage onset (s); NaN if unknown
    fhr_pre, up_pre         (N, 5280) float32  -- MIMO fill off; a gap reads 0
    kept_filled             (N,)      uint8    -- the gates keep the filled signals (a build
                                                  without the quality step reads these rows)
    kept_prefill            (N,)      uint8    -- the gates keep the pre-fill signals (a build
                                                  with the quality step reads these rows)
    fhr_repaired            (N, 5280) uint8    -- signal_quality.raw_fhr_repaired over the
                                                  record's kept_prefill rows; 0 elsewhere
    fhr_repair_exact        (N,)      uint8    -- 1 when that mask was read from the raw channel
    fhr_raw_match           (N,)      float32  -- its record-wide match rate; NaN without it

``minimal_guid_selection.csv`` beside the files lists every GUID read and how many segments it gave.

Run: edit :data:`RUN_ARGS` at the bottom and press Run, or override on the command line, e.g.
``python hdf5_dataset/create_minimal_hdf5.py --output_dir /data1/.../minimal_raw``.
"""

import argparse
import logging
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

#: Repository root: ``hdf5_dataset/create_minimal_hdf5.py`` -> up two.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the pipeline can be found.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Production: every file in one folder. Repository: the packages. Each falls back only when the
# module itself is absent, so an error inside it is not hidden.
try:
    import create_new_pipeline as cnp
except ModuleNotFoundError as _exc:
    if _exc.name != "create_new_pipeline":
        raise
    from hdf5_dataset.new_pipeline import create_new_pipeline as cnp
try:
    import profile_signal_quality as psq
except ModuleNotFoundError as _exc:
    if _exc.name != "profile_signal_quality":
        raise
    from hdf5_dataset.new_pipeline import profile_signal_quality as psq
try:
    from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args
except ModuleNotFoundError as _exc:
    if not str(_exc.name).startswith("teb_vae"):
        raise
    from launch import missing_required, resolve_launch_args

logger = logging.getLogger(__name__)

CLASS_NAMES = {1: "healthy", 2: "acidosis", 3: "hie"}
SELECTION_CSV = "minimal_guid_selection.csv"
#: Rows per chunk: several segments per chunk compress far better than the pipeline's one.
CHUNK_ROWS = 16
#: name -> (per-row shape, dtype)
FIELDS = {
    "fhr": ((cnp.SIGNAL_LENGTH,), "f4"),
    "up": ((cnp.SIGNAL_LENGTH,), "f4"),
    "weight": ((cnp.SEQUENCE_LENGTH,), "f4"),
    "epoch": ((), "f4"),
    "guid": ((), h5py.string_dtype(encoding="utf-8")),
    "class_label": ((), "u1"),
    "cs_label": ((), "u1"),
    "bg_label": ((), "u1"),
    "time_from_labor_onset": ((), "f4"),
    "second_stage_onset": ((), "f4"),
    "fhr_pre": ((cnp.SIGNAL_LENGTH,), "f4"),
    "up_pre": ((cnp.SIGNAL_LENGTH,), "f4"),
    "kept_filled": ((), "u1"),
    "kept_prefill": ((), "u1"),
    "fhr_repaired": ((cnp.SIGNAL_LENGTH,), "u1"),
    "fhr_repair_exact": ((), "u1"),
    "fhr_raw_match": ((), "f4"),
}
#: Per-row fields of :func:`read_record`, in :data:`FIELDS` order.
SIGNAL_FIELDS = ("fhr", "up", "weight", "fhr_pre", "up_pre", "kept_filled", "kept_prefill",
                 "fhr_repaired", "fhr_repair_exact", "fhr_raw_match")


def read_record(job: Tuple[str, str], verbose: bool = False) -> Dict[str, Any]:
    """Read one .mat file with and without the MIMO fill; keep what either gate keeps.

    ``profile_signal_quality.load_record`` does both reads, checks they agree on the segments and
    weights, de-duplicates and gates each. The FHR repair mask is computed as the build's
    ``_clean_record`` computes it: over the record's pre-fill-kept segments, falling back to none
    when the raw channel is missing or unreadable.

    Args:
        job: ``(record_path, subgroup)``.
        verbose: Unused; MIMO output stays suppressed.

    Returns:
        Dict with ``path``, ``subgroup``, ``epoch``, one array per :data:`SIGNAL_FIELDS` (empty
        when nothing survives) and ``error`` (``""`` on success).
    """
    path, subgroup = job
    try:
        r = psq.load_record(path)
        kept_filled = np.array([g == "kept" for g in r["gate"]], dtype=bool)
        kept_prefill = np.array([g == "kept" for g in r["gate_prefill"]], dtype=bool)
        n = len(kept_prefill)
        repaired = np.zeros((n, cnp.SIGNAL_LENGTH), dtype=np.uint8)
        exact = np.zeros(n, dtype=np.uint8)
        match = np.full(n, np.nan, dtype=np.float32)
        if kept_prefill.any():
            try:
                mask, info = psq.sq.raw_fhr_repaired(path, r["domain_start"][kept_prefill],
                                                     r["fhr_pre"][kept_prefill],
                                                     r["present"][kept_prefill])
            except Exception as exc:  # noqa: BLE001 -- the build's rules stand in, as there
                mask, info = None, {"raw_fhr": repr(exc)}
            if mask is not None:
                repaired[kept_prefill] = mask
                exact[kept_prefill] = 1
                match[kept_prefill] = info.get("raw_match", np.nan)
        keep = kept_filled | kept_prefill
        out = dict(fhr=r["fhr_now"], up=r["up_now"], weight=r["weight"], fhr_pre=r["fhr_pre"],
                   up_pre=r["up_pre"], kept_filled=kept_filled, kept_prefill=kept_prefill,
                   fhr_repaired=repaired, fhr_repair_exact=exact, fhr_raw_match=match)
        out = {k: np.asarray(v)[keep] for k, v in out.items()}
        return dict(path=path, subgroup=subgroup,
                    epoch=r["domain_start"][keep].astype(np.float32), error="", **out)
    except Exception as exc:  # noqa: BLE001 -- one bad record must not stop the build
        return dict(path=path, subgroup=subgroup, epoch=np.zeros(0, np.float32), error=str(exc))


def create_file(path: str, records_base_path: str, seed: int) -> h5py.File:
    """Create an empty minimal HDF5 file; refuses to overwrite an existing one."""
    h5f = h5py.File(path, "w-")
    for name, (shape, dtype) in FIELDS.items():
        is_str = name == "guid"
        h5f.create_dataset(
            name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype,
            chunks=(CHUNK_ROWS, *shape),
            compression=None if is_str else "gzip", shuffle=not is_str,
        )
    h5f.attrs["sampling_rate_hz"] = cnp.SAMPLING_RATE_HZ
    h5f.attrs["up_shift_secs"] = -20
    h5f.attrs["min_domain_start"] = cnp.MIN_DOMAIN_START_DATASET
    h5f.attrs["records_base_path"] = records_base_path
    h5f.attrs["seed"] = seed
    h5f.attrs["class_label_names"] = "1=healthy, 2=acidosis, 3=hie"
    return h5f


def append_record(h5f: h5py.File, rec: Dict[str, Any], labor_map: Dict[str, float],
                  ss_map: Dict[str, float]) -> int:
    """Append one record's segments; returns how many were written."""
    n = len(rec["epoch"])
    if n == 0:
        return 0
    guid = os.path.splitext(os.path.basename(rec["path"]))[0]
    key = cnp._normalize_guid(guid)
    target, cs, bg = cnp.SUBGROUP_META[rec["subgroup"]]
    epoch = rec["epoch"]
    rows = {
        **{k: rec[k] for k in SIGNAL_FIELDS}, "epoch": epoch,
        "guid": [guid] * n,
        "class_label": np.full(n, target, np.uint8),
        "cs_label": np.full(n, cs, np.uint8),
        "bg_label": np.full(n, bg, np.uint8),
        "time_from_labor_onset": epoch - labor_map.get(key, math.nan),
        "second_stage_onset": epoch - ss_map.get(key, math.nan),
    }
    for name, values in rows.items():
        dset = h5f[name]
        dset.resize(dset.shape[0] + n, axis=0)
        dset[-n:] = values
    return n


def build(records_base_path: str, output_dir: str, tlo_csv_path: Optional[str], seed: int,
          num_workers: int, verbose: bool) -> None:
    """Write ``acidosis.hdf5``, ``hie.hdf5``, ``healthy.hdf5`` and the selection CSV."""
    os.makedirs(output_dir, exist_ok=True)
    labor_map, ss_map = (cnp.load_csv_metadata(tlo_csv_path) if tlo_csv_path else ({}, {}))

    jobs: Dict[str, List[Tuple[str, str]]] = {name: [] for name in CLASS_NAMES.values()}
    for folder, subgroup in cnp.FOLDER_TO_SUBGROUP.items():
        cls = CLASS_NAMES[cnp.SUBGROUP_META[subgroup][0]]
        jobs[cls] += [(p, subgroup) for p in cnp._discover_mat_files(records_base_path, folder)]
    for cls, items in jobs.items():
        logger.info(f"found {len(items)} {cls} records")

    rows: List[Dict[str, Any]] = []
    pool = ProcessPoolExecutor(num_workers) if num_workers > 1 else None
    reader = partial(read_record, verbose=verbose)

    def run(cls: str, batch: List[Tuple[str, str]], h5f: h5py.File) -> int:
        """Read *batch* in order, append to *h5f*; returns how many GUIDs gave segments."""
        n_ok = 0
        results = pool.map(reader, batch) if pool else map(reader, batch)  # in order
        for rec in tqdm(results, total=len(batch), desc=cls):
            n_seg = append_record(h5f, rec, labor_map, ss_map)
            n_ok += n_seg > 0
            rows.append(dict(guid=os.path.splitext(os.path.basename(rec["path"]))[0], class_=cls,
                             subgroup=rec["subgroup"], n_segments=n_seg, error=rec["error"],
                             record_path=rec["path"]))
        return n_ok

    try:
        n_unhealthy = 0
        for cls in ("acidosis", "hie"):
            with create_file(os.path.join(output_dir, f"{cls}.hdf5"), records_base_path, seed) as h5f:
                n_unhealthy += run(cls, jobs[cls], h5f)

        healthy = sorted(jobs["healthy"])
        random.Random(seed).shuffle(healthy)
        n_healthy, pos = 0, 0
        with create_file(os.path.join(output_dir, "healthy.hdf5"), records_base_path, seed) as h5f:
            # Read only the shortfall each round, so the file stops at exactly n_unhealthy GUIDs.
            while n_healthy < n_unhealthy and pos < len(healthy):
                batch = healthy[pos:pos + n_unhealthy - n_healthy]
                pos += len(batch)
                n_healthy += run("healthy", batch, h5f)
        if n_healthy < n_unhealthy:
            logger.warning(f"only {n_healthy} healthy GUIDs gave segments (wanted {n_unhealthy})")
    finally:
        if pool:
            pool.shutdown()
        df = pd.DataFrame(rows).rename(columns={"class_": "class"})
        df.to_csv(os.path.join(output_dir, SELECTION_CSV), index=False)

    ok = df[df["n_segments"] > 0]
    logger.info("GUIDs / segments written:\n" + ok.groupby("class")["n_segments"]
                .agg(["count", "sum"]).to_string())
    logger.info(f"{int((df['error'] != '').sum())} records errored, "
                f"{int(((df['error'] == '') & (df['n_segments'] == 0)).sum())} gave no segment")


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser. No defaults here: :data:`RUN_ARGS` fills every flag not given."""
    parser = argparse.ArgumentParser(
        prog="python hdf5_dataset/create_minimal_hdf5.py",
        description="Raw-signal-only HDF5 files: all acidosis, all HIE, matched healthy subset.",
    )
    parser.add_argument("--records_base_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--tlo_csv_path", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--verbose", action="store_true", default=None)
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("records_base_path", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{k}={values[k]!r} (from {sources[k]})" for k in sorted(values)))
    build(
        records_base_path=values["records_base_path"],
        output_dir=values["output_dir"],
        tlo_csv_path=values["tlo_csv_path"],
        seed=cnp.RANDOM_STATE if values["seed"] is None else values["seed"],
        num_workers=values["num_workers"] or min(os.cpu_count() or 1, 8),
        verbose=bool(values["verbose"]),
    )
    return 0


#: **Edit the values below and press Run.** A command-line flag overrides its key.
RUN_ARGS: Dict[str, Any] = {
    # str -- root directory holding the StudyGroup subfolders. Required.
    "records_base_path": r"/data/deid/datafabric/fetal-heart-tracing/StudyGroup2022_v4/",
    # str -- NEW directory for acidosis.hdf5, hie.hdf5, healthy.hdf5. Existing files are never
    #        overwritten (the build refuses).  Required.
    "output_dir": r"/data1/fetal-heart-tracing/HDF5_Datasets/minimal_raw_signals",
    # str | None -- CSV with trace_guid, labor_onset_hours, second_stage_onset_hours.
    #        (None -> time_from_labor_onset and second_stage_onset are NaN)
    "tlo_csv_path": r"/data/deid/datafabric/fetal-heart-tracing/processing/labor_onset/labor_onset_by_dataset.csv",
    # int | None -- seed of the healthy draw.  (None -> 42, the pipeline's RANDOM_STATE)
    "seed": None,
    # int | None -- parallel record readers; 1 reads in this process.  (None -> min(cpu_count, 8))
    "num_workers": None,
    # bool | None -- show MIMO output per record.  (None -> False)
    "verbose": False,
}


if __name__ == "__main__":
    sys.exit(_cli())
