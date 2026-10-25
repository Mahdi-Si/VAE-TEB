r"""Build the k-fold and pretraining HDF5 datasets from a minimization fold folder.

``build_minimization_folds.py`` (run in the ``em_analysis`` checkout) writes a fold folder: the
partition of every classification GUID in every fold (``fold_partitions.csv``). This module turns
that folder into the two datasets ``create_new_pipeline.py`` builds, with three changes.

1. **The folds come from the fold folder**, not from Steps 1-3 of the pipeline. The folder is also
   written out as ``classification_dataset_records.pickle`` in the pipeline's own format, so every
   tool that reads that pickle works unchanged.
2. **Pretraining takes every healthy GUID outside classification**: the two BG subgroups and the
   two no-BG subgroups, each in its own directory. The pipeline took the BG subgroups only.
3. **Each GUID is transformed once.** The k-fold build writes one *unit* file per (block,
   subgroup) and then assembles every partition file by copying whole units. Block $b$ holds the
   pool GUIDs with ``fold_id`` $= b - 1$; the unit ``added_test`` holds the extra test GUIDs.
   ``create_new_pipeline.py`` transforms every pool GUID once per fold, so $K$ times.

The pipeline's own functions (``create_hdf5_for_masks`` and
``create_hdf5_dataset_from_records_list``) write every unit and every pretraining shard, so the
file layout and the attributes are those of ``create_new_pipeline.py``. The transform jobs run in
parallel, one process per entry of ``devices``. Every process computes the masks on the CPU, so
all files carry bit-identical attributes; the build checks this before it assembles anything.

Output layout::

    <output_base_path>/
        classification_dataset_records.pickle
        build_manifest.json
        pretraining_guids.csv
        pre_training_dataset/
            train/<subgroup>/shard_000.hdf5 ...
            test/<subgroup>/shard_000.hdf5 ...
            dataset_files.yaml          # the vae_train_datasets / vae_test_datasets lists
        k_fold_cross_validation_dataset/
            fold_<k>/{train,val,test}/<subgroup>.hdf5
            _units/                     # deleted after assembly unless keep_units

In holdout mode the fixed test set goes to ``k_fold_cross_validation_dataset/test/`` once and the
folds hold ``train`` and ``val`` only, as in ``create_new_pipeline.py``.

A rerun with the same inputs skips every file that is already complete. A file is written as
``<name>.partial`` and renamed only when it is complete, and a complete file is recognized by its
``source_guid_digest`` attribute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

#: Repository root: ``hdf5_dataset/new_pipeline/build_fold_datasets.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the pipeline can be found.
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402

logger = logging.getLogger(__name__)

PICKLE_NAME = "classification_dataset_records.pickle"
MANIFEST_NAME = "build_manifest.json"
PRETRAIN_DIR = "pre_training_dataset"
KFOLD_DIR = "k_fold_cross_validation_dataset"
UNITS_DIR = "_units"
ADDED_TEST_UNIT = "added_test"
PARTIAL_SUFFIX = ".partial"
STEPS = ("pretraining", "kfold")
PRETRAIN_ELIGIBILITY = ("eligible_3h", "screened")

#: Rows per read when a partition file is assembled from its units: about 54 MB at the
#: production widths.
COPY_BATCH_ROWS = 256


def _pipeline() -> Any:
    """Import ``create_new_pipeline`` on first use.

    Lazy, so that the parser and ``RUN_ARGS`` import without ``torch`` and without the
    production-only packages the pipeline needs.

    Returns:
        The ``create_new_pipeline`` module.
    """
    from hdf5_dataset.new_pipeline import create_new_pipeline

    return create_new_pipeline


@dataclass(frozen=True)
class Job:
    """One HDF5 file that the transform writes.

    Attributes:
        kind: ``'pretraining'`` for a pretraining shard, ``'unit'`` for a k-fold unit.
        group: The split (``'train'`` / ``'test'``) of a shard, or the unit name of a unit.
        subgroup: The subgroup of every record in the file.
        path: The final path of the file.
        records: The ``.mat`` paths, in the order the file is written.
    """

    kind: str
    group: str
    subgroup: str
    path: str
    records: Tuple[str, ...]


# =============================================================================
# Inputs
# =============================================================================
def _as_bool(values: pd.Series) -> pd.Series:
    """Parse a CSV column of ``True`` / ``False`` (or ``1`` / ``0``) into booleans.

    Raises:
        ValueError: If a value is neither.
    """
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        bad = values[parsed.isna()].unique()[:5].tolist()
        raise ValueError(f"column {values.name!r} holds values that are not booleans: {bad}")
    return parsed.astype(bool)


def _stem(path: str) -> str:
    """Return the GUID of a ``.mat`` path: its file name without the extension."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fold_folder(fold_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Read and check the fold folder.

    Args:
        fold_dir: The ``output_dir`` of ``build_minimization_folds.py``.

    Returns:
        ``(partitions, assignment, run_config)``: ``fold_partitions.csv``,
        ``fold_assignment.csv`` and ``run_config.json``.

    Raises:
        ValueError: If a column is missing, the test mode is unknown, or a GUID's
            ``record_path`` does not end in that GUID.
    """
    folder = Path(fold_dir)
    partitions = pd.read_csv(folder / "fold_partitions.csv",
                             dtype={"tracing_guid": str, "record_path": str, "subgroup": str})
    assignment = pd.read_csv(folder / "fold_assignment.csv",
                             dtype={"tracing_guid": str, "subgroup": str})
    with open(folder / "run_config.json", encoding="utf-8") as f:
        run_config = json.load(f)

    needed = {"fold", "partition", "subgroup", "tracing_guid", "record_path", "extra_test"}
    if needed - set(partitions.columns):
        raise ValueError(f"fold_partitions.csv lacks {sorted(needed - set(partitions.columns))}")
    if {"tracing_guid", "fold_id"} - set(assignment.columns):
        raise ValueError("fold_assignment.csv lacks tracing_guid or fold_id")
    if run_config.get("test_mode") not in ("augmented", "holdout"):
        raise ValueError(f"run_config.json test_mode={run_config.get('test_mode')!r}")
    partitions["extra_test"] = _as_bool(partitions["extra_test"])
    partitions["fold"] = partitions["fold"].astype(int)
    wrong = partitions[partitions["record_path"].map(_stem) != partitions["tracing_guid"]]
    if len(wrong):
        raise ValueError(f"{len(wrong)} rows whose record_path does not end in the GUID, e.g. "
                         f"{wrong[['tracing_guid', 'record_path']].head(3).values.tolist()}")
    return partitions, assignment, run_config


def unit_members(partitions: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Give every classification GUID its unit.

    Args:
        partitions: ``fold_partitions.csv``.
        assignment: ``fold_assignment.csv``.

    Returns:
        One row per GUID with ``tracing_guid``, ``subgroup``, ``record_path`` and ``unit``
        (``block_01`` ... ``block_K`` or :data:`ADDED_TEST_UNIT`).

    Raises:
        ValueError: If a GUID has two paths or two subgroups, is both a pool GUID and an extra
            test GUID, or is a pool GUID without a ``fold_id``.
    """
    columns = ["tracing_guid", "subgroup", "record_path"]
    core = partitions.loc[~partitions["extra_test"], columns].drop_duplicates()
    extra = partitions.loc[partitions["extra_test"], columns].drop_duplicates()
    both = set(core["tracing_guid"]) & set(extra["tracing_guid"])
    if both:
        raise ValueError(f"{len(both)} GUIDs are pool GUIDs and extra test GUIDs, e.g. "
                         f"{sorted(both)[:3]}")
    fold_id = assignment.drop_duplicates("tracing_guid").set_index("tracing_guid")["fold_id"]
    core = core.assign(unit=core["tracing_guid"].map(fold_id))
    if core["unit"].isna().any():
        raise ValueError(f"pool GUIDs without a fold_id, e.g. "
                         f"{core.loc[core['unit'].isna(), 'tracing_guid'].head(3).tolist()}")
    core["unit"] = core["unit"].map(lambda f: f"block_{int(f) + 1:02d}")
    members = pd.concat([core, extra.assign(unit=ADDED_TEST_UNIT)], ignore_index=True)
    dup = members["tracing_guid"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"GUIDs with two paths or two subgroups, e.g. "
                         f"{members.loc[dup, 'tracing_guid'].head(3).tolist()}")
    return members.sort_values(["unit", "subgroup", "tracing_guid"]).reset_index(drop=True)


def partition_units(partitions: pd.DataFrame,
                    members: pd.DataFrame) -> Dict[Tuple[int, str, str], List[str]]:
    """Map every partition file to the units it is copied from.

    Args:
        partitions: ``fold_partitions.csv``.
        members: Output of :func:`unit_members`.

    Returns:
        ``{(fold, partition, subgroup): [unit, ...]}``.

    Raises:
        ValueError: If a partition holds only part of a unit, so that copying whole units
            would add GUIDs to it.
    """
    unit_of = members.set_index("tracing_guid")["unit"]
    unit_guids = members.groupby(["unit", "subgroup"])["tracing_guid"].apply(frozenset)
    out: Dict[Tuple[int, str, str], List[str]] = {}
    for (fold, part, sg), grp in partitions.groupby(["fold", "partition", "subgroup"]):
        guids = set(grp["tracing_guid"])
        units = sorted(set(unit_of.loc[list(guids)]))
        covered = frozenset().union(*(unit_guids[(u, sg)] for u in units))
        if covered != guids:
            raise ValueError(f"fold {fold} {part} {sg} holds part of a unit: "
                             f"{len(covered - guids)} unit GUIDs are not in it")
        out[(int(fold), str(part), str(sg))] = units
    return out


def build_cv_result(partitions: pd.DataFrame, test_mode: str) -> Dict[str, Any]:
    """Write the folds in the format of ``create_cv_splits`` in ``create_new_pipeline.py``.

    Augmented mode: ``folds[fold_k][test]`` holds the pool test GUIDs and then the extra test
    GUIDs, and ``test_augmentation`` holds the extra test GUIDs. Holdout mode: the folds hold
    ``train`` and ``val`` only, and ``test`` holds the fixed test set.

    Args:
        partitions: ``fold_partitions.csv``.
        test_mode: ``'augmented'`` or ``'holdout'``.

    Returns:
        The ``cv_result`` dict.
    """
    ordered = partitions.sort_values(["fold", "partition", "subgroup", "extra_test",
                                      "tracing_guid"])
    parts = ("train", "val", "test") if test_mode == "augmented" else ("train", "val")
    folds: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for fold in sorted(ordered["fold"].unique()):
        rows = ordered[ordered["fold"] == fold]
        folds[f"fold_{fold}"] = {
            part: {sg: grp["record_path"].tolist()
                   for sg, grp in rows[rows["partition"] == part].groupby("subgroup")}
            for part in parts}
    extra = ordered[ordered["extra_test"]].drop_duplicates("tracing_guid")
    extra_dict = {sg: grp["record_path"].tolist() for sg, grp in extra.groupby("subgroup")}
    key = "test_augmentation" if test_mode == "augmented" else "test"
    return {"test_mode": test_mode, "folds": folds, key: extra_dict}


def select_pretraining(screening: pd.DataFrame, classification_guids: Set[str],
                       healthy: Sequence[str], eligibility: str) -> pd.DataFrame:
    """Return the healthy GUIDs that pretraining takes.

    Args:
        screening: ``guid_screening_results.csv`` of the pipeline run the pool came from. It
            holds every ``.mat`` file of every subgroup folder.
        classification_guids: Every GUID in the fold folder, pool and extra test.
        healthy: The healthy subgroups.
        eligibility: ``'eligible_3h'`` keeps the GUIDs with at least 3 h of valid signal in the
            last 6 h, the rule the classification pool and the old pretraining set used.
            ``'screened'`` keeps every GUID the screening read without an error; the
            per-segment quality filter of the transform still applies.

    Returns:
        One row per GUID with ``guid``, ``subgroup`` and ``record_path``, sorted.

    Raises:
        ValueError: On an unknown *eligibility* or a GUID listed twice.
    """
    if eligibility not in PRETRAIN_ELIGIBILITY:
        raise ValueError(f"pretrain_eligibility={eligibility!r}; use one of "
                         f"{PRETRAIN_ELIGIBILITY}")
    df = screening.assign(guid=screening["record_path"].map(_stem))
    keep = ~_as_bool(df["error"]) & df["subgroup"].isin(list(healthy))
    if eligibility == "eligible_3h":
        keep &= _as_bool(df["eligible_3h"])
    pool = df.loc[keep & ~df["guid"].isin(classification_guids),
                  ["guid", "subgroup", "record_path"]]
    if pool["guid"].duplicated().any():
        raise ValueError(f"the screening CSV lists GUIDs twice, e.g. "
                         f"{pool.loc[pool['guid'].duplicated(), 'guid'].head(3).tolist()}")
    return pool.sort_values(["subgroup", "guid"]).reset_index(drop=True)


def split_and_shard(pool: pd.DataFrame, test_fraction: float, guids_per_shard: int,
                    seed: int) -> pd.DataFrame:
    r"""Split each subgroup into train and test and cut each split into shards.

    Each subgroup is shuffled with its own generator, seeded by the string
    ``f"{seed}:{subgroup}"``, so the split of one subgroup does not change when another one
    changes. A subgroup of $n$ GUIDs gives $\lfloor (1 - f) n \rfloor$ train GUIDs, the rule of
    ``create_new_pipeline.py``. A split of $m$ GUIDs is cut into
    $\lceil m / s \rceil$ shards of near-equal size, where $s$ is ``guids_per_shard``.

    Args:
        pool: Output of :func:`select_pretraining`.
        test_fraction: The test fraction $f$.
        guids_per_shard: The most GUIDs in one shard, $s$.
        seed: Base seed.

    Returns:
        ``pool`` with the columns ``split`` and ``shard`` added.
    """
    frames = []
    for sg, grp in pool.groupby("subgroup", sort=True):
        order = grp.sort_values("guid").index.tolist()
        random.Random(f"{seed}:{sg}").shuffle(order)
        n_train = int(len(order) * (1.0 - test_fraction))
        for split, idx in (("train", order[:n_train]), ("test", order[n_train:])):
            if not idx:
                continue
            n_shards = max(1, math.ceil(len(idx) / guids_per_shard))
            for shard, chunk in enumerate(np.array_split(np.asarray(idx), n_shards)):
                frames.append(pool.loc[chunk].assign(split=split, shard=shard))
    if not frames:
        return pool.assign(split=pd.Series(dtype=str), shard=pd.Series(dtype=int))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Plan
# =============================================================================
def unit_path(output_base_path: str, unit: str, subgroup: str) -> str:
    """Return the path of one k-fold unit file."""
    return str(Path(output_base_path) / KFOLD_DIR / UNITS_DIR / unit / f"{subgroup}.hdf5")


def plan_jobs(output_base_path: str, members: pd.DataFrame, pretraining: pd.DataFrame,
              steps: Sequence[str]) -> List[Job]:
    """List every file the transform writes.

    Args:
        output_base_path: The output directory.
        members: Output of :func:`unit_members`.
        pretraining: Output of :func:`split_and_shard`.
        steps: Any of :data:`STEPS`.

    Returns:
        The jobs, pretraining shards first.
    """
    jobs: List[Job] = []
    if "pretraining" in steps:
        root = Path(output_base_path) / PRETRAIN_DIR
        for (split, sg, shard), grp in pretraining.groupby(["split", "subgroup", "shard"]):
            jobs.append(Job("pretraining", str(split), str(sg),
                            str(root / str(split) / str(sg) / f"shard_{int(shard):03d}.hdf5"),
                            tuple(grp["record_path"])))
    if "kfold" in steps:
        for (unit, sg), grp in members.groupby(["unit", "subgroup"]):
            jobs.append(Job("unit", str(unit), str(sg), unit_path(output_base_path, unit, sg),
                            tuple(grp["record_path"])))
    return jobs


# =============================================================================
# Transform jobs
# =============================================================================
_STATE: Dict[str, Any] = {}


def _prepare_state(device: Optional[str], settings: Mapping[str, Any]) -> None:
    """Load what every job of this process needs: the masks, the CSV maps and the device.

    The masks are computed on the CPU in every process, whatever the transform device, so
    that every file of the build carries the same attributes bit for bit. The loader refuses
    files whose causal attributes differ even in the last bit.
    """
    import torch

    cnp = _pipeline()
    masks = cnp.compute_scattering_masks(
        cnp.SIGNAL_LENGTH, scattering_T=16, device=torch.device("cpu"),
        transform=settings["transform"], leg_alignment=settings["leg_alignment"],
        phase_operator=settings["phase_operator"])
    labor_onset_map, second_stage_map = cnp.load_csv_metadata(settings["tlo_csv_path"],
                                                              verbose=False)
    _STATE.update(cnp=cnp, masks=masks, settings=dict(settings),
                  device=None if device is None else torch.device(device),
                  labor_onset_map=labor_onset_map, second_stage_map=second_stage_map)


def _init_worker(device_queue: Any, settings: Mapping[str, Any]) -> None:
    """Pool initializer: take one device from the queue and prepare this process."""
    device = device_queue.get()
    logging.basicConfig(level=logging.INFO, format=f"[{device}] %(message)s")
    _prepare_state(device, settings)


def _run_job(job: Job) -> Dict[str, Any]:
    """Transform the records of one job into its HDF5 file.

    Returns:
        A summary: path, record count, GUIDs with at least one segment, segment count, the
        records that failed and the run time.
    """
    import h5py

    cnp, masks, settings = _STATE["cnp"], _STATE["masks"], _STATE["settings"]
    final = Path(job.path)
    partial = final.with_name(final.name + PARTIAL_SUFFIX)
    partial.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cnp.create_hdf5_for_masks(str(partial), masks, len_sequence=cnp.SEQUENCE_LENGTH,
                              records_list=list(job.records),
                              source_pickle_path=settings["source_pickle_path"])
    target, cs, bg = cnp.SUBGROUP_META[job.subgroup]
    errors = cnp.create_hdf5_dataset_from_records_list(
        hdf5_path=str(partial), records_list=list(job.records), cs_label=cs, bg_label=bg,
        pre_defined_target=target, precomputed_masks=masks,
        labor_onset_map=_STATE["labor_onset_map"], second_stage_map=_STATE["second_stage_map"],
        base_block_size=cnp.BASE_BLOCK_SIZE, overlap_percentage=cnp.OVERLAP_PERCENTAGE,
        device=_STATE["device"], run_guid_analysis=False,
        scatter_batch_size=settings["scatter_batch_size"], verbose=settings["verbose"],
        transform=settings["transform"])
    with h5py.File(partial, "r") as f:
        n_segments = int(f["fhr"].shape[0])
        n_guids = len(set(f["guid"].asstr()[:]))
    os.replace(partial, final)
    return {"path": str(final), "kind": job.kind, "group": job.group, "subgroup": job.subgroup,
            "n_records": len(job.records), "n_guids_with_segments": n_guids,
            "n_segments": n_segments, "errors": list(errors),
            "seconds": round(time.time() - start, 1)}


def is_complete(path: str, digest: str, source_pickle_path: str) -> bool:
    """Return whether *path* is a finished file of this GUID set and this pickle.

    Args:
        path: The final path of the file.
        digest: ``guid_set_digest`` of the file's records.
        source_pickle_path: The pickle every file of the build records.
    """
    if not os.path.exists(path):
        return False
    import h5py

    with h5py.File(path, "r") as f:
        return (f.attrs.get("source_guid_digest") == digest
                and f.attrs.get("source_pickle_path") == str(source_pickle_path))


def run_jobs(jobs: Sequence[Job], settings: Mapping[str, Any],
             devices: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """Run every job that is not complete yet, largest first.

    With two or more ``devices`` each device gets its own process. A device may appear
    twice, which runs two processes on it; the MIMO preprocessing of a record is CPU work, so
    two processes per GPU can keep a GPU busier.

    Args:
        jobs: Output of :func:`plan_jobs`.
        settings: The transform settings and ``source_pickle_path``.
        devices: Torch devices, e.g. ``["cuda:0", "cuda:1"]``. ``None`` or one entry runs the
            jobs in this process.

    Returns:
        One summary per job that ran (see :func:`_run_job`).

    Raises:
        RuntimeError: If a job failed. The other jobs still run first, and a rerun repeats
            only the failed ones.
    """
    digest = _pipeline().guid_set_digest
    todo = [j for j in jobs if not is_complete(j.path, digest(list(j.records)),
                                               settings["source_pickle_path"])]
    logger.info(f"{len(jobs) - len(todo)} of {len(jobs)} files are complete; "
                f"{len(todo)} to build")
    todo.sort(key=lambda j: -len(j.records))
    results: List[Dict[str, Any]] = []
    failed: List[Tuple[str, str]] = []
    if not todo:
        return results
    if not devices or len(devices) == 1:
        _prepare_state(devices[0] if devices else None, settings)
        for i, job in enumerate(todo, 1):
            try:
                results.append(_run_job(job))
                _log_result(results[-1], i, len(todo))
            except Exception as e:  # noqa: BLE001 -- reported below, with the job's path
                logger.exception(f"FAILED {job.path}")
                failed.append((job.path, repr(e)))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            queue = manager.Queue()
            for device in devices:
                queue.put(device)
            with ProcessPoolExecutor(max_workers=len(devices), mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(queue, dict(settings))) as pool:
                futures = {pool.submit(_run_job, job): job for job in todo}
                for i, future in enumerate(as_completed(futures), 1):
                    job = futures[future]
                    try:
                        results.append(future.result())
                        _log_result(results[-1], i, len(todo))
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"FAILED {job.path}: {e!r}")
                        failed.append((job.path, repr(e)))
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(todo)} jobs failed; rerun to retry them: "
                           f"{failed[:5]}")
    return results


def _log_result(result: Mapping[str, Any], i: int, n: int) -> None:
    """Log one finished job."""
    logger.info(f"[{i}/{n}] {result['path']}: {result['n_records']} records, "
                f"{result['n_segments']} segments, {len(result['errors'])} errors, "
                f"{result['seconds']} s")


# =============================================================================
# Layout check and k-fold assembly
# =============================================================================
def _layout(path: str) -> Dict[str, Any]:
    """Return everything about a file that must agree across the build: the root attributes
    except ``source_guid_digest``, and each dataset's per-sample shape, dtype, chunks,
    compression and attributes."""
    import h5py

    with h5py.File(path, "r") as f:
        root = {k: v for k, v in f.attrs.items() if k != "source_guid_digest"}
        datasets = {name: {"shape": ds.shape[1:], "dtype": str(ds.dtype), "chunks": ds.chunks,
                           "compression": ds.compression, "attrs": dict(ds.attrs)}
                    for name, ds in f.items()}
    return {"root": root, "datasets": datasets}


def _first_difference(a: Any, b: Any, where: str = "") -> Optional[str]:
    """Return where two nested layouts first differ, or ``None`` if they are equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{where}: keys {sorted(set(a) ^ set(b))}"
        for key in a:
            found = _first_difference(a[key], b[key], f"{where}/{key}")
            if found:
                return found
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        nan_ok = a.dtype.kind in "fc" and b.dtype.kind in "fc"
        same = (a.shape == b.shape and a.dtype == b.dtype
                and np.array_equal(a, b, equal_nan=nan_ok))
        return None if same else f"{where}: arrays differ"
    return None if a == b else f"{where}: {a!r} != {b!r}"


def check_files_agree(paths: Sequence[str]) -> None:
    """Refuse a set of files whose layouts or attributes differ.

    The training loader requires this across every file it reads, and it compares the causal
    attributes bit for bit. Checking here finds a disagreement before hours of assembly and
    training, not after.

    Raises:
        ValueError: Naming the first file and the first attribute that differ.
    """
    reference, reference_path = None, ""
    for path in paths:
        layout = _layout(path)
        if reference is None:
            reference, reference_path = layout, path
            continue
        found = _first_difference(reference, layout)
        if found:
            raise ValueError(f"{path} disagrees with {reference_path} at {found}")


def _empty_like(template: Any, out: Any, digest: str) -> None:
    """Create in *out* every dataset of *template*, empty, with the same attributes."""
    for key, value in template.attrs.items():
        out.attrs[key] = value
    out.attrs["source_guid_digest"] = digest
    for name, ds in template.items():
        new = out.create_dataset(name, shape=(0,) + ds.shape[1:], maxshape=(None,) + ds.shape[1:],
                                 dtype=ds.dtype, chunks=ds.chunks, compression=ds.compression,
                                 compression_opts=ds.compression_opts, shuffle=ds.shuffle,
                                 fletcher32=ds.fletcher32)
        for key, value in ds.attrs.items():
            new.attrs[key] = value


def _append_rows(out: Any, source_path: str, batch_rows: int) -> Set[str]:
    """Append every row of *source_path* to the open file *out*; return the source's GUIDs."""
    import h5py

    with h5py.File(source_path, "r") as src:
        if set(src.keys()) != set(out.keys()):
            raise ValueError(f"{source_path} has datasets {sorted(src.keys())}, the partition "
                             f"file has {sorted(out.keys())}")
        n = int(src["fhr"].shape[0])
        start = int(out["fhr"].shape[0])
        for name in out:
            out[name].resize((start + n,) + out[name].shape[1:])
        for a in range(0, n, batch_rows):
            b = min(a + batch_rows, n)
            for name in out:
                rows = src[name].asstr()[a:b] if name == "guid" else src[name][a:b]
                out[name][start + a:start + b] = rows
        return set(src["guid"].asstr()[:]) if n else set()


@dataclass(frozen=True)
class AssemblyTask:
    """One partition file and the unit files it is copied from.

    Attributes:
        path: The partition file.
        sources: Its unit files, in copy order.
        digest: ``guid_set_digest`` of the partition's records, the value
            ``create_new_pipeline.py`` writes as ``source_guid_digest``.
        guids: The partition's GUIDs.
    """

    path: str
    sources: Tuple[str, ...]
    digest: str
    guids: frozenset


def assemble_file(task: AssemblyTask, source_pickle_path: str,
                  batch_rows: int = COPY_BATCH_ROWS) -> Dict[str, Any]:
    """Write one partition file as the concatenation of its unit files.

    The empty file copies the first unit's datasets and attributes, which
    ``create_hdf5_for_masks`` wrote; only ``source_guid_digest`` changes, to the digest of the
    partition's own records. Needs ``h5py`` only, so that the assembly processes stay light.

    Args:
        task: The partition file to write.
        source_pickle_path: The pickle every file of the build records.
        batch_rows: Rows per read.

    Returns:
        A summary of the file.

    Raises:
        ValueError: If the row count or the GUID set differs from the sum of the units, or a
            GUID is outside the partition.
    """
    import h5py

    dst, sources, digest = task.path, task.sources, task.digest
    if is_complete(dst, digest, source_pickle_path):
        return {"path": dst, "skipped": True}
    partial = dst + PARTIAL_SUFFIX
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    expected_rows, guids = 0, set()
    with h5py.File(sources[0], "r") as template, h5py.File(partial, "w", libver="latest") as out:
        _empty_like(template, out, digest)
        for source in sources:
            with h5py.File(source, "r") as src:
                expected_rows += int(src["fhr"].shape[0])
            guids |= _append_rows(out, source, batch_rows)
        n_rows = int(out["fhr"].shape[0])
        stored = set(out["guid"].asstr()[:]) if n_rows else set()
    if n_rows != expected_rows or stored != guids or not stored <= task.guids:
        raise ValueError(f"{dst}: {n_rows} rows against {expected_rows} in the units, or GUIDs "
                         f"outside the partition")
    os.replace(partial, dst)
    return {"path": dst, "n_units": len(sources), "n_segments": n_rows,
            "n_guids_with_segments": len(stored), "skipped": False}


def plan_assembly(output_base_path: str, partitions: pd.DataFrame, members: pd.DataFrame,
                  test_mode: str, digest: Callable[[List[str]], str]) -> List[AssemblyTask]:
    """List every partition file with its unit files.

    Args:
        output_base_path: The output directory.
        partitions: ``fold_partitions.csv``.
        members: Output of :func:`unit_members`.
        test_mode: ``'augmented'`` or ``'holdout'``.
        digest: ``guid_set_digest`` of ``create_new_pipeline.py``.

    Returns:
        One task per partition file. In holdout mode the fixed test set appears once, under
        ``k_fold_cross_validation_dataset/test/``.
    """
    kfold = Path(output_base_path) / KFOLD_DIR
    units = partition_units(partitions, members)
    records = partitions.groupby(["fold", "partition", "subgroup"])["record_path"].apply(list)
    targets: List[Tuple[Path, List[str], List[str]]] = []
    for (fold, part, sg), unit_names in units.items():
        if test_mode == "holdout" and part == "test":
            continue
        targets.append((kfold / f"fold_{fold}" / part / f"{sg}.hdf5",
                        [unit_path(output_base_path, u, sg) for u in unit_names],
                        list(records[(fold, part, sg)])))
    if test_mode == "holdout":
        extra = members[members["unit"] == ADDED_TEST_UNIT]
        for sg, grp in extra.groupby("subgroup"):
            targets.append((kfold / "test" / f"{sg}.hdf5",
                            [unit_path(output_base_path, ADDED_TEST_UNIT, str(sg))],
                            grp["record_path"].tolist()))
    return [AssemblyTask(str(path), tuple(sources), digest(recs),
                         frozenset(_stem(r) for r in recs))
            for path, sources, recs in targets]


def assemble_kfold(tasks: Sequence[AssemblyTask], source_pickle_path: str,
                   workers: int) -> List[Dict[str, Any]]:
    """Write every partition file, in *workers* processes (CPU work: LZF and disk).

    Returns:
        One summary per partition file.
    """
    if workers <= 1:
        return [assemble_file(t, source_pickle_path) for t in tasks]
    import multiprocessing as mp

    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(assemble_file, t, source_pickle_path) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if i % 20 == 0 or i == len(futures):
                logger.info(f"assembled {i}/{len(futures)} partition files")
    return results


# =============================================================================
# Records
# =============================================================================
def write_dataset_files(path: Path, jobs: Sequence[Job]) -> None:
    """Write the pretraining file lists in the config's own keys, ready to paste."""
    lines = ["# Paste into the training config. stat_path must be computed over ALL of these",
             "# files, train and test alike (hdf5_dataset/calculate_dataset_stats.py)."]
    for key, split in (("vae_train_datasets", "train"), ("vae_test_datasets", "test")):
        lines.append(f"{key}:")
        lines += [f"  - {j.path}" for j in sorted(jobs, key=lambda j: j.path)
                  if j.kind == "pretraining" and j.group == split]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_table(members: pd.DataFrame, pretraining: pd.DataFrame) -> pd.DataFrame:
    """GUIDs per subgroup: pool, extra test, pretraining train / test, and shards."""
    pool = members[members["unit"] != ADDED_TEST_UNIT].groupby("subgroup").size()
    extra = members[members["unit"] == ADDED_TEST_UNIT].groupby("subgroup").size()
    table = pd.DataFrame({"pool": pool, "extra_test": extra})
    for split in ("train", "test"):
        rows = pretraining[pretraining["split"] == split]
        table[f"pretrain_{split}"] = rows.groupby("subgroup").size()
        table[f"shards_{split}"] = rows.groupby("subgroup")["shard"].nunique()
    return table.fillna(0).astype(int).sort_index()


def _check_output_dir(out: Path, fold_digest: str) -> None:
    """Refuse a non-empty output directory that this build did not start.

    Raises:
        FileExistsError: If the directory holds files but no manifest of this build, or a
            manifest of a different fold folder.
    """
    if not out.exists() or not any(out.iterdir()):
        return
    manifest = out / MANIFEST_NAME
    if not manifest.exists():
        raise FileExistsError(f"{out} is not empty and holds no {MANIFEST_NAME}. Choose a NEW "
                              f"output_base_path; this build never writes into an existing "
                              f"dataset.")
    with open(manifest, encoding="utf-8") as f:
        previous = json.load(f)
    if previous.get("fold_partitions_sha256") != fold_digest:
        raise FileExistsError(f"{out} holds a build of a different fold folder. Choose a NEW "
                              f"output_base_path.")


# =============================================================================
# Entry point
# =============================================================================
def main(fold_dir: Optional[str] = None, screening_csv_path: Optional[str] = None,
         tlo_csv_path: Optional[str] = None, output_base_path: Optional[str] = None,
         steps: Optional[Sequence[str]] = None, devices: Optional[Sequence[str]] = None,
         transform: Optional[str] = None, leg_alignment: Optional[str] = None,
         phase_operator: Optional[str] = None, scatter_batch_size: Optional[int] = None,
         verbose: Optional[bool] = None, pretrain_eligibility: Optional[str] = None,
         pretrain_test_fraction: Optional[float] = None,
         pretrain_guids_per_shard: Optional[int] = None, seed: Optional[int] = None,
         assembly_workers: Optional[int] = None, keep_units: Optional[bool] = None,
         dry_run: Optional[bool] = None) -> int:
    """Apply the defaults, plan the build, and run it.

    Args:
        fold_dir: The ``output_dir`` of ``build_minimization_folds.py``.
        screening_csv_path: ``guid_screening_results.csv`` of the pipeline run the pool came
            from.
        tlo_csv_path: The CSV with the labour-onset and second-stage columns.
        output_base_path: A NEW directory for the whole build.
        steps: Any of :data:`STEPS`. Default: both.
        devices: Torch devices for the transform, one process each. Default: one process on
            the pipeline's default device.
        transform: Wavelet bank, as in ``create_new_pipeline.py``. Default: ``two_sided``.
        leg_alignment: Causal phase-pair leg alignment. Default: ``none``.
        phase_operator: Causal phase-harmonic operator. Default: ``ratio_power_v0``.
        scatter_batch_size: Segments per forward pass. Default: 128.
        verbose: Per-record progress output. Default: off.
        pretrain_eligibility: See :func:`select_pretraining`. Default: ``eligible_3h``.
        pretrain_test_fraction: Test fraction of each pretraining subgroup. Default: 0.1.
        pretrain_guids_per_shard: The most GUIDs in one pretraining shard. Default: 5000.
        seed: Seed of the pretraining split. Default: 42.
        assembly_workers: Processes that assemble the partition files. Default: 4.
        keep_units: Keep ``_units/`` after the assembly. Default: off.
        dry_run: Write the pickle, the manifest and the GUID lists, then stop. Default: off.

    Returns:
        The process exit code.
    """
    cnp = _pipeline()
    steps = list(steps or STEPS)
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        raise ValueError(f"unknown steps {unknown}; use {STEPS}")
    transform = transform or cnp.TWO_SIDED
    leg_alignment = leg_alignment or cnp.LEG_ALIGNMENT_MODES[0]
    phase_operator = phase_operator or cnp.PHASE_OPERATOR_LEGACY
    cnp.validate_transform(transform)
    cnp.validate_leg_alignment(leg_alignment)
    cnp.validate_phase_operator(phase_operator)
    eligibility = pretrain_eligibility or PRETRAIN_ELIGIBILITY[0]
    test_fraction = 0.1 if pretrain_test_fraction is None else float(pretrain_test_fraction)
    per_shard = int(pretrain_guids_per_shard or 5000)
    seed = 42 if seed is None else int(seed)

    out = Path(output_base_path)
    fold_digest = _sha256(Path(fold_dir) / "fold_partitions.csv")
    _check_output_dir(out, fold_digest)
    out.mkdir(parents=True, exist_ok=True)

    partitions, assignment, run_config = load_fold_folder(fold_dir)
    test_mode = run_config["test_mode"]
    members = unit_members(partitions, assignment)
    tasks = plan_assembly(str(out), partitions, members, test_mode, cnp.guid_set_digest)

    screening = pd.read_csv(screening_csv_path, dtype={"guid": str, "record_path": str})
    healthy = sorted(cnp.HEALTHY_SUBGROUPS)
    unscreened = (set(members.loc[members["subgroup"].isin(healthy), "tracing_guid"])
                  - set(screening["record_path"].map(_stem)))
    if unscreened:
        logger.warning(f"{len(unscreened)} healthy classification GUIDs are not in the "
                       f"screening CSV, e.g. {sorted(unscreened)[:3]}; is it the same run?")
    pool = select_pretraining(screening, set(members["tracing_guid"]), healthy, eligibility)
    pretraining = split_and_shard(pool, test_fraction, per_shard, seed)

    pickle_path = out / PICKLE_NAME
    with open(pickle_path, "wb") as f:
        pickle.dump(build_cv_result(partitions, test_mode), f, protocol=pickle.HIGHEST_PROTOCOL)
    pretraining.to_csv(out / "pretraining_guids.csv", index=False)
    jobs = plan_jobs(str(out), members, pretraining, steps)
    if "pretraining" in steps:
        (out / PRETRAIN_DIR).mkdir(exist_ok=True)
        write_dataset_files(out / PRETRAIN_DIR / "dataset_files.yaml", jobs)

    summary = _summary_table(members, pretraining)
    settings = {"transform": transform, "leg_alignment": leg_alignment,
                "phase_operator": phase_operator, "tlo_csv_path": str(tlo_csv_path),
                "scatter_batch_size": int(scatter_batch_size or 128), "verbose": bool(verbose),
                "source_pickle_path": str(pickle_path)}
    manifest = {
        "fold_dir": str(fold_dir), "fold_partitions_sha256": fold_digest,
        "fold_run_config": run_config, "screening_csv_path": str(screening_csv_path),
        "screening_csv_sha256": _sha256(Path(screening_csv_path)), "steps": steps,
        "devices": list(devices or []), "settings": settings,
        "pretrain": {"eligibility": eligibility, "test_fraction": test_fraction,
                     "guids_per_shard": per_shard, "seed": seed},
        "guids_per_subgroup": summary.to_dict(orient="index"),
        "jobs": [{**asdict(j), "records": len(j.records)} for j in jobs],
    }
    with open(out / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(summary.to_string())
    print(f"{len(jobs)} transform jobs, {len(tasks) if 'kfold' in steps else 0} partition files.")
    if dry_run:
        print(f"dry_run: wrote {PICKLE_NAME}, {MANIFEST_NAME} and the GUID lists to {out}")
        return 0

    # A unit is needed only while a partition file it feeds is incomplete: a rerun after a
    # finished build, whose units are deleted, must not transform them again.
    pending = [t for t in tasks if not is_complete(t.path, t.digest, str(pickle_path))]
    needed = {source for t in pending for source in t.sources}
    jobs = [j for j in jobs if j.kind != "unit" or j.path in needed]
    results = run_jobs(jobs, settings, devices)
    check_files_agree(sorted(j.path for j in jobs))
    assembled: List[Dict[str, Any]] = []
    if "kfold" in steps:
        assembled = assemble_kfold(pending, str(pickle_path), int(assembly_workers or 4))
        if not keep_units and (out / KFOLD_DIR / UNITS_DIR).exists():
            shutil.rmtree(out / KFOLD_DIR / UNITS_DIR)
        if pending:
            _run_fold_eda(out / KFOLD_DIR, test_mode)
    # Every final file of the build against every other: what the training loader requires.
    finals = [j.path for j in jobs if j.kind == "pretraining"] + (
        [t.path for t in tasks] if "kfold" in steps else [])
    check_files_agree(sorted(finals))

    manifest["results"] = {"transform": results, "assembly": assembled}
    with open(out / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Build finished: {out}")
    return 0


def _run_fold_eda(kfold: Path, test_mode: str) -> None:
    """Run the pipeline's fold EDA on the first fold, as ``create_new_pipeline.py`` does."""
    try:
        from hdf5_dataset.new_pipeline.fold_eda_analysis import run_fold_eda

        test_dir = str(kfold / "test") if test_mode == "holdout" else None
        run_fold_eda(str(kfold / "fold_1"), str(kfold / "fold_1" / "fold_eda"), test_dir=test_dir)
    except Exception as e:  # noqa: BLE001 -- optional report, as in the pipeline
        logger.error(f"Fold EDA failed: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    No argument is ``required`` and none carries a default, for the reasons given in
    ``create_new_pipeline.build_parser``: :data:`RUN_ARGS` fills every gap, and :func:`main`
    applies the real defaults after the merge.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/new_pipeline/build_fold_datasets.py",
        description="Build the k-fold and pretraining HDF5 datasets from a fold folder.")
    p.add_argument("--fold-dir", dest="fold_dir")
    p.add_argument("--screening-csv-path", dest="screening_csv_path")
    p.add_argument("--tlo-csv-path", dest="tlo_csv_path")
    p.add_argument("--output-base-path", dest="output_base_path")
    p.add_argument("--steps", dest="steps", nargs="+", choices=STEPS)
    p.add_argument("--devices", dest="devices", nargs="+")
    p.add_argument("--transform", dest="transform")
    p.add_argument("--leg-alignment", dest="leg_alignment")
    p.add_argument("--phase-operator", dest="phase_operator")
    p.add_argument("--scatter-batch-size", dest="scatter_batch_size", type=int)
    p.add_argument("--verbose", dest="verbose", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--pretrain-eligibility", dest="pretrain_eligibility",
                   choices=PRETRAIN_ELIGIBILITY)
    p.add_argument("--pretrain-test-fraction", dest="pretrain_test_fraction", type=float)
    p.add_argument("--pretrain-guids-per-shard", dest="pretrain_guids_per_shard", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--assembly-workers", dest="assembly_workers", type=int)
    p.add_argument("--keep-units", dest="keep_units", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--dry-run", dest="dry_run", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.

    Raises:
        SystemExit: If a required path was supplied by neither route.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("fold_dir", "screening_csv_path", "tlo_csv_path",
                                        "output_base_path"))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values)))
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
#:
#: Run once with ``dry_run=True`` first: it writes the pickle, the manifest and
#: ``pretraining_guids.csv`` and prints the GUID counts and the job count, without any transform.
RUN_ARGS: Dict[str, Any] = {
    # ================================ FILL IN: the paths ================================
    # str -- the output_dir of build_minimization_folds.py (fold_partitions.csv, ...). Required.
    "fold_dir": r"/path/to/minimization_folds_k10",

    # str -- guid_screening_results.csv of the pipeline run the pool came from. Required.
    "screening_csv_path": r"/path/to/output_base_path/guid_screening_results.csv",

    # str -- CSV with the labour-onset and second-stage columns. Required.
    "tlo_csv_path": r"/path/to/complete_labor_onset.csv",

    # str -- a NEW directory for the whole build. Required. A rerun into the same directory
    #        with the same fold folder skips every finished file.
    "output_base_path": r"/data1/fetal-heart-tracing/HDF5_Datasets/minimization_folds_causal_int",

    # ============================= The variant to build =============================
    # Same values and meanings as RUN_ARGS in create_new_pipeline.py.
    # "two_sided" | "causal"                              (None -> "two_sided")
    "transform": "causal",
    # "none" | "envelope"                                 (None -> "none"; causal only)
    "leg_alignment": "envelope",
    # "ratio_power_v0" | "integer_harmonic_v1"            (None -> "ratio_power_v0"; causal only)
    "phase_operator": "integer_harmonic_v1",

    # ================================ What to build ================================
    # list[str] | None -- any of "pretraining", "kfold".  (None -> both)
    "steps": None,

    # list[str] | None -- one transform process per entry, e.g. ["cuda:0", "cuda:1", "cuda:2"].
    #        A device may repeat, which runs two processes on it.
    #        (None -> one process on the first CUDA device, else the CPU)
    "devices": None,

    # "eligible_3h" | "screened"    (None -> "eligible_3h")
    #   "eligible_3h" : >= 3 h valid signal in the last 6 h, the classification pool's rule.
    #   "screened"    : every GUID the screening read without an error.
    "pretrain_eligibility": None,

    # float | None -- test fraction of each pretraining subgroup.  (None -> 0.1)
    "pretrain_test_fraction": None,

    # int | None -- the most GUIDs in one pretraining shard.  (None -> 5000)
    "pretrain_guids_per_shard": None,

    # int | None -- seed of the pretraining train/test split.  (None -> 42)
    "seed": None,

    # ========================= Everything below has a default =========================
    # int | None -- segments per forward pass of the transform.  (None -> 128)
    "scatter_batch_size": 128,
    # bool | None -- per-record progress output.  (None -> False)
    "verbose": False,
    # int | None -- processes that copy units into partition files.  (None -> 4)
    "assembly_workers": None,
    # bool | None -- keep k_fold_cross_validation_dataset/_units after the assembly.
    #        (None -> False)
    "keep_units": None,
    # bool | None -- plan only: pickle, manifest and GUID lists, no transform.  (None -> False)
    "dry_run": True,
}


if __name__ == "__main__":
    sys.exit(_cli())
