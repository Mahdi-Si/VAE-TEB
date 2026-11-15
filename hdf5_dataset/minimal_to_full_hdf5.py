r"""Rebuild the full k-fold HDF5 datasets from the files of ``create_minimal_hdf5.py``.

No ``.mat`` file and no MIMO package are needed: the stored segments stand in for the records.

The build is ``new_pipeline/build_fold_datasets.py`` run on a fold folder made here, with one
change. Its own functions plan the units, write each one with the pipeline's
``create_hdf5_dataset_from_records_list``, assemble the partition files and check that they all
agree, so the output has the same layout, datasets and attributes::

    <output_base_path>/
        fold_folder/            fold_partitions.csv, fold_assignment.csv, run_config.json
        tlo_from_minimal.csv    labour onset and second stage, recovered from the files
        classification_dataset_records.pickle
        build_manifest.json
        k_fold_cross_validation_dataset/fold_<k>/{train,val,test}/<subgroup>.hdf5

The one change: the pipeline reads each record through ``EarlyMaestraMimoAdaptor``, and here that
name is pointed at :class:`MinimalAdaptor`, which returns the GUID's stored segments. The pipeline
then runs its usual steps on them. Sanitising, de-duplication, the gates and the post-delivery cut
already ran when the minimal file was written, so they keep every segment, and the stored
``fhr``/``up`` are the signals the transform would have read from the ``.mat`` file.

Folds: each subgroup is split into ``n_folds`` blocks, stratified by labour duration like
``create_cv_splits`` when every duration bin has ``n_folds`` GUIDs, at random otherwise. A GUID in
block $g$ is in the test partition of fold $g + 1$, the val partition of fold
$((g + 1) \bmod K) + 1$ and the train partition of every other fold, the convention of
``build_minimization_folds.py`` (80 / 10 / 10 at $K = 10$). Every healthy GUID of the minimal
files is in the classification pool, so there is no pretraining set and no extra test set.

Not supported: ``signal_quality``. Its gates and bridge need the signals before the MIMO fill,
and the minimal files hold the filled ones.

Run: edit :data:`RUN_ARGS` at the bottom and press Run, or override a key on the command line.
A rerun into the same output directory skips every file that is already complete.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``hdf5_dataset/minimal_to_full_hdf5.py`` -> up two.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Launched with the Run button, this file's own directory goes on sys.path instead of the
# repository root, and neither the launch helper nor the pipeline can be found.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Production: every file in one folder. Repository: the packages. Each falls back only when the
# module itself is absent, so an error inside it is not hidden.
try:
    import build_fold_datasets as bfd
except ModuleNotFoundError as _exc:
    if _exc.name != "build_fold_datasets":
        raise
    from hdf5_dataset.new_pipeline import build_fold_datasets as bfd
try:
    from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args
except ModuleNotFoundError as _exc:
    if not str(_exc.name).startswith("teb_vae"):
        raise
    from launch import missing_required, resolve_launch_args

logger = logging.getLogger(__name__)


class MinimalAdaptor:
    """Stands in for ``EarlyMaestraMimoAdaptor``: serves a GUID's segments from a minimal file.

    Only the calls ``create_hdf5_dataset_from_records_list`` makes are provided:
    ``read_single_input(record)``, then ``.mimo.prepare_data()`` returning an object with
    ``block_input`` (UP in channel 0, FHR in channel 1), ``domain_start`` and ``sample_weights``.
    """

    #: GUID -> (minimal file, its rows in increasing order); filled by :func:`install_adaptor`.
    index: Dict[str, Tuple[str, np.ndarray]] = {}

    def __init__(self, **_: Any) -> None:
        self.mimo = self
        self._prepared: Any = None

    def read_single_input(self, record: str, **_: Any) -> None:
        path, rows = self.index[bfd._stem(record)]
        with h5py.File(path, "r") as f:
            # One contiguous read, then the rows: a GUID's rows lie in one block of the file.
            read = lambda k: f[k][rows[0]:rows[-1] + 1][rows - rows[0]]  # noqa: E731
            self._prepared = types.SimpleNamespace(
                block_input=np.stack([read("up"), read("fhr")], axis=-1),
                domain_start=[float(x) for x in read("epoch")],
                sample_weights=read("weight"),
            )

    def prepare_data(self, **_: Any) -> Tuple[Any, None]:
        return self._prepared, None


def _load_pipeline() -> Any:
    """Import ``create_new_pipeline``; without ``early_maestra`` on this box, stub it first.

    The stub is never called: :func:`install_adaptor` replaces the adaptor before any record is
    read.
    """
    try:
        return bfd._pipeline()
    except ModuleNotFoundError as exc:
        if not str(exc.name).startswith("early_maestra"):
            raise
    for name in ("early_maestra", "early_maestra.adaptor", "early_maestra.adaptor.mimo_adaptor"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["early_maestra.adaptor.mimo_adaptor"].EarlyMaestraMimoAdaptor = MinimalAdaptor
    return bfd._pipeline()


cnp = _load_pipeline()
#: (class target, cs, bg) -> subgroup; the eight triples are distinct.
META_TO_SUBGROUP = {meta: sg for sg, meta in cnp.SUBGROUP_META.items()}


def scan_minimal(paths: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, np.ndarray]]]:
    """Index every GUID of the minimal files.

    Returns:
        ``(guids, index)``: one row per GUID with ``tracing_guid``, ``subgroup``, ``record_path``,
        ``tlo_hours`` and ``ss_hours``; and GUID -> ``(file, rows)``.

    Raises:
        ValueError: If a GUID's rows are not contiguous or a GUID is in two files.
    """
    rows: List[Dict[str, Any]] = []
    index: Dict[str, Tuple[str, np.ndarray]] = {}
    for path in map(os.path.abspath, paths):
        with h5py.File(path, "r") as f:
            guids = f["guid"].asstr()[:]
            cols = {k: f[k][:] for k in ("class_label", "cs_label", "bg_label", "epoch",
                                         "time_from_labor_onset", "second_stage_onset")}
        cuts = [0, *(np.flatnonzero(guids[1:] != guids[:-1]) + 1), len(guids)]
        for a, b in zip(cuts[:-1], cuts[1:]):
            guid = str(guids[a])
            if guid in index:
                raise ValueError(f"GUID {guid} appears twice ({index[guid][0]} and {path})")
            index[guid] = (path, np.arange(a, b))
            epoch = float(cols["epoch"][a])
            rows.append(dict(
                tracing_guid=guid,
                subgroup=META_TO_SUBGROUP[(int(cols["class_label"][a]),
                                           bool(cols["cs_label"][a]), bool(cols["bg_label"][a]))],
                # A pseudo path whose stem is the GUID, which is all the pipeline takes from it.
                record_path=os.path.join(path, f"{guid}.mat"),
                tlo_hours=(epoch - float(cols["time_from_labor_onset"][a])) / 3600.0,
                ss_hours=(epoch - float(cols["second_stage_onset"][a])) / 3600.0,
            ))
    return pd.DataFrame(rows), index


def install_adaptor(paths: Sequence[str]) -> None:
    """Point the pipeline's record reader at the minimal files, in this process."""
    MinimalAdaptor.index = scan_minimal(paths)[1]
    cnp.EarlyMaestraMimoAdaptor = MinimalAdaptor


def assign_folds(guids: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Block id in ``[0, n_folds)`` per GUID, per subgroup, stratified by labour duration."""
    fold_id = pd.Series(0, index=guids.index, dtype=int)
    for _, grp in guids.sort_values("tracing_guid").groupby("subgroup", sort=True):
        keys = grp["tracing_guid"].tolist()
        bins = cnp._compute_duration_bins(keys, dict(zip(keys, grp["tlo_hours"])))
        counts = np.bincount(bins)
        ids = np.empty(len(grp), dtype=int)
        if len(grp) >= n_folds and counts[counts > 0].min() >= n_folds:
            splits = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(keys, bins)
            for k, (_, held) in enumerate(splits):
                ids[held] = k
        else:
            ids[np.random.RandomState(seed).permutation(len(grp))] = np.arange(len(grp)) % n_folds
        fold_id.loc[grp.index] = ids
    return fold_id


def build_partitions(guids: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """``fold_partitions.csv``: block g -> test of fold g+1, val of fold (g+1)%K+1, else train."""
    rows = []
    for g in guids.itertuples():
        for fold in range(1, n_folds + 1):
            part = ("test" if fold == g.fold_id + 1
                    else "val" if fold == (g.fold_id + 1) % n_folds + 1 else "train")
            rows.append(dict(fold=fold, partition=part, subgroup=g.subgroup,
                             tracing_guid=g.tracing_guid, record_path=g.record_path,
                             extra_test=False))
    return pd.DataFrame(rows).sort_values(["fold", "partition", "subgroup", "tracing_guid"])


def _init_worker(device_queue: Any, settings: Dict[str, Any], paths: List[str]) -> None:
    """Pool initializer: the minimal reader, then ``build_fold_datasets``' own initializer."""
    install_adaptor(paths)
    bfd._init_worker(device_queue, settings)


def run_jobs(jobs: Sequence[Any], settings: Dict[str, Any], devices: Optional[Sequence[str]],
             paths: List[str]) -> List[Dict[str, Any]]:
    """``build_fold_datasets.run_jobs``, with the minimal reader in every process."""
    install_adaptor(paths)
    if not devices or len(devices) < 2:
        return bfd.run_jobs(jobs, settings, devices)
    todo = [j for j in jobs if not bfd.is_complete(
        j.path, cnp.guid_set_digest(list(j.records)), settings["source_pickle_path"], None)]
    logger.info(f"{len(jobs) - len(todo)} of {len(jobs)} files are complete; {len(todo)} to build")
    results, failed = [], []
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        queue = manager.Queue()
        for device in devices:
            queue.put(device)
        with ProcessPoolExecutor(len(devices), mp_context=ctx, initializer=_init_worker,
                                 initargs=(queue, dict(settings), list(paths))) as pool:
            futures = {pool.submit(bfd._run_job, j): j
                       for j in sorted(todo, key=lambda j: -len(j.records))}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    results.append(future.result())
                    bfd._log_result(results[-1], i, len(todo))
                except Exception as e:  # noqa: BLE001 -- reported below; a rerun retries it
                    logger.error(f"FAILED {futures[future].path}: {e!r}")
                    failed.append((futures[future].path, repr(e)))
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(todo)} jobs failed; rerun to retry: {failed[:5]}")
    return results


def main(minimal_paths: Sequence[str], output_base_path: str, n_folds: Optional[int] = None,
         seed: Optional[int] = None, devices: Optional[Sequence[str]] = None,
         transform: Optional[str] = None, leg_alignment: Optional[str] = None,
         phase_operator: Optional[str] = None, scatter_batch_size: Optional[int] = None,
         verbose: Optional[bool] = None, assembly_workers: Optional[int] = None,
         keep_units: Optional[bool] = None) -> int:
    """Apply the defaults, write the fold folder, and run the k-fold build."""
    n_folds = int(n_folds or cnp.N_FOLDS)
    seed = cnp.RANDOM_STATE if seed is None else int(seed)
    transform = cnp.validate_transform(transform or cnp.TWO_SIDED)
    leg_alignment = cnp.validate_leg_alignment(leg_alignment or cnp.LEG_ALIGNMENT_MODES[0])
    phase_operator = phase_operator or cnp.PHASE_OPERATOR_LEGACY
    cnp.validate_phase_operator(phase_operator)
    paths = [os.path.abspath(p) for p in minimal_paths]

    guids, _ = scan_minimal(paths)
    guids["fold_id"] = assign_folds(guids, n_folds, seed)
    partitions = build_partitions(guids, n_folds)
    logger.info("GUIDs per subgroup:\n" + guids["subgroup"].value_counts().sort_index().to_string())

    # The digest is of the exact bytes written below, so build_fold_datasets' rerun check
    # recognizes this build and refuses any other content in the directory.
    text = partitions.to_csv(index=False, lineterminator="\n")
    fold_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = Path(output_base_path)
    bfd._check_output_dir(out, fold_digest)
    fold_dir = out / "fold_folder"
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / "fold_partitions.csv", "w", newline="", encoding="utf-8") as f:
        f.write(text)
    guids[["tracing_guid", "subgroup", "fold_id"]].to_csv(fold_dir / "fold_assignment.csv",
                                                          index=False)
    (fold_dir / "run_config.json").write_text(json.dumps(
        {"test_mode": "augmented", "n_folds": n_folds, "seed": seed, "minimal_paths": paths},
        indent=2), encoding="utf-8")
    tlo_csv = out / "tlo_from_minimal.csv"
    guids.rename(columns={"tracing_guid": "trace_guid", "tlo_hours": "labor_onset_hours",
                          "ss_hours": "second_stage_onset_hours"})[
        ["trace_guid", "labor_onset_hours", "second_stage_onset_hours"]].to_csv(tlo_csv,
                                                                                index=False)

    # From here on, build_fold_datasets.main with steps=["kfold"].
    partitions, assignment, run_config = bfd.load_fold_folder(str(fold_dir))
    members = bfd.unit_members(partitions, assignment)
    tasks = bfd.plan_assembly(str(out), partitions, members, "augmented", cnp.guid_set_digest)
    pickle_path = out / bfd.PICKLE_NAME
    with open(pickle_path, "wb") as f:
        pickle.dump(bfd.build_cv_result(partitions, "augmented"), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    settings = {"transform": transform, "leg_alignment": leg_alignment,
                "phase_operator": phase_operator, "tlo_csv_path": str(tlo_csv),
                "scatter_batch_size": int(scatter_batch_size or 128), "verbose": bool(verbose),
                "source_pickle_path": str(pickle_path), "signal_quality": False,
                "quality_version": None, "quality_valid_boxes": False}
    manifest = {"fold_dir": str(fold_dir), "fold_partitions_sha256": fold_digest,
                "minimal_paths": paths, "n_folds": n_folds, "seed": seed,
                "devices": list(devices or []), "settings": settings}
    with open(out / bfd.MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    pending = [t for t in tasks if not bfd.is_complete(t.path, t.digest, str(pickle_path))]
    needed = {source for t in pending for source in t.sources}
    jobs = [j for j in bfd.plan_jobs(str(out), members, None, ["kfold"]) if j.path in needed]
    results = run_jobs(jobs, settings, devices, paths)
    bfd.check_files_agree(sorted(j.path for j in jobs))
    assembled = bfd.assemble_kfold(pending, str(pickle_path), int(assembly_workers or 4))
    if not keep_units and (out / bfd.KFOLD_DIR / bfd.UNITS_DIR).exists():
        shutil.rmtree(out / bfd.KFOLD_DIR / bfd.UNITS_DIR)
    if pending:
        bfd._run_fold_eda(out / bfd.KFOLD_DIR, "augmented")
    bfd.check_files_agree(sorted(t.path for t in tasks))

    manifest["results"] = {"transform": results, "assembly": assembled}
    with open(out / bfd.MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Build finished: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser. No defaults here: :data:`RUN_ARGS` fills every flag not given."""
    p = argparse.ArgumentParser(
        prog="python hdf5_dataset/minimal_to_full_hdf5.py",
        description="Rebuild the full k-fold HDF5 datasets from the minimal raw-signal files.")
    p.add_argument("--minimal-paths", dest="minimal_paths", nargs="+")
    p.add_argument("--output-base-path", dest="output_base_path")
    p.add_argument("--n-folds", dest="n_folds", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--devices", dest="devices", nargs="+")
    p.add_argument("--transform", dest="transform")
    p.add_argument("--leg-alignment", dest="leg_alignment")
    p.add_argument("--phase-operator", dest="phase_operator")
    p.add_argument("--scatter-batch-size", dest="scatter_batch_size", type=int)
    p.add_argument("--verbose", dest="verbose", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--assembly-workers", dest="assembly_workers", type=int)
    p.add_argument("--keep-units", dest="keep_units", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("minimal_paths", "output_base_path"))
    if refusal:
        raise SystemExit(refusal)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("resolved arguments: " + ", ".join(
        f"{k}={values[k]!r} (from {sources[k]})" for k in sorted(values)))
    return main(**values)


_MINIMAL_DIR = r"/data1/fetal-heart-tracing/HDF5_Datasets/minimal_raw_signals"

#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # list[str] -- the files written by create_minimal_hdf5.py. Required.
    "minimal_paths": [os.path.join(_MINIMAL_DIR, f"{c}.hdf5")
                      for c in ("acidosis", "hie", "healthy")],
    # str -- a NEW directory for the build. Required. A rerun into it skips finished files.
    "output_base_path": r"/data1/fetal-heart-tracing/HDF5_Datasets/from_minimal_causal_int",

    # ======== The variant to build: same values and meanings as in build_fold_datasets.py ========
    # "two_sided" | "causal"                              (None -> "two_sided")
    "transform": "causal",
    # "none" | "envelope"                                 (None -> "none"; causal only)
    "leg_alignment": "envelope",
    # "ratio_power_v0" | "integer_harmonic_v1"            (None -> "ratio_power_v0"; causal only)
    "phase_operator": "integer_harmonic_v1",

    # ========================= Everything below has a default =========================
    # int | None -- folds.  (None -> 10: 80 / 10 / 10)
    "n_folds": None,
    # int | None -- seed of the fold assignment.  (None -> 42)
    "seed": None,
    # list[str] | None -- one transform process per entry, e.g. ["cuda:0", "cuda:1"].
    #        (None -> one process on the first CUDA device, else the CPU)
    "devices": None,
    # int | None -- segments per forward pass of the transform.  (None -> 128)
    "scatter_batch_size": 128,
    # bool | None -- per-record progress output.  (None -> False)
    "verbose": False,
    # int | None -- processes that copy units into partition files.  (None -> 4)
    "assembly_workers": None,
    # bool | None -- keep k_fold_cross_validation_dataset/_units after the assembly.  (None -> False)
    "keep_units": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
