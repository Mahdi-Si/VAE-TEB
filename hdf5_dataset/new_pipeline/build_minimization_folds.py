"""Assign the classification GUIDs to folds with the ``ctg_rules`` minimization splitter.

This step only decides which GUID goes into which fold and partition; it builds no HDF5 file.
It reads the train/val pool from an existing ``classification_dataset_records.pickle`` (written
by ``hdf5_dataset/new_pipeline/create_new_pipeline.py``), calls
:func:`ctg_rules.cohort_folds.build_cohort_folds` on that pool, and writes the GUIDs per fold
and partition.

Partitions follow the convention of the ``ctg_rules`` ``TieredCrossValidation`` splitter. A
GUID with minimization fold id $g \\in \\{0, \\dots, K-1\\}$ goes to:

* augmented mode: the test partition of fold $g + 1$, the val partition of fold
  $((g + 1) \\bmod K) + 1$, and the train partition of every other fold. At $K = 10$ this is
  80 / 10 / 10.
* holdout mode: the val partition of fold $g + 1$ and the train partition of every other fold.

The extra test GUIDs (``test_augmentation`` in augmented mode, the fixed ``test`` set in
holdout mode) are never split: they join the test partition of every fold, exactly as
``create_new_pipeline.py`` does.

The splitter offsets each dataset's seed by Python's ``hash()`` of a string, and that hash
changes between processes unless ``PYTHONHASHSEED`` is fixed. :func:`_cli` therefore relaunches
the script with ``PYTHONHASHSEED=0``, so a rerun reproduces the same folds.

The GUIDs passed to the splitter are the raw ``.mat`` file stems. The database matches
``tracing_guid`` as an exact string, so they are never normalized here.

A pool GUID that is absent from the database has no covariates, so the splitter cannot place
it. :func:`place_unstratified` puts each such GUID in the fold with the fewest GUIDs of its
subgroup, and ``fold_assignment.csv`` flags it ``stratified=False``. More than
``max_unstratified`` absent GUIDs stops the run, because that many points to a GUID format
problem rather than to a few missing recordings.

Outputs, in ``output_dir``:

* ``fold_assignment.csv``: one row per pool GUID with ``tracing_guid``, ``subgroup``,
  ``dataset_name`` (empty when the GUID is absent from the database), ``fold_id``,
  ``stratified``.
* ``fold_partitions.csv``: one row per (fold, GUID) with ``fold``, ``partition``,
  ``subgroup``, ``tracing_guid``, ``record_path``, ``extra_test``.
* ``fold_counts.csv``: GUID counts per fold, partition and subgroup.
* ``cohort_covariates.csv``: the covariate table the splitter fetched for the pool.
* ``strata_metadata.json``, ``minimization_report.json``, ``run_config.json``.

Run it with the Python environment of the ``em_analysis`` repository, on a machine that can
reach the KP research database. Edit :data:`RUN_ARGS` at the bottom and press Run, or pass
the same values as command-line flags.

This file sits in ``hdf5_dataset/new_pipeline/`` between the two steps it connects:
``create_new_pipeline.py`` (with ``selection_only=True``) writes its input pickle, and
``build_fold_datasets.py`` reads its ``output_dir``. It imports neither of them, so it runs
alone on the ``em_analysis`` box; copy only this file there. The extra test GUIDs may belong
to any healthy subgroup, BG ones included.
"""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# Copied from ``create_new_pipeline.py``, not imported: that module pulls in torch and the
# scattering stack, which the ``em_analysis`` environment this script runs in does not carry.
# The keys are also the dataset names in the database.
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
SUBGROUP_TO_FOLDER = {v: k for k, v in FOLDER_TO_SUBGROUP.items()}

#: The ``production_stratified`` axes of ``ctg_rules`` (``run_settings.yaml``), less its two
#: ``epoch_*`` signal-quality axes: those are defined on 2-min colour epochs, not on this
#: project's segments. :data:`QUALITY_COLUMN` takes their place when a screening CSV is given.
DEFAULT_STRATA_COLUMNS = (
    "hospital_guid",
    "gestational_age",
    "labor_length",
    "second_stage_length",
    "maternal_age",
    "prepregnancy_bmi",
    "cd_priority_label",
    "gestational_diabetes",
    "gestational_hypertension",
    "preeclampsia",
    "max_temperature",
)

#: Continuous axes, quantile-binned per dataset. Every other axis is balanced on its native
#: levels. Matches ``fold_stratify_bins`` of ``production_stratified``.
CONTINUOUS_BINS = {
    "gestational_age": 4,
    "labor_length": 4,
    "second_stage_length": 4,
    "maternal_age": 4,
    "prepregnancy_bmi": 4,
    "max_temperature": 4,
}

#: Signal-quality axis taken from ``estimated_valid_hours_6h`` of the screening CSV: the hours
#: of segments that pass the weight and flat-region filters in the last 6 h before delivery.
QUALITY_COLUMN = "valid_hours_6h"
QUALITY_BINS = 4


def _fold_number(fold_name: str) -> int:
    """Return the integer of a ``fold_<k>`` key."""
    return int(fold_name.rsplit("_", 1)[-1])


def _stem(record_path: str) -> str:
    """Return the raw ``.mat`` file stem, which is the database ``tracing_guid``."""
    return os.path.splitext(os.path.basename(record_path))[0]


def _rows(subgroups: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Flatten ``{subgroup: [record_path]}`` into one row per record."""
    records = [(_stem(p), sg, p) for sg, paths in subgroups.items() for p in paths]
    return pd.DataFrame(records, columns=["tracing_guid", "subgroup", "record_path"])


def load_pool(pickle_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read the train/val pool and the extra test GUIDs from a fold pickle.

    Every fold of the pickle must hold the same pool, which is checked, so the pool does not
    depend on which fold is read. The pool is sorted by subgroup and GUID; the splitter's
    result depends on row order, and this order does not depend on the old split.

    Args:
        pickle_path: ``classification_dataset_records.pickle`` from ``create_new_pipeline.py``.

    Returns:
        ``(pool, extra, test_mode)``. ``pool`` and ``extra`` have the columns
        ``tracing_guid``, ``subgroup`` and ``record_path``. ``extra`` holds the GUIDs that join
        every test partition unsplit. ``test_mode`` is ``'augmented'`` or ``'holdout'``.

    Raises:
        ValueError: If the folds disagree on the pool, a GUID is repeated, or a GUID is both
            in the pool and in the extra test set.
    """
    with open(pickle_path, "rb") as f:
        cv = pickle.load(f)
    # A legacy pickle is a flat dict of folds; the builder reads it as augmented mode too.
    if "test_mode" not in cv:
        cv = {"test_mode": "augmented", "folds": cv, "test_augmentation": {}}
    test_mode = cv["test_mode"]
    extra = _rows(cv.get("test_augmentation" if test_mode == "augmented" else "test", {}))
    extra_paths = set(extra["record_path"])

    pools: Dict[str, pd.DataFrame] = {}
    for fold_name in sorted(cv["folds"], key=_fold_number):
        fold_data = cv["folds"][fold_name]
        parts = [_rows(fold_data.get(part, {})) for part in ("train", "val", "test")]
        rows = pd.concat(parts, ignore_index=True)
        pools[fold_name] = rows[~rows["record_path"].isin(extra_paths)]

    first_name, pool = next(iter(pools.items()))
    reference = set(pool["record_path"])
    for fold_name, rows in pools.items():
        if set(rows["record_path"]) != reference:
            raise ValueError(
                f"{fold_name} holds a different train/val pool than {first_name}; "
                f"the pickle is not a consistent k-fold split.")

    for name, frame in (("pool", pool), ("extra test set", extra)):
        repeated = frame["tracing_guid"][frame["tracing_guid"].duplicated()]
        if len(repeated):
            raise ValueError(
                f"{len(repeated)} GUID(s) repeated in the {name}, e.g. "
                f"{repeated.unique()[:5].tolist()}.")
    both = set(pool["tracing_guid"]) & set(extra["tracing_guid"])
    if both:
        raise ValueError(
            f"{len(both)} GUID(s) are both in the pool and in the extra test set, e.g. "
            f"{sorted(both)[:5]}.")

    pool = pool.sort_values(["subgroup", "tracing_guid"], ignore_index=True)
    return pool, extra.reset_index(drop=True), test_mode


def load_quality(screening_csv_path: str, guids: Sequence[str]) -> pd.DataFrame:
    """Read the signal-quality covariate of every pool GUID from the screening CSV.

    Args:
        screening_csv_path: ``guid_screening_results.csv`` of the same pipeline run.
        guids: The pool GUIDs (raw file stems).

    Returns:
        Frame with ``tracing_guid`` and :data:`QUALITY_COLUMN`.

    Raises:
        ValueError: If a GUID occurs twice in the CSV or a pool GUID is absent from it,
            which means the CSV and the pickle come from different runs.
    """
    # dtype=str: an all-digit GUID would otherwise be read as a number.
    df = pd.read_csv(screening_csv_path, usecols=["guid", "estimated_valid_hours_6h"],
                     dtype={"guid": str})
    repeated = df["guid"][df["guid"].duplicated()]
    if len(repeated):
        raise ValueError(f"{len(repeated)} GUID(s) repeated in {screening_csv_path}, e.g. "
                         f"{repeated.unique()[:5].tolist()}.")
    missing = sorted(set(guids) - set(df["guid"]))
    if missing:
        raise ValueError(
            f"{len(missing)} pool GUID(s) are absent from {screening_csv_path}, e.g. "
            f"{missing[:5]}. Use the screening CSV of the run that wrote the pickle.")
    df = df[df["guid"].isin(set(guids))]
    return df.rename(columns={"guid": "tracing_guid",
                              "estimated_valid_hours_6h": QUALITY_COLUMN})


def check_datasets(pool: pd.DataFrame, covariates: pd.DataFrame,
                   dataset_names: Mapping[int, str]) -> pd.Series:
    """Check that the database puts every GUID in the dataset its folder names.

    Args:
        pool: The pool, with ``tracing_guid`` and ``subgroup``.
        covariates: The splitter's covariate table, indexed by ``tracing_guid``, with
            ``dataset_id``.
        dataset_names: ``{dataset_id: dataset_name}`` from the ``ctg_rules`` config.

    Returns:
        The database dataset name of every pool GUID, aligned with ``pool``.

    Raises:
        ValueError: If a GUID's database dataset differs from its folder.
    """
    ids = covariates.reindex(pool["tracing_guid"])["dataset_id"].to_numpy()
    names = pd.Series([dataset_names.get(int(float(d))) for d in ids], index=pool.index)
    expected = pool["subgroup"].map(SUBGROUP_TO_FOLDER)
    bad = pool[names != expected]
    if len(bad):
        example = bad.head(5).assign(database=names[bad.index].values)
        raise ValueError(
            f"{len(bad)} GUID(s) belong to a different dataset in the database than in "
            f"your folders:\n{example[['tracing_guid', 'subgroup', 'database']]}")
    return names


def place_unstratified(subgroups: np.ndarray, fold_id: np.ndarray, absent: np.ndarray,
                       n_folds: int) -> np.ndarray:
    """Place the GUIDs the splitter could not see, one at a time, in pool order.

    Each goes to the fold with the fewest GUIDs of its own subgroup; a tie goes to the fold
    with the fewest GUIDs overall, then to the lowest fold id. The rule never widens the
    spread of a subgroup's fold sizes beyond one more than the splitter left, and it is
    deterministic.

    Args:
        subgroups: Subgroup of every pool GUID.
        fold_id: Fold id of every pool GUID; any value for the absent ones.
        absent: True for the GUIDs the splitter did not place.
        n_folds: The number of folds $K$.

    Returns:
        A copy of ``fold_id`` with every absent GUID placed.
    """
    fold_id = np.where(absent, -1, fold_id).astype(np.int64)
    for i in np.flatnonzero(absent):
        placed = fold_id >= 0
        per_subgroup = np.bincount(fold_id[placed & (subgroups == subgroups[i])],
                                   minlength=n_folds)
        total = np.bincount(fold_id[placed], minlength=n_folds)
        fold_id[i] = min(range(n_folds), key=lambda f: (per_subgroup[f], total[f], f))
    return fold_id


def build_partitions(pool: pd.DataFrame, fold_id: np.ndarray, extra: pd.DataFrame,
                     test_mode: str, n_folds: int) -> pd.DataFrame:
    """Expand the fold ids into the train / val / test partitions of every fold.

    Args:
        pool: The pool, aligned with ``fold_id``.
        fold_id: Minimization fold id $g \\in \\{0, \\dots, K-1\\}$ of every pool GUID.
        extra: GUIDs that join the test partition of every fold.
        test_mode: ``'augmented'`` or ``'holdout'``.
        n_folds: The number of folds $K$.

    Returns:
        One row per (fold, GUID) with ``fold`` ($1 \\dots K$), ``partition``, ``subgroup``,
        ``tracing_guid``, ``record_path`` and ``extra_test``.
    """
    frames = []
    for k in range(n_folds):
        if test_mode == "augmented":
            # Fold k+1 tests minimization fold k and validates on fold k-1, the cyclic rule
            # of the ctg_rules TieredCrossValidation splitter.
            partition = np.where(fold_id == k, "test",
                                 np.where(fold_id == (k - 1) % n_folds, "val", "train"))
        else:
            partition = np.where(fold_id == k, "val", "train")
        frames.append(pool.assign(fold=k + 1, partition=partition, extra_test=False))
        frames.append(extra.assign(fold=k + 1, partition="test", extra_test=True))
    columns = ["fold", "partition", "subgroup", "tracing_guid", "record_path", "extra_test"]
    return pd.concat(frames, ignore_index=True)[columns]


def check_partitions(partitions: pd.DataFrame, test_mode: str, n_folds: int) -> List[str]:
    """Check the partition counts and list the folds that miss a subgroup.

    Args:
        partitions: Output of :func:`build_partitions`.
        test_mode: ``'augmented'`` or ``'holdout'``.
        n_folds: The number of folds $K$.

    Returns:
        One warning line per (fold, subgroup) whose held-out partition is empty.

    Raises:
        AssertionError: If a pool GUID is not in each partition the expected number of times.
    """
    core = partitions[~partitions["extra_test"]]
    expected = ({"test": 1, "val": 1, "train": n_folds - 2} if test_mode == "augmented"
                else {"val": 1, "train": n_folds - 1})
    counts = pd.crosstab(core["tracing_guid"], core["partition"]).reindex(
        columns=list(expected), fill_value=0)
    for part, n in expected.items():
        assert (counts[part] == n).all(), f"a GUID is not in {part} exactly {n} time(s)"

    held_out = "test" if test_mode == "augmented" else "val"
    held = core[core["partition"] == held_out]
    present = pd.crosstab(held["subgroup"], held["fold"]).reindex(
        index=sorted(core["subgroup"].unique()), columns=range(1, n_folds + 1), fill_value=0)
    return [f"fold_{fold}: no {sg} GUID in {held_out}"
            for sg, row in present.iterrows() for fold, n in row.items() if n == 0]


def run(pool: pd.DataFrame, extra: pd.DataFrame, test_mode: str, *, output_dir: str,
        dataset_names: Mapping[int, str], n_folds: int, seed: int, n_seed_trials: int,
        objective: str, strata_columns: Sequence[str], quality: Optional[pd.DataFrame],
        known_guids: Optional[Set[str]] = None, max_unstratified: int = 10,
        db_config: Optional[Mapping[str, Any]] = None, source: Any = None,
        provenance: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run the splitter on the pool and write every output file.

    Args:
        pool: Output of :func:`load_pool`.
        extra: Output of :func:`load_pool`.
        test_mode: Output of :func:`load_pool`.
        output_dir: Directory for the output files.
        dataset_names: ``{dataset_id: dataset_name}``.
        n_folds: The number of folds $K$.
        seed: Base seed of the splitter.
        n_seed_trials: Greedy passes of the multi-seed search.
        objective: ``'sum_of_ranges'`` or ``'kl_divergence'``.
        strata_columns: Covariate axes to balance.
        quality: Output of :func:`load_quality`, or ``None`` for no quality axis.
        known_guids: Every ``tracing_guid`` in the database's datasets. Pool GUIDs outside it
            are placed by :func:`place_unstratified`. ``None`` treats every GUID as known.
        max_unstratified: The most absent GUIDs accepted before the run stops.
        db_config: ``ctg_rules`` config with ``connection_str_f``; the production path.
        source: A ``CohortDataSource`` instead of the database; the offline test seam.
        provenance: Extra entries for ``run_config.json``.

    Returns:
        The partitions table (see :func:`build_partitions`).

    Raises:
        ValueError: If more than ``max_unstratified`` pool GUIDs are absent from the database.
    """
    from ctg_rules.cohort_folds import StratifiedFoldConfig, build_cohort_folds

    in_db = (np.ones(len(pool), dtype=bool) if known_guids is None
             else pool["tracing_guid"].isin(known_guids).to_numpy())
    absent = pool.loc[~in_db]
    if len(absent) > max_unstratified:
        raise ValueError(
            f"{len(absent)} of {len(pool)} pool GUIDs are absent from the database (limit "
            f"max_unstratified={max_unstratified}), e.g. {absent['tracing_guid'].head(5).tolist()}. "
            f"That many points to a GUID format problem; check it before raising the limit.")
    resolved = pool.loc[in_db]

    columns = tuple(strata_columns) + ((QUALITY_COLUMN,) if quality is not None else ())
    bins = {c: b for c, b in {**CONTINUOUS_BINS, QUALITY_COLUMN: QUALITY_BINS}.items()
            if c in columns}
    config = StratifiedFoldConfig(n_folds=n_folds, strata_columns=columns, strata_bins=bins,
                                  n_seed_trials=n_seed_trials, objective=objective, seed=seed)
    folds = build_cohort_folds(
        resolved["tracing_guid"].tolist(), config=config, source=source, db_config=db_config,
        want_covariates=True, want_dynamic_features=False, external_covariates=quality)

    names = pd.Series(None, index=pool.index, dtype=object)
    names.loc[resolved.index] = check_datasets(resolved, folds.covariates, dataset_names).values
    fold_id = np.full(len(pool), -1, dtype=np.int64)
    fold_id[in_db] = folds.fold_assignment.reindex(resolved["tracing_guid"]).to_numpy()
    fold_id = place_unstratified(pool["subgroup"].to_numpy(), fold_id, ~in_db, n_folds)
    partitions = build_partitions(pool, fold_id, extra, test_mode, n_folds)
    warnings = check_partitions(partitions, test_mode, n_folds)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool[["tracing_guid", "subgroup"]].assign(
        dataset_name=names.values, fold_id=fold_id, stratified=in_db) \
        .to_csv(out / "fold_assignment.csv", index=False)
    partitions.to_csv(out / "fold_partitions.csv", index=False)
    counts = (partitions.groupby(["fold", "partition", "subgroup"]).size()
              .rename("n_guids").reset_index())
    counts.to_csv(out / "fold_counts.csv", index=False)
    covariates = folds.covariates
    if folds.external_covariates is not None:
        covariates = covariates.join(folds.external_covariates)
    covariates.to_csv(out / "cohort_covariates.csv")
    for name, payload in (("strata_metadata.json", folds.strata_metadata),
                          ("minimization_report.json", folds.minimization_report)):
        with open(out / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
    with open(out / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            **(provenance or {}),
            "test_mode": test_mode,
            "n_folds": n_folds,
            "seed": seed,
            "n_seed_trials": n_seed_trials,
            "objective": objective,
            "strata_columns": list(columns),
            "strata_bins": bins,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "n_pool_guids": int(len(pool)),
            "n_extra_test_guids": int(len(extra)),
            "pool_per_subgroup": pool["subgroup"].value_counts().sort_index().to_dict(),
            "unstratified_guids": absent["tracing_guid"].tolist(),
        }, f, indent=2)

    unstratified = pool.loc[~in_db, ["tracing_guid", "subgroup"]].assign(
        fold_id=fold_id[~in_db])
    _report(partitions, folds.strata_metadata, warnings, test_mode, out, unstratified)
    return partitions


def _report(partitions: pd.DataFrame, metadata: Mapping[str, Any], warnings: List[str],
            test_mode: str, out: Path, unstratified: pd.DataFrame) -> None:
    """Print the held-out counts per fold, the axes the splitter shrank or dropped, and the
    GUIDs placed without stratification."""
    held_out = "test" if test_mode == "augmented" else "val"
    core = partitions[(~partitions["extra_test"]) & (partitions["partition"] == held_out)]
    print(f"\nPool GUIDs in {held_out} per fold (extra test GUIDs not counted):")
    print(pd.crosstab(core["subgroup"], core["fold"], margins=True).to_string())
    for column, meta in metadata.get("columns", {}).items():
        status = {ds: s for ds, s in meta.get("per_dataset_status", {}).items() if s != "ok"}
        if status:
            print(f"  axis {column}: {status}  (dataset_id -> status)")
    for row in unstratified.itertuples():
        print(f"  NOT STRATIFIED {row.tracing_guid} ({row.subgroup}): absent from the database, "
              f"placed by hand in fold_id {row.fold_id}")
    for line in warnings:
        print(f"  WARNING {line}")
    print(f"\nWrote the fold files to {out}")


def _sha256(path: str) -> str:
    """Return the SHA-256 of a file, recorded so the input pickle is identifiable later."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(pickle_path: str, config_yaml: str, output_dir: str,
         em_analysis_root: Optional[str] = None, screening_csv_path: Optional[str] = None,
         n_folds: int = 10, seed: int = 42, n_seed_trials: int = 32,
         objective: str = "sum_of_ranges", strata_columns: Optional[Sequence[str]] = None,
         limit: Optional[int] = None, max_unstratified: int = 10,
         overwrite: bool = False) -> int:
    """Load the pool, connect to the database through ``ctg_rules`` and write the folds.

    Args:
        pickle_path: ``classification_dataset_records.pickle``.
        config_yaml: ``ctg_rules`` config, e.g. ``ClinicalRules/config_EM.yaml``.
        output_dir: Directory for the output files.
        em_analysis_root: Root of the ``em_analysis`` checkout, put on ``sys.path``;
            ``None`` if ``ctg_rules`` is already importable (editable install).
        screening_csv_path: ``guid_screening_results.csv`` for the quality axis; ``None``
            for no quality axis.
        n_folds: The number of folds $K$.
        seed: Base seed of the splitter.
        n_seed_trials: Greedy passes of the multi-seed search.
        objective: ``'sum_of_ranges'`` or ``'kl_divergence'``.
        strata_columns: Covariate axes; ``None`` for :data:`DEFAULT_STRATA_COLUMNS`.
        limit: Use only the first $N$ pool GUIDs, to check the database connection quickly.
        max_unstratified: The most pool GUIDs absent from the database that are placed by
            hand before the run stops.
        overwrite: Replace an existing ``fold_assignment.csv`` in ``output_dir``.

    Returns:
        The process exit code.

    Raises:
        FileExistsError: If ``output_dir`` already holds folds and ``overwrite`` is false.
    """
    if (Path(output_dir) / "fold_assignment.csv").exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already holds fold_assignment.csv. A saved split is the only "
            f"record of it; choose a new output_dir or set overwrite=True.")
    if em_analysis_root is not None:
        root = Path(em_analysis_root)
        for sub in ("python/database", "ClinicalRules/Interpretation"):
            if not (root / sub).is_dir():
                raise FileNotFoundError(f"{root / sub} not found; check em_analysis_root.")
            if str(root / sub) not in sys.path:
                sys.path.insert(0, str(root / sub))
    from ctg_rules.cohort_folds import list_cohort_tracing_guids
    from ctg_rules.io.database.clinical_rules_config import ClinicalRulesConfig

    db_config = ClinicalRulesConfig(config_yaml).get_cfg()
    dataset_names = {int(info["dataset_id"]): name
                     for name, info in db_config["datasets"].items()}
    known_guids = set(list_cohort_tracing_guids(db_config=db_config,
                                                dataset_ids=sorted(dataset_names)))

    pool, extra, test_mode = load_pool(pickle_path)
    if limit is not None:
        print(f"limit={limit}: connection check only, these folds are not the real ones.")
        pool = pool.head(limit)
    quality = (load_quality(screening_csv_path, pool["tracing_guid"].tolist())
               if screening_csv_path else None)
    print(f"{len(pool)} pool GUIDs, {len(extra)} extra test GUIDs, mode={test_mode}")

    run(pool, extra, test_mode, output_dir=output_dir, dataset_names=dataset_names,
        n_folds=n_folds, seed=seed, n_seed_trials=n_seed_trials, objective=objective,
        strata_columns=strata_columns or DEFAULT_STRATA_COLUMNS, quality=quality,
        known_guids=known_guids, max_unstratified=max_unstratified, db_config=db_config,
        provenance={"pickle_path": str(pickle_path), "pickle_sha256": _sha256(pickle_path),
                    "screening_csv_path": screening_csv_path, "config_yaml": str(config_yaml),
                    "limit": limit})
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    No flag carries a default: a value that is not on the command line comes from
    :data:`RUN_ARGS`, so the Run button and the command line run the same build.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(description="Assign classification GUIDs to minimization folds.")
    p.add_argument("--pickle-path", dest="pickle_path")
    p.add_argument("--config-yaml", dest="config_yaml")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--em-analysis-root", dest="em_analysis_root")
    p.add_argument("--screening-csv-path", dest="screening_csv_path")
    p.add_argument("--n-folds", dest="n_folds", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--n-seed-trials", dest="n_seed_trials", type=int)
    p.add_argument("--limit", dest="limit", type=int)
    p.add_argument("--max-unstratified", dest="max_unstratified", type=int)
    p.add_argument("--overwrite", dest="overwrite", action="store_true", default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Fix the hash seed, merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    argv = sys.argv[1:] if argv is None else argv
    if os.environ.get("PYTHONHASHSEED") != "0":
        # The hash seed is fixed when the interpreter starts, so only a new process can set it.
        env = dict(os.environ, PYTHONHASHSEED="0")
        return subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
    values = dict(RUN_ARGS)
    values.update({k: v for k, v in vars(build_parser().parse_args(argv)).items()
                   if v is not None})
    missing = [k for k in ("pickle_path", "config_yaml", "output_dir") if not values.get(k)]
    if missing:
        raise SystemExit(f"Set {missing} in RUN_ARGS or on the command line.")
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
#: Use absolute paths.
RUN_ARGS: Dict[str, Any] = {
    # str -- the pickle of the pipeline run whose GUIDs you want to re-fold. Required.
    "pickle_path": r"/path/to/output_base_path/classification_dataset_records.pickle",

    # str | None -- guid_screening_results.csv of the SAME run, for the valid_hours_6h axis.
    #        None = no signal-quality axis.
    "screening_csv_path": r"/path/to/output_base_path/guid_screening_results.csv",

    # str -- the ctg_rules config with db_params and datasets. Required.
    "config_yaml": r"/path/to/em_analysis/ClinicalRules/config_EM.yaml",

    # str | None -- root of the em_analysis checkout. None = ctg_rules is already importable
    #        (pip install -e . at that root).
    "em_analysis_root": r"/path/to/em_analysis",

    # str -- a NEW directory for the fold files. Required.
    "output_dir": r"/path/to/output_base_path/minimization_folds_k10",

    # int -- number of folds K. Keep 10 to match the classifier's fold_1 .. fold_10.
    "n_folds": 10,

    # int -- base seed of the splitter.
    "seed": 42,

    # int -- greedy passes of the multi-seed search; the best one is kept.
    "n_seed_trials": 32,

    # "sum_of_ranges" | "kl_divergence"
    "objective": "sum_of_ranges",

    # list[str] | None -- covariate axes. None = DEFAULT_STRATA_COLUMNS.
    "strata_columns": None,

    # int | None -- use only the first N GUIDs, to test the database connection. None = all.
    "limit": None,

    # int -- pool GUIDs absent from the database are placed by hand (stratified=False in
    #        fold_assignment.csv), up to this many; more stops the run as a likely GUID format
    #        problem. The 2026-10 check found 1 of 5402 (77ED7F0BE9D5452D81ADCAF497B3FF70, hie_cs).
    "max_unstratified": 10,

    # bool -- replace existing fold files in output_dir.
    "overwrite": False,
}


if __name__ == "__main__":
    sys.exit(_cli())
