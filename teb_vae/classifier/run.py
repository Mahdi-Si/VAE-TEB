r"""Classifier CLI and run directory (SPEC §14).

    python -m teb_vae.classifier.run --config teb_vae/classifier/configs/smoke.yaml --stage cohort \
        [--set classifier.data.kfold_root=/path ...] [--run-dir PATH] [--folds 1,2] [--device cpu] \
        [--devices cuda:0,cuda:1,...]

    python -m teb_vae.classifier.run compare --runs RUN_A RUN_B [--policy np30] --out DIR   # §11.8, block Q

or edit :data:`RUN_ARGS` at the bottom and run the file. Stages live in :data:`STAGES`, a plain
dict of ``fn(cfg, run_dir, manifest) -> Optional[int]`` run in insertion order by ``--stage all``;
later phases add entries. A stage's return value is its exit code (None = 0): it is recorded as
``exit_code`` beside ``status: done`` in ``stage_state.json``, and the CLI exits with the largest
one. A stage marked ``done`` with exit code 0 is skipped; one with a non-zero exit code runs again.
A stage that runs marks every later stage ``pending`` (its inputs changed), so a retrained unit
reaches ``predict`` and everything after it on the next ``--stage all``. ``evaluate``, ``report``
and ``verify`` read only earlier outputs, so naming one explicitly always re-runs it. ``evaluate --only/--skip``
is a partial look (``evaluation/partial_<stamp>/``): it leaves ``stage_state.json`` untouched. An existing
run directory whose config digest differs raises, so a resume never mixes settings.

``train`` runs the §10.10.1 outer loop per fold: the sklearn baselines, then one neural unit per
``run.seeds`` (kind ``model``), the shuffled-label control on the first seed and, when the cohort flags a
missingness confound under ``context.auto_ablate_missing``, the ``noind`` ablation (§7.3.4). Each unit ends in
``selection_lock.json`` (val calibration + thresholds); a locked unit is skipped on resume, a failed
one is recorded (exit code 1; ``run.fail_fast`` raises instead) and ``evaluate`` refuses the run
unless ``--allow-partial``. Per-unit status: ``stage_state.json["train"]["units"]``. MLflow (§10.10.5):
one parent run per run directory, fail-closed; units nest under it; ``report`` logs the results to it.

Fold-parallel (§14.4): with ``run.devices`` (``--devices cuda:0,cuda:1,...``) the per-fold part of ``train`` and
``predict`` runs in one spawned process per fold, one device slot each (:mod:`~teb_vae.classifier.parallel`); the
parent keeps the baselines and the MLflow parent, merges each fold as its process ends, and runs ``evaluate``,
``report`` and ``verify`` once over every fold, exactly as the serial path does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
# Run as a file (IDE Run button), this directory would shadow the top-level `train` package once
# this package has a train.py; put the repo root first instead (as latent_pilot/run.py does).
if not __package__:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from teb_vae.classifier import baselines, cohort, metrics, parallel, report, sources, verify  # noqa: E402
from teb_vae.classifier import train as classifier_train  # noqa: E402
from teb_vae.classifier.baselines import (  # noqa: E402
    GUID_COLUMNS, LOCK, SEGMENT_COLUMNS, THREE_CLASS_COLUMNS, label_rows, lock_problem, read_lock, write_lock,
)
from teb_vae.classifier.config import (  # noqa: E402
    Classifier, Config, digest, load, ovr_enabled, resolve_path, selection_monitor, unit_config, unit_dir,
)
from teb_vae.classifier.data import (  # noqa: E402
    UnitData, build_unit, covariates_off, load_store, without_indicators,
)
from teb_vae.classifier.thresholds import (  # noqa: E402
    fit_stage_offsets, guid_level_policies, ovr_thresholds, select_thresholds,
)
from teb_vae.classifier.train import (  # noqa: E402
    ATTN_COLUMNS, ATTRIBUTION_COLUMNS, apply_stage_offsets, attribute_split, calibrate_frames, score_split, train_unit,
    unit_calibration,
)
from teb_vae.lag_attn.eval.report import json_safe  # noqa: E402
from teb_vae.lag_attn_cfs.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import software_record  # noqa: E402
from utils.custom_logger import setup_logging  # noqa: E402

VERSIONED = ("torch", "numpy", "pandas", "scikit-learn", "h5py", "pydantic", "lightning")
#: What a neural unit's lock digests: everything val selected, frozen before test is scored (L5).
UNIT_LOCKED = ("model_checkpoints/best.ckpt", "scaler.json", "calibration.json", "thresholds.json")
#: What a seed ensemble's lock digests (§10.7), besides its members' own ``selection_lock.json``: a retrained member
#: re-locks, which changes that digest, so a stale ensemble is re-selected by the next ``train``.
ENS_LOCKED = ("calibration.json", "thresholds.json")
#: The §11.3 stage-offset view (``eval.stage_offsets``) per unit: the offsets fitted on the val negatives and the
#: thresholds re-selected on the shifted val rows. Both enter the unit's lock; its prediction rows are ``<kind>_stage``.
STAGE_FILES = ("stage_offsets.json", "thresholds_stage.json")
STAGE_SUFFIX = "_stage"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def shard_record(path: str) -> Dict[str, Any]:
    """Size, mtime and the shard's ``source_guid_digest`` attribute (None when absent)."""
    stat = os.stat(path)
    with h5py.File(path, "r") as handle:
        guid_digest = handle.attrs.get("source_guid_digest")
    return {"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime,
            "source_guid_digest": None if guid_digest is None else str(guid_digest)}


# ---- neural units (§10.10.1 outer loop, steps 4-5) ------------------------------------------------
def neural_units(c: Classifier, confound: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, int]]:
    """``(kind, seed)`` per fold: a ``model`` unit per ``run.seeds``, then the §10.9.3 shuffled control (first seed),
    then, with ``context.auto_ablate_missing`` and any fold's missingness confound flagged (``confound``, the cohort
    record), the §7.3.4 ``noind`` ablation (first seed; every fold, so it pools), then under an online regime the
    §10.1 ``frozen`` baseline per seed (``config.frozen_baseline``: the same head on the frozen cache).
    ``noind`` runs the configured regime, the shuffled control the frozen one (§10.9.3)."""
    shuffled = [("shuffled", c.run.seeds[0])] if c.baselines.shuffled_control else []
    noind = [("noind", c.run.seeds[0])] if c.context.auto_ablate_missing and cohort.confound_flagged(confound) else []
    frozen = [("frozen", s) for s in c.run.seeds] if c.train.regime != "frozen_cached" else []
    return [("model", s) for s in c.run.seeds] + shuffled + noind + frozen


# ---- seed ensembles (§10.7) -----------------------------------------------------------------------
#: Raw score columns of :func:`score_split` averaged across seeds, besides every GUID ``score_*``.
ENS_MEAN = ("logit_seg", "logit_online", "attn_entropy", *ATTN_COLUMNS, "ord_score")


def ensembles(c: Classifier, confound: Optional[Mapping[str, Any]] = None) -> Dict[str, List[int]]:
    """``{kind: seeds}`` of every unit kind trained under more than one seed per fold: ``model`` with several
    ``run.seeds`` (and the online regimes' ``frozen`` baseline with them). Each gets a seed ensemble (§10.7)."""
    seeds: Dict[str, List[int]] = {}
    for kind, seed in neural_units(c, confound):
        seeds.setdefault(kind, []).append(seed)
    return {kind: s for kind, s in seeds.items() if len(s) > 1}


def ens_dir(run_dir: Path, fold: int, kind: str) -> Path:
    """``<run>/folds/fold_<k>/ens`` (kind ``model``) or ``.../ens/<kind>`` (§14.2)."""
    base = Path(run_dir) / "folds" / f"fold_{fold}" / "ens"
    return base if kind == "model" else base / kind


def ens_files(out: Path, members: Sequence[Path]) -> List[str]:
    """What the ensemble lock at ``out`` digests: :data:`ENS_LOCKED` and every member's lock, relative to ``out``."""
    return [*ENS_LOCKED, *(os.path.relpath(m / LOCK, out) for m in members)]


def ensemble_scores(scored: Sequence[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The seed ensemble (§10.7) of :func:`score_split` outputs over the same rows: every raw (prior-corrected,
    uncalibrated) logit, :data:`ENS_MEAN` and GUID ``score_*``, is the mean over seeds; class probabilities (``p_c*``,
    ``pseg_c*``) are the softmax of the mean log-probability, i.e. of the mean class logits up to a per-row constant.
    The CORAL g (``ord_score``) is averaged like a logit, so an ordinal ensemble keeps one ranking. Calibration and
    thresholds are then fitted once, on the ensemble's val rows (:func:`lock_ensemble`)."""
    out = []
    for key, parts in ((["split", "guid", "seg_pos"], [s for s, _ in scored]), (["split", "guid"], [g for _, g in scored])):
        base = parts[0].copy()
        if any(not p[key].reset_index(drop=True).equals(base[key].reset_index(drop=True)) for p in parts[1:]):
            raise ValueError(f"ensemble members scored different rows (keys {key})")
        cols = [c for c in base if c in ENS_MEAN or c.startswith("score_")]
        base[cols] = np.mean([p[cols].to_numpy(np.float64) for p in parts], 0)
        for pre in ("p_c", "pseg_c"):
            pc = [f"{pre}{k}" for k in range(3)]
            if set(pc) <= set(base):
                lp = np.mean([np.log(np.clip(p[pc].to_numpy(np.float64), 1e-300, None)) for p in parts], 0)
                base[pc] = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        out.append(base)
    return out[0], out[1]


def _fold_unit(cfg: Config, run_dir: Path, source: Mapping[str, Any], fold: int, kind: str,
               built: Dict[bool, UnitData]) -> UnitData:
    """The fold's data for ``kind``, built once per context variant into ``built``: ``noind`` drops the missing
    flags (:func:`without_indicators`); every other kind shares the plain unit (``shuffled`` differs in train labels
    only, which its own ``train_unit`` rebuilds)."""
    variant = kind == "noind"
    if variant not in built:
        built[variant] = build_unit(without_indicators(cfg.classifier) if variant else cfg, run_dir, source, fold)
    return built[variant]


def _cache_rows(source: Mapping[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
    """The cache index (for :func:`label_rows`) and valid steps per cache row."""
    return (sources.open_cache(source["cache_dir"], source["fingerprint"]),
            load_store(str(source["cache_dir"])).step_mask.sum(1))


def unit_rows(labels: pd.DataFrame, scored: Tuple[pd.DataFrame, pd.DataFrame], cal: Mapping[str, Any], kind: str,
              seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """CONTRACT segment and GUID rows of one neural unit on one split: ``labels`` (:func:`label_rows` of that
    split) joined with :func:`score_split`'s raw logits, the pooling-attention scalars (:data:`ATTN_COLUMNS`, after
    the CONTRACT columns), and every score's ``*_cal`` twin (:func:`calibrate_frames`). Optional columns:
    ``score_<agg>[_cal]`` (segment scope); :data:`THREE_CLASS_COLUMNS` under a 3-class calibration record
    (``labels.task: three_class``), the raw ``p_c0..2`` of a binary task's aux head."""
    seg_s, gd_s = scored
    keys = ["split", "guid", "seg_pos"]
    three = [c for c in seg_s if c.startswith(("p_c", "pseg_c")) or c == "ord_score"]
    seg = labels.merge(seg_s[keys + ["logit_seg", "logit_online", *ATTN_COLUMNS, *three]], on=keys,
                       validate="one_to_one")
    last = labels.groupby(["split", "guid"], sort=False).tail(1)  # label_rows is sorted by (split, guid, seg_pos)
    scores = [c for c in gd_s if c.startswith("score_")]
    gd = last.merge(gd_s[["split", "guid", *scores, *(c for c in three if c in gd_s)]], on=["split", "guid"],
                    validate="one_to_one")
    if (len(seg), len(gd)) != (len(labels), len(last)):
        raise ValueError(f"{kind} {seed}: scored {len(seg)}/{len(labels)} segments, {len(gd)}/{len(last)} GUIDs")
    ident = {"model_id": kind, "seed": str(seed)}
    seg, gd = (f.assign(**ident) for f in calibrate_frames(seg, gd, cal))
    p = THREE_CLASS_COLUMNS if "head" in cal else [c for c in three if c.startswith("p_c")]
    extra = [c for s in scores if s != "score_final" for c in (s, f"{s}_cal")]
    return seg[SEGMENT_COLUMNS + ATTN_COLUMNS + p], gd[GUID_COLUMNS + extra + p]


def lock_unit(cfg: Config, out: Path, unit: UnitData, labels: pd.DataFrame, kind: str, seed: Any,
              mlflow_run_id: Optional[str], *, scored: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
              files: Sequence[str] = UNIT_LOCKED,
              note: str = "best.ckpt, scaler, calibration and thresholds, fixed on val before test is scored"
              ) -> Dict[str, Any]:
    """Step 4: calibration fit on the val GUID rows (§10.8, :func:`unit_calibration`), every threshold policy on the
    calibrated val rows (§11.3; on ``three_class`` also per class one-vs-rest, ``thresholds.json["ovr"]``), then
    ``selection_lock.json`` last, digesting ``files``. The val selections go to the unit's MLflow child (§10.10.5).
    ``unit`` may be the fold's unshuffled data for the shuffled unit: only train labels differ, never val.
    A unit with no causal per-position score (non-causal, no segment head) is thresholded like the shortcut
    (:func:`~teb_vae.classifier.thresholds.guid_level_policies`); one without a segment score (``segment_head:
    false``) skips its ``segment``-basis policies (recorded ``skipped``). ``scored``: the val scores, when they do not
    come from ``out``'s own checkpoint (a seed ensemble, :func:`lock_ensemble`). With ``eval.stage_offsets`` and a
    causal online score, the §11.3 stage view is selected here too: :data:`STAGE_FILES`, digested by the same lock."""
    c = cfg.classifier
    scored = score_split(out, unit, "val") if scored is None else scored
    cal = unit_calibration(c, scored[1])
    seg, gd = unit_rows(labels, scored, cal, kind, seed)
    policies, skipped = [p.model_dump() for p in c.eval.thresholds], {}
    if seg[["logit_online_cal", "logit_seg_cal"]].isna().all(axis=None):
        policies, ids = guid_level_policies(policies)
        skipped = dict.fromkeys(ids, ("guid", "guid-level model"))
    elif seg["logit_seg_cal"].isna().all():  # no segment score (model.segment_head: false): no segment basis
        skipped = {p["id"]: ("segment", "no segment score") for p in policies if p["basis"] == "segment"}
        policies = [p for p in policies if p["id"] not in skipped]
    kw = dict(bin_h=c.eval.bin_h, exclude_last_min=c.eval.exclude_last_min,
              staleness_h=c.eval.snapshot_max_staleness_h)
    thr = select_thresholds(seg, gd, policies, **kw)
    for pid, (level, why) in skipped.items():
        thr.setdefault(level, {})[pid] = {"skipped": why}
    picked = {f"val/{name}_{pid}": e[key] for per in thr.values() for pid, e in per.items() if "skipped" not in e
              for name, key in (("thr", "threshold"), ("sens", "val_sens"), ("fpr", "val_fpr"))}
    if ovr_enabled(c):
        thr["ovr"] = ovr_thresholds(seg, gd, policies, **kw)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    (out / "thresholds.json").write_text(json.dumps(thr, indent=2))  # plain json: -Infinity is legal
    files = list(files)
    if c.eval.stage_offsets and seg["logit_online_cal"].notna().any():  # §11.3: the `<kind>_stage` view, val-selected
        off = fit_stage_offsets(seg)
        s2, g2 = apply_stage_offsets(seg, gd, off, scope=c.model.scope, aggregators=c.model.segment_aggregators,
                                     tau=c.model.lse_tau)
        thr_stage = select_thresholds(s2, g2, policies, **kw)
        for pid, (level, why) in skipped.items():
            thr_stage.setdefault(level, {})[pid] = {"skipped": why}
        (out / STAGE_FILES[0]).write_text(json.dumps(off, indent=2))
        (out / STAGE_FILES[1]).write_text(json.dumps(thr_stage, indent=2))
        files += list(STAGE_FILES)
    lock = write_lock(out, digest(cfg), files, fold=unit.fold, seed=seed, kind=kind, note=note)
    if mlflow_run_id:
        _log_metrics(_mlflow(cfg).get("tracking_uri"), mlflow_run_id,
                     picked | {"calib_temperature": cal.get("temperature")})
    return {"calibration": cal, "locked_utc": lock["locked_utc"]}


def lock_ensemble(cfg: Config, run_dir: Path, unit: UnitData, labels: pd.DataFrame, kind: str,
                  seeds: Sequence[int]) -> Dict[str, Any]:
    """§10.7 for one fold and ``kind``: the members' val scores (:func:`score_split` of each ``best.ckpt``) are
    averaged (:func:`ensemble_scores`), then calibrated once and thresholded once on those val rows by
    :func:`lock_unit`, into :func:`ens_dir` (``seed='ens'``). The lock digests the members' locks too
    (:func:`ens_files`). Call it only when every member is locked."""
    out = ens_dir(run_dir, unit.fold, kind)
    out.mkdir(parents=True, exist_ok=True)
    members = [unit_dir(run_dir, unit.fold, s, kind) for s in seeds]
    # ponytail: re-scores the members' val split (lock_unit already did once each); cache it if online scoring is slow
    scored = ensemble_scores([score_split(m, unit, "val") for m in members])
    return lock_unit(cfg, out, unit, labels, kind, "ens", None, scored=scored, files=ens_files(out, members),
                     note=f"mean raw logits of seeds {list(seeds)}; calibration and thresholds fixed on val before "
                          f"test is scored; the members' locks are digested")


def neural_predict(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], splits: Sequence[str] = ("val", "test")
                   ) -> Tuple[list, list, Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """Rows of every neural unit from its saved files: ``(segments, guids, thresholds, units, missing)``. A unit that
    failed in train is only listed in ``missing``; any other unit without an intact lock refuses (L5). With
    ``eval.covariates_off`` and covariates, every non-shuffled unit is also scored with every covariate missing
    (§7.3.5), under its own locked calibration and thresholds, as ``model_id = <kind>_covoff``. A kind with several
    seeds adds its seed ensemble (:func:`ensembles`) as ``seed='ens'`` rows under the ensemble's own lock, which is
    read before any row of the fold is scored; a failed member makes the ensemble ``missing``. One fold at a time
    (:func:`predict_fold_rows`)."""
    segs, gds, thr, units, missing = [], [], {}, [], []
    for fold in cfg.classifier.run.folds:
        s, g, t, u, m = predict_fold_rows(cfg, run_dir, manifest, fold, splits)
        segs += s
        gds += g
        thr |= t
        units += u
        missing += m
    return segs, gds, thr, units, missing


def _stage_rows(c: Classifier, out: Path, seg: pd.DataFrame, gd: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The ``<kind>_stage`` rows (§11.3) of a unit's ``(seg, gd)``: :func:`apply_stage_offsets` under the offsets the
    unit at ``out`` locked (:data:`STAGE_FILES`); every other column, the calibration included, is the unit's."""
    off = json.loads((out / STAGE_FILES[0]).read_text())
    s, g = apply_stage_offsets(seg, gd, off, scope=c.model.scope, aggregators=c.model.segment_aggregators,
                               tau=c.model.lse_tau)
    model_id = f"{seg['model_id'].iloc[0]}{STAGE_SUFFIX}"
    return s.assign(model_id=model_id), g.assign(model_id=model_id)


def predict_fold_rows(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int,
                      splits: Sequence[str] = ("val", "test")
                      ) -> Tuple[list, list, Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """:func:`neural_predict` of one fold, in the same row order: what a fold process scores (§14.4). A unit (or
    ensemble) that locked :data:`STAGE_FILES` also yields its ``<kind>_stage`` rows under ``thresholds_stage.json``."""
    c, want, source = cfg.classifier, digest(cfg), manifest["source"]
    index, n_valid = _cache_rows(source)
    segs, gds, thr, units, missing = [], [], {}, [], []
    ens = ensembles(c, manifest.get("confound"))
    built, labels, ens_locks, scored = {}, None, {}, {}
    for kind, seeds in ens.items():
        if any(_read_json(d / "fold_results.json").get("status") == "failed"
               for d in [ens_dir(run_dir, fold, kind), *(unit_dir(run_dir, fold, s, kind) for s in seeds)]):
            logger.warning(f"ensemble {kind}|ens|{fold} or a member failed in train: no ensemble rows")
            missing.append(f"{kind}|ens|{fold}")
        else:
            ens_locks[kind] = read_lock(ens_dir(run_dir, fold, kind), want)  # L5
    for kind, seed in neural_units(c, manifest.get("confound")):
        out, key = unit_dir(run_dir, fold, seed, kind), f"{kind}|{seed}|{fold}"
        if _read_json(out / "fold_results.json").get("status") == "failed":
            logger.warning(f"unit {key} failed in train: no prediction rows")
            missing.append(key)
            continue
        lock = read_lock(out, want)  # before any test row is scored (L5)
        unit = _fold_unit(cfg, run_dir, source, fold, kind, built)
        if labels is None:
            labels = label_rows(run_dir, fold, index, splits, n_valid, c.cohort.min_segments_per_guid,
                                strategy=c.labels.strategy)
        cal = json.loads((out / "calibration.json").read_text())
        passes = [(kind, unit)]
        if c.eval.covariates_off and unit.n_cov and kind != "shuffled":
            passes.append((f"{kind}_covoff", covariates_off(unit)))
        staged = (out / STAGE_FILES[0]).is_file()
        for model_id, u in passes:
            for split in splits:
                sc = score_split(out, u, split)
                if kind in ens_locks:
                    scored.setdefault((model_id, split), []).append(sc)
                s, g = unit_rows(labels[labels["split"] == split], sc, cal, model_id, seed)
                segs.append(s)
                gds.append(g)
                if staged and model_id == kind:
                    s2, g2 = _stage_rows(c, out, s, g)
                    segs.append(s2)
                    gds.append(g2)
            thr[f"{model_id}|{seed}|{fold}"] = json.loads((out / "thresholds.json").read_text())
            units.append({"model_id": model_id, "seed": str(seed), "fold": fold,
                          "lock_written_at": lock["locked_utc"]})
            if staged and model_id == kind:
                thr[f"{kind}{STAGE_SUFFIX}|{seed}|{fold}"] = json.loads((out / STAGE_FILES[1]).read_text())
                units.append({"model_id": f"{kind}{STAGE_SUFFIX}", "seed": str(seed), "fold": fold,
                              "lock_written_at": lock["locked_utc"]})
    for kind, lock in ens_locks.items():
        out = ens_dir(run_dir, fold, kind)
        cal = json.loads((out / "calibration.json").read_text())
        staged = (out / STAGE_FILES[0]).is_file()
        for model_id in dict.fromkeys(m for m, _ in scored if m in (kind, f"{kind}_covoff")):
            for split in splits:
                s, g = unit_rows(labels[labels["split"] == split], ensemble_scores(scored[model_id, split]), cal,
                                 model_id, "ens")
                segs.append(s)
                gds.append(g)
                if staged and model_id == kind:
                    s2, g2 = _stage_rows(c, out, s, g)
                    segs.append(s2)
                    gds.append(g2)
            thr[f"{model_id}|ens|{fold}"] = json.loads((out / "thresholds.json").read_text())
            units.append({"model_id": model_id, "seed": "ens", "fold": fold, "lock_written_at": lock["locked_utc"]})
            if staged and model_id == kind:
                thr[f"{kind}{STAGE_SUFFIX}|ens|{fold}"] = json.loads((out / STAGE_FILES[1]).read_text())
                units.append({"model_id": f"{kind}{STAGE_SUFFIX}", "seed": "ens", "fold": fold,
                              "lock_written_at": lock["locked_utc"]})
    return segs, gds, thr, units, missing


def _unit_state(run_dir: Path, key: str, **fields: Any) -> None:
    """§10.10.6: one unit's status and timestamps under ``stage_state.json["train"]["units"]``."""
    path = run_dir / "stage_state.json"
    state = _read_json(path)
    units = state.setdefault("train", {}).setdefault("units", {})
    units[key] = {**units.get(key, {}), **fields}
    _write_json(path, state)


def _progress(cfg: Config, run_dir: Path, rec: Mapping[str, Any]) -> None:
    """Step 5: one ``kfold_progress.log`` line per trained unit (a header when the file is new). The header names the
    run's selection monitor (``_gated`` under cotrain); frozen/shuffled units record their own in ``fold_results.json``."""
    path = run_dir / "kfold_progress.log"
    monitor = selection_monitor(cfg.advanced_config, cfg.classifier.train.regime)[0]
    auroc = (rec.get("best_val") or {}).get("val/guid_auroc")
    cells = [_now(), rec["fold"], rec["seed"], rec["kind"], rec["status"], rec.get("train_seconds", ""),
             rec.get("best_epoch", ""), f"{rec['best_score']:.4f}" if "best_score" in rec else "",
             "" if auroc is None else f"{auroc:.4f}", ((rec.get("error") or "").strip().splitlines() or [""])[-1]]
    with path.open("a") as fh:
        if fh.tell() == 0:
            fh.write(f"ts | fold | seed | kind | status | train_s | best_epoch | best_{monitor} | val/guid_auroc | "
                     f"error\n")
        fh.write(" | ".join(str(x).replace("|", "/") for x in cells) + "\n")


def kfold_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """§10.10.4: per-unit best metrics; mean/SD/min/max across folds of the best val metrics per ``kind|seed``."""
    units = [{k: r.get(k) for k in ("fold", "seed", "kind", "status", "best_epoch", "monitor", "best_score",
                                    "train_seconds", "unit_dir")}
             | {"val": {k: v for k, v in (r.get("best_val") or {}).items() if k.startswith("val/")}} for r in records]
    done = pd.DataFrame([{"unit": f"{u['kind']}|{u['seed']}", **u["val"]} for u in units if u["status"] == "done"])
    across = ({k: g.drop(columns="unit").agg(["mean", "std", "min", "max"]).to_dict() for k, g in done.groupby("unit")}
              if len(done) else {})
    return {"units": units, "across_folds": across,
            "missing_units": [f"{u['kind']}|{u['seed']}|{u['fold']}" for u in units if u["status"] != "done"]}


# ---- MLflow parent (§10.10.5) ---------------------------------------------------------------------
def _mlflow(cfg: Config) -> Dict[str, Any]:
    return (cfg.advanced_config.get("tracking") or {}).get("mlflow") or {}


def _mlflow_off(cfg: Config) -> Config:
    """``cfg`` with MLflow disabled, for the units only (the digest is always taken from ``cfg``)."""
    off = cfg.model_copy(deep=True)
    off.advanced_config.setdefault("tracking", {}).setdefault("mlflow", {})["enabled"] = False
    return off


def mlflow_parent(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> Optional[str]:
    """The parent run's id, created once per run directory. ``manifest["mlflow"]`` records ``{enabled, status:
    ok|disabled|unreachable, parent_run_id, reason}`` (F9; ``summary.md`` §1 prints it). Fail-closed: when the
    parent cannot be created, a WARNING and None; the caller then trains every unit with MLflow off, and ``report``
    logs nothing. Tags: the framework's provenance tags plus ``kind=parent``, task, source; params: the flattened
    ``classifier:`` block; artifact: ``config.resolved.yaml``."""
    settings, c = _mlflow(cfg), cfg.classifier
    if not settings.get("enabled"):
        manifest["mlflow"] = {"enabled": False, "status": "disabled", "parent_run_id": None,
                              "reason": "advanced_config.tracking.mlflow.enabled is false"}
        return None
    if (manifest.get("mlflow") or {}).get("parent_run_id"):
        return manifest["mlflow"]["parent_run_id"]
    uri = settings.get("tracking_uri")
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Param
        from train.graph_model_base import GraphModelBase

        client = MlflowClient(tracking_uri=uri)
        name = settings.get("experiment_name") or c.run.name
        exp = client.get_experiment_by_name(name)
        exp_id = exp.experiment_id if exp else client.create_experiment(
            name, artifact_location=settings.get("artifact_location"))
        unit = unit_config(cfg, run_dir=run_dir, fold=c.run.folds[0], seed=c.run.seeds[0], kind="model")
        fingerprint = (manifest.get("source") or {}).get("fingerprint") or {}
        tags = {**(settings.get("tags") or {}), **GraphModelBase._collect_provenance_tags(
            SimpleNamespace(config=unit, cuda_devices=unit["general_config"]["cuda_devices"])),
            "kind": "parent", "task": c.labels.task, "source_kind": c.source.kind,
            "vae_package": c.source.vae.package if c.source.kind == "vae" else "n/a",
            "vae_ckpt_sha": fingerprint.get("checkpoint_sha256") or "n/a", "n_folds": str(len(c.run.folds)),
            "seeds": ",".join(map(str, c.run.seeds)), "config_digest": manifest["config_digest"],
            "run_dir": str(run_dir)}
        run_id = client.create_run(exp_id, run_name=c.run.name, tags=tags).info.run_id
        params = [Param(k, v) for k, v in GraphModelBase._flatten_config(
            cfg.model_dump(mode="json")["classifier"], "classifier").items()]
        for i in range(0, len(params), 100):  # MLflow's batch cap
            client.log_batch(run_id, params=params[i:i + 100])
        client.log_artifact(run_id, str(run_dir / "config.resolved.yaml"))
    except Exception as exc:  # noqa: BLE001 - F9: tracking never stops a run
        logger.warning(f"F9: MLflow parent run could not be created at {uri} ({exc!r}); MLflow is DISABLED for "
                       f"every unit and for evaluate/report")
        manifest["mlflow"] = {"enabled": False, "status": "unreachable", "parent_run_id": None,
                              "reason": f"{uri}: {exc!r}"}
        _write_json(run_dir / "manifest.json", manifest)
        return None
    manifest["mlflow"] = {"enabled": True, "status": "ok", "parent_run_id": run_id, "reason": None}
    _write_json(run_dir / "manifest.json", manifest)
    logger.info(f"MLflow parent run {run_id} (experiment {name!r}, {uri})")
    return run_id


def _log_metrics(uri: Optional[str], run_id: str, values: Mapping[str, Any]) -> None:
    """Finite ``values`` to ``run_id`` in one batch per 1000; ``@`` (not a legal MLflow name character) becomes
    ``_at_``. Fail-closed: an error only warns."""
    try:
        from mlflow import MlflowClient
        from mlflow.entities import Metric

        stamp = int(time.time() * 1000)
        batch = [Metric(k.replace("@", "_at_"), float(v), stamp, 0) for k, v in values.items()
                 if isinstance(v, (int, float)) and math.isfinite(v)]
        client = MlflowClient(tracking_uri=uri)
        for i in range(0, len(batch), 1000):
            client.log_batch(run_id, metrics=batch[i:i + 1000])
    except Exception as exc:  # noqa: BLE001 - F9
        logger.warning(f"F9: MLflow metric logging to run {run_id} failed: {exc!r}")


def log_parent(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> None:
    """F8: the evaluation results to the parent via ``MlflowClient`` (fail-closed). Metrics from
    ``results.headline``: ``[<model>/][seed<s>/]{pooled_test,foldmean_test}/<level>/<metric>[_at_<policy>]``, where
    the model prefix is dropped for ``model`` and the seed one when the model has one seed or for its seed ensemble. Artifacts:
    ``summary.md``, ``summary.json``, ``tables/metrics.parquet``, ``figures/**``."""
    rec = manifest.get("mlflow") or {}
    if not rec.get("parent_run_id"):
        return
    ev, run_id, uri = run_dir / "evaluation", rec["parent_run_id"], _mlflow(cfg).get("tracking_uri")
    head = list(((_read_json(ev / "summary.json").get("results") or {}).get("headline") or {}).values())
    seeds: Dict[str, set] = {}
    for e in head:
        seeds.setdefault(e["model_id"], set()).add(e["seed"])
    values = {}
    for e in head:
        m = e["model_id"]
        prefix = ("" if m == "model" else f"{m}/") + ("" if len(seeds[m]) == 1 or e["seed"] == "ens" else
                                                      f"seed{e['seed']}/")
        name = f"{e['level']}/{e['metric']}" + ("" if e["policy_id"] == "threshold_free" else f"@{e['policy_id']}")
        values |= {f"{prefix}pooled_test/{name}": e["test"], f"{prefix}foldmean_test/{name}": e["fold_mean"]}
    _log_metrics(uri, run_id, values)
    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        for path, where in ((run_dir / report.SUMMARY_MD, None), (ev / "summary.json", None),
                            (ev / "tables" / "metrics.parquet", "tables")):
            if path.is_file():
                client.log_artifact(run_id, str(path), where)
        if (ev / "figures").is_dir():
            client.log_artifacts(run_id, str(ev / "figures"), "figures")
        client.set_terminated(run_id)
    except Exception as exc:  # noqa: BLE001 - F9
        logger.warning(f"F9: MLflow parent artifact logging failed: {exc!r}")
        rec["reason"] = f"report logging failed: {exc!r}"


def stage_cohort(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> None:
    """§6 tables and fold checks into ``cohort/``; shards, counts, exposure and confound into the manifest."""
    result = cohort.build_cohort(cfg.classifier, run_dir / "cohort")
    manifest["shards"] = [shard_record(path) for path in result.pop("shards")]
    manifest["exposure"] = result.pop("exposure")
    manifest["confound"] = result.pop("confound")
    manifest["cohort"] = result


def stage_extract(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> None:
    """§8.4: the (frozen) source once per unique cohort segment into the fingerprinted cache. Online regimes need it
    too: it carries the frames, the train scaler, the priors and their ``frozen`` baseline unit (§10.1)."""
    c = cfg.classifier
    segments = pd.read_parquet(run_dir / "cohort" / "segments.parquet")
    shards = {(k, s): p for k in c.run.folds for s, p in cohort.fold_shards(c.data, k).items()}
    source = sources.make_source(c.source, next(iter(shards.values()))[0], device=c.run.device)
    manifest["source"] = sources.extract(source, segments, shards, cache_root=c.source.cache_root,
                                         dtype=c.source.cache_dtype)


def train_fold(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int, parent: Optional[str], *,
               state: Optional[Callable[..., None]] = None,
               progress: Optional[Callable[[Mapping[str, Any]], None]] = None) -> Dict[str, Any]:
    """§10.10.1 for one fold: every neural unit (a locked one is skipped), then the fold's seed ensembles (§10.7).

    Args:
        cfg: The run's config.
        run_dir: The run directory.
        manifest: The run manifest (``source``, ``confound``).
        fold: The fold.
        parent: The MLflow parent run id; None trains every unit with MLflow off when MLflow is enabled (F9).
        state: ``state(key, **fields)`` records a unit's status; default :func:`_unit_state` (``stage_state.json``).
        progress: ``progress(record)`` records a unit trained here; default :func:`_progress` (``kfold_progress.log``).
            A fold process (§14.4) passes its own pair, because only the parent writes run-level files.

    Returns:
        ``{"records": [...], "ens_failed": [...]}``: the ``fold_results.json`` record of every unit, trained here or
        skipped as locked, and for each seed ensemble selected here whether it failed.
    """
    c = cfg.classifier
    state = state or (lambda key, **fields: _unit_state(run_dir, key, **fields))
    progress = progress or (lambda rec: _progress(cfg, run_dir, rec))
    ucfg = cfg if parent or not _mlflow(cfg).get("enabled") else _mlflow_off(cfg)  # F9: fail-closed
    want, source = digest(cfg), manifest["source"]
    index, n_valid = _cache_rows(source)
    records, ens_failed = [], []
    built, labels = {}, None  # the fold's data per context variant, shared by every seed (and by val selection)
    for kind, seed in neural_units(c, manifest.get("confound")):
        out, key = unit_dir(run_dir, fold, seed, kind), f"{kind}|{seed}|{fold}"
        if (out / LOCK).is_file():
            read_lock(out, want)  # a lock under another config or with changed files raises (L5)
            logger.info(f"unit {key}: locked at digest {want[:12]}, skipped")
            state(key, status="done", digest=want)
            records.append(_read_json(out / "fold_results.json"))
            continue
        unit = _fold_unit(cfg, run_dir, source, fold, kind, built)
        if labels is None:
            labels = label_rows(run_dir, fold, index, ["val"], n_valid, c.cohort.min_segments_per_guid,
                                strategy=c.labels.strategy)
        state(key, status="running", started=_now(), finished=None, digest=want)
        rec = {"fold": fold, "seed": seed, "kind": kind, "status": "failed"}
        try:
            rec = train_unit(ucfg, run_dir, manifest, fold=fold, seed=seed, kind=kind,
                             unit=None if kind == "shuffled" else unit, mlflow_parent_id=parent)
            if rec["status"] == "done":
                try:
                    rec |= lock_unit(cfg, out, unit, labels, kind, seed, rec.get("mlflow_run_id"))
                except Exception:
                    logger.exception(f"unit {key}: calibration / threshold selection failed")
                    rec |= {"status": "failed", "error": traceback.format_exc()}
                    (out / "fold_results.json").write_text(json.dumps(json_safe(rec), indent=2))
                    if c.run.fail_fast:
                        raise
        finally:
            state(key, status=rec["status"], finished=_now())
            progress(rec)
        records.append(rec)
    for kind, seeds in ensembles(c, manifest.get("confound")).items():  # §10.7, once every member is locked
        out, key = ens_dir(run_dir, fold, kind), f"{kind}|ens|{fold}"
        if not all((unit_dir(run_dir, fold, s, kind) / LOCK).is_file() for s in seeds):
            logger.warning(f"ensemble {key}: a member is not locked (it failed); no ensemble")
            continue
        if lock_problem(out, want) is None:  # a retrained member changed its lock: re-select below
            logger.info(f"ensemble {key}: locked at digest {want[:12]}, skipped")
            state(key, status="done", digest=want)
            continue
        if labels is None:
            labels = label_rows(run_dir, fold, index, ["val"], n_valid, c.cohort.min_segments_per_guid,
                                strategy=c.labels.strategy)
        state(key, status="running", started=_now(), finished=None, digest=want)
        rec = {"fold": fold, "seed": "ens", "kind": kind, "status": "failed"}
        out.mkdir(parents=True, exist_ok=True)
        try:
            (out / LOCK).unlink(missing_ok=True)
            rec |= lock_ensemble(cfg, run_dir, _fold_unit(cfg, run_dir, source, fold, kind, built), labels, kind,
                                 seeds) | {"status": "done"}
        except Exception:
            logger.exception(f"ensemble {key}: calibration / threshold selection failed")
            rec["error"] = traceback.format_exc()
            if c.run.fail_fast:
                raise
        finally:
            _write_json(out / "fold_results.json", rec)
            state(key, status=rec["status"], finished=_now())
        ens_failed.append(rec["status"] != "done")
    return {"records": records, "ens_failed": ens_failed}


def stage_train(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> int:
    """§14.1 train per fold: the sklearn baselines, then every neural unit (§10.10.1), fold after fold on
    ``run.device`` or, with ``run.devices``, in fold processes (§14.4); 1 iff a unit or a seed ensemble failed."""
    c = cfg.classifier
    manifest["baselines"] = baselines.train(cfg, run_dir, manifest["source"])
    parent = mlflow_parent(cfg, run_dir, manifest)
    if c.run.devices:
        results = fan_out(cfg, run_dir, manifest, "train", parent)
    else:
        results = {fold: train_fold(cfg, run_dir, manifest, fold, parent) for fold in c.run.folds}
    records = [rec for fold in c.run.folds for rec in results[fold]["records"]]
    _write_json(run_dir / "kfold_summary.json", kfold_summary(records))
    return int(any(r.get("status") != "done" for r in records)
               or any(failed for res in results.values() for failed in res["ens_failed"]))


def attribution_fold(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int,
                     split: str = "test") -> pd.DataFrame:
    """E6 (§11.12, ``eval.attribution.enabled``) of one fold: :func:`~teb_vae.classifier.train.attribute_split` of
    every locked ``model`` unit (each seed; an ensemble's IG is the mean of its members', IG being linear in the score)
    on ``split``, with ``eval.attribution.n_steps`` quadrature points. A unit that failed in train is skipped (it is
    already in ``missing_units``); any other must hold its lock (L5). Columns :data:`ATTRIBUTION_COLUMNS`."""
    c, want, source = cfg.classifier, digest(cfg), manifest["source"]
    frames, built = [], {}
    for seed in c.run.seeds:
        out = unit_dir(run_dir, fold, seed, "model")
        if _read_json(out / "fold_results.json").get("status") == "failed":
            continue
        read_lock(out, want)
        unit = _fold_unit(cfg, run_dir, source, fold, "model", built)
        frames.append(attribute_split(out, unit, split, int(c.eval.attribution["n_steps"]))
                      .assign(fold=fold, seed=str(seed), split=split))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).reindex(columns=ATTRIBUTION_COLUMNS)


def predict_fold(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int) -> Dict[str, Any]:
    """The neural part of ``predict`` for one fold: :func:`predict_fold_rows` of val and test as ``segments``,
    ``guids`` (lists of frames), ``thresholds``, ``units``, ``missing``, plus the fold's ``attribution``
    (:func:`attribution_fold`; None unless ``eval.attribution.enabled``)."""
    part = dict(zip(("segments", "guids", "thresholds", "units", "missing"),
                    predict_fold_rows(cfg, run_dir, manifest, fold)))
    enabled = cfg.classifier.eval.attribution["enabled"]
    return part | {"attribution": attribution_fold(cfg, run_dir, manifest, fold) if enabled else None}


def stage_predict(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> None:
    """§11.1: val and test rows of every model into ``predictions/`` with thresholds and provenance (L5). The neural
    units are scored per fold (:func:`predict_fold`), fold after fold or, with ``run.devices``, in fold processes
    (§14.4); either way the tables are concatenated in fold order, so they are the same rows in the same order."""
    c = cfg.classifier
    segments, guids, thresholds, units = baselines.predict(cfg, run_dir, manifest["source"])
    by_fold = (fan_out(cfg, run_dir, manifest, "predict") if c.run.devices else
               {fold: predict_fold(cfg, run_dir, manifest, fold) for fold in c.run.folds})
    parts = [by_fold[fold] for fold in c.run.folds]
    segments = pd.concat([segments, *(s for p in parts for s in p["segments"])], ignore_index=True)
    guids = pd.concat([guids, *(g for p in parts for g in p["guids"])], ignore_index=True)
    thresholds = thresholds | {k: v for p in parts for k, v in p["thresholds"].items()}
    units, missing = units + [u for p in parts for u in p["units"]], [m for p in parts for m in p["missing"]]
    out, written = run_dir / "predictions", baselines.utc_now()
    for unit in units:  # the lock was checked before any test row was scored; this records the order
        unit["test_written_at"] = written
        if not unit["lock_written_at"] < written:
            raise RuntimeError(f"L5: unit {unit} was locked after its test rows were written")
    out.mkdir(exist_ok=True)
    segments.to_parquet(out / "segments.parquet", index=False)
    guids.to_parquet(out / "guids.parquet", index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))  # plain json: -Infinity is legal
    (out / "attribution.parquet").unlink(missing_ok=True)  # never a stale table from an earlier predict
    if c.eval.attribution["enabled"]:
        frames = [p["attribution"] for p in parts if len(p["attribution"])]
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()).reindex(
            columns=ATTRIBUTION_COLUMNS).to_parquet(out / "attribution.parquet", index=False)
    _write_json(out / "provenance.json", {"config_digest": manifest["config_digest"], "run_id": run_dir.name,
                                          "written_at": written, "units": units, "missing_units": missing})


# ---- fold processes (§14.4) -----------------------------------------------------------------------
#: What a fold process runs; everything else (cohort, extract, the sklearn baselines, the MLflow parent, evaluate,
#: report, verify) runs once, in the parent.
FOLD_JOBS = ("train", "predict")


def worker_dir(run_dir: Path, fold: int, job: str) -> Path:
    """``<run>/folds/fold_<k>/worker/<job>/``: a fold process's ``output.log`` (stdout and stderr), its loguru pair
    ``fold.log``/``fold.jsonl``, and its result (``units.json`` + ``part.json`` for train, ``part.pkl`` for predict)."""
    return Path(run_dir) / "folds" / f"fold_{fold}" / "worker" / job


def fold_job(fold: int, device: str, job: str, cfg_json: Dict[str, Any], run_dir: str, manifest: Dict[str, Any],
             parent: Optional[str]) -> None:
    """Body of a fold process (:func:`~teb_vae.classifier.parallel.run_folds`): ``job`` of one fold on ``device``.

    ``train`` is :func:`train_fold`: each unit's status goes to ``units.json`` as it changes, the fold's result and
    the records of the units trained here (for ``kfold_progress.log``) to ``part.json`` at the end. ``predict`` is
    :func:`predict_fold` into ``part.pkl``, a pickle so the parent concatenates exactly the frames the serial path
    would. The process writes only under its fold's directories; :func:`fan_out` merges the result.

    Args:
        fold: The fold.
        device: This process's device (``run.device`` for everything it runs).
        job: One of :data:`FOLD_JOBS`.
        cfg_json: ``cfg.model_dump(mode="json")`` of the run's config.
        run_dir: The run directory.
        manifest: The run manifest.
        parent: The MLflow parent run id (train).

    Raises:
        RuntimeError: If the config does not reproduce the run's digest.
    """
    root = Path(run_dir)
    work = worker_dir(root, fold, job)
    classifier_train.PROCESS_LOGS = (str((work / "fold.log").relative_to(root)),
                                     str((work / "fold.jsonl").relative_to(root)))
    setup_logging(file_path=str(work / "fold.log"), json_path=str(work / "fold.jsonl"), compression=None)
    cfg = Config.model_validate(cfg_json)
    cfg.classifier.run.device = device
    if digest(cfg) != manifest["config_digest"]:
        raise RuntimeError(f"fold {fold}: the config reached the fold process with digest {digest(cfg)[:12]}, not "
                           f"the run's {manifest['config_digest'][:12]}")
    if job == "train":
        units: Dict[str, Dict[str, Any]] = {}
        trained: List[Mapping[str, Any]] = []

        def state(key: str, **fields: Any) -> None:
            units[key] = {**units.get(key, {}), **fields}
            _write_json(work / "units.json", units)

        result = train_fold(cfg, root, manifest, fold, parent, state=state, progress=trained.append)
        _write_json(work / "part.json", result | {"trained": trained, "device": device})
    elif job == "predict":
        pd.to_pickle(predict_fold(cfg, root, manifest, fold), work / "part.pkl")
    else:
        raise ValueError(f"fold job must be one of {FOLD_JOBS}, got {job!r}")
    logger.info(f"fold {fold}: {job} done on {device}")


def fan_out(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], job: str,
            parent: Optional[str] = None) -> Dict[int, Any]:
    """``job`` of every fold of ``run.folds`` in fold processes over ``run.devices`` (§14.4); ``{fold: result}``.

    Each fold's worker dir is emptied first, so no result outlives the process that wrote it. This process stays the
    only writer of run-level files. ``train``: as each fold process ends, its unit states and trained units are merged
    into ``stage_state.json`` and ``kfold_progress.log`` (:func:`_merge_train`); a process that died leaves its
    unlocked units ``failed`` (:func:`_dead_fold`), and under ``run.fail_fast`` the stage raises once every fold has
    ended. ``predict``: the result is the fold's :func:`predict_fold`; a fold whose process failed makes the stage
    raise once every fold has ended.

    Raises:
        RuntimeError: As above, naming each failed fold's ``output.log``.
    """
    c = cfg.classifier
    for fold in c.run.folds:
        shutil.rmtree(worker_dir(run_dir, fold, job), ignore_errors=True)
    results: Dict[int, Any] = {}

    def on_exit(fold: int, code: int) -> None:
        work = worker_dir(run_dir, fold, job)
        if job == "train":
            results[fold] = _merge_train(cfg, run_dir, manifest, fold, code, work)
        elif code == 0:
            results[fold] = pd.read_pickle(work / "part.pkl")
            (work / "part.pkl").unlink()  # the rows now live in predictions/; the logs stay

    codes = parallel.run_folds("teb_vae.classifier.run:fold_job", c.run.folds, c.run.devices,
                               (job, cfg.model_dump(mode="json"), str(run_dir), dict(manifest), parent),
                               lambda fold: worker_dir(run_dir, fold, job) / "output.log", on_exit)
    failed = sorted(fold for fold, code in codes.items() if code != 0)
    if failed and (job == "predict" or c.run.fail_fast):
        raise RuntimeError(f"{job} failed in the fold process of fold(s) {failed}; see "
                           f"{[str(worker_dir(run_dir, fold, job) / 'output.log') for fold in failed]}")
    return results


def _merge_train(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int, code: int,
                 work: Path) -> Dict[str, Any]:
    """A train fold process's result into the run-level files: its unit states into ``stage_state.json``, then one
    ``kfold_progress.log`` line per unit it trained. A process that died (non-zero exit, or no ``part.json``) is
    recorded by :func:`_dead_fold`. Returns the fold's :func:`train_fold` result."""
    for key, fields in _read_json(work / "units.json").items():
        _unit_state(run_dir, key, **fields)
    part = _read_json(work / "part.json") if code == 0 else {}
    if not part:
        part = _dead_fold(cfg, run_dir, manifest, fold,
                          f"fold process exited with code {code} before its result; see {work / 'output.log'}")
    for rec in part["trained"]:
        _progress(cfg, run_dir, rec)
    return part


def _dead_fold(cfg: Config, run_dir: Path, manifest: Mapping[str, Any], fold: int, error: str) -> Dict[str, Any]:
    """The train result of a fold whose process died. A unit (or seed ensemble) with an intact lock keeps its record;
    every other one is recorded ``failed`` with ``error``, in its ``fold_results.json`` and ``stage_state.json``, so
    ``predict`` lists it as missing instead of refusing an absent lock, and the next ``--stage train`` retrains it."""
    c, want = cfg.classifier, digest(cfg)
    records, trained, ens_failed = [], [], []
    for kind, seed in neural_units(c, manifest.get("confound")):
        out, key = unit_dir(run_dir, fold, seed, kind), f"{kind}|{seed}|{fold}"
        if lock_problem(out, want) is None:
            records.append(_read_json(out / "fold_results.json"))
            continue
        rec = _read_json(out / "fold_results.json") | {"fold": fold, "seed": seed, "kind": kind, "status": "failed",
                                                        "error": error}
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "fold_results.json", rec)
        _unit_state(run_dir, key, status="failed", finished=_now())
        records.append(rec)
        trained.append(rec)
    for kind in ensembles(c, manifest.get("confound")):
        out, key = ens_dir(run_dir, fold, kind), f"{kind}|ens|{fold}"
        if lock_problem(out, want) is None:
            continue
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "fold_results.json", {"fold": fold, "seed": "ens", "kind": kind, "status": "failed",
                                                "error": error})
        _unit_state(run_dir, key, status="failed", finished=_now())
        ens_failed.append(True)
    return {"records": records, "ens_failed": ens_failed, "trained": trained}


def stage_evaluate(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> int:
    """§11.12 analyses (fail-soft; 1 iff a step raised). ``--only``/``--skip`` run a partial evaluation into
    ``evaluation/partial_<stamp>/``; report and verify keep reading the last full one. A run with failed units is
    refused unless ``--allow-partial`` (§10.10.1)."""
    arguments = manifest.get("arguments", {})
    missing = _read_json(run_dir / "predictions" / "provenance.json").get("missing_units")
    if missing and not arguments.get("allow_partial"):
        logger.error(f"evaluate: {len(missing)} unit(s) failed in train: {missing}")
        raise RuntimeError(f"units {missing} failed in train; re-run --stage train, or pool the rest with "
                           f"--allow-partial")
    if missing:
        logger.warning(f"evaluate --allow-partial: pooling without the failed units {missing}")
    return metrics.evaluate(run_dir, cfg, only=arguments.get("only"), skip=arguments.get("skip"),
                            allow_partial=bool(arguments.get("allow_partial")))


def stage_report(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> int:
    """Figures and ``summary.md`` from the evaluation tables (fail-soft; 1 iff a step raised); then the
    results go to the MLflow parent (F8)."""
    code = report.report(run_dir, cfg)
    log_parent(cfg, run_dir, manifest)
    return code


def stage_verify(cfg: Config, run_dir: Path, manifest: Dict[str, Any]) -> int:
    """§11.15 gate over ``evaluation/summary.json``; 1 on any FAIL. Verdicts: ``evaluation/verify.json``."""
    return verify.main([str(run_dir), "--json-out", str(run_dir / "evaluation" / "verify.json")])


STAGES: Dict[str, Callable[[Config, Path, Dict[str, Any]], Optional[int]]] = {
    "cohort": stage_cohort, "extract": stage_extract, "train": stage_train, "predict": stage_predict,
    "evaluate": stage_evaluate, "report": stage_report, "verify": stage_verify}
#: Stages that only read earlier outputs: named explicitly, they run even when done.
RERUNNABLE = ("evaluate", "report", "verify")


def main(*, config: str, stage: str = "all", overrides: Optional[List[str]] = None,
         run_dir: Optional[str] = None, folds: Optional[str] = None,
         device: Optional[str] = None, devices: Optional[str] = None, only: Optional[str] = None,
         skip: Optional[str] = None, allow_partial: bool = False) -> Path:
    """Run ``stage`` (or every stage) into ``run_dir``; returns the run directory.

    Args:
        config: YAML path, relative to the repository root or absolute.
        stage: A key of :data:`STAGES`, or ``"all"``.
        overrides: ``classifier.x.y=value`` strings (YAML-parsed).
        run_dir: Existing or new directory; default ``<out_root>/<stamp>-<run.name>``.
        folds: Comma-separated fold ids, overriding ``run.folds``.
        device: Overrides ``run.device`` (not part of the digest).
        devices: Comma-separated device slots overriding ``run.devices`` (not part of the digest), e.g.
            ``cuda:0,cuda:1``: train and predict run one fold process per slot (§14.4).
        only: Comma-separated analysis ids for ``evaluate`` (default: all).
        skip: Comma-separated analysis ids ``evaluate`` leaves out.
        allow_partial: Let ``evaluate`` pool a run whose ``train`` had failed units.

    Raises:
        ValueError: If ``run_dir`` holds a run with a different config digest.
    """
    overrides = list(overrides or [])
    if folds:
        overrides.append(f"classifier.run.folds=[{folds}]")
    if device:
        overrides.append(f"classifier.run.device={device}")
    if devices:
        overrides.append(f"classifier.run.devices=[{devices}]")
    cfg = load(resolve_path(config), overrides)
    run = cfg.classifier.run
    run_path = (Path(run_dir) if run_dir else
                resolve_path(run.out_root) / f"{time.strftime('%Y-%m-%d--%H-%M-%S')}-{run.name}")
    run_path.mkdir(parents=True, exist_ok=True)
    setup_logging(file_path=str(run_path / "run.log"), json_path=str(run_path / "run.jsonl"),
                  compression=None)

    manifest_path, state_path = run_path / "manifest.json", run_path / "stage_state.json"
    manifest, state, config_digest = _read_json(manifest_path), _read_json(state_path), digest(cfg)
    if manifest.get("config_digest", config_digest) != config_digest:
        logger.error(f"digest mismatch in {run_path}: {manifest['config_digest'][:12]} on disk vs "
                     f"{config_digest[:12]} now")
        raise ValueError(f"{run_path} was run with a different config (digest "
                         f"{manifest['config_digest'][:12]} != {config_digest[:12]}); use a new "
                         f"--run-dir")
    (run_path / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    manifest.update(config_digest=config_digest, software=software_record(),
                    versions={name: version(name) for name in VERSIONED},
                    arguments={"config": str(config), "stage": stage, "overrides": overrides,
                               "only": only.split(",") if only else None,
                               "skip": skip.split(",") if skip else None, "allow_partial": allow_partial})
    _write_json(manifest_path, manifest)

    for name in list(STAGES) if stage == "all" else [stage]:
        state = _read_json(state_path)  # a stage writes its own per-unit records (train)
        record = state.get(name, {})
        if (record.get("status") == "done" and not record.get("exit_code")
                and not (name == stage and name in RERUNNABLE)):
            logger.info(f"stage {name}: already done at digest {config_digest[:12]}, skipped")
            continue
        if name == "evaluate" and (only or skip):  # a partial look (evaluation/partial_<stamp>/): state untouched
            code = STAGES[name](cfg, run_path, manifest)
            (logger.info if code == 0 else logger.warning)(f"partial evaluate: exit code {code}; not recorded")
            continue
        units ={k: {"units": rec["units"]} if "units" in rec else {} for k, rec in state.items()}
        later = list(STAGES)[list(STAGES).index(name) + 1:]  # their inputs are about to change
        state |= {k: {"status": "pending", **units[k]} for k in later if k in state}
        state[name] = {"status": "running", "started": _now(), **units.get(name, {})}
        _write_json(state_path, state)
        started = time.perf_counter()
        logger.info(f"stage {name}: start")
        try:
            code = STAGES[name](cfg, run_path, manifest) or 0
        except Exception:
            state = _read_json(state_path)
            state[name].update(status="failed", finished=_now())
            _write_json(state_path, state)
            raise
        state = _read_json(state_path)
        state[name].update(status="done", exit_code=code, finished=_now(),
                           seconds=round(time.perf_counter() - started, 2))
        _write_json(state_path, state)
        _write_json(manifest_path, manifest)
        (logger.info if code == 0 else logger.warning)(
            f"stage {name}: done in {state[name]['seconds']} s, exit code {code}")
    return run_path


def build_parser() -> argparse.ArgumentParser:
    """Every default is None, so :data:`RUN_ARGS` can fill what the command line leaves out."""
    parser = argparse.ArgumentParser(description="CTG outcome classifier (SPEC §14).")
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=[*STAGES, "all"])
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY.PATH=VALUE")
    parser.add_argument("--run-dir", dest="run_dir")
    parser.add_argument("--folds", help="comma-separated fold ids, e.g. 1,2")
    parser.add_argument("--device")
    parser.add_argument("--devices", help="fold-parallel train and predict, one fold process per slot, e.g. "
                                          "cuda:0,cuda:1,cuda:2,cuda:3 (a repeated device takes two folds)")
    parser.add_argument("--only", help="evaluate: comma-separated analysis ids, e.g. metrics,R1")
    parser.add_argument("--skip", help="evaluate: comma-separated analysis ids to leave out")
    parser.add_argument("--allow-partial", dest="allow_partial", action="store_true", default=None,
                        help="evaluate: pool a run whose train stage had failed units")
    return parser


def compare(runs: Sequence[Any], out: Any, *, policy: Optional[str] = None) -> int:
    """``compare`` (§11.8, §11.12 block Q) of 2-6 finished runs, the first the reference, into ``out``: Q2
    ``comparison.parquet`` (:func:`metrics.compare_runs`), then the Q1/Q2 figures and Q3 ``comparison.md``
    (:func:`report.compare_report`). It reads the runs and writes only under ``out``, and it creates no MLflow run
    (§10.10.5). It refuses runs whose cohort digests differ (:func:`metrics.cohort_digest`). Returns the report's exit
    code (1 iff a figure or the markdown raised)."""
    table = metrics.compare_runs(runs, policy=policy)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "comparison.parquet", index=False)
    logger.info(f"compare: {len(table)} rows in {out / 'comparison.parquet'}")
    return report.compare_report(runs, out, policy=policy)


def compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m teb_vae.classifier.run compare",
                                     description="Model comparison across finished runs (SPEC §11.8, block Q).")
    parser.add_argument("--runs", nargs="+", required=True, metavar="RUN_DIR",
                        help="2-6 finished runs sharing a cohort; the first is the reference")
    parser.add_argument("--out", required=True, help="output directory, the only place compare writes")
    # ponytail: GUID level only; a segment-level compare (R5 populations) when an ablation needs it
    parser.add_argument("--level", choices=["guid"], default="guid")
    parser.add_argument("--policy", help="compare this threshold policy only (default: every GUID-level policy the "
                                         "runs share; the figures draw the reference's primary policy)")
    return parser


def _cli(argv: Optional[List[str]] = None) -> int:
    """Run :func:`main`; the process exit code is the largest recorded stage exit code. ``compare ...`` runs
    :func:`compare` instead."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["compare"]:
        args = compare_parser().parse_args(argv[1:])
        return compare(args.runs, args.out, policy=args.policy)
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    run_path = main(**{key: value for key, value in values.items() if value is not None})
    codes = [rec.get("exit_code") or 0 for rec in _read_json(run_path / "stage_state.json").values()]
    if values.get("stage") == "evaluate" and (values.get("only") or values.get("skip")):  # not in stage_state
        codes.append(_read_json(max((run_path / "evaluation").glob("partial_*/summary.json"))).get("exit_code") or 0)
    return max(codes, default=0)


#: Arguments for an argument-less launch (the IDE Run button): edit the values below and press Run. On a command-line
#: launch the command line wins per key, and a key left at None here is simply not set. ``compare`` is not a stage and
#: has no entry here: ``python -m teb_vae.classifier.run compare --runs RUN_A RUN_B [--policy np30] --out DIR``.
RUN_ARGS: Dict[str, Any] = {
    # YAML config, relative to the repo root or absolute. Shipped (teb_vae/classifier/configs/), each `base:
    # default.yaml` plus a delta:
    #   default.yaml        trf_cfs VAE latents, sequence scope, binary adverse_vs_healthy; the reference setup
    #   segment_scope.yaml  one score per segment; GUID scores from post-hoc aggregators (max, mean, lse, ...)
    #   three_class.yaml    healthy / acidosis / HIE with a softmax head and sqrt-inverse class weights
    #   st_ph.yaml          ST/PH coefficients straight from the shards instead of the VAE (the no-VAE comparison)
    #   cotrain.yaml        co-train the classifier with the VAE objective (the fragile end; read against frozen)
    #   smoke.yaml          CPU, 2 folds, tiny model on the generated fixture tree; for checking the pipeline only
    "config": "teb_vae/classifier/configs/default.yaml",
    # Which stage to run: "cohort" | "extract" | "train" | "predict" | "evaluate" | "report" | "verify" | "all".
    #   cohort    segment/GUID tables, labels, fold checks (fast; run it first on new data and read the counts)
    #   extract   one VAE pass over every unique segment into the feature cache (GPU: `device`)
    #   train     sklearn baselines, then every neural unit per fold (fold-parallel with `devices`)
    #   predict   val and test rows of every model; refused for a unit without its selection lock
    #   evaluate  every analysis, pooled over all folds (CPU)
    #   report    figures and summary.md from the evaluation tables
    #   verify    the pass/fail gate over the evaluation
    #   all       every stage in that order; a stage already done in `run_dir` is skipped, so "all" also resumes
    # Naming evaluate, report or verify re-runs it even when done; any stage that runs marks the later ones pending.
    "stage": "cohort",
    # None or a list of "classifier.<path>=<value>" strings; each value is parsed as YAML, so numbers, lists and
    # dicts work. Anything in the config can be set this way. The ones a real run needs:
    #   ["classifier.data.kfold_root=/data/.../k_fold_cross_validation_dataset",
    #    "classifier.source.vae.checkpoint=/runs/.../model_checkpoints/best.ckpt"]
    # Others often touched: "classifier.run.num_workers=2" (loader workers per fold process),
    # "classifier.run.seeds=[42,43,44]" (a seed ensemble), "classifier.labels.task=hie_vs_rest",
    # "advanced_config.tracking.mlflow.enabled=true". Settings that change results are part of the config digest.
    "overrides": None,
    # None starts a new run directory, <run.out_root>/<YYYY-mm-dd--HH-MM-SS>-<run.name>. An existing run directory
    # resumes it: finished stages and locked units are skipped. It must have been run with the same config digest;
    # otherwise it is refused (use a new directory for a new setting).
    "run_dir": None,
    # None = every fold in run.folds (1-10 in default.yaml). A comma-separated subset, e.g. "1" for a quick look or
    # "1,2,3". Part of the digest: a run directory holds one fold set.
    "folds": None,
    # None keeps run.device (cuda:0 in default.yaml). "cpu" or "cuda:<k>": the device of extract, and of train and
    # predict too while `devices` is None. Not part of the digest.
    "device": None,
    # None runs the folds one after another on `device`. A comma-separated list of device slots runs train and
    # predict with one process per fold, one fold per slot (SPEC §14.4), e.g.
    #   "cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"   8 folds at once, one per GPU
    #   "cuda:0,cuda:0"                                             2 folds sharing one GPU
    #   "cpu,cpu"                                                   2 folds at once on CPU
    # Each fold process starts run.num_workers loader workers (Linux only when > 0). Not part of the digest, so a run
    # can resume on other GPUs or serially.
    "devices": None,
    # evaluate only: None runs every analysis. A comma-separated subset of analysis ids runs only those, as a
    # partial look in evaluation/partial_<stamp>/ (the full evaluation, report and verify are left as they were).
    # Ids (SPEC §11.12): C cohort, metrics, R1 R2 R5 R8 R9 R10 R11 R12 ROC/PR, T1-T4 thresholds, B1 baselines,
    # M time-resolved, A alarms, S1 S2 S4 S5 S6 subgroups, K K3 K7 calibration, X X9 X10 3-class, H heterogeneity,
    # E errors, E6 attribution, VF regime vs frozen. E.g. "metrics,R1".
    "only": None,
    # evaluate only: None skips nothing; a comma-separated list of analysis ids to leave out (also a partial look),
    # e.g. "S1,S2,S4,S5,S6" to skip the subgroup analyses.
    "skip": None,
    # evaluate only: None or False refuses to pool a run in which some unit failed in train; True pools the rest
    # anyway (the failed units are listed as missing, and verify reports INCONCLUSIVE instead of PASS).
    "allow_partial": None,
}


if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    sys.exit(_cli())
