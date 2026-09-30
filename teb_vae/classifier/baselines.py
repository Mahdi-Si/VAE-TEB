"""Mandatory sklearn baselines: the linear probe and the shortcut baseline (SPEC §10.9, FR-12).

Per fold, :func:`train` fits both on train, chooses the probe's ``C`` on val, selects every
threshold policy on val and saves everything under ``baselines/fold_<k>/``; ``selection_lock.json``
(config digest, SHA-256 of every saved file, timestamp) is written last and never overwritten, as
``latent_pilot/config.py:1409 lock_selection``. :func:`score_fold` re-scores from the saved files
only, and refuses test rows without an intact lock (L5).

* **Probe** (§10.9.1). Per cached segment: masked sum, count and max of the value channels
  (``role: attention`` cues live in the cache's ``attn`` and are never read). The features at
  position n are [masked mean || masked max] over the last ``baselines.probe_last_n`` retained
  segments <= n, scaled by the fold's train-only scaler (§8.5, L4; affine per channel, so the scaled
  mean/max are the mean/max of scaled steps). The GUID's features at its last position train the
  L2 ``LogisticRegression``; ``C`` in :data:`PROBE_C` minimises val GUID log loss.
* **Shortcut** (§10.9.2). ``LogisticRegression`` on :data:`SHORTCUT_FEATURES`, standardised on
  train. These metadata are never model inputs; the model is GUID-level only (NaN segment scores).

Scores are logits (``decision_function``); no calibration, so ``*_cal == raw`` (CONTRACT). On
``labels.task: three_class`` both are multinomial: the alarm logit is ``logit P(adverse)`` from
``predict_log_proba``, the probabilities fill :data:`THREE_CLASS_COLUMNS` (the probe's rows: its online
output; the shortcut's rows: NaN) and every policy also gets per-class OvR thresholds (``"ovr"``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import skops.io as sio
from loguru import logger
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from teb_vae.classifier.config import Config, digest, ovr_enabled
from teb_vae.classifier.sources import Scaler, fit_scaler, open_cache, read_rows
from teb_vae.classifier.thresholds import guid_level_policies, ovr_thresholds, select_thresholds
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import RunStateError

MODELS = ("probe", "shortcut")
SEED = "na"
PROBE_C = (0.01, 0.1, 1.0)
SHORTCUT_FEATURES = ["n_segments", "span_h", "has_tlo", "has_ss", "mean_valid_frac"]
LOCK = "selection_lock.json"
#: Files the lock digests: everything val selected, frozen before test is scored.
LOCKED = ("scaler.json", "probe.skops", "shortcut.skops", "thresholds_probe.json",
          "thresholds_shortcut.json")
SEGMENT_COLUMNS = [
    "run_id", "model_id", "seed", "fold", "split", "guid", "patient", "seg_pos", "slot", "epoch_s", "t_end_s",
    "hours_to_delivery", "tlo_end_h", "ss_rel_h", "stage", "class_code", "y", "in_eval_window",
    "label_weight", "cs", "bg", "clinical_class", "subgroup", "has_tlo", "has_ss", "shared_test",
    "logit_seg", "logit_seg_cal", "logit_online", "logit_online_cal", "valid_frac", "n_valid_steps",
]
GUID_COLUMNS = [
    "run_id", "model_id", "seed", "fold", "split", "guid", "patient", "class_code", "y", "cs", "bg",
    "clinical_class", "subgroup", "has_tlo", "has_ss", "shared_test", "n_segments", "first_epoch_s",
    "last_t_end_s", "score_final", "score_final_cal",
]
#: 3-class prediction columns (§11.1) of every model on ``labels.task: three_class``, segment and GUID rows: the class
#: probabilities (0 healthy, 1 acidosis, 2 hie; target ``class_code - 1``), calibrated twins, and the CORAL g of an
#: ordinal head (NaN otherwise). A binary task's aux head adds the raw ``p_c0..2`` only.
THREE_CLASS_COLUMNS = ["p_c0", "p_c1", "p_c2", "p_c0_cal", "p_c1_cal", "p_c2_cal", "ord_score"]


@dataclass(frozen=True)
class Features:
    """Per cache row: masked ``sum`` (N, C), valid-step ``count`` (N,), masked ``max`` (N, C), the
    step ``mask`` (N, T'); raw (unscaled) units."""

    cache_dir: str
    index: pd.DataFrame
    sum: np.ndarray
    count: np.ndarray
    max: np.ndarray
    mask: np.ndarray
    channels: Tuple[str, ...]


class _CacheValues:
    """``values[a:b]`` of chosen cache rows, read on demand (:func:`fit_scaler` reads 1024 at a time)."""

    def __init__(self, cache_dir: str, rows: Any) -> None:
        self.cache_dir, self.rows = cache_dir, np.asarray(rows)

    def __getitem__(self, part: slice) -> np.ndarray:
        return read_rows(self.cache_dir, self.rows[part]).values.numpy()


def load_features(cache: Mapping[str, Any], chunk: int = 1024) -> Features:
    """Per-row summaries of the whole frozen cache (the extract stage's manifest record)."""
    if not cache.get("cache_dir"):
        raise ValueError("the baselines read the frozen feature cache; train.regime must be frozen_cached")
    index = open_cache(cache["cache_dir"], cache["fingerprint"])
    n, parts = int(index["row"].max()) + 1, []
    for start in range(0, n, chunk):
        f = read_rows(cache["cache_dir"], np.arange(start, min(start + chunk, n)))
        v, m = f.values.numpy(), f.step_mask.numpy()
        parts.append((np.einsum("ntc,nt->nc", v, m.astype(v.dtype)).astype(np.float64), m.sum(1),
                      np.where(m[..., None], v, -np.inf).max(1).astype(np.float64), m))
    s, count, mx, mask = (np.concatenate(p) for p in zip(*parts))
    return Features(cache["cache_dir"], index, s, count, mx, mask, f.channels[:v.shape[-1]])


def fold_frame(run_dir: Path, fold: int, index: pd.DataFrame, n_valid_steps: Any,
               min_segments: int = 1, *, strategy: Optional[str] = None) -> pd.DataFrame:
    """Retained cohort segments of ``fold`` with their cache ``row`` and GUID strata, sorted by
    ``(split, guid, seg_pos)``: the one population of the baselines and the neural models.

    A segment whose cached ``step_mask`` is all False (``n_valid_steps``, per cache row, is 0: the source's
    warm-up / ``min_step`` masking can empty a window that ``low_valid_frac`` kept) is excluded as
    ``no_valid_steps`` with a WARNING; ``seg_pos`` and the GUID's ``n_segments``/``first_epoch_s``/``last_t_end_s``
    follow the kept segments. ``min_segments`` (``cohort.min_segments_per_guid``) is re-applied after the drop.
    Per-split counts of those segments, of the GUIDs left with none and of the GUIDs left with fewer than
    ``min_segments`` are in ``frame.attrs["no_valid_steps"]``. ``strategy`` (``labels.strategy``): the only
    position-dependent ω, ``final_only``'s last segment, is re-marked on the kept rows (``k_warm`` is applied later,
    from the kept ``seg_pos``, by ``data.GuidDataset``)."""
    co, keys = Path(run_dir) / "cohort", ["fold", "split", "guid", "epoch_s"]
    seg, gd = pd.read_parquet(co / "segments.parquet"), pd.read_parquet(co / "guids.parquet")
    seg = seg[(seg["fold"] == fold) & ~seg["excluded"]]
    gd = gd[(gd["fold"] == fold) & ~gd["excluded"]]
    frame = (seg.merge(index[keys + ["row"]], on=keys, how="left", validate="one_to_one")
             .merge(gd[["split", "guid", "has_tlo", "has_ss", "shared_test", "n_segments", "first_epoch_s",
                        "last_t_end_s", "patient"]], on=["split", "guid"], validate="many_to_one")
             .sort_values(["split", "guid", "seg_pos"], ignore_index=True))
    if frame["row"].isna().any():
        raise ValueError(f"fold {fold}: {int(frame['row'].isna().sum())} cohort segment(s) missing from the "
                         f"feature cache; rerun --stage extract")
    if not (frame.groupby(["split", "guid"]).cumcount() == frame["seg_pos"]).all():
        raise ValueError(f"fold {fold}: seg_pos is not 0..n-1 per GUID; rebuild the cohort stage")
    frame = frame.astype({"row": int})
    by = [frame["split"], frame["guid"]]
    empty = pd.Series(np.asarray(n_valid_steps)[frame["row"].to_numpy()] == 0)
    n_kept = (~empty).groupby(by).transform("sum")
    short = (n_kept > 0) & (n_kept < min_segments)
    per_split = lambda hit: hit.groupby(by).all().groupby(level=0).sum().astype(int).to_dict()  # noqa: E731
    record = {"segments": empty.groupby(frame["split"]).sum().astype(int).to_dict(),
              "guids": per_split(n_kept == 0), "min_segments": per_split(short)}
    if (empty | short).any():
        logger.warning(f"fold {fold}: no_valid_steps: excluded {record['segments']} segment(s) with an all-masked "
                       f"cached step_mask; GUIDs left with none: {record['guids']}; with fewer than {min_segments}: "
                       f"{sorted(frame.loc[short, 'guid'].unique())}")
        frame = frame[~(empty | short)].reset_index(drop=True)
        g = frame.groupby(["split", "guid"])
        frame = frame.assign(seg_pos=g.cumcount(), n_segments=g["seg_pos"].transform("size"),
                             first_epoch_s=g["epoch_s"].transform("min"), last_t_end_s=g["t_end_s"].transform("max"))
    if strategy == "final_only":  # ω marks the kept last segment: the cohort's may have been dropped just above
        last = frame["seg_pos"] == frame.groupby(["split", "guid"])["seg_pos"].transform("max")
        frame["label_weight"] = last.astype(float)
    frame.attrs["no_valid_steps"] = record
    return frame


def label_rows(run_dir: Path, fold: int, index: pd.DataFrame, splits: Sequence[str],
               n_valid_steps: np.ndarray, min_segments: int = 1, *, strategy: Optional[str] = None) -> pd.DataFrame:
    """:func:`fold_frame` rows of ``splits`` (``strategy``: :func:`fold_frame`'s) with the CONTRACT keys, eval
    clocks, labels, strata and quality columns every model's prediction rows share (no scores, no ``model_id``/``seed``). ``n_valid_steps`` is
    indexed by cache row. ``y`` is the binary target every binary analysis reads: the task's, with a 3-class
    target collapsed to adverse (class >= 1, §11.3); the 3-class target is ``class_code - 1``."""
    frame = fold_frame(run_dir, fold, index, n_valid_steps, min_segments, strategy=strategy)
    frame = frame[frame["split"].isin(splits)].reset_index(drop=True)
    return frame.assign(run_id=Path(run_dir).name, subgroup=frame["source_file"],
                        tlo_end_h=frame["tlo_end_s"] / 3600.0, ss_rel_h=frame["ss_rel_s"] / 3600.0,
                        n_valid_steps=np.asarray(n_valid_steps)[frame["row"].to_numpy()].astype(int),
                        class_code=frame["class_code"].astype(int), y=frame["y"].astype(int).clip(upper=1))


def probe_features(frame: pd.DataFrame, feats: Features, scaler: Scaler, last_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(online, segment)`` features per row of ``frame``: [mean || max] of the scaled value channels
    over the last ``last_n`` segments <= n, and over segment n alone. No valid step reads as 0."""
    rows, pos = frame["row"].to_numpy(), frame["seg_pos"].to_numpy()
    s0, n0, m0 = feats.sum[rows], feats.count[rows], feats.max[rows]
    s, n, m = s0.copy(), n0.copy(), m0.copy()
    for k in range(1, last_n):  # rows are sorted by (split, guid, seg_pos): row i-k is the same GUID iff pos >= k
        i = np.flatnonzero(pos >= k)
        s[i] += s0[i - k]
        n[i] += n0[i - k]
        m[i] = np.maximum(m[i], m0[i - k])

    def scaled(s: np.ndarray, n: np.ndarray, m: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.hstack([scaler.apply(s / n[:, None]), scaler.apply(m)])
        return np.nan_to_num(x, nan=0.0, neginf=0.0)

    return scaled(s, n, m), scaled(s0, n0, m0)


def shortcut_features(frame: pd.DataFrame) -> pd.DataFrame:
    """:data:`SHORTCUT_FEATURES` per ``(split, guid)``, from the cohort tables only."""
    g = frame.groupby(["split", "guid"])
    return pd.DataFrame({
        "n_segments": g.size(), "span_h": (g["t_end_s"].max() - g["epoch_s"].min()) / 3600.0,
        "has_tlo": g["has_tlo"].first().astype(float), "has_ss": g["has_ss"].first().astype(float),
        "mean_valid_frac": g["valid_frac"].mean(),
    })[SHORTCUT_FEATURES]


def _is_last(frame: pd.DataFrame) -> np.ndarray:
    return (frame["seg_pos"] == frame.groupby(["split", "guid"])["seg_pos"].transform("max")).to_numpy()


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def fold_dir(run_dir: Path, fold: int) -> Path:
    return Path(run_dir) / "baselines" / f"fold_{fold}"


def lock_problem(out: Path, config_digest: str) -> Optional[str]:
    """Why ``out``'s lock does not hold (L5), or None: no lock, another config, or a locked file changed after
    locking. Silent; :func:`read_lock` is the refusing form."""
    target = out / LOCK
    if not target.is_file():
        return f"L5: {target} is missing; test rows are written only after the selection is locked (run --stage train)"
    lock = json.loads(target.read_text())
    changed = [name for name, sha in lock["sha256"].items() if not (out / name).is_file() or _sha256(out / name) != sha]
    if lock["config_digest"] != config_digest or changed:
        return (f"L5: {target} was locked under config {lock['config_digest'][:12]} (now {config_digest[:12]}); files "
                f"changed or missing since: {changed}")
    return None


def read_lock(out: Path, config_digest: str) -> Dict[str, Any]:
    """The fold's lock, checked against the config digest and every locked file's SHA-256 (L5).

    Raises:
        RunStateError: No lock, another config, or a locked file changed after locking.
    """
    why = lock_problem(out, config_digest)
    if why:
        logger.error(why)  # §10.10.6: every refusal names its guard before raising
        raise RunStateError(why)
    return json.loads((out / LOCK).read_text())


def write_lock(out: Path, config_digest: str, files: Sequence[str], **extra: Any) -> Dict[str, Any]:
    """Write ``out/selection_lock.json`` (last, never before the files it digests): config digest, SHA-256 of
    every ``files`` path (relative to ``out``), UTC timestamp. :func:`read_lock` checks it."""
    lock = {"locked_utc": utc_now(), "config_digest": config_digest, **extra,
            "sha256": {name: _sha256(out / name) for name in files}}
    (out / LOCK).write_text(json.dumps(lock, indent=2))
    return lock


def _scores(model: Any, X: np.ndarray) -> Tuple[np.ndarray, Any]:
    """``(alarm logit, class probabilities | None)``: the decision function of a binary model; for K > 2 classes
    ``logit P(Y >= 1) = log Σ_{k>=1} p_k - log p_0`` from ``predict_log_proba``, and the probabilities (n, K)."""
    if len(model.classes_) == 2:
        return model.decision_function(X), None
    lp = model.predict_log_proba(X)
    return logsumexp(lp[:, 1:], axis=1) - lp[:, 0], np.exp(lp)


def _val_scores(y: np.ndarray, model: Any, X: np.ndarray) -> Dict[str, float]:
    """Val AUROC of the alarm logit against ``y > 0`` and log loss (binary, or multinomial over the classes)."""
    score, P = _scores(model, X)
    return {"val_auroc": float(roc_auc_score(y > 0, score)),
            "val_logloss": float(log_loss(y, 1.0 / (1.0 + np.exp(-score)), labels=[0, 1]) if P is None
                                 else log_loss(y, P, labels=model.classes_))}


def _three_class(P: Any, n: int) -> Dict[str, Any]:
    """:data:`THREE_CLASS_COLUMNS` of uncalibrated baseline probabilities (n, 3), all NaN when ``P`` is None; clipped to
    [1e-12, 1 - 1e-12] like the neural ``p_c<k>_cal`` (``train.P_EPS``), so every OvR ``logit(p_c<k>_cal)`` is finite."""
    P = np.full((n, 3), np.nan) if P is None else np.clip(P, 1e-12, 1.0 - 1e-12)
    return {**{f"p_c{k}{cal}": P[:, k] for cal in ("", "_cal") for k in range(3)}, "ord_score": np.nan}


def _fit(cfg: Config, run_dir: Path, fold: int, feats: Features, out: Path) -> None:
    """Scaler, probe (C on val) and shortcut of one fold, saved with their ``*_fit.json``."""
    c = cfg.classifier
    frame = fold_frame(run_dir, fold, feats.index, feats.count, c.cohort.min_segments_per_guid,
                       strategy=c.labels.strategy)
    excluded = frame.attrs["no_valid_steps"]  # counts only: every split, for the report
    frame = frame[frame["split"] != "test"].reset_index(drop=True)  # test stays unread until the lock
    tr = frame[frame["split"] == "train"]
    scaler = fit_scaler(tr[["fold", "split", "guid"]], _CacheValues(feats.cache_dir, tr["row"]),
                        feats.mask[tr["row"].to_numpy()], feats.channels)
    scaler.save(out / "scaler.json")
    online, _ = probe_features(frame, feats, scaler, c.baselines.probe_last_n)
    last = _is_last(frame)
    g = frame[last].reset_index(drop=True)
    X, y = online[last], g["y"].to_numpy(int)
    S = shortcut_features(frame).loc[pd.MultiIndex.from_frame(g[["split", "guid"]])].to_numpy(float)
    tr, va = (g["split"] == "train").to_numpy(), (g["split"] == "val").to_numpy()
    counts = {s: {str(k): int(v) for k, v in zip(*np.unique(y[(g["split"] == s).to_numpy()], return_counts=True))}
              for s in ("train", "val")}

    fits = {C: LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr]) for C in PROBE_C}
    loss = {C: _val_scores(y[va], m, X[va])["val_logloss"] for C, m in fits.items()}
    best = min(loss, key=loss.get)  # ties: the smallest C (strongest penalty)
    kept = [name for name, k in zip(scaler.channels, scaler.keep) if k]
    shortcut = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(S[tr], y[tr])
    coef = shortcut[-1].coef_
    records = {
        "probe": (fits[best], {"C": best, "val_logloss_by_C": {str(C): v for C, v in loss.items()},
                               "last_n": c.baselines.probe_last_n,
                               "features": [f"mean:{n}" for n in kept] + [f"max:{n}" for n in kept],
                               **_val_scores(y[va], fits[best], X[va])}),
        "shortcut": (shortcut, {"features": SHORTCUT_FEATURES,  # per class under a multinomial fit
                                "coef": dict(zip(SHORTCUT_FEATURES, (coef[0] if len(coef) == 1 else coef.T).tolist())),
                                **_val_scores(y[va], shortcut, S[va])}),
    }
    for name, (model, record) in records.items():
        sio.dump(model, out / f"{name}.skops")
        (out / f"{name}_fit.json").write_text(json.dumps({"fold": fold, "n_by_class": counts,
                                                          "no_valid_steps": excluded, **record}, indent=2))
    logger.info(f"baselines fold {fold}: probe C={best} val AUROC {records['probe'][1]['val_auroc']:.3f}, "
                f"shortcut val AUROC {records['shortcut'][1]['val_auroc']:.3f}")
    if records["shortcut"][1]["val_auroc"] > c.baselines.shortcut_warn_auroc:
        logger.warning(f"fold {fold}: shortcut (metadata-only) val AUROC {records['shortcut'][1]['val_auroc']:.3f} "
                       f"> {c.baselines.shortcut_warn_auroc}: length/missingness predicts the label")


def score_fold(cfg: Config, run_dir: Path, fold: int, feats: Features,
               splits: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """CONTRACT segment and GUID rows of both baselines on ``splits`` of ``fold``, from saved files.

    Returns:
        ``(segments, guids, lock)``; ``lock`` is ``{}`` when no test row is requested.

    Raises:
        RunStateError: ``"test"`` in ``splits`` without an intact lock (L5).
    """
    out = fold_dir(run_dir, fold)
    lock = read_lock(out, digest(cfg)) if "test" in splits else {}
    frame = label_rows(run_dir, fold, feats.index, splits, feats.count, cfg.classifier.cohort.min_segments_per_guid,
                       strategy=cfg.classifier.labels.strategy)
    scaler = Scaler.load(out / "scaler.json")
    probe, shortcut = (sio.load(out / f"{name}.skops") for name in MODELS)
    online, single = probe_features(frame, feats, scaler, cfg.classifier.baselines.probe_last_n)
    last = _is_last(frame)
    seg = frame.assign(seed=SEED)
    g = seg[last].reset_index(drop=True)
    S = shortcut_features(frame).loc[pd.MultiIndex.from_frame(g[["split", "guid"]])].to_numpy(float)
    (s_single, _), (s_on, p_on), (s_last, p_last) = (_scores(probe, x) for x in (single, online, online[last]))
    s_short, p_short = _scores(shortcut, S)
    scores = {"probe": (s_single, s_on, s_last, p_on, p_last), "shortcut": (np.nan, np.nan, s_short, None, p_short)}
    segs, guids = [], []
    for name, (s_seg, s_online, s_final, p_seg, p_gd) in scores.items():
        three = {} if p_gd is None else _three_class(p_seg, len(seg))
        segs.append(seg.assign(model_id=name, logit_seg=s_seg, logit_seg_cal=s_seg, logit_online=s_online,
                               logit_online_cal=s_online, **three)[SEGMENT_COLUMNS + list(three)])
        three = {} if p_gd is None else _three_class(p_gd, len(g))
        guids.append(g.assign(model_id=name, score_final=s_final, score_final_cal=s_final, **three)[
            GUID_COLUMNS + list(three)])
    return pd.concat(segs, ignore_index=True), pd.concat(guids, ignore_index=True), lock


def _thresholds(cfg: Config, seg: pd.DataFrame, gd: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Every policy on val per model; the GUID-level shortcut gets ``guid_level_policies`` (CONTRACT). On
    ``three_class`` (``eval.ovr_thresholds``) the same policies per class one-vs-rest go to ``"ovr"``."""
    ev = cfg.classifier.eval
    policies = [p.model_dump() for p in ev.thresholds]
    kw = dict(bin_h=ev.bin_h, exclude_last_min=ev.exclude_last_min, staleness_h=ev.snapshot_max_staleness_h)
    out = {}
    for name in MODELS:
        s, g = seg[seg["model_id"] == name], gd[gd["model_id"] == name]
        kept, skipped = (policies, []) if name == "probe" else guid_level_policies(policies)
        out[name] = select_thresholds(s, g, kept, **kw)
        for pid in skipped:
            out[name].setdefault("guid", {})[pid] = {"skipped": "guid-level model"}
        if ovr_enabled(cfg.classifier):
            out[name]["ovr"] = ovr_thresholds(s, g, kept, **kw)
    return out


def train(cfg: Config, run_dir: Path, cache: Mapping[str, Any]) -> Dict[str, Any]:
    """Fit, select on val, and lock every fold of ``run.folds``; a locked fold is skipped (resume).

    Returns:
        ``{fold: {model: *_fit.json minus the feature names}}`` for the manifest.
    """
    feats, config_digest = load_features(cache), digest(cfg)
    for fold in cfg.classifier.run.folds:
        out = fold_dir(run_dir, fold)
        if (out / LOCK).is_file():
            read_lock(out, config_digest)
            logger.info(f"baselines fold {fold}: already locked, skipped")
            continue
        out.mkdir(parents=True, exist_ok=True)
        _fit(cfg, run_dir, fold, feats, out)
        seg, gd, _ = score_fold(cfg, run_dir, fold, feats, ["val"])
        for name, thr in _thresholds(cfg, seg, gd).items():
            (out / f"thresholds_{name}.json").write_text(json.dumps(thr, indent=2))  # -Infinity is legal
        write_lock(out, config_digest, LOCKED, fold=fold,
                   note="every val-side choice (scaler, C, models, thresholds), fixed before test is scored")
        logger.info(f"baselines fold {fold}: selection locked at {out / LOCK}")
    fits = {str(k): {name: json.loads((fold_dir(run_dir, k) / f"{name}_fit.json").read_text()) for name in MODELS}
            for k in cfg.classifier.run.folds}
    return {k: {name: {key: v for key, v in rec.items() if key != "features"} for name, rec in per.items()}
            for k, per in fits.items()}


def predict(cfg: Config, run_dir: Path, cache: Mapping[str, Any], splits: Sequence[str] = ("val", "test")
            ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """Both baselines on ``splits`` of every fold: ``(segments, guids, thresholds, units)``.

    ``thresholds`` is keyed ``"<model_id>|<seed>|<fold>"``; each unit carries its ``lock_written_at``.
    """
    feats, segs, guids, thr, units = load_features(cache), [], [], {}, []
    for fold in cfg.classifier.run.folds:
        seg, gd, lock = score_fold(cfg, run_dir, fold, feats, splits)
        segs.append(seg)
        guids.append(gd)
        for name in MODELS:
            thr[f"{name}|{SEED}|{fold}"] = json.loads((fold_dir(run_dir, fold) / f"thresholds_{name}.json").read_text())
            units.append({"model_id": name, "seed": SEED, "fold": fold, "lock_written_at": lock.get("locked_utc")})
    return pd.concat(segs, ignore_index=True), pd.concat(guids, ignore_index=True), thr, units
