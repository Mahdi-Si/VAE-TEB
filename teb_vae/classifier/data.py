r"""Datasets, collates and loaders of the neural classifier (SPEC §7.1-7.2, §8.4-8.5, §10.4, §10.9.3).

:func:`build_unit` builds one :class:`UnitData` per fold (and per shuffled-label seed): the fold's retained
cohort rows joined to the frozen feature cache by ``baselines.fold_frame`` and projected at once onto
:data:`ALLOWED_COLUMNS` (L1); the train-only scaler over ``[values || attn]`` (§8.5, L4); the train-fold
GUID-level class counts and priors (L4). :class:`SegmentDataset` (segment scope) and :class:`GuidDataset`
(sequence scope) turn it into the batch dicts of the P3/P4 contract; :func:`make_loader` wraps either in a
seeded :class:`EpochBatches` batch sampler.

**Covariates (§7.3).** The cohort's raw ``cov:<name>`` / ``cov_age_h:<name>`` columns ride along with the frame
(observation times never get here: the as-of join returns value and age only);
:func:`fit_covariates` fits their encoding on train only (L4) and every dataset delivers it as ``batch["cov"]``
(``(B, D_cov)`` or ``(B, N, D_cov)``, padded with 0), with ModDrop covariate dropout on train loaders only
(``default_rng([seed, epoch, i, 1])``). :func:`covariates_off` and :func:`without_indicators` make the
``eval.covariates_off`` pass and the ``noind`` ablation unit.

**Online regimes (§10.1).** With an :class:`OnlineReader` (``make_loader(..., reader=)``) every batch also carries
``vae``: the collated raw HDF5 rows of its real segments, which the classifier's backbone turns into ``x``/``attn``
(``sources.OnlineFeatures``); the cached tensors ride along and keep the step mask.

**Forbidden inputs (L1).** ``epoch_s``, ``t_end_s`` and ``ss_rel_s`` are read, but reach tensors only as
within-GUID differences (``t_h``, ``delta_t``, ``elapsed``) and through ``time_in_ss``, which sees the window
length ``t_end_s - epoch_s`` and the positive part of ``ss_rel_s + window``; ``stage`` only as the ``in_ss`` flag
(straddle or second; first and unknown look alike, so ``has_ss`` does not leak from segment 0). ``hours_to_delivery``, ``cs``,
``bg``, ``source_file``, ``n_segments`` and ``first_epoch_s`` are never read.

**Memory.** The cache is memory-mapped once per process in its stored dtype (float16 by default) and shared by
every fold (:func:`load_store`): the OS pages in what a fold reads, so a cache larger than RAM still trains (a real
10-fold cache at the default keys is ~10-16 GB). Items are scaled to float32 on the fly. Loader workers are forked
(Python 3.14 defaults to forkserver, which would pickle the store into every worker).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler

from hdf5_dataset.hdf5_dataset import attribute_dict_collate
from hdf5_dataset.length_bucket_sampler import VariableBatchBucketSampler
from teb_vae.classifier.baselines import fold_frame
from teb_vae.classifier.cohort import COV, COV_AGE, context_features, fold_shards, warm_positions
from teb_vae.classifier.config import Classifier, CovariatesCfg
from teb_vae.classifier.sources import Scaler, fit_scaler, open_cache

#: The only cohort columns data.py keeps (L1), besides the declared covariates' raw ``cov:``/``cov_age_h:`` columns.
#: Identity and cache row; targets and loss weights; §7.1 context sources. The three clocks enter tensors only as
#: described in the module docstring.
ALLOWED_COLUMNS = ["fold", "split", "guid", "seg_pos", "row", "y", "class_code", "label_weight",
                   "tlo_end_s", "stage", "valid_frac", "ss_rel_s", "epoch_s", "t_end_s"]
#: Collate fill values of the per-segment keys (sequence scope).
_PAD = {"x": 0, "step_mask": False, "attn": 0, "ctx": 0, "cov": 0, "t_h": 0, "w": 0, "w_pos": 0, "row": -1}


def CONTEXT_COLUMNS(cfg: Any) -> List[str]:  # noqa: N802 (named as in the contract)
    """The §7.1 context vector of ``cfg`` (``Config`` or its ``classifier`` block), in batch ``ctx`` order."""
    c = getattr(cfg, "classifier", cfg)
    x = c.context
    wanted = {"tlo_psi": x.tlo.enabled, "tlo_missing": x.tlo.enabled and x.tlo.missing == "indicator",
              "in_ss": x.stage.enabled, "time_in_ss": x.time_in_ss.enabled,
              "valid_frac": x.valid_frac.enabled,
              "delta_t": x.delta_t.enabled and c.model.scope == "sequence", "elapsed": x.elapsed.enabled}
    return [name for name, on in wanted.items() if on]


def context_matrix(frame: pd.DataFrame, columns: Sequence[str], *, tlo_pre_onset: str = "clip") -> np.ndarray:
    """``(n, D_ctx)`` float32 in ``columns`` order. ``delta_t`` is 0 here (:class:`GuidDataset` fills it per
    item); ``elapsed`` is ``log1p`` hours since the GUID's first row of ``frame``; ``tlo_pre_onset``:
    ``context.tlo.pre_onset`` (:func:`~teb_vae.classifier.cohort.context_features`)."""
    window = frame.assign(epoch_s=0.0, t_end_s=frame["t_end_s"] - frame["epoch_s"])  # no absolute clock
    ctx = context_features(window, tlo_pre_onset=tlo_pre_onset)
    ctx["delta_t"] = 0.0
    ctx["elapsed"] = np.log1p((frame["t_end_s"] - frame.groupby("guid")["t_end_s"].transform("first")) / 3600.0)
    return ctx[list(columns)].to_numpy(np.float32)


def without_indicators(c: Classifier) -> Classifier:
    """``c`` with ``missing: no_indicator`` for TLO and the covariates: the ``noind`` unit of the §7.3.4 ablation."""
    x = c.context
    return c.model_copy(update={"context": x.model_copy(update={
        "tlo": x.tlo.model_copy(update={"missing": "no_indicator"}),
        "covariates": x.covariates.model_copy(update={"missing": "no_indicator"})})})


# ---- covariates (§7.3) -------------------------------------------------------------------------
@dataclass(frozen=True)
class Covariates:
    """The train-fold covariate encoding (§7.3, L4). Per variable: the standardised value (numeric) or a one-hot over
    the train vocabulary (categorical; unseen or missing = all 0), then ``<name>:missing`` (``missing: indicator``)
    and ``<name>:age_h`` = ``log1p`` age (a timed variable with ``age_feature``). Every missing entry is 0 after
    standardising. ``variables``: the fitted record per variable; ``sources``: the raw frame columns read;
    ``groups[j]``: the variable of output column j; ``missing_row``: the all-missing encoding."""

    variables: Tuple[Dict[str, Any], ...]
    sources: Tuple[str, ...]
    columns: Tuple[str, ...]
    groups: np.ndarray
    missing_row: np.ndarray

    def encode(self, frame: pd.DataFrame) -> np.ndarray:
        """``(n, D_cov)`` float32 of ``frame``'s raw covariate columns."""
        cols = []
        for v in self.variables:
            x = frame[COV + v["name"]]
            miss = x.isna().to_numpy()
            if v["kind"] == "numeric":
                cols.append(np.where(miss, 0.0, (x.to_numpy(float) - v["mean"]) / v["std"])[:, None])
            else:
                cols.append(x.to_numpy(object)[:, None] == np.asarray(v["vocab"], object)[None])
            if v["indicator"]:
                cols.append(miss[:, None])
            if v["age"]:
                cols.append(np.where(miss, 0.0, np.log1p(frame[COV_AGE + v["name"]].to_numpy(float)))[:, None])
        return np.concatenate(cols, 1).astype(np.float32)

    def drop(self, cov: np.ndarray, cfg: CovariatesCfg, rng: np.random.Generator) -> np.ndarray:
        """ModDrop (§7.3.5): each variable set missing with ``dropout_p``, the whole block with ``block_dropout_p``;
        ``cov`` is one item's ``(D_cov,)`` or ``(N, D_cov)``."""
        drop = rng.random(len(self.variables)) < cfg.dropout_p
        drop |= rng.random() < cfg.block_dropout_p
        return np.where(drop[self.groups], self.missing_row, cov).astype(np.float32)


def fit_covariates(train: pd.DataFrame, cfg: CovariatesCfg) -> Optional[Covariates]:
    """L4: the :class:`Covariates` encoding fitted on the train split (numeric mean/std over observed train segments,
    std floored to 1 when degenerate; categorical vocabulary sorted); None without variables."""
    if not cfg.variables:
        return None
    splits = sorted(set(train["split"]))
    if splits != ["train"]:
        raise ValueError(f"L4: covariate scalers and vocabularies are fitted on the train split only; got {splits}")
    variables, columns, groups = [], [], []
    for j, v in enumerate(cfg.variables):
        x = train[COV + v.name].dropna()
        rec = {"name": v.name, "kind": v.kind, "indicator": cfg.missing == "indicator",
               "age": cfg.age_feature and COV_AGE + v.name in train}
        if v.kind == "numeric":
            mean, std = (float(x.mean()), float(x.std(ddof=0))) if len(x) else (0.0, 1.0)
            rec |= {"mean": mean, "std": std if std > 1e-8 else 1.0}
            names = [v.name]
        else:
            rec["vocab"] = sorted(map(str, x.unique()))
            names = [f"{v.name}={k}" for k in rec["vocab"]]
        names += [f"{v.name}:missing"] * rec["indicator"] + [f"{v.name}:age_h"] * rec["age"]
        variables.append(rec)
        columns += names
        groups += [j] * len(names)
    sources = tuple(c for v in variables for c in (COV + v["name"], COV_AGE + v["name"]) if c in train)
    enc = Covariates(tuple(variables), sources, tuple(columns), np.asarray(groups, int),
                     np.zeros(len(columns), np.float32))
    return dataclasses.replace(enc, missing_row=enc.encode(pd.DataFrame({c: [np.nan] for c in sources}))[0])


# ---- the cache in memory -----------------------------------------------------------------------
@dataclass(frozen=True)
class Store:
    """The whole feature cache, stored dtype: ``values`` (N, T', C), ``step_mask`` (N, T'), ``attn``
    (N, T', C_a) or None; ``channels`` names the C value channels, then the C_a attention ones."""

    values: np.ndarray
    step_mask: np.ndarray
    attn: Optional[np.ndarray]
    channels: Tuple[str, ...]

    def stacked(self, rows: Any) -> np.ndarray:
        """``[values || attn]`` of ``rows``: what the scaler is fitted on and applied to."""
        v = self.values[rows]
        return v if self.attn is None else np.concatenate([v, self.attn[rows]], -1)


def _mapped(dset: Any, path: Path) -> np.ndarray:
    """A read-only ``np.memmap`` of a contiguous, uncompressed HDF5 dataset (what ``sources.extract`` writes); a
    chunked or compressed one is read whole."""
    offset = dset.id.get_offset()
    if dset.chunks is not None or dset.compression is not None or offset is None:
        return dset[()]
    return np.memmap(path, dtype=dset.dtype, mode="r", offset=offset, shape=dset.shape)


@functools.lru_cache(maxsize=1)
def load_store(cache_dir: str) -> Store:
    """``features.h5`` memory-mapped once per process; one cache per run, so another ``cache_dir`` evicts it."""
    path = Path(cache_dir) / "features.h5"
    with h5py.File(path, "r") as h5:
        return Store(values=_mapped(h5["values"], path), step_mask=_mapped(h5["step_mask"], path),
                     attn=_mapped(h5["attn"], path) if "attn" in h5 else None,
                     channels=tuple(json.loads(h5.attrs["channels"])))


class _Rows:
    """``store.stacked(rows[part])`` on demand: :func:`fit_scaler` reads 1024 rows at a time."""

    def __init__(self, store: Store, rows: np.ndarray) -> None:
        self.store, self.rows = store, rows

    def __getitem__(self, part: slice) -> np.ndarray:
        return self.store.stacked(self.rows[part])


# ---- the unit ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class UnitData:
    """One fold's data for a model kind. ``frames[split]``: retained rows, :data:`ALLOWED_COLUMNS` only,
    sorted by ``(guid, seg_pos)``; ``row`` addresses the cache. ``channels``/``attn_channels``: the value and
    attention channels the scaler kept. ``class_counts``: train GUIDs per task class (for
    ``losses.class_weights``); ``priors``: ``{"main": [K], "aux3": [3]}`` train GUID-level class priors;
    ``covariates``: the train-fold covariate encoding, None without covariates."""

    cfg: Classifier
    fold: int
    shuffle_seed: Optional[int]
    frames: Dict[str, pd.DataFrame]
    store: Store
    scaler: Scaler
    channels: Tuple[str, ...]
    attn_channels: Tuple[str, ...]
    context_columns: List[str]
    class_counts: List[int]
    priors: Dict[str, List[float]]
    covariates: Optional[Covariates] = None
    cache_dir: Optional[str] = None

    @property
    def n_values(self) -> int:
        return len(self.channels)

    @property
    def n_cov(self) -> int:
        return 0 if self.covariates is None else len(self.covariates.columns)

    @property
    def n_attn(self) -> int:
        return len(self.attn_channels)

    @property
    def n_ctx(self) -> int:
        return len(self.context_columns)

    @property
    def y_dtype(self) -> type:
        return np.float32 if self.cfg.labels.head == "binary" else np.int64

    def features(self, rows: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Scaled float32 ``x`` (n, T', C), ``step_mask`` (n, T') and ``attn`` (n, T', C_a) or None; 0 at
        masked steps."""
        keep = self.scaler.keep
        mask = self.store.step_mask[rows]
        z = ((self.store.stacked(rows)[..., keep].astype(np.float32) - self.scaler.center[keep].astype(np.float32))
             / self.scaler.scale[keep].astype(np.float32))
        z = np.where(mask[..., None], z, np.float32(0.0))
        return z[..., :self.n_values], mask, z[..., self.n_values:] if self.n_attn else None


def train_counts(train: pd.DataFrame, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """L4: train GUIDs per task class (``k``) and per ``class_code - 1`` (3)."""
    splits = sorted(set(train["split"]))
    if splits != ["train"]:
        raise ValueError(f"L4: class counts and priors are fitted on the train split only; got {splits}")
    g = train.groupby("guid")[["y", "class_code"]].first()
    return (np.bincount(g["y"].astype(int), minlength=k),
            np.bincount(g["class_code"].astype(int) - 1, minlength=3))


def shuffle_labels(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """§10.9.3: GUID labels (``y`` with its ``class_code``) permuted within the train split; every segment keeps
    the cohort's ``label_weight``. Rebuilt for the permuted targets, a time-dependent strategy would move each
    positive-labelled GUID's weight onto its late segments, so late-looking inputs would earn the positive label
    under any permutation and the control would beat 0.5 whenever the true signal is late (on the fixture:
    per-fold test AUROC 0.83-0.91). With ω fixed, the weighted label expectation of every segment is the train
    prevalence."""
    splits = sorted(set(train["split"]))
    if splits != ["train"]:
        raise ValueError(f"the shuffled-label control permutes the train split only; got {splits}")
    per_guid = train.groupby("guid")[["y", "class_code"]].first()
    permuted = per_guid.iloc[np.random.default_rng(seed).permutation(len(per_guid))].set_axis(per_guid.index)
    return train.assign(**{col: train["guid"].map(permuted[col]) for col in ("y", "class_code")})


def build_unit(cfg: Any, run_dir: Any, cache: Mapping[str, Any], fold: int, *,
               shuffle_seed: Optional[int] = None) -> UnitData:
    """The fold's :class:`UnitData`; ``shuffle_seed`` makes the §10.9.3 shuffled-label control.

    Args:
        cfg: ``Config`` or its ``classifier`` block.
        run_dir: The run directory (``cohort/`` is read).
        cache: The extract stage's manifest record (``manifest["source"]``).
        fold: Fold id.
        shuffle_seed: Permute train GUID labels with this seed; val and test are untouched.
    """
    c = getattr(cfg, "classifier", cfg)
    if not cache.get("cache_dir"):
        raise ValueError("data.py reads the frozen feature cache, which every regime builds; rerun --stage extract")
    index = open_cache(cache["cache_dir"], cache["fingerprint"])
    store = load_store(str(cache["cache_dir"]))
    frame = fold_frame(Path(run_dir), fold, index, store.step_mask.sum(1), c.cohort.min_segments_per_guid,
                       strategy=c.labels.strategy)
    names = [v.name for v in c.context.covariates.variables]
    lost = [COV + n for n in names if COV + n not in frame]
    if lost:
        raise ValueError(f"cohort/segments.parquet lacks the covariate column(s) {lost}; rerun --stage cohort")
    frame = frame[ALLOWED_COLUMNS + [col for n in names for col in (COV + n, COV_AGE + n) if col in frame]]  # L1
    frames = {split: g.reset_index(drop=True) for split, g in frame.groupby("split")}
    if shuffle_seed is not None:
        frames["train"] = shuffle_labels(frames["train"], shuffle_seed)
    train = frames["train"]
    rows = train["row"].to_numpy()
    scaler = fit_scaler(train[["fold", "split", "guid"]], _Rows(store, rows), store.step_mask[rows],
                        store.channels)
    names, n = np.asarray(store.channels), store.values.shape[-1]
    counts, aux = train_counts(train, 3 if c.labels.task == "three_class" else 2)
    unit = UnitData(
        cfg=c, fold=fold, shuffle_seed=shuffle_seed, frames=frames, store=store, scaler=scaler,
        channels=tuple(names[:n][scaler.keep[:n]]), attn_channels=tuple(names[n:][scaler.keep[n:]]),
        context_columns=CONTEXT_COLUMNS(c), class_counts=counts.tolist(),
        priors={"main": (counts / counts.sum()).tolist(), "aux3": (aux / aux.sum()).tolist()},
        covariates=fit_covariates(train, c.context.covariates), cache_dir=str(cache["cache_dir"]))
    logger.info(f"data fold {fold}: {', '.join(f'{s} {len(f)}' for s, f in frames.items())} segments; "
                f"{unit.n_values} value / {unit.n_attn} attention channels, ctx {unit.context_columns}, "
                f"{unit.n_cov} covariate columns; train "
                f"GUIDs per class {unit.class_counts}" + ("" if shuffle_seed is None
                                                          else f"; train labels shuffled (seed {shuffle_seed})"))
    return unit


def covariates_off(unit: UnitData) -> UnitData:
    """``unit`` with every covariate missing (the ``eval.covariates_off`` pass, §7.3.5): same fitted encoding, so
    ``batch["cov"]`` is its all-missing row."""
    if unit.covariates is None:
        return unit
    return dataclasses.replace(unit, frames={s: f.assign(**dict.fromkeys(unit.covariates.sources, np.nan))
                                             for s, f in unit.frames.items()})


# ---- online regimes (P7) -----------------------------------------------------------------------
class OnlineReader:
    """The raw HDF5 rows behind a unit's cached segments, for a source run online (§10.1).

    The cache index maps ``(fold, split, row)`` to the ``ds_index`` the cohort stage read, over the same shards
    (``cohort.fold_shards``); ``ds_index`` never enters a frame (it encodes shard order, L1). :meth:`bind` returns a
    frame's ``read(frame positions) -> [sample dicts]``, each checked against the frame's GUID. ``source`` is the
    :class:`~teb_vae.classifier.sources.VaeSource` whose loader contract opens the shards.
    """

    def __init__(self, source: Any, unit: UnitData) -> None:
        self.source, self.data = source, unit.cfg.data
        self.ds_index = open_cache(unit.cache_dir).set_index(["fold", "split", "row"])["ds_index"]
        self.datasets: Dict[Tuple[int, str], Any] = {}
        self.pid = os.getpid()

    def _dataset(self, fold: int, split: str) -> Any:
        """The ``(fold, split)`` dataset of the calling process. HDF5 handles are not fork-safe, so a forked loader
        worker first swaps every inherited dataset for a copy without handles (the dataset's own pickling protocol,
        ``__getstate__``: handles and locks dropped, reopened lazily in this process)."""
        if self.pid != os.getpid():
            self.datasets = {key: pickle.loads(pickle.dumps(ds)) for key, ds in self.datasets.items()}
            self.pid = os.getpid()
        if (fold, split) not in self.datasets:
            self.datasets[(fold, split)] = self.source.dataset(fold_shards(self.data, fold)[split])
        return self.datasets[(fold, split)]

    def bind(self, frame: pd.DataFrame) -> Any:
        keys = frame[["fold", "split"]].drop_duplicates()
        if len(keys) != 1:
            raise ValueError(f"an online frame holds one fold and split; got {keys.values.tolist()}")
        fold, split = int(keys.iloc[0, 0]), str(keys.iloc[0, 1])
        self._dataset(fold, split)  # open (and check) the shards now, in the loader's own process
        ds_index = self.ds_index.loc[pd.MultiIndex.from_frame(frame[["fold", "split", "row"]])].to_numpy()
        guids = frame["guid"].astype(str).to_numpy()

        def read(rows: np.ndarray) -> List[Any]:
            dataset = self._dataset(fold, split)  # per process: a forked worker reads through its own handles
            samples = [dataset[int(k)] for k in ds_index[rows]]
            if [str(s["guid"]) for s in samples] != guids[rows].tolist():
                raise ValueError(f"online rows of fold {fold} {split} do not read back as the cached GUIDs; "
                                 f"rebuild the cohort and cache")
            return samples

        return read


def _collate_vae(items: Sequence[Dict[str, Any]], out: Dict[str, Any]) -> Dict[str, Any]:
    """``out["vae"]``: every item's raw rows, collated in item then segment order (= ``seg_mask`` order)."""
    if "vae" in items[0]:
        out["vae"] = attribute_dict_collate([s for it in items for s in it["vae"]])
    return out


# ---- datasets ----------------------------------------------------------------------------------
def _key(key: Any) -> Tuple[int, Optional[int]]:
    """Dataset keys are ``i`` or ``(i, epoch)`` (:class:`EpochBatches`)."""
    return (int(key[0]), key[1]) if isinstance(key, tuple) else (int(key), None)


class _CovItems:
    """``cov`` of one item: the encoded rows, ModDrop'ed on train loaders (``default_rng([seed, epoch, i, 1])``, a
    stream apart from segment dropout's)."""

    def __init__(self, unit: UnitData, frame: pd.DataFrame, dropout: bool, seed: int) -> None:
        self.enc, self.cfg, self.dropout, self.seed = unit.covariates, unit.cfg.context.covariates, dropout, seed
        self.cov = None if self.enc is None else self.enc.encode(frame)

    def __call__(self, item: Dict[str, Any], rows: Any, i: int, epoch: Optional[int]) -> None:
        if self.cov is not None:
            cov = self.cov[rows]
            if self.dropout and epoch is not None:
                cov = self.enc.drop(cov, self.cfg, np.random.default_rng([self.seed, epoch, i, 1]))
            item["cov"] = cov


class SegmentDataset(Dataset):
    """Segment scope (§10.4): one item per retained segment of ``split``; ``w`` is ω over its GUID's Σω
    (0 when Σω = 0). ``guids[guid]`` names the batch's GUID index. ``cov_dropout``: covariate ModDrop (train).
    ``read`` (:meth:`OnlineReader.bind`): each item also carries its raw HDF5 row as ``vae`` (online regimes)."""

    def __init__(self, unit: UnitData, split: str, *, cov_dropout: bool = False, seed: int = 0,
                 read: Any = None) -> None:
        f = unit.frames[split]
        self.unit, self.rows, self.read = unit, f["row"].to_numpy(), read
        self.codes, self.guids = pd.factorize(f["guid"])
        self.ctx = context_matrix(f, unit.context_columns, tlo_pre_onset=unit.cfg.context.tlo.pre_onset)
        self.cov = _CovItems(unit, f, cov_dropout, seed)
        self.y = f["y"].to_numpy(unit.y_dtype)
        self.y3 = (f["class_code"].fillna(0) - 1).to_numpy(np.int64)
        omega, total = f["label_weight"].to_numpy(float), f.groupby("guid")["label_weight"].transform("sum").to_numpy()
        self.w = np.divide(omega, total, out=np.zeros_like(omega), where=total > 0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        i, epoch = _key(key)
        x, mask, attn = self.unit.features(self.rows[i:i + 1])
        item = {"x": x[0], "step_mask": mask[0], "ctx": self.ctx[i], "y": self.y[i], "y3": self.y3[i],
                "w": self.w[i], "guid": np.int64(self.codes[i]), "row": np.int64(i)}
        if attn is not None:
            item["attn"] = attn[0]
        self.cov(item, i, i, epoch)
        if self.read is not None:
            item["vae"] = self.read(np.array([i]))
        return item


class GuidDataset(Dataset):
    """Sequence scope (§10.4): one item per GUID, its retained segments by ``seg_pos``; ``w`` is raw ω (the
    segment-local term's weight), ``w_pos`` the per-position term's: ω with ``labels.k_warm`` (§6.5,
    :func:`~teb_vae.classifier.cohort.warm_positions`) zeroing positions ``seg_pos < k_warm``.

    With ``dropout > 0`` and a key ``(i, epoch)``, non-final segments are dropped with probability
    ``dropout`` (``default_rng([seed, epoch, i])``) before ``t_h`` (hours since the first kept segment's
    end), ``delta_t`` and ``elapsed`` are computed from the kept ends. ``cov_dropout``: covariate ModDrop, one draw
    per GUID (a dropped variable is missing at every position). ``read``: as :class:`SegmentDataset`'s, the kept
    segments' raw rows in order.
    """

    def __init__(self, unit: UnitData, split: str, *, dropout: float = 0.0, seed: int = 0,
                 cov_dropout: bool = False, read: Any = None) -> None:
        f = unit.frames[split]
        self.unit, self.dropout, self.seed, self.read = unit, dropout, seed, read
        self.cov = _CovItems(unit, f, cov_dropout, seed)
        self.rows, self.t_end = f["row"].to_numpy(), f["t_end_s"].to_numpy(float)
        codes, self.guids = pd.factorize(f["guid"])
        self.starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        self.lengths = np.diff(np.r_[self.starts, len(f)])
        self.ctx = context_matrix(f, unit.context_columns, tlo_pre_onset=unit.cfg.context.tlo.pre_onset)
        self.col = {k: unit.context_columns.index(k) for k in ("delta_t", "elapsed") if k in unit.context_columns}
        self.w = f["label_weight"].to_numpy(np.float32)
        self.w_pos = np.where(f["seg_pos"].to_numpy() < warm_positions(unit.cfg.labels, "sequence"), 0.0,
                              self.w).astype(np.float32)
        self.y = f["y"].to_numpy(unit.y_dtype)[self.starts]
        self.y3 = (f["class_code"].fillna(0) - 1).to_numpy(np.int64)[self.starts]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        i, epoch = _key(key)
        idx = np.arange(self.starts[i], self.starts[i] + self.lengths[i])
        if self.dropout and epoch is not None:
            keep = np.random.default_rng([self.seed, epoch, i]).random(len(idx)) >= self.dropout
            keep[-1] = True
            idx = idx[keep]
        t = self.t_end[idx]
        t_h = (t - t[0]) / 3600.0
        ctx = self.ctx[idx]
        if "delta_t" in self.col:
            ctx[:, self.col["delta_t"]] = np.log1p(np.diff(t, prepend=t[0]) / 3600.0)
        if "elapsed" in self.col:
            ctx[:, self.col["elapsed"]] = np.log1p(t_h)
        x, mask, attn = self.unit.features(self.rows[idx])
        item = {"x": x, "step_mask": mask, "ctx": ctx, "t_h": t_h.astype(np.float32), "w": self.w[idx],
                "w_pos": self.w_pos[idx],
                "row": idx.astype(np.int64), "y": self.y[i], "y3": self.y3[i], "guid": np.int64(i)}
        if attn is not None:
            item["attn"] = attn
        self.cov(item, idx, i, epoch)
        if self.read is not None:
            item["vae"] = self.read(idx)
        return item


def collate_segments(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Segment-scope items -> the contract batch (stacked; dtypes as the items'), plus ``vae`` when online."""
    return _collate_vae(items, {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0] if k != "vae"})


def collate_guids(items: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Sequence-scope items -> the contract batch, right-padded to the longest GUID (``seg_mask``)."""
    n, out = max(len(it["row"]) for it in items), {}
    for key, fill in _PAD.items():
        if key in items[0]:
            first = items[0][key]
            padded = np.full((len(items), n, *first.shape[1:]), fill, first.dtype)
            for b, it in enumerate(items):
                padded[b, :len(it[key])] = it[key]
            out[key] = torch.from_numpy(padded)
    out["seg_mask"] = out["row"] >= 0
    out |= {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in ("y", "y3", "guid")}
    return _collate_vae(items, out)


# ---- sampling and loaders ----------------------------------------------------------------------
def bucket_sizes(base: int) -> List[Tuple[Tuple[int, int], int]]:
    """``VariableBatchBucketSampler`` buckets: ``base`` GUIDs up to 16 segments, halved per doubling of N."""
    # ponytail: fixed halving (about constant padded tokens per batch); a config key if GPU memory needs it.
    return [((1, 16), base), ((17, 32), max(1, base // 2)), ((33, 64), max(1, base // 4)),
            ((65, 1 << 30), max(1, base // 8))]


class EpochBatches(Sampler):
    """Seeded batch sampler of ``(item, epoch)`` keys (§10.4).

    Epoch ``e`` draws from ``default_rng([seed, e])``: all items (permuted iff ``shuffle``), or with
    ``balanced`` as many GUID-level class-balanced draws with replacement (each class equally likely,
    then each GUID of it, then each item of the GUID). Batches are ``batch_size`` chunks, or
    ``VariableBatchBucketSampler`` batches over ``lengths`` when ``buckets`` is given. Every epoch has the
    natural (all items) batch count, ``len()``: balanced draws are repeated until it is filled, then cut to
    it (re-bucketed draws would change the count, which Lightning reads once). ``set_epoch`` fixes
    ``e``; otherwise each ``__iter__`` advances it. ``sampler`` is ``self`` so Lightning's
    ``_set_sampler_epoch`` (it looks at ``batch_sampler.sampler``) reaches :meth:`set_epoch`.
    """

    def __init__(self, y: Any, group: Any, *, batch_size: int, lengths: Any = None, buckets: Any = None,
                 shuffle: bool = False, balanced: bool = False, seed: int = 0) -> None:
        self.n, self.batch_size, self.lengths, self.buckets = len(y), batch_size, lengths, buckets
        self.shuffle, self.seed, self.epoch, self.sampler, self.p = shuffle, seed, 0, self, None
        if balanced:
            y, group = np.asarray(y).astype(int), np.asarray(group)
            guids_per_class = pd.Series(y).groupby(group).first().value_counts()
            p = 1.0 / (len(guids_per_class) * guids_per_class.loc[y].to_numpy() * np.bincount(group)[group])
            self.p = p / p.sum()
        self.n_batches = len(self._split(np.arange(self.n), np.random.default_rng(seed)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _split(self, items: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        if self.buckets is None:
            items = rng.permutation(items) if self.shuffle else items
            return [items[s:s + self.batch_size] for s in range(0, len(items), self.batch_size)]
        order = VariableBatchBucketSampler(self.lengths[items], self.buckets, shuffle=self.shuffle,
                                           seed=int(rng.integers(1 << 31)))
        return [items[b] for b in order]

    def _batches(self, epoch: int) -> List[List[Tuple[int, int]]]:
        rng = np.random.default_rng([self.seed, epoch])
        if self.p is None:
            batches = self._split(np.arange(self.n), rng)
        else:
            batches = []
            while len(batches) < self.n_batches:
                batches += self._split(rng.choice(self.n, self.n, p=self.p), rng)
        return [[(int(i), epoch) for i in b] for b in batches[:self.n_batches]]

    def __iter__(self):
        # A generator on purpose: torch's worker iterator calls iter() twice and consumes only the second, so
        # the epoch must be read and advanced on the first next(), not on iter().
        epoch, self.epoch = self.epoch, self.epoch + 1
        yield from self._batches(epoch)

    def __len__(self) -> int:
        return self.n_batches


def make_loader(unit: UnitData, split: str, *, train: bool, seed: int, reader: Optional[OnlineReader] = None
                ) -> DataLoader:
    """``split`` of ``unit`` as contract batches; with ``reader`` (online regimes) each batch also carries ``vae``,
    the raw HDF5 rows of its real segments (:class:`OnlineReader`).

    ``train``: shuffled, ``train.sampler`` honoured, segment dropout (sequence scope) and covariate dropout, all
    seeded by ``(seed, epoch)``. Otherwise deterministic and unshuffled: frame order (segment scope) or length buckets
    in frame order (sequence scope). Batch size ``train.batch_segments`` or, bucketed, ``train.batch_guids``.
    Workers persist across epochs for ``train`` loaders only: persistent eval workers each stall the unit's
    teardown by about 5 s.
    """
    c = unit.cfg
    read = None if reader is None else reader.bind(unit.frames[split])
    balanced = train and c.train.sampler == "class_balanced"
    if balanced and c.train.loss.weighting != "none":
        logger.warning(f"train.sampler=class_balanced with train.loss.weighting={c.train.loss.weighting}: "
                       f"the class imbalance is corrected twice (§10.4)")
    if c.model.scope == "sequence":
        dataset = GuidDataset(unit, split, dropout=c.train.segment_dropout if train else 0.0, seed=seed,
                              cov_dropout=train, read=read)
        sampler = EpochBatches(dataset.y, np.arange(len(dataset)), batch_size=c.train.batch_guids,
                               lengths=dataset.lengths, buckets=bucket_sizes(c.train.batch_guids),
                               shuffle=train, balanced=balanced, seed=seed)
        collate = collate_guids
    else:
        dataset = SegmentDataset(unit, split, cov_dropout=train, seed=seed, read=read)
        sampler = EpochBatches(dataset.y, dataset.codes, batch_size=c.train.batch_segments, shuffle=train,
                               balanced=balanced, seed=seed)
        collate = collate_segments
    workers = c.run.num_workers
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=workers,
                      multiprocessing_context="fork" if workers else None, persistent_workers=train and workers > 0)
