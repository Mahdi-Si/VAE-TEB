
import copy
import contextlib
import json
import math
import os
import random
import sys
import time
import importlib.util

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from tqdm import tqdm

import yaml

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    umap = None

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    GaussianHMM = None

try:
    from scipy.linalg import sqrtm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sqrtm = None

try:
    from tslearn.barycenters import dtw_barycenter_averaging  # type: ignore
    from tslearn.clustering import TimeSeriesKMeans  # type: ignore
    from tslearn.metrics import dtw  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dtw_barycenter_averaging = None
    TimeSeriesKMeans = None
    dtw = None

try:
    import ruptures as rpt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    rpt = None

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

plt.switch_backend("Agg")

LATENT_STEP_SECONDS = 4.0
EPS = 1e-9
DEFAULT_CLASS_NAMES = ["class_0", "class_1", "class_2"]



def _load_seqvae_graph_model():
    module_name = "seqvae_teb_graph_model_train"
    module_path = Path(__file__).resolve().parent / "graph_model_train.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SeqVAE graph model from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_graph_model_module = _load_seqvae_graph_model()
SeqVAEGraphModel = getattr(_graph_model_module, "SeqVAEGraphModel")


DEFAULT_ANALYSIS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "device": None,
    "which": "mu_post",
    "keep_te": True,
    "keep_uncertainty": True,
    "latent_dim": 16,
    "keys": {
        "y_st": "fhr_st",
        "y_ph": "fhr_ph",
        "x_ph": "fhr_up_ph",
        "signal_id": "guid",
        "epoch_idx": "epoch",
        "label": "target",
        "t0": None,
        "source_file": "source_file",
    },
    "class_names": [],
    "file_labels": {},
    "sampling": {
        "target_guids": None,
        "exclude_guids": None,
        "max_batches": None,
        "max_sequences": None,
        "max_epochs_per_signal": None,
        "stride": 1,
    },
    "storage": {
        "save_parquet": True,
        "save_csv": False,
        "compression": "snappy",
    },
    "performance": {
        "forward_chunk_size": None,
        "use_amp": True,
        "amp_dtype": "float16",
    },
    "reducers": {
        "pca": {
            "enabled": True,
            "n_components": 3,
            "standardize": False,
            "incremental": False,
            "batch_size": 8192,
        },
        "umap": {
            "enabled": False,
            "n_components": 2,
            "n_neighbors": 30,
            "min_dist": 0.1,
            "metric": "euclidean",
            "random_state": 0,
        },
        "tsne": {
            "enabled": False,
            "n_components": 2,
            "perplexity": 40,
            "n_iter": 1500,
            "learning_rate": "auto",
            "sample_size": 20000,
            "random_state": 0,
        },
    },
    "plotting": {
        "enabled": True,
        "guid_trajectory": {
            "enabled": True,
            "embedding": "pca",
            "color_by": "t_abs_s",
            "show_epoch_boundaries": True,
            "time_series": {
                "enabled": True,
                "include_dims": ["pc1", "pc2"],
                "include_speed": True,
                "include_te": True
            }
        },
        "signals": None,
        "max_signals": 12,
        "max_epochs_per_signal": 4,
        "epoch_examples": 16,
        "prefer_embeddings": ["pca", "umap"],
        "color_by": "te",
        "vector_field": {
            "enabled": True,
            "grid_size": 30,
            "neighbors": 20,
            "embedding": "pca",
        },
        "kmeans_states": {
            "enabled": True,
            "k": 6,
            "sample_size": 100000,
        },
        "hmm_states": {
            "enabled": False,
            "n_states": 6,
            "covariance_type": "diag",
            "sample_size": 60000,
        },
        "uncertainty_path": True,
        "dynamics_curves": True,
        "recurrence": True,
    },
    "metrics": {
        "enabled": True,
        "max_samples": 50000,
        "compute_pairwise": True,
    },
    "prototypes": {
        "enabled": False,
        "max_epochs_per_class": 30,
        "n_clusters": None,
    },
    "change_point": {
        "enabled": False,
        "feature": "speed",
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def merge_analysis_config(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_ANALYSIS_CONFIG)
    if user_cfg:
        cfg = _deep_update(cfg, copy.deepcopy(user_cfg))
    return cfg


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_guid(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value).strip() or "unknown"


def tensor_list_from_batch(value: Any, expected_len: int) -> List[Any]:
    if value is None:
        return [None] * expected_len
    if isinstance(value, torch.Tensor):
        value_cpu = value.detach().cpu()
        if value_cpu.ndim == 0:
            return [value_cpu.clone() for _ in range(expected_len)]
        return [value_cpu[i].clone() for i in range(min(expected_len, value_cpu.size(0)))]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [value.item()] * expected_len
        return [value[i] for i in range(min(expected_len, value.shape[0]))]
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) >= expected_len:
            return seq[:expected_len]
        if not seq:
            return [None] * expected_len
        pad_value = seq[-1]
        return seq + [pad_value] * (expected_len - len(seq))
    return [value for _ in range(expected_len)]


def to_numpy(sample: Any) -> np.ndarray:
    if sample is None:
        return np.asarray([])
    if isinstance(sample, torch.Tensor):
        return sample.detach().cpu().numpy()
    return np.asarray(sample)


def derive_label(sample: Any, class_names: Sequence[str]) -> Tuple[str, Optional[int]]:
    arr = to_numpy(sample)
    if arr.size == 0:
        return "unknown", None
    arr = np.nan_to_num(arr)
    arr = np.squeeze(arr)
    label_idx: Optional[int] = None

    if arr.ndim == 0:
        label_idx = int(round(float(arr)))
    elif arr.ndim == 1:
        if arr.dtype.kind in {"i", "u"}:
            values, counts = np.unique(arr.astype(int), return_counts=True)
            if counts.size:
                label_idx = int(values[np.argmax(counts)])
        elif arr.size <= len(class_names) and arr.max() <= 1.0 + EPS:
            label_idx = int(np.argmax(arr))
        else:
            rounded = np.round(arr).astype(int)
            rounded = rounded[rounded >= 0]
            if rounded.size:
                values, counts = np.unique(rounded, return_counts=True)
                label_idx = int(values[np.argmax(counts)])
    else:
        last_dim = arr.shape[-1]
        if last_dim <= len(class_names) and arr.max() <= 1.0 + EPS:
            flat = arr.reshape(-1, last_dim)
            votes = flat.argmax(axis=1)
            if votes.size:
                label_idx = int(np.bincount(votes).argmax())
        else:
            flat = np.round(arr.reshape(-1)).astype(int)
            flat = flat[flat >= 0]
            if flat.size:
                values, counts = np.unique(flat, return_counts=True)
                label_idx = int(values[np.argmax(counts)])

    if label_idx is None or label_idx < 0:
        return "unknown", None

    if label_idx < len(class_names):
        return str(class_names[label_idx]), label_idx
    return f"class_{label_idx}", label_idx


def safe_save_dataframe(
    df: Optional[pd.DataFrame],
    path: Path,
    *,
    save_parquet: bool,
    save_csv: bool,
    compression: str = "snappy",
) -> Optional[Path]:
    if df is None or df.empty:
        return None
    if save_parquet:
        try:
            df.to_parquet(path, index=False, compression=compression)
            return path
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning(f"Failed to save parquet to {path}: {exc}. Falling back to CSV.")
    if save_csv or not save_parquet:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    return None


def save_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def gaussian_mmd(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if sigma is None:
        concat = np.vstack([X, Y])
        if concat.shape[0] > 2000:
            concat = concat[np.random.default_rng(0).choice(concat.shape[0], 2000, replace=False)]
        dists = np.linalg.norm(concat[:, None, :] - concat[None, :, :], axis=-1)
        sigma = np.median(dists[dists > 0])
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
    gamma = 1.0 / (2.0 * sigma ** 2)
    def kernel(a, b):
        sq = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)
    Kxx = kernel(X, X).mean()
    Kyy = kernel(Y, Y).mean()
    Kxy = kernel(X, Y).mean()
    return float(Kxx + Kyy - 2.0 * Kxy)


def frechet_distance(mu1: np.ndarray, C1: np.ndarray, mu2: np.ndarray, C2: np.ndarray) -> float:
    if mu1.size == 0 or mu2.size == 0:
        return float("nan")
    diff = mu1 - mu2
    if sqrtm is None:  # pragma: no cover - optional dependency
        return float("nan")
    cov_prod = sqrtm(C1 @ C2)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    fid = diff.dot(diff) + np.trace(C1 + C2 - 2.0 * cov_prod)
    return float(fid)


def add_dynamics(df: pd.DataFrame, latent_dim: int = 16) -> pd.DataFrame:
    if df.empty:
        return df
    zcols = [f"z{i}" for i in range(latent_dim) if f"z{i}" in df.columns]
    if not zcols:
        return df

    def _one_epoch(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("t_in_epoch").copy()
        Z = g[zcols].to_numpy()
        d1 = np.diff(Z, axis=0, prepend=Z[[0]])
        d2 = np.diff(d1, axis=0, prepend=d1[[0]])
        g["speed"] = np.linalg.norm(d1, axis=1)
        g["accel"] = np.linalg.norm(d2, axis=1)
        max_t = float(g["t_in_epoch"].max()) if len(g) else 0.0
        g["t_norm"] = g["t_in_epoch"] / (max_t + EPS)
        return g

    return df.groupby(["signal_id", "epoch_idx"], group_keys=False).apply(_one_epoch)


def summarize_epoch(df_epoch: pd.DataFrame, latent_dim: int = 16) -> Dict[str, float]:
    zcols = [f"z{i}" for i in range(latent_dim) if f"z{i}" in df_epoch.columns]
    g = df_epoch.sort_values("t_in_epoch")
    Z = g[zcols].to_numpy()
    if Z.shape[0] < 2:
        cov = np.eye(len(zcols)) * EPS
        path_len = 0.0
    else:
        diffs = np.diff(Z, axis=0)
        path_len = float(np.linalg.norm(diffs, axis=1).sum())
        cov = np.cov(Z.T) + EPS * np.eye(len(zcols))
    sign, logdet = np.linalg.slogdet(cov)
    summary = {
        "path_length": path_len,
        "logdet_spread": float(logdet),
        "speed_mean": float(g.get("speed", pd.Series(dtype=float)).mean()),
        "speed_p95": float(g.get("speed", pd.Series(dtype=float)).quantile(0.95)),
        "accel_mean": float(g.get("accel", pd.Series(dtype=float)).mean()),
        "te_mean": float(g.get("te", pd.Series(dtype=float)).mean()) if "te" in g else float("nan"),
        "unc_mean": float(g.get("uncertainty", pd.Series(dtype=float)).mean()) if "uncertainty" in g else float("nan"),
    }
    return summary


def summarize_all_epochs(df: pd.DataFrame, latent_dim: int = 16) -> pd.DataFrame:
    rows = []
    for (sid, eidx), group in df.groupby(["signal_id", "epoch_idx"]):
        s = summarize_epoch(group, latent_dim)
        s.update(signal_id=sid, epoch_idx=eidx, label=group["label"].iloc[0], label_idx=group.get("label_idx", pd.Series(dtype=float)).iloc[0] if "label_idx" in group else np.nan)
        rows.append(s)
    return pd.DataFrame(rows)


def summarize_signal(df_signal: pd.DataFrame, latent_dim: int = 16) -> Dict[str, float]:
    zcols = [f"z{i}" for i in range(latent_dim) if f"z{i}" in df_signal.columns]
    g = df_signal.sort_values("t_abs_s")
    Z = g[zcols].to_numpy() if zcols else np.empty((len(g), 0))
    if Z.shape[0] < 2:
        path_len = 0.0
        cov = np.eye(len(zcols)) * EPS if zcols else np.empty((0, 0))
    else:
        diffs = np.diff(Z, axis=0)
        path_len = float(np.linalg.norm(diffs, axis=1).sum())
        cov = np.cov(Z.T) + EPS * np.eye(len(zcols)) if zcols else np.empty((0, 0))
    if cov.size:
        _, logdet = np.linalg.slogdet(cov)
    else:
        logdet = 0.0
    summary = {
        "path_length": path_len,
        "logdet_spread": float(logdet),
        "duration_s": float(g["t_abs_s"].max() - g["t_abs_s"].min() if not g["t_abs_s"].empty else 0.0),
        "epoch_count": int(g["epoch_idx"].nunique()),
        "speed_mean": float(g.get("speed", pd.Series(dtype=float)).mean()),
        "speed_p95": float(g.get("speed", pd.Series(dtype=float)).quantile(0.95)),
        "accel_mean": float(g.get("accel", pd.Series(dtype=float)).mean()),
        "te_mean": float(g.get("te", pd.Series(dtype=float)).mean()) if "te" in g else float("nan"),
        "unc_mean": float(g.get("uncertainty", pd.Series(dtype=float)).mean()) if "uncertainty" in g else float("nan"),
    }
    return summary


def summarize_all_signals(df: pd.DataFrame, latent_dim: int = 16) -> pd.DataFrame:
    rows = []
    for signal_id, group in df.groupby("signal_id"):
        s = summarize_signal(group, latent_dim)
        first = group.iloc[0]
        s.update(signal_id=signal_id, label=first.get("label"), label_idx=first.get("label_idx"))
        rows.append(s)
    return pd.DataFrame(rows)


def build_guid_epoch_trajectory(df: pd.DataFrame, latent_dim: int = 16) -> pd.DataFrame:
    zcols = [f"z{i}" for i in range(latent_dim) if f"z{i}" in df.columns]
    records = []
    for signal_id, group in df.groupby("signal_id"):
        epochs = []
        for epoch_idx, epoch_df in group.groupby("epoch_idx"):
            epoch_sorted = epoch_df.sort_values("t_in_epoch")
            start_time = float(epoch_sorted["t_abs_s"].min())
            end_time = float(epoch_sorted["t_abs_s"].max())
            entry_vec = epoch_sorted[zcols].iloc[0].to_numpy() if zcols else np.empty(0)
            exit_vec = epoch_sorted[zcols].iloc[-1].to_numpy() if zcols else np.empty(0)
            epochs.append({
                "signal_id": signal_id,
                "epoch_idx": epoch_idx,
                "t_abs_start": start_time,
                "t_abs_end": end_time,
                "duration_s": end_time - start_time,
                "entry_vec": entry_vec,
                "exit_vec": exit_vec,
                "label": epoch_sorted["label"].iloc[0],
                "label_idx": epoch_sorted.get("label_idx", pd.Series(dtype=float)).iloc[0] if "label_idx" in epoch_sorted else np.nan,
            })
        epochs.sort(key=lambda r: r["t_abs_start"])
        for order, epoch in enumerate(epochs):
            entry_vec = epoch.pop("entry_vec")
            exit_vec = epoch.pop("exit_vec")
            next_entry = epochs[order + 1]["entry_vec"] if order + 1 < len(epochs) else None
            next_start = epochs[order + 1]["t_abs_start"] if order + 1 < len(epochs) else None
            transition_norm = float(np.linalg.norm(exit_vec - next_entry)) if next_entry is not None and zcols else float("nan")
            gap = float(next_start - epoch["t_abs_end"]) if next_start is not None else float("nan")
            row = dict(epoch)
            row.update(epoch_order=order, next_gap_s=gap, transition_norm=transition_norm)
            if zcols:
                for idx, val in enumerate(entry_vec):
                    row[f"entry_z{idx}"] = float(val)
                for idx, val in enumerate(exit_vec):
                    row[f"exit_z{idx}"] = float(val)
            records.append(row)
    return pd.DataFrame(records)


def fit_global_pca(
    df: pd.DataFrame,
    zcols: Sequence[str],
    *,
    n_components: int = 3,
    standardize: bool = False,
    incremental: bool = False,
    batch_size: int = 8192,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    Z = df[zcols].to_numpy()
    scaler = None
    Z_fit = Z
    if standardize:
        scaler = StandardScaler().fit(Z)
        Z_fit = scaler.transform(Z)
    if incremental and Z_fit.shape[0] > batch_size:
        pca = IncrementalPCA(n_components=n_components)
        for start in range(0, Z_fit.shape[0], batch_size):
            pca.partial_fit(Z_fit[start:start + batch_size])
    else:
        pca = PCA(n_components=n_components, random_state=0).fit(Z_fit)
    X = pca.transform(Z_fit)
    out = df.copy()
    cols = []
    for i in range(n_components):
        col = f"pc{i + 1}"
        out[col] = X[:, i]
        cols.append(col)
    info = {
        "model": pca,
        "scaler": scaler,
        "cols": cols,
        "explained_variance": getattr(pca, "explained_variance_ratio_", None),
    }
    return info, out


def fit_global_umap(
    df: pd.DataFrame,
    zcols: Sequence[str],
    *,
    n_components: int = 2,
    random_state: int = 0,
    **kwargs: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    if umap is None:  # pragma: no cover - optional dependency
        logger.warning("UMAP is not installed; skipping UMAP embedding.")
        return None, None
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    Z = df[zcols].to_numpy()
    X = reducer.fit_transform(Z)
    out = df.copy()
    cols = []
    for i in range(n_components):
        col = f"umap{i + 1}"
        out[col] = X[:, i]
        cols.append(col)
    info = {"model": reducer, "cols": cols}
    return info, out


def fit_tsne(
    df: pd.DataFrame,
    zcols: Sequence[str],
    *,
    n_components: int = 2,
    perplexity: float = 40,
    n_iter: int = 1500,
    learning_rate: Any = "auto",
    random_state: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="pca",
        random_state=random_state,
        verbose=1,
    )
    Z = df[zcols].to_numpy()
    X = tsne.fit_transform(Z)
    out = df.copy()
    cols = []
    for i in range(n_components):
        col = f"tsne{i + 1}"
        out[col] = X[:, i]
        cols.append(col)
    info = {"model": tsne, "cols": cols}
    return info, out


def plot_epoch_trajectory(
    df_emb: pd.DataFrame,
    *,
    signal_id: str,
    epoch_idx: Any,
    xcol: str,
    ycol: str,
    color_by: str,
    save_path: Path,
) -> bool:
    subset = df_emb[(df_emb.signal_id == signal_id) & (df_emb.epoch_idx == epoch_idx)].sort_values("t_in_epoch")
    if subset.empty or xcol not in subset or ycol not in subset:
        return False
    X = subset[[xcol, ycol]].to_numpy()
    clr = subset.get(color_by)
    if clr is None:
        clr = subset["t_in_epoch"].to_numpy()
    else:
        clr = clr.to_numpy()
    points = X.reshape(-1, 1, 2)
    if len(points) < 2:
        return False
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis")
    lc.set_array(clr[:-1])
    lc.set_linewidth(2)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.add_collection(lc)
    ax.scatter(X[0, 0], X[0, 1], s=60, marker="o", label="start")
    ax.scatter(X[-1, 0], X[-1, 1], s=80, marker="X", label="end")
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title(f"Signal {signal_id} epoch {epoch_idx}")
    ax.autoscale()
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_dynamics(
    df_dyn: pd.DataFrame,
    *,
    signal_id: str,
    epoch_idx: Any,
    save_path_speed: Path,
    save_path_accel: Path,
) -> bool:
    g = df_dyn[(df_dyn.signal_id == signal_id) & (df_dyn.epoch_idx == epoch_idx)].sort_values("t_in_epoch")
    if g.empty or "speed" not in g or "accel" not in g:
        return False
    tsec = g["t_in_epoch"].to_numpy() * LATENT_STEP_SECONDS
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(tsec, g["speed"], label="speed ||dz||")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_speed, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(tsec, g["accel"], label="acceleration ||d²z||")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("accel")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_accel, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_uncertainty_path(
    df_emb: pd.DataFrame,
    *,
    signal_id: str,
    epoch_idx: Any,
    xcol: str,
    ycol: str,
    save_path: Path,
) -> bool:
    g = df_emb[(df_emb.signal_id == signal_id) & (df_emb.epoch_idx == epoch_idx)].sort_values("t_in_epoch")
    if g.empty or "uncertainty" not in g:
        return False
    X = g[[xcol, ycol]].to_numpy()
    u = g["uncertainty"].to_numpy()
    if not np.isfinite(u).any():
        return False
    u = (u - np.nanmin(u)) / (np.nanmax(u) - np.nanmin(u) + EPS)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(len(X) - 1):
        alpha = 0.2 + 0.8 * (1 - u[i])
        lw = 1.0 + 3.0 * u[i]
        ax.plot(X[i:i + 2, 0], X[i:i + 2, 1], alpha=alpha, linewidth=lw, color="tab:blue")
    ax.scatter(X[0, 0], X[0, 1], s=60, marker="o")
    ax.scatter(X[-1, 0], X[-1, 1], s=80, marker="X")
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title(f"Uncertainty path {signal_id} epoch {epoch_idx}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_recurrence(
    df_lat: pd.DataFrame,
    *,
    signal_id: str,
    epoch_idx: Any,
    latent_dim: int,
    save_path: Path,
) -> bool:
    zcols = [f"z{i}" for i in range(latent_dim) if f"z{i}" in df_lat.columns]
    g = df_lat[(df_lat.signal_id == signal_id) & (df_lat.epoch_idx == epoch_idx)].sort_values("t_in_epoch")
    if g.empty or not zcols:
        return False
    Z = g[zcols].to_numpy()
    D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(D, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title("Recurrence (pairwise latent distance)")
    ax.set_xlabel("t")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_guid_time_series(
    df_lat: pd.DataFrame,
    df_emb: pd.DataFrame,
    *,
    signal_id: str,
    dims: Sequence[str],
    show_speed: bool,
    show_te: bool,
    save_path: Path,
) -> bool:
    guid_lat = df_lat[df_lat.signal_id == signal_id]
    if guid_lat.empty:
        return False
    guid_lat = guid_lat.sort_values("t_abs_s")
    guid_emb = df_emb[df_emb.signal_id == signal_id].sort_values("t_abs_s")
    if guid_emb.empty:
        return False
    t_abs = guid_lat["t_abs_s"].to_numpy()
    if t_abs.size == 0:
        return False
    epoch_idx = guid_lat["epoch_idx"].to_numpy()
    boundaries = []
    if epoch_idx.size:
        last = epoch_idx[0]
        for idx, (ep, t) in enumerate(zip(epoch_idx, t_abs)):
            if idx > 0 and ep != last:
                boundaries.append(t)
                last = ep
    n_panels = 0
    dims_present = []
    for dim in dims:
        if dim in guid_emb:
            n_panels += 1
            dims_present.append(dim)
    if show_speed and "speed" in guid_lat:
        n_panels += 1
    if show_te and "te" in guid_lat:
        n_panels += 1
    if n_panels == 0:
        return False
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    panel = 0
    for dim in dims_present:
        axes[panel].plot(guid_emb["t_abs_s"], guid_emb[dim], label=dim)
        axes[panel].set_ylabel(dim.upper())
        axes[panel].grid(True)
        panel += 1
    if show_speed and "speed" in guid_lat:
        axes[panel].plot(t_abs, guid_lat["speed"], color="tab:orange", label="speed")
        axes[panel].set_ylabel("speed")
        axes[panel].grid(True)
        panel += 1
    if show_te and "te" in guid_lat:
        axes[panel].plot(t_abs, guid_lat["te"], color="tab:red", label="te")
        axes[panel].set_ylabel("TE")
        axes[panel].grid(True)
        panel += 1
    for ax in axes:
        for b in boundaries:
            ax.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0.0, color="red", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("absolute time (s)")
    fig.suptitle(f"Guid {signal_id} latent evolution")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True



def plot_guid_absolute_trajectory(
    df_emb: pd.DataFrame,
    *,
    signal_id: str,
    xcol: str,
    ycol: str,
    color_by: str = "t_abs_s",
    show_epoch_boundaries: bool = True,
    save_path: Path,
) -> bool:
    d = df_emb[df_emb.signal_id == signal_id]
    if d.empty or xcol not in d or ycol not in d:
        return False
    d = d.sort_values("t_abs_s")
    X = d[[xcol, ycol]].to_numpy()
    if len(X) < 2:
        return False
    if color_by in d:
        clr = d[color_by].to_numpy()
    else:
        clr = d["t_abs_s"].to_numpy()
    points = X.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis")
    lc.set_array(clr[:-1])
    lc.set_linewidth(2)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.add_collection(lc)
    ax.scatter(X[0, 0], X[0, 1], s=70, marker="o", label="start")
    ax.scatter(X[-1, 0], X[-1, 1], s=90, marker="X", label="end")
    if show_epoch_boundaries and "epoch_idx" in d:
        epochs = d["epoch_idx"].to_numpy()
        for idx in range(1, len(epochs)):
            if epochs[idx] != epochs[idx - 1]:
                ax.scatter(
                    X[idx, 0],
                    X[idx, 1],
                    s=40,
                    marker="s",
                    color="white",
                    edgecolors="black",
                    linewidths=0.6,
                )
    if "t_abs_s" in d:
        abs_times = d["t_abs_s"].to_numpy()
        if abs_times.size:
            zero_idx = int(np.argmin(np.abs(abs_times)))
            ax.scatter(X[zero_idx, 0], X[zero_idx, 1], s=60, marker="^", color="red", label="t~0 s")
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title(f"Absolute trajectory - signal {signal_id}")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True





def plot_signal_epochs(
    df_emb: pd.DataFrame,
    *,
    signal_id: str,
    xcol: str,
    ycol: str,
    save_path: Path,
) -> bool:
    d = df_emb[df_emb.signal_id == signal_id]
    if d.empty or xcol not in d or ycol not in d:
        return False
    fig, ax = plt.subplots(figsize=(7, 6))
    for epoch_idx, group in d.groupby("epoch_idx"):
        g = group.sort_values("t_in_epoch")
        ax.plot(g[xcol], g[ycol], alpha=0.8, label=f"epoch {epoch_idx}")
        ax.scatter(g[xcol].iloc[0], g[ycol].iloc[0], s=20)
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title(f"All epochs  signal {signal_id}")
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def fit_kmeans_states(
    df: pd.DataFrame,
    zcols: Sequence[str],
    *,
    k: int = 6,
    sample_size: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[KMeans, pd.DataFrame]:
    if sample_size and len(df) > sample_size:
        sample_df = df.sample(sample_size, random_state=random_state)
    else:
        sample_df = df
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(sample_df[zcols].to_numpy())
    assignments = km.predict(df[zcols].to_numpy())
    out = df.copy()
    out["state_kmeans"] = assignments
    return km, out


def plot_state_timeline(
    df_states: pd.DataFrame,
    *,
    signal_id: str,
    state_col: str,
    save_path: Path,
) -> bool:
    d = df_states[df_states.signal_id == signal_id]
    if d.empty or state_col not in d:
        return False
    fig, ax = plt.subplots(figsize=(10, 2 + 0.2 * d["epoch_idx"].nunique()))
    for epoch_idx, group in d.groupby("epoch_idx"):
        g = group.sort_values("t_abs_s")
        ax.step(g["t_abs_s"], g[state_col], where="post", linewidth=1.5, label=f"epoch {epoch_idx}")
    ax.set_xlabel("absolute time (s)")
    ax.set_ylabel("state")
    ax.set_title(f"Latent state timeline  signal {signal_id}")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_vector_field(
    df_emb: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    grid: int = 30,
    neighbors: int = 20,
    save_path: Path,
) -> bool:
    if df_emb.empty or xcol not in df_emb or ycol not in df_emb:
        return False
    df_sorted = df_emb.sort_values(["signal_id", "epoch_idx", "t_in_epoch"])
    P = df_sorted[[xcol, ycol]].to_numpy()
    pairs = df_sorted[["signal_id", "epoch_idx"]].to_numpy()
    diffs = np.vstack([P[1:] - P[:-1], np.zeros((1, 2))])
    continuity = np.all(pairs[1:] == pairs[:-1], axis=1)
    diffs[~continuity] = 0.0
    xg = np.linspace(P[:, 0].min(), P[:, 0].max(), grid)
    yg = np.linspace(P[:, 1].min(), P[:, 1].max(), grid)
    Xg, Yg = np.meshgrid(xg, yg)
    pts = np.c_[Xg.ravel(), Yg.ravel()]
    nn = NearestNeighbors(n_neighbors=min(neighbors, len(P))).fit(P)
    idxs = nn.kneighbors(pts, return_distance=False)
    Vx = diffs[idxs, 0].mean(axis=1)
    Vy = diffs[idxs, 1].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(P[:, 0], P[:, 1], s=2, alpha=0.2)
    ax.quiver(pts[:, 0], pts[:, 1], Vx, Vy, angles="xy", scale_units="xy", scale=1.0, width=0.002)
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title("Estimated latent flow field")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def plot_class_scatter(
    df_emb: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    save_path: Path,
) -> bool:
    if df_emb.empty or xcol not in df_emb or ycol not in df_emb or "label" not in df_emb:
        return False
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, group in df_emb.groupby("label"):
        ax.scatter(group[xcol], group[ycol], s=5, alpha=0.2, label=str(lab))
    ax.set_xlabel(xcol.upper())
    ax.set_ylabel(ycol.upper())
    ax.set_title("Class-wise latent occupancy")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def lda_projection(
    df: pd.DataFrame,
    zcols: Sequence[str],
    *,
    n_components: int = 2,
) -> Tuple[Optional[LinearDiscriminantAnalysis], Optional[pd.DataFrame], Optional[float]]:
    df_valid = df.dropna(subset=["label_idx"])
    if df_valid.empty:
        return None, None, None
    y = df_valid["label_idx"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None, None, None
    lda = LinearDiscriminantAnalysis(n_components=min(n_components, len(np.unique(y)) - 1))
    X = lda.fit_transform(df_valid[zcols].to_numpy(), y)
    out = df_valid.copy()
    cols = []
    for i in range(X.shape[1]):
        col = f"lda{i + 1}"
        out[col] = X[:, i]
        cols.append(col)
    try:
        acc = cross_val_score(lda, df_valid[zcols].to_numpy(), y, cv=5).mean()
    except Exception:  # pragma: no cover
        acc = None
    return lda, out, acc


@dataclass
class LatentCollectionResult:
    df_latents: pd.DataFrame
    epoch_metadata: pd.DataFrame
    sequence_count: int
    guid_counts: Dict[str, int]


class LatentTrajectoryAnalyzer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        dataloader: Iterable,
        output_dir: Path,
        config: Dict[str, Any],
        latent_dim: int,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.output_dir = ensure_dir(output_dir)
        self.config = merge_analysis_config(config)
        self.latent_dim = latent_dim
        self.keys = self.config.get("keys", {})
        class_names = self.config.get("class_names") or []
        if not class_names:
            class_names = DEFAULT_CLASS_NAMES.copy()
        self.class_names = list(class_names)
        storage_cfg = self.config.get("storage", {})
        self.save_parquet = bool(storage_cfg.get("save_parquet", True))
        self.save_csv = bool(storage_cfg.get("save_csv", not self.save_parquet))
        self.compression = storage_cfg.get("compression", "snappy")

        file_labels_cfg = self.config.get("file_labels", {}) or {}
        self.file_label_map = {}
        self.file_label_basename_map = {}
        for key, value in file_labels_cfg.items():
            label = str(value)
            norm_key = os.path.normpath(str(key))
            self.file_label_map[norm_key.lower()] = label
            self.file_label_basename_map[os.path.basename(norm_key).lower()] = label

        dataset = getattr(self.dataloader, "dataset", None)
        self.dataset_paths = []
        if dataset is not None and hasattr(dataset, "paths"):
            self.dataset_paths = [os.path.normpath(str(p)) for p in dataset.paths]

        self.device = self._resolve_device()
        self.model.to(self.device)
        self.model.eval()
        self._sequence_counter = 0
        self._signal_epoch_counts: Counter[str] = Counter()
        self._epoch_records: List[Dict[str, Any]] = []
        self.data_dir = ensure_dir(self.output_dir / "data")
        self.plots_dir = ensure_dir(self.output_dir / "plots")
        self.artifacts_dir = ensure_dir(self.output_dir / "artifacts")
        self.metrics_dir = ensure_dir(self.output_dir / "metrics")
        self.summary: Dict[str, Any] = {}
        self.latents_df: Optional[pd.DataFrame] = None
        self.latents_with_dyn: Optional[pd.DataFrame] = None
        self.epoch_summary_df: Optional[pd.DataFrame] = None
        self.signal_summary_df: Optional[pd.DataFrame] = None
        self.guid_epoch_df: Optional[pd.DataFrame] = None
        self.reductions: Dict[str, Dict[str, Any]] = {}
        self.reduction_frames: Dict[str, pd.DataFrame] = {}

        perf_cfg = self.config.get("performance", {})
        chunk_cfg = perf_cfg.get("forward_chunk_size")
        self.forward_chunk_size: Optional[int] = None
        if chunk_cfg:
            try:
                candidate = int(chunk_cfg)
                if candidate > 0:
                    self.forward_chunk_size = candidate
            except (TypeError, ValueError):
                logger.warning(f"Invalid forward_chunk_size value '{chunk_cfg}'; defaulting to full batch processing.")
        use_amp_default = self.device.type == "cuda"
        self.use_amp = bool(perf_cfg.get("use_amp", use_amp_default)) and self.device.type == "cuda"
        amp_dtype_name = str(perf_cfg.get("amp_dtype", "float16")).lower()
        if amp_dtype_name == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

    def _resolve_device(self) -> torch.device:
        device_cfg = self.config.get("device")
        if isinstance(device_cfg, str):
            if device_cfg.lower() == "cpu":
                return torch.device("cpu")
            if device_cfg.startswith("cuda") and torch.cuda.is_available():
                return torch.device(device_cfg)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def run(self) -> None:
        start = time.time()
        collection = self.collect_latents()
        if collection.df_latents.empty:
            logger.warning("Latent collection produced no data; aborting analysis.")
            return
        self.latents_df = collection.df_latents
        self.latents_with_dyn = add_dynamics(self.latents_df, latent_dim=self.latent_dim)
        self.epoch_summary_df = summarize_all_epochs(self.latents_with_dyn, latent_dim=self.latent_dim)
        self.signal_summary_df = summarize_all_signals(self.latents_with_dyn, latent_dim=self.latent_dim)
        self.guid_epoch_df = build_guid_epoch_trajectory(self.latents_with_dyn, latent_dim=self.latent_dim)
        self._epoch_metadata_df = collection.epoch_metadata
        self._persist_dataframes()
        self._run_reducers()
        if self.config.get("plotting", {}).get("enabled", True):
            self._generate_plots()
        if self.config.get("metrics", {}).get("enabled", True):
            metrics = self._compute_metrics()
        else:
            metrics = {}
        self.summary = {
            "sequence_count": collection.sequence_count,
            "signal_counts": collection.guid_counts,
            "signal_summary_rows": int(0 if self.signal_summary_df is None else len(self.signal_summary_df)),
            "guid_epoch_rows": int(0 if self.guid_epoch_df is None else len(self.guid_epoch_df)),
            "metrics": metrics,
            "outputs": {
                "data": str(self.data_dir),
                "plots": str(self.plots_dir),
                "artifacts": str(self.artifacts_dir),
            },
            "config": self.config,
            "runtime_sec": float(time.time() - start),
        }
        save_json(self.summary, self.output_dir / "analysis_summary.json")

    def collect_latents(self) -> LatentCollectionResult:
        sampling_cfg = self.config.get("sampling", {})
        target_guids = sampling_cfg.get("target_guids")
        if target_guids:
            target_guids = {normalize_guid(g) for g in target_guids}
        exclude_guids = sampling_cfg.get("exclude_guids")
        if exclude_guids:
            exclude_guids = {normalize_guid(g) for g in exclude_guids}
        max_batches = sampling_cfg.get("max_batches")
        max_sequences = sampling_cfg.get("max_sequences")
        max_epochs_per_signal = sampling_cfg.get("max_epochs_per_signal")
        stride = max(1, int(sampling_cfg.get("stride", 1)))

        rows: List[pd.DataFrame] = []
        device = self.device
        total_sequences = 0
        which_key = self.config.get("which", "mu_post")
        use_amp = self.use_amp and device.type == "cuda"
        max_sequences_reached = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Collecting latents")):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                y_st_src = getattr(batch, self.keys.get("y_st", "fhr_st"))
                y_ph_src = getattr(batch, self.keys.get("y_ph", "fhr_ph"))
                x_ph_src = getattr(batch, self.keys.get("x_ph", "fhr_up_ph"))

                if y_st_src is None or y_ph_src is None or x_ph_src is None:
                    continue

                B_total = y_st_src.shape[0]
                if B_total == 0:
                    continue

                signal_ids = tensor_list_from_batch(getattr(batch, self.keys.get("signal_id", "guid"), None), B_total)
                epoch_vals = tensor_list_from_batch(getattr(batch, self.keys.get("epoch_idx", "epoch"), None), B_total)
                labels_raw = tensor_list_from_batch(getattr(batch, self.keys.get("label"), None), B_total)
                t0_vals = tensor_list_from_batch(getattr(batch, self.keys.get("t0"), None), B_total)
                source_list = tensor_list_from_batch(getattr(batch, self.keys.get("source_file", "source_file"), None), B_total)

                chunk_size = self.forward_chunk_size or B_total
                chunk_size = max(1, min(chunk_size, B_total))

                for chunk_start in range(0, B_total, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B_total)
                    local_slice = slice(chunk_start, chunk_end)

                    y_st = y_st_src[local_slice].to(device, non_blocking=True)
                    y_ph = y_ph_src[local_slice].to(device, non_blocking=True)
                    x_ph = x_ph_src[local_slice].to(device, non_blocking=True)

                    autocast_ctx = (
                        torch.cuda.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
                        if use_amp
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        outputs_chunk = self.model(y_st=y_st, y_ph=y_ph, x_ph=x_ph)

                    outputs_cpu: Dict[str, Any] = {}
                    for key, value in outputs_chunk.items():
                        if torch.is_tensor(value):
                            value_cpu = value.detach().cpu()
                            if value_cpu.dtype in (torch.float16, torch.bfloat16):
                                value_cpu = value_cpu.float()
                            outputs_cpu[key] = value_cpu
                        else:
                            outputs_cpu[key] = value
                    del outputs_chunk

                    traj = outputs_cpu[which_key]
                    B_chunk, T, D = traj.shape

                    keep_indices: List[int] = []
                    guid_list: List[str] = []
                    epoch_list: List[Any] = []
                    label_names: List[str] = []
                    label_indices: List[Optional[int]] = []
                    t0_list: List[float] = []
                    source_values: List[Optional[str]] = []

                    for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
                        guid = normalize_guid(signal_ids[global_idx] if global_idx < len(signal_ids) else None)
                        if target_guids and guid not in target_guids:
                            continue
                        if exclude_guids and guid in exclude_guids:
                            continue
                        if max_epochs_per_signal is not None and self._signal_epoch_counts[guid] >= max_epochs_per_signal:
                            continue

                        source_val = source_list[global_idx] if global_idx < len(source_list) else None
                        if source_val is not None and not isinstance(source_val, str):
                            source_val = str(source_val)

                        label_val = labels_raw[global_idx] if global_idx < len(labels_raw) else None
                        label_name, label_idx = derive_label(label_val, self.class_names)
                        if label_name != "unknown" and label_idx is not None:
                            label_idx = self._ensure_label_index(label_name)
                        else:
                            mapped_label = self._label_from_source_path(source_val)
                            if mapped_label:
                                label_name = mapped_label
                                label_idx = self._ensure_label_index(mapped_label)
                            else:
                                label_idx = None

                        epoch_value_raw = epoch_vals[global_idx] if global_idx < len(epoch_vals) else None
                        if isinstance(epoch_value_raw, torch.Tensor):
                            epoch_value = float(epoch_value_raw.item())
                        elif isinstance(epoch_value_raw, np.ndarray):
                            epoch_value = float(np.asarray(epoch_value_raw).squeeze().item())
                        else:
                            epoch_value = float(epoch_value_raw) if epoch_value_raw is not None else float(self._signal_epoch_counts[guid])

                        t0_raw = t0_vals[global_idx] if global_idx < len(t0_vals) else None
                        if isinstance(t0_raw, torch.Tensor):
                            t0_value = float(t0_raw.item())
                        elif isinstance(t0_raw, np.ndarray):
                            t0_value = float(np.asarray(t0_raw).squeeze().item())
                        elif t0_raw is None:
                            t0_value = epoch_value
                        else:
                            t0_value = float(t0_raw)

                        keep_indices.append(local_idx)
                        guid_list.append(guid)
                        epoch_list.append(epoch_value)
                        label_names.append(label_name)
                        label_indices.append(label_idx if label_idx is not None else np.nan)
                        t0_list.append(t0_value)
                        source_values.append(source_val)
                        self._signal_epoch_counts[guid] += 1
                        self._epoch_records.append(
                            {
                                "sample_idx": self._sequence_counter,
                                "signal_id": guid,
                                "epoch_idx": epoch_value,
                                "label": label_name,
                                "label_idx": label_idx if label_idx is not None else np.nan,
                                "t0": t0_value,
                                "source_file": source_val,
                            }
                        )
                        self._sequence_counter += 1

                    if not keep_indices:
                        continue

                    traj_keep = traj[keep_indices].numpy()
                    B_keep = traj_keep.shape[0]
                    if stride > 1:
                        traj_keep = traj_keep[:, ::stride]
                        T_eff = traj_keep.shape[1]
                    else:
                        T_eff = T
                    flat_Z = traj_keep.reshape(-1, D)

                    te = None
                    if self.config.get("keep_te", True):
                        kld = self.model._kld_loss(
                            outputs_cpu["mu_prior"][keep_indices],
                            outputs_cpu["logvar_prior"][keep_indices],
                            outputs_cpu["mu_post"][keep_indices],
                            outputs_cpu["logvar_post"][keep_indices],
                            reduce_mean=False,
                        )
                        if stride > 1:
                            kld = kld[:, ::stride]
                        te = kld.sum(-1).detach().cpu().numpy().reshape(-1)

                    unc = None
                    if self.config.get("keep_uncertainty", True) and "logvar_post" in outputs_cpu:
                        logvar = outputs_cpu["logvar_post"][keep_indices]
                        if stride > 1:
                            logvar = logvar[:, ::stride]
                        unc = logvar.exp().sum(-1).detach().cpu().numpy().reshape(-1)

                    t_in_epoch = np.tile(np.arange(T_eff), B_keep)
                    if stride > 1:
                        t_in_epoch = t_in_epoch * stride
                    signal_rep = np.repeat(guid_list, T_eff)
                    epoch_rep = np.repeat(epoch_list, T_eff)
                    label_rep = np.repeat(label_names, T_eff)
                    label_idx_rep = np.repeat(label_indices, T_eff)
                    t0_rep = np.repeat(t0_list, T_eff)
                    source_rep = np.repeat(source_values, T_eff)
                    t_abs = t0_rep + LATENT_STEP_SECONDS * t_in_epoch
                    data: Dict[str, Any] = {
                        "signal_id": signal_rep,
                        "epoch_idx": epoch_rep,
                        "label": label_rep,
                        "label_idx": label_idx_rep,
                        "t_in_epoch": t_in_epoch,
                        "t_abs_s": t_abs,
                        "source_file": source_rep,
                        "sample_idx": np.repeat(np.arange(total_sequences, total_sequences + B_keep), T_eff),
                        "batch_idx": batch_idx,
                    }
                    total_sequences += B_keep
                    for d_idx in range(D):
                        data[f"z{d_idx}"] = flat_Z[:, d_idx]
                    if te is not None:
                        data["te"] = te
                    if unc is not None:
                        data["uncertainty"] = unc
                    df_batch = pd.DataFrame(data)
                    rows.append(df_batch)

                    if max_sequences is not None and total_sequences >= max_sequences:
                        max_sequences_reached = True
                        break

                if max_sequences_reached:
                    break

        df_latents = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        epoch_meta_df = pd.DataFrame(self._epoch_records)
        if not df_latents.empty:
            df_latents["label_idx"] = pd.to_numeric(df_latents["label_idx"], errors="coerce")
            df_latents["epoch_idx"] = pd.to_numeric(df_latents["epoch_idx"], errors="coerce")
        return LatentCollectionResult(
            df_latents=df_latents,
            epoch_metadata=epoch_meta_df,
            sequence_count=total_sequences,
            guid_counts=dict(self._signal_epoch_counts),
        )

    def _ensure_label_index(self, label_name: str) -> int:
        if label_name in self.class_names:
            return self.class_names.index(label_name)
        self.class_names.append(label_name)
        return len(self.class_names) - 1

    def _label_from_source_path(self, source_path: Any) -> Optional[str]:
        if not source_path:
            return None
        path_str = os.path.normpath(str(source_path))
        low = path_str.lower()
        if low in self.file_label_map:
            return self.file_label_map[low]
        base = os.path.basename(low)
        if base in self.file_label_basename_map:
            return self.file_label_basename_map[base]
        if not self.file_label_map and not self.file_label_basename_map:
            trimmed = os.path.splitext(base)[0]
            return trimmed or base
        return None

    def _persist_dataframes(self) -> None:
        if self.latents_df is None or self.latents_df.empty:
            return
        safe_save_dataframe(
            self.latents_df,
            self.data_dir / "latent_trajectories.parquet",
            save_parquet=self.save_parquet,
            save_csv=self.save_csv,
            compression=self.compression,
        )
        if self.latents_with_dyn is not None and not self.latents_with_dyn.empty:
            safe_save_dataframe(
                self.latents_with_dyn,
                self.data_dir / "latent_trajectories_with_dynamics.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )
        if self.epoch_summary_df is not None and not self.epoch_summary_df.empty:
            safe_save_dataframe(
                self.epoch_summary_df,
                self.data_dir / "latent_epoch_summary.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )
        if self.signal_summary_df is not None and not self.signal_summary_df.empty:
            safe_save_dataframe(
                self.signal_summary_df,
                self.data_dir / "latent_signal_summary.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )
        if self.guid_epoch_df is not None and not self.guid_epoch_df.empty:
            safe_save_dataframe(
                self.guid_epoch_df,
                self.data_dir / "latent_guid_epoch_trajectory.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )
        if hasattr(self, "_epoch_metadata_df") and not self._epoch_metadata_df.empty:
            safe_save_dataframe(
                self._epoch_metadata_df,
                self.data_dir / "latent_epoch_metadata.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )

    def _run_reducers(self) -> None:
        if self.latents_with_dyn is None or self.latents_with_dyn.empty:
            return
        reducers_cfg = self.config.get("reducers", {})
        zcols = [f"z{i}" for i in range(self.latent_dim) if f"z{i}" in self.latents_with_dyn.columns]
        if not zcols:
            return
        pca_cfg = reducers_cfg.get("pca", {})
        if pca_cfg.get("enabled", True):
            info, df_pca = fit_global_pca(
                self.latents_with_dyn,
                zcols,
                n_components=pca_cfg.get("n_components", 3),
                standardize=pca_cfg.get("standardize", False),
                incremental=pca_cfg.get("incremental", False),
                batch_size=pca_cfg.get("batch_size", 8192),
            )
            self.reductions["pca"] = info
            self.reduction_frames["pca"] = df_pca
            safe_save_dataframe(
                df_pca,
                self.data_dir / "latent_with_pca.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )
            if joblib is not None:
                joblib.dump(info, self.artifacts_dir / "pca_model.joblib")
        umap_cfg = reducers_cfg.get("umap", {})
        if umap_cfg.get("enabled", False):
            info_umap, df_umap = fit_global_umap(
                self.latents_with_dyn,
                zcols,
                n_components=umap_cfg.get("n_components", 2),
                n_neighbors=umap_cfg.get("n_neighbors", 30),
                min_dist=umap_cfg.get("min_dist", 0.1),
                metric=umap_cfg.get("metric", "euclidean"),
                random_state=umap_cfg.get("random_state", 0),
            )
            if info_umap and df_umap is not None:
                self.reductions["umap"] = info_umap
                self.reduction_frames["umap"] = df_umap
                safe_save_dataframe(
                    df_umap,
                    self.data_dir / "latent_with_umap.parquet",
                    save_parquet=self.save_parquet,
                    save_csv=self.save_csv,
                    compression=self.compression,
                )
                if joblib is not None:
                    joblib.dump(info_umap["model"], self.artifacts_dir / "umap_model.joblib")
        tsne_cfg = reducers_cfg.get("tsne", {})
        if tsne_cfg.get("enabled", False):
            sample_size = tsne_cfg.get("sample_size")
            if sample_size and sample_size < len(self.latents_with_dyn):
                df_sample = self.latents_with_dyn.sample(sample_size, random_state=tsne_cfg.get("random_state", 0))
            else:
                df_sample = self.latents_with_dyn
            info_tsne, df_tsne = fit_tsne(
                df_sample,
                zcols,
                n_components=tsne_cfg.get("n_components", 2),
                perplexity=tsne_cfg.get("perplexity", 40),
                n_iter=tsne_cfg.get("n_iter", 1500),
                learning_rate=tsne_cfg.get("learning_rate", "auto"),
                random_state=tsne_cfg.get("random_state", 0),
            )
            self.reductions["tsne"] = info_tsne
            self.reduction_frames["tsne"] = df_tsne
            safe_save_dataframe(
                df_tsne,
                self.data_dir / "latent_with_tsne.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )

    def _select_signals_for_plots(self) -> List[str]:
        plotting_cfg = self.config.get("plotting", {})
        signals = plotting_cfg.get("signals")
        if signals:
            return [normalize_guid(s) for s in signals]
        max_signals = plotting_cfg.get("max_signals", 12)
        if self.latents_df is None or self.latents_df.empty:
            return []
        counts = self.latents_df.groupby("signal_id")["epoch_idx"].nunique().sort_values(ascending=False)
        return list(counts.head(max_signals).index)

    def _generate_plots(self) -> None:
        plotting_cfg = self.config.get("plotting", {})
        if self.latents_with_dyn is None or self.latents_with_dyn.empty:
            return
        signals = self._select_signals_for_plots()
        if not signals:
            return
        prefer_embeddings = plotting_cfg.get("prefer_embeddings", ["pca", "umap", "tsne"])
        color_by = plotting_cfg.get("color_by", "te")
        guid_cfg = plotting_cfg.get("guid_trajectory", {})
        max_epochs = plotting_cfg.get("max_epochs_per_signal", 4)
        latent_dir = ensure_dir(self.plots_dir / "per_epoch")
        dynamics_dir = ensure_dir(self.plots_dir / "dynamics")
        uncertainty_dir = ensure_dir(self.plots_dir / "uncertainty")
        recurrence_dir = ensure_dir(self.plots_dir / "recurrence")
        stitched_dir = ensure_dir(self.plots_dir / "per_signal")
        class_dir = ensure_dir(self.plots_dir / "classes")
        for emb_name, df_emb in self.reduction_frames.items():
            if emb_name not in prefer_embeddings:
                continue
            if emb_name == "pca":
                xcol, ycol = "pc1", "pc2"
            elif emb_name == "umap":
                xcol, ycol = "umap1", "umap2"
            elif emb_name == "tsne":
                xcol, ycol = "tsne1", "tsne2"
            else:
                xcol = ycol = None
            if not xcol or xcol not in df_emb or ycol not in df_emb:
                continue
            for signal in signals:
                epochs = (
                    df_emb[df_emb.signal_id == signal]["epoch_idx"].drop_duplicates().sort_values().tolist()
                )
                for epoch in epochs[:max_epochs]:
                    epoch_path = latent_dir / emb_name / signal
                    ensure_dir(epoch_path)
                    plot_epoch_trajectory(
                        df_emb,
                        signal_id=signal,
                        epoch_idx=epoch,
                        xcol=xcol,
                        ycol=ycol,
                        color_by=color_by,
                        save_path=epoch_path / f"epoch_{epoch}.png",
                    )
                    if plotting_cfg.get("dynamics_curves", True):
                        dyn_path = dynamics_dir / signal
                        ensure_dir(dyn_path)
                        plot_dynamics(
                            self.latents_with_dyn,
                            signal_id=signal,
                            epoch_idx=epoch,
                            save_path_speed=dyn_path / f"speed_epoch_{epoch}.png",
                            save_path_accel=dyn_path / f"accel_epoch_{epoch}.png",
                        )
                    if plotting_cfg.get("uncertainty_path", True):
                        un_path = uncertainty_dir / emb_name / signal
                        ensure_dir(un_path)
                        plot_uncertainty_path(
                            df_emb,
                            signal_id=signal,
                            epoch_idx=epoch,
                            xcol=xcol,
                            ycol=ycol,
                            save_path=un_path / f"unc_epoch_{epoch}.png",
                        )
                    if plotting_cfg.get("recurrence", True):
                        rec_path = recurrence_dir / signal
                        ensure_dir(rec_path)
                        plot_recurrence(
                            self.latents_df,
                            signal_id=signal,
                            epoch_idx=epoch,
                            latent_dim=self.latent_dim,
                            save_path=rec_path / f"rec_epoch_{epoch}.png",
                        )
                stitched_path = stitched_dir / emb_name
                ensure_dir(stitched_path)
                plot_signal_epochs(
                    df_emb,
                    signal_id=signal,
                    xcol=xcol,
                    ycol=ycol,
                    save_path=stitched_path / f"{signal}.png",
                )
                if guid_cfg.get("enabled", True) and guid_cfg.get("embedding", "pca") == emb_name:
                    guid_path = ensure_dir(self.plots_dir / "per_guid_absolute" / emb_name)
                    plot_guid_absolute_trajectory(
                        df_emb,
                        signal_id=signal,
                        xcol=xcol,
                        ycol=ycol,
                        color_by=guid_cfg.get("color_by", "t_abs_s"),
                        show_epoch_boundaries=guid_cfg.get("show_epoch_boundaries", True),
                        save_path=guid_path / f"{signal}.png",
                    )
                    ts_cfg = guid_cfg.get("time_series", {})
                    if ts_cfg.get("enabled", True):
                        dims = ts_cfg.get("include_dims", [xcol, ycol])
                        guid_ts_path = ensure_dir(self.plots_dir / "per_guid_time_series")
                        plot_guid_time_series(
                            self.latents_with_dyn,
                            df_emb,
                            signal_id=signal,
                            dims=dims,
                            show_speed=ts_cfg.get("include_speed", True),
                            show_te=ts_cfg.get("include_te", True),
                            save_path=guid_ts_path / f"{signal}.png",
                        )
            vect_cfg = plotting_cfg.get("vector_field", {})
            if vect_cfg.get("enabled", True) and vect_cfg.get("embedding", "pca") == emb_name:
                vf_path = ensure_dir(self.plots_dir / "vector_field")
                plot_vector_field(
                    df_emb,
                    xcol=xcol,
                    ycol=ycol,
                    grid=vect_cfg.get("grid_size", 30),
                    neighbors=vect_cfg.get("neighbors", 20),
                    save_path=vf_path / f"vector_field_{emb_name}.png",
                )
            if "label" in df_emb:
                plot_class_scatter(
                    df_emb,
                    xcol=xcol,
                    ycol=ycol,
                    save_path=class_dir / f"class_scatter_{emb_name}.png",
                )
        zcols = [f"z{i}" for i in range(self.latent_dim) if f"z{i}" in self.latents_with_dyn.columns]
        if zcols:
            kmeans_cfg = plotting_cfg.get("kmeans_states", {})
            if kmeans_cfg.get("enabled", True):
                _, df_states = fit_kmeans_states(
                    self.latents_with_dyn,
                    zcols,
                    k=kmeans_cfg.get("k", 6),
                    sample_size=kmeans_cfg.get("sample_size"),
                )
                state_dir = ensure_dir(self.plots_dir / "states" / "kmeans")
                for signal in signals:
                    plot_state_timeline(
                        df_states,
                        signal_id=signal,
                        state_col="state_kmeans",
                        save_path=state_dir / f"{signal}.png",
                    )
            hmm_cfg = plotting_cfg.get("hmm_states", {})
            if hmm_cfg.get("enabled", False) and GaussianHMM is not None:
                df_sorted = self.latents_with_dyn.sort_values(["signal_id", "epoch_idx", "t_in_epoch"])
                if hmm_cfg.get("sample_size") and hmm_cfg["sample_size"] < len(df_sorted):
                    df_hmm = df_sorted.sample(hmm_cfg["sample_size"], random_state=0)
                else:
                    df_hmm = df_sorted
                Z = df_hmm[zcols].to_numpy()
                hmm = GaussianHMM(
                    n_components=hmm_cfg.get("n_states", 6),
                    covariance_type=hmm_cfg.get("covariance_type", "diag"),
                    random_state=0,
                    n_iter=200,
                )
                hmm.fit(Z)
                states = hmm.predict(self.latents_with_dyn[zcols].to_numpy())
                df_hmm_states = self.latents_with_dyn.copy()
                df_hmm_states["state_hmm"] = states
                hmm_dir = ensure_dir(self.plots_dir / "states" / "hmm")
                for signal in signals:
                    plot_state_timeline(
                        df_hmm_states,
                        signal_id=signal,
                        state_col="state_hmm",
                        save_path=hmm_dir / f"{signal}.png",
                    )
        lda_res = lda_projection(self.latents_with_dyn, zcols)
        if lda_res[1] is not None:
            lda_df = lda_res[1]
            lda_cols = [col for col in lda_df.columns if col.startswith("lda")]
            if len(lda_cols) >= 2:
                plot_class_scatter(
                    lda_df,
                    xcol=lda_cols[0],
                    ycol=lda_cols[1],
                    save_path=class_dir / "class_scatter_lda.png",
                )
            safe_save_dataframe(
                lda_df,
                self.data_dir / "latent_with_lda.parquet",
                save_parquet=self.save_parquet,
                save_csv=self.save_csv,
                compression=self.compression,
            )

    def _compute_metrics(self) -> Dict[str, Any]:
        if self.latents_df is None or self.latents_df.empty:
            return {}
        metrics_cfg = self.config.get("metrics", {})
        max_samples = metrics_cfg.get("max_samples")
        df = self.latents_df
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=0)
        zcols = [f"z{i}" for i in range(self.latent_dim) if f"z{i}" in df.columns]
        label_valid = df.dropna(subset=["label_idx"]) if "label_idx" in df else pd.DataFrame()
        metrics: Dict[str, Any] = {}
        if not label_valid.empty and len(label_valid["label_idx"].unique()) >= 2:
            Z = label_valid[zcols].to_numpy()
            y = label_valid["label_idx"].astype(int).to_numpy()
            try:
                metrics["silhouette"] = float(silhouette_score(Z, y))
            except Exception:
                metrics["silhouette"] = float("nan")
            try:
                metrics["davies_bouldin"] = float(davies_bouldin_score(Z, y))
            except Exception:
                metrics["davies_bouldin"] = float("nan")
            if metrics_cfg.get("compute_pairwise", True):
                class_metrics = {}
                for a, b in combinations(label_valid["label"].unique(), 2):
                    Za = label_valid[label_valid["label"] == a][zcols].to_numpy()
                    Zb = label_valid[label_valid["label"] == b][zcols].to_numpy()
                    mu_a, mu_b = Za.mean(axis=0), Zb.mean(axis=0)
                    cov_a = np.cov(Za.T) + EPS * np.eye(len(mu_a))
                    cov_b = np.cov(Zb.T) + EPS * np.eye(len(mu_b))
                    class_metrics[f"fid_{a}_vs_{b}"] = frechet_distance(mu_a, cov_a, mu_b, cov_b)
                    class_metrics[f"mmd_{a}_vs_{b}"] = gaussian_mmd(Za, Zb)
                metrics["pairwise"] = class_metrics
        save_json(metrics, self.metrics_dir / "latent_metrics.json")
        return metrics


class SeqVAEGraphModelTest(SeqVAEGraphModel):
    def __init__(self, config_file_path: Optional[str] = None):
        super().__init__(config_file_path)
        self.latent_analysis_config = self.config.get("latent_analysis", {})

    def create_model(self):
        """Override to pick an appropriate checkpoint before loading."""
        self.setup_config()
        self._prepare_test_checkpoint()
        self.load_checkpoint()

    def _prepare_test_checkpoint(self) -> None:
        """Resolve which checkpoint to use for evaluation.

        Preference order:
            1. model_config.base_model_checkpoint (already configured)
            2. seqvae_testing.test_checkpoint_path
            3. model_config.seqvae_checkpoint
        """
        candidate_specs = [
            ("model_config.base_model_checkpoint", getattr(self, "base_model_checkpoint", None)),
            ("seqvae_testing.test_checkpoint_path", getattr(self, "seqvae_testing_checkpoint", None)),
            ("model_config.seqvae_checkpoint", getattr(self, "seqvae_ckp", None)),
        ]

        for source, raw_path in candidate_specs:
            if not raw_path:
                continue
            path_value = os.path.normpath(raw_path)
            if os.path.exists(path_value):
                if source != "model_config.base_model_checkpoint":
                    logger.info(f"Using checkpoint from {source}: {path_value}")
                    self.base_model_checkpoint = path_value
                    self.config.setdefault("model_config", {})["base_model_checkpoint"] = path_value
                else:
                    logger.info(f"Using checkpoint: {path_value}")
                    self.base_model_checkpoint = path_value
                return

        logger.warning(
            "No valid checkpoint found for latent trajectory analysis; proceeding with randomly initialized weights."
        )

    def run_tests(self, test_loader, cuda_device: Optional[Any] = None) -> None:
        latent_dir = Path(self.test_results_dir) / "latent_trajectory"
        latent_dir.mkdir(parents=True, exist_ok=True)
        if cuda_device is not None:
            try:
                if isinstance(cuda_device, str) and cuda_device.lower() == "cpu":
                    self.set_cuda_devices([])
                else:
                    device_index = int(cuda_device)
                    if torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count():
                        self.set_cuda_devices([device_index])
            except Exception as exc:
                logger.warning(f"Failed to honor requested device {cuda_device}: {exc}")
        self.create_model()
        if self.pytorch_model is None:
            logger.error("PyTorch model could not be created or loaded; aborting latent trajectory analysis.")
            return
        analyzer = LatentTrajectoryAnalyzer(
            model=self.pytorch_model,
            dataloader=test_loader,
            output_dir=latent_dir,
            config=self.latent_analysis_config,
            latent_dim=self.latent_dim,
        )
        analyzer.run()


def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    config_file_path = 'SeqVAE-TEB-original/config_v.yaml'
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.isabs(config_file_path):
        config_file_path = os.path.join(project_root, config_file_path)
    config_file_path = os.path.normpath(config_file_path)
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found at: {config_file_path}")
        sys.exit(1)
    with open(config_file_path, 'r', encoding='utf-8') as yaml_file:
        config = yaml.safe_load(yaml_file)
    torch.set_float32_matmul_precision('high')

    def resolve_path(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return os.path.normpath(path_value)
        return os.path.normpath(os.path.join(project_root, path_value))

    dataset_cfg = config.get('dataset_config', {})
    for key in ['vae_train_datasets', 'vae_test_datasets']:
        if key in dataset_cfg:
            dataset_cfg[key] = [resolve_path(p) for p in dataset_cfg[key]]
    if 'stat_path' in dataset_cfg:
        dataset_cfg['stat_path'] = resolve_path(dataset_cfg['stat_path'])
    if 'seqvae_testing' in config and 'test_data_dir' in config['seqvae_testing']:
        config['seqvae_testing']['test_data_dir'] = resolve_path(config['seqvae_testing']['test_data_dir'])

    dataloader_cfg = dataset_cfg.get('dataloader_config', {})
    dataset_kwargs = dataloader_cfg.get('dataset_kwargs', {})
    normalize_fields = dataloader_cfg.get('normalize_fields')
    stat_path = dataset_cfg.get('stat_path')
    test_loader = create_optimized_dataloader(
        hdf5_files=dataset_cfg.get('vae_test_datasets', []),
        batch_size=config['general_config']['batch_size']['test'],
        num_workers=0,
        rank=0,
        world_size=1,
        stats_path=stat_path,
        normalize_fields=normalize_fields,
        pin_memory=True,
        **dataset_kwargs,
    )

    graph_model = SeqVAEGraphModelTest(config_file_path=config_file_path)
    graph_model.run_tests(test_loader)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

