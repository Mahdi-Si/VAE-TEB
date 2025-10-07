import contextlib
import copy
import importlib.util
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Optional imports
joblib, UMAP, GaussianHMM, sqrtm, sns = None, None, None, None, None

try:
    import joblib
except:
    pass

try:
    from umap import UMAP
except:
    try:
        from umap.umap_ import UMAP
    except:
        pass

try:
    from hmmlearn.hmm import GaussianHMM
except:
    pass

try:
    from scipy.linalg import sqrtm
except:
    pass

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
except:
    pass

from hdf5_dataset.hdf5_dataset import create_optimized_dataloader

plt.switch_backend("Agg")
if sns is None:
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3})

LATENT_STEP_SECONDS, EPS = 4.0, 1e-9
DEFAULT_CLASS_NAMES = ["class_0", "class_1", "class_2"]

DEFAULT_CONFIG = {
    "enabled": True, "device": None, "which": "mu_post", "keep_te": True, "keep_uncertainty": True, "latent_dim": 16,
    "keys": {"y_st": "fhr_st", "y_ph": "fhr_ph", "x_ph": "fhr_up_ph", "signal_id": "guid", "epoch_idx": "epoch", "label": "target", "t0": None, "source_file": "source_file"},
    "class_names": [], "file_labels": {},
    "sampling": {"target_guids": None, "exclude_guids": None, "max_batches": None, "max_sequences": None, "max_epochs_per_signal": None, "stride": 1},
    "storage": {"save_parquet": True, "save_csv": False, "compression": "snappy"},
    "performance": {"forward_chunk_size": None, "use_amp": True, "amp_dtype": "float16"},
    "reducers": {
        "pca": {"enabled": True, "n_components": 3, "standardize": False, "incremental": False, "batch_size": 8192},
        "umap": {"enabled": False, "n_components": 2, "n_neighbors": 30, "min_dist": 0.1, "metric": "euclidean", "random_state": 0},
        "tsne": {"enabled": False, "n_components": 2, "perplexity": 40, "n_iter": 1500, "learning_rate": "auto", "sample_size": 20000, "random_state": 0},
    },
    "plotting": {
        "enabled": True,
        "guid_trajectory": {"enabled": True, "embedding": "pca", "color_by": "t_abs_s", "show_epoch_boundaries": True,
                           "time_series": {"enabled": True, "include_dims": ["pc1", "pc2"], "include_speed": True, "include_te": True}},
        "signals": None, "max_signals": 12, "max_epochs_per_signal": 4, "prefer_embeddings": ["pca", "umap"], "color_by": "te",
        "vector_field": {"enabled": True, "grid_size": 30, "neighbors": 20, "embedding": "pca"},
        "kmeans_states": {"enabled": True, "k": 6, "sample_size": 100000},
        "hmm_states": {"enabled": False, "n_states": 6, "covariance_type": "diag", "sample_size": 60000},
        "uncertainty_path": True, "dynamics_curves": True, "recurrence": True,
    },
    "metrics": {"enabled": True, "max_samples": 50000, "compute_pairwise": True},
}


def _load_model():
    spec = importlib.util.spec_from_file_location("seqvae", Path(__file__).parent / "graph_model_train.py")
    if not spec or not spec.loader:
        raise ImportError("Unable to load model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "SeqVAEGraphModel")


SeqVAEGraphModel = _load_model()


def deep_update(base: Dict, updates: Dict) -> Dict:
    for k, v in updates.items():
        base[k] = deep_update(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_guid(v: Any) -> str:
    if v is None:
        return "unknown"
    return str(v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else v).strip() or "unknown"


def to_numpy(v: Any) -> np.ndarray:
    return np.asarray([]) if v is None else (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))


def tensor_to_list(v: Any, n: int) -> List[Any]:
    if v is None:
        return [None] * n
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return [v[i].clone() for i in range(min(n, v.size(0)))] if v.ndim > 0 else [v.clone()] * n
    if isinstance(v, np.ndarray):
        return [v[i] for i in range(min(n, v.shape[0]))] if v.ndim > 0 else [v.item()] * n
    if isinstance(v, (list, tuple)):
        lst = list(v)
        return lst[:n] if len(lst) >= n else lst + [lst[-1] if lst else None] * (n - len(lst))
    return [v] * n


def derive_label(s: Any, names: Sequence[str]) -> Tuple[str, Optional[int]]:
    arr = np.nan_to_num(np.squeeze(to_numpy(s)))
    if arr.size == 0:
        return "unknown", None

    idx = None
    if arr.ndim == 0:
        idx = int(round(float(arr)))
    elif arr.ndim == 1:
        if arr.dtype.kind in {"i", "u"}:
            vals, cnts = np.unique(arr.astype(int), return_counts=True)
            idx = int(vals[np.argmax(cnts)]) if cnts.size else None
        elif arr.size <= len(names) and arr.max() <= 1 + EPS:
            idx = int(np.argmax(arr))
        else:
            rnd = np.round(arr).astype(int)[np.round(arr).astype(int) >= 0]
            if rnd.size:
                vals, cnts = np.unique(rnd, return_counts=True)
                idx = int(vals[np.argmax(cnts)])
    else:
        dim = arr.shape[-1]
        if dim <= len(names) and arr.max() <= 1 + EPS:
            votes = arr.reshape(-1, dim).argmax(axis=1)
            idx = int(np.bincount(votes).argmax()) if votes.size else None
        else:
            flat = np.round(arr.reshape(-1)).astype(int)[np.round(arr.reshape(-1)).astype(int) >= 0]
            if flat.size:
                vals, cnts = np.unique(flat, return_counts=True)
                idx = int(vals[np.argmax(cnts)])

    if idx is None or idx < 0:
        return "unknown", None
    return (str(names[idx]), idx) if idx < len(names) else (f"class_{idx}", idx)


def save_df(df: Optional[pd.DataFrame], path: Path, pq: bool = True, csv: bool = False, comp: str = "snappy") -> Optional[Path]:
    if df is None or df.empty:
        return None
    if pq:
        try:
            df.to_parquet(path, index=False, compression=comp)
            return path
        except Exception as e:
            logger.warning(f"Parquet failed: {e}")
    if csv or not pq:
        p = path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p
    return None


def save_json(d: Dict, p: Path):
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def gaussian_mmd(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    if X.size == 0 or Y.size == 0:
        return float("nan")
    if sigma is None:
        c = np.vstack([X, Y])
        if c.shape[0] > 2000:
            c = c[np.random.default_rng(0).choice(c.shape[0], 2000, replace=False)]
        d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
        sigma = np.median(d[d > 0])
        sigma = 1.0 if not np.isfinite(sigma) or sigma <= 0 else sigma
    g = 1.0 / (2.0 * sigma ** 2)
    k = lambda a, b: np.exp(-g * np.sum((a[:, None] - b[None, :]) ** 2, axis=-1))
    return float(k(X, X).mean() + k(Y, Y).mean() - 2.0 * k(X, Y).mean())


def frechet_distance(mu1: np.ndarray, C1: np.ndarray, mu2: np.ndarray, C2: np.ndarray) -> float:
    if mu1.size == 0 or mu2.size == 0 or sqrtm is None:
        return float("nan")
    diff = mu1 - mu2
    cp = sqrtm(C1 @ C2)
    if np.iscomplexobj(cp):
        cp = cp.real
    return float(diff.dot(diff) + np.trace(C1 + C2 - 2.0 * cp))


def add_dynamics(df: pd.DataFrame, dim: int = 16) -> pd.DataFrame:
    if df.empty:
        return df
    zc = [f"z{i}" for i in range(dim) if f"z{i}" in df.columns]
    if not zc:
        return df

    def proc(g):
        g = g.sort_values("t_in_epoch").copy()
        Z = g[zc].to_numpy()
        d1 = np.diff(Z, axis=0, prepend=Z[[0]])
        d2 = np.diff(d1, axis=0, prepend=d1[[0]])
        g["speed"], g["accel"] = np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1)
        g["t_norm"] = g["t_in_epoch"] / (g["t_in_epoch"].max() + EPS)
        return g

    r = df.groupby(["signal_id", "epoch_idx"], group_keys=False).apply(proc)
    return r.reset_index(drop=False if isinstance(r.index, pd.MultiIndex) else True)


def fit_reducer(df: pd.DataFrame, zc: Sequence[str], method: str, **kw) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    Z = df[zc].to_numpy()

    if method == "pca":
        sc = StandardScaler().fit(Z) if kw.get("standardize") else None
        Zf = sc.transform(Z) if sc else Z
        if kw.get("incremental") and Zf.shape[0] > kw.get("batch_size", 8192):
            m = IncrementalPCA(n_components=kw.get("n_components", 3))
            for st in range(0, Zf.shape[0], kw.get("batch_size", 8192)):
                m.partial_fit(Zf[st:st + kw.get("batch_size", 8192)])
        else:
            m = PCA(n_components=kw.get("n_components", 3), random_state=0).fit(Zf)
        X = m.transform(Zf)
        pf, info = "pc", {"model": m, "scaler": sc, "var": getattr(m, "explained_variance_ratio_", None)}
    elif method == "umap":
        if UMAP is None:
            return None, None
        m = UMAP(n_components=kw.get("n_components", 2), random_state=kw.get("random_state", 0),
                n_neighbors=kw.get("n_neighbors", 30), min_dist=kw.get("min_dist", 0.1), metric=kw.get("metric", "euclidean"))
        X = m.fit_transform(Z)
        pf, info = "umap", {"model": m}
    elif method == "tsne":
        m = TSNE(n_components=kw.get("n_components", 2), perplexity=kw.get("perplexity", 40),
                n_iter=kw.get("n_iter", 1500), learning_rate=kw.get("learning_rate", "auto"),
                init="pca", random_state=kw.get("random_state", 0), verbose=1)
        X = m.fit_transform(Z)
        pf, info = "tsne", {"model": m}
    elif method == "lda":
        dv = df.dropna(subset=["label_idx"])
        if dv.empty:
            return None, None
        y = dv["label_idx"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return None, None
        m = LinearDiscriminantAnalysis(n_components=min(kw.get("n_components", 2), len(np.unique(y)) - 1))
        X = m.fit_transform(dv[zc].to_numpy(), y)
        pf = "lda"
        try:
            acc = cross_val_score(m, dv[zc].to_numpy(), y, cv=5).mean()
        except:
            acc = None
        info, df = {"model": m, "acc": acc}, dv
    else:
        return None, None

    out = df.copy()
    cols = [f"{pf}{i+1}" for i in range(X.shape[1])]
    for i, c in enumerate(cols):
        out[c] = X[:, i]
    info["cols"] = cols
    return info, out


class Plotter:
    def __init__(self, pdir: Path, dim: int):
        self.pdir = pdir
        self.dim = dim

    def _line(self, X: np.ndarray, clr: np.ndarray, xc: str, yc: str, t: str, cm: str = "viridis"):
        if len(X) < 2:
            return None, None
        pts = X.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cm, linewidths=2.5)
        lc.set_array(clr[:-1])

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.add_collection(lc)
        ax.scatter(X[0, 0], X[0, 1], s=100, marker="o", c='green', edgecolors='white', linewidth=2, label="start", zorder=5)
        ax.scatter(X[-1, 0], X[-1, 1], s=120, marker="X", c='red', edgecolors='white', linewidth=2, label="end", zorder=5)
        ax.set_xlabel(xc.upper(), fontweight='bold')
        ax.set_ylabel(yc.upper(), fontweight='bold')
        ax.set_title(t, fontsize=12, fontweight='bold')
        ax.autoscale()
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax

    def traj(self, df: pd.DataFrame, sid: str, eid: Any, xc: str, yc: str, cb: str, sp: Path) -> bool:
        sub = df[(df.signal_id == sid) & (df.epoch_idx == eid)].sort_values("t_in_epoch")
        if sub.empty or xc not in sub or yc not in sub:
            return False

        X = sub[[xc, yc]].to_numpy()
        clr = sub.get(cb, sub["t_in_epoch"]).to_numpy()

        fig, ax = self._line(X, clr, xc, yc, f"Trajectory: {sid} | Epoch {eid}", "plasma" if sns is None else "viridis")
        if fig:
            fig.colorbar(ax.collections[0], ax=ax, label=cb)
            fig.tight_layout()
            fig.savefig(sp, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return True
        return False

    def dyn(self, df: pd.DataFrame, sid: str, eid: Any, sd: Path) -> bool:
        g = df[(df.signal_id == sid) & (df.epoch_idx == eid)].sort_values("t_in_epoch")
        if g.empty or "speed" not in g or "accel" not in g:
            return False

        t = g["t_in_epoch"].to_numpy() * LATENT_STEP_SECONDS
        for m, y, i in [("speed", "Speed", 0), ("accel", "Accel", 2)]:
            fig, ax = plt.subplots(figsize=(9, 4))
            c = (sns.color_palette("husl", 8)[i] if sns else ('#1f77b4' if i == 0 else '#ff7f0e'))
            ax.plot(t, g[m], linewidth=2, color=c)
            ax.fill_between(t, g[m], alpha=0.2, color=c)
            ax.set_xlabel("Time (s)", fontweight='bold')
            ax.set_ylabel(y, fontweight='bold')
            ax.set_title(f"{y}: {sid} | Epoch {eid}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(sd / f"{m}_epoch_{eid}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        return True

    def scatter(self, df: pd.DataFrame, xc: str, yc: str, hc: str, t: str, sp: Path) -> bool:
        if df.empty or xc not in df or yc not in df or hc not in df:
            return False

        fig, ax = plt.subplots(figsize=(8, 7))
        if sns:
            sns.scatterplot(data=df, x=xc, y=yc, hue=hc, s=10, alpha=0.3, ax=ax, palette="Set2")
        else:
            for v, g in df.groupby(hc):
                ax.scatter(g[xc], g[yc], s=5, alpha=0.2, label=str(v))
            ax.legend()

        ax.set_xlabel(xc.upper(), fontweight='bold')
        ax.set_ylabel(yc.upper(), fontweight='bold')
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(sp, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True


@dataclass
class LatentCollectionResult:
    df_latents: pd.DataFrame
    epoch_metadata: pd.DataFrame
    sequence_count: int
    guid_counts: Dict[str, int]


class LatentTrajectoryAnalyzer:
    def __init__(self, model: torch.nn.Module, dataloader: Iterable, output_dir: Path, config: Dict[str, Any], latent_dim: int):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = ensure_dir(output_dir)
        self.config = deep_update(copy.deepcopy(DEFAULT_CONFIG), copy.deepcopy(config or {}))
        self.latent_dim = latent_dim
        self.config["latent_dim"] = latent_dim

        self.keys = self.config.get("keys", {})
        self.class_names = list(self.config.get("class_names") or DEFAULT_CLASS_NAMES.copy())

        st = self.config.get("storage", {})
        self.save_pq, self.save_csv, self.comp = st.get("save_parquet", True), st.get("save_csv", False), st.get("compression", "snappy")

        self.file_label_map = {}
        for k, v in (self.config.get("file_labels", {}) or {}).items():
            nk = os.path.normpath(str(k)).lower()
            self.file_label_map[nk] = str(v)
            self.file_label_map[os.path.basename(nk)] = str(v)

        self.device = self._get_device()
        self.model.to(self.device).eval()

        self._seq_cnt = 0
        self._sig_ep_cnts = Counter()
        self._ep_recs = []

        self.data_dir = ensure_dir(self.output_dir / "data")
        self.plots_dir = ensure_dir(self.output_dir / "plots")
        self.artifacts_dir = ensure_dir(self.output_dir / "artifacts")
        self.metrics_dir = ensure_dir(self.output_dir / "metrics")

        self.latents_df = None
        self.reductions, self.reduction_frames = {}, {}

        pf = self.config.get("performance", {})
        self.fwd_chunk = int(pf.get("forward_chunk_size", 0)) or None
        self.use_amp = pf.get("use_amp", self.device.type == "cuda") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if pf.get("amp_dtype", "float16").lower() == "bfloat16" else torch.float16

        self.plotter = Plotter(self.plots_dir, self.latent_dim)

    def _get_device(self) -> torch.device:
        dc = self.config.get("device")
        if isinstance(dc, str):
            if dc.lower() == "cpu":
                return torch.device("cpu")
            if dc.startswith("cuda") and torch.cuda.is_available():
                return torch.device(dc)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        st = time.time()
        col = self.collect_latents()

        if col.df_latents.empty:
            logger.warning("No data")
            return

        self.latents_df = add_dynamics(col.df_latents, self.latent_dim)

        for df, nm in [(self.latents_df, "latent_trajectories"), (col.epoch_metadata, "epoch_metadata")]:
            if df is not None and not df.empty:
                save_df(df, self.data_dir / f"{nm}.parquet", self.save_pq, self.save_csv, self.comp)

        self._run_reducers()

        if self.config.get("plotting", {}).get("enabled", True):
            self._gen_plots()

        mtr = self._compute_metrics() if self.config.get("metrics", {}).get("enabled", True) else {}

        sum = {"sequence_count": col.sequence_count, "signal_counts": col.guid_counts, "metrics": mtr, "runtime_sec": time.time() - st}
        save_json(sum, self.output_dir / "analysis_summary.json")
        logger.info(f"Done in {sum['runtime_sec']:.2f}s")

    def collect_latents(self) -> LatentCollectionResult:
        smp = self.config.get("sampling", {})
        tg = {normalize_guid(g) for g in smp.get("target_guids", [])} or None
        eg = {normalize_guid(g) for g in smp.get("exclude_guids", [])} or None
        mb, ms, meps = smp.get("max_batches"), smp.get("max_sequences"), smp.get("max_epochs_per_signal")
        stride = max(1, int(smp.get("stride", 1)))

        rows, wk, tot = [], self.config.get("which", "mu_post"), 0

        km = {k: self.keys.get(k, v) for k, v in [("y_st", "fhr_st"), ("y_ph", "fhr_ph"), ("x_ph", "fhr_up_ph"),
              ("signal", "guid"), ("epoch", "epoch"), ("label", "target"), ("t0", None), ("source", "source_file")]}

        with torch.no_grad():
            for bi, batch in enumerate(tqdm(self.dataloader, desc="Collecting")):
                if mb and bi >= mb:
                    break

                ys, yp, xp = [getattr(batch, km[k], None) for k in ["y_st", "y_ph", "x_ph"]]
                if ys is None or yp is None or xp is None:
                    continue

                B = ys.shape[0]
                if B == 0:
                    continue

                sids = tensor_to_list(getattr(batch, km["signal"], None), B)
                evals = tensor_to_list(getattr(batch, km["epoch"], None), B)
                lraws = tensor_to_list(getattr(batch, km["label"], None), B)
                t0s = tensor_to_list(getattr(batch, km["t0"], None) if km["t0"] else None, B)
                srcs = tensor_to_list(getattr(batch, km["source"], None), B)

                cs = min(self.fwd_chunk or B, B)

                for cst in range(0, B, cs):
                    cend = min(cst + cs, B)
                    sl = slice(cst, cend)

                    with (torch.cuda.amp.autocast(device_type="cuda", dtype=self.amp_dtype) if self.use_amp else contextlib.nullcontext()):
                        outs = self.model(y_st=ys[sl].to(self.device), y_ph=yp[sl].to(self.device), x_ph=xp[sl].to(self.device))

                    traj = outs[wk].detach().cpu().float().numpy()
                    Bc, T, D = traj.shape

                    ki, gl, el, ln, li, t0l, sv = [], [], [], [], [], [], []

                    for li_idx in range(Bc):
                        gi = cst + li_idx
                        guid = normalize_guid(sids[gi])

                        if (tg and guid not in tg) or (eg and guid in eg):
                            continue
                        if meps and self._sig_ep_cnts[guid] >= meps:
                            continue

                        sval = str(srcs[gi]) if srcs[gi] else None
                        lname, lidx = derive_label(lraws[gi], self.class_names)

                        if lname == "unknown" and sval:
                            mapped = self.file_label_map.get(os.path.normpath(sval).lower()) or self.file_label_map.get(os.path.basename(sval).lower())
                            if mapped:
                                lname, lidx = mapped, self._ensure_idx(mapped)
                        elif lname != "unknown" and lidx is not None:
                            lidx = self._ensure_idx(lname)

                        eval = evals[gi]
                        ev = float(eval.item() if isinstance(eval, torch.Tensor) else eval.item() if isinstance(eval, np.ndarray) else eval if eval is not None else self._sig_ep_cnts[guid])

                        t0v = t0s[gi]
                        t0val = float(t0v.item() if isinstance(t0v, torch.Tensor) else t0v.item() if isinstance(t0v, np.ndarray) else t0v if t0v is not None else ev)

                        ki.append(li_idx)
                        gl.append(guid)
                        el.append(ev)
                        ln.append(lname)
                        li.append(lidx if lidx is not None else np.nan)
                        t0l.append(t0val)
                        sv.append(sval)
                        self._sig_ep_cnts[guid] += 1

                        self._ep_recs.append({"sample_idx": self._seq_cnt, "signal_id": guid, "epoch_idx": ev,
                                             "label": lname, "label_idx": lidx if lidx is not None else np.nan,
                                             "t0": t0val, "source_file": sval})
                        self._seq_cnt += 1

                    if not ki:
                        continue

                    tk = traj[ki][:, ::stride] if stride > 1 else traj[ki]
                    Bk, Te, D = tk.shape
                    fz = tk.reshape(-1, D)

                    te = None
                    if self.config.get("keep_te", True):
                        kld = self.model._kld_loss(outs["mu_prior"][ki], outs["logvar_prior"][ki],
                                                  outs["mu_post"][ki], outs["logvar_post"][ki], reduce_mean=False)
                        te = (kld[:, ::stride] if stride > 1 else kld).sum(-1).detach().cpu().numpy().reshape(-1)

                    unc = None
                    if self.config.get("keep_uncertainty", True) and "logvar_post" in outs:
                        lv = outs["logvar_post"][ki]
                        unc = (lv[:, ::stride] if stride > 1 else lv).exp().sum(-1).detach().cpu().numpy().reshape(-1)

                    tie = np.tile(np.arange(Te) * stride, Bk)
                    dat = {
                        "signal_id": np.repeat(gl, Te), "epoch_idx": np.repeat(el, Te), "label": np.repeat(ln, Te),
                        "label_idx": np.repeat(li, Te), "t_in_epoch": tie,
                        "t_abs_s": np.repeat(t0l, Te) + LATENT_STEP_SECONDS * tie, "source_file": np.repeat(sv, Te),
                    }

                    for di in range(D):
                        dat[f"z{di}"] = fz[:, di]
                    if te is not None:
                        dat["te"] = te
                    if unc is not None:
                        dat["uncertainty"] = unc

                    rows.append(pd.DataFrame(dat))
                    tot += Bk

                    if ms and tot >= ms:
                        break

                if ms and tot >= ms:
                    break

        dl = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        logger.info(f"Collected {tot} seqs from {len(self._sig_ep_cnts)} signals")

        return LatentCollectionResult(df_latents=dl, epoch_metadata=pd.DataFrame(self._ep_recs),
                                      sequence_count=tot, guid_counts=dict(self._sig_ep_cnts))

    def _ensure_idx(self, ln: str) -> int:
        if ln not in self.class_names:
            self.class_names.append(ln)
        return self.class_names.index(ln)

    def _run_reducers(self):
        if self.latents_df is None or self.latents_df.empty:
            return

        reds = self.config.get("reducers", {})
        zc = [f"z{i}" for i in range(self.latent_dim) if f"z{i}" in self.latents_df.columns]
        if not zc:
            return

        for meth, cfg in reds.items():
            if not cfg.get("enabled", True):
                continue

            dfs = self.latents_df
            if meth == "tsne" and cfg.get("sample_size") and len(dfs) > cfg["sample_size"]:
                dfs = dfs.sample(cfg["sample_size"], random_state=cfg.get("random_state", 0))

            info, dfr = fit_reducer(dfs, zc, meth, **cfg)
            if info and dfr is not None:
                self.reductions[meth] = info
                self.reduction_frames[meth] = dfr
                save_df(dfr, self.data_dir / f"latent_with_{meth}.parquet", self.save_pq, self.save_csv, self.comp)
                if joblib and "model" in info:
                    joblib.dump(info["model"], self.artifacts_dir / f"{meth}_model.joblib")

    def _gen_plots(self):
        plt = self.config.get("plotting", {})
        if self.latents_df is None or self.latents_df.empty:
            return

        sigs = plt.get("signals") or []
        if not sigs:
            ms = plt.get("max_signals", 12)
            cnts = self.latents_df.groupby("signal_id")["epoch_idx"].nunique().sort_values(ascending=False)
            sigs = list(cnts.head(ms).index)

        if not sigs:
            return

        pe = plt.get("prefer_embeddings", ["pca", "umap"])
        cb = plt.get("color_by", "te")
        me = plt.get("max_epochs_per_signal", 4)

        for en, de in self.reduction_frames.items():
            if en not in pe:
                continue

            xc, yc = {"pca": ("pc1", "pc2"), "umap": ("umap1", "umap2"), "tsne": ("tsne1", "tsne2")}.get(en, (None, None))
            if not xc or xc not in de or yc not in de:
                continue

            for sig in sigs:
                eps = de[de.signal_id == sig]["epoch_idx"].drop_duplicates().sort_values().tolist()[:me]

                for ep in eps:
                    ed = ensure_dir(self.plots_dir / "per_epoch" / en / sig)
                    self.plotter.traj(de, sig, ep, xc, yc, cb, ed / f"epoch_{ep}.png")

                    if plt.get("dynamics_curves", True):
                        dd = ensure_dir(self.plots_dir / "dynamics" / sig)
                        self.plotter.dyn(self.latents_df, sig, ep, dd)

            if "label" in de:
                cd = ensure_dir(self.plots_dir / "classes")
                self.plotter.scatter(de, xc, yc, "label", "Class Distribution", cd / f"class_{en}.png")

    def _compute_metrics(self) -> Dict[str, Any]:
        if self.latents_df is None or self.latents_df.empty:
            return {}

        mc = self.config.get("metrics", {})
        ms = mc.get("max_samples")
        df = self.latents_df.sample(ms, random_state=0) if ms and len(self.latents_df) > ms else self.latents_df

        zc = [f"z{i}" for i in range(self.latent_dim) if f"z{i}" in df.columns]
        lv = df.dropna(subset=["label_idx"]) if "label_idx" in df else pd.DataFrame()

        mtr = {}
        if not lv.empty and len(lv["label_idx"].unique()) >= 2:
            Z = lv[zc].to_numpy()
            y = lv["label_idx"].astype(int).to_numpy()

            try:
                mtr["silhouette"] = float(silhouette_score(Z, y))
            except:
                mtr["silhouette"] = float("nan")

            try:
                mtr["davies_bouldin"] = float(davies_bouldin_score(Z, y))
            except:
                mtr["davies_bouldin"] = float("nan")

            if mc.get("compute_pairwise", True):
                cm = {}
                for a, b in combinations(lv["label"].unique(), 2):
                    Za, Zb = lv[lv["label"] == a][zc].to_numpy(), lv[lv["label"] == b][zc].to_numpy()
                    ma, mb = Za.mean(axis=0), Zb.mean(axis=0)
                    ca, cb = np.cov(Za.T) + EPS * np.eye(len(ma)), np.cov(Zb.T) + EPS * np.eye(len(mb))
                    cm[f"fid_{a}_vs_{b}"] = frechet_distance(ma, ca, mb, cb)
                    cm[f"mmd_{a}_vs_{b}"] = gaussian_mmd(Za, Zb)
                mtr["pairwise"] = cm

        save_json(mtr, self.metrics_dir / "latent_metrics.json")
        return mtr


class SeqVAEGraphModelTest(SeqVAEGraphModel):
    def __init__(self, config_file_path: Optional[str] = None):
        super().__init__(config_file_path)
        self.latent_analysis_config = self.config.get("latent_analysis", {})

    def create_model(self):
        self.setup_config()
        self._prep_ckpt()
        self.load_checkpoint()

    def _prep_ckpt(self):
        for src, p in [("base", getattr(self, "base_model_checkpoint", None)),
                      ("test", getattr(self, "seqvae_testing_checkpoint", None)),
                      ("seqvae", getattr(self, "seqvae_ckp", None))]:
            if p and os.path.exists(os.path.normpath(p)):
                self.base_model_checkpoint = os.path.normpath(p)
                if src != "base":
                    self.config.setdefault("model_config", {})["base_model_checkpoint"] = self.base_model_checkpoint
                logger.info(f"Checkpoint: {self.base_model_checkpoint}")
                return
        logger.warning("No checkpoint; using random weights")

    def run_tests(self, tl, cuda_device: Optional[Any] = None):
        ld = ensure_dir(Path(self.test_results_dir) / "latent_trajectory")

        if cuda_device is not None:
            try:
                if isinstance(cuda_device, str) and cuda_device.lower() == "cpu":
                    self.set_cuda_devices([])
                else:
                    di = int(cuda_device)
                    if torch.cuda.is_available() and 0 <= di < torch.cuda.device_count():
                        self.set_cuda_devices([di])
            except Exception as e:
                logger.warning(f"Device setup failed: {e}")

        self.create_model()
        if self.pytorch_model is None:
            logger.error("Model failed")
            return

        analyzer = LatentTrajectoryAnalyzer(self.pytorch_model, tl, ld, self.latent_analysis_config, self.latent_dim)
        analyzer.run()


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    cf = r"config_v.yaml"
    with open(cf, 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    torch.set_float32_matmul_precision('high')

    dc = c.get('dataset_config', {})
    dlc = dc.get('dataloader_config', {})

    tl = create_optimized_dataloader(
        hdf5_files=dc.get('vae_test_datasets', []),
        batch_size=c['general_config']['batch_size']['test'],
        num_workers=0, rank=0, world_size=1,
        stats_path=dc.get('stat_path'),
        normalize_fields=dlc.get('normalize_fields'),
        pin_memory=True,
        **dlc.get('dataset_kwargs', {}),
    )

    m = SeqVAEGraphModelTest(config_file_path=cf)
    m.run_tests(tl)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
