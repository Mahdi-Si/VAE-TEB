r"""Fine-tune the pretrained causal-feature conv-Transformer VAE on one fold's classification pool.

Pretraining saw healthy recordings only. The classification pool written by
``hdf5_dataset/new_pipeline/build_fold_datasets.py`` holds every acidosis/HIE recording and about as
many healthy ones, so ``fold_<k>/train`` is close to 1:1 by recording and its healthy half is the
replay that keeps the model's "normal" from drifting. Four steps; choose any subset, they always run
in this order:

* ``premise``  -- model-free and detector-free: per recording, the lagged cross-correlation of the
  high-passed UA and FHR, minus the same statistic on mismatched (UA of one recording, FHR of
  another) segment pairs, pooled per class and per stratum (hours before delivery, labour stage).
  Answers "is UA->FHR coupling stronger, or later, in the adverse recordings?" before any GPU time is
  spent. Lags are on the **stored** timeline, which the dataset declares canonical (the UA channel is
  stored 20 s early and nothing downstream corrects it), so a physiological reading adds that offset
  in the head, not in the numbers.
* ``finetune`` -- warm start from the pretrained checkpoint (``core_model_checkpoint``, strict load)
  on ``fold_<k>/train``, early stopping on ``fold_<k>/val``. The run's config is the pretrained one
  with five changes: the data, a **constant** beta at the pretrained run's final value (the trainer
  would otherwise restart the KL warm-up from 0 and leave the source pathway unregularised exactly
  while it sees new data), a lower LR, a short LR warm-up, and a shorter early-stopping patience.
  The best checkpoint and its ``resolved_config.yaml`` are copied to ``<output_dir>/core/``, the
  **core checkpoint** the evaluation and the classifier read (both look for the resolved config
  beside the checkpoint).
* ``evaluate`` -- ``teb_vae.lag_attn_transformer_cfs.eval.run`` on each chosen checkpoint, once per
  overrides file (the overrides file names the shards evaluated, e.g. the fold's test files).
* ``classify`` -- ``teb_vae.classifier.run`` on each chosen checkpoint, fold ``k`` only.

**Fold k only, for the fine-tuned model.** It has seen fold k's train and val GUIDs, and those include
the test GUIDs of every other fold, so it may be evaluated and classified on fold k alone (the
classifier's L3 guard refuses the others). A cross-validated result needs one fine-tune per fold:
run this module once per fold, each into its own ``output_dir``.

Run from the repository root::

    python -m teb_vae.lag_attn_transformer_cfs.adverse_finetune --steps premise finetune \
        --pretrained-checkpoint /runs/.../model_checkpoints/lag-attn-trf-cfs-epoch=120.ckpt \
        --kfold-root /data/.../k_fold_cross_validation_dataset --fold 1 --output-dir /runs/ft_fold1

or fill in :data:`RUN_ARGS` at the bottom and press Run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/adverse_finetune.py`` -> up three.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if not __package__ and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from loguru import logger  # noqa: E402

from teb_vae.lag_attn.config import _deep_merge, load_config  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import missing_required, resolve_launch_args  # noqa: E402

STEPS = ("premise", "finetune", "evaluate", "classify")
CHECKPOINT_LABELS = ("finetuned", "pretrained")
MODEL_CLASS = "SeqVaeLagAttnTrfCfs"
CHECKPOINT_STEM = "lag-attn-trf-cfs"
RESOLVED_CONFIG = "resolved_config.yaml"
CORE_DIR, CORE_NAME, CORE_MANIFEST = "core", "core_model.ckpt", "core_manifest.json"

#: Stored segment geometry (``create_new_pipeline.py``): 22 min at 4 Hz, ``epoch`` = segment start
#: in seconds relative to delivery.
FS_RAW = 4.0
SEGMENT_S = 1320.0
CLASSES = ("healthy", "acidosis", "hie")
#: Summary groups: each class, and the pooled adverse outcome.
GROUPS = {"healthy": ("healthy",), "acidosis": ("acidosis",), "hie": ("hie",),
          "adverse": ("acidosis", "hie")}


# =============================================================================
# Shared
# =============================================================================
def subgroup_class(subgroup: str) -> str:
    """``healthy`` / ``acidosis`` / ``hie`` from a subgroup file stem such as ``acidosis_cs``."""
    for name in CLASSES:
        if subgroup.startswith(name):
            return name
    raise ValueError(f"subgroup {subgroup!r} is none of {CLASSES}")


def fold_files(kfold_root: str, fold: int, split: str,
               subgroups: Optional[Sequence[str]] = None) -> List[Path]:
    """The subgroup files of ``<kfold_root>/fold_<fold>/<split>/``, optionally narrowed.

    Raises:
        FileNotFoundError: If the directory holds none of the requested files.
    """
    folder = Path(kfold_root) / f"fold_{fold}" / split
    files = sorted(folder.glob("*.hdf5"))
    if subgroups:
        files = [p for p in files if p.stem in set(subgroups)]
    if not files:
        raise FileNotFoundError(f"no subgroup files in {folder} (subgroups={subgroups})")
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(cmd: Sequence[str], env: Optional[Mapping[str, str]] = None) -> None:
    """Run a child process from the repo root; a non-zero exit raises."""
    logger.info("running: " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=_REPO_ROOT, check=True,
                   env={**os.environ, **(env or {})})


# =============================================================================
# Step 1: premise -- detector-free UA->FHR lagged cross-correlation
# =============================================================================
def high_pass(x: np.ndarray, valid: np.ndarray, *, fs_in: float, fs_out: float,
              detrend_s: float, min_valid_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""Decimate, remove a masked moving-mean trend, and z-score each row over its valid samples.

    Args:
        x: ``(S, N)`` signal at ``fs_in``.
        valid: ``(S, N)`` boolean validity.
        fs_in, fs_out: Input and output rates; ``fs_in / fs_out`` must be an integer.
        detrend_s: Width of the trend window, seconds. Slower changes (baseline, tone) are removed.
        min_valid_frac: A row with fewer valid output samples than this share is dropped entirely.

    Returns:
        ``(y, m)``: the standardised signal, zero wherever ``m`` is ``False``, and its validity.
    """
    from scipy.ndimage import uniform_filter1d

    k = int(round(fs_in / fs_out))
    rows, n = x.shape[0], x.shape[1] // k
    x = x[:, : n * k].reshape(rows, n, k).mean(-1)
    m = valid[:, : n * k].reshape(rows, n, k).all(-1)
    x = np.where(m, x, 0.0)
    width = max(int(round(detrend_s * fs_out)), 1)
    trend = (uniform_filter1d(x, width, axis=1, mode="nearest")
             / np.maximum(uniform_filter1d(m.astype(float), width, axis=1, mode="nearest"), 1e-6))
    y = np.where(m, x - trend, 0.0)
    count = m.sum(1)
    mean = y.sum(1) / np.maximum(count, 1)
    y = np.where(m, y - mean[:, None], 0.0)
    sd = np.sqrt((y ** 2).sum(1) / np.maximum(count, 1))
    keep = (count >= min_valid_frac * n) & (sd > 0)
    y = np.where(keep[:, None], y / np.where(sd > 0, sd, 1.0)[:, None], 0.0)
    return y, m & keep[:, None]


def lagged_sums(u: np.ndarray, mu: np.ndarray, f: np.ndarray, mf: np.ndarray,
                lags: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Per-row $\sum_t u_t f_{t+\ell}$ and the number of valid pairs, at each lag $\ell$.

    A positive lag is FHR **after** UA. Zero-padded FFT, so there is no circular wrap for
    $|\ell| < N$.

    Returns:
        ``(xc, n)``, both ``(S, len(lags))``.
    """
    width = u.shape[1]
    if np.abs(lags).max() >= width:
        raise ValueError(f"lags up to {np.abs(lags).max()} need rows longer than {width}")
    nfft = 1 << int(np.ceil(np.log2(2 * width)))

    def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        full = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
        return full[:, lags % nfft]

    return corr(u, f), np.rint(corr(mu.astype(float), mf.astype(float)))


def _segment_strata(f: Any, hour_edges: Sequence[float]) -> Dict[str, np.ndarray]:
    """Per-segment stratum labels: ``hours`` before delivery (segment midpoint) and labour ``stage``.

    The stage is the classifier's own rule (``stage_of`` over the untrimmed window, with an onset at
    delivery read as unknown), so a stratum here and a stage there name the same segments.
    """
    import pandas as pd

    from teb_vae.classifier.cohort import stage_of
    from teb_vae.lag_attn_cfs.eval.cohort import second_stage_eligibility

    epoch = f["epoch"][:].astype(float)
    guid = f["guid"].asstr()[:]
    hours = -(epoch + SEGMENT_S / 2.0) / 3600.0
    edges = list(hour_edges)
    labels = [f"{a:g}-{b:g}h" for a, b in zip(edges[:-1], edges[1:])]
    index = np.searchsorted(edges, hours, side="right") - 1
    hour_label = np.array([labels[i] if 0 <= i < len(labels) else "other" for i in index])
    if "second_stage_onset" in f:
        ss = pd.Series(f["second_stage_onset"][:].astype(float))
        onset = second_stage_eligibility(
            pd.DataFrame({"guid": guid, "epoch": epoch, "second_stage_onset": ss}))
        sentinel = set(onset["guid"][onset["onset_at_delivery"].astype(bool)])
        ss = ss.where(~pd.Series(guid).isin(sentinel))
        stage = stage_of(ss.to_numpy(), 0.0, SEGMENT_S)
    else:
        stage = np.full(len(guid), "unknown")
    return {"guid": guid, "hours": hour_label, "stage": np.asarray(stage, dtype=str)}


def _validity(f: Any, a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
    """UA and FHR raw validity of rows ``[a, b)``: covered steps, minus the quality build's invalid bits."""
    from hdf5_dataset.signal_quality import FHR_INVALID, FHR_INVALID_EXACT, UA_INVALID

    n = f["fhr"].shape[1]
    weight = f["weight"][a:b]
    covered = np.repeat(weight > 0, int(np.ceil(n / weight.shape[1])), axis=1)[:, :n]
    ua, fhr = covered.copy(), covered.copy()
    if "up_quality" in f:
        ua &= (f["up_quality"][a:b].astype(np.int64) & UA_INVALID) == 0
    if "fhr_quality" in f:
        exact = (f["fhr_repair_exact"][a:b].astype(bool) if "fhr_repair_exact" in f
                 else np.zeros(b - a, dtype=bool))
        bits = np.where(exact, FHR_INVALID_EXACT, FHR_INVALID)[:, None]
        fhr &= (f["fhr_quality"][a:b].astype(np.int64) & bits) == 0
    return ua, fhr


def accumulate_file(path: Path, lags: np.ndarray, *, fs_out: float, detrend_s: float,
                    min_valid_frac: float, hour_edges: Sequence[float],
                    chunk: int = 512) -> Dict[Tuple[str, str, str], np.ndarray]:
    r"""Lagged sums of one subgroup file, keyed ``(guid, stratum_kind, stratum)``.

    Each value is a ``(4, L)`` array: real $\sum uf$, real pair count, null $\sum uf$, null pair
    count. The null pairs each segment's UA with the FHR of the segment half a chunk away (rows are
    written recording by recording, so that is another recording), and drops any pair that lands on
    the same GUID.
    """
    import h5py

    acc: Dict[Tuple[str, str, str], np.ndarray] = defaultdict(lambda: np.zeros((4, lags.size)))
    with h5py.File(path, "r") as f:
        strata = _segment_strata(f, hour_edges)
        total = f["fhr"].shape[0]
        for a in range(0, total, chunk):
            b = min(a + chunk, total)
            ua_valid, fhr_valid = _validity(f, a, b)
            u, mu = high_pass(f["up"][a:b].astype(float), ua_valid, fs_in=FS_RAW, fs_out=fs_out,
                              detrend_s=detrend_s, min_valid_frac=min_valid_frac)
            x, mx = high_pass(f["fhr"][a:b].astype(float), fhr_valid, fs_in=FS_RAW, fs_out=fs_out,
                              detrend_s=detrend_s, min_valid_frac=min_valid_frac)
            real = lagged_sums(u, mu, x, mx, lags)
            partner = np.roll(np.arange(b - a), (b - a) // 2)
            null = lagged_sums(u, mu, x[partner], mx[partner], lags)
            guids = strata["guid"][a:b]
            same = guids == guids[partner]
            for i in range(b - a):
                block = np.stack([real[0][i], real[1][i],
                                  0.0 * null[0][i] if same[i] else null[0][i],
                                  0.0 * null[1][i] if same[i] else null[1][i]])
                for kind in ("all", "hours", "stage"):
                    label = "all" if kind == "all" else strata[kind][a + i]
                    acc[(guids[i], kind, label)] += block
    return dict(acc)


def _ratio(xc: np.ndarray, n: np.ndarray, min_pairs: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_pairs, xc / n, np.nan)


def guid_curves(acc: Mapping[Tuple[str, str, str], np.ndarray],
                min_pairs: int) -> Dict[Tuple[str, str, str], np.ndarray]:
    r"""Per-recording lagged correlation $r$; a lag with fewer than ``min_pairs`` pairs is ``NaN``."""
    return {key: _ratio(sums[0], sums[1], min_pairs) for key, sums in acc.items()}


def pooled_null(sums: Iterable[np.ndarray], min_pairs: int) -> np.ndarray:
    r"""The null correlation $r_0$ pooled over recordings: $\sum xc_0 / \sum n_0$.

    Pooled rather than per recording: one recording's few mismatched pairs give a noisy, smooth
    curve, and subtracting it moves the extremum it is meant to reveal.
    """
    total = np.sum(list(sums), axis=0)
    return _ratio(total[2], total[3], min_pairs)


def group_summary(curves: np.ndarray, rng: np.random.Generator,
                  n_boot: int) -> Dict[str, np.ndarray]:
    """GUID-mean curve with a GUID-bootstrap 95% band; ``curves`` is ``(G, L)``."""
    mean = np.nanmean(curves, axis=0)
    if curves.shape[0] < 2 or n_boot <= 0:
        return {"mean": mean, "lo": np.full_like(mean, np.nan), "hi": np.full_like(mean, np.nan)}
    draws = rng.integers(0, curves.shape[0], size=(n_boot, curves.shape[0]))
    boot = np.stack([np.nanmean(curves[d], axis=0) for d in draws])
    return {"mean": mean, "lo": np.nanpercentile(boot, 2.5, axis=0),
            "hi": np.nanpercentile(boot, 97.5, axis=0)}


def _extreme(curve: np.ndarray, lag_s: np.ndarray, window: Tuple[float, float],
             fn: Any) -> Tuple[float, float]:
    """``(value, lag)`` of ``fn`` (nanargmin / nanargmax) over the lag window; NaNs when empty."""
    inside = (lag_s >= window[0]) & (lag_s <= window[1]) & np.isfinite(curve)
    if not inside.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(inside)[fn(curve[inside])]
    return float(curve[idx]), float(lag_s[idx])


def run_premise(*, kfold_root: str, fold: int, splits: Sequence[str],
                subgroups: Optional[Sequence[str]], out_dir: Path, max_lag_s: float,
                min_lag_s: float, fs_out: float, detrend_s: float, min_valid_frac: float,
                min_pairs_s: float, hour_edges: Sequence[float], n_boot: int, seed: int) -> Path:
    """Write ``per_guid.csv``, ``summary.csv``, ``curves.npz`` and ``xcorr.png`` into ``out_dir``."""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    if "test" in splits:
        raise ValueError("premise reads train/val only: the test split stays untouched until the end")
    lags = np.arange(int(round(min_lag_s * fs_out)), int(round(max_lag_s * fs_out)) + 1)
    lag_s = lags / fs_out
    window = (0.0, float(max_lag_s))
    min_pairs = int(min_pairs_s * fs_out)
    # (split, subgroup, class, guid, kind, label, sums, r) for every recording-stratum
    found = []
    for split in splits:
        for path in fold_files(kfold_root, fold, split, subgroups):
            logger.info(f"premise: {path}")
            acc = accumulate_file(path, lags, fs_out=fs_out, detrend_s=detrend_s,
                                  min_valid_frac=min_valid_frac, hour_edges=hour_edges)
            for key, r in guid_curves(acc, min_pairs).items():
                if np.isfinite(r).any():
                    found.append((split, path.stem, subgroup_class(path.stem), *key, acc[key], r))
    if not found:
        raise RuntimeError("premise: no recording had enough valid UA/FHR pairs")

    per_group: Dict[Tuple[str, str, str], List[Any]] = defaultdict(list)
    for item in found:
        for group, members in GROUPS.items():
            if item[2] in members:
                per_group[(group, item[4], item[5])].append(item)
    nulls = {key: pooled_null((it[6] for it in items), min_pairs)
             for key, items in per_group.items()}

    rows = []
    for split, subgroup, cls, guid, kind, label, sums, r in found:
        d = r - nulls[(cls, kind, label)]
        d_min, lag_min = _extreme(d, lag_s, window, np.argmin)
        d_max, lag_max = _extreme(d, lag_s, window, np.argmax)
        rows.append({"split": split, "subgroup": subgroup, "class": cls, "guid": guid,
                     "stratum_kind": kind, "stratum": label,
                     "n_pairs_lag0": int(sums[1][lags == 0].sum()),
                     "d_min": d_min, "d_min_lag_s": lag_min, "d_max": d_max, "d_max_lag_s": lag_max})
    pd.DataFrame(rows).to_csv(out_dir / "per_guid.csv", index=False)

    rng = np.random.default_rng(seed)
    summary, arrays = [], {"lag_s": lag_s}
    for (group, kind, label), items in sorted(per_group.items()):
        r0 = nulls[(group, kind, label)]
        rs = np.stack([it[7] for it in items])
        d = group_summary(rs - r0, rng, n_boot)
        r = np.nanmean(rs, axis=0)
        tag = f"{group}|{kind}|{label}"
        arrays.update({f"{tag}|d_mean": d["mean"], f"{tag}|d_lo": d["lo"], f"{tag}|d_hi": d["hi"],
                       f"{tag}|r_mean": r, f"{tag}|r0_mean": r0})
        d_min, lag_min = _extreme(d["mean"], lag_s, window, np.argmin)
        d_max, lag_max = _extreme(d["mean"], lag_s, window, np.argmax)
        at = np.flatnonzero(lag_s == lag_min)
        summary.append({
            "group": group, "stratum_kind": kind, "stratum": label, "n_guids": len(items),
            "d_min": d_min, "d_min_lag_s": lag_min,
            "d_min_ci_lo": float(d["lo"][at][0]) if at.size else np.nan,
            "d_min_ci_hi": float(d["hi"][at][0]) if at.size else np.nan,
            "d_max": d_max, "d_max_lag_s": lag_max,
            "r_at_d_min": float(r[at][0]) if at.size else np.nan,
            "r0_at_d_min": float(r0[at][0]) if at.size else np.nan,
        })
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", **arrays)
    _premise_figure(per_group, arrays, out_dir / "xcorr.png")
    (out_dir / "premise_config.json").write_text(json.dumps({
        "kfold_root": str(kfold_root), "fold": fold, "splits": list(splits),
        "subgroups": list(subgroups or []), "lag_s": [float(min_lag_s), float(max_lag_s)],
        "fs_out": fs_out, "detrend_s": detrend_s, "min_valid_frac": min_valid_frac,
        "min_pairs_s": min_pairs_s, "hour_edges": list(hour_edges), "n_boot": n_boot,
        "seed": seed, "timeline": "stored (UA stored 20 s early; not corrected)"}, indent=2))
    logger.info(f"premise: wrote {out_dir}")
    return out_dir


def _premise_figure(per_group: Mapping[Tuple[str, str, str], Any], arrays: Mapping[str, np.ndarray],
                    path: Path) -> None:
    """One row per stratum kind, one panel per stratum, one line (with band) per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lag_s = arrays["lag_s"]
    kinds = ("all", "hours", "stage")
    panels = {k: sorted({label for (_g, kind, label) in per_group if kind == k}) for k in kinds}
    ncols = max(len(v) for v in panels.values())
    fig, axes = plt.subplots(len(kinds), ncols, figsize=(3.6 * ncols, 2.8 * len(kinds)),
                             sharex=True, sharey=True, squeeze=False)
    for row, kind in enumerate(kinds):
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(panels[kind]):
                ax.axis("off")
                continue
            label = panels[kind][col]
            for group in ("healthy", "acidosis", "hie"):
                tag = f"{group}|{kind}|{label}"
                if f"{tag}|d_mean" not in arrays:
                    continue
                n = len(per_group[(group, kind, label)])
                line, = ax.plot(lag_s, arrays[f"{tag}|d_mean"], label=f"{group} (n={n})")
                ax.fill_between(lag_s, arrays[f"{tag}|d_lo"], arrays[f"{tag}|d_hi"],
                                color=line.get_color(), alpha=0.2, linewidth=0)
            ax.axhline(0.0, color="0.5", linewidth=0.8)
            ax.axvline(0.0, color="0.5", linewidth=0.8, linestyle=":")
            ax.set_title(f"{kind}: {label}", fontsize=9)
            ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("lag, FHR after UA (s, stored timeline)")
    for row in axes:
        row[0].set_ylabel("r - r_null")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Step 2: finetune
# =============================================================================
def checkpoint_config_path(checkpoint: Path) -> Path:
    """The ``resolved_config.yaml`` the training run wrote beside (or above) ``checkpoint``."""
    root = checkpoint.parent.parent
    for candidate in (checkpoint.parent / RESOLVED_CONFIG, root / "train_results" / RESOLVED_CONFIG,
                      root / RESOLVED_CONFIG):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {RESOLVED_CONFIG} beside {checkpoint}; the core checkpoint and the "
                            f"evaluation both need the run's own config")


def _leaf_diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    """Dotted paths of the leaves where two nested mappings differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in _leaf_diff(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else str(k))]
    return [] if a == b else [prefix]


def final_beta(config: Mapping[str, Any]) -> float:
    """The KL weight the pretrained run ended on: a schedule's ``end``, else its constant value."""
    vae = config.get("model_config", {}).get("VAE_model", {})
    schedule = vae.get("beta_schedule")
    if isinstance(schedule, dict):
        if schedule.get("kind") == "linear_warmup":
            return float(schedule.get("end", 1.0))
        if schedule.get("value") is not None:
            return float(schedule["value"])
    return float(vae.get("kld_beta", 1.0))


def base_config(pretrained: Path, config: Optional[str], allow_mismatch: bool) -> Dict[str, Any]:
    """The config to fine-tune from: ``config`` if given, else the pretrained run's own.

    The model and the loader contract must match the checkpoint's: a ``VAE_model`` key the strict
    load cannot see (a mask switch, a loss weight) would otherwise fine-tune a different model.

    Raises:
        ValueError: If ``config`` disagrees with the checkpoint's resolved config on
            ``model_config.VAE_model``, ``stat_path`` or the loader's fields, unless allowed.
    """
    own = load_config(str(checkpoint_config_path(pretrained)))
    if config is None:
        return own
    given = load_config(config)
    keys = (("model_config", "VAE_model"), ("dataset_config", "stat_path"),
            ("dataset_config", "dataloader_config", "normalize_fields"),
            ("dataset_config", "dataloader_config", "dataset_kwargs"))
    diffs = []
    for path in keys:
        a, b = own, given
        for k in path:
            a, b = (a or {}).get(k), (b or {}).get(k)
        diffs += _leaf_diff(a, b, ".".join(path))
    if diffs and not allow_mismatch:
        raise ValueError(f"--config disagrees with the pretrained checkpoint's resolved config at "
                         f"{diffs[:10]}{' ...' if len(diffs) > 10 else ''}. Drop --config to use the "
                         f"checkpoint's own, or pass --allow-config-mismatch deliberately.")
    if diffs:
        logger.warning(f"--config differs from the checkpoint's config at {diffs}")
    return given


def finetune_config(base: Mapping[str, Any], *, pretrained: Path, train_files: Sequence[Path],
                    val_files: Sequence[Path], tag: str, out_dir_base: Path, lr: float,
                    lr_warmup_steps: int, beta: float, patience: int, max_epochs: int,
                    cuda_devices: Optional[Sequence[int]],
                    extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """``base`` with the fine-tune's data, schedule and identity, then ``extra`` merged last."""
    if set(map(str, train_files)) & set(map(str, val_files)):
        raise ValueError("a file is in both the fine-tune train and val lists")
    general: Dict[str, Any] = {"tag": tag, "lr": lr, "lr_warmup_steps": lr_warmup_steps,
                               "epochs": max_epochs, "folders_config": {"out_dir_base": str(out_dir_base)}}
    if cuda_devices:
        general["cuda_devices"] = list(cuda_devices)
    delta = {
        "general_config": general,
        "model_config": {"core_model_checkpoint": str(pretrained)},
        "dataset_config": {"vae_train_datasets": [str(p) for p in train_files],
                           "vae_test_datasets": [str(p) for p in val_files]},
        "advanced_config": {
            "callbacks": {"early_stopping": {"enabled": True, "patience": int(patience)}},
            "tracking": {"mlflow": {"run_name": tag, "tags": {"variant": tag}}}},
    }
    out = _deep_merge(dict(base), delta)
    # Replaced, not merged: a merge would leave the warm-up's start/end/warmup_epochs beside the
    # constant, a schedule that reads as two things.
    out["model_config"].setdefault("VAE_model", {})["beta_schedule"] = {
        "kind": "constant", "value": float(beta)}
    return _deep_merge(out, dict(extra or {}))


def best_checkpoint(ckpt_dir: Path, monitor: str) -> Path:
    """The best checkpoint of ``monitor``, read off the ModelCheckpoint state of the newest save.

    Lightning updates ``best_model_path`` before it writes a top-k file, so the newest file of the
    primary callback carries the run's final best. The secondary callback's files carry the monitor
    stem in their name and are not globbed.

    Raises:
        FileNotFoundError: If the directory holds no checkpoint, or the recorded best is gone.
    """
    import torch

    files = sorted(ckpt_dir.glob(f"{CHECKPOINT_STEM}-epoch=*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no {CHECKPOINT_STEM}-epoch=*.ckpt in {ckpt_dir}")
    blob = torch.load(files[-1], map_location="cpu", weights_only=False)
    for key, state in (blob.get("callbacks") or {}).items():
        if key.startswith("ModelCheckpoint") and f"'monitor': '{monitor}'" in key:
            best = Path(state["best_model_path"])
            local = ckpt_dir / best.name
            if local.is_file():
                return local
            raise FileNotFoundError(f"the recorded best {best} is not in {ckpt_dir}")
    raise FileNotFoundError(f"{files[-1]} records no ModelCheckpoint state for {monitor!r}")


def save_core(best: Path, run_dir: Path, core_dir: Path, record: Mapping[str, Any]) -> Path:
    """Copy the best checkpoint and its resolved config into ``core_dir``; write the manifest."""
    import torch

    from train.graph_models_utils import check_model_class

    check_model_class(torch.load(best, map_location="cpu", weights_only=False), MODEL_CLASS)
    core_dir.mkdir(parents=True, exist_ok=True)
    core = core_dir / CORE_NAME
    shutil.copy2(best, core)
    shutil.copy2(run_dir / "model_checkpoints" / RESOLVED_CONFIG, core_dir / RESOLVED_CONFIG)
    manifest = {**record, "source_checkpoint": str(best), "source_run_dir": str(run_dir),
                "core_checkpoint": str(core), "core_sha256": _sha256(core),
                "saved_at": datetime.now().isoformat(timespec="seconds")}
    (core_dir / CORE_MANIFEST).write_text(json.dumps(manifest, indent=2))
    logger.info(f"core checkpoint: {core}")
    return core


def run_finetune(*, pretrained: Path, config: Optional[str], allow_mismatch: bool,
                 kfold_root: str, fold: int, subgroups: Optional[Sequence[str]], out_dir: Path,
                 lr: float, lr_warmup_steps: int, beta: Optional[float], patience: int,
                 max_epochs: int, cuda_devices: Optional[Sequence[int]], launcher: str,
                 extra_overrides: Optional[str]) -> Path:
    """Write the fine-tune config, train, and save the core checkpoint. Returns its path."""
    core = out_dir / CORE_DIR / CORE_NAME
    if core.is_file():
        logger.info(f"finetune: {core} exists; skipped (delete {core.parent} to fine-tune again)")
        return core
    base = base_config(pretrained, config, allow_mismatch)
    beta = final_beta(base) if beta is None else float(beta)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{base['general_config']['tag']}_ft_fold{fold}_{stamp}"
    work = out_dir / "finetune"
    train_files = fold_files(kfold_root, fold, "train", subgroups)
    val_files = fold_files(kfold_root, fold, "val", subgroups)
    extra = load_config(extra_overrides) if extra_overrides else None
    cfg = finetune_config(base, pretrained=pretrained, train_files=train_files, val_files=val_files,
                          tag=tag, out_dir_base=work / "runs", lr=lr,
                          lr_warmup_steps=lr_warmup_steps, beta=beta, patience=patience,
                          max_epochs=max_epochs, cuda_devices=cuda_devices, extra=extra)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / f"finetune_config_{stamp}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    logger.info(f"finetune: config {cfg_path}; {len(train_files)} train / {len(val_files)} val "
                f"files, beta={beta}, lr={lr}")

    module = ["-m", "teb_vae.lag_attn_transformer_cfs.trainer", "--config", str(cfg_path)]
    if launcher == "torchrun":
        ranks = len(cfg["general_config"]["cuda_devices"])
        _run(["torchrun", f"--nproc_per_node={ranks}", *module],
             env={"TEB_RUN_STAMP": datetime.now().strftime("%Y-%m-%d--[%H-%M]")})
    elif launcher == "python":
        _run([sys.executable, *module])
    else:
        raise ValueError(f"launcher={launcher!r}; use 'python' or 'torchrun'")

    runs = list((work / "runs").glob(f"*-{tag}"))
    if len(runs) != 1:
        raise RuntimeError(f"expected one run directory *-{tag} under {work / 'runs'}, found {runs}")
    monitor = (cfg.get("advanced_config", {}).get("callbacks", {}).get("model_checkpoint", {})
               .get("monitor", "val/total_loss"))
    best = best_checkpoint(runs[0] / "model_checkpoints", monitor)
    return save_core(best, runs[0], out_dir / CORE_DIR, {
        "fold": fold, "kfold_root": str(kfold_root), "pretrained_checkpoint": str(pretrained),
        "finetune_config": str(cfg_path), "monitor": monitor, "beta": beta, "lr": lr,
        "train_files": [str(p) for p in train_files], "val_files": [str(p) for p in val_files]})


# =============================================================================
# Steps 3 and 4: evaluate and classify, by the packages' own entry points
# =============================================================================
def resolve_checkpoints(labels: Sequence[str], out_dir: Path, pretrained: Optional[Path],
                        finetuned: Optional[Path]) -> Dict[str, Path]:
    """``{label: checkpoint}`` for ``finetuned`` (the core, unless overridden) and ``pretrained``."""
    paths = {"finetuned": finetuned or out_dir / CORE_DIR / CORE_NAME, "pretrained": pretrained}
    out = {}
    for label in labels:
        path = paths[label]
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"{label} checkpoint {path} does not exist")
        out[label] = Path(path)
    return out


def run_evaluate(checkpoints: Mapping[str, Path], overrides: Sequence[str], out_dir: Path,
                 device: Optional[str]) -> None:
    """The model's evaluation once per (checkpoint, overrides file)."""
    if not overrides:
        raise ValueError("evaluate needs --eval-overrides: the file(s) naming the shards to evaluate")
    for label, ckpt in checkpoints.items():
        for path in overrides:
            target = out_dir / "evaluate" / label / Path(path).stem
            cmd = [sys.executable, "-m", "teb_vae.lag_attn_transformer_cfs.eval.run",
                   "--checkpoint", ckpt, "--output-dir", target, "--overrides", path]
            _run(cmd + (["--device", device] if device else []))


def run_classify(checkpoints: Mapping[str, Path], configs: Sequence[str], kfold_root: str,
                 fold: int, stage: str, out_dir: Path, device: Optional[str]) -> None:
    """The classifier once per (checkpoint, classifier config), on fold ``fold`` only."""
    if not configs:
        raise ValueError("classify needs --classifier-configs")
    for label, ckpt in checkpoints.items():
        for path in configs:
            target = out_dir / "classify" / label / Path(path).stem
            cmd = [sys.executable, "-m", "teb_vae.classifier.run", "--config", path,
                   "--stage", stage, "--folds", str(fold), "--run-dir", target,
                   "--set", f"classifier.data.kfold_root={kfold_root}",
                   "--set", f"classifier.source.vae.checkpoint={ckpt}"]
            _run(cmd + (["--device", device] if device else []))


# =============================================================================
# Entry point
# =============================================================================
def main(steps: Optional[Sequence[str]] = None, pretrained_checkpoint: Optional[str] = None,
         config: Optional[str] = None, kfold_root: Optional[str] = None, fold: Optional[int] = None,
         output_dir: Optional[str] = None, subgroups: Optional[Sequence[str]] = None,
         premise_splits: Optional[Sequence[str]] = None, premise_max_lag_s: Optional[float] = None,
         premise_min_lag_s: Optional[float] = None, premise_detrend_s: Optional[float] = None,
         premise_bootstrap: Optional[int] = None, premise_hour_edges: Optional[Sequence[float]] = None,
         lr: Optional[float] = None, lr_warmup_steps: Optional[int] = None,
         beta: Optional[float] = None, patience: Optional[int] = None,
         max_epochs: Optional[int] = None, cuda_devices: Optional[Sequence[int]] = None,
         launcher: Optional[str] = None, finetune_overrides: Optional[str] = None,
         allow_config_mismatch: Optional[bool] = None, finetuned_checkpoint: Optional[str] = None,
         compare: Optional[Sequence[str]] = None, eval_overrides: Optional[Sequence[str]] = None,
         classifier_configs: Optional[Sequence[str]] = None, classifier_stage: Optional[str] = None,
         device: Optional[str] = None, seed: Optional[int] = None) -> int:
    """Apply the defaults and run the chosen steps in order. Returns the exit code."""
    steps = [s for s in STEPS if s in set(steps or STEPS)]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pretrained = Path(pretrained_checkpoint) if pretrained_checkpoint else None
    fold = int(fold)
    seed = 42 if seed is None else int(seed)

    if "premise" in steps:
        run_premise(kfold_root=kfold_root, fold=fold, splits=list(premise_splits or ("train", "val")),
                    subgroups=subgroups, out_dir=out / "premise",
                    max_lag_s=240.0 if premise_max_lag_s is None else float(premise_max_lag_s),
                    min_lag_s=-60.0 if premise_min_lag_s is None else float(premise_min_lag_s),
                    fs_out=1.0,
                    detrend_s=300.0 if premise_detrend_s is None else float(premise_detrend_s),
                    min_valid_frac=0.5, min_pairs_s=600.0,
                    hour_edges=list(premise_hour_edges or (0, 1, 2, 4, 13)),
                    n_boot=1000 if premise_bootstrap is None else int(premise_bootstrap), seed=seed)
    if "finetune" in steps:
        if pretrained is None:
            raise ValueError("finetune needs --pretrained-checkpoint")
        run_finetune(pretrained=pretrained, config=config, allow_mismatch=bool(allow_config_mismatch),
                     kfold_root=kfold_root, fold=fold, subgroups=subgroups, out_dir=out,
                     lr=3e-5 if lr is None else float(lr),
                     lr_warmup_steps=200 if lr_warmup_steps is None else int(lr_warmup_steps),
                     beta=beta, patience=10 if patience is None else int(patience),
                     max_epochs=200 if max_epochs is None else int(max_epochs),
                     cuda_devices=cuda_devices, launcher=launcher or "python",
                     extra_overrides=finetune_overrides)
    if {"evaluate", "classify"} & set(steps):
        checkpoints = resolve_checkpoints(
            list(compare or CHECKPOINT_LABELS), out, pretrained,
            Path(finetuned_checkpoint) if finetuned_checkpoint else None)
        if "evaluate" in steps:
            run_evaluate(checkpoints, list(eval_overrides or []), out, device)
        if "classify" in steps:
            run_classify(checkpoints, list(classifier_configs or []), kfold_root, fold,
                         classifier_stage or "all", out, device)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """No argument carries a default or ``required``: :data:`RUN_ARGS` fills gaps, ``main`` applies
    the real defaults after the merge."""
    p = argparse.ArgumentParser(prog="python -m teb_vae.lag_attn_transformer_cfs.adverse_finetune",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", nargs="+", choices=STEPS)
    p.add_argument("--pretrained-checkpoint", dest="pretrained_checkpoint")
    p.add_argument("--config", help="training config; default: the pretrained run's resolved config")
    p.add_argument("--kfold-root", dest="kfold_root")
    p.add_argument("--fold", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--subgroups", nargs="+", help="subgroup file stems; default: all of them")
    p.add_argument("--premise-splits", dest="premise_splits", nargs="+", choices=("train", "val"))
    p.add_argument("--premise-max-lag-s", dest="premise_max_lag_s", type=float)
    p.add_argument("--premise-min-lag-s", dest="premise_min_lag_s", type=float)
    p.add_argument("--premise-detrend-s", dest="premise_detrend_s", type=float)
    p.add_argument("--premise-bootstrap", dest="premise_bootstrap", type=int)
    p.add_argument("--premise-hour-edges", dest="premise_hour_edges", nargs="+", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--lr-warmup-steps", dest="lr_warmup_steps", type=int)
    p.add_argument("--beta", type=float, help="constant KL weight; default: the pretrained run's final")
    p.add_argument("--patience", type=int)
    p.add_argument("--max-epochs", dest="max_epochs", type=int)
    p.add_argument("--cuda-devices", dest="cuda_devices", nargs="+", type=int)
    p.add_argument("--launcher", choices=("python", "torchrun"))
    p.add_argument("--finetune-overrides", dest="finetune_overrides",
                   help="YAML delta merged last over the fine-tune config")
    p.add_argument("--allow-config-mismatch", dest="allow_config_mismatch",
                   action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--finetuned-checkpoint", dest="finetuned_checkpoint",
                   help="evaluate/classify this instead of <output-dir>/core/core_model.ckpt")
    p.add_argument("--compare", nargs="+", choices=CHECKPOINT_LABELS)
    p.add_argument("--eval-overrides", dest="eval_overrides", nargs="+")
    p.add_argument("--classifier-configs", dest="classifier_configs", nargs="+")
    p.add_argument("--classifier-stage", dest="classifier_stage")
    p.add_argument("--device")
    p.add_argument("--seed", type=int)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("kfold_root", "fold", "output_dir"))
    if refusal:
        raise SystemExit(refusal)
    os.chdir(_REPO_ROOT)
    logger.info("resolved arguments: " + ", ".join(
        f"{k}={values[k]!r} (from {sources[k]})" for k in sorted(values)))
    return main(**values)


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # list[str] | None -- any of "premise", "finetune", "evaluate", "classify"; they run in that
    #        order. None -> all four.
    "steps": ["premise"],
    # str -- the pretrained (healthy-only) checkpoint, inside its run directory so its
    #        resolved_config.yaml is found. Needed by finetune, and by evaluate/classify
    #        whenever "pretrained" is in `compare`.
    "pretrained_checkpoint": None,
    # str | None -- training config to fine-tune from. None -> the pretrained run's own resolved
    #        config, which is the safe choice: a given config must agree with it on the model, the
    #        statistics and the loader fields, or the run is refused.
    "config": None,
    # str -- k_fold_cross_validation_dataset of the SAME build as the pretraining shards and stats.
    "kfold_root": None,
    # int -- the fold. Train on fold_<k>/train, early-stop on fold_<k>/val, never read the test.
    "fold": 1,
    # str -- this fold's working directory: premise/, finetune/, core/, evaluate/, classify/.
    "output_dir": None,
    # list[str] | None -- subgroup file stems to use, e.g. ["acidosis_cs", "healthy_bg_cs"].
    #        None -> every subgroup file in the split (adverse + healthy replay).
    "subgroups": None,

    # ---- premise ----
    "premise_splits": None,        # None -> ["train", "val"]
    "premise_max_lag_s": None,     # None -> 240
    "premise_min_lag_s": None,     # None -> -60 (negative lags: FHR before UA, a control)
    "premise_detrend_s": None,     # None -> 300 s moving-mean high-pass on both signals
    "premise_bootstrap": None,     # None -> 1000 GUID resamples
    "premise_hour_edges": None,    # None -> [0, 1, 2, 4, 13] hours before delivery

    # ---- finetune ----
    "lr": None,                    # None -> 3e-5 (pretraining shipped 3e-4)
    "lr_warmup_steps": None,       # None -> 200
    "beta": None,                  # None -> the pretrained run's final beta, held constant
    "patience": None,              # None -> 10 epochs on the checkpoint monitor
    "max_epochs": None,            # None -> 200 (early stopping ends it sooner)
    "cuda_devices": None,          # None -> the config's own list
    "launcher": None,              # None -> "python" (Lightning spawns DDP); or "torchrun"
    "finetune_overrides": None,    # YAML delta, e.g. spike_breaker margins
    "allow_config_mismatch": None,

    # ---- evaluate / classify ----
    "finetuned_checkpoint": None,  # None -> <output_dir>/core/core_model.ckpt
    "compare": None,               # None -> ["finetuned", "pretrained"]
    # list[str] -- eval override deltas naming the shards to evaluate, e.g. one naming
    #        fold_<k>/test (adverse + healthy) and one naming the pretraining test shards
    #        (healthy forgetting check).
    "eval_overrides": None,
    # list[str] -- classifier configs, e.g. teb_vae/classifier/configs/default.yaml (frozen) and
    #        teb_vae/classifier/configs/cotrain.yaml.
    "classifier_configs": None,
    "classifier_stage": None,      # None -> "all"
    "device": None,                # e.g. "cuda:0"; None -> each entry point's own default
    "seed": None,                  # None -> 42 (premise bootstrap)
}


if __name__ == "__main__":
    sys.exit(_cli())
