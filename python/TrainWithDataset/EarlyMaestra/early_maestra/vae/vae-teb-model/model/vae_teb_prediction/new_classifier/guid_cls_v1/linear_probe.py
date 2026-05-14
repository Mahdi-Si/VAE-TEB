"""Linear-probe diagnostic on cached VAE latents.

Loads the per-fold precomputed-latent cache, pools the last-K valid
segments of each GUID into a single per-GUID vector using the same
``hat_w`` mask the classifier sees, fits a logistic regression on the
train fold, and reports validation AUROC / Brier. Provides the
*linearly-accessible* ceiling for what the downstream GUID classifier
can extract from these latents.

Designed to be run in a separate terminal **in parallel** with training
— it only reads the on-disk cache, never touches the GPU, never writes
into the run directory.

Configuration lives in the ``__main__`` block at the bottom of this
file — edit those constants and run the file directly from PyCharm
(right-click ``linear_probe.py`` -> Run).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from model.vae_teb_prediction.new_classifier.guid_cls_v1.guid_dataset import (
    GuidSequenceDataset,
)


def _pool_guid_features(item: Dict[str, torch.Tensor], last_k: int) -> np.ndarray:
    """Pool the trailing ``last_k`` segments of one GUID into one vector.

    Pipeline mirroring the classifier's tokenizer but with **uniform**
    time-weights (so no learned attention confounds the probe):

    1. Restrict to the last ``min(last_k, S)`` segments.
    2. For each segment, take the ``hat_w``-weighted mean over time of
       $h^y$, $\\bar\\mu^q$, $\\Delta\\mu = \\bar\\mu^q - \\bar\\mu^p$, and
       $\\log(1 + K_t)$.
    3. Average those per-segment vectors across the kept segments.

    Returns:
        ``np.ndarray`` of shape ``(d_model_vae + 2 * d_z + 1,)``.
    """
    h_y: torch.Tensor = item["h_y"]                # (S, T, d_model_vae)
    mu_post: torch.Tensor = item["mu_post_norm"]   # (S, T, d_z)
    mu_prior: torch.Tensor = item["mu_prior_norm"] # (S, T, d_z)
    kld: torch.Tensor = item["kld_per_t"]          # (S, T)
    hat_w: torch.Tensor = item["hat_w"]            # (S, T)

    S = h_y.shape[0]
    k = min(last_k, S)
    h_y = h_y[-k:]
    mu_post = mu_post[-k:]
    mu_prior = mu_prior[-k:]
    kld = kld[-k:]
    hat_w = hat_w[-k:]
    delta_mu = mu_post - mu_prior

    w = hat_w / hat_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # (k, T)
    w_e = w.unsqueeze(-1)                                          # (k, T, 1)

    h_y_seg = (h_y * w_e).sum(dim=1)                # (k, d_model_vae)
    mu_post_seg = (mu_post * w_e).sum(dim=1)        # (k, d_z)
    delta_mu_seg = (delta_mu * w_e).sum(dim=1)      # (k, d_z)
    kld_seg = (kld * w).sum(dim=1)                  # (k,)

    parts = [
        h_y_seg.mean(dim=0).cpu().numpy(),
        mu_post_seg.mean(dim=0).cpu().numpy(),
        delta_mu_seg.mean(dim=0).cpu().numpy(),
        np.log1p(kld_seg.mean(dim=0).cpu().numpy()).reshape(1),
    ]
    return np.concatenate(parts).astype(np.float32)


def _build_xy(ds: GuidSequenceDataset, last_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise the (X, y) matrices for one partition."""
    y = np.zeros(len(ds), dtype=np.int64)
    feats = []
    for i in range(len(ds)):
        item = ds[i]
        feats.append(_pool_guid_features(item, last_k=last_k))
        y[i] = int(item["label_bin"])
    X = np.stack(feats, axis=0)
    return X, y


def run_linear_probe(
    cache_dir: Path,
    *,
    last_k: int = 3,
    C: float = 0.1,
    warmup_left: int = 30,
    warmup_right: int = 30,
    min_samples_per_guid: int = 5,
) -> Dict[str, float]:
    """Run the probe and return a dict of metrics.

    Args:
        cache_dir: folder containing ``train.hdf5`` and ``val.hdf5``.
        last_k: number of trailing valid segments to pool.
        C: inverse L2 strength for sklearn's logistic regression.
        warmup_left: per-segment mask left trim (steps).
        warmup_right: per-segment mask right trim (steps).
        min_samples_per_guid: matches the trainer's filter.

    Returns:
        ``{"train_auroc", "val_auroc", "train_brier", "val_brier",
           "n_train", "n_val", "elapsed_sec"}``
    """
    train_path = cache_dir / "train.hdf5"
    val_path = cache_dir / "val.hdf5"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected train.hdf5 and val.hdf5 in {cache_dir} "
            f"(found train={train_path.exists()}, val={val_path.exists()})"
        )

    t0 = time.time()
    print(f"[probe] loading datasets from {cache_dir}")
    train_ds = GuidSequenceDataset(
        train_path,
        warmup_left=warmup_left,
        warmup_right=warmup_right,
        min_samples_per_guid=min_samples_per_guid,
    )
    val_ds = GuidSequenceDataset(
        val_path,
        warmup_left=warmup_left,
        warmup_right=warmup_right,
        min_samples_per_guid=min_samples_per_guid,
    )
    print(f"[probe]   train GUIDs: {len(train_ds)}, val GUIDs: {len(val_ds)}")

    print(f"[probe] pooling features (last_k={last_k})")
    X_tr, y_tr = _build_xy(train_ds, last_k=last_k)
    X_va, y_va = _build_xy(val_ds, last_k=last_k)
    print(f"[probe]   X_tr={X_tr.shape}, X_va={X_va.shape}")
    print(f"[probe]   train pos rate = {y_tr.mean():.3f}, val pos rate = {y_va.mean():.3f}")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    print(f"[probe] fitting LogisticRegression(C={C})")
    clf = LogisticRegression(C=C, max_iter=5000, class_weight=None, solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    p_tr = clf.predict_proba(X_tr_s)[:, 1]
    p_va = clf.predict_proba(X_va_s)[:, 1]

    metrics = {
        "train_auroc": float(roc_auc_score(y_tr, p_tr)),
        "val_auroc": float(roc_auc_score(y_va, p_va)),
        "train_brier": float(brier_score_loss(y_tr, p_tr)),
        "val_brier": float(brier_score_loss(y_va, p_va)),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "elapsed_sec": float(time.time() - t0),
    }

    print()
    print("=" * 62)
    print(f"Linear probe   (last_k={last_k}, C={C})")
    print("=" * 62)
    print(f"  Train AUROC: {metrics['train_auroc']:.4f}   "
          f"Val AUROC: {metrics['val_auroc']:.4f}")
    print(f"  Train Brier: {metrics['train_brier']:.4f}   "
          f"Val Brier: {metrics['val_brier']:.4f}")
    print(f"  Elapsed: {metrics['elapsed_sec']:.1f}s")
    print()
    print("Interpretation guide (val AUROC):")
    print("  >= 0.70  -> classifier currently leaves signal on the table")
    print("  ~ 0.65   -> 3-layer transformer near the latent ceiling")
    print("  <= 0.58  -> latents themselves are the bottleneck")
    return metrics


if __name__ == "__main__":
    # =================================================================
    # Configuration — edit these and right-click "Run" in PyCharm.
    # =================================================================

    # Path to the per-fold precomputed-latent cache (the folder must
    # contain ``train.hdf5`` and ``val.hdf5``).
    CACHE_DIR = Path(
        "/data/deid/isilon/MS_model/new_vae_teb_cross_attention/"
        "classification_2/guid_cls_v1_run_frozen/precomputed_latents/fold_1"
    )

    # Number of trailing valid segments per GUID to pool.
    LAST_K = 3

    # Inverse L2 regularisation strength for sklearn LogisticRegression.
    # Try {0.01, 0.1, 1.0} to confirm the probe AUROC is not just a
    # regularisation artefact.
    C = 0.1

    # Mask trim widths (match the trainer's defaults).
    WARMUP_LEFT = 30
    WARMUP_RIGHT = 30

    # Drop GUIDs with fewer valid segments than this (matches trainer).
    MIN_SAMPLES_PER_GUID = 5

    run_linear_probe(
        cache_dir=CACHE_DIR,
        last_k=LAST_K,
        C=C,
        warmup_left=WARMUP_LEFT,
        warmup_right=WARMUP_RIGHT,
        min_samples_per_guid=MIN_SAMPLES_PER_GUID,
    )
