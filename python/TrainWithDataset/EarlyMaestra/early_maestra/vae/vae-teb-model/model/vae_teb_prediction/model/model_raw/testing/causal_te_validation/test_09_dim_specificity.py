"""Test 9 — Latent-dimension specificity.

Per-class mean of $K_{i,t,j}$ for each latent dimension $j$, plus the
contrasts $\\Delta_j^{A/H} = \\bar K_{\\mathrm{acidosis},j} - \\bar K_{\\mathrm{healthy},j}$
and $\\Delta_j^{HIE/H} = \\bar K_{\\mathrm{HIE},j} - \\bar K_{\\mathrm{healthy},j}$.

GUID-cluster bootstrap CIs ($B = 200$) on each contrast. Stability
score: how often does each dimension appear in the bootstrap top-3 by
$|\\Delta_j|$?

Verdict: at least 3 dims have CI($\\Delta_j$) excluding 0 (in either
direction) **and** $\\ge 2$ of the full-data top-3 are also stable in
the bootstrap top-3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.statistics import (
    guid_bootstrap_ci,
)


# Class labels in the v1 schema. Healthy is the reference class.
_LABEL_HEALTHY: int = 1
_LABEL_ACIDOSIS: int = 2
_LABEL_HIE: int = 3


def _per_class_means(
    df: pd.DataFrame, dim_cols: List[str],
) -> Dict[int, np.ndarray]:
    """Return ``{label: per-dim mean array}`` for each present class."""
    out: Dict[int, np.ndarray] = {}
    for cls in (_LABEL_HEALTHY, _LABEL_ACIDOSIS, _LABEL_HIE):
        sub = df[df["label"] == cls]
        if sub.empty:
            continue
        arr = sub[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if arr.size == 0:
            continue
        out[cls] = np.nanmean(arr, axis=0)
    return out


def _contrast_with_ci(
    df: pd.DataFrame,
    dim_cols: List[str],
    *,
    cls_a: int,
    cls_b: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-dim mean contrast $\\bar K_{a,j} - \\bar K_{b,j}$ with GUID CI."""
    rows: List[Dict[str, Any]] = []
    sub_a = df[df["label"] == cls_a]
    sub_b = df[df["label"] == cls_b]
    means_a = (
        sub_a[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not sub_a.empty else np.zeros((0, len(dim_cols)))
    )
    means_b = (
        sub_b[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not sub_b.empty else np.zeros((0, len(dim_cols)))
    )
    guids_a = sub_a["guid"].to_numpy() if "guid" in sub_a.columns else np.array([])
    guids_b = sub_b["guid"].to_numpy() if "guid" in sub_b.columns else np.array([])

    for j, col in enumerate(dim_cols):
        a_vals = means_a[:, j] if means_a.shape[0] > 0 else np.array([])
        b_vals = means_b[:, j] if means_b.shape[0] > 0 else np.array([])
        if a_vals.size == 0 or b_vals.size == 0:
            rows.append({
                "dim": int(col.split("_")[-1]),
                "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_a": int(a_vals.size), "n_b": int(b_vals.size),
            })
            continue
        delta = float(np.nanmean(a_vals) - np.nanmean(b_vals))
        # CI on the contrast: bootstrap each side independently, take the
        # difference of resampled means. Cluster by GUID on each side.
        seed_a, seed_b = int(seed) + 7 * j + 1, int(seed) + 7 * j + 2
        ci_lo_a, ci_hi_a = guid_bootstrap_ci(
            a_vals, guids_a, n_boot=int(n_boot), seed=seed_a,
        )
        ci_lo_b, ci_hi_b = guid_bootstrap_ci(
            b_vals, guids_b, n_boot=int(n_boot), seed=seed_b,
        )
        # Approximate contrast CI by combining endpoints (Welch-style
        # interval; conservative when sides are dependent, which they
        # are not here since classes are disjoint).
        ci_low = float(ci_lo_a - ci_hi_b) if all(
            np.isfinite([ci_lo_a, ci_hi_b])
        ) else float("nan")
        ci_high = float(ci_hi_a - ci_lo_b) if all(
            np.isfinite([ci_hi_a, ci_lo_b])
        ) else float("nan")
        rows.append({
            "dim": int(col.split("_")[-1]),
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_a": int(a_vals.size),
            "n_b": int(b_vals.size),
        })
    return pd.DataFrame(rows)


def _bootstrap_top3_stability(
    df: pd.DataFrame,
    dim_cols: List[str],
    *,
    cls_pos: int,
    cls_ref: int,
    n_boot: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Fraction of bootstrap iterations in which each dim is in the top-3 by |delta|.

    Returns a length-``len(dim_cols)`` vector of frequencies in $[0, 1]$.
    """
    rng = np.random.default_rng(int(seed))
    sub_pos = df[df["label"] == cls_pos]
    sub_ref = df[df["label"] == cls_ref]
    if sub_pos.empty or sub_ref.empty:
        return np.zeros(len(dim_cols), dtype=np.float64)
    arr_pos = sub_pos[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    arr_ref = sub_ref[dim_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    g_pos = sub_pos["guid"].to_numpy() if "guid" in sub_pos.columns else np.array([])
    g_ref = sub_ref["guid"].to_numpy() if "guid" in sub_ref.columns else np.array([])
    if g_pos.size == 0 or g_ref.size == 0:
        return np.zeros(len(dim_cols), dtype=np.float64)

    uniq_pos, inv_pos = np.unique(g_pos, return_inverse=True)
    uniq_ref, inv_ref = np.unique(g_ref, return_inverse=True)
    buckets_pos = [np.flatnonzero(inv_pos == j) for j in range(len(uniq_pos))]
    buckets_ref = [np.flatnonzero(inv_ref == j) for j in range(len(uniq_ref))]

    counts = np.zeros(len(dim_cols), dtype=np.float64)
    n_iter = 0
    for _ in range(int(n_boot)):
        s_pos = rng.integers(0, len(buckets_pos), size=len(buckets_pos))
        s_ref = rng.integers(0, len(buckets_ref), size=len(buckets_ref))
        idx_pos = np.concatenate([buckets_pos[s] for s in s_pos])
        idx_ref = np.concatenate([buckets_ref[s] for s in s_ref])
        if idx_pos.size == 0 or idx_ref.size == 0:
            continue
        delta = np.nanmean(arr_pos[idx_pos], axis=0) - np.nanmean(arr_ref[idx_ref], axis=0)
        if not np.any(np.isfinite(delta)):
            continue
        order = np.argsort(-np.abs(np.nan_to_num(delta, nan=0.0)))
        top3 = order[:3]
        counts[top3] += 1.0
        n_iter += 1
    if n_iter == 0:
        return np.zeros(len(dim_cols), dtype=np.float64)
    return counts / float(n_iter)


def _evidence_from_results(
    contrast_a_h: pd.DataFrame,
    contrast_hie_h: pd.DataFrame,
    stability_a_h: np.ndarray,
    stability_hie_h: np.ndarray,
    *,
    n_dims: int,
) -> Dict[str, Any]:
    """Translate per-dim contrasts + stability into the verdict evidence dict."""
    def _n_excludes_zero(df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        return int(((df["ci_low"] > 0) | (df["ci_high"] < 0)).sum())

    n_contrast_a = _n_excludes_zero(contrast_a_h)
    n_contrast_h = _n_excludes_zero(contrast_hie_h)

    def _top3_dims(df: pd.DataFrame) -> List[int]:
        if df.empty:
            return []
        order = np.argsort(-np.abs(df["delta"].fillna(0.0).to_numpy()))
        return df["dim"].iloc[order[:3]].tolist()

    top3_a = _top3_dims(contrast_a_h)
    top3_h = _top3_dims(contrast_hie_h)
    stable_a = (
        int(np.sum(stability_a_h[top3_a] >= 0.6))
        if top3_a and stability_a_h.size >= max(top3_a) + 1 else 0
    )
    stable_h = (
        int(np.sum(stability_hie_h[top3_h] >= 0.6))
        if top3_h and stability_hie_h.size >= max(top3_h) + 1 else 0
    )

    return {
        "n_contrastive_dims": int(max(n_contrast_a, n_contrast_h)),
        "n_contrastive_a_h": n_contrast_a,
        "n_contrastive_hie_h": n_contrast_h,
        "n_stable_in_top3": int(max(stable_a, stable_h)),
        "n_stable_a_h_top3": stable_a,
        "n_stable_hie_h_top3": stable_h,
        "n_dims": int(n_dims),
        "top3_a_h": top3_a,
        "top3_hie_h": top3_h,
    }


def run(
    *,
    histogram_csv: Path,
    output_dir: Path,
    n_boot: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Read ``histogram_metrics.csv``, fit per-dim contrasts, write the report.

    Args:
        histogram_csv: ``<output>/histograms/histogram_metrics.csv``.
        output_dir: ``<output>/causal_te_validation/dim_specificity``.
        n_boot: Bootstrap iterations for stability + CIs.
        seed: RNG seed.

    Returns:
        Dict with ``verdict``, ``evidence``, ``csv_paths``, ``figure_paths``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv = Path(histogram_csv)
    if not csv.is_file():
        return {
            "verdict": "missing", "evidence": {},
            "error": f"missing input: {csv}",
            "csv_paths": [], "figure_paths": [],
        }
    df = pd.read_csv(csv)
    dim_cols = sorted(
        (c for c in df.columns if c.startswith("kld_dim_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if not dim_cols or "label" not in df.columns or "guid" not in df.columns:
        return {
            "verdict": "missing", "evidence": {},
            "error": "histogram CSV missing kld_dim_*/label/guid columns",
            "csv_paths": [], "figure_paths": [],
        }
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    means = _per_class_means(df, dim_cols)
    contrast_a_h = _contrast_with_ci(
        df, dim_cols, cls_a=_LABEL_ACIDOSIS, cls_b=_LABEL_HEALTHY,
        n_boot=int(n_boot), seed=int(seed),
    )
    contrast_hie_h = _contrast_with_ci(
        df, dim_cols, cls_a=_LABEL_HIE, cls_b=_LABEL_HEALTHY,
        n_boot=int(n_boot), seed=int(seed) + 1,
    )
    stability_a_h = _bootstrap_top3_stability(
        df, dim_cols, cls_pos=_LABEL_ACIDOSIS, cls_ref=_LABEL_HEALTHY,
        n_boot=int(n_boot), seed=int(seed) + 100,
    )
    stability_hie_h = _bootstrap_top3_stability(
        df, dim_cols, cls_pos=_LABEL_HIE, cls_ref=_LABEL_HEALTHY,
        n_boot=int(n_boot), seed=int(seed) + 200,
    )

    n_dims = len(dim_cols)
    rows: List[Dict[str, Any]] = []
    for j, col in enumerate(dim_cols):
        dim_idx = int(col.split("_")[-1])
        a_h_row = contrast_a_h[contrast_a_h["dim"] == dim_idx]
        hie_h_row = contrast_hie_h[contrast_hie_h["dim"] == dim_idx]
        rows.append({
            "dim": dim_idx,
            "mean_healthy": (
                float(means[_LABEL_HEALTHY][j]) if _LABEL_HEALTHY in means else float("nan")
            ),
            "mean_acidosis": (
                float(means[_LABEL_ACIDOSIS][j]) if _LABEL_ACIDOSIS in means else float("nan")
            ),
            "mean_hie": (
                float(means[_LABEL_HIE][j]) if _LABEL_HIE in means else float("nan")
            ),
            "delta_A_H": float(a_h_row["delta"].iloc[0]) if not a_h_row.empty else float("nan"),
            "ci_A_H_low": float(a_h_row["ci_low"].iloc[0]) if not a_h_row.empty else float("nan"),
            "ci_A_H_high": float(a_h_row["ci_high"].iloc[0]) if not a_h_row.empty else float("nan"),
            "delta_HIE_H": float(hie_h_row["delta"].iloc[0]) if not hie_h_row.empty else float("nan"),
            "ci_HIE_H_low": float(hie_h_row["ci_low"].iloc[0]) if not hie_h_row.empty else float("nan"),
            "ci_HIE_H_high": float(hie_h_row["ci_high"].iloc[0]) if not hie_h_row.empty else float("nan"),
            "stability_A_H_top3_pct": float(stability_a_h[j]) if j < stability_a_h.size else 0.0,
            "stability_HIE_H_top3_pct": float(stability_hie_h[j]) if j < stability_hie_h.size else 0.0,
        })
    out_df = pd.DataFrame(rows)
    csv_path = output_dir / "per_dim_class_contrast.csv"
    out_df.to_csv(csv_path, index=False)

    evidence = _evidence_from_results(
        contrast_a_h, contrast_hie_h,
        stability_a_h, stability_hie_h, n_dims=n_dims,
    )

    from model.vae_teb_prediction.model.model_raw.testing.causal_te_validation.decision_rules import (
        verdict_test_09_dim_spec,
    )
    verdict = verdict_test_09_dim_spec(evidence)

    return {
        "verdict": verdict,
        "evidence": evidence,
        "csv_paths": [str(csv_path)],
        "figure_paths": [],
    }


__all__ = ["run"]
