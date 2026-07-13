"""Encoder-feature classifier probe for lag-attn v1.

Computes three time-averaged feature vectors per sample from the
encoder/latent outputs and evaluates how well each discriminates
between outcome classes via linear probes and clustering metrics from
:mod:`analyses.class_separation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.analyses.class_separation import (
    compute_cluster_quality_metrics,
    compute_linear_separability,
)
from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    FONT_LABEL,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _time_average(tensor: torch.Tensor, warmup: int) -> torch.Tensor:
    """Return per-sample time mean over valid anchors ``[warmup, T)``.

    Args:
        tensor: Shape ``(B, T, D)``.
        warmup: Number of initial anchors to drop.

    Returns:
        Shape ``(B, D)``.
    """
    T = int(tensor.size(1))
    warmup = max(0, min(int(warmup), T))
    if warmup >= T:
        return tensor.new_zeros(tensor.size(0), tensor.size(-1))
    return tensor[:, warmup:, :].mean(dim=1)


def run_encoder_probe(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 2000,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate class separability of encoder features.

    Collects three feature variants per sample:

    - ``z_mean`` — time-average of ``outputs["z"]`` (latent, ``d_z`` dim)
    - ``target_state_mean`` — time-average of ``outputs["target_state"]``
    - ``attended_source_mean`` — time-average of
      ``outputs["attended_source"]``

    For each variant it runs linear separability (logistic + LDA CV) and
    clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).

    Args:
        runner: Loaded :class:`TestRunner`.
        loader: PyTorch DataLoader.
        max_samples: Maximum samples to process.
        output_dir: Optional override (defaults to
            ``runner.ensure_dir("encoder_probe")``).

    Returns:
        Dict keyed by feature name with each sub-dict containing the
        linear and clustering metrics.
    """
    if max_samples <= 0:
        logger.info("encoder_probe: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("encoder_probe")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_rows: list = []
    tgt_rows: list = []
    att_rows: list = []
    meta_rows: list = []

    warmup = int(runner.warmup_steps)
    processed = 0
    with runner.inference_mode():
        for batch in runner.iter_batches(loader, max_samples):
            outputs = runner.forward(batch)
            z_mean = _time_average(outputs["z"], warmup)                   # (B, d_z)
            tgt_mean = _time_average(outputs["target_state"], warmup)      # (B, 128)
            att_mean = _time_average(outputs["attended_source"], warmup)   # (B, 128)

            batch_size = int(z_mean.size(0))
            z_np = z_mean.cpu().numpy()
            tgt_np = tgt_mean.cpu().numpy()
            att_np = att_mean.cpu().numpy()

            for idx in range(batch_size):
                if max_samples and processed >= max_samples:
                    break
                z_rows.append(z_np[idx])
                tgt_rows.append(tgt_np[idx])
                att_rows.append(att_np[idx])
                meta_rows.append({
                    "guid": _extract_guid(batch, idx),
                    "epoch": _extract_epoch(batch, idx),
                    "label": _extract_label(batch, idx),
                })
                processed += 1
            if max_samples and processed >= max_samples:
                break

    if not meta_rows:
        logger.warning("encoder_probe: no samples collected.")
        return {"n_samples": 0}

    meta_df = pd.DataFrame(meta_rows)
    valid_label_mask = meta_df["label"].notna() & (meta_df["label"] != 0)
    if not bool(valid_label_mask.any()):
        logger.warning("encoder_probe: no labelled samples; probes will be skipped.")
        return {"n_samples": int(len(meta_rows))}

    features = {
        "z_mean": np.asarray(z_rows),
        "target_state_mean": np.asarray(tgt_rows),
        "attended_source_mean": np.asarray(att_rows),
    }

    label_arr = meta_df["label"].fillna(0).astype(int).to_numpy()
    mask_np = valid_label_mask.to_numpy()

    # Dump the wide feature matrix for drill-down analyses.
    out_cols = {"guid": meta_df["guid"], "epoch": meta_df["epoch"], "label": meta_df["label"]}
    for name, mat in features.items():
        for d in range(mat.shape[1]):
            out_cols[f"{name}_{d}"] = mat[:, d]
    pd.DataFrame(out_cols).to_csv(output_dir / "feature_matrix.csv", index=False)

    results: Dict[str, Dict[str, float]] = {}
    probe_rows = []
    for name, mat in features.items():
        X = mat[mask_np]
        y = label_arr[mask_np]
        if X.shape[0] < 20 or len(np.unique(y)) < 2:
            logger.info(f"encoder_probe: not enough data for {name}; skipping.")
            continue
        try:
            ls = compute_linear_separability(X, y)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"encoder_probe: linear separability failed for {name}: {exc}")
            ls = {}
        try:
            cq = compute_cluster_quality_metrics(X, y)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"encoder_probe: cluster metrics failed for {name}: {exc}")
            cq = {}
        merged = {**ls, **cq}
        results[name] = merged
        probe_rows.append({"feature": name, **merged})

    if probe_rows:
        probe_df = pd.DataFrame(probe_rows)
        probe_df.to_csv(output_dir / "probe_results.csv", index=False)

        if plt is not None and "logreg_acc_mean" in probe_df.columns:
            fig, ax = plt.subplots(figsize=(5.4, 3.4))
            x = np.arange(len(probe_df))
            width = 0.35
            ax.bar(x - width / 2, probe_df["logreg_acc_mean"], width=width, color=COLOR_BLUE, label="logreg")
            if "lda_acc_mean" in probe_df.columns:
                ax.bar(x + width / 2, probe_df["lda_acc_mean"], width=width, color=COLOR_GREEN, label="lda")
            if "silhouette" in probe_df.columns:
                ax2 = ax.twinx()
                ax2.plot(x, probe_df["silhouette"], color=COLOR_ORANGE, marker="o", label="silhouette")
                ax2.set_ylabel("silhouette", fontsize=FONT_LABEL)
            ax.set_xticks(x)
            ax.set_xticklabels(probe_df["feature"], rotation=10)
            ax.set_ylabel("linear-probe accuracy", fontsize=FONT_LABEL)
            ax.set_title("Encoder probe class separability", fontsize=FONT_TITLE, fontweight="normal")
            ax.legend(loc="upper left", frameon=True)
            _style_axes(ax, grid="major", minor_ticks=False)
            fig.tight_layout()
            fig.savefig(output_dir / "encoder_probe_separation_bar.pdf", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)

    logger.info(f"encoder_probe: evaluated features {list(results.keys())}")
    return {"n_samples": int(len(meta_rows)), "features": results}
