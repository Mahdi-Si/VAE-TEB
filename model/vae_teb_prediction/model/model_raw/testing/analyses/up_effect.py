r"""Inference-time UP perturbation analysis for the raw model (S8-T02).

Asks whether the source-pure UP **waveform** actually helps the raw-FHR forecast, without a second
checkpoint, by comparing the true UP against perturbations of the same batch:

* ``normal``: the original raw ``up`` waveform
* ``zero``: the source waveform set to zero
* ``batch_permute``: the source waveform rolled across samples in the batch (a *wrong* UP)
* ``time_shuffle``: the source timesteps globally shuffled within each sample (UP energy kept,
  temporal coupling destroyed)

For each condition it forwards ``model(fhr_raw, up_perturbed, mask)`` and reports the **raw-waveform**
forecast degradation (``compute_raw_forecast_metrics``, no scattering/phase channel split), the KLD
change, residual usage, uplift, and attention concentration. Positive ``forecast_degradation`` means
the true UP lowered the raw forecast error relative to that perturbation; a positive ``kld_sum_drop``
means the true UP moved the posterior more than the perturbation did.

This is the raw analogue of the scattering ``up_effect``: it perturbs ``up_raw`` (not a separate
feature ``u_stream``) and scores ``raw_mse`` (not ``feat_mse_total``). Its ``condition_deltas.csv`` is
consumed by ``causal_te_validation/test_01_up_ablation_stats.py`` (whose metric map targets the
``raw_mse_delta_vs_normal`` column) and by the ``test_03_raw_uplift`` specificity leg.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from model.vae_teb_prediction.model.model_raw.testing.base import TestRunner
from model.vae_teb_prediction.model.model_raw.testing.collectors import (
    _extract_epoch,
    _extract_guid,
    _extract_label,
)
from model.vae_teb_prediction.model.model_raw.testing.metrics import (
    aggregate_te_lag_map,
    compute_attention_diagnostics,
    compute_kld_aggregates_per_sample,
    compute_raw_forecast_metrics,
    compute_residual_usage,
    compute_uplift_metrics,
)
from model.vae_teb_prediction.model.model_raw.testing.visualizers import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    FONT_LABEL,
    FONT_TITLE,
    SAVE_DPI,
    _style_axes,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _time_shuffle_index(length: int, seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic time-axis permutation index of a given length."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(length, generator=generator).to(device)


def _condition_streams(
    up_raw: torch.Tensor, mask: torch.Tensor, *, seed: int
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Return raw source-waveform variants paired with their **matching** validity mask.

    Every perturbation that reorders the source in time or batch reorders its mask the *same* way,
    so the source front end is always masked against the signal it is actually fed. Passing the
    original (unpermuted) mask instead would let gap-encoded 0-samples leak into mask-valid
    positions (and zero out genuine reordered UP), so the "energy-preserving shuffle" would neither
    preserve energy nor stay consistent with the model's own :meth:`permutation_kl` (which permutes
    ``mask[perm]`` with the source). The FHR/target path always keeps the original ``mask``; the
    per-condition mask is handed to the model via ``forward(..., up_mask=...)``.

    Returns a ``{condition: (up_stream, up_mask)}`` mapping; the original tensors are left untouched.
    """
    perm = _time_shuffle_index(up_raw.size(1), seed, up_raw.device)
    streams: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "normal": (up_raw, mask),
        "zero": (torch.zeros_like(up_raw), mask),
        "time_shuffle": (up_raw.index_select(1, perm), mask.index_select(1, perm)),
    }
    if up_raw.size(0) > 1:
        streams["batch_permute"] = (
            torch.roll(up_raw, shifts=1, dims=0),
            torch.roll(mask, shifts=1, dims=0),
        )
    else:
        # With batch size 1 there is no cross-sample permutation available.
        streams["batch_permute"] = streams["time_shuffle"]
    return streams


def _attention_concentration(outputs: Dict[str, torch.Tensor], warmup: int) -> np.ndarray:
    """Per-sample attention concentration, matching collect_metrics."""
    attn = outputs.get("attn_weights")
    if attn is None or attn.dim() != 4:
        batch_size = int(outputs["mu_full"].size(0))
        return np.full(batch_size, np.nan, dtype=float)
    diag = compute_attention_diagnostics(attn, warmup)
    ent = diag["entropy"].detach().cpu().numpy()
    # Warmup rows are entirely NaN -- suppress the expected
    # ``RuntimeWarning: Mean of empty slice`` from ``nanmean``.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        head_mean = np.nanmean(ent, axis=2)
        ent_mean = np.asarray(np.nanmean(head_mean, axis=1))
    L = int(attn.shape[-1])
    norm = math.log(L) if L > 1 else 1.0
    return 1.0 - ent_mean / max(norm, 1e-12)


def _te_lag_total_mass(outputs: Dict[str, torch.Tensor], warmup: int) -> np.ndarray:
    """Per-sample total mass of the lag-resolved TE attribution map."""
    te_lag_map = outputs.get("te_lag_map")
    if te_lag_map is None or te_lag_map.dim() != 3:
        batch_size = int(outputs["mu_full"].size(0))
        return np.full(batch_size, np.nan, dtype=float)
    agg = aggregate_te_lag_map(te_lag_map, warmup)
    return agg["te_lag_mean"].detach().cpu().numpy().sum(axis=-1)


def _plot_delta_boxplot(
    delta_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    *,
    ylabel: str,
    color: str,
) -> None:
    """Small condition-wise boxplot for perturbation deltas."""
    if plt is None or delta_df.empty or metric not in delta_df.columns:
        return
    plot_df = delta_df[delta_df["condition"] != "normal"].copy()
    if plot_df.empty:
        return
    conditions = [c for c in ("zero", "batch_permute", "time_shuffle") if c in set(plot_df["condition"])]
    if not conditions:
        return
    values = [
        plot_df.loc[plot_df["condition"] == cond, metric].dropna().to_numpy(dtype=float)
        for cond in conditions
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bp = ax.boxplot(values, tick_labels=conditions, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(COLOR_BLACK)
    ax.axhline(0.0, color=COLOR_BLACK, lw=0.7, ls="--")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(metric, fontsize=FONT_TITLE, fontweight="normal")
    _style_axes(ax, grid="major", minor_ticks=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def run_up_effect_analysis(
    runner: TestRunner,
    loader: Any,
    max_samples: int = 1000,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run inference-time UP-waveform perturbation tests on the raw model.

    Args:
        runner: Loaded raw :class:`TestRunner`.
        loader: Standard segment-level DataLoader (raw ``fhr``/``up``/``weight`` batches).
        max_samples: Maximum samples to process.
        output_dir: Optional output directory. Defaults to ``runner.ensure_dir("up_effect")``.
        seed: Deterministic seed for the time-shuffle perturbation.

    Returns:
        Summary dict with condition-wise mean deltas.
    """
    if max_samples <= 0:
        logger.info("up_effect: skipped (max_samples <= 0)")
        return {}

    if output_dir is None:
        output_dir = runner.ensure_dir("up_effect")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup, horizon = int(runner.warmup_steps), int(runner.horizon)
    records: List[Dict[str, Any]] = []
    processed = 0

    with runner.inference_mode():
        for batch_idx, batch in enumerate(runner.iter_batches(loader, max_samples)):
            fhr_raw, up_normal, mask = runner._build_raw_input(batch)
            batch_size = int(fhr_raw.size(0))
            x_plus = runner.build_future_target(batch)
            streams = _condition_streams(up_normal, mask, seed=seed + batch_idx)

            for condition, (up_stream, up_mask) in streams.items():
                # FHR/target keeps the original `mask`; the source is masked by its own reordered
                # `up_mask` so the perturbation stays energy-preserving and gap-consistent.
                outputs = runner.model(fhr_raw, up_stream, mask, up_mask=up_mask)
                fcst = compute_raw_forecast_metrics(
                    outputs["mu_full"], x_plus, warmup, horizon
                )
                uplift = compute_uplift_metrics(
                    outputs["mu_full"],
                    outputs["mu_base"],
                    x_plus,
                    warmup,
                    horizon,
                )
                usage = compute_residual_usage(
                    outputs["delta_mu_src"],
                    outputs["mu_full"],
                    warmup,
                    horizon,
                )
                kld = compute_kld_aggregates_per_sample(outputs, warmup)
                concentration = _attention_concentration(outputs, warmup)
                te_mass = _te_lag_total_mass(outputs, warmup)

                for idx in range(batch_size):
                    if max_samples and processed + idx >= max_samples:
                        break
                    records.append({
                        "sample_id": int(processed + idx),
                        "guid": _extract_guid(batch, idx),
                        "epoch": _extract_epoch(batch, idx),
                        "label": _extract_label(batch, idx),
                        "condition": condition,
                        "raw_mse": float(fcst["raw_mse"][idx].cpu().item()),
                        "raw_r2": float(fcst["raw_r2"][idx].cpu().item()),
                        "l_full": float(uplift["l_full"][idx].cpu().item()),
                        "l_base": float(uplift["l_base"][idx].cpu().item()),
                        "uplift_abs": float(uplift["uplift_abs"][idx].cpu().item()),
                        "uplift_rel": float(uplift["uplift_rel"][idx].cpu().item()),
                        "residual_ratio": float(usage["residual_ratio"][idx].cpu().item()),
                        "delta_src_norm": float(usage["delta_norm"][idx].cpu().item()),
                        "kld_mean": float(kld["kld_mean"][idx].cpu().item()),
                        "kld_sum": float(kld["kld_sum"][idx].cpu().item()),
                        "kld_l2": float(kld["kld_l2"][idx].cpu().item()),
                        "attention_concentration_mean": float(concentration[idx]),
                        "te_lag_total_mass": float(te_mass[idx]),
                    })

            processed += batch_size
            if max_samples and processed >= max_samples:
                break

    df = pd.DataFrame(records)
    per_sample_csv = output_dir / "per_sample.csv"
    df.to_csv(per_sample_csv, index=False)
    if df.empty:
        logger.warning("up_effect: no samples collected.")
        return {"n_samples": 0, "csv": str(per_sample_csv)}

    metric_cols = [
        "raw_mse",
        "uplift_rel",
        "residual_ratio",
        "delta_src_norm",
        "kld_mean",
        "kld_sum",
        "kld_l2",
        "attention_concentration_mean",
        "te_lag_total_mass",
    ]
    normal = (
        df[df["condition"] == "normal"]
        .set_index("sample_id")[metric_cols]
        .add_prefix("normal_")
    )
    delta_rows = []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        rec = row.to_dict()
        if sid in normal.index:
            for metric in metric_cols:
                nval = float(normal.loc[sid, f"normal_{metric}"])
                val = float(row[metric])
                rec[f"{metric}_delta_vs_normal"] = val - nval
                rec[f"{metric}_drop_from_normal"] = nval - val
        delta_rows.append(rec)
    delta_df = pd.DataFrame(delta_rows)
    delta_csv = output_dir / "condition_deltas.csv"

    # Positive forecast degradation means the true UP lowered raw error.
    delta_df["forecast_degradation"] = delta_df["raw_mse_delta_vs_normal"]
    delta_df["kld_sum_drop"] = delta_df["kld_sum_drop_from_normal"]
    delta_df["uplift_rel_drop"] = delta_df["uplift_rel_drop_from_normal"]
    delta_df.to_csv(delta_csv, index=False)

    _plot_delta_boxplot(
        delta_df,
        "forecast_degradation",
        output_dir / "forecast_degradation_by_condition.pdf",
        ylabel="raw_mse(condition) - raw_mse(normal)",
        color=COLOR_BLUE,
    )
    _plot_delta_boxplot(
        delta_df,
        "kld_sum_drop",
        output_dir / "kld_sum_drop_by_condition.pdf",
        ylabel="KLD_sum(normal) - KLD_sum(condition)",
        color=COLOR_PURPLE,
    )
    _plot_delta_boxplot(
        delta_df,
        "uplift_rel_drop",
        output_dir / "uplift_rel_drop_by_condition.pdf",
        ylabel="uplift_rel(normal) - uplift_rel(condition)",
        color=COLOR_GREEN,
    )
    _plot_delta_boxplot(
        delta_df,
        "residual_ratio_drop_from_normal",
        output_dir / "residual_ratio_drop_by_condition.pdf",
        ylabel="residual_ratio(normal) - residual_ratio(condition)",
        color=COLOR_ORANGE,
    )

    summary_by_condition: Dict[str, Dict[str, float]] = {}
    for condition, sub in delta_df.groupby("condition"):
        summary_by_condition[str(condition)] = {
            "n": int(len(sub)),
            "raw_mse_mean": float(sub["raw_mse"].mean()),
            "kld_sum_mean": float(sub["kld_sum"].mean()),
            "uplift_rel_mean": float(sub["uplift_rel"].mean()),
            "residual_ratio_mean": float(sub["residual_ratio"].mean()),
            "forecast_degradation_mean": float(sub["forecast_degradation"].mean()),
            "kld_sum_drop_mean": float(sub["kld_sum_drop"].mean()),
            "uplift_rel_drop_mean": float(sub["uplift_rel_drop"].mean()),
        }

    by_class_csv = None
    by_class_rows: List[Dict[str, Any]] = []
    if "label" in delta_df.columns and delta_df["label"].notna().any():
        for (label, condition), sub in delta_df.groupby(["label", "condition"]):
            by_class_rows.append({
                "label": int(label) if pd.notna(label) else None,
                "condition": str(condition),
                "n": int(len(sub)),
                "forecast_degradation_mean": float(sub["forecast_degradation"].mean()),
                "kld_sum_drop_mean": float(sub["kld_sum_drop"].mean()),
                "uplift_rel_drop_mean": float(sub["uplift_rel_drop"].mean()),
                "residual_ratio_drop_mean": float(
                    sub["residual_ratio_drop_from_normal"].mean()
                ),
            })
    if by_class_rows:
        by_class_df = pd.DataFrame(by_class_rows)
        by_class_csv = output_dir / "by_class_summary.csv"
        by_class_df.to_csv(by_class_csv, index=False)

    summary = {
        "n_samples": int(df["sample_id"].nunique()),
        "conditions": summary_by_condition,
        "per_sample_csv": str(per_sample_csv),
        "delta_csv": str(delta_csv),
        "by_class_csv": str(by_class_csv) if by_class_csv is not None else None,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        f"up_effect: n={summary['n_samples']}, "
        f"conditions={', '.join(sorted(summary_by_condition))}"
    )
    return summary
