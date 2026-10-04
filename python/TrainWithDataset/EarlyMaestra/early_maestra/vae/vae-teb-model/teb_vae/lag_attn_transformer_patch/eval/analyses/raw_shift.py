r"""``raw_shift`` -- does the lag readout move with the UP content? (E1-I).

**Question.** Shift raw UP against FHR and re-patchify. A model whose lag reads *content* moves its
lag centroid one-for-one with whole-patch shifts; one that attends at fixed lags does not move.
Sub-patch shifts test within-patch timing sensitivity.

**Shift convention.** ``s`` samples is the lag displacement of the UP content:
$u'(n) = u(n + s)$, so every UP event sits $s/16$ tokens **further back** at a fixed anchor and a
content tracker reads $\Delta\bar\ell = s/16$ tokens (the identity line, both axes in seconds). The
vacated samples are the segment's resting tone (``raw.resting_tone``), never zeros. ``s > 0`` hands
the model ``s`` samples of future UP: a non-causal probe, not a forecast. Shifts: ±1..15 samples
(sub-patch, 0.25-3.75 s) and ±1..10 patches (re-patchified after the shift).

**Readouts** per segment, shifted minus aligned, mean-decoded as ``time_shift`` does: ΔK (mean
``kld_per_t`` over the KL support), Δ``pred_gap`` (base minus full block nats per anchor; the base
does not move), and the shift of the attention centroid (per head and head-mean) and of the
``source_kl_lag_map`` centroid and argmax, each read off the profile averaged over the anchors with
the whole lag window in the segment. Reduced per recording; the spread is the std over recordings.
``fits`` regresses the centroid shift on the applied shift over per-recording points: slope, R².

**Cost.** Segments × (1 + 50) dense forwards. Cap: ``caps.raw_shift_segments`` (32, so 1632
segment-forwards at the default).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.attributions import symlog_legend
from teb_vae.lag_attn_cfs.eval.metrics import anchor_support
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "raw_shift"

#: ``(headline_name, key)`` pairs registered as ``("raw_shift", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("raw_shift_attention_centroid_slope", "attention_centroid_slope"),
    ("raw_shift_attention_centroid_r2", "attention_centroid_r2"),
    ("raw_shift_klmap_centroid_slope", "klmap_centroid_slope"),
)

CAP_SEGMENTS, DEFAULT_SEGMENTS = "raw_shift_segments", 32
SEED_OFFSET = 43
SAMPLES_PER_SECOND = 4.0
SUB_SHIFTS = tuple(s for s in range(-15, 16) if s)
PATCH_SHIFTS = tuple(16 * p for p in range(-10, 11) if p)
CAVEAT = "s > 0 moves UP content back in lag and shows the model s samples of future UP: a probe, not a forecast."


def shift_up(up: torch.Tensor, s: int, tone: torch.Tensor) -> torch.Tensor:
    """``u'(n) = u(n + s)``, the vacated end filled with the resting tone ``(B,)``."""
    if s == 0:
        return up
    pad = tone[:, None].expand(-1, abs(s))
    return torch.cat((up[:, s:], pad), dim=1) if s > 0 else torch.cat((pad, up[:, :s]), dim=1)


def profile_stats(profile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centroid and argmax (lags) of non-negative profiles ``(..., L)``."""
    lags = torch.arange(profile.shape[-1], device=profile.device, dtype=profile.dtype)
    centroid = (profile * lags).sum(-1) / profile.sum(-1).clamp_min(1e-12)
    return centroid, profile.argmax(-1).to(profile.dtype)


@torch.no_grad()
def score_batch(model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor, *,
                likelihood: str, shifts: Tuple[int, ...]) -> Dict[int, Dict[str, np.ndarray]]:
    """Every shift of one batch: per-segment Δ of KL, gap, attention and KL-map centroids (lags)."""
    r = int(model.raw_per_step)
    tone = raw.resting_tone(up, weight, raw_per_step=r, validity=model.source_validity)
    aligned = raw.forward_raw(model, fhr, up, weight)
    anchors = aligned["anchor_index"]
    target = raw.forecast_target(model, fhr, weight, anchors)
    mask, _coverage, kl_support = anchor_support(model, weight, aligned)
    contributing = (mask.amax(dim=-1) > 0).to(target.dtype)
    n_lags = aligned["attn_weights"].shape[-1]
    window = ((anchors >= n_lags - 1) & aligned["anchor_valid"]).to(target.dtype)  # whole lag window in the segment

    def block(latent: torch.Tensor) -> torch.Tensor:
        return raw.block_nll(model, *raw.decode_at(model, aligned, latent), target, mask, likelihood)

    def mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.view(*weights.shape, *([1] * (values.dim() - weights.dim())))
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    base = block(aligned["mu_prior"])

    def read(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        attn = mean(out["attn_weights"].gather(1, anchors[..., None, None].expand(-1, -1, *out["attn_weights"].shape[2:])), window)
        klmap = mean(out["source_kl_lag_map"].gather(1, anchors[..., None].expand(-1, -1, n_lags)), window)
        heads, _ = profile_stats(attn)                       # (B, M)
        pooled, _ = profile_stats(attn.mean(1))              # (B,)
        kl_centroid, kl_argmax = profile_stats(klmap)
        return {"kld": mean(out["kld_per_t"], kl_support), "gap": mean(base - block(out["mu_post"]), contributing),
                "attn_centroid": pooled, "attn_centroid_heads": heads,
                "klmap_centroid": kl_centroid, "klmap_argmax": kl_argmax}

    reference = read(aligned)
    deltas: Dict[int, Dict[str, np.ndarray]] = {}
    for s in shifts:
        moved = read(raw.forward_raw(model, fhr, shift_up(up, s, tone), weight))
        deltas[s] = {key: (moved[key] - reference[key]).double().cpu().numpy() for key in reference}
    return deltas


def fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    """Least-squares slope, intercept and R² of ``y`` on ``x`` over finite points."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return {"slope": None, "intercept": None, "r2": None, "n_points": int(ok.sum())}
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    residual = y[ok] - (slope * x[ok] + intercept)
    total = ((y[ok] - y[ok].mean()) ** 2).sum()
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1.0 - (residual ** 2).sum() / total) if total > 0 else None, "n_points": int(ok.sum())}


def build_figure(summary: pd.DataFrame, heads: int) -> Any:
    """ΔK, Δgap and the centroid shift against the applied shift; sub-patch left, whole-patch right."""
    figure, axes = figures.new_figure(3, 2, height_per_row=2.3)
    palette = (figures.COLOR_BLUE, figures.COLOR_ORANGE, figures.COLOR_GREEN, figures.COLOR_PURPLE)
    for col, kind in enumerate(("sub", "patch")):
        part = summary[summary["kind"] == kind].sort_values("shift_s")
        x = part["shift_s"].to_numpy()
        for row, (key, ylabel) in enumerate((("delta_kld", "ΔK (nats per step)"), ("delta_gap", "Δ pred_gap (nats per anchor)"))):
            ax = axes[row, col]
            m, s = part[f"{key}_mean"].to_numpy(), part[f"{key}_std"].to_numpy()
            ax.plot(x, m, color=figures.COLOR_BLUE, marker="o", markersize=figures.MARKER_SMALL,
                    linewidth=figures.LINE_REGULAR, label="mean over recordings")
            ax.fill_between(x, m - s, m + s, color=figures.COLOR_BLUE, alpha=0.15, linewidth=0, label="± std")
            ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
            ax.set(ylabel=ylabel, title=f"{'Sub-patch' if kind == 'sub' else 'Whole-patch'} shifts")
            symlog_legend(ax, m - s, m + s, legend=row == 0 and col == 0)
        ax = axes[2, col]
        ax.plot(x, x, color=figures.COLOR_BLACK, linestyle="--", linewidth=figures.LINE_THIN, label="identity (content tracking)")
        for m in range(heads):
            ax.plot(x, part[f"delta_centroid_attn_h{m}_mean"], color=palette[m % len(palette)],
                    linewidth=figures.LINE_THIN, label=f"attention head {m}")
        ax.plot(x, part["delta_centroid_attn_mean"], color=figures.COLOR_GRAY, linewidth=figures.LINE_EMPHASIS,
                marker="o", markersize=figures.MARKER_SMALL, label="attention, head mean")
        ax.plot(x, part["delta_centroid_klmap_mean"], color=figures.COLOR_VERMILLION, linewidth=figures.LINE_EMPHASIS,
                label="KL lag map centroid")
        ax.plot(x, part["delta_argmax_klmap_mean"], color=figures.COLOR_VERMILLION, linestyle=":",
                linewidth=figures.LINE_REGULAR, label="KL lag map argmax")
        ax.set(xlabel="applied lag shift s/4 (s)", ylabel="centroid shift (s)", title="Lag readout vs applied shift")
        if col == 1:
            for ax_ in axes[:, 1]:
                ax_.set_xticks(np.arange(-40, 41, 8))
                ax_.set_xticks(np.arange(-40, 41, 4), minor=True)
        else:
            figures.legend_with_headroom(ax, ncol=2)
        for ax_ in axes[:, col]:
            ax_.set_xlabel("applied lag shift s/4 (s)")
            figures.style_axes(ax_)
    figures.caveat_note(figure, CAVEAT)
    return figure


def run_raw_shift_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shift, score, reduce over recordings, fit the centroid against the shift, draw."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(NAME, "the shift re-runs the model on edited raw UP; an offline re-run has no model")
    cap = int((eval_config.get("caps") or {}).get(CAP_SEGMENTS) or DEFAULT_SEGMENTS)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    rows = raw.draw_segments(context, cap=cap, seed=seed)
    if rows.empty:
        return skip_record(NAME, "no scored segment could be located in the loader's dataset")

    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    shifts = SUB_SHIFTS + PATCH_SHIFTS
    pieces: List[pd.DataFrame] = []
    started = time.perf_counter()
    for chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        for s, delta in score_batch(model, fhr, up, weight, likelihood=likelihood, shifts=shifts).items():
            frame = chunk[[c for c in ("sample_index", "guid", labels.CLASS_COLUMN) if c in chunk.columns]].copy()
            frame["shift_samples"], frame["kind"] = s, "sub" if abs(s) < 16 else "patch"
            frame["delta_kld"], frame["delta_gap"] = delta["kld"], delta["gap"]
            # Lags → seconds: one token is 4 s.
            frame["delta_centroid_attn"] = 4.0 * delta["attn_centroid"]
            for m in range(delta["attn_centroid_heads"].shape[1]):
                frame[f"delta_centroid_attn_h{m}"] = 4.0 * delta["attn_centroid_heads"][:, m]
            frame["delta_centroid_klmap"] = 4.0 * delta["klmap_centroid"]
            frame["delta_argmax_klmap"] = 4.0 * delta["klmap_argmax"]
            pieces.append(frame)
    elapsed_s = time.perf_counter() - started

    per_segment = pd.concat(pieces, ignore_index=True)
    per_segment["shift_s"] = per_segment["shift_samples"] / SAMPLES_PER_SECOND
    measured = [c for c in per_segment.columns if c.startswith("delta_")]
    heads = sum(c.startswith("delta_centroid_attn_h") for c in measured)
    per_recording = per_segment.groupby(["shift_samples", "kind", "shift_s", "guid"], as_index=False)[measured].mean()
    grouped = per_recording.groupby(["shift_samples", "kind", "shift_s"])[measured]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std")).reset_index()
    summary["n_recordings"] = per_recording.groupby("shift_samples")["guid"].nunique().to_numpy()

    fits: Dict[str, Dict[str, Any]] = {}
    for kind in ("sub", "patch"):
        part = per_recording[per_recording["kind"] == kind]
        x = part["shift_s"].to_numpy(dtype=np.float64)
        fits[kind] = {c.removeprefix("delta_"): fit(x, part[c].to_numpy(dtype=np.float64))
                      for c in measured if c.startswith(("delta_centroid", "delta_argmax"))}

    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    per_segment.to_csv(directory / "raw_shift_per_segment.csv", index=False)
    summary.to_csv(directory / "raw_shift_summary.csv", index=False)
    figure = figures.render_figure(build_figure(summary, heads), directory / "raw_shift")
    patch = fits["patch"]
    return {
        "n_samples": int(rows.shape[0]),
        "composition": {"n_recordings": int(per_segment["guid"].nunique())},
        "plan": {"capped": True, "cap": cap, "seed": seed, "shifts_samples": list(shifts),
                 "fill": "resting tone", "decode": "latent mean, both arms",
                 "convention": "u'(n) = u(n + s); a content tracker moves its centroid by +s/4 s"},
        "cost": {"elapsed_s": float(elapsed_s), "segment_forwards": int(rows.shape[0] * (1 + len(shifts)))},
        "fits": fits,
        "summary": summary.to_dict(orient="records"),
        "headline": {
            "attention_centroid_slope": patch["centroid_attn"]["slope"],
            "attention_centroid_r2": patch["centroid_attn"]["r2"],
            "klmap_centroid_slope": patch["centroid_klmap"]["slope"],
        },
        "files": ["raw_shift_per_segment.csv", "raw_shift_summary.csv", str(figure.name)],
    }
