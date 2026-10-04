r"""``latent_descriptors`` -- which clinical descriptors of the recent raw trace the latent carries linearly (E1-L).

**Question.** Grouped-CV ridge R² from ``mu_prior``, ``mu_post`` and ``delta_mu = mu_post - mu_prior`` to
raw-signal descriptors of the window that ends at the latent's own step.

**Pass.** The collection tables hold no per-step latent, so a seeded draw of scored segments
(cap ``latent_descriptors_segments``, default :data:`DEFAULT_SEGMENTS`) is re-forwarded through
``raw.forward_raw`` (dense, no edit). Rows are every :data:`ANCHOR_STRIDE`-th decoded anchor ``a`` whose
window fits the segment (``a >= W - 1`` tokens); the latents are read at ``a``.

**Descriptors** (causal: window = tokens ``[a - W + 1, a]`` = raw samples ``[16(a + 1 - W), 16(a + 1))``,
``W`` = :data:`WINDOW_S` = 300 s = 75 tokens; FHR in bpm and UP in monitor units via
``raw.SummaryUnits``; an FHR sample is valid under the model's own rule, ``weight >= 1`` and finite;
``m_t`` = mean of token ``t``'s 16 samples, NaN unless all are valid):

* ``fhr_baseline_bpm`` -- median of the window's ``m_t``.
* ``fhr_stv_bpm`` -- mean ``|m_t - m_{t-1}|`` over the window's consecutive valid pairs: the
  Dawes-Redman STV construction with 4 s epochs (D-R uses 3.75 s) and bpm instead of ms.
* ``fhr_ltv_bpm`` -- mean over the window's five 1-min blocks of ``max - min`` of the ``m_t`` in
  the block (D-R's minute range).
* ``decel_depth_bpm`` -- largest ``depth_bpm`` (prominence) of the decelerations
  ``events.detect_decelerations`` finds on the segment whose nadir lies in the window, else 0.
* ``decel_area_bpm_s`` -- ``∫ max(baseline - FHR, 0) dt`` over valid window samples inside a
  detected deceleration span.
  *Caveat:* detection smooths and thresholds over the whole segment, so it is not causal; only the
  samples it reads off (nadir, area) lie inside the window.
* ``fhr_gap_frac`` -- fraction of invalid FHR samples in the window.
* ``up_contractions`` -- ``events.detect_contractions`` peaks in the window (same caveat).
* ``up_mean_tone`` -- mean of the finite UP samples in the window.
* ``up_since_peak_s`` -- seconds from the last contraction peak at or before the window's end to
  that end; NaN (row dropped for this descriptor) when the segment has none so far.

**Ridge.** Per descriptor and latent source: ``GroupKFold`` by recording (``min(5, n_recordings)``
folds), ``StandardScaler`` + ``RidgeCV`` fitted inside each training fold (alpha by RidgeCV's
efficient LOO on that fold; the outer grouped fold keeps the R² honest), R² on the held-out fold,
reported as mean ± sd over folds. A probe, not an attribution: linear, and the descriptors correlate
with each other.

**Outputs** (``latent_descriptors/``): ``latent_descriptors_r2.csv`` + ``.pdf`` (grouped bars),
``latent_descriptors_correlation.csv`` + ``.pdf`` (Pearson r, latent coordinate × descriptor, one panel
per source), ``latent_descriptors_rows.csv`` and ``latent_descriptors_latents.npz`` (the per-anchor rows
and latents, for offline re-fits).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "latent_descriptors"
CAP_NAME = f"{NAME}_segments"
DEFAULT_SEGMENTS = 512
SEED_OFFSET = 41
WINDOW_S = 300.0
LTV_BLOCK_S = 60.0
#: Every 4th decoded anchor (16 s apart): rows 4 s apart over a 300 s window are near-duplicates.
ANCHOR_STRIDE = 4
N_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-2, 4, 13)
SOURCES: Tuple[str, ...] = ("mu_prior", "mu_post", "delta_mu")
#: Descriptor -> axis label, in figure order (FHR, then UP).
DESCRIPTORS: Dict[str, str] = {
    "fhr_baseline_bpm": "baseline",
    "fhr_stv_bpm": "STV",
    "fhr_ltv_bpm": "LTV",
    "decel_depth_bpm": "decel depth",
    "decel_area_bpm_s": "decel area",
    "fhr_gap_frac": "gap frac",
    "up_contractions": "contractions",
    "up_mean_tone": "UP tone",
    "up_since_peak_s": "since peak",
}
SOURCE_COLORS = {"mu_prior": figures.COLOR_BLUE, "mu_post": figures.COLOR_ORANGE, "delta_mu": figures.COLOR_GREEN}

#: ``(headline_name, key)`` pairs registered as ``("latent_descriptors", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("latent_r2_baseline_mu_prior", "r2_fhr_baseline_bpm_mu_prior"),
    ("latent_r2_stv_mu_prior", "r2_fhr_stv_bpm_mu_prior"),
    ("latent_r2_decel_depth_delta_mu", "r2_decel_depth_bpm_delta_mu"),
    ("latent_r2_up_tone_delta_mu", "r2_up_mean_tone_delta_mu"),
)


# =============================================================================
# Descriptors
# =============================================================================
def segment_descriptors(
    fhr_bpm: np.ndarray, fhr_valid: np.ndarray, up: np.ndarray, anchors: np.ndarray, *, r: int
) -> Dict[str, np.ndarray]:
    """Every descriptor of the module docstring at each anchor, for one segment ``(L,)`` at 4 Hz."""
    fs = raw.events.FS_RAW
    w = int(round(WINDOW_S * fs / r))
    block = int(round(LTV_BLOCK_S * fs / r))
    steps = fhr_bpm.size // r
    tokens_valid = fhr_valid.reshape(steps, r).all(1)
    m = np.where(tokens_valid, fhr_bpm.reshape(steps, r).mean(1), np.nan)
    first = anchors - w + 1                                   # sliding-window row of each window
    end_raw, start_raw = r * (anchors + 1), r * (anchors + 1 - w)
    up_valid = np.isfinite(up)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN windows are NaN, by design
        window = sliding_window_view(m, w)[first]
        baseline = np.nanmedian(window, axis=1)
        stv = np.nanmean(sliding_window_view(np.abs(np.diff(m)), w - 1)[first], axis=1)
        minute = sliding_window_view(m, block)
        ranges = np.concatenate([np.full(block - 1, np.nan), np.nanmax(minute, 1) - np.nanmin(minute, 1)])
        ltv = np.nanmean(ranges[anchors[:, None] - block * np.arange(w // block)[None, :]], axis=1)
        up_tokens = np.nanmean(np.where(up_valid, up, np.nan).reshape(steps, r), axis=1)
        tone = np.nanmean(sliding_window_view(up_tokens, w)[first], axis=1)
    gap = 1.0 - sliding_window_view(fhr_valid.reshape(steps, r).mean(1), w)[first].mean(1)

    peaks = np.sort(raw.events.detect_contractions(np.nan_to_num(up), valid=up_valid)["peak_raw"])
    upto = np.searchsorted(peaks, end_raw)                  # peaks strictly before the window's end
    count = upto - np.searchsorted(peaks, start_raw)
    since = np.where(upto > 0, (end_raw - 1 - peaks[np.maximum(upto - 1, 0)]) / fs, np.nan) if peaks.size else np.full(anchors.size, np.nan)

    decels = raw.events.detect_decelerations(np.nan_to_num(fhr_bpm), valid=fhr_valid)
    depth, area = np.zeros(anchors.size), np.zeros(anchors.size)
    if decels["nadir_raw"].size:
        nadir, deep = decels["nadir_raw"], decels["depth_bpm"]
        inside = (nadir[None, :] >= start_raw[:, None]) & (nadir[None, :] < end_raw[:, None])
        depth = np.where(inside, deep[None, :], 0.0).max(1)
        span = np.zeros(fhr_bpm.size, dtype=bool)
        for onset, end in zip(decels["onset_raw"], decels["end_raw"]):
            span[int(onset):int(end) + 1] = True
        span &= fhr_valid
        for i, (lo, hi) in enumerate(zip(start_raw, end_raw)):
            sel = span[lo:hi]
            if sel.any() and np.isfinite(baseline[i]):
                area[i] = np.clip(baseline[i] - fhr_bpm[lo:hi][sel], 0.0, None).sum() / fs
    return {
        "fhr_baseline_bpm": baseline, "fhr_stv_bpm": stv, "fhr_ltv_bpm": ltv,
        "decel_depth_bpm": depth, "decel_area_bpm_s": area, "fhr_gap_frac": gap,
        "up_contractions": count.astype(np.float64), "up_mean_tone": tone, "up_since_peak_s": since,
    }


# =============================================================================
# The pass
# =============================================================================
@torch.no_grad()
def collect_rows(task: Any, loader: Any, rows: pd.DataFrame, units: raw.SummaryUnits) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Forward the drawn segments; return the per-anchor descriptor rows and the latents at those anchors."""
    model = task.orig_model
    r = int(model.raw_per_step)
    w = int(round(WINDOW_S * raw.events.FS_RAW / r))
    frames: List[pd.DataFrame] = []
    latents: Dict[str, List[np.ndarray]] = {"mu_prior": [], "mu_post": []}
    for chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        outputs = raw.forward_raw(model, fhr, up, weight)
        valid = raw.sample_validity(fhr, weight, raw_per_step=r, validity="fhr_weight").cpu().numpy()
        fhr_bpm = (fhr.double() * units.fhr_std + units.fhr_mean).cpu().numpy()
        up_phys = units.up_physical(up.double()).cpu().numpy()
        for b, (_, row) in enumerate(chunk.iterrows()):
            keep = outputs["anchor_valid"][b].bool() & (outputs["anchor_index"][b] >= w - 1)
            anchors = outputs["anchor_index"][b][keep].long()[::ANCHOR_STRIDE]
            if not anchors.numel():
                continue
            a = anchors.cpu().numpy()
            values = segment_descriptors(fhr_bpm[b], valid[b], up_phys[b], a, r=r)
            frame = pd.DataFrame({"guid": str(row["guid"]), "epoch": float(row["epoch"]), "anchor": a, **values})
            for column in labels.GROUP_COLUMNS:
                frame[column] = row.get(column)
            frames.append(frame)
            for name in latents:
                latents[name].append(outputs[name][b].index_select(0, anchors).double().cpu().numpy())
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stacked = {name: np.concatenate(parts) if parts else np.empty((0, model.d_z)) for name, parts in latents.items()}
    stacked["delta_mu"] = stacked["mu_post"] - stacked["mu_prior"]
    return table, stacked


# =============================================================================
# The probes
# =============================================================================
def grouped_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Held-out R² per grouped fold (empty when fewer than two recordings or a constant target)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ok = np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    n_groups = int(np.unique(groups).size)
    if n_groups < 2 or not np.ptp(y) > 0:
        return np.empty(0)
    scores = []
    for train, test in GroupKFold(n_splits=min(N_FOLDS, n_groups)).split(x, y, groups):
        if not (np.ptp(y[train]) > 0 and np.ptp(y[test]) > 0):
            scores.append(np.nan)  # R² is undefined on a constant held-out fold
            continue
        fit = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS)).fit(x[train], y[train])
        scores.append(r2_score(y[test], fit.predict(x[test])))
    return np.asarray(scores, dtype=np.float64)


def correlations(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r ``(d_z, n_descriptors)`` over the rows where each descriptor is finite; NaN where undefined."""
    out = np.full((x.shape[1], y.shape[1]), np.nan)
    for k in range(y.shape[1]):
        ok = np.isfinite(y[:, k])
        if ok.sum() < 3:
            continue
        xs = x[ok] - x[ok].mean(0)
        ys = y[ok, k] - y[ok, k].mean()
        denom = np.sqrt((xs ** 2).sum(0) * (ys ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k] = np.where(denom > 0, xs.T @ ys / denom, np.nan)
    return out


# =============================================================================
# Figures
# =============================================================================
def r2_figure(r2: pd.DataFrame) -> Any:
    """Grouped bars: R² (mean ± sd over folds) per descriptor, one bar per latent source."""
    fig, axes = figures.new_figure(1, 1, height_per_row=2.6)
    ax = axes[0, 0]
    names = list(DESCRIPTORS)
    width = 0.8 / len(SOURCES)
    lows = []
    for i, source in enumerate(SOURCES):
        cell = r2[r2["source"] == source].set_index("descriptor").reindex(names)
        mean, sd = cell["r2_mean"].to_numpy(float), cell["r2_sd"].to_numpy(float)
        ax.bar(np.arange(len(names)) + (i - 1) * width, mean, width, yerr=np.nan_to_num(sd),
               color=SOURCE_COLORS[source], label=source, error_kw={"linewidth": figures.LINE_THIN})
        lows.append(np.nanmin(mean - np.nan_to_num(sd)) if np.isfinite(mean).any() else 0.0)
    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
    ax.set_ylim(max(min(lows + [0.0]) - 0.05, -1.0), 1.0)
    ax.set_xticks(np.arange(len(names)), [DESCRIPTORS[n] for n in names], rotation=30, ha="right")
    ax.set_ylabel("held-out R² (grouped by recording)")
    ax.set_title(f"Linear read-out of {WINDOW_S:.0f} s raw descriptors from the latent at the step")
    figures.legend_with_headroom(ax, ncol=3)
    figures.style_axes(ax)
    figures.caveat_note(fig, "bars below R² = -1 are clipped; a probe, not an attribution (descriptors correlate)")
    return fig


def correlation_figure(corr: Dict[str, np.ndarray]) -> Any:
    """Pearson r, latent coordinate × descriptor, one panel per source on one shared [-1, 1] scale."""
    fig, axes = figures.new_figure(1, len(SOURCES), height_per_row=3.4)
    names = list(DESCRIPTORS)
    for ax, source in zip(axes[0], SOURCES):
        figures.heatmap_with_colorbar(
            fig, ax, corr[source], title=source, xlabel="", ylabel="latent coordinate",
            vlimits=(-1.0, 1.0), colorbar_label="Pearson r",
        )
        ax.set_xticks(np.arange(len(names)), [DESCRIPTORS[n] for n in names], rotation=60, ha="right")
    return fig


# =============================================================================
# The registry entry point
# =============================================================================
def run_latent_descriptors_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Re-forward a capped draw, compute the window descriptors, fit the grouped ridge probes, draw both figures."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(NAME, "per-step latents are on no table; this needs the model and the loader (offline re-run)")
    cap = int((eval_config.get("caps") or {}).get(CAP_NAME) or DEFAULT_SEGMENTS)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    rows = raw.draw_segments(context, cap=cap, seed=seed, max_hours=eval_config.get("max_hours_before_delivery"))
    if rows.empty:
        return skip_record(NAME, "no scored segment resolves to a dataset row")

    started = time.perf_counter()
    units = raw.SummaryUnits.from_model(task.orig_model, context.config, loader)
    table, latents = collect_rows(task, loader, rows, units)
    if table.empty:
        return skip_record(NAME, f"no anchor has a full {WINDOW_S:.0f} s window")
    groups = table["guid"].to_numpy()
    y = table[list(DESCRIPTORS)].to_numpy(np.float64)

    records, headline = [], {}
    for k, descriptor in enumerate(DESCRIPTORS):
        for source in SOURCES:
            folds = grouped_r2(latents[source], y[:, k], groups)
            finite = folds[np.isfinite(folds)]
            mean = float(finite.mean()) if finite.size else np.nan
            records.append({
                "descriptor": descriptor, "source": source, "r2_mean": mean,
                "r2_sd": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                "n_folds": int(finite.size), "n_rows": int(np.isfinite(y[:, k]).sum()),
                "n_recordings": int(pd.unique(groups[np.isfinite(y[:, k])]).size),
            })
            headline[f"r2_{descriptor}_{source}"] = raw.finite_or_none(mean)
    r2 = pd.DataFrame(records)
    corr = {source: correlations(latents[source], y) for source in SOURCES}
    elapsed_s = time.perf_counter() - started

    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    r2.to_csv(directory / f"{NAME}_r2.csv", index=False)
    pd.concat([
        pd.DataFrame(corr[s], columns=list(DESCRIPTORS)).rename_axis("coordinate").reset_index().assign(source=s)
        for s in SOURCES
    ]).to_csv(directory / f"{NAME}_correlation.csv", index=False)
    table.to_csv(directory / f"{NAME}_rows.csv", index=False)
    np.savez_compressed(directory / f"{NAME}_latents.npz", mu_prior=latents["mu_prior"], mu_post=latents["mu_post"])
    files = [
        f"{NAME}_r2.csv", f"{NAME}_correlation.csv", f"{NAME}_rows.csv", f"{NAME}_latents.npz",
        figures.render_figure(r2_figure(r2), directory / f"{NAME}_r2").name,
        figures.render_figure(correlation_figure(corr), directory / f"{NAME}_correlation").name,
    ]
    n_recordings = int(table["guid"].nunique())
    logger.info(
        f"{NAME}: {len(rows)} segment(s), {n_recordings} recording(s), {len(table)} anchor row(s), "
        f"{elapsed_s:.1f} s; R² baseline<-mu_prior {headline['r2_fhr_baseline_bpm_mu_prior']}"
    )
    classes = rows[labels.CLASS_COLUMN].value_counts().to_dict() if labels.CLASS_COLUMN in rows.columns else {}
    return {
        "n_samples": int(len(rows)),
        "composition": {"n_recordings": n_recordings, "by_class": {str(k): int(v) for k, v in classes.items()}},
        "plan": {"capped": True, "cap": cap, "seed": seed, "window_s": WINDOW_S, "folds": N_FOLDS,
                 "ridge_alphas": [float(a) for a in RIDGE_ALPHAS], "n_rows": int(len(table)),
                 "weak_cv": n_recordings < N_FOLDS},
        "cost": {"elapsed_s": float(elapsed_s), "seconds_per_segment": float(elapsed_s / max(1, len(rows)))},
        "r2": {
            descriptor: {
                rec["source"]: {"mean": raw.finite_or_none(rec["r2_mean"]), "sd": raw.finite_or_none(rec["r2_sd"]),
                                "n_folds": rec["n_folds"]}
                for rec in records if rec["descriptor"] == descriptor
            }
            for descriptor in DESCRIPTORS
        },
        "headline": {key: headline.get(key) for _, key in HEADLINE},
        "files": files,
    }
