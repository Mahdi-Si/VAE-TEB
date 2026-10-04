r"""``decelerations`` -- deceleration-centred skill, causal-contraction attention, implied delay and
dose-response (plan §2, E1-E; EVAL_MAP_LAG N3/N5, EVAL_MAP_FORECAST P2).

**Detection.** Contractions on raw UP (``events.detect_contractions``) and decelerations on raw FHR in
bpm (``events.detect_decelerations``, the D7 port; loader inverse from ``traces.raw_signal_scales``),
both ``weight``-masked as the collection pass does.

**Pairing.** Each contraction claims the deepest nadir in ``(peak, peak + W]`` (``W`` =
``caps.decelerations_pair_window_s``, default 120 s); a nadir claimed twice keeps the later
contraction. The measured delay is nadir − peak. Every nadir gets a context: ``paired``, ``preceded``
(a peak within ``W`` before it, but it was not that contraction's deepest) or ``unpreceded``. This is
contraction-centred rather than "latest peak before the nadir": with contractions every 2-4 min and a
delay near that spacing (the planted 180 s), the latest-peak rule pairs most nadirs with the wrong one.

**Readouts** (one mean-decoded dense forward per segment; per recording, then over recordings):

* *Skill around the nadir:* anchors whose horizon contains the nadir token, ``t ∈ [n − H, n − 1]``;
  level RMSE (bpm) of base, full and persistence on the steps within ±2 tokens of the nadir, and
  ``pred_gap``, against seconds to the nadir, per context.
* *Attention on the causal contraction (paired):* at those anchors, the head-mean attention (and
  ``source_kl_lag_map``) on lags ``|ℓ − (t − tok(peak))| ≤ 1`` per lag, over the mean per lag
  elsewhere in the support (enrichment; 1 = none). Control: the same with the recording's measured
  delays permuted across its paired events.
* *Implied delay:* ``d̂ = 4(ℓ̄ + 1 + τ_n)`` averaged over the window, ``ℓ̄`` the attention (or positive
  lag-map) centroid, ``τ_n`` the nadir's horizon step; Spearman ρ against the measured delay, pooled and
  per recording. A fixed-lag reader gives a constant ``d̂``; a contraction tracker gives ``d̂ ≈`` delay.
* *Dose-response (every contraction):* over anchors ``k ∈ [0, L − 1]`` tokens after the peak, ``K_t``,
  ``pred_gap``, attention and lag-map mass on the contraction's own tokens ``[onset, end]``, against
  amplitude above resting tone (UP units) and duration; per-recording OLS slopes, Wilcoxon against 0,
  class tests when there are classes.
* *Contraction-triggered FHR:* bpm relative to the 10 s before each peak; its minimum is the
  population's measured delay without any pairing rule.

Caps: ``decelerations_segments`` (256), ``decelerations_pair_window_s`` (120). Production cost: 256
dense forwards (about a minute on one GPU); the rest is numpy.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats

from teb_vae.lag_attn_cfs.eval import class_contrast
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.attributions import (
    EXAMPLE_PAGE_WIDTH,
    EXAMPLE_ROW_INCHES,
    UNREAD_COLOUR,
    symlog_legend,
    unsigned_log_norm,
)
from teb_vae.lag_attn_cfs.eval.metrics import anchor_support
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "decelerations"
DEFAULT_SEGMENTS, DEFAULT_PAIR_WINDOW_S = 256, 120
NADIR_HALF_WIDTH = 2               # horizon steps either side of the nadir scored for the level RMSE
BAND_HALF_WIDTH = 1                # lags either side of the contraction peak's lag
PRE_PEAK_S, TRIGGER_BIN_S = 10.0, 1.0
SEED_OFFSET = 43
CONTEXTS = ("paired", "preceded", "unpreceded")
DOSE_READOUTS = ("kld", "gap", "attn_mass", "lag_map_mass")
DOSE_COVARIATES = ("amplitude", "duration_s")

#: ``(headline_name, key)``, registered as ``("decelerations", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("decelerations_measured_delay_median_s", "measured_delay_median_s"),
    ("decelerations_triggered_fhr_min_s", "triggered_fhr_min_s"),
    ("decelerations_implied_delay_spearman", "implied_delay_spearman"),
    ("decelerations_implied_delay_attn_median_s", "implied_delay_attn_median_s"),
    ("decelerations_attention_enrichment", "attention_enrichment"),
    ("decelerations_attention_enrichment_control", "attention_enrichment_control"),
)


# =============================================================================
# The pass (mean-decoded through ``raw.latent_means``)
# =============================================================================
@torch.no_grad()
def dense_pass(model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor, likelihood: str) -> Dict[str, np.ndarray]:
    """One mean-decoded dense forward; level errors (standardized) for base, full and persistence."""
    with raw.latent_means(model):
        out = raw.forward_raw(model, fhr, up, weight)
    anchors = out["anchor_index"]
    summaries = model.summary_target(fhr, weight)
    target = model._build_forecast_target(summaries, anchors)
    mask, _coverage, _kl = anchor_support(model, weight, out)
    nll = {b: raw.block_nll(model, out[f"mu_{b}"], out[f"logvar_{b}"], target, mask, likelihood) for b in ("base", "full")}
    scored = mask > 0
    persistence = summaries.gather(1, anchors.long()[..., None].expand(-1, -1, 2))[..., 0]          # (B, A)
    nan = torch.tensor(float("nan"), device=target.device)
    host = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
    errors = {b: out[f"mu_{b}"][..., 0] - target[..., 0] for b in ("base", "full")}
    errors["persistence"] = persistence[..., None] - target[..., 0]
    return {
        "anchors": host(anchors[0]).astype(np.int64),
        "contributing": host(scored.any(dim=-1)).astype(bool),
        "attn": host(out["attn_weights"].mean(dim=2)),                                              # (B, T, L)
        "attn_heads": host(out["attn_weights"]),                                                    # (B, T, M, L)
        "lag_map": host(out["source_kl_lag_map"]),
        "kld": host(out["kld_per_t"]),
        "gap": host(nll["base"] - nll["full"]),
        **{f"err_{b}": host(torch.where(scored, e, nan)) for b, e in errors.items()},
    }


# =============================================================================
# Events
# =============================================================================
def pair_events(peaks: np.ndarray, nadirs: np.ndarray, depths: np.ndarray, window: int) -> np.ndarray:
    """Owner contraction per nadir (-1 = none): each peak claims its deepest nadir in ``(peak, peak + window]``.

    Peaks are visited in time order, so a nadir claimed twice keeps the later contraction.
    ponytail: greedy; the loser does not re-claim its second-deepest nadir.
    """
    owner = np.full(nadirs.size, -1, dtype=np.int64)
    for j in np.argsort(peaks):
        candidates = np.flatnonzero((nadirs > peaks[j]) & (nadirs <= peaks[j] + window))
        if candidates.size:
            owner[candidates[np.argmax(depths[candidates])]] = j
    return owner


def band_enrichment(profile: np.ndarray, centre: int, support: int, half_width: int = BAND_HALF_WIDTH) -> float:
    """Mean per lag on ``|ℓ − centre| ≤ w`` over the mean per lag elsewhere in ``ℓ ≤ support``."""
    lag = np.arange(profile.size)
    inside = lag <= support
    band = inside & (np.abs(lag - centre) <= half_width)
    rest = inside & ~band
    if not band.any() or not rest.any():
        return float("nan")
    denominator = profile[rest].mean()
    return float(profile[band].mean() / denominator) if denominator > 0 else float("nan")


def centroid(profile: np.ndarray, support: int) -> float:
    """Lag centroid of the non-negative part of ``profile`` over ``ℓ ≤ support``."""
    mass = np.clip(profile[: support + 1], 0.0, None)
    return float((np.arange(mass.size) * mass).sum() / mass.sum()) if mass.sum() > 0 else float("nan")


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of ``y`` on ``x`` over finite pairs; NaN below 3 pairs or with no spread."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


# =============================================================================
# Figures
# =============================================================================
def _band(ax: Any, x: np.ndarray, curves: np.ndarray, *, seed: int, colour: str, label: str) -> np.ndarray:
    mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
    ax.plot(x, mean, color=colour, linewidth=figures.LINE_REGULAR, label=label)
    if np.isfinite(lo).any():
        ax.fill_between(x, lo, hi, color=colour, alpha=0.2, linewidth=0)
    return mean


def build_summary_figure(out: Dict[str, Any], *, seed: int) -> Any:
    """Nine panels: measured delay (two ways), implied vs measured, enrichment, skill, gap, dose-response."""
    figure, axes = figures.new_figure(3, 3, height_per_row=2.4)
    colours = {"base": figures.COLOR_BLUE, "full": figures.COLOR_VERMILLION, "persistence": figures.COLOR_GRAY}
    ax = axes[0, 0]
    x, curves = out["trigger_axis"], out["trigger_curves"]
    mean = _band(ax, x, curves, seed=seed, colour=figures.COLOR_BLACK, label="FHR")
    if np.isfinite(mean).any():
        ax.axvline(out["triggered_fhr_min_s"], color=figures.COLOR_VERMILLION, linestyle=":", linewidth=figures.LINE_REGULAR,
                   label=f"min at {out['triggered_fhr_min_s']:.0f} s")
    ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
    ax.set(xlabel="seconds since contraction peak", ylabel="bpm vs the 10 s before the peak", title="Contraction-triggered FHR")
    figures.legend_with_headroom(ax, ncol=2)

    events = out["events"]
    paired = events[events["context"] == "paired"]
    ax = axes[0, 1]
    if len(paired):
        ax.hist(paired["delay_s"], bins=np.arange(0, out["window_s"] + 8, 8), color=figures.COLOR_GRAY)
        ax.axvline(paired["delay_s"].median(), color=figures.COLOR_VERMILLION, linestyle=":", label=f"median {paired['delay_s'].median():.0f} s")
        figures.legend_with_headroom(ax)
    ax.set(xlabel="contraction peak -> deceleration nadir (s)", ylabel="paired decelerations", title="Measured delay")

    ax = axes[0, 2]
    for column, marker, colour in (("implied_delay_attn_s", "o", figures.COLOR_BLUE), ("implied_delay_lag_map_s", "x", figures.COLOR_ORANGE)):
        rho = out["implied"].get(column, {}).get("rho_pooled", np.nan)
        ax.scatter(paired["delay_s"], paired[column], s=8, marker=marker, color=colour, linewidths=0.6,
                   label=f"{'attention' if 'attn' in column else 'lag map'} (rho {rho:.2f})")
    if len(paired):
        span = [0.0, float(np.nanmax(paired[["delay_s", "implied_delay_attn_s", "implied_delay_lag_map_s"]].to_numpy()))]
        ax.plot(span, span, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE, linestyle="--")
    ax.set(xlabel="measured delay (s)", ylabel="implied 4(centroid + 1 + tau) (s)", title="Implied vs measured delay")
    figures.legend_with_headroom(ax)

    ax = axes[1, 0]
    rec = out["enrichment_recordings"]
    for position, column in enumerate(("enrich_attn", "enrich_attn_control", "enrich_lag_map", "enrich_lag_map_control")):
        values = rec[column].to_numpy(float) if column in rec else np.zeros(0)
        values = values[np.isfinite(values)]
        ax.scatter(position + 0.08 * np.random.default_rng(0).standard_normal(values.size), values, s=6, color=figures.COLOR_GRAY)
        if values.size:
            ax.errorbar([position], [values.mean()], fmt="o", color=figures.COLOR_BLUE, markersize=figures.MARKER_SMALL)
    ax.axhline(1.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
    ax.set_xticks(range(4), ["attn", "attn\npermuted", "lag map", "lag map\npermuted"])
    ax.set(ylabel="band / rest, per lag", title="Mass on the causal contraction's lag")

    seconds = out["nadir_axis"]
    for ax, context in ((axes[1, 1], "paired"), (axes[1, 2], "unpreceded")):
        series = []
        for branch in ("base", "full", "persistence"):
            curves = out["nadir_rmse"][context][branch]
            if len(curves):
                series.append(_band(ax, seconds, curves, seed=seed, colour=colours[branch], label=branch))
        ax.set(xlabel="seconds from anchor to nadir", ylabel="level RMSE near the nadir (bpm)",
               title=f"Skill around {context} nadirs (n rec = {len(out['nadir_rmse'][context]['full'])})")
        if series:
            figures.legend_with_headroom(ax, ncol=3)

    ax = axes[2, 0]
    series = []
    for context, colour in zip(CONTEXTS, (figures.COLOR_VERMILLION, figures.COLOR_ORANGE, figures.COLOR_GRAY)):
        if len(out["nadir_gap"][context]):
            series.append(_band(ax, seconds, out["nadir_gap"][context], seed=seed, colour=colour, label=context))
    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
    ax.set(xlabel="seconds from anchor to nadir", ylabel="nats per anchor", title="pred_gap before the nadir")
    if series:
        symlog_legend(ax, *series, ncol=3)

    for ax, covariate in zip((axes[2, 1], axes[2, 2]), DOSE_COVARIATES):
        binned = out["dose_binned"][covariate]
        for readout in DOSE_READOUTS:
            if readout in binned:
                ax.plot(binned["bin_mid"], binned[readout], marker="o", markersize=figures.MARKER_SMALL,
                        linewidth=figures.LINE_REGULAR, label=readout)
        ax.axhline(1.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
        ax.set(xlabel=f"contraction {covariate.replace('_s', ' (s)')} (quintile mid)", ylabel="readout / pooled mean |readout|",
               title=f"Dose-response: {covariate.replace('_s', '')}")
        figures.legend_with_headroom(ax, ncol=2)
    for ax in axes.ravel():
        figures.style_axes(ax)
    return figure


def build_example_page(seg: Dict[str, Any], *, title: str) -> Any:
    """One segment on its own time axis (the samples-page layout): UP, FHR, attention, lag map, KL, gap."""
    rows = ["up", "fhr", "attn", "lag_map", "kld", "gap"]
    heights = [1.0, 1.0, 1.4, 1.4, 0.8, 0.8]
    figure = plt.figure(figsize=(EXAMPLE_PAGE_WIDTH, sum(heights) * EXAMPLE_ROW_INCHES * 0.8))
    grid = GridSpec(len(rows), 2, figure=figure, height_ratios=heights, width_ratios=[1.0, 0.022],
                    left=0.065, right=0.93, top=0.95, bottom=0.05, hspace=0.55, wspace=0.09)
    seconds = np.arange(seg["up"].size) / raw.events.FS_RAW
    anchor_s = 4.0 * (seg["anchors"] + 1)                  # each anchor's causal end
    lags = np.arange(seg["attn"].shape[1])
    first = None
    for position, name in enumerate(rows):
        ax = figure.add_subplot(grid[position, 0], sharex=first)
        first = first or ax
        cax = figure.add_subplot(grid[position, 1])
        if name in ("up", "fhr"):
            ax.plot(seconds, seg[name], color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
            if name == "up":
                for onset, end in zip(seg["onset_s"], seg["end_s"]):
                    ax.axvspan(onset, end, color=figures.COLOR_ORANGE, alpha=0.2, linewidth=0)
                ax.set_ylabel("UP")
            else:
                for nadir, peak, paired in zip(seg["nadir_s"], seg["pair_peak_s"], seg["paired"]):
                    y = seg["fhr"][int(nadir * raw.events.FS_RAW)]
                    ax.plot([nadir], [y], marker="v", color=figures.COLOR_VERMILLION if paired else figures.COLOR_GRAY, markersize=4)
                    if paired:
                        ax.annotate("", xy=(nadir, y), xytext=(peak, y), arrowprops={"arrowstyle": "->", "color": figures.COLOR_VERMILLION, "linewidth": 0.6})
                ax.set_ylabel("FHR (bpm)")
            cax.set_axis_off()
        elif name in ("attn", "lag_map"):
            field = seg[name][seg["anchors"]]
            field = np.where(lags[None, :] <= seg["anchors"][:, None], field, np.nan)
            norm = unsigned_log_norm(field)
            if norm is not None:
                field = np.where(np.isfinite(field), np.maximum(field, norm.vmin), np.nan)   # a zero is data, not a blank
                image = ax.imshow(field.T, origin="lower", aspect="auto", interpolation="none", norm=norm,
                                  cmap=plt.get_cmap("viridis").with_extremes(bad=UNREAD_COLOUR),
                                  extent=(anchor_s[0] - 2, anchor_s[-1] + 2, -2, 4.0 * lags[-1] + 2))
                figure.colorbar(image, cax=cax)
            for peak in seg["peak_s"]:      # where a tracker looks: lag = anchor time - peak time
                ax.plot([peak, peak + 4.0 * lags[-1]], [0.0, 4.0 * lags[-1]], color="white", linestyle="--", linewidth=figures.LINE_THIN)
            ax.set_ylim(-2, 4.0 * lags[-1] + 2)
            ax.set_ylabel("lag (s)")
            ax.set_title("head-mean attention" if name == "attn" else "source_kl_lag_map")
        else:
            values = seg[name][seg["anchors"]] if name == "kld" else seg[name]
            ax.plot(anchor_s, values, color=figures.COLOR_BLUE, linewidth=figures.LINE_THIN)
            ax.set_ylabel("nats" if name == "kld" else "pred_gap")
            cax.set_axis_off()
        figures.style_axes(ax, grid="none")
        if position < len(rows) - 1:
            ax.tick_params(labelbottom=False)
    first.set_xlim(0, seconds[-1])
    figure.axes[-2].set_xlabel("seconds in the segment (dashed: lag of each contraction peak)")
    figure.suptitle(title, fontsize=figures.FONT_NOTE)
    figures.mark_laid_out(figure)
    return figure


# =============================================================================
# Entry
# =============================================================================
def run_decelerations_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deceleration-centred skill, causal-contraction attention, implied delay, dose-response."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(NAME, "needs a model and a loader (an offline re-run has neither)")
    model = task.orig_model.eval()
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    units = raw.SummaryUnits.from_model(model, context.config, loader)
    caps = eval_config.get("caps") or {}
    cap = raw.cap(eval_config, f"{NAME}_segments", DEFAULT_SEGMENTS)
    rows = raw.draw_segments(context, cap=cap, seed=int(eval_config.get("seed", 0)) + SEED_OFFSET)
    if rows.empty:
        return skip_record(NAME, "no scored segment could be located in the loader")
    window_s = float(caps.get(f"{NAME}_pair_window_s") or DEFAULT_PAIR_WINDOW_S)
    R, L, H = int(model.raw_per_step), int(model.max_lag) + 1, int(model.horizon)
    fs = raw.events.FS_RAW
    window = int(round(window_s * fs))
    to_bpm = units.level_delta_bpm(1.0)
    trigger_axis = np.arange(-PRE_PEAK_S * 3, window_s + 60.0, TRIGGER_BIN_S)

    decels: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []                 # (decel event, anchor) rows
    contractions: List[Dict[str, Any]] = []
    trigger: Dict[str, List[np.ndarray]] = {}
    classes: Dict[str, str] = {}
    example: Optional[Dict[str, Any]] = None
    rng = np.random.default_rng(int(eval_config.get("seed", 0)) + SEED_OFFSET)
    started = time.perf_counter()
    for chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        seg = dense_pass(model, fhr, up, weight, likelihood)
        tone = raw.resting_tone(up, weight, raw_per_step=R, validity=model.source_validity).detach().cpu().numpy()
        fhr_bpm = fhr.detach().double().cpu().numpy() * units.fhr_std + units.fhr_mean
        up_np, w_np = up.detach().double().cpu().numpy(), weight.detach().double().cpu().numpy()
        anchors, anchor0 = seg["anchors"], int(seg["anchors"][0])
        lag = np.arange(L)
        for b, (_, row) in enumerate(chunk.iterrows()):
            guid = str(row["guid"])
            classes[guid] = str(row.get(labels.CLASS_COLUMN, "all"))
            valid = raw.events.raw_validity(w_np[b], decimation=R, raw_len=up_np.shape[1])
            con = raw.events.detect_contractions(up_np[b], valid=valid)
            dec = raw.events.detect_decelerations(fhr_bpm[b], valid=valid)
            order = np.argsort(con["peak_raw"])
            peaks, onsets, ends = (con[k][order] for k in ("peak_raw", "onset_raw", "end_raw"))
            owner = pair_events(peaks, dec["nadir_raw"], dec["depth_bpm"], window)
            attn, lag_map, kld, gap, contributing = seg["attn"][b], seg["lag_map"][b], seg["kld"][b], seg["gap"][b], seg["contributing"][b]
            heads = seg["attn_heads"][b]
            base = {"guid": guid, labels.CLASS_COLUMN: classes[guid], "sample_index": row.get("sample_index")}

            # -- contraction-triggered FHR (bpm relative to the 10 s before the peak)
            for peak in peaks:
                index = peak + np.round(trigger_axis * fs).astype(int)
                before = fhr_bpm[b][max(0, peak - int(PRE_PEAK_S * fs)):peak]
                if before.size and valid[max(0, peak - int(PRE_PEAK_S * fs)):peak].all():
                    ok = (index >= 0) & (index < fhr_bpm.shape[1])
                    curve = np.full(trigger_axis.size, np.nan)
                    curve[ok] = np.where(valid[index[ok]], fhr_bpm[b][index[ok]] - before.mean(), np.nan)
                    trigger.setdefault(guid, []).append(curve)

            # -- permuted-delay control: the segment's measured delays shuffled across its paired nadirs
            paired_index = np.flatnonzero(owner >= 0)
            delay_tok = dec["nadir_raw"][paired_index] // R - peaks[owner[paired_index]] // R
            control_tok = dict(zip(paired_index.tolist(), rng.permutation(delay_tok).tolist())) if paired_index.size >= 2 else {}

            # -- decelerations: context, skill around the nadir, enrichment, implied delay
            for i, (nadir, depth) in enumerate(zip(dec["nadir_raw"], dec["depth_bpm"])):
                n_tok, j = int(nadir) // R, int(owner[i])
                context_name = "paired" if j >= 0 else ("preceded" if ((peaks <= nadir) & (peaks >= nadir - window)).any() else "unpreceded")
                p_tok = int(peaks[j]) // R if j >= 0 else None
                event = {**base, "event": len(decels), "nadir_s": nadir / fs, "depth_bpm": float(depth), "context": context_name,
                         "peak_s": peaks[j] / fs if j >= 0 else np.nan, "delay_s": (nadir - peaks[j]) / fs if j >= 0 else np.nan}
                implied = {"attn": [], "lag_map": []}
                for t in range(n_tok - H, n_tok):
                    column = t - anchor0
                    if column < 0 or column >= anchors.size or not contributing[column]:
                        continue
                    tau_n = n_tok - t - 1
                    near = slice(max(0, tau_n - NADIR_HALF_WIDTH), min(H, tau_n + NADIR_HALF_WIDTH + 1))
                    record = {"event": event["event"], "guid": guid, "context": context_name, "steps_to_nadir": n_tok - t, "gap": float(gap[column])}
                    for branch in ("base", "full", "persistence"):
                        record[f"sq_{branch}"] = float(np.nanmean(seg[f"err_{branch}"][b, column, near] ** 2)) * to_bpm ** 2
                    support = min(t, L - 1)
                    for suffix, peak_tok in (("", p_tok), ("_control", n_tok - control_tok[i] if i in control_tok else None)):
                        if peak_tok is None or not 0 <= t - peak_tok <= support:
                            continue
                        record[f"enrich_attn{suffix}"] = band_enrichment(attn[t], t - peak_tok, support)
                        record[f"enrich_lag_map{suffix}"] = band_enrichment(lag_map[t], t - peak_tok, support)
                        for h in range(heads.shape[1]):
                            record[f"enrich_attn_head{h}{suffix}"] = band_enrichment(heads[t, h], t - peak_tok, support)
                    for kind, field in (("attn", attn), ("lag_map", lag_map)):
                        implied[kind].append(4.0 * (centroid(field[t], support) + 1 + tau_n))
                    pairs.append(record)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    event["implied_delay_attn_s"] = float(np.nanmean(implied["attn"])) if implied["attn"] else np.nan
                    event["implied_delay_lag_map_s"] = float(np.nanmean(implied["lag_map"])) if implied["lag_map"] else np.nan
                decels.append(event)

            # -- dose-response: every contraction
            for onset, peak, end in zip(onsets, peaks, ends):
                p_tok, on_tok, end_tok = int(peak) // R, int(onset) // R, int(end) // R
                values: Dict[str, List[float]] = {k: [] for k in DOSE_READOUTS}
                for t in range(p_tok, p_tok + L):
                    column = t - anchor0
                    if column < 0 or column >= anchors.size or not contributing[column]:
                        continue
                    span = (lag >= t - end_tok) & (lag <= t - on_tok) & (lag <= min(t, L - 1))
                    values["kld"].append(kld[t])
                    values["gap"].append(gap[column])
                    values["attn_mass"].append(attn[t][span].sum())
                    values["lag_map_mass"].append(lag_map[t][span].sum())
                if values["kld"]:
                    amplitude = units.up_std * (up_np[b][max(0, peak - 8):peak + 9].mean() - float(tone[b]))
                    contractions.append({**base, "peak_s": peak / fs, "duration_s": (end - onset) / fs, "amplitude": float(amplitude),
                                         **{k: float(np.mean(v)) for k, v in values.items()}})

            n_paired = int((owner >= 0).sum())
            if example is None or n_paired > example["n_paired"]:
                example = {"n_paired": n_paired, "guid": guid, "up": units.up_physical(up_np[b]), "fhr": fhr_bpm[b], "anchors": anchors,
                           "attn": attn, "lag_map": lag_map, "kld": kld, "gap": gap, "onset_s": onsets / fs, "end_s": ends / fs,
                           "peak_s": peaks / fs, "nadir_s": dec["nadir_raw"] / fs, "paired": owner >= 0,
                           "pair_peak_s": np.where(owner >= 0, peaks[np.clip(owner, 0, None)] / fs if peaks.size else np.nan, np.nan)}
    elapsed = time.perf_counter() - started
    if not decels:
        return skip_record(NAME, "no deceleration was detected on the drawn segments' raw FHR")

    seed = int(eval_config.get("seed", 0))
    resamples = int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES))
    events, anchor_rows, con_frame = pd.DataFrame(decels), pd.DataFrame(pairs), pd.DataFrame(contractions)
    n_heads = int(model.num_heads)
    enrich_columns = ["enrich_attn", "enrich_attn_control", "enrich_lag_map", "enrich_lag_map_control",
                      *(f"enrich_attn_head{h}{suffix}" for h in range(n_heads) for suffix in ("", "_control"))]
    for column in enrich_columns:
        if column not in anchor_rows:
            anchor_rows[column] = np.nan
    if len(anchor_rows):
        events = events.join(anchor_rows.groupby("event")[enrich_columns].mean(), on="event")
    else:
        events = events.assign(**{c: np.nan for c in enrich_columns})
    paired = events[events["context"] == "paired"]

    # -- implied vs measured delay
    implied: Dict[str, Dict[str, float]] = {}
    for column in ("implied_delay_attn_s", "implied_delay_lag_map_s"):
        ok = paired[["guid", "delay_s", column]].dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled = float(scipy_stats.spearmanr(ok["delay_s"], ok[column])[0]) if len(ok) >= 3 else float("nan")
            per_rec = [scipy_stats.spearmanr(g["delay_s"], g[column])[0] for _, g in ok.groupby("guid") if len(g) >= 3]
        implied[column] = {"rho_pooled": pooled, "rho_per_recording_mean": float(np.nanmean(per_rec)) if per_rec else float("nan"),
                           "n_events": int(len(ok)), "n_recordings_rho": len(per_rec),
                           "median_abs_error_s": float((ok[column] - ok["delay_s"]).abs().median()) if len(ok) else float("nan")}

    # -- per recording
    guids = sorted(classes)
    recordings = pd.DataFrame({"guid": guids, labels.CLASS_COLUMN: [classes[g] for g in guids]}).set_index("guid")
    recordings["n_decelerations"] = events.groupby("guid").size()
    for context_name in CONTEXTS:
        recordings[f"n_{context_name}"] = events[events["context"] == context_name].groupby("guid").size()
    recordings["recording_delay_median_s"] = paired.groupby("guid")["delay_s"].median()
    recordings = recordings.join(paired.groupby("guid")[enrich_columns].mean())
    steps = np.arange(1, H + 1)
    nadir_rmse: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in CONTEXTS}
    nadir_gap: Dict[str, np.ndarray] = {}
    curve_rows: List[Dict[str, Any]] = []
    for context_name in CONTEXTS:
        part = anchor_rows[anchor_rows["context"] == context_name] if len(anchor_rows) else anchor_rows
        grouped = part.groupby(["guid", "steps_to_nadir"]).mean(numeric_only=True) if len(part) else None
        present = sorted(part["guid"].unique()) if len(part) else []
        def per_rec(column: str) -> np.ndarray:
            return np.stack([grouped[column].xs(g, level="guid").reindex(steps).to_numpy(float) for g in present]) if present else np.zeros((0, H))
        for branch in ("base", "full", "persistence"):
            nadir_rmse[context_name][branch] = np.sqrt(per_rec(f"sq_{branch}"))
        nadir_gap[context_name] = per_rec("gap")
        for quantity, curves in [*((f"rmse_{b}_bpm", nadir_rmse[context_name][b]) for b in ("base", "full", "persistence")), ("pred_gap", nadir_gap[context_name])]:
            if len(curves):
                mean, lo, hi = class_contrast.mean_band(curves, seed=seed)
                curve_rows += [{"context": context_name, "quantity": quantity, "seconds_to_nadir": 4.0 * k, "mean": m, "ci_lo": a, "ci_hi": z,
                                "n_recordings": len(curves)} for k, m, a, z in zip(steps, mean, lo, hi)]
    trigger_curves = np.stack([np.nanmean(np.stack(trigger[g]), axis=0) for g in guids if g in trigger]) if trigger else np.zeros((0, trigger_axis.size))
    trigger_mean = np.nanmean(trigger_curves, axis=0) if len(trigger_curves) else np.full(trigger_axis.size, np.nan)
    after = trigger_axis >= 0
    triggered_min_s = float(trigger_axis[after][np.nanargmin(trigger_mean[after])]) if np.isfinite(trigger_mean[after]).any() else float("nan")

    # -- dose-response: per-recording slopes, Wilcoxon against zero, pooled quintile curves
    slope_rows, dose_binned = [], {}
    for covariate in DOSE_COVARIATES:
        for readout in DOSE_READOUTS:
            slopes = con_frame.groupby("guid").apply(lambda g: ols_slope(g[covariate].to_numpy(float), g[readout].to_numpy(float))) if len(con_frame) else pd.Series(dtype=float)
            recordings[f"slope_{readout}_vs_{covariate}"] = slopes
            test = shared_stats.wilcoxon_paired(slopes.to_numpy(float), np.zeros(len(slopes)), label_left="slope", label_right="zero")
            interval = shared_stats.bootstrap_ci(slopes.to_numpy(float), resamples=resamples, seed=seed)
            slope_rows.append({"covariate": covariate, "readout": readout, "mean_slope": interval["point"], "ci_lo": interval["lo"],
                               "ci_hi": interval["hi"], "n_recordings": test["n_pairs"], "wilcoxon_p": test["p_value"]})
        if len(con_frame) >= 5:
            bins = pd.qcut(con_frame[covariate], 5, duplicates="drop")
            table = con_frame.groupby(bins, observed=True)[list(DOSE_READOUTS)].mean()
            table = table / con_frame[list(DOSE_READOUTS)].abs().mean()
            table["bin_mid"] = [interval.mid for interval in table.index]
            dose_binned[covariate] = table.reset_index(drop=True)
        else:
            dose_binned[covariate] = pd.DataFrame()
    slopes_frame = pd.DataFrame(slope_rows)

    summary_rows = []
    for column in ["recording_delay_median_s", *enrich_columns]:
        interval = shared_stats.bootstrap_ci(recordings[column].to_numpy(float), resamples=resamples, seed=seed)
        summary_rows.append({"metric": column, **interval})
    summary = pd.DataFrame(summary_rows)
    point = summary.set_index("metric")["point"]

    # -- files
    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    events.to_csv(directory / "decelerations_events.csv", index=False)
    con_frame.to_csv(directory / "decelerations_contractions.csv", index=False)
    recordings.reset_index().to_csv(directory / "decelerations_recordings.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(directory / "decelerations_nadir_curves.csv", index=False)
    slopes_frame.to_csv(directory / "decelerations_dose_slopes.csv", index=False)
    summary.to_csv(directory / "decelerations_summary.csv", index=False)
    pd.DataFrame({"seconds_since_peak": trigger_axis, "fhr_bpm_vs_pre_peak": trigger_mean}).to_csv(directory / "decelerations_triggered_fhr.csv", index=False)
    files = ["decelerations_events.csv", "decelerations_contractions.csv", "decelerations_recordings.csv", "decelerations_nadir_curves.csv",
             "decelerations_dose_slopes.csv", "decelerations_summary.csv", "decelerations_triggered_fhr.csv"]
    if len(set(classes.values())) >= 2:
        metrics = recordings.reset_index()
        stats_frame, pairwise = class_contrast.class_tests(
            metrics, {NAME: ["recording_delay_median_s", "enrich_attn", "enrich_lag_map"],
                      f"{NAME}_dose": [c for c in metrics if c.startswith("slope_")]}, seed=seed)
        stats_frame.to_csv(directory / "decelerations_class_stats.csv", index=False)
        pairwise.to_csv(directory / "decelerations_class_pairwise.csv", index=False)
        files += ["decelerations_class_stats.csv", "decelerations_class_pairwise.csv"]
    figure = build_summary_figure({
        "events": events, "window_s": window_s, "implied": implied, "enrichment_recordings": recordings,
        "trigger_axis": trigger_axis, "trigger_curves": trigger_curves, "triggered_fhr_min_s": triggered_min_s,
        "nadir_axis": 4.0 * steps, "nadir_rmse": nadir_rmse, "nadir_gap": nadir_gap, "dose_binned": dose_binned,
    }, seed=seed)
    files.append(str(figures.render_figure(figure, directory / "decelerations_summary").name))
    if example is not None:
        page = build_example_page(example, title=f"{example['guid']}: {example['n_paired']} paired decelerations (red arrows: contraction peak -> nadir)")
        files.append(str(figures.render_figure(page, directory / "decelerations_example", tight=False).name))

    headline = {
        "measured_delay_median_s": float(paired["delay_s"].median()) if len(paired) else None,
        "triggered_fhr_min_s": triggered_min_s,
        "implied_delay_spearman": implied["implied_delay_attn_s"]["rho_pooled"],
        "implied_delay_attn_median_s": float(paired["implied_delay_attn_s"].median()) if len(paired) else None,
        "attention_enrichment": float(point["enrich_attn"]),
        "attention_enrichment_control": float(point["enrich_attn_control"]),
    }
    logger.info(f"{NAME}: {len(events)} decelerations ({len(paired)} paired), median delay {headline['measured_delay_median_s']}, "
                f"triggered-FHR min {triggered_min_s:.0f} s, implied-delay rho {headline['implied_delay_spearman']:.2f}, {elapsed:.1f} s")
    return {
        "n_samples": int(len(rows)),
        "composition": {"n_recordings": len(guids), "n_decelerations": int(len(events)), "n_contractions_scored": int(len(con_frame)),  # with >= 1 scored anchor after the peak
                        **{f"n_{c}": int((events["context"] == c).sum()) for c in CONTEXTS}},
        "plan": {"capped": True, "cap_segments": cap, "pair_window_s": window_s, "nadir_half_width_steps": NADIR_HALF_WIDTH,
                 "decode": "latent means", "deceleration_prominence_bpm": raw.events.DECELERATION_PROMINENCE_BPM},
        "implied_delay": implied,
        "summary": summary.to_dict(orient="records"),
        "dose_slopes": slopes_frame.to_dict(orient="records"),
        "cost": {"elapsed_s": float(elapsed), "n_segments": int(len(rows))},
        "headline": headline,
        "files": files,
    }
