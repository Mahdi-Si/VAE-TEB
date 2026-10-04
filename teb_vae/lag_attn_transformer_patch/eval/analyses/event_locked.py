r"""``event_locked`` -- contraction-locked lag maps and attribution (plan §2, E1-E; EVAL_MAP_LAG N2,
EVAL_MAP_CAPTUM P4).

**Question.** Does the attention follow a contraction as it recedes into the past (a diagonal
$\ell = s/4$ on a lag × time-since-peak map) or sit at fixed lags? Do $K_t$, the forecast gap and the
level error move with time since the peak? Which raw UP samples around a peak drive $K_t$ and the gap?

**Computation.** Contractions come from raw UP (``events.detect_contractions``, ``weight``-masked, as the
collection pass). Over a capped draw of segments, one dense forward each at the latent **means**:
every scored anchor $t$ gets $k = t - \mathrm{tok}(\text{latest peak} \le t) \in [0, L-1]$ and, separately,
$k = t - \mathrm{tok}(\text{next peak}) \in [-8, -1]$; $s = 4k$ s. Per $k$: per-head attention
$\alpha(k, m, \ell)$, ``source_kl_lag_map`` $\tilde K(k, \ell)$, $K_t$, ``pred_gap`` (base − full block
NLL, nats per anchor) and the base/full level RMSE per horizon step (bpm). Averaged per recording,
then over recordings. Lags $\ell > t$ (outside the anchor's support) are blank.

**Tracking index** (per head, head mean, and $\tilde K$): on the $k \ge 0$ map, mean per cell on the
diagonal band $|\ell - k| \le 1$ over mean per cell off it. Every lag column is covered on and off the
band alike, so a fixed-lag preference cancels: 1 = no tracking, > 1 = the head follows the contraction.
Per recording, then bootstrapped over recordings.

**Attribution.** For up to ``caps.event_locked_ig_events`` contractions, anchors at peak + $o$
(``caps.event_locked_ig_offsets`` offsets evenly over $[0, 4(L-1)]$ s): raw-UP IG
(``raw.RawReadout``, 128 steps by default, entry $10^{-3}$) of ``kld`` and ``pred_gap`` from a **resting-tone**
baseline (UP flat at the segment's 10th percentile: contractions removed, tone kept), re-indexed to
seconds from the peak (1 s bins), per recording then over recordings, plus the |IG| shares on the
rising limb ``[onset, peak)``, the falling limb ``[peak, end]`` and outside the event. Completeness
$|\delta| / \max(|f(x) - f(x_0)|, |f(x)|)$ per row.

**Gain by target time.** The base − full level-RMSE map over (time since peak, lead) collapsed onto
the target's own time since the peak, $s + 4(1 + \tau)$: a UP→FHR effect at delay $d$ shows as a
peak at $d$ whatever anchor saw it (``gain_peak_target_s``).

Caps (``eval_config.caps``): ``event_locked_segments`` (256), ``event_locked_ig_events`` (64),
``event_locked_ig_offsets`` (6), ``event_locked_ig_steps`` (module default 128; the overrides ship 512,
because the tone path is rougher than the zero one: on the planted model 128 steps left 140 of 396 rows over
the 1e-2 completeness tolerance and 512 left 5, max 0.136; read ``ig_completeness.n_rows_over_tolerance``).
Production cost at 512 steps: 256 dense forwards plus 64 × 6 × 2 = 768 IG rows, about 20 min on a GTX 1660 Ti.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib.gridspec import GridSpec

from teb_vae.lag_attn_cfs.eval import class_contrast
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval._reuse import stats as shared_stats
from teb_vae.lag_attn_cfs.eval.attributions import (
    EXAMPLE_PAGE_WIDTH,
    EXAMPLE_ROW_INCHES,
    UNREAD_COLOUR,
    signed_log_norm,
    symlog_legend,
    unsigned_log_norm,
)
from teb_vae.lag_attn_cfs.eval.attribution_pass import COMPLETENESS_TOLERANCE
from teb_vae.lag_attn_cfs.eval.metrics import anchor_support
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "event_locked"
PRE_TOKENS = 8                      # k < 0 bins: 32 s before the peak
DIAGONAL_HALF_WIDTH = 1             # tokens either side of l = k
DEFAULT_IG_STEPS = 128              # the plan's; caps.event_locked_ig_steps raises it (the tone path is rougher than the zero one)
IG_READOUTS = ("kld", "pred_gap")
IG_ROWS_PER_CALL = 16
X_BIN_S = 1.0                       # raw-time bin of the attribution map and the UP waveform
DEFAULT_SEGMENTS, DEFAULT_IG_EVENTS, DEFAULT_IG_OFFSETS = 256, 64, 6
SEED_OFFSET = 41

#: ``(headline_name, key)``, registered as ``("event_locked", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("event_locked_tracking_attention", "tracking_attention"),
    ("event_locked_tracking_attention_best_head", "tracking_attention_best_head"),
    ("event_locked_tracking_lag_map", "tracking_lag_map"),
    ("event_locked_gain_peak_target_s", "gain_peak_target_s"),
    ("event_locked_ig_completeness_max", "ig_completeness_max"),
)


# =============================================================================
# The pass (mean-decoded through ``raw.latent_means``)
# =============================================================================
@torch.no_grad()
def dense_pass(model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor, likelihood: str) -> Dict[str, np.ndarray]:
    """One mean-decoded dense forward: the per-anchor readouts this analysis averages, on the host."""
    with raw.latent_means(model):
        out = raw.forward_raw(model, fhr, up, weight)
    anchors = out["anchor_index"]
    target = raw.forecast_target(model, fhr, weight, anchors)
    mask, _coverage, _kl = anchor_support(model, weight, out)
    nll = {branch: raw.block_nll(model, out[f"mu_{branch}"], out[f"logvar_{branch}"], target, mask, likelihood)
           for branch in ("base", "full")}
    scored = mask > 0
    host = lambda x: x.detach().float().cpu().numpy()  # noqa: E731
    nan = torch.tensor(float("nan"), device=target.device)
    return {
        "anchors": host(anchors[0]).astype(np.int64),
        "contributing": host(scored.any(dim=-1)).astype(bool),
        "attn": host(out["attn_weights"]),
        "lag_map": host(out["source_kl_lag_map"]),
        "kld": host(out["kld_per_t"]),
        "gap": host(nll["base"] - nll["full"]),
        **{f"err_{branch}": host(torch.where(scored, out[f"mu_{branch}"][..., 0] - target[..., 0], nan))
           for branch in ("base", "full")},
    }


def contractions(up: np.ndarray, weight: np.ndarray, raw_per_step: int) -> Dict[str, np.ndarray]:
    """``detect_contractions`` on one raw UP row, validity from ``weight`` as the collection pass does."""
    valid = raw.events.raw_validity(weight, decimation=raw_per_step, raw_len=up.size)
    return raw.events.detect_contractions(up, valid=valid)


def tracking_index(field: np.ndarray, half_width: int = DIAGONAL_HALF_WIDTH) -> float:
    r"""Mean per cell on the band $|\ell - k| \le w$ over mean per cell off it; ``field`` is $(K \ge 0, L)$."""
    k, lag = np.indices(field.shape)
    on = np.abs(lag - k) <= half_width
    with np.errstate(all="ignore"):
        inside, outside = np.nanmean(np.where(on, field, np.nan)), np.nanmean(np.where(on, np.nan, field))
    return float(inside / outside) if np.isfinite(outside) and outside > 0 else float("nan")


# =============================================================================
# Accumulation, per recording
# =============================================================================
class _Sums:
    """Per-recording sums over anchors (k bins) and over IG rows (offset × raw-time bins)."""

    def __init__(self, n_bins: int, heads: int, lags: int, horizon: int, offsets: int, x_bins: int) -> None:
        self.n = np.zeros(n_bins)
        self.support = np.zeros((n_bins, lags))
        self.attn = np.zeros((n_bins, heads, lags))
        self.lag_map = np.zeros((n_bins, lags))
        self.kld = np.zeros(n_bins)
        self.gap = np.zeros(n_bins)
        self.sq = {b: np.zeros((n_bins, horizon)) for b in ("base", "full")}
        self.sq_n = np.zeros((n_bins, horizon))
        self.up = np.zeros(x_bins)
        self.up_n = np.zeros(x_bins)
        self.ig = {r: np.zeros((offsets, x_bins)) for r in IG_READOUTS}
        self.ig_n = {r: np.zeros((offsets, x_bins)) for r in IG_READOUTS}
        self.phase = {r: np.zeros((offsets, 3)) for r in IG_READOUTS}     # |IG| on rise, fall, outside
        self.coverage = np.zeros((offsets, 2))                             # samples in the event, in the window
        self.n_events = 0

    def means(self) -> Dict[str, np.ndarray]:
        with np.errstate(all="ignore"):
            support = np.where(self.support > 0, self.support, np.nan)
            n = np.where(self.n > 0, self.n, np.nan)
            out = {
                "attn": self.attn / support[:, None, :],
                "lag_map": self.lag_map / support,
                "kld": self.kld / n, "gap": self.gap / n,
                "up": self.up / np.where(self.up_n > 0, self.up_n, np.nan),
            }
            for branch in ("base", "full"):
                out[f"rmse_{branch}"] = np.sqrt(self.sq[branch] / np.where(self.sq_n > 0, self.sq_n, np.nan))
            for readout in IG_READOUTS:
                out[f"ig_{readout}"] = self.ig[readout] / np.where(self.ig_n[readout] > 0, self.ig_n[readout], np.nan)
                total = self.phase[readout].sum(axis=1, keepdims=True)
                out[f"phase_{readout}"] = self.phase[readout] / np.where(total > 0, total, np.nan)
            out["coverage"] = self.coverage[:, 0] / np.where(self.coverage[:, 1] > 0, self.coverage[:, 1], np.nan)
        return out


def accumulate_segment(sums: _Sums, seg: Dict[str, np.ndarray], peaks_tok: np.ndarray, n_lags: int) -> None:
    """Add one segment's anchors to its recording's k-bin sums (latest peak for k >= 0, next peak for k < 0)."""
    anchors, keep = seg["anchors"], seg["contributing"]
    if peaks_tok.size == 0:
        return
    position = np.searchsorted(peaks_tok, anchors, side="right")
    since = np.where(position > 0, anchors - peaks_tok[np.clip(position - 1, 0, None)], 10 ** 6)
    until = np.where(position < peaks_tok.size, anchors - peaks_tok[np.clip(position, None, peaks_tok.size - 1)], -10 ** 6)
    lag = np.arange(n_lags)
    for k in (since, until):
        use = keep & (k >= -PRE_TOKENS) & (k <= n_lags - 1)
        column, steps, bins = np.flatnonzero(use), anchors[use], k[use] + PRE_TOKENS
        if not column.size:
            continue
        np.add.at(sums.n, bins, 1.0)
        np.add.at(sums.support, bins, (lag[None, :] <= steps[:, None]).astype(float))
        np.add.at(sums.attn, bins, seg["attn"][steps])
        np.add.at(sums.lag_map, bins, seg["lag_map"][steps])
        np.add.at(sums.kld, bins, seg["kld"][steps])
        np.add.at(sums.gap, bins, seg["gap"][column])
        for branch in ("base", "full"):
            err = seg[f"err_{branch}"][column]
            np.add.at(sums.sq[branch], bins, np.nan_to_num(err ** 2))
        np.add.at(sums.sq_n, bins, np.isfinite(seg["err_full"][column]).astype(float))


def accumulate_waveform(sums: _Sums, up_phys: np.ndarray, peaks_raw: np.ndarray, x_lo: float, x_bins: int) -> None:
    """Event-triggered raw UP (physical units), binned at ``X_BIN_S`` around each peak."""
    per_bin = int(round(X_BIN_S * raw.events.FS_RAW))
    offsets = np.arange(x_bins * per_bin) + int(round(x_lo * raw.events.FS_RAW))
    for peak in peaks_raw:
        index = peak + offsets
        ok = (index >= 0) & (index < up_phys.size)
        np.add.at(sums.up, (np.flatnonzero(ok) // per_bin), up_phys[index[ok]])
        np.add.at(sums.up_n, (np.flatnonzero(ok) // per_bin), 1.0)


# =============================================================================
# Contraction-locked IG from the resting-tone baseline
# =============================================================================
# =============================================================================
# Figure: one page, every row on the time-since-peak axis
# =============================================================================
def _norm(field: np.ndarray) -> Any:
    finite = field[np.isfinite(field)]
    return signed_log_norm(field) if finite.size and finite.min() < 0 else unsigned_log_norm(field)


def build_figure(cohort: Dict[str, Any], *, s_anchor: np.ndarray, lag_s: np.ndarray, x_raw: np.ndarray,
                 offsets_s: np.ndarray, horizon_s: np.ndarray, title: str) -> Any:
    """The samples-page layout: one data column + a colour-bar column, every row sharing seconds-from-peak."""
    heads = cohort["attn"].shape[1]
    rows = (["up"] + [f"head{m}" for m in range(heads)]
            + ["lag_map", "diagonal", "kld", "gap", "rmse", "gain"]
            + [f"ig_{r}" for r in IG_READOUTS if np.isfinite(cohort[f"ig_{r}"]).any()])
    heights = [1.4 if r.startswith(("head", "lag_map", "gain", "ig_")) else 1.0 for r in rows]
    figure = plt.figure(figsize=(EXAMPLE_PAGE_WIDTH, sum(heights) * EXAMPLE_ROW_INCHES * 0.8))
    grid = GridSpec(len(rows), 2, figure=figure, height_ratios=heights, width_ratios=[1.0, 0.022],
                    left=0.065, right=0.93, top=0.97, bottom=0.03, hspace=0.6, wspace=0.09)
    first = None

    def heat(ax: Any, cax: Any, field: np.ndarray, x: np.ndarray, y: np.ndarray, label: str, cmap: str) -> None:
        norm = _norm(field)
        if isinstance(norm, mcolors.LogNorm):
            field = np.where(np.isfinite(field), np.maximum(field, norm.vmin), np.nan)   # a zero is data, not a blank
        if norm is None:
            ax.text(0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes, ha="center")
            cax.set_axis_off()
            return
        dx, dy = 0.5 * (x[1] - x[0]) if x.size > 1 else 0.5, 0.5 * (y[1] - y[0]) if y.size > 1 else 0.5
        image = ax.imshow(field.T, origin="lower", aspect="auto", interpolation="none", norm=norm,
                          cmap=plt.get_cmap(cmap).with_extremes(bad=UNREAD_COLOUR),
                          extent=(x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy))
        figure.colorbar(image, cax=cax).set_label(label, fontsize=figures.FONT_TINY)

    for position, name in enumerate(rows):
        ax = figure.add_subplot(grid[position, 0], sharex=first)
        first = first or ax
        cax = figure.add_subplot(grid[position, 1])
        if name == "up":
            ax.plot(x_raw, cohort["up"], color=figures.COLOR_BLACK, linewidth=figures.LINE_REGULAR)
            ax.set_ylabel("UP")
            ax.set_title("Raw UP around the contraction peak (mean over recordings)")
            cax.set_axis_off()
        elif name.startswith("head") or name == "lag_map":
            field = cohort["attn"][:, int(name[4:])] if name.startswith("head") else cohort["lag_map"]
            heat(ax, cax, field, s_anchor, lag_s, "attention" if name != "lag_map" else "nats", "viridis")
            ax.plot(s_anchor, s_anchor, color=figures.COLOR_BLACK, linestyle="--", linewidth=figures.LINE_REGULAR)
            ax.set_ylim(lag_s[0] - 2, lag_s[-1] + 2)
            ax.set_ylabel("lag (s)")
            index = cohort["tracking"].get(name if name == "lag_map" else f"attention_head{name[4:]}", np.nan)
            what = f"head {name[4:]} attention" if name != "lag_map" else "source_kl_lag_map"
            ax.set_title(f"{what} by time since peak (dashed: l = s/4; tracking index {index:.2f})")
        elif name == "diagonal":
            series = [cohort["diagonal"][:, m] for m in range(heads)]
            for m, curve in enumerate(series):
                ax.plot(s_anchor, curve, linewidth=figures.LINE_REGULAR, label=f"head {m}")
            ax.set_ylabel("mass on |l - s/4| <= 1")
            ax.set_title("Attention on the contraction's own lag band")
            symlog_legend(ax, *series, ncol=heads)
            cax.set_axis_off()
        elif name in ("kld", "gap"):
            series = []
            for cls, (mean, lo, hi) in cohort[f"{name}_by_class"].items():
                series.append(mean)
                line, = ax.plot(s_anchor, mean, linewidth=figures.LINE_REGULAR, label=f"{cls} (n={cohort['n_by_class'][cls]})")
                if np.isfinite(lo).any():
                    ax.fill_between(s_anchor, lo, hi, color=line.get_color(), alpha=0.2, linewidth=0)
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            ax.set_ylabel("nats per step" if name == "kld" else "nats per anchor")
            ax.set_title("$K_t$" if name == "kld" else "pred_gap (base - full, mean-decoded)")
            symlog_legend(ax, *series, ncol=4)
            cax.set_axis_off()
        elif name == "rmse":
            d, (mean, lo, hi) = cohort["gain_by_target_time"]
            ax.plot(d, mean, color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR)
            if np.isfinite(lo).any():
                ax.fill_between(d, lo, hi, color=figures.COLOR_BLUE, alpha=0.2, linewidth=0)
            ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_HAIRLINE)
            if np.isfinite(mean).any():
                ax.axvline(d[np.nanargmax(mean)], color=figures.COLOR_VERMILLION, linestyle=":", linewidth=figures.LINE_REGULAR)
            ax.set_ylabel("bpm")
            ax.set_title("Level RMSE gain (base - full) by target time since peak, s + 4(1+tau) (dotted: argmax)")
            cax.set_axis_off()
        elif name == "gain":
            heat(ax, cax, cohort["rmse_base"] - cohort["rmse_full"], s_anchor, horizon_s, "bpm", "RdBu_r")
            ax.set_ylabel("lead 4(1+tau) (s)")
            ax.set_title("Level RMSE gain (base - full) by time since peak and lead (a fixed UP->FHR delay is a line s + lead = d)")
        else:
            readout = name[3:]
            heat(ax, cax, cohort[name].T, x_raw, offsets_s, "readout units", "RdBu_r")
            ax.plot(offsets_s, offsets_s, linestyle="none", marker="|", color=figures.COLOR_BLACK, markersize=6)
            ax.set_ylabel("anchor at peak + (s)")
            ax.set_title(f"Raw-UP IG of {readout} from the resting-tone baseline (| marks the anchor)")
        figures.style_axes(ax, grid="none")
        if position < len(rows) - 1:
            ax.tick_params(labelbottom=False)
    first.set_xlim(x_raw[0], x_raw[-1])
    figure.axes[-2].set_xlabel("seconds relative to the contraction peak")
    figure.suptitle(title, fontsize=figures.FONT_NOTE)
    figures.mark_laid_out(figure)
    return figure


# =============================================================================
# Entry
# =============================================================================
def run_event_locked_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Contraction-locked maps, tracking index and tone-baseline IG; see the module docstring."""
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
    ig_events = int(caps.get(f"{NAME}_ig_events") or DEFAULT_IG_EVENTS)
    n_offsets = int(caps.get(f"{NAME}_ig_offsets") or DEFAULT_IG_OFFSETS)
    ig_steps = int(caps.get(f"{NAME}_ig_steps") or DEFAULT_IG_STEPS)

    R, L, H = int(model.raw_per_step), int(model.max_lag) + 1, int(model.horizon)
    heads = int(model.num_heads)
    offsets_tok = np.unique(np.round(np.linspace(0, L - 1, n_offsets)).astype(int))
    x_lo, x_hi = -4.0 * (L + 1), 4.0 * max(L + 1, L + H)       # covers the target times s + 4(1 + tau)
    x_bins = int(round((x_hi - x_lo) / X_BIN_S))
    sums: Dict[str, _Sums] = {}
    classes: Dict[str, str] = {}
    event_rows: List[Dict[str, Any]] = []
    completeness: Dict[str, List[float]] = {r: [] for r in IG_READOUTS}
    wrappers = {r: raw.RawReadout(raw.AnchorReadout(model, raw.ATTENTION_CELL, readout=r, likelihood=likelihood).eval())
                for r in IG_READOUTS}
    ig_budget = ig_events
    started = time.perf_counter()
    for chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        seg = dense_pass(model, fhr, up, weight, likelihood)
        up_np, w_np = up.detach().double().cpu().numpy(), weight.detach().double().cpu().numpy()
        tone = raw.resting_tone(up, weight, raw_per_step=R, validity=model.source_validity)
        anchor0 = int(seg["anchors"][0])
        ig_rows: List[Tuple[int, int, int, int, int, int]] = []      # (sample, peak, offset index, column, onset, end)
        for b, (_, row) in enumerate(chunk.iterrows()):
            guid = str(row["guid"])
            classes[guid] = str(row.get(labels.CLASS_COLUMN, "all"))
            acc = sums.setdefault(guid, _Sums(PRE_TOKENS + L, heads, L, H, offsets_tok.size, x_bins))
            found = contractions(up_np[b], w_np[b], R)
            order = np.argsort(found["peak_raw"])
            peaks, onsets, ends = found["peak_raw"][order], found["onset_raw"][order], found["end_raw"][order]
            acc.n_events += int(peaks.size)
            for onset, peak, end in zip(found["onset_raw"], found["peak_raw"], found["end_raw"]):
                event_rows.append({"guid": guid, labels.CLASS_COLUMN: classes[guid], "sample_index": row.get("sample_index"),
                                   "onset_s": onset / 4.0, "peak_s": peak / 4.0, "end_s": end / 4.0,
                                   "duration_s": (end - onset) / 4.0,
                                   "amplitude_above_tone": float(units.up_std * (up_np[b][max(0, peak - 8):peak + 9].mean() - float(tone[b])))})
            accumulate_segment(acc, {k: (v[b] if k not in ("anchors",) else v) for k, v in seg.items()}, peaks // R, L)
            accumulate_waveform(acc, units.up_physical(up_np[b]), peaks, x_lo, x_bins)
            for peak, onset, end in zip(peaks, onsets, ends):
                if ig_budget <= 0:
                    break
                columns = peak // R + offsets_tok - anchor0
                ok = (columns >= 0) & (columns < seg["contributing"].shape[1])
                ok[ok] &= seg["contributing"][b, columns[ok]]
                if ok.any():
                    ig_budget -= 1
                    ig_rows += [(b, int(peak), int(o), int(c), int(onset), int(end)) for o, c in zip(np.flatnonzero(ok), columns[ok])]
        if not ig_rows:
            continue
        summaries = model.summary_target(fhr, weight)
        for start in range(0, len(ig_rows), IG_ROWS_PER_CALL):
            part = np.asarray(ig_rows[start:start + IG_ROWS_PER_CALL])
            index = torch.as_tensor(part[:, 0], device=fhr.device)
            columns = torch.as_tensor(part[:, 3], device=fhr.device)
            # Finite raw for the IG path, with the original validity held fixed (a NaN stays masked).
            fhr3, up3, fhr_valid, up_valid = raw.RawReadout.prepare(model, fhr[index], up[index], weight[index])
            for readout, wrapper in wrappers.items():
                with torch.enable_grad():
                    ig = raw.raw_ig(wrapper, (fhr3, fhr3[..., :0], up3), (summaries[index], weight[index]), columns,
                                    torch.zeros_like(columns), stream="up",
                                    baseline=tone[index][:, None, None].expand_as(up3), n_steps=ig_steps,
                                    valid=(fhr_valid, up_valid))
                scale = torch.maximum((ig["value_input"] - ig["value_entry"]).abs(), ig["value_input"].abs()).clamp_min(1e-12)
                attr = ig["map"].flatten(1).float().cpu().numpy()
                rel = (ig["delta"].abs() / scale).float().cpu().numpy()
                completeness[readout] += rel.tolist()
                for (b, peak, o, column, onset, end), values in zip(part, attr):
                    t = anchor0 + column
                    first, last = max(0, R * (t - L + 1) - 1), R * t + R - 1
                    samples = np.arange(first, last + 1)
                    x = np.floor(((samples - peak) / raw.events.FS_RAW - x_lo) / X_BIN_S).astype(int)
                    ok = (x >= 0) & (x < x_bins)
                    acc = sums[str(chunk.iloc[b]["guid"])]
                    np.add.at(acc.ig[readout][o], x[ok], values[samples[ok]])
                    np.add.at(acc.ig_n[readout][o], x[ok], 1.0)
                    magnitude = np.abs(values[samples])
                    rise, fall = (samples >= onset) & (samples < peak), (samples >= peak) & (samples <= end)
                    acc.phase[readout][o] += [magnitude[rise].sum(), magnitude[fall].sum(), magnitude[~(rise | fall)].sum()]
                    if readout == IG_READOUTS[0]:
                        acc.coverage[o] += [(rise | fall).sum(), samples.size]
    elapsed = time.perf_counter() - started
    if not sums or not event_rows:
        return skip_record(NAME, "no contraction was detected on the drawn segments' raw UP")

    # ---- per recording, then over recordings
    guids = list(sums)
    per_rec = {guid: sums[guid].means() for guid in guids}
    stack = {key: np.stack([per_rec[g][key] for g in guids]) for key in per_rec[guids[0]]}
    post = slice(PRE_TOKENS, PRE_TOKENS + L)
    recordings = pd.DataFrame({"guid": guids, labels.CLASS_COLUMN: [classes[g] for g in guids],
                               "n_events": [sums[g].n_events for g in guids]})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        recordings["tracking_attention"] = [tracking_index(np.nanmean(m["attn"][post], axis=1)) for m in per_rec.values()]
        for h in range(heads):
            recordings[f"tracking_attention_head{h}"] = [tracking_index(m["attn"][post, h]) for m in per_rec.values()]
        recordings["tracking_lag_map"] = [tracking_index(m["lag_map"][post]) for m in per_rec.values()]
        recordings["kld_post_peak"] = np.nanmean(stack["kld"][:, post], axis=1)
        recordings["gap_post_peak"] = np.nanmean(stack["gap"][:, post], axis=1)
        recordings["kld_pre_peak"] = np.nanmean(stack["kld"][:, :PRE_TOKENS], axis=1)
        cohort = {key: np.nanmean(value, axis=0) for key, value in stack.items()}
    seed = int(eval_config.get("seed", 0))
    resamples = int(eval_config.get("bootstrap_resamples", shared_stats.DEFAULT_BOOTSTRAP_RESAMPLES))
    metric_columns = ["tracking_attention", *[f"tracking_attention_head{h}" for h in range(heads)],
                      "tracking_lag_map", "kld_post_peak", "gap_post_peak", "kld_pre_peak"]
    summary = pd.DataFrame([{"metric": c, **shared_stats.bootstrap_ci(recordings[c].to_numpy(float), resamples=resamples, seed=seed)}
                            for c in metric_columns])
    k_lag = np.arange(L)
    diagonal = np.stack([np.nansum(np.where(np.abs(k_lag[None, :] - k) <= DIAGONAL_HALF_WIDTH, cohort["attn"][k + PRE_TOKENS], 0.0), axis=-1)
                         if k >= 0 else np.full(heads, np.nan) for k in range(-PRE_TOKENS, L)])
    cohort["diagonal"] = diagonal
    cohort["tracking"] = {"lag_map": tracking_index(cohort["lag_map"][post]),
                          **{f"attention_head{h}": tracking_index(cohort["attn"][post, h]) for h in range(heads)}}
    s_anchor = 4.0 * np.arange(-PRE_TOKENS, L)
    target_time = s_anchor[:, None] + 4.0 * (1 + np.arange(H))[None, :]
    d_axis = np.unique(target_time)
    gain = stack["rmse_base"] - stack["rmse_full"]                                   # (R, K, H)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_rec_gain = np.stack([np.nanmean(np.where(target_time == d, gain, np.nan), axis=(1, 2)) for d in d_axis], axis=1)
    cohort["gain_by_target_time"] = (d_axis, class_contrast.mean_band(per_rec_gain, seed=seed))
    by_class = class_contrast.by_class([classes[g] for g in guids], np.arange(len(guids)))
    cohort["n_by_class"] = {c: int(len(i)) for c, i in by_class.items()}
    for key in ("kld", "gap"):
        cohort[f"{key}_by_class"] = {c: class_contrast.mean_band(stack[key][i], seed=seed) for c, i in by_class.items()}

    # ---- files
    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    lag_s, horizon_s = 4.0 * k_lag, 4.0 * (1 + np.arange(H))
    x_raw = x_lo + X_BIN_S * (np.arange(x_bins) + 0.5)
    pd.DataFrame(event_rows).to_csv(directory / "event_locked_events.csv", index=False)
    recordings.to_csv(directory / "event_locked_recordings.csv", index=False)
    summary.to_csv(directory / "event_locked_summary.csv", index=False)
    curves = pd.DataFrame({"seconds_since_peak": s_anchor, "kld": cohort["kld"], "pred_gap": cohort["gap"],
                           "rmse_level_base_bpm": np.nanmean(cohort["rmse_base"], axis=1),
                           "rmse_level_full_bpm": np.nanmean(cohort["rmse_full"], axis=1),
                           **{f"diagonal_mass_head{h}": diagonal[:, h] for h in range(heads)}})
    curves.to_csv(directory / "event_locked_curves.csv", index=False)
    np.savez_compressed(directory / "event_locked_maps.npz", seconds_since_peak=s_anchor, lag_s=lag_s, horizon_s=horizon_s,
                        raw_seconds=x_raw, ig_offsets_s=4.0 * offsets_tok, guids=np.asarray(guids),
                        gain_target_s=d_axis, gain_by_target_time=cohort["gain_by_target_time"][1][0],
                        **{k: v for k, v in cohort.items() if isinstance(v, np.ndarray)},
                        **{f"per_recording_{k}": v for k, v in stack.items() if k in ("attn", "lag_map")})
    pd.DataFrame({"target_seconds_since_peak": d_axis, "level_rmse_gain_bpm": cohort["gain_by_target_time"][1][0],
                  "ci_lo": cohort["gain_by_target_time"][1][1], "ci_hi": cohort["gain_by_target_time"][1][2]}
                 ).to_csv(directory / "event_locked_gain_by_target_time.csv", index=False)
    phase_rows = [{"readout": r, "anchor_after_peak_s": 4.0 * offsets_tok[o], "n_recordings": int(np.isfinite(stack[f"phase_{r}"][:, o, 0]).sum()),
                   **dict(zip(("share_rise", "share_fall", "share_outside"), cohort[f"phase_{r}"][o])),
                   "window_fraction_in_event": cohort["coverage"][o]}
                  for r in IG_READOUTS for o in range(offsets_tok.size)]
    pd.DataFrame(phase_rows).to_csv(directory / "event_locked_ig_phase.csv", index=False)
    files = ["event_locked_ig_phase.csv", "event_locked_gain_by_target_time.csv", "event_locked_events.csv", "event_locked_recordings.csv", "event_locked_summary.csv",
             "event_locked_curves.csv", "event_locked_maps.npz"]
    if len({c for c in classes.values()}) >= 2:
        stats, pairwise = class_contrast.class_tests(recordings, {NAME: metric_columns}, seed=seed)
        stats.to_csv(directory / "event_locked_class_stats.csv", index=False)
        pairwise.to_csv(directory / "event_locked_class_pairwise.csv", index=False)
        files += ["event_locked_class_stats.csv", "event_locked_class_pairwise.csv"]
    title = (f"Contraction-locked maps: {len(event_rows)} contractions, {len(guids)} recordings, "
             f"{len(rows)} segments (mean-decoded; IG {ig_steps} steps from the resting tone)")
    figure = build_figure(cohort, s_anchor=s_anchor, lag_s=lag_s, x_raw=x_raw, offsets_s=4.0 * offsets_tok,
                          horizon_s=horizon_s, title=title)
    files.append(str(figures.render_figure(figure, directory / NAME, tight=False).name))

    point = summary.set_index("metric")["point"]
    rel = {r: np.asarray(v, dtype=float) for r, v in completeness.items()}
    all_rel = np.concatenate(list(rel.values())) if any(v.size for v in rel.values()) else np.zeros(0)
    headline = {
        "tracking_attention": float(point["tracking_attention"]),
        "tracking_attention_best_head": float(max(point[f"tracking_attention_head{h}"] for h in range(heads))),
        "tracking_lag_map": float(point["tracking_lag_map"]),
        "gain_peak_target_s": float(d_axis[np.nanargmax(cohort["gain_by_target_time"][1][0])])
        if np.isfinite(cohort["gain_by_target_time"][1][0]).any() else None,
        "ig_completeness_max": float(all_rel.max()) if all_rel.size else None,
    }
    logger.info(f"{NAME}: tracking attention {headline['tracking_attention']:.3f}, lag map "
                f"{headline['tracking_lag_map']:.3f}; {len(event_rows)} contractions, {elapsed:.1f} s")
    return {
        "n_samples": int(len(rows)),
        "composition": {"n_recordings": len(guids), "n_contractions": len(event_rows),
                        "n_ig_rows": int(sum(v.size for v in rel.values()))},
        "plan": {"capped": True, "cap_segments": cap, "cap_ig_events": ig_events, "ig_offsets_s": (4 * offsets_tok).tolist(),
                 "ig_steps": ig_steps, "ig_baseline": "resting tone (10th percentile of valid UP, whole segment)",
                 "decode": "latent means", "seconds_since_peak": [float(s_anchor[0]), float(s_anchor[-1])]},
        "summary": summary.to_dict(orient="records"),
        "ig_completeness": {r: {"median": float(np.median(v)) if v.size else None, "max": float(v.max()) if v.size else None,
                                "n_rows_over_tolerance": int((v > COMPLETENESS_TOLERANCE).sum()), "tolerance": COMPLETENESS_TOLERANCE}
                            for r, v in rel.items()},
        "ig_phase_shares": phase_rows,
        "cost": {"elapsed_s": float(elapsed), "n_segments": int(len(rows))},
        "headline": headline,
        "files": files,
    }
