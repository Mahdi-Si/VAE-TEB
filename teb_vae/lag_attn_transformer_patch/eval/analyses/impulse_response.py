r"""``impulse_response`` -- the model's effective FIR kernel, read off a synthetic contraction (E1-I).

**Question.** With local K/V the lag attention is a data-dependent FIR filter over the UP history
(plan A.3). What forecast does one contraction elicit, as a function of delay? Inject a bump into raw
UP at a known time and read the response against

$$d = 4(t + 1 + \tau) - t_0 \ \text{s},$$

the delay from the injected peak $t_0$ (s) to the target patch $t + 1 + \tau$ of anchor $t$.

**Injection.** A raised cosine $A\,\tfrac12(1 + \cos 2\pi x/W)$, $|x| < W/2$, added to raw
loader-z UP with its peak on the first sample of token ``tok0`` ($t_0 = 4\,\mathrm{tok0}$ s, so
$d = 4(\ell_0 + 1 + \tau)$ with $\ell_0 = t - \mathrm{tok0}$ the lag the peak sits at).

* $A$ = median over the segment's detected contractions (``events.detect_contractions``) of
  UP at the peak minus the resting tone; $W$ = median onset-to-end span / 0.8 (the detector's flanks
  sit at 10 % of prominence, and a raised cosine's 10 % span is 0.795 W), clipped to
  :data:`WIDTH_LIMITS_S`. A segment with no detection takes :data:`DEFAULT_AMPLITUDE_Z` and
  :data:`DEFAULT_WIDTH_S`. The source of each value is counted in ``design``.
* Injection tokens: :data:`DEFAULT_INJECTIONS` evenly spaced from (first anchor + the widest
  half-width + 2), so a few anchors always precede the bump, to (last anchor − max lag), so every
  lag after the peak is decoded.
* Two backgrounds, each paired with itself without the bump: ``segment`` (the real UP) and ``rest``
  (UP held flat at the segment's resting tone, ``raw.resting_tone``), the cleaner probe.

**Readouts** (mean-decoded posterior, so both arms share the latent exactly; the prior and the base
forecast are target-only and do not move): Δ level in bpm (``SummaryUnits.level_delta_bpm``);
Δ variability as the log-ratio of rms-Δ (Δŷ·scale₁, never bpm); ΔK_t; Δ attention per head and
Δ ``source_kl_lag_map`` at the lag pointing at the peak, ℓ₀ = t − tok0. Level and variability are
binned on d for anchors with 0 ≤ ℓ₀ < L (the peak inside the window); KL and attention on ℓ₀.
Averaged over injections within a segment, then within a recording; the spread is the std over
recordings.

**Causality sanity.** Anchors whose last raw sample precedes the bump's first non-zero sample must
not move at all: ``causality`` holds the largest |Δ| there per readout (0 up to float noise).

**Cost.** Segments × 2 backgrounds × (1 + injections) dense forwards (64 × 12 = 768 segment-forwards
at the defaults). Caps: ``caps.impulse_response_segments`` (64), ``caps.impulse_response_injections`` (5).
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval.attributions import EXAMPLE_PAGE_WIDTH, signed_log_norm, symlog_legend
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.analyses import skip_record

NAME = "impulse_response"

#: ``(headline_name, key)`` pairs registered as ``("impulse_response", "headline", key)``. Read off
#: the ``rest`` background, the cleaner probe.
HEADLINE: Tuple[Tuple[str, str], ...] = (
    ("impulse_level_peak_delay_s", "level_peak_delay_s"),
    ("impulse_level_peak_bpm", "level_peak_bpm"),
    ("impulse_causality_max_abs", "causality_max_abs"),
)

CAP_SEGMENTS, DEFAULT_SEGMENTS = "impulse_response_segments", 64
CAP_INJECTIONS, DEFAULT_INJECTIONS = "impulse_response_injections", 5
SEED_OFFSET = 41

#: Fallback bump when a segment has no detected contraction: loader-z above resting tone, and the
#: full raised-cosine width (10 %-level span ≈ 72 s, a mid-labour contraction).
DEFAULT_AMPLITUDE_Z = 2.0
DEFAULT_WIDTH_S = 90.0
WIDTH_LIMITS_S = (40.0, 160.0)
FLANK_SPAN_FRAC = 0.8

SECONDS_PER_STEP = 4.0
#: Anchors before the peak kept on the KL axis: past the widest bump's rising flank, so the causal
#: zero is drawn. The injection grid guarantees these anchors exist.
PRE_TOKENS = math.ceil(WIDTH_LIMITS_S[1] / 2 / SECONDS_PER_STEP) + 2
BACKGROUNDS = ("segment", "rest")
CAVEAT = (
    "Synthetic UP bump; d = 4(t+1+tau) - t0 is peak-to-target-patch delay. Model sensitivity, not a "
    "physiological latency."
)


# =============================================================================
# Design
# =============================================================================
def bump_design(up: np.ndarray, tone: np.ndarray, *, fs: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Per-segment amplitude (loader z above tone), full width (s) and where they came from."""
    amplitude = np.full(len(up), DEFAULT_AMPLITUDE_Z)
    width = np.full(len(up), DEFAULT_WIDTH_S)
    source = ["default"] * len(up)
    for b, trace in enumerate(up):
        found = raw.events.detect_contractions(trace, fs=fs, valid=np.isfinite(trace))
        if not len(found["peak_raw"]):
            continue
        amp = float(np.median(trace[found["peak_raw"]] - tone[b]))
        if amp > 0.0:
            span = np.median(found["end_raw"] - found["onset_raw"]) / fs / FLANK_SPAN_FRAC
            amplitude[b], width[b], source[b] = amp, float(np.clip(span, *WIDTH_LIMITS_S)), "detected"
    return amplitude, width, source


def injection_tokens(first_anchor: int, last_anchor: int, n_lags: int, n: int) -> np.ndarray:
    """Evenly spaced peak tokens with anchors before the widest bump and every lag after the peak decoded."""
    lo = first_anchor + PRE_TOKENS
    hi = max(lo, last_anchor - (n_lags - 1))
    return np.unique(np.linspace(lo, hi, max(1, n)).round().astype(np.int64))


def raised_cosine(length: int, centre: int, width: torch.Tensor, amplitude: torch.Tensor) -> torch.Tensor:
    """``(B, length)`` bumps peaking at sample ``centre``, full width ``width`` samples ``(B,)``."""
    x = (torch.arange(length, device=width.device, dtype=width.dtype)[None] - centre) / width[:, None]
    return amplitude[:, None] * torch.where(x.abs() < 0.5, 0.5 * (1.0 + torch.cos(2.0 * math.pi * x)), 0.0)


# =============================================================================
# Readouts
# =============================================================================
@torch.no_grad()
def readouts(model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor) -> Dict[str, torch.Tensor]:
    """One dense forward, every readout gathered at the decoded anchors; the forecast mean-decoded."""
    out = raw.forward_raw(model, fhr, up, weight)
    idx = out["anchor_index"]

    def at(x: torch.Tensor) -> torch.Tensor:
        return x.gather(1, idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(-1, -1, *x.shape[2:]))

    mu, _ = raw.decode_at(model, out, out["mu_post"])
    return {
        "mu": mu, "kl": at(out["kld_per_t"]), "kl_head": at(out["kld_per_t_per_head"]),
        "attn": at(out["attn_weights"]), "klmap": at(out["source_kl_lag_map"]),
        "anchors": idx, "valid": out["anchor_valid"],
    }


def response(ref: Dict[str, torch.Tensor], inj: Dict[str, torch.Tensor], tok0: int, first_live: torch.Tensor,
             units: raw.SummaryUnits, raw_per_step: int) -> Dict[str, np.ndarray]:
    """The paired differences of one injection, on the host: level (bpm), variability (log-ratio), KL, attention."""
    anchors, valid = ref["anchors"], ref["valid"]
    n_lags = ref["attn"].shape[-1]
    lag0 = anchors - tok0
    pick = lag0.clamp(0, n_lags - 1)
    dattn_all = inj["attn"] - ref["attn"]
    dmu = inj["mu"] - ref["mu"]
    dlevel = units.level_delta_bpm(dmu[..., 0])
    dkl = inj["kl"] - ref["kl"]
    pre = valid & (anchors * raw_per_step + raw_per_step - 1 < first_live[:, None])

    def worst(x: torch.Tensor) -> float:
        x = x.abs().flatten(2).amax(-1) if x.dim() > 2 else x.abs()
        return float(torch.where(pre, x, torch.zeros_like(x)).max())

    host = lambda x: x.detach().double().cpu().numpy()
    return {
        "lag0": host(lag0).astype(np.int64), "valid": host(valid).astype(bool),
        "level": host(dlevel), "variability": host(dmu[..., 1] * units.scale[1]),
        "kl": host(dkl), "kl_head": host(inj["kl_head"] - ref["kl_head"]),
        "attn": host(dattn_all.gather(-1, pick[..., None, None].expand(-1, -1, dattn_all.shape[2], 1)).squeeze(-1)),
        "klmap": host((inj["klmap"] - ref["klmap"]).gather(-1, pick[..., None]).squeeze(-1)),
        "causality": {"level_bpm": worst(dlevel), "kl_nats": worst(dkl), "attention": worst(dattn_all)},
    }


class Bins:
    """Per-segment sums and counts on one integer axis, so a mean over injections is one division."""

    def __init__(self, n_segments: int, n_bins: int, extra: Tuple[int, ...] = ()) -> None:
        self.sum = np.zeros((n_segments, n_bins, *extra))
        self.count = np.zeros((n_segments, n_bins, *([1] * len(extra))))

    def add(self, index: np.ndarray, keep: np.ndarray, values: np.ndarray) -> None:
        """Add ``values[b, ...]`` at bin ``index[b, ...]`` where ``keep``; ``index`` (B, ...) broadcast to values' lead."""
        rows = np.broadcast_to(np.arange(index.shape[0]).reshape(-1, *([1] * (index.ndim - 1))), index.shape)
        lead = index.ndim
        flat = values.reshape(*values.shape[:lead], -1)
        np.add.at(self.sum.reshape(*self.sum.shape[:2], -1), (rows[keep], index[keep]), flat[keep])
        np.add.at(self.count.reshape(*self.count.shape[:2], -1), (rows[keep], index[keep]), 1.0)

    def mean(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.sum / np.maximum(self.count, 1.0), np.nan)


@torch.no_grad()
def score_batch(model: Any, fhr: torch.Tensor, up: torch.Tensor, weight: torch.Tensor, *,
                n_injections: int, units: raw.SummaryUnits, want_example: bool) -> Dict[str, Any]:
    """Every background × injection of one batch, reduced to per-segment binned means."""
    r = int(model.raw_per_step)
    batch, length = up.shape
    tone = raw.resting_tone(up, weight, raw_per_step=r, validity=model.source_validity)
    amplitude, width_s, source = bump_design(up.double().cpu().numpy(), tone.double().cpu().numpy(), fs=raw.events.FS_RAW)
    amp_t = torch.as_tensor(amplitude, dtype=up.dtype, device=up.device)
    width_t = torch.as_tensor(width_s * raw.events.FS_RAW, dtype=up.dtype, device=up.device)
    backgrounds = {"segment": up, "rest": tone[:, None].expand(-1, length).contiguous()}

    refs = {name: readouts(model, fhr, bg, weight) for name, bg in backgrounds.items()}
    anchors = refs["segment"]["anchors"][0]
    n_lags, horizon, heads = refs["segment"]["attn"].shape[-1], refs["segment"]["mu"].shape[2], refs["segment"]["attn"].shape[2]
    tokens = injection_tokens(int(anchors.min()), int(anchors.max()), n_lags, n_injections)
    tau = np.arange(horizon)
    bins = {
        name: {"level": Bins(batch, n_lags + horizon), "variability": Bins(batch, n_lags + horizon),
               "kl": Bins(batch, PRE_TOKENS + n_lags), "kl_head": Bins(batch, PRE_TOKENS + n_lags, (heads,)),
               "attn": Bins(batch, n_lags, (heads,)), "klmap": Bins(batch, n_lags)}
        for name in backgrounds
    }
    causality = {"level_bpm": 0.0, "kl_nats": 0.0, "attention": 0.0}
    example = None
    for k, tok0 in enumerate(tokens):
        bump = raised_cosine(length, int(tok0) * r, width_t, amp_t)
        first_live = (bump != 0).to(torch.int64).argmax(-1)
        for name, bg in backgrounds.items():
            res = response(refs[name], readouts(model, fhr, bg + bump, weight), int(tok0), first_live, units, r)
            causality = {key: max(causality[key], res["causality"][key]) for key in causality}
            lag0, valid = res["lag0"], res["valid"]
            seen = valid & (lag0 >= 0) & (lag0 < n_lags)
            delay = lag0[..., None] + 1 + tau
            spread = np.broadcast_to(seen[..., None], delay.shape)
            bins[name]["level"].add(delay, spread, res["level"])
            bins[name]["variability"].add(delay, spread, res["variability"])
            around = valid & (lag0 >= -PRE_TOKENS) & (lag0 < n_lags)
            bins[name]["kl"].add(lag0 + PRE_TOKENS, around, res["kl"])
            bins[name]["kl_head"].add(lag0 + PRE_TOKENS, around, res["kl_head"])
            bins[name]["attn"].add(lag0, seen, res["attn"])
            bins[name]["klmap"].add(lag0, seen, res["klmap"])
            if want_example and name == "segment" and k == len(tokens) // 2:
                example = {
                    "tok0": int(tok0), "anchors": anchors.cpu().numpy(), "level": res["level"][0],
                    "kl": res["kl"][0], "attn": res["attn"][0], "lag0": lag0[0],
                    "up": units.up_physical(up[0].double().cpu().numpy()),
                    "up_injected": units.up_physical((up[0] + bump[0]).double().cpu().numpy()),
                    "fhr_bpm": fhr[0].double().cpu().numpy() * units.fhr_std + units.fhr_mean,
                    "span_s": (float(first_live[0]) / raw.events.FS_RAW,
                               float(length - 1 - (bump[0].flip(0) != 0).to(torch.int64).argmax()) / raw.events.FS_RAW),
                    "amplitude_z": float(amplitude[0]), "width_s": float(width_s[0]),
                }
    return {
        "bins": {name: {key: b.mean() for key, b in group.items()} for name, group in bins.items()},
        "amplitude_z": amplitude, "width_s": width_s, "source": source, "tokens": tokens,
        "causality": causality, "example": example, "n_lags": n_lags, "horizon": horizon, "heads": heads,
    }


# =============================================================================
# Reduction
# =============================================================================
def recording_spread(values: np.ndarray, guids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Mean and std over recordings of per-recording means (axis 0 is segments)."""
    with warnings.catch_warnings():  # an all-NaN bin (no anchor reached it) stays NaN, silently
        warnings.simplefilter("ignore", RuntimeWarning)
        per_recording = np.stack([np.nanmean(values[guids == g], axis=0) for g in pd.unique(guids)])
        return np.nanmean(per_recording, axis=0), np.nanstd(per_recording, axis=0), len(per_recording)


def kernel_stats(x_s: np.ndarray, k: np.ndarray) -> Dict[str, Optional[float]]:
    """The extremum of a signed kernel, its sign, and the width of the run around it above half its size."""
    finite = np.isfinite(k)
    if not finite.any():
        return {"peak_delay_s": None, "peak": None, "sign": None, "width_at_half_max_s": None}
    i = int(np.nanargmax(np.abs(np.where(finite, k, np.nan))))
    above = finite & (np.sign(k) == np.sign(k[i])) & (np.abs(k) >= abs(k[i]) / 2.0)
    lo = hi = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(k) - 1 and above[hi + 1]:
        hi += 1
    step = float(np.median(np.diff(x_s))) if len(x_s) > 1 else 0.0
    return {"peak_delay_s": float(x_s[i]), "peak": float(k[i]), "sign": int(np.sign(k[i])),
            "width_at_half_max_s": float(x_s[hi] - x_s[lo] + step)}


# =============================================================================
# Figures
# =============================================================================
def build_kernel_figure(curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]], heads: int) -> Any:
    """Level and variability kernels against d; KL and per-head attention against time since the peak."""
    figure, axes = figures.new_figure(4, 1, height_per_row=2.1)
    styles = {"segment": ("-", figures.COLOR_BLUE), "rest": ("--", figures.COLOR_VERMILLION)}
    panels = (
        ("level", "delay d (s)", "Δ level (bpm)", "Level kernel k(d)"),
        ("variability", "delay d (s)", "Δ log rms-Δ", "Variability kernel"),
        ("kl", r"time since injected peak at the anchor, $4\ell_0$ (s)", "ΔK_t (nats)", "KL response"),
    )
    for ax, (key, xlabel, ylabel, title) in zip(axes[:3, 0], panels):
        drawn = []
        for name, (style, colour) in styles.items():
            x, mean, std = curves[name][key]
            ax.plot(x, mean, linestyle=style, color=colour, linewidth=figures.LINE_REGULAR, label=f"{name} background")
            ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)
            drawn += [mean - std, mean + std]
        ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        symlog_legend(ax, *drawn, legend=key == "level")
        figures.style_axes(ax)
    ax = axes[3, 0]
    palette = (figures.COLOR_BLUE, figures.COLOR_ORANGE, figures.COLOR_GREEN, figures.COLOR_PURPLE)
    drawn = []
    for name, (style, _colour) in styles.items():
        x, mean, _std = curves[name]["attn"]
        for m in range(heads):
            ax.plot(x, mean[:, m], linestyle=style, color=palette[m % len(palette)], linewidth=figures.LINE_THIN,
                    label=f"head {m}" if name == "segment" else None)
            drawn.append(mean[:, m])
    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=figures.LINE_HAIRLINE)
    ax.set(xlabel=r"time since injected peak at the anchor, $4\ell_0$ (s)", ylabel=r"Δ attention at $\ell_0$",
           title="Attention on the injected peak, per head (solid: segment, dashed: rest)")
    symlog_legend(ax, *drawn, ncol=heads)
    figures.style_axes(ax)
    figures.caveat_note(figure, CAVEAT)
    return figure


def build_example_figure(example: Dict[str, Any], heads: int) -> Any:
    """One segment, one injection, the samples-page layout: every row on one stored-time axis."""
    rows = (("up", 0.8), ("fhr", 0.8), ("level", 1.3), ("kl", 0.8), ("attn", 0.8))
    height = sum(h for _, h in rows) * 2.2 + 0.8
    figure = plt.figure(figsize=(EXAMPLE_PAGE_WIDTH, height))
    bottom = max(0.02, 0.35 / height) + figures.caveat_note(figure, CAVEAT)
    grid = GridSpec(len(rows), 2, figure=figure, height_ratios=[h for _, h in rows], width_ratios=[1.0, 0.022],
                    left=0.065, right=0.93, top=1.0 - 0.6 / height, bottom=bottom, hspace=0.55, wspace=0.09)
    seconds = np.arange(len(example["up"])) / raw.events.FS_RAW
    anchor_s = SECONDS_PER_STEP * (example["anchors"] + 1)
    peak_s = SECONDS_PER_STEP * example["tok0"]
    axes = []
    for i, (name, _h) in enumerate(rows):
        ax = figure.add_subplot(grid[i, 0], sharex=axes[0] if axes else None)
        cax = figure.add_subplot(grid[i, 1])
        axes.append(ax)
        if name == "up":
            ax.plot(seconds, example["up"], color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN, label="UP")
            ax.plot(seconds, example["up_injected"], color=figures.COLOR_VERMILLION, linewidth=figures.LINE_THIN,
                    label=f"UP + bump ({example['amplitude_z']:.2f} z, {example['width_s']:.0f} s)")
            ax.set_ylabel("UP")
            ax.legend(loc="upper left", fontsize=figures.FONT_TINY)
        elif name == "fhr":
            ax.plot(seconds, example["fhr_bpm"], color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN)
            ax.set_ylabel("FHR (bpm)")
        elif name == "level":
            field = example["level"]
            norm = signed_log_norm(field)
            image = ax.imshow(field.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm, interpolation="nearest",
                              extent=(anchor_s[0] - 2, anchor_s[-1] + 2, 2, 4 * field.shape[1] + 2))
            figure.colorbar(image, cax=cax).set_label("Δ level (bpm)")
            ax.set_ylabel("horizon (s)")
        elif name == "kl":
            ax.plot(anchor_s, example["kl"], color=figures.COLOR_BLUE, linewidth=figures.LINE_REGULAR)
            ax.set_ylabel("ΔK_t (nats)")
            symlog_legend(ax, example["kl"], legend=False)
        else:
            attn = np.where((example["lag0"] >= 0)[:, None], example["attn"], np.nan)
            for m in range(heads):
                ax.plot(anchor_s, attn[:, m], linewidth=figures.LINE_THIN, label=f"head {m}")
            ax.set_ylabel(r"Δ attention at $\ell_0$")
            symlog_legend(ax, attn, ncol=heads)
        if name != "level":
            cax.axis("off")
            ax.axvspan(*example["span_s"], color=figures.COLOR_LIGHT_GRAY, alpha=0.4, linewidth=0)
        ax.axvline(peak_s, color=figures.COLOR_VERMILLION, linewidth=figures.LINE_HAIRLINE)
        figures.style_axes(ax)
    axes[-1].set_xlabel("stored time (s); maps and curves at the anchor's last sample")
    axes[0].set_xlim(0.0, seconds[-1])
    axes[0].set_title(f"Injected contraction, peak at {peak_s:.0f} s (segment background)")
    figures.mark_laid_out(figure)
    return figure


# =============================================================================
# Entry
# =============================================================================
def run_impulse_response_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Inject, read the paired responses, bin on delay, reduce over recordings, draw and summarise."""
    del probe
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if task is None or loader is None:
        return skip_record(NAME, "the injection re-runs the model on edited raw UP; an offline re-run has no model")
    caps = eval_config.get("caps") or {}
    cap = int(caps.get(CAP_SEGMENTS) or DEFAULT_SEGMENTS)
    n_injections = int(caps.get(CAP_INJECTIONS) or DEFAULT_INJECTIONS)
    seed = int(eval_config.get("seed", 0)) + SEED_OFFSET
    rows = raw.draw_segments(context, cap=cap, seed=seed)
    if rows.empty:
        return skip_record(NAME, "no scored segment could be located in the loader's dataset")

    model = task.orig_model
    units = raw.SummaryUnits.from_model(model, context.config, loader)
    records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for _chunk, batch in raw.segment_batches(task, loader, rows):
        fhr, up, weight = raw.batch_signals(batch)
        records.append(score_batch(model, fhr, up, weight, n_injections=n_injections, units=units,
                                   want_example=not records))
    elapsed_s = time.perf_counter() - started

    first = records[0]
    n_lags, horizon, heads = first["n_lags"], first["horizon"], first["heads"]
    guids = rows["guid"].astype(str).to_numpy()
    delay_s = SECONDS_PER_STEP * np.arange(n_lags + horizon)
    since_s = SECONDS_PER_STEP * (np.arange(PRE_TOKENS + n_lags) - PRE_TOKENS)
    lag_s = SECONDS_PER_STEP * np.arange(n_lags)
    axes = {"level": delay_s, "variability": delay_s, "kl": since_s, "kl_head": since_s, "attn": lag_s, "klmap": lag_s}
    curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    table: List[pd.DataFrame] = []
    arrays: Dict[str, np.ndarray] = {}
    n_recordings = 0
    for name in BACKGROUNDS:
        curves[name] = {}
        for key, x in axes.items():
            values = np.concatenate([r["bins"][name][key] for r in records])
            arrays[f"{name}_{key}"] = values
            mean, std, n_recordings = recording_spread(values, guids)
            keep = slice(1, None) if x is delay_s else slice(None)  # d = 0 is never reached (ℓ₀ + 1 + τ ≥ 1)
            x, mean, std = x[keep], mean[keep], std[keep]
            curves[name][key] = (x, mean, std)
            # Per head for the head-resolved readouts, head -1 for the pooled ones.
            for h, (m, s) in ({-1: (mean, std)} if mean.ndim == 1 else dict(enumerate(zip(mean.T, std.T)))).items():
                table.append(pd.DataFrame({"background": name, "readout": key, "head": h, "x_s": x,
                                           "mean": m, "std": s, "n_recordings": n_recordings}))
    directory = Path(output_dir) / NAME
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(table, ignore_index=True).to_csv(directory / "impulse_response_kernel.csv", index=False)
    np.savez_compressed(directory / "impulse_response_per_segment.npz", guid=guids, delay_s=delay_s, since_s=since_s,
                        lag_s=lag_s, **arrays)

    amplitude = np.concatenate([r["amplitude_z"] for r in records])
    width = np.concatenate([r["width_s"] for r in records])
    sources = sum((r["source"] for r in records), [])
    causality = {key: max(r["causality"][key] for r in records) for key in first["causality"]}
    summary: Dict[str, Any] = {"backgrounds": {}}
    for name in BACKGROUNDS:
        level = kernel_stats(*curves[name]["level"][:2])
        attn_x, attn_mean, _ = curves[name]["attn"]
        summary["backgrounds"][name] = {
            "level_bpm": level,
            "variability_log_ratio": kernel_stats(*curves[name]["variability"][:2]),
            "kl_nats": kernel_stats(*curves[name]["kl"][:2]),
            "attention_peak_since_s_per_head": [kernel_stats(attn_x, attn_mean[:, m])["peak_delay_s"] for m in range(heads)],
        }
    rest = summary["backgrounds"]["rest"]["level_bpm"]
    summary.update({
        "causality_max_abs": causality,
        "design": {
            "n_detected": int(sources.count("detected")), "n_default": int(sources.count("default")),
            "amplitude_z_median": float(np.median(amplitude)), "width_s_median": float(np.median(width)),
            "amplitude_up_units_median": float(np.median(amplitude) * units.up_std),
            "injection_tokens": first["tokens"].tolist(), "peak_sample": "first sample of the injection token",
        },
    })
    (directory / "impulse_response_summary.json").write_text(json.dumps(summary, indent=2))

    files = ["impulse_response_kernel.csv", "impulse_response_per_segment.npz", "impulse_response_summary.json"]
    files.append(figures.render_figure(build_kernel_figure(curves, heads), directory / "impulse_response_kernel").name)
    if first["example"] is not None:
        files.append(figures.render_figure(build_example_figure(first["example"], heads),
                                           directory / "impulse_response_example").name)
    return {
        "n_samples": int(len(rows)),
        "composition": {"n_recordings": int(n_recordings)},
        "plan": {"capped": True, "cap": cap, "injections": n_injections, "seed": seed,
                 "backgrounds": list(BACKGROUNDS), "decode": "latent mean, both arms"},
        "cost": {"elapsed_s": float(elapsed_s), "segment_forwards": int(len(rows) * len(BACKGROUNDS) * (1 + len(first["tokens"])))},
        "summary": summary,
        "headline": {
            "level_peak_delay_s": rest["peak_delay_s"], "level_peak_bpm": rest["peak"],
            "causality_max_abs": max(causality.values()),
        },
        "files": [str(f) for f in files],
    }
