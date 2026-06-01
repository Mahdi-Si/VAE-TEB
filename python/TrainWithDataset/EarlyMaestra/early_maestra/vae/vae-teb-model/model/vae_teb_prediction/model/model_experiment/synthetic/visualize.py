r"""Visualisation utilities for cached synthetic TE datasets (v2; Decision V2-D2).

Renders a human-readable preview of a generated benchmark so the data can be
inspected before training and reused with confidence. The preview is a single
multi-panel ``preview.pdf`` (plus a ``preview.png``) written next to the cached
``.npz`` splits.

Public API:
    make_preview: build the one-page ``preview.{pdf,png}`` for a cached dataset.
    make_dataset_gallery: render a multi-figure ``figures/`` gallery (input /
        source heatmaps, native model fields, the forecast-target window, the
        lag structure, value distributions) so the inputs, outputs and transfer
        structure of a dataset can be inspected in full detail rather than from
        the single condensed preview.

The preview is **benchmark-aware**: panels 1, 2 and 5 are common to every
benchmark, while panels 3, 4 and 6 are dispatched on ``meta["benchmark"]``:

    * G1 / G1-rev (Gaussian state-space oscillator): target PSD overlay vs
      i.i.d. baseline, oscillator phase-portrait $(s_t, s_{t-1})$, and a
      Gaussian state-space parameter summary.
    * G2 (smooth AR(1)-ARX): delay-alignment scatter, per-channel lagged
      cross-correlation, and an ARX parameter summary.
    * G3 (slow categorical regime-switch): regime-strip imshow, target
      template overlay with regime colour-banding, and a parameter summary.

Note:
    Generators run with ``standardize=True``, which z-scores every channel to
    unit variance, so a raw variance-ratio bar carries no contrast. The
    standardisation-invariant **lagged cross-correlation** (G1 / G2) and a
    regime-strip imshow (G3) are plotted instead.

All figures use the shared publication style in :mod:`plot_style`.

Used by :mod:`build_dataset` and runnable standalone against any cached dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    plot_style as ps,
)

ps.apply_style()

_MAX_LAG = 90  # the model's attention window is lags 0..90 (max_lag + 1 = 91).

# Stable colour roles across every preview panel.
_C_INFORMATIVE = ps.COLOR_BLUE      # informative channels / lag-group 1
_C_DISTRACTOR = ps.COLOR_ORANGE     # non-informative distractor channels
_C_SOURCE = ps.COLOR_GREEN          # source-stream traces / lag-group 2
_C_BAND = ps.COLOR_VERMILLION       # true source-lag band
_C_DELAY = ps.COLOR_GRAY            # delay reference line
_C_ANALYTIC = ps.COLOR_BLACK        # analytic reference lines


def _load_split(npz_path: Path, n_sub: int) -> Dict[str, np.ndarray]:
    """Load the first ``n_sub`` samples of a cached split into RAM.

    Args:
        npz_path: Path to a ``{train,val,test}.npz``.
        n_sub: Number of leading samples to load (keeps the preview cheap).

    Returns:
        Dict with ``Y`` $(n, T, 87)$, ``U`` $(n, T, 101)$ and ``weight``
        $(n, T)$ as in-memory ``float32`` arrays.
    """
    with np.load(npz_path) as npz:
        n = min(int(n_sub), int(npz["fhr_st"].shape[0]))
        fhr_st = np.asarray(npz["fhr_st"][:n], dtype=np.float32)
        fhr_ph = np.asarray(npz["fhr_ph"][:n], dtype=np.float32)
        up_st = np.asarray(npz["up_st"][:n], dtype=np.float32)
        up_ph = np.asarray(npz["up_ph"][:n], dtype=np.float32)
        weight = np.asarray(npz["weight"][:n], dtype=np.float32)
    return {
        "Y": np.concatenate([fhr_st, fhr_ph], axis=-1),
        "U": np.concatenate([up_st, up_ph], axis=-1),
        "weight": weight,
    }


def _primary_delay(meta: Dict[str, Any]) -> int:
    """Return a representative delay for the trace panels.

    Each v2 benchmark exposes its delay under a different key: G1 / G1-rev
    carry a per-channel ``delays`` list (we return the first), G2 carries the
    scalar ``delay``, and G3 carries the source reveal-lead ``delta``. Under
    the variable per-sample-delay regime ``delay`` / ``delays`` may be absent
    or ``None`` and ``delay_max`` is used as the representative value.

    Args:
        meta: The dataset metadata dict.

    Returns:
        The benchmark-appropriate primary delay (0 if unset).
    """
    benchmark = str(meta.get("benchmark", ""))
    if benchmark in ("G1", "G1-rev"):
        delays = meta.get("delays") or []
        if delays:
            return int(delays[0])
        dmax = meta.get("delay_max")
        return int(dmax) if dmax is not None else 0
    if benchmark == "G3":
        return int(meta.get("delta", 0))
    # G2 family: fixed scalar ``delay`` or variable ``delay_max``.
    d = meta.get("delay")
    if d is None:
        d = meta.get("delay_max")
    return int(d) if d is not None else 0


def _lagged_corr_per_channel(
    U: np.ndarray, Y: np.ndarray, delay: int
) -> np.ndarray:
    r"""Pooled Pearson correlation $\mathrm{corr}(U_j(t-D),\,Y_j(t))$ per channel.

    Args:
        U: Source tensor $(n, T, C_u)$.
        Y: Target tensor $(n, T, C_y)$.
        delay: Source-to-target delay $D$.

    Returns:
        Correlation per target channel, shape $(C_y,)$ (each $Y$ channel
        paired with the source channel of the same index).
    """
    c_y = Y.shape[-1]
    src = U[:, : U.shape[1] - delay, :c_y].reshape(-1, c_y)
    tgt = Y[:, delay:, :].reshape(-1, c_y)
    src = src - src.mean(axis=0, keepdims=True)
    tgt = tgt - tgt.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(src, axis=0) * np.linalg.norm(tgt, axis=0)
    return np.where(denom > 0.0, (src * tgt).sum(axis=0) / (denom + 1e-12), 0.0)


def _panels_traces(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                   meta: Dict[str, Any]) -> None:
    """Fill panels 1-2 (target / source traces) -- common to every benchmark.

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    t_axis = np.arange(T)
    delay = _primary_delay(meta)
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    distractor = M if M < c_y else 0  # falls back to ch 0 in the easy variant.

    ax = axes[0, 0]
    ax.plot(t_axis, Y[0, :, 0], lw=0.9, color=_C_INFORMATIVE,
            label="informative ch 0")
    if M < c_y:
        ax.plot(
            t_axis, Y[0, :, distractor], lw=0.9, color=_C_DISTRACTOR,
            alpha=0.8, label=f"distractor ch {distractor}",
        )
    ax.axvline(delay, color=_C_DELAY, ls="--", lw=0.9, label=f"delay D={delay}")
    ax.set_title("Target $Y$ -- sample 0")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    ax = axes[0, 1]
    ax.plot(t_axis, U[0, :, 0], lw=0.9, color=_C_SOURCE, label="source ch 0")
    ax.axvline(delay, color=_C_DELAY, ls="--", lw=0.9, label=f"delay D={delay}")
    ax.set_title("Source $U$ -- sample 0")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right")
    ps.style_axes(ax)


def _panel_lag_band(ax: plt.Axes, meta: Dict[str, Any]) -> None:
    """Fill panel 5 (true source-lag band over the attention axis).

    Handles the single contiguous band that every v2 benchmark produces and
    the empty band of the G1-rev directionality variant.

    Args:
        ax: The target subplot axis.
        meta: The dataset metadata dict.
    """
    lag_axis = np.arange(_MAX_LAG + 1)
    band = sorted(set(meta.get("true_lag_band", [])))
    ax.bar(lag_axis, [1.0 if l in set(band) else 0.0 for l in lag_axis],
           width=1.0, color=_C_BAND)
    if band:
        title = (f"True source-lag band {band[0]}..{band[-1]}\n"
                 f"(model attention window 0..{_MAX_LAG})")
    else:
        title = ("No causal lag band\n"
                 "(reverse-roles directionality benchmark -- te_true=0)")
    ax.set_title(title)
    ax.set_xlabel("source lag $\\ell$")
    ax.set_ylabel("carries transfer")
    ax.set_xlim(-0.5, _MAX_LAG + 0.5)
    ax.set_ylim(0, 1.2)
    ps.style_axes(ax)


def _scatter_alignment(ax: plt.Axes, src: np.ndarray, tgt: np.ndarray,
                       color: str = _C_INFORMATIVE) -> float:
    """Scatter ``src`` against ``tgt`` (downsampled) and return their correlation.

    Args:
        ax: The target subplot axis.
        src: Flattened source values.
        tgt: Flattened target values.
        color: Marker colour.

    Returns:
        The empirical Pearson correlation of ``src`` and ``tgt``.
    """
    if src.size > 6000:
        sel = np.random.default_rng(0).choice(src.size, 6000, replace=False)
        src, tgt = src[sel], tgt[sel]
    ax.scatter(src, tgt, s=4, alpha=0.25, color=color, edgecolors="none")
    return float(np.corrcoef(src, tgt)[0, 1]) if src.size > 1 else 0.0


def _welch_psd(x: np.ndarray, n_seg: int = 256) -> tuple:
    r"""Compute a simple Welch-style PSD with 50%-overlap Hann windows.

    Avoids a SciPy dependency by using ``np.fft``. The mean is removed per
    segment before windowing so the PSD measures variability, not the DC.

    Args:
        x: 1-D real signal.
        n_seg: Segment length (in samples).

    Returns:
        Tuple ``(freqs, psd)`` of length ``n_seg // 2 + 1`` (Nyquist-normalised
        frequency axis in $[0, 0.5]$ and a positive PSD).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < n_seg:
        n_seg = max(8, x.size)
    window = np.hanning(n_seg)
    win_norm = (window * window).sum()
    hop = n_seg // 2
    n_segments = max(1, 1 + (x.size - n_seg) // hop)
    psd = np.zeros(n_seg // 2 + 1, dtype=np.float64)
    for k in range(n_segments):
        seg = x[k * hop : k * hop + n_seg]
        if seg.size < n_seg:
            break
        seg = (seg - seg.mean()) * window
        spec = np.fft.rfft(seg)
        psd += (spec.real * spec.real + spec.imag * spec.imag)
    psd /= max(1, n_segments) * win_norm
    freqs = np.fft.rfftfreq(n_seg, d=1.0)
    return freqs, psd


def _panels_state_space(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                        meta: Dict[str, Any]) -> None:
    r"""Fill panels 3, 4, 6 for the G1 / G1-rev Gaussian state-space benchmark.

    Args:
        axes: The $3 \times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    n = Y.shape[0]
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    delays = list(meta.get("delays") or [])
    delay = int(delays[0]) if delays else 0
    oscillators = list(meta.get("oscillators") or [])
    target_ar = float(meta.get("target_ar", 0.0))
    reverse = bool(meta.get("reverse_roles", False))
    distractor_ch = M if M < c_y else 1

    # --- Panel 3: target PSD overlay (informative vs distractor) ---------
    ax = axes[1, 0]
    n_sub = min(64, n)
    psd_inf = np.zeros(0)
    for s in range(n_sub):
        f, p = _welch_psd(Y[s, :, 0])
        psd_inf = p if psd_inf.size == 0 else psd_inf + p
    psd_inf /= max(1, n_sub)
    ax.semilogy(f, psd_inf + 1e-12, color=_C_INFORMATIVE, lw=1.2,
                label="target ch 0 (oscillator)")
    if M < c_y:
        psd_dist = np.zeros(0)
        for s in range(n_sub):
            _, p = _welch_psd(Y[s, :, distractor_ch])
            psd_dist = p if psd_dist.size == 0 else psd_dist + p
        psd_dist /= max(1, n_sub)
        ax.semilogy(f, psd_dist + 1e-12, color=_C_DISTRACTOR, lw=1.0,
                    label=f"distractor ch {distractor_ch}")
    # mark the dominant oscillator frequency.
    if oscillators:
        omega = float(oscillators[0][1])
        f_osc = omega / (2.0 * np.pi)
        ax.axvline(f_osc, color=_C_DELAY, ls=":", lw=1.0,
                   label=fr"$\omega/2\pi$={f_osc:.3f}")
    ax.set_title("Target PSD -- low-frequency oscillator vs distractor")
    ax.set_xlabel("frequency (cycles / step)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 4: oscillator phase-portrait (s_t vs s_{t-1}) -------------
    ax = axes[1, 1]
    # Use the source stream (which holds the AR(2) state s_t in the
    # forward orientation; in reverse-roles, the same coordinates show the
    # target-side process -- still a useful identifier).
    s_prev = U[:, :-1, 0].reshape(-1)
    s_curr = U[:, 1:, 0].reshape(-1)
    if s_prev.size > 8000:
        sel = np.random.default_rng(0).choice(s_prev.size, 8000, replace=False)
        s_prev = s_prev[sel]
        s_curr = s_curr[sel]
    ax.scatter(s_prev, s_curr, s=3, alpha=0.18, color=_C_SOURCE,
               edgecolors="none")
    portrait_title = (
        "Source phase-portrait $s_{t-1}$ vs $s_t$  (ch 0, oscillator orbit)"
        if not reverse
        else "Slot phase-portrait $s_{t-1}$ vs $s_t$  (reverse-roles slot)"
    )
    ax.set_title(portrait_title)
    ax.set_xlabel("$s_{t-1}$, channel 0")
    ax.set_ylabel("$s_t$, channel 0")
    ps.style_axes(ax)

    # --- Panel 6: text summary -------------------------------------------
    osc_str = (f"  oscillators      : {[(round(r, 3), round(w, 3)) for r, w in oscillators]}"
               if oscillators else "  oscillators      : (none)")
    extra = []
    if reverse:
        extra = [f"  direction        : {meta.get('direction')}",
                 f"  reverse_roles    : True"]
    _panel_text(axes[2, 1], meta, [
        f"  delays D         : {delays}",
        f"  target AR coeff  : {target_ar}",
        f"  B_y (couplings)  : {meta.get('B_y')}",
        f"  sigma2_y         : {meta.get('sigma2_y')}",
        f"  sigma2_eta       : {meta.get('sigma2_eta')}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
        f"  representative D : {delay}",
        osc_str,
        *extra,
    ])


def _panels_arx(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                meta: Dict[str, Any]) -> None:
    r"""Fill panels 3, 4, 6 for the G2 smooth AR(1)-ARX benchmark.

    Args:
        axes: The $3 \times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    # Variable per-sample delay: ``meta["delay"]`` is None, so use the
    # representative ``delay_max`` for the alignment panels.
    delay = _primary_delay(meta)
    variable = bool(meta.get("variable_delay", False))
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    c = float(meta["c"])
    reverse = bool(meta.get("reverse_roles", False))

    # --- Panel 3: delay-alignment scatter -------------------------------
    ax = axes[1, 0]
    src = U[:, : T - delay, 0].reshape(-1)
    tgt = Y[:, delay:, 0].reshape(-1)
    emp = _scatter_alignment(ax, src, tgt, color=_C_INFORMATIVE)
    note = "  (reverse roles: anti-causal, expect ~0)" if reverse else ""
    ax.set_title(
        f"Delay alignment ch 0:  $U(t-D)$ vs $Y(t)$\n"
        f"empirical corr={emp:.3f}{note}"
    )
    ax.set_xlabel(f"$U(t-{delay})$, channel 0")
    ax.set_ylabel("$Y(t)$, channel 0")
    ps.style_axes(ax)

    # --- Panel 4: per-channel lagged correlation ------------------------
    ax = axes[1, 1]
    corr = _lagged_corr_per_channel(U, Y, delay)
    colors = [_C_INFORMATIVE if j < M else _C_DISTRACTOR for j in range(c_y)]
    ax.bar(np.arange(c_y), corr, color=colors, width=1.0)
    ax.axhline(0.0, color=_C_ANALYTIC, ls="--", lw=0.8)
    ax.set_title(
        f"Per-channel lagged corr $\\mathrm{{corr}}(U_j(t-D), Y_j(t))$\n"
        f"informative=blue ($j<M={M}$), distractor=orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("lagged correlation")
    ax.set_xlim(-0.5, c_y - 0.5)
    ps.style_axes(ax)

    # --- Panel 6: text summary -----------------------------------------
    extra = []
    if reverse:
        extra = [f"  direction        : {meta.get('direction')}",
                 f"  reverse_roles    : True"]
    delay_line = (
        f"  delay range      : {meta.get('delay_min')}..{meta.get('delay_max')}"
        if variable else f"  delay D          : {delay}"
    )
    _panel_text(axes[2, 1], meta, [
        delay_line,
        f"  ARX coupling c   : {c}",
        f"  rho_u            : {meta.get('rho_u')}",
        f"  rho_y            : {meta.get('rho_y')}",
        f"  sigma2_eta       : {meta.get('sigma2_eta')}",
        f"  sigma2_eps       : {meta.get('sigma2_eps')}",
        f"  burn_in          : {meta.get('burn_in')}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
        *extra,
    ])


def _panels_regime_switch(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                          meta: Dict[str, Any]) -> None:
    r"""Fill panels 3, 4, 6 for the G3 slow categorical regime-switch benchmark.

    Args:
        axes: The $3 \times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    K = int(meta["K_classes"])
    delta = int(meta["delta"])
    p_switch = float(meta["p_switch"])

    # Decode the per-step regime from the per-channel source one-hot. The
    # source carries onehot(R_{t+delta}) for ch m at slots m*K..(m+1)*K.
    onehot = U[0, :, :K]                                  # (T, K) for ch 0
    decoded = np.argmax(onehot, axis=-1)                  # (T,)

    # --- Panel 3: regime strip (sample 0, ch 0 source decode) -----------
    ax = axes[1, 0]
    # Show the decoded regime as a thin horizontal strip alongside the
    # target trace; downsample if T is huge.
    t_axis = np.arange(T)
    strip = decoded[None, :]                              # (1, T)
    ax.imshow(strip, aspect="auto", cmap="tab10", vmin=0, vmax=max(1, K - 1),
              extent=(0, T, -0.5, 0.5), interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("time step $t$")
    ax.set_title(
        f"Decoded regime $R_t$ (source ch 0, $K$={K})\n"
        f"source leaks $R_{{t+\\delta}}$ at delta={delta}"
    )
    ps.style_axes(ax, grid="none")

    # --- Panel 4: target trace coloured by regime ------------------------
    ax = axes[1, 1]
    y0 = Y[0, :, 0]
    # plot baseline + colour transitions to mark regime spans
    ax.plot(t_axis, y0, lw=0.8, color=ps.COLOR_BLACK, alpha=0.6,
            label="target ch 0")
    # shade regime spans with the same tab10 colormap
    cmap = plt.get_cmap("tab10")
    boundaries = list(np.where(np.diff(decoded) != 0)[0] + 1)
    span_starts = [0, *boundaries]
    span_ends = [*boundaries, T]
    for lo, hi in zip(span_starts, span_ends):
        ax.axvspan(lo, hi, color=cmap(int(decoded[lo]) % 10), alpha=0.18)
    ax.set_xlim(0, T)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("target value")
    ax.set_title(
        "Target trace coloured by active regime (sample 0)"
    )
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 6: text summary ------------------------------------------
    _panel_text(axes[2, 1], meta, [
        f"  K_classes        : {K}",
        f"  p_switch         : {p_switch}",
        f"  source lead delta: {delta}",
        f"  shared_regime    : {meta.get('shared_regime')}",
        f"  template_period_min: {meta.get('template_period_min')}",
        f"  sigma2_y / sigma2_u: {meta.get('sigma2_y')} / {meta.get('sigma2_u')}",
        f"  informative M    : {M} / c_y={c_y}  (source needs M*K={M*K} <= c_u)",
    ])


def _panel_text(ax: plt.Axes, meta: Dict[str, Any], param_lines: list) -> None:
    """Fill panel 6 with the ground-truth + process-parameter text block.

    Args:
        ax: The target subplot axis.
        meta: The dataset metadata dict.
        param_lines: Benchmark-specific process-parameter lines.
    """
    ax.axis("off")
    sizes = meta.get("split_sizes", {})
    band = sorted(set(meta.get("true_lag_band", [])))
    band_str = f"{band[0]}..{band[-1]}" if band else "(none -- reverse/null)"
    lines = [
        "Ground truth",
        f"  benchmark        : {meta.get('benchmark')}",
        f"  te_true (block)  : {float(meta['te_true']):.4f} nats",
        f"  te_per_step      : {float(meta['te_per_step']):.4f} nats",
        f"  true_lag_band    : {band_str}  (H={meta.get('horizon')})",
        "",
        "Process parameters",
        *param_lines,
        f"  standardized     : {meta.get('standardized')}",
        "",
        "Splits",
        f"  train / val / test : "
        f"{sizes.get('train', '?')} / {sizes.get('val', '?')} / "
        f"{sizes.get('test', '?')}",
        f"  sequence length T  : {meta.get('sequence_length')}",
    ]
    ax.text(
        0.02, 0.98, "\n".join(lines), va="top", ha="left",
        family="monospace", fontsize=9.0, color=ps.COLOR_BLACK,
        transform=ax.transAxes,
    )


def make_preview(
    out_dir: Union[str, Path],
    meta: Dict[str, Any],
    *,
    split: str = "train",
    out_pdf: Optional[Union[str, Path]] = None,
    n_sub: int = 1000,
) -> Path:
    r"""Render a one-page ``preview.{pdf,png}`` for a cached synthetic dataset.

    The panel layout adapts to ``meta["benchmark"]`` (Phase 7): panels 3, 4 and
    6 are dispatched to a Gaussian (A / B / G), XOR (C) or two-lag (E) variant;
    panels 1, 2 and 5 are shared.

    Args:
        out_dir: Directory holding ``{split}.npz`` and ``meta.json``.
        meta: The dataset metadata dict (analytic ground truth + config).
        split: Which cached split to visualise (default ``"train"``).
        out_pdf: Output PDF path. Defaults to ``out_dir/preview.pdf``; a
            sibling ``.png`` is always written alongside it.
        n_sub: Leading samples used for the scatter / correlation panels.

    Returns:
        The path of the written PDF.
    """
    out_dir = Path(out_dir)
    npz_path = out_dir / f"{split}.npz"
    out_pdf = Path(out_pdf) if out_pdf is not None else out_dir / "preview.pdf"

    data = _load_split(npz_path, n_sub)
    Y, U = data["Y"], data["U"]
    benchmark = meta.get("benchmark", "?")
    tag = meta.get("tag", "?")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))
    fig.suptitle(
        f"Synthetic TE dataset preview  --  Benchmark {benchmark}  (tag: {tag})",
        fontsize=ps.FONT_TITLE,
        fontweight="bold",
    )

    _panels_traces(axes, Y, U, meta)
    if benchmark in ("G1", "G1-rev"):
        _panels_state_space(axes, Y, U, meta)
    elif benchmark == "G2":
        _panels_arx(axes, Y, U, meta)
    elif benchmark == "G3":
        _panels_regime_switch(axes, Y, U, meta)
    else:
        # Unknown benchmark -- leave the dispatched panels blank rather than
        # rendering misleading content from a stale v1 layout.
        for ax in (axes[1, 0], axes[1, 1], axes[2, 1]):
            ax.axis("off")
            ax.text(0.5, 0.5,
                    f"no preview panels for benchmark {benchmark!r}",
                    ha="center", va="center", transform=ax.transAxes)
    _panel_lag_band(axes[2, 0], meta)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ps.save_figure(fig, out_pdf.with_suffix(""))
    return out_pdf


# =============================================================================
# Multi-figure dataset gallery -- imshow-rich per-dataset inspection
# =============================================================================
#
# ``make_preview`` renders a single condensed page. The gallery below renders a
# *set* of standalone figures into ``<cache>/figures/`` so every facet of a
# dataset -- its raw inputs, the native model fields, the forecast-target
# window, the source-lag structure, the analytic-TE breakdown and the value
# distributions -- can be inspected at full size. Each figure is written as a
# ``.pdf`` + ``.png`` pair via :func:`plot_style.save_figure`.

_WARMUP_DEFAULT = 30    # model ``warmup_period`` -- steps excluded from the loss
_GALLERY_SAMPLES = 2    # per-sample heatmaps rendered for this many samples
_CORR_SAMPLE_CAP = 256  # samples used for the (slow) lagged-correlation panels
_HEATMAP_CMAP = "RdBu_r"   # diverging map for zero-centred standardised values

# Figure margins for the stacked-panel gallery figures: a lower ``top`` than
# the plot_style default leaves clear room for the bold suptitle above the
# first panel's own title.
_GALLERY_MARGINS = (0.10, 0.95, 0.92, 0.05)


def _channel_spans(meta: Dict[str, Any]) -> Dict[str, tuple]:
    """Return the ``{field: (lo, hi)}`` channel ranges of the native fields.

    Args:
        meta: The dataset metadata dict (its ``channel_map`` is used when
            present).

    Returns:
        A dict mapping each native field name (``fhr_st`` / ``fhr_ph`` /
        ``up_st`` / ``up_ph``) to its ``(lo, hi)`` half-open channel range.
    """
    cmap = meta.get("channel_map")
    if cmap:
        return {k: (int(v[0]), int(v[1])) for k, v in cmap.items()}
    return {
        "fhr_st": (0, 43), "fhr_ph": (43, 87),
        "up_st": (0, 43), "up_ph": (43, 101),
    }


def _symmetric_vlim(arr: np.ndarray, pct: float = 99.0, floor: float = 0.5) -> float:
    """Symmetric colour limit for a zero-centred (standardised) array.

    Args:
        arr: The array to scale.
        pct: Percentile of $|arr|$ used as the limit (robust to outliers).
        floor: Smallest limit returned, so near-constant panels stay readable.

    Returns:
        A positive ``float`` ``v`` for use as ``vmin=-v, vmax=+v``.
    """
    v = float(np.percentile(np.abs(np.asarray(arr, dtype=float)), pct))
    return max(floor, v)


def _heatmap(ax: plt.Axes, arr: np.ndarray, *, vlim: float,
             cmap: str = _HEATMAP_CMAP, vmin: Optional[float] = None):
    """imshow ``arr`` $(T, C)$ as a channel-(y) by time-(x) heatmap.

    Args:
        ax: The target axis.
        arr: A $(T, C)$ array (time-major); shown transposed.
        vlim: Symmetric colour limit (``vmin=-vlim`` unless ``vmin`` is given).
        cmap: Matplotlib colormap name.
        vmin: Optional explicit lower colour bound (for sequential maps).

    Returns:
        The ``AxesImage`` handle (pass to :func:`plot_style.attach_colorbar`).
    """
    T, C = arr.shape
    return ax.imshow(
        np.asarray(arr).T, aspect="auto", origin="lower", cmap=cmap,
        vmin=(-vlim if vmin is None else vmin), vmax=vlim,
        interpolation="nearest",
        extent=(0.0, float(T), -0.5, float(C) - 0.5),
    )


def _mark_informative(ax: plt.Axes, m: int, c: int, *, axis: str = "y") -> None:
    """Draw the informative / distractor channel split line on a heatmap.

    Args:
        ax: The target axis.
        m: Number of informative channels (``0 < m < c``).
        c: Total channel count.
        axis: ``"y"`` for a horizontal split (channel-on-y heatmaps).
    """
    if not 0 < m < c:
        return
    if axis == "y":
        ax.axhline(m - 0.5, color=ps.COLOR_BLACK, lw=1.1, ls="--")
        # Anchor the label *above* the split line (va="bottom"); for the small
        # informative band this keeps it clear of the x-axis ticks below.
        ax.text(
            0.012, (m - 0.5) / c + 0.02, "informative $j<M$",
            color=ps.COLOR_BLACK, fontsize=ps.FONT_TICK, ha="left", va="bottom",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
        )


def _lagged_corr_grid(
    U: np.ndarray, Y: np.ndarray, max_lag: int, n_cap: int = _CORR_SAMPLE_CAP
) -> np.ndarray:
    r"""Pooled Pearson correlation grid $\mathrm{corr}(U_j(t-\ell),\,Y_j(t))$.

    Args:
        U: Source tensor $(n, T, C_u)$.
        Y: Target tensor $(n, T, C_y)$.
        max_lag: Largest lag $\ell$ evaluated (the grid spans $0..\texttt{max\_lag}$).
        n_cap: Cap on the number of samples pooled (keeps the panel cheap).

    Returns:
        A $(C_y, \texttt{max\_lag}+1)$ array of per-channel, per-lag correlations
        (each $Y$ channel paired with the source channel of the same index).
    """
    n = min(int(n_cap), Y.shape[0])
    U, Y = U[:n], Y[:n]
    T, c_y = Y.shape[1], Y.shape[-1]
    grid = np.zeros((c_y, max_lag + 1), dtype=np.float64)
    for lag in range(max_lag + 1):
        if T - lag < 2:
            continue
        src = U[:, : T - lag, :c_y].reshape(-1, c_y).astype(np.float64)
        tgt = Y[:, lag:, :].reshape(-1, c_y).astype(np.float64)
        src = src - src.mean(axis=0, keepdims=True)
        tgt = tgt - tgt.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(src, axis=0) * np.linalg.norm(tgt, axis=0)
        grid[:, lag] = np.where(denom > 0.0, (src * tgt).sum(axis=0) / (denom + 1e-12), 0.0)
    return grid


def _fig_input_heatmaps(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 1 -- raw target $Y$ and source $U$ as channel-by-time heatmaps.

    Args:
        data: Loaded split dict (``Y`` / ``U`` / ``weight``).
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    n_show = min(_GALLERY_SAMPLES, Y.shape[0])
    delay = _primary_delay(meta)
    M, c_y, c_u = int(meta.get("M", 0)), int(meta["c_y"]), int(meta["c_u"])
    tag = meta.get("tag", "?")

    rows = [(stream, s) for s in range(n_show) for stream in ("Y", "U")]
    fig, axes, caxes = ps.stacked_figure(
        [1.5] * len(rows), colorbar=True, width=11.0, hspace=0.62,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Dataset inputs -- raw heatmaps  (Benchmark {meta.get('benchmark','?')}, "
        f"tag: {tag})", fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    for ax, cax, (stream, s) in zip(axes, caxes, rows):
        arr = Y[s] if stream == "Y" else U[s]
        c_tot = c_y if stream == "Y" else c_u
        vlim = _symmetric_vlim(arr)
        im = _heatmap(ax, arr, vlim=vlim)
        name = "target $Y=[fhr\\_st,fhr\\_ph]$" if stream == "Y" \
            else "source $U=[up\\_st,up\\_ph]$"
        ax.set_title(f"{name} -- sample {s}   ({c_tot} channels x {arr.shape[0]} steps)")
        ax.set_xlabel("time step $t$")
        ax.set_ylabel("channel $j$")
        ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
        _mark_informative(ax, M, c_tot)
        ps.style_axes(ax, grid="none")
        ps.attach_colorbar(fig, im, cax, label="standardised value")
    return ps.save_figure(fig, fig_dir / "01_input_heatmaps")


def _fig_model_fields(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 2 -- the four native model input fields for one sample.

    Shows ``fhr_st`` / ``fhr_ph`` / ``up_st`` / ``up_ph`` as separate heatmaps,
    i.e. exactly the tensors the dataloader hands the model.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"][0], data["U"][0]
    spans = _channel_spans(meta)
    fields = [
        ("fhr_st", Y, spans["fhr_st"], "target scattering  $fhr\\_st$"),
        ("fhr_ph", Y, spans["fhr_ph"], "target phase  $fhr\\_ph$"),
        ("up_st", U, spans["up_st"], "source scattering  $up\\_st$"),
        ("up_ph", U, spans["up_ph"], "source phase  $up\\_ph$"),
    ]
    fig, axes, caxes = ps.stacked_figure(
        [1.4] * 4, colorbar=True, width=11.0, hspace=0.6,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Native model input fields -- sample 0  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    delay = _primary_delay(meta)
    M = int(meta.get("M", 0))
    for ax, cax, (name, arr, (lo, hi), title) in zip(axes, caxes, fields):
        sub = arr[:, lo:hi]
        vlim = _symmetric_vlim(sub)
        im = _heatmap(ax, sub, vlim=vlim)
        ax.set_title(f"{title}   (channels {lo}:{hi}  ->  {hi - lo} wide)")
        ax.set_xlabel("time step $t$")
        ax.set_ylabel("field channel")
        ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
        # The informative channels live at the front of the *stream*; only the
        # first scattering field of each stream can carry them.
        if name in ("fhr_st", "up_st"):
            _mark_informative(ax, min(M, hi - lo), hi - lo)
        ps.style_axes(ax, grid="none")
        ps.attach_colorbar(fig, im, cax, label="standardised value")
    return ps.save_figure(fig, fig_dir / "02_model_input_fields")


def _fig_io_forecast(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    r"""Figure 3 -- the forecast contract: history, source window and target block.

    Panel 1 overlays an informative target channel with its delayed source
    drive. Panel 2 is the $(L, C_u)$ source-history window the lag attention
    consumes at one anchor. Panel 3 is the $(H_d, C_y)$ future block the
    decoder must predict at that anchor.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"][0], data["U"][0]
    T = Y.shape[0]
    delay = _primary_delay(meta)
    horizon = int(meta.get("horizon", 30))
    max_lag = _MAX_LAG
    a = meta.get("a")
    anchor_range = meta.get("clean_anchor_range", [delay - 1, T - horizon])
    t0 = int((anchor_range[0] + anchor_range[1]) // 2)
    anchors = sorted({anchor_range[0] + 5, t0, max(anchor_range[0] + 5, anchor_range[1] - 5)})

    fig, axes, caxes = ps.stacked_figure(
        [1.25, 1.5, 1.5], width=11.0, colorbar=[False, True, True], hspace=0.62,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Forecast contract -- inputs, source window and prediction target  "
        f"(tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: informative channel 0 history + delayed source drive ----
    ax = axes[0]
    t_axis = np.arange(T)
    ax.plot(t_axis, Y[:, 0], lw=0.9, color=ps.COLOR_BLUE, label="target $Y_0(t)$")
    if a is not None:
        drive = np.full(T, np.nan, dtype=float)
        drive[delay:] = float(a) * U[: T - delay, 0]
        ax.plot(t_axis, drive, lw=0.9, color=ps.COLOR_GREEN,
                label=f"delayed drive $a\\,U_0(t-{delay})$")
    ax.axvspan(0, _WARMUP_DEFAULT, color=ps.COLOR_GRAY, alpha=0.16,
               label=f"warm-up [0,{_WARMUP_DEFAULT})")
    for k, anc in enumerate(anchors):
        ax.axvspan(anc + 1, anc + horizon, color=ps.COLOR_VERMILLION, alpha=0.16,
                   label="forecast horizon" if k == 0 else None)
        ax.axvline(anc, color=ps.COLOR_VERMILLION, lw=0.8, ls="--")
    ax.set_title("informative channel 0: history, warm-up and forecast windows")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("standardised value")
    ps.tighten_xaxis(ax, t_axis)
    ax.legend(loc="upper right", ncol=2)
    ps.style_axes(ax)

    # --- Panel 2: source-history window the attention sees at anchor t0 ---
    ax = axes[1]
    lo = max(0, t0 - max_lag)
    src_win = U[lo : t0 + 1, :]                       # (L', C_u)
    vlim = _symmetric_vlim(src_win)
    im = _heatmap(ax, src_win, vlim=vlim)
    ax.set_title(
        f"source-history window at anchor $t={t0}$  "
        f"(lags 0..{t0 - lo}; model attention window 0..{max_lag})"
    )
    ax.set_xlabel(f"steps before anchor  (0 = $t-{t0 - lo}$ ... {src_win.shape[0]-1} = $t$)")
    ax.set_ylabel("source channel")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[1], label="standardised value")

    # --- Panel 3: the (H_d, C_y) future block the decoder predicts --------
    ax = axes[2]
    y_plus = Y[t0 + 1 : t0 + 1 + horizon, :]          # (H_d, C_y)
    vlim = _symmetric_vlim(y_plus)
    im = _heatmap(ax, y_plus, vlim=vlim)
    ax.set_title(
        f"forecast target $Y^+$ at anchor $t={t0}$   "
        f"(horizon $H_d$={horizon} x $C_y$={Y.shape[1]} -- the decoder output)"
    )
    ax.set_xlabel(r"forecast step $\tau = 1 \dots H_d$")
    ax.set_ylabel("target channel")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[2], label="standardised value")
    return ps.save_figure(fig, fig_dir / "03_forecast_contract")


def _fig_sample_traces(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 4 -- line traces of informative / distractor / source channels.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    T = Y.shape[1]
    t_axis = np.arange(T)
    delay = _primary_delay(meta)
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])
    n_tr = min(4, Y.shape[0])
    dist_ch = M if M < c_y else 1   # easy variant has no distractor -> ch 1

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.4), sharex=True)
    fig.suptitle(
        f"Sample traces -- {n_tr} sequences overlaid  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )
    panels = [
        (axes[0], Y, 0, "target $Y$ -- informative channel 0"),
        (axes[1], Y, dist_ch,
         f"target $Y$ -- {'distractor' if M < c_y else 'informative'} channel {dist_ch}"),
        (axes[2], U, 0, "source $U$ -- channel 0"),
    ]
    for ax, arr, ch, title in panels:
        for s in range(n_tr):
            ax.plot(t_axis, arr[s, :, ch], lw=0.8, alpha=0.85,
                    color=ps.PALETTE_EXTENDED[s % len(ps.PALETTE_EXTENDED)])
        ax.axvline(delay, color=ps.COLOR_GRAY, ls="--", lw=0.9,
                   label=f"delay $D$={delay}")
        ax.set_title(title)
        ax.set_ylabel("std. value")
        ps.tighten_xaxis(ax, t_axis)
        ax.legend(loc="upper right")
        ps.style_axes(ax)
    axes[-1].set_xlabel("time step $t$")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return ps.save_figure(fig, fig_dir / "04_sample_traces")


def _fig_lag_structure(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    r"""Figure 5 -- the source-lag structure the model's attention must recover.

    Panel 1 is the lagged cross-correlation $\mathrm{corr}(U_j(t-\ell), Y_j(t))$
    averaged over informative vs distractor channels. Panel 2 is the full
    $(C_y, L)$ correlation grid. Both shade the true lag band
    $\{D-H,\dots,D-1\}$.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])
    delay = _primary_delay(meta)
    band = sorted(set(meta.get("true_lag_band", [])))
    grid = _lagged_corr_grid(U, Y, _MAX_LAG)          # (C_y, L)
    lag_axis = np.arange(_MAX_LAG + 1)

    fig, axes, caxes = ps.stacked_figure(
        [1.15, 1.6], width=11.0, colorbar=[False, True], hspace=0.55,
        margins=_GALLERY_MARGINS,
    )
    fig.suptitle(
        f"Source-lag structure -- where the transfer lives  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: informative- vs distractor-mean lagged correlation ------
    ax = axes[0]
    inf_curve = grid[:M].mean(axis=0) if M > 0 else np.zeros(_MAX_LAG + 1)
    ax.plot(lag_axis, inf_curve, color=ps.COLOR_BLUE, lw=1.3,
            label=f"informative channels ($j<M={M}$)")
    if M < c_y:
        dist_curve = grid[M:].mean(axis=0)
        ax.plot(lag_axis, dist_curve, color=ps.COLOR_ORANGE, lw=1.1,
                label="distractor channels")
    if band:
        ax.axvspan(band[0], band[-1], color=ps.COLOR_VERMILLION, alpha=0.16,
                   label=f"true lag band {band[0]}..{band[-1]}")
    ax.axvline(delay, color=ps.COLOR_GRAY, ls="--", lw=0.9, label=f"delay $D$={delay}")
    ax.axhline(0.0, color=ps.COLOR_BLACK, lw=0.5)
    ax.set_title(r"lagged cross-correlation $\mathrm{corr}(U_j(t-\ell),\,Y_j(t))$")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel("correlation")
    ax.set_xlim(0, _MAX_LAG)
    ax.legend(loc="upper left", ncol=2)
    ps.style_axes(ax)

    # --- Panel 2: full (channel, lag) correlation grid --------------------
    ax = axes[1]
    vlim = max(0.05, float(np.percentile(np.abs(grid), 99.5)))
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=_HEATMAP_CMAP,
        vmin=-vlim, vmax=vlim, interpolation="nearest",
        extent=(-0.5, _MAX_LAG + 0.5, -0.5, c_y - 0.5),
    )
    if band:
        for edge in (band[0], band[-1]):
            ax.axvline(edge, color=ps.COLOR_BLACK, lw=0.8, ls="--")
    ax.axvline(delay, color=ps.COLOR_GRAY, ls=":", lw=1.0)
    _mark_informative(ax, M, c_y)
    ax.set_title(r"per-channel lagged-correlation grid (bright square = true transfer)")
    ax.set_xlabel(r"source lag $\ell$")
    ax.set_ylabel("target channel $j$")
    ps.style_axes(ax, grid="none")
    ps.attach_colorbar(fig, im, caxes[1], label="correlation")
    return ps.save_figure(fig, fig_dir / "05_lag_structure")


def _fig_distributions(
    data: Dict[str, np.ndarray], meta: Dict[str, Any], fig_dir: Path
) -> list:
    """Figure 7 -- value distributions and the per-channel standardisation check.

    Args:
        data: Loaded split dict.
        meta: The dataset metadata dict.
        fig_dir: Output directory for the figure pair.

    Returns:
        The list of written file paths.
    """
    Y, U = data["Y"], data["U"]
    M, c_y = int(meta.get("M", 0)), int(meta["c_y"])

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    fig.suptitle(
        f"Value distributions and standardisation check  (tag: {meta.get('tag','?')})",
        fontsize=ps.FONT_TITLE, fontweight="bold",
    )

    # --- Panel 1: value histograms ----------------------------------------
    ax = axes[0]
    bins = np.linspace(-5.0, 5.0, 80)
    ax.hist(Y[:, :, :M].reshape(-1), bins=bins, density=True, histtype="step",
            color=ps.COLOR_BLUE, lw=1.3, label="target informative")
    if M < c_y:
        ax.hist(Y[:, :, M:].reshape(-1), bins=bins, density=True, histtype="step",
                color=ps.COLOR_ORANGE, lw=1.1, label="target distractor")
    ax.hist(U.reshape(-1), bins=bins, density=True, histtype="step",
            color=ps.COLOR_GREEN, lw=1.1, label="source")
    ax.set_title("standardised value histogram")
    ax.set_xlabel("standardised value")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panels 2-3: per-channel std (standardisation should give ~1) -----
    for ax, arr, name, m_mark, n_ch in (
        (axes[1], Y, "target $Y$", M, c_y),
        (axes[2], U, "source $U$", M, int(meta["c_u"])),
    ):
        std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
        colors = [ps.COLOR_BLUE if j < m_mark else ps.COLOR_ORANGE
                  for j in range(n_ch)]
        ax.bar(np.arange(n_ch), std, color=colors, width=1.0)
        ax.axhline(1.0, color=ps.COLOR_BLACK, ls="--", lw=1.0,
                   label="unit std (target)")
        ax.set_title(f"{name} -- per-channel std")
        ax.set_xlabel("channel $j$")
        ax.set_ylabel("standard deviation")
        ax.set_xlim(-0.5, n_ch - 0.5)
        ax.set_ylim(0.0, max(1.3, float(std.max()) * 1.1))
        ax.legend(loc="lower right")
        ps.style_axes(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return ps.save_figure(fig, fig_dir / "07_distributions")


def make_dataset_gallery(
    out_dir: Union[str, Path],
    meta: Dict[str, Any],
    *,
    split: str = "train",
    n_sub: int = 1000,
) -> list:
    r"""Render the full multi-figure inspection gallery for a cached dataset.

    Writes a set of standalone ``.pdf`` + ``.png`` figures into
    ``<out_dir>/figures/``:

        * ``01_input_heatmaps``     -- raw $Y$ / $U$ channel-by-time heatmaps.
        * ``02_model_input_fields`` -- the four native fields (``fhr_st`` ...).
        * ``03_forecast_contract``  -- history, source window and $Y^+$ target.
        * ``04_sample_traces``      -- informative / distractor / source traces.
        * ``05_lag_structure``      -- lagged-correlation curve + channel-lag grid.
        * ``07_distributions``      -- value histograms + per-channel std check.

    (The v1 ``06_te_structure`` analytic-TE budget figure was removed alongside
    the v1 generators in Sprint 3 -- the v2 generators emit a richer
    ``meta.te_true`` block in ``meta.json`` that supersedes the per-channel
    bar / TE-vs-a curve.)

    Args:
        out_dir: Directory holding ``{split}.npz`` and ``meta.json``.
        meta: The dataset metadata dict (analytic ground truth + config).
        split: Which cached split to visualise (default ``"train"``).
        n_sub: Leading samples loaded for the panels.

    Returns:
        A flat list of every written file path (``.pdf`` and ``.png``).
    """
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = _load_split(out_dir / f"{split}.npz", n_sub)

    written: list = []
    written += _fig_input_heatmaps(data, meta, fig_dir)
    written += _fig_model_fields(data, meta, fig_dir)
    written += _fig_io_forecast(data, meta, fig_dir)
    written += _fig_sample_traces(data, meta, fig_dir)
    written += _fig_lag_structure(data, meta, fig_dir)
    written += _fig_distributions(data, meta, fig_dir)
    return written


if __name__ == "__main__":
    # Dual-mode dispatch (V2-D8): edit-and-run + self-check.
    #
    #   * EDIT-AND-RUN  -- set ``RUN_CONFIG['cache_dir']`` to an existing
    #     ``data/<benchmark>/<tag>/`` directory; the script reads its
    #     ``meta.json`` + ``<split>.npz`` and re-renders ``preview.{pdf,png}``
    #     (and, with ``gallery=True``, the standalone gallery under
    #     ``<cache_dir>/figures/``). No CLI flags needed -- run the file
    #     directly from PyCharm / VS Code.
    #
    #   * SELF-CHECK    -- with the default ``RUN_CONFIG['cache_dir'] = None``
    #     the script builds tiny G1 / G2 / G3 / G1-rev caches in temp dirs
    #     and renders their previews, exercising every panel path. Used as a
    #     smoke test in the v2 test suite.
    import json

    RUN_CONFIG = {
        # Set to an existing cache directory to re-render its preview, e.g.:
        #   "cache_dir": Path(
        #       r"C:/Users/mahdi/Desktop/teb_vae_model/model/"
        #       r"vae_teb_prediction/model/model_experiment/data/G1/G1_baseline"
        #   ),
        "cache_dir": None,           # None -> run the self-check below
        "split": "train",            # which split to render (train / val / test)
        "n_sub": 1000,               # leading samples used by the panels
        "gallery": False,            # True -> also write the gallery figures
    }

    if RUN_CONFIG["cache_dir"] is not None:
        cache_dir = Path(RUN_CONFIG["cache_dir"]).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"visualize: cache_dir does not exist: {cache_dir}"
            )
        meta_path = cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"visualize: meta.json not found in {cache_dir}"
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            cache_meta = json.load(fh)
        pdf = make_preview(
            cache_dir, cache_meta,
            split=str(RUN_CONFIG["split"]),
            n_sub=int(RUN_CONFIG["n_sub"]),
        )
        print(f"[preview] wrote {pdf}")
        if RUN_CONFIG["gallery"]:
            files = make_dataset_gallery(
                cache_dir, cache_meta,
                split=str(RUN_CONFIG["split"]),
                n_sub=int(RUN_CONFIG["n_sub"]),
            )
            print(
                f"[gallery] wrote {len(files)} files under "
                f"{cache_dir / 'figures'}"
            )
        sys.exit(0)

    # Self-check fallback: build tiny caches in temp dirs and render their
    # previews, exercising the G1 / G2 / G3 / G1-rev panel paths.
    import tempfile

    from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
        gen_regime_switch_smooth,
        gen_smooth_arx,
        gen_state_space_oscillator,
    )

    def _dump_and_preview(Y, U, meta, tag):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            np.savez(
                d / "train.npz",
                fhr_st=Y[..., :43].numpy(),
                fhr_ph=Y[..., 43:87].numpy(),
                up_st=U[..., :43].numpy(),
                up_ph=U[..., 43:101].numpy(),
                weight=np.ones((Y.shape[0], Y.shape[1]), dtype=np.float32),
            )
            m = dict(meta)
            m["tag"] = tag
            m["split_sizes"] = {"train": Y.shape[0], "val": 0, "test": 0}
            with open(d / "meta.json", "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=2)
            pdf = make_preview(d, m, n_sub=Y.shape[0])
            assert pdf.is_file() and pdf.stat().st_size > 0, pdf
            png = pdf.with_suffix(".png")
            assert png.is_file() and png.stat().st_size > 0, png
            gallery = make_dataset_gallery(d, m, n_sub=Y.shape[0])
            assert gallery and all(p.is_file() and p.stat().st_size > 0
                                   for p in gallery), gallery
            print(f"[preview {tag}] wrote {pdf.name} + {png.name}; "
                  f"gallery: {len(gallery)} files "
                  f"({len({p.stem for p in gallery})} figures x pdf+png)")

    # G1 -- Gaussian state-space oscillator (cheap MC for the smoke).
    _Y1, _U1, _m1 = gen_state_space_oscillator(
        n=32, T=300, oscillators=[(0.99, 0.05)], target_ar=0.95,
        delays=[60], B_y=[0.5], sigma2_y=1.0, sigma2_eta=0.01, M=4,
        te_n_samples=2_000, seed=0,
    )
    _dump_and_preview(_Y1, _U1, _m1, "smoke_G1")

    # G2 -- smooth AR(1) ARX.
    _Y2, _U2, _m2 = gen_smooth_arx(
        n=32, T=300, rho_u=0.99, rho_y=0.95, c=0.5,
        sigma2_eta=1.0, sigma2_eps=1.0, delay=60, M=4, seed=0,
    )
    _dump_and_preview(_Y2, _U2, _m2, "smoke_G2")

    # G3 -- slow categorical regime switch.
    _Y3, _U3, _m3 = gen_regime_switch_smooth(
        n=32, T=300, K_classes=10, p_switch=0.5, delta=60, M=4, seed=0,
    )
    _dump_and_preview(_Y3, _U3, _m3, "smoke_G3")

    # G1-rev -- directionality control (te_true = 0, true_lag_band = []).
    _Yr, _Ur, _mr = gen_state_space_oscillator(
        n=32, T=300, oscillators=[(0.99, 0.05)], target_ar=0.95,
        delays=[60], B_y=[0.5], sigma2_y=1.0, sigma2_eta=0.01, M=4,
        reverse_roles=True, te_n_samples=2_000, seed=0,
    )
    # Override benchmark id so the make_preview dispatch picks the same panel
    # set as the forward G1 cache.
    _mr["benchmark"] = "G1-rev"
    _dump_and_preview(_Yr, _Ur, _mr, "smoke_G1-rev")

    print("All visualize checks passed.")
