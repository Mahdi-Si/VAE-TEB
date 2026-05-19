r"""Visualisation utilities for cached synthetic TE datasets (Decision D7).

Renders a human-readable preview of a generated benchmark so the data can be
inspected before training and reused with confidence. The preview is a single
multi-panel ``preview.pdf`` (plus a ``preview.png``) written next to the cached
``.npz`` splits.

Public API:
    make_preview: build the one-page ``preview.{pdf,png}`` for a cached dataset.

The preview is **benchmark-aware** (Phase 7): panels 1, 2 and 5 are common to
every benchmark, while panels 3, 4 and 6 are dispatched on
``meta["benchmark"]``:

    * A / B / G (linear-Gaussian): delay-alignment scatter, per-channel lagged
      cross-correlation, and a Gaussian parameter summary.
    * C (delayed XOR): a bit-agreement scatter and per-channel bit-agreement
      rate $P(Y_j(t) = X_j(t-D))$.
    * E (two-lag Gaussian): a two-group delay-alignment scatter and per-channel
      lagged correlation evaluated at each group's own delay.

Note:
    Generators run with ``standardize=True``, which z-scores every channel to
    unit variance, so a raw variance-ratio bar carries no contrast. The
    standardisation-invariant **lagged cross-correlation** (and, for XOR, the
    sign-based bit-agreement rate) is plotted instead.

All figures use the shared publication style in :mod:`plot_style`.

Used by :mod:`build_dataset` and runnable standalone against any cached dataset.
"""

from __future__ import annotations

import math
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

    Args:
        meta: The dataset metadata dict.

    Returns:
        ``meta["delay"]`` for single-delay benchmarks, else ``meta["delay1"]``.
    """
    return int(meta.get("delay", meta.get("delay1", 0)))


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

    Handles the single band (A / B / C), the two bands (E), and the empty band
    of the reverse-roles directionality benchmark (G).

    Args:
        ax: The target subplot axis.
        meta: The dataset metadata dict.
    """
    lag_axis = np.arange(_MAX_LAG + 1)
    benchmark = meta.get("benchmark", "?")
    if benchmark == "E":
        b1 = set(meta.get("lag_band_1", []))
        b2 = set(meta.get("lag_band_2", []))
        ax.bar(lag_axis, [1.0 if l in b1 else 0.0 for l in lag_axis],
               width=1.0, color=_C_INFORMATIVE, label="band 1 ($D_1$)")
        ax.bar(lag_axis, [1.0 if l in b2 else 0.0 for l in lag_axis],
               width=1.0, color=_C_BAND, alpha=0.8, label="band 2 ($D_2$)")
        ax.legend(loc="upper right")
        ax.set_title(f"True source-lag bands (two-lag)\n"
                     f"(model attention window 0..{_MAX_LAG})")
    else:
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


def _panels_gaussian(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                    meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the linear-Gaussian benchmarks (A / B / G).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay = int(meta["delay"])
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    a = float(meta["a"])
    sigma2 = float(meta["sigma2"])
    reverse = bool(meta.get("reverse_roles", False))
    analytic_corr = (
        0.0 if reverse
        else (a / math.sqrt(a * a + sigma2) if (a * a + sigma2) > 0 else 0.0)
    )

    # --- Panel 3: delay-alignment scatter ---------------------------------
    ax = axes[1, 0]
    src = U[:, : T - delay, 0].reshape(-1)
    tgt = Y[:, delay:, 0].reshape(-1)
    emp = _scatter_alignment(ax, src, tgt, color=_C_INFORMATIVE)
    note = "  (reverse roles: anti-causal, expect ~0)" if reverse else ""
    ax.set_title(
        f"Delay alignment ch 0:  $U(t-D)$ vs $Y(t)$\n"
        f"empirical corr={emp:.3f}   analytic={analytic_corr:.3f}{note}"
    )
    ax.set_xlabel(f"$U(t-{delay})$, channel 0")
    ax.set_ylabel("$Y(t)$, channel 0")
    ps.style_axes(ax)

    # --- Panel 4: per-channel lagged correlation --------------------------
    ax = axes[1, 1]
    corr = _lagged_corr_per_channel(U, Y, delay)
    colors = [_C_INFORMATIVE if j < M else _C_DISTRACTOR for j in range(c_y)]
    ax.bar(np.arange(c_y), corr, color=colors, width=1.0)
    ax.axhline(analytic_corr, color=_C_ANALYTIC, ls="--", lw=1.0,
               label=f"analytic informative={analytic_corr:.3f}")
    ax.set_title(
        f"Per-channel lagged corr $\\mathrm{{corr}}(U_j(t-D), Y_j(t))$\n"
        f"informative=blue ($j<M={M}$), distractor=orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("lagged correlation")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    extra = []
    if meta.get("benchmark") == "B":
        extra = [f"  AR coeff rho     : {meta.get('rho')}",
                 f"  burn_in          : {meta.get('burn_in')}"]
    if reverse:
        extra = [f"  direction        : {meta.get('direction')}",
                 f"  reverse_roles    : {meta.get('reverse_roles')}"]
    _panel_text(axes[2, 1], meta, [
        f"  delay D          : {delay}",
        f"  transfer coeff a : {a}",
        f"  noise var sigma2 : {sigma2}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
        *extra,
    ])


def _panels_xor(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
               meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the delayed-XOR benchmark (C).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay = int(meta["delay"])
    M = int(meta["M"])
    c_y = int(meta["c_y"])
    q = float(meta["q"])

    # --- Panel 3: delay-alignment scatter ---------------------------------
    ax = axes[1, 0]
    src = U[:, : T - delay, 0].reshape(-1)
    tgt = Y[:, delay:, 0].reshape(-1)
    emp = _scatter_alignment(ax, src, tgt, color=_C_INFORMATIVE)
    ax.set_title(
        f"XOR delay alignment ch 0:  $U(t-D)$ vs $Y(t)$\n"
        f"empirical corr={emp:.3f}   analytic $1-2q$={1.0 - 2.0 * q:.3f}"
    )
    ax.set_xlabel(f"$U(t-{delay})$, channel 0 ($\\pm 1$ + noise)")
    ax.set_ylabel("$Y(t)$, channel 0")
    ps.style_axes(ax)

    # --- Panel 4: per-channel bit-agreement rate --------------------------
    ax = axes[1, 1]
    y_bit = Y[:, delay:, :] > 0.0
    x_bit = U[:, : T - delay, :c_y] > 0.0
    agree = (y_bit == x_bit).mean(axis=(0, 1))
    colors = [_C_INFORMATIVE if j < M else _C_DISTRACTOR for j in range(c_y)]
    ax.bar(np.arange(c_y), agree, color=colors, width=1.0)
    ax.axhline(1.0 - q, color=_C_ANALYTIC, ls="--", lw=1.0,
               label=f"analytic informative $1-q$={1.0 - q:.3f}")
    ax.axhline(0.5, color=_C_DELAY, ls=":", lw=1.0, label="chance 0.5")
    ax.set_title(
        f"Per-channel bit-agreement $P(\\mathrm{{sign}}\\,Y_j(t)="
        f"\\mathrm{{sign}}\\,X_j(t-D))$\n"
        f"informative=blue ($j<M={M}$), distractor=orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("bit-agreement rate")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    _panel_text(axes[2, 1], meta, [
        f"  delay D          : {delay}",
        f"  bit-flip prob q  : {q}",
        f"  obs_noise        : {meta.get('obs_noise')}",
        f"  informative M    : {M} / c_y={c_y}",
        f"  easy_variant     : {meta.get('easy_variant')}",
    ])


def _panels_two_lag(axes: np.ndarray, Y: np.ndarray, U: np.ndarray,
                   meta: Dict[str, Any]) -> None:
    """Fill panels 3, 4, 6 for the two-lag Gaussian benchmark (E).

    Args:
        axes: The $3 \\times 2$ array of subplot axes.
        Y: Target tensor $(n, T, C_y)$.
        U: Source tensor $(n, T, C_u)$.
        meta: The dataset metadata dict.
    """
    T = Y.shape[1]
    delay1, delay2 = int(meta["delay1"]), int(meta["delay2"])
    a1, a2 = float(meta["a1"]), float(meta["a2"])
    sigma2 = float(meta["sigma2"])
    M1, M2 = int(meta["M1"]), int(meta["M2"])
    m_total = M1 + M2
    c_y = int(meta["c_y"])

    # --- Panel 3: two-group delay-alignment scatter -----------------------
    ax = axes[1, 0]
    s1 = U[:, : T - delay1, 0].reshape(-1)
    t1 = Y[:, delay1:, 0].reshape(-1)
    e1 = _scatter_alignment(ax, s1, t1, color=_C_INFORMATIVE)
    s2 = U[:, : T - delay2, M1].reshape(-1)
    t2 = Y[:, delay2:, M1].reshape(-1)
    e2 = _scatter_alignment(ax, s2, t2, color=_C_SOURCE)
    ax.set_title(
        f"Two-lag delay alignment\n"
        f"group 1 ch 0 @ $D_1$={delay1} corr={e1:.3f}   "
        f"group 2 ch {M1} @ $D_2$={delay2} corr={e2:.3f}"
    )
    ax.set_xlabel("$U(t-D)$")
    ax.set_ylabel("$Y(t)$")
    ps.style_axes(ax)

    # --- Panel 4: per-channel lagged corr at each group's own delay -------
    ax = axes[1, 1]
    corr1 = _lagged_corr_per_channel(U, Y, delay1)
    corr2 = _lagged_corr_per_channel(U, Y, delay2)
    combined = corr1.copy()
    combined[M1:m_total] = corr2[M1:m_total]
    colors = []
    for j in range(c_y):
        if j < M1:
            colors.append(_C_INFORMATIVE)   # group 1
        elif j < m_total:
            colors.append(_C_SOURCE)        # group 2
        else:
            colors.append(_C_DISTRACTOR)    # distractor
    ax.bar(np.arange(c_y), combined, color=colors, width=1.0)
    ac1 = a1 / math.sqrt(a1 * a1 + sigma2)
    ac2 = a2 / math.sqrt(a2 * a2 + sigma2)
    ax.axhline(ac1, color=_C_INFORMATIVE, ls="--", lw=1.0,
               label=f"group 1 ~{ac1:.3f}")
    ax.axhline(ac2, color=_C_SOURCE, ls="--", lw=1.0,
               label=f"group 2 ~{ac2:.3f}")
    ax.set_title(
        "Per-channel lagged corr at the channel's own delay\n"
        f"group 1 ($j<{M1}$) blue, group 2 ($<{m_total}$) green, "
        "distractor orange"
    )
    ax.set_xlabel("target channel $j$")
    ax.set_ylabel("lagged correlation")
    ax.set_xlim(-0.5, c_y - 0.5)
    ax.legend(loc="upper right")
    ps.style_axes(ax)

    # --- Panel 6: text summary --------------------------------------------
    _panel_text(axes[2, 1], meta, [
        f"  delay D1 / D2    : {delay1} / {delay2}",
        f"  coeff a1 / a2    : {a1} / {a2}",
        f"  noise var sigma2 : {sigma2}",
        f"  M1 / M2          : {M1} / {M2}  (c_y={c_y})",
        f"  te_true_1        : {float(meta.get('te_true_1', 0.0)):.4f} nats",
        f"  te_true_2        : {float(meta.get('te_true_2', 0.0)):.4f} nats",
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
    if benchmark == "C":
        _panels_xor(axes, Y, U, meta)
    elif benchmark == "E":
        _panels_two_lag(axes, Y, U, meta)
    else:  # A / B / G and any unknown benchmark fall back to the Gaussian view.
        _panels_gaussian(axes, Y, U, meta)
    _panel_lag_band(axes[2, 0], meta)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ps.save_figure(fig, out_pdf.with_suffix(""))
    return out_pdf


if __name__ == "__main__":
    # Self-check: build tiny caches in temp dirs and render their previews,
    # exercising the Gaussian (A) and XOR (C) panel paths.
    import json
    import tempfile

    from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
        gen_delayed_gaussian,
        gen_delayed_xor,
        gen_two_lag_gaussian,
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
            print(f"[preview {tag}] wrote {pdf.name} + {png.name} "
                  f"({pdf.stat().st_size} bytes)")

    _Ya, _Ua, _ma = gen_delayed_gaussian(
        n=64, T=300, delay=60, a=1.0, sigma2=1.0, M=4, seed=0
    )
    _dump_and_preview(_Ya, _Ua, _ma, "smoke_A")

    _Yc, _Uc, _mc = gen_delayed_xor(n=64, T=300, delay=60, q=0.10, M=4, seed=0)
    _dump_and_preview(_Yc, _Uc, _mc, "smoke_C")

    _Ye, _Ue, _me = gen_two_lag_gaussian(
        n=64, T=300, delay1=50, delay2=80, a1=0.4, a2=0.25,
        sigma2=1.0, M1=4, M2=4, seed=0,
    )
    _dump_and_preview(_Ye, _Ue, _me, "smoke_E")

    _Yg, _Ug, _mg = gen_delayed_gaussian(
        n=64, T=300, delay=60, a=1.0, sigma2=1.0, M=4, reverse_roles=True, seed=0
    )
    _dump_and_preview(_Yg, _Ug, _mg, "smoke_G")

    print("All visualize checks passed.")
