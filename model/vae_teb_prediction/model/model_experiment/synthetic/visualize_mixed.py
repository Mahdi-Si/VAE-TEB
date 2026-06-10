r"""``visualize_mixed`` -- journal-quality data-anatomy figures for ``G1_mix``.

The mixed-population cache (:mod:`mixed_dataset`) pools cells that differ in
informative-channel count $M$, block transfer entropy $\mathrm{TE}$, and lag
band $[d_{\min}, d_{\max}]$ -- but the ``.npz`` arrays alone make it hard to
*see* what "transfer entropy through a drifting lag" actually looks like. This
module renders fully-annotated figures that make the data-generating process
legible at a glance:

1. **Sample anatomy** (``anatomy_<cell>``) -- one full sample of one cell:
   the true lag walk $d_t$ (top), one panel per informative channel pair
   showing the source $U_m$, the target $Y_m$, and the **lag-aligned source**
   $U_m(t - d_t)$ overlaid on the target, with colour-matched source/target
   section pairs and arrows visualising "this source section drives this
   target section $d_t$ steps later"; a zero-transfer contrast panel
   (distractor source / self-predictable target); and a lagged-correlation
   evidence panel with the true delay range shaded.
2. **TE $\times$ lag gallery** (``te_lag_gallery``) -- a (TE level) $\times$
   (lag band) grid of compact target-vs-aligned-source panels at one fixed
   $M$, so the *visual* effect of increasing transfer entropy (target tracks
   the lagged source ever more tightly) and of widening lag bands is directly
   comparable across cases, down to the $\mathrm{TE} = 0$ null row.
3. **Channel atlas** (``channel_atlas_<cell>``) -- channels $\times$ time
   heatmaps of the full $Y$ ($c_y = 87$) and $U$ ($c_u = 101$) buffers with
   the channel-role blocks (informative TE / self-predictable / small-noise;
   informative TE / AR(1) distractor / pure noise) separated and labelled, so
   the "needle in a haystack" structure of the channel budget is visible.

The figures follow the house style (:mod:`plot_style`) used by every other
synthetic figure and carry one-line "how to read this" captions.

Assumption: informative channels sit at indices $[0, M)$ of the concatenated
target / source buffers (the ``randomize_channel_layout: false`` default;
see ``model_validation_v3_mixed.md`` §4). The per-cell block layout is derived
from ``meta['channel_decomp']`` (the tails ``n_smallnoise`` / ``n_noise`` are
$M$-invariant).

Run modes (Decision V2-D8): both a CLI and an edit-and-run ``RUN_CONFIG``.

    python -m ...synthetic.visualize_mixed --cache-tag G1_mix_base
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from model.vae_teb_prediction.model.model_experiment.synthetic import (
    plot_style as ps,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.train_minimal import (
    apply_path_overrides,
    load_config,
    resolve_active_benchmark,
    resolve_user_path,
)

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _PKG_DIR / "config_synth.yaml"
_BENCHMARK = "G1_mix"
_PREVIEW_SUBDIR = "previews"

# Vertical offset separating the source trace from the target trace inside a
# channel-pair panel (both are per-channel z-scored, so +/-3 sigma covers the
# mass and an 8-sigma offset keeps the bands visually disjoint).
_PAIR_OFFSET = 8.0
# Half-width (steps) of the colour-matched source/target section pairs.
_SPAN_HALF_W = 6
# Hues for the three lag-annotation anchors (section pairs + arrows).
_ANCHOR_COLORS = (ps.COLOR_ORANGE, ps.COLOR_GREEN, ps.COLOR_SKY)


# =============================================================================
# Cache loading
# =============================================================================

@dataclass
class MixPreviewData:
    r"""In-memory view of one mixed-cache split for preview rendering.

    Attributes:
        Y: Target buffer $(n, T, c_y)$ -- ``fhr_st`` and ``fhr_ph``
            concatenated on the channel axis.
        U: Source buffer $(n, T, c_u)$ -- ``up_st`` and ``up_ph`` concatenated.
        lag_tt: Per-sample, per-step true lag $d_{i,t}$ $(n, T)$, or ``None``
            when the cache predates the lag-walk stamp.
        cell_id: Per-sample ``sample_cell_id`` $(n,)$.
        meta: The cache ``meta.json`` dict.
        cells: Manifest cells keyed by ``cell_id``.
    """

    Y: np.ndarray
    U: np.ndarray
    lag_tt: Optional[np.ndarray]
    cell_id: np.ndarray
    meta: Dict[str, Any]
    cells: Dict[int, Dict[str, Any]]

    @property
    def T(self) -> int:
        """Sequence length."""
        return int(self.Y.shape[1])


def load_mix_preview_data(cache_dir: Path, split: str = "test") -> MixPreviewData:
    """Load one split of a mixed cache plus its manifest.

    Args:
        cache_dir: The ``data/G1_mix/<tag>/`` cache directory.
        split: Which split to read (``test`` is the cheapest and sufficient).

    Returns:
        The assembled :class:`MixPreviewData`.

    Raises:
        FileNotFoundError: If the split ``.npz`` or ``meta.json`` is absent.
    """
    npz_path = Path(cache_dir) / f"{split}.npz"
    meta_path = Path(cache_dir) / "meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"mixed cache incomplete: need {npz_path} and {meta_path}. "
            f"Build it with mixed_dataset / run_mixed_pipeline first."
        )
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    with np.load(npz_path) as npz:
        Y = np.concatenate(
            [np.asarray(npz["fhr_st"]), np.asarray(npz["fhr_ph"])], axis=-1
        )
        U = np.concatenate(
            [np.asarray(npz["up_st"]), np.asarray(npz["up_ph"])], axis=-1
        )
        lag_tt = (
            np.asarray(npz["true_lag_tt"], dtype=int)
            if "true_lag_tt" in npz.files else None
        )
        cell_id = np.asarray(npz["sample_cell_id"], dtype=int)
    cells = {
        int(c["cell_id"]): c
        for c in (meta.get("mixture", {}) or {}).get("cells", [])
    }
    return MixPreviewData(Y=Y, U=U, lag_tt=lag_tt, cell_id=cell_id,
                          meta=meta, cells=cells)


def _cell_key(cell: Dict[str, Any]) -> str:
    """Filename-safe identity ``M<m>_TE<te>_<band>`` for a manifest cell."""
    te = float(cell.get("target_te", float("nan")))
    return f"M{int(cell['M'])}_TE{te:g}_{cell.get('band', '')}".replace(".", "p")


def _cell_layout(
    cell: Dict[str, Any], meta: Dict[str, Any],
) -> Dict[str, Dict[str, Tuple[int, int]]]:
    r"""Per-cell channel-role block boundaries ``[lo, hi)``.

    The informative block is $[0, M)$ on both sides; the small-noise / noise
    tails are $M$-invariant (``meta['channel_decomp']``), so the middle
    (self-predictable / distractor) block absorbs the difference.

    Args:
        cell: The manifest cell (carries ``M``).
        meta: The cache ``meta.json`` (carries ``c_y`` / ``c_u`` /
            ``channel_decomp``).

    Returns:
        ``{"Y": {"te", "self", "smallnoise"}, "U": {"te", "dist", "noise"}}``
        with half-open ``(lo, hi)`` index ranges.
    """
    decomp = meta.get("channel_decomp", {}) or {}
    c_y, c_u = int(meta["c_y"]), int(meta["c_u"])
    m = int(cell["M"])
    n_sn = int(decomp.get("n_smallnoise", 13))
    n_noise = int(decomp.get("n_noise", 17))
    return {
        "Y": {
            "te": (0, m),
            "self": (m, c_y - n_sn),
            "smallnoise": (c_y - n_sn, c_y),
        },
        "U": {
            "te": (0, m),
            "dist": (m, c_u - n_noise),
            "noise": (c_u - n_noise, c_u),
        },
    }


def _sample_for_cell(
    data: MixPreviewData, cell_id: int, sample_index: Optional[int] = None,
) -> int:
    r"""Return the global row index of a cell's sample to portray.

    Args:
        data: The loaded cache.
        cell_id: Manifest cell id.
        sample_index: Explicit index into the cell's samples. ``None`` picks
            the **most illustrative** sample -- the one whose lag walk $d_t$
            has the largest spread (a walk that visibly drifts demonstrates
            the within-signal lag regime far better than one that happens to
            hold a single value).

    Raises:
        ValueError: If the cell has no samples in this split.
    """
    rows = np.nonzero(data.cell_id == int(cell_id))[0]
    if rows.size == 0:
        raise ValueError(f"cell_id={cell_id} has no samples in this split.")
    if sample_index is not None:
        return int(rows[min(int(sample_index), rows.size - 1)])
    if data.lag_tt is None:
        return int(rows[0])
    spreads = data.lag_tt[rows].std(axis=1)
    return int(rows[int(np.argmax(spreads))])


def _sample_lag(data: MixPreviewData, row: int, cell: Dict[str, Any]) -> np.ndarray:
    """Per-step true lag for one sample (walk if stamped, else the band mean).

    Args:
        data: The loaded cache.
        row: Global sample row.
        cell: The sample's manifest cell (fallback delay range).

    Returns:
        Integer lag trajectory $(T,)$.
    """
    if data.lag_tt is not None:
        return data.lag_tt[row].astype(int)
    d = int(round(0.5 * (int(cell["delay_min"]) + int(cell["delay_max"]))))
    return np.full(data.T, d, dtype=int)


def _lag_aligned_source(u: np.ndarray, lag: np.ndarray) -> np.ndarray:
    r"""Shift the source by the per-step lag: $\tilde u_t = u_{t - d_t}$.

    Steps with $t < d_t$ (no valid past) are NaN so they simply do not draw.

    Args:
        u: Source channel $(T,)$.
        lag: Integer lag trajectory $(T,)$.

    Returns:
        The aligned source $(T,)$ with leading NaNs.
    """
    T = u.shape[0]
    idx = np.arange(T) - lag
    out = np.full(T, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = u[idx[valid]]
    return out


def _ar1_slope(y: np.ndarray, t0: int) -> float:
    r"""Per-channel OLS AR(1) coefficient $\hat\rho$ of a target channel.

    Args:
        y: Target channel $(T,)$.
        t0: First step of the fit window (skip the un-driven warm-up).

    Returns:
        $\hat\rho = \sum_t y_t y_{t+1} / \sum_t y_t^2$ over the window.
    """
    a, b = y[t0:-1], y[t0 + 1:]
    var = float(np.dot(a, a))
    return float(np.dot(a, b) / var) if var > 0 else 0.0


def _driven_component(
    y: np.ndarray, u_aligned: np.ndarray, *, t0: int,
) -> Tuple[np.ndarray, float]:
    r"""The target's predicted *driven component* from the aligned source.

    From the DGP $Y_t = a_y Y_{t-1} + B_y\, s_{t - d_t} + \varepsilon_t$, the
    target is the AR($a_y$)-**accumulation** of the lagged source, not the
    lagged source itself -- so the visually faithful overlay is the aligned
    source passed through the same AR filter,

    $$
    v_t = \hat\rho\, v_{t-1} + \tilde u_t,
    \qquad \tilde u_t = u_{t - d_t},
    $$

    with $\hat\rho$ estimated from the target by OLS, then z-scored. For a
    transfer-carrying channel $v$ traces the target closely; for a
    zero-transfer pair it is unrelated. (The raw aligned source correlates
    with $Y$ only weakly and unstably, because $Y$ integrates it.)

    Args:
        y: Target channel $(T,)$ (used only to estimate $\hat\rho$).
        u_aligned: Lag-aligned source $\tilde u_t$ with leading NaNs.
        t0: First step of the $\hat\rho$ fit window.

    Returns:
        ``(v, rho_hat)`` -- the z-scored driven component $(T,)$ (NaN before
        the first valid aligned step) and the estimated AR coefficient.
    """
    T = y.shape[0]
    rho = _ar1_slope(y, t0)
    v = np.full(T, np.nan)
    state = 0.0
    started = False
    for ti in range(T):
        u_t = u_aligned[ti]
        if not np.isfinite(u_t):
            if started:
                state = rho * state  # decay through a (rare) interior gap
                v[ti] = state
            continue
        started = True
        state = rho * state + float(u_t)
        v[ti] = state
    finite = np.isfinite(v)
    if finite.sum() >= 8 and np.nanstd(v) > 0:
        v = (v - np.nanmean(v)) / np.nanstd(v)
    return v, rho


def _nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the jointly-finite support (``nan`` if <8 pts)."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 8:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _innovation_xcorr_profile(
    y: np.ndarray, u: np.ndarray, *, lag_max: int, t0: int,
) -> np.ndarray:
    r"""Innovation cross-correlation $\rho(\ell) = \mathrm{corr}(Y_t -
    \hat\rho\,Y_{t-1},\ U_{t-\ell})$ for $\ell = 0, \dots,$ ``lag_max``.

    The raw lagged correlation $\mathrm{corr}(Y_t, U_{t-\ell})$ cannot
    localise the lag here: the source is a near-unit-root oscillator
    ($r = 0.99$, ACF $\approx r^\ell$), so shifting it a few steps barely
    changes the correlation and the profile is almost flat. The target's
    *innovation* strips the AR(1) self-memory: from the DGP $Y_t = a_y
    Y_{t-1} + B_y\, s_{t-d_t} + \varepsilon_t$, the residual $Y_t -
    \hat\rho\,Y_{t-1}$ (with $\hat\rho$ the per-channel OLS slope) is
    $\propto B_y\, s_{t-d_t} + \varepsilon_t$ -- it correlates with the
    source *only through the true delay*, so $\rho(\ell)$ peaks over the
    lags the walk actually occupied.

    Args:
        y: Target channel $(T,)$.
        u: Source channel $(T,)$.
        lag_max: Largest tested shift.
        t0: First target step used (skip the un-driven warm-up).

    Returns:
        Correlation profile $(\text{lag\_max} + 1,)$.
    """
    T = y.shape[0]
    rho_hat = _ar1_slope(y, t0)
    innov = np.full(T, np.nan)
    innov[1:] = y[1:] - rho_hat * y[:-1]
    out = np.full(lag_max + 1, np.nan)
    for ell in range(lag_max + 1):
        lo = max(t0, ell, 1)
        out[ell] = _nan_corr(innov[lo:T], u[lo - ell:T - ell])
    return out


def _caption(fig, text: str) -> None:
    """Add the italic "how to read this" caption used across mixed figures."""
    fig.text(0.5, -0.012, text, ha="center", va="top", fontsize=7.5,
             color=ps.COLOR_GRAY, style="italic", wrap=True)


# =============================================================================
# Figure 1 -- sample anatomy
# =============================================================================

def plot_sample_anatomy(
    data: MixPreviewData,
    cell_id: int,
    out_path: Path,
    *,
    sample_index: Optional[int] = None,
    n_channel_panels: int = 3,
) -> Optional[List[Path]]:
    r"""Render the fully-annotated single-sample anatomy figure for one cell.

    Panels, top to bottom:

    1. The true lag walk $d_t$ with the band $[d_{\min}, d_{\max}]$ shaded.
    2. One panel per shown informative channel pair $m$: source $U_m$ (upper
       band), target $Y_m$ (lower band), the **driven component** -- the
       lag-aligned source $U_m(t - d_t)$ passed through the target's own
       AR($\hat\rho$) filter (:func:`_driven_component`) -- dashed on the
       target band, three colour-matched source/target section pairs joined
       by "$d_t$ steps later" arrows, and the per-channel TE annotation.
    3. A zero-transfer contrast panel (AR(1) distractor source over a
       self-predictable target channel) -- what *no* transfer looks like.
    4. Innovation cross-correlation evidence
       (:func:`_innovation_xcorr_profile`) per shown channel with the true
       delay range shaded and the realised lag occupancy as bars.

    Args:
        data: The loaded cache.
        cell_id: Which manifest cell to portray.
        out_path: Output path *without* extension.
        sample_index: Which of the cell's samples to draw; ``None`` picks the
            most illustrative one (largest lag-walk spread).
        n_channel_panels: How many informative pairs get their own panel
            (capped at the cell's $M$).

    Returns:
        Written file paths, or ``None`` when the cell is missing.
    """
    cell = data.cells.get(int(cell_id))
    if cell is None:
        print(f"[visualize_mixed] cell {cell_id} not in manifest; skipped.")
        return None
    row = _sample_for_cell(data, cell_id, sample_index)
    layout = _cell_layout(cell, data.meta)
    T = data.T
    t = np.arange(T)
    lag = _sample_lag(data, row, cell)
    d_min, d_max = int(cell["delay_min"]), int(cell["delay_max"])
    M = int(cell["M"])
    te_cell = float(cell.get("te_cell_realised", cell.get("target_te", 0.0)))
    te_ch = te_cell / max(M, 1)
    horizon = int(data.meta.get("horizon", 30))
    K = max(1, min(int(n_channel_panels), M))

    heights = [0.85] + [1.5] * K + [1.5, 1.25]
    fig, axes, _ = ps.stacked_figure(heights, width=12.5)

    # --- Panel 0: the true lag walk -----------------------------------------
    ax = axes[0]
    ax.axhspan(d_min, d_max, color=ps.COLOR_LIGHT_GRAY, alpha=0.7, zorder=0)
    ax.fill_between(t, 0, lag, step="post", color=ps.COLOR_BLUE, alpha=0.18,
                    lw=0)
    ax.step(t, lag, where="post", color=ps.COLOR_BLUE, lw=1.3)
    for d_edge, name, va in ((d_min, r"$d_{\min}$", "bottom"),
                             (d_max, r"$d_{\max}$", "top")):
        ax.axhline(d_edge, ls="--", lw=0.7, color=ps.COLOR_GRAY)
        ax.text(0.004 * T, d_edge, f"{name}={d_edge}", fontsize=7,
                va=va, ha="left", color=ps.COLOR_GRAY)
    ax.axvline(T - horizon, ls=":", lw=0.9, color=ps.COLOR_VERMILLION)
    ax.text(T - horizon, d_max + 0.6, r"$T - H$", fontsize=7,
            color=ps.COLOR_VERMILLION, ha="center")
    ax.text(0.995, 0.92,
            rf"sample mean lag $= {float(np.mean(lag)):.1f}$ steps",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=ps.COLOR_GRAY)
    ax.set_ylabel(r"lag $d_t$ (steps)")
    ax.set_ylim(0.0, d_max + 2.0)
    ax.set_title(
        rf"True coupling lag $d_t$ (reflecting random walk):  "
        rf"$Y_{{m,t}} = a_y Y_{{m,t-1}} + B_y\, s_{{m,\,t - d_t}} + "
        rf"\varepsilon_t$ with $d_t$ drifting in $[{d_min}, {d_max}]$ "
        rf"(mean hold $\approx 50$ steps), shared by all $M={M}$ "
        rf"informative channels",
        fontsize=8.5,
    )
    ps.style_axes(ax)
    ps.tighten_xaxis(ax, t)
    ax.tick_params(labelbottom=False)

    # --- Panels 1..K: informative channel pairs ------------------------------
    # Three lag-annotation anchors inside the driven region; the hi-end guard
    # keeps them ordered even for a (hypothetical) very short sequence.
    anchor_lo = max(d_max + 15, 40)
    anchor_hi = max(anchor_lo + 2, T - horizon - 15)
    anchors = np.linspace(anchor_lo, anchor_hi, 3).astype(int)
    for k in range(K):
        ax = axes[1 + k]
        y_m = data.Y[row, :, k]
        u_m = data.U[row, :, k]
        u_al = _lag_aligned_source(u_m, lag)
        v_m, rho_hat = _driven_component(y_m, u_al, t0=d_max + 5)
        r_al = _nan_corr(y_m, v_m)

        # Faint band guides behind the two traces.
        for centre in (0.0, _PAIR_OFFSET):
            ax.axhspan(centre - 3.0, centre + 3.0, color=ps.COLOR_LIGHT_GRAY,
                       alpha=0.35, zorder=0)
        ax.plot(t, u_m + _PAIR_OFFSET, color=ps.COLOR_TEAL_DARK, lw=0.9,
                label=rf"source $U_{{{k}}}(t)$")
        ax.plot(t, y_m, color=ps.COLOR_PURPLE, lw=1.0,
                label=rf"target $Y_{{{k}}}(t)$")
        ax.plot(t, v_m, color=ps.COLOR_VERMILLION, lw=0.9, ls="--",
                alpha=0.95,
                label=(rf"driven component: AR($\hat\rho={rho_hat:.2f}$)-"
                       rf"filtered $U_{{{k}}}(t - d_t)$"))

        # Colour-matched section pairs + "d steps later" arrows.
        for a_i, t_star in enumerate(anchors):
            d_star = int(lag[t_star])
            color = _ANCHOR_COLORS[a_i % len(_ANCHOR_COLORS)]
            for x0, centre in (
                (t_star - d_star, _PAIR_OFFSET),   # source section
                (t_star, 0.0),                      # driven target section
            ):
                ax.fill_betweenx(
                    [centre - 3.0, centre + 3.0],
                    x0 - _SPAN_HALF_W, x0 + _SPAN_HALF_W,
                    color=color, alpha=0.30, zorder=1, lw=0,
                )
            ax.annotate(
                "", xy=(t_star, 3.2), xytext=(t_star - d_star, _PAIR_OFFSET - 3.2),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                connectionstyle="arc3,rad=-0.25"),
                zorder=4,
            )
            ax.text(t_star - d_star / 2.0, _PAIR_OFFSET / 2.0 - 0.2,
                    rf"$d_t={d_star}$", fontsize=7.2, color=color,
                    ha="center", va="center", fontweight="bold")

        ax.set_yticks([0.0, _PAIR_OFFSET])
        ax.set_yticklabels([rf"$Y_{{{k}}}$", rf"$U_{{{k}}}$"])
        ax.set_ylim(-4.2, _PAIR_OFFSET + 4.2)
        ax.text(
            0.995, 0.96,
            rf"informative pair $m={k}$:  TE/channel $\approx$ {te_ch:.3g} nats"
            rf"  |  corr$(Y,\ \mathrm{{driven}})$ = {r_al:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=ps.COLOR_GRAY, lw=0.5, alpha=0.9),
        )
        if k == 0:
            # Legend above the axes so it never collides with the traces.
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005),
                      fontsize=6.8, frameon=False, ncol=3)
        ps.style_axes(ax)
        ps.tighten_xaxis(ax, t)
        ax.tick_params(labelbottom=False)

    # --- Panel K+1: zero-transfer contrast ----------------------------------
    ax = axes[1 + K]
    y_self = data.Y[row, :, layout["Y"]["self"][0]]
    u_dist = data.U[row, :, layout["U"]["dist"][0]]
    v_null, _ = _driven_component(
        y_self, _lag_aligned_source(u_dist, lag), t0=d_max + 5,
    )
    r_null = _nan_corr(y_self, v_null)
    for centre in (0.0, _PAIR_OFFSET):
        ax.axhspan(centre - 3.0, centre + 3.0, color=ps.COLOR_LIGHT_GRAY,
                   alpha=0.35, zorder=0)
    ax.plot(t, u_dist + _PAIR_OFFSET, color=ps.COLOR_GRAY, lw=0.9,
            label="distractor source (AR(1), no coupling)")
    ax.plot(t, y_self, color=ps.COLOR_BLUE, lw=1.0,
            label="self-predictable target (zero TE)")
    ax.plot(t, v_null, color=ps.COLOR_GRAY, lw=0.8, ls="--", alpha=0.7,
            label="driven component of the distractor (no relationship)")
    ax.set_yticks([0.0, _PAIR_OFFSET])
    ax.set_yticklabels([r"$Y^{\mathrm{self}}$", r"$U^{\mathrm{dist}}$"])
    ax.set_ylim(-4.2, _PAIR_OFFSET + 4.2)
    ax.text(
        0.995, 0.96,
        rf"zero-transfer contrast:  TE $= 0$ by construction  |  "
        rf"corr = {r_null:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                  ec=ps.COLOR_GRAY, lw=0.5, alpha=0.9),
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), fontsize=6.8,
              frameon=False, ncol=3)
    ps.style_axes(ax)
    ps.tighten_xaxis(ax, t)
    ax.tick_params(labelbottom=False)

    # --- Panel K+2: innovation cross-correlation evidence --------------------
    # The raw corr(Y_t, U_{t-l}) is nearly flat in l (the source ACF decays as
    # 0.99^l), so the panel uses the target *innovation* Y_t - rho_hat*Y_{t-1},
    # which correlates with the source only through the true delay.
    ax = axes[2 + K]
    lag_max = d_max + 15
    t0 = d_max + 5
    profiles = []
    for k in range(K):
        prof = _innovation_xcorr_profile(
            data.Y[row, :, k], data.U[row, :, k], lag_max=lag_max, t0=t0,
        )
        profiles.append(prof)
        ax.plot(np.arange(lag_max + 1), prof, lw=0.8, alpha=0.6,
                color=ps.PALETTE_PRIMARY[k % len(ps.PALETTE_PRIMARY)],
                label=rf"pair $m={k}$")
    mean_prof = np.nanmean(np.stack(profiles, axis=0), axis=0)
    ax.plot(np.arange(lag_max + 1), mean_prof, lw=1.6, color=ps.COLOR_BLACK,
            label="mean over shown pairs")
    ax.axvspan(d_min, d_max, color=ps.COLOR_LIGHT_GRAY, alpha=0.8, zorder=0)
    ax.text(0.5 * (d_min + d_max), 0.02, "true delay\nrange",
            transform=ax.get_xaxis_transform(), fontsize=6.8,
            color=ps.COLOR_GRAY, ha="center", va="bottom")
    # Realised lag occupancy of this sample, as light bars on a twin axis,
    # with the occupancy-weighted mean lag marked -- the innovation-xcorr
    # mass should concentrate around it.
    vals, counts = np.unique(lag, return_counts=True)
    ax_h = ax.twinx()
    ax_h.bar(vals, counts / counts.sum(), width=0.85,
             color=ps.COLOR_ORANGE, alpha=0.35, zorder=1)
    ax_h.set_ylabel("lag occupancy", color=ps.COLOR_ORANGE, fontsize=7.5)
    ax_h.tick_params(axis="y", colors=ps.COLOR_ORANGE, labelsize=6.5)
    ax_h.grid(False)
    d_bar = float(np.average(vals, weights=counts))
    ax.axvline(d_bar, ls="--", lw=1.0, color=ps.COLOR_ORANGE)
    ax.text(d_bar + 0.3, 0.98, rf"occupied mean lag $\bar d={d_bar:.1f}$",
            transform=ax.get_xaxis_transform(), fontsize=7,
            color=ps.COLOR_ORANGE, ha="left", va="top")
    if np.isfinite(mean_prof).any():
        peak = int(np.nanargmax(mean_prof))
        ax.axvline(peak, ls=":", lw=0.9, color=ps.COLOR_VERMILLION)
        ax.text(peak, np.nanmax(mean_prof),
                rf" peak $\ell={peak}$", fontsize=7,
                color=ps.COLOR_VERMILLION, va="bottom")
    ax.set_xlabel(r"tested delay $\ell$ (steps)")
    ax.set_ylabel("innovation xcorr\n"
                  r"corr$(Y_t - \hat\rho Y_{t-1},\ U_{t-\ell})$")
    ax.set_xlim(0, lag_max)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    ps.style_axes(ax)

    held = "  [HELD-OUT]" if int(cell.get("held_out", 0)) else ""
    fig.suptitle(
        rf"G1_mix sample anatomy -- cell {cell['cell_id']}{held}:  $M={M}$ "
        rf"informative channels,  band `{cell.get('band', '')}` "
        rf"($d \in [{d_min}, {d_max}]$),  cell TE $= {te_cell:.3g}$ nats "
        rf"($\approx {te_ch:.3g}$/channel),  $B_y = "
        rf"{float(cell.get('B_y_scalar', float('nan'))):.4g}$",
        fontsize=11,
    )
    _caption(
        fig,
        "One sample, fully annotated. Top: the drifting true lag d_t. Middle: "
        "each informative source channel (upper band) drives its target "
        "channel (lower band) d_t steps later -- the dashed trace is the "
        "lag-aligned source passed through the target's own AR filter (the "
        "DGP integrates the lagged source, so this is the component Y "
        "actually follows), and the matching coloured sections show which "
        "source segment produces which target segment. The contrast panel "
        "shows what zero transfer looks like. Bottom: the innovation "
        "cross-correlation corr(Y_t - rho*Y_{t-1}, U_{t-l}) concentrates "
        "over the lags the walk occupied (orange bars / dashed mean).",
    )
    return ps.save_figure(fig, out_path)


# =============================================================================
# Figure 2 -- TE x lag-band gallery
# =============================================================================

def plot_te_lag_gallery(
    data: MixPreviewData,
    out_path: Path,
    *,
    gallery_m: Optional[int] = None,
    sample_index: Optional[int] = None,
) -> Optional[List[Path]]:
    r"""Render the (TE level) $\times$ (lag band) gallery at one fixed $M$.

    Every panel shows the same compact view -- the target $Y_0$ with the
    lag-aligned source $U_0(t - d_t)$ overlaid -- so scanning a column shows
    transfer entropy growing from the $\mathrm{TE} = 0$ null (no relationship)
    to tight tracking, and scanning a row shows the same TE realised through
    different lag bands.

    Args:
        data: The loaded cache.
        out_path: Output path *without* extension.
        gallery_m: The fixed informative-channel count; ``None`` picks the
            median $M$ present in the manifest.
        sample_index: Which sample of each cell to draw.

    Returns:
        Written file paths, or ``None`` when no suitable cells exist.
    """
    import matplotlib.pyplot as plt

    if not data.cells:
        print("[visualize_mixed] empty manifest; gallery skipped.")
        return None
    ms = sorted({int(c["M"]) for c in data.cells.values()})
    m_star = int(gallery_m) if gallery_m is not None else ms[len(ms) // 2]
    cells = [c for c in data.cells.values() if int(c["M"]) == m_star]
    if not cells:
        print(f"[visualize_mixed] no cells at M={m_star}; gallery skipped.")
        return None
    te_levels = sorted({float(c["target_te"]) for c in cells})
    band_order = {"short": 0, "mid": 1, "long": 2}
    bands = sorted({str(c["band"]) for c in cells},
                   key=lambda b: band_order.get(b, 99))
    lut = {(float(c["target_te"]), str(c["band"])): c for c in cells}

    n_rows, n_cols = len(te_levels), len(bands)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.1 * n_cols, 1.75 * n_rows),
        squeeze=False, sharex=True, sharey=True,
    )
    t = np.arange(data.T)
    for ri, te in enumerate(te_levels):
        for ci, band in enumerate(bands):
            ax = axes[ri][ci]
            cell = lut.get((te, band))
            if cell is None:
                ax.text(0.5, 0.5, "cell trimmed\n(below MC floor)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=7.5, color=ps.COLOR_GRAY)
                ps.style_axes(ax)
                continue
            row = _sample_for_cell(data, int(cell["cell_id"]), sample_index)
            lag = _sample_lag(data, row, cell)
            t0 = int(cell["delay_max"]) + 5
            # Per-channel corr is noisy (a single z-scored channel at T=300
            # with rho~0.95 has ~30 effective DOF, SE~0.18); the M-channel
            # average is the stable summary. Display the channel whose corr
            # is closest to that average (representative, not cherry-picked).
            corrs, driven = [], []
            for m in range(int(cell["M"])):
                v_m, _ = _driven_component(
                    data.Y[row, :, m],
                    _lag_aligned_source(data.U[row, :, m], lag), t0=t0,
                )
                driven.append(v_m)
                corrs.append(_nan_corr(data.Y[row, :, m], v_m))
            corrs_arr = np.asarray(corrs, dtype=float)
            if np.isfinite(corrs_arr).any():
                r_mean = float(np.nanmean(corrs_arr))
                r_std = float(np.nanstd(corrs_arr))
                m_show = int(np.nanargmin(np.abs(corrs_arr - r_mean)))
            else:  # degenerate (constant channels) -- still draw something
                r_mean = r_std = float("nan")
                m_show = 0
            te_real = float(cell.get("te_cell_realised", te))
            ax.plot(t, data.Y[row, :, m_show], color=ps.COLOR_PURPLE, lw=0.9,
                    label=r"target $Y_m$")
            ax.plot(t, driven[m_show], color=ps.COLOR_VERMILLION, lw=0.8,
                    ls="--", alpha=0.95,
                    label=r"driven component (AR-filtered $U_m(t-d_t)$)")
            held = int(cell.get("held_out", 0))
            ax.text(
                0.99, 0.94,
                (rf"TE$={te_real:.2f}$ nats,  corr$={r_mean:.2f}\pm{r_std:.2f}$"
                 rf" over $M$ ch."
                 + ("  [held-out]" if held else "")),
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=ps.COLOR_GRAY, lw=0.4, alpha=0.9),
            )
            if ri == 0:
                dmin, dmax = int(cell["delay_min"]), int(cell["delay_max"])
                ax.set_title(
                    rf"band `{band}`  ($d \in [{dmin}, {dmax}]$)",
                    fontsize=9,
                )
            if ci == 0:
                ax.set_ylabel(rf"target TE $= {te:g}$" + "\n(z-scored)",
                              fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel(r"timestep $t$")
            if ri == 0 and ci == 0:
                ax.legend(loc="lower left", fontsize=6.5, frameon=False,
                          ncol=2)
            ps.style_axes(ax)
            ps.tighten_xaxis(ax, t)
    fig.suptitle(
        rf"G1_mix transfer-entropy $\times$ lag-band gallery at $M={m_star}$ "
        rf"informative channels (representative channel pair, one sample "
        rf"per cell)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _caption(
        fig,
        "Rows: target block TE from the zero-coupling null upward; columns: "
        "lag band. Each panel overlays the driven component -- the "
        "lag-aligned source passed through the target's own AR filter -- on "
        "its target. Reading down a column, the target tracks the driven "
        "component ever more tightly (corr rises with TE); at TE=0 the "
        "dashed trace is unrelated to the target. Across a row the same "
        "information arrives through wider lag ranges.",
    )
    return ps.save_figure(fig, out_path)


# =============================================================================
# Figure 3 -- channel atlas
# =============================================================================

def plot_channel_atlas(
    data: MixPreviewData,
    cell_id: int,
    out_path: Path,
    *,
    sample_index: Optional[int] = None,
) -> Optional[List[Path]]:
    r"""Render channels $\times$ time heatmaps of the full $Y$ / $U$ buffers.

    One sample, every channel: the informative TE block, the self-predictable
    / distractor middle, and the small-noise / pure-noise tails are separated
    by dashed lines and labelled, so the channel budget the model must search
    through is visible at a glance.

    Args:
        data: The loaded cache.
        cell_id: Which manifest cell's sample to draw.
        out_path: Output path *without* extension.
        sample_index: Which of the cell's samples to draw.

    Returns:
        Written file paths, or ``None`` when the cell is missing.
    """
    cell = data.cells.get(int(cell_id))
    if cell is None:
        print(f"[visualize_mixed] cell {cell_id} not in manifest; skipped.")
        return None
    row = _sample_for_cell(data, cell_id, sample_index)
    layout = _cell_layout(cell, data.meta)
    T = data.T
    M = int(cell["M"])
    te_cell = float(cell.get("te_cell_realised", cell.get("target_te", 0.0)))

    fig, axes, caxes = ps.stacked_figure([1.7, 1.7], colorbar=[True, True],
                                         width=12.0)
    panels = (
        (axes[0], caxes[0], data.Y[row].T, "Y", "target  $Y$",
         (("te", rf"TE ({M})"), ("self", "self-predictable"),
          ("smallnoise", "small noise"))),
        (axes[1], caxes[1], data.U[row].T, "U", "source  $U$",
         (("te", rf"TE ({M})"), ("dist", "AR(1) distractor"),
          ("noise", "pure noise"))),
    )
    vmax = 3.0
    for ax, cax, img, side, label, blocks in panels:
        c_n = img.shape[0]
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(0, T, 0, c_n))
        for name, text in blocks:
            lo, hi = layout[side][name]
            if hi > lo:
                if lo > 0:
                    ax.axhline(lo, ls="--", lw=0.8, color=ps.COLOR_BLACK)
                ax.text(-0.012 * T, 0.5 * (lo + hi), text, fontsize=7.5,
                        rotation=90, ha="right", va="center",
                        color=ps.COLOR_BLACK)
        ax.set_ylabel(f"{label}  channel")
        ps.attach_colorbar(fig, im, cax, label="z-score")
        ps.style_axes(ax)
        ax.grid(False)
        ax.set_xlim(0, T)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel(r"timestep $t$")

    fig.suptitle(
        rf"G1_mix channel atlas -- cell {cell['cell_id']}: every channel of "
        rf"one sample  ($M={M}$ informative, band `{cell.get('band', '')}`, "
        rf"cell TE $= {te_cell:.3g}$ nats)",
        fontsize=11,
    )
    _caption(
        fig,
        "Channels x time for the full target (top, 87 ch) and source "
        "(bottom, 101 ch) buffers. Only the bottom TE block carries "
        "source-to-target transfer; the self-predictable / distractor middle "
        "is forecastable but transfer-free, and the small-noise / pure-noise "
        "tails are nuisance. The model must find the M informative rows "
        "among all channels.",
    )
    return ps.save_figure(fig, out_path)


# =============================================================================
# Driver
# =============================================================================

def _select_anatomy_cells(data: MixPreviewData, m_star: int) -> List[int]:
    r"""Pick a small, maximally-informative anatomy set at one $M$.

    The null cell ($\mathrm{TE} = 0$, first band) plus the highest-TE cell of
    every band -- the extremes that span the whole behaviour range.

    Args:
        data: The loaded cache.
        m_star: The fixed informative-channel count.

    Returns:
        Cell ids, de-duplicated, in (band, TE) order.
    """
    cells = [c for c in data.cells.values() if int(c["M"]) == m_star]
    band_order = {"short": 0, "mid": 1, "long": 2}
    chosen: List[int] = []
    nulls = sorted(
        (c for c in cells if float(c["target_te"]) == 0.0),
        key=lambda c: band_order.get(str(c["band"]), 99),
    )
    if nulls:
        chosen.append(int(nulls[0]["cell_id"]))
    for band in sorted({str(c["band"]) for c in cells},
                       key=lambda b: band_order.get(b, 99)):
        in_band = [c for c in cells if str(c["band"]) == band
                   and float(c["target_te"]) > 0.0]
        if in_band:
            top = max(in_band, key=lambda c: float(c["target_te"]))
            cid = int(top["cell_id"])
            if cid not in chosen:
                chosen.append(cid)
    return chosen


def render_mixed_previews(
    cache_dir: Path,
    out_dir: Optional[Path] = None,
    *,
    split: str = "test",
    gallery_m: Optional[int] = None,
    anatomy_cells: Optional[Sequence[int]] = None,
    sample_index: Optional[int] = None,
    n_channel_panels: int = 3,
) -> List[Path]:
    r"""Render every preview figure for one mixed cache.

    Writes, under ``<cache_dir>/previews/`` (or ``out_dir``):

    * ``anatomy_<M>_<TE>_<band>`` -- one per selected cell (default: the
      $\mathrm{TE} = 0$ null plus the highest-TE cell of each band at the
      median $M$);
    * ``te_lag_gallery`` -- the TE $\times$ band case grid;
    * ``channel_atlas_<...>`` -- the full-channel heatmap for the highest-TE
      anatomy cell.

    Args:
        cache_dir: The ``data/G1_mix/<tag>/`` cache directory.
        out_dir: Output directory (default ``<cache_dir>/previews``).
        split: Which split to read.
        gallery_m: Fixed $M$ for the gallery / anatomy selection (``None``
            picks the median $M$ in the manifest).
        anatomy_cells: Explicit cell ids for anatomy figures (overrides the
            automatic selection).
        sample_index: Which sample of each cell to draw.
        n_channel_panels: Informative pairs per anatomy figure.

    Returns:
        All written file paths.
    """
    ps.apply_style()
    data = load_mix_preview_data(Path(cache_dir), split=split)
    out_root = Path(out_dir) if out_dir is not None else Path(cache_dir) / _PREVIEW_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    ms = sorted({int(c["M"]) for c in data.cells.values()})
    if not ms:
        print(f"[visualize_mixed] no manifest cells in {cache_dir}; nothing to do.")
        return []
    m_star = int(gallery_m) if gallery_m is not None else ms[len(ms) // 2]
    cell_ids = (
        [int(c) for c in anatomy_cells] if anatomy_cells is not None
        else _select_anatomy_cells(data, m_star)
    )

    written: List[Path] = []
    for cid in cell_ids:
        cell = data.cells.get(cid)
        if cell is None:
            print(f"[visualize_mixed] cell {cid} not in manifest; skipped.")
            continue
        paths = plot_sample_anatomy(
            data, cid, out_root / f"anatomy_{_cell_key(cell)}",
            sample_index=sample_index, n_channel_panels=n_channel_panels,
        )
        written.extend(paths or [])
    written.extend(
        plot_te_lag_gallery(
            data, out_root / "te_lag_gallery",
            gallery_m=m_star, sample_index=sample_index,
        ) or []
    )
    if cell_ids:
        atlas_cell = data.cells.get(cell_ids[-1])
        if atlas_cell is not None:
            written.extend(
                plot_channel_atlas(
                    data, int(atlas_cell["cell_id"]),
                    out_root / f"channel_atlas_{_cell_key(atlas_cell)}",
                    sample_index=sample_index,
                ) or []
            )
    print(f"[visualize_mixed] wrote {len(written)} files -> {out_root}")
    return written


# =============================================================================
# CLI / edit-and-run
# =============================================================================

def main() -> None:
    """CLI entry point: resolve the cache path from the config and render."""
    parser = argparse.ArgumentParser(
        description="Journal-quality data-anatomy previews for a G1_mix cache."
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--cache-tag", type=str, default="G1_mix_base",
                        dest="cache_tag",
                        help="cache tag under data/G1_mix/")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--gallery-m", type=int, default=None, dest="gallery_m")
    parser.add_argument("--sample-index", type=int, default=None,
                        dest="sample_index",
                        help="explicit per-cell sample; omit to auto-pick "
                             "the sample with the most mobile lag walk")
    parser.add_argument("--n-channel-panels", type=int, default=3,
                        dest="n_channel_panels")
    parser.add_argument("--out-dir", type=Path, default=None, dest="out_dir")
    parser.add_argument("--data-dir", type=str, default=None, dest="data_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    config["experiment"]["benchmark"] = _BENCHMARK
    apply_path_overrides(config, {"data_dir": args.data_dir})
    resolve_active_benchmark(config)
    cache_dir = (
        resolve_user_path(config["paths"]["data_dir"]) / _BENCHMARK
        / args.cache_tag
    )
    render_mixed_previews(
        cache_dir, args.out_dir, split=args.split, gallery_m=args.gallery_m,
        sample_index=args.sample_index,
        n_channel_panels=args.n_channel_panels,
    )


if __name__ == "__main__":
    CONFIG_PATH = _DEFAULT_CONFIG
    RUN_CONFIG = {
        "cache_tag": "G1_mix_base",  # cache under data/G1_mix/
        "split": "test",
        "gallery_m": None,            # None -> median M in the manifest
        "sample_index": None,  # None -> auto-pick the most mobile lag walk
        "n_channel_panels": 3,
        "out_dir": None,              # None -> <cache>/previews/
        "data_dir": None,             # None -> config paths.data_dir
    }

    if len(sys.argv) > 1:
        main()
    else:
        config = load_config(CONFIG_PATH)
        config["experiment"]["benchmark"] = _BENCHMARK
        apply_path_overrides(config, {"data_dir": RUN_CONFIG["data_dir"]})
        resolve_active_benchmark(config)
        cache_dir = (
            resolve_user_path(config["paths"]["data_dir"]) / _BENCHMARK
            / RUN_CONFIG["cache_tag"]
        )
        render_mixed_previews(
            cache_dir, RUN_CONFIG["out_dir"], split=RUN_CONFIG["split"],
            gallery_m=RUN_CONFIG["gallery_m"],
            sample_index=RUN_CONFIG["sample_index"],
            n_channel_panels=RUN_CONFIG["n_channel_panels"],
        )
