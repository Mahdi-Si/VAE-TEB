r"""``samples`` -- the raw-trace diagnostic page; port, owner E1-F (``notes/EVAL_PLAN.md`` §2).

**Question.** What does the model forecast, in clinical units, against the trace a clinician
reads? The page draws the level forecast in bpm with its ±2σ band over the raw FHR, a variability
lane (rms-Δ bpm), and UP with the detected contractions shaded, on one shared time axis above the
shared latent, KL and lag rows.

**What is reused, and what is new.** The selection (a stratified draw, a class-balanced draw and
the extremes of three headline metrics), the dense re-forward, the identity checks, the two page
variants, the shared colour limits and the manifest all belong to the shared CFS ``samples``
analysis and run unchanged. That analysis draws through the task's three page seams
(``forecast_rows``, ``forecast_extra_rows``, ``input_stream_panels``). The patch task inherits
seams that draw CFS feature lanes. This module hands the shared analysis a thin view of the task
whose three seams are the patch ones below; every other attribute delegates to the real task.

**Rows** (full page, 10): ``raw`` (UP, mmHg, contractions shaded); ``forecast`` (raw FHR plus
the tiled level forecast, bpm); ``variability`` (rms-Δ bpm per 0.25 s, log axis, with the 0.25 bpm
monitor resolution and the ``variability_eps`` floor marked); ``input_target`` and
``input_source`` (the patch tokens the encoders read, missing tokens blank); then the shared
latent, per-dimension KL, ``K_t``, lag attention and KL-by-lag rows. The compact page keeps
``raw``, ``forecast``, both inputs, the latent, ``K_t`` and both lag maps.

**Tiling.** Anchors ``w, w + H, w + 2H, ...`` below ``T - H``, so windows abut without overlap
(``lag_attn_rws.sample_page.raw_forecast_rows``). Each window is decoded from one latent and is
drawn as 4 s steps, because the target is a 4 s patch summary. The band is the AR(1) marginal
±2σ, ``v_τ = σ²_τ + φ² v_{τ-1}``, when the model scores the AR(1) innovation. ``full`` is the
forward's own decode at one posterior draw, and ``base`` is decoded at the prior mean
(``base_decode: mean``).
"""
from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from teb_vae.lag_attn_cfs.eval import events, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval.analyses import samples as _shared
from teb_vae.lag_attn_cfs.eval.metrics import forecast_likelihood_terms
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD
from teb_vae.lag_attn_rws.sample_page import (
    FORECAST_ROW,
    RAW_ROW,
    ForecastRowInputs,
    InputStreamPanel,
    _input_stream_row,
)
from teb_vae.lag_attn_transformer_patch.eval import raw
from teb_vae.lag_attn_transformer_patch.eval.raw import SummaryUnits

#: ``(headline_name, key)`` pairs registered as ``("samples", "headline", key)``.
HEADLINE: Tuple[Tuple[str, str], ...] = ()

#: The rows this page adds below the forecast row, ``(name, height ratio)``: the variability lane,
#: then the two patch input streams. The inputs are drawn by this page's forecast seam rather than
#: the shared input rows, which shade the warm-up; a patch's warm-up restricts anchors, not inputs.
VARIABILITY_ROW = "variability"
INPUT_ROWS: Tuple[Tuple[str, float], ...] = (("input_target", 1.25), ("input_source", 1.25))
EXTRA_ROWS: Tuple[Tuple[str, float], ...] = ((VARIABILITY_ROW, 1.1), *INPUT_ROWS)

#: Rows on the full page: raw, forecast, variability, two inputs, five latent/lag rows.
EXPECTED_PAGE_ROWS = 2 + len(EXTRA_ROWS) + 5

#: Band half-width in predictive standard deviations (the shared raw page's).
BAND_SIGMAS = 2.0

#: The FHR monitor's resolution, bpm: the smallest non-zero beat-to-beat step.
FHR_RESOLUTION_BPM = 0.25

#: Fixed log limits of the variability lane, rms-Δ bpm per 0.25 s, so pages compare by eye.
VARIABILITY_LIMITS_BPM: Tuple[float, float] = (0.05, 20.0)

#: Margin around the segment's own FHR range on the zoomed forecast row, bpm.
FORECAST_ZOOM_MARGIN_BPM = 10.0

#: Colours: the shared raw page's (base grey dashed, full vermillion).
_BASE = figures.COLOR_GRAY
_FULL = figures.COLOR_VERMILLION
_CONTRACTION = figures.COLOR_GREEN


class PatchPageTask:
    """The task with the patch page seams; every other attribute is the real task's.

    The shared ``samples`` analysis reads its seams off the task by name (``page_seams``). This
    view swaps the three names and nothing else, so the forward, the device transfer and the
    hyper-parameters the analysis reads stay the loaded task's own.
    """

    def __init__(self, task: Any, *, forecast_rows: Any, input_stream_panels: Any) -> None:
        self._task = task
        self.forecast_rows = forecast_rows
        self.forecast_extra_rows = EXTRA_ROWS
        self.input_stream_panels = input_stream_panels

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task, name)


# =============================================================================
# Small helpers
# =============================================================================
def _field(batch: Any, name: str) -> Any:
    """One batch field, mapping or attribute style; ``None`` when absent."""
    return batch.get(name) if isinstance(batch, dict) else getattr(batch, name, None)


def tile_positions(anchors: np.ndarray, valid: np.ndarray, *, warmup: int, t_valid: int, horizon: int) -> List[int]:
    """Positions on the decoded anchor axis of the abutting tiles ``w, w + H, ...`` below ``T - H``."""
    where = {int(step): position for position, step in enumerate(anchors) if bool(valid[position])}
    return [where[step] for step in range(int(warmup), int(t_valid), int(horizon)) if step in where]


def tiled_series(
    block: np.ndarray, anchors: np.ndarray, positions: Sequence[int], n_steps: int
) -> np.ndarray:
    """Lay the tiles of one ``(A, H)`` block on the token grid ``(T,)``; ``NaN`` off the tiles.

    Horizon step ``τ`` of the anchor at step ``a`` is token ``a + 1 + τ``.
    """
    series = np.full(int(n_steps), np.nan)
    horizon = block.shape[1]
    for position in positions:
        steps = int(anchors[position]) + 1 + np.arange(horizon)
        keep = steps < n_steps
        series[steps[keep]] = block[position, keep]
    return series


def _stairs(ax: Any, series: np.ndarray, edges: np.ndarray, **kwargs: Any) -> None:
    """A token-grid series as 4 s steps; ``NaN`` tokens are gaps."""
    ax.stairs(series, edges, baseline=None, **kwargs)


def _band(ax: Any, lo: np.ndarray, hi: np.ndarray, edges: np.ndarray, **kwargs: Any) -> None:
    """A token-grid band as 4 s steps (``fill_between`` on the left edges, step ``post``)."""
    x = np.append(edges[:-1], edges[-1])
    ax.fill_between(x, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step="post", linewidth=0, **kwargs)


def contraction_spans(up: np.ndarray, valid: Optional[np.ndarray]) -> List[Tuple[float, float]]:
    """Contraction ``(onset, end)`` in seconds from the shared detector on loader-unit UP."""
    try:
        found = events.detect_contractions(up, valid=valid)
    except Exception:  # noqa: BLE001 - a page is worth drawing without its shading
        return []
    fs = float(events.FS_RAW)
    return [(float(on) / fs, float(end) / fs) for on, end in zip(found["onset_raw"], found["end_raw"])]


def _shade(ax: Any, spans: Sequence[Tuple[float, float]], *, label: bool = False) -> None:
    """Shade the contractions on one row."""
    for index, (onset, end) in enumerate(spans):
        ax.axvspan(onset, end, color=_CONTRACTION, alpha=0.12, linewidth=0, zorder=0,
                   label="contraction" if (label and index == 0) else None)


def _window_edges(ax: Any, anchors: np.ndarray, positions: Sequence[int], horizon: int, seconds: float) -> None:
    """Dashed rules where one tiled window ends and the next begins."""
    if not positions:
        return
    edges = [(int(anchors[p]) + 1) * seconds for p in positions]
    edges.append((int(anchors[positions[-1]]) + 1 + horizon) * seconds)
    for edge in edges:
        ax.axvline(edge, color=figures.COLOR_GRAY, linewidth=0.5, linestyle="--", alpha=0.7, zorder=1)


# =============================================================================
# The forecast seam
# =============================================================================
def patch_forecast_rows(
    rows: ForecastRowInputs, *, model: Any, units: SummaryUnits, phi: Optional[np.ndarray]
) -> None:
    """Draw the raw UP row, the level-forecast row, the variability lane and the two input rows.

    Args:
        rows: The shared layout's row inputs; ``rows.target`` is the standardized summary target
            ``(B, T, 2)`` (the eval task's ``_build_raw_target``), and ``rows.batch`` the loader
            batch, read for the raw ``fhr``/``up`` and ``weight``.
        units: Standardized summaries ↔ bpm.
        phi: The AR(1) coefficient per channel ``(2,)``, or ``None`` without the AR term.
        model: The eval view, for the patch streams of the input rows.
    """
    i = int(rows.sample_index)
    geometry = rows.geometry
    n_steps, horizon, warmup, t_valid = geometry.t, geometry.horizon, geometry.warmup, geometry.t_valid
    seconds = float(rows.t_max) / float(n_steps)
    edges = np.arange(n_steps + 1, dtype=np.float64) * seconds
    batch = rows.batch
    weight = raw.host(_field(batch, "weight")[i])
    fhr_z = raw.host(_field(batch, "fhr")[i])
    up_value = _field(batch, "up")
    up_z = None if up_value is None else raw.host(up_value[i])
    scales = {"fhr": (units.fhr_mean, units.fhr_std), "up": (units.up_mean, units.up_std)}
    raw_valid = traces.raw_validity_of(weight, fhr_z.size)
    fhr_bpm, fhr_unit = traces.physical_raw(fhr_z, "fhr", scales, raw_valid)
    spans = [] if up_z is None else contraction_spans(up_z, np.isfinite(up_z))

    # ---- raw: UP and its contractions -------------------------------------------------------
    ax, cax = rows.row_axes(RAW_ROW)
    if up_z is None:
        ax.text(0.5, 0.5, "UP not loaded", transform=ax.transAxes, ha="center", va="center")
    else:
        up_phys, up_unit = traces.physical_raw(up_z, "up", scales, np.isfinite(up_z))
        traces.draw_raw_signal(ax, np.asarray(rows.time_raw), up_phys, "up", up_unit)
    _shade(ax, spans, label=True)
    ax.set_title(f"Raw UP, {len(spans)} contraction(s) detected (shaded)", fontsize=9, pad=6)
    ax.set_xlabel("Time (s)", fontsize=8)
    if spans:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    figures.style_axes(ax)
    rows.finalise_time_axis(ax)
    cax.set_visible(False)

    outs = rows.outs
    anchors = raw.host(outs["anchor_index"][i]).astype(np.int64)
    valid = raw.host(outs["anchor_valid"][i]) > 0.5
    positions = tile_positions(anchors, valid, warmup=warmup, t_valid=t_valid, horizon=horizon)
    target = raw.host(rows.target[i])                                       # (T, 2) standardized
    token_valid = weight >= VALID_THRESHOLD
    target = np.where(token_valid[:, None], target, np.nan)
    tiled = np.isfinite(tiled_series(np.zeros((len(anchors), horizon)), anchors, positions, n_steps))

    def _branch(name: str, channel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mean, lo, hi)`` of one branch's channel on the token grid, standardized."""
        mu = raw.host(outs[f"mu_{name}"][i][..., channel])                 # (A, H)
        logvar = raw.host(outs[f"logvar_{name}"][i][..., channel])
        coef = None if phi is None else float(phi[channel])
        sigma = np.stack([raw.marginal_sigma(row, coef) for row in logvar])
        return tuple(
            tiled_series(block, anchors, positions, n_steps)
            for block in (mu, mu - BAND_SIGMAS * sigma, mu + BAND_SIGMAS * sigma)
        )

    # ---- forecast: raw FHR and the tiled level forecast, bpm -------------------------------
    if rows.wants(FORECAST_ROW):
        ax, cax = rows.row_axes(FORECAST_ROW)
        _shade(ax, spans)
        ax.plot(np.asarray(rows.time_raw), fhr_bpm, color=figures.COLOR_BLACK, linewidth=0.35,
                alpha=0.55, label="raw FHR", zorder=2)
        level_truth = units.level_bpm(np.where(tiled, target[:, 0], np.nan))
        _stairs(ax, level_truth, edges, color=figures.COLOR_BLACK, linewidth=0.9,
                label="true level (4 s mean)", zorder=4)
        for name, colour, style in (("base", _BASE, "--"), ("full", _FULL, "-")):
            mean, lo, hi = (units.level_bpm(series) for series in _branch(name, 0))
            _band(ax, lo, hi, edges, color=colour, alpha=0.18, zorder=1)
            _stairs(ax, mean, edges, color=colour, linewidth=0.9, linestyle=style,
                    label=f"{name} ({'$z^p$' if name == 'base' else '$z^q$'})", zorder=3)
        _window_edges(ax, anchors, positions, horizon, seconds)
        finite = fhr_bpm[np.isfinite(fhr_bpm)]
        if fhr_unit == traces.RAW_SIGNAL_UNITS["fhr"] and finite.size:
            low, high = np.percentile(finite, [1.0, 99.0])
            ax.set_ylim(max(traces.RAW_SIGNAL_LIMITS["fhr"][0], low - FORECAST_ZOOM_MARGIN_BPM),
                        min(traces.RAW_SIGNAL_LIMITS["fhr"][1], high + FORECAST_ZOOM_MARGIN_BPM))
        ax.set_title(
            f"Level forecast over raw FHR (bpm), tiled {horizon}-step windows, mean $\\pm$ "
            f"{BAND_SIGMAS:.0f}$\\sigma$" + (" (AR(1) marginal)" if phi is not None else "")
            + "; y zoomed to the segment",
            fontsize=9, pad=6,
        )
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel(f"FHR ({fhr_unit})", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95, ncol=4)
        figures.style_axes(ax)
        rows.finalise_time_axis(ax)
        cax.set_visible(False)

    # ---- variability: rms-Δ bpm per 0.25 s, log axis ----------------------------------------
    if rows.wants(VARIABILITY_ROW):
        ax, cax = rows.row_axes(VARIABILITY_ROW)
        floor = VARIABILITY_LIMITS_BPM[0]
        _shade(ax, spans)

        def _bpm(series: np.ndarray) -> np.ndarray:
            return np.clip(units.variability_bpm(series), floor, None)

        _stairs(ax, _bpm(target[:, 1]), edges, color=figures.COLOR_BLACK, linewidth=0.7,
                label="realised (every valid patch)", zorder=4)
        for name, colour, style in (("base", _BASE, "--"), ("full", _FULL, "-")):
            mean, lo, hi = (_bpm(series) for series in _branch(name, 1))
            _band(ax, lo, hi, edges, color=colour, alpha=0.18, zorder=1)
            _stairs(ax, mean, edges, color=colour, linewidth=0.9, linestyle=style, label=name, zorder=3)
        eps_bpm = units.eps * units.fhr_std
        # summary_stats.py derives eps from the 0.25 bpm resolution, so the two usually coincide.
        same = abs(eps_bpm - FHR_RESOLUTION_BPM) <= 0.05 * FHR_RESOLUTION_BPM
        ax.axhline(FHR_RESOLUTION_BPM, color=figures.COLOR_BLUE, linewidth=0.6, linestyle=":",
                   label=f"monitor resolution {FHR_RESOLUTION_BPM:g} bpm"
                   + (f" = variability_eps ({eps_bpm:.2g} bpm)" if same else ""))
        if not same and eps_bpm > floor:
            ax.axhline(eps_bpm, color=figures.COLOR_PURPLE, linewidth=0.6, linestyle=":",
                       label=f"variability_eps = {eps_bpm:.2g} bpm")
        _window_edges(ax, anchors, positions, horizon, seconds)
        ax.set_yscale("log")
        ax.set_ylim(*VARIABILITY_LIMITS_BPM)
        ax.set_title("Variability: rms of the within-patch first difference (bpm per 0.25 s), log scale",
                     fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("rms Δ (bpm)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95, ncol=5)
        figures.style_axes(ax)
        rows.finalise_time_axis(ax)
        cax.set_visible(False)

    # ---- the two patch streams, on the page's time axis, unshaded: every token is real input -----
    y_patch, u_patch = raw.patch_streams(model, _field(batch, "fhr"), _field(batch, "up"), _field(batch, "weight"))
    for panel in patch_stream_panels(model, (y_patch, y_patch[..., :0], u_patch), sample_index=i):
        if not rows.wants(f"input_{panel.name}"):
            continue
        ax, cax = rows.row_axes(f"input_{panel.name}")
        image = _input_stream_row(ax, panel, t_max=float(rows.t_max), seconds_per_step=seconds)
        rows.heatmap_spines(ax)
        rows.attach_cbar(cax, image, "value")
        ax.set_xlim(0.0, float(rows.t_max))


# =============================================================================
# The input-stream seam
# =============================================================================
def patch_stream_panels(model: Any, inputs: Sequence[Any], *, sample_index: int) -> Tuple[InputStreamPanel, ...]:
    """The two patch streams the encoders read, ``(T, 2R)`` = value | within-patch Δ, z units.

    A missing token (validity channel ``m_t - 1 = -1``) is blanked: it is replaced by the learned
    ``missing`` embedding and its stored zeros are not what the encoder saw.

    Args:
        model: The eval view (``raw_per_step``, ``source_validity``).
        inputs: ``(y_st, y_ph, u_stream)`` = ``(y_patch, empty, u_patch)``.
        sample_index: Which sample to draw.
    """
    r = int(model.raw_per_step)
    panels = []
    for name, stream, field, rule in (
        ("target", inputs[0], "FHR", "weight"),
        ("source", inputs[2], "UP", str(getattr(model, "source_validity", "finite"))),
    ):
        tokens = raw.host(stream[int(sample_index)])                       # (T, 2R + 1)
        values = tokens[:, : 2 * r].copy()
        missing = tokens[:, -1] < -0.5
        values[missing] = np.nan
        panels.append(InputStreamPanel(
            name=name,
            values=values,
            delays=np.zeros(2 * r, dtype=np.int64),
            center_hz=np.full(2 * r, np.nan),
            blocks=(("value", 0, r), ("Δ", r, 2 * r)),
            title=(f"{name.capitalize()} input: {field} patches ({r} samples + {r} in-patch Δ), z; "
                   f"{int(missing.sum())} missing token(s) blank (validity: {rule})"),
        ))
    return tuple(panels)


# =============================================================================
# Entry point
# =============================================================================
def run_samples_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render the shared selections through the patch page seams.

    Caps are the shared ones: ``caps.pages`` (stratified, default 10) and ``caps.pages_per_class``
    (default 10); the extremes take up to 10 per tail of three metrics. A pass without a model
    records the shared skip.
    """
    task = getattr(context, "task", None)
    if task is None or getattr(context, "loader", None) is None:
        return _shared.run_samples_analysis(context, eval_config=eval_config, output_dir=output_dir, probe=probe)
    model = task.orig_model
    units = SummaryUnits.from_model(model, getattr(context, "config", None), context.loader)
    ar_coef = forecast_likelihood_terms(model)["ar_coef"]
    phi = None if ar_coef is None else ar_coef.detach().cpu().double().numpy()
    view = PatchPageTask(
        task,
        forecast_rows=partial(patch_forecast_rows, model=model, units=units, phi=phi),
        input_stream_panels=lambda *_args, **_kwargs: (),  # drawn by patch_forecast_rows instead
    )
    result = _shared.run_samples_analysis(
        dataclasses.replace(context, task=view), eval_config=eval_config, output_dir=output_dir, probe=probe
    )
    plan = result.setdefault("plan", {})
    plan["expected_page_rows"] = EXPECTED_PAGE_ROWS
    plan["page"] = "patch raw-trace page (level bpm, variability lane, UP contractions)"
    plan["units"] = dataclasses.asdict(units)
    plan["band"] = f"±{BAND_SIGMAS:g}σ" + (" AR(1) marginal" if phi is not None else " per cell")
    return result
