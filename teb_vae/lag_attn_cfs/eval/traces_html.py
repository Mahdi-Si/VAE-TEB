r"""The per-recording trace as an interactive plotly page, beside the printed figure.

One HTML page per traced recording: the raw signals on top, then the same panels
:func:`~teb_vae.lag_attn_cfs.eval.traces.build_recording_figure` prints, on one shared, zoomable
time axis. The panel definitions are the cell's own :class:`~traces.HeatmapPanel` and
:class:`~traces.LinePanel` tuples, so the page and the PDF cannot come to show different things.

**Every series is painted onto one uniform grid** at the stored step (4 s) for the model
quantities and the raw rate for the signals, segment by segment in ``epoch`` order. Two things
fall out of that for free: a stretch the dataset holds no segment for is an empty column rather
than a neighbour stretched across it, and where two segments overlap -- a dataset stored with an
overlap between consecutive segments -- the later segment overwrites the earlier one, so every
instant carries the **latest** reading in time.

**Logarithmic colour** is a transform of the data, since plotly has no log colour axis: a
non-negative panel is drawn as $\log_{10}$ floored :data:`traces.LOG_PANEL_DECADES` decades
under its own maximum, a signed one as a symmetric signed log, and both colour bars are ticked
in the panel's own units.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hdf5_dataset.hdf5_dataset import RAW_SAMPLING_HZ
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import cohort, traces
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL

#: File extension of the page.
DASHBOARD_EXTENSION = ".html"

#: The raw signals drawn above the model panels, in order: the key in
#: :attr:`traces.SegmentTrace.raw`, the panel title and the axis label.
RAW_PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("fhr", "Fetal heart rate", "FHR"),
    ("up", "Uterine activity", "UA"),
)

#: Row heights in pixels, by row kind.
RAW_ROW_PX = 130
LINE_ROW_PX = 120
HEATMAP_ROW_PX = 170
_TITLE_PX = 90
_FOOTER_PX = 140
_ROW_GAP_PX = 34

#: Colour scales: one perceptual ramp for magnitude, one diverging pair about zero for a signed
#: quantity, as the printed figure.
SEQUENTIAL_SCALE = "Viridis"
DIVERGING_SCALE = "RdBu"

_FONT = dict(family="Inter, Helvetica Neue, Arial, sans-serif", size=11, color="#222222")
_GRID = "#EBEBEB"
_HOURS = 1.0 / cohort.SECONDS_PER_HOUR
_DX_HOURS = float(SECONDS_PER_STEP) * _HOURS


# =============================================================================
# The uniform grid
# =============================================================================
def _grid(t_first: float, t_last: float, step: float) -> Tuple[float, int]:
    """The origin and the column count of a uniform grid covering ``[t_first, t_last]``."""
    return float(t_first), int(round((float(t_last) - float(t_first)) / step)) + 1


def _paint(
    grid: np.ndarray, origin: float, step: float, t: np.ndarray, values: np.ndarray
) -> None:
    """Write ``values`` at times ``t`` into ``grid``; a later call overwrites an earlier one."""
    index = np.rint((np.asarray(t, dtype=np.float64) - origin) / step).astype(np.int64)
    keep = (index >= 0) & (index < grid.shape[0])
    grid[index[keep]] = values[keep]


def _segment_t(segment: traces.SegmentTrace) -> np.ndarray:
    """A segment's anchors on the absolute axis, in seconds."""
    return float(segment.epoch) + np.asarray(segment.anchor, dtype=np.float64) * float(SECONDS_PER_STEP)


def _anchor_axis(recording: traces.RecordingTrace) -> Tuple[float, int]:
    """The stored-step grid every model panel of the recording is painted on."""
    times = np.concatenate([_segment_t(s) for s in recording.segments if len(s.anchor)] or [np.zeros(1)])
    return _grid(times.min(), times.max(), float(SECONDS_PER_STEP))


def _kept_extents(recording: traces.RecordingTrace) -> List[Optional[Tuple[float, float]]]:
    """Each segment's span in seconds after the later segments have overwritten it, or ``None``."""
    starts = [(_segment_t(s)[0] if len(s.anchor) else np.nan) for s in recording.segments]
    extents: List[Optional[Tuple[float, float]]] = []
    for order, segment in enumerate(recording.segments):
        if not len(segment.anchor):
            extents.append(None)
            continue
        t = _segment_t(segment)
        later = [s for s in starts[order + 1:] if np.isfinite(s)]
        end = min(float(t[-1]), (min(later) - float(SECONDS_PER_STEP)) if later else np.inf)
        extents.append((float(t[0]), end) if end >= t[0] else None)
    return extents


def _painted_vector(recording: traces.RecordingTrace, panel: traces.HeatmapPanel, origin: float, n: int) -> Optional[np.ndarray]:
    """One heatmap panel's ``(n, K)`` grid, ``None`` when no segment carries the vector."""
    grid: Optional[np.ndarray] = None
    for segment in recording.segments:
        block = segment.vectors.get(panel.vector)
        if block is None or not len(segment.anchor):
            continue
        block = np.asarray(block, dtype=np.float64)
        if panel.subtract is not None:
            other = segment.vectors.get(panel.subtract)
            if other is None:
                continue
            block = block - np.asarray(other, dtype=np.float64)
        if grid is None:
            grid = np.full((n, block.shape[1]), np.nan)
        _paint(grid, origin, float(SECONDS_PER_STEP), _segment_t(segment), block)
    return grid


def _painted_scalar(recording: traces.RecordingTrace, column: str, origin: float, n: int, *, scored_only: bool = True) -> Optional[np.ndarray]:
    """One per-anchor column's ``(n,)`` grid, lifted at unscored anchors; ``None`` when absent."""
    frame = recording.anchors
    if column not in frame.columns:
        return None
    grid = np.full(n, np.nan)
    for _, cell in frame.groupby("segment_order", sort=True):
        values = np.asarray(cell[column], dtype=np.float64)
        if scored_only:
            values = np.where(np.asarray(cell["contributing"], dtype=bool), values, np.nan)
        _paint(grid, origin, float(SECONDS_PER_STEP), np.asarray(cell["t_abs_sec"], dtype=np.float64), values)
    return grid


def _painted_raw(recording: traces.RecordingTrace, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """A raw signal on its own uniform grid, ``(hours, values)``, or ``None`` when absent."""
    step = 1.0 / float(RAW_SAMPLING_HZ)
    present = [s for s in recording.segments if key in s.raw and len(s.raw[key])]
    if not present:
        return None
    spans = [(float(s.epoch), float(s.epoch) + (len(s.raw[key]) - 1) * step) for s in present]
    origin, n = _grid(min(a for a, _ in spans), max(b for _, b in spans), step)
    grid = np.full(n, np.nan)
    for segment in present:
        values = np.asarray(segment.raw[key], dtype=np.float64).reshape(-1)
        _paint(grid, origin, step, float(segment.epoch) + np.arange(values.size) * step, values)
    return (origin + np.arange(n) * step) * _HOURS, grid


# =============================================================================
# Colour transforms
# =============================================================================
def _log_colour(z: np.ndarray, *, symmetric: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Transform a panel to log colour and return the transformed grid with its colour-bar ticks."""
    finite = z[np.isfinite(z)]
    decades = float(traces.LOG_PANEL_DECADES)
    if symmetric:
        top = float(np.abs(finite).max()) if finite.size and np.abs(finite).max() > 0 else 1.0
        floor = top * 10.0 ** (-decades)
        with np.errstate(divide="ignore", invalid="ignore"):
            magnitude = np.clip(np.log10(np.abs(z) / floor), 0.0, decades)
        out = np.sign(z) * magnitude
        ticks = list(range(-int(decades), int(decades) + 1))
        text = [f"{'-' if k < 0 else ''}{floor * 10.0 ** abs(k):.2g}" if k else f"±{floor:.1g}" for k in ticks]
        return out, dict(tickvals=ticks, ticktext=text, zmin=-decades, zmax=decades,
                         colorscale=DIVERGING_SCALE, reversescale=True, zmid=0.0)
    positive = finite[finite > 0.0]
    top = float(positive.max()) if positive.size else 1.0
    lo, hi = np.log10(top) - decades, np.log10(top)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(z > 0.0, np.clip(np.log10(z), lo, hi), np.nan)
    ticks = [k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1) if lo <= k <= hi]
    return out, dict(tickvals=ticks, ticktext=[f"1e{k}" for k in ticks], zmin=lo, zmax=hi,
                     colorscale=SEQUENTIAL_SCALE)


def _linear_colour(z: np.ndarray, *, symmetric: bool) -> Dict[str, Any]:
    finite = z[np.isfinite(z)]
    if symmetric:
        limit = float(np.abs(finite).max()) if finite.size else 1.0
        return dict(zmin=-limit, zmax=limit, zmid=0.0, colorscale=DIVERGING_SCALE, reversescale=True)
    return dict(zmin=0.0, zmax=float(finite.max()) if finite.size else 1.0, colorscale=SEQUENTIAL_SCALE)


# =============================================================================
# The page
# =============================================================================
_TEX = {r"\mu": "μ", r"\ell": "ℓ", r"\|": "‖", r"\lVert": "‖", r"\rVert": "‖", "$-$": "−", "$": ""}


def _html_math(text: str) -> str:
    """The panel titles' matplotlib mathtext as HTML: a few symbols, sub- and superscripts."""
    for tex, html in _TEX.items():
        text = text.replace(tex, html)
    text = re.sub(r"\^\{([^}]*)\}|\^(\w)", lambda m: f"<sup>{m.group(1) or m.group(2)}</sup>", text)
    return re.sub(r"_\{([^}]*)\}|_(\w)", lambda m: f"<sub>{m.group(1) or m.group(2)}</sub>", text)


def _note_empty(figure: go.Figure, row: int, hours: np.ndarray) -> None:
    """Mark a row that had nothing to draw, with a placeholder trace so its axes still render."""
    figure.add_trace(go.Scatter(x=[float(hours[0])], y=[np.nan], mode="lines", showlegend=False,
                                hoverinfo="skip"), row=row, col=1)
    figure.add_annotation(text=figures.EMPTY_NOTE, xref="x domain", yref="y domain", x=0.5, y=0.5,
                          showarrow=False, font=dict(color=figures.COLOR_GRAY), row=row, col=1)


def _hover(label: str, unit: str = "") -> str:
    return f"{label}: %{{y:.4g}} {unit}<br>t = %{{x:.3f}} h<extra></extra>"


def build_recording_dashboard(
    recording: traces.RecordingTrace,
    *,
    panels: Sequence[Any],
    lag_seconds: np.ndarray,
    caveat: Optional[str] = None,
) -> go.Figure:
    """Draw one recording's trace as an interactive figure: raw signals, then every panel.

    Args:
        recording: The assembled recording.
        panels: :class:`traces.HeatmapPanel` and :class:`traces.LinePanel` entries, top to bottom.
        lag_seconds: The compensated lag axis, for the lag-resolved heatmaps.
        caveat: A sentence printed under the figure, or ``None``.

    Returns:
        The figure; the caller writes it.
    """
    raw_rows = [(key, title, unit) for key, title, unit in RAW_PANELS
                if any(key in s.raw for s in recording.segments)]
    rows: List[Any] = [*raw_rows, *panels]
    heights = [HEATMAP_ROW_PX if isinstance(r, traces.HeatmapPanel) else LINE_ROW_PX
               if isinstance(r, traces.LinePanel) else RAW_ROW_PX for r in rows]
    titles = [_html_math(r.title) if isinstance(r, (traces.HeatmapPanel, traces.LinePanel)) else r[1] for r in rows]
    figure = make_subplots(
        rows=len(rows), cols=1, shared_xaxes=True, row_heights=heights,
        vertical_spacing=_ROW_GAP_PX / max(sum(heights), 1), subplot_titles=titles,
    )
    origin, n = _anchor_axis(recording)
    hours = (origin + np.arange(n) * float(SECONDS_PER_STEP)) * _HOURS
    extents = _kept_extents(recording)
    legends: Dict[str, Dict[str, Any]] = {}

    for row, spec in enumerate(rows, start=1):
        yaxis = f"yaxis{row if row > 1 else ''}"
        legend = f"legend{row if row > 1 else ''}"
        entries = 0
        if not isinstance(spec, (traces.HeatmapPanel, traces.LinePanel)):
            key, _title, unit = spec
            painted = _painted_raw(recording, key)
            if painted is not None:
                figure.add_trace(go.Scattergl(
                    x0=float(painted[0][0]), dx=_HOURS / float(RAW_SAMPLING_HZ),
                    y=painted[1].astype(np.float32), mode="lines", name=unit,
                    line=dict(color=figures.COLOR_BLUE, width=1), connectgaps=False,
                    hovertemplate=_hover(unit), showlegend=False,
                ), row=row, col=1)
            figure.layout[yaxis].title = unit
        elif isinstance(spec, traces.HeatmapPanel):
            z = _painted_vector(recording, spec, origin, n)
            figure.layout[yaxis].title = COEFFICIENT_LAG_AXIS_LABEL if spec.lag_axis else spec.ylabel
            if z is None:
                _note_empty(figure, row, hours)
            else:
                if spec.log:
                    z, colour = _log_colour(z, symmetric=spec.symmetric)
                else:
                    colour = _linear_colour(z, symmetric=spec.symmetric)
                y = np.asarray(lag_seconds, dtype=np.float64) if spec.lag_axis else np.arange(z.shape[1])
                domain = figure.layout[yaxis].domain
                colorbar = dict(
                    x=1.005, xanchor="left", y=float(np.mean(domain)), len=float(domain[1] - domain[0]),
                    thickness=10, outlinewidth=0, tickfont=dict(size=9),
                    tickvals=colour.pop("tickvals", None), ticktext=colour.pop("ticktext", None),
                )
                figure.add_trace(go.Heatmap(
                    x0=float(hours[0]), dx=_DX_HOURS, y=y, z=np.ascontiguousarray(z.T, dtype=np.float32),
                    colorbar=colorbar, hoverongaps=False,
                    hovertemplate=("lag %{y:g} s" if spec.lag_axis else "coordinate %{y}")
                                  + ("<br>log10 colour %{z:.2f}" if spec.log else "<br>value %{z:.4g}")
                                  + "<br>t = %{x:.3f} h<extra></extra>",
                    **colour,
                ), row=row, col=1)
                if spec.argmax_column is not None:
                    peak = _painted_scalar(recording, spec.argmax_column, origin, n, scored_only=False)
                    if peak is not None and np.isfinite(peak).any():
                        if spec.lag_axis:
                            index = np.clip(np.nan_to_num(peak, nan=0.0), 0, len(lag_seconds) - 1).astype(int)
                            peak = np.where(np.isfinite(peak), np.asarray(lag_seconds)[index], np.nan)
                        figure.add_trace(go.Scattergl(
                            x0=float(hours[0]), dx=_DX_HOURS, y=peak.astype(np.float32), mode="lines",
                            name=spec.argmax_label, legend=legend,
                            line=dict(color=figures.COLOR_VERMILLION, width=1), connectgaps=False,
                            hovertemplate=_hover(spec.argmax_label, "s" if spec.lag_axis else ""),
                        ), row=row, col=1)
                        entries += 1
        else:
            figure.layout[yaxis].title = spec.ylabel
            drawn = 0
            for index, column in enumerate(spec.columns):
                values = _painted_scalar(recording, column, origin, n)
                if values is None or not np.isfinite(values).any():
                    continue
                label = _html_math(spec.labels[index] if index < len(spec.labels) else column)
                figure.add_trace(go.Scattergl(
                    x0=float(hours[0]), dx=_DX_HOURS, y=values.astype(np.float32), mode="lines",
                    name=label, legend=legend,
                    line=dict(color=traces.LINE_COLOURS[index % len(traces.LINE_COLOURS)], width=1.2),
                    connectgaps=False, hovertemplate=_hover(label, spec.ylabel),
                ), row=row, col=1)
                drawn += 1
            if drawn and spec.segment_mean and spec.columns[0] in recording.summary.columns:
                xs: List[float] = []
                ys: List[float] = []
                for extent, (_, summary_row) in zip(extents, recording.summary.sort_values("segment_order").iterrows()):
                    value = float(summary_row[spec.columns[0]])
                    if extent is None or not np.isfinite(value):
                        continue
                    xs += [extent[0] * _HOURS, extent[1] * _HOURS, np.nan]
                    ys += [value, value, np.nan]
                if xs:
                    figure.add_trace(go.Scattergl(
                        x=xs, y=ys, mode="lines", name="segment mean", legend=legend,
                        line=dict(color=figures.COLOR_BLACK, width=2), connectgaps=False,
                        hovertemplate=_hover("segment mean", spec.ylabel),
                    ), row=row, col=1)
                    drawn += 1
            entries = drawn
            if drawn == 0:
                _note_empty(figure, row, hours)
        if entries >= 2:
            domain = figure.layout[yaxis].domain
            legends[legend] = dict(x=1.0, xanchor="right", y=float(domain[1]), yanchor="bottom",
                                   orientation="h", font=dict(size=9), bgcolor="rgba(255,255,255,0.6)")
        else:
            for trace in figure.data:
                if getattr(trace, "legend", None) == legend:
                    trace.showlegend = False

    # Segment shading, breaks and clocks, across every row.
    breaks = (np.asarray(recording.summary["is_break"], dtype=bool)
              if "is_break" in recording.summary.columns and len(recording.summary) == len(extents)
              else np.zeros(len(extents), dtype=bool))
    previous: Optional[Tuple[float, float]] = None
    for order, extent in enumerate(extents):
        if extent is None:
            continue
        if order % 2 == 1:
            figure.add_vrect(x0=extent[0] * _HOURS, x1=extent[1] * _HOURS, fillcolor=traces.SEGMENT_SHADE,
                             opacity=0.6, line_width=0, layer="below")
        if previous is not None and breaks[order]:
            figure.add_vrect(x0=previous[1] * _HOURS, x1=extent[0] * _HOURS, fillcolor=traces.BREAK_SHADE,
                             opacity=0.8, line_width=0, layer="below")
        previous = extent
    for column, position in traces._clock_positions(recording).items():
        label, colour, style = traces.CLOCK_STYLES[column]
        figure.add_vline(x=-position, line=dict(color=colour, width=1, dash="dash" if style == "--" else "dot"))
        figure.add_annotation(x=-position, y=1.0, xref="x", yref="paper", text=label, showarrow=False,
                              xanchor="left", yanchor="bottom", font=dict(size=9, color=colour))

    n_breaks = int(breaks.sum())
    span = (float(recording.summary[cohort.HOURS_COLUMN].max() - recording.summary[cohort.HOURS_COLUMN].min())
            if len(recording.summary) else float("nan"))
    total_px = sum(heights) + _ROW_GAP_PX * (len(rows) - 1) + _TITLE_PX + _FOOTER_PX
    figure.update_layout(
        template="plotly_white", font=_FONT, height=total_px, autosize=True,
        title=dict(
            text=(f"<b>Recording {recording.guid}</b> · subgroup {recording.subgroup} · class {recording.clinical_class}"
                  f"<br><span style='font-size:11px;color:#555'>{len(recording.segments)} segment(s) over "
                  f"{span:.1f} h, {n_breaks} break(s). Later segments overwrite earlier ones where they "
                  f"overlap. Drag to zoom, double-click to reset, scroll to zoom the time axis.</span>"),
            x=0.0, xanchor="left", font=dict(size=15),
        ),
        margin=dict(l=70, r=90, t=_TITLE_PX, b=_FOOTER_PX), hovermode="x", dragmode="zoom",
        plot_bgcolor="white", paper_bgcolor="white", **legends,
    )
    figure.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, showspikes=True, spikemode="across",
                        spikesnap="cursor", spikethickness=1, spikecolor=figures.COLOR_GRAY, spikedash="dot",
                        ticks="outside", showline=True, linecolor="#BBBBBB")
    figure.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, ticks="outside", showline=True,
                        linecolor="#BBBBBB", title_font=dict(size=10))
    figure.update_xaxes(title_text="Time relative to delivery (h)", row=len(rows), col=1,
                        rangeslider=dict(visible=False))
    for annotation in figure.layout.annotations[:len(rows)]:
        annotation.update(x=0.0, xanchor="left", font=dict(size=11))
    if caveat:
        plot_px = total_px - _TITLE_PX - _FOOTER_PX
        figure.add_annotation(text="<i>" + "<br>".join(textwrap.wrap(caveat, 190)) + "</i>", xref="paper",
                              yref="paper", x=0.0, y=-50.0 / plot_px, xanchor="left", yanchor="top",
                              showarrow=False, align="left", font=dict(size=9, color="#555555"))
    return figure


def write_recording_dashboard(figure: go.Figure, path: Any) -> Path:
    """Write the page, with plotly.js loaded from its CDN so thirty pages do not embed it thirty times."""
    path = Path(path).with_suffix(DASHBOARD_EXTENSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        path, include_plotlyjs="cdn", full_html=True, default_width="100%",
        config=dict(displaylogo=False, scrollZoom=True, responsive=True,
                    toImageButtonOptions=dict(format="svg", filename=path.stem)),
    )
    return path


__all__ = ["DASHBOARD_EXTENSION", "RAW_PANELS", "build_recording_dashboard", "write_recording_dashboard"]
