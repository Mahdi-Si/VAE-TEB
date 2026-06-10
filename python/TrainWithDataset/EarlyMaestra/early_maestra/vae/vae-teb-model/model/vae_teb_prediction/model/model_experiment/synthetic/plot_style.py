r"""Journal-quality matplotlib style: rcParams, palette, and layout helpers.

This module is the single source of truth for the look of every figure in the
synthetic-TE validation pipeline (``visualize.py``, ``evaluate_te.py``,
``lag_recovery.py``, ``beta_sweep.py``, ``directionality.py``,
``final_report.py``). Call :func:`apply_style` once before plotting to get a
consistent, publication-grade look.

The headline helper is :func:`stacked_figure`, which builds a two-column
``GridSpec`` layout: each panel is one *wide* axes in column 0 plus an optional
*ultra-narrow* colorbar axes in column 1. Because the colorbar lives in its own
dedicated column, it never steals horizontal width from the plot, so every
panel keeps the same width and a vertical stack aligns edge-to-edge.

Why this matters: the naive ``fig.colorbar(im, ax=ax)`` shrinks the axes it is
attached to. In a multi-panel figure that leaves heatmap rows narrower than
line rows, and the stack looks ragged. Routing the colorbar into a
pre-allocated narrow axes ($\approx 2\%$ of the figure width) fixes this; for a
*single* axes use :func:`add_colorbar` instead.

Project adaptations (vs. the upstream ``mpl-style`` skill module):
    * The ``Agg`` backend is selected at import so every consumer is headless
      (no display needed for PDF/PNG rendering).
    * :func:`save_figure` defaults to ``("pdf", "png")`` -- the package
      ``.gitignore`` tracks PDF/PNG figures and ignores bulkier formats.

Example:
    >>> from model.vae_teb_prediction.model.model_experiment.synthetic import (
    ...     plot_style as ps)
    >>> ps.apply_style()
    >>> fig, ax = plt.subplots(figsize=(6.0, 5.0))
    >>> ax.plot(x, y, color=ps.COLOR_BLUE)
    >>> ps.style_axes(ax)
    >>> ps.save_figure(fig, "results/diagnostic")  # -> .pdf, .png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # headless: no display needed for PDF/PNG rendering.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colorbar import Colorbar  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

__all__ = [
    "apply_style",
    "style_axes",
    "stacked_figure",
    "attach_colorbar",
    "add_colorbar",
    "save_figure",
    "tighten_xaxis",
    "auto_ylim",
    "COLOR_BLUE",
    "COLOR_ORANGE",
    "COLOR_GREEN",
    "COLOR_SKY",
    "COLOR_PURPLE",
    "COLOR_VERMILLION",
    "COLOR_GRAY",
    "COLOR_BLACK",
    "COLOR_LIGHT_GRAY",
    "COLOR_SAGE",
    "COLOR_TEAL_DARK",
    "PALETTE_PRIMARY",
    "PALETTE_EXTENDED",
    "FONT_LABEL",
    "FONT_TITLE",
    "FONT_TICK",
    "FONT_LEGEND",
    "SAVE_DPI",
]

# -----------------------------------------------------------------------------
# Palette
# -----------------------------------------------------------------------------
# A fixed, muted-but-distinct palette. Using named constants instead of ad-hoc
# hex strings keeps colour meaning consistent across every figure in a project.
COLOR_BLUE = "#3F72AF"
COLOR_ORANGE = "#FFB200"
COLOR_GREEN = "#609966"
COLOR_SKY = "#00ADB5"
COLOR_PURPLE = "#112D4E"
COLOR_VERMILLION = "#EB5B00"
COLOR_GRAY = "#393E46"
COLOR_BLACK = "#222831"
COLOR_LIGHT_GRAY = "#EEEEEE"
COLOR_SAGE = "#9DC08B"
COLOR_TEAL_DARK = "#0D7377"

# Ordered palettes for multi-series plots. Pick PALETTE_PRIMARY for <=4 series
# and PALETTE_EXTENDED when more distinct colours are needed.
PALETTE_PRIMARY = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_SKY]
PALETTE_EXTENDED = [
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_SKY,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    COLOR_SAGE,
    COLOR_TEAL_DARK,
]

# Shared semantic maps for the mixed-population figures so every panel (in
# ``mixed_eval`` and ``mixed_calibration``) uses one M->colour / band->marker
# mapping. ``M_COLOR_CYCLE`` is the fallback order for M values not in the map
# (e.g. the M-extrapolation caches at M=4 / M=64).
M_COLORS = {4: COLOR_SKY, 8: COLOR_BLUE, 16: COLOR_VERMILLION,
            32: COLOR_GREEN, 64: COLOR_PURPLE}
M_COLOR_CYCLE = PALETTE_EXTENDED
BAND_MARKERS = {"short": "o", "mid": "s", "long": "^"}


def color_for_M(m: int) -> str:
    """Return the canonical colour for an informative-channel count $M$.

    Args:
        m: The informative-channel count.

    Returns:
        The mapped colour, or a deterministic fallback from
        :data:`M_COLOR_CYCLE` for an unmapped $M$.
    """
    if int(m) in M_COLORS:
        return M_COLORS[int(m)]
    return M_COLOR_CYCLE[int(m) % len(M_COLOR_CYCLE)]


SAVE_DPI = 600

# Explicit per-element font sizes (1.5x the rcParams base sizes below). Pass
# these to ``set_xlabel(..., fontsize=FONT_LABEL)`` etc. when a specific panel
# needs a larger label than the rcParams default, so sizing stays uniform.
FONT_LABEL = 12.0   # axes.labelsize 8 * 1.5
FONT_TITLE = 13.5   # axes.titlesize 9 * 1.5
FONT_TICK = 10.5    # xtick.labelsize 7 * 1.5
FONT_LEGEND = 10.5  # legend.fontsize 7 * 1.5

# -----------------------------------------------------------------------------
# rcParams
# -----------------------------------------------------------------------------
# A single source of truth for the publication look: serif fonts, thin black
# spines, inward ticks, a barely-there grid, and high-DPI raster output.
_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": SAVE_DPI,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": COLOR_BLACK,
    "axes.labelcolor": COLOR_BLACK,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "xtick.color": COLOR_BLACK,
    "ytick.color": COLOR_BLACK,
    "grid.alpha": 0.2,
    "grid.linewidth": 0.3,
    "grid.color": COLOR_LIGHT_GRAY,
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.fancybox": False,
    "legend.edgecolor": COLOR_GRAY,
    "legend.shadow": False,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "errorbar.capsize": 3,
    # "regular" mathtext renders ``$...$`` symbols in the body font (serif)
    # rather than Computer Modern, so inline maths matches the labels.
    "mathtext.default": "regular",
}


def apply_style() -> None:
    """Apply the publication rcParams to the global matplotlib state.

    Call this once, near the top of a plotting routine (right after importing
    matplotlib), so every figure created afterwards inherits the house style.
    It first resets to matplotlib's ``default`` style to guarantee a clean
    starting point, then overlays :data:`_RCPARAMS`. Re-calling is harmless.

    It also raises matplotlib's mathtext logger to ``WARNING``: with
    ``mathtext.default="regular"`` every figure that uses a glyph absent from
    the serif body font (e.g. ``\\mathcal{L}``, ``\\star``) emits an
    informational "Substituting symbol" record per render. The substitution is
    correct -- only the log spam is suppressed.
    """
    plt.style.use("default")
    plt.rcParams.update(_RCPARAMS)
    logging.getLogger("matplotlib._mathtext").setLevel(logging.WARNING)


def style_axes(ax: Axes, *, grid: str = "major", minor_ticks: bool = True) -> None:
    """Apply consistent spine, grid, and tick styling to one axes.

    Draws the grid behind the data, keeps all four spines as thin black lines,
    and optionally enables minor ticks. Call this on every axes so panels look
    identical regardless of which plotting call produced them.

    Args:
        ax: The axes to style.
        grid: Grid density -- ``"major"`` (default), ``"both"`` for major plus
            minor gridlines, or ``"none"`` to disable the grid entirely.
        minor_ticks: Whether to enable minor ticks (only takes effect when
            ``grid="both"``).
    """
    ax.set_axisbelow(True)

    if grid in ("both", "major"):
        ax.grid(True, which="major", alpha=0.25, linewidth=0.3,
                color=COLOR_LIGHT_GRAY)
    if grid == "both":
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.2,
                color=COLOR_LIGHT_GRAY)
        if minor_ticks:
            ax.minorticks_on()

    for spine in ("left", "bottom", "top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(COLOR_BLACK)
            ax.spines[spine].set_linewidth(0.6)


def stacked_figure(
    panel_heights: Sequence[float],
    *,
    width: float = 12.0,
    colorbar: Union[bool, Sequence[bool]] = False,
    hspace: float = 0.7,
    wspace: float = 0.015,
    cbar_width: float = 0.02,
    margins: tuple[float, float, float, float] = (0.10, 0.95, 0.955, 0.045),
    height_scale: float = 1.7,
    height_pad: float = 2.0,
) -> tuple[Figure, list[Axes], list[Optional[Axes]]]:
    """Build a two-column stacked-panel figure with aligned colorbar gutters.

    The figure is a ``GridSpec`` of ``n`` rows and 2 columns. Column 0 is the
    wide plotting area; column 1 is a narrow gutter (``cbar_width`` of the plot
    width) reserved for colorbars. Panels that need a colorbar get a dedicated
    axes in column 1; panels that do not leave it empty. Because the colorbar
    never lives *inside* the main axes, every panel keeps an identical width
    and the stack aligns cleanly on both edges.

    The figure height scales with the panel content so adding rows does not
    squash the others: ``height = height_scale * sum(panel_heights) +
    height_pad`` (inches).

    Args:
        panel_heights: Relative height of each panel, top to bottom. These
            become ``GridSpec`` ``height_ratios`` and also drive the total
            figure height. Typical values: a raw-signal trace $\\approx 1.4$,
            an image/heatmap $\\approx 1.6$, a dense one-line-per-dim row
            $\\approx 0.45$, a summary panel $\\approx 1.25$.
        width: Figure width in inches (default 12.0).
        colorbar: Either a single ``bool`` applied to all panels, or a
            per-panel sequence of ``bool`` the same length as
            ``panel_heights``. ``True`` allocates a colorbar axes for that row.
        hspace: Vertical gap between panels, in fractions of the mean panel
            height. The default 0.7 leaves clear room for titles.
        wspace: Horizontal gap between the plot column and the colorbar
            gutter. Keep this tiny so the colorbar hugs its panel.
        cbar_width: Width of the colorbar column relative to the plot column
            (the second ``width_ratios`` entry). $\\approx 0.018$--$0.02$
            reads as a slim bar.
        margins: ``(left, right, top, bottom)`` figure margins as fractions of
            the figure, passed straight to ``GridSpec``.
        height_scale: Inches of figure height per unit of summed panel height.
        height_pad: Constant inches added to the height for titles/margins.

    Returns:
        A ``(fig, main_axes, cbar_axes)`` tuple. ``main_axes[i]`` is the wide
        column-0 axes for panel ``i``. ``cbar_axes[i]`` is the column-1
        colorbar axes when that panel requested one, otherwise ``None``.

    Raises:
        ValueError: If ``panel_heights`` is empty, or a per-panel ``colorbar``
            sequence has a different length than ``panel_heights``.
    """
    heights = list(panel_heights)
    n = len(heights)
    if n == 0:
        raise ValueError("panel_heights must contain at least one panel.")

    if isinstance(colorbar, bool):
        cbar_flags = [colorbar] * n
    else:
        cbar_flags = list(colorbar)
        if len(cbar_flags) != n:
            raise ValueError(
                f"colorbar sequence has length {len(cbar_flags)}, expected "
                f"{n} to match panel_heights."
            )

    fig_h = height_scale * sum(heights) + height_pad
    fig = plt.figure(figsize=(width, fig_h))
    left, right, top, bottom = margins
    gs = GridSpec(
        n, 2, figure=fig,
        hspace=hspace, wspace=wspace,
        height_ratios=heights,
        width_ratios=[1.0, cbar_width],
        left=left, right=right, top=top, bottom=bottom,
    )

    main_axes: list[Axes] = []
    cbar_axes: list[Optional[Axes]] = []
    for i in range(n):
        main_axes.append(fig.add_subplot(gs[i, 0]))
        cbar_axes.append(fig.add_subplot(gs[i, 1]) if cbar_flags[i] else None)
    return fig, main_axes, cbar_axes


def attach_colorbar(
    fig: Figure,
    mappable,
    cax: Axes,
    *,
    label: Optional[str] = None,
    labelsize: float = 6.0,
) -> Colorbar:
    """Route a mappable's colorbar into a pre-allocated narrow axes.

    Use this with the column-1 axes returned by :func:`stacked_figure`. Unlike
    :func:`add_colorbar`, it does not resize the main panel -- the colorbar
    fills the dedicated ``cax`` instead.

    Args:
        fig: The parent figure.
        mappable: The artist to draw a colorbar for (e.g. the handle returned
            by ``ax.imshow`` or ``ax.pcolormesh``).
        cax: The pre-allocated colorbar axes (a ``cbar_axes`` entry from
            :func:`stacked_figure`).
        label: Optional colorbar label.
        labelsize: Tick label size for the colorbar; 6 pt suits the slim
            gutter without crowding.

    Returns:
        The created ``Colorbar``.
    """
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    if label:
        cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"])
    return cbar


def add_colorbar(
    fig: Figure,
    mappable,
    ax: Axes,
    *,
    label: Optional[str] = None,
    size: str = "3.5%",
    pad: float = 0.05,
) -> Colorbar:
    """Append a slim colorbar to the right of a single axes.

    For standalone (non-stacked) figures where there is no dedicated colorbar
    column. The colorbar is sized as a fraction of the axes via
    ``make_axes_locatable``, which keeps it the exact height of the axes.
    Prefer :func:`attach_colorbar` inside :func:`stacked_figure` layouts so
    multi-panel widths stay equal.

    Args:
        fig: The parent figure.
        mappable: The artist to draw a colorbar for.
        ax: The axes the colorbar should sit beside.
        label: Optional colorbar label.
        size: Colorbar width as a percentage of the axes width.
        pad: Gap between the axes and the colorbar, in inches.

    Returns:
        The created ``Colorbar``.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label, fontsize=plt.rcParams["axes.labelsize"],
                       color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=plt.rcParams["xtick.labelsize"],
                        colors=COLOR_BLACK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    return cbar


def save_figure(
    fig: Figure,
    path: Union[str, Path],
    *,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = SAVE_DPI,
    close: bool = True,
) -> list[Path]:
    """Save a figure to multiple formats and (optionally) close it.

    Saves PDF + PNG by default: PDF is the vector format for journal
    submission, PNG is a high-resolution raster for quick viewing and slides.
    The package ``.gitignore`` keeps both tracked.

    Args:
        fig: The figure to save.
        path: Output path *without* an extension (e.g.
            ``"results/diagnostic"``). One file per entry in ``formats`` is
            written, sharing this basename. Parent directories are created.
        formats: Extensions to write. Defaults to ``("pdf", "png")``.
        dpi: Raster resolution; only affects PNG (vector formats ignore it).
        close: Whether to close the figure afterwards to free memory. Set
            ``False`` to keep inspecting it interactively.

    Returns:
        The list of paths actually written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        written.append(out)
    if close:
        plt.close(fig)
    return written


def tighten_xaxis(ax: Axes, x: np.ndarray) -> None:
    """Clamp the x-limits to the data range with zero padding.

    Removes matplotlib's default horizontal margin so traces start and end
    flush with the axes edges -- important when stacking panels that should
    share a common x axis.

    Args:
        ax: The axes to adjust.
        x: The x-values; non-finite entries are ignored.
    """
    if x is None or len(x) == 0:
        return
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return
    ax.set_xlim(float(np.min(finite)), float(np.max(finite)))
    ax.margins(x=0.0)


def auto_ylim(
    ax: Axes,
    y: np.ndarray,
    *,
    pad_frac: float = 0.05,
    min_zero: bool = False,
    clamp: Optional[tuple[float, float]] = None,
) -> None:
    """Set y-limits from data with a small symmetric padding.

    Args:
        ax: The axes to adjust.
        y: The y-values; non-finite entries are ignored.
        pad_frac: Fraction of the data span added above and below.
        min_zero: Force the lower limit to 0 (useful for magnitudes).
        clamp: Optional ``(low, high)`` hard bounds the final limits are
            clipped into.
    """
    if y is None:
        return
    finite = np.asarray(y)[np.isfinite(y)]
    if finite.size == 0:
        return
    y_min = 0.0 if min_zero else float(np.min(finite))
    y_max = float(np.max(finite))
    span = max(y_max - y_min, 1e-12)
    y_min -= span * pad_frac
    y_max += span * pad_frac
    if clamp is not None:
        y_min = max(clamp[0], y_min)
        y_max = min(clamp[1], y_max)
    ax.set_ylim(y_min, y_max)
