r"""Static matplotlib primitives for the evaluation figures.

Three generic panels -- a histogram, a median-plus-IQR ribbon, and a heatmap with a colourbar --
plus the repository's figure conventions, re-exported so an analysis imports its plotting
surface from one place.

**The style is a journal figure's, and it is decided here once.** :func:`configure_figure_style`
applies ``utils.style``'s publication base and then :data:`STYLE_REFINEMENT` over it, which is what
turns a training-diagnostic look into a print one: an open frame (left and bottom spines only, box
frames on heatmaps), outward ticks, a 7 pt serif type scale with matching ``stix`` mathtext,
unframed legends, a hairline grid, and a double-column default width of :data:`FIGURE_WIDTH`
inches (183 mm, the wide-column width of the major journals) so a figure is designed at the size
it will be read rather than scaled down to it. Every panel here reads its weights and sizes from
the *active* ``rcParams`` rather than from literals, which is what lets a package's own refinement
reach inside them.

**The palette is the evaluation packages' own.** :data:`COLOR_BLUE` and its siblings are the
Okabe-Ito colour-blind-safe set, which is the palette the major journals recommend and which
survives a greyscale print; the training callbacks keep the brighter hues of
:mod:`teb_vae.lag_attn.figure_primitives`, so an evaluation figure and a training figure of the
same quantity are reconciled by legend rather than by hue. The names are kept so every analysis
keeps naming a colour by its job.

**Importing this module does not restyle anything.** ``apply_publication_style`` mutates global
``rcParams``, so calling it at import time would silently restyle any other figure produced in
the same process -- including a test's. The pipeline calls :func:`configure_figure_style` once
at startup instead.

**Every panel tolerates empty and all-``NaN`` input.** An analysis that legitimately found
nothing -- a fully masked split, a metric that is undefined for this checkpoint -- must produce
an empty, labelled figure rather than take down a multi-hour run at its final step.

**Every multi-panel figure gets panel letters, and every footnote gets its own room.**
:func:`render_figure` stamps bold lowercase letters on the data panels of a figure that has more
than one, and lays the figure out around the footnote :func:`footnote` reserved space for, so a
caveat printed under a figure never lands on the axis label above it.
"""
from __future__ import annotations

import string
import textwrap
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.backend_bases import FigureCanvasBase  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    attach_lag_seconds_axis,
    safe_vabs,
    shade_warmup,
    to_numpy,
)
from utils.style import (  # noqa: E402
    CLASS_COLORS_DEFAULT,
    SAVE_DPI,
    apply_publication_style,
    get_class_colors,
    save_figure,
)

__all__ = [
    "CLASS_COLORS_DEFAULT",
    "COLOR_BLACK",
    "COLOR_BLUE",
    "COLOR_GRAY",
    "COLOR_GREEN",
    "COLOR_LIGHT_GRAY",
    "COLOR_ORANGE",
    "COLOR_PURPLE",
    "COLOR_VERMILLION",
    "DEFAULT_FIGURE_FORMAT",
    "FIGURE_WIDTH",
    "SAVE_DPI",
    "STYLE_REFINEMENT",
    "SUPPORTED_FIGURE_FORMATS",
    "active_figure_format",
    "attach_lag_seconds_axis",
    "binned_violin_panel",
    "configure_figure_style",
    "figure_filename",
    "footnote",
    "frequency_scatter",
    "get_class_colors",
    "group_colors",
    "grouped_violin_figure",
    "heatmap_with_colorbar",
    "histogram_panel",
    "label_panels",
    "label_rows",
    "layout_rect",
    "legend_with_headroom",
    "mark_laid_out",
    "multi_line_panel",
    "render_figure",
    "ribbon_plot",
    "safe_vabs",
    "save_figure",
    "set_figure_format",
    "shade_warmup",
    "significance_strip",
    "style_axes",
    "to_numpy",
    "violin_panel",
]

# =============================================================================
# The palette
# =============================================================================
#: The Okabe-Ito set, named by the job each hue does in the figures rather than by its number.
#: Chosen because it is distinguishable under the common colour-vision deficiencies and keeps a
#: luminance order in greyscale, which the training callbacks' saturated hues do not.
COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_PURPLE = "#CC79A7"
COLOR_VERMILLION = "#D55E00"
#: Text and reference marks: dark enough to read at 6 pt, lighter than the data ink.
COLOR_GRAY = "#4D4D4D"
COLOR_BLACK = "#000000"
#: Grid, frames behind data, and the reference diagonal: the lightest mark on the page.
COLOR_LIGHT_GRAY = "#D9D9D9"

# =============================================================================
# The style
# =============================================================================
#: Default figure width in inches: 183 mm, the double-column width of Nature, Science, IEEE and
#: the major ML venues' two-column pages. A figure drawn at this width with 7 pt type is read at
#: the size it was designed at rather than shrunk to fit.
FIGURE_WIDTH = 7.2

#: The refinement applied over ``utils.style.apply_publication_style``. That base is shared with
#: the training callbacks and tuned for figures read on a screen during a run; this is the delta
#: that makes an evaluation figure a print one. Applied as a delta rather than a replacement so
#: the serif family, the DPI and the white background stay the repository's.
STYLE_REFINEMENT = {
    # Type: a four-step scale, 5.5-7 pt, inside the 5-8 pt band the journals ask for at final size.
    # Half a point below the earlier scale throughout, so tick labels and legends stop touching on
    # the dense multi-panel pages while the axis labels stay the largest text on the page.
    # Mathtext in the STIX face so a symbol in a label matches the Times body around it.
    "font.size": 6.5,
    "axes.titlesize": 7.0,
    "axes.titleweight": "normal",
    "axes.titlepad": 4.0,
    "axes.labelsize": 6.5,
    "axes.labelpad": 2.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "legend.title_fontsize": 5.5,
    "mathtext.fontset": "stix",
    # Frame: left and bottom spines only, hairline weight, ticks outward. The box frame heatmaps
    # need is restored per axes by ``style_axes``.
    "axes.linewidth": 0.4,
    "axes.edgecolor": COLOR_BLACK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.25,
    "ytick.minor.width": 0.25,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    # Grid: a hairline one can read a value against, faint enough to vanish under a shape.
    "grid.color": COLOR_LIGHT_GRAY,
    "grid.linewidth": 0.25,
    "grid.alpha": 0.5,
    # Legend: unframed. A box around a legend is the single most common mark of an unconsidered
    # figure, and the entries are short enough to read against the data.
    "legend.frameon": False,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.6,
    "legend.handleheight": 0.7,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.0,
    # Data defaults for every artist drawn without an explicit weight.
    "lines.linewidth": 0.7,
    "lines.markersize": 2.5,
    "lines.markeredgewidth": 0.3,
    "patch.linewidth": 0.35,
    "hatch.linewidth": 0.35,
    # Tick formatting: mathtext exponents rather than ``1e-3`` in a monospace face, and an offset
    # only when the axis really needs one.
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 4),
    "axes.formatter.useoffset": False,
    # Output: a tight bounding box with a hair of margin.
    "savefig.pad_inches": 0.02,
}

#: Drawn on an axes that has no finite data, in place of an empty frame that reads as a bug.
EMPTY_NOTE = "no finite values"

#: Weight in points of the dark outline every histogram bar and bar-chart bar carries. Thin:
#: about a third of the data line weight, enough to separate adjacent bins at print size and
#: light enough that a fill reads as one mark.
HISTOGRAM_EDGE_WIDTH = 0.25

#: Panel letters, stamped by :func:`render_figure` on every figure with more than one data panel,
#: in the order the axes were created. Lowercase and bold -- the convention of Nature and of the
#: major ML venues' multi-panel figures.
PANEL_LETTERS = string.ascii_lowercase

#: The name matplotlib gives a colourbar's own axes, which is how :func:`label_panels` tells a
#: data panel from the colourbar hanging off it.
_COLORBAR_AXES_LABEL = "<colorbar>"

#: The attribute a figure carries once :func:`footnote` has reserved room under it, read back by
#: :func:`render_figure` so the layout leaves that room. Module-private state on the figure rather
#: than a threaded parameter, because the footnote is written by one caller and the layout is done
#: by another, and neither owns the other's signature.
_FOOTNOTE_ATTRIBUTE = "_eval_footnote_fraction"

#: The attribute a builder stamps on a figure it laid out itself (a gridspec with explicit
#: margins, a shared colour-axis column), read back by :func:`render_figure` so ``tight_layout``
#: does not undo that layout. Stamped through :func:`mark_laid_out`.
_LAYOUT_DONE_ATTRIBUTE = "_eval_layout_done"

#: Footnote type size and line spacing, in points. 6 pt is the smallest size the journals accept
#: and the caveats these figures carry are read, not decorative, so they are not set below it.
FOOTNOTE_SIZE = 6.0
_FOOTNOTE_LINE_HEIGHT = 1.35
#: Average advance of a serif glyph at 1 pt, used to wrap a footnote to the figure width.
_GLYPH_ADVANCE_EM = 0.47


#: The format a run writes when its config names none. PDF because every committed
#: ``figure_manifest.json`` and ``FIGURE_GUIDE.md`` in this repository records ``.pdf`` names, and
#: those are compared against a real run: changing this default would invalidate all of them at
#: once. A run that wants another format says so in ``eval_config.figure_format``.
DEFAULT_FIGURE_FORMAT = "pdf"

#: The formats a run may ask for, taken from the live matplotlib build rather than written out
#: here. A hardcoded list is a list that goes stale against the installed backend -- and the point
#: of validating the config key at all is to reject a typo against what this machine can actually
#: write, not against what was true when the constant was typed.
SUPPORTED_FIGURE_FORMATS = frozenset(FigureCanvasBase.get_supported_filetypes())

#: The active format, set once per run by :func:`configure_figure_style` and read by
#: :func:`render_figure`.
#:
#: **Module state rather than a threaded parameter**, and deliberately: the format is a property of
#: the *run*, not of any one figure, and threading it would mean a new argument on roughly a
#: hundred call sites and every ``build_*_figure`` between them -- for a value that is constant for
#: the whole pass. This is the same shape ``apply_publication_style`` already has, which mutates
#: global ``rcParams`` from the same one-call-at-run-start hook for the same reason.
_ACTIVE_FIGURE_FORMAT = DEFAULT_FIGURE_FORMAT


def active_figure_format() -> str:
    """Return the format :func:`render_figure` currently writes.

    Returns:
        A matplotlib filetype such as ``"pdf"`` or ``"svg"``, without the leading dot.
    """
    return _ACTIVE_FIGURE_FORMAT


def figure_filename(stem: str) -> str:
    """Return the on-disk filename ``stem`` gets under the active format.

    The counterpart to :func:`render_figure` for anything that needs to *name* a figure without
    drawing it -- an analysis recording what it wrote, a test asserting the file exists. Both
    would otherwise hardcode an extension that ``eval_config.figure_format`` can change.

    Args:
        stem: A figure-name constant, without an extension.

    Returns:
        ``"<stem>.<format>"``.
    """
    return f"{stem}.{_ACTIVE_FIGURE_FORMAT}"


def set_figure_format(figure_format: str) -> str:
    """Set the format every subsequent :func:`render_figure` writes.

    Args:
        figure_format: A matplotlib filetype, with or without a leading dot; case-insensitive.

    Returns:
        The normalised format that was set.

    Raises:
        ValueError: If the format is not one this matplotlib build can write. The message names
            the supported set, because the realistic error here is a typo in a config file.
    """
    global _ACTIVE_FIGURE_FORMAT
    normalised = str(figure_format).strip().lstrip(".").lower()
    if normalised not in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(
            f"unsupported figure format {figure_format!r}; this matplotlib build writes "
            f"{sorted(SUPPORTED_FIGURE_FORMATS)}."
        )
    _ACTIVE_FIGURE_FORMAT = normalised
    return normalised


def configure_figure_style(figure_format: Optional[str] = None) -> None:
    """Apply the repository's publication rcParams and fix the run's figure format.

    Called once at pipeline start. Deliberately not an import side effect -- see the module
    docstring.

    Args:
        figure_format: The format every figure of this run is written in. ``None`` leaves the
            active format alone, which is :data:`DEFAULT_FIGURE_FORMAT` in a process that has not
            set one.

    Raises:
        ValueError: If ``figure_format`` is not a format this matplotlib build can write.
    """
    apply_publication_style()
    plt.rcParams.update(STYLE_REFINEMENT)
    if figure_format is not None:
        set_figure_format(figure_format)


def style_axes(ax: Any, *, grid: str = "major") -> None:
    """Style one axes the way every evaluation panel is styled.

    An open frame -- left and bottom spines, outward ticks -- for a panel that plots values, and a
    full box for one that does not draw a grid, which in these packages is always an image: a
    heatmap or a per-recording trace, whose extent the frame has to close. Every weight and colour
    comes from the active ``rcParams``, so a package's refinement reaches here without a second
    copy of the numbers.

    Args:
        ax: Target axes.
        grid: ``"major"`` for a hairline major grid, ``"both"`` to add a dotted minor one, or
            ``"none"`` for no grid and a boxed frame.
    """
    ax.set_axisbelow(True)
    width = float(plt.rcParams["axes.linewidth"])
    boxed = grid == "none"
    for name, spine in ax.spines.items():
        spine.set_linewidth(width)
        spine.set_color(plt.rcParams["axes.edgecolor"])
        if name in ("top", "right"):
            spine.set_visible(boxed)
    if boxed:
        ax.grid(False)
        return
    ax.grid(
        True, which="major", linestyle="-",
        linewidth=plt.rcParams["grid.linewidth"], alpha=plt.rcParams["grid.alpha"],
        color=plt.rcParams["grid.color"],
    )
    if grid == "both":
        ax.minorticks_on()
        ax.grid(
            True, which="minor", linestyle=":",
            linewidth=plt.rcParams["grid.linewidth"], alpha=plt.rcParams["grid.alpha"] * 0.6,
            color=plt.rcParams["grid.color"],
        )


def footnote(fig: Any, text: str) -> float:
    """Print a wrapped footnote under a figure and reserve the room it needs.

    The one place a caveat is written under a figure, so every figure carries it at one size and
    in one position. Wrapped to the figure's own width rather than to a fixed character count, and
    the room reserved is computed from the lines that result, so a four-line qualification under a
    short figure pushes the axes up instead of landing on their x label. :func:`render_figure`
    reads the reservation back when it lays the figure out.

    Args:
        fig: The figure.
        text: The footnote, one paragraph.

    Returns:
        The fraction of the figure height reserved under the axes.
    """
    width_in, height_in = (float(v) for v in fig.get_size_inches())
    # The line count is estimated here from the average glyph advance, for the reservation; the
    # wrapping itself is left to matplotlib, so the artist carries the caveat verbatim and a
    # reader of the figure's texts finds the sentence rather than a line-broken copy of it.
    per_line = max(int(width_in * 72.0 / (FOOTNOTE_SIZE * _GLYPH_ADVANCE_EM)), 20)
    n_lines = len(textwrap.wrap(str(text), width=per_line)) or 1
    line_in = FOOTNOTE_SIZE * _FOOTNOTE_LINE_HEIGHT / 72.0
    # The block plus a gap of one line above it, as a fraction of the figure.
    fraction = min((n_lines + 1) * line_in / height_in, 0.4)
    fig.text(
        0.0, 0.0, str(text), ha="left", va="bottom", wrap=True,
        fontsize=FOOTNOTE_SIZE, color=COLOR_GRAY, linespacing=_FOOTNOTE_LINE_HEIGHT,
    )
    setattr(fig, _FOOTNOTE_ATTRIBUTE, fraction)
    return fraction


def legend_with_headroom(
    ax: Any, *, ncol: int = 1, headroom: float = 0.3, below: bool = False, **kwargs: Any
) -> Any:
    """Place the legend in a strip of empty space made beside the data, not on top of it.

    A curve that spans the whole x axis leaves no corner for ``loc="best"`` to find, so the
    legend lands on the data wherever it goes. The room is made instead: the y range is extended
    by ``headroom`` on one side and the legend is put there. Above by default; below for a panel
    whose top strip is already in use, on data that cannot go under its own minimum.

    Args:
        ax: The axes, with everything already drawn.
        ncol: Legend columns.
        headroom: Fraction of the current y range added on the chosen side.
        below: Extend downward and place the legend at the bottom instead of the top.
        **kwargs: Passed to ``ax.legend``.

    Returns:
        The legend.
    """
    lo, hi = ax.get_ylim()
    room = (hi - lo) * float(headroom)
    if below:
        ax.set_ylim(lo - room, hi)
        return ax.legend(loc="lower right", ncol=ncol, **kwargs)
    ax.set_ylim(lo, hi + room)
    return ax.legend(loc="upper right", ncol=ncol, **kwargs)


def _data_axes(fig: Any) -> list:
    """The axes of ``fig`` that hold data: no colourbars, no secondary axes, no insets."""
    panels = []
    for ax in fig.axes:
        if ax.get_label() == _COLORBAR_AXES_LABEL:
            continue
        # A secondary axis or an inset is a child of a data axes rather than a panel of its own.
        if getattr(ax, "_secondary_axes_owner", None) is not None or any(
            ax in getattr(other, "child_axes", ()) for other in fig.axes if other is not ax
        ):
            continue
        panels.append(ax)
    return panels


def _take_title(ax: Any) -> str:
    """Return a panel's title and clear it, wherever the builder put it.

    Builders set titles with a bare ``set_title``, which is the centred slot, and a reader of the
    in-memory figure -- a test, an assertion in an analysis -- finds them with a bare
    ``get_title``. The left-aligned journal placement is therefore applied only here, at render
    time, so that convention costs no builder and no reader anything.
    """
    for location in ("center", "left", "right"):
        title = ax.get_title(loc=location)
        if title:
            ax.set_title("", loc=location)
            return title
    return ""


def label_panels(fig: Any) -> int:
    """Left-align every data panel's title, prefixed with a bold letter on a multi-panel figure.

    Letters go in creation order, which for every grid this module builds is row-major. A figure
    with one data panel gets no letter: there is nothing to refer to. Called at render time, so
    the in-memory figure a builder returns still carries its plain, centred titles.

    The letter is written **into the title** rather than pinned to the axes corner, and the
    reason is the top axis: a panel carrying a secondary axis along its top has its title pushed
    up above that axis, and a letter anchored to the frame corner would then sit a line below its
    own title. Set in bold mathtext, which the ``stix`` font set renders in the serif face of the
    title beside it; a panel without a title carries the letter alone.

    Args:
        fig: The figure.

    Returns:
        How many letters were stamped.
    """
    panels = _data_axes(fig)
    lettered = len(panels) >= 2
    for index, ax in enumerate(panels):
        title = _take_title(ax)
        if lettered and index < len(PANEL_LETTERS):
            letter = f"$\\mathbf{{{PANEL_LETTERS[index]}}}$"
            title = f"{letter}  {title}" if title else letter
        if title:
            ax.set_title(title, loc="left")
    return min(len(panels), len(PANEL_LETTERS)) if lettered else 0


def _finite(values: Any) -> np.ndarray:
    """Return the finite entries of ``values`` as a flat float array.

    Args:
        values: Any array, tensor or sequence.

    Returns:
        A 1-D array of the finite entries, possibly empty.
    """
    array = np.asarray(to_numpy(values), dtype=np.float64).ravel()
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _note_empty(ax: Any) -> None:
    """Mark an axes that had no finite data, so an empty panel is legible rather than puzzling."""
    ax.text(
        0.5,
        0.5,
        EMPTY_NOTE,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=plt.rcParams["axes.labelsize"],
        fontstyle="italic",
        color=COLOR_GRAY,
    )


def histogram_panel(
    ax: Any,
    values: Any,
    *,
    title: str = "",
    xlabel: str = "",
    bins: int = 40,
    color: str = COLOR_BLUE,
    reference: Optional[float] = None,
    reference_label: str = "",
) -> int:
    """Draw a histogram with a median line, tolerating empty and all-``NaN`` input.

    Args:
        ax: Target axes.
        values: The sample, any shape. Non-finite entries are dropped.
        title: Panel title.
        xlabel: X-axis label.
        bins: Histogram bin count.
        color: Bar colour.
        reference: Optional vertical reference line -- a threshold, or a null value the
            distribution should sit away from.
        reference_label: Legend label for the reference line.

    Returns:
        The number of finite values drawn, so a caller can record how much of a capped draw
        actually contributed rather than inferring it from the figure.
    """
    finite = _finite(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if finite.size == 0:
        _note_empty(ax)
        style_axes(ax)
        return 0

    # Bars outlined in a thin dark hairline, so adjacent bars read as separate bins rather than
    # as one filled area; the weight is well below the data line weight so the outline never
    # reads as a second series.
    ax.hist(
        finite, bins=int(bins), color=color, alpha=0.85, edgecolor=COLOR_BLACK,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )
    median = float(np.median(finite))
    ax.axvline(median, color=COLOR_VERMILLION, linestyle="--", linewidth=plt.rcParams["lines.linewidth"],
               label=f"median {median:.4g}")
    if reference is not None and np.isfinite(reference):
        ax.axvline(float(reference), color=COLOR_GRAY, linestyle=":", linewidth=plt.rcParams["lines.linewidth"],
                   label=reference_label or f"reference {float(reference):.4g}")
    ax.legend(loc="best")
    style_axes(ax)
    return int(finite.size)


def ribbon_plot(
    ax: Any,
    x: Any,
    values: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = COLOR_BLUE,
    label: str = "",
) -> int:
    r"""Draw a median line with an inter-quartile ribbon over a stack of per-sample curves.

    Median and IQR rather than mean and standard deviation: these profiles are routinely
    right-skewed -- a handful of poorly forecast recordings sit far above the rest -- and a mean
    band would be pulled off the bulk of the distribution and could extend below zero on a
    quantity that cannot be negative.

    Args:
        ax: Target axes.
        x: The shared x coordinate, $(N,)$.
        values: Per-sample curves, $(B, N)$. Columns that are entirely non-finite are skipped.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Line and ribbon colour.
        label: Legend label for the median line.

    Returns:
        The number of x positions that had at least one finite value.
    """
    curves = np.asarray(to_numpy(values), dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[None, :]
    axis_x = np.asarray(to_numpy(x), dtype=np.float64).ravel()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if curves.size == 0 or axis_x.size == 0 or not np.isfinite(curves).any():
        _note_empty(ax)
        style_axes(ax)
        return 0

    # An all-NaN column is legitimate -- the warm-up anchors are masked out by construction --
    # and nanpercentile warns on it rather than raising, so the warning is suppressed and the
    # column is left as NaN, which matplotlib renders as a gap.
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            low = np.nanpercentile(curves, 25, axis=0)
            median = np.nanpercentile(curves, 50, axis=0)
            high = np.nanpercentile(curves, 75, axis=0)

    ax.fill_between(axis_x, low, high, color=color, alpha=0.2, linewidth=0, label="IQR")
    ax.plot(axis_x, median, color=color, linewidth=plt.rcParams["lines.linewidth"], label=label or "median")
    # The ribbon spans the whole axis, so the legend gets headroom rather than a corner.
    legend_with_headroom(ax, ncol=2, headroom=0.2)
    style_axes(ax)
    return int(np.isfinite(median).sum())


def heatmap_with_colorbar(
    fig: Any,
    ax: Any,
    data: Any,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cmap: Optional[str] = None,
    symmetric: bool = True,
    vlimits: Optional[Tuple[float, float]] = None,
    colorbar_label: str = "",
    separator_row: Optional[int] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    interpolation: str = "nearest",
    norm: Any = None,
) -> Any:
    """Draw a heatmap with its colourbar, tolerating empty and all-``NaN`` input.

    Args:
        fig: The figure, needed to attach the colourbar.
        ax: Target axes.
        data: The field, $(\\mathrm{rows}, \\mathrm{cols})$.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Colormap name. ``None`` follows ``symmetric``: diverging ``RdBu_r`` for a signed
            field, sequential ``viridis`` for a non-negative one -- both perceptually ordered and
            colour-blind safe, unlike ``bwr`` and ``jet``. Tying the choice to ``symmetric`` is
            what keeps them from disagreeing -- a non-negative field drawn on a diverging map
            renders its *smallest* values saturated and its mid-range white, so the best-forecast
            channel looks extreme and the mediocre one looks neutral. The colourbar stays
            correct throughout, so nothing in the numbers gives the inversion away; only the
            at-a-glance ranking, which is what a heatmap is for, is backwards.
        symmetric: Use a symmetric limit about zero, via
            :func:`~teb_vae.lag_attn.figure_primitives.safe_vabs`. Right for a signed field such
            as a residual; wrong for a non-negative one, which should pass ``False``.
        vlimits: Optional ``(vmin, vmax)``, overriding the limits this field alone would give and
            taking precedence over ``symmetric``. For a *family* of panels that must be read
            against one another: three per-cohort panels each scaled to its own extremes paint the
            same colour for three different values, and the comparison the panels exist for is
            then the one thing they cannot support -- while every colourbar stays correct, so
            nothing in the numbers gives it away. ``None`` derives the limits from the field,
            which is what a single panel wants.
        colorbar_label: Label for the colourbar.
        separator_row: Row index of the last row of the upper feature block. A horizontal rule is
            drawn at ``separator_row + 0.5``, which is where the two blocks actually meet.
        extent: Optional imshow extent, for a panel sharing a physical axis with another.
        norm: A matplotlib colour normaliser -- a ``LogNorm`` or ``SymLogNorm`` -- that replaces
            the limits above entirely, for a field whose values span orders of magnitude. ``None``
            keeps the linear scale the limits describe.
        interpolation: What ``imshow`` does between cells. ``'nearest'`` resamples to the
            renderer's pixel grid; ``'none'`` emits the cells themselves, which is what a
            *vector* output wants -- in a PDF the resampling is done at a resolution the file
            does not carry, so a cell boundary can land half a cell away from where the data
            says it is. Pass ``'none'`` wherever the reader is expected to index a cell.

    Returns:
        The image handle, or ``None`` when there was nothing to draw.
    """
    field = np.asarray(to_numpy(data), dtype=np.float64)
    # Resolved from ``symmetric`` so the colour scale and the value range can never disagree.
    colormap = cmap if cmap is not None else ("RdBu_r" if symmetric else "viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if field.size == 0 or not np.isfinite(field).any():
        _note_empty(ax)
        style_axes(ax, grid="none")
        return None

    if vlimits is not None:
        # The caller's own limits, so a family of panels shares one scale rather than each being
        # scaled to its own extremes. Checked for collapse the same way a derived range is: a
        # caller computing its limits over panels that all turned out constant would otherwise
        # hand over vmin == vmax.
        vmin, vmax = float(vlimits[0]), float(vlimits[1])
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    elif symmetric:
        limit = safe_vabs(field)
        vmin, vmax = -limit, limit
    else:
        finite = field[np.isfinite(field)]
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            # A constant field would otherwise get a degenerate colour range, which matplotlib
            # renders as a single flat colour indistinguishable from an empty panel.
            vmin, vmax = vmin - 0.5, vmax + 0.5

    scale = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    image = ax.imshow(
        field, aspect="auto", origin="upper", cmap=colormap, interpolation=interpolation, extent=extent, **scale,
    )
    if separator_row is not None:
        ax.axhline(float(separator_row) + 0.5, color=COLOR_BLACK, linewidth=plt.rcParams["axes.linewidth"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.015, aspect=30)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=plt.rcParams["axes.labelsize"])
    colorbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], width=plt.rcParams["ytick.major.width"])
    colorbar.outline.set_linewidth(plt.rcParams["axes.linewidth"])
    style_axes(ax, grid="none")
    return image


#: The inner box plot's two marks, weighted in **points** as multiples of the active
#: ``lines.linewidth`` -- the inter-quartile bar first, then the whisker.
#:
#: Points rather than data units, which is the one geometric decision in this mark. A box drawn a
#: fixed fraction of a violin's width wide occupies a fixed share of the *data* axis, and the axis
#: holds one unit per group however many groups there are -- so the same box renders as a slab on
#: a three-cohort panel and as a sliver on an eight-subgroup one. Weighted in points it is the
#: same mark on both, which is what lets the two figures be read against each other.
#:
#: The multiples are chosen so that under the refined ``lines.linewidth`` of $0.7$ they come out
#: at the cfs seam's ``LINE_HEAVY`` ($2.0$) and ``LINE_THIN`` ($0.5$) -- the weights its
#: ``distributions`` strip already draws an inter-quartile span and a range with. One visual
#: vocabulary for "these are quartiles", across two figures built by different modules.
INNER_BOX_BAR_WEIGHT_RATIO = 2.86
INNER_BOX_WHISKER_WEIGHT_RATIO = 0.71

#: The median dot's diameter in points, and the weight of its outline as a multiple of the active
#: ``lines.linewidth``. Sized to sit inside the bar it marks rather than to straddle it.
INNER_BOX_MEDIAN_MARKERSIZE = 3.0
INNER_BOX_MEDIAN_EDGE_RATIO = 0.55

#: The whisker rule: the most extreme observation within $1.5 \times$ the inter-quartile range of
#: the quartiles -- Tukey's convention, and matplotlib's own default. Named rather than left
#: implicit because the alternative convention would be redundant here: matplotlib evaluates a
#: violin's kernel density between the data's own minimum and maximum, so the body already *is*
#: the full range and a whisker drawn to it would restate the outline it sits inside.
INNER_BOX_WHISKER_IQR = 1.5


def _draw_inner_box(ax: Any, centre: float, values: np.ndarray) -> None:
    """Draw one violin's interior box: the whisker, the quartile bar and the median dot.

    Extracted rather than inlined because **two** panels draw this mark -- the categorical
    :func:`violin_panel` and the binned :func:`binned_violin_panel` -- and the repository is
    better served by one visual vocabulary for "these are quartiles" than by two implementations
    that happen to agree today. The statistics come from the same function ``Axes.boxplot``
    computes its own from, so the whisker rule here and on any box plot elsewhere cannot drift
    apart; only the drawing is ours, and only because ``Axes.boxplot`` sizes its box in data
    units. See the weight constants above for why the marks are weighted in points.

    Args:
        ax: Target axes.
        centre: Where on the category or value axis the mark sits.
        values: That cell's values, already reduced to the finite ones -- ``boxplot_stats``
            propagates a ``NaN`` into every quantile it computes.
    """
    weight = float(plt.rcParams["lines.linewidth"])
    box = cbook.boxplot_stats(
        [np.asarray(values, dtype=np.float64)], whis=INNER_BOX_WHISKER_IQR
    )[0]
    ax.vlines(
        centre, box["whislo"], box["whishi"], color=COLOR_BLACK,
        linewidth=weight * INNER_BOX_WHISKER_WEIGHT_RATIO, zorder=3,
    )
    ax.vlines(
        centre, box["q1"], box["q3"], color=COLOR_BLACK,
        linewidth=weight * INNER_BOX_BAR_WEIGHT_RATIO, zorder=3,
    )
    ax.plot(
        [centre], [box["med"]], marker="o", markersize=INNER_BOX_MEDIAN_MARKERSIZE,
        markerfacecolor="white", markeredgecolor=COLOR_BLACK,
        markeredgewidth=weight * INNER_BOX_MEDIAN_EDGE_RATIO, zorder=4,
    )


def violin_panel(
    ax: Any,
    samples: Any,
    *,
    title: str = "",
    ylabel: str = "",
    color: str = COLOR_BLUE,
    colors: Optional[Any] = None,
    reference: Optional[float] = None,
    reference_label: str = "",
) -> int:
    r"""Draw one violin per group, each with a thin box plot inside it.

    A violin rather than a box: these distributions are routinely bimodal -- a set of recordings
    the model forecasts well and a tail it does not -- and a box plot renders both as the same
    five numbers.

    **The interior is a box plot rather than a bare median line**, which is the usual pairing and
    the one these panels need. A median tick answers "where is the centre" and nothing else, so
    the two questions asked of every violin here -- how wide is the middle half, and how far past
    it does the cohort reach -- had to be estimated by eye off the body's outline, which is a
    kernel density and therefore smoothed past exactly the quartiles being read. The box states
    all three: the heavy bar is $Q_1$ to $Q_3$, the hairline through it runs to Tukey's adjacent
    values, and the dot is the median.

    Three choices in it are deliberate. The median dot is **white** because the bar it sits on is
    drawn over cohort hues that include a red, against which this package's accent colour has too
    little contrast to read; white on black reads on every cohort and needs no per-cohort rule.
    **No fliers** are drawn: matplotlib evaluates the kernel density between the data's own
    extremes, so every outlier is already on the page as the body's tail, and a second marker for
    it would read as a second observation. And **no caps**, which at this weight are
    indistinguishable from the whisker they terminate.

    A group with no finite value is kept in position with an empty slot rather than dropped, so
    the categories stay aligned with their labels; dropping it would silently shift every label
    after it onto the wrong violin.

    Args:
        ax: Target axes.
        samples: Ordered mapping from group label to that group's values.
        title: Panel title.
        ylabel: Y-axis label.
        color: Body colour, used when ``colors`` is not supplied.
        colors: Optional label-to-colour mapping, for a grouped variant that must match a
            palette used elsewhere in the run.
        reference: Optional horizontal reference line.
        reference_label: Legend label for the reference line.

    Returns:
        The number of groups that had at least one finite value.
    """
    groups = dict(samples)
    labels = list(groups)
    finite = [_finite(values) for values in groups.values()]
    populated = [index for index, values in enumerate(finite) if values.size > 0]

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)) + 1.0)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=plt.rcParams["xtick.labelsize"])

    if not populated:
        _note_empty(ax)
        style_axes(ax)
        return 0

    drawn = [finite[index] for index in populated]
    centres = [index + 1.0 for index in populated]
    parts = ax.violinplot(
        drawn,
        positions=centres,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )
    for position, body in zip(populated, parts["bodies"]):
        hue = (colors or {}).get(labels[position], color)
        body.set_facecolor(hue)
        body.set_alpha(0.55)
        # Outlined in its own hue rather than black, so the body reads as one mark.
        body.set_edgecolor(hue)
        body.set_linewidth(0.4)

    # The interior, one mark per populated group, drawn by the helper the binned panel shares --
    # a convention with an edge case worth not owning twice.
    for centre, values in zip(centres, drawn):
        _draw_inner_box(ax, centre, values)

    if reference is not None and np.isfinite(reference):
        ax.axhline(float(reference), color=COLOR_GRAY, linestyle=":", linewidth=plt.rcParams["lines.linewidth"],
                   label=reference_label or f"reference {float(reference):.4g}")
        ax.legend(loc="best")
    style_axes(ax)
    return len(populated)


#: How much of its slot a violin body occupies, leaving a gap between adjacent groups. The same
#: fraction :func:`violin_panel` uses for its categorical bodies, so the two figures read alike.
BINNED_BODY_FRACTION = 0.8


def binned_violin_panel(
    ax: Any,
    samples_by_window: Sequence[Any],
    centres: Sequence[float],
    *,
    groups: Sequence[str],
    bin_width: float,
    min_body_size: int,
    colors: Optional[Any] = None,
    color: str = COLOR_BLUE,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> int:
    r"""Draw one violin per (window, group) cell on a **numeric** axis, dodged inside each window.

    :func:`violin_panel` puts one violin per group on a category axis; this puts $k$ of them
    inside every window of a continuous coordinate, which is what turns a trajectory of medians
    into a trajectory of *distributions*. The median line a trajectory figure draws is one number
    per cell, and three cohorts with the same median can be a uniform shift, a heavier tail, or a
    handful of recordings the model fails on completely -- three different findings that only the
    body shows.

    **The dodge is computed from the group count of the whole figure, not of the window.** Group
    $i$ of $k$ sits at

    $$x = c + \left(i - \frac{k - 1}{2}\right)\frac{w}{k + 1}$$

    with a body $0.8\,w/(k+1)$ wide. Taking $k$ from the figure is what makes a cohort absent from
    one window leave a **gap** there rather than shift its neighbours into its place -- the same
    rule :func:`violin_panel` follows when it keeps an empty group's slot, and for the same
    reason: a mark that moves between windows cannot be compared across them.

    **A cell too thin for a density draws its values instead.** Below ``min_body_size`` -- and
    whenever a cell's values are all equal, which is a singular covariance the kernel estimator
    cannot invert -- the points are plotted directly. matplotlib evaluates a violin's kernel
    density between the data's own extremes, so a "distribution" over two values is a shape the
    smoother invented; and the caller's test excludes exactly those cells, so the figure and the
    test agree about which cells carry evidence.

    **Every cell is annotated with its count.** That is what a violin hides at this size: a cell's
    body can move because the population changed rather than because the quantity did, and the
    only thing that says which is the number behind it.

    Args:
        ax: Target axes.
        samples_by_window: One ``{group: values}`` mapping per window, **positionally aligned**
            with ``centres`` -- the same per-window shape
            :func:`~teb_vae.lag_attn.eval.stats.windowed_group_comparisons` takes, so a caller
            builds it once and tests and draws the same cells.
        centres: The window centres, in the order they should be drawn.
        groups: The group labels in the order they should be dodged, left to right. A group with
            no data anywhere still occupies its slot.
        bin_width: Window width in the x coordinate's own units, which sets the dodge and the
            body width.
        min_body_size: Fewest values a cell may have and still be drawn as a density. **Required**
            rather than defaulted, because it has to be the same number the caller excludes a cell
            from its test at, and a default here would be a second definition of that threshold.
        colors: Optional label-to-colour mapping, so a caller's own cohort palette reaches the
            bodies rather than this module's default.
        color: Body colour for a label the mapping does not cover.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The number of cells that had at least one finite value. Zero draws :data:`EMPTY_NOTE`
        instead of an empty frame.

    Raises:
        ValueError: If ``samples_by_window`` and ``centres`` are of different lengths. Zipping the
            shorter of the two would silently drop windows off the figure while every axis label
            still said they were there.
    """
    windows = list(samples_by_window)
    positions_x = [float(value) for value in centres]
    if len(windows) != len(positions_x):
        raise ValueError(
            f"binned_violin_panel got {len(windows)} window(s) of samples and "
            f"{len(positions_x)} centre(s); they are positionally aligned, so a mismatch would "
            f"draw one window's values at another window's coordinate."
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    order = [str(group) for group in groups]
    if not order or not windows:
        _note_empty(ax)
        style_axes(ax)
        return 0

    slot = float(bin_width) / float(len(order) + 1)
    palette = dict(colors or {})
    # Passed through rather than scaled: outside a styled run this rcParam is the string
    # ``'medium'``, which matplotlib resolves for itself and arithmetic cannot.
    label_size = plt.rcParams["xtick.labelsize"]
    populated = 0
    legend: list = []

    for index, group in enumerate(order):
        offset = (float(index) - (len(order) - 1) / 2.0) * slot
        colour = palette.get(group, color)
        bodies: list = []
        body_positions: list = []
        point_x: list = []
        point_y: list = []
        for centre, samples in zip(positions_x, windows):
            values = _finite((samples or {}).get(group, []))
            if values.size == 0:
                continue
            populated += 1
            x = centre + offset
            # A constant cell is not merely thin: ``gaussian_kde`` inverts a covariance that is
            # singular there and raises, which would take down the figure at its final step.
            if values.size >= int(min_body_size) and float(np.ptp(values)) > 0.0:
                bodies.append(values)
                body_positions.append(x)
            else:
                point_x.extend([x] * int(values.size))
                point_y.extend(values.tolist())
            ax.annotate(
                str(int(values.size)), (x, float(values.max())),
                textcoords="offset points", xytext=(0, 2), ha="center",
                fontsize=label_size, color=colour,
            )

        if bodies:
            parts = ax.violinplot(
                bodies,
                positions=body_positions,
                widths=BINNED_BODY_FRACTION * slot,
                showmedians=False,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(colour)
                body.set_alpha(0.55)
                body.set_edgecolor(colour)
                body.set_linewidth(0.4)
            for position, values in zip(body_positions, bodies):
                _draw_inner_box(ax, position, values)
        if point_x:
            ax.plot(
                point_x, point_y, marker="o", linestyle="none",
                markersize=plt.rcParams["lines.markersize"], color=colour,
                markeredgewidth=0.0, alpha=0.85,
            )
        legend.append((group, colour))

    if populated == 0:
        _note_empty(ax)
        style_axes(ax)
        return 0

    # Proxy handles rather than the bodies themselves: a violin collection carries no usable
    # legend entry, and a group drawn only as points would otherwise be missing from the key.
    for group, colour in legend:
        ax.plot([], [], marker="s", linestyle="none", color=colour, label=group)
    ax.legend(loc="best")
    # Half a window of padding, so the outermost bodies are not clipped by the data limits.
    ax.set_xlim(min(positions_x) - float(bin_width), max(positions_x) + float(bin_width))
    style_axes(ax)
    return populated


#: Smallest $p$ the strip below will take a logarithm of. A rank test returns an exact zero at
#: large $n$, and $-\log_{10} 0$ is an infinite bar that rescales the axis until every other
#: window is a flat line -- so the one window with the strongest evidence would erase the evidence
#: everywhere else.
P_VALUE_FLOOR = 1e-300

#: How much of a window the significance bar occupies. Narrower than a violin body on purpose: the
#: strip is read against its own threshold line, not compared bar to bar.
SIGNIFICANCE_BAR_FRACTION = 0.4


def significance_strip(
    ax: Any,
    centres: Sequence[float],
    p_holm: Sequence[float],
    *,
    alpha: float,
    bin_width: float,
    title: str = "",
    xlabel: str = "",
) -> int:
    r"""Draw the corrected significance of each window as $-\log_{10} p$ against its threshold.

    $$-\log_{10}\tilde{p} \quad\text{against}\quad -\log_{10}\alpha$$

    A bar above the line is a window that survived the correction. This is the repository's
    established mark for "is there anything there" -- the cross-subgroup heatmap's upper panel and
    the ancestor's trajectory significance figure both draw it -- and it is preferred to asterisk
    codes because it shows *how far* past the threshold a window is, which a reader otherwise has
    to fetch from the CSV.

    **The adjusted $p$ is the one to pass.** The raw per-window $p$ drawn against $\alpha$ would
    report a family of one for every window of a family of twenty-five.

    Args:
        ax: Target axes.
        centres: The window centres, in drawing order.
        p_holm: The Holm-adjusted $p$ per window, positionally aligned with ``centres``.
            A non-finite entry is a window that could not be tested: it gets **no bar** -- a zero
            height would read as a window with no evidence -- and a grey cross at zero instead,
            so that "found nothing" and "never looked at" are distinguishable on the page and not
            only in the table.
        alpha: The family-wise error rate the correction controls; drawn as the threshold line.
        bin_width: Window width in the x coordinate's own units, which sets the bar width.
        title: Panel title.
        xlabel: X-axis label.

    Returns:
        The number of bars drawn -- the count of windows that were testable at all. Zero draws
        :data:`EMPTY_NOTE`.

    Raises:
        ValueError: If ``centres`` and ``p_holm`` are of different lengths, which would draw one
            window's evidence at another window's coordinate.
    """
    positions = [float(value) for value in centres]
    values = [float(value) for value in p_holm]
    if len(positions) != len(values):
        raise ValueError(
            f"significance_strip got {len(positions)} centre(s) and {len(values)} p-value(s); "
            f"they are positionally aligned, so a mismatch would draw one window's evidence at "
            f"another window's coordinate."
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$-\\log_{10}$ Holm-adjusted $p$")

    testable = [
        (position, value) for position, value in zip(positions, values) if np.isfinite(value)
    ]
    # The windows that could not be tested, marked rather than left blank. A window whose bar is
    # absent because its p is 1 and a window that was never tested are different statements, and
    # at any realistic window count they are otherwise the same empty stretch of axis: on a
    # twenty-two-window profile with eight surviving bars, a reader cannot tell whether the other
    # fourteen found nothing or were never looked at.
    untestable = [
        position for position, value in zip(positions, values) if not np.isfinite(value)
    ]
    if untestable:
        ax.plot(
            untestable, np.zeros(len(untestable)), marker="x", linestyle="none",
            markersize=plt.rcParams["lines.markersize"], color=COLOR_GRAY,
            markeredgewidth=plt.rcParams["lines.linewidth"] * 0.6,
            label="not testable", zorder=3,
        )
    if not testable:
        if untestable:
            # Above the row of crosses rather than through it, so the note and the markers that
            # explain it do not overprint.
            ax.text(
                0.5, 0.75, EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
                fontsize=plt.rcParams["axes.labelsize"], fontstyle="italic", color=COLOR_GRAY,
            )
            ax.legend(loc="best")
        else:
            _note_empty(ax)
        style_axes(ax)
        return 0

    heights = -np.log10(
        np.clip(np.array([value for _, value in testable], dtype=np.float64), P_VALUE_FLOOR, 1.0)
    )
    ax.bar(
        [position for position, _ in testable],
        heights,
        width=SIGNIFICANCE_BAR_FRACTION * float(bin_width),
        color=COLOR_BLUE,
        alpha=0.85,
        edgecolor=COLOR_BLACK,
        linewidth=HISTOGRAM_EDGE_WIDTH,
    )
    ax.axhline(
        -np.log10(float(alpha)),
        color=COLOR_VERMILLION,
        linestyle="--",
        linewidth=plt.rcParams["lines.linewidth"],
        label=f"alpha = {float(alpha):g} (Holm-adjusted)",
    )
    ax.legend(loc="best")
    ax.set_xlim(min(positions) - float(bin_width), max(positions) + float(bin_width))
    style_axes(ax)
    return len(testable)


def group_colors(groups: Sequence[str]) -> dict:
    """Return a stable colour per group label.

    Delegates to ``utils.style.get_class_colors``, which already ships the healthy / acidosis /
    hie mapping and falls back to a palette for anything else -- which is what makes one function
    serve both the class axis, whose labels it knows, and the subgroup axis, whose it does not.
    Delegating rather than restating is also what keeps an eval figure the same colour as the
    training figure of the same cohort.

    Args:
        groups: The group labels appearing in a figure.

    Returns:
        Label to hex colour.
    """
    return get_class_colors([str(group) for group in groups])


def grouped_violin_figure(
    values_by_metric: Any,
    groups: Sequence[str],
    *,
    title_prefix: str = "",
    references: Optional[Any] = None,
    colors: Optional[Any] = None,
) -> Tuple[Any, Any]:
    r"""Build a figure with one violin panel per metric, each split across the same groups.

    One row per metric rather than one figure per metric: the question a grouped variant answers
    is whether the *cohorts* differ, and that is read by scanning several metrics for the same
    group rather than by comparing two files.

    Args:
        values_by_metric: Ordered mapping from metric name to ``{group: values}``.
        groups: The group labels, in the order the violins should appear.
        title_prefix: Prepended to each panel title, typically the grouping axis.
        references: Optional metric-to-reference-value mapping for a horizontal line.
        colors: Optional label-to-colour mapping. ``None`` uses :func:`group_colors`; a caller
            supplies one when its package draws cohorts from a palette of its own, so that its
            grouped variants match the rest of its figures rather than this module's default.

    Returns:
        ``(fig, axes)``, unsaved and unclosed -- the caller owns both, so it can assert on the
        in-memory figure before it is written.
    """
    metrics = dict(values_by_metric)
    colors = dict(colors) if colors is not None else group_colors(groups)
    reference_table = dict(references or {})

    fig, axes = new_figure(max(len(metrics), 1), height_per_row=3.0)
    for row, (metric, samples) in enumerate(metrics.items()):
        reference = reference_table.get(metric)
        violin_panel(
            axes[row, 0],
            {group: samples.get(group, []) for group in groups},
            title=f"{title_prefix}{metric}" if title_prefix else str(metric),
            ylabel=str(metric),
            colors=colors,
            reference=reference,
            reference_label="" if reference is None else f"{metric} = {float(reference):g}",
        )
    return fig, axes


def multi_line_panel(
    ax: Any,
    x: Any,
    curves: Any,
    labels: Sequence[str],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> int:
    """Overlay one line per row of ``curves``, with a legend naming each.

    Used where a ribbon would not do: comparing several *groups* against each other on one axes,
    rather than showing the spread within a single group.

    Args:
        ax: Target axes.
        x: The shared x coordinate, $(N,)$.
        curves: One row per group, $(G, N)$. Rows that are entirely non-finite are skipped.
        labels: One label per row.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The number of rows actually drawn.
    """
    field = np.asarray(to_numpy(curves), dtype=np.float64)
    if field.ndim == 1:
        field = field[None, :]
    axis_x = np.asarray(to_numpy(x), dtype=np.float64).ravel()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if field.size == 0 or axis_x.size == 0 or not np.isfinite(field).any():
        _note_empty(ax)
        style_axes(ax)
        return 0

    drawn = 0
    palette = plt.get_cmap("viridis")
    for row in range(int(field.shape[0])):
        if not np.isfinite(field[row]).any():
            continue
        colour = palette(row / max(int(field.shape[0]) - 1, 1))
        label = str(labels[row]) if row < len(labels) else f"group {row}"
        ax.plot(axis_x, field[row], color=colour, linewidth=plt.rcParams["lines.linewidth"], label=label)
        drawn += 1
    # Every curve spans the whole axis, so the legend gets headroom rather than a corner; the
    # column count grows with the group count so the strip stays one or two rows deep.
    legend_with_headroom(ax, ncol=min(max(drawn, 1), 4), headroom=0.25)
    style_axes(ax)
    return drawn


def label_rows(ax: Any, labels: Sequence[str]) -> None:
    """Name a heatmap's rows, so a band heatmap does not read as anonymous row indices.

    Args:
        ax: Target axes, whose y-axis runs over the rows of an already-drawn heatmap.
        labels: One label per row, in row order.
    """
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([str(label) for label in labels], fontsize=plt.rcParams["ytick.labelsize"])


def frequency_scatter(
    fig: Any,
    ax: Any,
    frequencies: Any,
    values: Any,
    *,
    colour_by: Optional[Any] = None,
    colour_label: str = "",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> Any:
    r"""Scatter a per-channel quantity against channel centre frequency, on a log frequency axis.

    Log-scaled in $x$ because the channels are a geometric filter bank: at the production
    geometry they span $5 \times 10^{-4}$ to $1.5$ Hz, so on a linear axis the entire slow half of
    the bank -- everything below the deceleration band -- collapses onto the left-hand tick.

    Channels with no centre frequency are omitted rather than drawn at $0$, which a log axis
    cannot represent and which would in any case assert a frequency the provenance does not
    determine. The count of omitted points is returned via the axes' legend so their absence is
    visible.

    Args:
        fig: The figure, needed to attach the colourbar.
        ax: Target axes.
        frequencies: Per-channel centre frequency in Hz, $(C,)$. Non-finite entries are dropped.
        values: The quantity to plot, $(C,)$.
        colour_by: Optional third quantity to colour the points by, $(C,)$.
        colour_label: Label for the colourbar.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        The scatter handle, or ``None`` when there was nothing to draw.
    """
    hz = np.asarray(to_numpy(frequencies), dtype=np.float64).ravel()
    quantity = np.asarray(to_numpy(values), dtype=np.float64).ravel()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    usable = np.isfinite(hz) & (hz > 0.0) & np.isfinite(quantity)
    n_dropped = int((~usable).sum())
    if not usable.any():
        _note_empty(ax)
        style_axes(ax)
        return None

    shades = None
    if colour_by is not None:
        shades = np.asarray(to_numpy(colour_by), dtype=np.float64).ravel()[usable]
        if not np.isfinite(shades).any():
            shades = None

    handle = ax.scatter(
        hz[usable], quantity[usable],
        c=shades if shades is not None else COLOR_BLUE,
        cmap="viridis" if shades is not None else None,
        s=14, alpha=0.85, edgecolor=COLOR_BLACK, linewidth=HISTOGRAM_EDGE_WIDTH,
    )
    ax.set_xscale("log")
    if shades is not None:
        colorbar = fig.colorbar(handle, ax=ax, fraction=0.03, pad=0.015, aspect=30)
        if colour_label:
            colorbar.set_label(colour_label, fontsize=plt.rcParams["axes.labelsize"])
        colorbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])
        colorbar.outline.set_linewidth(plt.rcParams["axes.linewidth"])
    if n_dropped:
        # In the legend rather than only in a CSV: a panel silently missing 14 of 43 channels
        # looks complete.
        ax.plot([], [], linestyle="none", label=f"{n_dropped} channel(s) with no centre frequency")
        ax.legend(loc="best")
    style_axes(ax)
    return handle


def new_figure(n_rows: int, n_cols: int = 1, *, height_per_row: float = 2.2,
               width: float = FIGURE_WIDTH) -> Tuple[Any, Any]:
    """Create a figure and its axes grid at the pipeline's standard proportions.

    Args:
        n_rows: Number of stacked panels.
        n_cols: Number of columns.
        height_per_row: Height in inches allotted to each row. The default gives a single
            full-width panel the 3.3 : 1 aspect of a wide journal panel.
        width: Total width in inches; :data:`FIGURE_WIDTH` -- a double column -- by default.

    Returns:
        ``(fig, axes)`` with ``axes`` always a 2-D array, so a caller indexes it the same way
        whatever the grid shape -- ``squeeze=False`` rather than a shape check at every site.
    """
    fig, axes = plt.subplots(
        int(n_rows), int(n_cols),
        figsize=(float(width), float(height_per_row) * int(n_rows)),
        squeeze=False,
    )
    return fig, axes


def label_channel_blocks(ax: Any, n_scattering: int, n_total: int) -> None:
    """Mark the scattering / phase-harmonic boundary on a channel axis.

    Args:
        ax: Target axes, whose y-axis runs over feature channels.
        n_scattering: Width of the scattering block, from the batch rather than a literal.
        n_total: Total channel count $c_y$.
    """
    if 0 < int(n_scattering) < int(n_total):
        ax.axhline(float(n_scattering) - 0.5, color=COLOR_BLACK, linewidth=plt.rcParams["axes.linewidth"], linestyle="--")
        ax.text(
            0.005, float(n_scattering) - 0.5, " phase-harmonic below",
            transform=ax.get_yaxis_transform(), fontsize=plt.rcParams["ytick.labelsize"], va="bottom", color=COLOR_GRAY,
        )


def render_figure(fig: Any, path: Any, *, tight: bool = True) -> Any:
    """Save a figure in the run's configured format at the repository's DPI, and close it.

    ``path`` is a **stem**: the extension is this run's, not the caller's. Every figure-name
    constant in the eval packages is therefore extension-less, and the one place that decides
    what is actually written is :func:`set_figure_format` -- so a run's format is a single config
    key rather than a hundred literals that can disagree with each other.

    Two finishing steps every figure gets on its way out, so no builder has to remember them:
    panel letters on a figure with more than one data panel (:func:`label_panels`), and a layout
    that leaves the room a :func:`footnote` reserved. Stacked panels also get their y labels
    aligned, which is the difference between a column of panels and a stack of separate plots.

    Args:
        fig: The figure. It is closed whether or not the save succeeds; matplotlib holds every
            unclosed figure in a global registry, and a production pass draws hundreds.
        path: Destination **without** an extension. A :class:`~pathlib.Path` or a string.
        tight: Apply ``tight_layout`` before saving. A builder that laid the figure out itself
            passes ``False`` or stamps the figure with :func:`mark_laid_out`; the footnote room
            is honoured either way, because the builder that wrote the footnote is the one that
            laid the figure out around it.

    Returns:
        The path actually written, extension included.

    Raises:
        ValueError: If ``path`` already carries a format extension. That means a hardcoded suffix
            survived somewhere, and appending to it would silently write ``figure.pdf.svg``;
            failing here names the file so the stray literal can be found.
    """
    destination = Path(str(path))
    suffix = destination.suffix.lstrip(".").lower()
    if suffix in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(
            f"render_figure expects a stem without an extension, got {destination.name!r}. The "
            f"format is the run's ({active_figure_format()!r}), set from eval_config.figure_format."
        )
    destination = destination.with_name(f"{destination.name}.{_ACTIVE_FIGURE_FORMAT}")
    label_panels(fig)
    if tight and not getattr(fig, _LAYOUT_DONE_ATTRIBUTE, False):
        try:
            fig.tight_layout(rect=layout_rect(fig))
            fig.align_ylabels()
        except Exception:  # noqa: BLE001 - a layout warning must not lose a completed figure
            pass
    save_figure(fig, str(destination), dpi=SAVE_DPI, close=True)
    return destination


def layout_rect(fig: Any) -> Tuple[float, float, float, float]:
    """The ``tight_layout`` rectangle that keeps the axes clear of a footnote and a suptitle.

    ``tight_layout`` lays the axes out over the whole figure and knows nothing about a
    :func:`footnote` under them or a ``suptitle`` over them, so a page with either would have its
    first panel's title drawn through the suptitle and its x label through the footnote. The room
    for each is taken off the rectangle here instead.

    Args:
        fig: The figure, with or without a footnote or a suptitle.

    Returns:
        ``(left, bottom, right, top)`` in figure fractions.
    """
    bottom = float(getattr(fig, _FOOTNOTE_ATTRIBUTE, 0.0))
    top = 1.0
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        height_in = float(fig.get_size_inches()[1])
        lines = suptitle.get_text().count("\n") + 1
        # The title block plus half a line of clearance, in figure fractions.
        top = 1.0 - (lines + 0.5) * float(suptitle.get_fontsize()) * _FOOTNOTE_LINE_HEIGHT / 72.0 / height_in
    return (0.0, bottom, 1.0, max(top, bottom + 0.1))


def mark_laid_out(fig: Any) -> None:
    """Record that a builder laid ``fig`` out itself, so :func:`render_figure` leaves it alone.

    For a figure built on an explicit gridspec -- one with a shared colour-axis column, say --
    ``tight_layout`` would recompute every margin from the artists and, on a tall page, fail and
    fall back to matplotlib's default spacing, which is how a carefully placed page ends up
    crushed into the middle of a blank sheet. The panel lettering still happens at render time.

    Args:
        fig: The figure.
    """
    setattr(fig, _LAYOUT_DONE_ATTRIBUTE, True)


def sequence_axis(length: int) -> np.ndarray:
    """Return ``arange(length)`` as float, for a ribbon's x coordinate.

    Args:
        length: Number of positions.

    Returns:
        A $(\\mathrm{length},)$ float array.
    """
    return np.arange(int(length), dtype=np.float64)


def as_columns(values: Any, names: Sequence[str]) -> dict:
    """Zip a $(B, N)$ array into ``{name: column}`` for a DataFrame.

    Args:
        values: The array, $(B, N)$.
        names: One name per column; must match ``values.shape[1]``.

    Returns:
        A mapping from name to a $(B,)$ array.

    Raises:
        ValueError: If the name count does not match the column count. Silently zipping the
            shorter of the two would drop columns off the end of every emitted CSV.
    """
    array = np.asarray(to_numpy(values))
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(
            f"expected a (B, {len(names)}) array to match {len(names)} column name(s), got "
            f"shape {tuple(array.shape)}."
        )
    return {str(name): array[:, index] for index, name in enumerate(names)}
