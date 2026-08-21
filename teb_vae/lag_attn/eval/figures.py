r"""Static matplotlib primitives for the evaluation figures.

Three generic panels -- a histogram, a median-plus-IQR ribbon, and a heatmap with a colourbar --
plus the repository's figure conventions, re-exported so an analysis imports its plotting
surface from one place.

**Nothing here is reimplemented that already exists.** ``SAVE_DPI``,
:func:`~utils.style.apply_publication_style`, :func:`~utils.style.style_axes` and
:func:`~utils.style.save_figure` come from ``utils.style``; the conversions and the colour
literals come from :mod:`teb_vae.lag_attn.figure_primitives`, which the training callback
imports from too. The colours in particular are *not* taken from ``utils.style``: two of the
eight genuinely differ there (``COLOR_PURPLE`` and ``COLOR_BLACK``), and the figures depend on
the hues the model's own plots use, so that a training figure and an eval figure of the same
quantity are the same colour.

**Importing this module does not restyle anything.** ``apply_publication_style`` mutates global
``rcParams``, so calling it at import time would silently restyle any other figure produced in
the same process -- including a test's. The pipeline calls :func:`configure_figure_style` once
at startup instead.

**Every panel tolerates empty and all-``NaN`` input.** An analysis that legitimately found
nothing -- a fully masked split, a metric that is undefined for this checkpoint -- must produce
an empty, labelled figure rather than take down a multi-hour run at its final step.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cbook  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_VERMILLION,
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
    style_axes,
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
    "SAVE_DPI",
    "attach_lag_seconds_axis",
    "binned_violin_panel",
    "configure_figure_style",
    "frequency_scatter",
    "get_class_colors",
    "group_colors",
    "grouped_violin_figure",
    "heatmap_with_colorbar",
    "histogram_panel",
    "label_rows",
    "multi_line_panel",
    "ribbon_plot",
    "safe_vabs",
    "save_figure",
    "shade_warmup",
    "significance_strip",
    "style_axes",
    "to_numpy",
    "violin_panel",
]

#: Drawn on an axes that has no finite data, in place of an empty frame that reads as a bug.
EMPTY_NOTE = "no finite values"


def configure_figure_style() -> None:
    """Apply the repository's publication rcParams. Called once at pipeline start.

    Deliberately not an import side effect -- see the module docstring.
    """
    apply_publication_style()


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

    ax.hist(finite, bins=int(bins), color=color, alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.3)
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

    ax.fill_between(axis_x, low, high, color=color, alpha=0.25, linewidth=0, label="IQR")
    ax.plot(axis_x, median, color=color, linewidth=plt.rcParams["lines.linewidth"], label=label or "median")
    ax.legend(loc="best")
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
    colorbar_label: str = "",
    separator_row: Optional[int] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    interpolation: str = "nearest",
) -> Any:
    """Draw a heatmap with its colourbar, tolerating empty and all-``NaN`` input.

    Args:
        fig: The figure, needed to attach the colourbar.
        ax: Target axes.
        data: The field, $(\\mathrm{rows}, \\mathrm{cols})$.
        title: Panel title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        cmap: Colormap name. ``None`` follows ``symmetric``: diverging ``bwr`` for a signed
            field, sequential ``magma`` for a non-negative one. Tying the two together is what
            keeps them from disagreeing -- a non-negative field drawn on ``bwr`` renders its
            *smallest* values saturated blue and its mid-range white, so the best-forecast
            channel looks extreme and the mediocre one looks neutral. The colourbar stays
            correct throughout, so nothing in the numbers gives the inversion away; only the
            at-a-glance ranking, which is what a heatmap is for, is backwards.
        symmetric: Use a symmetric limit about zero, via
            :func:`~teb_vae.lag_attn.figure_primitives.safe_vabs`. Right for a signed field such
            as a residual; wrong for a non-negative one, which should pass ``False``.
        colorbar_label: Label for the colourbar.
        separator_row: Row index of the last row of the upper feature block. A horizontal rule is
            drawn at ``separator_row + 0.5``, which is where the two blocks actually meet.
        extent: Optional imshow extent, for a panel sharing a physical axis with another.
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
    colormap = cmap if cmap is not None else ("bwr" if symmetric else "magma")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if field.size == 0 or not np.isfinite(field).any():
        _note_empty(ax)
        style_axes(ax, grid="none")
        return None

    if symmetric:
        limit = safe_vabs(field)
        vmin, vmax = -limit, limit
    else:
        finite = field[np.isfinite(field)]
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            # A constant field would otherwise get a degenerate colour range, which matplotlib
            # renders as a single flat colour indistinguishable from an empty panel.
            vmin, vmax = vmin - 0.5, vmax + 0.5

    image = ax.imshow(
        field, aspect="auto", origin="upper", cmap=colormap, vmin=vmin, vmax=vmax,
        interpolation=interpolation, extent=extent,
    )
    if separator_row is not None:
        ax.axhline(float(separator_row) + 0.5, color=COLOR_BLACK, linewidth=plt.rcParams["axes.linewidth"])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    if colorbar_label:
        colorbar.set_label(colorbar_label, fontsize=plt.rcParams["axes.labelsize"])
    colorbar.ax.tick_params(labelsize=7)
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
#: The multiples are chosen so that under ``lag_attn_rws``'s style refinement they come out at
#: exactly that package's ``LINE_HEAVY`` ($2.4$) and ``LINE_THIN`` ($0.65$) -- the weights its
#: ``distributions`` strip already draws an inter-quartile span and a range with. One visual
#: vocabulary for "these are quartiles", across two figures built by different modules.
INNER_BOX_BAR_WEIGHT_RATIO = 2.67
INNER_BOX_WHISKER_WEIGHT_RATIO = 0.72

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
        body.set_facecolor((colors or {}).get(labels[position], color))
        body.set_alpha(0.65)
        body.set_edgecolor(COLOR_BLACK)
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
                body.set_alpha(0.65)
                body.set_edgecolor(COLOR_BLACK)
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
        _note_empty(ax)
        if untestable:
            ax.legend(loc="best")
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
    ax.legend(loc="best", ncol=2)
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
        s=18, alpha=0.85, edgecolor=COLOR_BLACK, linewidth=0.3,
    )
    ax.set_xscale("log")
    if shades is not None:
        colorbar = fig.colorbar(handle, ax=ax, fraction=0.025, pad=0.01)
        if colour_label:
            colorbar.set_label(colour_label, fontsize=plt.rcParams["axes.labelsize"])
        colorbar.ax.tick_params(labelsize=7)
    if n_dropped:
        # In the legend rather than only in a CSV: a panel silently missing 14 of 43 channels
        # looks complete.
        ax.plot([], [], linestyle="none", label=f"{n_dropped} channel(s) with no centre frequency")
        ax.legend(loc="best")
    style_axes(ax)
    return handle


def new_figure(n_rows: int, n_cols: int = 1, *, height_per_row: float = 2.6,
               width: float = 9.0) -> Tuple[Any, Any]:
    """Create a figure and its axes grid at the pipeline's standard proportions.

    Args:
        n_rows: Number of stacked panels.
        n_cols: Number of columns.
        height_per_row: Height in inches allotted to each row.
        width: Total width in inches.

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


def render_to_pdf(fig: Any, path: Any, *, tight: bool = True) -> Any:
    """Save a figure as PDF at the repository's DPI and close it.

    Args:
        fig: The figure.
        path: Destination path. The caller is responsible for the ``.pdf`` suffix.
        tight: Apply ``tight_layout`` before saving.

    Returns:
        The path written.
    """
    if tight:
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - a layout warning must not lose a completed figure
            pass
    save_figure(fig, str(path), dpi=SAVE_DPI, close=True)
    return path


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
