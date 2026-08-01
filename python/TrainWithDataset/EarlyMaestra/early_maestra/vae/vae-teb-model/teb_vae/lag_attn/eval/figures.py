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
    """Draw one violin per group, tolerating empty and all-``NaN`` groups.

    A violin rather than a box: these distributions are routinely bimodal -- a set of recordings
    the model forecasts well and a tail it does not -- and a box plot renders both as the same
    five numbers. The median is marked, so the central tendency is still readable at a glance.

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

    parts = ax.violinplot(
        [finite[index] for index in populated],
        positions=[index + 1.0 for index in populated],
        showmedians=True,
        showextrema=False,
        widths=0.8,
    )
    for position, body in zip(populated, parts["bodies"]):
        body.set_facecolor((colors or {}).get(labels[position], color))
        body.set_alpha(0.65)
        body.set_edgecolor(COLOR_BLACK)
        body.set_linewidth(0.4)
    if "cmedians" in parts:
        parts["cmedians"].set_color(COLOR_VERMILLION)
        parts["cmedians"].set_linewidth(plt.rcParams["lines.linewidth"])

    if reference is not None and np.isfinite(reference):
        ax.axhline(float(reference), color=COLOR_GRAY, linestyle=":", linewidth=plt.rcParams["lines.linewidth"],
                   label=reference_label or f"reference {float(reference):.4g}")
        ax.legend(loc="best")
    style_axes(ax)
    return len(populated)


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
