r"""The one plotting surface an analysis imports, and the one lag-axis label it may draw with.

The generic panels -- histogram, median-plus-IQR ribbon, heatmap with a colourbar, violin,
multi-line overlay -- are the shared ones, bound rather than forked. They already carry the two
properties that matter for an unattended multi-hour run, and both are asserted against this seam
rather than taken on trust:

**Two things here are this package's own rather than the sibling's**, and both are stated where
they are defined: the cohort palette (:data:`CLINICAL_CLASS_COLORS` -- green, amber, red by
severity, not ``utils.style``'s blue-first mapping) and a style refinement over the shared
publication rcParams (:data:`STYLE_REFINEMENT`, with the named weight and type scale beside it).
Neither reaches the sibling: the refinement is applied by this package's
:func:`configure_figure_style`, which a sibling run never calls.

The shared panel primitives now read their weights and sizes from the **active** ``rcParams``
rather than from literals, which is what lets that refinement reach inside them -- so a figure
drawn through this seam is lighter than the same builder produces for the sibling, without either
package owning a second copy of the builder.

* **Every panel tolerates empty and all-``NaN`` input.** An analysis that legitimately found
  nothing -- a fully masked split, a metric undefined for this checkpoint -- draws
  :data:`EMPTY_NOTE` on an otherwise empty axes instead of taking down the run at its final step.
* **Importing this module restyles nothing.** ``apply_publication_style`` mutates global
  ``rcParams``, so an import-time call would silently restyle every other figure in the process,
  including a test's. :func:`configure_figure_style` is called once at run start instead.

:func:`render_figure` *closes* the figure it saves. That is not tidiness: a production pass draws
one figure per analysis per grouping axis, and matplotlib holds every unclosed figure alive in
its global registry, so a leak is measured in hundreds of megabytes rather than in style.

**The lag axis is this package's own.** A lag index $\ell$ is not seconds, and the two seconds
figures it maps onto are different quantities -- $\tau_{\mathrm{compensated}} = 4(\ell + \delta)$
is the residual physiological delay, and only the *sensor* figure undoes the $20$ s the
preprocessing already removed, which it does by **subtracting** it: the preprocessing *advanced*
the source trace, so reaching the uncorrected timeline moves the figure down. :data:`COMPENSATED_LAG_AXIS_LABEL` names the first, and it is bound
here so that a figure drawn through this seam and the number reported beside it cannot disagree
about which of the two is shown.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from teb_vae.lag_attn_rws.eval._reuse import figures, labels
from teb_vae.lag_attn.nets.lag_report import COMPENSATED_LAG_AXIS_LABEL

#: Drawn on an axes that had no finite data, in place of an empty frame that reads as a bug.
EMPTY_NOTE = figures.EMPTY_NOTE

# =============================================================================
# The type and weight scale
# =============================================================================
#: Line weights, named by the job rather than by the number. Five steps, because a figure with
#: more than a handful of distinct weights reads as accidental: the eye takes a change of weight
#: as a change of meaning, so every weight has to mean something.
#:
#: Calibrated against the refined ``axes.linewidth`` of $0.5$ below: the frame is the lightest mark
#: on the page, a hairline sits at it, and the emphasis weight is roughly twice it. Nothing here is
#: heavier than :data:`LINE_HEAVY`, which exists for the one mark that is a *bar* rather than a
#: line -- the inter-quartile span of a per-recording strip.
LINE_HAIRLINE = 0.5
LINE_THIN = 0.65
LINE_REGULAR = 0.9
LINE_EMPHASIS = 1.05
LINE_HEAVY = 2.4

#: Type sizes, a step below the shared style's. Same principle: four steps, each with a job. The
#: body size comes from ``rcParams`` and is not restated here -- these are the sizes for text that
#: is deliberately smaller than the axis labels.
FONT_TINY = 4.5
FONT_SMALL = 5.5
FONT_LABEL = 6.0
FONT_NOTE = 7.5

#: The refinement this package applies on top of the shared publication style. Every value is a
#: reduction: the shared style is tuned for figures read at a page's width, and an evaluation run
#: draws eight-row stacks that are read at a screen's. Lighter frames, tighter type and a quieter
#: grid let the data carry the ink.
#:
#: Applied as a delta over ``apply_publication_style`` rather than replacing it, so the serif
#: family, the DPI and the white background stay the repository's.
STYLE_REFINEMENT = {
    # Type: one step down across the board, and the title no longer shouts over its own axes.
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "axes.titlepad": 4.0,
    "axes.labelpad": 2.5,
    # Frame and ticks: the lightest marks on the page, so nothing structural competes with data.
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    # Grid: present enough to read a value off, quiet enough to disappear while reading a shape.
    "grid.linewidth": 0.25,
    "grid.alpha": 0.18,
    # Legend: compact and unboxed-looking. A heavy frame around a legend is the most common way a
    # careful figure still looks unconsidered.
    "legend.framealpha": 0.85,
    "legend.edgecolor": figures.COLOR_LIGHT_GRAY,
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "legend.columnspacing": 1.0,
    # Data defaults, for every artist drawn without an explicit weight.
    "lines.linewidth": LINE_REGULAR,
    "lines.markersize": 2.5,
    "lines.markeredgewidth": 0.5,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
}


def configure_figure_style(figure_format: Optional[str] = None) -> None:
    """Apply the repository's publication style, then this package's refinement over it.

    Called once at run start, and deliberately not an import side effect: ``rcParams`` are global
    to the process, so an import-time call would restyle every other figure drawn in it --
    including a test's. The run's figure format rides along for the same reason -- it is one
    property of the whole pass, fixed once, rather than an argument on every drawing call.

    Args:
        figure_format: The image format every figure of this run is written in, as a matplotlib
            filetype. ``None`` leaves the active format alone, which is
            :data:`~teb_vae.lag_attn.eval.figures.DEFAULT_FIGURE_FORMAT` in a fresh process.

    Raises:
        ValueError: If ``figure_format`` is not a format this matplotlib build can write.
    """
    import matplotlib.pyplot as plt

    figures.configure_figure_style(figure_format)
    plt.rcParams.update(STYLE_REFINEMENT)


def style_axes(ax, *, grid: str = "major") -> None:
    """Style one axes, then re-apply the refined frame and grid weights over the shared ones.

    ``utils.style.style_axes`` sets the spine width and the grid weight as *literals*, so they
    survive the ``rcParams`` refinement above and would leave every panel framed a third heavier
    than the style asks for. Re-applying afterwards is what makes the refinement actually reach
    the page.

    Args:
        ax: Target axes.
        grid: One of ``'major'``, ``'both'``, ``'none'`` -- passed straight through.
    """
    import matplotlib.pyplot as plt

    figures.style_axes(ax, grid=grid)
    for spine in ax.spines.values():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
    if grid != "none":
        ax.grid(
            True, linestyle="-",
            alpha=plt.rcParams["grid.alpha"],
            linewidth=plt.rcParams["grid.linewidth"],
            color=figures.COLOR_LIGHT_GRAY,
        )

#: Figure construction and output. ``render_figure`` tight-layouts, saves at the repository's DPI,
#: closes the figure and returns the path it wrote.
new_figure = figures.new_figure
render_figure = figures.render_figure
figure_filename = figures.figure_filename

#: The generic panels. Each takes an axes and draws into it, so a figure builder composes them
#: rather than each analysis owning a layout.
binned_violin_panel = figures.binned_violin_panel
heatmap_with_colorbar = figures.heatmap_with_colorbar
histogram_panel = figures.histogram_panel
multi_line_panel = figures.multi_line_panel
ribbon_plot = figures.ribbon_plot
significance_strip = figures.significance_strip
violin_panel = figures.violin_panel

#: The grouped violin figure the by-class and by-subgroup variants are drawn with. The colours it
#: uses come from :func:`group_colors` below rather than from the shared default.
grouped_violin_figure = figures.grouped_violin_figure

#: The palette. Bound here rather than imported per analysis so a quantity keeps one colour across
#: the run: these are the hues the training figures use, not ``utils.style``'s, two of which
#: differ.
COLOR_BLACK = figures.COLOR_BLACK
COLOR_BLUE = figures.COLOR_BLUE
COLOR_GRAY = figures.COLOR_GRAY
COLOR_GREEN = figures.COLOR_GREEN
COLOR_LIGHT_GRAY = figures.COLOR_LIGHT_GRAY
COLOR_ORANGE = figures.COLOR_ORANGE
COLOR_PURPLE = figures.COLOR_PURPLE
COLOR_VERMILLION = figures.COLOR_VERMILLION

# =============================================================================
# The cohort palette
# =============================================================================
#: Clinical class to hex colour: green for healthy, amber for acidosis, red for HIE -- the
#: severity reading a clinician already carries, so a figure needs no legend lookup to be read
#: the right way round.
#:
#: **Deliberately not ``utils.style.CLASS_COLORS_DEFAULT``**, which paints ``healthy`` blue. That
#: table is shared with the ``lag_attn`` sibling and with ``model/transformer_experiment``, so
#: repainting it there would restyle two other projects' figures to satisfy this one's convention.
#: The cost of owning it here is stated rather than hidden: an ``lag_attn_rws`` eval figure of a
#: cohort is **not** the same colour as a training-callback figure of that cohort, and a reader
#: putting the two side by side must read the legend rather than the hue.
#:
#: The three are ordered by luminance as well as by hue ($0.42$ / $0.70$ / $0.36$), so the
#: severity axis survives a greyscale print, which a pure hue triple does not.
CLINICAL_CLASS_COLORS: Dict[str, str] = {
    "healthy": "#2E8B57",
    "acidosis": "#E8A33D",
    "hie": "#C0392B",
}

#: How far the first and last subgroup of a class are shaded off its class colour: positive blends
#: toward white, negative toward black. A subgroup is a *subdivision* of a class rather than a
#: cohort of its own, so it is drawn as a shade of that class -- on the eight-cohort figures that
#: keeps class membership legible, which eight unrelated hues do not.
SUBGROUP_TINT_RANGE = (0.45, -0.25)


def _blend(color: str, amount: float) -> str:
    """Blend a hex colour toward white or black.

    Args:
        color: ``'#rrggbb'``.
        amount: In $[-1, 1]$. Positive blends toward white, negative toward black, $0$ returns the
            colour unchanged.

    Returns:
        The blended colour as ``'#rrggbb'``.
    """
    channels = [int(color[index:index + 2], 16) for index in (1, 3, 5)]
    target = 255.0 if amount >= 0.0 else 0.0
    weight = abs(float(amount))
    return "#" + "".join(
        f"{int(round(value + (target - value) * weight)):02x}" for value in channels
    )


def _class_of(group: str) -> Optional[str]:
    """Return the clinical class a canonical subgroup belongs to.

    Args:
        group: A subgroup stem such as ``'healthy_bg_cs'``.

    Returns:
        The class name, or ``None`` when the stem names no known class. Matched on the class name
        as a prefix component rather than by a table, so the eight stems and the three classes
        cannot drift apart -- both come from ``labels``.
    """
    for name in CLINICAL_CLASS_COLORS:
        if group == name or group.startswith(f"{name}_"):
            return name
    return None


def _build_subgroup_colors() -> Dict[str, str]:
    """Shade each class's subgroups across :data:`SUBGROUP_TINT_RANGE`, in canonical order.

    Returns:
        Subgroup stem to hex colour. A class contributing one subgroup gets the class colour
        itself rather than an arbitrary end of the range.
    """
    members: Dict[str, List[str]] = {}
    for group in labels.CANONICAL_SUBGROUPS:
        name = _class_of(str(group))
        if name is not None:
            members.setdefault(name, []).append(str(group))

    lightest, darkest = SUBGROUP_TINT_RANGE
    colors: Dict[str, str] = {}
    for name, group_names in members.items():
        count = len(group_names)
        for index, group in enumerate(group_names):
            amount = (
                0.0 if count < 2
                else lightest + (darkest - lightest) * index / float(count - 1)
            )
            colors[group] = _blend(CLINICAL_CLASS_COLORS[name], amount)
    return colors


#: Subgroup stem to hex colour, built once from the two tables above.
SUBGROUP_COLORS: Dict[str, str] = _build_subgroup_colors()


def group_colors(groups: Sequence[str]) -> Dict[str, str]:
    """Return this package's colour for each cohort label.

    The mapping is a **table**, not an assignment pass, which is what makes it order-independent:
    a cohort asked for alone, or among others, or in another order, comes back the same colour, so
    two figures of overlapping cohorts can be put side by side. A label in neither table -- a
    non-canonical shard stem, an unknown class code -- falls back to the shared palette, which is
    an assignment pass and is therefore the only order-dependent part.

    Args:
        groups: The cohort labels appearing in a figure.

    Returns:
        Label to hex colour, covering every label given.
    """
    resolved: Dict[str, str] = {}
    unknown: List[str] = []
    for group in groups:
        name = str(group)
        colour = CLINICAL_CLASS_COLORS.get(name) or SUBGROUP_COLORS.get(name)
        if colour is None:
            unknown.append(name)
        else:
            resolved[name] = colour
    if unknown:
        resolved.update(figures.group_colors(unknown))
    return resolved


# =============================================================================
# The windowed comparison page
# =============================================================================
#: Width in inches of the page below, wider than every other figure this seam builds. Its x axis
#: carries one window per half hour of a whole labour, three violins deep; at the shared default of
#: $9$ in the bodies of a twenty-five-window profile come out narrower than the marks drawn inside
#: them, and the page stops being a distribution figure at all.
WINDOWS_FIGURE_WIDTH = 13.0

#: Height per row of that page. Every row is the same height because ``new_figure`` allots one, and
#: the significance strips are the rows that would be thinner if it did not.
WINDOWS_ROW_HEIGHT = 3.0


def windowed_comparison_figure(
    readouts: Sequence[Any],
    *,
    groups: Sequence[str],
    bin_width: float,
    min_body_size: int,
    xlabel: str,
    ylabel: str,
    delivery_orientation: bool,
    window_field: str = "time_bin",
) -> Any:
    r"""Build one clock's page: the distributions per window, their significance, and the effects.

    Three questions, one page and one $x$ axis, because they are read together and reading them
    across two files means holding a window's coordinate in mind while changing pages:

    * **What is the distribution** behind each point of the trajectory -- a violin per (window,
      cohort) cell over one value per recording;
    * **is it anything** -- the Holm-adjusted $p$ of that window's omnibus, against $\alpha$,
      directly beneath the violins it describes and on their coordinate;
    * **does it matter** -- Cliff's delta for every cohort pair that survived, in the bottom row.

    Built here rather than in an analysis because **both clocks draw it** and an analysis may not
    import another; and composed out of the shared panels rather than owning its own marks, so a
    violin here is the same mark as a violin anywhere else in the run.

    **The orientation is the caller's, and it is the difference between the two clocks.** Time
    before delivery decreases toward the event, so that axis is inverted and delivery sits at the
    right. Time relative to second-stage onset is signed and reads naturally left to right, with
    the onset marked where it falls.

    Args:
        readouts: One ``(name, cells, record)`` per readout. ``cells`` is that readout's
            ``{group: values}`` per window, **positionally aligned** with the record's own
            ``per_window`` list; ``record`` is the significance record the analysis produced.
        groups: Cohort labels in the order they should be dodged, left to right.
        bin_width: Window width in the x coordinate's units.
        min_body_size: Fewest recordings a cell may have and still be drawn as a density; the
            caller passes the same floor its test excludes a cell at.
        xlabel: X-axis label, naming the clock and its sign convention.
        ylabel: Y-axis label for the violin panels.
        delivery_orientation: ``True`` inverts the axis so delivery sits at the right; ``False``
            keeps the natural direction and marks $x = 0$.
        window_field: The key each per-window record carries its window identifier under, matching
            what the analysis passed to the shared inference.

    Returns:
        The figure, unsaved and unclosed -- the caller renders and closes it.

    Raises:
        ValueError: If a readout's cells and its record's windows are of different lengths, which
            would draw one window's distribution under another window's $p$-value.
    """
    figure, axes = new_figure(
        2 * max(len(readouts), 1) + 1,
        height_per_row=WINDOWS_ROW_HEIGHT,
        width=WINDOWS_FIGURE_WIDTH,
    )
    order = list(groups)
    colours = group_colors(order)
    # Where each cohort sits on the axis. The effect rows below are laid out by it rather than by
    # the alphabetical order of their labels, so a row reads in the same direction as the pair it
    # names -- more severe against less severe. A cohort the order does not know sorts after one
    # it does, exactly as ``cohort.ordered_groups`` places it.
    position = {group: index for index, group in enumerate(order)}
    effects: List[Dict[str, Any]] = []

    for index, (name, cells, record) in enumerate(readouts):
        per_window = list(record.get("per_window") or [])
        windows = list(cells)
        if len(windows) != len(per_window):
            raise ValueError(
                f"readout {name!r} carries {len(windows)} window(s) of values against "
                f"{len(per_window)} tested window(s); they are positionally aligned, so a "
                f"mismatch would draw one window's distribution under another window's p-value."
            )
        centres = [float(row["bin_center_h"]) for row in per_window]

        violins = axes[2 * index, 0]
        binned_violin_panel(
            violins, windows, centres,
            groups=order, bin_width=bin_width, min_body_size=min_body_size, colors=colours,
            title=f"{name} per window, by {labels.CLASS_COLUMN}",
            ylabel=ylabel,
        )
        strip = axes[2 * index + 1, 0]
        significance_strip(
            strip, centres, [row.get("p_holm", float("nan")) for row in per_window],
            alpha=float(record.get("alpha", 0.05)),
            bin_width=bin_width,
            title=f"{name}: cohort difference per window (Kruskal-Wallis, Holm)",
            xlabel=xlabel,
        )
        for axis in (violins, strip):
            if delivery_orientation:
                axis.invert_xaxis()
            else:
                axis.axvline(
                    0.0, color=COLOR_GRAY, linestyle=":", linewidth=LINE_REGULAR, zorder=0
                )

        centre_by_key = {
            str(row.get(window_field)): float(row["bin_center_h"]) for row in per_window
        }
        for key, comparisons in (record.get("pairwise") or {}).items():
            for item in comparisons:
                effects.append({
                    "row": f"{name}: {item['left']} vs {item['right']}",
                    # Readout first, then the two cohorts' places on the axis: the row order the
                    # heatmap draws, computed here because this is where both are known.
                    "rank": (
                        index,
                        position.get(str(item["left"]), len(order)),
                        position.get(str(item["right"]), len(order)),
                    ),
                    "centre": centre_by_key.get(str(key), float("nan")),
                    "delta": float(item.get("cliffs_delta", float("nan"))),
                })

    _draw_effect_heatmap(
        figure, axes[-1, 0], effects, xlabel=xlabel, descending=delivery_orientation
    )
    return figure


def _draw_effect_heatmap(
    figure: Any,
    ax: Any,
    effects: Sequence[Dict[str, Any]],
    *,
    xlabel: str,
    descending: bool,
) -> None:
    """Draw Cliff's delta for every surviving cohort pair, windows across and pairs down.

    The column order follows the panels above rather than the natural sort, so a column of this
    heatmap sits under the window it describes; the ancestor's version does not, and reading it
    against its own bar panel means reversing one of the two by eye. The **row** order is the
    cohort order the pairs were tested in rather than the alphabetical order of their labels, so
    ``hie vs acidosis`` sits above ``acidosis vs healthy`` here as it does in the pairwise CSV,
    and the whole page -- violins, columns and rows alike -- runs worst to least severe.

    Args:
        figure: The parent figure, for the colourbar.
        ax: Target axes.
        effects: ``{'row', 'rank', 'centre', 'delta'}`` per surviving comparison, ``rank`` being
            the caller's own (readout, left cohort, right cohort) position tuple.
        xlabel: X-axis label, matching the panels above.
        descending: Whether the windows run right to left, as the delivery clock does.
    """
    # Keyed on the smallest rank a row was seen with, because one pair can survive in several
    # windows and each of them carries the same rank anyway. A row the caller did not rank sorts
    # after every one it did and then by its label -- never dropped, the convention every cohort
    # order in this package follows for a label it does not know.
    rank_by_row: Dict[str, Any] = {}
    for item in effects:
        label = str(item["row"])
        rank = tuple(item.get("rank") or (float("inf"),))
        if label not in rank_by_row or rank < rank_by_row[label]:
            rank_by_row[label] = rank
    rows = sorted(rank_by_row, key=lambda label: (rank_by_row[label], label))
    columns = sorted(
        {float(item["centre"]) for item in effects if np.isfinite(item["centre"])},
        reverse=bool(descending),
    )
    if not rows or not columns:
        heatmap_with_colorbar(
            figure, ax, np.zeros((0, 0)),
            title="Cliff's delta (no window survived Holm)",
            symmetric=True, colorbar_label="Cliff's delta",
        )
        return

    field = np.full((len(rows), len(columns)), np.nan)
    for item in effects:
        if np.isfinite(item["centre"]):
            field[rows.index(str(item["row"])), columns.index(float(item["centre"]))] = item["delta"]
    heatmap_with_colorbar(
        figure, ax, field,
        title="Cliff's delta for the surviving cohort pairs",
        symmetric=True, colorbar_label="Cliff's delta",
    )
    figures.label_rows(ax, rows)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([f"{value:g}" for value in columns], fontsize=FONT_SMALL)
    ax.set_xlabel(xlabel)


__all__ = [
    "CLINICAL_CLASS_COLORS",
    "COLOR_BLACK",
    "COLOR_BLUE",
    "COLOR_GRAY",
    "COLOR_GREEN",
    "COLOR_LIGHT_GRAY",
    "COLOR_ORANGE",
    "COLOR_PURPLE",
    "COLOR_VERMILLION",
    "COMPENSATED_LAG_AXIS_LABEL",
    "EMPTY_NOTE",
    "FONT_LABEL",
    "FONT_NOTE",
    "FONT_SMALL",
    "FONT_TINY",
    "LINE_EMPHASIS",
    "LINE_HAIRLINE",
    "LINE_HEAVY",
    "LINE_REGULAR",
    "LINE_THIN",
    "STYLE_REFINEMENT",
    "SUBGROUP_COLORS",
    "SUBGROUP_TINT_RANGE",
    "WINDOWS_FIGURE_WIDTH",
    "WINDOWS_ROW_HEIGHT",
    "binned_violin_panel",
    "configure_figure_style",
    "figure_filename",
    "group_colors",
    "grouped_violin_figure",
    "heatmap_with_colorbar",
    "histogram_panel",
    "multi_line_panel",
    "new_figure",
    "render_figure",
    "ribbon_plot",
    "significance_strip",
    "style_axes",
    "violin_panel",
    "windowed_comparison_figure",
]
