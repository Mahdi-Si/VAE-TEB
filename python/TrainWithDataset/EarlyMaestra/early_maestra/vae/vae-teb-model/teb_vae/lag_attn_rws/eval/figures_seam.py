r"""The one plotting surface an analysis imports, and the one lag-axis label it may draw with.

The generic panels -- histogram, median-plus-IQR ribbon, heatmap with a colourbar, violin,
multi-line overlay -- are the shared ones, bound rather than forked. They already carry the two
properties that matter for an unattended multi-hour run, and both are asserted against this seam
rather than taken on trust:

* **Every panel tolerates empty and all-``NaN`` input.** An analysis that legitimately found
  nothing -- a fully masked split, a metric undefined for this checkpoint -- draws
  :data:`EMPTY_NOTE` on an otherwise empty axes instead of taking down the run at its final step.
* **Importing this module restyles nothing.** ``apply_publication_style`` mutates global
  ``rcParams``, so an import-time call would silently restyle every other figure in the process,
  including a test's. :func:`configure_figure_style` is called once at run start instead.

:func:`render_to_pdf` *closes* the figure it saves. That is not tidiness: a production pass draws
one figure per analysis per grouping axis, and matplotlib holds every unclosed figure alive in
its global registry, so a leak is measured in hundreds of megabytes rather than in style.

**The lag axis is this package's own.** A lag index $\ell$ is not seconds, and the two seconds
figures it maps onto are different quantities -- $\tau_{\mathrm{compensated}} = 4(\ell + \delta)$
is the residual physiological delay, and only the *sensor* figure adds back the $20$ s the
preprocessing already removed. :data:`COMPENSATED_LAG_AXIS_LABEL` names the first, and it is bound
here so that a figure drawn through this seam and the number reported beside it cannot disagree
about which of the two is shown.
"""
from __future__ import annotations

from teb_vae.lag_attn_rws.eval._reuse import figures
from teb_vae.lag_attn_rws.nets.lag_report import COMPENSATED_LAG_AXIS_LABEL

#: Drawn on an axes that had no finite data, in place of an empty frame that reads as a bug.
EMPTY_NOTE = figures.EMPTY_NOTE

#: Called once at run start. Deliberately not an import side effect -- see the module docstring.
configure_figure_style = figures.configure_figure_style

#: Figure construction and output. ``render_to_pdf`` tight-layouts, saves at the repository's DPI,
#: closes the figure and returns the path it wrote.
new_figure = figures.new_figure
render_to_pdf = figures.render_to_pdf

#: The generic panels. Each takes an axes and draws into it, so a figure builder composes them
#: rather than each analysis owning a layout.
heatmap_with_colorbar = figures.heatmap_with_colorbar
histogram_panel = figures.histogram_panel
multi_line_panel = figures.multi_line_panel
ribbon_plot = figures.ribbon_plot
violin_panel = figures.violin_panel

#: The grouped violin figure the by-class and by-subgroup variants are drawn with, and the stable
#: per-group colour map behind it -- which is ``utils.style``'s, so an eval figure of a cohort is
#: the same colour as the training figure of that cohort.
group_colors = figures.group_colors
grouped_violin_figure = figures.grouped_violin_figure

#: Axis styling, for the few panels a generic builder cannot draw -- a bar chart of skill scores
#: with its confidence whiskers, a profile with a shaded structural region. Bound rather than
#: reimplemented so a hand-drawn axes still looks like every other one in the run.
style_axes = figures.style_axes

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

__all__ = [
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
    "configure_figure_style",
    "group_colors",
    "grouped_violin_figure",
    "heatmap_with_colorbar",
    "histogram_panel",
    "multi_line_panel",
    "new_figure",
    "render_to_pdf",
    "ribbon_plot",
    "style_axes",
    "violin_panel",
]
