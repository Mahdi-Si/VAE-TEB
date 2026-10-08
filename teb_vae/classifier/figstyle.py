r"""The classifier's figure style: one palette, one set of ``rcParams`` and one way to finish and save a figure.

:mod:`teb_vae.classifier.report` draws through this module (``report._seam()``). Every name defined here overrides
the shared VAE seam (:mod:`teb_vae.lag_attn_cfs.eval.figures_seam`); every other name (``violin_panel``,
``heatmap_with_colorbar``, ``windowed_comparison_figure``, ``figures``, ...) falls through to the seam on first use
(:func:`__getattr__`). The seam itself is not changed, so the VAE evaluation figures keep their own style.

The rules of the style:

* **Type.** A sans-serif face (Arial, else DejaVu Sans) at 8 pt for labels and 7 pt for ticks and keys. These
  figures are read on a screen at full page width, not shrunk into a journal column.
* **Frame.** Left and bottom spines only, in a mid grey, with a light solid grid behind the data. The data is the
  darkest ink on the page.
* **Colour.** One qualitative series palette (:data:`LINE_PALETTE`) for models, policies and runs; fixed colours
  for the three rates (:data:`RATE_COLORS`); the clinical classes keep the seam's severity colours (green, amber,
  red), so a cohort has the same colour here and in the VAE figures. Reference marks (chance line, $\alpha$, zero)
  are thin and grey.
* **Layout.** Matplotlib's constrained layout on every builder-made figure. A figure carries at most one title
  (left-aligned, :func:`set_title`), one key shared by all its panels (:func:`add_key`, drawn under the panels) and
  one note line (:func:`caveat_note`). No panel letters: the panel titles and facet strips name each panel.

Nothing heavy is imported at module load, so the training callback can use :data:`RC` and the palette without the
evaluation stack.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---- palette ---------------------------------------------------------------------------------------------------
#: Text and the darkest marks.
INK = "#262A30"
#: Reference lines (chance, $\alpha$, zero), secondary text and notes.
MUTED = "#6B7280"
#: Light reference marks and unknown or missing members.
FAINT = "#A7AEB8"
#: The grid behind the data.
GRID = "#E6E8EB"
#: Axis spines and ticks.
SPINE = "#9AA1AB"
#: Background of a facet strip (the row label of a faceted page).
STRIP = "#EEF0F3"
#: The minor grid: one line per time bin or per 0.1 of a rate, lighter than :data:`GRID`.
GRID_MINOR = "#F2F3F5"
#: Edge of a filled marker on a trace: a thin dark ring, so the marker does not cut a white gap into its line.
EDGE = INK

BLUE = "#2F6DB5"
ORANGE = "#E3812B"
TEAL = "#2A9D8F"
ROSE = "#C8475B"
VIOLET = "#7B5EA7"
BROWN = "#8C6D46"
OCHRE = "#C9A227"
SLATE = "#5C6670"

#: Models, policies and compared runs, in this order. Strongly separated hues first; all stay readable as thin
#: lines on white and separate under the common colour-vision deficiencies by lightness as well as by hue.
LINE_PALETTE = (BLUE, ORANGE, TEAL, ROSE, VIOLET, BROWN, OCHRE, SLATE)

#: Sensitivity, specificity and FPR, with their markers. The same three colours on every rate figure.
RATE_COLORS = {"sens": BLUE, "spec": TEAL, "fpr": ROSE}

#: Names the report reads from the seam, bound to this palette.
COLOR_BLUE = BLUE
COLOR_ORANGE = ORANGE
COLOR_GREEN = TEAL
COLOR_PURPLE = VIOLET
COLOR_VERMILLION = ROSE
COLOR_GRAY = MUTED
COLOR_BLACK = INK
COLOR_LIGHT_GRAY = GRID

#: Sequential colour map of the confusion matrices: white to the series blue to a deep navy.
SEQUENTIAL_CMAP = "clf_blues"
_SEQUENTIAL_STOPS = ("#F7F9FC", "#C9DAF0", "#7FA6D8", BLUE, "#163A66")

# ---- weights and type --------------------------------------------------------------------------------------------
#: Line weights by job. The report draws a pooled estimate at ``LINE_EMPHASIS * 2``, a secondary line at
#: ``LINE_REGULAR``, per-fold lines at ``LINE_THIN`` (faded) and reference lines at ``LINE_HAIRLINE``.
LINE_HAIRLINE = 0.7
LINE_THIN = 0.8
LINE_REGULAR = 1.1
LINE_EMPHASIS = 0.95
LINE_HEAVY = 3.0
#: Marker diameter (pt) of a point estimate on a trace.
MARKER_SMALL = 3.2
#: Edge width (pt) of a filled marker (:data:`EDGE`).
MARKER_EDGE = 0.45
#: Text smaller than the axis labels.
FONT_TINY = 6.0
FONT_SMALL = 6.5
FONT_LABEL = 7.0
FONT_NOTE = 7.0
FONT_TITLE = 10.0

#: Figure width (in) of a page that runs on a time axis, and of a page of single panels.
WINDOWS_FIGURE_WIDTH = 12.0
FIGURE_WIDTH = 7.4

#: The ``rcParams`` of every classifier figure, applied over the seam's base style by :func:`configure_figure_style`.
RC: Dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Segoe UI", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.0,
    "text.color": INK,
    "axes.titlesize": 8.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlepad": 5.0,
    "axes.titlecolor": INK,
    "axes.labelsize": 8.0,
    "axes.labelcolor": INK,
    "axes.labelpad": 3.0,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.04,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.color": SPINE,
    "ytick.color": SPINE,
    "xtick.labelcolor": "#4B5260",
    "ytick.labelcolor": "#4B5260",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.8,
    "lines.solid_capstyle": "round",
    "patch.linewidth": 0.6,
    "hatch.linewidth": 0.5,
    "hatch.color": "white",
    "errorbar.capsize": 0.0,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-3, 4),
    "axes.formatter.useoffset": False,
    "figure.facecolor": "white",
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.06,
    "figure.constrained_layout.hspace": 0.035,
    "figure.constrained_layout.wspace": 0.035,
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "savefig.pad_inches": 0.05,
}

#: Text of an empty panel.
EMPTY_NOTE = "no finite values"

#: Figure attributes this module reads when it finishes a figure.
_TITLE, _NOTE, _KEY = "_clf_title", "_clf_note", "_clf_key"


def __getattr__(name: str) -> Any:
    """Fall through to the shared seam for every name this module does not define.

    Args:
        name: The attribute looked up.

    Returns:
        The seam's attribute of that name.
    """
    if name.startswith("__"):
        raise AttributeError(name)
    from teb_vae.lag_attn_cfs.eval import figures_seam

    return getattr(figures_seam, name)


def _register_cmap() -> None:
    """Register :data:`SEQUENTIAL_CMAP` once per process."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    if SEQUENTIAL_CMAP not in matplotlib.colormaps:
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list(SEQUENTIAL_CMAP, _SEQUENTIAL_STOPS))


def configure_figure_style(figure_format: Optional[str] = None) -> None:
    """Apply the seam's base style, then :data:`RC`, and fix the run's figure format.

    Args:
        figure_format: The format every later :func:`render_figure` writes; ``None`` keeps the active one.
    """
    import matplotlib.pyplot as plt
    from teb_vae.lag_attn_cfs.eval import figures_seam

    figures_seam.configure_figure_style(figure_format)
    plt.rcParams.update(RC)
    _register_cmap()


def group_colors(groups: Sequence[str]) -> Dict[str, str]:
    """The seam's class and subgroup colours; any other label takes :data:`LINE_PALETTE` in order.

    Args:
        groups: The labels in a figure.

    Returns:
        Label to hex colour.
    """
    from teb_vae.lag_attn_cfs.eval import figures_seam

    out, extra = {}, []
    for g in map(str, groups):
        col = figures_seam.CLINICAL_CLASS_COLORS.get(g) or figures_seam.SUBGROUP_COLORS.get(g)
        if col is None:
            extra.append(g)
        else:
            out[g] = col
    return out | {g: LINE_PALETTE[i % len(LINE_PALETTE)] for i, g in enumerate(extra)}


def style_axes(ax: Any, *, grid: str = "both") -> None:
    """Open frame and a light grid behind the data.

    Args:
        ax: The axes.
        grid: ``"both"`` (default), ``"x"`` or ``"y"`` for a grid on those axes; ``"none"`` boxes the frame and draws
            no grid (images).
    """
    import matplotlib.pyplot as plt

    ax.set_axisbelow(True)
    boxed = grid == "none"
    for name, spine in ax.spines.items():
        spine.set_linewidth(plt.rcParams["axes.linewidth"])
        spine.set_color(SPINE)
        if name in ("top", "right"):
            spine.set_visible(boxed)
    ax.grid(False)
    if not boxed:
        axis = {"major": "both"}.get(grid, grid)
        ax.grid(True, axis=axis, which="major", color=GRID, lw=0.6, ls="-")
        ax.grid(True, axis=axis, which="minor", color=GRID_MINOR, lw=0.45, ls="-")  # drawn where a minor locator is set


def tint(color: str, amount: float) -> str:
    """Blend a hex colour toward white by ``amount`` in $[0, 1]$: the lighter shade of a member that shares its
    colour with another (``unhealthy_cs_neg`` beside ``unhealthy_cs_pos``), used instead of a dashed line.

    Args:
        color: ``'#rrggbb'``.
        amount: $0$ keeps the colour, $1$ gives white.

    Returns:
        The blended colour as ``'#rrggbb'``.
    """
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(v + (255 - v) * amount):02x}" for v in rgb)


def set_title(fig: Any, text: str) -> None:
    """Set the one figure title, drawn left-aligned above the panels by :func:`finish`.

    Args:
        fig: The figure.
        text: The title; empty draws none.
    """
    setattr(fig, _TITLE, str(text))


def caveat_note(fig: Any, text: str = "") -> float:
    """Set the one note line, drawn under the key by :func:`finish`.

    Args:
        fig: The figure.
        text: The note.

    Returns:
        ``0.0``; :func:`finish` reserves the room.
    """
    setattr(fig, _NOTE, str(text))
    return 0.0


def add_key(fig: Any, handles: Iterable[Any], labels: Iterable[str]) -> None:
    """Add entries to the figure's one key (drawn under the panels by :func:`finish`); a label already in it is
    skipped.

    Args:
        fig: The figure.
        handles: Legend handles.
        labels: Their labels.
    """
    key = getattr(fig, _KEY, None) or ([], [])
    for h, lab in zip(handles, labels):
        if lab and not str(lab).startswith("_") and lab not in key[1]:
            key[0].append(h)
            key[1].append(lab)
    setattr(fig, _KEY, key)


def _key_columns(fig: Any, labels: List[str]) -> int:
    """Columns of the key: as many as fit across the figure on one row, else balanced rows."""
    width = fig.get_size_inches()[0] * 0.92
    item = max(len(str(lab)) for lab in labels) * 7.0 * 0.52 / 72.0 + 0.45
    per_row = max(1, int(width // item))
    rows = math.ceil(len(labels) / per_row)
    return math.ceil(len(labels) / rows)


def finish(fig: Any) -> Any:
    """Lay out ``fig`` and draw its title, key and note.

    A figure a builder laid out itself (``figures.mark_laid_out``) keeps its layout and only gets its note (bottom
    left). Every other figure gets constrained layout over the band the note leaves free, the key under its panels
    and the title over them.

    Args:
        fig: The figure.

    Returns:
        The figure.
    """
    import matplotlib.pyplot as plt

    w, h = fig.get_size_inches()
    note, title, key = getattr(fig, _NOTE, ""), getattr(fig, _TITLE, ""), getattr(fig, _KEY, None)
    if getattr(fig, "_eval_layout_done", False):
        if note:
            fig.text(0.1 / w, 0.08 / h, note, ha="left", va="bottom", fontsize=FONT_NOTE, color=MUTED)
        return fig
    # Bottom band, inches: the note line, then the key rows above it; the panels are laid out above both.
    note_in = 0.26 if note else 0.04
    ncol = _key_columns(fig, key[1]) if key and key[1] else 1
    key_in = math.ceil(len(key[1]) / ncol) * plt.rcParams["legend.fontsize"] * 1.75 / 72.0 + 0.1 if key and key[1] else 0.0
    band = (note_in + key_in) / h
    engine = fig.get_layout_engine()
    if engine is not None and type(engine).__name__ == "ConstrainedLayoutEngine":
        engine.set(rect=(0.0, band, 1.0, 1.0 - band))  # (left, bottom, width, height)
    else:  # a figure built elsewhere (the seam's windowed page) on a grid constrained layout cannot solve
        top = 1.0 - (0.38 / h if title else 0.0)
        fig.set_layout_engine("tight", rect=(0.0, band, 1.0, top))  # (left, bottom, right, top)
    if key and key[1]:
        fig.legend(key[0], key[1], loc="lower center", bbox_to_anchor=(0.5, note_in / h), ncol=ncol, frameon=False,
                   fontsize=plt.rcParams["legend.fontsize"], handlelength=2.0, borderaxespad=0.0)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=FONT_TITLE, fontweight="bold", color=INK)
    if note:
        fig.text(0.08 / w, 0.07 / h, note, ha="left", va="bottom", fontsize=FONT_NOTE, color=MUTED)
    return fig


def render_figure(fig: Any, path: Any, **_: Any) -> Path:
    """Finish ``fig`` (:func:`finish`), save it at the stem ``path`` in the run's active format and close it.

    Args:
        fig: The figure; closed whether or not the save succeeds.
        path: The destination stem, without an extension.
        **_: Ignored (the seam's ``tight`` / ``crop``): the layout is :func:`finish`'s and the page is saved at its
            own size.

    Returns:
        The path written.
    """
    import matplotlib.pyplot as plt
    from teb_vae.lag_attn_cfs.eval.figures_seam import figures

    dest = Path(str(path))
    dest = dest.with_name(f"{dest.name}.{figures.active_figure_format()}")
    try:
        finish(fig)
        fig.savefig(str(dest), dpi=figures.EVAL_SAVE_DPI, facecolor="white")
    finally:
        plt.close(fig)
    return dest
