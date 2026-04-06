"""Publication-quality plotting style for the transformer testing pipeline.

Extracted from ``model/transformer/training/plotting_callback.py`` to ensure
visual consistency between training diagnostics and test-time analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Colour palette (matches training/plotting_callback.py)
# ---------------------------------------------------------------------------
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

SAVE_DPI = 600  # Higher than training (300) for publication quality

# Full palette for automatic class colour assignment
_PALETTE = [
    COLOR_BLUE, COLOR_ORANGE, COLOR_VERMILLION, COLOR_GREEN,
    COLOR_SKY, COLOR_PURPLE, COLOR_SAGE, COLOR_GRAY,
]

# Default mapping for known classes
CLASS_COLORS_DEFAULT: Dict[str, str] = {
    "healthy": COLOR_BLUE,
    "acidosis": COLOR_ORANGE,
    "hie": COLOR_VERMILLION,
}

# Head colours (consistent across all forecast plots)
HEAD_COLORS: Dict[str, str] = {
    "self": COLOR_BLUE,
    "fused": COLOR_ORANGE,
    "te": COLOR_GREEN,
}

HEAD_LABELS: Dict[str, str] = {
    "self": "Self-only",
    "fused": "Fused",
    "te": "TE-augmented",
}


def get_class_colors(class_names: Sequence[str]) -> Dict[str, str]:
    """Return a colour mapping for any set of class names.

    Uses ``CLASS_COLORS_DEFAULT`` where available, assigns from ``_PALETTE``
    for unknown class names so the pipeline works with arbitrary keys.

    Args:
        class_names: Sequence of class name strings.

    Returns:
        Dictionary mapping each class name to a hex colour string.
    """
    colors: Dict[str, str] = {}
    used = set()
    for name in class_names:
        if name in CLASS_COLORS_DEFAULT:
            colors[name] = CLASS_COLORS_DEFAULT[name]
            used.add(CLASS_COLORS_DEFAULT[name])

    palette_idx = 0
    for name in class_names:
        if name not in colors:
            while palette_idx < len(_PALETTE) and _PALETTE[palette_idx] in used:
                palette_idx += 1
            if palette_idx < len(_PALETTE):
                colors[name] = _PALETTE[palette_idx]
                used.add(_PALETTE[palette_idx])
                palette_idx += 1
            else:
                colors[name] = COLOR_GRAY
    return colors


# ---------------------------------------------------------------------------
# Style helpers (identical to plotting_callback.py for consistency)
# ---------------------------------------------------------------------------

def apply_publication_style() -> None:
    """Apply publication-quality matplotlib style."""
    if not HAS_MPL:
        return
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "serif",
        "font.serif": [
            "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif",
        ],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLOR_BLACK,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.3,
        "grid.color": COLOR_LIGHT_GRAY,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": COLOR_GRAY,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def style_axes(ax: Any, *, grid: str = "major") -> None:
    """Apply clean styling with all four spines visible.

    Args:
        ax: Matplotlib Axes object.
        grid: One of ``"major"``, ``"both"``, or ``"none"``.
    """
    ax.set_axisbelow(True)
    if grid in ("both", "major"):
        ax.grid(True, linestyle="-", alpha=0.4, linewidth=0.4,
                color=COLOR_LIGHT_GRAY)
    if grid == "both":
        ax.grid(True, which="minor", linestyle=":", alpha=0.25,
                linewidth=0.3, color=COLOR_LIGHT_GRAY)
        ax.minorticks_on()
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)


def add_colorbar(fig: Any, mappable: Any, ax: Any, *,
                 label: Optional[str] = None) -> Any:
    """Attach an aligned colorbar to *ax*.

    Args:
        fig: Matplotlib Figure.
        mappable: The image/mappable returned by imshow, etc.
        ax: Axes to attach the colorbar to.
        label: Optional colorbar label text.

    Returns:
        The colorbar object.
    """
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=8, color=COLOR_BLACK)
    cbar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
    return cbar


def heatmap(ax: Any, data: np.ndarray, *, cmap: str = "bwr",
            origin: str = "upper", title: str = "", ylabel: str = "Channel",
            xlabel: str = "Time Steps", label: str = "Value",
            fig: Any = None) -> Any:
    """Draw a coefficient heatmap on *ax*.

    Args:
        ax: Matplotlib Axes.
        data: 2-D array to visualise.
        cmap: Colormap name.
        origin: ``"upper"`` or ``"lower"``.
        title: Axes title.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        label: Colorbar label.
        fig: If provided, a colorbar is attached via :func:`add_colorbar`.

    Returns:
        The ``AxesImage`` returned by ``imshow``.
    """
    vabs = np.nanmax(np.abs(data)) or 1.0
    im = ax.imshow(data, aspect="auto", cmap=cmap, origin=origin,
                   vmin=-vabs, vmax=vabs)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(False)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)
    if fig is not None:
        add_colorbar(fig, im, ax, label=label)
    return im


def save_figure(fig: Any, path: Any, *, dpi: Optional[int] = None,
                close: bool = True) -> None:
    """Save a figure and optionally close it.

    Args:
        fig: Matplotlib Figure.
        path: Output file path.
        dpi: Override DPI (defaults to ``SAVE_DPI``).
        close: Whether to close the figure after saving.
    """
    fig.savefig(str(path), dpi=dpi or SAVE_DPI, bbox_inches="tight",
                pad_inches=0.05)
    if close:
        plt.close(fig)
