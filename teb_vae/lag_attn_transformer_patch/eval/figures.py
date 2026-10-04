r"""Figure helpers the patch analyses share (house conventions: log scales only for magnitudes that span
decades; one segment's rows stacked on the samples-page time axis). Drawing primitives stay in
``teb_vae/lag_attn_cfs/eval/{figures_seam,attributions}.py``.

* :func:`spans_decades` -- do the finite magnitudes span ≥ ``decades`` decades (then use symlog/log)?
* :func:`share_axis` -- a positive log axis for non-negative shares, plus the compact legend.
* :func:`stacked_page` -- one anchor of one segment on the samples-page layout: raw FHR and UP rows,
  raw-resolution ``(T, R)`` maps on the same time axis, then free rows off it.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels


def spans_decades(*values: Any, decades: float = 2.0) -> bool:
    """True when the finite non-zero magnitudes span ``decades`` (max over median ``|x|``)."""
    stacked = np.concatenate([np.abs(np.asarray(v, dtype=np.float64)).ravel() for v in values])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    return bool(stacked.size) and float(stacked.max()) > 10.0 ** decades * float(np.median(stacked))


def share_axis(ax: Any, *series: Any, ncol: int = 2, legend: bool = True) -> None:
    """A positive log axis floored ``core.LOG_DECADES`` below the drawn maximum, a decade of headroom.

    For non-negative shares: a plain log axis, not the symmetric-log one, which would draw a negative
    region no share can reach.
    """
    values = np.concatenate([np.ravel(np.asarray(s, dtype=np.float64)) for s in series]) if series else np.zeros(0)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size:
        top = float(values.max())
        ax.set_yscale("log")
        ax.set_ylim(top * 10.0 ** (-core.LOG_DECADES), top * 10.0)
    if legend:
        ax.legend(loc="upper left", ncol=int(ncol), fontsize=figures.FONT_TINY)


def stacked_page(
    item: Mapping[str, Any],
    *,
    map_rows: Sequence[Tuple[Any, str, np.ndarray, str]],
    lower: Sequence[Tuple[str, Callable[[Any, Any], None]]],
    norms: Mapping[Any, Any],
    note: str,
) -> Any:
    """One anchor of one segment on the samples-page layout: raw rows, raw-resolution maps, then free rows.

    Args:
        item: ``anchor``, ``horizon``, ``steps``, ``guid``, the class (and subgroup) columns and the raw
            signals ``core._raw_row`` draws (``raw``, ``raw_units``).
        map_rows: ``(norm key, title, (T, R) field with unread samples NaN, colour-bar unit)``, drawn by
            the shared ``_stream_map`` (symmetric-log, unread cells blank, anchor ruled, block shaded).
        lower: ``(title, draw(ax, cax))`` rows off the time axis.
        norms: Colour norms by map key (shared across pages of one analysis).
        note: The caveat line under the page.
    """
    rows = [("raw_fhr", core.EXAMPLE_RAW_ROW), ("raw_up", core.EXAMPLE_RAW_ROW)]
    rows += [(key, core.EXAMPLE_MAP_ROW) for key, *_ in map_rows] + [(title, core.EXAMPLE_LAG_ROW) for title, _ in lower]
    heights = [height for _, height in rows]
    height = sum(heights) * core.EXAMPLE_ROW_INCHES
    figure = plt.figure(figsize=(core.EXAMPLE_PAGE_WIDTH, height))
    bottom = max(0.02, 0.35 / height) + figures.caveat_note(figure, note)
    grid = GridSpec(len(rows), 2, figure=figure, height_ratios=heights, width_ratios=[1.0, 0.022], left=0.065,
                    right=0.93, top=1.0 - core.EXAMPLE_HEADER_INCHES / height, bottom=bottom, hspace=0.55, wspace=0.09)
    anchor, horizon, steps = int(item["anchor"]), int(item["horizon"]), int(item["steps"])
    first = None
    n_time = 2 + len(map_rows)
    for position in range(n_time):
        ax = figure.add_subplot(grid[position, 0], sharex=first)
        first = first or ax
        cax = figure.add_subplot(grid[position, 1])
        if position < 2:
            core._raw_row(ax, item, "fhr" if position == 0 else "up", steps=steps)
            cax.set_axis_off()
        else:
            key, title, field, unit = map_rows[position - 2]
            core._stream_map(figure, ax, field, title=title, anchor=anchor, live=None, kind="signed", cax=cax,
                             horizon=horizon, norm=norms.get(key), colorbar_label=unit)
            ax.set_ylabel("sample in patch")
        if position < n_time - 1:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
    for offset, (title, draw) in enumerate(lower):
        ax = figure.add_subplot(grid[n_time + offset, 0])
        cax = figure.add_subplot(grid[n_time + offset, 1])
        draw(ax, cax)
        ax.set_title(title)
    subgroup = item.get(labels.SUBGROUP_COLUMN)
    figure.suptitle(f"guid {item['guid']}" + (f", subgroup {subgroup}" if subgroup is not None else "")
                    + f", class {item[labels.CLASS_COLUMN]}, anchor {anchor} (stored step)",
                    fontsize=figures.FONT_NOTE, y=1.0 - 0.3 * core.EXAMPLE_HEADER_INCHES / height)
    figures.mark_laid_out(figure)
    return figure


__all__ = ["share_axis", "spans_decades", "stacked_page"]
