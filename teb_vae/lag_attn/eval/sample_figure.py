r"""The per-sample multi-row diagnostic figure.

One page per recording, every row on the same physical-time x-axis in seconds, so a vertical
line cuts every panel at the same instant and a feature seen in the attention can be traced
straight down into the forecast. This is the figure a human actually reads when asking "what
did the model do on *this* recording", as opposed to the aggregate distributions the other
analyses emit.

Three things about the layout are load-bearing and none is obvious from a call site.

**The colorbar lives in its own gridspec column.** Attaching a colorbar to an axes steals width
from that axes, so a page mixing heatmap rows with line-plot rows ends up with two different
main-axes widths -- and the shared time axis, which is the whole point, stops lining up. A
two-column ``GridSpec`` reserves a narrow fixed-width slot in column 1 for every row; rows that
draw no colorbar hide theirs rather than omitting it, so column 0 is exactly as wide on every
row.

**Rows are declared, not drawn inline.** :data:`ROW_SPECS` is the single source for which rows
exist, in what order, at what height, and under what title prefix. A row whose data the batch
does not carry is dropped from that list *before* the gridspec is built, so the figure shrinks
by a row instead of carrying an empty panel -- and the row count, the titles and the height
ratios cannot disagree, because all three are read off the same list.

**A missing raw trace is expected, not exceptional.** ``fhr`` and ``up`` are optional
``load_fields``, and a config that omits them is a legitimate way to make the loader cheaper.
The figure degrades to one fewer row.

Every heatmap here draws with ``interpolation='none'`` rather than ``'nearest'``, which is both
faster and more truthful. In a vector backend ``'none'`` embeds the array unresampled, so one
data cell is one cell; ``'nearest'`` resamples to the axes' pixel size at ``SAVE_DPI``, which on
a page this wide means rasterising a $(c_y, T)$ field into tens of megapixels. Measured on the
production geometry that is $9.3$ s and $1.1$ MB per page against $3.7$ s and $290$ KB -- and
the resampled version can merge adjacent channels, which on a per-channel diagnostic is the one
artifact the row exists to rule out.

The figure builders here take numpy arrays and plain numbers rather than a batch and a model,
so the module has no dependency on the runner and the tests can drive it with synthetic
tensors of the right shape.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.gridspec import GridSpec

from teb_vae.lag_attn.eval.figures import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_LIGHT_GRAY,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_VERMILLION,
    attach_lag_seconds_axis,
    plt,
    safe_vabs,
    shade_warmup,
    style_axes,
    to_numpy,
)
from teb_vae.lag_attn.figure_primitives import (
    average_forecast_per_channel,
    kld_per_dim_np,
    time_axes,
)

#: Raw sampling rate of the ``fhr`` / ``up`` traces, in Hz. The decimated feature grid is
#: derived from it and the raw length, so a batch without raw fields falls back to the nominal
#: 16x decimation rather than losing its physical axis entirely.
FS_RAW = 4.0

#: Nominal decimation factor, used only to synthesise a time axis when the raw trace is absent.
DECIMATION = 16

#: Every row the figure can draw: ``(name, title_prefix, height_ratio)``, in page order.
#:
#: The single source for the row set. :func:`resolve_rows` filters it by what the batch carries;
#: the count, the titles and the height ratios are then all read off the filtered result, so
#: they cannot drift apart the way three parallel literals would.
ROW_SPECS: Tuple[Tuple[str, str, float], ...] = (
    ("raw", "Raw FHR / UP", 0.9),
    ("forecast", "Forecast", 1.4),
    ("truth", "Target", 1.4),
    ("residual", "Forecast residual", 1.4),
    ("latent", "Latent z", 1.0),
    ("kld_dims", "Per-dimension KL", 1.15),
    ("kld_total", "K_t", 0.9),
    ("attention", "Lag attention", 1.15),
    ("te_lag", "TE lag attribution", 1.15),
)

#: Rows whose data comes from an optional batch field. Everything else is produced by the
#: forward pass and is therefore always available.
OPTIONAL_ROWS: Dict[str, Tuple[str, ...]] = {"raw": ("fhr", "up")}


def resolve_rows(available: Sequence[str]) -> List[Tuple[str, str, float]]:
    """Return the row specs the given fields can actually fill, in page order.

    Args:
        available: Names of the optional inputs present -- typically ``('fhr', 'up')`` or
            nothing at all.

    Returns:
        The subset of :data:`ROW_SPECS` to draw. A row whose optional inputs are not all
        present is dropped rather than drawn empty.
    """
    present = set(available)
    return [
        spec
        for spec in ROW_SPECS
        if all(field in present for field in OPTIONAL_ROWS.get(spec[0], ()))
    ]


class RowGrid:
    """A figure whose rows share one physical-time axis and one main-axes width.

    Built from a resolved row list. Each row gets a ``(main, cax)`` pair; the ``cax`` is the
    reserved colorbar slot, which a line-plot row hides so its main axes stays exactly as wide
    as a heatmap row's.
    """

    def __init__(
        self,
        rows: Sequence[Tuple[str, str, float]],
        t_max: float,
        *,
        width: float = 14.0,
        height_per_row: float = 2.6,
    ) -> None:
        """Create the figure and its gridspec.

        Args:
            rows: The resolved row specs, ``(name, title_prefix, height_ratio)``.
            t_max: Full x-axis extent in seconds; every row is limited to ``(0, t_max)``.
            width: Figure width in inches.
            height_per_row: Height in inches per unit of height ratio.
        """
        self.rows = list(rows)
        self.t_max = float(t_max)
        self.names = [name for name, _, _ in self.rows]
        self.titles: Dict[str, str] = {name: title for name, title, _ in self.rows}
        ratios = [ratio for _, _, ratio in self.rows]

        self.figure = plt.figure(figsize=(float(width), float(height_per_row) * sum(ratios)))
        # Column 1 is the reserved colorbar slot -- see the module docstring. Its width is a
        # fraction of the figure, not of the row, so it is identical on every row.
        self.grid = GridSpec(
            len(self.rows), 2, figure=self.figure,
            height_ratios=ratios, width_ratios=[1.0, 0.022],
            left=0.065, right=0.93, top=0.97, bottom=0.03,
            hspace=0.55, wspace=0.03,
        )
        self._index = {name: position for position, name in enumerate(self.names)}
        self._axes: Dict[str, Any] = {}

    def has(self, name: str) -> bool:
        """Whether ``name`` is one of the rows this figure is drawing."""
        return name in self._index

    def axes(self, name: str) -> Tuple[Any, Any]:
        """Return the ``(main, cax)`` pair for a row, creating it on first use.

        Args:
            name: Row name from :data:`ROW_SPECS`.

        Returns:
            The main axes and its reserved colorbar axes.

        Raises:
            KeyError: If ``name`` is not a row of this figure -- which means the caller is
                drawing a row that :func:`resolve_rows` dropped, and would otherwise silently
                draw nothing.
        """
        if name in self._axes:
            return self._axes[name]
        position = self._index[name]
        pair = (
            self.figure.add_subplot(self.grid[position, 0]),
            self.figure.add_subplot(self.grid[position, 1]),
        )
        self._axes[name] = pair
        return pair

    def main_axes(self) -> List[Any]:
        """Every main axes created so far, in page order."""
        return [self._axes[name][0] for name in self.names if name in self._axes]

    def colorbar(self, cax: Any, image: Any, label: str) -> Any:
        """Attach a colorbar onto a row's reserved slot.

        Args:
            cax: The reserved colorbar axes.
            image: The image handle to describe.
            label: Colorbar label.

        Returns:
            The colorbar.
        """
        bar = self.figure.colorbar(image, cax=cax)
        bar.set_label(label, fontsize=8, color=COLOR_BLACK)
        bar.ax.tick_params(labelsize=7, colors=COLOR_BLACK)
        if bar.outline is not None:
            bar.outline.set_linewidth(0.6)
            bar.outline.set_edgecolor(COLOR_LIGHT_GRAY)
        return bar

    @staticmethod
    def hide_colorbar(cax: Any) -> None:
        """Hide an unused colorbar slot, keeping the main-axes width unchanged."""
        cax.set_visible(False)

    def finalise(self, ax: Any, name: str, detail: str = "", *, warmup: int = 0, T: int = 0) -> None:
        """Apply the shared time axis, the title and the warm-up shading to one row.

        Called at the end of every row so the alignment cannot be forgotten on one of them.

        Args:
            ax: The row's main axes.
            name: Row name, supplying the title prefix.
            detail: Appended to the title prefix after an em dash.
            warmup: Warm-up length in decimated steps, shaded when positive.
            T: Total decimated steps, for the step-to-seconds conversion.
        """
        title = self.titles.get(name, name)
        ax.set_title(f"{title} — {detail}" if detail else title, fontsize=9, pad=6)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_xlim(0.0, self.t_max)
        if warmup and T:
            shade_warmup(ax, warmup, self.t_max, T)


def _style_heatmap(ax: Any) -> None:
    """Draw all four spines on a heatmap axes and drop its grid."""
    ax.grid(False)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(COLOR_BLACK)
        ax.spines[spine].set_linewidth(0.6)


def _channel_heatmap(
    grid: RowGrid,
    name: str,
    field: np.ndarray,
    *,
    detail: str,
    colorbar_label: str,
    warmup: int,
    limit: Optional[float] = None,
    separator: Optional[int] = None,
    cmap: str = "bwr",
) -> None:
    r"""Draw one $(\mathrm{channel}, T)$ heatmap row on the shared time axis.

    Args:
        grid: The row grid.
        name: Row name.
        field: The field, $(\mathrm{rows}, T)$.
        detail: Title detail after the prefix.
        colorbar_label: Colorbar label.
        warmup: Warm-up length in decimated steps.
        limit: Symmetric colour limit. ``None`` derives one from the field, which is right for
            a single panel; a caller drawing two directly comparable panels passes the *same*
            limit to both, because two independently scaled heatmaps of the same quantity read
            as more similar than they are.
        separator: Row index of the last row of the upper feature block, drawn as a boundary
            line at ``separator + 0.5``.
        cmap: Colormap name.
    """
    ax, cax = grid.axes(name)
    n_rows, n_steps = field.shape
    vabs = safe_vabs(field) if limit is None else float(limit)
    image = ax.imshow(
        field, aspect="auto", cmap=cmap, origin="upper", vmin=-vabs, vmax=vabs,
        extent=[0.0, grid.t_max, n_rows - 0.5, -0.5], interpolation="none",
    )
    if separator is not None:
        ax.axhline(float(separator) + 0.5, color="white", linewidth=1.2, linestyle="--")
    ax.set_ylabel("Feature channel", fontsize=8)
    _style_heatmap(ax)
    grid.colorbar(cax, image, colorbar_label)
    grid.finalise(ax, name, detail, warmup=warmup, T=n_steps)


def build_sample_figure(
    *,
    outputs: Dict[str, Any],
    y_st: Any,
    y_ph: Any,
    fhr_raw: Optional[Any] = None,
    up_raw: Optional[Any] = None,
    warmup: int,
    horizon: int,
    guid: str = "unknown",
    epoch: Optional[int] = None,
    step_seconds: float = 4.0,
    up_shift_secs: float = 0.0,
    te_lag_label: str = "attribution",
) -> Any:
    r"""Build the multi-row diagnostic page for one sample.

    Every argument is already sliced to a single sample: ``outputs`` holds that sample's forward
    tensors without a batch dimension, and ``y_st`` / ``y_ph`` are $(T, \cdot)$.

    Args:
        outputs: One sample's forward outputs. Reads ``mu_full``, ``z``, ``mu_prior``,
            ``logvar_prior``, ``mu_post``, ``logvar_post``, ``kld_per_t``, ``attn_weights`` and
            ``te_lag_map``.
        y_st: Target scattering features, $(T, 43)$.
        y_ph: Target phase-harmonic features, $(T, 66)$.
        fhr_raw: Raw FHR trace $(R,)$, or ``None``. Absent drops the raw row.
        up_raw: Raw UP trace $(R,)$, or ``None``.
        warmup: Warm-up length $T_w$ in decimated steps.
        horizon: Forecast horizon $H_d$ in decimated steps.
        guid: Record identifier, for the page title.
        epoch: The recording's epoch, for the page title.
        step_seconds: Decimated step duration, for the lag second-axis.
        up_shift_secs: Dataset UP shift, for the lag second-axis.
        te_lag_label: ``'attribution'`` when ``head_structured_latent`` makes the TE lag map a
            rigorous attribution, ``'diagnostic'`` otherwise. Stated in the row title because a
            reader cannot tell the two apart from the picture.

    Returns:
        The figure. The caller saves and closes it.
    """
    y_st_np = np.asarray(to_numpy(y_st), dtype=np.float64)
    y_ph_np = np.asarray(to_numpy(y_ph), dtype=np.float64)
    T = int(y_st_np.shape[0])
    n_scattering = int(y_st_np.shape[-1])
    n_channels = n_scattering + int(y_ph_np.shape[-1])

    has_raw = fhr_raw is not None and up_raw is not None
    if has_raw:
        raw_fhr = np.asarray(to_numpy(fhr_raw), dtype=np.float64).ravel()
        raw_up = np.asarray(to_numpy(up_raw), dtype=np.float64).ravel()
        n_raw = int(raw_fhr.shape[0])
    else:
        # Without the raw trace the physical axis is still well defined via the nominal
        # decimation, so every other row keeps its seconds axis rather than falling back to
        # step indices and becoming incomparable with a run that did load the raw fields.
        raw_fhr = raw_up = np.zeros(0)
        n_raw = T * DECIMATION
    time_raw, time_dec, t_max = time_axes(T, n_raw, fs_raw=FS_RAW)

    rows = resolve_rows(("fhr", "up") if has_raw else ())
    grid = RowGrid(rows, t_max)

    # The figure exists from here on, so every exit below has to go through a close: a builder
    # that raises mid-page leaves its caller no handle to reclaim it, and pyplot keeps every
    # unclosed figure alive for the process.
    try:
        # ---- Raw FHR / UP context ------------------------------------------------
        if grid.has("raw"):
            ax, cax = grid.axes("raw")
            ax.plot(time_raw, raw_fhr, color=COLOR_BLUE, linewidth=0.8, label="FHR")
            twin = ax.twinx()
            twin.plot(time_raw, raw_up, color=COLOR_GREEN, linewidth=0.8, label="UP")
            ax.set_ylabel("FHR (normalised)", fontsize=8, color=COLOR_BLUE)
            ax.tick_params(axis="y", labelcolor=COLOR_BLUE)
            twin.set_ylabel("UP (normalised)", fontsize=8, color=COLOR_GREEN)
            twin.tick_params(axis="y", labelcolor=COLOR_GREEN)
            handles = ax.get_legend_handles_labels()
            twin_handles = twin.get_legend_handles_labels()
            ax.legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1],
                      loc="upper right", fontsize=7, framealpha=0.95)
            style_axes(ax, grid="both")
            grid.finalise(ax, "raw", "signals as loaded", warmup=warmup, T=T)
            twin.set_xlim(0.0, t_max)
            grid.hide_colorbar(cax)

        # ---- Forecast, truth, residual -------------------------------------------
        # The overlap-averaged rendering, so a per-anchor horizon tensor becomes something that can
        # share the recording's own time axis with every other row.
        forecast = average_forecast_per_channel(
            np.asarray(to_numpy(outputs["mu_full"]), dtype=np.float32), T, int(horizon), int(warmup)
        ).astype(np.float64)
        truth = np.concatenate([y_st_np, y_ph_np], axis=-1)
        residual = forecast - truth

        # One limit across the forecast and target rows: they are the same quantity, and scaling
        # them independently would make a badly-scaled forecast look like a well-scaled one.
        paired = np.concatenate([forecast[np.isfinite(forecast)], truth[np.isfinite(truth)]])
        shared_limit = safe_vabs(paired) if paired.size else 1.0

        separator = n_scattering - 1 if 0 < n_scattering < n_channels else None
        _channel_heatmap(
            grid, "forecast", forecast.T, warmup=warmup, limit=shared_limit, separator=separator,
            detail=f"overlap-averaged $\\mu_{{\\mathrm{{full}}}}$, {n_channels} channels, $H_d$={horizon}",
            colorbar_label="value",
        )
        _channel_heatmap(
            grid, "truth", truth.T, warmup=warmup, limit=shared_limit, separator=separator,
            detail=f"$Y$, {n_channels} channels (scattering above row {n_scattering}, phase below)",
            colorbar_label="value",
        )
        _channel_heatmap(
            grid, "residual", residual.T, warmup=warmup, separator=separator,
            detail="$\\mu_{\\mathrm{full}} - Y$, own colour range",
            colorbar_label="residual",
        )

        # ---- Latent z ------------------------------------------------------------
        latent = np.asarray(to_numpy(outputs["z"]), dtype=np.float64)
        d_z = int(latent.shape[-1])
        ax, cax = grid.axes("latent")
        vabs_z = safe_vabs(latent.T)
        image = ax.imshow(
            latent.T, aspect="auto", cmap="bwr", origin="lower", vmin=-vabs_z, vmax=vabs_z,
            extent=[0.0, t_max, -0.5, d_z - 0.5], interpolation="none",
        )
        ax.set_ylabel("Latent dim", fontsize=8)
        _style_heatmap(ax)
        grid.colorbar(cax, image, "z")
        grid.finalise(ax, "latent", f"$d_z$={d_z}, one seeded draw", warmup=warmup, T=T)

        # ---- Per-dimension KL ----------------------------------------------------
        kld_per_dim = kld_per_dim_np(
            np.asarray(to_numpy(outputs["mu_prior"]), dtype=np.float64),
            np.asarray(to_numpy(outputs["logvar_prior"]), dtype=np.float64),
            np.asarray(to_numpy(outputs["mu_post"]), dtype=np.float64),
            np.asarray(to_numpy(outputs["logvar_post"]), dtype=np.float64),
        )
        kld_image = np.where(np.isfinite(kld_per_dim), kld_per_dim, 0.0).T
        kld_max = float(kld_image.max()) if kld_image.size else 0.0
        ax, cax = grid.axes("kld_dims")
        image = ax.imshow(
            kld_image, aspect="auto", cmap="magma", origin="lower",
            vmin=0.0, vmax=kld_max if kld_max > 0.0 else 1.0,
            extent=[0.0, t_max, -0.5, d_z - 0.5], interpolation="none",
        )
        ax.set_ylabel("Latent dim", fontsize=8)
        _style_heatmap(ax)
        grid.colorbar(cax, image, "KL (nats)")
        grid.finalise(ax, "kld_dims", f"max {kld_max:.3g} nats", warmup=warmup, T=T)

        # ---- Total K_t, with the attention entropy alongside ---------------------
        kld_per_t = np.asarray(to_numpy(outputs["kld_per_t"]), dtype=np.float64).ravel()
        alpha = np.asarray(to_numpy(outputs["attn_weights"]), dtype=np.float64)
        mean_alpha = alpha.mean(axis=1)
        n_lags = int(mean_alpha.shape[-1])
        entropy = -(mean_alpha * np.log(mean_alpha + 1e-12)).sum(axis=-1)

        ax, cax = grid.axes("kld_total")
        ax.plot(time_dec, kld_per_t, color=COLOR_PURPLE, linewidth=1.0, label="$K_t$")
        ax.set_ylabel("KL (nats)", fontsize=8, color=COLOR_PURPLE)
        ax.tick_params(axis="y", labelcolor=COLOR_PURPLE)
        twin = ax.twinx()
        twin.plot(time_dec, entropy, color=COLOR_ORANGE, linewidth=0.9, alpha=0.85,
                  label="attention entropy")
        twin.set_ylabel("Entropy (nats)", fontsize=8, color=COLOR_ORANGE)
        twin.tick_params(axis="y", labelcolor=COLOR_ORANGE)
        handles = ax.get_legend_handles_labels()
        twin_handles = twin.get_legend_handles_labels()
        ax.legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1],
                  loc="upper right", fontsize=7, framealpha=0.95)
        style_axes(ax, grid="both")
        grid.finalise(ax, "kld_total", "per-step KL against attention sharpness", warmup=warmup, T=T)
        twin.set_xlim(0.0, t_max)
        grid.hide_colorbar(cax)

        # ---- Lag attention -------------------------------------------------------
        ax, cax = grid.axes("attention")
        image = ax.imshow(
            mean_alpha.T, aspect="auto", cmap="viridis", origin="lower",
            extent=[0.0, t_max, -0.5, n_lags - 0.5], interpolation="none",
        )
        ax.plot(time_dec, mean_alpha.argmax(axis=-1), color=COLOR_VERMILLION, linewidth=0.9,
                alpha=0.9, label="argmax lag")
        ax.set_ylabel(r"Lag $\ell$ (0 = current)", fontsize=8)
        # Negated, matching every other lag figure. ``attach_lag_seconds_axis`` maps
        # $\ell \mapsto s\ell + d$, while the pipeline's convention -- ``metrics.lag_to_seconds`` --
        # is $s\ell - \Delta_{UP}$, because recovering the delay in the original recording means
        # *undoing* the shift the dataset applied. Passing the shift through un-negated would label
        # the same lag 40 s differently here than in ``attention/attention_heatmaps.pdf``.
        attach_lag_seconds_axis(ax, step_seconds, -float(up_shift_secs))
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
        _style_heatmap(ax)
        grid.colorbar(cax, image, "attn prob")
        grid.finalise(
            ax, "attention", f"mean over {int(alpha.shape[1])} heads, $L$={n_lags}",
            warmup=warmup, T=T,
        )

        # ---- TE lag attribution --------------------------------------------------
        # Column-normalised: the per-step KL varies over orders of magnitude across a recording, so
        # a raw map is dominated by a few bright columns and the lag *selection* -- which is what
        # this row exists to show -- is invisible everywhere else. Columns whose KL is effectively
        # zero are left NaN so imshow draws them blank rather than amplifying rounding noise into a
        # confident-looking pattern.
        te_map = np.asarray(to_numpy(outputs["te_lag_map"]), dtype=np.float64).T
        te_map = np.where(np.isfinite(te_map) & (te_map > 0.0), te_map, 0.0)
        te_max = float(te_map.max()) if te_map.size else 0.0
        column_max = te_map.max(axis=0, keepdims=True)
        valid = column_max > max(1e-12, te_max * 1e-6)
        te_norm = np.where(valid, te_map / np.where(valid, column_max, 1.0), np.nan)

        ax, cax = grid.axes("te_lag")
        image = ax.imshow(
            te_norm, aspect="auto", cmap="viridis", origin="lower", vmin=0.0, vmax=1.0,
            extent=[0.0, t_max, -0.5, n_lags - 0.5], interpolation="none",
        )
        ax.set_ylabel(r"Lag $\ell$ (0 = current)", fontsize=8)
        # Negated, matching every other lag figure. ``attach_lag_seconds_axis`` maps
        # $\ell \mapsto s\ell + d$, while the pipeline's convention -- ``metrics.lag_to_seconds`` --
        # is $s\ell - \Delta_{UP}$, because recovering the delay in the original recording means
        # *undoing* the shift the dataset applied. Passing the shift through un-negated would label
        # the same lag 40 s differently here than in ``attention/attention_heatmaps.pdf``.
        attach_lag_seconds_axis(ax, step_seconds, -float(up_shift_secs))
        _style_heatmap(ax)
        grid.colorbar(cax, image, "column-norm")
        grid.finalise(
            ax, "te_lag",
            f"{te_lag_label}, column-normalised (max {te_max:.3g} nats)",
            warmup=warmup, T=T,
        )

        heading = f"guid {guid}" if epoch is None else f"guid {guid}, epoch {epoch}"
        grid.figure.suptitle(
            f"Sample diagnostics — {heading}", fontsize=11, color=COLOR_GRAY, y=0.995
        )
        return grid.figure
    except BaseException:
        plt.close(grid.figure)
        raise
