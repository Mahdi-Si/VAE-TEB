r"""What the warm-up budget cost, drawn twice: per run, and as the curve that chose it.

Two figures, from one resolved budget:

* **The run-level budget figure**, via :func:`build_warmup_budget_figure` -- every declared channel
  of both streams on one anchor-relative time axis, its warm-up drawn as a bar against the shaded
  forecast window. This answers *what did this run's budget keep, and what did it drop*, and it is
  a constant of the configuration rather than of a sample, so the diagnostic callback writes it
  once per run.
* **The budget tradeoff curve**, via :func:`build_tradeoff_figure` -- kept channels, anchors and
  tiles against every candidate budget, with the shipped one marked. This is the figure that
  *justifies* the budget, and it is a constant of the **shard** rather than of the run: writing an
  identical copy into every run directory would be waste, so it is produced offline, from
  ``causal_warmup.py``'s own entry point, exactly as ``channel_reach.py`` and ``input_budget.py``
  already check their arithmetic without starting a fit.

**The panel is the shipped one, fed a different quantity.**
:func:`~teb_vae.lag_attn_rws.input_budget._budget_panel` draws one horizontal bar per declared
channel spanning $[-\Delta\delta_c,\ -\Delta\delta_c + \rho_c]$, where $\delta_c$ is a per-channel
delay in steps and $\rho_c$ a forward reach in seconds. Feeding $\delta_c = W'_c$ and
$\rho_c = \Delta W'_c$ turns that into $[-\Delta W'_c,\ 0]$: a **backward settling length**, ending
exactly at the anchor's causal endpoint. The reading is the mirror image of the two-sided figure's
and is worth stating once -- there, a bar that ends after $0$ reaches into the window it is meant to
forecast; here, a bar that starts before $0$ is how long the channel spent becoming honest, and the
budget is the vertical line no kept bar may start left of.

The figure-level function is **not** shared, because the shipped one calls ``describe_streams``
itself, which builds the production two-sided Morlet bank and refuses these channel widths. The
panel is, by object identity, so the two figures cannot come to draw a bar differently.

**Dropped channels are drawn at $\delta_c = 0$**, which is the shipped panel's own convention and
is what makes them legible: their bar runs *forward* from the anchor for as long as they are still
warming up, straight through the forecast window. That is exactly why they were dropped, stated in
the same units as the forecast.

This module may depend on ``numpy`` and ``matplotlib``: it is model-layer, like ``plotting.py``,
imported by the task's figure seam rather than by ``nets/``. It builds no filter bank -- but reusing
the shipped panel means importing the module that owns it, which pulls ``kymatio`` at module scope
and costs about three seconds. **Every consumer therefore imports this module lazily**, inside the
call that draws, so the cost is paid once per run on the rank that writes the figure rather than by
every process that resolves a warm-up budget.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
)
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import StreamWarmup, WarmupBudget  # noqa: E402
from teb_vae.lag_attn_rws.input_budget import _budget_panel  # noqa: E402
from utils.style import style_axes  # noqa: E402

__all__ = [
    "BUDGET_FIGURE_STEM",
    "TRADEOFF_FIGURE_STEM",
    "BudgetPoint",
    "build_tradeoff_figure",
    "build_warmup_budget_figure",
    "budget_tradeoff",
    "write_tradeoff_figure",
    "write_warmup_budget_figure",
]

#: Filename stem of the run-level figure. Deliberately **not** the shipped
#: ``causal_input_budget``: the two describe different guards, and a model that somehow wrote both
#: would otherwise have one silently overwrite the other in the same directory.
BUDGET_FIGURE_STEM = "causal_warmup_budget"

#: Filename stem of the offline tradeoff curve.
TRADEOFF_FIGURE_STEM = "causal_warmup_tradeoff"


def _stream_channels(stream: StreamWarmup) -> Any:
    r"""Re-express one stream's warm-up as the record the shipped budget panel draws.

    Args:
        stream: The resolved stream, carrying its declared-width warm-up vector and its survivors.

    Returns:
        A :class:`~teb_vae.lag_attn_rws.input_budget.StreamChannels` whose ``delays`` are $W'_c$ in
        steps and whose ``reach_s`` is $\Delta W'_c$ in seconds, so each kept bar spans
        $[-\Delta W'_c, 0]$.
    """
    from teb_vae.lag_attn_rws.input_budget import StreamChannels

    declared = np.asarray(stream.declared_warmup_steps, dtype=int)
    return StreamChannels(
        name=stream.name,
        block_spans=stream.block_spans,
        reach_s=SECONDS_PER_STEP * declared.astype(float),
        # All non-finite, so the shared annotation draws no secondary axis: the centre frequencies
        # of the one-sided bank are carried nowhere on the model side, and a fabricated axis would
        # label every channel wrongly -- every tick would read "$S_0$", which names the order-0
        # scattering low-pass.
        center_hz=np.full(declared.size, np.nan),
        keep_index=np.asarray(stream.keep_index, dtype=int),
        delays=np.asarray(stream.warmup_steps, dtype=int),
    )


def build_warmup_budget_figure(budget: WarmupBudget, *, horizon: int) -> Any:
    r"""Draw the run-level figure: every input channel's warm-up against the forecast window.

    Args:
        budget: The resolved warm-up budget. The **declared**-width vectors are what makes this
            figure worth drawing: it exists to show the channels the budget dropped beside the ones
            it kept, and a description carrying only the survivors could not.
        horizon: $H$ in decimated steps, which sizes the shaded forecast window.

    Returns:
        The matplotlib ``Figure``. The caller saves and closes it.
    """
    horizon_s = float(horizon) * SECONDS_PER_STEP
    streams = (budget.target, budget.source)
    described = [_stream_channels(stream) for stream in streams]
    budget_s = -SECONDS_PER_STEP * float(budget.target.max_warmup)

    # Wide enough on the left that **every kept bar starts inside it** -- the source is not gated
    # and its slowest kept channel waits twice as long as the target's, so a limit taken from the
    # budget alone would cut the source panel's bars at the axis edge and make them read as though
    # they began there. The panel counts only what runs past the *right* edge, so a left cut would
    # be the one clipping nothing reports. The right edge holds the forecast window with room; the
    # dropped channels' bars run far past it by construction, and that they cross the window at all
    # is their informative property rather than where they end.
    longest_s = SECONDS_PER_STEP * max(float(stream.max_warmup) for stream in streams)
    x_limits = (-1.08 * longest_s, max(1.5 * horizon_s, 0.12 * longest_s))

    heights = [stream.declared_width for stream in described]
    figure, axes = plt.subplots(
        len(described), 1, figsize=(12, 3.0 + 0.055 * sum(heights)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.28},
        squeeze=False,
    )
    clipped = 0
    for ax, stream, resolved in zip(axes[:, 0], described, streams):
        clipped += _budget_panel(ax, stream, horizon_s=horizon_s, x_limits=x_limits)
        # The budget itself, on the axis the bars are drawn against: no kept target bar may start
        # left of it, which is the whole content of the guard and is otherwise something a reader
        # has to measure off the figure. Drawn on the source panel too, where it is *not* the
        # guard that shaped that stream -- the budget gates no source channel; the alignment
        # reference does -- and where its being crossed is exactly the compromise section the
        # design records.
        ax.axvline(
            budget_s, color=COLOR_BLUE, linewidth=1.0, linestyle="--",
            label=f"budget $-\\Delta B$ = {budget_s:g} s",
        )
        # The shared panel titles this a *delay*, which is what it is on the two-sided figure and
        # is not what these bars measure: they are a settling length, and the region behind the
        # boundary holds real values on no defined scale rather than a zero fill. The alignment
        # does introduce a genuine per-channel delay, and this figure deliberately does not draw
        # it -- a bar that silently became W' + d would make the two figures of this family measure
        # different quantities under one caption.
        blocks = ", ".join(
            f"{name} {kept}/{declared}" for name, kept, declared in resolved.block_counts()
        )
        ax.set_title(
            f"{resolved.name} stream — {blocks}; {resolved.kept_width}/"
            f"{resolved.declared_width} channels kept; warm-up 0-{resolved.max_warmup} steps "
            f"(0-{resolved.max_warmup * SECONDS_PER_STEP:g} s)",
            fontsize=9, pad=6,
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.95)

    figure.suptitle(
        f"Causal warm-up budget — each channel's warm-up $W'_c$ drawn backwards from the anchor's\n"
        f"causal endpoint, against the {horizon_s:.0f} s forecast window (shaded). A kept bar "
        f"spans $[-\\Delta W'_c, 0]$; a dropped channel is drawn at $\\delta_c = 0$, so its bar "
        f"runs forward\nthrough the window it was still warming up for. {clipped} bar(s) run past "
        f"the right edge and are clipped.",
        fontsize=10, y=0.995, va="top",
    )
    figure.subplots_adjust(top=1.0 - 1.0 / figure.get_figheight())
    return figure


def write_warmup_budget_figure(
    budget: WarmupBudget, directory: Path, *, horizon: int, file_format: str = "pdf"
) -> Path:
    """Draw the run-level figure and save it under ``directory``.

    Args:
        budget: The resolved warm-up budget.
        directory: Where the file goes; created if absent.
        horizon: $H$ in decimated steps.
        file_format: Figure extension, without the dot.

    Returns:
        The written path.
    """
    from utils.style import SAVE_DPI, save_figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{BUDGET_FIGURE_STEM}.{file_format}"
    save_figure(
        build_warmup_budget_figure(budget, horizon=horizon), path, dpi=SAVE_DPI, close=True
    )
    return path


# =================================================================================================
# The tradeoff curve
# =================================================================================================
class BudgetPoint(NamedTuple):
    r"""One candidate budget and what it buys.

    Attributes:
        budget_steps: The threshold $B$ in decimated steps.
        kept: Target channels whose $W'_c \le B$.
        anchors: Anchors the pairing then admits, $T_{\mathrm{valid}} - F$ with $F = B' - 1$ over
            the **survivors'** own maximum $B'$ -- not over the threshold. The distinction is the
            one the shipped floor rests on: a threshold of $151$ keeps the identical channels as
            $134$ and would otherwise read $17$ steps worse.
        tiles: Tiles a training step decodes at phase $0$, $\lceil \mathrm{anchors}/S \rceil$.
    """

    budget_steps: int
    kept: int
    anchors: int
    tiles: int


def _steps_to_seconds(steps: Any) -> Any:
    """Decimated steps to seconds, for the tradeoff curve's secondary axis."""
    return SECONDS_PER_STEP * steps


def _seconds_to_steps(seconds: Any) -> Any:
    """Seconds to decimated steps, the inverse the secondary axis needs."""
    return seconds / SECONDS_PER_STEP


def budget_tradeoff(
    declared_warmup_steps: Sequence[int],
    *,
    sequence_length: int,
    horizon: int,
    anchor_stride: int,
    thresholds: Optional[Sequence[int]] = None,
) -> List[BudgetPoint]:
    r"""What each candidate warm-up budget keeps, and what it costs in supervision.

    The three quantities move against each other and the choice is where they cross: a higher
    budget keeps more channels, but the floor it forces is $F = B' - 1$ over the survivors, and
    every step of floor is a step of $[F, T_{\mathrm{valid}})$ that no anchor can start in.

    Computed from a resolved vector rather than from constants, so it re-draws correctly against a
    dataset rebuilt at another ``causal_warmup_quantile`` -- which changes both the warm-ups and the
    stored channel count.

    **The curve is drawn at the unaligned pairing** $F = B' - 1$, and a run with a channel alignment
    configured decodes exactly one anchor fewer at every threshold, because a shifted channel is
    honest at $W'_c + d_c$ and the floor must clear that too. The offset is a constant one step, so
    the curve's shape -- which is what the budget is chosen from -- is unaffected; only its level is,
    and it is stated here rather than folded in, because a curve that took the alignment as given
    could no longer price the unaligned arm.

    Args:
        declared_warmup_steps: $W'_c$ per declared target channel, in decimated steps.
        sequence_length: $T$, the trimmed window the loader serves.
        horizon: $H$ in decimated steps.
        anchor_stride: $S$, the training stride the tile count is taken at.
        thresholds: Candidate budgets. Defaults to every distinct $W'_c$, which is the only set
            where anything changes -- the curve is a staircase and its corners are the data.

    Returns:
        One :class:`BudgetPoint` per threshold, ascending. A threshold admitting no anchor at all
        is included with ``anchors`` and ``tiles`` at $0$: that region is the figure's point.
    """
    declared = np.asarray(declared_warmup_steps, dtype=int)
    candidates = (
        np.unique(declared) if thresholds is None else np.asarray(thresholds, dtype=int)
    )
    t_valid = int(sequence_length) - int(horizon)

    points: List[BudgetPoint] = []
    for threshold in candidates.tolist():
        kept = declared[declared <= threshold]
        if not kept.size:
            points.append(BudgetPoint(threshold, 0, 0, 0))
            continue
        # The survivors' own maximum, not the threshold: the pairing the constructor enforces is
        # $F \ge \max_{c \in \mathrm{kept}} W'_c - 1$, and a floor read off the threshold would sit
        # above it wherever the staircase has a gap.
        floor = int(kept.max()) - 1
        anchors = max(t_valid - floor, 0)
        points.append(
            BudgetPoint(threshold, int(kept.size), anchors, -(-anchors // int(anchor_stride)))
        )
    return points


def build_tradeoff_figure(
    points: Sequence[BudgetPoint], *, shipped_budget: Optional[int] = None
) -> Any:
    r"""Draw the curve that justifies a budget: channels, anchors and tiles against $B$.

    Args:
        points: The tradeoff, from :func:`budget_tradeoff`.
        shipped_budget: The threshold to mark, or ``None`` to mark none.

    Returns:
        The matplotlib ``Figure``. The caller saves and closes it.
    """
    thresholds = np.array([point.budget_steps for point in points], dtype=float)
    figure, ax = plt.subplots(figsize=(10, 5.0))

    # The infeasible region first, so the curves are drawn over it: a budget whose floor leaves no
    # anchor is not a worse choice, it is not a choice.
    infeasible = [point.budget_steps for point in points if point.tiles == 0]
    if infeasible:
        ax.axvspan(
            min(infeasible) - 0.5, thresholds.max() + 0.5,
            color=COLOR_GRAY, alpha=0.18, zorder=0, label="no tile fits",
        )

    for values, colour, label in (
        ([point.kept for point in points], COLOR_ORANGE, "target channels kept"),
        ([point.anchors for point in points], COLOR_VERMILLION, "anchors admitted"),
        ([point.tiles for point in points], COLOR_BLUE, "tiles per sample at $\\varphi=0$"),
    ):
        ax.step(thresholds, values, where="post", color=colour, linewidth=1.2, label=label)

    if shipped_budget is not None:
        marked = [point for point in points if point.budget_steps == int(shipped_budget)]
        ax.axvline(float(shipped_budget), color=COLOR_BLACK, linewidth=1.0, linestyle="--")
        if marked:
            point = marked[0]
            ax.annotate(
                f"$B$={point.budget_steps}: {point.kept} ch, {point.anchors} anchors, "
                f"{point.tiles} tiles",
                xy=(float(point.budget_steps), float(point.kept)),
                xytext=(6, 8), textcoords="offset points", fontsize=8, color=COLOR_BLACK,
            )

    ax.set_xlabel("Warm-up budget $B$ (decimated steps)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(
        "Warm-up budget tradeoff — channels kept against supervision bought", fontsize=10, pad=8
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    style_axes(ax, grid="both")
    # Seconds beside steps, because the budget is argued about in minutes of recording and read off
    # a config in steps.
    secondary = ax.secondary_xaxis("top", functions=(_steps_to_seconds, _seconds_to_steps))
    secondary.set_xlabel("$\\Delta B$ (s)", fontsize=8)
    return figure


def write_tradeoff_figure(
    points: Sequence[BudgetPoint],
    directory: Path,
    *,
    shipped_budget: Optional[int] = None,
    file_format: str = "pdf",
) -> Path:
    """Draw the tradeoff curve and save it under ``directory``.

    Args:
        points: The tradeoff, from :func:`budget_tradeoff`.
        directory: Where the file goes; created if absent.
        shipped_budget: The threshold to mark.
        file_format: Figure extension, without the dot.

    Returns:
        The written path.
    """
    from utils.style import SAVE_DPI, save_figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{TRADEOFF_FIGURE_STEM}.{file_format}"
    save_figure(
        build_tradeoff_figure(points, shipped_budget=shipped_budget),
        path,
        dpi=SAVE_DPI,
        close=True,
    )
    return path


