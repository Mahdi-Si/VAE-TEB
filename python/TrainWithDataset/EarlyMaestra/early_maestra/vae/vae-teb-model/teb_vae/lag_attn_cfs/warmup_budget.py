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

**And one table, from the same resolved budget**, via :func:`source_reference_tradeoff` -- what
each candidate clock for the *source* stream buys and costs. The budget figures above price the
warm-up guard; this prices the **alignment reference**, which is a different decision made against
the same vectors: a reference is a clock, every source channel slower than it is dropped rather
than advanced, and the reference itself is how stale the freshest surviving source channel is at
the anchor. The third column is the one neither figure can draw -- where a physiological
uterine-activity-to-heart-rate delay lands on the lag axis, which follows from the inter-stream
offset and is therefore a property of the *pair* of references rather than of either. It is a
table rather than a figure because its three quantities are counts, one duration and a pair of lag
bounds, and the decision it feeds is a single pinned float.

This module may depend on ``numpy`` and ``matplotlib``: it is model-layer, like ``plotting.py``,
imported by the task's figure seam rather than by ``nets/``. It builds no filter bank -- but reusing
the shipped panel means importing the module that owns it, which pulls ``kymatio`` at module scope
and costs about three seconds. **Every consumer therefore imports this module lazily**, inside the
call that draws, so the cost is paid once per run on the rank that writes the figure rather than by
every process that resolves a warm-up budget.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/warmup_budget.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Launched as a script -- from an IDE's Run button, to print the source-reference table at the
# bottom of this file -- Python puts *this directory* on ``sys.path`` rather than the repository
# root, and every absolute import below fails before ``__main__`` is ever reached. Guarded rather
# than unconditional: as an imported module ``__package__`` is set and none of this is needed.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
)
from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    MECHANICAL_SHIFT_SECONDS,
    SECONDS_PER_STEP,
)
from teb_vae.lag_attn_cfs.causal_warmup import (  # noqa: E402
    ALIGNMENT_DELAY_FACTOR,
    StreamWarmup,
    WarmupBudget,
)
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_rws.input_budget import _budget_panel  # noqa: E402
from utils.style import style_axes  # noqa: E402

__all__ = [
    "BUDGET_FIGURE_STEM",
    "CANDIDATE_FLOOR_SECONDS",
    "MIN_UP_PH_KEPT",
    "PHYSIOLOGICAL_BAND_SECONDS",
    "TRADEOFF_FIGURE_STEM",
    "BudgetPoint",
    "SourceReferencePoint",
    "build_tradeoff_figure",
    "build_warmup_budget_figure",
    "budget_tradeoff",
    "format_source_reference_table",
    "source_reference_tradeoff",
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


# =================================================================================================
# The source alignment reference: what each candidate clock buys, and what it costs
# =================================================================================================
#: The uterine-activity-to-heart-rate delay the lag axis exists to resolve, in seconds. Written as
#: a band rather than a number because it is one: the contraction-to-deceleration interval is not a
#: constant, and a reference chosen to put a single value inside the window would put half the band
#: outside it.
#:
#: These two numbers are a **measurement setting** and not a launch convenience, which is why they
#: are a module constant rather than an argument of the entry point below: every candidate row is
#: read against them, and a value injected from a command line would appear in no artifact.
PHYSIOLOGICAL_BAND_SECONDS: Tuple[float, float] = (20.0, 60.0)

#: Fastest source channel a candidate reference may be taken from, in seconds. Below it the
#: envelope block is priced out entirely -- its fastest channel waits $150.786$ s on the shipped
#: bank -- so a faster candidate keeps no contraction channel at all and there is nothing left for
#: the lag search to find.
CANDIDATE_FLOOR_SECONDS = 150.0

#: The stored source block carrying the contraction envelope, and how many of its channels a
#: candidate must keep. The criterion is the whole reason for pricing the reference rather than
#: picking one: a faster clock improves the lag axis and pays for it in exactly this block, so a
#: candidate that reads well on every other column while dropping the envelope has improved the
#: axis by discarding the signal the axis exists to locate.
ENVELOPE_BLOCK = "up_ph"
MIN_UP_PH_KEPT = 8


class SourceReferencePoint(NamedTuple):
    r"""One candidate source clock $\tau^u_{\mathrm{ref}}$, and the three things it decides.

    Attributes:
        reference_s: The candidate itself, always one of the stream's own stored delays -- a
            reference between two of them is a clock no channel keeps.
        kept: Source channels at or below it, i.e. the ones a shift can reach without reading their
            own future.
        declared: The stream's declared width, so ``kept`` is readable without a second lookup.
        block_counts: ``(name, kept, declared)`` per stored source block, which is the resolution
            the envelope question is actually asked at.
        envelope_kept: Survivors of :data:`ENVELOPE_BLOCK`, pulled out of ``block_counts`` because
            it is the one the decision criterion is stated over.
        offset_s: $\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}$, the constant inter-stream bias
            a dual reference puts on the lag axis in place of the unaligned arm's pair-indexed
            smear. Zero under today's single reference; negative for every faster candidate.
        recency_s: How stale the freshest surviving source channel is at the anchor, as the shards
            report it -- which is the reference itself: every aligned channel reports the physical
            instant $\Delta t - \tau^u_{\mathrm{ref}}$.
        realised_recency_s: The same, at the energy centroid $\kappa\,\tau^u_{\mathrm{ref}}$ the
            channels' content actually sits at. Both are reported because
            :data:`~teb_vae.lag_attn_cfs.causal_warmup.ALIGNMENT_DELAY_FACTOR` is exactly why they
            differ and neither alone is the honest figure.
        band_lag_lo: The **smallest** lag index at which the physiological band is reported, which
            is the fastest delay at the furthest horizon step. Below $0$ it is censored at the near
            edge -- the geometry the whole revision starts from.
        band_lag_hi: The **largest**, i.e. the slowest delay at the first horizon step. Above
            $L - 1$ it is censored at the far edge.
        readable: Whether the whole band sits inside $[0, L-1]$ at every horizon step.
        meets_envelope_criterion: Whether ``envelope_kept`` reaches :data:`MIN_UP_PH_KEPT`.
    """

    reference_s: float
    kept: int
    declared: int
    block_counts: Tuple[Tuple[str, int, int], ...]
    envelope_kept: int
    offset_s: float
    recency_s: float
    realised_recency_s: float
    band_lag_lo: float
    band_lag_hi: float
    readable: bool
    meets_envelope_criterion: bool


def _lag_for_physical_delay(
    delay_s: float,
    *,
    offset_s: float,
    horizon_element: int,
    seconds_per_step: float = SECONDS_PER_STEP,
    mechanical_shift_seconds: float = MECHANICAL_SHIFT_SECONDS,
) -> float:
    r"""Which lag index reports a physiological delay of ``delay_s`` at horizon step $h$.

    The inverse of :func:`~teb_vae.lag_attn.nets.lag_report.physical_lag_seconds`, solved for
    $\ell$:

    $$\tau^{\mathrm{phys}}_{\ell,h} = \Delta(\ell + 1 + h) + \bigl(\tau^u_{\mathrm{ref}}
      - \tau^y_{\mathrm{ref}}\bigr) - \tau_{\mathrm{pre}}
    \quad\Longleftrightarrow\quad
    \ell = \frac{\tau^{\mathrm{phys}} + \tau_{\mathrm{pre}} - \mathrm{offset}}{\Delta} - 1 - h .$$

    Returned as a **float** rather than rounded to a bin: what this answers is whether the band
    clears the window's two censoring edges, and rounding first would turn a delay sitting $0.4$
    steps below lag $0$ into one sitting exactly on it.

    Args:
        delay_s: The physiological delay $\tau^{\mathrm{phys}}$ in seconds.
        offset_s: $\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}$, in seconds.
        horizon_element: $h$, which forecast step the delay is reported against.
        seconds_per_step: Seconds per decimated step $\Delta$.
        mechanical_shift_seconds: $\tau_{\mathrm{pre}}$, the sensor delay preprocessing removed.

    Returns:
        The lag index, unrounded and possibly negative.
    """
    return (
        (float(delay_s) + float(mechanical_shift_seconds) - float(offset_s))
        / float(seconds_per_step)
        - 1.0
        - float(horizon_element)
    )


def source_reference_tradeoff(
    source: StreamWarmup,
    *,
    target_reference_s: float,
    horizon: int,
    max_lag: int,
    band_seconds: Tuple[float, float] = PHYSIOLOGICAL_BAND_SECONDS,
    candidate_floor_s: float = CANDIDATE_FLOOR_SECONDS,
) -> List[SourceReferencePoint]:
    r"""Price every candidate clock for the source stream against the target's own.

    Three quantities move against each other and the choice is where they cross, exactly as they do
    for the warm-up budget above -- but they are different quantities and the direction is
    reversed. A **faster** reference brings the freshest source content closer to the anchor and
    pulls the physiological band up off the near censoring edge; it pays for both by dropping every
    source channel slower than itself, and the slow source channels are the contraction envelope.

    The drop rule is not a policy restated here: a channel above the reference could only be brought
    onto it by a negative shift, i.e. by being read from a *later* stored step, which reads raw
    signal after the anchor. That is
    :func:`~teb_vae.lag_attn_cfs.causal_warmup._align_stream`'s correctness requirement, and the
    survivor counts below are the same comparison it makes.

    **Candidates are the stream's own stored delays**, for the reason
    :func:`~teb_vae.lag_attn_cfs.causal_warmup._resolve_reference_delay` snaps an explicit float to
    one: the shift re-indexes a channel onto *some channel's* clock, and a reference landing between
    two of them is a clock no channel keeps, whose residual shows up as a fraction of a step on
    every channel at once rather than as a failure.

    Args:
        source: The resolved source stream, read for its declared delays and its block spans.
        target_reference_s: $\tau^y_{\mathrm{ref}}$, the clock the target keeps. It is also the
            slowest candidate: a source reference above it would move the source *later* than the
            target and push the band toward the far censoring edge instead of away from the near
            one.
        horizon: $H$ in decimated steps. The band is checked at every horizon step, because the lag
            reporting a fixed physiological delay moves one bin per step of $h$ -- which is what
            makes a delay readable at $h = 0$ and censored at $h = H - 1$.
        max_lag: The lag window's furthest searched lag, $L - 1$.
        band_seconds: The physiological delay band, ``(fastest, slowest)``.
        candidate_floor_s: Fastest stored delay a candidate may be taken from.

    Returns:
        One :class:`SourceReferencePoint` per candidate, ascending in the reference. The row at
        ``target_reference_s`` is today's single-reference behaviour and is always included, so
        every other row is read against it.

    Raises:
        ValueError: If the band is not ``(fastest, slowest)`` with both non-negative, or if no
            stored source delay falls in the candidate range at all -- the second being a stream
            this measurement has nothing to say about rather than an empty answer.
    """
    band_lo, band_hi = (float(band_seconds[0]), float(band_seconds[1]))
    if not 0.0 <= band_lo <= band_hi:
        raise ValueError(
            f"band_seconds={band_seconds} must be (fastest, slowest) with both >= 0. The band is "
            f"read as a closed interval of physiological delays, so a reversed pair would report "
            f"the lag bounds inverted and every readability verdict with them."
        )

    delays = np.asarray(source.declared_delay_s, dtype=np.float64)
    reference = float(target_reference_s)
    candidates = np.unique(
        delays[(delays >= float(candidate_floor_s)) & (delays <= reference)]
    )
    if not candidates.size:
        raise ValueError(
            f"no stored {source.name} delay falls in [{float(candidate_floor_s):g}, "
            f"{reference:.4f}] s, so there is no candidate reference to price: the stream's "
            f"delays run {float(delays.min()):.4f}-{float(delays.max()):.4f} s. A reference must "
            f"be one of them, so it is the range that is empty rather than the answer."
        )

    points: List[SourceReferencePoint] = []
    for candidate in candidates.tolist():
        # The alignment's own comparison, exactly: at or below the reference is reachable by a
        # non-negative shift, above it is not.
        keep = np.flatnonzero(delays <= candidate)
        blocks = tuple(
            (name, int(np.count_nonzero((keep >= start) & (keep < stop))), stop - start)
            for name, start, stop in source.block_spans
        )
        envelope = next((kept for name, kept, _ in blocks if name == ENVELOPE_BLOCK), 0)
        offset = candidate - reference
        # The fastest delay at the furthest horizon step is the smallest lag the band ever occupies,
        # and the slowest delay at the first step is the largest: the lag is affine, decreasing in
        # $h$ and increasing in the delay, so those two corners bracket the whole rectangle.
        lag_lo = _lag_for_physical_delay(
            band_lo, offset_s=offset, horizon_element=int(horizon) - 1
        )
        lag_hi = _lag_for_physical_delay(band_hi, offset_s=offset, horizon_element=0)
        points.append(
            SourceReferencePoint(
                reference_s=float(candidate),
                kept=int(keep.size),
                declared=int(delays.size),
                block_counts=blocks,
                envelope_kept=int(envelope),
                offset_s=float(offset),
                recency_s=float(candidate),
                realised_recency_s=float(ALIGNMENT_DELAY_FACTOR * candidate),
                band_lag_lo=float(lag_lo),
                band_lag_hi=float(lag_hi),
                readable=bool(lag_lo >= 0.0 and lag_hi <= float(max_lag)),
                meets_envelope_criterion=bool(envelope >= MIN_UP_PH_KEPT),
            )
        )
    return points


def format_source_reference_table(
    points: Sequence[SourceReferencePoint],
    *,
    max_lag: int,
    horizon: int,
    band_seconds: Tuple[float, float] = PHYSIOLOGICAL_BAND_SECONDS,
) -> str:
    """Lay the candidates out as one row each, with both decision criteria applied.

    Args:
        points: The candidates, from :func:`source_reference_tradeoff`.
        max_lag: The furthest searched lag $L - 1$, which is the far censoring edge.
        horizon: $H$, stated in the header because the lag bounds are taken across it.
        band_seconds: The band the bounds were computed for, stated for the same reason.

    Returns:
        The table, then the candidate the two criteria select and the number that goes into a
        config -- at the precision the resolver snaps at -- or the statement that none does.
    """
    band_lo, band_hi = (float(band_seconds[0]), float(band_seconds[1]))
    blocks = tuple(name for name, _, _ in points[0].block_counts) if points else ()
    lines = [
        f"source alignment reference: {len(points)} candidate(s), band {band_lo:g}-{band_hi:g} s "
        f"across h in [0, {int(horizon) - 1}], lag window [0, {int(max_lag)}]",
        f"  {'reference':>12}  {'kept':>10}  "
        + "  ".join(f"{name:>12}" for name in blocks)
        + f"  {'offset':>9}  {'recency':>16}  {'band lags':>18}  verdict",
    ]
    for point in points:
        counts = "  ".join(
            f"{kept:>5}/{declared:<6}" for _, kept, declared in point.block_counts
        )
        verdict = ("readable" if point.readable else "CENSORED") + (
            "" if point.meets_envelope_criterion else f", {ENVELOPE_BLOCK} short"
        )
        lines.append(
            f"  {point.reference_s:>12.4f}  {point.kept:>4}/{point.declared:<5}  {counts}  "
            f"{point.offset_s:>+9.2f}  {point.recency_s:>6.1f} ({point.realised_recency_s:>5.1f})"
            f"  {point.band_lag_lo:>8.1f} - {point.band_lag_hi:<6.1f}  {verdict}"
        )

    winners = [point for point in points if point.readable and point.meets_envelope_criterion]
    lines.append("")
    if winners:
        # The slowest survivor, not the fastest: every criterion is satisfied by all of them, and
        # among equals the one that discards the fewest source channels is the one to pin.
        chosen = winners[-1]
        declared_envelope = next(
            (declared for name, _, declared in chosen.block_counts if name == ENVELOPE_BLOCK), 0
        )
        lines.append(
            f"pinned: causal_align_reference_source = {chosen.reference_s:.4f} s -- "
            f"{chosen.kept}/{chosen.declared} source channels, {ENVELOPE_BLOCK} "
            f"{chosen.envelope_kept}/{declared_envelope} kept against {MIN_UP_PH_KEPT} required, "
            f"recency {chosen.recency_s:.1f} s reported / {chosen.realised_recency_s:.1f} s "
            f"realised, band at lags {chosen.band_lag_lo:.1f}-{chosen.band_lag_hi:.1f}"
        )
    else:
        lines.append(
            f"no candidate satisfies both criteria: the band must sit inside [0, {int(max_lag)}] "
            f"at every horizon step AND at least {MIN_UP_PH_KEPT} {ENVELOPE_BLOCK} channels must "
            f"survive. Widen the lag window, shorten the horizon, or accept a censored band."
        )
    return "\n".join(lines)


# =================================================================================================
# Entry point
# =================================================================================================
def build_parser() -> argparse.ArgumentParser:
    """The command line, with every default left at ``None``.

    A non-``None`` argparse default would be indistinguishable from a value the operator typed,
    which would make the matching :data:`RUN_ARGS` entry unreachable: the dict would be edited,
    nothing would change, and nothing would say why.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description="Price every candidate alignment reference for the source stream."
    )
    parser.add_argument(
        "--config",
        help=(
            "Config carrying the warm-up budget and the shards to resolve it against. A relative "
            "path is resolved against the repository root, not the working directory."
        ),
    )
    return parser


def main(config: str) -> int:
    """Resolve a config's budget and print the source-reference table.

    Args:
        config: Path to the config, absolute or repository-root-relative.

    Returns:
        The process exit code: $0$ on a printed table, $2$ on a configuration that has none.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.causal_warmup import BUDGET_KEY, resolve_warmup_budget

    path = config if os.path.isabs(config) else os.path.join(_REPO_ROOT, config)
    if not os.path.exists(path):
        print(
            f"--config {config!r} does not resolve to a file. Pass a path, or edit RUN_ARGS near "
            f"the bottom of {os.path.basename(__file__)}."
        )
        return 2

    loaded = load_config(path)
    resolved = resolve_warmup_budget(loaded)
    if resolved is None:
        print(f"{path} sets no {BUDGET_KEY}; there is no stream description to price against.")
        return 2
    if resolved.reference_delay_s is None:
        print(
            f"{path} runs unaligned (causal_align_reference: null), so there is no target "
            f"reference for a source reference to be offset against. The scheme is a pair of "
            f"clocks; price it against a config that configures the first one."
        )
        return 2

    vae_config = (loaded.get("model_config") or {}).get("VAE_model") or {}
    horizon, max_lag = int(vae_config["horizon"]), int(vae_config["max_lag"])
    print(resolved.summary())
    print()
    print(
        format_source_reference_table(
            source_reference_tradeoff(
                resolved.source,
                target_reference_s=resolved.reference_delay_s,
                horizon=horizon,
                max_lag=max_lag,
            ),
            max_lag=max_lag,
            horizon=horizon,
        )
    )
    return 0


#: Values used for arguments absent from the command line -- i.e. an IDE's Run button. Keyed by
#: argparse ``dest``, and merged per key, so a flag overrides one value and leaves the rest of the
#: dict standing.
#:
#: Nothing here has to be filled in: the file runs as it stands. ``planted.yaml`` is the default
#: for two reasons and both are load-bearing. Its shards are **committed**, and they carry the
#: production channel plan -- every causal fixture here is transformed by the real bank, so its
#: delay staircase is the production staircase -- which makes this table the table the pinning
#: decision is made from at no cost in shards. And it is the only committed config carrying the
#: production **lag window**: ``tiny.yaml`` shrinks ``max_lag`` to $8$, and every readability
#: verdict below is a statement about $[0, L-1]$, so a table drawn at that window would report
#: every candidate censored for a reason that is the config's rather than the geometry's.
RUN_ARGS: Dict[str, Any] = {
    "config": "teb_vae/lag_attn_cfs/configs/planted.yaml",
}


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS`, then print the table.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code.
    """
    values, sources = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    # The shard paths inside a config are repo-root-relative, and under an IDE Run button the
    # working directory is whatever the IDE chose -- where a relative path resolves to nothing and
    # the read fails as "no samples match the specified filters" with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)
    print(
        "resolved arguments: "
        + ", ".join(f"{key}={values[key]!r} (from {sources[key]})" for key in sorted(values))
    )
    return main(values["config"])


if __name__ == "__main__":
    sys.exit(_cli())
