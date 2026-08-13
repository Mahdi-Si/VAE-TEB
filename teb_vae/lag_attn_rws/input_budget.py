r"""What the model is actually given, and how it stands relative to the two minutes it forecasts.

Every other diagnostic in this package is about the model's output. This module is about its
input, and it exists because that input is not the raw signal: the net consumes the stored
scattering and phase-harmonic blocks, and under a reach budget it consumes a *subset* of them,
each read a channel-specific number of steps late. Between the shard and the encoder the stream
therefore loses channels and gains staleness, and neither of those is visible anywhere on the
per-sample page unless it is drawn.

Two figures, from one description of the channels:

* **Per-sample rows**, via :func:`stream_panels`, drawn into the diagnostic page between the
  forecast and the latent -- the gated streams themselves, as heatmaps, with the guard's delay
  staircase over them. This answers *what did the encoder read for this recording*.
* **A run-level figure**, via :func:`build_input_budget_figure` -- every declared channel of both
  streams on one anchor-relative time axis, its forward reach drawn as a bar against the shaded
  forecast window. This answers *what may this model be asked to predict*, and it is a constant
  of the configuration rather than of a sample, so the callback writes it once per run.

The second figure is the one that makes the tradeoff legible. A kept channel's bar ends at or
before $0$: its energy stops at the anchor's causal endpoint, which is what the delay bought. A
dropped channel's bar, drawn where it would sit if it were read at the anchor, crosses $0$ and
runs into the forecast window -- for the slowest scattering channels by several times the window's
own width. That crossing *is* the reason it was dropped, stated in the same units as the forecast.

**The arithmetic is imported, not restated.** :mod:`teb_vae.lag_attn.channel_reach` owns the
filter bank, the reaches and the budget resolution, and :mod:`teb_vae.lag_attn.eval.band_partition`
owns the clinical band edges. This module reads the guard off the model's own gates rather than
re-resolving the budget from a config, so a figure cannot describe a guard the model does not
have.

This module may depend on ``numpy``, ``kymatio`` (through ``channel_reach``) and ``matplotlib``:
it is model-layer, like ``plotting.py``, and is imported by the callback rather than by ``nets/``
or by the evaluation. The bank it builds is cached for the process and is already built by the
trainer, which resolves the same budget at start-up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from teb_vae.lag_attn.channel_reach import (  # noqa: E402
    SOURCE_BLOCKS,
    TARGET_BLOCKS,
    block_center_hz,
    block_reach_seconds,
)
from teb_vae.lag_attn.eval.band_partition import CLINICAL_BANDS  # noqa: E402
from teb_vae.lag_attn.figure_primitives import (  # noqa: E402
    COLOR_BLACK,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_VERMILLION,
    to_numpy,
)
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP  # noqa: E402
from teb_vae.lag_attn_rws.sample_page import (  # noqa: E402
    InputStreamPanel,
    annotate_channel_frequencies,
)
from utils.style import style_axes  # noqa: E402

__all__ = [
    "StreamChannels",
    "BUDGET_FIGURE_STEM",
    "build_input_budget_figure",
    "describe_streams",
    "stream_panels",
]

#: Filename stem of the run-level figure. Named here because the callback writes it and the tests
#: look for it, and a literal repeated in both is a filename that can drift in one of them.
BUDGET_FIGURE_STEM = "causal_input_budget"

#: Raw sampling rate in Hz, for turning the horizon into seconds.
_FS_RAW = 4.0


@dataclass(frozen=True)
class StreamChannels:
    r"""One input stream's channels: what each one is, and what the guard did to it.

    Declared-width vectors plus the surviving subset, rather than the surviving subset alone,
    because the run-level figure's whole point is to draw the channels that did **not** survive
    beside the ones that did.

    Attributes:
        name: ``'target'`` or ``'source'``.
        block_spans: ``(name, start, stop)`` per stored block, half-open, in declared coordinates.
        reach_s: Forward reach $L_{95}$ per declared channel, in seconds.
        center_hz: Representative centre frequency per declared channel, in Hz; ``nan`` for the
            order-$0$ scattering low-pass.
        keep_index: Surviving channel indices into the declared width, ascending.
        delays: One delay $\delta_c$ in decimated steps per survivor, same order as
            ``keep_index``.
    """

    name: str
    block_spans: Tuple[Tuple[str, int, int], ...]
    reach_s: np.ndarray
    center_hz: np.ndarray
    keep_index: np.ndarray
    delays: np.ndarray

    @property
    def declared_width(self) -> int:
        """Channels the stream declares, before the guard."""
        return int(self.reach_s.size)

    @property
    def kept_width(self) -> int:
        """Channels the encoder is given."""
        return int(self.keep_index.size)

    @property
    def max_delay(self) -> int:
        r"""The worst survivor's delay $\max_c \delta_c$, in steps."""
        return int(self.delays.max()) if self.delays.size else 0

    def kept_block_spans(self) -> Tuple[Tuple[str, int, int], ...]:
        """The block spans re-expressed in *surviving*-channel coordinates.

        Under a budget the blocks lose different fractions of their channels, so the boundary in
        the gated stream is not where the declared widths put it.

        Returns:
            One ``(name, start, stop)`` per block that kept at least one channel.
        """
        spans: List[Tuple[str, int, int]] = []
        for name, start, stop in self.block_spans:
            inside = int(np.count_nonzero((self.keep_index >= start) & (self.keep_index < stop)))
            if not inside:
                continue
            offset = int(np.count_nonzero(self.keep_index < start))
            spans.append((name, offset, offset + inside))
        return tuple(spans)

    def summary(self) -> str:
        """One line naming the surviving counts, the delay range and the frequency span kept."""
        blocks = ", ".join(
            f"{name} {int(np.count_nonzero((self.keep_index >= start) & (self.keep_index < stop)))}"
            f"/{stop - start}"
            for name, start, stop in self.block_spans
        )
        kept_hz = self.center_hz[self.keep_index]
        finite = kept_hz[np.isfinite(kept_hz)]
        span = f"{finite.min():.3g}–{finite.max():.3g} Hz" if finite.size else "no centre freq."
        return (
            f"{blocks}; {self.kept_width}/{self.declared_width} channels, {span}, "
            f"delay 0–{self.max_delay} steps "
            f"({self.max_delay * SECONDS_PER_STEP:g} s)"
        )


def _gate_of(model: Any, name: str) -> Any:
    """Return the model's gate for a stream, or ``None`` when the run is unguarded.

    Args:
        model: The net.
        name: ``'target'`` or ``'source'``.

    Returns:
        The :class:`~teb_vae.lag_attn.nets.delays.ChannelGate`, or ``None``.
    """
    return getattr(model, f"{name}_gate", None)


def describe_streams(model: Any) -> List[StreamChannels]:
    r"""Describe both input streams of ``model``: their channels, and the guard applied to them.

    The guard is read off the model's own gates. Re-resolving the budget from a configuration
    would let a figure describe a guard the loaded model does not have -- which is precisely the
    case a reader of this figure is checking.

    Args:
        model: The net. Its ``use_up_st`` selects the source blocks, and its ``target_gate`` /
            ``source_gate`` carry the resolved guard, or are ``None`` for an unguarded run.

    Returns:
        The target stream's description, then the source stream's.

    Raises:
        ValueError: If a stream's declared width disagrees with the production filter bank's
            channel count for the blocks it is built from. The keep-index is positional into that
            width, so a mismatch would label the wrong channels rather than fail.
    """
    reaches, frequencies = block_reach_seconds(), block_center_hz()
    source_blocks = SOURCE_BLOCKS if bool(getattr(model, "use_up_st", True)) else SOURCE_BLOCKS[1:]

    described: List[StreamChannels] = []
    for name, blocks, declared_key in (
        ("target", TARGET_BLOCKS, "c_y"),
        ("source", source_blocks, "c_u"),
    ):
        spans: List[Tuple[str, int, int]] = []
        offset = 0
        for block in blocks:
            width = len(reaches[block])
            spans.append((block, offset, offset + width))
            offset += width

        reach = np.array([value for block in blocks for value in reaches[block]], dtype=float)
        hz = np.array([value for block in blocks for value in frequencies[block]], dtype=float)

        # Required, not merely checked when present: a model that declares no channel widths is
        # not over this input representation at all, and treating its absent `c_y` as "no
        # opinion" would draw 109 scattering channels for a model that consumes the raw signal.
        declared = getattr(model, declared_key, None)
        if declared is None:
            raise ValueError(
                f"the model declares no {declared_key}, so it does not consume the stored "
                f"scattering and phase-harmonic blocks these figures describe. A model over "
                f"another input representation needs its own input figures."
            )
        if int(declared) != reach.size:
            raise ValueError(
                f"the model's {declared_key}={int(declared)} disagrees with the filter bank's "
                f"{reach.size} {name} channels for blocks {blocks}. The per-channel reach and "
                f"frequency vectors are positional into that width, so drawing them against this "
                f"model's channels would mislabel every channel rather than fail."
            )

        gate = _gate_of(model, name)
        if gate is None:
            keep_index = np.arange(reach.size)
            delays = np.zeros(reach.size, dtype=int)
        else:
            keep_index = to_numpy(gate.keep_index).astype(int).ravel()
            delays = to_numpy(gate.delay.delay_steps).astype(int).ravel()

        described.append(
            StreamChannels(
                name=name,
                block_spans=tuple(spans),
                reach_s=reach,
                center_hz=hz,
                keep_index=keep_index,
                delays=delays,
            )
        )
    return described


def stream_panels(
    model: Any,
    forward_inputs: Sequence[torch.Tensor],
    *,
    sample_index: int = 0,
) -> List[InputStreamPanel]:
    r"""Build the per-sample input rows from the tensors the net was actually fed.

    ``forward_inputs`` is the task's own ``_build_forward_inputs`` output rather than a
    re-assembly from the batch, and the gates are the model's own modules rather than a
    re-application of the budget, so the drawn stream is the encoder's input by construction and
    not by agreement between two pieces of code.

    Args:
        model: The net.
        forward_inputs: ``(y_st, y_ph, u_stream)``, exactly as splatted into ``forward``.
        sample_index: Which sample of the batch to draw.

    Returns:
        One panel per stream, target first.

    Raises:
        ValueError: If ``forward_inputs`` is not the three tensors this input representation
            uses, or if the streams' widths disagree with the filter bank -- see
            :func:`describe_streams`.
    """
    if len(forward_inputs) != 3:
        raise ValueError(
            f"expected the (y_st, y_ph, u_stream) inputs of the scattering/phase-harmonic input "
            f"representation, got {len(forward_inputs)} tensors. A model over another input "
            f"representation needs its own input rows; there is nothing to gate here."
        )
    y_st, y_ph, u_stream = forward_inputs
    streams = {"target": torch.cat([y_st, y_ph], dim=-1), "source": u_stream}

    panels: List[InputStreamPanel] = []
    for described in describe_streams(model):
        values = streams[described.name]
        # The declared width is what the reach and frequency vectors are positional into, so a
        # stream of another width would draw one model's data under another's channel labels.
        # Checked against the tensor rather than against the model's own `c_y`, because that is
        # the number the figure is about to label.
        if values.dim() != 3 or int(values.shape[-1]) != described.declared_width:
            raise ValueError(
                f"the {described.name} stream is {tuple(values.shape)} but this input "
                f"representation declares {described.declared_width} channels. The channel "
                f"labels are positional into that width and would describe other data."
            )
        gate = _gate_of(model, described.name)
        with torch.no_grad():
            gated = values if gate is None else gate(values)
        guard = (
            f"unguarded (every channel, no delay)"
            if gate is None
            else f"guarded, max delay {described.max_delay} steps "
            f"({described.max_delay * SECONDS_PER_STEP:g} s)"
        )
        panels.append(
            InputStreamPanel(
                name=described.name,
                values=to_numpy(gated[sample_index]),
                delays=described.delays,
                center_hz=described.center_hz[described.keep_index],
                blocks=described.kept_block_spans(),
                title=(
                    f"Model input — {described.name} stream as the encoder receives it: "
                    f"{described.summary()}; {guard}"
                ),
            )
        )
    return panels


def _budget_panel(
    ax: Any, described: StreamChannels, *, horizon_s: float, x_limits: Tuple[float, float]
) -> int:
    r"""Draw one stream's reach bars against the forecast window, and return the clipped count.

    Each channel gets one horizontal bar on an axis whose origin is the anchor's causal endpoint.
    A channel read $\delta_c$ steps late contributes its value from step $t - \delta_c$, whose own
    endpoint sits at $-\Delta \delta_c$, and that value's energy runs forward from there for
    $\mathrm{reach}_c$ seconds -- so the bar spans
    $[-\Delta\delta_c,\ -\Delta\delta_c + \mathrm{reach}_c]$ and the guard's condition
    $\mathrm{reach}_c \le \Delta\delta_c$ is exactly "the bar ends at or before $0$".

    Dropped channels are drawn at $\delta_c = 0$: they are not in the input at all, and the bar
    is the counterfactual that explains their absence -- where their energy *would* have reached
    had they been read at the anchor.

    Args:
        ax: The axes to draw into.
        described: The stream.
        horizon_s: The forecast window's length in seconds.
        x_limits: The drawn time range, in seconds relative to the anchor's causal endpoint.

    Returns:
        How many bars run past the right-hand limit and are therefore clipped.
    """
    kept = np.zeros(described.declared_width, dtype=bool)
    kept[described.keep_index] = True
    delay_steps = np.zeros(described.declared_width, dtype=int)
    delay_steps[described.keep_index] = described.delays

    starts = -SECONDS_PER_STEP * delay_steps.astype(float)
    ends = starts + described.reach_s
    channels = np.arange(described.declared_width)

    # The forecast window first, so every bar is drawn over it rather than under it.
    ax.axvspan(0.0, horizon_s, color=COLOR_VERMILLION, alpha=0.12, zorder=0)
    ax.axvline(0.0, color=COLOR_BLACK, linewidth=0.8, zorder=2)

    for mask, color, label in (
        (kept, COLOR_ORANGE, f"kept ({int(kept.sum())})"),
        (~kept, COLOR_GRAY, f"dropped ({int((~kept).sum())})"),
    ):
        if not mask.any():
            continue
        ax.barh(
            channels[mask], (ends - starts)[mask], left=starts[mask],
            height=1.0, color=color, edgecolor="none", label=label, zorder=1,
        )

    for name, start, stop in described.block_spans:
        if start:
            ax.axhline(start - 0.5, color=COLOR_BLACK, linewidth=0.6, linestyle="--")
        ax.text(
            x_limits[0] + 0.02 * (x_limits[1] - x_limits[0]), 0.5 * (start + stop) - 0.5,
            name, fontsize=7, va="center", ha="left", color=COLOR_BLACK,
        )

    ax.set_xlim(*x_limits)
    ax.set_ylim(-0.5, described.declared_width - 0.5)
    ax.set_title(f"{described.name} stream — {described.summary()}", fontsize=9, pad=6)
    ax.set_xlabel("Seconds relative to the anchor's causal endpoint", fontsize=8)
    ax.set_ylabel("Declared channel", fontsize=8)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    style_axes(ax, grid="major")
    # The same annotation the per-sample rows carry, so a channel index reads as a frequency on
    # both figures without the reader holding a conversion between them.
    annotate_channel_frequencies(ax, described.center_hz)
    return int(np.count_nonzero(ends > x_limits[1]))


def build_input_budget_figure(model: Any, *, geometry: Any = None) -> Any:
    r"""Draw the run-level figure: every input channel's reach against the forecast window.

    Args:
        model: The net, for its gates, its stream widths and (by default) its geometry.
        geometry: The trimmed-grid geometry, or ``None`` to read ``model.geometry``. Only the
            horizon and the raw samples per step are used, to size the forecast window.

    Returns:
        The matplotlib ``Figure``. The caller saves and closes it.
    """
    resolved = geometry if geometry is not None else model.geometry
    horizon_s = float(resolved.horizon * resolved.r) / _FS_RAW
    described = describe_streams(model)

    # Wide enough that the kept channels' delays are readable and the forecast window is not a
    # sliver, and *not* wide enough for the slowest scattering channel's 965 s reach: that bar's
    # informative end is where it crosses zero, and letting it set the scale would compress
    # everything the guard actually did into the leftmost centimetre.
    x_limits = (-1.3 * horizon_s, 1.6 * horizon_s)

    heights = [stream.declared_width for stream in described]
    figure, axes = plt.subplots(
        len(described), 1, figsize=(12, 3.0 + 0.055 * sum(heights)),
        gridspec_kw={"height_ratios": heights, "hspace": 0.28},
        squeeze=False,
    )
    clipped = sum(
        _budget_panel(ax, stream, horizon_s=horizon_s, x_limits=x_limits)
        for ax, stream in zip(axes[:, 0], described)
    )

    bands = ", ".join(
        f"{name} {low:g}–{high:g} Hz" if np.isfinite(high) else f"{name} $\\geq${low:g} Hz"
        for name, (low, high) in CLINICAL_BANDS.items()
    )
    figure.suptitle(
        f"Causal input budget — each channel's forward reach $L_{{95}}$ against the "
        f"{horizon_s:.0f} s forecast window (shaded)\n"
        f"a bar ending at or before 0 cannot see past the anchor; {clipped} bar(s) run past the "
        f"right edge and are clipped.  Clinical bands: {bands}",
        fontsize=10, y=0.995, va="top",
    )
    figure.subplots_adjust(top=1.0 - 1.0 / figure.get_figheight())
    return figure


def write_input_budget_figure(
    model: Any, directory: Path, *, file_format: str = "pdf", geometry: Any = None
) -> Path:
    """Draw the run-level figure and save it under ``directory``.

    Args:
        model: The net.
        directory: Where the file goes; created if absent.
        file_format: Figure extension, without the dot.
        geometry: Passed through to :func:`build_input_budget_figure`.

    Returns:
        The written path.
    """
    from utils.style import SAVE_DPI, save_figure

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{BUDGET_FIGURE_STEM}.{file_format}"
    save_figure(build_input_budget_figure(model, geometry=geometry), path, dpi=SAVE_DPI, close=True)
    return path


if __name__ == "__main__":
    # The budget of the shipped configuration, without a model: the same description built from a
    # resolved budget, so the figure can be checked by eye after a change to the reach arithmetic
    # without starting a fit.
    import types

    from teb_vae.lag_attn.channel_reach import resolve_stream_budgets
    from teb_vae.lag_attn.nets.delays import ChannelGate
    from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

    _budget = resolve_stream_budgets(
        {"causal_reach_budget_s": 120.0, "use_up_st": True, "warmup_period": 30}
    )
    assert _budget is not None
    _stub = types.SimpleNamespace(
        c_y=109,
        c_u=58,
        use_up_st=True,
        target_gate=ChannelGate(
            declared_width=109,
            keep_index=_budget.target_keep_index,
            delays=_budget.target_delays,
        ),
        source_gate=ChannelGate(
            declared_width=58,
            keep_index=_budget.source_keep_index,
            delays=_budget.source_delays,
        ),
        geometry=TrimmedRawGeometry(raw_len=4800, decimation=16, horizon=30, warmup=30),
    )
    for _stream in describe_streams(_stub):
        print(f"{_stream.name}: {_stream.summary()}")
    print(f"wrote {write_input_budget_figure(_stub, Path('output'))}")
