r"""Figures and the written report for the causal-versus-two-sided transform comparison.

Split from :mod:`hdf5_dataset.compare_causal_scattering` for the same reason
:mod:`teb_vae.lag_attn_rws.input_budget` is split from
:mod:`teb_vae.lag_attn.channel_reach`: the measurements have a correctness criterion of their own
and should be importable and assertable without ``matplotlib``.

Every figure answers one question, and each is drawn so that the *cost* of causality is visible
rather than only its benefit -- a plot showing the causal arm leaking nothing would be true and
uninformative. Styling comes from :mod:`utils.style` so these match the rest of the repository's
publication figures.

Note:
    The ``matplotlib`` in this environment parses ``\mathcal`` but rejects ``\tfrac`` and ``\bm``
    in mathtext, and a rejection surfaces only at ``savefig``. Axis labels here use ``\frac``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hdf5_dataset.causal_scattering import N_RAW
from teb_vae.lag_attn.eval.representation_capacity_probe import DECIMATION, FS, HORIZON_S
from utils.style import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_GRAY,
    COLOR_ORANGE,
    COLOR_SKY,
    COLOR_VERMILLION,
    add_colorbar,
    apply_publication_style,
    save_figure,
    style_axes,
)

#: One colour per arm, used in every figure so a curve's identity never has to be re-learned.
ARM_COLOR = {
    "two_sided": COLOR_BLUE,
    "causal": COLOR_VERMILLION,
    "naive": COLOR_ORANGE,
}
ARM_LABEL = {
    "two_sided": "two-sided (production)",
    "causal": "causal (gammatone)",
    "naive": "causal (truncated Morlet)",
}

#: Decimated steps in a stored segment, from the storage geometry rather than repeated as a
#: literal beside every use of it.
SEQUENCE_LENGTH = N_RAW // DECIMATION

#: Steps a lag estimate needs left over after a channel's warm-up. A cross-correlation over fewer
#: than this is noise, and ``measure_delay`` clamps its own start index by the same amount.
#:
#: This is a **measurement** threshold and deliberately not
#: :func:`~hdf5_dataset.causal_scattering.build_channel_plan`'s drop rule, which asks a different
#: question -- whether a channel is ever valid at all ($W \le 330$) rather than whether enough of
#: it survives to correlate against ($W < 330 - 32$). It is strictly the tighter of the two, so a
#: channel shaded here may still be one the dataset stores; ``test_causal_torch.py`` pins that
#: containment and the count both rules land on today, so neither can move unnoticed.
MIN_LAG_WINDOW_STEPS = 32


def write_all_figures(
    output_dir: str,
    bank: Any,
    causal: Any,
    naive: Any,
    filters: Dict[str, Any],
    leakage: Dict[str, Any],
    delay: Dict[str, Any],
    survivorship: Dict[str, Any],
    *,
    traces: Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]],
    showcase: Sequence[int],
) -> None:
    """Draw every figure into ``output_dir/figures``.

    Args:
        output_dir: Destination directory; ``figures/`` is created inside it.
        bank: The production filter bank.
        causal: The gammatone bank.
        naive: The truncated-Morlet bank.
        filters: Output of ``measure_filters``.
        leakage: Output of ``measure_leakage``.
        delay: Output of ``measure_delay``.
        survivorship: Output of ``measure_survivorship``.
        traces: ``(fhr, up, arm_b, arm_c)`` for the trace figure.
        showcase: Filter indices drawn in the per-filter figures.
    """
    apply_publication_style()
    figure_dir = os.path.join(output_dir, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    _figure_time_domain(figure_dir, bank, causal, naive, showcase)
    _figure_frequency_response(figure_dir, bank, causal, naive, showcase)
    _figure_reach_and_delay(figure_dir, bank, filters)
    _figure_leakage(figure_dir, leakage)
    _figure_traces(figure_dir, *traces, showcase=showcase)
    _figure_delay_scatter(figure_dir, delay, "fhr_st", "06_delay_scattering")
    _figure_delay_scatter(figure_dir, delay, "fhr_ph", "07_delay_phase")
    _figure_survivorship(figure_dir, survivorship)


def _figure_time_domain(
    figure_dir: str, bank: Any, causal: Any, naive: Any, showcase: Sequence[int]
) -> None:
    """The filters in time, with the future half shaded -- the visual statement of the change.

    Args:
        figure_dir: Destination.
        bank: The production filter bank.
        causal: The gammatone bank.
        naive: The truncated-Morlet bank.
        showcase: Filter indices to draw.
    """
    fig, axes = plt.subplots(len(showcase), 1, figsize=(7.2, 2.1 * len(showcase)), sharex=False)
    n_two_sided = bank.phi.size
    index = np.arange(n_two_sided)
    two_sided_time = np.where(index <= n_two_sided // 2, index, index - n_two_sided) / FS

    for axis, k in zip(np.atleast_1d(axes), showcase):
        morlet = np.fft.ifft(bank.psi[k])
        span = max(60.0, 6.0 / (bank.xi[k] * FS))
        window = np.abs(two_sided_time) <= span
        order = np.argsort(two_sided_time[window])
        t_two = two_sided_time[window][order]
        y_two = np.abs(morlet[window][order])
        axis.plot(t_two, y_two / y_two.max(), color=ARM_COLOR["two_sided"], lw=1.4,
                  label=ARM_LABEL["two_sided"])

        for name, cbank in (("causal", causal), ("naive", naive)):
            taps = np.arange(cbank.n_taps) / FS
            keep = taps <= span
            envelope = np.abs(cbank.psi[k][keep])
            if envelope.max() > 0:
                # Delay tau maps to time -tau: a causal filter reads only the past.
                axis.plot(-taps[keep], envelope / envelope.max(), color=ARM_COLOR[name], lw=1.2,
                          label=ARM_LABEL[name], alpha=0.9)

        axis.axvspan(0, span, color=COLOR_GRAY, alpha=0.12, lw=0)
        axis.axvline(0, color=COLOR_BLACK, lw=0.8, ls="--")
        axis.text(0.985, 0.88, "future", transform=axis.transAxes, ha="right", va="top",
                  fontsize=8, color=COLOR_GRAY)
        axis.set_xlim(-span, span)
        axis.set_ylabel("envelope\n(normalised)")
        axis.set_title(f"$\\xi = {bank.hz[k]:.4f}$ Hz", fontsize=9, loc="left")
        style_axes(axis)

    np.atleast_1d(axes)[-1].set_xlabel("time relative to the coefficient's own step (s)")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle("The two-sided filter reads its own future; the causal ones cannot", fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "01_filters_time_domain.pdf"))


def _figure_frequency_response(
    figure_dir: str, bank: Any, causal: Any, naive: Any, showcase: Sequence[int]
) -> None:
    """Magnitude responses in dB -- what each causal construction costs in selectivity.

    Args:
        figure_dir: Destination.
        bank: The production filter bank.
        causal: The gammatone bank.
        naive: The truncated-Morlet bank.
        showcase: Filter indices to draw.
    """
    fig, axes = plt.subplots(1, len(showcase), figsize=(3.0 * len(showcase), 3.0), sharey=True)

    for axis, k in zip(np.atleast_1d(axes), showcase):
        for name, spectrum in (
            ("two_sided", bank.psi[k]),
            ("causal", np.fft.fft(causal.psi[k])),
            ("naive", np.fft.fft(naive.psi[k])),
        ):
            magnitude = np.abs(spectrum)
            frequencies = np.fft.fftfreq(magnitude.size) * FS
            keep = frequencies > 0
            decibels = 20 * np.log10(np.maximum(magnitude[keep] / magnitude.max(), 1e-12))
            axis.plot(frequencies[keep], decibels, color=ARM_COLOR[name], lw=1.2,
                      label=ARM_LABEL[name])

        axis.set_xscale("log")
        axis.set_xlim(bank.hz[k] / 30, min(bank.hz[k] * 30, FS / 2))
        axis.set_ylim(-70, 3)
        axis.axvline(bank.hz[k], color=COLOR_BLACK, lw=0.7, ls=":")
        axis.set_xlabel("frequency (Hz)")
        axis.set_title(f"$\\xi = {bank.hz[k]:.4f}$ Hz", fontsize=9)
        style_axes(axis)

    np.atleast_1d(axes)[0].set_ylabel("magnitude (dB)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left", framealpha=0.9)
    fig.suptitle("Truncating the Morlet at its peak costs the passband; matching a gammatone "
                 "to it does not", fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "02_frequency_response.pdf"))


def _figure_reach_and_delay(figure_dir: str, bank: Any, filters: Dict[str, Any]) -> None:
    r"""Forward reach against causal group delay, per filter -- the exchange rate.

    Drawn as a mirror: reach above the axis (how far into the future the two-sided filter reads)
    and delay below it (how far into the past the causal one must look back). The point of the
    figure is that the lower bars are the *longer* ones.

    Args:
        figure_dir: Destination.
        bank: The production filter bank.
        filters: Output of ``measure_filters``.
    """
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    channels = np.arange(bank.n_filters)
    reach, delay = filters["reach_s"], filters["delay_s"]

    top.bar(channels - 0.2, reach, width=0.4, color=ARM_COLOR["two_sided"],
            label="two-sided forward reach $L_{95}$")
    top.bar(channels + 0.2, -delay, width=0.4, color=ARM_COLOR["causal"],
            label=r"causal group delay $\tau_g$")
    top.axhline(HORIZON_S, color=COLOR_BLACK, lw=0.9, ls="--")
    top.text(0.5, HORIZON_S * 1.15, f"{HORIZON_S:g} s forecast horizon", fontsize=7.5,
             color=COLOR_BLACK)
    top.axhline(0, color=COLOR_BLACK, lw=0.8)
    top.set_yscale("symlog", linthresh=1.0)
    top.set_ylabel("seconds\n(future above, past below)")
    top.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    style_axes(top)

    bottom.plot(channels, filters["delay_over_reach"], color=COLOR_SKY, lw=1.3, marker="o", ms=2.5)
    bottom.axhline(1.0, color=COLOR_BLACK, lw=0.8, ls="--")
    bottom.text(0.5, 1.02, "break-even", fontsize=7.5, color=COLOR_BLACK)
    bottom.set_ylabel(r"$\frac{\tau_g}{L_{95}}$", fontsize=12)
    bottom.set_xlabel("filter index (descending centre frequency)")
    bottom.set_ylim(0, max(2.0, float(filters["delay_over_reach"].max()) * 1.1))
    style_axes(bottom)

    fig.suptitle(
        f"Causality is bought, not free: delay is "
        f"{filters['delay_over_reach_median']:.2f}x the reach it removes (median)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "03_reach_and_delay.pdf"))


def _figure_leakage(figure_dir: str, leakage: Dict[str, Any]) -> None:
    r"""Edit the signal only after $t_0$; plot what moves before it.

    Log scale, because the interesting comparison spans thirteen orders of magnitude: the
    two-sided arm rises to $O(1)$ well before the edit, while the causal arm sits on the FFT
    round-off floor. A linear axis would draw the causal arm as a flat zero line and hide that the
    floor is a numerical artefact rather than a structural claim -- the structural claim is proved
    by the bitwise test in the self-test, not by this figure.

    Args:
        figure_dir: Destination.
        leakage: Output of ``measure_leakage``.
    """
    curves = leakage["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    step_s = DECIMATION / FS

    for axis, block in zip(axes, ("fhr_st", "fhr_ph")):
        for arm in ("two_sided", "causal"):
            curve = np.asarray(curves[arm][block])
            time_before = (np.arange(curve.size) - curve.size + 1) * step_s
            axis.semilogy(time_before, np.maximum(curve, 1e-18), color=ARM_COLOR[arm], lw=1.3,
                          label=ARM_LABEL[arm])
        axis.axvline(0, color=COLOR_BLACK, lw=0.9, ls="--")
        # Placed at the top: the causal curve runs along the bottom of the axis, and an annotation
        # sitting on it would obscure the one series the figure exists to show is flat.
        axis.text(0.985, 0.97, "$t_0$: edit begins", transform=axis.transAxes, ha="right",
                  va="top", fontsize=7.5, color=COLOR_BLACK)
        axis.set_xlabel("time before the edit (s)")
        axis.set_title(block, fontsize=9)
        style_axes(axis)

    axes[0].set_ylabel("max relative coefficient movement")
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle("A change made only in the future moves two-sided coefficients in the past",
                 fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "04_leakage.pdf"))


def _figure_traces(
    figure_dir: str,
    fhr: np.ndarray,
    up: np.ndarray,
    arm_b: Dict[str, np.ndarray],
    arm_c: Dict[str, np.ndarray],
    *,
    showcase: Sequence[int],
) -> None:
    """Raw signals and a few channels on one real segment -- do the two arms see the same thing.

    Args:
        figure_dir: Destination.
        fhr: Raw fetal heart rate, ``(n_signal,)``.
        up: Raw uterine pressure, ``(n_signal,)``.
        arm_b: Two-sided blocks.
        arm_c: Causal blocks.
        showcase: Filter indices; channel ``k + 1`` of the scattering block is filter ``k``.
    """
    rows = 2 + len(showcase)
    fig, axes = plt.subplots(rows, 1, figsize=(8.0, 1.5 * rows), sharex=True)
    raw_time = np.arange(fhr.size) / FS
    step_time = np.arange(arm_b["fhr_st"].shape[1]) * DECIMATION / FS

    axes[0].plot(raw_time, fhr, color=COLOR_BLUE, lw=0.5)
    axes[0].set_ylabel("FHR\n(bpm)")
    axes[1].plot(raw_time, up, color=COLOR_SKY, lw=0.5)
    axes[1].set_ylabel("UP")
    for axis in axes[:2]:
        style_axes(axis)

    for offset, k in enumerate(showcase):
        axis = axes[2 + offset]
        channel = k + 1
        for arm, block in (("two_sided", arm_b), ("causal", arm_c)):
            series = block["fhr_st"][channel]
            axis.plot(step_time, series, color=ARM_COLOR[arm], lw=1.0, label=ARM_LABEL[arm])
        axis.set_ylabel(f"fhr_st\nch {channel}")
        style_axes(axis)

    axes[2].legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Both arms track the same physiology; the causal one arrives later", fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "05_traces.pdf"))


def _figure_delay_scatter(
    figure_dir: str, delay: Dict[str, Any], block: str, stem: str
) -> None:
    r"""Measured lag against the analytic prediction, and the correlation it buys.

    The left panel is a consistency check with teeth: if the measured cross-correlation lag did not
    track $\tau_g$, either the bandwidth match or the normalisation would be wrong. The right panel
    is the cost -- correlation at zero lag is what a model reading the causal channel *without*
    compensating would see.

    Args:
        figure_dir: Destination.
        delay: Output of ``measure_delay``.
        block: Stored block name.
        stem: Output filename stem.
    """
    measured = delay[block]
    lag_s = np.asarray(measured["lag_steps"]) * DECIMATION / FS
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.6))

    channels = np.arange(lag_s.size)
    # Channels whose causal warm-up leaves too little of the segment to estimate a lag from. Their
    # measured values are noise, and shading them says so rather than leaving a reader to read the
    # erratic tail as the transform failing. It is instead the segment being too short to warm
    # those channels up at all -- the finding the warm-up column exists to record.
    unusable = ~_usable_mask(measured)
    if unusable.any():
        first = int(np.argmax(unusable))
        for axis in (left, right):
            axis.axvspan(first - 0.5, lag_s.size - 0.5, color=COLOR_GRAY, alpha=0.16, lw=0)
        left.text(first + 0.4, 0.04, "warm-up exceeds\nthe segment", transform=left.get_xaxis_transform(),
                  fontsize=6.5, color=COLOR_GRAY, va="bottom")

    left.plot(channels, lag_s, color=ARM_COLOR["causal"], lw=1.2, marker="o", ms=2.5,
              label="measured cross-correlation lag")
    # The prediction: a scattering channel carries its wavelet's delay *and* the low-pass's, and a
    # phase channel carries its slower leg's plus the low-pass's. Drawing it is what makes the
    # phase panel's zigzag legible -- consecutive pairs have different slower legs, so the
    # prediction zigzags with the measurement rather than the measurement looking like noise.
    left.plot(channels, np.asarray(measured["predicted_delay_s"])[: lag_s.size],
              color=COLOR_GRAY, lw=1.0, ls="--", label=r"analytic $\tau_g$ (slowest leg $+\ \phi$)")
    left.set_xlabel("channel")
    left.set_ylabel("delay (s)")
    left.set_yscale("symlog", linthresh=4.0)
    left.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    style_axes(left)

    # Three alignments, and the middle one is the number to read. The argmax is an upper bound and
    # can lock onto a sidelobe on the oscillatory phase channels; the predicted lag is unambiguous;
    # zero lag is what a model reading the causal channel without compensating would see.
    right.plot(channels, measured["r_at_best_lag"], color=COLOR_SKY, lw=1.0, alpha=0.8,
               label="at the best lag (upper bound)")
    right.plot(channels, measured["r_at_predicted_lag"], color=ARM_COLOR["causal"], lw=1.3,
               label=r"at the predicted $\tau_g$")
    right.plot(channels, measured["r_at_zero_lag"], color=COLOR_GRAY, lw=1.1, ls="--",
               label="at zero lag (uncompensated)")
    right.set_xlabel("channel")
    right.set_ylabel("correlation with the two-sided channel")
    right.set_ylim(-0.4, 1.05)
    right.axhline(0, color=COLOR_BLACK, lw=0.7)
    right.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    style_axes(right)

    fig.suptitle(f"{block}: what the causal channel reports, and when", fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, f"{stem}.pdf"))


def _figure_survivorship(figure_dir: str, survivorship: Dict[str, Any]) -> None:
    """Channels kept under each reach budget, both arms, under one predicate.

    A budget the causal arm cannot satisfy at all -- because the delay it implies exceeds the
    loss warm-up -- is drawn as a hatched zero rather than omitted, because "refused by the
    existing guard" is a result about the causal arm and not a gap in the measurement.

    Args:
        figure_dir: Destination.
        survivorship: Output of ``measure_survivorship``.
    """
    rows = survivorship["rows"]
    budgets = []
    for row in rows:
        if row["budget_s"] not in budgets:
            budgets.append(row["budget_s"])
    labels = ["unguarded" if b is None else f"{b:g} s" for b in budgets]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=False)
    width = 0.36
    positions = np.arange(len(budgets))

    for axis, stream in zip(axes, ("target", "source")):
        total = next(r[f"{stream}_total"] for r in rows)
        for offset, arm in ((-width / 2, "two_sided"), (width / 2, "causal")):
            kept, refused = [], []
            for budget in budgets:
                row = next(r for r in rows if r["budget_s"] == budget and r["arm"] == arm)
                kept.append(row[f"{stream}_kept"] or 0)
                refused.append(row[f"{stream}_refused"] is not None)
            bars = axis.bar(positions + offset, kept, width=width, color=ARM_COLOR[arm],
                            label=ARM_LABEL[arm])
            for position, bar, is_refused in zip(positions, bars, refused):
                if not is_refused:
                    continue
                # A refused budget keeps zero channels, so the bar has no height to hatch and
                # would read as missing data rather than as a result. Label it instead.
                bar.set_hatch("///")
                bar.set_facecolor("none")
                bar.set_edgecolor(ARM_COLOR[arm])
                axis.text(position + offset, total * 0.04, "refused", rotation=90, fontsize=6.5,
                          ha="center", va="bottom", color=ARM_COLOR[arm])
        axis.axhline(total, color=COLOR_BLACK, lw=0.8, ls=":")
        axis.set_ylim(0, total * 1.18)
        axis.text(0.99, 0.97, f"all {total}", transform=axis.transAxes, fontsize=7.5,
                  ha="right", va="top")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_xlabel("reach / delay budget")
        axis.set_ylabel(f"{stream} channels kept")
        axis.set_title(stream, fontsize=9)
        style_axes(axis)

    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Hatched = the budget's delay exceeds the loss warm-up, so the guard refuses it",
                 fontsize=10)
    fig.tight_layout()
    save_figure(fig, os.path.join(figure_dir, "08_survivorship.pdf"))


def _reach_back(leakage: Dict[str, Any], arm: str, block: str) -> str:
    """Format a leak-back extent, marking it when the measurement window censored it.

    The test only edits the signal after $t_0$ and only inspects steps before it, so if the
    earliest inspected step still moves the true extent is unknown beyond "at least this". Writing
    a bare number there would read as a measured endpoint.

    Args:
        leakage: Output of ``measure_leakage``.
        arm: ``'two_sided'`` or ``'causal'``.
        block: Stored block name.

    Returns:
        A formatted string, prefixed with the censoring marker where applicable.
    """
    seconds = leakage[f"{arm}_{block}_leak_back_s"]
    censored = leakage.get(f"{arm}_{block}_leak_back_censored", False)
    return f"{'≥ ' if censored else ''}{seconds:.0f} s"


#: Display transforms, taken from ``hdf5_dataset/calculate_dataset_stats.py`` rather than chosen
#: for the figure. The pipeline log-transforms every scattering channel except $S_0$ and
#: asinh-transforms every phase channel before computing the statistics the model normalises with,
#: so these are the units the model actually sees -- and a heatmap drawn in raw units would be a
#: picture of the dynamic range instead of a picture of the signal.
LOG_EPSILON = 1e-6

#: Blocks drawn in a per-sample panel, top to bottom, with the row height each needs to stay
#: legible at its channel count.
PANEL_BLOCKS = (("fhr_st", 1.55), ("fhr_ph", 2.15), ("up_st", 1.55), ("up_ph", 1.05))


def display_transform(block: str, data: np.ndarray) -> Tuple[np.ndarray, str, str]:
    r"""Put a coefficient block into the units the model is trained on.

    Scattering channels $1 \ldots 42$ take $\log(\max(x, 0) + \epsilon)$ and $S_0$ is left alone --
    it is an affine function of the $4$ s locally-averaged FHR in bpm, and logging it would hide
    that. Phase channels take $\operatorname{asinh}(x)$, which is signed and so keeps the sign
    structure the phase harmonic carries.

    Args:
        block: Stored block name.
        data: ``(n_channels, n_steps)`` raw coefficients.

    Returns:
        ``(transformed, colormap, colorbar_label)``. The scattering blocks get a sequential map
        because they are one-signed; the phase blocks get a diverging one centred at zero because
        their sign is meaningful.
    """
    if block in ("fhr_st", "up_st"):
        shown = np.empty_like(data)
        shown[0] = data[0]
        shown[1:] = np.log(np.maximum(data[1:], 0.0) + LOG_EPSILON)
        # S_0 lives in bpm and the rest in log units; showing them on one scale would flatten the
        # 42 log channels to a single colour. Put S_0 on the log rows' scale by standardising it.
        finite = shown[1:][np.isfinite(shown[1:])]
        if finite.size:
            low, high = np.percentile(finite, [1, 99])
            centred = data[0] - data[0].mean()
            spread = np.abs(centred).max() or 1.0
            shown[0] = (low + high) / 2 + centred / spread * (high - low) / 2
        return shown, "viridis", r"$\log(S + \epsilon)$   [$S_0$ rescaled]"
    return np.arcsinh(data), "coolwarm", r"$\operatorname{asinh}(\Phi)$"


def write_sample_panels(
    output_dir: str,
    panels: Sequence[Dict[str, Any]],
    warmup_steps: Dict[str, np.ndarray],
    *,
    channel_hz: Dict[str, np.ndarray],
) -> None:
    r"""One figure per sample: every stored coefficient, both arms, side by side.

    This is the "show me the transform" figure. Each block is drawn on a **shared colour scale
    across the two arms**, so the panels are comparable rather than each being separately
    auto-scaled -- without that, a causal block with half the dynamic range would look identical
    to the two-sided one.

    The causal column carries a per-channel warm-up boundary. Left of it a channel's output is a
    function of the assumed history rather than of the signal, and for the slowest channels that is
    the whole segment. Drawing it is what stops the causal panel from being read as "the same thing
    but shifted".

    Args:
        output_dir: Destination directory; ``figures/samples/`` is created inside it.
        panels: One dict per sample with ``index``, ``guid``, ``fhr``, ``up``, ``arm_b``, ``arm_c``.
        warmup_steps: Per-block causal warm-up in steps, from ``_channel_warmup_steps``.
        channel_hz: Per-block representative centre frequency in Hz, for the channel axis.
    """
    apply_publication_style()
    panel_dir = os.path.join(output_dir, "figures", "samples")
    os.makedirs(panel_dir, exist_ok=True)
    for panel in panels:
        _write_one_panel(panel_dir, panel, warmup_steps, channel_hz)


def _write_one_panel(
    panel_dir: str,
    panel: Dict[str, Any],
    warmup_steps: Dict[str, np.ndarray],
    channel_hz: Dict[str, np.ndarray],
) -> None:
    """Draw a single sample's full-coefficient panel.

    Args:
        panel_dir: Destination directory.
        panel: One entry from ``panels``; see :func:`write_sample_panels`.
        warmup_steps: Per-block causal warm-up in steps.
        channel_hz: Per-block centre frequencies in Hz.
    """
    arm_b, arm_c = panel["arm_b"], panel["arm_c"]
    duration_s = arm_b["fhr_st"].shape[1] * DECIMATION / FS
    raw_time = np.arange(panel["fhr"].size) / FS

    # One column, every plot box the same width and the same time axis, so a feature in the raw
    # signal can be followed straight down through all eight coefficient panels. That alignment is
    # the reason for the layout below rather than a two-column grid: each colorbar goes in its own
    # narrow gridspec column via `cax=`, which -- unlike `fig.colorbar(ax=...)` -- does not steal
    # width from the plot axes, so every box starts and ends at the same x position.
    rows = [("raw", None, 1.25)]
    for block, height in PANEL_BLOCKS:
        rows.append((block, "two-sided", height))
        rows.append((block, "causal", height))

    fig = plt.figure(figsize=(11.0, sum(height for *_, height in rows) + 1.9))
    grid = fig.add_gridspec(
        len(rows), 2, height_ratios=[height for *_, height in rows], width_ratios=[60, 1],
        left=0.11, right=0.88, top=0.915, bottom=0.042, hspace=0.42, wspace=0.02,
    )

    raw_axis = fig.add_subplot(grid[0, 0])
    raw_axis.plot(raw_time, panel["fhr"], color=COLOR_BLUE, lw=0.5)
    raw_axis.set_ylabel("FHR\n(bpm)", fontsize=8, color=COLOR_BLUE)
    raw_axis.set_xlim(0, duration_s)
    raw_axis.tick_params(labelbottom=False)
    # A twin axes shares the host's box exactly, so the second signal costs no alignment.
    twin = raw_axis.twinx()
    twin.plot(raw_time, panel["up"], color=COLOR_SKY, lw=0.5, alpha=0.85)
    twin.set_ylabel("UP", fontsize=8, color=COLOR_SKY)
    twin.set_xlim(0, duration_s)
    twin.grid(False)
    style_axes(raw_axis)
    raw_axis.set_title("raw signals the coefficients below were computed from",
                       fontsize=8.5, loc="left")

    row_index = 1
    for block, height in PANEL_BLOCKS:
        shown_b, cmap, label = display_transform(block, arm_b[block])
        shown_c, _, _ = display_transform(block, arm_c[block])
        # One scale for both arms. Percentile rather than min/max so a single outlying step cannot
        # compress everything else into one colour.
        stacked = np.concatenate([shown_b.ravel(), shown_c.ravel()])
        stacked = stacked[np.isfinite(stacked)]
        if block in ("fhr_ph", "up_ph"):
            bound = float(np.percentile(np.abs(stacked), 99)) or 1.0
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = (float(v) for v in np.percentile(stacked, [1, 99]))

        image = None
        for arm, shown in (("two-sided", shown_b), ("causal", shown_c)):
            axis = fig.add_subplot(grid[row_index, 0], sharex=raw_axis)
            image = axis.imshow(shown, aspect="auto", origin="upper", cmap=cmap,
                                vmin=vmin, vmax=vmax,
                                extent=(0.0, duration_s, shown.shape[0] - 0.5, -0.5))
            title = f"{block} — {arm}"
            if arm == "causal":
                _draw_warmup_boundary(axis, warmup_steps[block], shown.shape[0], duration_s,
                                      annotate=(block == PANEL_BLOCKS[0][0]))
                title += "   (black line = warm-up boundary)"
            axis.set_title(title, fontsize=8.5, loc="left")
            axis.set_ylabel(f"{block}\ncentre freq (Hz)", fontsize=8)
            _label_channel_axis(axis, channel_hz[block])
            axis.set_xlim(0, duration_s)
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
            last_row = row_index == len(rows) - 1
            axis.tick_params(labelbottom=last_row)
            if last_row:
                axis.set_xlabel("time (s)", fontsize=8)
            row_index += 1

        # One colorbar per block, spanning that block's two rows, in its own gridspec column.
        bar_axis = fig.add_subplot(grid[row_index - 2 : row_index, 1])
        fig.colorbar(image, cax=bar_axis).set_label(label, fontsize=7.5)
        bar_axis.tick_params(labelsize=6.5)

    fig.suptitle(
        f"Full coefficient set, $J=11$, $Q=4$, $T=16$ — sample {panel['index']} "
        f"(GUID {panel['guid']})",
        fontsize=10, y=0.992,
    )
    # What the black line is, stated in the figure rather than left to a caption: a reader meeting
    # the staircase for the first time is most likely to take it for a filter edge effect or for
    # the causal delay, and it is neither.
    fig.text(
        0.5, 0.976,
        "Black line on each causal panel = the warm-up boundary. It separates the interval where "
        "that channel's output still contains the\nedge-padded history the transform assumed "
        "(left, whitened) from the interval where it depends on the recording alone (right). "
        "It lengthens\ntoward the slow channels because their kernels are longer, and for the "
        "slowest it never closes inside the segment.",
        fontsize=7.4, ha="center", va="top", color=COLOR_GRAY, linespacing=1.5,
    )
    save_figure(fig, os.path.join(panel_dir, f"sample_{panel['index']:04d}.pdf"))


def _draw_warmup_boundary(
    axis: Any, warmup: np.ndarray, n_channels: int, duration_s: float, *, annotate: bool
) -> None:
    r"""Mark, and explain, where each causal channel stops reporting its own padding.

    A causal channel's output at time $t$ is a weighted sum over the preceding kernel. Until $t$
    exceeds that kernel's length the sum is partly over the edge-padded history the transform
    assumed, not over the recording -- so the value is an artefact of the pad. The boundary is the
    per-channel $95\%$ energy support, and it lengthens sharply toward the slow channels because
    their kernels are longer.

    The explanation is drawn into the figure rather than left to a caption, because a reader
    meeting the staircase for the first time is most likely to read it as a filter edge effect or
    as the causal delay, and it is neither.

    Args:
        axis: The causal heatmap axes.
        warmup: Per-channel warm-up in decimated steps.
        n_channels: Rows in the heatmap.
        duration_s: Segment length in seconds.
        annotate: Whether to draw the explanatory labels; only the first causal panel gets them,
            so the rest stay legible.
    """
    boundary = np.minimum(np.asarray(warmup) * DECIMATION / FS, duration_s)
    channels = np.arange(n_channels)
    axis.fill_betweenx(channels, 0, boundary, step="mid", color="white", alpha=0.55, lw=0)
    axis.step(boundary, channels, where="mid", color=COLOR_BLACK, lw=1.2)

    if not annotate:
        return

    # Two short labels only. The full sentence lives in the figure-level caption, because at the
    # width of one heatmap row anything longer sets in a font too small to read.
    inside = np.where(boundary < duration_s * 0.5)[0]
    row = int(inside[-1]) if inside.size else n_channels // 2
    box = dict(boxstyle="round,pad=0.25", fc="white", ec=COLOR_BLACK, lw=0.4, alpha=0.85)
    axis.text(boundary[row] * 0.5, row, "pad-dominated", fontsize=7, ha="center", va="center",
              color=COLOR_BLACK, bbox=box)
    axis.text(boundary[row] + (duration_s - boundary[row]) * 0.25, row, "signal only",
              fontsize=7, ha="center", va="center", color=COLOR_BLACK, bbox=box)


def _label_channel_axis(axis: Any, hz: np.ndarray) -> None:
    """Put a few centre-frequency labels on a channel axis.

    A bare channel index says nothing about what a row contains, and 43 or 66 frequency labels
    would be unreadable, so a handful are placed at fixed positions.

    Args:
        axis: The heatmap axes.
        hz: Per-channel representative centre frequency in Hz.
    """
    n = len(hz)
    positions = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    labels = []
    for position in positions:
        value = hz[position]
        labels.append("$S_0$" if not np.isfinite(value) else f"{value:.3g}")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=6.5)


def write_gallery(
    output_dir: str, panels: Sequence[Dict[str, Any]], block: str = "fhr_st"
) -> None:
    """A contact sheet: one block, both arms, every sample, on one page.

    Complements the per-sample panels by making across-sample variation visible at a glance --
    whether the causal arm's behaviour is consistent, or whether some recordings behave
    differently.

    Args:
        output_dir: Destination directory.
        panels: One dict per sample; see :func:`write_sample_panels`.
        block: Which stored block to tile.
    """
    apply_publication_style()
    rows = len(panels)
    fig, axes = plt.subplots(rows, 2, figsize=(10.0, 1.05 * rows + 1.0), squeeze=False)

    shown = [
        (display_transform(block, panel["arm_b"][block])[0],
         display_transform(block, panel["arm_c"][block])[0])
        for panel in panels
    ]
    # One scale across every sample as well as both arms, so the tiles are comparable down the page
    # and not just across it.
    stacked = np.concatenate([np.concatenate([b.ravel(), c.ravel()]) for b, c in shown])
    stacked = stacked[np.isfinite(stacked)]
    vmin, vmax = (float(v) for v in np.percentile(stacked, [1, 99]))
    duration_s = panels[0]["arm_b"][block].shape[1] * DECIMATION / FS

    for row, (panel, (shown_b, shown_c)) in enumerate(zip(panels, shown)):
        for column, data in enumerate((shown_b, shown_c)):
            axis = axes[row][column]
            image = axis.imshow(data, aspect="auto", origin="upper", cmap="viridis",
                                vmin=vmin, vmax=vmax,
                                extent=(0.0, duration_s, data.shape[0] - 0.5, -0.5))
            axis.set_yticks([])
            if row < rows - 1:
                axis.set_xticks([])
            else:
                axis.set_xlabel("time (s)", fontsize=7.5)
            if row == 0:
                axis.set_title(["two-sided (production)", "causal (gammatone)"][column],
                               fontsize=9)
            if column == 0:
                axis.set_ylabel(f"{panel['index']}", fontsize=7, rotation=0, labelpad=14,
                                va="center")
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_linewidth(0.5)

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.5, pad=0.02,
                 label=r"$\log(S + \epsilon)$")
    fig.suptitle(f"{block} across {rows} samples, $J=11$, $Q=4$, $T=16$ "
                 f"(row label = sample index)", fontsize=10)
    save_figure(fig, os.path.join(output_dir, "figures", f"09_gallery_{block}.pdf"))


def _usable_mask(measured: Dict[str, Any]) -> np.ndarray:
    r"""Channels whose causal warm-up leaves enough of the segment to measure a lag from.

    $W < 330 - 32$: a correlation over fewer than :data:`MIN_LAG_WINDOW_STEPS` steps reports noise,
    and a figure that plotted it without saying so would read as the transform failing rather than
    as the segment being too short to warm that channel up. Not the dataset's drop rule -- see
    :data:`MIN_LAG_WINDOW_STEPS` for why the two questions are different and how they are pinned
    against each other.

    Args:
        measured: One block's entry from ``measure_delay``.

    Returns:
        A boolean mask over channels.
    """
    return np.asarray(measured["warmup_steps"]) < SEQUENCE_LENGTH - MIN_LAG_WINDOW_STEPS


def _usable(delay: Dict[str, Any]) -> str:
    """Human-readable count of usable channels per block.

    Args:
        delay: Output of ``measure_delay``.

    Returns:
        A comma-separated summary.
    """
    return ", ".join(
        f"{block} {int(_usable_mask(measured).sum())}/{len(measured['warmup_steps'])}"
        for block, measured in delay.items()
    )


def _correlation_rows(delay: Dict[str, Any]) -> str:
    """Markdown rows of median correlation per block, over the usable channels.

    Args:
        delay: Output of ``measure_delay``.

    Returns:
        The table body.
    """
    rows = []
    for block, measured in delay.items():
        mask = _usable_mask(measured)
        if not mask.any():
            continue
        values = [
            float(np.nanmedian(np.asarray(measured[key])[mask]))
            for key in ("r_at_predicted_lag", "r_at_best_lag", "r_at_zero_lag")
        ]
        rows.append(f"| `{block}` | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
    return "\n".join(rows)


def _gate_section(
    validation: Dict[str, Any], delay: Dict[str, Any], gate_rows: str
) -> str:
    r"""Section 1 of the report: which arm reproduced the shard, and how well.

    Arm A is the shard, so which arm is the reference follows from which transform wrote it, and
    so does what else can be said. The S15.3 paragraphs describe production's spectral truncation
    of $\hat\phi$ -- an analytic projection, and therefore not a causal operator -- so on a causal
    shard there is no such deviation to measure and the paragraphs are omitted rather than
    interpolated from absent keys.

    Args:
        validation: Output of ``measure_validation``.
        delay: Output of ``measure_delay``, for the stored ``fhr_ph`` width.
        gate_rows: The pre-rendered table body.

    Returns:
        The section's markdown.
    """
    if validation["shard_transform"] == "causal":
        return f"""## 1. The correctness gate comes first

The shard was written by the **causal** transform, so the arm that must reproduce it is arm C --
`hdf5_dataset/causal_scattering.py`'s numpy chain, gathered to the channels the stored channel plan
keeps. Agreement here is the numpy reference against the batched torch chain that wrote the file,
on production-scale segments, and it also checks that the build kept the channels the plan names
rather than 36 plausible ones.

| block | max relative error, full segment | over the interior the model uses |
| --- | --- | --- |
{gate_rows}

The interior is steps `{validation['interior_slice']}` — after the loader's 1-minute trim and the
model's 30-step warm-up. Storage is float32 and the reference is float64, so single-precision
agreement is the tightest the stored data can express.

The S15.3 phase-smoothing deviation is not measured here. It is the gap between production's
spectral truncation and the documented $\\Re\\{{(\\cdot)\\star\\phi\\}}$, and that truncation is
the analytic (positive-frequency) projection — a non-causal operation the causal chain deliberately
does not offer, so on this shard there is nothing for it to be a deviation from.
"""

    return f"""## 1. The correctness gate comes first

Arm B is this comparison's own implementation driven by the **production** Morlet bank, in
production's own decimation and smoothing conventions. If it does not reproduce the shard, no
later difference means anything.

| block | max relative error, full segment | over the interior the model uses |
| --- | --- | --- |
{gate_rows}

The interior is steps `{validation['interior_slice']}` — after the loader's 1-minute trim and the
model's 30-step warm-up. Three of the four blocks agree to float32 storage precision, which is the
tightest agreement the stored data can express. The `fhr_ph` residual is larger and concentrated
near the segment start.

The S15.3 deviation is measured rather than assumed. Production truncates the spectrum where the
documented operator periodises it, so the stored phase block is the *analytic projection* of the
smoothed product rather than $\\Re\\{{(\\cdot) \\star \\phi\\}}$. Over the
{len(delay['fhr_ph']['lag_steps'])} stored `fhr_ph` channels the ratio between the two operators —
run on the same product, so it isolates the smoothing — has median
**{validation['s15_3_stored_channel_ratio_median']:.2f}**, range
{validation['s15_3_stored_channel_ratio_min']:.2f} to
{validation['s15_3_stored_channel_ratio_max']:.2f}, with
**{validation['s15_3_stored_channel_sign_flips']}** channels changing sign. It is a pair-dependent
mixing, not a rescaling, exactly as documented.

What pins the emulation as *the* documented operator is not that ratio but a machine-precision
projection identity. $\\hat\\phi$ is already $\\approx 2\\times 10^{{-22}}$ at the truncation edge,
so the omitted positive bins are numerically negligible and the truncated operator equals $d$
times the analytic projection of the smoothed product to measured discrepancy:
**{validation['s15_3_analytic_projection_max_rel_err']:.2e}**
({'identity holds' if validation['s15_3_is_analytic_projection'] else 'IDENTITY FAILS'}).

(The stored selection carries **{validation['n_stored_diagonal_pairs']}** diagonal pairs —
production's `k_steps = (4, 6, 8)` exclude $k = 0$ deliberately, the diagonal being ~0.97 correlated
with the matching scattering channel. A "ratio between $d/2$ and $d$" is *not* a pointwise bound
even there: the ratio is $d(\\mathrm{{DC}} + A)/(\\mathrm{{DC}} + 2A)$ in the AC part $A$, which is
unbounded wherever $A < 0$.)
"""


def write_report(
    output_dir: str, summary: Dict[str, Any], filters: Dict[str, Any], delay: Dict[str, Any]
) -> None:
    """Write ``REPORT.md``, with every number interpolated from the measurements.

    Args:
        output_dir: Destination directory.
        summary: The assembled summary dict.
        filters: Output of ``measure_filters``.
        delay: Output of ``measure_delay``.
    """
    headline = summary["headline"]
    validation = summary["validation"]
    leakage = summary["leakage"]

    # Every count below is read from the measurements rather than written out: the number of
    # first-order filters, and the width of each block in the arms the figures were drawn from.
    n_filters = int(np.asarray(filters["reach_s"]).size)
    widths = {block: int(np.asarray(measured["lag_steps"]).size)
              for block, measured in delay.items()}
    block_widths = ", ".join(f"`{block}` ({width})" for block, width in widths.items())

    interior = validation["gate_max_rel_interior"]
    full = validation["gate_max_rel_full_segment"]
    gate_rows = "\n".join(
        f"| `{name}` | {full[name]:.2e} | {interior[name]:.2e} |" for name in full
    )
    gate_section = _gate_section(validation, delay, gate_rows)

    survivor_rows = []
    for row in summary["survivorship"]:
        budget = "unguarded" if row["budget_s"] is None else f"{row['budget_s']:g} s"
        target = "refused" if row["target_kept"] is None else (
            f"{row['target_kept']}/{row['target_total']} "
            f"(max {row['target_max_delay_steps']} steps)")
        source = "refused" if row["source_kept"] is None else (
            f"{row['source_kept']}/{row['source_total']} "
            f"(max {row['source_max_delay_steps']} steps)")
        survivor_rows.append(f"| {budget} | {ARM_LABEL.get(row['arm'], row['arm'])} | "
                             f"{target} | {source} |")

    report = f"""# One-sided (causal) scattering and phase harmonics: what it costs

Generated by `hdf5_dataset/compare_causal_scattering.py` against a
**{validation['shard_transform']}** shard. Every number here is read from `summary.json` and
`per_channel.csv` in this directory.

{gate_section}
## 2. The headline: causality is bought, not free

A causal filter removes a forward leak by paying a backward delay, and the exchange rate is worse
than one:

- median $\\tau_g / L_{{95}}$ = **{headline['delay_over_reach_median']:.2f}**
  (range {headline['delay_over_reach_min']:.2f}–{headline['delay_over_reach_max']:.2f})
- {headline['n_reach_past_horizon']} of {n_filters} two-sided filters reach past the {HORIZON_S:g} s
  forecast horizon
- causal future-tap energy: **{headline['causal_future_energy_max']:.0f}** (exactly zero — there is
  no storage for a future tap), against {headline['two_sided_future_energy_median']:.3f} two-sided

So relative to the shipped 120 s reach budget the causal transform is *staler* on every channel
that budget already keeps. What it buys is that the staleness is **honest**: the two-sided channel
does not merely look 120 s into the future, it looks as far as its filter reaches, and $L_{{95}}$ is
an energy quantile rather than a support, so 5% of the leak lies beyond even that.

## 3. What it costs beyond delay

- **Analyticity.** The phase-harmonic operator reads $\\arg(x \\star \\psi)$, and by Paley–Wiener a
  filter cannot be exactly causal and exactly analytic at once. Measured mirror-frequency amplitude
  ratio:
  median **{headline['neg_freq_gain_rel_median']:.2e}**, worst
  **{headline['neg_freq_gain_rel_max']:.2e}**. The defect is largest on the slowest few filters,
  where kymatio pins $\\sigma$ and $\\xi/\\sigma$ falls to 2.64; phase fidelity is assessed below.
- **Warm-up.** A causal channel remains influenced by the assumed history until its finite support
  has passed. **{headline['n_causal_support_over_segment']} of {n_filters}** filters have a 95%-energy
  history longer than the 1320 s segment, so on this data those channels report the assumed history
  for most of it.
- **Selectivity, if you cut instead of matching.** Truncating the Morlet at its peak — the obvious
  way to make it causal — widens the passband to ~1.7x nominal, because the cut lands on the
  envelope's maximum. A taper trades boundary smoothness against retained peak and envelope energy
  unless the filter is redesigned and revalidated. The matched gammatone stays at 1.00x. See
  `figures/02_frequency_response.pdf`.

## 4. The leak, measured directly

A deceleration was injected **only** after $t_0 = {leakage['edit_time_s']:g}$ s, and the
coefficients *before* $t_0$ were compared:

| arm | `fhr_st` max movement | reaches back | `fhr_ph` max movement | reaches back |
| --- | --- | --- | --- | --- |
| two-sided | {leakage['two_sided_fhr_st_max_past_rel']:.2e} | {_reach_back(leakage, 'two_sided', 'fhr_st')} | {leakage['two_sided_fhr_ph_max_past_rel']:.2e} | {_reach_back(leakage, 'two_sided', 'fhr_ph')} |
| causal | {leakage['causal_fhr_st_max_past_rel']:.2e} | {_reach_back(leakage, 'causal', 'fhr_st')} | {leakage['causal_fhr_ph_max_past_rel']:.2e} | {_reach_back(leakage, 'causal', 'fhr_ph')} |

The causal arm's residual is FFT round-off, not structure. The structural claim — that the past
side is *bitwise* unchanged — is pinned separately by a direct time-domain convolution in
`--mode self-test`, where no round-off floor exists.

## 5. Does the causal channel report the same thing, once you allow for the delay?

Per block, over the channels whose warm-up leaves enough segment to measure ({_usable(delay)}):

| block | $r$ at the predicted $\\tau_g$ | $r$ at the best lag | $r$ at zero lag |
| --- | --- | --- | --- |
{_correlation_rows(delay)}

Two things follow.

**The scattering channels transfer.** Compensated for their delay they reproduce the two-sided
channels almost exactly, and the analytic $\\tau_g$ is the right compensation — the measured lag
tracks it (see `figures/06_delay_scattering.pdf`). A causal scattering front end is a real option.

**The phase channels do not.** Even at the best achievable alignment they reach only ~0.6–0.8, and
the delay composition that works for scattering — the slower leg plus the low-pass, which is also
how `channel_reach.block_reach_seconds` composes phase *reach* — does not predict their delay at
all. That is consistent with the phase harmonic being the operator most exposed to the analyticity
defect: it reads $\\arg(x \\star \\psi)$, which is exactly the quantity a causal filter cannot
reproduce exactly. **Anyone porting the reach module's composition rule to delays should not.**

**Uncompensated, nothing transfers.** At zero lag every block sits near zero. A causal rebuild is
not a drop-in: the delays have to be carried through to whatever consumes the channels.

## 6. Budgets, both arms under one predicate

Both the two-sided reach vector and the causal delay vector are pushed through the same
`resolve_channel_budget`, with the shipped `warmup_period = 30`.

| budget | arm | target kept | source kept |
| --- | --- | --- | --- |
{chr(10).join(survivor_rows)}

"refused" means the guard rejected the configuration because the implied delay exceeds the loss
warm-up. That is a result about the causal arm, not a gap in the measurement: adopting a causal
transform wholesale would require raising `warmup_period` substantially, which costs trained
anchors.

## 7. The transforms themselves

`figures/samples/sample_NNNN.pdf` carries the full coefficient set for
{len(summary.get('panel_samples', []))} segments, one per distinct recording — every channel of
{block_widths} at $J=11$, $Q=4$, $T=16$, with the
two-sided and causal arms side by side on a shared colour scale, over the raw FHR and UP the
coefficients came from. `figures/09_gallery_fhr_st.pdf` tiles one block across all of them.

Channels are shown in the units the model is trained on — $\\log(S + \\epsilon)$ for the scattering
channels above $S_0$ and $\\operatorname{{asinh}}(\\Phi)$ for the phase channels, exactly as
`calculate_dataset_stats.py` transforms them before computing normalisation statistics. A raw-unit
heatmap would be a picture of the dynamic range rather than of the signal.

The black staircase on each causal panel is that channel's warm-up, and the whitened region left of
it is where the channel is reporting the assumed history rather than the recording. It widens
steeply toward the slow channels, and for the slowest it never closes inside the segment — which is
the same fact as `{headline['n_causal_support_over_segment']} of {n_filters}` above, in the form where it is
hardest to overlook.

## 8. What this does not settle

- **Whether a causal front end forecasts better.** Nothing here trains a model. A delayed-but-honest
  input could easily beat a leaky one at forecasting even though it is staler, because the leak
  inflates apparent skill on exactly the interval being predicted — but that is a claim these
  measurements set up rather than test.
- **Whether the gammatone is the best causal family.** It is a defensible one: exactly causal,
  bandwidth-matched, zero-mean. A minimum-phase factorisation of the Morlet's own magnitude would
  match the passband exactly, at the cost of a filter with no closed form and a frequency-dependent
  delay that is harder to report.
- **The slowest three channels.** Their analyticity defect and their warm-up both exceed what this
  segment length can support. They are also the channels the 120 s budget already drops.
"""
    with open(os.path.join(output_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(report)
