r"""The one instrument whose dependence is planted in a raw signal and read through the real bank.

Every other generator writes stored coefficients directly, which is what lets it ask a clean
question about the architecture. This one asks the question the others structurally cannot: what
happens to a known source-to-target delay **after** the causal feature operator has summarised both
streams into coefficients?

The answer is not a detail. Each coefficient is the output of a filter whose support is long, so a
dependence planted at one raw instant reaches the stored grid spread across many stored steps -- a
spread the feature geometry fixes and no model-side change touches. A pointwise source encoder does
not undo it, and neither does an attention over lags. So this instrument's criterion is
**relevance**, not recovery: does the readout detect that the source helps at all, and how far does
its profile sit from where the plant actually is? The distance is the measurement. A readout that
peaked exactly on the planted band here would be the surprising result.

**Nothing is fabricated and nothing is written to disk.** The raw pair is synthesised, handed to the
production filter bank in memory, and the resulting coefficients carry the bank's own channel plan
-- its widths, its per-channel warm-up boundaries and its group delays. The warm-up budget is then
resolved from that plan by the same rule a run resolves it from a shard: rebase for the trim, keep
the channels that clear the budget. That is why this module builds no shard: a shard would have to
declare a warm-up boundary, and here the real one is available.

**The standardisation is the dataset's own fixed transforms**, applied before the channelwise
statistics: the first scattering channel stays linear, the rest are logged, the phase channels pass
through an inverse hyperbolic sine. The statistics are computed over each channel's warm region
only, which is the same exclusion a real normalisation uses and the reason a standardized zero is
the channel mean over the region the model reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from teb_vae.lag_slot_transformer_cfs.instruments.generators import (
    GeneratedBatch,
    Generator,
    InstrumentGeometry,
    Truth,
    direct_support,
)

#: Raw samples per stored step, which is the operator's own decimation.
DECIMATION = 16

#: Raw sampling rate, in hertz.
RAW_HZ = 4.0

#: Untrimmed stored steps the bank is sized for.
#:
#: The smallest length the production filter bank accepts: below it the bank's own normalisation
#: refuses, because the longest filter no longer fits the signal. That refusal is the bank's and is
#: left where it is -- an instrument that shrank the geometry until the bank stopped complaining
#: would be measuring a different operator.
UNTRIMMED_STEPS = 160

#: Stored steps removed from each end, matching the loader's own one-minute trim.
TRIM_STEPS = 15

#: Warm-up budget in untrimmed stored steps: a channel survives when its wait clears it.
#:
#: Chosen so that both streams keep their fast channels and drop the slow ones, which is the state
#: a production run is in. A budget that kept everything would leave the instrument reading
#: channels whose warm-up outlasts the record.
BUDGET_STEPS = 60

#: The planted source-to-target delay, in stored steps.
#:
#: Strictly above the horizon, so its readable band clears lag zero and a peak in the band is
#: distinguishable from a profile pinned at the near censoring edge.
PLANTED_DELAY_STEPS = 12

#: The slow drive's component frequencies, in hertz, and the faster carrier riding on it.
#:
#: Well below the bank's own low-pass so the envelope survives the transform, and mutually
#: irrational so no two components share a harmonic and a channel that hears one does not
#: automatically hear another.
ENVELOPE_HZ: Tuple[float, ...] = (0.00375, 0.00611, 0.00893, 0.01249)
CARRIER_HZ = 0.09
CARRIER_DEPTH = 0.25

#: The target's own uncoupled content, which the source explains none of. Without it the target is
#: a pure function of the source and any readout would find the plant trivially.
CONTROL_HZ: Tuple[float, ...] = (0.047, 0.076, 0.123, 0.199, 0.322)
CONTROL_AMPLITUDE = 4.0

#: Baselines and gains of the two synthesised signals, in their own physical units.
UP_BASE, UP_GAIN = 5.0, 70.0
FHR_BASE, FHR_GAIN = 145.0, 35.0

#: The floor inside the logarithm the dataset applies to every scattering channel but the first.
LOG_FLOOR = 1e-6

#: The denominator guard the dataset's channelwise standardisation carries.
SCALE_FLOOR = 1e-8


@dataclass(frozen=True)
class StreamChannels:
    """One stream's resolved channel plan, as the bank measured it and the budget cut it.

    Attributes:
        keep_index: Surviving channel indices into the declared width, ascending.
        warmup_steps: $W'_c$ per survivor, in **trimmed** stored steps.
        declared_width: Channels the stream declares before the budget.
    """

    keep_index: Tuple[int, ...]
    warmup_steps: Tuple[int, ...]
    declared_width: int


def synthesise_raw(
    segments: int, signal_len: int, delay_steps: int, seed: int
) -> Dict[str, np.ndarray]:
    r"""Two raw signals in which the target follows the source at a known delay.

    $$e_n(t) = \frac1K \sum_k \tfrac12\bigl(1 + \sin(2\pi f_k t + \varphi_{k,n})\bigr), \qquad
      u_n(t) = u_0 + g_u e_n(t)\bigl(1 + m\cos(2\pi f_{\mathrm{car}} t)\bigr),$$
    $$y_n(t) = y_0 - g_y\, e_n(t - \Delta) + a\sum_j \cos(2\pi f_j t + \psi_{j,n}).$$

    The delayed envelope is **evaluated at $t - \Delta$ as a function**, never shifted out of a
    buffer. A shifted buffer needs a fill for its leading region, and that fill is a seam the
    record does not declare -- which a filter bank would then summarise as a real event at the
    start of every segment.

    Args:
        segments: How many segments to synthesise.
        signal_len: Raw samples per segment.
        delay_steps: The planted delay in **stored** steps; the raw shift is that many times the
            decimation.
        seed: Seed for the per-segment phases, so a build is a function of its seed.

    Returns:
        ``{'fhr': (n, L) float32, 'up': (n, L) float32}``.

    Raises:
        ValueError: If the delay is not positive, or if the raw shift reaches beyond a segment --
            a plant longer than the record is one no lag search inside the record could find.
    """
    if int(delay_steps) <= 0:
        raise ValueError(
            f"delay_steps={delay_steps} must be positive: the plant is a source-to-target delay, "
            f"and a non-positive one asks the target to lead the source."
        )
    raw_shift = int(delay_steps) * DECIMATION
    if raw_shift >= int(signal_len):
        raise ValueError(
            f"delay_steps={delay_steps} is {raw_shift} raw samples against a {signal_len}-sample "
            f"segment, so the coupled source content lies outside every segment the model sees."
        )

    rng = np.random.default_rng(int(seed))
    seconds = np.arange(int(signal_len), dtype=np.float64) / RAW_HZ
    delayed = seconds - float(raw_shift) / RAW_HZ

    fhr = np.empty((int(segments), int(signal_len)), dtype=np.float64)
    up = np.empty((int(segments), int(signal_len)), dtype=np.float64)
    for index in range(int(segments)):
        envelope_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(ENVELOPE_HZ))
        control_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(CONTROL_HZ))
        now = _envelope(seconds, envelope_phases)
        then = _envelope(delayed, envelope_phases)
        carrier = 1.0 + CARRIER_DEPTH * np.cos(2.0 * np.pi * CARRIER_HZ * seconds)
        control = np.zeros_like(seconds)
        for frequency, phase in zip(CONTROL_HZ, control_phases):
            control += CONTROL_AMPLITUDE * np.cos(2.0 * np.pi * frequency * seconds + phase)
        up[index] = UP_BASE + UP_GAIN * now * carrier
        fhr[index] = FHR_BASE - FHR_GAIN * then + control
    return {"fhr": fhr.astype("f4"), "up": up.astype("f4")}


def _envelope(seconds: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """The slow drive of one segment, in $[0, 1]$, at the given times.

    Args:
        seconds: Times to evaluate at. May be negative, which is what lets the delayed copy be
            evaluated rather than shifted.
        phases: One phase per component.

    Returns:
        The envelope, same shape as ``seconds``.
    """
    total = np.zeros_like(seconds, dtype=np.float64)
    for frequency, phase in zip(ENVELOPE_HZ, phases):
        total += 0.5 * (1.0 + np.sin(2.0 * np.pi * frequency * seconds + phase))
    return total / float(len(ENVELOPE_HZ))


def bank_blocks(
    raw: Dict[str, np.ndarray], untrimmed_steps: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Run the production causal filter bank over a batch of raw segments.

    The one place this package touches the feature pipeline, and it touches the pipeline itself
    rather than a copy of its arithmetic: what an instrument claims about delay spread is a claim
    about *that* operator, so a second implementation of the bank would make the finding a claim
    about a reimplementation.

    Args:
        raw: ``{'fhr', 'up'}`` segments on the raw grid.
        untrimmed_steps: Stored steps the bank is sized for; the segments must be that many times
            the decimation long.

    Returns:
        ``(blocks, channel_plan)``: the four coefficient blocks, each $(n, C, T)$, and the bank's
        own per-block channel plan carrying each channel's warm-up and delay.
    """
    from hdf5_dataset.causal_scattering_torch import CausalTorchBank, transform_batch_numpy
    from hdf5_dataset.smoke_check_channel_selection import _import_pipeline

    pipeline = _import_pipeline()
    signal_len = int(untrimmed_steps) * DECIMATION
    device = torch.device("cpu")
    masks = pipeline.compute_scattering_masks(
        signal_len,
        scattering_T=DECIMATION,
        device=device,
        transform="causal",
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
    )
    blocks = transform_batch_numpy(
        CausalTorchBank(masks["causal_bank"], device, n_signal=signal_len),
        raw["fhr"],
        raw["up"],
        pipeline._selection_pairs(masks["fhr_ph_selection"]),
        pipeline._selection_pairs(masks["up_ph_selection"]),
        plan=masks["channel_plan"],
        leg_alignment="envelope",
        phase_operator="integer_harmonic_v1",
    )
    return blocks, masks["channel_plan"]


def resolve_stream(
    plan: Dict[str, Any],
    block_names: Sequence[str],
    *,
    trim_steps: int,
    budget_steps: int,
) -> StreamChannels:
    r"""Which channels of one stream survive the budget, and how long each survivor waits.

    Two rules, both the resolver's own. The trim rebases every wait,
    $W'_c = \max(W_c - \text{trim}, 0)$, because the model reads the trimmed window and a wait
    stated against the untrimmed one would announce a channel as cold for longer than it is. The
    budget then keeps the channels whose **untrimmed** wait clears it, which is the coordinate a
    configured budget is stated in.

    Args:
        plan: The bank's channel plan, keyed by block name.
        block_names: The stream's blocks, in the declared concatenation order.
        trim_steps: Stored steps removed from each end.
        budget_steps: The budget, in untrimmed stored steps.

    Returns:
        The resolved stream.

    Raises:
        ValueError: If the budget keeps no channel of the stream, which is a geometry that cannot
            be fitted rather than a run that would report nothing.
    """
    waits: List[int] = []
    for name in block_names:
        waits.extend(int(step) for step in plan[name].warmup_steps)
    keep = tuple(index for index, wait in enumerate(waits) if wait <= int(budget_steps))
    if not keep:
        raise ValueError(
            f"a budget of {budget_steps} untrimmed stored steps keeps no channel of blocks "
            f"{tuple(block_names)}, whose waits run from {min(waits)} to {max(waits)}. Raise the "
            f"budget or lengthen the record; a stream with no channel is not a stream."
        )
    return StreamChannels(
        keep_index=keep,
        warmup_steps=tuple(max(waits[index] - int(trim_steps), 0) for index in keep),
        declared_width=len(waits),
    )


def standardise(block: np.ndarray, *, logarithmic: bool, warm_from: Sequence[int]) -> np.ndarray:
    r"""Apply the dataset's fixed transform, then standardise each channel over its warm region.

    $$\bar x_c = \frac{g_c(x_c) - m_c}{s_c + 10^{-8}},$$

    with $g_c$ the logarithm on every scattering channel but the first, the inverse hyperbolic sine
    on a phase channel, and the statistics taken over the steps at or after that channel's own
    warm-up. Excluding the cold region is not a refinement: those coefficients are a filter running
    on assumed history, and statistics that included them would put the model's standardized zero
    somewhere no observation is.

    Args:
        block: One stored block $(n, C, T)$, untrimmed.
        logarithmic: Whether this is a scattering block, whose channels after the first are logged.
        warm_from: Each channel's own warm-up step, positional against the channel axis.

    Returns:
        The standardized block, same shape.
    """
    values = np.asarray(block, dtype=np.float64)
    transformed = np.empty_like(values)
    for channel in range(values.shape[1]):
        column = values[:, channel, :]
        if logarithmic and channel > 0:
            transformed[:, channel, :] = np.log(np.maximum(column, 0.0) + LOG_FLOOR)
        elif logarithmic:
            transformed[:, channel, :] = column
        else:
            transformed[:, channel, :] = np.arcsinh(column)

    standardized = np.empty_like(transformed)
    for channel in range(values.shape[1]):
        start = min(int(warm_from[channel]), values.shape[2] - 1)
        warm = transformed[:, channel, start:]
        mean, scale = float(warm.mean()), float(warm.std())
        standardized[:, channel, :] = (transformed[:, channel, :] - mean) / (scale + SCALE_FLOOR)
    return standardized


def raw_geometry(plan: Dict[str, Any]) -> InstrumentGeometry:
    """The stored grid this instrument runs at, read off the bank rather than declared.

    The widths are the bank's; the anchor floor is set above the slowest surviving target wait so
    the constructor's own geometry check passes, and the lag window is wide enough to hold the
    planted band with room on both sides.

    Args:
        plan: The bank's channel plan.

    Returns:
        The geometry.
    """
    target = resolve_stream(
        plan, ("fhr_st", "fhr_ph"), trim_steps=TRIM_STEPS, budget_steps=BUDGET_STEPS
    )
    return InstrumentGeometry(
        sequence_length=UNTRIMMED_STEPS - 2 * TRIM_STEPS,
        horizon=4,
        warmup_period=max(max(target.warmup_steps), 2 * PLANTED_DELAY_STEPS),
        max_lag=2 * PLANTED_DELAY_STEPS - 1,
        n_target_scattering=int(plan["fhr_st"].n_channels),
        n_target_phase=int(plan["fhr_ph"].n_channels),
        n_source=int(plan["up_st"].n_channels) + int(plan["up_ph"].n_channels),
        # The same three splits every other instrument runs. Fewer rows than a synthetic one
        # carries, because each row here costs a pass of the real filter bank rather than a
        # draw from a generator.
        segments=40,
        selection=10,
        holdout=16,
    )


def raw_instrument() -> Generator:
    """Build the filter-bank instrument, resolving its geometry from the bank once.

    The bank is run once here for its channel plan, which is a deterministic function of the
    geometry and the operator and not of any seed, and again inside every build for that build's
    own segments. Resolving the plan twice would let the declared widths and the coefficients
    describe two different operators.

    Returns:
        The generator, carrying its own geometry and the channel tuples the model is built with.
    """
    probe = synthesise_raw(1, UNTRIMMED_STEPS * DECIMATION, PLANTED_DELAY_STEPS, seed=0)
    _blocks, plan = bank_blocks(probe, UNTRIMMED_STEPS)
    geometry = raw_geometry(plan)
    target = resolve_stream(
        plan, ("fhr_st", "fhr_ph"), trim_steps=TRIM_STEPS, budget_steps=BUDGET_STEPS
    )
    source = resolve_stream(
        plan, ("up_st", "up_ph"), trim_steps=TRIM_STEPS, budget_steps=BUDGET_STEPS
    )

    def build(_geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
        """Synthesise, transform, trim and standardise one build's segments.

        Args:
            _geometry: Ignored; this instrument's grid is the bank's and travels on the generator.
            seed: The build's seed.

        Returns:
            The batch.
        """
        raw = synthesise_raw(
            geometry.rows, UNTRIMMED_STEPS * DECIMATION, PLANTED_DELAY_STEPS, seed
        )
        blocks, build_plan = bank_blocks(raw, UNTRIMMED_STEPS)
        pieces = {
            name: standardise(
                blocks[name],
                logarithmic=name.endswith("_st"),
                warm_from=[int(step) for step in build_plan[name].warmup_steps],
            )
            for name in ("fhr_st", "fhr_ph", "up_st", "up_ph")
        }
        # Trimmed after standardising, so each channel's statistics are taken over the same region
        # a real normalisation takes them over -- the untrimmed warm region -- and only then is the
        # window the model reads cut out of it.
        cut = slice(TRIM_STEPS, UNTRIMMED_STEPS - TRIM_STEPS)
        as_stream = {
            name: torch.from_numpy(piece[:, :, cut]).permute(0, 2, 1).contiguous().float()
            for name, piece in pieces.items()
        }
        return GeneratedBatch(
            y_st=as_stream["fhr_st"],
            y_ph=as_stream["fhr_ph"],
            u_stream=torch.cat([as_stream["up_st"], as_stream["up_ph"]], dim=-1),
            weight=torch.ones(geometry.rows, geometry.sequence_length),
        )

    band = direct_support([PLANTED_DELAY_STEPS], geometry.horizon, geometry.n_lags)
    return Generator(
        name="raw_filtered",
        build=build,
        truth=Truth(
            source_informative=True,
            direct_support=band,
            causal=True,
            recovery_expected=False,
            note=(
                "the delay is planted in the RAW signals and reaches the stored grid through the "
                "production filter bank, so every coefficient summarises a window far longer than "
                "one stored step. The band above is where the plant is, not where the readout "
                "should peak: the distance between them is the delay spread the feature geometry "
                "imposes, and it is the measurement. A profile peaking exactly on the band here "
                "would be the surprising result."
            ),
        ),
        checks="end-to-end delay spread and preprocessing causality",
        geometry=geometry,
        model_kwargs={
            "target_keep_index": target.keep_index,
            "target_warmup_steps": target.warmup_steps,
            "source_keep_index": source.keep_index,
            "source_warmup_steps": source.warmup_steps,
            "c_y": target.declared_width,
            "c_u": source.declared_width,
        },
    )


__all__ = [
    "BUDGET_STEPS",
    "DECIMATION",
    "PLANTED_DELAY_STEPS",
    "TRIM_STEPS",
    "UNTRIMMED_STEPS",
    "StreamChannels",
    "bank_blocks",
    "raw_geometry",
    "raw_instrument",
    "resolve_stream",
    "standardise",
    "synthesise_raw",
]
