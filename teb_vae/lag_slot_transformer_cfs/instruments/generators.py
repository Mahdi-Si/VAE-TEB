r"""The generators, and the ground truth each one is declared with.

Every generator here writes the **stored feature streams directly**: it produces the coefficient
tensors the model reads, rather than a raw signal and a filter bank. That is a decision with a
reason, not a shortcut.

**Why the synthetic instruments do not write a shard.** A stored-feature block synthesised here has
no filter bank behind it, so it has no real warm-up boundary, no real group delay and no real
novelty curve. Written into a causal shard it would carry those attributes anyway -- fabricated --
and the warm-up resolver reads exactly them. Every downstream number would then be computed against
a boundary nobody measured. So these generators declare what they are: standardized streams on an
untrimmed grid, every channel valid from the first step, and the campaign says so in its record.
What they measure is whether the architecture and its readouts recover a dependence **present in
the tensors the model reads**, which is the question they exist for. Warm-up handling, trimming and
normalisation provenance are covered by the suites that own them.

The one generator that does have a filter bank behind it lives beside this module and is built from
a raw process through the real causal operator, where the warm-up is real and is read off the bank.

**The common skeleton, and why every generator shares it.** Each builds a source stream, forms a
scalar driver from it (or from a hidden process), and drives a target autoregression:

$$u_{s,j} \sim \text{a declared source law}, \qquad
  y_s = \phi\, y_{s-1} + \beta\, D_s + \varepsilon_s,$$

with $D_s$ the generator's own driver. The target is then broadcast into channels at per-channel
gains with independent channel noise, and the source into channels of which only a declared few
carry the driver. Sharing the skeleton is what makes the generators comparable: two of them differ
in $D$ and in nothing else, so a difference between their measured rates is a difference of
dependence structure rather than of signal-to-noise.

**The autoregression is strong on purpose.** The failure mode these instruments exist to detect is a
source that appears to help because the baseline was weak. A target the base branch cannot predict
at all would manufacture exactly that, in every generator including the ones declared to carry no
source information -- so $\phi$ is large enough that a target-only forecaster is genuinely capable,
and the false-positive rates below mean something as a result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import torch

#: Autoregressive coefficient of the target process, shared by every generator.
#:
#: Large enough that a target-only forecaster is genuinely capable, which is what makes a
#: false-positive rate a measurement rather than a statement about a weak baseline.
TARGET_AR = 0.85

#: Weight the driver enters the target with. One value across every generator, so two generators
#: differ in the SHAPE of their dependence and not in its strength.
DRIVER_GAIN = 0.6

#: Standard deviation of the target's own innovation, and of the channel noise added on top.
TARGET_NOISE = 0.35
CHANNEL_NOISE = 0.15

#: How many source channels carry the driver. A handful rather than all of them, matching the
#: shape of a real source block where most channels say nothing about a given dependence.
INFORMATIVE_SOURCE_CHANNELS = 3


class GeneratedBatch(NamedTuple):
    r"""One generator's output, in the shapes the model's forward takes.

    Attributes:
        y_st: The target's first stored block $(B, T, \cdot)$.
        y_ph: The target's second stored block $(B, T, \cdot)$, concatenated after the first in the
            declared channel order.
        u_stream: The source stream $(B, T, C_U)$.
        weight: The decimated validity signal $(B, T)$, which the forecast mask is built from.
    """

    y_st: torch.Tensor
    y_ph: torch.Tensor
    u_stream: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class InstrumentGeometry:
    r"""The stored grid every generator writes on, and the anchor set the model decodes.

    Small in every width and faithful in the two things the instruments read: a lag window wide
    enough to hold a planted delay strictly inside it, and an anchor floor above that delay so the
    earliest anchor can still reach the source the plant refers to.

    Attributes:
        sequence_length: Stored steps $T$.
        horizon: Future steps per forecast $H$.
        warmup_period: The anchor floor $F$.
        max_lag: Furthest candidate lag, so $L = \texttt{max\_lag} + 1$.
        n_target_scattering: Channels in the target's first stored block.
        n_target_phase: Channels in its second.
        n_source: Source channels $C_U$.
        segments: Segments the model is **fitted** on.
        selection: Segments the fit is **stopped** on, disjoint from both other splits.
        holdout: Segments the verdicts are **scored** on, disjoint from both other splits.

    **Three splits and not two, and the third is not a refinement.** A gap read on the segments a
    model was fitted to measures how well it memorised them, and the source branch has strictly more
    capacity to memorise with than the target-only branch -- so an in-sample instrument reports a
    positive gap on a generator whose source carries nothing, which is exactly the false positive it
    exists to detect. That is what the second split fixes. The third fixes the next one down: a fit
    stopped at whichever step scored best on the set it is then reported on has had its stopping
    point chosen by the number it reports, and the rate would be a measurement of that choice.

    **The fit split is the largest of the three, and its size was measured rather than chosen.**
    At a third of it the fit memorises before its source pathway has learned anything: on a planted
    generator the gap is positive early and then decays through zero as the decoder drives its
    observation variance down, and the instrument has no power at all -- for a reason about the
    data volume rather than about the readout it exists to measure.
    """

    sequence_length: int = 96
    horizon: int = 4
    warmup_period: int = 40
    max_lag: int = 23
    n_target_scattering: int = 6
    n_target_phase: int = 4
    n_source: int = 8
    segments: int = 64
    selection: int = 16
    holdout: int = 24

    @property
    def rows(self) -> int:
        """Segments a builder writes: the three splits together."""
        return int(self.segments) + int(self.selection) + int(self.holdout)

    @property
    def n_lags(self) -> int:
        """$L$, the candidate lag count."""
        return int(self.max_lag) + 1

    @property
    def n_target(self) -> int:
        """Declared target channels, both blocks."""
        return int(self.n_target_scattering) + int(self.n_target_phase)


@dataclass(frozen=True)
class Truth:
    r"""What is actually true of one generator, declared before it is ever fitted.

    Attributes:
        source_informative: Whether the source carries information about the future target **beyond
            the available target history**. This is the conditional quantity, not a marginal
            correlation: a source that is a deterministic function of the target's own past is
            marginally informative and conditionally worthless, and the whole point of the
            redundant generator is that a readout must not confuse the two.
        direct_support: The inclusive lag band a planted delay is directly readable in, or ``None``
            where the generator plants no single referenced source time. Derived from the
            referenced offsets and the horizon by :func:`direct_support`, never written down.
        causal: Whether a detection would also be a statement about the direction of the
            dependence. ``False`` on the common-driver generator, where the source genuinely
            predicts the target and causes none of it -- so a readout that fires there is
            **correct** and a reader who calls it causal is not.
        recovery_expected: Whether the band above is something a readout should be scored against.
            ``True`` wherever the generator plants its dependence directly in the stored features,
            which is every stored-feature instrument. ``False`` on the instrument whose dependence
            is planted in a **raw** process and reaches the stored grid through the real feature
            operator: each coefficient there summarises a window far wider than the searched lag
            axis, so the band is reported as the plant's own location and the distance from it is
            the measurement, rather than a criterion the readout is failed against.
        note: What a reader has to know about this generator that its flags do not say.
    """

    source_informative: bool
    direct_support: Optional[Tuple[int, int]]
    causal: bool
    note: str
    recovery_expected: bool = True


@dataclass(frozen=True)
class Generator:
    r"""One instrument: a name, a builder, and the truth it was declared with.

    Attributes:
        name: The instrument's name, which every record is keyed by.
        build: Produces one batch from a geometry and a seed.
        truth: What is true of it, declared here and read by the campaign from here.
        checks: What this instrument is a check on, in the specification's own words.
        geometry: The stored grid this instrument runs at, or ``None`` to use the campaign's. Set
            only by an instrument whose grid is fixed by something other than a choice -- the one
            whose coefficients come from the real filter bank, whose widths and warm-up boundaries
            are the bank's rather than a setting.
        model_kwargs: Extra constructor keywords the instrument's own data requires, or ``None``.
            The channel tuples of the filter-bank instrument arrive this way, because they are a
            measurement of the bank and not a declaration.
    """

    name: str
    build: Callable[[InstrumentGeometry, int], GeneratedBatch]
    truth: Truth
    checks: str
    geometry: Optional[InstrumentGeometry] = None
    model_kwargs: Optional[Dict[str, object]] = None


def direct_support(
    offsets: Sequence[int], horizon: int, n_lags: int
) -> Optional[Tuple[int, int]]:
    r"""The lag band a set of directly referenced source offsets is readable in.

    For a process whose target at stored step $s$ references the source at $s - d$, the label at
    horizon $h$ of an anchor at $t$ is the target at $t + h$, which references the source at
    $t + h - d$ -- that is, at lag $\ell = d - h$. Pooling over the offsets $d \in \mathcal D$ and
    over $h \in \{1, \ldots, H\}$ gives

    $$\bigl[\min \mathcal D - H,\ \max \mathcal D - 1\bigr] \cap [0,\ L - 1].$$

    **This is direct structural support and not the whole of a predictive-importance profile.**
    Target propagation, source autocorrelation and the feature filtering all broaden a dependence
    across neighbouring lags, so a readout peaking just outside this band is not necessarily wrong.
    The band is what a recovery criterion is scored against because it is the only part of the
    profile the generator actually determines.

    Args:
        offsets: The referenced source offsets $\mathcal D$, in stored steps.
        horizon: $H$, the forecast length.
        n_lags: $L$, the candidate lag count.

    Returns:
        The inclusive band, or ``None`` when it falls entirely outside the searched window -- which
        is a property of the geometry rather than of the model, and which a recovery criterion must
        refuse to score rather than fail.

    Raises:
        ValueError: If no offsets are given. A band over an empty set of referenced times is not a
            band with no support; it is a question that was never asked.
    """
    if not len(offsets):
        raise ValueError(
            "direct_support needs at least one referenced source offset: a band over none is not "
            "an empty band, it is a generator that plants no delay and should declare None."
        )
    low = int(min(offsets)) - int(horizon)
    high = int(max(offsets)) - 1
    low, high = max(low, 0), min(high, int(n_lags) - 1)
    return None if low > high else (low, high)


# =================================================================================================
# The shared skeleton
# =================================================================================================
def _noise(shape: Tuple[int, ...], generator: torch.Generator, scale: float) -> torch.Tensor:
    """Seeded Gaussian noise at a declared scale.

    Args:
        shape: The tensor shape.
        generator: The seeded generator every draw in one build shares.
        scale: Standard deviation.

    Returns:
        The noise.
    """
    return torch.randn(*shape, generator=generator) * float(scale)


def _source_noise(
    geometry: InstrumentGeometry, generator: torch.Generator, *, correlated: bool = False
) -> torch.Tensor:
    r"""A source stream of independent channels, optionally autocorrelated in time.

    The autocorrelated variant matters to more than one instrument: a source whose neighbouring
    steps are nearly equal spreads a dependence across neighbouring lags whatever the model does,
    so an instrument that only ever used white source would report a lag resolution the physiology
    never offers.

    Args:
        geometry: The stored grid.
        generator: The seeded generator.
        correlated: Whether to pass the stream through a first-order recursion in time.

    Returns:
        The source stream $(B, T, C_U)$.
    """
    shape = (geometry.rows, geometry.sequence_length, geometry.n_source)
    stream = torch.randn(*shape, generator=generator)
    if not correlated:
        return stream
    smoothed = torch.empty_like(stream)
    smoothed[:, 0] = stream[:, 0]
    for step in range(1, geometry.sequence_length):
        smoothed[:, step] = 0.7 * smoothed[:, step - 1] + 0.7 * stream[:, step]
    return smoothed


def _driver_channels(geometry: InstrumentGeometry) -> slice:
    """Which source channels carry the driver.

    Args:
        geometry: The stored grid.

    Returns:
        The channel slice, the leading few of the block.
    """
    return slice(0, min(INFORMATIVE_SOURCE_CHANNELS, geometry.n_source))


def _read_driver(u_stream: torch.Tensor, geometry: InstrumentGeometry) -> torch.Tensor:
    r"""The scalar the informative source channels agree on, at every stored step.

    Averaged rather than summed so the driver's scale does not move with the number of informative
    channels, which would otherwise make two generators differ in signal strength as well as in
    structure.

    Args:
        u_stream: The source stream $(B, T, C_U)$.
        geometry: The stored grid.

    Returns:
        The driver $(B, T)$.
    """
    return u_stream[:, :, _driver_channels(geometry)].mean(dim=-1)


def _shift(values: torch.Tensor, offset: int) -> torch.Tensor:
    r"""Move a $(B, T)$ series ``offset`` steps later, filling the leading region with zeros.

    The fill is a declared boundary rather than a wrap: a wrapped series would put the end of the
    record at the start of it, and every lag readout would then find a dependence that the anchor
    floor was supposed to make unreachable.

    Args:
        values: The series $(B, T)$.
        offset: Steps to delay by; zero returns the input unchanged.

    Returns:
        The delayed series, same shape.
    """
    if int(offset) <= 0:
        return values
    delayed = torch.zeros_like(values)
    delayed[:, int(offset) :] = values[:, : -int(offset)]
    return delayed


def _autoregression(
    driver: torch.Tensor, geometry: InstrumentGeometry, generator: torch.Generator
) -> torch.Tensor:
    r"""Run the target's own recursion over a driver: $y_s = \phi y_{s-1} + \beta D_s + \varepsilon_s$.

    Args:
        driver: The generator's own driver $(B, T)$, already delayed where it should be.
        geometry: The stored grid.
        generator: The seeded generator.

    Returns:
        The scalar target process $(B, T)$.
    """
    innovation = _noise(
        (geometry.rows, geometry.sequence_length), generator, TARGET_NOISE
    )
    target = torch.empty_like(innovation)
    target[:, 0] = innovation[:, 0]
    for step in range(1, geometry.sequence_length):
        target[:, step] = (
            TARGET_AR * target[:, step - 1]
            + DRIVER_GAIN * driver[:, step]
            + innovation[:, step]
        )
    return target


def _as_batch(
    target: torch.Tensor,
    u_stream: torch.Tensor,
    geometry: InstrumentGeometry,
    generator: torch.Generator,
) -> GeneratedBatch:
    r"""Broadcast one scalar target process into the two stored blocks, and assemble the batch.

    Per-channel gains rather than one copy repeated: a block whose channels were identical would
    make the objective's channel weighting inert and would let a decoder fit every channel from any
    one of them. The gains alternate in sign so the plant is not a single common mode.

    The validity signal is all ones. The instruments read a lag profile, and a validity pattern
    would move the scored anchor set with it -- so the one thing they must not confound the readout
    with is the mask, which the suites that own masking already cover.

    Args:
        target: The scalar target process $(B, T)$.
        u_stream: The source stream.
        geometry: The stored grid.
        generator: The seeded generator.

    Returns:
        The batch.
    """
    channels = geometry.n_target
    index = torch.arange(channels, dtype=target.dtype)
    gains = (0.6 + 0.8 * index / max(channels - 1, 1)) * torch.where(
        index % 2 == 0, 1.0, -1.0
    )
    block = target.unsqueeze(-1) * gains[None, None, :] + _noise(
        (geometry.rows, geometry.sequence_length, channels), generator, CHANNEL_NOISE
    )
    return GeneratedBatch(
        y_st=block[:, :, : geometry.n_target_scattering].contiguous(),
        y_ph=block[:, :, geometry.n_target_scattering :].contiguous(),
        u_stream=u_stream,
        weight=torch.ones(geometry.rows, geometry.sequence_length),
    )


def _seeded(seed: int) -> torch.Generator:
    """One generator per build, so a build is a function of its seed and nothing else.

    Args:
        seed: The build's seed.

    Returns:
        The generator.
    """
    return torch.Generator().manual_seed(int(seed))


# =================================================================================================
# The generators
# =================================================================================================
#: The planted delay every single-delay instrument uses, in stored steps.
#:
#: Strictly above the horizon, which is what keeps its readable band clear of lag zero: a band
#: reaching the near edge would make "the readout found the plant" and "the readout is pinned at
#: the censoring edge" the same observation.
PLANTED_DELAY = 12

#: The second delay of the several-delays instrument, far enough from the first that the two are
#: separately resolvable at the geometry the campaign runs.
SECOND_DELAY = 20

#: Width of the broad response kernel, in stored steps.
KERNEL_WIDTH = 5


def _target_autoregression(
    geometry: InstrumentGeometry, seed: int
) -> GeneratedBatch:
    """A capable target autoregression with a source that does nothing at all.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator)
    target = _autoregression(torch.zeros(geometry.rows, geometry.sequence_length), geometry, generator)
    return _as_batch(target, u_stream, geometry, generator)


def _redundant_source(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""The source is a deterministic function of target history the model already has.

    Marginally the source predicts the future target very well; conditionally on the target's own
    permitted history it says nothing new. A readout that fires here is reporting redundancy as
    novelty, which is the second failure the evidence behind this architecture exhibits.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    target = _autoregression(
        torch.zeros(geometry.rows, geometry.sequence_length), geometry, generator
    )
    # The source at step $s$ is a function of the target at $s$ and $s-1$, both of which the
    # permitted history at any anchor that can read them already contains. No noise: a noisy copy
    # would carry a little conditional information and the generator would stop being a control.
    copied = torch.tanh(target) + 0.5 * _shift(target, 1)
    u_stream = _source_noise(geometry, generator)
    u_stream[:, :, _driver_channels(geometry)] = copied.unsqueeze(-1)
    return _as_batch(target, u_stream, geometry, generator)


def _single_delay(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    """One known delay, on a white source.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator)
    driver = _shift(_read_driver(u_stream, geometry), PLANTED_DELAY)
    return _as_batch(_autoregression(driver, geometry, generator), u_stream, geometry, generator)


def _several_delays(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    """Two planted delays at once, so the readout has competing source evidence to allocate.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator)
    source = _read_driver(u_stream, geometry)
    driver = 0.5 * (_shift(source, PLANTED_DELAY) + _shift(source, SECOND_DELAY))
    return _as_batch(_autoregression(driver, geometry, generator), u_stream, geometry, generator)


def _broad_kernel(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""A smooth response kernel over an autocorrelated source.

    The dependence is spread over a window of source times rather than concentrated at one, and the
    source itself is autocorrelated, so neighbouring lags carry nearly the same evidence. This is
    the instrument that says whether a readout reports a **band** or invents a point delay inside
    one.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator, correlated=True)
    source = _read_driver(u_stream, geometry)
    taps = torch.hann_window(KERNEL_WIDTH + 2)[1:-1]
    taps = taps / taps.sum()
    driver = torch.zeros_like(source)
    for offset, weight in enumerate(taps.tolist()):
        driver = driver + weight * _shift(source, PLANTED_DELAY + offset)
    return _as_batch(_autoregression(driver, geometry, generator), u_stream, geometry, generator)


def _state_dependent(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""The source acts only where the target's own state permits it.

    $$D_s = \mathbb 1[y_{s-1} > 0] \, u_{s - d_0},$$

    so neither stream predicts the effect alone and the two together do. This is the synergy case
    the specification is explicit about: conditional information includes synergy, and a design
    that required the source correction to be independent of the target would throw it away.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator)
    source = _shift(_read_driver(u_stream, geometry), PLANTED_DELAY)
    # The gate is built from the target's own recursion run once without the source, so the state
    # that gates the effect is not itself a function of the effect.
    gate = _autoregression(
        torch.zeros(geometry.rows, geometry.sequence_length), geometry, generator
    )
    driver = torch.where(_shift(gate, 1) > 0.0, source, torch.zeros_like(source))
    return _as_batch(_autoregression(driver, geometry, generator), u_stream, geometry, generator)


def _multi_lag_interaction(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""The effect is a product of two source times, which no additive proposal can represent.

    $$D_s = u_{s - d_0}\, u_{s - d_1}.$$

    Suppressing either lag alone destroys the whole effect, so the two lags are not separable in
    any additive sense. The architecture's proposals are additive before their limiter, so this
    instrument measures a **stated** representational restriction rather than looking for a defect.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    u_stream = _source_noise(geometry, generator)
    source = _read_driver(u_stream, geometry)
    driver = _shift(source, PLANTED_DELAY) * _shift(source, SECOND_DELAY)
    return _as_batch(_autoregression(driver, geometry, generator), u_stream, geometry, generator)


def _informative_zero(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""The value zero is the informative event, and it is an observation rather than an absence.

    The source takes three values and the target responds to exactly one of them:

    $$u \in \{-1, 0, 1\}, \qquad D_s = \mathbb 1[u_{s-d_0} = 0].$$

    A model that treated a standardized zero as "no observation" would have nothing to read here,
    and the architecture is explicit that a valid zero carries mask one. This instrument is what
    turns that from an assertion about the code into a measurement about the fit.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    levels = torch.randint(
        low=-1,
        high=2,
        size=(geometry.rows, geometry.sequence_length, geometry.n_source),
        generator=generator,
    ).to(torch.float32)
    driver = _shift((levels[:, :, _driver_channels(geometry)] == 0.0).all(dim=-1).float(), PLANTED_DELAY)
    return _as_batch(_autoregression(driver, geometry, generator), levels, geometry, generator)


def _constant_source(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    """The source is constant within a recording, so only its level and the clock vary.

    The target is a capable autoregression that ignores it entirely. A readout that fires here is
    reading the recording's own level, or the stored position, as source content.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    level = torch.randn(geometry.rows, 1, geometry.n_source, generator=generator)
    u_stream = level.expand(-1, geometry.sequence_length, -1).contiguous()
    target = _autoregression(
        torch.zeros(geometry.rows, geometry.sequence_length), geometry, generator
    )
    return _as_batch(target, u_stream, geometry, generator)


def _common_driver(geometry: InstrumentGeometry, seed: int) -> GeneratedBatch:
    r"""A hidden process moves both streams, and the source causes none of the target.

    $$u_s = z_s + \eta_s, \qquad D_s = z_{s - d_0}.$$

    The source is genuinely predictive of the future target given its history, so a readout that
    fires here is **right**: conditional predictive information is exactly what it measures. What
    the reader must not do is call it a causal effect, and the truth record for this generator says
    so rather than leaving it to be inferred from a firing rate that looks like the planted-delay
    instrument's.

    Args:
        geometry: The stored grid.
        seed: The build's seed.

    Returns:
        The batch.
    """
    generator = _seeded(seed)
    hidden = _source_noise(geometry, generator, correlated=True)[:, :, 0]
    u_stream = _source_noise(geometry, generator) * 0.4
    u_stream[:, :, _driver_channels(geometry)] += hidden.unsqueeze(-1)
    return _as_batch(
        _autoregression(_shift(hidden, PLANTED_DELAY), geometry, generator),
        u_stream,
        geometry,
        generator,
    )


def stored_feature_generators(geometry: InstrumentGeometry) -> Dict[str, Generator]:
    """Every generator that writes stored features directly, with its declared truth.

    A function of the geometry rather than a module constant, because two of the truth records
    carry a **band**, and a band is a function of the horizon and the lag window. Written down at
    one geometry it would be a claim about a different run.

    Args:
        geometry: The stored grid the campaign runs at.

    Returns:
        ``{name: Generator}``, in the order the specification's own table lists them.
    """
    horizon, n_lags = geometry.horizon, geometry.n_lags
    single = direct_support([PLANTED_DELAY], horizon, n_lags)
    several = direct_support([PLANTED_DELAY, SECOND_DELAY], horizon, n_lags)
    kernel = direct_support(
        [PLANTED_DELAY, PLANTED_DELAY + KERNEL_WIDTH - 1], horizon, n_lags
    )
    return {
        "target_autoregression": Generator(
            name="target_autoregression",
            build=_target_autoregression,
            truth=Truth(
                source_informative=False,
                direct_support=None,
                causal=False,
                note=(
                    "the source is independent noise and the target is a capable autoregression, "
                    "so every gain reported here is a false positive"
                ),
            ),
            checks="a false source gain against a capable target baseline",
        ),
        "redundant_source": Generator(
            name="redundant_source",
            build=_redundant_source,
            truth=Truth(
                source_informative=False,
                direct_support=None,
                causal=False,
                note=(
                    "the source is a deterministic function of target history the model already "
                    "holds, so it is marginally very predictive and conditionally worthless"
                ),
            ),
            checks="whether extra target capacity masquerades as source novelty",
        ),
        "single_delay": Generator(
            name="single_delay",
            build=_single_delay,
            truth=Truth(
                source_informative=True,
                direct_support=single,
                causal=True,
                note="one referenced source time, on a white source",
            ),
            checks="direct lag support and horizon indexing",
        ),
        "several_delays": Generator(
            name="several_delays",
            build=_several_delays,
            truth=Truth(
                source_informative=True,
                direct_support=several,
                causal=True,
                note=(
                    "two referenced source times of equal weight; the declared band spans both, "
                    "so a peak anywhere inside it counts and neither delay is privileged"
                ),
            ),
            checks="competing source evidence across lags",
        ),
        "broad_kernel": Generator(
            name="broad_kernel",
            build=_broad_kernel,
            truth=Truth(
                source_informative=True,
                direct_support=kernel,
                causal=True,
                note=(
                    "a smooth kernel over an autocorrelated source, so neighbouring lags carry "
                    "nearly the same evidence and a point delay is not a thing the data contains"
                ),
            ),
            checks="band recovery rather than an unjustified point delay",
        ),
        "state_dependent": Generator(
            name="state_dependent",
            build=_state_dependent,
            truth=Truth(
                source_informative=True,
                direct_support=single,
                causal=True,
                note=(
                    "the effect is gated by the target's own state, so neither stream predicts it "
                    "alone; this is synergy and a readout is expected to keep it"
                ),
            ),
            checks="whether conditioning the proposal on the target state is useful",
        ),
        "multi_lag_interaction": Generator(
            name="multi_lag_interaction",
            build=_multi_lag_interaction,
            truth=Truth(
                source_informative=True,
                direct_support=several,
                causal=True,
                note=(
                    "the effect is a product of two source times and is destroyed by removing "
                    "either, so it is not separable additively. A LOW recovery rate here is a "
                    "measurement of a stated representational restriction, not a defect"
                ),
            ),
            checks="the limits of additive parameter proposals",
        ),
        "informative_zero": Generator(
            name="informative_zero",
            build=_informative_zero,
            truth=Truth(
                source_informative=True,
                direct_support=single,
                causal=True,
                note=(
                    "the informative event is the source value zero, which is an observation and "
                    "not an absence; a model that conflated the two would find nothing here"
                ),
            ),
            checks="absence semantics against a real observed zero",
        ),
        "constant_source": Generator(
            name="constant_source",
            build=_constant_source,
            truth=Truth(
                source_informative=False,
                direct_support=None,
                causal=False,
                note=(
                    "the source is constant within a recording, so anything the readout finds is "
                    "the recording level or the stored position rather than source content"
                ),
            ),
            checks="clock and mask confounding",
        ),
        "common_driver": Generator(
            name="common_driver",
            build=_common_driver,
            truth=Truth(
                source_informative=True,
                direct_support=single,
                causal=False,
                note=(
                    "a hidden process moves both streams and the source causes none of the "
                    "target. A detection here is CORRECT as conditional predictive information "
                    "and is not a causal claim; this generator exists to make that distinction "
                    "a measured one rather than a caveat"
                ),
            ),
            checks="the limits of a predictive causal interpretation",
        ),
    }


__all__ = [
    "CHANNEL_NOISE",
    "DRIVER_GAIN",
    "INFORMATIVE_SOURCE_CHANNELS",
    "KERNEL_WIDTH",
    "PLANTED_DELAY",
    "SECOND_DELAY",
    "TARGET_AR",
    "TARGET_NOISE",
    "GeneratedBatch",
    "Generator",
    "InstrumentGeometry",
    "Truth",
    "direct_support",
    "stored_feature_generators",
]
