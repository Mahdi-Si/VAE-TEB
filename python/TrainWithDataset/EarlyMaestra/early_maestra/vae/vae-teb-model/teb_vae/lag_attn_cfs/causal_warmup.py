r"""How long each causal input channel waits before it is honest, and what a budget does about it.

A one-sided filter reads only the past, so before its warm-up has passed its output is a function
of the assumed pre-recording history rather than of the recording. Per channel, the size of that
region is $W'_c$, the leading delay enclosing $95\%$ of the kernel's energy, rebased into the
coordinates of the window the loader actually serves.

This module answers two questions, both of them properties of the shards rather than of the run.

*Given a warm-up budget, which channels may be used, and how long does each of them wait?* --
:func:`resolve_warmup_budget`, which reads the boundary off the configured shards, keeps the target
channels whose warm-up fits the budget, and returns the channel tuples the network constructor
takes.

*And on which clock does the encoder read them?* Every one-sided channel is stale by its own
composed group delay $\tau_c$, which the shards record as ``causal_delay_s`` and which spans
$13.3$ s to $791.0$ s across a stored block. Reading the whole vector at one step index therefore
asserts that its entries describe one instant, and they do not: they span thirteen minutes. The
repair is a per-channel shift onto a common reference,

$$d_c \;=\; \operatorname{round}\!\Bigl(\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr),
  \qquad \Delta = 4~\mathrm{s},$$

resolved here from the same attribute and applied by ``ChannelGate(delays=...)``, which already
exists and is built with ``delays=None`` when no reference is configured. A channel whose delay
exceeds the reference would need $d_c < 0$ -- to be read from its own future -- and is **dropped**
rather than advanced. The cost is provably nothing in warm-up: with $W_c \approx \rho\tau_c$,
$W_c + \Delta d_c = \tau_{\mathrm{ref}} + (\rho - 1)\tau_c \le W_{\mathrm{ref}}$, so aligning a
stream to its slowest kept channel costs no warm-up beyond that channel's own. That is checked on
the resolved vectors rather than assumed, because $\rho$ is only approximately constant.

*And may the two streams read on different clocks?* Yes, deliberately, through a second key. One
reference for both streams puts the source's freshest content $\tau_{\mathrm{ref}}$ before the
anchor, which at the target's own slowest kept channel is $402.2$ s -- far enough back that a
$20$-$60$ s physiological delay is reported **below lag $0$** at most horizon steps and is censored
rather than found. ``causal_align_reference_source`` aligns the source onto a chosen faster clock
$\tau^u_{\mathrm{ref}}$ while the target keeps its own $\tau^y_{\mathrm{ref}}$, which lifts the
readable band clear of that edge and pays for it in the source channels above the faster clock. The
price is a constant inter-stream bias $\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}$ on the lag
axis, and it is a **known** constant: it is carried on :class:`WarmupBudget`, stated in the startup
log, recorded in the run's causality disclosure, and is exactly the term
:func:`~teb_vae.lag_attn.nets.lag_report.physical_lag_seconds` already takes. That is what
distinguishes it from the per-stream *maximum* references this module rejects below, whose bias is
whatever the two streams happen to declare.

**The budget and the anchor floor are one decision.** A forecast at anchor $t$, horizon step $\tau$,
reads target time $t + 1 + \tau$; channel $c$ is valid there iff $t + 1 + \tau \ge W'_c$. Requiring
every kept channel to be valid across every anchor's whole window collapses to one inequality,

$$t + 1 \ge W'_c \quad \forall\, c \in \mathrm{kept}, \; \forall\, t \ge F,$$

satisfied exactly when the budget $B = \max_{c \in \mathrm{kept}} W'_c$ and the anchor floor
$F \ge B - 1$. The configuration exposes them as two keys so the floor may exceed the minimum the
budget requires -- a policy that withholds a forecast until some fixed observation time is then a
one-key change -- but the pairing itself is enforced where the model is built, not assumed here.
What this module owns is $B$: the keep-index and the per-channel wait.

**The boundary is read from the shards, never declared in YAML.** A dataset rebuilt at another
``causal_warmup_quantile`` changes both the warm-up vectors *and* the stored channel count, so a
config literal would silently describe a boundary the data no longer has. Every configured shard is
validated rather than only the first, because a test shard built at another quantile beside causal
training shards would otherwise resolve cleanly and be evaluated against the wrong geometry.

**Why not the two-sided reach guard.** ``teb_vae.lag_attn.channel_reach`` measures a different
quantity on a different bank: its ``block_reach_seconds`` builds the production two-sided kymatio
Morlets unconditionally, so it reads the same reaches whatever transform produced the data in front
of it, and it carries that bank into every process that imports it. Nothing here imports it, and a
configuration setting both guards at once is refused below.

The module reads HDF5 attributes and does arithmetic on integer vectors: no coefficient data, no
filter bank, milliseconds. It is not torch-free -- ``hdf5_dataset`` imports torch at module scope --
and does not need to be: the process that resolves a budget is about to build a network.

**Why the alignment rule is restated here rather than imported.**
``hdf5_dataset.causal_scattering.channel_alignment_delays`` is the canonical statement of $d_c$ and
is what the writer and the fidelity harness call. It cannot be imported from here: that module
imports ``kymatio`` at module scope, so importing it would build a filter-bank dependency into
every training process -- exactly what the reach-guard paragraph above refuses. The arithmetic is
one rounding and one refusal; ``tests/test_causal_warmup.py`` asserts the two agree entry for entry
on the committed fixture, so the duplication cannot drift silently.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_cfs/causal_warmup.py`` -> up three.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run directly -- from an IDE's Run button, to draw the tradeoff curve at the bottom of this file
# -- Python puts *this directory* on sys.path rather than the repository root, and the absolute
# import below fails before ``__main__`` is ever reached. Guarded rather than unconditional: as an
# imported module ``__package__`` is set and none of this is needed.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from hdf5_dataset.hdf5_dataset import (  # noqa: E402
    DECIMATION,
    RAW_SAMPLING_HZ,
    CausalWarmup,
    read_causal_warmup,
)

logger = logging.getLogger(__name__)

#: Seconds per decimated step, $\Delta$. Derived from the dataset's own two constants rather than
#: restated as ``4.0``: the shift below converts a delay in seconds into a step index, and a
#: hardcoded $\Delta$ would keep resolving shifts against a decimation the shards no longer have.
STEP_SECONDS = float(DECIMATION) / float(RAW_SAMPLING_HZ)

#: Fraction of a channel's reported group delay that its content actually sits at, $1 - 1/(2\gamma)$
#: $= 0.875$ at the shipped gammatone order $\gamma = 4$.
#:
#: ``causal_delay_s`` ships the phase group delay $\tau_g = \gamma/(2\pi b)$, which is the
#: envelope's *mean* and the right number to REPORT as a channel's staleness. It is not the right
#: number to ALIGN on. The delay a stored channel actually exhibits is the spectrum-weighted average
#: group delay -- equivalently the impulse response's energy centroid, $(2\gamma-1)/(4\pi b)$ --
#: because $\tau_g(\nu) = \tau_g(\xi)\,b^2/(b^2 + (\nu-\xi)^2)$ is *maximal* at the centre frequency,
#: so a channel's own passband contributes only downward departures. The spread is one-sided, not
#: the symmetric jitter it was long recorded as, and it therefore does not average out.
#:
#: Measured on the shipped bank over 30 segments, causal ``fhr_st`` against the centred block: the
#: median ratio of realised to reported lag is $0.903$ over all 30 resolved channels and $0.882$
#: over the nine slow ones where the $4$ s grid quantises by under $2.5\%$ -- against $0.875$
#: predicted here and $1.000$ predicted by $\tau_g$.
#:
#: Restated rather than imported, like :data:`TARGET_BLOCKS` above: deriving it from
#: ``hdf5_dataset.causal_scattering.GAMMATONE_ORDER`` would build the two-sided filter bank at
#: import and cost this module the torch-free, kymatio-free property the resolver depends on.
#: ``test_causal_warmup.py`` asserts the two agree, so they cannot drift.
GAMMATONE_ORDER = 4
ALIGNMENT_DELAY_FACTOR = 1.0 - 1.0 / (2.0 * GAMMATONE_ORDER)

#: The alignment reference resolved from the data: the slowest **kept target** channel's composed
#: delay. Written out because it appears in refusal messages and in the config.
REFERENCE_TARGET_MAX = "target_max"

#: The three clocks a forecast target may be scored on. ``stored`` is every run before the key
#: existed: channel $c$ is scored at its own stored index, $s_c = 0$. ``physical`` advances each
#: kept channel by $s_c = \mathrm{round}(\kappa(\tau_c - \tau_{\min})/\Delta) \ge 0$ so that every
#: horizon step scores content at one physical future instant; every element is then a strict
#: forecast for every channel, at the price of the trailing $\max_c s_c$ anchors. ``input`` delays
#: the scored element exactly as the encoder input is delayed, $s_c = -d_c \le 0$ -- the
#: continuation of the model's own aligned stream -- and is only resolvable against an aligned
#: target. The vector is signed and single-signed by construction, which is what lets one gather,
#: one floor formula and one ceiling serve all three.
FORECAST_CLOCK_STORED = "stored"
FORECAST_CLOCK_PHYSICAL = "physical"
FORECAST_CLOCK_INPUT = "input"
FORECAST_CLOCKS = (FORECAST_CLOCK_STORED, FORECAST_CLOCK_PHYSICAL, FORECAST_CLOCK_INPUT)

#: Stored block names, in the channel order the model concatenates them into each input stream.
#: Restated rather than imported from ``teb_vae.lag_attn.channel_reach``, which owns the same two
#: tuples: that module builds the two-sided filter bank at import, and it is the guard this family
#: replaces rather than the one it reuses. ``fhr_up_ph`` appears in neither -- a coefficient mixing
#: both signals would destroy the target-only / source-conditioned separation the design rests on,
#: and the causal variant does not store it at all.
TARGET_BLOCKS = ("fhr_st", "fhr_ph")
SOURCE_BLOCKS = ("up_st", "up_ph")

#: Configuration paths, written out because they appear in refusal messages. A message that names
#: the key the operator has to edit is the difference between a two-minute fix and a search.
BUDGET_KEY = "model_config.VAE_model.causal_warmup_budget_steps"
REACH_KEY = "model_config.VAE_model.causal_reach_budget_s"
ALIGN_KEY = "model_config.VAE_model.causal_align_reference"
ALIGN_SOURCE_KEY = "model_config.VAE_model.causal_align_reference_source"
TARGET_FORECAST_CLOCK_KEY = "model_config.VAE_model.causal_target_forecast_clock"
LEG_ALIGNMENT_KEY = "model_config.VAE_model.causal_leg_alignment"
SEQUENCE_KEY = "model_config.VAE_model.sequence_length"
TRIM_KEY = "dataset_config.dataloader_config.dataset_kwargs.trim_minutes"


@dataclass(frozen=True)
class StreamWarmup:
    r"""One input stream's warm-up: how long each declared channel waits, and which survive.

    The declared-width vector is stored and the surviving one derived, rather than the other way
    round, for the reason ``StreamChannels`` keeps its declared vectors: a figure of this guard
    draws the channels that did **not** survive beside the ones that did, and a stream that carried
    only its survivors could not say what it dropped. Everything derivable is a property, so no
    field can disagree with another.

    Attributes:
        name: ``'target'`` or ``'source'``.
        block_spans: ``(name, start, stop)`` per stored block, half-open, in declared coordinates.
        declared_warmup_steps: $W'_c$ per declared channel, in decimated steps of the trimmed
            window, in the concatenated channel order.
        declared_delay_s: $\tau_c$ per declared channel, in **seconds** and unrebased -- the
            shard's own ``causal_delay_s``. Declared rather than kept for the same reason the
            warm-up is, and one of its own: the channels the alignment drops are named in the
            startup log *by their delay*, which a survivors-only vector could not do.
        declared_novelty_frac: $\nu_c$ per declared channel -- the share of that coefficient drawn
            from raw samples the anchor has not seen, at the horizon the writer measured it over --
            or ``None`` when any of the stream's blocks was written before the attribute existed.
            ``None`` for the whole stream rather than per block, because the vector is consumed as
            one concatenated axis and a half-filled one would silently partition on the half that
            is there. Declared rather than kept for the same reason as above, and one of its own:
            the model gathers it through the keep-index, so a model built with no gate at all still
            receives a vector of the right width.
        keep_index: Surviving channel indices into the declared width, strictly ascending -- the
            order ``ChannelGate``'s gather requires.
        align_delays: $d_c$ per **survivor**, positional against :attr:`keep_index`, or ``None``
            when no reference is configured -- which is the stream the gate gathers without
            shifting. Stored rather than derived because the reference it was resolved against is
            a run's decision, not a property of the vectors kept here.
    """

    name: str
    block_spans: Tuple[Tuple[str, int, int], ...]
    declared_warmup_steps: Tuple[int, ...]
    declared_delay_s: Tuple[float, ...]
    declared_novelty_frac: Optional[Tuple[float, ...]]
    keep_index: Tuple[int, ...]
    align_delays: Optional[Tuple[int, ...]] = None

    @property
    def declared_width(self) -> int:
        """Channels the stream declares, before the budget."""
        return len(self.declared_warmup_steps)

    @property
    def kept_width(self) -> int:
        """Channels the encoder is given."""
        return len(self.keep_index)

    @property
    def warmup_steps(self) -> Tuple[int, ...]:
        r"""$W'_c$ per survivor, positional against :attr:`keep_index`."""
        return tuple(self.declared_warmup_steps[index] for index in self.keep_index)

    @property
    def dropped_index(self) -> Tuple[int, ...]:
        """Declared channel indices the budget removed, ascending."""
        kept = set(self.keep_index)
        return tuple(index for index in range(self.declared_width) if index not in kept)

    @property
    def delay_s(self) -> Tuple[float, ...]:
        r"""$\tau_c$ per survivor, in seconds, positional against :attr:`keep_index`."""
        return tuple(self.declared_delay_s[index] for index in self.keep_index)

    @property
    def combined_steps(self) -> Tuple[int, ...]:
        r"""When each survivor is honest **as the encoder reads it**: $W'_c + d_c$, in steps.

        The shift moves a channel's content later, so a gathered-and-delayed channel is a function
        of the recording only once the step index has reached *both* its own warm-up and its shift.
        This is the vector the availability adapter masks and announces with, and the vector the
        anchor floor's input-warmth half is taken over; the unshifted :attr:`warmup_steps` is
        neither. Identical to it when no reference is configured.
        """
        if self.align_delays is None:
            return self.warmup_steps
        return tuple(
            wait + shift for wait, shift in zip(self.warmup_steps, self.align_delays)
        )

    @property
    def max_align_delay(self) -> int:
        r"""The largest shift applied to any survivor, $\max_c d_c$, or $0$ with no reference.

        Attained by the **fastest** kept channel, which is the one furthest from the reference --
        so it is not an upper bound on how stale the stream is in physical time. It is the number
        ``ChannelGate.max_delay`` reports, and the number the leading steps of the delayed stream
        are zero-filled for.
        """
        return 0 if self.align_delays is None else max(self.align_delays, default=0)

    @property
    def max_warmup(self) -> int:
        r"""The slowest survivor's wait $B = \max_{c \in \mathrm{kept}} W'_c$, in steps.

        This -- not the configured threshold -- is the quantity the anchor floor must clear: a
        budget of $134$ against a staircase whose highest step below it is $112$ needs a floor of
        $111$, not $133$.
        """
        return max(self.warmup_steps, default=0)

    def block_counts(self) -> Tuple[Tuple[str, int, int], ...]:
        """Attribute the survivors back to the blocks they came from.

        The blocks lose different fractions of their channels, and "did ``fhr_ph`` survive this
        budget?" is the resolution at which the question is actually asked.

        Returns:
            One ``(name, kept, declared)`` triple per block, in concatenation order.
        """
        kept = np.asarray(self.keep_index, dtype=np.int64)
        return tuple(
            (name, int(np.count_nonzero((kept >= start) & (kept < stop))), stop - start)
            for name, start, stop in self.block_spans
        )

    def summary(self) -> str:
        """One line naming the surviving counts per block, the slowest wait and the shift range."""
        blocks = ", ".join(
            f"{name} {kept}/{declared}" for name, kept, declared in self.block_counts()
        )
        # Appended rather than always present: an unaligned run's line is the one every existing
        # log and every existing test reads, and a trailing "shift 0-0" on it would be noise that
        # says a mechanism ran when none did.
        shift = (
            ""
            if self.align_delays is None
            else (
                f", shift {min(self.align_delays)}-{self.max_align_delay} steps, "
                f"honest by 0-{max(self.combined_steps, default=0)}"
            )
        )
        return (
            f"{self.name}: {blocks}; {self.kept_width}/{self.declared_width} channels, "
            f"warm-up 0-{self.max_warmup} steps{shift}"
        )


@dataclass(frozen=True)
class WarmupBudget:
    r"""The resolved causal warm-up guard for one run: which channels survive, and how long each waits.

    Attributes:
        budget_steps: The configured threshold $B_{\mathrm{cfg}}$ in decimated steps. A channel
            survives when $W'_c \le B_{\mathrm{cfg}}$.
        trim_minutes: The trim the warm-up vectors are expressed against, which is the loader's own.
        quantile: The ``causal_warmup_quantile`` every configured shard was built at.
        reference_delay_s: $\tau^y_{\mathrm{ref}}$, the clock the **target** stream was shifted
            onto, in seconds; ``None`` when no reference is configured and neither stream is
            shifted. It is also the source's clock unless :attr:`source_reference_delay_s` names
            another one.
            **This -- not** ``source_delay_steps`` **-- is the physical constant a lag report
            needs.** The two are different quantities: the model's scalar is the largest *stored
            step* shift, attained by the fastest channel, while this is the physical instant every
            aligned channel reports at step $t$, namely $\Delta t - \tau_{\mathrm{ref}}$.
        source_reference_delay_s: $\tau^u_{\mathrm{ref}}$ where the source stream was aligned onto
            a clock of its own, in seconds; ``None`` where it was not, which is the single-reference
            scheme and is what every run before this key existed did. Never read on its own by a
            consumer computing a physical lag: use :attr:`source_clock_delay_s`, which resolves the
            fallback, since a ``None`` here means "the target's clock" rather than "unaligned".
        leg_alignment: Which phase-harmonic operator built the configured shards' phase blocks, as
            they record it. Carried so a run's startup log states which dataset variant it read;
            the *expected* value is a config key, checked in :func:`resolve_warmup_budget`.
        target: The target stream, which the budget gates.
        source: The source stream, which it does not; see :func:`resolve_warmup_budget`.
        target_forecast_clock: Which clock the forecast target is scored on -- one of
            :data:`FORECAST_CLOCKS`. ``'stored'`` is every run before the key existed.
        target_forecast_shift: $s_c$ per **kept** target channel, positional against
            ``target.keep_index``: the scored element at anchor $t$, horizon step $\tau$, reads
            stored step $t + 1 + \tau + s_c$. ``None`` under the stored clock, where the vector is
            identically zero and carrying it would make a byte-identical run's record differ.
        target_forecast_reference_s: The clock's reference in seconds -- $\tau_{\min}$ of the kept
            target channels under ``'physical'`` -- or ``None`` where the clock is another field's
            (``'input'`` reads :attr:`reference_delay_s`) or there is no shift at all.
    """

    budget_steps: int
    trim_minutes: Optional[float]
    quantile: Optional[float]
    reference_delay_s: Optional[float]
    source_reference_delay_s: Optional[float]
    leg_alignment: str
    target: StreamWarmup
    source: StreamWarmup
    target_forecast_clock: str = FORECAST_CLOCK_STORED
    target_forecast_shift: Optional[Tuple[int, ...]] = None
    target_forecast_reference_s: Optional[float] = None

    @property
    def max_forecast_advance(self) -> int:
        r"""$\max(0, \max_c s_c)$: how many trailing anchors the forecast clock costs.

        The anchor ceiling is $T_{\mathrm{valid}} - $ this, because the last scored element of an
        advanced channel reads stored step $t + H - 1 + s_c$ and must stay inside the window.
        $0$ under the stored clock and under ``'input'``, whose shifts are all $\le 0$.
        """
        if self.target_forecast_shift is None:
            return 0
        return max(0, max(self.target_forecast_shift, default=0))

    @property
    def source_clock_delay_s(self) -> Optional[float]:
        r"""$\tau^u_{\mathrm{ref}}$, the clock the source stream was actually shifted onto.

        The dual reference where one is configured, the target's own where it is not, ``None`` on an
        unaligned run. This is the quantity a physical lag is computed from -- it is what
        :func:`~teb_vae.lag_attn.nets.lag_report.physical_lag_seconds` calls ``source_reference_s``
        -- and it exists as a property so no consumer has to re-derive the fallback and get it
        wrong in one place out of four.
        """
        if self.reference_delay_s is None:
            return None
        return (
            self.reference_delay_s
            if self.source_reference_delay_s is None
            else self.source_reference_delay_s
        )

    @property
    def inter_stream_offset_s(self) -> Optional[float]:
        r"""$\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}$, the dual clock's bias on the lag axis.

        Exactly $0$ under the single-reference scheme, which is a *measured* zero rather than an
        absent quantity -- the two clocks coincide -- so it is reported as $0.0$ there and as
        ``None`` only on an unaligned run, where the bias is channel-pair-indexed and no single
        number stands in for it.
        """
        target_clock = self.reference_delay_s
        source_clock = self.source_clock_delay_s
        if target_clock is None or source_clock is None:
            return None
        return float(source_clock) - float(target_clock)

    def summary(self) -> str:
        """One line for the startup log, naming both streams and the threshold that produced them.

        The quantile is rounded for reading: it comes back from the shard as the ``float32`` it was
        stored as, and ``0.949999988079071`` in a log line is noise rather than provenance.

        The reference clause names **one** clock where one is configured, and the pair plus their
        offset where the source keeps its own -- rather than always printing the pair, because the
        single-reference line is the one every existing run's log carries and a trailing
        ``, source reference ... (offset +0.00 s)`` on it would report a mechanism that did not run.
        """
        quantile = "unknown" if self.quantile is None else f"{self.quantile:.3g}"
        reference = (
            "unaligned"
            if self.reference_delay_s is None
            else f"reference {self.reference_delay_s:.4f} s"
        )
        if self.source_reference_delay_s is not None:
            reference = (
                f"target reference {self.reference_delay_s:.4f} s, source reference "
                f"{self.source_reference_delay_s:.4f} s, inter-stream offset "
                f"{self.inter_stream_offset_s:+.4f} s"
            )
        # Appended only when a shift exists, for the reason the stream summaries append theirs: a
        # stored-clock run's line is the one every existing log carries, and a trailing "forecast
        # clock stored" on it would report a mechanism that did not run.
        forecast = ""
        if self.target_forecast_shift is not None:
            forecast = (
                f", forecast clock {self.target_forecast_clock} "
                f"(shift {min(self.target_forecast_shift)}..{max(self.target_forecast_shift)} "
                f"steps, ceiling -{self.max_forecast_advance})"
            )
        return (
            f"causal warm-up budget {self.budget_steps} steps "
            f"(quantile {quantile}, trim_minutes {self.trim_minutes}, "
            f"leg alignment {self.leg_alignment}, {reference}{forecast}): "
            f"{self.target.summary()}; {self.source.summary()}"
        )


def _require_int(vae_config: Mapping[str, Any], key: str) -> int:
    """Read a geometry key the budget cannot be resolved without.

    Deliberately no defaults. Every one of these keys also configures the network, so a default
    here would be a second declaration of the geometry that could disagree with the constructor's
    -- and the disagreement would move the anchor floor, which nothing downstream reports.

    Args:
        vae_config: The ``model_config.VAE_model`` block.
        key: The key to read.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If the key is absent or ``None``, naming its full config path.
    """
    value = vae_config.get(key)
    if value is None:
        raise ValueError(
            f"{BUDGET_KEY} is set but model_config.VAE_model.{key} is not. The warm-up budget is "
            f"resolved against the run's own geometry, and a default here would be a second "
            f"declaration of it that could silently disagree with the network's."
        )
    return int(value)


def _build_stream(
    name: str, blocks: Sequence[str], warmup: CausalWarmup, declared_width: int, declared_key: str
) -> Tuple[
    Tuple[Tuple[str, int, int], ...],
    Tuple[int, ...],
    Tuple[float, ...],
    Optional[Tuple[float, ...]],
]:
    """Concatenate a stream's blocks into one declared-width warm-up, delay and novelty vector.

    The three travel together because they are positional into the same width and are read off the
    same blocks in the same order: a stream assembled with one of them out of step would gate on
    one channel's warm-up and shift on another's, with every length still correct.

    The novelty vector is the one that may be missing. It is written by a strictly later writer than
    the other two, so a shard built before it exists carries neither the attribute nor any value
    that could stand in for it -- and it is reported as ``None`` for the **whole stream** the moment
    any one block lacks it, because a vector that is real over ``fhr_st`` and fabricated over
    ``fhr_ph`` would partition cleanly and mean nothing.

    Args:
        name: ``'target'`` or ``'source'``, for the refusal messages.
        blocks: The stored blocks making up this stream, in concatenation order.
        warmup: The boundary read off the shards.
        declared_width: The stream width the config declares.
        declared_key: The config key that declared it, for the refusal message.

    Returns:
        ``(block_spans, declared_warmup_steps, declared_delay_s, declared_novelty_frac)``, the last
        of which is ``None`` when any block of the stream carries no novelty vector.

    Raises:
        ValueError: If a block is absent from the shards, if the concatenated width disagrees with
            the declared one, or if a stored novelty vector is not of its block's width.
    """
    missing = [block for block in blocks if block not in warmup.warmup_steps]
    if missing:
        raise ValueError(
            f"the configured shards store no {missing} block, so the {name} stream cannot be "
            f"assembled; they store {sorted(warmup.warmup_steps)}."
        )

    spans: List[Tuple[str, int, int]] = []
    values: List[int] = []
    delays: List[float] = []
    novelty: Optional[List[float]] = [] if all(
        block in warmup.novelty_frac for block in blocks
    ) else None
    for block in blocks:
        vector = warmup.warmup_steps[block]
        spans.append((block, len(values), len(values) + int(vector.size)))
        values.extend(int(step) for step in vector)
        # Required by ``read_causal_warmup`` on every causal shard, and of the block's own width:
        # both are checked there, so an absent or mis-shaped vector never reaches this line.
        delays.extend(float(delay) for delay in warmup.delay_s[block])
        if novelty is not None:
            stored = warmup.novelty_frac[block]
            # Checked here rather than at the read, which takes the attribute as it finds it. A
            # short vector would concatenate into a stream of the right total width whenever
            # another block was long by the same amount, and every channel after the join would be
            # attributed the wrong novelty with no length disagreeing.
            if int(stored.size) != int(vector.size):
                raise ValueError(
                    f"the shards' '{block}' causal_novelty_frac has {int(stored.size)} entries "
                    f"against {int(vector.size)} channels in causal_warmup_steps. The two are "
                    f"positional into one channel axis, so a mismatch attributes one channel's "
                    f"novelty to another rather than fail."
                )
            novelty.extend(float(share) for share in stored)

    if declared_width != len(values):
        raise ValueError(
            f"{declared_key}={declared_width} disagrees with the shards' {len(values)} {name} "
            f"channels for blocks {tuple(blocks)}. The keep-index and the warm-up vector are "
            f"positional into the declared width, so the model would gather the wrong channels "
            f"and wait the wrong number of steps for each rather than fail."
        )
    return (
        tuple(spans),
        tuple(values),
        tuple(delays),
        None if novelty is None else tuple(novelty),
    )


def _check_leg_alignment(expected: Optional[Any], measured: str, paths: Sequence[str]) -> None:
    """Refuse shards built by a phase-harmonic operator this run did not ask for.

    The aligned and unaligned shard variants have **identical** widths, warm-ups, stored delays and
    block names, so nothing else in the resolution can tell them apart -- and only one of them makes
    the stored ``causal_delay_s`` true of the phase blocks. Under the unaligned operator the two
    legs of a pair are multiplied at one stored index although they report the signal at two
    different instants, so the composed delay the alignment below resolves its shifts from is a
    misprediction there: the measured best lag misses it by a median of $60.5$ steps on ``fhr_ph``.
    Shifting by a number the data does not have is worse than not shifting at all.

    Args:
        expected: The configured expectation, or ``None`` for no expectation at all -- which is
            what a run comparing the two variants, or measuring the shift mechanism against a
            legacy shard, wants.
        measured: What the configured shards record, with absence already read as ``'none'``.
        paths: The shards, for the refusal message.

    Raises:
        ValueError: If an expectation is configured and the shards disagree with it.
    """
    if expected is None:
        return
    if str(expected) != measured:
        raise ValueError(
            f"{LEG_ALIGNMENT_KEY}={str(expected)!r} but the configured shards record "
            f"causal_leg_alignment={measured!r} ({', '.join(str(path) for path in paths)}). The "
            f"two variants have identical widths, warm-ups and stored delays, so nothing else "
            f"here can tell them apart -- and the stored causal_delay_s is only true of the phase "
            f"blocks under 'envelope'. Point the run at the matching build, or set "
            f"{LEG_ALIGNMENT_KEY}: null to state that this run does not care."
        )


def _resolve_reference_delay(
    setting: Any, kept_target_delay_s: Sequence[float]
) -> Optional[float]:
    r"""Resolve ``causal_align_reference`` into $\tau_{\mathrm{ref}}$ in seconds.

    Three settings, and the explicit float is the one that needs a guard. ``'target_max'`` resolves
    the reference from the data -- the slowest channel the budget keeps, $402.1604$ s on the shipped
    bank -- which is the only value that keeps every kept target channel. A float is admitted so the
    every-lag-warm reference of $150.79$ s stays a one-key override, and it is checked against the
    stream's own delays: the shift is a re-indexing onto *some channel's* clock, and a reference
    landing between two of them is a clock no channel keeps, whose residual shows up as a fraction
    of a step on every channel at once rather than as a failure.

    **The matched channel's own delay is returned, not the configured number.** The drop rule
    compares delays against the reference exactly, and the stored vector is ``float32``: a config
    literal of ``150.78593`` sits $4\times10^{-6}$ s *below* the channel it names, which would drop
    that channel -- and both its harmonic siblings, and the whole ``up_ph`` block behind it -- for a
    rounding difference. Snapping removes the trap at its source rather than toleranced comparisons
    at three later ones, and it is what makes the shift exactly zero at the reference.

    Args:
        setting: The configured value: ``None``, ``'target_max'``, or a number of seconds.
        kept_target_delay_s: $\tau_c$ of the target channels the warm-up budget keeps, in seconds.

    Returns:
        The reference in seconds -- always one of ``kept_target_delay_s`` -- or ``None`` when no
        alignment is configured.

    Raises:
        ValueError: If the setting is neither ``'target_max'`` nor a number, or if an explicit
            float matches no kept target channel within half a step.
    """
    if setting is None:
        return None

    if isinstance(setting, str):
        if setting != REFERENCE_TARGET_MAX:
            raise ValueError(
                f"{ALIGN_KEY}={setting!r} is not a reference this resolver knows. Use "
                f"{REFERENCE_TARGET_MAX!r} to take the slowest kept target channel's delay from "
                f"the shards, an explicit delay in seconds, or null for no alignment."
            )
        return float(max(kept_target_delay_s))

    reference = float(setting)
    nearest = float(min(kept_target_delay_s, key=lambda delay: abs(delay - reference)))
    if abs(nearest - reference) > STEP_SECONDS / 2.0:
        raise ValueError(
            f"{ALIGN_KEY}={reference:g} s matches no target channel the budget keeps: the nearest "
            f"is {nearest:.4f} s, {abs(nearest - reference):.4f} s away, against a half-step "
            f"tolerance of {STEP_SECONDS / 2.0:g} s. The alignment re-indexes every channel onto "
            f"one channel's clock, so a reference between two of them is a clock no channel keeps "
            f"and leaves a residual on every channel at once. Use "
            f"{REFERENCE_TARGET_MAX!r}, or name a stored delay."
        )
    return nearest


def _resolve_source_reference_delay(
    setting: Any,
    target_reference_s: Optional[float],
    source_delay_s: Sequence[float],
) -> Optional[float]:
    r"""Resolve ``causal_align_reference_source`` into $\tau^u_{\mathrm{ref}}$ in seconds.

    Two settings rather than three, and the missing one is deliberate: there is no ``'source_max'``
    here. A per-stream *maximum* was rejected in the alignment work itself, because it restores a
    bias between the two streams' clocks that nothing knows the size of. What this key admits is a
    reference **chosen** by the measurement that priced it, whose offset against the target's is one
    constant carried explicitly through the lag arithmetic -- a different object from the stream's
    own maximum, arrived at for the opposite reason.

    ``None`` is the single-reference scheme: the source is aligned onto the target's clock, which is
    what every run before this key existed did and is byte-for-byte the resolution it produced.

    Args:
        setting: The configured value: ``None``, or a number of seconds.
        target_reference_s: $\tau^y_{\mathrm{ref}}$ as already resolved, or ``None`` on an unaligned
            run.
        source_delay_s: $\tau_c$ per declared source channel, in seconds. The whole declared width,
            because the warm-up budget deliberately gates no source channel: what the alignment is
            about to drop is decided by this reference and nothing before it.

    Returns:
        The reference in seconds -- always one of ``source_delay_s`` -- or ``None`` when the source
        keeps the target's clock.

    Raises:
        ValueError: If the setting is not a number; if it is set while the target stream is
            unaligned; or if it matches no stored source delay within half a step.
    """
    if setting is None:
        return None

    if isinstance(setting, str):
        raise ValueError(
            f"{ALIGN_SOURCE_KEY}={setting!r} is not a reference this resolver knows. It takes an "
            f"explicit delay in seconds, or null for the source to keep the target's clock. There "
            f"is deliberately no 'source_max': a stream's own maximum restores an inter-stream "
            f"bias of unknown size, which is the scheme this key was built to replace rather "
            f"than one of its settings."
        )

    if target_reference_s is None:
        raise ValueError(
            f"{ALIGN_SOURCE_KEY}={float(setting):g} s is set but {ALIGN_KEY} is null, so the "
            f"target stream is not aligned at all. The dual scheme is a PAIR of clocks whose "
            f"difference is the constant this run would put on the lag axis; against an "
            f"unaligned target that "
            f"difference is channel-pair-indexed and spans over a thousand seconds, so the source "
            f"reference would name a precision the run does not have. Set {ALIGN_KEY} to the "
            f"target's clock, or set this key to null."
        )

    reference = float(setting)
    nearest = float(min(source_delay_s, key=lambda delay: abs(delay - reference)))
    if abs(nearest - reference) > STEP_SECONDS / 2.0:
        raise ValueError(
            f"{ALIGN_SOURCE_KEY}={reference:g} s matches no stored source channel: the nearest is "
            f"{nearest:.4f} s, {abs(nearest - reference):.4f} s away, against a half-step "
            f"tolerance of {STEP_SECONDS / 2.0:g} s. The alignment re-indexes every channel onto one "
            f"channel's clock, so a reference between two of them is a clock no channel keeps and "
            f"leaves a residual on every source channel at once. Name a stored source delay, or "
            f"set this key to null."
        )
    # The matched channel's own delay, not the configured literal, for the reason
    # ``_resolve_reference_delay`` snaps: the drop rule compares delays against the reference
    # exactly and the stored vector is float32, so a literal sitting microseconds below the channel
    # it names would drop that channel and every harmonic sibling behind it.
    return nearest


def _align_stream(
    name: str,
    delay_s: Sequence[float],
    keep_index: Sequence[int],
    reference_s: Optional[float],
) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    r"""Drop a stream's above-reference channels and resolve the survivors' shifts.

    $$d_c \;=\; \operatorname{round}\!\Bigl(\kappa\,\frac{\tau_{\mathrm{ref}} - \tau_c}{\Delta}\Bigr)
      \ge 0, \qquad \kappa = 1 - \frac{1}{2\gamma} = 0.875 .$$

    **The factor $\kappa$ is not a fudge.** ``causal_delay_s`` reports $\tau_g$, the envelope mean,
    which is the honest staleness of a channel; the delay a channel's content actually sits at is
    the energy centroid $\kappa\,\tau_g$, because the group delay is maximal at the centre frequency
    and the passband can only pull the realised lag *down*. See
    :data:`ALIGNMENT_DELAY_FACTOR` for the measurement. Only the *difference* is scaled, so the
    reference channel still takes shift $0$ and the keep-index is untouched -- $\tau_c \le
    \tau_{\mathrm{ref}}$ is scale-invariant.

    **Rounding, not ceiling**, and the reason is not the warm-up's. Both directions are causally
    safe -- a shift only selects which *already-causal* stored step is read -- so the only criterion
    is residual misalignment, which rounding minimises at $\Delta/2 = 2$ s. (Contrast the warm-up
    itself, where the ceiling is load-bearing, because a step that is $40\%$ pad is not $40\%$
    valid.)

    **A channel above the reference is dropped, not advanced.** Its shift would be negative, i.e.
    it would be read from a *later* stored step, which reads raw signal after the anchor and
    destroys the one property the causal construction exists for. That is a correctness
    requirement and is what distinguishes these drops from the warm-up budget's, which are a policy
    about how much of a segment to spend waiting.

    Args:
        name: ``'target'`` or ``'source'``, for the log line.
        delay_s: $\tau_c$ per **declared** channel, in seconds.
        keep_index: The channels surviving everything before the alignment, ascending.
        reference_s: $\tau_{\mathrm{ref}}$, or ``None`` for no alignment.

    Returns:
        ``(keep_index, align_delays)``: the keep-index narrowed to the channels at or below the
        reference, and one shift per survivor -- or the keep-index unchanged and ``None`` when no
        reference is configured.

    Raises:
        ValueError: If the reference drops every channel of the stream.
    """
    if reference_s is None:
        return tuple(keep_index), None

    kept: List[int] = []
    dropped: List[int] = []
    for index in keep_index:
        (kept if delay_s[index] <= reference_s else dropped).append(int(index))

    # One line each rather than a summary count: which channels a run stopped reading is the fact
    # a reader of its log needs months later, and "4 dropped" does not survive a channel-plan
    # change while an index and a delay do.
    for index in dropped:
        logger.info(
            f"causal alignment drops {name} channel {index}: composed delay "
            f"{delay_s[index]:.4f} s is above the reference {reference_s:.4f} s, so aligning it "
            f"would need a negative shift, which reads the channel's own future."
        )

    if not kept:
        raise ValueError(
            f"{ALIGN_KEY} resolves to {reference_s:.4f} s, which is below every one of the "
            f"{len(keep_index)} {name} channels reaching this point (the fastest is "
            f"{min(delay_s[index] for index in keep_index):.4f} s). A stream with zero channels "
            f"builds a model that trains to completion having never read it."
        )

    shifts = tuple(
        int(round(ALIGNMENT_DELAY_FACTOR * (reference_s - delay_s[index]) / STEP_SECONDS))
        for index in kept
    )
    return tuple(kept), shifts


def _resolve_target_forecast_shift(
    setting: Any,
    kept_delay_s: Sequence[float],
    align_delays: Optional[Sequence[int]],
) -> Tuple[str, Optional[Tuple[int, ...]], Optional[float]]:
    r"""Resolve ``causal_target_forecast_clock`` into the signed per-channel shift $s_c$.

    The scored element at anchor $t$, horizon step $\tau$, kept channel $c$ reads stored step
    $t + 1 + \tau + s_c$. Three clocks:

    * ``'stored'`` (and the absent key): $s_c = 0$ and **no vector at all**, so a run written
      before the key existed resolves byte for byte what it did then.
    * ``'physical'``: $s_c = \mathrm{round}(\kappa(\tau_c - \tau_{\min})/\Delta) \ge 0$, the
      advance that puts every channel's scored content at one physical future instant --
      $\tau_{\min}$ being the fastest kept channel, whose own shift is exactly $0$. Advancing is
      admissible on a *target* where it is refused on an input: a target element is what the
      anchor is asked to predict, so reading a later stored step asks a strictly harder question
      rather than leaking anything.
    * ``'input'``: $s_c = -d_c \le 0$, the same delay the encoder input receives, so the model is
      scored on the continuation of its own aligned stream. Only resolvable against an aligned
      target -- with no $d_c$ there is no clock to copy.

    Args:
        setting: The configured value: ``None`` or one of :data:`FORECAST_CLOCKS`.
        kept_delay_s: $\tau_c$ per target channel surviving budget **and** alignment, in seconds,
            positional against the final keep-index.
        align_delays: $d_c$ per kept target channel as :func:`_align_stream` resolved them, or
            ``None`` on an unaligned run.

    Returns:
        ``(clock, shifts, reference_s)``: the normalised clock name; the signed shift vector, or
        ``None`` under the stored clock; and the physical clock's reference $\tau_{\min}$ in
        seconds, or ``None`` where the clock is not the physical one.

    Raises:
        ValueError: If the setting names no clock this resolver knows, or if ``'input'`` is set
            while the target stream is unaligned.
    """
    if setting is None or setting == FORECAST_CLOCK_STORED:
        return FORECAST_CLOCK_STORED, None, None

    if setting == FORECAST_CLOCK_PHYSICAL:
        reference_s = float(min(kept_delay_s))
        shifts = tuple(
            int(round(ALIGNMENT_DELAY_FACTOR * (delay - reference_s) / STEP_SECONDS))
            for delay in kept_delay_s
        )
        return FORECAST_CLOCK_PHYSICAL, shifts, reference_s

    if setting == FORECAST_CLOCK_INPUT:
        if align_delays is None:
            raise ValueError(
                f"{TARGET_FORECAST_CLOCK_KEY}={FORECAST_CLOCK_INPUT!r} is set but {ALIGN_KEY} is "
                f"null, so the target stream is not aligned and there is no input clock to score "
                f"on: the 'input' clock delays each scored channel by the same $d_c$ the encoder "
                f"input receives, and an unaligned gate has none. Set {ALIGN_KEY}, or use "
                f"{FORECAST_CLOCK_STORED!r} / {FORECAST_CLOCK_PHYSICAL!r}."
            )
        return FORECAST_CLOCK_INPUT, tuple(-int(delay) for delay in align_delays), None

    raise ValueError(
        f"{TARGET_FORECAST_CLOCK_KEY}={setting!r} is not a clock this resolver knows. Use "
        f"{FORECAST_CLOCK_PHYSICAL!r} to score every channel at one physical future instant, "
        f"{FORECAST_CLOCK_INPUT!r} to score the continuation of the encoder's own aligned "
        f"stream, or {FORECAST_CLOCK_STORED!r} (or null) for each channel's own stored index."
    )


def resolve_warmup_budget(config: Mapping[str, Any]) -> Optional[WarmupBudget]:
    r"""Resolve a run's configuration and its shards into concrete channel tuples.

    Pure but for one ``INFO`` line per channel the alignment drops: it reads the configuration and
    the shards' attributes and touches nothing else, so the experiment driver and the
    resolved-config record can each call it and cannot disagree. The drops are logged rather than
    returned quietly because which channels a run stopped reading is not recoverable from any
    metric it emits.

    **The source stream is never gated by the budget.** Its slowest channels are the ones carrying
    the contraction envelope, and dropping them to make every reachable lag warm would cost almost
    the whole ``up_ph`` block -- against a lag search that exists to find the $20$ to $120$ s
    contraction-to-deceleration delay. They are kept, the availability mechanism announces per step
    when each arrives, and the residual is measured rather than resolved.

    **The alignment drops for a different reason, and it is not a policy.** A channel whose composed
    delay exceeds $\tau_{\mathrm{ref}}$ can only be brought onto that clock by a negative shift,
    i.e. by being read from a later stored step, which reads raw signal after the anchor. Those
    channels are refused rather than advanced -- four of the $51$ source channels on the target's
    own clock, all fifteen ``up_ph`` surviving -- and the distinction from the paragraph above is
    load-bearing: the budget's drops are how much of a segment this design is willing to spend
    waiting, and these are a correctness requirement.

    **Which clock the source reads on is the one thing that is a decision**, and it is priced rather
    than picked: ``causal_align_reference_source`` moves the source onto a faster reference than the
    target's, which lifts a $20$-$60$ s physiological delay off the near censoring edge of the lag
    window and pays for it in exactly the channels above it -- twelve of $51$ at the shipped
    $288.2672$ s, nine of fifteen ``up_ph`` surviving. Null keeps the single clock and resolves
    byte-for-byte what it did before the key existed.

    Args:
        config: The resolved experiment config mapping. ``model_config.VAE_model`` supplies the
            budget, both alignment references, the expected leg alignment and the geometry;
            ``dataset_config`` supplies the shards and the trim.

    Returns:
        The resolved budget, or ``None`` when ``causal_warmup_budget_steps`` is absent or ``None``
        -- which is every two-sided run in the family and means no warm-up guard at all.

    Raises:
        ValueError: If a reach budget is configured alongside this one; if no shards are
            configured; if the shards are not causal, do not agree, or leave a channel with no
            valid step; if the trim the vectors were rebased at does not produce the declared
            sequence length; if a declared stream width disagrees with the shards; if the budget
            keeps no target channel; if the shards' recorded leg alignment disagrees with the
            configured expectation; if the alignment reference is neither ``'target_max'`` nor a
            number, matches no kept target channel, or leaves a stream with no channel at all; if
            the source reference is not a number, is set against an unaligned target, or matches no
            stored source channel; or if the floor and stride leave a phase with no anchor at all.
    """
    model_config = config.get("model_config") or {}
    vae_config = model_config.get("VAE_model") or {}

    budget_steps = vae_config.get("causal_warmup_budget_steps")
    if budget_steps is None:
        return None
    budget_steps = int(budget_steps)

    # Both guards at once is not a stricter run, it is an incoherent one: the reach quantile is
    # measured on the two-sided Morlet bank, which did not produce these coefficients, so the
    # delays it resolves describe another dataset -- and a delay is a shift, which would make the
    # model read every gated channel late on top of its warm-up.
    reach_budget_s = vae_config.get("causal_reach_budget_s")
    if reach_budget_s is not None:
        raise ValueError(
            f"{BUDGET_KEY}={budget_steps} and {REACH_KEY}={reach_budget_s} are both set. The "
            f"forward reach L95 is an energy quantile of a two-sided filter and is undefined on "
            f"one-sided features: it is measured on the production Morlet bank, which did not "
            f"produce these coefficients. Set {REACH_KEY}: null."
        )

    dataset_config = config.get("dataset_config") or {}
    paths = [
        *(dataset_config.get("vae_train_datasets") or []),
        *(dataset_config.get("vae_test_datasets") or []),
    ]
    if not paths:
        raise ValueError(
            f"{BUDGET_KEY}={budget_steps} is set but dataset_config names no shards under "
            f"vae_train_datasets or vae_test_datasets. The warm-up boundary is a property of the "
            f"data and there is nothing to read it from."
        )

    dataloader_config = dataset_config.get("dataloader_config") or {}
    dataset_kwargs = dataloader_config.get("dataset_kwargs") or {}
    trim_minutes = dataset_kwargs.get("trim_minutes")
    warmup = read_causal_warmup(paths, trim_minutes)

    # The one failure mode no metric can see. A uniformly wrong rebase moves the anchor floor and
    # the validity boundary together, so every "is the target warm at every scored step" readout
    # still reports a clean 1.0 while the model scores pad. The trim is therefore cross-checked
    # against the *other* declaration of the same geometry: the sequence length the network is
    # built at is the stored length minus twice the trim, and the two must agree.
    sequence_length = _require_int(vae_config, "sequence_length")
    for block, kept_steps in sorted(warmup.kept_steps.items()):
        if kept_steps != sequence_length:
            raise ValueError(
                f"{TRIM_KEY}={trim_minutes} trims the shards' {block} block to {kept_steps} "
                f"steps, but {SEQUENCE_KEY}={sequence_length}. The warm-up is rebased by that "
                f"trim, so the two must describe one window: a mismatch moves the anchor floor "
                f"and the validity boundary together and every warm-fraction readout still "
                f"reports 1.0."
            )

    horizon = _require_int(vae_config, "horizon")
    warmup_period = _require_int(vae_config, "warmup_period")
    # Required rather than defaulted, for the same reason as the four keys above and one more of
    # its own: the network takes ``anchor_stride`` too, and its default there is the *inert* one --
    # the dense range every sibling decodes -- so a default here would have to be either that value,
    # which makes the refusal below vacuous, or the shipped tiling, which would disagree with the
    # model built from the same config.
    anchor_stride = _require_int(vae_config, "anchor_stride")
    t_valid = sequence_length - horizon

    use_up_st = bool(vae_config.get("use_up_st", True))
    source_blocks = SOURCE_BLOCKS if use_up_st else SOURCE_BLOCKS[1:]
    target_spans, target_warmup, target_delay_s, target_novelty = _build_stream(
        "target", TARGET_BLOCKS, warmup, _require_int(vae_config, "c_y"), "c_y"
    )
    source_spans, source_warmup, source_delay_s, source_novelty = _build_stream(
        "source", source_blocks, warmup, _require_int(vae_config, "c_u"), "c_u"
    )

    target_keep = tuple(
        index for index, steps in enumerate(target_warmup) if steps <= budget_steps
    )
    if not target_keep:
        raise ValueError(
            f"{BUDGET_KEY}={budget_steps} keeps no target channel at all (the fastest of "
            f"{len(target_warmup)} waits {min(target_warmup)} steps). A stream with zero channels "
            f"builds a model that trains to completion having never read it."
        )

    # The shards' own record of which phase-harmonic operator built them, checked *before* any
    # shift is resolved from the delays it makes true or false.
    _check_leg_alignment(vae_config.get("causal_leg_alignment"), warmup.leg_alignment, paths)

    # The target's reference is resolved from the target stream alone. Under the single-reference
    # scheme it is applied to both streams, which is the whole point of it: a per-stream reference
    # taken as each stream's own MAXIMUM would keep every source channel and restore a known 389 s
    # bias between the two clocks, defeating the correction it looks like.
    reference_delay_s = _resolve_reference_delay(
        vae_config.get("causal_align_reference"),
        tuple(target_delay_s[index] for index in target_keep),
    )
    # The dual scheme differs from that rejection in kind rather than in degree, and the difference
    # is what makes it admissible: this reference is CHOSEN -- priced against channel survival, the
    # freshest-source recency and where a physiological delay lands in the (lag, horizon) window --
    # rather than read off the stream, and the bias it puts on the lag axis is one constant that
    # travels explicitly through the summary, the causality disclosure and the physical-lag
    # identity. A stream's own maximum carries a bias nothing states.
    source_reference_delay_s = _resolve_source_reference_delay(
        vae_config.get("causal_align_reference_source"),
        reference_delay_s,
        source_delay_s,
    )
    target_keep, target_align = _align_stream(
        "target", target_delay_s, target_keep, reference_delay_s
    )
    # The source keep-index is the identity **until** a reference is configured. It is not derived
    # from the warm-up budget and must not be: the source's slowest channels carry the contraction
    # envelope, and gating them on the budget would drop almost the whole ``up_ph`` block against a
    # lag search that exists to find the 20 to 120 s contraction-to-deceleration delay. What the
    # alignment removes is a different set for a different reason -- the channels *above* the
    # reference, which cannot be shifted onto it without reading their own future -- and that is a
    # correctness requirement rather than a warm-up policy.
    #
    # WHICH reference is the dual scheme's one decision, and it is priced in exactly this line. On
    # the target's clock all fifteen ``up_ph`` channels survive and four of 51 source channels drop;
    # on a faster source clock the survivors fall and the readable lag band lifts off the near
    # censoring edge. The drop is logged per channel below either way.
    source_keep, source_align = _align_stream(
        "source",
        source_delay_s,
        tuple(range(len(source_warmup))),
        reference_delay_s if source_reference_delay_s is None else source_reference_delay_s,
    )

    # The forecast clock is resolved AFTER the alignment, over the channels that survived it: the
    # physical clock's reference is the fastest channel the model actually scores, and the input
    # clock copies the shifts the alignment just resolved.
    forecast_clock, forecast_shift, forecast_reference_s = _resolve_target_forecast_shift(
        vae_config.get("causal_target_forecast_clock"),
        tuple(target_delay_s[index] for index in target_keep),
        target_align,
    )
    # A positive shift reads stored step t + H - 1 + s_c at the window's far end, so the anchor
    # ceiling moves down by the largest advance; a negative or absent shift moves nothing.
    max_advance = 0 if forecast_shift is None else max(0, max(forecast_shift, default=0))

    # The worst phase is the last one, whose first anchor is F + S - 1; if that anchor does not
    # exist there is a phase at which the sample contributes no forecast at all, and its share of
    # the epoch is silently dropped. Checked against the EFFECTIVE ceiling: under an advancing
    # forecast clock the last max_advance anchors do not exist either, and a feasibility check
    # taken over T_valid would admit a pairing the model itself refuses.
    if warmup_period + anchor_stride > t_valid - max_advance:
        advance = (
            ""
            if max_advance == 0
            else (
                f", less the {forecast_clock!r} forecast clock's largest advance "
                f"{max_advance}, whose trailing anchors read past the record"
            )
        )
        raise ValueError(
            f"warmup_period={warmup_period} with anchor_stride={anchor_stride} leaves no anchor "
            f"at phase {anchor_stride - 1}: the first would be "
            f"{warmup_period + anchor_stride - 1}, against an anchor ceiling of "
            f"{t_valid - max_advance} (T_valid={t_valid}, sequence_length {sequence_length} - "
            f"horizon {horizon}{advance}). Lower the floor, shorten "
            f"the stride, or shorten the horizon, which lengthens T_valid."
        )

    return WarmupBudget(
        budget_steps=budget_steps,
        trim_minutes=trim_minutes,
        quantile=warmup.quantile,
        reference_delay_s=reference_delay_s,
        source_reference_delay_s=source_reference_delay_s,
        leg_alignment=warmup.leg_alignment,
        target_forecast_clock=forecast_clock,
        target_forecast_shift=forecast_shift,
        target_forecast_reference_s=forecast_reference_s,
        target=StreamWarmup(
            name="target",
            block_spans=target_spans,
            declared_warmup_steps=target_warmup,
            declared_delay_s=target_delay_s,
            declared_novelty_frac=target_novelty,
            keep_index=target_keep,
            align_delays=target_align,
        ),
        source=StreamWarmup(
            name="source",
            block_spans=source_spans,
            declared_warmup_steps=source_warmup,
            declared_delay_s=source_delay_s,
            declared_novelty_frac=source_novelty,
            keep_index=source_keep,
            align_delays=source_align,
        ),
    )


#: Config the tradeoff curve below is drawn for when this module is run directly -- i.e. from an
#: IDE's Run button, with no command line at all. A relative path is resolved against the
#: repository root, not the working directory, because every config in this tree names its shards
#: repo-root-relative. Point it at ``configs/default.yaml`` to draw the curve of the production
#: shards; ``configs/tiny.yaml`` draws the committed fixture's, which is the same staircase.
RUN_CONFIG: str = "teb_vae/lag_attn_cfs/configs/tiny.yaml"

#: Where the curve is written. It is a constant of the **shard**, not of a run, which is why it is
#: produced here rather than into every run directory.
RUN_OUTPUT_DIR: str = "output"


def _cli() -> int:
    """Resolve :data:`RUN_CONFIG`'s budget and draw the tradeoff curve that justifies it.

    Kept out of the module body, and its matplotlib import kept inside it, so that resolving a
    budget -- which every training process does before it builds a network -- costs no figure
    machinery.

    Returns:
        The process exit code.
    """
    from teb_vae.lag_attn.config import load_config
    from teb_vae.lag_attn_cfs.warmup_budget import budget_tradeoff, write_tradeoff_figure

    # The shard paths inside a config are repo-root-relative, and under an IDE Run button the
    # working directory is whatever the IDE chose -- where a relative path resolves to nothing and
    # the read fails as "does not exist" with no mention of the real cause.
    if os.path.abspath(os.getcwd()) != _REPO_ROOT:
        os.chdir(_REPO_ROOT)

    config_path = (
        RUN_CONFIG if os.path.isabs(RUN_CONFIG) else os.path.join(_REPO_ROOT, RUN_CONFIG)
    )
    if not os.path.exists(config_path):
        print(
            f"RUN_CONFIG={RUN_CONFIG!r} does not resolve to a file. Edit RUN_CONFIG near the "
            f"bottom of {os.path.basename(__file__)} to name a config carrying "
            f"{BUDGET_KEY} and its shards."
        )
        return 2

    config = load_config(config_path)
    resolved = resolve_warmup_budget(config)
    if resolved is None:
        print(f"{config_path} sets no {BUDGET_KEY}; there is no budget to draw a tradeoff for.")
        return 2

    vae_config = (config.get("model_config") or {}).get("VAE_model") or {}
    points = budget_tradeoff(
        resolved.target.declared_warmup_steps,
        sequence_length=int(vae_config["sequence_length"]),
        horizon=int(vae_config["horizon"]),
        anchor_stride=int(vae_config["anchor_stride"]),
    )
    print(resolved.summary())
    for point in points:
        print(
            f"  B={point.budget_steps:>4}  kept={point.kept:>4}  anchors={point.anchors:>4}  "
            f"tiles={point.tiles:>3}"
        )
    path = write_tradeoff_figure(
        points, Path(RUN_OUTPUT_DIR), shipped_budget=resolved.budget_steps
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
