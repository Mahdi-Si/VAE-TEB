r"""How long each causal input channel waits before it is honest, and what a budget does about it.

A one-sided filter reads only the past, so before its warm-up has passed its output is a function
of the assumed pre-recording history rather than of the recording. Per channel, the size of that
region is $W'_c$, the leading delay enclosing $95\%$ of the kernel's energy, rebased into the
coordinates of the window the loader actually serves.

This module answers one question: *given a warm-up budget, which channels may be used, and how long
does each of them wait?* -- :func:`resolve_warmup_budget`, which reads the boundary off the
configured shards, keeps the target channels whose warm-up fits the budget, and returns the four
tuples the network constructor takes.

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
"""
from __future__ import annotations

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

from hdf5_dataset.hdf5_dataset import CausalWarmup, read_causal_warmup  # noqa: E402

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
        keep_index: Surviving channel indices into the declared width, strictly ascending -- the
            order ``ChannelGate``'s gather requires.
    """

    name: str
    block_spans: Tuple[Tuple[str, int, int], ...]
    declared_warmup_steps: Tuple[int, ...]
    keep_index: Tuple[int, ...]

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
        """One line naming the surviving counts per block and the slowest survivor's wait."""
        blocks = ", ".join(
            f"{name} {kept}/{declared}" for name, kept, declared in self.block_counts()
        )
        return (
            f"{self.name}: {blocks}; {self.kept_width}/{self.declared_width} channels, "
            f"warm-up 0-{self.max_warmup} steps"
        )


@dataclass(frozen=True)
class WarmupBudget:
    r"""The resolved causal warm-up guard for one run: which channels survive, and how long each waits.

    Attributes:
        budget_steps: The configured threshold $B_{\mathrm{cfg}}$ in decimated steps. A channel
            survives when $W'_c \le B_{\mathrm{cfg}}$.
        trim_minutes: The trim the warm-up vectors are expressed against, which is the loader's own.
        quantile: The ``causal_warmup_quantile`` every configured shard was built at.
        target: The target stream, which the budget gates.
        source: The source stream, which it does not; see :func:`resolve_warmup_budget`.
    """

    budget_steps: int
    trim_minutes: Optional[float]
    quantile: Optional[float]
    target: StreamWarmup
    source: StreamWarmup

    def summary(self) -> str:
        """One line for the startup log, naming both streams and the threshold that produced them.

        The quantile is rounded for reading: it comes back from the shard as the ``float32`` it was
        stored as, and ``0.949999988079071`` in a log line is noise rather than provenance.
        """
        quantile = "unknown" if self.quantile is None else f"{self.quantile:.3g}"
        return (
            f"causal warm-up budget {self.budget_steps} steps "
            f"(quantile {quantile}, trim_minutes {self.trim_minutes}): "
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
) -> Tuple[Tuple[Tuple[str, int, int], ...], Tuple[int, ...]]:
    """Concatenate a stream's blocks into one declared-width warm-up vector.

    Args:
        name: ``'target'`` or ``'source'``, for the refusal messages.
        blocks: The stored blocks making up this stream, in concatenation order.
        warmup: The boundary read off the shards.
        declared_width: The stream width the config declares.
        declared_key: The config key that declared it, for the refusal message.

    Returns:
        ``(block_spans, declared_warmup_steps)``.

    Raises:
        ValueError: If a block is absent from the shards, or if the concatenated width disagrees
            with the declared one.
    """
    missing = [block for block in blocks if block not in warmup.warmup_steps]
    if missing:
        raise ValueError(
            f"the configured shards store no {missing} block, so the {name} stream cannot be "
            f"assembled; they store {sorted(warmup.warmup_steps)}."
        )

    spans: List[Tuple[str, int, int]] = []
    values: List[int] = []
    for block in blocks:
        vector = warmup.warmup_steps[block]
        spans.append((block, len(values), len(values) + int(vector.size)))
        values.extend(int(step) for step in vector)

    if declared_width != len(values):
        raise ValueError(
            f"{declared_key}={declared_width} disagrees with the shards' {len(values)} {name} "
            f"channels for blocks {tuple(blocks)}. The keep-index and the warm-up vector are "
            f"positional into the declared width, so the model would gather the wrong channels "
            f"and wait the wrong number of steps for each rather than fail."
        )
    return tuple(spans), tuple(values)


def resolve_warmup_budget(config: Mapping[str, Any]) -> Optional[WarmupBudget]:
    r"""Resolve a run's configuration and its shards into concrete channel tuples.

    Pure: it reads the configuration and the shards' attributes and touches nothing else, so the
    experiment driver and the resolved-config record can each call it and cannot disagree.

    **The source stream is never gated.** Its keep-index is the identity by construction. Its
    slowest channels are the ones carrying the contraction envelope, and dropping them to make
    every reachable lag warm would cost almost the whole ``up_ph`` block -- against a lag search
    that exists to find the $20$ to $120$ s contraction-to-deceleration delay. They are kept, the
    availability mechanism announces per step when each arrives, and the residual is measured
    rather than resolved.

    Args:
        config: The resolved experiment config mapping. ``model_config.VAE_model`` supplies the
            budget and the geometry; ``dataset_config`` supplies the shards and the trim.

    Returns:
        The resolved budget, or ``None`` when ``causal_warmup_budget_steps`` is absent or ``None``
        -- which is every two-sided run in the family and means no warm-up guard at all.

    Raises:
        ValueError: If a reach budget is configured alongside this one; if no shards are
            configured; if the shards are not causal, do not agree, or leave a channel with no
            valid step; if the trim the vectors were rebased at does not produce the declared
            sequence length; if a declared stream width disagrees with the shards; if the budget
            keeps no target channel; or if the floor and stride leave a phase with no anchor at
            all.
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
    target_spans, target_warmup = _build_stream(
        "target", TARGET_BLOCKS, warmup, _require_int(vae_config, "c_y"), "c_y"
    )
    source_spans, source_warmup = _build_stream(
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

    # The worst phase is the last one, whose first anchor is F + S - 1; if that anchor does not
    # exist there is a phase at which the sample contributes no forecast at all, and its share of
    # the epoch is silently dropped.
    if warmup_period + anchor_stride > t_valid:
        raise ValueError(
            f"warmup_period={warmup_period} with anchor_stride={anchor_stride} leaves no anchor "
            f"at phase {anchor_stride - 1}: the first would be "
            f"{warmup_period + anchor_stride - 1}, against T_valid={t_valid} "
            f"(sequence_length {sequence_length} - horizon {horizon}). Lower the floor, shorten "
            f"the stride, or shorten the horizon, which lengthens T_valid."
        )

    return WarmupBudget(
        budget_steps=budget_steps,
        trim_minutes=trim_minutes,
        quantile=warmup.quantile,
        target=StreamWarmup(
            name="target",
            block_spans=target_spans,
            declared_warmup_steps=target_warmup,
            keep_index=target_keep,
        ),
        source=StreamWarmup(
            name="source",
            block_spans=source_spans,
            declared_warmup_steps=source_warmup,
            # Identity by construction, not by arithmetic that happens to keep everything: the
            # source is not gated, and a keep-index derived from the budget would quietly start
            # dropping the contraction envelope the moment the budget moved.
            keep_index=tuple(range(len(source_warmup))),
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
