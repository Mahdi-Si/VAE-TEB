r"""How far into its own future each input channel reads, and what a reach budget does about it.

The stored features are two-sided wavelet transforms. The value at decimated step $t$ of a
scattering or phase-harmonic channel is a weighted average over raw samples on *both* sides of
$t$, so a model conditioning on "the past up to $t$" is in fact conditioning on part of the
interval it is being asked to forecast. Per channel, the size of that violation is the
**forward reach** $L_{95}$: the smallest $D > 0$ enclosing $95\%$ of the filter's energy at taps
strictly after $t$.

This module answers two questions:

1. *How bad is it, channel by channel?* -- :func:`block_reach_seconds`, which rebuilds the
   production filter bank and reads the reaches straight off it.
2. *Given a budget, which channels may be used and how stale must each be?* --
   :func:`resolve_channel_budget`, which keeps the channels whose reach fits the budget and
   assigns each survivor the delay $\delta_c = \lceil \mathrm{reach}_c / \Delta \rceil$ steps
   that pushes its reach back behind the anchor's causal endpoint.

**The bank is built at the stored length $N = 5280$, not at the trimmed $4800$.** Reach is a
property of the filters that produced the coefficients, and those were computed on the untrimmed
segment; the loader's later crop moves which steps survive, not what each step contains.
Rebuilding at $4800$ changes the padded length, hence every realised filter, hence every reach --
silently and by enough to matter.

Everything here is numpy over a filter bank: no dataset, no I/O, milliseconds. There is therefore
no generated asset to keep in step with the code and no staleness problem -- the reaches are
recomputed whenever they are asked for, from the same constants the pipeline used.

This module sits outside ``nets/`` because it depends on ``numpy`` and ``kymatio``, which the
network layer may not. Nothing in ``nets/`` imports it; the resolved integers are passed in.

lean-limit: the reach bound is analytic (an energy quantile of the filter, not a hard support),
so a channel at its budget still leaks the $5\%$ tail; replace $L_{95}$ with a hard-support bound
when the transforms themselves are rebuilt causally.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from teb_vae.lag_attn.eval.representation_capacity_probe import (
    N_RAW,
    build_filter_bank,
    forward_reach,
    select_phase_pairs,
)

# Seconds per decimated step, imported rather than restated: this module divides by $\Delta$ to
# size the delay, and the lag report multiplies by it to add that delay back. Two definitions of
# one dataset fact would let the guard and the reported lag desynchronise silently. Importing a
# net module from outside `nets/` is the permitted direction -- the rule is that `nets/` must not
# import this module -- and `lag_report` is plain torch.
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: Band edges of the stored phase-harmonic selections, in Hz, as
#: ``hdf5_dataset/new_pipeline/create_new_pipeline.py`` sets them. The target block spans the
#: whole usable range; the source block is restricted to the deceleration band.
TARGET_PHASE_BAND_HZ = (0.008, 1.00)
SOURCE_PHASE_BAND_HZ = (0.008, 0.05)

#: Stored block names, in the channel order the model concatenates them.
TARGET_BLOCKS = ("fhr_st", "fhr_ph")
SOURCE_BLOCKS = ("up_st", "up_ph")


@lru_cache(maxsize=1)
def block_reach_seconds() -> Dict[str, Tuple[float, ...]]:
    r"""Forward reach $L_{95}$ in seconds, per channel, for each stored feature block.

    The scattering block is $S_0 = x \star \phi$ followed by the $42$ first-order envelopes, in
    that order, matching how the pipeline stores it. A phase-harmonic channel multiplies two
    wavelet responses at the same $t$ and then $\phi$-smooths the product, so its reach is the
    slower wavelet's reach plus the low-pass reach.

    Cached because the result is a constant of the production filter bank, and the bank costs a
    few hundred milliseconds of FFTs to rebuild.

    Returns:
        ``{'fhr_st': (43 reaches), 'fhr_ph': (66,), 'up_st': (43,), 'up_ph': (15,)}``, each in
        the stored channel order.
    """
    bank = build_filter_bank()
    reach_psi = np.array([forward_reach(bank, bank.psi[index]) for index in range(bank.n_filters)])
    reach_phi = forward_reach(bank, bank.phi)
    scattering = np.concatenate([[reach_phi], reach_psi])

    def phase_reach(band: Tuple[float, float]) -> np.ndarray:
        # select_phase_pairs emits pairs in ascending (i, j) index order, which is the order
        # KymatioPhaseScattering1D._build_coupling_indices builds them in and therefore the
        # order the shard writer's boolean mask preserves. See tests/test_channel_reach.py.
        pairs = select_phase_pairs(bank, band[0], band[1])
        return np.array([max(reach_psi[i], reach_psi[j]) + reach_phi for i, j in pairs])

    return {
        "fhr_st": tuple(float(value) for value in scattering),
        "fhr_ph": tuple(float(value) for value in phase_reach(TARGET_PHASE_BAND_HZ)),
        "up_st": tuple(float(value) for value in scattering),
        "up_ph": tuple(float(value) for value in phase_reach(SOURCE_PHASE_BAND_HZ)),
    }


def stream_reach_seconds(*, use_up_st: bool = True) -> Dict[str, Tuple[float, ...]]:
    r"""Per-channel reach for the two streams the model actually consumes.

    The model concatenates its blocks before the input adapters -- the target stream is
    ``[scattering, phase]`` and the source stream is the same pair, or the phase block alone
    under the ``use_up_st=False`` ablation -- so the reach vectors are concatenated the same way.

    Args:
        use_up_st: Whether the source stream includes its scattering block.

    Returns:
        ``{'target': (c_y reaches), 'source': (c_u reaches)}``.
    """
    blocks = block_reach_seconds()
    source_blocks = SOURCE_BLOCKS if use_up_st else SOURCE_BLOCKS[1:]
    return {
        "target": tuple(value for name in TARGET_BLOCKS for value in blocks[name]),
        "source": tuple(value for name in source_blocks for value in blocks[name]),
    }


def resolve_channel_budget(
    reach: Tuple[float, ...],
    budget_s: Optional[float],
    warmup_period: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    r"""Turn a reach budget into surviving channels and their delays.

    A channel survives when $\mathrm{reach}_c \le \mathrm{budget}$, and a survivor is read
    $\delta_c = \lceil \mathrm{reach}_c / \Delta \rceil$ steps late, which is the smallest delay
    for which the channel's forward reach no longer crosses the anchor's causal endpoint. The
    comparison against the budget is inclusive: a channel whose reach *equals* the budget
    satisfies it.

    Args:
        reach: Per-channel forward reach in seconds, in channel order.
        budget_s: Maximum admissible reach in seconds, or ``None`` for no guard -- every channel
            kept, every delay zero.
        warmup_period: The loss warm-up $w$ in steps. The first $\max_c \delta_c$ steps of a
            delayed stream are partly zero-filled, so they must fall inside the warm-up the loss
            already discards.

    Returns:
        ``(keep_index, delays)``: the surviving channel indices in ascending order, and one
        delay per survivor in the same order.

    Raises:
        ValueError: If the budget admits no channel at all, or if the resulting maximum delay
            exceeds ``warmup_period``. The warm-up comparison is strictly greater-than, not
            greater-or-equal: at the $120$ s budget the maximum delay is exactly $30$, which is
            also the shipped warm-up, and that configuration must be allowed.
    """
    if budget_s is None:
        return tuple(range(len(reach))), tuple(0 for _ in reach)

    budget = float(budget_s)
    keep_index = tuple(index for index, value in enumerate(reach) if value <= budget)
    if not keep_index:
        raise ValueError(
            f"causal_reach_budget_s={budget_s} keeps no channel at all (the fastest of "
            f"{len(reach)} reaches {min(reach):.1f} s). A stream with zero channels builds a "
            f"model that trains to completion having never read it."
        )
    delays = tuple(
        int(math.ceil(reach[index] / SECONDS_PER_STEP)) for index in keep_index
    )

    max_delay = max(delays)
    if max_delay > int(warmup_period):
        raise ValueError(
            f"causal_reach_budget_s={budget_s} needs a maximum delay of {max_delay} steps but "
            f"warmup_period is {int(warmup_period)}. The first max(delta) steps of a delayed "
            f"stream are partly zero-filled, so they must fall inside the warm-up the loss "
            f"already discards: raise warmup_period to at least {max_delay}, or lower "
            f"causal_reach_budget_s to at most {int(warmup_period) * SECONDS_PER_STEP:g}."
        )
    return keep_index, delays


def _split_counts(
    block_names: Tuple[str, ...],
    blocks: Dict[str, Tuple[float, ...]],
    keep_index: Tuple[int, ...],
) -> List[Tuple[str, int, int]]:
    """Attribute a stream's surviving channels back to the blocks they came from.

    The stream is the blocks concatenated in order, so block $k$ owns the half-open index range
    $[\\text{offset}, \\text{offset} + w_k)$ and its survivors are the keep-index entries falling
    in it.

    Args:
        block_names: The blocks making up this stream, in concatenation order.
        blocks: Per-block reach vectors, for their widths.
        keep_index: The stream's resolved surviving channel indices.

    Returns:
        One ``(name, kept, declared)`` triple per block.
    """
    survivors = set(keep_index)
    counts: List[Tuple[str, int, int]] = []
    offset = 0
    for name in block_names:
        width = len(blocks[name])
        counts.append(
            (name, sum(1 for i in range(offset, offset + width) if i in survivors), width)
        )
        offset += width
    return counts


@dataclass(frozen=True)
class ChannelBudget:
    r"""The resolved causal guard for one run: which channels survive, and how stale each is.

    Attributes:
        budget_s: The configured reach budget in seconds.
        target_keep_index: Surviving target channel indices, into the declared $c_y$.
        target_delays: One delay per surviving target channel, in steps.
        source_keep_index: Surviving source channel indices, into the declared $c_u$.
        source_delays: One delay per surviving source channel, in steps.
        block_counts: Surviving channel count per stored block, for the startup log. The counts
            are per block rather than per stream because that is the resolution at which the
            question "did ``up_ph`` survive this budget?" is asked.
    """

    budget_s: float
    target_keep_index: Tuple[int, ...]
    target_delays: Tuple[int, ...]
    source_keep_index: Tuple[int, ...]
    source_delays: Tuple[int, ...]
    block_counts: Tuple[Tuple[str, int, int], ...]

    @property
    def max_delay(self) -> int:
        """Largest delay over both streams, in steps."""
        return max((*self.target_delays, *self.source_delays), default=0)

    def as_record(self) -> Dict[str, Any]:
        """Render the budget as plain data, for writing into a run's resolved configuration.

        A record only: nothing reads it back. It exists so a run's causal standing is readable
        from the artefacts it left behind rather than re-derived from the code that produced it.

        Returns:
            A YAML-safe dict.
        """
        return {
            "causal_reach_budget_s": self.budget_s,
            "seconds_per_step": SECONDS_PER_STEP,
            "max_delay_steps": self.max_delay,
            "channels_kept_per_block": {
                name: {"kept": kept, "declared": declared}
                for name, kept, declared in self.block_counts
            },
            "target_channels_kept": len(self.target_keep_index),
            "source_channels_kept": len(self.source_keep_index),
            "target_keep_index": list(self.target_keep_index),
            "target_delays": list(self.target_delays),
            "source_keep_index": list(self.source_keep_index),
            "source_delays": list(self.source_delays),
        }

    def summary(self) -> str:
        """One line naming the surviving counts per block and the maximum delay."""
        blocks = ", ".join(
            f"{name} {kept}/{declared}" for name, kept, declared in self.block_counts
        )
        return (
            f"causal reach budget {self.budget_s:g} s: {blocks}; "
            f"c_y {len(self.target_keep_index)}, c_u {len(self.source_keep_index)}, "
            f"max delay {self.max_delay} steps "
            f"({self.max_delay * SECONDS_PER_STEP:g} s)"
        )


def resolve_stream_budgets(vae_config: Mapping[str, Any]) -> Optional[ChannelBudget]:
    r"""Resolve a ``model_config.VAE_model`` block's reach budget into concrete channel tuples.

    Pure: it reads the configuration and the filter bank and touches nothing else, so the
    experiment driver and the resolved-config record can each call it and cannot disagree.

    Args:
        vae_config: The ``VAE_model`` configuration block. ``causal_reach_budget_s`` selects the
            budget; ``use_up_st`` and ``warmup_period`` shape the streams and bound the delay.

    Returns:
        The resolved budget, or ``None`` when no budget is configured -- which is the unguarded
        default and means the model is built with no gather and no delay at all.

    Raises:
        ValueError: If the budget keeps no channel in a stream, if its maximum delay exceeds
            ``warmup_period``, or if the declared channel widths disagree with the filter bank's
            own channel counts (which would make the resolved indices point at the wrong
            channels).
    """
    budget_s = vae_config.get("causal_reach_budget_s")
    if budget_s is None:
        return None

    use_up_st = bool(vae_config.get("use_up_st", True))
    warmup_period = int(vae_config.get("warmup_period", 30))
    streams = stream_reach_seconds(use_up_st=use_up_st)

    # The keep-index is positional into the declared width, so a declared width that disagrees
    # with the bank would gather the wrong channels rather than fail.
    for name, declared_key in (("target", "c_y"), ("source", "c_u")):
        declared = vae_config.get(declared_key)
        if declared is not None and int(declared) != len(streams[name]):
            raise ValueError(
                f"causal_reach_budget_s is set but {declared_key}={declared} disagrees with the "
                f"filter bank's {len(streams[name])} {name} channels (use_up_st={use_up_st}). "
                f"The resolved keep-index is positional into the declared width, so it would "
                f"select the wrong channels rather than fail."
            )

    target_keep, target_delays = resolve_channel_budget(
        streams["target"], budget_s, warmup_period
    )
    source_keep, source_delays = resolve_channel_budget(
        streams["source"], budget_s, warmup_period
    )

    # Per-block counts are *split out of the resolved keep-indices* rather than recomputed with a
    # second comparison against the budget. The startup log and the run record would otherwise be
    # produced by an independent predicate, and could describe a different guard than the model
    # got if the boundary semantics ever diverged.
    blocks = block_reach_seconds()
    source_block_names = SOURCE_BLOCKS if use_up_st else SOURCE_BLOCKS[1:]
    block_counts = tuple(
        _split_counts(TARGET_BLOCKS, blocks, target_keep)
        + _split_counts(source_block_names, blocks, source_keep)
    )
    return ChannelBudget(
        budget_s=float(budget_s),
        target_keep_index=target_keep,
        target_delays=target_delays,
        source_keep_index=source_keep,
        source_delays=source_delays,
        block_counts=block_counts,
    )


if __name__ == "__main__":
    # The budget table, for checking the tradeoff by eye: how many channels each budget keeps and
    # how stale it makes them.
    _blocks = block_reach_seconds()
    print(f"forward reach L95 per block (filter bank at N = {N_RAW})")
    print(f"  {'block':>8} {'n':>4} {'min':>8} {'median':>8} {'max':>9}")
    for _name, _values in _blocks.items():
        _array = np.array(_values)
        print(
            f"  {_name:>8} {_array.size:>4} {_array.min():>8.1f} "
            f"{np.median(_array):>8.1f} {_array.max():>9.1f}"
        )
    # Widths computed, not written: this table is the source DESIGN.md's is regenerated from, so
    # a hardcoded denominator would print a stale figure straight into the design record after a
    # phase-selection change.
    _streams = stream_reach_seconds()
    _c_y, _c_u = len(_streams["target"]), len(_streams["source"])
    print("\nreach budget -> surviving channels and maximum delay")
    print(f"  {'budget s':>9} {'c_y':>8} {'c_u':>8} {'max delay':>10}")
    for _budget in (None, 240.0, 120.0, 60.0, 32.0):
        _resolved = resolve_stream_budgets(
            {"causal_reach_budget_s": _budget, "use_up_st": True, "warmup_period": 60}
        )
        if _resolved is None:
            print(f"  {'null':>9} {f'{_c_y}/{_c_y}':>8} {f'{_c_u}/{_c_u}':>8} {0:>10}")
            continue
        print(
            f"  {_resolved.budget_s:>9.0f} "
            f"{f'{len(_resolved.target_keep_index)}/{_c_y}':>8} "
            f"{f'{len(_resolved.source_keep_index)}/{_c_u}':>8} "
            f"{_resolved.max_delay:>10}"
        )
