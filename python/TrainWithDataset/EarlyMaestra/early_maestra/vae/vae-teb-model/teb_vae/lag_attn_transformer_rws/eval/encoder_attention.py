r"""The encoder self-attention the model computes and never materialises.

``CausalSelfAttention.forward`` attends through fused
``torch.nn.functional.scaled_dot_product_attention``, which returns $O_{h,t}$ without ever
building the $(B, H_e, T, T)$ probability tensor -- and the forward contract is pinned at exactly
twenty keys against the comparison model, so the probabilities cannot be returned either. This
module **recomputes** them.

**Every operand is read off the module.** A forward pre-hook captures each block's input, and the
recompute re-applies that module's own ``norm``, ``q_proj``, ``k_proj``, head reshape and ``rope``
before an explicit softmax under the module's own admitted-key mask. Nothing is reconstructed from
a config file and no geometry is re-derived, so a recompute that disagreed with what the model
computed would have to disagree with the model's own parameters. It is checked rather than
asserted: ``tests/test_eval_encoder_attention.py`` contracts these probabilities with $V$, pushes
them through ``out_proj``, and requires the module's actual output back.

**The mask comes from the module, not from a caller.** A windowed block carries its band as a
non-persistent ``attn_mask`` buffer and that buffer *is* what the fused kernel was handed, so it is
read rather than rebuilt. A full-prefix block carries none -- its causality comes from the kernel's
own ``is_causal`` flag -- and the lower triangle that flag means is built here. Both cases admit
$j = t$, so no row is fully masked and there is no NaN path through the softmax.

Two reductions run over the probabilities and neither retains them:

* **Per-head entropy against its truncation-aware ceiling.** At anchor $t$ a block admits
  $\min(t + 1, c)$ keys -- $c = T$ for the target, $c = W_U$ for the source -- so the attainable
  ceiling is $\operatorname{mean}_t \log \min(t + 1, c)$ and never $\log T$. The entropy is taken
  **per anchor and then averaged**, never as the entropy of the averaged profile: a mixture's
  entropy is at least the mean of the entropies mixed, so the second reports a head whose focus
  *shifts* across the segment as one that has no focus at all.
* **Attention mass by temporal distance** $t - j$, per head. This is what tests the design claim
  directly -- that the target encoder gives the prior content-dependent access to long-range
  history the recurrent branch could not, and that the source encoder stays inside its window.

Both reduce inside the batch loop, so peak memory is one block's map rather than the pass's, and
the accumulators are the same size whether the pass sees eight segments or eight thousand.

This module names no class of either package. It reads ``target_encoder`` / ``source_encoder``,
their ``attention_blocks`` and each block's ``attn`` structurally, which is what keeps the whole
coupling to this architecture inside ``binding.py``.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.metrics import batch_guids, model_inputs
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP

#: The two history streams, in the order every table and figure reports them.
TARGET_STREAM = "target"
SOURCE_STREAM = "source"
STREAMS: Tuple[str, str] = (TARGET_STREAM, SOURCE_STREAM)

#: The pooled cohort, beside the clinical classes. A label rather than a separate table, so a
#: reader groups on one column instead of joining two files; ``all`` is not a class name the
#: dataset assigns, so it cannot collide with one.
POOLED_CLASS = "all"

#: Quantiles of the mass-by-distance profile the measured reach is reported at. The median says
#: where the mass sits; the $95$th says how far the tail actually goes, which is the number the
#: structural bound is a bound on.
REACH_QUANTILES: Tuple[float, float] = (0.5, 0.95)


# =============================================================================
# Finding the blocks
# =============================================================================
@dataclass(frozen=True)
class BlockRef:
    """One causal self-attention block, and where it sits.

    Attributes:
        stream: ``'target'`` or ``'source'`` -- which encoder owns the block.
        index: Its position in that encoder's stack, from $0$.
        module: The ``CausalSelfAttention`` itself. Held rather than looked up again, so the
            recompute and the hook cannot end up on two different objects.
        window: The block's causal window $W$ in steps, or ``None`` for the full causal prefix.
            Read off the module, because that is what decided the mask the model attended under.
    """

    stream: str
    index: int
    module: Any
    window: Optional[int]

    @property
    def key(self) -> Tuple[str, int]:
        """``(stream, index)``, the key every accumulator and captured input is stored under."""
        return (self.stream, self.index)


def stream_encoder(model: Any, stream: str) -> Any:
    """Return one stream's encoder, or raise naming what is missing.

    Args:
        model: The rebuilt net.
        stream: ``'target'`` or ``'source'``.

    Returns:
        The encoder module.

    Raises:
        ValueError: If ``stream`` is neither of the two.
        AttributeError: If the model exposes no such encoder. A ``getattr`` default would report a
            model whose encoders were renamed as a model with no attention to describe.
    """
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {list(STREAMS)}, got {stream!r}")
    name = f"{stream}_encoder"
    if not hasattr(model, name):
        raise AttributeError(
            f"{type(model).__name__} carries no {name!r}, which the encoder-attention readout "
            f"reads. An evaluation cannot describe an encoder it cannot reach."
        )
    return getattr(model, name)


def attention_blocks(model: Any) -> List[BlockRef]:
    """Return every causal self-attention block under both encoders, in stream and stack order.

    Args:
        model: The rebuilt net.

    Returns:
        One :class:`BlockRef` per block, targets first.

    Raises:
        AttributeError: If an encoder exposes no ``attention_blocks``, or a block no ``attn``.
    """
    refs: List[BlockRef] = []
    for stream in STREAMS:
        encoder = stream_encoder(model, stream)
        if not hasattr(encoder, "attention_blocks"):
            raise AttributeError(
                f"the {stream} encoder carries no 'attention_blocks'; this readout profiles those "
                f"blocks and has nothing to hook without them"
            )
        for index, block in enumerate(encoder.attention_blocks):
            if not hasattr(block, "attn"):
                raise AttributeError(
                    f"{stream} attention block {index} carries no 'attn' submodule to profile"
                )
            module = block.attn
            refs.append(BlockRef(stream, index, module, module.window))
    return refs


@contextmanager
def captured_block_inputs(refs: Sequence[BlockRef]) -> Iterator[Dict[Tuple[str, int], torch.Tensor]]:
    """Capture each block's input for the duration of the block, then remove every hook.

    A **pre**-hook, so what is captured is the residual stream as the block received it -- before
    its own ``norm`` -- which is exactly the operand the recompute re-normalises. Removal is in a
    ``finally``: a hook that outlived a failed pass would keep firing on every later forward in the
    process, silently holding one batch's activations alive and re-filling the store from analyses
    that never asked for it.

    Args:
        refs: The blocks to hook.

    Yields:
        A dict from ``(stream, index)`` to the most recent input each block saw. Overwritten per
        forward, so a caller reads it between passes rather than accumulating it.
    """
    store: Dict[Tuple[str, int], torch.Tensor] = {}
    handles: List[Any] = []

    def _hook_for(ref: BlockRef) -> Any:
        def _hook(module: Any, args: Tuple[Any, ...]) -> None:
            store[ref.key] = args[0].detach()

        return _hook

    try:
        for ref in refs:
            handles.append(ref.module.register_forward_pre_hook(_hook_for(ref)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


# =============================================================================
# The recompute
# =============================================================================
def admitted_keys(module: Any, seq_len: int, *, device: Any) -> torch.Tensor:
    r"""Return the block's own $(T, T)$ boolean mask: ``True`` where key $j$ participates at $t$.

    The two cases are the two mechanisms ``CausalSelfAttention`` uses, never mixed:

    * **Windowed.** The band $0 \le t - j < W$ is a buffer the module built at construction and
      handed to the kernel verbatim, so it is sliced rather than rebuilt -- a second construction
      could agree today and drift tomorrow.
    * **Full prefix.** There is no buffer at all; causality came from the kernel's ``is_causal``
      flag, and the lower triangle is what that flag means.

    Args:
        module: The ``CausalSelfAttention`` block.
        seq_len: The batch's sequence length $T$.
        device: Where to build the triangle in the unwindowed case.

    Returns:
        A ``(seq_len, seq_len)`` boolean tensor. Every row admits $j = t$, in both cases, so no row
        is fully masked and the softmax below has no NaN path.

    Raises:
        ValueError: If a windowed module carries no mask buffer, or one too short for this batch.
    """
    if module.window is None:
        return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    mask = module.attn_mask
    if mask is None:
        raise ValueError(
            f"a block with window={module.window} carries no attn_mask buffer, so the band the "
            f"model attended under cannot be read off it"
        )
    if int(mask.shape[0]) < int(seq_len):
        raise ValueError(
            f"the block's mask covers {int(mask.shape[0])} steps and this batch is {seq_len}; the "
            f"mask is built once at max_seq_len, so a longer input was never servable"
        )
    return mask[:seq_len, :seq_len]


@torch.no_grad()
def attention_probabilities(module: Any, x: torch.Tensor) -> torch.Tensor:
    r"""Recompute one block's attention probabilities from its own parameters.

    $$
    e_{h,t,j} = \frac{\widetilde q_{h,t}^{\mathsf T}\widetilde k_{h,j}}{\sqrt{d_h}}, \qquad
    P_{h,t,\cdot} = \operatorname{softmax}_{j \in \mathcal A_t}\left(e_{h,t,\cdot}\right),
    $$

    with $\widetilde q$, $\widetilde k$ rotated by their absolute positions and $\mathcal A_t$ the
    module's own admitted keys. Every operand -- the norm, both projections, the rotary tables and
    the mask -- is the module's, so this is the model's computation rather than a model of it.

    Attention-probability dropout is structurally zero in this architecture, so ``train()`` and
    ``eval()`` give the same probabilities and the caller does not have to state which it is in.

    Args:
        module: The ``CausalSelfAttention`` block.
        x: The input that block received, ``(B, T, d)`` -- from :func:`captured_block_inputs`.

    Returns:
        ``(B, H_e, T, T)`` probabilities. Rows sum to $1$; masked entries are exactly $0$, because
        an ``-inf`` score exponentiates to zero rather than to something small.
    """
    batch, seq_len, _ = x.shape
    hidden = module.norm(x)
    shape = (batch, seq_len, module.num_heads, module.d_head)
    query = module.rope(module.q_proj(hidden).view(shape).transpose(1, 2))
    key = module.rope(module.k_proj(hidden).view(shape).transpose(1, 2))
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(module.d_head))
    mask = admitted_keys(module, seq_len, device=scores.device)
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)


# =============================================================================
# The streamed reductions
# =============================================================================
@dataclass(frozen=True)
class BlockSampleStats:
    r"""One block's reductions for one batch, per sample -- already free of the $(T, T)$ map.

    Attributes:
        entropy_nats: ``(B, H_e)`` mean over the scored anchors of the per-anchor entropy.
        ceiling_nats: ``(B,)`` mean over those anchors of $\log|\mathcal A_t|$, the most any head
            could carry there.
        entropy_ratio: ``(B, H_e)`` the first divided by the second, per sample, so the ratio is a
            measurement on that sample's own anchors rather than a pooled approximation.
        distance_mass: ``(B, H_e, T)`` attention mass by $t - j$, normalised per sample and head to
            sum to $1$.
        mean_distance: ``(B, H_e)`` the mass-weighted mean of $t - j$, in steps.
        n_anchors: How many anchors were scored, the same for every sample.
    """

    entropy_nats: np.ndarray
    ceiling_nats: np.ndarray
    entropy_ratio: np.ndarray
    distance_mass: np.ndarray
    mean_distance: np.ndarray
    n_anchors: int


def block_sample_stats(
    probs: torch.Tensor, *, anchors: Tuple[int, int], admitted_counts: torch.Tensor
) -> BlockSampleStats:
    r"""Reduce one block's probability map to per-sample statistics.

    The entropy is $H_{h,t} = -\sum_j P_{h,t,j}\log P_{h,t,j}$ **per anchor**, averaged afterwards.
    Taking the entropy of an anchor-averaged profile instead would be a different quantity and a
    systematically larger one -- entropy is concave, so a mixture's entropy is at least the mean of
    the entropies mixed -- and a head whose attention *moves* across the segment would read as one
    that never focuses.

    Args:
        probs: ``(B, H_e, T, T)`` probabilities from :func:`attention_probabilities`.
        anchors: ``(start, stop)`` half-open range of scored anchors.
        admitted_counts: ``(T,)`` number of keys admitted at each step, from the block's own mask.

    Returns:
        The per-sample statistics.

    Raises:
        ValueError: If the anchor range is empty. A block scored over no anchors measures nothing,
            and reporting a zero entropy for it would read as a perfectly focused head.
    """
    start, stop = int(anchors[0]), int(anchors[1])
    if stop <= start:
        raise ValueError(
            f"the scored anchor range [{start}, {stop}) is empty, so there is no attention row to "
            f"profile; the geometry's trained range is what this range comes from"
        )
    rows = probs[:, :, start:stop, :]
    batch, heads, n_anchors, seq_len = rows.shape

    entropy = torch.special.entr(rows).sum(dim=-1)
    entropy_mean = entropy.mean(dim=-1).to(torch.float64).cpu().numpy()

    # Constant across samples -- the encoder does no data-driven validity masking, so every sample
    # admits the same keys at the same step -- but computed rather than assumed, so a block that
    # ever gained one would report its own ceiling instead of a shared claim about it.
    ceiling = float(
        torch.log(admitted_counts[start:stop].to(torch.float64)).mean().item()
    )
    ceiling_nats = np.full(batch, ceiling, dtype=np.float64)

    positions = torch.arange(seq_len, device=rows.device)
    # Masked entries carry exactly zero mass, so the negative distances they would land on are
    # clamped onto bin 0 without contributing anything to it.
    distance = (positions[start:stop, None] - positions[None, :]).clamp(min=0)
    histogram = torch.zeros((batch * heads, seq_len), dtype=rows.dtype, device=rows.device)
    histogram.index_add_(1, distance.reshape(-1), rows.reshape(batch * heads, -1))
    distance_mass = (
        (histogram / float(n_anchors)).reshape(batch, heads, seq_len)
        .to(torch.float64).cpu().numpy()
    )

    steps = np.arange(seq_len, dtype=np.float64)
    # A ceiling of zero means every scored anchor admitted exactly one key, which is a segment
    # with nothing to attend over: the ratio is undefined there rather than infinite.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(ceiling > 0.0, entropy_mean / ceiling, np.nan)
    return BlockSampleStats(
        entropy_nats=entropy_mean,
        ceiling_nats=ceiling_nats,
        entropy_ratio=ratio,
        distance_mass=distance_mass,
        mean_distance=distance_mass @ steps,
        n_anchors=int(n_anchors),
    )


class BlockAccumulator:
    r"""What one $(\text{cohort}, \text{stream}, \text{block})$ cell keeps across the whole pass.

    Fixed size: one running sum per head for the entropy, its ceiling and their ratio, and one
    length-$T$ histogram per head. Peak memory is therefore independent of how many segments the
    pass sees, which is the property that lets the cap bound *time* rather than memory.
    """

    def __init__(self, n_heads: int, n_distances: int) -> None:
        """Start every accumulator at zero.

        Args:
            n_heads: Encoder attention heads $H_e$.
            n_distances: Distance bins, which is the sequence length $T$.
        """
        self.n_heads = int(n_heads)
        self.n_distances = int(n_distances)
        self.n_segments = 0
        self.n_anchors = 0
        self.entropy_sum = np.zeros(self.n_heads, dtype=np.float64)
        self.ratio_sum = np.zeros(self.n_heads, dtype=np.float64)
        self.ceiling_sum = 0.0
        self.distance_sum = np.zeros((self.n_heads, self.n_distances), dtype=np.float64)

    def update(self, stats: BlockSampleStats, rows: Sequence[int]) -> None:
        """Add the named samples of one batch.

        Args:
            stats: That batch's per-sample statistics.
            rows: Positions within the batch to add -- every sample for the pooled cohort, one
                class's samples for a class cohort.
        """
        selected = list(int(row) for row in rows)
        if not selected:
            return
        self.entropy_sum += stats.entropy_nats[selected].sum(axis=0)
        self.ratio_sum += stats.entropy_ratio[selected].sum(axis=0)
        self.ceiling_sum += float(stats.ceiling_nats[selected].sum())
        self.distance_sum += stats.distance_mass[selected].sum(axis=0)
        self.n_segments += len(selected)
        self.n_anchors = int(stats.n_anchors)

    @property
    def entropy_mean(self) -> np.ndarray:
        """Per-head mean entropy in nats, ``NaN`` per head when nothing was added."""
        return self._mean(self.entropy_sum)

    @property
    def ratio_mean(self) -> np.ndarray:
        """Per-head mean entropy ratio, ``NaN`` per head when nothing was added."""
        return self._mean(self.ratio_sum)

    @property
    def ceiling_mean(self) -> float:
        """Mean attainable ceiling in nats, ``NaN`` when nothing was added."""
        return self.ceiling_sum / self.n_segments if self.n_segments else float("nan")

    @property
    def distance_profile(self) -> np.ndarray:
        """Per-head mass by distance, each row summing to $1$; all-``NaN`` when empty."""
        if not self.n_segments:
            return np.full((self.n_heads, self.n_distances), np.nan, dtype=np.float64)
        return self.distance_sum / float(self.n_segments)

    def _mean(self, total: np.ndarray) -> np.ndarray:
        """Divide a running sum by the segment count, or return NaNs when there is none."""
        if not self.n_segments:
            return np.full(self.n_heads, np.nan, dtype=np.float64)
        return total / float(self.n_segments)


def mass_quantile(profile: np.ndarray, quantile: float) -> float:
    r"""The mass-weighted quantile of $t - j$, in steps.

    The profile is a discrete distribution over integer distances, so the quantile is the first bin
    whose cumulative mass reaches ``quantile`` rather than an interpolated position between two
    bins: a distance of $3.5$ steps is not a thing an attention row can have.

    Args:
        profile: Mass per distance bin, in bin order. Need not be normalised.
        quantile: In $[0, 1]$.

    Returns:
        The distance in steps, or ``NaN`` when the profile carries no positive finite mass.
    """
    values = np.asarray(profile, dtype=np.float64)
    finite = np.where(np.isfinite(values), values, 0.0)
    total = float(finite.sum())
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    cumulative = np.cumsum(finite) / total
    return float(min(int(np.searchsorted(cumulative, float(quantile), side="left")), values.size - 1))


# =============================================================================
# What the encoders are, structurally
# =============================================================================
def stream_geometry(model: Any, stream: str) -> Dict[str, Any]:
    r"""Return the structural facts one encoder's measured reach has to be read against.

    Args:
        model: The rebuilt net.
        stream: ``'target'`` or ``'source'``.

    Returns:
        The stem reach $R_{\mathrm{conv}}$, the block count, the window, and the structural bound
        $R_s = \min(R_{\mathrm{conv}} + N_s (W_s - 1),\ T)$ -- ``None`` for a full-prefix encoder,
        which is a bound that is *absent* rather than one that happens to equal $T$.
    """
    encoder = stream_encoder(model, stream)
    bound = encoder.receptive_field
    window = encoder.attention_window
    return {
        "conv_reach_steps": int(encoder.conv_reach),
        "n_attention_blocks": int(encoder.num_attention_blocks),
        "attention_window": None if window is None else int(window),
        "structural_bound_steps": None if bound is None else int(bound),
        "structural_bound_seconds": None if bound is None else int(bound) * SECONDS_PER_STEP,
        "structural_bound_absent": bound is None,
    }


def composed_reach(
    conv_reach_steps: int, block_distances: Sequence[float], *, sequence_length: int
) -> float:
    r"""Compose per-block measured distances into the reach of the whole stack.

    $$\widehat R_s = \min\!\left(R_{\mathrm{conv}} + \sum_b \widehat d_b,\ T\right),$$

    the same arithmetic as the structural bound
    $R_s = \min(R_{\mathrm{conv}} + N_s (W_s - 1),\ T)$ with each block's *measured* hop
    $\widehat d_b$ in place of the largest hop it was allowed. A stack whose every block put its
    mass at the far edge of its window therefore reproduces the structural bound exactly, which is
    what makes the two comparable at all.

    **The cap is the same $\min(\cdot, T)$ the structural formula applies**, and it is not
    cosmetic: on the full-prefix target encoder the per-block hops routinely sum past the segment,
    and an uncapped figure would report a reach of hundreds of steps more history than the segment
    contains. A stack that saturates it has reached the start of the segment, which is the most any
    reach can mean here.

    This is an estimate of what the encoder **uses**, deliberately not a re-derivation of what it
    *could* reach: ``tests/test_source_window.py`` already measures the structural bound by
    perturbation, and a second estimate of that number would be a worse copy of it.

    Args:
        conv_reach_steps: The stem's own reach $R_{\mathrm{conv}}$, itself included.
        block_distances: One measured distance per attention block, in steps.
        sequence_length: The segment's own $T$, which no reach can exceed.

    Returns:
        The composed reach in steps, or ``NaN`` when any block measured nothing.
    """
    values = np.asarray(list(block_distances), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    return float(min(float(conv_reach_steps) + float(values.sum()), float(sequence_length)))


# =============================================================================
# Choosing the segments
# =============================================================================
def dataset_shard_keys(dataset: Any) -> Optional[List[str]]:
    """Return the source shard of every position in ``dataset``, or ``None`` when it cannot be read.

    The stratum a cap has to be drawn within, resolved before anything is loaded. A ``Subset`` is
    unwrapped rather than refused: the pass this analysis is handed a loader for is frequently one
    the run's own sample cap already restricted, and a draw that fell back to "no strata" there
    would silently lose the coverage guarantee exactly where the population is already smallest.

    Args:
        dataset: The loader's dataset, possibly a ``Subset``.

    Returns:
        One shard basename per position, or ``None`` when the dataset does not expose its file
        layout -- which a caller reads as "draw without strata", never as "draw a prefix".
    """
    from pathlib import Path

    from torch.utils.data import Subset

    positions: Optional[List[int]] = None
    base = dataset
    if isinstance(base, Subset):
        positions = [int(value) for value in base.indices]
        base = base.dataset
    index_map = getattr(base, "index_map", None)
    paths = getattr(base, "paths", None)
    if not index_map or not paths:
        return None
    try:
        keys = [Path(str(paths[int(file_index)])).name for file_index, _ in index_map]
    except (IndexError, TypeError, ValueError):
        return None
    if positions is None:
        return keys
    try:
        return [keys[position] for position in positions]
    except IndexError:
        return None


def stratified_segment_loader(
    loader: Any, *, cap: int, seed: int
) -> Tuple[Any, Dict[str, Any]]:
    """Restrict a loader to a seeded, shard-stratified draw of at most ``cap`` segments.

    Never a prefix, and never the loader's own first batches. The evaluation split is eight
    concatenated per-subgroup files, so a prefix is one subgroup and one clinical class -- and this
    analysis cuts every readout by clinical class, which would then have exactly one cohort to cut
    into. Stratifying by shard gives every file a share of the cap proportional to its size, so a
    cap at or above the shard count reaches every shard rather than merely being likely to.

    Args:
        loader: The evaluation dataloader, read for its dataset, batch size and collation.
        cap: How many segments to score.
        seed: Seed, so a re-run scores the same segments.

    Returns:
        ``(loader, record)``. The record says what the draw did, including that it took everything,
        so a table from this analysis never has to be read as though it described the whole split.
    """
    from teb_vae.lag_attn_rws.eval._reuse import subsample_indices
    from torch.utils.data import DataLoader, Subset

    dataset = loader.dataset
    n_total = int(len(dataset))
    keys = dataset_shard_keys(dataset)
    drawn = subsample_indices(n_total, int(cap), int(seed), groups=keys)
    order = list(range(n_total)) if drawn is None else [int(value) for value in drawn.tolist()]
    record = {
        "cap": int(cap),
        "applied": drawn is not None,
        "n_total": n_total,
        "n_drawn": len(order),
        "stratified_by": "source_file_basename" if keys else None,
        "n_shards_drawn": len({keys[index] for index in order}) if keys else None,
    }
    return (
        DataLoader(
            Subset(dataset, order),
            batch_size=int(loader.batch_size or 1),
            shuffle=False,
            sampler=None,
            num_workers=0,
            collate_fn=loader.collate_fn,
        ),
        record,
    )


# =============================================================================
# The pass
# =============================================================================
@dataclass(frozen=True)
class PassResult:
    """Everything one bounded pass produced.

    Attributes:
        accumulators: ``(cohort, stream, block)`` to its accumulator. The pooled cohort is
            :data:`POOLED_CLASS`; the rest are clinical class names.
        per_segment: One record per scored segment -- its identity, its cohort labels and its
            per-stream mean entropy ratio and mean distance -- which is what the per-recording
            reduction and every grouped variant are built from.
        heatmaps: ``(stream, block)`` to one segment's head-averaged $(T, T)$ map, for the figure.
            One segment, because the map is the only thing here that is not a running sum.
        geometry: Per-stream structural facts, from :func:`stream_geometry`.
        anchor_range: The half-open range of scored anchors.
        n_heads: Encoder attention heads.
        seq_len: Sequence length $T$.
        n_segments: How many segments were scored.
        n_batches: How many batches they arrived in.
    """

    accumulators: Dict[Tuple[str, str, int], BlockAccumulator]
    per_segment: List[Dict[str, Any]]
    heatmaps: Dict[Tuple[str, int], np.ndarray]
    geometry: Dict[str, Dict[str, Any]]
    anchor_range: Tuple[int, int]
    n_heads: int
    seq_len: int
    n_segments: int
    n_batches: int


@torch.no_grad()
def run_encoder_attention_pass(task: Any, loader: Any) -> PassResult:
    r"""Run the model over ``loader``, recompute both encoders' attention, and reduce it.

    The **full** forward is what runs, not a re-assembly of the encoder path. The gate, the delay
    and the input adapter sit between the batch and the attention blocks, and a second copy of that
    sequence would profile attention over a stream the trained model never saw. The hooks fire
    wherever the forward reaches them.

    One block's map exists at a time: the loop recomputes, reduces and drops before moving on, so
    peak memory is $(B, H_e, T, T)$ for the widest block rather than for the stack.

    Args:
        task: The loaded task, in evaluation mode.
        loader: A loader over the segments to profile -- already capped and drawn by the caller,
            because which segments to score is a configuration decision and this is the mechanism.

    Returns:
        The pass's accumulators, per-segment records and retained maps.

    Raises:
        AttributeError: If the model exposes no encoders, or no blocks under them.
        ValueError: If the trained anchor range is empty.
    """
    model = task.orig_model
    refs = attention_blocks(model)
    anchor_range = model.geometry.valid_anchor_range()
    anchors = (int(anchor_range.start), int(anchor_range.stop))
    geometry = {stream: stream_geometry(model, stream) for stream in STREAMS}

    accumulators: Dict[Tuple[str, str, int], BlockAccumulator] = {}
    heatmaps: Dict[Tuple[str, int], np.ndarray] = {}
    per_segment: List[Dict[str, Any]] = []
    n_segments = 0
    n_batches = 0
    n_heads = 0
    seq_len = 0

    with captured_block_inputs(refs) as captured:
        for batch in loader:
            moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            y_st, y_ph, u_stream, _fhr_raw, _weight = model_inputs(task, moved)
            captured.clear()
            model(y_st, y_ph, u_stream)

            batch_size = int(y_st.shape[0])
            cohorts = labels.batch_labels(moved, batch_size)
            classes = [
                None if value is None else str(value)
                for value in cohorts[labels.CLASS_COLUMN]
            ]
            rows_by_cohort: Dict[str, List[int]] = {POOLED_CLASS: list(range(batch_size))}
            for row, name in enumerate(classes):
                if name is not None:
                    rows_by_cohort.setdefault(name, []).append(row)

            # Per (stream, sample): the ratio and distance summed over this stream's blocks and
            # heads, divided at the end -- the per-recording frame's own columns.
            segment_ratio: Dict[str, np.ndarray] = {
                stream: np.zeros(batch_size, dtype=np.float64) for stream in STREAMS
            }
            segment_distance: Dict[str, np.ndarray] = {
                stream: np.zeros(batch_size, dtype=np.float64) for stream in STREAMS
            }
            blocks_per_stream: Dict[str, int] = {stream: 0 for stream in STREAMS}

            for ref in refs:
                captured_input = captured.get(ref.key)
                if captured_input is None:
                    raise RuntimeError(
                        f"the forward never reached {ref.stream} attention block {ref.index}, so "
                        f"its input was not captured; the hook and the module have come apart"
                    )
                probs = attention_probabilities(ref.module, captured_input)
                seq_len = int(probs.shape[-1])
                n_heads = int(probs.shape[1])
                counts = admitted_keys(
                    ref.module, seq_len, device=probs.device
                ).sum(dim=-1)
                stats = block_sample_stats(probs, anchors=anchors, admitted_counts=counts)

                for cohort, rows in rows_by_cohort.items():
                    key = (cohort, ref.stream, ref.index)
                    if key not in accumulators:
                        accumulators[key] = BlockAccumulator(n_heads, seq_len)
                    accumulators[key].update(stats, rows)

                segment_ratio[ref.stream] += np.nanmean(stats.entropy_ratio, axis=1)
                segment_distance[ref.stream] += stats.mean_distance.mean(axis=1)
                blocks_per_stream[ref.stream] += 1

                # The first batch's first segment, head-averaged: the heatmap is a picture of one
                # recording rather than of the split, and retaining more would be the $(T, T)$
                # retention this analysis exists without.
                if ref.key not in heatmaps:
                    heatmaps[ref.key] = (
                        probs[0].mean(dim=0).to(torch.float64).cpu().numpy()
                    )
                del probs

            # The shared reader, so an unlabelled segment lands in the same ``'unknown'`` bucket
            # here as it does in every other per-recording reduction in the pipeline.
            guids = batch_guids(moved, batch_size)
            for row in range(batch_size):
                record: Dict[str, Any] = {
                    "guid": guids[row],
                    labels.CLASS_COLUMN: classes[row],
                    labels.SUBGROUP_COLUMN: cohorts[labels.SUBGROUP_COLUMN][row],
                }
                for stream in STREAMS:
                    count = max(blocks_per_stream[stream], 1)
                    record[f"encoder_entropy_ratio_{stream}"] = float(
                        segment_ratio[stream][row] / count
                    )
                    record[f"encoder_attention_distance_{stream}_steps"] = float(
                        segment_distance[stream][row] / count
                    )
                per_segment.append(record)

            n_segments += batch_size
            n_batches += 1

    return PassResult(
        accumulators=accumulators,
        per_segment=per_segment,
        heatmaps=heatmaps,
        geometry=geometry,
        anchor_range=anchors,
        n_heads=n_heads,
        seq_len=seq_len,
        n_segments=n_segments,
        n_batches=n_batches,
    )


