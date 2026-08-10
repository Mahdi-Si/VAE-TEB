r"""The evaluation-only oracle decoder: what a forecast costs when the bottleneck is removed.

``pred_gap`` is a difference between two *models*. Reading it as an information rate needs one
more number that no readout in this pipeline has carried until now: how much of the target's own
predictive content the latent throws away. Without it, a small gap is equally consistent with
"the source adds little" and with "the bottleneck discards so much that neither branch could have
used it".

So this module fits an **oracle**: the same decoder, at the same capacity, reading the target
encoder's own state $h^y_t$ instead of the $d_z$-wide latent $z_t$. It forecasts the identical
$H \cdot R = 480$-sample block, against the identical target and the identical mask, under the
identical likelihood. The only thing that differs is what it is conditioned on, so

$$\Delta_{\mathrm{suff}} = D_{\mathrm{base}} - D_{\mathrm{oracle}}$$

is the cost of the bottleneck and nothing else -- which is exactly the quantity the standing
limitation says is unmeasured.

**It is an estimate, not a bound, and both bias directions are carried in the output.**
Conditioning on ``target_state`` rather than on the raw target history omits the *encoder's* own
information loss, which biases the gap **down**; fitting the probe on the evaluation population
while $D_{\mathrm{base}}$ comes from a model trained on the disjoint, healthier pretraining cohort
biases it **up**, by a domain shift the probe does not suffer. The two oppose and neither is
measured, so no arithmetic here turns the number into a one-sided bound.

**The fit is held out at the level of the recording.** Fitting and scoring on the same recordings
makes the oracle look better than it is and the gap larger than it is, and a GUID contributes up
to ~37 overlapping segments, so a segment-level split is not a split at all. The GUIDs are
partitioned once, from the run's own seed, and the membership counts travel into the summary.

**The encoder states are cached, and that is an explicit amendment to the one-pass rule.** The
rest of the pipeline holds that the shared collection pass is the only model-touching cost. A
probe fit is thousands of passes over the same segments, so re-running the encoder for each would
be the dominant cost of the whole evaluation; instead one encoder pass writes
:class:`StateCache` -- roughly $153$ KiB of ``target_state`` per segment in fp32, plus the raw FHR
and its validity at about a tenth of that -- and every fit step reads it. The extra pass is
recorded in the analysis's ``plan`` rather than left as a surprise in a profile.

**The forecast target is rebuilt, never cached.** ``fhr_raw`` and ``weight`` are $21$ KiB per
segment against the $518$ KiB of the expanded $(T_{\mathrm{valid}}, H, R)$ block they generate, and
they are pushed through the *model's own* :func:`build_future_target` and :func:`forecast_mask` at
the model's own ``coverage_floor`` -- so the oracle is scored on the same anchors, against the same
samples, as the branch it is compared with.

**Nothing here can touch the production model.** The cached states are detached on the way out of
the forward, so the probe's optimizer step has no path back into the checkpoint's parameters; the
probe is a separate module with its own optimizer, and the global RNG is forked around the fit so
an analysis that runs afterwards sees the stream it would have seen anyway.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Subset

from teb_vae.lag_attn.nets.decoders import BaselineFutureDecoder, HorizonDecoderCore
from teb_vae.lag_attn_rws.eval._reuse import subsample_indices
from teb_vae.lag_attn_rws.eval.metrics import batch_field, batch_guids, model_inputs
from teb_vae.lag_attn_rws.nets.controls import NoCrossGroupPartner, make_derangement
from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

#: ``eval_config.caps`` name bounding how many segments are cached.
#:
#: Its default differs from every other cap in this pipeline and the difference is deliberate:
#: the retention caps in the collection pass are **opt-in**, because a missing entry there means
#: "keep nothing", which costs a figure. A missing entry here would mean "fit the probe on
#: nothing", which is not a cheaper measurement but no measurement at all. So absence means
#: *every* segment, and the cap exists only to bound the cache on a split large enough for
#: $159$ KiB per segment to matter.
CACHE_CAP_NAME = "oracle"

#: The fit budget, **in passes over the fit half** rather than in optimizer steps.
#:
#: A step count is not portable across populations: the same number that under-trains a probe on
#: two thousand segments overfits one on twenty, and every fixture in this repository is the second
#: case. Expressed in passes, one constant describes both -- the step count is derived in
#: :func:`resolve_budget`.
#:
#: A starting budget rather than a measured optimum, in the same sense as ``num_mc_samples``. What
#: makes a fixed budget honest is that the run reports its own held-out curve and raises
#: :data:`FitResult.converged` ``= False`` when that curve is still descending at the end -- so an
#: under-trained probe understates $\Delta_{\mathrm{suff}}$ *and says so*, rather than reporting a
#: small gap as a finding about the model.
DEFAULT_FIT_EPOCHS = 20
DEFAULT_FIT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3

#: Bounds on the derived step count. The floor keeps a tiny population from producing a curve with
#: no shape at all; the ceiling bounds a production run, where the probe is the one analysis whose
#: cost is a training loop rather than a forward pass.
MIN_FIT_STEPS = 4
MAX_FIT_STEPS = 3000

#: Points on the held-out curve. The convergence rule reads the last quarter of it, so this is
#: also the resolution that rule has.
DEFAULT_CURVE_POINTS = 12

#: Segments each curve point is measured over, per side.
#:
#: The curve is sampled rather than measured over everything, and the reason is arithmetic: at a
#: dozen evaluation points a full-population curve costs a sizeable fraction of the fit it is
#: describing. The sample is seeded and **fixed across every point of a run**, so the curve is a
#: sequence of comparable measurements rather than a walk over different segments; the reported
#: $D_{\mathrm{oracle}}$ is measured over the whole held-out half, once, at the end.
CURVE_SAMPLE_SEGMENTS = 64

#: Fraction of recordings held out of the fit. A half rather than a smaller slice: both sides need
#: enough recordings for the per-recording chain downstream to have a denominator, and the probe
#: is being asked whether the *bottleneck* costs anything, not to be the best possible forecaster.
HELD_OUT_FRACTION = 0.5

#: The capacity-adequacy check: refit at this width multiple and compare held-out scores.
CAPACITY_WIDTH_MULTIPLIER = 2

#: How much a doubled-width probe must improve held-out $D_{\mathrm{oracle}}$, in nats per anchor,
#: before the narrow probe is declared capacity-bound.
#:
#: One nat, matching ``prior_shuffle_min_nats``, and for the same reason: a capacity gap smaller
#: than the margin the coupling readout itself is judged against cannot change how the sufficiency
#: number is read, so calling it a capacity limitation would be reporting fit noise.
CAPACITY_MARGIN_NATS = 1.0

#: How much of the held-out curve's total improvement may still arrive in its final quarter before
#: the probe declares itself under-trained. A tenth: a probe that gains a tenth of everything it
#: ever gained in its last quarter is plainly still descending.
CONVERGENCE_TAIL_FRACTION = 0.1

#: The smallest population the split is attempted on. Fewer recordings than this cannot be halved
#: into two sides that both mean anything, and a probe fitted on one recording measures that
#: recording.
MIN_RECORDINGS = 4

#: The fewest segments each side of the split must carry.
MIN_SEGMENTS_PER_SIDE = 2

#: Offsets applied to ``eval_config.seed``, so the cache draw, the recording split, the probe's
#: initialisation and its batch draw are four independent streams derived from one value. Distinct
#: and non-zero for the same reason the runner's three are.
_SEED_OFFSET_CACHE, _SEED_OFFSET_SPLIT, _SEED_OFFSET_PROBE, _SEED_OFFSET_BATCH = 11, 12, 13, 14

#: The two directions the sufficiency estimate is biased in, neither of them measured. Carried in
#: the emitted record rather than only in this docstring: a reader meets the number in
#: ``summary.json``, and a caveat that lives only in a module nobody opens is not a caveat.
BIAS_DIRECTIONS: Tuple[Dict[str, str], ...] = (
    {
        "direction": "understates",
        "cause": (
            "the probe conditions on target_state rather than on the raw target history, so it "
            "inherits the encoder's own information loss and cannot recover what the encoder "
            "already discarded -- the measured gap is therefore the bottleneck's cost alone, not "
            "the whole context-sufficiency gap"
        ),
    },
    {
        "direction": "overstates",
        "cause": (
            "the probe is fitted on the evaluation population while D_base comes from a model "
            "trained on the disjoint, healthier pretraining cohort, so the probe does not suffer "
            "the domain shift the production decoder does and part of the gap is that shift "
            "rather than the bottleneck"
        ),
    },
)

#: The sentence the run carries beside the number, because the two biases oppose and neither is
#: measured. Fitting a second probe on a held-out split of the *training* cohort would remove the
#: second bias and is the stated path to a real one-sided bound.
ESTIMATE_NOT_A_BOUND = (
    "Delta_suff is an estimate, not a bound: one bias direction understates it and the other "
    "overstates it, neither is measured, and no arithmetic here resolves them."
)


# =============================================================================
# The probe
# =============================================================================
def build_probe(model: Any, *, width_multiplier: int = 1) -> nn.Module:
    r"""Build a decoder that mirrors the production one and reads ``target_state`` instead of $z$.

    No new decoder class: ``BaselineFutureDecoder`` already takes one conditioning tensor and one
    shared horizon core and emits $(B, T, H, R)$, so the oracle is that class built at the encoder
    state's width. Everything that decides *capacity* -- the hidden width, the core's depth, its
    kernel, its FiLM arrangement, its horizon-attention depth, the log-variance clamp -- is read off
    the model that is being measured rather than restated, which is what makes "the comparison
    isolates the bottleneck" a property rather than a claim.

    The core is a **fresh** one. Sharing the production core would give the probe the trained
    horizon dynamics for free and make $D_{\mathrm{oracle}}$ a statement about the projection layer
    alone.

    Two initialisation policies are mirrored, and one is applied unconditionally:

    * the horizon-step embedding is re-seeded at the model's own ``horizon_embed_std``, because the
      $H$ tokens are otherwise near-degenerate and per-block FiLM has nothing token-specific to
      modulate;
    * the output heads are calibrated onto the trivial $\mu = 0, \sigma = 1$ predictor whatever the
      model did, because the probe has a fixed step budget and an uncalibrated head spends a large
      part of it walking back from a random initial log-density. That choice makes the oracle
      *stronger*, hence the gap larger, and is named here for that reason.

    Args:
        model: The loaded :class:`~teb_vae.lag_attn_rws.nets.model.SeqVaeLagAttnRws`.
        width_multiplier: Multiple of the production hidden width to build at. ``1`` is the
            comparison probe; :data:`CAPACITY_WIDTH_MULTIPLIER` is the capacity-adequacy refit.

    Returns:
        The probe, on CPU and in training mode, holding no reference to the model's parameters.
    """
    reference = model.horizon_core
    d_hidden = int(reference.d_hidden) * int(width_multiplier)
    # Read from the built module rather than from a config: the core stores neither its kernel nor
    # its depth as an attribute, and a second copy of either could disagree with the model loaded.
    depth = len(reference.refine.blocks)
    kernel_size = int(reference.refine.blocks[0]["conv"].kernel_size[0])

    core = HorizonDecoderCore(
        d_hidden=d_hidden,
        horizon=int(reference.horizon),
        kernel_size=kernel_size,
        depth=depth,
        film=bool(reference.film),
        film_per_block=bool(reference.film_per_block),
        # The horizon attention is capacity like everything else here, so it is mirrored rather
        # than restated: a probe built without the blocks the loaded model has would answer a
        # question about a decoder nobody trained.
        attention_blocks=int(reference.attention_blocks),
        attention_heads=int(reference.attention_heads),
    )
    probe = BaselineFutureDecoder(
        core=core,
        d_model=int(model.d_model),
        out_channels=int(model.decoder.out_channels),
        d_hidden=d_hidden,
        # Zero, like the production decoder's: two invocations of one module draw two dropout
        # masks, and every score here is a difference between invocations.
        dropout=0.0,
        logvar_clamp=(float(model.logvar_clamp[0]), float(model.logvar_clamp[1])),
    )

    embed_std = float(getattr(model, "horizon_embed_std", 0.02))
    if embed_std != 0.02:
        nn.init.normal_(core.horizon_embedding, mean=0.0, std=embed_std)
    with torch.no_grad():
        probe.mean_head.weight.mul_(0.02)
        probe.logvar_head.bias.fill_(math.log(5.0 / 3.0))
        probe.logvar_head.weight.mul_(0.1)
    return probe


def parameter_count(module: nn.Module) -> int:
    """Return a module's trainable parameter count.

    Args:
        module: Any module.

    Returns:
        The number of elements across its trainable parameters.
    """
    return int(sum(int(p.numel()) for p in module.parameters() if p.requires_grad))


# =============================================================================
# The cache
# =============================================================================
@dataclass
class StateCache:
    """One encoder pass, kept so a probe fit does not become thousands of them.

    Attributes:
        target_state: Detached encoder states $(N, T_{\\mathrm{valid}}, d_{\\mathrm{model}})$ on
            CPU in fp32. Detached on the way out of the forward, which is what makes the probe's
            optimizer step structurally unable to reach the checkpoint.
        fhr_raw: The loader-normalized raw target $(N, L_{\\mathrm{raw}})$. The expanded forecast
            block is rebuilt from it per batch rather than cached: it is a twenty-fifth of the
            size and produces the block through the model's own index grid.
        weight: The decimated validity signal $(N, T)$, from which the forecast mask is rebuilt.
        guid: One recording identifier per segment, which is the unit the split is drawn on.
        epoch: One ``epoch`` per segment, so a scored segment can be joined back onto the
            per-sample table.
    """

    target_state: torch.Tensor
    fhr_raw: torch.Tensor
    weight: torch.Tensor
    guid: List[str]
    epoch: np.ndarray

    def __len__(self) -> int:
        return int(self.target_state.shape[0])

    @property
    def n_bytes(self) -> int:
        """Bytes held by the three tensors, so the amendment to the one-pass rule is a number."""
        return int(
            self.target_state.numel() * self.target_state.element_size()
            + self.fhr_raw.numel() * self.fhr_raw.element_size()
            + self.weight.numel() * self.weight.element_size()
        )

    def describe(self) -> Dict[str, Any]:
        """Return the cache as a record for the summary."""
        return {
            "n_segments": len(self),
            "n_recordings": int(len(set(self.guid))),
            "n_bytes": self.n_bytes,
            "state_shape": [int(value) for value in self.target_state.shape[1:]],
        }


@torch.no_grad()
def cache_target_states(
    task: Any, loader: Any, *, cap: Optional[int] = None, seed: int = 0
) -> StateCache:
    r"""Run the encoder once over the split and keep what the probe needs.

    The states come out of the **model's own forward** rather than out of a re-assembly of its
    first three lines. Re-assembling them would be cheaper -- the forward also decodes both
    branches -- but the target stream reaches the encoder through a channel gate, a delay and an
    adapter, and a state built by a second copy of that sequence is a state the production decoder
    never saw. The cost is one pass at roughly an eighth of the collection pass's, since that one
    decodes four branches at $K$ draws and this one decodes two at one.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader, read for its dataset, its batch size and its collation.
        cap: How many segments to cache, or ``None`` for all of them. Drawn over the **whole**
            index space, never as a prefix: the dataset is eight concatenated per-subgroup files,
            so a prefix is one subgroup and one clinical class.
        seed: Seed for that draw.

    Returns:
        The cache, in dataset order.
    """
    model = task.orig_model
    t_valid = int(model.geometry.t_valid)

    dataset = loader.dataset
    n_total = int(len(dataset))
    drawn = subsample_indices(n_total, cap, int(seed))
    order = list(range(n_total)) if drawn is None else [int(value) for value in drawn.tolist()]

    pages = DataLoader(
        Subset(dataset, order),
        batch_size=int(loader.batch_size or 1),
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=loader.collate_fn,
    )

    states: List[torch.Tensor] = []
    raws: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []
    guids: List[str] = []
    epochs: List[float] = []
    for batch in pages:
        moved = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
        y_st, y_ph, u_stream, fhr_raw, weight = model_inputs(task, moved)
        outputs = model(y_st, y_ph, u_stream)
        batch_size = int(y_st.shape[0])
        # Detached and on CPU: the probe reads a tensor with no graph behind it, so no optimizer
        # step it takes can reach a production parameter.
        states.append(outputs["target_state"][:, :t_valid].detach().to("cpu", torch.float32))
        raws.append(fhr_raw.detach().to("cpu", torch.float32))
        weights.append(weight.detach().to("cpu", torch.float32))
        guids.extend(batch_guids(moved, batch_size))
        epochs.extend(_epoch_values(batch_field(moved, "epoch"), batch_size))

    if not states:
        return StateCache(
            target_state=torch.zeros((0, t_valid, int(model.d_model))),
            fhr_raw=torch.zeros((0, 0)),
            weight=torch.zeros((0, 0)),
            guid=[],
            epoch=np.zeros((0,), dtype=np.float64),
        )
    cache = StateCache(
        target_state=torch.cat(states, dim=0),
        fhr_raw=torch.cat(raws, dim=0),
        weight=torch.cat(weights, dim=0),
        guid=guids,
        epoch=np.asarray(epochs, dtype=np.float64),
    )
    logger.info(
        f"oracle: cached {len(cache)} segment(s) from {len(set(cache.guid))} recording(s), "
        f"{cache.n_bytes / (1024 ** 2):.1f} MiB of encoder state"
    )
    return cache


def _epoch_values(field: Any, batch_size: int) -> List[float]:
    """Return one ``epoch`` per sample, NaN where the batch does not carry one.

    Args:
        field: The batch's ``epoch`` field, or ``None``.
        batch_size: Expected length.

    Returns:
        A list of floats of length ``batch_size``.
    """
    if field is None:
        return [float("nan")] * batch_size
    values = np.asarray(
        field.detach().cpu().numpy() if isinstance(field, torch.Tensor) else field,
        dtype=np.float64,
    ).reshape(-1)
    if values.size < batch_size:
        return list(values) + [float("nan")] * (batch_size - values.size)
    return [float(value) for value in values[:batch_size]]


# =============================================================================
# The held-out split
# =============================================================================
def guid_split(
    guids: Sequence[str], *, seed: int, held_out_fraction: float = HELD_OUT_FRACTION
) -> Tuple[np.ndarray, np.ndarray]:
    """Partition segment positions into a fit half and a held-out half, **by recording**.

    At the level of the recording rather than the segment, and that is the whole point: one
    delivery contributes tens of segments whose forecast windows overlap in 29 of their 30 steps,
    so a segment-level split leaves the probe scoring what it was fitted on under another name.

    Args:
        guids: One recording identifier per segment, in cache order.
        seed: Seed for the shuffle, so the split is reproducible from the summary.
        held_out_fraction: Share of the *recordings* withheld.

    Returns:
        ``(fit positions, held-out positions)``, both ascending and disjoint by construction --
        every position of a recording lands on the same side.
    """
    unique = sorted(set(str(value) for value in guids))
    if not unique:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    generator = torch.Generator().manual_seed(int(seed))
    shuffled = [unique[int(index)] for index in torch.randperm(len(unique), generator=generator)]
    n_held_out = int(round(len(unique) * float(held_out_fraction)))
    # Both sides non-empty whenever there are at least two recordings: a split that put every
    # recording on one side would report a held-out score fitted on itself.
    n_held_out = max(1, min(len(unique) - 1, n_held_out)) if len(unique) > 1 else 0
    held_out_guids = set(shuffled[:n_held_out])

    positions = np.arange(len(guids), dtype=np.int64)
    is_held_out = np.asarray([str(value) in held_out_guids for value in guids], dtype=bool)
    return positions[~is_held_out], positions[is_held_out]


# =============================================================================
# Scoring
# =============================================================================
def _batch_tensors(
    cache: StateCache,
    target_rows: torch.Tensor,
    state_rows: torch.Tensor,
    *,
    geometry: Any,
    future_index: torch.Tensor,
    coverage_floor: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Assemble one batch of ``(states, target, mask)`` from the cache.

    ``state_rows`` is separate from ``target_rows`` so the conditioning ablation can pair a
    segment's forecast target with **another recording's** encoder state through the same code
    path -- a control built out of a second, similar-looking loop is a control that can differ
    from the thing it controls.

    Args:
        cache: The cached pass.
        target_rows: Cache positions supplying the forecast target and its mask.
        state_rows: Cache positions supplying the conditioning state; equal to ``target_rows``
            unless the conditioning is deranged.
        geometry: The model's trimmed-grid geometry.
        future_index: The model's own $(T_{\mathrm{valid}}, H, R)$ raw index grid, on ``device``.
        coverage_floor: The model's own minimum valid fraction for an anchor to be scored.
        device: Where the batch is assembled.

    Returns:
        ``(states, target, mask)``.
    """
    states = cache.target_state[state_rows].to(device=device, dtype=torch.float32)
    fhr_raw = cache.fhr_raw[target_rows].to(device=device, dtype=torch.float32)
    weight = cache.weight[target_rows].to(device=device, dtype=torch.float32)
    target = build_future_target(fhr_raw, geometry, future_index=future_index)
    mask, _coverage = forecast_mask(weight, geometry, coverage_floor=float(coverage_floor))
    return states, target, mask


@torch.no_grad()
def score_rows(
    probe: nn.Module,
    cache: StateCache,
    rows: np.ndarray,
    *,
    geometry: Any,
    future_index: torch.Tensor,
    coverage_floor: float,
    likelihood: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Score the probe's forecast on the given cache positions, one figure per segment.

    The reduction is the pipeline's: an anchor's block score is summed over its $H \cdot R$ raw
    samples, then averaged over the segment's *contributing* anchors, so the result is in nats per
    anchor and directly comparable with ``nll_base_block``. A segment with no contributing anchor
    measured nothing and reads ``NaN``, never ``0.0``.

    Args:
        probe: The fitted probe.
        cache: The cached pass.
        rows: Cache positions to score.
        geometry: The model's trimmed-grid geometry.
        future_index: The model's raw index grid, on ``device``.
        coverage_floor: The model's minimum valid fraction per anchor.
        likelihood: ``'mse'`` or ``'gaussian_nll'``, the objective the checkpoint trained under.
        device: Where the scoring runs.
        batch_size: Segments per forward.

    Returns:
        ``(nats per anchor, contributing anchors)``, both of length ``len(rows)``.
    """
    probe.eval()
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    anchors = np.zeros(len(rows), dtype=np.float64)
    if not len(rows):
        return scores, anchors

    positions = torch.as_tensor(np.asarray(rows, dtype=np.int64), dtype=torch.long)
    for start in range(0, len(positions), int(batch_size)):
        chunk = positions[start : start + int(batch_size)]
        states, target, mask = _batch_tensors(
            cache, chunk, chunk,
            geometry=geometry, future_index=future_index,
            coverage_floor=coverage_floor, device=device,
        )
        mu, logvar = probe(states)
        block, contributing = masked_raw_block_per_anchor(
            mu, target, mask, likelihood=likelihood, logvar=logvar
        )
        counts = contributing.sum(dim=1).detach().cpu().to(torch.float64).numpy()
        totals = (block * contributing).sum(dim=1).detach().cpu().to(torch.float64).numpy()
        stop = start + len(chunk)
        anchors[start:stop] = counts
        scores[start:stop] = np.where(counts > 0.0, totals / np.maximum(counts, 1.0), np.nan)
    return scores, anchors


def _pooled(scores: np.ndarray, anchors: np.ndarray) -> float:
    """Return the anchor-weighted mean of per-segment scores, for the training curve.

    Anchor-weighted rather than per-recording: this is the quantity being *optimised*, and the
    curve should show the optimiser's own objective. The reported $D_{\\mathrm{oracle}}$ goes
    through the pipeline's per-recording chain instead, in the analysis that consumes this.

    Args:
        scores: Per-segment nats per anchor.
        anchors: Per-segment contributing-anchor counts.

    Returns:
        The pooled mean, or ``NaN`` when nothing was scored.
    """
    finite = np.isfinite(scores) & (anchors > 0.0)
    if not finite.any():
        return float("nan")
    return float(np.sum(scores[finite] * anchors[finite]) / np.sum(anchors[finite]))


# =============================================================================
# The fit
# =============================================================================
@dataclass
class FitResult:
    """One probe's fit, and everything needed to decide whether to believe it.

    Attributes:
        width_multiplier: The multiple of the production hidden width this probe was built at.
        n_parameters: Its trainable parameter count.
        steps: Optimizer steps run.
        curve: One record per evaluation point, carrying the step and both pooled scores. Both
            are measured over a *fixed subsample* of their side -- see
            :data:`CURVE_SAMPLE_SEGMENTS` -- so the curve is cheap and its points are comparable
            with each other, but not with the number below.
        final_held_out_nats: The final state's score over the **whole** held-out half, which is
            what $D_{\\mathrm{oracle}}$ is read from. Deliberately not the best point on the
            curve: selecting the step by held-out score would fit the held-out half through the
            back door.
        best_held_out_nats: The best point on the curve, and the step it happened at. Reported as
            a diagnostic -- a best far from the end is a probe that overfitted -- never as the
            answer.
        best_step: The step ``best_held_out_nats`` was reached at.
        converged: Whether the held-out curve had flattened by the end.
        convergence_detail: The arithmetic behind that flag, in words.
        shuffled_conditioning: Whether this fit was the conditioning control rather than the
            measurement.
    """

    width_multiplier: int
    n_parameters: int
    steps: int
    curve: List[Dict[str, float]]
    final_held_out_nats: float
    best_held_out_nats: float
    best_step: int
    converged: bool
    convergence_detail: str
    shuffled_conditioning: bool = False

    def describe(self) -> Dict[str, Any]:
        """Return the fit as a record for the summary."""
        return {
            "width_multiplier": int(self.width_multiplier),
            "n_parameters": int(self.n_parameters),
            "steps": int(self.steps),
            "final_held_out_nats": self.final_held_out_nats,
            "best_held_out_nats": self.best_held_out_nats,
            "best_step": int(self.best_step),
            "converged": bool(self.converged),
            "convergence_detail": self.convergence_detail,
            "shuffled_conditioning": bool(self.shuffled_conditioning),
            "curve": list(self.curve),
        }


def assess_convergence(curve: Sequence[Dict[str, float]]) -> Tuple[bool, str]:
    r"""Decide, mechanically, whether the held-out curve had stopped descending.

    Let $c_0$ be the first held-out score on the curve, $c_{3/4}$ the one nearest three quarters
    of the way through, and $c^\star$ the best. The probe counts as converged when

    $$c_{3/4} - c^\star \;\le\; \text{:data:`CONVERGENCE_TAIL_FRACTION`} \cdot (c_0 - c^\star),$$

    that is, when the final quarter contributed at most a tenth of everything the fit ever gained.
    A curve that gained nothing at all is **not** converged: a probe that never moved has not
    finished, it has failed to start, and reporting its score as an oracle would understate the
    gap by the whole of it.

    Args:
        curve: The evaluation points, in step order.

    Returns:
        ``(converged, detail)``; the detail states the three numbers so the flag can be checked.
    """
    values = [
        float(point["held_out_nats"])
        for point in curve
        if np.isfinite(float(point.get("held_out_nats", np.nan)))
    ]
    if len(values) < 3:
        return False, (
            f"the held-out curve carries {len(values)} finite point(s), too few to say whether it "
            f"had flattened"
        )
    first, best = values[0], min(values)
    tail = values[max(0, (3 * len(values)) // 4)]
    total_gain, tail_gain = first - best, tail - best
    if not (total_gain > 0.0):
        return False, (
            f"the held-out score never improved on its first evaluation "
            f"({first:.4g} nats/anchor); the probe did not fit"
        )
    converged = tail_gain <= CONVERGENCE_TAIL_FRACTION * total_gain
    return converged, (
        f"the final quarter of the fit contributed {tail_gain:.4g} of the {total_gain:.4g} "
        f"nats/anchor the held-out score improved by "
        f"({tail_gain / total_gain:.1%}, threshold {CONVERGENCE_TAIL_FRACTION:.0%})"
    )


def fit_probe(
    probe: nn.Module,
    cache: StateCache,
    fit_rows: np.ndarray,
    held_out_rows: np.ndarray,
    *,
    geometry: Any,
    future_index: torch.Tensor,
    coverage_floor: float,
    likelihood: str,
    device: torch.device,
    steps: int = MIN_FIT_STEPS,
    batch_size: int = DEFAULT_FIT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    eval_every: int = 1,
    seed: int = 0,
    width_multiplier: int = 1,
    shuffle_conditioning: bool = False,
) -> FitResult:
    r"""Fit the probe on the fit recordings, sampling the held-out curve as it goes.

    The objective is the checkpoint's own: the masked block score of
    :func:`~teb_vae.lag_attn_rws.nets.losses.masked_raw_block_per_anchor`, averaged over
    contributing anchors, so the probe is optimising exactly the quantity $D_{\mathrm{base}}$ is
    measured in.

    ``shuffle_conditioning`` is the control that separates "the probe learned to read the state"
    from "the probe learned the population mean": the fit pairs each segment's target with
    **another recording's** state, through the same batch assembly and the same loss, and is then
    scored on the held-out half with the states matched again. A probe that was reading the state
    loses; one that was not, does not.

    Args:
        probe: The probe, freshly built.
        cache: The cached pass.
        fit_rows: Cache positions the probe is fitted on.
        held_out_rows: Cache positions it is scored on.
        geometry: The model's trimmed-grid geometry.
        future_index: The model's raw index grid; moved to ``device`` by the caller.
        coverage_floor: The model's minimum valid fraction per anchor.
        likelihood: The objective the checkpoint trained under.
        device: Where the fit runs.
        steps: Optimizer steps.
        batch_size: Segments per step.
        learning_rate: Adam's learning rate.
        eval_every: Steps between held-out evaluations. The first and final steps are always
            evaluated, so the curve has endpoints whatever the interval.
        seed: Seed for the batch draw and, when asked for, the conditioning derangement.
        width_multiplier: Recorded on the result; the probe was already built at this width.
        shuffle_conditioning: Run the conditioning control instead of the measurement.

    Returns:
        The fit.
    """
    probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=float(learning_rate))
    generator = torch.Generator().manual_seed(int(seed))

    rows = torch.as_tensor(np.asarray(fit_rows, dtype=np.int64), dtype=torch.long)
    state_source = _conditioning_rows(rows, cache, generator) if shuffle_conditioning else rows

    curve: List[Dict[str, float]] = []
    steps = max(1, int(steps))
    eval_every = max(1, int(eval_every))
    # Fixed for the whole run, both sides: a curve whose points are measured over different
    # segments moves with the draw as much as with the fit.
    curve_held_out = _curve_sample(held_out_rows, seed=int(seed) + 1)
    curve_fit = _curve_sample(fit_rows, seed=int(seed) + 2)

    def _score(positions: np.ndarray) -> float:
        values, counts = score_rows(
            probe, cache, positions,
            geometry=geometry, future_index=future_index, coverage_floor=coverage_floor,
            likelihood=likelihood, device=device, batch_size=int(batch_size),
        )
        return _pooled(values, counts)

    def _evaluate(step: int) -> None:
        curve.append({
            "step": float(step),
            "held_out_nats": _score(curve_held_out),
            "fit_nats": _score(curve_fit),
        })
        probe.train()

    probe.train()
    _evaluate(0)
    with torch.enable_grad():
        for step in range(1, steps + 1):
            picks = torch.randint(len(rows), (int(batch_size),), generator=generator)
            states, target, mask = _batch_tensors(
                cache, rows[picks], state_source[picks],
                geometry=geometry, future_index=future_index,
                coverage_floor=coverage_floor, device=device,
            )
            mu, logvar = probe(states)
            block, contributing = masked_raw_block_per_anchor(
                mu, target, mask, likelihood=likelihood, logvar=logvar
            )
            loss = (block * contributing).sum() / contributing.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % eval_every == 0 or step == steps:
                _evaluate(step)
                logger.info(
                    f"oracle: step {step}/{steps} (width x{width_multiplier}), "
                    f"held-out {curve[-1]['held_out_nats']:.4g} nats/anchor"
                )

    held_out = [float(point["held_out_nats"]) for point in curve]
    finite = [value for value in held_out if np.isfinite(value)]
    best = min(finite) if finite else float("nan")
    best_step = (
        int(curve[held_out.index(best)]["step"]) if finite and best in held_out else -1
    )
    converged, detail = assess_convergence(curve)
    # The reported score is measured over the **whole** held-out half, once, at the final state --
    # not read off the curve, whose points are a fixed subsample, and not off the curve's best
    # point, which would be selecting the step by the held-out score it is about to report.
    final_scores, final_anchors = score_rows(
        probe, cache, np.asarray(held_out_rows, dtype=np.int64),
        geometry=geometry, future_index=future_index, coverage_floor=coverage_floor,
        likelihood=likelihood, device=device, batch_size=int(batch_size),
    )
    return FitResult(
        width_multiplier=int(width_multiplier),
        n_parameters=parameter_count(probe),
        steps=steps,
        curve=curve,
        final_held_out_nats=_pooled(final_scores, final_anchors),
        best_held_out_nats=best,
        best_step=best_step,
        converged=converged,
        convergence_detail=detail,
        shuffled_conditioning=bool(shuffle_conditioning),
    )


def _curve_sample(rows: np.ndarray, *, seed: int) -> np.ndarray:
    """Return the fixed subsample of ``rows`` every curve point is measured over.

    Args:
        rows: One side of the split.
        seed: Seed for the draw.

    Returns:
        At most :data:`CURVE_SAMPLE_SEGMENTS` positions, drawn over the whole set rather than as a
        prefix, ascending. The whole set when it is already small enough.
    """
    positions = np.asarray(rows, dtype=np.int64)
    drawn = subsample_indices(len(positions), CURVE_SAMPLE_SEGMENTS, int(seed))
    if drawn is None:
        return positions
    return positions[np.asarray(drawn.tolist(), dtype=np.int64)]


def _conditioning_rows(
    rows: torch.Tensor, cache: StateCache, generator: torch.Generator
) -> torch.Tensor:
    """Return the fit positions with their conditioning states deranged across recordings.

    Args:
        rows: The fit positions.
        cache: The cached pass, for the recording labels the derangement groups on.
        generator: The seeded generator.

    Returns:
        The same positions in an order that pairs no segment with its own recording's state --
        falling back to the ungrouped derangement, with a warning, when one recording holds more
        than half the fit half and no cross-recording pairing exists.
    """
    if len(rows) < 2:
        return rows
    groups = [cache.guid[int(index)] for index in rows.tolist()]
    try:
        permutation = make_derangement(len(rows), generator, groups=groups)
    except NoCrossGroupPartner:
        logger.warning(
            "oracle: the conditioning control could not pair every segment with another "
            "recording's state (one recording holds more than half the fit split); falling back "
            "to the ungrouped derangement, which is a weaker control."
        )
        permutation = make_derangement(len(rows), generator)
    return rows[permutation]


# =============================================================================
# The orchestration
# =============================================================================
def _fork_devices(device: torch.device) -> List[torch.device]:
    """Return the devices whose RNG state the fit must restore.

    The probe's initialisation and its batch draw run on the global generators, and an analysis
    that runs afterwards must see the stream it would have seen without this one -- otherwise
    ``--only samples`` and a full run would render different pages from the same checkpoint.

    Args:
        device: The device the fit runs on.

    Returns:
        ``[device]`` on CUDA, an empty list on CPU.
    """
    return [] if device.type != "cuda" else [device]


def resolve_budget(
    n_fit_segments: int,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    steps: Optional[int] = None,
    curve_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Turn the population size into the step count, the batch and the evaluation interval.

    Args:
        n_fit_segments: Segments the probe is fitted on.
        epochs: Passes over the fit half, or ``None`` for :data:`DEFAULT_FIT_EPOCHS`.
        batch_size: Segments per step, or ``None`` for :data:`DEFAULT_FIT_BATCH_SIZE`. Clamped to
            the fit set: a batch larger than the population draws the same segments repeatedly and
            pays for each one.
        steps: An explicit step count that overrides the derivation entirely.
        curve_points: Evaluation points, or ``None`` for :data:`DEFAULT_CURVE_POINTS`.

    Returns:
        ``steps``, ``batch_size``, ``learning_rate``, ``eval_every`` and the ``epochs`` the step
        count came from, ready to be recorded beside the number they produced.
    """
    resolved_epochs = DEFAULT_FIT_EPOCHS if epochs is None else int(epochs)
    resolved_batch = max(
        1,
        min(
            DEFAULT_FIT_BATCH_SIZE if batch_size is None else int(batch_size),
            max(1, int(n_fit_segments)),
        ),
    )
    if steps is None:
        derived = math.ceil(resolved_epochs * max(1, int(n_fit_segments)) / resolved_batch)
        resolved_steps = int(min(MAX_FIT_STEPS, max(MIN_FIT_STEPS, derived)))
    else:
        resolved_steps = max(1, int(steps))
    points = max(1, DEFAULT_CURVE_POINTS if curve_points is None else int(curve_points))
    return {
        "steps": resolved_steps,
        "batch_size": resolved_batch,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "eval_every": max(1, resolved_steps // points),
        "epochs": resolved_epochs,
    }


def run_oracle(
    task: Any,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    curve_points: Optional[int] = None,
    capacity_check: bool = True,
) -> Dict[str, Any]:
    r"""Cache the encoder states, split by recording, fit the probe, and score the held-out half.

    Args:
        task: The loaded task, in evaluation mode.
        loader: The evaluation dataloader.
        eval_config: The validated block, for the seed and ``caps.oracle``.
        epochs: Passes over the fit half, or ``None`` for :data:`DEFAULT_FIT_EPOCHS`.
        steps: An explicit optimizer-step count, overriding the derivation from ``epochs``.
        batch_size: Segments per step, or ``None`` for :data:`DEFAULT_FIT_BATCH_SIZE`.
        curve_points: Points on the held-out curve, or ``None`` for
            :data:`DEFAULT_CURVE_POINTS`.
        capacity_check: Refit at :data:`CAPACITY_WIDTH_MULTIPLIER` times the width and compare.
            A second fit at several times the first's per-step cost, which is why it is a
            parameter rather than a literal.

    Returns:
        The measurement, or ``{'skipped': True, 'reason': ...}`` when the population cannot carry
        one. On success: the cache record, the split, both fits, the capacity verdict, and the
        held-out segments' identities beside their scores, so a caller can join them onto the
        per-sample table rather than re-deriving them.
    """
    model = task.orig_model
    device = next(model.parameters()).device
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))
    seed = int(eval_config.get("seed", 0))
    cap = (eval_config.get("caps") or {}).get(CACHE_CAP_NAME)

    cache = cache_target_states(task, loader, cap=cap, seed=seed + _SEED_OFFSET_CACHE)
    n_recordings = len(set(cache.guid))
    if n_recordings < MIN_RECORDINGS:
        return _skip(
            f"the oracle needs at least {MIN_RECORDINGS} recordings to hold half of them out and "
            f"still have a denominator on each side; this pass cached {len(cache)} segment(s) "
            f"from {n_recordings}",
            cache=cache,
        )

    fit_rows, held_out_rows = guid_split(
        cache.guid, seed=seed + _SEED_OFFSET_SPLIT, held_out_fraction=HELD_OUT_FRACTION
    )
    if min(len(fit_rows), len(held_out_rows)) < MIN_SEGMENTS_PER_SIDE:
        return _skip(
            f"the recording split left {len(fit_rows)} fit and {len(held_out_rows)} held-out "
            f"segment(s), below the {MIN_SEGMENTS_PER_SIDE} each side needs",
            cache=cache,
        )

    fit_guids = {cache.guid[int(index)] for index in fit_rows}
    held_out_guids = {cache.guid[int(index)] for index in held_out_rows}
    # Asserted rather than assumed: the split is the only thing standing between this number and a
    # probe scored on what it was fitted to, and it is drawn by an ordinary function.
    overlap = sorted(fit_guids & held_out_guids)
    if overlap:
        raise ValueError(
            f"the oracle's recording split is not a split: {len(overlap)} recording(s) appear on "
            f"both sides ({overlap[:3]}...). A probe scored on recordings it was fitted to "
            f"reports the bottleneck as costing less than it does."
        )

    split = {
        "seed": seed + _SEED_OFFSET_SPLIT,
        "held_out_fraction": float(HELD_OUT_FRACTION),
        "n_fit_recordings": len(fit_guids),
        "n_held_out_recordings": len(held_out_guids),
        "n_fit_segments": int(len(fit_rows)),
        "n_held_out_segments": int(len(held_out_rows)),
        "recordings_disjoint": True,
    }
    budget = resolve_budget(
        len(fit_rows), epochs=epochs, batch_size=batch_size, steps=steps,
        curve_points=curve_points,
    )
    # ``epochs`` is recorded but is not a ``fit_probe`` argument -- it is what the step count was
    # derived from, which a reader of the summary needs and the fit itself does not.
    resolved = {name: value for name, value in budget.items() if name != "epochs"}
    shared = dict(
        geometry=model.geometry,
        future_index=model.future_index.to(device),
        coverage_floor=float(model.coverage_floor),
        likelihood=likelihood,
        device=device,
    )

    # Forked, so the probe's initialisation and batch draw cannot move the global stream the
    # analyses after this one draw from.
    with torch.random.fork_rng(devices=_fork_devices(device)):
        torch.manual_seed(seed + _SEED_OFFSET_PROBE)
        probe = build_probe(model, width_multiplier=1)
        fit = fit_probe(
            probe, cache, fit_rows, held_out_rows,
            seed=seed + _SEED_OFFSET_BATCH, width_multiplier=1, **shared, **resolved,
        )
        scores, anchors = score_rows(
            probe, cache, held_out_rows, batch_size=resolved["batch_size"], **shared
        )

        wide_fit: Optional[FitResult] = None
        if capacity_check:
            # The same initialisation stream as the narrow probe, so the two differ in width and
            # nothing else a seed could account for.
            torch.manual_seed(seed + _SEED_OFFSET_PROBE)
            wide = build_probe(model, width_multiplier=CAPACITY_WIDTH_MULTIPLIER)
            wide_fit = fit_probe(
                wide, cache, fit_rows, held_out_rows,
                seed=seed + _SEED_OFFSET_BATCH,
                width_multiplier=CAPACITY_WIDTH_MULTIPLIER, **shared, **resolved,
            )

    return {
        "skipped": False,
        "cache": {**cache.describe(), "cap": cap, "cap_name": CACHE_CAP_NAME},
        "split": split,
        "fit": fit.describe(),
        "capacity": capacity_verdict(fit, wide_fit),
        "settings": budget,
        "likelihood": likelihood,
        "bias_directions": [dict(entry) for entry in BIAS_DIRECTIONS],
        "estimate_not_a_bound": ESTIMATE_NOT_A_BOUND,
        "per_segment": {
            "guid": [cache.guid[int(index)] for index in held_out_rows],
            "epoch": cache.epoch[held_out_rows],
            "nll_oracle_block": scores,
            "oracle_n_anchors": anchors,
        },
    }


def capacity_verdict(fit: FitResult, wide: Optional[FitResult]) -> Dict[str, Any]:
    r"""Decide whether the comparison probe was limited by its own width.

    Mechanical, and in the direction that matters: a probe whose held-out score improves by more
    than :data:`CAPACITY_MARGIN_NATS` when its width doubles was not measuring the bottleneck, it
    was measuring itself -- and $\Delta_{\mathrm{suff}}$ is then a lower bound on what a larger
    probe would report.

    Args:
        fit: The comparison probe's fit.
        wide: The doubled-width refit, or ``None`` when the check was not run.

    Returns:
        The verdict, the two scores and the margin behind it.
    """
    if wide is None:
        return {
            "checked": False,
            "capacity_bound": None,
            "margin_nats": CAPACITY_MARGIN_NATS,
            "detail": "the doubled-width refit was not run, so capacity adequacy is unmeasured",
        }
    narrow_score, wide_score = fit.final_held_out_nats, wide.final_held_out_nats
    improvement = float(narrow_score) - float(wide_score)
    bound = bool(np.isfinite(improvement) and improvement > CAPACITY_MARGIN_NATS)
    return {
        "checked": True,
        "capacity_bound": bound,
        "width_multiplier": int(wide.width_multiplier),
        "n_parameters": int(fit.n_parameters),
        "n_parameters_wide": int(wide.n_parameters),
        "held_out_nats": narrow_score,
        "held_out_nats_wide": wide_score,
        "improvement_nats": improvement,
        "margin_nats": CAPACITY_MARGIN_NATS,
        "detail": (
            f"doubling the probe's width moved the held-out score by {improvement:.4g} "
            f"nats/anchor against a {CAPACITY_MARGIN_NATS:g} nat margin, so the reported gap is "
            + ("a lower bound on what a larger probe would find" if bound else "not width-limited")
        ),
        "wide_fit": wide.describe(),
    }


def _skip(reason: str, *, cache: Optional[StateCache] = None) -> Dict[str, Any]:
    """Return the recorded-skip shape, with whatever was measured before the guard fired.

    Args:
        reason: Why no measurement was made.
        cache: The cache, when one was built.

    Returns:
        The skip record.
    """
    logger.warning(f"oracle: skipped -- {reason}")
    return {
        "skipped": True,
        "reason": reason,
        "cache": None if cache is None else cache.describe(),
    }
