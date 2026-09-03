r"""The evaluation readouts, the Monte Carlo predictive scores, and the acceptance verdicts.

A fork of :mod:`teb_vae.lag_attn_rws.eval.metrics`, edited for a target domain that is
**wavelet-modulus and phase-harmonic coefficients on a decimated grid** rather than a $4$ Hz raw
signal, and for a forward that decodes a **gathered anchor set** rather than a contiguous prefix.
Five groups of quantity come out of a checkpoint.

**The readouts.** $\mu^p_t$, $\mu^q_t$, $\mu^q_t - \mu^p_t$, $K_t$, the attention $\alpha$ and the
per-lag KL attribution $\widetilde K_{t,\ell}$. They are summarised rather than dumped -- a run
over a real test set holds millions of anchors -- but every summary is of the tensor the model
actually produced, computed with the training objective's own functions.

**The predictive scores.** $D_{\mathrm{base}}$, $D_{\mathrm{full}}$ and $D_{\mathrm{shuffled}}$,
estimated by marginalising the latent over $K$ draws under **common random numbers**: one
$\epsilon$ per draw, shared by every branch, so the base-versus-full difference carries no
independent sampling noise. Under a Gaussian likelihood the marginal is
$\operatorname{logsumexp}_r \log p_r - \log K$ -- an average of *likelihoods*, not of log
likelihoods, which is a different and larger number.

**The baselines.** Persistence, climatology and the segment's own mean, rebuilt **in feature
space** (:func:`baseline_forecasts`) and scored through the same loss function over the same mask
at the same anchors. A summed-$2940$-coefficient block score is a large number under every
predictor -- its scale is set by the block, not by the model -- so it is only readable against
predictors that know nothing. Their observation variance is fixed at $\sigma = 1$ in the loader's
$z$ units and stated, because under a Gaussian likelihood a point predictor has no variance of its
own and the whole skill score would otherwise be decided by an unstated choice.

**The two controls this cell alone can have.** The permutation control answers *specificity* --
does the model read **this** recording's source -- and it structurally cannot see the hazard that
matters most here: the source availability pattern $m^u_{t,c}$ is a deterministic function of $t$,
identical in every row of a batch, and it enters $q(z \mid Y, U)$ but not $p(z \mid Y)$, so the
posterior can be pushed off the prior by the availability *clock* alone. No permutation of rows
removes something every row shares. The fifth branch re-encodes a **zeroed** source stream and
reports ``kld_source_null`` beside ``coupling_minus_clock``; see :func:`evaluate_batch`.

**The calibration.** The decoder's learned $\sigma^2$ is a claim about how wrong the forecast is,
and the whole block NLL is a log density only if that claim holds. It is checked over the scored
**coefficients** themselves -- probability-integral transform, central coverage against the exact
erf nominals, CRPS, and the gain over the best single constant variance fitted to the very
residuals being scored -- by streaming sums, because a real split holds $10^9$ of them and none is
retained.

**The verdicts.** Each is ``PASS``, ``FAIL`` or ``INCONCLUSIVE``, never a bare boolean, and each
carries the numbers that produced it. A label with no numbers behind it is a claim a reader
cannot check. Ten here against the sibling's eight; the two additions are
``coupling_exceeds_availability_clock`` and ``anchor_geometry_intact``.

One aggregation decision runs through all of it: **quantities are averaged per recording, then
across recordings.** Anchors are not independent samples of anything -- consecutive anchors'
forecast windows overlap in $14$ of their $15$ horizon steps at the dense evaluation geometry, and
a single long recording holds hundreds of them -- so a flat anchor mean weights recordings by their
length and reports an effective sample size far larger than the data supports. That chain --
support-weighted within a segment, unweighted over a recording's segments, unweighted across
recordings -- applies to the *vector* readouts too, not only the scalars.

**One pass produces all of it.** The decoder pass over four branches at $K$ draws is the dominant
cost of a run, so :func:`evaluate` takes an ``on_batch`` sink and hands it each batch's readouts
*before* their per-anchor and retained tensors are released.

Four things differ from the sibling beyond the target domain, and each is where a mechanical copy
would have produced a wrong number rather than an exception:

* **The forward is called densely and in exactly one place**, at
  :data:`DENSE_ANCHOR_GEOMETRY`. ``SeqVaeLagAttnCfsTask.resolve_anchor_geometry('test', batch)``
  returns that same pair; the training tiling exists for gradient decorrelation and activation
  memory, neither of which applies where there is no backward pass. A contract measured at the
  training tiling would name an $A_{\max}$ no evaluation ever produces.
* **The target and both masks are read off the forward's own anchor set**, never recomputed --
  ``anchor_index`` and ``anchor_valid`` come back from the forward for the reason
  ``causal_inputs.py`` gives for building them inside it: a second computation could disagree, and
  the disagreement would be a wrong number rather than an exception.
* **The lag-validity mask is the *model's*, not the attention module's.** This cell's
  ``build_lag_mask`` applies a configurable ``lag_floor`` on top of the causal one, so reading
  ``lag_attn.build_lag_mask`` -- which is what the sibling does, correctly, for a model with no
  floor -- would describe a support the attention was never computed over.
* **There is no bpm and no frequency-domain accumulator.** A scattering or phase-harmonic
  coefficient has no clinical unit, and it is a *modulus*: the analysing filter's phase was
  discarded before the value was stored, so the cross-spectral sufficient statistics the sibling
  accumulates have no analogue here at any window length.

Two lag quantities are reported side by side, and they answer different questions. The **raw**
attribution divides every lag bin by the same anchor support, so it keeps summing to $\bar K$ and
is the decomposition the identity test pins. The **support-corrected** profile divides each bin by
its own contributing-anchor count. At the shipped geometry both corrections are inert -- the
earliest decoded anchor is $F = 133$ against a furthest lag of $L - 1 = 90$, so every lag exists at
every scored anchor -- and that is a fact to **measure** rather than assume: an arm lowering the
floor below $90$ reintroduces truncation, and these are what would catch it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from teb_vae.lag_attn.nets.lag_report import (
    lag_compensated_seconds,
    lag_original_sensor_seconds,
)
from teb_vae.lag_attn_cfs.eval.lag_axis import GROUP_DELAY_CAVEAT
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.losses import (
    KLD_ACTIVE_EPS,
    masked_raw_block_per_anchor,
    raw_sample_score,
)

# The two bound-margin conventions, imported rather than restated: "the prior variance is pinned"
# must mean the same thing in a training log and in a summary, and a second literal here is how
# the two would stop meaning it.
from teb_vae.lag_attn_rws.nets.model import LOGVAR_FLOOR_MARGIN_FRAC, SATURATION_FRAC
from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD, forecast_mask, kl_mask

#: The anchor geometry every evaluation forward is called at: phase $0$, stride $1$ -- the dense
#: range ``resolve_anchor_geometry`` returns on ``val`` and ``test``.
#:
#: A constant rather than two literals at the call site, because ``eval/probe.py`` reports the
#: forward contract at this same pair and the two must be one decision. The evaluation deliberately
#: offers no tiled arm: the tiling exists for gradient decorrelation and activation memory, and
#: there is no backward pass here.
DENSE_ANCHOR_GEOMETRY: Tuple[int, int] = (0, 1)

#: Monte Carlo draws per anchor. The specification's starting value; more may be used for a
#: final analysis, and $K = 1$ reduces the estimator exactly to the training-path score.
DEFAULT_NUM_SAMPLES = 8

#: How much worse, in nats per anchor, a forecast from a *stranger's* prior latent must be before
#: the prior is credited with carrying the target's predictive state.
#:
#: Provisional. Where the boundary between "the prior latent is load-bearing" and "the decoder
#: largely ignores it" actually sits is an empirical question the first converged run answers, so
#: the verdict always reports the measured degradation next to its label.
DEFAULT_PRIOR_SHUFFLE_MIN_NATS = 1.0

#: Latent dimensions that must clear :data:`KLD_ACTIVE_EPS` for the latent to count as
#: uncollapsed. Two, matching the specification's "the KL does not collapse into only one or two
#: dimensions".
DEFAULT_MIN_ACTIVE_DIMS = 2

#: The margin $\Delta_{\mathrm{clock}}$ must clear, in nats per anchor. **Unset, deliberately.**
#:
#: A provisional number would decide a FAIL on the first production runs -- which is the run that
#: is supposed to *measure* it. A gate whose threshold is a guess either fails a healthy model or
#: passes a broken one, and nobody can tell which from the output. Unset makes the verdict
#: INCONCLUSIVE and emits the measured difference anyway, so the threshold can be set from data
#: rather than from judgement.
DEFAULT_CLOCK_MARGIN_MIN_NATS: Optional[float] = None

#: Below this total KL (nats per anchor) there is no coupling to be distributed over dimensions,
#: so a collapse verdict would be reporting the absence of a signal as a structural failure.
_COLLAPSE_INCONCLUSIVE_KL = 1e-6

#: How far a geometry guard may sit from its exact expected value before it counts as broken.
#: Both guarded quantities -- the decoded anchor count and the warm target fraction -- are means of
#: values that are *identical* on every sample, so the exact answer is reachable and this tolerance
#: guards only the float accumulation across recordings rather than admitting a real drift.
GEOMETRY_EXACT_TOLERANCE = 1e-9

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

#: The trivial forecast baselines, in reporting order. Every one is a *constant* over the anchor's
#: forecast block, per channel, which is what makes them trivial: they say nothing about the shape
#: of the next minute, only about its level in each frequency band.
BASELINE_NAMES: Tuple[str, ...] = ("persistence", "climatology", "segment_mean")

#: Every point forecast the run scores, model branches first.
FORECAST_BRANCHES: Tuple[str, ...] = ("base", "full", *BASELINE_NAMES)

#: The observation log-variance handed to a trivial baseline: $\sigma = 1$ in the loader's own
#: $z$ units, matching the decoder's head-init calibration.
#:
#: Fixed and stated rather than fitted, because under ``'gaussian_nll'`` a point predictor has no
#: variance of its own and the whole skill score is decided by whatever $\sigma$ it is given. That
#: is also why an MSE-space skill -- in which $\sigma$ cancels -- is reported beside the NLL-space
#: one: a learned-variance model otherwise beats a fixed-variance baseline partly on variance
#: modelling alone, and nothing in a single number would separate the two effects.
BASELINE_LOGVAR = 0.0

#: Bins of the probability-integral-transform histogram. Twenty over $(0, 1)$: enough that a
#: $\cup$ or $\cap$ shape is unmistakable, few enough that every bin is populated on a smoke run.
PIT_BINS = 20

#: Bins of the log-variance histogram, laid out over the model's own clamp range.
LOGVAR_BINS = 60

#: Central-coverage levels reported, in standard deviations.
COVERAGE_LEVELS: Tuple[int, ...] = (1, 2, 3)

#: Their exact nominal coverages, $\operatorname{erf}(k/\sqrt{2})$: $0.6827$, $0.9545$, $0.9973$.
#: Computed rather than written down -- the two-sigma figure is the one people quote as $0.95$,
#: which is a different number and would make a calibrated model look half a point miscalibrated.
COVERAGE_NOMINALS: Tuple[float, ...] = tuple(
    math.erf(float(level) / math.sqrt(2.0)) for level in COVERAGE_LEVELS
)

#: How far the observed **tail** mass may sit from its nominal, relatively, before the calibration
#: verdict fails: a factor of $1.5$ either way.
#:
#: Relative and on the tail rather than absolute and on the coverage, because the three levels'
#: nominals span two orders of magnitude in the tail: an absolute tolerance loose enough to admit
#: sampling noise at one sigma would accept a fifty-fold error at three.
#:
#: Provisional, like every other threshold here, and the verdict always reports the measured
#: coverages beside it.
DEFAULT_COVERAGE_TAIL_TOLERANCE = 0.5

#: How much of a bounded variance may sit within the margin of one of its clamps before the
#: variance counts as pinned there. A half is deliberately permissive: it is not a statement that
#: half a distribution on its bound is healthy, it is the point past which the *readout* built on
#: that variance stops meaning what it says.
DEFAULT_PINNED_VARIANCE_MAX_FRAC = 0.5


# =============================================================================
# Batch plumbing
# =============================================================================
def model_inputs(task: Any, batch: Any) -> Tuple[torch.Tensor, ...]:
    """Assemble the net's inputs from a batch, through the task's own builders.

    Not a re-implementation: the task's builders are what training uses, including their width
    checks, so an evaluation cannot end up feeding the model a differently assembled stream than
    the run it is evaluating did.

    The fourth element is the **feature** target, not a raw trace: ``_build_raw_target`` is
    inherited from ``teb_vae/lag_attn_fs/task.py`` and returns the two stored target blocks
    concatenated at $(B, T, c_y)$ -- the declared width the model's keep-index is positional into.
    The name is the family's and is kept so this seam stays one function across the grid.

    Args:
        task: The Lightning task wrapping the loaded net.
        batch: A batch from the data module.

    Returns:
        ``(y_st, y_ph, u_stream, target_features, weight)``.
    """
    y_st, y_ph = task._build_target_streams(batch)
    u_stream = task._build_source_stream(batch)
    target_features, weight = task._build_raw_target(batch)
    return y_st, y_ph, u_stream, target_features, weight


def batch_field(batch: Any, name: str) -> Any:
    """Read one field off a batch, tolerating both mapping and attribute access.

    The data module hands out an attribute-style batch and every stub in the test suite is a
    mapping or a ``SimpleNamespace``, so a reader that assumes one of the two silently returns
    ``None`` for the other -- which reads downstream as "the shard does not carry this field".

    Args:
        batch: A batch from the data module, or a stub.
        name: The field name.

    Returns:
        The field value, or ``None`` when the batch does not carry it.
    """
    if isinstance(batch, dict):
        return batch.get(name)
    return getattr(batch, name, None)


def batch_size_of(batch: Any) -> int:
    """Return the number of samples in a batch, from a field the model always requires.

    Read from a tensor field rather than from ``guid``, which is a ``list[str]`` a stub batch may
    not carry at all.

    Args:
        batch: A batch from the data module.

    Returns:
        The batch size, or ``0`` when the field is absent.
    """
    value = batch.get("fhr_st") if isinstance(batch, dict) else getattr(batch, "fhr_st", None)
    return 0 if value is None else int(value.shape[0])


def batch_recordings(batch: Any, batch_size: int) -> Optional[List[str]]:
    """Return one recording identifier per sample, or ``None`` when the batch carries none.

    The distinction is what the permutation control needs. ``None`` means *the grouping is
    unknown*, which is not the same as "every sample belongs to one recording": the first calls
    for the ungrouped derangement, the second for excluding the batch from the control entirely.
    Collapsing them would either exclude every stub batch or report a within-recording pairing
    rate of one where nothing is actually known.

    ``guid`` survives collation as a ``list[str]`` rather than a tensor, which is why it is never
    moved to a device and why it is read as a Python value here.

    Args:
        batch: A batch from the data module.
        batch_size: Number of samples, taken from a tensor field rather than from ``guid``
            itself, so a malformed identifier list cannot silently change the sample count.

    Returns:
        A list of length ``batch_size``, or ``None`` when the batch has no ``guid`` field.
    """
    field_value = batch.get("guid") if isinstance(batch, dict) else getattr(batch, "guid", None)
    if field_value is None:
        return None
    if isinstance(field_value, (list, tuple)):
        return [
            str(field_value[index]) if index < len(field_value) else "unknown"
            for index in range(batch_size)
        ]
    if isinstance(field_value, torch.Tensor):
        return [str(field_value[index].item()) for index in range(batch_size)]
    return [str(field_value)] * batch_size


def batch_guids(batch: Any, batch_size: int) -> List[str]:
    """Return one recording identifier per sample, falling back to ``'unknown'``.

    The aggregation needs a label for every sample -- an unlabelled one still has to land in some
    bucket rather than being dropped -- so the absent case becomes a single ``'unknown'``
    recording here. :func:`batch_recordings` is the accessor for callers that must distinguish
    "unknown" from "all one recording".

    Args:
        batch: A batch from the data module.
        batch_size: Number of samples.

    Returns:
        A list of length ``batch_size``; ``'unknown'`` wherever the batch carries no identifier.
    """
    found = batch_recordings(batch, batch_size)
    return ["unknown"] * batch_size if found is None else found


# =============================================================================
# Units
# =============================================================================
#: The only unit label this module exports. Everything stays in the loader's $z$ units.
#:
#: The sibling's ``BPM_UNIT``, ``to_bpm``, ``sigma_to_bpm`` and ``fhr_normalization`` are **deleted
#: rather than repointed**, and the deletion is the decision: a scattering or phase-harmonic
#: coefficient has no clinical unit, and inverting the per-channel statistics would put the $98$
#: channels on scales spanning orders of magnitude -- which destroys every pooled statistic, every
#: shared colour bar and the warm-up tertile split. A function that could be called would be
#: called, so there is none.
#:
#: The **other half of this decision lives in** ``collect.normalization_record``, which the
#: sibling points at ``("fhr", "up")`` and this package points at the four stored feature blocks:
#: the record still has to say what scale the numbers are on even though nothing converts them
#: off it. Named here rather than left to be noticed, because the two are one decision and a
#: record describing the raw traces beside coefficients in $z$ units would read as though the
#: coefficients had been denormalised.
NORMALISED_UNIT = "normalised"


# =============================================================================
# Monte Carlo predictive scores
# =============================================================================
def marginalise_block_scores(block_scores: torch.Tensor, likelihood: str) -> torch.Tensor:
    r"""Marginalise a stack of per-draw block scores over the latent.

    Under ``'gaussian_nll'`` a block score is a negative log-density, so the marginal predictive
    NLL is

    $$D = -\left[\operatorname{logsumexp}_{r=1}^{K}\left(-D_r\right) - \log K\right],$$

    the log of the *average likelihood*. This is not the average of the $D_r$ and is strictly
    smaller than it whenever the draws disagree, by Jensen -- which is the entire point of
    marginalising rather than averaging log scores.

    Under ``'mse'`` a block score is not a log-density and its exponential means nothing, so the
    marginal is the plain expectation over draws. Both cases are the expectation of the
    per-draw quantity taken in the space that quantity lives in.

    Args:
        block_scores: Per-draw per-anchor block scores $(K, B, A_{\max})$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The marginalised per-anchor score $(B, A_{\max})$.
    """
    if likelihood == "gaussian_nll":
        num_samples = int(block_scores.shape[0])
        return -(torch.logsumexp(-block_scores, dim=0) - math.log(float(num_samples)))
    return block_scores.mean(dim=0)


@torch.no_grad()
def mc_predictive_block(
    model: Any,
    branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    anchors: torch.Tensor,
    likelihood: str,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: Optional[torch.Generator] = None,
    persistence: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    r"""Score every branch's forecast under common random numbers, at the decoded anchors.

    One $\epsilon$ is drawn per Monte Carlo replicate and reused by **every** branch, so two
    branches with identical latent parameters produce bitwise identical scores and the
    base-versus-full difference is a difference of predictions rather than of noise.

    **The latent is gathered, never sliced**, and ``anchors`` is a required argument for exactly
    that reason. The sibling decodes ``latent[:, :t_valid]`` -- the contiguous prefix -- which is
    right for a model that decodes every anchor and wrong for one that decodes a set. Left as a
    slice here it would produce $(B, T_{\mathrm{valid}}, H, C)$ against a $(B, A_{\max}, H, C)$
    target: not broadcastable, so it fails loudly rather than silently, but four functions deep and
    $K$ draws into a multi-hour pass. The gather is the model's own, one line of its ``forward``.

    Args:
        model: The net, for its shared decoder.
        branches: ``{name: (mu, logvar)}`` latent parameters, each $(B, T, d_z)$. Every branch
            must share a shape; the first one's shape fixes the noise draw.
        target: The gathered forecast target $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        mask: The forecast mask $(B, A_{\max}, H)$.
        anchors: The decoded anchor index $(B, A_{\max})$, as the forward returned it.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        num_samples: Monte Carlo draws $K$. At $K = 1$ the result is exactly the training-path
            per-anchor score for the same draw.
        generator: Generator for $\epsilon$, on the same device as the latent parameters. The
            estimator is the only place an evaluation *adds* randomness of its own, so it takes
            an explicit stream rather than the global one: two runs of a checkpoint must report
            the same numbers, and a global draw makes that a property of whatever else in the
            process happened to draw first. ``None`` falls back to the global generator.
        persistence: The matched forward's own persistence input $(B, A_{\max}, C_{\mathrm{keep}})$
            on a model whose decoder was built with the residual, and ``None`` otherwise -- which is
            exactly what the decoder then expects, since it refuses either mismatch by name. Taken
            from ``forward_outputs['persistence']`` by the caller rather than re-gathered here: it
            is target-only and identical across branches and draws, so re-deriving it would be a
            second definition of a tensor the forward already produced.

    Returns:
        ``(scores, contributing)``: the marginalised per-anchor score of each branch, and the
        $0/1$ anchor indicator they share.

    Raises:
        ValueError: If ``branches`` is empty or ``num_samples`` is not positive.
    """
    if not branches:
        raise ValueError("mc_predictive_block needs at least one branch to score")
    if int(num_samples) < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    reference_mu = next(iter(branches.values()))[0]
    # Built once, outside both loops: it is a view-expanding index and identical for every branch
    # and every draw, and rebuilding it per draw would be the only per-draw allocation here that
    # carries no information.
    gather_index = anchors.to(torch.long)[:, :, None].expand(-1, -1, reference_mu.shape[-1])
    draws: Dict[str, List[torch.Tensor]] = {name: [] for name in branches}
    contributing: Optional[torch.Tensor] = None

    for _ in range(int(num_samples)):
        # Drawn once, outside the branch loop: this line is the common-random-numbers property.
        # ``normal_`` rather than ``randn_like`` only because the latter takes no generator; the
        # ``None`` branch keeps the original call so an unseeded draw stays bit for bit what it
        # was.
        epsilon = (
            torch.randn_like(reference_mu)
            if generator is None
            else torch.empty_like(reference_mu).normal_(generator=generator)
        )
        for name, (mu, logvar) in branches.items():
            latent = mu + epsilon * torch.exp(0.5 * logvar)
            # The same persistence tensor for every branch and every draw, as the matched forward
            # decoded with: it is target-only, so it is a property of the anchor rather than of the
            # latent, and a branch decoded without it would differ from the others by the residual
            # instead of by its own latent.
            forecast_mu, forecast_logvar = model.decoder(
                latent.gather(1, gather_index), persistence=persistence
            )
            block, contributing = masked_raw_block_per_anchor(
                forecast_mu, target, mask, likelihood=likelihood, logvar=forecast_logvar
            )
            draws[name].append(block)

    assert contributing is not None  # the loops above ran at least once
    scores = {
        name: marginalise_block_scores(torch.stack(blocks, dim=0), likelihood)
        for name, blocks in draws.items()
    }
    return scores, contributing


# =============================================================================
# Trivial forecast baselines, in feature space
# =============================================================================
def baseline_forecasts(
    target_features: torch.Tensor,
    weight: torch.Tensor,
    model: Any,
    anchors: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""Build the three trivial forecasts, each constant over its anchor's block, per channel.

    They exist to answer the question a block NLL alone cannot: is the forecast *good*, or merely
    arithmetically fine? A summed-$2940$-coefficient log-density is a large number under any
    predictor, so the only readable form of it is a comparison against predictors that know
    nothing.

    * **persistence** -- anchor $t$'s whole window is filled with the coefficient vector at the
      last **observed** step at or before $t$, per channel. "Last observed" rather than "last"
      because ``weight`` is the only trustworthy validity signal here: unlike the raw trace, the
      coefficients carry no gap sentinel at all, so a carried-forward invalid step is not an
      outlier that would show up -- it is an ordinary-looking number that would quietly measure the
      gap. The carry-forward is a running maximum over the valid step indices, so an anchor inside
      a gap reuses the last step before it.
    * **climatology** -- exactly $0$ per channel, which is the z-scored population mean. The
      statistics were accumulated *excluding* each channel's warm-up region, which is what makes
      zero the channel mean over the region the model reads.
    * **segment_mean** -- the mean of this segment's own observed steps, per channel. It is
      deliberately the stronger form: **not causal**, since it reads the segment's whole future, so
      a model that fails to beat it has learned nothing recording-specific that a constant could
      not say.

    **Every baseline is built on the gathered kept channels, never on the declared width.** The
    history each reads is ``index_select(target_features, -1, target_gate.keep_index)`` -- the same
    gather ``_build_forecast_target`` applies -- so a baseline block is positionally identical to
    the target it is scored against. Built on the declared $c_y = 102$ it would be scored against a
    $98$-channel target on a channel axis that is **not** aligned: the four dropped channels are
    interior to the declared order, so the mismatch would be a silent mis-pairing of channels
    wherever the shapes happened to survive rather than a clean shape error.

    Every forecast is returned at a *broadcastable* shape rather than expanded over the horizon,
    which the scorer does for free.

    Args:
        target_features: The loader-normalized target stream $(B, T, c_y)$, at the declared width.
        weight: The decimated validity signal $(B, T)$.
        model: The net, for its target gate.
        anchors: The decoded anchor index $(B, A_{\max})$.

    Returns:
        Baseline name to its forecast mean, broadcastable over
        $(B, A_{\max}, H, C_{\mathrm{keep}})$.
    """
    gathered = (
        target_features
        if model.target_gate is None
        else torch.index_select(target_features, -1, model.target_gate.keep_index)
    )
    channels = int(gathered.shape[-1])
    index = anchors.to(torch.long)
    valid = weight >= VALID_THRESHOLD                                        # (B, T)

    # Index of the most recent valid step at or before each step. ``cummax`` over the step
    # indices, with invalid steps sent to -1 so they never win the running maximum.
    steps = torch.arange(weight.shape[1], device=weight.device).expand_as(valid)
    last_valid = torch.cummax(torch.where(valid, steps, torch.full_like(steps, -1)), dim=1).values
    # An anchor with no valid step at or before it is fully masked by ``forecast_mask`` (its own
    # step is invalid), so the clamped fallback never reaches a scored term.
    at_anchor = last_valid.gather(1, index).clamp_min(0)                     # (B, A)
    persistence = gathered.gather(
        1, at_anchor[:, :, None].expand(-1, -1, channels)
    )[:, :, None, :]                                                         # (B, A, 1, C_keep)

    observed = valid.to(gathered.dtype)
    counts = observed.sum(dim=1, keepdim=True)                               # (B, 1)
    totals = (gathered * observed[:, :, None]).sum(dim=1)                    # (B, C_keep)
    # NaN rather than 0.0 where a segment carries no valid step at all: zero is the *climatology*
    # here, so a fabricated zero would silently report the population mean as this segment's mean
    # and score a second identical baseline under a different name. Such a segment scores no
    # anchors, so its whole row leaves the aggregation anyway.
    segment_mean = torch.where(
        counts > 0.0,
        totals / counts.clamp_min(1.0),
        torch.full_like(totals, float("nan")),
    )[:, None, None, :]                                                      # (B, 1, 1, C_keep)

    return {
        "persistence": persistence,
        "climatology": torch.zeros((), dtype=gathered.dtype, device=gathered.device),
        "segment_mean": segment_mean,
    }


def masked_raw_error_sums(
    mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Per-sample sums of the forecast residual, its magnitude and its square.

    $$e = \hat{x} - x, \qquad
    S^{1}_b = \sum m\,e, \quad S^{|1|}_b = \sum m\,|e|, \quad S^{2}_b = \sum m\,e^2,
    \quad n_b = C_{\mathrm{keep}} \sum_{a,\tau} m_{b,a,\tau}.$$

    Sums rather than finished statistics, and the reason is Jensen: an RMSE is the square root of
    a mean, and averaging finished per-sample RMSEs across a recording is biased **low** -- in the
    direction that flatters the model. So the squares accumulate unrooted here and the root is
    taken once, at the end of the aggregation chain.

    The residual is signed *forecast minus truth*, so a positive bias means the forecast runs
    high. In the loader's $z$ units, per coefficient; there is no bpm here.

    Args:
        mu: Forecast mean, broadcastable to $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        target: The gathered forecast target $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        mask: The forecast mask $(B, A_{\max}, H)$.

    Returns:
        ``sum_residual``, ``sum_abs``, ``sum_sq`` and ``n_coefficients``, each $(B,)$.
        ``n_coefficients`` is the scored coefficient count, which is the denominator every one of
        the three needs and which $H \cdot C_{\mathrm{keep}}$ over-states on any anchor with masked
        forecast steps.
    """
    residual = mu - target
    weights = mask[..., None]
    return {
        "sum_residual": (residual * weights).sum(dim=(1, 2, 3)),
        "sum_abs": (residual.abs() * weights).sum(dim=(1, 2, 3)),
        "sum_sq": ((residual**2) * weights).sum(dim=(1, 2, 3)),
        "n_coefficients": mask.sum(dim=(1, 2)) * float(target.shape[-1]),
    }


def branch_channel_scores(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
) -> torch.Tensor:
    r"""One branch's masked score, summed over the horizon and resolved per channel.

    $$S_{b,a,c} = \sum_\tau m_{b,a,\tau}\,
    \ell\!\left(x_{b,a,\tau,c}, \hat{x}_{b,a,\tau,c}\right),
    \qquad \sum_c S_{b,a,c} = D_{b,a}.$$

    The identity on the right holds by construction rather than by arithmetic coincidence: this and
    :func:`~teb_vae.lag_attn_rws.nets.losses.masked_raw_block_per_anchor` reduce the *same*
    elementwise term over different axes. It is what makes the per-channel gap vector, the two
    stored-block gaps and the three warm-up tertile gaps partial sums of the ``pred_gap`` they are
    read beside rather than five unrelated numbers.

    Args:
        mu: Forecast mean $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        logvar: Forecast log-variance, the same shape.
        target: The gathered forecast target, the same shape.
        mask: The forecast mask $(B, A_{\max}, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The per-anchor per-channel score $(B, A_{\max}, C_{\mathrm{keep}})$.
    """
    score = raw_sample_score(mu, target, likelihood=likelihood, logvar=logvar)
    return (score * mask[..., None]).sum(dim=2)


def masked_raw_block_per_horizon_step(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The per-anchor block score resolved by horizon step: summed over $c$, not over $\tau$.

    $$D_{b,a,\tau} = m_{b,a,\tau} \sum_{c} \ell\!\left(x_{b,a,\tau,c}, \hat{x}_{b,a,\tau,c}\right),
    \qquad \sum_\tau D_{b,a,\tau} = D_{b,a}.$$

    The counterpart of :func:`branch_channel_scores` on the other axis, and for the same reason.

    Args:
        mu: Forecast mean, broadcastable to $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        target: The gathered forecast target $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        mask: The forecast mask $(B, A_{\max}, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        logvar: Forecast log-variance, broadcastable to the same shape.

    Returns:
        The per-horizon-step block score $(B, A_{\max}, H)$.
    """
    per_sample = raw_sample_score(mu, target, likelihood=likelihood, logvar=logvar)
    return (per_sample * mask[..., None]).sum(dim=3)


def horizon_block_sums(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
) -> Dict[str, torch.Tensor]:
    r"""Accumulate one branch's horizon-resolved block score and its own denominator.

    $$S^{D}_\tau = \sum_{b,a} D_{b,a,\tau}, \qquad n^{a}_\tau = \sum_{b,a} m_{b,a,\tau}.$$

    The denominator is **per $\tau$**, not the per-anchor contributing indicator. That indicator
    is an ``amax`` over $\tau$, so using it would divide a late horizon's numerator -- which the
    mask has already zeroed wherever that step falls in a gap -- by a count that includes those
    zeros, and the late horizons would read artificially good exactly where the signal is worst.

    Args:
        mu: Forecast mean $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        logvar: Forecast log-variance, the same shape.
        target: The gathered forecast target, the same shape.
        mask: The forecast mask $(B, A_{\max}, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        ``sum_block`` and ``n_anchors``, each $(H,)$ in float64 -- a real split reaches $10^9$
        terms, where a float32 accumulator stops adding.
    """
    per_tau = masked_raw_block_per_horizon_step(
        mu, target, mask, likelihood=likelihood, logvar=logvar
    )
    return {
        "sum_block": per_tau.sum(dim=(0, 1), dtype=torch.float64),
        "n_anchors": mask.sum(dim=(0, 1), dtype=torch.float64),
    }


def horizon_residual_sums(
    mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Sum the residual and the log-variance over every scored coefficient, per horizon step.

    $$S^{\mathrm{sq}}_\tau = \sum_{b,a,c} m_{b,a,\tau}\,(x - \mu)^2, \qquad
    S^{z}_\tau = \sum_{b,a,c} m_{b,a,\tau}\,(x - \mu)^2 e^{-\log\sigma^2}, \qquad
    n_\tau = C_{\mathrm{keep}} \sum_{b,a} m_{b,a,\tau}.$$

    An accumulator rather than a retention. The residuals and log-variances themselves are
    $A_{\max} \times H \times C_{\mathrm{keep}}$ per sample -- about a megabyte each, tens of
    gigabytes over a real split -- while what a calibration or a horizon-resolved skill number
    needs from them is these four vectors of length $H$. $S^{z}_\tau / n_\tau$ is the standardised
    residual variance, which a calibrated learned variance puts at $1$.

    The $\tau$ resolution is the point: it is the one axis that survives on neither durable table,
    because both are keyed per anchor and $\tau$ lives *inside* an anchor. The denominator is
    $C_{\mathrm{keep}} \sum_{b,a} m_{b,a,\tau}$ rather than the per-anchor contributing indicator,
    which is an ``amax`` over $\tau$ and would count a masked forecast step as a scored zero.

    Sums accumulate in float64 -- a real split reaches $10^9$ terms, where float32 stops adding.

    Args:
        mu: Forecast mean $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        logvar: Forecast log-variance, the same shape.
        target: The gathered forecast target, the same shape.
        mask: The forecast mask $(B, A_{\max}, H)$.

    Returns:
        The four sums, each $(H,)$ in float64.
    """
    residual_sq = (target - mu) ** 2
    masked = mask[..., None]
    channels = float(target.shape[-1])
    return {
        "sum_sq": (residual_sq * masked).sum(dim=(0, 1, 3), dtype=torch.float64),
        "sum_standardised_sq": (residual_sq * torch.exp(-logvar) * masked).sum(
            dim=(0, 1, 3), dtype=torch.float64
        ),
        "sum_logvar": (logvar * masked).sum(dim=(0, 1, 3), dtype=torch.float64),
        "count": mask.sum(dim=(0, 1), dtype=torch.float64) * channels,
    }


# =============================================================================
# Per-batch evaluation
# =============================================================================
#: The vector readouts, by attribute name. Every one is per sample on a :class:`BatchReadout` and
#: a plain list on an :class:`Aggregate`, and they all travel the same aggregation chain, so the
#: chain is written once and driven from this tuple rather than repeated fourteen times.
#:
#: The last three are this cell's own and are the **channel** axis rather than the lag axis; see
#: :class:`BatchReadout` for why they are per sample and what they must sum to.
VECTOR_READOUTS: Tuple[str, ...] = (
    "kld_per_dim",
    "kld_per_head",
    "lag_profile",
    "lag_profile_support_corrected",
    "lag_profile_untruncated",
    "lag_profile_null",
    "lag_profile_null_untruncated",
    "lag_profile_per_head",
    "lag_support",
    "attention_profile",
    "attention_profile_support_corrected",
    "attention_profile_untruncated",
    "attention_profile_per_head",
    "attention_entropy_per_head",
    "gap_per_channel",
    "sq_error_per_channel_base",
    "sq_error_per_channel_full",
)


@dataclass
class BatchReadout:
    r"""Per-sample readouts from one batch, plus the anchor counts that weight them.

    Every scalar column is a per-sample mean over that sample's contributing anchors, and
    ``n_anchors`` is how many those were. Keeping the count is what lets an anchor-weighted total
    be reconstructed exactly -- which is how the evaluation is checked against the training loss
    -- while the per-sample values are what the per-recording aggregation needs.

    The vector readouts are **per sample** for the same reason the scalars are. Reduced over the
    batch here, they would reach the aggregate as one number per batch and be averaged with equal
    weight regardless of how many anchors or recordings the batch held -- so the per-dimension KL
    summed over dimensions would not equal the headline KL, which is the quantity it decomposes.

    Attributes:
        guids: Recording identifier per sample.
        columns: Named per-sample values, each a $(B,)$ tensor.
        n_anchors: Contributing anchors per sample, $(B,)$.
        kld_per_dim: Per-sample per-dimension KL over the sample's masked anchors, $(B, d_z)$.
        kld_per_head: The same KL split across the attention heads, $(B, M)$ -- the additive
            decomposition a head-structured posterior buys, which sums over heads to the sample's
            ``source_conditioned_kl_raw``.
        lag_profile: Per-sample raw per-lag KL attribution, $(B, L)$; sums over lags to the
            sample's ``source_conditioned_kl_raw``.
        lag_profile_support_corrected: The same attribution divided by each lag's own
            contributing-anchor count rather than by the common anchor total, $(B, L)$. At the
            shipped floor every lag exists at every scored anchor, so the correction is inert --
            which is a fact the run **measures** rather than assumes, because an arm lowering the
            floor below $L - 1$ reintroduces the truncation it corrects for.
        lag_profile_untruncated: The same attribution over the anchors at which **every** lag
            exists ($t \ge L - 1$), $(B, L)$. Identical to the unrestricted profile at the shipped
            floor, and not at a lowered one.
        lag_support: Contributing anchors per lag, $(B, L)$ -- the denominator above, carried so
            the correction can be checked and re-derived rather than trusted.
        lag_profile_null: The same attribution built from the **source-null** arm -- the
            posterior re-posed against a zeroed source stream, against its own attention over the
            lags, $(B, L)$. Sums over lags to ``kld_source_null``, so the difference against
            ``lag_profile`` sums to ``coupling_minus_clock``: it is the per-lag decomposition of
            the clock-excess coupling, which is the quantity the availability-clock verdict gates
            and the only lag readout here with the availability staircase removed.
        lag_profile_null_untruncated: The null attribution over the anchors at which **every** lag
            exists -- the partner of ``lag_profile_untruncated``, so a clock-excess formed against
            the untruncated matched profile is a difference of two profiles reduced over one
            anchor set rather than a mixture of two.
        lag_profile_per_head: The KL attribution **before** the sum over heads, flattened
            head-major to $(B, M \cdot L)$ so it travels the one-trailing-axis chain every other
            vector readout does; reshaped once, in the consumer. Summing it over $m$ returns
            ``lag_profile``'s numerator exactly, so it refines the shipped decomposition rather
            than restating it. It is **not** the product of ``kld_per_head`` and
            ``attention_profile_per_head``: those are two anchor means and this is the anchor mean
            of their product, and the gap between them is the within-segment covariance of a
            head's KL with its own attention.
        attention_profile: Per-sample head-averaged attention per lag, $(B, L)$.
        attention_profile_support_corrected: The same attention divided by each lag's own
            contributing-anchor count, $(B, L)$.
        attention_profile_untruncated: The head-averaged attention over the anchors at which
            **every** lag exists, $(B, L)$. The support correction fixes each bin's denominator; it
            cannot fix its numerator, because at a truncated anchor the probability mass that had
            nowhere to go among the long lags was renormalised onto the short ones. Restricting the
            anchor set is what removes that, at the cost of the anchors it drops -- so both travel.
        attention_profile_per_head: Per-head attention per lag, flattened head-major to
            $(B, M \cdot L)$. Flattened rather than kept $(B, M, L)$ so it travels the same
            one-trailing-axis aggregation chain every other vector readout does.
        attention_entropy_per_head: Per-head entropy of the attention over lags, in nats,
            $(B, M)$ -- averaged over the anchors rather than taken of the averaged profile, so
            it is comparable with the per-anchor attainable ceiling.
        gap_per_channel: The forecast gap $D_0 - D_1$ resolved per **surviving target channel**, in
            nats per anchor, $(B, C_{\mathrm{keep}})$. It sums over channels to the sample's
            ``pred_gap``, over each stored block to ``pred_gap_st`` / ``pred_gap_ph``, and over each
            warm-up tertile to ``pred_gap_warm_lo`` / ``_mid`` / ``_hi`` -- so the vector and those
            five scalars cannot disagree about the same decomposition. This is what the
            band-resolved skill analysis is computed from, and it is per sample because every
            statistic in this pipeline reduces per recording before it reduces anything else.
        sq_error_per_channel_base: Per-channel masked mean squared error of the target-only
            branch, $(B, C_{\mathrm{keep}})$; its mean over channels is ``sq_error_base``. A skill
            score resolved by frequency band needs a natural zero, and a pooled squared error has
            none.
        sq_error_per_channel_full: The same for the source-conditioned branch.
        n_control_pairs: Samples paired against a *known* other recording by the permutation
            control. Zero when the batch carries no identifiers, which is not the same as a
            within-recording pairing rate of zero.
        n_same_recording_pairs: How many of those pairs landed inside their own recording. Zero
            by construction under a grouped derangement, and reported anyway: a control that has
            silently stopped being a control looks exactly like one that works.
        per_anchor: The same quantities *before* the within-sample reduction, each
            $(B, A_{\max})$ -- what the per-anchor table is built from. ``anchor_index`` travels
            with them because the anchor axis here is a **gathered set**: a row's position in it is
            not the decimated step it scores, so a table keyed on position alone could not be
            joined against anything else in the run. Carried on the readout rather than recomputed
            because recomputing them means a second decoder pass, and released by :func:`evaluate`
            as soon as a sink has consumed them.
        per_anchor_vectors: Per-anchor **vector** quantities, each $(B, A_{\max}, L)$, gathered
            at the decoded anchors exactly as ``per_anchor`` is: ``kl_lag_map``, the pooled KL
            attribution $\widetilde K_{t,\ell} = \sum_m K^{(m)}_t lpha^{(m)}_{t,\ell}$ at
            every anchor, and ``attention_lag_map``, the head-averaged attention at the same
            anchors. They are what lets an analysis select anchors by their own $K_t$ and read
            the lag structure of the selection -- a question the per-sample profiles, which
            average every anchor together, cannot answer. Released by :func:`evaluate` with
            ``per_anchor`` once a sink has consumed them.
        retained: Whole model tensors a caller asked to keep, by their forward-output name.
            Empty unless ``retain`` named them: a retained forecast set here is four
            $(A_{\max}, H, C_{\mathrm{keep}})$ tensors, about $3.4$ MiB per sample.
        horizon_sums: Residual, log-variance and block-score sums resolved by horizon step, each
            $(H,)$, per branch. The $\tau$ axis lives inside an anchor, so it survives on neither
            table and cannot be recovered from either.
        calibration_sums: The observation model's calibration accumulators over the *full*
            branch's scored coefficients -- see :func:`calibration_sums`. Empty under ``'mse'``,
            where the decoder's log-variance head is never trained and a probability-integral
            transform of its output would be arithmetic over an untrained tensor.
    """

    guids: List[str]
    columns: Dict[str, torch.Tensor]
    n_anchors: torch.Tensor
    kld_per_dim: torch.Tensor
    lag_profile: torch.Tensor
    lag_profile_support_corrected: torch.Tensor
    lag_profile_untruncated: torch.Tensor
    lag_profile_null: torch.Tensor
    lag_profile_null_untruncated: torch.Tensor
    lag_profile_per_head: torch.Tensor
    lag_support: torch.Tensor
    attention_profile: torch.Tensor
    attention_profile_support_corrected: torch.Tensor
    attention_profile_untruncated: torch.Tensor
    attention_profile_per_head: torch.Tensor
    attention_entropy_per_head: torch.Tensor
    kld_per_head: torch.Tensor
    gap_per_channel: torch.Tensor
    sq_error_per_channel_base: torch.Tensor
    sq_error_per_channel_full: torch.Tensor
    n_control_pairs: int = 0
    n_same_recording_pairs: int = 0
    per_anchor: Dict[str, torch.Tensor] = field(default_factory=dict)
    per_anchor_vectors: Dict[str, torch.Tensor] = field(default_factory=dict)
    retained: Dict[str, torch.Tensor] = field(default_factory=dict)
    horizon_sums: Dict[str, torch.Tensor] = field(default_factory=dict)
    calibration_sums: Dict[str, torch.Tensor] = field(default_factory=dict)


def _per_sample_mean(per_anchor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average a per-anchor quantity within each sample, over its weighted anchors.

    Args:
        per_anchor: $(B, T_\\ast)$ values.
        weights: $(B, T_\\ast)$ non-negative weights; zero anchors drop out entirely.

    Returns:
        $(B,)$ per-sample means.
    """
    return (per_anchor * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _per_sample_element_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Average a per-coefficient quantity within each sample, over its scored coefficients.

    The forecast-side counterpart of :func:`_per_sample_mean`: the mask is per anchor and horizon
    step, and every one of the $C_{\mathrm{keep}}$ coefficients inside a horizon step shares its
    validity. The denominator is therefore $C_{\mathrm{keep}} \sum_{a,\tau} m_{a,\tau}$ -- the
    scored coefficient count -- and not $H \cdot C_{\mathrm{keep}}$, which over-states it on any
    anchor with masked forecast steps.

    Args:
        values: $(B, A_{\max}, H, C_{\mathrm{keep}})$ values, or anything broadcastable to that
            shape.
        mask: The forecast mask $(B, A_{\max}, H)$.

    Returns:
        $(B,)$ per-sample means.
    """
    weights = mask[..., None]
    channels = float(values.shape[-1])
    denominator = (mask.sum(dim=(1, 2)) * channels).clamp_min(1.0)
    return (values * weights).sum(dim=(1, 2, 3)) / denominator


def _per_sample_vector_mean(per_anchor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average a per-anchor *vector* quantity within each sample, over its weighted anchors.

    The same reduction as :func:`_per_sample_mean` with a trailing axis carried through, so a
    vector readout and the scalar it sums to cannot be averaged over two different supports.

    Args:
        per_anchor: $(B, T_\\ast, C)$ values.
        weights: $(B, T_\\ast)$ non-negative weights.

    Returns:
        $(B, C)$ per-sample means.
    """
    numerator = (per_anchor * weights.unsqueeze(-1)).sum(dim=1)
    return numerator / weights.sum(dim=1).clamp_min(1.0).unsqueeze(-1)


def lag_anchor_counts(support: torch.Tensor, lag_validity: torch.Tensor) -> torch.Tensor:
    r"""Count, per sample and per lag, the supported anchors at which that lag exists.

    $$n_{b,\ell} = \sum_t m^{\mathrm{KL}}_{b,t}\,\mathbb{1}[t - \ell \ge F_u].$$

    Lag $\ell$ refers to source step $t - \ell$, which does not exist for $\ell > t$ and is
    forbidden below this cell's configurable lag floor $F_u$, so a long lag is contributed to by
    fewer anchors than a short one wherever the anchor floor does not already cover the whole lag
    window. The validity indicator is **the model's own** ``build_lag_mask`` rather than the
    attention module's, because this cell's floors that mask: reading the module's would describe a
    support the attention was never computed over.

    Args:
        support: The KL anchor mask $(B, T)$.
        lag_validity: The lag-validity mask $(T, L)$, ``True`` where the lagged source step is
            readable.

    Returns:
        The per-lag anchor counts $(B, L)$.
    """
    return support @ lag_validity.to(support.dtype)


def lag_profiles(
    lag_map: torch.Tensor, support: torch.Tensor, lag_validity: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Reduce a per-anchor lag attribution to its raw and support-corrected per-sample profiles.

    The two differ only in the denominator, and that is the whole point:

    $$\text{raw}_{b,\ell} = \frac{\sum_t \widetilde K_{b,t,\ell}\, m_{b,t}}{\sum_t m_{b,t}},
    \qquad
    \text{corrected}_{b,\ell} = \frac{\sum_t \widetilde K_{b,t,\ell}\, m_{b,t}}{n_{b,\ell}}.$$

    The raw form keeps $\sum_\ell \text{raw}_{b,\ell} = \bar K_b$ -- it is a decomposition of the
    sample's own KL, and the identity is a pinned property. The corrected form is the per-anchor
    mean each lag actually earned, and sums to nothing in particular.

    **At the shipped geometry the two coincide**, because the earliest decoded anchor is $F = 133$
    against a furthest lag of $L - 1 = 90$, so every lag is valid at every scored anchor and every
    per-lag count is the common total. Both are still emitted: the equality is a property of this
    floor rather than of the domain, and an arm that lowers it makes them differ again.

    Args:
        lag_map: Per-anchor per-lag attribution $(B, T, L)$.
        support: The KL anchor mask $(B, T)$.
        lag_validity: The lag-validity mask $(T, L)$.

    Returns:
        ``(raw, corrected, counts)``, each $(B, L)$.
    """
    weighted = (lag_map * support.unsqueeze(-1)).sum(dim=1)
    total = support.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
    counts = lag_anchor_counts(support, lag_validity)
    return weighted / total, weighted / counts.clamp_min(1.0), counts


def untruncated_anchor_mask(
    seq_len: int, n_lags: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    r"""Anchors at which **every** lag exists: $t \ge L - 1$.

    The support correction of :func:`lag_profiles` fixes each bin's *denominator*. It cannot fix
    its numerator, and for a quantity that is a probability distribution over lags it does not
    even try: at a truncated anchor the mass that had no long lag to go to was renormalised onto
    the short ones, so the short bins are inflated by an amount no per-lag anchor count knows
    about. Restricting the anchor set is what removes that, at the cost of the anchors it drops.

    At the shipped geometry it drops **nothing**: $L = 91$ and the earliest decoded anchor is
    $F = 133$, so the restriction is a no-op and the restricted profile equals the unrestricted
    one. That is measured rather than assumed -- see ``eval/preflight.py::lag_support`` for the
    margin, and the ``attention`` and ``lag_kl`` analyses for the assertion.

    Args:
        seq_len: Sequence length $T$.
        n_lags: Lag window width $L$.
        device: Device to build the mask on.

    Returns:
        A $(T,)$ bool mask, ``True`` where the anchor's lag support is complete.
    """
    return torch.arange(int(seq_len), device=device) >= (int(n_lags) - 1)


def attainable_lag_entropy(
    seq_len: int,
    n_lags: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""The largest entropy an attention distribution can have at each anchor, in nats.

    $$H^{\max}_t = \log \min(t + 1, L).$$

    A uniform distribution over $n$ outcomes has entropy $\log n$, and at anchor $t$ only
    $\min(t + 1, L)$ lags exist at all. So $\log L$ is **not** the ceiling on a sequence whose
    early anchors are truncated. At this cell's shipped floor no scored anchor is truncated and
    the ceiling is exactly $\log L$ -- both are reported, because the gap between them is a
    property of the geometry rather than of the model and an arm can reopen it.

    Args:
        seq_len: Sequence length $T$.
        n_lags: Lag window width $L$.
        device: Device to build the vector on.
        dtype: Floating dtype, matched to the attention it will be compared against.

    Returns:
        A $(T,)$ tensor of per-anchor attainable entropies.
    """
    steps = torch.arange(int(seq_len), device=device, dtype=dtype)
    return torch.log(torch.clamp(steps + 1.0, max=float(n_lags)))


def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    r"""Per-anchor, per-head entropy of the attention over lags, in nats.

    $$H_{t}^{(m)} = -\sum_\ell \alpha^{(m)}_{t,\ell} \log \alpha^{(m)}_{t,\ell},$$

    with $0 \log 0 = 0$. That convention is load-bearing rather than pedantic: ``entmax15`` --
    which the shipped model uses -- assigns lags *exactly* zero, and so do the causal mask and this
    cell's lag floor at every anchor they exclude, so a naive $p \log p$ produces ``nan`` on the
    majority of rows.

    Taken per anchor and averaged afterwards, never of the averaged profile. The entropy of a
    mixture is at least the mean of the entropies mixed, so the second reads high by an amount
    that grows with how much the attention *moves* between anchors -- reporting a model whose lag
    focus shifts over time as one that has no focus at all.

    Args:
        attn_weights: Attention probabilities $(B, T, M, L)$ in lag order.

    Returns:
        The entropies $(B, T, M)$.
    """
    probabilities = attn_weights.clamp_min(0.0)
    logs = torch.where(
        probabilities > 0.0, probabilities.log(), torch.zeros_like(probabilities)
    )
    return -(probabilities * logs).sum(dim=-1)


def identity_residual_per_sample(
    decomposed: torch.Tensor, total: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    r"""The worst per-anchor disagreement between a decomposition and the total it decomposes.

    $$r_b = \max_{t\,:\,m_{b,t} > 0} \left| \sum_c x_{b,t,c} - y_{b,t} \right|.$$

    A **max**, not a mean. Both identities this measures -- the lag map summing to $K_t$ and the
    per-head KL summing to $K_t$ -- hold anchor by anchor or not at all, and an anchor-averaged
    residual lets a positive violation at one anchor cancel a negative one at another. The
    quantity that would break the identity is attention dropout, whose per-anchor error has zero
    mean by construction, so the mean is precisely the statistic that cannot see it.

    Args:
        decomposed: The decomposition $(B, T, C)$.
        total: The quantity it must sum to, $(B, T)$.
        support: The anchor mask $(B, T)$; unsupported anchors do not count.

    Returns:
        The per-sample residual $(B,)$. Zero for a sample with no supported anchor, which the
        collection pass blanks to ``NaN`` along with every other column of an unscored segment.
    """
    residual = (decomposed.sum(dim=-1) - total).abs() * (support > 0).to(total.dtype)
    return residual.amax(dim=1)


def calibration_sums(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    logvar_clamp: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    r"""Accumulate everything the observation model's calibration is judged from.

    The learned $\sigma^2$ is a claim about how wrong the forecast is, and the only way to check
    it is over the scored **coefficients** themselves -- of which a real split holds $10^9$. So
    nothing is retained: each quantity below is a sum or a histogram, and each is exact against a
    full-retention reference because addition is.

    With $z = (x - \mu)/\sigma$:

    * **PIT.** $u = \Phi(z)$ is uniform on $(0, 1)$ exactly when the observation model is right.
      Histogrammed rather than kept, because the shape is the whole content: a $\cup$ shape means
      the variance is too small and a $\cap$ shape means it is too large.
    * **Central coverage.** $\#\{|z| \le k\}$ for $k \in \{1, 2, 3\}$, against the erf nominals.
      Three counters rather than reading them off the PIT histogram, whose bins do not fall on
      those points.
    * **CRPS**, in closed form for a Gaussian:
      $\sigma\left[z(2\Phi(z) - 1) + 2\phi(z) - \pi^{-1/2}\right]$. A proper score that, unlike
      the NLL, is bounded and does not diverge on a single badly-placed coefficient.
    * **The three NLL sums**, which turn into the gain over the homoscedastic MLE fitted to these
      very residuals -- the comparison that says whether the *learned* variance earned anything
      over one constant $\sigma$. One constant across all $98$ channels, which is what makes it a
      floor rather than a flattering estimate: the channels are z-scored individually, so a single
      $\sigma$ is a stronger competitor here than it would be on a raw waveform.
    * **The log-variance histogram**, over the clamp's own range. A mean alone is equally
      consistent with a well-spread distribution and with half the mass pinned on each clamp.

    Args:
        mu: Forecast mean $(B, A_{\max}, H, C_{\mathrm{keep}})$.
        logvar: Forecast log-variance, the same shape.
        target: The gathered forecast target, the same shape.
        mask: The forecast mask $(B, A_{\max}, H)$.
        logvar_clamp: The model's own $(\mathrm{lo}, \mathrm{hi})$ log-variance bound, which fixes
            the histogram's range so two runs' histograms are comparable bin by bin.

    Returns:
        The sums and the two histograms, each reduced in float64 -- the counts alone reach $10^9$,
        where a float32 accumulator stops adding. The *elementwise* arithmetic stays in the
        model's own dtype, as every other accumulator here does: promoting the whole grid to
        float64 before reducing it would double the peak allocation of the pass for no accuracy
        the reduction does not already give.
    """
    weights = mask[..., None].expand_as(target).reshape(-1)
    residual = (target - mu).expand_as(target).reshape(-1)
    flat_logvar = logvar.expand_as(target).reshape(-1)
    sigma = torch.exp(0.5 * flat_logvar)
    standardised = residual / sigma

    normal_cdf = 0.5 * (1.0 + torch.erf(standardised / math.sqrt(2.0)))
    normal_pdf = torch.exp(-0.5 * standardised**2) / math.sqrt(2.0 * math.pi)
    crps = sigma * (
        standardised * (2.0 * normal_cdf - 1.0) + 2.0 * normal_pdf - 1.0 / math.sqrt(math.pi)
    )

    lo, hi = float(logvar_clamp[0]), float(logvar_clamp[1])
    sums: Dict[str, torch.Tensor] = {
        "count": weights.sum(dtype=torch.float64),
        "sum_residual_sq": (residual**2 * weights).sum(dtype=torch.float64),
        "sum_standardised_sq": (standardised**2 * weights).sum(dtype=torch.float64),
        "sum_logvar": (flat_logvar * weights).sum(dtype=torch.float64),
        "crps_sum": (crps * weights).sum(dtype=torch.float64),
        "pit_histogram": _weighted_histogram(normal_cdf, weights, PIT_BINS, 0.0, 1.0),
        "logvar_histogram": _weighted_histogram(flat_logvar, weights, LOGVAR_BINS, lo, hi),
    }
    for level in COVERAGE_LEVELS:
        sums[f"within_{level}_sigma"] = (
            (standardised.abs() <= float(level)).to(weights.dtype) * weights
        ).sum(dtype=torch.float64)
    return sums


def _weighted_histogram(
    values: torch.Tensor, weights: torch.Tensor, bins: int, low: float, high: float
) -> torch.Tensor:
    """Histogram ``values`` into ``bins`` equal bins over ``[low, high]``, weighted by ``weights``.

    ``torch.histc`` takes no weights, so the mask would have to be applied by indexing -- which
    materialises a copy of a $10^9$-element tensor. Bucketising and using ``bincount``'s weight
    argument keeps the peak allocation at one index array.

    The weights are promoted to float64 for the count, and that is not optional: a production
    split puts $\\approx 5 \\times 10^7$ values in a bin, and float32 stops representing
    consecutive integers at $2^{24} \\approx 1.7 \\times 10^7$ -- so the histogram would silently
    stop counting partway through the pass.

    Args:
        values: The sample, flattened.
        weights: Per-value weight, typically the $0/1$ mask.
        bins: Bin count.
        low: Left edge; anything below it lands in the first bin.
        high: Right edge; anything at or above it lands in the last bin.

    Returns:
        The weighted counts $(\\mathrm{bins},)$ in float64.
    """
    width = (float(high) - float(low)) / float(bins)
    index = ((values - float(low)) / width).floor().to(torch.int64).clamp_(0, int(bins) - 1)
    return torch.bincount(
        index, weights=weights.to(torch.float64), minlength=int(bins)
    ).to(torch.float64)


def calibration_report(
    sums: Mapping[str, Any], *, logvar_clamp: Optional[Sequence[float]] = None
) -> Dict[str, Any]:
    r"""Turn the accumulated calibration sums into the statistics they exist for.

    Pure arithmetic over the sums, so it runs identically on the tensors a pass produced and on
    the lists an earlier pass wrote into ``collection.json`` -- which is what lets the calibration
    analysis re-run offline against a finished directory.

    The NLL gain is the comparison worth the most here. The model's own mean negative log density
    per **coefficient** is $\tfrac{1}{2}[\log 2\pi + \overline{\log \sigma^2} + \overline{z^2}]$;
    the homoscedastic maximum-likelihood alternative fits **one** variance to the very residuals
    being scored, $\hat\sigma^2 = \overline{e^2}$, and scores
    $\tfrac{1}{2}[\log 2\pi + \log \hat\sigma^2 + 1]$. Their difference is what the *learned*,
    input-dependent variance earned over the best possible constant one -- and fitting the
    alternative on the scored residuals rather than on a held-out set is deliberate: it makes the
    baseline as strong as it can be, so the gain is a floor rather than a flattering estimate.

    **The scored unit is a coefficient, and the key names say so.** The sibling reports
    ``gain_per_raw_sample``; there is no raw sample anywhere in this pipeline for a gain to be per,
    and the two denominators differ by a factor of three at the shipped geometry -- so a column
    carried across under the sibling's name would be silently non-comparable with the sibling's
    number.

    Args:
        sums: What :func:`calibration_sums` accumulated, as tensors, arrays or plain lists.
        logvar_clamp: The bound the log-variance histogram was laid out over, for its bin edges.

    Returns:
        The PIT histogram and its worst departure from uniformity, central coverage against the
        erf nominals, mean CRPS, the standardised residual variance a calibrated model puts at
        $1$, the NLL gain per coefficient, and the log-variance histogram. Empty when nothing was
        scored -- a calibration statement over nothing is a skip, not a number.
    """
    def _scalar(name: str) -> float:
        value = sums.get(name)
        return float("nan") if value is None else float(np.asarray(value, dtype=np.float64))

    def _vector(name: str) -> np.ndarray:
        value = sums.get(name)
        return (
            np.zeros(0, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64).reshape(-1)
        )

    count = _scalar("count")
    if not np.isfinite(count) or count <= 0.0:
        return {}

    pit_counts = _vector("pit_histogram")
    pit_total = float(pit_counts.sum()) or 1.0
    # The empirical CDF at each bin edge against the uniform one it should follow: a
    # Kolmogorov-Smirnov statistic at histogram resolution, which is the one number that says how
    # far from uniform the PIT is without a reader having to interpret a shape.
    empirical_cdf = np.cumsum(pit_counts) / pit_total
    uniform_cdf = np.arange(1, pit_counts.size + 1, dtype=np.float64) / max(pit_counts.size, 1)
    logvar_counts = _vector("logvar_histogram")
    lo, hi = (
        (float(logvar_clamp[0]), float(logvar_clamp[1]))
        if logvar_clamp is not None and len(logvar_clamp) == 2
        else (float("nan"), float("nan"))
    )

    mean_standardised_sq = _scalar("sum_standardised_sq") / count
    mean_logvar = _scalar("sum_logvar") / count
    residual_variance = _scalar("sum_residual_sq") / count
    model_nll = 0.5 * (math.log(2.0 * math.pi) + mean_logvar + mean_standardised_sq)
    homoscedastic_nll = 0.5 * (
        math.log(2.0 * math.pi) + math.log(max(residual_variance, np.finfo(np.float64).tiny)) + 1.0
    )
    return {
        "n_coefficients": int(count),
        "pit": {
            "n_bins": int(pit_counts.size),
            "bin_edges": np.linspace(0.0, 1.0, pit_counts.size + 1).tolist(),
            "counts": pit_counts.tolist(),
            # Density rather than counts for the figure: a flat line at 1.0 is the calibrated
            # answer whatever the bin count, and a count histogram's height is not.
            "density": (pit_counts / pit_total * float(pit_counts.size)).tolist(),
            "max_cdf_deviation": float(np.max(np.abs(empirical_cdf - uniform_cdf)))
            if pit_counts.size
            else float("nan"),
        },
        "coverage": [
            {
                "level_sigma": int(level),
                "nominal": float(nominal),
                "observed": _scalar(f"within_{level}_sigma") / count,
                "observed_tail": 1.0 - _scalar(f"within_{level}_sigma") / count,
                "nominal_tail": 1.0 - float(nominal),
                "n_coefficients": int(count),
            }
            for level, nominal in zip(COVERAGE_LEVELS, COVERAGE_NOMINALS)
        ],
        "crps_normalised": _scalar("crps_sum") / count,
        # One when the learned variance is right on average; above one when it is too small.
        "mean_standardised_sq": mean_standardised_sq,
        "mean_logvar_full": mean_logvar,
        "residual_variance": residual_variance,
        "nll": {
            "model_per_coefficient": model_nll,
            "homoscedastic_per_coefficient": homoscedastic_nll,
            # Positive means the input-dependent variance beat the best constant one.
            "gain_per_coefficient": homoscedastic_nll - model_nll,
            "homoscedastic_sigma": float(np.sqrt(max(residual_variance, 0.0))),
        },
        "logvar": {
            "n_bins": int(logvar_counts.size),
            "bin_edges": np.linspace(lo, hi, logvar_counts.size + 1).tolist()
            if logvar_counts.size
            else [],
            "counts": logvar_counts.tolist(),
            "clamp": [lo, hi],
        },
        # Pooled over coefficients rather than chained per recording, and said so here: PIT and
        # coverage are statements about a distribution, not means of a per-recording quantity, so
        # this census weights a recording by how many coefficients it contributed. The
        # per-recording chained figures -- mean log-variance and both clamp fractions -- are
        # columns on the per-sample table, where the chain does apply.
        "weighting": "pooled over scored coefficients, not averaged per recording",
        "unit": NORMALISED_UNIT,
    }


# =============================================================================
# The two readouts only this cell has: the availability clock, and the source's warmth
# =============================================================================
@torch.no_grad()
def source_null_kld_per_sample(
    model: Any,
    outputs: Mapping[str, torch.Tensor],
    u_stream: torch.Tensor,
    kl_support: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""The KL a source carrying **no variation** induces, per sample, on the run's own support.

    $$\texttt{kld\_source\_null}_b \;=\;
    \frac{\sum_{t} m^{\mathrm{KL}}_{b,t}
    \sum_{d} \mathrm{KL}\!\left(q^{\mathrm{null}}_{b,t,d} \,\Vert\, p_{b,t,d}\right)}
    {\sum_t m^{\mathrm{KL}}_{b,t}},$$

    reduced exactly as ``source_conditioned_kl_raw`` is -- summed over $d_z$, masked by the same
    $(B, T)$ anchor support, divided by the same contributing-anchor count -- because the whole
    point is to subtract one from the other.

    **Why not call** :func:`~teb_vae.lag_attn_rws.nets.controls.source_null_kld` **directly.** That
    function exists, is what the task logs on ``val``, and is where the arm is *defined*; but it
    reduces a whole batch to one scalar, and what this table needs is the same quantity per sample.
    It is the situation ``prior_rate`` is in and it gets the same treatment: the expression is
    recomputed here over the identical re-encode, and ``tests/test_eval_controls.py`` pins the
    anchor-weighted batch reduction of this column equal to that function on the same inputs.

    The re-encode itself is **not** recomputed -- :func:`controls.source_null_forward_outputs` is
    called -- because that is where the arm's content lives: a zeroed stream is not a permutation
    of a real one, and every stage of the source pathway is nonlinear, so the gate and then the
    model's own key/value path must all be re-run. That function resolves the path off the model
    rather than naming a module, so this column measures whatever representation the lag attention
    actually reads. It draws no ``randn_like``, so it does not move the reparameterisation stream
    for the rest of the run.

    Args:
        model: The model the matched forward came from.
        outputs: That forward's dict.
        u_stream: The source stream it was given, $(B, T, c_u)$, at the **declared** width.
        kl_support: The KL anchor mask $(B, T)$ this run reduced its matched KL over.

    **The arm is also resolved by lag, and that costs no second re-encode.** The null forward
    already returns its own attention over the lags -- the query is the prior's mean, unchanged,
    but the keys are the null encode's -- so the same head-structured attribution the matched
    forward builds can be built here from tensors already in hand:

    $$\widetilde K^{\mathrm{null}}_{t,\ell}
    = \sum_m K^{(m),\mathrm{null}}_t\,\alpha^{(m),\mathrm{null}}_{t,\ell},
    \qquad \sum_\ell \widetilde K^{\mathrm{null}}_{t,\ell} = K^{\mathrm{null}}_t.$$

    The difference against the matched profile is the **clock-excess** attribution, and its lag sum
    is exactly ``coupling_minus_clock`` -- the scalar this whole arm exists to produce and the one
    ``clock_margin_min_nats`` gates. That identity is what makes a per-lag clock-excess a
    decomposition of the gated quantity rather than a second, differently-normalised reading of it,
    and :data:`IDENTITY_RESIDUAL_COLUMNS` measures it on every run rather than assuming it.

    **Why the difference is the right object and the null profile alone is not.** The availability
    staircase is a deterministic function of $t$ and is readable from the source state at *any*
    lag, so it enters the matched attribution at whatever lag the attention happens to sit on. Only
    subtracting an arm that carries the clock and no source content removes it, and no
    renormalisation of the matched profile can.

    ``te_analysis`` is invoked with ``head_structured=True``, which is what both causal cells' own
    forwards pass: under a flat latent the per-head split is an arbitrary slice of a shared latent
    and the product would be a diagnostic rather than a decomposition. Passing the model's own flag
    instead would let this arm and the matched forward disagree about what the attribution means.

    Returns:
        A mapping, because the arm produces four quantities off one re-encode and returning the
        scalar alone would mean paying for the encode again to get the rest:

        * ``kld_source_null`` -- the per-sample null KL $(B,)$, in nats per anchor;
        * ``lag_profile_null`` -- its per-lag decomposition $(B, L)$, which sums over lags to it;
        * ``lag_profile_null_untruncated`` -- the same over the anchors whose lag support is
          complete, the partner of ``lag_profile_untruncated``;
        * ``lag_map_null_identity_max_abs`` -- the worst per-anchor violation of the sum above.
    """
    nulled = controls.source_null_forward_outputs(model, dict(outputs), u_stream)
    kld_btd = model.kld_tensor(
        mu_prior=outputs["mu_prior"],
        logvar_prior=outputs["logvar_prior"],
        mu_post=nulled["mu_post"],
        logvar_post=nulled["logvar_post"],
    )
    # The null arm's own attribution, from the null arm's own attention. Head-structured, matching
    # both cells' forwards, so the split is a decomposition rather than a slice of a shared latent.
    kld_per_t_null, lag_map_null, _ = model.te_analysis(
        kld_btd, nulled["attn_weights"], head_structured=True
    )

    # The identical masks the matched profiles are reduced over, rebuilt here rather than passed:
    # this function runs before the lag block of ``evaluate_batch`` and reordering the two to share
    # a local would put the cheap re-encode after the expensive decode for no gain. ``build_lag_mask``
    # is the MODEL's, which is the floored one the attention was actually computed under.
    seq_len = int(kl_support.shape[1])
    n_lags = int(nulled["attn_weights"].shape[-1])
    lag_validity = model.build_lag_mask(seq_len, device=kl_support.device)
    untruncated = untruncated_anchor_mask(
        seq_len, n_lags, device=kl_support.device
    ).to(kl_support.dtype)
    profile_null, _, _ = lag_profiles(lag_map_null, kl_support, lag_validity)

    return {
        "kld_source_null": _per_sample_mean(kld_btd.sum(dim=-1), kl_support),
        "lag_profile_null": profile_null,
        "lag_profile_null_untruncated": _per_sample_vector_mean(
            lag_map_null, kl_support * untruncated
        ),
        "lag_map_null_identity_max_abs": identity_residual_per_sample(
            lag_map_null, kld_per_t_null, kl_support
        ),
    }


@torch.no_grad()
def source_lag_warmth_per_sample(
    model: Any,
    outputs: Mapping[str, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    r"""The share of attention mass landing on lags where each source block is warm, per sample.

    $$\texttt{source\_lag\_warmth\_frac}_b \;=\;
    \frac{\sum_{a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}\;
          \mathrm{warm}\!\left(t_{b,a} - \ell\right)}
         {\sum_{a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}},$$

    read at the anchors the forward decoded, with $v$ their validity. Normalising by the mass
    actually present rather than by a row count is what keeps the value in $[0, 1]$ when rows have
    no admissible lag at all: the attention normalises such a row to zero, and zero over zero would
    otherwise be the answer.

    This is the readout that sizes the compromise the design makes on the source. Lag attention
    searches $L$ lags back from an anchor, into a region where much of the source is still inside
    its own warm-up, and the design keeps every source channel rather than gating them. So the
    residual is measured instead of resolved, and a **small** value here is the expected finding
    rather than a failure -- ``lag_attn_cfs/DESIGN.md`` section 8 is the record, and the emitted
    analysis says so, so a reader does not treat it as a fault.

    The model's own ``_source_lag_warmth`` is the definition and reduces a whole batch to two
    scalars; this opens the batch axis and nothing else, and the **mass-weighted** recombination of
    this column is pinned equal to it by ``tests/test_eval_metrics.py`` -- mass-weighted rather than
    anchor-weighted because the model's denominator is the attention mass of the whole batch, which
    is the weight each sample's own fraction enters that ratio with. The warm patterns are read off
    the model rather than
    rebuilt: they are constants of the resolved budget, and a second resolution of them is a second
    partition that two batches of one run could disagree about.

    Args:
        model: The net, for ``source_block_warm_st`` and ``source_block_warm_ph``.
        outputs: The forward's dict, carrying ``attn_weights``, ``anchor_index`` and
            ``anchor_valid``.
        dtype: The floating dtype the columns are accumulated in.

    Returns:
        ``{'source_lag_warmth_frac_st', 'source_lag_warmth_frac_ph'}``, each $(B,)$.
    """
    alpha = outputs["attn_weights"]                                  # (B, T, M, L)
    heads, lags = int(alpha.shape[2]), int(alpha.shape[3])
    device = alpha.device

    anchors = outputs["anchor_index"].to(torch.long)                 # (B, A)
    live = outputs["anchor_valid"].to(dtype)                         # (B, A)

    index = anchors[:, :, None, None].expand(-1, -1, heads, lags)
    # (B, A, M, L): the attention rows of exactly the anchors that were decoded.
    at_anchor = alpha.gather(1, index).to(dtype) * live[:, :, None, None]
    total = at_anchor.sum(dim=(1, 2, 3)).clamp_min(torch.finfo(dtype).tiny)

    # (B, A, L): lag l at anchor t reads source step t - l. Negative entries carry exactly zero
    # attention -- the lag mask forbids them -- so they are clamped for the lookup and then
    # excluded, rather than relied upon to be harmless.
    lag_steps = anchors[:, :, None] - torch.arange(lags, device=device)[None, None, :]
    readable = lag_steps >= 0
    safe = lag_steps.clamp(min=0)

    warmth: Dict[str, torch.Tensor] = {}
    for name, pattern in (
        ("st", model.source_block_warm_st),
        ("ph", model.source_block_warm_ph),
    ):
        warm = readable & pattern.to(device)[safe]                   # (B, A, L)
        warmth[f"source_lag_warmth_frac_{name}"] = (
            at_anchor * warm[:, :, None, :].to(dtype)
        ).sum(dim=(1, 2, 3)) / total
    return warmth


def target_block_membership(model: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r"""A $0/1$ vector over the **kept** channels: $1$ where the channel came from the first block.

    Built from the gate's keep-index, which is positional into the declared width, so the split
    follows the resolved budget instead of assuming the survivors are contiguous -- they are not.
    The boundary is the model's own ``TARGET_BLOCK_SPLIT`` rather than a literal here, so a run
    whose stored blocks were written at other widths cannot be split at this one's.

    Args:
        model: The net, for its target gate and its block boundary.
        device: Device to build the vector on.
        dtype: Floating dtype of the vector it will multiply.

    Returns:
        A $(C_{\mathrm{keep}},)$ tensor.
    """
    keep_index = (
        torch.arange(model.c_y, device=device)
        if model.target_gate is None
        else model.target_gate.keep_index.to(device)
    )
    return (keep_index < int(model.TARGET_BLOCK_SPLIT)).to(dtype)


@torch.no_grad()
def evaluate_batch(
    task: Any,
    batch: Any,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    perm_generator: Optional[torch.Generator] = None,
    mc_generator: Optional[torch.Generator] = None,
    retain: Sequence[str] = (),
) -> BatchReadout:
    r"""Run one batch through the model and reduce it to per-sample readouts.

    Four latent branches are **decoded and scored** against the same feature future, under one
    shared set of noise draws:

    * ``base`` -- the target-only prior $p(z_t \mid Y_{\le t})$.
    * ``full`` -- the source-conditioned posterior $q(z_t \mid Y_{\le t}, U_{\le t})$.
    * ``shuffled`` -- the posterior rebuilt from a *stranger's* source, the negative control that
      makes a nonzero KL mean something.
    * ``base_shuffled_mu`` -- the base forecast from a stranger's *prior*, which is the check
      that the prior latent is carrying the target state at all rather than the decoder having
      learned a recording-independent average.

    A **fifth** arm is this cell's own and is deliberately **not decoded**: ``kld_source_null``
    re-encodes a zeroed source stream and reports the KL that survives, which needs only
    $(\mu^{q,\mathrm{null}}, \ell^{q,\mathrm{null}})$ and the support that is already built.
    Decoding it would pay for a forecast nothing reads. ``coupling_minus_clock`` beside it is the
    part of the coupling readout attributable to source *variation*, and it is the one hazard the
    permutation control structurally cannot see -- the availability pattern is identical in every
    row, and no permutation of rows removes what every row shares.

    Three latent-free forecasts are scored beside the four -- persistence, climatology and the
    segment mean (:func:`baseline_forecasts`), all in **feature space** -- through the same loss
    function and the same mask, because a summed-$2940$-coefficient block score is a large number
    under any predictor and only a comparison says whether the model is good or merely
    arithmetically fine.

    **The forward is called densely, here and nowhere else**, at :data:`DENSE_ANCHOR_GEOMETRY`.
    The target and both masks are then built from the anchor set the forward *returned*, never from
    a second derivation of it.

    The bound-variance diagnostics travel here rather than being left to the trainer's log,
    because the evaluation is where they are load-bearing: the KL carries
    $(\mu^q - \mu^p)^2 / \sigma_p^2$, so a prior variance pinned on its lower clamp inflates the
    coupling readout by orders of magnitude while ``mean_logvar_full`` and ``mean_logvar_base``
    -- which are *decoder* variances -- look perfectly healthy.

    Args:
        task: The Lightning task wrapping the loaded net.
        batch: A batch already on the model's device.
        num_samples: Monte Carlo draws $K$.
        perm_generator: Generator seeding the derangement, so a run is reproducible.
        mc_generator: Generator for the Monte Carlo $\epsilon$, on the model's device.
        retain: Forward-output names to carry back whole on the readout, plus ``'target'`` for
            the gathered feature future and ``'up_raw'`` / ``'weight'`` for the source trace and
            the validity behind it. Empty by default: a forecast tensor is
            $(B, A_{\max}, H, C_{\mathrm{keep}})$, about $0.9$ MiB per sample, so retaining one is
            a decision a caller makes rather than a default it inherits.

    Returns:
        The batch's per-sample readouts.

    Raises:
        NoCrossGroupPartner: If the batch carries recording identifiers but no cross-recording
            pairing exists -- one recording holding more than half the batch. Callers running a
            whole loader test this with
            :func:`~teb_vae.lag_attn_rws.nets.controls.groups_can_derange` and exclude such a
            batch, counting the exclusion.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))

    y_st, y_ph, u_stream, target_features, weight = model_inputs(task, batch)
    # The one forward call in this module, and the one place the anchor geometry is chosen. Both
    # keywords are named rather than positional so an argument order change in the net cannot
    # silently swap a phase for a stride.
    anchor_phase, anchor_stride = DENSE_ANCHOR_GEOMETRY
    outputs = model(
        y_st, y_ph, u_stream, anchor_phase=anchor_phase, anchor_stride=anchor_stride
    )

    # Read off the forward rather than recomputed, for the reason ``causal_inputs.py`` gives for
    # building the anchor set inside ``forward``: a second computation could disagree, and the
    # disagreement would be a wrong number rather than an exception.
    anchors, anchor_valid = outputs["anchor_index"], outputs["anchor_valid"]
    target = model._build_forecast_target(target_features, anchors)
    # The forecast clock's pooled validity -- the identity object on the stored clock -- so every
    # readout in this pass is scored under exactly the mask the training objective used.
    mask, coverage = forecast_mask(
        model.scored_weight(weight),
        model.geometry,
        coverage_floor=model.coverage_floor,
        anchors=anchors,
        anchor_valid=anchor_valid,
    )
    kl_support = kl_mask(mask, model.geometry, anchors=anchors, anchor_valid=anchor_valid)

    branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "base": (outputs["mu_prior"], outputs["logvar_prior"]),
        "full": (outputs["mu_post"], outputs["logvar_post"]),
    }
    batch_size = int(y_st.shape[0])
    recordings = batch_recordings(batch, batch_size)
    n_control_pairs = 0
    n_same_recording_pairs = 0
    shuffled_kl_per_t: Optional[torch.Tensor] = None
    if batch_size >= 2:
        # Grouped by recording where the batch says which recording each sample came from: an
        # unshuffled loader over per-recording shards puts a segment next to its own recording's
        # neighbouring segment, and pairing those two is not "a stranger's source".
        #
        # ``anchors=`` is not optional here even though this call site reads only the permuted
        # posterior's two distribution parameters: without it the control decodes the contiguous
        # prefix $[0, T_{\rm valid})$ instead of the decoded set, which is a $(B, 285, H, C)$
        # forecast nothing asked for -- silent, because the two parameters it *is* read for are
        # $(B, T, d_z)$ either way.
        permuted = controls.perm_forward_outputs(
            model, outputs, generator=perm_generator, groups=recordings, anchors=anchors
        )
        index = permuted["perm_index"]
        branches["shuffled"] = (permuted["mu_post"], permuted["logvar_post"])
        # The same derangement for both controls, so "a stranger's source" and "a stranger's
        # prior" name the same stranger and the two numbers are comparable.
        branches["base_shuffled_mu"] = (
            outputs["mu_prior"][index],
            outputs["logvar_prior"][index],
        )
        # The KL a stranger's source moves the posterior by. Recomputed from the permuted
        # distribution parameters -- two of ``controls.RECOMPUTED_KEYS`` -- rather than read off
        # the permuted dict's ``kld_per_t``, which is a shallow copy of the *matched* value and
        # would report the true coupling under the control's name with nothing failing.
        shuffled_kl_per_t = model.kld_tensor(
            mu_prior=outputs["mu_prior"],
            logvar_prior=outputs["logvar_prior"],
            mu_post=permuted["mu_post"],
            logvar_post=permuted["logvar_post"],
        ).sum(dim=-1)
        if recordings is not None:
            # Counted off the permutation that actually ran rather than asserted from the way it
            # was drawn: a grouped draw that silently stopped grouping is invisible otherwise.
            n_control_pairs = batch_size
            n_same_recording_pairs = sum(
                1
                for position, partner in enumerate(index.tolist())
                if recordings[position] == recordings[partner]
            )

    scores, contributing = mc_predictive_block(
        model, branches, target, mask, anchors=anchors, likelihood=likelihood,
        num_samples=num_samples, generator=mc_generator,
        # Absent on a model built without the decoder's persistence residual, which is the state
        # the decoder is then also in; present, it is the forward's own tensor rather than a
        # second gather of the same target.
        persistence=outputs.get("persistence"),
    )

    # The training-path score: one latent draw, the same functions the objective uses. Reported
    # alongside the marginalised one so the cost of the Monte Carlo marginalisation is visible.
    training_block, _ = masked_raw_block_per_anchor(
        outputs["mu_full"], target, mask, likelihood=likelihood, logvar=outputs["logvar_full"]
    )
    training_base_block, _ = masked_raw_block_per_anchor(
        outputs["mu_base"], target, mask, likelihood=likelihood, logvar=outputs["logvar_base"]
    )

    kld_btd = model.kld_tensor(
        mu_prior=outputs["mu_prior"],
        logvar_prior=outputs["logvar_prior"],
        mu_post=outputs["mu_post"],
        logvar_post=outputs["logvar_post"],
    )
    delta_mu = outputs["mu_post"] - outputs["mu_prior"]

    # Block scores only. The objective's ``nll_*_sample`` companion -- a fixed
    # $/(H \cdot C_{\rm keep})$ rescale of the block score -- is deliberately **not** emitted here:
    # it divides by the block's full width whatever the mask dropped, so it under-reports on every
    # anchor with a masked forecast step, and it is a constant multiple of a column already on the
    # table. A per-coefficient figure that is honest about its denominator exists and is
    # ``sq_error_*`` beside it, which divides by the coefficients actually scored.
    columns: Dict[str, torch.Tensor] = {
        "nll_base_block": _per_sample_mean(training_base_block, contributing),
        "nll_full_block": _per_sample_mean(training_block, contributing),
        "source_conditioned_kl_raw": _per_sample_mean(outputs["kld_per_t"], kl_support),
        # The prior's scale rate, on the same support and in the same nats-per-anchor units as the
        # divergence above it, so the two are addable exactly as they are in the objective. The
        # per-dimension expression is repeated from ``nets/losses.py::masked_prior_rate`` rather
        # than imported, for the reason every readout here is recomputed: that function reduces to
        # one scalar over a whole batch, and what this table needs is the same quantity resolved
        # per sample. A test pins the two equal on the same inputs.
        #
        # It is reported whether or not the run's objective weighted it: a prior collapsing onto
        # its clamp is visible here in any checkpoint, and a column that appeared only for
        # anchored runs would be missing from exactly the runs it diagnoses.
        "prior_rate": _per_sample_mean(
            (0.5 * (outputs["logvar_prior"].exp() - 1.0 - outputs["logvar_prior"])).sum(dim=-1),
            kl_support,
        ),
        "mu_prior_rms": _per_sample_mean(
            (outputs["mu_prior"] ** 2).mean(dim=-1), kl_support
        ).sqrt(),
        # Unrooted, and rooted beside it. An RMS is the square root of a mean, so averaging
        # finished per-segment roots across a recording is biased **low** by Jensen -- in the
        # direction that flatters the model. The square is what the aggregation chain must carry;
        # the rooted column stays because it is the figure the trainer logs and the headline
        # quotes, and the two now sit beside each other rather than one standing for both.
        "delta_mu_sq": _per_sample_mean((delta_mu**2).mean(dim=-1), kl_support),
        # Summed over $d_z$ *before* the mean, so this is the size of the belief shift per step
        # rather than a per-coordinate figure.
        "mu_post_prior_gap_sq": _per_sample_mean((delta_mu**2).sum(dim=-1), kl_support),
    }
    columns["delta_mu_rms"] = columns["delta_mu_sq"].sqrt()
    columns["pred_gap"] = columns["nll_base_block"] - columns["nll_full_block"]
    for name, value in scores.items():
        columns[f"mc_nll_{name}_block"] = _per_sample_mean(value, contributing)
    if "mc_nll_base_block" in columns and "mc_nll_full_block" in columns:
        columns["mc_pred_gap"] = columns["mc_nll_base_block"] - columns["mc_nll_full_block"]
    if shuffled_kl_per_t is not None:
        columns["source_conditioned_kl_shuffled_raw"] = _per_sample_mean(
            shuffled_kl_per_t, kl_support
        )

    # The availability-clock arm. Unconditional -- unlike the permutation controls it needs no
    # second sample in the batch, because the null is a zeroed stream rather than a stranger's --
    # so ``coupling_minus_clock`` is on every row of every run.
    # One re-encode, four quantities: the scalar the clock verdict gates, its per-lag
    # decomposition, that profile's untruncated partner, and the residual of the identity binding
    # the first two. The three lag tensors are held for the vector block below.
    null_arm = source_null_kld_per_sample(model, outputs, u_stream, kl_support)
    columns["kld_source_null"] = null_arm["kld_source_null"]
    columns["lag_map_null_identity_max_abs"] = null_arm["lag_map_null_identity_max_abs"]
    columns["coupling_minus_clock"] = (
        columns["source_conditioned_kl_raw"] - columns["kld_source_null"]
    )

    # The three trivial forecasts, scored through the model's *own* loss function with the
    # identical mask at the identical anchors -- so a skill score is a comparison of predictors
    # rather than of scoring conventions. Their observation variance is fixed and stated; see
    # BASELINE_LOGVAR.
    baselines = baseline_forecasts(target_features, weight, model, anchors)
    baseline_logvar = torch.full(
        (), BASELINE_LOGVAR, dtype=target.dtype, device=target.device
    )
    for name, baseline_mu in baselines.items():
        baseline_block, _ = masked_raw_block_per_anchor(
            baseline_mu, target, mask, likelihood=likelihood, logvar=baseline_logvar
        )
        columns[f"nll_{name}_block"] = _per_sample_mean(baseline_block, contributing)

    # Point-forecast error, in the loader's z units and per *scored coefficient* rather than per
    # anchor. The squares stay unrooted here -- see :func:`masked_raw_error_sums`.
    point_forecasts: Dict[str, torch.Tensor] = {
        "base": outputs["mu_base"], "full": outputs["mu_full"], **baselines
    }
    for name, point_mu in point_forecasts.items():
        sums = masked_raw_error_sums(point_mu, target, mask)
        scored = sums["n_coefficients"].clamp_min(1.0)
        columns[f"sq_error_{name}"] = sums["sum_sq"] / scored
        if name in ("base", "full"):
            # Only the model branches: the baselines exist to normalise the squared error, and a
            # constant predictor's bias is its own definition rather than a finding.
            columns[f"abs_error_{name}"] = sums["sum_abs"] / scored
            columns[f"signed_error_{name}"] = sums["sum_residual"] / scored

    # How far apart the two forecasts are, per scored coefficient, unrooted for the same Jensen
    # reason as the latent quantities above. Distinct from ``pred_gap``, which is a difference of
    # *scores*: two forecasts can differ everywhere and score identically.
    columns["forecast_difference_sq"] = _per_sample_element_mean(
        (outputs["mu_full"] - outputs["mu_base"]) ** 2, mask
    )

    # ---------------------------------------------------------------------
    # The channel axis: one reduction, six readouts
    # ---------------------------------------------------------------------
    # Both branches' masked scores resolved per anchor and per surviving channel. Everything below
    # is a different reduction of this one pair -- the per-channel gap vector, the two stored-block
    # gaps, the three warm-up tertile gaps and their per-anchor rows -- which is what makes all six
    # partial sums of the ``pred_gap`` they are read beside rather than six unrelated numbers.
    base_by_channel = branch_channel_scores(
        outputs["mu_base"], outputs["logvar_base"], target, mask, likelihood=likelihood
    )
    full_by_channel = branch_channel_scores(
        outputs["mu_full"], outputs["logvar_full"], target, mask, likelihood=likelihood
    )
    gap_by_anchor_channel = base_by_channel - full_by_channel        # (B, A, C_keep)
    anchor_totals = contributing.sum(dim=1).clamp_min(1.0)           # (B,)
    gap_per_channel = gap_by_anchor_channel.sum(dim=1) / anchor_totals[:, None]

    first_block = target_block_membership(model, target.device, gap_per_channel.dtype)
    columns["pred_gap_st"] = (gap_per_channel * first_block).sum(dim=1)
    columns["pred_gap_ph"] = (gap_per_channel * (1.0 - first_block)).sum(dim=1)

    # The warm-up tertiles, cutting the channel axis by filter speed rather than by stored block --
    # which the block split cannot, since both blocks span nearly the same rebased range. The
    # assignment is the model's own resolved partition, not a second ranking of the same vector.
    tertile = model.warm_tertile_id.to(target.device)
    tertile_names = ("lo", "mid", "hi")
    for group, name in enumerate(tertile_names):
        selector = (tertile == group).to(gap_per_channel.dtype)
        columns[f"pred_gap_warm_{name}"] = (gap_per_channel * selector).sum(dim=1)

    # The two geometry guards. ``anchors_per_sample`` is counted off ``anchor_valid`` rather than
    # off the mask, deliberately: it must report the **decoded** set, so a batch whose ``weight``
    # is entirely zero still says which anchors the forward built rather than reporting the
    # geometry as having collapsed. ``target_warm_frac`` is resolved at construction and echoed;
    # see ``CausalFeatureForecastTarget._resolve_target_warm_frac`` for why recomputing it per
    # batch would be a tautology, and what the column is actually for.
    columns["anchors_per_sample"] = anchor_valid.to(gap_per_channel.dtype).sum(dim=1)
    columns["target_warm_frac"] = torch.full(
        (batch_size,),
        float(model.target_warm_frac),
        dtype=gap_per_channel.dtype,
        device=target.device,
    )
    columns.update(source_lag_warmth_per_sample(model, outputs, gap_per_channel.dtype))

    # The bound-variance diagnostics, on the same aggregation chain as everything else. The model
    # computes the batch-level versions of these inside ``compute_loss`` and the trainer logs
    # them; the evaluation needs them per sample, so it recomputes them from the same tensors
    # against the same margin rather than restating the rule.
    lo, hi = float(model.logvar_clamp[0]), float(model.logvar_clamp[1])
    floor_threshold = lo + LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    ceil_threshold = hi - LOGVAR_FLOOR_MARGIN_FRAC * (hi - lo)
    dtype = outputs["logvar_prior"].dtype
    columns["mean_logvar_prior"] = _per_sample_mean(
        outputs["logvar_prior"].mean(dim=-1), kl_support
    )
    columns["mean_logvar_post"] = _per_sample_mean(
        outputs["logvar_post"].mean(dim=-1), kl_support
    )
    columns["logvar_prior_floor_frac"] = _per_sample_mean(
        (outputs["logvar_prior"] <= floor_threshold).to(dtype).mean(dim=-1), kl_support
    )
    columns["mean_logvar_full"] = _per_sample_element_mean(outputs["logvar_full"], mask)
    # Both ends, separately, and never inferred from the mean: one mean is equally consistent with
    # a well-spread distribution and with half the mass pinned on each clamp. The two ends also
    # fail differently -- on the floor the decoder is over-confident and the squared term
    # explodes; on the ceiling it has given up and is predicting noise, which reads as a healthy
    # falling NLL while ``pred_gap`` goes to zero.
    columns["logvar_full_floor_frac"] = _per_sample_element_mean(
        (outputs["logvar_full"] <= floor_threshold).to(dtype), mask
    )
    columns["logvar_full_ceil_frac"] = _per_sample_element_mean(
        (outputs["logvar_full"] >= ceil_threshold).to(dtype), mask
    )

    # The two saturation fractions, in both framings. The model's own are flat means over *every*
    # element -- warm-up prefix and untrained tail included -- and this package's masked/unmasked
    # convention is the opposite of the sibling's: the log-variance fractions above are already
    # masked, and these two are the only readouts that need a masked recomputation beside the raw
    # one. They may legitimately disagree, which is the point of emitting both.
    saturated_mu = (outputs["mu_prior"].abs() >= SATURATION_FRAC * model.mu_scale).to(dtype)
    saturated_delta = (delta_mu.abs() >= SATURATION_FRAC * model.delta_mu_scale).to(dtype)
    columns["mu_prior_sat_frac_raw"] = saturated_mu.mean(dim=(1, 2))
    columns["mu_prior_sat_frac_masked"] = _per_sample_mean(
        saturated_mu.mean(dim=-1), kl_support
    )
    columns["delta_mu_sat_frac_raw"] = saturated_delta.mean(dim=(1, 2))
    columns["delta_mu_sat_frac_masked"] = _per_sample_mean(
        saturated_delta.mean(dim=-1), kl_support
    )

    # Latent and lag summaries over the KL's own anchor support, so the untrained tail does not
    # dilute them, and per sample so they travel the same aggregation chain the scalars do.
    seq_len = int(kl_support.shape[1])
    # The **model's** lag mask, not ``lag_attn``'s: this cell floors it with ``lag_floor``, and the
    # attention was computed under the floored one.
    lag_validity = model.build_lag_mask(seq_len, device=kl_support.device)
    lag_profile, lag_profile_corrected, lag_support = lag_profiles(
        outputs["source_kl_lag_map"], kl_support, lag_validity
    )

    # The attention, on the identical footing: the same anchor support, the same two denominators,
    # and a third profile over the anchors whose lag support is complete -- see
    # :func:`untruncated_anchor_mask` for why the correction alone is not enough where the support
    # is truncated, and why nothing here is truncated at the shipped floor.
    attention = outputs["attn_weights"]
    n_lags = int(attention.shape[-1])
    head_averaged_attention = attention.mean(dim=2)
    attention_profile, attention_profile_corrected, _ = lag_profiles(
        head_averaged_attention, kl_support, lag_validity
    )
    untruncated = untruncated_anchor_mask(
        seq_len, n_lags, device=kl_support.device
    ).to(kl_support.dtype)
    attention_profile_untruncated = _per_sample_vector_mean(
        head_averaged_attention, kl_support * untruncated
    )
    # The KL attribution restricted to the same anchor set.
    lag_profile_untruncated = _per_sample_vector_mean(
        outputs["source_kl_lag_map"], kl_support * untruncated
    )
    # Head-major flattening, so one trailing axis reaches the aggregation chain; the head count
    # travels beside it in the lag report and the reshape happens once, in the consumer.
    per_head_attention = _per_sample_vector_mean(
        attention.reshape(attention.shape[0], seq_len, -1), kl_support
    )
    per_head_entropy = _per_sample_vector_mean(attention_entropy(attention), kl_support)
    # The KL attribution BEFORE the sum over heads, on the same head-major layout. Summing this
    # over $m$ returns ``source_kl_lag_map`` exactly, so it refines the shipped decomposition
    # rather than restating it -- and it is not recoverable from the two vectors beside it: the
    # product of the per-head KL's anchor mean with the per-head attention's anchor mean differs
    # from the anchor mean of their product by the within-segment covariance of the two, which is
    # precisely the quantity a head that concentrates its KL where it also concentrates its
    # attention would show.
    per_head_lag_map = outputs["kld_per_t_per_head"].unsqueeze(-1) * attention
    per_head_lag_profile = _per_sample_vector_mean(
        per_head_lag_map.reshape(per_head_lag_map.shape[0], seq_len, -1), kl_support
    )

    # The two lag-structure identities, measured on this run rather than assumed from the model
    # tests. Both hold anchor by anchor -- the lag map sums over lags to $K_t$ because each head's
    # attention sums to one, and the per-head KL sums over heads to $K_t$ because the latent
    # groups are head-aligned -- and the one thing that would break either is attention dropout,
    # whose error is zero-mean per anchor. Hence a max over anchors: see
    # :func:`identity_residual_per_sample`.
    columns["lag_map_identity_max_abs"] = identity_residual_per_sample(
        outputs["source_kl_lag_map"], outputs["kld_per_t"], kl_support
    )
    columns["head_kl_identity_max_abs"] = identity_residual_per_sample(
        outputs["kld_per_t_per_head"], outputs["kld_per_t"], kl_support
    )

    # Attention entropy against the ceiling it can actually reach. That ceiling is per sample
    # rather than a constant because it is a mean over *this sample's own supported anchors* of
    # $\log\min(t+1, L)$ -- so the two numbers describe the same anchor set and their ratio is a
    # measurement rather than an approximation. $\log L$ is the other ceiling, and is a constant;
    # at this cell's floor the two coincide, which the ``attention`` analysis asserts rather than
    # assumes.
    columns["attention_entropy_nats"] = per_head_entropy.mean(dim=-1)
    columns["attention_entropy_attainable_nats"] = _per_sample_mean(
        attainable_lag_entropy(
            seq_len, n_lags, device=kl_support.device, dtype=kl_support.dtype
        ).expand(batch_size, -1),
        kl_support,
    )

    # The same numbers as the columns above, before the within-sample reduction. Both score pairs
    # travel, so the per-anchor table recombines into the per-sample one exactly rather than into
    # one of the two conventions -- and neither name has to stand for both.
    #
    # ``anchor_index`` travels with them because this anchor axis is a *gathered set*: a row's
    # position in it is not the decimated step it scores, so a table keyed on position alone could
    # not be joined against the trajectory axis, the event table or anything else in the run.
    per_anchor: Dict[str, torch.Tensor] = {
        "anchor_index": anchors,
        "contributing": contributing,
        "coverage": coverage,
        # Gathered at the decoded anchors rather than sliced to a prefix: the KL support is dense
        # $(B, T)$ because the latent tensors it gates are produced at every step, and the rows
        # this table keeps are the anchors the forward chose out of it.
        "kld_per_t": outputs["kld_per_t"].gather(1, anchors),
        "nll_base_block": training_base_block,
        "nll_full_block": training_block,
        "pred_gap": training_base_block - training_block,
        # The lag the KL attribution peaks at, per anchor. Meaningful only where the anchor is
        # supported, which is exactly the set of rows the table keeps.
        "argmax_lag": outputs["source_kl_lag_map"]
        .gather(1, anchors[:, :, None].expand(-1, -1, n_lags))
        .argmax(dim=-1),
    }
    # The two per-anchor lag maps, gathered at the same anchors as the scalars above so a row of
    # the per-anchor table and a row of the per-anchor vector sidecar describe one anchor. The
    # pooled attribution rather than the per-head one: an anchor selection by $K_t$ is a selection
    # on the pooled KL, and the head split of the selected anchors is a second question. The
    # attention travels beside it for the reason every lag readout carries both -- the attribution
    # inherits the prior-variance inflation the attention is immune to.
    lag_gather = anchors[:, :, None].expand(-1, -1, n_lags)
    per_anchor_vectors: Dict[str, torch.Tensor] = {
        "kl_lag_map": outputs["source_kl_lag_map"].gather(1, lag_gather),
        "attention_lag_map": head_averaged_attention.gather(1, lag_gather),
    }
    # The three tertile gaps per anchor, so the per-anchor table recombines into the per-sample
    # columns of the same name -- which is what ``report_seam.RECOMBINED_COLUMNS`` checks.
    for group, name in enumerate(tertile_names):
        selector = (tertile == group).to(gap_by_anchor_channel.dtype)
        per_anchor[f"pred_gap_warm_{name}"] = (gap_by_anchor_channel * selector).sum(dim=2)
    for name in ("base", "full"):
        if name in scores:
            per_anchor[f"mc_nll_{name}_block"] = scores[name]
    if "base" in scores and "full" in scores:
        per_anchor["mc_pred_gap"] = scores["base"] - scores["full"]

    # The observation model's calibration census, over the full branch's scored coefficients.
    # Empty under ``'mse'``: the decoder's log-variance head is never fitted there, so a
    # probability integral transform of its output would be arithmetic over an untrained tensor.
    calibration = (
        calibration_sums(
            outputs["mu_full"], outputs["logvar_full"], target, mask,
            logvar_clamp=model.logvar_clamp,
        )
        if likelihood == "gaussian_nll"
        else {}
    )

    # Built only when asked for: the names resolve against the forward's own outputs plus the
    # gathered future, so a retained array cannot be a differently assembled version of what was
    # scored.
    retained: Dict[str, torch.Tensor] = {}
    if retain:
        available: Dict[str, torch.Tensor] = dict(outputs)
        available["target"] = target
        # The two raw traces beside the forecast, for the event analyses and the diagnostic page.
        # ``up_raw`` is the only signal in this pipeline the model never sees in raw form -- the
        # source reaches it as scattering and phase channels -- so a contraction can be located
        # nowhere else; and ``weight`` is what says which decimated steps are real, since the
        # coefficients carry no gap sentinel of their own.
        available["weight"] = weight
        up_raw = batch_field(batch, "up")
        if isinstance(up_raw, torch.Tensor):
            available["up_raw"] = up_raw
        fhr_raw = batch_field(batch, "fhr")
        if isinstance(fhr_raw, torch.Tensor):
            available["fhr_raw"] = fhr_raw
        retained = {name: available[name] for name in retain if name in available}

    return BatchReadout(
        guids=batch_guids(batch, batch_size),
        columns=columns,
        n_anchors=contributing.sum(dim=1),
        kld_per_dim=_per_sample_vector_mean(kld_btd, kl_support),
        lag_profile=lag_profile,
        lag_profile_support_corrected=lag_profile_corrected,
        lag_profile_untruncated=lag_profile_untruncated,
        lag_profile_null=null_arm["lag_profile_null"],
        lag_profile_null_untruncated=null_arm["lag_profile_null_untruncated"],
        lag_profile_per_head=per_head_lag_profile,
        lag_support=lag_support,
        attention_profile=attention_profile,
        attention_profile_support_corrected=attention_profile_corrected,
        attention_profile_untruncated=attention_profile_untruncated,
        attention_profile_per_head=per_head_attention,
        attention_entropy_per_head=per_head_entropy,
        kld_per_head=_per_sample_vector_mean(outputs["kld_per_t_per_head"], kl_support),
        gap_per_channel=gap_per_channel,
        # Per channel and per scored (anchor, horizon-step) pair, so the mean over channels is
        # exactly the pooled ``sq_error_*`` column beside it and a band-level skill has a zero.
        sq_error_per_channel_base=(
            ((outputs["mu_base"] - target) ** 2 * mask[..., None]).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1.0)[:, None]
        ),
        sq_error_per_channel_full=(
            ((outputs["mu_full"] - target) ** 2 * mask[..., None]).sum(dim=(1, 2))
            / mask.sum(dim=(1, 2)).clamp_min(1.0)[:, None]
        ),
        calibration_sums=calibration,
        n_control_pairs=n_control_pairs,
        n_same_recording_pairs=n_same_recording_pairs,
        per_anchor=per_anchor,
        per_anchor_vectors=per_anchor_vectors,
        retained=retained,
        horizon_sums={
            f"{branch}_{statistic}": value
            for branch, (branch_mu, branch_logvar) in (
                ("base", (outputs["mu_base"], outputs["logvar_base"])),
                ("full", (outputs["mu_full"], outputs["logvar_full"])),
            )
            for statistic, value in {
                **horizon_residual_sums(branch_mu, branch_logvar, target, mask),
                **horizon_block_sums(
                    branch_mu, branch_logvar, target, mask, likelihood=likelihood
                ),
            }.items()
        },
    )


# =============================================================================
# Aggregation
# =============================================================================
@dataclass
class Aggregate:
    r"""Readouts aggregated per recording and then across recordings.

    Attributes:
        per_recording: Per-guid means of every column.
        overall: The mean across recordings of each column -- the headline numbers.
        n_samples: Segments seen.
        n_samples_without_anchors: Segments excluded for scoring no anchors at all. Reported
            rather than silently dropped: a run where this is large measured far less than its
            segment count suggests.
        kld_per_dim: Per-dimension KL, per recording then across recordings.
        kld_per_head: The same KL split across the attention heads; sums over heads to
            ``overall['source_conditioned_kl_raw']``.
        lag_profile: Raw per-lag KL attribution on the same chain; sums to
            ``overall['source_conditioned_kl_raw']``.
        lag_profile_support_corrected: The per-lag attribution divided by each lag's own
            contributing-anchor count. Does not sum to the total KL.
        lag_profile_untruncated: The per-lag attribution over the anchors whose lag support is
            complete -- every scored anchor at the shipped floor.
        lag_support: Contributing anchors per lag, averaged on the same chain -- the correction's
            denominator, in anchors per segment.
        lag_profile_null: The source-null arm's per-lag attribution on the same chain; sums
            to ``kld_source_null``, so the difference against ``lag_profile`` sums to
            ``coupling_minus_clock``.
        lag_profile_null_untruncated: The null attribution restricted to the anchors whose lag
            support is complete, the partner of ``lag_profile_untruncated``.
        lag_profile_per_head: The per-head KL attribution, head-major and $M \cdot L$ wide;
            reshaped in :func:`lag_summary` and nowhere else.
        attention_profile: Per-lag attention on the same chain.
        attention_profile_support_corrected: The same attention on each lag's own denominator.
        attention_profile_untruncated: The same attention over the anchors whose lag support is
            complete.
        attention_profile_per_head: Per-head attention per lag, flattened head-major to
            $M \cdot L$ entries.
        attention_entropy_per_head: Per-head attention entropy in nats, one value per head.
        gap_per_channel: The forecast gap per surviving target channel, on the same chain; sums
            over channels to ``overall['pred_gap']``.
        sq_error_per_channel_base: Per-channel squared error of the target-only branch.
        sq_error_per_channel_full: The same for the source-conditioned branch.
    """

    per_recording: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_samples_without_anchors: int = 0
    kld_per_dim: List[float] = field(default_factory=list)
    kld_per_head: List[float] = field(default_factory=list)
    lag_profile: List[float] = field(default_factory=list)
    lag_profile_support_corrected: List[float] = field(default_factory=list)
    lag_profile_untruncated: List[float] = field(default_factory=list)
    lag_profile_null: List[float] = field(default_factory=list)
    lag_profile_null_untruncated: List[float] = field(default_factory=list)
    lag_profile_per_head: List[float] = field(default_factory=list)
    lag_support: List[float] = field(default_factory=list)
    attention_profile: List[float] = field(default_factory=list)
    attention_profile_support_corrected: List[float] = field(default_factory=list)
    attention_profile_untruncated: List[float] = field(default_factory=list)
    attention_profile_per_head: List[float] = field(default_factory=list)
    attention_entropy_per_head: List[float] = field(default_factory=list)
    gap_per_channel: List[float] = field(default_factory=list)
    sq_error_per_channel_base: List[float] = field(default_factory=list)
    sq_error_per_channel_full: List[float] = field(default_factory=list)

    @property
    def n_recordings(self) -> int:
        """How many distinct recordings contributed."""
        return len(self.per_recording)


def aggregate_by_recording(readouts: Sequence[BatchReadout]) -> Aggregate:
    r"""Average each column within a recording, then across recordings.

    Not a flat mean over anchors or over segments. Consecutive anchors' $15$-step forecast
    windows overlap in $14$ of them at the dense evaluation geometry, so anchors within a
    recording are very far from independent; averaging over them and reporting the result as if it
    had that many samples behind it overstates the precision of every number here, and weights the
    headline toward whichever recordings happen to be longest.

    The vector readouts of :data:`VECTOR_READOUTS` travel the identical chain, including the
    zero-anchor exclusion. Averaging them per *batch* instead -- which is what a stack-and-mean
    over batches does -- weights each batch equally however many anchors or recordings it held,
    and the per-dimension KL then no longer sums to the headline KL it decomposes.

    Args:
        readouts: Per-batch readouts.

    Returns:
        The aggregate. Empty when ``readouts`` is empty.

    Raises:
        ValueError: If the readouts do not agree on their column names, which would silently
            average different quantities together.
    """
    aggregate = Aggregate()
    if not readouts:
        return aggregate

    names = list(readouts[0].columns)
    for readout in readouts[1:]:
        if list(readout.columns) != names:
            raise ValueError(
                f"batches produced different readout columns: {names} vs "
                f"{list(readout.columns)}. A batch too small to derange skips the permutation "
                f"controls, so a run whose last batch has one sample must drop that batch "
                f"rather than average an inconsistent set."
            )

    # Sums and counts per recording, so a recording split across several batches is one unit.
    sums: Dict[str, Dict[str, float]] = {}
    vector_sums: Dict[str, Dict[str, torch.Tensor]] = {}
    counts: Dict[str, int] = {}
    for readout in readouts:
        aggregate.n_samples += len(readout.guids)
        for position, guid in enumerate(readout.guids):
            # A segment that scored no anchors -- every anchor gapped or below the coverage
            # floor -- measured nothing. Its columns are not small, they are absent: the
            # per-sample mean divides by a denominator clamped to 1, so an empty numerator
            # reads as exactly 0.0. Averaging that in would pull a summed-2940-coefficient block
            # score (hundreds of nats) toward zero and shrink pred_gap, with no other symptom.
            if float(readout.n_anchors[position]) <= 0.0:
                aggregate.n_samples_without_anchors += 1
                continue
            bucket = sums.setdefault(guid, {name: 0.0 for name in names})
            counts[guid] = counts.get(guid, 0) + 1
            for name in names:
                bucket[name] += float(readout.columns[name][position])
            # The vectors take the identical route -- same exclusion, same per-recording
            # denominator -- which is what keeps a decomposition equal to the scalar it
            # decomposes.
            vectors = vector_sums.setdefault(guid, {})
            for name in VECTOR_READOUTS:
                row = getattr(readout, name)[position].detach().to(torch.float64)
                vectors[name] = row.clone() if name not in vectors else vectors[name] + row

    aggregate.per_recording = {
        guid: {name: total / counts[guid] for name, total in bucket.items()}
        for guid, bucket in sums.items()
    }
    if not aggregate.per_recording:
        # Every segment scored zero anchors, so the pass measured nothing. `overall` stays empty,
        # exactly as it does for an empty `readouts` above. Dividing by a denominator clamped to 1
        # instead would publish a 0.0 for every column, and those zeros are not neutral: the loss
        # comparisons would FAIL on `0.0 == 0.0`, the two clamp verdicts would PASS on a
        # log-variance nothing ever wrote, and `anchor_geometry_intact` would FAIL on a geometry
        # no forward ever ran. Absent lets each verdict reach its own "not measured" branch.
        return aggregate
    n_recordings = float(len(aggregate.per_recording))
    aggregate.overall = {
        name: sum(values[name] for values in aggregate.per_recording.values()) / n_recordings
        for name in names
    }

    for name in VECTOR_READOUTS:
        per_recording_vectors = [
            vectors[name] / float(counts[guid]) for guid, vectors in vector_sums.items()
        ]
        if not per_recording_vectors:
            continue
        across = torch.stack(per_recording_vectors, dim=0).mean(dim=0)
        setattr(aggregate, name, [float(value) for value in across])
    return aggregate


def latent_health(aggregate: Aggregate) -> Dict[str, Any]:
    """Summarise how much of the latent is carrying source information.

    Args:
        aggregate: The aggregated readouts.

    Returns:
        Active-dimension count and fraction against :data:`KLD_ACTIVE_EPS`, the latent width, the
        full per-dimension KL distribution, and the share of the total KL held by the single
        largest dimension -- the number that says "collapsed into one dimension" directly.
    """
    per_dim = list(aggregate.kld_per_dim)
    d_z = len(per_dim)
    active = [value for value in per_dim if value > KLD_ACTIVE_EPS]
    total = sum(per_dim)
    return {
        "d_z": d_z,
        "active_dims": len(active),
        "active_frac": (len(active) / d_z) if d_z else 0.0,
        "activity_threshold_nats": KLD_ACTIVE_EPS,
        "kl_total_nats": total,
        "top_dimension_share": (max(per_dim) / total) if per_dim and total > 0.0 else 0.0,
        "kld_per_dimension": per_dim,
    }


def _argmax_of(profile: Sequence[float]) -> Optional[int]:
    """The index of a profile's largest bin, or ``None`` when there is no profile.

    ``None`` rather than a fallback to some other profile's argmax: the profiles here answer
    different questions, and a field silently holding a neighbouring one's number is the failure
    the corrections in this module exist to prevent.

    Args:
        profile: One value per lag.

    Returns:
        The argmax index, or ``None`` for an empty profile.
    """
    return max(range(len(profile)), key=profile.__getitem__) if len(profile) else None


def _seconds_of(lag: Optional[int], delay_steps: int) -> Optional[float]:
    """The compensated seconds of a lag index, or ``None`` when there is no lag.

    Args:
        lag: The lag index, or ``None``.
        delay_steps: The causal input delay $\\delta$.

    Returns:
        $4(\\ell + \\delta)$ seconds, or ``None``.
    """
    return None if lag is None else float(lag_compensated_seconds(lag, delay_steps=delay_steps))


#: The per-sample identity residual columns, and the name each is reported under. Both are worst
#: cases over anchors, so the pass reports the worst over *samples* rather than a mean: an
#: identity that fails on one recording has failed, and the mean of a max is not a max.
IDENTITY_RESIDUAL_COLUMNS: Dict[str, str] = {
    "lag_map_identity_max_abs": "lag_map_sums_to_kl_max_abs_nats",
    "head_kl_identity_max_abs": "per_head_kl_sums_to_kl_max_abs_nats",
    # The same identity on the source-null arm. It is measured rather than inherited from the
    # matched one because it is a DIFFERENT attention: the null forward re-poses the posterior
    # against a zeroed source, so its attention over the lags is its own and its rows summing to
    # one is its own property. The clock-excess profile is the difference of the two attributions,
    # so a violation here would silently make that difference something other than the per-lag
    # decomposition of ``coupling_minus_clock``.
    "lag_map_null_identity_max_abs": "null_lag_map_sums_to_kl_max_abs_nats",
}


def worst_identity_residuals(readouts: Sequence[BatchReadout]) -> Dict[str, float]:
    """The largest per-anchor identity residual any sample of the pass produced.

    Args:
        readouts: Every scored batch's readouts.

    Returns:
        One value per identity, by its reported name. Empty when the pass scored nothing, which
        the sanity block reads as "could not be evaluated" rather than as "held".
    """
    worst: Dict[str, float] = {}
    for column, reported in IDENTITY_RESIDUAL_COLUMNS.items():
        values = [
            float(readout.columns[column].max())
            for readout in readouts
            if column in readout.columns and readout.columns[column].numel()
        ]
        if values:
            worst[reported] = max(values)
    return worst


def lag_summary(
    aggregate: Aggregate,
    *,
    delay_steps: int = 0,
    identity_residuals: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    r"""Report where in the past the source informed the future, in every form that differs.

    **Four argmaxes, not one**, because three separate biases pull the answer short and each is
    removed by a different denominator or a different anchor set: the raw attribution's, the
    support-corrected one's, the attention's, and the attention's over the anchors at which every
    lag exists. Where two of them disagree, the difference *is* the corresponding bias, which is
    why they travel together rather than one replacing the rest. At this cell's shipped floor all
    three biases are zero and the four coincide; that is a property of $F \ge L - 1$ rather than of
    the domain, and an arm can reopen it.

    The **per-head** structure travels here too. The posterior is head-structured, so $K_t$ splits
    additively across heads and each head attends over its own lags; head-averaging before
    profiling discards exactly what that structure exists to expose.

    **Every figure here is in stored-coefficient time**, and
    :data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT` travels in the emitted record
    rather than only in a document beside it: the coefficients come from a strictly one-sided bank
    whose composed per-channel group delay is the same order as the lag search itself, so a peak's
    position on this axis is not a physiological latency.

    Args:
        aggregate: The aggregated readouts.
        delay_steps: The causal input delay $\delta$ applied to the source channels, in **stored
            steps**. Zero on an unaligned run of this cell, whose source is warm-up **masked**
            rather than delayed -- ``target_delays`` and ``source_delays`` are the only two
            constructor keywords the model removes -- and kept as an argument so the lag report is
            one function across the grid. Under a channel alignment it is the largest shift any
            surviving source channel received, which is what a stored-coefficient axis is indexed
            by; the alignment's physical constant $\tau_{\mathrm{ref}}$ is a **different number**
            and is deliberately not taken here, because applying it would restate this axis as a
            physiological latency, which is the claim the group-delay caveat refuses. It travels
            beside these numbers as ``source_reference_delay_s`` in the run's summary.
        identity_residuals: The worst per-anchor residual of each structural identity over the
            whole pass, by name. Recorded here rather than recomputed downstream because it is a
            maximum over samples and the aggregation chain reports means.

    Returns:
        The four argmaxes with their compensated seconds, the profiles themselves, the per-lag
        anchor counts behind the correction, the per-head split, the attention entropies, the
        identity residuals and the group-delay caveat. Empty when no lag profile was collected.
    """
    if not aggregate.lag_profile:
        return {}
    n_lags = len(aggregate.lag_profile)
    per_head_kl = list(aggregate.kld_per_head)
    num_heads = len(per_head_kl)
    kl_argmax = int(_argmax_of(aggregate.lag_profile) or 0)
    corrected = aggregate.lag_profile_support_corrected
    corrected_argmax = _argmax_of(corrected)
    untruncated_argmax = _argmax_of(aggregate.lag_profile_untruncated)
    attention_argmax = int(_argmax_of(aggregate.attention_profile) or 0)
    attention_corrected_argmax = _argmax_of(aggregate.attention_profile_support_corrected)
    attention_untruncated_argmax = _argmax_of(aggregate.attention_profile_untruncated)
    # Head-major, and reshaped exactly once -- here -- so no consumer has to know the layout.
    # An M*L flat list whose length does not factor is a mis-assembled profile rather than a
    # short one, so it is dropped whole rather than reshaped into a plausible wrong answer.
    flat_per_head = list(aggregate.attention_profile_per_head)
    per_head_profiles = (
        [flat_per_head[head * n_lags:(head + 1) * n_lags] for head in range(num_heads)]
        if num_heads and len(flat_per_head) == num_heads * n_lags
        else []
    )
    # The KL attribution on the same layout and through the same guard. A second reshape rather
    # than a shared helper because there are exactly two of them and the guard is three lines; a
    # third consumer is what would earn the helper.
    flat_kl_per_head = list(aggregate.lag_profile_per_head)
    kl_per_head_profiles = (
        [flat_kl_per_head[head * n_lags:(head + 1) * n_lags] for head in range(num_heads)]
        if num_heads and len(flat_kl_per_head) == num_heads * n_lags
        else []
    )
    # The clock-excess attribution: the matched profile less the source-null one, bin by bin.
    # **Signed, and published raw.** A lag at which the null exceeds the matched arm carries no
    # clock-exceeding coupling, and rectifying here would make the published vector stop summing
    # to ``coupling_minus_clock`` -- which is the one property that makes it a decomposition of
    # the gated scalar rather than a differently normalised profile. Whoever rectifies states the
    # rule and reports what it clipped.
    null_profile = list(aggregate.lag_profile_null)
    clock_excess = (
        [float(matched) - float(null) for matched, null in zip(aggregate.lag_profile, null_profile)]
        if len(null_profile) == n_lags
        else []
    )
    # The argmax of the POSITIVE part, because an argmax over a signed vector is a lag where the
    # excess is least negative wherever the profile is everywhere negative -- a bin, not a finding.
    clock_excess_argmax = _argmax_of([max(value, 0.0) for value in clock_excess])
    return {
        "delay_steps": int(delay_steps),
        # The source channels are masked individually and the maximum is what the model reports,
        # so every lag above is an upper bound. The flag travels with the numbers rather than
        # being stated once elsewhere, because a lag quoted without it reads as exact.
        "source_delay_is_max_over_channels": True,
        # The caveat that makes this axis readable, carried in the record rather than in a
        # document beside it: a reader given only the lag figures would have no way to know that
        # the group delay is the same order as the search.
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "n_lags": n_lags,
        "num_heads": num_heads,
        "kl_argmax_lag_step": kl_argmax,
        "kl_lag_compensated_seconds": float(
            lag_compensated_seconds(kl_argmax, delay_steps=delay_steps)
        ),
        "kl_lag_original_sensor_seconds": float(
            lag_original_sensor_seconds(kl_argmax, delay_steps=delay_steps)
        ),
        "kl_argmax_lag_step_support_corrected": corrected_argmax,
        "kl_lag_compensated_seconds_support_corrected": (
            None
            if corrected_argmax is None
            else float(lag_compensated_seconds(corrected_argmax, delay_steps=delay_steps))
        ),
        "kl_argmax_lag_step_untruncated": untruncated_argmax,
        "kl_lag_compensated_seconds_untruncated": _seconds_of(untruncated_argmax, delay_steps),
        "attention_argmax_lag_step": attention_argmax,
        "attention_lag_compensated_seconds": float(
            lag_compensated_seconds(attention_argmax, delay_steps=delay_steps)
        ),
        "attention_argmax_lag_step_support_corrected": attention_corrected_argmax,
        "attention_lag_compensated_seconds_support_corrected": _seconds_of(
            attention_corrected_argmax, delay_steps
        ),
        "attention_argmax_lag_step_untruncated": attention_untruncated_argmax,
        "attention_lag_compensated_seconds_untruncated": _seconds_of(
            attention_untruncated_argmax, delay_steps
        ),
        "kl_lag_profile": list(aggregate.lag_profile),
        "kl_lag_profile_support_corrected": list(corrected),
        "kl_lag_profile_untruncated": list(aggregate.lag_profile_untruncated),
        # The source-null arm, resolved by lag. Sums over lags to ``kld_source_null``.
        "kl_lag_profile_null": null_profile,
        "kl_lag_profile_null_untruncated": list(aggregate.lag_profile_null_untruncated),
        # ... and the difference, which sums to ``coupling_minus_clock``. This is the only lag
        # profile in this record with the availability staircase removed: the staircase is a
        # deterministic function of $t$, readable from the source state at ANY lag, so it enters
        # the matched attribution wherever the attention happens to sit and no renormalisation of
        # the matched profile can take it out. Signed; see the comment where it is built.
        "kl_lag_profile_clock_excess": clock_excess,
        "kl_argmax_lag_step_clock_excess": clock_excess_argmax,
        "kl_lag_compensated_seconds_clock_excess": _seconds_of(
            clock_excess_argmax, delay_steps
        ),
        # The KL attribution before the sum over heads. Reshaped here and nowhere else, exactly as
        # the attention's per-head profile is.
        "kl_lag_profile_per_head": kl_per_head_profiles,
        # Anchors per segment behind each corrected bin. Emitted beside the profile so a reader
        # can see which lags were averaged over how much, and re-derive either profile from the
        # other rather than taking the correction on trust.
        "kl_lag_anchor_counts": list(aggregate.lag_support),
        # Labelled head-averaged rather than left as "the" attention profile: it is a mean over
        # four distributions that need not agree, and the per-head profiles beside it are what
        # say whether they do.
        "attention_lag_profile": list(aggregate.attention_profile),
        "attention_lag_profile_support_corrected": list(
            aggregate.attention_profile_support_corrected
        ),
        "attention_lag_profile_untruncated": list(aggregate.attention_profile_untruncated),
        "attention_lag_profile_per_head": per_head_profiles,
        "attention_entropy_per_head_nats": list(aggregate.attention_entropy_per_head),
        # The additive per-head split of the KL: the quantity a head-structured posterior exists
        # to make meaningful, and it sums over heads to the headline KL exactly.
        "kld_per_head": per_head_kl,
        "kld_per_head_total_nats": sum(per_head_kl),
        # The worst per-anchor residual of each identity over the whole pass. Absent rather than
        # zero when the pass did not measure them: a zero here reads as "checked and exact".
        "identity_residuals": dict(identity_residuals or {}),
    }


# =============================================================================
# Verdicts
# =============================================================================
#: Every verdict this evaluation reports, in reporting order, with whether its status is promoted
#: into ``summary.json``'s headline block.
#:
#: The registry exists so that the *order* and the *promotion* are one declaration rather than two
#: conventions maintained in two files. :func:`build_verdicts` emits in this order and refuses to
#: emit a verdict that is not on it, so a new criterion is a line here plus the branch that decides
#: it -- and a criterion whose name drifts fails immediately rather than silently disappearing from
#: the headline.
#:
#: **Ten here against the sibling's eight.** The eight keep the sibling's positions so that two
#: summaries of two cells of the grid line up row for row, and the two this cell alone can have are
#: appended rather than interleaved -- which is also the order ``report_seam.HEADLINE_VERDICTS``
#: already declares and a test pins the two equal.
VERDICT_REGISTRY: Tuple[Tuple[str, bool], ...] = (
    ("predictive_improvement", True),
    # Between the gain and the specificity criteria because it sits between them in strength, and
    # because the three are read as a triple. ``source_specificity`` asks
    # $D_{\rm full} < D_{\rm base} < D_{\rm shuffled}$, which implies both of its neighbours; this
    # one drops the base branch entirely and asks only whether the matched source beat a
    # stranger's. The combination FAIL / PASS / FAIL is a real state and not a contradiction: no
    # predictive gain, and still source-specific.
    ("source_margin_positive", True),
    ("source_specificity", True),
    ("prior_carries_target_state", True),
    ("latent_not_collapsed", True),
    ("prior_variance_not_pinned", True),
    ("decoder_variance_not_pinned", True),
    ("calibration_near_nominal", True),
    # The fourth member of the source-pathway family, and the one no other cell has. It is not a
    # tightening of ``source_specificity``: that control deranges rows, and the availability
    # pattern is identical in every row, so no permutation of rows can remove it.
    ("coupling_exceeds_availability_clock", True),
    # Last, because it is a guard rather than a result: it says the population every number above
    # was computed over is the one the configuration states.
    ("anchor_geometry_intact", True),
)

#: Reporting order, derived so the two cannot disagree.
VERDICT_ORDER: Tuple[str, ...] = tuple(name for name, _ in VERDICT_REGISTRY)

#: The subset promoted into the headline block. ``report_seam`` restates these names rather than
#: importing them -- it must stay importable without ``torch`` -- and a test pins the two equal.
PROMOTED_VERDICTS: Tuple[str, ...] = tuple(
    name for name, promoted in VERDICT_REGISTRY if promoted
)


@dataclass(frozen=True)
class Verdict:
    """One acceptance criterion, its status, and the numbers behind it.

    Attributes:
        name: Criterion identifier.
        status: ``'PASS'``, ``'FAIL'`` or ``'INCONCLUSIVE'``.
        criterion: The criterion in words, so the summary is readable without this source.
        detail: Why this status, in one sentence.
        values: The numbers the status was decided from.
    """

    name: str
    status: str
    criterion: str
    detail: str
    values: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-shaped dict of this verdict."""
        return {
            "name": self.name,
            "status": self.status,
            "criterion": self.criterion,
            "detail": self.detail,
            "values": dict(self.values),
        }


def _score(overall: Dict[str, float], name: str) -> Optional[float]:
    """Return a marginalised branch score, or ``None`` when the branch did not run."""
    value = overall.get(f"mc_nll_{name}_block")
    return None if value is None else float(value)


def _finite(overall: Mapping[str, float], name: str) -> Optional[float]:
    """Return one headline readout, or ``None`` when it is absent or not finite.

    ``None`` rather than a ``NaN`` reaching a comparison: every comparison against a ``NaN`` is
    false, so a missing measurement would silently read as a failed criterion instead of an
    unevaluated one.

    Args:
        overall: The aggregate's across-recording means.
        name: The column to read.

    Returns:
        The value, or ``None``.
    """
    value = overall.get(name)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def source_specificity_verdict(
    d_base: Optional[float], d_full: Optional[float], d_shuffled: Optional[float]
) -> Verdict:
    r"""Decide the specificity criterion, $D_{\rm full} < D_{\rm base} < D_{\rm shuffled}$.

    **Three losses and nothing else.** The KL is deliberately not a parameter here, and that is
    the content of the criterion rather than a simplification of it: the posterior sees the source,
    so it reacts to *any* source, and a stranger's is out of distribution for a posterior trained
    only on matched pairs -- which routinely moves it **more**. A healthy model therefore has
    $K_{\rm shuffled} > K_{\rm true}$, so a criterion that read the KL would fail exactly the
    models it should pass. The discriminating comparison lives in prediction space.

    Args:
        d_base: The target-only branch's marginalised block score, or ``None``.
        d_full: The source-conditioned branch's, or ``None``.
        d_shuffled: The stranger's-source branch's, or ``None`` when the control did not run.

    Returns:
        The verdict, ``INCONCLUSIVE`` when any of the three is missing -- a control that could not
        run and a control that failed are different facts.
    """
    criterion = "D_full < D_base < D_shuffled"
    if d_base is None or d_full is None or d_shuffled is None:
        return Verdict(
            "source_specificity", INCONCLUSIVE, criterion,
            "the permutation control did not run; it needs a batch of at least two samples.",
            {},
        )
    ordered = d_full < d_base < d_shuffled
    return Verdict(
        "source_specificity", PASS if ordered else FAIL, criterion,
        "a stranger's source is worse than no source, so the model uses *this* recording's "
        "source rather than reacting to any source at all."
        if ordered
        else "the ordering does not hold, so a nonzero KL cannot be read as source-specific "
             "coupling.",
        {
            "d_base": float(d_base), "d_full": float(d_full), "d_shuffled": float(d_shuffled),
            "shuffle_penalty": float(d_shuffled) - float(d_base),
        },
    )


def source_margin_verdict(
    d_full: Optional[float], d_shuffled: Optional[float]
) -> Verdict:
    r"""Decide the margin criterion, $D_{\rm shuffled} > D_{\rm full}$.

    **Two losses, and the base branch is deliberately not one of them.** Every other predictive
    criterion here is referenced against $D_{\rm base}$, so every one of them inherits whatever
    the target-only forecast is doing -- and a model whose latent geometry charges more for the
    source than the source delivers fails all of them while still reading *this* recording's
    source rather than any source. This criterion changes only the source: prior, decoder and
    latent geometry are identical between the two branches. So it is the one predictive comparison
    that survives a negative predictive gain, and its passing beside a failing
    ``source_specificity`` is a state to report rather than a contradiction to resolve.

    Args:
        d_full: The source-conditioned branch's marginalised block score, or ``None``.
        d_shuffled: The stranger's-source branch's, or ``None`` when the control did not run.

    Returns:
        The verdict, ``INCONCLUSIVE`` when either is missing.
    """
    criterion = "D_shuffled > D_full"
    if d_full is None or d_shuffled is None:
        return Verdict(
            "source_margin_positive", INCONCLUSIVE, criterion,
            "the permutation control did not run; it needs a batch of at least two samples.",
            {},
        )
    margin = float(d_shuffled) - float(d_full)
    return Verdict(
        "source_margin_positive", PASS if margin > 0.0 else FAIL, criterion,
        "a stranger's source forecasts this recording worse than its own does, so the source "
        "pathway is reading this recording rather than reacting to any source."
        if margin > 0.0
        else "a stranger's source forecasts this recording at least as well as its own, so "
             "nothing the source pathway carries is specific to this recording.",
        {
            "d_full": float(d_full), "d_shuffled": float(d_shuffled),
            "source_margin": margin,
        },
    )


def availability_clock_verdict(
    coupling: Optional[float],
    clock: Optional[float],
    *,
    margin_min_nats: Optional[float] = DEFAULT_CLOCK_MARGIN_MIN_NATS,
    interval: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Verdict:
    r"""Decide whether the coupling readout exceeds the source *availability clock*.

    $$\Delta_{\mathrm{clock}} = \texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null}$$

    is the part of the coupling readout attributable to source **variation**. The hazard it sizes
    is the one no other control in this pipeline can see: the availability pattern $m^u_{t,c}$ is a
    deterministic function of $t$, identical in every row of a batch, and it enters
    $q(z \mid Y, U)$ but not $p(z \mid Y)$ -- so the posterior can be pushed off the prior by the
    clock alone, with no source information in it. The permutation control deranges rows, and no
    permutation of rows removes something every row shares.

    **The threshold ships unset, and that is the point.** Under ``margin_min_nats = None`` the
    status is ``INCONCLUSIVE`` and the measurement is emitted anyway. A provisional number would
    otherwise decide a FAIL on the first production runs -- which is the run that is supposed to
    *measure* it -- and a gate whose threshold is a guess either fails a healthy model or passes a
    broken one, with nothing in the output saying which. What is **not** conditional on the
    threshold is the number: ``coupling_minus_clock_nats`` is a headline scalar whatever the key
    says, so the arm tables carry it from the first run and the threshold can be set from them.

    **The decision is on the interval's lower end, not on the point estimate**, which is why an
    unsupplied interval leaves the verdict unevaluated rather than decided: a difference measured
    over fourteen recordings can clear any margin on its mean while its interval crosses zero.

    One thing the record must say, because it weakens the claim in the model's favour and nothing
    else would surface it: zeroing floors no source *variation*, and the encoder's response to a
    flat trajectory is not literally the availability pattern's response, so $\Delta$ is a slightly
    **weaker** statement than "the clock alone".

    Args:
        coupling: The matched coupling readout in nats per anchor, or ``None``.
        clock: The source-null readout on the same support, or ``None``.
        margin_min_nats: The margin the difference's lower interval end must clear. ``None`` --
            the shipped value -- makes this INCONCLUSIVE and still reports the measurement.
        interval: ``(lo, hi)`` bootstrap interval over recordings for the difference, or ``None``
            when the analysis that produces it has not run.

    Returns:
        The verdict, always carrying the measured difference and whatever interval exists beside
        the threshold -- a reader is never handed only a status.
    """
    criterion = (
        "source_conditioned_kl_raw - kld_source_null >= "
        f"{'(unset)' if margin_min_nats is None else f'{margin_min_nats:g}'} nats/anchor, "
        "on the lower end of the bootstrap interval over recordings"
    )
    caveat = (
        "the null zeroes the source stream, which floors its variation but is not literally the "
        "availability pattern alone -- the encoder is nonlinear, so this difference is a slightly "
        "weaker statement than 'the coupling exceeds the clock'; see lag_attn_cfs/DESIGN.md "
        "section 8."
    )
    if coupling is None or clock is None:
        return Verdict(
            "coupling_exceeds_availability_clock", INCONCLUSIVE, criterion,
            "this run reported no source-null readout, so how much of the coupling is the "
            f"availability clock is unmeasured rather than small. {caveat}",
            {} if margin_min_nats is None else {"margin_min_nats": float(margin_min_nats)},
        )

    difference = float(coupling) - float(clock)
    values: Dict[str, float] = {
        "source_conditioned_kl_raw": float(coupling),
        "kld_source_null": float(clock),
        "coupling_minus_clock_nats": difference,
    }
    lower = None if interval is None else interval[0]
    upper = None if interval is None else interval[1]
    if lower is not None and math.isfinite(float(lower)):
        values["interval_lo"] = float(lower)
    if upper is not None and math.isfinite(float(upper)):
        values["interval_hi"] = float(upper)
    if margin_min_nats is not None:
        values["margin_min_nats"] = float(margin_min_nats)

    if margin_min_nats is None:
        return Verdict(
            "coupling_exceeds_availability_clock", INCONCLUSIVE, criterion,
            "eval_config.clock_margin_min_nats is unset, so this criterion has no threshold to "
            "decide against; the measured difference travels here and in the headline so the "
            f"threshold can be set from the observed spread rather than guessed. {caveat}",
            values,
        )
    if "interval_lo" not in values:
        return Verdict(
            "coupling_exceeds_availability_clock", INCONCLUSIVE, criterion,
            "a threshold is set but no bootstrap interval over recordings reached this verdict, "
            "and the criterion is stated on the interval's lower end: a difference can clear any "
            f"margin on its mean while its interval crosses zero. {caveat}",
            values,
        )
    clears = values["interval_lo"] >= float(margin_min_nats)
    return Verdict(
        "coupling_exceeds_availability_clock", PASS if clears else FAIL, criterion,
        "the coupling readout exceeds the source availability clock by more than the stated "
        f"margin across recordings, so it is measuring source variation. {caveat}"
        if clears
        else "the coupling readout does not clear the availability clock by the stated margin, so "
             "part of what it reports may be a deterministic function of time rather than "
             f"anything the source carries. {caveat}",
        values,
    )


def anchor_geometry_verdict(
    anchors_per_sample: Optional[float],
    target_warm_frac: Optional[float],
    *,
    expected_anchors_per_sample: Optional[int] = None,
) -> Verdict:
    r"""Decide whether the run measured the population its configuration describes.

    Two exact numbers, and both are structural rather than statistical:

    * ``anchors_per_sample`` must be $\texttt{anchor\_ceiling} - F$ -- $137$ at the shipped
      stored-clock geometry, less the forecast clock's largest advance on a ``physical`` arm --
      because the evaluation decodes densely. A different count means the forward ran at the
      *training* tiling, and every number in the run was then computed over a different population
      with nothing else in the summary saying so.
    * ``target_warm_frac`` must be exactly $1.0$. Below it the objective scored assumed
      pre-recording history as signal, on coefficients normalised with constants that excluded
      exactly that region -- with every shape correct. The model's constructor refuses the pairing
      that would produce it, so a value off $1.0$ means the checkpoint predates that refusal.

    Compared at :data:`GEOMETRY_EXACT_TOLERANCE` rather than with ``==``: both quantities are means
    of values identical on every sample, so the exact answer is reachable and the tolerance guards
    the float accumulation across recordings rather than admitting a real drift.

    Args:
        anchors_per_sample: The decoded anchor count, averaged per recording then across.
        target_warm_frac: The warm target fraction, on the same chain.
        expected_anchors_per_sample: What the checkpoint's own geometry says the count must be.
            ``None`` when no model was available -- an offline re-run -- which makes the count half
            of this criterion unevaluated rather than assumed.

    Returns:
        The verdict, carrying both measurements and the expectation.
    """
    criterion = (
        "anchors_per_sample == anchor_ceiling - warmup_period"
        f"{'' if expected_anchors_per_sample is None else f' ({int(expected_anchors_per_sample)})'}"
        " and target_warm_frac == 1.0"
    )
    values: Dict[str, float] = {}
    if expected_anchors_per_sample is not None:
        values["expected_anchors_per_sample"] = float(expected_anchors_per_sample)
    if anchors_per_sample is not None:
        values["anchors_per_sample"] = float(anchors_per_sample)
    if target_warm_frac is not None:
        values["target_warm_frac"] = float(target_warm_frac)

    if anchors_per_sample is None or target_warm_frac is None:
        return Verdict(
            "anchor_geometry_intact", INCONCLUSIVE, criterion,
            "this run reported no anchor count or no warm-target fraction, so whether it measured "
            "the population its configuration describes is unknown rather than fine.",
            values,
        )
    if expected_anchors_per_sample is None:
        return Verdict(
            "anchor_geometry_intact", INCONCLUSIVE, criterion,
            "no checkpoint geometry reached this verdict, so the decoded anchor count has nothing "
            "to be checked against; an offline re-run with no --checkpoint reaches here.",
            values,
        )

    anchors_ok = (
        abs(float(anchors_per_sample) - float(expected_anchors_per_sample))
        <= GEOMETRY_EXACT_TOLERANCE
    )
    warm_ok = abs(float(target_warm_frac) - 1.0) <= GEOMETRY_EXACT_TOLERANCE
    if anchors_ok and warm_ok:
        return Verdict(
            "anchor_geometry_intact", PASS, criterion,
            "the run decoded every valid anchor and every scored target coefficient was past its "
            "channel's warm-up, so the population is the one the configuration states.",
            values,
        )
    reasons = []
    if not anchors_ok:
        reasons.append(
            f"the run decoded {float(anchors_per_sample):g} anchors per sample against the "
            f"{int(expected_anchors_per_sample)} its geometry requires, so every number here was "
            f"computed over a different population"
        )
    if not warm_ok:
        reasons.append(
            f"target_warm_frac is {float(target_warm_frac):g} rather than 1.0, so some scored "
            f"coefficient lay inside its channel's warm-up and was assumed pre-recording history "
            f"rather than signal"
        )
    return Verdict("anchor_geometry_intact", FAIL, criterion, "; ".join(reasons) + ".", values)


def order_verdicts(verdicts: Sequence[Verdict]) -> List[Verdict]:
    """Return the verdicts in :data:`VERDICT_ORDER`, refusing anything unregistered.

    Args:
        verdicts: The verdicts a builder produced, in whatever order it produced them.

    Returns:
        The same verdicts, in registry order.

    Raises:
        ValueError: If a verdict is not on the registry, if one is duplicated, or if a registered
            verdict is missing. All three are the same failure seen from different sides: the
            summary's verdict list is what the acceptance gate and the arm tables read by
            position and by name, so a silent gap in it reads as a criterion that passed.
    """
    # Read off the registry rather than off VERDICT_ORDER, so the registry is the single
    # declaration even when a caller has replaced it.
    order = tuple(name for name, _ in VERDICT_REGISTRY)
    produced = [verdict.name for verdict in verdicts]
    unknown = sorted(set(produced) - set(order))
    if unknown:
        raise ValueError(
            f"verdict(s) {unknown} are not in VERDICT_REGISTRY, whose entries are "
            f"{list(order)}. Add the name there -- the registry decides reporting order "
            f"and headline promotion, and a verdict absent from it reaches neither."
        )
    duplicated = sorted({name for name in produced if produced.count(name) > 1})
    if duplicated:
        raise ValueError(f"verdict(s) {duplicated} were produced more than once.")
    missing = [name for name in order if name not in produced]
    if missing:
        raise ValueError(
            f"VERDICT_REGISTRY names verdict(s) {missing} that this run did not produce. A "
            f"criterion that cannot be evaluated is reported {INCONCLUSIVE}, never omitted."
        )
    by_name = {verdict.name: verdict for verdict in verdicts}
    return [by_name[name] for name in order]


class StaleCachedVerdicts(RuntimeError):
    """A reused collection's verdict block was produced under a different registry.

    Distinct from a provenance mismatch: the tables describe the right run and are intact. What
    has moved is the *contract* -- a criterion was added or renamed since the pass that wrote
    them -- so the numbers are reusable and the verdict list is not.
    """


def check_cached_verdicts(cached: Optional[Sequence[Mapping[str, Any]]]) -> None:
    """Refuse a reused verdict block that the current registry no longer describes.

    The offline re-run path reads a finished directory's collection record and reports its
    ``verdicts`` verbatim -- that is what makes ``--only <analysis>`` cheap, and it is correct for
    as long as the registry has not moved. When it has, nothing downstream notices:
    :func:`order_verdicts` guards the list a *fresh* pass builds and is never reached on this
    path, so a directory collected under nine criteria would be re-reported as a nine-criterion
    run under a pipeline that declares ten. A summary silently missing a criterion reads exactly
    like one where that criterion passed.

    Recomputing the missing entries instead was considered and rejected. Only some criteria are
    decidable from what a collection record keeps -- the two predictive ones are, the calibration
    census is not -- so a repair path would work for the criteria that happen to be cheap and
    fail for the rest, which is a worse failure than a refusal because it is a partial one.

    Args:
        cached: The reused record's verdict list, or ``None`` when it carried none.

    Raises:
        StaleCachedVerdicts: Naming what moved in each direction, and the one way to fix it.
    """
    if cached is None:
        return
    registered = [name for name, _ in VERDICT_REGISTRY]
    produced = [str(entry.get("name")) for entry in cached if isinstance(entry, Mapping)]
    missing = [name for name in registered if name not in produced]
    unknown = [name for name in produced if name not in registered]
    if not missing and not unknown:
        return
    raise StaleCachedVerdicts(
        f"the collected tables here carry verdicts {produced}, which is not what this pipeline "
        f"declares ({registered})."
        + (f" Missing: {missing}." if missing else "")
        + (f" No longer registered: {unknown}." if unknown else "")
        + " The tables predate a change to the acceptance criteria, so their numbers are still "
          "good but their verdict block is not, and reporting it would omit a criterion rather "
          "than fail it. Re-collect: delete the collection from this directory, or point "
          "--output-dir at a new one, and pass --checkpoint so the pass has a model to collect "
          "with."
    )


def build_verdicts(
    aggregate: Aggregate,
    *,
    prior_shuffle_min_nats: float = DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    min_active_dims: int = DEFAULT_MIN_ACTIVE_DIMS,
    pinned_variance_max_frac: float = DEFAULT_PINNED_VARIANCE_MAX_FRAC,
    coverage_tail_tolerance: float = DEFAULT_COVERAGE_TAIL_TOLERANCE,
    clock_margin_min_nats: Optional[float] = DEFAULT_CLOCK_MARGIN_MIN_NATS,
    clock_interval: Optional[Tuple[Optional[float], Optional[float]]] = None,
    expected_anchors_per_sample: Optional[int] = None,
    logvar_margin: Optional[float] = None,
    calibration: Optional[Mapping[str, Any]] = None,
) -> List[Verdict]:
    r"""Turn the aggregated readouts into the acceptance verdicts.

    The two predictive criteria are the model's own: $D_{\mathrm{full}} < D_{\mathrm{base}}$, and
    $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$. The two representation
    criteria check that the thing being measured is where it is claimed to be. The three variance
    criteria check that the *numbers* mean what they say.

    The last two are this cell's own. ``coupling_exceeds_availability_clock`` is the only criterion
    in the grid that can separate a coupling readout from a deterministic availability clock;
    ``anchor_geometry_intact`` says the population every number above was computed over is the one
    the configuration describes.

    Args:
        aggregate: The aggregated readouts.
        prior_shuffle_min_nats: Minimum degradation from a shuffled prior latent.
        min_active_dims: Minimum active latent dimensions.
        pinned_variance_max_frac: How much of a bounded variance may sit within the margin of one
            of its clamps before it counts as pinned there.
        coverage_tail_tolerance: Relative tolerance on the observed tail mass at each coverage
            level.
        clock_margin_min_nats: ``eval_config.clock_margin_min_nats``. Ships unset; see
            :func:`availability_clock_verdict`.
        clock_interval: The bootstrap interval over recordings for
            $\Delta_{\mathrm{clock}}$, which the ``source_null`` analysis produces. Passed in
            rather than computed here so that one interval exists per run rather than two that
            could disagree; absent, the clock verdict is INCONCLUSIVE and still reports the
            measured difference.
        expected_anchors_per_sample: $T_{\mathrm{valid}} - F$ from the checkpoint's own geometry.
        logvar_margin: How close to a clamp counts as on it, in nats -- the model's own
            :data:`~teb_vae.lag_attn_rws.nets.model.LOGVAR_FLOOR_MARGIN_FRAC` of its clamp range,
            carried into the verdict so the fraction it reports is readable without the model.
        calibration: What :func:`calibration_report` produced, or ``None`` under a likelihood with
            no observation variance to calibrate.

    Returns:
        The verdicts, in :data:`VERDICT_REGISTRY` order.

    Raises:
        ValueError: Via :func:`order_verdicts`, if what this function produced and what the
            registry declares have drifted apart in either direction.
    """
    overall = aggregate.overall
    base, full = _score(overall, "base"), _score(overall, "full")
    shuffled, base_shuffled = _score(overall, "shuffled"), _score(overall, "base_shuffled_mu")
    verdicts: List[Verdict] = []

    if base is None or full is None:
        verdicts.append(
            Verdict(
                "predictive_improvement", INCONCLUSIVE,
                "D_full < D_base",
                "no batch produced both a base and a full predictive score.",
                {},
            )
        )
    else:
        verdicts.append(
            Verdict(
                "predictive_improvement", PASS if full < base else FAIL,
                "D_full < D_base",
                "the source-conditioned forecast scores better than the target-only one."
                if full < base
                else "the source-conditioned forecast is no better than the target-only one, so "
                     "the source contributed nothing the target's own past did not already say.",
                {"d_base": base, "d_full": full, "pred_gap": base - full},
            )
        )

    verdicts.append(source_margin_verdict(full, shuffled))
    verdicts.append(source_specificity_verdict(base, full, shuffled))

    if base is None or base_shuffled is None:
        verdicts.append(
            Verdict(
                "prior_carries_target_state", INCONCLUSIVE,
                f"D_base(shuffled mu_p) - D_base >= {prior_shuffle_min_nats} nats/anchor",
                "the prior-shuffle control did not run; it needs a batch of at least two "
                "samples.",
                {},
            )
        )
    else:
        degradation = base_shuffled - base
        if degradation <= 0.0:
            status, detail = FAIL, (
                "a stranger's prior latent forecasts this recording as well as its own, so "
                "the prior is not carrying the target's predictive state and every readout "
                "built on that reading is unsupported."
            )
        elif degradation < float(prior_shuffle_min_nats):
            status, detail = INCONCLUSIVE, (
                "shuffling the prior latent costs something, but less than the stated margin; "
                "the margin is provisional and this number is what revises it."
            )
        else:
            status, detail = PASS, (
                "shuffling the prior latent badly damages the baseline forecast, so the prior "
                "carries recording-specific target state."
            )
        verdicts.append(
            Verdict(
                "prior_carries_target_state", status,
                f"D_base(shuffled mu_p) - D_base >= {prior_shuffle_min_nats} nats/anchor",
                detail,
                {
                    "d_base": base, "d_base_shuffled_mu": base_shuffled,
                    "degradation": degradation, "margin": float(prior_shuffle_min_nats),
                },
            )
        )

    health = latent_health(aggregate)
    if health["kl_total_nats"] <= _COLLAPSE_INCONCLUSIVE_KL:
        status, detail = INCONCLUSIVE, (
            "the total KL is indistinguishable from zero, so there is no information to be "
            "distributed over dimensions; this is an untrained or collapsed source pathway "
            "rather than a badly shaped latent."
        )
    elif int(health["active_dims"]) >= int(min_active_dims):
        status, detail = PASS, "the KL is spread over more than one or two latent dimensions."
    else:
        status, detail = FAIL, (
            "the KL has collapsed onto fewer dimensions than the stated minimum, so the "
            "coupling readout rests on almost no latent structure."
        )
    verdicts.append(
        Verdict(
            "latent_not_collapsed", status,
            f"active latent dimensions >= {min_active_dims}",
            detail,
            {
                "active_dims": float(health["active_dims"]),
                "d_z": float(health["d_z"]),
                "min_active_dims": float(min_active_dims),
                "top_dimension_share": float(health["top_dimension_share"]),
                "kl_total_nats": float(health["kl_total_nats"]),
            },
        )
    )

    verdicts.append(
        _pinned_variance_verdict(
            "prior_variance_not_pinned",
            "the prior log-variance",
            floor_frac=overall.get("logvar_prior_floor_frac"),
            ceil_frac=None,
            mean_logvar=overall.get("mean_logvar_prior"),
            max_frac=pinned_variance_max_frac,
            margin=logvar_margin,
            pinned_detail=(
                "the prior variance sits on its lower clamp, and the KL carries "
                "(mu_q - mu_p)^2 / sigma_p^2 -- so every coupling number this run reports is "
                "inflated by an arbitrary factor while the decoder-side variances look healthy."
            ),
            healthy_detail=(
                "the prior variance is off its clamp, so the KL's denominator is a fitted "
                "quantity rather than a bound."
            ),
        )
    )
    verdicts.append(
        _pinned_variance_verdict(
            "decoder_variance_not_pinned",
            "the decoder log-variance",
            floor_frac=overall.get("logvar_full_floor_frac"),
            ceil_frac=overall.get("logvar_full_ceil_frac"),
            mean_logvar=overall.get("mean_logvar_full"),
            max_frac=pinned_variance_max_frac,
            margin=logvar_margin,
            pinned_detail=(
                "the decoder variance sits on a clamp: on the floor it is over-confident and the "
                "NLL's squared term explodes, on the ceiling it has given up and is predicting "
                "noise -- which reads as a healthy falling NLL while pred_gap goes to zero."
            ),
            healthy_detail="the decoder variance is off both clamps.",
        )
    )
    verdicts.append(
        _calibration_verdict(calibration, tail_tolerance=coverage_tail_tolerance)
    )
    verdicts.append(
        availability_clock_verdict(
            _finite(overall, "source_conditioned_kl_raw"),
            _finite(overall, "kld_source_null"),
            margin_min_nats=clock_margin_min_nats,
            interval=clock_interval,
        )
    )
    verdicts.append(
        anchor_geometry_verdict(
            _finite(overall, "anchors_per_sample"),
            _finite(overall, "target_warm_frac"),
            expected_anchors_per_sample=expected_anchors_per_sample,
        )
    )
    return order_verdicts(verdicts)


def _pinned_variance_verdict(
    name: str,
    subject: str,
    *,
    floor_frac: Optional[float],
    ceil_frac: Optional[float],
    mean_logvar: Optional[float],
    max_frac: float,
    margin: Optional[float],
    pinned_detail: str,
    healthy_detail: str,
) -> Verdict:
    """Decide whether a bounded log-variance has pinned itself against a clamp.

    Args:
        name: The verdict's registry name.
        subject: What is being judged, for the criterion sentence.
        floor_frac: Fraction of the quantity within the margin of the lower clamp, or ``None``
            when the run did not report it.
        ceil_frac: The same at the upper clamp, or ``None`` where only the floor is watched -- the
            prior's ceiling is a wide prior, which is uninformative rather than wrong.
        mean_logvar: The mean, carried for context. Deliberately not the decision: a mean is
            equally consistent with a spread distribution and with half the mass on each clamp.
        max_frac: The fraction past which the quantity counts as pinned.
        margin: How close to a clamp counts as on it, in nats. Carried into the verdict because
            "pinned" is meaningless without it: the bound is a sigmoid, so nothing ever reaches
            the clamp exactly and the fraction is entirely a statement about this margin.
        pinned_detail: What a failure means, in one sentence.
        healthy_detail: What a pass means.

    Returns:
        The verdict, ``INCONCLUSIVE`` when the fraction is absent -- which is what an offline
        re-run of tables collected before these columns existed produces, and is not a pass.
    """
    criterion = f"{subject} within its clamp margin for < {max_frac:g} of the masked support"
    values: Dict[str, float] = {"max_frac": float(max_frac)}
    if margin is not None and math.isfinite(float(margin)):
        values["clamp_margin_nats"] = float(margin)
    if mean_logvar is not None and math.isfinite(float(mean_logvar)):
        values["mean_logvar"] = float(mean_logvar)
    if floor_frac is None or not math.isfinite(float(floor_frac)):
        return Verdict(
            name, INCONCLUSIVE, criterion,
            f"this run reported no clamp fraction for {subject}, so whether it is pinned is "
            f"unmeasured rather than fine.",
            values,
        )

    values["floor_frac"] = float(floor_frac)
    worst = float(floor_frac)
    if ceil_frac is not None and math.isfinite(float(ceil_frac)):
        values["ceil_frac"] = float(ceil_frac)
        worst = max(worst, float(ceil_frac))
    pinned = worst >= float(max_frac)
    return Verdict(
        name, FAIL if pinned else PASS, criterion,
        pinned_detail if pinned else healthy_detail,
        values,
    )


def _calibration_verdict(
    calibration: Optional[Mapping[str, Any]], *, tail_tolerance: float
) -> Verdict:
    r"""Decide whether the observation model's central coverage matches the erf nominals.

    Judged on the **tail** mass and relatively, because the three levels' nominal tails span two
    orders of magnitude: an absolute tolerance loose enough to admit sampling noise at one sigma
    would accept a fifty-fold error at three, which is the end where an over-confident variance
    shows up first.

    Args:
        calibration: The calibration report, or ``None``/empty when there was none.
        tail_tolerance: Relative tolerance on each level's tail mass.

    Returns:
        The verdict, carrying every level's nominal and observed coverage whatever the status.
    """
    criterion = (
        f"observed central coverage at 1/2/3 sigma within {tail_tolerance:g} relative tail error "
        f"of erf(k/sqrt(2))"
    )
    levels = list((calibration or {}).get("coverage") or [])
    values: Dict[str, float] = {"tail_tolerance": float(tail_tolerance)}
    if not levels:
        return Verdict(
            "calibration_near_nominal", INCONCLUSIVE, criterion,
            "this run scored no Gaussian observation model, so there is no predictive "
            "distribution to calibrate -- an 'mse' checkpoint reaches here.",
            values,
        )

    worst_level, worst_error = None, 0.0
    for record in levels:
        level = int(record.get("level_sigma", 0))
        observed = float(record.get("observed", float("nan")))
        nominal = float(record.get("nominal", float("nan")))
        values[f"observed_{level}_sigma"] = observed
        values[f"nominal_{level}_sigma"] = nominal
        nominal_tail = 1.0 - nominal
        if not (math.isfinite(observed) and nominal_tail > 0.0):
            continue
        error = abs((1.0 - observed) - nominal_tail) / nominal_tail
        if error > worst_error:
            worst_level, worst_error = level, error
    values["worst_relative_tail_error"] = float(worst_error)
    if worst_level is not None:
        values["worst_level_sigma"] = float(worst_level)

    within = worst_error <= float(tail_tolerance)
    return Verdict(
        "calibration_near_nominal", PASS if within else FAIL, criterion,
        "the learned observation variance covers the truth at close to its nominal rates, so the "
        "block NLL is a log density rather than an arbitrary score."
        if within
        else f"central coverage misses its nominal worst at {worst_level} sigma, so the learned "
             f"variance is not the spread of the residuals it is meant to describe.",
        values,
    )


# =============================================================================
# Top level
# =============================================================================
def expected_anchors_per_sample(model: Any) -> int:
    r"""The dense anchor count a run of this checkpoint must decode.

    $\texttt{anchor\_ceiling} - F$: the ceiling is $T_{\mathrm{valid}}$ less the forecast clock's
    largest advance, so a ``physical``-clock checkpoint expects fewer anchors than a stored-clock
    one of the same geometry -- and a guard that read ``geometry.t_valid`` would fail every one of
    its rows for a reason that is not a defect. Derived from the checkpoint's own geometry rather
    than stated, so a legitimate arm -- ``sweep_horizon_15``, ``sweep_floor_150``, a forecast-clock
    arm -- moves the expectation with the model instead of failing a guard written against the
    shipped one.

    Args:
        model: The rebuilt net.

    Returns:
        The count, $137$ at the shipped stored-clock geometry.
    """
    ceiling = getattr(model, "anchor_ceiling", None)
    if ceiling is None:
        ceiling = model.geometry.t_valid
    return int(ceiling) - int(model.warmup_period)


@torch.no_grad()
def evaluate(
    task: Any,
    loader: Any,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    max_batches: Optional[int] = None,
    perm_generator: Optional[torch.Generator] = None,
    mc_generator: Optional[torch.Generator] = None,
    delay_steps: int = 0,
    prior_shuffle_min_nats: float = DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    min_active_dims: int = DEFAULT_MIN_ACTIVE_DIMS,
    clock_margin_min_nats: Optional[float] = DEFAULT_CLOCK_MARGIN_MIN_NATS,
    retain: Sequence[str] = (),
    on_batch: Optional[Callable[[Any, BatchReadout], None]] = None,
) -> Dict[str, Any]:
    """Evaluate a loaded task over a dataloader and assemble the JSON-shaped results.

    Two kinds of batch are skipped rather than partially scored, because a partially scored batch
    produces a different set of columns and averaging an inconsistent set together is how a
    control quietly stops being reported without anything failing: a batch too small to derange,
    and a batch whose samples come from too few distinct recordings for any pairing to be with a
    stranger. Both are counted, in batches and in samples -- the second in particular removes a
    non-random slice, since it is the longest recordings that fill a batch on their own.

    Args:
        task: The Lightning task wrapping the loaded net, already in ``eval`` mode.
        loader: A dataloader over the evaluation shards.
        num_samples: Monte Carlo draws $K$.
        max_batches: Stop after this many scored batches; ``None`` means the whole loader.
        perm_generator: Generator seeding the derangements.
        mc_generator: Generator for the Monte Carlo $\\epsilon$, on the model's device.
        delay_steps: The causal input delay applied to the source channels, for the lag report.
            Zero on this cell; see :func:`lag_summary`.
        prior_shuffle_min_nats: Verdict margin for the prior-shuffle control.
        min_active_dims: Verdict threshold for latent collapse.
        clock_margin_min_nats: ``eval_config.clock_margin_min_nats``, unset by default.
        retain: Forward-output names to carry back on each readout, for a sink that keeps them.
        on_batch: Called with ``(batch, readout)`` for every *scored* batch, before the readout's
            per-anchor and retained tensors are released. This is the seam the durable tables are
            built through: the decoder pass over four branches is the dominant cost of a run, and
            a second loop to write a table would double it.

    Returns:
        A dict of readouts, latent health, the lag report, the observation model's calibration,
        the control accounting, per-recording means and the verdicts.
    """
    was_training = task.training
    task.eval()
    readouts: List[BatchReadout] = []
    skipped = 0
    batches_without_partner = 0
    samples_without_partner = 0
    try:
        for batch in loader:
            batch = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            size = batch_size_of(batch)
            if size < 2:
                skipped += 1
                continue
            recordings = batch_recordings(batch, size)
            # Tested before the forward, not caught after it: the check is a Counter over a list
            # of strings, and a batch that cannot be controlled must not cost a decoder pass.
            if recordings is not None and not controls.groups_can_derange(recordings):
                batches_without_partner += 1
                samples_without_partner += size
                continue
            readout = evaluate_batch(
                task, batch, num_samples=num_samples,
                perm_generator=perm_generator, mc_generator=mc_generator, retain=retain,
            )
            readouts.append(readout)
            if on_batch is not None:
                on_batch(batch, readout)
            # Released as soon as the sink has had them. The readouts are held for the whole
            # loader so the aggregation can run over all of them at once, and the per-anchor and
            # retained tensors are two to three orders of magnitude larger than the per-sample
            # columns beside them -- keeping those alive for a full split would cost gigabytes for
            # values nothing reads again.
            readout.per_anchor = {}
            readout.per_anchor_vectors = {}
            readout.retained = {}
            if max_batches is not None and len(readouts) >= int(max_batches):
                break
    finally:
        task.train(was_training)

    aggregate = aggregate_by_recording(readouts)
    # Summed over batches here rather than in a sink, because the calibration statistics are the
    # one block a verdict reads that is not on the aggregation chain: a probability-integral
    # transform is a statement about a distribution over coefficients, not the mean of a
    # per-recording quantity.
    calibration_totals: Dict[str, torch.Tensor] = {}
    for readout in readouts:
        for name, value in readout.calibration_sums.items():
            calibration_totals[name] = (
                value if name not in calibration_totals else calibration_totals[name] + value
            )
    model = task.orig_model
    clamp = getattr(model, "logvar_clamp", None)
    calibration = calibration_report(
        {name: value.detach().cpu() for name, value in calibration_totals.items()},
        logvar_clamp=clamp,
    )
    verdicts = build_verdicts(
        aggregate,
        prior_shuffle_min_nats=prior_shuffle_min_nats,
        min_active_dims=min_active_dims,
        clock_margin_min_nats=clock_margin_min_nats,
        # From the checkpoint's own geometry, so an arm that legitimately moves the floor or the
        # horizon moves the expectation with it rather than failing a guard written against the
        # shipped numbers.
        expected_anchors_per_sample=expected_anchors_per_sample(model),
        # The model's own margin, so the fraction the pinned-variance verdicts report is readable
        # from the summary without going back to the checkpoint for what "on the clamp" meant.
        logvar_margin=(
            None if clamp is None
            else LOGVAR_FLOOR_MARGIN_FRAC * (float(clamp[1]) - float(clamp[0]))
        ),
        calibration=calibration,
    )
    control_pairs = sum(readout.n_control_pairs for readout in readouts)
    same_recording_pairs = sum(readout.n_same_recording_pairs for readout in readouts)
    return {
        "n_batches": len(readouts),
        "n_batches_skipped_too_small": skipped,
        "n_samples": aggregate.n_samples,
        "n_samples_without_anchors": aggregate.n_samples_without_anchors,
        "n_recordings": aggregate.n_recordings,
        "num_mc_samples": int(num_samples),
        "likelihood": str(task.hparams.get("likelihood", "gaussian_nll")),
        # The geometry every number below was produced at, recorded rather than assumed: the
        # training stride sits beside it because a table read against the training CSV would
        # otherwise be unreadable -- $A_{\max}$ differs by a factor of $S$ between the two.
        "anchor_geometry": {
            "anchor_phase": DENSE_ANCHOR_GEOMETRY[0],
            "anchor_stride": DENSE_ANCHOR_GEOMETRY[1],
            "anchors_per_sample_expected": expected_anchors_per_sample(model),
            "training_stride": int(model.anchor_stride),
            "target_kept_width": int(model.decoder_out_channels),
            "block_width": int(model.horizon) * int(model.decoder_out_channels),
        },
        "units": NORMALISED_UNIT,
        "readouts": dict(aggregate.overall),
        "latent_health": latent_health(aggregate),
        "lag": lag_summary(
            aggregate,
            delay_steps=delay_steps,
            identity_residuals=worst_identity_residuals(readouts),
        ),
        # Empty under a likelihood with no observation variance to calibrate, which is a skip
        # rather than a failure -- and is what the calibration analysis reports as one.
        "calibration": calibration,
        "controls": {
            # Zero by construction wherever the batch carried identifiers, and reported anyway:
            # this number is the only evidence that the control is still a control.
            "same_recording_pairing_rate": (
                (same_recording_pairs / control_pairs) if control_pairs else None
            ),
            "n_control_pairs": control_pairs,
            "n_same_recording_pairs": same_recording_pairs,
            # A batch too concentrated to pair across recordings is dropped whole. Silently, it
            # would remove the longest recordings preferentially and shrink every readout that
            # depends on them, with nothing in the output saying so.
            "n_batches_excluded_no_cross_recording_partner": batches_without_partner,
            "n_samples_excluded_no_cross_recording_partner": samples_without_partner,
        },
        "per_recording": aggregate.per_recording,
        "verdicts": [verdict.as_dict() for verdict in verdicts],
    }
