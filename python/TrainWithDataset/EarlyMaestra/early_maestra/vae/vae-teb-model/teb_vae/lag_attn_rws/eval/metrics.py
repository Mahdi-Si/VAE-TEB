r"""The evaluation readouts, the Monte Carlo predictive scores, and the acceptance verdicts.

Three groups of quantity come out of a checkpoint.

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

**The baselines.** Persistence, climatology and the segment's own mean, scored through the same
loss function over the same mask. A summed-$H \cdot R$-sample block score is a large number under every
predictor -- its scale is set by the block, not by the model -- so it is only readable against
predictors that know nothing. Their observation variance is fixed at $\sigma = 1$ in the loader's
$z$ units and stated, because under a Gaussian likelihood a point predictor has no variance of its
own and the whole skill score would otherwise be decided by an unstated choice.

**The bound diagnostics.** Both log-variances are smoothly bounded, and a bound that is always
active is a hyperparameter that has silently replaced a fitted quantity. The prior's floor is the
one that matters most and is the hardest to see: the KL carries
$(\mu^q - \mu^p)^2 / \sigma_p^2$, so a prior variance pinned on its lower clamp multiplies every
coupling number by an arbitrary factor while the *decoder* variances -- which are what a reader
looks at -- stay perfectly healthy. The fractions are recomputed here per sample, against the
model's own margin, and raised to a failable verdict rather than left in a training log.

**The calibration.** The decoder's learned $\sigma^2$ is a claim about how wrong the forecast is,
and the whole block NLL is a log density only if that claim holds. It is checked over the raw
samples themselves -- probability-integral transform, central coverage against the exact erf
nominals, CRPS, and the gain over the best single constant variance fitted to the very residuals
being scored -- by streaming sums, because a real split holds $10^9$ raw samples and none of them
is retained.

**The verdicts.** Each is ``PASS``, ``FAIL`` or ``INCONCLUSIVE``, never a bare boolean, and each
carries the numbers that produced it. A label with no numbers behind it is a claim a reader
cannot check.

One aggregation decision runs through all of it: **quantities are averaged per recording, then
across recordings.** Anchors are not independent samples of anything -- consecutive anchors'
forecast windows overlap in $H - 1$ of their $H$ horizon steps, and a single long recording holds
hundreds of them -- so a flat anchor mean weights recordings by their length and reports an
effective sample size far larger than the data supports. That chain -- support-weighted within a
segment, unweighted over a recording's segments, unweighted across recordings -- applies to the
*vector* readouts too, not only the scalars: a per-batch mean would weight batches equally
regardless of how many anchors or recordings each held, and the per-dimension KL summed over
dimensions would then no longer equal the headline KL it is a decomposition of.

**One pass produces all of it.** The decoder pass over four branches at $K$ draws is the dominant
cost of a run, so :func:`evaluate` takes an ``on_batch`` sink and hands it each batch's readouts
*before* their per-anchor and retained tensors are released. That is the seam the durable tables
are built through: a second loop to write them would double the only expensive part of an
evaluation, and a table assembled from a different forward would not be a table of these numbers.

Two lag quantities are reported side by side, and they answer different questions. The **raw**
attribution divides every lag bin by the same anchor support, so it keeps summing to $\bar K$ and
is the decomposition the identity test pins. But lag $\ell$ is causally valid only at anchors
$t \ge \ell$, so over the trained range the long lags are averaged over fewer anchors than the
short ones and the raw profile is biased toward short lags. The **support-corrected** profile
divides each bin by its own contributing-anchor count and is the one to read for "where in the
past did the source inform the future"; it does not sum to $\bar K$, which is why it is emitted
beside the raw attribution rather than in place of it.

The **attention** is read on the same footing and with two things the KL attribution does not
need. It is kept **per head**, because the posterior is head-structured and averaging the heads
before profiling discards exactly what that structure exists to expose -- four heads at four
delays and one head attending everywhere produce the same head-averaged profile. And it carries a
third profile restricted to the anchors at which *every* lag exists: an attention row is
renormalised per anchor, so at a truncated anchor the mass that had no long lag to reach was
pushed onto the short ones, and no per-lag denominator knows that happened. Its entropy is taken
per anchor and averaged afterwards, against a ceiling that is likewise per anchor -- $\log L$ is
unreachable wherever the lag support is truncated, and measured against it a model attending
uniformly over what it has reads as concentrated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn.nets.lag_report import (
    lag_compensated_seconds,
)
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
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

# The Welch layout, the taper and the one-sided weights live one layer down, because this module
# computes the cross-spectral sums and ``analyses/coherence`` turns them into readouts -- and an
# analysis may not import another. A second copy of the scaling convention in either place is how
# the two would come to disagree about what a spectrum is.
from teb_vae.lag_attn_rws.eval import spectra  # noqa: E402

#: Monte Carlo draws per anchor. The specification's starting value; more may be used for a
#: final analysis. $K = 1$ is one draw of the same estimator, NOT the training-path score: the
#: training path decodes the base branch at the prior MEAN under ``base_decode: mean`` (the
#: shipped setting) while this estimator samples every branch, so the two agree only on a branch
#: whose log-variance is pinned at $-\infty$ and only for the same $\epsilon$.
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

#: Below this total KL (nats per anchor) there is no coupling to be distributed over dimensions,
#: so a collapse verdict would be reporting the absence of a signal as a structural failure.
_COLLAPSE_INCONCLUSIVE_KL = 1e-6

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

#: The trivial forecast baselines, in reporting order. Every one is a *constant* over the anchor's
#: forecast block, which is what makes them trivial: they say nothing about the shape of the next
#: horizon, only about its level.
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

    Args:
        task: The Lightning task wrapping the loaded net.
        batch: A batch from the data module.

    Returns:
        ``(y_st, y_ph, u_stream, fhr_raw, weight)``.
    """
    y_st, y_ph = task._build_target_streams(batch)
    u_stream = task._build_source_stream(batch)
    fhr_raw, weight = task._build_raw_target(batch)
    return y_st, y_ph, u_stream, fhr_raw, weight


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
#: Unit labels. The ``normalised`` one is the honest fallback: without the loader's statistics the
#: numbers stay in $z$ units rather than being relabelled as something they are not.
BPM_UNIT = "bpm"
NORMALISED_UNIT = "normalised"

#: The epsilon ``denormalize_signal_data`` adds to the standard deviation before scaling. Restated
#: here only so the *sigma* inversion below uses the identical factor: a $\sigma$ converted with a
#: different scale from the mean it accompanies would draw a band that does not match its curve.
_DENORMALIZE_EPSILON = 1e-8


def fhr_normalization(
    normalization_stats: Optional[Dict[str, Any]]
) -> Optional[Tuple[float, float]]:
    """Return the loader's FHR z-scoring as ``(mean, scale)``, or ``None`` when it is unknown.

    Args:
        normalization_stats: The loader's statistics dict, as
            ``dataset.get_normalization_stats()`` returns it, or the ``{'fhr': {...}}`` subset the
            collection pass records. ``None`` and a dict without an ``'fhr'`` entry both mean the
            same thing: this run cannot express anything in bpm.

    Returns:
        ``(mean, std + eps)``, or ``None``.
    """
    block = (normalization_stats or {}).get("fhr")
    if not isinstance(block, Mapping):
        return None
    mean, std = block.get("mean"), block.get("std")
    if mean is None or std is None:
        return None
    return float(mean), float(std) + _DENORMALIZE_EPSILON


def to_bpm(
    values: Any, normalization_stats: Optional[Dict[str, Any]]
) -> Tuple[np.ndarray, str]:
    """Invert the loader's z-scoring on FHR *levels*.

    A level is affine in the normalisation: it scales **and** shifts. The conversion goes through
    ``denormalize_signal_data``, the repository's one supported z-to-bpm path, rather than through
    a second copy of the same two constants.

    Args:
        values: Values in loader units; anything ``numpy`` accepts.
        normalization_stats: The loader's statistics dict, or ``None``.

    Returns:
        ``(array, unit)``. Without statistics the values come back unchanged under the
        ``normalised`` label -- never relabelled bpm, which is the one failure mode a unit
        conversion has.
    """
    from train.graph_models_utils import denormalize_signal_data

    array = np.asarray(values, dtype=np.float64)
    if fhr_normalization(normalization_stats) is None:
        return array, NORMALISED_UNIT
    converted = denormalize_signal_data(
        torch.as_tensor(array), "fhr", dict(normalization_stats or {})
    )
    return np.asarray(converted.detach().cpu().numpy(), dtype=np.float64), BPM_UNIT


def sigma_to_bpm(
    values: Any, normalization_stats: Optional[Dict[str, Any]]
) -> Tuple[np.ndarray, str]:
    r"""Convert a *spread* from loader units to bpm: scale only, no offset.

    A standard deviation, a root-mean-square error and a mean signed error are all differences of
    levels, so the normalisation's additive term cancels and only its scale survives:

    $$\sigma_{\mathrm{bpm}} = \sigma_z \cdot (s + \varepsilon), \qquad
    \mu_{\mathrm{bpm}} = \mu_z \cdot (s + \varepsilon) + m.$$

    This function exists because the affine inversion is the plausible-looking wrong answer. An
    RMSE of $0.1$ z-units is $\approx 1$ bpm; put through the level conversion it becomes
    $\approx 141$ bpm, which is a physiologically reasonable number and therefore one nobody
    questions.

    Args:
        values: Spreads in loader units; anything ``numpy`` accepts.
        normalization_stats: The loader's statistics dict, or ``None``.

    Returns:
        ``(array, unit)``, with the ``normalised`` label when the statistics are unknown.
    """
    array = np.asarray(values, dtype=np.float64)
    resolved = fhr_normalization(normalization_stats)
    if resolved is None:
        return array, NORMALISED_UNIT
    return array * resolved[1], BPM_UNIT


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
        block_scores: Per-draw per-anchor block scores $(K, B, T_{\mathrm{valid}})$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.

    Returns:
        The marginalised per-anchor score $(B, T_{\mathrm{valid}})$.
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
    likelihood: str,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    r"""Score every branch's forecast under common random numbers.

    One $\epsilon$ is drawn per Monte Carlo replicate and reused by **every** branch, so two
    branches with identical latent parameters produce bitwise identical scores and the
    base-versus-full difference is a difference of predictions rather than of noise.

    Args:
        model: The net, for its shared decoder and its geometry.
        branches: ``{name: (mu, logvar)}`` latent parameters, each $(B, T, d_z)$. Every branch
            must share a shape; the first one's shape fixes the noise draw.
        target: The raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mask: The forecast mask $(B, T_{\mathrm{valid}}, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        num_samples: Monte Carlo draws $K$. At $K = 1$ the result is one sampled-latent score per
            branch. It equals the training path's own per-anchor score only for a branch the
            training path also SAMPLES and only under the same $\epsilon$; under
            ``base_decode: mean`` the training path decodes the base branch at $\mu^p$, which this
            estimator never does, so the base column here and ``nll_base_block`` are two different
            quantities at every $K$.
        generator: Generator for $\epsilon$, on the same device as the latent parameters. The
            estimator is the only place an evaluation *adds* randomness of its own, so it takes
            an explicit stream rather than the global one: two runs of a checkpoint must report
            the same numbers, and a global draw makes that a property of whatever else in the
            process happened to draw first. ``None`` falls back to the global generator.

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
    t_valid = model.geometry.t_valid
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
            forecast_mu, forecast_logvar = model.decoder(latent[:, :t_valid])
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
# Trivial forecast baselines
# =============================================================================
def baseline_forecasts(
    fhr_raw: torch.Tensor, weight: torch.Tensor, geometry: Any
) -> Dict[str, torch.Tensor]:
    r"""Build the three trivial forecasts, each constant over its anchor's block.

    They exist to answer the question a block NLL alone cannot: is the forecast *good*, or merely
    arithmetically fine? A summed-$H \cdot R$-sample log-density is a large number under any predictor,
    so the only readable form of it is a comparison against predictors that know nothing.

    * **persistence** -- hold the last *observed* raw sample forward across the whole block.
      "Observed" rather than "last", because a gap is stored as $0$ bpm, which after z-scoring is
      roughly $-11\sigma$: carrying that value forward would not measure persistence, it would
      measure the gap. The carry-forward is a running maximum over the valid step indices, so an
      anchor inside a gap reuses the last step before it.
    * **climatology** -- the normalisation's own centre, $\mu = 0$ in $z$ units. The predictor
      that has seen the population and nothing else.
    * **segment_mean** -- the mean of this segment's own observed raw samples. This stands in for
      the same-recording mean and is deliberately the stronger form: it is **not causal**, since
      it reads the segment's whole future, so a model that fails to beat it has learned nothing
      recording-specific that a constant could not say.

    Every forecast is returned at a *broadcastable* shape rather than expanded to
    $(B, T_{\mathrm{valid}}, H, R)$: half a megabyte per sample of repeated constants, when the
    scorer broadcasts them for free.

    Args:
        fhr_raw: The loader-normalized raw target $(B, L_{\mathrm{raw}})$.
        weight: The decimated validity signal $(B, T)$.
        geometry: The trimmed-grid geometry.

    Returns:
        Baseline name to its forecast mean, broadcastable over the forecast block.
    """
    decimation, t_valid = geometry.decimation, geometry.t_valid
    valid = weight >= VALID_THRESHOLD                                        # (B, T)

    # Index of the most recent valid decimated step at or before each step. ``cummax`` over the
    # step indices, with invalid steps sent to -1 so they never win the running maximum.
    steps = torch.arange(weight.shape[1], device=weight.device).expand_as(valid)
    last_valid = torch.cummax(torch.where(valid, steps, torch.full_like(steps, -1)), dim=1).values
    # An anchor with no valid step at or before it is fully masked by ``forecast_mask`` (its own
    # step is invalid), so the clamped fallback never reaches a scored term.
    endpoint = decimation * (last_valid[:, :t_valid].clamp_min(0) + 1) - 1   # (B, T_valid)
    persistence = fhr_raw.gather(1, endpoint)[:, :, None, None]

    # Raw-resolution validity: every raw sample of decimated step $k$ shares ``weight[k]``.
    raw_valid = valid.to(fhr_raw.dtype).repeat_interleave(decimation, dim=1)
    segment_mean = (
        (fhr_raw * raw_valid).sum(dim=1, keepdim=True)
        / raw_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    )[:, :, None, None]

    return {
        "persistence": persistence,
        "climatology": torch.zeros((), dtype=fhr_raw.dtype, device=fhr_raw.device),
        "segment_mean": segment_mean,
    }


def masked_raw_error_sums(
    mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Per-sample sums of the forecast residual, its magnitude and its square.

    $$e = \hat{x} - x, \qquad
    S^{1}_b = \sum m\,e, \quad S^{|1|}_b = \sum m\,|e|, \quad S^{2}_b = \sum m\,e^2,
    \quad n_b = R \sum_{t,\tau} m_{b,t,\tau}.$$

    Sums rather than finished statistics, and the reason is Jensen: an RMSE is the square root of
    a mean, and averaging finished per-sample RMSEs across a recording is biased **low** -- in the
    direction that flatters the model. So the squares accumulate unrooted here and the root is
    taken once, at the end of the aggregation chain.

    The residual is signed *forecast minus truth*, so a positive bias means the forecast runs
    high.

    Args:
        mu: Forecast mean, broadcastable to $(B, T_{\mathrm{valid}}, H, R)$.
        target: The raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.

    Returns:
        ``sum_residual``, ``sum_abs``, ``sum_sq`` and ``n_raw``, each $(B,)$. ``n_raw`` is the
        scored raw-sample count, which is the denominator every one of the three needs and which
        $H \cdot R$ over-states on any anchor with masked forecast steps.
    """
    residual = mu - target
    weights = mask[..., None]
    return {
        "sum_residual": (residual * weights).sum(dim=(1, 2, 3)),
        "sum_abs": (residual.abs() * weights).sum(dim=(1, 2, 3)),
        "sum_sq": ((residual**2) * weights).sum(dim=(1, 2, 3)),
        "n_raw": mask.sum(dim=(1, 2)) * float(target.shape[-1]),
    }


def masked_raw_block_per_horizon_step(
    mu: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    likelihood: str,
    logvar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""The per-anchor block score resolved by horizon step: summed over $r$, not over $\tau$.

    $$D_{b,t,\tau} = m_{b,t,\tau} \sum_{r} \ell\!\left(x_{b,t,\tau,r}, \hat{x}_{b,t,\tau,r}\right),
    \qquad \sum_\tau D_{b,t,\tau} = D_{b,t}.$$

    The identity on the right is the point, and it holds by construction rather than by
    arithmetic coincidence: this and
    :func:`~teb_vae.lag_attn_rws.nets.losses.masked_raw_block_per_anchor` reduce the *same*
    elementwise term over different axes. It is what makes the horizon curve readable as a
    decomposition of the headline score rather than as a second, differently-defined quantity.

    Args:
        mu: Forecast mean, broadcastable to $(B, T_{\mathrm{valid}}, H, R)$.
        target: The raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.
        likelihood: ``'mse'`` or ``'gaussian_nll'``.
        logvar: Forecast log-variance, broadcastable to the same shape.

    Returns:
        The per-horizon-step block score $(B, T_{\mathrm{valid}}, H)$.
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

    $$S^{D}_\tau = \sum_{b,t} D_{b,t,\tau}, \qquad n^{a}_\tau = \sum_{b,t} m_{b,t,\tau}.$$

    The denominator is **per $\tau$**, not the per-anchor contributing indicator. That indicator
    is an ``amax`` over $\tau$, so using it would divide a late horizon's numerator -- which the
    mask has already zeroed wherever that step falls in a gap -- by a count that includes those
    zeros, and the late horizons would read artificially good exactly where the signal is worst.

    Args:
        mu: Forecast mean $(B, T_{\mathrm{valid}}, H, R)$.
        logvar: Forecast log-variance, the same shape.
        target: The raw future target, the same shape.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.
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


# =============================================================================
# The tau-slice: a forecast block grid read as continuous time series, one per lead time
# =============================================================================
def tau_slices(block: torch.Tensor, *, warmup: int) -> torch.Tensor:
    r"""Read a $(B, T_{\mathrm{valid}}, H, R)$ forecast grid as $H$ continuous $4\,$Hz series.

    Fix a horizon step $\tau$ and concatenate over consecutive anchors. Because raw index
    $n(t, \tau, r) = R(t + 1) + R\tau + r$ advances by exactly $R$ when $t$ advances by one, and
    each anchor contributes exactly the $R$ samples $r \in [0, R)$, the concatenation is
    **contiguous, gap-free and non-overlapping**: slice $\tau$ is precisely the raw signal over
    $[R(w + 1 + \tau),\ R(T_{\mathrm{valid}} + 1 + \tau))$, sampled once each. At $\tau = H - 1$
    that upper end is exactly $L_{\mathrm{raw}}$, so the last slice closes on the record's end.

    This is the construction the whole coherence analysis rests on, and it is what makes a spectral
    reading of the forecast possible at all. Adjacent anchors' forecast *blocks* overlap in $H - 1$
    of their $H$ steps, so a spectrum taken over blocks would count almost every sample $H$
    times; and a single $H \cdot R$ block is $4H$ seconds, far too short to resolve the frequencies
    a fetal heart rate trace is read for. One $\tau$-slice is $A \cdot R$ samples -- $960$ s at the
    shipped geometry -- with every raw sample appearing exactly once.

    The trained-anchor prefix is dropped rather than masked: those anchors carry no loss term, so
    their forecasts are untrained output and including them would put untrained samples in the
    middle of a series whose spectrum is then attributed to the model.

    Args:
        block: $(B, T_{\mathrm{valid}}, H, R)$ -- the raw target, a branch's forecast mean, or any
            quantity on that grid.
        warmup: Leading anchors excluded from every loss, $w$.

    Returns:
        $(B, H, A R)$ with $A = T_{\mathrm{valid}} - w$, in strict time order along the last axis.
    """
    trained = block[:, int(warmup) :]
    # (B, A, H, R) -> (B, H, A, R) -> (B, H, A*R). The reshape is anchor-major and sub-sample-minor,
    # which is what puts the samples in time order; transposing the last two axes instead would
    # interleave the sub-samples of every anchor and produce a plausible-looking spectrum of
    # nothing.
    return trained.permute(0, 2, 1, 3).reshape(block.shape[0], block.shape[2], -1)


def tau_slice_window_validity(mask: torch.Tensor, *, layout: Any) -> torch.Tensor:
    r"""Which Welch windows of each $\tau$-slice contain no invalid sample.

    **A window touching a gap is dropped whole; nothing is ever interpolated.** ``events`` fills
    gaps before smoothing because a peak finder needs a continuous trace and the interpolant only
    has to not manufacture an edge. A spectral estimate cannot take that trade: the interpolant is
    a deterministic ramp whose own spectrum -- concentrated at low frequency, absent at high --
    would be attributed to the model, and in exactly the bands this analysis is read for.

    The test is an exact ``all()`` rather than a coverage threshold, and that is a consequence of
    the layout rather than a choice made here: the forecast mask is constant within a decimated
    step, and ``nperseg`` and the hop are integer multiples of $R$, so a window spans a whole
    number of anchors and is either entirely scored or not.

    Args:
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$ -- the pipeline's own scoring
            mask, which already folds the warm-up, anchor validity, forecast-step validity and the
            coverage floor.
        layout: The :class:`~teb_vae.lag_attn_rws.eval.spectra.SliceGeometry` for this run.

    Returns:
        $(B, H, W)$ of the mask's dtype, $1$ where every anchor the window spans is scored.
    """
    trained = mask[:, int(layout.warmup) :]
    # (B, A, H) -> (B, W, H, nperseg_steps): one entry per window, holding the anchors it spans.
    windows = trained.unfold(
        dimension=1, size=int(layout.nperseg_steps), step=int(layout.hop_steps)
    )
    return windows.amin(dim=-1).permute(0, 2, 1)


def source_tau_slices(up_raw: torch.Tensor, *, layout: Any) -> torch.Tensor:
    r"""Cut the raw UP trace into the same $\tau$-slice spans the forecast occupies.

    Slice $\tau$ covers raw $[R(w + 1 + \tau),\ R(T_{\mathrm{valid}} + 1 + \tau))$, so the source's
    slice is the *contemporaneous* uterine pressure -- the pressure during the window being
    forecast, not the pressure the model read. That is the point: the model conditions on UP only
    up to each anchor, so a forecast coherent with UP over its own horizon has anticipated the
    contraction rather than copied it.

    Args:
        up_raw: The raw UP trace $(B, L_{\mathrm{raw}})$, as the loader stores it (z-scored).
        layout: The :class:`~teb_vae.lag_attn_rws.eval.spectra.SliceGeometry` for this run.

    Returns:
        $(B, H, A R)$, aligned sample-for-sample with :func:`tau_slices` of the forecast grid.
    """
    stride = int(layout.raw_per_step)
    # Every start offset in steps of R, then the H that begin at R(w + 1 + tau).
    strided = up_raw.unfold(dimension=1, size=int(layout.n_samples), step=stride)
    first = int(layout.warmup) + 1
    return strided[:, first : first + int(layout.horizon)]


def _welch_segments(
    series: torch.Tensor, *, layout: Any, window: torch.Tensor
) -> torch.Tensor:
    r"""Cut a $\tau$-slice into overlapping Welch segments, detrended and tapered.

    Args:
        series: $(B, H, A R)$ from :func:`tau_slices`.
        layout: The slice layout.
        window: The periodic Hann window $(N,)$, on the series' device and dtype.

    Returns:
        $(B, H, W, N)$, each segment mean-removed and multiplied by the taper.
    """
    segments = series.unfold(dimension=-1, size=int(layout.nperseg), step=int(layout.hop))
    # Each segment's own mean, of that series alone -- the two series are detrended independently,
    # which is what makes the cross-spectrum the covariance of the two detrended segments.
    return (segments - segments.mean(dim=-1, keepdim=True)) * window


@torch.no_grad()
def cross_spectral_sums(
    target: torch.Tensor,
    mu_base: torch.Tensor,
    mu_full: torch.Tensor,
    up_raw: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    layout: Any,
) -> Dict[str, torch.Tensor]:
    r"""Accumulate the $\tau$-slice cross-spectral sufficient statistics for one batch.

    **Sufficient statistics, never ratios.** Every quantity returned is a sum over the windows this
    batch contributed, so a caller adds two batches', two segments' or two recordings' and ratios
    once at the end. That is not a stylistic preference:

    * $S_{ee} = S_{xx} + S_{yy} - 2\,\mathrm{Re}\,S_{xy}$ is linear in the three spectra, so the
      residual decomposition and the Parseval identity hold at *every* aggregation level only if
      the coherence, the gain and the phase all come from one aggregated triple.
    * Magnitude-squared coherence is exactly $1$ on a single window, for any two signals whatever.
      It carries no information until cross-spectra are averaged, so an implementation that ratioed
      here -- per batch, per segment, anywhere before the end -- would report perfect coherence
      everywhere and look entirely plausible doing it.

    **Float64 throughout, from model output that is float32.** Not to add precision the inputs do
    not have, but so the two sides of the Parseval identity -- one taken through the FFT, one
    accumulated in the time domain from the same float32 tensors -- differ by the estimator's
    correctness rather than by its arithmetic. At float32 they agree only to about $10^{-6}$, which
    is the order of the tolerance the identity is gated at, and a real normalisation bug would then
    be indistinguishable from round-off.

    **A window touching a gap contributes exactly zero and is not counted**; see
    :func:`tau_slice_window_validity` for why nothing is interpolated. Invalid samples are finite
    (a gap is stored as $0$ bpm, roughly $-11\sigma$ after z-scoring) so they cannot poison an
    accumulator through a ``NaN``; they are removed by the mask alone, which is why the mask is
    applied multiplicatively rather than by indexing.

    Args:
        target: The raw future target $(B, T_{\mathrm{valid}}, H, R)$.
        mu_base: The target-only forecast mean, the same shape.
        mu_full: The source-conditioned forecast mean, the same shape.
        up_raw: The raw UP trace $(B, L_{\mathrm{raw}})$, or ``None`` when the batch does not carry
            it -- in which case the four source-side statistics are absent rather than zero, so a
            reader cannot mistake "not collected" for "no coupling".
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.
        layout: The :class:`~teb_vae.lag_attn_rws.eval.spectra.SliceGeometry` for this run.

    Returns:
        Per-sample accumulators, each $(B, H, F)$ in float64 except the counts and the time-domain
        references, which are $(B, H)$:

        * ``sxx``, ``syy_base``, ``syy_full`` -- auto-spectra of the truth and the two branches.
        * ``sxy_base_re``/``_im``, ``sxy_full_re``/``_im`` -- $\overline{X}Y$, split into real and
          imaginary parts because the durable sidecar stores real arrays.
        * ``suu``, ``sux_truth_re``/``_im``, ``sux_base_re``/``_im``, ``sux_full_re``/``_im`` --
          the source side, absent when ``up_raw`` is ``None``.
        * ``n_windows``, ``n_windows_possible`` -- the honest denominator and what it would have
          been with no gaps.
        * ``ss_detrended_base``/``_full`` -- the time-domain residual sum of squares under the same
          taper and the same kept windows, which is the **exact** right-hand side of the Parseval
          identity.
        * ``ss_raw_base``/``_full`` -- the same without mean removal, for the loose magnitude
          reconciliation against the block scores. The difference between the two *is* the
          forecast's level error, which the spectrum by construction says nothing about.

        Empty when ``layout.n_windows`` is zero -- a geometry too short to hold one window measures
        nothing, and reporting zeros would be a measurement of that.
    """
    if int(layout.n_windows) <= 0:
        return {}

    device = target.device
    window = torch.as_tensor(
        spectra.welch_window(int(layout.nperseg)), dtype=torch.float64, device=device
    )
    weights = torch.as_tensor(
        spectra.one_sided_weights(int(layout.nperseg)), dtype=torch.float64, device=device
    )
    taper_power = float(window.pow(2).sum())
    denominator = float(layout.nperseg) * taper_power

    keep = tau_slice_window_validity(mask, layout=layout).to(torch.float64)  # (B, H, W)
    keep_bcast = keep[..., None]

    def segments_of(block: torch.Tensor) -> torch.Tensor:
        sliced = tau_slices(block.to(torch.float64), warmup=int(layout.warmup))
        return _welch_segments(sliced, layout=layout, window=window)

    truth = segments_of(target)
    base = segments_of(mu_base)
    full = segments_of(mu_full)

    fourier = {
        "truth": torch.fft.rfft(truth, dim=-1),
        "base": torch.fft.rfft(base, dim=-1),
        "full": torch.fft.rfft(full, dim=-1),
    }

    def accumulate(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        r"""$\sum_W \mathrm{keep} \cdot c_k \overline{L_k} R_k / (N U)$, summed over windows."""
        product = torch.conj(left) * right * weights
        return (product * keep_bcast).sum(dim=2) / denominator

    sums: Dict[str, torch.Tensor] = {
        "sxx": accumulate(fourier["truth"], fourier["truth"]).real,
        "syy_base": accumulate(fourier["base"], fourier["base"]).real,
        "syy_full": accumulate(fourier["full"], fourier["full"]).real,
        "n_windows": keep.sum(dim=2),
        "n_windows_possible": torch.full_like(keep.sum(dim=2), float(layout.n_windows)),
    }
    for name, spectrum in (("base", fourier["base"]), ("full", fourier["full"])):
        cross = accumulate(fourier["truth"], spectrum)
        sums[f"sxy_{name}_re"] = cross.real
        sums[f"sxy_{name}_im"] = cross.imag

    # The time-domain side of the Parseval identity, accumulated independently of the FFT over the
    # identical kept windows. `truth`/`base`/`full` are already detrended and tapered, so the
    # difference of two of them is the tapered, detrended residual -- which is exactly what the
    # left-hand side sums to, since removing each series' own mean removes the residual's.
    for name, segments in (("base", base), ("full", full)):
        detrended = segments - truth
        sums[f"ss_detrended_{name}"] = (
            (detrended.pow(2).sum(dim=-1) * keep).sum(dim=2) / taper_power
        )

    # The same without mean removal, for the magnitude check against the block scores. Re-cut from
    # the raw blocks rather than reconstructed, because the level is precisely what the detrended
    # form has discarded.
    raw_residual = tau_slices((mu_base - target).to(torch.float64), warmup=int(layout.warmup))
    raw_residual_full = tau_slices((mu_full - target).to(torch.float64), warmup=int(layout.warmup))
    for name, series in (("base", raw_residual), ("full", raw_residual_full)):
        segments = series.unfold(dimension=-1, size=int(layout.nperseg), step=int(layout.hop))
        tapered = (segments * window).pow(2).sum(dim=-1)
        sums[f"ss_raw_{name}"] = (tapered * keep).sum(dim=2) / taper_power

    if up_raw is not None:
        source = _welch_segments(
            source_tau_slices(up_raw.to(torch.float64), layout=layout),
            layout=layout,
            window=window,
        )
        source_fourier = torch.fft.rfft(source, dim=-1)
        sums["suu"] = accumulate(source_fourier, source_fourier).real
        for name in ("truth", "base", "full"):
            cross = accumulate(source_fourier, fourier[name])
            sums[f"sux_{name}_re"] = cross.real
            sums[f"sux_{name}_im"] = cross.imag

    return sums


# =============================================================================
# Per-batch evaluation
# =============================================================================
#: The vector readouts, by attribute name. Every one is per sample on a :class:`BatchReadout` and
#: a plain list on an :class:`Aggregate`, and they all travel the same aggregation chain, so the
#: chain is written once and driven from this tuple rather than repeated five times.
VECTOR_READOUTS: Tuple[str, ...] = (
    "kld_per_dim",
    "kld_per_head",
    "lag_profile",
    "lag_profile_support_corrected",
    "lag_profile_untruncated",
    "lag_support",
    "attention_profile",
    "attention_profile_support_corrected",
    "attention_profile_untruncated",
    "attention_profile_per_head",
    "attention_entropy_per_head",
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
            ``source_conditioned_kl_raw``. The model computes it every forward and nothing read
            it until now, which discarded exactly what the head structure exists to expose.
        lag_profile: Per-sample raw per-lag KL attribution, $(B, L)$; sums over lags to the
            sample's ``source_conditioned_kl_raw``.
        lag_profile_support_corrected: The same attribution divided by each lag's own
            contributing-anchor count rather than by the common anchor total, $(B, L)$.
        lag_profile_untruncated: The same attribution over the anchors at which **every** lag
            exists ($t \ge L - 1$), $(B, L)$. The attribution inherits the attention's
            per-anchor renormalisation, so it carries the same numerator bias the support
            correction cannot reach -- see ``attention_profile_untruncated`` below. This is the
            profile a per-cohort argmax claim rests on.
        lag_support: Contributing anchors per lag, $(B, L)$ -- the denominator above, carried so
            the correction can be checked and re-derived rather than trusted.
        attention_profile: Per-sample head-averaged attention per lag, $(B, L)$.
        attention_profile_support_corrected: The same attention divided by each lag's own
            contributing-anchor count, $(B, L)$.
        attention_profile_untruncated: The head-averaged attention over the anchors at which
            **every** lag exists ($t \ge L - 1$), $(B, L)$. The support correction fixes each
            bin's denominator; it cannot fix its numerator, because at a truncated anchor the
            probability mass that had nowhere to go among the long lags was renormalised onto
            the short ones. Restricting the anchor set is what removes that, at the cost of the
            anchors it drops -- so both travel.
        attention_profile_per_head: Per-head attention per lag, flattened head-major to
            $(B, M \cdot L)$. Flattened rather than kept $(B, M, L)$ so it travels the same
            one-trailing-axis aggregation chain every other vector readout does; the head count
            travels with it in the lag report, and the reshape is the consumer's one line.
        attention_entropy_per_head: Per-head entropy of the attention over lags, in nats,
            $(B, M)$ -- averaged over the anchors rather than taken of the averaged profile, so
            it is comparable with the per-anchor attainable ceiling.
        n_control_pairs: Samples paired against a *known* other recording by the permutation
            control. Zero when the batch carries no identifiers, which is not the same as a
            within-recording pairing rate of zero.
        n_same_recording_pairs: How many of those pairs landed inside their own recording. Zero
            by construction under a grouped derangement, and reported anyway: a control that has
            silently stopped being a control looks exactly like one that works.
        per_anchor: The same quantities *before* the within-sample reduction, each
            $(B, T_{\mathrm{valid}})$ -- what the per-anchor table is built from. Carried on the
            readout rather than recomputed because recomputing them means a second decoder pass,
            and released by :func:`evaluate` as soon as a sink has consumed them: they are two
            orders of magnitude larger than the per-sample columns and nothing needs them once
            the row is written.
        retained: Whole model tensors a caller asked to keep, by their forward-output name.
            Empty unless ``retain`` named them, because each is hundreds of kilobytes per sample.
        horizon_sums: Residual, log-variance and block-score sums resolved by horizon step, each
            $(H,)$, per branch. The $\tau$ axis lives inside an anchor, so it survives on neither
            table and cannot be recovered from either -- streaming it is what makes it available
            at all, and it is what the horizon-resolved skill and gap curves are built from.
        calibration_sums: The observation model's calibration accumulators over the *full*
            branch's raw samples -- see :func:`calibration_sums`. Empty under ``'mse'``, where the
            decoder's log-variance head is never trained and a probability-integral transform of
            its output would be arithmetic over an untrained tensor.
        spectral_sums: The $\tau$-slice cross-spectral sufficient statistics -- see
            :func:`cross_spectral_sums`. Per sample and at full frequency resolution, $(B, H, F)$,
            because the band collapse and the cohort pooling both need labels this layer does not
            have. **Unconditional**, unlike ``retained``: the coherence analysis must be able to
            run offline against a finished directory with no checkpoint, which is only possible if
            the pass always wrote them -- exactly as ``horizon_sums`` is always written. Empty only
            where the geometry cannot hold one Welch window.
    """

    guids: List[str]
    columns: Dict[str, torch.Tensor]
    n_anchors: torch.Tensor
    kld_per_dim: torch.Tensor
    lag_profile: torch.Tensor
    lag_profile_support_corrected: torch.Tensor
    lag_profile_untruncated: torch.Tensor
    lag_support: torch.Tensor
    attention_profile: torch.Tensor
    attention_profile_support_corrected: torch.Tensor
    attention_profile_untruncated: torch.Tensor
    attention_profile_per_head: torch.Tensor
    attention_entropy_per_head: torch.Tensor
    kld_per_head: torch.Tensor
    n_control_pairs: int = 0
    n_same_recording_pairs: int = 0
    per_anchor: Dict[str, torch.Tensor] = field(default_factory=dict)
    retained: Dict[str, torch.Tensor] = field(default_factory=dict)
    horizon_sums: Dict[str, torch.Tensor] = field(default_factory=dict)
    calibration_sums: Dict[str, torch.Tensor] = field(default_factory=dict)
    spectral_sums: Dict[str, torch.Tensor] = field(default_factory=dict)


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
    r"""Average a per-raw-sample quantity within each sample, over its scored raw samples.

    The forecast-side counterpart of :func:`_per_sample_mean`: the mask is per anchor and horizon
    step, and every one of the $R$ raw samples inside a horizon step shares its validity. The
    denominator is therefore $R \sum_{t,\tau} m_{t,\tau}$ -- the scored raw-sample count -- and not
    $H \cdot R$, which over-states it on any anchor with masked forecast steps.

    Args:
        values: $(B, T_{\mathrm{valid}}, H, R)$ values, or anything broadcastable to that shape.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.

    Returns:
        $(B,)$ per-sample means.
    """
    weights = mask[..., None]
    raw_per_step = float(values.shape[-1])
    denominator = (mask.sum(dim=(1, 2)) * raw_per_step).clamp_min(1.0)
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

    $$n_{b,\ell} = \sum_t m^{\mathrm{KL}}_{b,t}\,\mathbb{1}[t \ge \ell].$$

    Lag $\ell$ refers to source step $t - \ell$, which does not exist for $\ell > t$, so a long
    lag is contributed to by fewer anchors than a short one. The validity indicator is the
    attention module's own :meth:`~teb_vae.lag_attn.nets.attention.LagCrossAttention.build_lag_mask`
    rather than a second copy of the same inequality, so the denominator cannot describe a
    different support from the one the attention was actually computed over.

    Args:
        support: The KL anchor mask $(B, T)$.
        lag_validity: The lag-validity mask $(T, L)$, ``True`` where $t - \ell \ge 0$.

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
    sample's own KL, and the identity is a pinned property. It is also biased short: at the
    shipped geometry lags $0$--$30$ are averaged over all $240$ trained anchors while lag $90$ is
    averaged over $180$, a $25\%$ under-weight that moves the argmax toward short lags. The
    corrected form is the per-anchor mean each lag actually earned, and sums to nothing in
    particular.

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

    At the shipped geometry $L = 91$ and the trained range is $[30, 270)$, so $60$ of those $240$
    anchors are truncated -- a quarter of the evidence, which is why the restricted profile is
    reported *beside* the unrestricted one rather than in place of it.

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
    early anchors are truncated: measured against it, a model attending uniformly over everything
    available to it reads as increasingly concentrated the earlier the anchor. Both ceilings are
    reported, because the gap between them is a property of the geometry rather than of the model
    and a reader comparing against the wrong one draws the opposite conclusion.

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
    which the shipped model uses -- assigns lags *exactly* zero, and so does the causal mask at
    every truncated anchor, so a naive $p \log p$ produces ``nan`` on the majority of rows.

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
    it is over the raw samples themselves -- of which a real split holds $10^9$. So nothing is
    retained: each quantity below is a sum or a histogram, and each is exact against a
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
      the NLL, is bounded and does not diverge on a single badly-placed sample.
    * **The three NLL sums**, which turn into the gain over the homoscedastic MLE fitted to these
      very residuals -- the comparison that says whether the *learned* variance earned anything
      over one constant $\sigma$.
    * **The log-variance histogram**, over the clamp's own range. A mean alone is equally
      consistent with a well-spread distribution and with half the mass pinned on each clamp.

    Args:
        mu: Forecast mean $(B, T_{\mathrm{valid}}, H, R)$.
        logvar: Forecast log-variance, the same shape.
        target: The raw future target, the same shape.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.
        logvar_clamp: The model's own $(\mathrm{lo}, \mathrm{hi})$ log-variance bound, which fixes
            the histogram's range so two runs' histograms are comparable bin by bin.

    Returns:
        The sums and the two histograms, each reduced in float64 -- the counts alone reach $10^9$,
        where a float32 accumulator stops adding. The *elementwise* arithmetic stays in the
        model's own dtype, as every other accumulator here does: promoting a
        $(B, T_{\mathrm{valid}}, H, R)$ grid to float64 before reducing it would double the peak
        allocation of the pass for no accuracy the reduction does not already give.
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
    split puts $\\approx 5 \\times 10^7$ raw samples in a bin, and float32 stops representing
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
    per raw sample is $\tfrac{1}{2}[\log 2\pi + \overline{\log \sigma^2} + \overline{z^2}]$; the
    homoscedastic maximum-likelihood alternative fits **one** variance to the very residuals being
    scored, $\hat\sigma^2 = \overline{e^2}$, and scores
    $\tfrac{1}{2}[\log 2\pi + \log \hat\sigma^2 + 1]$. Their difference is what the *learned*,
    input-dependent variance earned over the best possible constant one -- and fitting the
    alternative on the scored residuals rather than on a held-out set is deliberate: it makes the
    baseline as strong as it can be, so the gain is a floor rather than a flattering estimate.

    Args:
        sums: What :func:`calibration_sums` accumulated, as tensors, arrays or plain lists.
        logvar_clamp: The bound the log-variance histogram was laid out over, for its bin edges.

    Returns:
        The PIT histogram and its worst departure from uniformity, central coverage against the
        erf nominals, mean CRPS, the standardised residual variance a calibrated model puts at
        $1$, the NLL gain, and the log-variance histogram. Empty when no raw sample was scored --
        a calibration statement over nothing is a skip, not a number.
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
        "n_raw_samples": int(count),
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
                "n_raw_samples": int(count),
            }
            for level, nominal in zip(COVERAGE_LEVELS, COVERAGE_NOMINALS)
        ],
        "crps_normalised": _scalar("crps_sum") / count,
        # One when the learned variance is right on average; above one when it is too small.
        "mean_standardised_sq": mean_standardised_sq,
        "mean_logvar_full": mean_logvar,
        "residual_variance": residual_variance,
        "nll": {
            "model_per_raw_sample": model_nll,
            "homoscedastic_per_raw_sample": homoscedastic_nll,
            # Positive means the input-dependent variance beat the best constant one.
            "gain_per_raw_sample": homoscedastic_nll - model_nll,
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
        # Pooled over raw samples rather than chained per recording, and said so here: PIT and
        # coverage are statements about a distribution, not means of a per-recording quantity, so
        # this census weights a recording by how many raw samples it contributed. The
        # per-recording chained figures -- mean log-variance and both clamp fractions -- are
        # columns on the per-sample table, where the chain does apply.
        "weighting": "pooled over scored raw samples, not averaged per recording",
    }


def horizon_residual_sums(
    mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Sum the residual and the log-variance over every scored raw sample, per horizon step.

    $$S^{\mathrm{sq}}_\tau = \sum_{b,t,r} m_{b,t,\tau}\,(x - \mu)^2, \qquad
    S^{z}_\tau = \sum_{b,t,r} m_{b,t,\tau}\,(x - \mu)^2 e^{-\log\sigma^2}, \qquad
    n_\tau = R \sum_{b,t} m_{b,t,\tau}.$$

    An accumulator rather than a retention. The residuals and log-variances themselves are
    $T_{\mathrm{valid}} \times H \times R$ per sample -- half a megabyte each, tens of gigabytes
    over a real split -- while what a calibration or a horizon-resolved skill number needs from
    them is these four vectors of length $H$. $S^{z}_\tau / n_\tau$ is the standardised residual
    variance, which a calibrated learned variance puts at $1$.

    The $\tau$ resolution is the point: it is the one axis that survives on neither durable table,
    because both are keyed per anchor and $\tau$ lives *inside* an anchor. The denominator is
    $\sum_{b,t} m_{b,t,\tau}$ rather than the per-anchor contributing indicator, which is an
    ``amax`` over $\tau$ and would count a masked forecast step as a scored zero.

    Sums accumulate in float64 -- a real split reaches $10^9$ terms, where float32 stops adding.

    Args:
        mu: Forecast mean $(B, T_{\mathrm{valid}}, H, R)$.
        logvar: Forecast log-variance, the same shape.
        target: The raw future target, the same shape.
        mask: The decimated forecast mask $(B, T_{\mathrm{valid}}, H)$.

    Returns:
        The four sums, each $(H,)$ in float64.
    """
    residual_sq = (target - mu) ** 2
    masked = mask[..., None]
    raw_per_step = float(mu.shape[-1])
    return {
        "sum_sq": (residual_sq * masked).sum(dim=(0, 1, 3), dtype=torch.float64),
        "sum_standardised_sq": (residual_sq * torch.exp(-logvar) * masked).sum(
            dim=(0, 1, 3), dtype=torch.float64
        ),
        "sum_logvar": (logvar * masked).sum(dim=(0, 1, 3), dtype=torch.float64),
        "count": mask.sum(dim=(0, 1), dtype=torch.float64) * raw_per_step,
    }


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

    Four latent branches are scored against the same raw future, under one shared set of noise
    draws:

    * ``base`` -- the target-only prior $p(z_t \mid Y_{\le t})$.
    * ``full`` -- the source-conditioned posterior $q(z_t \mid Y_{\le t}, U_{\le t})$.
    * ``shuffled`` -- the posterior rebuilt from a *stranger's* source, the negative control that
      makes a nonzero KL mean something.
    * ``base_shuffled_mu`` -- the base forecast from a stranger's *prior*, which is the check
      that the prior latent is carrying the target state at all rather than the decoder having
      learned a recording-independent average.

    Three latent-free forecasts are scored beside them -- persistence, climatology and the
    segment mean (:func:`baseline_forecasts`) -- through the same loss function and the same
    mask, because a summed-$H \cdot R$-sample block score is a large number under any predictor and
    only a comparison says whether the model is good or merely arithmetically fine.

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
            the raw future and ``'up_raw'`` / ``'weight'`` for the source trace and the validity
            behind it. Empty by default: the model tensors are $(B, T_{\mathrm{valid}}, H, R)$ or
            $(B, T, M, L)$, hundreds of kilobytes per sample, so retaining one is a decision a
            caller makes rather than a default it inherits.

    Returns:
        The batch's per-sample readouts.

    Raises:
        NoCrossGroupPartner: If the batch carries recording identifiers but no cross-recording
            pairing exists -- one recording holding more than half the batch. Callers running a
            whole loader test this with :func:`~teb_vae.lag_attn_rws.nets.controls.groups_can_derange`
            and exclude such a batch, counting the exclusion.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))

    y_st, y_ph, u_stream, fhr_raw, weight = model_inputs(task, batch)
    outputs = model(y_st, y_ph, u_stream)

    geometry = model.geometry
    target = build_future_target(fhr_raw, geometry, future_index=model.future_index)
    mask, coverage = forecast_mask(weight, geometry, coverage_floor=model.coverage_floor)
    kl_support = kl_mask(mask, geometry)

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
        permuted = controls.perm_forward_outputs(
            model, outputs, generator=perm_generator, groups=recordings
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
        model, branches, target, mask, likelihood=likelihood,
        num_samples=num_samples, generator=mc_generator,
    )

    # The training-path score: the forward's own decoded latents, the same functions the objective
    # uses. Under ``base_decode: mean`` ``mu_base`` was decoded at the prior MEAN while ``mu_full``
    # was decoded at one posterior sample, so ``nll_base_block - nll_full_block`` mixes a source
    # comparison with a decoding-policy difference; it is reported for objective parity only, and
    # the matched common-random-numbers scores above are the ones a source-gain claim reads.
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
        # rather than a per-coordinate figure. Proportional to the one above by $\sqrt{d_z}$ at
        # equal support, and reported separately because the two answer different questions and
        # an implementation that conflated them would differ by exactly that factor.
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

    # The three trivial forecasts, scored through the model's *own* loss function with the
    # identical mask -- so a skill score is a comparison of predictors rather than of scoring
    # conventions. Their observation variance is fixed and stated; see BASELINE_LOGVAR.
    baselines = baseline_forecasts(fhr_raw, weight, geometry)
    baseline_logvar = torch.full((), BASELINE_LOGVAR, dtype=fhr_raw.dtype, device=fhr_raw.device)
    for name, baseline_mu in baselines.items():
        baseline_block, _ = masked_raw_block_per_anchor(
            baseline_mu, target, mask, likelihood=likelihood, logvar=baseline_logvar
        )
        columns[f"nll_{name}_block"] = _per_sample_mean(baseline_block, contributing)

    # Point-forecast error, in the loader's z units and per *scored raw sample* rather than per
    # anchor: an RMSE is a statement about the waveform, not about the block. The squares stay
    # unrooted here -- see :func:`masked_raw_error_sums`.
    point_forecasts: Dict[str, torch.Tensor] = {
        "base": outputs["mu_base"], "full": outputs["mu_full"], **baselines
    }
    for name, point_mu in point_forecasts.items():
        sums = masked_raw_error_sums(point_mu, target, mask)
        scored = sums["n_raw"].clamp_min(1.0)
        columns[f"sq_error_{name}"] = sums["sum_sq"] / scored
        if name in ("base", "full"):
            # Only the model branches: the baselines exist to normalise the squared error, and a
            # constant predictor's bias is its own definition rather than a finding.
            columns[f"abs_error_{name}"] = sums["sum_abs"] / scored
            columns[f"signed_error_{name}"] = sums["sum_residual"] / scored

    # How far apart the two forecasts are, per scored raw sample, unrooted for the same Jensen
    # reason as the latent quantities above. Distinct from ``pred_gap``, which is a difference of
    # *scores*: two forecasts can differ everywhere and score identically.
    columns["forecast_difference_sq"] = _per_sample_element_mean(
        (outputs["mu_full"] - outputs["mu_base"]) ** 2, mask
    )

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
    lag_validity = model.lag_attn.build_lag_mask(seq_len, device=kl_support.device)
    lag_profile, lag_profile_corrected, lag_support = lag_profiles(
        outputs["source_kl_lag_map"], kl_support, lag_validity
    )

    # The attention, on the identical footing: the same anchor support, the same two denominators,
    # and a third profile over the anchors whose lag support is complete -- see
    # :func:`untruncated_anchor_mask` for why the correction alone is not enough here.
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
    # The KL attribution restricted to the same anchor set. It is the attention scaled by $K_t$,
    # so it inherits the renormalisation the support correction cannot undo; on this anchor set
    # every lag is valid, which is also why no per-lag denominator is needed here.
    lag_profile_untruncated = _per_sample_vector_mean(
        outputs["source_kl_lag_map"], kl_support * untruncated
    )
    # Head-major flattening, so one trailing axis reaches the aggregation chain; the head count
    # travels beside it in the lag report and the reshape happens once, in the consumer.
    per_head_attention = _per_sample_vector_mean(
        attention.reshape(attention.shape[0], seq_len, -1), kl_support
    )
    per_head_entropy = _per_sample_vector_mean(attention_entropy(attention), kl_support)

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
    # measurement rather than an approximation. $\log L$ is the other ceiling, and is a constant.
    columns["attention_entropy_nats"] = per_head_entropy.mean(dim=-1)
    columns["attention_entropy_attainable_nats"] = _per_sample_mean(
        attainable_lag_entropy(
            seq_len, n_lags, device=kl_support.device, dtype=kl_support.dtype
        ).expand(batch_size, -1),
        kl_support,
    )

    t_valid = geometry.t_valid
    # The same numbers as the columns above, before the within-sample reduction. Both score pairs
    # travel, so the per-anchor table recombines into the per-sample one exactly rather than into
    # one of the two conventions -- and neither name has to stand for both.
    per_anchor: Dict[str, torch.Tensor] = {
        "contributing": contributing,
        "coverage": coverage,
        "kld_per_t": outputs["kld_per_t"][:, :t_valid],
        "nll_base_block": training_base_block,
        "nll_full_block": training_block,
        "pred_gap": training_base_block - training_block,
        # The lag the KL attribution peaks at, per anchor. Meaningful only where the anchor is
        # supported, which is exactly the set of rows the table keeps.
        "argmax_lag": outputs["source_kl_lag_map"][:, :t_valid].argmax(dim=-1),
    }
    for name in ("base", "full"):
        if name in scores:
            per_anchor[f"mc_nll_{name}_block"] = scores[name]
    if "base" in scores and "full" in scores:
        per_anchor["mc_pred_gap"] = scores["base"] - scores["full"]

    # The observation model's calibration census, over the full branch's raw samples. Empty under
    # ``'mse'``: the decoder's log-variance head is never fitted there, so a probability integral
    # transform of its output would be arithmetic over an untrained tensor.
    calibration = (
        calibration_sums(
            outputs["mu_full"], outputs["logvar_full"], target, mask,
            logvar_clamp=model.logvar_clamp,
        )
        if likelihood == "gaussian_nll"
        else {}
    )

    # The tau-slice cross-spectra, over every sample of the pass. Unconditional, and deliberately
    # not behind a retention cap: `retained` exists because a (T_valid, H, R) tensor is hundreds of
    # kilobytes per sample, while these reduce to (H, F) per sample and reduce again to (H, B) on
    # the durable sidecar. Making them optional would mean the coherence analysis is silent under
    # the shipped `caps: {}` and cannot re-run offline, which is the property the whole
    # collect-then-analyse split exists to have.
    spectral = cross_spectral_sums(
        target,
        outputs["mu_base"],
        outputs["mu_full"],
        batch_field(batch, "up"),
        mask,
        layout=spectra.slice_geometry(
            t_valid=geometry.t_valid,
            warmup=geometry.warmup,
            horizon=geometry.horizon,
            raw_per_step=geometry.r,
        ),
    )

    # Built only when asked for: the names resolve against the forward's own outputs plus the
    # raw future, so a retained array cannot be a differently assembled version of what was
    # scored.
    retained: Dict[str, torch.Tensor] = {}
    if retain:
        available: Dict[str, torch.Tensor] = dict(outputs)
        available["target"] = target
        # The two raw traces beside the forecast, for the event analyses. `up_raw` is the only
        # signal in this pipeline the model never sees in raw form -- the source reaches it as
        # scattering and phase channels -- so a contraction can be located nowhere else; and
        # `weight` is what says which of those raw samples are real, since a gap is stored as a
        # value rather than as a sentinel. Both are three orders of magnitude smaller than the
        # forecast blocks they travel with, so they cost the retention plan nothing.
        available["weight"] = weight
        up_raw = batch_field(batch, "up")
        if isinstance(up_raw, torch.Tensor):
            available["up_raw"] = up_raw
        retained = {name: available[name] for name in retain if name in available}

    return BatchReadout(
        guids=batch_guids(batch, batch_size),
        columns=columns,
        n_anchors=contributing.sum(dim=1),
        kld_per_dim=_per_sample_vector_mean(kld_btd, kl_support),
        lag_profile=lag_profile,
        lag_profile_support_corrected=lag_profile_corrected,
        lag_profile_untruncated=lag_profile_untruncated,
        lag_support=lag_support,
        attention_profile=attention_profile,
        attention_profile_support_corrected=attention_profile_corrected,
        attention_profile_untruncated=attention_profile_untruncated,
        attention_profile_per_head=per_head_attention,
        attention_entropy_per_head=per_head_entropy,
        kld_per_head=_per_sample_vector_mean(outputs["kld_per_t_per_head"], kl_support),
        calibration_sums=calibration,
        spectral_sums=spectral,
        n_control_pairs=n_control_pairs,
        n_same_recording_pairs=n_same_recording_pairs,
        per_anchor=per_anchor,
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
            complete -- the profile free of the renormalisation bias the correction cannot reach.
        lag_support: Contributing anchors per lag, averaged on the same chain -- the correction's
            denominator, in anchors per segment.
        attention_profile: Per-lag attention on the same chain.
        attention_profile_support_corrected: The same attention on each lag's own denominator.
        attention_profile_untruncated: The same attention over the anchors whose lag support is
            complete.
        attention_profile_per_head: Per-head attention per lag, flattened head-major to
            $M \cdot L$ entries.
        attention_entropy_per_head: Per-head attention entropy in nats, one value per head.
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
    lag_support: List[float] = field(default_factory=list)
    attention_profile: List[float] = field(default_factory=list)
    attention_profile_support_corrected: List[float] = field(default_factory=list)
    attention_profile_untruncated: List[float] = field(default_factory=list)
    attention_profile_per_head: List[float] = field(default_factory=list)
    attention_entropy_per_head: List[float] = field(default_factory=list)

    @property
    def n_recordings(self) -> int:
        """How many distinct recordings contributed."""
        return len(self.per_recording)


def aggregate_by_recording(readouts: Sequence[BatchReadout]) -> Aggregate:
    r"""Average each column within a recording, then across recordings.

    Not a flat mean over anchors or over segments. Consecutive anchors' $30$-step forecast
    windows overlap in $29$ of them, so anchors within a recording are very far from independent;
    averaging over them and reporting the result as if it had that many samples behind it
    overstates the precision of every number here, and weights the headline toward whichever
    recordings happen to be longest.

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
            # reads as exactly 0.0. Averaging that in would pull a summed-(H*R)-sample block
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
        # comparisons would FAIL on `0.0 == 0.0` and the two clamp verdicts would PASS on a
        # log-variance nothing ever wrote. Absent lets each verdict reach its own "not measured"
        # branch and report INCONCLUSIVE, which is what a run that measured nothing means.
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
    removed by a different denominator or a different anchor set.

    * The **raw** KL attribution's argmax is what the decomposition of $\bar K$ peaks at. It
      divides every bin by the same anchor total, so a long lag -- causally valid at fewer anchors
      -- is averaged over anchors that could not contribute to it.
    * The **support-corrected** argmax divides each bin by its own contributing-anchor count.
    * The **attention** argmax is the same reading on the attention weights alone, without $K_t$
      weighting it.
    * The **untruncated** attention argmax restricts to anchors at which every lag exists. The
      support correction fixes each bin's denominator and cannot fix its numerator: at a truncated
      anchor the attention mass that had no long lag to reach was renormalised onto the short
      ones, and no per-lag count knows that happened.

    Where two of them disagree, the difference *is* the corresponding bias, which is why they
    travel together rather than one replacing the rest.

    The **per-head** structure travels here too. The posterior is head-structured, so $K_t$ splits
    additively across heads and each head attends over its own lags; head-averaging before
    profiling discards exactly what that structure exists to expose. The head-averaged profile is
    still emitted, named as such.

    Args:
        aggregate: The aggregated readouts.
        delay_steps: The causal input delay $\delta$ applied to the source channels.
        identity_residuals: The worst per-anchor residual of each structural identity over the
            whole pass, by name. Recorded here rather than recomputed downstream because it is a
            maximum over samples and the aggregation chain reports means.

    Returns:
        The four argmaxes with their compensated (residual physiological) seconds, the profiles
        themselves, the per-lag anchor counts behind the correction, the per-head split, the
        attention entropies against both ceilings, and the identity residuals. Empty when no lag
        profile was collected.
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
    return {
        "delay_steps": int(delay_steps),
        # The source channels are delayed individually and the maximum is what the model reports,
        # so every lag above is an upper bound. The flag travels with the numbers rather than
        # being stated once elsewhere, because a lag quoted without it reads as exact.
        "source_delay_is_max_over_channels": True,
        "n_lags": n_lags,
        "num_heads": num_heads,
        "kl_argmax_lag_step": kl_argmax,
        "kl_lag_compensated_seconds": float(
            lag_compensated_seconds(kl_argmax, delay_steps=delay_steps)
        ),
        # No sensor-timeline twin of the figure above: the stored UP/FHR timeline is canonical
        # and the builder's shift is never undone downstream.
        "kl_argmax_lag_step_support_corrected": corrected_argmax,
        "kl_lag_compensated_seconds_support_corrected": (
            None
            if corrected_argmax is None
            else float(lag_compensated_seconds(corrected_argmax, delay_steps=delay_steps))
        ),
        # The attribution's counterpart of the untruncated attention argmax below, and for the
        # same reason: the attribution is the attention scaled by $K_t$, so it carries the same
        # renormalisation the per-lag denominator cannot undo.
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
        # The claim that survives the truncation bias, and the only one an argmax statement about
        # the attention should rest on.
        "attention_argmax_lag_step_untruncated": attention_untruncated_argmax,
        "attention_lag_compensated_seconds_untruncated": _seconds_of(
            attention_untruncated_argmax, delay_steps
        ),
        "kl_lag_profile": list(aggregate.lag_profile),
        "kl_lag_profile_support_corrected": list(corrected),
        "kl_lag_profile_untruncated": list(aggregate.lag_profile_untruncated),
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
        # The additive per-head split of the KL, which nothing read before: it is the quantity a
        # head-structured posterior exists to make meaningful, and it sums over heads to the
        # headline KL exactly.
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
#: Every entry is promoted today. That is not the same as promotion being redundant: a later
#: diagnostic verdict may be worth reporting without being one of the numbers an acceptance gate
#: reads, and the column is what keeps that decision here rather than in the reporting layer.
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
    latent geometry are identical between the two branches, and the shuffled branch is the same
    posterior handed a stranger's. So it is the one predictive comparison that survives a negative
    predictive gain, and its passing beside a failing ``source_specificity`` is a state to report
    rather than a contradiction to resolve.

    It is strictly weaker than ``source_specificity``, which asks
    $D_{\rm full} < D_{\rm base} < D_{\rm shuffled}$ and therefore implies this: a run cannot pass
    that and fail this.

    Args:
        d_full: The source-conditioned branch's marginalised block score, or ``None``.
        d_shuffled: The stranger's-source branch's, or ``None`` when the control did not run.

    Returns:
        The verdict, ``INCONCLUSIVE`` when either is missing -- a control that could not run and a
        control that failed are different facts.
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
    path, so a directory collected under seven criteria would be re-reported as a seven-criterion
    run under a pipeline that declares eight. A summary silently missing a criterion reads exactly
    like one where that criterion passed.

    Recomputing the missing entries instead was considered and rejected. Only some criteria are
    decidable from what a collection record keeps -- the two predictive ones are, the calibration
    census is not -- so a repair path would work for the criteria that happen to be cheap and
    fail for the rest, which is a worse failure than a refusal because it is a partial one.

    Args:
        cached: The reused record's verdict list, or ``None`` when it carried none. ``None`` is
            accepted: a record written before the block existed at all is a different and older
            problem, and the analyses re-run against it still produce their own numbers.

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
    logvar_margin: Optional[float] = None,
    calibration: Optional[Mapping[str, Any]] = None,
) -> List[Verdict]:
    r"""Turn the aggregated readouts into the acceptance verdicts.

    The two predictive criteria are the model's own: $D_{\mathrm{full}} < D_{\mathrm{base}}$, and
    $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$. The two representation
    criteria check that the thing being measured is where it is claimed to be: that the prior
    latent carries the target state (shuffling it must hurt), and that the KL has not collapsed
    onto one or two dimensions.

    The three variance criteria check that the *numbers* mean what they say. A prior variance
    pinned on its lower clamp multiplies every coupling readout by an arbitrary factor while every
    decoder-side diagnostic stays healthy; a decoder variance pinned on either clamp means the
    observation model has stopped being one; and an observation model whose central coverage
    misses the erf nominals is one whose NLL is not a log density of anything.

    Args:
        aggregate: The aggregated readouts.
        prior_shuffle_min_nats: Minimum degradation from a shuffled prior latent.
        min_active_dims: Minimum active latent dimensions.
        pinned_variance_max_frac: How much of a bounded variance may sit within the margin of one
            of its clamps before it counts as pinned there.
        coverage_tail_tolerance: Relative tolerance on the observed tail mass at each coverage
            level.
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
        prior_shuffle_min_nats: Verdict margin for the prior-shuffle control.
        min_active_dims: Verdict threshold for latent collapse.
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
            # loader so the aggregation can run over all of them at once, and the per-anchor,
            # retained and cross-spectral tensors are two to three orders of magnitude larger than
            # the per-sample columns beside them -- keeping those alive for a full split would cost
            # gigabytes for values nothing reads again.
            #
            # ``spectral_sums`` is the largest of the three and the easiest to overlook, because
            # unlike ``retained`` it is populated unconditionally: fourteen $(B, H, F)$ float64
            # arrays, $0.8$ MiB per segment, still on the model's device -- the sink copies them to
            # host but that does not free the originals. Nothing after this loop reads it.
            readout.per_anchor = {}
            readout.retained = {}
            readout.spectral_sums = {}
            if max_batches is not None and len(readouts) >= int(max_batches):
                break
    finally:
        task.train(was_training)

    aggregate = aggregate_by_recording(readouts)
    # Summed over batches here rather than in a sink, because the calibration statistics are the
    # one block a verdict reads that is not on the aggregation chain: a probability-integral
    # transform is a statement about a distribution over raw samples, not the mean of a
    # per-recording quantity.
    calibration_totals: Dict[str, torch.Tensor] = {}
    for readout in readouts:
        for name, value in readout.calibration_sums.items():
            calibration_totals[name] = (
                value if name not in calibration_totals else calibration_totals[name] + value
            )
    clamp = getattr(task.orig_model, "logvar_clamp", None)
    calibration = calibration_report(
        {name: value.detach().cpu() for name, value in calibration_totals.items()},
        logvar_clamp=clamp,
    )
    verdicts = build_verdicts(
        aggregate,
        prior_shuffle_min_nats=prior_shuffle_min_nats,
        min_active_dims=min_active_dims,
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
