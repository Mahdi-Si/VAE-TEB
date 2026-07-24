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

**The verdicts.** Each is ``PASS``, ``FAIL`` or ``INCONCLUSIVE``, never a bare boolean, and each
carries the numbers that produced it. A label with no numbers behind it is a claim a reader
cannot check.

One aggregation decision runs through all of it: **quantities are averaged per recording, then
across recordings.** Anchors are not independent samples of anything -- consecutive anchors'
forecast windows overlap in $29$ of their $30$ horizon steps, and a single long recording holds
hundreds of them -- so a flat anchor mean weights recordings by their length and reports an
effective sample size far larger than the data supports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.lag_report import (
    lag_compensated_seconds,
    lag_original_sensor_seconds,
)
from teb_vae.lag_attn_rws.nets.losses import KLD_ACTIVE_EPS, masked_raw_block_per_anchor
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target

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

#: Below this total KL (nats per anchor) there is no coupling to be distributed over dimensions,
#: so a collapse verdict would be reporting the absence of a signal as a structural failure.
_COLLAPSE_INCONCLUSIVE_KL = 1e-6

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"


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


def batch_guids(batch: Any, batch_size: int) -> List[str]:
    """Return one recording identifier per sample in the batch.

    ``guid`` survives collation as a ``list[str]`` rather than a tensor, which is why it is never
    moved to a device and why it is read as a Python value here.

    Args:
        batch: A batch from the data module.
        batch_size: Number of samples, taken from a tensor field rather than from ``guid``
            itself, so a malformed identifier list cannot silently change the sample count.

    Returns:
        A list of length ``batch_size``; ``'unknown'`` wherever the batch carries no identifier.
    """
    field_value = batch.get("guid") if isinstance(batch, dict) else getattr(batch, "guid", None)
    if field_value is None:
        return ["unknown"] * batch_size
    if isinstance(field_value, (list, tuple)):
        return [
            str(field_value[index]) if index < len(field_value) else "unknown"
            for index in range(batch_size)
        ]
    if isinstance(field_value, torch.Tensor):
        return [str(field_value[index].item()) for index in range(batch_size)]
    return [str(field_value)] * batch_size


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
        num_samples: Monte Carlo draws $K$. At $K = 1$ the result is exactly the training-path
            per-anchor score for the same draw.

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
        epsilon = torch.randn_like(reference_mu)
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
# Per-batch evaluation
# =============================================================================
@dataclass
class BatchReadout:
    """Per-sample readouts from one batch, plus the anchor counts that weight them.

    Every scalar column is a per-sample mean over that sample's contributing anchors, and
    ``n_anchors`` is how many those were. Keeping the count is what lets an anchor-weighted total
    be reconstructed exactly -- which is how the evaluation is checked against the training loss
    -- while the per-sample values are what the per-recording aggregation needs.

    Attributes:
        guids: Recording identifier per sample.
        columns: Named per-sample values, each a $(B,)$ tensor.
        n_anchors: Contributing anchors per sample, $(B,)$.
        kld_per_dim: Mean per-dimension KL over this batch's masked anchors, $(d_z,)$.
        lag_profile: Mean per-lag KL attribution over this batch's masked anchors, $(L,)$.
        attention_profile: Mean head-averaged attention per lag over the same support, $(L,)$.
    """

    guids: List[str]
    columns: Dict[str, torch.Tensor]
    n_anchors: torch.Tensor
    kld_per_dim: torch.Tensor
    lag_profile: torch.Tensor
    attention_profile: torch.Tensor


def _per_sample_mean(per_anchor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Average a per-anchor quantity within each sample, over its weighted anchors.

    Args:
        per_anchor: $(B, T_\\ast)$ values.
        weights: $(B, T_\\ast)$ non-negative weights; zero anchors drop out entirely.

    Returns:
        $(B,)$ per-sample means.
    """
    return (per_anchor * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def evaluate_batch(
    task: Any,
    batch: Any,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    perm_generator: Optional[torch.Generator] = None,
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

    Args:
        task: The Lightning task wrapping the loaded net.
        batch: A batch already on the model's device.
        num_samples: Monte Carlo draws $K$.
        perm_generator: Generator seeding the derangement, so a run is reproducible.

    Returns:
        The batch's per-sample readouts.
    """
    model = task.orig_model
    likelihood = str(task.hparams.get("likelihood", "gaussian_nll"))

    y_st, y_ph, u_stream, fhr_raw, weight = model_inputs(task, batch)
    outputs = model(y_st, y_ph, u_stream)

    geometry = model.geometry
    target = build_future_target(fhr_raw, geometry, future_index=model.future_index)
    mask, _coverage = forecast_mask(weight, geometry, coverage_floor=model.coverage_floor)
    kl_support = kl_mask(mask, geometry)

    branches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
        "base": (outputs["mu_prior"], outputs["logvar_prior"]),
        "full": (outputs["mu_post"], outputs["logvar_post"]),
    }
    batch_size = int(y_st.shape[0])
    if batch_size >= 2:
        permuted = controls.perm_forward_outputs(model, outputs, generator=perm_generator)
        index = permuted["perm_index"]
        branches["shuffled"] = (permuted["mu_post"], permuted["logvar_post"])
        # The same derangement for both controls, so "a stranger's source" and "a stranger's
        # prior" name the same stranger and the two numbers are comparable.
        branches["base_shuffled_mu"] = (
            outputs["mu_prior"][index],
            outputs["logvar_prior"][index],
        )

    scores, contributing = mc_predictive_block(
        model, branches, target, mask, likelihood=likelihood, num_samples=num_samples
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

    columns: Dict[str, torch.Tensor] = {
        "nll_base_block": _per_sample_mean(training_base_block, contributing),
        "nll_full_block": _per_sample_mean(training_block, contributing),
        "source_conditioned_kl_raw": _per_sample_mean(outputs["kld_per_t"], kl_support),
        "mu_prior_rms": _per_sample_mean(
            (outputs["mu_prior"] ** 2).mean(dim=-1), kl_support
        ).sqrt(),
        "delta_mu_rms": _per_sample_mean((delta_mu**2).mean(dim=-1), kl_support).sqrt(),
    }
    columns["pred_gap"] = columns["nll_base_block"] - columns["nll_full_block"]
    for name, value in scores.items():
        columns[f"mc_nll_{name}_block"] = _per_sample_mean(value, contributing)
    if "mc_nll_base_block" in columns and "mc_nll_full_block" in columns:
        columns["mc_pred_gap"] = columns["mc_nll_base_block"] - columns["mc_nll_full_block"]

    # Latent and lag summaries over the KL's own anchor support, so the untrained tail does not
    # dilute them.
    support = kl_support.unsqueeze(-1)
    support_total = support.sum().clamp_min(1.0)
    kld_per_dim = (kld_btd * support).sum(dim=(0, 1)) / support_total
    lag_profile = (outputs["source_kl_lag_map"] * support).sum(dim=(0, 1)) / support_total
    attention_profile = (
        outputs["attn_weights"].mean(dim=2) * support
    ).sum(dim=(0, 1)) / support_total

    return BatchReadout(
        guids=batch_guids(batch, batch_size),
        columns=columns,
        n_anchors=contributing.sum(dim=1),
        kld_per_dim=kld_per_dim,
        lag_profile=lag_profile,
        attention_profile=attention_profile,
    )


# =============================================================================
# Aggregation
# =============================================================================
@dataclass
class Aggregate:
    """Readouts aggregated per recording and then across recordings.

    Attributes:
        per_recording: Per-guid means of every column.
        overall: The mean across recordings of each column -- the headline numbers.
        n_samples: Segments seen.
        n_samples_without_anchors: Segments excluded for scoring no anchors at all. Reported
            rather than silently dropped: a run where this is large measured far less than its
            segment count suggests.
        kld_per_dim: Mean per-dimension KL across batches.
        lag_profile: Mean per-lag KL attribution across batches.
        attention_profile: Mean per-lag attention across batches.
    """

    per_recording: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_samples_without_anchors: int = 0
    kld_per_dim: List[float] = field(default_factory=list)
    lag_profile: List[float] = field(default_factory=list)
    attention_profile: List[float] = field(default_factory=list)

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
    counts: Dict[str, int] = {}
    for readout in readouts:
        aggregate.n_samples += len(readout.guids)
        for position, guid in enumerate(readout.guids):
            # A segment that scored no anchors -- every anchor gapped or below the coverage
            # floor -- measured nothing. Its columns are not small, they are absent: the
            # per-sample mean divides by a denominator clamped to 1, so an empty numerator
            # reads as exactly 0.0. Averaging that in would pull a summed-480-sample block
            # score (hundreds of nats) toward zero and shrink pred_gap, with no other symptom.
            if float(readout.n_anchors[position]) <= 0.0:
                aggregate.n_samples_without_anchors += 1
                continue
            bucket = sums.setdefault(guid, {name: 0.0 for name in names})
            counts[guid] = counts.get(guid, 0) + 1
            for name in names:
                bucket[name] += float(readout.columns[name][position])

    aggregate.per_recording = {
        guid: {name: total / counts[guid] for name, total in bucket.items()}
        for guid, bucket in sums.items()
    }
    n_recordings = float(len(aggregate.per_recording)) or 1.0
    aggregate.overall = {
        name: sum(values[name] for values in aggregate.per_recording.values()) / n_recordings
        for name in names
    }

    def _stacked_mean(attribute: str) -> List[float]:
        """Mean of a per-batch vector readout across batches."""
        stack = torch.stack([getattr(readout, attribute) for readout in readouts], dim=0)
        return [float(value) for value in stack.mean(dim=0)]

    aggregate.kld_per_dim = _stacked_mean("kld_per_dim")
    aggregate.lag_profile = _stacked_mean("lag_profile")
    aggregate.attention_profile = _stacked_mean("attention_profile")
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


def lag_summary(aggregate: Aggregate, *, delay_steps: int = 0) -> Dict[str, Any]:
    r"""Report the dominant lag in both of the two seconds figures that may be quoted.

    Args:
        aggregate: The aggregated readouts.
        delay_steps: The causal input delay $\delta$ applied to the source channels.

    Returns:
        The argmax lag of the KL attribution and of the attention, each with its compensated
        (residual physiological) and original-sensor-timeline seconds. Empty when no lag profile
        was collected.
    """
    if not aggregate.lag_profile:
        return {}
    kl_argmax = max(range(len(aggregate.lag_profile)), key=aggregate.lag_profile.__getitem__)
    attention_argmax = max(
        range(len(aggregate.attention_profile)), key=aggregate.attention_profile.__getitem__
    )
    return {
        "delay_steps": int(delay_steps),
        "kl_argmax_lag_step": kl_argmax,
        "kl_lag_compensated_seconds": float(
            lag_compensated_seconds(kl_argmax, delay_steps=delay_steps)
        ),
        "kl_lag_original_sensor_seconds": float(
            lag_original_sensor_seconds(kl_argmax, delay_steps=delay_steps)
        ),
        "attention_argmax_lag_step": attention_argmax,
        "attention_lag_compensated_seconds": float(
            lag_compensated_seconds(attention_argmax, delay_steps=delay_steps)
        ),
        "kl_lag_profile": list(aggregate.lag_profile),
        "attention_lag_profile": list(aggregate.attention_profile),
    }


# =============================================================================
# Verdicts
# =============================================================================
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


def build_verdicts(
    aggregate: Aggregate,
    *,
    prior_shuffle_min_nats: float = DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    min_active_dims: int = DEFAULT_MIN_ACTIVE_DIMS,
) -> List[Verdict]:
    r"""Turn the aggregated readouts into the acceptance verdicts.

    The two predictive criteria are the model's own: $D_{\mathrm{full}} < D_{\mathrm{base}}$, and
    $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$. The two representation
    criteria check that the thing being measured is where it is claimed to be: that the prior
    latent carries the target state (shuffling it must hurt), and that the KL has not collapsed
    onto one or two dimensions.

    Args:
        aggregate: The aggregated readouts.
        prior_shuffle_min_nats: Minimum degradation from a shuffled prior latent.
        min_active_dims: Minimum active latent dimensions.

    Returns:
        The verdicts, in reporting order.
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

    if base is None or full is None or shuffled is None:
        verdicts.append(
            Verdict(
                "source_specificity", INCONCLUSIVE,
                "D_full < D_base < D_shuffled",
                "the permutation control did not run; it needs a batch of at least two samples.",
                {},
            )
        )
    else:
        ordered = full < base < shuffled
        verdicts.append(
            Verdict(
                "source_specificity", PASS if ordered else FAIL,
                "D_full < D_base < D_shuffled",
                "a stranger's source is worse than no source, so the model uses *this* "
                "recording's source rather than reacting to any source at all."
                if ordered
                else "the ordering does not hold, so a nonzero KL cannot be read as "
                     "source-specific coupling.",
                {
                    "d_base": base, "d_full": full, "d_shuffled": shuffled,
                    "shuffle_penalty": shuffled - base,
                },
            )
        )

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
    return verdicts


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
    delay_steps: int = 0,
    prior_shuffle_min_nats: float = DEFAULT_PRIOR_SHUFFLE_MIN_NATS,
    min_active_dims: int = DEFAULT_MIN_ACTIVE_DIMS,
) -> Dict[str, Any]:
    """Evaluate a loaded task over a dataloader and assemble the JSON-shaped results.

    Batches too small to derange are skipped rather than partially scored: they would produce a
    different set of columns, and averaging an inconsistent set together is how a control quietly
    stops being reported without anything failing.

    Args:
        task: The Lightning task wrapping the loaded net, already in ``eval`` mode.
        loader: A dataloader over the evaluation shards.
        num_samples: Monte Carlo draws $K$.
        max_batches: Stop after this many scored batches; ``None`` means the whole loader.
        perm_generator: Generator seeding the derangements.
        delay_steps: The causal input delay applied to the source channels, for the lag report.
        prior_shuffle_min_nats: Verdict margin for the prior-shuffle control.
        min_active_dims: Verdict threshold for latent collapse.

    Returns:
        A dict of readouts, latent health, the lag report, per-recording means and the verdicts.
    """
    was_training = task.training
    task.eval()
    readouts: List[BatchReadout] = []
    skipped = 0
    try:
        for batch in loader:
            batch = task.transfer_batch_to_device(batch, task.device, dataloader_idx=0)
            if batch_size_of(batch) < 2:
                skipped += 1
                continue
            readouts.append(
                evaluate_batch(
                    task, batch, num_samples=num_samples, perm_generator=perm_generator
                )
            )
            if max_batches is not None and len(readouts) >= int(max_batches):
                break
    finally:
        task.train(was_training)

    aggregate = aggregate_by_recording(readouts)
    verdicts = build_verdicts(
        aggregate,
        prior_shuffle_min_nats=prior_shuffle_min_nats,
        min_active_dims=min_active_dims,
    )
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
        "lag": lag_summary(aggregate, delay_steps=delay_steps),
        "per_recording": aggregate.per_recording,
        "verdicts": [verdict.as_dict() for verdict in verdicts],
    }
