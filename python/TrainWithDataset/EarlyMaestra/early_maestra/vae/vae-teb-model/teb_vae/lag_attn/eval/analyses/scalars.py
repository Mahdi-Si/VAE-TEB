r"""What is this checkpoint's test loss, metric by metric, against what training logged?

The plainest question about a checkpoint, and the only analysis that answers it. Everything else
in the pipeline reports a *per-sample* quantity; this one reports the **pooled** form -- one
global denominator over the whole split, exactly as ``compute_loss`` reduces -- so its numbers
are directly comparable with a row of the training run's ``metrics.csv``.

Two rules about what lands in the table.

**The metric set is the task's.** :data:`METRIC_SUFFIXES` mirrors
``trainer.py::_METRIC_SUFFIXES``, and a test asserts the two are equal, so a metric added to the
task and not here fails the suite rather than going missing from a table that looks complete.
It is mirrored rather than imported because ``trainer.py`` pulls in Lightning, and an eval run
has no business standing up a training framework to read a tuple of strings. A suffix this pass
cannot produce is recorded as *not applicable* with a reason rather than quietly omitted.

**The permutation-control metrics are omitted on batches where the control did not run, never
zero-filled.** ``task.py`` records why at length: a zero on a step the control skipped is not a
harmless placeholder, it scales the mean of ``feat_loss_shuffled`` and ``shuffle_penalty``
toward zero, which inverts the very ordering the control exists to check and reads as a
collapsed source pathway on a perfectly healthy model. Here the control is Sprint 6's, so its
keys are simply absent -- recorded as such.

**The reported $\beta$ is the schedule's, not the configured constant.** ``kld_beta`` in the
config is documented as the fallback for ``beta_schedule.kind == constant``; the shipped config
ships a ``linear_warmup`` reaching $0.1$, and ``task.py`` logs *that* under the name
``kld_beta``. Reporting the constant would put the wrong number in the ``kld_beta`` column and
would additionally shift ``total_loss`` and ``main_loss`` by
$(\beta_{\mathrm{eff}} - \beta_{\mathrm{cfg}})\,L_{KL}$ -- a hundredfold error in $\beta$ on the
shipped settings, in a row whose whole purpose is to line up against training.

The joint collapse verdict lives here too, because both of its components are pooled scalars.
It is deliberately *one* verdict rather than two independently near-zero numbers: the shipped
config documents collapse as ``pred_gap`` near zero **and** ``kld_raw`` near zero *together*,
and a run failing only one is inconclusive, not collapsed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from teb_vae.lag_attn.eval import masks
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "scalars"

#: Suffixes this pass cannot produce, and why. Recorded in the output rather than dropped, so a
#: reader can tell "not measured here" from "measured and zero".
NOT_APPLICABLE: Dict[str, str] = {
    "perm_loss": "permutation control not run by this analysis",
    "kld_shuffled": "permutation control not run by this analysis",
    "kld_shuffled_ratio": "permutation control not run by this analysis",
    "feat_loss_shuffled": "permutation control not run by this analysis",
    "shuffle_penalty": "permutation control not run by this analysis",
}

#: Thresholds for the joint collapse verdict. **Not** configurable, for the same reason
#: ``cross_subgroup``'s $\alpha$ is not: an operator who could raise these could make any run
#: read ``healthy``, and the verdict is promoted to the top of ``summary.json`` as the run's
#: headline conclusion. An earlier form read them from ``eval_config['collapse_thresholds']``,
#: a key ``config_schema.VALID_KEYS`` rejects -- so the branch was unreachable and the claim
#: that they were overridable was false in both directions at once.
#:
#: Chosen a priori and worth recalibrating against the first genuinely trained checkpoint; that
#: is an edit here, recorded in the diff, rather than a per-run knob.
DEFAULT_COLLAPSE_THRESHOLDS: Dict[str, float] = {"pred_gap": 1e-4, "kld_raw": 1e-3}

#: Metrics whose pooled value is a **root** of a pooled ratio, not a ratio. They are accumulated
#: as $\left(\sum m\,v^2,\ \sum m\right)$ and rooted once, at the end, because
#: $\mathrm{mean}\sqrt{x} \ne \sqrt{\mathrm{mean}\,x}$: by Jensen's inequality the mean of the
#: per-sample roots is *always* the smaller of the two, so averaging finished per-sample RMS
#: values reads systematically low against the single pooled root ``task.py`` logs -- about
#: $-9\%$ on realistic mask densities, in the direction that flatters the model.
RMS_METRICS: tuple = ("delta_mu_rms", "mu_post_prior_gap_rms")


#: Every metric the task logs, mirroring ``trainer.py::_METRIC_SUFFIXES``. Pinned equal to it by
#: a test -- see the module docstring for why it is mirrored rather than imported.
METRIC_SUFFIXES: tuple = (
    "total_loss", "main_loss", "feat_loss", "base_loss",
    "kld_loss", "kld_raw", "kld_train", "kld_active_frac",
    "perm_loss", "kld_shuffled", "kld_shuffled_ratio",
    "feat_loss_shuffled", "shuffle_penalty",
    "mean_logvar_full", "mean_logvar_base",
    "pred_gap", "delta_mu_rms", "mu_post_prior_gap_rms",
    "kld_beta", "lambda_full", "lambda_base",
    "lag_smoothness",
    "mu_prior_sat_frac", "delta_mu_sat_frac",
)


class _PooledAccumulator:
    r"""Accumulate a mask-weighted sum and its denominator across batches.

    A pooled mean over the split is *not* the mean of the per-batch means unless every batch has
    the same mask density -- the last batch of a split is routinely short, and gaps are not
    evenly spread. Carrying the numerator and denominator separately is what makes the final
    number equal to what ``compute_loss`` would report had the whole split been one batch.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def add(self, value: float, weight: float) -> None:
        """Add one batch's contribution.

        Args:
            value: The batch's mean.
            weight: The batch's denominator, i.e. how much that mean is worth.
        """
        if not np.isfinite(value) or weight <= 0:
            return
        self.total += float(value) * float(weight)
        self.weight += float(weight)

    def mean(self) -> float:
        """The pooled mean, or ``NaN`` when nothing contributed."""
        return self.total / self.weight if self.weight > 0 else float("nan")


def collapse_verdict(
    pred_gap: float, kld_raw: float, thresholds: Dict[str, float]
) -> Dict[str, Any]:
    r"""Combine the two collapse components into one named verdict.

    A collapsed run has $L_{\mathrm{base}} - L_{\mathrm{feat}}$ near zero **and** ``kld_raw``
    near zero together: the source neither changed the forecast nor moved the posterior. Either
    one alone is inconclusive and says so --

    * a near-zero KL with a real ``pred_gap`` is a decoder using a latent the KL under-reports;
    * a near-zero ``pred_gap`` with a real KL is a posterior carrying information the decoder
      does not act on.

    Both are findings worth chasing, and neither is the collapse the shipped config warns about.

    Args:
        pred_gap: The pooled $L_{\mathrm{base}} - L_{\mathrm{feat}}$.
        kld_raw: The pooled un-floored KL over its support.
        thresholds: Per-component near-zero thresholds.

    Returns:
        The verdict, its components, and the thresholds it was reached under.
    """
    gap_flat = bool(np.isfinite(pred_gap) and abs(pred_gap) < thresholds["pred_gap"])
    kld_flat = bool(np.isfinite(kld_raw) and abs(kld_raw) < thresholds["kld_raw"])

    if gap_flat and kld_flat:
        verdict, detail = "collapsed", (
            "both pred_gap and kld_raw are near zero: the source neither changed the forecast "
            "nor moved the posterior"
        )
    elif gap_flat:
        verdict, detail = "inconclusive", (
            "pred_gap is near zero but kld_raw is not: the posterior carries information the "
            "decoder does not act on"
        )
    elif kld_flat:
        verdict, detail = "inconclusive", (
            "kld_raw is near zero but pred_gap is not: the decoder is using a latent the KL "
            "under-reports"
        )
    else:
        verdict, detail = "healthy", "both components are away from zero"

    return {
        "verdict": verdict,
        "detail": detail,
        "pred_gap": float(pred_gap),
        "kld_raw": float(kld_raw),
        "pred_gap_near_zero": gap_flat,
        "kld_raw_near_zero": kld_flat,
        "thresholds": dict(thresholds),
    }


def run_scalar_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit the full test-split metric table under the checkpoint's own objective.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: Unused here -- this pass is uncapped by construction, since a scalar table is one
            row and a capped one would not reconcile with training.

    Returns:
        The metric table, the not-applicable map, the joint collapse verdict, and the $\beta$
        provenance the table's ``kld_beta`` row was built from.
    """
    del probe  # see the docstring: this pass never subsamples

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    # Resolved once, from the checkpoint's own schedule and epoch. It is a constant of the run,
    # not a per-batch quantity, and resolving it here means a schedule this pipeline cannot read
    # fails before the first forward rather than after the whole split has been scored.
    objective = runner.objective
    kld_beta = objective.effective_beta()

    accumulators: Dict[str, _PooledAccumulator] = {}
    n_samples = 0
    n_batches = 0

    for batch in runner.iter_batches(loader, max_samples=eval_config.get("max_samples")):
        outputs = runner.forward(batch)
        # beta=0 so total_loss is the unweighted sum of reported terms rather than the one
        # number that silently depends on which epoch the checkpoint came from.
        loss = runner.compute_loss(batch, outputs, beta=0.0)

        weight = get_field(batch, "weight")
        seq_len = int(outputs["mu_prior"].shape[1])
        batch_size = int(outputs["mu_prior"].shape[0])

        # Each metric is pooled against the denominator it was reduced over, so the split-level
        # mean equals what one giant batch would have produced.
        feature_weight = float(
            masks.feature_mask(
                runner.model, weight, batch_size, seq_len, device=outputs["mu_full"].device
            ).sum()
        )
        latent_weight = float(
            masks.kld_mask(
                runner.model, weight, batch_size, seq_len, device=outputs["mu_prior"].device
            ).sum()
        )

        values = _batch_scalars(runner, outputs, loss, kld_beta=kld_beta)
        for name, value in values.items():
            pooled_weight = _denominator_for(name, feature_weight, latent_weight, batch_size)
            accumulators.setdefault(name, _PooledAccumulator()).add(value, pooled_weight)

        # The two RMS diagnostics are pooled from their own numerator and denominator rather
        # than from a per-batch mean, so the root is taken once over the whole split. Fed here
        # rather than through `_batch_scalars` because the mask sums are the batch's, and a
        # value whose denominator came from `_denominator_for` would be pooled against a mask
        # built somewhere else.
        for name, (sum_squares, mask_sum) in _residual_diagnostics(
            runner, outputs, weight
        ).items():
            accumulators.setdefault(name, _PooledAccumulator()).add(
                sum_squares / mask_sum if mask_sum > 0.0 else float("nan"), mask_sum
            )

        n_samples += batch_size
        n_batches += 1

    table: Dict[str, Any] = {}
    not_applicable: Dict[str, str] = {}
    for suffix in METRIC_SUFFIXES:
        if suffix in accumulators:
            pooled = accumulators[suffix].mean()
            table[suffix] = float(np.sqrt(pooled)) if suffix in RMS_METRICS else pooled
        elif suffix in NOT_APPLICABLE:
            not_applicable[suffix] = NOT_APPLICABLE[suffix]
        else:
            not_applicable[suffix] = "not produced by this pass"

    verdict = collapse_verdict(
        table.get("pred_gap", float("nan")), table.get("kld_raw", float("nan")),
        DEFAULT_COLLAPSE_THRESHOLDS,
    )

    frame = pd.DataFrame(
        [{"metric": name, "value": value} for name, value in sorted(table.items())]
    )
    frame.to_csv(directory / "test_metrics.csv", index=False)
    with open(directory / "not_applicable.json", "w", encoding="utf-8") as handle:
        json.dump(not_applicable, handle, indent=2)

    # Named so neither can be read as the other. `kld_beta` in the table is the effective value,
    # matching what the task logs under that name; `kld_beta_configured` is the config constant,
    # which is only the effective value under a constant or absent schedule.
    beta_provenance = {
        "kld_beta_effective": float(kld_beta),
        "kld_beta_configured": float(objective.kld_beta),
        "beta_schedule": objective.beta_schedule,
        "checkpoint_epoch": objective.train_epoch,
    }

    logger.info(
        f"scalars: {len(table)} metric(s) over {n_samples} sample(s) in {n_batches} batch(es); "
        f"beta={kld_beta:.6g} at checkpoint epoch {objective.train_epoch}; "
        f"collapse verdict '{verdict['verdict']}'"
    )
    if verdict["verdict"] == "collapsed":
        logger.warning(f"collapse verdict: {verdict['detail']}")

    return {
        "n_samples": n_samples,
        "n_batches": n_batches,
        "metrics": table,
        "not_applicable": not_applicable,
        "collapse": verdict,
        "beta": beta_provenance,
    }


def _batch_scalars(
    runner: EvalRunner,
    outputs: Dict[str, torch.Tensor],
    loss: Dict[str, torch.Tensor],
    *,
    kld_beta: float,
) -> Dict[str, float]:
    r"""Assemble one batch's scalar metrics, matching the task's own metric dict.

    Built key by key rather than splatted from the loss dict. ``compute_loss`` echoes the
    ``likelihood`` string it was given, and a numeric consumer that receives it coerces it to a
    clean $0.0$ rather than raising -- ``runner.compute_loss`` already strips it, and this
    construction makes a reintroduction impossible.

    Args:
        runner: The loaded runner, for the objective's constants.
        outputs: The forward dict.
        loss: The loss dict, already stripped of ``likelihood``.
        kld_beta: The **effective** $\beta$ from
            :meth:`~teb_vae.lag_attn.eval.runner.Objective.effective_beta`, not the configured
            constant. Passed in rather than read off the objective here so the whole pass uses
            one resolved value, and so a caller cannot accidentally supply ``objective.kld_beta``
            without saying so.

    Returns:
        Metric name to float.
    """
    objective = runner.objective
    values: Dict[str, float] = {
        "feat_loss": float(loss["feat_loss"]),
        "base_loss": float(loss["base_loss"]),
        "kld_loss": float(loss["kld_loss"]),
        "kld_raw": float(loss["kld_raw"]),
        "kld_train": float(loss["kld_train"]),
        "kld_active_frac": float(loss["kld_active_frac"]),
        "mean_logvar_full": float(loss["mean_logvar_full"]),
        "mean_logvar_base": float(loss["mean_logvar_base"]),
        "lag_smoothness": float(loss["lag_smoothness"]),
        "pred_gap": float(loss["base_loss"] - loss["feat_loss"]),
        "mu_prior_sat_frac": float(outputs["mu_prior_sat_frac"]),
        "delta_mu_sat_frac": float(outputs["delta_mu_sat_frac"]),
        # Constants of the run rather than measurements, but the task logs them beside the
        # metrics and a table missing them cannot be lined up against a training row. `kld_beta`
        # is the schedule's value at the checkpoint's epoch, which is what task.py logs under
        # this name -- see the module docstring.
        "kld_beta": float(kld_beta),
        "lambda_full": float(objective.lambda_full),
        "lambda_base": float(objective.lambda_base),
    }

    # The training totals, reassembled under the checkpoint's own weights. `total_loss` here is
    # the perm-free main loss, which is what `main_loss` means in the task. The KL term carries
    # the effective beta, not the configured constant: under the shipped linear_warmup the two
    # differ by a factor of 100, and this total is the row a reader lines up against training.
    total = (
        objective.lambda_full * values["feat_loss"]
        + objective.lambda_base * values["base_loss"]
        + float(kld_beta) * values["kld_loss"]
        + objective.lambda_lag * values["lag_smoothness"]
    )
    values["total_loss"] = total
    values["main_loss"] = total
    return values


def _residual_diagnostics(
    runner: EvalRunner, outputs: Dict[str, torch.Tensor], weight: Optional[torch.Tensor]
) -> Dict[str, Tuple[float, float]]:
    r"""Reproduce the task's ``delta_mu_rms`` and ``mu_post_prior_gap_rms``, unrooted.

    Each uses the masking rules of the term it sits beside -- the first the feature window, the
    second the KL's own support -- so the numbers are comparable with what training logged rather
    than merely similarly named. Two things that makes concrete, both of which
    ``task.py::_compute_residual_diagnostics`` does and neither of which is optional:

    **The batch weight is folded into both masks.** ``masks.feature_mask`` and ``masks.kld_mask``
    skip the validity factor entirely when handed ``weight=None``, so omitting it counts the
    steps over gaps in the recording -- steps whose $\delta\mu$ and whose posterior-prior gap the
    loss never scored -- and divides by a denominator inflated by the same steps. The pooled
    denominators twelve lines up in :func:`run_scalar_analysis` already fold it in, so a
    ``None`` here would additionally pool a numerator and a denominator built over two different
    supports.

    **Neither is reduced to a finished RMS.** Returning
    $\left(\sum m\,v^2,\ \sum m\right)$ lets the caller pool across batches and take the root
    once, which is what ``task.py`` computes:

    $$\mathrm{rms} = \sqrt{\frac{\sum m\,v^2}{\max\left(\sum m,\ 1\right)}}$$

    The per-sample roots that :func:`~teb_vae.lag_attn.eval.metrics.residual_usage` and
    :func:`~teb_vae.lag_attn.eval.metrics.posterior_drift` return are a different quantity, and
    averaging them is strictly smaller by Jensen -- see :data:`RMS_METRICS`. Those functions keep
    their per-sample form: ``analyses/residual.py`` wants exactly that, one row per recording.

    Args:
        runner: The loaded runner.
        outputs: The forward dict.
        weight: Per-step validity $(B, T)$, or ``None`` when the batch carries none.

    Returns:
        Metric name to ``(masked sum of squares, mask sum)``.
    """
    delta = outputs["delta_mu_src"]
    batch_size, seq_len = int(delta.shape[0]), int(outputs["mu_prior"].shape[1])
    anchors = int(delta.shape[1]) - int(runner.model.horizon)

    feature_mask = masks.feature_mask(
        runner.model, weight, batch_size, seq_len, device=delta.device
    )
    latent_mask = masks.kld_mask(
        runner.model, weight, batch_size, seq_len, device=delta.device
    )

    # Channel-summed energy, then masked: the sum over $c$ inside the root is what makes this
    # `delta_mu_rms` rather than a per-channel RMS, and is the form task.py reduces.
    delta_energy = (delta[:, :anchors] ** 2).sum(dim=-1)
    gap_energy = ((outputs["mu_post"] - outputs["mu_prior"]) ** 2).sum(dim=-1)

    return {
        "delta_mu_rms": (
            float((delta_energy * feature_mask.squeeze(-1)).sum()),
            float(feature_mask.sum()),
        ),
        "mu_post_prior_gap_rms": (
            float((gap_energy * latent_mask).sum()),
            float(latent_mask.sum()),
        ),
    }


def _denominator_for(
    name: str, feature_weight: float, latent_weight: float, batch_size: int
) -> float:
    """Return the pooling weight for one metric, matching the support it was reduced over.

    Args:
        name: The metric name.
        feature_weight: The feature mask's sum for this batch.
        latent_weight: The KL mask's sum for this batch.
        batch_size: Samples in the batch.

    Returns:
        The weight this batch's value carries in the split-level pooled mean.
    """
    # The two RMS diagnostics are absent by design: they are pooled from their own numerator and
    # denominator in `run_scalar_analysis`, so routing them through here would give them a mask
    # sum built independently of the one their numerator was masked with.
    if name in ("kld_loss", "kld_raw", "kld_train"):
        return latent_weight
    if name in ("kld_beta", "lambda_full", "lambda_base", "kld_active_frac",
                "mu_prior_sat_frac", "delta_mu_sat_frac", "lag_smoothness"):
        # Per-sample or per-run quantities: weighting them by the mask would let a gappy batch
        # count for less on a number that has nothing to do with the mask.
        return float(batch_size)
    return feature_weight
