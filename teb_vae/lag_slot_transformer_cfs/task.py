r"""The training task: one diamond, and one step written out rather than inherited.

Two parents, each supplying what its own package added to the shared task:

* :class:`~teb_vae.lag_attn_cfs.task.SeqVaeLagAttnCfsTask` -- the anchor tiling's derived phase,
  the stage-to-geometry resolution that puts a per-segment phase and the training stride into a
  training forward and the dense range into an evaluation one, and the five-argument forward
  assembly;
* :class:`~teb_vae.lag_attn_transformer_rws.task.SeqVaeLagAttnTrfRwsTask` -- the step-granular
  learning-rate ramp a pre-normalised attention stack needs in its first few hundred optimizer
  steps, which an epoch-granularity schedule cannot address at all.

Everything else -- the optimizer, the divergence ramp, the spike-breaker wiring, the pre-clip
gradient-norm logging and the checkpoint contract -- is the shared task's, reached through both
parents at once.

**What is written here, and why none of it could be inherited.** The shared step is the family's,
and three of the tensors it names do not exist on this architecture:

* it reads ``delta_mu_sat_frac``, which is the lag-attentive posterior's bound. The bounds here are
  the two residual channels, and they saturate independently;
* its latent-gap readout pairs a dense stored-grid support with a latent produced at every step.
  This architecture's latent exists at the decoded anchors only, so the two would not even
  broadcast;
* it runs the permutation control through the lag attention and the head-structured posterior,
  neither of which is built.

So the step is written out. It is the same shape as the shared one -- forward, objective, metrics,
one refusal on a name collision -- and it calls the same hyperparameters by the same names, which is
what keeps a run of this model readable beside a run of its siblings.

**The diagnostic page is not enabled for this model, and that is a decision rather than an
omission.** The shared page's lag rows draw an attention matrix and a divergence-by-lag map, and
this architecture computes neither: there is no attention distribution, and no nonnegative per-lag
allocation of the divergence exists for it. The plotter is opt-in at the configuration, so a page is
simply not requested.

lean-limit: no per-epoch diagnostic figure for this architecture; replace with a page whose lag rows
draw proposal suppression and the cancellation ratio when the evaluation package's lag readouts have
been run on a real arm and their layout is settled.

**Two things the training loop measures that the shared step does not.** The first is the
predictive validation monitor: with ``validation_mc_draws`` set, every dense validation batch is
also scored as the unweighted $K$-draw predictive mixture of both branches,

$$D^{(K)}_t = -\operatorname{logsumexp}_{k}\bigl(-D^{(k)}_t\bigr) + \log K,$$

under a noise bank keyed on the batch's segment identities and the run seed, so the number a
checkpoint is selected on is the estimator the offline evaluation reports it under rather than the
one-draw weighted objective, and two validation passes at the same weights report the same value.
The second is the clip fraction: the shared hook logs whether *one sampled step* exceeded the clip,
and the epoch's CSV row therefore holds a zero or a one; here every optimizer step is counted and
the row holds the fraction of the epoch's steps the clip bound on.
"""
from __future__ import annotations

import hashlib
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn_cfs.task import (
    _KEY_SEPARATOR,
    DENSE_STAGES,
    SeqVaeLagAttnCfsTask,
    _as_float,
    _as_key,
)
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask
from teb_vae.lag_slot_transformer_cfs.nets.objective import _all_reduce_sum

#: Metric names this task adds to the objective's own surface, so a driver can track them by name
#: without importing the step. Written out rather than derived from a run, because a tracked name
#: that stops being produced should fail a test rather than leave an empty column.
#:
#: **Two of them are arm-dependent, and the tracked list is deliberately the union.** The scale
#: cancellation ratio is absent on the mean-only arm and both cancellation ratios are absent on a
#: comparator arm whose fusion normalises over lags rather than summing. A tracked name a run never
#: emits costs an empty column; a name a run emits and nothing tracks costs the number itself, and
#: this list exists to prevent the second.
TASK_METRIC_SUFFIXES: Tuple[str, ...] = (
    "main_loss",
    "mu_prior_sat_frac",
    "residual_mu_sat_frac",
    "residual_logsigma_sat_frac",
    "mu_post_prior_gap_rms",
    "cancellation_ratio_mean",
    "cancellation_ratio_scale",
    "lag_available_frac",
)

#: The hyperparameter that switches the predictive validation monitor on: the draw count $K$, or
#: ``None`` for the legacy validation surface. Named once, because the driver reads the same key
#: off the configuration and hands it to the task by this name.
VALIDATION_MC_DRAWS_KEY = "validation_mc_draws"

#: The monitor's three columns, produced on the dense stages alone and only when the draw count is
#: set. Kept apart from :data:`TASK_METRIC_SUFFIXES` for both reasons: a training batch never
#: carries them, and a legacy run must log exactly the columns it always did, so the driver tracks
#: these under ``val/`` only when the configuration asked for them.
VALIDATION_MONITOR_SUFFIXES: Tuple[str, ...] = (
    "pred_nll_full_mc",
    "pred_nll_base_mc",
    "pred_gap_mc",
)


def recording_grouped_totals(
    values: torch.Tensor, contributing: torch.Tensor, recordings: Sequence[Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Sum of per-recording means of a per-anchor score, and the number of recordings summed.

    Each recording's mean is over its own scored anchors in this batch, whichever samples they
    sit in; recordings are then weighted equally,

    $$\sum_{g} \frac{\sum_{b \in g}\sum_a c_{b,a}\, v_{b,a}}{\sum_{b \in g}\sum_a c_{b,a}},$$

    so a recording that contributed several segments to the batch does not outweigh one that
    contributed one. The sum and the count are returned rather than the mean so a caller can
    reduce them across ranks first and divide once.

    Args:
        values: A per-anchor score $(B, A)$.
        contributing: The $0/1$ scored-anchor indicator $(B, A)$.
        recordings: One recording identity per sample, in whatever form the batch carries it.

    Returns:
        ``(sum of recording means, recording count)``, both 0-d ``float64`` tensors on the
        values' device. Both are zero when no anchor was scored.

    Raises:
        ValueError: If the identity list is not one per sample.
    """
    if len(recordings) != int(values.shape[0]):
        raise ValueError(
            f"{len(recordings)} recording identities for a batch of {int(values.shape[0])} "
            f"samples; the two must be one per sample."
        )
    weights = contributing.to(torch.float64)
    per_sample_sum = (values.to(torch.float64) * weights).sum(dim=1)
    per_sample_count = weights.sum(dim=1)
    # One slot per distinct recording, in first-seen order; the GUID is normalised the way the
    # tiling phase normalises it so two spellings of one recording share a slot.
    slots: Dict[bytes, int] = {}
    slot_of = torch.tensor(
        [slots.setdefault(_as_key(recording), len(slots)) for recording in recordings],
        dtype=torch.long,
        device=values.device,
    )
    sums = torch.zeros(len(slots), dtype=torch.float64, device=values.device)
    counts = torch.zeros(len(slots), dtype=torch.float64, device=values.device)
    sums.index_add_(0, slot_of, per_sample_sum)
    counts.index_add_(0, slot_of, per_sample_count)
    scored = counts > 0
    means = sums[scored] / counts[scored]
    return means.sum(), scored.sum().to(torch.float64)


class SeqVaeLagResidualTrfCfsTask(SeqVaeLagAttnCfsTask, SeqVaeLagAttnTrfRwsTask):
    r"""Lightning task for
    :class:`~teb_vae.lag_slot_transformer_cfs.nets.model.SeqVaeLagResidualTrfCfs`.

    Defines a step, a latent-gap readout and two refusals. The tiling phase, the five-argument
    forward assembly and the target construction come from the first base; the step-granular
    learning-rate ramp from the second; and the optimizer, the divergence ramp, the spike breaker
    and the checkpoint contract from the shared ancestor both derive from.

    ``main_loss`` keeps its exact, unprefixed name, which is what the loss-spike breaker watches;
    the framework falls back to the returned loss when that key is missing, silently.
    """

    def __init__(
        self,
        base_model: Any,
        *,
        validation_mc_draws: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialize the task.

        Args:
            base_model: The net to wrap.
            validation_mc_draws: The draw count $K$ of the predictive validation monitor, or
                ``None`` to leave validation on the legacy one-draw weighted surface. Saved as a
                hyperparameter so a resumed run monitors what it monitored before it stopped; the
                driver applies the configured value after construction, by the route the seed
                takes.
            **kwargs: Every other keyword, forwarded to the inherited constructor unchanged.

        Raises:
            ValueError: If the draw count is set and is not a positive integer.
        """
        super().__init__(base_model, **kwargs)
        if validation_mc_draws is not None and (
            isinstance(validation_mc_draws, bool) or int(validation_mc_draws) < 1
        ):
            raise ValueError(
                f"{VALIDATION_MC_DRAWS_KEY}={validation_mc_draws!r} must be a positive integer "
                f"or null; the monitor is a mixture over that many draws and a mixture over "
                f"none is not a score."
            )
        self.save_hyperparameters(VALIDATION_MC_DRAWS_KEY)
        # The clip counters. Device tensors rather than Python numbers so the exceedance
        # indicator never forces a host synchronisation on the training step; reset at every
        # training epoch start and read at its last batch.
        self._clip_exceeded: Optional[torch.Tensor] = None
        self._clip_counted: int = 0

    @property
    def forecast_rows(self) -> Callable[..., None]:
        r"""The page's forecast rows, bound to this net's channel facts, tiling and weights.

        The causal parent's seven bindings, plus four this cell needs and that cell does not.
        **The extra four exist because this objective weights the block score** -- by channel,
        stating a ratio between the two stored target blocks, and by horizon step under a decaying
        half-life -- and the parent's per-window score row applies neither. Left inherited, the
        row would draw curves whose units are not the ``nll_base_block``, ``nll_full_block`` and
        ``pred_gap`` printed in the same figure's title and plotted on the training curve: a
        diagnostic disagreeing with the run it diagnoses about the very number it is drawing.

        The gather and the pooled validity are bound as the model's own **bound methods** rather
        than reconstructed on the page, so the block scored on the row is the block the objective
        scored. Both are read with ``getattr`` on the weights for the reason the model reads them
        that way: each is a buffer where it exists and absent otherwise, and ``None`` means the
        score skips the multiplication rather than multiplying by ones.

        Returns:
            A callable taking one
            :class:`~teb_vae.lag_attn_rws.sample_page.ForecastRowInputs` and drawing into it.
        """
        from teb_vae.lag_slot_transformer_cfs.sample_page import residual_forecast_rows

        model = self.orig_model
        gate = model.target_gate
        # The forecast clock's tau in seconds comes from the resolved budget, as the input clocks
        # do: the net stamps the per-channel step shifts and nothing from which tau can be
        # recovered. It reaches the page as a statement on the axis, not as a shift of anything.
        budget = self.warmup_budget
        return partial(
            residual_forecast_rows,
            keep_index=None if gate is None else gate.keep_index,
            block_split=int(model.TARGET_BLOCK_SPLIT),
            training_stride=int(model.anchor_stride),
            likelihood=str(self.hparams.get("likelihood", "gaussian_nll")),
            coverage_floor=float(model.coverage_floor),
            target_forecast_shift=model.target_forecast_shift,
            forecast_clock_delay_s=(
                None if budget is None else budget.target_forecast_clock_delay_s
            ),
            forecast_target=model._build_forecast_target,
            scored_weight=model.scored_weight,
            channel_weight=getattr(model, "target_channel_weight", None),
            horizon_weight=getattr(model, "horizon_weight", None),
        )

    def _mu_gap_rms(
        self, forward_outputs: Dict[str, torch.Tensor], weight: torch.Tensor
    ) -> torch.Tensor:
        r"""Masked root-mean-square of the latent mean gap, on the anchor axis.

        $$\sqrt{\frac{\sum_{b,a} c_{b,a} \lVert \mu^q_{b,a} - \mu^p_{b,a}\rVert^2}
                     {\sum_{b,a} c_{b,a}}}.$$

        Overridden against both inherited versions, which build a dense stored-grid support and
        multiply it into a latent produced at every step. Here the latent exists at the decoded
        anchors only, so the inherited expression would not broadcast -- and the fix is not to
        scatter the support back but to stop converting between two axes at all. The support is
        the reconstruction's own contributing set, which is what keeps this number and the
        divergence printed beside it averaged over one anchor set rather than two.

        Args:
            forward_outputs: The net's forward dict, carrying the anchor set and both latent means.
            weight: Per-step validity $(B, T)$, as the loader delivered it.

        Returns:
            A scalar tensor.
        """
        with torch.no_grad():
            model = self.orig_model
            mask, _coverage = forecast_mask(
                # The forecast clock's pooled validity, so this support is the one the objective
                # scored under; the identity object on the stored clock.
                model.scored_weight(weight),
                model.geometry,
                coverage_floor=model.coverage_floor,
                anchors=forward_outputs["anchor_index"],
                anchor_valid=forward_outputs["anchor_valid"],
            )
            support = contributing_anchors(mask)
            gap_sq = (
                (forward_outputs["mu_post"] - forward_outputs["mu_prior"]) ** 2
            ).sum(dim=-1)
            total = support.sum()
            if float(total) <= 0.0:
                return torch.zeros((), device=gap_sq.device, dtype=gap_sq.dtype)
            return torch.sqrt((gap_sq * support).sum() / total)

    def _added_metrics(
        self,
        inputs: Tuple[Any, ...],
        forward_outputs: Dict[str, torch.Tensor],
        weight: torch.Tensor,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        """No source control on the training loop, and the reason is worth stating.

        The inherited readout encodes a zeroed source stream through the source pathway and
        measures how far the divergence survives it. That pathway holds no parameters here, so
        there is nothing to encode: the analogous control is to re-run the **fusion** under a
        substituted source, which is a real intervention and belongs with the others in the
        evaluation package rather than as a lone copy inside a training step.

        The one control that is free here would say nothing: with every selector off the
        divergence is exactly zero by construction, so a column reporting it could not vary and
        would measure the invariant rather than the model.

        Args:
            inputs: The five positional arguments the forward was given.
            forward_outputs: That forward's dict.
            weight: Decimated validity signal $(B, T)$.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            An empty mapping.
        """
        return {}

    def _should_run_perm(self, batch_size: int, stage: str) -> bool:
        """Never, on this architecture.

        The inherited permutation control rebuilds the full branch by re-attending a deranged
        source state through the lag attention and the head-structured posterior. Neither module
        exists here. The control itself is meaningful for this model -- a cross-recording source is
        one of the specificity checks -- but it has to be rebuilt against the fusion, and it lives
        with the evaluation package where its matched scoring and its stratification live too.

        Args:
            batch_size: Samples in the batch.
            stage: The stage the step is running in.

        Returns:
            ``False``.
        """
        return False

    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the forward, score it, and assemble the reported surface.

        The shared step's shape, written out because three of the tensors it names do not exist on
        this architecture. What is *not* changed: the hyperparameters are read by the same names,
        the objective is reached through ``orig_model`` so the data-dependent reductions stay eager,
        ``main_loss`` is detached under exactly that name for the spike breaker, and a readout
        colliding with an objective metric is refused rather than merged.

        Args:
            batch: A batch from the data module.
            batch_idx: Index of the current batch.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            ``(loss, metrics)``.

        Raises:
            ValueError: If a readout below reuses a name the objective already reports, which
                would replace that column in the metric history and in the tracking backend with
                no error.
        """
        # The stage travels on the instance for exactly the length of one step, which is what lets
        # the inherited input builder resolve the anchor geometry without taking a stage argument
        # -- a signature the plotting seam and every sibling's test share.
        self._stage = stage
        try:
            inputs = self._build_forward_inputs(batch)
            target_features, weight = self._build_raw_target(batch)
            forward_outputs = self.model(*inputs)

            loss_metrics = self.orig_model.compute_loss(
                forward_outputs,
                target_features,
                weight=weight,
                beta=self._resolve_beta(self.current_epoch),
                beta_prior=float(self.hparams.get("beta_prior", 0.0)),
                lambda_full=float(self.hparams.get("lambda_full", 1.0)),
                lambda_base=float(self.hparams.get("lambda_base", 1.0)),
                likelihood=str(self.hparams.get("likelihood", "gaussian_nll")),
                free_bits=float(self.hparams.get("free_bits", 0.0)),
            )["metrics"]
        finally:
            self._stage = DENSE_STAGES[0]

        main_loss = loss_metrics["total_loss"]
        metrics: Dict[str, Any] = dict(loss_metrics)
        # Unprefixed and detached: the breaker watches this exact name and falls back to the
        # returned loss, silently, if it is missing.
        metrics["main_loss"] = main_loss.detach()
        metrics.update(self._residual_readouts(forward_outputs, weight))

        # The predictive monitor, on the dense stages only: a training batch is tiled and its
        # score would be over a different anchor set every epoch, and the monitor exists to be
        # compared across epochs. Absent entirely -- not zero -- when the draw count is unset, so
        # a legacy run logs exactly the columns it always did.
        draws = self.hparams.get(VALIDATION_MC_DRAWS_KEY)
        if draws is not None and stage in DENSE_STAGES:
            metrics.update(
                self._predictive_monitor(
                    batch, forward_outputs, target_features, weight, int(draws)
                )
            )

        added = self._added_metrics(inputs, forward_outputs, weight, stage)
        collisions = sorted(set(added) & set(metrics))
        if collisions:
            raise ValueError(
                f"_added_metrics returned {collisions}, which the objective already reports. A "
                f"readout reusing an objective metric's name replaces it in the metric history "
                f"and in the tracking backend with no error, so the column keeps its meaning "
                f"across the family only if the name is new. Rename it."
            )
        metrics.update(added)
        return main_loss, metrics

    def _residual_readouts(
        self, forward_outputs: Dict[str, torch.Tensor], weight: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        r"""The four numbers only this architecture has, over the scored anchor set.

        The two saturation fractions say whether either residual bound is binding, and they are
        reported separately because the two bounds are configured separately and a bound that is
        always active is a mis-set hyperparameter rather than a guard.

        The cancellation ratio says how much of the per-lag proposal mass survives the sum. It is
        the one readout here that cannot be inferred from anything else in the surface: a
        divergence penalty on the summed update cannot discipline proposals that cancel, so a
        pathway arguing with itself and a pathway that has switched off look identical in every
        other column.

        The lag-availability fraction is the exposure denominator. A per-lag readout computed over
        anchors where most lags are out of range or still cold is a measurement of the schedule.

        Averaged over the anchors the objective scored, so each of these is on the same support as
        the divergence printed beside it.

        Args:
            forward_outputs: The net's forward dict.
            weight: Per-step validity $(B, T)$.

        Returns:
            The readouts, keyed for the metric surface.
        """
        with torch.no_grad():
            model = self.orig_model
            mask, _coverage = forecast_mask(
                model.scored_weight(weight),
                model.geometry,
                coverage_floor=model.coverage_floor,
                anchors=forward_outputs["anchor_index"],
                anchor_valid=forward_outputs["anchor_valid"],
            )
            support = contributing_anchors(mask)
            total = support.sum()

            def masked_mean(values: torch.Tensor) -> torch.Tensor:
                """Average a per-anchor readout over the scored anchors, or report zero."""
                if float(total) <= 0.0:
                    return torch.zeros((), device=values.device, dtype=values.dtype)
                return (values * support).sum() / total

            readouts = {
                "mu_prior_sat_frac": forward_outputs["mu_prior_sat_frac"],
                "residual_mu_sat_frac": forward_outputs["residual_mu_sat_frac"],
                "residual_logsigma_sat_frac": forward_outputs["residual_logsigma_sat_frac"],
                "mu_post_prior_gap_rms": self._mu_gap_rms(forward_outputs, weight),
                "lag_available_frac": masked_mean(
                    forward_outputs["lag_valid"].to(support.dtype).mean(dim=-1)
                ),
            }
            # Each cancellation channel is reported only where the arm produces it, matching the
            # forward and for the same reason. The scale channel is absent on the mean-only arm,
            # and BOTH are absent on a comparator arm whose fusion normalises over lags: nothing
            # there is summed, every weight is non-negative and they add to one, so a ratio would
            # be a constant wearing a diagnostic's name.
            for channel in ("mean", "scale"):
                key = f"cancellation_ratio_{channel}"
                if key in forward_outputs:
                    readouts[key] = masked_mean(forward_outputs[key])
        return readouts

    # ------------------------------------------------------------------
    # The predictive validation monitor
    # ------------------------------------------------------------------
    def validation_noise_generator(self, batch: Any, device: torch.device) -> torch.Generator:
        r"""The monitor's noise bank for one batch: a generator seeded from what the batch is.

        The seed is a digest of the run seed and, in batch order, each segment's recording and
        floored start time -- the same two identity fields the tiling phase is keyed on, and for
        the same reason: a value that is a function of the data and the run, and of nothing that
        happened to be drawn earlier in the process. Two validation passes at the same weights
        therefore score the same draws and report the same number, which is what lets the monitor
        rank checkpoints at all; a global draw would move with every training step in between.

        The bank is fixed at fixed batching. A different batch size or a different loader order
        composes different batches and therefore different digests; that is a property of the
        monitor, and the evaluation package's scorer is the one to use for comparisons that must
        survive a rebatching.

        Args:
            batch: A batch from the data module, carrying ``guid`` and ``epoch``.
            device: Where the latent parameters live; the generator is built there.

        Returns:
            A seeded generator on ``device``.
        """
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(int(self.hparams.get("seed", 0))).encode("utf-8"))
        guids = self._phase_field(batch, "guid")
        starts = self._phase_field(batch, "epoch")
        for guid, start in zip(guids, starts):
            digest.update(_KEY_SEPARATOR)
            digest.update(_as_key(guid))
            digest.update(_KEY_SEPARATOR)
            digest.update(str(int(_as_float(start) // 1)).encode("utf-8"))
        generator = torch.Generator(device=device)
        generator.manual_seed(int.from_bytes(digest.digest(), "big") % (2**63 - 1))
        return generator

    def _predictive_monitor(
        self,
        batch: Any,
        forward_outputs: Dict[str, torch.Tensor],
        target_features: torch.Tensor,
        weight: torch.Tensor,
        num_draws: int,
    ) -> Dict[str, torch.Tensor]:
        r"""Score both branches as $K$-draw predictive mixtures, recording-grouped, batch-global.

        The scorer is the evaluation package's own, so the monitor is the same estimator the
        offline headline reads: one $\epsilon^{(k)}$ per draw shared by both branches, the
        decoder invoked on each, the block scored **unweighted** on the objective's own mask, and
        the negative log of the average likelihood taken per anchor. With one draw it is the
        unweighted single-draw conditional score of that draw; the Jensen gap to the average of
        per-draw scores opens only from the second draw on.

        The reduction is the objective's: per-recording means are summed and counted locally,
        the sums and the count cross the process group in one packed all-reduce, and the division
        happens once, so every rank reports one number and a rank holding fewer recordings does
        not weigh as much as one holding more. Across batches the framework averages these
        per-batch values, which is the same epoch estimand every other ``val/`` column has.

        Args:
            batch: The batch, for its recording identities and the noise bank's key.
            forward_outputs: The matched forward's dict, for the two latent parameter sets, the
                anchor set and the persistence input.
            target_features: The declared target stream $(B, T, c_y)$; the labels.
            weight: Decimated validity signal $(B, T)$.
            num_draws: The draw count $K$.

        Returns:
            ``pred_nll_full_mc``, ``pred_nll_base_mc`` and ``pred_gap_mc`` (base minus full),
            each a 0-d tensor; all three zero when no anchor was scored anywhere.
        """
        # Imported here rather than at module load: the scorer sits in the evaluation package,
        # which a training process otherwise never imports, and the first validation step is
        # the first moment it is needed.
        from teb_vae.lag_slot_transformer_cfs.eval.predictive import matched_predictive_scores

        model = self.orig_model
        anchors = forward_outputs["anchor_index"]
        target = model._build_forecast_target(target_features, anchors)
        mask, _coverage = forecast_mask(
            model.scored_weight(weight),
            model.geometry,
            coverage_floor=model.coverage_floor,
            anchors=anchors,
            anchor_valid=forward_outputs["anchor_valid"],
        )
        scored = matched_predictive_scores(
            model,
            {
                "base": (forward_outputs["mu_prior"], forward_outputs["logvar_prior"]),
                "full": (forward_outputs["mu_post"], forward_outputs["logvar_post"]),
            },
            target,
            mask,
            likelihood=str(self.hparams.get("likelihood", "gaussian_nll")),
            num_samples=int(num_draws),
            generator=self.validation_noise_generator(batch, anchors.device),
            # The forward's own persistence tensor: target-only, identical for both branches
            # and every draw, and the same object both decoder calls of the forward received.
            persistence=forward_outputs.get("persistence"),
        )
        recordings = self._phase_field(batch, "guid")
        contributing = scored["base"].contributing
        full_sum, count = recording_grouped_totals(scored["full"].marginal, contributing, recordings)
        base_sum, _count = recording_grouped_totals(scored["base"].marginal, contributing, recordings)
        totals = _all_reduce_sum(torch.stack([full_sum, base_sum, count]))
        dtype = target.dtype
        n_recordings = float(totals[2])
        if n_recordings <= 0.0:
            zero = torch.zeros((), device=anchors.device, dtype=dtype)
            return {name: zero.clone() for name in VALIDATION_MONITOR_SUFFIXES}
        full = (totals[0] / n_recordings).to(dtype)
        base = (totals[1] / n_recordings).to(dtype)
        return {
            "pred_nll_full_mc": full,
            "pred_nll_base_mc": base,
            "pred_gap_mc": base - full,
        }

    # ------------------------------------------------------------------
    # The clip fraction, counted over every optimizer step
    # ------------------------------------------------------------------
    def _on_train_epoch_start_hook(self) -> None:
        """Reset the clip counters, so each epoch's fraction is over that epoch's steps alone."""
        super()._on_train_epoch_start_hook()
        self._clip_exceeded = None
        self._clip_counted = 0

    def _record_clip_exceedance(self, grad_norm: torch.Tensor, clip_val: float) -> torch.Tensor:
        r"""Count one optimizer step against the clip and return the running fraction.

        $$f = \frac{\#\{\text{steps with } \lVert g \rVert_2 > c\}}{\#\{\text{steps}\}},$$

        both counted from the start of the current training epoch.

        Args:
            grad_norm: This step's pre-clip gradient norm, a 0-d tensor.
            clip_val: The clip threshold $c$, known positive.

        Returns:
            The running fraction, a 0-d tensor on the norm's device.
        """
        exceeded = (grad_norm.detach() > float(clip_val)).to(grad_norm.dtype)
        self._clip_exceeded = exceeded if self._clip_exceeded is None else self._clip_exceeded + exceeded
        self._clip_counted += 1
        return self._clip_exceeded / float(self._clip_counted)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        r"""Log the pre-clip gradient norm on the sampled steps and the clip fraction over all.

        Overrides the shared hook rather than extending it, because the shared one logs the
        exceedance **indicator** of the sampled step under the name the CSV reads: the metric
        history is collected from the bare key during validation, before the training epoch is
        reduced, so its row held a zero or a one and only their mean over epochs estimated the
        fraction. Here the norm is computed on every optimizer step -- one fused reduction over
        the gradients, negligible beside the backward that produced them -- and counted against
        the threshold; what is logged under ``train/grad_clip_frac`` is the running fraction of
        this epoch's steps, on the same sampled cadence as the norm and always on the epoch's
        last batch, where the running value is exactly the epoch's. The bare key the CSV reads
        therefore holds the fraction of the epoch's optimizer steps the clip bound on.

        ``train/grad_norm`` keeps the shared column's meaning and cadence. Both are omitted when
        the trainer configures no positive clip, as before: a fraction against no threshold
        answers no question. A batch the spike breaker skipped has a norm near zero and counts
        as a step the clip did not bind on; ``train/spike_skipped`` is the column that says so.

        Args:
            optimizer: The optimizer Lightning is about to step; unused, the gradients are read
                off the module's own parameters.
        """
        grads = [parameter.grad.detach() for parameter in self.parameters() if parameter.grad is not None]
        if not grads:
            return
        grad_norm = torch.nn.utils.get_total_norm(grads)
        trainer = self.trainer
        clip_val = trainer.gradient_clip_val
        clipping = clip_val is not None and float(clip_val) > 0.0
        fraction = self._record_clip_exceedance(grad_norm, float(clip_val)) if clipping else None

        if not (trainer.is_last_batch or trainer.global_step % self.GRAD_NORM_LOG_EVERY_N_STEPS == 0):
            return
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True, logger=True)
        if fraction is not None:
            # A running fraction has no meaningful epoch mean, so it is logged per step alone;
            # the value the CSV samples at the epoch's last batch is the epoch's own fraction.
            self.log("train/grad_clip_frac", fraction, on_step=True, on_epoch=False, logger=True)


__all__ = [
    "TASK_METRIC_SUFFIXES",
    "VALIDATION_MC_DRAWS_KEY",
    "VALIDATION_MONITOR_SUFFIXES",
    "SeqVaeLagResidualTrfCfsTask",
    "recording_grouped_totals",
]
