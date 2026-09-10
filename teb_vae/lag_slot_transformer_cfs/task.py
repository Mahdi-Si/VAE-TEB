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
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Tuple

import torch

from teb_vae.lag_attn_cfs.task import DENSE_STAGES, SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask

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


__all__ = ["TASK_METRIC_SUFFIXES", "SeqVaeLagResidualTrfCfsTask"]
