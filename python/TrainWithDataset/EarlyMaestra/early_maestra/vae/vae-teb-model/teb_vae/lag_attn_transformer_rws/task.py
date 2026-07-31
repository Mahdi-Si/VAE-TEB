r"""The training task: the objective, the metrics, and a step-granular learning-rate warm-up.

Everything about turning a batch into a loss is inherited unchanged from
:class:`~teb_vae.lag_attn_rws.task.SeqVaeLagAttnRwsTask`, and that is the point rather than an
economy. Two architectures are only comparable if they optimise the same thing, so the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}},
$$

its $\beta$ schedule, its metric surface, the validation-only permutation control, the spike-breaker
wiring, the pre-clip gradient-norm logging and the checkpoint contract are all the same code, not a
copy of it. That parent names no model class, so wrapping a different net needs no change to any of
it.

The one addition is :meth:`SeqVaeLagAttnTrfRwsTask.build_lr_scheduler`, and it exists because a
pre-normalised attention stack is fragile in exactly the first few hundred optimizer steps -- a
window an epoch-granularity schedule cannot address at all when one epoch is thousands of steps.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask


class SeqVaeLagAttnTrfRwsTask(SeqVaeLagAttnRwsTask):
    r"""Lightning task for
    :class:`~teb_vae.lag_attn_transformer_rws.nets.model.SeqVaeLagAttnTrfRws`.

    Adds exactly one method. The absence of everything else is deliberate and load-bearing: no
    ``training_step`` (the inherited one runs the config-gated loss-spike breaker), no
    ``configure_optimizers`` (the framework's calls :meth:`build_lr_scheduler`, which is the seam
    used here), no constructor (a second keyword schema for the same objective could only drift
    from the first).
    """

    def build_lr_scheduler(
        self, optimizer: Optimizer
    ) -> Optional[Union[_LRScheduler, Dict[str, Union[_LRScheduler, str, int]]]]:
        r"""Build the learning-rate schedule, with an optional step-granular warm-up.

        At ``lr_warmup_steps = 0`` -- the default, and what a config that never sets the key
        resolves to -- this delegates to the framework, so the inherited epoch-granularity
        ``lr_warmup_epochs`` path stays reachable from configuration and costs no code here.

        Above zero it returns a single ``LambdaLR`` stepped once per optimizer step, carrying both
        the ramp and the milestone decay:

        $$
        \mathrm{factor}(s) = \min\!\left(1, \frac{s+1}{S_{\mathrm{warm}}}\right)
          \cdot \gamma^{\,\left|\{m : m \le s\}\right|},
        $$

        where the milestones $m$ are the configured **epoch** milestones converted to steps
        through ``trainer.estimated_stepping_batches / trainer.max_epochs`` -- that property
        reports the optimizer steps for the whole run, not per epoch, and already accounts for
        gradient accumulation and the device count.

        One ``LambdaLR`` rather than a ``SequentialLR`` of a ramp and a ``MultiStepLR``, for two
        reasons that are each sufficient. ``LinearLR`` rejects ``start_factor=0.0``, so it cannot
        express a ramp that starts from nothing -- and the framework's epoch path starts at $0.1$,
        a tenfold discontinuity at step zero. And ``SequentialLR`` restarts the second scheduler's
        step counter at the switch, so its milestones need a compensating shift; a single lambda
        reads absolute step indices and needs none.

        Args:
            optimizer: The optimizer the schedule drives.

        Returns:
            ``None`` or whatever the base returns when no step warm-up is configured; otherwise a
            Lightning scheduler dict at ``interval: "step"``.

        Raises:
            ValueError: If ``lr_warmup_steps`` is negative.
        """
        warmup_steps = int(getattr(self.hparams, "lr_warmup_steps", 0) or 0)
        if warmup_steps < 0:
            raise ValueError(
                f"lr_warmup_steps must be >= 0 (0 disables the step warm-up and falls back to "
                f"the epoch-granularity schedule), got {warmup_steps}"
            )
        if warmup_steps == 0:
            return super().build_lr_scheduler(optimizer)

        milestones = [int(value) for value in (getattr(self.hparams, "lr_milestones", None) or [])]
        gamma = float(getattr(self.hparams, "lr_gamma", 0.1))

        # estimated_stepping_batches is the optimizer-step total for the WHOLE run; dividing by
        # max_epochs is what turns an epoch milestone into a step milestone. Floored at one so a
        # degenerate run (a single batch, or an epoch count Lightning reports as unlimited) still
        # produces a monotone milestone sequence rather than collapsing every milestone onto zero.
        #
        # Read as a float and tested for finiteness before it is used: an unlimited run --
        # `max_epochs: -1` to train until early stopping, or a dataloader of unknown length --
        # makes Lightning report the step total as infinity, and converting that to an int raises
        # rather than falling through to the floor. That failure would surface inside
        # `configure_optimizers`, i.e. after the run directory, the log sinks, the MLflow run and
        # every DDP rank are already up.
        trainer = self.trainer
        max_epochs = int(getattr(trainer, "max_epochs", 0) or 0)
        total_steps = float(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        bounded_run = max_epochs > 0 and math.isfinite(total_steps)
        steps_per_epoch = max(1, round(total_steps / max_epochs)) if bounded_run else 1
        step_milestones = [milestone * steps_per_epoch for milestone in milestones]

        def lr_factor(step: int) -> float:
            """Multiplier on the base learning rate at optimizer step ``step`` (zero-based)."""
            ramp = min(1.0, float(step + 1) / float(warmup_steps))
            decays = sum(1 for milestone in step_milestones if milestone <= step)
            return ramp * gamma**decays

        schedule: Dict[str, Any] = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_factor),
            # Per optimizer step, not per epoch: a ramp measured in steps stepped once per epoch
            # would take `lr_warmup_steps` EPOCHS to complete, silently.
            "interval": "step",
            "frequency": 1,
        }
        return schedule
