r"""The training task: the comparison model's task, fed raw signals instead of stored features.

Everything about turning a batch into a loss is inherited from
:class:`~teb_vae.lag_attn_transformer_rws.task.SeqVaeLagAttnTrfRwsTask`, which in turn inherits it
from the raw-signal model's task. That is the point rather than an economy: two architectures are
only comparable if they optimise the same thing, so the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p,
$$

its $\beta$ schedule, its metric surface, the validation-only permutation control, the
spike-breaker wiring, the pre-clip gradient-norm logging and the checkpoint contract are all the
same code, not a copy of it.

**The parent is the transformer task, not the raw-signal one**, and the distinction is load
bearing: the step-granular learning-rate ramp lives only on
:meth:`~teb_vae.lag_attn_transformer_rws.task.SeqVaeLagAttnTrfRwsTask.build_lr_scheduler`.
Subclassing the other one would leave the config requesting ``lr_warmup_steps: 2000``, the monitor
logging per step, and no ramp existing at all -- with nothing failing anywhere.

The one addition is :meth:`SeqVaeLagAttnTrfE2ETask._build_forward_inputs`, the seam the base class
exists to offer: an architecture over a different *input representation* keeps the whole objective
and overrides only what the net is handed.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch

from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask


class SeqVaeLagAttnTrfE2ETask(SeqVaeLagAttnTrfRwsTask):
    r"""Lightning task for
    :class:`~teb_vae.lag_attn_transformer_e2e.nets.model.SeqVaeLagAttnTrfE2E`.

    Adds exactly one method. The absence of everything else is deliberate and load-bearing: no
    ``training_step`` (the inherited one runs the config-gated loss-spike breaker), no
    ``configure_optimizers`` (the framework's calls ``build_lr_scheduler``, which is the seam the
    step warm-up hangs off), no ``build_lr_scheduler`` (that ramp is inherited, and a second copy
    of a schedule is a second schedule), and no constructor -- a second keyword schema for the same
    objective could only drift from the first, and it is what a later evaluation entry point would
    have to reconstruct the task from.
    """

    def _build_forward_inputs(self, batch: Any) -> Tuple[torch.Tensor, ...]:
        """Return ``(fhr, up, weight)``: the two raw signals and their shared validity.

        The whole difference between this model and the one it is compared against, expressed as
        the three tensors its ``forward`` takes. The feature blocks the sibling assembles --
        ``fhr_st``, ``fhr_ph``, ``up_st``, ``up_ph`` -- are not loaded at all here, so the two
        stream builders this overrides are unreachable rather than merely unused.

        ``fhr`` and ``weight`` come from :meth:`_build_raw_target`, the inherited single source of
        the reconstruction target, rather than from a second read of the batch. That is what makes
        the tensor the target front end consumes and the tensor the loss scores **the same
        object**: a model scored against a tensor other than the one it was shown would produce a
        plausible loss curve and a meaningless result. It also means the refusal for a missing
        ``fhr`` is that method's, which already names both config lists.

        Args:
            batch: A batch from the data module.

        Returns:
            ``(fhr, up, weight)``, splatted into the net's forward.

        Raises:
            RuntimeError: If ``up`` is absent, naming both config keys that put it there. Missing
                from ``load_fields`` the source stream does not exist; missing from
                ``normalize_fields`` it arrives in raw contraction units against a front end whose
                whole input contract is that the loader has already standardised it -- and nothing
                raises on that one.
        """
        fhr, weight = self._build_raw_target(batch)
        up = getattr(batch, "up", None)
        if up is None:
            raise RuntimeError(
                "batch has no `up` field, and the raw UP is this model's source stream -- the "
                "input the source-conditioned KL is a readout of. Add 'up' to "
                "dataset_kwargs.load_fields AND to dataloader_config.normalize_fields: the front "
                "end owns no statistics of its own and consumes what the loader produced, so an "
                "unnormalized source silently changes the operating point of every source-side "
                "quantity the model reports."
            )
        return fhr, up, weight
