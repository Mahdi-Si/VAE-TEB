r"""The training task: one diamond, and not a line of its own.

Two parents, and each supplies exactly what its own package added to the shared task:

* :class:`~teb_vae.lag_attn_crws.task.SeqVaeLagAttnCrwsTask` -- the anchor tiling's derived phase,
  the stage-to-geometry resolution that puts $(\varphi_b, S)$ into a training forward and $(0, 1)$
  into an evaluation one, the source-null KL readout, the anchor-aware latent-gap diagnostic, the
  run seed in ``save_hyperparameters`` and the three diagnostic-page seams a tiled model cannot
  inherit; and, through *its* parent, ``_build_raw_target``, which scores the net against the raw
  FHR future the whole raw-target row shares;
* :class:`~teb_vae.lag_attn_transformer_rws.task.SeqVaeLagAttnTrfRwsTask` -- ``build_lr_scheduler``,
  the step-granular learning-rate ramp a pre-normalised attention stack needs in exactly the first
  few hundred optimizer steps, which an epoch-granularity schedule cannot address at all.

Everything else -- the objective, its $\beta$ schedule, the metric surface, the validation-only
permutation control, the spike-breaker wiring, the pre-clip gradient-norm logging and the checkpoint
contract -- is the shared task's, reached through both parents at once. It is the same code here,
not a copy of it, which is what makes this model comparable to the seven it sits beside.

**Why the diamond is well-formed.** Both parents derive from
:class:`~teb_vae.lag_attn_rws.task.SeqVaeLagAttnRwsTask`, so ``TrfCrwsTask -> CrwsTask ->
TrfRwsTask -> RwsTask -> LightningModelBase`` is a valid linearisation. It is *not* well-formed
because the two branches happen to be disjoint -- they are, everything the causal-input cell adds
against $\{$``build_lr_scheduler``$\}$, but that is a fact about today's code rather than a property
of the construction. A future member defined on both sides would resolve to the causal side by order
alone, silently; ``tests/test_task.py`` asserts the linearisation as a list of class names and each
behaviour against the class the design names, so a reorder fails rather than trains something else.

**The empty class body is the guarantee.** There is no ``__init__``: the causal parent already takes
the run seed the tile phase is derived from and puts it in ``save_hyperparameters``, and a second
keyword schema for one shared objective could only drift from the first. There is no
``training_step`` either -- the framework's own step runs the config-gated loss-spike breaker, and a
subclass defining its own disables it in silence. And there is no ``forecast_rows``: the diagnostic
page's rows walk this cell's ``anchor_index``, which is the causal parent's own property and names
no encoder, so the two cells of this row draw one picture rather than two.
"""
from __future__ import annotations

from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask


class SeqVaeLagAttnTrfCrwsTask(SeqVaeLagAttnCrwsTask, SeqVaeLagAttnTrfRwsTask):
    r"""Lightning task for
    :class:`~teb_vae.lag_attn_transformer_crws.nets.model.SeqVaeLagAttnTrfCrws`.

    Defines nothing. The tiling phase, the five-argument forward, the source-null readout and the
    page-row seams come from the first base, the step-granular learning-rate ramp from the second,
    and the objective, the metric surface, the permutation control and the checkpoint contract from
    the shared ancestor both derive from.

    ``main_loss`` therefore keeps its exact, unprefixed name, which is what the loss-spike breaker
    watches; the framework falls back to the returned loss when that key is missing, silently.
    """


__all__ = ["SeqVaeLagAttnTrfCrwsTask"]
