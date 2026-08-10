r"""The training task: one diamond, and not a line of its own.

Two parents, and each supplies exactly what its own package added to the shared task:

* :class:`~teb_vae.lag_attn_fs.task.SeqVaeLagAttnFsTask` -- ``_build_raw_target``, which scores the
  net against the two stored target blocks concatenated in the declared order, and the
  ``forecast_rows`` property, which is the whole route by which a model in another target domain
  replaces two rows of the seven-row diagnostic page;
* :class:`~teb_vae.lag_attn_transformer_rws.task.SeqVaeLagAttnTrfRwsTask` -- ``build_lr_scheduler``,
  the step-granular learning-rate ramp a pre-normalised attention stack needs in exactly the first
  few hundred optimizer steps, which an epoch-granularity schedule cannot address at all.

Everything else -- the objective, its $\beta$ schedule, the metric surface, the validation-only
permutation control, the spike-breaker wiring, the pre-clip gradient-norm logging and the checkpoint
contract -- is the shared task's, reached through both parents at once. It is the same code here, not
a copy of it, which is what makes this model comparable to the three it sits beside.

**Why the diamond is well-formed.** Both parents derive from
:class:`~teb_vae.lag_attn_rws.task.SeqVaeLagAttnRwsTask`, so ``TrfFsTask -> FsTask -> TrfRwsTask ->
RwsTask -> LightningModelBase`` is a valid linearisation. It is *not* well-formed because the two
branches happen to be disjoint -- they are, $\{$``_build_raw_target``, ``forecast_rows``$\}$ against
$\{$``build_lr_scheduler``$\}$, but that is a fact about today's code rather than a property of the
construction. A future member defined on both sides would resolve to the feature side by order
alone, silently; ``tests/test_task.py`` asserts the linearisation as a list of class names and each
of the three behaviours against the class the design names, so a reorder fails rather than trains
something else.

**The empty class body is the guarantee.** There is no ``__init__``: a second keyword schema for one
shared objective could only drift from the first, and every hyperparameter a run is recoverable from
reaches ``self.hparams`` through the base's constructor. There is no ``training_step`` either -- the
framework's own step runs the config-gated loss-spike breaker, and a subclass defining its own
disables it in silence.
"""
from __future__ import annotations

from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_transformer_rws.task import SeqVaeLagAttnTrfRwsTask


class SeqVaeLagAttnTrfFsTask(SeqVaeLagAttnFsTask, SeqVaeLagAttnTrfRwsTask):
    r"""Lightning task for
    :class:`~teb_vae.lag_attn_transformer_fs.nets.model.SeqVaeLagAttnTrfFs`.

    Defines nothing. The feature target's batch-to-target builder and page-row seam come from the
    first base, the step-granular learning-rate ramp from the second, and the objective, the metric
    surface, the permutation control and the checkpoint contract from the shared ancestor both
    derive from.

    ``main_loss`` therefore keeps its exact, unprefixed name, which is what the loss-spike breaker
    watches; the framework falls back to the returned loss when that key is missing, silently.
    """
