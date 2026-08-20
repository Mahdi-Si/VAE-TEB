r"""The training task: the raw-signal sibling's, plus the tiling phase and the source-null arm.

Everything about turning a batch into a loss is inherited from
:class:`~teb_vae.lag_attn_rws.task.SeqVaeLagAttnRwsTask` -- **directly**, not through the
causal-feature cell's task, which descends from the feature-target task and would bring a feature
target's ``_build_raw_target`` with it. That base is the design rather than an economy: two models
are only comparable if they optimise the same thing, so the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
  + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}} + \lambda_{\Delta} \mathcal{L}_{\Delta},
$$

its $\beta$ schedule, its metric surface, the validation-only permutation control, the
spike-breaker wiring, the pre-clip gradient-norm logging, the checkpoint contract and the raw
target itself are the same *code* here, not a copy of it. ``_build_raw_target`` is the parent's
because the target **is** the parent's: the raw FHR future, gathered at this cell's decoded
anchors by the net rather than by the task.

What is new is everything that follows from the *anchor tiling*, and all of it lives on the seam
between the batch and the net rather than inside the objective. None of it is target-coupled, so
none of it is written here twice: eight of the members below -- the four of the tiling seam, the two
readouts and the two input-side page seams -- are **bound by reference** from
:class:`~teb_vae.lag_attn_cfs.task.SeqVaeLagAttnCfsTask`, which edits nothing there and leaves no
second definition to drift.

**The forward takes five tensors, not three.** The net decodes a tiled anchor set
$\mathcal{A}(\varphi) = \{F + \varphi + kS\}$, and both $\varphi$ and $S$ are arguments. They are
resolved from the ``stage`` string the framework's step dispatcher already passes: $(\varphi_b, S)$
on ``train``, $(0, 1)$ on ``val`` and ``test``, so both evaluation stages decode every valid anchor
and neither depends on a phase at all. Resolving the stride from ``self.training`` instead was
rejected where the net is written: the diagnostic callback calls ``eval()`` *during* training and
then the objective, so a mode-derived geometry would make ``total_loss`` a function of the dropout
switch.

**The phase is derived, never drawn.** A ``torch.randint`` here would consume the global RNG
stream, move the reparameterisation $\epsilon$ and break every bitwise comparison in the suite --
and would not survive a checkpoint resume. The bound ``anchor_phase`` hashes the segment's own
identity instead; see it for why each half of the key is load-bearing, and note that binding it
rather than copying it is also what keeps the cross-process stability guarantee **shared** with the
cell that already proved it, instead of re-proved against a second implementation of the same hash.

**Two readouts reach the decoded anchor set.** ``mu_post_prior_gap_rms`` is inherited from a task
whose model decodes every anchor, and its own docstring promises it uses "the KL's own anchor
support ... so the two cannot drift"; under tiling that promise fails, and the bound override
restores it. And ``kld_source_null`` -- the floor the source *availability clock* induces with no
source content in it -- needs the source stream, which is not in the forward dict, so it arrives
through the ``_added_metrics`` hook rather than through a second forward.

**Three page seams, and only one of them is this cell's own.** The forecast rows are re-pointed
because the anchor axis is sparse: the shipped rows walk a dense block and read it at an *anchor*,
and this model's forecast is indexed by position in the decoded set. The input rows and the
run-level budget figure are the causal-feature cell's, **bound by reference** like everything else
here -- those two seams describe the three input tensors and the resolved warm-up budget, and this
cell reads the identical tensors and resolves the identical budget, so a second implementation of
either could only differ from the first by being wrong. Both shipped builders consult the production
two-sided Morlet bank, which did not produce these coefficients and refuses these channel widths,
inside handlers that warn and continue -- so the cost of not re-pointing them is a green suite and a
page with two rows and a figure missing.

Only two members are written out beyond that seam. ``__init__`` and
:meth:`SeqVaeLagAttnCrwsTask.compute_loss_and_metrics` both call ``super()``, whose zero-argument
form closes over the class that *defines* it: bound onto a class outside that one's hierarchy it
raises ``TypeError: super(type, obj): obj must be an instance or subtype of type``, three frames from
anything that names the binding. So those two are this cell's own, and they are the two whose bodies
are three lines each.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from teb_vae.lag_attn_cfs.causal_warmup import WarmupBudget
from teb_vae.lag_attn_cfs.task import DENSE_STAGES, SeqVaeLagAttnCfsTask
from teb_vae.lag_attn_rws.sample_page import ForecastRowInputs
from teb_vae.lag_attn_rws.task import SeqVaeLagAttnRwsTask


class SeqVaeLagAttnCrwsTask(SeqVaeLagAttnRwsTask):
    r"""Lightning task for :class:`~teb_vae.lag_attn_crws.nets.model.SeqVaeLagAttnCrws`.

    Adds the tiling seam and the source-null readout, and the absence of everything else is
    deliberate and load-bearing: no ``training_step`` (the inherited one runs the config-gated
    loss-spike breaker), no ``compute_loss`` call of its own (the net owns the anchored gather), and
    above all no ``_build_raw_target`` -- the target is the raw FHR future and a second copy of that
    builder could only drift from the one the comparison model is scored through.

    Eight members are **bound by reference** from
    :class:`~teb_vae.lag_attn_cfs.task.SeqVaeLagAttnCfsTask`. Each is about the anchor set, the
    source stream or the three input tensors the page draws, and none of those notions changes with
    what the decoder emits, so the object here *is* the object there and drift is structurally
    impossible rather than merely test-detected.

    ``staticmethod(...)`` around ``_phase_field`` is load-bearing and its absence fails three frames
    away. ``Owner.some_staticmethod`` returns the **plain function** -- the descriptor has already
    resolved -- and a plain function assigned in a class body becomes an *instance* method, so
    ``self`` would arrive as its first positional argument and the batch as its second.
    """

    #: The tile phase and the five-tuple it travels in. All four read the batch, the model's own
    #: ``anchor_stride`` and the run seed, and name no target tensor at all.
    anchor_phase = SeqVaeLagAttnCfsTask.anchor_phase
    _phase_field = staticmethod(SeqVaeLagAttnCfsTask._phase_field)
    resolve_anchor_geometry = SeqVaeLagAttnCfsTask.resolve_anchor_geometry
    _build_forward_inputs = SeqVaeLagAttnCfsTask._build_forward_inputs

    #: The two readouts the tiling moves. ``_mu_gap_rms`` rebuilds the reconstruction and KL masks
    #: at the anchors the forward decoded, so the belief shift and the KL printed beside it are
    #: averaged over one anchor set; ``_added_metrics`` runs the source-null control on the
    #: evaluation stages only. Neither reads a target block.
    _mu_gap_rms = SeqVaeLagAttnCfsTask._mu_gap_rms
    _added_metrics = SeqVaeLagAttnCfsTask._added_metrics

    #: The two page seams that describe what the model was *given* rather than what it produced.
    #: Both are the causal-feature cell's, bound whole: this cell reads the identical three input
    #: tensors and resolves the identical warm-up budget, so the streams the first draws and the
    #: dropped channels the second draws are the same quantities here. ``input_stream_panels`` is a
    #: property, and a property accessed on its owning **class** returns the descriptor itself, so
    #: the binding carries the property rather than a resolved value.
    input_stream_panels = SeqVaeLagAttnCfsTask.input_stream_panels
    input_budget_figure = SeqVaeLagAttnCfsTask.input_budget_figure

    def __init__(self, base_model: Any, *, seed: int = 0, **kwargs: Any) -> None:
        r"""Initialize the task.

        Args:
            base_model: The net to wrap.
            seed: The run seed, one of the four halves of the tile-phase key. It is read once by
                the framework's own ``configure_determinism`` and handed to ``seed_everything``,
                and reaches no task in the family today -- so it is taken here explicitly and put
                into ``save_hyperparameters``, which is what makes a **resumed** run reproduce the
                phases it was drawing before it stopped. Defaults to $0$ so a unit-constructed task
                needs no run to exist.
            **kwargs: Every other keyword, forwarded to the inherited constructor unchanged. A
                second keyword schema for the same objective could only drift from the first.
        """
        super().__init__(base_model, **kwargs)
        self.save_hyperparameters("seed")

    @property
    def forecast_rows(self) -> Callable[[ForecastRowInputs], None]:
        r"""The page's first two rows, bound to this net's tiling.

        The raw sibling's rows cannot be inherited, and the reason is not the target domain -- the
        target *is* its target -- but the anchor axis. That implementation tiles through
        ``concat_single_forecasts``, which reads its per-anchor block at an **anchor** index; this
        model's forecast is $(A_{\max}, H, R)$ and is indexed by position in the decoded set. The two
        coincide only at floor $0$ and stride $1$. At the shipped geometry they do not even have
        compatible ranges -- $136$ positions read at anchors $134 \dots 269$ -- so the page dies
        inside a handler that warns and continues, and the run's diagnostics directory comes out
        empty; at a smaller floor it draws a real forecast at the wrong time with no exception in it.

        One value is bound, and it is one the page cannot recover from the arrays it is handed: the
        stride a training step tiles at, which the overlay draws beside the dense set this page is
        itself produced at.

        Returns:
            A callable taking one
            :class:`~teb_vae.lag_attn_rws.sample_page.ForecastRowInputs` and drawing into it.
        """
        from teb_vae.lag_attn_crws.sample_page import causal_raw_forecast_rows

        return partial(
            causal_raw_forecast_rows, training_stride=int(self.orig_model.anchor_stride)
        )

    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the inherited step, with the stage reachable from the input builder.

        The one thing this adds is the stage, and it is added *here* rather than threaded through
        ``_build_forward_inputs``'s signature because that signature is the family's shared seam
        between a batch and a net -- the plotting callback calls it with one argument, and so does
        every sibling's test. The attribute lives for exactly the length of one step and is set
        before anything reads it.

        The default matters: it is one of the two dense stages, so a caller reaching
        ``_build_forward_inputs`` outside a step -- the diagnostic callback does exactly that --
        gets the dense anchor set and a phase of zero, which is the reproducible geometry a figure
        should be drawn at, rather than a training tile grid that depends on the epoch.

        Written out rather than bound from the cell this task's tiling comes from, and the reason is
        the ``super()`` below: its zero-argument form closes over the class that *defines* it, so
        the bound copy would resolve ``super(SeqVaeLagAttnCfsTask, self)`` against an instance that
        is not one and raise ``TypeError`` on the first step of the first run.

        Args:
            batch: A batch from the data module.
            batch_idx: Index of the current batch.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            ``(loss, metrics)``, the inherited pair.
        """
        self._stage = stage
        try:
            return super().compute_loss_and_metrics(batch, batch_idx, stage)
        finally:
            self._stage = DENSE_STAGES[0]

    #: The stage the next ``_build_forward_inputs`` resolves its anchor geometry from. A class
    #: attribute so it exists before the first step and on a task nothing has stepped, and one of
    #: the dense stages so an out-of-step call -- the diagnostic callback's -- draws the dense,
    #: epoch-independent anchor set rather than a training tile grid.
    _stage: str = DENSE_STAGES[0]

    #: The resolved warm-up budget this run got, set by the experiment driver once it has built
    #: both. Read by the bound ``input_budget_figure`` and by nothing else, and deliberately **not**
    #: a hyperparameter: the four channel tuples the network needs are already in the checkpoint's
    #: ``model_kwargs``, and a second copy of them under another name is a second thing to keep
    #: true. ``None`` -- the default, so a hand-built task needs no budget to exist -- costs exactly
    #: the run-level figure and nothing else on the page, and says so by name when it is asked for.
    warmup_budget: Optional[WarmupBudget] = None


__all__ = ["DENSE_STAGES", "SeqVaeLagAttnCrwsTask"]
