r"""The training task: the feature-target sibling's, plus the tiling phase and the source-null arm.

Everything about turning a batch into a loss is inherited from
:class:`~teb_vae.lag_attn_fs.task.SeqVaeLagAttnFsTask`, which is itself the raw-signal task with one
builder re-pointed. That chain is the design rather than an economy: two models are only comparable
if they optimise the same thing, so the objective

$$
\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
  + \beta(e)\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p,
$$

its $\beta$ schedule, its metric surface, the validation-only permutation control, the spike-breaker
wiring, the pre-clip gradient-norm logging and the checkpoint contract are the same *code* here, not
a copy of it. ``_build_raw_target`` is the parent's for the same reason: this model forecasts the
same two stored blocks, one-sided rather than two-sided, and the block boundary it splits its
reported gaps at is a constant on the net rather than a decision the task makes.

**Three page seams are this task's own**, and each replaces a builder welded to something this
family does not have. The forecast rows are re-pointed because the anchor axis is sparse; the input
rows and the run-level budget figure because the shipped ones consult the production two-sided
Morlet bank, which did not produce these coefficients and refuses these channel widths -- inside
handlers that warn and continue, so the cost of not replacing them is a green suite and a page with
two rows missing.

What is new here is everything that follows from the *anchor tiling*, and all of it lives on the
seam between the batch and the net rather than inside the objective:

**The forward takes five tensors, not three.** The net decodes a tiled anchor set
$\mathcal{A}(\varphi) = \{F + \varphi + kS\}$, and both $\varphi$ and $S$ are arguments. They are
resolved from the ``stage`` string the framework's step dispatcher already passes: $(\varphi_b, S)$
on ``train``, $(0, 1)$ on ``val`` and ``test``, so both evaluation stages decode every valid anchor
and neither depends on a phase at all. Resolving the stride from ``self.training`` instead was
rejected where the net is written: the diagnostic callback calls ``eval()`` *during* training and
then the objective, so a mode-derived geometry would make ``total_loss`` a function of the dropout
switch.

**The phase is derived, never drawn.** A ``torch.randint`` here would consume the global RNG stream,
move the reparameterisation $\epsilon$ and break every bitwise comparison in the suite -- and would
not survive a checkpoint resume. :meth:`SeqVaeLagAttnCfsTask.anchor_phase` hashes the segment's own
identity instead; see it for why each half of the key is load-bearing.

**Two readouts are re-pointed at the decoded anchors.** ``mu_post_prior_gap_rms`` is inherited from
a task whose model decodes every anchor, and its own docstring promises it uses "the KL's own anchor
support ... so the two cannot drift". Under tiling that promise fails, so the override restores it.
And ``kld_source_null`` -- the floor the source *availability clock* induces with no source content
in it -- needs the source stream, which is not in the forward dict, so it arrives through the
``_added_metrics`` hook rather than through a second forward.
"""
from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn_cfs.causal_warmup import WarmupBudget
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_rws.nets import controls
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask, kl_mask
from teb_vae.lag_attn_rws.sample_page import ForecastRowInputs, InputStreamPanel

#: Field separator inside the tile-phase hash key. A byte that cannot occur in a GUID or in a
#: decimal integer, so no two distinct key tuples can render to one string -- ``"A" + "12"`` and
#: ``"A1" + "2"`` would otherwise collide and give two segments one grid.
_KEY_SEPARATOR = b"\x1f"

#: Stages whose anchor set is the dense range. Both evaluation stages decode every valid anchor
#: rather than one tile grid: a single phase is deterministic but *phase-biased*, so any structure
#: varying with position in the segment would be sampled at one offset from the segment start
#: forever. There is no gradient at either stage, so neither the redundancy argument nor the memory
#: argument that motivates the tiling applies.
DENSE_STAGES: Tuple[str, ...] = ("val", "test")


class SeqVaeLagAttnCfsTask(SeqVaeLagAttnFsTask):
    r"""Lightning task for :class:`~teb_vae.lag_attn_cfs.nets.model.SeqVaeLagAttnCfs`.

    Adds four members, and the absence of everything else is deliberate and load-bearing: no
    ``training_step`` (the inherited one runs the config-gated loss-spike breaker), no
    ``compute_loss_and_metrics`` (which is where the permutation control, the ``main_loss`` name the
    breaker watches and the latent-gap diagnostic live), and no ``_build_raw_target`` -- the target
    is the same two stored blocks concatenated, and a second copy of that builder could only drift
    from the one the comparison model is scored through.
    """

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

    # ------------------------------------------------------------------
    # The diagnostic page's three seams
    # ------------------------------------------------------------------
    @property
    def forecast_rows(self) -> Callable[[ForecastRowInputs], None]:
        r"""The page's first two rows, bound to this net's channel facts and its tiling.

        The two-sided sibling's rows cannot be inherited, and the reason is not the target domain
        -- it is the anchor axis. That implementation walks a dense $(T_{\mathrm{valid}}, H, C)$
        block and indexes an *anchor* into its first axis; this model's forecast is
        $(A_{\max}, H, C)$ and is indexed by position in the decoded set. The two agree only at
        $F = 0$ and stride $1$, and everywhere else the inherited rows would draw a real forecast
        at the wrong time, with no shape error anywhere in it.

        Five values are bound, and each is something the page cannot recover from the arrays it is
        handed: which declared channel each decoder output *is*, where the two stored blocks meet
        on that channel axis, the stride a training step tiles at -- which the page draws beside
        the dense set it is itself produced at -- and the two the per-window score row needs. Those
        last two are taken from where the objective takes them, the hyperparameter for the
        likelihood and the net for the coverage floor, so a window's height on that row is the
        block score this run computed rather than one drawn under some other assumption.

        Returns:
            A callable taking one
            :class:`~teb_vae.lag_attn_rws.sample_page.ForecastRowInputs` and drawing into it.
        """
        from teb_vae.lag_attn_cfs.sample_page import causal_forecast_rows

        model = self.orig_model
        gate = model.target_gate
        # The forecast clock's tau in seconds comes from the resolved budget, as the input clocks
        # do in `input_stream_panels`: the net stamps the per-channel step shifts and nothing
        # from which tau can be recovered. It reaches the page as a statement on the forecast
        # rows' time axis, not as a shift of anything drawn.
        budget = self.warmup_budget
        return partial(
            causal_forecast_rows,
            keep_index=None if gate is None else gate.keep_index,
            block_split=int(model.TARGET_BLOCK_SPLIT),
            training_stride=int(model.anchor_stride),
            likelihood=str(self.hparams.get("likelihood", "gaussian_nll")),
            coverage_floor=float(model.coverage_floor),
            # The model's own forecast clock, so the page's truth, its window scores and its mask
            # are the objective's -- None on the stored clock, where the page is what it was.
            target_forecast_shift=model.target_forecast_shift,
            forecast_clock_delay_s=(
                None if budget is None else budget.target_forecast_clock_delay_s
            ),
        )

    @property
    def forecast_extra_rows(self) -> Tuple[Tuple[str, float], ...]:
        """The rows :attr:`forecast_rows` draws beyond the two the layout always reserves.

        Resolved off the task by the callback and handed to the page builder, which is the only
        place a GridSpec row can be created -- the seam itself runs after the layout is fixed and
        can reach only rows that already exist. Returned from the drawing module's own constant
        rather than restated here, so the names reserved and the names drawn are one object: a
        name reserved and not drawn is a blank row on every page of the run, and a name drawn and
        not reserved is a ``KeyError`` raised inside a handler that swallows it.

        Returns:
            ``(name, height_ratio)`` per row, in drawing order.
        """
        from teb_vae.lag_attn_cfs.sample_page import CAUSAL_EXTRA_ROWS

        return CAUSAL_EXTRA_ROWS

    @property
    def input_stream_panels(self) -> Callable[..., Sequence[InputStreamPanel]]:
        """The page's input rows, which the shipped builder cannot draw for this model.

        That builder is welded to the production two-sided Morlet bank: it refuses these channel
        widths, and it raises inside a handler that warns and continues -- so leaving it in place
        costs two page rows and one log line, with a green suite. It also draws ``gate(values)``,
        and on this model the gate gathers and then shifts, with the warm-up mask one layer
        further on, inside the availability adapter.

        No binding is needed. Everything the replacement reads -- the gates, the adapters' own
        availability buffers, the warm-up vectors and the block splits -- is on the net it is
        handed, which is what keeps the drawn stream the encoder's input by construction rather
        than by two pieces of code agreeing.

        Returns:
            A builder with the signature of
            :func:`~teb_vae.lag_attn_rws.input_budget.stream_panels`.
        """
        from functools import partial

        from teb_vae.lag_attn_cfs.sample_page import causal_stream_panels

        # The clocks come from the resolved budget rather than from the net, which carries only
        # the per-channel shifts: tau_ref cannot be recovered from those without the stored
        # delays the net does not keep. Bound here rather than passed through the shared page
        # builder, whose hook signature is the two-sided family's and describes a guard that has
        # no reference at all. A task with no budget -- every hand-built one -- returns the plain
        # builder, and the rows then omit the clock clause instead of stating a wrong one.
        budget = self.warmup_budget
        if budget is None or budget.reference_delay_s is None:
            return causal_stream_panels
        return partial(
            causal_stream_panels,
            reference_delay_s={
                "target": budget.reference_delay_s,
                "source": budget.source_clock_delay_s,
            },
        )

    def input_budget_figure(self, directory: Any, *, file_format: str = "pdf") -> Path:
        r"""Write the run-level figure: every declared channel's warm-up against the window.

        Replaces the shipped ``causal_input_budget`` figure, which describes the two-sided *reach*
        guard and is built from the production Morlet bank. The two answer different questions and
        are written under different stems, so a directory holding both is readable rather than
        ambiguous.

        Drawn from :attr:`warmup_budget` rather than from the net, because the figure's subject is
        the channels the budget **dropped** beside the ones it kept -- and a dropped channel's own
        $W'_c$ is exactly what the checkpoint does not carry: ``model_kwargs`` stamps the survivors'
        vector, since that is what the constructor needs.

        A method rather than a property, deliberately. The callback resolves this seam with
        ``getattr(pl_module, ..., None)``, which does not swallow an exception raised *inside* a
        property -- so a task with no budget would take down the whole page rather than cost the
        one figure it can no longer draw. Raised on the call instead, inside the caller's own
        handler and behind its once-per-run latch.

        Args:
            directory: Where the figure goes; created if absent.
            file_format: Figure extension, without the dot.

        Returns:
            The written path.

        Raises:
            ValueError: If no resolved budget reached this task.
        """
        from teb_vae.lag_attn_cfs.warmup_budget import write_warmup_budget_figure

        if self.warmup_budget is None:
            raise ValueError(
                "no resolved warm-up budget reached this task, so the channels this run dropped "
                "-- which is what the figure is about -- cannot be recovered: the checkpoint "
                "carries the survivors' warm-up vector alone. The experiment driver sets "
                "`warmup_budget` on the task it builds; a task constructed by hand supplies it "
                "the same way."
            )
        return write_warmup_budget_figure(
            self.warmup_budget,
            Path(directory),
            horizon=int(self.orig_model.horizon),
            file_format=file_format,
        )

    # ------------------------------------------------------------------
    # Batch -> model inputs
    # ------------------------------------------------------------------
    def anchor_phase(self, batch: Any) -> torch.Tensor:
        r"""Derive $\varphi_b \in [0, S)$ per sample, as a stable hash of the segment's identity.

        $$\varphi_b = \mathrm{blake2b}\big(\texttt{guid}_b \,\Vert\, \lfloor\texttt{domain\_start}_b\rfloor
        \,\Vert\, \texttt{epoch} \,\Vert\, \texttt{seed}\big) \bmod S$$

        Three properties of that key are load-bearing, and each replaces something that does not
        work:

        * **``hashlib.blake2b``, not Python's ``hash()``.** ``hash()`` on a ``str`` is salted per
          process by ``PYTHONHASHSEED``, which is random by default, so a phase derived from it is
          stable neither across DDP ranks nor across a checkpoint resume -- and it fails *silently*,
          because $A_{\max}$ is a geometry constant either way and no shape or count would differ.
        * **``domain_start`` is in the key, not just ``guid``.** The GUID identifies the
          *recording*, not the segment, and an unshuffled loader over per-recording shards puts
          consecutive segments of one recording in one batch. Keying on the GUID alone would give
          every segment of a recording the same tile grid within an epoch, leaving in place exactly
          the within-batch gradient correlation the tiling exists to break. The batch's ``epoch``
          field is ``domain_start`` in seconds and is per segment.
        * **Nothing is drawn.** A ``torch.randint`` here would consume the global RNG stream and
          move the reparameterisation $\epsilon$ for every subsequent step.

        The training epoch is in the key so the grid *rotates* over epochs, which is what makes the
        claim that every anchor in $[F, T_{\mathrm{valid}})$ is eventually decoded true rather than
        aspirational. ``current_epoch`` is $0$ on a trainer-less task rather than raising, so a unit
        test needs no fit.

        Args:
            batch: A batch from the data module, carrying ``guid`` and ``epoch``.

        Returns:
            A $(B,)$ ``long`` tensor on the batch's own device.

        Raises:
            RuntimeError: If ``guid`` or ``epoch`` is absent, naming the config key that fixes it.
                Neither is optional: without them the phase degenerates to one grid for the whole
                dataset and nothing about the run would say so.
        """
        model = self.orig_model
        stride = int(model.anchor_stride)
        guids = self._phase_field(batch, "guid")
        starts = self._phase_field(batch, "epoch")
        train_epoch = int(self.current_epoch)
        seed = int(self.hparams.get("seed", 0))

        phases = []
        for guid, start in zip(guids, starts):
            key = _KEY_SEPARATOR.join(
                (
                    _as_key(guid),
                    # Floored to an integer: `domain_start` is stored as float32 seconds, and a
                    # value that round-trips differently through two builds of the same shard would
                    # otherwise re-tile a segment that had not moved.
                    str(int(_as_float(start) // 1)).encode("utf-8"),
                    str(train_epoch).encode("utf-8"),
                    str(seed).encode("utf-8"),
                )
            )
            digest = hashlib.blake2b(key, digest_size=8).digest()
            phases.append(int.from_bytes(digest, "big") % stride)
        # Built on the host, where the key already is: the net moves it to the device it builds the
        # anchor index on, so a device here would be a second decision about the same thing.
        return torch.tensor(phases, dtype=torch.long)

    @staticmethod
    def _phase_field(batch: Any, name: str):
        """Read one of the two phase-key fields off a batch, refusing an absent one by name.

        Args:
            batch: A batch from the data module.
            name: ``'guid'`` or ``'epoch'``.

        Returns:
            The field, as whatever sequence the loader delivered.

        Raises:
            RuntimeError: If the field is absent, naming the config list that carries it.
        """
        value = getattr(batch, name, None)
        if value is None:
            raise RuntimeError(
                f"batch has no `{name}` field, and the anchor tiling's phase is keyed on it. Add "
                f"'{name}' to dataset_config.dataloader_config.dataset_kwargs.load_fields -- that "
                f"list is honoured literally, with no forced additions, so dropping either key "
                f"leaves every segment on one tile grid forever with no shape or count differing."
            )
        return value

    def resolve_anchor_geometry(self, stage: str, batch: Any) -> Tuple[Any, int]:
        r"""Resolve $(\varphi, S)$ for one step from the stage string.

        Training tiles at the model's configured stride and a per-segment phase; ``val`` and
        ``test`` decode the dense range, where the stride is $1$ and the only admissible phase is
        $0$ -- the anchor set truncates rather than rotating there, so a non-zero phase would drop
        leading anchors rather than shift the grid.

        Args:
            stage: ``'train'``, ``'val'`` or ``'test'``.
            batch: A batch from the data module.

        Returns:
            ``(anchor_phase, anchor_stride)``, ready to splat into the net's forward.
        """
        if stage in DENSE_STAGES:
            return 0, 1
        return self.anchor_phase(batch), int(self.orig_model.anchor_stride)

    def _build_forward_inputs(self, batch: Any) -> Tuple[Any, ...]:
        """Return the positional arguments the net's ``forward`` takes, in order.

        Five, not the family's three: the two target blocks, the assembled source stream, the
        per-sample tile phase and the stride. Everything downstream of this method reads
        ``inputs[0]`` for the batch size and the device rather than a named tensor, so the arity
        change reaches nothing else -- the diagnostic callback star-splats it, and the objective
        never sees it at all.

        Args:
            batch: A batch from the data module.

        Returns:
            ``(y_st, y_ph, u_stream, anchor_phase, anchor_stride)``.
        """
        y_st, y_ph = self._build_target_streams(batch)
        source = self._build_source_stream(batch)
        # The stage is not an argument here -- the framework's step dispatcher owns it and this
        # method is called from the inherited step, which knows it. Reading it off ``self`` would
        # be the ``self.training`` mistake in another spelling, so the stage travels on the
        # instance for exactly the length of one step; see ``compute_loss_and_metrics``'s caller.
        phase, stride = self.resolve_anchor_geometry(self._stage, batch)
        return y_st, y_ph, source, phase, stride

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _mu_gap_rms(
        self, forward_outputs: Dict[str, torch.Tensor], weight: torch.Tensor
    ) -> torch.Tensor:
        r"""Masked RMS of the per-step latent mean gap, over the anchors the forward decoded.

        The inherited version rebuilds both masks with **no** anchor set, which is right for a model
        that decodes every anchor and wrong for one that decodes a tile: ``mu_post_prior_gap_rms``
        would then average the belief shift over all $T_{\mathrm{valid}} - F$ anchors while the
        ``source_conditioned_kl_raw`` printed beside it averages over the roughly ten the objective
        saw, and the two are read against each other. The override changes only the support, which
        is what restores the property the inherited docstring already claims -- that the gap and the
        KL cannot drift because they use one anchor set.

        Args:
            forward_outputs: The net's forward dict, carrying ``anchor_index`` and ``anchor_valid``.
            weight: Per-step validity, $(B, T)$.

        Returns:
            A scalar tensor.
        """
        with torch.no_grad():
            model = self.orig_model
            anchors = forward_outputs.get("anchor_index")
            anchor_valid = forward_outputs.get("anchor_valid")
            # The forecast clock's pooled validity, so this mask is the one the objective scored
            # under -- the identity object on the stored clock.
            forecast, _coverage = forecast_mask(
                model.scored_weight(weight),
                model.geometry,
                coverage_floor=model.coverage_floor,
                anchors=anchors,
                anchor_valid=anchor_valid,
            )
            support = kl_mask(
                forecast, model.geometry, anchors=anchors, anchor_valid=anchor_valid
            )
            gap_sq = (
                (forward_outputs["mu_post"] - forward_outputs["mu_prior"]) ** 2
            ).sum(dim=-1)
            return torch.sqrt((gap_sq * support).sum() / support.sum().clamp_min(1.0))

    def _added_metrics(
        self,
        inputs: Tuple[Any, ...],
        forward_outputs: Dict[str, torch.Tensor],
        weight: torch.Tensor,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        r"""The source-null KL floor, on the evaluation stages only.

        The hazard it sizes is specific and no other control can see it. The source availability
        pattern $m^u_{t,c}$ is a deterministic function of $t$, **identical in every row of the
        batch**, and it enters $q(z \mid Y, U)$ and not $p(z \mid Y)$ -- so it can push the
        posterior off the prior and inflate the coupling readout with no source information in it at
        all. The permutation control deranges rows, and no permutation of rows can remove something
        every row shares.

        $\texttt{source\_conditioned\_kl\_raw} - \texttt{kld\_source\_null}$ is therefore the part
        of the coupling readout attributable to source *variation*. If the two are equal, the
        readout is measuring a clock.

        Validation-only, like the permutation control and for the same two reasons: it is a readout
        that never enters the objective, and it costs one source encode per step that a training
        loop should not pay. It is **absent**, never zero-filled, on the steps that did not run it:
        the framework logs every metric with ``on_epoch=True``, whose epoch value is the mean over
        the steps that reported it, so a zero placeholder would scale the aggregate toward nothing.

        Args:
            inputs: The five positional arguments the forward was given; ``inputs[2]`` is the
                source stream at its **declared** width, which is not in the forward dict at all.
            forward_outputs: That forward's dict.
            weight: Decimated validity signal $(B, T)$.
            stage: ``'train'``, ``'val'`` or ``'test'``.

        Returns:
            ``{'kld_source_null': ...}`` on the evaluation stages, empty on ``train``.
        """
        if stage == "train":
            return {}
        return {
            "kld_source_null": controls.source_null_kld(
                self.orig_model, forward_outputs, inputs[2], weight
            )
        }

    # ------------------------------------------------------------------
    # Loss + metrics
    # ------------------------------------------------------------------
    def compute_loss_and_metrics(
        self, batch: Any, batch_idx: int, stage: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run the inherited step, with the stage reachable from the input builder.

        The one thing this adds is the stage, and it is added *here* rather than threaded through
        :meth:`_build_forward_inputs`'s signature because that signature is the family's shared seam
        between a batch and a net -- the plotting callback calls it with one argument, and so does
        every sibling's test. The attribute lives for exactly the length of one step and is set
        before anything reads it.

        The default matters: it is one of the two dense stages, so a caller reaching
        :meth:`_build_forward_inputs` outside a step -- the diagnostic callback does exactly that --
        gets the dense anchor set and a phase of zero, which is the reproducible geometry a figure
        should be drawn at, rather than a training tile grid that depends on the epoch.

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

    #: The stage the next :meth:`_build_forward_inputs` resolves its anchor geometry from. A class
    #: attribute so it exists before the first step and on a task nothing has stepped, and one of
    #: the dense stages so an out-of-step call -- the diagnostic callback's -- draws the dense,
    #: epoch-independent anchor set rather than a training tile grid.
    _stage: str = DENSE_STAGES[0]

    #: The resolved warm-up budget this run got, set by the experiment driver once it has built
    #: both. Read by :attr:`input_budget_figure` and by nothing else, and deliberately **not** a
    #: hyperparameter: the four channel tuples the network needs are already in the checkpoint's
    #: ``model_kwargs``, and a second copy of them under another name is a second thing to keep
    #: true. ``None`` -- the default, so a hand-built task needs no budget to exist -- costs
    #: exactly the run-level figure and nothing else on the page. The base task's
    #: ``on_save_checkpoint`` stamps its :meth:`~WarmupBudget.representation` into the blob under
    #: ``causal_representation`` when it is set, so a checkpoint names the representation its
    #: channel tuples were resolved under (CFS-03); no override here, by the member pin.
    warmup_budget: Optional[WarmupBudget] = None


def _as_key(value: Any) -> bytes:
    """Render one GUID -- ``str``, ``bytes`` or a 0-d tensor of either -- as hash-key bytes.

    The loader delivers GUIDs as Python strings, but an HDF5 read can hand back ``bytes`` and a
    collated batch can hand back a list of either. Normalising here rather than at the call site is
    what keeps the phase a function of the *recording* rather than of its representation: two
    spellings of one GUID would tile one recording two ways.

    Args:
        value: The GUID, in whatever form the batch carries.

    Returns:
        The key bytes.
    """
    if isinstance(value, bytes):
        return value
    if isinstance(value, torch.Tensor):
        return str(value.tolist()).encode("utf-8")
    return str(value).encode("utf-8")


def _as_float(value: Any) -> float:
    """Render one ``domain_start`` -- a Python number or a 0-d tensor -- as a float.

    Args:
        value: The segment's start time in seconds.

    Returns:
        The value as a ``float``.
    """
    if isinstance(value, torch.Tensor):
        return float(value.reshape(()).item())
    return float(value)


__all__ = ["DENSE_STAGES", "SeqVaeLagAttnCfsTask"]
