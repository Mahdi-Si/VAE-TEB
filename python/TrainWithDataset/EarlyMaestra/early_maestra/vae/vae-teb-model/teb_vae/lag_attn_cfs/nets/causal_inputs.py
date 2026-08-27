r"""The input-side half of one-sidedness: the warm-up mask, the lag floor and the anchor tiling.

:class:`CausalWarmupInputs` is the counterpart of
:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget`. That mixin
owns what changes about the *target* when the coefficients contain no future; this one owns what
changes about the *inputs* and about which anchors are decoded at all. Together they are the whole
of this target domain, and neither names an encoder -- which is what lets both cells of the encoder
axis compose the same two objects and differ only in the architecture underneath them.

Three additions, each of which exists because the coefficients are one-sided:

**The input warm-up.** A one-sided filter's output before $W'_c$ is a function of assumed
pre-recording history, and the stored values there are *real floats* -- not zeroed, not NaN --
normalised with constants accumulated while deliberately excluding exactly that region. They are
masked away and announced inside the availability adapter, which already holds the tensor that does
both. The vectors travel under **new** constructor keywords, ``target_warmup_steps`` and
``source_warmup_steps``: ``target_delays`` reaches ``ChannelDelay``, which *shifts* rather than
masks, and no model composing this mixin accepts it.

**The channel alignment.** A one-sided filter is also *stale* by its own composed group delay, and
those delays span thirteen minutes across a stream -- so reading the whole channel vector at one
step index asserts an instant the entries do not share. When a reference is configured the gate
stops being a pure gather and shifts each survivor by $d_c$ onto it, which is the one place in this
family where ``ChannelDelay`` does anything. The mask must then move with it:
:meth:`CausalWarmupInputs._build_adapter` announces $W'_c + d_c$, because a gathered-and-delayed
channel is honest only once the step index has reached both.

**The anchor tiling.** The forecast cannot begin at the model's own $30$-step warm-up, because the
slowest kept target channel is not honest until step $134$; and once the floor is that high the
dense anchor range costs a $(B, 136, H, C)$ tensor five times over for windows that overlap
$(H-1)/H$. The forward therefore decodes a *tiled* anchor set,

$$\mathcal A(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{valid}} \,\},$$

returning the indices and their validity companion so the objective, the diagnostics and the figures
all read the same set. Both $\varphi$ and $S$ are **arguments**, never derived from ``self.training``:
the diagnostic callback calls ``eval()`` during training and then the objective, so a mode-derived
geometry would make ``total_loss`` a function of the dropout switch.

**The lag validity floor.** Lag attention searches back $L - 1$ steps from an anchor, into a region
where much of the source is still inside its own warm-up. ``lag_floor`` generalises the mask from
$\mathbb 1[t - \ell \ge 0]$ to $\mathbb 1[t - \ell \ge F_u]$; it ships at $0$, where it is bitwise
the sibling's, and exists so the residual is measurable rather than argued about.

**Why the constructor is not here.** Every other member below is identical text for both cells, but
``__init__`` cannot be: the experiment driver builds a run's kwargs by sweeping
``inspect.signature(MODEL_CLS.__init__)``, so each cell has to write out *its own* architecture's
keyword list in full -- a ``**kwargs`` signature would forward four keys and silently build an
all-defaults model. What is shared instead is the work: :meth:`CausalWarmupInputs._set_causal_inputs`
before the base constructor and :meth:`CausalWarmupInputs._validate_causal_geometry` after it, so
each cell's constructor is a signature plus two calls and holds no validation of its own.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter
from teb_vae.lag_attn_rws.nets.model import SATURATION_FRAC

#: The constructor keywords this mixin owns, i.e. the ones a composing model must **not** forward to
#: its base under their own names. Named once here rather than repeated in each cell's ``locals()``
#: filter: a keyword added to the mixin and forgotten in one cell's filter would be passed to a base
#: that does not take it, which is loud -- but a keyword *removed* here and left in a filter would be
#: dropped on the floor, which is not.
#:
#: The last two are the exception that proves the rule: they *are* forwarded, but **renamed** to the
#: base's ``target_delays`` / ``source_delays``, which is why they must be excluded here or they
#: would arrive twice. Their own names are new rather than the base's because a run configures a
#: *reference* and gets shifts, while the base's names carry the two-sided reach guard's -- and
#: these cells still refuse those by name, since the reach quantile is measured on a bank that did
#: not produce these coefficients.
CAUSAL_ONLY_KEYWORDS: Tuple[str, ...] = (
    "target_warmup_steps",
    "source_warmup_steps",
    "anchor_stride",
    "lag_floor",
    "target_weight_st",
    "target_weight_ph",
    "target_align_delays",
    "source_align_delays",
    "target_novelty_frac",
)

#: What a composing constructor removes from its own ``locals()`` before forwarding the rest to the
#: base. ``__class__`` appears because a method referencing ``super()`` gets an implicit closure
#: cell of that name in its locals.
FORWARDED_EXCLUSIONS: Tuple[str, ...] = ("self", "__class__") + CAUSAL_ONLY_KEYWORDS


class CausalWarmupInputs:
    r"""The warm-up mask, the lag floor and the tiled anchor set, for any encoder architecture.

    A plain-object mixin, placed **first** in a composing model's bases so its ``_build_adapter``,
    ``build_lag_mask`` and ``forward`` win method resolution over the architecture's own. It is
    always composed alongside
    :class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget`, whose
    ``_check_anchor_floor``, ``_resolve_target_warm_frac``, ``_resolve_warm_tertiles``,
    ``_resolve_novelty_tertiles``, ``_resolve_block_warm_steps`` and ``SOURCE_BLOCK_SPLIT`` the
    geometry resolution below calls --
    the two mixins are the two halves of one target domain and neither is meaningful without the
    other.

    Attributes:
        target_warmup_steps: The per-survivor warm-up the target stream is masked and announced at,
            in decimated steps of the trimmed window, or ``None`` for a stream with no budget.
            Public because the diagnostic figures draw the boundary and must draw the one the model
            was built with, not a re-resolved guess at it.
        source_warmup_steps: The same for the source stream.
        anchor_stride: $S$, the spacing between decoded anchors.
        lag_floor: $F_u$, the earliest source step lag attention may read.
    """

    target_warmup_steps: Optional[Tuple[int, ...]]
    source_warmup_steps: Optional[Tuple[int, ...]]
    anchor_stride: int
    lag_floor: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _set_causal_inputs(
        self,
        *,
        horizon: int,
        target_keep_index: Optional[Sequence[int]],
        target_warmup_steps: Optional[Sequence[int]],
        source_keep_index: Optional[Sequence[int]],
        source_warmup_steps: Optional[Sequence[int]],
        anchor_stride: int,
        lag_floor: int,
    ) -> None:
        r"""Validate this mixin's four keywords and set them, **before** the base constructor runs.

        Before, and not after, because the base's constructor builds the input adapters and
        :meth:`_build_adapter` reads two of these to do it. Everything checked here is checkable
        without the geometry the base resolves; what needs it is in
        :meth:`_validate_causal_geometry`.

        Args:
            horizon: $H$, which bounds the stride -- above it the decoded windows leave gaps, so
                target steps between two tiles would never be scored at any phase.
            target_keep_index: The target stream's surviving-channel indices, read only to refuse
                the one arrangement that would misroute a warm-up vector.
            target_warmup_steps: $W'_c$ per **surviving** target channel, positional against
                ``target_keep_index``. ``None`` builds no warm-up mask, which is the ungated model.
            source_keep_index: The same for the source stream.
            source_warmup_steps: The same for the source stream.
            anchor_stride: $S$, the spacing between decoded anchors, in $[1, H]$.
            lag_floor: $F_u$, the earliest source step lag attention may read.

        Raises:
            ValueError: If ``anchor_stride`` is outside $[1, H]$, if ``lag_floor`` is negative, or
                if a warm-up vector arrives without its keep-index.
        """
        if not 1 <= int(anchor_stride) <= int(horizon):
            raise ValueError(
                f"anchor_stride must be in [1, horizon] = [1, {int(horizon)}], got "
                f"{anchor_stride}. Below 1 there is no anchor set; above the horizon the decoded "
                f"windows leave gaps, so target steps between two tiles would never be scored at "
                f"any phase."
            )
        if int(lag_floor) < 0:
            raise ValueError(f"lag_floor must be >= 0, got {lag_floor}")

        # The two vectors are positional against their keep-indices, and the adapters are told
        # apart by which gate they were built for -- so a warm-up arriving without its keep-index
        # would leave both gates ``None`` and route the target's vector into both streams.
        for stream, warmup, keep in (
            ("target", target_warmup_steps, target_keep_index),
            ("source", source_warmup_steps, source_keep_index),
        ):
            if warmup is not None and keep is None:
                raise ValueError(
                    f"{stream}_warmup_steps was given without {stream}_keep_index. The warm-up "
                    f"vector is positional against the surviving channels, so the two are "
                    f"resolved together and travel together; a keep-index covering every channel "
                    f"is still an explicit keep-index."
                )

        self.target_warmup_steps = None if target_warmup_steps is None else tuple(
            int(step) for step in target_warmup_steps
        )
        self.source_warmup_steps = None if source_warmup_steps is None else tuple(
            int(step) for step in source_warmup_steps
        )
        self.anchor_stride = int(anchor_stride)
        self.lag_floor = int(lag_floor)

    def _validate_causal_geometry(self) -> None:
        r"""Check what only the base's resolved geometry can decide, and resolve the readouts.

        Called immediately after ``super().__init__``. Two refusals and one resolution, in that
        order: a stride wider than the anchor span leaves some phase with no anchor at all; a floor
        below what the kept target channels require scores assumed pre-recording history as signal;
        and the four readout constants are functions of the resolved budget and the geometry, so
        they are resolved once here rather than per batch.

        The floor check is given the gate's **shifts** as well as the warm-up, read off the built
        gate rather than off a constructor argument for the reason :meth:`_build_adapter` gives.
        An unaligned gate hands it a vector of zeros, which is the argument's inert value and
        reproduces the refusal this cell has always made.

        Raises:
            ValueError: If the stride leaves a phase with no anchor, or if ``warmup_period`` is
                below the floor the kept channels require.
        """
        if self.anchor_stride > self.geometry.t_valid - self.warmup_period:
            raise ValueError(
                f"anchor_stride={self.anchor_stride} leaves no anchor at phase "
                f"{self.anchor_stride - 1}: the first would be "
                f"{self.warmup_period + self.anchor_stride - 1}, against T_valid="
                f"{self.geometry.t_valid} and warmup_period={self.warmup_period}. A sample drawn "
                f"at that phase would contribute no forecast at all, and nothing downstream "
                f"reports an empty anchor row."
            )
        self._check_anchor_floor(
            self.warmup_period,
            self.target_warmup_steps or (),
            ()
            if self.target_gate is None
            else tuple(int(shift) for shift in self.target_gate.delay.delay_steps),
        )
        self._resolve_warmup_readout_constants()

    def _resolve_warmup_readout_constants(self) -> None:
        r"""Resolve the five constants the added readouts are computed against.

        Every one of them is a function of the resolved budget and the geometry alone, so resolving
        them here rather than per batch is not an optimisation: ``target_warm_frac`` is identically
        $1.0$ under the constructor's own pairing refusal, and a per-batch recomputation of it would
        be a tautology evaluated once a step against a four-dimensional density the objective
        deliberately does not carry. The other four are partitions, and a partition recomputed per
        batch is a partition that can differ between two batches of one run.

        The four tensors are registered as **non-persistent** buffers, like every other
        budget-shaped tensor in the family: their contents follow the resolved budget, so a
        persistent copy would make a checkpoint trained at one budget fail to load at another and
        report it as misaligned keys rather than as a budget mismatch. Registering them at all --
        rather than keeping plain tensors -- is what carries them when the module moves device.
        """
        self.target_warm_frac = self._resolve_target_warm_frac(
            self.warmup_period,
            self.horizon,
            self.geometry.t_valid,
            self.target_warmup_steps or (),
        )
        declared_target = (
            torch.arange(self.c_y)
            if self.target_gate is None
            else self.target_gate.keep_index.cpu()
        )
        kept_target = int(declared_target.numel())

        self.register_buffer(
            "warm_tertile_id",
            torch.tensor(
                self._resolve_warm_tertiles(
                    self.target_warmup_steps or tuple(0 for _ in range(kept_target))
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )

        # The novelty split, gathered from a DECLARED-width vector through the same keep-index the
        # channel weights use, rather than taken per survivor. It is a readout and not part of the
        # guard, so the ungated comparison arm keeps it while dropping every resolved channel tuple
        # -- and a survivors-length vector would then be positional against a width that arm no
        # longer has.
        #
        # A model built without the vector at all -- the ungated arm, and every unit construction
        # -- falls back to a constant, which ranks the channels by declared index. That is the same
        # degenerate case the warm-up partition above is in on that arm, and it is a partition of
        # the channel axis rather than a measurement of novelty; the three columns stay present, so
        # the metric surface does not depend on the arm. A *gated* run cannot reach it: the mapping
        # from a resolved budget refuses a feature-target model whose shards carry no novelty
        # vector, rather than defaulting one.
        novelty = self.target_novelty_frac
        if novelty is not None and len(novelty) != int(self.c_y):
            raise ValueError(
                f"target_novelty_frac has {len(novelty)} entries against c_y={int(self.c_y)}. It "
                f"is positional into the DECLARED channel axis and gathered through the gate's "
                f"keep-index, so a survivors-length vector would be silently re-indexed and "
                f"partition the wrong channels."
            )
        self.register_buffer(
            "novelty_tertile_id",
            torch.tensor(
                self._resolve_novelty_tertiles(
                    tuple(0.0 for _ in range(kept_target))
                    if novelty is None
                    else tuple(novelty[index] for index in declared_target.tolist())
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )

        # The source split is taken in DECLARED channel coordinates, through the gate's keep-index
        # rather than positionally: the alignment drops the four source channels above the
        # reference, so the survivors' vector is 47 long against a declared 51 and the two no longer
        # agree. A split taken positionally at the boundary would put eleven channels of the
        # second stored source block into the first and report both warmth fractions against the
        # wrong denominators, with nothing failing -- which is the failure this indirection was
        # written for before it could happen.
        declared = (
            torch.arange(self.c_u)
            if self.source_gate is None
            else self.source_gate.keep_index.cpu()
        )
        split = self.SOURCE_BLOCK_SPLIT if self.use_up_st else 0

        # A channel is honest from step $W'_c + d_c$, not from $W'_c$: ``ChannelDelay`` makes
        # encoder step $t$ read stored step $t - d_c$, so the shift postpones the whole pattern.
        # The availability mask in ``_build_adapter`` and the floor in ``_check_anchor_floor``
        # both already add it; this was the third site and the only one that *reported* rather
        # than enforced, which is why nothing failed. Because $d_c \ge 0$ always, the omission
        # could only ever report the source as WARMER than it is -- and on the shipped aligned
        # configuration it made ``source_lag_warmth_frac_st`` identically $1.0$ for any attention
        # distribution, i.e. a column that could not vary and therefore measured nothing.
        waits = self.source_warmup_steps
        if waits is not None and self.source_gate is not None:
            waits = tuple(
                wait + int(shift)
                for wait, shift in zip(waits, self.source_gate.delay.delay_steps)
            )

        # Refused rather than zipped: ``declared`` is the gate's keep-index when there is a gate
        # and the full declared range when there is not, so a stream carrying a warm-up without a
        # gate would pair 47 waits against 51 indices and ``zip`` would silently drop the tail,
        # reporting both fractions against the wrong denominators with every shape still correct.
        if waits is not None and len(waits) != int(declared.numel()):
            raise ValueError(
                f"{len(waits)} source warm-up steps against {int(declared.numel())} declared "
                f"channel indices. These must agree: a stream with a warm-up must carry the gate "
                f"whose keep-index names the channels that warm-up belongs to."
            )

        for name, in_block in (
            ("st", declared < split),
            ("ph", declared >= split),
        ):
            block = (
                []
                if waits is None
                else [
                    int(step)
                    for step, keep in zip(waits, in_block.tolist())
                    if keep
                ]
            )
            self.register_buffer(
                f"source_block_warm_{name}",
                # An ungated stream has no warm-up to wait out, so every step is warm -- which is
                # the same all-True pattern an empty block gets, for a different reason.
                torch.ones(self.sequence_length, dtype=torch.bool)
                if waits is None
                else self._resolve_block_warm_steps(block, self.sequence_length),
                persistent=False,
            )

    # ------------------------------------------------------------------
    # The two overridden construction hooks
    # ------------------------------------------------------------------
    def _build_adapter(
        self, gate: Optional[ChannelGate], declared_width: int, dropout: float
    ) -> AvailabilityInputAdapter:
        r"""Build one stream's adapter at its **warm-up plus its shift**, not at either alone.

        $$\delta^{\mathrm{adapter}}_c \;=\; W'_c + d_c .$$

        Two corrections to the base, one per term. The base reads ``gate.delay.delay_steps`` alone,
        which is the right source for a reach-budget guard and is all zeros in every unaligned
        configuration here: the gate is then a pure gather, so ``max_delay`` would be $0$ and
        **neither** availability term would exist -- no mask, no announcement, and a stream whose
        leading region is real values on no defined scale entering the encoder as though it were
        signal. And the warm-up alone is right only while nothing is shifted: a channel gathered
        from step $t - d_c$ is a function of the recording only once $t - d_c \ge W'_c$, so an
        adapter told $W'_c$ against a shifted stream announces a channel warm by as much as $85$
        steps before it is -- with no crash, no shape change and no metric moving.

        Read off the **gate** rather than off a second copy of the shift, for the reason the base's
        own version gives: the gate fills in a missing delay vector with zeros and a missing
        keep-index with the identity, and either substitution would leave a separately-held vector
        out of step with what the stream actually received.

        The two calls are told apart by which gate they were handed. That is sound because a
        resolved budget always produces both keep-indices, so a stream with a warm-up always has a
        gate, and :meth:`_set_causal_inputs` refuses the one arrangement that would break it.

        Args:
            gate: The stream's channel gather, or ``None`` when it has none.
            declared_width: The stream's declared channel count, used when there is no gate.
            dropout: Dropout probability inside the projection stack.

        Returns:
            The adapter, carrying whichever availability terms the combined vector calls for --
            which under alignment is both, because the minimum of $W'_c + d_c$ is nonzero where the
            warm-up alone reaches $0$ -- $80$ steps at the causal-feature cells' shipped
            reference and $1$ at the raw-target cells' -- so the start-of-record embedding comes
            into existence.
        """
        warmup = (
            self.target_warmup_steps
            if gate is self.target_gate
            else self.source_warmup_steps
        )
        if warmup is None:
            return super()._build_adapter(gate, declared_width, dropout)

        width = declared_width if gate is None else gate.out_channels
        delays = list(warmup)
        if gate is not None:
            delays = [
                wait + int(shift) for wait, shift in zip(delays, gate.delay.delay_steps)
            ]
        return AvailabilityInputAdapter(
            in_dim=width,
            d_model=self.d_model,
            sequence_length=self.sequence_length,
            dropout=dropout,
            delays=delays,
        )

    def build_lag_mask(
        self, seq_len: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        r"""The lag-validity mask, floored: $m_{t,\ell} = \mathbb 1[t - \ell \ge F_u]$.

        At ``lag_floor = 0`` the base's mask is returned unchanged -- the same object, not an
        equal one -- so an unfloored causal model attends bitwise as the sibling does.

        A row with no admissible lag is left all-``False``. The attention normalises such a row to
        zero rather than to NaN, and zero is the right reading: no lag was attended because none was
        available. The consequence is worth stating where the floor is applied -- the lag map sums
        over lags to $K_t$ only where some lag is admissible, so rows below the floor carry a zero
        attribution against a non-zero $K_t$.

        Args:
            seq_len: Sequence length $T$.
            device: Device to build the mask on.

        Returns:
            A boolean $(T, L)$ mask, ``True`` where the lagged source step is readable.
        """
        mask = super().build_lag_mask(seq_len, device=device)
        if self.lag_floor == 0:
            return mask
        steps = torch.arange(seq_len, device=device)[:, None]
        lags = torch.arange(self.lag_attn.L, device=device)[None, :]
        return mask & (steps - lags >= self.lag_floor)

    # ------------------------------------------------------------------
    # The anchor set, and the forward that decodes at it
    # ------------------------------------------------------------------
    def _build_anchor_index(
        self,
        batch: int,
        device: torch.device,
        anchor_phase: Optional[Union[int, torch.Tensor]] = None,
        anchor_stride: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The tiled anchor set for one batch, with its validity companion.

        $$\mathcal A(\varphi) = \{\, F + \varphi + kS \;:\; k \ge 0,\; F + \varphi + kS < T_{\mathrm{valid}} \,\}$$

        The returned width is $A_{\max} = \lceil (T_{\mathrm{valid}} - F)/S \rceil$, a **geometry
        constant**: it does not vary with $\varphi$ or with the batch, so no rank can disagree about
        the shape and no shape is a function of the data. What does vary is how many entries are
        real, and short rows **repeat their last valid anchor** and mark it invalid.

        That padding convention is load-bearing rather than cosmetic. A padded slot holding a
        distinct *legal* anchor would produce a fully live row in the forecast mask, so its target
        block would be gathered and scored a second time -- while the KL support, which is a set,
        would count it once. The two per-anchor denominators would diverge and $\beta$ would quietly
        stop meaning what it means in every other cell of the grid.

        No random number is drawn here, and none may be: $\varphi$ arrives already derived, because
        a draw inside the forward would consume the global RNG stream, move the reparameterisation
        $\epsilon$ and break every bitwise comparison in the suite -- and would not survive a
        checkpoint resume.

        Args:
            batch: Batch size $B$ the anchor set is built for.
            device: Device to build the indices on.
            anchor_phase: $\varphi$ per sample: a $(B,)$ integer tensor, a single ``int`` applied to
                every sample, or ``None``. ``None`` is admitted only at stride $1$, where the set is
                the dense range and a phase would truncate it rather than rotate it.
            anchor_stride: $S$, or ``None`` for the model's configured stride.

        Returns:
            ``(anchor_index, anchor_valid)``: $(B, A_{\max})$ ``long`` and $(B, A_{\max})$ ``bool``.

        Raises:
            ValueError: If the stride is outside $[1, H]$, if a phase is missing at a stride above
                $1$, or if a phase is outside $[0, S)$ -- each naming the offending value.
        """
        stride = self.anchor_stride if anchor_stride is None else int(anchor_stride)
        if not 1 <= stride <= self.horizon:
            raise ValueError(
                f"anchor_stride must be in [1, horizon] = [1, {self.horizon}], got {stride}"
            )

        floor, t_valid = self.warmup_period, self.geometry.t_valid
        span = t_valid - floor

        if anchor_phase is None:
            # Refused rather than defaulted at a real stride: a forgotten phase would train every
            # sample of every epoch on one tile grid at a fixed offset from the segment start, and
            # A_max is a geometry constant either way, so nothing about the shapes would say so.
            if stride > 1:
                raise ValueError(
                    f"anchor_phase is required at anchor_stride={stride}: without it every sample "
                    f"would be decoded at the same tile grid forever, at a fixed offset from the "
                    f"segment start, and no shape or count would differ. Pass a (B,) phase, or "
                    f"decode densely with anchor_stride=1."
                )
            phase = torch.zeros(batch, dtype=torch.long, device=device)
        elif isinstance(anchor_phase, torch.Tensor):
            phase = anchor_phase.to(device=device, dtype=torch.long).reshape(-1)
            if phase.numel() != batch:
                raise ValueError(
                    f"anchor_phase has {phase.numel()} entries but the batch is {batch}; the "
                    f"phase is per sample, so a mismatch would tile one sample at another's grid"
                )
        else:
            phase = torch.full((batch,), int(anchor_phase), dtype=torch.long, device=device)

        if bool(((phase < 0) | (phase >= stride)).any()):
            offending = int(phase[(phase < 0) | (phase >= stride)][0])
            raise ValueError(
                f"anchor_phase {offending} is outside [0, anchor_stride) = [0, {stride}). The "
                f"anchor set truncates rather than rotating, so a phase at or above the stride "
                f"drops leading anchors instead of shifting the grid -- and at stride 1 the only "
                f"admissible phase is 0."
            )

        a_max = -(-span // stride)  # ceil, on ints
        steps = torch.arange(a_max, device=device, dtype=torch.long)
        anchors = floor + phase[:, None] + steps[None, :] * stride
        valid = anchors < t_valid

        # Short rows repeat their last real anchor. `span - phase >= 1` holds because the
        # constructor refuses a stride wider than the span, so every row has at least one.
        count = (span - phase + stride - 1) // stride
        last = floor + phase + (count - 1) * stride
        return torch.where(valid, anchors, last[:, None]), valid

    def forward(
        self,
        y_st: torch.Tensor,
        y_ph: torch.Tensor,
        u_stream: torch.Tensor,
        anchor_phase: Optional[Union[int, torch.Tensor]] = None,
        anchor_stride: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        r"""Run the full pipeline, decoding a tiled anchor set.

        The composed architecture's forward line for line, with two changes: the anchor set is built
        first, and the decoder is invoked on the latents **gathered at those anchors** rather than on
        the contiguous prefix $[0, T_{\mathrm{valid}})$. Everything on the anchor axis is sparse from
        here on -- the four forecast tensors, the target, the forecast mask and every per-anchor
        metric -- and the only place it is scattered back to dense is the KL support, which must be
        $(B, T)$ because the latent tensors it gates are produced at every step.

        The anchor set is built **inside** the forward, and returned, because the forecasts and the
        target must be gathered at the same anchors: a second computation elsewhere could disagree,
        and the disagreement would be a wrong number rather than an exception.

        The body names only submodules both architectures build under the same attribute names --
        the two encoders, the two gates, the two adapters, the prior and posterior heads, the lag
        attention and the shared decoder -- which is what makes one copy of it correct for both.
        Every architectural difference between the cells is inside those modules.

        Args:
            y_st: Target scattering features ``(B, T, 36)``.
            y_ph: Target phase-harmonic features ``(B, T, 66)``.
            u_stream: Source stream ``(B, T, c_u)``.
            anchor_phase: $\varphi$ per sample; see :meth:`_build_anchor_index`. Required once the
                resolved stride exceeds $1$.
            anchor_stride: $S$; ``None`` uses the model's configured stride.

        Returns:
            The base's twenty keys, with ``mu_base``, ``logvar_base``, ``mu_full`` and
            ``logvar_full`` at $(B, A_{\max}, H, C_{\mathrm{keep}})$ rather than
            $(B, T_{\mathrm{valid}}, H, R)$, plus two of this model's own:

            * ``anchor_index`` -- the decoded anchors $(B, A_{\max})$, ``torch.long``.
            * ``anchor_valid`` -- which of them are real $(B, A_{\max})$, ``torch.bool``. Padded
              slots repeat the row's last real anchor and are ``False`` here.
        """
        anchor_index, anchor_valid = self._build_anchor_index(
            batch=int(y_st.shape[0]),
            device=y_st.device,
            anchor_phase=anchor_phase,
            anchor_stride=anchor_stride,
        )

        # Concatenate at the declared widths, then gate: the surviving-channel indices are
        # positional into the full stream. The gate gathers the survivors and then delays each by
        # its own d_c, while the warm-up stays a leading mask rather than part of the gate -- and
        # the masking happens inside the adapter, which holds the same vector it announces with.
        target = torch.cat([y_st, y_ph], dim=-1)
        if self.target_gate is not None:
            target = self.target_gate(target)
        source = u_stream if self.source_gate is None else self.source_gate(u_stream)

        h_y = self.target_encoder(self.target_adapter(target))
        h_u = self.source_encoder(self.source_adapter(source))

        mu_prior, logvar_prior, raw_logvar_prior = self.prior_head(h_y)

        # The attended output (W_o's fused projection) is discarded: the head-structured posterior
        # consumes the per-head summaries, and W_o is frozen. The query is posed from the prior
        # belief -- mu^p, or [mu^p || logvar^p] under query_uses_logvar -- both target-only.
        query = (
            torch.cat([mu_prior, logvar_prior], dim=-1)
            if self.query_uses_logvar
            else mu_prior
        )
        _, alpha, attended_heads = self.lag_attn(
            self.query_proj(query), h_u, self.build_lag_mask(h_u.shape[1], h_u.device)
        )

        mu_post, logvar_post = self.posterior_head(
            h_y, attended_heads, mu_prior, raw_logvar_prior
        )
        z_prior, z_post = self._reparameterize_shared(
            mu_prior, logvar_prior, mu_post, logvar_post
        )

        # Saturation diagnostics: a bound that is always active is a bound that is binding, and a
        # binding bound is a silently mis-set hyperparameter.
        with torch.no_grad():
            mu_prior_sat_frac = (mu_prior.abs() >= (SATURATION_FRAC * self.mu_scale)).float().mean()
            delta_mu_sat_frac = (
                (mu_post - mu_prior).abs() >= (SATURATION_FRAC * self.delta_mu_scale)
            ).float().mean()

        # Decode at the tiled anchors. A gather rather than a slice, because the anchors are no
        # longer a contiguous prefix; and the same index for both branches, so base and full are
        # two latents through one decoder at one anchor set.
        gather_index = anchor_index[:, :, None].expand(-1, -1, self.d_z)
        mu_base, logvar_base = self.decoder(z_prior.gather(1, gather_index))
        mu_full, logvar_full = self.decoder(z_post.gather(1, gather_index))

        # The per-lag attribution: K_t says how much the source moved the belief, the attention
        # weights say from which lag. Head-structured, so the split is an additive decomposition
        # rather than an arbitrary slice of a shared latent.
        kld_btd = self.kld_tensor(
            mu_prior=mu_prior,
            logvar_prior=logvar_prior,
            mu_post=mu_post,
            logvar_post=logvar_post,
        )
        kld_per_t, source_kl_lag_map, kld_per_t_per_head = self.te_analysis(
            kld_btd, alpha, head_structured=True
        )

        return {
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "raw_logvar_prior": raw_logvar_prior,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "z_prior": z_prior,
            "z_post": z_post,
            "target_state": h_y,
            "source_state": h_u,
            "attended_source_heads": attended_heads,
            "attn_weights": alpha,
            "mu_base": mu_base,
            "logvar_base": logvar_base,
            "mu_full": mu_full,
            "logvar_full": logvar_full,
            "kld_per_t": kld_per_t,
            "kld_per_t_per_head": kld_per_t_per_head,
            "source_kl_lag_map": source_kl_lag_map,
            "mu_prior_sat_frac": mu_prior_sat_frac,
            "delta_mu_sat_frac": delta_mu_sat_frac,
            "anchor_index": anchor_index,
            "anchor_valid": anchor_valid,
        }


__all__ = ["CAUSAL_ONLY_KEYWORDS", "FORWARDED_EXCLUSIONS", "CausalWarmupInputs"]
