r"""The one-sided feature target: what changes when the coefficients contain no future.

:class:`CausalFeatureForecastTarget` is
:class:`~teb_vae.lag_attn_fs.nets.feature_target.FeatureForecastTarget` with two channel-layout
constants, one refusal, and the readouts one anchor set of its own makes possible. What it does
**not** override is the point: the decoder's width, the gathered-never-delayed target block and the
delegation that hands the shared objective both are inherited unchanged, and so is the anchor seam,
which lives in the shared objective and the shared masks rather than in any subclass.

**The block split moves.** The one-sided cascade drops the seven slowest scattering channels per
block at write time -- their warm-up outruns the stored segment at every trim -- so the first stored
block is $36$ channels wide here against the two-sided $43$. Nothing but two reported numbers depends
on it, which is exactly why it is checked against the data rather than left declared.

**The four resolved gaps must be computed at the anchors that were decoded.** The parent builds its
own mask with no anchor set, which is correct for a model that decodes every anchor and wrong for one
that decodes a tile. Left alone, all four splits would be averaged over the dense anchor range while
the ``pred_gap`` they are read against is averaged over the tiles, and the recomposition that makes
them a decomposition rather than four unrelated numbers would not hold.

**The ten added readouts live here for the same reason.** ``_resolved_forecast_gaps`` is the
family's per-package metric hook -- the parent's ``compute_loss`` merges whatever it returns -- and
it is the only one available, because the anchor seam is in the shared objective rather than in a
subclass. Two of the ten are geometry *guards* rather than results: ``target_warm_frac`` must
read exactly $1.0$ and ``anchors_per_sample`` must sit at its geometry-derived value, and a row
outside either means the geometry broke rather than that the model learned something. Six are two
three-way partitions of the same per-channel gap -- by warm-up rank and by the shard's stored
novelty share -- which recompose to ``pred_gap`` over one denominator and answer two different
questions about the same channels.

**What is deliberately not here.** ``_default_decoder_out_channels``, ``compute_loss`` and
``_build_forecast_target`` are the parent's, unmodified: the anchor set reaches all three through
``forward_outputs['anchor_index']``, so the seam is expressed once in the shared code rather than
three times in a subclass. And nothing here mentions an encoder, an adapter or a lag mask -- the
input-side half of causality lives on the model, which is what lets a second architecture compose
this same mixin.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask

#: How many contiguous groups the kept target channels are partitioned into by rank -- by warm-up
#: for one split and by novelty for the other. Three because the question both splits ask -- do the
#: slow channels forecast differently from the fast ones? -- needs a middle to say "monotone" rather
#: than "different", and because the two reported block gaps already cut the same axis a second way.
WARM_TERTILES = 3

#: What counts as a warm *block* at a step: at least half its channels past their own warm-up. A
#: fraction rather than "all" or "any" because both extremes are degenerate on the real vectors --
#: the first stored source block has a channel at $W' = 278$ against a $300$-step window, so "all"
#: is almost never true, and its fastest channel is warm at step $0$, so "any" is almost always
#: true.
WARM_BLOCK_FRACTION = 0.5


class CausalFeatureForecastTarget(FeatureForecastTarget):
    r"""The feature target, re-pointed at the one-sided channel layout and a tiled anchor set.

    Mixed in ahead of an encoder model exactly as its parent is, and for the same reason: it names
    no encoder, so both cells of the encoder axis compose it.

    Six things the composing model's constructor must resolve and set, declared here so the
    contract is written where the readouts that consume it are. Each is a constant of the resolved
    budget and the geometry, which is why none of them is recomputed per batch; the three patterns
    are tensors so that a model moved to another device carries them.

    Attributes:
        target_warm_frac: The constant :meth:`_resolve_target_warm_frac` returns.
        warm_tertile_id: ``(C_keep,)`` long, the tertile assignment of
            :meth:`_resolve_warm_tertiles`.
        target_novelty_frac: $\nu_c$ per **declared** target channel as
            :meth:`_set_target_novelty` stashed it, or ``None``.
        novelty_tertile_id: ``(C_keep,)`` long, the tertile assignment of
            :meth:`_resolve_novelty_tertiles`. Cuts the same channel axis by how much of each
            coefficient the anchor has *not* already seen, which is a different question from how
            long it took to become honest and is answered by a different stored vector.
        source_block_warm_st: ``(T,)`` bool, the **first** stored source block's per-step warmth
            from :meth:`_resolve_block_warm_steps`.
        source_block_warm_ph: ``(T,)`` bool, the same for the second.
    """

    target_warm_frac: float
    target_novelty_frac: Optional[Tuple[float, ...]]
    warm_tertile_id: torch.Tensor
    novelty_tertile_id: torch.Tensor
    source_block_warm_st: torch.Tensor
    source_block_warm_ph: torch.Tensor

    #: How many of the declared $c_y$ target channels belong to the **first** stored block.
    #:
    #: $36$, not the two-sided $43$: seven scattering channels per block were dropped at write time
    #: because their one-sided warm-up outruns the stored segment at every trim. Both phase blocks
    #: keep their full width -- their $0.008$ Hz band floor excludes those filters entirely -- so
    #: the second block is unchanged at $66$ and $c_y = 102$.
    #:
    #: Declared here rather than inherited, so that a change to the two-sided split cannot move
    #: this one and vice versa. It splits two reported numbers and feeds no loss, no shape and no
    #: parameter, which is why a wrong value would mislabel rather than fail -- and why the suite
    #: checks it against the width of the block the target is actually assembled from.
    TARGET_BLOCK_SPLIT: int = 36

    #: How many of the declared $c_u$ **source** channels belong to the **first** of the two stored
    #: source blocks -- the boundary ``source_lag_warmth_frac_st`` and ``_ph`` are reported either
    #: side of.
    #:
    #: The split is not decoration and a pooled figure would hide the thing it exists to show. The
    #: two blocks' rebased warm-ups are $0 \ldots 278$ and $41 \ldots 134$: the first is warm from
    #: step $0$ in its fastest channels, so pooling the two would let it carry the fraction while
    #: almost no channel of the second is warm at the far lags -- and the second is the block with
    #: the problem, its whole band being built from wavelets slower than $0.05$ Hz.
    #:
    #: A class constant for exactly the reasons :attr:`TARGET_BLOCK_SPLIT` is one, and resolved to
    #: $0$ on a model built without the first source block, where the second is the whole stream and
    #: there is no first block to report.
    SOURCE_BLOCK_SPLIT: int = 36

    @staticmethod
    def _check_anchor_floor(
        warmup_period: int,
        kept_warmup_steps: Sequence[int],
        kept_align_delays: Sequence[int] = (),
    ) -> None:
        r"""Refuse an anchor floor the kept channels do not admit, on either of two counts.

        $$F \;\ge\; \max\Bigl(\underbrace{B - 1}_{\text{scored target}},\;
          \underbrace{\max_c\bigl(W'_c + d_c\bigr)}_{\text{input warmth}}\Bigr),
          \qquad B = \max_{c \in \mathrm{kept}} W'_c .$$

        **The scored-target requirement**, which is the one this cell has always enforced. A
        forecast at anchor $t$, horizon step $\tau$, reads target time $t + 1 + \tau$, and channel
        $c$ is honest there only from $W'_c$ onwards; requiring every kept channel to be valid
        across every anchor's whole window collapses to $t + 1 \ge W'_c$ for all $t \ge F$, i.e.
        $F \ge B - 1$. The inequality is $\ge B - 1$ rather than $\ge B$ because the earliest
        target step an anchor reads is $t + 1$, not $t$; a floor of exactly $B - 1$ is the shipped
        configuration and must be admitted. The **target tile is never shifted** -- an aligned
        feature-domain forecast cannot be a strict forecast at its first horizon element -- so this
        half is taken over the unshifted warm-up whatever the input stream does.

        **The input-warmth requirement**, which binds only once the inputs are *shifted*. With
        $d_c = 0$ the input at step $t$ is the stored coefficient at $t$, a cold one is masked and
        announced inside the availability adapter, and that masking is the policy this family
        ships: it is why a floor of $133$ is admitted against $B = 134$ on an unshifted stream, with
        the slowest channel still cold at the anchor itself and honest exactly at $t + 1$. Under a nonzero
        shift the claim changes. The gathered channel vector at step $t$ asserts that its entries
        describe one physical instant, and that assertion is false while any of them has not
        arrived -- so the anchor must clear $W'_c + d_c$ for every kept channel, not $t+1$. At the
        shipped reference that is $134$, and the requirement costs exactly one anchor.

        Both halves are enforced rather than assumed, because both alternatives are silent: the
        objective's mask is $(B, A, H)$ and broadcasts over channels, so a floor one step too low
        scores the assumed pre-recording history of the slowest kept channel as though it were
        signal, with every shape correct and every warm-fraction readout still reporting $1.0$.

        Args:
            warmup_period: The anchor floor $F$ the model was built at.
            kept_warmup_steps: $W'_c$ per surviving target channel.
            kept_align_delays: $d_c$ per surviving target channel, positional against
                ``kept_warmup_steps``. Empty, or all zeros, is the unaligned stream -- where the
                second requirement does not apply at all, not where it happens to be satisfied.

        Raises:
            ValueError: If the floor is below either requirement, naming which one binds, the
                channel that binds it, and both numbers.
        """
        if not kept_warmup_steps:
            return
        waits = [int(step) for step in kept_warmup_steps]
        shifts = [int(shift) for shift in kept_align_delays]
        if shifts and len(shifts) != len(waits):
            raise ValueError(
                f"kept_align_delays has {len(shifts)} entries against {len(waits)} kept warm-up "
                f"steps. Both are positional over the same surviving channels, so a length "
                f"mismatch would pair one channel's wait with another's shift and refuse -- or "
                f"admit -- a floor computed for a stream that does not exist."
            )

        budget = max(waits)
        required = budget - 1
        binding = "the scored target's validity"
        detail = (
            f"the slowest kept target channel is honest only from step {budget}, and a forecast "
            f"at anchor t reads target step t + 1 at the earliest"
        )
        consequence = (
            "Below it the objective scores assumed pre-recording history as signal, on "
            "coefficients normalised with constants that excluded exactly that region"
        )

        # Emptiness and all-zeros are one case on purpose: an unshifted stream is exactly the one
        # whose channels are masked and announced rather than waited for, and a caller that passes
        # the gate's inert delay vector must get the same answer as one that passes nothing.
        if shifts and max(shifts) > 0:
            combined = [wait + shift for wait, shift in zip(waits, shifts)]
            index = max(range(len(combined)), key=combined.__getitem__)
            if combined[index] > required:
                required = combined[index]
                binding = "the shifted inputs' warmth"
                detail = (
                    f"kept target channel {index} is gathered from step t - {shifts[index]} and "
                    f"its own warm-up is {waits[index]} steps, so it is honest at the anchor only "
                    f"from step {required}"
                )
                consequence = (
                    "Below it the aligned channel vector claims one physical instant while an "
                    "entry of it has not arrived, which is the whole property the shift applies "
                    "the channels for"
                )

        if int(warmup_period) < required:
            raise ValueError(
                f"warmup_period={int(warmup_period)} is below the anchor floor {binding} "
                f"requires: {detail}, so the floor must be at least {required}. {consequence} -- "
                f"with every shape correct and nothing reporting it."
            )

    @staticmethod
    def _resolve_target_warm_frac(
        warmup_period: int,
        horizon: int,
        t_valid: int,
        kept_warmup_steps: Sequence[int],
    ) -> float:
        r"""The share of scored target coefficients whose channel is past its own warm-up.

        Over the triples the objective can ever score -- anchors $t \in [F, T_{\mathrm{valid}})$,
        horizon steps $\tau \in [0, H)$, kept channels $c$ -- the fraction satisfying
        $t + 1 + \tau \ge W'_c$.

        **Resolved once, from the geometry, and emitted as a constant column.** Given the
        constructor's pairing refusal and the anchor range this is identically $1.0$, so
        recomputing a would-be four-dimensional density every step would be a tautology evaluated
        per batch -- and the four-dimensional mask it would need is the one section-by-section
        rejected: every denominator of a loss term is an anchor count, so a valid-channel count
        that varied per anchor would make nats-per-anchor shrink silently with mask density.

        What the column is for is **provenance**: a value other than $1.0$ on a logged row means
        the checkpoint was built by code that predates the pairing refusal, which is a fact worth
        being able to read off a run months later.

        Args:
            warmup_period: The anchor floor $F$.
            horizon: $H$, forecast steps per anchor.
            t_valid: $T_{\mathrm{valid}} = T - H$, one past the last anchor.
            kept_warmup_steps: $W'_c$ per surviving target channel. Empty -- the ungated model --
                gives $1.0$, which is right: with no warm-up every coefficient is honest.

        Returns:
            The fraction, in $[0, 1]$.
        """
        anchors = range(int(warmup_period), int(t_valid))
        horizon = int(horizon)
        if not kept_warmup_steps or horizon <= 0 or len(anchors) == 0:
            return 1.0
        warm = 0
        for step in kept_warmup_steps:
            for anchor in anchors:
                # Horizon step tau reads target step t + 1 + tau, so the cold ones are exactly
                # tau < W'_c - t - 1, clipped into [0, H].
                warm += horizon - min(horizon, max(0, int(step) - anchor - 1))
        return warm / float(len(anchors) * horizon * len(kept_warmup_steps))

    @staticmethod
    def _rank_tertiles(values: Sequence[float]) -> Tuple[int, ...]:
        """Partition a per-channel vector into three contiguous groups by ascending rank.

        Shared by the two splits below rather than written twice, because the *rule* is the part
        that must not differ between them: two partitions computed by two implementations would be
        two different denominators, and both recompose to the same ``pred_gap``.

        The partition is by **rank**, not by value, into groups as equal in size as the count
        allows -- so the boundaries move when the resolved vector moves rather than sitting at
        declared thresholds that a rebuilt dataset would invalidate. Ties are broken by declared
        channel index, which is what makes the assignment a function of the vector alone.

        Args:
            values: One number per surviving target channel. Group $0$ takes the smallest.

        Returns:
            One tertile id in $[0, 3)$ per channel, positional against the input.
        """
        count = len(values)
        if count == 0:
            return ()
        order = sorted(range(count), key=lambda index: (float(values[index]), index))
        assignment = [0] * count
        for rank, channel in enumerate(order):
            # Floor rather than round: it puts any remainder in the last group, so the three sizes
            # differ by at most one whatever the count.
            assignment[channel] = min(WARM_TERTILES - 1, rank * WARM_TERTILES // count)
        return tuple(assignment)

    @classmethod
    def _resolve_warm_tertiles(cls, kept_warmup_steps: Sequence[int]) -> Tuple[int, ...]:
        r"""Assign each kept target channel to a warm-up tertile: group $0$ is the shortest wait.

        The three groups cut **across** the stored block boundary and are therefore not a
        restatement of ``pred_gap_st`` / ``pred_gap_ph``: at the shipped budget the kept set is
        $32$ channels of the first stored block plus all $66$ of the second, and the two span
        nearly the same rebased range.

        Args:
            kept_warmup_steps: $W'_c$ per surviving target channel.

        Returns:
            One tertile id in $[0, 3)$ per kept channel, positional against the kept axis.
        """
        return cls._rank_tertiles([int(step) for step in kept_warmup_steps])

    @classmethod
    def _resolve_novelty_tertiles(cls, kept_novelty_frac: Sequence[float]) -> Tuple[int, ...]:
        r"""Assign each kept target channel to a novelty tertile: group $0$ is the least new.

        $\nu_c$ is the share of a target coefficient drawn from raw samples the anchor has **not**
        seen, measured on the composed kernel at the shipped horizon and stored on the shard. It is
        not a restatement of the warm-up split even though both descend from the same filter
        ladder: the warm-up says when a channel became a function of the recording at all, and the
        novelty says how much of what it reports at a *scored* step is still ahead of the anchor.
        A channel can be warm for the whole window and carry $\nu = 0.026$.

        What the split is for. ``pred_gap`` is an unweighted sum over $H \cdot C_{\mathrm{keep}}$
        coefficients that mixes two different claims: on a high-$\nu$ channel a good score is a
        forecast, and on a low-$\nu$ one it is the model inverting its own differently-delayed
        history. Both are worth having and they are not the same thing, so the pooled number is
        reported split rather than only pooled.

        Args:
            kept_novelty_frac: $\nu_c$ per surviving target channel, each in $[0, 1]$.

        Returns:
            One tertile id in $[0, 3)$ per kept channel, positional against the kept axis.
        """
        return cls._rank_tertiles([float(share) for share in kept_novelty_frac])

    def _set_target_novelty(
        self, *, target_novelty_frac: Optional[Sequence[float]]
    ) -> None:
        r"""Stash the declared novelty vector, **before** the base constructor runs.

        Split from :meth:`_set_channel_weights` only because they are set from different keywords;
        split from ``_set_causal_inputs`` for the reason that method's own docstring gives -- the
        raw-target causal cells compose that mixin too, and a raw block's last axis counts samples
        of one signal, every one of which lies after the anchor, so there is no per-channel novelty
        to take. Putting this keyword there would make it a required argument of a call those cells
        make.

        In **declared** channel coordinates, gathered through the keep-index by the input mixin's
        ``_resolve_warmup_readout_constants``, exactly as ``target_channel_weight`` is. Positional over
        the survivors it could not be: the ungated comparison arm is built by removing the resolved
        channel tuples and keeping the readouts, and a survivors-length vector there would be
        positional against a width that no longer exists.

        Args:
            target_novelty_frac: $\nu_c$ per **declared** target channel, or ``None`` for a model
                built without one -- the ungated arm and every unit construction. See
                :meth:`_register_novelty_tertiles` for what ``None`` then reports.
        """
        self.target_novelty_frac = (
            None
            if target_novelty_frac is None
            else tuple(float(share) for share in target_novelty_frac)
        )

    def _set_channel_weights(self, *, target_weight_st: float, target_weight_ph: float) -> None:
        r"""Stash the two per-block weights, **before** the base constructor runs.

        Split from :meth:`_register_channel_weights` for the same reason the causal input mixin
        splits its own two calls: this half needs no geometry and that half needs a built
        ``nn.Module``. Split from ``_set_causal_inputs`` for a different and more important reason
        -- the *raw*-target causal cells compose that mixin too, and a raw block's last axis counts
        samples of one signal, so there are no stored blocks to weight and no split to take. Putting
        these keywords there would make them required arguments of a call those cells make.

        Args:
            target_weight_st: Relative reconstruction weight of the first stored target block.
            target_weight_ph: The same for the second.
        """
        self.target_weight_st = float(target_weight_st)
        self.target_weight_ph = float(target_weight_ph)

    def _register_channel_weights(self) -> None:
        r"""Resolve the stashed pair into the ``target_channel_weight`` buffer, after the base.

        In **declared** channel coordinates through the gate's keep-index rather than positionally:
        the survivors are not contiguous, so a positional split would put the block boundary in the
        wrong place the moment the budget drops a channel.

        Registered non-persistent, like every budget-shaped tensor in this family: its length is
        $C_{\mathrm{keep}}$, so a persistent copy would make a checkpoint trained at one budget fail
        to load at another and report it as a missing key rather than as a budget mismatch. The two
        scalars reach the checkpoint through ``model_kwargs`` instead, which is what makes a run's
        objective recoverable from its checkpoint alone.
        """
        declared = (
            torch.arange(self.c_y)
            if self.target_gate is None
            else self.target_gate.keep_index.cpu()
        )
        self.register_buffer(
            "target_channel_weight",
            self._resolve_channel_weights(
                declared.tolist(),
                weight_st=self.target_weight_st,
                weight_ph=self.target_weight_ph,
            ),
            persistent=False,
        )

    @classmethod
    def _resolve_channel_weights(
        cls,
        keep_index: Sequence[int],
        *,
        weight_st: float,
        weight_ph: float,
    ) -> torch.Tensor:
        r"""The per-kept-channel loss weight, **renormalised to leave the block scale alone**.

        The two stored target blocks are produced by different transforms and weighting them
        equally is a choice rather than a neutral default -- and at the shipped budget it is not
        even a neutral *count*: $66$ of the $98$ survivors are phase-harmonic, so a uniform
        objective already spends two-thirds of itself there. This returns $w_c$ per surviving
        channel, positional against ``keep_index``, from one weight per stored block.

        **The renormalisation is the load-bearing part.** The raw weights are scaled by
        $C_{\mathrm{keep}} / \sum_c w_c$, so the weighted block sums to the same magnitude the
        unweighted one did. Without it, $(1.0, 0.1)$ would shrink the reconstruction by $2.5\times$
        against a KL that did not move -- which is a $2.5\times$ increase in the effective $\beta$
        wearing a channel-weight's name, and it would put ``gradient_clip_val`` and
        ``additive_margin`` back out of date on both cfs cells. Only the *distribution* across
        channels moves; the ratio the configuration states is preserved exactly.

        Equal weights return exactly ones: the scale is then $C_{\mathrm{keep}}/C_{\mathrm{keep}}
        = 1$, so a model configured at $(1.0, 1.0)$ is bitwise the unweighted model rather than
        merely close to it.

        Args:
            keep_index: The surviving target channels' **declared** indices, which is what the
                block split is taken in -- the survivors are not contiguous.
            weight_st: Relative weight of the first stored block (scattering).
            weight_ph: Relative weight of the second (phase-harmonic).

        Returns:
            ``(C_keep,)`` float32, summing to $C_{\mathrm{keep}}$.

        Raises:
            ValueError: If either weight is negative, or if both are zero -- which would scale by
                infinity and leave an objective with no reconstruction term at all.
        """
        for name, value in (("target_weight_st", weight_st), ("target_weight_ph", weight_ph)):
            if not float(value) >= 0.0:  # catches NaN as well as negatives
                raise ValueError(f"{name} must be >= 0 and not NaN, got {value!r}")

        declared = torch.as_tensor(list(keep_index), dtype=torch.long)
        weights = torch.where(
            declared < cls.TARGET_BLOCK_SPLIT,
            torch.tensor(float(weight_st)),
            torch.tensor(float(weight_ph)),
        ).to(torch.float32)

        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError(
                "target_weight_st and target_weight_ph are both zero, which would leave the "
                "objective with no reconstruction term; at least one block must carry weight"
            )
        return weights * (float(weights.numel()) / total)

    @staticmethod
    def _resolve_block_warm_steps(
        block_warmup_steps: Sequence[int], sequence_length: int
    ) -> torch.Tensor:
        r"""Per step, whether at least half of one source block's channels are past their warm-up.

        $$\mathrm{warm}_s = \mathbb{1}\!\left[\,\left|\{c : W'_c \le s\}\right|
        \;\ge\; f\,\left|\{c\}\right|\,\right], \qquad f = 0.5.$$

        An **empty** block is warm at every step, vacuously: a constraint over no channels holds,
        and the alternative -- reporting a block that does not exist as permanently cold -- would
        put a zero in the CSV that reads as a measurement rather than as an absence.

        **Two configurations produce one**, and the second is now the common one. A source built
        without its first stored block leaves the second as the whole stream; and an alignment
        reference below every channel of a block drops that block entirely, which is what the
        raw-target cells do -- their reference sits far below the second block's fastest channel,
        so that block keeps no channels at all and its warmth fraction is $1.0$ over none of them.
        That is an absence, not a measurement, and it is why the fraction is read beside the
        block's kept width rather than alone.

        Args:
            block_warmup_steps: $W'_c$ for the block's channels.
            sequence_length: $T$, the window the pattern is built for.

        Returns:
            A ``(T,)`` boolean tensor.
        """
        steps = torch.arange(int(sequence_length))
        if not block_warmup_steps:
            return torch.ones(int(sequence_length), dtype=torch.bool)
        waits = torch.tensor([int(step) for step in block_warmup_steps])
        past = (steps[:, None] >= waits[None, :]).sum(dim=-1)
        return past.to(torch.float64) >= WARM_BLOCK_FRACTION * float(waits.numel())

    @torch.no_grad()
    def _gap_by_kept_channel(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        likelihood: str,
    ) -> torch.Tensor:
        r"""The forecast gap $D_0 - D_1$ resolved per surviving target channel, in nats per anchor.

        The per-element term is the objective's own :func:`raw_sample_score` and the denominator is
        the objective's own contributing-anchor count, so the vector this returns sums to
        ``pred_gap`` over the same mask -- which is the property both channel-axis splits reported
        beside it rest on.

        The two branches are reduced one at a time rather than differenced elementwise, for the
        reason the parent's version gives: one branch's score is a
        $(B, A, H, C_{\mathrm{keep}})$ tensor, and holding two of them plus their difference would
        triple that for a vector of $C_{\mathrm{keep}}$ numbers.

        Args:
            forward_outputs: The dict returned by ``forward``.
            target: The gathered forecast target $(B, A, H, C_{\mathrm{keep}})$.
            mask: The forecast mask $(B, A, H)$ the objective scored under.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.

        Returns:
            The per-channel gap $(C_{\mathrm{keep}},)$.
        """
        n_anchors = contributing_anchors(mask).to(target.dtype).sum().clamp_min(1.0)

        def _reduced(branch: str) -> torch.Tensor:
            score = raw_sample_score(
                forward_outputs[f"mu_{branch}"],
                target,
                likelihood=likelihood,
                logvar=forward_outputs[f"logvar_{branch}"],
                # The objective's own weight, so this vector still sums to the ``pred_gap``
                # printed beside it. Left out, every channel split would be a decomposition of a
                # quantity the objective does not optimise.
                channel_weight=self.target_channel_weight,
            ) * mask[..., None]
            return score.sum(dim=(0, 1, 2))

        return (_reduced("base") - _reduced("full")) / n_anchors

    @torch.no_grad()
    def _anchors_per_sample(
        self, forward_outputs: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        r"""How many anchors this step actually decoded, per batch element.

        A geometry guard, not a result: at the shipped $F = 134$, $S = H = 30$ and
        $T_{\mathrm{valid}} = 270$ it is $5$ for $\varphi \le 15$ and $4$ otherwise, mean
        $136/30$; at the validation stride of $1$ it is exactly $136$. A value off that band means
        the tiling is not the one the configuration states.

        Counted off ``anchor_valid`` rather than off the mask, deliberately: this must report the
        **decoded** set, so that a batch whose ``weight`` is entirely zero still says which anchors
        the forward built rather than reporting the geometry as having collapsed.

        Args:
            forward_outputs: The dict returned by ``forward``.
            target: The forecast target, read for its dtype and device only.

        Returns:
            A scalar tensor.
        """
        anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
        if anchors is None:
            # A model that decodes every anchor; the count is the dense range's own length.
            return torch.full(
                (), float(self.geometry.t_valid), device=target.device, dtype=target.dtype
            )
        valid: Optional[torch.Tensor] = forward_outputs.get("anchor_valid")
        if valid is None:
            return torch.full(
                (), float(anchors.shape[1]), device=target.device, dtype=target.dtype
            )
        return valid.to(target.dtype).sum() / float(max(1, anchors.shape[0]))

    @torch.no_grad()
    def _source_lag_warmth(
        self, forward_outputs: Dict[str, torch.Tensor], target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        r"""The share of attention mass landing on lags where a source block is warm, per block.

        $$\texttt{source\_lag\_warmth\_frac} \;=\;
        \frac{\sum_{b,a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}\;
              \mathrm{warm}\!\left(t_{b,a} - \ell\right)}
             {\sum_{b,a,m,\ell} v_{b,a}\,\alpha_{b,\,t_{b,a},\,m,\,\ell}}$$

        read at the anchors the forward decoded, with $v$ their validity. Normalising by the mass
        actually present rather than by a row count is what keeps the value in $[0, 1]$ when rows
        have no admissible lag at all: the attention normalises such a row to zero, and zero over
        zero would otherwise be the answer.

        This is the readout that sizes the compromise the design makes on the source. Lag attention
        searches $L$ lags back from an anchor, into a region where much of the source is still
        inside its own warm-up, and the design keeps every source channel rather than gating them
        -- the alternative costs almost the whole second source block, against a lag search that
        exists to find the $20$ to $120$ s contraction-to-deceleration delay. So the residual is
        measured instead of resolved, and a **small** value here is the expected finding rather
        than a failure.

        ``attn_weights`` is the forward's returned tensor, which is post-dropout: the attention
        applies dropout before the flip that produces it. At the shipped ``attn_dropout: 0.0`` --
        the same condition that makes the lag-map sum identity hold -- that is the pre-dropout
        tensor as well.

        Args:
            forward_outputs: The dict returned by ``forward``, carrying ``attn_weights``.
            target: The forecast target, read for its dtype only.

        Returns:
            ``{'source_lag_warmth_frac_st', 'source_lag_warmth_frac_ph'}``.
        """
        alpha = forward_outputs["attn_weights"]  # (B, T, num_heads, L)
        batch, _steps, heads, lags = alpha.shape
        device, dtype = alpha.device, target.dtype

        anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
        if anchors is None:
            dense = torch.arange(
                self.geometry.warmup, self.geometry.t_valid, device=device, dtype=torch.long
            )
            anchors = dense[None, :].expand(batch, -1)
            live = torch.ones(anchors.shape, device=device, dtype=dtype)
        else:
            anchors = anchors.to(torch.long)
            valid: Optional[torch.Tensor] = forward_outputs.get("anchor_valid")
            live = (
                torch.ones(anchors.shape, device=device, dtype=dtype)
                if valid is None
                else valid.to(dtype)
            )

        index = anchors[:, :, None, None].expand(-1, -1, heads, lags)
        # (B, A, num_heads, L): the attention rows of exactly the anchors that were decoded.
        at_anchor = alpha.gather(1, index).to(dtype) * live[:, :, None, None]
        total = at_anchor.sum().clamp_min(torch.finfo(dtype).tiny)

        # (B, A, L): lag l at anchor t reads source step t - l. Negative entries have exactly zero
        # attention -- the lag mask forbids them -- so they are clamped for the lookup and then
        # excluded, rather than relied upon to be harmless.
        lag_steps = anchors[:, :, None] - torch.arange(lags, device=device)[None, None, :]
        readable = lag_steps >= 0
        safe = lag_steps.clamp(min=0)

        warmth: Dict[str, torch.Tensor] = {}
        for name, pattern in (
            ("st", self.source_block_warm_st),
            ("ph", self.source_block_warm_ph),
        ):
            warm = readable & pattern.to(device)[safe]  # (B, A, L)
            warmth[f"source_lag_warmth_frac_{name}"] = (
                at_anchor * warm[:, :, None, :].to(dtype)
            ).sum() / total
        return warmth

    @torch.no_grad()
    def _resolved_forecast_gaps(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        weight: torch.Tensor,
        *,
        likelihood: str,
    ) -> Dict[str, torch.Tensor]:
        r"""The four inherited gaps, two three-way channel splits, and the family's four readouts.

        **Why the four inherited ones are recomputed here at all.** The parent's version builds its
        mask with no anchor set, which is right for a model that decodes every anchor and wrong for
        one that decodes a tile: the four splits would then be averaged over $T_{\mathrm{valid}}$
        anchors while the ``pred_gap`` printed beside them is averaged over the roughly ten the
        objective saw, and reading one against the other would be reading two different
        denominators. Only the mask changes -- the reduction is the parent's own
        :meth:`~teb_vae.lag_attn_fs.nets.feature_target.FeatureForecastTarget._forecast_gaps_from_mask`,
        so each of those four stays a partial sum of the gap it is read beside.

        **Why the other ten are emitted from here.** This method is the family's per-package
        metric hook: the parent's ``compute_loss`` merges whatever it returns, which is how the
        two-sided sibling ships its four added columns, and it is the only hook available because
        the anchor seam lives in the shared objective rather than in a subclass.

        The ten are two geometry guards (``target_warm_frac``, ``anchors_per_sample``), the
        two-block source warmth split, and **two** three-way channel splits. Both recompose to
        ``pred_gap`` over the same denominator exactly as the block split does, and both cut the
        channel axis in a way the block split cannot, since both stored blocks span nearly the same
        rebased range -- the first by filter speed, the second by how much of each coefficient the
        anchor has not already seen. The two are not restatements of one another: warm-up says when
        a channel became a function of the recording, novelty says how much of what it reports at a
        *scored* step is still ahead of the anchor, and the slowest kept channel is warm across the
        whole window while carrying $\nu = 0.026$.

        Args:
            forward_outputs: The dict returned by ``forward``, carrying ``anchor_index``,
                ``anchor_valid`` and ``attn_weights``.
            target: The gathered forecast target $(B, A, H, C_{\mathrm{keep}})$.
            weight: Decimated validity signal $(B, T)$.
            likelihood: ``'mse'`` or ``'gaussian_nll'``, matching the objective's.

        Returns:
            The parent's four gap splits, plus ``pred_gap_warm_lo`` / ``_mid`` / ``_hi``,
            ``pred_gap_novel_lo`` / ``_mid`` / ``_hi``, ``target_warm_frac``,
            ``anchors_per_sample``, ``source_lag_warmth_frac_st`` and ``source_lag_warmth_frac_ph``.
        """
        anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
        anchor_valid: Optional[torch.Tensor] = forward_outputs.get("anchor_valid")
        mask, _coverage = forecast_mask(
            weight,
            self.geometry,
            coverage_floor=self.coverage_floor,
            anchors=anchors,
            anchor_valid=anchor_valid,
        )
        metrics = self._forecast_gaps_from_mask(
            forward_outputs, target, mask, likelihood=likelihood
        )

        gap_by_channel = self._gap_by_kept_channel(
            forward_outputs, target, mask, likelihood=likelihood
        )
        # Two partitions of one per-channel gap vector, in one loop over the two buffers: the
        # column families differ only in which stored vector ranked the channels, and a second
        # loop written out would be the place the two reductions could come to differ.
        for prefix, assignment in (
            ("warm", self.warm_tertile_id),
            ("novel", self.novelty_tertile_id),
        ):
            tertile = assignment.to(gap_by_channel.device)
            for group, name in enumerate(("lo", "mid", "hi")):
                metrics[f"pred_gap_{prefix}_{name}"] = (
                    gap_by_channel * (tertile == group).to(gap_by_channel.dtype)
                ).sum()

        # Resolved at construction and echoed, not measured: see _resolve_target_warm_frac for why
        # recomputing it per batch would be a tautology, and what the column is actually for.
        metrics["target_warm_frac"] = torch.full(
            (), float(self.target_warm_frac), device=target.device, dtype=target.dtype
        )
        metrics["anchors_per_sample"] = self._anchors_per_sample(forward_outputs, target)
        metrics.update(self._source_lag_warmth(forward_outputs, target))
        return metrics
