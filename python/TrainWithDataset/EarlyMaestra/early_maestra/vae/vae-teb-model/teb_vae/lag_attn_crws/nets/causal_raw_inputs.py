r"""One-sided inputs over a raw target: the two members that change, and the anchored gather.

:class:`CausalRawInputs` extends
:class:`~teb_vae.lag_attn_cfs.nets.causal_inputs.CausalWarmupInputs` -- the input warm-up mask, the
lag validity floor, the tiled anchor set and the forward that decodes at it -- and overrides exactly
what a **raw** target changes about them. Five of that mixin's seven members are already
target-domain-free and are inherited untouched:

* ``_set_causal_inputs`` and ``_validate_causal_geometry`` -- pure writers over the four keywords
  this domain owns, and the stride-versus-span refusal;
* ``_build_adapter`` -- one stream's availability adapter at its warm-up rather than at its gate's
  delays, which are all zero under a pure gather;
* ``build_lag_mask`` -- the floored $\mathbb 1[t - \ell \ge F_u]$ mask;
* ``_build_anchor_index`` and ``forward`` -- the tiled anchor set and the decode at it. Their
  independence from the target is the load-bearing fact of this package: these cells take the
  **same three input tensors** as the causal-feature cells and differ only in what the decoder emits
  and what the objective scores, so not one line of the tiled forward is copied.

Three things live here, and each exists for a reason a raw target creates:

**The floor is a policy rather than a validity requirement.** In the causal-feature cells
$F \ge B - 1$ is enforced because below it the objective would score assumed pre-recording history
of a *stored coefficient* as though it were signal. A raw sample carries no such region, so the
inequality constrains nothing about the target. :meth:`CausalRawInputs._check_anchor_floor` keeps it
anyway, restated as the declared **input-warmth** policy -- every kept *input* channel is warm by
the first forecast step -- so the refusal says what it now enforces instead of inheriting a sentence
about a target that no longer exists.

**Only the source patterns are resolved.** The causal-feature readouts partition kept *target*
channels three ways; there are none here, so the target warm fraction and the warm tertiles are not
resolved at all, and :meth:`CausalRawInputs._resolve_warmup_readout_constants` registers the two
source-block patterns and nothing else.

**The raw future window is gathered at the decoded anchors.**
:func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_target` takes no anchor set -- the raw
target never needed one, because no raw-target cell tiled -- so :func:`gather_anchored_future_target`
is the one genuinely new piece of arithmetic in this cell, and its correctness rests on agreeing
with that dense builder at the dense anchor set.

Everything else the source stream needs is **bound by reference** from
:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` in the class
body below rather than copied. A bound member is read once at class creation and the owning class
never learns it has a second consumer, which is what keeps "no edit to any existing package" true
while leaving no second definition to drift.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.nets.causal_inputs import CausalWarmupInputs
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry
from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_raw_objective
from teb_vae.lag_attn_rws.nets.raw_targets import build_future_target


def gather_anchored_future_target(
    fhr_raw: torch.Tensor,
    geometry: TrimmedRawGeometry,
    anchors: torch.Tensor,
    *,
    future_index: torch.Tensor,
) -> torch.Tensor:
    r"""Gather each decoded anchor's raw future window: $(B, A, H, R)$.

    The index grid is the one
    :func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_index` already builds and the model
    already caches, read at the anchors rather than rebuilt:

    $$\mathrm{idx}[b, a, \tau, r] \;=\; \texttt{future\_index}\big[\,\mathcal A[b, a],\, \tau,\, r\,\big],
    \qquad X^{+}[b, a, \tau, r] \;=\; x\big[b,\, \mathrm{idx}[b, a, \tau, r]\,\big].$$

    **A ``gather``, not an ``index_select``, and that is the whole of the difference from the dense
    builder.** :func:`~teb_vae.lag_attn_rws.nets.raw_targets.build_future_target` may use
    ``index_select`` because its index is the *shared* $(T_{\mathrm{valid}}, H, R)$ grid -- the same
    rows for every sample of the batch. Here the anchor set is **per sample**, because the tile phase
    is derived per segment, so the index is $(B, A, H, R)$ and an ``index_select`` on dimension $1$
    would return $(B, B \cdot A \cdot H \cdot R)$ and fail the reshape.

    **The bounds check is not redundant with the objective's.** Advanced indexing on a negative
    index *wraps* rather than raising, so an anchor of $-1$ would silently gather the last legal
    window and return every shape correct; and an anchor at or above $T_{\mathrm{valid}}$ would
    raise from three frames down, naming a dimension rather than an anchor. What is deliberately
    **not** checked here is uniqueness among the valid entries: that refusal belongs to the mask,
    where the two per-anchor denominators it protects are built, and
    :func:`~teb_vae.lag_attn_rws.nets.raw_masks.forecast_mask` raises it on the same call.

    Args:
        fhr_raw: The raw target signal $(B, L_{\mathrm{raw}})$, loader-normalized.
        geometry: The trimmed-grid geometry; must match the raw length of ``fhr_raw``.
        anchors: The decoded anchor index $(B, A)$, integer, in $[0, T_{\mathrm{valid}})$. Padded
            slots repeat their row's last real anchor, which this function honours -- the padding is
            removed by the forecast mask, not here.
        future_index: The cached $(T_{\mathrm{valid}}, H, R)$ index grid.

    Returns:
        The anchored future raw target $(B, A, H, R)$.

    Raises:
        ValueError: If ``fhr_raw`` is not 2-D, if its length does not match ``geometry.raw_len`` --
            which is what a loader running at the wrong ``trim_minutes`` produces -- if ``anchors``
            is not 2-D, or if an anchor is outside $[0, T_{\mathrm{valid}})$, naming the value.
    """
    if fhr_raw.dim() != 2:
        raise ValueError(f"fhr_raw must be 2-D (B, L_raw), got shape {tuple(fhr_raw.shape)}")
    if fhr_raw.size(1) != geometry.raw_len:
        raise ValueError(
            f"fhr_raw length {fhr_raw.size(1)} != geometry.raw_len {geometry.raw_len}; "
            "this geometry assumes the loader's symmetric trim has already been applied "
            "(trim_minutes: 1.0 -> 4800 raw samples), so a mismatch means the loader ran "
            "at a different trim_minutes than the geometry was built for"
        )
    if anchors.dim() != 2:
        raise ValueError(f"anchors must be 2-D (B, A), got shape {tuple(anchors.shape)}")

    index = anchors.to(device=fhr_raw.device, dtype=torch.long)
    outside = (index < 0) | (index >= geometry.t_valid)
    if bool(outside.any()):
        offending = int(index[outside][0])
        raise ValueError(
            f"anchor {offending} is outside [0, T_valid) = [0, {geometry.t_valid}); the tail "
            f"{geometry.horizon} anchors have no fully observed forecast window, and a negative "
            f"index would wrap to a legal window rather than raising"
        )

    batch, count = int(index.shape[0]), int(index.shape[1])
    # (B, A, H, R): the cached grid's rows for this sample's own anchors.
    windows = future_index.to(fhr_raw.device)[index]
    gathered = fhr_raw.gather(1, windows.reshape(batch, -1))
    return gathered.reshape(batch, count, geometry.horizon, geometry.r)


class CausalRawInputs(CausalWarmupInputs):
    r"""One-sided inputs and a tiled anchor set, scoring the raw future at those anchors.

    A plain-object mixin, placed **first** in a composing model's bases so the inherited
    ``_build_adapter``, ``build_lag_mask`` and ``forward`` win method resolution over the
    architecture's own. Unlike the causal-feature cells there is no second mixin beside it: a raw
    target needs no width hook -- the architecture's own ``_default_decoder_out_channels`` already
    returns ``raw_per_step``, which is exactly what these cells want -- and its **absence** from this
    class is as load-bearing as anything present. Composing
    :class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` in by
    mistake builds the decoder at $C_{\mathrm{keep}}$ against a $(B, A, H, R)$ target, and the first
    symptom is ``raw_sample_score`` broadcasting two shapes that do not, three frames below the
    decision that caused it.

    Five members are **bound by reference** from that mixin rather than copied, because each is
    about the source stream or the anchor set and neither notion changes with the target. Binding
    edits no existing package and makes drift structurally impossible rather than merely
    test-detected: the object here *is* the object there.

    ``staticmethod(...)`` around ``_resolve_block_warm_steps`` is load-bearing and its absence fails
    three frames away. ``Owner.some_staticmethod`` returns the **plain function** -- the descriptor
    has already resolved -- and a plain function assigned in a class body becomes an *instance*
    method, so ``self`` would arrive as its first positional argument. The other two bound callables
    take ``self`` and bind as they are.

    Attributes:
        source_block_warm_st: ``(T,)`` bool, the **first** stored source block's per-step warmth.
        source_block_warm_ph: ``(T,)`` bool, the same for the second.
    """

    source_block_warm_st: torch.Tensor
    source_block_warm_ph: torch.Tensor

    #: The source channel layout, and the target feature block's. Both transfer unchanged, because
    #: these cells read the identical three input tensors: the target feature block is an **input**
    #: here rather than a target, and the diagnostic page's stream panels still split it.
    SOURCE_BLOCK_SPLIT = CausalFeatureForecastTarget.SOURCE_BLOCK_SPLIT
    TARGET_BLOCK_SPLIT = CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT

    #: Per-step warmth of one source block; static, and re-wrapped for the reason above.
    _resolve_block_warm_steps = staticmethod(
        CausalFeatureForecastTarget._resolve_block_warm_steps
    )

    #: The two kept readouts. ``_anchors_per_sample`` reads ``anchor_valid`` and the target's dtype
    #: only; ``_source_lag_warmth`` reads the attention weights, the anchors and the two source
    #: patterns. Neither mentions a target channel, which is why both survive the target change.
    _anchors_per_sample = CausalFeatureForecastTarget._anchors_per_sample
    _source_lag_warmth = CausalFeatureForecastTarget._source_lag_warmth

    # ------------------------------------------------------------------
    # The two target-coupled members
    # ------------------------------------------------------------------
    @staticmethod
    def _check_anchor_floor(warmup_period: int, kept_warmup_steps: Sequence[int]) -> None:
        r"""Refuse an anchor floor the kept **target-stream** channels' warm-up does not admit.

        $$F \;\ge\; B - 1, \qquad B = \max_{c \in \mathrm{kept}} W'_c.$$

        The inequality is the causal-feature cells', and what it enforces is not. There it is a
        *validity* requirement: the target is a stored coefficient, honest only from $W'_c$, so a
        floor one step too low scores assumed pre-recording history as signal with every shape
        correct. Here the target is a raw sample, honest at every step, so the objective is sound at
        any floor at all -- and the pairing is retained as the declared **input-warmth policy**,
        the statement that every kept **target-stream** input channel is warm by the first forecast
        step.

        **Which stream that is, is the whole of the wording, and the wider paraphrase is false.**
        The caller is ``_validate_causal_geometry``, which passes ``target_warmup_steps``: the
        survivors of the stream the *gate* selects. The **source** stream is never gated in this
        family -- its slowest channels carry the contraction envelope and dropping them would cost
        almost all of its second stored block -- so at the shipped budget it keeps channels waiting
        $162$, $194$, $233$ and $278$ steps, which are still cold hundreds of steps past this floor
        and are meant to be. That residual is *measured* rather than refused, by
        ``source_lag_warmth_frac_st`` and ``_ph``; a policy stated over every input channel would
        push $F$ to $277$ and cost about $144$ of the $152$ anchors to enforce something this design
        deliberately does not want.

        The remaining wording is exact and its obvious paraphrase is not. At the shipped $F = 133$
        against $B = 134$ the slowest kept target-stream channel is still cold *at the anchor
        itself*, and becomes honest exactly at $t + 1$, which is the first step the forecast covers.
        A policy stated as "warm at the anchor" would require $F \ge B$ and cost a tile boundary for
        nothing.

        Overridden rather than ``_validate_causal_geometry``, which would re-copy the
        stride-versus-span refusal and its message and let the two drift.

        Args:
            warmup_period: The anchor floor $F$ the model was built at.
            kept_warmup_steps: $W'_c$ per surviving **input** channel of the target feature stream.
                The source stream's vector is deliberately not passed; see above.

        Raises:
            ValueError: If the floor is below $B - 1$, naming both numbers, which stream they are
                computed over, and which reading of the policy it enforces.
        """
        if not kept_warmup_steps:
            return
        budget = max(int(step) for step in kept_warmup_steps)
        if int(warmup_period) < budget - 1:
            raise ValueError(
                f"warmup_period={int(warmup_period)} is below this cell's declared input-warmth "
                f"policy: every kept TARGET-STREAM input channel is warm by the first forecast "
                f"step. The slowest kept target-stream channel is honest only from step {budget}, "
                f"and a forecast at anchor t covers target steps from t + 1 onwards, so the floor "
                f"must be at least {budget - 1}. Two things this does not say. The policy is over "
                f"the gated target stream alone: the source stream is never gated and keeps "
                f"channels far colder than this floor by design, measured by "
                f"source_lag_warmth_frac_st / _ph rather than refused here, so do not raise the "
                f"floor to cover them. And the raw target is honest at every step, so a lower "
                f"floor would not corrupt the objective -- it would decode anchors whose inputs "
                f"are still partly pre-recording history, which is a different claim about the run "
                f"rather than a wrong number in it."
            )

    def _resolve_warmup_readout_constants(self) -> None:
        r"""Resolve the two constants the kept readouts are computed against.

        The causal-feature cells resolve four: a target warm fraction, a per-kept-channel tertile
        assignment, and the two source-block warmth patterns. The first two partition kept **target**
        channels, and this target has none -- its last axis counts raw samples, which have no
        warm-up, no filter and no order to rank by -- so they are dropped rather than re-pointed. A
        vacuous $1.0$ column and an empty tertile vector would both read as measurements.

        The two that remain are functions of the resolved budget and the geometry alone, so
        resolving them here rather than per batch is not an optimisation: a partition recomputed per
        batch is a partition that can differ between two batches of one run.

        Registered as **non-persistent** buffers, like every other budget-shaped tensor in the
        family: their contents follow the resolved budget, so a persistent copy would make a
        checkpoint trained at one budget fail to load at another and report it as misaligned keys
        rather than as a budget mismatch. Registering them at all -- rather than keeping plain
        tensors -- is what carries them when the module moves device.
        """
        # The split is taken in DECLARED channel coordinates, through the gate's keep-index rather
        # than positionally: the source is never gated in this family, so the two agree today, and a
        # split that assumed they always would is one that starts naming the wrong block the first
        # time they do not.
        declared = (
            torch.arange(self.c_u)
            if self.source_gate is None
            else self.source_gate.keep_index.cpu()
        )
        split = self.SOURCE_BLOCK_SPLIT if self.use_up_st else 0
        waits = self.source_warmup_steps
        for name, in_block in (
            ("st", declared < split),
            ("ph", declared >= split),
        ):
            block = (
                []
                if waits is None
                else [int(step) for step, keep in zip(waits, in_block.tolist()) if keep]
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
    # The objective
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        fhr_raw: torch.Tensor,
        *,
        weight: torch.Tensor,
        beta: float = 1.0,
        beta_prior: float = 0.0,
        lambda_full: float = 1.0,
        lambda_base: float = 1.0,
        likelihood: str = "gaussian_nll",
        free_bits: float = 0.0,
        lambda_ms: float = 0.0,
        lambda_deriv: float = 0.0,
        lambda_boundary: float = 0.0,
    ) -> Dict[str, Any]:
        r"""Compute the seven-term objective over the raw future, at the decoded anchors.

        The architecture's own ``compute_loss`` with one line changed: the raw window is gathered at
        ``anchor_index`` rather than at every anchor of $[0, T_{\mathrm{valid}})$. Every term, every
        reduction, every mask and every reported metric is the shared objective's -- including
        ``block_width``, which stays ``geometry.r`` because a horizon token still emits $R$ raw
        samples whatever the anchor set is.

        The **mask** side needs nothing at all:
        :func:`~teb_vae.lag_attn_rws.nets.losses.compute_loss` reads ``anchor_index`` and
        ``anchor_valid`` off the forward dict itself and threads them into
        :func:`~teb_vae.lag_attn_rws.nets.raw_masks.forecast_mask` and
        :func:`~teb_vae.lag_attn_rws.nets.raw_masks.kl_mask`, so the reconstruction support and the
        KL support are the decoded set by construction rather than by agreement.

        A forward dict carrying no anchor set falls back to the dense builder. That is not a
        convenience: it is what makes a stripped anchor set a **shape refusal** -- the dense target
        carries $T_{\mathrm{valid}}$ anchors against a forecast carrying $A_{\max}$ -- rather than a
        quietly mis-scored batch.

        Three readouts are merged onto the objective's metric dict, and all three are source- or
        anchor-side. The five the causal-feature cells report over kept target channels are dropped:
        this block's last axis counts raw samples.

        Args:
            forward_outputs: The dict returned by ``forward``, carrying ``anchor_index``,
                ``anchor_valid`` and ``attn_weights``.
            fhr_raw: The raw target signal $(B, L_{\mathrm{raw}})$, loader-normalized.
            weight: Decimated validity signal $(B, T)$.
            beta: Weight on the trained KL term.
            beta_prior: Weight on the prior scale rate; ``0.0`` leaves the historical three-term
                objective while ``prior_rate`` is still reported.
            lambda_full: Weight on the full-forecast reconstruction.
            lambda_base: Weight on the base-forecast reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-dimension per-step KL floor; enters the trained KL only.
            lambda_ms: Weight on the multiscale $L_1$ shape term; ``0.0`` skips it and reports an
                exact zero.
            lambda_deriv: Weight on the derivative Huber shape term, same contract.
            lambda_boundary: Weight on the boundary-continuity shape term, same contract. Refused
                outright by the shared objective when an anchor set is present.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``. ``metrics`` is the shared objective's key set
            plus ``anchors_per_sample`` -- a geometry guard rather than a result -- and the two
            source-block lag-warmth fractions.

        Raises:
            ValueError: On an unknown ``likelihood``, a raw length or a ``weight`` that does not
                match the geometry, an anchor outside $[0, T_{\mathrm{valid}})$ or duplicated among
                one row's valid entries, or a non-zero ``lambda_boundary`` with an anchor set.
        """
        anchors: Optional[torch.Tensor] = forward_outputs.get("anchor_index")
        target = (
            build_future_target(fhr_raw, self.geometry, future_index=self.future_index)
            if anchors is None
            else gather_anchored_future_target(
                fhr_raw, self.geometry, anchors, future_index=self.future_index
            )
        )
        result = compute_raw_objective(
            forward_outputs,
            target,
            weight=weight,
            geometry=self.geometry,
            # The raw block's last axis counts raw samples per horizon token, at every anchor set.
            block_width=self.geometry.r,
            coverage_floor=self.coverage_floor,
            logvar_clamp=self.logvar_clamp,
            beta=beta,
            beta_prior=beta_prior,
            lambda_full=lambda_full,
            lambda_base=lambda_base,
            likelihood=likelihood,
            free_bits=free_bits,
            lambda_ms=lambda_ms,
            lambda_deriv=lambda_deriv,
            lambda_boundary=lambda_boundary,
        )
        # Merged here rather than inside the objective, whose metric dict is pinned bitwise for the
        # four shipped forecasters; both readouts are this input domain's rather than the target's.
        result["metrics"]["anchors_per_sample"] = self._anchors_per_sample(
            forward_outputs, target
        )
        result["metrics"].update(self._source_lag_warmth(forward_outputs, target))
        return result


__all__ = ["CausalRawInputs", "gather_anchored_future_target"]
