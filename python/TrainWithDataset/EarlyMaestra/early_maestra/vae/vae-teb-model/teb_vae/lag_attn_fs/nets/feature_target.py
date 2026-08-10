r"""The feature forecast target: what a model forecasting stored coefficients decides differently.

Two things separate a feature-domain forecaster from the raw-signal one it is otherwise identical
to, and they are the same thing seen twice.

**The decoder's width.** Each horizon token emits one value per surviving *target channel* instead
of $R = 16$ raw samples. The width is not a configuration key: it is $C_{\mathrm{keep}}$, the reach
budget's surviving-channel count, and taking it from anywhere else would let a run decode a width
its target does not have. ``raw_per_step`` stays a geometry input -- the trimmed-grid geometry
validates its raw index identities against it -- it simply stops being the decoder width.

**The target.** The objective is handed a block gathered from the caller's feature stream rather
than from a raw grid. The gather takes the target gate's keep-index and **not** its delay: the
delay exists to push each *input* channel's forward reach behind the anchor's causal endpoint,
and applying it to the target would silently ask anchor $t$ to forecast the future of anchor
$t - \delta_c$, per channel, with every shape downstream unchanged.

Neither decision mentions an encoder, which is why they live here rather than in a model class.
:class:`FeatureForecastTarget` is a plain object carrying exactly those five members, and a model
becomes a feature-domain forecaster by listing it before its base:

.. code-block:: python

    class SeqVaeLagAttnFs(FeatureForecastTarget, SeqVaeLagAttnRws): ...

It defines no ``__init__`` and constructs nothing. Every attribute it reads -- ``target_gate``,
``c_y``, ``geometry``, ``horizon``, ``coverage_floor``, ``logvar_clamp`` and
``decoder_out_channels`` -- is set by the base constructor before the decoder is built, which is
what makes the width hook a legal override rather than a read of a half-built model.

What the objective computes is not restated here. ``lag_attn_rws/nets/losses.py`` is already
domain-neutral -- it reduces a $(B, T_{\mathrm{valid}}, H, X)$ block against a
$(B, T_{\mathrm{valid}}, H)$ mask and takes $X$ as an argument -- so this supplies the target and
the block width and delegates. A second copy of the objective would make every comparison between
the models partly a comparison of several losses.

The unit consequence is worth stating where the width is decided: the reconstruction is summed
over $H \cdot C_{\mathrm{keep}} = 2340$ coefficients at the shipped budget against the raw
models' $H \cdot R = 480$ samples, so the nats are not comparable to theirs, and -- since
$C_{\mathrm{keep}}$ moves with the budget -- not comparable across budgets within one model
either.

That same summation is why a feature-domain model reports four readouts the raw ones do not. A
scalar summed over $2340$ coefficients cannot separate a model that forecasts three easy channels
well from one that is uniformly mediocre, and it cannot separate forecasting from
*reconstruction*: a stored coefficient is an average over a window centred on its own step, so a
share of the short-horizon target is already determined by signal the model has legitimately
observed. Resolving the forecast gap by horizon step and by stored block is what makes those
distinguishable while there is no evaluation pipeline to ask the question properly.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from teb_vae.lag_attn_rws.nets.losses import compute_loss as compute_shared_objective
from teb_vae.lag_attn_rws.nets.losses import raw_sample_score
from teb_vae.lag_attn_rws.nets.raw_masks import contributing_anchors, forecast_mask


class FeatureForecastTarget:
    r"""The target-domain half of a feature forecaster, mixed in ahead of an encoder model.

    Not a base class and not an interface: a plain object with no ``__init__``, no abstract member
    and no state of its own, listed first in a model's bases so its five members win method
    resolution over the raw-target ones. Every model that mixes it in keeps its own constructor
    signature untouched, which is what the ``inspect.signature`` sweep in
    ``trainer._build_model_kwargs`` requires -- a narrowed ``__init__`` would forward no
    configuration at all and silently build an all-defaults model.
    """

    #: How many of the declared $c_y$ target channels belong to the **first** of the two stored
    #: blocks the caller concatenates, counted in the declared channel order the reach budget's
    #: keep-index is positional into.
    #:
    #: A class attribute rather than a constructor keyword, deliberately, and both halves of that
    #: matter. It is not a keyword because the constructor signature is swept with
    #: ``inspect.signature`` to build a run's kwargs, and a key that changed only which of two
    #: diagnostics a coefficient was counted in would look like an architecture decision in every
    #: checkpoint. It is not derived because it cannot be: $c_y$ is the *sum* of the two blocks'
    #: widths and nothing here can recover the split from it.
    #:
    #: It exists only to split :meth:`_resolved_forecast_gaps` -- no loss term, no shape, no
    #: parameter depends on it -- so a wrong value mislabels two reported numbers and breaks
    #: nothing. That is precisely why the task checks it against the data it assembles the target
    #: from rather than leaving it declared and unverified.
    TARGET_BLOCK_SPLIT: int = 43

    def _default_decoder_out_channels(self) -> int:
        r"""One output per surviving target channel: $C_{\mathrm{keep}}$, or $c_y$ ungated.

        Overriding the base's default rather than passing ``decoder_out_channels`` into it, for
        two reasons the base's own docstring records. The gate this reads is built by the base
        constructor, so nothing outside it can compute this value beforehand; and a subclass
        narrowing ``__init__`` to intercept the keyword would break the ``inspect.signature``
        sweep the trainer builds its kwargs with. It also keeps the width recoverable from a
        checkpoint: ``target_keep_index`` is stamped in ``model_kwargs``, and the width follows
        from it rather than needing a second field that could disagree.

        Returns:
            The decoder's output channel count.
        """
        return self.c_y if self.target_gate is None else self.target_gate.out_channels

    def _build_forecast_target(self, target_features: torch.Tensor) -> torch.Tensor:
        r"""Gather the surviving channels, then unfold each anchor's future window.

        $$Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]],$$

        for anchors $t \in [0, T - H)$ and horizon steps $\tau \in [0, H)$.

        The gather runs **before** the unfold. The two commute, and doing it first is what keeps
        the copy at $(B, T, C_{\mathrm{keep}})$ instead of $(B, T_{\mathrm{valid}}, H,
        C_{\mathrm{keep}})$ -- a factor of $H$ at production batch sizes, where the latter is a
        third of a gigabyte.

        The gate's **keep-index only**. Its delay reads channel $c$ at step $t - \delta_c$, which
        is the guard that keeps each input channel's forward reach behind the anchor's causal
        endpoint; the target is what the anchor is asked to predict, and delaying it would move
        the question rather than protect the answer.

        Args:
            target_features: The caller's target stream $(B, T, c_y)$, on the decimated grid.

        Returns:
            The forecast target $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$.

        Raises:
            ValueError: If the stream is not 3-D, if its length is not $T$ -- which is what a
                loader running at a different ``trim_minutes`` produces -- or if its width is not
                the declared $c_y$ the keep-index is positional into.
        """
        if target_features.dim() != 3:
            raise ValueError(
                f"target stream must be 3-D (B, T, c_y), got shape {tuple(target_features.shape)}"
            )
        if target_features.size(1) != self.geometry.t:
            raise ValueError(
                f"target stream length {target_features.size(1)} != geometry.t "
                f"{self.geometry.t}; this geometry assumes the trimmed loader "
                f"(trim_minutes: 1.0 -> T = {self.geometry.t} decimated steps), so a mismatch "
                f"means the loader ran at a different trim_minutes"
            )
        if target_features.size(2) != self.c_y:
            raise ValueError(
                f"target stream has {target_features.size(2)} channels but the model declares "
                f"c_y={self.c_y}; the surviving-channel index is positional into the declared "
                f"width, so a mismatch would gather the wrong channels rather than fail"
            )

        gathered = (
            target_features
            if self.target_gate is None
            else torch.index_select(target_features, -1, self.target_gate.keep_index)
        )
        # unfold appends the window as a new trailing axis, so the permute is what makes the
        # block horizon-major and the last axis the channel axis the decoder emits.
        return (
            gathered[:, 1:, :]
            .unfold(dimension=1, size=self.horizon, step=1)
            .permute(0, 1, 3, 2)
        )

    @torch.no_grad()
    def _resolved_forecast_gaps(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
        weight: torch.Tensor,
        *,
        likelihood: str,
    ) -> Dict[str, torch.Tensor]:
        r"""The forecast gap $D_0 - D_1$, resolved by horizon step and by stored block.

        Four scalars, each in the same nats-per-anchor units as ``pred_gap`` itself and each a
        partial sum of it: the summed gap restricted to horizon step $\tau = 0$, to
        $\tau = H - 1$, to the surviving channels of the first stored block, and to those of the
        second. Both splits recompose -- $\sum_\tau$ of the horizon curve and the two block terms
        each add back to ``pred_gap`` -- which is what makes them readable as a decomposition
        rather than as four unrelated numbers.

        **Why the objective does not report them and this does.** With no evaluation
        pipeline every readout is a scalar summed over $H \cdot C_{\mathrm{keep}}$ coefficients,
        and one scalar cannot answer the question this target domain raises. A stored coefficient
        at step $s$ is a weighted average of signal over a window *centred* at $s$, so the fraction
        of the target at horizon step $\tau$ that is already fixed by observed history is
        $\max(0, (\rho_c - 4\tau) / 2\rho_c)$: exactly $0.5$ at $\tau = 0$ for every channel, and
        $0.000$ by $\tau = 29$ on the surviving set. A gap that is real forecasting therefore
        survives to the far step; one that is reconstruction of the already-determined part does
        not, and the two are indistinguishable in the summed number. The block split asks the same
        question along the other axis, because the two stored blocks' filters have different
        reaches and therefore different blends.

        **The per-element term is the objective's own**, reached through
        :func:`~teb_vae.lag_attn_rws.nets.losses.raw_sample_score` rather than restated, and the
        mask is rebuilt through the same two functions the objective builds it from. A second
        definition of either would let these four stop being partial sums of the number they are
        read beside -- which is the only property that makes them worth reporting.

        The two branches are reduced one at a time rather than differenced elementwise: the score
        of one branch is a $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$ tensor, a third of a
        gigabyte at the production batch, and holding two of them plus their difference would
        triple that for four scalars.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            target: The gathered forecast target $(B, T_{\mathrm{valid}}, H, C_{\mathrm{keep}})$.
            weight: Decimated validity signal $(B, T)$.
            likelihood: ``'mse'`` or ``'gaussian_nll'``, matching the objective's.

        Returns:
            ``{'pred_gap_tau_first', 'pred_gap_tau_last', 'pred_gap_st', 'pred_gap_ph'}``.
        """
        mask, _coverage = forecast_mask(
            weight, self.geometry, coverage_floor=self.coverage_floor
        )
        # The objective's own denominator: anchors that contribute nothing leave the numerator and
        # the denominator together, so these stay per-anchor rather than scaling with mask density.
        n_anchors = contributing_anchors(mask).to(target.dtype).sum().clamp_min(1.0)

        def _reduced(branch: str) -> Tuple[torch.Tensor, torch.Tensor]:
            """Sum one branch's masked score over the anchor axes, keeping $\\tau$ then $c$."""
            score = raw_sample_score(
                forward_outputs[f"mu_{branch}"],
                target,
                likelihood=likelihood,
                logvar=forward_outputs[f"logvar_{branch}"],
            ) * mask[..., None]
            return score.sum(dim=(0, 1, 3)), score.sum(dim=(0, 1, 2))

        base_by_tau, base_by_channel = _reduced("base")
        full_by_tau, full_by_channel = _reduced("full")
        gap_by_tau = (base_by_tau - full_by_tau) / n_anchors
        gap_by_channel = (base_by_channel - full_by_channel) / n_anchors

        # Which of the surviving channels came from the first stored block. Built from the gate's
        # keep-index, which is positional into the declared width, so the split follows the reach
        # budget instead of assuming the survivors are contiguous -- they are not.
        keep_index = (
            torch.arange(self.c_y, device=target.device)
            if self.target_gate is None
            else self.target_gate.keep_index
        )
        first_block = (keep_index < self.TARGET_BLOCK_SPLIT).to(target.dtype)

        return {
            "pred_gap_tau_first": gap_by_tau[0],
            "pred_gap_tau_last": gap_by_tau[-1],
            "pred_gap_st": (gap_by_channel * first_block).sum(),
            "pred_gap_ph": (gap_by_channel * (1.0 - first_block)).sum(),
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, torch.Tensor],
        target_features: torch.Tensor,
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
        r"""Compute the seven-term objective, per anchor.

        $$\mathcal{L} = \lambda_{\mathrm{full}} D_1 + \lambda_{\mathrm{base}} D_0
        + \beta\,\mathrm{KL}_{\mathrm{train}} + \beta_p\,R_p
        + \lambda_{\mathrm{ms}} \mathcal{L}_{\mathrm{ms}}
        + \lambda_{\Delta} \mathcal{L}_{\Delta}
        + \lambda_{\mathrm{boundary}} \mathcal{L}_{\mathrm{boundary}}$$

        The three shape weights are forwarded like every other objective weight, and a
        feature-target run ships them at $0.0$: the terms read the block's last axis as a
        trajectory -- pooled neighbourhoods, first differences, a boundary sample identified with
        the anchor's last observed one -- and here that axis counts *channels*, which have no
        order and no continuity with anything. The keywords stay so the task plumbing and the
        config-parity comparison are uniform across the family, and the zeros-when-off contract
        keeps this model's columns honest zeros rather than raw-domain formulas evaluated over a
        channel axis.

        Delegates to :func:`~teb_vae.lag_attn_rws.nets.losses.compute_loss`, supplying the
        gathered feature block, this model's geometry, its block width and its two scalar bounds.
        Every term, every reduction and every reported metric is the raw model's; what this owns
        is the one thing a target domain decides -- how its target is gathered, and what its
        block's last axis counts.

        ``block_width`` is $C_{\mathrm{keep}}$, not ``geometry.r``. It feeds only the four
        per-element log-variance diagnostics, so passing the raw grid's $R$ here would change no
        loss, fail no shape check, and rescale exactly those four reported numbers by $4.9\times$
        at the shipped budget -- which is where ``logvar_clamp`` is re-derived from.

        Args:
            forward_outputs: The dict returned by :meth:`forward`.
            target_features: The target stream $(B, T, c_y)$, loader-normalized, in the declared
                channel order the keep-index is positional into.
            weight: Decimated validity signal $(B, T)$.
            beta: Weight on the trained KL term.
            beta_prior: Weight on the prior scale rate; ``0.0`` leaves the historical three-term
                objective while ``prior_rate`` is still reported.
            lambda_full: Weight on the full-forecast reconstruction.
            lambda_base: Weight on the base-forecast reconstruction.
            likelihood: ``'mse'`` or ``'gaussian_nll'``.
            free_bits: Per-dimension per-step KL floor; enters the trained KL only.
            lambda_ms: Weight on the multiscale $L_1$ shape term. Shipped ``0.0`` here; see
                above.
            lambda_deriv: Weight on the derivative Huber shape term. Shipped ``0.0`` here.
            lambda_boundary: Weight on the boundary-continuity shape term. Shipped ``0.0`` here.

        Returns:
            ``{'metrics': ..., 'likelihood': ...}``. ``metrics`` is the raw model's key set plus
            the four resolved forecast gaps of :meth:`_resolved_forecast_gaps`, which are partial
            sums of the ``pred_gap`` beside them rather than new quantities.

        Raises:
            ValueError: On an unknown ``likelihood``, a target stream that does not match the
                geometry or the declared width, or a ``weight`` that does not match the trimmed
                grid.
        """
        target = self._build_forecast_target(target_features)
        result = compute_shared_objective(
            forward_outputs,
            target,
            weight=weight,
            geometry=self.geometry,
            # The feature block's last axis counts surviving target channels, which is exactly
            # what the decoder emits -- so the two cannot disagree.
            block_width=self.decoder_out_channels,
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
        # Added here rather than inside the objective: the raw-signal models' block has one
        # physical channel and thirty horizon steps of one signal, so neither split says anything
        # there -- and the objective's metric dict is pinned bitwise for all of them.
        result["metrics"].update(
            self._resolved_forecast_gaps(
                forward_outputs, target, weight, likelihood=likelihood
            )
        )
        return result
