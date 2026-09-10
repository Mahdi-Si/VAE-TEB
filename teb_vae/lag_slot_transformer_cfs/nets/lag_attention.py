r"""The attention comparator's fusion head: the same update, formed by a distribution over lags.

The recommended architecture forms its latent update as an **explicit sum**: one local proposal per
lag, added up, scaled by a fixed convention and bounded afterwards. This module is the arm that
replaces that sum with a learned convex combination -- a query posed from the anchor's target state,
scored against every lag's source vector, normalised over the lag axis, and read out through one
projection.

**Everything downstream of this module is the recommended arm's, unchanged.** The bound is the same
limiter at the same two scales, the full distribution is the same prior-relative residual, the
divergence is the same residual form, the sampling is the same shared draw and the decoder is the
same one invoked twice. That is what makes a difference between the two arms attributable to the
fusion rather than to a package of five changes at once, and it is the specification's own condition
on an attention comparator: replace how the residual inputs are formed and nothing else.

**Three things this head deliberately does not do, each of which the sibling architecture does.**

It builds **no head-structured posterior**. The per-head summaries are read out through one output
projection into the same $d_z$-wide update the local arm produces; nothing partitions the latent by
attention head, and the latent width is therefore free of the head count exactly as it is on every
other arm.

It carries **no sparsifying normaliser and no seeded long-lag penalty**. Both are real settings on
the sibling and both change what the distribution over lags looks like, so switching either on
alongside the fusion would be a second declared change inside a comparison built to isolate one.
Plain softmax with a learned per-lag key bias is the minimum that makes the arm an attention arm.

It **does not export its attention distribution**, and that is the decision most worth stating. The
weights exist here and are real, unlike anything the recommended arm could offer under that name.
What they are not is an allocation of the divergence or a measurement of when the source mattered:
the comparison this arm exists for is a difference of predictive scores, and a distribution over
lags published beside it would be read as a lag readout by every reader and every downstream table.
Both arms are therefore interrogated the same way -- by suppressing a band through the selector and
rescoring -- so that two arms' band margins are produced by one intervention interface.

**The suppression semantics differ between the two fusions, and the difference is not cosmetic.**
Removing a lag from an explicit sum removes its term and leaves every other term where it was.
Removing a lag from a normalised distribution removes it from the denominator too, so the surviving
weights grow. That is what "the model cannot read this lag" means for an attention head and there is
no version of it that leaves the others untouched, but it does mean a band margin measured on this
arm and one measured on the local arm are not the same quantity and must not be differenced. The
evaluation records which fusion produced each margin for exactly that reason.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

#: Spread the per-lag key bias is drawn at, matching the lag embedding of the local fusion head so
#: that neither arm starts with a stronger opinion about lag identity than the other.
LAG_BIAS_STD = 0.02


class LagAttentionFusion(nn.Module):
    r"""Aggregate one anchor's lag window by attention, and read out the bounded update.

    The query is the anchor's conditioning state $h_t$ -- the same tensor the prior heads read and
    the same one the local arm's proposal head conditions on. Keys and values are projected from
    each lag's source vector, whatever representation produced it, and a learned per-lag key bias
    carries lag identity in the way the local arm's lag embedding does.

    $$\alpha_{t,\ell} = \operatorname{softmax}_\ell
        \frac{\langle q_t, k_{t,\ell}\rangle + \langle q_t, r_\ell\rangle}{\sqrt{d_{\mathrm{head}}}},
      \qquad
      \bar a_t,\ \bar b_t = \operatorname{out}\Bigl(W_o \sum_\ell \alpha_{t,\ell} v_{t,\ell}\Bigr).$$

    Attributes:
        d_model: Width $d_h$ of the conditioning state and of the attended output.
        d_z: Latent width $d_z$ each update is emitted at.
        n_lags: $L$, the number of lag slots the key bias covers.
        source_dim: Width of one lag's source vector, as its encoder reports it.
        num_heads: Attention heads.
        d_head: Per-head width, so ``num_heads * d_head == d_model``.
        mean_only: Whether the scale half of the update was built at all.
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_z: int,
        n_lags: int,
        source_dim: int,
        num_heads: int = 4,
        mean_only: bool = False,
    ) -> None:
        r"""Initialize the fusion head.

        Args:
            d_model: Width $d_h$ of the conditioning state.
            d_z: Latent width $d_z$.
            n_lags: $L$, the number of lag slots.
            source_dim: Flattened width of one lag's source vector, which the source encoder
                reports as its own ``source_dim`` so the two cannot disagree.
            num_heads: Attention heads. Must divide ``d_model``; it partitions the *attention*
                and nothing else, and in particular nothing in the latent.
            mean_only: Build no scale half of the update at all. The output projection emits $d_z$
                rather than $2 d_z$, so the arm is a different module tree and a different state
                dict, matching the local fusion's treatment of the same arm.

        Raises:
            ValueError: If any width is not positive, or if the head count does not divide the
                model width -- which would leave one head narrower than the rest with every shape
                still correct.
        """
        super().__init__()
        for name, value in (
            ("d_model", d_model),
            ("d_z", d_z),
            ("n_lags", n_lags),
            ("source_dim", source_dim),
            ("num_heads", num_heads),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(
                f"num_heads={num_heads} must divide d_model={d_model}; an uneven split would "
                f"leave one head a different width from the rest and would still run."
            )

        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.n_lags = int(n_lags)
        self.source_dim = int(source_dim)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        self.mean_only = bool(mean_only)
        self.scale = 1.0 / math.sqrt(float(self.d_head))

        # Pre-norm on both sides, which is what keeps the dot products in a range the softmax has
        # not already saturated on. The key-value side normalises the source vector at its own
        # width, so the two source representations reach identical downstream shapes without this
        # module knowing which of them produced the window.
        self.q_norm = nn.LayerNorm(self.d_model)
        self.kv_norm = nn.LayerNorm(self.source_dim)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.source_dim, self.d_model)
        self.v_proj = nn.Linear(self.source_dim, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)

        # Lag identity, as a per-head key bias scored against the query. The counterpart of the
        # local arm's lag embedding: without it a lag-blind score would make the head invariant to
        # permuting the source times, which is the property lag identity exists to break. An
        # ``nn.Parameter`` rather than an ``nn.Embedding`` because it is scored rather than looked
        # up -- and unlike a linear layer, the family's generic initialisation pass leaves it alone.
        self.lag_bias = nn.Parameter(torch.zeros(self.n_lags, self.num_heads, self.d_head))
        nn.init.normal_(self.lag_bias, mean=0.0, std=LAG_BIAS_STD)

        self.output_proj = nn.Linear(
            self.d_model, self.d_z if self.mean_only else 2 * self.d_z
        )
        self.zero_output()

    def zero_output(self) -> None:
        """Zero the final projection, so the update starts at exactly zero.

        Called from the constructor and **again** from a composing model's post-initialisation
        block, because the family's generic pass xavier-fills every ``nn.Linear`` after the modules
        are built. Named and shaped exactly as the local fusion head's, so one hook serves both and
        an arm cannot lose the zero start by being a different class. Idempotent.

        Initialisation only. Calling it on a trained model would discard the source pathway.
        """
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        target_state: torch.Tensor,
        source_window: torch.Tensor,
        *,
        lag_valid: Optional[torch.Tensor] = None,
        selector: Optional[torch.Tensor] = None,
        lag_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Attend over one anchor set's lag window and emit the raw update.

        Args:
            target_state: The anchor's conditioning state $h_t$, $(B, A, d_h)$.
            source_window: One lag window per anchor, $(B, A, L, \ldots)$ with any trailing shape
                whose product is :attr:`source_dim`.
            lag_valid: $v_{t,\ell}$, $(B, A, L)$, true where the lag carries any available channel.
                ``None`` treats every lag as available.
            selector: $s_{t,\ell}$, broadcastable to $(B, A, L)$, externally set and never learned.
                A zero **removes the lag from the distribution**, so the surviving weights
                renormalise over what is left; there is no version of removing a lag from a
                normalised aggregation that leaves the other weights where they were.
            lag_index: Which lag slots this call covers, as a ``long`` tensor. ``None`` covers
                every slot in order.

        Returns:
            ``(raw_mean_update, raw_scale_update)``, each $(B, A, d_z)$, the second ``None`` on the
            mean-only arm. Both are already zeroed at anchors where no lag survived, so an anchor
            with every selector off or every lag unavailable reproduces the prior exactly rather
            than receiving the output projection's response to an all-zero summary.

        Raises:
            ValueError: If the target state is not 3-D or the wrong width, if the window's lag axis
                does not match the requested slots, if its flattened width is not
                :attr:`source_dim`, or if the two disagree about the batch or anchor axes.
        """
        if target_state.dim() != 3:
            raise ValueError(
                f"target_state must be 3-D (B, A, d_model), got shape "
                f"{tuple(target_state.shape)}"
            )
        if target_state.shape[-1] != self.d_model:
            raise ValueError(
                f"target_state has width {target_state.shape[-1]} against d_model="
                f"{self.d_model}"
            )
        if source_window.dim() < 4:
            raise ValueError(
                f"source_window must be at least 4-D (B, A, L, ...), got shape "
                f"{tuple(source_window.shape)}"
            )
        batch, n_anchors = target_state.shape[0], target_state.shape[1]
        if source_window.shape[:2] != (batch, n_anchors):
            raise ValueError(
                f"source_window leading axes {tuple(source_window.shape[:2])} disagree with the "
                f"target state's {(batch, n_anchors)}; the two describe one anchor set."
            )
        lags = (
            torch.arange(self.n_lags, device=target_state.device)
            if lag_index is None
            else lag_index.to(device=target_state.device, dtype=torch.long)
        )
        if source_window.shape[2] != lags.numel():
            raise ValueError(
                f"source_window has {source_window.shape[2]} lags against {lags.numel()} "
                f"requested lag slots; the key bias is indexed by slot, so a mismatch would score "
                f"the wrong slot's identity rather than fail."
            )
        if bool(((lags < 0) | (lags >= self.n_lags)).any()):
            raise ValueError(
                f"lag_index has entries outside [0, n_lags) = [0, {self.n_lags}); the key bias "
                f"covers the configured slots only."
            )
        flat_source = source_window.reshape(batch, n_anchors, lags.numel(), -1)
        if flat_source.shape[-1] != self.source_dim:
            raise ValueError(
                f"source_window flattens to width {flat_source.shape[-1]} against source_dim="
                f"{self.source_dim}; that width is the encoder's own, so a mismatch means the "
                f"window came from a differently configured encoder."
            )

        n_lags = lags.numel()
        query = self.q_proj(self.q_norm(target_state)).view(
            batch, n_anchors, self.num_heads, self.d_head
        )
        normed = self.kv_norm(flat_source)
        keys = self.k_proj(normed).view(
            batch, n_anchors, n_lags, self.num_heads, self.d_head
        )
        values = self.v_proj(normed).view(
            batch, n_anchors, n_lags, self.num_heads, self.d_head
        )

        # Content score plus the per-lag identity bias, both against the same query.
        scores = torch.einsum("bamd,balmd->balm", query, keys)
        scores = scores + torch.einsum("bamd,lmd->balm", query, self.lag_bias[lags])
        scores = scores * self.scale

        admissible = self._admissible(lag_valid, selector, batch, n_anchors, n_lags, scores.device)
        scores = scores.masked_fill(~admissible[..., None], float("-inf"))
        weights = F.softmax(scores, dim=2)
        # An anchor with no admissible lag scores all minus infinity, which normalises to NaN. Zero
        # is the right reading -- no lag was attended because none could be -- and the gate below
        # is what makes the resulting update exactly zero rather than the projection's response to
        # an all-zero summary, which is a learned constant.
        weights = torch.nan_to_num(weights, nan=0.0)

        attended = torch.einsum("balm,balmd->bamd", weights, values)
        summary = self.o_proj(attended.reshape(batch, n_anchors, self.d_model))
        raw = self.output_proj(summary)
        raw = raw * admissible.any(dim=2, keepdim=True).to(raw.dtype)

        if self.mean_only:
            return raw, None
        mean_update, scale_update = raw.split(self.d_z, dim=-1)
        return mean_update, scale_update

    @staticmethod
    def _admissible(
        lag_valid: Optional[torch.Tensor],
        selector: Optional[torch.Tensor],
        batch: int,
        n_anchors: int,
        n_lags: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Which lags may enter the distribution: available, and not suppressed.

        Args:
            lag_valid: The availability indicator, or ``None`` for all available.
            selector: The intervention selector, or ``None`` for the unintervened forward. Any
                nonzero entry admits its lag; the selector is a switch here rather than a gain,
                because a fractional weight on a normalised distribution is not a suppression of
                anything.
            batch: Batch size.
            n_anchors: Anchors in this call.
            n_lags: Lag slots in this call.
            device: Device to build the default on.

        Returns:
            A boolean $(B, A, L)$ indicator.
        """
        admissible = (
            torch.ones(batch, n_anchors, n_lags, dtype=torch.bool, device=device)
            if lag_valid is None
            else lag_valid.to(torch.bool)
        )
        if selector is not None:
            admissible = admissible & (selector != 0)
        return admissible

    def extra_repr(self) -> str:
        """Report the widths and the head split, which is what a reader checks first."""
        return (
            f"d_model={self.d_model}, d_z={self.d_z}, n_lags={self.n_lags}, "
            f"source_dim={self.source_dim}, num_heads={self.num_heads}, "
            f"mean_only={self.mean_only}"
        )


__all__ = ["LAG_BIAS_STD", "LagAttentionFusion"]
