r"""Paired interventions on the source pathway, and what each of them does and does not measure.

Four arms, each producing the **latent parameters** of an intervened full branch rather than a
score. That return type is the point: every arm is scored later, in one draw loop, against the same
$\epsilon^{(k)}$ as the matched arm, so a reported margin is a difference of predictions and never a
difference of noise. An arm that scored itself would be free to draw its own.

The four ask genuinely different questions and are not interchangeable:

``suppress``
    Set $s_{t,\ell} = 0$ over a band of lags with the target state, the metadata, the original
    per-channel masks, the remaining proposals and $c_L$ all held fixed. It measures how much the
    **fitted computation** relies on the proposals it removed. It is not an intervention on the
    biological system, and it does not identify a lag's contribution: proposals admit a
    reparameterisation $r_\ell \mapsto r_\ell + k_\ell(h_t)$ with $\sum_\ell k_\ell \equiv 0$ that
    leaves the sum, the divergence and every full-model prediction identical while changing what
    removing a single lag does. See :data:`SUPPRESSION_QUALIFICATION`, which is written into the
    artifact rather than only into this docstring.

``replace``
    Substitute the source **values** with the selectors left **enabled**, which is what makes it a
    different question from suppression: the head still runs, still reads the availability
    announcement and still reads the clock, and only what the source said has changed. A valid
    standardized zero is an observation, so this arm's margin is not "the source removed" -- it is
    "the source replaced by this particular alternative".

``permute``
    Pair each recording with a **different** recording's source, preserving within-source time
    order. It tests recording specificity. A positive margin here is necessary and not sufficient:
    both the correct and the shuffled source can be worse than a target-only forecaster, which is
    what the evidence motivating this architecture actually found.

``silence``
    Every selector zero. It verifies the equality invariant and **nothing else**. Its margin is
    exactly zero by construction, so a nonzero one is a defect report rather than a measurement.

**Only the first arm recomputes from cached proposals.** Suppression removes a subset of a sum that
was already evaluated, so the fusion downstream of it -- the scaling, the limiter, the
prior-relative parameters -- is recomputed exactly and no encoder or head runs again. The other
three change what the proposal head *reads*, and a proposal is a function of the source values it
was given, so no cached set can answer a question about different ones. Those three re-run the
forward under a substituted stream, matched in every other argument.

**And only under the local fusion.** The comparator arms aggregate their lags with a normalised
distribution, which has no per-lag term to subtract: removing a lag there means removing it from
the *denominator* as well, so the surviving weights grow. Those arms are suppressed by handing the
forward a band selector instead, through :func:`band_selector`, and the result is a different
quantity from the local arm's under the same band name. It answers the same question -- what does
this model do when it cannot read these lags -- and the two must not be differenced against each
other, which is why the evaluation records the fusion beside every margin.
"""
from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Sequence, Tuple

import torch

from teb_vae.lag_attn.nets.controls import (
    NoCrossGroupPartner,
    groups_can_derange,
    make_derangement,
)
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    bound_update,
    residual_kl,
    residual_parameters,
)

#: The qualification every reported suppression margin travels with, in the artifact and in the
#: analysis docstring rather than in a planning document alone.
#:
#: Three sentences because the three claims fail differently. Locality is a **structural** property
#: this architecture has and the evidence for it is a Jacobian. Suppression is a **measurement**,
#: and what it measures is a fitted parameterisation. A physiological delay is neither: the feature
#: pipeline spreads a dependence across times far wider than the searched window, and no pointwise
#: encoder undoes that.
SUPPRESSION_QUALIFICATION: str = (
    "Locality identifies which stored source time a proposal head can read. Proposal suppression "
    "measures how much the fitted computation depends on the proposals removed. Neither "
    "establishes a unique functional decomposition over lags, and neither is a physiological "
    "delay: an exactly zero-sum reallocation across lags leaves the summed update, the divergence "
    "and every prediction unchanged while changing what removing a single lag does."
)

#: The replacement laws this module implements, and what each one holds fixed.
#:
#: ``zeros`` puts a standardized zero at every available position. On the normalization this task
#: uses, zero is the training-split channel mean over the region the model reads, so it is the
#: least informative value on the coefficient's own scale rather than an arbitrary one.
#:
#: ``constant`` replaces each channel by its own per-sample mean over the stored axis. It removes
#: every temporal variation while leaving the recording's own level standing, which is the arm that
#: isolates *temporal* source content from level.
#:
#: ``mask_only`` asks what the head does with the availability announcement and no value at all. On
#: the recommended arm it is **the same intervention as** ``zeros``, and by construction rather
#: than by coincidence: the encoding is $[x^{\rm safe}, m]$, the value coordinate is the identity,
#: and a zeroed stream therefore leaves exactly the mask. It is named separately because that
#: identity stops holding under the scalar lift, where $\phi_j(0)$ is a learned constant and not
#: zero -- and there the arm is refused rather than silently reported as the zeros arm.
REPLACEMENT_MODES: Tuple[str, ...] = ("zeros", "constant", "mask_only")


def band_lag_mask(
    n_lags: int, lo: int, hi: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    r"""Which lags an inclusive band $[\mathrm{lo}, \mathrm{hi}]$ removes.

    A band is stated in **lags** -- stored steps back from the anchor -- so the band names source
    steps $t - \mathrm{hi}, \ldots, t - \mathrm{lo}$ and is empty exactly when $\mathrm{lo} >
    \mathrm{hi}$. An empty band is legal here and is the reference arm: it removes nothing, and the
    suppression of nothing must reproduce the matched forward exactly, which is the cheapest check
    that a margin is a measurement rather than an artifact of a second code path.

    Args:
        n_lags: The candidate lag count $L$, so lags run over $0, \ldots, L-1$.
        lo: First lag removed, inclusive.
        hi: Last lag removed, inclusive.
        device: Device to build the mask on.

    Returns:
        A boolean $(L,)$ mask, ``True`` at every lag the band removes.

    Raises:
        ValueError: If the band reaches past the last candidate lag, which would name a wider
            window than the model searched, or if either end is negative.
    """
    if int(lo) < 0 or int(hi) < 0:
        raise ValueError(
            f"a lag band's ends are counts of stored steps back and must be >= 0, "
            f"got [{lo}, {hi}]"
        )
    if int(hi) >= int(n_lags):
        raise ValueError(
            f"lag band [{lo}, {hi}] reaches lag {hi}, past the model's last candidate lag "
            f"{int(n_lags) - 1}. A band wider than the searched window would be reported under a "
            f"name that overstates what was removed."
        )
    lags = torch.arange(int(n_lags), device=device)
    return (lags >= int(lo)) & (lags <= int(hi))


def band_selector(
    removed: torch.Tensor, batch: int, n_anchors: int, *, dtype: torch.dtype
) -> torch.Tensor:
    r"""The selector that removes a band, for an arm whose fusion must be re-run to suppress it.

    $$s_{t,\ell} = \mathbb 1[\ell \notin \mathcal B].$$

    Materialised at the full $(B, A, L)$ shape rather than broadcast, because it is handed to a
    forward that slices it by anchor chunk, and a shape that broadcasts in one place and is sliced
    in another is a shape that eventually gets sliced along the wrong axis.

    An **empty** band gives an all-ones selector, which every fusion here treats as the
    unintervened forward: the local head multiplies by one and the attention head admits every lag
    it already admitted. That is what makes the empty-band arm reproduce the matched forward on
    every arm rather than only on the one whose suppression is subtractive.

    Args:
        removed: Boolean mask of the removed lags, $(L,)$.
        batch: Batch size $B$.
        n_anchors: Decoded anchors $A$.
        dtype: Floating dtype the forward's selector is expected in.

    Returns:
        The selector $(B, A, L)$.
    """
    keep = (~removed.to(torch.bool)).to(dtype)
    return keep.view(1, 1, -1).expand(int(batch), int(n_anchors), -1).contiguous()


def suppressed_parameters(
    model: Any, outputs: Dict[str, torch.Tensor], removed: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""Recompute the full branch with a band of proposals removed, from the cached set.

    The removal is written as a **subtraction** of the removed band from the matched raw update,

    $$\bar a^{\setminus\mathcal B}_t
      = \bar a_t - c_L \sum_{\ell \in \mathcal B} r^\mu_{t,\ell},$$

    and everything downstream of it -- the limiter, the prior-relative parameters, the divergence
    -- is then recomputed exactly rather than adjusted. The subtraction is what makes an **empty**
    band bitwise identical to the matched arm: the removed sum is exactly $0$, so nothing is
    recomputed at all. The mirror case is written out rather than subtracted: a band removing
    *every* lag leaves a sum over an empty index set, which is zero by definition, and reaching it
    by cancelling a chunk-accumulated total against a single-pass one would leave a residue of a
    few units in the last place and a full branch that merely nearly equals the prior.

    $c_L$ is held at its configured value. Renormalising it by the surviving lag count would make
    the suppressed arm weigh its remaining evidence differently from the matched arm, and the
    margin would then contain that reweighting as well as the removal.

    Args:
        model: The net, for $c_L$ and the two residual bounds.
        outputs: A forward's dict, taken with ``return_proposals=True``. Reads ``mean_proposals``,
            ``raw_update_mean``, ``mu_prior``, ``logvar_prior`` and, on an arm that has one, the
            two scale twins.
        removed: Boolean mask of the removed lags, $(L,)$ or broadcastable to $(B, A, L)$.

    Returns:
        ``{'mu_post', 'logvar_post', 'update_mean', 'kld_per_anchor_dim', 'kld_per_anchor'}``,
        plus ``'update_logsigma'`` where the arm has a scale channel.

    Raises:
        ValueError: If the model's fusion is not the local sum. A normalised aggregation has no
            per-lag term to subtract, and subtracting one from its output would produce a number
            with no interpretation at all rather than a suppression. Those arms go through
            :func:`band_selector` and a re-run forward.
        KeyError: If the forward was run without ``return_proposals``, naming the flag. The
            proposals are the whole input to this computation, and a suppression computed from the
            summed update alone would be a suppression of nothing.
    """
    if str(getattr(model, "lag_fusion", "local")) != "local":
        raise ValueError(
            f"suppressed_parameters subtracts a band from a sum of per-lag updates, and this "
            f"model's lag_fusion is {getattr(model, 'lag_fusion')!r}, which produces no such sum: "
            f"its lags enter a normalised distribution and removing one changes the weight of "
            f"every other. Suppress that arm with band_selector and a re-run forward instead."
        )
    if "mean_proposals" not in outputs:
        raise KeyError(
            "suppressed_parameters needs the per-lag proposals: call the forward with "
            "return_proposals=True. The summed update alone cannot say what a band contributed."
        )
    proposals = outputs["mean_proposals"]
    keep_nothing = bool(removed.all())

    def suppress(raw_matched: torch.Tensor, per_lag: torch.Tensor) -> torch.Tensor:
        """Remove the band from one proposal channel's matched raw update."""
        if keep_nothing:
            # A sum over no lags, written as one. See the docstring: the subtractive form would
            # leave a floating-point residue here rather than the exact zero the invariant needs.
            return torch.zeros_like(raw_matched)
        band = removed.to(per_lag.dtype)
        if band.dim() == 1:
            band = band.view(1, 1, -1)
        # The trailing axis is the latent one and the band says nothing about it, so the mask
        # broadcasts over it rather than being repeated $d_z$ times.
        return raw_matched - float(model.lag_scale) * (per_lag * band.unsqueeze(-1)).sum(dim=2)

    raw_a = suppress(outputs["raw_update_mean"], proposals)
    a = bound_update(raw_a, model.residual_mu_scale)

    b: Optional[torch.Tensor] = None
    if "scale_proposals" in outputs and "raw_update_logsigma" in outputs:
        raw_b = suppress(outputs["raw_update_logsigma"], outputs["scale_proposals"])
        b = bound_update(raw_b, model.residual_logsigma_scale)

    mu_post, logvar_post = residual_parameters(
        outputs["mu_prior"], outputs["logvar_prior"], a, b
    )
    kld_dim = residual_kl(a, b)
    result: Dict[str, torch.Tensor] = {
        "mu_post": mu_post,
        "logvar_post": logvar_post,
        "update_mean": a,
        "kld_per_anchor_dim": kld_dim,
        "kld_per_anchor": kld_dim.sum(dim=-1),
    }
    if b is not None:
        result["update_logsigma"] = b
    return result


def replaced_source_stream(
    u_stream: torch.Tensor, mode: str, *, scalar_lift: bool = False
) -> torch.Tensor:
    r"""Build the substituted source stream one replacement arm reads.

    Every mode leaves the stream's **shape, dtype, device and finiteness** untouched, which is what
    keeps the availability announcement fixed: the per-channel mask is the product of the in-range
    indicator, the warm-up indicator and the accepted-finite indicator, and none of the three sees
    a value. The metadata clock reads stored position alone and cannot move either. Both are
    asserted at the call site rather than trusted here.

    Args:
        u_stream: The declared source stream $(B, T, c_u)$.
        mode: One of :data:`REPLACEMENT_MODES`.
        scalar_lift: Whether the model's encoder carries the optional per-channel lift, which is
            what decides whether ``mask_only`` is a distinct intervention at all.

    Returns:
        The substituted stream, shaped like ``u_stream``.

    Raises:
        ValueError: On an unknown mode, or on ``mask_only`` over a lifted encoder, where zeroing
            the stream does **not** leave only the mask -- $\phi_j(0)$ is a learned constant -- so
            the arm would silently be a fourth thing reported under a name that describes a
            different one.
    """
    if mode not in REPLACEMENT_MODES:
        raise ValueError(
            f"unknown replacement mode {mode!r}; the arms are {list(REPLACEMENT_MODES)}"
        )
    if mode == "constant":
        # Per sample and per channel, over the stored axis: every temporal variation removed, the
        # recording's own level left standing.
        return u_stream.mean(dim=1, keepdim=True).expand_as(u_stream).contiguous()
    if mode == "mask_only" and scalar_lift:
        raise ValueError(
            "the mask_only arm zeroes the stream, which leaves only the mask on the identity "
            "encoder and does not on a lifted one: the lift of a zero is a learned constant, not "
            "zero. Run this arm on the recommended encoder, or add an encoding-level hook that "
            "drops the value coordinates -- do not report the lifted result under this name."
        )
    return torch.zeros_like(u_stream)


def cross_recording_index(
    recordings: Sequence[Hashable],
    *,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""Draw the pairing the permutation arm uses, or refuse to draw one.

    Grouped by recording rather than merely fixed-point-free. An unshuffled loader over
    per-recording shards puts a segment beside its own recording's next segment, and a plain
    derangement pairs those two happily -- which is not a stranger's source, and weakens the
    control by an amount nothing downstream reports.

    The permutation is over the batch's **rows**, so each substituted source keeps its own
    within-source time order intact. Shuffling individual source steps instead would destroy the
    autocorrelation that makes a lag window mean anything, and would test a different claim.

    Args:
        recordings: One recording identifier per batch element.
        generator: Generator for the draw, so a run is reproducible.
        device: Device of the returned index.

    Returns:
        A $(B,)$ long tensor $\pi$, with $g_{\pi(i)} \neq g_i$ everywhere.

    Raises:
        NoCrossGroupPartner: If one recording holds more than half the batch, where no
            cross-recording pairing exists at all. A caller running a whole loader excludes such a
            batch and **counts** the exclusion: a control that silently stopped being a control
            looks exactly like one that works.
    """
    return make_derangement(
        len(recordings), generator=generator, device=device, groups=list(recordings)
    )


def same_recording_pairs(recordings: Sequence[Hashable], index: torch.Tensor) -> int:
    r"""Count pairings that landed inside their own recording.

    Counted off the permutation that actually ran rather than asserted from the way it was drawn.
    Zero by construction under a grouped draw, and reported anyway, because the failure this
    catches is a grouped draw that silently stopped being grouped.

    Args:
        recordings: One recording identifier per batch element.
        index: The pairing $\\pi$.

    Returns:
        How many positions were paired with their own recording.
    """
    return sum(
        1
        for position, partner in enumerate(index.tolist())
        if recordings[position] == recordings[partner]
    )


__all__ = [
    "NoCrossGroupPartner",
    "REPLACEMENT_MODES",
    "SUPPRESSION_QUALIFICATION",
    "band_lag_mask",
    "band_selector",
    "cross_recording_index",
    "groups_can_derange",
    "replaced_source_stream",
    "same_recording_pairs",
    "suppressed_parameters",
]
