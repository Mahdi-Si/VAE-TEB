r"""Band suppression margins, the cancellation ratio, and per-lag exposure -- with their limits.

Three readouts, and the first two are useless without the third.

**The band suppression margin** is
$J_{\mathcal B} = \widehat{\mathbb E}\bigl[D^{(K)}_{q \setminus \mathcal B} - D^{(K)}_q\bigr]$: how
much worse the fitted model predicts when a band's proposals are set aside. It is **not** normalised
to sum to the total gain or to the divergence, and it is not made to. Its parts do not add up,
because the limiter is applied after the summation and the update is bounded rather than linear, and
a normalisation would present a set of numbers that add to something as a decomposition of that
something. It can be negative, which is a real result rather than a failure: a fitted model can
predict better without a band it was fitted with.

**The cancellation ratio** is
$\kappa_t = \lVert \sum_\ell r_{t,\ell}\rVert_2 / (\sum_\ell \lVert r_{t,\ell}\rVert_2 +
\varepsilon)$, and it travels with **both** of its parts because the ratio alone is ambiguous in
exactly the way that matters. A near-zero $\kappa$ means either that the proposals cancel or that
they are all near zero, and those are a source pathway arguing with itself and a source pathway that
has switched off. The denominator tells them apart and nothing else does.

**Exposure** is the denominator of everything above. A per-lag readout computed over anchors where
most lags are out of range, or where the source channels have not warmed up, measures the
availability schedule. Every band reports how many anchors and how many channel-anchor pairs it
actually had, and a bin with no support is recorded as **missing** rather than as a measured zero:
the diagnosis this architecture answers reported bands whose intervals spanned zero, and an
unsupported bin presented as a zero is the one reading that cannot be distinguished from an absent
effect afterwards.

Every artifact this module writes carries
:data:`~teb_vae.lag_slot_transformer_cfs.nets.controls.SUPPRESSION_QUALIFICATION` verbatim. That is
deliberate placement rather than belt and braces: a caveat that lives only in a planning document is
one edit away from being dropped from the thing a reader actually opens.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from teb_vae.lag_slot_transformer_cfs.nets.controls import (
    SUPPRESSION_QUALIFICATION,
    band_lag_mask,
)

#: What a bin with no support is recorded as. ``None`` survives JSON as ``null``, which a reader and
#: every downstream tool can tell from a zero; a NaN does not survive ``json.dump`` at all, and a
#: zero is the reading this module exists to prevent.
MISSING = None


def band_masks(
    bands: Mapping[str, Sequence[int]], n_lags: int, *, device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Turn the configured inclusive lag bands into removal masks, plus the two reference arms.

    The two arms are added here rather than by each caller, because they are what makes a margin
    readable. ``none`` removes nothing and must reproduce the matched arm exactly; ``all`` removes
    every declared band at once, which on a partition of the window is every lag, and must reproduce
    the target-only prior. **Whole-band and joint removals are scored before any single-lag result
    is reported**, which is what this ordering encodes: a single-lag peak read off a window whose
    joint removal does nothing is a peak in noise.

    Args:
        bands: ``{name: [lo, hi]}``, inclusive, in stored steps back from the anchor.
        n_lags: The candidate lag count $L$.
        device: Device to build the masks on.

    Returns:
        ``{name: (L,) bool}`` in declaration order, with ``none`` first and ``all`` last.

    Raises:
        ValueError: If a band is empty, or reaches past the last candidate lag. Both would be
            reported under a name that describes something other than what was removed.
    """
    masks: Dict[str, torch.Tensor] = {
        "none": torch.zeros(int(n_lags), dtype=torch.bool, device=device)
    }
    for name, pair in bands.items():
        lo, hi = int(pair[0]), int(pair[1])
        if lo > hi:
            raise ValueError(
                f"lag band {name!r} is empty ([{lo}, {hi}]): it removes nothing, so its margin "
                f"would be identically zero and read as 'the source did not matter there'."
            )
        masks[name] = band_lag_mask(n_lags, lo, hi, device=device)
    if bands:
        joint = torch.zeros(int(n_lags), dtype=torch.bool, device=device)
        for name in bands:
            joint = joint | masks[name]
        masks["all"] = joint
    return masks


def lag_exposure(
    lag_valid: torch.Tensor,
    channel_mask: torch.Tensor,
    contributing: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    r"""How much source each lag actually had, over the anchors the run scored.

    Two counts, because index support and feature warm-up are different conditions that fail at
    different anchors. ``anchors_per_lag`` counts scored anchors at which the lag carried **any**
    available channel; ``channels_per_lag`` counts available channels summed over those anchors. A
    lag can be perfectly in range at every anchor and carry two of forty-six channels, and the first
    count alone would report it as fully exposed.

    Args:
        lag_valid: Whether each anchor-lag pair carries any available channel, $(B, A, L)$.
        channel_mask: The per-channel availability of the gathered window, $(B, A, L, C_U)$.
        contributing: The $0/1$ scored-anchor indicator, $(B, A)$.

    Returns:
        ``{'anchors_per_lag', 'channels_per_lag', 'scored_anchors'}``, the first two $(L,)$ and the
        last a scalar tensor. Summed over the batch, so a caller accumulates them across batches by
        adding.

    Raises:
        ValueError: If the two masks disagree about the anchor or lag axes, which would count one
            geometry's availability against another's anchors.
    """
    if lag_valid.shape != channel_mask.shape[:3]:
        raise ValueError(
            f"lag_valid is {tuple(lag_valid.shape)} against a channel mask of "
            f"{tuple(channel_mask.shape)}; the first three axes must agree."
        )
    scored = contributing.to(torch.float64)
    return {
        "anchors_per_lag": (lag_valid.to(torch.float64) * scored[:, :, None]).sum(dim=(0, 1)),
        "channels_per_lag": (
            channel_mask.to(torch.float64).sum(dim=-1) * scored[:, :, None]
        ).sum(dim=(0, 1)),
        "scored_anchors": scored.sum(),
    }


def channel_exposure(
    channel_mask: torch.Tensor, contributing: torch.Tensor
) -> torch.Tensor:
    r"""How many scored anchor-lag pairs each source channel was available at.

    The companion of :func:`lag_exposure` on the other axis. A channel whose warm-up outlasts the
    anchor floor is unavailable at every early anchor however well the lag index behaves, and a
    per-lag profile pooled over channels hides that entirely.

    Args:
        channel_mask: The per-channel availability of the gathered window, $(B, A, L, C_U)$.
        contributing: The $0/1$ scored-anchor indicator, $(B, A)$.

    Returns:
        A $(C_U,)$ tensor of counts, summed over the batch.
    """
    scored = contributing.to(torch.float64)[:, :, None, None]
    return (channel_mask.to(torch.float64) * scored).sum(dim=(0, 1, 2))


def cancellation_totals(
    outputs: Mapping[str, torch.Tensor], contributing: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    r"""One batch's cancellation sums, per proposal channel, over the scored anchors.

    Sums rather than means, so a pass accumulates them across batches by adding and the final
    figure is the mean over every scored anchor of the split. A mean of per-batch means would
    weight a batch holding one segment equally with a batch holding thirty-two.

    Reported for the mean proposals and, where the arm has one, for the scale proposals. The two
    saturate and cancel independently, and a mean channel that has switched off beside a scale
    channel that has not is a state one pooled number cannot show.

    Args:
        outputs: A forward's dict, carrying ``cancellation_ratio_*``, ``cancellation_numerator_*``
            and ``cancellation_denominator_*``.
        contributing: The $0/1$ scored-anchor indicator, $(B, A)$.

    Returns:
        ``{channel: {'ratio_sum', 'numerator_sum', 'denominator_sum', 'scored_anchors'}}``.
    """
    weights = contributing.to(torch.float64)
    total = float(weights.sum())
    totals: Dict[str, Dict[str, float]] = {}
    for channel in ("mean", "scale"):
        if f"cancellation_ratio_{channel}" not in outputs:
            continue
        block = {"scored_anchors": total}
        for part in ("ratio", "numerator", "denominator"):
            values = outputs[f"cancellation_{part}_{channel}"].to(torch.float64)
            block[f"{part}_sum"] = float((values * weights).sum())
        totals[channel] = block
    return totals


def merge_cancellation(
    left: Optional[Dict[str, Dict[str, float]]], right: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Accumulate two batches of cancellation sums.

    Args:
        left: The running totals, or ``None`` on the first batch.
        right: This batch's sums.

    Returns:
        The summed totals.
    """
    if left is None:
        return {channel: dict(block) for channel, block in right.items()}
    merged = {channel: dict(block) for channel, block in left.items()}
    for channel, block in right.items():
        target = merged.setdefault(channel, {name: 0.0 for name in block})
        for name, value in block.items():
            target[name] = target.get(name, 0.0) + float(value)
    return merged


def cancellation_summary(
    totals: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, Any]]:
    r"""Reduce the accumulated sums to the ratio and both of its parts.

    **All three travel, and that is the whole point of this readout.** A near-zero $\kappa$ means
    either that the proposals cancel or that they are all near zero, and those are a source pathway
    arguing with itself and a source pathway that has switched off. Only the denominator separates
    them, and a summary carrying the ratio alone leaves the two indistinguishable.

    Args:
        totals: The accumulated sums from :func:`merge_cancellation`.

    Returns:
        ``{channel: {'ratio', 'numerator', 'denominator', 'scored_anchors'}}``, each figure
        :data:`MISSING` where nothing was scored.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for channel, block in totals.items():
        scored = float(block.get("scored_anchors", 0.0))
        entry: Dict[str, Any] = {"scored_anchors": scored}
        for part in ("ratio", "numerator", "denominator"):
            entry[part] = MISSING if scored <= 0.0 else float(block[f"{part}_sum"]) / scored
        summary[channel] = entry
    return summary


def band_suppression_block(
    headline: Mapping[str, Any],
    band_exposure: Mapping[str, Mapping[str, float]],
    *,
    matched_key: str = "nll_full",
    suppressed_prefix: str = "nll_suppress:",
) -> Dict[str, Any]:
    r"""Each band's margin against the matched full branch, with its usable counts beside it.

    $$J_{\mathcal B}
      = \widehat{\mathbb E}\bigl[D^{(K)}_{q\setminus\mathcal B,t}\bigr]
      - \widehat{\mathbb E}\bigl[D^{(K)}_{q,t}\bigr],$$

    a difference of two arms' equal-recording means. Each arm's own bootstrap interval travels
    beside the margin rather than an interval of the difference, so a reader sees both ends of both
    arms; the margin is a difference of point estimates and is labelled as one.

    **Nothing here is renormalised.** The margins do not sum to the gap, to the divergence, or to
    each other: the limiter is applied after the summation, so the update is bounded rather than
    linear in the proposals, and a normalisation would present numbers that add to something as a
    decomposition of that something.

    A band whose available-channel count is zero had nothing to remove -- it lies wholly in the
    warm-up staircase, or past the start of the record -- and its margin is recorded as
    :data:`MISSING`. A zero there is the one reading that cannot afterwards be told apart from a
    fully available band that did not matter.

    Args:
        headline: The bootstrapped per-column block, keyed by column name.
        band_exposure: ``{band: {'anchors', 'channels'}}`` from :func:`band_exposure`.
        matched_key: The column the margins are taken against.
        suppressed_prefix: The prefix marking a suppressed arm's column.

    Returns:
        ``{band: {'margin_nats', 'suppressed_nll', 'band_anchors', 'band_channels'}}``.
    """
    matched = (headline.get(matched_key) or {}).get("point")
    block: Dict[str, Any] = {}
    for name, record in headline.items():
        if not name.startswith(suppressed_prefix):
            continue
        band = name[len(suppressed_prefix):]
        counts = band_exposure.get(band, {})
        channels = float(counts.get("channels", 0.0))
        # The reference arm removes nothing, so a zero channel count there is not an absence of
        # support. Every other band with no available source had nothing to remove.
        unsupported = band != "none" and channels <= 0.0
        block[band] = {
            "margin_nats": (
                MISSING
                if unsupported or matched is None
                else float(record["point"]) - float(matched)
            ),
            "suppressed_nll": record,
            "band_anchors": float(counts.get("anchors", 0.0)),
            "band_channels": channels,
        }
    return block


def qualified_report(blocks: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the qualification every lag readout must be read with.

    The text is copied into the artifact rather than referenced, so a reader with the JSON and
    nothing else has it. A test asserts its presence in what a run writes, which is what stops it
    being dropped by a later edit that tidied the summary.

    Args:
        blocks: The assembled lag readout blocks.

    Returns:
        The same blocks with the qualification beside them.
    """
    return {**blocks, "qualification": SUPPRESSION_QUALIFICATION}


def merge_counts(
    left: Optional[Dict[str, torch.Tensor]], right: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Accumulate two batches' exposure counts.

    Args:
        left: The running totals, or ``None`` on the first batch.
        right: This batch's counts.

    Returns:
        The summed counts.
    """
    if left is None:
        return {name: value.clone() for name, value in right.items()}
    return {name: left[name] + value for name, value in right.items()}


def band_exposure(
    masks: Mapping[str, torch.Tensor], counts: Mapping[str, torch.Tensor]
) -> Dict[str, Dict[str, float]]:
    """Reduce the per-lag exposure onto the declared bands.

    Args:
        masks: ``{band: (L,) bool}`` removal masks.
        counts: The accumulated :func:`lag_exposure` totals.

    Returns:
        ``{band: {'anchors', 'channels'}}``, summed over the band's own lags.
    """
    anchors, channels = counts["anchors_per_lag"], counts["channels_per_lag"]
    return {
        band: {
            "anchors": float(anchors[mask].sum()),
            "channels": float(channels[mask].sum()),
        }
        for band, mask in masks.items()
    }


__all__ = [
    "MISSING",
    "band_exposure",
    "band_masks",
    "band_suppression_block",
    "cancellation_summary",
    "cancellation_totals",
    "channel_exposure",
    "lag_exposure",
    "merge_cancellation",
    "merge_counts",
    "qualified_report",
]
