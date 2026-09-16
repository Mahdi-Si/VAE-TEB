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

**The per-lag latent profile** is the fine-grained companion of the band margins, and it is read
in latent space rather than in the forecast because that is where it is cheap enough to take at
every lag: for each lag $\ell$ the bounded update is recomputed with that lag's proposals alone
removed, from the cached array, and three things are reported -- the proposal's own norm
$\lVert r_{t,\ell} \rVert_2$, the shift $\lVert a_t - a_t^{\setminus \ell} \rVert_2$ it makes to
the bounded mean update, and the drop $K_t - K_t^{\setminus \ell}$ it makes to the divergence. None
of the three is a per-lag allocation: the shifts and drops of different lags do not sum to anything,
and the reallocation the qualification names changes every one of them while changing no prediction.

**Every margin travels with a paired interval over recordings.** Two arms scored under one set of
draws differ per recording, so the interval that belongs on a margin is the interval of those
differences and not two overlapping intervals of the two arms' own scores, which is wider by exactly
the shared variation the pairing removes.

Every artifact this module writes carries
:data:`~teb_vae.lag_slot_transformer_cfs.nets.controls.SUPPRESSION_QUALIFICATION` verbatim. That is
deliberate placement rather than belt and braces: a caveat that lives only in a planning document is
one edit away from being dropped from the thing a reader actually opens.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from teb_vae.lag_attn.eval.stats import MIN_GROUP_SIZE, bootstrap_ci
from teb_vae.lag_slot_transformer_cfs.nets.controls import (
    SUPPRESSION_QUALIFICATION,
    band_lag_mask,
)
from teb_vae.lag_slot_transformer_cfs.nets.lag_updates import (
    bound_update,
    residual_kl,
)

#: What a bin with no support is recorded as. ``None`` survives JSON as ``null``, which a reader and
#: every downstream tool can tell from a zero; a NaN does not survive ``json.dump`` at all, and a
#: zero is the reading this module exists to prevent.
MISSING = None

#: Prefix that marks a scored arm as the suppression of one lag band. One place, read by the pass
#: that names the arms, the analyses that list them and the figures that draw them, so the three
#: cannot come to spell it differently.
SUPPRESSION_PREFIX = "suppress:"

#: Lags recomputed at once by the per-lag latent profile. The removed-lag update is one array of
#: the proposals' own shape per chunk, so the chunk bounds the transient at a fraction of the
#: proposal array that is already held rather than at several copies of it.
LAG_PROFILE_CHUNK = 16


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
    suppressed_prefix: str = f"nll_{SUPPRESSION_PREFIX}",
    intervals: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    r"""Each band's margin against the matched full branch, with its usable counts beside it.

    $$J_{\mathcal B}
      = \widehat{\mathbb E}\bigl[D^{(K)}_{q\setminus\mathcal B,t}\bigr]
      - \widehat{\mathbb E}\bigl[D^{(K)}_{q,t}\bigr],$$

    a difference of two arms' equal-recording means. Each arm's own bootstrap interval travels
    beside the margin so a reader sees both ends of both arms, and where the caller hands in the
    paired interval of the per-recording differences (:func:`paired_margin`) that travels too,
    under ``margin_interval`` -- it is the interval a claim about the margin rests on, since the
    two arms' own intervals overlap by the shared variation the pairing removes.

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
        intervals: ``{band: paired bootstrap record}`` from :func:`paired_margin`, or ``None``
            when the caller has no per-recording table to pair over.

    Returns:
        ``{band: {'margin_nats', 'margin_interval', 'suppressed_nll', 'band_anchors',
        'band_channels'}}``.
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
            # Absent rather than a point-only record when no pairing was possible, and absent on
            # an unsupported band for the reason the point is: an interval on a band that had
            # nothing to remove would be an interval on the availability schedule.
            "margin_interval": (
                MISSING
                if unsupported or intervals is None or band not in intervals
                else dict(intervals[band])
            ),
            "suppressed_nll": record,
            "band_anchors": float(counts.get("anchors", 0.0)),
            "band_channels": channels,
        }
    # In declaration order -- the order the exposure carries, which is the masks' -- rather than
    # in the alphabetical order the headline happens to hold its columns in, so the figures and
    # the tables list the bands as the delta declared them. A band the exposure does not name
    # keeps its place after them rather than being dropped.
    ordered = [band for band in band_exposure if band in block]
    ordered += [band for band in block if band not in band_exposure]
    return {band: block[band] for band in ordered}


def paired_margin(
    per_recording: Mapping[str, Mapping[str, float]],
    intervened: str,
    matched: str,
    *,
    resamples: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    r"""The interval of an arm's margin, drawn over the per-recording **differences**.

    $$\Delta_g = \bar D_{g}^{\mathrm{intervened}} - \bar D_{g}^{\mathrm{matched}}$$

    per recording $g$, bootstrapped over recordings. Both arms were scored on the same recordings
    under the same draws, so their difference exists per recording and the shared variation --
    which recordings are hard to forecast at all -- cancels inside it. Two separate intervals on
    the two arms would each carry that variation whole, and reading their overlap as "no
    difference" is the specific misreading the pairing exists to remove.

    Args:
        per_recording: ``{recording: {column: value}}``, the table the pass aggregated.
        intervened: The intervened arm's column.
        matched: The matched arm's column.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        The bootstrap record of the mean difference, with ``n_paired`` beside it; the point and
        the bounds are ``NaN`` with a ``note`` below :data:`~teb_vae.lag_attn.eval.stats.MIN_GROUP_SIZE`
        paired recordings, and :data:`MISSING` when neither column exists in the table.
    """
    differences: List[float] = []
    for row in per_recording.values():
        left, right = row.get(intervened), row.get(matched)
        if left is None or right is None:
            continue
        differences.append(float(left) - float(right))
    if not differences:
        return MISSING
    record = bootstrap_ci(differences, resamples=int(resamples), seed=int(seed))
    record["n_paired"] = len(differences)
    record["pairing"] = "per-recording difference of the intervened and matched arms"
    return record


def bootstrap_curve(
    rows: Mapping[str, Any],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    r"""A recording-level percentile bootstrap of a per-recording **vector**, one draw set for all.

    The same interval :func:`~teb_vae.lag_attn.eval.stats.bootstrap_ci` builds for a scalar,
    taken at every position of the vector under **one** resampling of the recordings. One rather
    than one per position, and that is the point: a horizon curve or a lag profile whose positions
    were resampled independently would have an interval at each step that a reader cannot follow
    from one step to the next, because the recordings behind adjacent steps would differ.

    Args:
        rows: ``{recording: vector}``, every vector the same length $N$. A recording whose vector
            is not entirely finite is dropped and counted.
        resamples: Bootstrap resamples.
        seed: Seed for the resampling.
        confidence: Coverage of the interval.

    Returns:
        ``{'point', 'lo', 'hi'}`` as lists of length $N$, with ``n``, ``n_dropped``,
        ``resamples``, ``seed``, ``confidence`` and ``method`` beside them. The lists are ``NaN``
        throughout, with a ``note``, below :data:`~teb_vae.lag_attn.eval.stats.MIN_GROUP_SIZE`
        recordings.

    Raises:
        ValueError: If the vectors disagree in length, if ``confidence`` is outside $(0, 1)$, or
            if ``resamples`` is not positive.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence!r}.")
    if int(resamples) < 1:
        raise ValueError(f"resamples must be positive, got {resamples!r}.")
    vectors = [np.asarray(list(vector), dtype=np.float64) for vector in rows.values()]
    lengths = {int(vector.size) for vector in vectors}
    if len(lengths) > 1:
        raise ValueError(
            f"every per-recording vector must have one length, got lengths {sorted(lengths)}."
        )
    width = lengths.pop() if lengths else 0
    finite = [vector for vector in vectors if vector.size and np.isfinite(vector).all()]
    alpha = 1.0 - float(confidence)
    record: Dict[str, Any] = {
        "statistic": "mean",
        "n": len(finite),
        "n_dropped": len(vectors) - len(finite),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "seed": int(seed),
        "method": "percentile bootstrap over recordings, one resampling for every position",
        "point": [float("nan")] * width,
        "lo": [float("nan")] * width,
        "hi": [float("nan")] * width,
    }
    if len(finite) < MIN_GROUP_SIZE:
        record["note"] = (
            f"only {len(finite)} finite recording(s); below the minimum of {MIN_GROUP_SIZE} a "
            f"bootstrap interval reproduces the sample rather than estimating its spread"
        )
        return record
    matrix = np.stack(finite, axis=0)
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, matrix.shape[0], size=(int(resamples), matrix.shape[0]))
    means = matrix[draws].mean(axis=1)
    record["point"] = matrix.mean(axis=0).tolist()
    record["lo"] = np.quantile(means, alpha / 2.0, axis=0).tolist()
    record["hi"] = np.quantile(means, 1.0 - alpha / 2.0, axis=0).tolist()
    return record


def per_lag_latent_maps(
    model: Any, outputs: Mapping[str, torch.Tensor], contributing: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""One batch's per-lag latent readouts at **every scored anchor**, before any reduction.

    For every lag $\ell$ the bounded update is recomputed with that lag's proposals alone
    removed, from the cached array and by the same subtraction the band arms use:

    $$\bar a_t^{\setminus \ell} = \bar a_t - c_L\, r^\mu_{t,\ell}, \qquad
      a_t^{\setminus \ell} = a_{\max}\tanh\bigl(\bar a_t^{\setminus \ell} / a_{\max}\bigr),$$

    and likewise for the scale channel where the arm has one. Three maps come back, each
    $(B, A, L)$:

    * ``proposal_norm``: $\lVert r^\mu_{t,\ell} \rVert_2$, what the head emitted;
    * ``update_shift``: $\lVert a_t - a_t^{\setminus \ell} \rVert_2$, what removing it does to
      the bounded mean update -- zero where the limiter has saturated, however large the
      proposal, which is what separates this from the first;
    * ``divergence_drop``: $K_t - K_t^{\setminus \ell}$, what removing it does to the
      divergence. Signed: removing a lag can raise the divergence when its proposal was
      cancelling another's.

    ``scale_proposal_norm`` joins them on an arm with a scale channel, and ``live`` is the
    $(B, A, L)$ weight -- the lag carried an available channel at a scored anchor -- that every
    reduction of these maps is taken under. An out-of-range or cold lag has an exactly zero
    proposal, so its three entries are exactly zero, and the weight is what keeps a zero that
    means "nothing was there" out of a mean over the lags that carried source.

    **None of the three is an allocation.** The reallocation the qualification names --
    $r_\ell \mapsto r_\ell + k_\ell(h_t)$ with $\sum_\ell k_\ell \equiv 0$ -- leaves the update,
    the divergence and every prediction unchanged while changing all three readouts at every lag.

    The maps are what the collection pass writes to the per-anchor sidecar, the per-sample
    profiles are their within-segment means, and :func:`per_lag_latent_totals` is their sum over
    the batch -- one computation behind the three, so the sidecar, the table and the pooled
    profile cannot come to disagree.

    Args:
        model: The net, for $c_L$ and the two residual bounds.
        outputs: A forward's dict taken with ``return_proposals=True`` on the local fusion.
        contributing: The $0/1$ scored-anchor indicator, $(B, A)$.

    Returns:
        ``{'proposal_norm', 'update_shift', 'divergence_drop', 'live'}`` and, where the arm has
        a scale channel, ``'scale_proposal_norm'``, each $(B, A, L)$ in float64.

    Raises:
        KeyError: If the forward was run without ``return_proposals``, naming the flag.
    """
    if "mean_proposals" not in outputs:
        raise KeyError(
            "per_lag_latent_maps needs the per-lag proposals: call the forward with "
            "return_proposals=True."
        )
    proposals = outputs["mean_proposals"]  # (B, A, L, d_z)
    scale_proposals = outputs.get("scale_proposals")
    raw_b_full = outputs.get("raw_update_logsigma")
    n_lags = int(proposals.shape[2])
    scale = float(model.lag_scale)
    # Weighted by the anchors the lag was live at: an out-of-range or cold lag has an exactly
    # zero proposal, and averaging its zero shift over every scored anchor would dilute a live
    # lag's figure by the schedule rather than by the source.
    live = (outputs["lag_valid"].to(torch.float64) * contributing.to(torch.float64)[:, :, None])

    a_full = outputs["update_mean"]
    kl_full = outputs["kld_per_anchor"]
    raw_a_full = outputs["raw_update_mean"]

    proposal_norm = proposals.norm(dim=-1).to(torch.float64)  # (B, A, L)
    update_shift = torch.zeros_like(proposal_norm)
    divergence_drop = torch.zeros_like(proposal_norm)
    for start in range(0, n_lags, LAG_PROFILE_CHUNK):
        stop = min(start + LAG_PROFILE_CHUNK, n_lags)
        raw_a = raw_a_full[:, :, None, :] - scale * proposals[:, :, start:stop]
        a_minus = bound_update(raw_a, model.residual_mu_scale)
        b_minus: Optional[torch.Tensor] = None
        if scale_proposals is not None and raw_b_full is not None:
            raw_b = raw_b_full[:, :, None, :] - scale * scale_proposals[:, :, start:stop]
            b_minus = bound_update(raw_b, model.residual_logsigma_scale)
        update_shift[:, :, start:stop] = (a_full[:, :, None, :] - a_minus).norm(dim=-1)
        divergence_drop[:, :, start:stop] = (
            kl_full[:, :, None].to(torch.float64)
            - residual_kl(a_minus, b_minus).sum(dim=-1).to(torch.float64)
        )

    maps = {
        "proposal_norm": proposal_norm,
        "update_shift": update_shift,
        "divergence_drop": divergence_drop,
        "live": live,
    }
    if scale_proposals is not None:
        maps["scale_proposal_norm"] = scale_proposals.norm(dim=-1).to(torch.float64)
    return maps


def per_lag_latent_totals(
    model: Any, outputs: Mapping[str, torch.Tensor], contributing: torch.Tensor
) -> Dict[str, torch.Tensor]:
    r"""One batch's per-lag latent readouts, summed over the scored anchors the lag was live at.

    The batch sum of :func:`per_lag_latent_maps` under its ``live`` weight: ``<name>_sum`` is
    $\sum_{b,t} \mathrm{live}_{b,t,\ell}\, m_{b,t,\ell}$ for each map $m$. Sums rather than
    means, so a pass accumulates them with :func:`merge_counts` and divides once by the per-lag
    anchor count the exposure readout already carries.

    Args:
        model: The net, for $c_L$ and the two residual bounds.
        outputs: A forward's dict taken with ``return_proposals=True`` on the local fusion.
        contributing: The $0/1$ scored-anchor indicator, $(B, A)$.

    Returns:
        ``{'proposal_norm_sum', 'update_shift_sum', 'divergence_drop_sum'}`` and, where the arm
        has a scale channel, ``'scale_proposal_norm_sum'``, each $(L,)$ in float64.

    Raises:
        KeyError: If the forward was run without ``return_proposals``, naming the flag.
    """
    return latent_totals_from_maps(per_lag_latent_maps(model, outputs, contributing))


def latent_totals_from_maps(maps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sum :func:`per_lag_latent_maps` over the batch, under its own ``live`` weight.

    Split out so the collection pass, which needs the maps themselves for the sidecar, sums them
    once rather than recomputing every per-lag removal a second time.

    Args:
        maps: The maps, carrying ``live``.

    Returns:
        ``{'<name>_sum': (L,)}`` for every map but ``live``.
    """
    live = maps["live"]
    return {
        f"{name}_sum": (values * live).sum(dim=(0, 1))
        for name, values in maps.items()
        if name != "live"
    }


def per_segment_lag_profiles(maps: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    r"""Reduce :func:`per_lag_latent_maps` to one profile per segment, per map.

    Each segment's profile is the mean of the map over the scored anchors at which the lag was
    live, lag by lag:

    $$\bar m_{b,\ell} = \frac{\sum_t \mathrm{live}_{b,t,\ell}\, m_{b,t,\ell}}
                              {\sum_t \mathrm{live}_{b,t,\ell}},$$

    ``NaN`` where a segment had no live anchor at a lag -- a lag that was never read rather than
    a lag that was read and found empty, which the shape statistics keep apart by dropping the
    non-finite bin from the mass.

    Args:
        maps: The maps, carrying ``live``.

    Returns:
        ``{'proposal_norm', 'divergence_drop', ...}``, each $(B, L)$ in float64.
    """
    live = maps["live"]
    count = live.sum(dim=1)  # (B, L)
    profiles: Dict[str, torch.Tensor] = {}
    for name, values in maps.items():
        if name == "live":
            continue
        summed = (values * live).sum(dim=1)
        profiles[name] = torch.where(
            count > 0.0, summed / count.clamp_min(1.0), torch.full_like(summed, float("nan"))
        )
    return profiles


def lag_profile_summary(
    totals: Optional[Mapping[str, torch.Tensor]], anchors_per_lag: Optional[torch.Tensor]
) -> Dict[str, Any]:
    """Reduce the accumulated per-lag sums to per-lag means, missing where a lag had no support.

    Args:
        totals: The accumulated :func:`per_lag_latent_totals`, or ``None`` when nothing ran.
        anchors_per_lag: The per-lag live-anchor count from :func:`lag_exposure`, or ``None``.

    Returns:
        ``{quantity: [per-lag mean or MISSING]}`` with ``anchors_per_lag`` beside it, empty when
        nothing was accumulated.
    """
    if not totals or anchors_per_lag is None:
        return {}
    counts = anchors_per_lag.to(torch.float64).cpu()
    summary: Dict[str, Any] = {"anchors_per_lag": counts.tolist()}
    for name, total in totals.items():
        values = total.to(torch.float64).cpu()
        summary[name[: -len("_sum")]] = [
            MISSING if float(count) <= 0.0 else float(value) / float(count)
            for value, count in zip(values.tolist(), counts.tolist())
        ]
    return summary


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
    # The masks live on the model device because they are applied inside the forward; the
    # counts are accumulated on the CPU with every other per-batch record. This reduction is
    # the one place the two meet, so the mask follows the counts rather than the pass carrying
    # a second copy of the masks.
    return {
        band: {
            "anchors": float(anchors[mask.to(anchors.device)].sum()),
            "channels": float(channels[mask.to(channels.device)].sum()),
        }
        for band, mask in masks.items()
    }


__all__ = [
    "LAG_PROFILE_CHUNK",
    "MISSING",
    "SUPPRESSION_PREFIX",
    "band_exposure",
    "band_masks",
    "band_suppression_block",
    "bootstrap_curve",
    "cancellation_summary",
    "cancellation_totals",
    "channel_exposure",
    "lag_exposure",
    "lag_profile_summary",
    "merge_cancellation",
    "merge_counts",
    "paired_margin",
    "per_lag_latent_totals",
    "qualified_report",
]
