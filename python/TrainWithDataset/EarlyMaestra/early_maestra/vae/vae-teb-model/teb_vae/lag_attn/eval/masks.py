r"""The one place in the evaluation pipeline that constructs a mask.

Every reported number is a masked mean, so a mask that disagrees with the training loss by one
step is a number that disagrees with training and cannot be reconciled with it. Keeping the
construction in a single module is what makes the parity test meaningful: there is exactly one
definition to compare against ``compute_loss``, and an analysis that wants a different window
narrows this one rather than writing its own.

Two rules the rest of the package depends on.

**The feature mask mirrors ``compute_loss`` exactly**, including its shape. The trailing
singleton channel axis is not decoration -- it is what makes the denominator
$\mathrm{mask.sum()} \times C$ count entries rather than channels, and a mask broadcast to
$(B, T_{\mathrm{valid}}, H_d, C)$ instead would inflate every denominator by $C$ and silently
divide every loss by it.

**The KL support is read off the model, never rebuilt.** ``task.py`` documents why: under
``kld_support='anchor'`` the support additionally drops the final $H_d$ steps, and a local
copy that forgot to would average over exactly the anchors whose posterior is pulled to the
prior with nothing pulling back. The two would then disagree systematically, in the direction
that looks like a healthier model.

The lag-band helpers live here for the same reason -- a band keep-mask is a mask -- and because
the dead-anchor arithmetic they carry is the thing the lag ablation is most likely to get
wrong. When a kept band excludes lag $0$, every causally valid lag at anchors
$t < \min(\mathrm{band})$ is removed; the model forces lag $0$ back on there purely to keep
``entmax15`` well-posed, so those anchors carry a forecast that the ablation did not actually
ablate. Scoring must exclude them, and it must exclude the *same* anchors for every band or the
bands are not comparable.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch


def valid_anchor_range(model: Any, seq_len: int) -> Tuple[int, int]:
    r"""Return the supervised anchor range $[\mathrm{warmup},\ T - H_d)$ as a half-open pair.

    Anchors below the warm-up have encoder states conditioned on almost nothing; anchors at or
    above $T - H_d$ have no fully observed forecast window. Both are excluded from every
    feature-space number the pipeline reports.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        seq_len: Sequence length $T$.

    Returns:
        ``(start, stop)``, half-open. ``stop <= start`` means no anchor is supervised, which a
        caller should treat as an empty analysis rather than as an error.
    """
    horizon = int(model.horizon)
    stop = max(int(seq_len) - horizon, 0)
    start = min(int(model._warmup_steps(int(seq_len))), stop)
    return start, stop


def feature_mask(
    model: Any,
    weight: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Build the feature-loss mask, $(B,\ T - H_d,\ H_d,\ 1)$.

    Elementwise identical to the mask ``compute_loss`` builds internally:

    $$m = \mathbb{1}[t \ge \mathrm{warmup}] \cdot w_{\mathrm{anchor}} \cdot w_{\mathrm{target}}$$

    where $w_{\mathrm{anchor}}$ is ``weight[:, :T-H_d]`` and $w_{\mathrm{target}}$ is
    ``weight[:, 1:]`` unfolded over the horizon -- an entry counts only if both its anchor and
    every step of its forecast target are valid.

    ``batch_size`` is required rather than inferred, because with ``weight=None`` there is no
    tensor to read it from and the mask must still expand to a real batch: an analysis that
    divided by a $(1, T-H_d, H_d, 1)$ sum would under-count by a factor of $B$.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        weight: Per-step validity $(B, T)$, or ``None`` when the batch carries none.
        batch_size: Batch size $B$.
        seq_len: Sequence length $T$.
        device: Device for the mask. Defaults to ``weight``'s, else CPU.
        dtype: Floating dtype. fp32 throughout the pipeline.

    Returns:
        The mask, $(B,\ T - H_d,\ H_d,\ 1)$.
    """
    horizon = int(model.horizon)
    valid_steps = max(int(seq_len) - horizon, 0)
    if device is None:
        device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")

    warmup = int(model._warmup_steps(int(seq_len)))
    warmup_t = torch.zeros(valid_steps, dtype=dtype, device=device)
    if warmup < valid_steps:
        warmup_t[warmup:] = 1.0

    if weight is None:
        return warmup_t[None, :, None, None].expand(int(batch_size), valid_steps, horizon, 1)

    step_weight = weight.to(device=device, dtype=dtype)
    anchor_weight = step_weight[:, :valid_steps]
    target_weight = step_weight[:, 1:].unfold(dimension=1, size=horizon, step=1)
    return (
        warmup_t[None, :, None, None]
        * anchor_weight[:, :, None, None]
        * target_weight[:, :, :, None]
    )


def kld_support(
    model: Any,
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Return the KL time support $(T,)$, delegating to the model.

    A thin wrapper on purpose. The delegation is the point: ``kld_support='anchor'`` drops the
    final $H_d$ steps as well as the warm-up prefix, and a reimplementation that tracked only
    the warm-up would read systematically low against ``kld_raw`` while looking correct.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        seq_len: Sequence length $T$.
        device: Device for the mask.
        dtype: Floating dtype.

    Returns:
        $1.0$ in support and $0.0$ outside, $(T,)$.
    """
    return model._kld_support_mask(int(seq_len), device=device, dtype=dtype)


def kld_mask(
    model: Any,
    weight: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Return the full KL mask $(B, T)$: the time support intersected with ``weight``.

    The same composition ``_kld_loss`` performs before reducing, exposed so an analysis can
    reduce the per-dimension KL itself -- per sample, per step, per head -- and still land on
    ``kld_raw`` when it reduces the whole thing.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        weight: Per-step validity $(B, T)$, or ``None``.
        batch_size: Batch size $B$.
        seq_len: Sequence length $T$.
        device: Device for the mask. Defaults to ``weight``'s, else CPU.
        dtype: Floating dtype.

    Returns:
        The mask, $(B, T)$.
    """
    if device is None:
        device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
    support = kld_support(model, int(seq_len), device=device, dtype=dtype)
    mask = support.unsqueeze(0).expand(int(batch_size), int(seq_len))
    if weight is not None:
        mask = mask * weight.to(device=device, dtype=dtype)
    return mask


# ---------------------------------------------------------------------------
# Lag bands
# ---------------------------------------------------------------------------
def lag_band_keep_mask(
    band: Sequence[int], num_lags: int, *, device: Optional[torch.device] = None
) -> torch.Tensor:
    r"""Build the boolean keep-mask for one inclusive lag band, $(L,)$.

    Index $\ell$ is lag $\ell$, matching ``LagAttention.build_lag_mask``'s lag ordering, so the
    result can be handed straight to ``forward(..., lag_band_mask=...)``.

    Args:
        band: Inclusive ``(lo, hi)`` pair in model-lag units.
        num_lags: The window width $L = \mathrm{max\_lag} + 1$.
        device: Device for the mask.

    Returns:
        ``True`` at the kept lags, $(L,)$ bool.

    Raises:
        ValueError: If the band is not a pair, falls outside $[0, L)$, or is empty. An empty
            band masks every lag at every anchor, and ``entmax15`` -- which the shipped config
            enables -- raises on a zero-support row rather than degrading like ``softmax``.
    """
    if len(band) != 2:
        raise ValueError(f"a lag band must be an inclusive (lo, hi) pair, got {band!r}.")
    low, high = int(band[0]), int(band[1])
    if low > high:
        raise ValueError(
            f"lag band ({low}, {high}) is empty (lo > hi). An empty band removes every lag at "
            f"every anchor, leaving the attention with no valid support."
        )
    if low < 0 or high >= int(num_lags):
        raise ValueError(
            f"lag band ({low}, {high}) falls outside the model's window [0, {int(num_lags) - 1}] "
            f"(L = {int(num_lags)})."
        )
    mask = torch.zeros(int(num_lags), dtype=torch.bool, device=device)
    mask[low : high + 1] = True
    return mask


def dead_before(band: Sequence[int]) -> int:
    r"""Return the first anchor at which a band's ablation is real, $\min(\mathrm{band})$.

    At anchors $t < \min(\mathrm{band})$ every causally valid lag is outside the band, so the
    model forces lag $0$ back on to keep the attention activation well-posed. Those anchors ran
    with a source the ablation did not remove, and scoring them would dilute the band's effect
    toward zero -- most severely for the long-lag bands, which is exactly the comparison the
    ablation exists to make.

    Args:
        band: Inclusive ``(lo, hi)`` pair.

    Returns:
        The first anchor at which every kept lag is causally available.
    """
    return int(band[0])


def common_scoring_start(model: Any, bands: Dict[str, Sequence[int]], seq_len: int) -> int:
    r"""Return the first anchor scorable under **every** band, plus the warm-up.

    $$t_0 = \max\left(\mathrm{warmup},\ \max_{b} \min(b)\right)$$

    One support shared by all bands, not a per-band one. Per-band supports would score each band
    over a different anchor set, so a band's number would reflect both its ablation *and* the
    anchors it happened to be scored on, and the difference between two bands would confound the
    two. Sharing the strictest support costs anchors and buys comparability.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        bands: Band name to inclusive ``(lo, hi)`` pair. Empty means the warm-up alone.
        seq_len: Sequence length $T$.

    Returns:
        The first scorable anchor. May equal the end of the anchor range when a band starts
        beyond it, which the caller should read as "no anchor is common to every band".
    """
    start, _ = valid_anchor_range(model, int(seq_len))
    if not bands:
        return start
    return max(start, max(dead_before(band) for band in bands.values()))


def anchor_slice_mask(
    mask_feat: torch.Tensor, start: int, stop: Optional[int] = None
) -> torch.Tensor:
    """Zero a feature mask outside ``[start, stop)`` on the anchor axis.

    Narrowing rather than slicing keeps the tensor's shape, so a narrowed mask still multiplies
    a full ``(B, T-H_d, H_d, C)`` error tensor and the two cannot fall out of alignment.

    Args:
        mask_feat: A feature mask from :func:`feature_mask`.
        start: First anchor to keep.
        stop: One past the last anchor to keep. ``None`` keeps through the end.

    Returns:
        A new mask, zeroed outside the range.
    """
    narrowed = torch.zeros_like(mask_feat)
    n_anchors = int(mask_feat.shape[1])
    lo = max(int(start), 0)
    hi = n_anchors if stop is None else min(int(stop), n_anchors)
    if lo < hi:
        narrowed[:, lo:hi] = mask_feat[:, lo:hi]
    return narrowed


def live_anchor_mask(
    attn_weights: torch.Tensor, *, tolerance: float = 1e-4
) -> torch.Tensor:
    r"""Return the anchors whose attention rows actually carry mass, $(B, T)$ bool.

    An anchor is *dead* when the model zeroed its attention rather than normalising them --
    ``_ablate_dead_anchors`` sets $\alpha_{t} = 0$ wherever a band mask left no causally valid
    lag, deliberately without renormalising. A dead row therefore sums to $0$ while $K_t$ stays
    positive, and the attribution identity $\sum_\ell \widetilde{TE}_{t,\ell} = K_t$ does not
    hold there.

    That makes this the mask every lag-resolved readout must intersect with before averaging.
    Averaging a dead anchor into a per-sample lag profile mixes an all-zero row into a set of
    distributions, which drags the profile toward zero in proportion to how many anchors the
    band killed -- worst for exactly the long-lag bands an ablation most wants to compare.

    Every head is required to be live, not merely one. At a genuinely dead anchor all heads are
    zeroed together, so the two agree there; requiring all of them additionally catches a row
    the ``entmax15`` NaN guard zeroed on its own, which happens per head rather than per anchor.

    Args:
        attn_weights: The forward's ``attn_weights``, $(B, T, M, L)$, in lag order.
        tolerance: How far a row sum may fall below $1$ and still count as live. Loose enough to
            absorb fp32 summation error over $L = 91$ lags, far tighter than the gap to $0$.

    Returns:
        ``True`` at anchors where every head's row sums to $1$, $(B, T)$.
    """
    row_sums = attn_weights.sum(dim=-1)
    return (row_sums >= 1.0 - float(tolerance)).all(dim=-1)


def lag_readout_support(
    model: Any,
    attn_weights: torch.Tensor,
    weight: Optional[torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""The anchor support every lag-resolved readout shares, $(B, T)$.

    Three conditions intersected, each excluding anchors for a different reason:

    * the **KL support**, because $K_t$ is only defined where the model reduces it -- and under
      ``kld_support='anchor'`` that drops the trailing $H_d$ steps as well as the warm-up;
    * the per-step validity **weight**, because an anchor over a gap in the recording carries an
      attention row fitted to interpolated nothing;
    * **liveness**, because a band-ablated anchor's row was zeroed rather than renormalised.

    Sharing one definition between the attention diagnostics and the ``te_lag_map`` analysis is
    what makes their numbers reconcile: the identity $\sum_\ell \widetilde{TE}_{t,\ell} = K_t$
    is checked on this support, so an attention profile averaged over a different one would
    describe a different set of anchors from the KL it is supposed to attribute.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        attn_weights: The forward's ``attn_weights``, $(B, T, M, L)$.
        weight: Per-step validity $(B, T)$, or ``None``.
        dtype: Floating dtype.

    Returns:
        $1.0$ where the anchor is usable and $0.0$ elsewhere, $(B, T)$.
    """
    batch_size, seq_len = int(attn_weights.shape[0]), int(attn_weights.shape[1])
    support = kld_mask(
        model, weight, batch_size, seq_len, device=attn_weights.device, dtype=dtype
    )
    return support * live_anchor_mask(attn_weights).to(dtype)


def band_exclusion_counts(
    model: Any, bands: Dict[str, Sequence[int]], seq_len: int
) -> Dict[str, Dict[str, int]]:
    r"""Per band, how many supervised anchors the common scoring support gives up.

    The lag ablation scores every band on one shared anchor set -- see
    :func:`common_scoring_start` -- so a band whose own ablation becomes real early still gets
    scored only from the latest band's first real anchor onward. That is the price of
    comparability, and it must be *visible*: a band table that reported only losses would hide
    the fact that a long-lag band forced thirty anchors off the front of every other band.

    Args:
        model: The rebuilt ``SeqVaeLagAttn``.
        bands: Band name to inclusive ``(lo, hi)`` pair.
        seq_len: Sequence length $T$.

    Returns:
        Per band: ``dead_before`` (its own first real anchor), ``excluded_by_common_support``
        (anchors it could have scored but gives up to the shared support), and ``n_scored``.

    Raises:
        ValueError: If the common support starts at or beyond the end of the supervised anchor
            range, which leaves every band with nothing to score. Raised rather than returned as
            a zero count, because the resulting table would be a page of ``NaN`` that reads as a
            broken analysis rather than as a misconfigured band set.
    """
    start, stop = valid_anchor_range(model, int(seq_len))
    common = common_scoring_start(model, bands, int(seq_len))
    if common >= stop:
        raise ValueError(
            f"the common scoring support is empty: it starts at anchor {common} but the "
            f"supervised anchor range is [{start}, {stop}). The band with the largest minimum "
            f"lag decides this -- "
            f"{max(bands.items(), key=lambda item: dead_before(item[1]))[0] if bands else '(none)'}"
            f" starts at lag {max((dead_before(band) for band in bands.values()), default=0)}, "
            f"which exceeds what a sequence of length {int(seq_len)} can supply. Narrow the "
            f"bands in eval_config.bands, or evaluate a checkpoint with a shorter max_lag."
        )
    return {
        name: {
            "dead_before": dead_before(band),
            "excluded_by_common_support": max(common - max(dead_before(band), start), 0),
            "n_scored": stop - common,
        }
        for name, band in bands.items()
    }


def subsample_indices(
    n_total: int, cap: Optional[int], seed: int, *, groups: Optional[Iterable[Any]] = None
) -> Optional[torch.Tensor]:
    r"""Draw a seeded index subsample over the **whole** index space.

    Never a prefix. The test loader is built ``shuffle=False`` over eight concatenated
    per-subgroup files, so a prefix cap draws file $0$ alone -- one subgroup, one clinical
    class -- which is the predecessor's "only 1 class found" failure arriving by a second route.
    Its only recorded workaround was "do not use a cap".

    When ``groups`` is supplied the draw is stratified: each group receives a share of the cap
    proportional to its size, with the remainder going to the largest groups. That guarantees
    every file appears whenever the cap is at least the group count, rather than merely making
    it overwhelmingly likely.

    Args:
        n_total: Size of the index space.
        cap: Maximum indices to draw. ``None`` or a cap at least ``n_total`` returns ``None``,
            meaning "take everything" -- which callers can test for without materialising an
            arange.
        seed: Seed, so a rerun draws the same subsample.
        groups: Optional per-index group key -- typically the source file basename -- to
            stratify over.

    Returns:
        Sorted indices to retain, or ``None`` when nothing is dropped. Sorted so a capped pass
        still visits the loader in its natural order.
    """
    if cap is None or int(cap) >= int(n_total):
        return None
    cap = int(cap)
    generator = torch.Generator().manual_seed(int(seed))

    if groups is None:
        drawn = torch.randperm(int(n_total), generator=generator)[:cap]
        return torch.sort(drawn).values

    by_group: Dict[Any, list] = {}
    for index, key in enumerate(groups):
        by_group.setdefault(key, []).append(index)

    # Largest group first: the proportional split's remainder then lands where it costs the least
    # relative coverage, and a cap smaller than the group count still resolves deterministically
    # rather than depending on dict ordering.
    ordered = sorted(by_group.items(), key=lambda item: (-len(item[1]), str(item[0])))
    quotas = _allocate_quotas([len(members) for _, members in ordered], cap)

    picked: list = []
    for (_, members), take in zip(ordered, quotas):
        if take <= 0:
            continue
        member_index = torch.tensor(members, dtype=torch.long)
        chosen = member_index[torch.randperm(len(members), generator=generator)[:take]]
        picked.extend(int(value) for value in chosen.tolist())
    return torch.sort(torch.tensor(picked, dtype=torch.long)).values


def _allocate_quotas(sizes: Sequence[int], cap: int) -> list:
    """Split ``cap`` across groups in proportion to ``sizes``, giving every group at least one.

    The "at least one" floor is what turns a stratified cap into a coverage guarantee: without
    it a small shard rounds to a quota of zero and disappears from the draw entirely, which is
    the failure mode the stratification exists to prevent. When the cap is smaller than the
    group count the floor cannot hold for everyone, and the largest groups keep their slot.

    The floor is allocated **first** and never given back. An earlier form applied the floor
    inside a proportional expression and then repaired the overshoot by trimming the smallest
    groups, which undid the floor on precisely the groups it exists to protect:
    ``sizes=[500, 200, 100, 50, 20, 10, 5, 3], cap=8`` produced ``[4, 1, 1, 1, 1, 0, 0, 0]``,
    dropping three shards at exactly the cap the docstring promised full coverage for. On the
    shipped eight-subgroup k-fold split those are the rare clinical shards -- ``hie_cs``,
    ``acidosis_cs`` -- so the qualitative pages silently came only from the common subgroups.

    Args:
        sizes: Group sizes, largest first.
        cap: Total indices to draw. Assumed no larger than ``sum(sizes)``.

    Returns:
        Per-group quotas, in ``sizes`` order, summing to ``cap``. Every group with at least one
        member receives at least one whenever ``cap >= len(sizes)``.
    """
    count = len(sizes)
    # Floor pass: one index per group while the cap lasts. ``sizes`` arrives largest first, so a
    # cap below the group count deterministically keeps the largest groups rather than depending
    # on iteration order.
    quotas = [1 if position < min(cap, count) else 0 for position in range(count)]
    remaining = cap - sum(quotas)

    # Remainder pass: distribute what the floor did not consume in proportion to each group's
    # *unclaimed* members, so a group can never be pushed above its own size and the floor is
    # never a candidate for trimming.
    if remaining > 0:
        headroom = [sizes[position] - quotas[position] for position in range(count)]
        total_headroom = sum(headroom)
        if total_headroom > 0:
            for position in range(count):
                if remaining <= 0:
                    break
                share = min(headroom[position], remaining * headroom[position] // total_headroom)
                quotas[position] += share
                remaining -= share
        # The flooring division leaves a handful unallocated; place them largest-first.
        for position in range(count):
            if remaining <= 0:
                break
            take = min(sizes[position] - quotas[position], remaining)
            quotas[position] += take
            remaining -= take
    return quotas
