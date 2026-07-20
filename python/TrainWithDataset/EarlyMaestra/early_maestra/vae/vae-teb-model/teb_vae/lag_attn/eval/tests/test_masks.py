"""Tests for the single mask builder.

The mask's *numerical* agreement with ``compute_loss`` is asserted in ``test_parity.py``, which
is where the load-bearing check lives. What is asserted here is the structure the rest of the
package relies on -- the shape, the warm-up and anchor boundaries, the weight product, and the
lag-band and dead-anchor arithmetic the ablation is built on.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.eval import masks
from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import SEQ_LEN, TINY_KWARGS


@pytest.fixture
def model() -> SeqVaeLagAttn:
    """A tiny model under the default ``'full'`` KL support."""
    torch.manual_seed(0)
    return SeqVaeLagAttn(**TINY_KWARGS)


@pytest.fixture
def anchor_model() -> SeqVaeLagAttn:
    """A tiny model under ``kld_support='anchor'``."""
    torch.manual_seed(0)
    return SeqVaeLagAttn(**dict(TINY_KWARGS, kld_support="anchor"))


# ---------------------------------------------------------------------------
# Feature mask
# ---------------------------------------------------------------------------
def test_feature_mask_has_the_shape_compute_loss_builds(model: SeqVaeLagAttn) -> None:
    """The trailing singleton channel axis is what makes the denominator count entries."""
    batch, horizon = 3, int(model.horizon)
    mask = masks.feature_mask(model, None, batch, SEQ_LEN)
    assert mask.shape == (batch, SEQ_LEN - horizon, horizon, 1)


def test_feature_mask_expands_to_the_batch_when_weight_is_absent(model: SeqVaeLagAttn) -> None:
    """With no ``weight`` there is no tensor to read $B$ from, so it must come from the argument.

    A mask left at $B = 1$ would make every per-sample denominator short by a factor of $B$.
    """
    batch = 5
    mask = masks.feature_mask(model, None, batch, SEQ_LEN)
    assert int(mask.shape[0]) == batch
    assert float(mask.sum()) == pytest.approx(
        batch * (SEQ_LEN - model.horizon - model.warmup_period) * model.horizon
    )


def test_feature_mask_zeroes_the_warmup_prefix_and_nothing_after_it(model: SeqVaeLagAttn) -> None:
    """The boundary is at ``warmup`` exactly -- off by one either way changes every loss."""
    warmup = int(model.warmup_period)
    mask = masks.feature_mask(model, None, 2, SEQ_LEN)
    assert float(mask[:, :warmup].sum()) == 0.0
    assert bool((mask[:, warmup:] == 1.0).all())


def test_feature_mask_requires_both_the_anchor_and_its_whole_target_window(
    model: SeqVaeLagAttn,
) -> None:
    """An entry counts only if its anchor *and* every step of its forecast target are valid.

    Zeroing one step of ``weight`` must therefore knock out that step as an anchor and as a
    target of each of the ``horizon`` anchors that forecast it.
    """
    batch, horizon = 2, int(model.horizon)
    weight = torch.ones(batch, SEQ_LEN)
    dead_step = SEQ_LEN - 2
    weight[:, dead_step] = 0.0

    mask = masks.feature_mask(model, weight, batch, SEQ_LEN)
    full = masks.feature_mask(model, None, batch, SEQ_LEN)

    # The dead step is a target of anchors t where t + 1 <= dead_step <= t + horizon, in every
    # sample of the batch -- the weight was zeroed for all of them.
    lost = float(full.sum() - mask.sum())
    expected_lost = batch * sum(
        1
        for anchor in range(int(model.warmup_period), SEQ_LEN - horizon)
        for h in range(horizon)
        if anchor + 1 + h == dead_step
    )
    assert lost == pytest.approx(expected_lost)


def test_feature_mask_is_multiplicative_in_a_fractional_weight(model: SeqVaeLagAttn) -> None:
    """A fractional ``weight`` produces a weighted mask, not a thresholded one.

    Whether production ``weight`` is ever fractional is an open question the loader probe
    answers; the mask must be right either way.
    """
    batch = 2
    weight = torch.full((batch, SEQ_LEN), 0.5)
    mask = masks.feature_mask(model, weight, batch, SEQ_LEN)
    binary = masks.feature_mask(model, torch.ones(batch, SEQ_LEN), batch, SEQ_LEN)
    # Both the anchor and the target factor contribute 0.5.
    assert torch.allclose(mask, binary * 0.25)


# ---------------------------------------------------------------------------
# KL support
# ---------------------------------------------------------------------------
def test_kld_support_is_the_models_own_under_full(model: SeqVaeLagAttn) -> None:
    """Delegation, not reimplementation -- asserted by identity against the model."""
    support = masks.kld_support(model, SEQ_LEN)
    assert torch.equal(support, model._kld_support_mask(SEQ_LEN))
    assert float(support[: model.warmup_period].sum()) == 0.0
    assert bool((support[model.warmup_period :] == 1.0).all())


def test_kld_support_drops_the_untrained_tail_under_anchor(anchor_model: SeqVaeLagAttn) -> None:
    """``'anchor'`` additionally masks the final $H_d$ steps.

    Rebuilding the support locally and forgetting this is the documented way the reported KL and
    ``kld_raw`` diverge, in the direction that looks like a healthier model.
    """
    support = masks.kld_support(anchor_model, SEQ_LEN)
    horizon = int(anchor_model.horizon)
    assert float(support[-horizon:].sum()) == 0.0
    assert float(support.sum()) == pytest.approx(
        SEQ_LEN - anchor_model.warmup_period - horizon
    )


def test_kld_mask_intersects_the_support_with_weight(anchor_model: SeqVaeLagAttn) -> None:
    """The composition ``_kld_loss`` performs before reducing."""
    batch = 3
    weight = torch.ones(batch, SEQ_LEN)
    weight[0, :] = 0.0
    mask = masks.kld_mask(anchor_model, weight, batch, SEQ_LEN)
    assert mask.shape == (batch, SEQ_LEN)
    assert float(mask[0].sum()) == 0.0
    assert float(mask[1].sum()) == pytest.approx(
        float(masks.kld_support(anchor_model, SEQ_LEN).sum())
    )


# ---------------------------------------------------------------------------
# Anchor range
# ---------------------------------------------------------------------------
def test_valid_anchor_range_is_the_supervised_window(model: SeqVaeLagAttn) -> None:
    """$[\\mathrm{warmup},\\ T - H_d)$, half open."""
    start, stop = masks.valid_anchor_range(model, SEQ_LEN)
    assert (start, stop) == (int(model.warmup_period), SEQ_LEN - int(model.horizon))


def test_valid_anchor_range_collapses_rather_than_inverting_on_a_short_sequence(
    model: SeqVaeLagAttn,
) -> None:
    """A sequence shorter than the horizon must give an empty range, not a negative one."""
    start, stop = masks.valid_anchor_range(model, int(model.horizon) - 1)
    assert start == stop == 0


# ---------------------------------------------------------------------------
# Lag bands
# ---------------------------------------------------------------------------
def test_lag_band_keep_mask_keeps_exactly_the_inclusive_span() -> None:
    """Index $\\ell$ is lag $\\ell$, matching the attention's lag ordering."""
    mask = masks.lag_band_keep_mask((2, 5), 9)
    assert mask.dtype == torch.bool
    assert mask.shape == (9,)
    assert mask.nonzero().flatten().tolist() == [2, 3, 4, 5]


def test_lag_band_keep_mask_handles_a_lag_zero_inclusive_band() -> None:
    """A band including lag $0$ has no dead anchors -- lag $0$ is causally valid everywhere."""
    mask = masks.lag_band_keep_mask((0, 3), 9)
    assert bool(mask[0])
    assert masks.dead_before((0, 3)) == 0


def test_lag_band_keep_mask_handles_a_long_lag_only_band() -> None:
    """A band excluding lag $0$ is exactly the case that creates dead anchors."""
    mask = masks.lag_band_keep_mask((6, 8), 9)
    assert not bool(mask[0])
    assert masks.dead_before((6, 8)) == 6


def test_an_empty_band_raises_rather_than_producing_a_zero_support_row() -> None:
    """``entmax15`` raises on a zero-support row rather than degrading like ``softmax``."""
    with pytest.raises(ValueError, match="empty"):
        masks.lag_band_keep_mask((5, 2), 9)


def test_a_band_beyond_the_window_raises() -> None:
    """Naming the real window, so the message is actionable."""
    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        masks.lag_band_keep_mask((3, 9), 9)


def test_common_scoring_start_is_the_strictest_band_plus_the_warmup(
    model: SeqVaeLagAttn,
) -> None:
    """One support for every band, so a band difference is not confounded with its anchor set."""
    bands = {"near": (0, 3), "far": (6, 8)}
    start = masks.common_scoring_start(model, bands, SEQ_LEN)
    assert start == max(int(model.warmup_period), 6)


def test_common_scoring_start_falls_back_to_the_warmup_with_no_bands(
    model: SeqVaeLagAttn,
) -> None:
    """No bands means no dead anchors to exclude."""
    assert masks.common_scoring_start(model, {}, SEQ_LEN) == int(model.warmup_period)


def test_anchor_slice_mask_narrows_without_reshaping(model: SeqVaeLagAttn) -> None:
    """Narrowing rather than slicing keeps the mask aligned with a full-width error tensor."""
    mask = masks.feature_mask(model, None, 2, SEQ_LEN)
    narrowed = masks.anchor_slice_mask(mask, 5)
    assert narrowed.shape == mask.shape
    assert float(narrowed[:, :5].sum()) == 0.0
    assert torch.equal(narrowed[:, 5:], mask[:, 5:])


# ---------------------------------------------------------------------------
# Capped draws
# ---------------------------------------------------------------------------
def test_no_cap_returns_none_rather_than_an_arange() -> None:
    """``None`` lets a caller test for "keep everything" without materialising an index."""
    assert masks.subsample_indices(100, None, seed=0) is None
    assert masks.subsample_indices(100, 100, seed=0) is None
    assert masks.subsample_indices(100, 500, seed=0) is None


def test_a_cap_draws_across_the_whole_index_space_not_a_prefix() -> None:
    """The assertion that would fail under prefix truncation.

    A prefix cap over the eight concatenated per-subgroup test shards draws file $0$ alone --
    one subgroup and one clinical class -- which is the predecessor's worst bug by a second
    route.
    """
    drawn = masks.subsample_indices(1000, 50, seed=7)
    assert drawn is not None
    assert len(drawn) == 50
    # A prefix draw would have max < 50; a whole-space draw reaches the far end.
    assert int(drawn.max()) > 500


def test_a_capped_draw_is_seeded_and_sorted() -> None:
    """A rerun with the same seed must retain the same samples, in loader order."""
    first = masks.subsample_indices(500, 40, seed=11)
    second = masks.subsample_indices(500, 40, seed=11)
    other = masks.subsample_indices(500, 40, seed=12)
    assert first is not None and second is not None and other is not None
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.equal(first, torch.sort(first).values)


def test_a_stratified_cap_reaches_every_group() -> None:
    """Stratification upgrades "very probably covers every file" into a guarantee."""
    # Eight groups of wildly unequal size, as the k-fold subgroup shards are.
    groups = ["a"] * 500 + ["b"] * 200 + ["c"] * 100 + ["d"] * 50 + ["e"] * 20 + [
        "f"
    ] * 10 + ["g"] * 5 + ["h"] * 3
    drawn = masks.subsample_indices(len(groups), 60, seed=3, groups=groups)
    assert drawn is not None
    assert len(drawn) == 60
    covered = {groups[int(index)] for index in drawn.tolist()}
    assert covered == set(groups), f"a stratified draw missed {set(groups) - covered}"


def test_a_stratified_cap_at_the_group_count_still_reaches_every_group() -> None:
    """The boundary the shipped config actually sits on: ``caps.samples: 8`` over eight shards.

    A generous cap covers every group even with a broken allocator, so the guarantee has to be
    tested where it is tight. An earlier allocator applied the per-group floor inside the
    proportional expression and then repaired the overshoot by trimming the *smallest* groups,
    returning ``[4, 1, 1, 1, 1, 0, 0, 0]`` here -- three shards silently absent from the draw,
    and the three rarest at that.
    """
    sizes = {"a": 500, "b": 200, "c": 100, "d": 50, "e": 20, "f": 10, "g": 5, "h": 3}
    groups = [name for name, size in sizes.items() for _ in range(size)]
    for cap in (len(sizes), len(sizes) + 1, 2 * len(sizes)):
        drawn = masks.subsample_indices(len(groups), cap, seed=3, groups=groups)
        assert drawn is not None
        assert len(drawn) == cap
        covered = {groups[int(index)] for index in drawn.tolist()}
        assert covered == set(sizes), (
            f"a stratified draw at cap={cap} over {len(sizes)} groups missed "
            f"{set(sizes) - covered}"
        )


def test_a_stratified_cap_stays_within_every_group_and_hits_the_cap_exactly() -> None:
    """The quota split must not over-draw a small group or come up short overall."""
    groups = ["a"] * 10 + ["b"] * 3
    drawn = masks.subsample_indices(13, 9, seed=5, groups=groups)
    assert drawn is not None
    assert len(drawn) == 9
    counts = {"a": 0, "b": 0}
    for index in drawn.tolist():
        counts[groups[int(index)]] += 1
    assert counts["a"] <= 10 and counts["b"] <= 3
    assert counts["a"] + counts["b"] == 9


# ---------------------------------------------------------------------------
# Band exclusion accounting
# ---------------------------------------------------------------------------
def test_the_common_support_is_the_max_over_bands_including_a_long_lag_only_one(model) -> None:
    """One band starting late costs every other band the same anchors, and that must be visible.

    The tiny geometry is ``warmup_period=2``, ``max_lag=8``, $T = 16$, $H_d = 4$, so the
    supervised range is $[2, 12)$ and a band starting at lag $6$ pushes every band's first
    scorable anchor to $6$.
    """
    bands = {"early": (0, 2), "mid": (3, 5), "late": (6, 8)}
    assert masks.common_scoring_start(model, bands, SEQ_LEN) == 6

    counts = masks.band_exclusion_counts(model, bands, SEQ_LEN)
    assert set(counts) == set(bands)
    # `early` could have scored from the warm-up at anchor 2 and gives up four anchors to the
    # shared support; `late` was never going to score before 6 and gives up none.
    assert counts["early"]["excluded_by_common_support"] == 4
    assert counts["mid"]["excluded_by_common_support"] == 3
    assert counts["late"]["excluded_by_common_support"] == 0
    assert counts["early"]["dead_before"] == 0 and counts["late"]["dead_before"] == 6


def test_every_band_is_scored_on_an_identical_anchor_count(model) -> None:
    """The whole point of a shared support: a band's number reflects its ablation, not its window."""
    counts = masks.band_exclusion_counts(
        model, {"a": (0, 1), "b": (2, 4), "c": (5, 8)}, SEQ_LEN
    )
    assert len({record["n_scored"] for record in counts.values()}) == 1


def test_a_band_starting_beyond_the_anchor_range_raises_rather_than_scoring_nothing(
    model,
) -> None:
    """An empty support would emit a page of NaN that reads as a broken analysis."""
    # The supervised range ends at T - H_d = 12, so a band whose minimum lag is 12 leaves nothing.
    with pytest.raises(ValueError, match="common scoring support is empty"):
        masks.band_exclusion_counts(model, {"far": (12, 12)}, SEQ_LEN)


def test_the_warmup_still_binds_when_every_band_starts_at_lag_zero(model) -> None:
    """With no band excluding lag 0 the support is the plain warm-up boundary."""
    counts = masks.band_exclusion_counts(model, {"a": (0, 3), "b": (0, 8)}, SEQ_LEN)
    warmup, stop = masks.valid_anchor_range(model, SEQ_LEN)
    assert all(record["excluded_by_common_support"] == 0 for record in counts.values())
    assert all(record["n_scored"] == stop - warmup for record in counts.values())
