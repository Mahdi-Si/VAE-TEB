r"""The seven added readouts, and the eighth that is not a reduction of anything already in hand.

Almost all of them are partial sums or fractions of quantities the objective already computes, so
what is checked is exactly that: that each is the partial sum it claims to be, over the denominator
it claims, of the number it is printed beside. A readout that recomposes to nothing is four
unrelated numbers with suggestive names.

Two of the seven are **guards** rather than results and are tested as such. ``target_warm_frac``
must read exactly $1.0$ and ``anchors_per_sample`` must sit at its geometry-derived value; a row
outside either means the geometry broke, not that the model learned something. Both are asserted
against values re-derived here from the geometry rather than against literals, so a horizon or
budget change moves the expectation with the model.

``kld_source_null`` is the exception, and the one whose *design* is the claim. The source
availability pattern is a deterministic function of $t$ and identical in every row of the batch, so
it enters $q(z \mid Y, U)$ and not $p(z \mid Y)$, and no permutation of rows can remove it -- which
is why the shuffle control cannot see it and why this arm exists. The property that makes it a
different control is asserted directly: it does not move under a derangement of the source.
"""
from __future__ import annotations

import math

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import (
    WARM_BLOCK_FRACTION,
    WARM_TERTILES,
    CausalFeatureForecastTarget,
)
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    CAUSAL_C_U,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    SHIPPED_WARMUP_PERIOD,
    TINY_STRIDE,
    build,
    make_streams,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)
from teb_vae.lag_attn_rws.nets import controls

#: The two per-block source-warmth columns, and the three warm-up tertiles, named once.
_WARMTH_KEYS = ("source_lag_warmth_frac_st", "source_lag_warmth_frac_ph")
_TERTILE_KEYS = ("pred_gap_warm_lo", "pred_gap_warm_mid", "pred_gap_warm_hi")

#: The tolerance the two channel-axis splits are compared against each other at. Both are reduced
#: from the same elementwise term over the same mask and the same denominator, so they agree to
#: float32 rounding on the *difference*, which is what this pins.
_SPLIT_TOL = 1e-6

#: Why the recomposition against ``pred_gap`` itself carries a looser bound. ``pred_gap`` is
#: computed as ``nll_base_block - nll_full_block``, a difference of two sums over $H \cdot C$
#: coefficients that run to $10^{3}$ nats, so it loses roughly five decimal digits to
#: cancellation before either split is even formed. The splits difference the two branches
#: *elementwise* and therefore do not. The bound below is float32's own epsilon against the
#: magnitude of the operands that cancel, with slack -- not a tolerance chosen to make a test pass.
_CANCELLATION_EPS = 2.0e-6


def _weight(model, batch: int = BATCH, value: float = 1.0) -> torch.Tensor:
    """A uniform decimated weight at a model's own sequence length."""
    return torch.full((batch, model.geometry.t), float(value))


def _tiled(stride: int = TINY_STRIDE, **overrides):
    """The tiny model at a tiling, its three input tensors, and its concatenated target stream."""
    kwargs = tiny_warmup_kwargs(anchor_stride=stride, **overrides)
    model = build(kwargs).eval()
    y_st, y_ph, u_stream = make_streams(kwargs)
    return model, (y_st, y_ph, u_stream), torch.cat([y_st, y_ph], dim=-1)


def _metrics(model, streams, features, phase, *, weight=None, likelihood="mse"):
    """One forward and its full metric dict."""
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(*streams, phase)
    weight = _weight(model) if weight is None else weight
    return out, model.compute_loss(
        out, features, weight=weight, likelihood=likelihood
    )["metrics"]


def _synthetic_gaps(model, phase, stride, batch: int):
    """The added readouts alone, over a zero-valued forward at a real geometry.

    The two geometry guards are functions of the resolved budget and the decoded anchor set and of
    nothing the network computes, so they can be exercised at the **production** geometry without
    paying for a production forward -- which is what makes ``anchors_per_sample`` testable at the
    batch width its shipped value is a mean over.
    """
    device = torch.device("cpu")
    anchors, valid = model._build_anchor_index(batch, device, phase, stride)
    channels = model.decoder_out_channels
    zeros = torch.zeros(batch, anchors.shape[1], model.horizon, channels)
    lags = model.lag_attn.L
    forward_outputs = {
        "anchor_index": anchors,
        "anchor_valid": valid,
        "mu_base": zeros,
        "logvar_base": zeros,
        "mu_full": zeros,
        "logvar_full": zeros,
        "attn_weights": torch.full(
            (batch, model.geometry.t, model.lag_attn.num_heads, lags), 1.0 / lags
        ),
    }
    return model._resolved_forecast_gaps(
        forward_outputs,
        zeros,
        torch.ones(batch, model.geometry.t),
        likelihood="mse",
    )


# =================================================================================================
# target_warm_frac: the pairing, stamped
# =================================================================================================
def test_the_warm_fraction_is_exactly_one_at_the_shipped_budget_and_floor() -> None:
    r"""Exactly $1.0$, not approximately: under the pairing $F \ge B - 1$ every scored triple is
    past its channel's warm-up by construction, so anything else means the checkpoint was built by
    code that predates the constructor's refusal."""
    model = build(shipped_warmup_kwargs())

    assert model.target_warm_frac == 1.0
    assert model.warmup_period >= max(model.target_warmup_steps) - 1


def test_the_warm_fraction_is_a_constant_column_not_a_measurement() -> None:
    """Resolved once at construction, so it does not move with the batch, the weight, the phase or
    the stride. A column that moved with any of those would be reporting something other than the
    provenance fact it exists to carry."""
    model, streams, features = _tiled()
    gapped = _weight(model)
    gapped[:, : model.geometry.t // 2] = 0.0
    dense = build(tiny_warmup_kwargs(anchor_stride=1)).eval()

    _out, tiled_metrics = _metrics(model, streams, features, torch.tensor([0, 3]))
    _out, gapped_metrics = _metrics(
        model, streams, features, torch.tensor([1, 2]), weight=gapped
    )
    _out, dense_metrics = _metrics(dense, streams, features, None)

    assert float(tiled_metrics["target_warm_frac"]) == 1.0
    assert float(gapped_metrics["target_warm_frac"]) == 1.0
    assert float(dense_metrics["target_warm_frac"]) == 1.0


def test_the_warm_fraction_probe_is_not_vacuous_against_the_resolver() -> None:
    r"""Proved against the resolver with a hand-built vector, **not** against a model: the
    constructor refuses to build one whose floor and budget disagree, so a "would it notice?" test
    at model level cannot exist.

    A channel honest only from step $20$ against a floor of $5$ and a horizon of $4$ leaves the
    early anchors' windows partly inside the assumed pre-recording history, and the fraction must
    say so.
    """
    honest = CausalFeatureForecastTarget._resolve_target_warm_frac(5, 4, 20, (0, 3, 6))
    violating = CausalFeatureForecastTarget._resolve_target_warm_frac(5, 4, 20, (0, 3, 20))

    assert honest == 1.0
    assert 0.0 < violating < 1.0
    # And the hand computation, so the number is not merely "less than one": channel W' = 20 is
    # cold at (t, tau) exactly when t + 1 + tau < 20, over anchors 5..19 and tau 0..3.
    cold = sum(
        1
        for anchor in range(5, 20)
        for tau in range(4)
        if anchor + 1 + tau < 20
    )
    assert violating == pytest.approx(1.0 - cold / (15 * 4 * 3))


def test_a_floor_that_violates_the_pairing_cannot_be_constructed() -> None:
    """The other half: the resolver can express a violating geometry, and the model cannot be
    built at one. Without this the test above would be checking arithmetic nothing enforces."""
    with pytest.raises(ValueError, match="warmup_period"):
        build(tiny_warmup_kwargs(warmup_period=1))


# =================================================================================================
# anchors_per_sample: the tiling, actually firing
# =================================================================================================
def test_the_anchor_count_is_the_geometry_derived_tile_count_at_train_stride() -> None:
    r"""Every phase in $[0, S)$ at once, so the reported mean is the real one:
    $\lceil (T_{\mathrm{valid}} - F - \varphi)/S \rceil$, which at the shipped geometry is $11$
    for $\varphi \le 1$ and $10$ otherwise, summing to $152$ and averaging to $152/15$.

    The numbers are re-derived here from the geometry rather than written down, so a horizon change
    moves the expectation with the model instead of failing this test.
    """
    model = build(shipped_warmup_kwargs())
    stride = model.horizon
    span = model.geometry.t_valid - model.warmup_period
    phase = torch.arange(stride)

    metrics = _synthetic_gaps(model, phase, stride, batch=stride)

    per_phase = [-(-(span - value) // stride) for value in range(stride)]
    assert span == 152 and stride == SHIPPED_HORIZON
    assert min(per_phase) == 10 and max(per_phase) == 11
    assert sum(per_phase) == span
    assert float(metrics["anchors_per_sample"]) == pytest.approx(span / stride, rel=1e-6)


def test_the_anchor_count_is_the_whole_valid_range_at_the_validation_stride() -> None:
    r"""Validation and test decode every valid anchor, so the count is
    $T_{\mathrm{valid}} - F = 152$ exactly -- not a tile set at one fixed phase, which would sample
    the same $11$ positions of every segment forever."""
    model = build(shipped_warmup_kwargs())
    span = model.geometry.t_valid - model.warmup_period

    metrics = _synthetic_gaps(model, None, 1, batch=2)

    assert span == 152
    assert float(metrics["anchors_per_sample"]) == float(span)


def test_the_anchor_count_reports_the_decoded_set_on_a_fully_masked_batch() -> None:
    """Counted off ``anchor_valid``, not off the mask. A batch whose weight is entirely zero has no
    contributing anchor at all, and this column must still say which anchors the forward *built* --
    otherwise a gap-heavy validation batch would read as the geometry having collapsed."""
    model, streams, features = _tiled()
    weight = _weight(model, value=0.0)

    out, metrics = _metrics(model, streams, features, torch.tensor([0, 3]), weight=weight)

    assert float(metrics["anchors_per_sample"]) == float(out["anchor_valid"].sum()) / BATCH
    for name in ("target_warm_frac", *_WARMTH_KEYS, *_TERTILE_KEYS):
        assert math.isfinite(float(metrics[name])), name


# =================================================================================================
# source_lag_warmth_frac: the compromise, sized
# =================================================================================================
def test_the_source_warmth_split_is_taken_at_the_resolved_block_boundary() -> None:
    """The split point is where the two stored source blocks meet, and the two patterns must
    really differ -- the whole reason for splitting is that the first block is warm from step $0$
    in its fastest channels while the second is not warm at all until far later, and a pooled
    figure would let the first carry the fraction."""
    kwargs = shipped_warmup_kwargs()
    model = build(kwargs)
    split = CausalFeatureForecastTarget.SOURCE_BLOCK_SPLIT
    waits = kwargs["source_warmup_steps"]

    first = [step for index, step in enumerate(waits) if index < split]
    second = [step for index, step in enumerate(waits) if index >= split]
    assert len(first) == split and len(second) == CAUSAL_C_U - split

    for pattern, block in (
        (model.source_block_warm_st, first),
        (model.source_block_warm_ph, second),
    ):
        expected = [
            sum(1 for step in block if step <= s) >= WARM_BLOCK_FRACTION * len(block)
            for s in range(SHIPPED_SEQUENCE_LENGTH)
        ]
        assert pattern.tolist() == expected

    # Not the same pattern twice: the second block warms much later, which is the fact the split
    # exists to keep visible.
    first_warm = int(model.source_block_warm_st.to(torch.uint8).argmax())
    second_warm = int(model.source_block_warm_ph.to(torch.uint8).argmax())
    assert first_warm < second_warm


def test_the_two_warmth_fractions_are_shares_of_the_attention_mass() -> None:
    """Both in $[0, 1]$, and both a share of the mass actually present -- a row with no admissible
    lag normalises to zero, so a denominator counting rows rather than mass would report a
    fraction of something that was never attended."""
    model, streams, features = _tiled()
    _out, metrics = _metrics(model, streams, features, torch.tensor([0, 3]))

    for name in _WARMTH_KEYS:
        value = float(metrics[name])
        assert 0.0 <= value <= 1.0, (name, value)


def test_the_warmth_fractions_move_when_the_lag_floor_moves() -> None:
    r"""A non-zero ``lag_floor`` forbids the nearest lags, so the surviving mass shifts to older
    source steps -- which are colder. A readout that did not move here would be reporting
    something other than where the attention actually landed."""
    model, streams, features = _tiled()
    floored = build(
        tiny_warmup_kwargs(anchor_stride=TINY_STRIDE, lag_floor=6)
    ).eval()

    _out, unfloored_metrics = _metrics(model, streams, features, torch.tensor([0, 3]))
    _out, floored_metrics = _metrics(floored, streams, features, torch.tensor([0, 3]))

    assert model.lag_floor == 0 and floored.lag_floor == 6
    moved = [
        name
        for name in _WARMTH_KEYS
        if float(unfloored_metrics[name]) != float(floored_metrics[name])
    ]
    assert moved, "neither block's warmth responded to the lag floor"


def test_an_ungated_source_is_warm_everywhere_and_says_so() -> None:
    """A model with no budget has no warm-up to wait out, so every step is warm and both columns
    read $1.0$. Worth pinning: the alternative -- a zero, from an empty pattern -- would look like
    a measurement of a cold source rather than like the absence of a guard."""
    from teb_vae.lag_attn_cfs.tests.conftest import TINY_KWARGS

    kwargs = dict(TINY_KWARGS)
    model = build(kwargs).eval()
    y_st, y_ph, u_stream = make_streams(kwargs)

    _out, metrics = _metrics(model, (y_st, y_ph, u_stream), torch.cat([y_st, y_ph], -1), None)

    assert model.source_warmup_steps is None
    assert bool(model.source_block_warm_st.all()) and bool(model.source_block_warm_ph.all())
    for name in _WARMTH_KEYS:
        assert float(metrics[name]) == pytest.approx(1.0, rel=1e-6)


# =================================================================================================
# The warm-up tertiles
# =================================================================================================
def test_the_tertiles_partition_the_kept_channels_by_warm_up_rank() -> None:
    r"""Three contiguous groups as equal in size as the count allows, by **rank** of $W'$ rather
    than by its value -- so the boundaries follow the resolved vector instead of sitting at
    declared step counts a rebuilt dataset would invalidate."""
    model = build(shipped_warmup_kwargs())
    assignment = model.warm_tertile_id.tolist()
    waits = list(model.target_warmup_steps)

    counts = [assignment.count(group) for group in range(WARM_TERTILES)]
    assert sum(counts) == len(waits) == model.decoder_out_channels
    assert max(counts) - min(counts) <= 1

    # Monotone in the warm-up: no channel of a later group waits less than one of an earlier group.
    by_group = [
        [waits[index] for index, group in enumerate(assignment) if group == value]
        for value in range(WARM_TERTILES)
    ]
    for lower, upper in zip(by_group, by_group[1:]):
        assert max(lower) <= min(upper)


def test_the_tertile_boundaries_move_when_the_budget_moves() -> None:
    """A function of the resolved vector, not of a fixed offset: a narrower budget drops the slow
    channels, so the three groups re-partition what is left."""
    wide = build(shipped_warmup_kwargs())
    narrow_kwargs = shipped_warmup_kwargs()
    keep = [
        index
        for index, step in zip(narrow_kwargs["target_keep_index"], narrow_kwargs["target_warmup_steps"])
        if step <= 100
    ]
    steps = [step for step in narrow_kwargs["target_warmup_steps"] if step <= 100]
    narrow = build(
        dict(
            narrow_kwargs,
            target_keep_index=tuple(keep),
            target_warmup_steps=tuple(steps),
            warmup_period=max(steps) - 1,
        )
    )

    def _top_group(model) -> tuple:
        waits = [
            step
            for step, group in zip(model.target_warmup_steps, model.warm_tertile_id.tolist())
            if group == WARM_TERTILES - 1
        ]
        return min(waits), max(waits)

    assert narrow.decoder_out_channels < wide.decoder_out_channels
    # The slowest tertile is where a budget change shows: the wide budget's top group reaches the
    # budget itself, the narrow one's cannot.
    assert _top_group(wide)[1] == max(wide.target_warmup_steps) == 134
    assert _top_group(narrow)[1] <= 100
    assert _top_group(narrow) != _top_group(wide)


@pytest.mark.parametrize("likelihood", ["gaussian_nll", "mse"])
def test_both_channel_splits_recompose_to_the_gap_they_are_read_beside(
    perturb_posterior, likelihood
) -> None:
    r"""The criterion the three tertile columns exist to satisfy, and the inherited block split
    beside it: both are partitions of the same per-channel gap over the same denominator, so both
    add back to ``pred_gap``.

    The two are compared against **each other** at $10^{-6}$ relative, which is the sharp test --
    they are two reductions of one elementwise difference -- and against ``pred_gap`` itself at a
    bound derived from the cancellation in its own subtraction. See :data:`_CANCELLATION_EPS`: the
    limit there is float32, not the split.
    """
    model, streams, features = _tiled()
    perturb_posterior(model)
    _out, metrics = _metrics(
        model, streams, features, torch.tensor([0, 3]), likelihood=likelihood
    )

    total = float(metrics["pred_gap"])
    assert total != 0.0, "the probe is vacuous on an unperturbed model"
    tertiles = sum(float(metrics[name]) for name in _TERTILE_KEYS)
    blocks = float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"])

    assert tertiles == pytest.approx(blocks, rel=_SPLIT_TOL, abs=1e-9)
    slack = _CANCELLATION_EPS * abs(float(metrics["nll_base_block"]))
    assert tertiles == pytest.approx(total, abs=slack)
    assert blocks == pytest.approx(total, abs=slack)


def test_the_tertiles_are_not_the_block_split_under_another_name() -> None:
    r"""They cut **across** the stored block boundary: at the shipped budget the kept set is $32$
    channels of the first block plus all $66$ of the second, and both span nearly the same rebased
    range -- so no tertile is a block and the two splits are answering different questions."""
    model = build(shipped_warmup_kwargs())
    keep = model.target_gate.keep_index.tolist()
    first_block = [
        index
        for index, declared in enumerate(keep)
        if declared < CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT
    ]

    assert len(first_block) == 32 and len(keep) - len(first_block) == 66
    for group in range(WARM_TERTILES):
        members = {index for index, value in enumerate(model.warm_tertile_id.tolist()) if value == group}
        assert members != set(first_block)
        assert members & set(first_block), (
            f"tertile {group} holds no channel of the first stored block, so the two splits do "
            f"not cross after all"
        )


def test_the_tertiles_are_zero_at_init_like_the_gap_they_decompose(perturb_posterior) -> None:
    """At initialisation the posterior is the prior, so every part of the gap is zero -- and a
    split that read the wrong branch, or scored against a differently-built target, would be
    nonzero here while ``pred_gap`` stayed at zero. Paired with the perturbed control, so the
    zeros are a property of the model rather than of the split."""
    model, streams, features = _tiled()
    _out, fresh = _metrics(model, streams, features, torch.tensor([0, 3]))

    for name in _TERTILE_KEYS:
        assert float(fresh[name]) == pytest.approx(0.0, abs=1e-6), name

    perturb_posterior(model)
    _out, moved = _metrics(model, streams, features, torch.tensor([0, 3]))
    assert any(float(moved[name]) != 0.0 for name in _TERTILE_KEYS)


def test_the_tertiles_carry_no_gradient() -> None:
    """They are diagnostics. A term that reached the graph would be an objective term no weight in
    any config controls."""
    model, streams, features = _tiled()
    model.train()

    out = model(*streams, torch.tensor([0, 3]))
    metrics = model.compute_loss(out, features, weight=_weight(model))["metrics"]

    for name in (*_TERTILE_KEYS, *_WARMTH_KEYS, "target_warm_frac", "anchors_per_sample"):
        assert not metrics[name].requires_grad, name


# =================================================================================================
# kld_source_null: the floor the availability clock induces
# =================================================================================================
def _null_and_matched(model, streams, features, phase):
    """``(source_conditioned_kl_raw, kld_source_null)`` from one forward."""
    out, metrics = _metrics(model, streams, features, phase)
    null = controls.source_null_kld(model, out, streams[2], _weight(model))
    return out, float(metrics["source_conditioned_kl_raw"]), float(null)


def test_the_null_readout_is_a_finite_nonnegative_rate(perturb_posterior) -> None:
    """A KL over the same support and in the same nats-per-anchor units as the coupling readout it
    is printed beside, which is the only reason subtracting one from the other means anything."""
    model, streams, features = _tiled()
    perturb_posterior(model)

    _out, matched, null = _null_and_matched(model, streams, features, torch.tensor([0, 3]))

    assert math.isfinite(null) and null >= 0.0
    assert math.isfinite(matched)


def test_the_null_differs_from_the_coupling_readout_when_the_source_is_read(
    perturb_posterior,
) -> None:
    """The first direction of the non-vacuity pair: on a model whose posterior responds to the
    source, replacing the source with a flat trajectory must change the divergence. Equality here
    would mean the readout was measuring the clock all along -- which is exactly the finding this
    arm exists to be able to report, and therefore exactly what must not be true by construction."""
    model, streams, features = _tiled()
    perturb_posterior(model)

    _out, matched, null = _null_and_matched(model, streams, features, torch.tensor([0, 3]))

    assert matched != pytest.approx(null, rel=1e-6)


def test_the_null_equals_the_coupling_readout_when_the_posterior_ignores_the_source(
    perturb_posterior,
) -> None:
    r"""The second direction, and the sharper one. The attended source enters the posterior only
    through ``a_head_norm``, so zeroing that norm's affine parameters makes the fusion's source
    half identically zero whatever the source was -- a posterior that reads the target and nothing
    else. Both readouts must then agree exactly, because they differ only in the source.

    Without this direction the test above would pass for a null arm that computed *anything*
    different from the matched one.
    """
    model, streams, features = _tiled()
    perturb_posterior(model)
    with torch.no_grad():
        model.posterior_head.a_head_norm.weight.zero_()
        model.posterior_head.a_head_norm.bias.zero_()

    _out, matched, null = _null_and_matched(model, streams, features, torch.tensor([0, 3]))

    assert matched > 0.0, "the posterior collapsed onto the prior; the probe is vacuous"
    assert matched == pytest.approx(null, rel=1e-6)


def test_the_null_does_not_move_under_a_derangement_of_the_source(perturb_posterior) -> None:
    """The property that makes this a control the shuffle is not.

    The permutation arm deranges ``source_state`` across the batch, and every row carries the same
    availability pattern, so no permutation can remove it. This arm replaces the stream instead, so
    a derangement of that stream leaves it exactly where it was -- while the shuffled readout, by
    construction, does not.
    """
    model, streams, features = _tiled()
    perturb_posterior(model)
    y_st, y_ph, u_stream = streams
    deranged = u_stream.flip(0)

    _out, _matched, null = _null_and_matched(model, streams, features, torch.tensor([0, 3]))
    _out, _matched_perm, null_perm = _null_and_matched(
        model, (y_st, y_ph, deranged), features, torch.tensor([0, 3])
    )

    assert not torch.equal(u_stream, deranged), "the derangement was a no-op"
    assert null == pytest.approx(null_perm, rel=1e-6)


def test_the_null_is_read_over_the_tiled_anchor_support(perturb_posterior) -> None:
    """The same anchors the coupling readout is averaged over, so the difference of the two is a
    difference of one quantity rather than of two denominators. Two strides decode different sets,
    so a null computed over a fixed support would not move between them."""
    model, streams, features = _tiled()
    perturb_posterior(model)
    weight = _weight(model)

    torch.manual_seed(0)
    with torch.no_grad():
        tiled = model(*streams, torch.tensor([0, 3]), TINY_STRIDE)
        dense = model(*streams, None, 1)

    assert tiled["anchor_index"].shape != dense["anchor_index"].shape
    assert float(controls.source_null_kld(model, tiled, streams[2], weight)) != pytest.approx(
        float(controls.source_null_kld(model, dense, streams[2], weight)), rel=1e-9
    )
