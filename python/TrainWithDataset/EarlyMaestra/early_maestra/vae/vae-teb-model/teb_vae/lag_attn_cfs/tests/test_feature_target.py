r"""The causal target domain: the block split, and the gaps computed where the loss was.

:class:`~teb_vae.lag_attn_cfs.nets.causal_feature_target.CausalFeatureForecastTarget` inherits the
decoder's width unchanged, and its overrides of ``_build_forecast_target`` and ``compute_loss``
exist only for the forecast clock: with no ``target_forecast_shift`` both delegate to the parent
-- the gather bitwise, the loss under the identical ``weight`` object -- and the anchor seam still
reaches the objective through ``forward_outputs['anchor_index']``. Tested here because a
delegation that stopped delegating would be a fork of shared code free to drift, with every shape
unchanged.

The gap override is the load-bearing one. The parent builds its mask with no anchor set, so on a
tiled model all four splits would be averaged over $T_{\mathrm{valid}}$ anchors while the
``pred_gap`` printed beside them is averaged over the tiles -- and reading a decomposition against a
total computed over a different denominator is exactly the kind of mistake nothing else here would
catch.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_cfs.nets.causal_feature_target import CausalFeatureForecastTarget
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    CAUSAL_ST_WIDTH,
    TINY_STRIDE,
    TINY_TARGET_ALIGN_DELAYS,
    TINY_TARGET_KEEP_INDEX,
    TINY_TARGET_WARMUP_STEPS,
    build,
    make_streams,
    tiny_align_kwargs,
    tiny_warmup_kwargs,
)
from teb_vae.lag_attn_fs.nets.feature_target import FeatureForecastTarget

_TOL = 1e-5

#: Multiplier separating the step index from the channel index in a planted pattern. It only has to
#: exceed the widest channel count the suite builds, and $1000$ keeps the value readable in a
#: failure message: $t = 7$, $c = 52$ reads as $7052$.
_PATTERN_STEP_SCALE = 1000.0


def _patterned(batch: int, length: int, channels: int) -> torch.Tensor:
    r"""A stream whose every element names its own $(b, t, c)$ position, exactly."""
    sample = torch.arange(batch, dtype=torch.float32).view(-1, 1, 1)
    step = torch.arange(length, dtype=torch.float32).view(1, -1, 1)
    channel = torch.arange(channels, dtype=torch.float32).view(1, 1, -1)
    return sample * (_PATTERN_STEP_SCALE * length) + step * _PATTERN_STEP_SCALE + channel


def _weight(batch: int, length: int) -> torch.Tensor:
    """An all-valid decimated weight; the gaps are the objective's own tests' business."""
    return torch.ones(batch, length)


# =================================================================================================
# What the mixin defines, and what it must not
# =================================================================================================
def test_the_block_split_is_the_causal_scattering_width() -> None:
    """$36$, not the two-sided $43$: seven scattering channels per block never survive the build.

    Defined on this class rather than inherited, so a change to the two-sided split cannot move
    this one -- and asserted against the width the target is actually assembled from rather than
    against a second literal.
    """
    assert "TARGET_BLOCK_SPLIT" in vars(CausalFeatureForecastTarget)
    assert CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT == CAUSAL_ST_WIDTH
    assert CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT != FeatureForecastTarget.TARGET_BLOCK_SPLIT


def test_the_decoder_width_is_not_redefined() -> None:
    """The anchor seam is expressed once, in the shared objective and the shared masks.

    A subclass copy would be a second place the gate's width has to be threaded, and the second
    place is the one that goes stale.
    """
    member = "_default_decoder_out_channels"
    assert member not in vars(CausalFeatureForecastTarget), member
    assert getattr(CausalFeatureForecastTarget, member) is getattr(
        FeatureForecastTarget, member
    )


def test_the_stored_clock_gather_is_the_parents_bitwise(tiny_warmup) -> None:
    """With no forecast shift, ``_build_forecast_target`` is the parent's, by delegation.

    The override exists only for the forecast clock; on the stored clock it must hand back
    exactly what the parent hands back, or every historical cell's target moved silently.
    """
    model = build(tiny_warmup)
    assert model.target_forecast_shift is None
    stream = _patterned(BATCH, model.geometry.t, model.c_y)
    anchors = torch.tensor([[5, 9], [6, 10]])
    assert torch.equal(
        model._build_forecast_target(stream, anchors),
        FeatureForecastTarget._build_forecast_target(model, stream, anchors),
    )


def test_the_stored_clock_scored_weight_is_the_same_object(tiny_warmup) -> None:
    """``scored_weight`` is the identity object -- not an equal tensor -- on the stored clock.

    ``compute_loss``'s one substitution is this call, so identity here is what makes the
    stored-clock objective bitwise the parent's.
    """
    model = build(tiny_warmup)
    weight = _weight(BATCH, model.geometry.t)
    assert model.scored_weight(weight) is weight


def test_the_resolved_gaps_are_redefined() -> None:
    """The one member that must fork, and the family's per-package metric hook besides."""
    assert "_resolved_forecast_gaps" in vars(CausalFeatureForecastTarget)


# =================================================================================================
# The target block
# =================================================================================================
def test_the_target_is_gathered_at_the_keep_index_and_never_delayed(tiny_warmup) -> None:
    r"""$Y^{+}[b, a, \tau, k] = Y[b,\, t_a + 1 + \tau,\, \mathrm{keep}[k]]$, position by position.

    A delayed target would ask anchor $t$ to forecast the future of anchor $t - \delta_c$, per
    channel, with every shape downstream unchanged -- which is why the pattern names its own
    position rather than being random.
    """
    model = build(tiny_warmup)
    length = model.geometry.t
    stream = _patterned(BATCH, length, model.c_y)
    anchors = torch.tensor([[5, 9], [6, 10]])

    block = model._build_forecast_target(stream, anchors)
    assert tuple(block.shape) == (BATCH, 2, model.horizon, len(TINY_TARGET_KEEP_INDEX))

    for sample in range(BATCH):
        for position in range(2):
            anchor = int(anchors[sample, position])
            for tau in range(model.horizon):
                for slot, declared in enumerate(TINY_TARGET_KEEP_INDEX):
                    assert float(block[sample, position, tau, slot]) == float(
                        stream[sample, anchor + 1 + tau, declared]
                    )


def test_the_dense_and_anchored_targets_agree_where_they_overlap(tiny_warmup) -> None:
    """The gather is the unfold's restriction, not a different window."""
    model = build(tiny_warmup)
    stream = _patterned(BATCH, model.geometry.t, model.c_y)

    dense = model._build_forecast_target(stream)
    anchors = torch.tensor([[3, 7, 11]] * BATCH)
    gathered = model._build_forecast_target(stream, anchors)

    for position, anchor in enumerate((3, 7, 11)):
        assert torch.equal(gathered[:, position], dense[:, anchor])


# =================================================================================================
# The forecast clock
# =================================================================================================
#: A tiny all-advance shift, one per surviving channel: the shape the resolver's ``physical``
#: clock produces (min exactly 0, staircase upward), small enough that the tiny ceiling
#: $T_{\mathrm{valid}} - 3 = 17$ leaves the stride-4 tiling feasible.
TINY_FORECAST_ADVANCE = tuple(
    min(slot // 8, 3) for slot in range(len(TINY_TARGET_KEEP_INDEX))
)

#: The tiny ``input`` clock: the alignment's own delays, negated -- exactly what the resolver
#: emits for ``causal_target_forecast_clock: input``.
TINY_FORECAST_DELAY = tuple(-delay for delay in TINY_TARGET_ALIGN_DELAYS)


def test_the_physical_clock_advances_each_channel_by_its_own_shift() -> None:
    r"""$Y^{+}[b, a, \tau, k] = Y[b,\, t_a + 1 + \tau + s_k,\, \mathrm{keep}[k]]$, position by position."""
    model = build(tiny_warmup_kwargs(target_forecast_shift=TINY_FORECAST_ADVANCE))
    stream = _patterned(BATCH, model.geometry.t, model.c_y)
    anchors = torch.tensor([[5, 9], [6, 10]])

    block = model._build_forecast_target(stream, anchors)
    assert tuple(block.shape) == (BATCH, 2, model.horizon, len(TINY_TARGET_KEEP_INDEX))

    for sample in range(BATCH):
        for position in range(2):
            anchor = int(anchors[sample, position])
            for tau in range(model.horizon):
                for slot, declared in enumerate(TINY_TARGET_KEEP_INDEX):
                    assert float(block[sample, position, tau, slot]) == float(
                        stream[sample, anchor + 1 + tau + TINY_FORECAST_ADVANCE[slot], declared]
                    )


def test_the_input_clock_delays_each_channel_like_the_encoder_input() -> None:
    r"""Under ``input``, the scored element reads $t + 1 + \tau - d_c$: the aligned stream's own continuation."""
    model = build(tiny_align_kwargs(target_forecast_shift=TINY_FORECAST_DELAY))
    stream = _patterned(BATCH, model.geometry.t, model.c_y)
    anchors = torch.tensor([[7, 11], [8, 12]])

    block = model._build_forecast_target(stream, anchors)
    for sample in range(BATCH):
        for position in range(2):
            anchor = int(anchors[sample, position])
            for tau in range(model.horizon):
                for slot, declared in enumerate(TINY_TARGET_KEEP_INDEX):
                    assert float(block[sample, position, tau, slot]) == float(
                        stream[sample, anchor + 1 + tau + TINY_FORECAST_DELAY[slot], declared]
                    )


def test_the_persistence_gather_clamps_to_the_scored_clocks_past() -> None:
    r"""$Y^{0}$ reads $t_a + \min(s_c, 0)$: the anchor's own step advancing, the aligned step delaying.

    An unclamped advance would hand the decoder stored steps after the anchor -- future data --
    and an unclamped delay would hand horizon element $\tau = d_c - 1$ its own answer.
    """
    anchors = torch.tensor([[7, 11], [8, 12]])

    advanced = build(tiny_warmup_kwargs(target_forecast_shift=TINY_FORECAST_ADVANCE))
    stream = _patterned(BATCH, advanced.geometry.t, advanced.c_y)
    values = advanced._anchor_target_values(stream, anchors)
    for sample in range(BATCH):
        for position in range(2):
            anchor = int(anchors[sample, position])
            for slot, declared in enumerate(TINY_TARGET_KEEP_INDEX):
                assert float(values[sample, position, slot]) == float(
                    stream[sample, anchor, declared]
                )

    delayed = build(tiny_align_kwargs(target_forecast_shift=TINY_FORECAST_DELAY))
    values = delayed._anchor_target_values(stream, anchors)
    for sample in range(BATCH):
        for position in range(2):
            anchor = int(anchors[sample, position])
            for slot, declared in enumerate(TINY_TARGET_KEEP_INDEX):
                assert float(values[sample, position, slot]) == float(
                    stream[sample, anchor + TINY_FORECAST_DELAY[slot], declared]
                )


def test_the_anchor_ceiling_moves_by_the_largest_advance_and_only_then() -> None:
    """An advancing clock loses exactly its largest advance in trailing anchors; a delaying one none."""
    stored = build(tiny_warmup_kwargs())
    assert stored.anchor_ceiling == stored.geometry.t_valid

    advanced = build(tiny_warmup_kwargs(target_forecast_shift=TINY_FORECAST_ADVANCE))
    assert advanced.anchor_ceiling == advanced.geometry.t_valid - max(TINY_FORECAST_ADVANCE)
    anchors, valid = advanced._build_anchor_index(
        batch=1, device=torch.device("cpu"), anchor_phase=0, anchor_stride=1
    )
    assert int(anchors[0, valid[0]].max()) == advanced.anchor_ceiling - 1

    delayed = build(tiny_align_kwargs(target_forecast_shift=TINY_FORECAST_DELAY))
    assert delayed.anchor_ceiling == delayed.geometry.t_valid


def test_the_pooled_weight_is_conservative_over_the_shift_span() -> None:
    """A gap poisons exactly the steps whose shift span reaches it, and nothing else."""
    model = build(tiny_warmup_kwargs(target_forecast_shift=TINY_FORECAST_ADVANCE))
    span = max(TINY_FORECAST_ADVANCE)
    weight = _weight(1, model.geometry.t)
    gap = 12
    weight[0, gap] = 0.0

    pooled = model.scored_weight(weight)
    for step in range(model.geometry.t):
        expected = 0.0 if gap - span <= step <= gap else 1.0
        assert float(pooled[0, step]) == expected, step


def test_a_mixed_sign_forecast_shift_is_refused() -> None:
    """No resolver clock produces one, so a mixed vector is two clocks spliced together."""
    mixed = tuple(
        1 if slot == 0 else -1 if slot == 1 else 0
        for slot in range(len(TINY_TARGET_KEEP_INDEX))
    )
    with pytest.raises(ValueError, match="mixes signs"):
        build(tiny_warmup_kwargs(target_forecast_shift=mixed))


# =================================================================================================
# The gaps, at the anchors the loss saw
# =================================================================================================
def _forward_and_gaps(kwargs, phase=1, perturb=None):
    """A forward at a real tiling, plus the four gaps and the objective's own metrics.

    ``perturb`` is not optional in spirit: the posterior deltas are zero-initialised, so on a fresh
    model base and full are bitwise identical and every gap is exactly $0.0$ -- which satisfies a
    recomposition test and a difference test alike, on a model wired to nothing.
    """
    model = build(kwargs).eval()
    if perturb is not None:
        perturb(model)
    y_st, y_ph, u_stream = make_streams(kwargs)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model(y_st, y_ph, u_stream, phase)
    features = torch.cat([y_st, y_ph], dim=-1)
    weight = _weight(BATCH, model.geometry.t)
    result = model.compute_loss(out, features, weight=weight, likelihood="mse")
    return model, out, features, weight, result["metrics"]


def test_the_parents_version_cannot_even_run_on_a_tiled_model(tiny_warmup) -> None:
    """Which is why the override exists, and why leaving it inherited would not have been quiet.

    It is quiet in the way that matters, though: the shapes only collide because this fixture's
    tiling is short. At a stride of one -- validation's resolution -- the dense mask and the
    gathered one differ by the warm-up prefix alone, and the parent's version would broadcast
    cleanly and report four numbers averaged over the wrong denominator.
    """
    kwargs = tiny_warmup_kwargs(tiny_warmup, anchor_stride=TINY_STRIDE)
    model, out, features, weight, _metrics = _forward_and_gaps(kwargs)
    target = model._build_forecast_target(features, out["anchor_index"])

    with pytest.raises(RuntimeError):
        FeatureForecastTarget._resolved_forecast_gaps(
            model, out, target, weight, likelihood="mse"
        )


def test_the_gaps_recompose_to_the_gap_they_are_read_beside(
    tiny_warmup, perturb_posterior
) -> None:
    r"""Both splits are partial sums of ``pred_gap``: $\sum_c$ of the block split and the whole
    horizon curve each add back to it.

    This is the criterion the override exists for. Computed over the dense anchor range while
    ``pred_gap`` is over the tiles, neither sum would land -- and the two numbers would be read
    against each other anyway, because they are printed in the same row.
    """
    kwargs = tiny_warmup_kwargs(tiny_warmup, anchor_stride=TINY_STRIDE)
    _model, _out, _features, _weight, metrics = _forward_and_gaps(
        kwargs, perturb=perturb_posterior
    )

    total = float(metrics["pred_gap"])
    assert total != 0.0, "the probe is vacuous on an unperturbed model"
    blocks = float(metrics["pred_gap_st"]) + float(metrics["pred_gap_ph"])
    assert blocks == pytest.approx(total, rel=_TOL, abs=1e-6)
    # The horizon split reports its two ends rather than the whole curve, so what is checked is
    # that both ends are inside the total's own scale rather than a second summation.
    assert abs(float(metrics["pred_gap_tau_first"])) <= abs(total) + _TOL
    assert abs(float(metrics["pred_gap_tau_last"])) <= abs(total) + _TOL


def test_the_gaps_move_with_the_tiling(tiny_warmup, perturb_posterior) -> None:
    """The paired control: two phases decode different anchors, so the four numbers must differ.

    Without it, an override that silently ignored the anchor set -- or a model whose gaps were
    structurally zero -- would satisfy every assertion above.
    """
    kwargs = tiny_warmup_kwargs(tiny_warmup, anchor_stride=TINY_STRIDE)
    _m, _o, _f, _w, first = _forward_and_gaps(kwargs, phase=0, perturb=perturb_posterior)
    _m, _o, _f, _w, second = _forward_and_gaps(kwargs, phase=2, perturb=perturb_posterior)

    assert float(first["pred_gap_st"]) != float(second["pred_gap_st"])
    assert float(first["nll_base_block"]) != float(second["nll_base_block"])


def test_the_block_split_counts_the_channels_the_budget_kept(tiny_warmup) -> None:
    """The split follows the keep-index rather than assuming the survivors are contiguous."""
    model = build(tiny_warmup)
    assert model.target_gate is not None
    keep = model.target_gate.keep_index
    first_block = int((keep < CausalFeatureForecastTarget.TARGET_BLOCK_SPLIT).sum())

    assert first_block == sum(1 for index in TINY_TARGET_KEEP_INDEX if index < CAUSAL_ST_WIDTH)
    assert 0 < first_block < len(TINY_TARGET_KEEP_INDEX), (
        "the tiny budget kept channels from one block only; the split is untested"
    )


# =================================================================================================
# The pairing refusal
# =================================================================================================
def test_the_pairing_refusal_names_both_numbers() -> None:
    """A static check, so it is testable without a model that the constructor would refuse."""
    with pytest.raises(ValueError) as error:
        CausalFeatureForecastTarget._check_anchor_floor(10, (0, 4, 12))
    message = str(error.value)
    assert "warmup_period=10" in message and "12" in message and "11" in message

    # Exactly $B - 1$ is admitted, and so is anything above it.
    CausalFeatureForecastTarget._check_anchor_floor(11, (0, 4, 12))
    CausalFeatureForecastTarget._check_anchor_floor(200, (0, 4, 12))
    # And an ungated stream has no floor to clear.
    CausalFeatureForecastTarget._check_anchor_floor(0, ())


def test_the_pairing_refusal_names_which_of_the_two_requirements_binds() -> None:
    r"""The floor is $\max(B - 1, \max_c(W'_c + d_c))$, and the two halves come from different
    places: the first from the **scored target**, which is never shifted, and the second from the
    **inputs**, which are. A message that named only a number would leave an operator raising the
    floor against the wrong one."""
    with pytest.raises(ValueError) as error:
        CausalFeatureForecastTarget._check_anchor_floor(10, (0, 4, 12))
    assert "scored target" in str(error.value)

    with pytest.raises(ValueError) as error:
        CausalFeatureForecastTarget._check_anchor_floor(12, (0, 4, 12), (5, 3, 1))
    message = str(error.value)
    assert "shifted inputs" in message
    assert "warmup_period=12" in message and "at least 13" in message
    # The binding channel is named, and it is not the slowest one: channel 0 waits nothing and is
    # shifted 5, channel 2 waits 12 and is shifted 1, so 13 comes from the last.
    assert "channel 2" in message and "t - 1" in message


def test_a_zero_shift_vector_is_the_unshifted_case_and_not_a_shift_that_ran() -> None:
    r"""The distinction the second half rests on. With $d_c = 0$ the input at step $t$ *is* the
    stored coefficient at $t$: a cold one is masked and announced inside the availability adapter,
    which is the policy this family ships and is why $F = B - 1$ is admitted with the slowest kept
    channel still cold at the anchor itself. Applying the input-warmth half unconditionally would
    refuse the shipped configuration, so an empty vector and an all-zero one must give one answer.
    """
    CausalFeatureForecastTarget._check_anchor_floor(11, (0, 4, 12), ())
    CausalFeatureForecastTarget._check_anchor_floor(11, (0, 4, 12), (0, 0, 0))
    with pytest.raises(ValueError, match="scored target"):
        CausalFeatureForecastTarget._check_anchor_floor(10, (0, 4, 12), (0, 0, 0))


def test_any_shift_at_all_costs_exactly_the_one_anchor_the_boundary_step_bought() -> None:
    r"""Once *anything* is shifted the floor is at least $B$, never $B - 1$, and no arrangement of
    the shifts can avoid it.

    $\max_c(W'_c + d_c) \ge \max_c W'_c = B$ because every $d_c \ge 0$, so the input-warmth half
    always exceeds the scored half by at least one. That is the whole price of the alignment on the
    anchor axis, it is the same one anchor at the shipped reference, and it is a structural
    statement rather than a property of the vectors below -- which is why it is asserted over three
    different shift arrangements including the one that puts the shift on the slowest channel.
    """
    waits = (0, 4, 12)
    for shifts in ((8, 6, 0), (1, 1, 0), (0, 0, 3)):
        required = max(wait + shift for wait, shift in zip(waits, shifts))
        assert required >= max(waits), shifts
        CausalFeatureForecastTarget._check_anchor_floor(required, waits, shifts)
        with pytest.raises(ValueError, match="shifted inputs"):
            CausalFeatureForecastTarget._check_anchor_floor(required - 1, waits, shifts)


def test_the_shipped_tiny_guard_satisfies_its_own_pairing(tiny_warmup) -> None:
    """The fixture is not accidentally exempt from the rule it is meant to exercise."""
    assert max(TINY_TARGET_WARMUP_STEPS) > 0
    assert int(tiny_warmup["warmup_period"]) >= max(TINY_TARGET_WARMUP_STEPS) - 1
