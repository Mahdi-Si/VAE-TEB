r"""The smeared target: what it is, why it is not a leak, and the numbers the argument rests on.

A stored coefficient at decimated step $s$ is not a value *at* $s$. It is a weighted average of
raw signal over a window **centred** at raw index $16 s$ with half-width $\rho_c$, the channel's
$L_{95}$ energy reach. So a forecast target at short horizons is partly a deterministic function
of raw signal the model has already observed. The blend fraction

$$b(\tau, \rho_c) \;=\; \max\!\left(0,\; \frac{\rho_c - 4\tau}{2\rho_c}\right)$$

is exactly $0.5$ at $\tau = 0$ for every channel and falls linearly to zero at step $\rho_c / 4$.

**That is not a causality violation, and the distinction is the whole point of this file.** Two
separate claims carry it, and each is asserted rather than argued:

1. *Nothing the model reads reaches past the anchor.* Per surviving channel, the model reads
   channel $c$ at step $t - \delta_c$ at the latest, and that coefficient's own forward reach
   ends at or before the anchor's causal endpoint $n_{\mathrm{raw}}(t)$. Arithmetic over the
   reach table and the resolved delays -- and a forward probe showing the model is bitwise
   causal step by step, which the arithmetic assumes.
2. *What is smeared into the target came from history the model legitimately holds.* Writing
   $Y^+ = (A, B)$ with $A$ the component fixed by observed history, $A$ is
   $\sigma(Y^-)$-measurable, so $I(U^-; A \mid Y^-) = 0$ and the readout is unbiased. What the
   blend does affect is optimisation (a share of the summed NLL is a component the source cannot
   help with) and interpretation.

The measured table of section 1.6 is recomputed here from the shipped filter bank rather than
copied, so the preprint prints figures some test reproduces -- and so that a change to the
selection moves the recorded numbers instead of silently invalidating them.

The empirical, transform-level version of claim 1 -- perturb a raw signal, recompute the real
scattering transform, measure what the delayed features saw -- already exists as
``lag_attn_rws/tests/test_causal_leak.py`` and is deliberately not repeated: this model reads the
same features through the same gate, so a second copy would be a second copy of one measurement.
"""
from __future__ import annotations

import statistics

import pytest
import torch

from teb_vae.lag_attn.channel_reach import SECONDS_PER_STEP, stream_reach_seconds
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.tests.conftest import (
    SHIPPED_KWARGS,
    SHIPPED_REACH_BUDGET_S,
    make_stub_batch,
    resolve_target_budget,
)
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

#: The production geometry, and the horizon the table is read over.
_GEOMETRY = TrimmedRawGeometry(
    raw_len=SHIPPED_KWARGS["sequence_length"] * SHIPPED_KWARGS["raw_per_step"],
    decimation=SHIPPED_KWARGS["raw_per_step"],
    horizon=SHIPPED_KWARGS["horizon"],
    warmup=SHIPPED_KWARGS["warmup_period"],
)

#: Raw samples per second of the stored signal. $4\,\mathrm{Hz}$, and $16$ raw samples per $4\,$s
#: decimated step, so the two are the same statement twice.
_FS_HZ = SHIPPED_KWARGS["raw_per_step"] / SECONDS_PER_STEP

#: The recorded table of section 1.6: horizon step -> (all mean blend, kept mean blend, clean
#: channels). "Clean" is the same count on both sets because every channel fast enough to be clean
#: by step 29 ($\rho \le 116\,$s) is also fast enough to survive the $120\,$s budget.
_RECORDED_TABLE = {
    0: (0.500, 0.500, 0),
    3: (0.317, 0.254, 14),
    6: (0.229, 0.141, 33),
    12: (0.153, 0.055, 55),
    18: (0.115, 0.020, 63),
    24: (0.090, 0.005, 71),
    29: (0.075, 0.000, 75),
}

#: The recorded means over the whole $H_d = 30$ horizon.
_RECORDED_HORIZON_MEAN_ALL = 0.173
_RECORDED_HORIZON_MEAN_KEPT = 0.091


def _blend(reach_s: float, tau: int) -> float:
    r"""The fraction of horizon step $\tau$'s coefficient support lying in observed history.

    Args:
        reach_s: The channel's forward reach $\rho_c$, in seconds.
        tau: The horizon step.

    Returns:
        $b(\tau, \rho_c) \in [0, 0.5]$.
    """
    return max(0.0, (reach_s - SECONDS_PER_STEP * tau) / (2.0 * reach_s))


def _reaches():
    """``(all, kept)`` per-channel target reaches in seconds, from the shipped filter bank."""
    reach = stream_reach_seconds()["target"]
    keep_index = resolve_target_budget(SHIPPED_REACH_BUDGET_S).target_keep_index
    return reach, tuple(reach[index] for index in keep_index)


# ---------------------------------------------------------------------------------------
# The blend fraction itself
# ---------------------------------------------------------------------------------------
def test_the_blend_is_one_half_at_the_first_horizon_step_for_every_channel():
    """A two-sided window centred on the step: at $\\tau = 0$ exactly half of it lies at or before
    the anchor, whatever the channel's reach. The channel-independence is the point -- it is why
    the first horizon step is the *least* informative one about forecasting skill."""
    reach, _ = _reaches()

    assert {round(_blend(value, 0), 12) for value in reach} == {0.5}


def test_the_blend_falls_linearly_and_reaches_zero_at_the_channels_own_step():
    r"""Zero at $\tau = \rho_c / 4$: past that step the coefficient's whole support lies after the
    anchor, and forecasting it is forecasting."""
    for reach_s in (12.0, 28.0, 117.25):
        clean_step = reach_s / SECONDS_PER_STEP
        assert _blend(reach_s, 0) == pytest.approx(0.5)
        assert _blend(reach_s, clean_step) == pytest.approx(0.0)
        assert _blend(reach_s, clean_step + 1) == 0.0
        # Linear in between: the midpoint is a quarter.
        assert _blend(reach_s, clean_step / 2.0) == pytest.approx(0.25)


def test_the_blend_never_exceeds_one_half():
    """It is a share of a two-sided window, so a value above $0.5$ would mean the coefficient drew
    more from the past than its own support holds -- which is how a sign error in the formula
    would first show."""
    reach, _ = _reaches()

    for tau in range(_GEOMETRY.horizon):
        assert max(_blend(value, tau) for value in reach) <= 0.5


# ---------------------------------------------------------------------------------------
# The recorded table
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("tau", sorted(_RECORDED_TABLE))
def test_the_recorded_blend_table_is_reproduced_from_the_shipped_filter_bank(tau):
    """Row by row, so a failure names the horizon step rather than the table."""
    reach, kept = _reaches()
    expected_all, expected_kept, expected_clean = _RECORDED_TABLE[tau]

    all_blend = [_blend(value, tau) for value in reach]
    kept_blend = [_blend(value, tau) for value in kept]

    assert statistics.fmean(all_blend) == pytest.approx(expected_all, abs=5e-4)
    assert statistics.fmean(kept_blend) == pytest.approx(expected_kept, abs=5e-4)
    # The clean count is the same on both sets: a channel clean by step 29 reaches at most 116 s
    # and therefore also clears the 120 s budget.
    assert sum(1 for value in all_blend if value == 0.0) == expected_clean
    assert sum(1 for value in kept_blend if value == 0.0) == expected_clean


def test_restricting_the_target_to_the_survivors_roughly_halves_the_mean_blend():
    """The measurement the target restriction rests on: $0.091$ against $0.173$ over the horizon.
    This is what "the far horizon is genuinely clean" buys, and it is the reason the target is the
    gated subset rather than all $109$ channels."""
    reach, kept = _reaches()

    mean_all = statistics.fmean(
        _blend(value, tau) for tau in range(_GEOMETRY.horizon) for value in reach
    )
    mean_kept = statistics.fmean(
        _blend(value, tau) for tau in range(_GEOMETRY.horizon) for value in kept
    )

    assert mean_all == pytest.approx(_RECORDED_HORIZON_MEAN_ALL, abs=5e-4)
    assert mean_kept == pytest.approx(_RECORDED_HORIZON_MEAN_KEPT, abs=5e-4)
    assert mean_kept < mean_all / 1.8


def test_the_kept_sets_worst_channel_comes_clean_just_past_the_horizon():
    """$117.25 / 4 = 29.31$, against $H_d = 30$. The full set never reaches a fully clean step at
    all, because it retains a channel reaching $965.5$ s -- step $241$."""
    reach, kept = _reaches()

    assert max(kept) / SECONDS_PER_STEP == pytest.approx(29.3125)
    assert max(kept) / SECONDS_PER_STEP < _GEOMETRY.horizon
    assert max(reach) / SECONDS_PER_STEP > _GEOMETRY.horizon


# ---------------------------------------------------------------------------------------
# Nothing the model reads reaches past the anchor
# ---------------------------------------------------------------------------------------
def test_every_surviving_channel_is_read_from_behind_the_anchors_causal_endpoint():
    r"""The guard's defining property, in raw samples rather than in seconds.

    The model reads channel $c$ at step $t - \delta_c$ at the latest. That coefficient is centred
    at raw index $16(t - \delta_c)$ and reaches forward $4 \rho_c$ samples, so the latest raw
    sample it can contain is $16(t - \delta_c) + 4\rho_c$, against an anchor causal endpoint of
    $n_{\mathrm{raw}}(t) = 16t + 15$. The anchor index cancels, so the statement holds at every
    anchor and is checked per channel.
    """
    reach, _ = _reaches()
    budget = resolve_target_budget(SHIPPED_REACH_BUDGET_S)
    anchor = 137  # any anchor; the index cancels, and a concrete one makes a failure readable

    endpoint = _GEOMETRY.n_raw(anchor)
    for channel, delay in zip(budget.target_keep_index, budget.target_delays):
        latest_read = _GEOMETRY.decimation * (anchor - delay) + _FS_HZ * reach[channel]
        assert latest_read <= endpoint, (
            f"channel {channel} (reach {reach[channel]} s, delay {delay} steps) can contain raw "
            f"sample {latest_read} at anchor {anchor}, past the causal endpoint {endpoint}"
        )


def test_without_the_delay_almost_every_surviving_channel_reaches_past_the_anchor():
    r"""The negative control. If the undelayed features already stopped at the anchor, the guard
    would be solving a problem that does not exist and the test above would prove nothing.

    "Almost" is exact and worth stating: $72$ of the $78$ survivors reach past the endpoint
    undelayed. The six that do not are the fastest channels, whose whole forward reach fits inside
    the $15$ raw samples separating a step's centre from the anchor's causal endpoint -- so they
    are causal *without* a delay, and a control that demanded all $78$ would be asserting
    something false about them.
    """
    reach, _ = _reaches()
    budget = resolve_target_budget(SHIPPED_REACH_BUDGET_S)
    anchor = 137
    endpoint = _GEOMETRY.n_raw(anchor)
    #: The forward reach, in seconds, that exactly fills the gap between a step's centre and the
    #: anchor's causal endpoint: $15$ raw samples at $4\,$Hz.
    self_causal_reach_s = (_GEOMETRY.decimation - 1) / _FS_HZ

    offenders = [
        channel
        for channel in budget.target_keep_index
        if _GEOMETRY.decimation * anchor + _FS_HZ * reach[channel] > endpoint
    ]
    already_causal = [
        channel
        for channel in budget.target_keep_index
        if reach[channel] <= self_causal_reach_s
    ]

    assert self_causal_reach_s == pytest.approx(3.75)
    assert len(offenders) == 72
    assert len(offenders) + len(already_causal) == len(budget.target_keep_index) == 78
    assert set(offenders).isdisjoint(already_causal)


def test_the_delay_is_the_smallest_one_that_achieves_it():
    r"""$\delta_c = \lceil \rho_c / \Delta \rceil$: one step less and the channel reads past the
    anchor again. A guard rounded down would be a guard that does not guard."""
    reach, _ = _reaches()
    budget = resolve_target_budget(SHIPPED_REACH_BUDGET_S)

    for channel, delay in zip(budget.target_keep_index, budget.target_delays):
        assert delay * SECONDS_PER_STEP >= reach[channel]
        assert (delay - 1) * SECONDS_PER_STEP < reach[channel]


def test_the_model_reads_no_step_after_the_anchor(tiny_gated):
    """What the arithmetic above assumes: the encoders are causal step by step, so "the model
    reads channel $c$ at step $t - \\delta_c$ at the latest" is a fact about the network and not
    only about the gate.

    Asserted under ``causal_norm: true``, which the shipped configuration sets and the tiny
    keyword set does not. The next test is why that qualifier is load-bearing.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(tiny_gated, causal_norm=True, dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(2, 16, 43, generator=generator)
    y_ph = torch.randn(2, 16, 66, generator=generator)
    u_stream = torch.randn(2, 16, 58, generator=generator)
    cut = 8

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    perturbed_st, perturbed_ph = y_st.clone(), y_ph.clone()
    perturbed_st[:, cut + 1 :] = torch.randn(
        perturbed_st[:, cut + 1 :].shape, generator=generator
    )
    perturbed_ph[:, cut + 1 :] = torch.randn(
        perturbed_ph[:, cut + 1 :].shape, generator=generator
    )
    torch.manual_seed(0)
    with torch.no_grad():
        moved = model(perturbed_st, perturbed_ph, u_stream)

    assert torch.equal(reference["mu_prior"][:, : cut + 1], moved["mu_prior"][:, : cut + 1])
    assert torch.equal(reference["mu_base"][:, : cut + 1], moved["mu_base"][:, : cut + 1])
    # The paired control: the perturbation did reach the model, so the bit-stability above is
    # a statement about causality rather than about a dead pathway.
    assert not torch.equal(reference["mu_prior"][:, -1], moved["mu_prior"][:, -1])


def test_without_causal_norm_the_step_wise_claim_does_not_hold(tiny_gated):
    """The time-pooling normaliser inside each encoder mixes the whole sequence, so an unguarded
    configuration is *not* causal step by step -- which is exactly what ``causal_norm`` exists to
    fix and why the shipped configuration sets it. Recorded here so the qualifier above reads as
    a measured requirement rather than as a convenience of the fixture."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**dict(tiny_gated, causal_norm=False, dropout=0.0)).eval()
    generator = torch.Generator().manual_seed(0)
    y_st = torch.randn(2, 16, 43, generator=generator)
    y_ph = torch.randn(2, 16, 66, generator=generator)
    u_stream = torch.randn(2, 16, 58, generator=generator)
    cut = 8

    torch.manual_seed(0)
    with torch.no_grad():
        reference = model(y_st, y_ph, u_stream)
    perturbed = y_st.clone()
    perturbed[:, cut + 1 :] = torch.randn(perturbed[:, cut + 1 :].shape, generator=generator)
    torch.manual_seed(0)
    with torch.no_grad():
        moved = model(perturbed, y_ph, u_stream)

    assert model.n_causalized_norms == 0
    assert not torch.equal(reference["mu_prior"][:, : cut + 1], moved["mu_prior"][:, : cut + 1])


# ---------------------------------------------------------------------------------------
# What the blend does and does not affect
# ---------------------------------------------------------------------------------------
def test_the_blended_component_is_scored_identically_in_both_branches(tiny_gated):
    r"""The cancellation the not-a-leak argument names. $D_0$ and $D_1$ come from one shared
    decoder under one shared $\epsilon$, so at initialisation -- where the two latents are equal
    -- the component of the target fixed by observed history contributes the *same* amount to
    both, and ``pred_gap`` is exactly zero however smeared the target is.

    The smear can therefore inflate $D_0$ and $D_1$ together; what it cannot do is manufacture a
    gap between them, which is the quantity every coupling claim is read off.
    """
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**tiny_gated).eval()
    batch = make_stub_batch()

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(batch.fhr_st, batch.fhr_ph, torch.cat([batch.up_st, batch.up_ph], dim=-1))
    metrics = model.compute_loss(
        out, torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1), weight=batch.weight
    )["metrics"]

    assert float(metrics["pred_gap"]) == 0.0
    assert torch.equal(metrics["nll_full_block"], metrics["nll_base_block"])
