r"""Turning a reach budget in seconds into surviving channels and their delays.

The resolution is one small pure function, and every property below is one the rest of the guard
silently assumes: that the unguarded default really is "everything, undelayed"; that the shipped
$120$ s budget produces the channel counts the design was costed against; that a delay which
would outrun the loss warm-up is refused rather than quietly zero-padding trained anchors; and
that the source phase-harmonic block behaves as the structural all-or-nothing it is.
"""
from __future__ import annotations

import math

import pytest

from teb_vae.lag_attn_rws.channel_reach import (
    SECONDS_PER_STEP,
    block_reach_seconds,
    resolve_channel_budget,
    resolve_stream_budgets,
    stream_reach_seconds,
)

#: The shipped warm-up, in steps. Also exactly the maximum delay at the 120 s budget, which is
#: the boundary case the warm-up comparison has to admit.
_WARMUP = 30


def _shipped(budget_s, **overrides):
    """A ``VAE_model``-shaped config block at the shipped widths and warm-up."""
    return dict(
        {
            "causal_reach_budget_s": budget_s,
            "use_up_st": True,
            "warmup_period": _WARMUP,
            "c_y": 109,
            "c_u": 58,
        },
        **overrides,
    )


# ---------------------------------------------------------------------------------------
# The unguarded default
# ---------------------------------------------------------------------------------------
def test_no_budget_keeps_every_channel_undelayed():
    reach = stream_reach_seconds()["target"]

    keep_index, delays = resolve_channel_budget(reach, None, _WARMUP)

    assert keep_index == tuple(range(len(reach)))
    assert set(delays) == {0}


def test_no_budget_resolves_to_no_guard_at_all():
    """``None`` must produce *nothing*, not an identity guard: the model builds no gather and no
    delay module, which is what makes the unguarded run an architectural baseline."""
    assert resolve_stream_budgets(_shipped(None)) is None


# ---------------------------------------------------------------------------------------
# The shipped budget
# ---------------------------------------------------------------------------------------
def test_the_120_second_budget_gives_the_costed_channel_counts():
    """78 target and 29 source channels at a maximum delay of 30 steps -- the figures the design
    was costed against, and the reason the shipped ``warmup_period`` is 30."""
    budget = resolve_stream_budgets(_shipped(120.0))

    assert len(budget.target_keep_index) == 78
    assert len(budget.source_keep_index) == 29
    assert budget.max_delay == 30


def test_the_per_block_counts_add_up_to_the_stream_counts():
    """The startup log reports per block; the model is gated per stream. A disagreement would
    make the log a description of something other than the run."""
    budget = resolve_stream_budgets(_shipped(120.0))

    counts = {name: kept for name, kept, _ in budget.block_counts}

    assert counts == {"fhr_st": 27, "fhr_ph": 51, "up_st": 27, "up_ph": 2}
    assert counts["fhr_st"] + counts["fhr_ph"] == len(budget.target_keep_index)
    assert counts["up_st"] + counts["up_ph"] == len(budget.source_keep_index)


def test_every_survivors_delay_covers_its_own_reach():
    r"""The defining property: $\Delta\,\delta_c \ge \mathrm{reach}_c$, and $\delta_c$ is the
    *smallest* such delay -- one step less would leave the channel reading past the anchor."""
    reach = stream_reach_seconds()["target"]

    keep_index, delays = resolve_channel_budget(reach, 120.0, _WARMUP)

    for channel, delay in zip(keep_index, delays):
        assert delay * SECONDS_PER_STEP >= reach[channel]
        assert (delay - 1) * SECONDS_PER_STEP < reach[channel]


def test_the_budget_bound_is_inclusive():
    """A channel whose reach is exactly the budget satisfies it. Not cosmetic: the slowest
    source phase channel sits at exactly 100.0 s, so the boundary decides whether a 100 s budget
    keeps a channel or none."""
    keep_index, _ = resolve_channel_budget((10.0, 20.0, 30.0), 20.0, _WARMUP)

    assert keep_index == (0, 1)


# ---------------------------------------------------------------------------------------
# The warm-up bound
# ---------------------------------------------------------------------------------------
def test_a_delay_equal_to_the_warmup_is_allowed():
    """Strictly greater-than, not greater-or-equal. At the 120 s budget the maximum delay is
    exactly 30 and so is the shipped warm-up; a `>=` test would refuse the shipped
    configuration."""
    budget = resolve_stream_budgets(_shipped(120.0, warmup_period=30))

    assert budget.max_delay == 30


def test_a_delay_beyond_the_warmup_raises_naming_both_values():
    with pytest.raises(ValueError) as excinfo:
        resolve_stream_budgets(_shipped(120.0, warmup_period=29))

    message = str(excinfo.value)
    assert "causal_reach_budget_s" in message
    assert "30" in message and "29" in message


def test_a_budget_that_keeps_nothing_raises():
    """A stream of zero channels builds a model that trains to completion having never read
    it -- and then reports its KL as a measurement of it."""
    with pytest.raises(ValueError, match="keeps no channel"):
        resolve_channel_budget((100.0, 200.0), 10.0, _WARMUP)


# ---------------------------------------------------------------------------------------
# The source phase block
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("budget_s", [32.0, 60.0, 99.0])
def test_budgets_below_a_hundred_seconds_drop_the_source_phase_block_entirely(budget_s):
    """Structural, not incidental: the source phase band tops out at 0.05 Hz, so every one of
    its pairs is built from two slow wavelets. Below 100 s the source stream is ``up_st`` alone,
    which is what the reach-budget sweep answers "does ``up_ph`` earn its place" with."""
    budget = resolve_stream_budgets(_shipped(budget_s, warmup_period=60))

    counts = {name: kept for name, kept, _ in budget.block_counts}

    assert counts["up_ph"] == 0
    assert len(budget.source_keep_index) == counts["up_st"]


def test_the_source_phase_block_survives_at_a_hundred_seconds():
    """The other side of the boundary, so the test above is not passing on an off-by-one."""
    budget = resolve_stream_budgets(_shipped(100.0, warmup_period=25))

    assert dict((name, kept) for name, kept, _ in budget.block_counts)["up_ph"] == 1


def test_the_ablation_stream_is_resolved_against_its_own_width():
    """With ``use_up_st=False`` the source stream is the phase block alone, so the keep-index
    must index into 15 channels rather than into 58."""
    budget = resolve_stream_budgets(
        _shipped(120.0, use_up_st=False, c_u=15, warmup_period=_WARMUP)
    )

    assert max(budget.source_keep_index) < 15
    assert len(budget.source_keep_index) == 2


# ---------------------------------------------------------------------------------------
# Guards against a misaligned index
# ---------------------------------------------------------------------------------------
def test_a_declared_width_that_disagrees_with_the_bank_raises():
    """The keep-index is positional into the declared width, so a stale ``c_u`` would gather
    the wrong channels rather than fail."""
    with pytest.raises(ValueError, match="c_u=15"):
        resolve_stream_budgets(_shipped(120.0, use_up_st=True, c_u=15))


def test_the_record_written_into_a_runs_config_is_plain_data():
    """It is read by a human and by yaml.safe_dump, and by nothing else."""
    record = resolve_stream_budgets(_shipped(120.0)).as_record()

    assert record["causal_reach_budget_s"] == 120.0
    assert record["max_delay_steps"] == 30
    assert record["target_channels_kept"] == 78
    assert record["source_channels_kept"] == 29
    assert record["channels_kept_per_block"]["up_ph"] == {"kept": 2, "declared": 15}
    assert len(record["target_delays"]) == len(record["target_keep_index"]) == 78
    assert all(
        isinstance(value, (int, float, str, list, dict)) for value in record.values()
    )


def test_the_delay_formula_is_a_ceiling_not_a_rounding():
    """Rounding down by one step would leave the channel reading a quarter-step past the anchor,
    which is exactly the leak the guard removes."""
    reach = block_reach_seconds()["up_ph"]
    # 100.0 s is an exact multiple of the 4 s step; 191.25 s is not.
    keep_index, delays = resolve_channel_budget(reach, 300.0, warmup_period=80)

    for channel, delay in zip(keep_index, delays):
        assert delay == math.ceil(reach[channel] / SECONDS_PER_STEP)
