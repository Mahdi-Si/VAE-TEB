r"""The forecast target: unfold the feature future, gather the surviving channels, never delay.

Everything here is the *reference* construction, assembled from two pieces that already exist and
deliberately reaching **no** module of this package. :func:`~teb_vae.lag_attn.figure_primitives
.future_target` concatenates the two stored target blocks and unfolds them into
$(B, T_{\mathrm{valid}}, H_d, c_y)$ -- it is what ``lag_attn``'s evaluation and plotting are
already pinned against -- and ``torch.index_select`` on the last axis keeps the channels the reach
budget admits.

That independence is the point, and the arrival of ``nets/feature_target.py`` is what makes it
worth stating. The net's builder cannot be the reference for the net's builder; and the net's
builder cannot even take the same *signature*, because the framework-free rule forbids the two
stored block names anywhere under ``nets/``, docstrings included -- so it receives one already
concatenated $(B, T, c_y)$ tensor and this file assembles the block from the two blocks by name.
``test_objective.py`` is where the net's builder is pinned equal to what is built here.

Two claims carry the whole construction, and each is asserted against a **hand-written** slice-
and-stack of the planted pattern rather than against the unfold that produced it -- a wrong
formula shared by the builder and its check would agree with itself perfectly:

1. $Y^{+}[b, t, \tau, k] = Y[b,\, t + 1 + \tau,\, \mathrm{keep}[k]]$, the index identity.
2. The gather is taken from the input gate and the **delay is not**. This is the sharpest
   correctness trap in the model: applying the gate wholesale would make anchor $t$'s target the
   future of anchor $t - \delta_c$, per channel, and every shape check downstream would still
   pass.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.figure_primitives import future_target
from teb_vae.lag_attn_fs.tests.conftest import (
    PATTERN_STEP_SCALE,
    SHIPPED_KWARGS,
    SHIPPED_REACH_BUDGET_S,
    build_target_gate,
    make_patterned_batch,
)
from teb_vae.lag_attn_rws.nets.geometry import TrimmedRawGeometry

_BATCH = 2

#: The production geometry, sourced from the constructor keyword set rather than restated, so a
#: change to either reaches the other. $T = 300$, $H_d = 30$, hence $T_{\mathrm{valid}} = 270$.
_GEOMETRY = TrimmedRawGeometry(
    raw_len=SHIPPED_KWARGS["sequence_length"] * SHIPPED_KWARGS["raw_per_step"],
    decimation=SHIPPED_KWARGS["raw_per_step"],
    horizon=SHIPPED_KWARGS["horizon"],
    warmup=SHIPPED_KWARGS["warmup_period"],
)

#: Surviving target channels at the shipped budget, and the full declared width. Written out
#: because they are the figures the decoder width and the $H_d \cdot C$ block cardinality were
#: costed against; ``test_budget_width.py`` re-derives them from the filter bank.
_KEPT_CHANNELS = 78
_ALL_CHANNELS = 109


@pytest.fixture(scope="module")
def batch():
    """A batch at the production geometry whose target blocks carry the known pattern."""
    return make_patterned_batch(_BATCH, _GEOMETRY.t)


@pytest.fixture(scope="module")
def gate():
    """The target stream's input guard at the shipped reach budget."""
    return build_target_gate(SHIPPED_REACH_BUDGET_S)


def _stream(batch) -> torch.Tensor:
    """The concatenated target stream $(B, T, c_y)$, in the declared block order."""
    return torch.cat([batch.fhr_st, batch.fhr_ph], dim=-1)


def _gathered_target(batch, gate) -> torch.Tensor:
    r"""The forecast target: unfold the whole stream, then keep the surviving channels.

    The gather is ``index_select`` against the gate's own keep-index rather than a call to the
    gate, which would also apply the delay. :meth:`ChannelGate.forward` is
    ``self.delay(torch.index_select(x, -1, self.keep_index))`` -- there is no gather-only method
    to call, so the target does the gather itself.

    Args:
        batch: A batch carrying the two stored target blocks.
        gate: The stream's input guard.

    Returns:
        The target block $(B, T_{\mathrm{valid}}, H_d, C_{\mathrm{keep}})$.
    """
    block = future_target(batch.fhr_st, batch.fhr_ph, _GEOMETRY.horizon)
    return torch.index_select(block, -1, gate.keep_index)


def _stacked_block(stream: torch.Tensor) -> torch.Tensor:
    r"""The same block built by slicing and stacking, independently of ``unfold``.

    Horizon step $\tau$ of every anchor is one contiguous slice of the stream, so the whole block
    is $H_d$ slices stacked on a new axis. It shares no arithmetic with the unfold under test.

    Args:
        stream: A feature stream $(B, T, C)$.

    Returns:
        The block $(B, T_{\mathrm{valid}}, H_d, C)$.
    """
    return torch.stack(
        [
            stream[:, 1 + tau : 1 + tau + _GEOMETRY.t_valid, :]
            for tau in range(_GEOMETRY.horizon)
        ],
        dim=2,
    )


# ---------------------------------------------------------------------------------------
# Shape and the index identity
# ---------------------------------------------------------------------------------------
def test_the_block_has_the_costed_shape(batch, gate):
    """$(B, 270, 30, 78)$: the decoder's output shape, and the block the NLL sums over."""
    block = _gathered_target(batch, gate)

    assert block.shape == (_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon, _KEPT_CHANNELS)
    assert _GEOMETRY.t_valid == _GEOMETRY.t - _GEOMETRY.horizon == 270


def test_the_first_valid_anchor_forecasts_the_thirty_steps_after_it(batch, gate):
    r"""Anchor $0$'s window is steps $1 \ldots 30$ -- it starts at $t + 1$, not at $t$."""
    block = _gathered_target(batch, gate)
    kept = torch.index_select(_stream(batch), -1, gate.keep_index)

    for tau in (0, 1, _GEOMETRY.horizon - 1):
        assert torch.equal(block[:, 0, tau, :], kept[:, 1 + tau, :])


def test_the_last_valid_anchor_ends_on_the_final_stored_step(batch, gate):
    r"""Anchor $269$, step $\tau = 29$, reads step $299$: the reason $T_{\mathrm{valid}}$ stops
    $H_d$ short of $T$ rather than at $T$."""
    block = _gathered_target(batch, gate)
    kept = torch.index_select(_stream(batch), -1, gate.keep_index)
    last = _GEOMETRY.t_valid - 1

    assert torch.equal(block[:, last, _GEOMETRY.horizon - 1, :], kept[:, _GEOMETRY.t - 1, :])


def test_an_interior_anchor_reads_its_own_window(batch, gate):
    r"""One $(t, \tau)$ written out in full, so the identity is pinned somewhere the boundaries
    cannot rescue it: anchor $137$, step $11$, reads stored step $149$."""
    block = _gathered_target(batch, gate)
    kept = torch.index_select(_stream(batch), -1, gate.keep_index)

    assert torch.equal(block[:, 137, 11, :], kept[:, 137 + 1 + 11, :])


def test_the_index_identity_holds_at_every_position(batch, gate):
    """The whole block against a slice-and-stack that shares no arithmetic with ``unfold``."""
    block = _gathered_target(batch, gate)
    expected = _stacked_block(torch.index_select(_stream(batch), -1, gate.keep_index))

    assert torch.equal(block, expected)


def test_the_target_values_name_the_steps_and_channels_they_came_from(batch, gate):
    r"""The planted pattern read back. Element $(0, t, \tau, k)$ is
    $(t + 1 + \tau)\,S + \mathrm{keep}[k]$, so dividing by $S$ recovers the *stored step* and the
    remainder recovers the *stored channel* -- which must be $\mathrm{keep}[k]$, not $k$. A gather
    of the first $78$ channels, or of the wrong $78$, fails here and nowhere else.

    Read off sample $0$ alone: the per-sample offset is itself a multiple of $S$, so it would land
    in the recovered step and say nothing new.
    """
    block = _gathered_target(batch, gate)[0]
    recovered_step = torch.div(block, PATTERN_STEP_SCALE, rounding_mode="floor")
    recovered_channel = block - recovered_step * PATTERN_STEP_SCALE

    expected_step = (
        torch.arange(_GEOMETRY.t_valid).view(-1, 1)
        + 1
        + torch.arange(_GEOMETRY.horizon).view(1, -1)
    ).float()
    assert torch.equal(recovered_step, expected_step.unsqueeze(-1).expand_as(block))
    assert torch.equal(
        recovered_channel, gate.keep_index.float().view(1, 1, -1).expand_as(block)
    )


# ---------------------------------------------------------------------------------------
# The unfold, and why the permute is load-bearing
# ---------------------------------------------------------------------------------------
def test_the_unfold_emits_the_horizon_last_and_the_permute_puts_it_back(batch):
    r"""``Tensor.unfold`` appends the window as a *new trailing* axis, so the raw result is
    $(B, T_{\mathrm{valid}}, c_y, H_d)$ -- channel-major. The shared helper's ``permute`` is what
    makes the block horizon-major, and it is asserted against that helper rather than restated.
    """
    raw = _stream(batch)[:, 1:, :].unfold(dimension=1, size=_GEOMETRY.horizon, step=1)

    assert raw.shape == (_BATCH, _GEOMETRY.t_valid, _ALL_CHANNELS, _GEOMETRY.horizon)
    assert torch.equal(
        future_target(batch.fhr_st, batch.fhr_ph, _GEOMETRY.horizon), raw.permute(0, 1, 3, 2)
    )


def test_the_two_stored_blocks_enter_in_the_declared_order(batch):
    """``fhr_st`` then ``fhr_ph``: the order the keep-index, the reach table and the decoder's
    output channels are all positional against. Swapping them would move every phase-harmonic
    channel by $43$ places and change which channels the budget appears to have kept."""
    block = future_target(batch.fhr_st, batch.fhr_ph, _GEOMETRY.horizon)

    assert block.shape[-1] == _ALL_CHANNELS
    assert torch.equal(block[..., : batch.fhr_st.shape[-1]], _stacked_block(batch.fhr_st))
    assert torch.equal(block[..., batch.fhr_st.shape[-1] :], _stacked_block(batch.fhr_ph))


# ---------------------------------------------------------------------------------------
# The gather is taken from the gate; the delay is not
# ---------------------------------------------------------------------------------------
def test_every_surviving_channel_carries_a_nonzero_delay(gate):
    """What makes the negative test below specific rather than "the two are not equal": at the
    shipped budget **all 78** survivors are delayed, the fastest by one step and the slowest by
    thirty, so a target built through the gate is wrong in every channel it contains."""
    delays = gate.delay.delay_steps

    assert delays.numel() == _KEPT_CHANNELS
    assert int((delays > 0).sum()) == _KEPT_CHANNELS
    assert int(delays.min()) == 1
    assert int(delays.max()) == _GEOMETRY.warmup == 30


def test_the_target_is_not_what_the_gate_emits(batch, gate):
    """``ChannelGate.forward`` is ``self.delay(index_select(...))`` -- there is no gather-only
    method -- so a target that calls the gate is delayed, and nothing downstream would fail."""
    stream = _stream(batch)

    assert not torch.equal(gate(stream), torch.index_select(stream, -1, gate.keep_index))


def test_a_delayed_target_would_forecast_the_wrong_anchors_window(batch, gate):
    r"""The failure spelled out. A block built from the gate's output satisfies

    $$Y^{+}_{\mathrm{delayed}}[t, \tau, k] = Y^{+}[t - \delta_k, \tau, k],$$

    so channel $k$ of anchor $t$ would be scored against the future of an anchor up to thirty
    steps -- two minutes -- earlier, per channel and with no shape change anywhere.
    """
    correct = _gathered_target(batch, gate)
    delayed = _stacked_block(gate(_stream(batch)))
    delays = gate.delay.delay_steps

    assert delayed.shape == correct.shape  # the mistake is invisible to every shape check
    # The slowest survivor, so the displacement is the full 30 steps rather than a single one.
    channel = int(torch.argmax(delays))
    shift = int(delays[channel])
    anchor = 200
    assert torch.equal(delayed[:, anchor, :, channel], correct[:, anchor - shift, :, channel])
    assert not torch.equal(delayed[:, anchor, :, channel], correct[:, anchor, :, channel])
    # And the fastest survivor is displaced too -- by one step, four seconds, still wrong.
    fastest = int(torch.argmin(delays))
    assert not torch.equal(delayed[:, anchor, :, fastest], correct[:, anchor, :, fastest])


# ---------------------------------------------------------------------------------------
# The unguarded arm
# ---------------------------------------------------------------------------------------
def test_without_a_budget_the_target_is_every_declared_channel(batch):
    """With no gate there is nothing to gather from, so the block keeps all $109$ channels and
    the decoder width follows $c_y$. The unguarded arm is a configuration, not a special case."""
    assert build_target_gate(None) is None
    block = future_target(batch.fhr_st, batch.fhr_ph, _GEOMETRY.horizon)

    assert block.shape == (_BATCH, _GEOMETRY.t_valid, _GEOMETRY.horizon, _ALL_CHANNELS)
    assert torch.equal(block, _stacked_block(_stream(batch)))
