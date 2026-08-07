r"""The per-channel delay is an index operation, and is tested as one.

Nothing here is about wavelets or reaches -- :mod:`teb_vae.lag_attn.channel_reach` owns
those. What matters here is that the shift is exactly the shift it claims: channel $c$'s output
at step $t$ is its input at $t - \delta_c$, positions with no source are zero, and a zero delay
vector changes nothing at all. That last property is what makes the unguarded configuration a
genuine architectural baseline rather than a nearly-identical one.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.delays import ChannelDelay

_BATCH, _STEPS, _CHANNELS = 2, 12, 5


@pytest.fixture
def stream() -> torch.Tensor:
    """A ``(B, T, C)`` stream with no zeros, so a zeroed position is unambiguous."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(_BATCH, _STEPS, _CHANNELS, generator=generator) + 10.0


def test_a_zero_delay_is_a_bitwise_no_op(stream):
    """Not "close to": the unguarded model must be *the* model with no delay in it."""
    delay = ChannelDelay(num_channels=_CHANNELS, delays=[0] * _CHANNELS)

    assert torch.equal(delay(stream), stream)


def test_each_channel_is_read_at_its_own_offset(stream):
    delays = [0, 1, 2, 3, 4]
    delay = ChannelDelay(num_channels=_CHANNELS, delays=delays)

    out = delay(stream)

    for channel, offset in enumerate(delays):
        for step in range(offset, _STEPS):
            assert torch.equal(out[:, step, channel], stream[:, step - offset, channel]), (
                f"channel {channel} at step {step} should read input step {step - offset}"
            )


def test_positions_before_the_delay_are_zero(stream):
    delays = [0, 1, 2, 3, 4]
    delay = ChannelDelay(num_channels=_CHANNELS, delays=delays)

    out = delay(stream)

    for channel, offset in enumerate(delays):
        assert torch.equal(out[:, :offset, channel], torch.zeros(_BATCH, offset))


def test_the_delay_does_not_wrap_around(stream):
    """A modular index would fill the leading positions from the *end* of the sequence -- the
    far worse failure, since it is a leak from the future dressed as a guard."""
    delay = ChannelDelay(num_channels=_CHANNELS, delays=[_STEPS - 1] * _CHANNELS)

    out = delay(stream)

    assert torch.equal(out[:, -1], stream[:, 0])
    assert float(out[:, :-1].abs().max()) == 0.0


def test_the_index_buffer_stays_out_of_the_state_dict():
    """A persistent budget-shaped buffer would make a checkpoint trained at one reach budget
    unloadable at another, and the failure would name misaligned keys, not the budget."""
    delay = ChannelDelay(num_channels=_CHANNELS, delays=[0, 1, 2, 3, 4])

    assert delay.state_dict() == {}


def test_the_buffer_still_moves_with_the_module():
    """Non-persistent is not the same as detached: it must still follow ``.to()``, or a CUDA
    forward would gather with a CPU index."""
    delay = ChannelDelay(num_channels=_CHANNELS, delays=[0, 1, 2, 3, 4]).to(torch.device("cpu"))

    assert delay.delay_steps.device.type == "cpu"
    assert list(delay.buffers())


def test_a_mismatched_delay_vector_raises():
    with pytest.raises(ValueError, match="num_channels"):
        ChannelDelay(num_channels=_CHANNELS, delays=[0, 1])


def test_a_negative_delay_raises():
    """It would read the channel from its own future -- the leak the module exists to close."""
    with pytest.raises(ValueError, match="future"):
        ChannelDelay(num_channels=3, delays=[0, -1, 2])


def test_a_stream_of_the_wrong_width_raises(stream):
    delay = ChannelDelay(num_channels=_CHANNELS + 1, delays=[0] * (_CHANNELS + 1))

    with pytest.raises(ValueError, match=r"\(B, T, 6\)"):
        delay(stream)


def test_the_delay_is_differentiable_through_the_kept_positions(stream):
    """The guard sits between the data and the encoders, so gradient must pass through it."""
    delays = [0, 2, 2, 2, 2]
    delay = ChannelDelay(num_channels=_CHANNELS, delays=delays)
    x = stream.clone().requires_grad_(True)

    delay(x).sum().backward()

    # Steps that no output position reads (the last delta_c of each delayed channel) get no
    # gradient; every other position gets exactly one unit.
    assert float(x.grad[:, :, 0].min()) == 1.0
    assert float(x.grad[:, -2:, 1].abs().max()) == 0.0
    assert float(x.grad[:, :-2, 1].min()) == 1.0
