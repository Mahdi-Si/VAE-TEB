r"""The availability adapter's input mask: one vector zeroes the stream and announces it.

The adapter already held $m_{t,c} = \mathbb 1[t \ge \delta_c]$, and it now multiplies that same
pattern into the stream before the first linear. What this file pins is the pair of claims that
makes the change safe to have made:

* **It is inert wherever the stream was already zero there.** Every model that builds availability
  terms reaches this adapter through :class:`~teb_vae.lag_attn.nets.delays.ChannelDelay`, which
  returns ``gathered * available`` under the *same* $\delta$ -- so the multiply meets exact zeros
  and changes nothing. Asserted against a hand-written copy of the projection without it, rather
  than against a recorded number, so the claim survives a move between machines.
* **It is not inert otherwise.** Handed a stream that is non-zero everywhere -- which is what a
  one-sided transform's warm-up region actually contains: real values, normalised with constants
  accumulated while excluding exactly that region -- the masked positions must reach no encoder
  input at all. That is asserted by gradient as well as by value, because a value check alone
  passes on a projection that happens to map those numbers near zero.

A separate masking module beside the adapter was the alternative, and it would make the mask
vector and the announcement vector two copies of one quantity that can silently disagree.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn.nets.delays import ChannelGate
from teb_vae.lag_attn.nets.encoders import AvailabilityInputAdapter

#: Probe geometry. Six channels carry a mixed delay vector readably; the sequence is long enough
#: that the longest delay leaves plenty of fully available steps after it.
_IN_DIM, _D_MODEL, _BATCH, _SEQ_LEN = 6, 32, 2, 16

#: Delays with a zero in them, and delays without. The distinction decides whether the start
#: embedding is built, and both must be inert under an already-delayed stream.
_MIXED_DELAYS = (0, 3, 5, 0, 2, 4)
_POSITIVE_DELAYS = (1, 3, 5, 2, 4, 6)


def _adapter(delays=None, *, seed: int = 0) -> AvailabilityInputAdapter:
    """A seeded adapter at the probe geometry, in eval mode so dropout is out of the way."""
    torch.manual_seed(seed)
    return AvailabilityInputAdapter(
        in_dim=_IN_DIM,
        d_model=_D_MODEL,
        sequence_length=_SEQ_LEN,
        dropout=0.0,
        delays=delays,
    ).eval()


def _stream(seed: int = 0) -> torch.Tensor:
    """A $(B, T, C)$ stream that is non-zero at every position.

    The offset matters: a ``randn`` draw is non-zero with probability one but small somewhere, and
    the point of these probes is that the masked region holds values of ordinary size.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(_BATCH, _SEQ_LEN, _IN_DIM, generator=generator) + 3.0


def _projection_without_the_mask(adapter: AvailabilityInputAdapter, x: torch.Tensor):
    """The adapter's forward as it read before the input mask, written out by hand.

    Args:
        adapter: The adapter under test.
        x: Its input stream.

    Returns:
        The projected stream, with the announcement terms added but the input unmasked.
    """
    embedded = adapter.linear(x)
    if adapter.mask_proj is not None:
        embedded = embedded + adapter.mask_proj(adapter.availability[: x.shape[1]] - 1.0)
    if adapter.start_embed is not None:
        embedded = embedded + adapter.start_indicator[: x.shape[1]] * adapter.start_embed
    return adapter.res_mlp(adapter.drop(adapter.act(adapter.norm(embedded))))


@pytest.mark.parametrize(
    "delays", [_MIXED_DELAYS, _POSITIVE_DELAYS], ids=["mixed", "all-positive"]
)
def test_the_mask_is_bitwise_inert_on_a_stream_that_came_through_the_delay(delays):
    r"""The inertness claim, at the composition every guarded model actually builds.

    ``ChannelDelay`` returns ``gathered * available`` under the same $\delta$ the adapter's
    pattern is built from, so the positions the multiply touches are exactly zero already and
    $0 \times 0$ is $0$. ``torch.equal``: a guarded model must be unaffected to the last bit.
    """
    adapter = _adapter(delays)
    gate = ChannelGate(declared_width=_IN_DIM, keep_index=None, delays=delays)
    delayed = gate(_stream(seed=1))

    # The premise, stated rather than assumed: the multiply meets a tensor that is already zero.
    assert torch.equal(delayed * adapter.availability, delayed)
    assert torch.equal(adapter(delayed), _projection_without_the_mask(adapter, delayed))


def test_an_ungated_adapter_builds_no_mask_and_is_untouched():
    """With no delays there is no availability buffer, so there is nothing to multiply by -- the
    branch is on a module built at construction, never on tensor content, which is the distinction
    ``find_unused_parameters=False`` cares about."""
    adapter = _adapter()
    x = _stream(seed=2)

    assert adapter.mask_proj is None
    assert not hasattr(adapter, "availability")
    assert torch.equal(adapter(x), _projection_without_the_mask(adapter, x))


@pytest.mark.parametrize("delays", [None, _POSITIVE_DELAYS])
def test_a_stream_of_the_wrong_width_is_refused_by_both_guard_states(delays):
    """The check the multiply would otherwise have removed on exactly the guarded streams.

    ``self.linear(x)`` refuses a wrong channel count itself, but ``x * available`` *broadcasts*:
    a squeezed or mis-sliced single-channel stream would fan out to every channel and produce a
    plausible encoding. Asserted for both guard states so the two cannot come to disagree about
    what a valid input is -- which is the shape the bug had, the guarded model being the lenient
    one.
    """
    adapter = _adapter(delays)
    narrow = _stream(seed=7)[..., :1]

    with pytest.raises(ValueError, match="channels"):
        adapter(narrow)
    # Not vacuous: the full width still goes through, and a too-wide stream is refused too.
    adapter(_stream(seed=7))
    with pytest.raises(ValueError, match="channels"):
        adapter(torch.cat([_stream(seed=7), narrow], dim=-1))


def test_the_masked_prefix_of_an_unzeroed_stream_reaches_no_encoder_input():
    """The case the mask exists for: values of ordinary size inside the unavailable region.

    Perturbing them must move nothing at all. Bitwise, because the multiply is exact.
    """
    adapter = _adapter(_POSITIVE_DELAYS)
    x = _stream(seed=3)
    planted = x.clone()
    for channel, delay in enumerate(_POSITIVE_DELAYS):
        planted[:, :delay, channel] = 1.0e6

    assert torch.equal(adapter(planted), adapter(x))
    # Not vacuous: a value planted one step *past* the delay does move the output.
    beyond = x.clone()
    beyond[:, max(_POSITIVE_DELAYS), 0] = 1.0e6
    assert not torch.equal(adapter(beyond), adapter(x))


def test_no_gradient_reaches_the_masked_positions():
    """The value check above passes on a projection that merely maps those numbers near zero;
    this one does not. A masked input position must have an exactly zero derivative."""
    adapter = _adapter(_POSITIVE_DELAYS)
    x = _stream(seed=4).requires_grad_(True)

    adapter(x).sum().backward()

    assert x.grad is not None
    for channel, delay in enumerate(_POSITIVE_DELAYS):
        masked = x.grad[:, :delay, channel]
        assert float(masked.abs().max()) == 0.0, f"channel {channel} leaked below its delay"
        # And the first available step of that channel does carry gradient, so the probe is
        # measuring the boundary rather than a dead projection.
        assert float(x.grad[:, delay, channel].abs().max()) > 0.0


def test_the_change_added_no_buffer_and_no_submodule():
    """The mask is the announcement's own pattern. A second buffer would be a second vector, free
    to describe a different region with every shape still correct."""
    adapter = _adapter(_POSITIVE_DELAYS)

    assert {name for name, _ in adapter.named_buffers()} == {
        "availability",
        "start_indicator",
    }
    assert {name for name, _ in adapter.named_children()} == {
        "linear",
        "norm",
        "act",
        "drop",
        "res_mlp",
        "mask_proj",
    }
