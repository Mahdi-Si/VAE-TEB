r"""The pointwise source encoder and its lag gather.

Four properties are load-bearing and none of them is checked anywhere else in the repository:

* the encoding is **pointwise**, which is the architectural claim the whole design rests on and is
  asserted here by a Jacobian rather than argued from the absence of a convolution;
* a lag reaching before the record begins must not **wrap** to the end of it, which would gather
  real future data with every shape correct and no metric obviously wrong;
* **index support and feature warm-up are different conditions**, and conflating them reports a
  channel as readable for as long as its own warm-up lasts;
* a nonfinite value where the availability rules say valid is **refused**, because normalising it to
  zero would present a fabricated standardized-zero coefficient that no downstream readout could
  tell apart from a real one.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_slot_transformer_cfs.nets.pointwise_source import (
    IDENTITY_WIDTH,
    LIFT_WIDTH,
    PointwiseSourceEncoder,
    gather_lag_window,
    lag_validity,
)
from teb_vae.lag_slot_transformer_cfs.tests.conftest import (
    TINY_BATCH,
    TINY_C_U,
    TINY_FLOOR,
    TINY_N_LAGS,
    TINY_SEQ_LEN,
    TINY_SOURCE_WARMUP,
)


def build_encoder(**overrides) -> PointwiseSourceEncoder:
    """Build an encoder at the tiny geometry, with any keyword replaced.

    Args:
        **overrides: Constructor keywords to replace.

    Returns:
        The encoder.
    """
    kwargs = dict(c_u=TINY_C_U, warmup_steps=TINY_SOURCE_WARMUP)
    kwargs.update(overrides)
    return PointwiseSourceEncoder(**kwargs)


# =================================================================================================
# The encoding
# =================================================================================================
def test_the_recommended_arm_holds_no_parameters() -> None:
    """The identity-plus-mask encoder is parameter-free, and the lift is the only arm that is not.

    Stated as a property of the built module rather than of the source text, because a checkpoint
    is where the claim has to be verifiable and the state dict is what carries it. The plain arm's
    state dict is *empty*: the warm-up vector is a non-persistent buffer, so a checkpoint trained at
    one budget cannot fail to load at another and report it as a missing key.
    """
    plain = build_encoder()
    assert not plain.has_parameters()
    assert dict(plain.state_dict()) == {}
    assert plain.warmup_vector.tolist() == list(TINY_SOURCE_WARMUP)
    assert plain.out_width == IDENTITY_WIDTH
    assert plain.source_dim == TINY_C_U * IDENTITY_WIDTH

    lifted = build_encoder(scalar_lift=True)
    assert lifted.has_parameters()
    assert lifted.out_width == IDENTITY_WIDTH + LIFT_WIDTH
    assert lifted.source_dim == TINY_C_U * (IDENTITY_WIDTH + LIFT_WIDTH)


def test_the_encoding_is_the_value_beside_its_availability_bit(
    source_stream: torch.Tensor,
) -> None:
    """Available coefficients come back exactly; unavailable ones come back as an exact zero."""
    encoder = build_encoder()
    encoded, mask = encoder(source_stream)

    assert encoded.shape == (TINY_BATCH, TINY_SEQ_LEN, TINY_C_U, IDENTITY_WIDTH)
    assert mask.shape == (TINY_BATCH, TINY_SEQ_LEN, TINY_C_U)
    assert mask.dtype == torch.bool

    # The mask coordinate and the returned mask are one quantity, not two that agree today.
    assert torch.equal(encoded[..., 1], mask.to(encoded.dtype))

    available = mask
    assert torch.equal(encoded[..., 0][available], source_stream[available])
    assert torch.all(encoded[..., 0][~available] == 0.0)


def test_availability_is_the_channel_warm_up_and_nothing_else(
    source_stream: torch.Tensor,
) -> None:
    """Each channel becomes available at its own $W'_j$, not at a stream-wide boundary."""
    encoder = build_encoder()
    _, mask = encoder(source_stream)

    for channel, wait in enumerate(TINY_SOURCE_WARMUP):
        column = mask[0, :, channel]
        assert not bool(column[:wait].any()), f"channel {channel} warm before its own W'"
        assert bool(column[wait:].all()), f"channel {channel} still cold after its own W'"


def test_an_absent_warm_up_vector_makes_every_step_available(
    source_stream: torch.Tensor,
) -> None:
    """The ungated arm has nothing to wait out, and says so rather than defaulting a staircase."""
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    assert bool(mask.all())
    assert torch.equal(encoded[..., 0], source_stream)


def test_the_encoding_is_pointwise_by_jacobian(source_stream: torch.Tensor) -> None:
    r"""$\partial e_{s,j} / \partial \bar U_{r,k}$ is exactly zero off the diagonal.

    The architectural claim, measured. A convolution, a pooling layer or a temporal normaliser
    added to this module later would leave every shape correct and every other test in this file
    passing; this is the one that would fail.

    Run on a deliberately small slice, because a full Jacobian at the fixture geometry is a
    dense matrix of every coefficient against every other and the property is local.
    """
    encoder = PointwiseSourceEncoder(c_u=3, warmup_steps=None)
    slim = source_stream[:1, :4, :3].clone().requires_grad_(True)

    def encode_values(stream: torch.Tensor) -> torch.Tensor:
        """Return only the value coordinate, which is the half a derivative can be nonzero in."""
        return encoder(stream)[0][..., 0]

    jacobian = torch.autograd.functional.jacobian(encode_values, slim)
    # (1, 4, 3, 1, 4, 3): output index against input index. Pointwise means the only nonzero
    # entries sit where the two indices agree.
    flat = jacobian.reshape(slim.numel(), slim.numel())
    off_diagonal = flat - torch.diag(torch.diagonal(flat))
    assert torch.all(off_diagonal == 0.0)
    assert torch.all(torch.diagonal(flat) == 1.0)


def test_the_scalar_lift_retains_the_identity_and_mask_coordinates(
    source_stream: torch.Tensor,
) -> None:
    """The lift appends; it never replaces, so an available coefficient stays recoverable."""
    encoder = build_encoder(scalar_lift=True)
    encoded, mask = encoder(source_stream)

    assert encoded.shape[-1] == IDENTITY_WIDTH + LIFT_WIDTH
    available = mask
    assert torch.equal(encoded[..., 0][available], source_stream[available])
    assert torch.equal(encoded[..., 1], mask.to(encoded.dtype))


def test_the_scalar_lift_stays_pointwise(source_stream: torch.Tensor) -> None:
    """No lift coordinate depends on another channel's coefficient at the same step."""
    encoder = PointwiseSourceEncoder(c_u=3, warmup_steps=None, scalar_lift=True)
    stream = source_stream[:1, :4, :3].clone()

    baseline, _ = encoder(stream)
    moved = stream.clone()
    moved[0, 2, 1] += 5.0
    perturbed, _ = encoder(moved)

    difference = (perturbed - baseline).abs()
    # Only the perturbed coefficient's own representation may move.
    assert bool(difference[0, 2, 1].max() > 0)
    difference[0, 2, 1] = 0.0
    assert torch.all(difference == 0.0)


# =================================================================================================
# Refusals
# =================================================================================================
def test_a_nonfinite_value_in_the_available_region_is_refused(
    source_stream: torch.Tensor,
) -> None:
    """It is a data error, named by batch element, step and channel."""
    encoder = build_encoder()
    broken = source_stream.clone()
    # Channel 0 is warm from step 0, so this position is unambiguously inside the valid region.
    broken[1, 9, 0] = float("nan")

    with pytest.raises(ValueError, match=r"batch element 1, stored step 9, channel 0"):
        encoder(broken)


def test_a_nonfinite_value_inside_a_channel_warm_up_is_tolerated_and_sanitised(
    source_stream: torch.Tensor,
) -> None:
    """The leading warm-up region holds whatever the builder left there, and never reaches the graph.

    Sanitisation must be a substitution rather than a multiplication: a nonfinite value scaled by a
    zero mask is still nonfinite, in the forward and in every gradient that touches it.
    """
    encoder = build_encoder()
    broken = source_stream.clone()
    # The slowest channel is cold until well past this step.
    broken[0, 0, TINY_C_U - 1] = float("inf")
    broken[0, 1, TINY_C_U - 1] = float("nan")

    encoded, mask = encoder(broken)
    assert bool(torch.isfinite(encoded).all())
    assert not bool(mask[0, 0, TINY_C_U - 1])
    assert not bool(mask[0, 1, TINY_C_U - 1])

    leaf = broken.clone().requires_grad_(True)
    values, _ = encoder(leaf)
    values.sum().backward()
    assert leaf.grad is not None
    assert bool(torch.isfinite(leaf.grad).all())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(c_u=0), "c_u must be > 0"),
        (dict(warmup_steps=(0, 1)), "positional against the source channel axis"),
        (dict(warmup_steps=tuple([-1] + [0] * (TINY_C_U - 1))), "negative entries"),
        (dict(scalar_lift=True, lift_hidden=0), "lift_hidden must be > 0"),
    ],
)
def test_the_constructor_refuses_a_mis_specified_stream(kwargs, message: str) -> None:
    """Each refusal names what is wrong, because each alternative is a wrong number."""
    with pytest.raises(ValueError, match=message):
        build_encoder(**kwargs)


def test_a_stream_of_the_wrong_width_is_refused(source_stream: torch.Tensor) -> None:
    """The warm-up vector is positional, so a width mismatch waits out the wrong channels."""
    encoder = build_encoder()
    with pytest.raises(ValueError, match="declares c_u"):
        encoder(source_stream[..., :-1])


# =================================================================================================
# The gather
# =================================================================================================
def test_the_gather_reads_the_lagged_step(
    source_stream: torch.Tensor, anchors: torch.Tensor
) -> None:
    r"""Anchor $t$ and lag $\ell$ read stored step $t - \ell$, and only that step."""
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    window, window_mask = gather_lag_window(
        encoded, mask, anchors, n_lags=TINY_N_LAGS
    )

    n_anchors = anchors.shape[1]
    assert window.shape == (
        TINY_BATCH,
        n_anchors,
        TINY_N_LAGS,
        TINY_C_U,
        IDENTITY_WIDTH,
    )
    assert window_mask.shape == (TINY_BATCH, n_anchors, TINY_N_LAGS, TINY_C_U)

    for position, anchor in enumerate(anchors[0].tolist()):
        for lag in range(TINY_N_LAGS):
            step = anchor - lag
            if step < 0:
                continue
            assert torch.equal(window[0, position, lag], encoded[0, step])


def test_a_lag_before_the_record_is_masked_and_never_wraps(source_stream: torch.Tensor) -> None:
    """The failure this function exists to prevent: a negative index reading the end of the record.

    A wrapped index gathers real data from a stored step *after* the anchor, with every shape
    correct, the mask fully set and no metric visibly wrong.
    """
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    # Anchor 1 with five lags reaches back to step -3.
    early = torch.ones(TINY_BATCH, 1, dtype=torch.long)
    window, window_mask = gather_lag_window(encoded, mask, early, n_lags=TINY_N_LAGS)

    assert bool(window_mask[:, 0, 0].all())  # lag 0 reads step 1
    assert bool(window_mask[:, 0, 1].all())  # lag 1 reads step 0
    assert not bool(window_mask[:, 0, 2:].any())  # lags 2..4 read steps -1, -2, -3
    assert torch.all(window[:, 0, 2:] == 0.0)

    # And specifically not the tail of the record, which is what a wrap would have returned.
    tail = encoded[:, -1]
    assert not torch.allclose(window[:, 0, 2], tail)


def test_index_support_and_feature_warm_up_are_separate_conditions(
    source_stream: torch.Tensor,
) -> None:
    """A lag can be in range while the channel it reads is still cold.

    At the first eligible anchor the slowest channel has not warmed up, so every lag from it is
    masked on that channel while the fastest channel's are all available. A single stream-wide
    availability boundary would make the two columns identical.
    """
    encoder = build_encoder()
    encoded, mask = encoder(source_stream)
    first = torch.full((TINY_BATCH, 1), TINY_FLOOR, dtype=torch.long)
    _, window_mask = gather_lag_window(encoded, mask, first, n_lags=TINY_N_LAGS)

    slowest = TINY_C_U - 1
    assert TINY_SOURCE_WARMUP[0] == 0 and TINY_SOURCE_WARMUP[slowest] > TINY_FLOOR
    # Every lag is in range at this anchor, so index support is uniformly true here.
    assert TINY_FLOOR - (TINY_N_LAGS - 1) >= 0
    assert bool(window_mask[:, 0, :, 0].all())
    assert not bool(window_mask[:, 0, :, slowest].any())


def test_a_ruled_out_position_is_zeroed_in_every_coordinate(
    source_stream: torch.Tensor,
) -> None:
    """Including the availability bit, which the surrogate index would otherwise have set.

    The surrogate reads a legal step whose coefficients are real and whose mask bit is set. If the
    gather zeroed only the value coordinate, the fusion head would be told a coefficient is present
    at a position where none is.
    """
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    early = torch.zeros(TINY_BATCH, 1, dtype=torch.long)  # anchor 0: every lag but 0 is invalid
    window, window_mask = gather_lag_window(encoded, mask, early, n_lags=TINY_N_LAGS)

    assert torch.all(window[:, 0, 1:] == 0.0)
    assert not bool(window_mask[:, 0, 1:].any())
    # The mask coordinate of the encoding agrees with the returned mask everywhere, still.
    assert torch.equal(window[..., 1], window_mask.to(window.dtype))


def test_the_lag_floor_narrows_the_readable_range(source_stream: torch.Tensor) -> None:
    """A floor above zero rules out steps that are in the record but below it."""
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    anchor = torch.full((TINY_BATCH, 1), 4, dtype=torch.long)

    _, unfloored = gather_lag_window(encoded, mask, anchor, n_lags=TINY_N_LAGS)
    _, floored = gather_lag_window(encoded, mask, anchor, n_lags=TINY_N_LAGS, lag_floor=2)

    assert bool(unfloored[:, 0, :].all())  # steps 4, 3, 2, 1, 0 all exist
    assert bool(floored[:, 0, :3].all())  # steps 4, 3, 2 clear the floor
    assert not bool(floored[:, 0, 3:].any())  # steps 1 and 0 do not


def test_lag_validity_is_any_available_channel(source_stream: torch.Tensor) -> None:
    """The factor a proposal is multiplied by, so an empty lag contributes an exact zero."""
    encoder = build_encoder()
    encoded, mask = encoder(source_stream)
    early = torch.ones(TINY_BATCH, 1, dtype=torch.long)
    _, window_mask = gather_lag_window(encoded, mask, early, n_lags=TINY_N_LAGS)

    valid = lag_validity(window_mask)
    assert valid.shape == (TINY_BATCH, 1, TINY_N_LAGS)
    assert torch.equal(valid, window_mask.any(dim=-1))
    # At anchor 1, lags 0 and 1 read steps 1 and 0, where the two fastest channels are warm.
    assert bool(valid[:, 0, :2].all())
    assert not bool(valid[:, 0, 2:].any())


@pytest.mark.parametrize(
    "anchor_value, message",
    [(-1, r"outside \[0, T\)"), (TINY_SEQ_LEN, r"outside \[0, T\)")],
)
def test_an_out_of_range_anchor_is_refused(
    source_stream: torch.Tensor, anchor_value: int, message: str
) -> None:
    """The clamp below would otherwise turn it into a legal read of the wrong step."""
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    bad = torch.full((TINY_BATCH, 1), anchor_value, dtype=torch.long)
    with pytest.raises(ValueError, match=message):
        gather_lag_window(encoded, mask, bad, n_lags=TINY_N_LAGS)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(n_lags=0), "n_lags must be >= 1"),
        (dict(n_lags=TINY_N_LAGS, lag_floor=-1), "lag_floor must be >= 0"),
    ],
)
def test_the_gather_refuses_a_mis_specified_window(
    source_stream: torch.Tensor, anchors: torch.Tensor, kwargs, message: str
) -> None:
    """Each argument whose wrong value is a wrong number rather than an exception."""
    encoder = build_encoder(warmup_steps=None)
    encoded, mask = encoder(source_stream)
    with pytest.raises(ValueError, match=message):
        gather_lag_window(encoded, mask, anchors, **kwargs)


def test_the_gather_accepts_any_anchor_subset(
    source_stream: torch.Tensor, anchors: torch.Tensor
) -> None:
    """Chunking the anchor axis is a caller decision, and the gather must not notice it.

    Two halves gathered separately and concatenated equal one gather over the whole set, which is
    what lets the model bound its peak memory without changing a number.
    """
    encoder = build_encoder()
    encoded, mask = encoder(source_stream)
    whole, whole_mask = gather_lag_window(encoded, mask, anchors, n_lags=TINY_N_LAGS)

    split = anchors.shape[1] // 2
    first, first_mask = gather_lag_window(
        encoded, mask, anchors[:, :split], n_lags=TINY_N_LAGS
    )
    second, second_mask = gather_lag_window(
        encoded, mask, anchors[:, split:], n_lags=TINY_N_LAGS
    )

    assert torch.equal(whole, torch.cat([first, second], dim=1))
    assert torch.equal(whole_mask, torch.cat([first_mask, second_mask], dim=1))
