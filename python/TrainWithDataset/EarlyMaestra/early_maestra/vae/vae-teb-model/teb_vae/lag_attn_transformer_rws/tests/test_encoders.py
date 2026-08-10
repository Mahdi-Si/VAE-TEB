r"""The availability-aware adapter and the causal conv-Transformer encoder.

The adapter's job is small and its failure mode is not. When the causal delay guard is active the
first $\max_c \delta_c$ steps of a channel are exact zeros, and an exactly zero token entering
repeated normalisation layers is what drives the encoder this replaces to global gradient norms
around $10^{26}$. So the tests here are as much about the delayed prefix as about the projection:
which parameters exist under which delay vectors, that they are not inert where they must act, that
they are exactly inert where they must not, and that a fully unavailable step differentiates
finitely.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.nets.encoders import InputAdapter
from teb_vae.lag_attn_transformer_rws.nets.encoders import (
    START_EMBED_STD,
    AvailabilityInputAdapter,
    conv_receptive_field,
)
from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    build_stream_encoder,
    relative_change,
)

#: Adapter geometry the delay probes run at. A six-channel stream is enough to carry a mixed delay
#: vector, and small enough that a hand-written availability loop is readable.
ADAPTER_IN_DIM, ADAPTER_D_MODEL, BATCH = 6, 32, 2

#: A delay vector with a zero in it, and one without. The distinction is load-bearing: the start
#: embedding exists only when *every* channel is delayed, so the first of these leaves it
#: permanently inert while still calling for the mask projection.
MIXED_DELAYS = (0, 3, 5, 0, 2, 4)
POSITIVE_DELAYS = (1, 3, 5, 2, 4, 6)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _stream(seed: int = 0, *, in_dim: int = ADAPTER_IN_DIM, seq_len: int = SEQ_LEN):
    """A seeded $(B, T, C)$ feature stream."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, seq_len, in_dim, generator=generator)


def _adapter(delays=None, *, d_model: int = ADAPTER_D_MODEL, seed: int = 0):
    """A seeded adapter at the probe geometry, in eval mode so dropout is out of the way."""
    torch.manual_seed(seed)
    return AvailabilityInputAdapter(
        in_dim=ADAPTER_IN_DIM,
        d_model=d_model,
        sequence_length=SEQ_LEN,
        dropout=0.0,
        delays=delays,
    ).eval()


def _gated(x: torch.Tensor, delays) -> torch.Tensor:
    """Zero the unavailable prefix of each channel, exactly as the channel gate emits it."""
    gated = x.clone()
    for channel, delay in enumerate(delays):
        gated[:, :delay, channel] = 0.0
    return gated


# ---------------------------------------------------------------------------------------
# The adapter against the one it replaces
# ---------------------------------------------------------------------------------------


def test_the_adapter_reproduces_the_sibling_when_no_terms_are_built():
    """Only terms were added, and this is the claim that pins it.

    The submodules are named for name and built shape for shape, so with no availability terms the
    two state dicts are interchangeable -- and a matched pair must then agree bitwise.
    """
    mine = _adapter()
    theirs = InputAdapter(
        in_dim=ADAPTER_IN_DIM,
        d_model=ADAPTER_D_MODEL,
        dropout=0.0,
        post_residual_activation=False,
    ).eval()

    assert set(mine.state_dict()) == set(theirs.state_dict())
    mine.load_state_dict(theirs.state_dict())
    x = _stream(seed=1)

    assert torch.equal(mine(x), theirs(x))


# ---------------------------------------------------------------------------------------
# The availability terms
# ---------------------------------------------------------------------------------------


def test_the_availability_terms_act_where_channels_are_missing_and_nowhere_else():
    r"""The paired probe: the terms must not be inert, and must not leak past the delayed prefix.

    The projection reads $m_t - \mathbf 1$, so it is *exactly* zero wherever every channel is
    available. That is asserted bitwise rather than within a tolerance, because it is the property
    that keeps the availability mechanism from quietly shifting the representation on the part of
    the sequence where nothing is missing.
    """
    guarded = _adapter(POSITIVE_DELAYS)
    plain = _adapter()
    # The availability terms are constructed after the projection stack, so at a common seed the
    # two adapters draw bitwise identical shared weights. Asserted rather than assumed: without it
    # the comparison below would be measuring two different random adapters.
    shared = dict(plain.named_parameters())
    for name, parameter in guarded.named_parameters():
        if name in shared:
            assert torch.equal(parameter, shared[name]), f"{name} differs before the comparison"
    assert set(shared) < set(dict(guarded.named_parameters()))
    x = _stream(seed=2)

    with_terms = guarded(x)
    without_terms = plain(x)
    max_delay = max(POSITIVE_DELAYS)

    for step in range(max_delay):
        movement = relative_change(without_terms[:, step], with_terms[:, step])
        assert movement > MOVEMENT_TOL, (
            f"step {step} has unavailable channels but the terms moved it by only "
            f"{movement:.3e}"
        )
    for step in range(max_delay, SEQ_LEN):
        assert torch.equal(with_terms[:, step], without_terms[:, step]), (
            f"step {step} has every channel available, so the availability terms must contribute "
            f"exactly nothing there"
        )


@pytest.mark.parametrize(
    "delays,wants_projection,wants_start",
    [
        (None, False, False),
        ((0, 0, 0, 0, 0, 0), False, False),
        (MIXED_DELAYS, True, False),
        (POSITIVE_DELAYS, True, True),
    ],
    ids=["no-gate", "all-zero", "mixed", "all-positive"],
)
def test_which_availability_parameters_exist(delays, wants_projection, wants_start):
    r"""$W_m$ when $\max_c \delta_c > 0$, $e_{\mathrm{start}}$ when $\min_c \delta_c > 0$.

    The mixed case is the one worth having: it satisfies $\max > 0$ while leaving the start token
    permanently inert, because the indicator fires only where *every* channel is missing.
    """
    names = dict(_adapter(delays).named_parameters())

    assert ("mask_proj.weight" in names) is wants_projection
    assert ("start_embed" in names) is wants_start


def test_the_start_embedding_is_drawn_from_the_specified_normal():
    """$\\mathcal N(0, 0.02^2)$, per the architecture's initialisation list."""
    adapter = _adapter(POSITIVE_DELAYS, d_model=512, seed=7)
    assert adapter.start_embed is not None

    measured = float(adapter.start_embed.std())

    assert START_EMBED_STD == 0.02
    assert 0.01 <= measured <= 0.04, f"start embedding std {measured:.4f} is outside [0.01, 0.04]"


def test_the_availability_buffer_matches_a_hand_written_loop():
    adapter = _adapter(MIXED_DELAYS)

    expected = torch.zeros(SEQ_LEN, ADAPTER_IN_DIM)
    for step in range(SEQ_LEN):
        for channel, delay in enumerate(MIXED_DELAYS):
            expected[step, channel] = 1.0 if step >= delay else 0.0

    assert torch.equal(adapter.availability, expected)


def test_the_availability_buffers_are_non_persistent_and_absent_when_inert():
    """Their width is the surviving-channel count, so a persistent copy would make a checkpoint
    trained at one reach budget fail to load at another as misaligned keys."""
    guarded = _adapter(POSITIVE_DELAYS)
    assert {"availability", "start_indicator"} <= set(dict(guarded.named_buffers()))
    assert not any(name.startswith(("availability", "start_indicator")) for name in guarded.state_dict())

    plain = _adapter()
    assert not any(
        name.startswith(("availability", "start_indicator"))
        for name in dict(plain.named_buffers())
    )


def test_the_start_indicator_fires_exactly_on_the_fully_unavailable_prefix():
    adapter = _adapter(POSITIVE_DELAYS)
    minimum = min(POSITIVE_DELAYS)

    fired = adapter.start_indicator.squeeze(-1).bool()

    assert bool(fired[:minimum].all())
    assert not bool(fired[minimum:].any())


def test_a_fully_unavailable_prefix_step_differentiates_finitely():
    r"""The failure the delayed-input representation exists to prevent.

    An exact all-zero token entering repeated normalisation layers produces derivatives of order
    $1/\sqrt{\epsilon}$; the encoder this replaces reaches global gradient norms around $10^{26}$
    that way on every finite reach budget.
    """
    adapter = _adapter(POSITIVE_DELAYS)
    gated = _gated(_stream(seed=3), POSITIVE_DELAYS)

    embedded = adapter(gated)
    assert torch.isfinite(embedded).all()

    embedded[:, 0].pow(2).sum().backward()

    gradients = [
        parameter.grad for parameter in adapter.parameters() if parameter.grad is not None
    ]
    assert gradients, "nothing received a gradient, so the probe proves nothing"
    total = torch.sqrt(torch.stack([(gradient**2).sum() for gradient in gradients]).sum())
    assert torch.isfinite(total), f"global gradient norm is {float(total)}"
    assert float(total) < 1e3, f"global gradient norm {float(total):.3e} is implausibly large"


def test_delays_of_the_wrong_length_raise_naming_both_counts():
    with pytest.raises(ValueError, match="7 entries but the adapter reads 6"):
        _adapter((0, 1, 2, 3, 4, 5, 6))


def test_a_negative_delay_raises():
    """A negative delay reads a channel from its own future, which is the leak the guard removes."""
    with pytest.raises(ValueError, match=">= 0"):
        _adapter((0, -1, 2, 3, 4, 5))


def test_the_adapter_refuses_a_sequence_longer_than_its_pattern():
    adapter = _adapter(POSITIVE_DELAYS)
    with pytest.raises(ValueError, match=f"{SEQ_LEN + 1} steps"):
        adapter(_stream(seq_len=SEQ_LEN + 1))


def test_the_adapter_accepts_a_prefix():
    """Prefix equivalence needs the constant patterns sliced, not re-derived."""
    adapter = _adapter(POSITIVE_DELAYS)
    assert adapter(_stream(seq_len=5)).shape == (BATCH, 5, ADAPTER_D_MODEL)


# ---------------------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", ["target", "source"])
@pytest.mark.parametrize("kwargs", [TINY_KWARGS, SHIPPED_KWARGS], ids=["tiny", "shipped"])
def test_the_encoder_preserves_shape(stream, kwargs):
    encoder = build_stream_encoder(stream, kwargs)
    seq_len, d_model = int(kwargs["sequence_length"]), int(kwargs["d_model"])
    x = torch.randn(1, seq_len, d_model)

    assert encoder(x).shape == (1, seq_len, d_model)


@pytest.mark.parametrize("stream", ["target", "source"])
def test_the_encoder_parameter_count_is_its_blocks_plus_the_final_norm(stream):
    r"""$\sum_b (3d^2 + 3d + dk_b) + N_s(4d^2 + 3d\,d_{\mathrm{ff}} + 4d) + d$.

    Built from the per-block arithmetic rather than from a literal total, so a legitimate change to
    a block cannot be absorbed here without the block's own test noticing first.
    """
    encoder = build_stream_encoder(stream, SHIPPED_KWARGS)
    d_model = int(SHIPPED_KWARGS["d_model"])
    d_ff = int(SHIPPED_KWARGS["encoder_d_ff"])
    blocks = int(SHIPPED_KWARGS[f"{stream}_attention_blocks"])

    stem_cost = sum(
        3 * d_model**2 + 3 * d_model + d_model * kernel
        for kernel in SHIPPED_KWARGS["encoder_conv_kernels"]
    )
    attention_cost = blocks * (4 * d_model**2 + 3 * d_model * d_ff + 4 * d_model)
    measured_blocks = sum(
        parameter.numel()
        for block in list(encoder.conv_blocks) + list(encoder.attention_blocks)
        for parameter in block.parameters()
    )

    total = sum(parameter.numel() for parameter in encoder.parameters())
    assert measured_blocks == stem_cost + attention_cost
    assert total == measured_blocks + d_model


def test_the_shipped_encoders_match_the_architecture_totals():
    """$1{,}676{,}928$ and $888{,}960$: the two numbers the parameter budget is built from."""
    assert (
        sum(p.numel() for p in build_stream_encoder("target", SHIPPED_KWARGS).parameters())
        == 1_676_928
    )
    assert (
        sum(p.numel() for p in build_stream_encoder("source", SHIPPED_KWARGS).parameters())
        == 888_960
    )


def test_conv_receptive_field_matches_the_architecture_arithmetic():
    r"""$R_{\mathrm{conv}} = 1 + 4 \cdot 1 + 8 \cdot 2 = 21$ steps, or $84$ s."""
    assert conv_receptive_field((5, 9), (1, 2)) == 21
    assert conv_receptive_field((), ()) == 1
    assert conv_receptive_field((3,), (4,)) == 9


def test_mismatched_stem_schedules_raise_naming_both_lengths():
    with pytest.raises(ValueError, match="got 2 and 1"):
        conv_receptive_field((5, 9), (1,))


def test_the_source_bound_is_reported_and_the_target_is_unbounded():
    r"""$R_U = \min(R_{\mathrm{conv}} + N_U(W_U - 1),\ T)$; the target has no number to report."""
    target = build_stream_encoder("target", SHIPPED_KWARGS)
    source = build_stream_encoder("source", SHIPPED_KWARGS)

    assert target.receptive_field is None
    assert source.receptive_field == 21 + 3 * (16 - 1) == 66


def test_the_reported_bound_clamps_at_the_sequence_length():
    """A window wide enough to reach past the segment is bounded by the segment, not by the sum."""
    encoder = build_stream_encoder(
        "source", TINY_KWARGS, attention_window=int(TINY_KWARGS["sequence_length"])
    )
    assert encoder.receptive_field == int(TINY_KWARGS["sequence_length"])


def test_an_empty_stem_drops_exactly_the_stem_cost():
    """The stem-free architecture arm needs a working encoder with no convolution blocks at all."""
    full = build_stream_encoder("source", SHIPPED_KWARGS)
    stemless = build_stream_encoder(
        "source", SHIPPED_KWARGS, conv_kernels=(), conv_dilations=()
    )
    d_model = int(SHIPPED_KWARGS["d_model"])
    stem_cost = sum(
        3 * d_model**2 + 3 * d_model + d_model * kernel
        for kernel in SHIPPED_KWARGS["encoder_conv_kernels"]
    )

    assert len(stemless.conv_blocks) == 0
    assert stemless.conv_reach == 1
    assert sum(p.numel() for p in full.parameters()) - sum(
        p.numel() for p in stemless.parameters()
    ) == stem_cost

    x = torch.randn(1, int(SHIPPED_KWARGS["sequence_length"]), d_model)
    assert stemless(x).shape == x.shape


def test_the_two_encoders_share_no_parameter_tensor():
    """Separate instances, so a target gradient cannot reach the source state or the reverse."""
    target = build_stream_encoder("target")
    source = build_stream_encoder("source")

    target_ids = {id(parameter) for _, parameter in target.named_parameters()}
    source_ids = {id(parameter) for _, parameter in source.named_parameters()}

    assert target_ids and source_ids
    assert target_ids.isdisjoint(source_ids)


def test_a_non_positive_window_raises_naming_the_value():
    with pytest.raises(ValueError, match="got 0"):
        build_stream_encoder("source", attention_window=0)
    with pytest.raises(ValueError, match="got -4"):
        build_stream_encoder("source", attention_window=-4)


def test_an_encoder_with_no_attention_blocks_raises():
    with pytest.raises(ValueError, match="at least 1, got 0"):
        build_stream_encoder("source", num_attention_blocks=0)


def test_extra_repr_states_the_counts_the_window_and_the_bound():
    """``print(model)`` is where the source-locality claim is read off a built model."""
    description = repr(build_stream_encoder("source", SHIPPED_KWARGS))

    assert "conv_blocks=2" in description
    assert "attention_blocks=3" in description
    assert "causal window 16" in description
    assert "receptive_field=66 steps" in description
    assert "full causal prefix" in repr(build_stream_encoder("target", SHIPPED_KWARGS))


def test_the_module_entry_point_prints_the_table_and_exits_zero():
    """The demonstration the architecture's source-locality argument rests on."""
    completed = subprocess.run(
        [sys.executable, "-m", "teb_vae.lag_attn_transformer_rws.nets.encoders"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for expected in ("21 steps / 84 s", "66 steps / 264 s", "unbounded", "90 steps / 360 s"):
        assert expected in completed.stdout, f"{expected!r} missing from:\n{completed.stdout}"
