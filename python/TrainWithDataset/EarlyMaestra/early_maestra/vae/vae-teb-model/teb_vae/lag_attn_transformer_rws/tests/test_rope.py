r"""Rotary position encoding: the relative-position property, and that it is not a no-op.

The second half is the whole point of this file. A rotation that never happens -- all frequencies
zero, the tables indexed but discarded, the reshape pairing the wrong coordinates -- satisfies the
relative-position identity trivially, passes every causality test in the package, passes prefix
equivalence, and leaves the encoder permutation-equivariant with nothing downstream to catch it.
So every invariance assertion here is paired with a variation assertion.

The probe vectors are all ones rather than random draws. With $q = k = \mathbf 1$ the score
collapses to $2\sum_i \cos(\omega_i (t - j))$, which is a closed form: the expected separations
between displacements are properties of the frequency schedule, not of a seed.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.blocks import ROPE_BASE, RotaryPositionEncoding

#: Head width and table length the probes run at. $d_h = 8$ gives four frequency pairs spanning
#: $\omega \in \{1, 10^{-1}, 10^{-2}, 10^{-3}\}$ -- fast enough to separate small displacements,
#: slow enough that the largest is not yet aliased.
D_HEAD, MAX_SEQ_LEN = 8, 32

#: float32 round-off on an $O(d_h)$ inner product of $O(1)$ terms.
_INVARIANCE_TOL = 1e-5

#: Separation two displacements must exceed to count as distinguishable.
_SEPARATION_TOL = 1e-3


def _rotated_constant(rope: RotaryPositionEncoding, seq_len: int) -> torch.Tensor:
    """Rotate the same all-ones vector at every position; returns ``(T, d_head)``."""
    return rope(torch.ones(seq_len, rope.d_head))


def _score(rotated_q: torch.Tensor, rotated_k: torch.Tensor, t: int, j: int) -> float:
    """The unnormalised attention score between query position ``t`` and key position ``j``."""
    return float(rotated_q[t] @ rotated_k[j])


def test_the_inner_product_depends_only_on_the_displacement():
    r"""$\langle R(t)q, R(j)k\rangle$ is a function of $t - j$ alone."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    rotated = _rotated_constant(rope, MAX_SEQ_LEN)

    for t, j in ((5, 2), (9, 0), (20, 17)):
        base = _score(rotated, rotated, t, j)
        for shift in (1, 3, 8):
            shifted = _score(rotated, rotated, t + shift, j + shift)
            assert abs(base - shifted) < _INVARIANCE_TOL, (
                f"score at ({t}, {j}) is {base:.6f} but at ({t + shift}, {j + shift}) it is "
                f"{shifted:.6f}; the encoding is not purely relative"
            )


def test_the_inner_product_actually_varies_with_the_displacement():
    """The non-degeneracy half: a rotation that does nothing passes every other test here."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    rotated = _rotated_constant(rope, MAX_SEQ_LEN)

    offsets = (0, 1, 4, 16)
    scores = {offset: _score(rotated, rotated, offset, 0) for offset in offsets}
    for index, left in enumerate(offsets):
        for right in offsets[index + 1 :]:
            assert abs(scores[left] - scores[right]) > _SEPARATION_TOL, (
                f"displacements {left} and {right} both score {scores[left]:.6f} -- the rotation "
                f"carries no positional information"
            )


def test_the_rotation_moves_every_position_except_the_first():
    """$R(0)$ is the identity by construction; $R(t)q \\neq q$ for $t > 0$ is what must be true."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    rotated = _rotated_constant(rope, MAX_SEQ_LEN)
    ones = torch.ones(D_HEAD)

    assert torch.allclose(rotated[0], ones, atol=_INVARIANCE_TOL)
    for position in (1, 2, 7, MAX_SEQ_LEN - 1):
        assert not torch.allclose(rotated[position], ones, atol=_SEPARATION_TOL), (
            f"R({position}) left the vector unchanged"
        )


def test_the_encoding_is_deterministic_and_carries_no_parameters():
    """It must stay independent of the lag attention's learned lag biases, which it would not be
    if it had parameters of its own to co-adapt with them."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    assert list(rope.parameters()) == []


def test_the_tables_are_non_persistent_buffers():
    """They are derivable from three constructor arguments, so a checkpoint must not carry them --
    and must not refuse to load when ``max_seq_len`` changes."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    assert rope.state_dict() == {}
    assert {name for name, _ in rope.named_buffers()} == {"cos_table", "sin_table"}


def test_the_tables_follow_a_dtype_move():
    """Buffer membership is what makes ``.to(device)`` reach them; ``.to(dtype)`` is checkable
    here without a second device."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    assert rope.cos_table.dtype == torch.float32

    rope.to(torch.float64)

    assert rope.cos_table.dtype == torch.float64
    assert rope.sin_table.dtype == torch.float64
    assert rope(torch.ones(4, D_HEAD, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_the_tables_follow_a_device_move():
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN).to("cuda")
    assert rope.cos_table.is_cuda and rope.sin_table.is_cuda


def test_an_odd_head_width_raises_naming_it():
    with pytest.raises(ValueError, match="d_head=7"):
        RotaryPositionEncoding(7, MAX_SEQ_LEN)


def test_a_sequence_longer_than_the_tables_raises_naming_both_lengths():
    """No dynamic cache growth: $T$ is a constructor argument, so this is an error rather than a
    silent reallocation that would make the memory footprint input-dependent."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    with pytest.raises(ValueError, match=f"{MAX_SEQ_LEN + 1} steps"):
        rope(torch.ones(MAX_SEQ_LEN + 1, D_HEAD))


def test_a_wrong_head_width_raises():
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="head width"):
        rope(torch.ones(4, D_HEAD + 2))


def test_the_frequency_schedule_is_the_standard_geometric_one():
    r"""$\omega_i = \theta^{-2i/d_h}$, read back off the tables at $t = 1$."""
    rope = RotaryPositionEncoding(D_HEAD, MAX_SEQ_LEN)
    expected = torch.tensor(
        [ROPE_BASE ** (-2.0 * index / D_HEAD) for index in range(D_HEAD // 2)],
        dtype=torch.float32,
    )
    assert torch.allclose(rope.cos_table[1], expected.cos(), atol=_INVARIANCE_TOL)
    assert torch.allclose(rope.sin_table[1], expected.sin(), atol=_INVARIANCE_TOL)
