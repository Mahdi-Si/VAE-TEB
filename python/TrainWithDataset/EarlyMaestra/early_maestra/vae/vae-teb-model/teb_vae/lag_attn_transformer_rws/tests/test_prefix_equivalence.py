r"""Prefix equivalence: $\mathcal E(X_{0:T-1})_t = \mathcal E(X_{0:t})_t$.

Token causality says the output at $t$ ignores everything after $t$. Prefix equivalence says
something stronger and separately falsifiable: the output at $t$ is the *same function* of the
prefix however long the sequence around it happens to be. Causality can hold while this fails --
if positions were computed relative to the sequence end, if the rotary table were indexed from an
offset that depends on the input length, or if a convolution padded on the right and the padding
happened to be masked out downstream.

The encoder this replaces never needed the property, because a fixed dilation schedule and a
unidirectional LSTM cannot express an end-relative position. Rotary encoding plus a growing
key-value context can, which is what makes it newly worth checking.

The negative control is built here, in the test file, rather than as a switch in production code:
a rotary encoding whose positions run $T-1-t$ instead of $t$. That is precisely the failure the
property exists to catch, and under it the assertion must fail.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_rws.nets.blocks import RotaryPositionEncoding
from teb_vae.lag_attn_transformer_rws.tests.conftest import (
    SEQ_LEN,
    TINY_KWARGS,
    build_stream_encoder,
    relative_change,
)

BATCH = 2
D_MODEL = int(TINY_KWARGS["d_model"])
SOURCE_WINDOW = int(TINY_KWARGS["source_attention_window"])

#: The full and prefix runs compute the same quantity by different reduction orders -- a masked key
#: contributes an exact zero in one and is simply absent in the other -- and
#: ``scaled_dot_product_attention`` may pick a different kernel at a different sequence length. So
#: this is a tolerance rather than a bitwise assertion.
_PREFIX_TOL = 1e-5

#: Anchors, including both ends. $t = 0$ is where a convolution's left padding and the rotary
#: table's zero offset meet; $t = T-1$ is where the prefix is the whole sequence and the assertion
#: is trivially true, which is worth having as a self-check on the harness.
ANCHORS = (0, 1, SEQ_LEN // 2, SEQ_LEN - 1)


def _sequence(seed: int = 0) -> torch.Tensor:
    """A seeded $(B, T, d)$ encoder input."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL, generator=generator)


def _prefix_movement(encoder, x: torch.Tensor, anchor: int) -> float:
    """Relative difference between the full run read at ``anchor`` and the prefix run's last step."""
    full = encoder(x)[:, anchor]
    prefix = encoder(x[:, : anchor + 1])[:, -1]
    return relative_change(full, prefix)


def _rotate_at(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by the rotary angles at the given table rows, in the module's own convention."""
    cos = self.cos_table[index].to(x.dtype)
    sin = self.sin_table[index].to(x.dtype)
    pairs = x.reshape(*x.shape[:-1], self.d_head // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.reshape(x.shape)


def _end_relative_rope(self, x: torch.Tensor) -> torch.Tensor:
    """Positions running $L-1-t$ instead of $t$, where $L$ is the length of *this* input."""
    seq_len = int(x.shape[-2])
    return _rotate_at(self, x, torch.arange(seq_len - 1, -1, -1))


def _length_normalised_rope(self, x: torch.Tensor) -> torch.Tensor:
    """Positions stretched to fill the table, so the step-to-position map depends on the length."""
    seq_len = int(x.shape[-2])
    stretch = max(self.max_seq_len - 1, 1) / max(seq_len - 1, 1)
    index = (torch.arange(seq_len, dtype=torch.float32) * stretch).round().long()
    return _rotate_at(self, x, index.clamp_max(self.max_seq_len - 1))


@pytest.mark.parametrize("anchor", ANCHORS)
@pytest.mark.parametrize("stream", ["target", "source"])
def test_encoding_a_prefix_reproduces_the_full_run_at_that_step(stream, anchor):
    encoder = build_stream_encoder(stream)

    movement = _prefix_movement(encoder, _sequence(), anchor)

    assert movement < _PREFIX_TOL, (
        f"{stream} encoder: reading step {anchor} of the full sequence differs from encoding "
        f"X[0:{anchor + 1}] and reading its last step, by {movement:.3e} relative"
    )


@pytest.mark.parametrize("anchor", [0, 1, SOURCE_WINDOW - 1])
def test_the_windowed_encoder_holds_where_the_window_is_truncated_by_the_start(anchor):
    """Below the window the band mask is cut short by the sequence start, so the prefix run and the
    full run attend to different-sized key sets. Both must still be the same $t$-step function."""
    encoder = build_stream_encoder("source")
    assert anchor < SOURCE_WINDOW

    assert _prefix_movement(encoder, _sequence(seed=1), anchor) < _PREFIX_TOL


def test_an_end_relative_rotary_encoding_is_not_a_control(monkeypatch):
    r"""Recorded so nobody re-adds the obvious control and believes it.

    Positions running $L-1-t$ look like exactly the failure this property exists to catch, and they
    are not one. A rotary score depends only on
    $\operatorname{pos}(t) - \operatorname{pos}(j) = (L-1-t) - (L-1-j) = j - t$: the $L$-dependent
    part is a *uniform* shift of every position in the run, and a uniform shift cancels in every
    difference. The sign flips; nothing else does. So this scheme is still prefix-equivalent, and
    a control built on it would silently assert nothing.

    A control has to make the step-to-position map depend on the length **non-uniformly**, which is
    what the next test does.
    """
    monkeypatch.setattr(RotaryPositionEncoding, "forward", _end_relative_rope)
    encoder = build_stream_encoder("target")

    assert _prefix_movement(encoder, _sequence(seed=2), SEQ_LEN // 2) < _PREFIX_TOL


def test_the_property_fails_under_a_length_normalised_rotary_encoding(monkeypatch):
    """A test that cannot fail is not a test.

    Positions stretched to fill the rotary table are the real failure class: two steps a fixed
    number apart sit at different rotary displacements depending on how long the sequence is, so
    the same prefix encodes differently inside a longer one.

    Checked away from the endpoints: at $t = T-1$ the prefix *is* the full sequence, so the
    assertion is trivially true no matter how positions are indexed.
    """
    monkeypatch.setattr(RotaryPositionEncoding, "forward", _length_normalised_rope)
    encoder = build_stream_encoder("target")

    movement = _prefix_movement(encoder, _sequence(seed=2), SEQ_LEN // 2)

    assert movement > _PREFIX_TOL, (
        f"a length-dependent rotary encoding still passed prefix equivalence at {movement:.3e}, "
        f"so the probe is not reading the positions at all"
    )


def test_the_monkeypatch_leaves_the_last_step_alone():
    """After the patch is undone the property must hold again, so a leaked patch cannot make a
    later test in this session pass or fail for the wrong reason."""
    encoder = build_stream_encoder("target")
    assert _prefix_movement(encoder, _sequence(seed=3), SEQ_LEN // 2) < _PREFIX_TOL
